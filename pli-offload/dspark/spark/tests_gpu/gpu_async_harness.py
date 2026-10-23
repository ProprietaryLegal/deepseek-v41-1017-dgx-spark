#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU regression and soak for the async remote round (lane A2), on one CUDA device.

    CUDA_VISIBLE_DEVICES=GPU-<uuid> python3 gpu_async_harness.py --pkg <dir with remote_dspark> \\
        --tests <dir with sim.py> --mode regress|soak|selftest [--rounds N] [--out result.json]

--pkg selects the implementation, so the same harness runs the first async design (an export of 57c6474: proxy thread
with a side stream, CudaStreamSync) and the host-word design (HostFlagSync). The engine side is the offline simulator
(tests/sim.py) with its tensors on the GPU and the reference worker (mock_worker.TcpStubServer) on localhost.

regress: after each round is submitted (the engine stream parked on the round), the engine thread loads a module the
         process has never seen (cuModuleLoadData of a fresh PTX kernel), as Triton does at a kernel's first use. The
         first design deadlocks there (run it under `timeout -s KILL`); the host-word design must complete every round.
soak:    N rounds, pipelined (the engine thread runs up to --depth rounds ahead of verification), with fresh module loads
         while rounds are parked, injected worker delays past the timeout, dropped replies and worker restarts
         (resyncs). Every usable draft must equal the fake drafter over the exact window (bit-exact ids and
         confidences); failed requests must carry zeros.
selftest: the hand-off's own startup self-test only.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import sys
import threading
import time

import numpy as np
import torch


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------- fresh modules (PTX)

_PTX = """.version 7.0
.target sm_80
.address_size 64
.visible .entry {name}(.param .u64 p)
{{
  .reg .u64 %rd<3>;
  .reg .u32 %r<2>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, {val};
  st.global.u32 [%rd2], %r1;
  ret;
}}
"""


class ModuleLoader:
    def __init__(self):
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.cu.cuModuleLoadData.restype = ctypes.c_int
        self.cu.cuModuleUnload.argtypes = [ctypes.c_void_p]
        self.cu.cuModuleUnload.restype = ctypes.c_int
        self.n = 0
        self.loaded: list[ctypes.c_void_p] = []
        self.durations: list[float] = []

    def load_fresh(self) -> float:
        """Load a module this process has never loaded; returns the call's duration (s)."""
        self.n += 1
        ptx = _PTX.format(name=f"a2_fresh_{os.getpid()}_{self.n}", val=self.n).encode() + b"\0"
        mod = ctypes.c_void_p()
        t0 = time.perf_counter()
        r = self.cu.cuModuleLoadData(ctypes.byref(mod), ptx)
        dt = time.perf_counter() - t0
        if r != 0:
            raise RuntimeError(f"cuModuleLoadData failed: CUresult {r}")
        self.loaded.append(mod)
        self.durations.append(dt)
        return dt

    def unload_all(self) -> None:
        torch.cuda.synchronize()
        for m in self.loaded:
            r = self.cu.cuModuleUnload(m)
            if r != 0:
                raise RuntimeError(f"cuModuleUnload failed: CUresult {r}")
        self.loaded.clear()


# ------------------------------------------------------------------------------------------- setup

def setup(args):
    sys.path.insert(0, args.tests)
    sys.path.insert(0, args.pkg)
    import sim  # noqa: E402

    wire = importlib.import_module("remote_dspark.wire")
    cfgmod = importlib.import_module("remote_dspark.config")
    mw = importlib.import_module("remote_dspark.mock_worker")
    rmod = importlib.import_module("remote_dspark.round")
    ss = importlib.import_module("remote_dspark.streamsync")
    ident = bytes(range(32))
    hello = lambda: wire.Hello(hidden=sim.H, row_bytes=sim.N_TAPS * sim.H * 2, drafter_identity=ident)  # noqa: E731
    env = {"VLLM_DSPARK_REMOTE": "tcp://127.0.0.1:{port}", "VLLM_DSPARK_REMOTE_IDENTITY": ident.hex(),
           "VLLM_DSPARK_REMOTE_POLICY": "capacity", "VLLM_DSPARK_REMOTE_TIMEOUT_MS": "150",
           "VLLM_DSPARK_REMOTE_BACKOFF_MS": "2", "VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS": "4",
           "VLLM_DSPARK_REMOTE_LOG_EVERY": "0", "VLLM_DSPARK_REMOTE_STATS_FILE": "",
           "VLLM_DSPARK_REMOTE_ASYNC": "1"}
    return sim, wire, cfgmod, mw, rmod, ss, hello, env


def make_sync(ss, dev, args):
    if hasattr(ss, "HostFlagSync"):
        sync = ss.HostFlagSync(dev, lib_path=args.hostflag_lib)
        impl = "host_flags"
    else:
        sync = ss.CudaStreamSync(dev)
        impl = "side_stream (first design)"
    ok, why = sync.self_test()
    log(f"hand-off {impl}: self-test {ok} ({why})")
    if not ok:
        raise SystemExit(f"self-test failed: {why}")
    return sync, impl


def to_dev(sd, dev):
    """The simulator's step on the GPU: pinned + non_blocking copies (never a pageable copy, which would block the CPU
    on the parked stream and hide run-ahead)."""
    def d(t):
        return t.pin_memory().to(dev, non_blocking=True)
    b = sd.batch
    b.positions = d(b.positions)
    b.idx_mapping = d(b.idx_mapping)
    sd.taps = [d(t) for t in sd.taps]
    sd.num_sampled = d(sd.num_sampled)
    sd.num_rejected = d(sd.num_rejected)
    sd.last_sampled = d(sd.last_sampled)
    sd.next_prefill_tokens = d(sd.next_prefill_tokens)
    return sd


def churn(sim, eng, rng):
    for slot, r in enumerate(eng.slots):
        if r is None:
            if rng.random() < 0.5:
                eng.new_request(slot, hit=int(rng.choice([0, 0, 300, 2000])))
            continue
        if not r.prefilling and r.decode_left <= 0:
            if rng.random() < 0.3:
                eng.readd(slot)
                r.decode_left = int(rng.integers(2, 7))
            else:
                eng.slots[slot] = None


def expected(sim, mw, eng, sd):
    """Per batch row: None (no draft needed) or (ids, conf) the worker must return."""
    out = []
    for i, r in enumerate(sd.reqs):
        if not sd.need_draft[i]:
            out.append(None)
        else:
            out.append(mw.fake_draft(eng.expected_window(r), r.anchor, r.computed, sim.N, sim.VOCAB))
    return out


def verify(sim, layout, snap: np.ndarray, n: int, exp) -> tuple[int, int, int]:
    """(checked exact, failed-with-zeros, status) for one round's broadcast snapshot."""
    N = sim.N
    status = int(snap[0])
    okm = snap[layout.o_ok:layout.o_ok + n]
    ids = snap[layout.o_ids:layout.o_ids + n * N].reshape(n, N)
    conf = snap[layout.o_conf:layout.o_conf + n * N].view(np.float32).reshape(n, N)
    checked = zeros = 0
    for i, e in enumerate(exp):
        if e is None or not okm[i]:
            if ids[i].any() or conf[i].any():
                raise AssertionError(f"row {i}: no usable draft but non-zero ids/conf")
            zeros += e is not None
            continue
        want_ids, want_conf = e
        if not np.array_equal(ids[i], want_ids):
            raise AssertionError(f"row {i}: drafts {ids[i].tolist()} != {want_ids.tolist()}")
        if not np.array_equal(conf[i].view(np.int32), want_conf.view(np.int32)):
            raise AssertionError(f"row {i}: confidences differ")
        checked += 1
    if status == 2 and okm.any():
        raise AssertionError("round failed but ok_mask set")
    return checked, zeros, status


def wait_ready(rnd, s=5.0):
    t = time.monotonic() + s
    while not rnd.client.ready and time.monotonic() < t:
        time.sleep(0.002)
    if not rnd.client.ready:
        raise SystemExit("worker connection not ready")


# ------------------------------------------------------------------------------------------- modes

def run(args):
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    sim, wire, cfgmod, mw, rmod, ss, hello, env = setup(args)
    sync, impl = make_sync(ss, dev, args)
    result = {"impl": impl, "pkg": args.pkg, "mode": args.mode, "device": torch.cuda.get_device_name(dev)}
    if args.mode == "selftest":
        result["ok"] = True
        return result
    srv = mw.TcpStubServer(mw.WorkerCore(hello()), "127.0.0.1", 0).start()
    e = dict(env)
    e["VLLM_DSPARK_REMOTE"] = env["VLLM_DSPARK_REMOTE"].format(port=srv.port)
    cfg = cfgmod.RemoteConfig.from_env(e)
    rnd = rmod.RemoteRound(cfg, hello(), device=dev, tp_rank=0, broadcast_fn=None, background=True,
                           stream_sync=sync)
    wait_ready(rnd)
    loader = ModuleLoader()
    eng = sim.Engine(seed=args.seed)
    for sl in range(sim.M):
        eng.new_request(sl, prompt_len=400)
    rng = np.random.default_rng(args.seed)
    L = rnd.layout
    depth = 1 if args.mode == "regress" else args.depth
    snaps = [torch.zeros(L.size, dtype=torch.int32, device=dev) for _ in range(depth)]
    snaps_host = [torch.zeros(L.size, dtype=torch.int32, pin_memory=True) for _ in range(depth)]
    pending: list[tuple[int, int, list, int]] = []   # (round, n, expected, snapshot slot)
    totals = {"checked": 0, "zeros": 0, "ok": 0, "some_failed": 0, "failed": 0, "loads_parked": 0}
    t_start = time.monotonic()
    last_progress = t_start
    events: list[str] = []

    def drain():
        torch.cuda.synchronize(dev)
        for (t, n, exp, k) in pending:
            c, z, st = verify(sim, L, snaps_host[k].numpy(), n, exp)
            totals["checked"] += c
            totals["zeros"] += z
            totals[{0: "ok", 1: "some_failed", 2: "failed"}[st]] += 1
        pending.clear()

    for step in range(args.rounds):
        churn(sim, eng, rng)
        if not any(eng.slots):
            eng.new_request(0)
        if args.mode == "soak":
            if step % 400 == 200:
                srv.core.inject.delay_ms = 400            # past the 150 ms timeout: a failed round
                events.append(f"{step}: delay 400 ms")
            elif step % 400 == 201:
                srv.core.inject.delay_ms = 0
            if step % 1000 == 500:
                srv.core = mw.WorkerCore(hello())          # worker restart: its windows are gone -> resync
                events.append(f"{step}: worker restart")
            if step % 700 == 350:
                srv.core.inject.drop_next_step = True
                events.append(f"{step}: drop reply")
        elif args.mode == "regress":
            srv.core.inject.delay_ms = args.regress_delay_ms   # the proxy is mid-round when the module loads
        sd = eng.step()
        exp = expected(sim, mw, eng, sd)
        sd = to_dev(sd, dev)
        k = step % depth
        res = rnd.run(sd.batch, sd.taps, sd.num_sampled, sd.num_rejected, sd.last_sampled, sd.next_prefill_tokens)
        snaps[k].copy_(res.bcast)                   # stream-ordered after the reply copy
        snaps_host[k].copy_(snaps[k], non_blocking=True)
        pending.append((step, sd.batch.num_reqs, exp, k))
        load_now = (args.mode == "regress") or (args.load_every and step % args.load_every == args.load_every // 2)
        if load_now:
            if args.mode == "regress":
                log(f"round {step + 1}: submitted, stream parked; loading a fresh module")
            dt = loader.load_fresh()
            totals["loads_parked"] += 1
            if args.mode == "regress":
                log(f"round {step + 1}: module loaded in {dt * 1e3:.1f} ms")
        if len(pending) == depth:
            drain()
        if args.mode == "soak" and step % 2000 == 1999:
            drain()
            loader.unload_all()
        now = time.monotonic()
        if now - last_progress > 10:
            last_progress = now
            log(f"step {step + 1}/{args.rounds}: {totals}")
    drain()
    srv.core.inject.delay_ms = 0
    snap = rnd.stats.snapshot()
    rnd.close()
    srv.stop()
    loader.unload_all()
    elapsed = time.monotonic() - t_start
    lat = snap.get("latency", {})
    durs = sorted(loader.durations)
    result.update({
        "ok": True, "rounds": args.rounds, "elapsed_s": round(elapsed, 2), "totals": totals,
        "counters": snap.get("counters", {}), "events": events[:50],
        "module_loads": {"n": len(durs), "p50_ms": round(durs[len(durs) // 2] * 1e3, 3) if durs else None,
                         "max_ms": round(durs[-1] * 1e3, 3) if durs else None},
        "latency_us": {k: lat.get(k) for k in ("rank0_round_us", "rank0_submit_us", "gather_us", "rtt_us")},
        "words": list(sync.values()) if hasattr(sync, "values") else None,
    })
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--mode", choices=("regress", "soak", "selftest"), required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--load-every", type=int, default=37)
    ap.add_argument("--regress-delay-ms", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hostflag-lib", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--gate-file", default=os.path.expanduser("~/dsv41-offload/A2_GPU_OK"),
                    help="GPU work only while this file exists (coordinator's permission); checked every 0.5 s")
    args = ap.parse_args()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis.startswith("GPU-DRAFTER0") or "," in vis:
        raise SystemExit(f"refusing: CUDA_VISIBLE_DEVICES must be exactly GPU-DRAFTER0-... (is {vis!r})")
    if not os.path.exists(args.gate_file):
        raise SystemExit(f"refusing: {args.gate_file} does not exist (no GPU slot from the coordinator)")

    def gate_watch():
        while True:
            time.sleep(0.5)
            if not os.path.exists(args.gate_file):
                log(f"GATE CLOSED: {args.gate_file} disappeared; exiting now")
                os._exit(3)          # also works while the main thread is stuck in a driver call

    threading.Thread(target=gate_watch, daemon=True, name="gate-watch").start()
    log(f"start {args.mode} pkg={args.pkg} rounds={args.rounds}")
    res = run(args)
    log("RESULT " + json.dumps(res))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
