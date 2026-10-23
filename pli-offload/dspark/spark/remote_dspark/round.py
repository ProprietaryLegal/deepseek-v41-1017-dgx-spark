# SPDX-License-Identifier: Apache-2.0
"""One remote draft round, vLLM-agnostic (the TP broadcast is injected).

Rank 0: gather rows (gather.py) -> slot bookkeeping (state.py) -> STEP (client.py) ->
pack [status, n, ok_mask, drafts, confidences] -> broadcast. Other ranks: broadcast only.
Both ranks then read the same device tensor, so every draft token and every fallback
decision is identical across TP ranks (CONTRACT 4.1).
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from . import wire
from .breaker import CircuitBreaker
from .client import RemoteDraftClient, StepOutcome
from .config import RemoteConfig
from .gather import BatchView, RowGatherer, batch_view_from_input_batch
from .state import SlotTable
from .stats import Stats

log = logging.getLogger("vllm.remote_dspark")

STATUS_OK = 0            # every request that needed drafts got usable remote drafts
STATUS_SOME_FAILED = 1   # some requests did not (see ok_mask)
STATUS_ROUND_FAILED = 2  # no remote drafts this round
STATUS_NAMES = {STATUS_OK: "ok", STATUS_SOME_FAILED: "some_failed",
                STATUS_ROUND_FAILED: "round_failed"}


class BroadcastLayout:
    """int32 tensor: [status, n, ok_mask[M], drafts[M*N], conf_bits[M*N]]."""

    def __init__(self, max_reqs: int, num_spec: int):
        self.M, self.N = max_reqs, num_spec
        self.o_ok = 2
        self.o_ids = 2 + max_reqs
        self.o_conf = self.o_ids + max_reqs * num_spec
        self.size = self.o_conf + max_reqs * num_spec

    def drafts(self, t: torch.Tensor, n: int) -> torch.Tensor:
        return t[self.o_ids:self.o_ids + n * self.N].view(n, self.N)

    def conf(self, t: torch.Tensor, n: int) -> torch.Tensor:
        return t[self.o_conf:self.o_conf + n * self.N].view(torch.float32).view(n, self.N)

    def ok_mask(self, t: torch.Tensor, n: int) -> torch.Tensor:
        return t[self.o_ok:self.o_ok + n]


@dataclass
class RoundResult:
    n: int
    bcast: torch.Tensor                 # device, layout above
    layout: BroadcastLayout
    status_host: int | None = None      # rank 0 only (others: read_status())
    reply: wire.StepReply | None = None  # rank 0 only
    need_draft: np.ndarray | None = None  # rank 0 only

    @property
    def drafts(self) -> torch.Tensor:
        return self.layout.drafts(self.bcast, self.n)

    @property
    def conf(self) -> torch.Tensor:
        return self.layout.conf(self.bcast, self.n)

    @property
    def ok_mask(self) -> torch.Tensor:
        return self.layout.ok_mask(self.bcast, self.n)

    def read_status(self) -> int:
        """Host value of the round status on any rank (syncs on non-zero ranks)."""
        if self.status_host is None:
            self.status_host = int(self.bcast[0].item())
        return self.status_host


class RemoteRound:
    def __init__(
        self,
        cfg: RemoteConfig,
        hello: wire.Hello,
        *,
        device: torch.device,
        tp_rank: int,
        broadcast_fn: Callable[[torch.Tensor], None] | None,
        stats: Stats | None = None,
        client: RemoteDraftClient | None = None,
        transport_factory=None,
        background: bool = True,
        stream_sync=None,
        n_out_by_reqs: list[int] | None = None,
    ):
        self.cfg = cfg
        self.hello = hello
        self.device = torch.device(device)
        self.tp_rank = tp_rank
        self.is_sender = tp_rank == 0
        self.broadcast_fn = broadcast_fn
        self.stats = stats or Stats()
        self.N = hello.num_spec
        self.W = hello.window
        self.M = hello.max_slots
        self.layout = BroadcastLayout(self.M, self.N)
        self.bcast = torch.zeros(self.layout.size, dtype=torch.int32, device=self.device)
        # CONTRACT 6.3 n_out by request count (index n, clamped); None = always N
        self.n_out_by_reqs = list(n_out_by_reqs) if n_out_by_reqs else None
        self.rounds = 0
        self._last_epoch = 0
        self._last_log = time.monotonic()
        self._err_logged = 0.0
        self.client: RemoteDraftClient | None = None
        self._bcast_ev = None
        self.gatherer: RowGatherer | None = None
        self.slots: SlotTable | None = None
        self.breaker: CircuitBreaker | None = None
        if self.is_sender:
            pin = self.device.type == "cuda"
            self.bcast_host = torch.zeros(self.layout.size, dtype=torch.int32, pin_memory=pin)
            self.bcast_np = self.bcast_host.numpy()
            # recorded after every copy out of bcast_host; waited on before the host rewrites
            # it (normally already complete: each round synchronizes on its gather first).
            # Async mode has none: the proxy thread makes no CUDA call, and stream order already
            # guarantees the previous reply copy ran before the next round is gathered (streamsync.py)
            self._bcast_ev = torch.cuda.Event() if pin and stream_sync is None else None
            self.gatherer = RowGatherer(max_slots=self.M, window=self.W, hidden=hello.hidden,
                                        n_taps=hello.n_taps, max_rows=hello.max_rows_per_step,
                                        device=self.device, fused=cfg.fused_gather)
            if self.gatherer.row_bytes != hello.row_bytes:
                raise ValueError(f"row bytes {self.gatherer.row_bytes} != {hello.row_bytes}")
            self.slots = SlotTable(self.M, self.W)
            self.breaker = CircuitBreaker(cfg.trip, cfg.cooldown)
            self.stats.set("breaker", self.breaker.snapshot())
            self.client = client or RemoteDraftClient(
                cfg, hello, stats=self.stats, transport_factory=transport_factory,
                background=background)
            self.stats.set("contract_version", wire.CONTRACT_VERSION)
            self.stats.set("policy", cfg.policy)
            self.stats.set("endpoint", str(cfg.endpoint))
            self.stats.set("rank0_buffers", {"device_bytes": self.gatherer.device_bytes,
                                             "pinned_host_bytes": self.gatherer.host_bytes})
            self.stats.set("fused_gather", {"on": self.gatherer.fused,
                                            "check": self.gatherer.fused_check})
            self.stats.set("n_out_by_reqs", self.n_out_by_reqs)
        # async mode (rank 0, capacity policy): the engine stream and the proxy thread hand each
        # round over through two host words (streamsync.HostFlagSync); the proxy thread makes no
        # CUDA call, so a module load on the engine thread can never wait on it
        self.sync = stream_sync if self.is_sender else None
        self.async_mode = self.sync is not None
        self._q: queue.Queue | None = None
        self._staged: frozenset[int] = frozenset()
        if self.async_mode:
            if not getattr(self.sync, "cuda_free_proxy", False):
                raise ValueError("async rounds need a CUDA-free hand-off (streamsync.HostFlagSync)")
            assert self.gatherer is not None
            # resync windows: the engine stream copies a requested slot's whole ring window here
            # after the round's gather; the proxy builds resync rows from it on the CPU
            self.resync_host = torch.zeros(self.M, self.W, self.gatherer.row_elems,
                                           dtype=self.gatherer.dtype,
                                           pin_memory=self.device.type == "cuda")
            self.resync_np = self.resync_host.view(torch.uint8).numpy()
            self._resync_lock = threading.Lock()
            self._resync_req: set[int] = set()
            self._stop = threading.Event()
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self._async_main, daemon=True,
                                            name="dsv41-remote-dspark-round")
            self._thread.start()
            self.stats.set("async", True)
            self.stats.set("async_handoff", self.sync.describe())

    # ------------------------------------------------------------------ run

    def run(self, input_batch, aux_hidden_states, num_sampled: torch.Tensor,
            num_rejected: torch.Tensor, last_sampled: torch.Tensor,
            next_prefill_tokens: torch.Tensor) -> RoundResult:
        n = input_batch.num_reqs
        res = RoundResult(n=n, bcast=self.bcast, layout=self.layout)
        if self.async_mode:
            self._submit_async(input_batch, aux_hidden_states, num_sampled, num_rejected,
                               last_sampled, next_prefill_tokens)
        elif self.is_sender:
            t0 = time.perf_counter()
            try:
                bv = batch_view_from_input_batch(input_batch)
                status = self._sender_round(bv, input_batch, aux_hidden_states, num_sampled,
                                            num_rejected, last_sampled, next_prefill_tokens,
                                            res)
            except Exception as e:  # never skip the broadcast: rank 1 waits for it
                status = STATUS_ROUND_FAILED
                self._pack_failed(n)
                self.stats.inc("fallback_internal_error")
                if self.slots is not None:
                    self.slots.mark_all_dirty()
                now = time.monotonic()
                if now - self._err_logged > 10.0:
                    self._err_logged = now
                    log.exception("remote DSpark round failed on rank 0: %s", e)
            res.status_host = status
            self.bcast.copy_(self.bcast_host, non_blocking=True)
            self._bcast_copied()
            self.stats.observe("rank0_round_us", (time.perf_counter() - t0) * 1e6)
            self.rounds += 1
            self._maybe_log()
        if self.broadcast_fn is not None:
            self.broadcast_fn(self.bcast)
        return res

    def _bcast_copied(self) -> None:
        if self._bcast_ev is not None:
            self._bcast_ev.record()

    def _bcast_writable(self) -> None:
        """Wait until the last copy out of bcast_host has run (pinned memory: the host must
        not rewrite it under a pending non_blocking copy). Rank 0's own streams only."""
        if self._bcast_ev is not None:
            self._bcast_ev.synchronize()

    def n_out_for(self, n: int) -> int:
        """ReqEntry.n_out for a round of n requests (0 = all N positions)."""
        t = self.n_out_by_reqs
        if not t:
            return 0
        c = int(t[min(n, len(t) - 1)])
        return 0 if c >= self.N else max(1, c)

    def _pack_failed(self, n: int) -> None:
        self._bcast_writable()
        b = self.bcast_np
        b[:] = 0
        b[0] = STATUS_ROUND_FAILED
        b[1] = n

    def _sender_round(self, bv: BatchView, input_batch, aux, num_sampled, num_rejected,
                      last_sampled, next_prefill_tokens, res: RoundResult) -> int:
        assert self.gatherer is not None
        res.need_draft = bv.need_draft
        tg = time.perf_counter()
        g = self.gatherer.gather(bv, list(aux), input_batch.positions, num_rejected,
                                 num_sampled, input_batch.idx_mapping, last_sampled,
                                 next_prefill_tokens)
        self.stats.observe("gather_us", (time.perf_counter() - tg) * 1e6)
        return self._after_gather(bv, g, res, step_id=self.rounds + 1)

    def _after_gather(self, bv: BatchView, g, res: RoundResult, step_id: int) -> int:
        assert self.gatherer is not None and self.slots is not None and self.client is not None
        n = bv.n
        # rank-0 bookkeeping of the ring (CONTRACT 7.1, 7.3). ready + epoch are read once, together:
        # every entry below is built for this connection, and step() refuses any newer one
        epoch = self.client.connection_epoch()
        if epoch is not None and epoch != self._last_epoch:
            self._last_epoch = epoch
            self.slots.mark_all_dirty()
        slots = self.slots
        anchors_pos = g.first_pos + g.valid
        for i in range(n):
            s = int(bv.slots[i])
            st = slots[s]
            slots.begin_request(s, bv.req_ids[i])
            v = int(g.valid[i])
            if v <= 0:
                st.rng.clear()
                st.dirty = True
                continue
            try:
                st.rng.apply(int(g.first_pos[i]), v, int(anchors_pos[i]))
            except ValueError as e:
                log.warning("slot %d ring bookkeeping: %s", s, e)
                st.rng.clear()
                st.dirty = True
        assert self.breaker is not None
        if not self.breaker.allow():
            # open breaker: no network this round (CONTRACT 10); the ring above stays exact
            if self.breaker.state == "half_open":
                self.stats.set("breaker", self.breaker.snapshot())
            return self._fail_round(bv, "cooldown")
        if epoch is None:
            reason = "disabled" if self.client.disabled else "not_ready"
            return self._fail_round(bv, reason)
        entries: list[wire.ReqEntry] = []
        spans: list[tuple[int, int, int]] = []
        resync = False
        n_out = self.n_out_for(n)
        for i in range(n):
            s = int(bv.slots[i])
            st = slots[s]
            v = int(g.valid[i])
            anchor_pos = int(anchors_pos[i])
            flags = wire.RF_NEED_DRAFT if bv.need_draft[i] else 0
            first, k = int(g.first_pos[i]), v
            if st.dirty or not st.announced:
                # the worker's window is unknown or new: (re)send the whole ring window,
                # which already holds this round's rows (CONTRACT 7.4)
                flags |= wire.RF_NEW_GEN
                if not st.rng.empty:
                    pos = st.rng.positions(anchor_pos)
                    if (pos.start, len(pos)) != (first, k):
                        first, k = pos.start, len(pos)
                        flags |= wire.RF_RESYNC
                        resync = True
            if k == 0:
                # nothing to send for this request (cannot happen for a scheduled
                # request); an empty NEW_GEN entry would only fail BAD_ENTRY
                return self._fail_round(bv, "no_rows")
            entries.append(wire.ReqEntry(slot=s, generation=st.generation, first_pos=first,
                                         n_rows=k, anchor_token=int(g.anchor[i]),
                                         anchor_pos=anchor_pos, flags=flags,
                                         k_window=int(min(bv.k_window[i], 255)),
                                         n_out=n_out))
            spans.append((s, first, k))
        n_rows = sum(e.n_rows for e in entries)
        if resync:
            if self.async_mode:
                # the whole message is rebuilt from the ring: every slot's window must have been
                # staged by the engine stream this round; otherwise ask for it and fail this round
                need = {sl for sl, _, _ in spans}
                if not need <= self._staged:
                    with self._resync_lock:
                        self._resync_req |= need
                    return self._fail_round(bv, "resync_pending")
                got = self._rows_from_staged(n, spans)
            else:
                got = self.gatherer.gather_from_ring(n, spans)
            if got != n_rows:
                raise RuntimeError(f"resync gathered {got} rows, expected {n_rows}")
            self.stats.inc("resync_rounds")
            self.stats.inc("resync_rows", n_rows)
        elif n_rows != g.n_valid:
            raise RuntimeError(f"{n_rows} rows to send but {g.n_valid} valid rows staged")
        step_flags = wire.SF_WANT_MARGIN if self.cfg.want_margin else 0
        out: StepOutcome = self.client.step(self.gatherer.msg_np, entries, n_rows,
                                            step_id=step_id, step_flags=step_flags, epoch=epoch)
        if not out.ok:
            if out.reason not in ("not_ready", "disabled", "reconnected"):   # sent, and failed
                self._breaker_record(False, out.reason or "unknown")
            return self._fail_round(bv, out.reason or "unknown")
        self.stats.observe("rtt_us", out.rtt_us)
        self.stats.observe("worker_us", out.reply.worker_us)
        self.stats.inc("rows_sent", n_rows)
        if n_out:
            self.stats.inc("rounds_n_out_cut")
        status = self._accept_reply(bv, entries, out.reply, res)
        needed = bv.need_draft[:n].astype(bool)
        if needed.any():
            # failed for the breaker when every request that needed drafts failed on the
            # worker. A round with nothing to draft (prefill appends only) never exercises
            # drafting: neutral, so it neither resets the count nor closes a probe
            usable = np.isin(out.reply.status, list(wire.USABLE_REQ_STATUS))
            self._breaker_record(bool(usable[needed].any()), "every request failed")
        return status

    def _breaker_record(self, ok: bool, why: str) -> None:
        assert self.breaker is not None
        change = self.breaker.record(ok)
        if change is None:
            return
        self.stats.set("breaker", self.breaker.snapshot())
        if change == "tripped":
            self.stats.inc("breaker_trips")
            log.warning("remote DSpark circuit breaker open (%s; trip %d): no network for the "
                        "next %d rounds, then one probe", why, self.breaker.trips,
                        self.breaker.cooldown)
        else:
            self.stats.inc("breaker_recoveries")
            log.info("remote DSpark circuit breaker closed: the probe round succeeded")

    # ------------------------------------------------------------------ async mode

    def _submit_async(self, input_batch, aux, num_sampled, num_rejected, last_sampled,
                      next_prefill_tokens) -> None:
        """Rank 0, async: enqueue the gather (plus any requested resync windows), the gathered
        word, the wait for the release and the reply copy; hand the round to the proxy thread;
        return without blocking the CPU (CONTRACT 4.2, async variant; streamsync.py)."""
        assert self.gatherer is not None and self.sync is not None and self._q is not None
        t = self.rounds + 1
        tq = time.perf_counter()
        staged: frozenset[int] = frozenset()
        try:
            bv = batch_view_from_input_batch(input_batch)
            R = self.gatherer.enqueue(bv, list(aux), input_batch.positions, num_rejected,
                                      num_sampled, input_batch.idx_mapping, last_sampled,
                                      next_prefill_tokens)
            with self._resync_lock:
                want = sorted(self._resync_req)
            if want:
                W, ring = self.W, self.gatherer.ring
                for sl in want:
                    self.resync_host[sl].copy_(ring[sl * W:(sl + 1) * W], non_blocking=True)
                staged = frozenset(want)
                self.stats.inc("resync_windows_staged", len(want))
            item = (t, bv, R, staged)
        except Exception as e:
            # nothing reaches the thread for this round: it releases t with a failed reply
            self.stats.inc("fallback_internal_error")
            log.exception("remote DSpark async enqueue failed: %s", e)
            item = (t, None, 0, frozenset())
        self.rounds = t
        # stream order: the gather's copies (and staged windows), then the word the proxy waits for
        self.sync.gathered(t)
        self._q.put(item)
        self.sync.wait_release(t)
        self.bcast.copy_(self.bcast_host, non_blocking=True)
        self.stats.observe("rank0_submit_us", (time.perf_counter() - tq) * 1e6)

    def _wait_gathered(self, t: int) -> bool:
        """Proxy thread: True once the engine stream finished round t's gather; False when
        stopping. A prefill chunk's forward can take seconds, so there is no give-up."""
        assert self.sync is not None
        t0 = time.monotonic()
        next_log = 30.0
        while not self._stop.is_set():
            if self.sync.wait_gathered(t, 0.1):
                return True
            waited = time.monotonic() - t0
            if waited >= next_log:
                next_log += 30.0
                log.warning("remote DSpark async: the engine stream has not finished round %d's "
                            "gather after %.0f s (words: %s)", t, waited, self.sync.values())
        return False

    def _async_main(self) -> None:
        """Proxy thread. CPU only: no CUDA call of any kind (streamsync.py)."""
        assert self._q is not None and self.sync is not None
        while True:
            item = self._q.get()
            if item is None:
                return
            t, bv, R, staged = item
            with self._resync_lock:
                self._resync_req -= staged     # a request made below persists to the next round
            t0 = time.perf_counter()
            try:
                if not self._wait_gathered(t):
                    return                      # stopping (finally releases t)
                self.stats.observe("gather_us", (time.perf_counter() - t0) * 1e6)
                if bv is None:
                    raise RuntimeError("enqueue failed on the engine thread")
                g = self.gatherer.collect(bv.n, R)
                res = RoundResult(n=bv.n, bcast=self.bcast, layout=self.layout)
                self._staged = staged
                self._after_gather(bv, g, res, step_id=t)
            except Exception as e:  # the stream must always be released
                n = bv.n if bv is not None else 0
                self._pack_failed(n)
                if bv is not None:
                    self.stats.inc("fallback_internal_error")
                if self.slots is not None:
                    self.slots.mark_all_dirty()
                now = time.monotonic()
                if now - self._err_logged > 10.0:
                    self._err_logged = now
                    log.exception("remote DSpark async round %d failed: %s", t, e)
            finally:
                self._staged = frozenset()
                self.sync.release(t)
            self.stats.observe("rank0_round_us", (time.perf_counter() - t0) * 1e6)
            self._maybe_log()

    def _rows_from_staged(self, n_req: int, spans: list[tuple[int, int, int]]) -> int:
        """Async resync: the STEP rows for spans [(slot, first_pos, n_rows)] from the windows
        the engine stream staged (= gatherer.gather_from_ring on the CPU, same ring indexing)."""
        assert self.gatherer is not None
        g, W = self.gatherer, self.W
        total = sum(k for _, _, k in spans)
        if total > g.max_rows:
            raise ValueError(f"{total} resync rows > max_rows {g.max_rows}")
        off = wire.APP_HEADER_SIZE + wire.rows_offset_for(n_req)
        rows = g.msg_np[off:off + total * g.row_bytes].reshape(total, g.row_bytes)
        o = 0
        for sl, p, k in spans:
            rows[o:o + k] = self.resync_np[sl][(p + np.arange(k, dtype=np.int64)) % W]
            o += k
        return total

    def close(self) -> None:
        if self._q is not None:
            self._stop.set()
            self._q.put(None)
            self._thread.join(timeout=5)
            if self.sync is not None:
                # nothing may stay parked on a round the stopped thread never released; the
                # reply bytes do not matter then (drafts never change verified output)
                self.sync.release(self.rounds)
        if self.client is not None:
            self.client.close()

    def _fail_round(self, bv: BatchView, reason: str) -> int:
        assert self.slots is not None
        for s in bv.slots:
            self.slots[int(s)].dirty = True
        self.stats.inc(f"fallback_{reason}")
        self.stats.inc("rounds_failed")
        self._pack_failed(bv.n)
        return STATUS_ROUND_FAILED

    def _accept_reply(self, bv: BatchView, entries: list[wire.ReqEntry],
                      reply: wire.StepReply, res: RoundResult) -> int:
        assert self.slots is not None
        n, N, L = bv.n, self.N, self.layout
        self._bcast_writable()
        b = self.bcast_np
        b[:] = 0
        b[1] = n
        okm = b[L.o_ok:L.o_ok + n]
        ids = b[L.o_ids:L.o_ids + n * N].reshape(n, N)
        conf = b[L.o_conf:L.o_conf + n * N].view(np.float32).reshape(n, N)
        all_ok = True
        for i, e in enumerate(entries):
            rs = int(reply.status[i])
            st = self.slots[e.slot]
            self.stats.inc(f"req_status_{wire.REQ_STATUS_NAMES.get(rs, rs)}".lower())
            usable = rs in wire.USABLE_REQ_STATUS
            if usable:
                st.announced = True
                # a partial window is resent only if the ring holds more than the worker
                # has; otherwise it is genuine (rows never existed) and only counted
                st.dirty = (rs == wire.RS_PARTIAL_WINDOW
                            and int(reply.window_rows[i]) < st.rng.rows_for_anchor(
                                e.anchor_pos))
            else:
                st.dirty = True
            if not bv.need_draft[i]:
                continue
            if usable:
                okm[i] = 1
                ids[i] = reply.draft_ids[i]
                conf[i] = reply.confidence[i]
            else:
                all_ok = False
        res.reply = reply
        stale = [e.slot for i, e in enumerate(entries)
                 if int(reply.status[i]) == wire.RS_STALE_GENERATION]
        if stale:
            # rank 0's generations only grow (CONTRACT 7.1), so the worker's slot state does
            # not match ours; a resync under the same session would stay stale
            self.stats.inc("stale_generation_new_sessions")
            self.client.new_session(f"worker reported STALE_GENERATION for slots {stale}")
        self.stats.inc("rounds_ok")
        status = STATUS_OK if all_ok else STATUS_SOME_FAILED
        if not all_ok:
            self.stats.inc("rounds_some_failed")
        b[0] = status
        return status

    # ------------------------------------------------------------------ logging

    def _maybe_log(self) -> None:
        every = self.cfg.log_every
        if every <= 0 or self.rounds % every:
            return
        if self.breaker is not None:
            self.stats.set("breaker", self.breaker.snapshot())
        snap = self.stats.snapshot()
        snap["rounds"] = self.rounds
        c = snap["counters"]
        lat = snap["latency"]
        log.info(
            "remote DSpark %s: %d rounds, ok %d, failed %d, resync %d; rtt p50/p99 %s/%s us, "
            "worker p50 %s us, gather p50 %s us, rank-0 round p50 %s us",
            self.cfg.policy, self.rounds, c.get("rounds_ok", 0), c.get("rounds_failed", 0),
            c.get("resync_rounds", 0), lat.get("rtt_us", {}).get("p50_us"),
            lat.get("rtt_us", {}).get("p99_us"), lat.get("worker_us", {}).get("p50_us"),
            lat.get("gather_us", {}).get("p50_us"), lat.get("rank0_round_us", {}).get("p50_us"))
        self.write_stats()

    def write_stats(self) -> None:
        if not self.is_sender or not self.cfg.stats_file:
            return
        self.stats.set("rounds", self.rounds)
        if self.breaker is not None:
            self.stats.set("breaker", self.breaker.snapshot())
        try:
            self.stats.write_json(self.cfg.stats_file)
        except OSError as e:
            log.warning("remote DSpark stats file %s: %s", self.cfg.stats_file, e)
