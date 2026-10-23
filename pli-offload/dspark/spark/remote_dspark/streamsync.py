# SPDX-License-Identifier: Apache-2.0
"""Hand-off between the engine stream and the round's proxy thread (async mode).

The engine stream must not run past a remote round before its drafts are in device memory, and the CPU must not block
either. Two 32-bit words in registered, device-mapped host memory carry the hand-off:

    engine thread (enqueues, never blocks):
        gather(t) -> D2H rows/meta -> [WRITE gathered = t] -> [WAIT released >= t] -> H2D reply -> TP broadcast ...
    proxy thread (plain CPU, no CUDA call):
        wait gathered >= t (acquire) -> network round -> write the reply into pinned host memory -> released = t (release)

The proxy thread makes NO CUDA call. The first async design (a side stream on the proxy thread wrote a device flag after
an H2D copy) deadlocked on GB10 twice: a CUDA call on the proxy thread waited behind a module load on the engine thread,
and the module load waited for the device, whose stream was parked on the flag (offload/results/C/lazy-repro/RESULT.md).
With host words the parked stream is released by a CPU store, which needs no CUDA lock. The engine's Engram spin reader
uses the same handshake on GB10 (engram_disk.cpp enable_spin / poll): malloc memory registered Mapped|Portable (not
cudaHostAlloc, which GB10's CPU reaches over an uncached path), cuStreamWriteValue32 / cuStreamWaitValue32 on its device
pointer, CPU acquire loads / release stores.

Ordering:
  * gathered: CU_STREAM_WRITE_VALUE_DEFAULT puts a stream-scoped system fence before the write, so the D2H rows and
    metadata enqueued before it are visible to the CPU once the word is; the proxy's acquire load orders its reads after.
  * released: the proxy's release store orders its reply writes before the word; the engine stream's H2D copy of the
    reply is enqueued after the wait, so the copy engine reads the host bytes only after the wait passed.
  * reuse of the pinned reply and staging buffers: the proxy sees gathered >= t+1 only after the stream ran everything
    enqueued before round t+1's gather, including round t's H2D reply copy, so it may rewrite them for round t+1.

The CPU side of the words lives in hostflag.c (ctypes.CDLL releases the GIL for every call): an acquire-load spin wait
and a release store. ``FakeHostFlagSync`` stands in on CPU tests (the "stream" wait blocks the caller).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time

import torch

CU_STREAM_WAIT_VALUE_GEQ = 0x0          # (int32)(*addr - value) >= 0: wrap-safe
CU_STREAM_WRITE_VALUE_DEFAULT = 0x0     # with a memory fence before the write
CU_MEMHOSTREGISTER_PORTABLE = 0x01
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02

HOSTFLAG_LIB = "libdsv41_hostflag.so"
HOSTFLAG_VERSION = 1
PAGE = 4096
OFF_GATHERED = 0        # written by the engine stream
OFF_RELEASED = 128      # written by the proxy thread (its own cache line)
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hostflag.c")


class StreamSyncError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------- the C helper

def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.dsv41_hf_version.restype = ctypes.c_int
    lib.dsv41_hf_version.argtypes = []
    lib.dsv41_hf_alloc.restype = ctypes.c_void_p
    lib.dsv41_hf_alloc.argtypes = [ctypes.c_size_t]
    lib.dsv41_hf_free.restype = None
    lib.dsv41_hf_free.argtypes = [ctypes.c_void_p]
    lib.dsv41_hf_load.restype = ctypes.c_uint32
    lib.dsv41_hf_load.argtypes = [ctypes.c_void_p]
    lib.dsv41_hf_store.restype = None
    lib.dsv41_hf_store.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.dsv41_hf_wait_geq.restype = ctypes.c_int
    lib.dsv41_hf_wait_geq.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64, ctypes.c_int64,
                                      ctypes.c_int64]
    v = lib.dsv41_hf_version()
    if v != HOSTFLAG_VERSION:
        raise StreamSyncError(f"{lib._name}: hostflag version {v}, expected {HOSTFLAG_VERSION}")
    return lib


def _private_dir(path: str) -> None:
    """Create (or accept) a build directory only this user can write."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise StreamSyncError(f"hostflag build directory {path} is not a private directory of uid {os.getuid()}")


def _compile(cc: str, out_dir: str) -> str:
    with open(_SRC, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    d = os.path.join(out_dir, f"dsv41-hostflag-{os.getuid()}-{digest}")
    _private_dir(d)
    so = os.path.join(d, HOSTFLAG_LIB)
    if not os.path.exists(so):
        tmp = so + f".tmp{os.getpid()}"
        r = subprocess.run([cc, "-O2", "-Wall", "-shared", "-fPIC", "-o", tmp, _SRC],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise StreamSyncError(f"compiling {_SRC} with {cc} failed ({r.returncode}): {r.stderr.strip()[:400]}")
        os.replace(tmp, so)
    return so


def load_hostflag(lib_path: str = "") -> tuple[ctypes.CDLL, str]:
    """libdsv41_hostflag.so: an explicit path, else VLLM_DSPARK_REMOTE_OFL_DIR's copy, else hostflag.c compiled with
    the system C compiler into a private directory under the temp dir. Returns (library, where it came from)."""
    if lib_path:
        if not os.path.exists(lib_path):
            raise StreamSyncError(f"VLLM_DSPARK_REMOTE_HOSTFLAG_LIB={lib_path} does not exist")
        return _bind(ctypes.CDLL(lib_path)), lib_path
    ofl_dir = (os.environ.get("VLLM_DSPARK_REMOTE_OFL_DIR") or "").strip()
    if ofl_dir and os.path.exists(os.path.join(ofl_dir, HOSTFLAG_LIB)):
        p = os.path.join(ofl_dir, HOSTFLAG_LIB)
        return _bind(ctypes.CDLL(p)), p
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise StreamSyncError(f"no {HOSTFLAG_LIB} (set VLLM_DSPARK_REMOTE_HOSTFLAG_LIB or put it in "
                              "VLLM_DSPARK_REMOTE_OFL_DIR) and no C compiler to build hostflag.c")
    p = _compile(cc, tempfile.gettempdir())
    return _bind(ctypes.CDLL(p)), f"{p} (compiled at startup)"


# ---------------------------------------------------------------------------------------------------- CUDA side

class _Driver:
    """The few driver entry points, from the already loaded libcuda (no link-time dependency)."""

    def __init__(self):
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise StreamSyncError(f"libcuda.so.1: {e}") from e

        def sym(*names):
            for n in names:
                f = getattr(lib, n, None)
                if f is not None:
                    return f
            raise StreamSyncError(f"libcuda has none of {names}")

        self.wait = sym("cuStreamWaitValue32_v2", "cuStreamWaitValue32")
        self.write = sym("cuStreamWriteValue32_v2", "cuStreamWriteValue32")
        for f in (self.wait, self.write):
            f.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
            f.restype = ctypes.c_int
        self.register = sym("cuMemHostRegister_v2", "cuMemHostRegister")
        self.register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.register.restype = ctypes.c_int
        self.unregister = sym("cuMemHostUnregister")
        self.unregister.argtypes = [ctypes.c_void_p]
        self.unregister.restype = ctypes.c_int
        self.devptr = sym("cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer")
        self.devptr.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        self.devptr.restype = ctypes.c_int


def _check(r: int, what: str) -> None:
    if r != 0:
        raise StreamSyncError(f"{what} failed: CUresult {r}")


class HostFlagSync:
    """Rank 0's async hand-off (see the module docstring). Engine-thread calls (gathered, wait_release, self_test,
    reset, close) may use CUDA; proxy-thread calls (wait_gathered, release) never do."""

    cuda_free_proxy = True

    def __init__(self, device: torch.device, lib_path: str = "", spin_us: int = 1000, nap_us: int = 0):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise StreamSyncError("HostFlagSync needs a CUDA device")
        self.lib, self.lib_origin = load_hostflag(lib_path)
        self.drv = _Driver()
        self.spin_ns = int(spin_us) * 1000
        self.nap_ns = int(nap_us) * 1000
        # the driver calls below need the device's primary context current on this thread
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        page = self.lib.dsv41_hf_alloc(PAGE)
        if not page:
            raise StreamSyncError("hostflag page allocation failed")
        self.page = int(page)
        try:
            _check(self.drv.register(ctypes.c_void_p(self.page), PAGE,
                                     CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP),
                   "cuMemHostRegister")
        except StreamSyncError:
            self.lib.dsv41_hf_free(ctypes.c_void_p(self.page))
            raise
        d = ctypes.c_uint64(0)
        _check(self.drv.devptr(ctypes.byref(d), ctypes.c_void_p(self.page), 0), "cuMemHostGetDevicePointer")
        self.page_dev = int(d.value)
        self.gathered_host = self.page + OFF_GATHERED
        self.released_host = self.page + OFF_RELEASED
        self.gathered_dev = self.page_dev + OFF_GATHERED
        self.released_dev = self.page_dev + OFF_RELEASED

    # -------------------------------------------------- engine thread (stream-ordered, never blocks the CPU)

    def gathered(self, value: int, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue "gathered = value" after everything already on the stream (fenced)."""
        s = stream or torch.cuda.current_stream(self.device)
        _check(self.drv.write(ctypes.c_void_p(s.cuda_stream), self.gathered_dev, value & 0xFFFFFFFF,
                              CU_STREAM_WRITE_VALUE_DEFAULT), "cuStreamWriteValue32(gathered)")

    def wait_release(self, value: int, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue a wait until released >= value (wrap-safe)."""
        s = stream or torch.cuda.current_stream(self.device)
        _check(self.drv.wait(ctypes.c_void_p(s.cuda_stream), self.released_dev, value & 0xFFFFFFFF,
                             CU_STREAM_WAIT_VALUE_GEQ), "cuStreamWaitValue32(released)")

    # -------------------------------------------------- proxy thread (CPU only)

    def wait_gathered(self, value: int, timeout_s: float) -> bool:
        """True once gathered >= value; False after timeout_s. Releases the GIL while it waits."""
        r = self.lib.dsv41_hf_wait_geq(ctypes.c_void_p(self.gathered_host), value & 0xFFFFFFFF,
                                       int(timeout_s * 1e9), self.spin_ns, self.nap_ns)
        return r == 0

    def release(self, value: int) -> None:
        self.lib.dsv41_hf_store(ctypes.c_void_p(self.released_host), value & 0xFFFFFFFF)

    def values(self) -> tuple[int, int]:
        return (int(self.lib.dsv41_hf_load(ctypes.c_void_p(self.gathered_host))),
                int(self.lib.dsv41_hf_load(ctypes.c_void_p(self.released_host))))

    # -------------------------------------------------- setup (engine thread, no round in flight)

    def reset(self) -> None:
        torch.cuda.synchronize(self.device)
        self.lib.dsv41_hf_store(ctypes.c_void_p(self.gathered_host), 0)
        self.lib.dsv41_hf_store(ctypes.c_void_p(self.released_host), 0)

    def describe(self) -> dict:
        return {"mechanism": "host_flags", "lib": self.lib_origin, "spin_us": self.spin_ns // 1000,
                "nap_us": self.nap_ns // 1000}

    def self_test(self, timeout_s: float = 2.0) -> tuple[bool, str]:
        """On this device: (1) the CPU sees data the stream copied before "gathered" once it sees the word; (2) the
        stream does not pass the wait before the CPU's release, and its copy after the wait reads the bytes the CPU
        wrote before releasing. The CPU never needs CUDA to unblock the stream, so a failed test cannot leave it
        parked: the rescue is a release store."""
        dev = self.device
        main = torch.cuda.current_stream(dev)
        self.reset()
        y = torch.zeros(1, dtype=torch.int32, device=dev)
        x = torch.zeros(1, dtype=torch.int32, device=dev)
        h_out = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        h_in = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        # every kernel this test launches runs once before anything parks the stream (lazy module loading)
        torch.cuda._sleep(1)
        y.fill_(5)
        h_out.copy_(y, non_blocking=True)
        x.copy_(h_in, non_blocking=True)
        torch.cuda.synchronize(dev)
        y.zero_()
        torch.cuda.synchronize(dev)
        try:
            # (1) gathered: a ~10 ms kernel, then the value the CPU must see with the word
            torch.cuda._sleep(20_000_000)
            y.fill_(7)
            h_out.copy_(y, non_blocking=True)
            self.gathered(1, main)
            if not self.wait_gathered(1, timeout_s):
                return False, "gathered word not written within the timeout"
            if int(h_out.numpy()[0]) != 7:
                return False, f"CPU saw gathered before the copied data ({int(h_out.numpy()[0])})"
            # (2) released: the stream parks; the CPU writes the input, then releases
            self.wait_release(1, main)
            x.copy_(h_in, non_blocking=True)
            done = torch.cuda.Event()
            done.record(main)
            time.sleep(0.02)
            if done.query():
                return False, "stream passed the wait before the release"
            h_in.numpy()[0] = 9
            self.release(1)
            t0 = time.monotonic()
            while not done.query():
                if time.monotonic() - t0 > timeout_s:
                    return False, "stream did not pass the wait within the timeout after the release"
                time.sleep(0.001)
            if int(x.item()) != 9:
                return False, f"stream copied {int(x.item())} after the release, expected 9"
            return True, "ok"
        finally:
            self.release(0x7FFFFFFF)           # never leave the stream parked
            torch.cuda.synchronize(dev)
            self.reset()

    def close(self) -> None:
        """Engine thread, after the proxy thread stopped."""
        if self.page:
            torch.cuda.synchronize(self.device)
            self.drv.unregister(ctypes.c_void_p(self.page))
            self.lib.dsv41_hf_free(ctypes.c_void_p(self.page))
            self.page = 0


# ---------------------------------------------------------------------------------------------------- CPU stand-in

class FakeHostFlagSync:
    """CPU tests: "gathered" is immediate (CPU copies are synchronous) and the engine's "stream wait" blocks the
    calling thread until the proxy released."""

    cuda_free_proxy = True

    def __init__(self, timeout_s: float = 10.0):
        self._g = 0
        self._r = 0
        self._cv = threading.Condition()
        self.timeout_s = timeout_s

    def gathered(self, value: int, stream=None) -> None:
        with self._cv:
            self._g = max(self._g, value)
            self._cv.notify_all()

    def wait_release(self, value: int, stream=None) -> None:
        with self._cv:
            if not self._cv.wait_for(lambda: self._r >= value, timeout=self.timeout_s):
                raise StreamSyncError(f"fake wait for release {value} timed out (released {self._r})")

    def wait_gathered(self, value: int, timeout_s: float) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: self._g >= value, timeout=timeout_s)

    def release(self, value: int) -> None:
        with self._cv:
            self._r = max(self._r, value)
            self._cv.notify_all()

    def values(self) -> tuple[int, int]:
        with self._cv:
            return self._g, self._r

    def reset(self) -> None:
        with self._cv:
            self._g = self._r = 0

    def describe(self) -> dict:
        return {"mechanism": "fake"}

    def self_test(self, timeout_s: float = 2.0) -> tuple[bool, str]:
        return True, "fake"

    def close(self) -> None:
        pass
