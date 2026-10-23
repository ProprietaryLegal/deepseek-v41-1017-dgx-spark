// SPDX-License-Identifier: Apache-2.0
// Host-memory flag words for the async remote round (streamsync.HostFlagSync).
//
// The engine stream and the round's proxy thread hand a round back and forth through two 32-bit words in registered,
// device-mapped host memory: the stream writes "gathered = t" (cuStreamWriteValue32) and waits for "released >= t"
// (cuStreamWaitValue32); the proxy thread waits for the first and writes the second with plain CPU atomics. The proxy
// thread therefore makes no CUDA call at all. A CUDA call there can deadlock against a module load on the engine thread
// while the stream is parked (offload/results/C/lazy-repro/RESULT.md); the engine's Engram spin reader uses the same
// handshake on GB10 (engram_disk.cpp, enable_spin / poll).
//
// Called through ctypes.CDLL, which releases the GIL for the whole call. Build (any host, no dependencies):
//   cc -O2 -Wall -shared -fPIC -o libdsv41_hostflag.so hostflag.c
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DSV41_HF_VERSION 1

static inline int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline void cpu_relax(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("isb" ::: "memory");  // as engram_disk.cpp's spin loop
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

int dsv41_hf_version(void) { return DSV41_HF_VERSION; }

// One zeroed, page-aligned block for the flag words (registered by the caller; malloc memory, not cudaHostAlloc: GB10
// reaches cudaHostAlloc memory over an uncached path, engram_disk.cpp). NULL on failure.
void* dsv41_hf_alloc(size_t bytes) {
  void* p = NULL;
  if (bytes == 0 || posix_memalign(&p, 4096, bytes) != 0) return NULL;
  memset(p, 0, bytes);
  return p;
}

void dsv41_hf_free(void* p) { free(p); }

uint32_t dsv41_hf_load(const uint32_t* addr) { return __atomic_load_n(addr, __ATOMIC_ACQUIRE); }

// Release store: every store this thread made before the call (the reply bytes) is visible before the word is.
void dsv41_hf_store(uint32_t* addr, uint32_t value) { __atomic_store_n(addr, value, __ATOMIC_RELEASE); }

// Wait until (int32)(*addr - value) >= 0, the wrap-safe comparison of CU_STREAM_WAIT_VALUE_GEQ. Acquire loads, so the
// caller's later reads (the gathered rows) are ordered after the word. Spins for spin_ns, then sleeps nap_ns between
// loads (nap_ns <= 0: spin until the deadline). Returns 0 when reached, 1 when timeout_ns passed first.
int dsv41_hf_wait_geq(const uint32_t* addr, uint32_t value, int64_t timeout_ns, int64_t spin_ns, int64_t nap_ns) {
  const int64_t t0 = now_ns();
  int64_t now = t0;
  for (uint32_t i = 1;; i++) {
    if ((int32_t)(__atomic_load_n(addr, __ATOMIC_ACQUIRE) - value) >= 0) return 0;
    const int napping = nap_ns > 0 && now - t0 >= spin_ns;
    if (napping || (i & 63u) == 0) {
      now = now_ns();
      if (now - t0 >= timeout_ns) return 1;
    }
    if (napping) {
      struct timespec ts = {(time_t)(nap_ns / 1000000000LL), (long)(nap_ns % 1000000000LL)};
      nanosleep(&ts, NULL);
    } else {
      cpu_relax();
    }
  }
}
