# SPDX-License-Identifier: Apache-2.0
"""Rank-0 device work of a remote round (CONTRACT 4.2, 4.3, 7.3).

Per round, on the engine stream:
  1. candidate rows = the last min(q_i, W) rows of every request's span (CPU-known);
  2. valid rows = the first (n_i - num_rejected_i) of them (num_rejected is on the GPU:
     compaction is done there, no host sync);
  3. rows = cat(taps)[candidates] compacted to the front of a staging tensor, and written
     into the per-slot ring at (slot, pos mod W);
  4. one copy of rows + metadata (valid counts, first positions, anchors) into pinned
     host memory, at the offset the STEP message needs (zero-copy send), then one sync.

Works on CPU tensors too (offline tests): copies are then synchronous and no event is
used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import wire

import logging

log = logging.getLogger("vllm.remote_dspark")


@dataclass
class BatchView:
    """CPU-side facts of a batch, from InputBatch (batch order)."""
    req_ids: list[str]
    slots: np.ndarray       # [n] request-state index (idx_mapping_np)
    qsl: np.ndarray         # [n+1] query_start_loc_np
    need_draft: np.ndarray  # [n] bool
    k_window: np.ndarray    # [n] drafts verified this step (0 for prefill)

    @property
    def n(self) -> int:
        return len(self.req_ids)


@dataclass
class GatherResult:
    valid: np.ndarray       # [n] valid rows per request (rows sent)
    first_pos: np.ndarray   # [n] position of the first candidate (= first valid) row
    anchor: np.ndarray      # [n] anchor token
    sampled: np.ndarray     # [n] bool: num_sampled > 0
    n_candidates: int

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


def batch_view_from_input_batch(input_batch, max_model_len: int | None = None) -> BatchView:
    n = input_batch.num_reqs
    qsl = np.asarray(input_batch.query_start_loc_np[: n + 1], dtype=np.int64)
    q = np.diff(qsl)
    is_pref = np.asarray(input_batch.is_prefilling_np[:n], dtype=bool)
    done = (np.asarray(input_batch.num_computed_prefill_tokens_np[:n], dtype=np.int64) + q
            >= np.asarray(input_batch.prefill_len_np[:n], dtype=np.int64))
    need = ~is_pref | done
    nd = input_batch.num_draft_tokens_per_req
    k_window = (np.zeros(n, dtype=np.int64) if nd is None
                else np.asarray(nd[:n], dtype=np.int64))
    return BatchView(req_ids=list(input_batch.req_ids[:n]),
                     slots=np.asarray(input_batch.idx_mapping_np[:n], dtype=np.int64),
                     qsl=qsl, need_draft=need, k_window=k_window)


class RowGatherer:
    def __init__(self, *, max_slots: int, window: int, hidden: int, n_taps: int,
                 max_rows: int, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 fused: bool | None = None):
        self.max_slots, self.window, self.hidden, self.n_taps = max_slots, window, hidden, n_taps
        self.max_rows = max_rows
        self.device = torch.device(device)
        self.dtype = dtype
        self.row_elems = n_taps * hidden
        self.row_bytes = self.row_elems * torch.tensor([], dtype=dtype).element_size()
        cuda = self.device.type == "cuda"
        self.pin = cuda
        # ring: max_slots x W rows + 1 dump row (invalid candidates land there)
        self.ring_dump = max_slots * window
        self.ring = torch.zeros(max_slots * window + 1, self.row_elems, dtype=dtype,
                                device=self.device)
        self.stage = torch.empty(max_rows + 1, self.row_elems, dtype=dtype, device=self.device)
        self.msg_capacity = (wire.APP_HEADER_SIZE + wire.rows_offset_for(max_slots)
                             + max_rows * self.row_bytes)
        self.msg = torch.empty(self.msg_capacity, dtype=torch.uint8, pin_memory=self.pin)
        self.msg_np = self.msg.numpy()
        # Index uploads use a fresh pinned tensor per round: in async mode the host runs
        # ahead of the GPU, and torch's caching host allocator keeps a pinned block until the
        # non_blocking copy that read it has completed (a reused buffer would be overwritten
        # before its copy ran). The ring gather (proxy thread) has buffers of its own.
        # Every host<->device copy has contiguous source and destination: a strided copy goes
        # through a temporary, and a pageable host temporary makes it synchronous. The index
        # buffers are therefore flat and viewed as [3, R] / [3, n] per round.
        self.idx_dev = torch.empty(3 * max_rows, dtype=torch.int64, device=self.device)
        self.req_dev = torch.empty(3 * max_slots, dtype=torch.int64, device=self.device)
        self.ring_idx_dev = torch.empty(max_rows, dtype=torch.int64, device=self.device)
        # persistent: gather_from_ring runs in the async proxy thread while the engine stream
        # may be parked on the stream flag, so it must not allocate pinned memory there; the
        # buffer is reusable because gather_from_ring waits for its copy before returning
        self.ring_idx_host = torch.empty(max_rows, dtype=torch.int64, pin_memory=self.pin)
        self.ring_stage = torch.empty(max_rows, self.row_elems, dtype=dtype, device=self.device)
        self.meta_dev = torch.zeros(4, max_slots, dtype=torch.int64, device=self.device)
        self.meta_host = torch.zeros(4, max_slots, dtype=torch.int64, pin_memory=self.pin)
        self.meta_np = self.meta_host.numpy()   # collect() runs on the async proxy thread: no torch call there
        self.event = torch.cuda.Event() if cuda else None
        # one-launch Triton gather (fused_gather.py): default on CUDA when Triton imports
        if fused is None:
            fused = cuda and n_taps == 3 and dtype == torch.bfloat16
        if fused:
            from . import fused_gather

            fused = fused_gather.available() and n_taps == 3 and max_slots <= fused_gather.MAX_REQ
        self.fused = bool(fused)
        self.fused_check: str = "off"
        if self.fused and cuda:
            ok, why = self.self_check()
            self.fused_check = why
            if not ok:
                log.warning("remote DSpark fused gather disabled at startup (%s); torch gather", why)
                self.fused = False
        # warm the resync path (index upload, ring index_select, rows copy) at startup
        self.gather_from_ring(1, [(0, 0, 1)])

    # (n requests, num_sampled per request): every batch size up to 4 (the fused kernel
    # masks requests >= n), and a request that sampled nothing (prefill anchor branch)
    _SELF_CHECK_CASES = ((4, (4, 6, 2, 0)), (3, (4, 0, 2)), (2, (0, 6)), (1, (4,)))

    def self_check(self) -> tuple[bool, str]:
        """Fused vs torch path on synthetic 1..4-request batches on this device (bit-identical
        rows, ring rows and metadata expected). Also the fused kernel's JIT warmup."""
        dev = self.device
        H, W = self.hidden, self.window
        gen = torch.Generator(device="cpu").manual_seed(1234)
        q_all = np.array([6, 6, 5, min(W + 9, self.max_rows - 17)])
        nrej = torch.tensor([2, 0, 3, 0], dtype=torch.int32, device=dev)
        ls = torch.arange(10, 10 + self.max_slots, dtype=torch.int64, device=dev).view(-1, 1)
        npf = torch.arange(20, 20 + self.max_slots, dtype=torch.int32, device=dev).view(1, -1)
        try:
            for n, ns in self._SELF_CHECK_CASES:
                if n > self.max_slots:
                    continue
                q = q_all[-n:] if n < 4 else q_all
                T = int(q.sum())
                taps = [torch.randn(T, H, generator=gen).to(torch.bfloat16).to(dev)
                        for _ in range(3)]
                qsl = np.concatenate([[0], np.cumsum(q)])
                bv = BatchView([f"selfcheck{i}" for i in range(n)], np.arange(n)[::-1].copy(),
                               qsl, np.ones(n, bool),
                               np.array([0 if s == 0 else 4 for s in ns]))
                positions = torch.cat([torch.arange(1000 * (i + 1), 1000 * (i + 1) + int(q[i]))
                                       for i in range(n)]).to(dev)
                # num_sampled is sized for max_slots in the engine: entries >= n must be ignored
                nsamp = torch.full((self.max_slots,), 7, dtype=torch.int32, device=dev)
                nsamp[:n] = torch.tensor(ns, dtype=torch.int32)
                nrej_n = torch.full((self.max_slots,), 9, dtype=torch.int32, device=dev)
                nrej_n[:n] = nrej[:n]
                idx = torch.as_tensor(bv.slots, dtype=torch.int32, device=dev)
                args = (bv, taps, positions, nrej_n, nsamp, idx, ls, npf)
                self.ring.zero_()
                a = self.gather(*args)                   # fused (self.fused is True)
                if not self.fused:
                    return False, "fused path raised (see warning)"
                rows_f = self.rows_view(n, a.n_valid).clone()
                ring_f = self.ring.clone()
                self.ring.zero_()
                self.fused = False
                b = self.gather(*args)                   # torch reference
                self.fused = True
                same = (np.array_equal(a.valid, b.valid)
                        and np.array_equal(a.first_pos, b.first_pos)
                        and np.array_equal(a.anchor, b.anchor)
                        and np.array_equal(a.sampled, b.sampled)
                        and torch.equal(rows_f.view(torch.int16),
                                        self.rows_view(n, b.n_valid).view(torch.int16))
                        and torch.equal(ring_f[:-1].view(torch.int16),
                                        self.ring[:-1].view(torch.int16)))
                if not same:
                    return False, f"fused and torch gathers differ ({n} requests)"
        except Exception as e:  # noqa: BLE001
            return False, f"self-check error: {e}"
        finally:
            self.ring.zero_()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        return True, "ok"

    @property
    def device_bytes(self) -> int:
        return ((self.ring.numel() + self.stage.numel() + self.ring_stage.numel())
                * self.ring.element_size())

    @property
    def host_bytes(self) -> int:
        return (self.msg_capacity + self.ring_idx_host.numel() * 8
                + self.meta_host.numel() * 8)

    def rows_view(self, n_req: int, n_rows: int) -> torch.Tensor:
        """[n_rows, row_elems] view of the pinned message buffer where STEP rows go."""
        off = wire.APP_HEADER_SIZE + wire.rows_offset_for(n_req)
        nbytes = n_rows * self.row_bytes
        return self.msg[off:off + nbytes].view(self.dtype).view(n_rows, self.row_elems)

    def _sync(self) -> None:
        if self.event is not None:
            self.event.record()
            self.event.synchronize()

    @torch.inference_mode()
    def gather(
        self,
        bv: BatchView,
        taps: list[torch.Tensor],
        positions: torch.Tensor,
        num_rejected: torch.Tensor,
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> GatherResult:
        """Enqueue + wait + read back (sync mode)."""
        R = self.enqueue(bv, taps, positions, num_rejected, num_sampled, idx_mapping,
                         last_sampled, next_prefill_tokens)
        self._sync()
        return self.collect(bv.n, R)

    def collect(self, n: int, n_candidates: int) -> GatherResult:
        """Read the metadata of the last enqueue (after its copy completed)."""
        mh = self.meta_np
        return GatherResult(valid=mh[0, :n].copy(), first_pos=mh[1, :n].copy(),
                            anchor=mh[2, :n].copy(), sampled=mh[3, :n].astype(bool),
                            n_candidates=n_candidates)

    @torch.inference_mode()
    def enqueue(
        self,
        bv: BatchView,
        taps: list[torch.Tensor],
        positions: torch.Tensor,
        num_rejected: torch.Tensor,
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> int:
        """Enqueue the gather, ring update and the copies into pinned memory on the current
        stream; no host sync. Returns the number of candidate rows."""
        n, W = bv.n, self.window
        if not 1 <= n <= self.max_slots:
            raise ValueError(f"{n} requests, max_slots {self.max_slots}")
        if len(taps) != self.n_taps:
            raise ValueError(f"{len(taps)} taps, expected {self.n_taps}")
        q = np.diff(bv.qsl)
        if (q <= 0).any():
            raise ValueError(f"empty request span in batch: {q.tolist()}")
        nc = np.minimum(q, W)
        cand_start = bv.qsl[1:] - nc
        R = int(nc.sum())
        if R > self.max_rows:
            raise ValueError(f"{R} candidate rows > max_rows {self.max_rows}")
        if self.fused:
            try:
                from .fused_gather import fused_enqueue

                fused_enqueue(self, bv, taps, positions, num_rejected, num_sampled,
                              last_sampled, next_prefill_tokens, nc, cand_start, R)
            except Exception as e:  # compile / launch problem: use the torch path from now on
                log.warning("remote DSpark fused gather disabled (%s); torch gather", e)
                self.fused = False
            else:
                self.rows_view(n, R).copy_(self.stage[:R], non_blocking=True)
                self.meta_host.copy_(self.meta_dev, non_blocking=True)
                return R
        cand_req = np.repeat(np.arange(n, dtype=np.int64), nc)
        cand_r = np.arange(R, dtype=np.int64) - np.repeat(np.cumsum(nc) - nc, nc)
        cand_src = np.repeat(cand_start, nc) + cand_r
        idx_host = torch.empty(3, R, dtype=torch.int64, pin_memory=self.pin)
        ih = idx_host.numpy()
        ih[0], ih[1], ih[2] = cand_src, cand_req, cand_r
        req_host = torch.empty(3, n, dtype=torch.int64, pin_memory=self.pin)
        rh = req_host.numpy()
        rh[0], rh[1], rh[2] = nc, cand_start, bv.slots
        idx = self.idx_dev[:3 * R].view(3, R)
        idx.copy_(idx_host, non_blocking=True)
        req = self.req_dev[:3 * n].view(3, n)
        req.copy_(req_host, non_blocking=True)
        src, creq, cr = idx[0], idx[1], idx[2]
        nc_d, start_d, slot_d = req[0], req[1], req[2]

        nrej = num_rejected[:n].to(torch.int64)
        valid = torch.minimum(torch.clamp(nc_d - nrej, min=0), nc_d)
        off = torch.cumsum(valid, 0) - valid
        is_valid = cr < valid.index_select(0, creq)
        dest = torch.where(is_valid, off.index_select(0, creq) + cr,
                           torch.full_like(cr, R))
        rows = torch.cat([t.index_select(0, src) for t in taps], dim=1)
        if rows.dtype != self.dtype:
            raise TypeError(f"taps are {rows.dtype}, expected {self.dtype}")
        self.stage.index_copy_(0, dest, rows)
        pos_c = positions.index_select(0, src).to(torch.int64)
        ring_dest = torch.where(is_valid, slot_d.index_select(0, creq) * W + pos_c % W,
                                torch.full_like(cr, self.ring_dump))
        self.ring.index_copy_(0, ring_dest, rows)

        idx_req = idx_mapping[:n].to(torch.int64)
        sampled = num_sampled[:n] > 0
        anchor = torch.where(sampled,
                             last_sampled.reshape(-1).index_select(0, idx_req).to(torch.int64),
                             next_prefill_tokens[0].index_select(0, idx_req).to(torch.int64))
        meta = self.meta_dev[:, :n]
        meta[0].copy_(valid)
        meta[1].copy_(positions.index_select(0, start_d).to(torch.int64))
        meta[2].copy_(anchor)
        meta[3].copy_(sampled.to(torch.int64))
        self.rows_view(n, R).copy_(self.stage[:R], non_blocking=True)
        self.meta_host.copy_(self.meta_dev, non_blocking=True)
        return R

    @torch.inference_mode()
    def gather_from_ring(self, n_req: int, spans: list[tuple[int, int, int]]) -> int:
        """Fill the STEP rows from the ring: spans = [(slot, first_pos, n_rows)] in entry
        order. Returns the total rows written (resync rounds, CONTRACT 7.4)."""
        W = self.window
        ring_idx = np.concatenate(
            [s * W + (np.arange(p, p + k, dtype=np.int64) % W) for s, p, k in spans]
        ) if spans else np.zeros(0, dtype=np.int64)
        total = int(ring_idx.size)
        if total > self.max_rows:
            raise ValueError(f"{total} resync rows > max_rows {self.max_rows}")
        if total:
            self.ring_idx_host.numpy()[:total] = ring_idx
            host = self.ring_idx_host[:total]
            idx = self.ring_idx_dev[:total]
            idx.copy_(host, non_blocking=True)
            torch.index_select(self.ring, 0, idx, out=self.ring_stage[:total])
            self.rows_view(n_req, total).copy_(self.ring_stage[:total], non_blocking=True)
            self._sync()
        return total
