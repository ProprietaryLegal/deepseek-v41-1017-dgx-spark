# PLI Labs x86-host offloads for this image (branch `pli/dgx-spark-offload`, on upstream `b1726644`, image v5)

Two offloads for the two-Spark DeepSeek-V4.1-Flash-1017 serve, measured with image v5 and this repo's patches 0022–0050
on two DGX Spark (GB10) plus an x86 host (AMD EPYC 7302P, 503 GiB RAM, ConnectX-5 Ex at 40 Gb/s to each Spark, RTX 3090s):

| folder | what | measured |
|---|---|---|
| `engram/` + `transport/` | Engram rows (layers 1, 14) served from host RAM over RDMA; overlay of `engram.py` keeps the NVMe reader as exact fallback | TTFT −5.4 % @32K, −5.7 % @150K; decode level; rows byte-identical; 0 fallbacks |
| `dspark/` (+ `transport/`) | the DSpark drafter on an RTX 3090 (worker) + the Spark-side `remote_dspark` speculator overlay; "capacity" policy removes the drafter from the Sparks | KV 989,460 → 1,604,926 tokens at 6 GiB/rank; 4 × 300K concurrent; single-stream decode −3 to −9 %; acceptance unchanged |

Serve-script hooks (all inert when unset), as used on top of `serve/serve_example.sh`:
1. append `${OFFLOAD_DOCKER_ENV:+ $OFFLOAD_DOCKER_ENV}${OFFLOAD_DSPARK_ENV:+ $OFFLOAD_DSPARK_ENV}` to `DOCKER_ENV`;
2. `KV_BYTES=${PROD_KV_BYTES:-4026531840}` and `MAX_LEN=${PROD_MAX_LEN:-327680}`;
3. launch with a launcher that accepts `EXTRA_DOCKER_ARGS` (see the matching sudoingX fork branch) when
   `OFFLOAD_DOCKER_ARGS` is set (read-only bind mounts of `remote_dspark/`, the `spec_decode/__init__.py` overlay and the
   transport directory); refuse a half-enabled drafter configuration.
Example settings: `production-D.env.example`. Full procedures: `engram/INTEGRATION.md`, `dspark/INTEGRATION.md`;
wire contract: `dspark/CONTRACT.md`.

Two notes for this patch set:
- **GB10 stream-wait deadlock.** Parking a stream on `cuStreamWaitValue32` while another thread must make any CUDA call to
  release it deadlocks against first-time module loads (lazy-loaded kernels, and Triton's `cuModuleLoadData`, which
  `CUDA_MODULE_LOADING=EAGER` does not cover). The Engram spin reader is safe because its release is a CPU store; our async
  drafter now uses the same pattern. Since measured on GB10 as a separate, torch-free proxy process: −0.9 to −3.9 % per
  decode round with one drafter card, −3.9 to −6.5 % with two, time to first token at parity; in production since
  2026-10-23:
  https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff
- **0057/0060 pairing reproduction (allocator only, not yet run on GB10).** The chain 0051→0053→0057→0059(code, NVFP4
  off)→0060 from afe4d5e applies to v5 without rejects; the engine's own allocator run offline reproduces 1,346,928 and
  1,531,258 exactly and gives 2,502,600 tokens at 6 GiB/rank with the drafter removed.

Entries: [combined stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
[findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings) ·
[GitHub](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload).
All credit for the image, patches and quant to HeNryous; this folder is Apache-2.0 like the repo, by PLI Labs (ProprietaryLegal).
