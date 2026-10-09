# Measurement detail

All numbers from a single Intel Arc B580 12 GB (oneAPI 2026.0, Windows 11), llama.cpp SYCL
around `b10569` / master `9588757`, unless stated otherwise. Decode = token generation rate
from `llama-server` timings; prefill = prompt-processing rate.

## 1. `GGML_SYCL_FA_PB` flash-attention knob (patch 0001)

### test-backend-ops A/B — 153 comparable cases, PB=160 vs auto

- 42 cases improved by >5%, up to **−53.5%** time
  (`hsk=256 kv=32768`: 1121 µs → 522 µs; `kv=16384`: −51.6%; `kv=4096`: −39.4%; `hsk=576 kv=65536`: −27%)
- 39 cases worsened by >5% (all small head sizes, `hsk=64`, up to +135%)
- 72 neutral, median **−1.9%**
- Net win exactly on the hsk 256/512/576 family (Qwen 35B-A3B/27B class) at medium-long context.
- A bug found during the sweep was fixed in the same patch: a fixed override could allocate
  `parallel_blocks × KQV` scratch and OOM the device (`UR_RESULT_ERROR_DEVICE_LOST` at
  hsk=576); the shipped patch caps the scratch at 2 GB.

### End-to-end (35B-A3B, `--n-cpu-moe 24`, c=44K)

| Config | short decode | prefill @44K | decode @44K |
|---|---|---|---|
| FA_PB auto (upstream) | 51.95 t/s | 865.5 t/s | 43.03 t/s |
| FA_PB = 160 | 53.15 t/s (+2.3%) | 867.2 t/s | **46.03 t/s (+7.0%)** |

## 2. `--n-cpu-moe` context sweeps (35B-A3B class)

Runs in this section use a 10.9 GB community mixed quant (declared file type IQ2_XXS;
per-tensor mix Q3_K/Q4_K/IQ2_XXS by depth). Excerpts of recorded sweeps (`-ub 1024 -b 2048`, `-fa on`; VRAM = dedicated usage of
`llama-server` after load):

| Config | context | short decode | prefill @43.5K | decode @43.5K | VRAM | RAM |
|---|---|---|---|---|---|---|
| n-cpu-moe 16 | 163,840 | 55.8 t/s | 739 t/s | 48.9 t/s | 11.7 GB | 77% |
| n-cpu-moe 20 | 180,224 | 52.0 t/s | 705 t/s | 47.1 t/s | 11.2 GB | 76% |
| n-cpu-moe 24 | 200,000 | 50.3 t/s | 690 t/s | 43.0 t/s | 10.8 GB | 77% |
| n-cpu-moe 20 | 200,000 | 53.2 t/s | 718 t/s | 41.4 t/s | 11.7 GB | 93% — too tight |
| n-cpu-moe 16 | 180,224 | 38.1 t/s | 639 t/s | 34.2 t/s | 11.7 GB | 96% — degraded |

Post-reboot end-to-end, desktop clean (200K context):

| `--n-cpu-moe` | short decode | prefill @44K | decode @44K | VRAM |
|---|---|---|---|---|
| 24 | 49–53 t/s | 826–833 t/s | 44.8–47.0 t/s | 9.8 GB |
| 22 | 39–52 t/s | 846 t/s | 47.9 t/s | 10.2 GB |
| 20 | **56 t/s** | 858 t/s | 47.8 t/s | 10.6 GB |
| 18 | 55–58 t/s | 886 t/s | 49.8 t/s | 11.0 GB (marginal) |
| 16 | 30 t/s | 628 t/s | 26.7 t/s | overflow |

Short-context best (`n-cpu-moe 8`, ~65K context): **63.0 t/s** short decode.

Mixed-quant variant (12.8 GB "APEX-I-Mini" recipe, declared Q3_K_M; Q3_K/Q4_K/IQ2_S by depth),
`n-cpu-moe 28`, c=160K → 46.08 / 48.29 / 48.99 t/s short (three runs, 250 tokens each) and
43.01 t/s at 43,560 input tokens with 638.42 t/s prefill.

## 3. `GGML_SYCL_F16=ON` build

| Metric | F16 build OFF | F16 build ON |
|---|---|---|
| pp2048 | 691 t/s | **801 t/s (+16%)** |
| pp8192 | 672 t/s | **776 t/s (+15%)** |
| short decode | 54.0 t/s | 53.6 t/s (equal) |
| PPL wikitext-2 (ctx 2048) | 6.336 | 6.344 (unchanged within noise) |

## 4. Prompt cache

Repeated session prefixes: **63.9% hit rate** (915/1431 tokens), wall time 12.5 s → 5.3 s
(−58%) on the observed session.

## 5. RAM behaviour (SYCL driver)

| Setting | RAM held with model loaded | Decode impact |
|---|---|---|
| `GGML_SYCL_HOST_PINNED_MEM` default (pinned) | ~8.5 GB, never returned while the process lives | none measured |
| `GGML_SYCL_HOST_PINNED_MEM=0` | ~1–1.5 GB at rest | none (80–81 t/s both ways on a 4B) |

With pinned memory off there is still a slow accumulation of ~110 MB per request
(measured: working set 137 → 251 → 363 → 474 MB over four requests). A server restart
after a long batch run returns it; ~8 s at a 4B model load time.

## 6. Desktop VRAM

Measured with the server stopped, the B580 already had 2+ GB of dedicated VRAM in use:
`dwm` alone reached 3.36 GB during a work session. Effect on the same production config:
36 t/s instead of ~53 t/s. Remedies, most effective first:

1. Connect the monitor to the motherboard output (iGPU); the desktop stops using the B580.
2. Restart `dwm` / reboot; close browsers and capture tools before long sessions.
3. Windows Settings → Display → Graphics: set browsers to "Power saving" (iGPU).

## 7. Family coverage (Gated DeltaNet MoE, `qwen35moe`)

Same recipe (`--n-cpu-moe`, `GGML_SYCL_FA_PB=160`) applied to six checkpoints of the same
architecture family — base, distilled and community abliterated derivatives. All runs on the
same B580 machine in the same week; every 35B-A3B weight file is a community mixed quant
(declared types Q3_K_M / IQ2_XXS), per-tensor composition read from the GGUF headers.

| Model | Weights | `--n-cpu-moe` | context | short decode | decode @depth | prefill |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B ("Nano" mix) | IQ2_XXS, 10.9 GB | 24 | 200,000 | 50.3 t/s | 43.0 @43.5K | 690 @43.5K |
| Qwen3.6-35B-A3B ("Mini" mix) | Q3_K_M, 12.8 GB | 28 | 163,840 | 47.8 t/s | 43.0 @43.5K | 638 @43.5K |
| Occamy-1.0 (abliterated 35B-A3B) | Q3_K_M, 12.5 GB | 24 | 200,000 | 50.5 t/s | - | - |
| Qwen3.8-35B-Distill (abliterated, MTP) | 13.7 GB mix | 32 | 131,072 | 41.5 t/s | 31.6 @53K | 454-456 @4K/53K |
| Ornith-1.5-35B-A3B (abliterated) | 10.9 GB mix | - | 200,192 | - | 45.0 @43.5K | 890 @43.5K |
| Qwen3.8-4B-Distill (dense GDN) | Q4_K_M, 2.8 GB | - | 16,384 | ~90 t/s | - | - |

Notes: Occamy-1.0 50.5 t/s is short-context decode with the production config and passed the
3-task quality gate at 1.000. The Qwen3.8 distill runs carry a separate MTP draft head
(acceptance 90–100%). Dense GDN (non-MoE) 4B runs fully on the GPU.

## 8. Bonsai 2 27B (ternary) — `arc-b580` fork, Oct 2026

Different runtime from sections 1–7: these runs use the
[arc-b580 fork](https://github.com/Torchit1/llama.cpp) (commit `fc18b1a`), required for PrismML's
ternary `PTQ1_0` format. Same B580, Windows 11, oneAPI 2026.0, level-zero driver `32.0.101.9034`,
community MTP weight file (6.53 GB). Full sweep: [BONSAI.md](BONSAI.md).

196K context, decode of a large code-edit task (server timings, thinking off), by weight repack:

| `GGML_SYCL_PTQ1_T2` | spill (`llama-server` shared) | decode |
|---|---|---|
| `all` (`-ub 1024`) | ~2,360 MB | 24.8 t/s |
| `ffn` (`-ub 1024`) | ~1,920 MB | 24.6 t/s |
| unset (`-ub 1024`) | ~1,190 MB | 66.8 t/s |
| unset (`-ub 512`) | ~730 MB | **74.2 t/s** |

Quality vs thinking budget (hard subset, code with executed unit tests, greedy): 256 → 0.750;
2048 → 0.950 (199 s); 6144 → 0.950 (478 s); 12288 → 0.742 (902 s — overthinking degrades).
Full 9-task suite: thinking off 0.694; thinking on @2048 0.903–1.000; @6144 0.978.

Speculative decoding (64K, codegen / small edit / large edit, t/s): MTP4 + n-gram 256 →
82.7/87.5/96.6; **MTP4 only → 86.1/88.9/115.2**; MTP6 → 75.1/–/110.3; n-gram only → 46.5/–/–.

Needle test at 196K: exact recall on a 20K-token document, 26 s end-to-end (warm).
Warmup after load: first prefill 568 t/s vs 94 t/s cold (first agent turn ~42 s → seconds).
`llama-bench` short-context: pp512 1013 t/s / tg128 42.2 t/s (repack off), F16 build no gain.
