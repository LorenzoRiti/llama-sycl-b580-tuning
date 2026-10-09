# llama.cpp SYCL patches and tuning for Intel Arc B580 (12 GB)

**[Project page](https://lorenzoriti.github.io/llama-sycl-b580-tuning/)** ·
**[Benchmark on Intelinside](https://intelinside.ai/results/229)** ·
**[Full measurements](docs/RESULTS.md)**

Patches and measured recipes for running large MoE models on a **single Intel Arc B580 (12 GB)**
with the llama.cpp **SYCL** backend on Windows.

Headline result: a **35B-parameter MoE model (3B active) at 47–50 t/s decode and 160–200K context
on one 12 GB card**, by offloading the MoE expert layers to the CPU and keeping everything else
on the GPU. The same recipe applies to the whole Gated-DeltaNet MoE family — base models,
distilled and community derivatives — not just one checkpoint.

Second recipe on the same card: **PrismML's Ternary Bonsai 2 27B** (dense, ternary weights,
~1.75 bits/weight) at **196K context** — 53–74 t/s plain, ~49 t/s with a short thinking budget.
That one runs on the `arc-b580` SYCL fork, see [docs/BONSAI.md](docs/BONSAI.md).

Target setup used for all measurements:

- GPU: Intel Arc B580 12 GB (Battlemage), Level Zero, oneAPI 2026.0
- CPU/RAM: AMD Ryzen 5 9600X, 32 GB DDR5-6000
- OS: Windows 11
- Runtime: llama.cpp SYCL, builds around `b10569` / master `9588757` (September 2026)

Nothing here is vendor code or a fork release: each file is a small, focused patch against
contemporary upstream, written to be easy to rebase onto newer `master` (`git apply -3`).

---

## Patches

| # | Patch | What it does | Measured effect |
|---|-------|--------------|-----------------|
| 0001 | `sycl-fattn-parallel-blocks-knob` | Adds the `GGML_SYCL_FA_PB` environment knob: overrides the number of parallel blocks along the KV axis in the SYCL flash-attention kernel (`0` = upstream auto). Includes a 2 GB scratch cap so large overrides cannot exhaust device memory. | +7.0% decode at 44K context (hsk 256 models); flash-attn decode case `hsk=256 kv=32768` 1121 µs → 522 µs (−53.5%) |
| 0002 | `tests-fattn-35b-shapes` | Adds 32 flash-attention perf cases with the exact decode shapes of a 35B-A3B (hs 256, 2 KV heads, GQA 8, KV 4K→88K, F16/Q8_0, 1–4 batch) | Makes the 0001 gain visible and guarded in `test-backend-ops` |
| 0003 | `sycl-unsupported-type-cpu-fallback` | `GGML_ABORT` → `GGML_LOG_WARN` + `nullptr` for unsupported convert/`mul_mat_vec_q` types, so the op falls back to CPU instead of killing the process | Mixed-quant models load instead of aborting; warn + CPU fallback |
| 0004 | `ngram-cache-const-ref` | Replaces per-step copies of ngram parts with `const&` in `common/ngram_cache_*` | Drafting 8.54 µs → 0.90–0.96 µs/token (~9.5×) |
| 0005 | `reasoning-budget-soft-ramp` | Reasoning-budget sampler: progressive end-sequence bias near the budget wall (soft wind-down instead of hard forcing) + a short anti-empty window (EOS suppressed for 3 content tokens after the block closes) | Cleaner reasoning-budget behaviour; no more injected end-tokens mid-word, no empty answers after short reasoning |
| 0006 | `msvc-avx512bf16-include` | Adds the missing `#include <immintrin.h>` for the MSVC build with `__AVX512BF16__` | MSVC + AVX-512 BF16 builds compile |
| 0007 | `tests-moe-expert-shapes` | `MMID_BENCH=1` switches `test-backend-ops` to MoE expert matmul shapes (256 experts, 8 active, 2048×512 and transpose) over IQ2_XXS / IQ2_S / Q3_K / Q4_K | Reproducible MoE-kernel benchmarking |

Apply from the llama.cpp root, in order:

```bash
for p in patches/00*.patch; do git apply -3 "$p"; done
```

## Build (Windows + oneAPI)

```bat
cmake -B build -G Ninja ^
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_CURL=OFF
cmake --build build -j
```

`GGML_SYCL_F16=ON` alone measured **+16% prefill** on this card (pp2048 691 → 801 t/s,
pp8192 672 → 776 t/s) with identical perplexity (PPL wikitext-2: 6.336 → 6.344).

## Runtime environment

```bat
set ONEAPI_DEVICE_SELECTOR=level_zero:0
set GGML_SYCL_FA_PB=160        :: patch 0001: best for hsk 256 (35B-A3B class)
set GGML_SYCL_HOST_PINNED_MEM=0 :: see "RAM" below — ~7 GB less RAM held by the driver
```

## Recipes

### 35B-A3B on a 12 GB card (MoE experts on CPU)

The model does not fit in VRAM; `--n-cpu-moe` keeps the expert FFNs on the CPU while attention,
Gated DeltaNet and shared experts stay on the GPU. Everything is read at DDR5 speed for the
expert part, GPU speed for the rest.

Measured on 35B-A3B (3B active), `-ub 1024 -b 2048`, `-fa on`:

| Quant | `--n-cpu-moe` | context | short decode | decode @43.5K | notes |
|---|---|---|---|---|---|
| IQ2_XXS (mixed) | 24 | 200,000 | 50.3 t/s | 43.0 t/s | daily driver, VRAM ~10.8 GB |
| IQ2_XXS (mixed) | 20 | 180,224 | 52.0 t/s | 47.1 t/s | needs ~11.2 GB VRAM |
| IQ2_XXS (mixed) | 16 | 163,840 | 55.8 t/s | 48.9 t/s | ~11.7 GB VRAM, RAM 77% |
| Q3_K_M (mixed) | 28 | 163,840 | 47.8 t/s | 43.0 t/s | 12.8 GB mix: Q3_K/Q4_K/IQ2_S by depth |

Both weight files are community **mixed quants** (declared file types: Q3_K_M / IQ2_XXS): higher
precision on early layers, IQ2 on the deepest ones — that mix is part of why these numbers hold up.
Quant column shows the declared type; per-tensor composition verified from the GGUF headers.
Same recipes work with single-type quants, expect somewhat different weights/VRAM.

Prefill at 43.5K context: **638–867 t/s** depending on quant and `n-cpu-moe`.

Short-context sweet spot (65K context, `--n-cpu-moe 8`): **63 t/s**.

### Works across the family (Gated DeltaNet MoE, `qwen35moe`)

The same recipe and the FA_PB knob apply to the whole Gated-DeltaNet MoE family, not just one
model: base, distilled and community (abliterated) derivatives all share the architecture
(`qwen35moe`, 256 experts x 2.6B, ~3B active) and the same attention shapes. Measured on the
same single B580:

| Model | Weights | `--n-cpu-moe` | context | short decode | decode @depth | prefill | notes |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B ("Nano" mix) | IQ2_XXS, 10.9 GB | 24 | 200,000 | 50.3 t/s | 43.0 t/s @43.5K | 690 t/s @43.5K | production daily driver |
| Qwen3.6-35B-A3B ("Mini" mix) | Q3_K_M, 12.8 GB | 28 | 163,840 | 47.8 t/s | 43.0 t/s @43.5K | 638 t/s @43.5K | Q3_K/Q4_K/IQ2_S mix by depth |
| Occamy-1.0 (abliterated 35B-A3B) | Q3_K_M, 12.5 GB | 24 | 200,000 | 50.5 t/s | - | - | quality gate 1.000/3 tasks |
| Qwen3.8-35B-Distill (abliterated) | 13.7 GB mix | 32 | 131,072 | 41.5 t/s | 31.6 t/s @53K | 454-456 t/s @4K/53K | + MTP draft head, acceptance 90-100%, VRAM 11 GB |
| Ornith-1.5-35B-A3B (abliterated) | 10.9 GB mix | - | 200,192 | - | 45.0 t/s @43.5K | 890 t/s @43.5K | |
| Qwen3.8-4B-Distill (dense GDN) | Q4_K_M, 2.8 GB | - | 16,384 | ~90 t/s | - | - | always-on resident model |

Prefix prefill for the Qwen3.8 distill measured at 4K and 53K context (454/456 t/s), decode
41.5 t/s short and 31.6 t/s at 53K- that shape of the curve is typical for the family with
`--n-cpu-moe 32`. All 35B-A3B weight files above are community mixed quants (declared file
types Q3_K_M / IQ2_XXS); composition verified per tensor from the GGUF headers.

### Ternary 27B (Bonsai 2) at 196K — needs the `arc-b580` fork

A different shape of model on the same card: **PrismML's Ternary Bonsai 2 27B** (ternary weights,
~1.75 bits/weight, 5.9 GiB; derived from Qwen3.8-27B). Stock llama.cpp cannot read the ternary
`PTQ1_0` format — this model runs on the [arc-b580 fork](https://github.com/Torchit1/llama.cpp)
(XMX ternary kernels, q4_0 decode attention, MTP drafting).

Measured at 196K on the same B580: **53–74 t/s** plain, **~49 t/s** with a 2K thinking budget,
**115 t/s** on large code edits at 64K, exact needle recall at 196K. Three findings matter more
than the flags:

- the fork's XMX weight repack (`GGML_SYCL_PTQ1_T2`) costs ~+1.6 GB VRAM and is a **net loss at
  196K** — leaving it off is 3× faster (74 vs 25 t/s) because it eliminates the spill;
- keep `-b 2048`: shrinking the logical batch (`-b 512`) cuts spill but collapses decode 4×;
- thinking budgets: this model thinks **short** — 2K–6K reasoning tokens is the quality sweet
  spot; 12K+ measurably degrades it and costs 4–5× the time.

Full sweep and all tables: [docs/BONSAI.md](docs/BONSAI.md).

### Small always-on model

A 4B-class Q4_K_M (Gated DeltaNet family) runs at **~90 t/s** fully in VRAM with a 16K context —
useful as a resident background assistant while the big model is unloaded.

### Keeping the desktop off the GPU

With the display attached to the iGPU (Ryzen iGPU on the motherboard output), the B580 stays
fully dedicated to the model: +2–4 GB of usable VRAM and no `dwm` VRAM growth. When `dwm`
accumulates multiple GB of dedicated VRAM the same server drops from ~53 to ~36 t/s —
worth checking in Task Manager → GPU → Dedicated memory before trusting a slow number.

## RAM notes (SYCL driver)

`GGML_SYCL_HOST_PINNED_MEM` (default behaviour) makes the Level Zero driver hold large
**pinned host buffers**: measured ~8.5 GB of physical RAM with the model loaded, that never
returns. With `GGML_SYCL_HOST_PINNED_MEM=0` the cost at rest is ~1–1.5 GB and decode speed is
unchanged (80–81 t/s both ways on a 4B). There is a smaller accumulation of ~110 MB per
request that stays until the server restarts; a restart after a long batch run returns the
memory (~8 s).

## Results in detail

Full measurement tables (A/B flash-attention sweep, context/`n-cpu-moe` sweeps, prefill,
prompt-cache behaviour): see [docs/RESULTS.md](docs/RESULTS.md).

## Limitations

- Single-card, Windows-only measurements; other systems will differ.
- Numbers depend on desktop VRAM usage (see above) and on RAM pressure.
- Patches target upstream around September 2026; later `master` may need `git apply -3` or a
  small rebase. Kernel work upstream moves fast — always re-run `test-backend-ops`.

## License

MIT. The patches are derived from [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT, © The ggml authors). See [LICENSE](LICENSE).
