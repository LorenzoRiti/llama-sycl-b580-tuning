# Bonsai 2 27B (ternary) on the B580 — measured tuning

Second recipe for the same machine: a **dense 27B at 196K context** on the single 12 GB card.
Ternary weights change the memory maths but the constraints are the same ones as the MoE page:
keep the desktop off the GPU, mind the spill, choose the runtime path by context length.

All numbers from the same B580 machine as the rest of this repository — Intel Arc B580 12 GB,
Windows 11, oneAPI 2026.0, level-zero driver `32.0.101.9034` (Oct 2026).

**Runtime: this model needs the [arc-b580 fork](https://github.com/Torchit1/llama.cpp)**
(branch `arc-b580`, commit `fc18b1a`, the v0.8 kernel set). Stock llama.cpp rejects the ternary
`PTQ1_0` format, and the fork carries the XMX ternary kernels, the q4_0 decode attention and the
MTP draft path used here. The patches in this repository (0001–0007) are **not** part of the Bonsai
numbers; they apply to upstream trees around September 2026.

Model: **Ternary Bonsai 2 27B** by PrismML (derived from Qwen3.8-27B; matrix weights in
{−1, 0, +1} with FP16 group scales, ~1.75 bpw; hybrid attention, ~75% linear / 25% full).
Weight file used: community build with the MTP head included, **6.53 GB (5.87 GiB)**.

## Recipe (196K on 12 GB)

```bat
set ONEAPI_DEVICE_SELECTOR=level_zero:0
set GGML_SYCL_HOST_PINNED_MEM=0      :: ~7 GB less RAM held by the driver (same as MoE recipe)
set GGML_SYCL_FA_DEC_DPAS=1          :: decode/verify attention on XMX from the q4_0 cache (Xe2)
set GGML_SYCL_T2_W8A8_MIN=1024       :: long prompt batches via oneDNN int8 GEMM
set GGML_SYCL_FA_ONEDNN_MAX_KV=98304 :: chunked attention above ~96K keys (safe at 128K+)
:: GGML_SYCL_PTQ1_T2: leave UNSET (see finding 1)
:: SYCL_CACHE_PERSISTENT: leave unset (JIT segfault on this driver, intel/llvm #22853)

llama-server.exe -m Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf ^
  -ngl 99 -c 196608 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -np 1 ^
  -ub 512 -b 2048 --jinja ^
  --spec-type draft-mtp --spec-draft-n-max 4 ^
  --reasoning-budget 2048
```

## 1. The XMX weight repack is a net loss at long context on 12 GB

The fork can repack the ternary weights in place into a 2-bit XMX layout (`GGML_SYCL_PTQ1_T2`),
which is faster at short context — but it costs **~+1.6 GB of VRAM on the whole model**
(all tensors), and at 196K that pushes the server over the card and into shared memory.
Measured at 196K (`-ub 1024`, same decode task, server timings, thinking off):

| `GGML_SYCL_PTQ1_T2` | spill (`llama-server` shared) | decode |
|---|---|---|
| `all` | ~2,360 MB | 24.8 t/s |
| `ffn` | ~1,920 MB | 24.6 t/s |
| **unset (off)** | ~1,190 MB | **66.8 t/s** |
| unset, `-ub 512` | **~730 MB** | **74.2 t/s** |

At short context the repack is a win (`llama-bench`, both residual in VRAM):
pp512 1521 t/s / tg128 45.4 t/s with `all` vs pp512 1013 t/s / tg128 42.2 t/s with off.
At 196K the spill it causes costs more than the kernels gain: **off is 3× faster**.

## 2. Batch settings

- `-ub 512` beats `-ub 1024` at 196K: 74.2 vs 66.8 t/s, spill 0.7 vs 1.4 GB (same task).
- `-b 512` (shrinking the logical batch) cut spill further (~0.5 GB) **but collapsed decode to
  ~12 t/s** — a 4× loss, reproducible, likely a tile-selection threshold in the fork's ternary
  kernels tied to `n_batch`. Keep **`-b 2048`**. `-b 1024` measured identical spill and no gain.
- Draft batch (`LLAMA_ARG_SPEC_DRAFT_UBATCH`): 512 vs 256, no measurable difference.

## 3. Speculative decoding: MTP only, depth 4

64K context, temperature 0 (server timings, t/s):

| Draft config | codegen | small edit | large edit |
|---|---|---|---|
| MTP 4 + n-gram-mod 256 (fork default advice) | 82.7 | 87.5 | 96.6 |
| **MTP only, n-max 4** | **86.1** | **88.9** | **115.2** |
| MTP only, n-max 6 | 75.1 | – | 110.3 |
| MTP only, n-max 8 | 67.5 | – | 111.4 |
| n-gram-mod only | 46.5 | – | – |

- The n-gram drafter is a net negative on this card/workloads; removing it also cut the 128K
  spill from ~2.5 GB to ~0.9 GB (it keeps its own draft buffers).
- Deeper MTP drafts (6/8) spend more verify compute than they recover: acceptance is ~45–60%
  on reasoning text, ~90% on edit/repetition text.
- MTP depth 3 is ~20–30% faster on thinking turns, but showed a deterministic quality regression
  on one hard task at budget 6144 (see finding 4); depth 4 is the safe choice.
- The larger MTP-head file (7.01 GB) measured +2% on edits but −0.66 GB of VRAM budget:
  not worth it at 196K.

## 4. Thinking budget: this model thinks SHORT

Quality on four hard tasks (code with executed unit tests + SQL, greedy, same seed), MTP 4:

| Budget | score | wall time |
|---|---|---|
| 256 | 0.750 (one code task broke) | 162 s |
| **2048** | **0.950** | **199 s** |
| 6144 | 0.950 | 478 s |
| 12288 | **0.742** (same task as 256 fails, 9.8K thinking tokens) | 902 s |

Full 9-task suite (5 tasks with executed tests): **thinking off 0.694; on @2048 0.903–1.000;
on @6144 0.978**. Conclusion: the sweet spot is **2K–6K thinking tokens**; giving this model
10K+ tokens of reasoning *degrades* quality (overthinking drift) and costs 4–5× the time.
Default `--reasoning-budget 2048` covers plain clients; adaptive clients (per-request budgets
1K–6K) get the best quality/time mix.

## 5. Context on 12 GB

- Native max is 262,144; every step up to 256K loads on the card (with spill). 196K used daily.
- Footprint at 196K: weights ~6 GB + q4_0 KV ~3.4 GB + draft/buffers ≈ 10.5–11.5 GB against
  ~11.9 GB usable → **0.7–1.8 GB spill depending on desktop apps** (browser + `dwm`). The spill
  does not break correctness; it costs throughput (all decode numbers above are with the
  production desktop state).
- **Needle test at 196K**: exact recall (passphrase at 10/50/90% depth of a 20K-token document),
  measured 26 s end-to-end on the warm path (prefill ~570–770 t/s, warm).
- q8_0 KV rejected at 196K: spill +1.7 GB and decode 15–25 t/s. Draft KV q8_0: no gain.

## 6. First response: warm the server at start

Cold first turn pays the one-time kernel JIT plus the full system-prompt prefill at ~94 t/s
(~42 s for a 4K-token agent prompt). Sending a ~1K-token warmup request right after load
(prefill then runs at **568 t/s**) removes the JIT from the first real turn; subsequent turns
reuse the prompt cache (~3 s on an agent session).

```bat
:: after /health returns ok — warm the W8A8/oneDNN + XMX kernels
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"model\":\"bonsai\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":8}"
```

## 7. Throughput summary (final recipe, 196K, production desktop state)

| Workload | t/s |
|---|---|
| chat, thinking capped at 2048 | ~49 |
| chat, thinking off | 53–74 |
| agent codegen tasks (thinking on) | 40–70 |
| agent large edits, thinking off (64K) | 115 |
| prefill, short prompts (warm) | 500–770 |
| `llama-bench` tg128 / pp512 (short ctx) | 42 / 1013 |

## 8. Knobs measured with no effect

For this model, on this build: `GGML_SYCL_F16=ON` build (1589/46.2 vs 1564/46.0 pp512/tg128),
`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`, `GGML_SYCL_ENABLE_GRAPH`,
`GGML_SYCL_DEVICE_ARCH=bmg`, `GGML_SYCL_ENABLE_VMM=0`, `GGML_SYCL_PRIORITIZE_DMMV`,
`GGML_SYCL_DYNAMIC_PRECISION`, draft ubatch 256 — all neutral within noise.

## Notes

- Single-card, Windows-only measurements; desktop VRAM state moves these numbers by ±20%
  (see the MoE pages for the `dwm` effect — it applies identically here).
- The fork is a moving branch: re-verify `test-backend-ops -o ARCLAB` after rebasing.
- Quality numbers come from a local suite with executed unit tests; they are indicative,
  single-seed, and not a substitute for your own eval.
