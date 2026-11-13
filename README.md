# BLT-Speed

**Are byte-latent transformers faster than BPE transformers at equal quality? A pre-registered, small-scale controlled study.**

**字节潜在 transformer 能否在质量相当时比 BPE transformer 更快？一项预注册的小规模受控研究。**

Status: completed on 12 November 2026. The result is negative for the tested configurations. The full report is in Chinese: [`REPORT.md`](REPORT.md).

---

## Abstract

Byte-latent models (BLT-style) group raw bytes into variable-length patches and run a large global transformer once per patch rather than once per token. This study asked whether such a model can process text faster than a BPE transformer while keeping held-out bits per byte (BPB) within 0.03 (and 3%). The core model grid and the primary criteria were frozen internally before training ([`docs/preregistration.md`](docs/preregistration.md), published here retrospectively together with the results). Protocol details specified later, amendments and exploratory controls are identified separately in the report.

We implemented both model families from scratch. Their global trunks use the same transformer-block implementation (RMSNorm, SwiGLU, RoPE, GQA with four KV heads, static KV cache). Byte models add small causal local encoders and decoders, delimiter-based causal patching (P≈6 or 8 bytes per patch) and hashed n-gram embeddings. We trained 13 models, with 7.6M–109M parameters, on 3 GB of FineWeb-Edu each. All timing was measured as wall-clock time on a single RTX 4070 (12 GB).

Main findings for the primary pair M‑P6 / M‑BPE16K, which share the same global trunk (lower BPB is better; a speed ratio above 1 means the byte model is faster):

- **Quality.** The byte model's BPB is higher by about 0.23 (byte minus BPE on the test split: +0.229 for seed 0, +0.232 for seed 1). The gap narrows with training (0.357 at 0.375 GB, 0.228 at 3 GB) but remains 7.6–11.9 times the tolerance.
- **Prefill (primary claim): fails.** No byte model enters the equal-quality band. In batch‑1, end-to-end prefill, the paired speed ratio is 1.13× at 8 KiB and 1.25× at 16 KiB, against a requirement of ≥1.20× at both. Including the exploratory XS controls added after inspecting the registered results, every tested byte-model point is dominated at 8 and 16 KiB: a smaller BPE model is both better and faster. Among the registered models alone, S‑BPE50K dominates the M‑scale byte models, but S‑P6 is not dominated.
- **Decode: fails.** The byte model generates more slowly: 0.57× at batch 1 and 0.37× at batch 8. The batch, lengths and comparator were specified after the validation and prefill results, before the decode measurement.
- **Tasks.** No model reached the pre-registered capability floor on PIQA/HellaSwag, so task retention cannot be supported. Byte models score 2.0–2.6 points lower.
- **Long context: screen fails.** Both primary models failed the ≥90% greedy exact-generation screen on a synthetic five-digit retrieval task, so no long-context continuation was run. Separately, in ten-candidate ranking the byte models stayed near chance (about 0.10), whereas M‑BPE16K ranked ≥99.5% correctly up to 4 KiB. Two scoring checks on audited cases (cached versus full-forward scoring; substituting the hidden number) found no scoring error.
- **Representations: fail** on the operational probe and robustness tests. The probe design is structurally unfavourable to byte models; see the report.
- **Off-the-shelf models.** Bolmo-1B has lower BPB than OLMo-2-0425-1B on our validation split (0.694 vs 0.737), but its prompt prefill is 2.2–2.7 times slower. The pre-registered screening estimate (0.99×) did not justify building a slimmed "Thin-Bolmo", so it was not run. A no-gradient router-editing experiment on Bolmo-1B was inconclusive.

These conclusions apply to our implementation, training budget (3 GB per model) and hardware. They are not a general rejection of byte-latent models, and the study did not locate the source of the quality gap.

## 摘要

字节潜在模型（BLT 式）把原始字节切成长度不等的 patch，大的全局 transformer 每个 patch 只算一次，而不是每个 token 算一次。本研究要回答：这种模型能否在留出 BPB 相差不超过 0.03（且相对不超过 3%）的前提下，比 BPE transformer 处理文本更快？核心模型配置和主要判定标准在训练前已在内部冻结（[`docs/preregistration.md`](docs/preregistration.md)，与结果一起事后公开）。后来补定的协议细节、修改和探索性对照，报告里都分开标明。

我们从头实现了两族模型，全局主干用同一个 transformer block 实现。字节模型另有小的因果局部编码器和解码器、按分隔符严格因果地切 patch（每个 patch 约 6 或 8 字节），以及哈希 n‑gram 嵌入。共训练 13 个模型，每个模型 760 万到 1.09 亿参数、训练 3GB FineWeb‑Edu 文本。所有速度都在一张 RTX 4070（12GB）上按墙钟时间实测。

主要发现针对全局主干相同的主配对 M‑P6 / M‑BPE16K（BPB 越低越好；速度比大于 1 表示字节模型更快）：

- **质量**：字节模型的 BPB 比 BPE 高约 0.23（测试集，字节减 BPE：种子 0 为 +0.229，种子 1 为 +0.232）。差距随训练缩小，但仍是容差的 7.6–11.9 倍。
- **prefill（主判定）不通过**：没有任何字节模型进入等质量带。batch 1、端到端口径下，配对速度比 8 KiB 为 1.13 倍、16 KiB 为 1.25 倍，要求两种长度都至少 1.20 倍。算上看过预注册结果之后才加的探索性 XS 对照，所有字节模型在 8 和 16 KiB 都被压过：有更小的 BPE 模型质量更好、速度也更快。只看预注册模型时，S‑BPE50K 压过 M 规模字节模型，但 S‑P6 没有被压过。
- **生成（decode）不通过**：字节模型更慢，batch 1 时 0.57 倍，batch 8 时 0.37 倍。batch、长度和对照方式是在验证集和 prefill 结果之后、解码测量之前补定的。
- **常识任务**：没有模型达到预注册的能力门槛，任务保持无法支持。字节模型低 2.0–2.6 分。
- **长上下文筛查不通过**：在合成的 5 位随机数检索上，两个主模型都没达到"贪心生成完全正确 ≥90%"的筛查标准，所以没有做长上下文续训。另外，在 10 选 1 的候选排序中，字节模型接近随机（约 0.10），M‑BPE16K 在 4 KiB 内排对 ≥99.5%。对审计样例做的两项评分核对（缓存与全前向打分对比、替换藏的数字）没有发现评分错误。
- **内部表示**：按操作化的探针和鲁棒性测试不通过。探针设计在结构上对字节模型不利，详见报告。
- **现成模型**：Bolmo‑1B 的 BPB 低于 OLMo‑2‑0425‑1B（0.694 对 0.737），但读入长提示慢 2.2–2.7 倍。预注册的筛查估计为 0.99 倍，不支持继续做"Thin‑Bolmo"，所以没有执行。在 Bolmo‑1B 上不用梯度的路由编程实验没有得出结论。

这些结论只针对我们的实现、训练量（每个模型 3GB）和硬件，不是对字节潜在模型的一般否定，也没有查明质量差距的来源。

---

## Repository layout / 仓库结构

| Path | Contents |
|---|---|
| [`REPORT.md`](REPORT.md) | Final report (Chinese): verdicts, results, limitations, resource ledger, methods |
| [`docs/preregistration.md`](docs/preregistration.md) | Research plan frozen on 8 November 2026 (Chinese), with the redactions listed at its top |
| [`results/`](results/) | Structured results cited by the report; see [`results/README.md`](results/README.md) |
| [`artifacts/`](artifacts/) | Small artifacts needed to rebuild the experiment: data-supply manifest and split indices, the trained 16K tokenizer, probe examples, and per-run configurations; see [`artifacts/README.md`](artifacts/README.md) |
| `src/blt/` | Models, patcher, hashes, data pipeline, training, scoring, benchmarking, decoding, probes, Track A and router code |
| `scripts/` | Entry points (training, scoring, timing, tasks, probes, Track A, A2) and analysis and figure scripts in `scripts/diag/` |
| `tests/` | Unit and integration tests (causality, cache consistency, resume, scoring, timing protocol) |

The Thin-Bolmo code (`scripts/thin_bolmo_*.py`) was written before the gate decision and was not used for any reported result. Code comments refer to internal review rounds ("Sol round N") and to the lab notebook ("STATUS"); neither record is part of this repository.

## Environment and reproduction / 环境与复现

The experiments ran under WSL2 (Ubuntu 24.04) on one RTX 4070 (12 GB). The scripts assume a working root at `~/blt` (`~/blt/data`, `~/blt/runs`, `~/blt/models`, `~/blt/results`). The recorded versions are PyTorch 2.9.1 (CUDA 12.8), transformers 4.57.3, xlstm 2.0.5 and mlstm-kernels 2.0.2. The versions of the other packages were not recorded, and `pyproject.toml` declares no dependencies. An environment along these lines should work:

```bash
python3.12 -m venv ~/blt/venv && source ~/blt/venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.3 xlstm==2.0.5 mlstm-kernels==2.0.2 tokenizers tiktoken numpy pyarrow scikit-learn matplotlib pytest
pip install -e .
```

transformers, xlstm and mlstm-kernels are needed only for the Bolmo/OLMo experiments (Track A); scikit-learn is used by the probes.

- Data: FineWeb-Edu `sample-10BT`, materialised into immutable byte shards with document-level splits (`scripts/prepare_data.py`). The manifest in `artifacts/data/supply-v1/` lists the source files and the SHA-256 of every split, so a rebuilt supply can be checked. The 16K tokenizer used in the study is included (`artifacts/tokenizer16k-v1/`); `scripts/train_tokenizer16k.py` trains it.
- Training and evaluation commands are generated by `scripts/make_jobs.py` and executed by `scripts/queue_daemon.py`. The final measurement jobs are `scripts/final_timing.sh`, `decode_timing.sh`, `task_eval.sh`, `probe_run.sh`, `trackA_run.sh`, `longctx_screen.sh` and `router_run.sh`.
- Figures: `python scripts/diag/report_figures.py results` regenerates `results/figures/` from the included results.
- Raw per-request prefill timings are compressed in `results/final-timing/raw-timing.tar.gz` and `results/final-timing-seed1/raw-timing.tar.gz`. HellaSwag per-item results are in `results/tasks/hellaswag-items.tar.gz`.
- The included results support re-running the statistical summaries (BPB intervals, timing summaries, task and probe summaries, figures). Retraining reproduces the experiment approximately, not bit for bit, because some CUDA kernels are nondeterministic.

Model checkpoints are not distributed with this repository.

## How this study was produced / 研究方式说明

The study was carried out with AI assistance. Claude (Anthropic) ran the experiments, reviewed all code, made the weight edits and wrote the report. Most backend code was written by GPT-6.1 Terra and reviewed by Claude. Intermediate results and the report were reviewed by GPT-6.1 Sol; its comments were treated as advice and independently checked before adoption. The research lead set the goals and made the scope decisions recorded in the report.

## License

The code, documentation and results produced in this study are released under the [Apache License 2.0](LICENSE). Third-party resources keep their own licenses:

- Bolmo-1B and OLMo-2-0425-1B are not redistributed here.
- PIQA and HellaSwag texts are not included. Per-item task results store a SHA-256 hash of each question and candidate answer in place of the text.
- FineWeb-Edu (ODC-By 1.0) is not redistributed, apart from the short validation-split excerpts used as probe examples in `artifacts/data/probe-v1/`.
