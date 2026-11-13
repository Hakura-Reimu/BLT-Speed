# 结果文件说明

本目录收录报告（`../REPORT.md`）引用的结构化结果。文件中的路径（`~/blt/...`）是原运行环境里的路径，只作溯源，不需要存在。所有 BPB 都是越低越好；速度比 >1 表示字节模型更快。

## 收录内容

| 目录或文件 | 内容 | 报告章节 |
|---|---|---|
| `main-scores/` | 验证集评分：13 个模型在 0.375 / 0.75 / 1.5 / 3GB 训练量的 BPB，12 个模型在 1 小时活跃训练时间点的 BPB，以及 3 个探索性 pilot 的早期检查点；含逐文档 NLL | §3.1 |
| `test-scores/` | 测试集评分（模型选择冻结后只打开一次） | §3.1 |
| `final-timing/` | 最终配对 prefill 测速：`summary-end_to_end.md/json`（主口径）、`summary-gpu_resident_preprocessed.*`（GPU 常驻口径）、冻结的 graph 缓存策略 `policy/`、运行清单。逐请求原始计时在 `raw-timing.tar.gz`（三轮 `s1`–`s3`、锚点 `anchor`、长度曲线 `curve`），用 `tar -xzf raw-timing.tar.gz` 解压到本目录即可重算 | §3.2 |
| `final-timing-seed1/` | 换一组提示（种子 1）的单轮敏感性测速；原始记录同样在 `raw-timing.tar.gz` | §3.2 |
| `decode/` | 解码测速：`primary/`（三轮主测量和 `summary.md`）、`diag/`（单轮诊断格子）、`diag-fp32/`（CPU FP32 平局检查）、`recheck-0212.json` | §3.3 |
| `tasks/` | PIQA 逐题输出和全部 13 个模型的汇总 `summary.json`（HellaSwag 逐题输出每个约 12MB，未收录） | §3.4 |
| `task-scoring-check/` | 候选答案打分器的独立核对 | 附录 A |
| `longctx-screen/` | 长上下文检索筛查，`summary.json` 为判定；`diag/` 为短上下文与纯复制检查（`sanity.json`）和评分审计（`audit.json`） | §3.5 |
| `probes/` | 预注册的线性探针与鲁棒性结果（两个种子） | §3.6 |
| `probes-posthoc/` | 事后探索性的去哈希探针 | §3.6 |
| `trackA/` | Bolmo‑1B 对 OLMo‑2‑0425‑1B：验证集 BPB（`bpb-val.json`）、prefill 测速（`prefill-partial-timeout.json`，16 格中的 15 格，最后一格因超时缺失）、Thin‑Bolmo 闸门计算（`gate.txt`） | §3.7 |
| `router/a2.json` | Track A2 路由编程：标定、各臂的压缩率与 BPB、张量哈希审计 | §3.8 |
| `diag/` | 探索性诊断：冻结读出、解码器容量续训、300MB 检查点上的损失分解 | §3.9 |
| `lr-screen/`、`lr-confirm/`、`lr-xs/` | 学习率筛选与确认（选定值的依据） | §2.2 |
| `gate/`、`pilot3/` | 0.375GB 质量闸门与训练吞吐 pilot（选定 3GB 训练量和微批的依据） | §2.2 |
| `graph_stress/` | CUDA graph 重放压力测试（全部命中与 eager 逐位一致） | 附录 A |
| `tokenizer_check_*.json` | 快速 GPT‑2 / 16K 分词器与参考实现的逐 token 一致性和速度 | 附录 A |
| `bf16_*.json` | bf16 下分块交叉熵与梯度精度检查（促成训练改用 fp32 归约） | 附录 A |
| `frozen-analysis.json` | 冻结的分析规则，含 9 个分析代码文件的 SHA‑256 | 附录 A |
| `ledger-training.json` | 各训练运行的活跃时间（GPU 账目的依据） | §6 |
| `figures/` | 报告中的三张图，由 `scripts/diag/report_figures.py` 从本目录生成 | §3 |

## 未收录的内容

- 模型检查点和优化器状态。
- 训练逐步日志（`log.jsonl`）和各作业的运行日志（`*.log`）。
- 作废或被取代的测速：第一次训练吞吐 pilot（BPE 窗口填充错误）、按最坏情况补齐的 graph 测速、因 autocast 缓存导致非法显存访问而崩溃的一次重测、随机权重和临时测速。这些运行的 GPU 时间计入报告 §6 的估计。
- 第一次解码运行（一致性检查未过、改为平局规则之前）的记录。
- HellaSwag 逐题输出。
