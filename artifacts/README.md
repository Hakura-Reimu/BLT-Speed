# 复现用的小型产物

这些文件用来重建实验用的数据切分、分词器和模型配置。体积大的产物（训练字节分片、模型检查点）不在本仓库。

| 路径 | 内容 |
|---|---|
| `data/supply-v1/manifest.json` | 数据供给清单：FineWeb‑Edu `sample-10BT` 的源 parquet 文件、文档数、去重数，以及 `train/val/test` 分片的大小和 SHA‑256。本文件**保持逐字节原样**（含原运行环境的绝对路径），因此它的 SHA‑256（`7f7921e5…`）与结果文件里记录的 `data_manifest_sha256` 一致 |
| `data/supply-v1/val.idx`、`test.idx` | 验证集、测试集的文档边界索引（与清单中的哈希一致） |
| `data/supply-v1/tokenized/*/meta.json` | 两个 BPE 分词器的预分词缓存元数据 |
| `tokenizer16k-v1/` | 训练好的 16K 字节级 BPE 分词器（`tokenizer.json`，SHA‑256 `7e5a793c…`，与结果文件里的 `tokenizer_sha256` 一致）及其训练元数据 |
| `data/probe-v1/` | 表示探针用的样例（`examples.jsonl`）、清单和鲁棒性扰动文本。样例取自 FineWeb‑Edu 验证集的短片段 |
| `runs/<run>/` | 13 个主模型的训练配置（`config.json`）、数值与填充策略，字节模型另有切 patch 的标定结果（`calibration.json`） |

FineWeb‑Edu 由 Hugging Face 以 ODC‑By 1.0 许可发布；`data/probe-v1/` 中的文本片段来自该数据集，按其许可署名。

重建数据的方式：下载清单列出的四个 FineWeb‑Edu parquet 分片，运行 `scripts/prepare_data.py`，再用清单里的 SHA‑256 核对生成的分片。训练使用的 CUDA kernel 不保证逐位确定，所以重训得到的是同一实验的近似复现，不是逐位相同的检查点。
