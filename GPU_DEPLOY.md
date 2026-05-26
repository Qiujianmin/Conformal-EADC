# GPU 实验环境部署指南

## 1. 本地目录结构

```
lunwen/
├── src/                        # Python 源码
│   ├── data_utils.py           # FineHarm 数据加载
│   ├── eprocess.py             # 核心 e-process 算法
│   ├── baselines.py            # Baseline 方法 (FixedThreshold, SWMA, DelayK, NaiveSPRT)
│   ├── train_probe.py          # 探针训练脚本
│   ├── offline_profiler.py     # 离线分数 profiling (备选，pipeline已集成)
│   ├── run_pipeline.py         # ★ 主流水线 (训练+评分+评测 一体化)
│   ├── run_streaming.py        # 分步评测 (备选)
│   └── visualize.py            # 生成 6 张论文图
├── outputs/                    # 实验输出
│   ├── best_probe/             # 训练好的 RoBERTa-base 探针 (~476MB)
│   │   ├── config.json
│   │   ├── model.safetensors   # 模型权重
│   │   ├── tokenizer.json
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   └── tokenizer_config.json
│   ├── score_cache/            # 预计算的分数缓存 (~28MB)
│   │   ├── cal_safe_scores.npy # 校准集安全分数 (659,595个)
│   │   ├── cal_safe_lengths.npy# 校准集对应前缀长度
│   │   ├── drift_profiler.json # DKW drift ε_t 参数
│   │   └── test_scores.json    # 测试集每个token位置的探针分数 (2685样本)
│   ├── figures/                # 论文图表 (PNG + PDF)
│   │   ├── fig1_pvalue_diagnostics.*
│   │   ├── fig2_evidence_trajectories.*
│   │   ├── fig3_pareto_fpr_leakage.*
│   │   ├── fig4_pareto_pir_leakage.*
│   │   ├── fig5_alpha_sweep.*
│   │   └── fig6_detection_delay.*
│   ├── streaming_results.json  # 完整实验结果
│   ├── trajectories.json       # 50条样本轨迹 (用于可视化)
│   ├── training_history.json   # 探针训练记录
│   ├── data_split.json         # 训练/校验集划分信息
│   └── experiment_results.md   # 实验结果文档
├── data-FineHarm/              # FineHarm 数据集
│   ├── FineHarm-train.json
│   ├── FineHarm-val.json
│   └── FineHarm-test.json
└── GPU_DEPLOY.md               # 本文件
```

## 2. GPU 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 3090 / 4090 或更高 (需要 ~8GB VRAM) |
| Python | 3.8+ |
| CUDA | 11.x+ |
| 磁盘 | ~2GB 可用空间 |
| 推荐镜像 | AutoDL: PyTorch 2.0 + Python 3.8 |

### Python 依赖

```
torch>=2.0
transformers>=4.30
numpy
scipy
scikit-learn
matplotlib
tqdm
```

## 3. 新 GPU 部署步骤

### 方式 A: 完整恢复（推荐，节省 ~2 小时）

如果你已有本地的 `outputs/` 和 `src/`，直接上传即可跳过训练和评分步骤。

```bash
# 1. 上传源码
scp -P <端口> -r src/ root@<地址>:/root/

# 2. 上传数据集
scp -P <端口> -r data-FineHarm/ root@<地址>:/root/

# 3. 上传已训练的探针
scp -P <端口> -r outputs/best_probe/ root@<地址>:/root/outputs/

# 4. 上传分数缓存 (跳过最耗时的评分步骤, 节省 ~90 分钟)
scp -P <端口> outputs/score_cache/*.npy root@<地址>:/root/outputs/score_cache/
scp -P <端口> outputs/score_cache/*.json root@<地址>:/root/outputs/score_cache/

# 5. 运行评测 (使用 --skip-profiling 跳过已有缓存)
ssh -p <端口> root@<地址>
cd /root
python -u src/run_pipeline.py --skip-profiling
```

### 方式 B: 从零开始（不需要本地 outputs/）

```bash
# 1. 上传源码和数据
scp -P <端口> -r src/ root@<地址>:/root/
scp -P <端口> -r data-FineHarm/ root@<地址>:/root/

# 2. 安装依赖
ssh -p <端口> root@<地址>
pip install torch transformers numpy scipy scikit-learn matplotlib tqdm

# 3. 如果在国内 GPU (AutoDL)，设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 4. 训练探针 (~15 分钟)
python -u src/train_probe.py

# 5. 运行完整 pipeline (~90-120 分钟)
python -u src/run_pipeline.py
```

## 4. 各步骤耗时参考 (RTX 4090)

| 步骤 | 耗时 | 说明 |
|------|------|------|
| 探针训练 | ~15 分钟 | RoBERTa-base, 3 epochs |
| 校准集全量评分 | ~65 分钟 | 4653 样本 × 每个token位置 |
| 漂移分析评分 | ~4 分钟 | 200 验证集样本 |
| 测试集全量评分 | ~90 分钟 | 2685 样本 × 每个token位置 |
| **评测计算** | **~3 分钟** | 5 配置 × 6 alpha × 2685 样本 |
| 图表生成 | <1 分钟 | matplotlib |
| **总计 (从零)** | **~3 小时** | |
| **总计 (有缓存)** | **~5 分钟** | 仅跑评测 + 画图 |

## 5. 关键文件说明

### run_pipeline.py 参数

```bash
python src/run_pipeline.py --skip-profiling    # 跳过已有缓存，直接评测
python src/run_pipeline.py                      # 完整运行 (重新评分)
```

### run_pipeline.py 内部流程

1. **Step 1**: 校准集评分 → `score_cache/cal_safe_scores.npy`
2. **Step 2**: 漂移分析 → `score_cache/drift_profiler.json`
3. **Step 3**: 测试集评分 → `score_cache/test_scores.json` (最耗时)
4. **Step 4**: 评测 → `streaming_results.json` + `trajectories.json`

### 评测配置扫描

当前代码会自动扫描 5 种配置：

| 配置名 | epsilon_multiplier | t_min | 说明 |
|--------|:-:|:-:|------|
| v2 (baseline) | 1.0 | 1 | 原始 ε_t |
| eps_x2 | 2.0 | 1 | ε_t 加倍 |
| **eps_x3 (当前最优)** | **3.0** | **1** | **ε_t 三倍** |
| t_min=5 | 1.0 | 5 | 跳过前 5 个 token |
| eps_x2+t5 | 2.0 | 5 | 组合 |

自动选择 α=0.05 时 FPR 达标且 Power 最高的配置。

## 6. 如果要修改/继续实验

### 调优方向

- **降低 FPR**: 增大 `epsilon_multiplier` (试 4.0, 5.0) 或增大 `t_min`
- **提高 Power**: 减小 `epsilon_multiplier` 或调整 `kappa_min` / `ewma_beta`
- **修改 EADC 调度**: 调整 `eadc_C_max` (最大跳过步数) 和 `eadc_rho` (曲线形状)

### 修改位置

- 算法核心: `src/eprocess.py` 中的 `EProcessEngine` 类
- 评测逻辑: `src/eprocess.py` 中的 `evaluate_batch` 函数
- Baseline 方法: `src/baselines.py`
- 配置扫描: `src/run_pipeline.py` 中的 `configs` 列表
- 图表样式: `src/visualize.py`

### 重新跑某个步骤

```bash
# 只重跑评测 (修改了 eprocess.py 后)
rm /root/outputs/streaming_results.json /root/outputs/trajectories.json
python -u src/run_pipeline.py --skip-profiling

# 重跑校准 (修改了校准策略后)
rm /root/outputs/score_cache/cal_safe_*.npy /root/outputs/score_cache/drift_profiler.json
python -u src/run_pipeline.py --skip-profiling

# 完全重跑 (包括测试集评分)
rm -rf /root/outputs/score_cache/
python -u src/run_pipeline.py
```

## 7. AutoDL 网络注意事项

- **HuggingFace 被墙**: 使用 `export HF_ENDPOINT=https://hf-mirror.com`
- **Python 路径**: AutoDL 默认 `python` 命令不可用，需用 `/root/miniconda3/bin/python`
- **输出缓冲**: 用 `python -u` 关闭缓冲，否则 `tail -f` 看不到输出
- **后台运行**: 用 `nohup python -u ... > log 2>&1 &`

## 8. 当前实验结果 (2025-05-23)

最优配置: **eps_x3** (ε_t × 3)

| α | Prefix-FPR | 达标? | Power | Leakage | PIR |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.01 | 0.0615 | VIOLATED | 0.809 | 62.0 | 0.040 |
| 0.05 | 0.0708 | VIOLATED | 0.794 | 57.6 | 0.034 |
| 0.10 | **0.0734** | **OK** | **0.796** | **54.6** | **0.030** |
| 0.20 | 0.0808 | OK | 0.783 | 52.8 | 0.025 |

详细结果见 `outputs/experiment_results.md`。
