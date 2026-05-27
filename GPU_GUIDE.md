# GPU 实验环境快速部署指南

本文档用于在新租用的 GPU 服务器上快速复现 Conformal-EADC 的所有实验。

## 1. 环境要求

- Python 3.8+
- NVIDIA GPU（CUDA 12.x 兼容）
- 磁盘空间 ≥ 2GB

## 2. 一键环境搭建

```bash
# 安装 Python 依赖
pip3 install torch transformers scikit-learn scipy numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若默认源超时，使用清华镜像（如上）。耗时约 2-5 分钟。

## 3. 拉取代码

```bash
git clone https://github.com/Qiujianmin/Conformal-EADC.git
cd Conformal-EADC
```

## 4. 传输必要数据（本地 → 服务器）

以下数据被 `.gitignore` 排除，需从本地手动传输。在**本地机器**执行：

```bash
# 设置变量
REMOTE="root@<服务器地址>"
PORT=<端口>
PASS="<密码>"

# 传输校准分数和漂移配置（约 28MB，所有实验必需）
scp -P $PORT -r outputs/score_cache $REMOTE:~/Conformal-EADC/outputs/

# 传输训练好的 RoBERTa 探针模型（约 500MB，BeaverTails 实验必需）
scp -P $PORT -r outputs/best_probe $REMOTE:~/Conformal-EADC/outputs/

# 传输 BeaverTails 测试集（约 500KB，仅 M3 实验需要）
scp -P $PORT -r data-BeaverTails $REMOTE:~/Conformal-EADC/

# 传输 FineHarm 数据集（仅训练探针需要，通常不需要传输）
# scp -P $PORT -r data-FineHarm $REMOTE:~/Conformal-EADC/
```

> **依赖关系图**：
> - M1/M2/M4/M5 实验：仅需 `score_cache/`
> - M3 (BeaverTails)：需要 `score_cache/` + `best_probe/` + `data-BeaverTails/`
> - 探针训练：需要 `data-FineHarm/`（已训练好，通常不需要）

## 5. 运行实验

### 仅需 CPU 的实验（无需 GPU）

```bash
cd ~/Conformal-EADC

# M1: ACI 验证（~2秒）
python3 src/run_m1_aci.py

# M2: EADC 消融（~5秒）
python3 src/run_m2_ablation.py

# M4: 多种子鲁棒性（~10秒）
python3 src/run_m4_multiseed.py

# M5: 超参敏感性（~30秒）
python3 src/run_m5_sensitivity.py

# 补充: 组合 ACI 实验（~2秒，用于 ε_t×3+ACI 联合配置）
python3 src/run_combined_aci.py
```

### 需要 GPU 的实验

```bash
# M3: BeaverTails 跨数据集验证（约 2 分钟，RTX 3090）
# 需要先传输 best_probe/ 和 data-BeaverTails/
python3 src/run_m3_beavertails_v2.py
```

## 6. 实验输出

所有结果保存在 `outputs/` 目录：

| 文件 | 实验 | 说明 |
|------|------|------|
| `m1_aci_results.json` | M1 | ACI vs ε_t 对比（5个α × 多个η） |
| `m2_ablation_results.json` | M2 | EADC+Baseline 消融 |
| `m3_beavertails_results.json` | M3 | BeaverTails 跨数据集（含 Traditional FPR） |
| `m4_multiseed_results.json` | M4 | 5-seed 鲁棒性 + McNemar 检验 |
| `combined_aci_results.json` | 补充 | ε_t×3+ACI 联合实验 |

## 7. 论文中表格与实验的对应关系

| 论文表格 | 数据来源 |
|----------|----------|
| Table 1 (ε_t ablation) | `m1_aci_results.json` 的 eps_x1/eps_x3 行 |
| Table 2 (ACI validation) | `combined_aci_results.json` |
| Table 3 (α sweep) | `m1_aci_results.json` 的 eps_x3 行 |
| Table 4 (EADC ablation) | `m2_ablation_results.json` |
| Table 5 (Main comparison) | `m4_multiseed_results.json` 的 aggregated |
| Table 6 (BeaverTails) | `m3_beavertails_results.json` |
| Table 7 (Multi-seed) | `m4_multiseed_results.json` |
| Table 8 (Sensitivity) | `run_m5_sensitivity.py` 输出 + η 手动运行 |

## 8. 审稿历史与修改记录

| 轮次 | 关键修改 |
|------|----------|
| R1→R2 | 补充 ACI 理论模块 |
| R2→R3 | 新增 M1-M5 全套实验、EADC 消融、跨数据集验证、多种子统计检验 |
| R3→R4 | 修复 BeaverTails 方法论（Trad. FPR 作主指标）、贡献声明校准、算法补 ACI |
| R4 终审 | 修 PIR 不一致、补 ACI err_t 讨论、补 η 敏感性。**论文已通过终审** |

## 9. 常见问题

**Q: 如何重新训练探针？**
需要 `data-FineHarm/` 数据集（需向原作者申请），运行 `python3 src/train_probe.py`。

**Q: 如何修改超参数重新运行实验？**
直接编辑对应的 `run_m*.py` 脚本中的参数列表，然后重新运行。

**Q: 论文 LaTeX 在哪？**
`paper/` 目录（被 `.gitignore` 排除，仅存本地）。编译：`cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`。

**Q: GitHub 推送失败？**
使用 token 认证：`git push https://ghp_xxx@github.com/Qiujianmin/Conformal-EADC.git master`。若网络不稳定，尝试 `git config --global http.version HTTP/1.1`。
