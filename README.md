
# 代码说明

面向 A 股 K 线多步预测的对比实验研究：**LSTM、原生 Transformer、Kronos 预训练模型（微调 / zero-shot）、LightGBM 基线**在同一数据口径与同一选股策略下的系统对比。预测目标：根据过去的日 K 线预测沪深300未来五天收益最高的股票组合（最多不超过 5 只，实际按三池策略最多输出 3 只）。

## 环境配置

- Python 3.10
- PyTorch 2.0.1 / CUDA 11.7
- 其他依赖见 `requirements.txt`：`pip install -r requirements.txt`（LightGBM 基线需 `lightgbm>=4.0`）

## 目录结构

### 本地开发布局（当前仓库）

```
LSTM-VanillaTransformer-Kronos/
├── code/
│   ├── kronos/         # Kronos 模型代码（featurework.py / train.py / test.py）
│   ├── lstm/           # LSTM 对照模型（train.py / test.py）
│   ├── transformer/    # Vanilla Transformer 对照模型（train.py / test.py）
│   ├── lgbm/           # LightGBM 表格基线（train.py / test.py）
│   └── common/         # 公共层（模型无关，单向依赖，见下文"公共模块架构"）
│       ├── paths.py        # 路径与环境变量单点配置
│       ├── config.py       # 种子/确定性/TF32/设备/并行度统一配置
│       ├── data_io.py      # 数据读取/清洗/交易日历对齐（纯 pandas，无 torch）
│       ├── dataset.py      # 随机种子 / 训练样本生成 / K 线 Dataset（torch，训练专用）
│       ├── featurework.py  # 13 维特征工程（6 原始 + 7 衍生，训练/推理端到端复用）
│       ├── strategy.py     # 三池选股 + 预测/组合结果持久化（各 test.py 共用）
│       ├── logger_utils.py # 训练日志 / 配置快照 / 指标 CSV
│       ├── evaluate.py     # 多模型统一评估（IC / RankIC / 回测）
│       └── data_utils.py   # 兼容壳（re-export 上述模块，保持旧 import 可用）
├── utils/              # 数据获取工具（get_data.py 抓取、merge_stock_data.py 合并）
├── data/               # stock_data.csv、hs300_stock_list.csv、trade_calendar.json、样本缓存 pkl
├── model/              # 各模型训练权重与结构配置（kronos / lstm / transformer / lgbm）
├── output/             # 各模型预测结果（kronos / lstm / transformer / lgbm）与评估汇总
├── logs/               # 训练日志、配置快照、指标 CSV（见"日志与实验记录"）
├── temp/               # 临时文件（比赛镜像约定目录）
├── requirements.txt    # Python 依赖
└── README.md
```

> 注：`Dockerfile`、`train.sh`、`test.sh`、`init.sh` 等入口脚本位于比赛镜像内（`/app/` 布局），不在本仓库；`pretrained/` 预训练权重体积过大不入库（镜像内含离线缓存，本地可用环境变量指向任意位置，见"Kronos 算法"节）。

## 数据

使用沪深300成分股历史日K线数据（2020-01-01 至 2026-07-24），包含开盘价、最高价、最低价、收盘价、成交量、成交额。数据通过 baostock 平台获取（`utils/get_data.py` 抓取各年份分片，`utils/merge_stock_data.py` 合并为 `data/stock_data.csv`）。

- 默认训练使用区间内全部数据（Kronos 口径，不划分验证集）
- **时间划分防泄露**：论文实验设置环境变量 `TRAIN_START_DATE` / `TRAIN_END_DATE` 控制训练数据区间（样本缓存按划分自动区分），确保训练期与测试期不重叠

## 预训练模型

使用 Kronos-small 预训练模型（NeoQuasar/Kronos-small）和 Kronos-Tokenizer-base（NeoQuasar/Kronos-Tokenizer-base），通过 Hugging Face 获取。模型权重文件的 MD5 值已报备至赛事方邮箱。比赛镜像内预训练权重存放于 `/app/pretrained/` 目录下；本地运行 Kronos 训练时可通过环境变量指向权重位置：

```bash
KRONOS_TOKENIZER_PATH=/path/to/Kronos-Tokenizer-base \
KRONOS_PREDICTOR_PATH=/path/to/Kronos-small \
python code/kronos/train.py
```

## 公共模块架构

`code/common/` 按职责分层，依赖方向单向（外层模型脚本 → 公共层），无循环依赖：

```
paths.py（路径配置，零依赖）   config.py（种子/确定性/设备/并行度，零依赖）
  ↑
data_io.py / featurework.py / logger_utils.py / strategy.py（纯 pandas/numpy + logging）
  ↑
dataset.py（torch，仅训练链路）
  ↑
code/{kronos, lstm, transformer, lgbm}/{train, test}.py（模型层，只依赖 common）
```

- **数据口径单点维护**：训练样本生成（`dataset.generate_samples`）与推理预处理（`data_io.load_stock_dataframe`）使用同一份数据读取与日历对齐逻辑，训练/推理分布严格一致
- **选股策略单点维护**：四个模型的 `test.py` 共用 `strategy.select_portfolio()`（三池策略一份实现）
- **配置单点维护**：种子/确定性/TF32/并行度统一由 `config.py` 提供；路径统一由 `paths.py` 提供；窗口参数由各模型 `config.json` 提供（推理时自动读取，不会与训练脱钩）
- **兼容性**：`data_utils.py` 为 re-export 壳，旧的 `from common.data_utils import ...` 写法继续有效

## 统一配置表

四个模型的配置来源完全一致，仅模型专属超参不同：

| 配置项 | 唯一来源 | 环境变量 | Kronos | LSTM | Transformer | LightGBM |
|---|---|---|---|---|---|---|
| 随机种子 | `common/config.py` | `SEED` | 42 | 42 | 42 | 42 |
| 训练设备 | `common/config.py` | — | 自动（cuda 优先） | 同 | 同 | CPU |
| 确定性开关 | `common/config.py` | — | cuDNN 确定性 + 确定性算法 + `CUBLAS_WORKSPACE_CONFIG` | 同 | 同 | `deterministic=True` |
| TF32 | `common/config.py` | `USE_TF32` | 默认关 | 默认关 | 默认关 | n/a |
| DataLoader 进程 | `common/config.py` | `NUM_WORKERS`（Kronos 用 `KRONOS_NUM_WORKERS`） | 0 | 4 | 4 | n/a |
| 权重目录 | `common/paths.py` | `MODEL_DIR` | `model/kronos` | `model/lstm` | `model/transformer` | `model/lgbm` |
| 输出目录 | `common/paths.py` | `OUTPUT_DIR` | `output/kronos` | `output/lstm` | `output/transformer` | `output/lgbm` |
| 数据文件 | `common/paths.py` | `DATA_FILE` / `HS300_FILE` / `DATA_DIR` | 四模型共用同一份 | ← | ← | ← |
| 日志目录 | `common/paths.py` | `LOG_DIR` | `logs/` | ← | ← | ← |
| 时间划分 | `dataset.py` / 各 train | `TRAIN_START_DATE` / `TRAIN_END_DATE` | 支持 | 支持 | 支持 | 支持 |
| 输入窗口 | 各模型 `config.json` | — | 512 | 60 | 60 | 60 |
| 预测窗口 | 各模型 `config.json` | — | 5 | 5 | 5 | 5 |
| 选股策略 | `common/strategy.py` | — | 四模型共用三池策略 | ← | ← | ← |
| 结果文件 | `common/strategy.py` | — | `result.csv` / `result_full.csv` / `result_portfolio.csv` | ← | ← | ← |
| 模型配置写盘 | 各 train | — | 训练中写 | **训练开始即写** | **训练开始即写** | **训练开始即写** |

> `config.json` 训练开始即写盘：即使训练中断，`test.py` 也能正常加载模型结构与窗口参数。

### 硬件适配建议（RTX 4060 Laptop 8G + i9 + 16G 内存）

| 项 | 建议 | 说明 |
|---|---|---|
| `NUM_WORKERS` | 4（默认） | i9 多核；LSTM/Transformer 的训练瓶颈是单进程数据准备 |
| `USE_TF32` | 跑 Kronos 时设 1 | 4060 支持 TF32，Kronos 是耗时大头（预期提速 20-40%）；为保证历史结果可复现，默认关闭 |
| LSTM/Transformer `batch_size` | 256 | 8GB 显存充裕 |
| Kronos `batch_size` | 12 | seq 512 + Kronos-small，8GB 刚好；若 OOM 降至 8 |
| 内存 | 无压力 | 376MB 样本缓存 + 4 worker 副本约 2GB |

## featurework.py 来源

`code/kronos/featurework.py` 由 Kronos 官方开源代码中的 `kronos.py` 和 `module.py` 两个文件合并而成。`kronos.py` 包含 KronosTokenizer、Kronos 模型类和 KronosPredictor 推理封装类，`module.py` 包含 TransformerBlock、MultiHeadAttention、BSQuantizer 等底层模块。合并时未做任何逻辑修改，仅将两个文件的 import 语句统一放到文件头部。

（注意与 `code/common/featurework.py` 区分：后者是 LSTM/Transformer/LightGBM 共用的 13 维特征工程模块，二者用途不同。）

## Kronos 算法

### 整体思路介绍
采用 Kronos 时序生成模型对沪深300成分股进行未来5日价格预测。Kronos 将连续K线数据通过 Tokenizer 离散化为 token，再由基于 Transformer 的 Predictor 自回归预测未来 token，最后解码为价格。微调后的模型对所有股票进行预测，按预期收益率排序后进行多池筛选，选出3只股票构建投资组合。

### 网络结构
- **Tokenizer**：编码器-解码器结构，结合 Binary Spherical Quantization (BSQ) 将连续数据压缩为离散 token
- **Predictor**：多层 Transformer Decoder，包含层次化 Embedding、旋转位置编码 (RoPE)、Dual Head 预测头

### 损失函数
- Tokenizer：MSE 重建损失 + BSQ 量化损失
- Predictor：交叉熵损失（s1 和 s2 两个层级的 token 分类）

### 数据扩增
输入数据采用 3 倍标准差缩尾处理（Winsorize），将超过均值±3倍标准差的数值截断，防止极端值干扰模型预测。

### 算法的其他细节
- 上下文窗口 512 个交易日（约2年）
- 预测窗口 5 个交易日
- 训练时使用固定随机种子（42）确保可复现
- 所有 cuDNN 确定性开关已开启（`cudnn.deterministic=True`、`use_deterministic_algorithms=True`）
- DataLoader 使用固定 generator 和 worker_init_fn
- 设置了 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 确保矩阵乘法确定性

### 训练流程
1. `generate_finetune_data()`：从 `stock_data.csv` 读取原始数据，过滤沪深300成分股，对齐交易日历填补停牌日，以滑动窗口方式（步长10）切分训练样本（512天输入→5天输出），保存为 pkl 文件
2. `train_tokenizer()`：加载预训练 Tokenizer，用训练样本微调 12 轮，使用 AdamW 优化器（lr=5e-6）和余弦退火学习率调度，保存最佳模型到 `model/kronos/tokenizer/`；设置环境变量 `KRONOS_SKIP_TOKENIZER_TRAIN=1` 可跳过此步（消融实验）
3. `train_predictor()`：加载预训练 Predictor，冻结 Tokenizer 参数后微调 25 轮，使用 AdamW 优化器（lr=5e-6），早停策略为验证损失连续 8 轮不下降即停止，保存最佳模型到 `model/kronos/predictor/`
4. 注意力计算使用 SDPBackend.MATH 后端，确保完全确定性

### 推理流程
1. 数据预处理：加载并清洗股票数据，过滤沪深300成分股，填充缺失值
2. 遍历每只股票，取最近 512 天K线数据，经缩尾处理后输入模型
3. 模型以温度 T=0.8、top_p=0.9 进行单次采样预测，输出未来 5 天预测值
4. 计算每只股票的预期收益率 `(T+5_open - T+1_open) / T+1_open`
5. 按三池筛选策略选出最多3只股票，分配权重（0.4/0.3/0.3）
6. 输出结果到 `output/kronos/result.csv`，格式为 `stock_id,weight`

## 选股策略（全模型统一）

选股采用三池筛选策略，结合高收益与风险对冲（实现在 `code/common/strategy.py`，所有模型共用）：

1. **超高收益池**（预期收益率 > 10%）：取距离池内均值最近的1只股票，权重 0.4
2. **高收益池**（预期收益率 5%~10%）：先过滤掉与池均值偏差超过 1% 的股票，再取距离过滤后均值最近的1只，权重 0.3
3. **负收益对冲池**（预期收益率 -2%~0%）：从收益绝对值在 0.01~0.018 之间的股票中，计算 V = (超高收益 × 0.4 + 高收益 × 0.3) / 5，选取 V + roi 为正且最接近 0 的股票，权重 0.3

三只股票权重合计 1.0，如某池无满足条件的股票则输出不足3只（满足"最多不超过 5 只"约束）。

## LSTM / Vanilla Transformer 对照实验

为验证 Kronos 的效果，实现了两个端到端训练的对照模型，均预测未来 5 日 K 线（**13 维特征 = 6 原始 + 7 衍生**），供同一套三池选股策略使用：

- **LSTM**（`code/lstm/train.py` + `test.py`）：2 层 LSTM（hidden=256）取最后时刻隐状态，全连接层直接回归未来 5 日序列
- **Vanilla Transformer**（`code/transformer/train.py` + `test.py`）：线性嵌入 + 可学习位置编码 + 因果掩码的 3 层 Transformer 编码器（d_model=128, 8 heads, Pre-Norm），末端全连接回归头

两者数据口径与 Kronos 一致（沪深300成分股过滤、交易日历对齐、按输入窗口自身均值/标准差归一化并截断 ±3σ、固定种子 42 全套确定性配置），保证对比公平：

- 输入窗口 60 个交易日，预测窗口 5 个交易日，滑动步长 5
- **13 维特征工程**（`code/common/featurework.py`）：
  - 原始 6 列：`open / high / low / close / volume / amount`
  - 衍生 7 列：`log_return / range_pct / close_ma5 / close_ma10 / close_ma20 / volatility_5 / volume_ratio`
- AdamW + 余弦退火调度 + 梯度裁剪 + 训练损失早停（patience=8）
- 样本缓存为 `data/samples_lb60_pw5_step5_feat.pkl`（已 gitignore，可自动重建），两模型共用
- 权重与结构配置保存到 `model/lstm/`、`model/transformer/`（`best_model.pt` + `config.json`），test.py 据此自动重建模型

**推理与选股**：读取最近 60 天原始 K 线 → 特征工程 → 归一化 → 模型预测 5 天 → 反归一化取 `T+1_open` / `T+5_open` → 计算 `expected_roi` → 三池选股（`common/strategy.py`）→ 输出 `output/{lstm|transformer}/result.csv`。

## LightGBM 表格基线

`code/lgbm/train.py` / `test.py`：39 维表格特征（最新截面 13 维 + 60 日窗口均值/标准差 26 维），直接回归未来 5 日实际 ROI（与深度模型的 T+1/T+5 定义一致），按时间切分最后 10% 交易日为验证集早停；输出与其他模型同构，纳入统一评估。

## 统一评估（论文实验）

各模型 test.py 除输出竞赛格式 `result.csv` 外，还会累积保存两类文件（按 `pred_date` 去重覆盖，支持滚动多日评测）：

- `output/{model}/result_full.csv`：全部股票的预测 ROI（预测精度评估用）
- `output/{model}/result_portfolio.csv`：每期组合及权重（组合回测用）

统一评估入口：

```bash
py code/common/evaluate.py                  # 自动评估 output/ 下所有有结果的模型
py code/common/evaluate.py lstm transformer # 指定模型
```

产出 `output/evaluation_summary.csv`（论文汇总表）与 `evaluation_detail.csv`（按日期明细）：

- **预测精度**：IC / RankIC（主指标）/ ICIR / t-stat / 方向准确率 / RMSE / MAE
- **组合回测**：期均收益 / 年化 / 夏普 / 最大回撤 / 胜率 / 相对等权基准超额

**Kronos zero-shot 评测**：不微调直接用预训练权重推理（Docker 内执行）：

```bash
docker compose run --rm -e TOKENIZER_DIR=/app/pretrained/Kronos-Tokenizer-base \
    -e MODEL_DIR=/app/pretrained/Kronos-small bdc2026 python /app/code/kronos/test.py
```

**Kronos 消融实验**：三组对比——完整微调（默认）/ 只调 Predictor（`KRONOS_SKIP_TOKENIZER_TRAIN=1` 训练）/ zero-shot（上式推理）。

## 日志与实验记录（logs/）

各模型训练过程统一通过 `code/common/logger_utils.py` 记录到 `logs/` 目录，同一次训练的三类文件使用同一时间戳（`YYYYMMDD_HHMMSS`）互相对应：

1. **训练日志** `logs/train_{模型}_{时间戳}.log`：控制台与文件双通道输出，逐轮记录平均损失、学习率、耗时，以及最佳权重保存事件和早停轮次（Kronos 分 tokenizer / predictor 两个阶段记录）
2. **配置快照** `logs/train_config/{模型}_{时间戳}.json`：完整超参数、随机种子、运行环境（Python / PyTorch / GPU 型号），以及训练代码和输入数据文件的 MD5——任何一次训练结果都可溯源到确切的代码与数据版本
3. **指标数据** `logs/metrics_{模型}_{时间戳}.csv`：逐轮的 `epoch, avg_loss, lr, epoch_time_sec`，可直接用于绘制各模型收敛曲线对比

路径与日志根目录均支持环境变量覆盖（`DATA_DIR / MODEL_ROOT / OUTPUT_ROOT / LOG_DIR` 等，见 `code/common/paths.py`）。

## Docker 使用（比赛镜像）

以下入口脚本位于比赛镜像内（本仓库不含）：

```bash
# 构建镜像
docker compose build

# 交互式进入容器
docker compose run --rm bdc2026 bash

# 训练（在容器内执行）
docker compose run --rm bdc2026 /app/train.sh          # Kronos
docker compose run --rm bdc2026 /app/train.sh lstm     # LSTM 对照

# 预测（在容器内执行）
docker compose run --rm bdc2026 /app/test.sh           # Kronos
docker compose run --rm bdc2026 /app/test.sh lstm      # LSTM 对照
```

### 镜像内目录结构（/app）

```
/app/
├── code/            # 与本仓库 code/ 一致（kronos / lstm / transformer / lgbm / common）
├── data/            # 股票数据、沪深300成分股、交易日历（共享）、样本缓存 pkl
├── pretrained/      # Kronos 预训练权重（离线缓存，MD5 已报备）
├── model/           # 训练权重（kronos / lstm / transformer / lgbm）
├── output/          # 预测结果（kronos / lstm / transformer / lgbm）
├── logs/            # 训练日志、配置快照、指标 CSV
├── temp/            # 临时文件
├── init.sh          # 环境初始化脚本
├── train.sh         # 训练入口脚本（参数: kronos | lstm | transformer）
├── test.sh          # 预测入口脚本（参数: kronos | lstm | transformer）
└── requirements.txt # Python 依赖
```

## 其他注意事项
- 训练和推理均可离线运行，不依赖网络
- 所有随机性已通过固定种子消除，训练和推理过程可复现
- 预训练权重来自 Hugging Face，MD5 已报备，镜像内包含离线缓存
>>>>>>> 03b31b2 (基础准备)
