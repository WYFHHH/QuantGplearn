# QuantGplearn GPU 因子挖掘框架文档

本文档说明本次重构后新增的 GPU/Torch 路径。原有 CPU 版 `SymbolicTransformer`、`SymbolicRegressor`、`SymbolicClassifier` 保持兼容；新增入口为：

```python
from QuantGplearn.gpu_transformer import GpuSymbolicTransformer
```

## 1. 设计目标

本次重构的目标是把 QuantGplearn 从 Pandas/NumPy 的逐表达式执行框架，扩展为适用于时序与截面量化因子挖掘的 GPU 框架：

```text
QuantGplearn GP 树生成、交叉、变异
        ↓
TensorPanelData: [T, N, F] dense tensor
        ↓
_Program.execute_tensor(): Torch GPU 表达式执行
        ↓
TensorFitness: IC / RankIC / ICIR / Sharpe proxy
        ↓
ProgramEvaluator: 缓存、归一化、异常值处理
        ↓
GpuSymbolicTransformer: GPU 因子挖掘入口
```

其中：

- `T`：时间长度；
- `N`：标的数量；
- `F`：特征数量；
- 单个因子表达式输出形状为 `[T, N]`。

## 2. 新增模块

| 文件 | 作用 |
|---|---|
| `QuantGplearn/tensor_data.py` | 面板 DataFrame 与 Tensor 数据互转 |
| `QuantGplearn/torch_functions.py` | Torch/GPU 版本基础算子、时序算子、截面算子 |
| `QuantGplearn/tensor_fitness.py` | GPU IC、RankIC、ICIR、long-short Sharpe 代理目标 |
| `QuantGplearn/evaluator.py` | ProgramEvaluator，负责执行、归一化、缓存和 fitness 计算 |
| `QuantGplearn/alpha_pool.py` | alphagen 风格的 TensorFactorCalculator 与 GPAlphaPool |
| `QuantGplearn/gpu_transformer.py` | 新增 GPU 版 `GpuSymbolicTransformer` |
| `tests/test_gpu_transformer.py` | GPU 路径 smoke tests，可在 CPU tensor 上运行 |

## 3. 数据格式

输入建议为 MultiIndex DataFrame：

```text
index = [datetime, symbol]
columns = feature columns + target
```

也支持 `datetime`、`symbol` 作为普通列，框架会自动设置为 MultiIndex。

示例：

```python
feature_names = [
    "open", "high", "low", "close", "volume", "quote_volume",
    "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "vwap",
]

model.fit_panel(panel_df, target_col="target")
```

内部会转换为：

```python
values: torch.Tensor  # [T, N, F]
target: torch.Tensor  # [T, N]
mask:   torch.Tensor  # [T, N]
```

## 4. 快速使用

```python
from QuantGplearn.gpu_transformer import GpuSymbolicTransformer
from QuantGplearn.torch_functions import GPU_SAFE_PANEL_FUNCTIONS

model = GpuSymbolicTransformer(
    population_size=1024,
    generations=20,
    hall_of_fame=100,
    n_components=20,
    tournament_size=100,
    function_set=GPU_SAFE_PANEL_FUNCTIONS,
    objective="icir",          # ic, rank_ic, icir, rank_icir, long_short_sharpe
    feature_names=feature_names,
    time_series_index="datetime",
    security_index="symbol",
    device="cuda:0",
    random_state=2025,
    verbose=1,
)

model.fit_panel(panel_df, target_col="target")

factor_df = model.transform_panel(output="dataframe")
expr_df = model.get_factor_expressions()
```

如果机器没有 CUDA，`device="cuda:0"` 会自动回退到 CPU tensor 执行，并给出 warning。生产训练建议使用 CUDA。

## 5. GPU 支持的函数集

当前支持以下 GPU 安全函数：

```text
add, sub, mul, div, sqrt, log, abs, neg, inv, max, min, sin, cos, tan, sig

ts_shift, ts_delta, ts_mom, ts_min, ts_max, ts_argmax, ts_argmin,
ts_rank, ts_sum, ts_std, ts_corr, ts_mean, ts_zscore, ts_freq,
ts_cdlbodym, ts_bar_bs, ts_adx, ts_aroon, ts_bopr, ts_cmo, ts_ema,
ts_macd, ts_rsi, ts_stochf, ts_xs_ratio, ts_one_ols_k,
ts_one_ols_resid, ts_skew, ts_kurt, ts_atr, ts_hedge, ts_bband

cs_rank, cs_zscore, cs_demean, cs_scale, cs_winsorize
```

推荐直接使用：

```python
from QuantGplearn.torch_functions import GPU_SAFE_PANEL_FUNCTIONS
```

## 6. 训练目标 objective

| objective | 含义 | 适用场景 |
|---|---|---|
| `ic` / `pearson` | 每期截面 Pearson IC 的均值 | 快速挖掘线性相关因子 |
| `rank_ic` / `spearman` | 每期截面 RankIC 的均值 | 更关注排序稳定性 |
| `icir` | IC 均值 / IC 标准差 | 偏向稳定 IC 因子 |
| `rank_icir` | RankIC 均值 / RankIC 标准差 | 偏向稳定排序因子 |
| `long_short_sharpe` / `sharpe` | GPU top/bottom 多空组合 Sharpe 代理 | 偏策略收益目标 |

训练期的 `long_short_sharpe` 是快速代理目标，不替代独立、无前视的完整回测验证。

## 7. 与 alphagen 的对应关系

| alphagen 设计 | 本项目实现 |
|---|---|
| `StockData` | `TensorPanelData` |
| `Expression.evaluate(data)` | `_Program.execute_tensor(data)` |
| `TensorAlphaCalculator` | `TensorFactorCalculator` |
| `batch_pearsonr` / `batch_spearmanr` | `tensor_fitness.batch_pearsonr` / `batch_spearmanr` |
| `LinearAlphaPool` | `GPAlphaPool` |
| RL 生成公式 | 暂不迁移，保留 QuantGplearn GP 进化 |

## 8. 关键参数建议

中等规模 GPU 训练建议：

```python
GpuSymbolicTransformer(
    population_size=512,
    generations=20,
    hall_of_fame=100,
    n_components=20,
    tournament_size=64,
    init_depth=(2, 5),
    p_crossover=0.5,
    p_subtree_mutation=0.2,
    p_hoist_mutation=0.1,
    p_point_mutation=0.1,
    max_length=24,
    tolerable_corr=0.7,
    objective="icir",
    device="cuda:0",
)
```

大规模训练建议：

- 先用 `objective="ic"` 或 `objective="rank_ic"` 快速筛选；
- 再用 `objective="icir"` 或 `long_short_sharpe` 做精筛；
- 复杂 rolling window 会增加显存占用；
- 高频长历史数据建议控制 `population_size`、`generations` 和最大表达式长度。

## 9. 已知限制

1. 第一版 GPU 路径不支持分类特征和 category-return 函数。
2. `cs_rank` / `ts_rank` 的 tie 处理采用 Torch ordinal rank，和 Pandas `average rank` 可能存在微小差异。
3. `torch.unfold` 实现 rolling 窗口，对超长窗口和超大面板会产生较大中间张量。
4. GPU 模式下没有使用 joblib 多进程执行 fitness，避免 CUDA 多进程上下文和显存复制问题。
5. 复杂 rolling 技术指标在长历史和大面板上可能产生较大的中间张量。

## 10. 测试

```bash
PYTHONPATH=. pytest -q tests/test_gpu_transformer.py
PYTHONPATH=. pytest -q tests/test_panel_program.py tests/test_functions.py
```

在没有 CUDA 的环境中，GPU transformer 测试会使用 CPU tensor，因此可作为功能 smoke test。
