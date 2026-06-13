# 本次重构摘要

## 重构原则

本次没有把 QuantGplearn 改写成 alphagen 的 RL 框架，而是采用更稳健的分层迁移方案：

```text
保留 QuantGplearn GP 生成与遗传操作
新增 alphagen 风格 Tensor 数据层、Calculator、AlphaPool
新增 Torch/GPU 执行后端
新增 GPU 训练入口
```

## 代码变更

### 修改

- `QuantGplearn/functions.py`
  - `_Function` 新增 `torch_function` 字段；
  - 新增 `call_torch()`；
  - 不影响原有 `__call__()` NumPy/Pandas 逻辑。

- `QuantGplearn/_program.py`
  - 新增 `execute_tensor(data)`；
  - 保留原有 `execute(X)`；
  - `generate_my_output()` 兼容 `np.integer` 和 `np.floating` 常数。

- `QuantGplearn/__init__.py`
  - 导出 `GpuSymbolicTransformer`。

### 新增

- `tensor_data.py`
- `torch_functions.py`
- `tensor_fitness.py`
- `evaluator.py`
- `alpha_pool.py`
- `gpu_transformer.py`
- `tests/test_gpu_transformer.py`
- `docs/GPU_FACTOR_MINING.md`

## 兼容性

原有 CPU 路径保持可用：

```python
from QuantGplearn.genetic import SymbolicTransformer
```

新增 GPU 路径：

```python
from QuantGplearn.gpu_transformer import GpuSymbolicTransformer
```

二者可以并存。旧代码不需要改动，除非用户主动迁移到 `GpuSymbolicTransformer`。

## 推荐后续扩展

1. 为 `cs_rank` 和 `ts_rank` 实现严格 average-tie rank。
2. 增加 chunked rolling，降低超大面板显存占用。
3. 增加多 GPU evaluator worker。
4. 将 `GPAlphaPool` 扩展为线性组合权重优化器。
