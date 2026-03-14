# Sequence Parallelism BasePattern 迁移设计文档

## 1. 背景与目标

### 1.1 现状

- `SequenceParallelismPass` 和 `SequenceParallelismMoePass` 的 pattern 使用 `pm.register_replacement` 直接注册到 Inductor，**未**调用 `torchair.register_replacement`
- 启用 npugraph_ex 时，graph 直接交给 TorchAir，不经过 GraphFusionPassManager，导致 SP 的 pattern 从未生效
- 与 `BasePattern` 不同：继承 BasePattern 的 pass（如 MatmulAllReduceAddRMSNormPass）会同时注册到 Inductor 和 TorchAir，在两种编译路径下均可生效

### 1.2 目标

将 Sequence Parallelism 相关 pattern 迁移到继承 `BasePattern`，实现：

- **enable_npugraph_ex=True** 时：pattern 通过 `torchair.register_replacement` 在 npugraph_ex 内部生效
- **enable_npugraph_ex=False** 时：pattern 通过 `pm.register_replacement` 在 GraphFusionPassManager 路径下生效
- 行为与现有实现保持一致，不引入功能回归

---

## 2. 技术方案

### 2.1 BasePattern 双注册机制

`BasePattern.register(pm_pass)` 会同时执行：

```python
pm.register_replacement(pattern_fn, replacement_fn, example_inputs, pm.fwd_only, pm_pass)
torchair.register_replacement(search_fn=pattern_fn, replace_fn=replacement_fn, example_inputs=example_inputs, extra_check=...)
```

因此，只要 pattern 继承 BasePattern 并实现 `get_inputs()`、`get_pattern()`、`get_replacement()`，即可在两种路径下生效。

### 2.2 迁移范围

| Pass | Pattern 数量 | scalar_workaround | 迁移可行性 |
|------|-------------|-------------------|------------|
| SequenceParallelismPass | 3 | 无 | ✅ 已完成 |
| SequenceParallelismMoePass | 4 | 3 个 pattern 使用 | ✅ 已完成 |

TorchAir 的 `register_replacement` 支持 `scalar_workaround` 参数（参见 `torchair/npugraph_ex/npugraph_ex/patterns/pattern_pass_manager.py`），BasePattern 已扩展 `get_scalar_workaround()` 并传递给双端注册。

### 2.3 SequenceParallelismPass 涉及的 Pattern

| Pattern 类 | 功能 |
|------------|------|
| MiddleAllReduceRMSNormPattern | all_reduce + AddRMSNormBias → reduce_scatter + AddRMSNormBias + all_gather（中间层） |
| LastAllReduceRMSNormPattern | 同上，最后一层（无 residual 回传） |
| Qwen3VLMiddleAllReduceRMSNormPattern | all_reduce + add + AddRMSNormBias → reduce_scatter + chunk + add + AddRMSNormBias + all_gather（Qwen3-VL） |

---

## 3. 实现设计

### 3.1 继承关系与辅助类

```
BasePattern (abc)
    └── SequenceParallelBasePattern (新增)
            ├── 继承 BasePattern + _SequenceParallelPatternHelper
            └── 提供 get_extra_stream_scope_check()，包含 min_tokens 检查

SequenceParallelismPass:
  MiddleAllReduceRMSNormPattern(SequenceParallelBasePattern)
  LastAllReduceRMSNormPattern(SequenceParallelBasePattern)
  Qwen3VLMiddleAllReduceRMSNormPattern(SequenceParallelBasePattern)

SequenceParallelismMoePass:
  MiddleLayerAllgatherAddRMSNormPattern(SequenceParallelBasePattern)  # get_scalar_workaround
  LastLayerAllgatherRMSNormPattern(SequenceParallelBasePattern)       # get_scalar_workaround
  Qwen3VLMiddleLayerAllgatherAddRMSNormPattern(SequenceParallelBasePattern)  # get_scalar_workaround
  AllGatherChunkNoOpPattern(SequenceParallelBasePattern)
```

`_SequenceParallelPatternHelper` 保留为独立辅助类，提供：

- `_all_reduce`, `_reduce_scatter`, `_all_gather`
- `empty(shape)`
- `tp_group`, `tp_size`, `tp_rank`

`SequenceParallelBasePattern` 通过组合或继承使用这些能力，并实现 BasePattern 的抽象接口。

### 3.2 BasePattern 接口适配

需实现：

- `get_inputs() -> list[torch.Tensor]`：构造与 pattern 参数一致的 example tensors
- `get_pattern() -> Callable`：返回 pattern 函数
- `get_replacement() -> Callable`：返回 replacement 函数
- `get_extra_stream_scope_check()`：返回 `extra_check`，可组合 stream 检查与 token 数量检查
- `get_scalar_workaround() -> dict | None`（可选）：返回 `scalar_workaround` 字典，用于含标量/符号输入的 pattern

### 3.3 min_tokens 检查（extra_check）

#### 3.3.1 简化方案（参考 allreduce_rmsnorm_fusion_pass）

`allreduce_rmsnorm_fusion_pass` 在 `extra_check` 中直接使用 `get_pass_context().compile_range`，且该 pass 已通过 BasePattern 注册到 TorchAir 并正常工作，说明 TorchAir 执行 `extra_check` 时 pass context 已被 vLLM 设置好。SP 可采用相同模式，无需从 shape 推断 token 数。

在 `sequence_parallelism.py` 中新增工厂函数：

```python
def get_sp_compile_range_and_extra_stream_check(min_tokens: int):
    """Same pattern as allreduce_rmsnorm_fusion_pass.get_compile_range_and_extra_stream_check."""
    def check_func(match: Match) -> bool:
        compile_range = get_pass_context().compile_range
        return extra_stream_scope_check(match) and compile_range.start >= min_tokens

    return check_func
```

在 `SequenceParallelBasePattern` 中：

```python
def get_extra_stream_scope_check(self):
    return get_sp_compile_range_and_extra_stream_check(get_sp_min_token_num(self.vllm_config))
```

需添加 import：`from vllm.compilation.passes.inductor_pass import get_pass_context` 和 `Match`。

### 3.4 与 compile_ranges_split_points 的关系

- 迁移后，pattern 会在 npugraph_ex 内部应用，不再依赖「在 npugraph_ex 前跑 GraphFusionPassManager」
- `ascend_config.update_compile_ranges_split_points` 中，当 `enable_npugraph_ex=True` 且 `enable_sp=True` 时，仍需将 `sp_min_token_num` 加入 `compile_ranges_split_points`，以便按 token 范围拆分不同编译产物
- 具体逻辑与 `enable_npugraph_ex=False` 分支保持一致（参考现有 `update_compile_ranges_split_points`）

### 3.5 scalar_workaround 支持（SequenceParallelismMoePass）

TorchAir 的 `register_replacement` 支持 `scalar_workaround` 参数，BasePattern 已扩展：

- `get_scalar_workaround() -> dict | None`：默认返回 `None`，需要时覆盖返回如 `{"num_tokens": 8}`
- `register()` 中将 `scalar_workaround` 传递给 `pm.register_replacement` 和 `torchair.register_replacement`

MoE 的 3 个 pattern（`MiddleLayerAllgatherAddRMSNormPattern`、`LastLayerAllgatherRMSNormPattern`、`Qwen3VLMiddleLayerAllgatherAddRMSNormPattern`）已迁移为继承 `SequenceParallelBasePattern`，并实现 `get_scalar_workaround()` 返回 `{"num_tokens": 8}`。`AllGatherChunkNoOpPattern` 无需 scalar_workaround。

---

## 4. 文件与修改清单

| 文件 | 修改内容 |
|------|----------|
| `vllm_ascend/compilation/passes/base_pattern.py` | 新增 `get_scalar_workaround()` 方法，`register()` 中将 `scalar_workaround` 传递给 pm 和 torchair |
| `vllm_ascend/compilation/passes/sequence_parallelism.py` | 1) 新增 `get_sp_compile_range_and_extra_stream_check(min_tokens)`；2) 新增 `SequenceParallelBasePattern`；3) 将 3 个 Pattern 迁移为继承 `SequenceParallelBasePattern` |
| `vllm_ascend/compilation/passes/sequence_parallelism_moe.py` | 将 4 个 Pattern 迁移为继承 `SequenceParallelBasePattern`，3 个实现 `get_scalar_workaround()` |
| `vllm_ascend/ascend_config.py` | 在 `update_compile_ranges_split_points` 的 `enable_npugraph_ex` 分支中，当 `enable_sp=True` 时添加 `sp_min_token_num` |
| `tests/e2e/multicard/2-cards/test_sp_pass.py` | 增加 `enable_npugraph_ex=True` + `enable_sp=True` 的用例，验证输出与 `enable_npugraph_ex=False` 一致 |
| `docs/source/user_guide/feature_guide/sequence_parallelism.md` | 说明 SP 可与 npugraph_ex 同时启用 |

### 4.1 不修改的部分

- `vllm_ascend/compilation/compiler_interface.py`：不再需要在 npugraph_ex 前手动执行 pass manager
- `vllm_ascend/compilation/graph_fusion_pass_manager.py`：SP 相关 pass 注册逻辑保持不变

---

## 5. 测试策略

1. **现有用例**：`test_sp_pass.py` 中 `enable_npugraph_ex=False` 的用例保持通过
2. **新增用例**：`enable_npugraph_ex=True` + `enable_sp=True`，与无 SP 或 SP+非 npugraph_ex 的输出对比，确保一致
3. **回归**：覆盖 Dense / MoE、VL / 非 VL 等配置，确保无功能退化

---

## 6. 风险与回退

| 风险 | 缓解 |
|------|------|
| TorchAir 路径下 `get_pass_context()` 未设置 | allreduce_rmsnorm_fusion_pass 已用相同方式在 TorchAir 下工作，可认为 context 已就绪 |
| 多继承/组合引入复杂度 | `SequenceParallelBasePattern` 明确封装 helper 与 BasePattern 的融合 |

若出现严重问题，可暂时在 `SequenceParallelBasePattern.register` 中只调用 `pm.register_replacement`，不调用 `torchair.register_replacement`，回退为仅 inductor 生效。

---

## 7. 验证脚本

当前使用的验证脚本 `run_qwen_vl_moe.sh`：

```bash
export PYTHONPATH=/home/l00659631/vllm:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_TORCH_PROFILER_DIR=/home/l00659631/profile
export HCCL_OP_EXPANSION_MODE=AIV
export TORCH_COMPILE_DEBUG=1

vllm serve /home/weights/Qwen3-VL-30B-A3B-Instruct  \
    --port 8001 \
    --max-model-len 262144 \
    --max-num-batched-tokens 32768 \
    --served-model-name auto \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --allowed-local-media-path /workspace \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8], "pass_config": {"enable_sp": true}}' \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/l00659631/profile"}'
```

- 模型：Qwen3-VL-30B-A3B-Instruct（VL MoE）
- 配置：TP=2, DP=2, enable_expert_parallel, enable_sp
- 可选：`--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'` 用于对比 npugraph_ex 开/关

---

## 8. 已知问题

### 9.1 AllGatherChunkNoOpPattern 未生效

**现象**：`AllGatherChunkNoOpPattern`（将 `all_gather` + `sequence_parallel_chunk_impl` 折叠为 identity）在实际运行中未生效。

**怀疑原因**：`NoOpEliminationPass` 未生效。

**分析**：

1. `NoOpEliminationPass` 仅在 `SequenceParallelismPass.__call__()` 中显式调用，用于消除 pattern 重写后产生的冗余 view/reshape 节点。
2. 当 `enable_npugraph_ex=True` 时，graph 直接交给 npugraph_ex，**不经过** `GraphFusionPassManager`，因此 `SequenceParallelismPass.__call__` 和 `SequenceParallelismMoePass.__call__` 均不会执行。
3. `NoOpEliminationPass` 作为显式图变换，不是通过 `register_replacement` 注册的 pattern，在 npugraph_ex 路径下**从未被执行**。
4. 若 `all_gather` 与 `sequence_parallel_chunk_impl` 之间存在多余的 view/reshape 等节点，会阻碍 `AllGatherChunkNoOpPattern` 匹配；而本应由 `NoOpEliminationPass` 清理的这些节点，在 npugraph_ex 路径下未被清理。

**待验证**：

- 通过 `TORCH_COMPILE_DEBUG=1` dump FX graph，确认 `all_gather` 与 `sequence_parallel_chunk_impl` 之间的图结构。
- 若存在需由 NoOpEliminationPass 清理的节点，需在 npugraph_ex 路径下增加对 NoOpEliminationPass 的调用（例如在 `npugraph_ex_compile` 中、将 graph 传给 npugraph_ex 之前执行）。

---

## 9. 参考

- [BasePattern 实现](vllm_ascend/compilation/passes/base_pattern.py)
- [TorchAir pattern_pass_manager（支持 scalar_workaround）](/home/l00659631/torchair/npugraph_ex/npugraph_ex/patterns/pattern_pass_manager.py)
- [allreduce_rmsnorm_fusion_pass 的 BasePattern 使用方式](vllm_ascend/compilation/passes/allreduce_rmsnorm_fusion_pass.py)
- [TorchAir register_replacement 文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00088.html)
- [Sequence Parallelism 用户文档](docs/source/user_guide/feature_guide/sequence_parallelism.md)
