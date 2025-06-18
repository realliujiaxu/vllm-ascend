import torch
import torch_npu

from vllm_ascend.ops.npu_moe_token_unpermute import npu_moe_token_unpermute

dtype = torch.bfloat16
permuted_tokens = torch.tensor([[1., 1., 1.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [3., 3., 3.],
                                    [2., 2., 2.],
                                    [1., 1., 1.],
                                    [2., 2., 2.],
                                    [3., 3., 3.]]).npu().to(dtype).requires_grad_(True)
sorted_indices = torch.tensor([0, 6, 7, 5, 3, 1, 2, 4], dtype=torch.int32).npu()
indices = torch.tensor([[0, 4], [4, 3], [4, 2], [1, 1]]).npu()
probs = torch.ones_like(indices) / 2
probs = probs.npu().to(dtype).requires_grad_(True)

# 正向接口案例
unpermuted_tokens = npu_moe_token_unpermute(
    permuted_tokens, sorted_indices, probs=probs)
