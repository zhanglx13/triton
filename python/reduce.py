import torch
import triton
import triton.language as tl
from numpy.random import RandomState

@triton.jit
def seg_fault_kernel(X, Z, N, BLOCK: tl.constexpr):
    np = tl.num_programs(1)
    x = tl.load(X + tl.arange(0, BLOCK))
    z = tl.sum(x, axis=0) + np
    tl.store(Z, z)

device = 'cuda'
shape = 128
dtype_str = torch.float16


rs = RandomState(17)
x_tri = torch.randn(shape, dtype=dtype_str, device=device)
z_tri = torch.randn(1, dtype=dtype_str, device=device)
seg_fault_kernel[(1, )](x_tri, z_tri, shape, BLOCK=shape)
print(f"torch sum: {torch.sum(x_tri)}")
print(f"triton sum: {z_tri[0]-1}")
