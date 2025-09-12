import torch

# import os
# import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.join(project_root, 'src'))
# from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward

device = "xpu"

def test_compile(bsz, seq_len, hidden_size, intermediate_size, dtype):
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)

    # initialize weights
    G = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    a = x1 @ G
    b = x1 @ U

    print(f"a: {a.shape}, b: {b.shape}")

    a, b, c = swiglu_forward(a, b)
    print(f"output: {c.shape}")

    dc = torch.randn_like(c)
    a, b = swiglu_backward(a, b, dc)
    print(f"grad: {a.shape}, {b.shape}")

if __name__ == "__main__":
    test_compile(
        bsz=2,
        seq_len=128,
        hidden_size=512,
        intermediate_size=2048,
        dtype=torch.bfloat16,
    )