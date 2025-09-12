# import os
# import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.join(project_root, 'src'))
# from liger_kernel.ops.cross_entropy import cross_entropy_forward, cross_entropy_backward

import torch
device = "xpu"

def test_compile(
    B,
    T,
    V,
    scalar,
    dtype,
    reduction="mean",
):
    _input = torch.randn(B * T, V, device=device, dtype=dtype) * scalar

    x1 = _input.clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)
    weight = torch.randn(V, device=device, dtype=dtype)

    result = cross_entropy_forward(
        _input,
        target,
        weight,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction=reduction,
        softcap=None,
        return_z_loss=False,
    )

    print("Cross Entropy result:", result)

    grad_output = torch.randn_like(x1, device=device)
    x1 = cross_entropy_backward(x1, grad_output)
    print("Cross Entropy grad:", x1)


if __name__ == "__main__":
    test_compile(
        B=2,
        T=128,
        V=512,
        scalar=0.1,
        dtype=torch.float32,
        reduction="mean",
    )

    test_compile(
        B=2,
        T=128,
        V=512,
        scalar=0.1,
        dtype=torch.float32,
        reduction="sum",
    )