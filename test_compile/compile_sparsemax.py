import torch
dims = [1, -1]
x = torch.randn((2, 128, 512), dtype=torch.float32, device="xpu")
for dim in dims:
    y, out_flat = _sparsemax_forward(x, dim=dim)
    _sparsemax_backward(grad_out=y, out_flat=out_flat, dim=dim)