import torch
device = "xpu"

def test_compile(
    B,
    T,
    V,
    scalar,
    dtype,
):
    _input = torch.randn(B * T, V, device=device, dtype=dtype) * scalar

    x1 = _input.clone().requires_grad_(True)

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    cross_entropy_forward(
        _input,
        target,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss=False,
    )

    grad_output = torch.randn_like(x1, device=device)
    cross_entropy_backward(_input, grad_output)


if __name__ == "__main__":
    test_compile(
        B=2,
        T=128,
        V=512,
        scalar=0.1,
        dtype=torch.float32,
    )