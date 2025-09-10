class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    :param label_smoothing: label_smoothing to apply on target
    :param lse_square_scale: scaler of lse ^ 2 to compute z loss
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ce_loss = CrossEntropyWithZLoss(
            weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            return_z_loss=return_z_loss,
        )
        self.softcap = softcap

    def forward(self, x, y):
        logits = self.lin(x).to(torch.float32)
        if self.softcap is not None and self.softcap != 0.0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return self.ce_loss(logits, y)
