class CrossEntropyWithZLoss(torch.nn.Module):
    def __init__(
        self,
        weight=None,
        lse_square_scale=0.0,
        reduction="mean",
        ignore_index=-100,
        label_smoothing=0.0,
        return_z_loss=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.weight = weight
        self.lse_square_scale = lse_square_scale
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.return_z_loss = return_z_loss
        self.label_smoothing = label_smoothing
        self.dtype = dtype

    def forward(self, logits, targets):
        # Loss calculations are all in float32
        logits = logits.to(torch.float32)

        target_mask = targets != self.ignore_index

        # Standard cross entropy loss
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

        # Compute log-sum-exp term
        lse = torch.logsumexp(logits, dim=-1)

        # Z-loss term
        z_loss = torch.where(targets != self.ignore_index, self.lse_square_scale * (lse**2), 0.0)

        if self.reduction == "mean":
            z_loss = z_loss.sum() / target_mask.sum()
        elif self.reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss
        ce_loss = ce_loss.to(self.dtype)
        z_loss = z_loss.to(self.dtype)

        # Final loss: cross-entropy loss + Z-loss
        total_loss = ce_loss + z_loss
        if self.return_z_loss:
            return total_loss, z_loss
        else:
            return total_loss