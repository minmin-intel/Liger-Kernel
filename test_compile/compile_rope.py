import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# import os
# import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.join(project_root, 'src'))
# from liger_kernel.ops.rope import rope_forward, rope_backward

def transformers_version_dispatch(
    required_version: str,
    before_fn,
    after_fn,
    before_args: tuple = (),
    after_args: tuple = (),
    before_kwargs: dict = None,
    after_kwargs: dict = None,
):
    """
    Dispatches to different functions based on package version comparison.

    Args:
        required_version: Version to compare against (e.g. "4.48.0")
        before_fn: Function to call if package_version < required_version
        after_fn: Function to call if package_version >= required_version
        before_args: Positional arguments for before_fn
        after_args: Positional arguments for after_fn
        before_kwargs: Keyword arguments for before_fn
        after_kwargs: Keyword arguments for after_fn

    Returns:
        Result from either before_fn or after_fn

    Example:
        >>> rotary_emb = transformers_version_dispatch(
        ...     "4.48.0",
        ...     LlamaRotaryEmbedding,
        ...     LlamaRotaryEmbedding,
        ...     before_args=(head_dim,),
        ...     after_args=(LlamaConfig(head_dim=head_dim),),
        ...     before_kwargs={'device': device},
        ...     after_kwargs={'device': device}
        ... )
    """
    from packaging import version
    from transformers import __version__ as transformers_version

    before_kwargs = before_kwargs or {}
    after_kwargs = after_kwargs or {}

    if version.parse(transformers_version) < version.parse(required_version):
        return before_fn(*before_args, **before_kwargs)
    else:
        return after_fn(*after_args, **after_kwargs)

device = "xpu"
def test_compile(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    expand_position_ids,
):
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    _tensor_q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    _tensor_k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)


    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k1, pos_ids)

    # validate forward pass
    rope_forward(q1, k1, cos, sin)


    # validate backward pass
    dq, dk = (
        torch.randn_like(q1, device=device),
        torch.randn_like(k1, device=device).to(dtype),
    )

    rope_backward(dq, dk, cos, sin)

if __name__ == "__main__":
    test_compile(
        bsz=2,
        seq_len=128,
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=64,
        dtype=torch.float32,
        expand_position_ids=True,
    )

    test_compile(
        bsz=2,
        seq_len=128,
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=64,
        dtype=torch.float32,
        expand_position_ids=False,
    )