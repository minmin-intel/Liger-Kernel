import torch
import torch.nn as nn

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))
from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward

device = "xpu"

def test_correctness_llamamlp(bsz, seq_len, hidden_size, intermediate_size, dtype):
    _input = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    x1 = _input.clone().requires_grad_(True)

    # initialize weights
    G = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    a = G.T @ x1
    b = U.T @ x1

    output = swiglu_forward(a, b)
    y = D.T @ output

    dy = torch.randn_like(y)
    swiglu_backward(a, b, dy)