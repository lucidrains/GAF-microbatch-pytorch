import pytest
from copy import deepcopy

import torch
from torch import nn
torch.set_default_dtype(torch.float64)

from GAF_microbatch_pytorch import GAFWrapper, set_filter_gradients_

def test_gaf():

    net = nn.Sequential(
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.Linear(256, 128)
    )

    gaf_net = GAFWrapper(
        deepcopy(net),
        filter_distance_thres = 2.
    )

    x = torch.randn(8, 1024, 512)
    y = x.clone()

    x.requires_grad_()
    y.requires_grad_()

    out1 = net(x)
    out2 = gaf_net(y)

    out1.sum().backward()
    out2.sum().backward()

    grad = net[0].weight.grad
    grad_filtered = gaf_net.net[0].weight.grad

    assert torch.allclose(grad, grad_filtered, atol = 1e-6)

def test_gaf():

    net = nn.Sequential(
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.Linear(256, 128)
    )

    gaf_net = GAFWrapper(
        deepcopy(net),
        filter_distance_thres = 0.
    )

    x = torch.randn(8, 1024, 512)
    y = x.clone()

    x.requires_grad_()
    y.requires_grad_()

    out1 = net(x)
    out2 = gaf_net(y)

    out1.sum().backward()
    out2.sum().backward()

    grad = net[0].weight.grad
    grad_filtered = gaf_net.net[0].weight.grad

    assert not torch.allclose(grad, grad_filtered, atol = 1e-6)
