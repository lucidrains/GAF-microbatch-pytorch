from __future__ import annotations

from functools import partial
from typing import Literal, Callable
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import Module, Linear
from torch.autograd import Function
import torch.nn.functional as F

from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.func import functional_call, vjp, vmap

from einops import einsum, rearrange, repeat, reduce

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# distance used for gradient agreement
# they found cosine distance to work the best, at a threshold of ~0.96

def l2norm(t):
    return F.normalize(t, p = 2, dim  = -1)

def cosine_sim_distance(grads):
    grads = rearrange(grads, 'b ... -> b (...)')
    normed = l2norm(grads)
    dist = einsum(normed, normed, 'i d, j d -> i j')
    return 1. - dist

def filter_gradients_by_agreement(
    grads,
    threshold,
    strategy: Literal[
        'accept_max_neighbors',
        'accept_min_neighbors'
    ] = 'accept_max_neighbors',
    accept_batch_frac = 0.2
):
    """ main gradient filtering function """

    batch = grads.shape[0]

    dist = cosine_sim_distance(grads) # (batch, batch) cosine sim gradient distance

    accept_mask = dist < threshold

    num_neighbors_within_dist = accept_mask.sum(dim = -1)

    if (num_neighbors_within_dist == 1).all():
        return torch.zeros_like(grads)

    # take the most naive approach

    if strategy == 'accept_max_neighbors':
        # accept the gradient and its neighbors that is the majority

        center_ind = num_neighbors_within_dist.argmax(dim = -1)

        accept_mask = accept_mask[center_ind]

    elif strategy == 'accept_min_neighbors':
        # reject any gradients that does not have at least `batch * accept_batch_frac` similar gradients within the same batch

        accept_mask = num_neighbors_within_dist >= max(batch * accept_batch_frac, 2)
    else:
        raise ValueError(f'unknown strategy {strategy}')

    if accept_mask.sum().item() <= 1:
        return torch.zeros_like(grads)

    if accept_mask.all():
        return grads

    renorm_scale = batch / accept_mask.sum().item()

    # filter out the gradients

    grads[~accept_mask] = 0.

    # renormalize based on how many accepted

    grads *= renorm_scale

    return grads

# custom linear

class GAF(Function):

    @classmethod
    def forward(self, ctx, tree_spec, *tree_nodes):

        package = tree_unflatten(tree_nodes, tree_spec)

        net = package['net']
        params, buffers = package['params_buffers']
        filter_gradients_fn = package['filter_gradients_fn']
        exclude_from_filtering = package['exclude_from_filtering']
        inp_tensor, args, kwargs = package['inputs']

        batch = inp_tensor.shape[0]

        def fn(params, buffers, inp_tensor):
            return functional_call(net, (params, buffers), (inp_tensor, *args), kwargs)

        fn = vmap(fn, in_dims = (0, None, 0))

        params = {name: repeat(t, '... -> b ...', b = batch) for name, t in params.items()}

        output, vjpfunc = vjp(fn, params, buffers, inp_tensor)

        ctx._saved_info_for_backwards = (
            vjpfunc,
            filter_gradients_fn,
            args,
            kwargs,
            exclude_from_filtering
        )

        return output

    @classmethod
    def backward(self, ctx, do):

        (
            vjp_func,
            filter_gradients_fn,
            args,
            kwargs,
            exclude_from_filtering
        ) = ctx._saved_info_for_backwards

        dparams, dbuffers, dinp = vjp_func(do)

        # filter gradients for each parameter tensor
        # unless it is in `exclude_from_filtering`

        filtered_dparams = dict()

        for name, dparam in dparams.items():
            if name in exclude_from_filtering:
                filtered_dparams[name] = dparam
                continue

            filtered_dparams[name] = filter_gradients_fn(dparam)

        # tree flatten back out

        package = dict(
            net = None,
            params_buffers = (filtered_dparams, dbuffers),
            inputs = (dinp, None, None),
            filter_gradients_fn = None,
            exclude_from_filtering = None
        )

        tree_nodes, _ = tree_flatten(package)

        return (None, *tree_nodes)

gaf_function = GAF.apply

# main function

class GAFWrapper(Module):
    """
    a wrapper for a neural network that automatically starts filtering all the gradients by their intra-batch agreement - not across machines as in the paper
    """
    def __init__(
        self,
        net: Module,
        filter_distance_thres = 0.97,
        filter_gradients = True,
        filter_gradients_fn: Callable | None = None,
        exclude_from_filtering: Sequence[str] = ()
    ):
        super().__init__()

        self.net = net
        assert not any([m for m in net.modules() if isinstance(m, GAFWrapper)]), 'GAF wrapper cannot contain another network that is already wrapped'

        self.exclude_from_filtering = set(exclude_from_filtering)

        # gradient agreement filtering related

        self.filter_gradients = filter_gradients
        self.filter_distance_thres = filter_distance_thres

        if not exists(filter_gradients_fn):
            filter_gradients_fn = partial(filter_gradients_by_agreement, threshold = filter_distance_thres)

        self.filter_gradients_fn = filter_gradients_fn

    def forward(
        self,
        inp_tensor,
        *args,
        **kwargs
    ):
        only_one_dim_or_no_batch = inp_tensor.ndim == 1 or inp_tensor.shape[0] == 1

        if not self.filter_gradients or only_one_dim_or_no_batch:
            return self.net(inp_tensor, *args, **kwargs)

        params = dict(self.net.named_parameters())
        buffers = dict(self.net.named_buffers())

        package = dict(
            net = self.net,
            params_buffers = (params, buffers),
            inputs = (inp_tensor, args, kwargs),
            filter_gradients_fn = self.filter_gradients_fn,
            exclude_from_filtering = self.exclude_from_filtering
        )

        tree_nodes, tree_spec = tree_flatten(package)

        out = gaf_function(tree_spec, *tree_nodes)
        return out

# helper functions for disabling GAF wrappers within a network
# for handy ablation, in the case subnetworks within a neural network were wrapped

def set_filter_gradients_(
    m: Module,
    filter_gradients: bool,
    filter_distance_thres = None
):
    for module in m.modules():
        if not isinstance(module, GAFWrapper):
            continue

        module.filter_gradients = filter_gradients

        if exists(filter_distance_thres):
            module.filter_distance_thres = filter_distance_thres
