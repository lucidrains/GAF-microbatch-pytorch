## Gradient Agreement Filtering (microbatch) - Pytorch

Implementation of [Gradient Agreement Filtering](https://arxiv.org/abs/2412.18052), from Chaubard et al. of Stanford, but for single machine microbatches, in Pytorch.

Whether it is just a means to filter out outlier label noise, or actually has some ties to better generalization, thought it was worth exploring either way.

The official repository that does filtering done for macrobatches is [here](https://github.com/Fchaubard/gradient_agreement_filtering)

## Install

```bash
$ pip install GAF-microbatch-pytorch
```

## Usage

```python
import torch

# mock network

from torch import nn

net = nn.Sequential(
    nn.Linear(512, 256),
    nn.SiLU(),
    nn.Linear(256, 128)
)

# import the gradient agreement filtering (GAF) wrapper

from GAF_microbatch_pytorch import GAFWrapper

# just wrap your neural net

gaf_net = GAFWrapper(
    net,
    filter_distance_thres = 0.97
)

# your batch of data

x = torch.randn(16, 1024, 512)

# forward and backwards as usual

out = gaf_net(x)

out.sum().backward()

# gradients should be filtered by set threshold comparing per sample gradients within batch, as in paper

```

## Citations

```bibtex
@inproceedings{Chaubard2024BeyondGA,
    title   = {Beyond Gradient Averaging in Parallel Optimization: Improved Robustness through Gradient Agreement Filtering},
    author  = {Francois Chaubard and Duncan Eddy and Mykel J. Kochenderfer},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:274992650}
}
```
