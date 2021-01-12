#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn.functional as F

# Negative Cosine Similarity
def D(p,z):
    """
    z = z.detach()
    p = F.normalize(p,p=2,dim=1)
    z = F.normalize(z,p=2,dim=1)
    return -(p*z).sum(dim=1).mean()
    """
    # "faster version" by Patrick Hua (actually faster...?)
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

