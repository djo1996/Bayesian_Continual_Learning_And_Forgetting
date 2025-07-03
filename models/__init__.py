
#!/usr/bin/env python3                     
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT AND BSD-3-Clause 
#
# Code for “Bayesian continual learning and forgetting in neural networks”
# (arXiv:2504.13569)
# Portions adapted from the PyTorch project (BSD-3-Clause) 
#
# Author: Djohan Bonnet  <djohan.bonnet@gmail.com>
# Date: 2025-04-18
"""
Init file
"""


from .LayersBBB import *
from .LayersBBB_GP import *
from .LayersMetaBayes import *

from .Models_BBB import *
from .Models_BBB_GP import *
from .Models_MetaBayes import *
from .Models_DET import *