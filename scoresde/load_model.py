
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from scoresde.losses import get_optimizer
from scoresde.models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
from scoresde.utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import scoresde.models
from scoresde.models import utils as mutils
from scoresde.models import ddpm as ddpm_model
from scoresde.models import ncsnpp 
from scoresde.models import layers
from scoresde.models import normalization
import scoresde.sampling
from scoresde.likelihood import get_likelihood_fn
from scoresde.sde_lib import VESDE, VPSDE, subVPSDE
from scoresde.sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import scoresde.datasets as datasets
#import scoresde.cifar10_ddpmpp_deep_continuous as configs 
import scoresde.cifar10_ddpmpp_continuous as configs 



def create_model(device):
    ckpt_filename = "models/checkpoint_8.pth"
    config = configs.get_config()
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
    model = mutils.create_model(config)
    optimizer = get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(model.parameters(),
                               decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
             model=model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, device)
    ema.copy_to(model.parameters())
    return model
