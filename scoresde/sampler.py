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
from scoresde.models.utils import get_noise_fn

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
from tqdm.auto import tqdm
import io
from scoresde.utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import scoresde.models
from scoresde.models import utils as mutils
from scoresde.models import ddpm as ddpm_model
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
from util.img_utils import clear_color
#import scoresde.cifar10_ddpmpp_deep_continuous as configs 
import scoresde.cifar10_ddpmpp_continuous as configs 
from guided_diffusion.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

def create_sampler():
    config = configs.get_config()
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
    sigmas = mutils.get_sigmas(config)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    img_size = config.data.image_size
    channels = config.data.num_channels
    batch_size = 1
    #shape = (batch_size, channels, img_size, img_size)
    predictor = EulerMaruyamaPredictor
    probability_flow = False
    #scale = kwargs.get('scale', 1.0)
    sampler = Sampler(sde,
                      predictor, 
                      probability_flow, 
                      config.training.continuous, 
                      sampling_eps,
                      inverse_scaler)
    return sampler

class Sampler():
    def __init__(self, 
                sde, 
                predictor, 
                probability_flow, 
                continuous, 
                eps,
                inverse_scaler,
                denoise=True):
        self.sde = sde
        self.predictor = predictor
        self.predictor_update_fn = functools.partial(scoresde.sampling.shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
        self.eps = eps
        self.denoise = denoise
        self.inverse_scaler = inverse_scaler
        #self.operator = operator
        #self.scale = scale
    
    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root):
        img = x_start
        device = x_start.device
        noise_schedule = NoiseScheduleVP(schedule='linear', continuous_beta_0=0.1, continuous_beta_1=20.)
        noise_pred_fn = get_noise_fn(self.sde, model, train=False, continuous=True)
        model_fn = model_wrapper(
                noise_pred_fn,
                noise_schedule,
                model_type="noise",
                guidance_type="dps",
                condition=measurement,
                classifier_fn=measurement_cond_fn
            )
        solver_type = "dpmsolver"
        #skip_type = "time_uniform"
        #order = 3
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type=solver_type)

        
        img = dpm_solver.sample(
                img,
                steps=50,
                order=3,
                skip_type="time_uniform",
                method="singlestep",
        )
        #skip_type = "time_uniform"
        #order = 3
        #dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type=solver_type)

        #timesteps_outer, orders =  dpm_solver.get_orders_and_timesteps_for_singlestep_solver(
        #    steps=50, 
        #    order=order, 
        #    skip_type=skip_type,
        #    t_T=dpm_solver.noise_schedule.T, 
        #    t_0=1e-3, 
        #    device=device)

        #for step, order in enumerate(orders):
        #    s, t = timesteps_outer[step], timesteps_outer[step + 1]
        #    timesteps_inner = dpm_solver.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
        #    lambda_inner = dpm_solver.noise_schedule.marginal_lambda(timesteps_inner)
        #    h = lambda_inner[-1] - lambda_inner[0]
        #    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
        #    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
        #    x = dpm_solver.singlestep_dpm_solver_update(img, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
        #    vec_s = torch.ones(img.shape[0], device=device) * s
        #    noisy_measurement, _ = self.sde.marginal_prob(measurement, t=vec_s)
        #    mean, std = self.sde.marginal_coef(vec_s)
        #    for _ in range(1):
        #        with torch.enable_grad():
        #            img = img.requires_grad_()
        #            s = model(img, vec_s)
        #            x_0_hat = (img + std*std*s)/mean
                    #x_0_hat = dpm_solver.sample(
                    #            img,
                    #            steps=20,
                    #            order=3,
                    #            skip_type="time_uniform",
                    #            method="singlestep",
                    #            t_start=s
                    #) 
        #            img, distance = measurement_cond_fn(x_t=x,
        #                              measurement=measurement,
        #                              noisy_measurement=noisy_measurement,
        #                              x_prev=img,
        #                              x_0_hat=x_0_hat)
        #            img = img.detach_()
            #img = x
        for _ in range(100):
            with torch.enable_grad():
                img = img.requires_grad_()
                vec_s = torch.ones(img.shape[0], device=device) * 1e-3
                s = model(img, vec_s)
                mean, std = self.sde.marginal_coef(vec_s)
                x_0_hat = (img + std*std*s)/mean
                    #x_0_hat = dpm_solver.sample(
                    #            img,
                    #            steps=20,
                    #            order=3,
                    #            skip_type="time_uniform",
                    #            method="singlestep",
                    #            t_start=s
                    #) 
                x = img.detach()
                img, distance = measurement_cond_fn(x_t=x,
                                      measurement=measurement,
                                      noisy_measurement=measurement,
                                      x_prev=img,
                                      x_0_hat=x_0_hat)
                img = img.detach_()







        #pbar = tqdm(list(range(self.sde.N)))
        #timesteps = torch.linspace(self.sde.T, self.eps, self.sde.N, device=device)
        #for idx in range(self.sde.N):
        #    t = timesteps[idx]
        #    vec_t = torch.ones(img.shape[0], device=t.device) * t
        #    img = img.requires_grad_()
            #x, _, s  = self.predictor_update_fn(img, vec_t, model=model)
            # Give condition.
            #noisy_measurement, _ = self.sde.marginal_prob(measurement, t=vec_t)
            #mean, std = self.sde.marginal_coef(vec_t)
            #x_0_hat = (img + std * std * s) / mean
            # TODO: how can we handle argument for different condition method?
            #img, distance = measurement_cond_fn(x_t=x,
            #                          measurement=measurement,
            #                          noisy_measurement=noisy_measurement,
            #                          x_prev=img,
            #                          x_0_hat=x_0_hat)

            #img = img.detach_()
           
            #pbar.set_postfix({'distance': distance.item()}, refresh=False)
            #if record:
            #    if idx % 10 == 0:
            #        file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
            #        plt.imsave(file_path, clear_color(img))
        #self.inverse_scaler(x_mean if self.denoise else x)

        return img