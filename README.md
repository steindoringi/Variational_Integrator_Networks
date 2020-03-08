# Variational Integrator Networks

## Overview

### Link to paper

![Variational Integrator Networks for Physically Structured Embeddings](https://arxiv.org/abs/1910.09349)

### Run experiment
`python run_exp.py root_dir model_name system_name observations num_train_traj num_train_steps seed`</br>
e.g. </br>
`python run_exp.py experiments VIN_VV pendulum pixels 1 60 1`

### Dependencies
- tensorflow 2.1
- tensorflow_probability
- gin-config
- see `requirements.txt`

## Example: Ideal Pendulum, Noisy Observations

__Setup__
- Train on 15s of observations (150 datapoints)
- Test on noisless initial state, forecast for 10s

<div style="display: flex; justify-content: row;">
  <img src="figures/pendulum_resnet_noisy.gif" width=45%>
  <img src="figures/pendulum_vinvv_noisy.gif" width=45%>
</div>

#### Recurrent Residual Network (Left) / Variational Integrator Network (Right) 

## Example: Ideal Pendulum, Pixel Observations

__Setup__
- Train on 6s of 28x28 pixel observations (60 datapoints)
- Infer latent initial state from 1s of data
- Forecast for 10s, reconstruct latent path

<div style="display: flex; justify-content: row;">
<img src="figures/pendulum_resnet_pixels.gif" width=32% title="Test"/>
<img src="figures/pendulum_vinvv_pixels.gif" width=32% />
<img src="figures/pendulum_vinso2_pixels.gif" width=32% />
</div>

#### Recurrent ResNet (Left) / VIN (Middle) / VIN on SO(2) Manifold (Right)