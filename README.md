# Variational_Integrator_Networks (Work in Progress)

- [ ] Add HNN Baseline
- [ ] Sampling/Uncertainty predictions plots
- [ ] Documentation

## Ideal Pendulum: Noisy States

### Test setup: given initial state, forecast the evolution for 10s.

### 3s Training data

#### ResNet

<p float="left">
  <img src="figures/pendulum/noisy-30/ResNet/1/eval/figures/traj_0.gif" width="300" />
  <img src="figures/pendulum/noisy-30/ResNet/1/eval/figures/traj_1.gif" width="300" /> 
  <img src="figures/pendulum/noisy-30/ResNet/1/eval/figures/traj_2.gif" width="300" />
</p>

#### Variational Integrator Network (Velocity Verlet)

<p float="left">
  <img src="figures/pendulum/noisy-30/VIN_VV/1/eval/figures/traj_0.gif" width="300" />
  <img src="figures/pendulum/noisy-30/VIN_VV/1/eval/figures/traj_1.gif" width="300" /> 
  <img src="figures/pendulum/noisy-30/VIN_VV/1/eval/figures/traj_2.gif" width="300" />
</p>

### 15s Training data

#### ResNet

<p float="left">
  <img src="figures/pendulum/noisy-150/ResNet/1/eval/figures/traj_0.gif" width="300" />
  <img src="figures/pendulum/noisy-150/ResNet/1/eval/figures/traj_1.gif" width="300" /> 
  <img src="figures/pendulum/noisy-150/ResNet/1/eval/figures/traj_2.gif" width="300" />
</p>

#### Variational Integrator Network (Velocity Verlet)

<p float="left">
  <img src="figures/pendulum/noisy-150/VIN_VV/1/eval/figures/traj_0.gif" width="300" />
  <img src="figures/pendulum/noisy-150/VIN_VV/1/eval/figures/traj_1.gif" width="300" /> 
  <img src="figures/pendulum/noisy-150/VIN_VV/1/eval/figures/traj_2.gif" width="300" />
</p>

## Pixel Observations

### 6s Training data. Test setup: given initial image observations, forecast the evolution for 10s.

#### ResNet, VIN-SO2, VIN-VV

<p float="left">
  <img src="figures/pendulum/pixels-60/ResNet/1/eval/figures/traj_0.gif" width="300" />
  <img src="figures/pendulum/pixels-60/VIN_SO2/1/eval/figures/traj_0.gif" width="300" />
  <img src="figures/pendulum/pixels-60/VIN_VV/1/eval/figures/traj_0.gif" width="300" />
</p>