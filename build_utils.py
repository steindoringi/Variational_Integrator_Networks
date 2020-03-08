import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

import gin
import models
import systems


@gin.configurable
def create_model(
    observations,
    model_name,
    step_size=0.1,
    horizon=10,
    dim_obs=784,
    dim_latent=10,
    inf_horizon=1,
    infer_qdot=True,
    **model_param
    ):

    if model_name == 'ResNet':
        dynamics = models.ResNet(step_size, horizon, name=model_name, **model_param)
    elif model_name == 'VIN_VV':
        dynamics = models.VIN_VV(step_size, horizon, name=model_name, **model_param)
    elif model_name == 'VIN_SV':
        dynamics = models.VIN_SV(step_size, horizon, name=model_name, **model_param)
    elif model_name == 'VIN_SO2':
        dynamics = models.VIN_SO2(step_size, horizon, name=model_name, **model_param)
    elif model_name == 'FeedForward':
        dynamics = models.FeedForward(step_size, 1, name=model_name, **model_param)
    elif model_name in ['VAE', 'LG_VAE']:
        dynamics = None
    else:
        raise NotImplementedError()

    if model_name in ['VIN_SV', 'VIN_SO2']:
        infer_qdot = False

    if observations == 'observed':
        model = dynamics
    elif observations == 'noisy':
        dim_dec_in = dynamics.dim_state
        dim_obs = dynamics.dim_state
        model = models.NoisyLDDN(dim_obs, dim_dec_in, dynamics, inf_horizon,
            name="NoisyLDDN", infer_qdot=infer_qdot, **model_param)
    elif observations == 'pixels':
        if model_name in ['ResNet', 'FeedForward']:
            dim_dec_in = dynamics.dim_state
        elif 'VIN' in model_name:
            dim_dec_in = dynamics.dim_Q

        if model_name in ['VIN_SO2', 'LG_VAE']:
            dec_inp_fn = lambda x: tf.concat([tf.sin(x), tf.cos(x)], 1)
        else:
            dec_inp_fn = None
        if dynamics is not None:
            model = models.PixelLDDN(dim_obs, dim_dec_in, dynamics, inf_horizon,
                name="PixelLDDN", infer_qdot=infer_qdot, dec_inp_fn=dec_inp_fn, **model_param)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    
    return compile_model(model, observations)

@gin.configurable
def compile_model(
    model,
    observations,
    learning_rate=3e-4):

    if observations == 'observed':
        loss = 'mse'
    elif observations in ['noisy', 'pixels']:
        loss = model.loss

    optimizer = tfk.optimizers.Adam(learning_rate)
    model.compile(optimizer, loss=loss)

    return model


@gin.configurable
def create_system(
    system_name,
    observations,
    seed,
    **system_param
    ):

    if system_name == 'pendulum':
        system = systems.Pendulum(observations, seed, **system_param)
    elif system_name == 'springmass':
        system = systems.SpringMass(observations, seed, **system_param)
    else:
        raise NotImplementedError()

    return system

@gin.configurable
def create_dataset(
    system,
    horizon,
    num_traj,
    num_steps,
    step_size=0.1,
    num_train_traj=None,
    num_epochs=1000,
    batch_size=100,
    shuffle_buffer=10000,
    reshuffle=True,
    min_delta=1e-3,
    patience=5,
    restore_weights=True
    ):

    obs, states = system.run(num_traj, num_steps, step_size)
    obs_windowed = []
    for t in range(obs.shape[1]-horizon):
        qqd_w = obs[:, t:t+horizon+1]
        obs_windowed.append(qqd_w[:, None])
    obs_windowed = np.concatenate(obs_windowed, axis=1)

    obs_train = obs_windowed[:num_train_traj]
    obs_train = obs_train.reshape(-1, horizon+1, *obs.shape[2:])

    if system.observations == 'observed':
        x_train, y_train = obs_train[:, :1], obs_train[:, 1:]
    elif system.observations in ['noisy', 'pixels']:
        x_train, y_train = obs_train, obs_train

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=reshuffle)
    train_dataset = train_dataset.batch(batch_size)

    if num_train_traj is not None and num_traj - num_train_traj > 0:
        obs_valid = obs_windowed[num_train_traj:]
        obs_valid = obs_valid.reshape(-1, horizon+1, *obs.shape[2:])
        if system.observations in ['observed', 'noisy']:
            x_valid, y_valid = obs_valid[:, :1], obs_valid[:, 1:]
        elif system.observations == 'pixels':
            x_valid, y_valid = obs_valid, obs_valid
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        valid_dataset = valid_dataset.batch(batch_size)
    else:
        valid_dataset = None

    if valid_dataset is not None:
        monitor = 'val_loss'
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=min_delta,
                mode='auto',
                patience=patience,
                verbose=1,
                restore_best_weights=restore_weights)
            ]
    else:
        callbacks = None

    

    return train_dataset, valid_dataset, callbacks, num_epochs, obs, states
