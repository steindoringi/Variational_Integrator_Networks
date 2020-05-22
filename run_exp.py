import sys
import os

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

from tensorflow_probability import distributions as tfd

import gin
import build_utils
import plot_utils

import matplotlib.pyplot as plt
import seaborn as sns

COLORS = list(sns.color_palette())
MODEL_COLORS = {
    "ResNet": COLORS[0],
    "VIN_VV": COLORS[1],
    "VIN_SV": COLORS[2],
    "VIN_SO2": COLORS[3],
    "VINF_VV": COLORS[4]
}


@gin.configurable
def eval_model(eval_dir, model, system, model_name,
    num_traj=1, num_steps=100, step_size=0.1,
    sample=False, use_mean=True,
    create_animations=False,
    plot_pred=False):

    plot_dir = os.path.join(eval_dir, 'figures/')
    pred_dir = os.path.join(eval_dir, 'preds/')

    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)

    eval_obs, eval_states = system.run(num_traj, num_steps, step_size)

    if system.observations == 'pixels':
        train_dat = np.load(eval_dir+'/obs_states.npz')
        train_obs = train_dat['obs']
        train_states = train_dat['states']

        x0 = train_states[:, 0]
        train_obs, train_states = system.run(x0.shape[0], num_steps, step_size, y0=x0)

        eval_obs = np.vstack([train_obs, eval_obs])
        eval_states = np.vstack([train_states, eval_states])

    if system.observations == 'pixels':
        y0 = eval_obs[:, :model.inf_horizon]
    elif system.observations == 'noisy':
        #NOTE: Use ground truth state for easier comparison.
        # Not testing encoder network.
        if model.infer_qdot:
            y0 = eval_states[:, :1]
        else:
            y0 = eval_states[:, :2]
    else:
        y0 = eval_obs[:, :1]

    y0 = tf.constant(y0, dtype=tfk.backend.floatx())

    np.savez(pred_dir+'eval_obs_states.npz', obs=eval_obs, states=eval_states)

    pred_obs = []
    x = []

    for n in range(eval_obs.shape[0]):

        if system.observations == 'pixels':
            pred_obs_n, x_n = model.forward(y0[n:n+1], model.step_size, num_steps-1, sample=sample, use_mean=use_mean)
        elif system.observations == 'noisy':
            x_n = model.dynamics_network.forward(y0[n:n+1], model.step_size, num_steps-1)
            pred_obs_n = x_n
        else:
            pred_obs_n = model.forward(y0[n:n+1], model.step_size, num_steps-1)
            x_n = pred_obs_n

        pred_obs_n = pred_obs_n.numpy()
        x_n = x_n.numpy()

        pred_obs.append(pred_obs_n)
        x.append(x_n)

        if create_animations:

            save_path = plot_dir + 'traj_{}.gif'.format(n)
            print("Plotting {}".format(save_path))

            if system.observations == 'pixels':

                p_pred_obs_n = tfd.Bernoulli(logits=pred_obs_n)
                pred_obs_n = p_pred_obs_n.mean().numpy()
                pred_obs_n[pred_obs_n >= 0.5] = 1.0
                pred_obs_n[pred_obs_n < 0.5] = 0.

                plot_utils.animate_pixel_predictions(
                    [eval_obs[n].reshape(num_steps, 28, 28), pred_obs_n[0].reshape(num_steps, 28, 28)],
                    step_size, ["black", MODEL_COLORS[model_name]],
                    ["Ground Truth", model_name], save_path)
            else:

                obs_lim = np.max(np.abs(eval_obs[n]), axis=0)
                f_lim_x = obs_lim[0] + 0.2*obs_lim[0]
                f_lim_y = obs_lim[1] + 0.2*obs_lim[1]
                f_lim = (f_lim_x, f_lim_y)

                qqdots = [eval_states[n], pred_obs_n[0]]
                colors = ["black", MODEL_COLORS[model_name]]
                linestyles = ["-", "-o"]
                labels = ["Ground Truth", model_name]

                if system.observations == 'noisy':
                    qqdots.append(eval_obs[n])
                    colors.append("red")
                    linestyles.append("x")
                    labels.append("Observations")

                plot_utils.animate_predictions(
                    qqdots, step_size, colors, linestyles, labels,
                    save_path=save_path, f_lim=f_lim)

            if system.observations in ['observed', 'noisy']:
                gt_E = system.compute_energy(eval_states[n])
                model_E = system.compute_energy(x_n[0])

                colors = ["black", MODEL_COLORS[model_name]]
                linestyles = ["-", "-o"]
                labels = ["Ground Truth", model_name]

                save_path = plot_dir + 'energy_{}.gif'.format(n)
                print("Plotting {}".format(save_path))

                plot_utils.animate_energy([gt_E, model_E], step_size, colors, labels, save_path=save_path)

    pred_obs = np.vstack(pred_obs)
    x = np.vstack(x)

    if plot_pred:
        for n, x_n in enumerate(x):
            if 'VIN' in model_name:
                if system.observations in ['noisy', 'pixels']:
                    U = model.dynamics_network.potential(x_n[:, :1]).numpy()
                else:
                    U = model.potential(x_n[:, :1]).numpy()
                f, ax = plt.subplots(x_n.shape[1]+1)
                ax[x_n.shape[1]].plot(U[:, 0], color='blue')
            else:
                f, ax = plt.subplots(x_n.shape[1])
            for d in range(x_n.shape[1]):
                ax[d].plot(x_n[:, d], color='red')
                if system.observations in ['observed', 'noisy']:
                    ax[d].plot(eval_states[n, :, d], color='black')
            plt.savefig(plot_dir + 'pred_{}.png'.format(n))

    if system.observations in ['observed', 'noisy']:
        mse = np.mean(np.sum((pred_obs - eval_states)**2, axis=2), axis=0)
    else:
        mse = np.mean(np.sum((pred_obs - eval_obs)**2, axis=2), axis=0)
    np.savez(eval_dir+'/mse.npz', mse)
    model_str_list = eval_dir.replace('eval', '').split('/')
    print("{} MSE: {:.2f}".format(' '.join(model_str_list), np.mean(mse)))


@gin.configurable
def main(root_dir, model_name, system_name, observations, num_train_traj, num_train_steps, seed,
        train_flag=True, eval_flag=True, save_every=100):

    exp_dir = os.path.join(
        root_dir, system_name,
        observations + '-{}'.format(num_train_traj*num_train_steps),
        model_name, str(seed))

    checkpoint_dir = os.path.join(exp_dir, 'training/')
    eval_dir = os.path.join(exp_dir, 'eval/')

    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    model = build_utils.create_model(observations, model_name)
    system = build_utils.create_system(system_name, observations, seed)

    if train_flag:

        checkpoint_path = checkpoint_dir + 'cp-{epoch:04d}.ckpt'
        cp_callback = tfk.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq=save_every)

        model.save_weights(checkpoint_path.format(epoch=0))

        train_dataset, valid_dataset, callbacks, num_epochs, obs, states =\
            build_utils.create_dataset(system, model.horizon, num_train_traj, num_train_steps)


        np.savez(eval_dir+'/obs_states.npz', obs=obs, states=states)

        if callbacks is not None:
            callbacks += [cp_callback]
        else:
            callbacks = [cp_callback]

        model.fit(train_dataset, validation_data=valid_dataset,
            epochs=num_epochs, callbacks=callbacks + [cp_callback])

    if eval_flag:

        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        eval_model(eval_dir, model, system, model_name)


if __name__ == '__main__':

    root_dir = sys.argv[1]
    model_name = sys.argv[2]
    system_name = sys.argv[3]
    observations = sys.argv[4]
    num_train_traj = int(sys.argv[5])
    num_train_steps = int(sys.argv[6])
    seed = int(sys.argv[7])

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    #NOTE: Configs may overwrite arguments in prev. files
    gin.parse_config_file('./config/base.gin')
    gin.parse_config_file('./config/{}.gin'.format(observations))
    if model_name == 'VIN_SO2' and observations == 'pixels':
        # Overwrite step-size for SO2 VIN
        gin.parse_config_file('./config/pixels_SO2.gin')
    gin.parse_config_file('./config/{}.gin'.format(system_name))

    main(root_dir, model_name, system_name, observations, num_train_traj, num_train_steps, seed)