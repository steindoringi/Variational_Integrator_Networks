import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
from matplotlib import animation, rc
from matplotlib.lines import Line2D
rc('animation', html='jshtml')
rc('text', usetex=True)
font = {'size'   : 14}
rc('font', **font)


def create_diff_img(obs, pred):

    pred_ind = pred == 1.
    obs_ind = obs == 1.
    diff = np.zeros_like(obs)
    diff[obs_ind] = 1.
    diff[pred_ind] = 0.5

    return diff


def animate_energy(Es, dt, colors, labels, save_path=None):

    labels = [l.replace('_', ' ') for l in labels]

    sys_E = Es[0][0]
    m_E = np.mean(Es[1])
    scale_Es = [sys_E, m_E]
    Es = [E - scale_Es[e] for e, E in enumerate(Es)]

    x_max = dt * Es[0].shape[0]
    y_max = 10
    T = np.linspace(0., x_max, len(Es[0]))

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0., x_max), ylim=(-y_max, y_max))

    lines = [ax.plot([], [], '-', color=colors[l], label=labels[l])[0] for l in range(len(Es))]
    time_template = '$t = %.1f$s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, size=16)

    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return lines, time_text

    def animate(i):

        for l, line in enumerate(lines):
            x = T[:i]
            y = Es[l][:i, 0]
            line.set_data(x, y)

        time_text.set_text(time_template % (i * dt))
        return lines, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(Es[0])),
                                interval=30, blit=False, init_func=init)

    plt.xlabel('Time', fontdict={"size": 18})
    plt.ylabel('Shifted Energy', fontdict={"size": 18})

    plt.legend(bbox_to_anchor=(0.5,1.15), loc="upper center", ncol=len(Es))

    if save_path is not None:
        ani.save(save_path, writer='imagemagick', fps=25)

    return ani


def animate_predictions(qqdots, dt, colors, linestyles, labels, save_path=None, f_lim=None):

    labels = [l.replace('_', ' ') for l in labels]

    qqdot = qqdots[0]
    qqdot_max = np.max(np.abs(qqdot), axis=0)
    if f_lim is None:
        x_max = qqdot_max[0] + 2.0
        y_max = qqdot_max[1] + 2.0
    else:
        x_max = f_lim[0]
        y_max = f_lim[1]

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-x_max, x_max), ylim=(-y_max, y_max))

    lines = [ax.plot([], [], linestyles[l], color=colors[l], label=labels[l])[0] for l in range(len(qqdots))]
    time_template = '$t = %.1f$s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, size=16)
    mse_template = 'MSE$ = %.1f$'
    mse_text = ax.text(0.75, 0.9, '', transform=ax.transAxes, size=16, color='red')

    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return lines, time_text

    def animate(i):

        for l, line in enumerate(lines):
            x = qqdots[l][:i, 0]
            y = qqdots[l][:i, 1]
            line.set_data(x, y)

        mse = np.mean(np.sum((qqdots[0][:i] - qqdots[1][:i])**2, axis=1))
        time_text.set_text(time_template % (i * dt))
        mse_text.set_text(mse_template % mse)
        return lines, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(qqdot)),
                                interval=30, blit=False, init_func=init)

    plt.xlabel('$\\theta$', fontdict={"size": 18})
    plt.ylabel('$\\dot{\\theta}$', fontdict={"size": 18})

    plt.legend(bbox_to_anchor=(0.5,1.15), loc="upper center", ncol=len(qqdots))

    if save_path is not None:
        ani.save(save_path, writer='imagemagick', fps=25)

    return ani


def animate_pixel_predictions(ys, dt, colors, labels, save_path=None, title=None):

    labels = [l.replace('_', ' ') for l in labels]

    y_plot = ys[0]
    fig, ax = plt.subplots(1, figsize=(4, 4), dpi=6*28)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    time_template = '$t = %.1f$s'
    ims = []
    if title is not None:
        ax.set_title(title)

    if len(ys) > 1:
        y_plot = create_diff_img(ys[0], ys[1])
        cmap = matplotlib.colors.ListedColormap(['white', colors[1], colors[0]])
        bounds = [0., 0.05, 0.95, 1.0]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    else:
        cmap = matplotlib.colors.ListedColormap(['white', colors[0]])
        bounds = [0., 0.1, 1.0]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    custom_lines = [
        Line2D([0], [0], color=colors[0], lw=4),
        Line2D([0], [0], color=colors[1], lw=4)]

    for t in range(y_plot.shape[0]):

        time_text = ax.text(0.05, 0.9, time_template % (t * dt), transform=ax.transAxes, size=16)
        im = ax.imshow(y_plot[t],
            interpolation='nearest', cmap=cmap, norm=norm)
        
        ax.legend(custom_lines, labels, bbox_to_anchor=(0.5,1.15), loc="upper center", ncol=2)

        ims.append([im, time_text])

    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=100)

    if save_path is not None:
        ani.save(save_path, writer='imagemagick', fps=25)

    return ani