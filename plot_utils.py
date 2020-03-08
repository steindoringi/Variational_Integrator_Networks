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

    fig = plt.figure(figsize=(6, 4), dpi=150)
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

    plt.xlabel('Time')
    plt.ylabel('Shifted Energy')

    plt.legend(bbox_to_anchor=(0.6,1), loc="upper center", ncol=len(Es))

    plt.tight_layout()

    if save_path is not None:
        #ani.save(save_path, writer='imagemagick', fps=25)
        #with open(save_path.replace('gif', 'html'), 'w') as f:
        #    print(ani.to_jshtml(), file=f)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, bitrate=3600, codec='h264')
        #extra_args=['-vcodec', 'h264',  '-pix_fmt', 'yuv420p']
        ani.save(save_path.replace('gif', 'mp4'), writer=writer)

    return ani


def animate_predictions(qqdots, dt, colors, linestyles, labels, save_path=None, f_lim=None, plot_mse=True, num_active=10):

    labels = [l.replace('_', ' ') for l in labels]

    qqdot = qqdots[0]
    qqdot_max = np.max(np.abs(qqdot), axis=0)
    if f_lim is None:
        x_max = qqdot_max[0] + 2.0
        y_max = qqdot_max[1] + 2.0
    else:
        x_max = f_lim[0]
        y_max = f_lim[1]

    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-x_max, x_max), ylim=(-y_max, y_max))

    lines = [ax.plot([], [], linestyles[l], color=colors[l], label=labels[l])[0] for l in range(len(qqdots))]
    points = [ax.plot([], [], 'o', color=colors[l])[0] for l in range(len(qqdots))]
    time_template = '$t = %.1f$s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, size=16)
    if len(qqdots) > 1 and plot_mse:
        mse_template = 'MSE$ = %.1f$'
        mse_text = ax.text(0.75, 0.9, '', transform=ax.transAxes, size=16, color='red')
    else:
        mse_template = None

    def init():
        for line in lines:
            line.set_data([], [])
        for point in points:
            point.set_data([], [])

        time_text.set_text('')
        return lines, points, time_text

    def animate(i):

        for l, line in enumerate(lines):
            if labels[l] == 'Observations':
                ind = max(i-num_active, 0)
                x = qqdots[l][ind:i+1, 0]
                y = qqdots[l][ind:i+1, 1]
                line.set_data(x, y)
            elif labels[l] == 'Ground Truth':
                x = qqdots[l][:i, 0]
                y = qqdots[l][:i, 1]
                line.set_data(x, y)
            else:
                ind = max(i-num_active, 0)
                x = qqdots[l][ind:i+1, 0]
                y = qqdots[l][ind:i+1, 1]
                points[l].set_data(x, y)
                x_line = qqdots[l][:ind+1, 0]
                y_line = qqdots[l][:ind+1, 1]
                line.set_data(x_line, y_line)
                line.set_alpha(0.33)

        time_text.set_text(time_template % (i * dt))
        if mse_template is not None:
            mse = np.mean(np.sum((qqdots[0][:i] - qqdots[1][:i])**2, axis=1))
            mse_text.set_text(mse_template % mse)

        return lines

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(qqdot)),
                                interval=50, blit=False, init_func=init)

    plt.xlabel('$\\theta$', labelpad=-1)
    plt.ylabel('$\\dot{\\theta}$', labelpad=-1)

    plt.legend(bbox_to_anchor=(0.5,1.15), loc="upper center", ncol=len(qqdots))

    if save_path is not None:
        #ani.save(save_path, writer='imagemagick', fps=20)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, bitrate=3600, codec='h264')
        #extra_args=['-vcodec', 'h264',  '-pix_fmt', 'yuv420p']
        ani.save(save_path.replace('gif', 'mp4'), writer=writer)


        #with open(save_path.replace('gif', 'html'), 'w') as f:
        #    print(ani.to_jshtml(), file=f)

    return ani

def animate_latent(qqdots, dt, colors, linestyles, labels, save_path=None, f_lim=None):

    labels = [l.replace('_', ' ') for l in labels]

    qqdot = qqdots[0]
    qqdot_max = np.max(np.abs(qqdot), axis=0)
    if f_lim is None:
        x_max = qqdot_max[0] + 2.0
        y_max = qqdot_max[1] + 2.0
    else:
        x_max = f_lim[0]
        y_max = f_lim[1]

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-x_max, x_max), ylim=(-y_max, y_max))

    lines = [ax.plot([], [], linestyles[l], color=colors[l], label=labels[l])[0] for l in range(len(qqdots))]
    
    time_template = '$t = %.1f$s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, size=16)

    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return lines, time_text

    def animate(i):

        for l, line in enumerate(lines):
            x = qqdots[l][:i, 0]
            if 'SO2' in labels[l]:
                y = (x - qqdots[l][:i, 1]) / dt
            else:
                y = qqdots[l][:i, 1]
            line.set_data(x, y)
        time_text.set_text(time_template % (i * dt))

        return lines, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(qqdot)),
                                interval=30, blit=False, init_func=init)

    plt.xlabel('$q$', fontdict={"size": 16})
    plt.ylabel('$\\dot{q}$', fontdict={"size": 16})

    plt.legend(bbox_to_anchor=(0.5,1.15), loc="upper center", ncol=len(qqdots))

    if save_path is not None:
        ani.save(save_path, writer='imagemagick', fps=25)

        #with open(save_path.replace('gif', 'html'), 'w') as f:
        #    print(ani.to_jshtml(), file=f)

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

    custom_lines = [Line2D([0], [0], color=colors[l], lw=4) for l in range(len(ys))]

    for t in range(y_plot.shape[0]):

        time_text = ax.text(0.05, 0.9, time_template % (t * dt), transform=ax.transAxes, size=16)
        im = ax.imshow(y_plot[t],
            interpolation='nearest', cmap=cmap, norm=norm)
        
        ax.legend(custom_lines, labels, bbox_to_anchor=(0.5,1.15), loc="upper center", ncol=2)

        ims.append([im, time_text])

    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=100)

    if save_path is not None:
        #ani.save(save_path, writer='imagemagick', fps=25)
        #with open(save_path.replace('gif', 'html'), 'w') as f:
        #    print(ani.to_jshtml(), file=f)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, bitrate=3600, codec='h264')
        #extra_args=['-vcodec', 'h264',  '-pix_fmt', 'yuv420p']
        ani.save(save_path.replace('gif', 'mp4'), writer=writer)

    return ani


def animate_pixel_latent_predictions(ys, qqdots, dt, colors, linestyles, labels, save_path=None):

    labels = [l.replace('_', ' ') for l in labels]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=100)

    """ Pixels """
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    time_template = '$t = %.1f$s'

    y_plot = create_diff_img(ys[0], ys[1])
    cmap = matplotlib.colors.ListedColormap(['white', colors[1], colors[0]])
    bounds = [0., 0.05, 0.95, 1.0]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    custom_lines = [Line2D([0], [0], color=colors[l], lw=4) for l in range(len(ys))]

    time_text_1 = ax[0].text(0.05, 0.9, '', transform=ax[0].transAxes, size=16)
    ax[0].legend(custom_lines, labels, bbox_to_anchor=(0.5,1.15), loc="upper center", ncol=2)

    im = ax[0].imshow(y_plot[0],
            interpolation='nearest', cmap=cmap, norm=norm)

    """ Latent """
    num_active = 10
    qqdot = qqdots[0]
    qqdot_max = np.max(np.abs(qqdot), axis=0)
    x_max = qqdot_max[0] + 2.0
    y_max = qqdot_max[1] + 2.0

    ax[1].set_xlim(-x_max, x_max)
    ax[1].set_ylim(-y_max, y_max)

    lines = [ax[1].plot([], [], '-', color=colors[l+1], label=labels[l+1] + ' Latent', alpha=0.5)[0] for l in range(len(qqdots))]
    points = [ax[1].plot([], [], 'o', color=colors[l+1])[0] for l in range(len(qqdots))]

    if 'VIN' in labels[1]:
        ax[1].set_xlabel('$q$', fontdict={"size": 16})
        ax[1].set_ylabel('$\\dot{q}$', fontdict={"size": 16})
    else:
        ax[1].set_xlabel('$x_1$', fontdict={"size": 16})
        ax[1].set_ylabel('$x_2$', fontdict={"size": 16})
    ax[1].legend(bbox_to_anchor=(0.5,1.15), loc="upper center", ncol=1)

    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    time_text_2 = ax[1].text(0.05, 0.9, '', transform=ax[1].transAxes, size=16)

    def init():
        im.set_data(np.zeros_like(y_plot[0]))
        time_text_1.set_text('')
        for line in lines:
            line.set_data([], [])
        for point in points:
            point.set_data([], [])
        time_text_2.set_text('')
        return im, lines, points, time_text_1, time_text_2

    def animate(i):

        im.set_data(y_plot[i])
        time_text_1.set_text(time_template % (i * dt))

        for l, line in enumerate(lines):
            x = qqdots[l][:i, 0]
            if 'SO2' in labels[l+1] or 'SV' in labels[l+1]:
                y = (x - qqdots[l][:i, 1]) / dt
            else:
                y = qqdots[l][:i, 1]

            line.set_data(x, y)
            #print(line)
            #line.set_alpha(0.5)

            point = points[l]
            point.set_data(x[-num_active:], y[-num_active:])

        time_text_2.set_text(time_template % (i * dt))

        return im, lines, points, time_text_1, time_text_2

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(qqdot)),
                                interval=30, blit=False, init_func=init)

    if save_path is not None:
        ani.save(save_path, writer='imagemagick', fps=30)
        #with open(save_path.replace('gif', 'html'), 'w') as f:
        #    print(ani.to_jshtml(), file=f)
        
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=30, bitrate=3600, codec='h264')
        #extra_args=['-vcodec', 'h264',  '-pix_fmt', 'yuv420p']
        #ani.save(save_path.replace('gif', 'mp4'), writer=writer)

    return ani