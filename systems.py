import numpy as np
from scipy.integrate import ode

import plot_utils
import matplotlib.pyplot as plt


class System(object):

    def __init__(self, observations, seed, **kwargs):

        self.observations = observations
        self.RNG = np.random.RandomState(seed)
        self.system_param = kwargs
        self._init_system(**kwargs)

    def _init_system(self, **kwargs):
        raise NotImplementedError()

    def _random_init_state(self):
        raise NotImplementedError()

    def _ODE(self, t, y):
        raise NotImplementedError()

    def _state_to_observations(self, qqdot):
        if self.observations == "observed":
            return qqdot
        elif self.observations == "noisy":
            noise = self.RNG.randn(*qqdot.shape) * self.system_param["noise_std"]
            return qqdot + noise
        elif self.observations == "pixels":
            return self._state_to_pixels(qqdot)

    def _state_to_pixels(self, qqdot):
        raise NotImplementedError()

    def compute_energy(self, qqdot):
        raise NotImplementedError()

    def integrate_ODE(self, num_steps, step_size, y0=None, rtol=1e-12):

        T = num_steps * step_size
        t = np.linspace(0.0, T, num_steps)

        if y0 is None:
            y0 = self._random_init_state()

        solver = ode(self._ODE).set_integrator('dop853', rtol=rtol)
        sol = np.empty((len(t), 2))
        sol[0] = y0
        solver.set_initial_value(y0)
        k = 1
        while solver.successful() and solver.t < T:
            solver.integrate(t[k])
            sol[k] = solver.y
            k += 1
        
        return sol

    def run(self, num_traj, num_steps, step_size, y0=None):

        qqd = []
        for n in range(num_traj):
            if y0 is not None:
                y0_n = y0[n]
            else:
                y0_n = None
            qqd_n = self.integrate_ODE(num_steps, step_size, y0=y0_n)
            qqd.append(qqd_n[None])
        qqd = np.vstack(qqd)

        return self._state_to_observations(qqd), qqd

class Pendulum(System):

    def _init_system(self, mass=1.0, length=1.0, g=9.82, friction=0., **kwargs):

        def pendulum_ODE(t, y):
            q, qdot = y
            dydt = [qdot, -friction*qdot - (g/length)*np.sin(q)]
            return dydt

        self._ODE = pendulum_ODE

    def _random_init_state(self):

        sign = [-1., 1.]
        q0 = np.pi - self.RNG.uniform(low=0.1, high=1./2.*np.pi)
        q0 = self.RNG.choice(sign)*q0
        qdot0 = self.RNG.uniform(low=-1., high=1.)
        y0 = np.hstack([q0, qdot0])
        return y0

    def _state_to_pixels(self, qqdot):

        length = self.system_param["length"]
        q, qdot = np.split(qqdot, 2, axis=-1)
        x = length * np.sin(q)
        y = -length * np.cos(q)

        f, ax = plt.subplots(figsize=(1, 1), dpi=28)
        f_lim = length+0.2

        ys = []
        for traj in range(q.shape[0]):
            y_traj = []
            for t in range(q.shape[1]):
                plt.cla()
                ax.set_xlim(-f_lim, f_lim)
                ax.set_ylim(-f_lim, f_lim)
                ax.plot([0, x[traj, t]],[0, y[traj, t]], linewidth=12, color="black")
                ax.axis('off')
                f.canvas.draw()
                y_t = np.array(f.canvas.renderer.buffer_rgba())[:, :, :1]
                y_t[y_t > 0] = 255
                y_t[y_t == 0] = 1.
                y_t[y_t == 255] = 0.
                y_traj.append(np.float32(y_t[None]))
            y_traj = np.vstack(y_traj)
            ys.append(y_traj[None])
        ys = np.vstack(ys)
        ys = ys.reshape(*ys.shape[:2], 784)

        return ys

    def compute_energy(self, qqdot):

        mass = self.system_param["mass"]
        length = self.system_param["length"]
        g = self.system_param["g"]

        q, qdot = np.split(qqdot, 2, axis=-1)
        K = 0.5 * mass * (length**2) * qdot**2 
        U = -mass * g * length * np.cos(q)
        E = K + U

        return E

class SpringMass(System):

    def _init_system(self, mass=1.0, k=0.75, **kwargs):

        def mass_spring_ODE(t, y):
            q, qdot = y
            dydt = [qdot, -(k/mass)*q]
            return dydt

        self._ODE = mass_spring_ODE

    def _random_init_state(self):

        q0 = self.RNG.uniform(low=1.0, high=1.5)
        sign = [-1., 1.]
        q0 = self.RNG.choice(sign) * q0
        qdot0 = self.RNG.uniform(low=-0.5, high=0.5)
        qdot0 = self.RNG.choice(sign) * qdot0
        y0 = np.hstack([q0, qdot0])
        return y0

    def _state_to_pixels(self, qqdot):

        q, qdot = np.split(qqdot, 2, axis=-1)
        x = q

        f, ax = plt.subplots(figsize=(1, 1), dpi=28)
        f_lim = 2.5

        ys = []
        for traj in range(q.shape[0]):
            y_traj = []
            for t in range(q.shape[1]):
                plt.cla()
                ax.set_xlim(-f_lim, f_lim)
                ax.set_ylim(-f_lim, f_lim)
                ax.scatter([x[traj, t]], [0.], marker='o', s=400, color="black")
                ax.axis('off')
                f.canvas.draw()
                y_t = np.array(f.canvas.renderer.buffer_rgba())[:, :, :1]
                y_t[y_t > 0] = 255
                y_t[y_t == 0] = 1.
                y_t[y_t == 255] = 0.
                y_traj.append(np.float32(y_t[None]))
            y_traj = np.vstack(y_traj)
            ys.append(y_traj[None])
        ys = np.vstack(ys)
        ys = ys.reshape(*ys.shape[:2], 784)

        return ys
    
    def compute_energy(self, qqdot):

        mass = self.system_param["mass"]
        k = self.system_param["k"]

        q, qdot = np.split(qqdot, 2, axis=-1)
        K = 0.5 * mass * qdot**2 
        U = -0.5 * k * q**2
        E = K + U

        return E