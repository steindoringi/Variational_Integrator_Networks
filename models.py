import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import numpy as np


class DDN(tfk.Model):
    """ Deep Dynamical Network """

    def __init__(self, step_size, horizon, name="DDN", **kwargs):

        super(DDN, self).__init__(name=name)

        self._step_size = step_size
        self._horizon = horizon

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self._build_network(**kwargs)

        inputs = tfk.Input(shape=(None, self.dim_state), dtype=tfk.backend.floatx())
        outputs = self.call(inputs)

        self._set_input_attrs(inputs)
        self._set_output_attrs(outputs)

    @property
    def step_size(self):
        return self._step_size

    @property
    def horizon(self):
        return self._horizon

    def _build_network(self, **kwargs):
        raise NotImplementedError()

    def step(self, x, step_size, t):
        raise NotImplementedError()

    def call(self, x0):
        return self.forward(x0, self.step_size, self.horizon)[:, 1:]

    def forward(self, x0, step_size, horizon):

        x = [x0]
        for t in range(horizon):
            x_t = x[-1][:, -1]
            x_next = self.step(x_t, step_size, t)
            x.append(x_next[:, None])

        return tf.concat(x, 1)


class FeedForward(DDN):
    """ Feed Forward Network """

    def _build_network(self, dim_state=10, dim_h=500, activation='relu', **kwargs):

        self.network = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation),
            tfk.layers.Dense(dim_state)
        ])

    def step(self, x, step_size, t):
        return self.network(x)


class ResNet(DDN):
    """ Residual Network """

    def _build_network(self, dim_state=10, dim_h=500, activation='relu', **kwargs):

        self.network = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation),
            tfk.layers.Dense(dim_state)
        ])

    def step(self, x, step_size, t):
        dxdt = self.network(x)
        return x + step_size * dxdt


class VIN(DDN):
    """ Variational Integrator Network """

    def _build_network(self, dim_state=10, dim_h=500, activation='relu', learn_inertia=False, **kwargs):

        self.dim_Q = self.dim_state // 2

        self.potential = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation),
            tfk.layers.Dense(1, use_bias=False)
        ])

        self.learn_inertia = learn_inertia
        if self.learn_inertia:
            num_w = int(self.dim_Q * (self.dim_Q + 1) / 2)
            self.L_param = tf.Variable(num_w*[0.], dtype=tfk.backend.floatx())
        else:
            self.L_param = None

    @property
    def M_inv(self):
        if self.learn_inertia:
            L = tfp.math.fill_triangular(self.L_param)
            M_inv = tf.transpose(L) @ L
        else:
            M_inv = tf.linalg.diag(tf.ones((self.dim_Q,), dtype=tfk.backend.floatx()))
        return M_inv

    def grad_potential(self, q):

        with tf.GradientTape() as g:
            g.watch(q)
            U = self.potential(q)

        return g.gradient(U, q)

    def step(self, x, step_size, t):
        raise NotImplementedError()


class VIN_SV(VIN):
    """ Störmer-Verlet VIN """

    def step(self, x, step_size, t):

        q = x[:, :self.dim_Q]
        q_prev = x[:, self.dim_Q:]
        dUdq = self.grad_potential(q)

        qddot = tf.einsum('jk,ik->ij', self.M_inv, dUdq)
        q_next = 2 * q - q_prev - (step_size**2) * qddot

        return tf.concat([q_next, q], 1)

    def forward(self, q0, step_size, horizon):

        x0 = tf.concat([q0[:, :1], tf.zeros_like(q0[:, :1])], 2)
        x1 = tf.concat([q0[:, 1:2], q0[:, :1]], 2)
        x0 = tf.concat([x0, x1], 1)
        x = [x0]
        for t in range(horizon-1):
            x_t = x[-1][:, -1]
            x_next = self.step(x_t, step_size, t)
            x.append(x_next[:, None])

        return tf.concat(x, 1)


class VIN_VV(VIN):
    """ Velocity-Verlet VIN """

    def step(self, x, step_size, t):

        q = x[:, :self.dim_Q]
        qdot = x[:, self.dim_Q:]
        dUdq = self.grad_potential(q)
        
        qddot = tf.einsum('jk,ik->ij', self.M_inv, dUdq)

        q_next = q + step_size * qdot - 0.5 * (step_size**2) * qddot
        dUdq_next = self.grad_potential(q_next)

        dUdq_mid = dUdq + dUdq_next
        qddot_mid = tf.einsum('jk,ik->ij', self.M_inv, dUdq_mid)

        qdot_next = qdot - 0.5 * step_size * qddot_mid

        return tf.concat([q_next, qdot_next], 1)


class VIN_SO2(VIN):
    """ SO2 VIN """

    def _build_network(self, dim_state=10, dim_h=500, activation='relu', learn_inertia=False, **kwargs):

        self.dim_Q = self.dim_state // 2

        self.potential = tfk.Sequential([
            tfk.layers.Dense(dim_h, activation=activation),
            tfk.layers.Dense(1, activation=tf.sin)
        ])

        if learn_inertia:
            num_w = int(self.dim_Q * (self.dim_Q + 1) / 2)
            self.L_param = tf.Variable(num_w*[0.], dtype=tfk.backend.floatx())
        else:
            self.L_param = None

    def step(self, x, step_size, t):

        eps = 1e-4

        q = x[:, :self.dim_Q]
        q_prev = x[:, self.dim_Q:]
        dUdq = self.grad_potential(q)

        qddot = tf.einsum('jk,ik->ij', self.M_inv, dUdq)
        
        sin_q_delta = tf.sin(q - q_prev) + (step_size**2) * qddot
        q_delta = tf.math.asin(tf.clip_by_value(sin_q_delta, -(1.-eps), 1.-eps))
        q_next = q + q_delta

        return tf.concat([q_next, q], 1)

    def forward(self, q0, step_size, horizon):

        x0 = tf.concat([q0[:, :1], tf.zeros_like(q0[:, :1])], 2)
        x1 = tf.concat([q0[:, 1:2], q0[:, :1]], 2)
        x0 = tf.concat([x0, x1], 1)
        x = [x0]
        for t in range(horizon-1):
            x_t = x[-1][:, -1]
            x_next = self.step(x_t, step_size, t)
            x.append(x_next[:, None])

        return tf.concat(x, 1)


class LDDN(tfk.Model):
    """ Latent Deep Dynamical Network """

    def __init__(self, dim_obs, dim_dec_in, dynamics_network, inf_horizon, name="LDDN", **kwargs):

        super(LDDN, self).__init__(name=name)

        self.dim_obs = dim_obs
        self.dim_dec_in = dim_dec_in
        self.inf_horizon = inf_horizon
        self.dynamics_network = dynamics_network

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self._build_network(**kwargs)

        inputs = tfk.Input(shape=(None, self.dim_obs), dtype=tfk.backend.floatx())
        outputs = self.call(inputs)

        self._set_input_attrs(inputs)
        self._set_output_attrs(outputs)

    @property
    def step_size(self):
        return self.dynamics_network._step_size

    @property
    def horizon(self):
        return self.dynamics_network._horizon

    def _build_network(self, **kwargs):
        raise NotImplementedError()

    def encode(self, y, sample=False, use_mean=False):
        
        batch_size = tf.shape(y)[0]
        seq_len = tf.shape(y)[1]
        y_inp = tf.reshape(y, [batch_size*seq_len, self.dim_obs])
        x_param = self.q_inf_network(y_inp)
        x_param = tf.reshape(x_param, [batch_size, seq_len, x_param.shape[1]])

        if self.infer_qdot:
            x_param = self.qdot_inf_network(x_param)

        if sample:
            x_mu = x_param[:, :, :self.dim_latent]
            x_var = tf.nn.softplus(x_param[:, :, self.dim_latent:])
            qx = tfd.Normal(x_mu, x_var)
            x_sample = qx.sample()
            if self.infer_qdot:
                return x_sample[:, :1]
            else:
                return x_sample[:, :2]
        elif use_mean:
            x_mu = x_param[:, :, :self.dim_latent]
            if self.infer_qdot:
                return x_mu[:, :1]
            else:
                return x_mu[:, :2]
        else:
            return x_param

    def decode(self, x):

        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size*seq_len, self.dynamics_network.dim_state])
        x_inp = x[:, :self.dim_dec_in]
        y_rec = self.observation_network(x_inp)
        y_rec = tf.reshape(y_rec, [batch_size, seq_len, self.dim_obs])
        return y_rec

    def log_likelihood_y(self, y, y_rec):
        raise NotImplementedError()

    def call(self, y):

        y0 = y[:, :self.inf_horizon]
        x0 = self.encode(y0, sample=True)
        x = self.dynamics_network.forward(x0, self.step_size, self.horizon)
        return self.decode(x)

    def loss(self, y, y_rec):

        log_py = self.log_likelihood_y(y, y_rec)
        x_param = self.encode(y[:, :self.inf_horizon])
        kl_x = self.kullback_leibler_x(x_param)
        return -log_py + kl_x

    def forward(self, y0, step_size, horizon, sample=True, use_mean=False):

        x0 = self.encode(y0, sample=sample, use_mean=use_mean)
        x = self.dynamics_network.forward(x0, step_size, horizon)
        return self.decode(x), x


class NoisyLDDN(LDDN):
    """ Noisy Latent DDN """

    def _build_network(self, infer_qdot=False, dim_h_inf=100, dim_h_LSTM=20, activation='relu', **kwargs):

        self.q_inf_network = tfk.Sequential([
            tfk.layers.Dense(dim_h_inf, activation='relu')
        ])

        if infer_qdot:
            self.q_inf_network.add(tfk.layers.Dense(2*self.dynamics_network.dim_state))
            self.qdot_inf_network = tfk.Sequential([
                tfk.layers.LSTM(dim_h_LSTM, return_sequences=True, go_backwards=True),
                tfk.layers.Dense(2*self.dynamics_network.dim_state)
            ])
            self.dim_latent = self.dynamics_network.dim_state
        else:
            self.q_inf_network.add(tfk.layers.Dense(2*self.dynamics_network.dim_Q))
            self.qdot_inf_network = None
            self.dim_latent = self.dynamics_network.dim_Q

        self.observation_network = tfk.Sequential([
            tfk.layers.Lambda(lambda x: x)
            ])

        self.log_noise_var = tf.Variable([0.], dtype=tfk.backend.floatx())

    def call(self, y):

        y0 = y[:, :self.inf_horizon]
        x0 = self.encode(y0, sample=False, use_mean=True)
        x = self.dynamics_network.forward(x0, self.step_size, self.horizon)
        return self.decode(x)

    def kullback_leibler_x(self, x_param):
        x_mu = x_param[:, :, :self.dim_latent]
        x_var = tf.nn.softplus(x_param[:, :, self.dim_latent:])
        qx = tfd.Normal(x_mu, x_var)
        
        px_mu = tf.zeros_like(x_mu)
        px_var = tf.ones_like(x_var)
        px = tfd.Normal(px_mu, px_var)

        kl_x = tfd.kl_divergence(qx, px)
        kl_x = tf.reduce_sum(kl_x, [2, 1])
        kl_x = tf.reduce_mean(kl_x)
        return kl_x

    def log_likelihood_y(self, y, y_rec):
        noise_var = tf.nn.softplus(self.log_noise_var) * tf.ones_like(y_rec)
        py = tfd.Normal(y_rec, noise_var)
        log_py = py.log_prob(y)
        log_py = tf.reduce_sum(log_py, [2])
        log_lik = tf.reduce_mean(log_py)
        return log_lik

    def loss(self, y, y_rec):

        log_py = self.log_likelihood_y(y, y_rec)
        return -log_py

class PixelLDDN(LDDN):
    """ Pixel Latent DDN """

    def _build_network(self, infer_qdot=False, dec_inp_fn=None, dim_h_dec=1000, dim_h_inf=1000, dim_h_LSTM=20, activation='relu', **kwargs):

        self.q_inf_network = tfk.Sequential([
            tfk.layers.Dense(dim_h_inf, activation='relu'),
            tfk.layers.Dense(dim_h_inf, activation='relu')
        ])

        if infer_qdot:
            self.q_inf_network.add(tfk.layers.Dense(2*self.dynamics_network.dim_state))
            self.qdot_inf_network = tfk.Sequential([
                tfk.layers.LSTM(dim_h_LSTM, return_sequences=True, go_backwards=True),
                tfk.layers.Dense(2*self.dynamics_network.dim_state)
            ])
            self.dim_latent = self.dynamics_network.dim_state
        else:
            self.q_inf_network.add(tfk.layers.Dense(2*self.dynamics_network.dim_Q))
            self.qdot_inf_network = None
            self.dim_latent = self.dynamics_network.dim_Q

        if dec_inp_fn is None:
            dec_inp_fn = lambda x: x

        self.observation_network = tfk.Sequential([
            tfk.layers.Lambda(dec_inp_fn),
            tfk.layers.Dense(dim_h_dec, activation='relu'),
            tfk.layers.Dense(dim_h_dec, activation='relu'),
            tfk.layers.Dense(self.dim_obs)
        ])

    def kullback_leibler_x(self, x_param):
        x_mu = x_param[:, :, :self.dim_latent]
        x_var = tf.nn.softplus(x_param[:, :, self.dim_latent:])

        if not self.infer_qdot:
            delta_x_mu = x_mu[:, 1:] - x_mu[:, :-1]
            delta_x_var = x_var[:, 1:] + x_var[:, :-1]
            qx_mu = tf.concat([x_mu[:, :2], delta_x_mu[:, 1:]], 1)
            qx_var = tf.concat([x_var[:, :2], delta_x_var[:, 1:]], 1)
        else:
            qx_mu = x_mu
            qx_var = x_var

        qx = tfd.Normal(qx_mu, qx_var)

        if not self.infer_qdot:
            px_mu = tf.zeros_like(x_mu[:, :2])
            px_var = tf.ones_like(x_var[:, :2])
            px_delta_mu = tf.zeros_like(delta_x_mu[:, 1:])
            px_delta_var = tf.ones_like(delta_x_var[:, 1:]) * self.step_size
            px_mu = tf.concat([px_mu, px_delta_mu], 1)
            px_var = tf.concat([px_var, px_delta_var], 1)
        else:
            px_mu = tf.zeros_like(x_mu)
            if 'VIN' in self.dynamics_network.name:
                px_var = tf.ones_like(x_var)[:, :, :self.dynamics_network.dim_Q]
                px_var = tf.concat([px_var, self.step_size*px_var], 2)
            else:
                px_var = tf.ones_like(x_var)

        px = tfd.Normal(px_mu, px_var)
        kl_x = tfd.kl_divergence(qx, px)
        kl_x = tf.reduce_sum(kl_x, [2, 1])
        kl_x = tf.reduce_mean(kl_x)
        return kl_x

    def log_likelihood_y(self, y, y_rec):
        py = tfd.Bernoulli(logits=y_rec)
        log_py = py.log_prob(y)
        log_py = tf.reduce_sum(log_py, [2, 1])
        log_lik = tf.reduce_mean(log_py)
        return log_lik
