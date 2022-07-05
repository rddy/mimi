from __future__ import division

import numpy as np
from skopt import gp_minimize
from skopt.learning.gaussian_process.kernels import Matern
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor

from . import utils


def make_pol_sim_kernel(gp_optimizer, kernel_obses):
  '''policy similarity kernel, inspired by https://arxiv.org/abs/2106.10251'''
  X_to_A = lambda X: np.array([kernel_obses.dot(gp_optimizer.W_from_params(X[i, :])).ravel() for i in range(X.shape[0])])

  class PolSimKernel(Matern):

    def gradient_x(self, x, X_train):
      a = X_to_A(x[np.newaxis, :])[0, :]
      A_train = X_to_A(X_train)
      g = super().gradient_x(a, A_train)
      return np.array([kernel_obses.T.dot(g[i].reshape((kernel_obses.shape[0], -1))).ravel() for i in range(g.shape[0])])

    def __call__(self, X, Y=None, eval_gradient=False):
      A = X_to_A(X)
      B = None if Y is None else X_to_A(Y)
      return super().__call__(A, Y=B, eval_gradient=eval_gradient)

  return PolSimKernel()


class GP(object):

  def __init__(
    self,
    env,
    reward_model,
    use_env_obs=False,
    param_bounds=(-1., 1.),
    n_policy_params=None,
    W_from_w=None,
    kernel_obses=None
    ):
    self.env = env
    self.reward_model = reward_model
    self.use_env_obs = use_env_obs
    self.W_from_w = W_from_w

    self.n_obs_dim = self.env.n_obs_dim if use_env_obs else self.env.n_user_obs_dim
    if n_policy_params is None:
      n_policy_params = self.env.n_act_dim * self.n_obs_dim
    if type(param_bounds) == tuple and len(param_bounds) == 2:
      param_bounds = [param_bounds] * n_policy_params
    self.param_bounds = param_bounds

    self.base_estimator = None
    if kernel_obses is not None:
      kernel = make_pol_sim_kernel(self, kernel_obses)
      self.base_estimator = GaussianProcessRegressor(kernel=kernel)

  def W_from_params(self, policy_params):
    if self.W_from_w is None:
      return np.array(policy_params).reshape((self.env.n_act_dim, self.n_obs_dim))
    else:
      return self.W_from_w(policy_params)

  def policy_from_params(self, policy_params):
    W = self.W_from_params(policy_params)
    policy = lambda obs: W.dot(obs)
    if not self.use_env_obs:
      return (lambda obs: policy(self.env.extract_user_obses(obs[np.newaxis])[0]))
    return policy

  def cost_of_policy_params(self, policy_params):
    policy = self.policy_from_params(policy_params)
    rollouts = utils.rollout_policy(
      policy,
      self.env,
      self.n_steps_per_pol,
      n_eps=self.n_eps_per_pol,
      ep_kwargs=self.ep_kwargs
    )
    reward = self.reward_model(rollouts)
    self.eval_data_of_pol.append((policy_params, rollouts, reward))
    if self.verbose:
      true_reward = np.mean([sum(x[2] for x in r) for r in rollouts])
      print(reward, true_reward, policy_params)
    return -reward

  def run(
    self,
    n_pols=10,
    n_steps_per_pol=100,
    n_eps_per_pol=None,
    gp_min_kwargs={},
    ep_kwargs={},
    reward_model_train_kwargs={},
    verbose=False
    ):
    self.verbose = verbose
    self.n_steps_per_pol = n_steps_per_pol
    self.n_eps_per_pol = n_eps_per_pol
    self.ep_kwargs = ep_kwargs
    self.reward_model_train_kwargs = reward_model_train_kwargs
    self.eval_data_of_pol = []
    res = gp_minimize(
      self.cost_of_policy_params,
      self.param_bounds,
      n_calls=n_pols,
      base_estimator=self.base_estimator,
      **gp_min_kwargs
    )
    policy = self.policy_from_params(res.x)
    return policy, res
