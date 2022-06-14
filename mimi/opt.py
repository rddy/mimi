from __future__ import division

import numpy as np
from skopt import gp_minimize

from . import utils


class GP(object):

  def __init__(
    self,
    env,
    reward_model,
    use_env_obs=False,
    use_bias=False,
    param_bounds=(-1., 1.),
    n_policy_params=None,
    W_from_w=None
    ):
    self.env = env
    self.reward_model = reward_model
    self.use_env_obs = use_env_obs
    self.use_bias = use_bias
    self.W_from_w = W_from_w

    self.n_obs_dim = self.env.n_obs_dim if use_env_obs else self.env.n_user_obs_dim
    if n_policy_params is None:
      n_policy_params = self.env.n_act_dim * self.n_obs_dim
      if self.use_bias:
        n_policy_params += self.env.n_act_dim
    if type(param_bounds) == tuple and len(param_bounds) == 2:
      param_bounds = [param_bounds] * n_policy_params
    self.param_bounds = param_bounds

  def policy_from_params(self, policy_params):
    if self.use_bias:
      w = policy_params[:-self.env.n_act_dim]
      b = policy_params[-self.env.n_act_dim:]
    else:
      w = policy_params
      b = 0
    if self.W_from_w is None:
      W = np.array(w).reshape((self.env.n_act_dim, self.n_obs_dim))
    else:
      W = self.W_from_w(w)
    b = np.array(b)
    policy = lambda obs: W.dot(obs) + b
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
      **gp_min_kwargs
    )
    policy = self.policy_from_params(res.x)
    return policy, res
