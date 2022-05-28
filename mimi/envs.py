from __future__ import division

from collections import defaultdict
from copy import deepcopy
import time

import numpy as np
import gym

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt
import matplotlib as mpl


from . import utils


class MIMIEnv(gym.Env):

  def __init__(
    self,
    sess,
    user_model,
    max_ep_len=1000,
    render_on_step=False,
    reset_delay=0,
    step_delay=0
    ):

    self.sess = sess
    self.user_model = user_model
    self.max_ep_len = max_ep_len
    self.render_on_step = render_on_step
    self.reset_delay = reset_delay
    self.step_delay = step_delay

    self.user_obs = None
    self.prev_obs = None
    self.timestep = None
    self.goal = None
    self.pos = None
    self.viewer = None

  @property
  def n_obs_dim(self):
    return self.n_env_obs_dim + self.n_user_obs_dim


  def extract_env_obses(self, obses):
    return obses[:, :self.n_env_obs_dim]

  def extract_user_obses(self, obses):
    return obses[:, self.n_env_obs_dim:]

  def obs(self):
    return np.concatenate((self.pos, self.user_obs))

  def get_user_obs(self):
    return self.user_model(self.pos, self.goal)

  def reset(self):
    self.user_model.reset()
    self.user_obs = self.get_user_obs()
    self.prev_obs = self.obs()
    self.timestep = 0

    if self.reset_delay > 0:
      self.render()
      time.sleep(self.reset_delay)

    return self.prev_obs

  def step(self, action, r, done, info):
    self.timestep += 1
    obs = self.obs()
    prev_user_obs = self.extract_user_obses(self.prev_obs[np.newaxis])[0]
    info['goal'] = self.goal
    self.prev_obs = obs
    if self.render_on_step:
      self.render()
    if self.step_delay > 0:
      time.sleep(self.step_delay)
    return obs, r, done, info


class X2TEnv(MIMIEnv):

  def __init__(self):
    self.n_act_dim = 8
    self.n_env_obs_dim = self.n_act_dim
    self.n_user_obs_dim = 128


class ASHASwitchEnv(MIMIEnv):

  def __init__(self):
    self.n_act_dim = 7
    self.n_env_obs_dim = 14
    self.n_user_obs_dim = 128


class ASHABottleEnv(MIMIEnv):

  def __init__(self):
    self.n_act_dim = 7
    self.n_env_obs_dim = 23
    self.n_user_obs_dim = 128


class ISQLEnv(MIMIEnv):

  def __init__(self):
    self.n_act_dim = 6
    self.n_env_obs_dim = 8
    self.n_user_obs_dim = self.n_act_dim


class DeepAssistEnv(MIMIEnv):

  def __init__(self):
    self.n_act_dim = 6
    self.n_env_obs_dim = 1
    self.n_user_obs_dim = self.n_act_dim


class CursorEnv(MIMIEnv):

  def __init__(
    self,
    *args,
    goal_dist_thresh=0.15,
    speed=0.01,
    win_dims=(640,480),
    **kwargs
    ):

    super().__init__(*args, **kwargs)

    self.name = 'cursor'

    self.win_dims = win_dims
    self.speed = speed
    self.goal_dist_thresh = goal_dist_thresh
    self.n_act_dim = 2
    self.n_env_obs_dim = 2
    self.n_user_obs_dim = self.user_model.n_user_obs_dim

    self.init_pos = np.array([0.5, 0.5])
    self.prev_action = None

  def eval_metrics(self, obs, goal):
    env_obs = self.extract_env_obses(obs[np.newaxis])[0]
    goal_dist = np.linalg.norm(env_obs - goal)
    metrics = {}
    metrics['succ'] = goal_dist <= self.goal_dist_thresh
    metrics['goal_dist'] = goal_dist
    return metrics

  def reset(self):
    self.goal = np.random.normal(0, 1, 2)
    self.goal = self.goal / np.linalg.norm(self.goal) * 0.5 + self.init_pos
    self.pos = deepcopy(self.init_pos)
    self.prev_action = np.zeros(2)
    if self.viewer is not None:
      self.viewer.window.activate()
    return super().reset()

  def step(self, action, eps=1e-9):
    action = action / (eps + np.linalg.norm(action)) * self.speed
    self.prev_action = action
    if (self.pos + action >= 0).all() and (self.pos + action < 1).all():
      self.pos += action
    self.user_obs = self.get_user_obs()
    obs = self.obs()
    info = self.eval_metrics(obs, self.goal)
    done = self.timestep >= self.max_ep_len or info['succ']
    r = -1
    if info['succ']:
      r += self.max_ep_len
    return super().step(action, r, done, info)

  def render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if self.viewer is None:
      self.viewer = rendering.Viewer(width=self.win_dims[0], height=self.win_dims[1])
      self.viewer.window.activate()

    b = 0.1
    m = 0.8
    scale = lambda v: m*v+b

    X = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    X = m*X+b
    X = X * np.array(self.win_dims)[np.newaxis, :]
    box = self.viewer.draw_polyline([tuple(X[i, :]) for i in range(X.shape[0])])
    box.set_color(0, 0, 0)

    ang = -np.arctan2(*self.user_obs)
    t = rendering.Transform(rotation=ang, translation=tuple(scale(np.ones(2)*0.5)*self.win_dims))
    ctrl = self.viewer.draw_polygon([(-10, -10), (0, 40), (10, -10)])
    ctrl.set_color(0, 0, 0)
    ctrl.add_attr(t)

    t = rendering.Transform(translation=tuple(scale(self.pos)*self.win_dims))
    pos = self.viewer.draw_circle(10)
    pos.set_color(0, 0, 0)
    pos.add_attr(t)

    t = rendering.Transform(translation=tuple(scale(self.goal)*self.win_dims))
    goal = self.viewer.draw_circle(10)
    goal.set_color(0, 1, 0)
    goal.add_attr(t)

    return self.viewer.render()


class LanderEnv(MIMIEnv):

  def __init__(
    self,
    *args,
    max_ep_len=500,
    **kwargs
    ):

    super().__init__(*args, **kwargs, max_ep_len=max_ep_len)

    self.name = 'lander'

    self.env = gym.make('LunarLander-v2')

    self.n_act_dim = self.env.action_space.low.size
    self.n_user_obs_dim = self.user_model.n_user_obs_dim
    self.n_env_obs_dim = self.env.observation_space.low.size
    self.n_min_env_obs_dim = 3

  def get_user_obs(self):
    return self.user_model(self.pos, self.goal)

  def extract_min_env_obses(self, obses):
    return obses[:, [0, 1, 4]]

  def reset(self):
    self.pos = self.env.reset()
    self.goal = self.env.helipad_idx
    return super().reset()

  def step(self, action):
    self.pos, r, done, info = self.env.step(action)
    self.user_obs = self.get_user_obs()
    if self.timestep >= self.max_ep_len:
      done = True
    return super().step(action, r, done, info)

  def render(self, *args, **kwargs):
    self.env.render(*args, **kwargs)
