from __future__ import division

from collections import defaultdict
from copy import deepcopy
import time
from queue import Queue

import numpy as np
import gym

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.transform


from . import utils
from . import models


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

    self.n_user_obs_dim = self.user_model.n_user_obs_dim

    self.user_obs = None
    self.prev_obs = None
    self.timestep = None
    self.goal = None
    self.pos = None
    self.viewer = None
    self.prev_step = None

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
    self.user_model.reset(self.pos)
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
    info['goal'] = self.goal
    self.prev_step = (self.prev_obs, action, r, obs, done, info)
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
    min_pos=0,
    max_pos=1,
    **kwargs
    ):

    super().__init__(*args, **kwargs)

    self.name = 'cursor'

    self.win_dims = win_dims
    self.speed = speed
    self.goal_dist_thresh = goal_dist_thresh
    self.min_pos = min_pos
    self.max_pos = max_pos

    self.n_env_obs_dim = 2
    self.n_act_dim = self.n_env_obs_dim

    self.init_pos = np.ones(self.n_env_obs_dim) * ((self.max_pos - self.min_pos) / 2 + self.min_pos)

  def _goal_dist(self, obs):
    return np.linalg.norm(obs - self.goal)

  def eval_metrics(self, obs):
    env_obs = self.extract_env_obses(obs[np.newaxis])[0]
    goal_dist = self._goal_dist(env_obs)
    metrics = {}
    metrics['goal_dist'] = goal_dist
    metrics['succ'] = goal_dist <= self.goal_dist_thresh
    return metrics

  def _sample_goal(self):
    goal = np.random.normal(0, 1, self.init_pos.shape)
    goal = goal / np.linalg.norm(goal) * (self.max_pos - self.min_pos) / 2 + self.init_pos
    return goal

  def reset(self):
    self.goal = self._sample_goal()
    self.pos = deepcopy(self.init_pos)
    return super().reset()

  def _update_pos(self, pos, action, eps=1e-9):
    action = action / (eps + np.linalg.norm(action)) * self.speed
    if (pos + action >= self.min_pos).all() and (pos + action < self.max_pos).all():
      return pos + action
    else:
      return pos

  def step(self, action):
    self.pos = self._update_pos(self.pos, action)
    self.user_obs = self.get_user_obs()
    obs = self.obs()
    info = self.eval_metrics(obs)
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

    b = 0.1
    m = 0.8
    scale = lambda v: m*v+b

    X = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    X = m*X+b
    X = X * np.array(self.win_dims)[np.newaxis, :]
    box = self.viewer.draw_polyline([tuple(X[i, :]) for i in range(X.shape[0])])
    box.set_color(0, 0, 0)

    if self.user_obs.size == 2:
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


class RewardModelCursorEnv(CursorEnv):

  def __init__(
    self,
    *args,
    update_freq=10,
    buffer_size=10,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    self.observation_space = gym.spaces.Box(
      np.concatenate((np.ones(self.n_env_obs_dim) * self.min_pos, np.ones(self.n_user_obs_dim) * self.user_model.obs_low)),
      np.concatenate((np.ones(self.n_env_obs_dim) * self.max_pos, np.ones(self.n_user_obs_dim) * self.user_model.obs_high))
    )
    self.action_space = gym.spaces.Box(
      -np.ones(self.n_env_obs_dim),
      np.ones(self.n_env_obs_dim)
    )

    self.update_freq = update_freq
    self.rollouts = Queue(maxsize=buffer_size)
    self.n_eps_since_update = 0
    self.rollout = None

  def set_reward_model(self, reward_model):
    self.reward_model = reward_model
    self.reward_model.model.init_tf_vars()

  def reset(self, *args, **kwargs):
    rtn = super().reset(*args, **kwargs)
    if self.rollout is not None:
      if self.rollouts.full():
        self.rollouts.get()
      self.rollouts.put(self.rollout)
    self.rollout = []
    self.n_eps_since_update += 1
    if self.n_eps_since_update % self.update_freq == 0:
      data = self.reward_model.format_rollouts(self.rollouts.queue)
      self.reward_model.train(data)
      self.n_eps_since_update = 0
    return rtn

  def step(self, *args, **kwargs):
    obs, r, done, info = super().step(*args, **kwargs)
    self.rollout.append(self.prev_step)
    r = self.reward_model.compute_step_rewards(list(self.rollouts.queue) + [self.rollout])[-1][0]
    return obs, r, done, info


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


class LatentExplorationEnv(CursorEnv):

  def __init__(self, model, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = model
    self.n_act_dim = self.model.latent_dim
    self.n_env_obs_dim = self.model.latent_dim
    self.init_pos = np.zeros(self.model.latent_dim)
  '''
  def _update_pos(self, pos, action):
    pos = action
    pos = np.minimum(self.max_pos, pos)
    pos = np.maximum(self.min_pos, pos)
    return pos
  ''' # DEBUG
  def _visualize_pos(self, pos):
    return self.model.decode(pos[np.newaxis, :])[0]

  def _visualize(self):
    return self._visualize_pos(self.pos)

  def render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if self.viewer is None:
      self.viewer = rendering.SimpleImageViewer()

    img = self._visualize()
    self.viewer.imshow(img)


class MNISTEnv(LatentExplorationEnv):

  def __init__(self, *args, **kwargs):
    model = models.BTCVAEEncoder('mnist')
    super().__init__(model, *args, **kwargs)

    self.name = 'mnist'

    self.clf = utils.make_mnist_clf()

  def _sample_goal(self):
    goal = np.random.choice(list(range(10)))
    return goal

  def _visualize(self, win_size=512):
    img = super()._visualize()
    img = skimage.transform.resize(img, (win_size, win_size))
    img = (img * 255).astype('uint8')
    img = np.concatenate([img] * 3, axis=2)
    return img

  def _classify(self, pos):
    return self.clf.predict_proba(self._visualize_pos(pos).reshape(1, -1))[0]

  def _goal_dist(self, pos):
    probs = self._classify(pos)
    non_goal_probs = np.concatenate((probs[:self.goal], probs[self.goal+1:]))
    goal_prob = probs[self.goal]
    next_prob = np.max(non_goal_probs)
    dist = next_prob - goal_prob
    return dist
