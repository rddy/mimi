# Adapted from https://github.com/rddy/ASE/blob/master/sensei/utils.py

from __future__ import division

from collections import defaultdict
from copy import deepcopy
import collections
import os
import time
import types
import functools

from IPython.core.display import display
from IPython.core.display import HTML
from matplotlib import animation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

home_dir = os.path.expanduser('~')
mimi_dir = os.path.join(home_dir, 'mimi')
data_dir = os.path.join(mimi_dir, 'data')
if not os.path.exists(data_dir):
  os.makedirs(data_dir)


tf_init_vars_cache = {}


def make_tf_session(gpu_mode=False):
  if not gpu_mode:
    kwargs = {'config': tf.ConfigProto(device_count={'GPU': 0})}
  else:
    kwargs = {}
  sess = tf.InteractiveSession(**kwargs)
  return sess


def get_tf_vars_in_scope(scope):
  return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def init_tf_vars(sess, scopes=None, use_cache=False):
  """Initialize TF variables"""
  if scopes is None:
    sess.run(tf.global_variables_initializer())
  else:
    global tf_init_vars_cache
    init_ops = []
    for scope in scopes:
      if not use_cache or scope not in tf_init_vars_cache:
        tf_init_vars_cache[scope] = tf.variables_initializer(
            get_tf_vars_in_scope(scope))
      init_ops.append(tf_init_vars_cache[scope])
    sess.run(init_ops)


def save_tf_vars(sess, scope, save_path):
  """Save TF variables"""
  saver = tf.train.Saver(
      [v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.save(sess, save_path=save_path)


def load_tf_vars(sess, scope, load_path):
  """Load TF variables"""
  saver = tf.train.Saver(
      [v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.restore(sess, load_path)


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=1,
              size=256,
              activation=None,
              output_activation=None,
              **kwargs):
  """Build MLP model"""
  out = input_placeholder
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    for _ in range(n_layers):
      out = tf.layers.dense(out, size, activation=activation, **kwargs)
    out = tf.layers.dense(out, output_size, activation=output_activation, **kwargs)
  return out


def onehot_encode(i, n):
  x = np.zeros(n)
  x[i] = 1
  return x


def onehot_decode(x):
  return np.argmax(x)


col_means = lambda x: np.nanmean(x, axis=0)
col_stderrs = lambda x: np.nanstd(
    x, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(x), axis=0))
err_bar_mins = lambda x: col_means(x) - col_stderrs(x)
err_bar_maxs = lambda x: col_means(x) + col_stderrs(x)


def make_perf_mat(perf_evals, y_key, smooth_win=10):
  n = len(perf_evals[0][y_key])
  max_len = max(len(perf_eval[y_key]) for perf_eval in perf_evals)

  def pad(lst, n):
    if len(lst) < n:
      #p = np.nan
      p = np.mean(lst[-smooth_win:])
      lst += [p] * (n - len(lst))
    return lst

  return np.array([pad(perf_eval[y_key], max_len) for perf_eval in perf_evals])


def smooth(xs, win=10):
  win = min(len(xs), win)
  psums = np.concatenate((np.zeros(1), np.cumsum(xs)))
  rtn = (psums[win:] - psums[:-win]) / win
  rtn[0] = xs[0]
  return rtn


def plot_perf_evals(perf_evals,
                    x_key,
                    y_key,
                    label='',
                    smooth_win=None,
                    color=None):
  y_mat = make_perf_mat(perf_evals, y_key)
  y_mins = err_bar_mins(y_mat)
  y_maxs = err_bar_maxs(y_mat)
  y_means = col_means(y_mat)

  if smooth_win is not None:
    y_mins = smooth(y_mins, win=smooth_win)
    y_maxs = smooth(y_maxs, win=smooth_win)
    y_means = smooth(y_means, win=smooth_win)

  xs = max([perf_eval[x_key] for perf_eval in perf_evals], key=lambda x: len(x))
  xs = xs[:len(y_means)]

  kwargs = {}
  if color is not None:
    kwargs['color'] = color

  plt.fill_between(
      xs,
      y_mins,
      y_maxs,
      where=y_maxs >= y_mins,
      interpolate=True,
      label=label,
      alpha=0.5,
      **kwargs)
  plt.plot(xs, y_means, **kwargs)


def stderr(xs):
  n = (~np.isnan(xs)).sum()
  return np.nanstd(xs) / np.sqrt(n)


def converged(val_losses, ftol, min_iters=2, eps=1e-9):
  return len(val_losses) >= max(2, min_iters) and (
      val_losses[-1] == np.nan or abs(val_losses[-1] - val_losses[-2]) /
      (eps + abs(val_losses[-2])) < ftol)


def sample_from_categorical(logits):
  noise = np.random.gumbel(loc=0, scale=1, size=logits.size)
  return (logits + noise).argmax()


def elts_at_idxes(x, idxes):
  if type(x) == list:
    return [x[i] for i in idxes]
  else:
    return x[idxes]


def sample_batch(size, data, data_keys, idxes_key, class_idxes_key=None):
  if class_idxes_key is not None and class_idxes_key not in data:
    class_idxes_key = None
  if size < len(data[idxes_key]):
    def samp(idxes, size):
      kwargs = {}
      if 'samp_probs' in data:
        p = data['samp_probs'][idxes]
        p /= p.sum()
        kwargs['p'] = p
      return np.random.choice(idxes, size, **kwargs)
    if class_idxes_key is None:
      idxes = samp(data[idxes_key], size)
    else:
      # sample class-balanced batch
      idxes = []
      idxes_of_class = data[class_idxes_key]
      n_classes = len(idxes_of_class)
      for c, idxes_of_c in idxes_of_class.items():
        k = int(np.ceil(size / n_classes))
        if k > len(idxes_of_c):
          idxes_of_c_samp = idxes_of_c
        else:
          idxes_of_c_samp = samp(idxes_of_c, k)
        idxes.extend(idxes_of_c_samp)
      if len(idxes) > size:
        np.random.shuffle(idxes)
        idxes = idxes[:size]
  else:
    idxes = data[idxes_key]
  batch = {k: elts_at_idxes(data[k], idxes) for k in data_keys}
  return batch


def split_data(data, train_frac=0.9, n_samples=None, bal_keys=None, bal_vals=None, idxes=None):
  """Train-test split
  Useful for sample_batch
  """
  if n_samples is None:
    n_samples = len(list(data.values())[0])
  if idxes is None:
    idxes = list(range(n_samples))
  np.random.shuffle(idxes)
  n_train_examples = int(train_frac * len(idxes))
  n_val_examples = len(idxes) - n_train_examples

  if bal_keys is not None:
    assert len(bal_keys) == len(bal_vals)
    for bal_key, bal_val in zip(bal_keys, bal_vals):
      def proc_idxes(idxes):
        idxes_of_val = defaultdict(list)
        for idx in idxes:
          idxes_of_val[bal_val(data[bal_key][idx])].append(idx)
        idxes_of_val = dict(idxes_of_val)
        return idxes_of_val
      idxes_of_val = proc_idxes(idxes)
      if train_frac is not None:
        train_idxes = []
        val_idxes = []
        for v, v_idxes in idxes_of_val.items():
          n_train_v_examples = n_val_examples // (len(idxes_of_val) * len(bal_keys))
          train_idxes.extend(v_idxes[n_train_v_examples:])
          val_idxes.extend(v_idxes[:n_train_v_examples])
      else:
        train_idxes = idxes
        val_idxes = idxes
      train_idxes_of_val = proc_idxes(train_idxes)
      val_idxes_of_val = proc_idxes(val_idxes)
      data['train_idxes_of_%s' % bal_key] = train_idxes_of_val
      data['val_idxes_of_%s' % bal_key] = val_idxes_of_val
  else:
    if train_frac is not None:
      train_idxes = idxes[:n_train_examples]
      val_idxes = idxes[n_train_examples:]
    else:
      train_idxes = idxes
      val_idxes = idxes

  data.update({
    'train_idxes': np.array(train_idxes),
    'val_idxes': np.array(val_idxes)
  })
  return data


def set_class_idxes_key(data, key):
  data['train_idxes_of_bal_val'] = data['train_idxes_of_%s' % key]
  data['val_idxes_of_bal_val'] = data['val_idxes_of_%s' % key]
  return data


def slice_data(data, idxes):
  sliced_data = {}
  new_idx = dict({x: i for i, x in enumerate(idxes)})
  for k, v in data.items():
    if any(k.startswith(x) for x in ['train_idxes', 'val_idxes']):
      sliced_v = [new_idx[idx] for idx in v if idx in new_idx]
    else:
      sliced_v = v[idxes]
    sliced_data[k] = sliced_v
  return sliced_data


def default_batch_acc(all_outputs, outputs):
  return np.concatenate((all_outputs, outputs), axis=0)


def batch_op(inputs, batch_size, op, acc=default_batch_acc):
  v = list(inputs.values())[0]
  n_batches = int(np.ceil(len(v) / batch_size))
  batch_idx = 0
  all_outputs = None
  for batch_idx in range(n_batches):
    batch = {k: v[batch_idx*batch_size:(batch_idx+1)*batch_size] for k, v in inputs.items()}
    outputs = op(batch)
    if all_outputs is None:
      all_outputs = outputs
    else:
      all_outputs = acc(all_outputs, outputs)
  return all_outputs


def agg_idxes_of_elt(arr):
  idxes_of_elt = defaultdict(list)
  for idx, elt in enumerate(arr):
    idxes_of_elt[elt].append(idx)
  return idxes_of_elt


def bal_weights_of_batch(batch_elts):
  batch_size = len(batch_elts)
  weights = np.ones(batch_size)
  idxes_of_elt = agg_idxes_of_elt(batch_elts)
  for elt, idxes in idxes_of_elt.items():
    weights[idxes] = 1. / len(idxes)
  return weights


def run_ep(policy, env, max_ep_len=1000, render=False, init_delay=0):
  if max_ep_len is None or max_ep_len > env.max_ep_len:
    max_ep_len = env.max_ep_len

  try:
    policy.reset()
  except:
    pass

  def render_env():
    if render:
      try:
        env.render()
      except NotImplementedError:
        pass

  obs = env.reset()
  if init_delay > 0:
    render_env()
    time.sleep(init_delay)
  done = False
  prev_obs = deepcopy(obs)
  rollout = []
  for _ in range(max_ep_len):
    if done:
      break
    action = policy(prev_obs)
    obs, r, done, info = env.step(action)
    rollout.append(deepcopy((prev_obs, action, r, obs, done, info)))
    prev_obs = deepcopy(obs)
    render_env()
  return rollout


def compute_perf_metrics(rollouts, env):
  metrics = {}
  metrics['rew'] = np.mean([sum(r for s, a, r, ns, d, i in rollout) for rollout in rollouts])
  for key in ['succ']:
    inds = [
      1 if x[-1].get(key, False) else 0
      for rollout in rollouts
      for x in rollout
    ]
    metrics[key] = np.mean(inds)
  metrics['rolloutlen'] = np.mean([len(rollout) for rollout in rollouts])
  return metrics


def play_nb_vid(frames, figsize=(10, 5), dpi=500):
  fig = plt.figure(figsize=figsize, dpi=dpi)
  plt.axis('off')
  ims = [[plt.imshow(frame, animated=True)] for frame in frames]
  plt.close()
  anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
  display(HTML(anim.to_html5_video()))
  return anim


def viz_cursor_rollout(rollout):
  traj = np.array([x[0][:2] for x in rollout])
  goal = rollout[0][-1]['goal']
  plt.scatter(traj[:, 0], traj[:, 1], c=list(range(len(traj))))
  plt.scatter(goal[0], goal[1], color='green')
  plt.xlim([-0.1, 1.1])
  plt.ylim([-0.1, 1.1])
  plt.show()


def rotate_vec(vec, ang):
  rot_mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
  return rot_mat.dot(vec)


def prep_env_for_human_user(env, user_model):
  assert env.name == 'cursor'
  env.reset()
  env.render()
  env.viewer.window.on_mouse_motion = user_model.on_mouse_motion


def format_rollouts(rollouts, env):
  data = {
    'obses': [],
    'actions': [],
    'next_obses': [],
    'rewards': []
  }
  for rollout_idx, rollout in enumerate(rollouts):
    for i, x in enumerate(rollout):
      next_obs_idxes = [i]
      for j in next_obs_idxes:
        data['obses'].append(x[0])
        data['actions'].append(x[1])
        data['next_obses'].append(rollout[j][3])
  data = {k: np.array(v) for k, v in data.items()}
  data['env_obses'] = env.extract_env_obses(data['obses'])
  data['next_env_obses'] = env.extract_env_obses(data['next_obses'])
  data['user_obses'] = env.extract_user_obses(data['obses'])
  try:
    data['min_env_obses'] = env.extract_min_env_obses(data['obses'])
    data['min_next_env_obses'] = env.extract_min_env_obses(data['next_obses'])
  except:
    pass
  return data


def rollout_policy(policy, env, n_steps, n_eps=None, ep_kwargs={}):
  t = 0
  rollouts = []
  while (n_eps is None and t < n_steps) or (n_eps is not None and len(rollouts) < n_eps):
    rollout = run_ep(policy, env, **ep_kwargs)
    rollouts.append(rollout)
    t += len(rollout)
  return rollouts


class ObsNormalizer(object):

  def __init__(self, obs_shape):
    self.n = 0
    self.mean = np.zeros(obs_shape)
    self.sq = np.ones(obs_shape)
    self.inv_std = np.ones(obs_shape)

  def update(self, obses):
    samp_mean = np.mean(obses, axis=0)
    samp_sq = np.mean(obses**2, axis=0)
    samp_n = len(obses)
    w = self.n / (self.n + samp_n)
    self.mean = self.mean * w + samp_mean * (1 - w)
    self.sq = self.sq * w + samp_sq * (1 - w)
    self.n += samp_n

    var = self.sq - self.mean
    var = np.maximum(1e-8, var)
    self.inv_std = 1 / np.sqrt(var)

  def __call__(self, obs):
    return self.inv_std * (obs - self.mean)


def compute_rews_of_rollouts(rollouts_of_pol, reward_models, verbose=True):
  rewards_of_pol = []
  for pol_idx, rollouts in enumerate(rollouts_of_pol):
    rewards = [reward_model(rollouts) for reward_model in reward_models]
    rewards_of_pol.append(rewards)
    if verbose:
      print(pol_idx, rewards)
  return np.array(rewards_of_pol)
