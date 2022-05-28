from __future__ import division

import numpy as np

from . import utils
from .models import MIModel


class MIRewardModel(object):

  def __init__(
    self,
    env,
    mi_model_init_args,
    mi_model_init_kwargs,
    mi_model_train_kwargs,
    use_min_env_obs=False,
    use_next_env_obs=True
    ):
    self.env = env
    self.mi_model_train_kwargs = mi_model_train_kwargs

    input_vars = ['user_obses']
    ss = ['']
    if use_next_env_obs:
      ss.append('next_')
    for s in ss:
      input_vars.append('%s%senv_obses' % ('min_' if use_min_env_obs else '', s))

    self.model = MIModel(
      *mi_model_init_args,
      shuffle_var='user_obses',
      input_vars=input_vars,
      **mi_model_init_kwargs
    )

  def format_rollouts(self, rollouts):
    data = utils.format_rollouts(rollouts, self.env)
    return utils.split_data(data, train_frac=0.9)

  def train(self, data):
    self.model.train(data, **self.mi_model_train_kwargs)

  def evaluate(self, data):
    val_data = utils.slice_data(data, data['val_idxes'])
    return self.model.compute_mi(val_data)

  def __call__(self, rollouts):
    data = self.format_rollouts(rollouts)
    self.train(data)
    return self.evaluate(data)
