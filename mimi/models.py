# Adapted from https://github.com/rddy/ASE/blob/master/sensei/models.py

from __future__ import division

from queue import Queue
import pickle
import uuid
import os

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from . import utils

import sys
sys.path.append(utils.dvae_dir)
import disvae.utils.modelIO


class TFModel(object):

  def __init__(
    self,
    sess,
    scope_file=None,
    tf_file=None,
    scope=None,
    *args,
    **kwargs
    ):

    if scope is None:
      if scope_file is not None and os.path.exists(scope_file):
        with open(scope_file, 'rb') as f:
          scope = pickle.load(f)
      else:
        scope = str(uuid.uuid4())

    self.sess = sess
    self.tf_file = tf_file
    self.scope_file = scope_file
    self.scope = scope

    self.loss = None
    self.grads = None
    self.is_trained = False
    self.is_initialized = False

  def save(self):
    if self.scope_file is None:
      return

    with open(self.scope_file, 'wb') as f:
      pickle.dump(self.scope, f, pickle.HIGHEST_PROTOCOL)

    utils.save_tf_vars(self.sess, self.scope, self.tf_file)

  def load(self):
    if self.scope_file is None:
      return

    with open(self.scope_file, 'rb') as f:
      self.scope = pickle.load(f)

    self.init_tf_vars()
    utils.load_tf_vars(self.sess, self.scope, self.tf_file)
    self.is_initialized = True

  def init_tf_vars(self):
    utils.init_tf_vars(self.sess, [self.scope])
    self.is_initialized = True

  def compute_batch_loss(self, feed_dict, update=True):
    args = [self.loss]
    if update:
      args.append(self.update_op)
    loss_eval = self.sess.run(args, feed_dict=feed_dict)[0]
    return loss_eval

  def train(
    self,
    data,
    iterations=100000,
    ftol=1e-4,
    batch_size=32,
    learning_rate=1e-3,
    beta1=0.9,
    val_update_freq=100,
    verbose=False,
    show_plots=None,
    warm_start=False
    ):

    if self.loss is None:
      return

    if show_plots is None:
      show_plots = verbose

    var_list = utils.get_tf_vars_in_scope(self.scope)
    opt_scope = str(uuid.uuid4())
    with tf.variable_scope(opt_scope, reuse=tf.AUTO_REUSE):
      self.update_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(self.loss, var_list=var_list)

    init_scopes = [opt_scope]
    if not (warm_start and self.is_initialized):
      init_scopes.append(self.scope)
    utils.init_tf_vars(self.sess, init_scopes)

    val_losses = []
    val_batch = utils.sample_batch(
      size=len(data['val_idxes']),
      data=data,
      data_keys=self.data_keys,
      idxes_key='val_idxes'
    )
    formatted_val_batch = self.format_batch(val_batch)

    if verbose:
      print('-----')
      print('iters total_iters train_loss val_loss')

    train_losses = []
    for t in range(iterations):
      batch = utils.sample_batch(
        size=batch_size,
        data=data,
        data_keys=self.data_keys,
        idxes_key='train_idxes',
        class_idxes_key='train_idxes_of_bal_val'
      )

      formatted_batch = self.format_batch(batch)
      train_loss = self.compute_batch_loss(formatted_batch, update=True)
      train_losses.append(train_loss)

      min_val_loss = None
      if val_update_freq is not None and t % val_update_freq == 0:
        val_loss = self.compute_batch_loss(formatted_val_batch, update=False)
        train_losses = []

        if verbose:
          print('%d %d %f %f' % (t, iterations, train_loss, val_loss))

        val_losses.append(val_loss)

        if ftol is not None and utils.converged(val_losses, ftol):
          break

    if verbose:
      print('-----\n')

    if show_plots:
      plt.xlabel('Gradient Steps')
      plt.ylabel('Validation Loss')
      grad_steps = np.arange(0, len(val_losses), 1) * val_update_freq
      plt.plot(grad_steps, val_losses)
      plt.show()

    self.is_trained = True


class BaseModel(TFModel):

  def __init__(
    self,
    *args,
    n_env_obs_dim,
    n_user_obs_dim,
    n_act_dim,
    input_vars,
    n_layers=2,
    layer_size=32,
    output_var=None,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    self.n_layers = n_layers
    self.layer_size = layer_size
    self.input_vars = input_vars
    self.output_var = output_var
    self.n_user_obs_dim = n_user_obs_dim
    self.n_act_dim = n_act_dim
    if output_var == 'user_obses':
      self.n_output_dim = n_user_obs_dim
    elif output_var in ['next_env_obses', 'env_obses']:
      self.n_output_dim = n_env_obs_dim
    elif output_var == 'actions':
      self.n_output_dim = n_act_dim
    else:
      self.n_output_dim = 1

    self.data_keys = list(self.input_vars)
    if self.output_var is not None:
      self.data_keys.append(self.output_var)

    self.s_ph = tf.placeholder(tf.float32, [None, n_env_obs_dim])
    self.s_next_ph = tf.placeholder(tf.float32, [None, n_env_obs_dim])
    self.x_ph = tf.placeholder(tf.float32, [None, n_user_obs_dim])
    self.a_ph = tf.placeholder(tf.float32, [None, n_act_dim])
    self.ph_of_var = {
      'user_obses': self.x_ph,
      'env_obses': self.s_ph,
      'next_env_obses': self.s_next_ph,
      'min_env_obses': self.s_ph,
      'min_next_env_obses': self.s_next_ph,
      'actions': self.a_ph
    }
    self.input_phs = [self.ph_of_var[var] for var in self.input_vars]
    if self.output_var is not None:
      self.output_ph = self.ph_of_var[self.output_var]

  def format_batch(self, batch):
    feed_dict = {self.ph_of_var[k]: batch[k] for k in self.data_keys if k in batch and k in self.ph_of_var}
    return feed_dict

  def build_model(
    self,
    *args,
    scope=None,
    out_dim=None,
    n_layers=None
    ):
    if n_layers is None:
      n_layers = self.n_layers
    if scope is None:
      scope = self.scope
    if out_dim is None:
      out_dim = self.n_output_dim
    cat_in = tf.concat(args, axis=1)
    return utils.build_mlp(
      cat_in,
      out_dim,
      scope,
      n_layers=self.n_layers,
      size=self.layer_size,
      activation=tf.nn.relu,
      output_activation=None
    )


class MIModel(BaseModel):

  def __init__(
    self,
    *args,
    shuffle_var,
    n_mine_samp=32,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    left = self.build_model(*self.input_phs)
    shuffled_stats = []
    shuffle_var_idx = self.input_vars.index(shuffle_var)
    for _ in range(n_mine_samp):
      self.input_phs[shuffle_var_idx] = tf.random.shuffle(self.input_phs[shuffle_var_idx])
      shuffled_out = self.build_model(*self.input_phs)
      shuffled_stats.append(shuffled_out)
    shuffled_stats = tf.stack(shuffled_stats, axis=1)
    a_phs = [ph for ph, var in zip(self.input_phs, self.input_vars) if var != shuffle_var]
    a_scope = self.scope + '/a'
    a = self.build_model(*a_phs, scope=a_scope)
    right = tf.reduce_mean(tf.exp(shuffled_stats), axis=1) / tf.exp(a) + a - 1
    self.mi_lb = left - right
    self.loss = -tf.reduce_mean(self.mi_lb)

  def compute_mi(self, batch):
    feed_dict = self.format_batch(batch)
    loss = self.compute_batch_loss(feed_dict, update=False)
    return -loss

  def compute_fine_mi(self, batch):
    feed_dict = self.format_batch(batch)
    return self.sess.run(self.mi_lb, feed_dict=feed_dict)


class InvDynModel(BaseModel):

  def __init__(self, *args, **kwargs):
    super().__init__(
      *args,
      input_vars=['env_obses', 'next_env_obses'],
      output_var='user_obses',
      **kwargs
    )

    self.x_pred = self.build_model(*self.input_phs)
    self.loss = tf.reduce_mean((self.x_pred-self.output_ph)**2)

  def __call__(self, env_obses, next_env_obses):
    batch = {
      'env_obses': env_obses,
      'next_env_obses': next_env_obses
    }
    feed_dict = self.format_batch(batch)
    pred_user_obses = self.sess.run(self.x_pred, feed_dict=feed_dict)
    return pred_user_obses


class EnsembleInvDynModel(object):

  def __init__(
    self,
    model_init_args,
    model_init_kwargs={},
    intuitive_models=[],
    n_models=2,
    buffer_size=0,
    temp=1
    ):
    self.temp = temp
    self.models = intuitive_models
    for _ in range(n_models-len(intuitive_models)):
      model = InvDynModel(*model_init_args, **model_init_kwargs)
      model.init_tf_vars()
      self.models.append(model)
    self.logits = [Queue(maxsize=buffer_size) for _ in range(n_models)]

  def batch_call(self, env_obses, next_env_obses):
    batch = {
      'env_obses': env_obses,
      'next_env_obses': next_env_obses
    }
    feed_dict = {}
    for model in self.models:
      feed_dict.update(model.format_batch(batch))
    x_preds = [model.x_pred for model in self.models]
    pred_user_obses = self.models[0].sess.run(x_preds, feed_dict=feed_dict)
    return np.array(pred_user_obses)

  def update(self, env_obs, user_obs, next_env_obs):
    pred_user_obses = self.batch_call(env_obs[np.newaxis, :], next_env_obs[np.newaxis, :])[:, 0, :]
    logits = -np.mean((pred_user_obses-user_obs)**2, axis=1)
    for model_idx, q in enumerate(self.logits):
      if q.full():
        q.get()
      q.put(logits[model_idx])

  def __call__(self, *args, **kwargs):
    logits = np.array([np.mean(q.queue) for q in self.logits])
    logits *= self.temp
    model_idx = utils.sample_from_categorical(logits)
    return self.models[model_idx](*args, **kwargs)


class BTCVAEEncoder(object):

  def __init__(self, dataset):
    model_dir = os.path.join(utils.dvae_dir, 'results', 'btcvae_%s' % dataset)
    self.model = disvae.utils.modelIO.load_model(model_dir)
    self.model.eval()
    self.latent_dim = self.model.latent_dim
    self.device = next(self.model.parameters()).device

  def encode(self, images, batch_size=32):
    data = utils.front_img_ch(images)
    def op(batch):
      torched_batch = utils.numpy_to_torch(batch).to(self.device)
      batch_latents, _ = self.model.encoder(torched_batch)
      return utils.torch_to_numpy(batch_latents)
    return utils.batch_op(data, batch_size, op)

  def decode(self, latents, batch_size=32):
    def op(batch):
      torched_latents = utils.numpy_to_torch(batch).to(self.device)
      batch_images = self.model.decoder(torched_latents)
      return utils.torch_to_numpy(batch_images)
    images = utils.batch_op(latents, batch_size, op)
    images = utils.back_img_ch(images)
    return images
