from __future__ import division

import time

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import mediapipe as mp
from PIL import Image

from . import models
from . import utils


class User(object):

  def __init__(self, n_user_obs_dim=2):
    self.n_user_obs_dim = n_user_obs_dim

  def reset(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)


class HumanMouseUser(User):

  def __init__(self, win_dims=(640,480), step_delay=0):
    super().__init__(n_user_obs_dim=2)

    self.step_delay = step_delay
    self.win_dims = np.array(win_dims)

    self.win_ctr = np.ones(2) * 0.5
    self.action = None

  def reset(self, *args, **kwargs):
    self.action = np.zeros(2)

  def call(self, *args, **kwargs):
    if self.step_delay > 0:
      time.sleep(self.step_delay)
    return self.action

  def on_mouse_motion(self, x, y, dx, dy):
    self.action = np.array([x, y])
    self.action = self.action / self.win_dims
    self.action -= self.win_ctr
    eps = 1e-9
    self.action /= np.linalg.norm(self.action) + eps


class HumanHandUser(User):

  def __init__(self, mode=False, max_hands=2, detection_conn=0.5, track_conn=0.5):
    self.n_user_obs_dim = 4#21*2
    self.prev_user_obs = np.zeros(self.n_user_obs_dim)
    self.cam = cv2.VideoCapture(0)
    self.mp_hands = mp.solutions.hands
    self.mp_draw = mp.solutions.drawing_utils
    self.hands = self.mp_hands.Hands(mode, max_hands, detection_conn, track_conn)

    self.max_dist = 1e-9
    self.min_dist = 1e9

    self.prev_user_obs = None
    self.prev_img = None

  def _featurize_landmarks(self, landmarks):
    HL = self.mp_hands.HandLandmark
    tip_idxes = [HL.THUMB_TIP, HL.INDEX_FINGER_TIP, HL.MIDDLE_FINGER_TIP, HL.RING_FINGER_TIP, HL.PINKY_TIP]
    tips = [np.array(landmarks[tip_idx]) for tip_idx in tip_idxes]
    dists = [np.linalg.norm(a-tips[0]) for a in tips[1:]]
    dists = np.array(dists)
    self.max_dist = max(self.max_dist, np.max(dists))
    self.min_dist = min(self.min_dist, np.min(dists))
    dists = (dists - self.min_dist) / (self.max_dist - self.min_dist)
    return dists * 2 - 1

  def _process_hands(self, img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return self.hands.process(rgb_img)

  def _get_landmarks(self, img):
    results = self._process_hands(img)
    landmarks = []
    if results.multi_hand_landmarks is not None:
      hand = results.multi_hand_landmarks[0]
      for lm in hand.landmark:
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        landmarks.append(np.array([cx, cy]))
    return landmarks

  def draw(self, img):
    results = self._process_hands(img)
    bkg_img = np.ones(img.shape)
    if results.multi_hand_landmarks is not None:
      hand = results.multi_hand_landmarks[0]
      self.mp_draw.draw_landmarks(bkg_img, hand, self.mp_hands.HAND_CONNECTIONS)
    return bkg_img

  def _featurize_img(self, img):
    landmarks = self._get_landmarks(img)
    return self._featurize_landmarks(landmarks) if len(landmarks) > 0 else None

  def _downscale_img(self, img, downscale=2):
    im = Image.fromarray(img)
    w, h = im.size
    resized_im = im.resize((w//downscale, h//downscale), Image.LANCZOS)
    return np.array(resized_im)

  def call(self, *args, **kwargs):
    _, img = self.cam.read()
    img_feats = self._featurize_img(img)
    if img_feats is not None:
      self.prev_img = self._downscale_img(img)
      self.prev_user_obs = img_feats
    return self.prev_user_obs


class SimUser(User):

  def __init__(
    self,
    *args,
    inv_dyn_model_init_args,
    inv_dyn_model_init_kwargs={},
    inv_dyn_ens_init_kwargs={},
    noise_std=1,
    **kwargs):
    super().__init__(*args, **kwargs)

    self.noise_std = noise_std

    self.inv_dyn_model = models.EnsembleInvDynModel(
      inv_dyn_model_init_args,
      model_init_kwargs=inv_dyn_model_init_kwargs,
      **inv_dyn_ens_init_kwargs
    )

  def reset(self, *args, **kwargs):
    self.prev_obs = None
    self.prev_user_obs = None

  def _desired_next_obs(self, obs, goal):
    raise NotImplementedError

  def _normalize_user_obs(self, user_obs):
    return user_obs

  def call(self, obs, goal):
    desired_next_obs = self._desired_next_obs(obs, goal)
    user_obs = self.inv_dyn_model(obs[np.newaxis, :], desired_next_obs[np.newaxis, :])[0]
    user_obs += np.random.normal(0, self.noise_std, user_obs.shape)
    user_obs = self._normalize_user_obs(user_obs)
    if self.prev_obs is not None:
      self.inv_dyn_model.update(self.prev_obs, self.prev_user_obs, obs)
    self.prev_obs = obs
    self.prev_user_obs = user_obs
    return user_obs


class SimCursorUser(SimUser):

  def __init__(self, speed, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.speed = speed
    self.obs_low = -1
    self.obs_high = 1

  def _normalize_speed(self, v):
    return v / np.linalg.norm(v) * self.speed

  def _desired_next_obs(self, obs, goal):
    return obs + self._normalize_speed(goal - obs)

  def _normalize_user_obs(self, user_obs, eps=1e-9):
    return user_obs / np.linalg.norm(user_obs) + eps


class SimMNISTUser(SimCursorUser):

  def __init__(self, goal_latents, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.goal_latents = goal_latents

  def _desired_next_obs(self, obs, goal):
    return super()._desired_next_obs(obs, self.goal_latents[goal])
