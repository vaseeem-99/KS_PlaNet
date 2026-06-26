# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from planet import control
from planet import tools


Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components')


def dummy(config, params):
  action_repeat = params.get('action_repeat', 1)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = lambda: control.wrappers.ActionRepeat(
      control.DummyEnv, action_repeat)
  return Task('dummy', env_ctor, max_length, state_components)


def cartpole_balance(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'balance',
      params)
  return Task('cartpole_balance', env_ctor, max_length, state_components)


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup',
      params)
  return Task('cartpole_swingup', env_ctor, max_length, state_components)


def finger_spin(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'finger', 'spin', params)
  return Task('finger_spin', env_ctor, max_length, state_components)


def cheetah_run(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cheetah', 'run', params)
  return Task('cheetah_run', env_ctor, max_length, state_components)


def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch',
      params)
  return Task('cup_catch', env_ctor, max_length, state_components)


def walker_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'height', 'orientations', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'walker', 'walk', params)
  return Task('walker_walk', env_ctor, max_length, state_components)


def reacher_easy(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'to_target']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'reacher', 'easy', params)
  return Task('reacher_easy', env_ctor, max_length, state_components)


def gym_cheetah(config, params):
  # Works with `isolate_envs: process`.
  action_repeat = params.get('action_repeat', 1)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'state']
  env_ctor = tools.bind(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'HalfCheetah-v3')
  return Task('gym_cheetah', env_ctor, max_length, state_components)


def gym_racecar(config, params):
  # Works with `isolate_envs: thread`.
  action_repeat = params.get('action_repeat', 1)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = tools.bind(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'CarRacing-v0', obs_is_image=True)
  return Task('gym_racing', env_ctor, max_length, state_components)


def _dm_control_env(
    action_repeat, max_length, domain, task, params, normalize=False,
    camera_id=None):
  if isinstance(domain, str):
    from dm_control import suite
    env = suite.load(domain, task)
  else:
    assert task is None
    env = domain()
  if camera_id is None:
    camera_id = int(params.get('camera_id', 0))
  env = control.wrappers.DeepMindWrapper(env, (64, 64), camera_id=camera_id)
  env = control.wrappers.ActionRepeat(env, action_repeat)
  if normalize:
    env = control.wrappers.NormalizeActions(env)
  env = control.wrappers.MaximumDuration(env, max_length)
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env


def _gym_env(action_repeat, min_length, max_length, name, obs_is_image=False):
  import gym
  env = gym.make(name)
  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.NormalizeActions(env)
  env = control.wrappers.MinimumDuration(env, min_length)
  env = control.wrappers.MaximumDuration(env, max_length)
  if obs_is_image:
    env = control.wrappers.ObservationDict(env, 'image')
    env = control.wrappers.ObservationToRender(env)
  else:
    env = control.wrappers.ObservationDict(env, 'state')
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env

# ============================================================================
# APPEND PlaNet tasks: ks_chaos (1D) and kolmogorov_flow (2D)
# ============================================================================

def ks_chaos(config, params):
  """KS chaos task -- 1D Kuramoto-Sivashinsky.

  DummyKSEnv satisfies PlaNet's Task interface.  No live simulation runs
  during training; data is loaded from pre-converted NPZ files.
  """
  import gym
  import gym.spaces

  max_length       = params.get('max_length', 140)
  state_components = ['reward']

  class DummyKSEnv(gym.Env):
    def __init__(self):
      self.observation_space = gym.spaces.Dict({
          'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
      self.action_space = gym.spaces.Box(
          -1.0, 1.0, (1,), dtype=np.float32)

    def reset(self):
      return {'image': np.zeros((64, 64, 3), dtype=np.uint8)}

    def step(self, action):
      return {'image': np.zeros((64, 64, 3), dtype=np.uint8)}, 0.0, True, {}

    def render(self, mode='rgb_array'):
      return np.zeros((64, 64, 3), dtype=np.uint8)

  env_ctor = lambda: control.wrappers.MaximumDuration(DummyKSEnv(), max_length)
  return Task('ks_chaos', env_ctor, max_length, state_components)


def kolmogorov_flow(config, params):
  """Kolmogorov 2D flow task.

  Default max_length=2 assumes independent (X, Y) pair episodes (single-step
  prediction).  Override via params {max_length: 51} when the converter
  reports sequential data and rebuilds trajectories of length 51.
  """
  import gym
  import gym.spaces

  max_length       = params.get('max_length', 2)
  state_components = ['reward']

  class DummyKolmogorovEnv(gym.Env):
    def __init__(self):
      self.observation_space = gym.spaces.Dict({
          'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)})
      self.action_space = gym.spaces.Box(
          -1.0, 1.0, (1,), dtype=np.float32)

    def reset(self):
      return {'image': np.zeros((64, 64, 3), dtype=np.uint8)}

    def step(self, action):
      return {'image': np.zeros((64, 64, 3), dtype=np.uint8)}, 0.0, True, {}

    def render(self, mode='rgb_array'):
      return np.zeros((64, 64, 3), dtype=np.uint8)

  env_ctor = lambda: control.wrappers.MaximumDuration(
      DummyKolmogorovEnv(), max_length)
  return Task('kolmogorov_flow', env_ctor, max_length, state_components)
