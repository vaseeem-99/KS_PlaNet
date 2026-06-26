"""Evaluate a trained PlaNet RSSM on PDE episode files.

This restores the actual TensorFlow checkpoint and writes an ``.npz`` archive
containing a true open-loop rollout, the matching ground truth, and RSSM
posterior/open-loop latent states.  It is deliberately separate from training
so evaluation cannot change a completed checkpoint.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import numpy as np
import tensorflow as tf

from planet import tools
from planet.scripts import configs


def _checkpoint(logdir):
  """Find the newest checkpoint in a PlaNet Experiment log directory."""
  candidates = [logdir]
  candidates += sorted(glob.glob(os.path.join(logdir, '*')), reverse=True)
  for directory in candidates:
    if os.path.isdir(directory):
      checkpoint = tf.train.latest_checkpoint(directory)
      if checkpoint:
        return checkpoint
  raise ValueError('No TensorFlow checkpoint found under {}'.format(logdir))


def _config(name, logdir, context):
  params = tools.AttrDict(_unlocked=True)
  params.logdir = logdir
  params.open_loop_context = context
  return getattr(configs, name)(tools.AttrDict(), params)


def _build_graph(config):
  """Recreate the trained encoder, RSSM, and image decoder scopes exactly."""
  image = tf.placeholder(tf.uint8, [1, None, 64, 64, 3], name='image')
  action = tf.placeholder(tf.float32, [1, None, 1], name='action')
  data = {'image': config.preprocess_fn(image), 'action': action}

  cell = config.cell()
  encoder = tf.make_template('encoder', config.encoder, create_scope_now_=True)
  image_head = tf.make_template(
      'head_image', config.heads.image, data_shape=[64, 64, 3],
      create_scope_now_=True)

  # PlaNet creates decoder variables before the encoder/RSSM variables.
  dummy = cell.features_from_state(cell.zero_state(1, tf.float32))
  image_head(dummy)
  embedded = encoder(data)
  posterior = tools.unroll.closed_loop(
      cell, embedded, data['action'], config.debug)[1]
  open_state = tools.unroll.open_loop(
      cell, embedded, data['action'], config.open_loop_context, config.debug)
  prediction = config.postprocess_fn(
      image_head(cell.features_from_state(open_state)).mean())
  return image, action, prediction, posterior, open_state


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', required=True)
  parser.add_argument('--episode-dir', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--config', default='ks_config',
                      choices=['ks_config', 'kolmogorov_config'])
  parser.add_argument('--context', type=int, default=None)
  parser.add_argument('--episode-index', type=int, default=0)
  parser.add_argument('--horizon', type=int, default=0,
                      help='0 evaluates the complete episode.')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  context = args.context
  if context is None:
    context = 20 if args.config == 'ks_config' else 1
  files = sorted(glob.glob(os.path.join(args.episode_dir, 'episode_*.npz')))
  if args.episode_index >= len(files):
    raise ValueError('Episode index {} is unavailable; found {} files.'.format(
        args.episode_index, len(files)))
  source = files[args.episode_index]
  episode = np.load(source)
  truth = episode['image']
  if args.horizon:
    truth = truth[:args.horizon]
  if len(truth) <= context:
    raise ValueError('Episode length must exceed context ({} <= {}).'.format(
        len(truth), context))
  action_values = episode['action'][:len(truth)]

  tf.reset_default_graph()
  tf.set_random_seed(args.seed)
  # ``training.utility.train`` constructs the saved model inside this scope.
  # Restoring at the root scope would look for ``encoder/...`` instead of the
  # checkpoint's ``graph/encoder/...`` variables.
  with tf.variable_scope('graph', use_resource=True):
    image, action, prediction, posterior, open_state = _build_graph(
        _config(args.config, args.logdir, context))
  checkpoint = _checkpoint(args.logdir)
  saver = tf.train.Saver()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as session:
    saver.restore(session, checkpoint)
    fetches = [prediction, posterior['mean'], posterior['belief'],
               open_state['mean'], open_state['belief']]
    values = session.run(fetches, {image: truth[None], action: action_values[None]})

  directory = os.path.dirname(args.output)
  if directory:
    os.makedirs(directory, exist_ok=True)
  np.savez_compressed(
      args.output,
      prediction=np.clip(values[0][0], 0.0, 1.0).astype(np.float32),
      truth=truth.astype(np.float32) / 255.0,
      posterior_mean=values[1][0].astype(np.float32),
      posterior_belief=values[2][0].astype(np.float32),
      open_loop_mean=values[3][0].astype(np.float32),
      open_loop_belief=values[4][0].astype(np.float32),
      context=np.int32(context),
      episode_file=np.array(os.path.basename(source)))
  print('Restored checkpoint: {}'.format(checkpoint))
  print('Evaluated episode:   {}'.format(source))
  print('Saved rollout:       {}'.format(args.output))


if __name__ == '__main__':
  main()
