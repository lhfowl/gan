# coding=utf-8
# Copyright 2020 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Trains a Self-Attention GAN using Estimators.

This code is based on the original code from https://arxiv.org/abs/1805.08318.

"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import logging
import os

flags.DEFINE_string('model_dir', '/tmp/tfgan_logdir/sagan-estimator',
                    'Optional location to save model. If `None`, use a '
                    'default provided by tf.Estimator.')


# ML Hparams.
flags.DEFINE_integer(
    'train_batch_size', 32,
    'The number of images in each train batch. From go/tpu-pitfalls: "The '
    'batch size of any model should always be at least 64 (8 per TPU core), '
    'since the TPU always pads the tensors to this size. The ideal batch size '
    'when training on the TPU is 1024 (128 per TPU core), since this '
    'eliminates inefficiencies related to memory transfer and padding."')
flags.DEFINE_integer('z_dim', 128,
                     'Dimensions of the generator noise vector.')
flags.DEFINE_integer('gf_dim', 64, 'Dimensionality of gf. [64]')
flags.DEFINE_integer('df_dim', 64, 'Dimensionality of df. [64]')
flags.DEFINE_float('generator_lr', 0.0001, 'The generator learning rate.')
flags.DEFINE_float('discriminator_lr', 0.0004,
                   'The discriminator learning rate.')
flags.DEFINE_float('beta1', 0.0, 'Momentum term of adam. [0.0]')


# ML Infra.
flags.DEFINE_enum(
    'mode', None, ['train', 'continuous_eval', 'train_and_eval', 'intra_fid_eval'],
    'Mode to run in. `train` just trains the model. `continuous_eval` '
    'continuously looks for new checkpoints and computes eval metrics and '
    'writes sample outputs to disk. `train_and_eval` does both. '
    'If not set, will deduce mode from the TF_CONFIG environment variable.')
flags.DEFINE_integer('max_number_of_steps', 50000,
                     'The maximum number of train steps.')
flags.DEFINE_integer(
    'train_steps_per_eval', 1000,
    'Number of train steps before writing some sample images.')
flags.DEFINE_integer('num_eval_steps', 32, 'The number of evaluation steps.')
flags.DEFINE_integer('eval_batch_size', 32,
                     'The number of images in each eval batch.')
flags.DEFINE_integer('predict_batch_size', 80,
                     'The number of images in each predict batch.')

# Debugging.
flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or CPU.')
flags.DEFINE_bool('eval_on_tpu', False, 'Whether eval is run on TPU.')
flags.DEFINE_integer(
    'continuous_eval_timeout_secs', None,
    'None, or number of seconds to wait for a checkpoint '
    'before stopping.')

# TPU params.
flags.DEFINE_bool(
    'use_tpu_estimator', False,
    'Whether to use TPUGANEstimator or GANEstimator. This is useful if, for '
    'instance, we want to run the eval job on GPU.')
flags.DEFINE_string('tpu', None, 'A string corresponding to the TPU to use.')
flags.DEFINE_string('gcp_project', None,
                    'Name of the GCP project containing Cloud TPUs.')
flags.DEFINE_string('tpu_zone', None, 'Zone where the TPUs are located.')
flags.DEFINE_integer(
    'tpu_iterations_per_loop', 1000,
    'Steps per interior TPU loop. Should be less than '
    '--train_steps_per_eval.')
    
# MHingeGAN
flags.DEFINE_integer(
    'image_size', 32,
    'The size of the images in the dataset.')
flags.DEFINE_string('dataset_val_split_name', 'test',
    'For the datset that is being used, the name of the '
    'split that should be used for validation. In imagenet it is "validation",'
    'other times it is "test".')
flags.DEFINE_integer(
    'num_classes', 10,
    'The number of classes in the dataset.')
flags.DEFINE_string('data_dir', None,
                    'A directory for a TFDS datset. If `None`, use default.')
flags.DEFINE_string('dataset_name', 'imagenet2012',
                    'Which dataset to use. imagenet2012 is default.')
flags.DEFINE_float('aux_cond_generator_weight', None,
                   'How to scale generator ACGAN loss relative to WGAN loss, default is None. Try 0.1')
flags.DEFINE_float('aux_cond_discriminator_weight', None, 
                   'How to scale the critic ACGAN loss relative to WGAN loss, default is None. Try 1.0')
flags.DEFINE_float('aux_mhinge_cond_generator_weight', None,
                   '..., default is None.')
flags.DEFINE_float('aux_mhinge_cond_discriminator_weight', None, 
                   '..., default is None.')
flags.DEFINE_enum(
    'critic_type', 'acgan', ['acgan', 'kplusone_fm', 'kplusone_wgan', 'acgan_noproj', 'acgan_multiproj'],
    '...')
flags.DEFINE_float('kplusone_mhinge_cond_discriminator_weight', None, 
                   '..., default is None.')
# unlabelled data
flags.DEFINE_string('unlabelled_dataset_name', None,
                    'If set use unlabelled data.')
flags.DEFINE_string('unlabelled_dataset_split_name', 'unlabelled',
                    '...')
flags.DEFINE_float('kplusone_mhinge_ssl_cond_discriminator_weight', None, 
                   '..., default is None.')
flags.DEFINE_enum(
    'generator_loss_fn', None, ['kplusone_wasserstein_generator_loss', 'kplusone_featurematching_generator_loss', 'kplusone_ssl_featurematching_generator_loss'],
    '...')
flags.DEFINE_integer(
    'tpu_gan_estimator_d_step', 1,
    '...')
flags.DEFINE_float('generator_margin_size', 1.0, 'Used in achingegan_generator_loss.')
flags.DEFINE_integer( 'intra_fid_eval_start', 0, '...')
flags.DEFINE_integer( 'intra_fid_eval_end', None, '...')
flags.DEFINE_integer( 'tfdf_num_parallel_calls', 16, '...')


FLAGS = flags.FLAGS

def main(_):
  from tensorflow_gan.examples.self_attention_estimator import train_experiment
  
  # get TF logger
  log = logging.getLogger('tensorflow')
  log.setLevel(logging.INFO)
  
  # create formatter and add it to the handlers
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  
  # create file handler
  logging_dir = os.environ['HOME'] if FLAGS.use_tpu else FLAGS.model_dir
  fh = logging.FileHandler(logging_dir + '/tensorflow.log')
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)
  log.addHandler(fh)
    
  tpu_location = FLAGS.tpu
  hparams = train_experiment.HParams(
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size,
      generator_lr=FLAGS.generator_lr,
      discriminator_lr=FLAGS.discriminator_lr,
      beta1=FLAGS.beta1,
      gf_dim=FLAGS.gf_dim,
      df_dim=FLAGS.df_dim,
      num_classes=FLAGS.num_classes,
      shuffle_buffer_size=10000,
      z_dim=FLAGS.z_dim,
      model_dir=FLAGS.model_dir,
      max_number_of_steps=FLAGS.max_number_of_steps,
      train_steps_per_eval=FLAGS.train_steps_per_eval,
      num_eval_steps=FLAGS.num_eval_steps,
      debug_params=train_experiment.DebugParams(
          use_tpu=FLAGS.use_tpu,
          eval_on_tpu=FLAGS.eval_on_tpu,
          fake_nets=False,
          fake_data=False,
          continuous_eval_timeout_secs=FLAGS.continuous_eval_timeout_secs,
      ),
      tpu_params=train_experiment.TPUParams(
          use_tpu_estimator=FLAGS.use_tpu_estimator,
          tpu_location=tpu_location,
          gcp_project=FLAGS.gcp_project,
          tpu_zone=FLAGS.tpu_zone,
          tpu_iterations_per_loop=FLAGS.tpu_iterations_per_loop,
      ),
  )
  if FLAGS.mode == 'train':
    train_experiment.run_train(hparams)
  elif FLAGS.mode == 'continuous_eval':
    train_experiment.run_continuous_eval(hparams)
  elif FLAGS.mode == 'intra_fid_eval':
    train_experiment.run_intra_fid_eval(hparams)
  elif FLAGS.mode == 'train_and_eval' or FLAGS.mode is None:
    train_experiment.run_train_and_eval(hparams)
  else:
    raise ValueError('Mode not recognized: ', FLAGS.mode)


if __name__ == '__main__':
  app.run(main)
