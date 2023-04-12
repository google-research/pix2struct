# Copyright 2023 The pix2struct Authors.
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

# Lint as: python3
"""Preprocessors."""
from typing import Callable, Dict, Mapping, Tuple

import seqio
import tensorflow as tf

TensorMapping = Callable[[tf.Tensor], tf.Tensor]
FeaturesDict = Dict[str, tf.Tensor]
FeaturesMapping = Callable[[FeaturesDict], FeaturesDict]


def map_feature(key: str, map_fn: TensorMapping) -> FeaturesMapping:
  @seqio.utils.map_over_dataset
  def _mapper(features: FeaturesDict) -> FeaturesDict:
    features[key] = map_fn(features[key])
    return features
  return _mapper


def image_decoder(key: str, channels: int) -> FeaturesMapping:
  return map_feature(
      key=key,
      map_fn=lambda f: tf.io.decode_png(f, channels=channels))


def read_image(key: str, image_dir: str) -> FeaturesMapping:
  return map_feature(
      key=key,
      map_fn=lambda f: tf.io.read_file(tf.strings.join([image_dir, "/", f])))


def normalize_image(key: str) -> FeaturesMapping:
  return map_feature(
      key=key,
      map_fn=tf.image.per_image_standardization)


def sample_one(key: str) -> FeaturesMapping:
  return map_feature(
      key=key,
      map_fn=lambda v: tf.random.shuffle(v)[0])


def patch_sequence(
    image: tf.Tensor,
    max_patches: int,
    patch_size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
  """Extract patch sequence."""
  patch_height, patch_width = patch_size
  image_shape = tf.shape(image)
  image_height = image_shape[0]
  image_width = image_shape[1]
  image_channels = image_shape[2]
  image_height = tf.cast(image_height, tf.float32)
  image_width = tf.cast(image_width, tf.float32)

  # maximize scale s.t.
  # ceil(scale * image_height / patch_height) *
  # ceil(scale * image_width / patch_width) <= max_patches
  scale = tf.sqrt(
      max_patches *
      (patch_height / image_height) *
      (patch_width / image_width))
  num_feasible_rows = tf.maximum(tf.minimum(
      tf.math.floor(scale * image_height / patch_height),
      max_patches), 1)
  num_feasible_cols = tf.maximum(tf.minimum(
      tf.math.floor(scale * image_width / patch_width),
      max_patches), 1)
  resized_height = tf.maximum(
      tf.cast(num_feasible_rows * patch_height, tf.int32), 1)
  resized_width = tf.maximum(
      tf.cast(num_feasible_cols * patch_width, tf.int32), 1)

  image = tf.image.resize(
      images=image,
      size=(resized_height, resized_width),
      preserve_aspect_ratio=False,
      antialias=True)

  # [1, rows, columns, patch_height * patch_width * image_channels]
  patches = tf.image.extract_patches(
      images=tf.expand_dims(image, 0),
      sizes=[1, patch_height, patch_width, 1],
      strides=[1, patch_height, patch_width, 1],
      rates=[1, 1, 1, 1],
      padding="SAME")

  patches_shape = tf.shape(patches)
  rows = patches_shape[1]
  columns = patches_shape[2]
  depth = patches_shape[3]

  # [rows * columns, patch_height * patch_width * image_channels]
  patches = tf.reshape(patches, [rows * columns, depth])

  # [rows * columns, 1]
  row_ids = tf.reshape(
      tf.tile(tf.expand_dims(tf.range(rows), 1), [1, columns]),
      [rows * columns, 1])
  col_ids = tf.reshape(
      tf.tile(tf.expand_dims(tf.range(columns), 0), [rows, 1]),
      [rows * columns, 1])

  # Offset by 1 so the ids do not contain zeros, which represent padding.
  row_ids += 1
  col_ids += 1

  # Prepare additional patch information for concatenation with real values.
  row_ids = tf.cast(row_ids, tf.float32)
  col_ids = tf.cast(col_ids, tf.float32)

  # [rows * columns, 2 + patch_height * patch_width * image_channels]
  result = tf.concat([row_ids, col_ids, patches], -1)

  # [max_patches, 2 + patch_height * patch_width * image_channels]
  result = tf.pad(result, [[0, max_patches - (rows * columns)], [0, 0]])

  original_shape = tf.stack(
      [rows, columns, patch_height, patch_width, image_channels])
  return result, original_shape


def image_to_patches(
    key: str,
    patch_size: Tuple[int, int] = (16, 16)):
  """Image to patches."""

  @seqio.utils.map_over_dataset
  def _mapper(features: FeaturesDict,
              sequence_length: Mapping[str, int]) -> FeaturesDict:
    inputs, original_shape = patch_sequence(
        image=features[key],
        max_patches=sequence_length[key],
        patch_size=patch_size)
    features[key] = inputs
    features["original_shape"] = original_shape
    return features

  return _mapper
