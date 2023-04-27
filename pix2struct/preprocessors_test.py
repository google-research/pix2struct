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

"""Tests for preprocessors."""
import random
import numpy as np
from pix2struct import preprocessors
import tensorflow as tf


class PreprocessorsTest(tf.test.TestCase):

  def test_patch_sequence_divisible(self):
    max_patches = 512
    patch_size = (16, 16)
    expected_depth = (patch_size[0] * patch_size[1] * 3) + 2

    # Perfectly divisible without resizing.
    random_image = tf.random.uniform((512, 256, 3))
    patches, original_shape = preprocessors.patch_sequence(
        image=random_image,
        max_patches=max_patches,
        patch_size=patch_size)
    valid_patches = patches[patches[:, 0] > 0]
    positions = valid_patches[:, :2]
    self.assertAllEqual(patches.shape, [max_patches, expected_depth])
    self.assertAllEqual(valid_patches.shape, [max_patches, expected_depth])
    self.assertAllEqual(original_shape.shape, [5])
    self.assertAllGreater(positions, 0)
    self.assertAllLessEqual(positions, valid_patches.shape[0])

    # Perfectly divisible after scaling up.
    random_image = tf.random.uniform((1, 2, 3))
    patches, original_shape = preprocessors.patch_sequence(
        image=random_image,
        max_patches=max_patches,
        patch_size=patch_size)
    valid_patches = patches[patches[:, 0] > 0]
    positions = valid_patches[:, :2]
    self.assertAllEqual(patches.shape, [max_patches, expected_depth])
    self.assertAllEqual(valid_patches.shape, [max_patches, expected_depth])
    self.assertAllEqual(original_shape.shape, [5])
    self.assertAllGreater(positions, 0)
    self.assertAllLessEqual(positions, valid_patches.shape[0])

    # Perfectly divisible after scaling down.
    random_image = tf.random.uniform((2048, 1024, 3))
    patches, original_shape = preprocessors.patch_sequence(
        image=random_image,
        max_patches=max_patches,
        patch_size=patch_size)
    valid_patches = patches[patches[:, 0] > 0]
    positions = valid_patches[:, :2]
    self.assertAllEqual(patches.shape, [max_patches, expected_depth])
    self.assertAllEqual(valid_patches.shape, [max_patches, expected_depth])
    self.assertAllEqual(original_shape.shape, [5])
    self.assertAllGreater(positions, 0)
    self.assertAllLessEqual(positions, valid_patches.shape[0])

  def test_patch_sequence_random(self):
    # Test that random image sizes always respect the `max_patches` constraint
    # and always fills up at least half of the capacity.
    total_padding = 0
    num_trials = 100
    max_patches = 512
    patch_size = (16, 16)
    expected_depth = (patch_size[0] * patch_size[1] * 3) + 2

    for _ in range(num_trials):
      random_width = random.randint(1, 10000)
      random_height = random.randint(1, 10000)
      random_image = tf.random.uniform((random_width, random_height, 3))
      patches, original_shape = preprocessors.patch_sequence(
          image=random_image,
          max_patches=max_patches,
          patch_size=patch_size)
      valid_patches = patches[patches[:, 0] > 0]
      positions = valid_patches[:, :2]
      total_padding += patches.shape[0] - valid_patches.shape[0]
      self.assertAllEqual(patches.shape, [max_patches, expected_depth])
      self.assertLessEqual(valid_patches.shape[0], max_patches)
      self.assertAllEqual(original_shape.shape, [5])
      self.assertGreaterEqual(valid_patches.shape[0], max_patches / 2)
      self.assertAllGreater(positions, 0)
      self.assertAllLessEqual(positions, valid_patches.shape[0])

    # Average padding should be between 0 and half the sequence length.
    mean_padding = total_padding / num_trials
    self.assertGreater(mean_padding, 0)
    self.assertLess(mean_padding, max_patches / 2)

  def test_image_to_patches(self):
    random_image = tf.random.uniform((4, 4, 3))
    preprocessor = preprocessors.image_to_patches(
        key="inputs",
        patch_size=(1, 1))
    sequence_length = {"inputs": 7}
    dataset = tf.data.Dataset.from_tensors({"inputs": random_image})
    dataset = preprocessor(dataset, sequence_length=sequence_length)
    np.set_printoptions(threshold=np.inf)
    print(list(dataset.as_numpy_iterator()))

if __name__ == "__main__":
  tf.test.main()
