# Copyright 2024 The pix2struct Authors.
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

"""Tests for pix2struct.models."""
from absl.testing import absltest
import gin
import jax
import numpy as np
from t5x import partitioning
from t5x import trainer as trainer_lib
from t5x import utils


class ModelsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    gin.clear_config()
    gin.add_config_file_search_path("pix2struct/configs")
    gin.parse_config_file("models/pix2struct.gin")
    gin.parse_config_file("optimizers/adafactor.gin")
    gin.parse_config_file("sizes/tiny.gin")
    gin.parse_config_file("init/random_init.gin")

    # Our Adafactor implementation requires knowing the total number of steps.
    # Don't use a real output vocab to keep this test hermetic.
    gin.parse_config("""
      TRAIN_STEPS = 1
      models.ImageToTextModel.output_vocabulary = @seqio.PassThroughVocabulary()
    """)
    gin.finalize()
    self.model = gin.query_parameter("%MODEL").scoped_configurable_fn()

    self.input_data = {
        "encoder_input_tokens": np.ones(shape=(8, 4, 5), dtype=np.float32),
        "decoder_input_tokens": np.ones(shape=(8, 3), dtype=np.int32),
        "decoder_target_tokens": np.ones(shape=(8, 3), dtype=np.int32)
    }
    self.partitioner = partitioning.PjitPartitioner(num_partitions=1)
    self.train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=self.model.optimizer_def,
        init_fn=self.model.get_initial_variables,
        input_shapes={k: v.shape for k, v in self.input_data.items()},
        partitioner=self.partitioner)
    self.train_state = self.train_state_initializer.from_scratch(
        jax.random.PRNGKey(0))

  def test_image_encoder_text_decoder_train(self):
    trainer = trainer_lib.Trainer(
        self.model,
        train_state=self.train_state,
        partitioner=self.partitioner,
        eval_names=[],
        summary_dir=None,
        train_state_axes=self.train_state_initializer.train_state_axes,
        rng=jax.random.PRNGKey(0),
        learning_rate_fn=lambda x: 0.001,
        num_microbatches=1)

    trainer.train(
        batch_iter=iter([self.input_data]),
        num_steps=1)

  def test_image_encoder_text_decoder_predict(self):
    predictions = self.model.predict_batch(
        params=self.train_state.params,
        batch=self.input_data)
    self.assertSequenceEqual(predictions.shape, [8, 3])

if __name__ == "__main__":
  absltest.main()
