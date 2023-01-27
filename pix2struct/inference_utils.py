# Copyright 2022 The pix2struct Authors.
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

"""Inference utils."""

from typing import Any, Callable, Dict, Iterable, Mapping
import seqio
from t5x import models
from t5x import partitioning
from t5x import utils
import tensorflow as tf


def get_inference_fns(
    task_name: str,
    batch_size: int,
    sequence_length: Mapping[str, int],
    model: models.BaseTransformerModel,
    checkpoint_path: str,
    partitioner: partitioning.BasePartitioner
    ) -> Dict[str, Callable[[tf.data.Dataset], Iterable[Any]]]:
  """Get inference function."""
  task = seqio.get_mixture_or_task(task_name)
  feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)

  def _task_to_dataset(t: seqio.Task) -> tf.data.Dataset:
    d = t.get_dataset(
        sequence_length=sequence_length,
        split=task.splits[0],
        shuffle=False,
        num_epochs=1,
        use_cached=False)
    return feature_converter(d, sequence_length)

  input_shapes = {
      k: (batch_size,) + spec.shape for k, spec in
      _task_to_dataset(task).element_spec.items()
  }
  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=None,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner)
  restore_checkpoint_cfg = utils.RestoreCheckpointConfig(
      path=checkpoint_path,
      mode="specific",
      strict=False)

  train_state = train_state_initializer.from_checkpoint(
      [restore_checkpoint_cfg])
  assert train_state is not None

  def _dataset_to_batches(dataset: tf.data.Dataset) -> Iterable[Any]:
    temp_task = seqio.Task(
        name="tmp",
        source=seqio.FunctionDataSource(
            dataset_fn=lambda split, shuffle_files: dataset,
            splits=["tmp"]),
        output_features=task.output_features,
        preprocessors=task.preprocessors)
    temp_dataset = _task_to_dataset(temp_task)
    temp_dataset = temp_dataset.batch(batch_size)
    return temp_dataset.as_numpy_iterator()

  vocabulary = task.output_features["targets"].vocabulary
  def _predict(dataset: tf.data.Dataset) -> Iterable[str]:
    for batch in _dataset_to_batches(dataset):
      for token_ids in model.predict_batch(train_state.params, batch):
        yield vocabulary.decode(token_ids)

  def _intermediates(dataset: tf.data.Dataset) -> Iterable[Any]:
    for batch in _dataset_to_batches(dataset):
      _, intermediates = model.score_batch(
          train_state.params, batch, return_intermediates=True)
      yield batch, intermediates

  return {
      "predict": _predict,
      "intermediates": _intermediates,
  }
