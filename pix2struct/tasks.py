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

"""Pix2Struct tasks."""
import functools
import os
from typing import Any, Callable, List, Optional
from pix2struct import metrics
from pix2struct import postprocessors
from pix2struct import preprocessors
import seqio
import tensorflow as tf

OUTPUT_FEATURES = dict(
    inputs=seqio.ContinuousFeature(rank=2, dtype=tf.float32),
    targets=seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(
            "gs://pix2struct-data/sentencepiece.model")))

KEY_MAP = dict(
    inputs="image",
    targets="parse",
    parse="parse",
    image="image",
    id="id",
    group_id="group_id")

PREPROCESSORS = [
    functools.partial(seqio.preprocessors.rekey, key_map=KEY_MAP),
    preprocessors.sample_one(key="targets"),
    preprocessors.image_decoder(key="inputs", channels=3),
    preprocessors.normalize_image(key="inputs"),
    preprocessors.image_to_patches(key="inputs"),
    seqio.preprocessors.tokenize_and_append_eos,
]

FEATURE_DESCRIPTION = {
    "id": tf.io.FixedLenFeature([], tf.string, default_value="no-id"),
    "image": tf.io.FixedLenFeature([], tf.string),
    "parse": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "group_id": tf.io.FixedLenFeature(
        [], tf.string, default_value="no-group-id"),
}


def add_pix2struct_task(
    name: str,
    base_dir: str,
    train_file_pattern: str,
    valid_file_pattern: str,
    test_file_pattern: Optional[str] = None,
    metric_fns: Optional[List[seqio.dataset_providers.MetricFnCallable]] = None,
    postprocess_fn: Optional[Callable[..., Any]] = None):
  """Add pix2struct task."""
  split_to_filepattern = {
      "train": os.path.join(base_dir, train_file_pattern),
      "validation": os.path.join(base_dir, valid_file_pattern)
  }
  if test_file_pattern is not None:
    split_to_filepattern["test"] = os.path.join(base_dir, test_file_pattern)

  for v in split_to_filepattern.values():
    if not tf.io.gfile.glob(v):
      print(f"[{name}] No files matching {v}."
            "Must run data preprocessing first to use this task.")

  seqio.TaskRegistry.add(
      name=name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern=split_to_filepattern,
          feature_description=FEATURE_DESCRIPTION),
      preprocessors=PREPROCESSORS,
      output_features=OUTPUT_FEATURES,
      postprocess_fn=postprocess_fn or postprocessors.multi_target,
      metric_fns=metric_fns or [metrics.pix2struct_metrics])


# MNIST dataset for debugging.
add_pix2struct_task(
    name="mnist",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="mnist/train.tfr*",
    valid_file_pattern="mnist/test.tfr*")

# TextCaps dataset from https://textvqa.org/textcaps/.
add_pix2struct_task(
    name="textcaps",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="textcaps/processed/train.tfr*",
    valid_file_pattern="textcaps/processed/val.tfr*",
    test_file_pattern="textcaps/processed/test.tfr*")

# Screen2Words dataset.
add_pix2struct_task(
    name="screen2words",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="screen2words/processed/train.tfr*",
    valid_file_pattern="screen2words/processed/dev.tfr*",
    test_file_pattern="screen2words/processed/test.tfr*",
)

# DocVQA (https://arxiv.org/abs/2007.00398).
add_pix2struct_task(
    name="docvqa",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="docvqa/processed/train.tfr*",
    valid_file_pattern="docvqa/processed/val.tfr*",
    test_file_pattern="docvqa/processed/test.tfr*")

add_pix2struct_task(
    name="infographicvqa",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="infographicvqa/processed/train.tfr*",
    valid_file_pattern="infographicvqa/processed/val.tfr*",
    test_file_pattern="infographicvqa/processed/test.tfr*")

add_pix2struct_task(
    name="ocrvqa",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="ocrvqa/processed/train.tfr*",
    valid_file_pattern="ocrvqa/processed/val.tfr*",
    test_file_pattern="ocrvqa/processed/test.tfr*")

add_pix2struct_task(
    name="refexp",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="refexp/processed/train.tfr*",
    valid_file_pattern="refexp/processed/val.tfr*",
    test_file_pattern="refexp/processed/test.tfr*",
    metric_fns=[functools.partial(
        metrics.instance_ranking_metrics,
        group_fn=lambda t: t["group_id"],
        correct_fn=lambda t: t["parse"][0] == "true",
        ranking_fn=lambda p, s: (p == "true", s * (1 if p == "true" else -1))
    )],
    postprocess_fn=postprocessors.group_target)

add_pix2struct_task(
    name="widget_captioning",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="widget_captioning/processed/train.tfr*",
    valid_file_pattern="widget_captioning/processed/val.tfr*",
    test_file_pattern="widget_captioning/processed/test.tfr*")

add_pix2struct_task(
    name="chartqa_augmented",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="chartqa/processed_augmented/train.tfr*",
    valid_file_pattern="chartqa/processed_augmented/val.tfr*",
    test_file_pattern="chartqa/processed_augmented/test.tfr*")

add_pix2struct_task(
    name="chartqa_human",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="chartqa/processed_human/train.tfr*",
    valid_file_pattern="chartqa/processed_human/val.tfr*",
    test_file_pattern="chartqa/processed_human/test.tfr*")

seqio.MixtureRegistry.add(
    "chartqa",
    ["chartqa_human", "chartqa_augmented"],
    default_rate=1.0)

add_pix2struct_task(
    name="ai2d",
    base_dir=os.environ["PIX2STRUCT_DIR"] + "/data",
    train_file_pattern="ai2d/processed/train.tfr*",
    valid_file_pattern="ai2d/processed/val.tfr*",
    test_file_pattern="ai2d/processed/test.tfr*")
