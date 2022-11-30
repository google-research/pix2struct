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

r"""Convert OCR-VQA data to the common Pix2Struct format.
"""


import json
import logging
import os
from typing import Iterable

from absl import app
from absl import flags
import apache_beam as beam
from PIL import Image
from pix2struct.preprocessing import preprocessing_utils
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir",
    None,
    "Directory containing OCR-VQA data.")


class ProcessSplit(beam.PTransform):
  """Process split."""

  # fixed extension since all images have already been converted to .jpg
  def __init__(self, split: str, extension: str = ".jpg"):
    self._split = split
    split_indexes = {"train": 1, "val": 2, "test": 3}
    self._split_index = split_indexes[split]
    self._extension = extension
    self._data_dir = FLAGS.data_dir
    self._error_counter = beam.metrics.Metrics.counter(
        "example", "errors")
    self._processed_counter = beam.metrics.Metrics.counter(
        "example", "processed")

  def read_image(self, filename):
    with tf.io.gfile.GFile(os.path.join(
        self._data_dir, filename), "rb") as f:
      return Image.open(f)

  def convert_to_tf_examples(self, json_example) -> Iterable[tf.train.Example]:
    """Returns a list, either empty or with a single example."""

    # OutOfRangeError happens due to unknown reasons on some training examples;
    # we discard those.
    try:
      image = self.read_image(json_example["image"])
    except tf.errors.OutOfRangeError:
      self._error_counter.inc()
      return

    tf_example = tf.train.Example()
    image_with_question = preprocessing_utils.render_header(
        image, json_example["question"])
    preprocessing_utils.add_bytes_feature(
        tf_example, "image",
        preprocessing_utils.image_to_bytes(image_with_question))
    preprocessing_utils.add_text_feature(
        tf_example, "id", str(json_example["questionId"]))
    for answer in json_example["answers"]:
      preprocessing_utils.add_text_feature(tf_example, "parse", answer)
    self._processed_counter.inc()
    yield tf_example

  def expand(self, root):
    data_path = os.path.join(
        self._data_dir, "dataset.json")
    with tf.io.gfile.GFile(data_path) as data_file:
      json_data = json.load(data_file)
    data = []
    question_id = 0
    for image_id, image_data in json_data.items():
      if image_data["split"] != self._split_index:
        continue
      for question, answer in zip(image_data["questions"],
                                  image_data["answers"]):
        data.append({
            "question": question,
            "questionId": str(question_id),
            "answers": [answer],
            "image": f"images/{image_id}{self._extension}"
        })
        question_id += 1

    output_path = os.path.join(
        self._data_dir, "processed", f"{self._split}.tfr")
    return (root
            | "Create" >> beam.Create(data)
            | "Convert" >> beam.FlatMap(self.convert_to_tf_examples)
            | "Shuffle" >> beam.Reshuffle()
            | "Write" >> beam.io.WriteToTFRecord(
                output_path, coder=beam.coders.ProtoCoder(tf.train.Example)))


def pipeline(root):
  _ = (root | "ProcessTrain" >> ProcessSplit("train"))
  _ = (root | "ProcessVal" >> ProcessSplit("val"))
  _ = (root | "ProcessTest" >> ProcessSplit("test"))


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:])) as root:
    pipeline(root)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  flags.mark_flag_as_required("data_dir")
  app.run(main)
