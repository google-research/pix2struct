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

r"""Convert DocVQA/InfographicVQA data to the common Pix2Struct format.
"""

import json
import logging
import os

from absl import app
from absl import flags
import apache_beam as beam
from PIL import Image
from pix2struct.preprocessing import preprocessing_utils
import tensorflow as tf

flags.DEFINE_string(
    "data_dir",
    None,
    "Directory containing the DocVQA or InfographicVQA data.")


class ProcessSplit(beam.PTransform):
  """Process split."""

  def __init__(self, split: str):
    self._split = split
    self._data_dir = flags.FLAGS.data_dir

  def read_image(self, filename):
    with tf.io.gfile.GFile(os.path.join(
        self._data_dir, self._split, filename), "rb") as f:
      return Image.open(f)

  def convert_to_tf_examples(self, json_example) -> tf.train.Example:
    if "image" in json_example:
      image = self.read_image(json_example["image"])
    else:
      image = self.read_image(json_example["image_local_name"])

    tf_example = tf.train.Example()
    image_with_question = preprocessing_utils.render_header(
        image, json_example["question"]
    )
    preprocessing_utils.add_bytes_feature(
        tf_example,
        "image",
        preprocessing_utils.image_to_bytes(image_with_question),
    )
    preprocessing_utils.add_text_feature(
        tf_example, "id", str(json_example["questionId"])
    )
    # "N/A" parse for the test set where the answers are not available
    for parse in json_example.get("answers", ["N/A"]):
      preprocessing_utils.add_text_feature(tf_example, "parse", parse)
    return tf_example

  def expand(self, root):
    data_path = os.path.join(
        self._data_dir, self._split, f"{self._split}_v1.0.json"
    )
    with tf.io.gfile.GFile(data_path) as data_file:
      data = json.load(data_file)
    assert data["dataset_name"] in ("docvqa", "infographicVQA")
    assert data["dataset_version"] == "1.0"
    assert data["dataset_split"] == self._split

    output_path = os.path.join(
        self._data_dir, "processed", f"{self._split}.tfr")
    return (root
            | "Create" >> beam.Create(data["data"])
            | "Convert" >> beam.Map(self.convert_to_tf_examples)
            | "Shuffle" >> beam.Reshuffle()
            | "Write" >> beam.io.WriteToTFRecord(
                output_path,
                coder=beam.coders.ProtoCoder(tf.train.Example)))


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
