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

r"""Convert ChartQA data to the common Pix2Struct format.
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

flags.DEFINE_string("data_dir",
                    None,
                    "Directory containing the ChartQA data.")


class ProcessSplit(beam.PTransform):
  """Process split."""

  def __init__(self, split: str, version: str):
    self._split = split
    self._data_dir = flags.FLAGS.data_dir
    self._version = version

  def convert_to_tf_examples(self, example_id,
                             json_example) -> tf.train.Example:
    with tf.io.gfile.GFile(
        os.path.join(self._data_dir, self._split, "png",
                     json_example["imgname"]), "rb") as f:
      image = Image.open(f)

    tf_example = tf.train.Example()
    image_with_question = preprocessing_utils.render_header(
        image, json_example["query"])
    preprocessing_utils.add_bytes_feature(
        tf_example, "image",
        preprocessing_utils.image_to_bytes(image_with_question))
    preprocessing_utils.add_text_feature(
        tf_example, "id", f"{self._split}_{self._version}_{example_id}")
    preprocessing_utils.add_text_feature(tf_example, "parse",
                                         json_example["label"])
    return tf_example

  def expand(self, root):
    assert self._version in ("human", "augmented")
    data_path = os.path.join(self._data_dir, self._split,
                             f"{self._split}_{self._version}.json")
    with tf.io.gfile.GFile(data_path) as data_file:
      data = json.load(data_file)

    output_path = os.path.join(self._data_dir, f"processed_{self._version}",
                               f"{self._split}.tfr")
    return (root
            | "Create" >> beam.Create(enumerate(data))
            | "Convert" >> beam.MapTuple(self.convert_to_tf_examples)
            | "Shuffle" >> beam.Reshuffle()
            | "Write" >> beam.io.WriteToTFRecord(
                output_path,
                coder=beam.coders.ProtoCoder(tf.train.Example)))


def pipeline(root):
  _ = (root | "ProcessTrainHuman" >> ProcessSplit("train", "human"))
  _ = (root | "ProcessValHuman" >> ProcessSplit("val", "human"))
  _ = (root | "ProcessTestHuman" >> ProcessSplit("test", "human"))

  _ = (root | "ProcessTrainAugmented" >> ProcessSplit("train", "augmented"))
  _ = (root | "ProcessValAugmented" >> ProcessSplit("val", "augmented"))
  _ = (root | "ProcessTestAugmented" >> ProcessSplit("test", "augmented"))


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:])) as root:
    pipeline(root)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  flags.mark_flag_as_required("data_dir")
  app.run(main)
