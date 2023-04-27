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

r"""Convert AI2D data.
"""
import json
import logging
import os
import string
from typing import Iterable

from absl import app
from absl import flags
import apache_beam as beam
from PIL import Image
from pix2struct.preprocessing import preprocessing_utils
import tensorflow as tf

flags.DEFINE_string(
    "data_dir",
    None,
    "Directory containing the AI2D data.")

flags.DEFINE_string(
    "test_ids_path",
    None,
    "Path to CSV file containing the ids of the test datapoints.")


def convert(input_path: str, data_dir: str) -> Iterable[tf.train.Example]:
  """Convert example."""
  with tf.io.gfile.GFile(input_path) as f:
    data = json.load(f)
  with tf.io.gfile.GFile(
      os.path.join(data_dir, "images", data["imageName"]), "rb") as f:
    image = Image.open(f)
  with tf.io.gfile.GFile(
      os.path.join(data_dir, "annotations", f"{data['imageName']}.json")) as f:
    annotation = json.load(f)

  image_with_placeholders = image.copy()
  for v in annotation["text"].values():
    preprocessing_utils.render_text_on_bounding_box(
        text=v["replacementText"],
        bounding_box=v["rectangle"],
        image=image_with_placeholders)

  for k, v in data["questions"].items():
    example = tf.train.Example()

    # The `image_id` field is only used to ensure correct splitting of the data.
    preprocessing_utils.add_text_feature(example, "image_id", data["imageName"])
    options = " ".join(
        f"({string.ascii_lowercase[i]}) {a}"
        for i, a in enumerate(v["answerTexts"])
    )

    image_with_header = preprocessing_utils.render_header(
        image=image_with_placeholders if v["abcLabel"] else image,
        header=f"{k} {options}",
    )
    preprocessing_utils.add_bytes_feature(
        example, "image", preprocessing_utils.image_to_bytes(image_with_header)
    )
    parse = v["answerTexts"][v["correctAnswer"]]
    preprocessing_utils.add_text_feature(example, "parse", parse)
    yield example


def pipeline(root):
  with tf.io.gfile.GFile(flags.FLAGS.test_ids_path) as f:
    test_ids = {f"{l.strip()}.png" for l in f if l.strip()}
  _ = (root
       | "Create" >> beam.Create(tf.io.gfile.glob(
           os.path.join(flags.FLAGS.data_dir, "questions", "*.json")))
       | "Convert" >> beam.FlatMap(convert, data_dir=flags.FLAGS.data_dir)
       | "Write" >> preprocessing_utils.SplitAndWriteTFRecords(
           output_dir=os.path.join(flags.FLAGS.data_dir, "processed"),
           key="image_id",
           validation_percent=1,
           is_test=lambda x: x in test_ids))


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:])) as root:
    pipeline(root)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  flags.mark_flag_as_required("data_dir")
  app.run(main)
