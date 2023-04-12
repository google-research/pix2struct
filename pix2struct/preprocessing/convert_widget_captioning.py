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

r"""Convert widget captioning data to the common Pix2Struct format.
"""
import csv
import io
import json
import logging
import os
from typing import Iterable

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pix2struct.preprocessing import preprocessing_utils
import tensorflow as tf

flags.DEFINE_string(
    "data_dir", None,
    "Directory containing the widget captioning data.")

flags.DEFINE_string("data_file", "widget_captions.csv", "Data file name.")

flags.DEFINE_integer("rico_canvas_y", 2560,
                     "Dataset property indicating the y-dim of the canvas.")

flags.DEFINE_string(
    "image_dir",
    None,
    "Directory containing the images referenced in refexp data.")

flags.DEFINE_string(
    "processed_dir",
    "processed",
    "Sub-directory containing the refexp processed data.")


class ProcessSplit(beam.PTransform):
  """Process split."""

  def __init__(self, split: str):
    self._split = split
    self._data_dir = flags.FLAGS.data_dir
    self._image_dir = flags.FLAGS.image_dir
    self._rico_canvas_y = flags.FLAGS.rico_canvas_y

  def get_node_box(self, screen_id, node_id, image_dims):
    index_list = [int(i) for i in node_id.split(".")[1:]]
    with tf.io.gfile.GFile(os.path.join(self._image_dir,
                                        screen_id + ".json")) as f:
      view = json.load(f)
    curr_node = view["activity"]["root"]
    for index in index_list:
      curr_node = curr_node["children"][index]
    normalized_bounds = map(lambda x: x * image_dims[0] / self._rico_canvas_y,
                            curr_node["bounds"])
    return normalized_bounds

  def convert_to_tf_examples(self, screen_id, node_id,
                             captions) -> Iterable[tf.train.Example]:
    # get image
    with tf.io.gfile.GFile(
        os.path.join(self._image_dir, screen_id + ".jpg"), "rb") as f:
      image = Image.open(f)
    image_dims = np.asarray(image).shape
    # get bounding box coordinates
    xmin, ymin, xmax, ymax = self.get_node_box(screen_id, node_id, image_dims)
    # draw bounding box
    img_draw = ImageDraw.Draw(image, "RGBA")
    img_draw.rectangle(
        xy=((xmin, ymax),
            (xmax, ymin)),
        fill=(0, 0, 255, 0),
        outline=(0, 0, 255, 255))
    tf_example = tf.train.Example()
    # Convert the image to bytes.
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    preprocessing_utils.add_bytes_feature(tf_example, "image",
                                          img_byte_arr.getvalue())
    preprocessing_utils.add_text_feature(tf_example, "id",
                                         str(f"{screen_id}_{node_id}"))
    for caption in captions.split("|"):
      preprocessing_utils.add_text_feature(tf_example, "parse", caption)
    yield tf_example

  def expand(self, root):
    # read split screen ids
    split_screen_ids = set()
    with tf.io.gfile.GFile(os.path.join(self._data_dir,
                                        self._split + ".txt")) as f:
      for line in f:
        split_screen_ids.add(line.strip())

    data = []
    with tf.io.gfile.GFile(os.path.join(self._data_dir,
                                        "widget_captions.csv")) as f:
      reader = csv.DictReader(f, delimiter=",")
      for row in reader:
        if row["screenId"] in split_screen_ids:
          data.append((row["screenId"], row["nodeId"], row["captions"]))

    output_path = os.path.join(
        self._data_dir, flags.FLAGS.processed_dir, f"{self._split}.tfr")
    return (root
            | "Create" >> beam.Create(data)
            | "Convert" >> beam.FlatMapTuple(self.convert_to_tf_examples)
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
  flags.mark_flag_as_required("image_dir")
  app.run(main)
