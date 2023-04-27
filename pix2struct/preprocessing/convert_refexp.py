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

r"""Convert refexp data to the common Pix2Struct format.
"""
import logging
import os
import random
from typing import Iterable

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pix2struct.preprocessing import preprocessing_utils
import tensorflow as tf

flags.DEFINE_string("data_dir", None, "Directory containing the refexp data.")

flags.DEFINE_string(
    "image_dir",
    None,
    "Directory containing the images referenced in refexp data.")

flags.DEFINE_integer(
    "num_negative_samples",
    5,
    "Number of negative samples per instance.")


class ProcessSplit(beam.PTransform):
  """Process split."""

  def __init__(self, split: str):
    self._split = split
    self._data_dir = flags.FLAGS.data_dir
    self._image_dir = flags.FLAGS.image_dir

  def get_image(self, image_id):
    filename = image_id + ".jpg"
    with tf.io.gfile.GFile(os.path.join(self._image_dir, filename), "rb") as f:
      return Image.open(f)

  def draw_bounding_box(self, image, candidate_idx, example):
    def _get_coordinate(key, max_value):
      float_val = example.features.feature[key].float_list.value[candidate_idx]
      return round(float_val * max_value)
    image_dims = np.asarray(image).shape
    xmin = _get_coordinate("image/object/bbox/xmin", image_dims[1])
    xmax = _get_coordinate("image/object/bbox/xmax", image_dims[1])
    ymin = _get_coordinate("image/object/bbox/ymin", image_dims[0])
    ymax = _get_coordinate("image/object/bbox/ymax", image_dims[0])
    img_draw = ImageDraw.Draw(image, "RGBA")
    img_draw.rectangle(
        xy=((xmin, ymax),
            (xmax, ymin)),
        fill=(0, 0, 255, 0),
        outline=(0, 0, 255, 255))
    return image

  def convert_to_tf_examples(self, record_id, record
                             ) -> Iterable[tf.train.Example]:
    raw_example = tf.train.Example().FromString(record.numpy())
    record_id = record_id.numpy().item()
    try:
      label = preprocessing_utils.get_int_feature(raw_example,
                                                  "image/ref_exp/label")
      num_candidates = int(
          preprocessing_utils.get_float_feature(raw_example,
                                                "image/object/num"))
      query = preprocessing_utils.get_text_feature(raw_example,
                                                   "image/ref_exp/text")
      image_id = preprocessing_utils.get_text_feature(raw_example, "image/id")
      image = self.get_image(image_id)
    except (IndexError, tf.errors.NotFoundError):
      return

    if flags.FLAGS.num_negative_samples and self._split == "train":
      num_negative_samples = flags.FLAGS.num_negative_samples
    else:
      num_negative_samples = num_candidates

    candidates = list(cand for cand in range(num_candidates) if cand != label)
    random.shuffle(candidates)
    candidates = candidates[:num_negative_samples] + [label]
    for candidate_idx in candidates:
      tf_example = tf.train.Example()
      candidate_image = image.copy()
      candidate_image = self.draw_bounding_box(candidate_image, candidate_idx,
                                               raw_example)
      candidate_image = preprocessing_utils.render_header(
          candidate_image, query)
      is_correct = label == candidate_idx
      # pix2struct features
      preprocessing_utils.add_bytes_feature(
          tf_example, "image",
          preprocessing_utils.image_to_bytes(candidate_image))
      preprocessing_utils.add_text_feature(
          tf_example, "parse", str(is_correct).lower())
      preprocessing_utils.add_text_feature(
          tf_example, "id", str(f"{record_id}_{candidate_idx}"))
      # pix2box features
      preprocessing_utils.add_text_feature(
          tf_example, "group_id", str(record_id))
      preprocessing_utils.add_text_feature(
          tf_example, "candidate_id", str(candidate_idx))
      yield tf_example

  def expand(self, root):
    data_path = os.path.join(
        self._data_dir, f"{self._split}.tfrecord")
    raw_dataset = tf.data.TFRecordDataset([data_path])
    # get a unique id per record
    raw_dataset = raw_dataset.enumerate(start=0)
    output_path = os.path.join(
        self._data_dir, "processed", f"{self._split}.tfr")

    return (root
            | "Create" >> beam.Create(raw_dataset)
            | "Convert" >> beam.FlatMapTuple(self.convert_to_tf_examples)
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
  flags.mark_flag_as_required("image_dir")
  app.run(main)
