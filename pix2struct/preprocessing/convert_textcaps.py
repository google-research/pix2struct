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

r"""Convert TextCaps data.
"""
import collections
import io
import json
import logging
import os
from typing import Iterable, List, Tuple
from absl import app
from absl import flags
import apache_beam as beam
import PIL
from pix2struct.preprocessing import preprocessing_utils
import tensorflow as tf

flags.DEFINE_string("textcaps_dir", None, "Train captions path.")

flags.DEFINE_string("output_dir", None, "Output path.")


class ConvertDataForSplit(beam.PTransform):
  """Convert data for split."""

  def __init__(self, split: str, image_dir: str):
    self._split = split
    self._textcaps_dir = flags.FLAGS.textcaps_dir
    self._output_dir = flags.FLAGS.output_dir
    self._image_dir = image_dir

  def json_to_image_ids_and_captions(self) -> Iterable[Tuple[str, List[str]]]:
    with tf.io.gfile.GFile(os.path.join(
        self._textcaps_dir, f"TextCaps_0.1_{self._split}.json")) as f:
      data = json.load(f)
    assert data["dataset_name"] == "textcaps"
    assert data["dataset_type"] == self._split
    example_dict = collections.defaultdict(list)
    for example in data["data"]:
      example_dict[example["image_id"]].append(
          example.get("caption_str", "N/A"))
    for image_id, captions in example_dict.items():
      yield image_id, captions

  def image_id_and_caption_to_example(self, image_id: str,
                                      captions: List[str]) -> tf.train.Example:
    image_bytes = io.BytesIO()
    with tf.io.gfile.GFile(os.path.join(
        self._textcaps_dir, self._image_dir, f"{image_id}.jpg"), "rb") as f:
      PIL.Image.open(f).convert("RGB").save(image_bytes, format="PNG")
    example = tf.train.Example()
    preprocessing_utils.add_text_feature(example, "id", image_id)
    preprocessing_utils.add_bytes_feature(
        example, "image", image_bytes.getvalue())
    for caption in captions:
      preprocessing_utils.add_text_feature(example, "parse", caption)
    return example

  def expand(self, pcoll):
    return (pcoll
            | "Read" >> beam.Create(self.json_to_image_ids_and_captions())
            | "Reshuffle" >> beam.Reshuffle()
            | "ToExample" >> beam.MapTuple(self.image_id_and_caption_to_example)
            | "Write" >> beam.io.WriteToTFRecord(
                os.path.join(self._output_dir, f"{self._split}.tfr"),
                coder=beam.coders.ProtoCoder(tf.train.Example)))


def pipeline(root):
  """Pipeline."""
  _ = (root | "ConvertTrain" >> ConvertDataForSplit("train", "train_images"))
  _ = (root | "ConvertVal" >> ConvertDataForSplit("val", "train_images"))
  _ = (root | "ConvertTest" >> ConvertDataForSplit("test", "test_images"))


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:])) as root:
    pipeline(root)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  flags.mark_flag_as_required("textcaps_dir")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
