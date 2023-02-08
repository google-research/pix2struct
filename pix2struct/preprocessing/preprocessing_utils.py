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

"""Preprocessing utils."""
import hashlib
import io
import os
import random
import textwrap
from typing import Any, Callable, Iterable, List, Optional

import apache_beam as beam
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorflow as tf

DEFAULT_FONT_PATH = "arial.ttf"


def add_int_feature(example: tf.train.Example,
                    key: str,
                    value: int) -> None:
  example.features.feature[key].int64_list.value.append(value)


def add_bytes_feature(example: tf.train.Example,
                      key: str,
                      value: bytes) -> None:
  example.features.feature[key].bytes_list.value.append(value)


def add_text_feature(example: tf.train.Example, key: str, value: str) -> None:
  add_bytes_feature(example, key, value.encode("utf-8"))


def get_bytes_feature(example: tf.train.Example, key: str) -> bytes:
  return example.features.feature[key].bytes_list.value[0]


def get_text_feature(example: tf.train.Example, key: str) -> str:
  return get_bytes_feature(example, key).decode("utf-8")


def get_text_features(example: tf.train.Example, key: str) -> List[str]:
  return [v.decode("utf-8")
          for v in example.features.feature[key].bytes_list.value]


def get_int_feature(example: tf.train.Example, key: str) -> int:
  return example.features.feature[key].int64_list.value[0]


def get_float_feature(example: tf.train.Example, key: str) -> float:
  return example.features.feature[key].float_list.value[0]


def get_hash(key: str) -> int:
  return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)


def keep_every(_: Any, ratio: float) -> bool:
  return random.random() < ratio


def deterministic_sample(items: Iterable[Any], value_fn) -> Any:
  return max(items, key=lambda x: get_hash(value_fn(x)))


def image_to_bytes(image: Image.Image) -> bytes:
  img_byte_arr = io.BytesIO()
  image.save(img_byte_arr, format="PNG")
  return img_byte_arr.getvalue()


def render_header(image: Image.Image, header: str) -> Image.Image:
  """Renders a header on a PIL image and returns a new PIL image."""
  header_image = render_text(header)
  new_width = max(header_image.width, image.width)

  new_height = int(image.height *  (new_width / image.width))
  new_header_height = int(
      header_image.height * (new_width / header_image.width))

  new_image = Image.new(
      "RGB",
      (new_width, new_height + new_header_height),
      "white")
  new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
  new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

  return new_image


def render_text(text: str,
                text_size: int = 36,
                text_color: str = "black",
                background_color: str = "white",
                left_padding: int = 5,
                right_padding: int = 5,
                top_padding: int = 5,
                bottom_padding: int = 5,
                font_path: str = DEFAULT_FONT_PATH) -> Image.Image:
  """Render text."""
  # Add new lines so that each line is no more than 80 characters.
  wrapper = textwrap.TextWrapper(width=80)
  lines = wrapper.wrap(text=text)
  wrapped_text = "\n".join(lines)

  if tf.io.gfile.exists(font_path):
    with tf.io.gfile.GFile(font_path, "rb") as font_file:
      font = ImageFont.truetype(font_file, encoding="UTF-8", size=text_size)
  else:
    font = ImageFont.truetype(font_path, encoding="UTF-8", size=text_size)

  # Use a temporary canvas to determine the width and height in pixels when
  # rendering the text.
  temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))
  _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

  # Create the actual image with a bit of padding around the text.
  image_width = text_width + left_padding + right_padding
  image_height = text_height + top_padding + bottom_padding
  image = Image.new("RGB", (image_width, image_height), background_color)
  draw = ImageDraw.Draw(image)
  draw.text(
      xy=(left_padding, top_padding),
      text=wrapped_text,
      fill=text_color,
      font=font)
  return image


def render_text_on_bounding_box(
    text: str,
    bounding_box: Iterable[Iterable[int]],
    image: Image.Image):
  """Render text on top of a specific bounding box."""
  draw = ImageDraw.Draw(image)
  (x0, y0), (x1, y1) = bounding_box
  draw.rectangle(xy=[(x0, y0), (x1, y1)], fill=(255, 255, 255, 255))

  fontsize = 1
  def _can_increment_font(ratio=0.95):
    next_font = ImageFont.truetype(
        DEFAULT_FONT_PATH, encoding="UTF-8", size=fontsize + 1)
    width, height = next_font.getsize(text)
    return width < ratio * (x1 - x0) and height < ratio * (y1 - y0)

  while _can_increment_font():
    fontsize += 1
  font = ImageFont.truetype(DEFAULT_FONT_PATH, encoding="UTF-8", size=fontsize)

  draw.text(
      xy=((x0 + x1)/2, (y0 + y1)/2),
      text=text,
      font=font,
      fill="black",
      anchor="mm")


def increment_counter(item, counter):
  counter.inc()
  return item


class SplitAndWriteTFRecords(beam.PTransform):
  """Split and write TFRecords."""

  def __init__(self,
               output_dir: str,
               key: str,
               validation_percent: Optional[int] = 10,
               train_file_name: str = "train.tfr",
               val_file_name: str = "val.tfr",
               test_file_name: str = "test.tfr",
               is_test: Optional[Callable[[str], bool]] = None):
    self._output_dir = output_dir
    self._key = key
    self._validation_percent = validation_percent
    self._train_file_name = train_file_name
    self._val_file_name = val_file_name
    self._test_file_name = test_file_name
    self._is_test = is_test
    self._train_counter = beam.metrics.Metrics.counter(
        "SplitAndWriteTFRecords", "train")
    self._val_counter = beam.metrics.Metrics.counter(
        "SplitAndWriteTFRecords", "val")
    self._test_counter = beam.metrics.Metrics.counter(
        "SplitAndWriteTFRecords", "test")

  def _partition_index(self,
                       example: tf.train.Example,
                       num_partitions: int) -> int:
    assert num_partitions == 3
    key_feature = get_text_feature(example, self._key)
    if self._is_test is not None and self._is_test(key_feature):
      return 2
    else:
      return int(get_hash(key_feature) % 100 < self._validation_percent)

  def expand(self, pcoll):
    train, val, test = (pcoll
                        | "Shuffle" >> beam.Reshuffle()
                        | "Partition" >> beam.Partition(
                            self._partition_index, 3))
    _ = (train
         | "CountTrain" >> beam.Map(increment_counter, self._train_counter)
         | "WriteTrain" >> beam.io.WriteToTFRecord(
             os.path.join(self._output_dir, self._train_file_name),
             coder=beam.coders.ProtoCoder(tf.train.Example)))
    _ = (val
         | "CountVal" >> beam.Map(increment_counter, self._val_counter)
         | "WriteVal" >> beam.io.WriteToTFRecord(
             os.path.join(self._output_dir, self._val_file_name),
             coder=beam.coders.ProtoCoder(tf.train.Example)))
    if self._is_test is not None:
      _ = (test
           | "CountTest" >> beam.Map(increment_counter, self._test_counter)
           | "WriteTest" >> beam.io.WriteToTFRecord(
               os.path.join(self._output_dir, self._test_file_name),
               coder=beam.coders.ProtoCoder(tf.train.Example)))


class DeterministicSamplePerKey(beam.PTransform):
  """Deterministic sample per key."""

  def __init__(self,
               key_fn: Callable[[Any], str],
               value_fn: Callable[[Any], str]):
    self._key_fn = key_fn
    self._value_fn = value_fn

  def expand(self, pcoll):
    return (pcoll
            | "AddKeys" >> beam.WithKeys(self._key_fn)
            | "SampleOne" >> beam.CombinePerKey(
                deterministic_sample, value_fn=self._value_fn)
            | "DropKeys" >> beam.Values())
