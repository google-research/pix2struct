# Copyright 2024 The pix2struct Authors.
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

"""Demo utils."""

import io
from typing import Any, Callable, Iterable

import PIL.Image
from pix2struct.preprocessing import preprocessing_utils
import tensorflow as tf


def maybe_add_question(question, image_bytes):
  if question:
    # If it exists, add a question as a header.
    image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    output_image = preprocessing_utils.render_header(image, question)
    output_image_bytes = io.BytesIO()
    output_image.save(output_image_bytes, format="PNG")
    return output_image_bytes.getvalue()
  else:
    return image_bytes


def apply_single_inference(
    inference_fn: Callable[[tf.data.Dataset], Iterable[Any]], image_bytes: bytes
) -> Any:
  dataset = tf.data.Dataset.from_tensors(
      {"id": "", "group_id": "", "image": image_bytes, "parse": [""]}
  )
  return next(iter(inference_fn(dataset)))
