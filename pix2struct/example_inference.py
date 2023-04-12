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

"""Example script for Pix2Struct Inference."""
from absl import flags
import gin
from pix2struct import demo_utils
from pix2struct import inference_utils
from t5x import gin_utils
import tensorflow as tf

flags.DEFINE_multi_string("gin_file", None, "Gin files.")
flags.DEFINE_multi_string("gin_bindings", [], "Individual gin bindings.")
flags.DEFINE_list("gin_search_paths", ["."], "Gin search paths.")
flags.DEFINE_string("image", "", "Path to the image file.")
flags.DEFINE_string("text", None, "Optional text (e.g. question).")

FLAGS = flags.FLAGS


def main(_) -> None:
  get_inference_fns_using_gin = gin.configurable(
      inference_utils.get_inference_fns
  )
  gin_utils.parse_gin_flags(
      gin_search_paths=FLAGS.gin_search_paths,
      gin_files=FLAGS.gin_file,
      gin_bindings=FLAGS.gin_bindings,
  )
  inference_fns = get_inference_fns_using_gin()
  predict_fn = inference_fns["predict"]

  with tf.io.gfile.GFile(FLAGS.image, "rb") as f:
    image_bytes = f.read()
  image_bytes = demo_utils.maybe_add_question(FLAGS.text, image_bytes)
  prediction = demo_utils.apply_single_inference(predict_fn, image_bytes)
  print(prediction)

if __name__ == "__main__":
  gin_utils.run(main)
