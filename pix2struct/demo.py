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

"""Web demo of Pix2Struct."""
import base64
import html
import io
import os
import wsgiref.simple_server

from absl import flags
import gin
import jinja2
import PIL.Image
from pix2struct import demo_utils
from pix2struct import inference_utils
from t5x import gin_utils
import tensorflow as tf
import tornado.web
import tornado.wsgi


flags.DEFINE_multi_string("gin_file", None, "Gin files.")
flags.DEFINE_multi_string("gin_bindings", [], "Individual gin bindings.")
flags.DEFINE_list("gin_search_paths", ["."], "Gin search paths.")
flags.DEFINE_integer("port", 8080, "Port number for localhost.")


FLAGS = flags.FLAGS


class ScreenshotHandler(tornado.web.RequestHandler):
  """Main handler."""
  _tmpl = None
  _demo_fn = None

  def initialize(self,
                 env=None,
                 demo_fn=None):
    self._demo_fn = demo_fn
    self._tmpl = env.get_template("demo_screenshot.html")

  def get(self):
    self.post()

  def post(self):
    if "image" in self.request.files:
      image_bytes = self.request.files["image"][0]["body"]
      image_bytes = demo_utils.maybe_add_question(
          question=self.get_argument("question", default=""),
          image_bytes=image_bytes,
      )
      prediction = html.escape(
          demo_utils.apply_single_inference(self._demo_fn, image_bytes)
      )
      image = tf.compat.as_str(base64.b64encode(image_bytes))
    else:
      prediction = ""
      image = ""
    self.write(self._tmpl.render(
        image=image,
        prediction=prediction))


def main(_):
  get_demo_fns_using_gin = gin.configurable(inference_utils.get_inference_fns)
  gin_utils.parse_gin_flags(
      gin_search_paths=FLAGS.gin_search_paths,
      gin_files=FLAGS.gin_file,
      gin_bindings=FLAGS.gin_bindings)
  demo_fn = get_demo_fns_using_gin()["predict"]

  print("Warming up demo function...")
  placeholder_bytes = io.BytesIO()
  PIL.Image.new("RGB", size=(1, 1)).save(placeholder_bytes, "png")
  demo_utils.apply_single_inference(demo_fn, placeholder_bytes.getvalue())
  print("Done warming up demo function.")

  web_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")
  env = jinja2.Environment(
      loader=jinja2.FileSystemLoader(os.path.join(web_path, "templates")))
  application = tornado.wsgi.WSGIApplication([
      (r"/", ScreenshotHandler, {
          "env": env,
          "demo_fn": demo_fn,
      }),
      (r"/static/(.*)", tornado.web.StaticFileHandler, {
          "path": os.path.join(web_path, "static")
      })
  ])
  server = wsgiref.simple_server.make_server("", FLAGS.port, application)
  print("<READY!>")
  server.serve_forever()

if __name__ == "__main__":
  gin_utils.run(main)
