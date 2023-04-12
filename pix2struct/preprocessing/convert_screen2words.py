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

r"""Convert Screen2Words data to a format that can be streamed.
"""
import logging
import os
from typing import Iterable, Tuple

from absl import app
from absl import flags
import apache_beam as beam
from pix2struct.preprocessing import preprocessing_utils
import tensorflow as tf

flags.DEFINE_string("screen2words_dir", None,
                    "Directory containing Screen2Words data.")

flags.DEFINE_string("rico_dir", None, "Directory containing RICO data.")


def parse_summary_line(line: str) -> Iterable[Tuple[str, str]]:
  line = line.strip()
  if line and line != "screenId,summary":
    screen_id, summary = line.split(",", 1)
    yield screen_id, summary


class ProcessSplit(beam.PTransform):
  """Examples from task."""

  def __init__(self, split: str):
    self._split = split
    self._rico_dir = flags.FLAGS.rico_dir
    self._screen2words_dir = flags.FLAGS.screen2words_dir

  def get_image(self, screen_id: str) -> bytes:
    with tf.io.gfile.GFile(
        os.path.join(self._rico_dir, f"{screen_id}.jpg"), "rb") as f:
      return f.read()

  def convert_to_tf_examples(
      self,
      screen_id: str,
      summaries_and_dummy: Tuple[Iterable[str], Iterable[bool]]
      ) -> Iterable[tf.train.Example]:
    """Convert the results of joining examples and splits to TF examples."""
    summaries, dummy = summaries_and_dummy
    # Only yield examples if there was a non-empty join with the intended split.
    if any(dummy):
      example = tf.train.Example()
      preprocessing_utils.add_bytes_feature(
          example, "image", self.get_image(screen_id))
      for summary in summaries:
        preprocessing_utils.add_text_feature(
            example, "parse", summary)
      yield example

  def expand(self, root_and_summaries):
    root, summaries = root_and_summaries
    screens_path = os.path.join(flags.FLAGS.screen2words_dir, "split",
                                f"{self._split}_screens.txt")
    screen_ids_for_split = (root
                            | "Read" >> beam.io.ReadFromText(screens_path)
                            | "Parse" >> beam.Map(lambda l: (l.strip(), True)))
    output_path = os.path.join(
        self._screen2words_dir, "processed", f"{self._split}.tfr")
    return ((summaries, screen_ids_for_split)
            | "Join" >> beam.CoGroupByKey()
            | "Convert" >> beam.FlatMapTuple(self.convert_to_tf_examples)
            | "Shuffle" >> beam.Reshuffle()
            | "Write" >> beam.io.WriteToTFRecord(
                output_path,
                coder=beam.coders.ProtoCoder(tf.train.Example)))


def pipeline(root):
  """Pipeline."""
  summaries = (
      root
      | "ReadSummaries" >> beam.io.ReadFromText(
          os.path.join(flags.FLAGS.screen2words_dir, "screen_summaries.csv"))
      | "ParseSummaries" >> beam.FlatMap(parse_summary_line))
  _ = ((root, summaries) | "ProcessTrain" >> ProcessSplit("train"))
  _ = ((root, summaries) | "ProcessDev" >> ProcessSplit("dev"))
  _ = ((root, summaries) | "ProcessTest" >> ProcessSplit("test"))


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:])) as root:
    pipeline(root)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  flags.mark_flag_as_required("screen2words_dir")
  flags.mark_flag_as_required("rico_dir")
  app.run(main)
