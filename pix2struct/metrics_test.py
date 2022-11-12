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

"""Tests for metrics."""
from absl.testing import absltest
from pix2struct import metrics


class MetricsTest(absltest.TestCase):

  def test_instance_ranking_metrics(self):
    eval_dict, is_correct = metrics.instance_ranking_metrics(
        predictions=[
            # Rely on score ranking between 'true' labels.
            "true",
            "true",
            "true",
            # Rely on score ranking between 'false' labels.
            "false",
            "false",
            "false",
            # Rely on predicted label regardless of score.
            "false",
            "false",
            "true",
            # Rely on both predicted label and score.
            "false",
            "true",
            "true",
        ],
        aux_values={"scores": [
            # Rely on score ranking between all 'true' predictions.
            -1,
            1,
            0,
            # Rely on score ranking between all 'false' predictions.
            1,
            2,
            3,
            # Rely on predicted label regardless of score.
            0,
            0,
            0,
            # Rely on both predicted label and score.
            2,
            0,
            1,
        ]},
        targets=[
            # Rely on score ranking between 'true' labels.
            {"group_id": "0", "id": "0_0", "parse": ["false"]},
            {"group_id": "0", "id": "0_1", "parse": ["true"]},
            {"group_id": "0", "id": "0_2", "parse": ["false"]},
            # Rely on score ranking between 'false' labels.
            {"group_id": "1", "id": "1_0", "parse": ["true"]},
            {"group_id": "1", "id": "1_1", "parse": ["false"]},
            {"group_id": "1", "id": "1_2", "parse": ["false"]},
            # Rely on predicted label regardless of score.
            {"group_id": "2", "id": "2_0", "parse": ["false"]},
            {"group_id": "2", "id": "2_1", "parse": ["false"]},
            {"group_id": "2", "id": "2_2", "parse": ["true"]},
            # Rely on both predicted label and score.
            {"group_id": "3", "id": "3_0", "parse": ["false"]},
            {"group_id": "3", "id": "3_1", "parse": ["false"]},
            {"group_id": "3", "id": "3_2", "parse": ["true"]},
        ],
        group_fn=lambda t: t["group_id"],
        correct_fn=lambda t: t["parse"][0] == "true",
        ranking_fn=lambda p, s: (p == "true", s * (1 if p == "true" else -1)),
        return_correctness=True)
    self.assertEqual([True, True, True, True], is_correct)
    self.assertEqual(
        {
            "group_accuracy": 100.0,
            "total_groups": 4
        },
        eval_dict)

  def test_pix2struct_metrics(self):
    eval_dict = metrics.pix2struct_metrics(
        predictions=[
            "abc",
            "abc",
            "Abc",
            "100%",
            "100%",
            "100%",
            "100%",
            "Don't",
        ],
        targets=[
            ["abc"],
            ["Abc"],
            ["abc"],
            ["96%"],
            ["94%"],
            ["0.96"],
            ["0.94"],
            ["Won't"],
        ])
    for k, v in {
        "exact_match": 12.5,
        "anls": 47.5,
        "relaxed_accuracy": 62.5,
        "cider": 128.6
    }.items():
      self.assertAlmostEqual(v, eval_dict[k], places=1)

if __name__ == "__main__":
  absltest.main()
