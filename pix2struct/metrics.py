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

"""Metrics."""
import collections
import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
import editdistance


def aggregate_metrics(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str],
    metric_fn: Callable[[str, str], Any],
    normalize_fn: Callable[[str], str] = lambda v: v) -> float:
  """Aggregate target-prediction pair metrics over a dataset."""
  assert len(targets) == len(predictions)
  total = 0
  for prediction, target in zip(predictions, targets):
    p = normalize_fn(prediction)
    total += max(metric_fn(normalize_fn(t), p) for t in target)
  return (100.0 * total) / len(targets)


def cider(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str]) -> float:
  """Compute CIDEr score."""
  coco_tokenizer = PTBTokenizer()

  scorer = Cider()
  avg_score, _ = scorer.compute_score(
      gts=coco_tokenizer.tokenize({
          str(i): [{"caption": t} for t in target]
          for i, target in enumerate(targets)
      }),
      res=coco_tokenizer.tokenize({
          str(i): [{"caption": prediction}]
          for i, prediction in enumerate(predictions)
      }))
  return float(avg_score) * 100.0


def anls_metric(target: str, prediction: str, theta: float = 0.5):
  """Calculates ANLS for DocVQA and InfographicVQA.

  Official evaluation script at https://rrc.cvc.uab.es/?ch=17&com=downloads
  (Infographics VQA Evaluation scripts).

  Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

  Args:
    target: Target string.
    prediction: Predicted string.
    theta: Filter threshold set to 0.5 for DocVQA.

  Returns:
    ANLS score.
  """

  edit_distance = editdistance.eval(target, prediction)
  normalized_ld = edit_distance / max(len(target), len(prediction))
  return 1 - normalized_ld if normalized_ld <= theta else 0


def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
  """Calculates relaxed correctness.

  The correctness tolerates certain error ratio defined by max_relative_change.
  See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
  “Following Methani et al. (2020), we use a relaxed accuracy measure for the
  numeric answers to allow a minor inaccuracy that may result from the automatic
  data extraction process. We consider an answer to be correct if it is within
  5% of the gold answer. For non-numeric answers, we still need an exact match
  to consider an answer to be correct.”

  Args:
    target: Target string.
    prediction: Predicted string.
    max_relative_change: Maximum relative change.

  Returns:
    Whether the prediction was correct given the specified tolerance.
  """

  def _to_float(text: str) -> Optional[float]:
    try:
      if text.endswith("%"):
        # Convert percentages to floats.
        return float(text.rstrip("%")) / 100.0
      else:
        return float(text)
    except ValueError:
      return None

  prediction_float = _to_float(prediction)
  target_float = _to_float(target)
  if prediction_float is not None and target_float:
    relative_change = abs(prediction_float - target_float) / abs(target_float)
    return relative_change <= max_relative_change
  else:
    return prediction.lower() == target.lower()


def pix2struct_metrics(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str]) -> Mapping[str, float]:
  """Calculates evaluation metrics.

  Args:
    targets: list of list of strings.
    predictions: list of strings.

  Returns:
    dictionary with metric names as keys and metric value as values.
  """
  return dict(
      exact_match=aggregate_metrics(
          targets=targets,
          predictions=predictions,
          metric_fn=lambda x, y: x == y),
      anls=aggregate_metrics(
          targets=targets,
          predictions=predictions,
          metric_fn=anls_metric,
          normalize_fn=lambda v: v.lower()),
      relaxed_accuracy=aggregate_metrics(
          targets=targets,
          predictions=predictions,
          metric_fn=relaxed_correctness),
      cider=cider(
          targets=targets,
          predictions=predictions))


def instance_ranking_metrics(
    targets: List[Dict[str, Any]],
    predictions: List[str],
    aux_values: Dict[str, Any],
    group_fn: Callable[[Any], Any],
    correct_fn: Callable[[Any], bool],
    ranking_fn: Callable[[str, float], Any],
    return_correctness: bool = False
    ) -> Union[Mapping[str, float], Tuple[Mapping[str, float], List[bool]]]:
  """Compute accuracy of instance ranking.

  Args:
    targets: List of dictionaries after the postprocessor is applied.
    predictions: List of predicted strings.
    aux_values: Dictionary where the "scores" entry has a list of float scores.
    group_fn: Function that maps a target to a grouping key.
    correct_fn: Function that maps a target to a boolean indicating correctness.
      Must return `True` for exactly one instance per group.
    ranking_fn: Function that maps a (prediction, score) pair to a something
      that can be used as a key to rank instances.
    return_correctness: Whether or not to also return a list of judgments of
      about correctness. Used for testing only.
  Returns:
    Dictionary with metric names as keys and metric value as values. Optionally
    also returns a list of correctness if specified.
  """
  Instance = collections.namedtuple(
      "Instance", ["target", "prediction", "score"])
  assert len(targets) == len(predictions) == len(aux_values["scores"])
  instances = [Instance(t, p, s) for t, p, s in
               zip(targets, predictions, aux_values["scores"])]
  is_correct = []
  total_groups = 0
  for _, group in itertools.groupby(
      sorted(instances, key=lambda i: group_fn(i.target)),
      lambda i: group_fn(i.target)):
    group = list(group)
    best_idx, _ = max(
        enumerate(group),
        key=lambda idx_i: ranking_fn(idx_i[1].prediction, idx_i[1].score))
    (true_idx,) = [idx for idx, i in enumerate(group)
                   if correct_fn(i.target)]
    is_correct.append(best_idx == true_idx)
    total_groups += 1
  eval_dict = dict(
      group_accuracy=sum(is_correct) * 100.0 / total_groups,
      total_groups=total_groups)
  if return_correctness:
    return eval_dict, is_correct
  else:
    return eval_dict
