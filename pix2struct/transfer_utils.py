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

"""Transfer utils."""
import dataclasses
import os
from typing import Optional
import gin

import optax
from t5x import utils


@dataclasses.dataclass
class TransferRestoreCheckpointConfig(utils.RestoreCheckpointConfig):
  """Transfer restore checkpoint config."""
  steps: Optional[int] = None

  def __post_init__(self):
    super().__post_init__()
    if self.steps is not None:
      assert self.mode == "specific"
      self.path = os.path.join(self.path, f"checkpoint_{self.steps}")


def transfer_warmup_cosine_decay_schedule(
    peak_value: float,
    warmup_steps: int,
    start_step: int,
    end_step: int,
    end_value: float = 0.0,
    cycle_length_ratio: float = 1.0,
) -> optax.Schedule:
  """Warmup cosine decay schedule with offset."""
  assert end_step >= start_step

  # Optionally adjust cycle length to overshoot the actually number of steps in
  # order to not stop at exactly 0. See https://arxiv.org/abs/2203.15556.
  decay_steps = int((end_step - start_step) * cycle_length_ratio)

  schedules = [
      optax.linear_schedule(
          init_value=0,
          end_value=0,
          transition_steps=start_step),
      optax.warmup_cosine_decay_schedule(
          init_value=0,
          peak_value=peak_value,
          warmup_steps=warmup_steps,
          decay_steps=decay_steps,
          end_value=end_value)]
  return optax.join_schedules(schedules, [start_step])


@gin.configurable
def add(a: int = gin.REQUIRED, b: int = gin.REQUIRED):
  return a + b
