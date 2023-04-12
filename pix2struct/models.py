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

"""Models."""
from typing import Callable

from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import seqio
from t5x import models
import tensorflow as tf

from flaxformer import types
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import embedding


class ImageToTextFeatureConverter(seqio.EncDecFeatureConverter):
  """Feature converter for an image-to-text encoder-decoder architecture."""

  TASK_FEATURES = {
      "inputs": seqio.FeatureConverter.FeatureSpec(dtype=tf.float32, rank=2),
      "targets": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens": seqio.FeatureConverter.FeatureSpec(
          dtype=tf.float32, rank=2),
      "decoder_target_tokens": seqio.FeatureConverter.FeatureSpec(
          dtype=tf.int32),
      "decoder_input_tokens": seqio.FeatureConverter.FeatureSpec(
          dtype=tf.int32),
      "decoder_loss_weights": seqio.FeatureConverter.FeatureSpec(
          dtype=tf.int32),
  }


class ImageToTextModel(models.EncoderDecoderModel):
  """ImageToTextModel."""

  FEATURE_CONVERTER_CLS = ImageToTextFeatureConverter


class ImageEncoder(t5_architecture.Encoder):
  """ImageEncoder."""

  def __call__(self,
               inputs,
               inputs_positions=None,
               encoder_mask=None,
               *,
               segment_ids=None,
               enable_dropout: bool = True):

    assert inputs.ndim == 3
    # We assume `inputs_positions` and `segment_ids` are not present because
    # (1) we do not support packing and (2) positional information is encoded as
    # the first several channels of the `inputs`.
    assert inputs_positions is None
    assert segment_ids is None

    assert encoder_mask is not None

    embedded_inputs = self.embedder(token_ids=inputs)
    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)
    encoder_outputs = self.encode_from_continuous_inputs(
        embedded_inputs,
        encoder_mask=encoder_mask,
        enable_dropout=enable_dropout)
    return encoder_outputs


class ImageEncoderTextDecoder(t5_architecture.EncoderDecoder):
  """ImageEncoderTextDecoder."""

  def setup(self):
    # Having a shared token embedder for images and text doesn't make sense.
    assert self.shared_token_embedder_factory is None
    self.token_embedder = None
    self.encoder = self.encoder_factory()
    self.decoder = self.decoder_factory()

  def _make_padding_attention_mask(self,
                                   query_tokens: types.Array,
                                   key_tokens: types.Array) -> types.Array:
    del query_tokens

    # Use padding from the positional information from the first channel to
    # detect padding.
    row_ids = key_tokens[:, :, 0].astype(jnp.int32)
    key_mask = row_ids > 0

    # Add singleton axis -3 for broadcasting to the attention heads and
    # singleton axis -2 for broadcasting to the queries.
    return jnp.expand_dims(key_mask, axis=(-3, -2)).astype(self.dtype)


class PatchEmbed(nn.Module, embedding.Embedder[types.Array]):
  """Patch embed."""
  # In addition to the patches with the pixels, the first `num_extra_embedders`
  # channels in the inputs are assumed to contain additional ids that represent
  # any metadata such as positional information.
  num_extra_embedders: int
  embedder_factory: Callable[[], embedding.Embed]
  patch_projection_factory: Callable[[], nn.Module]

  def setup(self):
    self.patch_projection = self.patch_projection_factory()
    self.embedders = [self.embedder_factory()
                      for _ in range(self.num_extra_embedders)]

  def __call__(self, inputs, **kwargs):
    # Inputs: [id_0, id_1, ..., id_{num_extra_embedders},
    #          pixel_0, pixel_1, ..., pixel_{patch_size}]
    split_inputs = jnp.split(
        inputs, np.arange(self.num_extra_embedders) + 1, -1)

    ids = split_inputs[:-1]
    patches = split_inputs[-1]
    embeddings = [embedder(i.astype(jnp.int32).squeeze(-1)) for embedder, i in
                  zip(self.embedders, ids)]
    embeddings.append(self.patch_projection(patches))
    return sum(embeddings)
