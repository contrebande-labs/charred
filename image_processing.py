# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for ByT5."""

from transformers import CLIPImageProcessor

# TODO: change that for JAX equivalents
from torchvision import transforms

# TODO: hard-code values from: https://huggingface.co/openai/clip-vit-base-patch32/blob/main/preprocessor_config.json
# TODO: JAXify the code in CLIPImageProcessor (nice to have)
class ByT5ImageProcessor(CLIPImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def setup_train_transforms(args):
  return transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )