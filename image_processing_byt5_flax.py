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

from transformers.utils import is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL


# TODO: hard-code values from: https://huggingface.co/openai/clip-vit-base-patch32/blob/main/preprocessor_config.json
# TODO: JAXify the code in CLIPImageProcessor (nice to have)
class ByT5ImageProcessor(CLIPImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)