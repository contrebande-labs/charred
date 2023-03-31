# CHARacter-awaRE Diffusion: Multilingual Character-Aware Encoders for Font-Aware Diffusers That Can Actually Spell

Tired of text-to-image models that can't spell or deal with fonts and typography correctly ? [The secret seems to be in the use of multilingual, tokenization-free, character-aware transformer encoders](https://arxiv.org/abs/2212.10562) such as [ByT5](https://arxiv.org/abs/2105.13626) and [CANINE-c](https://arxiv.org/abs/2103.06874). To this, and as per [Peter von Platen](https://github.com/patrickvonplaten)'s suggestion, we also introduce a typographic [ControlNet](https://arxiv.org/abs/2302.05543) trained on an automatically-generated dataset of rasterized text layout images paired with multilingual specifications of the font, weight, kerning, leading, slant and any other typographic attributes supported by the [CSS3](https://www.w3.org/Style/CSS/) [Text](https://www.w3.org/TR/css-text-3/), [Fonts](https://www.w3.org/TR/css-fonts-3) & [Writing Modes](https://www.w3.org/TR/css-writing-modes-3/) modules as implemented by the latest version of [Chromium](https://www.chromium.org/Home/).

The short term realistic objective is to first port the [Stable UnCLIP 2.1](https://arxiv.org/abs/2204.06125) [Karlo](https://github.com/kakaobrain/karlo)-[derived](https://github.com/Stability-AI/stablediffusion/blob/main/doc/UNCLIP.MD) [code base](https://github.com/Stability-AI/stablediffusion) to [JAX](https://github.com/google/jax)/[FLAX](https://github.com/google/flax), and replacing [CLIP](https://arxiv.org/abs/2103.00020)'s [text encoder implementation](https://github.com/openai/CLIP) with [ByT5](https://github.com/google-research/t5x/tree/main/t5x/examples/t5/byt5)'s (all the while keeping an eye on the [legacy ByT5 impl](https://github.com/google-research/byt5)). Secondly, we will integrate the [Hugging-Face JAX/FLAX ControlNet implementation](https://github.com/huggingface/community-events/tree/main/jax-controlnet-sprint) into the mix for better typographic control over the generated images.

In the long term, we shall explore the applicable improvements brought on by [Imagen](https://arxiv.org/abs/2205.11487), CANINE-c & the likes to the aforementioned technologies.
