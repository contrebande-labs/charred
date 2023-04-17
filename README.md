<h1 style="text-align:center"><img
src="https://repository-images.githubusercontent.com/621984782/64f865ef-5858-4ce0-bdcc-c51f53552545"
     alt="Markdown Monster icon" /></h1>

# CHARacter-awaRE Diffusion: Multilingual Character-Aware Encoders for Font-Aware Diffusers That Can Actually Spell

Tired of text-to-image models that can't spell or deal with fonts and typography correctly ? [The secret seems to be in the use of multilingual, tokenization-free, character-aware transformer encoders](https://arxiv.org/abs/2212.10562) such as [ByT5](https://arxiv.org/abs/2105.13626) and [CANINE-c](https://arxiv.org/abs/2103.06874).

## Replace CLIP with ByT5 in HF's `text-to-image` Pipeline

AS part of the [Hugging Face JAX Diffuser Sprint](https://github.com/huggingface/community-events/tree/main/jax-controlnet-sprint), we will replace [CLIP](https://arxiv.org/abs/2103.00020)'s tokenizer and encoder with [ByT5](https://arxiv.org/abs/2105.13626)'s in the [HF's JAX/FLAX text-to-image pre-training code](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py) and run it on the sponsored TPU ressources provided by Google for the event.

More specifically, here are the main tasks we will try to accomplish during the sprint:

- Pre-training dataset preparation: we are NOT going to train on `lambdalabs/pokemon-blip-captions`. So what is it going to be, what are the options? [Anything in here](https://analyticsindiamag.com/top-used-datasets-for-text-to-image-synthesis-models/) or [here](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image#head3) takes your fancy? Or maybe [DiffusionDB](https://poloclub.github.io/diffusiondb/)? Or a savant mix of many datasets? We probably will need to combine many datasets as we are looking to cover these requirements:
  - We need samples for which there is text in the scene that is explicitely specified in the caption and the priority is to do that in full scene photos. If we can't find enough, we will integrate more specialized datasets for OCR;
  - Approximately the same language distribution as ByT5, but also include indonesian (not in ByT5) to see how character-awareness works when text in the prompt is specified in a language. We need to build testing facilities around the languages that are spoken by team members and friends: indonesian, japanese, french, amharic, arabic, norwegian, swedish, hindi, urdu and english.
- Improvements to the [original code](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py):
  - ~~Make sure we can run the original code as-is on the TPU VM.~~
  - Audit and optimize the code for the Google Cloud TPU v4-8 VM: [`jnp`](https://jax.readthedocs.io/en/latest/jax.numpy.html) (instead of np) [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), [`pmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html), [`pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html) everywhere! And we should make sure we do not miss any [optimization made in the sprint code](https://github.com/huggingface/community-events/blob/main/jax-controlnet-sprint/training_scripts/train_controlnet_flax.py ) either.
  - Instrumentation for TPU remote monitoring with [Open Telemetry](https://opentelemetry.io/docs/instrumentation/python/), [TensorBoard](https://www.tensorflow.org/tensorboard/), [Perfetto](https://perfetto.dev), [Weights & Biases](https://wandb.ai) and [JAX's own profiler](https://jax.readthedocs.io/en/latest/profiling.html).
  - Implement checkpoint milestone snapshot uploading to cloud storage: we need to be able to download the model for local inference benchmarking to make sure we are on the right track. There seems to be [rudimentary checkpoint support in the original code](https://huggingface.co/docs/diffusers/training/text2image#save-and-load-checkpoints).
  - ~~No time for politics. NSFW filtering will be turned off. So we get `FlaxStableDiffusionSafetyChecker` out of the way.~~
- Replace CLIP with ByT5 in [original code](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py):
  - ~~Replacing `CLIPTokenizer` with `ByT5Tokenizer`. Since this will run on the CPUs, there is no need for JAX/FLAX unless there is hope for huge performance improvements. This should be trivial.~~ Merged. Needs testing.
  - ~~Replacing `FlaxCLIPTextModel` with `FlaxT5EncoderModel`. This *might* be almost as easy as replacing the tokenizer.~~ Merged. Needs testing.
  - ~~Rewrite `CLIPImageProcessor` for ByT5. This is still under investigation. It's unclear how hard it will be.~~ Done. Needs testing.
  - ~~RAdapt `FlaxAutoencoderKL` and `FlaxUNet2DConditionModel` for ByT5 if necessary.~~ Done. Needs testing.
  - ~~Break down the main pretraining loop into many functions in different source files for readability and easier maintenance.~~

## Introducing a Calligraphic & Typographic ControlNet

Secondly, we will integrate to the above a [Hugging-Face JAX/FLAX ControlNet implementation](https://github.com/huggingface/diffusers/tree/main/examples/controlnet) for better typographic control over the generated images. To the orthographically-enanced SD above and as per [Peter von Platen](https://github.com/patrickvonplaten)'s suggestion, we also introduce the idea a typographic [ControlNet](https://arxiv.org/abs/2302.05543) trained on an synthetic dataset of images paired with multilingual specifications of the textual content, font taxonomy, weight, kerning, leading, slant and any other typographic attribute supported by the [CSS3](https://www.w3.org/Style/CSS/) [Text](https://www.w3.org/TR/css-text-3/), [Fonts](https://www.w3.org/TR/css-fonts-3) and [Writing Modes](https://www.w3.org/TR/css-writing-modes-3/) modules, as implemented by the latest version of [Chromium](https://www.chromium.org/Home/).
