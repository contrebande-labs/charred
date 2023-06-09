# CHARRED Roadmap

## Use cases

Artistic rendition of missing, loss, or stolen textual artifacts (text-to-image), OCR (image-to-text), statistical reconstitution of damaged textual artifacts (image-to-image, inpainting to predict the missing characters, ex: MARI).

Low-resource languages, low-resource domains. (ex: perfumery)

## Training

Are we generating the input embeddings correctly?

FlaxT5PreTrainedModel.encode(): input_ids=jnp.array(input_ids, dtype="i4") ? uint32 Unicode char or uint8 UTF-8 byte ?
array = np.frombuffer(string.encode("UTF-8", errors="ignore"), dtype=np.uint8) + 3
string = (ndarray - 3).tobytes().decode("UTF-8", errors="ignore")

Use T5x lib instead of "transformers"

Should the shape of the latents in the VAE/UNet be bigger to accomodate for more tokens ?

Would it be possible to have a vocab-less, raw UTF-8 byte character aware decoder-only language model ?

Write the tests first: https://github.com/deepmind/chex

VAE/U-Net hyperparameters to accommodate byt5's character-awareness better

Try to run original code

1. DONE: Implement JAX/FLAX SD 2.1 training pipeline with ByT5-Base instead of CLIP: https://github.com/patil-suraj/stable-diffusion-jax https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py https://huggingface.co/google/byt5-base https://huggingface.co/blog/stable_diffusion_jax
2. DONE: WandB monitoring
3. DONE: Implement Mini-SNR loss rebalancing: https://arxiv.org/abs/2303.09556
4. DONE: Implement on-the-fly validation: https://huggingface.co/docs/diffusers/en/conceptual/evaluation
5. DONE: save checkpoints to disk
6. Get rid of PytorchDataloader Flax Linen and HF libraries (transformers, diffusers, datasets), use JAX's new Array code, and write pure functions
7. Make the code independent from device topology (might be hardcoded to 8xTPUv4 at the moment)
8. Implement streaming from the network (instead of from the disk), mini-batching and gradient accumulation with image aspect ratio and tokenized caption size bucketing. Freezed models caption text embeddings (ByT5) and image embeddings (VAE) caching with bfloat16 half-precision (ByT5 and VAE) and explore using ByT5 XXL float32 (51.6GB), XXL bfloat16 (26GB), or XL float32 (15GB) and discard anything unnecessary from the freezed models (eg: ByT5 decoder weights) to lower the memory requirements: https://github.com/google/jax/issues/1408 https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html https://huggingface.co/google/byt5-xxl https://github.com/google-research/t5x/blob/main/docs/models.md#byt5-checkpoints https://github.com/google-research/t5x/blob/main/t5x/scripts/convert_tf_checkpoint.py https://optax.readthedocs.io/en/latest/gradient_accumulation.html https://optax.readthedocs.io/en/latest/api.html#optax.MultiSteps
9. Better strategy to load and save checkpoints using JAX-native methods: https://flax.readthedocs.io/en/latest/api_reference/flax.training.html https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#save-checkpoints https://arxiv.org/abs/1604.06174

## Inference

1. DONE: Implement JAX/FLAX text-to-image inference pipeline and Gradio demo with ByT5-Base instead of CLIP: https://huggingface.co/docs/diffusers/training/text2image https://github.com/patil-suraj/stable-diffusion-jax
2. Implement AUTOMATIC1111 and Gradio UIs: https://github.com/AUTOMATIC1111/stable-diffusion-webui
3. Load checkpoints using JAX-native methods https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html
4. Implement OCR and Document understanging inference pipeline with ByT5 text decoder
5. Implement text encoding CPU offloading with int8 precision and GPU-accelerated U-Net prediction and VAE decoding with int8 precision https://jax.readthedocs.io/en/latest/multi_process.html https://github.com/TimDettmers/bitsandbytes https://huggingface.co/blog/hf-bitsandbytes-integration

## MLOps

### XLA, IREE, HLO, MLIR

https://medium.com/@shivvidhyut/a-brief-introduction-to-distributed-training-with-gradient-descent-a4ba9faefcea
https://www.kaggle.com/code/grez911/tutorial-efficient-gradient-descent-with-jax/notebook
https://github.com/kingoflolz/mesh-transformer-jax
https://github.com/kingoflolz/swarm-jax
https://github.com/openxla/iree https://openxla.github.io/iree/
https://github.com/openxla/xla/blob/main/xla/xla.proto
https://github.com/openxla/xla/blob/main/xla/xla_data.proto
https://github.com/openxla/xla/blob/main/xla/service/hlo.proto
https://github.com/openxla/xla/tree/main/third_party/tsl/tsl/protobuf
https://github.com/openxla/xla/blob/main/xla/pjrt/distributed/protocol.proto
https://github.com/apache/tvm/
https://github.com/openai/triton
https://github.com/onnx/onnx-mlir
https://github.com/plaidml/plaidml
https://github.com/llvm/torch-mlir
https://github.com/pytorch/pytorch/tree/main/torch/_dynamo
https://github.com/pytorch/pytorch/tree/main/torch/_inductor
https://github.com/llvm/llvm-project/tree/main/mlir/ https://mlir.llvm.org/
https://github.com/tensorflow/mlir-hlo
https://github.com/openxla/stablehlo
https://github.com/llvm/torch-mlir
https://research.google/pubs/pub48035/
https://iq.opengenus.org/mlir-compiler-infrastructure/
https://www.youtube.com/watch?v=Z8knnMYRPx0 https://mlir.llvm.org/OpenMeetings/2023-03-23-Nelli.pdf
https://mlir.llvm.org/docs/Tutorials/Toy/
https://mlir.llvm.org/getting_started/
Production AOT with IREE over Java JNI/JNA/Panama https://github.com/openxla/iree https://github.com/iree-org/iree-jax https://jax.readthedocs.io/en/latest/aot.html https://jax.readthedocs.io/en/latest/_autosummary/jax.make_jaxpr.html https://jax.readthedocs.io/en/latest/_autosummary/jax.xla_computation.html https://github.com/openxla/stablehlo https://github.com/openxla/xla https://github.com/openxla/openxla-pjrt-plugin https://github.com/iml130/iree-template-cpp
hlo/mlir compiler/interpreter/simulator/emulator in java: https://github.com/oracle/graal/tree/master/sulong https://github.com/oracle/graal/tree/master/visualizer https://github.com/oracle/graal/tree/master/truffle https://github.com/graalvm/simplelanguage https://github.com/graalvm/simpletools https://openjdk.org/jeps/442 https://openjdk.org/jeps/448

### Java pipeline

https://vertx.io/docs/vertx-web-api-service/java/
https://github.com/vert-x3/vertx-infinispan https://github.com/infinispan/infinispan
https://github.com/eclipse-vertx/vertx-grpc
https://github.com/vert-x3/vertx-service-discovery
https://github.com/vert-x3/vertx-service-proxy

https://kafka.apache.org/ https://github.com/provectus/kafka-ui
https://github.com/vert-x3/vertx-kafka-client
https://github.com/vert-x3/vertx-stomp https://github.com/stomp-js/stompjs https://activemq.apache.org/ https://developers.cloudflare.com/queues/
https://github.com/apache/pulsar
https://github.com/datastax/kafka-examples
https://github.com/datastax/kafka-sink
https://github.com/datastax/starlight-for-kafka

https://github.com/apache/arrow/tree/main/java
https://github.com/apache/thrift
https://github.com/apache/avro
https://github.com/apache/orc
https://github.com/apache/parquet-mr
https://github.com/msgpack/msgpack-java
https://github.com/irmen/pickle
https://github.com/jamesmudd/jhdf

https://github.com/apache/iceberg
https://github.com/eclipse/jnosql
https://github.com/trinodb/trino
https://github.com/apache/druid/
https://github.com/apache/hudi
https://github.com/delta-io/delta
https://github.com/apache/pinot

### HA & Telemetry

OpenTelemetry/Graphana monitoring instead of WandB, Perfetto or Tensorbord, attach JAX profiler artifacts
https://github.com/resilience4j/resilience4j https://resilience4j.readme.io/docs/micrometer https://vertx.io/docs/vertx-micrometer-metrics/java/
https://github.com/open-telemetry/opentelemetry-java-instrumentation
https://github.com/grafana/JPProf https://jax.readthedocs.io/en/latest/device_memory_profiling.html https://jax.readthedocs.io/en/latest/_autosummary/jax.profiler.device_memory_profile.html https://github.com/google/pprof/tree/main/proto https://github.com/grafana/pyroscope https://github.com/grafana/otel-profiling-go https://github.com/grafana/metrictank https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/extension/pprofextension
https://jax.readthedocs.io/en/latest/profiling.html https://github.com/google/perfetto/tree/master/protos
https://vertx.io/docs/vertx-opentelemetry/java/
https://vertx.io/docs/vertx-micrometer-metrics/java/

## Dataset preprocessing

Make the most of cheap Kubernetes clusters: https://github.com/murphye/cheap-gke-cluster

1. Synthetic data generation: HTML/SVG/CSS graphic/layout/typography
2. Dataset merging: synthetic data, LAION-HR: https://huggingface.co/datasets/laion/laion-high-resolution, WiT dataset: https://huggingface.co/datasets/google/wit https://huggingface.co/datasets/wikimedia/wit_base, handwritten and printed documents scans, graphemes-in-the-wild, etc. with language re-sampling to match ByT5's C4 training distribution so as not to loose the multilingual balance: https://huggingface.co/datasets/allenai/c4 https://huggingface.co/datasets/mc4. More image datasets: https://huggingface.co/datasets/facebook/winoground https://huggingface.co/datasets/huggan/wikiart https://huggingface.co/datasets/kakaobrain/coyo-700m https://github.com/unsplash/datasets https://huggingface.co/datasets/red_caps https://huggingface.co/datasets/gigant/oldbookillustrations
3. Complementary downloads from dataset URLs (mostly images) and JPEG XL archiving (see IIIF)
4. Deduplication of images with fingerprinting and of captions with sentence embeddings (all the sentence-transformers disappeared on May 8 2023)
5. Scene segmentation, document layout understanding and caption NER. Because NER is to text what segmentation is to a scene and what layout understanding is to a document, we need to annotate all of these to be able to detect captions-within-a-caption (captions that spell out text within the image, for instance) and also score captions based on how exhaustive is the "coverage" of the scene or document they describe: https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887 https://huggingface.co/docs/transformers/model_doc/flan-ul2 https://pytorch.org/text/main/tutorials/t5_demo.html https://towardsdatascience.com/guide-to-fine-tuning-text-generation-models-gpt-2-gpt-neo-and-t5-dc5de6b3bc5e https://programming-review.com/machine-learning/t5/ https://colab.research.google.com/drive/1syXmhEQ5s7C59zU8RtHVru0wAvMXTSQ8 https://github.com/ttengwang/Caption-Anything https://github.com/facebookresearch/segment-anything https://github.com/facebookresearch/detectron2 https://huggingface.co/datasets/joelito/lextreme ttps://registry.opendata.aws/lei/ https://huggingface.co/datasets/jfrenz/legalglue https://huggingface.co/datasets/super_glue https://huggingface.co/datasets/klue https://huggingface.co/datasets/NbAiLab/norne https://huggingface.co/datasets/indic_glue https://huggingface.co/datasets/acronym_identification https://huggingface.co/datasets/wikicorpus https://huggingface.co/datasets/multi_woz_v22 https://huggingface.co/datasets/wnut_17 https://huggingface.co/datasets/msra_ner https://huggingface.co/datasets/conll2012_ontonotesv5 https://huggingface.co/datasets/conllpp
6. Image aesthetics and caption exhaustiveness (based on #5) meaningfulness (CoLa) evaluation and filtering: https://github.com/google-research/google-research/tree/master/vila https://github.com/google-research/google-research/tree/master/musiq https://github.com/christophschuhmann/improved-aesthetic-predictor https://www.mdpi.com/2313-433X/9/2/30 https://paperswithcode.com/dataset/aesthetic-visual-analysis https://www.tandfonline.com/doi/full/10.1080/09540091.2022.2147902 https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping https://github.com/rmokady/CLIP_prefix_caption https://github.com/google-research-datasets/Image-Caption-Quality-Dataset https://github.com/gchhablani/multilingual-image-captioning https://ai.googleblog.com/2022/10/crossmodal-3600-multilingual-reference.html https://www.cl.uni-heidelberg.de/statnlpgroup/wikicaps/ https://huggingface.co/docs/transformers/main/tasks/image_captioning https://www.mdpi.com/2076-3417/13/4/2446 https://arxiv.org/abs/2201.12723 https://laion.ai/blog/laion-aesthetics/ https://github.com/JD-P/simulacra-aesthetic-captions
7. Bucketing and batching (similar caption lengths for padding and truncating, similar image ratio for up/downsampling): https://github.com/NovelAI/novelai-aspect-ratio-bucketing
8. Images preprocessing with JAX-native methods: https://jax.readthedocs.io/en/latest/jax.image.html https://dm-pix.readthedocs.io/ https://github.com/4rtemi5/imax https://github.com/rolandgvc/flaxvision

## CharT5 (ByT5 v2)

Pretrain a better ByT5 with innovations from the T5 family and other character-aware language transformer models:

- Early character-aware language models: https://arxiv.org/abs/1508.06615 https://arxiv.org/abs/2011.01513
- CANINE-C encoder only character-aware language model (https://arxiv.org/abs/2103.06874, https://github.com/google-research/language/tree/master/language/canine, https://huggingface.co/google/canine-c, https://huggingface.co/vicl/canine-c-finetuned-cola)
- Switch/MOE https://arxiv.org/abs/2101.03961 https://github.com/google-research/t5x/tree/main/t5x/contrib/moe https://towardsdatascience.com/understanding-googles-switch-transformer-904b8bf29f66, https://huggingface.co/google/switch-c-2048, https://towardsdatascience.com/the-switch-transformer-59f3854c7050 https://arxiv.org/abs/2208.02813
- FLAN/PALM/PALM-E/PALM 2/UL2 https://arxiv.org/abs/2210.11416 https://arxiv.org/abs/2301.13688 https://arxiv.org/abs/2109.01652 https://github.com/lucidrains/PaLM-jax https://github.com/conceptofmind/PaLM-flax https://github.com/google-research/t5x/tree/main/t5x/examples/decoder_only https://huggingface.co/docs/transformers/main/model_doc/flan-t5 https://huggingface.co/google/flan-ul2 https://arxiv.org/abs/2205.05131v3 https://github.com/google-research/google-research/tree/master/ul2 https://ai.googleblog.com/2022/10/ul2-20b-open-source-unified-language.html
- T5 v1.1 https://arxiv.org/abs/2002.05202
- CALM https://github.com/google-research/t5x/tree/main/t5x/contrib/calm https://arxiv.org/abs/2207.07061
- FlashAttention https://github.com/HazyResearch/flash-attention https://arxiv.org/abs/2205.14135
- T5x & Seqio https://arxiv.org/abs/2203.17189
- LongT5 https://github.com/google-research/longt5
- WT5 https://github.com/google-research/google-research/tree/master/wt5
- NanoT5 https://github.com/PiotrNawrot/nanoT5
- Tensor Considered Harmful https://nlp.seas.harvard.edu/NamedTensor https://github.com/stanford-crfm/levanter https://github.com/harvardnlp/NamedTensor
- FasterTransformers https://github.com/NVIDIA/FasterTransformer
- FlaxFormers https://github.com/google/flaxformer

# Beyond SD 2.1

Integrate and port to JAX as much improvements and ideas from Imagen, SDXL, Deep Floyd, Big Vision, Vision Transformer, etc. as possible : https://github.com/lucidrains/imagen-pytorch https://github.com/deep-floyd/IF https://stable-diffusion-art.com/sdxl-beta/ https://huggingface.co/docs/diffusers/api/pipelines/if https://huggingface.co/spaces/DeepFloyd/IF https://huggingface.co/DeepFloyd/IF-I-XL-v1.0 https://huggingface.co/DeepFloyd/IF-II-L-v1.0 https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler https://huggingface.co/DeepFloyd/IF-notebooks/tree/main https://huggingface.co/blog/if https://huggingface.co/docs/diffusers/main/en/api/pipelines/if https://stability.ai/blog/deepfloyd-if-text-to-image-model https://deepfloyd.ai/ https://www.assemblyai.com/blog/how-imagen-actually-works/ https://www.youtube.com/watch?v=af6WPqvzjjk https://www.youtube.com/watch?v=xqDeAz0U-R4 https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/ https://github.com/google-research/big_vision https://github.com/google-research/vision_transformer https://github.com/microsoft/unilm/tree/master/beit https://arxiv.org/abs/2106.04803 https://arxiv.org/abs/2210.01820 https://arxiv.org/abs/2103.15808 https://arxiv.org/abs/2201.10271 https://arxiv.org/abs/2209.15159 https://arxiv.org/abs/2303.14189 https://arxiv.org/abs/2010.11929 https://arxiv.org/abs/2208.10442 https://arxiv.org/abs/2012.12877 https://arxiv.org/abs/2111.06377v3 https://arxiv.org/abs/2107.06263 https://arxiv.org/abs/1906.00446 https://arxiv.org/abs/2110.04627 https://arxiv.org/abs/2208.06366 https://arxiv.org/abs/2302.00902 https://arxiv.org/abs/2212.03185 https://arxiv.org/abs/2212.07372 https://arxiv.org/abs/2209.09002 https://arxiv.org/abs/2301.00704 https://arxiv.org/abs/2211.09117 https://arxiv.org/abs/2302.05917
