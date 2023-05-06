# CHARRED Roadmap

## Use cases

OCR
Statistical reconstitution of damaged textual artifacts. (inpainting to predict the missing characters, ex MARI)

## MlOps
https://airflow.apache.org/
https://github.com/confluentinc/confluent-kafka-python
https://cloud.google.com/tpu/docs/preemptible
https://cloud.google.com/tpu/docs/kubernetes-engine-setup
https://medium.com/@shivvidhyut/a-brief-introduction-to-distributed-training-with-gradient-descent-a4ba9faefcea
https://www.kaggle.com/code/grez911/tutorial-efficient-gradient-descent-with-jax/notebook
https://github.com/kingoflolz/mesh-transformer-jax
https://www.kubeflow.org/
https://mlflow.org/
https://metaflow.org/
https://kedro.org/
https://zenml.io/
https://www.iguazio.com/open-source/mlrun/
https://cml.dev/
https://www.seldon.io/solutions/open-source-projects/core
https://h2o.ai/platform/h2o-automl/
https://github.com/pachyderm/pachyderm

### XLA, IREE, HLO, MLIR
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

### Java pipeline
https://kafka.apache.org/ https://github.com/provectus/kafka-ui
https://kogito.kie.org/trustyai/
https://www.optaplanner.org/
https://www.drools.org/
https://vertx.io/docs/vertx-web-api-service/java/
https://vertx.io/docs/vertx-web-proxy/java/
https://vertx.io/docs/vertx-infinispan/java/
https://vertx.io/docs/vertx-stomp/java/ https://github.com/stomp-js/stompjs https://activemq.apache.org/ https://developers.cloudflare.com/queues/
https://vertx.io/docs/vertx-service-proxy/java/
https://vertx.io/docs/vertx-grpc/java/
https://vertx.io/docs/vertx-service-discovery/java/
https://github.com/deepjavalibrary/djl
https://openjdk.org/jeps/442
https://openjdk.org/jeps/448
https://github.com/apache/arrow/tree/main/java
https://github.com/apache/thrift
https://github.com/apache/avro
https://github.com/apache/orc
https://github.com/apache/parquet-mr
https://github.com/strategicblue/parquet-floor
https://github.com/apache/iceberg

### Testing
https://github.com/deepmind/chex

### HA & Telemetry
https://github.com/resilience4j/resilience4j https://resilience4j.readme.io/docs/micrometer https://vertx.io/docs/vertx-micrometer-metrics/java/
https://github.com/open-telemetry/opentelemetry-java-instrumentation
https://github.com/grafana/JPProf https://jax.readthedocs.io/en/latest/device_memory_profiling.html https://jax.readthedocs.io/en/latest/_autosummary/jax.profiler.device_memory_profile.html https://github.com/google/pprof/tree/main/proto https://github.com/grafana/pyroscope https://github.com/grafana/otel-profiling-go https://github.com/grafana/metrictank https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/extension/pprofextension
https://jax.readthedocs.io/en/latest/profiling.html https://github.com/google/perfetto/tree/master/protos
https://vertx.io/docs/vertx-opentelemetry/java/
https://vertx.io/docs/vertx-micrometer-metrics/java/

## Dataset

Make the most of cheap Kubernetes clusters: https://github.com/murphye/cheap-gke-cluster

1. Synthetic data generation: HTML/SVG/CSS graphic/layout/typography
2. Dataset merging: synthetic data, LAION-HR: https://huggingface.co/datasets/laion/laion-high-resolution, WiT dataset: https://huggingface.co/datasets/google/wit, handwritten and printed documents scans, graphemes-in-the-wild, etc.
3. Language augmentation and sampling to match ByT5's C4 training distribution so as not to loose the multilingual balance: https://huggingface.co/datasets/allenai/c4 https://huggingface.co/datasets/mc4
4. Complementary downloads from dataset URLs (mostly images) and JPEG XL archiving (see IIIF)
5. Deduplication of images with fingerprinting and of captions with sentence embeddings: https://huggingface.co/sentence-transformers/sentence-t5-base https://huggingface.co/sentence-transformers/sentence-t5-xl https://tfhub.dev/google/collections/sentence-t5 https://www.sbert.net/examples/training/multilingual/README.html https://arxiv.org/abs/2108.08877 
6. Scene segmentation, document layout understanding and caption NER. Because NER is to text what segmentation is to a scene and what layout understanding is to a document, we need to annotate all of these to be able to detect captions-within-a-caption (captions that spell out text within the image, for instance) and also score captions based on how exhaustive is the "coverage" of the scene or document they describe: https://github.com/google-research/t5x/blob/main/docs/usage/finetune.md https://huggingface.co/dbmdz/t5-base-conll03-english https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887 https://github.com/SairamNaragoni/named-entity-recognition-T5 https://github.com/MarSanTeam/Complex_NER_SemEval https://huggingface.co/docs/transformers/tasks/token_classification https://colab.research.google.com/drive/1obr78FY_cBmWY5ODViCmzdY6O1KB65Vc https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb https://www.kaggle.com/code/prithvijaunjale/t5-multi-label-classification https://pytorch.org/text/main/tutorials/t5_demo.html https://github.com/pedro-r-marques/tutorial-t5-fine-tune https://towardsdatascience.com/guide-to-fine-tuning-text-generation-models-gpt-2-gpt-neo-and-t5-dc5de6b3bc5e https://github.com/monologg/EncT5 https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb https://colab.research.google.com/github/enzoampil/t5-intro/blob/master/t5_qa_training_pytorch_span_extraction.ipynb https://programming-review.com/machine-learning/t5/ https://colab.research.google.com/drive/1syXmhEQ5s7C59zU8RtHVru0wAvMXTSQ8 https://github.com/ttengwang/Caption-Anything https://github.com/facebookresearch/segment-anything https://github.com/facebookresearch/detectron2 https://huggingface.co/datasets/wikiann https://huggingface.co/datasets/xtreme https://huggingface.co/datasets/joelito/lextreme https://huggingface.co/datasets/polyglot_ner https://huggingface.co/datasets/xglue https://huggingface.co/datasets/euronews https://huggingface.co/datasets/Babelscape/wikineural https://huggingface.co/datasets/Babelscape/multinerd https://huggingface.co/datasets/tner/multinerd https://huggingface.co/datasets/tner/wikineural https://huggingface.co/datasets/universal_dependencies
https://huggingface.co/datasets/MultiCoNER/multiconer_v2 https://surfacesyntacticud.github.io https://registry.opendata.aws/fast-ai-nlp https://registry.opendata.aws/lowcontext-ner-gaz https://registry.opendata.aws/code-mixed-ner https://registry.opendata.aws/lei/
7. Image aesthetics and caption exhaustiveness (based on #5) meaningfulness (CoLa) evaluation and filtering: https://github.com/google-research/google-research/tree/master/vila https://github.com/google-research/google-research/tree/master/musiq https://github.com/christophschuhmann/improved-aesthetic-predictor https://www.mdpi.com/2313-433X/9/2/30 https://paperswithcode.com/dataset/aesthetic-visual-analysis https://www.tandfonline.com/doi/full/10.1080/09540091.2022.2147902 https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping https://github.com/rmokady/CLIP_prefix_caption https://github.com/google-research-datasets/Image-Caption-Quality-Dataset https://github.com/gchhablani/multilingual-image-captioning https://ai.googleblog.com/2022/10/crossmodal-3600-multilingual-reference.html https://www.cl.uni-heidelberg.de/statnlpgroup/wikicaps/ https://huggingface.co/docs/transformers/main/tasks/image_captioning https://www.mdpi.com/2076-3417/13/4/2446 https://arxiv.org/abs/2201.12723 https://laion.ai/blog/laion-aesthetics/ 
8. Bucketing and batching (similar caption lengths for padding and truncating, similar image ratio for up/downsampling)
9. Images preprocessing with JAX-native methods: https://jax.readthedocs.io/en/latest/jax.image.html https://dm-pix.readthedocs.io/ https://github.com/4rtemi5/imax https://github.com/rolandgvc/flaxvision
10.  Freezed models caption text embeddings (ByT5) and image embeddings (VAE) caching

## Training

DONE: Implement JAX/FLAX SD 2.1 training pipeline with ByT5-Base instead of CLIP: https://github.com/patil-suraj/stable-diffusion-jax https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py https://huggingface.co/google/byt5-base https://huggingface.co/blog/stable_diffusion_jax
DONE: WandB monitoring
DONE: Implement Mini-SNR loss rebalancing: https://arxiv.org/abs/2303.09556
DONE: Implement on-the-fly validation: https://huggingface.co/docs/diffusers/en/conceptual/evaluation
Use CPU offloading more/better: https://github.com/google/jax/issues/1408 https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html
Better to use ByT5 XXL float32 (51.6GB), XXL bfloat16 (26GB), or XL float32 (15GB) ? https://huggingface.co/google/byt5-xxl https://github.com/google-research/t5x/blob/main/docs/models.md#byt5-checkpoints https://github.com/google-research/t5x/blob/main/t5x/scripts/convert_tf_checkpoint.py
Redo ByT5 with innovations on models from the T5 family or other character-aware models: Switch/MOE (https://arxiv.org/abs/2101.03961, https://github.com/google-research/t5x/tree/main/t5x/contrib/moe, https://towardsdatascience.com/understanding-googles-switch-transformer-904b8bf29f66, https://huggingface.co/google/switch-c-2048, https://towardsdatascience.com/the-switch-transformer-59f3854c7050, https://arxiv.org/abs/2208.02813), FLAN (https://arxiv.org/abs/2210.11416, https://arxiv.org/abs/2301.13688, https://arxiv.org/abs/2109.01652), UL2 (https://arxiv.org/abs/2205.05131v3, https://github.com/google-research/google-research/tree/master/ul2, https://ai.googleblog.com/2022/10/ul2-20b-open-source-unified-language.html), T5 v1.1 (https://arxiv.org/abs/2002.05202), CANINE-C (https://arxiv.org/abs/2103.06874, https://github.com/google-research/language/tree/master/language/canine, https://huggingface.co/google/canine-c, https://huggingface.co/vicl/canine-c-finetuned-cola), CALM (
https://github.com/google-research/t5x/tree/main/t5x/contrib/calm, https://arxiv.org/abs/2207.07061), FlashAttention (https://github.com/HazyResearch/flash-attention, https://arxiv.org/abs/2205.14135), Gradient Checkpointing (https://arxiv.org/abs/1604.06174), FastT5 (https://github.com/Ki6an/fastT5), T5x & Seqio (https://arxiv.org/abs/2203.17189), LongT5 (https://github.com/google-research/longt5), WT5 (https://github.com/google-research/google-research/tree/master/wt5)
OpenTelemetry monitoring including JAX profiler tracing artifact uploading
Integrate Big Vision optimizations: https://github.com/google-research/big_vision
Implement streaming, mini-batching and gradient accumulation with image aspect ratio and tokenized caption size bucketing: https://github.com/NovelAI/novelai-aspect-ratio-bucketing https://optax.readthedocs.io/en/latest/gradient_accumulation.html https://optax.readthedocs.io/en/latest/api.html#optax.MultiSteps

Port to JAX and Integrate Imagen, SDXL and Deep Floyd improvements: https://github.com/lucidrains/imagen-pytorch https://github.com/deep-floyd/IF https://stable-diffusion-art.com/sdxl-beta/ https://huggingface.co/docs/diffusers/api/pipelines/if https://huggingface.co/spaces/DeepFloyd/IF https://huggingface.co/DeepFloyd/IF-I-XL-v1.0 https://huggingface.co/DeepFloyd/IF-II-L-v1.0 https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler https://huggingface.co/DeepFloyd/IF-notebooks/tree/main https://huggingface.co/blog/if https://huggingface.co/docs/diffusers/main/en/api/pipelines/if https://stability.ai/blog/deepfloyd-if-text-to-image-model https://deepfloyd.ai/ https://www.assemblyai.com/blog/how-imagen-actually-works/ https://www.youtube.com/watch?v=af6WPqvzjjk https://www.youtube.com/watch?v=xqDeAz0U-R4 https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/
Save checkpoints using JAX-native methods: https://flax.readthedocs.io/en/latest/api_reference/flax.training.html https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#save-checkpoints


## Inference

DONE: Implement JAX/FLAX text-to-image inference pipeline with ByT5-Base instead of CLIP: https://huggingface.co/docs/diffusers/training/text2image https://github.com/patil-suraj/stable-diffusion-jax
Production AOT with IREE over Java JNI/JNA/Panama: https://github.com/openxla/iree https://github.com/iree-org/iree-jax https://jax.readthedocs.io/en/latest/aot.html https://jax.readthedocs.io/en/latest/_autosummary/jax.make_jaxpr.html https://jax.readthedocs.io/en/latest/_autosummary/jax.xla_computation.html https://github.com/openxla/stablehlo https://github.com/openxla/xla https://github.com/openxla/openxla-pjrt-plugin https://github.com/iml130/iree-template-cpp
Load checkpoints using JAX-native methods https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#id1
Implement OCR and Document understanging inference pipeline with ByT5 text decoder
Implement text encoding CPU offloading with int8 precision
Implement accelerated U-Net prediction and VAE decoding with int8 precision : https://github.com/TimDettmers/bitsandbytes https://huggingface.co/blog/hf-bitsandbytes-integration

## Exploration

https://github.com/google-research/google-research/tree/master/vrdu
https://github.com/google-research/google-research/tree/master/invariant_explanations
https://github.com/google-research/google-research/tree/master/pali
https://laion.ai/blog/paella/
https://laion.ai/blog/open-flamingo/
https://laion.ai/blog/datacomp/
https://aclanthology.org/2022.semeval-1.226/
https://github.com/gsarti/t5-flax-gcp
https://github.com/PiotrNawrot/nanoT5
https://huggingface.co/mesolitica/finetune-dependency-t5-base-standard-bahasa-cased
https://arxiv.org/abs/2208.14536
https://github.com/Alibaba-NLP/KB-NER https://github.com/modelscope/AdaSeq
https://github.com/mckysse/gain