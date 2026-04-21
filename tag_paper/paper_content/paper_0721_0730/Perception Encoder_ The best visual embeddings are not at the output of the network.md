Title: Perception Encoder: The best visual embeddings are not at the output of the network
Abstract: We introduce Perception Encoder (PE), a family of state-of-the-art vision encoders for image and video understanding. Traditionally, vision encoders have relied on a variety of pretraining objectives, each excelling at different downstream tasks. Surprisingly, after scaling a carefully tuned image pretraining recipe and refining with a robust video data engine, we find that contrastive vision-language training alone can produce strong, general embeddings for all of these downstream tasks. There is only one caveat: these embeddings are hidden within the intermediate layers of the network. To draw them out, we introduce two alignment methods: language alignment for multimodal language modeling, and spatial alignment for dense prediction. Together, our PE family of models achieves state-of-the-art results on a wide variety of tasks, including zero-shot image and video classification and retrieval; document, image, and video Q&A; and spatial tasks such as detection, tracking, and depth estimation. We release our models, code, and novel dataset of synthetically and human-annotated videos: https://github.com/facebookresearch/perception_models Category: Animals Caption: The video shows a white and gray adult cat and two kittens. The adult cat is grooming the kitten closest to it with its tongue, and the kitten is looking around. A hand reaches out from the frame's upper left to pet the two kittens. Category: Outdoor ScenesCaption: The video shows a tall, pointed structure in the middle of a field. and the structure is surrounded by trees and other vegetation. The field is divided into sections, with some areas covered in green grass and others covered in white material. The video shows the structure and the field from a distance, with the camera moving around it. Category: Work ActivitiesCaption: The video shows a person using a shovel to clean the ashes from a fireplace. They are scooping up the ashes and removing them from the fireplace. Category: Food Preparation Caption:The video shows a person cutting an green color item into small pieces. They are using a knife to slice the pickle into thin pieces, and then chopping those pieces into smaller cubes. The person is working on a wooden cutting board, and the Hands are visible from the left side of the frame with pink nail paint on their nails. Category: Object Interactions Caption:The video shows a black and white spiral that is spinning. The spiral is made up of alternating black and white stripes that are evenly spaced and symmetrical. Category: Hand ActionsCaption: The video captures a closeup shot of person typing on a keyboard. The camera moves from the left side of the keyboard to the right, an animation of the revolving globe and some numbers can be seen in the frame and the video ends. Category: Object HandlingCaption: The video shows a person putting a bowl of something into an oven. The person then closes the oven door. The background is blurry. Category: Water ScenesCaption: The video shows a large school of fish swimming in a water body towards the right frame. The camera too pans a little to the right. Category: Nature ScenesCaption: The video shows a pile of branches and leaves on fire in a field. The fire is burning brightly, with flames licking at the edges of the pile. The smoke from the fire rises into the air, billowing up into the sky.

Section: Introduction
For the last decade in computer vision, pretrained vision encoders have been the core building block for most applications requiring perception. From million-scale ImageNet [25] pretrained convolutional networks [41,60,79,121,128] to billion-scale web-pretrained transformers [19,24,28,53,127,156], the dominant strategy in vision has been to adapt large-scale pretrained encoders to downstream tasks. Today, these pretraining objectives come in several flavors: vision-language contrastive losses [103,158] learn a global vision and language embedding well-suited for zero-shot classification and retrieval as well as provide vision-language alignment for open-world [68,92] and generative tasks [105,111]; captioning losses [36,134] learn to predict image descriptions using a language decoder, which transfers well to downstream multimodal language model (MLLM) tasks; and spatially self-supervised losses [43,96] learn dense spatial correspondences without language supervision, making them useful for tasks requiring precise localization like object detection. Many works are now attempting to combine two or more of these techniques in different ways [19,33,34,36,44,88,107,156]. While many have been successful, the complexity of these strategies grows exponentially with number of use cases, which can make scaling difficult. There has not yet been shown a single, simple, and easily scalable pretraining technique that can learn state-of-the-art features for all downstream tasks.
In this work, we discover that global vision-language contrastive learning alone can be one such approach. We begin by building PE core (Fig. 1, left), a large-scale contrastively pretrained model with state-of-the-art zero-shot performance on both image and video ( §2). To accomplish this, we first focus on developing a strong image-only contrastive pretraining recipe to extract general knowledge from billion-scale image-text data ( §2.1). We then use the resulting model as a frame-based encoder to develop a video data engine ( §2.2) for generating well-aligned video captions. Finetuning on this synthetic video-text data substantially improves performance on both image and video classification 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Image Classify Images # Class Idx Classify Videos # Class Idx PE Core PE Language OCR Q&A T Text Captioning T Text Video Q&A T Text Grounding Box PE Spatial Detect Box Estimate Depth Depth Map Track Masklet Segment Mask Alignment Tuning Large-Scale Contrastive Pretraining State-of-the-Art Language Encoder State-of-the-Art Spatial Encoder §2 §3 §4 §5
Figure 1: Perception Encoder (PE) is a family of large-scale vision encoder models with stateof-the-art performance on a large variety of vision tasks. By using a robust contrastive pretraining recipe and finetuning on synthetically aligned videos, PE not only outperforms all existing models on classification and retrieval ( §2), but it also internally produces strong, general features that scale for downstream tasks ( §3). PE unlocks the ability for large-scale contrastive pretraining to transfer to downstream tasks with alignment tuning to capitalize on those general features ( §4, §5). and retrieval tasks. Finally, we scale our robust image pretraining and well-aligned video finetuning strategy to 2B parameters to produce PE core G ( §2.3), a single unified encoder that outperforms SigLIP2 [135] on zero-shot image tasks and InternVideo2 [143] on most zero-shot video tasks.
After analyzing the performance of PE core G, we found a surprising result: inside the model were specific features aligned to OCR, VQA, grounding, detection, depth estimation, and tracking ( §3). Compared to the state-of-the-art models with captioning [36] and spatially self-supervised [96] pretraining, our contrastive encoder has specific layers that, when used as frozen features, matches or exceeds the performance of the other two pretraining techniques on tasks they should be the best at.
The only problem is-these features exist at different layers for each task.
By exploiting this phenomenon with alignment tuning (Fig. 1, right), we show it is possible to align these features to the end of the network to create state-of-the-art encoders for downstream MLLM ( §4) and spatial ( §5) tasks-all following the same easily scalable contrastive pretraining. Thus, Perception Encoder unlocks the potential to scale one simple pretraining method to solve many downstream vision tasks. We will release our models, code, and novel PE Video Dataset of 1M high-quality stock footage videos and 120K human-refined captions.
this section cite: ['b24', 'b40', 'b59', 'b78', 'b120', 'b127', 'b18', 'b23', 'b27', 'b52', 'b126', 'b155', 'b102', 'b157', 'b67', 'b91', 'b104', 'b110', 'b35', 'b133', 'b42', 'b95', 'b18', 'b32', 'b33', 'b35', 'b43', 'b87', 'b106', 'b155', 'b134', 'b142', 'b35', 'b95']

Section: Perception Encoder: Core
To build Perception Encoder (PE), we start by training a large-scale, robust, and performant visionlanguage contrastive model for image and video. We have two objectives: to enhance the scalability and data efficiency of contrastive training, and to create a unified model for image and video.
We decouple image and video training into two stages. We first develop a strong image pretraining recipe ( §2.1) with several regularization techniques to create a robust starting point. Then we use the resulting image model as a frame encoder to develop a video data engine ( §2.2) supported by our novel human-refined video-text dataset to generate aligned captions for video clips. Finally, we train the image encoder on the resulting aligned video data ( §2.3). Using our data engine design, this short training step substantially improves both image and video performance.
this section cite: []

Section: Robust Image Pretraining
In the first stage of pretraining, we want to learn as much visual information as possible from a large set of image-text data with high regularization, stability, and training efficiency in mind.
this section cite: []

Section: Setup.
We track our changes with OpenCLIP [50] ViT-L/14 at 224 resolution as a baseline (Fig. 2.1). We fix a training budget of around 1T GFLOPs (i.e., a ZFLOP), and ablate on a fixed 2.3B noisy image-text dataset curated using the MetaCLIP [150] text-only curation pipeline, and start by training for 12B samples seen. To assess generality, we report ImageNet val [25] zero-shot classification results as well as an average of 6 common robustness metrics: ImageNet val [25], ImageNet v2 [109], ObjectNet [4], ImageNet Adversarial [46], ImageNet Rendition [45], and ImageNet Sketch [140].
Training. Motivated by [69,70,77,128,133], we begin by improving training efficiency with progressive resolution (Fig. 2.2). By evenly splitting the baseline 12B sample run into 98, 154, and 224 resolution stages (4B per stage), we half training FLOPs while maintaining performance. We then use the extra budget to double global batch size (Fig. 2.3) from 32K to 64K, increasing total samples from 12B to 24B. This makes hard negatives more probable, increasing the
78.9 78.9 79.5 79.9 80.4 80.7 81.0 81.1 81.3 76.2 76.9 78.3 79.2 80.1 80.8 80.9 75.3 Baseline Prog. Res LAMB High Res RoPE Attn Pool Data Aug Mask Reg Batch Sz 75.1 1.0 0.5 1.1 1.1 1.2 1.2 1.2 1.2 1.2
8.
this section cite: ['b49', 'b149', 'b24', 'b24', 'b108', 'b3', 'b45', 'b44', 'b139', 'b68', 'b69', 'b76', 'b127', 'b132']

Section: 9.
Robustness avg of 6 / ImageNet val Training ZFLOPs
this section cite: []

Section: Figure 2: Robust Image Pretraining.
We tune our pretraining recipe ( §2.1) to maximize performance on a fixed set of data, starting with an OpenCLIP [50] ViT-L/14 model. We report cumulative zero-shot classification results for each modification. The inner bars show robustness evaluation, calculated as the average of 6 robustness benchmarks [4,25,45,46,109,140], and the outer bars show ImageNet val [25] alone. Several changes significantly improve robustness, indicating that ImageNet val scales more with data, while robustness can scale with refined training techniques.
"task difficulty" of CLIP. Finally, we switch from AdamW to LAMB [154] (Fig. 2.4), which allows us to stably increase learning rate from 5 × 10 -4 to 2 × 10 -3 and better fit the CLIP objective. Overall, these changes improve +1.0% on ImageNet val and a similar +1.6% on robustness.
Modeling. To assist with scalability [35,128], we add a higher resolution (Fig. 2.5) stage at 336 pixels. To keep FLOPs the same, we adjust the schedule to 10B samples at 98 resolution, 8B at 154, 4B at 224, and 2B at 336. To improve extrapolation, we also add 2D RoPE [124] (Fig. 2.6) to each attention layer, keeping the original position embedding. Finally, we follow [158] in constructing the CLIP embedding using an attention pooling transformer block (Fig. 2.7). Surprisingly, we found keeping the class token as an input to this block is important for small model performance. These changes improve ImageNet val by +1.1% but robustness threefold, by +3.2%.
Regularization. Despite training on billions of samples, we find data augmentation (Fig. 2.8) still important. Adding heavy random cropping, brightness/saturation jitter, and horizontal flip generally improves robustness without adverse downstream effects (e.g., for OCR). Finally, we add mask regularization (Fig. 2.9) by duplicating and masking 1/16th of the input batch. At the output, the masked tokens are aligned to their unmasked counterparts by maximizing cosine similarity. Together, these regularization changes improved ImageNet val by +0.3% and robustness by +0.8%.
Overall, our recipe improves ImageNet val by +2.4% and robustness by a significant +5.6% while keeping FLOPs similar and maintaining or improving scaling behavior (see Appendix C.1). LLM Title + Description + Metadata Frame 1 caption + Frame 2 caption + … + Frame Captions Frame n caption + Video Caption Video Captioner Image Captioner Aligned Caption T Text Figure 3: Video Data Engine.
this section cite: ['b49', 'b3', 'b24', 'b44', 'b45', 'b108', 'b139', 'b24', 'b153', 'b34', 'b127', 'b123', 'b157']

Section: Bootstrapping a Video Data Engine with Perception Encoder
We use a PE-based video captioner for video-level captions and an existing image captioner [80] on sampled frames. We use these components along with the video metadata to synthesize short captions with a text-only LLM [80]. Our next step is to extend the image-only encoder to video. Unlike web-scale image-text data, which comes in many cases with human-generated descriptive alt-text information, videos with aligned language annotation are inherently scarce and often low quality. Inspired by the recent success of image data engines [57, 63, 94, 108, 149], we address the lack of high quality aligned video captions by developing a robust video data engine to generate them. Our approach (Fig. 3) represents the first large-scale exploration of this kind.
this section cite: ['b79', 'b79']

Section: Video Data Engine.
We build our data engine in 3 parts: (1) we construct a video captioning model using an early image-only version of PE as a frame-level encoder and Llama [80] as the language decoder. We train with the PLM [21] MLLM training recipe and data mix. In total, the mix consists of 64.7M images and videos covering natural images, charts, documents, exocentric and egocentric videos. (2) to further boost captioning performance, we collect a set of 265K videos (part of which we release as PE Video Dataset, see Appendix A.1), caption them with our base video captioner, and ask human raters to refine the captions. We then finetune our video captioner with this human refined data, significantly improving captioning quality (see Appendix C.2). (3) finally, we synthesize the aligned video captions by incorporating captions from our video captioner, Llama 3.2 [80] as a per-frame image captioner, and the existing title and description metadata (Fig. 3) summarized with a Llama 3.3 70B text model (see Appendix A.2.4 for prompts).
this section cite: ['b79', 'b20', 'b79']

Section: Video Training.
We use the resulting data engine to generate information-dense captions for a diverse set of 22M videos, with which we finetune the image-only PE model. To do so, we use PE as an frame-level encoder: for each video, we uniformly sample 8 frames, extract the CLIP embedding for each, and average pool to obtain a single video embedding for text embedding alignment. Despite its simplicity, we find this technique produces a strong joint image-video encoder.
Image Zero-Shot Video Zero-Shot
Title Description Video Caption Frame Caption Average Image ImageNet val [25] ImageNet v2 [109] ObjectNet IN Classes [4] MS-COCO txt→img [74] MS-COCO img→txt [74] Average Video Kinetics 400 [54] Kinetics 600 [54] MSR-VTT txt→vid [151] MSR-VTT vid→txt [151] 72.6 83.3 77.8 85.8 49.4 66.8 50.9 69.7 68.4 38.0 27.3 ✓ ✓ 75.4 83.2 78.2 87.1 47.3 66.0 56.0 74.1 73.5 39.0 37.3 ✓ ✓ ✓ 78.2 83.5 78.4 86.8 56.0 74.3 60.9 73.8 73.4 47.6 48.8 ✓ ✓ ✓ * ✓ 78. 1 83.7 79.0 87.7 54.1 73.0 60.9 75.4 75.1 46.7 46 2 83.7 79.0 87.5 54.6 73.2 61.6 75.8 75.5 47.4 48.1Table 1: Video Data Engine Ablation. We ablate our video data engine in Fig. 3 by finetuning on an in-development image-only version of PE by averaging the frame embeddings to create a single video CLIP embedding. Video captions are generated by our captioner trained with or without (✓ * ) human-refined data. Frame captions are generated by the Llama 3.2 vision model [80]. Taken together, the result is a huge boost to both image and video zero-shot performance. See Appendix C.2 for more ablations and scaling behavior.
Ablations. In Tab. 1, we ablate the impact of each component of the video data engine by finetuning an intermediate image-only PE core checkpoint on the recaptioned videos. Compared to the image-only baseline encoder (first row), our video data engine significantly enhances zero-shot classification and retrieval performance for both image (72.6→78.2) and video (50.9→61.6). Notably, using videolevel and frame-level captions provides significant improvements over relying solely on metadata such as video title and description (second row), highlighting the importance of building a robust video data engine to compensate for noise in web videos.
this section cite: ['b79']

Section: A Unified Encoder for Image and Video
Using a robust, scalable image pretraining recipe and video-pretraining data recaptioned by the proposed video data engine, in this section we present PE core , a unified image-and-video encoder.
Scale Tower Params Width Depth MLP Heads CLIP Dim B Vision 0.09B 768 12 3072 12 1024 Text 0.31B 1024 24 4096 16 L Vision 0.32B 1024 24 4096 16 1024 Text 0.31B 1024 24 4096 16 G Vision 1.88B 1536 50 8960 16 1280 Text 0.47B 1280 24 5120 20
Table 2: PE Model Configurations. Model Architecture. To capitalize on the promising scaling behavior observed in §2.1, we scale the largest PE core model to 2B parameters (G scale). Tab. 2 shows the detailed model configuration of the vision and text transformers and the dimension of the output clip embedding space.
this section cite: []

Section: Model Training.
We train PE core in three stages:
1. Image pretraining. We scale up image pretraining data to 5.4B publicly available image alt-text pairs curated with MetaCLIP [150] and a total of 86B samples seen to ensure convergence (58B for B and L). We use a global batch size of 131K, with progressive resolution from 98 to up to 448 depending on the model. 2. Image and video finetuning. Following the initial pretraining, we subsequently finetune the model at max resolution with a short schedule for 50M samples on the image pretraining data (as cooldown) followed by 22M samples on the recaptioned videos with a smaller learning rate and batch size. The video captions are produced using the proposed video data engine ( §2.2). For each video clip, we uniformly sample 8 frames, encode them, take their average to produce a single video embedding, and align them with the corresponding video captions using the same contrastive objective in image training. 3. Smaller model distillation. We distill the 2B model (G scale) into smaller contrastive pretrained models at B and L scales under their final resolutions, using a short finetuning schedule that covers approximately 4B samples seen (∼8% of the pretraining schedule) with a lower learning rate. We still perform stages 1 and 2 for small models (see Appendix C.3).
Detailed training configurations and setups are listed in Appendix B.
1.1. Model Zero-Shot Classification Zero-Shot Fine-Grained Classification Zero-Shot Retrieval Encoder Params Resolution Data Avg Class. ImageNet val [25] ImageNet v2 [109] ObjectNet IN Classes [4] ImageNet Adversarial [46] ImageNet Renditions [45] ImageNet Sketch [140] Avg Fine. Flowers Oxford [95] Cars Stanford [58] Aircrafts FGVC [86] Countries 211 [130] Scenes SUN397 [148] Satellite RESISC [20]
Avg Retrieval MS-COCO txt→img [74] MS-COCO img→txt [74] Flickr-30k txt→img [155] Flickr-30k img→txt [155] SigLIP-B/16 † [158] 0.1B 224 10B 69.9 76.2 69.5 70.7 45.1 90.2 67.9 61.8 85.2 90.8 44.0 15.9 70.0 64.6 69.8 47.2 64.5 77.989.6 SigLIP2-B/16 † [135] 0.1B 224 10B 73.1 78.2 71.4 73.6 55.0 91.7 68.9 66.2 85.7 93.4 54.8 19.2 72.7 71.1 73.7 52.1 68.9 80.7 93.0 PEcoreB 0.1B 224 5.4B 73.2 78.4 71.7 71.9 62.4 88.7 66.1 68.8 86.5 92.1 57.0 30.5 74.0 72.7 74.3 50.9 71.0 80.8 94.4 SigLIP-L/16 † [158] 0.3B 384 10B 80.7 82.1 75.9 80.9 76.5 95.0 73.6 67.1 89.4 94.8 53.2 24.7 72.5 67.9 74.7 52.8 70.5 82.6 92.9 SigLIP2-L/16 † [135] 0.3B 384 10B 83. 3 83.1 77.4 84.4 84.3 95.7 75.5 72.5 90.0 95.8 67.0 31.6 74.8 75.5 76.7 55.3 71.4 85.095.2 PEcoreL 0.3B 336 5.4B 83.9 83.5 77.9 84.7 89.0 95.2 73.4 74.6 87.2 93.7 67.8 45.6 77.4 75.7 78.8 57.1 75.9 85.5 96.6 DFN-H+ † [32] 0.6B 378 5B 81.6 84.3 78.3 79.6 79.6 93.6 73.3 75.2 91.6 96.0 72.5 37.9 77.4 75.9 75.8 55.6 71.8 82.1 93.6 InternVL-C [19] 5.5B 224 5B 82.5 83.2 77.3 80.6 83.8 95.7 74.3 69.9 85.8 94.4 53.3 35.1 76.3 74.4 78.6 58.6 74.9 85.0 95.7 EVA 18B [127] 17.5B 224 2B 83.6 83.8 77.9 82.2 87.3 95.7 74.7 73.1 86.0 94.9 59.7 43.1 77.7 76.9 77.5 56.2 73.6 83.3 96.7 SigLIP2-g-opt † [135] 1.1B 384 10B 86. 2 85.0 79.8 88.0 90.5 96.6 77.4 75.6 91.5 95.9 73.6 40.1 76.3 75.9 78.0 56.1 72.8 86.0 95.4  PEcoreG (image only) 1.9B 448 5.4B 86.0 85.2 80.2 87.1 91.2 96.1 76.1 78.2 91.0 94.6 76.7 57.3 77.5 71.8 74.9 53.1 70.9 81.6 93.9  PEcoreG  1.9B 448 5.4B 86.6 85.4 80.2 88.2 92.6 96.5 76.5 79.4 91.4 94.7 78.2 57.6 78.5 75.8 78.9 58.1 75.4 85.7 962 Table 4: Zero-Shot Video Results. Video performance of PE core compared to recent video and image encoders. PE core obtains state-of-the-art in video classification and comparable performance on retrieval benchmarks while using only 22M videos. † SigLIP2 evaluated by us (see Appendix B.1.2).
this section cite: ['b149', 'b73', 'b73', 'b154', 'b154']

Section: Zero-Shot Image Results.
In Tab. 3, we present PE core 's performance on zero-shot image benchmarks for classification and retrieval vs. the strongest open models, including SigLIP2 [135]. PE core outperforms all other contrastive models across the board on all zero-shot tasks, including the highly competitive average of zero-shot ImageNet robustness metrics [4,25,45,46,109,140]. This marks a significant achievement, as we are the first to accomplish this in over 3 years without access to Google's internal JFT-3B [28] or WebLI [17] datasets. And at the same time, PE core also exceeds the existing state-of-the-art on image-text retrieval and significantly improves on fine-grained classification-the first to simultaneously hold state-of-the-art on all common zero-shot categories. Notably, this dominant image performance is made possible by our video finetuning. Compared to image only, the video finetuned PE core G obtains +0.6% general classification, +1.2% fine-grained classification, and a significant +4.0% boost on retrieval. Thus, well-aligned video text data does not just improve video performance-it creates a strictly better model for both videos and images. Zero-Shot Video Results. We present video results in Tab. 4. Our base image encoder already outperforms all other image-only encoders on both zero-shot classification and retrieval, including SigLIP2-g-opt. With video finetuning, PE core G significantly outperforms even native video models that use full temporal attention on video classification, and it nearly matches the state-of-the-art on video retrieval despite being a simple frame-level encoder. This result underscores the importance of our video data engine, resulting in +3.9% on average zero-shot video classification, and a massive +11.1% on video retrieval. Moreover, PE core does this with fewer videos compared to other videobased approaches like InternVideo2 [143], highlighting the benefits of a joint image-video encoder. See Appendix C.4 for additional zero-shot and probing results. 3 General Features in a Contrastive Disguise PE core has strong results on zero-shot classification and retrieval, but these are tasks contrastive encoders specialize in. More important is whether or not this strong performance generalizes to downstream tasks. To find out, we compare PE core G to state-of-the-art models for other pretraining techniques: captioning (AIMv2-3B [29]) and self-supervised learning (DINOv2-g [96]). Best 89.8 AIMv2 Layer DINOv2 Layer PE core Layer AIMv2 Layer DINOv2 Layer PE core Layer 24 1 40 1 50 1 PE core Layer DINOv2 Layer AIMv2 Layer 24 1 40 1 50 1 24 1 40 1 50 1 Detection COCO Box AP Depth NYU RMSE (Flipped) OCR Q&A Avg of 4 % Accuracy Visual Q&A Avg of 4 % Accuracy Grounding RefCOCO /g/+ P@0.5 Tracking DAVIS Zero-Shot J&F Best 77.5 Best 62.1 Best 70.6 58.5 Layer 40/40 56.8 Layer 32/50 54.7 Layer 13/24 0.28 Layer 27/40 0.25 Layer 39/50 43.2 Layer 38/50 42.9 Layer 34/40 39.1 Layer 17/24 67.9 Layer 37/24 70.6 Layer 31/40 68.7 Layer 38/50 75.3 Layer 47/50 77.5 Layer 24/24 61.4 Layer 20/24 38.9 Layer 23/40 Attn Probe ImageNet 1k % Accuracy 89.8 Layer 50/50 86.9 Layer 38/40 89.5 Layer 24/24 0.31 Layer 16/24 Language Modeling Spatial Tasks Classification 62.1 Layer 38/50 Best 43.2 Best 0.25 Best 58.5 63.7 Layer 27/40 Layerwise Feature Analysis. We perform frozen feature analysis of each encoder in Fig. 4 for several downstream benchmarks in 3 categories: classification, language modeling, and spatial tasks. For classification, we probe each model using a randomly initialized cross attention transformer block. For language alignment, we learn a projector and finetune a decoder-only LLM (see §4), and for spatial tasks we train with several different decoders (ViTDet [71] Mask-RCNN [42] with Absolute Win [7] for detection, DPT [106] for depth, and zero-shot feature correspondence for tracking [51]). For each experiment, we sweep over the layers of the model as the optimal features are not necessarily the last. In each case, we use an equivalent image size (window size for detection) of 32 × 32 tokens. In each plot, we normalize performance by the maximum and minimum performance across models on that task.
this section cite: ['b134', 'b3', 'b24', 'b44', 'b45', 'b108', 'b139', 'b27', 'b16', 'b70', 'b41', 'b6', 'b105', 'b50']

Section: General Features in Disguise.
This analysis reveals several insights. First, as expected, AIMv2 performs well at classification and the best at visual Q&A language tasks. Similarly, DI-NOv2 performs the well on spatial tasks like detection, depth, and even grounding through an LLM. Then as already established by other works: DINOv2 performs poorly on OCR tasks [131]. But interestingly, its performance peaks in the middle of the network and then drops by the end.
And so do the others on several tasks (AIMv2: tracking, grounding, detection; DINOv2: VQ&A, grounding). PE core exhibits similar behavior, but with unexpected results: it can perform well on all tasks, often matching or exceeding the leading models. Remarkably, PE has intermediate layers that perform near to or on par with AIMv2 for language tasks and DINOv2 for spatial tasks, despite being trained with a global contrastive loss. Depth estimation is particularly noteworthy, as contrastive encoders are not typically considered state-of-the-art in that area. In fact, CLIP models are notorious for poor spatial performance [107].
An Alignment Problem. However, PE core 's strong general performance diminishes rapidly towards the end of the network, such as for LLM-based grounding. This behavior is less pronounced the closer the downstream task is to the pretraining method, suggesting an alignment problem. Thus, a well-tuned large-scale contrastive model can learn general embeddings in the process of fitting its objective, but it fails to output them. We address this issue with alignment tuning in §4 and §5 and analyze why our CLIP model has these general features and its scaling behavior in Appendix C.5.
Analysis. The finding that pure CLIP models possess features which match the performance of state-of-the-art pretraining methods in their specialized domains is new. In fact, recent work [30] has shown the opposite-that CLIP models fail to scale on downstream tasks. We next investigate how our approach yields these results.
34.4 34.2 33.2 35.7 37.0 37.1 38.0 38.2 28.6 32.3 30.7 30.0 32.3 33.2 31.7 31.9 32.7 26.0 Baseline Prog. Res LAMB High Res RoPE Attn Pool Data Aug Mask Reg Batch Sz 14/24 16/24 16/24 16/24 16/24 16/24 18/24 18/24 18/24 1. 2. 3. 4. 5. 6. 7. 8.
this section cite: ['b130', 'b106', 'b29']

Section: 9.
Last Layer COCO Box / Best Layer COCO Box Argmax Layer  2 evaluated as frozen features on COCO [74] using Mask R-CNN [42]. We report the last layer performance, best layer performance, and the best layer's index.
To start, we perform layerwise frozen feature analysis on COCO detection. PE core was particularly "peaky" on this task in Fig. 4, with its best layer on par with DI-NOv2, but last layer significantly worse. We already ablate each change we made from vanilla CLIP in Fig. 2 using a ViT-L/14 model. So to retrace our steps, we run frozen feature analysis on those checkpoints. For efficiency, we use a lower resolution and only sample even layers for this experiment. In Fig. 5, we report COCO box mAP for the last and best layers for each cumulative ablation, along with the index of the best layer. Further, we plot the layerwise mAP for each change in Fig. 6.
Surprisingly, the simple changes we made to CLIP pretraining in §2.1 overall improved the best layer's performance by almost 10 mAP! Some improvements are expected like with high resolution (5) and RoPE (6), but unexpectedly data augmentation (8) and especially progressive resolution (2) help considerably. It is possible that contrastive pretraining overfits to a specific resolution through "global tokens" [23], thus changing the resolution during training forces the model to be more robust.
1 Vanilla 2 8 7 6 5 4 3 Ours 9 Frozen Encoder Layer COCO Box mAP Next, both progressive resolution (2) and attention pooling (7) move the argmax layer deeper into the network (rightmost column of Fig. 5). Attention pooling in particular alters the whole shape of the layerwise performance curve (Fig. 6). Finally, some changes reduced performance: increasing the batch size (3) and using LAMB with a high learning rate (4). Both help fit the CLIP loss better, which after a point may not improve the general features. Moreover, while the best layer improved significantly, the last layer performance stagnated after (2). This suggests that constructing the CLIP token requires a specialized decoder. Yet, this does not prevent the model from learning general features-just outputting them.
Scaling Behavior. Evidently, our robust recipe can enable contrastive pretraining to produce general features. But, does it scale? In Fig. 7, we answer this by performing frozen feature analysis across S/14, B/14, and L/14 models trained with the same schedule with either the vanilla CLIP recipe or our recipe (see Fig. 14). Immediately, we see a stark contrast between their scaling behaviors: while the vanilla recipe quickly plateaus at L scale (300M), the best layer of our robust pretraining recipe demonstrates scaling to G scale (2B)-despite being trained with a decidedly non-spatially aligned global contrastive loss. Though note this is the best layer. The last layer still stagnates for both. Thus, CLIP loss obfuscates its general features even with our recipe, placing them several layers deep.
this section cite: ['b73', 'b41', 'b7', 'b22']

Section: COCO Box mAP

this section cite: []

Section: Frozen Encoder Depth
Frozen Encoder Depth
this section cite: []

Section: Perception Encoder: Language Alignment
In §3 we have seen that PE core already possesses useful features for Multimodal Large Language Models (MLLMs), but those features are not aligned to the end of the network. In this section, we lift these features through alignment tuning to construct a new, MLLM-specialized encoder: PE lang .
this section cite: []

Section: Alignment Method.
Aligning a vision encoder to an LLM is relatively straightforward. We follow the approaches of [18,21,36], where the vision encoder is unfrozen and finetuned as part of an MLLM. In our case, we align PE core to a pretrained Llama3.2 3B text-only decoder with both the encoder and decoder unfrozen, connected with a 2-layer MLP. We discard the last 3 layers of PE core , as suggested by [18] and regularize the encoder with LayerScale [132] and DropPath [49]. We train with next token prediction on 70M total samples across OCR Q&A, Captioning, Visual Q&A, and Video Q&A (following [21]), and finally extract the vision encoder only as PE lang . More training details are available in Appendix B.2 and ablations of this recipe are conducted in Appendix D.1.
this section cite: ['b17', 'b20', 'b35', 'b17', 'b131', 'b48', 'b20']

Section: Accuracy (Avg of 4) Accuracy (Avg of 4)
CIDEr Score (Avg of 3)
Frozen Encoder Layer RefCOCO /g/+ P@0.5
this section cite: []

Section: Frozen Encoder Layer
Frozen Encoder Layer Frozen Encoder Layer Model OCR / Chart / Doc. Q&A Visual Q&A Captioning Video Avg. OCR QA ChartQA Acc.
OCR Q&A Visual Q&A Captioning Grounding PEcore G PElang G PEcore G PElang G PEcore G PElang G PEcore G PElang G
[162]
this section cite: []

Section: DocVQA
Acc.
[89]
Info. QA Acc.
[90]
this section cite: []

Section: AI2D
Acc.
[
Avg. VQA TextVQA Acc.
[122]
OK-VQA Acc.
[115]
this section cite: []

Section: POPE
Acc.
[72]
this section cite: []

Section: VQAv2
Acc.
[39]
Avg. Cap. Flicker CIDEr [155] COCO CIDEr [74] No Cap CIDEr [1] Avg. Ground.
RefCOCO/g/+ [55] Avg. Video VideoMME Acc.
[37]
this section cite: ['b154', 'b73', 'b0', 'b54']

Section: STAR
Acc.
[
TGIF-QA Acc.
[52]
this section cite: []

Section: EgoSchema
Acc.
[87]
this section cite: []

Section: MVBench
Acc.
[67]
this section cite: []

Section: PerceptionTest
Acc.
[102]
576 Tokens per Image CLIP-L [103] 53.5 61.7 49.5 32.8 70.1 72.7 60.7 63.9 87.3 78.9 113.3 92.0 132.9 115.0 65.0 54.2 46.3 52.1 68.6 57.4 48.5 52.3 AIMv2-L Distill [36] 53.7 61.1 49.4 31.5 72.7 74.1 62.8 64.8 88.3 80.3 117.8 94.7 137.5 121.2 62.6 53.8 44.3 52.4 65.0 57.4 50.0 53.6  SigLIP2-so400M [135] 58.9 69.0 58.3 35.2 73.1 76.8 69.8 67.2 88.7 81.6 116.5 92.1 137.7 119.8 67.4 54.5 45.5 53.1 67.2 57.6 49.3 54.5SigLIP2-g-opt [135] 56. 2 63.1 55.3 34.0 72.4 77.0 70.3 66.7 89.6 81.6 117.7 94.9 137.8 120.3 66.5 53.9 46.2 53.9 66.6 53.8 48.554.7 PE lang G † 66.9 76.8 73.6 41.1 76.1 76.2 68.5 66.0 89.1 81.3 119.7 96.1 139.6 123.4 68.9 58.1 48.7 58.9 70.5 61.8 52.7 55.9 1024 Tokens per Image InternViT2.5-L [18] 60. 6 74.1 59.2 35.9 73.1 74.2 65.4 64.4 87.6 79.6 112.3 88.4 133.7 114.9 66.9 50.6 45.2 44.8 62.7 54.2 46.0 50.5  SigLIP2-so400M [135] 63.3 72.1 69.3 39.0 72.7 77.9 74.8 66.0 89.0 81.8 117.4 93.5 138.3 120.2 69.6 55.8 46.2 55.4 67.0 62.0 50.0 54.5  PEcoreL  59.4 68.7 62.5 36.6 69.7 74.7 67.7 64.3 88.3 78.7 112.7 89.6 133.4 114.9 59.7 50.9 41.7 51.2 61.6 52.6 47.450.6 PE lang L 71.1 81.0 81.9 46.4 75.0 77.1 73.0 65. 5 89.3 80.8 117.3 94.3 137.3 120.1 70.5 56.5 47.0 57.2 68.0 59.8 52.3 547 DINOv2-g [96] 30.0 19.6 14.7 24.2 61.5 61.0 19.3 60.4 88.6 75.8 109.4 86.5 131.6 110.1 64.9 49.5 39.7 52.1 60.1 46.8 47.4 50.8 AIMv2-3B [36] 48.9 40.5 53.9 33.9 67.2 73.0 64.1 64.0 85.2 78.9 115.7 93.8 135.2 118.1 36.1 54.6 45.1 54.5 66.7 55.4 51.7 54.3 InternViT2.5-6B [18] 59.9 72.3 59.4 35.2 72.5 75.5 68.9 64.9 88.2 80.2 115.0 92.2 136.3 116.3 Table 5: MLLM Results. We benchmark PE lang vs. other frozen vision encoders with Llama 3.1instruct 8B [80] as the LLM. PE lang shows strong performance across all benchmarks, outperforming much larger models. † Interpolated without extra training. See Appendix D.4 for more results.
this section cite: ['b35', 'b134', 'b95', 'b35', 'b17']

Section: Perception Encoder: Spatial Alignment
Unlike for language alignment with an MLLM, the best way to spatially align a model is not obvious. However, the path becomes clear when we study an apparent dichotomy in §3 for PE core : higher level spatial tasks like detection and depth estimation perform optimally around layer 40, while low level tasks like tracking perform the best at around layer 30. Upon analyzing the features directly (see Appendix E.1), we find that locality begins to deteriorate starting at layer 33 due to global tokens [23].
Alignment Method. Following these insights, we design our spatial alignment method with two goals in mind: (1) keep the high level features around layer 40 in tact while (2) improving the locality of the features for lower level tasks. To address (1), we simply finetune PE core using its own frozen layer 41 features as a teacher with heavy regularization (DropPath [49], LayerScale [132], 75% masking [144]). Then, we enforce spatial correspondence for (2) using SAM 2.1 [108] mask logits.
That is, unlike [44,107,116], we do not directly use SAM features but instead sample 32×32 points in a grid and concatenate the SAM 2.1 mask logit for each into a single feature map. As shown in Appendix Fig. 19, this provides features with strong locality. See Appendix B.3.1 for training details.
this section cite: ['b22', 'b48', 'b131', 'b143', 'b107', 'b43', 'b106', 'b115']

Section: COCO Box mAP
Frozen Encoder Layer
this section cite: []

Section: Detection ADE20k mIoU
Frozen Encoder Layer Semantic Segm.
DAVIS Zero-Shot J&F Frozen Encoder Layer Tracking NYU RMSE Loss Frozen Encoder Layer Depth (↓) PEcore G aligned to SAM Mask Logits PEspatial G PEcore Layer 41 aligned to both Effects. In Fig. 9, we compare layerwise performance of the original PE core G checkpoint compared to aligning to the teachers described above.
We denote aligning to both teachers as PE spatial G.
Aligning to PE core G layer 41 alone performs generally well on all tasks, but has lackluster performance on tracking, where percise locality is necessary to define boundaries. In contrast, aligning to SAM 2.1 mask logits lowers last layer performance on every task but tracking. Thus, the optimal approach is to combine both teachers. As a result, PE spatial G not only lifts the features for all tasks to the end of the network, but it also improves over self-alignment, especially on tracking and semantic segmentation. Notably, PE spatial G's tracking performance is lower than the SAM-aligned model, but it is still ahead of other methods while being generally good, see results. Tracking Segmentation Depth DAVIS (↑) [101] ADE20k (↑) [164] NYU (↓) [120] Encoder Best Last Idx Best Last Idx Best Last Idx SigLIP-so400M [158] 48.7 36.3 16/27 40.1 38.3 22/27 .339 .369 21/27 SigLIP2-so400M [135] 51.4 45.3 15/27 44.0 42.9 24/27 .306 .329 25/27 DINOv2-L [96] 58.7 58.2 23/24 47.3 47.3 24/24 .297 .308 23/24 DINOv2-g [96] 58.5 58.5 40/40 48.7 48.4 37/40 .279 .290 27/40 PEcoreG 56.8 42.8 32/50 41.5 38.6 44/50 .249 .309 39/50 PE spatial G 61.5 61.5 50/50 49.3 48.9 49/50 .262 .275 46/50  Table 8: SOTA Setting Detection on COCO val. Recipe in Appendix B.3.5.
Results. In Tab. 6, we compare performance on dense tasks with a frozen encoder with a fixed 448 resolution, reporting both best layer performance and last layer performance. Across the board, PE spatial G outperforms other state-of-the-art models, with its features well aligned to the last layer. In Tab. 7, the same is true when end-to-end finetuning for detection on both LVIS [40] and COCO [74] with a fixed 1024 resolution using Mask-RCNN [42] and ViTDet [71]. Finally, in Tab. 8, we provide a system-level comparison vs. the absolute state-of-the-art on COCO val2017. With only Object365 [117] as extra detection data, PE spatial G can match the performance of more complex models tuned for detection, while only using a simple DETR-style decoder [11,97]. PE spatial G marks the first general, contrastively pretrained model to accomplish this.
this section cite: ['b39', 'b73', 'b41', 'b70', 'b116', 'b10', 'b96']

Section: Related Work
Vision-language pretrained models have served as foundation for zero-shot image classification and image-text retrieval [50,103,114], open-vocabulary detection [62,92,93] and segmentation [22,27], and multimodal large language models (MLLMs) [3,5,76,91,98,131]. PE iterates on this paradigm.
this section cite: ['b49', 'b102', 'b113', 'b61', 'b91', 'b92', 'b21', 'b26', 'b2', 'b4', 'b75', 'b90', 'b97', 'b130']

Section: Contrastive Language-Image Pretraining.
The early works of Virtex [26], ICMLM [112], and ConVIRT [161] developed the techniques for learning through contrastive objectives between vision and language modalities. Subsequently, vision encoders such as CLIP [50,103] and ALIGN [53] scaled these techniques to much larger datasets and model sizes, popularizing vision-language contrastive learning. A series of open-weight contrastive models have been developed to enhance the performance and robustness of CLIP [32,70,114,126,150,158]. PE is among this effort.
this section cite: ['b25', 'b111', 'b160', 'b49', 'b102', 'b52', 'b31', 'b69', 'b113', 'b125', 'b149', 'b157']

Section: Existing Techniques.
Various techniques used in this work have been explored before. BASIC [99] and LAION [114] explored scaling the batch size up to 160K, and shows the benefits of large batch sizes during training. EVA-CLIP [127] uses LAMB optimizer [154] for large batch training of clip models. Rotary positional embedding (RoPE) [124] has been successfully adopted in large language models. In vision transformers [2, 47] adopted 2D rotatory positional embeddings. For data engine, a series of works focus on large-scale sourcing and filtering through efficient data curation [32,38,114,150] and explore recaptioning training images using MLLMs or VLMs [31,63,94,149]. We extend these concepts to create a robust training recipe and to extend data engines to video.
Intermediate Layers Are Better. Most vision encoders rely on the last layer to extract features. However, when trained on proxy or self-supervised tasks, the last layer is often not the ideal candidate for other tasks [8,15,16,29,83,104,118,125,139,157,163]. This has been shown for image coloration [160,163], next token prediction [15,29,104], image generation [83,157], and to a limited extent in CLIP models [125]. In contrast to these works, we first show the same behaviors across multiple classes of models simultaneously. Then we study this behavior for PE specifically in depth, and show it is possible for CLIP training to produce rich spatial and language features in intermediate layers on par with the best existing models for each. Finally, we show how to align these features with short finetuning steps to obtain state-of-the-art on a wide variety of tasks. Unlike other alignment [3, 18, 19, 65, 80, 129, 141] and feature combination [44,107,116,157] methods, our main goal is not to instill a large amount of new knowledge into the model, but instead to bring out and refine the latent strong general features that already exist in the original PE model.
this section cite: ['b98', 'b113', 'b126', 'b153', 'b123', 'b31', 'b37', 'b113', 'b149', 'b30', 'b62', 'b93', 'b148', 'b7', 'b14', 'b15', 'b28', 'b82', 'b103', 'b117', 'b124', 'b138', 'b156', 'b162', 'b159', 'b162', 'b14', 'b28', 'b103', 'b82', 'b156', 'b124', 'b43', 'b106', 'b115', 'b156']

Section: Conclusion
In this work, we have presented Perception Encoders (PE), a family of best-in-class foundation models comprising PE core , PE lang , and PE spatial . We have shown that PE core can outperform the leading models in zero-shot image recognition, while also excelling in zero-shot video recognition. We have demonstrated that PE lang outperforms the best vision encoders for use in multimodal large language models, often by a large margin. We have established that PE spatial outperforms the longstanding state-of-the-art in object detection with a simpler decoder. Throughout all of this, one conclusion is abundantly clear: Perception Encoder unlocks the potential to scale simple contrastive vision-language pretraining to address a wide range of downstream vision tasks.
this section cite: []

Section: References
Ref_id:b0 Title: Nocaps: Novel object captioning at scale Year: (2019)
Ref_id:b1 Title:  Year: (2024)
Ref_id:b2 Title: Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond Year: (2023)
Ref_id:b3 Title: ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models Year: (2019)
Ref_id:b4 Title:  Year: (2024)
Ref_id:b5 Title: Soft-NMS-Improving object detection with one line of code Year: (2017)
Ref_id:b6 Title: Window attention is bugged: how not to interpolate position embeddings Year: (2023)
Ref_id:b7 Title: Guillotine regularization: Why removing layers is needed to improve generalization in self-supervised learning Year: (2022)
Ref_id:b8 Title: The OpenCV library. Dr. Dobb's Journal: Software Tools for the Professional Programmer Year: (2000)
Ref_id:b9 Title: Activitynet: A large-scale video benchmark for human activity understanding Year: (2015)
Ref_id:b10 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b11 Title: AuroraCap: Efficient, performant video detailed captioning and a new benchmark Year: (2025)
Ref_id:b12 Title: Collecting highly parallel data for paraphrase evaluation Year: (2011)
Ref_id:b13 Title: Hybrid task cascade for instance segmentation Year: (2019)
Ref_id:b14 Title: Generative pretraining from pixels Year: (2020)
Ref_id:b15 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b16 Title: A jointly-scaled multilingual language-image model Year: (2023)
Ref_id:b17 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2024)
Ref_id:b18 Title: InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b19 Title: Remote sensing image scene classification: Benchmark and state of the art Year: (2017)
Ref_id:b20 Title: Perceptionlm: Open-access data and models for detailed visual understanding Year: (2025)
Ref_id:b21 Title: CAT-Seg: Cost aggregation for open-vocabulary semantic segmentation Year: (2024)
Ref_id:b22 Title: Vision transformers need registers Year: (2024)
Ref_id:b23 Title: ICML Year: (2023)
Ref_id:b24 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b25 Title: VirTex: Learning visual representations from textual annotations Year: (2021)
Ref_id:b26 Title: Decoupling zero-shot semantic segmentation Year: (2022)
Ref_id:b27 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b28 Title: Scalable pre-training of large autoregressive image models Year: (2024)
Ref_id:b29 Title: Scaling language-free visual representation learning Year: (2025)
Ref_id:b30 Title: Improving CLIP training with language rewrites Year: (2023)
Ref_id:b31 Title: Data filtering networks Year: (2024)
Ref_id:b32 Title: EVA: Exploring the limits of masked visual representation learning at scale Year: (2023)
Ref_id:b33 Title: EVA-02: A visual representation for neon genesis Year: (2024)
Ref_id:b34 Title: X3D: Expanding architectures for efficient video recognition Year: (2020)
Ref_id:b35 Title: Multimodal autoregressive pre-training of large vision encoders Year: (2025)
Ref_id:b36 Title: Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2024)
Ref_id:b37 Title: DataComp: In search of the next generation of multimodal datasets Year: (2023)
Ref_id:b38 Title: Making the v in VQA matter: Elevating the role of image understanding in visual question answering Year: (2017)
Ref_id:b39 Title: LVIS: A dataset for large vocabulary instance segmentation Year: (2019)
Ref_id:b40 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b41 Title: Mask R-CNN Year: (2017)
Ref_id:b42 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b43 Title: RADIOv2.5: Improved baselines for agglomerative vision foundation models Year: (2025)
Ref_id:b44 Title: The many faces of robustness: A critical analysis of out-of-distribution generalization Year: (2021)
Ref_id:b45 Title: Natural adversarial examples Year: (2021)
Ref_id:b46 Title: Rotary position embedding for vision transformer Year: (2024)
Ref_id:b47 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b48 Title: Deep networks with stochastic depth Year: (2016)
Ref_id:b49 Title:  Year: (2021)
Ref_id:b50 Title: Space-time correspondence as a contrastive random walk Year: (2020)
Ref_id:b51 Title: TGIF-QA: Toward spatiotemporal reasoning in visual question answering Year: (2017)
Ref_id:b52 Title: Scaling up visual and vision-language representation learning with noisy text supervision Year: (2021)
Ref_id:b53 Title: The kinetics human action video dataset Year: (2017)
Ref_id:b54 Title: Referitgame: Referring to objects in photographs of natural scenes Year: (2014)
Ref_id:b55 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b56 Title: Segment anything Year: (2023)
Ref_id:b57 Title: 3d object representations for fine-grained categorization Year: (2013)
Ref_id:b58 Title: Visual genome: Connecting language and vision using crowdsourced dense image annotations Year: (2017)
Ref_id:b59 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b60 Title: HMDB: a large video database for human motion recognition Year: (2011)
Ref_id:b61 Title: F-VLM: open-vocabulary object detection upon frozen vision and language models Year: (2023)
Ref_id:b62 Title: VeCLIP: Improving CLIP training via visual-enriched captions Year: (2024)
Ref_id:b63 Title: What matters when building visionlanguage models Year: (2024)
Ref_id:b64 Title: LLaVA-OneVision: Easy visual task transfer Year: (2025)
Ref_id:b65 Title: Unmasked teacher: Towards training-efficient video foundation models Year: (2023)
Ref_id:b66 Title: MVBench: A comprehensive multi-modal video understanding benchmark Year: (2024)
Ref_id:b67 Title: Grounded languageimage pre-training Year: (2022)
Ref_id:b68 Title: An inverse scaling law for CLIP training Year: (2023)
Ref_id:b69 Title: CLIPA-v2: Scaling CLIP training with 81.1% zero-shot imagenet accuracy within a $10,000 budget Year: (2023)
Ref_id:b70 Title: Exploring plain vision transformer backbones for object detection Year: (2022)
Ref_id:b71 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b72 Title: Binsformer: Revisiting adaptive bins for monocular depth estimation Year: (2024)
Ref_id:b73 Title: Microsoft COCO: Common objects in context Year: (2014)
Ref_id:b74 Title: LLaVA-NeXT: Improved reasoning, ocr, and world knowledge Year: (2024)
Ref_id:b75 Title:  Year: (2024)
Ref_id:b76 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b77 Title: Swin transformer v2: Scaling up capacity and resolution Year: (2022)
Ref_id:b78 Title: A ConvNet for the 2020s Year: (2022)
Ref_id:b79 Title: The llama 3 herd of models Year: (2024)
Ref_id:b80 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b81 Title: CLIP4Clip: An empirical study of clip for end to end video clip retrieval Year: (2021)
Ref_id:b82 Title: SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b83 Title: Video-ChatGPT: Towards detailed video understanding via large vision and language models Year: (2024)
Ref_id:b84 Title: VideoGPT+: Integrating image and video encoders for enhanced video understanding Year: (2024)
Ref_id:b85 Title: Fine-grained visual classification of aircraft Year: (2013)
Ref_id:b86 Title: Egoschema: A diagnostic benchmark for very long-form video language understanding Year: (2024)
Ref_id:b87 Title: Tips: Text-image pretraining with spatial awareness Year: (2024)
Ref_id:b88 Title: DocVQA: A dataset for vqa on document images Year: (2021)
Ref_id:b89 Title:  Year: (2022)
Ref_id:b90 Title: MM1: methods, analysis and insights from multimodal LLM pre-training Year: (2024)
Ref_id:b91 Title: Simple open-vocabulary object detection with vision transformers Year: (2022)
Ref_id:b92 Title: Scaling open-vocabulary object detection Year: (2023)
Ref_id:b93 Title: Improving multimodal datasets with image captioning Year: (2023)
Ref_id:b94 Title: Automated flower classification over a large number of classes Year: (2008)
Ref_id:b95 Title: Learning robust visual features without supervision Year: (2024)
Ref_id:b96 Title: NMSstrikes back Year: (2022)
Ref_id:b97 Title: Kosmos-2: Grounding multimodal large language models to the world Year: (2023)
Ref_id:b98 Title: Combined scaling for zero-shot transfer learning Year: (2023)
Ref_id:b99 Title: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models Year: (2015)
Ref_id:b100 Title: The 2017 DAVIS challenge on video object segmentation Year: (2017)
Ref_id:b101 Title: Perception test: A diagnostic benchmark for multimodal video models Year: (2024)
Ref_id:b102 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b103 Title: An empirical study of autoregressive pre-training from videos Year: (2025)
Ref_id:b104 Title: Hierarchical text-conditional image generation with CLIP latents Year: (2022)
Ref_id:b105 Title: Vision transformers for dense prediction Year: (2021)
Ref_id:b106 Title: AM-RADIO: Agglomerative vision foundation model-reduce all domains into one Year: (2024)
Ref_id:b107 Title: SAM 2: Segment anything in images and videos Year: (2024)
Ref_id:b108 Title: Do imagenet classifiers generalize to imagenet Year: (2019)
Ref_id:b109 Title: The dollar street dataset: images representing the geographic and socioeconomic diversity of the world Year: (2022)
Ref_id:b110 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b111 Title: Learning visual representations with caption annotations Year: (2020)
Ref_id:b112 Title: UNIC: Universal classification models via multi-teacher distillation Year: (2024)
Ref_id:b113 Title: LAION-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b114 Title: A-OKVQA: A benchmark for visual question answering using world knowledge Year: (2022)
Ref_id:b115 Title: Theia: Distilling diverse vision foundation models for robot learning Year: (2024)
Ref_id:b116 Title: Objects365: A large-scale, high-quality dataset for object detection Year: (2019)
Ref_id:b117 Title: Objectives matter: Understanding the impact of self-supervised objectives on vision transformer representations Year: (2023)
Ref_id:b118 Title: Textcaps: a dataset for image captioning with reading comprehension Year: (2020)
Ref_id:b119 Title: Indoor segmentation and support inference from rgbd images Year: (2012)
Ref_id:b120 Title: Very deep convolutional networks for large-scale image recognition Year: (2015)
Ref_id:b121 Title: Towards VQA models that can read Year: (2019)
Ref_id:b122 Title: A dataset of 101 human actions classes from videos in the wild Year: (2012)
Ref_id:b123 Title: RoFormer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b124 Title: CLIPer: Hierarchically improving spatial representation of CLIP for open-vocabulary semantic segmentation Year: (2024)
Ref_id:b125 Title: EVA-CLIP: Improved training techniques for clip at scale Year: (2023)
Ref_id:b126 Title: EVA-CLIP-18B: Scaling clip to 18 billion parameters Year: (2024)
Ref_id:b127 Title: EfficientNet: Rethinking model scaling for convolutional neural networks Year: (2019)
Ref_id:b128 Title: Gemma 3 technical report Year: (2025)
Ref_id:b129 Title: YFCC100M: The new data in multimedia research Year: (2016)
Ref_id:b130 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b131 Title: Going deeper with image transformers Year: (2021)
Ref_id:b132 Title: DeiT III: Revenge of the ViT Year: (2022)
Ref_id:b133 Title: Image captioners are scalable vision learners too Year: (2023)
Ref_id:b134 Title: SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features Year: (2025)
Ref_id:b135 Title: A picture is worth more than 77 text tokens: Evaluating CLIP-style models on dense captions Year: (2024)
Ref_id:b136 Title: The inaturalist species classification and detection dataset Year: (2018)
Ref_id:b137 Title: Attention is all you need Year: (2017)
Ref_id:b138 Title: Teaching matters: Investigating the role of supervision in vision transformers Year: (2023)
Ref_id:b139 Title: Learning robust global representations by penalizing local predictive power Year: (2019)
Ref_id:b140 Title: Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b141 Title: InternImage: Exploring large-scale vision foundation models with deformable convolutions Year: (2023)
Ref_id:b142 Title: InternVideo2: Scaling foundation models for multimodal video understanding Year: (2024)
Ref_id:b143 Title: Masked feature prediction for self-supervised visual pre-training Year: (2022)
Ref_id:b144 Title: STAR: A benchmark for situated reasoning in real-world videos Year: (2021)
Ref_id:b145 Title:  Year: (2019)
Ref_id:b146 Title:  Year: (2019)
Ref_id:b147 Title: SUN database: Exploring a large collection of scene categories Year: (2014)
Ref_id:b148 Title: Altogether: Image captioning via re-aligning alt-text Year: (2024)
Ref_id:b149 Title: Demystifying clip data Year: (2024)
Ref_id:b150 Title: MSR-VTT: A large video description dataset for bridging video and language Year: (2016)
Ref_id:b151 Title:  Year: (2024)
Ref_id:b152 Title:  Year: (2024)
Ref_id:b153 Title: Large batch optimization for deep learning: Training BERT in 76 minutes Year: (2020)
Ref_id:b154 Title: From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions Year: (2014)
Ref_id:b155 Title: CoCa: Contrastive captioners are image-text foundation models Year: (2022)
Ref_id:b156 Title: Representation alignment for generation: Training diffusion transformers is easier than you think Year: (2025)
Ref_id:b157 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b158 Title: DINO: DETR with improved denoising anchor boxes for end-to-end object detection Year: (2023)
Ref_id:b159 Title: Colorful image colorization Year: (2016)
Ref_id:b160 Title: Contrastive learning of medical visual representations from paired images and text Year: (2022)
Ref_id:b161 Title: Advancing chart question answering with robust chart component recognition Year: (2025)
Ref_id:b162 Title: Good practice in cnn feature transfer Year: (2016)
Ref_id:b163 Title: Scene parsing through ADE20K dataset Year: (2017)
Ref_id:b164 Title: DETRs with collaborative hybrid assignments training Year: (2023)
Ref_id:b165 Title: have shown SAM to not be an effective teacher when distilling from multiple sources (though recently [44] has shown it can help with some tricks). However, upon observation of the raw features of SAM 2.1-L (Fig. 19), the main problem may be the same one we are currently trying to solve: SAM has global tokens as well Year: ()
Ref_id:b166 Title: Table 36: Raw Spatial Layer Analysis Results. The raw values for the plots in Fig Year: ()
