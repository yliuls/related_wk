Title: PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding
Abstract: Vision-language models are integral to computer vision research, yet many highperforming models remain closed-source, obscuring their data, design and training recipe. The research community has responded by using distillation from black-box models to label training data, achieving strong benchmark results, at the cost of measurable scientific progress. However, without knowing the details of the teacher model and its data sources, scientific progress remains difficult to measure. In this paper, we study building a Perception Language Model (PLM) in a fully open and reproducible framework for transparent research in image and video understanding. We analyze standard training pipelines without distillation from proprietary models and explore large-scale synthetic data to identify critical data gaps, particularly in detailed video understanding. To bridge these gaps, we release 2.8M humanlabeled instances of fine-grained video question-answer pairs and spatio-temporally grounded video captions. Additionally, we introduce PLM-VideoBench, a suite for evaluating challenging video understanding tasks focusing on the ability to reason about "what", "where", "when", and "how" of a video. We make our work fully reproducible by providing data, training recipes, code and models at https://github.com/facebookresearch/perception_models. Scaling Laws with Synthetic DataWe examine scaling properties of our synthetic data under controlled setup and establish scaling laws. 10 12 25 30 35 40 45 50 55 Avg. Error No Syth. (1B) No Syth. (3B) No Syth. (8B) Video QA Err. = (2.5e+03 x GFLOPs) 0.15 1B 3B 8B 10 12 GFLOPs No Syth. (1B) No Syth. (3B) No Syth. (8B) OCR QA Err. = (6.5e+03 x GFLOPs) 0.20 10 12

Section: Introduction
Vision-language models (VLMs) are now a key part of computer vision research and are widely used in both academia and industry. Many of the strongest performing VLMs are closed-source, meaning their design, training methods, and the data they use are not publicly shared. To stay competitive, the research community has started to catch up to the proprietary models by using a straightforward approach -distillation from black-box models [1][2][3][4][5], where proprietary models are directly used to label training data [3,6,7], directly leading to strong benchmark results.
Although distillation will unlock strong performance, there are two main issues for basic research. First, it makes it hard to track scientific progress. Specifically, we cannot tell if better results on benchmarks are due to advances in model design or training, or simply because the proprietary teacher models were trained on the evaluation sets of widely used benchmarks or internal data collected to resemble them -this information is not available. Second, the heavy reliance on distillation leads to a fundamental misunderstanding of the effectiveness of current methods for training VLMs from scratch. Several key questions remain unanswered, including the significance of each training stage,  the influence of synthetic data , the data gaps that the research community should prioritize, and which of these gaps are currently being artificially addressed by distillation from proprietary models.
To better understand these challenges, we develop the Perception Language Model (PLM), a fully open and reproducible model for transparent research in image and video understanding (Fig. 1 right). PLM consists of a vision encoder with a small scale (<8B parameters) LLM decoder. We start by an analysis of standard training pipelines with available data, without any proprietary model distillation. We investigate large-scale synthetic data and establish key scaling laws to identify critical data gaps that limit video understanding performance, especially for spatio-temporal reasoning and fine-grained understanding tasks.
To fill these gaps, we create 2.8M high-quality human-labeled instances of fine-grained video QA and spatio-temporally grounded video captions, see Fig. 1. This release is nearly an order of magnitude larger than the largest existing video datasets of each type [8,9]. Our model, dataset and benchmark push the boundaries of video understanding, and provide a foundation for reproducible and transparent training and evaluation of VLM research. Across 40 image and video benchmarks, we achieve comparable performance with existing state-of-the-art open-weight models (e.g., InternVL2.5 [10]), without distilling from proprietary models, and greatly outperform fully open models (i.e., Molmo [11]).
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b2', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10']

Section: PLM: Overview
In this section, we overview the model, training stages and training data involved in the development of PLM. Please refer to Fig. 8 for a detailed overview and Appendix A for additional details.
Stage 1 Stage 2 Stage 3 Warmup Midtraining SFT Modality Image Image + Video Image + Video Data 1M Synthetic 72M Mix 19M Mix Training Projector Full Full Downsampling -2 × 2 2 × 2 Tiles/Frames 1/-16/16 36/32
Table 1: Summary of three training stages to train PLM.
See Appendix Table 8 and Table 9 for data splits.
Model. PLM consists of a vision encoder and language decoder, where a pre-trained Perception Encoder (PE) [12] is connected to the Llama 3 [13] language decoder (1B, 3B, or 8B parameters) with a 2-layer MLP projector. We use PE L/14 for Llama3.2 1B and 3B, and PE G/14 for Llama3.1 8B. For image input, PLM incorporates dynamic tiling to support high resolution images for up to 36 tiles of 448 2 resolution, where each tile undergoes 2 × 2 average pooling to compress the visual tokens. For video input, PLM uses 32 frames at 448 2 resolution, where the same pooling is applied across the spatial dimensions of each video frame.
Data. The data used to train the PLM consists of synthetic and human-annotated samples. Synthetic data enhances the general capabilities of PLM, while human-annotated data broadens these capabilities to encompass more complex tasks. Synthetic data is sourced from a diverse array of image and video datasets, covering fundamental VLM capabilities such as OCR, chart/document/diagram understanding, image/video captioning, and visual question answering.
We design data engines for each data modality (e.g., natural images, charts, documents, figures, egocentric and exocentric videos) to efficiently scale up, creating ∼66.1M samples ( §3). The synthetic data can be noisy, but is available at large scale; on the other hand, human-annotated data provides rich, high-quality supervision for image and video tasks. Here, we combine existing human annotations of diverse image and video sources, with our own collected human-annotated data, specifically geared towards fine-grained video understanding and spatio-temporally grounded reasoning ( §4).
Training stages. PLM trains in three stages:
1. Projector warm-up. First, we freeze the vision encoder and LLM and only train the vision projector on a small amount of synthetic image data. This warms-up the newly initialized parameters in the projector and improves stability for later stages. We use 1M images from SA-1B [14] with the image captions generated by our data engine ( §3).
this section cite: ['b11', 'b12', 'b13']

Section: Samples Type Stage
Our Human-annotated (2.87M) PLM-FGQA 2.4M Fine-grained 3 PLM-STC 476.2K R(D)Cap + RTL 3 Our Synthetic (66.1M) Natural Images 15.9M Caption 1,2,3 Charts & Documents 31.9M Caption 2,3 Videos Mix 17.5M Mix. 2,3 Ego4D 880K Cap. + QA 2,3 Existing Open Source (6.52M) Image (92 datasets) 5.6M Diverse 2,3 Video (27 datasets) 920K Diverse 2,3
Table 2: Summary of the data mix for training PLM. See Table 10 for the full data blend.
2. Large-scale midtraining with synthetic data. Next, we train PLM on diverse domains of images and videos at scale, using a maximum of 16 tiles for images and 16 frames for videos. PLM sees around 64.7M images and videos with synthetically generated captions and questionanswer pairs. We employ our data engine to scale up synthetic data generation (see §3).
3. Supervised fine-tuning with human-annotated data.
Finally, we train PLM with higher image resolutions and more video frames, using up to 36 tiles for images and 32 frames for videos. In this stage, we tackle more challenging video tasks, including fine-grained QA and spatiotemporally grounded reasoning.
Table 1 shows an overview of our training setup for each stage. Appendix A.1 provides the complete training recipe for each stage, including hyperparameters and data sources. The compute cost of each PLM training stage is detailed in Tab. 7 in the Appendix.
this section cite: []

Section: Synthetic Data Generation and Scaling
The predominant paradigm for VLM training is to generate synthetic annotations as cheap alternatives to human-labeled data [1,10,11,[15][16][17][18]. Although seemingly promising to get the best results on benchmarks, the majority of such data shared in the community is derived from proprietary models. This trend makes it hard to decouple scientific progress from proprietary distillation impact. In this section, we explore the efficacy of the current paradigm for VLM training in a transparent manner. We design our data engine entirely from open-source models and scale the synthetic data generation to around 66.1M samples of images and videos. We establish the scaling laws of training from synthetic data on standard VLM tasks, including image, OCR/document, and video tasks.
this section cite: ['b0', 'b9', 'b10', 'b14', 'b15', 'b16', 'b17']

Section: Data Engine
Our data engine is designed to target base capabilities of VLMs for image and video understanding.
Image Data Engine. We generate short and long captions, as well as question-answer pairs, for natural images and those containing documents, diagrams, and text recognizable by optical character recognition (OCR). We prompt openly accessible Llama 3 [13] model to produce factual, detailed image captions while minimizing hallucinations. To create informative question-answer pairs, we utilize OCR data, captions, and other metadata, which are fed into the prompt of a text-only LLM.
this section cite: ['b12']

Section: Video Data Engine.
For videos, we first use an off-the-shelf scene detector [19] to extract video clips of approximately 30 seconds duration. Then, we extract the keyframes and generate frame-level captions using Llama 3, and video captions using our initial PLM trained with Stage 1 and Stage 3 data as shown in Table 2. We then employ an LLM to refine the frame-level and video captions by incorporating existing video metadata (e.g., action labels, time tags) into a cohesive, detailed video-level caption. Similarly, we generate question-answer pairs from the video-level captions.
The resulting synthetic data is large-scale and diverse -66.1M samples carefully curated from a variety of image and video sources including natural images, in-the-wild text, chart, figures, documents, egocentric and exocentric videos. Additional details are in Appendix I.
No Syth. (1B) No Syth. (3B) No Syth. (8B) Natural QA Err. = (6.3e+02 x GFLOPs) 0.11
this section cite: ['b18']

Section: Power Law Fit
Figure 2: Synthetic Scaling Plots. Relationship between Average Error across benchmarks and training compute (in floating-point operations) for various PLM models. We report average errors across Video QA tasks [8,[20][21][22][23][24], OCR QA tasks [25][26][27][28], and Natural Images tasks [29][30][31][32][33][34]. Model's performance using only human-labeled data subset are reported (No Syth.) as well as the actual power-law fit of each subcategory.
this section cite: ['b7', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32']

Section: Setup.
To establish power-law relationship between compute and validation-set errors of downstream benchmarks, we vary the scale of synthetic data, language model decoders (1B, 3B, and 8B), vision encoders (300M and 2B), and resolution/number of frames. For each configuration, we train a model with the 66.1M synthetic data from our data engine and 6.5M publicly available human-labeled data, following stage 2 training described in §2. At every 2M samples, we evaluate PLM on three categories of downstream benchmarks (VideoQA, OCR QA, Natural QA), constructed from 20 vision-language understanding benchmarks that provide a comprehensive and general evaluation of multi-modal large language models. We compute the pareto frontier of these data points and fit a power law relationship: Err. = (β × FLOP) α and compare the exponents α of the power function as scalability of each setup, where a smaller α implies better scaling.
Scaling with decoder size. Fig. 2 shows the scaling behavior of PLM across various LLM sizes. We show validation-set errors and training compute on a logarithmic scale, with the black linear line representing the power-law relationship between them. Different colors (green, turquoise, and blue) represent different language model scales (1B, 3B, 8B) while keeping the vision encoder size constant at 300M. As described in the setup section above, we show the power law fit of the pareto frontier in each benchmark category. We also show the results of PLM only trained on 4M human-labeled datasets as baselines, denoted with horizontal lines of each color. The gap from the horizontal line to the data point marks the impact of the synthetic data. Interestingly, all three categories of benchmarks demonstrate clear power-law relationship between compute and average benchmark errors, with the power law exponent (α) of -0.15, -0.20, and -0.11 for Video QA, OCR QA, and Natural Image QA, respectively. In Appendix B, we provide more details and extend the analysis to (1) scaling the encoder size, and (2) scaling the image resolution and video frames.
this section cite: []

Section: 1B 3B 8B

this section cite: []

Section: 2 4
Power Law Fit Challenging video tasks (HardQA [35][36][37][38][39][40][41]) do not scale well with synthetic data.
Limitation of synthetic data. In Fig. 3, we evaluate stage 2 on an extended set of video benchmarks. Specifically, we show the result of 7 challenging video tasks on fine-grained activity understanding [35][36][37][38][39], temporal grounding [40] and long-video reasoning [41]. Unlike generic, high-level understanding (e.g., "what is happening in this video"), the "challenging" tasks require a thorough understanding of video in space and time, and fine-grained semantic details. As shown, the challenging video tasks ("HardQA" in lavender, plum, magenta) show a poor scaling trend (-0.03) compared to general video QA (-0.15). The stark difference between the two power law fits shows that scaling synthetic data is only effective for established, base tasks. Extending VLMs to these more challenging, complex tasks still remain unsolved. Next, we address this challenge with high-quality human-annotated video data, PLM-FGQA and PLM-STC.
Existing QA Datasets PLM-FGQA (ours) # Samples 100K 200K 300K 400K 1x 58x 30x 26x 3x 18x 47x 9x 47x 7x 2x Action Rec. Object Rec. Mov. Direction Counting Obj. State. Pose Obj. Attributes Obj. Location Spatial Rel. Speed / Force Action Seq.
this section cite: ['b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39']

Section: Question
How many chakli snacks does the person flip?
this section cite: []

Section: Answer
The person flips three chakli snacks with a long metal skewer.
this section cite: []

Section: Question
Where is the metal skewer located at the beginning?
this section cite: []

Section: Answer
Resting on top of the pan, which is positioned on the left burner of the portable stove.
this section cite: []

Section: Question
In which direction is the person moving the sandpaper?
this section cite: []

Section: Answer
From the bottom of the baluster to the top in a vertical, oscillating motion.
this section cite: []

Section: Human-annotated High Quality Data
As shown in Fig. 3, the current paradigm with synthetic data has run out of steam. Training from tens of millions of synthetically annotated data hardly improves our model on new, challenging video benchmarks. Beyond standard VLM tasks, these benchmarks focus on advanced capabilities such as fine-grained activity understanding, temporal grounding, and long video understanding. Perhaps, the knowledge that these benchmarks examine is simply not present in the initial training set of our data engine nor in existing human-annotated data. Our community lacks high quality datasets for detailed visual understanding to start from, that covers what, where, when, and how of activities in video. To address this gap, we introduce two large-scale, human-annotated video datasets: PLM-FGQA is a fine-grained video QA dataset collected by asking human annotators to watch a short video segment and answer model-generated questions which focus on "what" activities humans perform and "how" they perform these activities. Question types include fine-grained recognition (action and object), fine-grained temporal perception (direction of movements, repetition counts, hand pose etc.), and fine-grained spatial understanding (object locations and spatial relationships). We use a multi-stage data engine to first extract video segments with salient actions from untrimmed videos through temporal clustering and shot-detection. Next, we generate questions and answers using either a text-only LLM or an early version of PLM. Finally, we refine the answers by asking humans to verify or replace them if they are incorrect, resulting in a high-quality QA pairs. Overall, we collect 2.4M question answer pairs from various open-access video datasets [42][43][44][45][46][47] spanning over 780k unique video clips from diverse domains (e.g., cooking, DIY, carpentry, automotive and bike repair) and viewpoints (egocentric and third-person); refer to Fig. 13 for domain statistics. This is nearly 8 times larger than the size of the largest existing human-annotated video QA dataset in the community [48]. Moreover, as illustrated by the breakdown of question typesfoot_0 in Fig. 4 (top-right), PLM-FGQA contains a large number of annotations about fine-grained details that have been largely missing in existing training video QA datasets [24,[49][50][51][52][53][54][55][56]. Please refer to Table 19 for comparison with existing datasets Table 20 for dataset examples and Appendix F for further details.
PLM-STC is a spatio-temporal video captioning dataset that offers detailed activity descriptions for each video. It includes timestamps ("when") of each activity and focuses on specific subjects identified by a masklet ("where"). We employ a two-stage annotation process to improve efficiency in collecting PLM-STC. In the first stage, annotators select interesting objects that exhibit significant motion changes in the video and use SAM 2 [57] to generate initial mask tublets, which they then refine to ensure high-quality spatial-temporal segmentation. For segments where the subject is out of frame, we automatically supplement "out of frame" caption. In the second stage, a separate set of annotators write temporally localized descriptions of the highlighted subject focusing on the changes in action across time in relation to the whole video.
this section cite: ['b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b23', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55']

Section: Spatio-temporal Captions (STC)
[0, 11] Out of frame. [12,67] The person wearing a jacket is running on a snow covered ground. She stops and turns to look the other person.
[0, 19] The man moves gracefully, using his hand gestures that closely resemble a dance in most of his actions. [20,31] The person moves from right to left.
[0, 81] A little girl moves back as a beluga whale approaches her face. [82,85] Out of frame. [86,98] She tries to feed the whale. Overall, we collect 194.2K spatio-temporal captions as the first existing large-scale dense video-region captioning dataset. We convert these spatio-temporal captions into three tasks for training: RCap (194.2K): Given the video region and timestamps, the model generates a caption; RTLoc (194.2K): Given the video region and caption, the model localizes the action; and RDCap (122.3K): Given the video region, the model generates dense, localized captions. In total, we construct 194.2K + 194.2K + 122.3K = 522.7K samples, of which 476.2K are used for training and the rest for constructing PLM-VideoBench. Please refer to Fig. 5 for dataset examples, Table 22 for comparison with existing datasets, Table 23 for dataset statistics and Appendix G for further details.
this section cite: ['b11', 'b65', 'b19', 'b29', 'b80', 'b83', 'b84', 'b96']

Section: PLM-VideoBench
Our high-quality human-annotated data offers VLMs to train for broader range of capabilities for holistic video understanding. However, existing video benchmarks are not adequately equipped to evaluate these. To this end, we introduce PLM-VideoBench, a novel benchmark focusing on specific activities (what) and their execution details (how) within spatio-temporal contexts (where and when).
Fine-Grained Question Answering (FGQA). In this task, a model must answer a multiple-choice question (MCQ) that probes nuanced, fine-grained activity understanding (e.g., painting "vertically" vs. "horizontally" in Fig. 6, first). We report multi-binary accuracy (MBAcc) [39] where each question is split into multiple binary choice questions. Our test set consists of 4,371 question-answer pairs. For more information, including statistics on video clips, segment duration, question types, and benchmark construction, see Table 21 and §F.2.
this section cite: ['b37']

Section: Smart Glasses Question Answering (SGQA).
In this task, a model must answer open-ended questions about activities and objects visible in an egocentric video stream recorded by a smartglasses device (see Fig. 6, second). The questions are designed to simulate real-world scenarios where a user would ask for assistance from their smart glasses. We manually collect the videos using commercially available smart glasses, providing a completely new, unique dataset that reflects modern use-cases such as online AI video assistance and activity coaching. For evaluation, we use LLM-judge accuracy with an open-access model (Llama3.3 70B). The test set consists of 665 human-annotated question-answer pairs. See Appendix H for more details.
Video Region Captioning (RCap). In this task, a model must generate a detailed description of an event involving a subject of interest in the video. Given a region masklet and a specified time interval, the model is required to output a caption that accurately describes the event occurring within that interval. Compared to traditional video captioning [47,58,59] where the aim is to generate a video-level caption, the goal is to generate a region-level caption tied to a specific subject (e.g., a person, object or animal) (see Fig. 6, third). The test set contains 10,060 human-annotated instances and we report LLM-judge accuracy with Llama3.3 70B. See Appendix C.3 for details.
this section cite: ['b45', 'b56', 'b57']

Section: Region Temporal Localization (RTLoc).
In this task, a model must identify the precise time interval within the video when the specified event takes place for the given subject. Given a video, a region masklet and a text description of the event, the model is required to output the start and
FGQA Question How is the person moving the paintbrush to paint the wall? Choices (A) Vertically from top to bottom. (B) Horizontally from left to right. Answer (A) Vertically from top to bottom.
this section cite: []

Section: Question
Does this pasta look strained enough?
this section cite: []

Section: Answer
The pasta looks perfectly strained, with excess water removed.
this section cite: []

Section: Question
Describe all the actions performed by the highlighted object between frames [0, 31].
this section cite: []

Section: Answer
A black and white dog runs towards the camera, spins around and runs back away
this section cite: []

Section: Question
Given the shaded region, when does "The cat gets up and moves in front, towards the toy." occur in the video. [55,64] Question Provide a dense caption for the region indicated by the shaded mask with the start and end frames of all actions of the subject, along with a brief description of each action.
this section cite: ['b53', 'b62']

Section: Answer

this section cite: []

Section: Answer
[0, 1]: The person to swing in the monkey bar. [81,98] The person hang from the toller monkey bar.
this section cite: ['b79', 'b96']

Section: SGQA
RCap RTLoc RDCap end timestamps that correspond to the occurrence of the event (see Fig. 6 fourth). Notably, this task is the inverse of RCap -instead of generating the caption, the model receives it as input and generates the corresponding time interval. We filter the test set to include only the captions that are unambiguously localized, i.e., they map to a single time window in the video. As a result, the test set size is reduced to 7,910 instances compared to RCap. We report average recall@1 over IoU thresholds (0.3, 0.5, 0.7, 0.9). See Appendix C.3 for details.
this section cite: []

Section: Region Dense Video Captioning (RDCap).
In this task, a model must generate a detailed description of all events involving a specific subject of interest (e.g., person, animal, or object) in a video. Given a video and a region masklet, the model must produce a sequence of (start, end, caption) tuples that cover the entire duration of the video, including periods when the subject is not visible (see Fig. 6, last). This task is a composition of RTLoc and RCap, requiring the model to produce both temporal windows for events as well as captions directly from the video. The test set contains 2,620 samples and we report the SODA score [60] which uses an LLM judge. See Appendix C.3 for details.
this section cite: ['b58']

Section: Experiments
We first overview the baselines and evaluation setting ( §5.1). We then compare benchmark results of PLMs with the baselines on a broad collection of image ( §5.2) and video ( §5.3) tasks as well as on our PLM-VideoBench ( §5.4). Finally, we provide analyses on data and model ablations ( §5.5).
this section cite: []

Section: Setup
We compare PLMs against the following two classes of baselines:
• Proprietary models such as GPT-4o [61] (gpt-4o-2024-11-20), Gemini-Pro 1.5 [62] and Gemini-Flash 2.0 [63]. We use API calls to evaluate these models.
• Open-access models such as Molmo-O [11], LLaVA-OneVision [64], Qwen2.5-VL [15] and InternVL2.5 [10] -state-of-the-art open-access models, for which model scale, architecture and inference code are available. We use the official inference code for all models.
Inference protocol. For mask inputs in PLM-VideoBench, we overlay a colored box on the video frames to specify the regions. We report validation set performance unless specified (in brackets) under the benchmark name. Metrics marked with † use LLM as a judge. Complete implementation details including inference hyper-parameters, task prompts, judge prompts and proprietary model evaluation protocol can be found in Appendix C.4.
this section cite: ['b59', 'b60', 'b61', 'b10', 'b62', 'b14', 'b9']

Section: Image Benchmark Results
We evaluate PLM on a total of 20 image benchmarks. Charts, Diagrams and Documents: answer questions that require parsing images of documents and diagrams; Image Captioning: generate a short/detailed caption, Perception and Reasoning: answer questions of varying difficulty about objects, actions, functional correspondence, multi-view reasoning, spatial layout etc. and Hallucination: evaluate robustness to hallucinated details. More details are in Appendix C.1. Table 4: Video benchmark results. PLM versus proprietary models and open-access baselines of comparable scale. Cells with * are reported numbers from literature and the remaining are reproduced using official code.
Table 3 shows our results. Overall, PLM shows strong performance on a wide spectrum of image benchmarks with solely from open-access data with a white-box data engine. Additionally, we report Image Grounding task results on RefCOCO/+/g [77] datasets in Appendix Table 15, and show that PLM outperforms both specialist models as well as the VLM baselines in all model scales.
this section cite: ['b75']

Section: Video Benchmark Results
We evaluate PLM on a total of 25 video benchmarks. We divide these into the following categories.
Video Captioning: generate a short caption for a video, or a dense description of all events; Short video QA: answer a question about a short video (few seconds to a minute), either by selecting from a list of options, or providing a free-form answer; Long video QA: answer a question as before, about a much longer video (minutes to hours); Fine-grained QA: answer detailed questions about spatial location, motion, temporal information etc.; and Hallucination: evaluate the robustness of video models to hallucinated details about objects and events.
Table 4 shows video captioning, video QA, fine-grained video QA, and video hallucination results. We achieve strong results on widely adopted benchmarks, despite only using open-access data mix free from proprietary model artifacts, outperforming both the open-access and proprietary models.
Further, we achieve competitive performance on the majority of challenging benchmarks, such as EgoSchema (68.8 %), MotionBench (61.4 %), TOMATO (33.2 %), TempCompass (72.7 %), TemporalBench (28.3 &), Charades-STA (58.6 %), and more. All our model scales show strong performance against both proprietary models as well as open-access baselines of same scale.
Lastly, we also show that PLMs at all scale greatly outperform existing approaches on captioning tasks and hallucination detection tasks, owing to our focus on detailed, fine-grained spatio-temporal annotations in our human-annotated data collection.
this section cite: []

Section: PLM-VideoBench Results
We report the result on our proposed benchmark PLM-VideoBench from §4.1 in Table 5. We evaluate our PLM as well as (proprietary and open-access) baselines. In addition, we provide human performance of each subtask in the first row. The results show a significant gap between the baselines and PLM. Proprietary baselines and open-source baselines alike perform reasonably on FGQA tasks, though still 6.5 points lower than PLM (61.2 vs 67.7). Note that the human performance varies based on the nature of the task and evaluation metrics. For example, FGQA human scores are naturally higher than RCap because the task is structured (select the correct option vs. open-ended) and the metric is objective (accuracy vs. LLM-judge accuracy).
this section cite: []

Section: Ablation Studies

this section cite: []

Section: Setup.
We perform an ablation study to assess the importance of each of our proposed data, both synthetic and human-annotated. We start with PLM 3B after stage 2 training, and finetune on 4M short image and video SFT data mixfoot_1 for the data ablation. We evaluate and report average video benchmark performance across five categories -video captioning, short video QA, fine-grained QA, and video hallucination, as well as spatial and temporal tasks, PLM-VideoBench and three image categories -image OCR, image captioning, and image perception. Full details are in Appendix A.3.
Discussion. First, we observe that stage 2 synthetic data training boosts model performance across the board. Moreover, adding our PLM-STC data further improves a variety of benchmarks, including PLM-STC (+27.4 points), video captioning (+2.4 points), and most importantly, spatial and temporal tasks (+6.8 points). Adding our PLM-FGQA data improves a distinct set of categories for finegrained activity understanding; PLM-FGQA (+13.1 points), PLM-SGQA (+7.3 points), Fine-grained video tasks (+1.3 points), video hallucination tasks (+3.0 points), and spatial and temporal tasks (+2.2 points). Using our human-annotated data altogether results in the best performance overall. Further in Fig. 7, we show that our human-annotated data improves upon HardQA [35][36][37][38][39][40][41], effectively addressing the limitations of synthetic data discussed in §3.2.
Finally, we conduct an ablation study (Tab. 17 in Appendix), where Stage 2 and 3 are merged into a single training phase using a combined data blend under the Stage 3 setup (36 tiles, 32 frames). Despite nearly doubling the total training FLOPs, this unified setting yields worse results than the three-stage pipeline-dropping 1.2 points on image benchmarks and 2.0 points on video benchmarks.
this section cite: ['b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39']

Section: Related Work
Vision-Language Models. Building on the strengths of large language models (LLMs), several visionlanguage models (VLMs) have recently been proposed for image understanding [1, 13, 18, 72, 78-82], video understanding [52,[83][84][85][86][87][88][89] and joint understanding of both images and videos [10,16,64,90]. These works employ several modeling advancements such as dynamic high resolution inputs [78], adaptive token compression [87,91], and multimodal positional embeddings [16].
Open source, open data VLMs. Training data is a key component in developing powerful VLMs. Many existing approaches train on proprietary data that is not released to the community [61-63, 92, 93] or on data generated using proprietary models (e.g., GPT4o) [3], effectively distilling the closed models. Doing so make measuring scientific progress difficult and limits research on how to train VLMs ground-up. Molmo [11] proposes a class of open-data models, however, they are image VLMs trained on relatively small-scale data, limiting their performance as our experiments will show.
VLM Benchmarks. Several benchmarks have been proposed to assess the capabilities of VLMs. Popular image benchmarks cover broad perception and reasoning [29, 33, 68, 69, 71, 72, 94-100] as well as capabilities like image captioning [101][102][103], document/diagram understanding [26-28, 65-67, 70, 104-106], mathematical reasoning [107][108][109], visual grounding [77,110] and hallucination [32,111]. Popular video benchmarks cover video question answering [8, 20, 21, 23, 24, 50-52, 84, 112-117], video captioning [47,59,74,[118][119][120][121], and hallucination in videos [37,76]. Many of these video benchmarks remain image-centric -they have questions that can be answered with a few frames. Video-centric reasoning in benchmarks has been relatively neglected with benchmarks proposed only recently for long video understanding [22,35,41,48,[122][123][124][125][126] and fine-grained, temporal reasoning [36,38,39,75,127]. We introduce PLM-VideoBench-a benchmark suite aimed at the core, video-centric capabilities that current benchmarks neglect, namely fine-grained activity understanding and spatio-temporally grounded reasoning.
this section cite: ['b50', 'b81', 'b82', 'b83', 'b84', 'b85', 'b86', 'b87', 'b9', 'b15', 'b62', 'b88', 'b76', 'b85', 'b89', 'b15', 'b2', 'b10', 'b99', 'b100', 'b101', 'b105', 'b106', 'b107', 'b75', 'b108', 'b30', 'b109', 'b45', 'b57', 'b72', 'b116', 'b117', 'b118', 'b119', 'b35', 'b74', 'b21', 'b33', 'b39', 'b46', 'b120', 'b121', 'b122', 'b123', 'b124', 'b34', 'b36', 'b37', 'b73', 'b125']

Section: Conclusion
This work presents Perception Language Model (PLM), a fully-reproducible vision-language model to transparently tackle visual perception tasks without distillation of private black-box models. We trained PLM using data from existing open-access datasets and synthetic samples generated by our data engine. We identified gaps in detailed video understanding capabilities that cannot be filled with synthetic data. In response, we collected 2.8M human-labels for fine-grained video question answering and spatio-temporally grounded captioning, and created a new benchmark, PLM-VideoBench, to evaluate these capabilities. We hope our open dataset, benchmark, and models will foster transparent research in visual perception. 6.52M Grounding (41 K) 6.52M Existing Data Total 69.1M PLM Image and Video Data Image QA Q: Which Place is shown in the image? A: A serene lake scene with a monument featuring a man on horseback. Image OCR Q: What location does this map represent? A: The map is of Kansai International Airport (KIX), which is located in Japan. Multi Image Q: Spot the difference between these two images. A: The color of the pole on the dog's right changes from black to light red. Video QA Q: What is the person doing in the video? A: Person is wearing ring in left hand and cutting the onions with a knife. Video Cap. Q: Briefly caption the video. A: The video shows a person making avocado sandwich by mashing the avocado and spreading it on bread. PLM-FGQA Q: How does a person pour the tea into the mug? A: The person moves the teapot up and down three times while pouring the tea into the mug. PLM-RTLoc Q: Localize the event "Person facing towards the camera is dancing." for the shaded region? A: [0, 13] PLM-RCap Q: Describe the highlighted subject in the video between frames [0, 13]. A: A person facing towards the camera is dancing. PLM-RDCap Q: Provide a dense caption for the shaded region. A: [0, 12] Kid is playing hide and seek and hides behind a tree. [13, 16] The kid reappears. Others (500K) Figure 8: The figure provides an overview of the datasets used in the paper. PLM is trained with 47.8M synthetic image and 18.4M synthetic video, and 2.9M human-labeled video samples. Our data enables PLM to perform a variety of tasks, including standard tasks like Image, Multi-image, and Video QA, as well as new video tasks such as Fine-grained QA (FGQA), Region Temporal Localization (RTLoc), Region Captioning (RCap), and Region Detailed Captioning (RDCap).
this section cite: []

Section: References
Ref_id:b0 Title: Visual instruction tuning Year: (2023)
Ref_id:b1 Title:  Year: (2024)
Ref_id:b2 Title: Sharegpt4v: Improving large multi-modal models with better captions Year: (2024)
Ref_id:b3 Title: Finevideo: behind the scenes Year: (2024)
Ref_id:b4 Title: Video instruction tuning with synthetic data Year: (2024)
Ref_id:b5 Title: Sharegpt4video: Improving video understanding and generation with better captions Year: (2024)
Ref_id:b6 Title: Eagle-2: Faster inference of language models with dynamic draft trees Year: (2024)
Ref_id:b7 Title: Hierarchical encoder for video+ language omni-representation pre-training Year: (2020)
Ref_id:b8 Title: Where does it exist: Spatio-temporal video grounding for multi-form sentences Year: (2020)
Ref_id:b9 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2024)
Ref_id:b10 Title: Open weights and open data for state-of-the-art multimodal models Year: (2024)
Ref_id:b11 Title: Perception encoder: The best visual embeddings are not at the output of the network Year: (2025)
Ref_id:b12 Title: The llama 3 herd of models Year: (2024)
Ref_id:b13 Title: Segment anything Year: (2023)
Ref_id:b14 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b15 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b16 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b17 Title: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models Year: (2023)
Ref_id:b18 Title:  Year: ()
Ref_id:b19 Title: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2024)
Ref_id:b20 Title: Star: A benchmark for situated reasoning in real-world videos Year: ()
Ref_id:b21 Title: Egoschema: A diagnostic benchmark for very long-form video language understanding Year: (2024)
Ref_id:b22 Title: Mvbench: A comprehensive multi-modal video understanding benchmark Year: (2024)
Ref_id:b23 Title: Perception test: A diagnostic benchmark for multimodal video models Year: (2024)
Ref_id:b24 Title: Chartqa: A benchmark for question answering about charts with visual and logical reasoning Year: (2022-05)
Ref_id:b25 Title: Docvqa: A dataset for vqa on document images Year: (2021)
Ref_id:b26 Title: 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Year: (2022)
Ref_id:b27 Title: Ocrbench: on the hidden mystery of ocr in large multimodal models Year: (2024)
Ref_id:b28 Title: Ok-vqa: A visual question answering benchmark requiring external knowledge Year: (2019)
Ref_id:b29 Title: Making the v in vqa matter: Elevating the role of image understanding in visual question answering Year: (2017)
Ref_id:b30 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b31 Title: Vizwiz: nearly real-time answers to visual questions Year: (2010)
Ref_id:b32 Title: Towards vqa models that can read Year: (2019)
Ref_id:b33 Title: Cg-bench: Clue-grounded question answering benchmark for long video understanding Year: (2024)
Ref_id:b34 Title: Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models Year: (2024)
Ref_id:b35 Title: Eventhallusion: Diagnosing event hallucinations in video llms Year: (2024)
Ref_id:b36 Title: Motionbench: Benchmarking and improving fine-grained video motion understanding for vision language models Year: (2025)
Ref_id:b37 Title: Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models Year: (2024)
Ref_id:b38 Title: Tall: Temporal activity localization via language query Year: (2017)
Ref_id:b39 Title: Lvbench: An extreme long video understanding benchmark Year: (2024)
Ref_id:b40 Title: Howto100m: Learning a text-video embedding by watching hundred million narrated video clips Year: (2019)
Ref_id:b41 Title: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Year: ()
Ref_id:b42 Title: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Year: (2023)
Ref_id:b43 Title: Coin: A large-scale dataset for comprehensive instructional video analysis Year: (2019)
Ref_id:b44 Title: Crosstask weakly supervised learning from instructional videos Year: (2019)
Ref_id:b45 Title: Towards automatic learning of procedures from web instructional videos Year: (2018)
Ref_id:b46 Title: A long video question answering dataset and benchmark Year: (2024)
Ref_id:b47 Title: Encoding and controlling global semantics for long-form video question answering Year: (2024)
Ref_id:b48 Title: Next-qa: Next phase of question-answering to explaining temporal actions Year: (2021)
Ref_id:b49 Title: Activitynet-qa: A dataset for understanding complex web videos via question answering Year: (2019)
Ref_id:b50 Title: Video-chatgpt: Towards detailed video understanding via large vision and language models Year: (2023)
Ref_id:b51 Title: Clevrer: Collision events for video representation and reasoning Year: (2019)
Ref_id:b52 Title: The kinetics human action video dataset Year: (2017)
Ref_id:b53 Title: The" something something" video database for learning and evaluating visual common sense Year: (2017)
Ref_id:b54 Title: Connecting vision and language with video localized narratives Year: (2023)
Ref_id:b55 Title: Segment anything in images and videos Year: (2024)
Ref_id:b56 Title: Activitynet: A largescale video benchmark for human activity understanding Year: (2015)
Ref_id:b57 Title: Vatex: A large-scale, high-quality multilingual dataset for video-and-language research Year: (2019)
Ref_id:b58 Title: Soda: Story oriented dense video captioning evaluation framework Year: (2020)
Ref_id:b59 Title: Gpt-4o system card Year: (2024)
Ref_id:b60 Title: Gemini: a family of highly capable multimodal models Year: (2023)
Ref_id:b61 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b62 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b63 Title: Advancing chart question answering with robust chart component recognition Year: (2024)
Ref_id:b64 Title: Towards vqa models that can read Year: (2019)
Ref_id:b65 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b66 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b67 Title: A-okvqa: A benchmark for visual question answering using world knowledge Year: (2022)
Ref_id:b68 Title: Seed-bench: Benchmarking multimodal llms with generative comprehension Year: (2023)
Ref_id:b69 Title: Blink: Multimodal large language models can see but not perceive Year: (2025)
Ref_id:b70 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b71 Title: Visual spatial reasoning Year: (2023)
Ref_id:b72 Title: Recipes for training and evaluating large video description models Year: (2024)
Ref_id:b73 Title: Tempcompass: Do video llms really understand videos? arXiv preprint Year: (2024)
Ref_id:b74 Title: Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models Year: (2024)
Ref_id:b75 Title: Revisiting referring expression comprehension evaluation in the era of large multimodal models Year: (2024)
Ref_id:b76 Title: Llava-next: Improved reasoning, ocr, and world knowledge Year: (2024-01)
Ref_id:b77 Title: mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration Year: (2024)
Ref_id:b78 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b79 Title: On pretraining for visual language models Year: (2024)
Ref_id:b80 Title: Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding Year: (2024)
Ref_id:b81 Title: Videochat: Chat-centric video understanding Year: (2023)
Ref_id:b82 Title: Videogpt+: Integrating image and video encoders for enhanced video understanding Year: (2024)
Ref_id:b83 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b84 Title: A powerful video-language model supporting long-context video input Year: (2024)
Ref_id:b85 Title: Spatiotemporal adaptive compression for long video-language understanding Year: (2024)
Ref_id:b86 Title: Efficient long video understanding via large language models Year: (2025)
Ref_id:b87 Title: Long-form video understanding with large language model as agent Year: (2024)
Ref_id:b88 Title: Anymal: An efficient and scalable any-modality augmented language model Year: (2024)
Ref_id:b89 Title: Don't look twice: Faster video transformers with run-length tokenization Year: (2024)
Ref_id:b90 Title: Gpt-4v(ision) system card Year: (2023)
Ref_id:b91 Title: The claude 3 model family: Opus, sonnet, haiku Year: (2024)
Ref_id:b92 Title: Mmbench: Is your multi-modal model an all-around player? Year: (2024)
Ref_id:b93 Title: A comprehensive evaluation benchmark for multimodal large language models Year: (2023)
Ref_id:b94 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b95 Title: Are we on the right way for evaluating large vision-language models? arXiv preprint Year: (2024)
Ref_id:b96 Title: Wildvision: Evaluating vision-language models in the wild with human preferences Year: (2024)
Ref_id:b97 Title: Interleaved multi-image instruction tuning Year: (2024)
Ref_id:b98 Title: A comprehensive benchmark for robust multi-image understanding Year: (2024)
Ref_id:b99 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b100 Title: Nocaps: Novel object captioning at scale Year: (2019)
Ref_id:b101 Title: From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions Year: (2014)
Ref_id:b102 Title: Charting gaps in realistic chart understanding in multimodal llms Year: (2024)
Ref_id:b103 Title: From recognition to cognition: Visual commonsense reasoning Year: (2019)
Ref_id:b104 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b105 Title: Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? Year: (2025)
Ref_id:b106 Title: Evaluating mathematical reasoning of foundation models in visual contexts Year: (2023)
Ref_id:b107 Title: Measuring multimodal mathematical reasoning with math-vision dataset Year: (2024)
Ref_id:b108 Title: Visual genome: Connecting language and vision using crowdsourced dense image annotations Year: (2017)
Ref_id:b109 Title: Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models Year: (2024)
Ref_id:b110 Title: Tgif-qa: Toward spatiotemporal reasoning in visual question answering Year: (2017)
Ref_id:b111 Title: Tvqa: Localized, compositional video question answering Year: (2018)
Ref_id:b112 Title: Videobench: A comprehensive benchmark and toolkit for evaluating video-based large language models Year: (2023)
Ref_id:b113 Title: Scrutinizing lmms over dense temporal reasoning with short videos Year: (2024)
Ref_id:b114 Title: Mmbench-video: A long-form multi-shot benchmark for holistic video understanding Year: (2024)
Ref_id:b115 Title: Tvbench: Redesigning video-language evaluation Year: (2024)
Ref_id:b116 Title: Msr-vtt: A large video description dataset for bridging video and language Year: (2016)
Ref_id:b117 Title: Collecting highly parallel data for paraphrase evaluation Year: (2011)
Ref_id:b118 Title: Dense-captioning events in videos Year: (2017)
Ref_id:b119 Title: Auroracap: Efficient, performant video detailed captioning and a new benchmark Year: (2024)
Ref_id:b120 Title: Movieqa: Understanding stories in movies through question-answering Year: (2016)
Ref_id:b121 Title: Longvideobench: A benchmark for long-context interleaved video-language understanding Year: (2025)
Ref_id:b122 Title: Moviechat: From dense token to sparse memory for long video understanding Year: (2024)
Ref_id:b123 Title: Mlvu: A comprehensive benchmark for multi-task long video understanding Year: (2024)
Ref_id:b124 Title: Apollo: An exploration of video understanding in large multimodal models Year: (2024)
Ref_id:b125 Title: Actionatlas: A videoqa benchmark for domain-specialized action recognition Year: (2024)
Ref_id:b126 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b127 Title: Pytorch fsdp: experiences on scaling fully sharded data parallel Year: (2023)
Ref_id:b128 Title: Flashattention-2: Faster attention with better parallelism and work partitioning Year: (2023)
Ref_id:b129 Title:  Year: (2017)
Ref_id:b130 Title: Pdf association dataset (pdfa) Year: (2024)
Ref_id:b131 Title: Industry documents library (idl) Year: (2024)
Ref_id:b132 Title: Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models Year: (2024)
Ref_id:b133 Title: Objects365: A large-scale, high-quality dataset for object detection Year: (2019)
Ref_id:b134 Title: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV Year: (2020)
Ref_id:b135 Title: Merlot: Multimodal neural script knowledge models Year: (2021)
Ref_id:b136 Title: Spoken moments: Learning joint audio-visual representations from video descriptions Year: (2021)
Ref_id:b137 Title: Hollywood in homes: Crowdsourcing data collection for activity understanding Year: (2016)
Ref_id:b138 Title: Localizing moments in video with natural language Year: (2017)
Ref_id:b139 Title: Reasoning in the wild with 2.8 m challenging questions Year: (2025)
Ref_id:b140 Title: Dvqa: Understanding data visualizations via question answering Year: (2018)
Ref_id:b141 Title: Plotqa: Reasoning over scientific plots Year: (2020-03)
Ref_id:b142 Title: Mapqa: A dataset for question answering on choropleth maps Year: (2022)
Ref_id:b143 Title: Ocr-vqa: Visual question answering by reading text in images Year: (2019)
Ref_id:b144 Title: Connecting vision and language with localized narratives Year: (2020)
Ref_id:b145 Title: Figureqa: An annotated figure dataset for visual reasoning Year: (2018)
Ref_id:b146 Title: The hateful memes challenge: Detecting hate speech in multimodal memes Year: (2020)
Ref_id:b147 Title: Clevr: A diagnostic dataset for compositional language and elementary visual reasoning Year: (2016)
Ref_id:b148 Title: Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning Year: (2021)
Ref_id:b149 Title: Geomverse: A systematic evaluation of large models for geometric reasoning Year: (2023)
Ref_id:b150 Title: Robut: A systematic study of table qa robustness against human-annotated adversarial perturbations Year: (2023-07)
Ref_id:b151 Title: Unlocking the conversion of web screenshots into html code with the websight dataset Year: (2024)
Ref_id:b152 Title: Visual7w: Grounded question answering in images Year: (2016)
Ref_id:b153 Title: Tallyqa: Answering complex counting questions Year: (2019)
Ref_id:b154 Title: Automatikz: Text-guided synthesis of scientific vector graphics with tikz Year: (2024)
Ref_id:b155 Title: Exploring models and data for image question answering Year: (2015)
Ref_id:b156 Title: Chart-to-text: Generating natural language descriptions for charts by adapting the transformer model Year: (2020-12)
Ref_id:b157 Title: Vistext: A benchmark for semantically rich chart captioning Year: ()
Ref_id:b158 Title: Finqa: A dataset of numerical reasoning over financial data Year: (2021-11)
Ref_id:b159 Title: Scene text visual question answering Year: (2019)
Ref_id:b160 Title: Tat-qa: A question answering benchmark on a hybrid of tabular and textual content in finance Year: (2021-08)
Ref_id:b161 Title:  Year: (2024)
Ref_id:b162 Title: Raven: A dataset for relational and analogical visual reasoning Year: (2019)
Ref_id:b163 Title: The iam-database: An english sentence database for offline handwriting recognition Year: ()
Ref_id:b164 Title: Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning Year: ()
Ref_id:b165 Title: Textcaps: a dataset for image captioning with reading comprehension Year: (2020)
Ref_id:b166 Title: Screen2words: Automatic mobile ui summarization with multimodal learning Year: (2021)
Ref_id:b167 Title: Visual spatial reasoning Year: (2023)
Ref_id:b168 Title: Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension Year: (2017)
Ref_id:b169 Title: Visualmrc: Machine reading comprehension on document images Year: (2021)
Ref_id:b170 Title: A dataset of clinically generated visual questions and answers about radiology images Year: (2018)
Ref_id:b171 Title: Hitab: A hierarchical table dataset for question answering and natural language generation Year: (2022-05)
Ref_id:b172 Title: Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning Year: ()
Ref_id:b173 Title:  Year: (2023)
Ref_id:b174 Title: Mimic-it: Multi-modal in-context instruction tuning Year: (2023)
Ref_id:b175 Title: Multihiertt: Numerical reasoning over multi hierarchical tabular and textual data Year: (2022-05)
Ref_id:b176 Title: A corpus for reasoning about natural language grounded in photographs Year: (2019-07)
Ref_id:b177 Title: Learning to describe differences between pairs of similar images Year: (2018-11)
Ref_id:b178 Title: Vision datasets: A benchmark for vision-based industrial inspection Year: (2023)
Ref_id:b179 Title: Imagine this! scripts to compositions to videos Year: (2018)
Ref_id:b180 Title: Image retrieval from contextual descriptions Year: (2022)
Ref_id:b181 Title: Discovering states and transformations in image collections Year: (2015)
Ref_id:b182 Title: Webqa: Multihop and multimodal qa Year: (2022)
Ref_id:b183 Title: Neural naturalist: Generating fine-grained image comparisons Year: (2019)
Ref_id:b184 Title: Abstract encoding of stories, objects, and pictures Year: (2021)
Ref_id:b185 Title: Recipeqa: A challenge dataset for multimodal comprehension of cooking recipes Year: (2018)
Ref_id:b186 Title: Robust change captioning Year: (2019)
Ref_id:b187 Title: iedit: Localised text-guided image editing with weak supervision Year: (2024)
Ref_id:b188 Title: Compositional semantic parsing on semi-structured tables Year: (2015-07)
Ref_id:b189 Title: Syntax-aware network for handwritten mathematical expression recognition Year: (2022)
Ref_id:b190 Title: Docci: Descriptions of connected and contrasting images Year: (2024)
Ref_id:b191 Title: A picture is worth more than 77 text tokens: Evaluating clip-style models on dense captions Year: (2024)
Ref_id:b192 Title: Image captioning via re-aligning alt-text Year: (2024)
Ref_id:b193 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b194 Title: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models Year: (2015)
Ref_id:b195 Title: Referitgame: Referring to objects in photographs of natural scenes Year: (2014)
Ref_id:b196 Title: Mementos: A comprehensive benchmark for multimodal large language model reasoning over image sequences Year: (2024)
Ref_id:b197 Title: Hierarchical video-moment retrieval and step-captioning Year: (2023)
Ref_id:b198 Title: Human-centric spatio-temporal video grounding with visual transformers Year: (2021)
Ref_id:b199 Title:  Year: (2023)
Ref_id:b200 Title: Towards interpretable math word problem solving with operation-based formalisms Year: (2019)
Ref_id:b201 Title: Less is more for alignment Year: (2023)
Ref_id:b202 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b203 Title: Finetuned language models are zero-shot learners Year: ()
Ref_id:b204 Title: Free dolly: Introducing the world's first truly open instruction-tuned llm Year: (2023)
Ref_id:b205 Title: Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing Year: (2024)
Ref_id:b206 Title: Cider: Consensus-based image description evaluation Year: (2015)
Ref_id:b207 Title: Lmms-eval: Reality check on the evaluation of large multimodal models Year: (2024)
Ref_id:b208 Title: Vlmevalkit: An open-source toolkit for evaluating large multimodality models Year: (2024)
Ref_id:b209 Title: Grounding dino: Marrying dino with grounded pre-training for open-set object detection Year: (2023)
Ref_id:b210 Title: Universal instance perception as object discovery and retrieval Year: (2023)
Ref_id:b211 Title: One-peace: Exploring one general representation model toward unlimited modalities Year: (2023)
Ref_id:b212 Title: Language-image models with 3d understanding Year: (2025)
Ref_id:b213 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b214 Title: Ego4d goal-step: Toward hierarchical understanding of procedural activities Year: (2023)
Ref_id:b215 Title: HT-step: Aligning instructional articles with how-to videos Year: (2023)
Ref_id:b216 Title: Learning to ground instructional articles in videos through narrations Year: (2023-10)
Ref_id:b217 Title: Internvideo2: Scaling video foundation models for multimodal video understanding Year: (2024)
Ref_id:b218 Title: Uboco: Unsupervised boundary contrastive learning for generic event boundary detection Year: (2021)
Ref_id:b219 Title: Fast and unsupervised action boundary detection for action segmentation Year: (2022)
Ref_id:b220 Title: PySceneDetect: Video Cut Detection and Analysis Tool Year: ()
Ref_id:b221 Title: Out of time: automated lip sync in the wild Year: (2016)
Ref_id:b222 Title: Unlocking exocentric video-language data for egocentric video representation learning Year: (2024)
Ref_id:b223 Title: Understanding human hands in contact at internet scale Year: (2020)
Ref_id:b224 Title: Scaling open-vocabulary object detection Year: (2023)
Ref_id:b225 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b226 Title: Perceiver io: A general architecture for structured inputs & outputs Year: (2022)
Ref_id:b227 Title: A benchmark dataset and evaluation methodology for video object segmentation Year: (2016)
Ref_id:b228 Title: The 2019 davis challenge on vos: Unsupervised multi-object segmentation Year: (2019)
Ref_id:b229 Title: Weakly supervised actor-action segmentation via robust multi-task ranking Year: (2017)
Ref_id:b230 Title: Unsupervised deep metric learning via orthogonality based probabilistic loss Year: (2020)
Ref_id:b231 Title: Grounded video description Year: (2019)
Ref_id:b232 Title: Urvos: Unified referring video object segmentation network with a large-scale benchmark Year: (2020)
Ref_id:b233 Title: Human-centric spatio-temporal video grounding with visual transformers Year: (2021)
Ref_id:b234 Title: Mevis: A large-scale benchmark for video segmentation with motion expressions Year: (2023)
Ref_id:b235 Title: Merlot reserve: Neural script knowledge through vision and language and sound Year: (2022)
