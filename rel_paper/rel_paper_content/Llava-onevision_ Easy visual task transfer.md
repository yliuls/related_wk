Title: LLaVA-OneVision: Easy Visual Task Transfer
Abstract: We present LLaVA-OneVision, a family of open large multimodal models (LMMs) developed by consolidating our insights into data, models, and visual representations in the LLaVA-NeXT blog series. Our experimental results demonstrate that LLaVA-OneVision is the first single model that can simultaneously push the performance boundaries of open LMMs in three important computer vision scenarios: single-image, multi-image, and video scenarios. Importantly, the design of LLaVA-OneVision allows strong transfer learning across different modalities/scenarios, yielding new emerging capabilities. In particular, strong video understanding and cross-scenario capabilities are demonstrated through task transfer from images to videos.

Section: Introduction
It is a core aspiration in AI to build general-purpose assistants with Large Multimodal Models (LMM) [67]. LLaVA-OneVision is an open model, continuing to advance the line of research in building large vision-and-language assistant (LLaVA) [83] that can follow diverse instructions to complete a variety of computer vision tasks in the wild. As a cost-efficient recipe, it is typically developed by connecting vision encoders with large language models (LLM) using a simple connection module.
The first LLaVA model [83] demonstrates impressive multimodal chat abilities, sometimes exhibiting the behaviors similar to GPT-4V on previously unseen images and instructions for the first time. LLaVA-1.5 [81] significantly expands and improves the capabilities by incorporating more academicrelated instruction data, achieving SoTA performance on a dozens of benchmarks with a data-efficient recipe. LLaVA-NeXT [82] inherits this property, further pushing performance boundaries through three key techniques: AnyRes for handling high-resolution images, expanding high-quality instruction data, and utilizing the best open LLM available at the time.
LLaVA-NeXT provides an extendable and scalable prototype, which facilitates several parallel explorations, reported in the LLaVA-NeXT blog series [82,169,65,64,68]: https://llava-vl.github.io/blog/
• The Video blog [169] shows that the image-only-trained LLaVA-NeXT model is surprisingly strong on video tasks with zero-shot modality transfer, due to the design of AnyRes to digest any vision signals as a sequence of images.
• The Stronger blog [65] demonstrates the LLM model scaling succuss of this cost-efficient strategy. By simply scaling up the LLM, it achieves performance comparable to GPT-4V on selected benchmarks.
• The Ablation blog [64] summarizes our empirical exploration except the visual instruction data itself, including the choice of architectures (scaling of LLM & vision encoder), visual representations (resolution & #tokens), as well as training strategies (trainable modules & high-quality data) in the pursuit of data scaling success.
• The Interleave blog [68] describes the strategies to extend and improve the capability in new scenarios including multi-image, multi-frame (video) and multi-view (3D), while maintaining the single-image performance.
These explorations, conducted within a fixed compute budget, aimed to offer useful insights along the way as we navigate the project, rather than push performance limits. During the process, we have also been accumulating and curating a large collection of the high-quality datasets from January to June. By consolidating these insights and execute the experiments with "yolo run" on newly accumulated larger datasets, we introduce LLaVA-OneVision. We implement the new model with the available compute, without extensively de-risking individual components. This leaves room for further improvements in capabilities through additional data and model scaling following our recipe, Please see the detailed development timeline in Section A. In particular, our paper makes the following contributions:
• Large multimodal models. We develop LLaVA-OneVision, a family of open large multimodal models (LMMs) that improves the performance boundaries of open LMMs in three important vision settings, including single-image, multi-image, and video scenarios.
• Emerging Capabilities with Task Transfer. Our design in modeling and data representations allow task transfer across different scenarios, suggesting a simple approach to yield new emgerging capabilities. In particular, LLaVA-OneVision demonstrate strong video understanding through task transfer from images.
• Open-source. To pave the way towards building a general-purpose visual assistant, we release the following assets to the public: the generated multimodal instruction data, the codebase, the model checkpoints, and a visual chat demo.
this section cite: ['b66', 'b82', 'b82', 'b80', 'b81', 'b81', 'b167', 'b64', 'b63', 'b67', 'b167', 'b64', 'b63', 'b67']

Section: Related Work
The SoTA proprietary LMMs, such as GPT-4V [109], GPT-4o [110], Gemini [131] and Claude-3.5 [3], exhibit excellent performance in versertile vision scenarios, including single-image, multi-image and video settings. In the open research community, existing works typically develop models tailored to each individual scenario separately. Specifically, most focus on pushing the performance limits in single-image scenarios [26,83,173,73,164,35], only a few recent papers have begun to explore multi-image scenarios [70,47]. While video LMMs excel in video understanding, they often do so at the expense of image performance [72,76]. It is rare to have a single open model that reports excellent performance in all three scenarios. LLaVA-OneVision aims to fill this gap by demonstrating state-of-the-art performance across a broad range of tasks, and showcasing interesting emerging capabilities through cross-scenario task transfer and composition.
To the best of our knowledge, LLaVA-NeXT-Interleave [68] is the first attempt to report good performance in all three scenarios, LLaVA-OneVision inherits its training recipe and data for improved performance. Other versatial open LMMs with potentials to excel include VILA [77], InternLM-XComposer-2.5 [162]. Unfortunately, their results are not fully evaluated and reported; we compare with them in the experiments. In addition to building systems with versatial capabilities, LLaVA-OneVision is benefited from large-scale high-quality data training, including model-synthesized knowledge and the new collection of diverse instruction tuning data. For the former, we inherit all the knowledge learning data in [64]. For the latter, our are motivated by FLAN [136,88,145]. The data collection process is con-current with Idefics2 [63] and Cambrian-1 [133], but we focus on a smaller but more carefully curated collection of datasets. A similar conclusion is observed: a large amount of visual instruction tuning data can significantly improve performance. For comprehensive investigations on design choices of LMMs, we refer to several recent studies [51,63,64,104,133,10]. x N 2 N 0 o J + R 9 e P C j i 1 f / i z X / j t s 1 B W x 8 M P N 6 b Y W Z e m H C m j e t + O 6 W V 1 b X 1 j f J m Z W t 7 Z 3 e v u n / Q 1 n G q C G 2 R m M e q G 2 J N O Z O 0 Z Z j h t J s o i k X I a S c c 3 0 z 9 z i N V m s X y 3 k w S G g g 8 l C x i B B s r P U T 9 z A + j D P n J i O V 5 v 1 p z 6 + 4 M a J l 4 B a l B g W a / + u U P Y p I K K g 3 h W O u e 5 y Y m y L A y j H C a V / x U 0 w S T M R 7 S n q U S C 6 q D b H Z 1 j k 6 s M k B R r G x J g 2 b q 7 4 k M C 6 0 n I r S d A p u R X v S m 4 n 9 e L z X R V Z A x m a S G S j J
J m V U G J I y 1 L Y V k r v 6 e y G h k z C Q K b G d E c W S W v Z n 4 n 9 d N M b z 2 M 6 G S F L l i i 0 V h K g n G Z P Y 8 G Q j N G c q J J Z R p Y W 8 l b E Q 1 Z W g j K t k Q v O W X V 0 n r o u r V q
n I 0 + u W v 3 i D m a Q Q K u W T G d D 0 3 Q T 9 j G g W X M C 3 1 U g M J 4 2 M 2 h K 6 l i k V g / G x + 8 Z S e W W V A w 1 j b U k j n 6 u + J j E X G T K L A d k Y M
this section cite: ['b108', 'b129', 'b2', 'b25', 'b82', 'b171', 'b72', 'b162', 'b34', 'b69', 'b46', 'b71', 'b75', 'b67', 'b76', 'b160', 'b63', 'b134', 'b87', 'b143', 'b62', 'b131', 'b50', 'b62', 'b63', 'b103', 'b131', 'b9']

Section: Modeling

this section cite: []

Section: Network Architecture
The model architecture inherits the minimalism design of LLaVA series, whose primary goals are (i) effectively leverage the pre-trained capabilities of both the LLM and visual model, as well as (ii) facilitate strong scaling behavior in terms of both data and model. The network archtecture is illustrated in Figure 1.
• LLM. We choose Qwen-2 [148] as our LLM f ϕ (•) parameterized by ϕ, as it offers various model size and exhibits strong language capabilities to date among publicly available checkpoints.
• Vision Encoder. We consider the SigLIP [158] as the visual encoder g ψ (•) parameterized by ψ, encoding an input image X v into its visual feature Z v = g(X v ). The grid features before and after the last Transformer layer are considered in our experiments.
• Projector. We consider a 2-layer MLP [81] p θ (•) parameterized by θ, to project image features into the word embedding space, yielding a sequence of visual tokens
H v = p(Z v ).
The model choice is based on our empirical insights in [65,64] that stronger LLM typically supercharge stronger multimodal capabilities in the wild, while SigLIP yields higher LMM performance among open vision encoders.
For a sequence of length L, we compute the probability of the target answers X a by:
p(X a |X v , X q ) = L i=1 p(x i |X v , X q,<i , X a,<i ),(1)
where X q,<i and X a,<i are the instruction and answer tokens in all turns before the current prediction token x i , respectively. For the conditionals in (1), we explicitly add X v to emphasize the fact that the visual signal is grounded for all answers. As explained in Section 3.2, the form of visual signal X v is general. The visual input fed into the vision encoder depends on the corresponding scenarios: the invidiual image crop in the single-image sequence, the invidiual image in a multi-image sequence and the invidiual frame in the video sequence, respectively.
this section cite: ['b146', 'b156', 'b80', 'b64', 'b63']

Section: Visual Representations
The representation of visual signals is key to the success of the visual encoding. It relates to two factors, the resolution in the raw pixel space and the number of tokens in the feature space, leading to the visual input representation configuration (resolution, #token). The scaling of both factors leads to improved performance, especially on tasks that require visual details. To strike a balance of performance and cost, we observe that the scaling of resolution is more effective than that of token numbers, and recommend an AnyRes strategy with pooling. The comparison is illustrated in Figure 2.   Single-Image Multi-Image Video 729 + N * 729 Tokens N * 729 Tokens … N Images … N Crops … N Frames N * 196 Tokens Max Tokens 32 * 196 = 6272 Tokens 12 * 729 = 8748 Tokens (1 + 9) * 729 = 7290 Tokens Example on Token Strategy The maximum number of visual tokens across different scenarios is designed to be similar, ensuring balanced visual representations to accommodate cross-scenario capability transfer. Note that 729 is the #tokens for SigLIP to encode a visual input of resolustion 384×384.
For AnyRes with a configuration of width a, height b, it divides the image into a × b crops, each with the shape (a, b). Each crop has the same resolution suitable for the vision encoder. Assuming there are T tokens per crop, the total number of visual tokens is L = (a × b + 1) × T , where the base image is resized before being fed into the vision encoder. We consider a threshold τ , and reduce the #token per crop, using bilinear interpolation if needed:
T new = τ (a×b+1) if L > τ T if L ≤ τ(2)
A set of spatial configurations (a, b) is defined to specify various methods for cropping images, thereby accommodating images of different resolutions and aspect ratios. Among them, the configuration that requires a minimum number of crops is selected. Please see our detailed ablations of visual representation in [64].
The proposed Higher AnyRes strategy can serve as a flexible visual representation framework, adaptable for multi-image and video representation. The optimal configuration for performance and cost can be adjusted accordingly. We illustratie the configuration in Figure 3, describe the detailed in Section C.1 and provide high-level encoding strategies as below:
• Single-image. We consider a large maximum spatial configuration (a, b) for single-image representation to maintain the original image resolution without resizing. Additionally, we purposefully allocate a large number of visual tokens per image, resulting in a long sequence to effectively represent the visual signal. This is based on the observation that there is a larger number of high-quality training samples with diverse instructions for images compared to videos. By representing an image with a long sequence that mimics video representation, we facilitate a smoother capability transfer from image to video understanding [169,64].
• Multi-image. Only the base image resolution is considered and fed into the vision encoder to obtain feature maps, eliminating the need for multi-crop of high resolution image and thus saving computational resources [68].
• Video. Each frame of the video is resized to the base image resolution and processed by the vision encoder to generate feature maps. Bilinear interpolation is employed to reduce the number of tokens, allowing the consideration of a larger number of frames by reducing tokens per frame. Empirical evidence suggests this provides a better trade-off between performance and computational cost [169].
These representation configurations are designed for capability transfer with a fixed compute budget in our experiments. With increased computational resources, the number of tokens per image or frame can be increased during both training and inference stages to boost performance.
this section cite: ['b63', 'b167', 'b63', 'b67', 'b167']

Section: Data
In the realm of multimodal training from LLM, the axiom "quality over quantity" is especially true. This principle is paramount due to the extensive knowledge stored within pre-trained LLMs and Vision Transformers (ViTs). While it is essential to accumulate balanced, diverse, and highquality instruction data by the end of the LMM's training lifecycle, an often-overlooked aspect is the continuous exposure of the model to new, high-quality data for further knowledge acquisition whenever it is available. In this section, we discuss the data sources and strategies for high-quality knowledge learning and visual instruction tuning.
this section cite: []

Section: High-Quality Knowledge
The web-scale public image-text data is often of low-quality, rendering the data scaling of multimodal pre-training less efficient. Instead, we recommend to focus on high-quality knowledge learning, given a limited compute budget. This approach acknowledges that the pre-trained LLMs and ViTs already possess a substantial knowledge base, and the goal is to refine and enhance this knowledge with carefully curated data. By prioritizing the quality of data, we can maximize compute efficiency.
We consider data from three major categories for high-quality knowledge learning:
• Re-Captioned Detailed Description Data. LLaVA-NeXT-34B [82] is known for its strong detailed caption ability among open-source LMMs. We used the model to generate new captions for the images from the following datasets: COCO118K, BLIP558K, and CC3M. We combined them to form the Re-Captioned Detailed Description Data, totaling 3.5M samples. This can be viewed as an simple attempt of self-improvement AI, where the training data is generated by an early version of the model itself.
• Document / OCR Data. We utilized the Text Reading subset from the UReader dataset, totaling 100K, which is easily accessible through PDF rendering. We used this text reading data along with the SynDOG EN/CN, to form the Document / OCR Data, totaling 1.1M samples.
• Chinese and Language Data. We used the original ShareGPT4V [20] images and utilized GPT-4V provided by the Azure API to generate 92K detailed Chinese caption data, aiming to improve the model's capability in Chinese. Since we used a large portion of detailed caption data, we also aim to balance the model's language understanding ability. We collected 143K samples from the Evo-Instruct dataset [16].
It is interesting to note that almost all (accounting for 99.8%) of the high-quality knowledge data is synthetic. This is due to the high cost and copyright constraints associated with collecting large-scale, high-quality data in the wild. In contrast, synthetic data can be easily scaled. We believe that learning from large-scale synthetic data is becoming a trend as AI models continue to grow more powerful.
this section cite: ['b81', 'b19', 'b15']

Section: Visual Instruction Tuning Data
Visual instruction tuning [83] refers to the capability of an LMM to understand and act upon visual instructions. These instructions can be in the form of language, combined with visual media such as images and videos, which the LMM processes and follows to perform a task or provide a response. This involves integrating visual understanding with natural language processing to interpret the instructions and execute the required responses.
this section cite: ['b82']

Section: Data Collection and Curation.
As demosntrated in previous works [81,133,63], visual instruction tuning data is crutial for LMM capaiblity. Therefore, maintaining a high-quality dataset collection is crucial and beneficial to the community. We started to collect a large pool of instruction tuning datasets from various original sources, with an unbalanced data ratio among categories. Additionally, we utilize a few new subsets from the Cauldron [63] and Cambrian [133] dataset collections.
We categorize the data based on a three-level hierachy: vision, instruction, and response.
• Vision Input. Three vision scenarios are considered, depding which visual input is considered in the multimodal sequence, including single-image, multi-image, video.
• Language Instruction. The instructions, which often appears as questions, define the tasks to perform to deal with the visual input. We classify the data into five major categories: General QA, General OCR, Doc/Chart/Screen, Math Reasoning, and Language. These instructions define the skill sets that a trained LMM could cover. We use task categorization to help maintain and balance the skill distribution.
• Language Response. The answer not only responds the user request, but also specifies the model behavior. It can be broadly categorized into free-form and fixed-form.
Free-form data is typically annotated by advanced models like GPT-4V/o and Gemini, while fixedform data is derived from academic datasets, e.g. VQAv2, GQA, Visual Genome. For free-form data, we keep the original answers. However, for fixed-form data, we manually review the content and make necessary corrections to the question and answer formats. We adhere to the LLaVA-1.5 prompting strategy for multiple-choice data, short answer data, and specific task data (e.g., OCR). This step is crucial for guiding the model's behavior to correctly balance QA performance, conversational ability, and reasoning skills in more complicated tasks, as well as preventing potential conflicts from different data sources. We list the full details about each dataset in our collection, and their categorization and formatting prompt in Appendix E.3.
We divide the instruction data into two separate groups: one for single-image scenario and the other for all vision scenarios. This division is based on insights from our earlier studies [68,169], which highlight the relationship between image and video models: a stronger image model can better transfer to multi-image and video tasks. Additionally, the quantity and quality of training datasets available for single images are significantly higher than those for videos and multi-image tasks.
Single-Image Data. Since single-image data is crucial for multimodal capabilities, we explicitly compile a large single-image data collection for model learning. We select from collected data sources to form a balanced collection, resulting in a total of 3.2 million samples. The overall distribution of single-image data is shown in Figure 4, with detailed information and the roadmap of data collection presented in Appendix E.1.
OneVision Data. In addition to the single-image stage training, we further fine-tune the model using a mixture of video, image, and multi-image data. We introduce a total of 1.6 million mixed data samples, comprising 560K multi-image data from OneVision 1.6M Single-Image (31.2%) Magpie Pro (90.0K) Vision FLAN (filtered) (55.8K) Image Textualization (49.8K) Cauldron (40.2K) UReader (39.9K) ShareGPT4V (21.0K) ALLaVA Inst. (21.0K) Cambrian (filtered GPT4o) (24.9K) LLAVA-Wild (train) (10.9K) LAION-GPT4V (8.0K) LLAVA-158K (7.0K) Geo170K-QA (6.8K) Geo170K-Align (6.0K) ShareGPT4o (5.7K) TabMWP (4.5K) LLAVAR GPT4 (4.0K) MapQA (4.3K) MathQA (3.0K) TextOCR (GPT4V) (2.5K) TextCaps (2.2K) ScienceQA (1.9K) FigureQA (1.8K) GeoQA+ (1.7K) AI2D (InternVL) (1.2K) UniGeo (1.2K) IconQA (1.1K) LRV-Normal (filtered) (1.1K) TQA (1.0K) Geometry3K (1.0K) Super-CLEVR (0.9K) AI2D (GPT4V) (0.7K) VizWiz (0.7K) VQA-AS (0.6K) CLEVR-Math (0.5K) PlotQA (0.5K) GEOS (0.5K) InfoVQA (0.9K) PMC-VQA (0.4K) Geo3K (0.2K) VQA-RAD (0.2K) LRV-Chart (0.2K) Multi-Image (43.0%) NLVR (86.4K) Co-Instruct (50.0K) ScanNet (49.9K) RAVEN (35.0K) IconQA (34.6K) VIST (26.0K) ScanQA (25.6K) ContrastiveCaption (25.2K) ALFRED (22.6K) FlintstonesSV (22.3K) ImageCode (16.6K) DreamSim (15.9K) Birds-to-Words (14.3K) PororoSV (12.3K) Spot-the-Diff (10.8K) nuScenes (9.8K) VISION (9.9K) WebQA (9.3K) RecipeQA-VisualCloze (8.7K) RecipeQA-ImageCoherence (8.7K) TQA (MI) (8.2K) AESOP (6.9K) HQ-Edit-Diff (7.0K) MagicBrush-Diff (6.7K) COMICS-Dialogue (5.9K) MultiVQA (5.0K) VizWiz (MI) (4.9K) CLEVR-Change (3.9K) NextQA (3.9K) IEdit (3.5K) Star (3.0K) DocVQA (MI) (1.9K) MIT-PropertyCoherence (1.9K) MIT-StateCoherence (1.9K) OCR-VQA (MI) (1.9K) Video (25.9%) ActivityNet (6.5K) Charades (23.6K) Ego4D (0.8K) NextQA (9.5K) ShareGPT4Video (255.0K) Youcook2 (41.9K) in [68]. The data distribution and details are presented in Figure 5, with additional information available in Appendix E.2.
this section cite: ['b80', 'b131', 'b62', 'b62', 'b131', 'b67', 'b167', 'b67']

Section: Training Strategies
To enable LLM for multimodal capabilities, we identify three critical functionalities, and systematically divide them into three distinct learning stages for the purpose of ablation studies. As with most existing research, prior LLaVA models mainly explore the single-image instruction tuning. However, other parts are less frequently investigated and therefore constitute the primary focus of this section.
We train the model via a curriculum learning principle, where training objectives and examples of increasing difficulty are observed in a stage-wise manner. With a fixed compute budget, this strategy helps decompose the training process and produces immediate checkpoints that can be re-used in more experiment trails.
• Stage-1: Language-Image Alignment. The goal is to well align the visual features into the word embedding space of LLMs.
• Stage-1.5: High-Quality Knowledge Learning. To strike a balance between compute-efficiency and injecting new knowledge into LMMs, we recommend to consider the high-quality knowledge for LMM learning. The training configuration mirrors the settings used in Stage-2, ensuring consistency and allowing the model to integrate new information seamlessly.
• Stage-2: Visual Instruction Tuning. To teach LMM to solve a diverse set of visual task with preferred responces, we organize the instruction data into different groups, described in Section 4.2. The model is scheduled to train on these groups in order.
Specifically, the visual instruction tuning process consists of two phases: (i) Single-Image Training:
The model is first trained on 3.2 million single-image instructions, resulting in a model with strong performance in following a diverse set of instructions to complete visual tasks using a single image.
(ii) OneVision Training: The model is then trained on a mixture of video, single-image, and multiimage data. In this phase, the model expands its capabilities from single-image scenarios to diverse scenarios. It learns to follow instructions to complete tasks in each new scenario and transfer the learned knowledge across different scenarios, resulting in new emergent capabilities. Note that the proposed OneVision training in the post-training stage is probably the simplest and most cost-efficient way to empower the LMMs with the multi-image and video understanding capabilities.
The training strategy is summarized in Table 1. We progressively train the model to deal with long sequence training. The maximum image resolution and the number of visual tokens gradually increase as training progresses. In Stage-1, the base image representation is considered with 729 tokens. In Stages 1.5 and 2, AnyRes is considered with up to 5 times and 10 times more visual tokens, respectively. Regarding trainable modules, Stage-1 updates only the projector, while the subsequent stages update the full model. It is also noted that the learning rate for the vision encoder is 5 times smaller than that for the LLM.
this section cite: []

Section: Language-Image Alignment High-Quality Knowledge Learning Visual Instruction Tuning
Stage
this section cite: []

Section: Experimental Results
We conduct standardized and reproducible evaluations for LLaVA-OneVision models on all benchmarks using LMMs-Eval [161]. For fair comparison with other leading LMMs, we primarily report results from original papers. When results are unavailable, we onboard the models in LMMs-Eval and evaluate them using consistent settings. All our results are reported with greedy decoding and 0-shot settings unless otherwise specified.
To reveal the generality and effectiveness of the designed paradigm, we comprehensively evaluate our LLaVA-OneVision models across different modalities in Table 2, including single-image, multiimage, and video benchmarks. Detailed results for each modality are presented in Table 3, Table 4, and Table 5, respectively. We denote the the model checkpoint trained after the single-image stage and one-vision stage as LLaVA-OV (SI) or LLaVA-OV, respectively Three model sizes are provided (0.5B, 7B and 72B), to accomodate applications with different performance-throughput trade-off, ranging from edge device to cloud serving. The GPT-4V and GPT-4o results are presented as references. Our largest model LLaVA-OneVision-72B yields superior performance between GPT-4V and GPT-4o on most benchmarks. It suggests that the proposed recipe is effecitve, revealing a promising path for further scaling. However, a relatively larger gap remains in complex tasks such as visual chat scenarios, we leave it as future research in stronger LLMs, larger training data and better preference learning.
this section cite: ['b159']

Section: Single-Image Benchmarks
To validate the performance for single-image tasks in real-world scenories, we consider a comprehensive set of image benchmarks in Table 3. It can be categorized into three classes:
(1) Chart, Diagram, and Document Understanding. As the main visual formats for structured OCR data, we evaluate the results on AI2D [54], ChartQA [101], DocVQA [103], and InfoVQA [102] benchmarks. Though current open-source models such as InternVL [22] and Cambrian [133] achieve performance comparable to commercial models, LLaVA-OneVision goes a step further, surpassing GPT-4V [109] and approaching the performance level of GPT-4o [110].
(2) Perception and Multi-discipline Reasoning. Including visual perception scenarios, we reveal the potentials of our model for more complex and challenging reasoning tasks. Specifically, we adopt the perception benchmarks including MME [151], MMBench [86], and MMVet [154], and reasoning benchmarks such as MathVerse [165], MathVista [90], and MMMU [157]. The results of LLaVA-OneVision significantly outperforms GPT-4V on various benchmarks, and comparable to GPT-4o on MathVista. This further confirms the superiority of our framework in visual perception and reasoning tasks.
(3) Real-world Understanding and Visual Chat. We consider the evaluation of LMMs as generalpurpose assistant in the wild as the most important metrics, beyond the lab environments. To validate the capabilities in real-world scenarios, we utilize several widely-adopted benchmarks, including RealworldQA [141], Vibe-Eval [111], MM-LiveBench [161], and LLaVA-Bench-Wilder [65]. While our model still has room for improvement compared to GPT-4V and GPT-4o, it achieves competitive performance with open-source models of similar parameter size. Notably, our model performs well on MM-LiveBench [161], a benchmark for real-world internet content with constantly updated content, demonstrating the model's broad world knowledge and strong generalization abilities.
this section cite: ['b53', 'b100', 'b102', 'b101', 'b21', 'b131', 'b108', 'b149', 'b85', 'b152', 'b163', 'b89', 'b155', 'b139', 'b109', 'b159', 'b64', 'b159']

Section: Multi-Image Benchmarks
We further evaluate LLaVA-OneVision in multi-image interleaved settings, where users may ask questions between multiples images. In particular, we perform comprehensive assessment on the diverse subtasks of LLaVA-Interleave Bench [68], such as Spot the Difference [45], Image Edit Instruction (IEI) [68], Visual Storytelling (VST) [40], Text-rich VQA (TR-VQA) [85], Multi-image VQA (MI-VQA) [117], Raven Puzzle [24], Q-Bench (QB) [139], and NLVR2 [125]). We also utilize several multi-view benchmarks for evaluation, which depict 3D environments with multiple viewpoints, including 3D Dialogue (3D-Chat) and Task Decomposition (3D-TD) from 3D-LLM [38], ScanQA [5], ALFRED [122], and nuScenes VQA [9]. We refer to these datasets as in-domain evaluations, since our training data includes the training split of them.
Moreover, we conduct evaluations on different out-domain tasks, which reveals the generalization capability of our approach. They include the multi-image split of math QA benchmark MathVerse [165] and science QA benchmark SciVerse [34], multi-image perception benchmark BLINK [31], MMMU-(multi-image) [157] that contains all multi-image QA in MMMU, and MuirBench [135] spanning 12 diverse multi-image tasks.
As shown in Table 4, LLaVA-OneVision (SI) consistently outperforms existing multi-image LMMs in all benchmarks. After additional tuning on multi-image and video data, LLaVA-OneVision shows a marked improvement over GPT-4V in specific areas, with significant margins. This highlights its strong performance in complex tasks such as multi-image reasoning, identifying differences, and understanding 3D environments. In addition, we observe a consistent performance enhancement on after the one-vision training stage, which is more evident on multi-view benchmarks that are absent Table 3: LLaVA-OneVision performance on single-image benchmarks. * GPT-4V reports 4-shot results on ChartQA. All results are reported as 0-shot accuracy.
in single-image data. This demonstrates the significance of our one-vision paradigm for empowering LMMs with comprehensive visual capbalities.
this section cite: ['b67', 'b44', 'b67', 'b39', 'b84', 'b115', 'b23', 'b137', 'b123', 'b37', 'b4', 'b120', 'b8', 'b163', 'b33', 'b30', 'b155', 'b133']

Section: Video Benchmarks
Video is also a common modality to build world model, capturing the dynamic nature of the real world over time. We conduct experiments on several open-ended and multi-choice video benchmarks. These include ActivityNet-QA [155] that contains human-annotated action-related QA pairs derived from ActivityNet dataset, EgoSchema [98] and MLVU [170] focusing on long video understanding, PerceptionTest [115] designed to evaluate the perception skills, VideoMME [29] and NeXTQA [142] containing diverse video domains and durations (from minutes to hours), VideoDetailCaption [87] and Video-ChatGPT [96] for video detailed description and visua chat, respectively.
As shown in
Table 5, LLaVA-OneVision achieves comparable or better results than previous open source models with much larger LLMs. The superiority of LLaVA-OneVision is particularly evident in complex benchmarks such as EgoSchema and VideoMME. Even compared to the advanced commercial model GPT-4V, LLaVA-OneVision performs competitively on the ActivityNet-QA, MLVU, and VideoMME benchmarks. Model IEI MI-VQA NLVR2 Puzzle Q-Bench Spot-Diff TR-VQA VST 3D-Chat 3D-TD ScanQA ALFRED nuScenes BLINK Mantis MathVerse MuirBench SciVerse in-domain multi-image in-domain multi-view out-domain GPT-4V [109] 11.0 52.0 88.8 17.1 76.5 12.5 54.5 10.9 31.2 35.4 32.6 10.3 63.7 51.1 62.7 60.3 62.3 66.9 LLaVA-N-Image-7B † [82] 13.2 39.4 68.0 9.0 51.0 12.9 59.6 10.1 -----41.8 46.1 13.5 -12.2 VPG-C-7B [70] 15.2 46.8 73.2 2.4 57.6 27.8 38.9 21.5 -----43.1 52.4 24.3 -23.1 Mantis-7B [47] 11.2 52.5 87.4 25.7 69.9 17.6 45.2 12.5 2.60 14.7 16.1 14.0 46.2 46.4 59.5 27.2 36.1 29.3 LLaVA-N-Inter-7B [68] 24.3 87.5 88.8 48.7 74.2 37.1 76.1 33.1 -----52.6 62.7 32.8 38.9 31.6 LLaVA-N-Inter-14B [68] 24.5 95 .0 91.1 59.9 76.7 40.5 78.6 33.3 70.6 52.2 34.5 62.0 76.7 52.1 66.4 33.4 40.7 32.7LLaVA-OV-0.5B (SI) 15.6 44.8 56.1 30.0 45.8 8.5 36.7 7.6 22.1 22.1 16.9 25.5 8.2 37.9 38.2 20.9 22.7 26.7 LLaVA-OV-0.5B 17.1 48.7 63.4 35.4 48.8 36.4 65.0 29.8 60.0 48.0 29.4 62.2 70.5 52.1 39.6 60.0 25.5 29.1 LLaVA-OV-7B (SI) 20.5 60.3 75.9 24.6 56.0 7.9 52.8 8.4 24.5 29.9 22.1 32.0 70.8 45.6 54.2 26.3 32.7 30.0 LLaVA-OV-7B 22.2 90.2 89.4 53.3 74.5 39.2 80.1 31.7 62.8 52.6 30.1 61.0 79.8 48.2 64.2 67.6 41.8 791 LLaVA-OV-72B (SI) 22.1 61.2 78.9 44.2 61.5 15.6 67.9 12.1 30.8 25.4 21.9 43.5 75.5 46.0 56.8 58.6 33.2 658 LLaVA-OV-72B 22.5 95.3 93.8 63.4 83.2 43.3 83.7 34.5 63.2 53.3 35.8 66.3 78.8 55.4 77.6 91.6 54.8 949 Table 5: LLaVA-OneVision performance on video benchmarks. We report the score out of 5 for VideoDC, VideoChatGPT while other results are reported in accuracy. All results are reported as 0-shot accuracy.
Within the LLaVA-OV split, the smallest performance difference occurs in PerceptionTest, with a minimal improvement of 0.5 points when scaling the LLM from 0.5B to 7B. This contrasts with at least a 5-point improvement in other datasets. The modest gain at PerceptionTest suggests that LLaVA-OV's perception capabilities may mainly depend on its vision module, supporting findings from recent studies such as those by Qiao et al. [116], which separate the roles of the image encoder and the LLM in perception and reasoning tasks. Notably, for datasets like EgoSchema that demand significant reasoning, a larger LLM substantially enhances performance.
Moreover, in comparing LLaVA-OV-7B (SI) with LLaVA-OV-7B, the smallest improvement is seen with ActivityNet-QA. This suggests that LLaVA-OV-7B (SI), which is trained only on images, can already perform well on this dataset. Delving into ActivityNet-QA, it becomes apparent that many questions can be answered by observing just a single frame from the video. For instance, the question "What's the color of the ball?" can be answered throughout the video as the ball is visible from start to finish. This scenario does not require the model to understand the video sequence, allowing LLaVA-OV-7B (SI) to perform well.
this section cite: ['b153', 'b97', 'b168', 'b113', 'b28', 'b140', 'b86', 'b95', 'b114']

Section: Emerging Capabilities with Task Transfer
In addition to reporting the LLaVA-OneVision's capabilities across various benchmarks, we also observe the emerging behaviors of the proposed model with task transfer and composition, paving a promising way to generalize to tackle real-world computer vision tasks in the wild. We illustrate several emerging capabilities using examples as below.
this section cite: []

Section: S1: Joint understanding of diagram and chart (Transfer from single-image to multi-image)
The capability to understand tables and charts are seperately learned from single image diagram and single-image chart understanding data, and the joint understanding task of table and chart do not appear in multi-image data. As shown in Table 6, LLaVA-OneVision is capable of understanding and reasoning over the joint of diagram and chart.
S2: GUI for multi-modal agent (Transfer from single-image and multi-image). Understanding GUIs and applying multimodal models to agentic tasks is of great value. In Table 7, LLaVA-OneVision recognizes the graphical user interface (GUI) screenshots of an iPhone and provides operational instructions to search for and open the TikTok app. This task requires strong OCR capabilities learned from single-image scenarios and relational reasoning skills developed from multiimage scenarios. The example highlights LLaVA-OneVision's proficiency in GUI understanding and task execution.
this section cite: []

Section: S3: Set-of-mark Prompting (Transfer from single-image task composition).
Different from existing open LLMs, LLaVA-OneVision demonstrates excellent set-of-marks (SoM) reasoning [149], an emerging capability shown in Table 8. To the best of our knowledge, this is the first time that open LMMs report good emerged SoM ability, as we observe that LLaVA-OneVision is able to produce SoM reasoning for many examples in [149]. This task is not explicitly included in our training data, it is hypothsized that the ability is composed by visual referring and OCR.
this section cite: ['b147', 'b147']

Section: S4: Image-to-Video Editing Instruction (Transfer from single-image and video).
LLaVA-OneVision could generate detailed video creation prompts based on a static image in Table 9. Given an image and a target video, the model constructs a coherent and vivid narrative for the video, detailing elements such as characters, actions, background settings, and scene specifics. This task leverages both single-image analysis and video comprehension. It is hypothesized that this ability is generalized from the composition of single-image editing instruction task and video detailed description task.
S5: Video-to-Video Difference (Transfer from multi-image and video). Understanding differences in images is a common ability in recent large multimodal models (LMMs), but our models extend this capability to videos. Table 10 showcases LLaVA-OneVision's ability to analyze differences between two video sequences with the same beginning frame but different endings. The model provides a detailed comparison, describing characters, actions, and scene changes. In Table 11, LLaVA-OneVision's describe the differences one by one between videos with a similar background but different main object in the foreground. This task leverages spot the difference in the multi-image analysis to generalize to video scenarios.
S6: Multi-camera Video Understanding in Self-driving (Transfer from single-image and multiimage to video). Understanding videos in a normal aspect ratio is straightforward, what about the videos with multi-views? In Table 12, we observe that LLaVA-OneVision could analyze and interprets multi-camera video footage from self-driving cars. Given video showing four camera views, the model describes each view in detail and plans the ego car's next move. This task combines multi-panel comprehension, video detailed description, and spatial-temporal reasoning.
S7: Composed Sub-video Understanding (Transfer from multi-image to video). Besides multiview video, we see our model generalize to vertical videos with two sub-scenes. Table 13 demonstrates LLaVA-OneVision's ability to understand and describe the content and layout of a composed subvideo. Given a vertical video with a series of frames featuring a consistent background and a person in the foreground, the model provides a detailed analysis of visual elements, their arrangement, and the narrative context. This task requires single-image analysis, multi-image sequence comprehension, and contextual reasoning.
S8: Visual prompting in video (Task transfer from single-image to video). In Table 14, LLaVA-OneVision is able to understand the highlighed area with a semi-transparent circle in the video, and clearly see the number "10" on the back of the player. The capability of understanding visual prompts and OCR is a capablity of single-image LMMs. Our model displays the capablity of understanding visual prompts in videos, without training on video data with visual prompts.
this section cite: []

Section: S9: Visual Referring in Image in Video Understanding.
The ability to refer to image query when answering questions about a video as shown in Table 15. This capbility is not seen in LLaVA-NeXT or LLaVA-Interleave, this is proabably because strong base single-image training is required for such capabilty to appear.
this section cite: []

Section: Conclusions
LLaVA-OneVision is a new, open LMM that shines when transferred to a broad range of tasks in the scenarios of single-image, multi-image and videos. The model is developed by consolidating the insights in the LLaVA-NeXT blog series, and is trained by scaling the recipe with a larger dataset and stronger LLMs. Our design allows new capabilities to emerge, through training multiple scenarios together and task transfer, eg, strong visual understanding ability from image to video. Our results demonstrate that LMMs trained with this open recipe and resources achieve state-of-the-art performance across various benchmarks. We also hope that LLaVA-OneVision serves as a valuable starting point for the community to build specific applications, and develop stronger LMMs for diverse vision scenarios through further scaling. Table 6: LLaVA-OneVision transfers its ability to understand diagram and table to multi-image scenarios, interpreting multiple images in a coherent manner.
this section cite: []

Section: References
Ref_id:b0 Title: Tallyqa: Answering complex counting questions Year: (2019)
Ref_id:b1 Title: Towards interpretable math word problem solving with operation-based formalisms Year: (2019)
Ref_id:b2 Title:  Year: (2024-05-03)
Ref_id:b3 Title: Vqa: Visual question answering Year: (2015)
Ref_id:b4 Title: Scanqa: 3d question answering for spatial scene understanding Year: (2022)
Ref_id:b5 Title: Scanqa: 3d question answering for spatial scene understanding Year: (2022)
Ref_id:b6 Title: Vision datasets: A benchmark for vision-based industrial inspection Year: (2023)
Ref_id:b7 Title: Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond Year: (2023)
Ref_id:b8 Title: Visual question answering on image sets Year: (2020)
Ref_id:b9 Title: A versatile 3b vlm for transfer Year: (2024)
Ref_id:b10 Title: Scene text visual question answering Year: (2019)
Ref_id:b11 Title: nuscenes: A multimodal dataset for autonomous driving Year: (2020)
Ref_id:b12 Title: Textocr-gpt4v Year: ()
Ref_id:b13 Title: Mapqa: A dataset for question answering on choropleth maps Year: (2022)
Ref_id:b14 Title: Webqa: Multihop and multimodal qa Year: (2021)
Ref_id:b15 Title: Allava: Harnessing gpt4vsynthesized data for a lite vision-language model Year: (2024)
Ref_id:b16 Title: Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression Year: (2022)
Ref_id:b17 Title: Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning Year: (2022)
Ref_id:b18 Title: Are we on the right way for evaluating large visionlanguage models? arXiv preprint Year: (2024)
Ref_id:b19 Title: Sharegpt4v: Improving large multi-modal models with better captions Year: (2023)
Ref_id:b20 Title: Sharegpt4video: Improving video understanding and generation with better captions Year: (2024)
Ref_id:b21 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2023)
Ref_id:b22 Title: Hitab: A hierarchical table dataset for question answering and natural language generation Year: (2022)
Ref_id:b23 Title: Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns Year: (2024)
Ref_id:b24 Title: Scannet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017)
Ref_id:b25 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning Year: (2024)
Ref_id:b26 Title: Neural naturalist: Generating fine-grained image comparisons Year: (2019)
Ref_id:b27 Title: Mme: A comprehensive evaluation benchmark for multimodal large language models Year: (2024)
Ref_id:b28 Title: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2024)
Ref_id:b29 Title: Dreamsim: Learning new dimensions of human visual similarity using synthetic data Year: (2023)
Ref_id:b30 Title: Blink: Multimodal large language models can see but not perceive Year: (2024)
Ref_id:b31 Title: G-llava: Solving geometric problem with multi-modal large language model Year: (2023)
Ref_id:b32 Title:  Year: (2022)
Ref_id:b33 Title:  Year: (2024)
Ref_id:b34 Title: Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following Year: (2023)
Ref_id:b35 Title: Imagine this! scripts to compositions to videos Year: (2018)
Ref_id:b36 Title: Vizwiz grand challenge: Answering visual questions from blind people Year: (2018)
Ref_id:b37 Title: 3d-llm: Injecting the 3d world into large language models Year: (2023)
Ref_id:b38 Title: Image change captioning by learning from an auxiliary task Year: (2021)
Ref_id:b39 Title: Visual storytelling Year: (2016)
Ref_id:b40 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b41 Title: Hq-edit: A high-quality dataset for instruction-based image editing Year: (2024)
Ref_id:b42 Title: Discovering states and transformations in image collections Year: (2015)
Ref_id:b43 Title: The amazing mysteries of the gutter: Drawing inferences between panels in comic book narratives Year: (2017)
Ref_id:b44 Title: Learning to describe differences between pairs of similar images Year: (2018)
Ref_id:b45 Title: Learning to describe differences between pairs of similar images Year: (2018)
Ref_id:b46 Title: Interleaved multi-image instruction tuning Year: (2024)
Ref_id:b47 Title: Clevr: A diagnostic dataset for compositional language and elementary visual reasoning Year: (2017)
Ref_id:b48 Title: Dvqa: Understanding data visualizations via question answering Year: (2018)
Ref_id:b49 Title: Figureqa: An annotated figure dataset for visual reasoning Year: (2018)
Ref_id:b50 Title: Prismatic vlms: Investigating the design space of visually-conditioned language models Year: (2024)
Ref_id:b51 Title: Geomverse: A systematic evaluation of large models for geometric reasoning Year: (2023)
Ref_id:b52 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b53 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b54 Title: Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension Year: (2017)
Ref_id:b55 Title: Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension Year: (2017)
Ref_id:b56 Title: The hateful memes challenge: Detecting hate speech in multimodal memes Year: (2020)
Ref_id:b57 Title: Ocr-free document understanding transformer Year: (2022)
Ref_id:b58 Title: Visual genome: Connecting language and vision using crowdsourced dense image annotations Year: (2016)
Ref_id:b59 Title: Image retrieval from contextual descriptions Year: (2022-05)
Ref_id:b60 Title: Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o Year: (2023)
Ref_id:b61 Title: A dataset of clinically generated visual questions and answers about radiology images Year: (2018)
Ref_id:b62 Title: What matters when building vision-language models? Year: (2006)
Ref_id:b63 Title: Llava-next: What else influences visual instruction tuning beyond data? Year: (2005)
Ref_id:b64 Title: Llava-next: Stronger llms supercharge multimodal capabilities in the wild Year: (2024-05-01)
Ref_id:b65 Title: Seed-bench: Benchmarking multimodal llms with generative comprehension Year: (2023)
Ref_id:b66 Title: Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision Year: (2024)
Ref_id:b67 Title: Llava-next: Tackling multi-image, video, and 3d in large multimodal models Year: (2024-06-01)
Ref_id:b68 Title: Fine-tuning multimodal llms to follow zero-shot demonstrative instructions Year: (2024)
Ref_id:b69 Title: Siliang Tang, and Yueting Zhuang. Empowering vision-language models to follow interleaved vision-language instructions Year: (2023)
Ref_id:b70 Title: Mvbench: A comprehensive multi-modal video understanding benchmark Year: (2023)
Ref_id:b71 Title: Llama-vid: An image is worth 2 tokens in large language models Year: (2024)
Ref_id:b72 Title: Mini-gemini: Mining the potential of multi-modality vision language models Year: (2024)
Ref_id:b73 Title: Storygan: A sequential conditional gan for story visualization Year: (2019)
Ref_id:b74 Title: Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning Year: (2023)
Ref_id:b75 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b76 Title: On pre-training for visual language models Year: (2024)
Ref_id:b77 Title: Microsoft coco: Common objects in context Year: (2015)
Ref_id:b78 Title: Visual spatial reasoning Year: (2023)
Ref_id:b79 Title: Aligning large multi-modal model with robust instruction tuning Year: (2023)
Ref_id:b80 Title: Improved baselines with visual instruction tuning Year: (2006)
Ref_id:b81 Title: Llava-next: Improved reasoning, ocr, and world knowledge Year: (2024-01-01)
Ref_id:b82 Title: Visual instruction tuning Year: (2006)
Ref_id:b83 Title: What large language models bring to text-rich vqa? Year: (2023)
Ref_id:b84 Title: What large language models bring to text-rich vqa? arXiv preprint Year: (2023)
Ref_id:b85 Title: Mmbench: Is your multi-modal model an all-around player? Year: (2023)
Ref_id:b86 Title: Video detail caption Year: (2024)
Ref_id:b87 Title: The flan collection: Designing data and methods for effective instruction tuning Year: ()
Ref_id:b88 Title: Deepseek-vl: towards real-world vision-language understanding Year: (2024)
Ref_id:b89 Title: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models Year: (2023)
Ref_id:b90 Title: Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning Year: (2021)
Ref_id:b91 Title: Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning Year: (2021)
Ref_id:b92 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b93 Title: Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning Year: (2023)
Ref_id:b94 Title: Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning Year: (2021)
Ref_id:b95 Title: Video-chatgpt: Towards detailed video understanding via large vision and language models Year: (2023)
Ref_id:b96 Title: Video-chatgpt: Towards detailed video understanding via large vision and language models Year: (2024)
Ref_id:b97 Title: Egoschema: A diagnostic benchmark for very long-form video language understanding Year: (2024)
Ref_id:b98 Title: Ok-vqa: A visual question answering benchmark requiring external knowledge Year: (2019)
Ref_id:b99 Title: The iam-database: an english sentence database for offline handwriting recognition Year: (2002)
Ref_id:b100 Title: Chartqa: A benchmark for question answering about charts with visual and logical reasoning Year: (2022)
Ref_id:b101 Title: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Year: (2022)
Ref_id:b102 Title: Docvqa: A dataset for vqa on document images Year: ()
Ref_id:b103 Title: Mm1: Methods, analysis & insights from multimodal llm pre-training Year: (2024)
Ref_id:b104 Title: Scene text recognition using higher order language priors Year: (2012)
Ref_id:b105 Title: Ocr-vqa: Visual question answering by reading text in images Year: (2019)
Ref_id:b106 Title: Ocr-vqa: Visual question answering by reading text in images Year: (2019)
Ref_id:b107 Title: Chart-to-text: Generating natural language descriptions for charts by adapting the transformer model Year: (2020)
Ref_id:b108 Title: Hello gpt Year: (2009)
Ref_id:b109 Title: Vibe-eval: A hard evaluation suite for measuring progress of multimodal language models Year: (2024)
Ref_id:b110 Title: Vibe-eval: A hard evaluation suite for measuring progress of multimodal language models Year: (2024)
Ref_id:b111 Title: Robust change captioning Year: (2019)
Ref_id:b112 Title: Image textualization: An automatic framework for creating accurate and detailed image descriptions Year: (2024)
Ref_id:b113 Title: Perception test: A diagnostic benchmark for multimodal video models Year: (2023)
Ref_id:b114 Title: Prism: A framework for decoupling and assessing the capabilities of vlms Year: (2024)
Ref_id:b115 Title: Multi-image visual question answering Year: (2021)
Ref_id:b116 Title: Aesop: Abstract encoding of stories, objects, and pictures Year: (2021)
Ref_id:b117 Title: A-okvqa: A benchmark for visual question answering using world knowledge Year: (2022)
Ref_id:b118 Title: Solving geometry problems: Combining text and diagram interpretation Year: (2015)
Ref_id:b119 Title:  Year: (2023)
Ref_id:b120 Title: Alfred: A benchmark for interpreting grounded instructions for everyday tasks Year: (2020)
Ref_id:b121 Title: Textcaps: a dataset for image captioning with reading comprehension Year: (2020)
Ref_id:b122 Title: Hollywood in homes: Crowdsourcing data collection for activity understanding Year: (2016)
Ref_id:b123 Title: A corpus of natural language for visual reasoning Year: (2017)
Ref_id:b124 Title: A corpus for reasoning about natural language grounded in photographs Year: (2019)
Ref_id:b125 Title: Expressing visual relationships via language Year: (2019)
Ref_id:b126 Title: Visualmrc: Machine reading comprehension on document images Year: (2021)
Ref_id:b127 Title: Vistext: A benchmark for semantically rich chart captioning Year: (2023)
Ref_id:b128 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b129 Title: Gemini: a family of highly capable multimodal models Year: (2023)
Ref_id:b130 Title:  Year: (2016)
Ref_id:b131 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2009)
Ref_id:b132 Title: Screen2words: Automatic mobile ui summarization with multimodal learning Year: (2021)
Ref_id:b133 Title: A comprehensive benchmark for robust multi-image understanding Year: (2024)
Ref_id:b134 Title: Finetuned language models are zero-shot learners Year: (2021)
Ref_id:b135 Title: wendlerc/renderedtext Year: (2023)
Ref_id:b136 Title: Longvideobench: A benchmark for long-context interleaved video-language understanding Year: (2024)
Ref_id:b137 Title: Q-bench: A benchmark for general-purpose foundation models on low-level vision Year: (2023)
Ref_id:b138 Title: Towards open-ended visual quality comparison Year: (2024)
Ref_id:b139 Title: Grok-1.5 vision preview Year: ()
Ref_id:b140 Title: Next-qa: Next phase of questionanswering to explaining temporal actions Year: (2011)
Ref_id:b141 Title: Pllava: Parameter-free llava extension from images to videos for video dense captioning Year: (2024)
Ref_id:b142 Title: Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing Year: (2024)
Ref_id:b143 Title: Vision-flan: Scaling human-labeled tasks in visual instruction tuning Year: (2024)
Ref_id:b144 Title: Vision-flan: Scaling human-labeled tasks in visual instruction tuning Year: (2024)
Ref_id:b145 Title: Erkut Erdem, and Nazli Ikizler-Cinbis. Recipeqa: A challenge dataset for multimodal comprehension of cooking recipes Year: (2018)
Ref_id:b146 Title: Qwen2 technical report Year: (2024)
Ref_id:b147 Title: Setof-mark prompting unleashes extraordinary visual grounding in gpt-4v Year: (2023)
Ref_id:b148 Title: Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model Year: (2023)
Ref_id:b149 Title: A survey on multimodal large language models Year: (2023)
Ref_id:b150 Title: Modeling context in referring expressions Year: (2016)
Ref_id:b151 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b152 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b153 Title: Activitynet-qa: A dataset for understanding complex web videos via question answering Year: (2019)
Ref_id:b154 Title: Syntax-aware network for handwritten mathematical expression recognition Year: (2022)
Ref_id:b155 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b156 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b157 Title: Raven: A dataset for relational and analogical visual reasoning Year: (2019)
Ref_id:b158 Title: Magicbrush: A manually annotated dataset for instruction-guided image editing Year: (2024)
Ref_id:b159 Title: Lmms-eval: Reality check on the evaluation of large multimodal models Year: (2024)
Ref_id:b160 Title: Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output Year: (2024)
Ref_id:b161 Title: Long context transfer from language to vision Year: (2024)
Ref_id:b162 Title: Llama-adapter: Efficient fine-tuning of language models with zero-init attention Year: (2023)
Ref_id:b163 Title: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint Year: (2024)
Ref_id:b164 Title: Mathematical visual instruction tuning Year: (2024)
Ref_id:b165 Title: Direct preference optimization of video large multimodal models from language model reward Year: (2024)
Ref_id:b166 Title: Enhanced visual instruction tuning for text-rich image understanding Year: (2023)
Ref_id:b167 Title: Llava-next: A strong zero-shot video understanding model Year: (2024-04-01)
Ref_id:b168 Title: Mlvu: A comprehensive benchmark for multi-task long video understanding Year: (2024)
Ref_id:b169 Title: Mlvu: A comprehensive benchmark for multi-task long video understanding Year: (2024)
Ref_id:b170 Title: Towards automatic learning of procedures from web instructional videos Year: (2017)
Ref_id:b171 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b172 Title: Visual7w: Grounded question answering in images Year: (2016)
