Title: Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence
Abstract: You are a robot beginning at the stairs and facing into the room. You want to navigate to the nightstand closer to the window. You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'):1. Go forward until the foot of the bed. 2. [please fill in] 3. Go forward until the wall. 4. [please fill in] 5. Go forward until the nightstand. You have reached the final destination. A. Turn Left, Turn Left B. Turn Left, Turn Right C. Turn Back, Turn Left D. Turn Right, Turn Left <Thinking> The goal is to reach the nightstand closer to the window. From my current position to the left side of the bed, I need to find this specific nightstand. I've already gone past a lamp on the left, so I should be looking around the wall area before the bed. Given the window is on the right side of the bed, it seems logical that the nightstand is further along that side. ... ...To get there, I need to turn right. The wall will be on my right side. Once I'm along the wall, I need to turn left when I reached the window, which is where the nightstand is located. </Thinking> Current Position? Walk along wall --> window from window to nightstand Get to where? window..How? <answer> D </answer> Figure 1: We propose Spatial-MLLM, a method that significantly enhances the visual-based spatial intelligence of existing video MLLMs. As shown, Spatial-MLLM is capable of understanding and reasoning about the underlying scene from video input, achieving state-of-the-art performance across a wide range of tasks.

Section: Introduction
Multimodal Large Language Models (MLLMs) [1,2,3] have achieved significant progress in processing multimodal inputs to generate contextually aware and semantically coherent responses. While proprietary models such as Gemini [4] and GPT-4o [5] exhibit state-of-the-art performance, the open-source community continues to advance the field by improving these models' ability to interpret diverse content modalities, including images [6,7,8], videos [9,10,11,12,13,14], and audio [15,16,17]. Although these models excel at a wide range of 2D tasks, their capacity to perceive, understand, and reason about 3D scenes, i.e., 3D spatial intelligence, remains limited [18,19].
The requirement of spatial understanding and reasoning typically arises in two scenarios. In the first scenario, the model has access to additional 3D or 2.5D data (e.g., point clouds, camera parameters, or depth maps) alongside 2D visual inputs (e.g., images or videos). These supplementary modalities enhance the model's spatial awareness, enabling more accurate spatial reasoning. However, this setup limits the model's applicability in many real-world scenarios where only monocular video of the scene is available, which is the second scenario. The model's ability to perform spatial understanding and reasoning under such conditions is referred to as visual-based 3D spatial intelligence [18,20]. A major challenge in this setting is that each frame provides only a partial observation of the scene, and no global representation (e.g., the point clouds [21,22,23] or posed depth maps [24,25]) is available as input. This requires the model to infer the global spatial layout from incomplete cues and internally integrate these partial observations into a coherent and implicit global representation, which demands strong spatial awareness. However, most existing video MLLMs pretrain their visual encoders on image-text pairs-primarily image-caption data [13,14,26]-following the CLIP [27] paradigm. This makes the visual encoder excel at capturing high-level semantic content but lack structure and spatial information when only 2D video inputs are available [28,29,30]. Consequently, current video MLLMs generally perform worse on spatial reasoning tasks than on other tasks, such as temporal understanding. Moreover, their performance still significantly lags behind human capabilities [18].
In this paper, we introduce Spatial-MLLM, a method that significantly improves the visual-based 3D spatial intelligence of existing video MLLMs. To address the limitations of visual encoders in general-purpose video MLLMs, our key insight is to unleash the strong structure prior provided by the feed-forward visual geometry foundation model [31,32,33,34]. These models, typically trained on pixel-point pairs, complement the general-purpose video MLLM visual encoders that are trained primarily on image-text data [14]. Based on this insight, we design a dual-encoder architecture consisting of a 2D encoder-initialized from the visual encoder of a general-purpose video MLLM-to extract 2D semantic information, and a spatial encoder-leveraging the VGGT feature extractor [32]-to recover implicit 3D structural information from 2D video inputs. We then use a connector to integrate features from both branches into unified visual tokens. The resulting representation enables the Large Language Model (LLM) backbone to perform effective spatial reasoning without requiring explicit 3D data as input.
Furthermore, we fully exploit the additional information provided by the introduced feed-forward visual geometry model [32], and propose a space-aware frame sampling strategy at inference time, which selects the most spatially informative frames from the video sequence when the total number of input frames is limited (e.g., due to the VRAM limitation). Specifically, we first feed a relatively large number of frames into the spatial encoder and decode the resulting 3D features into voxels. The frame selection task is then reformulated as a maximum coverage problem over these voxels, which we solve using a greedy algorithm. To train Spatial-MLLM, we construct a visual-based 3D spatial question-answering dataset and perform supervised fine-tuning on it. We further apply a simple cold-start [35] to help the model adapt to the correct reasoning format, and then train it using Group Relative Policy Optimization (GRPO) [36,35] to enhance its long-chain-of-thought (long-CoT) spatial reasoning capability [37]. We conduct extensive evaluations on the VSI-Bench [18], 2 Related Work
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b17', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b12', 'b13', 'b25', 'b26', 'b27', 'b28', 'b29', 'b17', 'b30', 'b31', 'b32', 'b33', 'b13', 'b31', 'b31', 'b34', 'b35', 'b34', 'b36', 'b17']

Section: MLLMs for Video Understanding
Multimodal Large Language Models (MLLMs) have made significant progress in integrating vision and language. Early works such as BLIP-2 [2] and Flamingo [1] introduce token-level fusion (e.g., Q-Former) and feature-level fusion (e.g., cross-attention layers) to bridge modalities. Other approaches, including the LLaVA series [3,40], MiniGPT-4 [41], and subsequent models [13,42,43], leverage MLPs to project visual features into the language space. Recent advancements in MLLMs have extended their capabilities from images to videos, typically by introducing video-language alignment through large-scale pretraining [9,44]. Later models, such as Qwen2.5-VL [14], enhance temporal reasoning via dynamic resolution and absolute time encoding. Although existing video MLLMs excel at capturing high-level semantics and temporal patterns, they struggle to interpret the underlying 3D scene from video input, which inspires our work to enhance their spatial understanding capabilities.
this section cite: ['b1', 'b0', 'b2', 'b39', 'b40', 'b12', 'b41', 'b42', 'b8', 'b43', 'b13']

Section: 3D MLLMs for Scene Understanding
Recent advances in MLLMs have sparked interest in extending their capabilities from 2D to 3D scene understanding [23,24,25,45,46,47,22,48,49,50,51,52]. LL3DA [23] extracts scene-level features from point clouds using a Q-Former, while Grounded 3D-LLM [45] integrates 3D detectors to generate object proposals. Methods like Chat3D [46], LEO [47], and Chat-Scene [22] first segment 3D objects and encode object-centric features for fusion. Alternatively, 3D-LLM [48] and Scene-LLM [49] aggregate CLIP features from pre-segmented multi-view object patches into 3D point representations, leveraging multi-view images and camera parameters. LLaVA-3D [24] projects 2D multi-view patch features into voxel space for 3D-aware aggregation, and GPT4Scene [50] enhances 3D reasoning by first reconstructing scenes and then using BEV images as input. While these methods advance 3D scene understanding, most of them require additional 3D or 2.5D inputs, which is difficult to acquire in real-world scenarios. In contrast, our approach only requires 2D videos as input.
this section cite: ['b22', 'b23', 'b24', 'b44', 'b45', 'b46', 'b21', 'b47', 'b48', 'b49', 'b50', 'b51', 'b22', 'b44', 'b45', 'b46', 'b21', 'b47', 'b48', 'b23', 'b49']

Section: Visual-based 3D Spatial Intelligence
Visual-based 3D spatial intelligence aims to enable video MLLMs to perceive, infer, and reason about 3D structures and spatial relationships purely from 2D visual inputs. Despite recent advances, most existing video MLLMs are still primarily designed for 2D understanding tasks, and their extension to visual-based 3D reasoning, e.g., 3D question answering [6,53] and robotic manipulation [54], remains relatively underexplored. To address this limitation, a new wave of specialized benchmarks has emerged to systematically evaluate the spatial reasoning capabilities of these models [18,20,55,56,57,58]. Among them, VSI-Bench [18] serves as a pioneering benchmark that comprehensively assesses visual-spatial intelligence across multiple dimensions. STI-Bench [20] 𝑵𝑵𝒌𝒌 frames
this section cite: ['b5', 'b52', 'b53', 'b17', 'b19', 'b54', 'b55', 'b56', 'b57', 'b17', 'b19']

Section: Connector

this section cite: []

Section: 3D Features (Inference)
Selected for 𝑵𝑵𝒌𝒌 frames
this section cite: []

Section: Space-aware Frame Sample
𝑵𝑵𝒌𝒌 frames
this section cite: []

Section: 3D Features

this section cite: []

Section: 𝑵𝑵𝒎𝒎 frames

this section cite: []

Section: Tokenize

this section cite: []

Section: Answer

this section cite: []

Section: Large Language Model

this section cite: []

Section: Decode

this section cite: []

Section: 2D Encoder

this section cite: []

Section: 3D Spatial Encoder
A.
table "Measuring from the closest point of each object, which of these objects (table, sofa, stool, chair) is the closest to the dishwasher?" A. table B. sofa C. stool D. chair
this section cite: []

Section: Language Instructions

this section cite: []

Section: 2D Features

this section cite: []

Section: 𝑵𝑵𝒌𝒌 frames

this section cite: []

Section: 3D Features (Inference)
for 𝑵𝑵𝒎𝒎 frames
this section cite: []

Section: Index Selection

this section cite: []

Section: Spatial Video

this section cite: []

Section: Spatial Encoder

this section cite: []

Section: Input

this section cite: []

Section: Global Attention

this section cite: []

Section: Frame Attention

this section cite: []

Section: *L Times

this section cite: []

Section: 3D Features

this section cite: []

Section: Inference Training
Figure 2: Overview of Spatial-MLLM. Our model is composed of a 2D visual encoder E2D, a 3D spatial encoder ESpatial, which is initialized from a feed-forward visual geometry foundation model, a connector, and a large language model backbone. At inference time, we incorporate a space-aware frame sampling strategy to select spatially informative frames when the number of input frames is limited due to GPU memory constraints.
introduces physics-aware questions, such as velocity estimation, to quantify a model's spatial and kinematic reasoning abilities. Ego-ST Bench [55] evaluates the model's spatial intelligence from an egocentric perspective, while VLM4D [56] emphasizes motion dynamics, such as trajectory prediction, to probe 4D spatiotemporal interactions. Collectively, these benchmarks signify a shift toward a more holistic evaluation of visual-based spatial intelligence in video MLLMs.
this section cite: ['b54', 'b55']

Section: Method
In this section, we introduce Spatial-MLLM. Given a video of N frames depicting a scene, denoted as V = {f i } N i=1 , where f i ∈ R H×W ×3 , Spatial-MLLM is designed to understand spatial relationships, perform spatial reasoning, and generate appropriate responses. We begin by describing the model architecture in Section 3.1, which comprises a 2D visual encoder, a 3D spatial encoder, a connector, and a large language model backbone. Then we present the space-aware frame sampling strategy in Section 3.2, which selects N k spatially informative frames f k i N k i=1 , where N k ≪ N . Finally, we introduce the training dataset construction process and two-stage training pipeline in Section 3.3.
this section cite: []

Section: Spatial-MLLM Architecture
In this section, we present the architecture of Spatial-MLLM, which is shown in Figure 2. We adopt Qwen2.5-VL-3B [14] as our base model and explore strategies to enhance its spatial understanding and reasoning capability. Before diving into the details, we first briefly introduce the key insights that motivate our design.
What hinders visual-based spatial intelligence in existing video MLLMs? Existing video MLLMs [14,13,12] typically employ a pre-trained 2D visual encoder E 2D to extract 2D patch features e 2D . These features are then projected into visual tokens through a lightweight connection module. A large language model backbone f θ subsequently generates the final response by conditioning on both visual and textual tokens. A critical bottleneck in this process lies in the nature of the visual features extracted. The required type of information varies by task: high-level semantic representations are essential for 2D recognition and understanding, whereas fine-grained structural cues are crucial for spatial reasoning. However, the visual encoders used in current video MLLMs are primarily pre-trained on image-text datasets (mainly image-caption pairs) [14,26] following the CLIP [27] paradigm. As a result, these models predominantly capture semantic content and often lack spatial awareness when no additional 3D or 2.5D data are available [28,29,30]. To address this, our key insight is to unleash feed-forward visual geometry foundation models [32], which are trained on pixel-point pairs and can recover rich 3D structural information from 2D inputs, which complements the semantic features extracted by the 2D visual encoder. We design a dual-encoder architecture that exploits the strengths of both models and a connector to fuse semantic and structural information into unified visual tokens. Below, we introduce the core components of our design.
this section cite: ['b13', 'b13', 'b12', 'b11', 'b13', 'b25', 'b26', 'b27', 'b28', 'b29', 'b31']

Section: Dual-Encoder.
The proposed dual-encoder consists of a 2D encoder E 2D and a 3D spatial encoder E Spatial . For the 2D encoder branch, we adopt the same design as the visual encoder of Qwen2.5-VL [14] to encode input frames into semantically rich features:
e 2D = E 2D {f i } N k i=1 , e 2D ∈ R N k ′ × H p 2D × W p 2D ×d2D ,(1)
where p 2D and d 2D denote the patch size and feature dimension of the 2D visual encoder, respectively. The two consecutive frames are grouped for video input, thus N k ′ = ⌈N k /2⌉.
For the spatial encoder branch, we utilize the feature backbone of VGGT [32]. Specifically, given N k frames of the scene video, we first patchify the input and then extract 3D features with alternating frame-wise self-attention and global self-attention [59]. This process allows E spatial to aggregate spatial information across different frames to get the final 3D features:
e 3D , e c , e register = E spatial {f i } N k i=1 , e 3D ∈ R N k × H p 3D × W p 3D ×d3D ,(2)
where e 3D , e c , and e register represent the dense 3D feature, the camera feature for each frame, and the register tokens [60], respectively. We only use e 3D in the feature fusion stage as it captures the dense structure information of the input frames.
Connector. After obtaining the 2D and 3D features, we use a connector to integrate the semantic and structural information from both branches. Specifically, we first align e 3D with e 2D in both spatial and temporal dimensions:
e ′ 3D = Rearrange(e 3D ), e ′ 3D ∈ R N k ′ × H p 2D × W p 2D ×d ′ 3D .(3)
Here, the spatially and temporally adjacent information in e 3D is aggregated into the feature channel dimension, enabling alignment with e 2D . Next, we employ a lightweight connector to fuse the information to obtain the unified visual tokens:
e = Connector(e 2D , e ′ 3D ),(4)
where e ∈ R S×d llm denotes the final visual tokens and S = N k ′ × H p2D × W p2D is the sequence length. In practice, we adopt a MLP-based design (detailed in Section B.2). Although more complex feature fusion methods, e.g., cross-attention [59,61,51], could be applied, we find that this approach is effective to enhance the model's spatial understanding and reasoning capabilities. We leave the exploration of more advanced fusion strategies for future work.
this section cite: ['b13', 'b31', 'b58', 'b59', 'b58', 'b60', 'b50']

Section: Space-Aware Frame Sampling
Due to GPU memory constraints, video MLLMs can process only a limited subset of frames from a scene video sequence. For example, in the VSI-Bench setup [18], only 8 to 32 frames are sampled as input to the video MLLM, while a typical scene video in VSI-Bench contains over 2,000 frames. A widely adopted solution is uniform frame sampling [13,14,18], which is effective for generalpurpose video understanding. However, as spatial videos represent 3D scenes, the sampling strategy for spatial understanding tasks should focus on capturing most information of the underlying scene, which uniform sampling fails to achieve.
Benefiting from the feed-forward visual geometry foundation model, we design a straightforward space-aware frame sampling strategy at inference time. Specifically, given a scene video V = {f i } N i=1 , our objective is to select N k frames, {f k i } N k i=1 that have most coverage of the underlying scene. To achieve this, we first uniformly subsample N m frames, {f m i } Nm i=1 , where N m satisfies N k < N m < N , and is determined by the available GPU memory. In practice, we choose N m = 128 and N k = 16. We then leverage E 3D to extract their corresponding 3D features e m 3D and camera features e m c . Subsequently, we use the pretrained camera head f c and depth head f d of the VGGT model [32] to decode a set of camera parameters and depth maps:
{E m i , K m i } Nm i=1 = f c (e c ),and
{D m i } Nm i=1 = f d (e 3D ).(5)
This allows us to calculate the voxels V (f m i ) covered by each frame f m i , and formulate frame selection as a maximum coverage problem [62], i.e., select N k frames {f k i } N k i=1 ⊆ {f m i } Nm i=1 such that the total number of unique covered voxels
N k i=1 V (f k i ) is maximized.
In practice, we apply a greedy algorithm to accelerate computation [63,25]. Once the N k frames are selected, it is not necessary to recompute their 3D features e k 3D and the corresponding features from the precomputed set e m 3D can be directly reused. We provide the complete algorithm and detailed explanation in Section B.1.
this section cite: ['b17', 'b12', 'b13', 'b17', 'b31', 'b61', 'b62', 'b24']

Section: Training
Training Data Construction. We first construct a visual-based 3D spatial question-answering dataset. The dataset has approximately 120k QA pairs and is constructed from three sources: the training set of ScanQA [38], SQA3D [39], as well as additional self-created spatial QA data. All items in our training dataset are derived from scenes in the training set of ScanNet [64] and are each represented as a quadruple I i = ⟨Q i , A i , V i , M i ⟩, denoting the question, answer, video ID, and meta-information (e.g., task type), respectively. For the self-created QA data, we follow the data processing pipeline proposed in VSI-Bench [18]. Specifically, we first convert ScanNet scenes into continuous video clips at 24 FPS and 640 × 480 resolution. Then we generate spatial reasoning QA pairs leveraging the meta-annotations of Scannet. The generated QA pairs cover various spatial understanding and reasoning tasks, including object counting, object size, room size, absolute distance, appearance order, relative distance, and relative direction. Since the QA pair construction process is similar to that of VSI-Bench [18], we exclude the QA pair I i if its scene video V i is used in VSI-Bench (these videos are sourced from the validation set of Scannet) to prevent data leakage. Finally, the self-created data contains approximately 70k QA pairs in total. We provide additional details on training data construction in the section B.3. Figure 3 shows a brief summary of key statistics of the training dataset. Supervised Fine-tuning. Leveraging the constructed training dataset, we first perform supervised fine-tuning (SFT) on our model. Since E 2D and E spatial are pre-trained on large-scale image-text and pixel-point pairs, respectively, we freeze them to preserve their ability to extract rich semantic and structural information. We jointly train the connection module and the LLM backbone to enable the model to adaptively fuse 2D and 3D features and enhance its spatial understanding and reasoning capability. During this stage, we employ the standard cross-entropy loss L ce between the model-generated answers and the ground-truth annotations:
L ce (θ) = - i log P (o (i) | o (1:i-1) , q, {f i } N k i=1 ) (6
)
where
{f i } N k i=1
denotes input video frames, q denotes the system prompt and question, o (i) represents the i-th token in the ground-truth answer, and o (1:i-1) denotes the preceding answer tokens.
this section cite: ['b37', 'b38', 'b63', 'b17', 'b17']

Section: RL Training.
Following the SFT stage, we first perform a simple cold start [35] to help the model adapt to the correct reasoning format. Then we train the model using Group Relative Policy Optimization (GRPO) [36] to enhance its long-CoT [37] spatial reasoning capability. During training, we first sample a set of output {o 1 , o 2 , . . . , o G } for each question q from the policy model π θold . Then we optimize the policy model by maximizing the following objective:
J GRPO (θ) = Eq,o i 1 G G i=1 min π θ (oi | q) π θ old (oi | q) Ai, clip( π θ (oi | q) π θ old (oi | q) , 1 ± ϵ)Ai -β KL[π θ ∥π ref ](7)
where
A i = r1-mean(r1,r2,...,r G ) std(r1,r2,...,r G )
is the advantage function computed using the group rewards.
In GRPO, the design of the reward function is critical. In addition to a formatting reward applied to all task types, we introduce task-dependent reward modeling to ensure that it accurately reflects Table 1: Evaluation Results on VSI-Bench [18]. For Spatial-MLLM and Qwen2.5-VL series [14], we use 16 frames as input and report micro average scores. For other open-source methods and GPT-4o [5], the number of frames is the same as VSI-Bench setting (ranging from 8 to 32 frames). For Gemini-1.5 Pro [4], it samples video frames at 1 FPS. Bold and underline denote the best-performing and second-best-performing open-source models, respectively.
this section cite: ['b34', 'b35', 'b36', 'b17', 'b13', 'b4', 'b3']

Section: Methods
Numerical Question Multiple-Choice Question Avg. Rank Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order Proprietary Models GPT-4o [5] 46.2 5.3 43.8 38.2 37.0 41.3 31.5 28.5 34.0 7 Gemini-1.5 Pro [4] 56.2 30.9 64.1 43.6 51.3 46.3 36.0 34.6 45.4 2 Open-source Models InternVL2-40B [7] 34.9 26.9 46.5 31.8 42.1 32.2 34.0 39.6 36.0 6 LongVILA-8B [66] 29.1 9.1 16.7 0.0 29.6 30.7 32.5 25.5 21.6 12 VILA-1.5-40B [67] 22.4 24.8 48.7 22.7 40.5 25.7 31.5 32.9 31.2 9 LongVA-7B [68] 38.0 16.6 38.9 22.2 33.1 43.3 25.4 15.7 29.2 11 LLaVA-OneVision-72B [6] 43.5 23.9 57.6 37.5 42.5 39.9 32.5 44.6 40.2 4 LLaVA-Video-72B [12] 48.9 22.8 57.4 35.3 42.4 36.7 35.0 48.6 40.9 3 Spatial-MLLM and Qwen2.5-VL Series Qwen2.5-VL-3B [14] 24.3 24.7 31.7 22.6 38.3 41.6 26.3 21.2 30.6 10 Qwen2.5-VL-7B [14] 40.9 14.8 43.4 10.7 38.6 38.5 33.0 29.8 33.0 8 Qwen2.5-VL-72B [14] 25.1 29.3 57.9 29.4 41.7 37.0 23.2 29.0 37.0 5 Spatial-MLLM-4B 65.3 34.8 63.1 45.1 41.3 46.2 33.5 46.3 48.4 1
the proximity between the predicted and ground-truth answers. Specifically, we categorize the data into three types based on answer format: numeric answer questions, multiple-choice questions, and verbal answer questions. For numeric questions, we compute the mean relative accuracy [18]. For multiple-choice questions, we employ an exact match reward. For verbal answer questions, we use fuzzy matching based on Levenshtein distance. Further details on reward calculation are provided in Section B.5.
this section cite: ['b17']

Section: Experiments

this section cite: []

Section: Implementation Details
Training details. Spatial-MLLM is built on Qwen2.5-VL [14] and VGGT [32] and has approximately 4.9B parameters in total. We use the visual encoder of Qwen2.5-VL [14] to initialize E 2D , and the LLM backbone of it to initialize f θ . We then use the feature backbone of VGGT [32] to initialize E spatial . During training, we use 640 × 480 resolution and limit video frames to 16. In the SFT stage, we train the model using Adam optimizer [65] for one epoch. We set the global batch size to 16 and use a linear learning-rate schedule, with a peak value of 10 -5 . In the cold start stage, we first construct a small CoT dataset. Specifically, we prompt Qwen2.5-VL-72B [14] to generate multiple thinking processes and answers according to the scene video and question. Then we use the GT answer to filter a correct thinking-answer pair (more details are provided in Section B.4). We use a similar setting as in the SFT stage to train the model for 200 steps. In the RL stage, we perform 8 rollouts per question and set the default sampling temperature to 1. The KL divergence coefficient, β, is set to 0.04. Due to computational resource limitations, we train the model for 1,000 steps with a learning rate of 10 -6 . We show the training curve of SFT Stage and RL Stage in Figure 4.
Inference Details. During inference, we set N m = 128 and N k = 16 for space-aware frame sampling. Since spatial reasoning requires a certain level of determinism, we set the temperature to 0.1 and the top-p to 0.001. The default input resolution from the scene video is 640 × 480.
this section cite: ['b13', 'b31', 'b13', 'b31', 'b64', 'b13']

Section: Comparisons on VSI-Bench
Setup. VSI-Bench [18] contains more than 5,000 question-answer pairs derived from egocentric videos sourced from ScanNet [64], ScanNet++ [69], and ARKitScenes [70]. The task types are divided into Multiple-Choice Answer (MCA) and Numerical Answer (NA). For the MCA tasks, we compute mean accuracy, and for the NA tasks, we calculate relative accuracy across confidence thresholds C = {0.5, 0.55 . . . , 0.95}. We report the final average score and individual metrics on eight task types of VSI-Bench, including: (1) configurational reasoning tasks (object counting, relative direction, absolute direction, and route planning), (2) measurement estimation tasks (object size, room size, and absolute distance), and (3) spatiotemporal reasoning tasks (appearance order). For Spatial-MLLM Table 2: Evaluation Results on ScanQA [38] and SQA3D [39]. We use the val set of ScanQA and test set of SQA3D for evaluation following common practice [22,47,25]. Bold and underline denote the best-performing and second-best-performing models in each model category, respectively.
this section cite: ['b17', 'b63', 'b68', 'b69', 'b37', 'b38', 'b21', 'b46', 'b24']

Section: Methods
ScanQA (val) SQA3D (test) Video-Input Only BLEU-1 BLEU-4 METEOR ROUGE-L CIDEr EM-1 EM-R1 Task-Specific Models ScanQA [38] 30.2 10.1 13.1 33.3 64.9 47.2 -✗ SQA3D [39] 30.5 11.2 13.5 34.5 -46.6 -✗ 3D-Vista [71] --13.9 35.7 -48.5 -✗ 3D/2.5D-Input Models 3D-LLM [48] 39.3 12.0 14.5 35.7 69.4 --✗ LL3DA [23] -13.5 15.9 37.3 76.8 --✗ Chat-Scene [22] 43.2 14.3 18.0 41.6 87.7 54.6 57.5 ✗ 3D-LLaVA [21] -17.1 18.4 43.1 92.6 54.5 56.6 ✗ Video-3D LLM [25] 47.1 16.2 19.8 49.0 102.1 58.6 -✗ Video-Input Models Qwen2.5-VL-3B [14] 26.4 7.5 12.2 33.2 62.7 43.4 45.9 ✓ Qwen2.5-VL-7B [14] 26.2 9.6 12.7 34.2 64.9 46.5 49.8 ✓ Qwen2.5-VL-72B [14] 26.8 12.0 13.0 35.2 66.9 47.0 50.9 ✓ LLaVA-Video-7B [12] 39.7 3.1 17.7 44.6 88.7 48.5 -✓ Oryx-34B [53] 38.0 -15.0 37.3 72.3 --✓ Spatial-MLLM-4B 44.4 14.8 18.4 45.0 91.8 55.9 58.7 ✓
and Qwen2.5-VL series, we report micro average scores in Table 1 and macro average scores in Table 5.
Baseline Models. We compare our model with a broad range of video-input MLLMs. For proprietary model, we include GPT-4o [5] and Gemini-1.5 Pro [4] for comparison. For open-source video-input MLLMs, we compare our model with InternVL2 [7], LongVILA [66], VILA [67], LongVA [68], LLaVA-NeXT-Video [12], LLaVA-OneVision [6], and the Qwen2.5-VL [14] series. The parameter count of the baseline models is reported in Table 1.
Results. We present the quantitative results on VSI-Bench [18] in Table 1 and Table 5. Despite having 4.9B parameters, Spatial-MLLM significantly outperforms all proprietary and open-source MLLMs, including those with substantially larger parameter counts (e.g., 32B or 72B). Among the remaining models, the best-performing one is the proprietary Gemini-1.5 Pro [4]. Notably, Spatial-MLLM is provided with only 16 input frames per video, while Gemini-1.5 Pro [4] samples videos at 1 FPS (i.e., an average of 85 frames per video on VSI-Bench) according to its API instructions [18]. Despite the significantly lower number of input frames, Spatial-MLLM still achieves higher average accuracy than Gemini-1.5 Pro [4].
this section cite: ['b4', 'b3', 'b6', 'b65', 'b66', 'b67', 'b11', 'b5', 'b13', 'b17', 'b3', 'b17', 'b3']

Section: Comparison on ScanQA and SQA3D
Setup. ScanQA [38] and SQA3D [39] are two 3D question-answering benchmarks built upon ScanNet [64]. Since the authors did not provide a test set for ScanQA, we evaluate it using the validation set, which consists of 4,675 QA pairs focused on understanding spatial relationships such as object alignment and orientation, as well as the ability to accurately identify objects in 3D scenes based on textual questions. We follow standard practice [25,50] by evaluating answer quality using the following metrics: CiDEr, BLEU-1, BLEU-4, METEOR, and ROUGE-L. For SQA3D, we evaluate the model on its test set, which contains 3,519 QA pairs. The task requires the model to first understand its position and orientation within the 3D scene, as described by text, then reason about its environment and answer a question under those conditions. Since SQA3D contains definitive answers, we use exact match accuracy (EM) and its refined version (EM-R) as evaluation metrics. We provide the evaluation results using additional metrics for both benchmarks in Section C.2.
Baselines. Since both the ScanQA [38] and SQA3D [39] benchmarks provide additional 3D annotations (e.g., point clouds and depth maps of the scene), we compare Spatial-MLLM with several other model types in addition to video-input MLLM. These includes task-specific models designed for 3D question-answering tasks, such as ScanQA [38], SQA3D [39], 3D-VisTA [71], and LLMs that require point clouds or depth maps as input, such as Chat-Scene [22], Video-3D LLM [25], and 3D-LLaVA [21]. Results. We present the quantitative results on the ScanQA [38] and SQA3D [39] benchmarks in Table 2. As shown, Spatial-MLLM significantly outperforms all video-input models across all metrics on both ScanQA and SQA3D. Our model also surpasses all task-specific models. Among models utilizing 3D or 2.5D input, only 3D-LLaVA [21] (on ScanQA) and Video-3D-LLM [25] (on ScanQA and SQA3D) achieve better performance than Spatial-MLLM. However, 3D-LLaVA requires additional point cloud input, and Video-3D-LLM depends on depth maps. Despite not relying on any additional 3D or 2.5D input, our model still outperforms other 3D-dependent models such as 3D-LLM [48], LL3DA [23], and Chat-Scene [22].
this section cite: ['b37', 'b38', 'b63', 'b24', 'b49', 'b37', 'b38', 'b37', 'b38', 'b70', 'b21', 'b24', 'b20', 'b37', 'b38', 'b20', 'b24', 'b47', 'b22', 'b21']

Section: Ablation Study and Analysis
Ablation on Input Frame Number. We evaluate the effect of the number of input frames on VSI-Bench across different models, including Spatial-MLLM, Gemini-1.5 Pro [4], and Qwen2.5-VL-3B [14]. The result is shown in Table . 4. For the 1 fps setting of Gemini-1.5 Pro, we upload the entire video to the model following the VSI-Bench [18], where the video is sampled at 1 fps according to the API instructions. For the 0.1 fps and 0.25 fps settings, we first manually sample the video frames and then upload these sampled frames to the model. As shown, all models exhibit improved performance as the number of input frames increases, particularly when the number of frames is small.
this section cite: ['b3', 'b13', 'b17']

Section: Effectiveness of Space-aware Frame Sampling.
We evaluate different frame sampling configurations in Table 4, including 8, 16, and 32 frames using uniform sampling and space-aware frame sampling. As shown, increasing the number of sampled frames improves performance for both space-aware frame sampling and uniform sampling. Compared with uniform sampling, space-aware frame sampling consistently outperforms it when the number of input frames is the same.
We further provide a visualization of our space-aware frame sampling in Fig. 5, which shows the point maps (predicted by the VGGT [32]) corresponding to the frames selected by different sampling strategies. As shown, the proposed space-aware frame sampling strategy consistently yields more spatial coverage than uniform sampling, which often overlooks transient regions that appear briefly in the video and tends to produce redundant viewpoints when the camera remains static. Effectiveness of RL Training. We evaluate Spatial-MLLM's performance before and after GRPO training on VSI-Bench. The results are presented in the second (SFT + GRPO) and third (SFT) rows of Table 3.
As shown, even though we conduct only small scale RL training (i.e., 1,000 steps), the GRPO-trained model still achieves performance gains, suggesting that long chain-of-thought reasoning enhances the spatial reasoning capabilities required by VSI-Bench [18].
this section cite: ['b31', 'b17']

Section: Effectiveness of the Spatial-MLLM Architecture and Training Dataset.
We compare the supervised fine-tuned version of Spatial-MLLM, two supervised fine-tuned versions of Qwen2.5-VL-3B [14] (the base model of Spatial-MLLM) and original Qwen2.5-VL-3B model in Table 3. † denotes results obtained with the R1-V [72] training framework. ‡ denotes results which we further apply a question token mask during the loss computation process within R1-V [72], which aligns better with Spatial-MLLM training process. As shown 3, both SFT versions of Qwen2.5-VL-3B  show improvements, indicating the effectiveness of our proposed dataset to enhance the model's spatial reasoning capabilities. Furthermore, both models underperform compared to the supervised fine-tuned version of Spatial-MLLM, which validates the effectiveness of the proposed architecture.
this section cite: ['b13', 'b71', 'b71']

Section: Conclusion
We introduce Spatial-MLLM, a method that enables effective spatial understanding and reasoning from purely 2D visual inputs. By combining a semantic 2D encoder with a structure-aware spatial encoder initialized from a visual geometry foundation model, our dual-encoder design captures both semantic and spatial cues. Additionally, our proposed space-aware frame sampling strategy further enhances performance under limited input constraints. Trained on the collected dataset, our model achieves state-of-the-art results across multiple benchmarks.
Limitations and Future Work. Although Spatial-MLLM demonstrates significant improvements over previous video MLLMs across a wide range of visual-based spatial understanding and reasoning tasks, there remains room to scale Spatial-MLLM further in terms of model size and training data. Moreover, as this work primarily addresses visual-based spatial intelligence, we have trained and evaluated our model specifically on relevant datasets and benchmarks. An interesting direction for future work would be to explore how integrating spatial structural information might further benefit general video understanding and reasoning tasks. Justification: We discuss possible limitations of our method in Sec. 5.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: References
Ref_id:b0 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b1 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b2 Title: Visual instruction tuning Year: (2024)
Ref_id:b3 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b4 Title: Gpt-4o system card Year: (2024)
Ref_id:b5 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b6 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b7 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b8 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b9 Title: Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms Year: (2024)
Ref_id:b10 Title: Streaming long video understanding with large language models Year: (2024)
Ref_id:b11 Title: Video instruction tuning with synthetic data Year: (2024)
Ref_id:b12 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b13 Title: Qwen2.5-vl technical report Year: (2025)
Ref_id:b14 Title: Audiogpt: Understanding and generating speech, music, sound, and talking head Year: (2023)
Ref_id:b15 Title: Salmonn: Towards generic hearing abilities for large language models Year: (2023)
Ref_id:b16 Title: Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment Year: (2025)
Ref_id:b17 Title: Thinking in space: How multimodal large language models see, remember, and recall spaces Year: (2024)
Ref_id:b18 Title: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities Year: (2024)
Ref_id:b19 Title: Sti-bench: Are mllms ready for precise spatial-temporal world understanding? Year: (2025)
Ref_id:b20 Title: 3d-llava: Towards generalist 3d lmms with omni superpoint transformer Year: (2025)
Ref_id:b21 Title: Chat-scene: Bridging 3d scene and large language models with object identifiers Year: (2024)
Ref_id:b22 Title: Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning Year: (2024)
Ref_id:b23 Title: Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness Year: (2024)
Ref_id:b24 Title: Video-3d llm: Learning position-aware video representation for 3d scene understanding Year: (2024)
Ref_id:b25 Title: Datacomp: In search of the next generation of multimodal datasets Year: (2023)
Ref_id:b26 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b27 Title: Tulip: Towards unified language-image pretraining Year: (2025)
Ref_id:b28 Title: Beyond semantics: Rediscovering spatial awareness in vision-language models Year: (2025)
Ref_id:b29 Title: Long-clip: Unlocking the long-text capability of clip Year: (2024)
Ref_id:b30 Title: Dust3r: Geometric 3d vision made easy Year: (2023)
Ref_id:b31 Title: Vggt: Visual geometry grounded transformer Year: (2025)
Ref_id:b32 Title: Megasam: Accurate, fast, and robust structure and motion from casual dynamic videos Year: (2024)
Ref_id:b33 Title: permutation-equivariant visual geometry learning Year: (2025)
Ref_id:b34 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b35 Title: Deepseekmath: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b36 Title: Chain of thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b37 Title: Scanqa: 3d question answering for spatial scene understanding Year: (2021)
Ref_id:b38 Title: Sqa3d: Situated question answering in 3d scenes Year: (2022)
Ref_id:b39 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b40 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b41 Title: Pandagpt: One model to instruction-follow them all Year: (2023)
Ref_id:b42 Title: Detgpt: Detect what you need via reasoning Year: (2023)
Ref_id:b43 Title: Videochat: Chat-centric video understanding Year: (2023)
Ref_id:b44 Title: Grounded 3d-llm with referent tokens Year: (2024)
Ref_id:b45 Title: Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes Year: (2023)
Ref_id:b46 Title: An embodied generalist agent in 3d world Year: (2023)
Ref_id:b47 Title: 3d-llm: Injecting the 3d world into large language models Year: (2023)
Ref_id:b48 Title: Scene-llm: Extending language model for 3d visual understanding and reasoning Year: (2024)
Ref_id:b49 Title: Gpt4scene: Understand 3d scenes from videos with vision-language models Year: (2025)
Ref_id:b50 Title: Vlm-3r: Visionlanguage models augmented with instruction-aligned 3d reconstruction Year: (2025)
Ref_id:b51 Title: Spacer: Reinforcing mllms in video spatial reasoning Year: (2025)
Ref_id:b52 Title: Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution Year: (2024)
Ref_id:b53 Title: Videoagent: Long-form video understanding with large language model as agent Year: (2024)
Ref_id:b54 Title: St-think: How multimodal large language models reason about 4d worlds from ego-centric videos Year: (2025)
Ref_id:b55 Title: Vlm4d: Towards spatiotemporal awareness in vision language models Year: (2025)
Ref_id:b56 Title: Multi-modal situated reasoning in 3d scenes Year: (2024)
Ref_id:b57 Title: Spatial mental modeling from limited views Year: (2025)
Ref_id:b58 Title: Attention is all you need Year: (2017)
Ref_id:b59 Title: Vision transformers need registers Year: (2023)
Ref_id:b60 Title: Transfer between modalities with metaqueries Year: (2025)
Ref_id:b61 Title: An analysis of approximations for maximizing submodular set functions-i Year: (1978)
Ref_id:b62 Title: Approximating covering and packing problems: set cover, vertex cover, independent set, and related problems Year: (1996)
Ref_id:b63 Title: Scannet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017)
Ref_id:b64 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b65 Title: Longvila: Scaling long-context visual language models for long videos Year: (2024)
Ref_id:b66 Title: Vila: On pre-training for visual language models Year: (2023)
Ref_id:b67 Title: Long context transfer from language to vision Year: (2024)
Ref_id:b68 Title: Scannet++: A high-fidelity dataset of 3d indoor scenes Year: (2023)
Ref_id:b69 Title: ARKitscenes -a diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data Year: ()
Ref_id:b70 Title: 3d-vista: Pre-trained transformer for 3d vision and text alignment Year: (2023)
Ref_id:b71 Title: R1-v: Reinforcing super generalization ability in visionlanguage models with less than $3 Year: ()
Ref_id:b72 Title: Open3d: A modern library for 3d data processing Year: (2018)
Ref_id:b73 Title: Indoor segmentation and support inference from rgbd images Year: (2012)
Ref_id:b74 Title: Perceptual organization and recognition of indoor scenes from rgb-d images Year: (2013)
