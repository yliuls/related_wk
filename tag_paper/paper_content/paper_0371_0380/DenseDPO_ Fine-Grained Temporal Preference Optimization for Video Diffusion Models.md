Title: DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models
Abstract: Direct Preference Optimization (DPO) has recently been applied as a post-training technique for text-to-video diffusion models.To obtain training data, annotators are asked to provide preferences between two videos generated from independent noise. However, this approach prohibits fine-grained comparisons, and we point out that it biases the annotators towards low-motion clips as they often contain fewer visual artifacts. In this work, we introduce DenseDPO, a method that addresses these shortcomings by making three contributions. First, we create each video pair for DPO by denoising corrupted copies of a ground truth video. This results in aligned pairs with similar motion structures while differing in local details, effectively neutralizing the motion bias. Second, we leverage the resulting temporal alignment to label preferences on short segments rather than entire clips, yielding a denser and more precise learning signal. With only one-third of the labeled data, DenseDPO greatly improves motion generation over vanilla DPO, while matching it in text alignment, visual quality, and temporal consistency. Finally, we show that DenseDPO unlocks automatic preference annotation using off-the-shelf Vision Language Models (VLMs): GPT accurately predicts segment-level preferences similar to task-specifically fine-tuned video reward models, and DenseDPO trained on these labels achieves performance close to using human labels. Additional results are available at https://snap-research.github.io/DenseDPO/.pre-collected human preference pairs, bypassing the need for online reward feedback. Building on this paradigm, subsequent works have explored improving comparison data pairs [28,32,88,90], better DPO objectives [29,41,73], and credit assignment over denoising timesteps [43,94].Preference learning for video diffusion. Early approaches to preference learning in video diffusion directly borrow techniques from image diffusion, such as direct reward optimization [40,56,60,68,87] and training loss re-weighting [19]. However, they often rely on image reward models [33,75] to provide supervision. Recent papers thus focus on developing better video reward models [46,47,80]. One strategy aggregates multiple video quality assessment metrics [21,30] to a final score [47,91]. However, existing metrics are only effective for short videos [46,80], limiting their applicability for modern video generators that produce long videos [7,71,84]. To address this limitation, LiFT [74], VisionReward [80], and VideoAlign [46]  collect a large number of videos from advanced video generators, label human feedback, and fine-tune VLMs to predict preferences. With a powerful video reward model, they apply weighted training [74] or DPO [46,80] to improve video generation. In contrast to prior works, we focus on DPO for video diffusion using direct human annotations, i.e., without an explicit reward model. Analogous to the verbosity bias observed in language model preference learning, where annotators favor longer outputs [64, 67], we identify a motion bias in video preference labels, where slow-motion videos are often preferred. To mitigate this, we propose a better data pair construction strategy to address this bias via guided video generation.

Section: Introduction
Recent advances in diffusion models [23] have enabled high-quality text-guided video generation [2, 7,24,35,58,66,71]. Despite tremendous progress, existing video generators still fall short on temporal coherence, visual fidelity, and prompt alignment [89], impeding their industry-level applications.
Inspired by the success of learning from human feedback in language models [3,52] and image diffusion [5,17,70], recent works have explored preference alignment in video diffusion [40,60,87]. Among them, methods based on Direct Preference Optimization (DPO) [61] stand out as they bypass the need for an explicit reward model [10,47,50,65]. However, existing DPO methods for video diffusion are largely adapted from their image-based counterparts, without addressing the unique challenges inherent to video generation. Typically, these methods first generate videos from independent noise maps, followed by human preference labeling to construct comparison pairs. Yet, human preferences in video are influenced by multiple, sometimes inversely correlated, factors, such as the visual quality (i.e., pixel-level fidelity) and the dynamic degree (i.e., strength of global motion). Indeed, current video generation models excel at producing high-quality slow-motion videos, while struggling to synthesize more challenging dynamic scenes [8]. As a result, when annotators are asked to express preferences, they often favor artifact-free slow-motion clips. Applying DPO training on such preference data further reinforces video generators' bias toward slow-motion content, ultimately suppressing the model's ability to generate dynamic and motion-rich videos.
this section cite: ['b20', 'b4', 'b21', 'b32', 'b54', 'b62', 'b67', 'b85', 'b0', 'b48', 'b2', 'b14', 'b66', 'b36', 'b56', 'b83', 'b57', 'b7', 'b43', 'b46', 'b61', 'b5']

Section: 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
"A young adult male doing a handstand on the beach" "A woman doing push-up exercise on a beach at sunset" "A monkey performs a jump on a skateboard at the skate park, landing smoothly" "A panda breakdancing in a neon-lit urban alley" A natural approach to enhance competing factors, drawing inspiration from Pareto optimization, is to fix some attributes within each video pair while varying others. Motivated by the guided image synthesis approach in SDEdit [51], we generate a pair by introducing different partial noise to a ground-truth video and perform denoising. The resulting videos in each pair share high-level semantics and motion trajectories while differing in local visual details [8,72], allowing us to reduce spurious correlations. However, guided sampling inherently reduces the diversity across generated video pairs, leading to degraded DPO performance [53,63]. A straightforward solution might be to annotate more data pairs. Instead, we propose extracting richer and more accurate supervision from each video pair by collecting segment-level preference labels.
Prior works show that multi-dimensional scores are superior to a single label in preference alignment [42,80,93]. Unlike images, videos have a unique temporal dimension [78]. In practice, we observe that human preferences over video pairs often vary across time, as artifacts may appear at different timestamps in each video, leading to inconsistent preferences. This issue is more severe with modern video generators as they produce longer videos. Therefore, we split videos into short segments (e.g., 1s slices), and collect per-segment preference labels. Thanks to temporally aligned videos from guided sampling, there is a clear one-to-one correspondence between segment pairs, simplifying the annotation process. Segment feedback also reduces the amount of ties when both videos contain artifacts, and provides more accurate supervision. In addition, it allows us to apply existing vision-language models (VLMs) [1,4] which can produce reliable judgment on short segments.
Our main contributions are threefold: (i) A DenseDPO framework tailored towards video generation, with improved data construction and preference granularity over vanilla DPO; (ii) DenseDPO retains the motion strength of the base model while matching other metrics of vanilla DPO, with significantly higher data efficiency; (iii) We show that existing VLMs fail to label preference over long videos (e.g., 5s), but they perform well in segment-level preference, achieving results close to human labels.
this section cite: ['b47', 'b5', 'b68', 'b49', 'b59', 'b38', 'b76', 'b89', 'b74', 'b1']

Section: Related Work
Preference learning for image diffusion. Inspired by the success of human feedback learning in language modeling [3,52], similar approaches have been adapted to image generation. One major line of work focuses on training reward models from human preference labels [33,75,76,79,93], these models can be used as loss functions to optimize generators by direct gradient backpropagation [9,13,31,59,77,79] or policy gradients [5,17]. Another line of work utilizes predicted rewards on training data to re-weight the diffusion loss [38], or trains only on high-scoring samples [14,15,39]. However, all these methods require an explicitly trained reward model, and may suffer from the reward hacking issue [13,79]. In contrast, Diffusion-DPO [70] and D3PO [83] directly optimize the model on Rich feedback for alignment. While early preference alignment methods treated human feedback as a single binary label, recent works begin to exploit rich, multi-dimensional feedback [42,80,92]. In image generation, MPS [93] learns a reward model that evaluates images on four dimensions including aesthetics, semantics, detail, and overall quality, improving its alignment with humans. On the other hand, Liang et al. [42] curates a dataset that localizes regions of artifacts and misaligned words in the text prompt, leading to better DPO performance. Multi-aspect feedback is even more critical for video generation due to its inherently higher dimensionality. Recent works all explicitly model dimensions such as visual fidelity, text relevance, and motion consistency [46,47,80]. However, a notable limitation is that they still aggregate feedback at the whole-video level, neglecting the finegrained temporal dimension of preferences. In contrast, our DenseDPO partitions videos into short, temporally aligned segments, and collects preferences for each segment. This is conceptually similar to the sentence-level preference label used in language models [34,86]. By localizing feedback to brief windows, we obtain more accurate and denser supervision signals for DPO training.
this section cite: ['b0', 'b48', 'b30', 'b71', 'b72', 'b75', 'b89', 'b6', 'b10', 'b28', 'b55', 'b73', 'b75', 'b2', 'b14', 'b34', 'b11', 'b12', 'b35', 'b10', 'b75', 'b66', 'b79', 'b38', 'b76', 'b88', 'b89', 'b38', 'b42', 'b43', 'b76', 'b31', 'b82']

Section: Method
We build upon diffusion models and the standard Direct Preference Optimization (DPO) framework, which we refer to as VanillaDPO (Sec. 3.1). We discuss the motion bias inherent in using this naïve approach for video generation and introduce StructuralDPO, a method that optimizes human preferences on structurally similar video pairs (Sec. 3.2). To address the reduction of diversity induced from using structurally similar videos, we propose DenseDPO, which enables fine-grained human preference alignment along the temporal axis of videos (Sec. 3.3).
this section cite: []

Section: Background: Video Diffusion and DPO
Rectified-flow diffusion models. Let x ∈ R T ×H×W denote a video sample of length T with spatial dimensions H × W . We follow the rectified flow framework [44,48], which learns a transport map from the standard normal distribution ϵ ∼ N (0, I) to the distribution of real videos x ∼ p data with a denoiser. The forward diffusion process produces a noisy input x t at time t ∈ [0, 1] via a linear interpolation with noise ϵ: x t = (1 -t)x 0 + tϵ. The denoiser G θ (x t , t, c), implemented as a neural network parameterized by θ, is trained to reverse this process with the following objective:
min θ E t∼p(t),x∼pdata,ϵ∼N (0,I) ∥(ϵ -x) -G θ (x t , t, c)∥ 2 ,(1)
where p(t) is the distribution of noise levels (following [16], we adopt the logit-normal one) and c refers to the auxiliary conditioning variable such as text embeddings.
this section cite: ['b40', 'b44', 'b13']

Section: VanillaDPO.
In the direct preference optimization framework [61,70], a generative model is trained to align its outputs with human preferences. Typically, these preferences are defined by a dataset D = {(c, x 0 , x 1 , l)}, where each sample consists of two videos {x 0 , x 1 } per input condition c and their preference label l ∈ {-1, +1}. The preference function is defined as:
l(x 0 , x 1 ) = +1, if x 0 ≻ x 1 (i.e., x 0 is preferred over x 1 ) -1, if x 1 ≻ x 0 (i.e., x 1 is preferred over x 0 ) (2
)
The Bradley-Terry (BT) model [6] defines pairwise preference using a reward function r(x, c) which computes the alignment score between the sample x and the input condition c. The corresponding probabilistic preference can be expressed as:
p BT (x 0 ≻ x 1 |c) = σ(r(x 0 , c) -r(x 1 , c)); p BT (x 1 ≻ x 0 |c) = σ(r(x 1 , c) -r(x 0 , c)),(3)
where σ(•) is the sigmoid function.
Rafailov et al. [61] defines the binary preference optimization as explicitly optimizing the binary reward objective log σ(l(x 0 , x 1 ) * (r(x 0 , c) -r(x 1 , c))) in conjunction with a Kullback-Leibler (KL) divergence regularization to control the deviation from a reference model. Wallace et al. [70] re-formulated the preference optimization framework for diffusion models assuming the presence of a reference model G ref , which is further extended to rectified flow models in [46]. Given a sample (c, x 0 , x 1 , l), the denoiser G θ , and the reference model G ref , we can define an implicit reward as:
s(x * , c, t, θ) = ∥(ϵ * -x * ) -G θ (x * t , t, c)∥ 2 2 -∥(ϵ * -x * ) -G ref (x * t , t, c)∥ 2 2 ,(4)
where
x * t = (1 -t)x * + tϵ * , ϵ * ∼ N (0, I) is a noisy latent for input x * (either x 0 or x 1
) at time t. With the implicit reward function, the VanillaDPO objective is defined as follows:
L(θ) = -E (c,x 0 ,x 1 ,l)∼D, t∼p(t), ϵ∼N (0,I) log σ -β * l(x 0 , x 1 ) * s(x 0 , c, t, θ) -s(x 1 , c, t, θ) . (5)
this section cite: ['b57', 'b66', 'b3', 'b57', 'b66', 'b42']

Section: StructuralDPO: Preference Learning over Structurally Similar Videos
Motion bias in VanillaDPO. In the standard VanillaDPO pipeline, preference pairs are created by independently sampling two videos (x 0 , x 1 ) from different noise seeds under the same conditioning c (see Algo. 1), followed by human preference annotation. While this approach works reasonably well for images, its direct extension to videos introduces new issues due to the presence of the new temporal dimension. Independent noises often result in videos with significantly different motion patterns and global layouts (see Fig. 2 (a)). For example, in a typical preference pair, one video may be nearly static but visually clean, while the other contains the desired motion but also introduces artifacts, such as distorted limbs or flickering. η=0.85 η=0.75 η=0. 65
this section cite: ['b61']

Section: GT video No Guidance
Figure 3: Guided video generation with different η. Lower η means more guidance. We sample one frame per video for visualization. η = 0.75 is enough to maintain the motion trajectory and high-level semantics of the ground-truth video. For slow-motion videos (top), a high η suffices to generate artifact-free videos, while videos with challenging motion (bottom) require more guidance.
We empirically observe that this is a common bias in generated video data. Indeed, video models often excel at producing high-quality slow-motion clips, while dynamic videos usually contain visible artifacts [8,89]. Since humans typically perceive clean static videos as more realistic than artifactprone dynamic ones, this often leads to a preference dataset that systematically over-represents static content. Consequently, a model trained with DPO on this dataset would produce videos with reduced motion. In our preliminary DPO experiments, we observed a substantial drop in dynamic degree using our base model (see Tab. 1 and Tab. 2). The same issue has also been observed in prior works, e.g., Tab.10 of [80] using CogVideoX [84] and Tab. 2 of [47] using VideoCrafter-v2 [11].
this section cite: ['b5', 'b85', 'b76', 'b80', 'b43', 'b8']

Section: StructuralDPO.
To address the static bias in VanillaDPO, we propose StructuralDPO, which modifies the data curation strategy using guided generation [51] to obtain pairs of videos with similar motion trajectories. Specifically, in VanillaDPO, we start from independent noise, i.e., x 0 N ∼ N (0, I); x 1 N ∼ N (0, I) and then denoise from step N till step 1. Instead, we propose to denoise from a partially noised real video to generate video pairs. Concretely, given a ground truth video x and the guidance level η ∈ [0, 1], we obtain the corrupted noisy video at step n = round(η * N ) as:
x 0 n = (1 -η)x + ηϵ 0 ; x 1 n = (1 -η)x + ηϵ 1 ; where ϵ 0 , ϵ 1 ∼ N (0, I). (6
)
The corrupted videos are then denoised from step n to 1, as outlined in Algo. 2. Here, η governs the structural similarity between the two samples. Since early diffusion steps control the global motion [8], this approach preserves the overall dynamics while allowing variations in local details. We compare videos of varying motion strengths generated with different η in Fig. 3.
StructuralDPO applies the standard DPO formulation (Eq. ( 5)) on this structurally consistent dataset, which helps to focus the preference on the temporal artifacts and visual inconsistencies, anchoring other dimensions like dynamic degree. Additionally, guided sampling simplifies the generation task, allowing the model to produce artifact-free highly dynamic videos more reliably. Finally, guided sampling reduces data construction costs as it requires fewer sampling steps.
Algorithm 1 Vanilla Paired Video Generation Input: Denoiser G θ , Input Condition c, Infer- ence Steps N Init: ∆ t = 1 N Init: x 0 N ∼ N (0, I) Init: x 1 N ∼ N (0, I) for i = N to 1 do t = i N x 0 i-1 = x 0 i + G θ (x 0 i , t, c) * ∆ t x 1 i-1 = x 1 i + G θ (x 1 i , t, c) * ∆ t end for return Video Pair (x 0 0 , x 1 0 ) Algorithm 2 Guided Paired Video Generation Input: Denoiser G θ , Input Condition c, Inference Steps N , Real Video x, Guidance Level η ∈ [0, 1] Init: ∆ t = 1 N , n = round(η * N ) Init: x 0 n = (1 -η)x + ηϵ 0 , ϵ 0 ∼ N (0, I) Init: x 1 n = (1 -η)x + ηϵ 1 , ϵ 1 ∼ N (0, I) for i = n to 1 do t = i N x 0 i-1 = x 0 i + G θ (x 0 i , t, c) * ∆ t x 1 i-1 = x 1 i + G θ (x 1 i , t, c) * ∆ t end for return Video Pair (x 0 0 , x 1 0 )
this section cite: ['b47', 'b5']

Section: DenseDPO: Rich Temporal Feedback with Segment-Level Preferences
Although StructuralDPO effectively preserves dynamic degree, models trained with it tend to generate videos with lower visual quality and weaker text alignment compared to VanillaDPO. This performance gap is because video pairs are structurally similar, which reduces diversity in the curated DPO dataset, and, as we discuss in Appendix B, can unintentionally drive the model to diverge from the real data distribution [53,63]. A straightforward solution is to obtain more labeled data, but it increases the annotation cost compared to VanillaDPO. Instead, we explore an alternative approach to increase data and annotation diversity without increasing the number of labeled video pairs.
DenseDPO. In VanillaDPO and StructuralDPO, a scalar preference label l ∈ {-1, +1} is obtained for the entire video of length T . Instead, DenseDPO annotates preferences on shorter temporal segments. Since guided video generation (Algo. 2) yields structurally similar video pairs, the same time period in both videos has a clear correspondence, making comparison feasible. We show an example in Fig. 2 (b) with the intervals being a single frame: for frame 1 and 2, the first video is better, while for frame 3, the second video is better. Formally, given two videos (x 0 , x 1 ) and the interval length s, we can break down videos into F = ceil( T s ) temporal segments of length s by splitting along the time dimension. The resulting video pairs ({x 0 f , x 1 f } F f =1 ) are annotated with preferences over each segment, yielding segment-level dense preference labels l ∈ {-1, +1} F , i.e., l(x 0 , x 1 ) = [l(x 0 f , x 1 f )] F f =1 . Thus, following Eq. ( 5), we can formulate the DenseDPO objective as:
L(θ) = -E (c,x 0 ,x 1 ,l)∼D t∼p(t), ϵ∼N (0,I) log σ   -β F f =1 l(x 0 f , x 1 f ) * s(x 0 , c, t, θ) f -s(x 1 , c, t, θ) f   , (7
)
where s(•) f is the implicit reward value on the f -th video segment.
In our collected dense preference data, we find that over 60% of video pairs have both winning and losing labels in l. In regular preference annotation such pairs will either be treated as ties or choose the video with fewer artifacts. In the latter case, this encourages the model to minimize loss on videos with artifacts in Eq. ( 5), degrading the model performance. In contrast, DenseDPO assigns preference labels more accurately over time, only optimizing models on segments with a clear difference.
Segment preference annotation with VLMs. Another benefit of the DenseDPO is that it allows us to use off-the-shelf VLMs for automatic preference labeling. Prior works point out that existing VLMs struggle at assessing long videos (e.g., 5s) [21,80], often requiring task-specific fine-tuning and large-scale human annotations to train effective video reward models. Instead, we show that pre-trained VLMs are already capable of processing short clips (e.g., 1s). Given two temporally aligned videos, we feed in pairs of segments into a VLM, and ask it to identify the better one. As we will demonstrate in the experiments (Tab. 3), GPT-o3 [1] achieves high accuracy on segment-level preferences, leading to DPO results competitive with using human preference labels.
this section cite: ['b49', 'b59', 'b18', 'b76']

Section: Experiments
Our experiments aim to answer the following questions: (i) How does DenseDPO perform against VanillaDPO? (Sec. 4.2) (ii) Can we leverage existing VLMs to produce high-quality preference labels? (Sec. 4.3) (iii) What is the impact of each component in our framework? (Sec. 4.4)
this section cite: []

Section: Experimental Setup
We list some key aspects of our experimental setup here. For full details, please refer to Appendix A.
Preference learning data. We curate a high-quality video dataset from existing datasets, resulting in around 55k videos. This is done by filtering the length, visual quality, and motion score of videos similar to [58], and prompting GPT-4o [3] to classify if the text prompt contains events of meaningful dynamics. This naturally gives us text prompts and corresponding ground-truth videos.
Baselines. Our pre-trained text-to-video generator is a DiT [55]-based latent flow model. We finetune it on the curated high-quality data, termed the SFT baseline. For VanillaDPO, we randomly select 30k text prompts from the curated dataset, generate 2 videos of 5s per prompt with Algo. 1, and ask human labelers to annotate preferences. This leads to around 10k winning-losing pairs after Table 1: Quantitative results on VideoJAM-bench [8]. We report automatic metrics from VBench [30] and VisionReward [80]. DenseDPO significantly outperforms Vanilla DPO in dynamic degree, while achieves similar performance in other dimensions.  5). Following prior works [46,80], we set β to 500 and apply LoRA [27] with rank 128 to fine-tune the video model. We train with the AdamW optimizer [49] and a global batch size of 256 for 1000 steps.
this section cite: ['b54', 'b0', 'b51', 'b5', 'b27', 'b76', 'b42', 'b76', 'b24', 'b45']

Section: Method

this section cite: []

Section: VBench Metrics VisionReward Metrics

this section cite: []

Section: Aesthetic

this section cite: []

Section: DenseDPO implementation details.
For fair comparison with baselines, we only take 10k video pairs from the StructuralDPO training data to label dense preferences, which costs a similar amount of human annotation time. The segment length s is set to 1s. Overall, more than 80% of video pairs have at least 1 non-tie segment and can be used in DPO training, greatly improving the data efficiency over using global preferences. All other hyper-parameters are the same as DPO baselines.
Evaluation datasets. We utilize two benchmarks to evaluate the performance of text-to-video generation. VideoJAM-bench [8] contains 128 prompts focusing on real-world scenarios with challenging motion, ranging from human actions to physical phenomena. We also construct MotionBench, which collects more diverse prompts from existing prompt sets [35,58,74] such as MovieGenBench. We run GPT-4o to select prompts with dynamic human actions, resulting in 419 prompts.
Evaluation metrics. We aim to measure the visual quality, text alignment, and motion quality of videos. Specifically, we want to evaluate both the smoothness and strength of the motion. Therefore, we adopt VBench [30] and a state-of-the-art video quality assessment model, VisionReward [80].
this section cite: ['b5', 'b32', 'b54', 'b70', 'b27', 'b76']

Section: DPO with Human Labels
Tab. 1 and Tab. 2 present the quantitative results on VideoJAM-bench and MotionBench. In addition, it matches all aspects of VanillaDPO and scores a significantly higher dynamic degree, despite using only one-third of labeled videos (10k vs. 30k). Please refer to Appendix C.1 for comparison with more baselines including online RL-based methods.
For more qualitative results, please check out Appendix C.2 and our project page for video results. DenseDPO is the only method that generates correct limbs, large dynamics, and high quality visuals.
Please check out our project page for video results of baselines and our methods.
TA VQ TC DD 16.6% 42.7% 36.0% 43.0% 74.9% 36.7% 35.1% 46.4% 8.6% 20.6% 28.9% 10.7% DenseDPO wins Ties StructuralDPO wins TA VQ TC DD 18.4% 38.0% 36.4% 63.9% 63.0% 26.0% 27.6% 23.5% 18.6% 35.9% 36.0% 12.6% DenseDPO wins Ties VanillaDPO wins Human evaluation. We conduct a user study using all prompts from VideoJAM-bench in Fig. 5. We ask the participants to express their preference when presented with paired samples from our method and each baseline. DenseDPO consistently outperforms StructuralDPO in all dimensions. Compared to VanillaDPO, we achieve significantly higher dynamic degree, and are on par in other aspects. Please refer to Appendix C.2 for more user study results.
this section cite: []

Section: DPO with VLM Labels
VLM labelers. As discussed in Sec. 3.3, we aim to evaluate the effectiveness of pre-trained VLMs in video preference learning. We take VLMs designed specifically for the video quality assessment task, VideoScore [21], LiFT [74], VideoReward [46], and VisionReward [80]. They have been fine-tuned on large-scale human preference labels. We also utilize the state-of-the-art visual reasoning model, GPT o3 [1], to explore the limits of models without task-specific training. Finally, we design GPT o3 Segment that partitions long videos into short segments to process separately, and aggregates results via majority voting. Please refer to Appendix A.6 for more implementation details.
Evaluation setup. We first test the preference prediction accuracy of VLMs on two types of videos: Short Segment partitions videos into 1s clips, and compares them separately. We report the accuracy on the 10k human preference labels used in DenseDPO. Long Video directly runs the model on the entire video except for GPT o3 Segment that aggregates segment-level results. We report the accuracy on the 30k human preference labels used in StructuralDPO. In addition, we conduct DPO training using these VLM-generated labels, and report the VisionReward score on VideoJAM-bench. Notably, we run VLMs to label video pairs generated from all 55k training data for better performance.
Results. Tab. 3a presents the results of preference prediction. With task-specific fine-tuning, stateof-the-art video reward models outperform the advanced GPT o3 in assessing short clips. Yet, their  performance on long videos degrades drastically. Our further analysis in Appendix C.3 reveals that these models might be biased towards video content rather than temporal motion. Thanks to the temporal alignment of video pairs, we can run GPT to compare short segments individually, and then aggregate the results. This leads to higher accuracy in long video preference prediction.
We further show the DPO alignment results in Tab. 3b. Due to a higher preference accuracy, DenseDPO with GPT o3 Segment labels outperforms StructuralDPO with binary preferences from other VLMs. It even matches DenseDPO trained with human labels on text alignment, visual quality, and dynamic degree. Yet, please note that GPT label has 5.5× more videos than human label.
this section cite: ['b18', 'b70', 'b42', 'b76']

Section: Ablation Study
We study the effect of each component in DenseDPO. All results are evaluated on VideoJAM-bench.
Human labeling bias. We first study if there is a systematic bias in the annotation pipeline, e.g., labelers may produce higher quality preference labels on short segments than long videos. To verify it, we aggregate labels of all segments within a video to obtain its binary preference label via majority voting, and then train StructuralDPO on these labels. Tab. 4 shows that it achieves similar results compared to StructuralDPO trained on globally labeled preferences. This proves that DenseDPO's superior performance comes from segment-level preference supervision instead of labeler bias.
this section cite: []

Section: Human label quality.
We study how label noise affects DenseDPO performance. As shown in Tab. 4, we randomly flip 20% and 40% winning or losing labels. This results in a clear drop in all the metrics.
this section cite: []

Section: Human label quantity.
We ablate different amounts of segment preference labels used in DenseDPO. Tab. 4 shows that scaling to 2× labels leads to the best results across all metrics. Interestingly, results
Table 5: Ablation on segment length s of dense preference labels. We report VBench [30] and VisionReward [80] metrics on VideoJAM-bench [8]. All models are only trained on 5k videos. trained with 0.5× labels are clearly better than results trained with 40% labels flipped. This may indicate that the quality of labels has a larger impact than the quantity of labels.
this section cite: ['b27', 'b76', 'b5']

Section: Method

this section cite: []

Section: VBench Metrics VisionReward Metrics

this section cite: []

Section: Aesthetic

this section cite: []

Section: Ground-truth video in guided generation.
In our experiments, we randomly sample 10k videos from the 55k high-quality dataset to serve as guidance videos. Here, we randomly sample another set of 10k videos without overlap with the previously selected ones and conduct DenseDPO training. Tab. 4 (New GT Video row) shows that both runs achieve similar results, proving that our method is robust to the selection of ground-truth data as long as they are high-quality videos.
this section cite: []

Section: Dense label granularity.
We study the impact of the segment length s in dense preference labels. By default, s = 1 is used in all our experiments. Here, we tested s = 0.5 and s = 2. Due to the high annotation cost, we only label 5k videos for each segment setting in this study. Tab. 5 compares DenseDPO trained on 5k videos using different s. s = 1 consistently outperforms s = 2 due to more fine-grained preference annotation. Interestingly, s = 0.5 performs similarly to s = 1. We hypothesize that this is because 0.5s is too short-a longer context window is needed to assess the temporal aspect of videos. In addition, labeling dense preference at s = 0.5 is 2× expensive compared to s = 1. Therefore, we choose to label 1s segments in our experiments.
this section cite: []

Section: VLM label quantity.
Finally, we study DenseDPO performance when scaling up automatic VLM labels in Tab. 6. Since VLM labels are noisy, DenseDPO with 10k GPT labels achieves only marginal improvement compared to the pre-trained model. Nevertheless, as the amount of VLM labels grows, we see consistent improvement in all metrics. This trend reveals that the quantity of VLM labels compensates for their quality, and our method scales well with the quantity of automatic labels. Since VLM itself is an actively evolving field, we view this as a promising future direction to further scale up automatic preference learning with more data and better off-the-shelf VLMs.
this section cite: []

Section: Conclusion
We present DenseDPO, an improved preference optimization framework for video generation. We address two critical aspects of video DPO-comparison data curation and preference labeling. Our guided video generation mechanism and fine-grained preference labeling significantly improve over vanilla DPO. Furthermore, we show that segment-level labels unlock automatic annotation with off-the-shelf VLMs without task-specific training. We discuss our limitations in Appendix D.
Answer: [NA] Justification: The main paper does not include theoretical results. We have an analysis of a potential failure cause of StructuralDPO in Appendix B with detailed derivations. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. GPT-4 technical report Year: (2023)
Ref_id:b1 Title: Qwen2.5-VL technical report Year: (2025)
Ref_id:b2 Title: Training diffusion models with reinforcement learning Year: ()
Ref_id:b3 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b4 Title: Video generation models as world simulators Year: (2024)
Ref_id:b5 Title: VideoJAM: Joint appearance-motion representations for enhanced motion generation in video models Year: (2025)
Ref_id:b6 Title: Enhancing diffusion models with text-encoder reinforcement learning Year: (2024)
Ref_id:b7 Title: SkyReels-V2: Infinite-length film generative model Year: (2025)
Ref_id:b8 Title: VideoCrafter2: Overcoming data limitations for high-quality video diffusion models Year: (2024)
Ref_id:b9 Title: Panda-70M: Captioning 70m videos with multiple cross-modality teachers Year: (2024)
Ref_id:b10 Title: Directly fine-tuning diffusion models on differentiable rewards Year: (2023)
Ref_id:b11 Title: Emu: Enhancing image generation models using photogenic needles in a haystack Year: (2023)
Ref_id:b12 Title: RAFT: Reward ranked finetuning for generative foundation model alignment Year: (2023)
Ref_id:b13 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b14 Title: DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models Year: (2023)
Ref_id:b15 Title: Towards analyzing and understanding the limitations of dpo: A theoretical perspective Year: (2024)
Ref_id:b16 Title: Improving dynamic object interactions in text-to-video generation with ai feedback Year: (2024)
Ref_id:b17 Title: On the content bias in fréchet video distance Year: (2024)
Ref_id:b18 Title: VideoScore: Building automatic metrics to simulate fine-grained human feedback for video generation Year: (2009)
Ref_id:b19 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b20 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b21 Title: Imagen Video: High definition video generation with diffusion models Year: (2022)
Ref_id:b22 Title: Marginaware preference optimization for aligning diffusion models without reference Year: (2025)
Ref_id:b23 Title: CogVLM2: Visual language models for image and video understanding Year: (2024)
Ref_id:b24 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b25 Title: D-Fusion: Direct preference optimization for aligning diffusion models with visually consistent samples Year: (2025)
Ref_id:b26 Title: PatchDPO: Patch-level dpo for finetuning-free personalized image generation Year: (2025)
Ref_id:b27 Title: VBench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b28 Title: CoMat: Aligning text-to-image diffusion model with image-to-text concept matching Year: (2024)
Ref_id:b29 Title: Scalable ranked preference optimization for text-to-image generation Year: (2024)
Ref_id:b30 Title: Pick-a-Pic: An open dataset of user preferences for text-to-image generation Year: (2023)
Ref_id:b31 Title: SDPO: Segment-level direct preference optimization for social agents Year: (2025)
Ref_id:b32 Title: HunyuanVideo: A systematic framework for large video generative models Year: (2024)
Ref_id:b33 Title: Open-Sora-Plan Year: (2024-04)
Ref_id:b34 Title: Aligning text-to-image models using human feedback Year: (2023)
Ref_id:b35 Title: Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation Year: (2024)
Ref_id:b36 Title: T2V-Turbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design Year: (2025)
Ref_id:b37 Title: Aligning diffusion models by optimizing human utility Year: (2024)
Ref_id:b38 Title: Rich human feedback for text-to-image generation Year: (2024)
Ref_id:b39 Title: Aesthetic post-training diffusion models from generic preferences with step-by-step preference optimization Year: (2025)
Ref_id:b40 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b41 Title: Flow-GRPO: Training flow matching models via online rl Year: (2025)
Ref_id:b42 Title: Improving video generation with human feedback Year: (2025)
Ref_id:b43 Title: VideoDPO: Omni-preference alignment for video diffusion generation Year: (2025)
Ref_id:b44 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b45 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b46 Title: Step-Video-T2V technical report: The practice, challenges, and future of video foundation model Year: (2025)
Ref_id:b47 Title: SDEdit: Guided image synthesis and editing with stochastic differential equations Year: ()
Ref_id:b48 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b49 Title: Fixing failure modes of preference optimisation with dpo-positive Year: (2024)
Ref_id:b50 Title: PyTorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b51 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b52 Title: Training-free diffusion model alignment with sampling demons Year: (2025)
Ref_id:b53 Title: SDXL: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b54 Title: Movie Gen: A cast of media foundation models Year: (2024)
Ref_id:b55 Title: Aligning text-toimage diffusion models with reward backpropagation Year: (2023)
Ref_id:b56 Title: Video diffusion alignment via reward gradients Year: (2024)
Ref_id:b57 Title: Direct Preference Optimization: Your language model is secretly a reward model Year: (2004)
Ref_id:b58 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b59 Title: Unintentional Unalignment: Likelihood displacement in direct preference optimization Year: (2025)
Ref_id:b60 Title: Verbosity bias in preference labeling by large language models Year: (2024)
Ref_id:b61 Title:  Year: (2025)
Ref_id:b62 Title:  Year: (2024)
Ref_id:b63 Title: A long way to go: Investigating length correlations in rlhf Year: (2024)
Ref_id:b64 Title: Tuning-free alignment of diffusion models with direct noise optimization Year: (2024)
Ref_id:b65 Title: Towards accurate generative models of video: A new metric & challenges Year: ()
Ref_id:b66 Title: Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization Year: (2024)
Ref_id:b67 Title: Wan: Open and advanced large-scale video generative models Year: (2025)
Ref_id:b68 Title: Diffusion models generate images like painters: an analytical theory of outline first, details later Year: (2023)
Ref_id:b69 Title: Diffusion-NPO: Negative preference optimization for better preference aligned generation of diffusion models Year: (2025)
Ref_id:b70 Title: Leveraging human feedback for text-to-video model alignment Year: (2009)
Ref_id:b71 Title: Human Preference Score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis Year: (2023)
Ref_id:b72 Title: Human Preference Score: Better aligning text-to-image models with human preference Year: (2023)
Ref_id:b73 Title: Deep reward supervisions for tuning text-to-image diffusion models Year: (2024)
Ref_id:b74 Title: Mind the Time: Temporally-controlled multi-event video generation Year: (2025)
Ref_id:b75 Title: ImageReward: Learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b76 Title: VisionReward: Fine-grained multi-dimensional human preference learning for image and video generation Year: (2024)
Ref_id:b77 Title: Advancing high-resolution video-language representation with large-scale video transcriptions Year: (2022)
Ref_id:b78 Title: Unleashing grpo on visual generation Year: (2025)
Ref_id:b79 Title: Using human feedback to fine-tune diffusion models without any reward model Year: (2024)
Ref_id:b80 Title: CogVideoX: Text-to-video diffusion models with an expert transformer Year: ()
Ref_id:b81 Title: Language model beats diffusion-tokenizer is key to visual generation Year: (2023)
Ref_id:b82 Title: RLHF-V: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback Year: (2024)
Ref_id:b83 Title: InstructVideo: Instructing video diffusion models with human feedback Year: (2024)
Ref_id:b84 Title: Self-play fine-tuning of diffusion models for text-to-image generation Year: (2024)
Ref_id:b85 Title: The dawn of video generation: Preliminary explorations with sora-like models Year: (2024)
Ref_id:b86 Title: Semi-policy preference optimization for diffusion alignment Year: (2024)
Ref_id:b87 Title: OnlineVPO: Align video diffusion model with online video-centric preference optimization Year: (2024)
Ref_id:b88 Title: Perceptual artifacts localization for image synthesis tasks Year: (2023)
Ref_id:b89 Title: Learning multi-dimensional human preference for text-to-image generation Year: (2024)
Ref_id:b90 Title: Shiming Xiang, and Chunhong Pan Year: (2025)
Ref_id:b91 Title: Open-Sora: Democratizing efficient video production for all Year: (2024-03)
