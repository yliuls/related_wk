Title: OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers
Abstract: Lip synchronization is the task of aligning a speaker's lip movements in video with corresponding speech audio, and it is essential for creating realistic, expressive video content. However, existing methods often rely on reference frames and masked-frame inpainting, which limit their robustness to identity consistency, pose variations, facial occlusions, and stylized content. In addition, since audio signals provide weaker conditioning than visual cues, lip shape leakage from the original video will affect lip sync quality. In this paper, we present OmniSync, a universal lip synchronization framework for diverse visual scenarios. Our approach introduces a mask-free training paradigm using Diffusion Transformer models for direct frame editing without explicit masks, enabling unlimited-duration inference while maintaining natural facial dynamics and preserving character identity. During inference, we propose a flow-matching-based progressive noise initialization to ensure pose and identity consistency, while allowing precise mouth-region editing. To address the weak conditioning signal of audio, we develop a Dynamic Spatiotemporal Classifier-Free Guidance (DS-CFG) mechanism that adaptively adjusts guidance strength over time and space. We also establish the AIGC-LipSync Benchmark, the first evaluation suite for lip synchronization in diverse AI-generated videos.* Work done during an internship at Kling Team, Kuaishou Technology. 39th Conference on Neural Information Processing Systems (NeurIPS 2025).Extensive experiments demonstrate that OmniSync significantly outperforms prior methods in both visual quality and lip sync accuracy, achieving superior results in both real-world and AI-generated videos.

Section: Introduction
Lip synchronization, matching mouth movements with speech audio, is essential for creating compelling visual content across film dubbing [51], digital avatars [32,31,55,5], and teleconferencing [24,28,53]. With the rise of AI-generated content, this technology has evolved from a specialized technique to a fundamental aspect of the video generation landscape [47,34,20]. Despite significant advances in text-to-video (T2V) models [4,2,48,16,39] creating increasingly realistic footage, achieving precise and natural lip synchronization remains an unsolved challenge.
Traditional lip synchronization approaches rely heavily on reference frames combined with maskedframe inpainting [33,51,11,10]. This methodology extracts appearance information from reference frames to inpaint masked regions in target frames-a process that introduces several critical limitations. These methods struggle with head pose variations, identity preservation, and artifact elimination, especially when target poses differ significantly from references [30,29].
Furthermore, the dependence on explicit masks cannot fully prevent unwanted lip shape leakage, compromising synchronization quality and restricting applicability across diverse visual representations [1]. The challenges intensify in the context of audio-driven generation. Unlike strong visual cues, audio signals provide relatively weak conditioning, making precise lip synchronization difficult [40]. Additionally, existing methods rely on face detection and alignment [3] techniques that break down when applied to stylized characters and non-human entities, precisely the diverse content that modern text-to-video models excel at generating. This technical gap is compounded by the absence of standardized evaluation frameworks for lip sync in stylized videos. Current benchmarks [52,42] focus almost exclusively on photorealistic human faces in controlled settings, failing to capture the visual diversity inherent in AI-generated videos.
To address these challenges, we introduce OmniSync, a universal lip synchronization framework designed for diverse videos. Our approach eliminates reliance on reference frames and explicit masks through a diffusion-based direct video editing paradigm. In addition, we establish AIGC-LipSync Benchmark, the first comprehensive evaluation framework for lip synchronization across diverse AIGC contexts. OmniSync's technical approach is built upon three key innovations: First, we implement a mask-free training paradigm using Diffusion Transformers (DiT) [26] for direct cross-frame editing. Our model learns a mapping function (V cd , A ab ) → V ab , where V represents video frames and A represents audio. The indices (a : b, c : d) represent different segments sampled from the same video. The model modifies only speech-relevant regions according to target audio without requiring explicit masks or references. This approach enables unlimited-duration inference while maintaining natural facial dynamics and preserving character identity.
Second, we introduce a flow-matching-based progressive noise initialization strategy during inference. Rather than beginning with random noise [38], we inject controlled noise into original frames using Flow Matching [19], then execute only the final denoising steps. This approach maintains spatial consistency between source and generated frames while allowing sufficient flexibility for precise mouth region modifications, effectively mitigating pose inconsistency and identity drift.
Third, we develop a dynamic spatiotemporal Classifier-Free Guidance (CFG) framework [13] that provides fine-grained control over the generation process. By adaptively adjusting guidance strength across both temporal and spatial dimensions: temporally reducing guidance strength as denoising progresses, and spatially applying Gaussian-weighted control centered on mouth-relevant regions. This balanced approach ensures precise lip synchronization without disturbing unrelated areas.
Our contributions can be summarized as follows:
• A universal lip synchronization framework that eliminates reliance on reference frames and explicit masks, enabling accurate speech synchronization across diverse visual representations.
• A flow-matching-based progressive noise initialization strategy during inference, effectively stabilizing the early denoising process and mitigating pose inconsistency and identity drift.
• A dynamic spatiotemporal CFG framework that provides fine-grained control over audio influence, addressing the weak signal problem in audio-driven generation.
• A comprehensive AIGC-LipSync Benchmark for evaluating lip synchronization in AIgenerated content, including stylized characters and non-human entities.
2 Related Work
this section cite: ['b50', 'b31', 'b30', 'b54', 'b4', 'b23', 'b27', 'b52', 'b46', 'b33', 'b19', 'b3', 'b1', 'b47', 'b15', 'b38', 'b32', 'b50', 'b10', 'b9', 'b29', 'b28', 'b0', 'b39', 'b2', 'b51', 'b41', 'b25', 'b37', 'b18', 'b12']

Section: Audio-driven Lip Synchronization

this section cite: []

Section: GAN-based Lip Synchronization.
Traditional GAN-based [9] methods [33,41,7,23,12] have established important foundations in lip synchronization. Wav2Lip [33] pioneered the use of pretrained SyncNet to supervise generator training, setting a benchmark for subsequent research. DINet [51] enhanced synchronization quality by performing spatial deformation on reference image feature maps, better preserving high-frequency details. IP-LAP [54] introduced a two-stage approach that first infers landmarks from audio before rendering them into facial images. ReSyncer [10] incorporated 3D mesh priors for facial motion, effectively reducing artifacts.
this section cite: ['b8', 'b32', 'b40', 'b6', 'b22', 'b11', 'b32', 'b50', 'b53', 'b9']

Section: Diffusion-based Lip Synchronization.
Recent advances in diffusion models [25,50,17,22] have enabled significant progress in audio-driven lip synchronization. LatentSync [17] represents an end-to-end framework based on audio-conditioned latent diffusion models without intermediate motion representation. SayAnything [22] employs a denoising UNet architecture that processes video latents with audio conditioning. MuseTalk [50] proposes a novel sampling strategy that selects reference images with head poses closely matching the target.
However, these methods still rely on reference frames combined with masked-frame inpainting, leading to head pose limitations, identity preservation issues, and blurry edge generation. Our OmniSync framework addresses these limitations through a mask-free training paradigm that enables application across diverse visual representations.
this section cite: ['b24', 'b49', 'b16', 'b21', 'b16', 'b21', 'b49']

Section: Audio-driven Portrait Animation
Audio-driven portrait animation [38,46,15,6,14,27,37,49] differs fundamentally from lip sync. Portrait animation [45,8] follows an image-to-video framework without constraints on head poses or facial expressions, eliminating the need to integrate generated content back into original video. This approach is unsuitable for post-generation lip synchronization in video generation pipelines. In contrast, lip synchronization [33,17] operates within a video-to-video framework, modifying only lip movements while maintaining compatibility with existing footage. This represents a more constrained task, requiring precise modification of lip regions while preserving surrounding facial features.
Recent models like OmniHuman-1 [18] and Mocha [44] use audio directly as a conditioning signal for image-to-video or text-to-video frameworks. However, due to limitations in talking head datasets, their generative capabilities don't match the versatility of advanced video generation models. This gap highlights why specialized lip synchronization for AI-generated videos remains critical.
this section cite: ['b37', 'b45', 'b14', 'b5', 'b13', 'b26', 'b36', 'b48', 'b44', 'b7', 'b32', 'b16', 'b17', 'b43']

Section: Method

this section cite: []

Section: Overview
In this section, we present OmniSync, a universal lip synchronization framework designed for diverse visual content (Fig. 2). Our approach comprises three key components: 1) a mask-free training paradigm that eliminates dependency on reference frames and explicit masks, 2) a flow-matchingbased progressive noise initialization strategy for enhanced inference stability, and 3) dynamic spatiotemporal Classifier-Free Guidance (CFG) that optimizes lip sync while preserving facial details.
The following subsections provide comprehensive explanations of each component.
this section cite: []

Section: Mask-Free Training Paradigm
Traditional lip synchronization methods [33,50] rely on masked-frame inpainting, isolating the mouth region before generating content based on audio input. Despite their prevalence, these approaches
this section cite: ['b32', 'b49']

Section: Self Attention Temporal Attention

this section cite: []

Section: Audio Cross Attention

this section cite: []

Section: FFN
DiT Block × N
this section cite: []

Section: Text Cross Attention
"A girl with clear facial and tooth movements."
this section cite: []

Section: Text Encoder

this section cite: []

Section: 3D VAE Decoder
Flow Matching Noise Generator
this section cite: []

Section: OmniSync

this section cite: []

Section: Progressive Noise Initialization

this section cite: []

Section: Diffusion Timestep
Audio Encoder
Text Features Audio Features Video Description Early Middle Late Text Features Audio Features Dynamic Spatiotemporal Classifier-Free Guidance Timestep-Dependent Sampling Strategy 1000 999 998 997 •••• 600 ••• 500 ••• 400 •••• 3 2 1 0 Focus on: Pose & Identity Pseudo-Paired Dataset Arbitrary Training Dataset
this section cite: []

Section: Inference Stage
Video Frames Audio Input:
Target:
Input:
this section cite: []

Section: Diffusion Timestep

this section cite: []

Section: Early Stage Middle Stage Late Stage
Focus on: Lip Shape An alternative approach is direct frame editing, which aims to transform frames according to target audio without relying on masks or references. However, this approach requires perfectly paired training data with identical head poses and identity-differing only in lip movements. Such paired data is extremely rare and would severely restrict the model's generalizability to diverse visual results.
To address these limitations, we leverage the progressive denoising characteristic of diffusion models, introducing a novel training strategy that varies data sampling based on diffusion timesteps. This allows for stable learning without requiring perfectly paired examples. Our goal is to learn a conditional generation process mapping (V cd , A ab ) → V ab through iterative denoising, where V represents video frames and A represents audio.
We employ Flow Matching [19] as our training objective. Given an input video segment V cd from frames c to d, and a target audio segment A ab from frames a to b, our model generates the corresponding video segment V ab via the diffusion process:
x t-1 = DiT(x t , V cd , A ab , t),(1)
where x t represents the noised version of target video V ab at timestep t, and DiT denotes our diffusion transformer, which predicts the denoised state at timestep t -1.
The CFM loss used for training is defined as:
L CF M (θ) = E t,xt,V cd ,A ab ,V ab ∥v θ (x t , V cd , A ab , t) -u t (x t |V ab )∥ 2 2 ,(2)
where v θ (x t , V cd , A ab , t) is the learned velocity field predicted by DiT with conditioning on input video V cd and target audio A ab , u t (x t |V ab ) is the conditional velocity field typically defined as u t (x t |V ab ) = (V ab -x t )/(1 -t) for the linear interpolation path x t = (1 -t)x 0 + tV ab .
this section cite: ['b18']

Section: Timestep-Dependent Sampling Strategy.
A critical insight in our approach is recognizing that the diffusion process can be decomposed into distinct phases, each with different learning requirements. Specifically, early timesteps focus on generating fundamental facial structure, including pose and identity information; middle timesteps primarily generate lip movements driven by audio; while late timesteps refine identity details and textures. To capitalize on this natural progression, we utilize different datasets for distinct timesteps.
For early timesteps (approximately t ≈ T ), responsible for generating overall facial structure, we employ pseudo-paired data from controlled laboratory settings. These samples maintain nearly identical pose information, with variations only in lip movements, providing stable learning signals for structural features and ensuring pose alignment between input and output.
For middle and late timesteps, we transition to more diverse data, sampling from arbitrary videos. During middle timesteps, the model learns to generate lip shapes guided by audio input, whereas in late timesteps (approximately t ≈ 0), it focuses on refining identity details and ensuring texture consistency. This timestep-dependent training strategy can be formalized as:
p(V cd , V ab |t) = p pseudo-paired (V cd , V ab ) if t > t threshold , p arbitrary (V cd , V ab ) otherwise.(3)
Here, p pseudo-paired indicates sampling from controlled datasets with minimal pose variations, while p arbitrary signifies sampling from our diverse collection of videos. The conditional generation process can be expressed mathematically as:
p θ (V ab |V cd , A ab ) = p θ (V ab |x 0 )p θ (x 0 |V cd , A ab )dx 0 ,(4)
where p θ (V ab |x 0 ) represents the mapping from the fully denoised state to the output video, and p θ (x 0 |V cd , A ab ) captures the relationship between input conditions and the denoised state. Here, x 0 refers to the completely denoised latent representation (at timestep t = 0).
This progressive training approach aligns well with the natural learning progression of diffusion models. By providing appropriate training signals at each stage, we enable stable learning even without perfectly paired data, allowing our model to generalize effectively to diverse real-world scenarios while maintaining identity consistency.
this section cite: []

Section: Progressive Noise Initialization
Standard diffusion-based generation [38] typically begins from random noise (timestep T ) and progressively denoises toward the final output (timestep 0). However, this approach often results in subtle but noticeable pose misalignments between generated content and original video frames, creating undesirable boundary artifacts and compromising identity preservation.
The fundamental issue lies in error accumulation during the early stages of diffusion. Even minor deviations in early timesteps-when basic facial structure is being formed-can lead to significant misalignments in the final output. This problem is relevant for lip synchronization, where the goal is to modify only speech-relevant regions while maintaining perfect spatial consistency elsewhere. To address this challenge, we introduce a flow-matching-based progressive noise initialization strategy that transforms the traditional diffusion process.
this section cite: ['b37']

Section: Flow-Matching Noise Initialization.
Rather than starting the diffusion process from random noise at timestep T , we initialize from original video frames with a controlled level of noise. This simulates an intermediate state in the diffusion trajectory, corresponding to a normalized parameter τ . The initialization is performed by adding this controlled noise to the original video frame:
x init = FM add (V source , τ ) = (1 -τ )V source + τ ϵ,(5)
where x init is the initial noised state derived using the parameter τ , V source is the source video frame, and ϵ ∼ N (0, I) is random noise. Let t start be the discrete timestep corresponding to this initialization point (T is the total number of diffusion steps, and τ ∈ [0, 1]).
This initialization strategy provides two significant advantages. First, it bypasses the early stages of diffusion (from T down to t start ) where general facial structure is formed. This ensures that head pose and global structure are directly inherited from the source frame. Second, it reduces computational requirements by performing denoising only for the remaining steps, from t start down to 0.
The complete progressive denoising process can be expressed as:
x t = x init if t = t start , DiT(x t+1 , V source , A target , t + 1) if t start > t ≥ 0,(6)
where A target is the target audio used to guide the denoising process, and t here represents discrete diffusion timesteps.
This approach effectively creates a two-stage process: (1) initialization using the flow-matchinginspired noise addition (Eq. 5) to reach a state equivalent to timestep t start , and (2) guided denoising from t start to 0 that focuses on modifying mouth regions according to the target audio while preserving the overall facial structure, identity features, and head pose from the source frame. By skipping the early noisy stages where basic structures form, we maintain spatial consistency while allowing sufficient flexibility for precise mouth region modifications.
this section cite: []

Section: Dynamic Spatiotemporal Classifier-Free Guidance
Audio-driven lip synchronization faces a fundamental challenge: audio signals provide relatively weak conditioning compared to visual cues [40]. Standard Classifier-Free Guidance (CFG) [13] can enhance audio conditioning, but applying uniform guidance across spatial and temporal dimensions creates an unavoidable dilemma: higher guidance scales produce more accurate lip sync but introduce texture artifacts, while lower scales preserve visual fidelity but yield less precise lip movements.
To resolve this tension, we introduce Dynamic Spatiotemporal Classifier-Free Guidance (DS-CFG), a novel approach that provides fine-grained control over the generation process across both spatial and temporal dimensions. Our method applies varying guidance strengths to different regions of the frame and different stages of the diffusion process, achieving an optimal balance between lip synchronization accuracy and overall visual quality.
this section cite: ['b39', 'b12']

Section: Spatially-Adaptive Guidance.
The key insight for spatial adaptation is that audio information primarily affects the mouth region, while other facial areas should remain largely unchanged. We implement this through a Gaussian-weighted spatial guidance matrix that concentrates guidance strength around speech-relevant regions:
G spatial (x, y) = ω base + (ω peak -ω base ) • exp - (x -x m ) 2 + (y -y m ) 2 2σ 2(7)
where (x m , y m ) represents the mouth center, σ controls the spread of the Gaussian distribution, ω base is the baseline guidance strength applied to non-mouth regions, and ω peak is the peak strength applied at the mouth center. This spatial adaptation ensures that audio conditions strongly influence lip and surrounding regions while minimally affecting other facial features.
Temporally-Adaptive Guidance. We observe that audio conditioning plays different roles at different stages of the diffusion process. In early diffusion timesteps, strong guidance helps establish correct lip shapes, while in later stages, excessive guidance can disrupt fine texture details. [2,43,35] To address this, we implement a temporally decreasing guidance schedule:
ω(t) = ω peak • t T γ (8
)
where t is the current diffusion timestep, T is the total number of timesteps, ω peak is the maximum guidance scale, and γ controls the decay rate, with a value of 1.5. This temporal adaptation ensures strong guidance during early and middle diffusion stages when coarse structures are formed, gradually reducing influence during later stages when fine details and textures are refined.
Unified Dynamic Spatiotemporal CFG. Combining both spatial and temporal adaptations, our DS-CFG approach modifies the standard CFG formulation to:
εθ (x t , c, t) = ϵ θ (x t , ∅, t) + G spatial • ω(t) • (ϵ θ (x t , c, t) -ϵ θ (x t , ∅, t)) (9
)
where ϵ θ (x t , c, t) and ϵ θ (x t , ∅, t) are the noise predictions with and without conditioning, respectively.
Through this DS-CFG, our method achieves precise control over the generation process, effectively addressing the weak audio signal problem in audio-driven generation.
this section cite: ['b1', 'b42', 'b34']

Section: Experiments

this section cite: []

Section: Experimental Settings
Datasets. We trained OmniSync using the MEAD dataset [42] and a 400-hour dataset collected from YouTube. MEAD's controlled laboratory recordings with diverse facial expressions but minimal head movement provided ideal data for training early denoising stages, while the YouTube dataset enhanced generalization across varied real-world conditions for middle and late stages.
To address the limitations of existing benchmarks that focus solely on real-world videos with frontal views and stable lighting, we created the AIGC-LipSync Benchmark. This comprehensive evaluation framework comprises 615 human-centric videos generated by state-of-the-art text-tovideo models such as Kling, Dreamina, Wan [39], and Hunyuan [16]. The benchmark specifically captures challenging visual scenarios such as large facial movements, profile views, variable lighting, occlusions, and stylized characters-conditions that traditional benchmarks fail to address. Details about benchmark construction can be found in the supplementary materials.
this section cite: ['b41', 'b38', 'b15']

Section: Implementation Details.
We implement our OmniSync model using the Diffusion Transformer architecture. The model is trained on a combined dataset for 80,000 steps using AdamW optimizer [21] with a learning rate of 1e-5. Training is completed in 80 hours using 64 NVIDIA A100 GPUs with a batch size of 64. Audio features are extracted via a pre-trained Whisper encoder, and text conditioning utilizes a T5 encoder. Training employs the timestep-dependent sampling threshold t threshold = 850.
The experimental results indicate that excessive thresholds induce significant misalignment while insufficient values will leak the original lip shape. During inference we adopt our flow-matching-based progressive noise initialization starting at τ = 0.92, followed by 50 denoising steps.
this section cite: ['b20']

Section: Quantitative Evaluation
We evaluate OmniSync against state-of-the-art methods including Wav2Lip [33], VideoReTalking [7], TalkLip [41], IP-LAP [54], Diff2Lip [25], MuseTalk [50], and LatentSync [17] using a comprehensive suite of metrics. For visual quality assessment, we employ FID (Fréchet Inception Distance) to measure frame-level fidelity, FVD (Fréchet Video Distance) for temporal consistency, and CSIM (Cosine Similarity) to quantify identity preservation. Perceptual quality is assessed using no-reference metrics including NIQE (Natural Image Quality Evaluator), BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator), and HyperIQA [36]. For audio-visual synchronization, we measure LMD (Landmark Distance) between predicted and ground truth facial landmarks in the mouth region, and LSE-C (Lip Sync Error -Confidence) to evaluate lip synchronization quality. For the AIGC-LipSync benchmark, we report the Generation Success Rate across all 615 videos and specifically for stylized characters. This metric shows the percentage of videos that are successfully synchronized and pass human verification. This evaluation is essential for universal lip synchronization in AI-generated content, where traditional metrics may not fully capture the challenges of stylized characters, extreme poses, and other atypical visual conditions.
The experimental results in Tab. 1 and Tab. 2 demonstrate that our approach achieves superior performance on multiple metrics. On the HDTF dataset, our method reduced FID by 7.8% and FVD by 8.0% compared to LatentSync, with a remarkable 23.2% improvement in BRISQUE over Diff2Lip. For lip synchronization, we achieved the lowest LMD, outperforming IP-LAP by 7.8%, while LatentSync maintained a slight edge in LSE-C due to its SyncNet-based loss constraint.
On the challenging AIGC-LipSync benchmark, OmniSync demonstrated exceptional capabilities with a 30.5% FID reduction and 19.7% FVD reduction compared to LatentSync, alongside improved identity preservation. Most significantly, our method achieved a 97.40% Generation Success Rate across all videos-substantially higher than MuseTalk (92.20%) and other methods (below 75%). For stylized characters, our success rate of 87.78% outperformed MuseTalk (67.78%), demonstrating OmniSync's capability to handle diverse visual representations including stylized characters.
this section cite: ['b32', 'b6', 'b40', 'b53', 'b24', 'b49', 'b16', 'b35']

Section: Qualitative Evaluation
We present qualitative comparisons between OmniSync and existing methods in Fig. 3. Our approach produces more natural facial expressions and superior lip synchronization. Due to lip shape leakage, IP-LAP [54] and LatentSync [17] frequently fail at mouth shape modification, resulting in poor lip synchronization effects. MuseTalk [50] and VideoReTalking [7] modify lip movements but  frequently lose identity and visual quality. Diff2Lip [25] and Wav2Lip [33] commonly exhibit lip sync errors, mouth artifacts, and identity drift, particularly in challenging or stylized cases. In contrast, OmniSync consistently maintains identity details and generates realistic, expressive lip movements, demonstrating robust performance. Our approach effectively balances audio and visual cues, addressing the challenge of weak audio conditioning.
User Study.
To assess perceptual quality, we conducted a user study with 39 participants evaluating 32 video sets generated by OmniSync and seven competing methods, with a standardized Cronbach's α coefficient of 0.98. Participants rated each video on a 5-point Likert scale across five criteria: Lip Sync Accuracy, Character Identity preservation, Timing Stability, Image Quality, and Video Realism. As shown in Tab. 3, OmniSync outperformed all competitors across all metrics, achieving superior scores in Lip Sync Accuracy (3.923 vs. 3.812 for LatentSync), Character Identity (4.128 vs. 3.658), Timing Stability (4.043 vs. 3.581), Image Quality (4.051 vs. 3.632), and Video Realism (3.872 vs. 3.453). These results confirm OmniSync's superior ability to generate high-quality talking videos.
this section cite: ['b53', 'b16', 'b49', 'b6', 'b24', 'b32']

Section: Ablation Study
To clarify the contributions of each core component in our framework, we conduct an ablation study targeting three key modules: the timestep-dependent data sampling strategy, progressive noise initialization, and the Dynamic Spatiotemporal Classifier-Free Guidance (DS-CFG) mechanism. Quantitative results are presented in Tab. 4, and corresponding visual examples are shown in Fig. 4.
Removing the timestep-dependent sampling strategy results in a significant drop in identity preservation and pose consistency, with a 10.7% decrease in CSIM and substantial increases in FID and FVD. As shown in Fig. 4, without this sampling strategy, the generated faces often exhibit clear mismatches with the original image, including noticeable facial misalignment issues. This validates our design choice of aligning pseudo-paired data with early diffusion steps, which proves critical for generating structurally stable outputs. Similarly, removing progressive noise initialization leads to evident temporal inconsistencies and an increase in FVD, confirming the importance of our flow-matching initialization in preserving spatial anchoring and motion coherence.
We also compare our proposed DS-CFG with both low and high static CFG settings. As illustrated in Fig. 4, low CFG provides insufficient audio conditioning, resulting in under-articulated lip movements (LSE-C: 4.16), whereas high CFG improves synchronization (LSE-C: 7.10) but introduces noticeable artifacts and distortions in facial details. In contrast, DS-CFG achieves an optimal balance by applying strong localized guidance in early diffusion stages and gradually reducing it in later steps. These results confirm that dynamic control across temporal and spatial dimensions is essential for producing expressive and visually coherent lip synchronization in generative video content.
this section cite: []

Section: Conclusion
In this paper, we introduce OmniSync, a universal lip synchronization framework for diverse content that addresses critical limitations of traditional approaches. Our three key innovations-a maskfree training paradigm eliminating mask dependencies, a flow-matching-based progressive noise initialization strategy ensuring identity preservation, and dynamic spatiotemporal Classifier-Free Guidance balancing synchronization with visual quality-collectively enable precise lip movements across diverse visual representations. To support systematic evaluation in this field, we establish the AIGC-LipSync Benchmark, the first comprehensive framework for assessing lip synchronization in varied AIGC contexts. Extensive experiments demonstrate OmniSync's superior performance across challenging scenarios, establishing a robust foundation for integrating precise lip synchronization into the broader AI video generation ecosystem.
this section cite: []

Section: References
Ref_id:b0 Title: Keysync: A robust approach for leakage-free lip synchronization in high resolution Year: (2025)
Ref_id:b1 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b2 Title: How far are we from solving the 2d & 3d face alignment problem? Year: (2017)
Ref_id:b3 Title: Videocrafter1: Open diffusion models for high-quality video generation Year: (2023)
Ref_id:b4 Title: Cafe-talk: Generating 3d talking face animation with multimodal coarse-and fine-grained control Year: (2025)
Ref_id:b5 Title: Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions Year: (2025)
Ref_id:b6 Title: Videoretalking: Audio-based lip synchronization for talking head video editing in the wild Year: (2022)
Ref_id:b7 Title: Hallo2: Long-duration and high-resolution audio-driven portrait image animation Year: (2024)
Ref_id:b8 Title: Generative adversarial networks Year: (2020)
Ref_id:b9 Title: Resyncer: Rewiring style-based generator for unified audio-visually synced facial performer Year: (2024)
Ref_id:b10 Title: Stylesync: High-fidelity generalized and personalized lip sync in style-based generator Year: (2023)
Ref_id:b11 Title: Towards generating ultra-high resolution talking-face videos with lip synchronization Year: (2023)
Ref_id:b12 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b13 Title: Shifting focus to global audio perception in portrait animation Year: (2024)
Ref_id:b14 Title: Taming audio-driven portrait avatar with long-term motion dependency Year: (2024)
Ref_id:b15 Title: A systematic framework for large video generative models Year: (2024)
Ref_id:b16 Title: Latentsync: Audio conditioned latent diffusion models for lip sync Year: (2024)
Ref_id:b17 Title: Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models Year: (2025)
Ref_id:b18 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b19 Title: Evalcrafter: Benchmarking and evaluating large video generation models Year: (2024)
Ref_id:b20 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b21 Title: Sayanything: Audio-driven lip synchronization with conditional video diffusion Year: (2025)
Ref_id:b22 Title: Styletalk: One-shot talking head generation with controllable speaking styles Year: (2023)
Ref_id:b23 Title: A comprehensive taxonomy and analysis of talking head synthesis: Techniques for portrait generation, driving mechanisms, and editing Year: (2024)
Ref_id:b24 Title: Diff2lip: Audio conditioned diffusion models for lip-synchronization Year: (2024)
Ref_id:b25 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b26 Title: Temporally-aware precise action control for talking avatars Year: (2025)
Ref_id:b27 Title: Dualtalk: Dual-speaker interaction for 3d talking head conversations Year: (2025)
Ref_id:b28 Title: Synctalk++: High-fidelity and efficient synchronized talking heads synthesis using gaussian splatting Year: (2025)
Ref_id:b29 Title: Synctalk: The devil is in the synchronization for talking head synthesis Year: (2024)
Ref_id:b30 Title: Selftalk: A self-supervised commutative training diagram to comprehend 3d talking faces Year: (2023)
Ref_id:b31 Title: Emotalk: Speech-driven emotional disentanglement for 3d face animation Year: (2023)
Ref_id:b32 Title: A lip sync expert is all you need for speech to lip generation in the wild Year: (2020)
Ref_id:b33 Title: Make-a-video: Text-to-video generation without text-video data Year: (2022)
Ref_id:b34 Title: Diffused heads: Diffusion models beat gans on talking-face generation Year: (2024)
Ref_id:b35 Title: Blindly assess image quality in the wild guided by a self-adaptive hyper network Year: (2020)
Ref_id:b36 Title: Streaming diffusion models for real-time interactive human avatars Year: (2025)
Ref_id:b37 Title: Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions Year: (2024)
Ref_id:b38 Title: Open and advanced large-scale video generative models Year: (2025)
Ref_id:b39 Title: Conditional dropout for progressive training of portrait video generation Year: (2024)
Ref_id:b40 Title: Seeing what you said: Talking face generation guided by a lip reading expert Year: (2023)
Ref_id:b41 Title: Mead: A large-scale audio-visual dataset for emotional talking-face generation Year: (2020)
Ref_id:b42 Title: Analysis of classifier-free guidance weight schedulers Year: (2024)
Ref_id:b43 Title: Towards movie-grade talking character synthesis Year: (2025)
Ref_id:b44 Title: Aniportrait: Audio-driven synthesis of photorealistic portrait animation Year: (2024)
Ref_id:b45 Title: Hierarchical audio-driven visual synthesis for portrait image animation Year: (2024)
Ref_id:b46 Title: Videogpt: Video generation using vq-vae and transformers Year: (2021)
Ref_id:b47 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2024)
Ref_id:b48 Title: Uniavgen: Unified audio and video generation with asymmetric cross-modal interactions Year: (2025)
Ref_id:b49 Title: Real-time high quality lip synchronization with latent space inpainting Year: (2024)
Ref_id:b50 Title: Dinet: Deformation inpainting network for realistic face visually dubbing on high resolution video Year: (2023)
Ref_id:b51 Title: Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset Year: (2021)
Ref_id:b52 Title: Morpheus: A neural-driven animatronic face with hybrid actuation and diverse emotion control Year: (2025)
Ref_id:b53 Title: Identity-preserving talking face generation with landmark and appearance priors Year: (2023)
Ref_id:b54 Title: Meta-learning empowered meta-face: Personalized speaking style adaptation for audio-driven 3d talking face animation Year: (2024)
