Title: MoCha: Towards Movie-Grade Talking Character Generation
Abstract: Camera Movement Control"Two distinct streams of tears trail down her cheeks as she speaks with an angry expression…" "A medium shot of a man interacting warmly with an elephant. the man talks to the camera…" "A tilt up shot of a man standing in a dimly lit room, speaking to the camera…" Action Control Multi-Character Turn-based Talk Talking Character "Close-up shot of a doctor in a white lab coat over blue scrubs, speaking…" Emotion Control Figure 1: MoCha is an end-to-end dialogue-centric video generation model that takes only speech and text as input, without requiring any auxiliary conditions.

Section: Introduction
Automating film production holds immense commercial potential, promising to democratize cinematic-level storytelling by enabling content creators to effortlessly generate films through natural language [1][2][3][4]. In film, dialogue plays a central role in conveying narratives. Ideally, creators should be able to specify rich storylines involving multiple characters-whether realistic humans or stylized cartoons-that engage in meaningful dialogue, express emotions, and perform full-body actions. Such talking characters serve as powerful mediums for delivering impactful messages, communicating ideas, and engaging audiences. Beyond film, they also enable a wide range of downstream applications, including digital assistants, virtual avatars, advertising, and educational content.
Despite impressive progress in video generation, current video foundation models are primarily designed for narration-style, non-dialogue scenes. Models such as SoRA, Pika, Luma, Hailuo, and Kling [5][6][7][8][9][10][11][12] produce characters with arbitrary lip movements and disconnected facial expressions, lacking control over actual speech content. As a result, these characters appear lifeless and fail to deliver messeage to the audience, limiting the models' applicability in real-life cinematic production.
Meanwhile, speech-conditioned video generation is still in its infancy, primarily focused on simplified talking-head scenarios. Models such as Loopy, Hallo3, and EMO [13][14][15][16][17][18][19] are limited to cropped face regions with static cameras, ignoring essential elements such as full-body actions, camera motion, and multi-character interactions. These limitations hinder their characters' expressiveness and make them unsuitable for realistic, engaging storytelling.
To bridge the gap between non-dialogue video generation and constrained talking-head synthesis, we introduce the novel task of Talking Characters, which directly targets the goal of automating dialogue-centric film production. The task is defined as generating lifelike digital characters from natural language and speech inputs that express synchronized speech, realistic emotions, and full-body actions under dynamic camera movements (see section 2). To tackle this task, we propose MoCha, the first end-to-end Diffusion Transformer (DiT) model designed to produce high-quality, movie-grade talking character videos. MoCha demonstrates compelling storytelling capabilities. As shown on the project website ahttps://congwei1230.github.io/MoCha/, we lightly edited MoCha-generated clips into a 1-minute, emotionally engaging narrative, illustrating its potential for real-world filmmaking.
MoCha introduces several key technical innovations tailored for this task:
• First End-to-End DiT Without Auxiliary Control Signals: MoCha is the first DiT-based model to demonstrate that high-quality lip synchronization and natural character motion can be achieved using only text and speech-without relying on external control signals such as reference images, pose skeletons, or facial keypoints [14,15,20,13,16,21,17]. Contrary to the dominant belief that audio alone is insufficient, we show that strong audio-visual alignment can emerge purely from end-to-end training.
• Localized Audio Attention: We propose a attention mechanism tailored for DiT-based dialogue-diven video generation, which addresses the temporal mismatch between compressed video tokens and high-resolution audio inputs (see Sec. 3.2). This design significantly improves lip-sync accuracy and speech-video alignment.
Talking Characters differs from conventional talking-head generation-which is restricted to a single face, fixed camera, and square crop-by enabling full-body character synthesis across a range of shot sizes (e.g., close-up, medium, wide) with dynamic camera motion. It supports the generation of one or more characters situated within a contextually appropriate scene.
The task is formally defined by the input-output specification and evaluation protocol, in Table 1.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b13', 'b14', 'b19', 'b12', 'b15', 'b20', 'b16']

Section: Aspect Description Input
Text Prompt y: Natural language description including (1)environment, (2)character appearance and (3)actions, (4)facing direction and emotion, (5)frame position, (6)camera movement, and (7)shot size. Speech Audio a ∈ R L : Raw waveform signal that drives characters' lip movements, facial expressions, and body motions.
Output Video ν ∈ R T ×H×W ×3 : A rendered video of one or more talking characters (human, 3D cartoon, or animal), where T is the number of frames and (H, W ) is the spatial resolution.
this section cite: []

Section: Eval
The generated characters are expected to perform well across the following five axes:
(1) Lip-Sync Quality: Accurate and temporally aligned lip movements with respect to α.
(2) Expression Naturalness: Expressive and coherent facial emotions that align with both y and a.
(3) Action Naturalness: Realistic body gestures reflect described actions in y, synchronized with a.
(4) Text Alignment: Coherence between visual content and y, including Visual layout, character appearance, and camera motion.
(5) Visual Quality: High fidelity and temporal consistency, free from visual artifacts.
Table 1: Task definition of Talking Characters: formalized inputs (y, a), output ν, and evaluation criteria.
this section cite: []

Section: Model: MoCha
In this section, we introduce the MoCha model, the first model to generate talking characters. We begin by outlining its architecture in subsection 3.1, followed by the localized audio attention mechanism in subsection 3.2. Next, we describe the method of generating multiple clips in subsection 3.3. Finally we provide explanation of the training strategy in subsection 3.4.
this section cite: []

Section: Window Cross Attention

this section cite: []

Section: DiT block
Cross
this section cite: []

Section: Audio + Text to Video Diffusion Transformers
Figure 2 presents the overall architecture of MoCha. Unlike prior works that employ text-to-image (T2I) U-Net [14,15,17,22] for talking head generation, MoCha is a diffusion transformer (DiT) [23].
this section cite: ['b13', 'b14', 'b16', 'b21', 'b22']

Section: Model Architecture.
MoCha adopts a fully tokenized design, where both text and speech inputs are projected into token sequences and integrated with video tokens through cross-attention. Given an video ν ∈ R T ×H×W ×3 with T frames, we encode it into a latent representation x 0 ∈ R τ ×h×w×c using a 3D VAE, which down-samples the video spatially and temporally. We define the temporal down-sampling ratio as r = T τ . Next, x 0 is flattened into a sequence of tokens of size (τ × h × w) × c and passed to the DiT model f θ (•). Within each DiT block, the model first applies self-attention to the video tokens x 0 , followed by sequential cross-attention with the text condition tokens y and audio condition tokens α. The audio tokens α ∈ R T ×c is derived from raw waveforms a using Wav2Vec2 [24] and processed through MLPs to align its feature dimension with the video tokens.
Training Objective. We adopt Flow Matching [25], which enables efficient simulation of continuous-time dynamics, to train our model. Given a latent video representation x 1 ∈ R τ ×h×w×c (encoded from the input video), random noise ϵ ∼ N (0, I), and a continuous time step t ∈ [0, 1], we construct an intermediate latent x t by interpolating between ϵ and x 1 :
x t = (1 -t) ϵ + t x 1 .
(1) The model is trained to predict the velocity, defined as the difference between the data and noise:
v t = dx t dt = x 1 -ϵ.(2)
The training loss is then:
L = E ϵ∼N (0,I), x1, y, α, t∈[0,1] f θ x t , y, α, t -(x 1 -ϵ) 2 2 ,(3)
where x 1 is the latent video, y and α are text and audio conditions, and f θ (•) is the DiT model. Unlike prior works [10,14,15,20,13,16,21,17], MoCha does not rely on auxiliary objectives such as face or body masking. Instead, it learns speech-video correlations purely from data, using a fully tokenized and end-to-end training approach.
this section cite: ['b23', 'b24', 'b9', 'b13', 'b14', 'b19', 'b12', 'b15', 'b20', 'b16']

Section: Localized Audio Attention
Most existing talking head generation models rely on 2D diffusion architectures (e.g., U-Net) that generate T frames autoregressively, with each frame ν i conditioned directly on its corresponding audio token α i ∈ R c . This one-to-one mapping between audio and video timesteps naturally ensures tight synchronization. In contrast, diffusion transformer (DiT) architectures depart from this design in two critical ways that complicate temporal alignment:
Audio Token (Key/ Value) α 3D VAE f r a m e 1 Noisy Video ν ∈ ℝ 12×h×w×3 f r a m e 5 f r a m e 9 f r a m e 1 Noisy Video Token (Query) x f r a m e 5 f r a m e 9 α 1 x 1 x 2 x 3 f r a m e 12 Wav2Vec2 + Projector Temp Down r=4 α 4 α 5 α 8 α 9 α 12 f r a m e 12 1. Temporal Resolution Mismatch: DiT-based models operate on latent representations produced by a 3D VAE, which temporally compresses videos by a factor of r (commonly r = 4 or 8 in recent T2V models [8,10]). As a result, video tokens span only τ = T /r steps, while audio remains at the original resolution T , eliminating direct alignment. 2. Fully Parallel Decoding: Unlike autoregressive designs, DiT generates all τ latent frames in parallel. Without constraints, naïve cross-attention permits each video token to attend globally to the full audio sequence, which can lead to incorrect associations.
To address these challenges, we propose a Localized Audio Attention mechanism that introduces a temporal inductive bias. Inspired by the observation that lip movements depend on short-term phonetic patterns while body gestures and expressions reflect longer-term textual semantics, we constrain each video token's attention to a limited audio span.
Specifically, for a latent video frame x (i) ∈ R h×w×c at timestep i ∈ {1, . . . , τ }, we restrict its cross-attention to audio tokens α (j) in the window:
j ∈ [max(1, (i-1)r-1), min(T, ir+1)] . (4
)
This r+2-token window corresponds to the uncompressed temporal segment that x (i) summarizes, with one token of padding on each side for context smoothing. This simple yet effective constraint encourages alignment between speech and video while preserving local continuity across frames.
this section cite: ['b7', 'b9']

Section: Multi-character Turn-based Conversation
Thanks to the clean and fully tokenized design of MoCha (see subsection 3.1), the model supports multi-clip video generation in exactly the same way as single-clip generation-without any additional architectural modifications. As illustrated in Figure 4, unlike video extension methods that rely on autoregressive generation conditioned on previously generated content, MoCha generates all clips in parallel. It leverages self-attention across video tokens to maintain character consistency across clips and preserve coherence in the surrounding environment.
Assuming only one character speaks at a time, we observe that speaker changes in the audio implicitly guide MoCha to transition between clips-without requiring any explicit indicators such as clip tokens [26]. Simultaneously, the text condition y specifies the content of each clip.
Formally, given an audio sequence α = {α 1 , α 2 , . . . , α T }, where the tokens correspond to two speakers segmented as α = [α 1 , . . . , α rk ] ∥ [α rk+1 , . . . , α T ], with r being the VAE temporal downsampling ratio and k being an integer index. The model generates the latent video sequence in parallel as: x = f θ (y, α), where x = {x 1 , x 2 , . . . , x τ }, τ = T /r. The output sequence x can be segmented into two clips aligned with the respective speaker turns:
x = [x 1 , . . . , x k ] ∥ [x k+1 , . . . , x τ ].
While MoCha supports seamless multi-clip generation, a challenge lies in prompting-ensuring that character attributes are consistently grounded across clips. This becomes especially difficult when characters interact or reappear in different clips. Naive captioning models typically rely on visual descriptions to refer to characters. As a result, they must repeat detailed appearance descriptions each time a character is mentioned, leading to long, redundant, and confusing prompts. For example in Figure 5. This verbosity not only increases the risk of exceeding token limits (e.g., 256 tokens) but also confuses the model during generation-especially in multi-clip scenarios.
" Two video clips. Characters:
-Person1: Woman with short grey hair, medium skin tone, wearing a maroon sweater and apron.
-Person2: Man with a beard, darker skin tone, wearing a black t-shirt and jeans. Projector Text Encoder Window Cross Attention Cross Attention Self Attention Speech x t 0s 5s Person2 Speech 3D VAE x Projector Speech Encoder Video Token Person1 Speech 2s 2s Timeline 0s 5s Clip 2: Person2 Scene Clip 1: Person1 Scene 2s 2s Timeline x 1 Figure 4: Multi-character Conversation and Character Tagging. MoCha supports generating multi-character conversion with scene cuts. We design a specialized prompt template: it first specifies the number of clips, then introduces the characters along with their descriptions and tags. Each clip is subsequently described using only the character tags, simplifying the prompt while preserving clarity. MoCha leverages self-attention across video tokens to ensure character and environment consistency. The audio signal implicitly guides MoCha when the transition between clips happens. "The first video clip captures a girl aged 10-15 in a green dress waving to another girl dressed in a green shirt with braided hair holding a book... another girl dressed in a green shirt in the middle of the frame responds with a smile... Nearby, another girl aged 10-15 in a blue hoodie points toward the whiteboard while looking at the girl dressed in the girl in a green shirt with braided hair. Clip1 Clip2 naïve captioning is long, redundant, and confusing 😊 😔 "Two Video Clip. Characters: (Person1)A girl aged 10-15 in a green dress (Person2)A girl in a green shirt with braided hair (Person3)A girl aged 10-15 in a blue hoodie  We address this by introducing a structured prompt template with fixed keywords and a character tagging mechanism that promotes clarity, compactness, and consistency:
• "Two Video Clip. Characters:" Defines a list of characters, each described by visual attributes and assigned a unique tag [27] (e.g., Person1, Person2).
• "First clip", "Second clip" Each video segment is described using only the defined character tags.
This design significantly reduces redundancy and helps the model reliably associate visual attributes with character actions, even across multiple clips.
this section cite: ['b25', 'b26']

Section: MoCha Training Strategy
We identify two key challenges in training high-quality talking character models:
(i) Data Scarcity and Limited Diversity: Speech-annotated video datasets are relatively scarce and often lack sufficient visual and semantic diversity, making it difficult to directly train a MoCha model on them. (ii) Varying Speech Influence Across Shot Types: The impact of speech on video generation differs between spatial scales: it strongly governs lip and facial movements in close-up shots but has a diminishing influence in medium or wide shots involving full-body motion. Training across all shot types simultaneously may lead to slow convergence. To address both issues, we propose a Curriculum-based Multi-modal Training strategy that integrates modality-aware supervision with progressive visual complexity. Mixed-Modal Sampling. Our data consists of a balanced mix of multimodal and unimodal data: • 80% Multimodal (speech+text): The majority enables fine-grained audiovisual grounding. Stage 0. Stage 1. Text Close-Up Shot (1 Character) 80% Stage 2. Video-Speech Alignment Strong Weak Easy Task Complexity 40% Close-Up Shot (1 Character) Medium Close Shot (1 Character) 40% Stage 3. 20% Close-Up Shot (1 Character) Text 20% 20% Medium Close Shot (1 Character) Medium Shot (1 Character) 40% Text 20% Text 20% Stage 4. 10% Close-Up Shot (1 Character) Text 20% 10% Medium Close Shot (1 Character) 20% Medium Shot (1 Character) 2 Character Multi-Clip Shot 40% Hard 100% Figure 6: Progressive Curriculum for Multimodal Training in MoCha. MoCha is first pretrained on text-only video data to acquire general visual generation capability. The model is then progressively exposed to speech-conditioned data across different task difficulties.
• 20% Unimodal (text-only): Text-only video samples offer broader visual diversity and varied camera movements, helping the model retain strong generalization capabilities. In this setting, the speech embedding is replaced with a zero vector before the audio projector.
Shot-Type-Based Curriculum. We organize training into multiple stages based on shot complexity:
• Stage 0: We pretrain on large-scale text-only datasets to establish strong visual priors.
• Stages 1-N: We begin with close-up shots, which have high speech-visual correlation, and progressively incorporate more challenging scenarios such as medium/wide shots and multi-character scenes. At each stage, we halve the share of easier examples from the previous stage while maintaining the 80%/20% multimodal-unimodal ratio.
This combined strategy allows MoCha to benefit from both abundant text-only data and limited multimodal supervision while progressively mastering harder generation tasks.
this section cite: []

Section: Experiment
In this section, we first describe the details of our model in subsection 4.1. We then introduce MoCha-Bench for Talking Characters task and benchmark MoCha against baseline methods in subsection 4.2, and finally, we present an ablation study to analyze the impact of key design choices in
this section cite: []

Section: Implementation Details
MoCha builds upon a pretrained 30B-parameter MovieGen backbone [8, 10], which we extend for speech-conditioned video generation in a fully tokenized, end-to-end setting. The model is configured to produce 128 frames at 24 frames per second, resulting in 5.3-second video clips. Training data is standardized to a spatial resolution of approximately 720 × 720, with flexible aspect ratios to support various shot types. To enable multimodal learning, we curate two complementary datasets:
(1) a large-scale collection of O(100) million text-captioned videos and (2) a smaller set of O(800) thousand speech-annotated samples [28]. Training is distributed across 64 compute nodes. More details of the implementation and the data processing pipeline are provided in the Appendix.  Table 3: Ablation on MoCha-Bench. Removing Localized Audio Attn. degrades lip-sync, Curriculum Training improves generalization.
this section cite: ['b27']

Section: Evaluation
Baselines. We compare our method with state-of-the-art audio-driven talking face generation models, including SadTalker [29], AniPortrait [30], and Hallo3 [28]. These models generate talking faces conditioned on audio and auxiliary signals, such as the first frame, facial keypoints, or pose skeletons. In contrast, MoCha generates talking characters directly from raw speech and text.
Benchmark. We introduce MoCha-bench, a benchmark tailored for the Talking Character generation task. It contains 200 diverse samples, each comprising a text prompt and corresponding audio clip. The dataset spans various camera shot angles and camera movement-for example, closeup shots emphasize facial expressions and lip-sync, while medium shots highlight hand gestures and body movement. Scenes cover a wide range of human activities and object interactions (e.g., woman holding a coffee cup, professor talking to student), with characters speaking with various emotions and facing directions. All prompts were manually curated and further enriched using the publicly released LLaMA-3 [31] model to enhance expressiveness and diversity.
Qualitative Experiments We present qualitative results of MoCha in Figure 1, demonstrating its ability to generate diverse and realistic human motion while maintaining precise speech synchronization, even during complex actions. Contrary to the traditional view that treats audio as a weak conditioning signal requiring auxiliary supervision (e.g., pose annotations), our results show that strong audio-visual alignment can emerge purely from end-to-end training. More examples can be found in Appendix and the https://congwei1230.github.io/MoCha/.
this section cite: ['b28', 'b29', 'b27', 'b30']

Section: Figure 8 illustrates a side-by-side evaluation of MoCha against baseline methods on MoCha-Bench.
Since MoCha generates video directly from speech and text, while baselines operate in an imageto-video (I2V) setup, we ensure fair comparison by providing each baseline with the first frame generated by MoCha (cropped/resized to focus on the head region as needed ). Additional results are provided in the Appendix. MoCha not only produces lip movements that closely align with the speech-enhancing both articulation and naturalness-but also generates expressive facial emotion that accurately follow the textual prompt. In contrast, SadTalker and AniPortrait exhibit minimal head motion and limited lip synchronization. While Hallo3 achieves mostly consistent lip-syncing, it suffers from inaccurate articulation and erratic head movements.
this section cite: []

Section: Quantitative Experiments
We evaluate video quality using the automatic metrics to measure the lip-sync quality. Table 2 presents a comparison on the MoCha-Bench. Our model achieves the best scores across all lip-sync metrics, demonstrating the effectiveness of MoCha 's end-to-end DiT design. These results further confirm that strong audio-visual alignment can emerge purely from audio conditioning without any auxiliary signal.
this section cite: []

Section: Human Evaluations.
We conduct a comprehensive human evaluation to compare MoCha against baseline methods on the MoCha-Bench dataset. The evaluation is based on five axes tailored for the Talking Characters task (see section 2), with scores ranging from 1 to 4 (see Appendix). Each model output received 5 independent ratings per example, resulting in over 1000 responses per model. As illustrated in Figure 7, MoCha significantly outperforms all baselines across all five axes, with average scores approaching 4-indicating performance that is nearly indistinguishable from real video or cinematic production.
this section cite: []

Section: Ablation Studies
We conduct ablation experiments to assess the individual contributions of MoCha 's core components. Table 3 presents the impact of each component. (i)We disable our localized audio attention mechanism during training, which results in a noticeable drop in Sync-C and increased Sync-D. We also have design variant comparison with RoPE + global audio attention in the Appendix. (ii)We train MoCha exclusively on speech-annotated data. This results in a noticeable drop in lip-sync quality, indicating degraded generalization due to the reduced diversity of the dataset.
this section cite: []

Section: Related Work

this section cite: []

Section: Talking Head Generation
Given an audio sequence and a reference face, pioneer talking-head generation works typically utilize biometric signals such as facial keypoints [32][33][34][35], or 3D priors [36-40, 29, 41] as intermediate motion representation to animate the reference face while ensuring lip synchronization. For example, SadTalker [29] first extracts 3DMM coefficients from audio and then renders the face in a 3D-aware manner. AniPortrait [30] predicts 2D facial landmarks from audio and then uses diffusion models to generate a portrait video from the 2D landmark maps. VLOGGER [42] predicts both 3D expression coefficient and 3D body pose from speech and enables the simultaneous generation of talking-face animations and upper-body gestures. Although effective, videos generated by these methods often lack expressiveness and naturalness due to the limited representation of 2D/3D priors. Recently works, such as EMO [18] and Hallo [17], generate audio-driven portrait videos end-to-end using diffusion models, which eliminate intermediate facial representations and learn natural motion from data [18,17,22,16,43]. Hallo3 [28] builds upon pretrained transformer-based video diffusion models to animate faces with dynamic head poses and background elements. Although these methods can generate natural expressions, they rely on complex auxiliary signals-such as reference images or keypoints-which not only limit the naturalness and flexibility of facial expressions and body movements but also limit the generalization ablity of those methods.
this section cite: ['b31', 'b32', 'b33', 'b34', 'b28', 'b29', 'b41', 'b17', 'b16', 'b17', 'b16', 'b21', 'b15', 'b42', 'b27']

Section: Video Diffusion Model
Recent diffusion-based video models have focused on improving visual quality and temporal coherence, particularly in text-guided synthesis. Early works such as Make-A-Video [44], Tune-A-Video [45], Video LDM [46], MagicVideo [47], and AnimateDiff [48] adapt text-to-image (T2I) backbones to model motion dynamics. More recent methods based on diffusion transformers-such as HunyuanVideo [10], CogVideoX [7], MovieGen [8]-along with open-source frameworks like VideoCrafter [6], ModelScopeT2V [49], and Pyramidal Flow [50], further enhance spatio-temporal consistency and generation fidelity.
However, these models lack mechanisms for aligning speech with character behavior, often producing disjointed lip motions or gestures that fail to reflect spoken content. In contrast, MoCha pioneers a new direction by jointly conditioning on both speech and text to drive character animation-bridging the gap between realistic motion synthesis and dialogue-driven storytelling.
this section cite: ['b43', 'b44', 'b45', 'b46', 'b47', 'b9', 'b6', 'b7', 'b5', 'b48', 'b49']

Section: Conclusion
In summary, our work pioneers the task of Talking Characters Generation, pushing beyond traditional talking head synthesis to enable full-body, multi-character animations directly driven by speech and text. We present MoCha, the first framework to address this challenging task, introducing key innovations such as the localized audio attention mechanism for precise audio-visual alignment and a curriculum-based multi-modal training strategy that leverages both speech-and text-labelled data for enhanced generalization. Additionally, our structured prompt design unlocks multi-character, turn-based dialogues with contextual awareness. Comprehensive experiments and human evaluations demonstrate that MoCha delivers state-of-the-art performance in dialogue-driven video generation, marking a significant step toward scalable, cinematic AI storytelling.
this section cite: []

Section: References
Ref_id:b0 Title: Storydiffusion: Consistent self-attention for long-range image and video generation Year: (2024)
Ref_id:b1 Title: Mocogan: Decomposing motion and content for video generation Year: (2018)
Ref_id:b2 Title: Video diffusion models Year: (2022)
Ref_id:b3 Title: Videogpt: Video generation using vq-vae and transformers Year: (2021)
Ref_id:b4 Title: Goku: Flow based video generative foundation models Year: (2025)
Ref_id:b5 Title: Videocrafter1: Open diffusion models for high-quality video generation Year: (2023)
Ref_id:b6 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2024)
Ref_id:b7 Title: Movie gen: A cast of media foundation models Year: (2024)
Ref_id:b8 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b9 Title: A systematic framework for large video generative models Year: (2024)
Ref_id:b10 Title: Videojam: Joint appearance-motion representations for enhanced motion generation in video models Year: (2025)
Ref_id:b11 Title: Video generation models as world simulators Year: (2024)
Ref_id:b12 Title: Towards striking, simplified, and semi-body human animation Year: (2024)
Ref_id:b13 Title: Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions Year: (2024)
Ref_id:b14 Title: Emo2: End-effector guided audio-driven avatar video generation Year: (2025)
Ref_id:b15 Title: Loopy: Taming audio-driven portrait avatar with long-term motion dependency Year: (2025)
Ref_id:b16 Title: Hierarchical audio-driven visual synthesis for portrait image animation Year: (2024)
Ref_id:b17 Title: Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions Year: (2024)
Ref_id:b18 Title: Generating infinite talking videos with your words and voice Year: (2025)
Ref_id:b19 Title: Shifting focus to global audio perception in portrait animation Year: (2024)
Ref_id:b20 Title: Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models Year: (2025)
Ref_id:b21 Title: Hallo2: Long-duration and high-resolution audio-driven portrait image animation Year: (2025)
Ref_id:b22 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b23 Title: wav2vec 2.0: A framework for self-supervised learning of speech representations Year: (2020)
Ref_id:b24 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b25 Title: Mind the time: Temporally-controlled multi-event video generation Year: (2025)
Ref_id:b26 Title: Movie weaver: Tuning-free multi-concept video personalization with anchored prompts Year: (2025)
Ref_id:b27 Title: Hallo3: Highly dynamic and realistic portrait image animation with diffusion transformer networks Year: (2024)
Ref_id:b28 Title: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation Year: (2023)
Ref_id:b29 Title: Aniportrait: Audio-driven synthesis of photorealistic portrait animation Year: (2024)
Ref_id:b30 Title: The llama 3 herd of models Year: (2024)
Ref_id:b31 Title: A lip sync expert is all you need for speech to lip generation in the wild Year: (2020)
Ref_id:b32 Title: Difftalk: Crafting diffusion models for generalized audio-driven portraits animation Year: (1982)
Ref_id:b33 Title: Moda: Mapping-once audiodriven portrait animation with dual attentions Year: (2023)
Ref_id:b34 Title: Identity-preserving talking face generation with landmark and appearance priors Year: (2023)
Ref_id:b35 Title: Headgan: One-shot neural head synthesis and editing Year: (2021)
Ref_id:b36 Title: Dreamtalk: When expressive talking head generation meets diffusion probabilistic models Year: (2023)
Ref_id:b37 Title: Audio-driven emotional video portraits Year: (2021)
Ref_id:b38 Title: Vividtalk: One-shot audio-driven talking head generation based on 3d hybrid prior Year: (2023)
Ref_id:b39 Title: Efficient emotional adaptation for audio-driven talking-head generation Year: (2023)
Ref_id:b40 Title: Real3d-portrait: One-shot realistic 3d talking portrait synthesis Year: (2024)
Ref_id:b41 Title: Multimodal diffusion for embodied avatar synthesis Year: (2024)
Ref_id:b42 Title: Conditional dropout for progressive training of portrait video generation Year: (2024)
Ref_id:b43 Title: Make-a-video: Text-to-video generation without text-video data Year: (2022)
Ref_id:b44 Title: Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation Year: (2022)
Ref_id:b45 Title: Align your latents: High-resolution video synthesis with latent diffusion models Year: (2023)
Ref_id:b46 Title: Efficient video generation with latent diffusion models Year: (2022)
Ref_id:b47 Title: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning Year: (2023)
Ref_id:b48 Title: Modelscope text-to-video technical report Year: (2023)
Ref_id:b49 Title: Pyramidal flow matching for efficient video generative modeling Year: (2024)
Ref_id:b50 Title:  Year: ()
Ref_id:b51 Title: wav2vec: Unsupervised pre-training for speech recognition Year: (2019)
Ref_id:b52 Title: Introducing meta llama 3: The most capable openly available llm to date Year: (2024)
