Title: INSTILLING AN ACTIVE MIND IN AVATARS VIA COG-NITIVE SIMULATION
Abstract: Current video avatar models can generate fluid animations but struggle to capture a character's authentic essence, primarily synchronizing motion with low-level audio cues instead of understanding higher-level semantics like emotion or intent. To bridge this gap, we propose a novel framework for generating character animations that are not only physically plausible but also semantically rich and expressive. Our model is built on two technical innovations. First, we employ Multimodal Large Language Models to generate a structured textual representation from input conditions, providing high-level semantic guidance for creating contextually and emotionally resonant actions. Second, to ensure robust fusion of multimodal signals, we introduce a specialized Multimodal Diffusion Transformer architecture featuring a novel Pseudo Last Frame design. This allows our model to accurately interpret the joint semantics of audio, images and text, generating motions that are deeply coherent with the overall context. Comprehensive experiments validate the superiority of our method, which achieves compelling results in lip-sync accuracy, video quality, motion naturalness, and semantic consistency. The approach also shows strong generalization to challenging scenarios, including multi-person and non-human subjects. Project page is linked here.

Section: INTRODUCTION
"System 1 operates automatically and quickly, with little or no effort and no sense of voluntary control. System 2 allocates attention to the effortful mental activities that demand it, including complex computations." -Daniel Kahneman, Thinking, Fast and Slow
The field of video avatars (He et al., 2023;Tian et al., 2025c;Xu et al., 2024a;Wang et al., 2024a;Chen et al., 2024c;Xu et al., 2024b;Stypulkowski et al., 2024;Jiang et al., 2025;Lin et al., 2025a;Gan et al., 2025;Kong et al., 2025;Wang et al., 2025b;Hu, 2024b;Lin et al., 2025b;Qiu et al., 2025) aims to synthesize realistic character videos from driving signals, with the goal of creating lifelike digital humans capable of reasoned action and authentic emotion. Recent years have seen rapid progress, evolving from early lip-sync (Jiang et al., 2024;Zhang et al., 2023;Wang et al., 2021;Zhao & Zhang, 2022;Siarohin et al., 2019;2021) and portrait animation (Jiang et al., 2025;Tian et al., 2025c;Zhong et al., 2024;Xu et al., 2024b;a;Cui et al., 2024) to half-body (Lin et al., 2025a;Tian et al., 2025a) and full-body generation (Lin et al., 2025b). As the scope of generation expands, so does the expectation for models to move beyond mere physical likeness and capture a character's authentic essence, their underlying personality, emotion, and intent, as shown in Figure 1.
A recent wave of audio-driven methods based on Diffusion Transformers (DiT) (Peebles & Xie, 2023;Esser et al., 2024b;Seawead et al., 2025;Kong et al., 2024;Wang et al., 2025a) can generate human motion synchronized with audio (Lin et al., 2025b;Kong et al., 2025;Gan et al., 2025;Wang et al., 2025b;Qiu et al., 2025). However, these models typically learn only low-level correlations, resulting in accurate lip movements but simple, repetitive gestures. Their outputs lack the contextual appropriateness and plausibility of authentic human behavior, revealing a significant gap between current capabilities and the goal of creating convincing avatars. We attribute this gap to a failure to model higher-order reasoning. Drawing inspiration from the dual-process theory of human cognition (Kahneman, 2011;Kahneman & Tversky, 2013), which distinguishes between fast, intuitive "System 1" thinking and slow, deliberative "System 2" reasoning, we observe that existing models operate mainly at the level of System 1. They excel at reactive mappings, like audio-to-lipmovement, but cannot perform the goal-oriented, contextual reasoning characteristic of System 2. We argue that the path forward lies in emulating both systems. To this end, we propose leveraging the powerful reasoning capabilities (Wei et al., 2022;Wang et al.;Yao et al., 2023;Schick et al., 2023;Park et al., 2023;Yuan et al., 2024;Wu et al., 2025b) of Multimodal Large Language Models (Team et al., 2023;OpenAI, 2024) to explicitly simulate the deliberative processes of System 2.
However, integrating MLLM-generated textual guidance into an avatar framework is non-trivial. The text, which articulates high-level reasoning, introduces a new modality that can conflict with existing signals like audio (for rhythm) and a reference image (for identity). Naive fusion can lead to modal interference, where, for example, low-level audio cues might disrupt high-level, text-guided semantic actions, and reference image conditioning could alter their motion magnitude. Therefore, a novel architecture is required to effectively manage these interdependencies and mitigate conflicts, enabling the simultaneous simulation of both System 1 and System 2.
Motivated by this analysis, we propose a Multimodal DiT framework with two key designs. First, an MLLM-powered agent generates a high-level, logically coherent "System 2" signal by reasoning over multimodal inputs. * Second, we introduce a specialized MMDiT architecture and training methodology to fuse these inputs and mitigate interference. This architecture incorporates a novel pseudo-last-frame strategy that preserves identity without constraining dynamic, content-driven motion, by leveraging the model's temporal extrapolation capabilities to maintain identity, instead of directly conditioning on the reference image during training. Our main contributions are as follows:
A New Perspective on Avatar Modeling. We are the first to frame the video avatar problem through the cognitive science lens of System 1 and System 2, identifying the limitations of current models and proposing a holistic approach that models both.
this section cite: ['b20', 'b65', 'b32', 'b15', 'b39', 'b55', 'b31', 'b97', 'b79', 'b99', 'b62', 'b32', 'b101', 'b10', 'b53', 'b13', 'b59', 'b38', 'b39', 'b15', 'b55', 'b33', 'b34', 'b84', 'b93', 'b58', 'b52', 'b96', 'b67', 'b49']

Section: A Framework for Dual-System Simulation.
We propose a novel framework featuring MLLMbased agents for deliberative "System 2" guidance and a specialized MMDiT architecture with a pseudo-last-frame strategy to synergistically fuse this guidance with reactive "System 1" signals, resolving critical modal conflicts.
this section cite: []

Section: State-of-the-Art Performance and Generalization.
Our method achieves highly competitive results on multiple benchmarks and is significantly preferred in user studies for its contextual naturalness. Its versatility is further demonstrated by its successful extension to complex multi-person and non-human scenarios.
this section cite: []

Section: RELATED WORK
Video Generation. The field of video generation has rapidly advanced, largely building on the success of diffusion models in visual synthesis (Ho et al., 2020;Song et al., 2020). Current approaches can be broadly categorized by their architecture. The first category adapts pre-trained text-to-image U-Nets (Esser et al., 2024a;Chen et al., 2024a) by inserting temporal modules and fine-tuning on video data (Guo et al., 2023;Wang et al., 2023b). While leveraging powerful image priors, these methods can be constrained by their original image-centric design. The second category employs Diffusion Transformer (DiT) architectures (Brooks et al., 2024;Yang et al., 2024;Zheng et al., 2024;Kong et al., 2024;Ma et al., 2025;Polyak et al., 2024;Chen et al., 2024b;Menapace et al., 2024), which treat video as a sequence of spatiotemporal patches. This unified approach has demonstrated superior scalability and flexibility, enabling high-resolution video generation with variable durations. A third, emerging direction involves integrating Large Language Models (LLMs) to enhance logical coherence and narrative structure. While this approach is still nascent in video generation, it has shown great promise in the image domain (Yu et al., 2023b;Koh et al., 2023;Pan et al., 2025;Wu et al., 2025a;2024;Shi et al., 2024) and holds significant potential for our field.
this section cite: ['b22', 'b64', 'b19', 'b2', 'b92', 'b100', 'b38', 'b46', 'b54', 'b47', 'b37', 'b51', 'b61']

Section: Video Avatars.
Video avatar models aim to create realistic human videos from driving signals, with audio being a primary focus of recent research. These audio-driven animation methods must tackle the dual challenges of motion generation and rendering. A common paradigm is a two-stage pipeline: first, a model translates audio into an intermediate motion representation (e.g., keypoints, meshes, or tokens) (Zhuang et al., 2024;Meng et al., 2025;Deng et al., 2025;Wei et al., 2024;Hogue et al., 2024;Tian et al., 2025b); then, a rendering model synthesizes the final video. This rendering stage, which animates a character from an explicit motion sequence, constitutes an independent research area known as pose-driven animation (Zhang et al., 2024;Shao et al., 2024;Chang et al., 2023;Tu et al., 2024;Wang et al., 2024b;Karras et al., 2023;Xu et al., 2024c), where the primary focus is on rendering fidelity. In contrast to these two-stage methods, recent end-to-end approaches directly generate video from audio for better synchronization (Lin et al., 2025a;b;Wang et al., 2025c;Liang et al., 2025). Despite their architectural diversity, all these audio-driven methods fundamentally treat motion generation as a direct, reactive mapping. This process lacks an explicit cognitive phase of planning and reasoning, which our work introduces to generate more plausible and intelligent behaviors.
LLMs for Cognitive Simulation and Control. Large Language Models (LLMs) have evolved from foundational models (Brown et al., 2020;Ouyang et al., 2022;Achiam et al., 2023) to powerful cognitive engines with advanced reasoning (Hurst et al., 2024;Jaech et al., 2024;Guo et al., 2025a), unlocked by prompting techniques such as Chain-of-Thought (Wei et al., 2022;Wang et al.;Yao et al., 2023) and other applications (Liang et al., 2024;Kannan et al., 2024;Qu et al., 2023). This has empowered autonomous agents to strategize and simulate believable human behaviors (Schick et al., 2023;Hong et al., 2023;Wang et al., 2023a;Park et al., 2023). Beyond standalone agents, LLM reasoning is also used to steer generative models, acting as a universal planner for controllable image and video synthesis (Brooks et al., 2023;Pan et al., 2025;Hu et al., 2024a;Yuan et al., 2024;Li et al., 2024;Wu et al., 2025b). However, applying the reasoning capabilities of these powerful models to generate fine-grained, intelligent avatar behavior remains a largely unexplored area which our work directly addresses.
this section cite: ['b104', 'b48', 'b11', 'b83', 'b23', 'b98', 'b60', 'b4', 'b72', 'b36', 'b94', 'b41', 'b3', 'b50', 'b0', 'b29', 'b30', 'b84', 'b93', 'b42', 'b35', 'b56', 'b58', 'b24', 'b52', 'b1', 'b51', 'b96', 'b40']

Section: APPROACH
3.1 OVERVIEW Our goal is to generate character animations that are both visually realistic and logically coherent with multimodal inputs. To this end, we introduce a framework that simulates both reactive (System 1) and deliberative (System 2) cognitive processes, as illustrated in Figure 2. The core of our model is a Diffusion Transformer (DiT) backbone (Seawead et al., 2025;Gao et al., 2025;Esser et al., 2024b;Peebles & Xie, 2023), pre-trained on general video tasks to acquire foundational synthesis capabilities. We enhance this base model with two critical designs.
To simulate System 2 (Sec. 3.2), we employ MLLM-based agents to reason over the input context and generate high-level semantic guidance for deliberative control. To simulate System 1 (Sec. 3.3), a specialized Multimodal DiT (MMDiT) architecture (Esser et al., 2024b) fuses this guidance with reactive signals like audio. To mitigate modal conflicts within the MMDiT, we introduce a pseudo-last-frame strategy that prevents the static reference image from interfering with dynamic motion generation.
Following common practices for simplicity (Lin et al., 2025b;Gan et al., 2025;Kong et al., 2025), our framework operates in the latent space of a pre-trained 3D VAE (Yu et al., 2023a) and is trained with a flow matching objective (Liu et al., 2022). To support long-form video synthesis, the model can operate autoregressively by conditioning new segments on the final frames of the previous one (Stypulkowski et al., 2024). We omit further discussion of these standard components to focus on our core contributions.
this section cite: ['b59', 'b16', 'b13', 'b53', 'b15', 'b39', 'b45', 'b65']

Section: AGENTIC REASONING FOR DELIBERATIVE CONTROL

this section cite: []

Section: Agentic Reasoning for Deliberative Guidance.
To simulate the deliberative nature of "System 2", our agentic reasoning module processes multimodal inputs to generate high-level, logically coherent guidance. The module takes the character's reference image, the audio clip, and an optional text prompt describing the desired behavior. It outputs this guidance as reasoning text, the explicit textual output generated by the MLLM agents in response to our designed step-by-step guided probing, which directly conditions the synthesis model.
this section cite: []

Section: Multi-Step Reasoning Pipeline.
As shown in Figure 2 (top-right), this guidance is generated by a two-stage MLLM pipeline. First, an Analyzer MLLM receives the reference image, its caption, the audio, and the user prompt. Guided by step-by-step probing prompts, the model infers the character's speech content, emotional state, and intent. These insights are then consolidated into a structured JSON object. This output is then passed to a Planner MLLM, which uses this context to devise a detailed action plan. This plan is structured as a sequence of shots, with each shot defining the character's expressions and actions for a single generation pass. This collaborative process yields a comprehensive motion schedule that ensures a coherent character persona across the entire video.
Details and examples are provided in Appendix D.
this section cite: []

Section: Framework Extensibility.
The flexibility of our agentic framework also enables various design explorations. For instance, to enhance long-form coherence, the Planner can incorporate a reflective re-planning step to correct for semantic drift during synthesis. We also explored an alternative conditioning method using reasoning-infused audio latents. While we leave a full investigation of these extensions to future work, our primary experiments focus on the straightforward reasoning text approach for its proven robustness and effectiveness. We provide preliminary examples of these explorations in Appendix E to illustrate the broader potential of our framework. This chosen design enables our agent to formulate a global, coherent plan, integrating deliberative "System 2" reasoning to provide thoughtful, top-down guidance absent in purely reactive "System 1" methods.
this section cite: []

Section: REACTIVE RENDERING VIA MULTIMODAL DIFFUSION
In this section, we detail how our diffusion model synthesizes the final video. It synergistically fuses high-level guidance from the System 2 agents (represented as text) with low-level, reactive signals from the audio input.
this section cite: []

Section: Rethinking Reference Image Conditioning.
A critical input in video avatar models is the reference image, which serves two distinct purposes: first, providing an initial frame as a conditioning prefix for the generated sequence, and second, maintaining identity consistency. While the former is a necessary function, the latter (using a static reference image to enforce identity) is problematic.
Prior methods, whether using dedicated networks (Hu, 2024a;Tian et al., 2025c;Zhu et al., 2023) or parameter reuse (Lin et al., 2025b;Kong et al., 2025), typically condition the model on a reference image sampled from the training video. This training strategy teaches the model a spurious correlation: as illustrated in Figure 3, it learns that the reference image must appear literally within the generated sequence. We argue this is a critical artifact that severely restricts motion dynamics and conflicts with other driving signals. The root cause is that the reference image is an artificial construct, not a condition native to the video data itself. This leads to a training dilemma regarding the semantic distance of the reference image. Sampling a semantically "close" reference (e.g., from the same clip) creates the static artifact, while sampling a semantically "distant" one can teach the model to exhibit excessive, identity-altering variation.
Our solution is to discard the reference image entirely during training and introduce a novel guidance mechanism. As shown in Figure 2 (bottom-right), we instead probabilistically condition the model on the GT first and last frames of the video clip, both native signals, each with a dropout probability of 0.1. During inference, we repurpose this mechanism by placing the user's reference image in the last frame's position, creating a "pseudo-last-frame". Crucially, we shift its positional encoding, RoPE (Su et al., 2024), by assigning it a positional index corresponding to a fixed temporal distance beyond the final generated frame. This pseudo frame functions as a "carrot on a stick": it guides the model toward the target identity without ever forcing it to replicate the static image, which is discarded after synthesis. As our experiments show, this approach eliminates training artifacts and mitigates autoregressive error, achieving a superior trade-off between motion dynamics and identity stability.
this section cite: ['b103', 'b39', 'b66']

Section: Symmetric Fusion and Modality Warm-Up.
Having established data-native conditions, we now address their joint modeling. We adopt an MMDiT backbone but introduce a dedicated audio branch, architecturally symmetric to the video and text branches. Instead of using cross-attention, all three modalities are fused within each transformer block by concatenating their tokens and applying a sin- gle shared multi-head self-attention mechanism. This symmetric design enables true joint modeling, as tokens from all modalities mutually attend to one another, allowing for iterative refinement and deep semantic alignment.
However, this architecture introduces a critical training challenge: naive joint training causes the model to over-rely on the dense audio signal, which washes out text guidance and disrupts the pretrained video branch's patterns, degrading overall synthesis capability. To resolve this, we propose a two-stage warm-up strategy. In stage 1, we train the full three-branch model jointly, forcing the model to learn an optimal division of labor where the audio branch specializes in its core competencies (e.g., lip sync, speech mannerisms). In stage 2, we initialize the text and video branches with their original weights and the audio branch with its specialized weights from stage 1, then fine-tune the entire model. This strategy provides each branch with a strong prior, mitigating modality conflict and preserving each input's distinct conditioning power.
Ultimately, our redesigned reference conditioning and symmetric audio fusion enable the model to effectively execute the deliberative plan of System 2, translating high-level guidance while maintaining the reactive fidelity of System 1.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETUP
Implementation and Training. Our model, built upon the MMDiT architecture, generates 120frame clips at 24 fps. For efficiency, most ablation studies are conducted at a 480p resolution. For comparison with prior work, these clips are then upscaled to 720p to match common evaluation setting. The model underwent a three-stage training process: an audio branch warm-up, a main phase on 15,000 hours of video, and a final fine-tuning stage on a 100-hour high-quality subset. Further implementation and training details are provided in the Appendix B.
Evaluation Datasets and Metrics. To rigorously test generalization, we constructed two challenging custom benchmarks: a diverse single-subject set (150 cases) and a multi-subject set (57 cases), both featuring a wide range of characters and complex audio. For fair comparison with prior work, we also evaluate on the CelebV-HQ (Zhu et al., 2022) and CyberHost (Lin et al., 2025a) test sets.
Our evaluation protocol includes both objective and subjective metrics. We measure image quality (FID, IQA), video quality (FVD), lip-sync (Sync-C), and gesture dynamics (HKC, HKV). Crucially, to assess perceptual quality, we conducted user studies with 40 participants, using pairwise comparisons (GSB score) and flaw identification (LSI, MU, ID) to capture nuances that objective metrics miss. A detailed description of our datasets and evaluation protocols can be found in the Appendix C.
this section cite: ['b102']

Section: ABLATION STUDIES
We conduct a series of ablation studies to validate the contributions of our proposed components, using a custom single-subject test set of 150 video clips. Our analysis systematically isolates the impact of two key elements: (1) the agentic reasoning module and (2) our proposed conditioning architecture. For a comprehensive assessment, we employ both quantitative metrics and subjective user studies to evaluate performance, perceptual quality, and user preference.
this section cite: []

Section: Effectiveness of Agentic Reasoning.
We analyze our Agentic Reasoning module by progressively ablating its components: first removing multi-step reasoning, then the Analyzer, and finally the full module for a "System 1 only" baseline. As shown in Table 1, standard metrics like IQA and Sync-C show minimal variation. This is expected, as these metrics measure low-level fidelity, are largely saturated, and are insensitive to higher-level semantics. A more telling trend emerges from Hand Keypoint Variance (HKV), which progressively decreases as reasoning is removed, indicating that the generated motion becomes more static and less expressive. This directly demonstrates the value of our reasoning module.
To assess the module's semantic impact, we conducted subjective evaluations. As shown in Table 2 (a), our full model shows a substantial advantage. Agentic reasoning yields a 29% gain in GSB and a significant reduction in perceived motion unnaturalness (MU), while maintaining strong lipsync (LSI) and image quality (ID). These findings confirm our reasoning module enhances motion plausibility and naturalness, qualities not captured by standard objective metrics. This is further corroborated by an MLLM-based analysis (Appendix D.2) confirming our method's superior contextual coherence. We strongly encourage viewing the supplementary material and project page, which showcase compelling visual results unattainable with existing "System 1" models. Effectiveness of Proposed Conditioning Modules. We also ablate our core architectural designs in Tables 1 and 2, keeping the agentic reasoning module fixed. We ablate our core architectural designs by evaluating three key changes: replacing our MM-Attention-based audio conditioning with standard cross-attention, removing the MM-Warmup stage, and substituting our pseudo-last-frame strategy with conventional reference image attention. Our full model consistently leads in objective metrics, with superior HKC/HKV scores highlighting enhanced motion dynamics. Subjectively, our method also significantly outperforms a baseline using cross-attention for both audio and reference image conditioning (akin to OmniHuman-1 (Lin et al., 2025b)), showing clear advantages in lip-sync (LSI), motion naturalness (MU), and visual quality (ID). These results validate our unique conditioning architecture as a robust foundation for executing agentic plans. Further visual analysis in the appendix demonstrates how PLF maintains identity consistency during dynamic motion.
Text-Conditioning Fidelity. As a supplementary experiment, we verified that our multi-modal training does not degrade text-following capabilities. Under text-only conditioning, a user study (Table 2) confirms our model achieves on-par text alignment (TA) with the base model, while demonstrating significantly improved motion naturalness (MU) and visual quality (ID). This validates that our approach enhances generation quality without sacrificing core text-conditioning fidelity.
this section cite: []

Section: FURTHER EXPLORATION ON APPLICATIONS
Generalization to Diverse and Multi-Person Scenarios. Our model demonstrates robust generalization, enabled by a dual-system framework whose agentic reasoning provides context-aware motion guidance for non-human subjects and interprets conversational turn-taking (Figure 4). To further showcase this extensibility, we conducted a preliminary study on multi-person animation by incorporating a predicted speaker mask to guide the system (details in Appendix B). In a focused comparison (Table 3), our full model significantly outperforms baselines like InterActHuman (Wang et al., 2025c) and an ablation without reasoning. The observed improvements in gesture dynamics (HKC/HKV), lip-sync (Sync-D), and driving accuracy (DA) primarily validate the effectiveness of our reasoning component in enhancing coordination in such complex settings. These results underscore our method's broad applicability and potential for creating interactive avatars (see supplementary material for video results).  Method Top-1 (%) Skyreel-A1 5% FantasyTalking 8% OmniAvatar 14% MultiTalk 18% OmniHuman-1 22% Ours 33% (a) Best-Choice Selection. (b) GSB against leading proprietary models. 4.4 COMPARISON AMONG RECENT METHODS Comparisons with State-of-the-Art Methods. We conduct a comprehensive evaluation against leading academic baselines across two scenarios: portrait and full-body generation. For portraits, we compare on the CelebV-HQ test set against specialized and DiT-based methods, including SadTalker (Zhang et al., 2023), EchoMimic (Chen et al., 2024c), Hallo (Xu et al., 2024a),
Hallo3 (Cui et al., 2024), Loopy (Jiang et al., 2025), and OmniHuman-1 (Lin et al., 2025b). For the more challenging full-body synthesis task, we evaluate on the CyberHost test set against recent DiT models like Skyreels-A1 (Fei et al., 2025), FantasyTalking (Wang et al., 2025b), OmniAvatar (Gan et al., 2025), MultiTalk (Kong et al., 2025), and OmniHuman-1.
Quantitative results (Table 4) show our method consistently ranking among the top two. While on par with OmniHuman-1 in portraits, a setting where limited motion challenges objective metrics, our advantages become pronounced in the full-body setting (see also Table 2b and Figure 12). In this setting, our model excels at generating dynamic, large-scale movements (high HKV) while preserving local detail (competitive HKC). To assess perceptual quality, we conducted user studies against a top academic method (Figure 5a) and four leading, anonymized commercial systems (Figure 5b).
In user studies, human evaluators consistently and significantly preferred our results overall, a preference we attribute to high-level qualities like contextual coherence that objective metrics fail to capture but are critical for perceptual realism.
this section cite: ['b10', 'b32', 'b14', 'b15', 'b39']

Section: CONCLUSION
Inspired by the dual-system theory of human cognition, we introduce a new paradigm for video avatars. We argue that existing methods, by simulating only reactive "System 1" thinking, fail to align motion with high-level intent, resulting in behaviors that lack contextual appropriateness and logical coherence. To address this, we propose a novel framework that models deliberative "System 2" processes via two innovations: MLLM-based agents for semantic planning and a specialized MMDiT for high-fidelity synthesis. Experimental results validate that our approach generates more expressive and logically coherent motions in both single-and multi-person scenarios, achieving leading performance on both subjective and objective metrics across multiple benchmarks. We hope this cognitive agency perspective offers the community a promising path toward creating video avatars that are not just visually realistic, but truly believable.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Instructpix2pix: Learning to follow image editing instructions Year: (2023)
Ref_id:b2 Title: Video generation models as world simulators Year: (2024)
Ref_id:b3 Title: Language models are few-shot learners Year: (2020)
Ref_id:b4 Title: Realistic human poses and facial expressions retargeting with identity-aware diffusion Year: (2023)
Ref_id:b5 Title: Pixart-{\delta}: Fast and controllable image generation with latent consistency models Year: (2024)
Ref_id:b6 Title: Gentron: Diffusion transformers for image and video generation Year: (2024)
Ref_id:b7 Title: Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions Year: (2024)
Ref_id:b8 Title: Out of time: automated lip sync in the wild Year: (2016)
Ref_id:b9 Title: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities Year: (2025)
Ref_id:b10 Title: Hallo3: Highly dynamic and realistic portrait image animation with diffusion transformer networks Year: (2024)
Ref_id:b11 Title: Stereo-talker: Audio-driven 3d human synthesis with priorguided mixture-of-experts Year: (2025)
Ref_id:b12 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b13 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b14 Title: Skyreels-a2: Compose anything in video diffusion transformers Year: (2025)
Ref_id:b15 Title: Omniavatar: Efficient audiodriven avatar video generation with adaptive body animation Year: (2025)
Ref_id:b16 Title: Seedance 1.0: Exploring the boundaries of video generation models Year: (2025)
Ref_id:b17 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b18 Title: Seed1. 5-vl technical report Year: (2025)
Ref_id:b19 Title: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning Year: (2023)
Ref_id:b20 Title: Zero-shot talking avatar generation Year: (2023)
Ref_id:b21 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b22 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b23 Title: Diffted: One-shot audio-driven ted talk video generation with diffusion-based co-speech gestures Year: (2024)
Ref_id:b24 Title: Metagpt: Meta programming for a multi-agent collaborative framework Year: (2023)
Ref_id:b25 Title: Animate anyone: Consistent and controllable image-to-video synthesis for character animation Year: (2024)
Ref_id:b26 Title: Animate anyone: Consistent and controllable image-to-video synthesis for character animation Year: (2024)
Ref_id:b27 Title: Storyagent: Customized storytelling video generation via multi-agent collaboration Year: (2024)
Ref_id:b28 Title: Unveiling the potential of small language models with scalable training strategies Year: (2024)
Ref_id:b29 Title: Gpt-4o system card Year: (2024)
Ref_id:b30 Title: Openai o1 system card Year: (2024)
Ref_id:b31 Title: Real-time one-shot neural head avatars on mobile devices Year: (2024)
Ref_id:b32 Title: Loopy: Taming audio-driven portrait avatar with long-term motion dependency Year: (2025)
Ref_id:b33 Title: Thinking, fast and slow Year: (2011)
Ref_id:b34 Title: Prospect theory: An analysis of decision under risk Year: (2013)
Ref_id:b35 Title: Smart-llm: Smart multi-agent robot task planning using large language models Year: (2024)
Ref_id:b36 Title: Dreampose: Fashion image-to-video synthesis via stable diffusion Year: (2023)
Ref_id:b37 Title: Generating images with multimodal language models Year: (2023)
Ref_id:b38 Title: A systematic framework for large video generative models Year: (2024)
Ref_id:b39 Title: Let them talk: Audio-driven multi-person conversational video generation Year: (2025)
Ref_id:b40 Title: Anim-director: A large multimodal model powered agent for controllable animation video generation Year: (2024)
Ref_id:b41 Title: Alignhuman: Improving motion and fidelity via timestep-segment preference optimization for audio-driven human animation Year: (2025)
Ref_id:b42 Title: Wavcraft: Audio editing and generation with large language models Year: (2024)
Ref_id:b43 Title: Cyberhost: A one-stage diffusion framework for audio-driven talking body generation Year: ()
Ref_id:b44 Title: Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models Year: (2025)
Ref_id:b45 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b46 Title: Step-video-t2v technical report: The practice, challenges, and future of video foundation model Year: (2025)
Ref_id:b47 Title: Snap video: Scaled spatiotemporal transformers for text-to-video synthesis Year: (2024)
Ref_id:b48 Title: Echomimicv2: Towards striking, simplified, and semi-body human animation Year: (2025)
Ref_id:b49 Title:  Year: (2024)
Ref_id:b50 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b51 Title: Transfer between modalities with metaqueries Year: (2025)
Ref_id:b52 Title: Generative agents: Interactive simulacra of human behavior Year: (2023)
Ref_id:b53 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b54 Title: Movie gen: A cast of media foundation models Year: (2024)
Ref_id:b55 Title: Skyreels-a1: Expressive portrait animation in video diffusion transformers Year: (2025)
Ref_id:b56 Title: Layoutllm-t2i: Eliciting layout guidance from llm for text-to-image generation Year: (2023)
Ref_id:b57 Title: Robust speech recognition via large-scale weak supervision Year: (2023)
Ref_id:b58 Title: Toolformer: Language models can teach themselves to use tools Year: (2023)
Ref_id:b59 Title: Seaweed-7b: Cost-effective training of video generation foundation model Year: (2025)
Ref_id:b60 Title: Human4dit: Free-view human video generation with 4d diffusion transformer Year: (2024)
Ref_id:b61 Title: Lmfusion: Adapting pretrained language models for multimodal generation Year: (2024)
Ref_id:b62 Title: First order motion model for image animation Year: (2019)
Ref_id:b63 Title: Motion representations for articulated animation Year: (2021)
Ref_id:b64 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b65 Title: Diffused heads: Diffusion models beat gans on talking-face generation Year: (2024)
Ref_id:b66 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b67 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b68 Title: Raft: Recurrent all-pairs field transforms for optical flow Year: (2020)
Ref_id:b69 Title: Emo2: End-effector guided audiodriven avatar video generation Year: (2025)
Ref_id:b70 Title: Emo2: End-effector guided audiodriven avatar video generation Year: (2025)
Ref_id:b71 Title: Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions Year: (2025)
Ref_id:b72 Title: Editing video motion via lightweight score-guided diffusion Year: (2024)
Ref_id:b73 Title: Fvd: A new metric for video generation Year: (2019)
Ref_id:b74 Title: Open and advanced large-scale video generative models Year: (2025)
Ref_id:b75 Title: Conditional dropout for progressive training of portrait video generation Year: (2024)
Ref_id:b76 Title: Voyager: An open-ended embodied agent with large language models Year: (2023)
Ref_id:b77 Title: Modelscope text-to-video technical report Year: (2023)
Ref_id:b78 Title: Fantasytalking: Realistic talking portrait generation via coherent motion synthesis Year: (2025)
Ref_id:b79 Title: One-shot free-view neural talking-head synthesis for video conferencing Year: (2021)
Ref_id:b80 Title: Unianimate: Taming unified video diffusion models for consistent human image animation Year: (2024)
Ref_id:b81 Title: Self-consistency improves chain of thought reasoning in language models Year: ()
Ref_id:b82 Title: Interacthuman: Multi-concept human animation with layout-aligned audio conditions Year: (2025)
Ref_id:b83 Title: Aniportrait: Audio-driven synthesis of photorealistic portrait animation Year: (2024)
Ref_id:b84 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b85 Title: Qwen-image technical report Year: (2025)
Ref_id:b86 Title: Q-align: Teaching lmms for visual scoring via discrete text-defined levels Year: (2023)
Ref_id:b87 Title: Next-gpt: Any-to-any multimodal llm Year: (2024)
Ref_id:b88 Title: Automated movie generation via multi-agent cot planning Year: (2025)
Ref_id:b89 Title: Hierarchical audio-driven visual synthesis for portrait image animation Year: (2024)
Ref_id:b90 Title: Vasa-1: Lifelike audio-driven talking faces generated in real time Year: (2024)
Ref_id:b91 Title: Magicanimate: Temporally consistent human image animation using diffusion model Year: (2024)
Ref_id:b92 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2024)
Ref_id:b93 Title: Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems Year: (2023)
Ref_id:b94 Title: Language model beats diffusiontokenizer is key to visual generation Year: (2023)
Ref_id:b95 Title: Scaling autoregressive multi-modal models: Pretraining and instruction tuning Year: (2023)
Ref_id:b96 Title: Enabling generalist video generation via a multi-agent framework Year: (2024)
Ref_id:b97 Title: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation Year: (2023)
Ref_id:b98 Title: Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance Year: (2024)
Ref_id:b99 Title: Thin-plate spline motion model for image animation Year: (2022)
Ref_id:b100 Title: Open-sora: Democratizing efficient video production for all Year: (2024)
Ref_id:b101 Title: Fada: Fast diffusion avatar synthesis with mixed-supervised multi-cfg distillation Year: (2024)
Ref_id:b102 Title: Celebv-hq: A large-scale video facial attributes dataset Year: (2022)
Ref_id:b103 Title: Tryondiffusion: A tale of two unets Year: (2023)
Ref_id:b104 Title: Vlogger: Make your dream a vlog Year: (2024)
