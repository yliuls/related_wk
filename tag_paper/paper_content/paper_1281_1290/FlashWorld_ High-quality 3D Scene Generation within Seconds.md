Title: FLASHWORLD: HIGH-QUALITY 3D SCENE GENERA-TION WITHIN SECONDS
Abstract: We propose FlashWorld, a generative model that produces 3D scenes from a single image or text prompt in seconds, 10 ∼ 100× faster than previous works while possessing superior rendering quality. Our approach shifts from the conventional multi-view-oriented (MV-oriented) paradigm, which generates multi-view images for subsequent 3D reconstruction, to a 3D-oriented approach where the model directly produces 3D Gaussian representations during multi-view generation. While ensuring 3D consistency, 3D-oriented method typically suffers poor visual quality. FlashWorld includes a dual-mode pre-training phase followed by a cross-mode post-training phase, effectively integrating the strengths of both paradigms. Specifically, leveraging the prior from a video diffusion model, we first pre-train a dual-mode multi-view diffusion model, which jointly supports MV-oriented and 3D-oriented generation modes. To bridge the quality gap in 3D-oriented generation, we further propose a cross-mode post-training distillation by matching distribution from consistent 3D-oriented mode to high-quality MV-oriented mode. This not only enhances visual quality while maintaining 3D consistency, but also reduces the required denoising steps for inference. Also, we propose a strategy to leverage massive single-view images and text prompts during this process to enhance the model's generalization to out-of-distribution inputs. Extensive experiments demonstrate the superiority and efficiency of our method. Our code is released at https://github.com/imlixinyang/FlashWorld.

Section: Text & Image Prompt
Seconds High-quality 3DGS Scene  (Gao et al., 2024), Bolt3D (Szymanowicz et al., 2025), Wonderland (Liang et al., 2025), and Ours w/ MV-Diff) suffer from noisy textures due to multi-view inconsistency. MV-oriented distillation further exacerbates this flaw (i.e., Ours w/ MV-Dist). 3D-oriented diffusion methods (i.e., Ours w/ 3D-Diff) suffer from blurry visual effect. Our cross-mode distillation model (i.e., Ours) simultaneously solves these, making the quality of the novel view close to the input view. The time cost per scene, tested on a single GPU, is presented at the bottom of each method.
Early methods typically relied on assembling pre-existing 3D assets (Xu et al., 2002;Yu et al., 2011;Wu et al., 2018;Feng et al., 2023;C ¸elen et al., 2024;Yang et al., 2024c;Deng et al., 2025) or iteratively reconstructing scenes from inpainted images and depth maps (Cai et al., 2023;Fridman et al., 2023;Höllein et al., 2023;Lei et al., 2023;Yu et al., 2024a;Zhang et al., 2024a;b;Chung et al., 2023;Yu et al., 2025;Shriram et al., 2025;Ni et al., 2025). Yet, without holistic scene-level understanding or multi-view consistency constraints, these approaches often struggle to produce semantically coherent and visually realistic scenes. To address this, scalable data-driven approaches have emerged. The dominant paradigm is a two-stage, multi-view-oriented (MV-oriented) pipeline (Gao et al., 2024;Sun et al., 2024;Wallingford et al., 2024;Zhao et al., 2025;Szymanowicz et al., 2025;Yang et al., 2025;Go et al., 2025a;b): a diffusion model first generates multiple views from text or reference images, and then a 3D reconstruction is performed. However, the lack of explicit 3D constraints during view synthesis often causes geometric and semantic inconsistencies in generated views, leading to a noticeable visual quality gap between synthesized views and the reconstructed 3D scene. Moreover, the considerable computational overhead of both the diffusion and reconstruction stages leads to generation latencies of several minutes to hours, as shown in Fig. 2. These limitations compromise the effectiveness and efficiency of current 3D scene generation methods, blocking their applications.
One promising but relatively less explored direction is the 3D-oriented scene generation pipeline (Xu et al., 2023;Li et al., 2024a;b;Tang et al., 2025;Cai et al., 2024;Zuo et al., 2024). These methods combine differentiable rendering (Mildenhall et al., 2020;Wang et al., 2021;Kerbl et al., 2023) with diffusion models, allowing for direct 3D scene generation without additional reconstruction. However, these generated 3D scenes often suffer from visual artifacts and blurry content. Consequently, they often require an additional refinement stage, which significantly degrades generation efficiency.
To enhance efficiency of diffusion models, post-training distillation techniques, such as consistency model distillation (Song et al., 2023) and distribution matching distillation (Yin et al., 2024b;a;Xie et al., 2024), are often used. However, directly applying distillation amplifies each framework's inherent limitations: e.g., it exacerbates multi-view inconsistency in the MV-oriented pipeline.
In this work, we introduce a novel framework that combines the strengths of both paradigms through distillation, achieving substantial gains in 3D consistency and visual fidelity while significantly accelerating inference speed. Our contributions are briefly summarized as follows:
• We introduce a dual mode pre-training strategy built on a video diffusion model to train a multi-view diffusion model capable of operating in both MV-oriented and 3D-oriented modes.
• We propose a cross-mode post-training strategy, where the MV-oriented mode serves as the teacher to improve visual quality, while the 3D-oriented mode acts as the student to ensure 3D consistency.
• To improve out-of-distribution generalization ability, we introduce a novel strategy that can leverage massive unlabeled image data and text prompts with randomly simulated camera trajectories during post-training, enhancing the model's adaptability to diverse inputs, as shown in Fig. 1.
this section cite: ['b10', 'b48', 'b27', 'b60', 'b71', 'b57', 'b8', 'b3', 'b6', 'b1', 'b9', 'b18', 'b24', 'b5', 'b70', 'b45', 'b35', 'b10', 'b47', 'b51', 'b77', 'b48', 'b64', 'b61', 'b50', 'b2', 'b79', 'b34', 'b54', 'b22', 'b46', 'b59']

Section: PRELIMINARY
Diffusion models (Ho et al., 2020) generate data by progressively transforming samples from a standard Gaussian distribution p(x T ) ∼ N (0, I) into samples from a target data distribution p(x), which have been widely applied across multiple domains, including image synthesis (Rombach et al., 2022), multi-view generation (Shi et al., 2023;Tang et al., 2023), video generation (Blattmann et al., 2023;Wan et al., 2025), and panoramic 3D scenes (HunyuanWorld, 2025). The core methodology involves training a denoising network with optimizable parameters to reconstruct the original data by removing the injected Gaussian noise ϵ from x according to a predefined noise schedule. The forward process is formulated as: x t = F (x, t) = α t x + σ t ϵ, where α t and σ t jointly control the signal-to-noise ratio at each timestep t. The denoising network can be trained to predict clean data x from noisy input x t by minimizing the following objective:
L = E x,t,ϵ ∥x -xθ (x t , t)∥ 2 .(1)
Alternative training objectives include predicting noise ϵ (Ho et al., 2020) or a linear combination of x 0 and ϵ, known as v-prediction (Salimans & Ho, 2022). All predictions can be converted to the denoised estimate µ(x t , t) and represent the gradient of the log probability of the distribution:
s(x t , t) = ∇ xt log p t (x t ) = - x t -α t µ(x t , t) σ 2 t .
(2) Distribution matching distillation (DMD) (Yin et al., 2024b;a) is an advanced technique designed to distill a slow, multi-step teacher diffusion model into a fast, few-step student model with comparable generation capabilities. The key component is to minimize the approximate KL divergence across randomly sampled timesteps t and noise inputs z between the smoothed real data distribution p real (x t ) and the student generator's output distribution p fake (x t ) by:
∇L DMD = -E t (s real (F (G θ (z), t), t) -s fake (F (G θ (z), t), t)) dG θ (z) dθ dz ,(3)
where s real and s fake are approximated scores using diffusion models µ real and µ fake trained on their respective distributions (Eq. 1). DMD uses a frozen pre-trained diffusion model µ real as the teacher, and dynamically updates µ fake while training G θ , using diffusion loss on samples from the generator.
this section cite: ['b17', 'b39', 'b44', 'b49', 'b0', 'b52', 'b17', 'b40']

Section: METHOD
The core of our framework lies in leveraging DMD to transfer knowledge from a MV-oriented multi-view diffusion model, one well-established for high visual quality, to a 3D-oriented few-step multi-view generator, which is inherently endowed with 3D consistency. However, this paradigm introduces two key challenges: First, for open-world 3D scene generation, the 3D-oriented few-step generator requires a sufficiently robust prior and strong generative capacity from the start. Without this, the training process is prone to collapse. Second, due to the limited quantity and diversity of high-quality multi-view datasets, it becomes critical to develop a strategy that effectively handles scenarios with diverse styles, object categories, and camera trajectories. Specifically, To address these challenges, we first design a dual-mode pre-training strategy as detailed in Sec. 3.1. This strategy
Novel Views 3DGS Decoder Dual-mode Pre-training Novel Cameras Rendering Cameras DIT Blocks DIT Blocks Real Score Cross-mode Model Distillation DiT Blocks Input Views Add Noise 3DGS Decoder DIT Blocks 3DGS Novel Cameras Lcon Fake Images Noisy Fake Images add noise Fake Score Ldm 3DGS Target Images Render 3DGS Decoder Cross-mode Post-training Novel Cameras DiT Blocks Render DiT Blocks DiT Blocks Add Noise Real Score Fake Score Noise A lego excavator.
this section cite: []

Section: 3DGS Conditions

this section cite: []

Section: -

this section cite: []

Section: Input Cameras
Figure 3: Method overview. We first pre-train a dual-mode multi-view latent diffusion model using multi-view datasets, and then employ an cross-mode distillation post-training strategy to accelerate generation while enhancing visual quality and inheriting 3D consistency. yields a multi-view diffusion model that operates in two distinct modes: a MV-oriented mode for high visual fidelity and a 3D-oriented mode for inherent 3D consistency. Subsequently, in Sec. 3.2, we present a cross-mode post-training framework to bridge these two modes: the MV-oriented mode acts as the teacher, supplying score distillation gradients to ensure visual quality; the 3D-oriented mode serves as the student, learning to inherit the teacher's distribution while preserving 3D consistency. Furthermore, to explicitly tackle out-of-distribution generalization, in Sec. 3.3, we introduce a strategy that can leverage single-view image data, text prompts, and pre-defined camera trajectories, boosting the model's adaptability to diverse scenarios.
this section cite: []

Section: DUAL-MODE PRE-TRAINING
In this stage, we pre-train a dual-mode multi-view latent diffusion model using multi-view datasets, as illustrated in Fig. 3 (left). For each training iteration, we sample a batch containing multi-view images X , their corresponding camera parameters C, and additional conditioning information y (such as a text prompt or a single-view image). The multi-view images are first encoded into the latent space to obtain multi-view latents Z = E(X ). A forward diffusion process is then applied to produce noisy multi-view latents Z t = α t Z + σ t ϵ at a randomly sampled timestep t.
The noisy latents Z t , together with the camera parameters C and conditioning y, are input to the denoising network for reverse denoising training. We represent cameras using Reference-Point Plücker Coordinates (Cai et al., 2024) raymaps. The denoising network is a Diffusion Transformer (DiT) (Peebles & Xie, 2023) enhanced with 3D attention blocks, and outputs both a denoised estimate ẐMV and an auxiliary multi-view feature F. For the MV-oriented mode, we optimize:
L MV = E X ,t,ϵ,y,C Z -ẐMV 2 .
(4)
To enable 3D-oriented generation, we decode 3D Gaussian parameters from the multi-view feature F using a 3DGS decoder: {τ, q, s, α, c} = D G (F), where τ , q, s, α, and c represent the depth, rotation quaternion, scale, opacity, and spherical harmonics coefficients of the 3D Gaussians, respectively. The 3DGS decoder D G is initialized from the original latent decoder D, with its first and last convolutional layers re-initialized to accommodate the additional features and output channels required for the Gaussian parameters. The predicted depth is then converted to pixel-aligned Gaussian points via µ = o + τ d, where o and d denote the camera origin and ray direction, respectively. For the 3D-oriented mode, we optimize the following loss:
L 3D = E X ,t,ϵ,y,C ∥X novel -R(G, C novel )∥ 2 ,(5)
where R denotes the rendering operation, G = {µ, q, s, α, c} is the set of 3D Gaussians, and X novel , C novel are the ground-truth novel-view images and their associated cameras, respectively. During inference, both MV-oriented and 3D-oriented modes can be used for denoising (Li et al., 2024a;b). In particular, for the 3D-oriented mode, the model predicts the estimated clean multi-view latents as Ẑ3D = E(R(G, C)).
In contrast to previous methods (Li et al., 2024a;b) that are initialized from image diffusion models (Rombach et al., 2022), we initialize our framework with a video diffusion model (Wan et al., 2025). We observe that this video model not only converges more rapidly, but also features a powerful VAE with a higher compression rate, enabling support for a larger number of views (i.e., 24) and higher output resolutions (i.e., 480P).
this section cite: ['b2', 'b36', 'b39', 'b52']

Section: CROSS-MODE POST-TRAINING
After pre-training, we employ an asymmetric distillation strategy to accelerate generation while enhancing visual quality and inheriting 3D consistency, as shown in Fig 3 (right). Specifically, we observe that while the MV-oriented mode exhibits poor consistency, it can generate multi-view images with high visual quality; thus, we leverage the MV-oriented mode of our dual-mode multi-view latent diffusion model as the real teacher µ real : this teacher model is frozen, tasked with computing the real score gradient. Another copy of the model µ fake is dynamically updated to estimate the fake score corresponding to the current distribution of the distilled generator. Meanwhile, our few-step student model is initialized with the 3D-oriented mode of our dual-mode multi-view latent diffusion model.
The 3D-oriented multi-view generation process alternates between denoising and noise injection steps to enhance sample quality of the 3D scenes following LCMs (Luo et al., 2023). Specifically, we first define a schedule of N timesteps, denoted as {t 1 , t 2 , • • • , t N }, where N is typically small (e.g., 4). Starting from a randomly sampled noise Z t1 = z ∼ N (0, I), we alternate between 3Doriented denoising updates Ẑti = E(R(G θ,3D (Z ti , t i , y, C), C)) and forward diffusion steps Z ti+1 = α ti+1 Ẑti + σ ti+1 ϵ where G θ,3D is the 3DGS generator and ϵ ∼ N (0, I), until obtaining the 3D Gaussians at the final step (i.e., G θ,3D (Z t N , t N , y, C)). At each step, the multi-view denoising update is performed based on rendering, thereby ensuring that 3D consistency is maintained throughout the process.
During distillation training, we adopt the DMD2 algorithm (Yin et al., 2024a), which includes a DMD objective (i.e., Eq. 3) and a standard non-saturating GAN objective (Goodfellow et al., 2020), where the logits value required by the GAN loss is obtained by adding an extra classification branch with several convolutional layers at the end of the fake score network. We adopt the estimated R1 regularization (Lin et al., 2025a) to stabilize the GAN training. The DMD objective and the GAN objective are employed to optimize both the original and novel views.
We also observe that relying solely on the above strategy can lead to the generation of scenes with unstable floating artifacts. We hypothesize that this instability arises from the challenges in optimizing with noisy gradients introduced by Gaussian rendering and latent encoding. To address this, during post-training, we additionally update an MV-oriented student model at a lower frequency. This model shares the same DiT backbone as the 3D-oriented student model. To encourage alignment between the two modes, we introduce a cross-mode consistency loss:
L CMC = E z,t,ϵ,y,C,i λ ∥E(R(G θ,3D (Z ti , t i , y, C), C)) -G θ,MV (Z ti , t i , y, C)∥ 2 , (6
)
where λ is a small weighting factor (i.e., 0.1). Because the MV-oriented mode prediction are less affected by unstable rendering gradients, this consistency loss regularizes the 3D-oriented mode to produce more stable and reliable generations.
this section cite: ['b33', 'b13']

Section: OUT-OF-DISTRIBUTION DATA CO-TRAINING.
During pre-training, it is common to jointly train on image and video generation tasks to enhance the model's generalization ability. While this approach benefits the DiT backbone, it does not optimize the 3DGS decoder, potentially limiting the range of inputs the 3DGS decoder can effectively process.
To address this, in the post-training phase, we introduce a strategy to broaden the model's input distribution and improve generalization to diverse scenes, even when multi-view data is limited in quantity and variety. Specifically, we combine image or text conditions sampled from image datasets with random camera trajectories, which can be drawn either from multi-view sequences or from a set of predefined trajectories. Importantly, we omit the GAN loss during this co-training process to prevent distribution mismatches. This approach not only enhances the model's generalization to a wide range of input images and text prompts, but also increases its robustness when encountering out-of-distribution camera trajectories. The details of this strategy are provided in Appendix A .
this section cite: []

Section: EXPERIMENTS
In this section, we evaluate the performance of our method on various benchmarks, including imageto-3D scene generation, text-to-3D scene generation, and WorldScore benchmark. For implementation details, please refer to Appendix A.
this section cite: []

Section: COMPARISON ON IMAGE-TO-3D SCENE GENERATION
We present a qualitative comparison with state-of-the-art image-to-3D scene generation methods in Fig. 4. These baselines are MV-oriented, including: CAT3D (Gao et al., 2024), which generates novel views via multi-view diffusion followed by optimization-based 3D reconstruction; Bolt3D (Szymanowicz et al., 2025), which synthesizes both appearance and geometry for novel views and then applies a feed-forward 3D reconstruction; and Wonderland (Liang et al., 2025), a leading approach that leverages a powerful video diffusion model and latent-based feed-forward 3D reconstruction. As these methods are not open-sourced, we utilize the video results provided in their respective project pages for visualization. We employ ViPE (Huang et al., 2025) to estimate camera poses and intrinsics from the baseline videos. CAT3D struggles to generate complex scenes, resulting in blurry outputs and missing geometric details. Bolt3D also exhibits inaccurate geometric details, such as imprecise tree branches and needle-like leaves. Wonderland suffers from repeated and distorted Gaussian artifacts, especially under large camera pose changes. Overall, these MV-oriented methods fail to generate complex scenes, primarily due to insufficient multi-view consistency. In contrast, our model produces high-fidelity, detailed scenes and successfully recovers intricate structures (e.g., leaves, iron fences, and tentacles), highlighting the advantages of our 3D-oriented pipeline.
this section cite: ['b10', 'b27', 'b19']

Section: COMPARISON ON TEXT-TO-3D SCENE GENERATION
We compare our method against several state-of-the-art text-to-3D scene generation approaches, including Director3D (Li et al., 2024b), Prometheus (Yang et al., 2025), SplatFlow (Go et al., 2025a), and VideoRFSplat (Go et al., 2025a). A qualitative comparison is presented in Fig. 5. Director3D relies on per-scene refinement, which frequently introduces blurry and wave-like artifacts in the generated results. In contrast, our model produces accurate objects with fine-grained details, such as animal fur, while preserving realistic backgrounds. Prometheus does not utilize refinement, and due to the inherent inconsistency of its MV-oriented pipeline, the generated scenes are often blurry and may exhibit incorrect object geometries (e.g., chair legs). Our approach, however, is Input A fluffy, orange cat.
A spacious kitchen with wooden floors, white countertops, and an island in the center.
A traditional wooden gate with red lanterns.
A large stone fountain surrounded by lush greenery and a clear blue sky.
this section cite: ['b64']

Section: Ours

this section cite: []

Section: Director3D
Prometheus SplatFlow VideoRFSplat Baselines Method T3Bench-200 DL3DV-200 WorldScore-200 Time Cost Q-Align IQA Q-Align IAA CLIP IQA+ CLIP Aesthetic CLIP Score Q-Align IQA Q-Align IAA CLIP IQA+ CLIP Aesthetic CLIP Score Q-Align IQA Q-Align IAA CLIP IQA+ CLIP Aesthetic the method is the best , second best , or third best on this metric.
capable of generating structurally rich and precise objects in complex scenes, even under large camera movements. SplatFlow and VideoRFSplat also suffer from blurry artifacts and have difficulty reproducing fine details, such as those found in floors and grass. In comparison, our model generates realistic details while maintaining semantic consistency with the input text prompt.
We further perform a comprehensive quantitative evaluation for this task. Specifically, we sample 600 text prompts from T3Bench (He et al., 2023), DL3DV (Ling et al., 2024), and WorldScore (Duan et al., 2025), covering object-centric and general scenes. As all compared methods are based on 3D Gaussian representations, metrics related to camera control and 3D consistency are not applicable in this setting. Accordingly, we concentrate on the quality evaluation metrics utilized, including CLIP IQA+ (Wang et al., 2023), CLIP Aesthetic (Schuhmann, 2022), the text-image alignment score (CLIP Score) (Hessel et al., 2021), as well as the latest LMM-based Q-Align (Wu et al., 2024) image quality metric. The quantitative results are summarized in Tab. 1. It is evident that our model achieves superior performance on the majority of quality evaluation metrics. For CLIP-Aesthetic, we note that this metric sometimes favor smooth outputs, which may not always align with the detailed and realistic results produced by our method. Our method also attains the highest CLIP Score for two subsets, demonstrating the strong text alignment ability of our method. In addition, we report the average time required to generate a single scene for each method on a single H20 GPU. Our method demonstrates a substantial speed advantage over other approaches. Remarkably, this efficiency is
Ours WonderWorld LucidDreamer WonderJourney Figure 6: 3D scene generation results of different methods on WorldScore benchmark. Method 3D Consistency Photometric Consistency Object Control Content Alignment Style Consistency Subjective Quality Average Time Cost WonderJourney 80.60 79.03 34.81 38.37 67.52 61.49 60.30 6 min LucidDreamer 90.37 90.20 43.48 59.41 66.41 48.02 66.32 6 min WonderWorld 86.91 85.56 52.09 56.82 75.92 41.28 66.43 10 sec Ours 85.87 86.72 49.61 53.96 81.52 54.63 68.72 9 sec
Table 2: Quantitative comparison on WorldScore benchmark. Note that the time cost of the baselines is tested on 1× H100 GPU, while our time cost is tested on 1× H20 GPU.
maintained even when our method produces results with higher resolution and a greater number of frames. In addition, our approach leverages a unified model that seamlessly handles both image-to-3D and text-to-3D tasks without requiring separate training processes. This unified framework not only simplifies the overall workflow but also substantially reduces the training cost.
this section cite: ['b15', 'b30', 'b53', 'b43']

Section: COMPARISON ON WORLDSCORE BENCHMARK
We further conduct a comprehensive evaluation on the recent WorldScore (Duan et al., 2025) benchmark. The static subset of WorldScore comprises 2,000 test examples, encompassing a diverse array of worlds with varying styles, scenarios, and objects. Each test case provides an input image, a text prompt, and a camera trajectory as conditions for generation. The evaluation protocol is designed to assess two primary aspects of world generation: controllability and quality. For baselines, we select three state-of-the-art 3D generation methods: WonderJourney (Yu et al., 2024a), which iteratively completes novel-view images and depth maps based on point clouds; LucidDreamer (Chung et al., 2023), which also performs iterative novel view completion but utilizes 3DGS for rendering; and WonderWorld (Yu et al., 2025), which improves generation quality through the use of layered Gaussian surfels. Since our comparison focuses exclusively on 3D generation methods, the "Camera Control" metric primarily reflects the robustness of the evaluation protocol for each method, and is thus less informative in this context. Accordingly, we omit this metric from our comparison. Additionally, the original WorldScore benchmark evaluates most metrics only on anchor frames, which is suboptimal for 3D world generation tasks that require novel view synthesis. To ensure a fairer comparison, we re-evaluate these metrics by randomly sampled frames within specific intervals. Qualitative and quantitative comparisons are shown in Fig. 6 and Tab. 2, respectively. Our method achieves the highest average score and the fastest inference speed among all compared approaches.
In particular, our model achieves the best "Style Consistency" and secures the second place in "Pho-
Config. T3Bench-200 DL3DV-200 WorldScore-200 Q-Align IQA Q-Align IAA CLIP IQA+ CLIP Aesthetic CLIP Score Q-Align IQA Q-Align IAA CLIP IQA+ CLIP Aesthetic CLIP Score Q-Align IQA Q-Align IAA CLIP IQA+ CLIP Aesthetic CLIP Score A 3.11 2.03 0.41 4.36 25.34 2.64 2.09 0.39 4.60 24.49 2.48 2.10 0.35 4.78 27.40 B 2.61 1.68 0.37 4.11 22.92 2.71 1.96 0.40 4.54 22.71 2.74 2.16 0.33 4.83 26.11 C 3.46 2.12 0.45 4.42 26.95 2.99 2.05 0.42 4.57 26.41 3.06 2.18 0.42 4.92 28.71 D 4.12 2.31 0.52 4.52 27.59 4.02 2.35 0.51 4.80 27.90 3.90 2.71 0.51 5.12 29.09 E 3.98 2.50 0.53 4.58 27.04 3.89 2.35 0.50 4.82 27.45 3.66 2.56 0.47 4.95 28.76 F 4.12 2.26 0.54 4.49 27.68 3.96 2.27 0.50 4.77 27.63 3.76 2.55 0.49 5.08 29.13
Table tometric Consistency", "Object Control", and "Subjective Quality", reflecting a well-balanced and robust capability across controllability and quality. While our method yields relatively lower scores in "3D Consistency" and "Content Alignment", these results can be attributed to methodological differences: for "3D Consistency", all baselines utilize monocular depth estimation models that are closely aligned with the evaluation protocol, whereas our approach relies solely on RGB supervision without explicit depth guidance; for "Content Alignment", our method does not directly manipulate the anchor frame content, in contrast to the baselines. Qualitative analysis further reveals that baseline methods frequently exhibit unnatural transitions, discontinuous content, and visible holes in the generated scenes, which may not be fully reflected by the current metrics. Overall, our approach demonstrates superior consistency and faithful generation over existing methods.
this section cite: ['b5', 'b70']

Section: ABLATION STUDY
In Fig. 2, we show the generation results of various ablation models for image-to-3D scene generation.
The outcomes align well with our expectations: both the MV-oriented diffusion model (w/ MV-Diff) and the MV-oriented distillation model (w/ MV-Dist) exhibit noisy 3D reconstruction due to multiview inconsistency, while the 3D-oriented diffusion model (w/ 3D-Diff) produces blurry visual results.
To further validate the effectiveness of each proposed strategy, we conduct more comprehensive ablation studies on text-to-3D scene generation. Quantitative and qualitative results are summarized in Tab. 3 and Fig. 7, respectively. Consistently, the first three ablation models continue to demonstrate worse visual quality and weaker text alignment. The model without cross-mode consistency loss (w/o CMC) achieves competitive, and in some cases superior, scores on most quantitative metrics compared to our full model. However, qualitative analysis reveals that this model is susceptible to floating and duplicated artifacts. The model without out-of-distribution data (w/o OOD) is more prone to semantic misalignment (e.g., "field") and exhibits a drop in quantitative text alignment metrics. This issue is exacerbated on T3Bench and WorldScore, which differ in distribution from the original multi-view data, highlighting the importance of incorporating OOD data to improve generalization.
this section cite: []

Section: CONCLUSION
We propose an efficient yet powerful model for 3D scene generation, named FlashWorld. At the core of our approach is a novel distillation strategy, which transfers high visual fidelity from a multi-view-oriented diffusion model to a 3D-oriented multi-view generative model endowed with perfect 3D consistency. To achieve this, we design a dual-mode pre-training phase and a cross-mode post-training phase, and introduce an out-of-distribution data co-training strategy to boost the model's generalization. Our method achieves state-of-the-art performance on multiple tasks, while offering significant advantages in inference speed. The efficiency and effectiveness of our approach are well-positioned to advance applications of 3D scene generation. Future work includes incorporating autoregressive generation and extending our framework to dynamic 4D scene generation tasks.
this section cite: []

Section: References
Ref_id:b0 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b1 Title: Diffdreamer: Towards consistent unsupervised single-view scene extrapolation with conditional diffusion models Year: (2023)
Ref_id:b2 Title: Baking gaussian splatting into diffusion denoiser for fast and scalable single-stage image-to-3d generation Year: (2024)
Ref_id:b3 Title: I-design: Personalized llm interior designer Year: (2024)
Ref_id:b4 Title: Video depth anything: Consistent depth estimation for super-long videos Year: (2025)
Ref_id:b5 Title: Domain-free generation of 3d gaussian splatting scenes Year: (2023)
Ref_id:b6 Title: Global-local tree search in vlms for 3d indoor scene generation Year: (2025)
Ref_id:b7 Title: Worldscore: A unified evaluation benchmark for world generation Year: (2025)
Ref_id:b8 Title: Layoutgpt: Compositional visual planning and generation with large language models Year: (2023)
Ref_id:b9 Title: Scenescape: Text-driven consistent scene generation Year: (2023)
Ref_id:b10 Title: Cat3d: create anything in 3d with multi-view diffusion models Year: (2024)
Ref_id:b11 Title: Splatflow: Multi-view rectified flow model for 3d gaussian splatting synthesis Year: (2025)
Ref_id:b12 Title: Videorfsplat: Direct scene-level text-to-3d gaussian splatting generation with flexible pose and multi-view joint modeling Year: (2025)
Ref_id:b13 Title: Generative adversarial networks Year: (2020)
Ref_id:b14 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b15 Title: T3bench: Benchmarking current progress in text-to-3d generation Year: (2023)
Ref_id:b16 Title: CLIPScore: a referencefree evaluation metric for image captioning Year: ()
Ref_id:b17 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b18 Title: Text2room: Extracting textured 3d meshes from 2d text-to-image models Year: (2023)
Ref_id:b19 Title: Video pose engine for 3d geometric perception Year: (2025)
Ref_id:b20 Title: Team HunyuanWorld. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels Year: (2025)
Ref_id:b21 Title: Feed-forward 3d gaussian splatting from unconstrained views Year: (2025)
Ref_id:b22 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023-07)
Ref_id:b23 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b24 Title: Rgbd2: Generative scene synthesis via incremental view inpainting using rgbd diffusion models Year: (2023)
Ref_id:b25 Title: Dual3d: Efficient and consistent text-to-3d generation with dual-mode multi-view latent diffusion Year: (2024)
Ref_id:b26 Title: Real-world camera trajectory and 3d scene generation from text Year: (2024)
Ref_id:b27 Title: Wonderland: Navigating 3d scenes from a single image Year: (2025)
Ref_id:b28 Title: Diffusion adversarial post-training for one-step video generation Year: (2025)
Ref_id:b29 Title: Autoregressive adversarial post-training for real-time interactive video generation Year: (2025)
Ref_id:b30 Title: Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision Year: (2024)
Ref_id:b31 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b32 Title: Knowledge distillation in iterative generative models for improved sampling speed Year: (2021)
Ref_id:b33 Title: Latent consistency models: Synthesizing high-resolution images with few-step inference Year: (2023)
Ref_id:b34 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2020)
Ref_id:b35 Title: Generating interactive 3d world in 0.72 seconds Year: (2025)
Ref_id:b36 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b37 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b38 Title: Anigs: Animatable gaussian avatar from a single image with inconsistent gaussian reconstruction Year: (2025)
Ref_id:b39 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b40 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b41 Title: Fast high-resolution image synthesis with latent adversarial diffusion distillation Year: (2024)
Ref_id:b42 Title: Adversarial diffusion distillation Year: (2024)
Ref_id:b43 Title: Clip+ mlp aesthetic score predictor Year: (2022)
Ref_id:b44 Title: Multi-view diffusion for 3d generation Year: (2023)
Ref_id:b45 Title: Realmdreamer: Text-driven 3d scene generation with inpainting and depth diffusion Year: (2025)
Ref_id:b46 Title: Consistency models Year: (2023)
Ref_id:b47 Title: Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion Year: (2024)
Ref_id:b48 Title: Generating 3d scenes in seconds Year: (2025)
Ref_id:b49 Title: Mvdiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion Year: (2023)
Ref_id:b50 Title: Cycle3d: High-quality and consistent image-to-3d generation via generationreconstruction cycle Year: (2025)
Ref_id:b51 Title: From an image to a scene: Learning to imagine the world from a million 360 videos Year: (2024)
Ref_id:b52 Title: Open and advanced large-scale video generative models Year: (2025)
Ref_id:b53 Title: Exploring clip for assessing the look and feel of images Year: (2023)
Ref_id:b54 Title: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction Year: (2021)
Ref_id:b55 Title: History-aware novel view streaming without temporal training Year: (2025)
Ref_id:b56 Title: Q-align: Teaching lmms for visual scoring via discrete text-defined levels Year: ()
Ref_id:b57 Title: Miqp-based layout design for building interiors Year: (2018)
Ref_id:b58 Title: Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos Year: (2024)
Ref_id:b59 Title: Em distillation for one-step diffusion models Year: (2024)
Ref_id:b60 Title: Constraint-based automatic placement for scene composition Year: (2002)
Ref_id:b61 Title: Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model Year: (2023)
Ref_id:b62 Title: Depth anything: Unleashing the power of large-scale unlabeled data Year: (2024)
Ref_id:b63 Title: Depth anything v2 Year: (2024)
Ref_id:b64 Title: Prometheus: 3d-aware latent diffusion models for feed-forward text-to-3d scene generation Year: (2025)
Ref_id:b65 Title: Language guided generation of 3d embodied ai environments Year: (2024)
Ref_id:b66 Title: Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation Year: (2025)
Ref_id:b67 Title: Improved distribution matching distillation for fast image synthesis Year: (2024)
Ref_id:b68 Title: One-step diffusion with distribution matching distillation Year: (2024)
Ref_id:b69 Title: Wonderjourney: Going from anywhere to everywhere Year: (2024)
Ref_id:b70 Title: Wonderworld: Interactive 3d scene generation from a single image Year: (2025)
Ref_id:b71 Title: Make it home: automatic optimization of furniture arrangement Year: (2011)
Ref_id:b72 Title: Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis Year: (2024)
Ref_id:b73 Title: Mvimgnet: A large-scale dataset of multi-view images Year: (2023)
Ref_id:b74 Title: Text2nerf: Text-driven 3d scene generation with neural radiance fields Year: (2024)
Ref_id:b75 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b76 Title: 3d-scenedreamer: Text-driven 3d-consistent scene generation Year: (2024)
Ref_id:b77 Title: Genxd: Generating any 3d and 4d scenes Year: (2025)
Ref_id:b78 Title: Stereo magnification: Learning view synthesis using multiplane images Year: (2018)
Ref_id:b79 Title: Videomv: Consistent multi-view generation based on large video generative model Year: (2024)
