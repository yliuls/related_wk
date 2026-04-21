Title: LET FEATURES DECIDE THEIR OWN SOLVERS: HYBRID FEATURE CACHING FOR DIFFUSION TRANSFORMERS
Abstract: Diffusion Transformers offer state-of-the-art fidelity in image and video synthesis, but their iterative sampling process remains a major bottleneck due to the high cost of transformer forward passes at each timestep. To mitigate this, feature caching has emerged as a training-free acceleration technique that reuses hidden representations. However, existing methods often apply a uniform caching strategy across all feature dimensions, ignoring their heterogeneous dynamic behaviors. Therefore, we adopt a new perspective by modeling hidden feature evolution as a mixture of ODEs across dimensions, and introduce HyCa, a Hybrid ODE solver inspired caching framework that applies dimension-wise caching strategies. HyCa achieves near-lossless acceleration across diverse tasks and models, including 5.55× speedup on FLUX, 5.56× speedup on HunyuanVideo, 6.24× speedup on Qwen-Image and Qwen-Image-Edit without retraining. Our Project Page.

Section: 
Diffusion Transformers (DiTs) have recently achieved impressive success across image and video generation tasks, demonstrating strong modeling capacity and generation quality. However, the iterative nature of diffusion sampling presents a significant bottleneck, as each output demands multiple transformer passes. This high computational cost hinders deployment in scenarios with strict latency or resource constraints, driving the ongoing research on efficient inference methods.
To address this challenge, two primary acceleration directions have emerged: reducing the total number of sampling steps via algorithmic advancements (Lu et al., 2022a), and lowering the cost of each step through architectural optimization (Yuan et al., 2024;Zhao et al., 2024). Among these,
this section cite: ['b38', 'b41']

Section: Hunyuan Video
Qwen-Image training-free feature caching has emerged as a promising solution. It exploits the temporal coherence of hidden representations by reusing features, thereby reducing redundant computation. Early works such as DeepCache (Ma et al., 2024) demonstrated the feasibility of this idea in U-Net backbones, recent methods such as FORA (Selvaraju et al., 2024), ToCa (Zou et al., 2024a), TaylorSeer (Liu et al., 2025a) extended caching to transformer-based architectures and showed that feature caching can be effectively viewed as solving the temporal evolution of hidden features. Despite progress, current approaches are still limited in critical ways.
Existing methods implicitly assume that all hidden dimensions evolve under a single, unified system. However, this assumption is untenable in DiTs, where the feature space is high-dimensional and exhibits complicated behaviors. Such complexity is unlikely to be captured by a single process. To further investigate, we analyze how each feature dimension changes over timesteps and group them into clusters based on their dynamics. As shown in Fig. 2(a), some dimensions fluctuate sharply with oscillatory patterns, indicating stiffness or multimodal behavior, while others evolve smoothly and predictably, reflecting stable dynamics on Fig. 2(b). These observations suggest that the feature space of DiTs is better described as a complex system, where different groups of dimensions follow distinct temporal patterns, highlighting the need for tailored solvers rather than a one-size-fits-all approach.
Therefore, we introduce HyCa, a hybrid caching framework that models hidden feature evolution as a mixture of ODEs and applies suitable solvers for every dimension. Hyca begins with unsupervised clustering, grouping dimensions with similar dynamic behaviors, and modeling them into a shared ODE. Then, HyCa assigns the most suitable solver to each cluster. Normally, identifying the best solver would require running inference on a large set of images and comparing quantitative metrics. However, surprisingly, we found that cluster assignments are highly stable across resolutions, timesteps, and even prompts. As shown in Fig. 2(c)(d), this invariance allows us to evaluate solver performance on a single prompt at a single timestep to reliably identify the best solver, achieving results comparable to large-scale evaluation. Thus, with "One-Time Choosing" performed offline for each model, "All-Time Solving" becomes possible without any additional cost during inference.
HyCa provides robust and adaptive feature prediction across diverse tasks and architectures. Without retraining, it achieves near-lossless acceleration of 5.56× on FLUX and Hunyuan Video, 6.24× on Qwen-Image and Qwen-Image-Edit. Moreover, it is also fully compatible with distillation, reaching up to 24.4× speedup on FLUX and 12.2× on Qwen-Image while maintaining strong image quality. In summary, our main contributions are:
• Heterogeneous Feature Dynamics. We show that feature dimensions in DiTs do not follow a single unified system but exhibit heterogeneous dynamic behaviors that are better described as a mixture of ODEs. Through dynamics clustering analysis across multiple settings, we further reveal that these cluster's distributions are consistent and input-invariant.
• HyCa Framework. Inspired by hybrid ODE solvers in numerical analysis, we propose HyCa, a training-free framework that groups feature dimensions by their dynamics and automatically assigns the most suitable solver to each group, with minimal overhead.
• Outstanding Performance. We evaluate HyCa across diverse architectures and tasks, including Drawbench on FLUX and Qwen-Image, Vbench on HunyuanVideo, GEdit-Bench on Qwen-Image-Edit, and even distilled models. In all settings, HyCa delivers state-of-the-art performance.
this section cite: ['b22', 'b28']

Section: RELATED WORK
Diffusion models (Sohl-Dickstein et al., 2015;Ho et al., 2020) have achieved strong image/video generation quality. Early U-Net backbones (Ronneberger et al., 2015) faced scaling limits that Diffusion Transformers (DiT) (Peebles & Xie, 2023b) alleviated, enabling rapid progress across modalities and resolutions (Chen et al., 2024b;a;Zheng et al., 2024;Yang et al., 2025). Nevertheless, the iterative nature of sampling remains a key inference bottleneck. Two complementary research lines thus emerge: (i) reducing the number of steps and (ii) reducing the cost per step. Beyond speed, a central challenge is maintaining stability and fidelity under aggressive acceleration, especially when feature dynamics are heterogeneous across dimensions and timesteps.
2.1 SAMPLING TIMESTEP REDUCTION DDIM (Song et al., 2021) introduced deterministic few-step sampling that preserves perceptual quality. Higher-order ODE solvers (DPM-Solver and variants) (Lu et al., 2022a;b;Zheng et al., 2023) improve accuracy-cost trade-offs via multi-step/multi-stage discretizations with carefully controlled local truncation error. Rectified Flow (Liu et al., 2023) shortens transport paths, while distillation (Salimans & Ho, 2022) compresses long trajectories into compact generators. Consistency models (Song et al., 2023) enable few-step synthesis by learning a direct noise-to-clean mapping.
this section cite: ['b30', 'b7', 'b25', 'b44', 'b37', 'b31', 'b42', 'b17', 'b27', 'b32']

Section: DENOISING NETWORK ACCELERATION

this section cite: []

Section: Model Compression Acceleration.
Pruning (Fang et al., 2023;Zhu et al., 2024), quantization (Li et al., 2023b;Shang et al., 2023;Kim et al., 2025), distillation (Li et al., 2024), and token reduction (Bolya & Hoffman, 2023;Kim et al., 2024;Zhang et al., 2024;2025;Cheng et al., 2025) reduce compute with limited runtime overhead. While effective, they typically require additional training and may degrade robustness under domain shifts if the compression is too aggressive.
this section cite: ['b5', 'b45', 'b29', 'b10', 'b13', 'b0', 'b9', 'b39', 'b4']

Section: Feature Caching Acceleration.
Feature caching reuses activations to avoid redundant computation. Early U-Net methods (Li et al., 2023a;Ma et al., 2024) inspired DiT-specific designs: FasterCache (Lv et al., 2025), FORA (Selvaraju et al., 2024), ∆-DiT (Chen et al., 2024c), TeaCache (Liu et al., 2024), and FoCa (Zheng et al., 2025). Dynamic updates (ToCa/DuCa) (Zou et al., 2024a;b), unified cache-prune pipelines (Sun et al., 2025), and region-adaptive sampling (Liu et al., 2025c) further improve efficiency. Among these advances, TaylorSeer (Liu et al., 2025a) exemplifies the cache-thenforecast paradigm by polynomial extrapolation from cached neighbors.
this section cite: ['b22', 'b21', 'b28', 'b14', 'b43', 'b33']

Section: METHOD

this section cite: []

Section: PRELIMINARY
Diffusion Models. Diffusion models (Ho et al., 2020;Song et al., 2021) generate structured data by progressively refining random noise through a series of denoising steps. At each timestep t, the model predicts a conditional Gaussian distribution over x t-1 given x t , where both the mean and variance are parameterized. This generative process can be formulated as:
p θ (x t-1 |x t ) = N x t-1 ; 1 √ α t x t - 1 -α t √ 1 -ᾱt τ θ (x t , t) , β t I ,(1)
where N denotes a normal distribution, α t and β t are noise schedule parameters, and τ θ (x t , t) denotes the model's estimate of the noise component. Sampling begins from a pure noise vector and proceeds by repeatedly drawing samples from these intermediate distributions until a clean image is produced.
this section cite: ['b7', 'b31']

Section: Diffusion Transformer Architecture.
The Diffusion Transformer (DiT) (Peebles & Xie, 2023a) adopts a hierarchical design, expressed as a composition of modules
G = g 1 • g 2 • • • • • g L .
Each module g l consists of a self-attention layer (F l SA ), a cross-attention layer (F l CA ), and a feedforward MLP (F l MLP ). These components are dynamically modulated across timesteps to accommodate the evolving noise levels during generation. The input x t = {x i } H×W i=1 is represented as a sequence of patch tokens. Each module includes a residual update of the form F(x) = x + AdaLN • f (x), where AdaLN (adaptive layer normalization) conditions the normalization parameters on the noise timestep, allowing for more effective denoising across varying noise scales.
Timesteps Dimensions Feature Space of 1 Token t t-1 t-2 t-3 t-4 t-5 t-6 t-7 D1 D2 D3 D4 D5 D6 D7 D8 Indicators for Temporal Dynamics I I1 I2 I3 I4 I5 I6 I7 I8 First-order difference, Second-order difference, Energy, Jerk ratio, Curvature ratio, Spectral flatness, …… Clustering D1 D4 D2 D6 D8 D3 D5 D7 Timesteps t t-1 t-2 t-3 t-4 t-5 t-6 t-7 Cluster1(C1) Cluster2(C2) Cluster3(C3) Clustering based on Indicators e.g., Details of I2 Runge-Kutta (RK) Adams-Moulton (AM) Taylor Fomula (TF) et al.
For each timestep (e.g., t-1) Find the Best Solver for Each Cluster Real Feature t-1 Box of Solvers: Predicted Feature t-1 RK AM TF Compare Each Cluster VS. Min Diffe -rence Compute Skip Skip Skip RK Predictor Compute Skip Skip Skip AM Predictor Compute Skip Skip Skip TF Predictor t t-1 t-2 t-3 t t-1 t-2 t-3 t t-1 t-2 t-3 For D in Cluster1 For D in Cluster2 For D in Cluster3 g., differences, curvature). For each cluster, candidate solvers generate predicted features, then compared against real computed features; the solver with minimum error is then assigned to that cluster. (b) Inference: once assigned, each cluster consistently reuses its solver, enabling efficient prediction by skipping redundant computations while maintaining accuracy.
Feature Caching. Feature caching aims to reduce the cost of diffusion sampling by avoiding repeated computation of hidden features across timesteps. At each timestep t, the model produces hidden features F t = {F l t } L l=1 , and a caching function C(F A , k) estimates features Fk at a future timestep k using cached features. A common strategy is to reuse features from the last computed step:
Fk = C(F t , k) := F t , ∀k ∈ (t, t + n -1],
which provides up to (n-1)× speedup. Recent methods improve reuse by forecasting future features, yet their reliance on a uniform prediction strategy across all dimensions often proves unstable in DiT's complex hidden feature space. In this work, we propose a hybrid approach that assigns suitable solvers for every dimension according to their dynamic behaviors.
this section cite: []

Section: FEATURE CACHING AS HYBRID ODE SOLVING
During reverse-time denoising in diffusion models, the hidden features evolve across timesteps. Let x(τ ) be the latent variable at continuous time τ , and let F(x(τ )) denote the hidden feature extracted from it. Since the generative model is differentiable and x(τ ) follows a continuous reverse-time trajectory, the composite feature map τ → F(x(τ )) is also differentiable. By the chain rule and the probability flow ODE governing x(τ ), the feature dynamics satisfy:
d dτ F(x(τ )) = g θ F(x(τ )), τ ,(3)
where g θ captures the implicit time-dependent vector field induced by the underlying network weights and structure. Although g θ is not directly accessible, we can sample the trajectory {F(x(τ k ))} on a discrete timestep grid, enabling numerical integration using only cached feature values. This perspective naturally casts feature caching as a numerical ODE solving problem. Instead of performing full forward computation at every (discrete) sampler step, we aim to solve the next feature value using prior ones: Ft+1 ≈ Solver(F t , F t-1 , . . . ), (4) where the solver is applied locally to the residual output of each transformer block at skip steps. To accommodate diverse local feature dynamics, from smooth near-linear segments to rapidly varying regions, we adopt a hybrid solver strategy. Concretely, we define a predictor pool S that includes both explicit and implicit numerical solvers with different stability and accuracy properties, including Runge-Kutta (RK), Adams-Bashforth (AB), Taylor Formula (TF), Backward Differentiation Formula (BDF) and Adams-Moulton (AM). This diverse solver set enables HyCa to assign methods tailored to local feature dynamics. Please refer to A.2.2 and A.6 for detailed implementation.
this section cite: []

Section: HYCA FRAMEWORK
Building on this foundation, HyCa is designed as a feature caching framework that models hidden dynamics as a mixture of ODEs and automatically assigns the most suitable solver to each cluster through a one-time optimization procedure. HyCa begins by analyzing the temporal dynamic behavior of each feature dimension. During a probe pass on a single prompt at the first few timesteps, we extract a descriptor vector ϕ d ∈ R k for each feature dimension d ∈ {1, . . . , D}, capturing dynamic indicators such as Jerk ratio and curvature ratio. Then, we apply k-means clustering to obtain a partition {c(d)}, where each dimension d is assigned to a cluster c ∈ {1, . . . , C}. These cluster assignments remain stable across prompts, timesteps and resolutions, thus reused throughout inference.
The resulting clusters represent groups of dimensions that share similar temporal behaviors, enabling solver assignments to be conducted independently for each cluster. Given a solver pool S, HyCa selects the optimal solver s ⋆ c ∈ S for each cluster c by minimizing the average next-step prediction error across all dimensions in that cluster:
min {sc∈S} C c=1 C c=1 1 |c| d∈c F(sc,d) t+1 -F (d) t+1 2 2 ,(5)
where F(sc,d)
t+1
denotes the predicted feature for dimension d at timestep t + 1 using solver s c . This formulation enables per-cluster solver selection via a one-time probing pass, ensuring that HyCa combines efficiency with the adaptability of hybrid solvers.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENT SETTINGS

this section cite: []

Section: Model Configurations.
We conduct experiments on four representative diffusion-based models: the text-to-image models FLUX.1-dev (Labs, 2024) and Qwen-Image (Wu et al., 2025), the text-to-video model HunyuanVideo (Sun et al.), and the image editing model Qwen-Image-Edit (Wu et al., 2025). To further assess compatibility with model compression techniques, we also evaluate our method on distilled models: FLUX.1-schnell and Qwen-Image-Lightning. All models are evaluated under official or recommended configurations on standard public checkpoints.
this section cite: ['b35', 'b35']

Section: Evaluation and Metrics.
For text-to-image generation, we follow the DrawBench (Saharia et al., 2022) protocol and evaluate all models on a fixed set of 200 prompts. We evaluate images using ImageReward (Xu et al., 2023) for photorealism, CLIP Score (Hessel et al., 2021) for text-image alignment, and PSNR, SSIM, LPIPS for fidelity. For text-to-video generation, we evaluate Hunyuan-Video on VBench (Huang et al., 2023), which provides multi-dimensional human-aligned assessments of motion quality, visual appearance, and semantic consistency. For image editing tasks, we use GEdit-Bench (Liu et al., 2025b) to evaluate model performance across a diverse set of edit types and prompts. Unless otherwise specified, all evaluations are conducted using fixed random seeds and default inference settings. Additional implementation details please refer to A.1.    4.3 RESULTS ON TEXT-TO-VIDEO GENERATION As shown in Table 3, our method delivers the best performance on Hunyuan Video. With N =6, it achieves the highest acceleration (5.56× FLOPs reduction) while maintaining a strong VBench score (80.25), marginly lower than the original (80.66) at full 50-step inference. In contrast, TaylorSeer reaches only 4.16× with 79.93, TeaCache drops to 79.36, and DuCa/ToCa degrade further. This demonstrates a superior speed-quality trade-off and strong generalization to video generation. Convert the image to a Japanese manga style. Reference Prompts HyCa Original 10 Steps FORA Change the weather to snow. ×6.24 ×1.00 ×5.00 ×5.00
TaylorSeer Use abstract color blocks and lines to express the composition.
this section cite: ['b26', 'b36', 'b6', 'b8']

Section: Generate a collage-style artwork.
×5.00
this section cite: []

Section: DISCUSSION

this section cite: []

Section: References
Ref_id:b0 Title: Token merging for fast stable diffusion Year: (2023)
Ref_id:b1 Title: Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation Year: (2024)
Ref_id:b2 Title: Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2024)
Ref_id:b3 Title: δ-dit: A training-free acceleration method tailored for diffusion transformers Year: (2024)
Ref_id:b4 Title: Cat pruning: Cluster-aware token pruning for text-to-image diffusion models Year: (2025)
Ref_id:b5 Title: Structural pruning for diffusion models Year: (2023)
Ref_id:b6 Title: Clipscore: A referencefree evaluation metric for image captioning Year: (2021)
Ref_id:b7 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b8 Title: VBench: Comprehensive Benchmark Suite for Video Generative Models Year: (2023-11)
Ref_id:b9 Title: Token fusion: Bridging the gap between token pruning and token merging Year: (2024)
Ref_id:b10 Title: Mincheol Park, and Won Woo Ro. Ditto: Accelerating diffusion model via temporal value similarity Year: (2025)
Ref_id:b11 Title: Faster diffusion: Rethinking the role of unet encoder in diffusion models Year: (2023)
Ref_id:b12 Title: Q-diffusion: Quantizing diffusion models Year: (2023)
Ref_id:b13 Title: Snapfusion: Text-to-image diffusion model on mobile devices within two seconds Year: (2024)
Ref_id:b14 Title: Timestep embedding tells: It's time to cache for video diffusion model Year: (2024)
Ref_id:b15 Title: From reusing to forecasting: Accelerating diffusion models with taylorseers Year: (2025)
Ref_id:b16 Title: Step1x-edit: A practical framework for general image editing Year: (2025)
Ref_id:b17 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b18 Title: Region-adaptive sampling for diffusion transformers Year: (2025)
Ref_id:b19 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b20 Title: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models Year: (2022)
Ref_id:b21 Title: Fastercache: Training-free video diffusion model acceleration with high quality Year: (2025)
Ref_id:b22 Title: Deepcache: Accelerating diffusion models for free Year: (2024)
Ref_id:b23 Title: Scalable Diffusion Models with Transformers Year: (2023-03)
Ref_id:b24 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b25 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b26 Title: Photorealistic text-to-image diffusion models with deep language understanding Year: (2022)
Ref_id:b27 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b28 Title: Fora: Fast-forward caching in diffusion transformer acceleration Year: (2024)
Ref_id:b29 Title: Post-training quantization on diffusion models Year: (2023)
Ref_id:b30 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b31 Title: Denoising diffusion implicit models Year: (2021)
Ref_id:b32 Title: Consistency models Year: (2023)
Ref_id:b33 Title: Unicp: A unified caching and pruning framework for efficient video generation Year: (2025)
Ref_id:b34 Title: Hunyuan-large: An open-source MoE model with 52 billion activated parameters by tencent Year: ()
Ref_id:b35 Title: Qwen-image technical report Year: (2024)
Ref_id:b36 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b37 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2025)
Ref_id:b38 Title: Attention compression for diffusion transformer models Year: (2024)
Ref_id:b39 Title: Token pruning for caching better: 9 times acceleration on stable diffusion for free Year: (2024)
Ref_id:b40 Title: Training-free and hardware-friendly acceleration for diffusion models via similarity-based token pruning Year: (2025)
Ref_id:b41 Title: Real-time video generation with pyramid attention broadcast Year: (2024)
Ref_id:b42 Title: DPM-solver-v3: Improved diffusion ODE solver with empirical model statistics Year: (2023)
Ref_id:b43 Title: Forecast then calibrate: Feature caching as ode for efficient diffusion transformers Year: (2025-08)
Ref_id:b44 Title: Open-sora: Democratizing efficient video production for all Year: (2024-03)
Ref_id:b45 Title: Dip-go: A diffusion pruner via few-step gradient optimization Year: (2024)
Ref_id:b46 Title: Accelerating diffusion transformers with token-wise feature caching Year: (2024)
Ref_id:b47 Title: Accelerating diffusion transformers with dual feature caching Year: (2024)
