Title: LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation
Abstract: We present LeMiCa, a training-free and efficient acceleration framework for diffusion-based video generation. While existing caching strategies primarily focus on reducing local heuristic errors, they often overlook the accumulation of global errors, leading to noticeable content degradation between accelerated and original videos. To address this issue, we formulate cache scheduling as a directed graph with error-weighted edges and introduce a Lexicographic Minimax Path Optimization strategy that explicitly bounds the worst-case path error. This approach substantially improves the consistency of global content and style across generated frames. Extensive experiments on multiple text-to-video benchmarks demonstrate that LeMiCa delivers dual improvements in both inference speed and generation quality. Notably, our method achieves a 2.9× speedup on the Latte model and reaches an LPIPS score of 0.05 on Open-Sora, outperforming prior caching techniques. Importantly, these gains come with minimal perceptual quality degradation, making LeMiCa a robust and generalizable paradigm for accelerating diffusion-based video generation. We believe this approach can serve as a strong foundation for future research on efficient and reliable video synthesis. Our code is available at https://github.com/UnicomAI/LeMiCa

Section: Introduction
Diffusion models [10,38] have made significant advancements in video generation [24,53,45], particularly with DiT-based architectures [29], which greatly enhance visual quality. However, these methods are often hindered by high memory usage, substantial computational costs, and long inference latencies, limiting their use in interactive applications. This has led to increased interest in more efficient and cost-effective generation strategies.
Existing approaches such as model distillation [39,30,42], pruning [7,27], and quantization [34,37,8,18] have been widely adopted to accelerate inference. While effective, these methods require careful architectural design and retraining on large datasets, incurring high costs. Caching mechanisms [35,26], in contrast, offer a retraining-free alternative for accelerating diffusion model inference. The core idea is to reuse model outputs from specific timesteps during sampling to reduce redundant computations and speed up the process [20,44]. Selecting optimal cache timesteps, while balancing video quality and inference speed, remains an open problem in video generation.
Ideally, a lossless video acceleration method should meet two essential criteria: (i) High visual quality and (ii) Consistency between accelerated and original videos. However, existing cache-based methods [20,51] maintain a certain level of visual quality, but they often introduce content deviations and loss of high-frequency details, increasing the risk of uncontrolled degradation.
Timesteps 𝒙 𝑻 𝒙 𝟎 𝒙 𝑻 𝒙 𝑻-𝟏 𝒙 𝑻-𝟐 𝒙 𝟎 … Original Traditional LeMiCa No Cache Local-Greedy Cache Global Outcome-Aware Cache B = 30 B = 12 B = 10 𝒘 𝒊𝒋 Directed Acyclic Graph (DAG) Traditional (TeaCache) LeMiCa Original Speed up (w/o) Speed up 2.1x
this section cite: ['b9', 'b37', 'b23', 'b52', 'b44', 'b28', 'b38', 'b29', 'b41', 'b6', 'b26', 'b33', 'b36', 'b7', 'b17', 'b34', 'b25', 'b19', 'b43', 'b19', 'b50']

Section: Speed up 2.64x
Figure 1: Comparison between our globally controlled cache mechanism (LeMiCa) and traditional local greedy cache methods. Top: The second row shows the traditional Local-Greedy approach, which uses local error estimation and fixed thresholds for caching decisions. It assumes uniform denoising contributions across time steps and ignores temporal heterogeneity and error propagation.
Our method (third row) introduces a Global Outcome-Aware Cache, evaluating cache segment impacts through multiple prompts along a fixed sampling path, creating a static directed acyclic graph (DAG). We then use Lexicographic MiniMax Path Optimization (LeMiCa) to find the optimal cache path under a fixed inference budget (B, model forward steps). Bottom: LeMiCa outperforms traditional methods (e.g., TeaCache) in maintaining structural consistency with faster inference and better control over cache errors and distortions.
Upon further analysis, we identify two key limitations. First, representative methods [44,20] typically compute local errors between adjacent timesteps and apply fixed thresholds to decide whether to cache. However, the diffusion denoising process exhibits significant temporal heterogeneity, with varying noise levels and semantic richness across timesteps. Applying a uniform threshold throughout the process may disrupt semantic alignment and introduce inconsistencies in decision-making, leading to inaccurate caching behavior. Second, these methods mainly focus on minimizing local differences between consecutive steps-what we refer to as Local-Greedy error. While this may reduce shortterm discrepancies, it overlooks how small errors accumulate over time, potentially resulting in a dual loss in both video quality and content consistency. These issues are evident in TeaCache (a state-of-the-art Local-Greedy method), as shown in Figure 1, particularly with the three frames in the top and second rows, where caching introduces noticeable content deviations and visual quality degradation.
To address these limitations, we propose Lexicographic Minimax Caching (LeMiCa), a static caching framework that is model-agnostic and architecture-independent. Instead of using local greedy strategies, LeMiCa treats cache scheduling as a global path planning problem. This is based on the observation that well-trained diffusion models remain stable along a fixed sampling path.
LeMiCa takes a global view of error by introducing the Global Outcome-Aware error, which quantifies the impact of each cache segment on the final output, effectively eliminating temporal heterogeneity and mitigating error propagation. Based on this metric, LeMiCa constructs a Directed Acyclic Graph (DAG), where each edge represents a possible cache segment and is weighted by its global impact on output quality. This graph is generated offline using multiple prompts and full sampling trajectories.
We then apply lexicographic minimax optimization to identify the path that minimizes worst-case degradation. Among all feasible paths under a fixed budget, the one with the smallest maximum error is selected. If multiple paths have the same maximum error, the next largest error is compared, and so on. This strategy explicitly constrains the worst-case error, effectively preventing global degradation caused by locally unstable cache decisions, and significantly improving content consistency and video quality in accelerated generation.
In summary, the contributions of this paper are:
• We propose LeMiCa, a novel, training-free cache scheduling framework that formulates the generation process as a globally optimized DAG traversal task, offering a principled alternative to heuristic and locally greedy approaches.
• We conduct an in-depth analysis of the cache optimization problem and appropriately introduce the Lexicographic Minimax Path Optimization strategy to solve the graph under a fixed cache budget, effectively suppressing error peaks and enhancing global consistency.
• Experiments show that, compared to existing cache techniques, ours achieves dual improvements in inference speed and generation quality across various base models, such as a 2.9X speedup on Latte and an LPIPS of 0.05 on Open-Sora.
2 Related Work Diffusion Model Acceleration. Diffusion models exhibit strong versatility across domains, but their iterative nature incurs high computational costs, positioning inference acceleration as a central research challenge. Current efforts to accelerate diffusion model sampling focus primarily on reducing sampling steps via schedulers. Denoising Diffusion Implicit Models (DDIM) [38] represents one of the earliest attempts to accelerate sampling by extending the original Denoising Diffusion Probabilistic Model (DDPM) [10] to non-Markovian settings. The Efficient Denoising Model (EDM) [13] introduces a design framework that optimizes specific aspects of the diffusion process. Concurrently, there is growing attention to more efficient and accurate methods for solving stochastic differential equations (SDEs) and ordinary differential equations (ODEs) [40,12,21,3]. Other approaches introduce knowledge distillation [9], training a student model to condense the multi-step outputs of the original diffusion model into fewer steps [22], including Progressive Distillation [30], Consistency Distillation [39,14,6,42,52], Adversarial Diffusion Distillation [32,31], and Score Distillation Sampling [47,46]. Additionally, methods such as quantization [17,36,34], pruning [7,27], optimization [19], and parallelism [50,15,5,4] have been proposed and applied to various diffusion-based generative tasks. However, these methods often require large amounts of computational resources and data for training or intricate engineering designs, which increases the complexity of their application.
Cache in Diffusion Models. Caching mechanisms [35] have recently attracted attention as a retraining-free alternative for accelerating diffusion model inference [44,25]. The core idea is to reuse model outputs from certain timesteps during sampling to reduce redundant computations [33]. DeepCache [26] accelerates the Unet structure using manually set rules. T-GATE [49] and ∆-DiT [2] apply this idea to DiT-based networks [29], achieving advanced image generation acceleration [54,16].
With the breakthrough of Sora [28] in video generation, researchers have extended this acceleration concept from image generation to video generation. In this context, PAB [51] observed a U-shaped pattern in attention differences across timesteps in the diffusion process, and based on this, proposed a strategy to cache and broadcast intermediate features at various timestep intervals. FasterCache [23] realized the significant redundancy in conditional generation (CFG) and further enhanced inference speed by utilizing a dynamic feature-based caching mechanism. TeaCache [20] leverages the correlation between timestep embeddings and model outputs, incorporating threshold-based indicators and polynomial fitting to guide caching. Although these methods have improved the efficiency of diffusion-based generation, the core challenge remains in how to accelerate inference while maintaining content consistency and preserving details.
this section cite: ['b43', 'b19', 'b37', 'b9', 'b12', 'b39', 'b11', 'b20', 'b2', 'b8', 'b21', 'b29', 'b38', 'b13', 'b5', 'b41', 'b51', 'b31', 'b30', 'b46', 'b45', 'b16', 'b35', 'b33', 'b6', 'b26', 'b18', 'b49', 'b14', 'b4', 'b3', 'b34', 'b43', 'b24', 'b32', 'b25', 'b48', 'b1', 'b28', 'b53', 'b15', 'b27', 'b50', 'b22', 'b19']

Section: Method

this section cite: []

Section: Background: Denoising Diffusion Models
Denoising Diffusion Models achieve generative modeling by simulating the gradual noising and denoising process of data. The core of these models consists of two key stages: diffusion and denoising. During the forward diffusion process, the model starts from a real sample x 0 ∼ q(x) and gradually adds Gaussian noise over T timesteps. The noised sample x t at timestep t is given by:
x t = √ α t x t-1 + √ 1 -α t z t , z t ∼ N (0, I), t = 1, . . . , T,(1)
where α t controls the noise strength at each step. As t increases, the samples converge to a standard normal distribution N (0, I). In the reverse denoising process, the model reconstructs the original data distribution by iteratively denoising through a neural network. The conditional probability for each step is modeled as:
p θ (x t-1 | x t ) = N (x t-1 ; µ θ (x t , t), Σ θ (x t , t)),(2)
where µ θ and Σ θ are learned mean and covariance functions. Due to the multi-step nature of denoising, diffusion models typically incur significant computational overhead during generation. The traditional Local-Greedy (L1 rel ) strategy uses fixed thresholds on local output differences between adjacent timesteps to decide when to cache. This assumes uniform temporal sensitivity, which can be misleading-for instance, caching at t 2 yields lower final error than t 1 , despite t 1 seeming smoother locally. This highlights the role of temporal heterogeneity. (b) Our Global Outcome-Aware (segment-wise error) strategy estimates final output error when caching outputs over segments of length len, starting from timestep i. The plot shows that early caches cause greater error, supporting an outcome-sensitive, trajectory-aware strategy over fixed local heuristics.
this section cite: []

Section: Rethinking Cache in Diffusion Sampling
Traditional cache reuse in diffusion sampling typically adopts a Local-Greedy strategy (Figure 2a), where caching is based on local differences between adjacent model outputs, often measured by the relative L1 distance [20]:
L1 rel (O, t) = ∥O t -O t+1 ∥ 1 ∥O t+1 ∥ 1(3)
where O t is the output at timestep t. High local differences prompt full inference; low differences lead to cache reuse. This step-wise strategy assumes uniform importance across timesteps.
However, diffusion processes are inherently temporally heterogeneous-early steps shape global structure, while later steps refine details. Thus, as illustrated in Figure 2a, a seemingly minor change at an early step (e.g., t 1 ) can have a larger impact on the final output than a larger change at a later step (e.g., t 2 ). Local metrics fail to account for this asymmetric error propagation, motivating a rethinking of cache strategies.
To address this, we propose a Global Outcome-Aware view that considers the long-term impact of cache reuse over time. Specifically, we define a cache segment (i, j) means full inference is performed at timesteps i and j, while all intermediate steps t ∈ (i, j) reuse cached outputs:
L1 glob (i → j) = 1 N x cache(i→j) 0 -x original 0 1 (4
)
Here, x original 0 is the output with no caching, and
x cache(i→j) 0
is the output with segment-level cache. As shown in Figure 2b, the global error depends not just on segment length but also on its temporal position-early caches induce amplified downstream errors, while later caches are less disruptive.
These findings reveal two key insights: (1) Global error propagation is non-uniform and timedependent, invalidating fixed-threshold heuristics; (2) The position of the cache segment matters more than its length. Building on these insights, we formulate cache planning as a graph-based constrained path optimization problem over the sampling trajectory.
this section cite: ['b19']

Section: Lexicographic Minimax Path Caching
Based on the rethinking of cache in Sec 3.2, we propose LeMiCa, a method that integrates sparse directed graph construction with optimal graph search under peak error control.
this section cite: []

Section: Graph Construction.
We construct a directed acyclic graph, as shown in Figure 1, where each edge represents a candidate cache segment along the original sampling trajectory. To reduce complexity, we impose a maximum skip length based on the prior that long-range reuse typically leads to large errors, thus avoiding full graph construction. Edge weights are evaluated by replaying cached segments using intermediate states from a full denoising pass. To ensure generality, we build a static graph by averaging edge errors across diverse prompts and noise seeds.
this section cite: []

Section: Graph Optimization.
Given a directed acyclic graph G with globally error-weighted edges, we frame the caching problem as selecting a path from source s to target t that includes exactly B full computation steps and an arbitrary number of cached segments. This budget-constrained formulation allows flexible reuse while bounding computational cost.
As shown in Figure 2b, early-stage cache errors amplify exponentially during denoising, while late-stage errors remain more localized. This asymmetric error propagation renders traditional shortest-path heuristics-which minimize only additive cost-suboptimal, as they fail to control the dominant sources of degradation.
To better address this imbalance, we adopt a lexicographic minimax criterion that explicitly minimizes the highest cache error along the path, followed by the second highest, and so on. Unlike training-based approaches such as ShortDF [3], which directly seek the shortest error path, our formulation-commonly used in control systems for robust worst-case optimization-offers improved stability in error-sensitive settings. Formally, the optimization problem is defined as:
min P ∈P (B) s→t LexMax (sort_desc ({w(e) | e ∈ P cache }))(5)
Here, P
s→t denotes the set of all paths from s to t with exactly B full steps, and P cache ⊂ P are the cached segments within a given path P . The operator LexMax lexicographically minimizes the sorted error vector, ensuring worst-case robustness. The detailed algorithm pseudocode is provided in the Appendix (Section A).
this section cite: ['b2']

Section: Experiments

this section cite: []

Section: Experimental Setup
Metrics For fair comparison, we follow prior works and report both efficiency and visual quality metrics. Efficiency is measured by FLOPs and latency. Visual quality is evaluated using VBench [11] (human preference), LPIPS [48] (perceptual similarity), SSIM [43] (structural consistency), and PSNR (pixel-level accuracy).
this section cite: ['b10', 'b47', 'b42']

Section: Baselines and Compared Methods
We evaluate our method on representative diffusion-based video models: Open-Sora [53], Latte [24], and CogVideoX [45]. Baselines include ∆-DiT [2], T-GATE [49], PAB [51], and TeaCache [20]. Among them, T-GATE and ∆-DiT are designed for images, while PAB and TeaCache target video. Accordingly, we compare against PAB and TeaCache on CogVideoX, and against all four baselines on Open-Sora and Latte.
Implementation Details Experiments are conducted on NVIDIA H100 GPUs using PyTorch.
To construct the DAG for Global Outcome-Aware error modeling, we sample 70 prompts (10 per attribute) from T2V-CompBench [41], following standard practice [41,20]. The DAG construction and forward inference use distinct datasets to ensure fair and robust evaluation. Sampling is repeated 10 times with different seeds, and results are averaged to reduce bias.
this section cite: ['b52', 'b23', 'b44', 'b1', 'b48', 'b50', 'b19', 'b40', 'b40', 'b19']

Section: Comparison with State-of-the-Art Methods

this section cite: []

Section: Quantitative Comparison
Table 1 compares LeMiCa with baselines across four metrics: VBench, LPIPS, SSIM, and PSNR. LeMiCa includes two variants: LeMiCa-slow (fidelity-focused) and LeMiCa-fast (speed-focused). It consistently outperforms training-free acceleration baselines across models, schedulers, resolutions, and video lengths. LeMiCa-slow achieves the best reconstruction quality, reducing LPIPS from 0.134 to 0.05 on Open-Sora and from 0.195 to 0.091 on Latte-over 2× improvement vs. TeaCache-slow. LeMiCa-fast improves inference speed from 2.60× to 2.93× on Latte compared to TeaCache-fast, while preserving visual quality. Unlike prior methods relying on online greedy strategies, LeMiCa precomputes its caching policy, eliminating runtime overhead. Overall, LeMiCa provides efficient video generation with minimal perceptual quality degradation.
this section cite: []

Section: Visualization
We compare video acceleration methods from both quality and speed perspectives. As shown in Fig. 3, under the fidelity-focused setting, LeMiCa excels in preserving content consistency and fine details, as highlighted in red boxes. This demonstrates its ability to maintain high-quality visuals even when prioritizing fidelity. In contrast, Fig. 4 illustrates that under the speed-focused se-Original (w/o Speed up) TeaCache LeMiCa (ours) Open-sora Latte CogVideoX Latency: 26.54s Latency: 12.63s (2.10x) Latency: 10.86s (2.44x) Latency: 11.18s Latency: 4.30s (2.60x) Latency: 3.81s (2.93x) Latency: 43.08s Latency: 17.58s (2.45x) Latency:16.48s (2.61x) tting, LeMiCa-fast significantly outperforms TeaCache-fast, achieving superior acceleration rates while still maintaining competitive performance. These results highlight LeMiCa's ability to balance quality and speed across different configurations. Additional qualitative examples can be found in the Appendix (Section E).
this section cite: []

Section: Ablation Studies
Acceleration vs. Performance trade-off Figure 5 presents the quality-latency trade-off between our proposed LeMiCa and TeaCache. To ensure comparable computational budgets, LeMiCa is configured with 19, 12, 9, and 7 inference steps (i.e., inference budget B), corresponding to TeaCache thresholds of 0.1, 0.2, 0.3, and 0.5, respectively. Across all latency regimes, LeMiCa consistently achieves a superior quality-efficiency balance, outperforming TeaCache on all reference-based metrics. Importantly, under extreme acceleration (latencies below 8 seconds), LeMiCa maintains robust and high-quality performance.
this section cite: []

Section: Sample Requirements for Graph Construction
To investigate how many samples LeMiCa requires to offline construct the DAG, we randomly select n ∈ {1, 5, 10, 20} from the original 350 samples (70 prompts × 5 seeds), and compute the optimal caching path under the lexicographic minimax criterion. Each setting is repeated 20 times to reduce randomness. Importantly, distinct datasets are used for DAG construction and forward inference to guarantee fairness and robustness in evaluation. Video quality is then evaluated on 50 selected VBench prompts, with average results reported. Table 2 shows that LeMiCa achieves strong performance with a single sample (e.g., PSNR 24.51), rapidly approaching the upper bound with 10 samples and essentially saturating at 20 samples across all metrics. This demonstrates LeMiCa's ability to construct high-quality cache paths with minimal samples and underscores the robustness of its static caching strategy across varying prompts and seeds. Trajectory Robustness Since the cache mechanism is inherently tied to the original denoising trajectory, it is essential to assess whether a training-free cache method remains effective when the trajectory changes. To this end, we vary the trajectory scale parameter in the sampling schedule from its default value of 1.0 to several alternative values (0.5, 0.75, 1.25, 1.5), introducing different diffusion paths during inference. As shown in Figure 6, the left panel illustrates the effect of trajectory scaling on the denoising paths, while the right panel demonstrates that LeMiCa consistently outperforms the current state-of-the-art method, TeaCache, across all trajectories in terms of LPIPS. These results confirm that our method remains effective even under varying denoising paths. Shortest Path vs. Lexicographic MiniMax Path We compare the performance of the Shortest Path strategy and the Lexicographic MiniMax Path strategy in video reconstruction tasks. As shown in Table 3, the MiniMax Path strategy consistently outperforms the baseline Shortest Path strategy in both VBench scores and reconstruction metrics. This observation is consistent with our analysis: the edge errors cached during the sampling process are not independent and thus cannot be simply accumulated linearly.
this section cite: []

Section: Performance at different resolutions and lengths
Our method incorporates Dynamic Sequence Parallelism (DSP) [51]to support high-resolution long-video generation across multiple GPUs. To assess its sampling acceleration performance across varying video sizes, we conducted tests on videos with different lengths and resolutions. As shown in Figure 7, our method maintains stable acceleration even as video resolution and frame count increase, highlighting its potential for handling longer and higher-resolution videos.
this section cite: ['b50']

Section: Conclusion
We propose LeMiCa, a general and efficient caching framework for accelerating diffusion-based video generation. Unlike locally greedy strategies, LeMiCa formulates cache scheduling as a global path optimization problem using lexicographic minimax over a static DAG, effectively constraining worst-case degradation. With the introduction of the Global Outcome-Aware error, our method captures the long-term impact of caching decisions, mitigating temporal heterogeneity and error accumulation. Extensive experiments demonstrate that LeMiCa consistently improves both efficiency and visual quality across diverse diffusion models. More broadly, LeMiCa offers a new perspective on structured caching in generative modeling, which may inspire future research in other domains such as 3D, multi-view, or multi-modal generation where controllable acceleration remains an open challenge.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2025)
Ref_id:b1 Title: Delta dit: A training-free acceleration method tailored for diffusion transformers Year: (2024)
Ref_id:b2 Title: Optimizing for the shortest path in denoising diffusion model Year: ()
Ref_id:b3 Title: Parallelizing diffusion models by asynchronous denoising Year: (2024)
Ref_id:b4 Title: xdit: an inference engine for diffusion transformers (dits) with massive parallelism Year: (2024)
Ref_id:b5 Title: Consistency models made easy Year: (2024)
Ref_id:b6 Title: Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding Year: (2015)
Ref_id:b7 Title: Ptqd: Accurate post-training quantization for diffusion models Year: (2024)
Ref_id:b8 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b9 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b10 Title: Vbench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b11 Title: Gotta go fast when generating data with score-based models Year: (2021)
Ref_id:b12 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b13 Title: Consistency trajectory models: Learning probability flow ode trajectory of diffusion Year: (2023)
Ref_id:b14 Title: Distrifusion: Distributed parallel inference for high-resolution diffusion models Year: (2024)
Ref_id:b15 Title: Faster diffusion: Rethinking the role of unet encoder in diffusion models. arXiv e-prints Year: (2023)
Ref_id:b16 Title: Q-dm: An efficient low-bit quantized diffusion model Year: (2023)
Ref_id:b17 Title: Q-dm: An efficient low-bit quantized diffusion model Year: (2024)
Ref_id:b18 Title: Oms-dpm: Optimizing the model schedule for diffusion probabilistic models Year: (2023)
Ref_id:b19 Title: Timestep embedding tells: It's time to cache for video diffusion model Year: (2024)
Ref_id:b20 Title: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models Year: (2022)
Ref_id:b21 Title: Latent consistency models: Synthesizing high-resolution images with few-step inference Year: (2023)
Ref_id:b22 Title: Fastercache: Training-free video diffusion model acceleration with high quality Year: (2024)
Ref_id:b23 Title: Latte: Latent diffusion transformer for video generation Year: (2025)
Ref_id:b24 Title: Learning-to-cache: Accelerating diffusion transformer via layer caching Year: (2024)
Ref_id:b25 Title: Accelerating diffusion models for free Year: (2023)
Ref_id:b26 Title: Llm-pruner: On the structural pruning of large language models Year: (2023)
Ref_id:b27 Title:  Year: (2024)
Ref_id:b28 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b29 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b30 Title: Fast high-resolution image synthesis with latent adversarial diffusion distillation Year: (2024)
Ref_id:b31 Title: Adversarial diffusion distillation Year: (2024)
Ref_id:b32 Title: Fora: Fast-forward caching in diffusion transformer acceleration Year: (2024)
Ref_id:b33 Title: Post-training quantization on diffusion models Year: (2023)
Ref_id:b34 Title: Cache memories Year: (1982)
Ref_id:b35 Title: Temporal dynamic quantization for diffusion models Year: (2023)
Ref_id:b36 Title: Temporal dynamic quantization for diffusion models Year: (2024)
Ref_id:b37 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b38 Title: Consistency models Year: (2023)
Ref_id:b39 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b40 Title: T2v-compbench: A comprehensive benchmark for compositional text-to-video generation Year: (2024)
Ref_id:b41 Title: Target-driven distillation: Consistency distillation with target timestep selection and decoupled guidance Year: (2025)
Ref_id:b42 Title: A universal image quality index Year: (2002)
Ref_id:b43 Title: Cache me if you can: Accelerating diffusion models through block caching Year: (2023)
Ref_id:b44 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2024)
Ref_id:b45 Title: Improved distribution matching distillation for fast image synthesis Year: (2024)
Ref_id:b46 Title: One-step diffusion with distribution matching distillation Year: (2024)
Ref_id:b47 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b48 Title: Cross-attention makes inference cumbersome in text-to-image diffusion models Year: (2024)
Ref_id:b49 Title: Dsp: Dynamic sequence parallelism for multi-dimensional transformers Year: (2024)
Ref_id:b50 Title: Real-time video generation with pyramid attention broadcast Year: (2024)
Ref_id:b51 Title: Changxing Ding, Dacheng Tao, and Tat-Jen Cham. Trajectory consistency distillation: Improved latent consistency distillation by semi-linear consistency function with trajectory mapping Year: (2024)
Ref_id:b52 Title: Open-sora: Democratizing efficient video production for all Year: (2024)
Ref_id:b53 Title: LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation Year: (2024)
