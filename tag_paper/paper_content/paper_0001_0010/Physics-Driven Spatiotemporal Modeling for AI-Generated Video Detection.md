Title: Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection
Abstract: AI-generated videos have achieved near-perfect visual realism (e.g., Sora), urgently necessitating reliable detection mechanisms. However, detecting such videos faces significant challenges in modeling high-dimensional spatiotemporal dynamics and identifying subtle anomalies that violate physical laws. In this paper, we propose a physics-driven AI-generated video detection paradigm based on probability flow conservation principles. Specifically, we propose a statistic called Normalized Spatiotemporal Gradient (NSG), which quantifies the ratio of spatial probability gradients to temporal density changes, explicitly capturing deviations from natural video dynamics. Leveraging pre-trained diffusion models, we develop an NSG estimator through spatial gradients approximation and motion-aware temporal modeling without complex motion decomposition while preserving physical constraints. Building on this, we propose an NSG-based video detection method (NSG-VD) that computes the Maximum Mean Discrepancy (MMD) between NSG features of the test and real videos as a detection metric. Last, we derive an upper bound of NSG feature distances between real and generated videos, proving that generated videos exhibit amplified discrepancies due to distributional shifts. Extensive experiments confirm that NSG-VD outperforms state-of-the-art baselines by 16.00% in Recall and 10.75% in F1-Score, validating the superior performance of NSG-VD.

Section: Introduction
The rapid advancement of generative models [1,2,3,4,5,6], particularly diffusion-based frameworks (e.g., Sora [3]), has achieved unprecedented capabilities in synthesizing photorealistic video content. While these breakthroughs enable transformative applications in content creation for creative industries [7,8,9,10], they simultaneously pose critical societal risks through malicious manipulation (e.g., deepfake disinformation [11,12,13,14,15,16,17], synthetic media fraud [7,18]). As AI-generated videos become increasingly realistic in both spatial and temporal domains, developing effective video detection methods becomes critically urgent for preserving societal trust in digital media.
A fundamental challenge for AI-generated video detection lies in modeling the spatiotemporal dynamics of video evolution. Intuitively, natural videos inherently obey physical laws like motion coherence and texture continuity [19,20], while AI-generated videos often exhibit subtle yet systematic inconsistencies in spatiotemporal coherence [21]. This observation raises a crucial question: Motion Appearance (a) Traditional Spatiotemporal Modeling (a) Traditional (Changes in raw video) (Appearance, Motion) Real (b) Physics-Driven Probability Flow (Ours) Fake (, ) + •((, )•(, )) = Video Frames Motion Appearance (, ) (, ) Video Frames Velocity Probability (b) Physics-Driven Spatiotemporal Modeling (Ours) Figure 1: Comparisons of traditional and physics-driven paradigms for spatiotemporal modeling in AI-generated video detection. (a) Traditional methods [22,23,24] often rely on specific artifacts like appearance consistency and optical flow-based motion modeling, struggling with highly realistic content yet physically implausible (e.g., Sora). (b) Our physics-driven approach explicitly models video dynamics via physics conservation laws, effectively identifying violations of physical laws.
Two critical difficulties arise: 1) Video content inherently contains complex spatial domain correlations (e.g., texture structure) and temporal domain dependencies (e.g., motion trajectories), requiring modeling frameworks that jointly capture both spatial structures and temporal dynamics characteristics. 2) AI-generated videos are rapidly approaching the perceptual quality of natural videos, with discrepancies that may become vanishingly subtle in both visual appearance and temporal evolution.
Existing AI-generated video detection methods primarily rely on local feature inconsistencies (e.g., optical flow-based motion modeling [22], appearance consistency modeling [23]) or supervised learning with large-scale datasets [25,24,26,27]. However, they often ignore physics-driven constraints governing spatiotemporal evolution inherent to natural videos. This limitation exhibits inherent vulnerabilities when confronting synthetic anomalies that violate physical laws, e.g., nonphysical motion patterns in Sora-generated videos [3] (Figure 1-a), leading to inferior performance.
In this paper, we propose a physics-driven paradigm based on probability flow conservation principles [28,29]. By modeling video dynamics as fluid mechanics, we formulate video evolution through a probability flow velocity field governed by continuity equations (see Figure 1-b and Section 3.1). This reveals a key insight: natural video dynamics preserve the product between the velocity field and the ratio of spatial probability gradients to temporal density changes. Inspired by this, we introduce a Normalized Spatiotemporal Gradient (NSG) statistic, which quantifies the ratio of spatial probability gradients to temporal density changes. NSG captures fundamental discrepancies in how videos adhere to physical constraints while eliminating reliance on specific artifacts, enabling sensitive detection even when visual differences are imperceptible to humans or conventional models.
To enable practical estimation, we develop an NSG estimator leveraging pre-trained diffusion models' inherent gradient estimation ability [30,31] in Section 3.2. By approximating spatial gradients with learned score functions (i.e., the gradient of the log probability density) from the diffusion models and temporal derivatives through motion-aware temporal dynamics via a brightness constancy constraint [32], our method avoids explicit flow computation while preserving essential physical constraints. This estimator eliminates reliance on complex motion modeling by physics-inspired priors while maintaining sensitivity to subtle spatiotemporal inconsistencies inherent to synthetic content.
Building on this foundation, we propose an NSG-based video detection method (NSG-VD) in Section 3.3, which computes the Maximum Mean Discrepancy (MMD) [33,34] between NSG features of real videos and the test video as the detection metric, as illustrated in Figure 2. We further theoretically derive an upper bound of the distance between NSG features of real and generated data in Section 3.4, showing this bound expands with increasing distribution shifts in generated videos. This implies that the MMD between NSGs of real videos tends to be smaller than that between real and generated videos, establishing the theoretical basis for the effectiveness of NSG-VD. Extensive experiments show that NSG-VD achieves 16.00% higher Recall and 10.75% higher F1-score than baselines, validating the superior performance of NSG-VD. Our contributions are summarized as:
• A physics-driven NSG statistic: We formulate the video evolution through a probability flow velocity field with a continuity equation and propose a novel statistic Normalized Spatiotemporal Gradient (NSG) that explicitly models spatiotemporal dynamics of videos. By quantifying the ratio of spatial probability gradients to temporal density changes, NSG fundamentally captures violations of physical continuity in AI-generated videos without reliance on artifact-specific supervision.
• A diffusion-guided NSG estimation with physical priors: We develop an NSG estimator by spatial gradients approximation and motion-aware temporal dynamics modeling using pre-trained diffusion models. By avoiding explicit flow modeling and instead enforcing brightness constancy constraints, our method achieves effective NSG approximation without domain-specific motion modeling.
• An AI-generated video detection method with theoretical and empirical justifications: We propose an NSG-based video detection method (NSG-VD), which quantifies distributional shifts in NSG features using Maximum Mean Discrepancy (MMD). We derive an upper bound of NSG feature distances between real and generated videos, proving that generated videos exhibit amplified discrepancies under distribution shifts. Empirical results also show the superiority of our NSG-VD.
2 Related Work AI-Generated Video Detection. Early generated video detection methods primarily focus on identifying synthetic facial videos. Yang et al. [35] and Amerini et al. [22] exploit auxiliary facial motion cues (landmark dynamics vs. optical flow) for deepfake detection. Gu et al. [36] separately model spatial and temporal inconsistencies, and introduce a vertical slicing feature fusion mechanism to establish a more comprehensive spatial-temporal representation. Wang et al. [23] propose an alternating-freezing strategy with spatiotemporal augmentation for facial consistency modeling. Xu et al. [24] transform consecutive frames into a predefined layout via masking/resizing to enable efficient spatiotemporal modeling. Peng et al. [37] integrate multi-feature fusion of facial perspectives, textures, and attributes. While most methods utilize facial priors, their reliance on domain-specific features limits their generalizability to more general AI-generated content detection.
With the rapid advancement in video generation, detecting general AI-generated content has become challenging. Bai et al. [38] fuse frame-level and optical flow predictions to detect spatial-temporal anomalies. To jointly capture spatiotemporal cues, Ma et al. [39] and Chen et al. [25] propose Transformer-and mamba-based frameworks to model spatiotemporal relationships in video frame features for detection. Song et al. [40] exploit the cross-modal perception and reasoning in visionlanguage large models to learn general forgery features. Despite this progress, these methods mainly focus on appearance inconsistencies, while overlooking the intrinsic spatiotemporal dynamics cues, thereby struggling to tackle visual cues from diverse video generative models.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b2', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b6', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b21', 'b22', 'b24', 'b23', 'b25', 'b26', 'b2', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b21', 'b35', 'b22', 'b23', 'b36', 'b37', 'b38', 'b24', 'b39']

Section: Diffusion Models.
Diffusion models [41,30,42,31] have emerged as powerful probabilistic generative models, benefiting from their diffusion-denoising paradigm that perturb data into noise through Gaussian processes and reconstruct samples via iterative denoising. Intuitively, the highquality and diverse generative capabilities of diffusion models come from their ability to capture and exploit the distributional characteristics of natural data, enabling effective discrimination between natural samples and outliers. Motivated by this, a growing body of research has leveraged diffusion models for the detection of adversarial [43,44,45] and generated samples [46,47,48], wherein the score model emerges as a powerful discriminative tool. Nevertheless, it remains challenging to simultaneously capture and integrate spatiotemporal features when relying solely on score models. log x, x ~ℚ x ~ℙ ∇ x log x, Inter-frame Differences ∆x ∆ ∆x ∆ Diffusion Model x x Figure 2: Overview of the proposed NSG-VD. Given a reference set of real videos {x re } and a test video x te , we estimate their spatial gradients ∇ x log p(x, t) and temporal derivatives ∂ t log p(x, t) via a pre-trained diffusion model s θ , from which we derive their Normalized Spatiotemporal Gradients (NSGs) and calculate the MMD between NSG features of real and test videos as a detection metric.
Method Overview. To address these challenges, we propose a physics-driven method based on physical conservation principles to model spatiotemporal dynamics and introduce a novel statistic Normalized Spatiotemporal Gradient (NSG), which quantifies the ratio of spatial probability gradients to temporal density changes, capturing subtle anomalies in videos (Section 3.1). Leveraging diffusion models, we develop an effective NSG estimator by spatial gradients approximation and motion-aware temporal dynamics modeling (Section 3.2). Building on this, we develop an NSGbased video detection method (NSG-VD), which computes the Maximum Mean Discrepancy (MMD) between NSG features of the test video and real videos as a detection characteristic (Section 3.3), where its framework is shown in Figure 2. Last, we theoretically show that the MMD between NSGs of real videos tends to be smaller than that between real and generated videos (Section 3.4).
this section cite: ['b40', 'b29', 'b41', 'b30', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47']

Section: Modeling Spatiotemporal Dynamics via Normalized Spatiotemporal Gradient
Detecting AI-generated videos requires capturing both spatial irregularities and temporal inconsistencies in synthetic content. Inspired by conservation laws in physics (e.g., mass or energy transport), we initially formulate the probability flow velocity field v(x, t) to model the evolution of probability density p(x, t), which satisfies a continuity equation for global consistency across spatiotemporal domains. However, solving v faces challenges due to its underdetermined nature (Eqn. (4)). To address this, we propose Normalized Spatiotemporal Gradient (NSG) g(x, t), a dual field statistic of v combining both spatial gradients and temporal dynamics of p(x, t), as defined in Eqn. (6).
Probability Flow Velocity Field v(x, t). We begin to conceptualize the probability flow (also called probability current) as the movement of probability mass of x over time t [50,51]. To formalize this flow, we define the probability flow density J(x, t), analogous to fluid mechanics [52]:
J(x, t) = p(x, t) • v(x, t),(1)
where p(x, t) denotes the probability density and v(x, t) represents the velocity field guiding the flow of probability mass. The conservation of probability mass [28,29] implies the continuity equation:
∂p(x, t) ∂t + ∇ x • J(x, t) = 0,(2)
where ∇ x • J= i ∂Ji ∂xi denotes the divergence of the vector field J [51]. This is not a video-specific assumption but a universal mathematical formulation of probability mass conservation, which holds for any time-evolving probability density p(x, t) [53,29]. Intuitively, this equation shows that the rate of change in probability density ∂ t p at a point equals the difference between inflow (negative divergence) or outflow (positive divergence) of the probability flow J. Substituting J(x, t) into Eqn.
(2), dividing by p(x, t), and applying the chain rule to log p(x, t), yields:
∂ t log p(x, t) + ∇ x • v(x, t) + v(x, t) • ∇ x log p(x, t) = 0.(3)
This expression reveals how the velocity field v(x, t) simultaneously encodes temporal evolution (∂ t log p(x, t)) and spatial gradients (∇ x log p(x, t)) of the probability distribution.
Normalized Spatiotemporal Gradient g(x, t) as Dual Field of v(x, t). To solve v(x, t), we focus on the dominant components of Eqn. (3). Assuming that the divergence term ∇ x • v is subdominant in smoothly varying distributions (e.g., incompressible flow approximations [28,54]), a condition commonly used in fluid dynamics [28] and quantum mechanics [55], Eqn. (3) simplifies to: v(x, t) • ∇ x log p(x, t) ≈ -∂ t log p(x, t).
Considering the non-uniqueness of solutions to v(x, t) in Eqn. (4), we normalize both sides into v(x, t) • ∇ x log p(x, t) -∂ t log p(x, t) ≈ 1.
Definition 1. (Normalized Spatiotemporal Gradient (NSG).) The relation in Eqn. (5) reveals that natural video dynamics preserve the product between the velocity field and the ratio of spatial probability gradients to temporal density changes. We formalize this constrained ratio as the Normalized Spatiotemporal Gradient (NSG), defined as:
g(x, t) = ∇ x log p(x, t) -∂ t log p(x, t) + λ .(6)
Here, λ > 0 prevents numerical instability. Eqn. ( 5) and ( 6) imply that g(x, t) acts as a dual field to v(x, t), satisfying v • g ≈ 1. The formulation of g(x, t) bypasses the ill-posed velocity v(x, t) inversion problem while preserving the critical information about spatiotemporal gradient dynamics.
this section cite: ['b5', 'b49', 'b50', 'b51', 'b27', 'b28', 'b50', 'b52', 'b28', 'b2', 'b27', 'b53', 'b27', 'b54']

Section: Interpretation and Advantages.
The NSG statistic g(x, t) quantifies the directional sensitivity of probability flow per unit temporal variation, driven by both spatial gradients (∇ x log p(x, t)) and temporal derivatives (∂ t log p(x, t)). This statistic captures both spatial irregularities (via ∇ x log p(x, t)) and temporal inconsistencies (via ∂ t log p(x, t)), enabling comprehensive analysis of video dynamics. Moreover, by modeling fundamental probability flow dynamics, NSG avoids dependencies on specific artifacts, making it suitable for detecting generated videos across diverse generation paradigms.
this section cite: []

Section: Estimating NSG with Diffusion Models
The NSG statistic g(x, t) in Eqn. (6) requires estimating two key components: the spatial gradients ∇ x log p(x, t) and the temporal derivatives ∂ t log p(x, t). Using diffusion models' inherent gradient estimation ability [30,31], we propose an effective estimator combining spatial gradients from pre-trained diffusion models with motion-aware temporal dynamics using Eqn. (8) and (9), yielding:
g(x, t) ≈ s θ (x t ) s θ (x t ) • xt+∆t-xt ∆t + λ ,(7)
where s θ denotes the learned score function from diffusion models and x t represents the t-th video frame. This estimator eliminates the need for explicit flow computation while preserving critical spatiotemporal dynamics through physics-inspired modeling. Below, we detail its derivation.
this section cite: ['b29', 'b30']

Section: Spatial Gradients Estimation.
Diffusion models [31,56] explicitly learn a score network s θ through score matching [30] or denoising diffusion modeling [41] to approximate ∇ x log p(x, t). For a given video x at t-th frame, the spatial gradient is estimated by:
∇ x log p(x, t) ≈ s θ (x t ),(8)
where x t is the t-th frame of the video x. Here, we omit the diffusion timestep to make the notation clearer. This estimation allows direct computation of ∇ x log p(x, t) in NSG via a single forward pass of the pre-trained diffusion model, eliminating the need for numerical differentiation.
this section cite: ['b30', 'b55', 'b29', 'b40']

Section: Temporal Derivatives Approximation.
To estimate ∂ t log p(x, t), we exploit the temporal coherence of video sequences under the brightness constancy assumption [32], which posits that the probability density along motion trajectories remains constant. This leads to the following approximation: Proposition 1. Under the brightness constancy assumption p(x + ∆x, t + ∆t) ≈ p(x, t) with small inter-frame motion (∆t → 0) and inter-frame displacement (∆x → 0), we have
∂ t log p(x, t) ≈ - ∇ x log p(x, t) • ∆x ∆t .(9)
this section cite: ['b31']

Section: Exploring NSG for Detecting AI-Generated Videos
To effectively distinguish AI-generated content from real videos, it is crucial to design metrics that capture subtle distributional discrepancies in high-dimensional spatiotemporal features. Recent studies show Maximum Mean Discrepancy (MMD) [33]-a non-parametric statistic for distribution alignment-has demonstrated remarkable capabilities in measuring distributional differences, e.g., AItext detection [48,46] and adversarial samples detection [57,58]. Building upon MMD's theoretical foundation and NSG's unique strength in modeling spatiotemporal dynamics, we propose an NSGbased video detection method (NSG-VD) that integrates MMD with the NSG feature representation.
this section cite: ['b32', 'b47', 'b45', 'b56', 'b57']

Section: MMD Formulation with NSG Features.
We aggregate NSG features across T frames in each video as G(x)={g(x, t)} T t=1 . Let S re P ={x (i) } n i=1 denote a reference set of real videos and S te Q ={ỹ} represent a test video. The MMD [33] between S re P and S te Q in terms of NSG is computed as:
MMD 2 b S re P , S te Q ; H k = 1 n 2 n i,j=1 k G (i) , G (j) - 2 n n i=1 k G (i) , G (test) +k G (test) , G (test) , (10
)
where
G (i) = G(x (i) )
and G (test) = G(ỹ) are NSG features extracted from real and test videos. The kernel k : G × G → R maps NSG features to a reproducing kernel Hilbert space (RKHS) H k , such as the Gaussian kernel k (a, b) = exp -∥a -b∥ 2 / 2σ 2 . Note that while MMD is conventionally used for distribution-level comparisons, recent studies [48, 46, 58] validate its efficacy in single-sample detection by quantifying deviations from reference distributions. Crucially, while MMD provides a viable solution for distributional comparison, the core advantage of NSG-VD stems from the NSG itself modeling fundamental spatiotemporal dynamics (see details in Appendix E.3).
Detection Protocol with MMD Metric. Let f (ỹ; S P , k ω , τ ) = I MMD 2 b > τ , where I is the indicator function and τ is a threshold for the decision. Given a test video ỹ, we compute the MMD with NSG against a referenced real video set and give the decision:
f (ỹ) = Fake, if f (ỹ; S P , k ω , τ ) = 1, Real, if f (ỹ; S P , k ω , τ ) = 0. (11
)
Optimization for NSG-VD. To enhance discriminative power, we use a deep kernel [34] for MMD:
k ω (x, y) = [(1 -ϵ)κ (ϕ G (x), ϕ G (y)) + ϵ] • Φ (G(x), G(y)) ,(12)
where ϕ G (x) = ϕ(G(x)) is a deep neural network, κ and Φ are Gaussian kernels with bandwidths σ ϕ and σ Φ , and ϵ ∈ (0, 1). The kernel parameters ω={ϵ, ϕ, σ ϕ , σ Φ } will be optimized by Eqn. (13) to maximize the detection ability. Considering the multiple-population scenarios across diverse video distributions [48], we adopt a multi-population aware optimization for the kernel training:
k * ω = arg max kω MPP u (S tr P , S tr Q ; k ω ) σ2 (S tr P , S tr Q ; k ω ) + λ , σ2 = 4 N 3 N i=1   N j=1 H * ij   2 - 4 N 4   N i=1 N j=1 H * ij   2 , (13
)
where S tr P and S tr Q denote the training real and generated videos, respectively, MPP u (S tr P , S tr Q ;
k ω ) = 1 N (N -1) i̸ =j H * ij and H * ij =k ω (x i , x j )-k ω (x i , y j )-k ω (y i , x j ).
this section cite: ['b32', 'b33', 'b12', 'b47']

Section: Theoretical Guarantees for NSG-VD
The effectiveness of NSG-VD relies on ensuring the MMD between NSG features of real videos is smaller than that between real and generated videos. To formalize this, we analyze the MMD formulation in Eqn. (10), where the key discriminative information lies in the cross-term k G (i) , G (test) since the first and third terms remain invariant for fixed reference sets. Under the Gaussian kernel, this cross-term is dominated by the exponential squared distance between NSG features. Note that analyzing ∇x log p(x,t) -∂t log p(x,t)+λ under practical distributions can be very difficult and infeasible, we adopt a common practice [59,60] by assuming Gaussian-distributed data to derive theoretical insights. Below, we first characterize the NSG statistics for real and generated videos under Gaussian assumptions.
Proposition 2. Let the real video distribution be p(x, t) = N (0, σ(t) 2 I d ) and the generated video distribution be q(y, t)=N (µ, σ(t) 2 I d ), respectively, where I d ∈ R d×d is an identity matrix and µ̸ =0 ∈ R d is the distribution shift and σ(t)̸ =0, the NSG g(x, t) and g(y, t) satisfy:
g(x, t) = - x/σ(t) 2 D r (x) , - x σ(t) 2 ∼ N 0, σ(t) 2 I d , D r (x) ∼ λ + d σ(t) σ(t) - σ(t) σ(t) χ 2 (d); g(y, t) = - y/σ(t) 2 D f (y) , - y σ(t) 2 ∼ N - µ σ(t)
, σ(t
) 2 I d , D f (y) ∼ λ + d σ(t) σ(t) - σ(t) σ(t) χ 2 (d, φ), where D r (x) = λ + d σ(t) σ(t) -∥x∥ 2 σ(t) σ(t) 3 , D f (y) = λ + d σ(t) σ(t) -∥y∥ 2 σ(t) σ(t) 3 , σ(t) ≜ d dt σ(t)
, and φ = ∥µ∥ 2 σ(t) 2 , χ 2 (d) is the central chi-squared distribution with d degrees of freedom and χ 2 (d, φ) is the noncentral chi-squared distribution with noncentrality parameter φ and d degrees of freedom [61].
Proposition 2 reveals that the distribution shift µ in generated videos introduces deviations in both the numerator and denominator of the NSG, i.e., noncentral Gaussian and chi-squared distributions. To quantify this deviation, we derive an upper bound on the squared distance between NSGs: Theorem 1. Let the real video distribution be x∼N (0, σ(t) 2 I d ) and the generated video distribution be y∼N (µ, σ(t) 2 I d ), respectively, where I d ∈R d×d is an identity matrix and µ̸ =0∈R d is the distribution shift. Given G(x)={g(x, t)} T t=1 , denote φ=∥µ∥ 2 /σ(t) 2 and assume |-∂ t log p(x, t) + λ| ≥ C > 0 and |-∂ t log p(y, t) + λ| ≥ C > 0, with probability at least 1 -δ, we have
∥G(x)-G(y)∥ 2 ≤O T C 4 σ(t) 2 φd + d 2 + φ + log T δ •(φ + d) + log 2 T δ .
Theorem 1 reveals that the bound of the squared distance between NSG features of real and fake data will be smaller if the distribution shift term φ = ∥µ∥ 2 /σ(t) 2 is closer to zero for a given δ. This formalizes the intuition that small distribution shifts produce small geometric distortions in NSG space, while significant deviations in synthetic content lead to large separations from real data. Under the Gaussian kernel, this implies that the real data have a larger k(G(x), G(y)) than the fake data since the distribution shift term φ = 0 for real data. Therefore, when substituted into Eqn. (10), the MMD between NSG features of real videos is smaller than that between real and generated videos.
this section cite: ['b9', 'b58', 'b59', 'b60', 'b9']

Section: Experiments
Datasets. We evaluate our methods on the GenVideo benchmark [25], a large-scale dataset for AI-generated video detection that includes diverse real-world videos and synthetic content from multiple generative models. We use Kinetics-400 [62] as the real video source, SEINE [63] or Pika [64] as the AI-generated videos for training. The test set comprises MSR-VTT [65] and 10 diverse AI-generated datasets from different generation paradigms. More details are in Appendix C.1.
this section cite: ['b24', 'b61', 'b62', 'b63', 'b64']

Section: Evaluation Metrics.
We evaluate the performance of video detection on Recall, Accuracy, F1-score [66] and AUROC [67] metrics. More details are provided in Appendix C.2. We use bold numbers to indicate the best results and underlined numbers to denote the second-best results in tables.
Baselines. We compare our NSG-VD with following baselines: TALL [24], NPR [27], STIL [36], and Demamba [25]. These baselines are implemented based on the codebase provided by Demamba [25].
this section cite: ['b65', 'b66', 'b23', 'b26', 'b35', 'b24', 'b24']

Section: Comparisons on Standard Evaluation
We start by comparing our NSG-VD with baselines using 10, 000 real videos from Kinetics-400 and 10, 000 generated videos from Pika (Table 1) and SENIE (Table 2) for training, respectively.
Results on Trained with Kinetics-400 and Pika. From Table 1, existing methods exhibit critical limitations. For instance, Demamba struggles with generative paradigms like HotShot (40.60% Recall) and Sora (48.21% Recall), while NPR shows unstable performance with Accuracy ranging from 57.20% to 98.20%. TALL fails on synthetic outliers (e.g., 25.00% Recall on Sora) and STIL collapses completely on critical cases (e.g., 1.40% Recall on HotShot and 1.79% Recall on Sora), revealing limitations of their inherent dependencies on generator-specific artifacts.  20 80.60 34.60 35.71 43.20 57.35  Accuracy 79.80  89.20  98.20  57.20  65.70 94.80 89.50 66.50 67.86 70.8077.96 F1 75.18 88.11 98.20 27.21 49.03 94.61 88.47 50.81 52.63 59.67 68.39 AUROC 93.05 97.18 99.66 82.97 90.50 99.13 97.87 87.54 90.47 91.84 93.02 TALL Recall 51.20 65.20 93.40 32.00 61.60 94.80 81.80 49.20 25.00 53.60 60.78 Accuracy 75.10 82.10 96.20 65.50 80.30 96.90 90.40 74.10 61.61 76.30 79.85 F1 67.28 78.46 96.09 48.12 75.77 96.83 89.50 65.51 39.44 69.34 72.63 AUROC 95.82 97.14 99.73 92.55 97.36 99.79 99.09 94.84 86.67 93.75 95.67 STIL Recall 73.80 70.80 43.40 1.40 2.00 45.00 13.20 7.20 1.79 11.60 27.02 Accuracy 86.90 85.40 71.70 50.70 51.00 72.50 56.60 53.60 50.89 55.80 63.51 F1 84.93 82.90 60.53 2.76 3.92 62.07 23.32 13.43 3.51 20.79 35.82 AUROC 96.43 97.77 99.34 86.66 90.56 98.88 97.04 88.16 92.57 87.52 93.49 NSG-VD (Ours) Recall 68.33 98.33 100.00 92.50 87.50 80.00 98.33 94.17 78.57 82.50 88.02 Accuracy 81.67 98.33 96.67 91.67 90.83 88.33 95.83 94.17 88.39 88.75 91.46 F1 78.85 98.33 96.77 91.74 90.52 87.27 95.93 94.17 87.13 88.00 90.87 AUROC 92.26 98.66 98.15 94.45 96.38 94.83 98.16 97.41 96.40 94.73 96.14 Results on Trained with Kinetics-400 and SENIE. As shown in Table 2, our NSG-VD achieves superior detection performance across all metrics compared to baselines. Notably, it attains nearperfect Recall (≥ 98.33%) on models like MoonValley, HotShot and Show1, while maintaining balanced performance across diverse domains (e.g., ModelScope, WildScrape). These results are consistent with the results on Pika in Table 1, further demonstrating the effectiveness of our proposed method. In contrast, existing baselines exhibit pronounced limitations under this setting. Demamba's performance is more constrained (≤ 85.60% Recall on most models), and NPR's F1-score varies widely (41.83% ∼ 89.43%). TALL shows instability on models like Sora (33.93% Recall), while  20  45.20  41.20  26.20  33.80 60.20 60.20 22.60 25.00 18.2036.08 Accuracy 64.00 72.50 70.50 63.00 66.80 80.00 80.00 61.20 62.50 59.00 67.95 F1 43.93 62.17 58.27 41.46 50.45 75.06 75.06 36.81 40.00 30.74 51.40 AUROC 93.34 94.56 94.25 91.64 91.63 94.99 97.60 91.46 84.92 85.20 91.96 STIL Recall 25.80 64.80  68.40  46.20  29.20 67.20 70.00 44.40 26.79 25.00 46.78  Accuracy 62.70  82.20  84.00  72.90  64.40 83.40 84.80 72.00 63.39 62.30 7321 F1 40.89 78.45 81.04 63.03 45.06 80.19 82.16 61.33 42.25 39.87 61.43 AUROC 85.14 95.74 96.87 89.46 83.22 96.09 96.23 90.36 89.89 78.99 90.20 NSG-VD (Ours) Recall 85.83 99.17 100.00 99.17 97.50 95.83 99.17 91.67 82.14 81.67 93.21 Accuracy 84.58 92.50 93.75 89.58 89.58 90.83 92.50 90.00 86.61 81.67 89.16 F1 84.77 92.97 94.12 90.49 90.35 91.27 92.97 90.16 85.98 81.67 89.48 AUROC 90.76 98.18 98.18 95.03 95.48 96.97 96.53 95.11 95.73 87.13 94.91 STIL fails entirely on critical cases (e.g., 19.00% Recall on WildScrape). These failures highlight the fragility of artifact-based approaches in capturing subtle spatiotemporal inconsistencies. Quantitatively, NSG-VD surpasses Demamba by 25.88% ↑ in average Recall (97.13% vs. 71.25%) and NPR by 16.12% ↑ in average F1-score (87.45% vs. 71.33%). On closed-source models like Sora, it achieves 94.64% Recall-nearly twice Demamba's (42.86%) and sextuple STIL's (14.29%). This improvement highlights NSG-VD's sensitivity to synthetic anomalies, especially in near-photorealistic videos (e.g., Sora), where subtle spatiotemporal inconsistencies are amplified by the NSG but not effectively captured by baselines, indicating reliable detection across diverse generation paradigms.
this section cite: []

Section: Comparisons on Challenging Data-Imbalanced Scenarios
In real-world scenarios, natural videos are often abundant and accessible, while collecting sufficient AI-generated videos remains challenging due to rapidly evolving generation techniques. To thoroughly assess reliability under these conditions, we train all models using 10, 000 Kinetics-400 real videos and only 1, 000 SENIE-generated videos. As shown in To investigate the impact of the spatial gradients ∇ x log p(x, t) and temporal derivatives ∂ t log p(x, t) for our NSG-VD, we evaluate these components as independent detection statistics for AI-generated video detection. To this end, we train these separate models with 10, 000 real videos from Kinetics-400 and generated videos from Pika. From Table 4, the spatial gradient achieves moderate performance (e.g., 87.99% Recall, 83.40% F1-score), suggesting its ability to capture spatial anomalies, which may arise from its sensitivity to localized variations in texture or geometry. The temporal derivative, however, shows limited detection power (e.g., 60.35% Recall, 66.97% F1-score), likely due to its sensitivity to transient noise in dynamic modeling. In contrast, our NSG-VD integrating both components achieves significantly enhanced performance (e.g., 88.02% Recall, 90.87% F1-score). This demonstrates that the interplay between spatial gradients and temporal derivatives formalized via physical conservation principles is critical for video detection. We evaluate the decision threshold τ in Eqn. (11) for NSG-VD by testing τ ∈ [0.4, 1.3] under the same settings as Table 1. As shown in Figure 3, our NSG-VD maintains remarkably stable performance across a wide range of τ values without requiring fine-grained tuning. Specifically, NSG-VD consistently shows high detection performance as τ ∈ [0.7, 1.1] for average Recall, Accuracy and F1-Score across diverse generators. These results indicate that NSG features create a clear separation between real and fake distributions. We set τ = 1.0 as the default throughout all settings.
this section cite: ['b10']

Section: Conclusion
In this paper, we propose a physics-driven AI-generated video detection paradigm by modeling spatiotemporal dynamics through the Normalized Spatiotemporal Gradient (NSG), a novel statistic based on probability flow conservation principles. Leveraging pre-trained diffusion models, we propose an NSG-based video detection method (NSG-VD). Theoretical analyses and extensive experiments validate the superiority of our NSG-VD in detecting advanced generated videos.
this section cite: []

Section: A Theoretical Analysis

this section cite: []

Section: A.1 Basic Theorems and Corollaries Related to Statistics
We start to provide some basic theoretical results, laying the foundation for establishing the bounds of the statistics in Appendix A.6 and A.7. Theorem 2. Let X ∼ χ 2 (d, φ) follow a noncentral chi-squared distribution with d degrees of freedom and noncentrality parameter φ. For any t > 0, the following tail bounds hold:
P X -(d + φ) ≥ 2 (d + 2φ)t + 2t ≤ e -t , P X -(d + φ) ≤ -2 (d + 2φ)t ≤ e -t .
Proof. The moment-generating function of X satisfies
E[e sX ] = e φs 1-2s (1 -2s) d/2 (s < 1/2).
The log-moment generating function of X -(d + φ) is:
log E[e s(X-(d+φ)) ] = log E[e sX ] -s(d + φ) = - d 2 log(1 -2s) + φs 1 -2s -s(d + φ).(14)
For 0 < s < 1/2, we have
-s - 1 2 log(1 -2s) ≤ s 2 1 -2s ,(15)
which holds because the function ψ(s
) = -s -1 2 log(1 -2s) -s 2 1-2s satisfies ψ ′ (s) = -1 + 1 1-2s - 2s-s 2 (1-2s) 2 = -2s 2 (1-2s) 2 ≤ 0, implying max 0<s<1/2 ψ(s) < ψ(0 + ) =
0, i.e., ψ(s) ≤ 0. Substituting Eqn. (15) into Eqn.(14), we get log E[e s(X-(d+φ)) ] ≤ ds 2 1 -2s + 2φs 2 1 -2s = (d + 2φ)s 2 1 -2s . (
)16
According to the result in [68], if ∃ v, c > 0, s.t.log E[e uZ ] ≤ vu 2 2(1-cu) , then for ∀ t > 0, the following inequality holds:
P (Z ≥ ct + √ 2vt) ≤ e -t .
Applying this result to Eqn. (16), we set Z = X -(d + φ), v = 2(d + 2φ) and c = 2, then
P X -(d + φ) ≥ 2 (d + 2φ)t + 2t ≤ e -t .
For -1/2 < s < 0, we have
-s - 1 2 log(1 -2s) ≤ s 2 ,(17)
which holds because the function h(s
) = -s-1 2 log(1-2s)-s 2 satisfies h ′ (s) = -1+ 1 1-2s -2s = 4s 2 1-2s ≥ 0, implying max -1/2<s<0 ψ(s) < ψ(0 -) = 0, i.e., h(s) ≤ 0. Substituting Eqn. (17) into Eqn.(14), we get log E[e s(X-(d+φ)) ] ≤ ds 2 + 2φs 2 1 -2s = (d + 2φ 1 -2s )s 2 ≤ (d + 2φ)s 2 . (18
)
According to the result in [68], if ∃ v > 0, s.t., log E[e sZ ] ≤ vs 2 2 , then for ∀ t > 0, the following inequality holds:
P Z ≤ - √ 2vt ≤ e -t .
Applying this result to Eqn. (18), we set Z = X -(d + φ), v = 2(d + 2φ), then P X -(d + φ) ≤ -2 (d + 2φ)t ≤ e -t .
this section cite: ['b67', 'b67']

Section: Corollary 1.
Given X ∼ χ 2 (d, φ), a noncentralchi-squared distribution with d degrees of freedom and the noncentrality parameter φ, with probability at least 1 -δ, we have
|X| ≤ d + φ + 4(d + 2φ) log 2 δ + 2 log 2 δ .
Proof. By Theorem 2, setting e -t = δ 2 yields the following inequalities:
P X -(d + φ) ≥ 2 (d + 2φ) log 2 δ + 2 log 2 δ ≤ δ 2 , P X -(d + φ) ≤ -2 (d + 2φ) log 2 δ ≤ δ 2 .
Combining these two inequalities, we obtain:
P X ≥ d+φ+2 (d+2φ) log 2 δ +2 log 2 δ or X ≤ d+φ-2 (d+2φ) log 2 δ ≤δ.
Taking the complement of the above event, we have:
P d+φ-2 (d+2φ) log 2 δ ≤ X ≤ d+φ+2 (d+2φ) log 2 δ +2 log 2 δ ≥1 -δ.
By relaxing the lower bound of X, we conclude
P |X| ≤ d+φ+2 (d+2φ) log 2 δ +2 log 2 δ ≥1 -δ.
this section cite: []

Section: A.2 Proof of Proposition 1
Proposition 1. Under the brightness constancy assumption p(x + ∆x, t + ∆t) ≈ p(x, t) with small inter-frame motion (∆t → 0) and inter-frame displacement (∆x → 0), we have
∂ t log p(x, t) ≈ - ∇ x log p(x, t) • ∆x ∆t .(19)
Proof. We apply the Taylor expansion log p(x + ∆x, t + ∆t) around (x, t) to first order:
log p(x + ∆x, t + ∆t) = log p(x, t) + ∇ x log p(x, t) • ∆x + ∂ t log p(x, t) • ∆t + o(∥∆x∥ 2 + ∆t 2 ),
where o(∥∆x∥ 2 + ∆t 2 ) represents higher-order infinitesimal terms.
By assumption, log p(x + ∆x, t + ∆t) ≈ log p(x, t). Subtracting log p(x, t) from both sides:
∇ x log p(x, t) • ∆x + ∂ t log p(x, t) • ∆t + o(∥∆x∥ 2 + ∆t 2 ) ≈ 0.
Under ∆t → 0 and ∆x → 0, we obtain:
∇ x log p(x, t) • ∆x + ∂ t log p(x, t) • ∆t ≈ 0.
Rearranging terms gives: ∂ t log p(x, t) ≈ -∇ x log p(x, t) • ∆x ∆t .
this section cite: []

Section: A.3 Derivations of Chain Rule to the Conservation of Probability Mass
For ∂p ∂t + ∇ x • J = 0, we substitute J = pv and then divide the entire equation by p (which is strictly positive everywhere in its support), yielding:
1 p ∂ t p + 1 p ∇ x • (pv) = 0.
Applying the vector calculus product rule
∇ x • (pv) = p(∇ x • v) + v • (∇ x p)
and the chain rule of calculus, 1 p ∂p ∂t = ∂ t log p and ∇xp p = ∇ x log p, we obtain Eqn.(3):
∂ t log p + ∇ x • v + v • ∇ x log p = 0.
This transformation does not alter the underlying fluid constraint-it is a variable change making explicit how velocity couples to log-density's temporal and spatial gradients.
this section cite: []

Section: A.4 Derivations of Gradients in NSG
In the following, we provide the specific forms of the terms ∇ x log p(x, t), -∂ t log p(x, t) + λ and g(x, t) when the data are from Gaussian distributions, which will be used in Appendix A.5, A.6, A.7. Proposition 3. Given the real video distribution p(x, t) = N (0, σ(t) 2 I d ) and the generated video distribution q(y, t)=N (µ, σ(t) 2 I d ) with µ̸ =0 and σ(t)̸ =0, the gradients ∇ x log p(x, t) and ∇ y log p(y, t) are:
∇ x log p(x, t) = - x σ(t) 2 ∼ N (0, 1 σ(t) 2 I d ), ∇ y log p(y, t) = - y σ(t) 2 ∼ N (- µ σ(t)
, σ(t) 2 I d ).
Proof. Recall that for a Gaussian distribution p(z) = N (ν, σ 2 I d ), the probability density function is
p(z) = 1 (2πσ 2 ) d/2 exp - ∥z -ν∥ 2 2σ 2 .
The log-density is log p(z) = -d 2 log(2πσ 2 ) -∥z -ν∥ 2 2σ 2 . Thus, we have ∇ z log p(z) = -zν σ 2 . For the real video distribution p(x, t) = N (0, σ(t) 2 I d ), we have ν = 0. Taking the gradient w.r.t. x:
∇ x log p(x, t) = ∇ x - ∥x∥ 2 2σ(t) 2 = - x σ(t) 2 ∼ N 0, 1 σ(t) 2 I d .
For the generated video distribution q(y, t) = N (µ, σ(t) 2 I d ), evaluated under p(y, t)), the gradient w.r.t. y is
∇ y log p(y, t) = - y σ(t) 2 ∼ N - µ σ(t) 2 , 1 σ(t) 2 I d .
Proposition 4. Given the real video distribution p(x, t) = N (0, σ(t) 2 I d ) and the generated video distribution q(y, t)=N (µ, σ(t) 2 I d ) with µ̸ =0 and σ(t)̸ =0, the partial derivatives -∂ t log p(x, t) and -∂ t log p(y, t) are:
-∂ t log p(x, t) = d σ(t) σ(t) - ∥x∥ 2 σ(t) σ(t) 3 ∼ d σ(t) σ(t) - σ(t) σ(t) χ 2 (d), -∂ t log p(y, t) = d σ(t) σ(t) - ∥y∥ 2 σ(t) σ(t) 3 ∼ d σ(t) σ(t) - σ(t) σ(t) χ 2 (d, φ),
where σ(t) ≜ d dt σ(t) and φ = ∥µ∥ 2 σ(t) 2 . Here, χ 2 (d) is the central chi-squared distribution and χ 2 (d, φ) is the noncentral chi-squared distribution with noncentrality parameter φ [61].
Proof. We first derive the expression for the real video distribution p(x, t). The log-density is log p(x, t) = -d 2 log(2πσ(t) 2 ) -∥x∥ 2 2σ(t) 2 .
Taking the time derivative ∂ t (denoted by dot notation): 3 .
∂ t log p(x, t) = - d 2 • 1 2πσ(t) 2 • 2π • 2σ(t) σ(t) + ∥x∥ 2 σ(t) 3 σ(t) = - d σ(t) σ(t) + ∥x∥ 2 σ(t) σ(t)
Thus, we get
-∂ t log p(x, t) = d σ(t) σ(t) - ∥x∥ 2 σ(t) σ(t) 3 ∼ d σ(t) σ(t) - σ(t) σ(t) χ 2 (d),
where the last formula is based on
∥ x σ(t) ∥ 2 ∼ χ 2 (d).
For the generated video distribution q(y, t) = N (µ, σ(t) 2 I d ), under p(y, t) with φ = ∥µ∥ 2 σ(t) 2 , we have
-∂ t log p(y, t) = d σ(t) σ(t) - ∥y∥ 2 σ(t) σ(t) 3 ∼ d σ(t) σ(t) - σ(t) σ(t) χ 2 (d, φ),
where the last formula is based on ∥ y σ(t) ∥ 2 ∼ χ 2 (d, φ).
this section cite: ['b60']

Section: A.5 Proof of Proposition 2
Proposition 2. Let the real video distribution be p(x, t) = N (0, σ(t) 2 I d ) and the generated video distribution be q(y, t)=N (µ, σ(t) 2 I d ), respectively, where I d ∈ R d×d is an identity matrix and µ̸ =0 ∈ R d is the distribution shift and σ(t)̸ =0, the NSG g(x, t) and g(y, t) satisfy:
g(x, t) = - x/σ(t) 2 D r (x) , - x σ(t) 2 ∼ N 0, σ(t) 2 I d , D r (x) ∼ λ + d σ(t) σ(t) - σ(t) σ(t) χ 2 (d); g(y, t) = - y/σ(t) 2 D f (y) , - y σ(t) 2 ∼ N - µ σ(t)
, σ(t
) 2 I d , D f (y) ∼ λ + d σ(t) σ(t) - σ(t) σ(t) χ 2 (d, φ), where D r (x) = λ + d σ(t) σ(t) -∥x∥ 2 σ(t) σ(t) 3 , D f (y) = λ + d σ(t) σ(t) -∥y∥ 2 σ(t) σ(t) 3 , σ(t) ≜ d dt σ(t)
, and φ = ∥µ∥ 2 σ(t) 2 , χ 2 (d) is the central chi-squared distribution with d degrees of freedom and χ 2 (d, φ) is the noncentral chi-squared distribution with noncentrality parameter φ and d degrees of freedom [61].
Proof. According to the definition of NSG,
g(x, t) = ∇ x log p(x, t) -∂ t log p(x, t) + λ ,(20)
we can substitute the results of ∇ x log p(x, t) in Proposition 3 and -∂ t log p(x, t) in Proposition 4 into Eqn. (20) and directly contribute to the results.
this section cite: ['b60', 'b19']

Section: A.6 Derivations of Upper Bounds for Gradients
Next, we present some propositions on the upper bounds that will be used in Appendix A.7. Proposition 5. Given the real video distribution p(x, t) = N (0, σ(t)
2 I d ) and the generated video distribution q(y, t)=N (µ, σ(t) 2 I d ) with µ̸ =0 and σ(t)̸ =0, let D r (x) = λ -∂ t log p(x, t) and D f (y) = λ -∂ t log p(y, t), with probability at least 1 -δ, we have |D r (x) -D f (y)| ≤ φ + 2 (d + 2φ) log 4 δ + 2 d log 4 δ + 2 log 4 δ , where φ = ∥µ∥ 2 σ(t) 2 .
Proof. From Proposition 4, we obtain
|D r (x) -D f (y)| = | σ(t)| σ(t) 3 ∥x∥ 2 -∥y∥ 2 .
where σ(t) ≜ d dt σ(t).
Let Z = ∥x∥ 2 σ(t) 2 ∼ χ 2 (d) and W = ∥y∥ 2 σ(t) 2 ∼ χ 2 (d, φ), where φ = ∥µ∥ 2 σ(t) 2 . The difference becomes
|D r (x) -D f (y)| = | σ(t)| σ(t) |W -Z| .
To bound |W -Z|, we use concentration inequalities for chi-squared distributions by Theorem 2. For Z ∼ χ 2 (d), we have
P Z -d ≥ 2 d log 4 δ + 2 log 4 δ ≤ δ 4 , P Z -d ≤ -2 d log 4 δ ≤ δ 4 .
Combining these two events, we obtain
P -2 d log 4 δ ≤ Z -d ≤ 2 d log 4 δ + 2 log 4 δ ≥ 1 - δ 2 .(21)
Similarly, for W ∼ χ 2 (d, φ), we have
P φ -2 (d + 2φ) log 4 δ ≤ W -d ≤ φ + 2 (d + 2φ) log 4 δ + 2 log 4 δ ≥ 1 - δ 2 .(22)
Combining the bounds in Eqn. ( 22) and ( 21), we have with probability 1 -δ:
|W -Z| ≤ φ + 2 (d + 2φ) log 4 δ + 2 d log 4 δ + 2 log 4 δ .
this section cite: []

Section: Substituting this into the expression for |D r (x) -D f (y)| completes the proof.
Proposition 6. Given the real video distribution p(x, t) = N (0, σ(t) 2 I d ) and the generated video distribution q(y, t)=N (µ, σ(t) 2 I d ) with µ̸ =0 and σ(t)̸ =0, the following inequalities hold with probability at least 1 -δ:
1) ∥x∥ 2 σ(t) 2 ≤ d + 4d log 2 δ + 2 log 2 δ , 2) ∥y∥ 2 σ(t) 2 ≤ d + φ + 4(d + 2φ) log 2 δ + 2 log 2 δ , 3) ∥x -y∥ 2 2σ(t) 2 ≤ d + φ 2 + 4(d + φ) log 2 δ + 2 log 2 δ ,
where φ = ∥µ∥ 2 σ(t) 2 .
Proof. 1) Since x ∼ N (0, σ(t) 2 I d ), ∥x∥ 2 σ(t) 2 follows a central chi-squared distribution χ 2 (d). By the concentration inequality for central chi-squared distributions (Corollary 1), with probability 1 -δ:
∥x∥ 2 σ(t) 2 ≤ d + 4d log 2 δ + 2 log 2 δ .
2) For y ∼ N (µ, σ(t) 2 I d ), ∥y∥ 2 σ(t) 2 follows a noncentral chi-squared distribution χ 2 (d, φ), where φ = ∥µ∥ 2 σ(t) 2 . By Corollary 1, with probability 1 -δ, we have
∥y∥ 2 σ(t) 2 ≤ d + φ + 4(d + 2φ) log 2 δ + 2 log 2 δ .
3) Since x ∼ N (0, σ(t) 2 I d ) and y ∼ N (µ, σ(t) 2 I d ), their difference z satisfies:
xy ∼ N (-µ, 2σ(t) 2 I d ).
Thus, ∥x-y∥ 2 2σ(t) 2 follows a noncentral chi-squared distribution χ 2 (d, φ/2), where φ/2 = ∥µ∥ 2 2σ(t) 2 . By Corollary 1, with probability 1 -δ, we have ∥x -y∥ 2 2σ(t) 2 ≤ d + φ 2 + 4 (d + φ) log 2 δ + 2 log 2 δ .
this section cite: []

Section: A.7 Proof of Theorem 1
Given a video x ∈ R T ×d , its NSG Feature is G(x) = {g(x, t)} T t=1 , where g(x, t) is defined as:
g(x, t) = ∇ x log p(x, t) -∂ t log p(x, t) + λ ,(23)
Here, λ > 0 and p(x, t) is the probability density of the real video parameterized by time t.
Note that Theorem 1 share a common lower bound C on both D r (x)=λ-∂ t log p(x, t) and D f (y)=λ-∂ t log p(y, t). We first derive the conditions for D r (x)>C>0 and D f (y)>C>0 to hold in Proposition 7. The same analytical approach can be extended to examine other cases. Proposition 7. Let the real video distribution be x ∼ N (0, σ(t) 2 I d ) and the generated video distribution be y ∼ N (µ, σ(t) 2 I d ), respectively, where I d ∈ R d×d is an identity matrix and µ ̸ = 0 ∈ R d is the distribution shift, we have D r (x) > C > 0 and D f (y) > C > 0 with probability at least 1 -δ, provided C and λ meet the following conditions:
1) Case 1 ( σ(t) σ(t) > 0):
C = λ - σ(t) σ(t) • φ + 2 σ(t) σ(t) d log 2 δ + 2 σ(t) σ(t) log 2 δ , λ > σ(t) σ(t) (d + φ) - 2 σ(t) σ(t) d log 2 δ - 2 σ(t) σ(t) log 2 δ .
where φ = ∥µ∥ 2 σ(t) 2 .
2) Case 2 ( σ(t) σ(t) < 0):
C = λ - 2 σ(t) σ(t) d log 2 δ , λ > 2 σ(t) σ(t) d log 2 δ .
Proof. Let W = ∥y∥ 2 σ(t) 2 ∼ χ 2 (d, φ), where φ = ∥µ∥ 2 σ(t) 2 . From Theorem 2, we have
P W -(d + φ) ≥ 2 (d + 2φ) log 2 δ + 2 log 2 δ ≤ δ 2 , (24
) P W -(d + φ) ≤ -2 (d + 2φ) log 2 δ ≤ δ 2 . (25
) 1) Case 1 ( σ(t) σ(t) > 0): Substituting D f (y) = λ -σ(t) σ(t) (W -d) into the bound Eqn. (24):
P D f (y) ≤ λ - σ(t) σ(t) • φ + 2 σ(t) σ(t) (d + 2φ) log 2 δ + 2 σ(t) σ(t) log 2 δ ≤ δ 2 .
Thus, the following inequalities hold with probability at least 1 -δ/2:
D f (y) ≥ λ - σ(t) σ(t) • φ + 2 σ(t) σ(t) (d + 2φ) log 2 δ + 2 σ(t) σ(t) log 2 δ , D r (x) ≥ λ + 2 σ(t) σ(t) d log 2 δ + 2 σ(t) σ(t) log 2 δ .
To ensure D r (x) > C > 0 and D f (y) > C > 0, we can select C and λ as:
C = λ - σ(t) σ(t) • φ + 2 σ(t) σ(t) d log 2 δ + 2 σ(t) σ(t) log 2 δ , λ > σ(t) σ(t) (d + φ) - 2 σ(t) σ(t) d log 2 δ - 2 σ(t) σ(t) log 2 δ .
2) Case 2 ( σ(t) σ(t) < 0):
Substituting D f (y) = λ -σ(t) σ(t) (W -d) into the bound Eqn. (25
): P D f (y) ≤ λ - σ(t) σ(t) • φ - 2 σ(t) σ(t) (d + 2φ) log 2 δ ≤ δ 2 .
Thus, the following inequalities hold with probability at least 1 -δ/2:
D f (y) ≥ λ - σ(t) σ(t) • φ - 2 σ(t) σ(t) (d + 2φ) log 2 δ , D r (x) ≥ λ - 2 σ(t) σ(t) d log 2 δ .
To ensure D r (x) > C > 0 and D f (y) > C > 0, we can select C and λ as:
C = λ - 2 σ(t) σ(t) d log 2 δ , λ > 2 σ(t) σ(t) d log 2 δ .
Building upon the established Propositions 2, 5, 6 and 7, we next prove Theorem 1.
Theorem 1. Let the real video distribution be x∼N (0, σ(t) 2 I d ) and the generated video distribution be y∼N (µ, σ(t) 2 I d ), respectively, where I d ∈R d×d is an identity matrix and µ̸ =0∈R d is the distribution shift. Given G(x)={g(x, t)} T t=1 , denote φ=∥µ∥ 2 /σ(t) 2 and assume |-∂ t log p(x, t) + λ| ≥ C > 0 and |-∂ t log p(y, t) + λ| ≥ C > 0, with probability at least 1 -δ, we have
∥G(x)-G(y)∥ 2 ≤O T C 4 σ(t) 2 φd + d 2 + φ + log T δ •(φ + d) + log 2 T δ .
Proof. Based on the definition of G(x), we have
∥G(x) -G(y)∥ 2 = T t=1 ∥g(x, t) -g(y, t)∥ 2 .(26)
Next, we focus on the bound of g(x, t) -g(y, t).
Let D r (x) = λ -∂ t log p(x, t) and D f (y) = λ -∂ t log p(y, t), by Propositions 3 and 4, we have
∥g(x, t) -g(y, t)∥ 2 = x/σ(t) 2 D r (x) - y/σ(t) 2 D f (y) 2 = x/σ(t) 2 D r (x) - x/σ(t) 2 D f (y) + x/σ(t) 2 D f (y) - y/σ(t) 2 D f (y) 2 ≤2 x/σ(t) 2 D r (x) - x/σ(t) 2 D f (y) 2 + 2 x/σ(t) 2 D f (y) - y/σ(t) 2 D f (y) 2 =2 1 D r (x) - 1 D f (y) 2 • ∥x∥ 2 σ(t) 4 + 2 1 D f (y) 2 • ∥x -y∥ 2 σ(t) 4 =2 D f (y) -D r (x) D r (x)D f (y) 2 • ∥x∥ 2 σ(t) 4 + 2 1 D f (y) 2 • ∥x -y∥ 2 σ(t) 4 ≤2 |D f (y) -D r (x)| 2 C 4 • ∥x∥ 2 σ(t) 4 + 2 C 2 • ∥x -y∥ 2 σ(t) 4 .(27)
For the first term in Eqn. (27), according to Proposition 5, with probability at least 1 -2δ/3, we have
2 |D f (y) -D r (x)| 2 C 4 • ∥x∥ 2 σ(t
) 4 ≤ 2 C 4 σ(t) 2 φ + 2 (d + 2φ) log 12 δ + 2 d log 12 δ + 2 log 12 δ • d + 4d log 6 δ + 2 log 6 δ ≤ 2 C 4 σ(t) 2 φ + 4 (d + 2φ) log 12 δ + 2 log 12 δ • d + 4d log 6 δ + 2 log 6 δ . (
)28
For the second term in Eqn. (27), applying Proposition 6, with probability at least 1 -δ/3, we have
2 C 2 • ∥x -y∥ 2 σ(t) 4 ≤ 4 C 2 σ(t) 2 d + φ 2 + 4(d + φ) log 6 δ + 2 log 6 δ .(29)
For simplicity, let L = log 12 δ . Since log 6 δ = L -log 2 < L, we have
2 |D f (y) -D r (x)| 2 C 4 • ∥x∥ 2 σ(t) 4 ≤ 2 C 4 σ(t) 2 φ + 4 (d + 2φ)L + 2L • d + 2 √ dL + 2L ≤ 2 C 4 σ(t) 2 (φ + 2(d + 2φ) + L + 2L) • (d + d + L + 2L) = 2 C 4 σ(t) 2 (5φ + 2d + 3L)(2d + 3L) = 2 C 4 σ(t) 2 10φd + 4d 2 + 3L • (5φ + 4d) + 9L 2 . (30
)
2 C 2 • ∥x -y∥ 2 σ(t) 4 ≤ 4 C 2 σ(t) 2 d + φ 2 + 2 (d + φ)L + 2L ≤ 4 C 2 σ(t) 2 d + φ 2 + (d + φ)L + 2L = 4 C 2 σ(t) 2 d + φ 2 + L(d + φ + 2) .(31)
Combining Eqn. ( 30) and ( 31) and ( 27), and substituting L = log 12 δ , with probability at least 1 -δ, we have ∥g(x, t) -g(y, t)∥
2 ≤ 2 C 4 σ(t) 2 10φd + 4d 2 + 3L • (5φ + 4d) + 9L 2 + 4 C 2 σ(t) 2 d + φ 2 + L(d + φ + 2) .
For further simplicity, note that 1 C 2 ≤ 1 C 4 , we obtain ∥g(x, t) -g(y, t)∥
2 ≤ 2 C 4 σ(t) 2 10φd + 4d 2 + 2d + φ + L • (17φ + 14d + 4) + 9L 2 = 2 C 4 σ(t) 2 10φd + 4d 2 + 2d + φ + log 12 δ • (17φ + 14d + 4) + 9 log 2 12 δ . (32
)
Summing over time steps t = 1, . . . , T , and replacing δ in Eqn. (32) into δ/T , we get:
∥G(x)-G(y)∥ 2 ≤ 2T C 4 σ(t) 2 10φd + 4d 2 + 2d + φ + log 12T δ •(17φ + 14d + 4) + 9 log 2 12T δ .
Thus, we obtain the final results
∥G(x)-G(y)∥ 2 ≤O T C 4 σ(t) 2 φd + d 2 + φ + log T δ •(φ + d) + log 2 T δ .
this section cite: ['b26', 'b26', 'b31']

Section: B More Related Work
Maximum Mean Discrepancy (MMD). Maximum Mean Discrepancy (MMD) is an effective metric for two-sample testing to assess whether two samples originate from the same distribution [69,33,70,34,71,72,73]. Originally introduced by Müller et al. [72] as an instance of an integral probability metric, MMD admits several sample-based estimators. Particularly, Gretton et al. [33] introduce the U-statistic estimator, which is unbiased for the squared MMD and achieves nearminimal variance among all unbiased alternatives. Further, Tolstikhin et al. [74] derive lower bounds on the estimation error of MMD under finite samples when employing a radial universal kernel.
Building upon the traditional formulation of MMD, recent advancements incorporate learnable kernels to enhance its discriminative capability. Liu et al. [34] develop a data-splitting strategy for kernel optimization and selection, effectively addressing the kernel adaptation challenges for complex-data scenarios. Kim et al. [75] propose an adaptive two-sample test designed for comparing two Hölder densities supported on the d-dimensional unit ball. In addition, Zhang et al. [48] introduce MMD-MP, a multi-population aware optimization framework to further improve the stability of kernel-based MMD training. At present, MMD has been extensively applied to distributional measurement and discrepancy detection tasks across both textual and visual modalities [45,48,46].
this section cite: ['b68', 'b32', 'b69', 'b33', 'b70', 'b71', 'b72', 'b71', 'b32', 'b73', 'b33', 'b74', 'b47', 'b44', 'b47', 'b45']

Section: C More Details for Experiment Settings

this section cite: []

Section: C.1 More Details on Datasets
GenVideo [25] is a large-scale benchmark for AI-generated video detection, comprising 1.22 million real videos and 1.05 million AI-generated videos. The real video collection aggregates content from three established datasets: MSR-VTT (web video clips) [65], Kinetics-400 (human action videos) [62], and Youku-mPLUG (diverse online videos captured from Youku.com) [76]. The AI-generated portion features videos produced by 19 distinct generation models, including both open-source implementations (ZeroScope [77], I2VGen-XL [78], SVD [1], VideoCrafter [7], DynamiCrafter [8], Stable Diffusion(SD) [79], SEINE [63], Latte [80], OpenSora [81], ModelScope [82], HotShot [83], Show-1 [84], Gen2 [85], Crafter [86], Lavie [87]) and commercial closed-source systems (Pika, Sora, MoonValley, MorphStudio). This dataset spans multiple generation paradigms, including text-to-video and image-to-video synthesis. Throughout all experiments, we filter videos with less than 8 frames and only uniformly sample 8 frames for each video during training and testing.
this section cite: ['b24', 'b64', 'b61', 'b75', 'b76', 'b77', 'b0', 'b6', 'b7', 'b78', 'b62', 'b79', 'b80', 'b81', 'b82', 'b83', 'b84', 'b85', 'b86']

Section: C.2 More Details on Evaluation Metrics
Video generation detection is inherently a binary classification task. Here, we introduce four fundamental evaluation metrics of binary classification: True Positive (TP) means correctly predicted positive instances (ground truth is positive, prediction is positive).True Negative (TN) means correctly predicted negative instances (ground truth is negative, prediction is negative). False Positive (FP) means incorrectly predicted positive instances (ground truth is negative, prediction is positive). False Negative (FN) means incorrectly predicted negative instances (ground truth is positive, prediction is negative).
AUROC denotes the Area Under the Receiver Operating Characteristic Curve [67,88], which is a widely used statistic for assessing the discriminatory capacity of distribution models. Formally, AU ROC = T P (t)F P (t)dt, where T P (t) = T P (t)/(T P (t) + F N (t)) is the true positive rate and F P (t) = F P (t)/(F P (t) + T N (t)) is false positive rate with a threshed t.
Accuracy (ACC) measures the model's overall correctness in classification by calculating the ratio of correctly predicted instances (both true positive and true negative) to the total instances. Accuracy = T P + T N T P + T N + F P + F N .
Recall [66] evaluates the model's ability to identify all relevant instances of a class, measuring the proportion of true positives among all actual positive instances. It emphasizes minimizing false negatives, ensuring comprehensive coverage of positive cases.
Recall = T P T P + F N .
Precision [66] quantifies the model's capability to avoid false positives by calculating the proportion of true positives among all predicted positive instances. It ensures reliability in positive predictions.
Precision = T P T P + F P .
F1-score [66] balances Precision and Recall using their harmonic mean, providing a robust metric for scenarios with imbalanced class distributions. It penalizes extreme biases toward either precision or recall.
F1-score = 2 × Precision × Recall Precision + Recall .
this section cite: ['b66', 'b87', 'b65', 'b65', 'b65']

Section: C.3 Implementation Details on NSG-VD
In our NSG-VD, we employ the pre-trained diffusion model s θ of Guided Diffusion using the 256 × 256 unconditional checkpoint from the guided-diffusion libraryfoot_0 following [56]. For a given video x at t-th frame, we compute its score feature ∇ x log p(x, t) by diffusing x t at diffusion timestep 5/1, 000 and passing it through s θ . For the deep kernel ϕ G , we employ a single-layer of Swin transformer [89], mapping input features of dimension 8 × 224 × 224 to a 300-dimensional output.
We conduct our experiments on a server with 1× NVIDIA RTX 3090 GPU using Python 3.10.17 and Pytorch 2.7.0. We use Adam optimizer [90] to optimize the kernel parameters ω with batchsize 24, learning rate 0.0001, weight decay 0.1, σ ϕ = 0.1 and σ Φ = 100. For the testing, we set the decision threshold τ = 1 in Eqn. (11). The overall algorithms for training and testing are in Alg. 1 and 2.
C.4 Pseudo Code of NSG-VD Algorithm 1 Training deep kernel of MMD Input: Real and generated videos S tr P , S tr Q ; ω ← ω0; λ ← 10 -10 ; learning rate η; for r = 1, 2, . . . , rmax do kω ← kernel function using Eqn. (12); M (ω) ← MPPu(S tr P , S tr Q ; kω); V λ (ω) ← σ2 (S tr P , S tr Q ; kω) using Eqn. (13); Ĵλ (ω) ← M (ω)/ V λ (ω); ω ← ω + η∇Adam Ĵλ (ω); end for Output: k * ω Algorithm 2 Detecting videos with NSG-VD Input: Referenced videos S re P , testing videos S te Q ; decision f (•); deep kernel kω; threshold τ ; for xi in S te Q do Qi← MMD 2 b (S re P , {xi}; kω) using Eqn. (10); f (xi; S re P , kω, τ ) = I (Qi > τ ); Obtaining f (xi) using Eqn. (11); end for Output: Predictions {f (xi)} of the testing set
this section cite: ['b55', 'b88', 'b89', 'b10']

Section: D More Experimental Results

this section cite: []

Section: D.1 More Results on Standard Evaluation
To demonstrate the statistical robustness and reproducibility of our proposed NSG-VD method, we report standard deviations of four metrics across 10 datasets with three different seeds (Table 5). The table shows that our NSG-VD achieves consistently high performance with minimal variance (e.g., 0.41% for Recall, 0.87% for Accuracy), indicating strong reliability and repeatability of our methods.
this section cite: []

Section: D.2 More Results on Impact of Spatial Gradients and Temporal Derivatives
To comprehensively analyze how spatial gradients and temporal derivatives contribute to detection performance across diverse generative paradigms, we include detailed results across 10 diverse generative paradigms in Table 6. The spatial gradients achieve strong performance across most generated models (e.g., 81.67% Recall on ModelScope, 97.50% Recall on MorphStudio), with only minor performance gaps on models like HotShot (72.23% Recall) and Show1 (74.17% Recall).
In contrast, the temporal derivatives show complementary strengths and relatively better detection capabilities on these challenging cases, notably achieving 75.40% Recall on HotShot and 77.60% Recall on Show1, where temporal dynamics (e.g., rapid motion transitions in HotShot dataset) may play a more pronounced role in exposing synthetic anomalies.
Notably, our proposed NSG-VD demonstrates superior reliability by integrating both components, achieving an average F1-score of 90.87%, a significant improvement over individual features. This highlights the complementary nature of spatiotemporal dynamics modeling, enabling consistent detection performance even when individual cues exhibit dataset-specific limitations. 20 0 20 40 60 80 100 120 140 Number of Tunable Parameters (M) 30 40 50 60 70 80 90 100 F1-score (%) NPR STIL DeMamba-XCLIP TALL Ours B e t t e r Method All Params(M) ↓ Tun. Params(M) ↓ F1(%) ↑ DeMamba 119.89 119.89 80.12
TALL 82.89 82.89 72.63 STIL 21.63 21.63 35.82 NPR 1.37 1.37 68.39 NSG-VD (Ours) 527.45 0.25 90.87 Existing baselines struggle to balance parameter scale and performance. NPR achieves only 68.39% F1-score despite its minimal trainable parameters (1.37 M), while STIL requires 21.63 M parameters to attain 35.82% F1-score-a suboptimal trade-off compared to NSG-VD's superior performance with 100× fewer parameters. These results underscore the limitations of conventional artifact-driven frameworks in effective parameter budget utilization, further validating the importance of our physicsguided spatiotemporal modeling paradigm for AI-generated video detection.
this section cite: []

Section: Performance vs. Inference Time Analysis.
We evaluate the efficiency of our NSG-VD by analyzing both detection performance and computational overhead under the same setting as Table 1. As shown in Table 7, our NSG-VD achieves superior detection performance (e.g., 97.13% Recall, 87.45% F1-score) with a practical inference latency of 0.3605s per video, which remains viable for non-realtime applications (e.g., judicial video evidence analysis) despite being slower than other baselines. This latency stems from our current implementation of pre-trained diffusion models for gradient estimation-a design choice prioritizing theoretical validation over computational optimization.
Importantly, this current implementation prioritizes accuracy over speed to establish the theoretical and empirical validity of physics-guided spatiotemporal modeling. Empirically, we observe that the inference speed can be greatly enhanced with minimal performance degradation by scaling the resolution of pre-trained diffusion models, e.g., reducing resolution to 128 × 128 and 64 × 64 cuts inference time by 67.73% and 91.73%, respectively, while retaining over 98.63% and 94.57% of the original AUROC (Table 7). This trade-off underscores the flexibility of our approach in balancing accuracy and efficiency according to application needs. Future work may further boost efficiency via diffusion model compression [91,92,93] or efficient architecture design [94,95], highlighting NSG-VD's potential for practical deployment as video generation and detection requirements advance. 10 3 10 2 10 1 10 0 0 10 0 10 1 10 2 10 3 10 4 Values in t logp(x, t) 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 Frequency Kinetics-400 SEINE To ensure the numerical stability of the NSG statistic with the temporal derivatives ∂ t log p(x, t) in its denominator, we examine the distribution of values in ∂ t log p(x, t) across 10, 000 real and generated videos from Kinetics-400 and SEINE, respectively. From Figure 5, nearly all values lie outside the critical near-zero range [-0.1, 0.1]. This indicates that almost no value in ∂ t log p(x, t) approaches zero in practice, effectively mitigating instability risks from division by vanishingly small values.
The observed distribution aligns with the physical intuition that temporal density changes in real or synthetic videos are inherently non-stationary, resulting in measurable temporal derivatives. Additionally, the regularization term λ > 0 in the NSG denominator (Eqn. 6) further safeguards against edge cases where ∂ t log p(x, t) might marginally approach zero. These design choices collectively ensure robust numerical stability for NSG across diverse video distributions.
this section cite: ['b90', 'b91', 'b92', 'b93', 'b94']

Section: E.3 Impact of MMD for NSG-VD
To validate the inherent superiority of the NSG statistic independent of the training objective, we compare our framework trained with both Maximum Mean Discrepancy (NSG-VD) and standard binary cross-entropy loss (NSG-BCE) against baselines using BCE. From Table 8, NSG-BCE achieves superior performance across all metrics compared to state-of-the-art baselines, even when adopting a conventional training paradigm. For example, it achieves 77.67% average Recall and 82.70% F1-score, significantly outperforming Demamba by 6.42% ↑ in Recall and 1.83% ↑ in F1-score, and TALL by 16.20% ↑ in Recall and 8.65% ↑ in F1-score. This demonstrates that the NSG statistic's ability to capture spatiotemporal dynamics remains effective regardless of the training objective.
Notably, NSG-BCE excels in challenging scenarios where other methods struggle. For instance, it achieves 64.29% Recall on Sora (vs. 42.86% for Demamba) and 63.60% Recall on WildScrape (vs. 48.00% for Demamba), highlighting its ability to generalize beyond superficial artifacts. The performance gap widens further in NSG-VD (97.13% Recall), where MMD explicitly models distributional shifts by the NSG feature in a producing kernel Hilbert space and enables more precise separation between real and synthetic videos. These results confirm that the NSG statistic's physicsdriven design captures fundamental spatiotemporal dynamics, providing an intrinsic advantage over conventional features regardless of the training strategy.  60.20  62.00 77.80 88.20 43.80 33.93 35.80 61.47  Accuracy 78.80  87.00  89.20  79.60  80.50 88.40 93.60 71.40 66.07 67.40 80.20F1 73.43 85.23 88.03 74.69 76.07 87.02 93.23 60.50 50.00 52.34 74.05 AUROC 97.10 98.12 98.63 96.37 96.45 97.76 99.38 94.80 83.35 89.45 95.14 STIL Recall 28.60 57.40 78.40 46.80 18.80 66.40 69.00 24.80 14.29 19.00 42.35 Accuracy 64.20 78.60 89.10 73.30 59.30 83.10 84.40 62.30 57.14 59.40 71.08 F1 44.41 72.84 87.79 63.67 31.60 79.71 81.56 39.68 25.00 31.88 55.81 AUROC 95.53 97.91 99.40 96.49 92.79 98.06 98.86 91.00 92.79 86.58 94.94 NSG-BCE (Ours) Recall 53.40 96.40 94.80 90.60 77.60 79.40 83.20 73.40 64.29 63.60 77.67 Accuracy 72.70 94.20 93.40 91.30 84.80 85.70 87.60 82.70 74.11 77.80 84.43 F1 66.17 94.32 93.49 91.24 83.62 84.74 87.03 80.93 71.29 74.13 82.70 AUROC 84.67 98.79 97.77 96.90 92.69 93.00 93.86 91.32 83.58 87.99 92.06 NSG-VD (Ours) Recall 91.67 100.00 100.00 100.00 100.00 98.33 100.00 97.50 94.64 89.17 97.13 Accuracy 82.50 88.33 89.58 84.58 86.25 87.08 86.67 87.92 89.29 78.33 86.05 F1 83.97 89.55 90.57 86.64 87.91 88.39 88.24 88.97 89.83 80.45 87.45 AUROC 90.67 97.62 98.38 95.88 96.69 97.87 97.64 95.09 96.14 88.65 95.46 E.4 Impact of Size of Reference Set for NSG-VD
We investigate the impact of reference set size by evaluating subsets containing between 10 and 500 samples, with other settings remaining consistent with Table 1. Performance is assessed using comprehensive criteria, including AUROC, Accuracy, F1 Score, and Recall. Intuitively, a larger reference set enables more accurate estimation of the underlying distribution of real videos, thereby supporting more stable and reliable detection. In contrast, smaller reference sets may introduce substantial sampling and estimation biases. As shown in Figure 6, our NSG-VD demonstrates consistently robust performance across varying reference set sizes, with the exception of extreme cases involving very limited samples (e.g., n = 10). Consequently, we set n = 100 for all experiments.
this section cite: []

Section: E.5 Impact of Diversity of Real Videos in the Reference Set
We conduct additional ablation studies on real-domain mixed reference sets, revealing a key strength of NSG-VD: strong generalization to unseen generated video domains when most real test samples are covered by the reference distribution. Specifically, we train on Kinetics-400 (real) and SEINE (generated) videos, and test on MSR-VTT (real) and 10 generated videos using reference sets with varying ratios of MSR-VTT and Kinetics-400. From Table 9, even a small proportion (3 : 7) yields satisfactory performance (84.19% of Accuray, 81.12% of F1-Score) compared with baselines, which quickly saturates. This confirms that NSG-VD needs only modest in-domain real coverage, while the fake side can remain highly heterogeneous. These results will be included in our revision.
this section cite: []

Section: E.6 Discussions on Assumption of the Divergence Term
We assume ∇ x • v is subdominant in smoothly varying video distributions for three reasons: First, its direct estimation is ill-posed in high-dimensional video data. Solving ∂ t x = v(x, t) is an underdetermined inverse problem, and video noise (e.g., blur or compression) further amplifies estimation errors, making explicit divergence computation unstable and infeasible [32,29]. Second, many physical flows approximate incompressibility (∇ x • v ≈ 0), a simplification grounded in fluid dynamics [29] and quantum mechanics [55] that preserves physical interpretability. Third, our NSG remains robust even if ∇ x • v ̸ = 0, as it captures cumulative spatiotemporal inconsistencies across all terms in Eqn. (3). Experiments confirm the resilience of NSG-VD to deviations from this assumption.
this section cite: ['b31', 'b28', 'b28', 'b54', 'b2']

Section: F Limitations and Future Directions
While our proposed NSG-VD method demonstrates strong performance across diverse AI-generated video detection scenarios, several limitations and opportunities for future work remain:
Limitations. First, the current formulation of the NSG statistic relies on simplified physical assumptions (e.g., the incompressible flow approximation in continuity equations), which may fail to capture highly dynamic or discontinuous motion patterns in complex real-world scenarios. Second, the effectiveness of NSG-VD critically depends on the quality of pre-trained diffusion models used for score estimation; domain shifts or limited training data may degrade the reliability of estimated NSG features. Third, while NSG-VD achieves competitive performance, its reliance on diffusion models introduces computational overhead, making it less suitable for large-scale real-time detection tasks. Lastly, while our deep kernel design improves detection performance, its architecture could be further optimized to better adapt to heterogeneous spatiotemporal patterns.
this section cite: []

Section: Future Directions.
To address these limitations, future work could explore more sophisticated physical models that account for compressible flows or discontinuous motion dynamics [54], enhancing the NSG statistic's adaptability to complex scenarios. Additionally, developing effective domainspecific fine-tuning strategies [96,97,98,99] could improve the reliability of score estimation under distribution shifts. For real-time deployment, investigating lightweight diffusion model compression techniques (e.g., pruning [91,100], quantization [92,93]) would reduce computational costs. Finally, advancing the design of the deep kernel network-such as incorporating attention mechanisms [101] or hierarchical feature fusion [89]-could further optimize the MMD-based detection framework, enabling better performance for AI-generated video detection.
this section cite: ['b53', 'b95', 'b96', 'b97', 'b98', 'b90', 'b99', 'b91', 'b92', 'b100', 'b88']

Section: G Broader Impacts
The development of AI-generated video detection methods like NSG-VD has significant societal, ethical, and technical implications. Our work contributes to mitigating the risks of malicious deepfake content, such as misinformation, identity theft, and political manipulation, by enabling more reliable verification of video authenticity. By leveraging physics-informed principles, NSG-VD provides a reliable framework for detecting synthetic videos that may otherwise evade traditional artifact-based detection methods. This could strengthen trust in digital media, support content moderation efforts, and aid legal or journalistic investigations involving video evidence.
This research aligns with broader efforts to establish trustworthy multimedia ecosystems. By bridging physics principles with machine learning, NSG-VD advances interpretable detection mechanisms-a critical step toward auditing AI-generated content while fostering public awareness of synthetic media risks. We encourage interdisciplinary collaboration among researchers, ethicists, and legislators to ensure such technologies serve as safeguards rather than instruments of control.
this section cite: []

Section: H Visualizations
Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Real Real Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Real Real Real Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Real Real Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Real Real Real Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Real Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Real Figure 7: Results of the detection on real videos from the MSR-VTT dataset. Demamba NPR TALL STIL NSG-VD (Ours) Real Fake Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Fake Real Fake Figure 8: Results of the detection on generated videos from the Crafter dataset. Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Fake Fake Figure 9: Results of the detection on generated videos from the Gen2 dataset. Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Figure 10: Results of the detection on generated videos from the HotShot dataset. Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Fake Real Fake Figure 11: Results of the detection on generated videos from the Lavie dataset. Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Fake Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Fake Real Fake Fake Figure 12: Results of the detection on generated videos from the ModelScope dataset. Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Fake Fake Real Fake Figure 13: Results of the detection on generated videos from the MoonValley dataset. Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Real Fake Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Real Fake Fake Figure 14: Results of the detection on generated videos from the MorphStudio dataset. Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Figure 15: Results of the detection on generated videos from the Show1 dataset. Demamba NPR TALL STIL NSG-VD(Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Fake Real Fake Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Figure 16: Results of the detection on generated videos from the Sora dataset. Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Figure 17: Results of the detection on generated videos from the Seaweed dataset. Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Fake Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Figure 18: Results of the detection on generated videos from the Seaweed dataset. Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake Demamba NPR TALL STIL NSG-VD (Ours) Real Real Real Real Fake To further demonstrate the excellent performance of our NSG-VD, we present visual detection results on both real and generated videos across all 10 datasets. As illustrated in Figures 78910111213141516171819, both the baselines and NSG-VD demonstrate satisfactory detection on real video samples. For generated videos, the existing baselines achieve reasonable performance on early generation models (e.g., Crafter, Gen2, and MoonValley), but exhibit significant performance degradation when applied to more advanced generative models (e.g., Show1, Sora, and WildScrape). In contrast, NSG-VD consistently achieves strong detection performance across all generation levels.
On this basis, we consider the recently proposed Seaweed [102] method (as shown in Figures 17, 18), which generates highly realistic long-form videos. All four baselines exhibit near-complete failure on this dataset, whereas NSG-VD continues to deliver effective detection performance.
this section cite: ['b101']

Section: Answer: [Yes]
Justification: The paper provides complete theoretical results for the NSG feature lower bound (Theorem 1 in Section 3.4), including assumptions (e.g., Gaussian-distributed real/fake videos and temporal derivatives constraints). The proof is detailed in Appendix A.7, including all mathematical derivations and references to supporting lemmas. All theorems and lemmas are numbered and cross-referenced, and proofs are accessible in Appendix A.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We fully disclose all information needed to reproduce the main experimental results of the paper, see Section 4 and Appendix C.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in
this section cite: []

Section: 
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: We discuss the limitations in Appendix F.
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

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Justification: We will release our code upon acceptance.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We provide full experimental detail content in our experimental settings (see Section 4 and Appendix C).
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We evaluate statistical significance by reporting mean and standard deviation across multiple runs with different random seeds (Appendix D.1).
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes]
Justification: We provide detailed information about on computing resources in Appendix C.3.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: The paper meets the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: We discuss broader impacts in Appendix G.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: This paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We strictly follow the license of the assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: This paper does not involve LLMs. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b1 Title: Align your latents: High-resolution video synthesis with latent diffusion models Year: (2023)
Ref_id:b2 Title: Video generation models as world simulators Year: (2024)
Ref_id:b3 Title: Customcrafter: Customized video generation with preserving motion and concept composition abilities Year: (2025)
Ref_id:b4 Title: Unveiling causal reasoning in large language models: Reality or mirage? Year: (2024)
Ref_id:b5 Title: Fine-grained controllable video generation via object appearance and context Year: (2025)
Ref_id:b6 Title: Videocrafter2: Overcoming data limitations for high-quality video diffusion models Year: (2024)
Ref_id:b7 Title: Animating open-domain images with video diffusion priors Year: (2024)
Ref_id:b8 Title: Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling Year: (2024)
Ref_id:b9 Title: Image conductor: Precision control for interactive video synthesis Year: (2025)
Ref_id:b10 Title: Towards open-set identity preserving face synthesis Year: (2018)
Ref_id:b11 Title: Pulid: Pure and lightning id customization via contrastive alignment Year: (2024)
Ref_id:b12 Title: Out-of-distribution detection learning with unreliable out-of-distribution sources Year: (2023)
Ref_id:b13 Title: Hififace: 3d shape and semantic prior guided high fidelity face swapping Year: (2021)
Ref_id:b14 Title: Diffswap: High-fidelity and controllable face swapping via 3d-aware masked diffusion Year: (2023)
Ref_id:b15 Title: Avff: Audio-visual feature fusion for video deepfake detection Year: (2024)
Ref_id:b16 Title: Learning defense transformations for counterattacking adversarial examples Year: (2023)
Ref_id:b17 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b18 Title: Vbench: Comprehensive benchmark suite for video generative models Year: (2024-06)
Ref_id:b19 Title: Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness Year: (2025)
Ref_id:b20 Title: Codef: Content deformation fields for temporally consistent video processing Year: (2024)
Ref_id:b21 Title: Deepfake video detection through optical flow based cnn Year: (2019)
Ref_id:b22 Title: Altfreezing for more general video face forgery detection Year: (2023)
Ref_id:b23 Title: Tall: Thumbnail layout for deepfake video detection Year: (2023)
Ref_id:b24 Title: Demamba: Ai-generated video detection on million-scale genvideo benchmark Year: (2024)
Ref_id:b25 Title: Thinking in frequency: Face forgery detection by mining frequency-aware clues Year: (2020)
Ref_id:b26 Title: Rethinking the up-sampling operations in cnn-based generative network for generalizable deepfake detection Year: (2024)
Ref_id:b27 Title: An introduction to fluid dynamics Year: (2000)
Ref_id:b28 Title: Fluid dynamics: an introduction Year: (2014)
Ref_id:b29 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b30 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b31 Title: Determining optical flow Year: (1981)
Ref_id:b32 Title: A kernel two-sample test Year: (2012)
Ref_id:b33 Title: Learning deep kernels for non-parametric two-sample tests Year: (2020)
Ref_id:b34 Title: Exposing deep fakes using inconsistent head poses Year: (2019)
Ref_id:b35 Title: Spatiotemporal inconsistency learning for deepfake video detection Year: (2021)
Ref_id:b36 Title: Where deepfakes gaze at? spatial-temporal gaze inconsistency analysis for video face forgery detection Year: (2024)
Ref_id:b37 Title: Ai-generated video detection via spatialtemporal anomaly learning Year: (2024)
Ref_id:b38 Title: Generated video detection via frame consistency: The first benchmark dataset Year: (2024)
Ref_id:b39 Title: On learning multi-modal forgery representation for diffusion generated video detection Year: ()
Ref_id:b40 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b41 Title: Improved techniques for training score-based generative models Year: (2020)
Ref_id:b42 Title: Diffusion models for adversarial purification Year: (2022)
Ref_id:b43 Title: Adversarial purification with score-based generative models Year: (2021)
Ref_id:b44 Title: Detecting adversarial data by probing multiple perturbations using expected perturbation score Year: (2023)
Ref_id:b45 Title: Deep kernel relative test for machine-generated text detection Year: (2025)
Ref_id:b46 Title: Dire for diffusion-generated image detection Year: (2023)
Ref_id:b47 Title: Detecting machine-generated texts by multi-population aware optimization for maximum mean discrepancy Year: (2024)
Ref_id:b48 Title: Generating long videos of dynamic scenes Year: (2022)
Ref_id:b49 Title: Quantum field theory Year: (1999)
Ref_id:b50 Title: Tensor calculus Year: (1978)
Ref_id:b51 Title: Electron spin and probability current density in quantum mechanics Year: (2014)
Ref_id:b52 Title: Fokker-planck equation Year: (1989)
Ref_id:b53 Title: Incompressible flow Year: (2024)
Ref_id:b54 Title: Quantum mechanics: foundations and applications Year: (2013)
Ref_id:b55 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b56 Title: Maximum mean discrepancy test is aware of adversarial attacks Year: (2021)
Ref_id:b57 Title: Detecting adversarial data by probing multiple perturbations using expected perturbation score Year: (2023)
Ref_id:b58 Title: Toward understanding generative data augmentation Year: (2023)
Ref_id:b59 Title: Embedding trajectory for out-of-distribution detection in mathematical reasoning Year: (2024)
Ref_id:b60 Title: Approximate formulae for the percentage points and the probability integral of the non-central χ 2 distribution Year: (1954)
Ref_id:b61 Title: The kinetics human action video dataset Year: (2017)
Ref_id:b62 Title: Seine: Short-to-long video diffusion model for generative transition and prediction Year: (2023)
Ref_id:b63 Title: Pika: Empowering non-programmers to author executable governance policies in online communities Year: (2024)
Ref_id:b64 Title: Msr-vtt: A large video description dataset for bridging video and language Year: (2016)
Ref_id:b65 Title: Muc-5 evaluation metrics Year: (1993)
Ref_id:b66 Title: Using auc and accuracy in evaluating learning algorithms Year: (2005)
Ref_id:b67 Title: Minimum contrast estimators on sieves: exponential bounds and rates of convergence Year: (1998)
Ref_id:b68 Title: Kernel-based tests for likelihood-free hypothesis testing Year: (2023)
Ref_id:b69 Title: Nyström m-hilbert-schmidt independence criterion Year: (2023)
Ref_id:b70 Title: Trustworthy machine learning: From data to models Year: (2025)
Ref_id:b71 Title: Integral probability metrics and their generating classes of functions Year: (1997)
Ref_id:b72 Title: TOHAN: A one-step approach towards few-shot hypothesis adaptation Year: (2021)
Ref_id:b73 Title: Minimax estimation of maximum mean discrepancy with radial kernels Year: (2016)
Ref_id:b74 Title: Minimax optimality of permutation tests Year: (2022)
Ref_id:b75 Title: Youku-mplug: A 10 million large-scale chinese video-language dataset for pre-training and benchmarks Year: (2023)
Ref_id:b76 Title:  Year: (2023)
Ref_id:b77 Title: I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models Year: (2023)
Ref_id:b78 Title: Your personalized image animator via plug-and-play modules in text-to-image models Year: (2024)
Ref_id:b79 Title: Latte: Latent diffusion transformer for video generation Year: (2024)
Ref_id:b80 Title: Open-sora: Democratizing efficient video production for all Year: (2024)
Ref_id:b81 Title: Modelscope text-to-video technical report Year: (2023)
Ref_id:b82 Title:  Year: (2023)
Ref_id:b83 Title: Show-1: Marrying pixel and latent diffusion models for text-to-video generation Year: (2023)
Ref_id:b84 Title: Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models Year: (2023)
Ref_id:b85 Title: Videocrafter1: Open diffusion models for high-quality video generation Year: (2023)
Ref_id:b86 Title: Lavie: High-quality video generation with cascaded latent diffusion models Year: (2024)
Ref_id:b87 Title: Insights into the area under the receiver operating characteristic curve (auc) as a discrimination measure in species distribution modelling Year: (2012)
Ref_id:b88 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b89 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b90 Title: Structural pruning for diffusion models Year: (2023)
Ref_id:b91 Title: Ptq4dit: Posttraining quantization for diffusion transformers Year: (2024)
Ref_id:b92 Title: Absorbing outliers by low-rank components for 4-bit diffusion models Year: (2025)
Ref_id:b93 Title: Mobilediffusion: Instant text-to-image generation on mobile devices Year: (2024)
Ref_id:b94 Title: Light-t2m: A lightweight and fast model for text-to-motion generation Year: (2025)
Ref_id:b95 Title: Difffit: Unlocking transferability of large diffusion models via simple parameterefficient fine-tuning Year: (2023)
Ref_id:b96 Title: Deft: Efficient fine-tuning of diffusion models by learning the generalised h-transform Year: (2024)
Ref_id:b97 Title: Domain generalization enables general cancer cell annotation in single-cell and spatial transcriptomics Year: (2024)
Ref_id:b98 Title: Deep transfer learning enables lesion tracing of circulating tumor cells Year: (2022)
Ref_id:b99 Title: Deepcache: Accelerating diffusion models for free Year: (2024)
Ref_id:b100 Title: Attention is all you need Year: (2017)
Ref_id:b101 Title: Seaweed-7b: Cost-effective training of video generation foundation model Year: (2025)
