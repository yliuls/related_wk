Title: Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes
Abstract: In this paper, we introduce a geometric framework to analyze memorization in diffusion models through the sharpness of the log probability density. We mathematically justify a previously proposed score-difference-based memorization metric by demonstrating its effectiveness in quantifying sharpness. Additionally, we propose a novel memorization metric that captures sharpness at the initial stage of image generation in latent diffusion models, offering early insights into potential memorization. Leveraging this metric, we develop a mitigation strategy that optimizes the initial noise of the generation process using a sharpness-aware regularization term. The code is publicly available at https://github.com/Dongjae0324/  sharpness_memorization_diffusion.

Section: Introduction
Recent advancements in generative models have significantly improved data generation across various domains, including image synthesis (Rombach et al., 2022), natural language processing (Achiam et al., 2023;Touvron et al., 2023), and molecular design (Alakhdar et al., 2024). Among these, diffusion models (Ho et al., 2020;Song et al., 2021c) have emerged as powerful frameworks, achieving state-ofthe-art results by iteratively refining noisy samples to approximate complex data distributions (Song et al., 2021b;Saharia et al., 2022;Rombach et al., 2022).
Despite their successes, diffusion models suffer from memorization, where they replicate training samples instead of generating novel outputs (Carlini et al., 2023;Somepalli et al., 2023b;Webster, 2023). This issue is especially con-cerning when models are trained on sensitive data, leading to privacy risks (Orrick, 2023;Joseph Saveri, 2023). Addressing memorization is critical for ensuring the responsible deployment of generative models in real-world applications.
Previous work has sought to analyze memorization using various approaches, including probability manifold analysis via Local Intrinsic Dimensionality (LID) (Ross et al., 2024;Kamkari et al., 2024), spectral characterizations (Ventura et al., 2024;Stanczuk et al., 2024), and score-based discrepancy measures (Wen et al., 2024). Additionally, attentionbased methods have been used to examine memorization at the feature level (Ren et al., 2024;Chen et al., 2024).
In this work, we propose a general sharpness-based framework for understanding memorization in diffusion models. Specifically, we observe that memorization correlates with regions of sharpness in the probability landscape, which can be quantified via the Hessian of the log probability. Large negative eigenvalues of the Hessian indicate sharp, isolated regions in the learned distribution, providing a mathematically grounded explanation of memorization. Furthermore, we show that the trace-based eigenvalue statistics can serve as a robust early-stage indicator of memorization, enabling detection at the initial sampling step of generation.
Our framework also provides a justification for score based metric by interpreting it through the lens of sharpness, reinforcing its validity as a memorization detection metric. Building on this, we propose an enhanced sharpness measure with additional Hessian components, improving sensitivity, particularly at the earliest stages of sampling.
Beyond detection, we introduce an inference-time mitigation strategy that reduces memorization by selecting initial diffusion noise from regions of lower sharpness. Our method, Sharpness-Aware Initialization for Latent Diffusion (SAIL), utilizes our sharpness metric to identify initializations that avoid trajectories leading to memorization. By simply adjusting the initial noise, SAIL steers the diffusion process toward smoother probability regions, mitigating memorization without requiring retraining. Unlike prompt modifications, which can negatively affect generation quality, SAIL reduces memorization by carefully selecting the initial noise while fully preserving the conditioning inputs.
We validate our approach through experiments on a 2D toy dataset, MNIST, and Stable Diffusion. Our results show that Hessian eigenvalues effectively differentiate memorized from non-memorized samples, and our sharpness measure provides a reliable metric for memorization detection. Additionally, we demonstrate that SAIL mitigates memorization while preserving generation quality, offering a simple yet effective solution for reducing memorization.
In summary, our key contributions are:
• We propose a sharpness-based framework for analyzing memorization in diffusion models, examining the patterns of Hessian eigenvalues and their aggregate statistics to characterize memorized samples.
• We provide a theoretical justification for the memorization detection metric introduced by Wen et al. (2024) through sharpness analysis.
• We introduce a new sharpness measure that enables early-stage memorization detection during the diffusion process.
• We propose SAIL, a simple yet effective mitigation strategy that selects initial noise leading to smoother probability regions, reducing memorization without altering model parameters or prompts.
this section cite: ['b30', 'b1', 'b41', 'b3', 'b12', 'b32', 'b30', 'b6', 'b24', 'b16', 'b31', 'b17', 'b42', 'b39', 'b29', 'b8']

Section: Related works
Understanding and Explaining Memorization. The memorization behavior of diffusion models (DMs) has been extensively studied (Somepalli et al., 2023b;Carlini et al., 2023;Wen et al., 2024), with prior work examining contributing factors such as prompt conditioning (Somepalli et al., 2023b), data duplication (Carlini et al., 2023;Somepalli et al., 2023a), and dataset size or complexity (Gu et al., 2023). Some studies have approached this issue from a geometric standpoint, drawing on the manifold learning conjecture (Fefferman et al., 2016;Pope et al., 2021), where exact memorization is associated with data points lying on a zero-dimensional manifold (Ross et al., 2024;Ventura et al., 2024;Pidstrigach, 2022).
This geometric perspective has led to efforts to estimate Local Intrinsic Dimensionality (LID) at the sample level (Stanczuk et al., 2024;Kamkari et al., 2024;Horvat & Pfister, 2024;Wenliang & Moran, 2023;Tempczyk et al., 2022), which has been used to characterize memorization (Ross et al., 2024;Ventura et al., 2024).
While our work is inspired by prior studies, it introduces several key distinctions. Unlike approaches that define memorization in terms of overall model behavior (Yoon et al., 2023;Gu et al., 2023), we focus on sample-specific behavior manifested in the learned probability density. Although our perspective is conceptually aligned with recent geometric interpretations (Ross et al., 2024;Bhattacharjee et al., 2023), our methodology diverges fundamentally by analyzing sharpness in the learned density, without relying on assumptions about an inaccessible ground-truth distribution. In contrast to manifold-based analyses that track variations in individual feature components (Ventura et al., 2024;Achilli et al., 2024), we show that sharpness, treated as an aggregated statistic, can be effectively estimated and used for detecting memorization. Moreover, unlike LIDbased methods (Ross et al., 2024) that are restricted to the final denoising step, our approach reveals that memorized samples persistently occupy high-sharpness regions throughout the diffusion process. This allows for earlier detection and targeted intervention, enabling a more proactive and interpretable strategy for mitigating memorization.
Detecting and Mitigating Memorization. Detecting and mitigating memorization during the generative process remains a challenging problem. Previous studies have explored various approaches to identify prompts that induce memorization in text-conditional DMs by comparing generated images to training data. For instance, Somepalli et al. (2023a) employed feature-based detectors like SSCD (Pizzi et al., 2022) and DINO (Caron et al., 2021), while Carlini et al. (2023) and Yoon et al. (2023) used calibrated ℓ 2 distance in pixel space to quantify memorization. Webster (2023) developed both white-box and black-box attacks, analyzing edges and noise patterns in generated images. While these methods provide valuable insights, their computational cost makes real-time detection impractical. To address this limitation, heuristic-based alternatives have been proposed. Wen et al. (2024) introduced a metric based on the magnitude of text-conditional score predictions, leveraging the observation that memorized prompts exhibit stronger text guidance. Similarly, Ren et al. (2024) identified memorization via anomalously high attention scores on specific tokens, while Chen et al. (2024) focused on patterns in end tokens of text embeddings.
Since memorization in DMs is often linked to specific text prompts, most mitigation strategies have focused on modifying prompts or adjusting attention mechanisms to reduce their influence (Wen et al., 2024;Ren et al., 2024;Ross et al., 2024). For example, Ross et al. (2024) rephrased prompts using GPT-4 to mitigate memorization. However, these interventions often degrade image quality or compromise user intent by altering model-internal components.
In contrast, our approach offers a principled and modelagnostic alternative by optimizing the initial noise input instead of modifying the text prompt or trained model parameters. By selecting initial noise that leads to smoother probability regions, our method mitigates memorization while preserving both user prompts and model fidelity, ensuring minimal impact on generation quality.
this section cite: ['b6', 'b6', 'b10', 'b9', 'b27', 'b31', 'b42', 'b25', 'b39', 'b17', 'b13', 'b40', 'b31', 'b42', 'b10', 'b31', 'b5', 'b42', 'b2', 'b31', 'b26', 'b7', 'b29', 'b31', 'b31']

Section: Preliminaries
Score-based Diffusion Models. Diffusion models (DMs) (Sohl-Dickstein et al., 2015;Ho et al., 2020;Song et al., 2021c) generate images by iteratively refining random noise into samples that approximate the data distribution p 0 (x 0 ). The process begins with the forward process, where the training data is progressively corrupted by the addition of Gaussian noise. At each timestep t, the conditional distribution of the noisy data is given by:
q t|0 (x t |x 0 ) = N (x t | √ α t x 0 , (1 -α t )I),
where x t represents the noisy data at timestep t, and α t decreases monotonically over time in the variance-preserving case, with α T becoming sufficiently small such that the resulting distribution closely resembles pure Gaussian noise:
q T |0 (x T |x 0 ) ≈ N (0, I).
This process can be equivalently represented as a stochastic differential equation (SDE):
dx t = f (x t , t)dt + g(t)dw t ,
where w t is a standard Brownian motion.
The reverse process, which reconstructs the data distribution p 0 (x 0 ) from noise, is then formulated as:
dx t = f (x t , t) -g 2 (t)∇ xt log p t (x t ) dt + g(t)d wt ,
where wt denotes a standard Brownian motion in reverse time, and p t (x t ) is the marginal distribution at timestep t.
The only unknown term in the reverse process is the score function over timesteps, ∇ xt log p t (x t ) := s(x t ), which is often parameterized by a neural network with s θ (x t ).
In many applications the data x 0 is often represented with an associated label c (e.g., prompts or class labels). In these scenarios, the additional condition c is incorporated into the model as s θ (x t , c), allowing it to estimate the score of the conditional density ∇ xt log p t (x t |c) := s(x t , c) via classifier free guidance (Ho & Salimans, 2021). In this work, we examine the memorization by analyzing the Hessian of log p t (x t ), which corresponds to the Jacobian of the score function. We denote it as H(x t ) := ∇ 2 xt log p t (x t ) for the unconditional case and H(x t , c) := ∇ 2 xt log p t (x t |c) for the conditional case. The Hessian estimated by the model is denoted as H θ (x t ) and H θ (x t , c).
this section cite: ['b33', 'b12', 'b11']

Section: Understanding Memorization via Sharpness

this section cite: []

Section: Memorization: Sharpness in Probability Landscape
Sharpness quantifies the concentration of learned log density log p(x) around point x, which can be analyzed through the eigenvalues of its Hessian matrix. Large negative eigenvalues indicate sharp peaks in the distribution, suggesting memorization of specific data points. Conversely, small magnitude or positive eigenvalues characterize broader, smoother regions that facilitate better generalization.
Local Intrinsic Dimensionality (LID) (Kamkari et al., 2024) quantifies the effective dimensionality of a point in its local neighborhood, characterizing local sample space geometry. At the final generation step (t ≈ 0), LID serves as a memorization indicator (Ross et al., 2024). Exact Memorization (EM) shows near-zero LID, indicating pure reproduction of training samples, while Partial Memorization (PM) exhibits small but nonzero LID, reflecting limited stylistic variations. In contrast, properly generalized samples demonstrate moderate LID values, indicating more diverse representations.
While both sharpness and LID characterize curvature properties of probability density, LID is limited to analyzing sample space at t ≈ 0, where the generated image emerges. In contrast, we extend memorization detection across all timesteps by leveraging sharpness via Hessian eigenvalues as a more versatile metric, enabling continuous monitoring throughout the generative process rather than relying solely on final output characteristics. Partial Mem Non Mem Exact Mem ------------------Figure 3: Left: Eigenvalue
distribution of H θ (x t , c) across memorization categories in Stable Diffusion v1.4 at initial sampling step (t = T -1) with range clipped. (top) 30 prompts per category with identical initialization. (bottom) Fixed prompt set with three different initializations. Both plots reveal stronger memorization correlates with fewer non-negative eigenvalues. Right: Eigenvalue distribution of H θ (x t , c) across memorization categories in Stable Diffusion v1.4 at final sampling step (t = 1). Generated images shown with original training counterparts (outlined in red). Eigenvalues are approximated via Arnoldi iteration (Arnoldi, 1951), details in Appendix A.2.
this section cite: ['b17', 'b31']

Section: Mem
Non-mem  To validate our approach on real data, we conduct experiments on MNIST by inducing memorization through repeated exposure to a single "9" image while maintaining all "3" images as a general class (Figure 2). The eigenvalue distributions at t = 1 clearly differentiate memorized from nonmemorized samples: memorized samples show consistently large negative eigenvalues indicating sharp peaks, while non-memorized samples exhibit positive eigenvalues, reflecting locally convex regions that allow sample variations. Notably, these clear distributional differences emerged even at the initial sampling step (t = T -1), confirming that sharpness-based memorization detection is effective from the very beginning of the generation process.
We further validate our approach on Stable Diffusion (Rombach et al., 2022), analyzing its 16, 384-dimensional latent space. Figure 3 reveals distinct patterns in both the number of non-negative eigenvalues and the magnitude of negative eigenvalues across different memorization categories (EM, PM, and non-memorized) at both initial and final sampling step. These patterns not only align with LID-based analysis at t ≈ 0 but also demonstrate sharpness as a more generalizable memorization measure, capturing distinctive characteristics at generation onset.
this section cite: []

Section: Score Norm as a Sharpness Measure
While sharpness serves as a fundamental measure of memorization in generative models, directly computing the full spectrum of Hessian eigenvalues in high-dimensional distri-butions, such as those in Stable Diffusion, is computationally intractable. A practical alternative is to approximate sharpness using the trace of the Hessian, a single scalar quantity that represents the sum of all eigenvalues, where large negative traces indicate sharp, highly localized regions.
A key observation is that the norm of the score function ∥s(x)∥ inherently encodes information about the probability landscape's curvature. In Gaussian distributions, the score norm is directly connected to the Hessian trace, as shown in the following result. (Appendix B.2).
Lemma 4.1. For a Gaussian vector x ∼ N (µ, Σ),
E ∥s(x)∥ 2 = -tr(H(x)),
where H(x) ≡ -Σ -1 is the Hessian of the log density.
This result extends to non-Gaussian distributions under mild regularity assumptions (Appendix B.2). For theoretical clarity and ease of analysis, however, we focus on the Gaussian case. While the distribution x t in diffusion processes is not strictly Gaussian at every timestep, recent studies show that at moderate to high noise levels, corresponding to the early and middle stages of the reverse process-the learned score is predominantly governed by its Gaussian component (Wang & Vastola, 2024). This approximation is further justified in latent diffusion models, where the latent variable z t is explicitly regularized toward a Gaussian prior (Kingma, 2013;Rombach et al., 2022), despite the complexity of the original data distribution.
Under this Gaussian assumption at relevant sampling steps, the score norm ∥s θ (x t )∥ 2 provides an unbiased estimate of the negative Hessian trace -tr(H θ (x t )), offering an efficient measure of the sharpness of the probability landscape.  even in the later stages of the diffusion process, suggesting that score norm can serve as a computationally efficient sharpness measure throughout generation. This perspective provides a theoretical foundation for interpreting sharpness in generative models through score norm based statistic, enabling efficient memorization detection and analysis without requiring costly Hessian eigenvalue decompositions. 4.3. Wen's Metric as a Sharpness Measure Wen et al. (2024) characterized memorization through the norm of difference between conditional and unconditional score functions:
∥s ∆ θ (x t )∥ := ∥s θ (x t , c) -s θ (x t )∥.
This difference vector s ∆ θ (x t ) determines the sampling direction in classifier-free guidance. Their approach is based on the observation that memorized prompts consistently guide generation toward specific images, resulting in larger magnitudes of s ∆ θ (x t ) due to stronger text-driven guidance. While the theoretical foundations of this heuristic remain to be fully understood, it has proven to be one of the most effective detection metrics thus far.
Notably, the structure of ∥s ∆ θ (x t )∥ bears a strong resemblance to the score norm, which we previously identified as a measure of sharpness. This similarity hints at the possibility of interpreting Wen's metric as a sharpness measure, encapsulating the impact of conditioning on the probability distribution's curvature. To rigorously establish this connection, we proceed to analyze the Hessian of the log-density, following the same approach as in the preceding analysis.
Lemma 4.2. For x ∼ N (µ, Σ) and x|c ∼ N (µ c , Σ c ):
E x∼p(x|c) ∥s(x, c) -s(x)∥ 2 = ∥H(x)(µ -µ c )∥ 2 + tr((H(x) -H c (x)) 2 H -1 c (x)),
where H(x) ≡ -Σ -1 and H c (x) ≡ -Σ -1 c . Additionally, when Σ and Σ c commute (i.e., ΣΣ c = Σ c Σ) and mean vectors are the same (µ = µ c ), this reduces to
E x∼p(x|c) ∥s(x, c) -s(x)∥ 2 = d i=1 (λ i -λ i,c ) 2 λ i,c ,
where λ i , λ i,c are eigenvalues of H(x) and H c (x).
This result demonstrates that Wen's metric measures sharpness differences through squared eigenvalue differences of the conditional and unconditional Hessian. During early timesteps, when the latent distribution remains close to an isotropic Gaussian, this metric directly captures the extent to which conditioning induces sharpness. At later timesteps, when Σ t and Σ t,c do not generally commute, the metric can be interpreted through generalized eigenvalues, revealing how conditioning sharpens the learned distribution in similar manner. The details are provided in Appendix A.3. Figure 5 shows the eigenvalue disparities between conditional and unconditional Hessians across timesteps, revealing how conditioning shapes the probability distribution's geometry. For memorized samples, the eigenvalue gap is notably large, showing that conditioning creates a more constrained probability landscape. At intermediate timesteps (t = 20), the differences are subtle but noticeable, indicating early conditioning effects. Near the end (t = 1), the eigenvalue gap widens substantially, demonstrating conditioning's growing influence on the learned density. In contrast, non-memorized samples show minimal eigenvalue variations throughout, indicating little conditioning influence. These findings support our theoretical framework and confirm Wen's metric effectively measures sharpness.
this section cite: ['b18', 'b30']

Section: Upscaling Eigenvalue Statistics via Hessian
While Wen's metric reveals eigenvalue disparities at intermediate timesteps, identifying and mitigating memorization during the initial generation stage remains challenging. The probability landscape maintains a nearly uniform character since the latent distribution approximates an isotropic Gaussian, making structural sharpness differences subtle. Conventional metrics struggle to capture these fine-grained distributional variations, limiting early-stage applications.
To address this limitation, we introduce a curvature-aware scaling that enhances Wen's metric through Hessian-based weighting. By multiplying the Hessian with the score function, we amplify high-curvature directions, rendering sharp regions more distinguishable within a smooth probability landscape. This approach significantly improves the eigenvalue gap at the earliest generation stage, advancing memorization detection in the diffusion process. The following lemma shows that the Hessian-score product provides an amplified measure of the Hessian trace, thereby increasing sensitivity to distributional sharpness. Lemma 4.3. For a Gaussian vector x ∼ N (µ, Σ),
E ∥H(x)s(x)∥ 2 = -tr((H(x)) 3 )
where H(x) ≡ -Σ is the Hessian of the log density.
This relationship, empirically verified in Figure 4, demonstrates the curvature-sensitive scaling effect of the Hessian score product. Building on this principle, we propose an enhanced version of Wen's metric that improves early-stage sensitivity through second-order sharpness characterization:
∥H ∆ θ (x t , c)s ∆ θ (x t , c)∥ 2 , where H ∆ θ (x t , c) = H θ (x t , c) -H θ (x t ), and s ∆ θ (x t , c) = s θ (x t , c) -s θ (x t ).
To provide intuition, assuming identical means (µ = µ c ) and that Σ t and Σ t,c commute, the expected value of our metric simplifies to:
E xt∼pt(xt|c) ∥H ∆ θ (x t , c)s ∆ θ (x t , c)∥ 2 = d i=1 (λ i -λ i,c ) 4 λ i,c
, where λ i , λ i,c are eigenvalues of H(x t ) and H(x t , c).
Compared to Wen's metric in Lemma 4.2, this refinement substantially improves sensitivity by amplifying the difference in sharpness, thereby enabling more effective detection of memorization at earlier stages.
this section cite: []

Section: Detecting Memorization in Stable Diffusion
Experimental Setup. To evaluate our metric, we use 500 memorized prompts identified by Webster (2023) for Stable Diffusion v1.4, and 219 prompts for v2.0. As a complementary set, we include 500 non-memorized prompts sourced from COCO (Lin et al., 2014), Lexica (Lexica, 2024), Tuxemon (HuggingFace, 2024), and GPT-4 (Achiam et al., 2023) Similarly, in v2.0, our approach attains an AUC of 0.991 without full-step sampling, underscoring its effectiveness.
Importantly, our metric can be efficiently computed using Hessian-vector products without explicitly forming the full Hessian matrix. Leveraging automatic differentiation frameworks such as PyTorch, a single Hessian-vector product suffices for detection, incurring minimal overhead.
this section cite: ['b21', 'b1']

Section: Sharpness Aware Memorization Mitigation

this section cite: []

Section: Sharpness Aware Initialization Sampling
Motivation. In Section 4, we observed that memorized samples exhibit a sharp conditional density, p t (x t |c), even at the very beginning of the generation process (i.e., at t = T -1; note that sampling proceeds in reverse order, starting from t = T ). This is substantiated by the strong detection performance of both Wen's metric and our metric at the initial sampling step, which quantifies the sharpness gap between p t (x t |c) and p t (x t ). This phenomenon, linked to the deterministic nature of ODE samplers (a one-to-one mapping between noise and image), implies that initializations from sharper densities remain in sharper regions at each intermediate timestep of the generation process, thereby increasing the likelihood of producing memorized images. In contrast, initializations from smoother regions tend to yield non-memorized images.
Thus, we argue that sampling with noise from smoother densities could effectively mitigate memorization. While manually searching for such initializations is a straightforward approach, it becomes infeasible in high-dimensional Gaussian space due to the sheer size and complexity of the search domain. Consequently, we propose to directly optimize the initial noise x T as a more scalable and systematic way to address this challenge.
this section cite: []

Section: Sharpness Aware Initialization.
We propose Sharpness-Aware Initialization for Latent Diffusion (SAIL), an inference-time mitigation strategy that optimizes initializations x T by minimizing the sharpness gap at the starting step (t = T -1). SAIL identifies initial seeds on non-memorized sampling trajectories by selecting x T from smoother regions while maintaining a reasonable density under the isotropic Gaussian prior. The objective function is defined as: where p G is the density of an isotropic Gaussian distribution.
∥H ∆ θ (x T )s ∆ θ (x T )∥ 2 -α log p G (x T ),
While ∥H ∆ θ (x T )s ∆ θ (x T )∥ 2 can be efficiently computed using Hessian-vector products, the gradient backpropagation required for optimization introduces computational overhead. To overcome the burden, we approximate the term using a Taylor expansion around x T :
∥H ∆ θ (x T )s ∆ θ (x T )∥ 2 ≈ s ∆ θ x T + δs ∆ θ (x T ) -s ∆ θ (x T ) 2 δ 2 .
This leads to the final objective for SAIL:
L SAIL (x T ) := ∥s ∆ θ x T + δs ∆ θ (x T ) -s ∆ θ (x T )∥ 2 + α∥x T ∥ 2
, where α balances the sharpness of the density and the original likelihood. To ensure initializations remain close to the Gaussian distribution, we employ early stopping based on a threshold ℓ thres , limiting number of optimization steps.
this section cite: []

Section: Mitigating Memorization in Stable Diffusion.
Experimental Setup. To evaluate mitigation strategies, we use the same memorized prompt set employed in the detection experiments described in Section 4.5. However, since verifying mitigation effects requires access to training images, we exclude prompts whose corresponding training samples are unavailable. Further details are in Appendix D.
We employ two key metrics following (Wen et al., 2024;Somepalli et al., 2023a): the SSCD similarity score (Pizzi et al., 2022), which quantifies memorization by comparing model-based features of generated images to their corresponding training data, and the CLIP score (Radford et al., 2021), which evaluates prompt-image alignment. Results are averaged over five generations per prompt.
For comparison, we implement four recent mitigation algorithms. Somepalli et al. (2023b) propose Random Token Addition (RTA) and Random Number Addition (RNA), which perturb original prompts to mitigate memorization. Wen et al. (2024) introduce a method that optimizes text embeddings to reduce the influence of memorization-inducing tokens. Ren et al. (2024) propose a strategy that adjusts attention scores of text embeddings for mitigation.
For a fair comparison, all methods are evaluated using five distinct hyperparameter settings and optimized with the Adam optimizer at a learning rate of 0.05. For a detailed experimental settings, refer to Appendix D.2.
Results. Figure 6 (left) demonstrates that SAIL significantly improves both SSCD and CLIP metrics for Stable Diffusion v1.4 and v2.0. By optimizing the noise initialization x T without altering model components like text embeddings or attention weights, SAIL effectively mitigates memorized content while preserving model behavior and user prompts, ensuring high-quality, non-memorized outputs.
The advantage of SAIL is evident in Figure 6 (right), where it generates images that faithfully preserve key prompt details, such as celebrity names and primary objects. In contrast, methods that modify text-conditional components often reduce the influence of those components during mitigation, leading to degraded alignment with the original prompt and potentially diminishing user utility. Additional qualitative results for algorithms are provided in Appendix E.
this section cite: ['b26', 'b28', 'b29']

Section: Conclusion
We propose a sharpness-based framework for detecting and mitigating memorization in diffusion models. Our analysis identifies Hessian-based sharpness as a reliable indicator of memorization and introduces an efficient proxy based on the score norm. This perspective also provides a theoretical interpretation of the memorization detection metric proposed by Wen et al. (2024). Building on this foundation, we introduce Sharpness-Aware Initialization for Latent Diffusion (SAIL), an inference-time method that reduces memorization by selecting low-sharpness initial noise. Experiments on synthetic 2D data, MNIST, and Stable Diffusion demonstrate that our approach enables early detection and effective mitigation, all without degrading generation quality.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2005)
Ref_id:b1 Title: Gpt-4 technical report Year: (2023)
Ref_id:b2 Title: Losing dimensions: Geometric memorization in generative diffusion Year: (2024)
Ref_id:b3 Title: Diffusion models in de novo drug design Year: (2024)
Ref_id:b4 Title: The principle of minimized iterations in the solution of the matrix eigenvalue problem Year: (1951)
Ref_id:b5 Title: Datacopying in generative models: a formal framework Year: (2023)
Ref_id:b6 Title: Extracting training data from diffusion models Year: (2023)
Ref_id:b7 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b8 Title: Exploring local memorization in diffusion models via bright ending attention Year: (2024)
Ref_id:b9 Title: Testing the manifold hypothesis Year: (2016)
Ref_id:b10 Title: On memorization in diffusion models Year: (2023)
Ref_id:b11 Title: Classifier-free diffusion guidance Year: (2021)
Ref_id:b12 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b13 Title: On gauge freedom, conservativity and intrinsic dimensionality estimation in diffusion models Year: (2024)
Ref_id:b14 Title:  Year: (2024)
Ref_id:b15 Title: Estimation of non-normalized statistical models by score matching Year: (2005)
Ref_id:b16 Title: Stable diffusion litigation Year: (2023)
Ref_id:b17 Title: A geometric view of data complexity: Efficient local intrinsic dimension estimation with diffusion models Year: (2024)
Ref_id:b18 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b19 Title: An iteration method for the solution of the eigenvalue problem of linear differential and integral operators Year: (1950)
Ref_id:b20 Title:  Year: (2024)
Ref_id:b21 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b22 Title: Maximum likelihood training for score-based diffusion odes by high order denoising score matching Year: (2022)
Ref_id:b23 Title: Estimating high order gradients of the data distribution by denoising Year: (2021)
Ref_id:b24 Title:  Year: (2023)
Ref_id:b25 Title: Score-based generative models detect manifolds Year: (2022)
Ref_id:b26 Title: A self-supervised descriptor for image copy detection Year: (2022)
Ref_id:b27 Title: The intrinsic dimension of images and its impact on learning Year: (2021)
Ref_id:b28 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b29 Title: Unveiling and mitigating memorization in textto-image diffusion models through cross attention Year: (2024)
Ref_id:b30 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b31 Title: A geometric framework for understanding memorization in generative models Year: (2024)
Ref_id:b32 Title: Photorealistic text-to-image diffusion models with deep language understanding Year: (2022)
Ref_id:b33 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b34 Title: Diffusion art or digital forgery? investigating data replication in diffusion models Year: (2023)
Ref_id:b35 Title: Understanding and mitigating copying in diffusion models Year: ()
Ref_id:b36 Title: Denoising diffusion implicit models Year: ()
Ref_id:b37 Title: Maximum likelihood training of score-based diffusion models Year: ()
Ref_id:b38 Title: Score-based generative modeling through stochastic differential equations Year: ()
Ref_id:b39 Title: Diffusion models encode the intrinsic dimension of data manifolds Year: (2024)
Ref_id:b40 Title: Lidl: Local intrinsic dimension estimation using approximate likelihood Year: (2022)
Ref_id:b41 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b42 Title: random matrices and spectral gaps: The geometric phases of generative diffusion Year: (2024)
