Title: Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is Secretly a GAN Discriminator
Abstract: While likelihood-based generative models, particularly diffusion and autoregressive models, have achieved remarkable fidelity in visual generation, the maximum likelihood estimation (MLE) objective, which minimizes the forward KL divergence, inherently suffers from a mode-covering tendency that limits the generation quality under limited model capacity. In this work, we propose Direct Discriminative Optimization (DDO) as a unified framework that integrates likelihood-based generative training and GAN-type discrimination to bypass this fundamental constraint by exploiting reverse KL and self-generated negative signals. Our key insight is to parameterize a discriminator implicitly using the likelihood ratio between a learnable target model and a fixed reference model, drawing parallels with the philosophy of Direct Preference Optimization (DPO). Unlike GANs, this parameterization eliminates the need for joint training of generator and discriminator networks, allowing for direct, efficient, and effective finetuning of a well-trained model to its full potential beyond the limits of MLE. DDO can be performed iteratively in a self-play manner for progressive model refinement, with each round requiring less than 1% of pretraining epochs. Our experiments demonstrate the effectiveness of DDO by significantly advancing the previous SOTA diffusion model EDM, reducing FID scores from 1.79/1.58/1.96 to new records of 1.30/0.97/1.26 on CIFAR-10/ImageNet-64/ImageNet 512×512 datasets without any guidance mechanisms, and by consistently improving both guidance-free and CFG-enhanced FIDs of visual autoregressive models on ImageNet 256×256.

Section: Introduction
Modeling the distribution of high-dimensional data is a fundamental challenge in machine learning (Bishop & Nasrabadi, 2006;Goodfellow et al., 2016). Recent years have witnessed the domination of diffusion (Ho et al., 2020;Song et al., 2021b) and autoregressive (Van Den Oord et al., 2016) paradigms in generative modeling of continuous data and discrete data. They have achieved both theoretical and empirical success in visual tasks including image and video synthesis (Dhariwal & Nichol, 2021;Esser et al., 2021;Ramesh et al., 2021;Karras et al., 2022;Ho et al., 2022;Rombach et al., 2022;Balaji et al., 2022;Gupta et al., 2023;Esser et al., 2024;Brooks et al., 2024;Bao et al., 2024;Tian et al., 2024), forming the cornerstone of large-scale generation systems in the era of AI-generated content.
Diffusion and autoregressive models are representatives of likelihood-based generative models. Compared to Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) which often face unstable training and mode collapse issues, these models are distinguished by their stability, scalability, and generalizability. Besides, their iterative generation process imposes fewer constraints on network Lipschitzness, potentially facilitating superior generative capability over GAN's single-step generation. Likelihood-based generative models aim to learn the underlying data distribution p data by maximizing the likelihood of the observed data under a parameterized probabilistic model p θ , which is equivalent to minimizing the forward Kullback-Leibler (KL) divergence:
max θ E pdata(x) [log p θ (x)] ⇐⇒ min θ D KL (p data ∥ p θ )
However, this maximum likelihood estimation (MLE) objective entails inherent limitations. Forward KL is known to prioritize "mode-covering" and imposes extreme penalties if the model severely underestimates the likelihood of any training sample (Karras et al., 2024a). Under limited model capacity, this property forces the learned density to spread out excessively (Figure 2(a)), potentially leading to blurry samples-a phenomenon commonly observed in Variational Autoencoders (VAEs) (Kingma & Welling, 2014) and in likelihood training of diffusion models (Song et al., 2021a;Kingma et al., 2021;Lu et al., 2022a;Zheng et al., 2023b). Consequently, these models often rely heavily on guidance methods (Ho & Salimans, 2021;Kim et al., 2023a;Karras et al., 2024a) to steer the samples away from unlikely lowprobability regions and toward the core of the data manifold in order to improve overall generation quality. In contrast, GANs, which are theoretically grounded in Jensen-Shannon (JS) divergence or Wasserstein distance (Arjovsky et al., 2017), tend to produce sharper and more realistic samples.
To bypass MLE's mode-covering nature, we aim to leverage GAN-type loss to discriminate between the model and data distributions and produce contrastive forces that guide the model. However, typical GANs require parameterizing extra discriminator networks and alternating optimization, creating engineering complications. Applying GAN-type training trivially to diffusion or autoregressive models is especially inefficient due to their iterative sampling processes.
In this work, we introduce Direct Discriminative Optimization (DDO), a framework that bridges likelihood-based generative models with GANs to push their performance beyond the limits of MLE. Our key insight is to implicitly parameterize the discriminator using the likelihood ratio between a learnable target model and a fixed reference model, both initialized from the pretrained model. This parameterization, inspired by Direct Preference Optimization (DPO) (Rafailov et al., 2024), offers theoretical guarantees of optimality, divergence bounds, and connections to guidance methods. It also enables direct finetuning of the pretrained model without altering the network architecture or inference protocol and supports iterative refinement via multi-round self-play. DDO achieves significant performance gains for both diffusion and autoregressive models sufficiently pretrained on standard image benchmarks. By finetuning state-of-the-art diffusion models EDM (Karras et al., 2022) and EDM2 (Karras et al., 2024b), we achieve unprecedented guidancefree FID scores of 1.30/0.97/1.26 on CIFAR-10/ImageNet-64/ImageNet 512×512 datasets. Finetuning the visual autoregressive model VAR (Tian et al., 2024) on ImageNet 256×256 reduces the FID from 1.92 to 1.73 while removing sampling tricks. Notably, even without classifier-free guidance (CFG) (Ho & Salimans, 2021), the finetuned model achieves an FID of 1.79, surpassing the CFG-enhanced pretrained model while cutting the inference cost by half.
this section cite: ['b6', 'b24', 'b31', 'b75', 'b21', 'b61', 'b38', 'b32', 'b63', 'b3', 'b27', 'b22', 'b9', 'b4', 'b72', 'b25', 'b45', 'b46', 'b1', 'b60', 'b38', 'b72']

Section: Background

this section cite: []

Section: Likelihood-Based Generative Models
Likelihood-based generative models parameterize a probability distribution p θ to learn the data distribution p data , enabling explicit likelihood evaluation and density estimation. Among them, diffusion and autoregressive models are two prominent types that excel in visual generation.
Autoregressive (AR) models (Van Den Oord et al., 2016;Brown et al., 2020) learn discretefoot_0 data distributions via the next-token prediction mechanism:
log p θ (x) = d n=1 log p θ (x (n) |x (<n) )(1)
where d denotes the data dimension (sequence length). It factorizes the joint distribution into a product of conditional probabilities, allowing exact likelihood computation. Each p θ (•|x (<n) ) is parameterized via a Softmax operation over the model's output logits and optimized using cross-entropy loss against the ground-truth token. In visual autoregressive modeling, images are first quantized to discrete tokens within a compact latent space using autoencoders (Van Den Oord et al., 2017;Esser et al., 2021).
Diffusion models (Ho et al., 2020;Song et al., 2021b) learn continuous data distributions by gradually perturbing clean data x 0 ∼ p data with Gaussian noise, which generates a trajectory {x t } T t=0 , and then learning to reverse this process. The forward and backward dynamics can be formulated as either stochastic or ordinary differential equations (SDEs or ODEs) (Song et al., 2021b). The forward process follows a closed-form transition kernel q t|0 (x t |x 0 ) = N (α t x 0 , σ 2 t I) with predefined noise schedule α t , σ t , enabling reparameterization as x t = α t x 0 + σ t ϵ, ϵ ∼ N (0, I). The model is typically parameterized as a noise prediction network ϵ θ (x t , t) trained to estimate ϵ via mean squared error (MSE) regression, which forms an evidence (or variational) lower bound (ELBO) on the likelihood (Song et al., 2021a):
log p θ (x 0 ) ≥ C-E t∼p(t),ϵ∼N (0,I) w(t)∥ϵ θ (x t , t) -ϵ∥ 2 2
(2) where x t = α t x + σ t ϵ, C is a constant irrelevant to θ, and p(t), w(t) are certain time distribution and weighting function. The ELBO provides a reasonable likelihood approximation compared to the exact but cumbersome instantaneous change-of-variable formula in neural ODEs (Chen et al., 2018). Moreover, while the likelihood bound is tight only for specific p(t), w(t), alternative choices share the same optimum and can serve as surrogate objectives (Kingma & Gao, 2024).
From the perspective of score matching (Song et al., 2021b), the optimal noise predictor is linked to the score function s * (x t , t) := ∇ xt log q t (x t ) by ϵ * (x t ) = -σ t s * (x t , t), where q t denotes the marginal distribution at time t in the forward process. Due to the properties of MSE, the network can be parameterized in alternative yet theoretically equivalent forms, such as a velocity predictor (Salimans & Ho, 2022;Zheng et al., 2023b) that estimates the tangent of the diffusion trajectory, commonly known as flow matching (Lipman et al., 2022). In our experiments, we adopt the more generalized F -parameterization introduced in EDM (Karras et al., 2022) (detailed in Appendix C).
this section cite: ['b75', 'b10', 'b76', 'b21', 'b31', 'b14', 'b44', 'b66', 'b50', 'b38']

Section: GANs
GANs (Goodfellow et al., 2014) do not explicitly model the likelihood p θ but instead directly optimize the data generation process through adversarial training. Specifically, the optimization involves an adversarial interplay between a generator network g θ : R dz → R d that maps latent variables z ∈ R dz ∼ p(z) (typically Gaussian noise) into synthetic samples, and a discriminator network d ϕ : R d → [0, 1] that classifies samples as real or fake:
min θ max ϕ E pdata(x) [log d ϕ (x)] + E p θ (x) [log(1 -d ϕ (x))] .
(3) Here p θ (x) is the generator distribution, whose exact density is intractable but can be easily sampled from via x = g θ (z), z ∼ p(z). In the inner loop, the discriminator is optimized using binary cross-entropy loss (also known as noise contrastive estimation (NCE) (Gutmann & Hyvärinen, 2010)), and its optimal solution can be derived as:
d * (x) = p data (x) p data (x) + p θ (x)(4)
under which the minimax game becomes
min θ 2D JS (p data ∥ p θ ) -2 log 2(5)
where
D JS (p ∥ q) = 1 2 D KL (p ∥ p+q 2 ) + 1 2 D KL (q ∥ p+q2
) is the Jensen-Shannon (JS) divergence. This theoretically ensures that the optimal generator distribution matches the data distribution. However, in practice, training instability arises due to gradient vanishing and mode collapse, inspiring variants such as Wasserstein GANs (Arjovsky et al., 2017).
GANs can be incorporated to enhance other generative models. For example, Discriminator Guidance (Kim et al., 2023a) utilizes the gradient information from the discriminator as a corrective term to refine the score function in diffusion models (discussed in Section 4.2). Additionally, GANs are commonly employed as an auxiliary loss to improve one-step generation such as in diffusion distillation (Kim et al., 2023b;Yin et al., 2024;Zhou et al., 2024b).
this section cite: ['b25', 'b28', 'b1', 'b85']

Section: Direct Discriminative Optimization
Motivated by the benefits of adversarial training in enhancing generation quality, we aim to bridge likelihood-based generative models with GANs to derive an alternative training paradigm to MLE. Unlike prior works that incorporate GAN as an auxiliary loss and require additional engineering overhead, our approach seeks to (1) directly optimize likelihood-based generative models without modifying the network architecture, adding extra discriminators, complicating the training procedure or increasing inference costs, and (2) eliminate the need for backpropagation through the sampling process, making it applicable to diffusion and autoregressive models that rely on iterative sampling.
this section cite: []

Section: Your Likelihood-Based Generative Model is
Secretly a Discriminator
Unlike one-step generators that learn a direct mapping from noise to data, likelihood-based generative models are s 𝑝 𝜃 (Target) max 𝜃 𝔼 𝑝 data (𝑥) log 𝑑 𝜃 (𝑥) + 𝔼 𝑝 𝜃 ref (𝑥) log(1 -𝑑 𝜃 𝑥 ) init s 𝑝 𝜃 ref (Pretrained) : frozen : trainable 𝜎 log 𝑝 𝜃 𝑝 𝜃 ref = def 𝑑 𝜃 Discriminator Real/Fake? GAN discriminator loss ~𝑝data ~𝑝𝜃 ref log 𝜎 log 𝑝 𝜃 𝑝 𝜃 ref log 1 -𝜎 log 𝑝 𝜃 𝑝 𝜃 ref Training Dataset Model Samples grounded in the probabilistic definition of the likelihood function p θ , which enables both the generation of samples x ∼ p θ and the evaluation of the likelihood p θ (x), either exactly or approximately, while retaining the tractability of backpropagation through the likelihood computation. This inspires us to utilize the likelihood information embedded in the optimal discriminator (Eqn. (4)).
Specifically, consider a pretrained model p θref as a reference to generate fake samples. The optimal discriminator d θ for
min θ -E pdata(x) [log d θ (x)]-E p θ ref (x) [log(1 -d θ (x))] (6)
can be rewritten as:
d * (x) = p data (x) p data (x) + p θref (x) = σ log p data (x) p θref (x)(7)
where σ(x) = 1 1+e -x is the Sigmoid function. The data distribution p data is available from d * . Therefore, if we parameterize the discriminator d θ using a likelihood-based target generative model p θ as:
d θ (x) := σ log p θ (x) p θref (x)(8)
then the optimal target model that minimizes the GAN discriminator loss matches the data distribution. We formalize this induced objective in the following theorem.
Theorem 3.1 (Optimality). With unlimited model capacity, the optimal likelihood-based model p θ under the objective
min θ L(θ) = -E p data (x) log σ log p θ (x) p θ ref (x) -E p θ ref (x) log 1 -σ log p θ (x) p θ ref (x) (9) satisfies p θ * = p data .
In contrast to previous GAN-based methods that introduce a separate discriminator network d ϕ , our approach implicitly defines the discriminator through a target generative model p θ . While it is theoretically feasible to initialize θ, θ ref arbitrarily and train from scratch, strong initial conditions facilitate optimization (Section 3.2). In practice, we initialize θ, θ ref from widely available pretrained models, promoting steady improvement. We refer to this approach as Direct Discriminative Optimization (DDO), drawing parallels with Direct Preference Optimization (DPO) (Rafailov et al., 2024), which aligns language models with human preferences by expressing the reward model in terms of the likelihood ratio between two policies (discussed in Section 4.1). The DDO pipeline is illustrated in Figure 3.
this section cite: ['b60']

Section: What does the DDO update do?
For a mechanistic understanding of DDO, we can analyze the gradient of the loss function with respect to parameters θ:
∇ θ L(θ) = (1 -d θ (x) ∈[0,1] )(p θ (x) -p data (x) p θ (x)↑ when <0 )∇ θ log p θ (x)dx(
10) Intuitively, gradient descent increases the model likelihood p θ (x) for data points x that satisfy p θ (x) < p data (x), and decreases it otherwise, pushing p θ closer to p data . Furthermore, the gradient magnitude is weighted by both the distance |p θ (x) -p data (x)| and 1 -d θ (x), assigning higher weights to samples discriminated as fake.
this section cite: []

Section: Theoretical Analysis
Apart from the optimality guarantee, we also examine the behavior of the DDO objective when θ is not optimal. Specifically, we investigate the following question:
Is p θ closer to p data with a lower L(θ)?
Under certain assumptions, we can establish bounds on the divergence between p θ and p data in terms of the difference between L(θ) to the optimal loss value L * , as formalized in the following theorem.
D KL (p data ∥ p θ ) ≤ C 1 L(θ) -L * (11
) D KL (p θ ∥ p data ) ≤ C 2 L(θ) -L * (12
)
The assumption of bounded log p θ p θ ref implies that the optimized distribution does not deviate significantly from the reference distribution, which is reasonable when finetuning for a short duration. The assumption of bounded log
p θ ref pdata
imposes a constraint to the reference model regarding its mutual density coverage with the data distribution. We can expect log p θ ref pdata to be lower bounded, i.e., p θref sufficiently covers p data , which aligns with the characteristics of MLE-trained models. Under this condition, the forward KL D KL (p data ∥ p θ ) remains bounded by L(θ) -L * . However, bounding the reverse KL requires an upper bound on p θ ref pdata , which imposes a stronger constraint on p θref .
this section cite: []

Section: Practical Implementation
We introduce several practical techniques that make DDO applicable to high-dimensional real-world data and diffusion models whose likelihood computation is expensive. 2   Generalized Objective with Extra Coeffecients The loglikelihood log p θ (x) of likelihood-based generative models often scales with the data dimension and can reach magnitudes of 10 3 . As the DDO objective in Eqn. (9) involves a Sigmoid operation on log p θ (x), this can lead to gradient vanishing and numerical precision issues. To address this, we add hyperparameters α, β to control the relative weights of loss terms and scale the probability ratio:
L α,β (θ) = -E pdata(x) log σ β log p θ (x) p θref (x) -αE p θ ref (x) log 1 -σ β log p θ (x) p θref (x) (13
) The modified loss retains the same optimization trend, namely, increasing p θ (x) for x ∼ p data and decreasing p θ (x) for x ∼ p θref , but the optimum may "overshoot" the data distribution for β < 1. Specifically, we have: Theorem 3.3. With unlimited model capacity, the optimal likelihood-based generative model θ that minimizes L α,β (θ)
satisfies p θ * ∝ p 1-1/β θ ref p 1/β data for certain α.
This establishes a deep connection with guidance methods (discussed in Section 4.2). In practice, we observe that α and β across a wide range of values 3 yield reasonable 2 A code example is provided in Appendix D. 3 Typical choices are α ∈ [0.5, 50] and β ∈ [0.01, 0.1], while the optimal values depend on the specific model and settings.
performance. We sweep over them for the best results.
Handling Compute-Intensive Likelihood Evaluating the model likelihood for a specific data point can be computationally intensive. In particular, unlike autoregressive models, which only require a single forward pass through the network to compute log p θ (x) (Eqn. (1)) due to the causal structure imposed by attention masks, diffusion models necessitate multiple forward passes over different timesteps to approximate log p θ (x) through the ELBO (Eqn. (2)). Specifically, the log-likelihood ratio in the DDO loss is:
log p θ (x) p θref (x) ≈ E t,ϵ [∆ xt,t,ϵ ](14)
where x t = α t x + σ t ϵ and
∆ xt,t,ϵ = -w(t) ∥ϵ θ (x t , t) -ϵ∥ 2 2 -∥ϵ θref (x t , t) -ϵ∥ 2 2 (15
) We apply Jensen's inequality pointwise to derive an upper bound for the loss using the convexity of the function -a log σ(x) -b log(1 -σ(x)) for any a, b ≥ 0:
L(θ) ≈ -E pdata(x) log σ (E t,ϵ [∆]) -E pθ ref (x) log(1 -σ (E t,ϵ [∆])) ≤ -E t,ϵ E pdata(x) log σ(∆) + E pθ ref (x) log(1 -σ(∆))) (16)
This treatment, analogous to the one used in Diffusion-DPO (Wallace et al., 2024), enables us to approximate the diffusion DDO loss using a single forward pass for each x.
Multi-Round Refinement via Self-Play Due to the practical modifications for applicability, the optimization process of DDO provides useful gradient information in the early stage but does not converge to the data distribution in the final. To maximize the fine-tuning performance, we adopt a multi-round refinement strategy, where the reference model p θref is iteratively updated by replacing it with an improved version from the previous round:
Round n: . . . → p θ * n-1 Reference → σ β log p θn p θ * n-1 Discriminator Round n + 1: → p θ * n Reference → . . .
where θ * n represents the best-performing model across different hyperparameter configurations and training iterations in round n. In each round, the reference model acts as a fixed generator, making the multi-round optimization analogous to the generator-discriminator interplay in GANs. However, unlike GANs, where both networks are explicitly optimized, we never update the reference (generator) model directly. Instead, the generator is obtained from the discriminator in the previous round, leading to a form of self-play. This iterative refinement process is conceptually similar to Iterative DPO (Xu et al., 2023) and SPIN (Chen et al., 2024c), which extend DPO for better language model alignment.
this section cite: ['b77', 'b82']

Section: Discussion
Connection to RL At a high level, DDO enables visual generative models to utilize negative signals from selfgenerated samples-a characteristic deeply rooted in reinforcement learning (RL) that underpins modern language models (Achiam et al., 2023;Guo et al., 2025). Distinguished from works that employ a similar contrastive loss merely to off-the-shelf data (Chen et al., 2024a), DDO can fundamentally improve the base model's ability.
Extension to f -divergence The GAN discriminator loss can be generalized to f -divergences (Nowozin et al., 2016):
D f (p ∥ q) = sup T E p(x) [T (x)] -E q(x) [f * (T (x))] (17
)
where f * is the convex conjugate of f . DDO can be extended to this family as the optimal
T * (x) = f ′ p(x) q(x)
explicitly involves the density ratio.
this section cite: ['b0', 'b26', 'b57']

Section: References
Ref_id:b0 Title: Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b2 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b3 Title: ediffi: Text-to-image diffusion models with an ensemble of expert denoisers Year: (2022)
Ref_id:b4 Title: Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models Year: (2024)
Ref_id:b5 Title: A note on the inception score Year: (2018)
Ref_id:b6 Title: Pattern recognition and machine learning Year: (2006)
Ref_id:b7 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b8 Title: Large scale gan training for high fidelity natural image synthesis Year: (2018)
Ref_id:b9 Title: Video generation models as world simulators Year: (2024)
Ref_id:b10 Title: Language models are few-shot learners Year: (2020)
Ref_id:b11 Title: Masked generative image transformer Year: (2022)
Ref_id:b12 Title: Toward guidance-free ar visual generation via condition contrastive alignment Year: (2024)
Ref_id:b13 Title: Deep compression autoencoder for efficient high-resolution diffusion models Year: (2024)
Ref_id:b14 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b15 Title: Self-play fine-tuning converts weak language models to strong language models Year: (2024)
Ref_id:b16 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b17 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b18 Title: Diffusion models beat GANs on image synthesis Year: (2021)
Ref_id:b19 Title: Density estimation using real nvp Year: (2016)
Ref_id:b20 Title: Implicit generation and modeling with energy based models Year: (2019)
Ref_id:b21 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b22 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b23 Title: Mask-predict: Parallel decoding of conditional masked language models Year: (2019)
Ref_id:b24 Title:  Year: (2016)
Ref_id:b25 Title: Generative adversarial nets Year: (2014)
Ref_id:b26 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b27 Title: Photorealistic video generation with diffusion models Year: (2023)
Ref_id:b28 Title: Noise-contrastive estimation: A new estimation principle for unnormalized statistical models Year: (2010)
Ref_id:b29 Title: GANs trained by a two time-scale update rule converge to a local Nash equilibrium Year: (2017)
Ref_id:b30 Title: Classifier-free diffusion guidance Year: (2021)
Ref_id:b31 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b32 Title: Imagen video: High definition video generation with diffusion models Year: (2022)
Ref_id:b33 Title: simple diffusion: End-to-end diffusion for high resolution images Year: (2023)
Ref_id:b34 Title: The gan is dead; long live the gan! a modern gan baseline Year: (2025)
Ref_id:b35 Title: Scalable adaptive computation for iterative generation Year: (2022)
Ref_id:b36 Title: Scaling up gans for text-to-image synthesis Year: (2023)
Ref_id:b37 Title: Training generative adversarial networks with limited data. Advances in neural information processing systems Year: (2020)
Ref_id:b38 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b39 Title: Guiding a diffusion model with a bad version of itself Year: (2024)
Ref_id:b40 Title: Analyzing and improving the training dynamics of diffusion models Year: (2024)
Ref_id:b41 Title: Refining generative process with discriminator guidance in score-based diffusion models Year: (2023)
Ref_id:b42 Title: Consistency trajectory models: Learning probability flow ode trajectory of diffusion Year: (2023)
Ref_id:b43 Title: Progressive growing of a one-step generator from a low-resolution diffusion teacher Year: (2024)
Ref_id:b44 Title: Understanding diffusion objectives as the elbo with simple data augmentation Year: (2024)
Ref_id:b45 Title: Auto-encoding variational bayes Year: (2014)
Ref_id:b46 Title: Variational diffusion models Year: (2021)
Ref_id:b47 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b48 Title: Autoregressive image generation using residual quantization Year: (2022)
Ref_id:b49 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b50 Title: Flow matching for generative modeling Year: (2022)
Ref_id:b51 Title: Discrete diffusion language modeling by estimating the ratios of the data distribution Year: (2023)
Ref_id:b52 Title: Maximum likelihood training for score-based diffusion odes by high order denoising score matching Year: (2022)
Ref_id:b53 Title: Dpmsolver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b54 Title: Open-magvit2: An open-source project toward democratizing auto-regressive visual generation Year: (2024)
Ref_id:b55 Title: Exploring flow and diffusionbased generative models with scalable interpolant transformers Year: (2024)
Ref_id:b56 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b57 Title: Training generative neural samplers using variational divergence minimization Year: (2016)
Ref_id:b58 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b59 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b60 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2024)
Ref_id:b61 Title: Zero-shot text-toimage generation Year: (2021)
Ref_id:b62 Title: Beyond next-token: Next-x prediction for autoregressive visual generation Year: (2025)
Ref_id:b63 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b64 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b65 Title: The diffusion duality Year: (2025)
Ref_id:b66 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b67 Title: Stylegan-xl: Scaling stylegan to large diverse datasets Year: (2022)
Ref_id:b68 Title: Simplified and generalized masked diffusion for discrete data Year: (2024)
Ref_id:b69 Title: Maximum likelihood training of score-based diffusion models Year: (2021)
Ref_id:b70 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b71 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b72 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b73 Title: Generative infinite-vocabulary transformers Year: (2024)
Ref_id:b74 Title: Score-based generative modeling in latent space Year: (2021)
Ref_id:b75 Title: Pixel recurrent neural networks Year: (2016)
Ref_id:b76 Title: Neural discrete representation learning Year: (2017)
Ref_id:b77 Title: Diffusion model alignment using direct preference optimization Year: (2024)
Ref_id:b78 Title: Diffusion-gan: Training gans with diffusion Year: (2022)
Ref_id:b79 Title: Embedding-free image generation via bit tokens Year: (2024)
Ref_id:b80 Title: Tackling the generative learning trilemma with denoising diffusion GANs Year: (2022)
Ref_id:b81 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2024)
Ref_id:b82 Title: Some things are more cringe than others: Preference optimization with the pairwise cringe loss Year: (2023)
Ref_id:b83 Title: Disco-diff: Enhancing continuous diffusion models with discrete latents Year: (2024)
Ref_id:b84 Title: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models Year: (2025)
Ref_id:b85 Title: Improved distribution matching distillation for fast image synthesis Year: (2024)
Ref_id:b86 Title: Vector-quantized image modeling with improved vqgan Year: (2021)
Ref_id:b87 Title: Language model beats diffusion-tokenizer is key to visual generation Year: (2023)
Ref_id:b88 Title: Randomized autoregressive visual generation Year: (2024)
Ref_id:b89 Title: Representation alignment for generation: Training diffusion transformers is easier than you think Year: (2024)
Ref_id:b90 Title: Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization Year: ()
Ref_id:b91 Title: Sageattention3: Microscaling fp4 attention for inference and an exploration of 8-bit training Year: (2025)
Ref_id:b92 Title: Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration Year: (2025)
Ref_id:b93 Title: Accurate sparse attention accelerating any model inference Year: (2025)
Ref_id:b94 Title: Improved diffusion ode solver with empirical model statistics Year: (2023)
Ref_id:b95 Title: Improved techniques for maximum likelihood estimation for diffusion odes Year: (2023)
Ref_id:b96 Title: Masked diffusion models are secretly timeagnostic masked models and exploit inaccurate categorical sampling Year: (2024)
Ref_id:b97 Title: Transfusion: Predict the next token and diffuse images with one multi-modal model Year: (2024)
Ref_id:b98 Title: Adversarial score identity distillation: Rapidly surpassing the teacher in one step Year: (2024)
