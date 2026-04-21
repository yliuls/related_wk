Title: UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control
Abstract: Recent advances in diffusion bridge models leverage Doob's h-transform to establish fixed endpoints between distributions, demonstrating promising results in image translation and restoration tasks. However, these approaches frequently produce blurred or excessively smoothed image details and lack a comprehensive theoretical foundation to explain these shortcomings. To address these limitations, we propose UniDB, a unified framework for diffusion bridges based on Stochastic Optimal Control (SOC). UniDB formulates the problem through an SOC-based optimization and derives a closed-form solution for the optimal controller, thereby unifying and generalizing existing diffusion bridge models. We demonstrate that existing diffusion bridges employing Doob's h-transform constitute a special case of our framework, emerging when the terminal penalty coefficient in the SOC cost function tends to infinity. By incorporating a tunable terminal penalty coefficient, UniDB achieves an optimal balance between control costs and terminal penalties, substantially improving detail preservation and output quality. Notably, UniDB seamlessly integrates with existing diffusion bridge models, requiring only minimal code modifications. Extensive experiments across diverse image restoration tasks validate the superiority and adaptability of the proposed framework. Our code is available at https://github.com/  UniDB-SOC/UniDB/.⇒ x T ∼ N e ft:T x t + me fT ht:T , e 2 fT ḡ2 t:T I , (43) ⇒ ∇ xt log p(x T |x t ) = -∇ xt (x T -e ft:T x t -me fT ht:T ) 2 2e 2 fT ḡ2 t:T

Section: Introduction
The diffusion model has been extensively utilized across a range of applications, including image generation and edit-* Equal contribution 1 ShanghaiTech University 2 MoE Key Laboratory of Intelligent Perception and Human-Machine Collaboration 3 Fudan University. Correspondence to: Ye Shi <shiye@shanghaitech.edu.cn>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
ing (Ho et al., 2020;Kawar et al., 2022;Song et al., 2020;Xia et al., 2023;Li et al., 2023), imitation learning (Wu et al., 2024;Chi et al., 2023;Ze et al., 2024) and reinforcement learning (Yang et al., 2023;Ding et al., 2024a), etc. Despite its versatility, the standard diffusion model faces limitations in transitioning between arbitrary distributions due to its inherent assumption of a Gaussian noise prior. To overcome this problem, diffusion models (Dhariwal & Nichol, 2021;Ho & Salimans, 2022;Murata et al., 2023;Ding et al., 2024b;Chung et al., 2022;Tang et al., 2024) often rely on meticulously designed conditioning mechanisms and classifier/loss guidance to facilitate conditional sampling and ensure output alignment with a target distribution. However, these methods can be cumbersome and may introduce manifold deviations during the sampling process. Meanwhile, Diffusion Schrödinger Bridge (Shi et al., 2024;De Bortoli et al., 2021;Somnath et al., 2023) involves constraints that hinder direct optimization of the KL divergence, resulting in slow convergence and limited model fitting capability.
To address this challenge, DDBMs (Zheng et al., 2024) proposed a diffusion bridge model using Doob's h-transform. This framework is specifically designed to establish fixed endpoints between two distinct distributions by learning the score function of the diffusion bridge from data, and then solving the stochastic differential equation (SDE) based on these learned scores to transition from one endpoint distribution to another. However, the forward SDE in DDBMs lacks the mean information of the terminal distribution, which restricts the quality of the generated images, particularly in image restoration tasks. Subsequently, GOUB (Yue et al., 2023) extends this framework by integrating Doob's h-transform with a mean-reverting SDE, achieving better results compared to DDBMs. Despite the promising results in diffusion bridge with Doob's h-transform, two fundamental challenges persist: 1) the theoretical mechanisms by which Doob's h-transform governs the bridging process remain poorly understood, lacking a rigorous framework to unify its empirical success; and 2) while effective for global distribution alignment, existing methods frequently degrade high-frequency details-such as sharp edges and fine textures-resulting in outputs with blurred or oversmoothed artifacts that compromise perceptual fidelity. These limitations underscore the need for both theoretical grounding and enhanced detail preservation in diffusion bridges.
this section cite: ['b21', 'b24', 'b45', 'b51', 'b28', 'b50', 'b9', 'b57', 'b52', 'b20', 'b31', 'b10', 'b46', 'b41', 'b11', 'b43', 'b59', 'b55']

Section: UniDB
Diffusion Bridge via Doob's h-transform is a special case when 𝛾 → ∞ in UniDB!
this section cite: []

Section: Diffusion Bridge via Doob's h-transform

this section cite: []

Section: Ground Truth Doob's h-transform

this section cite: []

Section: UniDB Ground Truth Doob's h-transform UniDB
High-Quality (HQ) Low-Quality (LQ)
Figure 1. Recent advances in diffusion bridge models leverage Doob's h-transform to establish fixed endpoints between distributions, which introduces an h function into the forward process of a standard stochastic differential equation (SDE) to forcibly match the two endpoints. However, as shown in the figure, this method can lead to local blurring and distortion in the generated images. UniDB formulates the forward process as a stochastic optimal control problem and employs the penalty coefficient γ, balancing realistic SDE trajectories and target endpoint matching, to produce images with more realistic details. We also find that Doob's h-transform is a special case in our framework when γ → ∞. Therefore, our framework can seamlessly integrate with existing diffusion bridge models (with Doob's h-transform).
In this paper, we revisit the diffusion bridges through the lens of stochastic optimal control (SOC) by introducing a novel framework called UniDB, which formulates an optimization problem based on SOC principles to implement diffusion bridges. It enables the derivation of a closed-form solution for the optimal controller, along with the corresponding training objective for the diffusion bridge. UniDB identifies Doob's h-transform as a special case when the terminal penalty coefficient in the SOC cost function approaches infinity. This explains why Doob's h-transform may result in suboptimal solutions with blurred or distorted details. To address this limitation, UniDB utilizes the penalty coefficient in SOC to adjust the expressiveness of the image details and enhance the authenticity of the generated outputs. Our main contributions are as follows:
• We introduce UniDB, a novel unified diffusion bridge framework based on stochastic optimal control. This framework generalizes existing diffusion bridge models like DDBMs and GOUB, offering a comprehensive understanding and extension of Doob's h-transform by incorporating general forward SDE forms.
• We derive closed-form solutions for the SOC problem, demonstrating that Doob's h-transform is merely a special case within UniDB when the terminal penalty coefficient in the SOC cost function approaches infinity. This insight reveals inherent limitations in the existing diffusion bridge approaches, which UniDB overcomes. Notably, the improvement of UniDB requires minimal code modification, ensuring easy implementation.
• UniDB achieves state-of-the-art results in various image restoration tasks, including super-resolution (DIV2K), inpainting (CelebA-HQ), and deraining (Rain100H), which highlights the framework's superior image quality and adaptability across diverse scenarios.
this section cite: []

Section: Related Work Diffusion with Guidance.
This technique tackles conditional generative tasks by leveraging a differentiable loss function for guidance without the need for additional training (Chung et al., 2022;Shenoy et al., 2024;Bradley & Nakkiran, 2024). However, it often yields suboptimal image quality and a prolonged sampling process due to the necessity of small step sizes. Most importantly, the sampling process is prone to manifold deviations and detail losses (Yang et al., 2024). Furthermore, enhancing the guidance of the diffusion typically requires the introduction of additional modules, thereby increasing the model's computational complexity.
this section cite: ['b10', 'b40', 'b6', 'b53']

Section: Diffusion Schrödinger Bridge.
This approach aims to determine a stochastic process, π * that facilitates probabilistic transport between a given initial distribution P prior , and a terminal distribution P data (Liu et al., 2023;Shi et al., 2024;De Bortoli et al., 2021) while minimizing the Kullback-Leibler (KL) divergence. However, its training process is usually intricate, involving constraints that hinder direct optimization of the KL divergence, resulting in slow convergence and limited model fitting capability. For instance, DSB (Somnath et al., 2023) requires two independent forward passes during training to obtain the target distribution, thereby increasing both the complexity and time cost of training.
this section cite: ['b29', 'b41', 'b11', 'b43']

Section: Diffusion Bridge with Doob's h-transform.
Recent advances in diffusion bridging have demonstrated the efficacy of Doob's h-transform in enhancing transition quality between arbitrary distributions. Notably, DDBMs (Zhou et al., 2023) pioneered this approach by employing a linear SDE combined with Doob's h-transform to construct direct diffusion bridges. Subsequently, GOUB (Yue et al., 2023) extends this framework by integrating Doob's h-transform with a mean-reverting SDE, achieving state-of-the-art performance in image restoration tasks. Despite these empirical successes, the theoretical foundations of Doob's h-transform in this context remain insufficiently explored. In addition, these methods often result in images with blurred or oversmoothed features, particularly affecting the capture of highfrequency details crucial for perceptual fidelity.
this section cite: ['b60', 'b55']

Section: Diffusion with Stochastic Optimal Control.
The integration of SOC principles into diffusion models has emerged as a promising paradigm for guiding distribution transitions. DIS (Berner et al., 2022) established a foundational theoretical linkage between diffusion processes and SOC, while RB-Modulation (Rout et al., 2024) operationalized SOC via a simplified SDE structure for training-free style transfer using pre-trained diffusion models. Close to our work, DBFS (Park et al., 2024) leveraged SOC to construct diffusion bridges in infinite-dimensional function spaces and also established equivalence between SOC and Doob's h-transform. However, DBFS primarily extends Doob's h-transform to infinite Hilbert spaces via SOC, without addressing its intrinsic limitations. Our analysis reveals a critical insight: Doob's h-transform corresponds to a suboptimal solution that can inherently lead to artifacts such as blurred or distorted details. To resolve this, we introduce a unified SOC framework that jointly optimizes trajectory costs and terminal constraints, enhancing detail preservation and image quality.
this section cite: ['b4', 'b38', 'b34']

Section: Preliminaries

this section cite: []

Section: Denoising Diffusion Bridge Models
Starting with an initial d-dimensional data distribution x 0 ∼ q data (x), diffusion models (Song et al., 2020;Ho et al., 2020;Sohl-Dickstein et al., 2015;Song & Ermon, 2019) construct a diffusion process, which can be achieved by defining a forward stochastic process evolving from x 0 through a stochastic differential equation (SDE):
dx t = f (x t , t)dt + g t dw t ,(1)
where t ranges over the interval [0, T ], f : R d ×[0, T ] → R d is the vector-valued drift function, g : [0, T ] → R signifies the scalar-valued diffusion coefficient and w t ∈ R d is the Wiener process, also known as Brownian motion. To promise the transition probability p(x t | x s ) remains Gaussian, almost all the diffusion SDEs take the following linear form (Zheng et al., 2024) in (1):
f (x t , t) = f (t)x t ,(2)
where f (t) is some scalar-valued function. To realize transition between arbitrary distributions, DDBMs introduces Doob's h-transform (Särkkä & Solin, 2019), a mathematical technique applied to stochastic processes, which rectifies the drift term of the forward diffusion process to pass through a preset terminal point x T ∈ R d . Precisely, the forward process of diffusion bridges after Doob's h-transform becomes:
dx t = f (x t , t) + g 2 t h(x t , t, x T , T ) dt + g t dw t , (3
)
where h(x t , t, x T , T ) = ∇ xt log p(x T | x t ) is the h function. The diffusion bridge can connect the initial x 0 to any given terminal x T and thus is promising for various image restoration tasks. Meanwhile, its backward reverse SDE (Anderson, 1982) is given by
dx t = f (x t , t) + g 2 t ∇ xt log p(x T | x t ) -g 2 t ∇ xt log p(x t | x T ) dt + g t d wt .(4)
where wt is the reverse-time Wiener process and the unknown term ∇ xt log p(x t | x T ) can be estimated by a score prediction neural network s θ (Song et al., 2020).
this section cite: ['b45', 'b21', 'b42', 'b44', 'b59', 'b39', 'b3', 'b45']

Section: Generalized Ornstein-Uhlenbeck Bridge
Generalized Ornstein-Uhlenbeck (GOU) process describes a mean-reverting stochastic process commonly used in finance, physics, and other fields in the following SDE form (Ahmad, 1988;Pavliotis & Pavliotis, 2014;Wang et al., 2018):
dx t = θ t (µ -x t ) dt + g t dw t ,(5)
where µ is a given state vector, θ t denotes a scalar drift coefficient and g t represents the diffusion coefficient with θ t , g t satisfying the specified relationship g 2 t = 2λ 2 θ t where λ 2 is a given constant scalar. Based on this, Generalized Ornstein-Uhlenbeck Bridge (GOUB) is a diffusion bridge model (Yue et al., 2023), which can address image restoration tasks without the need for specific prior knowledge if we consider the initial state x 0 to represent a high-quality image and the corresponding low-quality image x T = µ as the final condition. With the introduction of µ, x t tends to µ as time t progresses. Through Doob's h-transform, denote θs:t = t s θ z dz, θt = t 0 θ z dz for simplification when s = 0 and σ2 s:t = λ 2 (1 -e -2 θs:t ), the forward process of GOUB is formed as:
dx t = θ t + g 2 t e -2 θt:T σ2 t:T (x T -x t ) dt + g t dw t . (6)
And the forward transition p(
x t | x 0 , x T ) is given by p(x t | x 0 , x T ) = N ( μ′ t , σ′2 t I), μ′ t = e -θt σ2 t:T σ2 T x 0 + (1 -e -θt σ2 t:T σ2 T )x T , σ′2 t = σ2 t σ2 t:T σ2 T .
(7) Also, GOUB presents a new reverse ODE called Mean-ODE, which directly neglects the Brownian term of (4):
dx t = f (x t , t) + g 2 t ∇ xt log p(x T | x t ) -g 2 t ∇ xt log p(x t | x T ) dt.(8)
this section cite: ['b1', 'b35', 'b48', 'b55']

Section: Stochastic Optimal Control
Stochastic Optimal Control (SOC) is a mathematical discipline that focuses on determining optimal control strategies for dynamic systems under uncertainty. By integrating stochastic processes with optimization theory, SOC seeks to identify the best control strategies in scenarios involving randomness, as commonly encountered in fields like finance (Geering et al., 2010) and style transfer (Rout et al., 2024).
Considering the dynamics described in (1), let us examine the following Linear Quadratic SOC problem (Bryson, 2018;O'Connell, 2003;Kappen, 2008;Chen et al., 2023):
min ut,γ ∈U E T 0 1 2 ∥u t,γ ∥ 2 2 dt + γ 2 ∥x u T -x T ∥ 2 2 s.t. dx t = (f (x t , t) + g t u t,γ ) dt + g t dw t , x u 0 = x 0 ,(9)
where x u t is the diffusion process under control, x 0 and x T represent for the initial state and the preset terminal respectively, ∥u t,γ ∥ 2 2 is the instantaneous cost, γ 2 ∥x u T -x T ∥ 2 2 is the terminal cost with its penalty coefficient γ. The SOC problem aims to design the controller u t,γ to drive the dynamic system from x 0 to x T with minimum cost.
this section cite: ['b17', 'b38', 'b7', 'b33', 'b22', 'b8']

Section: Methods

this section cite: []

Section: Diffusion Bridges Constructed by SOC Problem
The forward SDE of the Diffusion Bridge with Doob's htransform is enforced to pass from the predetermined origin x 0 to the terminal x T . With a similar purpose, UniDB constructs a SOC problem where the constraints are an arbitrary linear SDE of the forward diffusion with a given initial state, while the objective incorporates a penalty term steering the forward diffusion trajectory towards the predetermined terminal x T . Meanwhile, compared with the linear drift term (2), we combined a given state vector term m with the same dimension as x t and its related coefficient h t which is a simple reformulation and generalization of the parameters in GOU process (5):
f (x t , t) = f t x t + h t m.(10)
Accordingly, our SOC problem with unified linear SDE (10) is formed as:
min ut,γ ∈U E T 0 1 2 ∥u t,γ ∥ 2 2 dt + γ 2 ∥x u T -x T ∥ 2 2 s.t.dx t = f t x t + h t m + g t u t,γ dt + g t dw t , x u 0 = x 0 .(11)
According to the certainty equivalence principle (Chen et al., 2023;Rout et al., 2024), the addition of noise or perturbations to a linear system with quadratic costs does not change the optimal control. Therefore, we can modify the SOC problem with the deterministic ODE condition to obtain the optimal controller u * t,γ as follows,
min ut,γ ∈U T 0 1 2 ∥u t,γ ∥ 2 2 dt + γ 2 ∥x u T -x T ∥ 2 2 s.t. dx t = f t x t + h t m + g t u t,γ dt, x u 0 = x 0 .(12)
We can derive the closed-form solution to the problem (12), which leads to the following Theorem 4.1:
Theorem 4.1. Consider the SOC problem (12), denote d t,γ = γ -1 + e 2 fT ḡ2 t:T , fs:t = t s f z dz, hs:t = t s e -fz h z dz and ḡ2 s:t = t s e -2 fz g 2 z dz, denote ft , ht and ḡ2
t for simplification when s = 0, then the closed-form optimal controller u * t,γ is
u * t,γ = g t e ft:T x T -e ft:T x t -me fT ht:T d t,γ ,(13)
and the transition of x t from x 0 and x T is
x t = e ft d t,γ d 0,γ x 0 + e fT ḡ2 t d 0,γ x T + ht - e 2 fT hT ḡ2 t d 0,γ m .(14)
The proof of Theorem 4.1 is provided in Appendix A.1. With Theorem 4.1, we can obtain an optimally controlled forward SDE connected from x 0 to the neighborhood of the terminal x T and the transition of x t for the forward process. As for the backward process, similar to ( 4) and ( 8), the backward reverse SDE and Mean-ODE are respectively formulated as:
dx t = f t x t + h t m + g t u * t,γ -g 2 t ∇ xt log p(x t | x T ) dt + g t d wt ,(15)
dx t = f t x t + h t m + g t u * t,γ -g 2 t ∇ xt log p(x t | x T ) dt.(16)
this section cite: ['b8', 'b38']

Section: Connections between SOC and Doob's h-transform
We can intuitively see from the SOC problem that when γ → ∞ in Theorem 4.1, it means that the target of SDE process is precisely the predetermined endpoint (Chen et al., 2023), which is also the purpose of Doob's h-transform and facilitates the following theorem:
Theorem 4.2. For the SOC problem (12), when γ → ∞, the optimal controller becomes u * t,∞ = g t ∇ xt log p(x T | x t ), and the corresponding forward and backward SDE with the linear SDE form (10) are the same as Doob's h-transform as in (3) and (4).
The proof of Theorem 4.2 is presented in Appendix A.2. This theorem shows that existing diffusion bridge models using Doob's h-transform are merely special instances of our UniDB framework, which offers a unified approach to diffusion bridges through the lens of SOC. Furthermore, using Doob's h-transform in diffusion bridge models is not necessarily optimal, as letting the terminal penalty coefficient γ → ∞ eliminates the consideration of control costs in SOC. To support this argument, we present Proposition 4.3, which asserts that the diffusion bridge with Doob's h-transform is not the most effective choice.
Proposition 4.3. Consider the SOC problem (12), denote J (u t,γ , γ) ≜ T 0 1 2 ∥u t,γ ∥ 2 2 dt+ γ 2 ∥x u T -x T ∥
2 2 as the overall cost of the system, u * t,γ as the optimal controller (13), then
J (u * t,γ , γ) ≤ J (u * t,∞ , ∞).(17)
Detailed proof of Proposition 4.3 is provided in Appendix A.3. Proposition 4.3 shows that finite γ achieves a lower total cost not by sacrificing performance, but by optimally trading minor terminal mismatches for significantly smoother and more natural diffusion paths. Doob's h-transform requires larger controller ∥u * t,∞ ∥ 2 2 ≥ ∥u * t,γ ∥ 2 2 in SDE trajectory to force exact endpoint matching (the controlled target is precisely the preset endpoint ∥x u * T -x T ∥ 2 2 = 0 when γ → ∞), which may disrupt the inherent continuity and smoothness of images. Prioritizing pixel-perfect endpoints over smooth trajectories leads to "mathematically correct but visually unrealistic" outputs. As shown in Figure 1, Doob's h-transform can lead to artifacts along edges and unnatural patterns in smooth regions. Therefore, maintaining the penalty coefficient γ as a hyperparameter is a more effective approach.
this section cite: ['b8']

Section: Training objective of UniDB
In this section, we focus on constructing the training objective of UniDB. According to maximum log-likelihood (Ho et al., 2020) and conditional score matching (Song et al., 2020), the training objective is based on the forward transition p(x t | x 0 , x T ). Thus, we begin by deriving this probability. The closed-form expression in ( 14) represents the mean value of the forward transition after applying reparameterization techniques. However, this expression lacks a noise component after the transformation based on the certainty equivalence principle. To address this issue, we employ stochastic interpolant theory (Albergo et al., 2023) to introduce a noise term σ′ t ϵ with σ′ 0 = σ′ T = 0. We define σ′2 t = σ2 t σ2 t:T /σ 2 T similar to ( 7), leading to the following forward transition:
p(x t | x 0 , x T ) = N ( μt,γ , σ′2 t I), μt,γ = e ft d t,γ d 0,γ x 0 + e fT ḡ2 t d 0,γ x T + ht - e 2 fT hT ḡ2 t d 0,γ m ,σ2
s:t = e 2 ft ḡ2 s:t , σ′2 t = σ2 t σ2 t:T σ2 T .
(18) The derailed derivation is provided in Appendix A.4. Similar to (Yue et al., 2023) using the l 1 loss form to bring improved visual quality and details at the pixel level (Boyd, 2004;Hastie et al., 2017), we can derive the training objective. Denote a t,γ = e ft d t,γ , assuming µ t-1,θ , σ 2 t-1,θ and µ t-1,γ , σ 2 t-1,γ are respectively the mean values and variances of p θ (x t-1 | x t , x T ) and p(
x t-1 | x 0 , x t , x T ), suppose the score ∇ xt log p(x t | x T ) is parameterized as -ϵ θ (x t , x T , t)/σ ′
t , the final training objective is as follows,
L θ = E t,x0,xt,x T 1 2σ 2 t-1,θ µ t-1,θ -µ t-1,γ 1 , µ t-1,θ = x t -f t x t -h t m -g t u * t,γ + g 2 t σ′ t ϵ θ (x t , x T , t), µ t-1,γ = μt-1,γ + σ′2 t-1 a t,γ σ′2 t a t-1,γ (x t -μt,γ ), σ t-1,θ = g t .(19)
Please refer to Appendix A.5 for detailed derivations. Therefore, we can recover or generate the origin image x0 through Euler sampling iterations. So far, we've built the UniDB framework, which establishes and expands the forward and backward process of the diffusion bridge model through SOC and comprises Doob's h-transform as a special case.
this section cite: ['b21', 'b45', 'b2', 'b55', 'b5', 'b18']

Section: UniDB unifies diffusion bridge models
Our UniDB is a unified framework for existing diffusion bridge models: DDBMs (VE) (Zhou et al., 2023), DDBMs (VP) (Zhou et al., 2023) and GOUB (Yue et al., 2023). Proposition 4.4. UniDB encompasses existing diffusion bridge models by employing different hyper-parameter spaces H as follows:
• DDBMs (VE) corresponds to UniDB with hyperparameter H VE (f t = 0, h t = 0, γ → ∞)
• DDBMs (VP) corresponds to UniDB with hyperparameter
H VP (f t = -1 2 g 2 t , h t = 0, γ → ∞) • GOUB corresponds to UniDB with hyper-parameter H GOU (f t = θ t , h t = -θ t , m = µ, γ → ∞)
Details of the proposition 4.4 are provided in Appendix A.6.
this section cite: ['b60', 'b60', 'b55']

Section: An Example: UniDB-GOU
It is evident that these diffusion bridge models like DDBMs (VE), DDBMs (VP) and GOUB all based on Doob's htransform are all special cases of UniDB with γ → ∞. However, according to Proposition 4.3, these models are not the effective choices. Therefore, we introduce UniDB based on the GOU process (5), hereafter referred to as UniDB-GOU, which retains the penalty coefficient γ as the hyper-parameter. Considering the SOC problem with GOU process (5), the optimally controlled forward SDE is:
dx t = θ t + g 2 t e -2 θt:T γ -1 + σ2 t:T (x T -x t )dt+g t dw t , (20)
and the mean value of forward transition p(
x t | x 0 , x T ) is μt,γ = e -θt 1 + γ σ2 t:T 1 + γ σ2 T x 0 + 1 -e -θt 1 + γ σ2 t:T 1 + γ σ2 T x T . (21)
Please refer to Appendix A.7 for detailed proof.
Remark 1. It's worth noting that our UniDB model can be a plugin module to the existing diffusion bridge with Doob's h-transform. Taking UniDB-GOU as an example, we highlight the key difference between UniDB-GOU and GOUB (the coefficient of x 0 in the mean value of forward transition and h-function term) as follows:
e -θt σ2 t:T σ2 T ⇒ e -θt γ -1 + σ2 t:T γ -1 + σ2 T g t h = g t e -2 θt:T (x T -x t ) σ2 t:T GOUB ⇒ u * t,γ = g t e -2 θt:T (x T -x t ) γ -1 + σ2 t:T UniDB-GOU (22
)
Algorithm 1 UniDB Training repeat Take a pair of images x 0 = x 0 and
x T = x T t ∼ Uniform({1, ..., T }) σ t-1,θ = g t a t,γ = e -θt σ2 t:T σ2 T ← GOUB a t,γ = e -θt γ -1 +σ 2 t:T γ -1 +σ 2 T ← UniDB-GOU x t = a t,γ x 0 + (1 -a t,γ ) x T + σ′ t ϵ μt,γ = a t,γ x 0 + (1 -a t,γ ) x T µ t-1,θ = x t -θ t + g 2 t e -2 θt:T σ2 t:T (x T -x t ) + g 2 t σ′2 t ϵ θ (x t , x T , t) ← GOUB µ t-1,θ = x t -θ t + g 2 t e -2 θt:T γ -1 +σ 2 t:T (x T -x t ) + g 2 t σ′2 t ϵ θ (x t , x T , t) ← UniDB-GOU µ t-1,γ = μt-1,γ + σ′ t-1 at,γ σ′2 t at-1,γ (x t -μt,γ ) Take gradient descent step on ∇ θ L θ until converged
Hence, only a few lines of code need to be adjusted to generate more realistic images using the same training method. We provide pseudo-code Algorithm 1 and Algorithm 2 for the training and sampling process of UniDB-GOU, respectively. The two algorithms encapsulate the core methodologies employed by our model to learn and explain how to restore HQ images from LQ images. Also, the red and the green parts highlight the main difference between UniDB and GOUB. Beyond the GOUB model, our UniDB framework can be similarly extended to other diffusion bridge models, such as DDBMs (VE) and DDBMs (VP). For detailed information on UniDB-VE and UniDB-VP, please refer to Appendix A.8.
this section cite: []

Section: Algorithm 2 UniDB Sampling
Input: Low-Quality images x T = x T . for t = T to 1 do z ∼ N (0, I) if t > 1, else z = 0 Building upon equations ( 20) and ( 21), we further present a proposition to characterize how the penalty coefficient γ affects the controlled terminal distribution as follows: Proposition 4.5. Denote the initial state distribution x 0 , the terminal distribution x u T by the controller and the predefined terminal distribution x T , then
x t-1 = x t -θ t + g 2 t e -2 θt:T σ2 t:T (x T -x t ) + g 2 t σ′2 t ϵ θ (x t , x T , t) -g t z ← GOUB x t-1 = x t -θ t + g 2 t e -2 θt:T γ -1 +σ 2 t:T (x T -x t ) + g 2 t σ′2 t ϵ θ (x t , x T , t) -g t z ← UniDB-GOU end for Return High-Quality images x0
∥x u T -x T ∥ 2 2 = e -2 θT 1 + γλ 2 (1 -e -2 θT ) 2 ∥x T -x 0 ∥ 2 2 . (23
)
The detailed derivations of proposition 4.5 are provided in Appendix A.9. Notably, as γ approaches infinity, the control terminal converges to the predefined endpoint. However, as analyzed in Proposition 4.3, this can result in suboptimal outcomes with blurry or overly smoothed image details. To address this, it is crucial to balance the control cost and terminal term by selecting the value of γ. In the following section, we will present comprehensive experiments to evaluate the impact of different γ values on the results.
this section cite: []

Section: Experiments
In this section, we evaluate our models in image restoration tasks including Image 4×Super-resolution, Image Deraining, and Image Inpainting. We take four evaluation metrics: Peak Signal-to-Noise Ratio (PSNR, higher is better) (Fardo et al., 2016), Structural Similarity Index (SSIM, higher is better) (Wang et al., 2004), Learned Perceptual Image Patch Similarity (LPIPS, lower is better) (Zhang et al., 2018) and Fréchet Inception Distance (FID, lower is better) (Heusel et al., 2017). For simple expressions in the following sections, UniDB (SDE) and UniDB (ODE) are applied to represent the UniDB-GOU with reverse SDE and reverse Mean-ODE, respectively. Please refer to Appendix B and C for all related implementation details and more experiment results, respectively.
this section cite: ['b15', 'b49', 'b58', 'b19']

Section: Experiments Setup
According to Proposition 4.5, we first quantitatively analyze the l 2 -norm distances between the two terminal distributions depicted in Figure 4. We computed the average distances  between high-quality and low-quality images in the three datasets (CelebA-HQ, Rain100H, and DIV2K) related to the subsequent experimental section as the distances ∥x T -x 0 ∥ 2 2 in ( 23). As can be seen, for all three datasets, these distances remain relatively small, ranging from 10 -4 to 10 -10 when γ is within the range of 1 × 10 5 to 1 × 10 9 . Therefore, our subsequent experiments will focus on the γ of this range to further investigate the performance of UniDB-GOU.
this section cite: []

Section: Experimental Details
Image 4×Super-Resolution Tasks. In super-resolution, we evaluated our models based on DIV2K dataset (Agustsson & Timofte, 2017), which contains 2K-resolution high-quality images. During the experiment, all low-resolution images were 4× bicubic upscaling to the same image size as the paired high-resolution images. For comparison, we choose Bicubic interpolantion (Kawar et al., 2022), DDRM (Kawar et al., 2022), IR-SDE (Luo et al., 2023), GOUB (SDE) (Yue et al., 2023) and GOUB (Mean-ODE) (Yue et al., 2023) following abbreviated as GOUB (ODE) as the baselines. The qualitative and quantitative results are illustrated in Table 1 and Figure 2. Visually, our proposed model demonstrates a significant improvement over the baseline across various metrics. It also excels by delivering superior performance in both visual quality and detail compared to other results.
Image Deraining Tasks. For image deraining tasks, we conducted the experiments based on Rain100H datasets (Yang et al., 2017). Particularly, to be consistent with other deraining models (Ren et al., 2019;Zamir et al., 2021;Luo et al., 2023;Yue et al., 2023), PSNR and SSIM scores on the Y channel (YCbCr space) are selected instead of the origin PSNR and SSIM. MAXIM (Tu et al., 2022), MHNet (Gao et al., 2025), IR-SDE (Luo et al., 2023), GOUB (SDE) (Yue et al., 2023) and GOUB (ODE) (Yue et al., 2023) are chosen as the baselines. The relevant experimental results are shown in the Table 1 and Figure 3. Similarly, our model achieved state-of-the-art results in the deraining task. Visually, it can also be observed that our model excels in capturing details such as the eyebrows, eye bags, and lips.
Image Inpainting Tasks. In image inpainting tasks, we evaluated our methods on CelebA-HQ 256×256 datasets  Karras, 2017). For comparison, we choose DDRM (Kawar et al., 2022), PromptIR (Potlapalli et al., 2023), IR-SDE (Luo et al., 2023), GOUB (SDE) (Yue et al., 2023) and GOUB (ODE) (Yue et al., 2023) as the baselines. As for mask type, we take 100 thin masks consistent with the baselines. The relevant experimental results are shown in Table 1 and Figure 3. It is observed that our model achieved state-of-the-art results in all indicators and also delivered highly competitive outcomes on other metrics. From a visual perspective, our model excels in capturing details such as faces, eyes, chins, and noses.
(
this section cite: ['b0', 'b24', 'b24', 'b30', 'b55', 'b55', 'b54', 'b37', 'b56', 'b30', 'b55', 'b47', 'b16', 'b30', 'b55', 'b55', 'b23', 'b24', 'b36', 'b30', 'b55', 'b55']

Section: Ablation Study
Penalty Coefficient γ. To evaluate the specific impact of different penalty coefficients γ on model performance, we conducted the experiments above with several different γ.
The final results are shown in Table 2. The results across all tasks show that the choice of γ significantly influences the model's performance on all tasks, different optimal γ for different tasks, and our UniDB achieves the best performance in almost all metrics. Particularly in super-resolution tasks, we focus more on the significantly better perceptual scores (LPIPS and FID) (Luo et al., 2023), demonstrating that UniDB ensures to capture and preserve more intricate image details and features as shown in Figure 2. These findings underscore the importance of carefully tuning γ to achieve the best performance for specific tasks.
this section cite: ['b30']

Section: Conclusion
In this paper, we presented UniDB, a unified diffusion bridge framework based on stochastic optimal control principles, offering a novel perspective on diffusion bridges. Through this framework, we unify and extend existing diffu-sion bridge models with Doob's h-transform like DDBMs and GOUB. Moreover, we demonstrate that the diffusion bridge with Doob's h-transform can be viewed as a specific case within UniDB when the terminal penalty coefficient approaches infinity. This insight helps elucidate why Doob's h-transform may lead to suboptimal image restoration, often resulting in blurred or distorted details. By simply adjusting this terminal penalty coefficient, UniDB achieves a marked improvement in image quality with minimal code modifications. Our experimental results underscore UniDB's superiority and versatility across various image processing tasks, particularly in enhancing image details for more realistic outputs. Despite these advantages, UniDB, like other standard diffusion bridge models, faces the challenge of computationally intensive sampling processes, especially with high-resolution images or complex restoration tasks. Future work will focus on developing strategies to accelerate the sampling process, enhancing UniDB's practicality, particularly for real-time applications.
Solving the Equation ( 28), we have:
p t = p 0 e -ft , p T = p 0 e -fT .(31)
Solve the Equation ( 27):
dx t dt = f t x t + h t m -g 2 t p t ⇒ d(e -ft x t ) dt = e -ft h t m -e -ft g 2 t p t , ⇒ e -ft x t -x 0 = m ht -p 0 ḡ2 t , ⇒ e -ft x t -x 0 = m ht -p 0 ḡ2 t .
Hence, we can get:
x T = e fT x 0 + me fT hT -p T e 2 fT ḡ2 T ,(32)
and
x t = e ft x 0 + me ft ht -p T e ft e fT ḡ2 t .(33)
Take the Equation ( 32) into the Equation ( 30) and solve p T ,
p T = γ e fT x 0 + me fT hT -p T e 2 fT ḡ2 T -x T (34
) ⇒ p T = γ e fT x 0 + me fT hT -x T 1 + γe 2 fT ḡ2 T .(35)
Also, take the Equation ( 34) into the equation ( 33),
x t = e ft x 0 + me ft ht -e ft e fT ḡ2 t e fT x 0 + me fT hT -x T γ -1 + e 2 fT ḡ2 T = e ft d t,γ d 0,γ x 0 + e fT ḡ2 t d 0,γ x T + ht - e 2 fT hT ḡ2 t d 0,γ m .(36)
Preserve γ,
u * t,γ = -g t p t = -g t e -ft e fT e fT x 0 + me fT hT -x T γ -1 + e 2 fT ḡ2 T = g t e ft:T x T -e ft:T x t -me fT ht:T d t,γ ,(37)
with the fact (36)
x t = e ft d t,γ d 0,γ x 0 + e fT ḡ2 t d 0,γ x T + ht - e 2 fT hT ḡ2 t d 0,γ m ,(38)
which concludes the proof of the Proposition 4.1.
= e ft:T x T -e ft:T x t -me fT ht:T e 2 fT ḡ2 t:T ,(44)
⇒ u * t,∞ = g t e ft:T x T -e ft:T x t -me fT ht:T e 2 fT ḡ2 t:T = g t ∇ xt log p(x T |x t ) = g t h(x t , t, x T , T ). (45
)
The forward SDEs obtained through SOC and Doob's h-transform are both formed as
dx t =   f t x t + h t m + g 2 t e ft:T x T -e ft:T x t -me fT ht:T e 2 fT ḡ2 t:T   dt + g t dw t ,(46)
and the both backward SDEs are
dx t =   f t x t + h t m + g 2 t e ft:T x T -e ft:T x t -me fT ht:T e 2 fT ḡ2 t:T -g 2 t ∇ xt p(x t |x T )   dt + g t dw t ,(47)
which concludes the proof of the Theorem 4.2.
this section cite: []

Section: References
Ref_id:b0 Title: Ntire 2017 challenge on single image super-resolution: Dataset and study Year: (2017)
Ref_id:b1 Title: Introduction to stochastic differential equations Year: (1988)
Ref_id:b2 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b3 Title: Reverse-time diffusion equation models Year: (1982)
Ref_id:b4 Title: An optimal control perspective on diffusion-based generative modeling Year: (2022)
Ref_id:b5 Title: Convex optimization Year: (2004)
Ref_id:b6 Title: Classifier-free guidance is a predictor-corrector Year: (2024)
Ref_id:b7 Title: Applied optimal control: optimization, estimation and control Year: (2018)
Ref_id:b8 Title: Generative modeling with phase stochastic bridges Year: (2023)
Ref_id:b9 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b10 Title: Diffusion posterior sampling for general noisy inverse problems Year: (2022)
Ref_id:b11 Title: Diffusion schrödinger bridge with applications to score-based generative modeling Year: (2021)
Ref_id:b12 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b13 Title: Diffusion-based reinforcement learning via q-weighted variational policy optimization Year: (2024)
Ref_id:b14 Title: Continuous conditional diffusion models for image generation Year: (2024)
Ref_id:b15 Title: A formal evaluation of psnr as quality measurement parameter for image segmentation algorithms Year: (2016)
Ref_id:b16 Title: Mixed hierarchy network for image restoration Year: (2025)
Ref_id:b17 Title: Stochastic optimal control with applications in financial engineering. Optimization and Optimal Control: Theory and Applications Year: (2010)
Ref_id:b18 Title: The elements of statistical learning: data mining, inference, and prediction Year: (2017)
Ref_id:b19 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b20 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b21 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b22 Title: Stochastic optimal control theory Year: (2008)
Ref_id:b23 Title: Progressive growing of gans for improved quality, stability, and variation Year: (2017)
Ref_id:b24 Title: Denoising diffusion restoration models Year: (2022)
Ref_id:b25 Title: A method for stochastic optimization Year: (2014)
Ref_id:b26 Title: Optimal control theory: an introduction Year: (2004)
Ref_id:b27 Title: Optimal control theory: An introduction Year: (1972)
Ref_id:b28 Title: Diffusion models for image restoration and enhancement-a comprehensive survey Year: (2023)
Ref_id:b29 Title: A. I2sb: image-to-image schrödinger bridge Year: (2023)
Ref_id:b30 Title: Image restoration with mean-reverting stochastic differential equations Year: (2023)
Ref_id:b31 Title: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration Year: (2023)
Ref_id:b32 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b33 Title: Conditioned random walks and the rsk correspondence Year: (2003)
Ref_id:b34 Title: Stochastic optimal control for diffusion bridges in function spaces Year: (2024)
Ref_id:b35 Title: Introduction to stochastic differential equations Year: (2014)
Ref_id:b36 Title: Prompting for all-in-one blind image restoration Year: (2023)
Ref_id:b37 Title: Progressive image deraining networks: A better and simpler baseline Year: (2019)
Ref_id:b38 Title: Rb-modulation: Trainingfree personalization of diffusion models using stochastic optimal control Year: (2024)
Ref_id:b39 Title: Applied stochastic differential equations Year: (2019)
Ref_id:b40 Title: Gradient-free classifier guidance for diffusion model sampling Year: (2024)
Ref_id:b41 Title: Diffusion schrödinger bridge matching Year: (2024)
Ref_id:b42 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b43 Title: Aligned diffusion schrödinger bridges Year: (2023)
Ref_id:b44 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b45 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b46 Title: A unified diffusion framework for scene-aware human motion estimation from sparse signals Year: (2024)
Ref_id:b47 Title: Multi-axis mlp for image processing Year: (2022)
Ref_id:b48 Title: A stochastic differential equation sis epidemic model incorporating ornstein-uhlenbeck process Year: (2018)
Ref_id:b49 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b50 Title: Generalizable diffusion policy with transferable affordance Year: (2024)
Ref_id:b51 Title: Diffir: Efficient diffusion model for image restoration Year: (2023)
Ref_id:b52 Title: Policy representation via diffusion probability model for reinforcement learning Year: (2023)
Ref_id:b53 Title: Guidance with spherical gaussian constraint for conditional diffusion Year: (2024)
Ref_id:b54 Title: Deep joint rain detection and removal from a single image Year: (2017)
Ref_id:b55 Title: Image restoration through generalized ornstein-uhlenbeck bridge Year: (2023)
Ref_id:b56 Title: Multi-stage progressive image restoration Year: (2021)
Ref_id:b57 Title: 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations Year: (2024)
Ref_id:b58 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b59 Title: Diffusion bridge implicit models Year: (2024)
Ref_id:b60 Title: Denoising diffusion bridge models Year: (2023)
