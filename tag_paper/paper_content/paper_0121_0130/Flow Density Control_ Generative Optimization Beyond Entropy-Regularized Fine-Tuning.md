Title: Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning
Abstract: Adapting large-scale foundational flow and diffusion generative models to optimize task-specific objectives while preserving prior information is crucial for real-world applications such as molecular design, protein docking, and creative image generation. Existing principled fine-tuning methods aim to maximize the expected reward of generated samples, while retaining knowledge from the pre-trained model via KL-divergence regularization. In this work, we tackle the significantly more general problem of optimizing general utilities beyond average rewards, including risk-averse and novelty-seeking reward maximization, diversity measures for exploration, and experiment design objectives among others. Likewise, we consider more general ways to preserve prior information beyond KL-divergence, such as optimal transport distances and Rényi divergences. To this end, we introduce Flow Density Control (FDC), a simple algorithm that reduces this complex problem to a specific sequence of simpler fine-tuning tasks, each solvable via scalable established methods. We derive convergence guarantees for the proposed scheme under realistic assumptions by leveraging recent understanding of mirror flows. Finally, we validate our method on illustrative settings, text-to-image, and molecular design tasks, showing that it can steer pre-trained generative models to optimize objectives and solve practically relevant tasks beyond the reach of current fine-tuning schemes.

Section: Introduction

this section cite: []

Section: Utility

this section cite: []

Section: Current
This work Divergence Large-scale generative modeling has recently seen remarkable advancements, with flow [30,31] and diffusion models [51,52,23] standing out for their ability to produce high-fidelity samples across a wide range of applications, from chemistry [24] and biology [9] to robotics [8]. However, approximating the data distribution is insufficient for real-world applications such as scientific discovery [6,59], where one typically wishes to generate samples optimizing specific utilities, e.g., molecular stability and diversity, while preserving certain information from a pre-trained model. This problem has recently been tackled via fine-tuning in the case where the utility corresponds to the expected reward of generated samples, and pre-trained model information is retained via KLdivergence regularization, as shown in Fig. 1 (left). Crucially, this specific fine-tuning problem can be solved via entropy-regularized control formulations [e.g., 14,55,53] with successful applications in real-world domains such as image generation [14], molecular design [56], or protein engineering [56].
Unfortunately, many practically relevant tasks cannot be captured by this formulation. For instance, consider the tasks of risk-averse and novelty-seeking reward maximization. In the former case, one wishes to steer the generative model toward distributions with controlled worst-case rewards, thereby improving validity and safety. In the latter case, one aims to control the upper tail of the reward distribution to maximize the probability of generating exceptionally promising designs, e.g., for scientific discovery. Other applications that cannot be captured via maximization of simple expectations include manifold exploration [12], model de-biasing [13], and optimal experimental design [38,10] among others. Similarly, preserving prior information via a KL divergence has known drawbacks. For instance, it can lead to missing of low-probability yet valuable modes [29,43], and it prevents from leveraging the geometry of the space even when this is known, e.g., in protein docking [9]. Replacing KL with alternative divergences can address these shortcomings. Driven by these motivations, in this work we aim to answer the following fundamental question (see Fig. 1): How can we provably fine-tune a flow or diffusion model to optimize any user-specified utility while preserving prior information via an arbitrary divergence?
Answering this would contribute to the algorithmic-theoretical foundations of generative optimization.
Our approach We tackle this challenge by first introducing the formal problem of generative optimization via fine-tuning. Then, we shed light on why this formulation is strictly more expressive than current fine-tuning problems [14,53], and present a sample of novel practically relevant utilities and divergences (Sec. 3). Next, we introduce Flow Density Control (FDC), a simple sequential scheme that can fine-tune models to optimize general objectives beyond the reach of entropyregularized control methods. This is achieved by leveraging recent machinery from Convex [20] and General Utilities RL [60] (Sec. 4). We provide rigorous convergence guarantees for the proposed algorithm in both a simplified scenario, via convex optimization analysis [42,33], and in a realistic setting, by building on recent understanding of mirror flows [25] (Sec. 5). Finally, we provide an experimental evaluation of the proposed method, demonstrating its practical relevance on both synthetic and high-dimensional image and molecular generation tasks, showing how it can steer pre-trained models to solve tasks beyond the inherent limits of current fine-tuning schemes (Sec. 6).
this section cite: ['b29', 'b30', 'b50', 'b51', 'b22', 'b23', 'b8', 'b7', 'b5', 'b58', 'b13', 'b54', 'b52', 'b13', 'b55', 'b55', 'b11', 'b12', 'b37', 'b9', 'b28', 'b42', 'b8', 'b13', 'b52', 'b19', 'b59', 'b41', 'b32', 'b24']

Section: Our contributions
To sum up, in this work we contribute
• A formalization of the generative optimization problem, which extends current fine-tuning formulations beyond linear utilities and general divergences (Sec. 3).
• Flow Density Control (FDC), a principled algorithm capable of optimizing functionals beyond the reach of current fine-tuning schemes based on entropy-regularized control/RL (Sec. 4).
• Convergence guarantees for the presented algorithm both under simplified and realistic assumptions leveraging recent understanding of mirror flows (Sec. 5).
• An experimental evaluation of FDC showcasing its practical relevance on both illustrative and high-dimensional text-to-image and molecular design tasks, showing how it can steer pre-trained models to solve tasks beyond the capabilities of current fine-tuning schemes. (Sec. 6).
this section cite: []

Section: Background and Notation
General Notation. We denote with X ⊆ R d an arbitrary set. Then, we indicate the set of Borel probability measures on X with P(X ), and the set of functionals over the set of probability measures P(X ) as F(X ). Given an integer N , we define [N ] := {1, . . . , N }.
this section cite: []

Section: Generative Flow Models.
Generative models aim to approximately sample novel data points from a data distribution p data . Flow models tackle this problem by transforming samples X 0 = x 0 from a source distribution p 0 into samples X 1 = x 1 from the target distribution p data [31,17]. Formally, a flow is a time-dependent map ψ : [0, 1] × R d → R such that ψ : (t, x) → ψ t (x). A generative flow model is a continuous-time Markov process {X t } 0≤t≤1 obtained by applying a flow ψ t to X 0 ∼ p 0 as X t = ψ t (X 0 ), t ∈ [0, 1], such that X 1 = ψ 1 (X 0 ) ∼ p data . In particular, the flow ψ can be defined by a velocity field u : [0, 1] × R d → R d , which is a vector field related to ψ via the following ordinary differential equation (ODE), typically referred to as flow ODE: with initial condition ψ 0 (x) = 0. A flow model X t = ψ t (X 0 ) induces a probability path of marginal densities p = {p t } 0≤t≤1 such that at time t we have that X t ∼ p t . Given a velocity field u and marginal densities p, we say that u generates the marginal densities p = {p t } 0≤t≤1 if X t = ψ t (X 0 ) ∼ p t for all t ∈ [0, 1). This is the case if the pair (u, p) satisfy the Continuity Equation:
d dt ψ t (x) = u t (ψ t (x))(1)
d dt p t (x) + div(p t u t )(x) = 0 (2
)
In this case, we denote by p u the probability path of marginal densities induced by the velocity field u.
Flow matching [30,32,1,31] can estimate a velocity field u θ s.t. the induced marginal densities p u θ satisfy p u θ 0 = p 0 and p u θ 1 = p data , where p 0 denotes the source distribution, and p data the target data distribution. Interestingly, diffusion models [52] (DMs) admit an equivalent ODE-based formulation with identical marginal densities to their original SDE dynamics [31,Chapter 10]. Consequently, although in this work we adopt the notation of flow models, our contributions carry over directly to DMs.
this section cite: ['b30', 'b16', 'b29', 'b31', 'b0', 'b30', 'b51', 'b30']

Section: Continuous-time Reinforcement Learning.
We formulate finite-horizon continuous-time reinforcement learning (RL) as a specific class of optimal control problems [57,26,54,61]. Given a state space X and an action space A, we consider the transition dynamics governed by the following ODE:
d dt ψ t (x) = a t (ψ t (x)) (3
)
where a t ∈ A is a selected action. We consider a state space X := R d × [0, 1], and denote by (Markovian) deterministic policy a function π t (X t ) := π(X t , t) ∈ A mapping a state (x, t) ∈ X to an action a ∈ A such that a t = π(X t , t), and denote with p π t the marginal density at time t induced by policy π. Pre-trained Flow Models as an RL policy. A pre-trained flow model with velocity field u pre can be interpreted as an action process a pre t := u pre (X t , t), where a pre t is determined by a continuous-time RL policy via a pre t = π pre (X t , t) [12]. Therefore, we can express the flow ODE induced by a pre-trained flow model by replacing a t with a pre in Eq. ( 3), and denote the pre-trained model by its (implicit) policy π pre , which induces a marginal density p pre 1 := p π pre 1 approximating p data .
this section cite: ['b56', 'b25', 'b53', 'b60', 'b11']

Section: Formal Problem: a General Framework for Generative Optimization
In this section, we aim to formally introduce the general problem of generative optimization (GO) via fine-tuning. Formally, we wish to adapt a pre-trained generative flow model π pre to obtain a new model π * inducing an ODE:
d dt ψ t (x) = a * t (ψ t (x)) with a * t = π * (x, t),(4)
such that instead of imitating the data distribution p data , as typically in generative modeling, it induces a marginal density p π * 1 that maximizes a utility measure F : P(X ) → R, while preserving information from the pre-trained model π pre via regularization with an arbitrary divergence D(• ∥ p pre ). This algorithmic problem is illustrated in Fig. 2a, and formalized in the following.
this section cite: []

Section: Generative Optimization via Flow
Model Fine-Tuning arg max π F (p π 1 ) -αD(p π 1 ∥ p pre 1 ) s.t. d dt p t (x) + div(p t a t )(x) = 0 with a t = π(x, t) (5) APPLICATION FUNCTIONAL F / D LINEAR GO NON-LINEAR GO CONVEX GENERAL REWARD OPTIMIZATION [14, 55] E x∼p π [r(x)] ✓ ✓ ✓ MANIFOLD EXPLORATION [12] GEN. MODEL DE-BIASING H(p π
) := -E x∼p π [log p π (x)] ✗ ✓ ✓ RISK-AVERSE OPTIMIZATION CVaR r β (p π ) := E x∼p π [r(x) | r(x) ≤ q r β (p π )] ✗ ✓ ✓ E x∼p π [r(x)] -Var(p π ) ✗ ✗ ✓ NOVELTY-SEEKING OPTIMIZATION SQ r β (p π ) := E x∼p π [r(x) | r(x) ≥ q r β (p π )] ✗ ✗ ✓ OPTIMAL EXPERIMENT DESIGN s E x∼p π [Φ(x)Φ(x) ⊤ -λI] ✗ ✓ ✓ s(•) ∈ {log det(•), -Tr(•) -1 , -λ max (•)} DIVERSE MODES DISCOVERY -E z [D KL (p π,z ∥ E k p π,k )] ✗ ✗ ✓ LOG-BARRIER CONSTRAINED GENERATION E x∼p π [r(x)] -β log (⟨p π , c⟩ -C) ✗ ✓ ✓ KULLBACK-LEIBLER DIVERGENCE [14, 55] D KL (p π ∥ p pre ) = p π (x) log p π (x) p pre (x) dx ✓ ✓ ✓ RÉNYI DIVERGENCES D β (p π ∥ p pre ) := 1 β -1 log (p π (x)) β (p pre ) 1-β dx ✗ ✗ ✓ OPTIMAL TRANSPORT DISTANCES W p (p π ∥ p pre ) := inf γ∈Γ(p π ,p pre ) E (x,y)∼γ [d(x, y) p ] 1 p ✗ ✗ ✓ MAXIMUM MEAN DISCREPANCY MMD k (p π ∥ p pre ) := ∥µ p π -µ p pre ∥, µ p := E x∼p [k(x, •)] ✗ ✓ ✓
Table 1: Examples of practically relevant utilities F (blue) and divergences D (orange). Apx. A provides mathematical details and practical applications for each functional. Notice that besides H, all non-linear functionals presented are novel in the context of fine-tuning of diffusion and flow models.
In this formulation, F and D are both functionals mapping the marginal density p π 1 induced by policy π to a scalar real number, namely F, D : P(X ) → R. The constraint in Eq. ( 5) is the (controlled) Continuity Equation (see Eq. ( 2)), which relates the control policy π to the induced marginal density p π 1 .
this section cite: []

Section: The sub-case of KL-regularized reward maximization via entropy-regularized control
Current fine-tuning schemes for flow generative models based on RL and control-theoretic formulations [e.g., 14, 55] aim to tackle the following problem, where we omit the flow constraint for clarity:
Linear Generative Optimization via Flow Model Fine-Tuning arg max π E x∼p π 1 [r(x)] -αD KL (p π 1 ∥ p pre 1 )(6)
Crucially, the common problem in Eq. ( 6), which we denote by Linearfoot_0 GO, is the specific sub-case of the generative optimization problem in Eq. ( 5), where the utility F is a linear functional corresponding to the expectation of a (reward) function r : X → R, and D is the Kullback-Leibler divergence:
F(p π 1 ) = ⟨p π 1 , r⟩ = E x∼p π 1 [r(x)] and D(p π 1 ∥ p pre 1 ) = D KL (p π 1 ∥ p pre 1 )(7)
This specific fine-tuning problem can be solved via entropy-regularized (or relaxed) control [14].
this section cite: ['b13']

Section: Beyond Linear Generative Optimization: an Expressivity Viewpoint
Let G(p π 1 ) = F(p π 1 ) -α D(p π 1 ∥ p pre 1 ) be the functional in Eq. ( 5). Then we denote by Convex GO the case where G is concave in p π 1 , and by General GO the case for arbitrary, possibly non-convex functionals 2 . In terms of expressivity Linear GO ⊂ Convex GO ⊂ General GO, as depicted in Fig. 2b (left). In Table 1 we classify into these tree tiers a sample of practically relevant utilities (F, blue) and divergences (D, orange). In Apx. A we report complete definitions and applications. Except for entropy [12] and KL, all non-linear functionals in Table 1 are to our knowledge explicitly used for the first time in the flow and diffusion model fine-tuning literature, while vastly employed in other areas. Moreover, the framework presented in this work for GO (Eq. 5) applies to any new choice of F or D.
this section cite: ['b11']

Section: Algorithm 1 Flow Density Control (FDC)
1: input: G : general utility functional, K : number of iterations, π pre : pre-trained flow generative model, {η k } K k=1 regularization coefficients 2: Init: π0 := π pre 3: for k = 1, 2, . . . , K do 4:
Estimate: ∇xg k = ∇xδG(p k-1 1 ) 5:
Compute π k via first-order linear fine-tuning: π k ← ENTROPYREGULARIZEDCONTROLSOLVER(∇xg k , η k , π k-1 ) 6: end for 7: output: policy π := πK Given the generality of generative optimization (Eq.( 5)), a natural question arises: how can it be solved algorithmically? In the next section, we answer this by leveraging recent machinery from Convex [20] and General-Utilities RL [60], to derive a fine-tuning scheme that handles both convex and general GO, thus going beyond current entropy-regularized control methods, as illustrated in Fig. 2b (right).
this section cite: ['b19', 'b59']

Section: Algorithm: Flow Density Control
In this section, we introduce Flow Density Control (FDC), see Alg. 1, which provably solves the generative optimization problem in Eq. ( 5) via sequential fine-tuning of the pre-trained model π pre . To this end, we recall the notion of first variation of a functional over a space of probability measures [25]. A functional G ∈ F(X ), where G : P(X ) → R, has first variation at µ ∈ P(X ) if there exists a function δG(µ) ∈ F(X ) such that for all µ ′ ∈ P(X ) it holds that:
G(µ + ϵµ ′ ) = G(µ) + ϵ⟨µ ′ , δG(µ)⟩ + o(ϵ).
where the inner product has to be interpreted as an expectation. Intuitively, the first variation of G at µ, namely δG(µ), can be interpreted as an infinite-dimensional gradient in the space of probability measures. Given this notion, and a pair of generative models represented via policies π and π ′ , we can now state the following entropy-regularized first variation maximization fine-tuning problem.
this section cite: ['b24']

Section: Entropy-Regularized First Variation Maximization
arg max
π ⟨δG p π ′ 1 , p π 1 ⟩ -ηD KL (p π 1 ∥ p π ′ 1 )(8)
Crucially, we can introduce a function g : X → R defined for all x ∈ X such that:
g(x) := δG p π ′ 1 (x) and E x∼p π [g(x)] = ⟨δG p π ′ 1 , p π 1 ⟩ (9
)
As a consequence, by rewriting Eq. ( 8) expressing the first term via an expectation as shown in Eq. ( 9), it corresponds to a common Linear GO problem (see Eq. ( 6)), which can be optimized by utilizing established entropy-regularized control methods [e.g., 56,14,61].
We can finally present Flow Density Control (FDC), see Alg. 1, a mirror descent (MD) scheme [42] that reduces optimization of non-linear functionals G to a specific sequence of Linear GO problems. FDC takes three inputs: a pre-trained flow or diffusion model π pre , the number of iterations K, and a sequence of regularization weights {η k } K k=1 . At each iteration, FDC first estimates the gradient of the functional first variation at the previous policy π k-1 , i.e., ∇ x δG p k-1 1 (line 4). Then, it updates the flow model π k by solving the fine-tuning problem in Eq. ( 8) via an entropy-regularized control solver such as Adjoint Matching [14], using ∇ x g k := ∇ x δG p k-1 1 as in Eq. ( 9) (line 5). Ultimately, it returns a final policy π := π K . We report a detailed implementation of FDC in Apx. D.
Gradient of first variation: computation and estimation. Surprisingly, estimating ∇ x g k in Alg. 1 (line 4) rarely requires density estimation. Among the functionals in Table 1, only the Rényi divergence does, for which one can leverage the recent Itô density estimator [50]. All other functionals admit straightforward plug-in or sample-based approximations detailed in Apx. A. As an illustrative example, in the following we showcase three examples from Table 1:
∇ x δQ(p π )(x) =    -∇ x log p π (x) Entropy (H) ∇ x r(x) • 1{r(x) ≤ q r β (p π )} CVaR ∇ x ϕ * (x) where ϕ * = arg max ϕ:∥∇xϕ∥≤1 ⟨ϕ, p π -p pre ⟩ Wasserstein-1 (W 1 )
Here Q denotes either a utility F or a divergence D, and q r β (p π ) is the β-quantile of Z = r(X) with X ∼ p π [47]. These gradients can be easily implemented. For entropy, the score term can be approximated via the score network in the case of diffusion models [12], and obtained via a known linear transformation of the learned velocity field in the case of flows [14,Eq.(8)]. For CVaR, any standard sample-based estimator of q r β (p π ) [47] can be used. For Wasserstein-1, ϕ * actually corresponds to the discriminator in Wasserstein-GAN, which can be learned with established methods [2]. In Apx. A, we report the gradient of the first variation for all functionals in Table 1, explain their practical estimation, and present a tutorial to derive the first variation of any new functionals not mentioned within Table 1.
Given the approximate gradient estimates and the generality of the objective functions, it is still unclear whether the proposed algorithm provably converges to the optimal flow model π * . In the next section, we answer this question by developing a theoretical analysis via recent results on mirror flows [25].
this section cite: ['b55', 'b13', 'b60', 'b41', 'b13', 'b49', 'b46', 'b11', 'b13', 'b46', 'b1', 'b24']

Section: Guarantees for Generative Optimization via Flow Density Control
In this section, we recast (5) as constrained optimization over stochastic processes, where the constraint is given by the Continuity Equation (2). This formulation enables the application of mirror descent for constrained optimization and the notion of relative smoothness [3]. In our framework, convergence speed is governed by: 1. the structural complexity of the functional G (cf. Section 4), 2. the accuracy of the estimator g from ( 9), and 3. the quality of the oracle ENTROPYREGULARIZED-CONTROLSOLVER in Alg. 1. To handle these cases, we will analyze two representative regimes:
• Idealized. G is concave, and both g and ENTROPYREGULARIZEDCONTROLSOLVER are exact. In this setting, classical results yield sharp step-size prescriptions and fast convergence rates.
• General. G is non-concave, with g and the oracle subject to noise and bias. While fast convergence is generally out of reach [34,27], convergence to a stationary point remains attainable under mild assumptions. Theoretical analysis: Idealized setting. We now present a framework leading to convergence guarantees for FDC (i.e., Alg. 1) for concave functionals G ∈ F(X ). We start by recalling the notion of Bregman divergence induced by a functional Q ∈ F(X ) between densities µ, ν ∈ P(X ), namely:
D Q (µ ∥ ν) := Q(µ) -Q(ν) -⟨δQ(ν), µ -ν⟩
Next, we introduce two structural properties for our analysis.
Definition 1 (Relative smoothness and relative strong concavity [33]). Let G : P(X ) → R a concave functional. We say that G is L-smooth relative to Q ∈ F(X
) over P(X ) if ∃ L scalar s.t. for all µ, ν ∈ P(X ): G(ν) ≥ G(µ) + ⟨δG(µ), ν -µ⟩ -LD Q (ν ∥ µ)(10)
and we say that G is l-strongly concave relative to Q ∈ F(X
) over P(X ) if ∃ l ≥ 0 scalar s.t. for all µ, ν ∈ P(X ): G(ν) ≤ G(µ) + ⟨δG(µ), ν -µ⟩ -lD Q (ν ∥ µ)(11)
In the following, we interpret line (6) of FDC as a step of mirror ascent [42], and the KL divergence term as the Bregman divergence induced by an entropic mirror map Q = H, i.e., D KL (µ, ν) = D H (µ ∥ ν). We can finally state the following set of assumptions as well as the convergence guarantee for an arbitrary functional G(•) = F(•) -αD(• ∥ p pre ) ∈ F(X ). Assumption 5.1 (Exact estimation and optimization). We consider the following assumptions: 8) is solved exactly. Theorem 5.1 (Convergence guarantee of Flow Density Control with concave functionals). Given Assumptions 5.1, fine-tuning a pre-trained model π pre via FDC (Algorithm 1) with η k = L ∀k ∈ [K], leads to a policy π inducing a marginal distribution p π
1. Exact estimation: ∇ x δG(p k 1 ) is estimated exactly ∀k ∈ [K]. 2. The optimization problem in Eq. (
1 such that:
G(p * 1 ) -G(p π 1 ) ≤ L -l K D KL (p * 1 ∥ p pre 1 )(12)
where p * 1 := p π * 1 is the marginal distribution induced by the optimal policy π * ∈ arg max π G(p π 1 ) := F(p π 1 ) -αD(p π 1 ∥ p pre 1 ). Theorem 5.1 provides a fast convergence rate under a specific step-size choice (η k = L). However, it critically depends on Assumption 5.1, which typically does not hold in practice. To address this limitation, we now consider a more general scenario where this key assumption is relaxed.
this section cite: ['b2', 'b33', 'b26', 'b32', 'b41']

Section: Theoretical analysis: General setting.
Recall that p k 1 := p π k 1 represents the (stochastic) density produced by the ENTROPYREGULARIZEDCONTROLSOLVER oracle at the k-th step of FDC, and consider the following mirror ascent iterates, where 1/λ k = η k in Algorithm 1:
p k ♯ := arg max p∈P(Ωpre) ⟨δG p π k-1 T , p⟩ - 1 γ k D KL (p ∥ p π k-1 T ) (MD k )
In realistic settings, where only noisy and biased approximations of (MD k ) are available, it is essential to quantify the deviations from the idealized iterates in (MD k ). To this end, denote by T k the filtration up to step k, and consider the decomposition of the oracle into its noise and bias parts:
b k := E δG(p π k T ) -δG(p k ♯ ) | T k , U k := δG(p π k T ) -δG(p k ♯ ) -b k (13
)
Conditioned on T k , U k has zero mean, while b k captures the systematic error. We then impose: Assumption 5.2 (Noise and Bias). The following events happen almost surely:
∥b k ∥ ∞ → 0, k E γ 2 k ∥b k ∥ 2 ∞ + ∥U k ∥ 2 ∞ < ∞, k γ k ∥b k ∥ ∞ < ∞ (14
)
The first condition is a necessary requirement for convergence since when violated, it is easy to construct scenarios where no practical algorithm can solve the generative optimization problem.
The second and third inequalities manage the trade-off between accuracy of the approximate oracle ENTROPYREGULARIZEDCONTROLSOLVER and aggressiveness of the step sizes, γ k . Intuitively, lower noise and bias in the oracle enable the use of larger step sizes. To this end, Assumption 5.2 provides a concrete criterion that guarantees the success of finding the optimal policy with probability one.
this section cite: []

Section: Theorem 5.2 (Convergence guarantee of Flow Density Control for general functionals).
Given the Robbins-Monro step-size rule: k γ k = ∞, k γ 2 k < ∞, under Assumption 5.2 and technical assumptions (see Appendix C), the sequence of marginal densities p k 1 induced by the iterates π k of Algorithm 1 converges weakly to a stationary point p1 of G almost surely, formally: p k 1 ⇀ p1 a.s..
this section cite: []

Section: Experimental Evaluation
We analyze the ability of Flow Density Control (FDC) to induce policies optimizing complex non-linear objectives, and compare its performance with Adjoint Matching (AM) [14], a classic fine-tuning method. We present two types of experiments: (i) Illustrative settings to provide insights via visual interpretability, and (ii) High-dimensional real-world applications, namely (a) noveltyseeking molecular design for single-point energy minimization [18], and (b) manifold exploration for text-to-image creative bridge design generation. Additional details are provided in Apx. E. Risk-averse reward maximization for better worst-case validity or safety. We fine-tune a pretrained policy π pre (see Fig. 3a) by optimizing the CVaR β utility i.e., expected outcome in the β-worstcase (see Tab. 1) with KL regularization, and costs interpreted as negative rewards. The cost has three regions: a high-cost plateau (dark orange), where the initial density lies; a moderate-cost left area (light orange); and a predominantly low-cost right zone (yellow) punctuated by narrow, but catastrophic red-stripes. As shown in Fig. 3b, AM moves the model density into the yellow region, lowering average cost but exposing it to rare extreme costs. In contrast, FDC, run with K = 2 iterations and β = 0.01, successfully steers density into the safer, moderate-cost area, cutting the 1%-worst-case cost from 288.2 achieved by AM to 90.0, well below the initial 262.5, as shown in Fig. 3c and 3d.
Novelty-seeking reward maximization for discovery. We fine-tune a pre-trained policy π pre to maximize the SQ β utility, i.e., expected outcome in the β-best-case (see Tab. 1). The reward shown in Fig. 3e has a moderately high-reward left area (light gray), a medium-reward central plateau (darker gray) where the initial density lies, and a low-reward right region (black) with sparse, extreme-reward spikes depicted by thin white lines. As shown in Fig. 3f, AM drifts the density into the safer left basin -improving the average reward but only reaching a best-1% expected reward of 55.5, as shown in Fig. 3g and Fig. 3h. In contrast, FDC, run for K = 2 iterations and β = 0.99, pushes the density rightwards, elevating the top-1% reward to 596.1 (see Fig. 3h) -far above both AM and the initial 66.6. Reward maximization regularized via optimal transport distance. We fine-tune the pre-trained model with density in Fig. 3i to maximize a reward function that increases moving top right. We consider two W 1 distances induced by two ground metrics: d A , which makes vertical moves more costly than horizontal ones, and d B , which does the opposite. Under d A , both AM and the OT-regularized model reach an expected reward of 35.0, but FDC-A incurs only W A 1 = 1.95 versus 4.67 for AM, Expected rewards maximization under optimal transport distance regularization. Crucially, FDC can optimize well these complex objectives, while AM [14], a classic fine-tuning scheme, fails at this. and achieves a mean shift that is 280% larger in the horizontal than in the vertical direction (Fig. 3j and Tab. 3l). By contrast, FDC-B under d B preferentially shifts the density upward (Fig. 3k).
this section cite: ['b13', 'b17', 'b13']

Section: Conservative manifold exploration.
We tackle manifold exploration [12] by fine-tuning a pre-trained model π pre to maximize the entropy utility (H in Tab. 1) under a KL regularization of strength α, a capability not possible with prior methods [12]. As in previous work, we consider the common setting where the pre-trained model density p pre 1 concentrates most of its mass in a specific region as shown in Fig. 4a, where N = 10000 samples are shown. By fine-tuning π pre via FDC, the density of the fine-tuned model shifts into low-coverage areas (see Fig. 4b and 4c). In particular, Fig. 4d demonstrates that reducing α from 0.5 to 0.0 yields progressively higher Monte Carlo entropy estimates (7.00 at α = 0.5, 7.14 at α = 0), thus enabling control of the trade-off between preserving the original distribution and exploring novel regions, a capability not supported by prior methods [12]. Molecular design for single-point energy minimization. We fine-tune FlowMol [15], pre-trained on QM9 [46], to discover molecules minimizing the single-point total energy computed via extended tight-binding at the GFN1-xTB level of theory [18]. Concretely, we maximize the negative energy. We do not aim to maximize the average sample reward, but rather that of the top 0.2% samples. We employ FDC with novelty-seeking SQ utility (see Tab. 1) with β = 0.998, and make 2 gradient steps per K = 10 iterations. We compare it with AM run for 240 steps. Fig. 4j shows that while AM generates better samples in average (namely 29.1 over 27.5 of FDC), the average quality of the top 0.2% molecules, indicated by SQ β is higher for FDC than for AM (namely 41.8 over 39.7 of AM). This confirms (see Fig. 4i and 4h) that FDC can sacrifice the average reward to generate a few truly high-reward designs. Text-to-image bridge designs conservative exploration. We perform manifold exploration by fine-tuning Stable Diffusion (SD) 1.4 [49] with prompt "A creative bridge design.". To this end, we maximize the KL-regularized entropy (see Tab. 1) with α = 0.001 via FDC for K = 2 steps. As a diversity metric, we utilize the Vendi score [19] with cosine similarity kernel on the extracted CLIP [21] features from a sample of 100 images and compared it to the baseline pre-trained model in Fig. 4g. Beyond increasing the Vendi score, FDC also increases the CLIP score of the initial model.
this section cite: ['b11', 'b11', 'b11', 'b14', 'b45', 'b17', 'b48', 'b18', 'b20']

Section: Related Works
Flow and diffusion models fine-tuning via optimal control. Recent works have framed fine-tuning of diffusion and flow models to maximize expected reward under KL regularization as an entropy-regularized optimal control problem [e.g., 55,53,56,14]. Crucially, as shown in Sec. 3, the problem tackled by these studies is the specific sub-case of generative optimization (Eq. ( 5)),  [15]. FDC shows enhanced control capabilities for optimizing such complex objectives than AM, a classic fine-tuning scheme.
where the utility F is linear, and D = D KL . In this work, we propose a principled method with guarantees for the far more general class of non-linear utilities and divergences beyond KL, including the ones listed in Tab. 1. The framework introduced has strictly higher expressive power and control capabilities for fine-tuning generative model (see Sec. 3). This renders possible to tackle relevant tasks e.g., scientific discovery, beyond the capabilities of the aforementioned fine-tuning schemes. Convex and General Utilities Reinforcement Learning. Convex and General (Utilities) RL [20,58,60] generalizes RL to the case where one wishes to maximize a concave [20,58], or general [60,4] functional of the state distribution induced by a policy over a dynamical system's state space. The introduced generative optimization problem (in Eq. ( 5)) is related, with p π 1 representing the state distribution induced by policy π over a subset of the state space. Recent works tackled the finite samples budget setting [e.g., 41, 39, 40, 44, 11]. Ultimately, to our knowledge, this is the first work leveraging an algorithmic scheme resembling General RL for the practically relevant task of generative optimization of general non-linear functionals via fine-tuning of diffusion and flow models. Optimization over probability measures via mirror flows. Recently, there has been a growing interest in building theoretical guarantees for optimization problems over spaces of probability measures in a variety of applications. These include GANs [25], optimal transport [3,28,27], kernelized methods [16], and manifold exploration [12]. We present the first use of this framework to establish guarantees for the generative optimization problem in Eq. ( 5). This novel link to probability-space optimization sheds new light on large-scale flow and diffusion models fine-tuning.
this section cite: ['b54', 'b52', 'b55', 'b13', 'b14', 'b19', 'b57', 'b59', 'b19', 'b57', 'b59', 'b3', 'b24', 'b2', 'b27', 'b26', 'b15', 'b11']

Section: Conclusion
This work tackles the fundamental challenge of fine-tuning pre-trained flow and diffusion generative models on arbitrary task-specific utilities and divergences while retaining prior knowledge. We introduce a unified generative optimization framework that strictly generalizes existing formulations and propose a rich class of new practically relevant objectives. We then propose Flow Density Control, a mirror-descent algorithm that reduces complex generative optimization to a sequence of standard fine-tuning steps, each solvable by scalable off-the-shelf methods. Leveraging convex analysis and recent advances in mirror flows theory, we prove convergence under general conditions. Empirical results on synthetic benchmarks, molecular design, and image generation, demonstrate that our approach can steer pre-trained models to optimize objectives beyond the reach of current fine-tuning techniques. As for limitations, while our framework is general, future work will need to assess to what extent the flexibility in selecting utilities and divergences yields concrete gains in specific applications.
this section cite: []

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2022)
Ref_id:b1 Title:  Year: (2017)
Ref_id:b2 Title: Mirror descent with relative smoothness in measure spaces, with application to sinkhorn and em Year: (2022)
Ref_id:b3 Title: Reinforcement learning with general utilities: Simpler variance reduction and large state-action space Year: (2023)
Ref_id:b4 Title: Dynamics of stochastic approximation algorithms Year: (2006)
Ref_id:b5 Title: Generative models for molecular discovery: Recent advances and challenges Year: (2022)
Ref_id:b6 Title: Bayesian experimental design: A review Year: (1995)
Ref_id:b7 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b8 Title: Diffusion steps, twists, and turns for molecular docking Year: (2022)
Ref_id:b9 Title: Geometric active exploration in markov decision processes: the benefit of abstraction Year: (2024)
Ref_id:b10 Title: Global reinforcement learning: Beyond linear and convex rewards via submodular semi-gradient methods Year: (2024)
Ref_id:b11 Title: Provable maximum entropy manifold exploration via diffusion models Year: ()
Ref_id:b12 Title: Debiasing synthetic data generated by deep generative models Year: (2024)
Ref_id:b13 Title: Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control Year: (2024)
Ref_id:b14 Title: Mixed continuous and categorical flow matching for 3d de novo molecule generation Year: (2024)
Ref_id:b15 Title: Analysis of kernel mirror prox for measure optimization Year: (2024)
Ref_id:b16 Title: Temporal difference flows Year: (2025)
Ref_id:b17 Title: dxtb-an efficient and fully differentiable framework for extended tight-binding Year: ()
Ref_id:b18 Title: The vendi score: A diversity evaluation metric for machine learning Year: (2022)
Ref_id:b19 Title: Provably efficient maximum entropy exploration Year: (2019)
Ref_id:b20 Title: Clipscore: A reference-free evaluation metric for image captioning Year: (2021)
Ref_id:b21 Title: Fundamentals of convex analysis Year: (2004)
Ref_id:b22 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b23 Title: Equivariant diffusion for molecule generation in 3d Year: (2022)
Ref_id:b24 Title: Finding mixed nash equilibria of generative adversarial networks Year: (2019)
Ref_id:b25 Title: Policy evaluation and temporal-difference learning in continuous time and space: A martingale approach Year: (2022)
Ref_id:b26 Title: Sinkhorn flow as mirror flow: A continuous-time framework for generalizing the sinkhorn algorithm Year: (2024)
Ref_id:b27 Title: A gradient descent perspective on sinkhorn Year: (2021)
Ref_id:b28 Title: Rényi divergence variational inference Year: (2016)
Ref_id:b29 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b30 Title: Flow matching guide and code Year: (2024)
Ref_id:b31 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b32 Title: Relatively smooth convex optimization by first-order methods, and applications Year: (2018)
Ref_id:b33 Title: A unified stochastic approximation framework for learning in games Year: (2024)
Ref_id:b34 Title: Envelope theorems for arbitrary choice sets Year: (2002)
Ref_id:b35 Title: Kernel mean embedding of distributions: A review and beyond Year: (2017)
Ref_id:b36 Title: Modern Adaptive Experiment Design: Machine Learning Perspective Year: (2024)
Ref_id:b37 Title: Active exploration via experiment design in Markov chains Year: (2023)
Ref_id:b38 Title: Challenging common assumptions in convex reinforcement learning Year: (2022)
Ref_id:b39 Title: Convex reinforcement learning in finite trials Year: (2023)
Ref_id:b40 Title: The importance of non-markovianity in maximum state entropy exploration Year: (2022)
Ref_id:b41 Title: Arkadij Semenovič Nemirovskij and David Borisovich Yudin. Problem complexity and method efficiency in optimization Year: (1983)
Ref_id:b42 Title: Heavy-tailed diffusion models Year: (2024)
Ref_id:b43 Title: Submodular reinforcement learning Year: (2023)
Ref_id:b44 Title: Optimal design of experiments Year: (2006)
Ref_id:b45 Title: Quantum chemistry structures and properties of 134 kilo molecules Year: (2014)
Ref_id:b46 Title: Conditional value-at-risk for general loss distributions Year: (2002)
Ref_id:b47 Title: Optimization of conditional value-at-risk Year: (2000)
Ref_id:b48 Title: High-resolution image synthesis with latent diffusion models Year: (2021)
Ref_id:b49 Title: The superposition of diffusion models using the it\ˆo density estimator Year: (2024)
Ref_id:b50 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b51 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b52 Title: Fine-tuning of diffusion models via stochastic control: entropy regularization and beyond Year: (2024)
Ref_id:b53 Title: Efficient exploration in continuous-time model-based reinforcement learning Year: (2023)
Ref_id:b54 Title: Finetuning of continuous-time diffusion models as entropy-regularized control Year: (2024)
Ref_id:b55 Title: Feedback efficient online fine-tuning of diffusion models Year: (2024)
Ref_id:b56 Title: Reinforcement learning in continuous time and space: A stochastic control approach Year: (2020)
Ref_id:b57 Title: Reward is enough for convex mdps Year: (2021)
Ref_id:b58 Title: a generative model for inorganic materials design Year: (2023)
Ref_id:b59 Title: Variational policy gradient method for reinforcement learning with general utilities Year: (2020)
Ref_id:b60 Title: Scores as actions: a framework of fine-tuning diffusion models by continuous-time reinforcement learning Year: (2024)
