Title: Convergence of Mean-Field Langevin Stochastic Gradient Descent-Ascent for Distributional Minimax Optimization
Abstract: We study convergence properties of the discretetime Mean-Field Langevin Stochastic Descent-Ascent (MFL-SDA) algorithm for solving distributional minimax optimization. These problems arise in various applications, such as zero-sum games, generative adversarial networks and distributionally robust learning. Despite the significance of MFL-SDA in these contexts, the discretetime convergence rate remains underexplored. To address this gap, we establish a last-iterate convergence rate of O( 1ϵ log 1 ϵ ) for MFL-SDA. This rate is nearly optimal when compared to the complexity lower bound of its Euclidean counterpart. This rate also matches the complexity of mean-field Langevin stochastic gradient descent for distributional minimization and the outer-loop iteration complexity of an existing double-loop algorithm for distributional minimax problems. By leveraging an elementary analysis framework that avoids PDE-based techniques, we overcome previous limitations and achieve a faster convergence rate.

Section: Introduction
In this paper, we study a distributional minimax optimization of the form
min µ∈P1 max ν∈P2 E(µ, ν),(1)
where P 1 , P 2 are sets of probability measures and E is a convex-concave probability functional. This formulation generalizes minimax optimization problems in Euclidean spaces and arises widely in many important applications, including generative adversarial networks (GANs) (Goodfellow et al., 2020;Arjovsky et al., 2017), distributionally robust learning (M ądry et al., 2017), as well as zerosum games with mixed Nash equillibrium (Daskalakis & Panageas, 2018).
Solving such optimization problems over distributional spaces is inherently challenging. Unlike in Euclidean spaces, the convexity/concavity of the objective function does not directly guarantee the convergence of gradient-based methods.
To address this, mean-field Langevin dynamics (MFLD) has emerged as a powerful theoretical framework (Mei et al., 2018;Sirignano & Spiliopoulos, 2020;Hu et al., 2021;Chizat, 2022;Nitanda et al., 2022a;Suzuki et al., 2024). In this framework, an entropy-regularized distributional convex minimization is considered:
min µ∈P(Θ) E(µ) -τ H(µ),
where H(µ) = -E µ [log µ] is the entropy functional. The mean-field Langevin gradient descent method in discrete time proceeds as follows:
θ k+1 = θ k -η∇ δE δµ [µ k ](θ k ) + 2ητ ξ k ,
where µ k is the distribution of θ k ; δE δµ [µ k ] is the first variation of E with respect to µ at µ k ; and ξ k is an independent injected standard Gaussian noise. Notably, the gradient in this context is often replaced with an unbiased stochastic gradient estimator to accommodate practical settings such as stochastic gradient descent. By leveraging the uniform log-Sobolev inequality, corresponding to the Polyak-Łojasiewicz (PL) condition (Karimi et al., 2016) in the distributional space, recent works have established exponential (linear) convergence for MFLD in continuous time (Chizat, 2022;Nitanda et al., 2022a) sublinear convergence of the order O( 1 ϵ log 1 ϵ ) in discrete time (Nitanda et al., 2022a) and with a stochastic gradient oracle (Suzuki et al., 2024).
The MFLD framework can be extended to distributional minimax problems (1), giving rise to the Mean-Field Langevin Descent-Ascent (MFL-DA) algorithm. For instance, under two-sided Polyak-Łojasiewicz conditions, Lu (2023) established a linear convergence rate for MFL-DA in continuous time. Furthermore, Kim et al. (2024) proposed two variants: the Mean-Field Langevin Averaged Gradient method, which guarantees average-iterate convergence, and the Mean-Field Langevin Anchored Best Response, a symmetric doubleloop algorithm whose outer loop achieves linear last-iterate convergence. Despite recent advances, the convergence properties of single-loop discrete-time algorithms remain largely unexplored. A notable exception is the discretetime stochastic Mean-Field Langevin Averaged Gradient algorithm introduced by Kim et al. (2024), which achieves an average-iterate convergence rate of O(ϵ -O(1/α) ), where α is the log-Sobolev constant. However, this rate appears suboptimal. Furthermore, last-iterate convergence is often more desirable in minimax optimization due to its practical significance. To our knowledge, the nearly optimal complexity of last-iterate convergence for discrete-time Mean-Field Langevin Stochastic Descent-Ascent (MFL-SDA) has not yet been investigated.
In contrast, minimax problems in Euclidean spaces have been extensively studied. For stochastic strongly convexstrongly concave functions, an optimal complexity of O(1/ϵ) is achieved by a variant of stochastic gradient descent-ascent (Yan et al., 2020;Zhang & Hu, 2025). Similarly, for problems with two-sided PL conditions using a stochastic oracle, an O(1/ϵ) convergence rate is attained by alternating stochastic gradient descent-ascent (Yang et al., 2020). In view of this, it is natural to ask whether MFL-SDA can achieve a similar last-iterate convergence rate in the distributional space under the two-sided distributional PL condition. These gaps in understanding motivate our central question: Can the discrete time MFL-SDA for (1) achieve a last-iterate complexity similar to that of minimax problems in the Euclidean space?
1.1. Related Work Mean Field Langevin Dynamics Our primary motivation stems from recent advancements in applying MFLD to neural networks. The pioneering works of Mei et al. (2018); Chizat & Bach (2018); Sirignano & Spiliopoulos (2020) leveraged MFLD to establish global convergence guarantees for (noisy) gradient descent in optimizing two-layer neural networks. Building on these foundations, Chizat (2022); Nitanda et al. (2022a) demonstrated that, under the log-Sobolev inequality-the distributional counterpart of the PL condition, continuous-time MFLD for distributional minimization can achieve an exponential convergence rate. Furthermore, Nitanda et al. (2022a) showed that the discrete-time MFLD enjoys a convergence guarantee with a complexity of O( 1ϵ log( 1 ϵ )). This rate also extends to set-tings involving stochastic gradient oracles and finite-particle approximations, as established by Suzuki et al. (2024). Our discrete-time analysis draws inspiration from a recent work by Wang et al. (2024), which established the convergence of MFLD in discrete time for one-hidden layer neural network with softmax activations and a square loss.
Distributional Minimax Optimization Compared to distributional minimization problems, the study of MFLD in distributional minimax problems remains limited, particularly in the discrete-time setting. The most relevant work in this area is by Kim et al. (2024), who investigated the convergence of general convex-concave functionals in both continuous and discrete time. Rather than analyzing the standard MFL-DA algorithm, they proposed two alternative methods: the Mean-Field Langevin Averaged Gradient (MFL-AG) (Tao et al., 2021) and the Mean-Field Langevin Anchored Best Response (MFL-ABR) algorithm (Lascu et al., 2023). MFL-AG updates are computed using a weighted average of past gradients instead of the current gradient.
For the discrete-time setting with a stochastic oracle, they demonstrated that the average-iterate convergence rate to an ϵ-optimal saddle point is O(ϵ -1-O(1/α) ), where α is the log-Sobolev constant. They derived an O(1/d) bound for α and noted that this dependence on the dimension of the sample space can sometimes be avoided. MFL-ABR is a double-loop algorithm, with the outer loop having an iteration complexity of O( 1 ϵ log 1 ϵ ). Additionally, Cai et al. (2024) analyzed MFL-SDA with a bilinear functional and a strongly convex-concave interaction function, an assumption not applicable in our context. Wang & Chizat (2022) introduced and analyzed a particle-based method inspired by the mirror prox algorithm in Euclidean space.
In addition, there has been some convergence analysis for MFL-DA in continuous time. Domingo-Enrich et al. (2020), Ma & Ying (2021) and Lu (2023) studied its convergence for finding the saddle point of an entropy-regularized objective. Qualitative convergence results were established in Domingo-Enrich et al. (2020), while Ma & Ying (2021) proved the asymptotic convergence under quasi-static conditions, where the ascent dynamics is infinitely faster or slower than the descent dynamics. Lu (2023) provided nonasymptotic exponential convergence rates with based on a two-sided PL condition. Zhu et al. (2024) demonstrated a sublinear convergence rate for the stochastic gradient descent-ascent algorithm for solving functional minimax optimization using mean-field neural networks, and showed that the discrete-time algorithm converges to its continuoustime counterpart.
Eucliean Minimax Optimization There is a substantial body of research on minimax optimization problems in Euclidean space (e.g., see recent works of Daskalakis & Panageas (2018); Doan (2022); Jin et al. (2020); Lin et al.  (2020); Li et al. (2022)), encompassing both the convexconcave setting and more general scenarios. Among these studies, some provide valuable insights for our work. For instance, Doan (2022) analyzed the convergence properties in Euclidean space under the PL condition using continuoustime analysis. Similarly, Yang et al. (2020) proposed a (stochastic) alternating gradient descent-ascent algorithm under the PL condition. We aim to achieve a similar convergence result in the distributional space.
this section cite: ['b10', 'b1', 'b7', 'b23', 'b29', 'b12', 'b5', 'b32', 'b14', 'b5', 'b32', 'b20', 'b16', 'b16', 'b39', 'b42', 'b40', 'b23', 'b6', 'b29', 'b5', 'b32', 'b16', 'b33', 'b17', 'b3', 'b36', 'b9', 'b21', 'b20', 'b21', 'b43', 'b7', 'b8', 'b40']

Section: Main contributions
We establish the last-iterate convergence rate of the MFL-SDA algorithm in discrete time. Our O( 1ϵ log 1 ϵ ) bound is nearly optimal, comparable to the O( 1ϵ ) bound for minimax problems in Euclidean space, and aligns with the O( 1ϵ log 1 ϵ ) complexity of mean-field Langevin stochastic gradient descent in distributional minimization problems (Suzuki et al., 2024) and the outer-loop iteration complexity of the doubleloop MFL-ABR algorithm (Kim et al., 2023).
Unlike previous studies that rely on PDE-based techniques, our proof is elementary and bears more resemblance to analyses conducted in Euclidean space. Our perturbation analysis is flexible enough to overcome the limitations of existing discrete-time algorithm analyses and thereby improves the convergence rate.
We apply our findings to several applications, including zero-sum games and mean field neural networks, and verify the essential assumptions required to ensure the theoretical convergence of MFL-SDA in these contexts.
this section cite: ['b32', 'b15']

Section: Preliminaries

this section cite: []

Section: Problem Setup
Let P(R dΘ ) (resp., P(R dΩ )) denote the space of probability distributions on R dΘ (resp., R dΩ ) where the entropy and second-order moment are well-defined. Let J : P(R dΘ ) × P(R dΩ ) → R be a convex-concave functional, in the sense that for any distributions µ 1 , µ 2 ∈ P(R dΘ ) and ν 1 , ν 2 ∈ P(R dΩ ) and any t ∈ [0, 1], the following conditions hold
J(tµ 1 + (1 -t)µ 2 , ν 1 ) ≤ tJ(µ 1 , ν 1 ) + (1 -t)J(µ 2 , ν 1 ), J(µ 1 , tν 1 + (1 -t)ν 2 ) ≥ tJ(µ 1 , ν 1 ) + (1 -t)J(µ 1 , ν 2 ).
We consider the following energy functional
E(µ, ν) := J(µ, ν) + τ KL(µ|ρ µ ) -τ KL(ν|ρ ν ), (2) where KL(µ|ρ µ ) = E µ log dµ dρ µ
denotes the Kullback-Leibler (KL) divergence; the reference distributions ρ µ , ρ ν are assumed to be standard Gaussian for simplicity, though they can be generalized to distributions with a strongly convex potential. The hyperparameter τ > 0 controls the regularization. This setup aligns with Kim et al. (2023) but differs slightly from Lu (2023), who considered entropy regularization. When the reference distributions are standard Gaussians, the KL divergence is equivalent to the entropy plus the second-order moment of the distribution. With KL (or entropy) regularization, the energy functional E(µ, ν) is strongly convex in µ and strongly concave in ν, which ensures the existence and uniqueness of the mixed Nash equilibrium (µ * , ν * ) (Kim et al., 2024, Proposition 2.1), where
E(µ * , ν) ≤ E(µ * , ν * ) ≤ E(µ, ν * ), ∀µ, ν.(3)
Denote by δJ δµ [µ, ν](•) and δJ δν [µ, ν](•) the first variations of J with respect to µ and ν, respectively, which are assumed to be well-defined throughout. Convexity/concavity can be also defined via first variations, as detailed in Appendix A.1. Denote by θ and ω the random variables with distributions µ and ν, respectively. Denote by ∇ δJ δµ [µ, ν](•) and ∇ δJ δν [µ, ν](•) the Wasserstein gradients of J with respect to µ and ν, respectively. For more details on Wasserstein gradient flow, see Ambrosio et al. (2008); Santambrogio (2015) .
The continuous-time mean-field gradient flow of (2) is given by
dθ t = -∇ δJ δµ [µ t , ν t ](θ t )dt -τ θ t dt + √ 2τ dB 1 t , dω t = η • ∇ δJ δν [µ t , ν t ](ω t )dt -τ ω t dt + √ 2τ dB 2 t .
(4) Here, µ t and ν t are the distributions of θ t and ω t at time t, respectively. The drift terms ∇ δJ δµ [µ t , ν t ](θ t ) and ∇ δJ δν [µ t , ν t ](ω t ) are the Wasserstein gradients with respect to µ t and ν t at time t, while {B 1 t } t , {B 2 t } t are two independent Brownian motions initialized at zero. The decay terms τ θ t and τ ω t correspond to the second-order moment regularization associated with the KL regularization. To simplify the presentation, we set the scaling factor and the weight decay to be the same as the hyperparameter for KL regularization, corresponding to the standard Gaussian, but our analysis can be easily extended to other choices. The scaling factor η > 0 follows from the formulation in Lu (2023), who derived an exponential convergence of (4) to the saddle point.
Algorithm 1 Mean field Langevin Stochastic Descent- Ascent (MFL-SDA) 1: Initialize µ0, ν0, K 2: for k = 1 to K -1 do 3: θ k+1 ← θ k -η1( ∇ δJ δµ [µ k , ν k ](θ k ) + τ θ k ) + √ 2η1τ ξ 1 k 4: ω k+1 ← ω k +η2( ∇ δJ δν [µ k+1 , ν k ](ω k )-τ ω k )+ √ 2η2τ ξ 2 k 5: end for
In this paper, we focus on the discrete-time counterpart of (4), as outlined in Algorithm 1. In this algorithm, ∇ denotes an unbiased stochastic gradient estimator, and ξfoot_0 k and ξ 2 k are independent standard normal random variables sampled at each iteration k. The algorithm employs dynamics with two timescales, η 1 , η 2 > 0, aligning with the continuoustime updates in Lu (2023) and the discrete-time Euclidean updates in Yang et al. (2020). Additionally, the updates for µ and ν are performed alternately, following a strategy similar to Yang et al. (2020) for Euclidean spaces. Our primary goal of this work is to analyze the convergence of Algorithm 1 to the saddle point (µ * , ν * ).
this section cite: ['b15', 'b20', 'b0', 'b28', 'b20', 'b20', 'b40', 'b40']

Section: Log-Sobolev Inequality
Unlike minimax problems in Euclidean space, the convexconcave property of the energy functional does not provide immediate algorithmic benefits for gradient-based methods in the Wasserstein space 1 . Conversely, a more generalized notion, the Polyak-Łojasiewicz (PL) condition, has proven useful for analyzing the gradient flow in the distribution space. In Euclidean space, the PL condition facilitates a linear convergence rate (Karimi et al., 2016). Extending this notion to distribution spaces leads to the log-Sobolev inequality, which has been employed in the analysis of optimization over distribution spaces with MFLD (Chizat, 2022;Nitanda et al., 2022a;Lu, 2023;Kim et al., 2023).
Definition 1. A distribution ν satisfies the log-Sobolev inequality with parameter α > 0 if, for all µ ≪ ν, the following holds:
KL(µ|ν) ≤ 1 2α I(µ|ν),(LSI)
where the relative Fisher information I(µ|ν) is defined as
I(µ|ν) := E µ ∇ log dµ dν 2 .
this section cite: ['b14', 'b5', 'b20', 'b15']

Section: Gibbs Distributions
To define the optimizer of the inner maximization in (2), we introduce several Gibbs distributions. We define the Gibbs operators K + µ [•] and K - ν [•] as
K + µ [ν](ω) ∝ exp τ -1 δJ δν [µ, ν](ω) - ∥ω∥ 2 2 2 ,(5)
K - ν [µ](θ) ∝ exp -τ -1 δJ δµ [µ, ν](θ) - ∥θ∥ 2 2 2 . (6)
When the energy functional J(µ, ν) is bilinear in µ and ν, Lu (2023) demonstrated that K + µ [ν] = arg max ν∈P(R d Ω ) E(µ, ν). However, this equality no longer holds when J(µ, ν) is a general nonlinear functional.
To address this, we consider the fixed point of (5), defined as
K + * [µ](ω) := 1 Z * (µ) exp τ -1 δJ δν µ, K + * [µ] (ω)- ∥ω∥ 2 2 2 ,
where
Z * (µ) := R d Ω exp τ -1 δJ δν µ, K + * [µ] (ω) - ∥ω∥ 2 2 2 dω
is a normalization constant. It can be verified that K + * [µ] satisfies the following equation of ν:
δJ δν [µ, ν](ω) -τ ∥ω∥ 2 2 2 -τ log ν(ω) = const, ∀ω,
where the constant is independent of ω but may depend on µ.
Observe that the above equation is the first-order condition for the problem max ν∈P(R d Ω ) E(µ, ν). Thus we have
E µ, K + * [µ] = max ν∈P(R d Ω ) E(µ, ν) =: E * (µ).(7)
This will be frequently used in our convergence analysis.
this section cite: ['b20']

Section: Main Results
The goal of this section is to establish the convergence of MFL-SDA (Algorithm 1). Following Yang et al. (2020); Lu (2023), we introduce the Lyapunov function
L(µ, ν) := L 1 (µ) + λL 2 (µ, ν),
where λ > 0 is a fixed constant, and
L 1 (µ) := max ν ′ ∈P(R d Ω ) E(µ, ν) -min µ ′ ∈P(R d Θ ) max ν ′ ∈P(R d Ω ) E(µ ′ , ν ′ ), L 2 (µ, ν) := max ν ′ ∈P(R d Ω ) E(µ, ν ′ ) -E(µ, ν).
(8) Note that L 1 and L 2 are both non-negative and vanish if and only if (µ, ν) = (µ * , ν * ) in the weak sense. We begin by stating our main assumptions.
this section cite: ['b40']

Section: Assumptions
We impose the following assumptions.
this section cite: []

Section: Assumption 1 (Initial condition). The initial iterate (µ
0 , ν 0 ) satisfies E(µ 0 , ν 0 ) < ∞, and the initial third-order mo- ments satisfy E µ0 [∥θ 0 ∥ 4 2 ], E ν0 [∥ω 0 ∥ 4 2 ] < ∞.
Assumption 2 (Regularity of functional J). The functional J(µ, ν) is convex in µ and concave in ν. The first variations of J have bounded derivatives up to the fourth order: ∥∇ i δJ δµ ∥ F , ∥∇ i δJ δν ∥ F ≤ M i , i = 1, . . . , 4, where ∥ • ∥ F is Frobenius norm. Additionally, J has a bounded cross second-order variation: ∥ δ 2 J δµδν ∥ ∞ ≤ C 0 . Moreover, the Hessian of its second variations are bounded:
∥∇ θ ∇ ⊤ θ ′ δ 2 J δµ 2 ∥ F , ∥∇ θ ∇ ⊤ ω δ 2 J δµδν ∥ F , ∥∇ ω ∇ ⊤ ω ′ δ 2 J δν 2 ∥ F ≤ C 1 , ∥∇ θ δ 2 J δµδν ∥ ∞ , ∥∇ ω δ 2 J δµδν ∥ ∞ ≤ C 2 .
Assumption 3 (Log-Sobolev inequality). For any µ ∈ P(R dΘ ), ν ∈ P(R dΩ ), the measures K + µ [ν], K - ν [µ] satisfy LSI with parameter α.
We will validate these assumptions in various applications, as discussed in Section 4 and Appendix B.1. Remark 1. Using Suzuki et al. (2024); Kim et al. (2023), Asssumption 2 implies Assumption 3 with a conservative LSI constant; see Lemma 5 in the Appendix. Remark 2. For simplicity, we assume d Θ = d Ω = d in proofs related to convergence rates.
this section cite: ['b32', 'b15']

Section: Convergence Analysis
At a high level, our analysis framework parallels that of stochastic gradient descent-ascent in Euclidean spaces (e.g., Yang et al. (2020)). However, analyzing the evolution of distributions in our problem presents greater challenges. Specifically, the noise terms ξ 1 k and ξ 2 k introduce timediscretization error terms that do not appear in Euclidean problems, even when the gradient oracle is exact. To show that these errors are negligible higher-order terms, most existing analyses leverage the connection between discretetime updates and continuous-time gradient flow (4) (e.g., Vempala & Wibisono (2022 2024)). In contrast, we directly analyze the discrete-time updates and carefully bound the higher-order terms in Taylor expansions through integration by parts and the divergence theorem. As will be seen from Theorem 2, our approach is flexible enough to handle the stochastic gradient oracle in a straightforward manner, similar to the Euclidean case.
Below, to illustrate the main idea, we first present the convergence analysis of MFL-DA, where the algorithm has access to exact gradients, then extend our results to MFL-SDA in Section 3.3.
Observe that L(µ k+1 , ν k+1 ) -L(µ k , ν k ) =L 1 (µ k+1 ) -L 1 (µ k ) + λ(L 2 (µ k+1 , ν k+1 ) -L 2 (µ k , ν k )).
In the following, we bound the two terms
L 1 (µ k+1 ) - L 1 (µ k ) and L 2 (µ k+1 , ν k+1 ) -L 2 (µ k , ν k ) in
Sections 3.2.1 and 3.2.2, respectively. Once these bounds are established, we sum them over all iterations and apply a telescoping argument to derive the overall convergence rate in Section 3.2.3.
To simplify the presentation, we define
g k := ∇ δJ δµ [µ k , ν k ](θ k ) + τ θ k , h k := ∇ δJ δν [µ k+1 , ν k ](ω k ) -τ ω k , f k := ∇ δJ δµ [µ k , K + * [µ k ]](θ k ) + τ θ k ,
where f k , depending on the Gibbs distribution K + * [µ k ], is only used for theoretical analysis but not in the implementation of algorithm. Moreover, we omit the constants appearing in the higher-order error bounds, whose explicit expressions are provided in Appendix A.3.
3.2.1. BOUNDING L 1 (µ k+1 ) -L 1 (µ k ) By definition of L 1 in (8) and E * in (7), it holds that L 1 (µ k+1 ) -L 1 (µ k ) = E * (µ k+1 ) -E * (µ k ).
We have the following result.
Proposition 1. Assume Assumptions 1-3 hold. Let η 1 < 1 C1 . Then it holds that L 1 (µ k+1 ) -L 1 (µ k ) ≤ - η 1 2 E µ k [∥f k + ∇ log µ k ∥ 2 2 ] -E µ k [∥g k -f k ∥ 2 2 ] + Γ 0 η 2 1 , where remainder Γ 0 is defined in Appendix A.3. Proposition 1 establishes that the difference L 1 (µ k+1 ) - L 1 (µ k ), or equivalently, E * (µ k+1 ) -E * (µ k ), is bounded by the squared Wasserstein gradient norm of E * (µ) at µ k .
The bound consists of two dominant terms, corresponding to the squared norm of the partial Wasserstein gradients of E µ, K + * [µ] -an equivalent form of E * (µ)-with respect to the first and second arguments of the energy functional E. The first term arises from the outer minimization problem in (2) with respect to µ, while the second term serves as a correction due to the inner maximization in (2) over ν. This result parallels the convergence analysis in Euclidean space (Yang et al., 2020). However, obtaining the O(η 2 1 ) remainder requires significant effort, as it involves analyzing the smoothness of the functional E * (µ), which, in turn, depends on the smoothness of the operator K + * [µ]; see Lemma 9 in the Appendix for details.
3.2.2. BOUNDING L 2 (µ k+1 , ν k+1 ) -L 2 (µ k , ν k ) By definition, we have L 2 (µ k+1 , ν k ) -L 2 (µ k , ν k ) = E * (µ k+1 ) -E * (µ k ) + E(µ k , ν k ) -E(µ k+1 , ν k ) .
We have already established an upper bound on the first difference above in Proposition 1. It remains to bound the second difference. Lemma 1. Assume Assumptions 1-3 hold. Then we have
E(µ k+1 , ν k+1 ) -E(µ k+1 , ν k ) ≥ η 2 E ν k [∥h k -τ ∇ log ν k ∥ 2 2 ] -Γ 2 η 2 2 . Similarly, we have E(µ k+1 , ν k ) -E(µ k , ν k ) ≤ -η 1 E µ k [∥g k + ∇ log µ k ∥ 2 2 ] + Γ 1 η 2 1 .
Here, the remainders Γ 1 and Γ 2 are defined in Appendix A.3.
This lemma provides a lower bound on the per-step objective improvement for the inner gradient ascent and outer gradient descent in solving (2). The leading terms in both bounds correspond to the squared Wasserstein gradient norm with respect to ν k and µ k , respectively. The O(η 2 1 ) and O(η 2 2 ) bias terms arise from time discretization and are consistent with the results of Nitanda et al. (2022a) for distributional convex minimization problems.
Using Lemma 1 and Assumption 3, the following result is immediate.
Lemma 2. Assume Assumptions 1-3 hold. Then we have
L 2 (µ k+1 , ν k+1 ) ≤ (1 -2η 2 τ α)L 2 (µ k+1 , ν k ) + Γ 2 η 2 2 .
Using the bound in Proposition 1 and combining Lemma 1 and Lemma 2, we obtain the following bound on the difference between L 2 (µ k+1 , ν k+1 ) and L 2 (µ k , ν k ).
Proposition 2. Assume Assumptions 1-3 hold. Then we have
L 2 (µ k+1 , ν k+1 ) ≤ (1 -2η 2 τ α) L 2 (µ k , ν k ) + η 1 E µ k [∥g k + ∇ log µ k ∥ 2 2 ] - η 1 2 E µ k [∥f k + ∇ log µ k ∥ 2 2 ] + η 1 2 E µ k [∥g k -f k ∥ 2 2 ] + Γ 2 η 2 2 + (1 -2η 2 τ α)(Γ 1 + Γ 0 )η 2 1 .
Combining Propositions 1 and 2 yields a recursive bound on the Lyapunov function L.
this section cite: ['b40']

Section: CONVERGENCE OF MFL-DA
By applying Propositions 1 and 2, and using an argument similar to that in the Euclidean case (Yang et al., 2020), we can establish the following convergence result for MFL-DA. Note that different learning rates η 1 , η 2 are employed to ensure convergence, as is commonly done (Yang et al., 2020;Lu, 2023).
Theorem 1. Assume Assumptions 1-3 hold. Set τ < 1 2C 2 1 , η 1 ≤ 1 C1 , η 2 ≤ 1 2τ α and η 1 = min{λ, 0.2, 1 τ α }τ αη 2 , then it holds that L(µ K , ν K ) ≤ (1 -2η 1 τ α) K L(µ 0 , ν 0 ) + R 1 , (9) where R 1 = λ(Γ2η 2 2 +(1-2η2τ α)(Γ1+Γ0)η 2 1 )+Γ1η 2 1 η1τ α .
Theorem 1 demonstrates that the Lyapunov function L(µ, ν) converges to a bias R 1 at a geometric rate. This geometric decay aligns with the exponential decay observed in the continuous-time case, as established by Lu (2023). The bias term R 1 , which results from time discretization, is of order O(η 1 ). To assess the algorithm's complexity, since in practical algorithm we often assume τ, η 1 , η 2 is small, then the remainder r g4 , r h4 in Appendix A.3 caused by the fourth moment has max{O(1), τ 2 d 2 } scale. Substitute into Γ 0 , Γ 1 , Γ 2 we can get an estimation of these bias term:
Γ 0 = max{O(1), τ d, τ 2 d 2 , d α 1/2 τ } , Γ 1(2) = max{O(1), τ d, τ 2 d 2 } . Replace them into R 1 we can get a worst bound R 1 = O( dη1 τ 3 α 3 ). Let R 1 = ϵ, then choose η 1 = O( ϵτ 3 α 3 d ) to get a sample complexity K = O( d ϵτ 4 α 4 log 1 ϵ )
. This complexity matches that of MFLD for distributional convex minimization (Nitanda et al., 2022a) and the outerloop complexity of discrete-time MFL-ABR (Kim et al., 2024), and the sample complexity K = O( d ϵτ 2 α 2 log 1 ϵ ) in (Nitanda et al., 2022b) who discussed about discrete-time MFLD in the single minimization problem, the higher order of τ, α is because the two-timescale optimization scheme in minimax problem, hence the efficiency of this algorithm mainly depends on the slower part, which is the descent part in our paper.
Comparatively, in the Euclidean case, the (exact) gradient descent-ascent method (Yang et al., 2020, Theorem 3.1 with σ = 0) achieves a linear convergence rate of O( 1ϵ ), where the bias term is absent. This is because, in the Euclidean setting, higher-order terms in the Taylor expansion can be absorbed into the first-order squared gradient norm, resulting in a contraction of the Lyapunov function. However, in our distributional case, the randomness introduced by Gaussian noise prevents the absorption of higher-order terms into the first-order term, leading to a sublinear convergence rate. Remark 3. Similar to Lu (2023), we can also consider the Lyapunov function L 3 (ν) + λL 4 (µ, ν), where
L 3 (ν) := max ν ′ ∈P(R d Ω ) min µ ′ ∈P(R d Θ ) E(µ ′ , ν ′ ) -min µ ′ ∈P(R d Θ ) E(µ ′ , ν), L 4 (µ, ν) := E(µ, ν) -min µ ′ ∈P(R d Θ ) E(µ ′ , ν),
which is useful for max-min problem. The result is similar to Theorem 1 but with a reverse scaling of η 1 and η 2 .
this section cite: ['b40', 'b40', 'b20', 'b20', 'b16', 'b40', 'b20']

Section: Convergence of MFL-SDA
In the previous subsection, we assumed the availability of exact gradients. In practice, however, we often work with stochastic gradients, where the exact gradient is replaced by an unbiased estimate. Thanks to the similar high-level structure of our analysis to the Euclidean case (Yang et al., 2020), our results can be extended to the stochastic gradient setting in a straightforward manner. We can show that the convergence rate of MFL-SDA is analogous to that of MFL-DA, with the same geometric decay rate and bias term. The main difference is that the bias term now depends on the higher-order moments of the stochastic gradients.
this section cite: ['b40']

Section: Assumption 4 (Bounded moments). There exists
ζ ≥ 0 such that E[∥ĝ k -g k ∥ 4 |µ k , ν k ], E[∥ ĥk -h k ∥ 4 |µ k+1 , ν k ] ≤ ζ,
We have the following result for MFL-SDA analogous to Theorem 1. Theorem 2. Under the same setup as in Theorem 1, and assume further that Assumption 4 holds. Then we have
E[L(µ K , ν K )] ≤ (1 -2η 1 τ α) K L(µ 0 , ν 0 ) + R 2 , where R 2 = λ( Γ2η 2 2 +(1-2η2τ α)( Γ1+ Γ0)η 2 1 )+ Γ1η 2 1 η1τ α
with Γ0 , Γ1 , Γ2 defined in (17), and the expectation is taken over the randomness in the stochastic gradients.
This result demonstrates that the convergence rate of MFL-SDA (Algorithm 1) is O( 1 ϵ log 1 ϵ ). This rate is nearly optimal when compared to the O( 1ϵ ) convergence rate of stochastic gradient descent-ascent for minimax optimization with a two-sided PL condition in Euclidean space, as discussed in Yang et al. (2020).
Proof Sketch. Let H k be σ-field generated by the random gradients up to iteration k -1. Similar to the Euclidean case, at iteration k, we analyze the impact of using stochastic gradient oracles on the bound E[L(µ k+1 , ν k+1 ) | H k ] -L(µ k , ν k ). By controlling the second-and thirdorder moments of the stochastic gradients, we ensure that the convergence properties established in Section 3.2.3 remain valid, albeit with additional error terms. Specifically, thanks to the unbiasedness of the stochastic gradients and the moment bounds in Assumption 4, the error term introduced by stochastic gradients is O(η 2 ) or higher order. This holds for both the squared gradient norm and the entropy regularization. By carefully bounding these error terms, we show that the overall convergence result holds with a modified remainder term that accounts for the inexactness of the gradient. Furthermore, leveraging the connection between the Lyapunov function and KL-divergence, we can also obtain convergence rate in terms of KL-divergence to (µ * , ν * ) and further in terms of Wasserstein distance. Corollary 1 (Convergence in KL-divergence / Wasserstein distance). Under the same setup as in Theorem 2, suppose that K + * [µ] and K - * [ν] satisfy LSI with constant α 1 , then it holds that
2τ α 1 (W 2 2 (µ k , µ * ) + W 2 2 (ν k , ν * )) ≤ τ (KL(µ K |µ * ) + KL(ν K |ν * )) ≤ (1 -η 1 τ α) K QL(µ 0 , ν 0 ) + QR 1 , where Q = 1 + ( 2 λ + 4C 2 0 τ 2 )
and W 2 denotes the 2-Wasserstein distance.
Similar to Remark 1, Assumption 2 implies a conservative upper bound on the LSI constant α 1 . Kim et al. (2023) established that the stochastic MLF-AG achieves a convergence rate of O(ϵ -1-O(α -1 ) ) in terms of the squared 1-Wasserstein distance, where α is the LSI constant in Assumption 3. Since W 2 upper bounds W 1 , Corollary 1 improves the existing complexity bound for stochastic MLF-AG, particularly when the log-Sobolev constant α is small. Notably, it surpasses the conservative α = O(1/d) bound established in Proposition 3.2 of their paper.
this section cite: ['b40', 'b15']

Section: Applications

this section cite: []

Section: Zero-Sum Games
Zero-sum games are widely applicable in economics, operations research, and reinforcement learning. These games involve a payoff function G(θ, ω), which defines the interaction between two players' strategies θ and ω. While finding a pure Nash equilibrium can be challenging or even impossible when G is nonconvex-nonconcave, a mixed Nash equilibrium (MNE) often exists. In an MNE, players optimize their mixed strategies, represented as probability distributions over available actions.
Following Lu (2023), consider a bilinear distributional minimax optimization problem, where the strategies of two players are represented by probability distributions µ ∈ P 1 and ν ∈ P 2 . The energy functional J(µ, ν), which captures the expected payoff, is expressed as
J(µ, ν) = E µ⊗ν [G(θ, ω)]. (10
)
The goal is to find an MNE (µ * , ν * ) that satisfies µ * ∈ arg min µ∈P1 E(µ, ν * ) and ν * ∈ arg max ν∈P2 E(µ * , ν). This formulation extends the classical minimax problem to a distributional setting, where players optimize over probability measures rather than deterministic strategies.
Example 1 (GAN). Consider the following generative adversarial network with an integral probability metric:
min µ∈P(R d Θ ) max f ∈F E µ [f ] -E p data [f ] ,
where p data is the real data distribution, µ is the distribution of the generated data, and f is a discriminator function. Suppose that the discriminator function f is parameterized by a (infinite-width) two-layer neural network with an activation function σ(•, ω) parameterized by ω under the mean-field scaling, so that every function f ∈ F can be expressed as
f (θ) = E ω∼ν [σ(θ, ω)].
Then the generative adversarial network can be formulated as a distributional minimax optimization problem with
J(µ, ν) = E µ E ν [σ(θ, ω)] -E p data E ν [σ(θ, ω)] ,
which is a bilinear functional. ♢
The bilinear nature of the functional J simplifies the analysis, as the Wasserstein gradients are given by
∇ θ δJ δµ [µ, ν] = E ω∼ν [∇ θ G(θ, ω)], ∇ ω δJ δν [µ, ν] = E θ∼µ [∇ ω G(θ, ω)].
and the Gibbs distribution K + µ [ν] satisfies the first-order optimality condition of the inner maximization problem in (1). Under mild regularity conditions on G, we can verify that Assumptions 2-3 are satisfied.
Proposition 3. Assume the payoff function G satisfies that ∥∇ i G∥ F ≤ G i , i = 0, 1, . . . , 4. Then the functional J in (10) satisfies Assumption 2, and Assumption 3 holds with α = 1 exp(2G0τ -1 ) .
this section cite: ['b20']

Section: Mean-Field Neural Networks
Consider a functional minimax problem
min f max g E z∼D [F (f (z), g(z), z)],(11)
where f and g are functions of a variable z. The objective function F is convex in f and concave in g, and the expectation is taken with respect to z ∼ D. We parameterize f and g as infinite-width two-layer neural networks with activations σ 1 and σ 2 , respectively:
f (z) = E θ∼µ [σ 1 (θ, z)], g(z) = E ω∼ν [σ 2 (ω, z)].
This transforms the original problem into a minimax problem in distributional space:
min µ max ν E z∼D F (E θ∼µ [σ 1 (z, θ)], E ω∼ν [σ 2 (z, ω)], z) .(12)
Example 2 (Functional Conditional Moment Equations).
The conditional moment equation is a fundamental problem in econometrics and statistics. Given a dataset z = (X, Y ) ∼ D, the goal is to find a function f that solves the following functional equation involving the conditional distribution of X given Z:
E Y |X [Φ(f (X, Y ), Y ) | X = x] = 0, ∀x,
where Φ is a known function that is convex in f . Examples of Φ include conditional moment equations in nonparametric instrumental variable regression, policy evaluation in reinforcement learning, and asset pricing models in finance (Zhu et al., 2024). Using a Lagrangian dual function g, this problem can be formulated as a distributional minimax optimization problem by setting g(z) = g(X) and
F (f, g, z) = gΦ(f, Y ). ♢
Example 3 (Feature-based Policy Learning). Feature-based decision-making (Yang et al., 2022) aims to find a policy f from a set of features X to an action f (X). Given a data set z = (X, Y ) ∼ D, where Y is some exogenous random variable, the goal is to minimize the expected loss E D [ℓ(f (X), Y )], subject to feasibility constraints Af (X) ≤ b for every X. By introducing a Lagrangian dual function g(z) = g(X), this problem can be formulated as a distributional minimax optimization problem with the objective
F (f, g, Z) = ℓ(f, Z) + g(Af -b). ♢
We introduce regularity assumptions on F and σ 1 , σ 2 , which implies Assumption 2.
Assumption 5. The function F (x, y) is convex-concave, L-smooth in both x and y, and has bounded derivatives (i.e., ∥F ′ x ∥, ∥F ′ y ∥ ≤ F 1 ). Moreover, we assume that σ 1 , σ 2 has bounded gradients up to fourth-order, i.e.,
∥∇ i σ 1 ∥, ∥∇ i σ 2 ∥ ≤ m i , i = 0, 1, . . . , 4.
Under this regularity condition, we can also show that the Gibbs distributions K + µ [ν] and K - ν [µ] satisfy the log-Sobolev inequality, which verifies Assumption 3.
Proposition 4. Under Assumption 5, (12) satisfies Assumption 2. Meanwhile, for J(µ, ν) defined in (12), we have K + µ [ν] ∈ L 1 (R dΩ ) and K - ν [µ] ∈ L 1 (R dΘ ) and they both satisfy the Log-Sobolev inequality with parameter α 1 = 1 exp(2F1m0τ -1 ) .
this section cite: ['b43', 'b41']

Section: Conclusion
In this paper, we establish an Õ(1/ϵ) last-iterate convergence guarantee for the Mean-Field Langevin Stochastic Descent Ascent (MFL-SDA) algorithm. We also explore several common applications, including zero-sum games and mean-field neural networks.
There are several directions for future research. First, in practical applications, the MFL-SDA algorithm often requires finite-particle approximation, which calls for further analysis to establish uniform-in-time propagation of chaos. To address this issue, our analysis can be combined with the techniques from Chen et al. (2022); Suzuki et al. (2023;2024); Nitanda (2024); Kim et al. (2023). Second, while our results apply to a broad class of nonlinear functionals, more specialized analyses for specific functionals-such as bilinear forms or convex-concave functions of expectationsmay lead to sharper convergence guarantees. Lastly, our current algorithm follows a two-time-scale framework with a relatively large time-scale ratio, in line with Yang et al. (2020); Lu (2023). It remains an open question whether a single-timescale approach could be advantageous in certain settings, as highlighted in Wang & Chizat (2024).
this section cite: ['b4', 'b31', 'b24', 'b15', 'b40', 'b20', 'b37']

Section: References
Ref_id:b0 Title: Gradient flows: in metric spaces and in the space of probability measures Year: (2008)
Ref_id:b1 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b2 Title: Diffusions hypercontractives Year: (2006)
Ref_id:b3 Title: Convergence of the min-max langevin dynamics and algorithm for zerosum games Year: (2024)
Ref_id:b4 Title: Uniform-in-time propagation of chaos for mean field langevin dynamics Year: (2022)
Ref_id:b5 Title: Mean-field langevin dynamics : Exponential convergence and annealing Year: (2022)
Ref_id:b6 Title: On the global convergence of gradient descent for over-parameterized models using optimal transport Year: (2018)
Ref_id:b7 Title: Last-iterate convergence: Zero-sum games and constrained min-max optimization Year: (2018)
Ref_id:b8 Title: Convergence rates of two-time-scale gradient descent-ascent dynamics for solving nonconvex min-max problems Year: (2022)
Ref_id:b9 Title: A mean-field analysis of two-player zerosum games Year: (2020)
Ref_id:b10 Title: Generative adversarial networks Year: (2020)
Ref_id:b11 Title: Logarithmic sobolev inequalities and stochastic ising models Year: (1986)
Ref_id:b12 Title: Mean-field langevin dynamics and energy landscape of neural networks Year: (2021)
Ref_id:b13 Title: What is local optimality in nonconvex-nonconcave minimax optimization Year: (2020)
Ref_id:b14 Title: Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition Year: (2016)
Ref_id:b15 Title: Symmetric mean-field langevin dynamics for distributional minimax problems Year: (2023)
Ref_id:b16 Title: Symmetric mean-field langevin dynamics for distributional minimax problems Year: (2024)
Ref_id:b17 Title: Entropic meanfield min-max problems via best response flow Year: (2023)
Ref_id:b18 Title: Nonsmooth nonconvexnonconcave minimax optimization: Primal-dual balancing and iteration complexity analysis Year: (2022)
Ref_id:b19 Title: On gradient descent ascent for nonconvex-concave minimax problems Year: (2020)
Ref_id:b20 Title: Two-scale gradient descent ascent dynamics finds mixed nash equilibria of continuous games: A mean-field perspective Year: (2023)
Ref_id:b21 Title: Provably convergent quasistatic dynamics for mean-field two-player zero-sum games Year: (2021)
Ref_id:b22 Title: Towards deep learning models resistant to adversarial attacks Year: (2017)
Ref_id:b23 Title: A mean field view of the landscape of two-layer neural networks Year: (2018)
Ref_id:b24 Title: Improved particle approximation error for mean field neural networks Year: (2024)
Ref_id:b25 Title: Convex analysis of the mean field langevin dynamics Year: (2022)
Ref_id:b26 Title: Convex Analysis of the Mean Field Langevin Dynamics, February 2022b Year: ()
Ref_id:b27 Title: Generalization of an inequality by talagrand and links with the logarithmic sobolev inequality Year: (2000)
Ref_id:b28 Title: Progress in Nonlinear Differential Equations and Their Applications Year: (2015)
Ref_id:b29 Title: Mean field analysis of neural networks: A law of large numbers Year: (2020)
Ref_id:b30 Title: Some inequalities satisfied by the quantities of information of fisher and shannon Year: (1959)
Ref_id:b31 Title: Uniform-in-time propagation of chaos for the mean-field gradient langevin dynamics Year: (2023)
Ref_id:b32 Title: Mean-field langevin dynamics: Time-space discretization, stochastic gradient, and variance reduction Year: (2024)
Ref_id:b33 Title: Gradient descent averaging and primal-dual averaging for strongly convex optimization Year: (2021)
Ref_id:b34 Title: Rapid Convergence of the Unadjusted Langevin Algorithm: Isoperimetry Suffices Year: (2022-03)
Ref_id:b35 Title: On a New Class of Weak Solutions to the Spatially Homogeneous Boltzmann and Landau Equations Year: (1998-09)
Ref_id:b36 Title: An exponentially converging particle method for the mixed nash equilibrium of continuous games Year: (2022)
Ref_id:b37 Title: Open problem: Convergence of single-timescale mean-field Langevin descent-ascent for two-player zero-sum games Year: (2024-06)
Ref_id:b38 Title: Neural-network mixed logit choice model: Statistical and optimality guarantees Year: (2024)
Ref_id:b39 Title: Optimal epoch stochastic gradient descent ascent methods for minmax optimization Year: (2020)
Ref_id:b40 Title: Global convergence and variance reduction for a class of nonconvex-nonconcave minimax problems Year: (2020)
Ref_id:b41 Title: Decisionmaking with side information: A causal transport robust approach Year: (2022)
Ref_id:b42 Title: Avoid overclaims: Summary of complexity bounds for algorithms in minimization and minimax optimization Year: (2025)
Ref_id:b43 Title: A Mean-Field Analysis of Neural Stochastic Gradient Descent-Ascent for Functional Minimax Optimization Year: (2024-10)
