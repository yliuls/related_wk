Title: Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems
Abstract: Designing algorithms for solving high-dimensional Bayesian inverse problems directly in infinite-dimensional function spaces-where such problems are naturally formulated-is crucial to ensure stability and convergence as the discretization of the underlying problem is refined. In this paper, we contribute to this line of work by analyzing a widely used sampler for linear inverse problems: Langevin dynamics driven by score-based generative models (SGMs) acting as priors, formulated directly in function space. Building on the theoretical framework for SGMs in Hilbert spaces, we give a rigorous definition of this sampler in the infinite-dimensional setting and derive, for the first time, error estimates that explicitly depend on the approximation error of the score. As a consequence, we obtain sufficient conditions for global convergence in Kullback-Leibler divergence on the underlying function space. Preventing numerical instabilities requires preconditioning of the Langevin algorithm and we prove the existence and the form of an optimal preconditioner. The preconditioner depends on both the score error and the forward operator and guarantees a uniform convergence rate across all posterior modes. Our analysis applies to both Gaussian and a general class of non-Gaussian priors. Finally, we present examples that illustrate and validate our theoretical findings.

Section: Introduction
Inverse problems arise in many challenging applications, such as X-ray computed tomography, seismic tomography, inverse heat conduction, and inverse scattering. These problems share a common goal: to estimate unknown parameters from noisy observations or measurements [1]. What makes them difficult is that they are often ill-posed in the sense of Hadamard [2]: they may have multiple solutions, no solutions at all, or solutions that are highly sensitive to small perturbations in the data. A possible approach to address these difficulties is to cast the problem in a probabilistic framework known as Bayesian inference. In the Bayesian approach, one first specifies a prior distribution that encodes knowledge about the unknown before any data is observed, along with a model for the observational noise. Bayes' rule is then used to update this prior knowledge in light of the measurements, yielding the so-called posterior distribution, which describes the distribution of the unknown conditioned on the data. By sampling from the posterior one can extract statistical information and quantify uncertainty in the solution [3][4][5][6].
A central challenge in applying Bayesian inference to inverse problems is that in many casesespecially those governed by partial differential equations (PDEs)-the unknowns to be estimated are functions that lie in a suitable function space, typically an infinite-dimensional Hilbert space. It is therefore crucial to design Bayesian inference algorithms that are both theoretically sound and computationally effective in arbitrarily high dimensions. A way to achieve this is by lifting these problems to an infinite-dimensional space and designing inference methods directly in that setting. This approach, sometimes referred to as "apply-algorithm-then-discretize"-or, in the context of Bayesian inference, "Bayesianize-then-discretize"-allows for the development of algorithms that are inherently discretization-invariant, as the Bayes formula and algorithms are properly defined on Hilbert spaces [3,7]. In contrast, the opposite approach-"discretize-then-Bayesianize"-can lead to several issues, such as instability as the discretization of the underlying problem is refined, or worse, methods that seem stable but whose results are theoretically implausible [8,9]. These considerations manifest clearly even in simple scenarios. In Figure 1 we consider two examples involving a vanilla diffusion Langevin sampler. In the first one, we sample from a Gaussian posterior. While the method appears numerically stable and produces samples with seemingly reasonable behavior, a closer inspection shows that the samples carry infinite energy-they do not belong to the infinite-dimensional Hilbert space. That is, the algorithm is producing objects that are not valid functions in the limit of refined discretization. In the second example, we attempt to fix this by choosing a trace-class prior, which ensures that samples have finite energy and are well-defined in a Hilbert space. This theoretically-motivated structure, however, comes at a cost: without adjustments, the drift of the vanilla Langevin sampler may diverge at fine scales. Figure 1: We consider the toy linear inverse problem y j = A jj X (j) 0 + n j in the basis (v j ) of the Hilbert space H, with A jj = e -0.1j and n j ∼ N (0, 0.05 2 ), for j ≤ 100. In the top row, we sample the posterior using an identity prior covariance on X 0 . The Langevin diffusion seems stable, but the eigenvalues of the posterior covariance do not decay at infinity and therefore the samples do not belong to the Hilbert space H. In the bottom row, we use a trace-class covariance prior with diagonal terms ∼ 1/j 2 ; the drift of the vanilla Langevin sampler starts diverging at fine scales. These types of challenges, intrinsic to the infinite-dimensional setting, have long been studied in the Bayesian inverse problems community, but are now receiving renewed attention with the rise of deep learning methods for posterior sampling. One popular class of methods that still lacks a complete theoretical understanding in this context is score-based generative models (SGMs), which generate samples from complex distributions by first learning the (Stein) score-the gradient of the log-density [10]-and then using it in various sampling algorithms [11,12]. For example, [13] employs the learned score in a Langevin-based sampler, while [14] unified SGMs and diffusion-based methods [15,16] through a stochastic differential equation (SDE) framework, known as score-based diffusion models. After their introduction, SGMs have been applied successfully to Bayesian inverse problems, either by learning the score conditioned on data [17][18][19], or by using the score of the prior distribution-the unconditional score model-within Langevin-type samplers. Crucially, with a few exceptions [20][21][22][23], these approaches assume that the posterior is supported in a finite-dimensional space, leaving the challenges of infinite dimensions to heuristics and ad hoc solutions.
In this work, we present a detailed analysis of SGMs for Bayesian inference of linear inverse problems, going beyond the common assumption that the posterior is supported on a finite-dimensional space. We focus on a widely used posterior sampling technique that combines SGMs-used as powerful learned priors to capture complex features-with a Langevin-based sampler [24][25][26]. Lifting the problem directly to function spaces is not a mere technicality: we show that, to provably sample the posterior, the Langevin diffusion must be modified by a preconditioning operator C acting on the Hilbert space. This preconditioner is not an ad hoc fix but rather built into the fabric of the infinitedimensional setting: it first appears in the forward diffusion process (5) whose time-reversal is used to learn the prior, and must then be carried through into the Langevin sampler to ensure convergence to the correct posterior (Section 3). Crucially, C cannot be the identity: for the time-reversed diffusion to remain Hilbert-space-valued, C must be trace class. Setting C = I leads to the same theoretical and numerical issues as highlighted in Figure 1 above.
The importance of preconditioning in function spaces has been well established in the context of posterior sampling [3,8,[27][28][29][30], but its implications have not yet fully explored for infinitedimensional SGMs. In this setting, we characterize the interplay between the preconditioner C, the trace-class prior, the score approximation error, and the linear forward map of the inverse problem. In particular, we analyze the impact of the score approximation error at small times-where the score is learned in practice-and identify a theoretically optimal preconditioner that ensures uniform convergence rates across posterior modes (Section 4). We carry out the analysis by focusing on two cases: a Gaussian prior measure, and a more general class of priors which are absolutely continuous with respect to a Gaussian measure (Section 5). Illustrations are provided in Section 6.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b2', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b2', 'b7', 'b26', 'b27', 'b28', 'b29']

Section: Related Work.
There exists a large body of literature on infinite-dimensional MCMC algorithms [3,8,[27][28][29][30][31][32][33][34][35][36][37][38][39][40], which include a variety of preconditioning strategies for posterior sampling. However, these works precede the recent wave of papers on SGMs and therefore do not address the central focus of our analysis: the interplay between the score approximation error, the preconditioning operator, the trace-class prior, and the sampler convergence, which we study in detail in both the Gaussian and non-Gaussian settings.
Closest to our work are the papers that use SGMs for posterior sampling, such as [21,[24][25][26], which employ SGMs as learned priors in a Langevin-type diffusion algorithm. Among these works, the theoretical analysis of [25] is the most directly related to ours. However, there are key differences. [25] analyze Langevin dynamics with SGMs for posterior sampling in finite dimensions, as their results provide convergence error estimates that explicitly depend on the score approximation error but diverge as the dimension of the problem increases. In contrast, our error analysis, since it is formulated directly in infinite dimensions, provides conditions to ensure global boundedness (Theorem 3.1). Moreover, the finite-dimensional setting of [25] does not address the role of preconditioning, which becomes essential in infinite dimensions. Other related works include [41,42] which investigate preconditioning in Langevin dynamics with SGMs. However, these analyses are also finite-dimensional and do not account for the score approximation error. As a result, they do not capture the critical role of preconditioning, which-as we show in Section 4-becomes crucial in function spaces.
As we have pointed out several times, the learned score plays a key role in our analysis. Among the theoretical frameworks defining SGMs in infinite dimensions [23,[43][44][45][46][47][48], we follow those of [20,49] for continuous-time diffusions. An important contribution of our work is to show that the convergence bound depends explicitly on the accuracy of the approximated score, and that controlling this error is key to designing a preconditioner that ensures convergence in function spaces (Theorem 4.1).
Finally, we note that [21] also explores the role of preconditioning to ensure convergence in infinite dimensions in the context of SGMs. Their analysis is conducted in a more complex setting-nonlinear inverse problems. Their argument builds on the proof of [25], but the difficulties of the nonlinear setting prevent them from identifying an optimal preconditioner. In contrast, our work takes full advantage of the linear setting, where the distributions at play admit explicit formulas. This allows us to derive detailed error estimates in the small diffusion time regime, where the score is typically learned, and discuss the impact of the score approximation error on posterior sampling-including the effects of preconditioning on the bias error. Furthermore, their analysis focuses only on Gaussian priors, while we generalize and consider non-Gaussian priors (Section 5).
this section cite: ['b2', 'b7', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b20', 'b23', 'b24', 'b25', 'b24', 'b24', 'b24', 'b40', 'b41', 'b22', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b19', 'b48', 'b20', 'b24']

Section: Langevin Posterior Sampling with Score-Based Generative Priors
We work in the setting of a linear Bayesian inverse problem formulated in infinite dimension. Let H be a separable Hilbert space with inner product ⟨• , •⟩, and let C, C µ : H → H be trace-class, positive-definite, symmetric covariance operators. The unknown quantity of interest is an H-valued random variable X 0 ∼ µ, where the prior measure µ is assumed to be absolutely continuous with respect to a Gaussian reference measure N (0, C µ ), with density
dµ dN (0, C µ ) (X) ∝ exp -Φ(X) .(1)
The observations y ∈ R N are modeled as
y = AX 0 + n,
where A : H → R N is a linear operator, and n ∼ N (0, σ 2 I N ) is Gaussian observational noise independent of X 0 . Since we consider an observational model corresponding to observing a finitedimensional subspace of H, there exists an orthonormal basis (v j ) of H such that Av j = 0 for all j > N . Let (e j ) denote the standard basis of R N . Then the observation model can be written as
y i = N j=1 A ij X (j) 0 + n i , where A ij = ⟨e i , A v j ⟩, y i = ⟨y, e i ⟩, X(j)
0 = ⟨X 0 , v j ⟩, and n i = ⟨n, e i ⟩. The posterior distribution π y of X 0 conditioned on the observations y is absolutely continuous with respect to N (0, C µ ):
dπ y dN (0, C µ ) (X) ∝ exp -Φ(X) - 1 2σ 2 ∥AX -y∥ 2 .(2)
The goal of infinite-dimensional Bayesian inference is to design sampling methods for π y whose performance remains stable as the underlying discretization is refined. To this end, we study a widely used sampler-a Langevin-type diffusion driven by score-based generative priors-this time formulated directly in infinite dimensions rather than in the usual finite-dimensional setting. In particular, we consider the continuous-time SDE
dX t = S θ (X t , τ ; µ)dt + C∇ X log ρ(y -AX t )dt + √ 2CdW t ,(3)
where ρ is the noise density, C acts as a preconditioner, ∇ X denotes the Fréchet derivative with respect to X, W t is a Wiener process on H, and S θ (X t , τ ; µ) is a neural network approximation of the score function
S(X, τ ; µ) = -(1 -e -τ ) -1 (X -e -τ /2 E[X 0 |X τ = X]),(4)
which corresponds to the drift term in the time-reversed SDE of the Hilbert-space-valued forward diffusion
dX τ = - 1 2 X τ dτ + √ CdW τ , X 0 ∼ µ.(5)
There are two important aspects to note here. First, both the Langevin SDE (3) and the forward diffusion (5) are driven by a C-Wiener process, where C is trace-class, which is crucial for ensuring that the samples are supported on the Hilbert space. Most of the technical difficulties in infinite dimensions arise from this. Second, although the score is often expressed as ∇ log p τ in finitedimensional settings, the density p τ is not defined in infinite dimensions, since a Lebesgue reference measure does not exist. For this reason, in the following we adopt the conditional expectation representation of the score-or, more precisely, an equivalent formulation derived from it, as stated in the next proposition, which was first proved in [49]. Proposition 2.1. The score (4) can be written as
S(X, τ ; µ) = -e τ /2 E[C(C µ C -1 τ ) -1 ∇Φ(X 0 ) | X τ = X] -CC -1 τ X,(6)
where C τ = e -τ C µ + (1 -e -τ )C.
The idea behind samplers like (3) is simple yet powerful. By training S θ (X, τ ; µ) to approximate S(X, τ ; µ), one can effectively learn potentially complex priors µ-since, once S θ (X, τ ; µ) is known, one can sample from µ by simulating the backward-in-time dynamics-and then incorporate such priors within a Langevin sampling scheme. What remains less understood, however, is how this approach extends to the infinite-dimensional setting, particularly in relation to the error introduced by approximating the score and whether the sampler remains stable. In the sections that follow, we address this gap-we prove convergence of (3) to the correct posterior and derive error bounds, along with conditions ensuring a globally bounded convergence error. We also elucidate the role of the preconditioner C. Our analysis is divided into two parts: one addressing the case of Gaussian priors, and the other the non-Gaussian case.
this section cite: ['b48']

Section: Error Analysis in the Gaussian Setting
We begin our analysis of the continous-time Langevin SDE (3) in the infinite-dimensional setting by examining the case where the prior of X 0 is a Gaussian measure. While this case may seem to defeat the purpose of using a score-based generative model to learn a simple prior, it provides illuminating insights, as it allows us to detail the impact of the score approximation error on the stationary distribution of (3), offers a clear interpretation of the infinite-dimensional difficulties, and paves the way for the derivation of an explicit form of the optimal preconditioner (Section 4).
We assume in this section that Φ = 0. The posterior (2) is Gaussian:
π y = N C -1 µ + σ -2 A ⊤ A -1 σ -2 A ⊤ y, C -1 µ + σ -2 A ⊤ A -1 .
We also assume that both C and C µ are diagonal in the basis (v j ), with eigenvalues (λ j ) and (µ j ), respectively. We can make a few remarks: • In the (v j ) basis, the posterior decomposes into a Gaussian π N y on the span of the first N observed modes and a product of marginal Gaussian over the unobserved modes j > N .
• For the observed modes-i.e., those j ≤ N influenced by the data through the forward operator A-the distribution is
π N y = N C -1 µ,N + σ -2 A ⊤ N A N -1 σ -2 A ⊤ N y, C -1 µ,N + σ -2 A ⊤ N A N -1
, with C µ,N = Diag 1≤j≤N (µ j ) .
• For the unobserved modes j > N , which lie in the nullspace of A, the posterior coincides with the prior: π (j) y = N (0, µ j ).
• The score function is S(X, τ ; µ) =j s j (τ ; µ)X (j) v j , with s j (τ ; µ) = λj e -τ µj +(1-e -τ )λj . The block diagonalization of the system by (v j ) justifies the following assumption on the form of the score approximation error.
this section cite: []

Section: Assumption 1.
We consider an approximate score S θ (X, τ ; µ) such that
S(X, τ ; µ) -S θ (X, τ ; µ), v j = ε a j (τ )X (j) + ε b j (τ ).
Define X N = N j=1 X (j) v j and similarly let W N t denote the projection of W t onto the first N modes. By Assumption 1, for the observed modes j ≤ N , the preconditioned Langevin dynamics (3) become
dX N t = -Diag 1≤j≤N (s j (τ ; µ)) + Diag 1≤j≤N ε a j (τ ) + C N σ -2 A ⊤ N A N X N t dt + C N σ -2 A ⊤ N y -Diag 1≤j≤N ε b j (τ ) dt + 2C N dW N t ,
with C N = Diag 1≤j≤N (λ j ). For the unobserved modes j > N , we have
dX (j) t = -s j (τ ; µ) + ε a j (τ ) X (j) t dt -ε b j (τ )dt + 2λ j dW (j)
t . We are now ready to derive the stationary distribution of the continuous-time SDE (3). The following proposition makes explicit the dependence on the score approximation error; its proof is included in Appendix A.1. (v j (τ )). For the observed modes j ≤ N , we have
vN (τ ) = C -1 N Diag 1≤j≤N (s j (τ ; µ)) + σ -2 A ⊤ N A N + C -1 N Diag 1≤j≤N ε a j (τ ) -1 ,(7)
mN (τ ) = vN (τ ) σ -2 A ⊤ N y -C -1 N Diag 1≤j≤N ε b j (τ ) ,(8)
while for the unobserved modes j > N , we have
vj (τ ) = λ -1 j s j (τ ; µ) + λ -1 j ε a j (τ ) -1 , mj (τ ) = -v j (τ )λ -1 j ε b j (τ ).(9)
Based on Proposition 3.1, we make a few comments:
• If we have access to the perfect score, that is, ε a j = ε b j = 0 for all j, then
m(τ ) = C -1 τ + σ -2 A ⊤ A -1 σ -2 A ⊤ y τ →0 → C -1 µ + σ -2 A ⊤ A -1 σ -2 A ⊤ y, v(τ ) = C -1 τ + σ -2 A ⊤ A -1 τ →0 → C -1 µ + σ -2 A ⊤ A -1 .
That is, we recover the posterior π y given the data. It does not depend on the preconditioner C.
• The error ε a j can have an impact on the stationary distribution of X
t , but as long as it is smaller than λ j /µ j (i.e., the relative error in the approximation of the score is small), the impact is small.
• The error ε b j can induce a bias. The bias can be large because the mean of the j-th mode marginal of π(j) y is amplified by λ -1 j . The preconditioner cannot prevent from this bias.
We can make our analysis more quantitative by presenting mode-by-mode and global convergence error estimates for the preconditioned Langevin sampler in the Gaussian setting. To simplify the discussion, the following theorem is stated by assuming that A ⊤ N A N is diagonal. Theorem 3.1. We define p j = λ j /µ j for all j. Let π(j) y and π (j) y denote the j-th mode marginals of the approximate and true posterior distributions, πy and π y , respectively. Suppose that p -1 j ε a j (τ ) = O(τ ), λ -1 j ε b j (τ ) = O(1). Then, for j ≤ N , the Kullback-Leibler divergence satisfies
D KL π(j) y π (j) y = 1 2 λ -2 j ε b j (τ ) 2 - λ -1 j ε b j (τ ) 1 + σ -2 µ j (A ⊤ N A N ) jj σ -2 (A ⊤ N y) j -λ -1 j ε b j (τ ) (p j -1)τ -p -1 j ε a j (τ ) + O(τ 2 ). (10
)
For j > N , we have D KL π(j
) y π (j) y = λ -2 j ε b j (τ ) 2 1 2 + (p j -1)τ -p -1 j ε a j (τ ) + O(τ 2 ).
Proof. The proof relies on Proposition 3.1 and the fact that the j-th mode marginals π(j) y and π (j) y are Gaussian, N ( mj (τ ), vj (τ )) and N (m j , v j ), respectively, hence the Kullback-Leibler divergence has an explicit form and standard perturbation arguments lead to the desired estimates. Full details are provided in Appendix A.2.
Remark 3.1. Note that Theorem 3.1 can provide a set of sufficient conditions that ensure that the global convergence error of the sampler is bounded in infinite dimensions: j λ -1 j ε b j (τ ) < ∞, p -1 j ε a j (τ ) < C 1 , and (A ⊤ N y) j < C 2 , where C 1 , C 2 do not depend on j.
this section cite: []

Section: The Essence of Preconditioning
We now elucidate the role of the preconditioner C in the infinite-dimensional Gaussian setting introduced in the previous section. We begin with two preliminary remarks:
• In our analysis, C first appears in the forward diffusion (5), whose time-reversal learns the prior, and must be carried through the Langevin sampler (3) to target the correct posterior.
• C cannot be the identity: it must be trace-class to keep the diffusion well-posed and to stabilize the Langevin updates across all modes. Indeed, if C = Diag(λ j ), the drift in the j-th mode contains the factor λ j [e -τ µ j + (1 -e -τ )λ j ] -1 , which, unless λ j decays sufficiently fast, blows up like µ -1 j as j → ∞, making the sampler unstable at fine scales. This is a consequence of the infinite-dimensional setting, where C µ must be trace-class. Since the preconditioner plays a role in the rate of convergence across all posterior modes, it is natural to ask whether there exists a C that ensures a uniform convergence rate for the Langevin sampler. To this aim, in the next propositionwe derive the mean reversion rate κ of the preconditioned Langevin dynamics (3); the proof is given in Appendix B.1. Proposition 4.1. Assume that A ⊤ N A N is diagonal. For the observed modes j ≤ N , the mean reversion rate is
κ (j) = λ j e -τ µ j + (1 -e -τ )λ j -1 + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) ,(11)
while for the unobserved modes j > N ,
κ (j) = λ j [e -τ µ j + (1 -e -τ )λ j ] -1 + λ -1 j ε a j (τ ) .
We can make a few comments:
• For the unobserved modes j > N , the convergence rate is λ j [e -τ µ j + (1 -e -τ )λ j ] -1 (≃ λ j /µ j for small τ ) when the error ε a j is negligible, and therefore we should choose λ j = µ j for all j to get a convergence uniform in j, that is to say, C = C µ .
• For the observed modes j ≤ N , the convergence rates for those modes such that
µ j ≪ σ 2 /(A ⊤ N A N ) jj (or (A ⊤ N A N ) jj = 0) are λ j /µ j , whereas for modes such that µ j ≫ σ 2 /(A ⊤ N A N ) jj the convergence rates are λ j σ -2 (A ⊤ N A N ) jj . We should then choose λ j = [µ -1 j + σ -2 (A ⊤ N A N ) jj ] -1 , or equivalently C = [C -1 µ + σ -2 A ⊤ A] -1
. We now refine our analysis of the preconditioner by incorporating a first-order correction that accounts for the score approximation error at small τ , the regime in which the score is typically learned. The proof of the following theorem relies on a straightforward perturbation argument; full details are given in Appendix B.2. Theorem 4.1. In addition to Assumption 1, we further suppose that A ⊤ N A N is diagonal, and that ε a j (τ ) = ε a j τ + O(τ 2 ). Under these conditions, the optimal preconditioner C is also diagonal in the basis (v j ), with eigenvalues that admit the expansion λ j = λ (0) j + λ (1) j τ + O(τ 2 ). For the observed modes j ≤ N , we have
λ (0) j = µ -1 j + σ -2 (A ⊤ N A N ) jj -1 , λ(1)
j = λ (0) j 3 µ -2 j -λ (0) j 2 µ -1 j -ε a j λ (0) j .(12)
For the unobserved modes j > N , we have λ
j = µ j , λ(0)
j = -µ j ε a j .(1)
Based on Theorem 4.1, we make a few comments:
• To compute the preconditioner C, one would need information on A ⊤ A, σ, C µ , and the score approximation error. Our analysis, however, suggests a simple and practical choice: take C as close as possible to the prior covariance C µ . For higher-order modes, the leading-order term of the preconditioner coincides with C µ , and this approximation is particularly justified when the prior decays quickly, so that µ -1 j ≫ σ -2 (A ⊤ N A N ) for the low-order modes. Any available knowledge of the posterior covariance or score error can then be used to refine this first approximation.
• This is not the first occurrence in the literature of an optimal preconditioner for diffusion models in infinite dimensions. For example, while analyzing the convergence error of time-reversed SDE dynamics in infinite dimensions, Pidstrigach et al. [49] derived a similar result for the optimal C by minimizing the Wasserstein-2 distance between the true data distribution and the learned sample distribution. Interestingly, assuming no data model and a perfect score, our framework yields the same optimal C, with a few caveats: in our case, the preconditioner arises directly from the mean-reversion rate of the Langevin dynamics. Hence, the optimal covariance we identify does not merely minimize an upper bound: it represents, under the stated assumptions, the best achievable choice in practice for ensuring a uniform rate of convergence across all modes.
this section cite: ['b48']

Section: Non-Gaussian Sampling
We can generalize the results of Sections 3 and 4 by considering the case of a general class of prior measures µ assumed to be absolutely continuous with respect to a Gaussian reference measure N (0, C µ ) with density proportional to exp(-Φ). We present the main ideas here, relegating the more quantitative results and proofs in Appendix C.
To reproduce the approach of the Gaussian setting, one first needs to diagonalize the Langevin SDE system, which in turn requires diagonalizing the score function S(X, τ ; µ). Proposition 5.1. We assume that C and C µ have the same basis of eigenfunctions (v j ) and we define X (j) = ⟨X, v j ⟩. We assume that Φ(X) = j ϕ j (X (j) ). The score function (6) can be written as S(X, τ ; µ) = j S (j) (X (j) , τ ; µ)v j , where
S (j) (X (j) , τ ; µ) = -λ j ∂ j φj (X (j) , τ ) -s j (τ, µ)X (j) ,(13)
with φj (X (j) , τ
) = -log E exp(-ϕ j ( X (j) 0 )) | X (j) τ = X (j)
, and
X (j) 0 X (j) τ ∼ N 0, µ j e -τ /2 µ j e -τ /2 µ j e -τ µ j + (1 -e -τ )λ j .(14)
We assume a more general form of the score approximation error. Assumption 2. We consider an approximate score S θ (X, τ ; µ) such that
S(X, τ ; µ) -S θ (X, τ ; µ), v j = ε a j (τ ) X (j) + ∂ j ϕ j (X (j) ) + ε b j (τ ).
With the learned score, the preconditioned Langevin SDE for the observed modes j ≤ N becomes
dX N t = -M N X N t dt + b N dt + 2C N dW N t ,where
M N = Diag 1≤j≤N (s j (τ ; µ)) + C N σ -2 A ⊤ N A N + Diag 1≤j≤N ε a j (τ ) , b N =C N σ -2 A ⊤ N y -C N Diag 1≤j≤N ∂ j φj -Diag 1≤j≤N ε a j (τ )∂ j ϕ j (X (j) t ) -Diag 1≤j≤N ε b j (τ ) .
The SDE for the unobserved modes j > N can be obtained by taking A N = 0 above.
The stationary distribution of the preconditioned Langevin SDE is derived in the following proposition, which makes explicit the dependence on the score approximation error. Proposition 5.2. Let Assumption 2 hold true. Under the hypotheses of the previous proposition, the preconditioned Langevin with approximate score in the drift term has πy as its stationary distribution. It is absolutely continuous with respect to N ( m(τ ), v(τ )), and is given by the density
dπ y dN ( m(τ ), v(τ )) (X, τ ) ∝ exp -Φ(X, τ ) .(15)
For the observed modes j ≤ N , the negative log-density is
ΦN (X N , τ ) = N j=1 φj (X (j) , τ ) + λ -1 j ε a j (τ )ϕ j (X (j)
t ) , the covariance vN (τ ) and mean mN (τ ) are given by (7)(8). For the unobserved modes j > N , the negative log-density is Φ
(j) (X (j) , τ ) = φj (X (j) , τ ) + λ -1 j ε a j (τ )ϕ j (X (j)
t ), the covariance v(j) (τ ) and mean m(j) (τ ) are given by (9).
The interested reader can find in Appendix C a quantitative analysis of this general case, including Theorem C.1, which provides an error analysis analogous to Theorem 3.1 for the Gaussian case. Here we make a few qualitative comments:
• If ε a j = ε b j = 0 for all j, then Φ(X, τ ) τ →0 → j ϕ j (X (j) ), v(τ ) τ →0 → C -1 µ + σ -2 A ⊤ A -1 , m(τ ) τ →0 → C -1 µ + σ -2 A ⊤ A -1 σ -2 A ⊤ y
, that is to say, we get the posterior given the data.
• The preconditioner influences the convergence rate. Consider the case in which ϕ j is convex, i.e., ∂ 2 j ϕ j ≥ C ϕj > 0. In this case, for the unobserved modes j > N , the convergence rate of the j-th mode is ≃ λ j [µ -1 j + C ϕj ] for small τ , assuming the error ε a j is negligible. To achieve a convergence rate that is uniform in j, we should then choose λ j = [µ -1 j + C ϕj ] -1 . For the observed modes j ≤ N , and assuming A ⊤ N A N is diagonal for simplicity, we should instead choose
λ j = [µ -1 j + σ -2 (A ⊤ N A N ) jj + C ϕj ] -1 .
• If the error |ε a j | ≪ λ j , its impact on the stationary distribution of X (j) t is small. Like the Gaussian case, the error ε b j can induce a bias. It can be large since λ -1 j → +∞ as j → +∞.
this section cite: ['b6', 'b7', 'b8']

Section: Illustrations
We verify our theory by applying the preconditioned Langevin dynamics with SGM (3) to two linear inverse problems: one based on the Karhunen-Loève (KL) expansion of the Brownian sheet [50], and the other one on an inverse source problem for the heat equation [3]. Both examples are consistent with the theory of the paper. Implementation and further details are provided in Appendix D.
this section cite: ['b49', 'b2']

Section: Brownian sheet.
We illustrate the discretization-invariance property of our approach on the Brownian sheet, represented by its truncated KL expansion B N (x) = N j,k=1 ϕ j,k (x)η j,k , x ∈ [0, 1] 2 , where η j,k ∼ N (0, µ j,k ) and (ϕ j,k , µ j,k ) are the KL eigenpairs. For M ≤ N , the inverse problem consists of reconstructing the KL coefficients and Brownian sheet from noisy data y j,k = ηj,k + ε j,k , j, k ≤ M , with ηj,k ∼ N (0, µ j,k ), ε j,k ∼ N (0, 0.01 2 ) (i.e. A jk = 1 if j = k ≤ M and 0 otherwise). Figure 2 shows the robustness of our approach with respect to the number of modes M 2 . Heat equation. We verify the benefits of the optimal preconditioner C from Theorem 4.1 by considering the ill-posed inverse problem of recovering the initial condition u(x, 0), x ∈ [0, 1] 2 , of the heat equation from noisy observations of the solution u(x, T ) at time T = 0.1. Expanding in the eigenpairs (ψ j,k , ζ j,k ) of the Dirichlet Laplacian, one finds that u(x, t) = j,k e -ζ j,k t g j,k ψ j,k (x), where g j,k = ⟨u(•, 0), ψ j,k ⟩. The inverse problem diagonalizes: we observe y j,k = e -ζ j,k T g j,k +ε j,k , j, k ≤ M , with g j,k ∼ N (0, e -βζ j,k ), β = 0.1, ε j,k ∼ N (0, 0.005 2 ). In Figure 3 we compare reconstructions using Langevin dynamics preconditioned with the optimal C (3rd column) and vanilla Langevin (4th column). Both samplers use a score perturbed by a relative error ε a j ∼ N (0, 0.1 2 ) scaled by a small τ , as assumed in Theorem 4.1. The results support our theory: (i) the preconditioned sampler is robust to score approximation error, as expected from the design of C (Theorem 4.1); ii) as shown in the autocorrelation plot in Figure 4 (corresponding to the top row of Figure 3), the modes converge faster and more uniformly than with the vanilla dynamics, since C targets the optimal mean reversion rates (Proposition 4.1) ; iii) vanilla Langevin deteriorates as the score error increases (Figure 3) due to amplification at fine scales, reproducing the pathological behavior seen in Figure 1.
this section cite: []

Section: Discussion and Future Work
We studied a popular sampler-a Langevin-type diffusion driven by score-based generative priorsdirectly in the infinite-dimensional Bayesian setting, rather than in the usual finite-dimensional one. We showed that naïvely applying standard techniques in infinite dimensions leads to several issues. To ensure provable posterior sampling and discretization-invariance, our analysis shows that  preconditioning the vanilla Langevin is necessary. We prove detailed convergence error estimates and the existence (and form) of an optimal preconditioner-depending on both the forward map A and the score error-that yields uniform convergence across all modes.
As is standard in infinite-dimensional analysis, our results rely on some simplifying assumptions: finite-dimensional data, and co-diagonalizability of the prior and the diffusion's noise covariance. In some parts, we also assumed that A ⊤ A is diagonal, a common assumption in the theory of linear Bayesian inverse problems [3,4]. This is not merely a technical convenience: in many classical linear inverse problems, such as the heat equation, tomography, or inverse scattering for Schrödinger-type operators under the Born approximation, the forward operator A is compact, and hence one can always find a basis in which A ⊤ A is diagonal.
Nevertheless, the main conclusions of our analysis remain valid even without the diagonalization assumption. For example, the asymptotic expansion in Eq. ( 10) of Theorem 3.1 can be extended to non-diagonal by replacing scalar expansions with their corresponding matrix-series counterparts, while the arguments for the higher-order modes remains the same. Likewise, in Section 4, one can verify that under a perfect score function the optimal preconditioner still takes the form C = [C -1 µ + σ -2 A ⊤ A] -1 . Several open questions remain. In particular, how do these results extend to nonlinear inverse problems? Extending the analysis of [21] to determine an optimal preconditioner for nonlinear inverse problems represents an important direction for future work.
this section cite: ['b2', 'b3', 'b20']

Section: A Proofs of Section 3 A.1 Proof of Proposition 3.1
By Assumption 1, for the observed modes j ≤ N , the preconditioned Langevin dynamics (3) reduces to the SDE
dX N t = -Diag 1≤j≤N (s j (τ ; µ)) + Diag 1≤j≤N ε a j (τ ) + C N σ -2 A ⊤ N A N X N t dt + C N σ -2 A ⊤ N y -Diag 1≤j≤N ε b j (τ ) dt + 2C N dW N t ,(16)
with C N = Diag 1≤j≤N (λ j ). For each unobserved mode j > N , we have
dX (j) t = -s j (τ ; µ) + ε a j (τ ) X (j) t dt -ε b j (τ )dt + 2λ j dW (j) t .(17)
Both ( 16) and ( 17) are Ornstein-Uhlenbeck (OU) processes. In particular, X N t t→∞ → X N ∞ in distribution, where the distribution of X N ∞ is the stationary distribution of ( 16):
X N ∞ ∼ N mN (τ ), vN (τ ) , with mN (τ ) = Diag 1≤j≤N (s j (τ ; µ)) + Diag 1≤j≤N ε a j (τ ) + C N σ -2 A ⊤ N A N -1 C N σ -2 A ⊤ N y -Diag 1≤j≤N ε b j (τ ) = C -1 N Diag 1≤j≤N (s j (τ ; µ))+C -1 N Diag 1≤j≤N ε a j (τ ) +σ -2 A ⊤ N A N -1 σ -2 A ⊤ N y-C -1 N Diag 1≤j≤N ε b j (τ ) ,
and vN (τ ) is such that it solves the Lyapunov equation
Diag 1≤j≤N (s j (τ ; µ)) + Diag 1≤j≤N ε a j (τ ) + C N σ -2 A ⊤ N A N vN (τ ) + vN (τ ) Diag 1≤j≤N (s j (τ ; µ)) + Diag 1≤j≤N ε a j (τ ) + C N σ -2 A ⊤ N A N ⊤ = 2C N .
Then
vN (τ ) = C -1 N Diag 1≤j≤N (s j (τ ; µ)) + C -1 N Diag 1≤j≤N ε a j (τ ) + σ -2 A ⊤ N A N -1 , since Diag 1≤j≤N (s j (τ ; µ)) + Diag 1≤j≤N ε a j (τ ) + C N σ -2 A ⊤ N A N vN (τ ) = C N vN (τ ) -1 vN (τ ) = C N ,and
vN (τ ) C -1 N Diag 1≤j≤N (s j (τ ; µ)) + C -1 N Diag 1≤j≤N ε a j (τ ) + σ -2 A ⊤ N A N C N = vN (τ ) vN (τ ) -1 C N = C N .
For each j > N , (17) is a one-dimensional OU process with rate s j (τ ; µ) + ε a j (τ ), mean shift -[s j (τ ; µ) + ε a j (τ )] -1 ε b j (τ ), and noise 2λ j . Hence X
(j) t t→∞ → X (j)
∞ in distribution, where the distribution of X (j) ∞ is the stationary distribution of ( 17)
X (j) ∞ ∼ N - ε b j (τ ) s j (τ ; µ) + ε a j (τ ) , λ j s j (τ ; µ) + ε a j (τ )
.
These results are valid as soon as s j (τ ; µ) + ε a j (τ ) is a positive number for all j.
this section cite: []

Section: A.2 Proof of Theorem 3.1
For each mode j, define μj = µ j e -τ + (1 -e -τ )p j .
The proof can be divided into two cases: one for the observed modes j ≤ N , and one for the unobserved modes j > N . Since the KL divergence for the unobserved modes can be obtained by taking A N = 0 in the expression for the observed modes, we focus only on the latter.
The marginal distributions of the j-th mode for the approximate and true posterior, for j ≤ N , are given respectively by
π(j) y = N 1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) -1 σ -2 (A ⊤ N y) j -λ -1 j ε b j (τ ) , 1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) -1 ,and
π (j) y = N 1 µ j + σ -2 (A ⊤ N A N ) jj -1 σ -2 (A ⊤ N y) j , 1 µ j + σ -2 (A ⊤ N A N ) jj -1 .
The KL divergence between these two Gaussian distributions admits the explicit formula
D KL π(j) y π (j) y = log 1 µ j + σ -2 (A ⊤ N A N ) jj -1 1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) - 1 2 + 1 2 1 µ j + σ -2 (A ⊤ N A N ) jj 2 1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) -2 + 1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) -1 σ -2 (A ⊤ N y) j -λ -1 j ε b j (τ ) - 1 µ j + σ -2 (A ⊤ N A N ) jj -1 σ -2 (A ⊤ N y) j 2    .
We study its limiting behavior as τ → 0. From the first term, we derive
log 1 µ j + σ -2 (A ⊤ N A N ) jj -1 1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) = log 1 + σ -2 µ j (A ⊤ N A N ) jj -1 1 e -τ + (1 -e -τ )p j + σ -2 µ j (A ⊤ N A N ) jj + p -1 j ε a j (τ ) = log 1 + 1 1 + σ -2 µ j (A ⊤ N A N ) jj (1 -e -τ ) + (e -τ -1)p j e -τ + (1 -e -τ )p j + p -1 j ε a j (τ ) = log 1 + 1 1 + σ -2 µ j (A ⊤ N A N ) jj -τ (p j -1) + p -1 j ε a j (τ ) + O(τ 2 ) = 1 1 + σ -2 µ j (A ⊤ N A N ) jj -τ (p j -1) + p -1 j ε a j (τ ) + O(τ 2 ).(18)
We now consider the other terms of the KL divergence. First, we derive
1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) -1 × σ -2 (A ⊤ N y) j -λ -1 j ε b j (τ ) - 1 µ j + σ -2 (A ⊤ N A N ) jj -1 σ -2 (A ⊤ N y) j = µ j 1 + σ -2 µ j (A ⊤ N A N ) jj 1 + 1 1 + σ -2 µ j (A ⊤ N A N ) jj (1 -e -τ ) + (e -τ -1)p j e -τ + (1 -e -τ )p j + p -1 j ε a j (τ ) -1 σ -2 (A ⊤ N y) j -λ -1 j ε b j (τ ) -σ -2 (A ⊤ N y) j = µ j 1 + σ -2 µ j (A ⊤ N A N ) jj 1 + 1 1 + σ -2 µ j (A ⊤ N A N ) jj -τ (p j -1) + p -1 j ε a j (τ ) + O(τ 2 ) -1 × σ -2 (A ⊤ N y) j -λ -1 j ε b j (τ ) -σ -2 (A ⊤ N y) j = µ j 1 + σ -2 µ j (A ⊤ N A N ) jj 1 + 1 1 + σ -2 µ j (A ⊤ N A N ) jj τ (p j -1) -p -1 j ε a j (τ ) + O(τ 2 ) × σ -2 (A ⊤ N y) j -λ -1 j ε b j (τ ) -σ -2 (A ⊤ N y) j = µ j 1 + σ -2 µ j (A ⊤ N A N ) jj -λ -1 j ε b j (τ ) - λ -1 j ε b j (τ ) 1 + σ -2 µ j (A ⊤ N A N ) jj τ (p j -1) -p -1 j ε a j (τ ) + σ -2 (A ⊤ N y) j 1 + σ -2 µ j (A ⊤ N A N ) jj τ (p j -1) -p -1 j ε a j (τ ) + O(τ 2 ).
Taking the square yields
1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) -1 × σ -2 (A ⊤ N y) j + λ -1 j ε b j (τ ) - 1 µ j + σ -2 (A ⊤ N A N ) jj -1 σ -2 (A ⊤ N y) j 2 = µ 2 j (1 + σ -2 µ j (A ⊤ N A N ) jj ) 2 λ -2 j ε b j (τ ) 2 + 2λ -1 j ε b j (τ ) 1 + σ -2 µ j (A ⊤ N A N ) jj λ -1 j ε b j (τ ) -σ -2 (A ⊤ N y) j τ (p j -1) -p -1 j ε a j (τ ) + O(τ 2 ).(19)
Next, we consider
1 μj + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) -2 = µ 2 j (1 + σ -2 µ j (A ⊤ N A N ) jj ) 2 1 + 1 1 + σ -2 µ j (A ⊤ N A N ) jj -τ (p j -1) + p -1 j ε a j (τ ) + O(τ 2 ) -2 = µ 2 j (1 + σ -2 µ j (A ⊤ N A N ) jj ) 2 1 + 2 1 + σ -2 µ j (A ⊤ N A N ) jj τ (p j -1) -p -1 j ε a j (τ ) + O(τ 2 ) .(20)
Putting ( 18), (19), and (20) together, we obtain
D KL π(j) y π (j) y = 1 2 λ -2 j ε b j (τ ) 2 - λ -1 j ε b j (τ ) 1 + σ -2 µ j (A ⊤ N A N ) jj σ -2 (A ⊤ N y) j -λ -1 j ε b j (τ ) τ (p j -1) -p -1 j ε a j (τ ) + O(τ 2 ).
this section cite: ['b18']

Section: B Proofs of Section 4 B.1 Proof of Proposition 4.1
If we assume that A ⊤ N A N is diagonal in (v j ), for the observed modes j ≤ N the preconditioned Langevin dynamics 3 becomes
dX (j) t = -s j (τ ; µ) + ε a j (τ ) + λ j σ -2 (A ⊤ N A N ) jj X (j) t dt -λ j σ -2 (A ⊤ N y) j -ε b j (τ ) dt + 2λ j dW (j) t ,(21)
while for each unobserved mode j > N , one obtains
dX (j) t = -s j (τ ; µ) + ε a j (τ ) X (j) t dt -ε b j (τ )dt + 2λ j dW (j) t . (22
) Let m (j) (t) = E[X (j) t ].
For the observed modes j ≤ N , taking the expectation in the SDE above gives the linear ODEs
dm (j) dt = -s j (τ ; µ) + ε a j (τ ) + λ j σ -2 (A ⊤ N A N ) jj m (j) (t) + λ j σ -2 (A ⊤ N y) j -ε b j (τ )
while for each unobserved mode j > N one obtains
dm (j) dt = -s j (τ ; µ) + ε a j (τ ) m (j) (t) -ε b j (τ ).
Given m (j) (0) = m (j) 0 , both have unique solution. For j ≤ N ,
m (j) (t) = m (j) (0) - λ j σ -2 (A ⊤ N y) j -ε b j (τ ) s j (τ ; µ) + ε a j (τ ) + λ j σ -2 (A ⊤ N A N ) jj e -[sj (τ ;µ)+ε a j (τ )+λj σ -2 (A ⊤ N A N )jj ]t + λ j σ -2 (A ⊤ N y) j -ε b j (τ ) s j (τ ; µ) + ε a j (τ ) + λ j σ -2 (A ⊤ N A N ) jj
, which for t → ∞ decays exponentially fast to the mean
λ j σ -2 (A ⊤ N y) j -ε b j (τ ) s j (τ ; µ) + ε a j (τ ) + λ j σ -2 (A ⊤ N A N ) jj , withrate
κ (j) = s j (τ ; µ) + ε a j (τ ) + λ j σ -2 (A ⊤ N A N ) jj = λ j e -τ µ j + (1 -e -τ )λ j -1 + σ -2 (A ⊤ N A N ) jj + λ -1 j ε a j (τ ) .
For the unobserved modes j > N ,
m (j) (t) = m (j) (0) + ε b j (τ ) s j (τ ; µ) + ε a j (τ ) e -[sj (τ ;µ)+ε a j (τ )]t - ε b j (τ ) s j (τ ; µ) + ε a j (τ )
, which converge to the mean
- ε b j (τ ) s j (τ ; µ) + ε a j (τ )
, with rate
κ (j) = s j (τ ; µ) + ε a j (τ ) = λ j e -τ µ j + (1 -e -τ )λ j -1 + λ -1 j ε a j (τ ) .
this section cite: []

Section: B.2 Proof of Theorem 4.1
We consider only the case of the observed modes j ≤ N , since the case for the unobserved modes j > N follows directly by setting A N = 0.
By Proposition 4.1, ensuring uniform convergence rate for (3) using an approximate score functionas described in Assumption 1-amounts to solving the equation
λ j μj + λ j σ -2 (A ⊤ N A N ) jj + ε a j (τ ) = 1,(23)
where μj = µ j [e -τ + (1 -e -τ )p j ].
Assume the expansions
λ j = λ (0) j + λ (1
)
j τ + O(τ 2 ), ε a j (τ ) = ε a j τ + O(τ 2
). Then we compute
μj = µ j + µ j (p j -1)τ + O(τ 2 ) = µ j + λ (0) j τ -µ j τ + O(τ 2 ),
using that p j = λ j /µ j Substituting into (23), we obtain that λ (0) j and λ (1) j must satisfy
µ -1 j (λ (0
) j + λ (0) j [1 -µ -1 j λ (0) j ]τ + λ (1
) j τ ) + σ -2 (A ⊤ N A N ) jj (λ (0
) j + λ (1
) j τ ) + ε a j τ + O(τ 2 ) = 1.
Rearranging the terms, we get λ
(0) j µ -1 j + λ (0) j σ -2 (A ⊤ N A N ) jj = 1, which gives λ (0
) j = µ -1 j + σ -2 (A ⊤ N A N ) jj -1 ,and
µ -1 j -λ (0) j µ -1 j λ (0) j -1 + λ (1) j + σ -2 (A ⊤ N A N ) jj λ (1
) j + ε a j = 0, yielding λ (1
)
j = λ (0) j µ -1 j λ (0) j µ -1 j λ (0) j -1 -ε a j .
this section cite: ['b22']

Section: C Non-Gaussian Sampling: Technical Details

this section cite: []

Section: C.1 Proof of Proposition 5.1
By Φ(X) = j ϕ j (X (j) ), the prior µ has Radon-Nikodym derivative with respect to the Gaussian N (0, C µ ) given by dµ dN (0, C µ ) (X) ∝ j exp -ϕ j (X (j) ) .
Since C and C µ are both diagonalized by the same basis (v j ), the prior factorizes as a product of independent one-dimensional marginals in the coordinates X (j) :
dµ dN (0, C µ ) (X) = j dµ (j) dN (0, µ j ) (X (j) ),
where
dµ (j) dN (0, µ j ) (X (j) ) ∝ exp -ϕ j (X (j) ) .
We can then work mode by mode. For each j, define the one-dimensional OU process
X (j) 0 ∼ N (0, µ j ), X (j) τ = e -τ /2 X (j) 0 + √ 1 -e -τ ξ (j) , with ξ (j) ∼ N (0, λ j ) independent of X (j) 0 . Notice that X (j) 0 X (j) τ ∼ N 0, µ j e -τ /2 µ j e -τ /2 µ j e -τ µ j + (1 -e -τ )λ j . (24
)
The OU transition kernel is
p( X (j) τ = x τ | X (j) 0 = x 0 ) = N (e -τ /2 x 0 , (1 -e -τ )λ j )(x τ
), where N (µ, σ 2 )(x) is the density at x of the normal distribution with mean µ and variance σ 2 . We push forward the prior e -ϕj N (0, µ j ) through the OU kernel. Mode by mode, its density is
µ (j) τ (x τ ) ∝ exp(-ϕ j (x 0 ))N (0, µ j )(x 0 )p(x τ | x 0 )dx 0 = exp(-ϕ j (x 0 ))p 0,τ (x 0 , x τ )dx 0 ,
where p0,τ (x 0 , x τ ) denotes the joint density of ( X
(j) 0 , X(j)
τ ). Let μj = e -τ µ j + (1 -e -τ )λ j . Dividing by the marginal Gaussian density of X (j) τ , we get
µ (j) τ (x τ ) N (0, μj )(x τ ) ∝ exp(-ϕ j (x 0 )) p0,τ (x 0 , x τ ) pτ (x τ ) dx 0 = exp(-ϕ j (x 0 ))p(x 0 | x τ )dx 0 = E[exp(-ϕ j ( X (j) 0 )) | X (j) τ = x τ ]. Since S (j) (x τ , τ ; µ j ) = λ j ∂ j log µ (j) τ (xτ ) N (0,μj ) + λ j N (0, μj )(x τ ), we obtain S (j) (x τ , τ ; µ) = λ j ∂ j log E[exp(-ϕ j ( X (j) 0 )) | X (j) τ = x τ ] - λ j e -τ µ j + (1 -e -τ )λ j X (j) .
this section cite: []

Section: C.2 Proof of Proposition 5.2
For each mode j, define j) . Under Assumption 2, the first N coordinates of the preconditioned Langevin dynamics (3) corresponding to the observed modes j ≤ N satisfy
φj (X (j) , τ ) = -log E exp(-ϕ j ( X (j) 0 )) | X (j) τ = X (
dX N t = -Diag 1≤j≤N (s j (τ ; µ)) + C N σ -2 A ⊤ N A N + Diag 1≤j≤N ε a j (τ ) X N t dt + C N σ -2 A ⊤ N y -C N Diag 1≤j≤N ∂ j φj (X (j) t ) -Diag 1≤j≤N ε a j (τ )∂ j ϕ j (X (j) t ) -Diag 1≤j≤N ε b j (τ ) dt + 2C N dW N t .(25)
The SDE for the unobserved modes j > N can be obtained by taking A N = 0 above:
dX (j) t = -s j (τ ; µ) + ε a j (τ ) X (j) t dt + -λ j ∂ j φj (X (j) t ) -ε a j (τ )∂ j ϕ j (X (j) t ) -ε b j (τ ) dt + 2λ j dW (j) t .(26)
Both ( 25) and ( 26) are preconditioned overdamped Langevin SDEs. In particular, one checks that (25) can be written as
dX N t = -C N ∇U N (X N t )dt + 2C N dW N t , where the potential U N is U N (X N ) = 1 2 X N ⊤ C -1 N Diag 1≤j≤N (s j (τ ; µ)) + σ -2 A ⊤ N A N + C -1 N Diag 1≤j≤N ε a j (τ ) X N -σ -2 A ⊤ N y -C -1 N Diag 1≤j≤N ε b j (τ ) X N + N j=1 φj (X (j) t ) -λ -1 j ε a j (τ )ϕ j (X (j) t ) .
Its stationary distribution is πN y , which is absolutely continuous with respect to the Lebesgue measure over R N :
dπ N y (X N ) dX N ∝ exp(-U N (X N )).
We split U N into quadratic and non-quadratic terms. Hence
dπ N y (X N ) dX N ∝ exp -ΦN (X N , τ ) N ( mN (τ ), vN (τ ))(X N ),
where N ( mN (τ ), vN (τ ))(X N ) is the density at X N of the multivariate Gaussian with mean mN (τ ) and covariance vN (τ ), with
ΦN (X N , τ ) = N j=1 φj (X (j) , τ ) + λ -1 j ε a j ϕ j (X (j) t ) , vN (τ ) = C -1 N Diag 1≤j≤N (s j (τ ; µ)) + σ -2 A ⊤ N A N + C -1 N Diag 1≤j≤N ε a j (τ ) -1 , mN (τ ) = vN (τ ) σ -2 A ⊤ N y -C -1 N Diag 1≤j≤N ε b j (τ ) .
By the same argument, for each j > N , the one-dimensional potential of ( 26) is
U j (X (j) ) = λ -1 j s j (τ ; µ) + λ -1 j ε a j (τ ) X (j) t 2 2 + λ -1 j ε b j (τ )X (j) t + φj (X (j) t ) + λ -1 j ε a j (τ )ϕ j (X (j) t ).
Its stationary distribution is therefore
dπ (j) y (X (j) ) dX (j) ∝ exp -Φ(j) (X (j) , τ ) N ( m(j) (τ ), v(j) (τ ))(X (j) ),
where
N ( m(j) (τ ), v(j) (τ ))(X (j)
) is the density at X (j) of the multivariate Gaussian with mean m(j) (τ ) and covariance v(j) (τ ), with
Φ(j) (X (j) , τ ) = φj (X (j) , τ ) + λ -1 j ε a j ϕ j (X (j) t ), v(j) (τ ) = λ -1 j s j (τ ; µ) + λ -1 j ε a j (τ ) -1 , m(j) (τ ) = -v (j) (τ )λ j -1 ε b j (τ ).
this section cite: []

Section: C.3 Error Analysis in the Non-Gaussian Setting
For the sake of completeness, we present a result analogous to Theorem 3.1 for the non-Gaussian case. As the interested reader will notice, the calculations are significantly more involved, but remain relatively straightforward. Theorem C.1. We assume ε a j (τ ) = O(τ ), ε b j (τ ) = O(1), and A N = Diag 1≤j≤N (A jj ). The Kullback-Leibler divergence between π(j) y and π
y is given by
D KL π(j) y π (j) y = B j (τ ) + E j (τ ),
where B j (τ ) is a bias term given by
B j (τ ) = -2λ -1 j ε b j (τ )E π(j) y [x] + λ -1 j p -1 j (ε b j (τ )) 2 + log e -ϕj (z)-1 2σ 2 [Ajj z-yj ] 2 N (0, µ j )(z)dz -log e -ϕj (z)-1 2σ 2 [Ajj z-yj ] 2 N (-p -1 j ε b j (τ ), µ j )(z)dz,and
E j (τ ) is an error term E j (τ ) = E(1)
j (τ )τ + E(2)
j (τ )λ -1 j ε a j (τ ) + O(τ 3/2 ),where
E (1) j (τ ) = E π(j) y x 2 µ j -λ -1 j p -1 j ε b j (τ ) 2 (1 -p j ) + 1 2 E π(j) y λ j ϕ ′ j (x) 2 -ϕ ′′ j (x) -ϕ ′ j (x)(1 -2p j )x -Z(ε b j (τ ), A jj , y j ) -1 e -ϕj (z)-1 2σ 2 [Ajj z-yj ] 2 N (-p -1 j ε b j (τ ), µ j )(z) × 1 2 λ j (ϕ ′ (z) 2 -ϕ ′′ j (z)) -ϕ ′ j (z)(1 -2p j )z + z + p -1 j ε b j (τ ) µ j + (z + p -1 j ε b j (τ )) 2 µ j - 1 2 (1 -p j ) dz,and
E (2
)
j (τ ) = E π(j) y x 2 -ϕ j (x) -p -2 j ε b j (τ ) 2 -Z(ε b j (τ ), A jj , y j ) -1 e -ϕj (z)-1 2σ 2 [Ajj z-yj ] 2 N (-p -1 j ε b j (τ ), µ j )(z) × -ϕ j (z) + z + p -1 j ε b j (τ ) + z + p -1 j ε b j (τ ) 2 - µ j 2 dz,with
Z(ε b j (τ ), A jj , y j ) = e -ϕj (z)-1 2σ 2 [Ajj z-yj ] 2 N (-p -1 j ε b j (τ ), µ j )(z)dz.
Proof. In the following E π(j) y [ψ] and E π(j) y [ψ(x)] stand for ψ(x)dπ (j) y (x). Recall that for j ≤ N the j-th mode marginal of the approximate posterior distribution πy is
dπ (j) y (X (j) ) = 1 Z π(j) y exp -φj (X (j) ) -λ -1 j ε a j (τ )ϕ j (X (j) t ) - 1 2σ 2 A jj X (j) -y j 2 × dN - 1 μj + λ -1 j ε a j (τ ) -1 λ -1 j ε b j (τ ), 1 μj + λ -1 j ε a j (τ ) -1 (X (j) ),
while the true posterior is
dπ (j) y (X (j) ) = 1 Z π(j) y exp -ϕ j (X (j) ) - 1 2σ 2 A jj X (j) -y j 2 dN (0, µ j ).
For the unobserved modes j > N , we set A jj = 0. For each j, we have
D KL π(j) y π (j) y = E π(j) y log N - 1 μj + λ -1 j ε a j (τ ) -1 λ -1 j ε b j (τ ), 1 μj + λ -1 j ε a j (τ ) -1 -log N (0, µ j ) first term + E π(j) y [1 -λ -1 j ε a j (τ )]ϕ j -φj second term + log Z π (j) y Z π(j) y third term .
First term We derive
log N - 1 μj + λ -1 j ε a j (τ ) -1 λ -1 j ε b j (τ ), 1 μj + λ -1 j ε a j (τ ) -1 (x) = - 1 2 log(2π) - 1 2 log 1 μj + λ -1 j ε a j (τ ) -1 - 1 2 1 μj + λ -1 j ε a j (τ ) x + 1 μj + λ -1 j ε a j (τ ) -1 λ -1 j ε b j (τ ) 2 ,
and
log N (0, µ j )(x) = - 1 2 log(2π) - 1 2 log µ j - 1 2 x 2 µ j .
Hence
E π(j) y log N - 1 μj + λ -1 j ε a j (τ ) -1 λ -1 j ε b j (τ ), 1 μj + λ -1 j ε a j (τ ) -1 -log N (0, µ j ) = 1 2 log 1 e -τ + (1 -e -τ )p j + p -1 j ε a j (τ ) - 1 2 E π(j) y 1 µ j 1 e -τ + (1 -e -τ )p j + p -1 j ε a j (τ ) × x + 1 e -τ + (1 -e -τ )p j + p -1 j ε a j (τ ) -1 p -1 j ε b j (τ ) 2 - x 2 µ j   .
We have
1 2 log 1 e -τ + (1 -e -τ )p j + p -1 j ε a j (τ ) = 1 2 (1 -p j )τ + p -1 j ε a j (τ ) + O(τ 2 ),and
E π(j) y 1 µ j 1 e -τ + (1 -e -τ )p j + p -1 j ε a j (τ ) × x + 1 e -τ + (1 -e -τ )p j + p -1 j ε a j (τ ) -1 p -1 j ε b j (τ ) 2 - x 2 µ j   = (1 -p j )τ + p -1 j ε a j (τ ) E π(j) y x 2 µ j + 2λ -1 j ε b j (τ )E π(j) y [x] + λ -1 j p -1 j ε b j (τ ) 2 × 1 -(1 -p j )τ -p -1 j ε a j (τ ) + O(τ 2 ).
this section cite: []

Section: Second term
We have
E exp -ϕ j ( X 0 ) | X τ = x = exp(-ϕ j (z)) 1 √ 2πv τ exp - (z -m τ (x)) 2 2v τ dz,where
m τ (x) = e -τ /2 µ j e -τ µ j + (1 -e -τ )λ j x = 1 + 1 2 -p j τ + p j 2 - 1 8 τ 2 x + O(τ 3 ),and
√ v τ = µ j - e -τ µ 2 j e -τ µ j + (1 -e -τ )λ j = λ j τ 1 - τ 4 - τ 2 32 + O(τ 5/2 ) .
this section cite: []

Section: By the change of variable w = z-mτ (x)
√ vτ , we obtain
E exp -ϕ j ( X 0 ) | X τ = x = exp(-ϕ j (z)) 1 √ 2πv τ exp - (z -m τ (x)) 2 2v τ dz = exp(-ϕ j ( √ v τ w + m τ (x))) 1 √ 2π exp(-w 2 /2)dw = exp(-ϕ j (x)) 1 + λ j ϕ ′ j (x) 2 -ϕ ′′ j (x) -ϕ ′ j (x)(1 -2p j )x τ 2 + O(τ 3/2 )
where we used the Taylor expansion for exp(-ϕ j √ v τ w + m τ (x) ) as √ τ → 0 and
w √ 2π exp(-w 2 /2) = 0, w 2 √ 2π exp(-w 2 /2) = 1.
Hence
E π(j) y (ϕ j (x) -φj (x)) = E π(j) y log 1 + λ j ϕ ′ j (x) 2 -ϕ ′′ j (x) -ϕ ′ j (x)(1 -2p j )x τ 2 + O(τ 3/2 ) .
Third term For the unobserved modes j > N , we analyze
e -ϕj (z) N (0, µ j )(z)dz e -φj (z)-λ -1 j ε a j (τ )ϕj (z) N (-[μ -1 j + λ -1 j ε a j (τ )] -1 λ -1 j ε b j (τ ), [μ -1 j + λ -1 j ε a j (τ )] -1 )(z)dz . (27
)
We use that φj (z
) = ϕ j (z) -λ j (ϕ ′ j (z) 2 -ϕ ′′ j (z)) -ϕ ′ j (z)(1 -2p j )z τ 2 + O(τ 3/2 ), which implies e -φj (z)-λ -1 j ε a j (τ )ϕj (z) = e -ϕj (z) 1 + λ j (ϕ ′ j (z) 2 -ϕ ′′ j (z)) -ϕ ′ j (z)(1 -2p j )z τ 2 -λ -1 j ε a j ϕ j (z) + O(τ 3/2 ) .
Now we consider the density of N -[μ -1
j + λ -1 j ε a j (τ )] -1 λ -1 j ε b j (τ ), [μ -1 j + λ -1 j ε a j (τ )] -1 : 1 2π[μ -1 j + λ -1 j ε a j (τ )] -1 exp   - z + [μ -1 j + λ -1 j ε a j (τ )] -1 λ -1 j ε b j (τ ) 2 2[μ -1 j + λ -1 j ε a j (τ )] -1    .(28)
We have 1
2π[μ -1 j + λ -1 j ε a j (τ )] -1 = 1 2πµ j 1 - 1 2 (1 -p j )τ - 1 2 p -1 j ε a j (τ ) + O(τ 2 ) .(29)
We now look at the exponent of (28). Its numerator reduces to
(z + p -1 j ε b j (τ )) 2 -2(z + p -1 j ε b j (τ ))((1 -p j )τ + p -1 j ε a j (τ )) + O(τ 2
), while the reciprocal of its denominator (28) reduces to 1 2µ j 1 + (1 -p j )τ + p -1 j ε a j (τ ) + O(τ 2 ) .
Then the exponent of ( 28) can be expanded as
- 1 2µ j z 2 - 1 2µ j p -2 j ε b j (τ ) 2 -zλ -1 j ε b j (τ ) + (z + p -1 j ε b j (τ )) µ j + (z + p -1 j ε b j (τ )) 2 2µ j ((1 -p j )τ + p -1 j ε a j (τ )) + O(τ 2 ),
and the exponential term in (28) becomes
exp - (z + p -1 j ε b j (τ )) 2 2µ j × 1 + z + p -1 j ε b j (τ ) µ j + (z + p -1 j ε b j (τ )) 2 2µ j ((1 -p j )τ + p -1 j ε a j (τ )) + O(τ 2 ) .(30)
Putting together (29) and (30) we get that the Gaussian density (28) is expanded as
N (-p -1 j ε b j (τ ), µ j )(z) × 1 + z + p -1 j ε b j (τ ) µ j + (z + p -1 j ε b j (τ )) 2 µ j - 1 2 ((1 -p j )τ + p -1 j ε a j (τ )) + O(τ 2 ) .
We can now expand for small τ
e -φj -λ -1 j ε a j (τ )ϕj dN (-[μ -1 j + λ -1 j ε a j (τ )] -1 λ -1 j ε b j (τ ), [μ -1 j + λ -1 j ε a j (τ )] -1 ) -1 . Let Z j (ε b j (τ )) = e -ϕj (z) N (-p -1 j ε b j (τ ), µ j )(z)dz. We derive Z j (ε b j (τ )) -1 1 -Z j (ε b j (τ )) -1 × e -ϕj (z) N (-p -1 j ε b j (τ ), µ j )(z) λ j (ϕ ′ (z) 2 -ϕ ′′ j (z)) -ϕ ′ j (z)(1 -2p j )z τ 2 -λ -1 j ε a j (τ )ϕ j (z) + z + p -1 j ε b j (τ ) µ j + (z + p -1 j ε b j (τ )) 2 µ j - 1 2 (1 -p j )τ + p -1 j ε a j (τ ) dz + O(τ 3/2
). Then (27) can be expanded as
log Z j (0) -log Z j (ε b j (τ )) -Z j (ε b j (τ )) -1 e -ϕj (z) N (-p -1 j ε b j (τ ), µ j )(z) λ j (ϕ ′ (z) 2 -ϕ ′′ j (z)) -ϕ ′ j (z)(1 -2p j )z τ 2 -λ -1 j ε a j (τ )ϕ j (z) + z + p -1 j ε b j (τ ) µ j + (z + p -1 j ε b j (τ )) 2 µ j - 1 2 × ((1 + p j )τ + p -1 j ε a j (τ )) dz + O(τ 3/2 ). Now let Z j (ε b j (τ ), A jj , y j ) = e -ϕj (z)-1 2σ 2 [Ajj z-yj ] 2 N (-p -1 j ε b j (τ ), µ j )(z)dz. For the observed modes j ≤ N , we get log Z j (0, A jj , y j ) -log Z j (ε b j (τ ), A jj , y j ) -Z j (ε b j (τ ), A jj , y j ) -1 e -ϕj (z)-1 2σ 2 [Ajj z-yj ] 2 N (-p -1 j ε b j (τ ), µ j )(z) λ j (ϕ ′ (z) 2 -ϕ ′′ j (z)) -ϕ ′ j (z)(1 -2p j )z τ 2 -λ -1 j ε a j (τ )ϕ j (z) + z + p -1 j ε b j (τ ) µ j + (z + p -1 j ε b j (τ )) 2 µ j - 1 2 × ((1 -p j )τ + p -1 j ε a j (τ )) dz + O(τ 3/2 ).
Remark C.1. If ϕ j is smooth and ε b j (τ ) = O(1), then E j (τ ) → 0 as τ → 0.
this section cite: ['b27', 'b27', 'b28', 'b29', 'b27']

Section: D Illustrations: Additional Details
Here we provide additional details on the theoretical setup underlying the illustrations. All illustrations were generated on Google Colab (13 GB of RAM), and all code executions took less than one minutefoot_0 .
this section cite: []

Section: D.1 Recovering the KL coefficients of the Brownian sheet
The Brownian sheet B(x 1 , x 2 ) is a Gaussian process with zero mean and covariance Cov(B(x 1 , x 2 ), B(y 1 , y 2 )) = min(x 1 , y 1 ) min(x 2 , y 2 ).
Its Karhunen-Loève expansion [50] is
B(x 1 , x 2 ) = j,k ϕ j,k (x 1 , x 2 )η j,k , (x 1 , x 2 ) ∈ [0, 1] 2 ,
where η j,k ∼ N (0, µ j,k ) are independent Gaussian random variables, and
ϕ j,k (x 1 , x 2 ) = 2 sin π j - 1 2 x 1 sin π k - 1 2 x 2 , µ j,k = j - 1 2 π k - 1 2 π -2 .
In Section 6, we truncate the KL expansion after N modes
B N (x 1 , x 2 ) = N j,k=1 ϕ j,k (x 1 , x 2 )η j,k ,
and consider the inverse problem of recovering the first N 2 coefficients from noisy observations corresponding to the first M 2 ≤ N 2 modes y j,k = ηj,k + ε j,k , j, k ≤ M, where the prior is ηj,k ∼ N (0, µ j,k ) and the noise ε j,k ∼ N (0, σ 2 ). This setup satisfies the assumptions of our theory, since the prior diagonal in the KL basis (ϕ j,k ) and the forward map is simply the projection onto these modes, so that A j,k,j ′ ,k ′ = δ j,j ′ δ k,k ′ , j, j ′ , k ′ , k ≤ M, and zero otherwise. As a result, the posterior for each coefficient remains Gaussian η j,k | y j,k ∼ π (j,k) y j,k = N (m j,k , v j,k ), with, for j, k ≤ M , v j,k = µ -1 j,k + σ -2 -1 = µ j,k σ 2 µ j,k + σ 2 , m j,k = µ j,k µ j,k + σ 2 y j,k , and for j > M or k > M (unobserved modes) the posterior simply coincides with the prior, v j,k = µ j,k , m j,k = 0.
this section cite: ['b49']

Section: Experimental details
In Figure 2, within the theoretical setup described above, we set the noise level σ = 10 -2 , chose N = 200, and varied the number of observed modes M 2 = 75 2 , 200 2 to illustrate the discretization-invariance of the preconditioned Langevin sampler. This is confirmed by the small errors reported in the fourth column of Figure 2. The preconditioned Langevin dynamics, using the preconditioner C M = Diag 1≤j,k≤M (λ j,k ), λ j,k = [µ -1 j,k + σ -2 ] -1 , was run for 5 • 10 3 iterations with a fixed step-size of 5 • 10 -1 . We assumed access to the exact score function, i.e., ϕ = τ = 0 in (6).
this section cite: []

Section: D.2 Inverse source problem for the heat equation
Let Ω = [0, 1] 2 ⊆ R 2 . Consider u : Ω × [0, T ] → R solving the heat equation    ∂ t u(x, t) = ∆u(x, t), (x, t) ∈ Ω × (0, T ], u(x, 0) = g(x),
x ∈ Ω, u(x, t) = 0,
x ∈ ∂Ω × (0, T ].
Set u(x 1 , x 2 ; t) = ∞ j,k=1 u j,k (t) ψ j,k (x 1 , x 2 ), where (ψ j,k , ζ j,k ) are the Dirichlet eigenpairs of -∆ on [0, 1] 2 :
-∆ψ j,k (
x 1 , x 2 ) = ζ j,k ψ j,k (x 1 , x 2 ), (x 1 , x 2 ) ∈ [0, 1] 2 , ψ j,k | ∂[0,1] 2 = 0.
We have ψ j,k (x 1 , x 2 ) = 2 sin(jπx 1 ) sin(kπx 2 ), ζ j,k = π 2 (j 2 + k 2 ).
The coefficients evolve as
u j,k (t) = e -ζ j,k t g j,k , g j,k = ⟨g, ϕ j,k ⟩.
In Section 6, we consider the the so-called backward heat equation-the ill-posed inverse problem of recovering the initial condition g from noisy measurements of u(•, T ) inside Ω y j,k = e -ζ j,k T g j,k + ε j,k , j, k ≤ M, by adopting a Bayesian approach [3]. We assume a Gaussian prior g j,k ∼ N (0, e -βζ j,k ) and independent Gaussian noise ε j,k ∼ N (0, σ 2 ). The forward map is diagonal in (ψ j,k ), with
A j,k,j ′ ,k ′ = e -ζ j,k T δ j,j ′ δ k,k ′ , j, j ′ , k, k ′ ≤ M.
As a result, the posterior for each coefficient remains Gaussian
g j,k | y j,k ∼ N (m j,k , v j,k ),
with, for j, k ≤ M ,
v j,k = e -βζ j,k σ 2 e -(β+2T )ζ j,k + σ 2 , m j,k = µ j,k e -(β+2T )ζ j,k + σ 2 e -ζ j,k T y j,k .
For j > M or k > M (unobserved modes), the posterior simply coincides with the prior.
this section cite: ['b2']

Section: Experimental details
In Figure 3, within the theoretical setup described above, we fixed the noise level at σ = 5 • 10 -3 , chose M = 15 (i.e. 225 observed modes), and set T = 0.1. We then ran the preconditioned Langevin sampler-with the optimal preconditioner C from Theorem 4.1-using the exact score function perturbed by a relative error ε a j ∼ N (0, 0.1 2 ), scaled by τ = 10 -3 in the top row of Figure 3 and by 10 -1 in its bottom row, and with zero bias (i.e. ε b j = 0) to simulate a learned score. This sampler was run for 5 • 10 3 iterations with a fixed step-size of 10 -2 . For comparison, we also executed the vanilla Langevin sampler for 1.5 • 10 4 iterations with a fixed step-size of 10 -6 . To further illustrate the quality of our preconditioned posterior samples, Figure 5 below shows uncertainty quantification for Figure 3. For the first 35 modes, we plot the conditional posterior mean (red), the 95% credible interval (orange shading), and the ground truth (dotted black line). NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes]
Justification: This work focuses on a Langevin-type diffusion algorithm driven by scorebased generative priors in infinite dimensions, proving convergence error estimates and existence and form of an optimal preconditioner.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: See the last section.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Code is uploaded in a single zip file along with additional supplementary materials (e.g. Appendix).
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: See main text, Appendix, and code.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: The posterior mean and standard deviation intervals presented throughout the main paper and Appendix reflect the inferred uncertainty, while the posterior samples illustrate the variability and structure of the distribution.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
this section cite: []

Section: 
Answer: [Yes] Justification: All the details of the proof are provided in the Appendix.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: See Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: 5.
Open access to data and code
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: Our experiments require less than a minute per run on a laptop. Details are reported in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We conform to the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: Our paper mainly focuses on a theoretical analysis of a popular sampler in infinite dimensions.
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
Answer: [NA] Justification: This work poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA] Justification: We use open-source packages.
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
Answer: [No] Justification: No new asset is released.
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
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The paper is mainly a theoretical analysis and does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Inverse problem theory and methods for model parameter estimation Year: (2005)
Ref_id:b1 Title: Lectures on Cauchy's problem in linear partial differential equations Year: (1923)
Ref_id:b2 Title: Inverse problems: a bayesian perspective Year: (2010)
Ref_id:b3 Title: Bayesian inverse problems with Gaussian priors Year: (2011)
Ref_id:b4 Title: Uncertainty quantification and weak approximation of an elliptic inverse problem Year: (2011)
Ref_id:b5 Title: Uncertainty quantification in bayesian inversion Year: (2014)
Ref_id:b6 Title: The bayesian approach to inverse problems Year: (2013)
Ref_id:b7 Title: MCMC methods for functions: Modifying old algorithms to make them faster Year: (2013)
Ref_id:b8 Title: Can one use total variation prior for edge-preserving bayesian inversion? Inverse problems Year: (2004)
Ref_id:b9 Title: A kernelized stein discrepancy for goodness-of-fit tests Year: (2016)
Ref_id:b10 Title: A connection between score matching and denoising autoencoders Year: (2011)
Ref_id:b11 Title: Improved techniques for training score-based generative models Year: (2020)
Ref_id:b12 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b13 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b14 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b15 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b16 Title: Conditional image generation with score-based diffusion models Year: (2021)
Ref_id:b17 Title: Snips: Solving noisy inverse problems stochastically Year: (2021)
Ref_id:b18 Title: Robust compressed sensing mri with deep generative priors Year: (2021)
Ref_id:b19 Title: Conditional score-based diffusion models for bayesian inference in infinite dimensions Year: (2024)
Ref_id:b20 Title: Taming scorebased diffusion priors for infinite-dimensional nonlinear inverse problems Year: (2024)
Ref_id:b21 Title: Conditional optimal transport on function spaces Year: (2023)
Ref_id:b22 Title: Multilevel diffusion: Infinite dimensional score-based diffusion models for image generation Year: (2023)
Ref_id:b23 Title: Score-based diffusion models as principled priors for inverse imaging Year: (2023)
Ref_id:b24 Title: Provable probabilistic imaging using score-based generative priors Year: (2024)
Ref_id:b25 Title: Provably robust score-based diffusion posterior sampling for plug-and-play image reconstruction Year: (2024)
Ref_id:b26 Title: Conditional path sampling of sdes and the langevin mcmc method Year: (2004)
Ref_id:b27 Title: Analysis of spdes arising in path sampling. part i: The gaussian case Year: (2005)
Ref_id:b28 Title: Analysis of spdes arising in path sampling part ii: The nonlinear case Year: (2007)
Ref_id:b29 Title: Geometric mcmc for infinitedimensional inverse problems Year: (2017)
Ref_id:b30 Title: Infinite dimensional adaptive mcmc for gaussian processes Year: (2018)
Ref_id:b31 Title: High-dimensional Bayesian inference via the unadjusted Langevin algorithm Year: (2019)
Ref_id:b32 Title: Nonasymptotic convergence analysis for the unadjusted Langevin algorithm Year: (2017)
Ref_id:b33 Title: Theoretical guarantees for approximate sampling from smooth and log-concave densities Year: (2017)
Ref_id:b34 Title: Spectral gaps for a Metropolis-Hastings algorithm in infinite dimensions Year: (2014)
Ref_id:b35 Title: Dimension-independent likelihood-informed mcmc Year: (2016)
Ref_id:b36 Title: Multilevel dimension-independent likelihood-informed mcmc for large-scale inverse problems Year: (2024)
Ref_id:b37 Title: Multilevel sequential monte carlo with dimension-independent likelihood-informed proposals Year: (2018)
Ref_id:b38 Title: Localization for mcmc: sampling highdimensional posterior distributions with local structure Year: (2019)
Ref_id:b39 Title: Mcmc methods for diffusion bridges Year: (2008)
Ref_id:b40 Title: Accelerating score-based generative models with preconditioned diffusion sampling Year: (2022)
Ref_id:b41 Title: Preconditioned score-based generative models Year: (2025)
Ref_id:b42 Title: Continuoustime functional diffusion processes Year: (2024)
Ref_id:b43 Title: Generative diffusion models in infinite dimensions: a survey Year: (2025)
Ref_id:b44 Title: Score-based diffusion models in function space Year: (2023)
Ref_id:b45 Title: Diffusion generative models in infinite dimensions Year: (2022)
Ref_id:b46 Title: Score-based generative modeling through stochastic evolution equations in hilbert spaces Year: (2024)
Ref_id:b47 Title: ∞-diff: Infinite resolution diffusion with subsampled mollified states Year: (2023)
Ref_id:b48 Title: Infinite-dimensional diffusion models Year: (2024)
Ref_id:b49 Title: Karhunen-Loeve expansions and their applications Year: (2008)
