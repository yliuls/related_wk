Title: FAST ESCAPE, SLOW CONVERGENCE: LEARNING DY-NAMICS OF PHASE RETRIEVAL UNDER POWER-LAW DATA
Abstract: Scaling laws describe how learning performance improves with data, compute, or training time, and have become a central theme in modern deep learning. We study this phenomenon in a canonical nonlinear model: phase retrieval with anisotropic Gaussian inputs whose covariance spectrum follows a power law. Unlike the isotropic case, where dynamics collapse to a two-dimensional system, anisotropy yields a qualitatively new regime in which an infinite hierarchy of coupled equations governs the evolution of the summary statistics. We develop a tractable reduction that reveals a three-phase trajectory: (i) fast escape from low alignment, (ii) slow convergence of the summary statistics, and (iii) spectral-tail learning in low-variance directions. From this decomposition, we derive explicit scaling laws for the mean-squared error, showing how spectral decay dictates convergence times and error curves. Experiments confirm the predicted phases and exponents. These results provide the first rigorous characterization of scaling laws in nonlinear regression with anisotropic data, highlighting how anisotropy reshapes learning dynamics.

Section: INTRODUCTION
Scaling laws quantify how the performance of a learning algorithm varies with resources such as training time, dataset size, or model capacity. Empirically, losses often follow simple power laws across wide ranges of data and computation, enabling forecasting from a handful of measurements (Hestness et al., 2017;Kaplan et al., 2020;Hoffmann et al., 2022). These regularities naturally raise the fundamental question: when and how do such laws emerge from first principles?
Despite their central role in modern deep learning practice, neural scaling laws remain theoretically poorly understood. A notable exception is provided by linear models, where the scaling of the generalization error has been thoroughly analysed within the classical kernel literature, encompassing both ridge regression (Caponnetto & De Vito, 2007;Rudi & Rosasco, 2017) and stochastic gradient descent (Yao et al., 2007;Ying & Pontil, 2008;Carratino et al., 2018;Pillaud-Vivien et al., 2018;Kunstner & Bach, 2025). Recent developments in this direction, driven by the empirical observations of cross-overs and bottlenecks in the context of neural networks, demonstrate that analogous phenomena are already present in linear settings (Cui et al., 2021;Defilippis et al., 2024;Bahri et al., 2024;Maloney et al., 2022;Atanasov et al., 2024;Paquette et al., 2024;Bordelon et al., 2024;Lin et al., 2024). By contrast, nonlinear settings, ubiquitous in practice (e.g., functional data, learned feature maps, or embeddings with heavy spectral tails), remain far less understood.
We address this gap in the canonical nonlinear regression problem of phase retrieval:
y = ⟨x, w ⋆ ⟩ 2 + ξ,
x ∼ N (0, Q), where w ⋆ ∈ R d is the target vector, and Q has eigenvalues (λ i ) d i=1 obeying a power law λ i ∝ i -a with a > 1 and noise ξ. This model captures two core difficulties: (i) a nonconvex landscape, and (ii) strong anisotropy that induces highly unbalanced learning across directions.
Figure 1: Evolution of the MSE during training with online SGD for different spectral exponents a (log-log scale). For a > 1, convergence is markedly slower than the exponential decay seen in the isotropic case, reflecting the difficulty of learning directions associated with small eigenvalues.
Figure 1 illustrates this phenomenon: under the same initialization, noise level, stepsize, and optimization algorithm (SGD), the mean-square error (MSE) behaves differently depending on a. Intuitively, when a is larger, directions associated with small eigenvalues λ i are harder to learn, so the MSE decays more slowly. By contrast, in isotropic designs, all directions progress at comparable rates, causing the error to drop sharply. These observations motivate our central question:
How does the input spectrum govern finite-time convergence in nonlinear regression, and can we predict the learning curve from the spectral decay?
Concretely, we seek to (i) analyze the mechanisms behind the plateau-drop structure in anisotropic phase retrieval, and (ii) derive scaling laws that quantify MSE decay as functions of the spectral parameter a and time t.
1.1 CONTRIBUTIONS.
• New phenomena in the anisotropic case. We show that anisotropy challenges several aspects of the intuition developed in isotropic settings. In the isotropic setting, the dynamics collapse to a low-dimensional ODE, and the main challenge is escaping mediocrity, i.e. the regime where the correlation with the signal w * is vanishing, before convergence accelerates. Under anisotropy, by contrast, the dynamics form an infinite hierarchy of coupled equations. This structural change flips the qualitative behavior, yielding an escape-convergence trade-off : escaping from mediocrity can be faster, but convergence to a low MSE is slowed by the difficulty of learning directions associated with small eigenvalues, as illustrated by Figure 1 (a large a is associated with a slow decay of the MSE) and Figure 3 (a large a leads quickly to constant order correlation with the signal). Numerical experiments confirm this phase-level contrast between isotropic and anisotropic regimes.
• Analytical framework. We first obtain a closed-form representation of the dynamics via Duhamel's formula. We then introduce a phase decomposition of the trajectory, isolating regimes where different approximations become valid. This allows us to analyze the qualitative behavior of the ODE hierarchy phase by phase: (i) fast escape from mediocrity, (ii) macroscopic convergence of summary statistics, and (iii) spectral-tail learning of small-eigenvalue directions. The combination of Duhamel representation and phase-specific approximations provides a systematic way to make infinite-dimensional dynamics tractable.
• Scaling laws. As a byproduct of this analysis, we derive explicit scaling laws for anisotropic phase retrieval. These formulas quantify how the eigenvalue decay governs the MSE. Numerical results corroborate the predicted exponents across different spectral profiles.
this section cite: ['b28', 'b30', 'b29', 'b15', 'b40', 'b50', 'b51', 'b16', 'b38', 'b31', 'b18', 'b21', 'b7', 'b34', 'b5', 'b36', 'b11', 'b32']

Section: OTHER RELATED WORK
Theory of scaling laws. The study of risk scaling in problems with power-law structure is a classical theme in the kernel literature, where it falls under the framework of source and capacity conditions. It has been extensively investigated for kernel ridge regression (Caponnetto & De Vito, 2007;Cui et al., 2021), random features regression (Rudi & Rosasco, 2017;Defilippis et al., 2024), and also for (S)GD (Yao et al., 2007;Ying & Pontil, 2008;Carratino et al., 2018;Pillaud-Vivien et al., 2018). This line of work has recently gained renewed relevance in the context of neural scaling laws, with linear models emerging as theoretical testbeds to explain the plateaux and crossovers observed in practice (Bahri et al., 2024;Maloney et al., 2022;Atanasov et al., 2024;Bordelon et al., 2024;Worschech & Rosenow, 2024;Paquette et al., 2024;Lin et al., 2024). More recently, Wortsman & Loureiro (2025) have investigated kernel ridge regression under anisotropic power-law inputs, showing that in some regimes the scaling of the covariates transfer to the features. Sums of orthogonal single-index models have been analysed in the feature-learning regime, albeit still under isotropic input distributions (Ren et al., 2025;Ben Arous et al., 2025;Defilippis et al., 2025).
In short, while the linear setting with anisotropic spectra (e.g. power-law decays) is by now well understood, the nonlinear regime has remained largely isotropic. Our work addresses this gap by deriving compute-error scaling laws in a nonlinear model with anisotropic Gaussian inputs.
Phase retrieval and quadratic neural networks. Phase retrieval (PR) is a classical inverse problem motivated by imaging: reconstruct a signal from intensity-only measurements; see Dong et al. (2023) for a recent tutorial. A rich algorithmic literature includes spectral initializations and nonconvex Wirtinger-flow refinements (Ma et al., 2021;Candès et al., 2015;Tan & Vershynin, 2019;2023;Davis et al., 2020). PR can be viewed as learning a single neuron with a quadratic activation, connecting it to the broader theory of quadratic networks and their training dynamics (Sarao Mannelli et al., 2020;Arnaboldi et al., 2023a;Martin et al., 2024;Erba et al., 2025). In this context, Ben Arous et al. (2025) have studied scaling laws for one-pass SGD and Defilippis et al. (2025) for full-batch ERM, in a quadratic net model where the target coefficients follow a power-law. Most theoretical analyses of PR assume isotropic sub-Gaussian measurements; the impact of anisotropic covariances on the learning curve has received less attention. Our analysis isolates precisely this aspect and quantifies how the input spectrum shapes the three-phase trajectory and the resulting scaling laws.
this section cite: ['b15', 'b18', 'b40', 'b21', 'b50', 'b51', 'b16', 'b38', 'b7', 'b34', 'b5', 'b11', 'b48', 'b36', 'b32', 'b49', 'b39', 'b9', 'b22', 'b23', 'b33', 'b14', 'b44', 'b20', 'b43', 'b35', 'b22']

Section: Non-convex optimization and feature learning.
A complementary line of work studies gradient-based training in multi-index and shallow networks, characterizing feature learning, convergence phases, and computational-statistical trade-offs, predominantly under isotropic designs (Saad & Solla, 1995a;b;Goldt et al., 2019;Veiga et al., 2022;Arnaboldi et al., 2023b;2024;Collins-Woodfin et al., 2024;Ben Arous et al., 2022;Abbe et al., 2022;2023;Bietti et al., 2025;Dandi et al., 2024;Bruna & Hsu, 2025). Results with anisotropic inputs are more limited: some works consider spiked covariances and rely on preconditioned methods (Ba et al., 2023), while others treat a broader but weaker anisotropic regime that does not include power-law spectra (Goldt et al., 2020;Braun et al., 2025). By contrast, we analyze the unpreconditioned gradient flow in a strongly anisotropic regime and show that anisotropy destroys the finite-dimensional closure of the dynamics, leading to an infinite hierarchy whose Duhamel-Volterra reduction yields explicit scaling laws.
this section cite: ['b30', 'b25', 'b47', 'b17', 'b8', 'b0', 'b10', 'b19', 'b13', 'b6', 'b26', 'b12']

Section: NOTATIONS
We use ∥•∥ and ⟨•, •⟩ to denote the Euclidean norm and scalar product, respectively. When applied to a matrix, ∥•∥ refers to the operator norm. Any positive definite matrix Q induces a scalar product defined by ⟨x, y⟩ Q = x ⊤ Qy. The Frobenius norm of a matrix A is denoted by ∥A∥ F . The d × d identity matrix is represented by I d . The (d -1)-dimensional unit sphere is denoted by S d-1 . We use the notation
a n ≲ b n (or a n ≳ b n ) for sequences (a n ) n≥1 and (b n ) n≥1 if there exists a constant C > 0 such that a n ≤ Cb n (or a n ≥ Cb n ) for all n.
If the inequalities hold only for sufficiently large n, we write
a n = O(b n ) (or a n = Ω(b n )). We denote by C ∞ b (R ≥0 ; ℓ 2
) the Banach space of bounded and infinitely differentiable functions from R ≥0 to ℓ 2 := {(x n ) n≥0 :
x 2 n < ∞}, the Hilbert space of square summable sequences. We write * for convolution, and for a function f we denote by f its Laplace transform.
Data distribution. By orthogonal invariance of the Gaussian distribution, we may assume without loss of generality that the covariance matrix is diagonal: Q = diag(λ 1 , . . . , λ d ) ∈ R d×d . We assume a power-law spectrum,
λ i = i -a d j=1 j -a , a > 1,
so that tr(Q) = 1. This assumption reflects the slow, heavy-tailed eigenvalue decay observed in many empirical covariance spectra (e.g., images, text embeddings, kernel features). From a theoretical perspective, the exponent a provides a simple parametrization of anisotropy: it controls the balance between a few dominant directions and a long tail of weak ones. It is widely used as a canonical model for scaling laws in learning dynamics. We consider the phase retrieval setting
y = ⟨x, w ⋆ ⟩ 2 + ξ, x ∼ N (0, Q),
where the target weights are generated by w ⋆ = u ∥Q 1/2 u∥ , u ∼ N (0, I d ). This construction ensures that ∥Q 1/2 w ⋆ ∥ = 1. For numerical experiments, we sometimes adopt the alternative normalization ∥w ⋆ ∥ 2 = d, so that all curves start from the same baseline when comparing different decay exponents a; both normalizations are of similar order. The noise term ξ ∼ N (0, σ 2 ) is independent of x. Since our analysis focuses on the population dynamics, label noise only shifts the loss by a constant and can be set to zero without loss of generality.
Model. We consider estimators of the form ⟨x, w⟩ 2 for w ∈ R d and the population loss used to train our model
L(w) = E x (x ⊤ w) 2 -(x ⊤ w ⋆ ) 2 2 .
However, under anisotropy, this loss can be misleading: a vector w that aligns with w ⋆ only along the directions corresponding to the largest eigenvalues may still achieve a small loss. To evaluate how well w actually recovers the signal direction, a more natural metric is the MSE defined as
MSE(w, w ⋆ ) = 1 d min ∥w -w ⋆ ∥ 2 , ∥w + w ⋆ ∥ 2 .
Optimization. The learner maintains a weight vector w(t) ∈ R d , initialized uniformly at random on the unit sphere S d-1 . Its evolution follows the gradient flow dynamics
ẇ(t) = -∇ w L(w(t))
which can be viewed as the continuous-time counterpart of gradient descent.
this section cite: []

Section: LOSS GEOMETRY AND EVOLUTION OF SUMMARY STATISTICS
In this section, we describe the main characteristics of the loss landscape and establish ODEs to describe the evolution of the key summary statistics. The proofs are in the appendix, Section B. For convenience, we introduce the following notations: s := ∥w∥ 2 Q = w ⊤ Qw, s ⋆ := ∥w ⋆ ∥ 2 Q , and u := ⟨w, w ⋆ ⟩ Q = w ⊤ Qw ⋆ .
this section cite: []

Section: LOSS SIMPLIFICATION
Although the loss is defined on a d-dimensional parameter space, it depends only on two summary statistics: ∥w∥ 2 Q and ⟨w, w ⋆ ⟩ 2 Q . This dimensional reduction makes the geometry of the loss landscape transparent, as captured in the following proposition. Proposition 1. Let x ∼ N (0, Q) ∈ R d , where Q ∈ R d×d is a symmetric positive definite diagonal matrix. Let w, w ⋆ ∈ R d . The population loss can be rewritten as
L(w) = 3∥w∥ 4 Q + 3∥w ⋆ ∥ 4 Q -4⟨w, w ⋆ ⟩ 2 Q -2∥w∥ 2 Q • ∥w ⋆ ∥ 2 Q = 3s 2 + 3s 2 ⋆ -4u 2 -2s ⋆ s
Next, we compute the population gradient and characterize the critical points.
this section cite: []

Section: GRADIENT, HESSIAN, AND CRITICAL POINTS
Proposition 2. The population gradient and Hessian of the loss L are given by
∇L(w) = 12sQw -4s ⋆ Qw -8uQw ⋆ , (3.1) ∇ 2 L(w) = 24(Qw)(Qw) ⊤ + (12s -4s ⋆ )Q -8(Qw ⋆ )(Qw ⋆ ) ⊤ . (3.2)
The set of critical points is {0, w ⋆ , -w ⋆ } ∪ {w : (u, s) = (0, s ⋆ /3)}. Among them, 0 is a strict local maximum, ±w ⋆ are strict global minima, and every w with (u, s) = (0, s ⋆ /3) is a saddle point. Remark 1. The loss landscape contains no spurious local minima: the only minima are the global optima ±w ⋆ . At the same time, the origin is a strict local maximum, and the remaining critical points are saddles. Nevertheless, the dynamics exhibit a long plateau phase. At initialization, one typically has u(0) ≈ d -1/2 ≈ 0. In this low-correlation regime, the gradient is small, so the iterates take a long time to escape. Even after leaving this regime, convergence remains slower than in the isotropic case. Remark 2. The anisotropic gradient flow can be viewed as a Q-preconditioned version of the isotropic flow. Under the reparametrization z = Q 1/2 w, define the isotropic loss L I (z) = L(Q -1/2 z). By the chain rule, ż(t) = -Q∇L I (z(t)). Thus, while the critical points coincide with those in the isotropic case, the dynamics differ substantially due to the preconditioning by Q.
this section cite: []

Section: INFINITE-DIMENSIONAL STRUCTURE OF GRADIENT FLOW
As shown in Proposition 1, the loss depends only on the summary statistics u(t) and s(t). Controlling these two quantities already yields a faithful description of the loss trajectory.
A crucial distinction, however, arises between the isotropic and anisotropic settings. In the isotropic case (Q = I), the dynamics of the loss can be expressed entirely in terms of two scalars: the signal overlap u(t) and the energy s(t). This leads to a closed, two-dimensional ODE system.
In contrast, as soon as Q ̸ = I, the situation changes qualitatively. Proposition 3 shows that the evolution of u(t) and s(t) necessarily involves higher-order weighted overlaps,
s (k) (t) := ∥w(t)∥ 2 Q k , s (k) ⋆ := ∥w ⋆ ∥ 2 Q k , u (k) (t) := ⟨w(t), w ⋆ ⟩ Q k ,
with the conventions s = s (1) (t), u = u (1) (t), and s ⋆ = s
⋆ = 1. The resulting system is an infinite hierarchy of coupled ODEs. Unlike the isotropic case, no finitedimensional closure exists: the time derivative of order-k statistics depends on order-(k+1) statistics, and so on. Understanding the anisotropic dynamics thus requires working with this infinitedimensional structure. Proposition 3. The gradient flow dynamics satisfy
ẇi (t) = 4λ i s ⋆ -3s(t) w i (t) + 8λ i u(t)w ⋆ i , i = 1, . . . , d, (3.3) ṡ(k) (t) = 8 (s ⋆ -3s(t)) s (k+1) (t) + 16 u(t) u (k+1) (t), (3.4) u(k) (t) = 4 (s ⋆ -3s(t)) u (k+1) (t) + 8 u(t) s (k+1) ⋆ .
(3.5)
In particular, setting k = 1 recovers the dynamics of s(t) and u(t).
this section cite: []

Section: THREE-PHASE STRUCTURE OF THE DYNAMICS
The system of ODEs from Proposition 3 involves higher-order moments, preventing a closed-form analysis as in the isotropic case. To guide our theory, we first examine the empirical evolution of the key observables u(t) and s(t); see Figure 2. The trajectories display a three-phase structure:
(i) Phase I: Escape from mediocrity, s(t) ≈ 0 (t ≤ T 1 ). At initialization, u(0) ≈ 1/ √ d, there is almost no correlation with the signal. During this "warm-up" stage, the correlation u(t) escapes exponentially from zero and reaches a small but fixed constant u(T 1 ) = δ > 0. The overall MSE, however, remains essentially unchanged, since only a few easy directions-those aligned with large eigenvalues-have been learned so far.
(ii) Phase II: Convergence u(t), s(t) → 1 (T 1 ≤ t < T 2 ). Here, the signal alignment strengthens and the summary statistics u(t), s(t) approach their limiting values. Two distinct episodes appear:
• (IIa) Transition (T 1 ≤ t ≤ T ′ 1 ). The energy s(t) crosses the critical threshold 1/3 and stays above it.
• (IIb) Asymptotic convergence (T ′ 1 < t < T 2 ). Both u(t) and s(t) approach 1, but convergence is slower than in the isotropic case (see Section F), reflecting the difficulty of learning directions corresponding to small eigenvalues of Q.
(iii) Phase III: Spectral-tail learning u(t), s(t) ≈ 1 (t > T 2 ). After u(t) and s(t) have essentially stabilized, the MSE continues to decrease as the flow progressively learns directions associated with the small eigenvalues of Q.
This empirical decomposition provides a roadmap for the analysis: by isolating each phase, the infinite-dimensional system becomes amenable to tractable approximations.
this section cite: []

Section: MAIN RESULTS: THREE-PHASE GRADIENT FLOW DYNAMICS
The experiments in Section 3.4 revealed a characteristic trajectory with three successive phases: (i) escape from mediocrity, (ii) convergence of the summary statistics, and (iii) spectral-tail learning. We now show that this qualitative picture admits a rigorous derivation from the gradient flow equations. The following theorems make precise the stopping times, plateau behavior, and tail-driven decay observed empirically.
this section cite: []

Section: PHASE I-II: ESCAPE AND APPROXIMATE CONVERGENCE OF THE SUMMARY STATISTICS
The next theorem formalizes escape and approximate convergence, under initialization conditions whose full statement is deferred to Appendix C (see Assumption A1).
Theorem 1 (Phases I and II). For d sufficiently large, let ε ≳ d -(a-1)/2 . Under mild conditions on initialization (e.g. u(0) ≍ d -1/2 ), there exist stopping times T 1 = O(log d), T ′ 1 = T 1 + O(1), and
T 2 = T ′ 1 + O(ε -2a/(a-1) log(1/ε)
) such that:  • Initialization and mediocrity. The sign of u(t) is fixed at initialization; w.l.o.g. one may assume it positive by replacing w ⋆ with -w ⋆ . Under random isotropic initialization, |u(0)| is typically of order d -1/2 (as in the isotropic case), hence the escape mediocrity time satisfies
T 1 ≍ log(1/|u(0)|) = Θ(log d).
In particular, anisotropy does not remove this logarithmic barrier. A table summarizing the different stopping times can be found in Section A.
• Anisotropic vs. isotropic. In isotropic models, u(0) directly controls early growth and convergence is relatively fast; in the anisotropic setting, the Volterra reduction shows that post-plateau escape is controlled by a linear functional of the initialization (via residue/resolvent calculus), and convergence after escape is tail-controlled and slower.
• Accuracy vs. time. For ε ≳ d -(a-1)/2 stabilization occurs within O(ε -2a/(a-1) log(1/ε)), while demanding ε ≪ d -(a-1)/2 leads to even higher time costs (see Phase III analysis).
Remark 4. The analysis of Phase I reveals that the u(t) grows exponentially at a rate depending on a: the convergence is faster when a is large (see Proposition 9 in the appendix, and Figure 3 for a numerical illustration).
this section cite: []

Section: SPECTRAL-TAIL LEARNING AND PHASE III
While Phases I-II describe the evolution of summary statistics (u, s), these quantities do not capture how the estimator w(t) approaches the ground truth entrywise. The mean squared error
MSE(t) = 1 d d i=1 (w i (t) -w ⋆ i ) 2
is the natural measure of recovery: it is the quantity plotted in Fig. 1, and its log-log decay rates yield the scaling laws highlighted in the introduction. Moreover, the MSE aggregates contributions from all eigen-directions, making it the right observable to expose the effect of the spectral tail.
Up to time T 2 , while the summary statistics are still converging, the MSE shows almost no decrease and stays essentially at its initial level. The following proposition captures this plateau behavior.
Proposition 4. Let σ 2 ⋆ = 1 d d i=1 (w ⋆ i ) 2 . Under the assumptions of Theorem 1 we have MSE(T 2 ) -σ 2 ⋆ ≲ ε -a d 1/3 + log d d 1/3 . After T 2 , progress is governed by coordinates aligned with the smallest eigenvalues. Let e i (t) = w i (t) -w ⋆ i be the coordinate error and define π i = e i (T 2 ) 2 / j e j (T 2 ) 2 . The weighted average S d (τ ) = i π i e -16λiτ describes the spectral decay of the error, while the uniform benchmark S d (τ ) = d -1 i e -16λiτ admits explicit asymptotics. We also set s
⋆ = i λ 2 i (w ⋆ i ) 2 , H d,a = d j=1 j -a , β d = 16/H d,a(2)
, and x d = (β d τ ) 1/a . Theorem 2 (Phase III: spectral-tail learning). Under the assumptions of Theorem 1, the MSE satisfies for every τ ≥ 0,
MSE(T 2 + τ ) = 1 + O(ε) MSE(T 2 ) S d (τ ) + O ε 2 τ 2 1 d s (2) ⋆ .
(4.1)
In particular, S d (τ ) = (1 + o(1))S d (τ ), and S d (τ ) admits the asymptotics
S d (τ ) =          1 -16 d τ + O( τ 2 d ), β d τ ≪ 1, 1 -Γ(1 -1 a ) x d d + o( x d d ), 1 ≪ x d ≪ d, ≤ exp(-β d τ d -a ), x d ≳ d.
Thus, the plateau persists until T 2 , after which the MSE decays at a rate controlled by the spectral tail. The exponent a determines the slope of this decay on log-log scales, accounting for the scaling laws observed in Fig. 1.
this section cite: []

Section: PROOF OUTLINE
We briefly sketch the main ingredients of our analysis, deferring complete proofs to the appendix: Section C (Phase I), Section D (Phase II), and Section E (Phase III).
this section cite: []

Section: PHASE I: ESCAPE FROM MEDIOCRITY
In the anisotropic setting, the summary statistics (u, s) do not form a closed system; instead, they are coupled to an infinite hierarchy of correlations. Our strategy is to lift the dynamics to an infinitedimensional space, solve the system there, and then project back to obtain a closed Volterra representation for u(t), which we can analyze directly.
this section cite: []

Section: Step 1: Infinite-dimensional formulation.
The core difficulty is the infinite hierarchy of coupled correlations. To tame it, we collect them into the vector
U (t) = (u (1) (t), u (2) (t), . . . ) ⊤ ∈ H := C ∞ b (R ≥0 ; ℓ 2 ).
Let B be the right-shift operator on sequences, defined by (Bx) k := x k+1 and let S := (Bs ∞ ⋆ ) e ⊤ 1 be a rank one operator with
s ∞ ⋆ = (s (1) ⋆ , s(2)
⋆ , . . . ) ⊤ ∈ ℓ 2 and e 1 = (1, 0, 0, . . . ) ⊤ . Then the correlation dynamics (3.5) collapse into the compact operator form
U (t) = 4 1 -3s(t) B + 8S U (t).
(5.1) This is the key structural observation: the entire infinite system is generated by the shift operator B plus a rank-one perturbation S. Applying Duhamel's formula yields the following representation.
Lemma 1. The unique solution U ∈ C ∞ b (R ≥0 ; ℓ 2 ) of (5.1) satisfies, for all t ≥ 0,
U (t) = e 4B Θ(t) U 0 + 8 t 0 e 4B (Θ(t)-Θ(τ )) (Bs ∞ ⋆ ) u (1) (τ ) dτ,(5.2)
where u (1) (τ ) := ⟨e 1 , U (τ )⟩ = u(t), and Θ(t) := t 0 (1 -3s(τ )) dτ.
this section cite: []

Section: Step 2: Reduction to a Volterra equation.
Let δ > 0 be arbitrarily small, and let T 1 denote the stopping time such that s(t) ≤ δ for all t ≤ T 1 . On this time interval we may approximate Θ(t) ≈ t.
Projecting the representation (5.2) onto e 1 then yields the Volterra equation
u(t) = a 0 (t) + 8 t 0 K(t -τ ) u(τ ) dτ, (5.3) where a 0 (t) = i w i (0)w ⋆ i λ i e 4λit , K(t) = i (w ⋆ i ) 2 λ 2 i e 4λit .
Applying the Laplace transform gives û(p) = â0(p)
1-8 K(p) . The growth of u(t) is therefore governed by the rightmost pole of û(p) (see Section C and Chapter VII of Gripenberg et al. (1990)). Computing K(p) and solving the equation 1 = 8 K(p) yields the following. Lemma 2 (Exponential growth rate). The correlation u(t) grows at rate e ρtruet , where ρ true > 4λ 1 is the unique positive solution of 1 -8 K(ρ) = 0.
Step 3: Control of the higher-order correlations. A similar analysis shows that the influence of all higher-order terms is dominated by the leading mode u(t). Without loss of generality, we assume that u(t) grows positively (the case of negative growth is analogous). The key point is that at time T 1 , the second correlation u (2) (T 1 ) has the same sign as u(T 1 ), a fact that will be critical in the analysis of Phase IIa. Proposition 5 (Control of higher-order correlations). For all k ≥ 1 and all t ∈ [0, T 1 ], we have
-λ k 1 log d d e 4λ1t + 8 s (k) ⋆ t 0 u(τ ) dτ ≤ u (k) (t) ≤ λ k 1 log d d e 4λ1t + λ k-1 1 u(t) -u(0) .
(5.4) In particular, for d sufficiently large there exists a constant c 1 > 0 such that u (2) (T 1 ) ≥ c 1 .
this section cite: ['b27']

Section: PHASE II: CONVERGENCE OF SUMMARY STATISTICS
Since the proof techniques differ, we separate the analysis of Phase IIa and Phase IIb.
this section cite: []

Section: ANALYSIS OF PHASE IIA
The first step is to show that s(t), which at time T 1 is still small (s(T 1 ) = δ ′ 2 ), must increase up to the critical threshold 1/3. Indeed, from
ṡ(t) = 8 1 -3s(t) s (2) (t) + 16 u(t)u (2) (t),
both terms on the right-hand side are positive whenever s(t) ≤ 1/3, so s(t) is driven upward and crosses 1/3 in finite time.
The second step establishes stability beyond the threshold: once s(t) has passed 1/3, it cannot fall back below. Close to the boundary, the positive contribution 16u(t)u (2) (t) dominates the negative drift from the first term. A careful comparison shows that s(t) remains uniformly above 1/3 + δ for some constant δ > 0. The detailed proof of these two points is given in Section D.1.
this section cite: []

Section: ANALYSIS OF PHASE IIB
We now study the error ∆(t) := 1 -u(t) ≥ 0 for t ≥ T ′ 1 through the Volterra equation
∆(t) = b Θ (t) + t T ′ 1 K Θ (t, τ ) ∆(τ ) dτ,
where the source term b Θ is detailed in Appendix D.2. Since after T ′ 1 we have Θ(t) -Θ(τ ) ≤ -s 0 (t -τ ), the kernel K Θ is decreasing, so the integral contribution diminishes over time.
To control the positive part of b Θ , we split the spectrum at a cutoff λ c : directions with λ i < λ c form the tail, which is small but slow to learn, while λ i ≥ λ c form the head, easier but requiring more training time when λ c is small. Balancing these two effects yields the following result.
Proposition 6. Let ε ≳ d -a-1 2 , and set
T 2 (ε) = T ′ 1 + C 4s 0 ε - 2a a-1 log 1 ε ,
with C > 0 sufficiently large. Then ∆(T 2 ) ≤ ε.
Remark 5. The condition ε ≳ d -a-1 2 prevents ε from being too small. Otherwise the required time becomes much larger; for instance, ε = d -1 already forces T ≳ d a . Note that the exponent 2a a-1 decreases as a increases, so T 2 (ε) is shorter for faster spectral decay. Intuitively, for larger a, the lighter tail means fewer small eigenvalues to learn, so the error contracts more rapidly. In contrast, for a close to 1 the heavy tail creates many slow directions, delaying convergence. This trend is confirmed numerically in Section F.
this section cite: []

Section: PHASE III: SPECTRAL-TAIL LEARNING AND THE SCALING LAW
After T 2 , both u(t) and s(t) are close to one, so the coordinate dynamics reduce to
ẇi (t) ≈ 8λ i w ⋆ i -w i (t) , whose solution shows exponential relaxation of each w i toward w ⋆ i .
The resulting MSE is a weighted spectral average S d (τ ); under the heuristic π i ≈ d -1 , this reduces to the benchmark S d (τ ), whose asymptotics follow from classical sum-integral comparisons and reveal distinct decay regimes. The approximation error is controlled by a Duhamel (variation-of-constants) representation, together with Phase IIb bounds on 1 -u(t) and 1 -s(t). A Grönwall argument shows that these corrections remain small, so the full trajectory closely tracks the idealized one. Altogether, this yields the scaling law of Theorem 2, see Section F.1 for an illustration of the approximation.
this section cite: []

Section: DISCUSSION

this section cite: []

Section: References
Ref_id:b0 Title: The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks Year: (2022)
Ref_id:b1 Title: Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics Year: (2023)
Ref_id:b2 Title: Escaping mediocrity: how two-layer networks learn hard generalized linear models with sgd Year: (2023)
Ref_id:b3 Title: From high-dimensional & mean-field dynamics to dimensionless odes: A unifying approach to sgd in two-layers networks Year: (2023)
Ref_id:b4 Title: Repetita iuvant: Data repetition allows sgd to learn high-dimensional multi-index functions Year: (2024)
Ref_id:b5 Title: Scaling and renormalization in high-dimensional regression Year: (2024)
Ref_id:b6 Title: Learning in the presence of low-dimensional structure: A spiked random matrix perspective Year: (2023)
Ref_id:b7 Title: Explaining neural scaling laws Year: (2024)
Ref_id:b8 Title: High-dimensional limit theorems for sgd: Effective dynamics and critical scaling Year: (2022)
Ref_id:b9 Title: Learning quadratic neural networks in high dimensions Year: (2025)
Ref_id:b10 Title: On learning gaussian multi-index models with gradient flow part i: General properties and two-timescale learning Year: (2025)
Ref_id:b11 Title: A dynamical model of neural scaling laws Year: (2024)
Ref_id:b12 Title: Learning a single index model from anisotropic data with vanilla stochastic gradient descent Year: (2025)
Ref_id:b13 Title: Survey on algorithms for multi-index models Year: (2025)
Ref_id:b14 Title: Phase retrieval via wirtinger flow: Theory and algorithms Year: (1985)
Ref_id:b15 Title: Optimal rates for the regularized least-squares algorithm Year: (2007)
Ref_id:b16 Title: Advances in neural information processing systems Year: (2018)
Ref_id:b17 Title: Hitting the high-dimensional notes: An ode for sgd learning dynamics on glms and multi-index models Year: (2024)
Ref_id:b18 Title: Generalization error rates in kernel regression: The crossover from the noiseless to noisy regime Year: (2021)
Ref_id:b19 Title: The benefits of reusing batches for gradient descent in two-layer networks: Breaking the curse of information and leap exponents Year: (2024-07)
Ref_id:b20 Title: The nonsmooth landscape of phase retrieval Year: (2020)
Ref_id:b21 Title: Dimension-free deterministic equivalents and scaling laws for random feature regression Year: (2024)
Ref_id:b22 Title: Scaling laws and spectra of shallow neural networks in the feature learning regime Year: (2025)
Ref_id:b23 Title: Phase retrieval: From computational imaging to machine learning: A tutorial Year: (2023)
Ref_id:b24 Title: The nuclear route: Sharp asymptotics of erm in overparameterized quadratic networks Year: (2025)
Ref_id:b25 Title: Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup Year: (2019)
Ref_id:b26 Title: Modeling the influence of data structure on learning in neural networks: The hidden manifold model Year: (2020)
Ref_id:b27 Title: Volterra Integral and Functional Equations. Encyclopedia of Mathematics and its Applications Year: (1990)
Ref_id:b28 Title:  Year: (2017)
Ref_id:b29 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b30 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b31 Title: Scaling laws for gradient descent and sign descent for linear bigram models under Zipf's law Year: (2025)
Ref_id:b32 Title: Scaling laws in linear regression: Compute, parameters, and data Year: (2024)
Ref_id:b33 Title: Spectral method for phase retrieval: An expectation propagation perspective Year: (2021)
Ref_id:b34 Title: A solvable model of neural scaling laws Year: (2022)
Ref_id:b35 Title: On the impact of overparameterization on the training of a shallow neural network in high dimensions Year: (2024)
Ref_id:b36 Title: 4+3 phases of computeoptimal neural scaling laws Year: (2024)
Ref_id:b37 Title: Semigroups of linear operators and applications to partial differential equations Year: (2012)
Ref_id:b38 Title: Statistical optimality of stochastic gradient descent on hard learning problems through multiple passes Year: (2018)
Ref_id:b39 Title: Emergence and scaling laws in sgd learning of shallow neural networks Year: (2025)
Ref_id:b40 Title: Generalization properties of learning with random features Year: (2017)
Ref_id:b41 Title: Dynamics of on-line gradient descent learning for multilayer neural networks Year: (1995)
Ref_id:b42 Title: On-line learning in soft committee machines Year: (1995)
Ref_id:b43 Title: Optimization and generalization of shallow neural networks with quadratic activation functions Year: (2020)
Ref_id:b44 Title: Phase retrieval via randomized kaczmarz: theoretical guarantees. Information and Inference: A Year: (2019)
Ref_id:b45 Title: Online stochastic gradient descent with arbitrary initialization solves non-smooth, non-convex phase retrieval Year: (2023)
Ref_id:b46 Title: Ordinary differential equations and dynamical systems Year: (2012)
Ref_id:b47 Title: Phase diagram of stochastic gradient descent in high-dimensional two-layer neural networks Year: (2022)
Ref_id:b48 Title: Analyzing neural scaling laws in two-layer networks with power-law data spectra Year: (2024)
Ref_id:b49 Title: Kernel ridge regression under power-law data: spectrum and generalization Year: (2025)
Ref_id:b50 Title: On early stopping in gradient descent learning Year: (2007)
Ref_id:b51 Title: Online gradient descent learning algorithms Year: (2008)
