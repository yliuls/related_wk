Title: Graph-Smoothed Bayesian Black-Box Shift Estimator and Its Information Geometry
Abstract: Label shift adaptation aims to recover target class priors when the labelled source distribution P and the unlabelled target distributionClassical black-box shift estimators invert an empirical confusion matrix of a frozen classifier, producing a brittle point estimate that ignores sampling noise and similarity among classes. We present Graph-Smoothed Bayesian BBSE (GS-B 3 SE), a fully probabilistic alternative that places Laplacian-Gaussian priors on both target log-priors and confusion-matrix columns, tying them together on a label-similarity graph. The resulting posterior is tractable with HMC or a fast block Newton-CG scheme. We prove identifiability, N -1/2 contraction, variance bounds that shrink with the graph's algebraic connectivity, and robustness to Laplacian misspecification. We also reinterpret GS-B 3 SE through information geometry, showing that it generalizes existing shift estimators.

Section: Introduction
Modern machine-learning systems are rarely deployed in exactly the same environment in which they were trained. When the distribution of class labels drifts but the class-conditional features remain stable, a phenomenon known as label shift, even a high-capacity model can produce arbitrarily biased predictions [54,12,41,29,44,45,51]. Practical examples include sudden changes in click-through behaviour of online advertising, evolving pathogen prevalence in medical diagnostics, and seasonal increments of certain object categories in autonomous driving. Because re-labelling target data is often prohibitively expensive, methods that recover the new class priors from a small unlabelled sample are indispensable precursors to reliable downstream decisions.
A popular remedy, the Black-Box Shift Estimator (BBSE) is to keep a single, frozen classifier ĥ : X → Y trained on labelled source data and to link its predictions on the target domain to the unknown target priors through the confusion matrix C [39,7]. The resulting formulation converts density shift into a linear system q = Cq that can be solved in closed form, after plugging in (i) an empirical estimate C computed on a small labelled validation set and (ii) the empirical prediction histogram q measured on unlabelled target instances. The elegance of BBSE has made it the default baseline for label-shift studies [45,13,38,30,42,19].
Despite its popularity, the BBSE pipeline overlooks two sources of uncertainty that become debilitating in realistic, high-class-count regimes. (i) Finite-sample noise. Each column of C is estimated from at most a few hundred examples, so the matrix inversion layer can amplify small fluctuations into large errors on q. Regularised variants such as RLLS [7] or MLLS [18] damp variance but still return a point estimate whose uncertainty remains opaque to the user. (ii) Semantic structure. Classes in vision and language problems live on rich ontologies [35]: car and bus are more alike than car and daisy. Standard BBSE fits each class independently and cannot borrow statistical strength across such related labels, leading to particularly fragile estimates for rare classes.
We introduce Graph-Smoothed Bayesian BBSE (GS-B 3 SE), a fully probabilistic alternative that attacks both weaknesses in a single hierarchical model. The key idea is to couple both the target log-prior vector and the columns of the confusion matrix through a Gaussian Markov random field defined on a label-similarity graph. Graph edges are obtained once from off-the-shelf text or image embeddings, and the resulting Laplacian precision shrinks parameters of semantically adjacent classes towards each other. Sampling noise is handled naturally by Bayesian inference: we place Gamma-Laplacian hyper-priors on the shrinkage strengths and sample the joint posterior with either Hamiltonian Monte Carlo [8,9,16] or a fast block Newton-conjugate-gradient optimizer [23,10]. Moreover, we provide interpretation of GS-B 3 SE through the lens of information geometry [5,6]. The analysis of statistical procedures or algorithms in this framework is known to help us understand them better [4,1,40,28,27,25,26,2,24].
Contributions. i) We formulate the joint Bayesian model that simultaneously regularizes the target prior and the confusion matrix with graph-based smoothness, reducing variance without hand-tuned penalties (in Section 4). ii) We provide theoretical guarantees: (a) posterior identifiability, (b) N -1/2 contraction, (c) class-wise variance bounds that tighten with the graph's algebraic connectivity, and (d) robustness to Laplacian misspecification (in Section 5). Moreover, we provide interpretation of GS-B 3 SE through the lens of information geometry framework and it shows that our algorithm is a natural generalization of existing methods. iii) An empirical study on several datasets demonstrates that GS-B 3 SE produces sharper prior estimates and improves downstream accuracy after Saerens correction compared to state-of-the-art baselines (in Section 7).
2 Related Literature Classical estimators under label shift. Saerens et al. [47] proposed an EM algorithm that alternates between estimating the target prior and re-weighting posterior probabilities calculated by a fixed classifier. Lipton et al. [39] later formalised the Black-Box Shift Estimator (BBSE), showing that a single inversion of the empirical confusion matrix suffices when P (X | Y ) is preserved. Subsequent refinements introduced regularisation to cope with ill-conditioned inverses: RLLS adds an ℓ 2 penalty to the normal equations [7], while MLLS frames the problem as a constrained maximum-likelihood optimisation [18]. All three methods remain point estimators and ignore uncertainty in C.
Bayesian and uncertainty-aware approaches. Caelen [11] derived posterior credible intervals for precision and recall by coupling Beta priors with multinomial counts; Ye et al. [53] extended the idea to label-shift estimation under class imbalance. Most of these models assume that classes are a priori independent, so posterior variance remains high for rare labels. Our work instead imposes structured Gaussian Markov random field (GMRF) priors that borrow strength across semantically related classes.
Graph-based smoothing and Laplacian priors. In spatial statistics, Laplacian-Gaussian GMRFs are a standard device for sharing information among neighbouring regions [49]. Recent machine learning studies exploit the same idea for discrete label graphs: Alsmadi et al. [3] introduced a graph-Dirichlet-multinomial model for text classification, and Ding et al. [15] used graph convolutions to smooth class logits. We follow this line but couple both the prior vector and every confusion-matrix column to the same similarity graph, yielding a joint posterior amenable to HMC and Newton-CG.
this section cite: ['b53', 'b11', 'b40', 'b28', 'b43', 'b44', 'b50', 'b38', 'b6', 'b44', 'b12', 'b37', 'b29', 'b41', 'b18', 'b6', 'b17', 'b34', 'b7', 'b8', 'b15', 'b22', 'b9', 'b4', 'b5', 'b3', 'b0', 'b39', 'b27', 'b26', 'b24', 'b25', 'b1', 'b23', 'b46', 'b38', 'b6', 'b17', 'b10', 'b52', 'b48', 'b2', 'b14']

Section: Domain-shift benchmarks and failure modes.
Large-scale empirical studies such as WILDS [30] and Mandoline [13] describe how brittle point estimators become under distribution drift and class imbalance. Rabanser et al. [45] demonstrated that label-shift detectors without calibrated uncertainty frequently produce over-confident but wrong alarms. By delivering credible intervals whose width shrinks with the graph's algebraic connectivity, our method directly tackles this shortcoming.
Positioning of this work. GS-B 3 SE unifies three strands of research: (i) black-box label-shift estimation, (ii) Bayesian confusion-matrix modelling, and (iii) graph-structured smoothing. To justify our proposed method, we show posterior identifiability and N -1/2 contraction, and derive variance bounds that scale with λ 2 (L). Empirically, our method plugs seamlessly into existing shift-benchmark pipelines, providing calibrated uncertainty absent from earlier regularised or EM-style alternatives.
this section cite: ['b29', 'b12', 'b44']

Section: Preliminarily
Problem Setting Let X be an input space and Y = {1, . . . , K} be a label set, where K ≥ 2 is the number of classes. For source and target distributions P and Q, the label shift assumption states that the class-conditional feature laws remain unchanged while the class priors may differ:
P (X | Y = i) = Q(X | Y = i), ∀i ∈ Y, and P (Y = i) =:pi ̸ = Q(Y = i) =:qi in general. (1)
Let p = (p 1 , . . . , p K ) ⊤ and q = (q 1 , . . . , q K ) ⊤ , where i p i = i q i = 1.
Black-Box Shift Estimator Train once, on source data, an arbitrary measurable classifier ĥ : X → Y. Denote its confusion matrix under P by C ∈ (0, 1) K×K , where C j,i = Pr P ĥ(X) = j | Y = i and j C j,i = 1. Notice that C depends only on the source distribution and can be estimated on a held-out validation set with known labels. Write the empirical estimate as C. Let m labelled source-validation points (x i , y i ) m i=1 be used to form C. Apply the same fixed ĥ to unlabeled target instances {x ′ t } n ′ t=1 of size n ′ . Let qj = Pr Q ĥ(X) = j , q = (q 1 , . . . , qK ) ⊤ . Because the class-conditionals are shared, Bayes' rule gives
Pr Q ĥ(X) = j = K i=1 Pr Q ĥ(X) = j | Y = i • q i = K i=1 C j,i q i ,(2)
and in vector form, it can be written as q = Cq. Eq. 2 is the identifiability equation for label shift.
Assume C is invertible or full column rank. Then the population target prior is q = C -1 q. Under this observation, BBSE framework [39] converts black-box classifier predictions on unlabeled target data into an estimate of the unknown target class prior by solving the linear system.
this section cite: ['b38']

Section: Methodology
The usual BBSE treats the confusion matrix C as fixed. In practice C is estimated from a finite validation set and is itself ill-conditioned when some classes are rare. To address this problem, we consider extending BBSE framework by the joint Bayesian model for both confusion matrix and target priors with the graph coupling. Let n S i be a number of source examples with label i, and n S i = (n S 1,i , . . . , n S K,i ) ⊤ be the counts of ĥ(X) = j among those n S i . Also, let ñ = (ñ 1 , . . . , ñK ) ⊤ be the counts of ĥ(X) = j on unlabeled target data. For each true class i, n S i | C ∼ Multi n S i , C :,i , where C :,i is the i-column of C and Multi(•, •) is the multinomial distribution. Similarly,
ñ | C, q ∼ Multi (n ′ , Cq) .
The complete-data likelihood factorises
p (data | C, q) = K i=1 Multi n S i ; n S i , C :,i Multi ( ñ; n ′ , Cq) .
We now consider to utilize similarity information between classes. Let G = (Y, E, W ) be a similarity graph on labels with the weight matrix W , E is the edges and L is the graph Laplacian. In G, each vertex corresponds to a class label; an edge (i, j) ∈ E indicates that labels i and j are semantically or visually similar. Weights W ij ∈ [0, 1] quantify that similarity, with W ij = 0 when no edge is present. In our methodology, we use the unnormalised Laplacian L = D -W where D ii = j W ij . Because we enforce connectivity, L has exactly one zero eigenvalue, making λ 2 (L) (the algebraic connectivity) strictly positive as required by our theory. Introduce log-odds vector
θ i = log q i - 1 K K k=1 log q k , θ ∈ R K , θ ⊤ 1 = 0,
and consider the following Gaussian Markov random field (GMRF) prior
p(θ | τ q ) ∝ exp - τ q 2 θ ⊤ Lθ , τ q ∼ Gamma(a q , b q ),(3)
where a q , b q > 0 are hyper-parameters. A Laplacian-based precision shrinks log-odds differences along graph edges, promoting smooth class priors across semantically similar labels [31], and recover q = softmax(θ).
Treat each column C :,i as a latent simplex vector with Dirichlet-log-normal hierarchy:
i) Latent log-odds ϕ i ∈ R K , ϕ ⊤ i 1 = 0. ii) Conditional prior p(ϕ i | τ C ) ∝ exp -τ C 2 ϕ ⊤ i Lϕ i .
All ϕ i share the same Laplacian L over predicted labels so that columns corresponding to neighbouring predicted classes exhibit similar shape.
iii) Transformation to the simplex C j,i = exp ϕj,i K ℓ=1 exp ϕ ℓ,i for i = 1, . . . , K.
iv) Hyper-prior τ C ∼ Gamma(a C , b C ), with a C , b C > 0.
The resulting distribution on each C :,i is a logistic-Normal on the simplex and it reduces to an ordinary Dirichlet when L = 0 but gains graph-coupled precision for L ̸ = 0. The full hierarchical model is as follows.
τ q ∼ Gamma(a q , b q ), θ | τ q ∼ N (0, (τ q L) † ), q = softmax(θ), τ C ∼ Gamma(a C , b C ), ∀i : ϕ i | τ C ∼ N (0, (τ C L) † ), C :,i = softmax(ϕ i ), ∀i : n S i | C ∼ Multi(n S i , C :,i ), ñ | C, q ∼ Multi(n ′ , Cq).
Here, the Moore-Penrose pseudoinverse L † appears because L is singular, and the constraint θ ⊤ 1 = 0 ensures uniqueness.
For the posterior inference, consider the following log-joint distribution.
All terms are differentiable, enabling Hamiltonian Monte Carlo (HMC) in the unconstrained variables. Because Eq. 4 is concave in each block after reparameterization, a block-Newton scheme alternates, i) update {ϕ i } K i=1 by one Newton-CG step using sparse Laplacian Hessian, ii) update θ likewise, iii) closed-form updates for τ C , τ q from Gamma posteriors. Convergence is super-linear due to the strict convexity induced by the Laplacian energies. The posterior predictive distribution of the confusion-weighted target counts is
ñ * | data = Multi(n ′ , Cq)p(C, q | data)dCdq.(5)
Credible intervals for each q i reflect both sampling noise and model-induced graph smoothing, an advantage over plug-in BBSE. If no prior similarity information exists one may default to W ij = 1{i = j}, in which case our model reduces to an independent logistic-Normal prior and all theoretical guarantees still hold.
this section cite: ['b30']

Section: Relationship to Existing Work
• Replaces the point estimate of BBSE C -1 q with a full posterior, relating Bayesian confusion matrix treatments [11].
• Laplacian GMRFs generalise classical Dirichlet priors by borrowing strength along graph edges, extending recent graph-Dirichlet-multinomial models [31].
• When L = 0 and Gamma hyper-priors degenerate to delta masses, our hierarchical model reduces exactly to deterministic BBSE.
this section cite: ['b10', 'b30']

Section: Theory
This section provides the theoretical foundations of the proposed method. See Appendix A for the detailed proofs. First, the following lemma on identifiability is introduced. Although an analogous statement has been implicitly argued in the prior work [39], the full proof is included in the Appendix A to make this study self-contained. Lemma 1. Let C and C ′ be two column-stochastic matrices with strictly positive entries: C j,i > 0,
C ′ j,i > 0, K j=1 C j,i = K j=1 C ′ j,i = 1 for 1 ≤ i ≤ K.
In addition, assume C and C ′ are invertible, or equivalently, det C ̸ = 0 and det C ′ ̸ = 0. For any deterministic sample sizes n S i ∈ {1, 2, . . . } and n ′ ∈ {1, 2, . . . }, define the data-generating distributions
N S i | C ∼ Multi(n S i , C :,i ), N S i | C ′ ∼ Multi(n S i , C ′ :,i ), Ñ | C, q ∼ Multi(n ′ , Cq), Ñ | C ′ , q ′ ∼ Multi(n ′ , C ′ q ′ ).
Suppose that, for every choice of the sample sizes {n S i=1 } K i=1 and n ′ ,
{N S i } K i=1 , Ñ d = {N S ′ i } K i=1 , Ñ ′ , as random vectors in N K 2 +K
, where the left-hand side is generated by (C, q) and the right-hand side by (C ′ , q ′ ). Then, C = C ′ and q = q ′ . Lemma 1 implies that the mapping (q, C) → {{n S i }, ñ} is injective up to measure-zero label permutations when the graph is connected and all source classes appear.
Let ∆ K-1 be the (K -1)-dimensional probability simplex:
∆ K-1 := q = (q 1 , . . . , q K ) ∈ R K : q i ≥ 0 for every i, K i=1 q i = 1 .
The following lemma provides the support condition needed in statements described later. Lemma 2. Let (q 0 , C 0 ) be the true parameter pair, where q 0 ∈ ∆ K-1 and C 0 ∈ (0, 1) K×K with det C 0 ̸ = 0. Define the Euclidean small ball as B ϵ (q 0 , C 0 ) := {(q, C) : ∥q -q 0 ∥ 2 < ϵ, ∥C -C 0 ∥ F < ϵ} , for some radius ϵ > 0 small enough that all vectors in the ball stay strictly inside the simplex. Then, for every ϵ, the joint prior distribution Π on (q, C) assigns strictly positive mass to the ball:
Π (B ϵ (q 0 , C 0 )) > 0.
This lemma states that positivity of Gaussian density and the smooth bijection yield the positive push-forward density, and it is the classical strategy used for logistic-Gaussian process priors in density estimation [50].
Lemmas 1, 2 and the classical results from Ghosal et al. [20], Van der Vaart [52] gives the following statement. Proposition 1. Let K ≥ 2 be fixed and (q 0 , C 0 ) be the true parameters pair with q 0 ∈ ∆K-1 and C 0 ∈ (0, 1) K×K , where ∆K-1 is the interior of ∆ K-1 , and det C 0 ̸ = 0. Suppose that the data consist of {N S i } K i=1 and Ñ where conditionally on (q 0 , C 0 ),
N S i ∼ Multi n S i , C 0,i , Ñ ∼ Multi (n ′ , C 0 q 0 ) .
Also suppose that sample sizes diverge with the single index: N := n ′ + K i=1 n S i → ∞, and min i n S i → ∞. Then, for every ϵ > 0, Π (B c ϵ | data)
P (q 0 ,C 0 ) -----→ N →∞ 0.
Under the same assumption in Proposition 1, the following statements about the posterior contraction rate are obtained. Theorem 1. Let the data-generating model, true parameter pair (q 0 , C 0 ), and diverging sample sizes N = n ′ + K i=1 n S i → ∞ satisfy the setup spelled out before Proposition 1. Define the Euclidean radius ϵ N := M/ √ N , M > 0 arbitrary but fixed. Let
B c N = {(q, C) : ∥q -q 0 ∥ 2 + ∥C -C 0 ∥ F > ϵ N } .
Under Lemma 1 and 2, the joint posterior Π(• | data) for the Laplacian-Gaussian hierarchy satisfies
Π (B c N | data) P (q 0 ,C 0 ) -----→ N →∞ 0.
That is, the posterior contracts around the truth at the parametric rate N -1/2 . Corollary 1. Retain the setting and notation of Theorem 1. For each class i, write
Var N (q i ) := Var (q i | data of size N ) ,
under the joint posterior Π(• | data). Let L be the connected-graph Laplacian used in the GMRF prior and let λ 2 (L) := min {λ > 0 : λ is an eigenvalue of L} be its algebraic connectivity. Assume the hyper-parameter τ q is fixed, or sampled from a Gamma prior independent of N . Then, there exists a constant C > 0, depending only on the true (q 0 , C 0 ) and on K, such that for every sample size N large enough,
Var N (q i ) ≤ C λ 2 (L)N
, ∀i ∈ {1, . . . , K}.
Finally, we can show the following statement about the robustness to graph Laplacian misspecification. Proposition 2. Let L 0 be the true Laplacian, and
F 0 := diag(C 0 q 0 ) -(C 0 q 0 )(C 0 q 0 ) ⊤ ⪰ 0 be the Fisher information of θ in the target multinomial likelihood. For L ̸ = L 0 , let θN := E[θ | data]
be the posterior mean of θ under the misspecified prior. Then, for all sample sizes N large enough,
∥ θN -θ 0 ∥ 2 ≤ (N F 0 + τ q L) -1 2 sampling + prior precision τ q (L -L 0 )θ 0 2 graph-misspecification bias +O P (N -1 ).(6)
In particular,
∥ θN -θ 0 ∥ 2 ≤ τ q N λ min (F 0 ) + τ q λ 2 (L) ∥(L -L 0 )θ 0 ∥ 2 + O P (N -1 ),(7)
where λ min (F 0 ) > 0 and λ 2 (L) > 0 are, respectively, the smallest eigenvalue of F 0 and the algebraic connectivity of L.
Thus, we can see that the bias decays as N -1 when L ̸ = L 0 and if the graphs coincide the leading term vanishes and the posterior mean is unbiased up to the usual N -1/2 noise. Moreover, Proposition 2 states that a larger algebraic connectivity λ 2 (L) reduces bias, emphasising the benefit of rich similarity structures.
this section cite: ['b38', 'b49', 'b19', 'b51']

Section: Interpretation via Information Geometry
The basic notations of information geometry used in this section are summarized in Appendix B. The K -1 simplex ∆ K-1 := {q > 0 :
1 ⊤ K q = 1} is a Riemannian manifold when equipped with the Fisher-Rao metric g q (v, w) = K i=1 viwi qi , for v, w ∈ T q ∆ K-1 , where T q ∆ K-1 := {v : 1 ⊤ v = 0}
is the tangent space. The natural potential on this manifold is minus entropy ψ(q) = i q i log q i whose Euclidean gradient is the centred log-odds vector θ used in the previous section. These facts allow us to cast GS-B 3 SE as a Riemannian penalised likelihood. The dual affine coordinates are m-coordinates q i (mixture parameters) and e-coordinates θ i = log q i -1 K j log q j (centred log-odds). The convex potential ψ(q) = K i=1 q i log q i is minus Shannon entropy and satisfies ∇ Euc q ψ(q) = θ; together (ψ, θ) endow ∆ K-1 with the classical dually-flat structure of information geometry [5,6]. See standard textbooks for detailed explanation of concepts in differential geometry and Riemannian manifold [36,37,34,17,43,21,33].
Denote by r = ñ/n ′ and M = Cq : q ∈ ∆ K-1 the empirical prediction histogram and the m-flat sub-manifold induced by the frozen classifier. The negative log-posterior derived in Section 4 can be written as
F (q) = n ′ D KL r ∥ Cq + τ q 2 θ ⊤ Lθ + const. (8
)
Thus Eq. ( 8) is a sum of an m-convex and an e-convex potential, so it is geodesically convex under the Fisher-Rao metric (see Table 1). Theorem 2 (Geodesic convexity of F ). For every q ∈ ∆K-1 the Riemannian Hessian of F satisfies
Hess FR q F ⪰ n ′ λ min (F 0 ) + τ q λ 2 (L) g q ,
where
F 0 = diag(Cq) -(Cq)(Cq) ⊤
is the Fisher information of the multinomial likelihood and λ 2 (L) the algebraic connectivity of the label graph. Hence F is α-strongly geodesically convex with
α = n ′ λ min (F 0 ) + τ q λ 2 (L) > 0.
The proof in Appendix A explicitly decomposes any tangent direction into an m-straight and an e-straight component and shows that the lower bound remains positive because both components contribute additively.
this section cite: ['b4', 'b5', 'b35', 'b36', 'b33', 'b16', 'b42', 'b20', 'b32']

Section: Natural-Gradient Dynamics
The natural gradient of F is
grad FR F (q) = g -1 q ∇ (m) F (q) = q ⊙ ∇ q F -(∇ q F ) ⊤ q ,
where ⊙ is component-wise product. The associated flow q(t) = -grad FR F q(t) is the steepest-descent curve in the Fisher-Rao geometry.
Proposition 3 (Natural-gradient flow of the penalised objective). Under the Fisher-Rao metric
g q (v, w) = K i=1 v i w i /q i the natural gradient grad FR F (q) of F is grad FR F (q) = diag(q) n ′ C ⊤ 1 -r r(q) + τ q L θ ,(9)
where the division is element-wise. Consequently the un-constrained natural-gradient flow
qt = -grad FR F (q t ) = -diag(q t ) n ′ C ⊤ 1 -r r(qt) + τ q L θ t (10
)
preserves the simplex and coincides with the replicator-Laplacian dynamical system: qt,j = -q t,j n ′ [C ⊤ (1 -r/r)] + τ q [Lθ t ] j .
Remarks i) Role of Laplacian When the Laplacian term is absent (τ q = 0), the flow reduces to the classical replicator equation that drives every class-probability q j proportionally to the (signed) log-likelihood residual [C ⊤ (1 -r/r)] j . The graph-Laplacian contribution -τ q q j [Lθ] j plays the role of a mutation / diffusion force that mixes mass along edges of the label graph and prevents degenerate solutions.
ii) Role of the algebraic connectivity. From Theorem 2 the strong-convexity modulus is α = n ′ λ min (F 0 ) + τ q λ 2 (L). Along the flow we have d dt F (q t ) = -∥grad FR F (q t )∥ 2 gq t ≤ -2α(F (q t ) -F * ), so F decays exponentially fast with rate proportional to the algebraic connectivity λ 2 (L); a denser-connected label graph therefore accelerates convergence.
iii) Link to Saerens EM correction. If we freeze the confusion matrix and drop the Laplacian term, the stationary condition C ⊤ (r/r) = 1 is exactly the fixed point solved (iteratively) by the Saerens EM method [47].
Table 2: Baseline methods and their key ideas.
this section cite: ['b46']

Section: Method Key idea
BBSE [39] Solve Ĉ q = ŷ with the empirical confusion matrix (no re-training).
EM [47] Expectation-Maximization that iteratively re-estimates priors and re-weights posteriors.
RLLS [7] Adds an ℓ 2 penalty to the BBSE normal equations to control variance for small n ′ .
this section cite: ['b38', 'b46', 'b6']

Section: MLLS [18]
Maximum-likelihood estimation of the label-ratio vector; unifies BBSE & RLLS and optimizes q directly.
this section cite: []

Section: GS-B 3 SE (ours)
Joint Bayesian inference of both target priors q and confusion matrix C. The hierarchical model couples classes along a label-similarity graph, shrinking estimates in low-count regimes and yielding full posterior credible intervals.
this section cite: []

Section: Dual Projections and the Pythagorean Identity
Let Π m (r) be the m-projection of the data onto M and Π e (q 0 ) the e-projection of the hyper-prior center onto the same manifold. At the optimum q ⋆ we have Π m (r) = Π e (q 0 ) = Cq ⋆ , and the generalized Pythagorean theorem [5] gives
D KL r ∥ q 0 = D KL r ∥ Cq ⋆ + D KL Cq ⋆ ∥ q 0 ,
this section cite: ['b4']

Section: Experiments

this section cite: []

Section: Experimental Protocol and Implementation
Datasets and synthetic label shifts. We evaluate on MNIST (K = 10) [14], CIFAR-10 (K = 10) and CIFAR-100 (K = 100) datasets [32]. For each dataset we treat the official training split as the source domain and the official test split as the pool from which an unlabelled target domain is drawn. Source class-priors are kept uniform p = (1/K, . . . , 1/K). Target priors are deliberately perturbed:
q = Dirichlet(α × u K ) (MNIST), i -b K j=1 j -b , i = 1, .
. . , K (CIFAR-10 and CIFAR-100), where u K = (1, . . . , K) ⊤ . In our experiments, we set α = 0.05 and b = 1.1. The procedure is: i) Source set: Sample 10, 000 instances from the training partition according to p and train a backbone classifier (ResNet-18 [22,48]) for 100 epochs with standard data-augmentation. ii) Validation set: Hold out 5, 000 labelled source instances, stratified by p, to estimate the empirical confusion matrix C. iii) Target set: Draw n ′ = 10, 000 unlabelled instances from the test partition using probabilities q. These labels are revealed only for evaluation.
this section cite: ['b13', 'b31', 'b21', 'b47']

Section: Graph construction on labels.
For every dataset we embed the class names with the frozen CLIP ViT-B/32 text encoder [46], obtain {e i } K i=1 ⊂ R 512 , |e i | 2 = 1, and build a k-nearest-neighbour graph E = {(i, j) | e j is among the k nearest neighbors of e i } , k = 4 (K = 10), 8 (K = 100).
Edge weights are W ij = exp -∥e i -e j ∥ 2 2 /σ 2 with σ set to the median pairwise distance inside E. The resulting k-NN graph is connected, so its unnormalised Laplacian L = D -W satisfies λ 2 (L) > 0. For MNIST, where class names are single digits, we instead construct E from 4-NN in the Euclidean space of 128-d penultimate-layer features averaged over the training images.
Hyper-priors and inference. Gamma hyper-priors: a q = b q = a C = b C = 1, giving vague Gamma(1, 1) on τ q and τ C . Four independent HMC chains, each with 500 warm-up (NUTS) and 1,000 posterior iterations; leap-frog step-size adaptively tuned. Block Newton-CG inner optimizer: Table 3: Label shift estimation and downstream performance. Lower is better for ∥ q -q∥ 1 ; higher is better for post-correction accuracy. Best results are bold. ± shows one bootstrap standard error. (1 000 resamples).
this section cite: ['b45']

Section: MNIST (K=10)
CIFAR-10 (K=10) CIFAR-100 (K=100)
Method ∥ q -q∥1 ↓ Acc ↑ ∥ q -q∥1 ↓ Acc ↑ ∥ q -q∥1 ↓ Acc ↑
BBSE 0.038 ± 0.007 0.942 ± 0.002 0.112 ± 0.015 0.781 ± 0.004 1.62 ± 0.05 0.690 ± 0.006 EM 0.052 ± 0.015 0.935 ± 0.008 0.194 ± 0.033 0.732 ± 0.012 2.10 ± 0.14 0.632 ± 0.026 RLLS 0.016 ± 0.004 0.959 ± 0.003 0.072 ± 0.010 0.803 ± 0.004 0.92 ± 0.03 0.712 ± 0.006 MLLS 0.010 ± 0.003 0.963 ± 0.002 0.052 ± 0.008 0.812 ± 0.004 0.71 ± 0.03 0.734 ± 0.006 GS-B 3 SE 0.002 ± 0.001 0.986 ± 0.002 0.025 ± 0.004 0.844 ± 0.003 0.22 ± 0.02 0.783 ± 0.005 tolerance 10 -4 , at most eight iterations per Newton step, stop when the relative change of the joint log-density falls below 10 -3 . All routines implemented in PyMC and run on a single NVIDIA T4.
Baselines. We compare against a) plug-in BBSE [39], b) the EM-style Saerens re-weighting [47], c) RLLS [7] with ℓ 2 -regularisation and d) MLLS [18] tuned on a held-out split. All baselines receive the same C and target predictions ĥ(x). Table 2 summarizes the baseline methods and their key ideas, including our method.
Evaluation. We report prior-error | q -q| 1 and downstream accuracy after Saerens likelihood correction using the estimated priors. Significance is assessed with 1,000 paired bootstrap resamples of the target set.
this section cite: ['b38', 'b46', 'b6', 'b17']

Section: Main Empirical Findings
Table 3 compares
this section cite: []

Section: Better downstream accuracy.
Feeding the estimated priors into Saerens post-processing improves final accuracy in proportion to the quality of the prior. GS-B 3 SE attains 0.986 on MNIST, 0.844 on CIFAR-10 and 0.783 on CIFAR-100-absolute gains of +2.3,pp, +3.2,pp and +4.9,pp over the strongest non-Bayesian competitor (MLLS) on the respective datasets.
this section cite: []

Section: Conclusion
We presented GS-B 3 SE, a graph-smoothed Bayesian generalization of the classical black-box shift estimator. By tying both the target prior q and every column of the confusion matrix C together through a Laplacian-Gaussian hierarchy, the model simultaneously i) shares statistical strength across semantically related classes, ii) quantifies all uncertainty arising from finite validation and target samples, and admits scalable inference with either HMC or a Newton-CG variational surrogate. We proved that the resulting posterior is identifiable, contracts at the optimal N -1/2 rate, and that its class-wise variance decays inversely with the graph's algebraic connectivity λ 2 (L). A robustness bound further shows that even with a misspecified graph the bias vanishes as N -1 . Because our approach is a pure post-processing layer that needs only a frozen classifier, a tiny labelled validation set, and a pre-computed label graph, it can be retro-fitted to virtually any deployed model.
Limitations and future work. i) Our current graph is built from CLIP or feature embeddings; learning the graph jointly with the posterior could adapt it to the task. ii) Although inference is already tractable, further speed-ups via structured variational approximations would make GS-B 3 SE attractive for extreme-label settings. iii) As declared in Section 7.1, our experiments used a single NVIDIA T4; scaling to larger datasets remains future work.
this section cite: []

Section: References
Ref_id:b0 Title: The e-pca and m-pca: Dimension reduction of parameters by information geometry Year: (2004)
Ref_id:b1 Title: Information geometry of contrastive divergence Year: (2008)
Ref_id:b2 Title: Hybrid topic modeling method based on dirichlet multinomial mixture and fuzzy match algorithm for short text clustering Year: (2024)
Ref_id:b3 Title: Natural gradient works efficiently in learning Year: (1998)
Ref_id:b4 Title: Information geometry and its applications Year: (2016)
Ref_id:b5 Title: Methods of information geometry Year: (2000)
Ref_id:b6 Title: Regularized learning for domain adaptation under label shifts Year: (2019)
Ref_id:b7 Title: A conceptual introduction to hamiltonian monte carlo Year: (2017)
Ref_id:b8 Title: Hamiltonian monte carlo for hierarchical models Year: (2015)
Ref_id:b9 Title: A combined conjugate-gradient quasi-newton minimization algorithm Year: (1978)
Ref_id:b10 Title: A bayesian interpretation of the confusion matrix Year: (2017)
Ref_id:b11 Title: Word sense disambiguation with distribution estimation Year: (2005)
Ref_id:b12 Title: Mandoline: Model evaluation under distribution shift Year: (2021)
Ref_id:b13 Title: The mnist database of handwritten digit images for machine learning research Year: (2012)
Ref_id:b14 Title: Class-imbalanced graph convolution smoothing for hyperspectral image classification Year: (2024)
Ref_id:b15 Title: Hybrid monte carlo Year: (1987)
Ref_id:b16 Title: Riemannian geometry Year: (1997)
Ref_id:b17 Title: A unified view of label shift estimation Year: (2020)
Ref_id:b18 Title: Rlsbench: Domain adaptation under relaxed label shift Year: (2023)
Ref_id:b19 Title: Convergence rates of posterior distributions Year: (2000)
Ref_id:b20 Title:  Year: (2012)
Ref_id:b21 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b22 Title: Methods of conjugate gradients for solving linear systems Year: (1952)
Ref_id:b23 Title: Geometry of em and related iterative algorithms Year: (2024)
Ref_id:b24 Title: Generalized t-sne through the lens of information geometry Year: (2021)
Ref_id:b25 Title: Density ratio estimation via sampling along generalized geodesics on statistical manifolds Year: (2025)
Ref_id:b26 Title: α-geodesical skew divergence Year: (2021)
Ref_id:b27 Title: Information geometrically generalized covariate shift adaptation Year: (2022)
Ref_id:b28 Title: A short survey on importance weighting for machine learning Year: (2024)
Ref_id:b29 Title: Wilds: A benchmark of in-the-wild distribution shifts Year: (2021)
Ref_id:b30 Title: Discrete parametric graphical models with dirichlet type priors Year: (2023)
Ref_id:b31 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b32 Title: Differential geometry Year: (2015)
Ref_id:b33 Title: Differential and Riemannian manifolds Year: (1995)
Ref_id:b34 Title: Adjusting the outputs of a classifier to new a priori probabilities may significantly improve classification accuracy: Evidence from a multi-class problem in remote sensing Year: (2001)
Ref_id:b35 Title: Riemannian manifolds: an introduction to curvature Year: (2006)
Ref_id:b36 Title: Introduction to Riemannian manifolds Year: (2018)
Ref_id:b37 Title: A comprehensive survey on test-time adaptation under distribution shifts Year: (2025)
Ref_id:b38 Title: Detecting and correcting for label shift with black box predictors Year: (2018)
Ref_id:b39 Title: Information geometry of u-boost and bregman divergence Year: (2004)
Ref_id:b40 Title: Continuous target shift adaptation in supervised learning Year: (2016)
Ref_id:b41 Title: Label shift adapter for testtime adaptation under covariate and label shifts Year: (2023)
Ref_id:b42 Title: Riemannian geometry Year: (2006)
Ref_id:b43 Title: Dataset shift in machine learning Year: (2022)
Ref_id:b44 Title: Failing loudly: An empirical study of methods for detecting dataset shift Year: (2019)
Ref_id:b45 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b46 Title: Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure Year: (2002)
Ref_id:b47 Title: Deep residual learning for image recognition: A survey Year: (2022)
Ref_id:b48 Title: First steps with mrf smooths Year: (2017)
Ref_id:b49 Title: Posterior consistency of logistic gaussian process priors in density estimation Year: (2007)
Ref_id:b50 Title: Unbiased look at dataset bias Year: (2011)
Ref_id:b51 Title:  Year: (2000)
Ref_id:b52 Title: Label shift estimation for class-imbalance problem: A bayesian approach Year: (2024)
Ref_id:b53 Title: Domain adaptation under target and conditional shift Year: (2013)
