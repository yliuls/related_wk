Title: Sharper Convergence Rates for Nonconvex Optimisation via Reduction Mappings
Abstract: Many high-dimensional optimisation problems exhibit rich geometric structures in their set of minimisers, often forming smooth manifolds due to over-parametrisation or symmetries. When this structure is known, at least locally, it can be exploited through reduction mappings that reparametrise part of the parameter space to lie on the solution manifold. These reductions naturally arise from inner optimisation problems and effectively remove redundant directions, yielding a lowerdimensional objective. In this work, we introduce a general framework to understand how such reductions influence the optimisation landscape. We show that well-designed reduction mappings improve curvature properties of the objective, leading to better-conditioned problems and theoretically faster convergence for gradient-based methods. Our analysis unifies a range of scenarios where structural information at optimality is leveraged to accelerate convergence, offering a principled explanation for the empirical gains observed in such optimisation algorithms.

Section: Introduction
First-order gradient methods are the workhorse for large-scale optimisation in machine learning and data science due to their simplicity and scalability. However, the objective functions encountered in these settings often exhibit highly non-convex and intricate loss landscapes, stemming from various factors such as over-parametrisation, compositions of nonlinear functions, and underlying data distributions, to name a few [1][2][3][4]. Despite this complexity, gradient-based methods perform remarkably well in practice, achieving local linear convergence under mild regularity conditionssuch as the widely studied Polyak-Łojasiewicz (PŁ) condition [5][6][7][8][9]. This apparent tension between theoretical difficulty and empirical success motivates a deeper understanding of the geometry of loss landscapes and its role in shaping optimisation dynamics.
Let us consider the geometry of the solution spaces, which are shown to have manifold-like structures, rather than isolated points for many machine learning problems due to over-parametrisation, symmetries, or latent invariances [10,6]. Such structured sets of minimisers are not merely theoretical curiosities-they naturally arise in a variety of real-world applications. For instance, in deep neural networks, over-parametrisation often leads to entire manifolds of local minima [10] related by symmetries [11][12][13]. Furthermore, recent findings in neural collapse [14] reveal that optimal solutions often exhibit highly regular, symmetric, low-dimensional structures across layers [15][16][17][18][19][20]. In matrix factorisation problems [21] such as dictionary learning [22][23][24][25], low-rank matrix completion [26,27], and tensor decomposition problems [28,29], solutions are only identifiable up to scaling or orthogonal
S x 1 x 2 Ambient Surface of f f (x1, x2) = x 2 1 + 2(x2 -x1) 2 F1(x1) = f (x1, Ψ1(x1)) M F 1 = {(x1, x1) : x1 ∈ R} F2(x1) = f (x1, Ψ2(x1)) M F 2 = {(x1, 0) : x1 ∈ R} F3(x1) = f (x1, Ψ3(x1))
M F 3 = {(x1, x1 + 2 sin(x1)) : x1 ∈ R} Figure 1: Illustration of how well-designed reduction mappings iron out worst-case curvature. The opaque surface depicts the graph of the function f : R 2 → R, lifted above the ambient domain (grey plane) for visualisation purposes. The function has a single global minimum S = {(0, 0)}. In general, S can be a set of non-isolated points. The blue curve shows the restriction of f along the mapping Ψ 1 : x 1 → x 1 , where the high-curvature quadratic component cancels, yielding a flatter profile in x 1 . In contrast, the orange curve corresponds to the mapping Ψ 2 : x 1 → 0, which preserves most of the steep curvature of f . The grey-green curve traces the restriction along a nonlinear sinusoidal mapping-an example of a poorly designed reduction, which introduces additional curvature into the problem. Dashed curves on the ambient domain represent the images of these mappings as one-dimensional submanifolds M. The submanifolds have a non-empty intersection with S.
transformations [30], reflecting the inherent symmetries of these models. Similar invariances exist in problems such as phase retrieval [31][32][33][34] and blind deconvolution [35][36][37][38][39][40]. These symmetries induce structured sets of minimisers, often forming smooth manifolds or discrete equivalence classes, reflecting invariance under these transformations [30]. These examples all share a common theme: the objective exhibits invariances that endow the set of solutions with rich geometric structure.
In this work, we study how the geometric structure of the solution set can be systematically exploited to accelerate convergence. We focus on reduction mappings-reparametrisations that encode known components of the solution manifold, and study their effect on the local curvature. In practice, these reductions often arise from inner optimisation problems, thereby reformulating the original problem into a bi-level optimisation problem and yielding a lower-dimensional objective. While intuitively promising, not all such reduction mappings are beneficial. Even if optimal in value, a poorly designed reduction can distort local curvature and hinder convergence. We develop a rigorous framework to identify when and how these mappings lead to provable improvements in outer iteration complexity for gradient descent.
To build geometric intuition, Figure 1 illustrates how different reduction mappings affect the local curvature of the objective along their respective subspaces. For more details on this example, refer to Appendix A. While some mappings reveal a well-conditioned, flattened profile, others retain or even exaggerate steep curvature. This highlights a central idea of our work: the convergence behaviour of gradient methods depends not just on the presence of structure, but on how effectively it is incorporated into the optimisation process.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b5', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b29']

Section: Contributions
We provide a systematic characterisation of when reduction mappings lead to provable gains in optimisation efficiency. Specifically, we identify conditions under which the reduced objective exhibits a strictly smaller smoothness constant and a strictly larger sharpness constant 1 . Together, these improvements yield a strictly better condition number, leading to faster worst-case convergence rates for gradient-based methods applied to the reduced problem. The results hold under standard regularity assumptions and apply to both affine and nonlinear mappings. Specifically we contribute the following:
• Theorem 1 shows that for affine reduction mappings, the smoothness constant of the reduced objective is strictly smaller than that of the full objective.
• Theorem 2 extends this result to general nonlinear mappings, establishing improved smoothness under mild regularity assumptions.
• Theorem 3 demonstrates that the sharpness constant of the reduced problem is strictly larger, implying stronger curvature near the minimisers.
• Corollary 3 combines the above to show that the condition number of the reduced problem is strictly better, leading to faster convergence of gradient descent under the PŁ condition.
For completeness, we include a number of supporting lemmas and additional theoretical results in the appendices.
this section cite: []

Section: Related Work
Reparametrisations and geometry-aware optimisation have long been used to exploit structure and improve conditioning in nonconvex problems [41]. Examples include normalisation methods [42,43], which can be interpreted as explicit reduction mappings improving feature geometry and convergence [44]; neural collapse [14] formulations, where our framework captures both fixed classifier parametrisations [45] and dynamic equiangular tight frame (ETF) projections [46]; and gauge-fixing strategies in problems with symmetries, offering a lightweight alternative to quotient manifold optimisation [47,48]. Preconditioned methods also relate closely to our approach, with connections to natural gradient descent [49,50], adaptive preconditioning [51,52], and studies of parameter space geometry under reparametrisation [53]. Our framework generalises these by treating reduction mappings as intrinsic geometry-aware preconditioners, unifying and extending prior work on preconditioning and geometry adaptation. Finally, classical variable elimination and bilevel optimisation methods [54,55] can be seen as special cases of reduction mappings, where our analysis goes beyond dimensionality reduction to provide precise improvements in conditioning and convergence rates. A more detailed discussion is provided in Appendix B.
this section cite: ['b40', 'b41', 'b42', 'b43', 'b13', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54']

Section: Setup, Notation, and Preliminaries
We consider general unconstrained optimisation problems of the form
minimise x∈R n f (x) ,(1)
where f : R n → R is a C 2 (twice differentiable), possibly non-convex function. We assume that the set of minimisers of f is not discrete, but instead forms a non-isolated set. Specifically, we define the set of all local minima in a neighbourhood as
S = {x ∈ R n : x is a local minimum of f and f (x) = c} .(2)
Without loss of generality, we assume the minimum of f is zero, i.e., c = 0. We further assume that, locally around any minimiser, the function f satisfies the PŁ condition with constant µ > 0 (µ-PŁ), that is,
f (x) ≤ 1 2µ ∥∇f (x)∥ 2 . (PŁ)
Other related conditions commonly imposed in non-convex optimisation include the quadratic growth (QG) condition [56] and the error bound (EB) condition [57]. These can be defined as follows, where dist is the classical Euclidean distance: f satisfies the quadratic growth condition with µ > 0 around a minimum if
f (x) ≥ µ 2 dist 2 (x, S) .(QG)
Also, f satisfies the error bound condition with µ > 0 around a minimum if
µ dist(x, S) ≤ ∥∇f (x)∥ .(EB)
Numerous works derive convergence rates for gradient-based methods under these assumptions [58][59][60][61]. Another important condition is the Morse-Bott property [62][63][64], which generalises the classical Morse theory framework [65]. While Morse functions require all critical points to be isolated and nondegenerate, the Morse-Bott condition relaxes this by allowing the set of critical points to form smooth manifolds, as long as the Hessian is non-degenerate in directions normal to these manifolds [62]. This setting is particularly relevant in optimisation problems where symmetries or invariances naturally give rise to non-isolated minimisers lying on structured sets and hence singular Hessians.
Definition 1 (Morse-Bott Property) Let x be a local minimiser of f with associated set of minimisers S and T xS be the tangent space of S at x. Then f satisfies the Morse-Bott property at x if S is a C 1 submanifold around x and ker ∇ 2 f (x) = T xS .
(MB) Furthermore, if there exists a uniform µ > 0 such that for all vectors v normal to S one has ⟨v, ∇ 2 f (x) v⟩ ≥ µ ∥v∥ 2 , then we say that f satisfies the µ-MB property.
Notably, for C 2 functions, it has been shown [64] that these conditions are essentially equivalent-up to potential degradations in the constants or reductions in the neighbourhoods where they hold. Therefore, throughout this work, assuming any one of these conditions allows us to invoke the others interchangeably. Under (MB), we refer to set of minimisers S as the solution manifold.
this section cite: ['b55', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b61', 'b63']

Section: Assumption 1 (Standing Assumptions on Function f
) Suppose there exist constants L, β, µ > 0 and a compact neighbourhood N ⊂ R n around a minimiser x such that the following hold:
this section cite: []

Section: Reparametrisation via Reduction Mappings
We begin by decomposing the variable x of the function f into two components, x 1 ∈ R n1 and x 2 ∈ R n2 , such that n = n 1 + n 2 , with n 1 < n. Whenever the problem structure or prior knowledge permits, we assume that the component x 2 exhibits a known geometric structure at optimality. This allows us to introduce an appropriate mapping-typically a projection or an implicit parametrisationthat explicitly encodes this structure by setting x 2 as a function of x 1 . This leads to a reduction mapping, where x 2 is fixed to lie on its optimal structure, enabling us to reformulate the problem by optimising only over the remaining variables x 1 .
Definition 2 (Reduction Mapping and Reduced Function) Let Ψ : R n1 → R n2 be a C 2 mapping representing the known geometric structure of x 2 at optimality. We define the reduction mapping Φ : R n1 → R n as Φ(x 1 ) := x 1 , Ψ(x 1 ) .
(3) We then define the reduced objective F : R n1 → R as the pullback of f along Φ as
F (x 1 ) := f Φ(x 1 ) = f x 1 , Ψ(x 1 ) .(4)
Since f and Φ are C 2 , the reduced function F is also C 2 .
this section cite: []

Section: Definition 3 (Graph Manifold)
We define the graph manifold of the reduction mapping Φ as
M F := {Φ(x 1 ) = x 1 , Ψ(x 1 ) | x 1 ∈ R n1 } . (5
)
Assuming Ψ is C 2 , the manifold M F , which we will refer to as the feasible manifold, is a globally embedded C 2 submanifold of R n of dimension n 1 , since Φ is globally injective and C 2 .
Assumption 2 (Standing Assumptions on the Reduced Function F ) Assume the following hold within a compact neighbourhood N ⊂ R n defined around a minimiser x:
1. The reduced objective F is β F -smooth on N . 2. The intersection M F ∩ S is non-empty. We denote this set as S F := M F ∩ S, which forms the set of minimisers of F .
this section cite: []

Section: Notation
In the sequel, we use the subscript f , to denote all constants associated with the original objective f (e.g., β f for the smoothness constant, µ f for the (PŁ)). Similarly, all constants related to the reduced objective F will be subscripted accordingly (e.g., β F , µ F ).
this section cite: []

Section: Main Results
We now present the main theoretical contributions of this work. Our results formally demonstrate how exploiting known geometric structure at optimality via reduction mappings leads to improved smoothness and sharpness properties, which in turn result in improved convergence rates for gradientbased methods. Among the four equivalent sharpness conditions discussed previously, we will, for the purpose of this section, adopt the (MB) condition as our reference framework for comparing sharpness constants. When discussing convergence rates, we will switch to the (PŁ) condition, which is more directly linked to iteration complexity. For clarity of exposition, we organise the results into two parts: smoothness improvements and improvements of the Morse-Bott constant.
this section cite: []

Section: Smoothness Improvement under Affine Reduction Mappings
We first establish that affine reduction mappings, by eliminating alignment with the worst-case curvature directions of the original problem, yield strictly improved smoothness constants for the reduced problem. This result is made precise in the following theorem. A noteworthy special case of affine mappings are constant mappings, whose analysis is deferred to Appendix A.
Theorem 1 (Sharper Smoothness Constant for Reduced Functions with Affine Mappings) Let f : R n → R be a C 2 function satisfying Assumption 1 on the compact neighbourhood N . Let Ψ : R n1 → R n2 be an affine mapping, and define the reduction mapping Φ(x 1 ) = x 1 , Ψ(x 1 ) , the reduced function F , and the feasible manifold M F as in Definitions 2 and 3. We consider the local feasible manifold M loc F := M F ∩ N . Let σ 1 , . . . , σ n be the singular values of ∇ 2 f (x) arranged in descending order. Suppose that, for every x ∈ M loc F , the largest singular value (denoted σ max ) has multiplicity p ≥ 1, with associated dominant subspace Σ max . Assume the following:
1. There exists a uniform constant ε ∈ (0, 1] such that for all unit vectors v ∈ Σ max , ∥P TxM loc F v∥ ≤ 1 -ε , where P TxM loc F is the orthogonal projection onto the tangent space T x M loc F .
2. There is a uniform spectral gap:
∆ max := inf x∈M loc F σ max ∇ 2 f (x) -σ p+1 ∇ 2 f (x) > 0 .
Then, equipping R n1 with the pullback metric induced by the embedding Φ, the Riemannian gradient of F is Lipschitz continuous with constant β F satisfying:
β F ≤ β f -∆ max (2ε -ε 2 ) < β f .
this section cite: []

Section: Proof Sketch of Theorem 1
The proof relies on viewing the smoothness of the reduced function F as the largest curvature of f along the feasible manifold M loc F . Since the dominant curvature directions of f are not contained in the tangent space of M loc F , the restriction of the Hessian to M loc F exhibits strictly smaller operator norm. By combining this observation with the spectral gap assumption, we obtain a strict improvement in the smoothness constant. The argument is formalised in Appendix C.
this section cite: []

Section: ■
This result establishes that the smoothness constant of the reduced function F is strictly smaller than that of the original function f , with the improvement governed by the geometric properties of the intersection between the dominant curvature directions and the feasible manifold. These assumptions correspond to generic properties under mild conditions, as discussed in Appendices E.3 and E.4.
We now translate this result into the standard Euclidean setting, where the role of the pullback metric becomes explicit through the following corollary.
Corollary 1 (Euclidean Smoothness Bound under Affine Mappings) Under the setting of Theorem 1, define the Euclidean smoothness constant of the reduced function F as β (E) F := sup x1 ∇ 2 F , where the derivative of Φ is D Φ = I D Ψ and M (Φ) := λ max D Φ ⊤ D Φ . Then,
β (E) F ≤ M (Φ) β f -∆ max (2ε -ε 2 ) < M (Φ) β f .
In particular, when Ψ is an orthogonal projection, M (Φ) = 2, and the sufficient condition ensuring
β (E) F < β f reduces to ∆ max (2ε -ε 2 ) > 1 2 β f . (⋆)
Condition (⋆) requires the Hessian of f to exhibit a significant spectral gap between its largest and subsequent eigenvalues. This scenario corresponds to highly anisotropic curvature, where a small number of dominant directions govern the largest curvatures of the loss landscape. Such spectral structures are commonly observed in over-parametrised models in machine learning, where empirical studies have consistently reported Hessians with a few large outlier eigenvalues and a bulk of nearzero eigenvalues [66][67][68][69][70]. This phenomenon supports the practical relevance of our condition in many deep learning settings. We note that the degradation observed in the Euclidean smoothness constant in Corollary 1 arises because the reduction mapping induces a non-Euclidean geometry on the reduced space. The natural metric in this setting is the pullback metric induced by the mapping itself, under which the improvement in smoothness is directly captured, as shown in Theorem 1. We will return to this point and its implications for convergence rates in Section 4.
this section cite: ['b65', 'b66', 'b67', 'b68', 'b69']

Section: Smoothness Improvement under Nonlinear Reduction Mappings
We now extend the previous result to the more interesting case where the reduction mapping Ψ(x 1 ) is nonlinear. In this setting, the feasible manifold M loc F becomes curved within the ambient space, introducing an additional curvature contribution to the reduced function F . This contribution is captured by the correction term C(x 1 ) in the Hessian (see Lemma 3). Intuitively, this term quantifies the bending effect of M loc F and the extent to which the nonlinearity of Ψ adds curvature to F . When this correction remains sufficiently small relative to the spectral gap between the largest curvature directions of f and their restriction to M loc F , the reduction in the smoothness constant is preserved. This is formalised in the following theorem and the corresponding corollary for the Euclidean case.
Theorem 2 (Sharper Smoothness Constant for Reduced Functions with Nonlinear Mappings) Let f , Φ, F , and M loc F be as in Theorem 1, except that Ψ(x 1 ) is now a general C 2 mapping. Assume:
1. There exist constants Q, Z > 0 such that for all x 1 ∈ M loc F , ∥ D 2 Ψ(x 1 )∥ ≤ Q and
∥∇ x2 f (x 1 , Ψ(x 1 ))∥ ≤ Z.
2. The correction term satisfies Q Z m (Φ) < δ, where m (Φ) = λ min D Φ ⊤ D Φ and δ is the curvature gap defined as
δ := σ max ∇ 2 f (Φ(x 1 )) -σ max ∇ 2 f (Φ(x 1 )) T Φ(x 1 ) M loc F .
Then, the reduced function F has a Lipschitz continuous Riemannian gradient with
β F ≤ β f -∆ max (2ε -ε 2 ) + Q Z m (Φ) < β f .
In particular, when Ψ is an orthogonal projection, m (Φ) = 1.
this section cite: []

Section: Proof Sketch of Theorem 2
The proof extends the affine case by accounting for the additional curvature induced by the nonlinearity of Ψ. This is captured by the correction term C(x 1 ) in the Hessian of F , which arises due to the curvature of the feasible manifold M loc F . By controlling ∥C(x 1 )∥ via bounds on D 2 Ψ and ∇ x2 f , and ensuring that it remains strictly smaller than the curvature gap δ obtained from the projection step, we show that the overall smoothness constant of F is still strictly lower than that of f . The result follows by combining these bounds and applying Weyl's inequality for perturbed operators. Full details are given in Appendix C.
this section cite: []

Section: ■
Corollary 2 (Euclidean Smoothness Bound under Nonlinear Mappings) Under the setting of Theorem 2, the Euclidean smoothness constant of the reduced function F satisfies
β (E) F ≤ M (Φ) β f -∆ max (2ε -ε 2 ) + Q Z < M (Φ) β f ,
where M (Φ) is the metric distortion factor defined as in Corollary 1. In particular, when Ψ is an orthogonal projection, M (Φ) = 2, and the sufficient condition ensuring
β (E) F < β f simplifies to ∆ max (2ε -ε 2 ) > 1 2 (β f + Q Z) .
Reduction mappings often naturally arise in bilevel optimisation settings, where x 2 = Ψ(x 1 ) is implicitly defined as the solution to an inner problem. Depending on the problem structure, these mappings can be affine or nonlinear, and our results apply equally in both cases. We formalise this in the remark below.
this section cite: []

Section: Remark 1 (Inner Mappings as Argmin Problems)
Reduction mappings often arise when Ψ(x 1 ) is defined implicitly as a local solution to an inner optimisation problem
Ψ(x 1 ) ∈ arg min u∈C G(x 1 , u) .
Under standard regularity conditions, such as constraint qualifications and strict second-order sufficiency (SSOSC), classical sensitivity results ensure that Ψ(x 1 ) is locally C 2 [71]. Thus, our previous theorems for affine and nonlinear mappings apply directly in such settings.
To better understand the bounds on the correction term, when the mapping Ψ arises from an inner argmin problem, the constants Q and Z have been explicitly quantified in Theorem 4 in Appendix D.
this section cite: ['b70']

Section: Morse-Bott Constant Improvement under Reduction Mappings
In Appendix D, we establish that if the original function f satisfies the (MB) property, then the reduced function F obtained via reduction mappings also satisfies the (MB) property, albeit potentially with a different constant. In the following theorem, we strengthen this result by showing that the (MB) constant of the reduced problem is in fact strictly improved compared to that of the original problem.
Theorem 3 (Strict Improvement of the Morse-Bott Constant under Smooth Reduction) Let f : R n → R be a C 2 function where the solution manifold S satisfies the µ f -MB property within a compact neighbourhood N . Let Ψ : R n1 → R n2 be a C 2 mapping and define the reduced function F (x 1 ) = f (x 1 , Ψ(x 1 )), with local feasible manifold M loc F = M F ∩ N . At each x ∈ S ∩ M loc F , let H x denote the restriction of ∇ 2 f (x) to N xS . Assume:
1. The smallest eigenvalue λ min (H x) has multiplicity m, with eigenspace E min , and for all v ∈ E min , ∥v∥ = 1, ∥P TxM loc F v∥ ≤ 1 -ε , holds uniformly for some ε ∈ (0, 1]. 2. The spectral gap ∆ min := inf
x∈S∩M loc F [λ n-m (H x) -λ min (H x)] > 0.
Then, for the (MB) property of the reduced function F , we obtain a strictly improved constant
µ F ≥ µ f + ∆ min (2ε -ε 2 ) > µ f .
Proof The result follows by applying the same geometric argument as in Theorem 1, now to the positive definite Hessian H x restricted to N xS . Since we operate entirely within the normal space and on the solution manifold, only eigenvalues and eigenspaces matter, and the nonlinearity of Ψ has no effect. ■
In the Euclidean setting, the improvement in the Morse-Bott constant is further scaled by the pullback metric. Specifically, since the Euclidean constant is related to the intrinsic pullback constant via µ (E) F ≥ m (Φ) µ F , the distortion introduced by the metric, through its smallest eigenvalue m (Φ) ≥ 1, amplifies the effective (MB) constant. This shows that the pullback metric does not only improve the intrinsic condition number but also results in a better constant when measured under the Euclidean metric. In particular, when Ψ is a projection mapping, we have m (Φ) = 1, and the Euclidean result coincides exactly with the pullback metric case.
Remark 2 (Morse-Bott Constant Equivalence [64]) Since in Theorem 3 we showed that µ F > µ f for the (MB) property, by equivalence it follows that the constants µ for the (PŁ), (EB), and (QG) conditions will follow the same strict inequality.
this section cite: ['b63']

Section: Convergence Rate Results and Discussion
In this section, we discuss the algorithmic implications of our geometric analysis of reduction mappings. In particular, we focus on the impact of the improved smoothness and (MB)-and by equivalence (PŁ)-constants obtained in Theorems 1, 2, and 3 on the convergence behaviour of first-order methods.
When the reduced problem is equipped with the pullback metric induced by the reduction mapping Φ, our results show that the condition number of the reduced function F is strictly improved relative to that of the original problem f . This theoretical gain directly translates into faster local linear convergence rates when applying geometrically preconditioned gradient descent (GeoPrecGD) on the reduced objective. We formally state this result below. A broader discussion on metric choices, algorithmic implications, and practical considerations follows.
Corollary 3 (Faster Linear Convergence of the Reduced Function) Under the settings of Theorems 1, 2, and 3, equipping R n1 with the pullback metric induced by Φ yields a strictly improved condition number for the reduced problem
κ F := β F µ F < β f µ f =: κ f .
As a result, preconditioned gradient descent applied to F achieves a strictly faster local linear convergence rate under (PŁ) condition compared to gradient descent on f [5]. Specifically:
• The rate factor improves from O(exp(-t/κ f )) to O(exp(-t/κ F )).
• The iteration complexity to achieve accuracy ϵ improves from O(κ f log(1/ϵ)) to O(κ F log(1/ϵ)).
Below, we make explicit the form of the geometrically preconditioned gradient descent algorithm applied to the reduced problem F under the pullback metric induced by Φ
x (t+1) 1 = x (t) 1 -η R -1 ∇F x (t) 1 , with R := D Φ ⊤ D Φ .(GeoPrecGD)
This method performs steepest descent under the pullback metric, aligning the descent directions with the intrinsic geometry induced by the reduction mapping-a fundamental principle in optimisation [72]. While preconditioning introduces additional computational cost due to the inversion of R, such overhead can often be mitigated by exploiting the structure of R, as shown in prior works [73,74,50,75,76,52,51]. In particular, for affine mappings, R is constant and cheap to apply, whereas for nonlinear mappings, structured or approximate solvers can be used. Designing efficient implementations of these methods is primarily mapping dependent and beyond the scope of this work, where our focus is on the iteration complexity benefits arising from the improved conditioning.
An instructive special case is when Ψ is an orthogonal projection; a common choice for many reparametrisations. If Ψ is affine, then R simplifies, and its inverse is given by R -1 = I -1 2 D Ψ ⊤ D Ψ, implying no additional computational overhead. More generally, if Ψ is nonlinear but still defines an orthogonal projection onto a manifold, the situation remains favourable, particularly near convergence in the defined neighbourhood. As iterates approach the solution manifold, the derivative D Φ locally behaves as a linear orthogonal projection onto the tangent space of the solution manifold, with the condition number of R remaining close to two. This favourable and stable spectrum near the solution means the preconditioned linear system within the (GeoPrecGD) step (Rz = ∇F ) can be solved efficiently. Specifically, iterative solvers (e.g., Conjugate Gradient) applied to this system benefit from rapid convergence due to the low condition number. Also, the structure of R allows for an efficient and cheap inversion via the Woodbury formula, depending on whether n 1 > n 2 or vice versa.
We validate our theory with synthetic experiments in Appendix G. Notably, for quadratic objectives, our geometrically preconditioned gradient descent is equivalent to the full Newton method on the reduced problem; and for nonlinear least-squares, it is equivalent to the Gauss-Newton method applied to the reduced problem. Thus, our approach recovers Newton and Gauss-Newton as special cases while extending beyond them to more general settings.
While preconditioning is necessary and recommended [53] to fully exploit the improved geometry induced by the reduction mapping, improvements can still be observed under Euclidean gradient descent. Specifically, if the condition (⋆) holds, we still have β (E) F < β f , leading to a better condition number even without preconditioning. Even if this condition fails, the improvement in the sharpness constant µ (E) F > µ f may still result in a better overall condition number κ (E) F compared to κ f . Thus, gains are still possible, although they may be limited.
If a practitioner desires additional control over the condition number in Euclidean space for convergence guarantees, the reduction mapping can be modified to induce approximate isometries via a simple small-slope method. This approach is general and may warrant a more systematic study; however, a full investigation of such strategies lies outside the scope of this paper. We sketch the idea below.
Remark 3 (Small-Slope Method and Approximate Isometry) When using reduction mappings, the pullback metric takes the form of R = I + D Ψ ⊤ D Ψ, which in general is not identity, leading to potential degradation in the Euclidean smoothness constant proportional to λ max (R). To mitigate this, we propose a simple small-slope method, where the mapping is rescaled as Ψ α (x 1 ) := αΨ(x 1 ) for a small α > 0. This modifies the metric to R α = I + α 2 D Ψ ⊤ D Ψ, whose condition number becomes
κ(R α ) = 1 + α 2 λ max (D Ψ ⊤ D Ψ) 1 + α 2 λ min (D Ψ ⊤ D Ψ) .
Thus by making α sufficiently small, R α becomes approximately isometric, improving the likelihood that β (E) F ≲ β f . This method offers a simple yet effective trade-off between improving conditioning and preserving geometric fidelity. Notably, if the geometric structure encoded by Ψ is not scaleinvariant, excessively small α may distort the mapping, potentially degrading its ability to capture the correct geometry of the x 2 variables. This trade-off is less critical in many common cases-such as subspaces, cones, or orthogonal structures-where the geometry is inherently scale-invariant. The parameter α can be treated as a user-defined knob, balancing conditioning improvement against the precision of the reduction.
this section cite: ['b4', 'b71', 'b72', 'b73', 'b49', 'b74', 'b75', 'b51', 'b50', 'b52']

Section: Limitations and Future Directions
Our analysis assumes that the reduction mapping Ψ can be evaluated exactly. When Ψ is defined implicitly, this idealisation neglects the approximation errors that may arise in practice, potentially introducing bias in the reduced gradients and weakening the theoretical convergence guarantees. Moreover, the computational cost of evaluating Ψ(x 1 ) is inherently problem-specific and is not explicitly captured in the iteration complexity analysis. A rigorous treatment that jointly considers the benefits of reduction mappings and their evaluation cost remains an open, problem-dependent direction for future work.
Beyond these considerations, our analysis is confined to local properties and deterministic firstorder methods. Extending the theory to global convergence settings, particularly in nonconvex landscapes, is an important future direction. In this context, it would be valuable to explore how reduction mappings reshape the landscape globally, including their potential to eliminate or introduce saddle points and spurious minima, extending the framework of [41]. Another promising direction is applying our geometric framework to specific structured problems where the objective admits compositional forms, such as matrix factorisation, nonlinear regression, or neural network training, to obtain practical insights. Finally, extending the analysis to stochastic settings, for example by studying stochastic gradient descent (SGD) under reduction mappings and characterising its regularity conditions, would enhance the relevance of our results in large-scale applications.
this section cite: ['b40']

Section: Conclusion
In this work, we presented a unified geometric framework for analysing reduction mappings in optimisation problems with structured solution sets. By explicitly incorporating mappings that encode known geometric structures at optimality, we showed that the resulting reduced problems exhibit strictly improved smoothness and sharpness properties. This leads to enhanced local condition numbers and provably faster convergence rates when applying appropriately preconditioned first-order methods. Our analysis generalises seamlessly from affine to nonlinear mappings, carefully accounting for the additional curvature induced by the bending of the feasible manifold. While the improvements are fundamentally intrinsic to the pullback geometry, we further showed that, under appropriate constructions and trade-offs, these gains can also manifest in the Euclidean metric. Throughout, we emphasised the generality and flexibility of our framework, illustrating that reduction mappings may be given explicitly or arise implicitly as solutions to inner optimisation problems, thereby connecting our approach to classical bilevel and composition formulations via implicit differentiation. We believe this geometric perspective offers a principled and broadly applicable toolset for designing more efficient optimisation algorithms that better exploit problem structure, with potential impact across areas such as matrix factorisation, deep learning, and other structured nonconvex problems.
this section cite: []

Section: References
Ref_id:b0 Title: The loss surfaces of multilayer networks Year: (2015)
Ref_id:b1 Title: Deep learning without poor local minima Year: (2016)
Ref_id:b2 Title: Gradient descent only converges to minimizers: Non-isolated critical points and invariant regions Year: (2016)
Ref_id:b3 Title: Gradient descent provably optimizes overparameterized neural networks Year: (2018)
Ref_id:b4 Title: Zhurnal vychislitel'noi matematiki i matematicheskoi fiziki Year: (1963)
Ref_id:b5 Title: Loss landscapes and optimization in over-parameterized nonlinear systems and neural networks Year: (2022)
Ref_id:b6 Title: On exponential convergence of sgd in non-convex overparametrized learning Year: (2018)
Ref_id:b7 Title: Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition Year: (2016)
Ref_id:b8 Title: Convergence of the iterates of descent methods for analytic cost functions Year: (2005)
Ref_id:b9 Title: Global minima of overparameterized neural networks Year: (2021)
Ref_id:b10 Title: The role of permutation invariance in linear mode connectivity of neural networks Year: (2021)
Ref_id:b11 Title: Sharp minima can generalize for deep nets Year: (2017)
Ref_id:b12 Title: Path-sgd: Path-normalized optimization in deep neural networks Year: (2015)
Ref_id:b13 Title: Prevalence of neural collapse during the terminal phase of deep learning training Year: (2020)
Ref_id:b14 Title: Deep neural collapse is provably optimal for the deep unconstrained features model Year: (2023)
Ref_id:b15 Title: Neural collapse vs. low-rank bias: Is deep neural collapse really optimal? Year: (2024)
Ref_id:b16 Title: The persistence of neural collapse despite low-rank bias: An analytic perspective through unconstrained features Year: (2024)
Ref_id:b17 Title: Feature learning in deep classifiers through intermediate neural collapse Year: (2023-07)
Ref_id:b18 Title: Extended unconstrained features model for exploring deep neural collapse Year: (2022)
Ref_id:b19 Title: Equiangular tight frames that contain regular simplices Year: (2018)
Ref_id:b20 Title: Nonconvex optimization meets low-rank matrix factorization: An overview Year: (2019)
Ref_id:b21 Title: Complete dictionary recovery over the sphere i: Overview and the geometric picture Year: (2016)
Ref_id:b22 Title: Complete dictionary recovery over the sphere ii: Recovery by riemannian trust-region method Year: (2016)
Ref_id:b23 Title: Efficient dictionary learning with gradient descent Year: (2019)
Ref_id:b24 Title: Analysis of the optimization landscapes for overcomplete representation learning Year: (2019)
Ref_id:b25 Title: No spurious local minima in nonconvex low rank problems: A unified geometric analysis Year: (2017)
Ref_id:b26 Title: Matrix completion has no spurious local minimum Year: (2016)
Ref_id:b27 Title: On the optimization landscape of tensor decompositions Year: (2017)
Ref_id:b28 Title: Escaping from saddle points-online stochastic gradient for tensor decomposition Year: (2015)
Ref_id:b29 Title: From symmetry to geometry: Tractable nonconvex problems Year: (2020)
Ref_id:b30 Title: Phase retrieval via matrix completion Year: (2015)
Ref_id:b31 Title: Phase retrieval via wirtinger flow: Theory and algorithms Year: (2015)
Ref_id:b32 Title: A geometric analysis of phase retrieval Year: (2018)
Ref_id:b33 Title: The numerics of phase retrieval Year: (2020)
Ref_id:b34 Title: Blind deconvolution meets blind demixing: Algorithms and performance bounds Year: (2017)
Ref_id:b35 Title: Structured local minima in sparse blind deconvolution Year: (2018)
Ref_id:b36 Title: Geometry and symmetry in short-and-sparse deconvolution Year: (2019)
Ref_id:b37 Title: Short-and-sparse deconvolution-a geometric approach Year: (2019)
Ref_id:b38 Title: Global geometry of multichannel sparse blind deconvolution on the sphere Year: (2018)
Ref_id:b39 Title: A nonconvex approach for exact and efficient multichannel sparse blind deconvolution Year: (2019)
Ref_id:b40 Title: The effect of smooth parametrizations on nonconvex optimization landscapes Year: (2025)
Ref_id:b41 Title: Batch normalization: Accelerating deep network training by reducing internal covariate shift Year: (2015)
Ref_id:b42 Title: Layer normalization Year: (2016)
Ref_id:b43 Title: How does batch normalization help optimization? Year: (2018)
Ref_id:b44 Title: A geometric analysis of neural collapse with unconstrained features Year: (2021)
Ref_id:b45 Title: Guiding neural collapse: Optimising towards the nearest simplex equiangular tight frame Year: (2024)
Ref_id:b46 Title: Optimization Algorithms on Matrix Manifolds Year: (2008)
Ref_id:b47 Title: An introduction to optimization on smooth manifolds Year: (2023)
Ref_id:b48 Title: Natural gradient works efficiently in learning Year: (1998)
Ref_id:b49 Title: Fast convergence of natural gradient descent for over-parameterized neural networks Year: (2019)
Ref_id:b50 Title: Shampoo: Preconditioned stochastic tensor optimization Year: (2018)
Ref_id:b51 Title: Muon: An optimizer for hidden layers in neural networks Year: (2024)
Ref_id:b52 Title: The geometry of neural nets' parameter spaces under reparametrization Year: (2023)
Ref_id:b53 Title: Nonlinear programming Year: (1997)
Ref_id:b54 Title: On differentiating parameterized argmin and argmax problems with application to bi-level optimization Year: (2016)
Ref_id:b55 Title: Second-order sufficiency and quadratic growth for nonisolated minima Year: (1995)
Ref_id:b56 Title: Error bounds and convergence analysis of feasible descent methods: a general approach Year: (1993)
Ref_id:b57 Title: Degenerate nonlinear programming with a quadratic growth condition Year: (2000)
Ref_id:b58 Title: Linear convergence of first order methods for nonstrongly convex optimization Year: (2019)
Ref_id:b59 Title: Asynchronous stochastic coordinate descent: Parallelism and convergence properties Year: (2015)
Ref_id:b60 Title: Error bounds, quadratic growth, and linear convergence of proximal methods Year: (2018)
Ref_id:b61 Title: Nondegenerate critical manifolds Year: (1954)
Ref_id:b62 Title: Lectures on Morse theory, old and new Year: (1982)
Ref_id:b63 Title: Fast convergence to non-isolated minima: four equivalent conditions for C 2 functions Year: ()
Ref_id:b64 Title:  Year: (1969)
Ref_id:b65 Title: Measurements of three-level hierarchical structure in the outliers in the spectrum of deepnet hessians Year: (2019)
Ref_id:b66 Title: Traces of class/cross-class structure pervade deep learning spectra Year: (2020)
Ref_id:b67 Title: Eigenvalues of the hessian in deep learning: Singularity and beyond Year: (2016)
Ref_id:b68 Title: Empirical analysis of the hessian of over-parametrized neural networks Year: (2017)
Ref_id:b69 Title: An investigation into neural net optimization via hessian eigenvalue density Year: (2019)
Ref_id:b70 Title: Perturbation analysis of optimization problems Year: (2013)
Ref_id:b71 Title: Old optimizer, new norm: An anthology Year: (2024)
Ref_id:b72 Title: The power of preconditioning in overparameterized low-rank matrix sensing Year: (2023)
Ref_id:b73 Title: Gram-gauss-newton method: Learning overparameterized neural networks for regression problems Year: (2019)
Ref_id:b74 Title: Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model Year: (2019)
Ref_id:b75 Title: Accelerating ill-conditioned low-rank matrix estimation via scaled gradient descent Year: (2021)
Ref_id:b76 Title: The prevalence of neural collapse in neural multivariate regression Year: (2024)
Ref_id:b77 Title: Numerical optimization Year: (2006)
Ref_id:b78 Title: Optimizing neural networks with kronecker-factored approximate curvature Year: (2015)
Ref_id:b79 Title: Perturbation Theory for Linear Operators. Classics in Mathematics Year: (1995)
Ref_id:b80 Title: On the codimension of the variety of symmetric matrices with multiple eigenvalues Year: (2006)
Ref_id:b81 Title: The measure of the critical values of differentiable maps Year: (1942)
Ref_id:b82 Title: Graduate Texts in Mathematics Year: (1997)
Ref_id:b83 Title: The analysis of linear partial differential operators III: Pseudo-differential operators Year: (2007)
Ref_id:b84 Title: Computable, obstructed morse homology for clean intersections Year: (2024)
