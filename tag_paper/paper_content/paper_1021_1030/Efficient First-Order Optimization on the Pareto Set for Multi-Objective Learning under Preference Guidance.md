Title: Efficient First-Order Optimization on the Pareto Set for Multi-Objective Learning under Preference Guidance
Abstract: Multi-objective learning under user-specified preference is common in real-world problems such as multi-lingual speech recognition under fairness. In this work, we frame such a problem as a semivectorial bilevel optimization problem, whose goal is to optimize a pre-defined preference function, subject to the constraint that the model parameters are weakly Pareto optimal. To solve this problem, we convert the multi-objective constraints to a single-objective constraint through a merit function with an easy-to-evaluate gradient, and then, we use a penalty-based reformulation of the bilevel optimization problem. We theoretically establish the properties of the merit function, and the relations of solutions for the penalty reformulation and the constrained formulation. Then we propose algorithms to solve the reformulated single-level problem, and establish its convergence guarantees. We test the method on various synthetic and real-world problems. The results demonstrate the effectiveness of the proposed method in finding preference-guided optimal solutions to the multi-objective problem.

Section: Introduction
Many machine learning tasks naturally involve multiple objectives, which may include diverse performance metrics such as accuracy, fairness, and privacy, or even the same metrics evaluated across different datasets (Sener & Koltun, 2018). A common approach to tackling such multiobjective problems is to learn a shared model that performs well across all objectives simultaneously. Compared to training separate models for each objective, this approach offers 1 Rensselaer Polytechnic Institute, Troy, United States 2 Kyoto University, Kyoto, Japan 3 Peking University, Beijing, China. Correspondence to: Lisha Chen <lishachen9577@gmail.com>, Tianyi Chen <chentianyi19@gmail.com>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). significant benefits -most notably, it reduces model size and inference time, making it more efficient and scalable. Multiobjective optimization facilitates this by enabling the learning of models that minimize vector-valued objectives (Miettinen, 1998;Ehrgott, 2005). In practical scenarios, it is often desirable to obtain solutions that provide controlled trade-offs or reflect specific preferences among competing objectives, rather than treating all objectives equally.
To further illustrate, we use one example on multi-lingual speech or language processing problem in Figure 1. The goal of this problem is to minimize multiple losses from different languages, while satisfying the user-specified preferences. Preferences can control trade-offs among multiple losses and enhance steerability, enabling the solver to return diverse solutions on the Pareto front. Analytically, the preferences can be defined as constraints or objectives, see, e.g., (Lin et al., 2019;Curtis et al., 2023;Chen et al., 2024a). To prioritize finding the optimal solutions of the multi-lingual losses over satisfying the preferences, we model the preference as a secondary scalarvalued objective. We first optimize the vector-valued objective formed by concatenating the multi-lingual losses, and then optimize the scalar-valued preference objective. More formally, let f 1 , . . . , f M : X → R be the objective functions, and f 0 : X → R be the preference function, e.g., the discrepancy between different objectives, with X ⊆ R q being nonempty, closed and convex. Depending on the problem, X can be compact or R q . Let F = (f 1 , . . . , f M ) : X → R M be a vector-valued function. Then the optimization problem is
min x∈X f 0 (x) s.t. x ∈ arg min x∈X F (x)(OPS)
where for minimizing the vector-valued objective F (x), we consider the widely used Pareto optimality, whose formal definition is deferred to Section 2. Then problem (OPS) is also known as Optimization on the Pareto Set (OPS), or a semivectorial simple bilevel optimization problem (Bolintineanu, 1993a;b).
The OPS or semivectorial bilevel optimization problem is generally very difficult to solve (Bolintineanu, 1993b). Existing studies usually make strong assumptions such as the convexity of the objectives F (Roy et al., 2023), or the algorithms require evaluating the second-order derivatives of the objectives, and is thus inefficient for large-scale problems (Chen et al., 2022;Roy et al., 2023). To address these challenges, we introduce reformulations of the OPS problem, establish relations of the reformulations and the original problem, and propose algorithms to solve the reformulated problem with convergence guarantees.
Our contributions can be summarized as follows:
First, we propose a smoothed merit function to convert the vector-valued objective F to a scalar-valued objective, with easy-to-evaluate gradient. We prove that the smoothed merit function preserves the equivalence with approximate weak Pareto optimality. Then we use the smoothed merit function as a penalty function, prove its error bound properties, and establish the relation of the global/local/stationary solutions of the penalty-based reformulation and the original problem. Based on the reformulation, we propose an efficient first-order algorithm with convergence rate guarantees.
Experiments are conducted on various synthetic and real datasets with possibly nonconvex objectives to demonstrate its effectiveness.
Technically, we address the following challenges:
T1 The original OPS problem in general has a non-convex non-smooth structure, posing challenges to evaluating the subdifferential and developing convergent algorithms. We propose to use a smoothed merit function as a penalty so that it has easy-to-evaluate gradient, and the convergence of the developed algorithm can be analyzed.
T2 Instead of directly assuming the lower-level merit (penalty) function satisfies the error bound as in existing (simple) bilevel optimization literature, we prove the proposed merit (penalty) function satisfies the desired error bound when the objectives satisfy certain conditions such as subanalyticity.
T3 We define a stationary condition for the simple bilevel problem with provable calmness condition under the Kurdyka-Łojasiewicz inequality, weaker than the assumptions in existing results, and thus applicable to a wider range of problems including ours. Based on this, we establish the relation of the stationary solution to the penalty problem and that to the simple bilevel problem.
this section cite: ['b71', 'b63', 'b26', 'b51', 'b19', 'b69', 'b69']

Section: Problem Setup and Preliminaries
For the optimization problem min x∈X F (x) in (OPS), we use the standard definitions for Pareto optimality (Miettinen, 1998). Given two vectors v and w, we use v < w and v ≤ w to denote v i < w i and v i ≤ w i for all i, respectively. We use v ⪇ w to denote v ≤ w and v ̸ = w, and define >, ≥, ⪈ analogously. We use 1 M to denote an all-one vector with dimension M , where M is sometimes ommitted if it is clear from the context. Then Pareto dominance and weak Pareto optimality are formally defined below.
Definition 2.1 (Pareto dominance and optimality). Given v, w ∈ R M , we say v strictly dominates w if and only if v-w < 0. Correspondingly, a point x ∈ X is weakly Pareto optimal if there is no x ′ ∈ X such that, F (x ′ ) < F (x). In addition, a point x ∈ X is ϵ-weakly Pareto optimal if there exists no x ′ ∈ X and x ′ ̸ = x such that, F (x ′ ) < F (x) -ϵ1.
Throughout the paper, we assume f m (x), m = 0, . . . , M are proper and bounded below. And we assume they are twice continuously and directionally differentiable. Denote the directional derivative of f m at point x along direction d as f ′ m (x; d), defined as
f ′ m (x; d) := lim α↓0 f m (x + αd) -f m (x) α .(1)
Then the Pareto stationarity is defined as follows.
Definition 2.2 (Pareto stationarity e.g. (Ehrgott, 2005)). A point x ∈ X is Pareto stationary if max m∈[M ] f ′ m (x; zx) ≥ 0 for all z ∈ X , where [M ] = {1, . . . , M }.
Denote W P (F ) ⊆ X as the weak Pareto set of F , which contains all the weakly Pareto optimal solutions for min x∈X F (x). Then problem (OPS) is equivalent to min x∈W P (F ) f 0 (x). We say that the solutions to (OPS) are preferred Pareto optimal. We then discuss in the next section the problem formulation to find these solutions.
this section cite: ['b63', 'b26']

Section: Problem Reformulation
In this section, we first convert the original problem (OPS) with multiple lower-level objectives to an equivalent problem with a single lower-level objective using a merit function. Then we discuss its penalty-based reformulation.
this section cite: []

Section: A smoothed merit function and its properties
A merit function associated with the multi-objective optimization problem min x∈X F (x) is non-negative, and returns zero only at the weakly Pareto optimal solutions (Auslender, 1976;Hearn, 1982). Under this require-ment, assuming lower semicontinuity of F , then ū(x) := sup y∈X min m∈ [M ] {f m (x) -f m (y)}
(2) is a merit function in the sense of weak Pareto optimality (Tanabe et al., 2024, Theorem 3.1). In other words, ū(x) = 0 if and only if x is weakly Pareto optimal. Given this equivalence, it is desirable to convert the original lowerlevel multi-objective optimization problem to minimizing the scalar-valued function ū(x). However, ū(x) in general can be non-differentiable due to its max-min structure, posing challenges to directly applying gradient-based approaches to minimize ū(x). To address this challenge, we propose the following smoothed and regularized merit function v l,τ (x) given l ≥ 0, τ > 0.
Note that, v l,τ can be seen as a smoothed and regularized function of ū. Specifically, when l = 0, v 0,τ smoothes the maximization operation over m ∈ [M ] in ū with the logsum-exponential (LSE) function (Nesterov, 2005). And it uniformly converges to ū as τ converges to zero. Besides, adding the regularization with l > 0 can further lift a weakly convex objective to a strongly convex one, so that not only the minimization min y∈X h l,τ (x, y) in (3b) enjoys a unique solution, but also v l,τ (x) is smooth and has easy-to-evaluate gradient. To further illustrate, we provide a visualization of ū, v l,τ on a simple example in Figure 2. It shows that v l,τ with smaller τ or l approximates ū better, while larger τ or l makes v l,τ smoother. Too large τ or l could possibly change the shape of v l,τ significantly compared to ū.
-1.0 -0.5 0.0 0.5 1.0 -0.05
(x) = 6 (x + 1 2 ) 2 + 1 8 , 6 (x -1 2 ) 2 + 1 8 ⊤ .
Then we discuss the properties of the smoothed merit function and outline the procedure for computing its gradient.
Properties of the smoothed merit function. We establish how the value of the smoothed merit function changes when x reaches the weak Pareto optimality condition. In the prior work (Tanabe et al., 2024), such properties have been established for the merit functions ū and u l (c.f. Appendix B.1). We will show that the smoothed version v l,τ can still preserve these properties approximately depending on the hyperparameters l and τ . For our analysis, we make the following assumptions that are common in multiobjective optimization (Fliege et al., 2019;Liu & Vicente, 2021). Note that we do not require all the following assumptions to hold for all our results, which will be specified correspondingly.
Assumption 1. For all m ∈ {0, . . . , M }, f m (x) is locally Lipschitz on any bounded set in X .
Definition 3.1 (Weak convexity). A locally Lipschitz func-
tion f : X → R is µ-weakly convex if f (x) -µ 2 ∥x∥ 2 is convex for x ∈ X .
Assumption 2. For all m ∈ [M ], f m (x) is locally Lipschitz and µ-weakly convex on X .
Before proceeding to the theoretical results, we introduce the following definition, which relaxes the commonly used convexity assumption of the objective functions.
Definition 3.2 (Point strong quasar-convex functions (Hardt et al., 2018
, Definition 2.1)). A function f : X → R is (c q , µ)-point strong quasar-convex with c q ∈ (0, 1], µ ≥ 0 at x * ∈ X if for all x ∈ X , f (x * ) ≥ f (x) + 1 cq ∇f (x) ⊤ (x * -x) + µ 2 ∥x * -x∥ 2 . (4
)
The point (strong) quasar-convexity in Definition 3.2 is a relaxation of the (strong) convexity. When c q = 1, the quasarconvexity implies point star-convexity (Lee & Valiant, 2016).
And if the point star-convexity holds at all x ∈ X , then it implies convexity. There are many examples of nonconvex but point quasar-convex functions, see the discussions in, e.g., (Hinder et al., 2020). In machine learning, typical examples that satisfy the quasar convexity include the linear dynamical systems identification (Hardt et al., 2018) and generalized linear models with leaky ReLU or logistic activation functions (Wang & Wibisono, 2023).
Based on the above assumptions and definition, we introduce the properties of the smoothed merit function v l,τ in Proposition 3.3.
Proposition 3.3 (Properties of v l,τ ). Suppose Assumption 2 holds. The merit function v l,τ (x) defined in (3b) satisfies the following properties:
1. ū(x) -τ ln M ≤ v 0,τ (x) ≤ ū(x). Furthermore, min x∈X v l,τ (x) = -τ ln M . 2. If x is weakly Pareto optimal, then v l,τ (x) ≤ 0. Con- versely, if a) l = 0, v l,τ (x) ≤ 0, then x is ϵ-weakly Pareto optimal with ϵ = τ ln M ; b) l > 0, v l,τ (x) ≤ -τ ln M
, and for all m ∈ [M ], f m are (1, 0)-point quasar-convex at x, then x is weakly Pareto optimal.
The proof of Proposition 3.3 is deferred to Appendix B.3. In Appendix B.4, we provide some examples of nonconvex F that satisfies condition 2-b) in Proposition 3.3.
Next we discuss how to compute the gradient of the smoothed merit function v l,τ , so that we can use gradientbased methods to directly minimize v l,τ .
this section cite: ['b1', 'b36', 'b65', 'b74', 'b29', 'b56', 'b35', 'b49', 'b37', 'b35', 'b77']

Section: Gradient of the smoothed merit function.
Under Assumption 2, all the objectives f m are µ-weakly convex, i.e., f m (x) -µ 2 ∥x∥ 2 is convex on X . The LSE function preserves weak convexity (c.f. Lemma B.3), thus h l,τ (x, y) is strongly convex w.r.t. y if l + µ > 0. Then y * l,τ (x) := arg min y∈X h l,τ (x, y) is a singleton, and is continuous w.r.t. x, so the Danskin-type theorem can be applied here to compute the gradient of v l,τ , given by
∇v l,τ (x) = M m=1 π m (x)∇f m (x) -l(x -y * l,τ (x)), (5a
) with π m (x) := e 1 τ (fm(y * l,τ (x))-fm(x)) M m=1 e 1 τ (fm(y * l,τ (x))-fm(x)) .(5b)
Reformulation of problem (OPS). We then consider the following optimization problem by approximating the Pareto set constraint x ∈ W P (F ) through a merit function constraint using v l,τ .
min x∈X f0(x), s.t. x ∈ X * v l,τ :={x ∈ X | v l,τ (x) + τ ln M ≤ 0}.(CP)
We name the above program a constrained program (CP) reformulation. By Proposition 3.3, S(F ) ⊆ W P (F ), and these two sets become equal as l, τ ↓ 0. To solve (CP), we further consider a penalty-based program (PP γ ) with penalty parameter γ, θ > 0 below
min x∈X φ γ (x) := f 0 (x) + γp(x) (PP γ ) with p(x) := (v l,τ (x) + τ ln M ) θ .
For (PP γ ), when γ → ∞, any limit point of the sequence of solutions to the approximation problem (PP γ ) is a solution to the problem (CP).
this section cite: []

Section: Relation of different formulations
To establish the relations of the solutions to (PP γ ), (CP), and (OPS), without loss of generality, we assume there exists at least one x * ∈ arg min x∈X v l,τ (x) that x * is bounded, and that the function value f m and gradient ∇f m at x * for m = 0, . . . , M are also bounded. We also introduce the following global subanalyticity assumptions on the objectives below.
Assumption 3 (Subanalyticity of f m (x)). For all m ∈ [M ], f m (x) is subanalytic on X .
Due to space limit, the definitions on (global) subanalytic functions and the related properties are provided in Appendix C. Subanalyticity can be generally satisfied by many widely-used objective functions (Dries & Miller, 1996;Bolte et al., 2007). For example, the ℓ p -norm with p ≥ 1, and the LSE and polynomial functions defined on a bounded set, all satisfy the subanalyticity. More discussions and examples are provided in Appendix C.2 and Table 6. Intuitively speaking, global subanalytic functions can be described by finite combinations of locally analytic functions. They exhibit a "tame" geometry, thus stability under basic operations, and desirable properties for optimization. One of them is the Hölderian error bound defined below.
Definition 3.4 ((ϱ, η)-Hölderian error bound). For a func-
tion v : X → R, let X * v := arg min x∈X v(x). Then v satisfies the (ϱ, η)-Hölderian error bound (HEB) if ϱ v(x) -min x∈X v(x) ≥ dist(x, X * v ) η (6
)
where dist(x, S) is the Euclidian distance from a point x to a set S, and ϱ, η > 0.
The HEB in Definition 3.4 generalizes the widely used Quadratic Growth (QG) condition with η = 2 in optimization (Karimi et al., 2016;Drusvyatskiy & Lewis, 2018), and the weak sharp minima condition with η = 1 (Burke & Ferris, 1993). This condition ensures the point is close to the solution set if the function value gap at the point is small. In our problem, it is desirable that the function v l,τ also satisfies such a condition, so that it satisfies HEB near its solution set. This can be proved based on the properties of subanalytic functions, as described in Lemma 3.5 below.
Lemma 3.5 (Subanalyticity of X * v l,τ and v l,τ (x)). Under Assumption 3, and that X is subanalytic, given a compact subanalytic set X C , suppose f m (x) is continuous and bounded on X C ∩ X for all m ∈ [M ]. Then both X * v l,τ ∩ X C and v l,τ (x) on X C ∩ X are globally subanalytic. Consequently, v l,τ (x), p(x) satisfy the (ϱ, η) and (ϱ p , η p )-HEB in Definition 3.4 on X C ∩ X , respectively, with some ϱ, η > 0, η p = θη, and ϱ p = ϱ θ .
Definition 3.4 and Lemma 3.5 are crucial for establishing the relations of the global/local/stationary solutions of the penalty reformulation (PP γ ) and the constrained formulation (CP). Below we first define the global and local solutions to (CP), then we discuss their relation with global and local solutions to (PP γ ).
Definition 3.6 (Global and local solutions). We say x is an (ϵ, δ)-global solution to (CP) on X S ⊆ X if it satisfies
f 0 (x) -min x∈X S ∩X δ f 0 (x) ≤ ϵ, x ∈ X S ∩ X δ ,(7a)
with
X δ := {x ∈ X | p(x) ≤ δ}.(7b)
Let B(x, r) denote the neighborhood of x with radius r.
We say x is an (ϵ, δ)-local solution to (CP) on X if it is an (ϵ, δ)-global solution on X S = B(x, r) ∩ X .
In Theorem 3.7, we establish the relation of solutions to (CP) with a smoothed merit function and those to the original problem (OPS). We also establish the relations of the global/local solutions of the penalty reformulation (PP γ ) and the constrained formulation (CP).
Theorem 3.7 (Relation of ϵ-global/local solutions to (OPS), (CP) and (PP γ )). Suppose Assumptions 1 and 3 hold. Then with proper choices of ϵ ′ , ϵ, δ ′ , γ depending on δ for all δ > 0, we have 1. (Relation of (OPS) and (CP)) The (ϵ, δ)-global/local solutions to (CP) with δ ≥ τ ln M are (ϵ ′ , δ)-global/local solutions to (OPS). Conversely, the (ϵ, δ)-global/local solutions to (OPS) are (ϵ ′ , δ ′ )-global/local solutions to (CP).
this section cite: ['b24', 'b7', 'b44', 'b25', 'b11']

Section: (Relation of (CP) and (PP γ ))
The ϵ-global/local solution to (PP γ ), denoted as x γ , is an (ϵ, δ +2ϵ * )-global/local solution to (CP), where ϵ * = 0 for the global case, and ϵ * = inf x∈B(xγ ,r)∩X p(x) for the local case. Conversely, an (ϵ ′ , ϵ)-global/local solution to (CP) is a δ-global/local solution to (PP γ ).
The proof of Theorem 3.7-2 is deferred to Appendix D.2. It extends the result from (Shen et al., 2025) using the HEB proved in Lemma 3.5 on a compact subanalytic set instead of directly assuming the QG condition on the whole domain X . Furthermore, it does not rely on convexity assumptions of v l,τ (x). The less restrictive assumptions make it applicable to a much wider set of problems. A more detailed comparison is given in Appendix A.1.
Theorem 3.7-2 states that if v l,τ satisfies HEB, and there exists p(x) ≤ ϵ * in the neighborhood of the local solution, then the local solution of (PP γ ) is a local solution of (CP).
In e.g., (Shen et al., 2025;Chen et al., 2024b), these conditions can be satisfied under certain assumptions on the lower-level objective, such as the PL inequality or convexity. However, in our problems (PP γ ) and (CP), we cannot directly assume such conditions hold for v l,τ . Therefore, next we will show that the conditions hold under additional conditions specified in Proposition 3.9. We first introduce the Kurdyka-Łojasiewicz inequality below.
Definition 3.8 (Kurdyka-Łojasiewicz inequality). A proper and lower semicontinuous function f : R q → (-∞, +∞] satisfies the (c, α)-Kurdyka-Łojasiewicz (KL) inequality at x if there exist ν ∈ (0, +∞], c > 0, α > 1, a neighborhood B(x), such that for all x ∈ B(x) and f (x) < f (x) < f (x) + ν, the following inequality holds
c dist(0, ∂f (x)) α ≥ ∥f (x) -f (x)∥.(8)
Moreover, if f satisfies the (c, α)-KL inequality for every pair of points (x, x) on a set X C with f (x) = min x∈X C f (x), then we say f is (c, α)-KL on X C .
Proposition 3.9. Let x ∈ X be a bounded ϵ-stationary point of min x∈X v l,τ (x). If there exists x * ∈ X * v l,τ and
x ∈ B(x * ) with KL inequality at x * , then v l,τ satisfies the condition in Theorem 3.7-2. The above condition holds if a) for all m ∈ [M ], f m satisfies the (1, 0)-point quasarconvexity at x, and (1, µ)-point strong quasar-convexity at y * l,τ (x) = arg min y∈X h l,τ (x, y); or b) v l,τ (x)+τ ln M ≤ ν in Lemma C.9. The condition in Theorem 3.7-2 requires a local solution to min x∈X v l,τ (x) is also ϵ-globally optimal to this problem. Such a condition holds if F satisfies point convexity or subanalyticity near the stationary points of v l,τ (x), as shown in Proposition 3.9. The proof is deferred to Appendix D.2. Next, we define the KKT condition, its necessity, and establish the relation of stationary solutions.
Definition 3.10 ((ϵ, δ)-stationary condition, e.g. (Liu et al., 2022;Xiao et al., 2023b)). Let X = R q . For the function p such that ∇p(x) and ∇ 2 p(x) exist, and ∇p(x) = 0 implies p(x) = 0, a gradient-based reformulation of the problem (CP) is
min x∈R q f 0 (x), s.t. ∇p(x) = 0,(9)
The (ϵ, δ)-KKT condition of the problem (9) is
∥∇f 0 (x) + ∇ 2 p(x)w∥ ≤ ϵ, ∥∇p(x)∥ ≤ δ(10)
where w ∈ R q is bounded.
Theorem 3.11 (Relation of ϵ-stationary solutions). Let X = R q , and θ = 1. Let x γ be a bounded ϵ-stationary solution to (PP γ ). Then there exists a compact subanalytic set X C ⊂ R q with X C ∩X * v l,τ ̸ = ∅ and x γ ∈ X C . Suppose Assumption 1 holds, and that on X C , ∇v l,τ (x) exists and is ℓ v,2 -smooth, and v l,τ (x) is (c v , α v )-KL with α v ≥ 2. Then with proper choice of parameter γ, ϵ depending on δ, x γ is an (ϵ + δ, δ)-KKT point to the problem (9).
The ℓ v,2 -smoothness of ∇v l,τ can be justified under additional assumptions of the Hessian of f m for m ∈ [M ]. See a detailed discussion in Lemma D.13. Theorem 3.11 shows that for a simple bilevel problem where the lower-level objective v l,τ (x) satisfies the KL inequality with exponent α v ≥ 2, a stationary solution to the penalty reformulation approximates the KKT solution to the constrained formulation in (9). This indicates that though the KKT solution to a bilevel problem often requires the second-order information of the lower-level objective as shown in (10) and discussed in e.g., (Roy et al., 2023;Liu et al., 2022;Xiao et al., 2023b), one can still use first-order methods to approximate such solutions. The proof of Theorem 3.11 is provided in Appendix D.3. We first prove that the calmness condition holds in the above settings, ensuring the KKT conditions are necessary for global optimality.
this section cite: ['b72', 'b72', 'b53', 'b69', 'b53']

Section: Comparison with existing methods
To address the preference-guided MOL problem, one commonly used formulation is linear scalarization (LS) where Table 1. A recipe to choose hyperparameters θ, γ to obtain solutions to (PPγ) and thus the solutions to (CP). Denote ηp, αp as the HEB, KL exponents of p(x), and η, αv as the HEB, KL exponents of v l,τ (x) on a subanalytic and compact set, respectively.
(PPγ) X v l,τ (x) property θ p(x) property γ (CP) ϵ-global/local (Theorem 3.7-2) compact or R q η > 0 θ = ηp η ηp ≥ 1 ϵ 1 ηp -1 (ϵ, ϵ)-global/local (ϵ, ϵ)-global/local ϵ-stat. (Theorem 3.11) R q αv ≥ 2 θ = 1 αp ≥ 2 ϵ -1 (ϵ, ϵ)-KKT to (9) 0.0 0.2 0.4 0.6 0.8 1.0 1 0.0 0.2 0.4 0.6 0.8 1.0 2 LS output (a) LS 0.0 0.2 0.4 0.6 0.8 1.0 1 0.0 0.2 0.4 0.6 0.8 1.0 2 FERERO output (b) FERERO 0.0 0.2 0.4 0.6 0.8 1.0 1 0.0 0.2 0.4 0.6 0.8 1.0 2 Ours output (c) FOOPS Figure 3. Results of LS, FERERO, and FOOPS on Example 3.12. The black dashed lines is the preference defined by H(x) = 0. Green dots represent initial values, blue markers represent converged values for different methods.
the preference is modeled by different weights of the objectives. Another commonly used formulation is constrained vector optimization (Lin et al., 2019;Chen et al., 2024a), given by
min x∈X F (x), s.t. G(x) ≤ 0, H(x) = 0(11)
where G, H are vector-valued functions defined by userspecified perferences. Though the above formulation has successful applications in, e.g., multi-task learning, it is only guaranteed to converge to a KKT point of (11), which may be far away from the optimal solutions to min x∈X F (x). See Example 3.12, and the corresponding results in Figure 3. An intuitive explanation for this suboptimality is that (11) puts constraints at the lower level, emphasizing more on satisfying the constraints rather than minimizing the objectives.
Example 3.12. Let M = 2, q = 1, and M g = 0, M h = 1. Let X = R. The objective F and constraint H for the preference-constained formulation (11) is defined as
F (x) = 1 -e -∥x-1q∥ 2 2 , 1 -e -∥x+1q∥ 2 2 , (12a) H(x) =5f 1 (x) -4f 2 (x).(12b)
The corresponding preference function f 0 in formulation (OPS) to minimize the constraint violation of H is defined as f 0 (x) = ∥H(x)∥ 2 . In Figure 3, it is easy to see that there exists a solution x * in the Pareto set whose objective F (x * ) is at the intersection of the dashed line defined by H(x) = 0 and the solid curve representing the Pareto front. Therefore, x * is an optimal solution to both (OPS) and (11).
In this example, linear scalarization (LS) fails to converge to the preferred region that H(x) = 0, even after enumerating different weights. Indeed, we could prove that with  11) that satisfies ∇F (x)λ = 0 with λ ∈ ∆ M is a local maximum point of the objective λ ⊤ F (x). It is also worth noting that, there are some other examples where LS cannot find all points on the Pareto front even after enumerating all possible weights (Osyczka, 1984;Athan, 1994;Hu et al., 2023). See a detailed discussion in Appendix A.2. Proposition 3.13. Under certain initializations, there exists no λ ∈ ∆ M such that gradient descent algorithm on LS objective with weight λ converges to the solution of (12).
Moreover, algorithms developed under the preferenceconstrained formulation (11), such as PMTL (Lin et al., 2019) and FERERO (Chen et al., 2024a) (with the partial order cone being a nonnegative orthant cone R M + ), converge to a KKT point to (11), which is not necessarily Pareto optimal or Pareto stationary. For a more detailed discussion on this example, see Appendix A.2. Proposition 3.14 is provided to further support the claim that the KKT solution to ( 11) can be suboptimal for min x∈X F (x), with its proof in Appendix A.2. Proposition 3.14. The KKT solution to (11) is not necessarily Pareto stationary to min x∈X F (x).
Besides modeling preference by weights or constraints, there are also some works which model the preference by objectives, and formulate the problem as optimization on the Pareto set (OPS), as in (OPS). Among these methods, Preference-Based Pareto Descent Optimization (PB-PDO) (Kamani et al., 2021) and Pareto Navigation Gradient descent (PNG) (Ye & Liu, 2022) use a descent-type algorithm to ensure the output of the algorithm satisfies the preference and decreases the objectives F at each iteration. However, it has been shown in (Roy et al., 2023) that the stationary condition derived in (Ye & Liu, 2022) is not a necessary optimality condition for (OPS). Furthermore, as discussed in (Roy et al., 2023, Proposition 3), a nontrivial stationarity condition for (OPS) typically requires second-order derivatives of the objectives f m , m ∈ [M ]. Different from these methods, Target-Aware Weighted Training (TAWT) (Chen et al., 2022) and Pareto Majorization-Minimization (PMM) (Roy et al., 2023) convert the lowerlevel vector-valued objective to a scalar-valued objective through linear scalarization (LS), and optimize both the scalarization weight and the model parameter. However, as discussed in Example 3.12 and Appendix A.2, optimality for LS is not necessary for (OPS) unless f m are convex for all m ∈ [M ].
We summarize in Table 2 the key differences of our work compared to existing methods for OPS. See also a detailed review in Section 5 and Appendix A.
this section cite: ['b51', 'b67', 'b0', 'b39', 'b51', 'b43', 'b53', 'b69', 'b53', 'b69', 'b69']

Section: Algorithms and Analysis
In this section, we introduce practical first-order gradientbased algorithms to solve (PP γ ). We first update y to obtain an estimate for y * l,τ (x), and thus an estimate for v l,τ (x) = -h l,τ (x, y * l,τ (x)). Then we update x based on the estimated penalty function φ γ .
At the t-th outer iteration and the k-th inner iteration, we iteratively update x t and y t,k as follows.
y t,k+1 = U y (y t,k , ∆y t,k (y t,k , x t ); β t,k , k);
(13a)
x t+1 = U x (x t , ∆x t (x t , y t+1 ); α t , t) (13b
)
where U is some gradient based oracle, and the gradient vectors with respect to y and x are defined as
∆y t,k (y t,k , xt) = ∇yh l,τ (xt, y t,k ) ∆xt(xt, yt+1) = ∇f0(xt)-γtθ sign(vt)|vt| θ-1 ∇xh l,τ (xt, yt+1)
with y t+1 = y t,Kt and v t = τ ln M -h l,τ (x t , y t+1 ).
A meta algorithm with the above updates is summarized in Algorithm 1. We name it First-Order Optimization on the Pareto Set (FOOPS) algorithm.
Algorithm 1 The meta FOOPS algorithm with oracles
1: Initialize t = 0, x 0 , y 0 , set step sizes {α t , β t }, penalty parameter {γ t }, inner-loop iterations {K t }. 2: while ∥∇φ γt (x t )∥ 2 > ϵ do 3: Set k = 0; 4: for k = 0, . . . , K t -1 do 5:
Update y t,k by (13a);
6: end for 7: Set y t+1 = y t,Kt ; 8: Update x t by (13b); 9:
Set t = t + 1; 10: end while
We then discuss the choices of update oracles and the nonasymptotic convergence rate of Algorithm 1 with different oracles below. Let w represent the updated parameter, which can be either x and y, ∆w denote the gradient vector, α the stepsize, and t is the iteration number.
We give examples of oracles using projected gradient desent (PGD), and momentum updates (Momentum) in ( 14). The Nesterov's acceleration and Adam update are also applicable, which is detailed in Appendix E, one can also see in e.g., (Wang et al., 2024).
PGD: U(w, ∆w; α t , t) = Proj X (w -α t ∆w) (14a
) Momentum: U(w, ∆w; α t , t) = Proj X (w -α t v t ), with v t = αv t-1 + ∆w (14b
)
Discussion about the convergence. Since h l,τ (x, y) is µ hy -strongly convex w.r.t. y as detailed in the proof of Lemma E.4, when we choose U y as projected gradient descent, the momentum updates and Nesterov's acceleration, it gives linear convergence rate for the inner-loop of y. For the outer-loop w.r.t. x, if θ ≥ 1, then the objective φ γ (x) is differentiable and thus projected gradient descent, momentum, Nesterov's acceleration, Adam updates give O(1/T ) convergence rate in the deterministic setting and O(1/ √ T ) convergence rate in the stochastic setting according to (Wang et al., 2024). Therefore, combining outer-loop and innerloop update oracles together, Algorithm 1 converges. We provide a proof for the convergence of Algorithm 1 in Appendix E, Theorem E.6, when choosing both U y and U x as the PGD oracle and choosing θ ≥ 1, and assuming the objectives f m (x) for m = 0, . . . , M are smooth. For θ < 1, p(x) can be nonsmooth. The convergence for Algorithm 1 can be built upon nonsmooth optimization (Kiwiel, 2004;Davis et al., 2018), but possibly under additional assumptions. Some discussions about algorithms for nonsmooth lower-level objective are provided in (Chen et al., 2024b). When f 0 (x) and p(x) are Lipschitz, the algorithm can be applied to our problem. We leave a more detailed study of algorithm development in nonsmooth cases for future research.
Single-loop and stochastic variants. When we take the exact penalty method with η p = 1, γ t and K t can be upper bounded by a constant. In fact, under Assumption 2, when l > ℓ f,1 , a single-loop variant of Algorithm 1 that takes K t = 1 also has non-asymptotic convergence guarantees, see e.g. (Chen et al., 2021). Besides, stochastic variants of Algorithm 1 can also be derived which replace the deterministic gradients with their unbiased stochastic estimates. We leave the development and convergence analysis of such variants for future work.
this section cite: ['b76', 'b76', 'b45', 'b20']

Section: Related Works
We discuss recent works that are most related to ours. An extended discussion is provided in Appendix A.1. Preference-vector-guided multi-objective learning. Preferences in multi-objective optimization can be represented using weights, thresholds, or preference vectors. Scalarization methods, such as linear and Tchebycheff scalarization, convert vector objectives into scalar objectives by applying weighted norms (Miettinen, 1998). Alternatively, ϵ-constraint methods impose thresholds on objectives to convert the problem to a constrained optimization problem (Curtis et al., 2023). There are also other approaches which represent preferences with vectors in the objective space, focusing on finding optimal solutions satisfying constraints defined by these vectors (Lin et al., 2019;Chen et al., 2024a) or minimizing distances to the vectors (Mahapatra & Rajan, 2020;Momma et al., 2022). See also a comprehensive review in (Chen et al., 2025). Optimization on the Pareto Set. In machine learning, there are specific instantiations of the OPS problem. For example, EPO (Mahapatra & Rajan, 2020) and Preference-Based Pareto Descent Optimization (PB-PDO) (Kamani et al., 2021) find a Pareto model such that the objective values satisfy a ratio constraint by minimizing the non-uniformity score. Specifically, EPO finds update directions to improve the objective values, or to reduce the constraint violations, or both. PB-PDO finds common descent directions for the lower-level multi-objectives and the upper-level objective. They are not guaranteed to converge to a necessary condition of the OPS problem. Target-Aware Weighted Training (TAWT) (Chen et al., 2022) learns a multi-task model that minimizes the discrepancy of task representations to ensure they are similar. TAWT converts the lower-level objectives to a linearly scalarized objective, and optimizes the scalarization weights in the upper level. This approach has the limitation of introducing undesired local solutions. (Simple) bilevel optimization. Problem (CP) is a constrained reformulation of the simple bilevel program with a generally nonconvex lower-level (LL) objective. For nonconvex LL objectives, algorithms are proposed in e.g., (Liu et al., 2021b;Huang, 2023;Xiao et al., 2023b). However, they usually require second-order derivatives in the algorithms, which can be expensive to implement. More recently, first-order Hessian-free approaches were proposed to address this by the value function reformulation of BLO. For example, sequential quadratic programming (Liu et al., 2022), and smoothed Lagrangian method (Lu, 2023) were used to solve the constrained problem. Later on, a penaltybased algorithm was proposed (Shen et al., 2025). Variants such as Moreau Envelope based algorithms (Kwon et al., 2024;Liu et al., 2024) and adaptive algorithms (Chen et al., 2024b) were proposed. However, none of these works tackle BLO with vector-valued LL objective. Moreover, even af-ter converting the vector-valued LL objective to a scalarvalued one through the merit function v l,τ , the LL objective is generally nonconvex and non-PL even if the objectives f m , m ∈ [M ] are all convex or PL. Therefore, it is difficult to directly apply the existing analysis or algorithms to our problem.
this section cite: ['b63', 'b19', 'b51', 'b60', 'b64', 'b17', 'b60', 'b43', 'b40', 'b53', 'b72', 'b48', 'b55']

Section: Experiments
In this section, we conduct experiments to verify our theory and show the applicability of the algorithms to preferenceguided multi-task learning. We use LS, PMTL (Lin et al., 2019), EPO (Mahapatra & Rajan, 2020), XWC-MGDA (XM) (Momma et al., 2022), FERERO (Chen et al., 2024a) as baselines for comparison. For a preference vector-guided MOL problem, we define f 0 (x) = ∥H(x)∥ 2 , where H(x) is the equality constraint function derived from the preference vector (Chen et al., 2024a).
this section cite: ['b51', 'b60', 'b64']

Section: Metrics.
Objective loss and accuracy. We report the objective losses and accuracies in classification. Hypervolume. Let F ′ ∈ R M denote the Nadir point, i.e., the worst performance on single-task baselines, and S denote a set of objective function values of the obtained models. Hypervolume measures the size of the dominated space of S relative to F ′ , which can be computed by H
(S) = Λ({q ∈ R M | ∃F ∈ S : F ≤ q ≤ F ′ }),
where Λ(•) denotes the Lebesgue measure. Additional details. The implementation and additional experiments can be found in Appendix F. 0.0 0.2 0.4 0.6 0.8 1.0 1 0.0 0.2 0.4 0.6 0.8 1.0 2 Initial values PMTL output (a) PMTL 0.0 0.2 0.4 0.6 0.8 1.0 1 0.0 0.2 0.4 0.6 0.8 1.0 2 FERERO output (b) FERERO 0.0 0.2 0.4 0.6 0.8 1.0 1 0.0 0.2 0.4 0.6 0.8 1.0 2 Ours output (c) FOOPS Example 3.12. Following (Lin et al., 2019;Mahapatra & Rajan, 2020), the first objective we consider is (12a) in Example 3.12, but with q = 20. The results for the experiments with hard initialization are displayed in Figure 4. They show that under certain initializations and preferences, algorithms developed under (11) such as PMTL and FERERO (with A = I therein) could fail to reach the Pareto front. It further justifies our Proposition 3.14 that the KKT solution to ( 11), with preferences modelled by the constraints at the lower level, can be suboptimal for min x∈X F (x). In contrast, FOOPS successfully converges to preferred Pareto optimal solutions under different preferences. It demonstrates the benefit of the OPS formulation over (11), by prioritizing the
0.3 0.4 0.5 0.6 0.7 0.3 0.4 0.5 0.6 0.7 (a) Multi-MNIST loss 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.4 0.5 0.6 0.7 0.8 0.9 1.0 (b) Multi-Fashion loss LS EPO PMTL XM FERERO FOOPS 0.2 0.3 0.4 0.5 0.6 0.7 0.2 0.3 0.4 0.5 0.6 0.7 (c) Multi-F+M loss  attainment of weak Pareto optimality instead of the optimality of the preference function.
Multi-patch image classification. Following (Momma et al., 2022), we use Multi-MNIST (Mt-M), Multi-Fashion (Mt-F), and Multi-Fashion+MNIST (Mt-F+M) for image classification. The two tasks or objectives in all three datasets are to classify the top-left and the bottom-right images, respectively. We use LeNet as the backbone neural network. The losses of different methods given different preference vectors are plotted in Figure 5. Experiments for our method are repeated 5 times. Hypervolumes with means and standard deviations are reported in Table 3. The results for other methods in Table 3 are referenced from (Momma et al., 2022;Chen et al., 2024a). The results show that FOOPS is better at obtaining large hypervolumes, see Table 3, but worse at aligning with preferences compared to other methods, see Figure 5.
this section cite: ['b51', 'b60', 'b64', 'b64']

Section: Multi-lingual speech recognition.
We use the proposed method to fine-tune a pre-trained multi-lingual speech recognition model. The datasets include Librispeech (100 hours) (Panayotov et al., 2015), and AISHELL v1 (Bu et al., 2017).
The model architecture is a conformer with 8 blocks. The speech recognition Connectionist Temporal Classification (CTC) losses in Chinese and English are denoted as f ch t and f en t , respectively. We also use the self-supervised Contrastive Predictive Coding (CPC) loss f p for representation learning, i.e., min f0(x) := ∥f ch t (x) -f en t (x)∥ 2 (15a) s.t. x ∈ arg min F (x) := fp(x), f ch t (x), f en t (x) ⊤ (15b)
where the lower-level objective f p ensures the model learns a good representation, and the upper-level objective ensures the difference of the performances on both languages is small; see more details in Appendix F. The results are reported in Table 4, which show that FOOPS demonstrate competitive average performance on different languages, but is less good at optimizing the fairness preference function f 0 . This observation is consistent with that in Figure 5.
this section cite: ['b68', 'b10']

Section: Conclusions
In this work, we cast preference-guided multi-objective learning as an optimization on the Pareto set (OPS) problem, which is essentially a semivectorial simple bilevel optimization problem, with a lower-level vector-valued objective, and an upper-level scalar-valued preference objective. We propose a first-order penalty method to solve the problem, where the penalty function is the polynomial of a smoothed merit function. For the theoretical analysis, first we establish the properties of the merit function, including its relation to weak Pareto optimality, and the Hölderian error bound. Then we discuss the relation of solutions to the penalty reformulation and the original OPS problem. Part of our theoretical analysis is of independent interest to simple bilevel optimization problem. Interestingly and perhaps surprisingly, our analysis shows that although the stationary condition of the OPS problem usually requires second-order derivative information, it can be approximated using firstorder methods. Based on the results, we develop first-order algorithms to solve the penalty problem and provide their convergence guarantees. For the empirical experiments, we apply the proposed method to synthetic and real-world problems, which demonstrate the effectiveness of the proposed method in finding preference-guided optimal solutions.
For nonconvex LL objectives, a pessimistic algorithm with asymptotic convergence guarantee was proposed in (Liu et al., 2021b), the stationary metric was studied for BLO with lower-level PL objective, and an alternating descent algorithm with non-asymptotic analysis was proposed in (Xiao et al., 2023b). In Table 5, we provide a summary of the bilevel optimization works with possibly nonconvex lower-level objectives which may satisfy the Hölderian error bound (HEB).
Semivectorial BLO. OPS can be seen as a semivectorial simple BLO problem, where the bilevel program has a vectorvalued lower-level (LL) objective and a scalar-valued upper-level (UL) objective. To solve such problems, one straightforward approach is to convert the LL vector-valued objective to a scalar-valued objective through scalarization, and to optimize the scalarization parameter in the upper level, see, e.g., (Roy et al., 2023). However, it has been shown that this reformulation could induce additional local minimizers or stationary solutions (Dempe & Mehlitz, 2019;Benko & Mehlitz, 2021). Furthermore, this reformulation might require stronger constraint qualifications than the original problem (Benko & Mehlitz, 2021). Alternatively, penalty-based reformulations have been considered (Bonnel & Morgan, 2006), where the penalty function is defined as the maximum improvement amount of the vector-valued objective. However, no practical implementation, or relation of solutions to the original problem in nonconvex settings, or non-asymptotic convergence analysis are provided for the reformulation. Recently, in (Giovannelli et al., 2024), deterministic and stochastic risk-neutral and risk-averse algorithms are proposed, under the assumption that the LL objective is strictly convex w.r.t. the LL variable, and requiring the second-order derivative of the objective.
Another line of research study the problem with vector-valued upper-level (UL) objective, and scalar-valued LL objective (Ye et al., 2021;Gu et al., 2023;Ye et al., 2024;Yang et al., 2024). It is sometimes also referred to as the multi-objective BLO problem. For a more detailed review of multi-objective BLO algorithms, see a survey (Mejía-De-Dios et al., 2023) and the references therein. Besides, single-level multi-objective learning algorithms have been extensively studied, these include the variants of the multi-gradient descent algorithm (Fliege & Svaiter, 2000;Sener & Koltun, 2018;Liu et al., 2021a;Liu & Vicente, 2021;Fernando et al., 2023;Chen et al., 2023;Xiao et al., 2023a) that are designed to avoid gradient conflicts during the optimization procedure. For a more detailed review of multi-objective learning algorithms, see a survey (Chen et al., 2025) and the references therein.
Comparison of the theory in Section 3.2 to existing works. The works most related to ours regarding the theory in Section 3.2 include (Ye et al., 1997;Luo et al., 1996b) and recent works (Shen et al., 2025;Chen et al., 2024b). However, a major difference is that they directly assume the lower-level objective satisfies HEB, while in our work, we prove the property holds on a bounded set for v l,τ (x) and p(x) when the objective F is subanalytic. The two works (Ye et al., 1997;Luo et al., 1996b) focus on the cases with exact penalty. Other differences include that the results in (Shen et al., 2025) only consider LL objective satisfies HEB with η = 2, while we consider more general η. Also, we do not require the global convexity or Lipschitz assumption as in (Chen et al., 2024b), which is generally not satisfied by our merit function. Furthermore, we provide the relation between the stationary solution of the penalty formulation and the KKT solution of the constrained formulation under the general KL inequality, which is not discussed in either of the two works.
this section cite: ['b69', 'b21', 'b3', 'b3', 'b9', 'b33', 'b82', 'b80', 'b28', 'b71', 'b52', 'b56', 'b27', 'b12', 'b17', 'b84', 'b59', 'b72', 'b84', 'b59', 'b72']

Section: A.2. Limitation of linear scalarization and preference as constraint
Besides using empirical results on Example 3.12, we also provide theoretical justifications to show the limitations of LS and KKT solutions for preference-guided MOL.
Limitation of linear scalarization. We first discuss the limitation of linear scalarization (LS) for preference-guided MOL.
It is known that LS is not good at handling nonconvex Pareto front. Prior works have shown that the optimality condition of LS is not a necessary condition for Pareto optimality, see e.g., (Athan, 1994, Proposition 3.3). As a result, the solution set of LS, even by enumerating all possible weights of objectives, does not include all Pareto optimal solutions, and thus it does not include Pareto optimal solutions under certain preferences.
Below we provide the proof of Proposition 3.13, which shows that under certain initializations, gradient descent on the LS objective does not converge to a preferred Pareto optimal solution in Example 3.12.
Proof of Proposition 3.13. In Example 3.12, there exists a solution x * ∈ (-1, 1), which is an optimal solution to both (OPS) and ( 11). And there exists
λ = [λ 1 , λ 2 ] ⊤ ∈ ∆ 2 with λ 2 = λ 1 (1-x * )e -(x * -1) 2 (1+x * )e -(x * +1) 2 that ∇F (x * )λ = 0.
To show this, let c denote Table 5. Comparison with existing methods for (simple) bilevel optimization with lower-level scalar objective, "SC" and "C" represent "strongly convex" and "convex", respectively; "comp" represents "compact set"; "Lip" represents "Lipschitz continuous". For non-simple bilevel optimization problem, the lower-level and upper-level properties are all w.r.t. the lower-level variable for a meaningful comparison.
The lower-level objective in our problem is v l,τ (x) + τ ln M .
Method lower-level HEB other lower-level properties upper-level first-order non-simple bilevel optimization IAPTT-GM (Liu et al., 2021b) smooth, comp smooth, comp ✗ BOME (Liu et al., 2022) η = 2 PL, Lip, smooth Lip, smooth ✓ PBGD (Shen et al., 2025) η = 2 PL, smooth Lip, smooth ✓ C, smooth Lip, smooth GALET (Xiao et al., 2023b) η = 2 PL, smooth Lip, smooth ✗ AGILS (Bai et al., 2024) η ≥ 1 KL, weakly C (composite) smooth ✓ MEHA (Liu et al., 2024) -smooth smooth ✓ weakly C (composite) smooth SLM (Lu, 2023) η = 2 PL, smooth smooth, comp ✓ simple bilevel optimization CG-BiO (Jiang et al., 2023) η ≥ 1 C, smooth C, smooth ✓ C, smooth non-C, smooth R-APM (Samadi et al., 2024) η = 1 C, composite C, smooth ✓ PB-APG (Chen et al., 2024b)
η ≥ 1 C, composite C, composite ✓ C, composite SC, composite nonsmooth, Lip nonsmooth, Lip FOOPS (ours) η > 0, (modified by θ = ηp η ≥ 1 η ) subanalytic (provable) locally Lip ✓ KL (provable)
locally Lip a positive constant, note that
∇F (x)λ =2λ 1 e -(x-1) 2 (x -1) + 2λ 2 e -(x+1) 2 (x + 1) =c 2e -(x-1) 2 (x -1)(1 + x * )e -(x * +1) 2 + 2e -(x+1) 2 (x + 1)(1 -x * )e -(x * -1) 2 =2c(x + 1)(x * + 1)e -(x+1) 2 -(x * +1) 2 e 4x x -1 x + 1 - x * -1 x * + 1 e 4x * . (16
)
Let r(x) = x-1 x+1 e 4x . Then r(x) > r(x * ) ⇐⇒ ∇F (x)λ > 0 and vice versa for r(x) < r(x * ). The above equation implies that when r(x ′ ) = r(x * ), ∇F (x ′ )λ = 0. Therefore, ∇F (x * )λ = 0. Also observing that there exists different points
-1 < x 1 < x * < x 2 < 1 that r(x 1 ) = r(x 2 ) = r(x * ).(17)
This means that x 1 , x 2 are all stationary points of λ ⊤ F (x). Furthermore, 1. for
x ′ ∈ (x 1 , x * ) ∪ (x 2 , 1), r(x ′ ) > r(x * ), thus ∇F (x ′ )λ > 0, then gradient descent (GD) on λ ⊤ F (x) starting from x ′ ∈ (x 1 , x *
) with sufficiently small step size converges to x 1 , and it converges to x 2 if starting from x ′ ∈ (x 2 , 1); 2. for
x ′ ∈ (-1, x 1 ) ∪ (x * , x 2 ), r(x ′ ) < r(x * ), thus ∇F (x ′ )λ < 0, then GD on λ ⊤ F (x)
starting from x ′ ∈ (-1, x 1 ) with sufficiently small step size converges to x 1 , and it converges to x 2 if starting from x ′ ∈ (x * , x 2 ).
This proves that they will not converge to x * .
Limitation of preference as constraint. We then discuss in more detail of the limitation of modeling preference by constraints. We verify Proposition 3.14 by constructing an example with a KKT but non-Pareto stationary solution. Using Example 3.12, and noticing that one solution that PMTL converges to, denoted as x * , has objective value that is approximately F (x * ) ≈ [0.80; 0.99], which satisfies the feasibility condition that H(x * ) = 0. Furthermore, its gradient at x * can be computed approximately as
∇f 1 (x * ) ≈0.5062; (18a) ∇f 2 (x * ) ≈0.0002.(18b)
Clearly, x * is not Pareto stationary. By the KKT stationarity condition, we further have
∇F (x)λ f + ∇H(x)λ h = ∇F (x) λ f + 5 -4 λ h =∇F (x) λ f + 5λ h -4λ h = 0, for some λ f ∈ ∆ 2 , λ h ∈ R.(19)
It can be verified that 0.5062(λ f,1 + 5λ h ) + 0.0002(1 -λ f,1 -4λ h ) = 0 has solutions, e.g., λ f = [0; 1], λ h ≈ -7.9 × 10 -5 . Therefore, x * is a KKT point.
We use another example with strongly convex objectives to prove Proposition 3.14 that a KKT point to ( 11) is not necessarily Pareto stationary.
Example A.1. Let M = 2, q = 2, and
M g = 0, M h = 1. Let X = R 2 , and x = [x 1 ; x 2 ].
The objective F and constraint H is defined as
F (x) = (x 1 -1) 2 + x 2 2 , 0.5x 2 1 + x 2 2 (20a) H(x) =9f 1 (x) -8f 2 (x)(20b)
Proof of Proposition 3.14. In Example A.1, the gradient of F can be computed by
∇F (x) = 2(x 1 -1) x 1 2x 2 2x 2 .(21)
For x = [3; 0], ∇F (x) = 4 3 0 0 . Let ∆ M denote the (M -1)-simplex. Apparently, there exists no λ ∈ ∆ 2 such that ∇F (x)λ = 0. Therefore, x is not Pareto stationary.
Then we check whether x satisfies the KKT condition. First, it satisfies the feasiblity condition since H(x) = 0. Second, by invoking the KKT stationarity condition, for some λ f ∈ ∆ 2 , λ h ∈ R, we have
∇F (x)λ f + ∇H(x)λ h = ∇F (x) λ f + 9 -8 λ h = 4 3 0 0 λ f + 9λ h -8λ h = 0. (22
)
It can then be verified that the above holds true when λ f = [0, 1] ∈ ∆ 2 , and λ h = -0.25.
Therefore, x is a KKT point but not a Pareto stationary point. The proof is complete.
this section cite: ['b53', 'b72', 'b2', 'b55', 'b42', 'b70']

Section: B. Proof of the properties of the merit functions
For convenience, we define the merit function u l (x) and restate the smoothed merit function v l,τ (x) below.
u l (x) := max y∈X min m∈[M ] f m (x) -f m (y) - l 2 ∥x -y∥ 2 (23) v l,τ (x) := -min y∈X τ ln M m=1 e fm(y)-fm (x) τ + l 2 ∥x -y∥ 2 .(24)
Correspondingly, we define h l,τ (x, y) below for analysis. Note that v l,τ (x) = -min y∈X h l,τ (x, y). Proposition B.2 (Smoothness implies weak convexity). If a locally Lipschitz function f is ℓ f,1 -smooth, then it is also -ℓ f,1 -weakly convex. Lemma B.3 (Log-sum-exp function preserves weak convexity). Let f m (x), m ∈ [M ] be weakly convex with modulus µ m ∈ R. Let μ = min m∈[M ] µ m . Then ln M m=1 e fm(x) is weakly convex with modulus μ.
Proof of Lemma B.3. By definition, and since μ = min m∈[M ] µ m , we have f m (x) -μ 2 ∥x∥ 2 is convex for all m ∈ [M ]. Also because the Log-sum-exp function preserves convexity, we have that ln M m=1 e fm(x)-μ 2 ∥x∥ 2 is convex. Further rearranging this function, we have
ln M m=1 e fm(x)-μ 2 ∥x∥ 2 = ln e -μ 2 ∥x∥ 2 M m=1 e fm(x) = ln M m=1 e fm(x) - μ 2 ∥x∥ 2(26)
which is convex. By definition, this implies that ln M m=1 e fm(x) is weakly convex with modulus μ. The proof is complete.
Corollary B.4. If f m (x), m ∈ [M ] is weakly convex with modulus µ m ∈ R, and l + min m∈[M ] µ m > 0, then h l,τ (x, y) defined in (25) is strictly convex w.r.t. y, and the solution to min y∈X h l,τ (x, y) is a singleton. is μ-weakly convex w.r.t. y, where μ = min m∈[M ] µ m . Since l + min m∈[M ] µ m = l + μ > 0, h l,τ (x, y) is strictly convex w.r.t. y, and the solution to min y∈X h l,τ (x, y) is unique. The proof is complete.
Lemma B.5. If f m (x), m ∈ [M ] is continuous and weakly convex with modulus µ m , and l ≥ -min m∈[M ] µ m , then there exists x ∈ X such that v l,τ (x) = -τ ln M .
Proof of Lemma B.5. From Corollary B.4, since l ≥ -min m∈[M ] µ m , h l,τ (x, y) is convex w.r.t. y. Let P be the indicator function defined on X . Then, 0 ∈ ∇ y h l,τ (x, y) + ∂P (y) if and only if y = arg min y∈X h l,τ (x, y).
By the definition of h l,τ (x, y), the gradient ∇ y h l,τ (x, y) can be derived as
∇ y h l,τ (x, y) = M m=1 e fm(y)-fm(x) τ M m=1 e fm(y)-fm (x) τ ∇f m (y) + l(y -x).(27)
When y = x, it can be further derived that
∇ y h l,τ (y, y) = 1 M M m=1 ∇f m (y).(28)
Recall that λ ⊤ F (x) is lower bounded for all λ ∈ ∆ M . And λ ⊤ F (x) is continuous since f m are continuous for all m ∈ [M ].
We assume either X is compact, or X = R q and f m is coercive for all m ∈ [M ]. Then the solution to
min x∈X λ ⊤ F (x) exists. Let x * = arg min x∈X 1 M M m=1 f m (x), which implies 0 ∈ 1 M M m=1 ∇f m (x * ) + ∂P (x * ).(29)
Combining ( 29) with (28), we have that 0 ∈ ∇ y h l,τ (x, y) + ∂P (y) | (x,y)=(x * ,x * ) . Therefore, x * ∈ arg min y∈X h l,τ (x * , y), and thus
min y∈X h l,τ (x * , y) = h l,τ (x * , x * ) = τ ln M m=1 e fm(x * )-fm (x * ) τ = τ ln M.(30)
By definition, v l,τ (x * ) can be computed by
v l,τ (x * ) = -min y∈X h l,τ (x * , y) = -τ ln M(31)
which completes the proof. For any sequence {x t } ⊆ X satisfying lim t→∞ x t = x ∈ X , given any ϵ > 0, let ȳ ∈ X satisfy h l,τ (x, ȳ) ≤ min y∈X h l,τ (x, y) + ϵ. As h l,τ is continuous at (x, ȳ), there exists T > 0 such that
min y∈X h l,τ (x t , y) ≤ h l,τ (x t , ȳ) ≤ h l,τ (x, ȳ) + ϵ ≤ min y∈X h l,τ (x, y) + 2ϵ, ∀t > T,(32)
and thus
lim sup t→∞ min y∈X h l,τ (x t , y) ≤ min y∈X h l,τ (x, y) + 2ϵ.(33)
As the above inequality holds for any ϵ > 0, we obtain,
lim sup t→∞ min y∈X h l,τ (x t , y) ≤ min y∈X h l,τ (x, y)(34)
which proves that v l,τ (x) is lower semi-continuous.
Proof of 2). We prove the Lipschitz continuity of h l,τ (x, y) below. We define π m (x, y) := e
1 τ (fm(y)-fm(x)) M m=1 e 1 τ (fm(y)-fm(x))
.
(35)
Note that
∥∇ x h l,τ (x, y)∥ ≤ M m=1 π m (x, y)∇f m (x) + ∥l(x -y)∥ ≤ ℓ f + l(∥x∥ + ∥y∥) ≤ ℓ f + 2lℓ x ,(36)
∥∇ y h l,τ (x, y)∥ ≤ M m=1 π m (x, y)∇f m (y) + ∥l(x -y)∥ ≤ ℓ f + l(∥x∥ + ∥y∥) ≤ ℓ f + 2lℓ x .(37)
Therefore, h l,τ (x, y) is (ℓ f + 2lℓ x )-Lipschitz continuous w.r.t. both x ∈ X C and y ∈ X C .
Next we prove the Lipschitz continuity of the merit function v l,τ under additional assumptions. From Assumption 1, the functions f m (x), m = 0, . . . , M are ℓ f -Lipschitz on a bounded set X C where ∥x∥ ≤ ℓ x . From (5), we can compute ∇v l,τ (x). We then derive the bound of ∥∇v l,τ (x)∥ below.
∥∇v l,τ (x)∥ = M m=1 π m (x)∇f m (x) -l(x -y * l,τ (x)) ≤ℓ f + l∥x∥ + l∥y * l,τ (x)∥ ≤ ℓ f + 2lℓ x (38
) which proves that v l,τ (x) is (ℓ f + 2lℓ x )-Lipschitz continuous on X . Recall that p(x) = v l,τ (x) + τ ln M θ . For θ ≥ 1, the gradient of p(x) is given by ∇p(x) = θ v l,τ (x) + τ ln M θ-1 ∇v l,τ (x)(39)
Note that v l,τ (x) + τ ln M is bounded on a compact set since v l,τ (x) is Lipschitz on this set, i.e.,
v l,τ (x) + τ ln M ≤ ℓ v l,τ ∥x -x * ∥ ≤ 2ℓ v l,τ ℓ x with ℓ v l,τ = ℓ f + 2lℓ x .(40)
Then ∥∇p(x)∥ can be bounded by
∥∇p(x)∥ ≤θℓ v l,τ v l,τ (x) + τ ln M θ-1 ≤ θℓ v l,τ v l,τ (x) + τ ln M θ-1 (41
)
≤θℓ v l,τ 2ℓ v l,τ ℓ x θ-1 = θ 2ℓ x θ-1 ℓ θ v l,τ(42)
which proves that p(x) is θ 2ℓ x θ-1 ℓ θ v l,τ -Lipschitz continuous on X .
y * l,τ (x) := arg min y∈X h l,τ (x, y) = arg min y∈X τ ln M m=1 e fm(y)-fm (x) τ + l 2 ∥x -y∥ 2 . (43
) For l -ℓ f,1 ≥ µ hy > 0, there exists ℓ y * l,τ = 2M ℓ f τ ℓ 2 f τ + ℓ f,1 + 4M ℓ 3 f τ 2 > 0 that for all x, x ′ ∈ X C , the following holds ∥y * l,τ (x) -y * l,τ (x ′ )∥ ≤ ℓ y * l,τ ∥x -x ′ ∥.(44)
Proof. By Corollary B.4, for l + min m∈[M ] µ m ≥ µ hy > 0, the function h l,τ (x, y) is µ hy -strongly convex w.r.t. y. Therefore, from (Dontchev & Rockafellar, 2009, Theorem 2F.7), or using similar arguments for the proof in (Chen et al., 2023, Lemma 15), we can derive that
∥y * l,τ (x) -y * l,τ (x ′ )∥ ≤ µ -1 hy ∥∇ 2 yy h l,τ (x, y) -∇ 2 yy h l,τ (x ′ , y)∥.(45)
Let I q ∈ R q×q denote the identity matrix, then ∇ 2 yy h l,τ (x, y) can be further computed by
∇ 2 yy h l,τ (x, y) = ∇ y M m=1 e fm(y)-fm (x) τ M m=1 e fm(y)-fm (x) τ πm(x,y) ∇f m (y) + l(y -x) =∇ y ∇F (y)π(x, y) S(x,y) +l(y -x) with π(x, y) = [π 1 (x, y), . . . , π M (x, y)] ⊤ = 1 τ M m=1 π m (x, y)∇f m (y)∇f m (y) ⊤ - 1 τ S(x, y)S(x, y) ⊤ + M m=1 π m (x, y)∇ 2 f m (y) + lI q . (46
)
We first bound ∥S(x, y)S(x, y) ⊤ -S(x ′ , y)S(x ′ , y) ⊤ ∥ by
∥S(x, y)S(x, y) ⊤ -S(x ′ , y)S(x ′ , y) ⊤ ∥ ≤ ∥S(x, y)∥ + ∥S(x ′ , y)∥ ∥S(x, y) -S(x ′ , y)∥.(47)
Then from Assumptions 4 and 1, we can bound ∥∇ 2 yy h l,τ (x, y) -∇ 2 yy h l,τ (x ′ , y)∥ by
∥∇ 2 yy h l,τ (x, y) -∇ 2 yy h l,τ (x ′ , y)∥ ≤ 1 τ M m=1 ∥π m (x, y) -π m (x ′ , y)∥∥∇f m (y)∥ 2 + 1 τ ∥S(x, y)∥ + ∥S(x ′ , y)∥ ∥S(x, y) -S(x ′ , y)∥ + M m=1 ∥π m (x, y) -π m (x ′ , y)∥∥∇ 2 f m (y)∥ ≤ M m=1 ∥π m (x, y) -π m (x ′ , y)∥ ℓ 2 f τ + ℓ f,1 + 2ℓ f τ ∥S(x, y) -S(x ′ , y)∥(48)
where ∥π m (x, y) -π m (x ′ , y)∥ can be further bounded by
∥π m (x, y) -π m (x ′ , y)∥ ≤ 2ℓ f τ ∥x -x ′ ∥.(49)
Similarly, ∥S(x, y) -S(x ′ , y)∥ can be further bounded by
∥S(x, y) -S(x ′ , y)∥ ≤ M m=1 π m (x, y) -π m (x ′ , y) ∇f m (y) ≤ 2M ℓ 2 f τ ∥x -x ′ ∥. (50
)
The proof is complete with
ℓ y * l,τ = 2M ℓ f τ ℓ 2 f τ + ℓ f,1 + 4M ℓ 3 f τ 2 .
this section cite: ['b22']

Section: B.3. Proof of Proposition 3.3: relations of v l,τ and weak Pareto optimality
Proof of Proposition 3.3. We prove each property as follows.
Property 1. For the first argument, by the property of the Log-sum-exp function (Nesterov, 2005), and since taking min y∈X preserves inequality, we have that
u l (x) -τ ln M ≤ v l,τ (x) ≤ u l (x).(51)
This implies that, as τ ↓ 0, v l,τ (x) uniformly converges to u l (x). Also recall from Lemma B.1 that x is weakly Pareto optimal if and only if u l = 0. Therefore, x is weakly Pareto optimal if and only if lim τ ↓0 v l,τ (x) = 0. The first argument is proved.
For the second argument, from (Nesterov, 2005), we have that
τ ln M m=1 e fm (y)-fm(x) τ + l 2 ∥x -y∥ 2 ≤ τ ln M + max m∈[M ] {f m (y) -f m (x)} + l 2 ∥x -y∥ 2 . (52
)
Since taking min y∈X preserves inequality, it implies that
min y∈X τ ln M m=1 e fm(y)-fm (x) τ + l 2 ∥x -y∥ 2 ≤ τ ln M + min y∈X max m∈[M ] {f m (y) -f m (x)} + l 2 ∥x -y∥ 2(53)
which proves that
v l,τ (x) ≥ -min y∈X max m∈[M ] {f m (y) -f m (x)} + l 2 ∥x -y∥ 2 -τ ln M =u l (x) -τ ln M ≥ -τ ln M (54
)
where the last inequality holds because u l (x) ≥ 0. Furthermore, there exists x ∈ X such that v l,τ (x) = -τ ln M by Lemma B.5. Then min x∈X v l,τ (x) = -τ ln M .
Property 2. For the first argument, by Lemma B.1, if x is weakly Pareto optimal, then u l (x) = 0. Furthermore, by Property-1, u l (x) ≥ v l,τ (x), which proves v l,τ (x) ≤ 0.
Conversely, for the second argument, for condition a),
l = 0, v 0,τ (x) ≤ 0 implies that ū(x) ≤ v 0,τ (x) + τ ln M ≤ τ ln M.(55)
For all z ∈ X , the directional derivative of v l,τ , denoted as v ′ l,τ (x; z -x), can be computed by
v ′ l,τ (x; z -x) = M m=1 π m (x, y * l,τ (x))f ′ m (x; z -x) -l x -y * l,τ (x) ⊤ (z -x).(63)
Proof of Lemma B.11. Recall that we have defined
h l,τ (x, y) = τ ln M m=1 e fm (y)-fm(x) τ + l 2 ∥x -y∥ 2 . (64
) By definition, v l,τ (x) = -min y∈X h l,τ (x, y) = -h l,τ (x, y * l,τ (x)), with y * l,τ (x) ∈ arg min y∈X h l,τ (x, y). If l + min m∈[M ] µ m ≥ c > 0, by Corollary B.4, y * l,τ (x) is unique. Furthermore, y * l,τ (x) is continuous w.r.t.
x. By the extended Danskin-type theorem in e.g., (Shen et al., 2025, Proposition 5), v l,τ (x) is differentiable. Its gradient can be computed by
∇v l,τ (x) = -∇ x h l,τ (x, y * l,τ (x)) = M m=1 e fm(y * l,τ (x))-fm(x) τ M m=1 e fm(y * l,τ (x))-fm(x) τ ∇f m (x) -l(x -y * l,τ (x)).(65)
Then given all x, z ∈ X , the directional derivative of v l,τ (x) can be computed by
v ′ l,τ (x; z -x) = M m=1 π m (x, y * l,τ (x))f ′ m (x; z -x) -l x -y * l,τ (x) ⊤ (z -x) for all z ∈ X (66
)
where π m (x, y) =
e fm (y)-fm(x) τ M m=1 e fm(y)-fm(x) τ
. The proof is complete.
this section cite: ['b65', 'b65']

Section: C. Subanalyticity and related properties
In this section, we first discuss some preliminaries on subanalyticity, and then prove the global subanalyticity of the merit function v l,τ (x).
Definition C.1 (Subanalyticity (Bierstone & Milman, 1988)). 1) A subset S ⊂ R q is called semianalytic if each point of R q admits a neighborhood V for which S ∩ V assumes the following form
I i=1 J j=1 {x ∈ V : f ij (x) = 0, g ij (x) > 0} ,(67)
where
f ij , g ij : V → R are real analytic functions for 1 ≤ i ≤ I, 1 ≤ j ≤ J. 2) A subset S ⊂ R q is called subanalytic if each point of R q admits a neighborhood V such that S ∩ V = {x ∈ R q | (x, y) ∈ B} (68
) where B is a bounded semianalytic subset of R q × R m . 3) A function f : R q → R ∪ {+∞} is called subanalytic if its graph is a subanalytic subset of R q × R.
Definition C.2 (Global subanalyticity (Dries & Miller, 1996, p. 506)). Let x = [x 1 , . . . , x q ] ⊤ ∈ R q . Define the function
Φ q (x) := x 1 1 + x 2 1 , . . . , x q 1 + x 2 q . (69
) 1) A subset S of R q is called globally subanalytic if its image under Φ q is a subanalytic subset of R q . 2) A function f : R q → R ∪ {+∞} is called globally subanalytic if its graph is a globally subanalytic subset of R q × R.
Proposition C.3 ( (Bolte et al., 2007)). Globally subanalytic sets are subanalytic, and conversely, any bounded subanalytic set is globally subanalytic.
Lemma C.4 ( (Dries & Miller, 1996)). The image or the preimage of a globally subanalytic set by a globally subanalytic function (respectively, globally subanalytic multivalued operator) is globally subanalytic.
Lemma C.5 (Projection theorem (Dries & Miller, 1996)). Let Π(x 1 , . . . , x n+1 ) = (x 1 , . . . , x n ) be the canonical projection from R n+1 onto R n . If S is a globally subanalytic subset of R n+1 , then so is Π(S) in R n .
Lemma C.6 (Lojasiewicz factorization lemma (Bierstone & Milman, 1988, Theorem 6.4)). If X * v l,τ := arg min x∈X v l,τ (x) is globally subanalytic, and v l,τ (x) is continuous and globally subanalytic, then the (ϱ, η)-Hölderian error bound holds for v l,τ (x) for some ϱ, η > 0.
C.1. Proof of global subanalyticity of v l,τ , p, and u l Lemma C.7. Let X and Y be two bounded subanalytic subsets in R q . Let f : X × Y → R be a bounded subanalytic function. Then 1) the function ϕ below is bounded and subanalytic, thus globally subanalytic.
ϕ : X ∋ x → max y∈Y f (x, y) ∈ R.(70)
2) Given x, the solution set Y * (x) := arg max y∈Y f (x, y) is bounded and subanalytic, thus globally subanalytic.
Proof. The proof mainly adopts the proof idea of (Kosiba, 2025, Lemma 4.18). The key difference is that instead of using local boundedness of the functions, we start from global boundedness of f on X and derive the global boundedness of ϕ, which, combined with subanalyticity, leads to the global subanalyticity of ϕ on X .
Let Gr(•) denote the graph of a function. Since f is bounded, let c f , c f denote its upper and lower bound, respectively. And
define W := {w ∈ R | c f ≤ w ≤ c f }. Then W ⊂ R is bounded. Consider the following set A = {(x, y, z, w) ∈ X × Y × R × W | (x, y, z) ∈ Gr(f (x, y)), z ≤ w}.(71)
Note that x, y are bounded, and so is z because of the boundedness of f . Furthermore, w is bounded by definition. Thus A is bounded. Also note that A is subanalytic because f is subanalytic, its graph is subanalytic, and the Cartesian product of two subanalytic sets is subanalytic. Therefore, A is globally subanalytic.
Define the following projection Π A and the set B by canonical projection of A.
Π A : X × Y × R × W ∋ (x, y, z, w) → (x, y, w) ∈ X × Y × W.(72)
B := Π A (A) = {(x, y, w) ∈ X × Y × W | f (x, y) ≤ w}. (73
)
Then B is globally subanalytic based on Lemma C.5.
We also define the following auxiliary sets
B R = {(x, y, w) ∈ X × Y × W}, (74
) B R \B = {(x, y, w) ∈ X × Y × W | f (x, y) > w}.(75)
Then we define the projection Π B and define the following set C by canonical projection of B.
Π B : X × Y × W ∋ (x, y, w) → (x, w) ∈ X × W,(76)
C := Π B (B)\Π B (B R \B) = {(x, w) ∈ X × W | sup y∈Y f (x, y) ≤ w}. (77
)
Since C is bounded and subanalytic, it is globally subanalytic.
We further define the following subanalytic set and projection
D := {(x, w 1 , w 2 ) ∈ X × W × W | (x, w 1 ) ∈ C, (x, w 2 ) ∈ C, w 1 > w 2 },(78)
Π D : X × W × W ∋ (x, w 1 , w 2 ) → (x, w 1 ) ∈ X × W.(79)
Finally, observe that C\Π D (D) ∋ (x, w) ⇐⇒ there exists no w ′ such that (x, w ′ ), (x, w) ∈ C and w > w ′ (80) ⇐⇒ there exists no w ′ such that sup y∈Y f (x, y) ≤ w ′ and sup y∈Y f (x, y) ≤ w and w > w ′ (81) which means that Gr(ϕ(x)) = C\Π D (D). Since Gr(ϕ(x)) is bounded and subanalytic, thus it is globally subanalytic. By definition, ϕ is also globally subanalytic.
Next we proceed to prove the global subanalyticity of Y * (x) given x. Define the following set E and projection
Π E E := {(x, y, w) ∈ X × Y × W | (x, w) ∈ Gr(ϕ(x)), f (x, y) = w}, (82) Π E : X × Y × W ∋ (x, y, w) → (x, y) ∈ X × Y. (83
)
Then Π E (E) = {(x, y) ∈ X × Y | y ∈ Y * (x)
} is globally subanalytic. Furthermore, given x = c x ∈ X for any c x ∈ X , we have
Π E (E) ∩ {(x, y) ∈ X × Y | x = c x } = {(x, y) ∈ X × Y | y ∈ Y * (x), x = c x } (84
)
which is globally subanalytic. Taking projection Π : X × Y ∋ (x, y) → y ∈ Y yields that Y * (x) given x = c x for any c x ∈ X is also globally subanalytic.
Proof of Lemma 3.5. Part 1: By Assumption 3, f m is globally subanalytic for all m = 0, . . . , M . By Definition C.1 and Lemma C.4, subanalyticity is preserved under the subanalytic LSE function.
Recall the definition of h l,τ (x, y) in ( 25). By Assumption 1 and Lemma B.6, h l,τ (x, y) is Lipschitz continuous w.r.t. both x, y ∈ X C , thus h l,τ (x, y) is bounded for (x, y) ∈ X C × X C . Combining the above arguments, h l,τ (x, y) is bounded and subanalytic on X C , thus globally subanalytic on X C by Proposition C.3.
Part 2: Next, we prove that v l,τ (x) = -min y∈X h l,τ (x, y) and p(x) = v l,τ (x) + τ ln M θ are both globally subanalytic on X C . This directly follows by applying Lemma C.7-1).
Part 3: Finally, we prove that X * v l,τ = arg min x∈X v l,τ (x) is also globally subanalytic on X C . This directly follows by applying Lemma C.7-2).
Consequently, the merit function v l,τ (x) satisfies the (ϱ, η)-Hölderian error bound for some ϱ, η > 0 on a bounded set X C by Lemma C.6. The penalty function p(x) also satisfies the (ϱ p , η p )-Hölderian error bound with ϱ p = ϱ θ and η p = θη.
Following similar arguments as the above proof, we can obtain that the original merit function u l (x) is globally subanalytic on a bounded subanalytic set, and thus satisfies the HEB on the set. This result is formally stated below.
Corollary C.8. Under Assumption 3, and that X is subanalytic, given a compact subanalytic set X C , suppose f m (x) is continuous and bounded on X C ∩ X for all m ∈ [M ]. Then both X * v l,τ ∩ X C and v l,τ (x) on X C ∩ X are globally subanalytic. Consequently, the merit function u l (x) in (23) without smoothing satisfies the (ϱ u , η u )-HEB in Definition 3.4 on X C ∩ X , with some ϱ u , η u > 0.
Lemma C.9 (KL inequality (Kurdyka, 1998, Theorem 1)). Let f : Ω → R be a subanalytic function which is differentiable in Ω\f -1 (0), where Ω is an open bounded subset of R q . Then there exist c > 0, ν > 0 and α > 1 such that: c∥∇f (x)∥ α ≥ |f (x)|, for each x ∈ Ω such that |f (x)| ∈ (0, ν). If in addition lim x→a f (x) = 0 for some a ∈ Ω, then the above inequality holds for each x ∈ Ω\f -1 (0) close to a.
this section cite: ['b4', 'b7', 'b24', 'b24', 'b4', 'b47']

Section: C.2. Examples of globally subanalytic functions
We summarize some commonly used globally subanalytic functions and their corresponding Hölderian error bound (HEB) in Table 6 below. The first few examples of convex functions in Table 6 are directly referenced from (Doron & Shtern, 2023, Table 2) and (Chen et al., 2024b, Table 2).
Below, we prove the HEB of the last four bounded functions in Table 6.
{⟨ai, x⟩ -bi} ai ∈ R q , b ∈ R m Polytope 1 max i∈[m] {∥ai∥ -1 } ∥x -x0∥Q = (x -x0) ⊤ Q(x -x0) Q ∈ S q , Q ≻ 0, x0 ∈ R q Q-norm (Ellipsoid) 1 (λmin(Q)) -1 2 ∥x -x0∥p x0 ∈ R q , p ≥ 1 ℓp-norm 1 1 1 m m i=1 log(1 + exp(-a ⊤ i xbi)) ai ∈ R q , b ∈ R m , A ∈ R m×q Logistic loss 2 (λmin(Q)) -1 , Q is a function of ai, b f (x) + σ 2 ∥x∥ 2 f convex, σ > 0 strongly-convex 2 2 σ
Bounded and possibly nonconvex functions sin(x)
x ∈ [0,
1 2 π] Sine ≥ 1 ( π 2 ) η x p x ∈ [0, 1], p > 0 Polynomial p 1 e x x ∈ [0, 1] Exponential ≥ 1 1 ln(x + 1) x ∈ [0, 1] Logarithmic ≥ 1 1 ln(2)
Proof of HEB of bounded functions in Table 6. We prove the (ϱ, η)-Hölderian error bound (HEB) of the last four functions in Table 6 one by one as follows.
1) Sine function. For x ∈ [0, 1 2 π], x * = 0 is the unique minimizer of sin(x), thus for η ≥ 1, dist(x, X * sin ) η = |x| η ≤ ( π 2 ) η sin(x) for all x ∈ [0, 1 2 π]. Therefore, sin(x) satisfies (ϱ, η)-HEB for x ∈ [0, 1 2 π] with η ≥ 1 and ϱ = ( π 2 ) η . 2) Polynomial function. For x ∈ [0, 1], x * = 0 is the unique minimizer of x p . Thus dist(x, X * sin ) p = |x| p ≤ x p for all
x ∈ [0, 1]. Therefore, x p satisfies (ϱ, η)-HEB for x ∈ [0, 1] with η = p and ϱ = 1.
3) Exponential function. For x ∈ [0, 1], x * = 0 is the unique minimizer of e x . Thus dist(x, X * sin ) η = |x| η ≤ e x -e 0 for
all x ∈ [0, 1], and η ≥ 1. Therefore, e x satisfies (ϱ, η)-HEB for x ∈ [0, 1] with η ≥ 1 and ϱ = 1. 2) ln(x + 1) for all x ∈ [0, 1], and η ≥ 1. Therefore, e x satisfies (ϱ, η)-HEB for x ∈ [0, 1] with η ≥ 1 and ϱ = 1 ln(2) . The proof is complete.
4) Logarithmic function. For x ∈ [0, 1], x * = 0 is the unique minimizer of ln(x + 1). Thus dist(x, X * sin ) η = |x| η ≤ 1 ln(
Remark C.10. Note that, even though we have shown in the above that the exponential function satisfies HEB in a compact and subanalytic set x ∈ [0, 1], it is known that the exponential function is not globally subanalytic on R. In the following proofs which require the HEB property, it suffices to show the global subanalyticity holds on a compact and subanalytic set constructed from the corresponding problem, which leads to the HEB property on the compact and subanalytic set. And HEB of p(x) on a compact and subanalytic set within X is sufficient to show the relations of global/local/stationary solutions of the penalty problem to the bilevel problem as long as there exists bounded solutions to min x∈X p(x), even if X is not bounded.
this section cite: []

Section: C.3. Relations of Hölderian error bound, quadratic growth, proximal PL, and proximal error bound
We further discuss the relations of Hölderian error bound (HEB), also known as Hölderian growth, with other commonly used conditions such as quadratic growth (QG), proximal error bound (EB), proximal PL inequality, and strong convexity (SC). Below, we consider a general function for the discussion.
ϕ(x) := f (x) + g(x) (85
)
with f smooth and g convex. In our specific problem, g is an indicator function on X . Let X * ϕ denote the set of minimizers for ϕ. We first review the formal definitions of the above conditions, and then discuss their relations.
Definition C.11 (Proximal error bound). The function ϕ in (85) satisfies proximal error bound if for t > 0, there exists
c ϕ > 0 that dist(x, X * ϕ ) ≤ c ϕ t -1 ∥x -prox tg (x -t∇f (x))∥.(86)
Definition C.12 (Proximal PL). The function ϕ in (85) satisfies proximal PL inequality if for t > 0, there exists c > 0 that
ϕ(x) -ϕ(x * ) ≤ cD g (x, t) with x * ∈ X * ϕ ,and
D g (x, t) := -2t min y ⟨∇f (x), y -x⟩ + t 2 ∥y -x∥ 2 + g(y) -g(x) .(87)
By definition, QG is a special case of HEB with the exponent η = 2. And similarly, proximal KL generalizes the concept of proximal PL. The relations of SC, proximal EB, proximal PL, QG have been studied in existing literature. We summarize these relations using the equation below.
f is SC (a) =⇒ proximal EB (b) ⇐⇒ proximal PL (c) =⇒ QG ⇓ ⇓ proximal EB ⇐⇒ proximal KL =⇒ HEB (88
)
where (a) has been proved in e.g., (Karimi et al., 2016, Appendix F-2); (b) has been proved in e.g., (Karimi et al., 2016, Appendix G); (c) has been proved in e.g., (Liao et al., 2024, Theorem 3.1) with additional conditions that ϕ is closed and weakly convex, or in (Karimi et al., 2016, Theorem 2) with additional conditions that g is a constant.
Furthermore, when the function f is convex, we have that
proximal EB/PL (d) ⇐⇒ QG ⇓ ⇓ proximal EB/KL (f ) ⇐⇒ HEB (89
)
where (d) has been proved in e.g., (Drusvyatskiy & Lewis, 2018, Corollary 3.6); (f ) has been proved in e.g., (Bolte et al., 2016, Theorem 5). In this paper, we provide a proof of the relations between KL, HEB, and EB in (88) for nonconvex f and constant g in Lemma C.13 and Lemma C.14 below.
Lemma C.13 (KL implies HEB). Consider the function ϕ in (85) with g(x) = 0, and min ϕ(x
) = 0. If ϕ satisfies the (c ϕ , α ϕ )-KL inequality on Ω with exponent α ϕ > 1, then it also satisfies the (ϱ ϕ , η ϕ )-HEB on Ω with exponent η ϕ = α ϕ α ϕ -1 , and ϱ ϕ = 1 -1 α ϕ - α ϕ α ϕ -1 1 c ϕ -1 α ϕ -1 .
Proof of Lemma C.13. We first define an auxiliary function
L(x) := ϕ(x) 1-1 α ϕ .(90)
Since ϕ satisfies the KL inequality, then for any x / ∈ X * ϕ , and thus ϕ(x) ̸ = 0, we have
∥∇L(x)∥ 2 = 1 - 1 α ϕ 2 ∇ϕ(x) (ϕ(x)) 1 α ϕ 2 = 1 - 1 α ϕ 2 ∥∇ϕ(x)∥ 2 (ϕ(x)) 2 α ϕ ≥ 1 - 1 α ϕ 2 1 c ϕ 2 α ϕ µ L .(91)
Also ϕ satisfies the KL inequality implies that ϕ is an invex function and thus L is a non-negative invex function with a closed optimal solution set and zero optimal value. or any point x 0 / ∈ X * L , consider solving the following differential equation for x(t) / ∈ X * ϕ :
dx(t) dt = -∇L(x(t))(92)
x(t = 0) = x 0 (93
)
Following similar arguments for proving PL implies QG, see e.g., (Karimi et al., 2016, Theorem 2, Appendix A), there exists a T such that x(T ) ∈ X * L (and at this point the differential equation ceases to be defined). Then
L(x 0 ) -L(x t ) = x0 xt ⟨∇L(x), dx⟩ = - xt x0 ⟨∇L(x), dx⟩ = - T 0 ⟨∇L(x(t)), dx(t) dt ⟩dt (94
) = T 0 ∥∇L(x(t))∥ 2 dt ≥ T 0 µ L dt = µ L T.(95)
As L(x t ) ≥ 0, this shows we need to have T ≤ L(x 0 )/µ L , so there must be a T with x(T ) ∈ X * L . The length of the orbit x(t) starting at x 0 , which we'll denote by D(x 0 ), is given by
D(x 0 ) = T 0 ∥dx(t)/dt∥dt = T 0 ∥∇L(x(t))∥dt ≥ ∥x 0 -x p ∥(96)
where x p is the projection of x 0 onto X * L and the inequality follows because the orbit is a path from x 0 to a point in X * L (and thus it must be at least as long as the projection distance).
Then we can further bound
L(x 0 ) = L(x 0 ) -L(x T ) (95) = T 0 ∥∇L(x(t))∥ 2 dt (91) ≥ √ µ L T 0 ∥∇L(x(t))∥dt (96
) ≥ √ µ L ∥x 0 -x p ∥. (97
)
The proof is complete.
Lemma C.14 (KL and HEB imply EB). For ϕ(x) in (85) with g(x) = 0 that satisfies the KL inequality with exponent α ϕ , and the HEB with exponent η ϕ , i.e.,
KL: c ϕ ∥∇ϕ(x)∥ α ϕ ≥ ϕ(x), HEB: ϕ(x) ≥ ϱ -1 ϕ dist(x, X * ϕ ) η ϕ(98)
Then it holds that
∥∇ϕ(x)∥ ≥ ϱ -1 h dist(x, X * ϕ ) η h with η h = η ϕ α ϕ , and ϱ h = ϱ ϕ c ϕ 1 α ϕ .(99)
Proof of Lemma C.14. The proof directly follows from combining the two inequalities from KL and HEB that
c ϕ ∥∇ϕ(x)∥ α ϕ ≥ ϕ(x) ≥ ϱ -1 ϕ dist(x, X * ϕ ) η ϕ .(100)
Rearranging the above inequalities proves the result.
this section cite: ['b44', 'b50', 'b44', 'b25', 'b44']

Section: D. Proof of relations of different formulations
In this section, we prove the relations of the solutions of the bilevel problem and the penalty reformulation. Recall that we denote the penalty function with exponent θ as
p(x) = v l,τ (x) + τ ln M θ .(101)
Recall that we let
X * v l,τ := arg min x∈X v l,τ (x). Similarly, we define X * φγ := arg min x∈X φ γ (x). For ϵ ≥ 0, define the ϵ-approximate solution set (level set) below X ϵ := {x ∈ X | p(x) ≤ ϵ}.(102)
Then X * v l,τ = X 0 ⊆ X ϵ .
We use X C to denote a compact subanalytic set, X S ⊆ X to denote a closed subanalytic subset of X , and define z ϵ (x) ∈ arg min z∈Xϵ∩X C ∥z -x∥.
this section cite: []

Section: D.1. Proof of Theorem 3.7-1: the ϵ-global/local solutions relation of the smoothed problem
Proof of Theorem 3.7-1. We use u l as an auxiliary merit function, defined below
u l (x) := sup y∈X min m∈[M ] {f m (x) -f m (y) - l 2 ∥y -x∥ 2 }.(103)
Under the conditions in Proposition 3.3-2, solving (OPS) is equivalent to solving
min x∈X f 0 (x), s.t. x ∈ X * u l := {x ∈ X | u l (x) ≤ 0}.(104)
Part 1: proof of global solutions relation. Let x δ be an (ϵ, δ)-global solution to (CP), then
x δ ∈ X δ , i.e., v l,τ (x δ ) + τ ln M ≤ δ. From Proposition 3.3-2, we have that u l (x δ ) ≤ v l,τ (x δ ) + τ ln M ≤ δ.
Let x * be a global solution to (OPS), and let
x p := Proj X * u l (x δ ). By the (ϱ u , η u )-HEB of the function u l in Corollary C.8, see Appendix C.1, we have ∥x δ -x p ∥ = dist(x δ , X * u l ) ≤ (ϱ u δ) 1 ηu . For δ ≥ τ ln M , X * u l ⊆ X * δ , thus f 0 (x * ) -f 0 (x δ ) ≥ 0. By the ℓ f -local Lipschitz continuity of f 0 , it holds that f 0 (x * ) -f 0 (x δ ) ≤ f 0 (x p ) -f 0 (x δ ) ≤ ℓ f ∥x δ -x p ∥ ≤ ℓ f (ϱ u δ) 1 ηu .(105)
This proves that x δ is an (ϵ ′ , δ)-global solution to (OPS) with
ϵ ′ = ℓ f (ϱ u δ) 1 ηu .
Conversely, if x ′ is an (ϵ, δ)-global optimal solution to (OPS), then by definition,
f 0 (x ′ ) -min x∈X u l ,δ f 0 (x) ≤ ϵ, and x ′ ∈ X u l ,δ .(106)
In other words,
u l (x ′ ) ≤ δ, which implies v l,τ (x ′ )+τ ln M ≤ δ +τ ln M , i.e., x ′ ∈ X δ ′ with δ ′ = δ +τ ln M . Furthermore, since X u l ,δ ⊆ X δ ′ , min x∈X δ ′ f 0 (x) ≤ min x∈X u l ,δ f 0 (x), therefore f 0 (x ′ ) -min x∈X δ ′ f 0 (x) =f 0 (x ′ ) -min x∈X u l ,δ f 0 (x) + min x∈X u l ,δ f 0 (x) -min x∈X δ ′ f 0 (x) ≤ϵ + min x∈X u l ,δ f 0 (x) -min x∈X δ ′ f 0 (x)(107)
where letting
x * δ ′ ,0 ∈ arg min x∈X δ ′ f 0 (x), and x * p,δ ′ ,0 = Proj X u l ,δ (x), then min x∈X u l ,δ f 0 (x) -min x∈X δ ′ f 0 (x) can be further bounded by min x∈X u l ,δ f 0 (x) -min x∈X δ ′ f 0 (x) = min x∈X u l ,δ f 0 (x) -f 0 (x * δ ′ ,0 ) ≤f 0 (x * p,δ ′ ,0 ) -f 0 (x * δ ′ ,0 ) ≤ ℓ f ∥x * δ ′ ,0 -x * p,δ ′ ,0 ∥ ≤ ℓ f dist(x * δ ′ ,0 , X * u l ) ≤ ℓ f (ϱ u δ ′ ) 1 ηu . (108
) Therefore, x ′ is an (ϵ ′ , δ ′ ) solution to (CP) with ϵ ′ = ϵ + ℓ f (ϱ u δ ′ ) 1 ηu and δ ′ = δ + τ ln M .
Part 2: proof of local solutions relation. Let x δ be an (ϵ, δ)-local solution to (CP), then x δ ∈ X δ , and x δ is an (ϵ, δ)-global solution to (CP) on the set X ∩ B(x δ , r). Applying the results in Part 1 we get that x δ is an (ϵ ′ , δ)-global solution to (OPS) on the set X ∩ B(x δ , r), thus an (ϵ ′ , δ)-local solution to (OPS) on the set X . Similar arguments can be used to prove the converse statement.
Combining Part 1 and Part 2 completes the proof.
this section cite: []

Section: D.2. Proof of Theorem 3.7-2: the ϵ-global/local solutions relation
We first present some auxiliary lemmas below, then prove the main results.
Lemma D.1. Let X C ⊆ R q be a compact subanalytic set with X C ∩ X 0 ̸ = ∅, and thus X C ∩ X ϵ ̸ = ∅ for some ϵ ≥ 0. Suppose f 0 (x) is ℓ f -Lipschitz continuous on X C ∩ X with some ℓ f > 0. If v l,τ is globally subanalytic on X C ∩ X , then it satisfies the (ϱ, η)-HEB on X C ∩ X with some ϱ, η > 0, and p(x) satisfies the (ϱ p , η p )-HEB on X C ∩ X with ϱ p = ϱ θ and η p = θη. Given x ∈ X C ∩ X , it holds for any ϵ ≥ 0 that f 0 (x) + γp(x) -f 0 (z ϵ (x)) ≥ -ϵ γ :=    ℓ f ℓ f ϱp γηp 1 ηp -1 ( 1 ηp -1), η p > 1, γ > 0; 0, η p = 1, γ ≥ ϱ p ℓ f .(109)
Proof of Lemma D.1. Since v l,τ (x) and p(x) are globally subanalytic on X C ∩ X , there exists ϱ, η > 0 that v l,τ (x) satisfies the (ϱ, η)-HEB, thus p(x) satisfies the (ϱ p , η p )-HEB with ϱ p = ϱ θ and η p = θη on X C ∩ X , which yields that for all ϵ ≥ 0,
ϱ p p(x) ≥ ∥z 0 (x) -x∥ ηp ≥ ∥z ϵ (x) -x∥ ηp .(110)
Since X C is bounded, we have that for all x, x ′ ∈ X C , there exists
ℓ f > 0 such that f 0 (x) -f 0 (x ′ ) ≥ -ℓ f ∥x -x ′ ∥.
Combined with the above inequality, it holds that
f 0 (x) + γp(x) -f 0 (z ϵ (x)) ≥ γ ϱ p ∥z ϵ (x) -x∥ ηp -ℓ f ∥z ϵ (x) -x∥ ≥ inf ζ∈R+ γ ϱ p ζ ηp -ℓ f ζ -ϵγ .(111)
Analyzing -ϵ γ separately under η p > 1 and η p = 1 proves the result. When η p > 1, the result is obtained by solving the optimal ζ through the first-order optimality condition. When η p = 1, the optimal value is achieved at ζ = 0.
Lemma D.2. Given x γ ∈ X S ⊆ X with p(x γ ) = ϵ γ , and that x γ is an ϵ-global solution to (PP γ ) on X S , then f 0 (x γ ) - inf x∈X S ∩Xϵ γ f 0 (x) ≤ ϵ (112
) Proof of Lemma D.2. Since x γ is an ϵ-global solution of (PP γ ) on X S , by definition we have that for all x ∈ X S ∩ X ϵγ , f 0 (x γ ) + γp(x γ ) ≤ f 0 (x) + γp(x) + ϵ.(113)
Recall p(x γ ) = ϵ γ , and p(x) ≤ ϵ γ since x ∈ X S ∩ X ϵγ . Plugging these into the above inequality yields (112).
Lemma D.3. Let X C ⊆ R q be a compact subanalytic set with X C ∩ X 0 ̸ = ∅, and thus X C ∩ X ϵ ̸ = ∅ for some ϵ ≥ 0. Let B(x, r) denote the neighborhood of x with radius r > 0 for some x ∈ X C ∩ X . If there exists x ∈ B(x, r) ∩ X ∩ X C such that p(x) ≤ ϵ, then there exists z ϵ (x) ∈ arg min z∈Xϵ∩X C ∥z -x∥ such that z ϵ (x) ∈ B(x, r) ∩ X ϵ ∩ X C .
Proof of Lemma D.3. As X ϵ ∩ X C for ϵ ≥ 0 is closed and nonempty, there exists z ϵ (x) ∈ arg min x∈Xϵ∩X C ∥z -x∥. Also since x ∈ B(x, r) ∩ X , and p(x) ≤ ϵ, thus x ∈ B(x γ , r) ∩ X ϵ ∩ X C , then
∥z ϵ (x) -x∥ = min z∈Xϵ∩X C ∥z -x∥ ≤ ∥x -x∥ ≤ r (114
)
which implies that z ϵ (x) ∈ B(x, r), combined with z ϵ (x) ∈ X ϵ ∩ X C proves the result.
Lemma D.4. Let X S = X or X S = X ∩ B(x, r). Given x ϵ b ∈ X S ⊆ X , which is also an (ϵ b , ϵ)-global solution to (CP) on X S . Suppose X S ∩ X ϵ ′ ̸ = ∅
with ϵ ≥ ϵ ′ ≥ 0, and there exist bounded points in X S ∩ X ϵ ′ , then for ϵ γ specified in (109) in Lemma D.1, we have
f 0 (x ϵ b ) + γp(x ϵ b ) ≤ inf x∈X S f 0 (x) + γp(x) + ϵ γ + ϵ b + γϵ.(115)
Proof of Lemma D.4. By the definition of (ϵ b , ϵ)-global solution to (CP) on X S , we have
p(x ϵ b ) ≤ ϵ, and f 0 (x ϵ b ) ≤ inf x∈Xϵ∩X S f 0 (x) + ϵ b .(116)
Therefore,
f 0 (x ϵ b ) + γp(x ϵ b ) ≤ inf x∈Xϵ∩X S f 0 (x) + ϵ b + γϵ. (117
) Recall that X S ∩ X ϵ ′ ̸ = ∅. Let X C be a compact subanalytic set such that X C ∩ X S ∩ X ϵ ′ ̸ = ∅, and X C ∩ X * S,φγ ̸ = ∅. The set X C exists because there exist bounded points in X S ∩ X ϵ ′ and X * S,φγ , respectively. For x ∈ X C ∩ X , recall we define z ϵ (x) ∈ arg min z∈Xϵ∩X C ∥z -x∥. When X S = X , z ϵ (x) ∈ X S ; when X S = X ∩ B(x, r), z ϵ (x) ∈ X S by Lemma D.3. Let x * C ∈ arg min x∈X C ∩X S φ γ (x). Then inf x∈Xϵ∩X S f 0 (x) can be further bounded by inf x∈Xϵ∩X S f 0 (x) ≤ inf x∈X ϵ ′ ∩X C ∩X S f 0 (x) since X ϵ ′ ∩ X C ∩ X S ⊆ X ϵ ∩ X S ≤f 0 (z ϵ ′ (x * C )) since z ϵ ′ (x * C ) ∈ X ϵ ′ ∩ X C ∩ X S ≤f 0 (x * C ) + γp(x * C ) + ϵ γ from Lemma D.1 = inf x∈X S f 0 (x) + γp(x) + ϵ γ . since X C ∩ X * S,φγ ̸ = ∅ (118
)
Plugging ( 118) into (117) yields
f 0 (x ϵ b ) + γp(x ϵ b ) ≤ inf x∈X S f 0 (x) + γp(x) + ϵ γ + ϵ b + γϵ. (119
)
The proof is complete.
Proof of Theorem 3.7-2. 1) Given γ > 0, let x γ be a bounded ϵ-global solution of (PP γ ) on X S , with X S = X or X S = B(x γ , r) ∩ X , then for all x ∈ X S ,
f 0 (x γ ) + γp(x γ ) ≤ f 0 (x) + γp(x) + ϵ. (120) Let x * ∈ X S ∩ X ϵ * denote a bounded (0, ϵ * )-global solution of (CP) on X S .
Then there exists a compact and subanalytic set X C such that x γ , x * ∈ X C and
X C ∩ X ϵ * ̸ = ∅. Let z ϵ * (x) ∈ arg min z∈X ϵ * ∩X C ∥z -x∥. When 1) X S = X , z ϵ * (x γ ) ∈ X S ∩ X ϵ * ; when 2) X S = B(x γ , r), z ϵ * (x γ ) ∈ X S ∩ X ϵ *
according to Lemma D.3. Then by Lemma D.1, given γ ′ > 0, in both cases we have that 120), and combining with the above inequality, we obtain that
f 0 (x γ ) + γ ′ p(x γ ) -f 0 (z ϵ * (x γ )) ≥ -ϵ γ ′ . (121
) Because f 0 (z ϵ * (x γ )) ≥ f 0 (x * ), (121) indicates that f 0 (x γ ) + γ ′ p(x γ ) ≥ f 0 (x * ) -ϵ γ ′ . Plugging x = x * in (
f 0 (x γ ) + γp(x γ ) ≤f 0 (x * ) + γϵ * + ϵ since p(x * ) ≤ ϵ * ≤f 0 (x γ ) + γ ′ p(x γ ) + ϵ γ ′ + γϵ * + ϵ from (121) (122
)
which further implies
(γ -γ ′ )p(x γ ) ≤ ϵ γ ′ + γϵ * + ϵ. (123
) Define ϵ γ := p(x γ ), then x γ ∈ X ϵγ := {x ∈ X | p(x ϵγ ) ≤ ϵ γ }. The above inequality implies ϵ γ ≤ ϵ γ ′ +γϵ * +ϵ γ-γ ′
. Further, from Lemma D.2 we have
f 0 (x γ ) - inf x∈X S ∩Xϵ γ f 0 (x) ≤ ϵ.(124)
Combining ( 123) and (124) proves that x γ is an (ϵ, ϵ γ )-approximate global solution to (CP) on X S with ϵ γ ≤
ϵ γ ′ +γϵ * +ϵ γ-γ ′ . When η p > 1, choosing γ ′ = γ 2 , γ ≥ 2ℓ f δ - ηp -1 ηp ( ϱp ηp ) 1 ηp (1 -1 ηp ) ηp -1 ηp , and ϵ ≤ ℓ f δ 1 ηp ( ϱp 2ηp ) 1 ηp (1 -1 ηp ) ηp -1 ηp , we further have that ϵ γ ≤ ϵ γ ′ + ϵ + γϵ * γ -γ ′ = 2ℓ f γ ηp ηp -1 ϱ p η p 1 ηp -1 (1 - 1 η p ) + 2ϵ γ + 2ϵ * ≤ δ + 2ϵ * (125
) which proves that the ϵ-global solution of (PP γ ) on X S with γ ≥ 2ℓ f δ - ηp -1 ηp ( ϱp ηp ) 1 ηp (1-1 ηp ) ηp -1 ηp and ϵ ≤ ℓ f δ 1 ηp ( ϱp 2ηp ) 1 ηp (1- 1 ηp ) ηp -1 ηp , is an (ϵ, δ + 2ϵ * )-global solution to (CP) on X S . When η p = 1, from Lemma D.1, for γ ′ ≥ ϱ p ℓ f , ϵ γ ′ = 0. Choosing γ ≥ ϱ p ℓ f + 1, we have ϵ γ ≤ ϵ. Therefore, the ϵ-global solution of (PP γ ) on X S is an (ϵ, ϵ + 2ϵ * )-global solution to (CP) on X S . The set X S can be X or B(x γ , r) ∩ X , which
corresponds to the global solution on X , or the local solution in the neighborhood of x γ , respectively.
2) Next we prove the converse. Define x ϵ b to be a bounded (ϵ b , ϵ)-global solution to (CP) on X S , and recall X ϵ := {x ∈ X | p(x) ≤ ϵ}. Then by Lemma D.4, we have
f 0 (x ϵ b ) + γp(x ϵ b ) ≤ inf x∈X S f 0 (x) + γp(x) + ϵ γ + ϵ b + γϵ ≤ inf x∈X S f 0 (x) + γp(x) + δ (126
)
where the last inequality holds by choosing ϵ ≤ δ 3γ , ϵ b ≤ δ 3 , and γ = ℓ f ϱp ηp
3ℓ f (1-1 ηp ) δ ηp-1 when η p > 1, γ = ϱ p ℓ f when η p = 1.
The set X S can be X or B(x ϵ b , r) ∩ X , which corresponds to the global solution on X , or the local solution in the neighborhood of x ϵ b , respectively. The converse is proved. Lemma D.5. If x γ ∈ X is an ϵ-stationary solution to (PP γ ), and ∥∇f 0 (x γ )∥ ≤ ℓ f , then it is also an ϵ γ -stationary solution to min x∈X p(x) with ϵ γ = ϵ+ℓ f γ . Furthermore, for θ ≥ 1, let c v (x γ ) := θ(v l,τ (x γ ) + τ ln M ) θ-1 . Then either one of the following two conditions holds. 1) c v (x γ ) = 0 and x γ is an optimal solution, thus also a stationary solution to min x∈X v l,τ (x); 2) c v (x γ ) > 0, and x γ is an ϵ ′ γ -stationary solution to min x∈X v l,τ (x) with ϵ ′ γ = ϵ+ℓ f γcv(xγ ) .
Proof of Lemma D.5. Since x γ ∈ X is an ϵ-stationary solution to (PP γ ), for α > 0, we have
1 α ∥x γ -Proj X (x γ -α∇φ γ (x γ ))∥ ≤ ϵ.(127)
By the definition that φ γ (x) = f 0 (x) + γp(x), we further have that
∥x γ -Proj X (x γ -αγ∇p(x γ ))∥ ≤∥x γ -Proj X (x γ -α∇φ γ (x γ ))∥ + ∥Proj X (x γ -α∇φ γ (x γ )) -Proj X (x γ -αγ∇p(x γ ))∥ ≤∥x γ -Proj X (x γ -α∇φ γ (x γ ))∥ + ∥α∇φ γ (x γ ) -αγ∇p(x γ )∥ =∥x γ -Proj X (x γ -α∇φ γ (x γ ))∥ + α∥∇f 0 (x γ )∥.(128)
Dividing both sides by 1 αγ of the above inequality yields
1 αγ ∥x γ -Proj X (x γ -αγ∇p(x γ ))∥ ≤ ϵ + ℓ f γ (129
) which proves that x γ is an ϵ γ -stationary solution to min x∈X p(x) with ϵ γ = ϵ+ℓ f γ . By the definition that p(x γ ) = (v l,τ (x γ ) + τ ln M ) θ , for θ ≥ 1, we further have 1 αγ ∥x γ -Proj X (x γ -αγ θ(v l,τ (x γ ) + τ ln M ) θ-1 cv(xγ ) ∇v l,τ (x γ ))∥ ≤ ϵ + ℓ f γ (130
)
where c v (x γ ) ≥ 0. This implies that either v l,τ (x γ ) + τ ln M = 0, or c v (x γ ) > 0, and
1 αγc v (x γ ) ∥x γ -Proj X (x γ -αγc v (x γ )∇v l,τ (x γ ))∥ ≤ ϵ + ℓ f γc v (x γ ) . (131
)
The proof is complete.
Lemma D.6. Suppose Assumption 4 holds. Recall that y * l,τ (x) := arg min y∈X h l,τ (x, y). For l -ℓ f,1 > 0, we have
l -ℓ f,1 2 ∥x -y * l,τ (x)∥ 2 ≤ v l,τ (x) + τ ln M ≤ 3ℓ 2 h l,τ ,1 + 6α -2 2(l -ℓ f,1 ) ∥x -y * l,τ (x)∥ 2 . (132
)
Proof of Lemma D.6. By definition, v l,τ (x) = -h l,τ (x, y * l,τ (x)), and h l,τ (x, x) = τ ln M . Therefore,
v l,τ (x) + τ ln M = -h l,τ (x, y * l,τ (x)) + h l,τ (x, x) ≥ l -ℓ f,1 2 ∥x -y * l,τ (x)∥ 2 (133
)
where the last inequality follows from the (l -ℓ f,1 )-strong convexity of h l,τ (x, •), thus the quadratic growth. The first inequality in (132) is proved.
For the second inequality in (132), we have
v l,τ (x) + τ ln M = -h l,τ (x, y * l,τ (x)) + h l,τ (x, x) (a) ≤ 1 2(l -ℓ f,1 )α 2 ∥x -Proj X (x -α∇ y h l,τ (x, x))∥ 2 (b) ≤ 1 2(l -ℓ f,1 )α 2 ∥x -Proj X (x -α∇ y h l,τ (x, x)) -y * l,τ (x) + Proj X (y * l,τ (x) -α∇ y h l,τ (x, y * l,τ (x)))∥ 2
Applying Lemma D.6 and letting α = O(1), we have
v l,τ (x) + τ ln M ≤ 3ℓ 2 h l,τ ,1 + 6α -2 2(l -ℓ f,1 ) ∥x -y * l,τ (x)∥ 2 ≤ 3ℓ 2 h l,τ ,1 + 6α -2 2(l -ℓ f,1 )µ 2 ϵ 2 = O(ϵ 2 ).(141)
Then there exists x ∈ N (x γ , r) that p(x) = O(ϵ 2θ ).
this section cite: []

Section: D.3. Proof of Theorem 3.11: the ϵ-stationary solutions relation
We first discuss the stationary condition of (CP) when X = R q , and the calmness condition that ensures the KKT condition is a necessary condition. Then we prove Theorem 3.11, the relation of ϵ-stationary solutions to (PP γ ) and (CP). Consider a general constrained problem below
min x∈R q f 0 (x) s.t. H(x) = 0 (142
)
where f 0 : R q → R, and H : R q → R d h with d h ≥ 1.
Definition D.8 (KKT condition of ( 142)). The KKT condition of ( 142) is
H(x) = 0 feasibility , ∇f 0 (x) + ∇H(x)w = 0 stationarity , with w ∈ R d h .(143)
Correspondingly, the (ϵ ′ , ϵ)-KKT condition of ( 142) is
∥∇f 0 (x) + ∇H(x)w∥ ≤ ϵ ′ , ∥H(x)∥ ≤ ϵ, with w ∈ R d h .(144)
Definition D.9 (Calmness (Clarke, 1990, Definition 6.4.1)). Let x * be the global minimizer of ( 142). If there exist ϵ, c > 0 such that for any u ∈ R d h with ∥u∥ ≤ ϵ and any x that ∥x -x * ∥ ≤ ϵ which satisfies H(x) + u = 0, one has
f 0 (x) -f 0 (x * ) + c∥u∥ ≥ 0.(145)
Then the problem (142) is said to be calm with c.
Lemma D.10 ((Ye, 2000, Theorem 3.6)). If the problem (142) is calm at a global solution x * , then x * satisfies the KKT condition in Definition D.8.
Below is a lemma to show that if the objective is Lipschitz and the constraint satisfies error bound with exponent no greater than one, then the calmness condition holds. Similar results have been discussed in (Ye, 2000, Proposition 4.2) with exponent equal to one. This result connects error bound with the calmness condition, and thus the necessity of KKT condition.
Lemma D.11. Let x * be a global minimizer of problem (142). For ϵ < 1, consider any u ∈ R d h and ∥u∥ ≤ ϵ, and any x that ∥x -x * ∥ ≤ ϵ and H(x
) + u = 0. Define x p = Proj X * H (x), where X * H = {x ∈ R q | H(x) = 0}.
If H(x) satisfies an error bound that ∥H(x)∥ ≥ ϱ h ∥x -x p ∥ η h and f 0 is ℓ f -Lipschitz for all x ∈ B(x * , 2ϵ) with ϵ < 1, then the calmness condition in Definition D.9 for problem (142) holds.
Proof of Lemma D.11. By definition, for x ∈ B(x * , ϵ) with ϵ ≤ 1,
ϵ ≥ ∥u∥ = ∥H(x)∥ ≥ ϱ h ∥x -x p ∥ η h .(146)
Since x * is a global minimizer, and ∥u∥ ≤ ϵ ≤ 1, for η h ≤ 1,
f 0 (x) -f 0 (x * ) ≥ f 0 (x) -f 0 (x p ) (a) ≥ -ℓ f ∥x -x p ∥ ≥ - ℓ f ϱ h ∥u∥ 1 η h ≥ - ℓ f ϱ h ∥u∥.(147)
where (a) holds because of the ℓ f -Lipschitz continuity of f 0 on a bounded set that includes x and x p . Therefore, the calmness condition in Definition D.9 holds with c =
ℓ f ϱ h . Proposition D.12. Recall that p(x) = v l,τ (x) + τ ln M θ with θ > 0. If v l,τ satisfies the (c v , α v )-KL inequality on Ω
with α v > 1, then p satisfies the (c p , α p )-KL inequality on Ω with α p =
θ θ-1+ 1 αv > 1, and c p = θ - θ θ-1+ 1 αv • c θ (θ-1)αv +1 v .
Proof of Proposition D.12. For x ∈ Ω and that v l,τ (x) + τ ln M > 0, we have
∥∇p(x)∥ =θ(v l,τ (x) + τ ln M ) θ-1 ∥∇v l,τ (x)∥ ≥θ(c v ) -1 αv v l,τ (x) + τ ln M θ-1+ 1 αv = θ(c v ) -1 αv p(x) θ-1+ 1 αv θ .(148)
Rearranging the above inequality proves the result.
Theorem 3.11 requires the assumption of the ℓ v,2 -smoothness of ∇v l,τ on a bounded set. Below we provide a sufficient condition, which shows that under additional assumptions of f m , m ∈ [M ], the ℓ v,2 -smoothness of ∇v l,τ on a bounded set X C can be justified.
Lemma D.13 (Smoothness of ∇v l,τ ). Suppose Assumption 2 holds, and l + µ > 0. If ∇f m is ℓ f,2 -smooth for all m ∈ [M ] on a bounded set X C , and there exists
x ′ ∈ X C that ∇ 2 f m for all m ∈ [M ] is bounded, then ∇v l,τ is ℓ v,2 -smooth on X C , with ℓ v,2 = ℓ hxx,2 (1 + ℓ y * l,τ ) + ℓ y * l,τ ,1 ℓ hxy,1 + ℓ y * l,τ ℓ hxy,2 (1 + ℓ y * l,τ ).
Proof of Lemma D.13. With similar arguments as Lemma E.1, since ∇f m is ℓ f,2 -smooth for all m ∈ [M ] on a bounded set X C , and there exists x ′ ∈ X C that ∇ 2 f m (x ′ ) for all m ∈ [M ] is bounded, therefore, there exists ℓ f,1 > 0 such that for x ∈ X C , and for all m ∈ [M ], ∥∇ 2 f m (x)∥ ≤ ℓ f,1 .
Under Assumption 2, and with l + µ > 0, first recall from (5) that ∇v l,τ (x) can be computed by
∇v l,τ (x) = -∇ x h(x, y) | y=y * l,τ (x) = M m=1 π m (x, y)∇f m (x) -l(x -y) | y=y * l,τ (x) .(149)
Because of the twice continuous differentiability of f m for m ∈ [M ], ∇ 2 v l,τ (x) exists and can be computed by
∇ 2 v l,τ (x) = -∇ 2 xx h(x, y) -∇y * l,τ (x)∇ 2 xy h(x, y) | y=y * l,τ (x) .(150)
For simplicity, we simplify y * l,τ (x) as y * (x), and h l,τ (x, y) as h(x, y) in the following derivations. Then ∥∇ 2 v l,τ (x) -∇ 2 v l,τ (x ′ )∥ can be bounded by
∥∇ 2 v l,τ (x) -∇ 2 v l,τ (x ′ )∥ ≤ ∥∇ 2 xx h(x, y * (x)) -∇ 2 xx h(x ′ , y * (x ′ ))∥ J1 + ∥∇y * (x) -∇y * (x ′ )∥∥∇ 2 xy h(x, y * (x))∥ J2 + ∥∇y * (x)∥∥∇ 2 xy h(x, y * (x)) -∇ 2 xy h(x ′ , y * (x ′ ))∥ J3(151)
where J 1 can be further bounded by
J 1 ≤ ℓ hxx,2 ∥x -x ′ ∥ + ∥y * (x) -y * (x ′ )∥ ≤ ℓ hxx,2 (1 + ℓ y * l,τ )∥x -x ′ ∥ (152
) with ℓ hxx,2 denoting the Lipschitz continuity of ∇ 2 xx h(x, y) w.r.t. [x; y]. Similarly, with ℓ hxy,2 denoting the Lipschitz continuity of ∇ 2 xy h(x, y) w.r.t. [x; y], J 3 can be bounded by
J 3 ≤ ℓ y * l,τ ℓ hxy,2 (1 + ℓ y * l,τ )∥x -x ′ ∥.(153)
And J 2 can be bounded by
J 2 ≤ ℓ y * l,τ ,1 ℓ hxy,1 ∥x -x ′ ∥.(154)
Therefore,
∥∇ 2 v l,τ (x) -∇ 2 v l,τ (x ′ )∥ ≤ ℓ hxx,2 (1 + ℓ y * l,τ ) + ℓ y * l,τ ,1 ℓ hxy,1 + ℓ y * l,τ ℓ hxy,2 (1 + ℓ y * l,τ ) ℓv,2 ∥x -x ′ ∥.(155)
The derivation of ℓ y * l,τ is discussed in Lemma B.8. We next discuss the derivation for ℓ hxy,1 , ℓ hxx,2 , ℓ hxy,2 and ℓ y * l,τ ,1 .
To compute ℓ y * l,τ ,1 , from implicit differentiation, ∇y * l,τ (x) can be computed by
∇y * l,τ (x) = -∇ 2 xy h(x, y * (x)) ∇ 2 yy h(x, y * (x)) -1 .(156)
Then ∥∇y * l,τ (x) -∇y * l,τ (x ′ )∥ can be bounded by
∥∇y * l,τ (x) -∇y * l,τ (x ′ )∥ ≤ ∥∇ 2 xy h(x, y * (x)) -∇ 2 xy h(x ′ , y * (x ′ ))∥ ∇ 2 yy h(x, y * (x)) -1 + ∥∇ 2 xy h(x ′ , y * (x ′ ))∥∥[∇ 2 yy h(x, y * (x))] -1 ∥∥[∇ 2 yy h(x ′ , y * (x ′ ))] -1 ∥ ∇ 2 yy h(x, y * (x)) -∇ 2 yy h(x ′ , y * (x ′ )) ≤ ℓ hxy,2 (l + µ) -1 + ℓ hxy,1 (l + µ) -2 ℓ hyy,2 (1 + ℓ y * l,τ ) ℓ y * l,τ ,1 ∥x -x ′ ∥.(157)
We then proceed to bound ℓ hxy,1 .
∇ 2 xy h(x, y) = M m=1 ∇ y π m (x, y)∇f m (x) ⊤ + lI q = 1 τ ∇F (y) π(x, y)π(x, y) ⊤ -diag(π(x, y)) ∇F (x) ⊤ -lI q .(158)
We have that
∥∇ 2 xy h(x, y)∥ ≤ ℓ f + l := ℓ hxy,1 .(159)
Furthermore, to bound ℓ hxy,2 , we have
∥∇ 2 xy h(x, y) -∇ 2 xy h(x ′ , y ′ )∥ ≤ 1 τ ∥∇F (y) -∇F (y ′ )∥ π(x, y)π(x, y) ⊤ -diag(π(x, y)) ∇F (x) ⊤ + ∥∇F (y ′ )∥∥π(x, y) -π(x ′ , y ′ )∥(∥π(x, y)∥ + ∥π(x ′ , y ′ )∥ + 1)∥∇F (x)∥ + ∇F (y ′ ) π(x ′ , y ′ )π(x ′ , y ′ ) ⊤ -diag(π(x ′ , y ′ )) ∥∇F (x) -∇F (x ′ )∥ ≤ 1 τ 2ℓ f,1 + 3ℓ f ℓ π M ℓ f ℓ hxy ,2 ∥x -x ′ ∥ + ∥y -y ′ ∥ .(160)
To compute ℓ π , recall that
∇ x π(x, y) = - 1 τ ∇F (x) diag(π(x, y)) -π(x, y)π(x, y) ⊤ ,(161)
∇ y π(x, y) = 1 τ ∇F (y) diag(π(x, y)) -π(x, y)π(x, y) ⊤(162)
from which we have
max{∥∇ x π(x, y)∥, ∥∇ y π(x, y)∥} ≤ 2 τ √ M ℓ f := ℓ π .(163)
Next we bound ℓ hxx,2 , and ℓ hyy,2 . The Hessian of h(x, y) can be computed by
∇ 2 yy h(x, y) = - 1 τ ∇F (y) π(x, y)π(x, y) ⊤ -diag(π(x, y)) ∇F (y) ⊤ + M m=1 π m (x, y)∇ 2 f m (y) + lI q , (164
)
∇ 2 xx h(x, y) = - 1 τ ∇F (x) π(x, y)π(x, y) ⊤ -diag(π(x, y)) ∇F (x) ⊤ + M m=1 π m (x, y)∇ 2 f m (x) + lI q . (165
)
Then we have
∥∇ 2 yy h(x, y) -∇ 2 yy h(x ′ , y ′ )∥ ≤ 1 τ ∥∇F (y) -∇F (y ′ )∥ π(x, y)π(x, y) ⊤ -diag(π(x, y)) ∇F (y) ⊤ + ∥∇F (y ′ )∥∥π(x, y) -π(x ′ , y ′ )∥(∥π(x, y)∥ + ∥π(x ′ , y ′ )∥ + 1)∥∇F (y)∥ + ∇F (y ′ ) π(x ′ , y ′ )π(x ′ , y ′ ) ⊤ -diag(π(x ′ , y ′ )) ∥∇F (y) -∇F (y ′ )∥ + ∥∇ 2 F (y) -∇ 2 F (y ′ )∥∥π(x, y)∥ + ∥∇ 2 F (y ′ )∥∥π(x, y) -π(x ′ , y ′ )∥ ≤ 1 τ 2ℓ f,1 + 3ℓ f ℓ π M ℓ f + √ M ℓ f,2 + √ M ℓ f,1 ℓ π ℓ hyy ,2 ∥x -x ′ ∥ + ∥y -y ′ ∥ . (166)
With similar derivations as the above, we have ℓ hxx,2 = ℓ hyy,2 .
Collecting the results in ( 155), ( 157), ( 159), ( 160), ( 163), ( 166) completes the proof.
Proof of Theorem 3.11. Since x γ is an ϵ-stationary solution to (PP γ ), thus
∥∇f 0 (x γ ) + γ∇p(x γ )∥ ≤ ϵ.(167)
By Lemma D.5, it is also an ϵ γ -stationary solution to min x∈R q p(x), and thus
∥∇p(x γ )∥ ≤ ϵ γ = ϵ + ℓ f γ .(168)
By Lemma C.14, the KL condition implies that ϱ
h dist(x γ , X * p ∩ X C ) η h ≤ ∥∇p(x γ )∥.
And since X * p ∩ X C is closed, the above implies that there exists x * ∈ X * p ∩ X C such that
∥x γ -x * ∥ = dist(x γ , X * p ∩ X C ) ≤ ϱ -1 η h h ∥∇p(x γ )∥ 1 η h = (ϱ -1 h ϵ γ ) 1 η h = O(ϵ 1 η h γ ).(169)
Taking Taylor expansion of ∇p(x) at x * and by the ℓ p,2 -smoothness of ∇p(x) on X C , we have
∥∇p(x γ ) -∇ 2 p(x * )(x γ -x * )∥ =∥∇p(x γ ) -∇p(x * ) -∇ 2 p(x * )(x γ -x * )∥ ≤ℓ p,2 ∥x γ -x * ∥ 2 ≤ ℓ p,2 (ϱ -1 h ϵ γ ) 2 η h = O(ϵ 2 η h γ ).(170)
Plugging the above inequality into (167), we have
∥∇f 0 (x γ ) + γ∇ 2 p(x * )(x γ -x * )∥ ≤ ϵ + γℓ p,2 (ϱ -1 h ϵ γ ) 2 η h = O(ϵ + γ 1-2 η h ).(171)
Letting w = γ(x γ -x * ), then ∥w∥ = O(γ
1-1 η h ) ≤ O(1) is bounded since η h ≤ 1. We can further bound ∥∇f 0 (x γ ) + ∇ 2 p(x γ )w∥ by ∥∇f 0 (x γ ) + ∇ 2 p(x γ )w∥ ≤∥∇f 0 (x γ ) + ∇ 2 p(x * )w∥ + ∥∇ 2 p(x γ ) -∇ 2 p(x * )∥∥w∥ ≤∥∇f 0 (x γ ) + ∇ 2 p(x * )w∥ + ℓ p,2 γ∥x γ -x * ∥ 2 ≤ϵ + 2γℓ p,2 (ϱ -1 h ϵ γ ) 2 η h = O(ϵ + γ 1-2 η h ).(172)
Recall that η h ≤ 1, thus choosing γ = Ω(δ -1 ), and ϵ ≤ ℓ f , we have ϵ γ ≤ δ, and ∥∇f 0 (x γ ) + ∇ 2 p(x γ )w∥ ≤ ϵ + δ, which proves the result. In this case, since we require
1 ≥ η h = η p α p = 1 α p -1(173)
which implies α p ≥ 2, and thus θ θ-1+ 1 αv ≥ 2 from Proposition D.12, implying θ < 2. Remark D.14. Note that, the HEB and KL exponents may not be unique (c.f. Appendix C.2). In such cases, there may exist 0 < η p ̸ = αp αp-1 . And the above theorem still requires η h = ηp αp ≤ 1, thus α p ≥ η p . If we further have that p(x) is smooth around x * , then the smoothness implies η p ≥ 2 ≥ α p , combining which with α p ≥ η p , implies that we require α p = η p = 2. This condition can be relaxed, or the exponent can take a wider range if the local Lipschitz continuity of f 0 is replaced by the Hölder continuity with larger exponent as in (Ye et al., 1997, Definition 2.8). We leave a detailed discussion to future work.
ℓ f τ ∥x -x ′ ∥ + e fm(y)-fm (x ′ ) τ M m=1 e fm(y)-fm(x) τ M m=1 e fm(y)-fm (x) τ M m=1 e fm(y)-fm(x ′ ) τ ℓ f τ ∥x -x ′ ∥ (178
)
where x is on the line segment of x and x ′ . Taking M m=1 of the above inequality yields
M m=1 ∥π m (x, y) -π m (x ′ , y)∥ ≤ M m=1 e fm (y)-fm(x) τ M m=1 e fm (y)-fm(x) τ • ℓ f τ ∥x -x ′ ∥ + M m=1 e fm(y)-fm (x) τ M m=1 e fm(y)-fm (x) τ • ℓ f τ ∥x -x ′ ∥ ≤2e ℓ f ℓx τ ℓ f τ ∥x -x ′ ∥(179)
where the last inequality uses the fact that ∥f m (x) -f m (x)∥ ≤ ℓ f ℓ x . Combining the above arguments yields
∥∇ x h l,τ (x, y) -∇ x h l,τ (x ′ , y)∥ ≤ℓ h l,τ ,1 ∥x -x ′ ∥(180)
with ℓ h l,τ ,1 = l + ℓ f,1 + 2e ℓ f ℓx τ ℓ 2 f τ . Similarly, given x, y, y ′ ∈ X , we can bound ∥∇h l,τ (x, y) -∇h l,τ (x, y ′ )∥ by
∥∇ x h l,τ (x, y) -∇ x h l,τ (x, y ′ )∥ ≤ℓ h l,τ ,1 ∥y -y ′ ∥ (181) ∥∇ y h l,τ (x, y) -∇ y h l,τ (x, y ′ )∥ ≤ℓ h l,τ ,1 ∥y -y ′ ∥.(182)
The gradient of v l,τ (x) can be computed by
∇v l,τ (x) = M m=1 π m (x, y * l,τ (x))∇f m (x) + l(x -y * l,τ (x)).(183)
Given x, x ′ ∈ X , we can bound ∥∇v l,τ (x) -∇v l,τ (x ′ )∥ by
∥∇v l,τ (x) -∇v l,τ (x ′ )∥ ≤ ℓ h l,τ ,1 (∥x -x ′ ∥ + ∥y * l,τ (x) -y * l,τ (x ′ )∥) ≤ ℓ h l,τ ,1 (1 + ℓ y * l,τ )∥x -x ′ ∥. (184
)
The proof is complete.
Lemma E.3 (Smoothness of the penalized function). Suppose Assumptions 4, 5 hold. Then φ f h,γ is ℓ φ f h,γ ,1 -smooth w.r.t.
x and y on the trajectory, with ℓ φ f h,γ ,1 = ℓ f,1 + γℓ h l,τ ,1 .
Proof of Lemma E.3. Given x, x ′ , y, y ′ ∈ X , we can bound the difference ∥∇ x φ f h,γ (x, y) -∇ x φ f h,γ (x ′ , y)∥ by
∥∇ x φ f h,γ (x, y) -∇ x φ f h,γ (x ′ , y)∥ ≤∥∇f 0 (x) -∇f 0 (x ′ )∥ + γ∥∇h l,τ (x, y) -∇h l,τ (x ′ , y)∥ ≤ℓ f,1 ∥x -x ′ ∥ + γℓ h l,τ ,1 ∥x -x ′ ∥.(185)
Similarly, we can bound the difference ∥∇ y φ f h,γ (x, y) -∇ y φ f h,γ (x, y ′ )∥ by
∥∇ y φ f h,γ (x, y) -∇ y φ f h,γ (x, y ′ )∥ ≤γ∥∇h l,τ (x, y) -∇h l,τ (x, y ′ )∥ ≤ γℓ h l,τ ,1 ∥y -y ′ ∥. (186
)
The proof is complete.
Lemma E.4 (Contraction of y t,k ). Suppose Assumptions 4, 5 hold, and l -ℓ f,1 ≥ µ hy > 0. Recall that y * l,τ (x) := arg min y h l,τ (x, y). The sequence {y t,k } K k=1 produced by Algorithm 1 satisfies ∥y t,k+1 -y * l,τ (x t )∥ 2 ≤ (1 -µ hy β t,k )∥y t,k -y * l,τ (x t )∥ 2 .
Proof of Lemma E.4. Recall that the update of y t,k in (13a) takes the projected gradient descent (PGD) on h l,τ (x, y). By Corollary B.4, for l + min m∈[M ] µ m ≥ µ hy > 0, the function h l,τ (x, y) is µ hy -strongly convex w.r.t. y.
Leveraging the convergence result of PGD on strongly convex functions, we have
∥y t,k+1 -y * l,τ (x t )∥ 2 ≤ (1 -µ hy β t,k )∥y t,k -y * l,τ (x t )∥ 2 . (188
)
The proof is complete.
Corollary E.5. Suppose Assumptions 4, 5 hold, and l -ℓ f,1 ≥ µ hy > 0. Recall that y * l,τ (x) := arg min y h l,τ (x, y). The sequence {y t,k } K k=1 produced by Algorithm 1 satisfies
∥y t,K -y * l,τ (x t )∥ 2 ≤ K-1 k=0 (1 -µ hy β t,k )∥y t,0 -y * l,τ (x t )∥ 2 .(189)
Proof of Corollary E.5. The result directly follows from the update of y t,k in (13a), and by applying Lemma E.4 iteratively from k = 0, . . . , K -1.
this section cite: ['b18']

Section: E.2. Convergence of the meta algorithm
Theorem E.6 (Convergence of Algorithm 1 with projected gradient descent). Suppose Assumptions 4 and 5 hold. The sequence {x t , y t } T t=0 produced by Algorithm 1 with α t = α = Θ(1), β t = β = Θ(1), γ t = O(1 + t), K t = O(1 + t) satisfies
1 T T -1 t=0 1 α 2 t x t -Proj X x t -α t ∇φ γt (x t ) 2 = O 1 T .(190)
Proof of Theorem E.6. We first prove (190), the convergence of the penalty reformulation (PP γ ). Recall that at each outer-loop iteration, Algorithm 1 does the following update
x t+1 = Proj X x t -α t (∇f 0 (x t ) -γ t ∇ x h l,τ (x t , y t+1 ))(191)
where y t+1 = y t,K approximates y * l,τ (x t ) with sufficiently large K based on Corollary E.5. Choosing β t,k = β t ≤ 1/µ hy for all k = 0, . . . , K -1, it then follows that
∥y t+1 -y * l,τ (x t )∥ 2 = ∥y t,K -y * l,τ (x t )∥ 2 ≤ (1 -µ hy β t ) K ∥y t,0 -y * l,τ (x t )∥ 2 .(192)
Let ℓ f h,1,t denote the smoothness constant for f 0 (x) -γ t h l,τ (x, y). Define the Lyapunov function V t to be
V t := f 0 (x t ) -γ t h l,τ (x t , y t+1 ).(193)
Applying the convergence of PGD for general nonconvex smooth objective yields
V t+1 -V t ≤ ⟨∇f 0 (x t ) -γ t ∇h l,τ (x t , y t+1 ), x t+1 -x t ⟩ + ℓ f h,1,t 2 ∥x t+1 -x t ∥ 2 .(194)
By the property of projection, and the update of x t , we further have
⟨∇f 0 (x t ) -γ t ∇h l,τ (x t , y t+1 ), x t+1 -x t ⟩ ≤ - 1 α t ∥x t+1 -x t ∥ 2(195)
We use the Pymoo 0.6.1 library to compute the hypervolume. The Nadir points for the hypervolume computation are given in Table 9. For a fair comparison, the Nadir points we use are the same with (Momma et al., 2022;Chen et al., 2024a).   8 below. They show that choosing τ or l to be too large or too small could degrade the performance. Nevertheless, the performances of FOOPS under suboptimal choice of the hyperparameters are still better than the baselines. Multi-lingual speech recognition. We follow the same experiment settings in (Chen et al., 2024a). We use two datasets, Librispeech and AISHELL v1. Librispeech is an English speech dataset that consists of 960 hours of labeled audio data. For our experiments, we use the "train-clean-100" subset of the Librispeech dataset for supervised training, which contains 100 hours of clean training data. Additionally, we use the full 960 hours of data for self-supervised training. AISHELL v1 is a 178-hour Mandarin speech corpus designed for various speech and speaker processing tasks. We use the full AISHELL v1 dataset for both self-supervised and supervised training. We combine these two datasets for our multi-lingual speech recognition experiments.
We use the conformer (Gulati et al., 2020) model with 8 conformer blocks as the encoder. Each block contains 512 hidden units and 8 attention heads. Each attention head has dimension 64. The convolutional kernel size is 31. Two classification heads are used. They contain two linear layers, one with 1000 output size for English, and another with 5000 output size for Chinese. The total number of parameters is around 64.5M with 58.4M encoder layer parameters and the rest being the classification layer parameters.
The loss functions we use include the Contrastive Predictive Coding (CPC) loss, and the Connectionist Temporal Classification (CTC) loss. The CPC loss (Oord et al., 2018) is a self-supervised loss to learn robust representations from unlabeled speech data. The CPC loss is designed to maximize the probability of a future sample given a contextual representation generated from the current speech sequence. The CTC loss is defined as the negative log-likelihood of the model parameter given the input sequence and the label sequence.
For all methods including the baselines, we use the step sizes α t,1 = 5 × 10 -4 for training the backbone conformer parameters and α t,2 = 5 × 10 -5 for training the classification head parameters.
this section cite: ['b64', 'b34', 'b66']

Section: Comparison of run time and memory cost.
In Table 11 we summarize the average run time and number of iterations or epochs of different methods on different datasets. The results show that FOOPS generally requires shorter run time than FERERO, but longer run time than LS. Furthermore, we summarize the memory cost in Table 12. It shows slightly higher memory cost than LS or FERERO on smaller-scale experiments. This is because although FOOPS does not compute M gradients per-iteration while FERERO does, which saves some memory, FOOPS requires storing the model parameters for both x and y while FERERO does not, which introduces extra memory cost. We leave it for future work to further reduce the memory cost and improve the efficiency of the algorithms.
this section cite: []

Section: References
Ref_id:b0 Title: A quasi-Monte Carlo method for multicriteria optimization Year: (1994)
Ref_id:b1 Title: Optimisation: Méthodes Numériques Year: (1976)
Ref_id:b2 Title: Alternating gradient-type algorithm for bilevel optimization with inexact lower-level solutions via moreau envelope-based reformulation Year: (2024)
Ref_id:b3 Title: On implicit variables in optimization theory Year: (2021)
Ref_id:b4 Title: Semianalytic and subanalytic sets Year: (1988)
Ref_id:b5 Title: Necessary conditions for nonlinear suboptimization over the weakly-efficient set Year: (1993)
Ref_id:b6 Title: Optimality conditions for minimization over the (weakly or properly) efficient set Year: ()
Ref_id:b7 Title: The Łojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems Year: (2007)
Ref_id:b8 Title: From error bounds to the complexity of first-order descent methods for convex functions Year: (2016)
Ref_id:b9 Title: Semivectorial bilevel optimization problem: Penalty approach Year: (2006)
Ref_id:b10 Title: Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline Year: (2017)
Ref_id:b11 Title: Weak sharp minima in mathematical programming Year: (1993)
Ref_id:b12 Title: Threeway trade-off in multi-objective learning: Optimization, generalization and conflict-avoidance Year: (2023)
Ref_id:b13 Title: FERERO: A flexible framework for preference-guided multi-objective learning Year: (2024)
Ref_id:b14 Title: Penalty-based methods for simple bilevel optimization under Hölderian error bounds Year: (2024)
Ref_id:b15 Title: Weighted training for cross-task learning Year: ()
Ref_id:b16 Title: Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems Year: ()
Ref_id:b17 Title: Gradient-based multi-objective deep learning: Algorithms, theories, applications, and beyond Year: (2025)
Ref_id:b18 Title: Optimization and Nonsmooth Analysis Year: (1990)
Ref_id:b19 Title: Fair machine learning through constrained stochastic optimization and an epsilon-constraint method Year: (2023)
Ref_id:b20 Title: Subgradient methods for sharp weakly convex functions Year: (2018-09)
Ref_id:b21 Title: Semivectorial bilevel programming versus scalar bilevel programming Year: (2019)
Ref_id:b22 Title: Implicit Functions and Solution Mappings Year: (2009)
Ref_id:b23 Title: Methodology and first-order algorithms for solving nonsmooth and non-strongly convex bilevel optimization problems Year: ()
Ref_id:b24 Title: Geometric categories and o-minimal structures Year: (1996)
Ref_id:b25 Title: Error bounds, quadratic growth, and linear convergence of proximal methods Year: (2018)
Ref_id:b26 Title: Multicriteria optimization Year: (2005)
Ref_id:b27 Title: Mitigating gradient bias in multiobjective learning: A provably convergent stochastic approach Year: (2023-05)
Ref_id:b28 Title: Steepest descent methods for multicriteria optimization Year: (2000)
Ref_id:b29 Title: Complexity of gradient descent for multi-objective optimization Year: (2019)
Ref_id:b30 Title: Approximation methods for bilevel programming Year: (2018)
Ref_id:b31 Title: A projection-free method for solving convex bilevel optimization problems Year: (2024)
Ref_id:b32 Title: Bilevel optimization with a multi-objective lower-level problem: Risk-neutral and risk-averse formulations Year: ()
Ref_id:b33 Title: Min-max bilevel multi-objective optimization with applications in machine learning Year: (2023)
Ref_id:b34 Title: Conformer: Convolution-augmented transformer for speech recognition Year: (2020)
Ref_id:b35 Title: Gradient descent learns linear dynamical systems Year: (2018)
Ref_id:b36 Title: The gap function of a convex program Year: (1982)
Ref_id:b37 Title: Near-optimal methods for minimizing star-convex functions and beyond Year: (2020-07)
Ref_id:b38 Title: A twotimescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actorcritic Year: ()
Ref_id:b39 Title: Revisiting scalarization in multi-task learning: A theoretical perspective Year: (2023)
Ref_id:b40 Title: On momentum-based gradient methods for bilevel optimization with nonconvex lower-level Year: (2023)
Ref_id:b41 Title: Bilevel optimization: Convergence analysis and enhanced design Year: ()
Ref_id:b42 Title: A conditional gradient-based method for simple bilevel optimization with convex lower-level problem Year: (2023)
Ref_id:b43 Title: Pareto efficient fairness in supervised learning: From extraction to tracing Year: (2021)
Ref_id:b44 Title: Linear convergence of gradient and proximal-gradient methods under the polyak-Łojasiewicz condition Year: (2016)
Ref_id:b45 Title: Convergence of approximate and incremental subgradient methods for convex optimization Year: (2004)
Ref_id:b46 Title: The generalized Łojasiewicz inequality for definable and subanalytic multifunctions Year: (2025)
Ref_id:b47 Title: On gradients of functions definable in ominimal structures Year: (1998)
Ref_id:b48 Title: On penalty methods for nonconvex bilevel optimization and firstorder stochastic approximation Year: (2024-05)
Ref_id:b49 Title: Optimizing star-convex functions Year: (2016)
Ref_id:b50 Title: Error bounds, PL condition, and quadratic growth for weakly convex functions, and linear convergences of proximal point methods Year: (2024-07)
Ref_id:b51 Title: Pareto multi-task learning Year: (2019-12)
Ref_id:b52 Title: Conflict-Averse Gradient Descent for Multi-task Learning Year: (2021-12)
Ref_id:b53 Title: BOME! Bilevel optimization made easy: A simple first-order approach Year: (2022)
Ref_id:b54 Title: Towards gradientbased bilevel optimization with non-convex followers and beyond Year: ()
Ref_id:b55 Title: Moreau envelope for nonconvex bi-level optimization: A singleloop and Hessian-free solution strategy Year: (2024)
Ref_id:b56 Title: The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning Year: (2021)
Ref_id:b57 Title: A smoothed first-order lagrangian method for structured constrained nonconvex optimization Year: (2023)
Ref_id:b58 Title: Mathematical Programs with Equilibrium Constraints Year: (1996)
Ref_id:b59 Title: Exact penalization and stationarity conditions of mathematical programs with equilibrium constraints Year: (1996)
Ref_id:b60 Title: Multi-task learning with user preferences: Gradient descent with controlled ascent in Pareto optimization Year: (2020)
Ref_id:b61 Title: Multiobjective bilevel optimization: A survey of the state-of-the-art Year: ()
Ref_id:b62 Title: Convex bi-level optimization problems with nonsmooth outer objective function Year: ()
Ref_id:b63 Title: Nonlinear Multiobjective Optimization Year: (1998)
Ref_id:b64 Title: A multi-objective/multitask learning framework induced by Pareto stationarity Year: (2022)
Ref_id:b65 Title: Smooth minimization of non-smooth functions Year: (2005-05)
Ref_id:b66 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b67 Title: Multicriterion optimization in engineering with FORTRAN programs Year: (1984)
Ref_id:b68 Title: Librispeech: an ASR corpus based on public domain audio books Year: (2015)
Ref_id:b69 Title: Optimization on Pareto sets: On a theory of multi-objective optimization Year: (2023)
Ref_id:b70 Title: Achieving optimal complexity guarantees for a class of bilevel convex optimization problems Year: (2024)
Ref_id:b71 Title: Multi-task learning as multiobjective optimization Year: (2018-12)
Ref_id:b72 Title: On penalty-based bilevel gradient descent method Year: (2025-02)
Ref_id:b73 Title: Market Structure and Equilibrium Year: (1952)
Ref_id:b74 Title: New merit functions for multiobjective optimization and their properties Year: (2024)
Ref_id:b75 Title: Bilevel and multilevel programming: A bibliography review Year: (1994)
Ref_id:b76 Title: On the convergence of adam under nonuniform smoothness Year: (2024)
Ref_id:b77 Title: Continuized acceleration for quasar convex functions in non-convex optimization Year: (2023)
Ref_id:b78 Title: Direction-oriented multiobjective learning: Simple and provable stochastic algorithms Year: (2023)
Ref_id:b79 Title: An alternating optimization method for bilevel problems under the Polyak-Łojasiewicz condition Year: (2023)
Ref_id:b80 Title: Gradientbased algorithms for multi-objective bi-level optimization Year: (2024)
Ref_id:b81 Title: Multi-objective meta learning Year: ()
Ref_id:b82 Title: A first-order multi-gradient algorithm for multi-objective bi-level optimization Year: (2024)
Ref_id:b83 Title: Constraint qualifications and necessary optimality conditions for optimization problems with variational inequality constraints Year: (2000)
Ref_id:b84 Title: Exact penalization and necessary optimality conditions for generalized bilevel programming problems Year: (1997)
Ref_id:b85 Title: Pareto navigation gradient descent: a first-order algorithm for optimization in Pareto set Year: ()
