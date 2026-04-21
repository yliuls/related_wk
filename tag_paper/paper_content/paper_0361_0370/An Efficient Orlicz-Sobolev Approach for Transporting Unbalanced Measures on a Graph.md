Title: An Efficient Orlicz-Sobolev Approach for Transporting Unbalanced Measures on a Graph
Abstract: We investigate optimal transport (OT) for measures on graph metric spaces with different total masses. To mitigate the limitations of traditional L p geometry, Orlicz-Wasserstein (OW) and generalized Sobolev transport (GST) employ Orlicz geometric structure, leveraging convex functions to capture nuanced geometric relationships and remarkably contribute to advance certain machine learning approaches. However, both OW and GST are restricted to measures with equal total mass, limiting their applicability to real-world scenarios where mass variation is common, and input measures may have noisy supports, or outliers. To address unbalanced measures, OW can either incorporate mass constraints or marginal discrepancy penalization, but this leads to a more complex two-level optimization problem. Additionally, GST provides a scalable yet rigid framework, which poses significant challenges to extend GST to accommodate nonnegative measures. To tackle these challenges, in this work we revisit the entropy partial transport (EPT) problem. By exploiting Caffarelli & McCann [12]'s insights, we develop a novel variant of EPT endowed with Orlicz geometric structure, called Orlicz-EPT. We establish theoretical background to solve Orlicz-EPT using a binary search algorithmic approach. Especially, by leveraging the dual EPT and the underlying graph structure, we formulate a novel regularization approach that leads to the proposed Orlicz-Sobolev transport (OST). Notably, we demonstrate that OST can be efficiently computed by simply solving a univariate optimization problem, in stark contrast to the intensive computation needed for Orlicz-EPT. Building on this, we derive geometric structures for OST and draw its connections to other transport distances. We empirically illustrate that OST is several-order faster than Orlicz-EPT. Furthermore, we show initial evidence on the advantages of OST for measures on a graph in document classification and topological data analysis.

Section: Introduction
Orlicz-Wasserstein (OW) extends L p geometry by leveraging a specific class of convex functions for Orlicz geometric structure. Intuitively, OW is an instance of optimal transport (OT), which utilizes Orlicz metric as its ground cost [67,32,28,3,41]. Building on this foundation, OW has proven instrumental in advancing certain machine learning approaches. For example, recent works have leveraged OW to tackle challenging problems: Altschuler & Chewi [3] use OW as a metric shift for Rényi divergence, enabling novel differential-privacy-inspired techniques to overcome longstanding challenges for fast convergence of hypocoercive differential equations, while Guha et al. [28] employ OW metric to significantly improve Bayesian contraction rates in hierarchical Bayesian nonparametric models by overcoming limitations raised from the usage of traditional OT with Euclidean ground cost. However, OW's high computational complexity, stemming from its two-level optimization formula, poses a significant limitation. To address this challenge, Le et al. [41] introduce generalized Sobolev transport (GST), a scalable variant of OW suitable for practical application domains, especially for large-scale settings. Moreover, Orlicz geometric structure has been successfully applied to various machine learning problems, including linear regression [4,66], scalable approaches [21] for reinforcement learning, kernelized support vector machines, and clustering. Additionally, Orlicz metrics play a crucial role in deriving deviation bounds for polynomialgrowth functions to approximate kernel derivatives [13], and have been used as regularization in OT problems [46]. For in-depth studies on Orlicz functions, see [2,60].
When dealing with input measures having different total masses, various approaches have been proposed in the literature to address this challenge [30,29,6,12,25,44,57,58,26,33,45,18,8,27,63,64,56,62,15,5,48,36,24,16,65,40,52,7,14,72]. These approaches for unbalanced measures have proven effective in various domains, including color transfer [8], shape matching [8], imageto-image translation [72], multi-label learning [26], positive-unlabeled learning [15], point-cloud gradient flow [72], natural language processing [36,40], topological data analysis (TDA) [36,40], generative modeling [5,72], domain adaptation [5], and robust approaches for handling noisy supports, outliers [26,5,48], or noisy ground cost [54,42].
In this work, we focus on the OT problem with Orlicz geometric structure for unbalanced measures supported on a graph metric space. On one hand, OW naturally extends OT's flexibility to handle unbalanced measures by incorporating either mass constraints or marginal difference penalization, formulated as partial OT (POT) or unbalanced OT (UOT) respectively. However, these approaches result in a more complex two-level optimization problem, analogous to POT/UOT with Orlicz metric cost, which poses significant computational challenges. On the other hand, although GST provides a scalable alternative to the computationally intensive OW, it still assumes equal-mass input measures. Moreover, due to GST's definition as an optimization over the critic function, extending it to unbalanced measures is nontrivial. To address these limitations, we revisit the entropy partial transport (EPT) problem [36,40,72] and leverage insights from Caffarelli & McCann [12] to reformulate EPT as a standard complete OT problem. Then by carefully calibrating the corresponding ground cost for its nonnegativity, we propose Orlicz-EPT and establish a theoretical foundation for solving it by a binary search algorithmic approach. Furthermore, by exploiting the dual EPT and underlying graph structure, we introduce a novel regularization approach, leading to Orlicz-Sobolev transport (OST), which scales Orlicz-EPT for practical applications.
Contribution. In summary, our contributions are two-fold as follows:
• We revisit the EPT problem and leverage Caffarelli & McCann [12]'s insights to reformulate EPT as a standard complete OT, leading to the development of the proposed Orlicz-EPT. We establish its theoretical foundations, enabling a binary search algorithmic approach for its computation. Additionally, we develop a novel regularization approach, resulting in the proposed OST. We show that OST can be efficiently computed by simply solving a univariate optimization problem, unlike the computationally intensive Orlicz-EPT.
• We derive geometric structures for OST and establish its connections to other transport distances. Our empirical results demonstrate that OST is several-order faster than Orlicz-EPT. We also provide initial evidence on the advantages of OST for document classification and TDA.
Organization. In §2, we briefly review relevant background and notions. We revisit EPT problem and propose Orlicz-EPT in §3. In §4, we introduce the computationally efficient OST. Then we derive geometric structures for OST and draw its connections to other transport distances in §5. In §6, we discuss related work. Empirical results are presented in §7, followed by concluding remarks in §8. Proofs of key theoretical results and additional materials are deferred to the Appendices. Furthermore, we have released code for our proposed approaches. 2
this section cite: ['b66', 'b31', 'b27', 'b2', 'b40', 'b2', 'b27', 'b40', 'b3', 'b65', 'b20', 'b12', 'b45', 'b1', 'b59', 'b29', 'b28', 'b5', 'b11', 'b24', 'b43', 'b56', 'b57', 'b25', 'b32', 'b44', 'b17', 'b7', 'b26', 'b62', 'b63', 'b55', 'b61', 'b14', 'b4', 'b47', 'b35', 'b23', 'b15', 'b64', 'b39', 'b51', 'b6', 'b13', 'b71', 'b7', 'b7', 'b71', 'b25', 'b14', 'b71', 'b35', 'b39', 'b35', 'b39', 'b4', 'b71', 'b4', 'b25', 'b4', 'b47', 'b53', 'b41', 'b35', 'b39', 'b71', 'b11']

Section: Preliminaries
In this section, we introduce notations, and briefly review graph, and Orlicz functions.
Graph. We follow the graph setting as in [39]. Let V, E be the sets of nodes and edges respectively. We consider a connected, undirected, and physical 3 graph G = (V, E) with positive edge lengths {w e } e∈E . For continuous graph setting, we regard G as the set of all nodes in V and all points forming the edges in E. We equip G with graph metric d G (x, y) which equals to the length of the shortest path between x and y in G. Additionally, we assume that there exists a fixed root node z 0 ∈ V such that the shortest path connecting z 0 and x is unique for any x ∈ G, i.e., the uniqueness property of the shortest paths. We denote P(G) (resp. P(G × G)) as the set of all nonnegative Borel measures on G (resp. G × G) with a finite mass. Let [x, z] be the shortest path connecting x and z in G. For x ∈ G, edge e ∈ E, define the sets Λ(x) and γ e as follows:
Λ(x) := y ∈ G : x ∈ [z 0 , y] , γ e := y ∈ G : e ⊂ [z 0 , y] .
∥f ∥ LΦ := inf t > 0 | G Φ |f (x)| t ω(dx) ≤ 1 .(2)
3 Orlicz-EPT: Entropy Partial Transport with Orlicz Geometric Structure
In this section, we revisit the entropy partial transport (EPT) problem [36,40,72], then develop Orlicz-EPT as a variant of EPT endowed with Orlicz geometric structure.
this section cite: ['b38', 'b35', 'b39', 'b71']

Section: Entropy Partial Transport (EPT)
Let γ 1 , γ 2 be the first and second marginals of γ ∈ P(G × G) respectively. For unbalanced measures µ, ν ∈ P(G), we consider the set Π ≤ (µ, ν) := {γ : γ 1 ≤ µ, γ 2 ≤ ν}. 4 Additionally, let f 1 , f 2 be the Radon-Nikodym derivatives of γ 1 w.r.t. µ and of γ 2 w.r.t. ν respectively, i.e., γ 1 = f 1 µ (0 ≤ f 1 ≤ 1, µ-a.e.) and γ 2 = f 2 ν (0 ≤ f 2 ≤ 1, ν-a.e.).
For convex and lower semicontinuous entropy functions F 1 , F 2 : [0, 1] → (0, ∞), and nonnegative weight functions w 1 , w 2 : G → [0, ∞), we consider the weighted relative entropies
Following [40, §3], by using entropy functions F 1 (s) = F 2 (s) := |s -1| and considering a Lagrange multiplier λ ∈ R conjugate to the constraint γ(G × G) = m, we instead study the problem 3 In the sense that V is a subset of Euclidean space R n , and each edge e ∈ E is the standard line segment in R n connecting the two vertices of the edge e. 4 γ1 ≤ µ means that γ1(B) ≤ µ(B) for all Borel set B ⊂ G, similarly for γ2 ≤ ν.
ET λ (µ, ν) = inf γ∈Π ≤ (µ,ν) C λ (γ),(4)
where
C λ (γ) = G w 1 µ(dx) + G w 2 ν(dx) - G w 1 γ 1 (dx) - G w 2 γ 2 (dx) + b G×G [d G (x,y
)λ]γ(dx, dy). 5   EPT as a standard OT. Following Caffarelli & McCann [12]'s insights, we can reformulate problem (4) as the standard complete OT problem. However, it is not guarantee that the corresponding standard OT has a nonnegative ground cost, e.g., see [12,36,40,72]. Therefore, such OT reformulation may not be applicable to derive corresponding OW as in [67,32,28,3,41] since N -function is only defined for nonnegative domain ( §2). Therefore, it is essential to carefully calibrate the ground cost of the corresponding standard OT of EPT to ensure its nonnegativity.
Precisely, following [40, Theorem 3.1], we henceforth consider λ ≥ 0. 6 Then let ŝ be a point outside graph G, i.e., ŝ / ∈ G, and extend graph metric cost d G on G to a new nonnegative cost function ĉ with bλ-deviation on Ĝ := G ∪ {ŝ} as follows:
ĉ(x, y) :=      b d G (x, y) if x, y ∈ G, w 1 (x) + bλ if x ∈ G and y = ŝ, w 2 (y) + bλ if x = ŝ and y ∈ G, bλ if x = y = ŝ.(5)
For unbalanced measures µ, ν, we construct corresponding probability (balanced
) measures μ = µ+ν(G)δ ŝ µ(G)+ν(G) and ν = ν+µ(G)δ ŝ µ(G)+ν(G)
. Let Π(μ, ν) := γ ∈ P( Ĝ × Ĝ) : μ(U ) = γ(U × Ĝ), ν(U ) = γ( Ĝ × U ) for all Borel sets U ⊂ Ĝ , then one can recast EPT (4) as a standard OT with cost ĉ.
Proposition 3.1. Consider the standard OT W ĉ between probability measures μ, ν with cost ĉ,
W ĉ( μ, ν) := inf γ∈Π(μ,ν) Ĝ× Ĝ ĉ(x, y)γ(dx, dy),(6)
then we have
KT(µ, ν) := (µ(G) + ν(G)) (W ĉ( μ, ν) -bλ) = ET λ (µ, ν). (7
)
The proof is placed in Appendix §A.2.1.
Thus, we have reformulated EPT (4) for unbalanced measures as a corresponding standard complete OT (7) with nonnegative ground cost. Consequently, we bypass the technical challenges inherent in unbalanced settings and can leverage abundant existing results and approaches in the standard balanced setting for OT problems with unbalanced measures on a graph. Remark 3.2 (Nonnegativity). Unlike existing approaches, e.g., as in [12,36,40,72], the new ground cost ĉ of the corresponding standard OT problem (7) for EPT is guaranteed to be nonnegative. Our calibration is essential for developing the associated OW from its standard OT as in [67,32,28,17].
this section cite: ['b11', 'b11', 'b35', 'b39', 'b71', 'b66', 'b31', 'b27', 'b2', 'b40', 'b6', 'b11', 'b35', 'b39', 'b71', 'b66', 'b31', 'b27', 'b16']

Section: Orlicz-EPT
Following the approaches in [67,32,28,17], we define Orlicz-EPT, which is EPT endowed with an Orlicz geometric structure, based on the standard OT problem (7) as follows:
OE Φ (µ, ν) := (µ(G) + ν(G)) (W Φ (μ, ν) -bλ),(8)
where
W Φ (μ, ν) := inf γ∈Π(μ,ν) inf t > 0 : Ĝ× Ĝ Φ ĉ(x,y) t dγ(x, y) ≤ 1 .
It should be noted that, similar to OW, Orlicz-EPT (8) is derived from the standard OT problem (7), which circumvents all challenges coming from the setting of unbalanced measures.
We next show that the objective function of Orlicz-EPT is monotone non-increasing w.r.t. t.
this section cite: ['b66', 'b31', 'b27', 'b16', 'b6']

Section: Proposition 3.3 (Monotonicity).
Let Φ be an N -function, and let ĉ be the cost given by (5). For any probability measures μ and ν on Ĝ, define
A(t; μ, ν) := inf γ∈Π(μ,ν) Ĝ× Ĝ Φ ĉ(x, y) t dγ(x, y) for t > 0. (9
)
Then the function t ∈ (0, +∞) -→ A(t; μ, ν) is monotone non-increasing.
The proof is placed in Appendix §A.2.2. 5 The relationship between Problem (3) and Problem (4) is established in [40,Theorem A.1]. 6 The dual EPT result is the foundation for developing Orlicz-Sobolev transport in §4, where λ is nonnegative.
this section cite: ['b4', 'b39']

Section: Computation.
Observe that for a fixed t, A is a standard OT problem between μ and ν with the cost function Φ ĉ(•,•) t . For computational efficiency, 7 we consider its corresponding entropic regularization [20], and show that the monotonicity is preserved.
this section cite: ['b19']

Section: Proposition 3.4 (Entropic regularization). Define the entropic regularization of A as
A ε (t; μ, ν) := inf γ∈Π(μ,ν) Ĝ× Ĝ Φ ĉ(x, y) t dγ(x, y) -εH(γ) ,(10)
where ε ≥ 0 and H is Shannon entropy defined by H(γ) := -Ĝ× Ĝ(log γ(x, y) -1)dγ(x, y). Then the function t ∈ (0, +∞) -→ A ε (t; μ, ν) is monotone non-increasing.
The proof is placed in Appendix §A.2.3.
In addition, we obtain the following upper and lower bounds for A ε .
this section cite: []

Section: Proposition 3.5 (Bounds).
Let supp(•) be a set of supports of a measure, then we have
A ε W ĉ( μ, ν) Φ -1 (1 + ε [H(μ) + H(ν) -1])
; μ, ν ≥ 1, and
A ε L μ,ν Φ -1 (1 + ε) ; μ, ν ≤ 1,
where L μ,ν := max x∈supp(μ),y∈supp(ν) ĉ(x, y).
The proof is placed in Appendix §A.2.4.
Thanks to the monotonicity of A ε in Proposition 3.4 and the limits of A ε in Proposition 3.5, we can leverage the binary search approach to compute the entropic regularized Orlicz-EPT, which corresponds to the original Orlicz-EPT (8). Precisely, this entropic regularization is defined as
OE Φ,ε (µ, ν) := (µ(G) + ν(G)) (W Φ,ε (μ, ν) -bλ),(11)
where
W Φ,ε (μ, ν) := inf γ∈Π(μ,ν) inf t > 0 : Ĝ× Ĝ Φ ĉ(x,y) t dγ(x, y) -εH(γ) ≤ 1 .
Discussions. Orlicz-EPT is a novel variant of EPT that incorporates Orlicz geometric structure. [12]'s insights and carefully calibrating the ground cost of the corresponding standard OT to ensure its nonnegativity, we are able to bypass all challenges of unbalanced measures and derive the proposed Orlicz-EPT from the standard OT, similar to OW [67]. We note that OE Φ,ε (11) performs binary search with quadratic complexity A ε (10), instead of dealing with super-cubic complexity A (9) in OE Φ (8). Unfortunately, the two-level optimization structure of OE Φ,ε still retains significant complexity, severely limiting its practical applications, particularly in large-scale settings. To address this computational challenge, in the next section we exploit the dual EPT and graph structure to develop a novel regularization approach, resulting in the proposed Orlicz-Sobolev transport. This approach adopts the Orlicz geometric structure used in Orlicz-EPT, but offers a much more efficient computation.
this section cite: ['b7', 'b11', 'b66', 'b10', 'b7']

Section: Leveraging Caffarelli & McCann

this section cite: []

Section: Orlicz-Sobolev Transport: A Scalable Variant of Orlicz-EPT
In this section, we leverage the dual EPT and underlying graph structure to develop a novel regularization approach, resulting in the proposed Orlicz-Sobolev transport (OST).
Dual EPT. For b-Lipschitz w 1 , w 2 (w.r.t. d G ), from [40, Corollary 3.2], the dual EPT is ET λ (µ, ν) = sup f ∈U G f (µ -ν) - bλ 2 µ(G) + ν(G) ,(12)
where y) . Let Ψ be the complement N -function of Φ and ω be a nonnegative Borel measure on G. Then let WL Ψ (G, ω) be the graph-based Orlicz-Sobolev space [41, Definition 3.1] associated to Ψ and ω. Inspired by the approach of GST [41], we consider the critic function f ∈ U within WL Ψ (G, ω). Consequently, the b-Lipschitz constraint on the critic function f ∈ U is replaced by ∥f ′ ∥ LΨ ≤ b.
U := f ∈ C(G) : -w 2 -bλ 2 ≤ f ≤ w 1 + bλ 2 , |f (x) -f (y)| ≤ b d G (x,
For f ∈ WL Ψ (G, ω), we have f (x) = f (z 0 ) + [z0,x] f ′ (y)ω(dy), ∀x ∈ G. Let 1 be the indicator function. Then by using the generalized Hölder inequality [2, §8.11] and ∥f ′ ∥ LΨ ≤ b, we can control the integral part over the generalized graph derivative f ′ for f (x) as follows:
[z0,x] f ′ (y)ω(dy) ≤ 2 ∥f ′ ∥ LΨ 1 [z0,x] LΦ ≤ 2b 1 [z0,x] LΦ ≤ 2b Φ -1 (1/ω(G)) ,(13)
where the last inequality is due to the increasing property of N -function Φ. Therefore, instead of the bounded constraint on the critic function f in U, we constraint only on f (z 0 ). Definition 4.1 (Orlicz-Sobolev transport (OST)). For α ∈ [0, 1 2 (bλ + w 1 (z 0 ) + w 2 (z 0 ))], let I α := -w 2 (z 0 ) -bλ 2 + α, w 1 (z 0 ) + bλ 2 -α . The Orlicz-Sobolev transport for µ, ν ∈ P(G) is defined
OS Φ,α (µ, ν) := sup f ∈UΨ,α G f (x)µ(dx) - G f (x)ν(dx) ,(14)
where
U Ψ,α := f ∈ WL Ψ (G, ω) : ∥f ′ ∥ LΨ ≤ b, f (z 0 ) ∈ I α .
Intuitively, U Ψ,α is the collection of all functions f expressed by f (x) = s + [z0,x] h(y)ω(dy), ∀x ∈ G, where s ∈ I α , and ∥h∥ LΨ ≤ b. The upper bound constraint on α is to ensure that I α is nonempty. When α = 0, I α is the largest interval. Also, OST is an instance of the integral probability metric [49].
Computation. Given unbalanced measures µ, ν ∈ P(G), for brevity let us define
Θ := w 1 (z 0 ) + bλ 2 -α if µ(G) ≥ ν(G), w 2 (z 0 ) + bλ 2 -α if µ(G) < ν(G).(15)
Theorem 4.2 (Univariate optimization problem for OST). OST can be computed as follows
OS Φ,α (µ, ν) = Θ|µ(G) -ν(G)| + inf k>0 1 k 1 + G Φ (kb |µ(Λ(x)) -ν(Λ(x))|) ω(dx) . (16
)
The proof is placed in Appendix §A.2.5.
We derive the discrete case for OST which provides an explicit expression for the integral in (63).
this section cite: ['b40', 'b48', 'b62']

Section: Corollary 4.3 (Discrete case).
Let ω be the length measure of graph G, and assume that input measures µ, ν are supported on nodes in V of graph G. 8 Then, we have
OS Φ,α (µ, ν) = Θ|µ(G) -ν(G)| + inf k>0 1 k 1 + e∈E w e Φ(kb |µ(γ e ) -ν(γ e )|) . (17
)
The proof is placed in Appendix §A.2.6.
Therefore, OST can be efficiently computed by simply solving the univariate optimization problem (17), thanks to the proposed novel regularization for critic functions in U Ψ,α . Remark 4.4 (Non-physical graph). In §2, G is assumed to be a physical graph.
Corollary 4.3 implies that OST only depends on graph structure (V, E) and edge weights w e when input measures are supported on nodes in V of G. Hence, OST is applicable for non-physical graph G for such cases. Remark 4.5 (Complementary pairs of N -functions). Corollary 4.3 also implies that one can compute OST with N -function Φ without involving its complementary N -function Ψ (17), unlike its definition (14). The univariate optimization formula (17)) for OST requires that Ψ is finite-valued, which is satisfied for any N -function Φ as it grows faster than linear. Preprocessing for γ e . Similar to the GST computation [41], we precompute set γ e (1) for all edge e in G. More concretely, we apply the Dijkstra algorithm to recompute the shortest paths from z 0 to all other vertices in V with complexity O(|E| + |V | log |V |), where | • | denotes the set cardinality. Sparsity. Observe that for every x ∈ supp(µ), its mass is gathered into µ(γ e ) if and only if e ⊂ [z 0 , x] [41]. Let E µ,ν := {e ∈ E | ∃z ∈ (supp(µ) ∪ supp(ν)), e ⊂ [z 0 , z]} ⊂ E. Then it suffices to compute the summation only over edges e ∈ E µ,ν in (17) for OST, i.e., screen out all edges e ∈ E \ E µ,ν .
this section cite: []

Section: Theoretical Properties
In this section, we leverage the computational efficiency of OST to derive its geometric structure and explore its connections to other transport distances.
Geometric structures of OST. Proposition 5.1 (Geometric structure). Let 0 ≤ α < bλ 2 +min{w 1 (z 0 ), w 2 (z 0 )} and µ, ν, σ ∈ P(G). i) OS Φ,α (µ + σ, ν + σ) = OS Φ,α (µ, ν).
ii) OS Φ,α is a divergence, 9 and OS Φ,α (µ, ν) ≤ OS Φ,α (µ, σ) + OS Φ,α (σ, ν).
iii) With an additional assumption w 1 (z 0 ) = w 2 (z 0 ), then OS Φ,α is a metric.
We next establish connections of OST with other transport distances, including GST [41], Sobolev transport (ST) [39], unbalanced Sobolev transport (UST) [40].
Connection of OST with GST. Denote GS Φ for the GST with N -function Φ. Proposition 5.2. For µ(G) = ν(G), b = 1, then OS Φ,α (µ, ν) = GS Φ (µ, ν).
this section cite: ['b40', 'b38', 'b39']

Section: Connection of OST with ST.
Denote S p for the p-order ST, for 1 < p < ∞.
Proposition 5.3. For µ(G) = ν(G), b = 1, and Φ(t) = (p-1) p-1 p p t p , then OS Φ,α (µ, ν) = S p (µ, ν).
Connection of OST with UST. Denote US p,α for UST, for 1 < p < ∞. Proposition 5.4. For N -function Φ(t) = (p-1) p-1 p p t p , then OS Φ,α (µ, ν) = US p,α (µ, ν). Additionally, we investigate the limit case for N -function, i.e., Φ(t) = t, 10 for OST and Orlicz-EPT. Proposition 5.5 (Limit case for OST). For Φ(t) = t, and with the same assumptions as in Corollary 4.3, then OST yields a closed-form expression as follows:
OS Φ,α (µ, ν) = b e∈E w e |µ(γ e ) -ν(γ e )| + Θ|µ(G) -ν(G)|.(18)
Proposition 5.6 (Limit case for Orlicz-EPT). For Φ(t) = t, then we have OE Φ (µ, ν) = KT(µ, ν) for every µ, ν ∈ P(G). Proposition 5.7 (Relation of OST and Orlicz-EPT). For Φ(t) = t, length measure ω on G, b-Lipschitz w 1 , w 2 (w.r.t. d G ), α = 0, and p = 1, then OS Φ,α (µ, ν) ≥ OE Φ (µ, ν)
+ bλ 2 (µ(G) + ν(G)).
The proofs for these theoretical results (in §5) are respectively placed in §A.2.7- §A.2.13.
this section cite: []

Section: Related Works and Discussions
In this section, we discuss relations between our proposals with related works in the literature.
GST [41]. Proposition 5.2 shows that OST provably generalizes GST [41] for unbalanced measures. We emphasize that GST is restricted for balanced measures and is defined as an optimization over the critic function, making it nontrivial to directly extend it to accommodate unbalanced measures.
EPT [36,40,72]. Orlicz-EPT and OST are developed from the primal and dual EPT respectively. Notably, the corresponding standard OT following [12] is not guarantee nonnegativity for ground cost, see [12,36,40]. Additionally, Caffarelli & McCann [12]'s insights may not be applicable to certain other UOT formulations, such as those proposed in [6,26,18,63,64,27,5,48,52]. The calibration is essential to guarantee nonnegativity for ground cost of the corresponding standard OT for EPT, paving ways to develop Orlicz-EPT. Similar to OW, Orlicz-EPT is derived from standard OT, thereby circumventing the challenges associated with unbalanced measures. Furthermore, we derive a novel regularization, resulting in the proposed OST with an efficient computation.
UST [40] and ST [39] Proposition 5.4 shows that OST provably generalizes UST to a more general collection of N -functions. Consequently, OST also provably generalizes ST to unbalanced measures, and to a more general set of N -functions (see Proposition 5.3).
Measures on a graph. We study OT problem between two unbalanced measures supported on the same graph, a setting also explored in [40]. One should distinguish our considered problem with the research lines on computing either distances/discrepancies [55,74,73,22,11,47] or kernels [10,34,53,61] between two (different) input graphs.
this section cite: ['b40', 'b40', 'b35', 'b39', 'b71', 'b11', 'b11', 'b35', 'b39', 'b11', 'b5', 'b25', 'b17', 'b62', 'b63', 'b26', 'b4', 'b47', 'b51', 'b39', 'b38', 'b39', 'b54', 'b73', 'b72', 'b21', 'b10', 'b46', 'b9', 'b33', 'b52', 'b60']

Section: Experiments
In this section, we illustrate that the computation of Orlicz-EPT is costly. Especially, OST is severalorder faster than Orlicz-EPT. Following the problem setups in [40], we evaluate OST for unbalanced measures supported a given graph, 11 and show initial evidences on its advantages for document classification and topological data analysis (TDA).
this section cite: ['b39']

Section: Document classification.
We use 4 real-world document datasets: TWITTER, RECIPE, CLASSIC, and AMAZON as in [40], and summarize their characteristics in Figure 2. By regarding each word in a document as a support with a unit mass, we represent each document as a nonnegative measure. Consequently, the representations of documents with different lengths are measures with different total mass. We apply the same word embedding procedure in [40] to map words into vectors in R 300 .
this section cite: ['b39', 'b39']

Section: TDA.
We consider orbit recognition on Orbit dataset [1], and object shape classification on MPEG7 dataset [35] as in [40]. We summarize these dataset characteristics in Figure 3. We use persistence diagrams (PD), a multiset of 2-dimensional data points summarized topological features, to represent objects of interest. We then consider each data point in PD as a support with a unit-mass, and represent PD as nonnegative measures. As a result, PD having different numbers of topological features are presented as measures with different total mass. 12   Graph. Following [40], we use the graphs G Log and G Sqrt [39, §5] for our experiments, 13 which empirically satisfy the assumptions in §2. Additionally, we set M = 10 4 for the number of nodes for these graphs, except experiments on MPEG7 dataset with M = 10 3 due to its small size.
this section cite: ['b0', 'b34', 'b39', 'b39']

Section: N -function.
Following [41], we consider two N -functions: Φ 1 (t) = exp(t) -t -1, and Φ 2 (t) = exp(t 2 ) -1, and the limit case of N -functions, i.e., Φ 0 (t) = t.
Parameters. For simplicity, we follow the experimental setup in [40]. We set λ = b = 1, α = 0, and consider the weight functions w 1 (x) = w 2 (x) = a 1 d G (z 0 , x) + a 0 where a 1 = b and a 0 = 1. The entropic regularization ε is chosen from {0.01, 0.1, 1, 10}, typically via cross validation.
Optimization algorithm. For OST, we use fmincon MATLAB solver with trust-region-reflective algorithm, for solving the univariate optimization problem (17).
this section cite: ['b40', 'b39', 'b16']

Section: SVM classification.
For document classification and TDA, we use support vector machine (SVM) with kernels exp(-t d(•, •)), where d is a distance/discrepancy (e.g., OST, Orlicz-EPT) for unbalanced measures on a graph, and t > 0. We regularize Gram matrices by adding a sufficiently large diagonal term for indefinite kernels [20]. Additionally, we note that there are more than 29M pairs for AMAZON which we need to evaluate distances/discrepancies for SVM in each run to illustrate the experiment scale. 14   Set up. We randomly split each dataset into 70%/30% for training and test, and use 10 repeats. We basically choose hyper-parameters via cross validation. More concretely, we choose kernel hyperparameter from {1/q s , 0.5/q s , 0.2/q s } with s = 10, 20, . . . , 90, where q s is the s% quantile of a subset of distances observed on a training set; SVM regularization hyperparameter from {0.01, 0.1, 1, 10}; root node z 0 from a random 10-root-node subset of V in graph G. Note that reported time consumption includes all preprocessing procedures, e.g., preprocessing for γ e for OST. 11 One should distinguish the considered problem, i.e., compare two input unbalanced measures supported in the same graph, with either OT or Gromov-Wasserstein problem between two different input graphs ( §6). 12 We distinguish our problem setup with [41], where objects are represented as probability measures instead. 13 Due to the space limitation, corresponding experimental results for graph GLog are placed in §B.3. 14 See Table 1 in §B.2 for the details. We compare the time consumption of OST and Orlicz-EPT with Φ 1 , Φ 2 , and with the limit case Φ 0 .
Set up. We randomly sample 10 4 pairs of nonnegative measures on AMAZON dataset for evaluation. We consider M = 10 3 for graphs, and ε = 0.1 for Orlicz-EPT.
this section cite: ['b19', 'b40']

Section: Results.
We illustrate the time consumptions on G Sqrt in Figure 1. OST is several-order faster than Orlicz-EPT, i.e., at least 250×, 13800×, 11200× for Φ 0 , Φ 1 , Φ 2 respectively. Notably, for Nfunctions Φ 1 , Φ 2 , Orlicz-EPT takes at least 2.6 days, while OST takes less than 21 seconds. Note that for the limit case Φ 0 , Orlicz-EPT is equal to EPT on a graph (Proposition 5.6), and OST admits a closed-form expression (Proposition 5.5) for a fast computation. Consequently, Orlicz-EPT and OST with Φ 0 is more computationally efficient than those with Φ 1 , Φ 2 . Set up. We evaluate OST with Φ 0 , Φ 1 , Φ 2 ( §7.1), denote them as OST-Φ i for i = 0, 1, 2. We exclude Orlicz-EPT due to their heavy computations ( §7.1). Additionally, following [40], we consider UOT [26,63] with ground cost d G , 15 and special cases with tree-structure graph. More concretely, we randomly sample a tree from the given graph G, then consider the regularized EPT and d 0 , denoted as d 0 -Tree and regEPT-Tree [36, Proposition 3.8, Equation ( 9)].
this section cite: ['b39', 'b25', 'b62']

Section: Document Classification
Results. We show SVM results and time consumptions of kernels on G Sqrt in Figure 2. The performances of OST with all Φ functions are comparable to UOT, but the computation of UOT is more costly than OST. Additionally, OST outperforms d 0 -Tree and regEPT-Tree. However, the computations of OST-Φ 1 , OST-Φ 2 are more expensive while the computation of OST-Φ 0 is comparative to those fast-computational variants of UOT on tree (i.e., d 0 -Tree and regEPT-Tree). Moreover, OST-Φ 1 and OST-Φ 2 improve performances of OST-Φ 0 , but their computational time is several-order higher, which may imply that Orlicz geometric structure in OST may be helpful for document classification. The performances of UOT also agree with observations in [40]. Results. We illustrate SVM results and time consumptions of kernels on G Sqrt in Figure 3. The performances of OST with all Φ functions compare favorably with other transport distance approaches. Especially, the performances of OST-Φ 1 and OST-Φ 2 compare favorably with those of OST-Φ 0 , but it comes with higher computational cost (i.e., OST-Φ 0 has a closed-form expression (Proposition 5.5)), which may imply that Orlicz geometric structure may be also helpful for TDA tasks.
this section cite: ['b39']

Section: Conclusion
In this work, we propose novel approaches to extend OW/GST for unbalanced measures on a graph. Building on the EPT problem and leveraging Caffarelli & McCann [12]'s insights, we derive Orlicz-EPT by recasting it as a standard OT with a carefully calibrated ground cost, thereby bypassing challenges raised from unbalanced measures. Furthermore, by exploiting dual EPT and the underlying geometric structure, we formulate a novel regularization, resulting in the proposed OST, which is efficient in computation. It provably suffices to compute OST by simply solving a univariate optimization problem, unlike the computationally intensive Orlicz-EPT. Moreover, we illustrate empirical evidence on the advantages of OST in document classification and topological data analysis.
this section cite: ['b11']

Section: References
Ref_id:b0 Title: Persistence images: A stable vector representation of persistent homology Year: (2017)
Ref_id:b1 Title:  Year: (2003)
Ref_id:b2 Title: Faster high-accuracy log-concave sampling via algorithmic warm starts Year: (2024)
Ref_id:b3 Title: Subspace embedding and linear regression with Orlicz norm Year: (2018)
Ref_id:b4 Title: Robust optimal transport with applications in generative modeling and domain adaptation Year: (2020)
Ref_id:b5 Title: Numerical resolution of an "unbalanced" mass transport problem Year: (2003)
Ref_id:b6 Title: Slicing unbalanced optimal transport Year: (2024)
Ref_id:b7 Title: SPOT: sliced partial optimal transport Year: (2019)
Ref_id:b8 Title: Sliced and Radon Wasserstein barycenters of measures Year: (2015)
Ref_id:b9 Title: Graph kernels: State-of-the-art and future challenges Year: (2020)
Ref_id:b10 Title: Learning to predict graphs with fused gromov-wasserstein barycenters Year: (2022)
Ref_id:b11 Title: Free boundaries in optimal transport and Monge-Ampere obstacle problems Year: (2010)
Ref_id:b12 Title: Orlicz random Fourier features Year: (2020)
Ref_id:b13 Title: One for all and all for one: Efficient computation of partial Wasserstein distances on the line Year: (2025)
Ref_id:b14 Title: Partial optimal tranport with applications on positiveunlabeled learning Year: (2020)
Ref_id:b15 Title: Unbalanced optimal transport through non-negative penalized linear regression Year: (2021)
Ref_id:b16 Title: An optimization perspective on log-concave sampling and beyond Year: (2023)
Ref_id:b17 Title: Unbalanced optimal transport: Dynamic and Kantorovich formulations Year: (2018)
Ref_id:b18 Title: Elements of information theory Year: (1999)
Ref_id:b19 Title: Sinkhorn distances: Lightspeed computation of optimal transport Year: (2013)
Ref_id:b20 Title: Fast distance oracles for any symmetric norm Year: (2022)
Ref_id:b21 Title: COPT: Coordinated optimal transport on graphs Year: (2020)
Ref_id:b22 Title: Persistent homology -A survey Year: (2008)
Ref_id:b23 Title: Unbalanced minibatch optimal transport; applications to domain adaptation Year: (2021)
Ref_id:b24 Title: The optimal partial transport problem. Archive for rational mechanics and analysis Year: (2010)
Ref_id:b25 Title: Learning with a Wasserstein loss Year: (2015)
Ref_id:b26 Title: Unnormalized optimal transport Year: (2019)
Ref_id:b27 Title: On excess mass behavior in Gaussian mixture models with Orlicz-Wasserstein distances Year: (2023)
Ref_id:b28 Title: Extended Kantorovich norms: a tool for optimization Year: (2002)
Ref_id:b29 Title: Kantorovich-Rubinstein norm and its application in the theory of Lipschitz spaces Year: (1992)
Ref_id:b30 Title: DNA microarrays: Design principles for maximizing ergodic, chaotic mixing Year: (2007)
Ref_id:b31 Title: On interpolation and curvature via Wasserstein geodesics Year: (2017)
Ref_id:b32 Title: A new optimal transport distance on the space of finite Radon measures Year: (2016)
Ref_id:b33 Title: A survey on graph kernels Year: (2020)
Ref_id:b34 Title: Shape descriptors for non-rigid shapes with a single closed contour Year: (2000)
Ref_id:b35 Title: Entropy partial transport with tree metrics: Theory and practice Year: (2021)
Ref_id:b36 Title: Tree-sliced variants of Wasserstein distances Year: (2019)
Ref_id:b37 Title: Flow-based alignment approaches for probability measures in different spaces Year: (2021)
Ref_id:b38 Title: Sobolev transport: A scalable metric for probability measures with graph metrics Year: (2022)
Ref_id:b39 Title: Scalable unbalanced Sobolev transport for measures on a graph Year: (2023)
Ref_id:b40 Title: Generalized Sobolev transport for probability measures on a graph Year: (2024)
Ref_id:b41 Title: Optimal transport for measures with noisy tree metric Year: (2024)
Ref_id:b42 Title: Scalable Sobolev IPM for probability measures on a graph Year: (2025)
Ref_id:b43 Title: Imaging with Kantorovich-Rubinstein discrepancy Year: (2014)
Ref_id:b44 Title: Optimal entropy-transport problems and a new Hellinger-Kantorovich distance between positive measures Year: (2018)
Ref_id:b45 Title: Orlicz space regularization of continuous optimal transport problems Year: (2022)
Ref_id:b46 Title: Fused Gromov-Wasserstein graph mixup for graph-level classifications Year: (2023)
Ref_id:b47 Title: Outlier-robust optimal transport Year: (2021)
Ref_id:b48 Title: Integral probability metrics and their generating classes of functions Year: (1997)
Ref_id:b49 Title: Orlicz spaces and modular spaces Year: (2006)
Ref_id:b50 Title: Sliced Wasserstein with random-path projecting directions Year: (2024)
Ref_id:b51 Title: On unbalanced optimal transport: Gradient methods, sparsity and approximation error Year: (2023)
Ref_id:b52 Title: Graph kernels: A survey Year: (2021)
Ref_id:b53 Title: Subspace robust Wasserstein distances Year: (2019)
Ref_id:b54 Title: GOT: An optimal transport framework for graph comparison Year: (2019)
Ref_id:b55 Title: On unbalanced optimal transport: An analysis of Sinkhorn algorithm Year: (2020)
Ref_id:b56 Title: Generalized Wasserstein distance and its application to transport equations with source Year: (2014)
Ref_id:b57 Title: On properties of the generalized Wasserstein distance Year: (2016)
Ref_id:b58 Title: Wasserstein barycenter and its application to texture mixing Year: (2011)
Ref_id:b59 Title: Theory of Orlicz spaces Year: (1991)
Ref_id:b60 Title: Tree structure for the categorical Wasserstein Weisfeiler-Lehman graph kernel Year: ()
Ref_id:b61 Title: Fast unbalanced optimal transport on tree Year: (2020)
Ref_id:b62 Title: Sinkhorn divergences for unbalanced optimal transport Year: (2019)
Ref_id:b63 Title: Faster unbalanced optimal transport: Translation invariant Sinkhorn and 1-D Frank-Wolfe Year: (2022)
Ref_id:b64 Title: Unbalanced optimal transport, from theory to numerics. Handbook of Numerical Analysis Year: (2023)
Ref_id:b65 Title: Efficient symmetric norm regression via linear sketching Year: (2019)
Ref_id:b66 Title: Generalized Orlicz spaces and Wasserstein distances for convex-concave scale functions Year: (2011)
Ref_id:b67 Title: Tree-sliced Wasserstein distance with nonlinear projection Year: (2025)
Ref_id:b68 Title: Spherical tree-sliced Wasserstein distance Year: (2025)
Ref_id:b69 Title: Distance-based tree-sliced Wasserstein distance Year: (2025)
Ref_id:b70 Title: Tree-sliced Wasserstein distance: A geometric perspective Year: (2025)
Ref_id:b71 Title: Tree-sliced Entropy Partial Transport Year: (2025)
Ref_id:b72 Title: Scalable Gromov-Wasserstein learning for graph partitioning and matching Year: (2019)
Ref_id:b73 Title: Gromov-Wasserstein learning for graph matching and node embedding Year: (2019)
