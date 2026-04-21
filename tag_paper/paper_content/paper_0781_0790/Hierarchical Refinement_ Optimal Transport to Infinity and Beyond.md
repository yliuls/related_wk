Title: Hierarchical Refinement: Optimal Transport to Infinity and Beyond
Abstract: Optimal transport (OT) has enjoyed great success in machine learning as a principled way to align datasets via a least-cost correspondence, driven in large part by the runtime efficiency of the Sinkhorn algorithm (Cuturi, 2013). However, Sinkhorn has quadratic space and time complexity in the number of points, limiting scalability to larger datasets. Low-rank OT achieves linear complexity, but by definition, cannot compute a oneto-one correspondence between points. When the optimal transport problem is an assignment problem between datasets then an optimal mapping, known as the Monge map, is guaranteed to be a bijection. In this setting, we show that the factors of an optimal low-rank coupling co-cluster each point with its image under the Monge map. We leverage this invariant to derive an algorithm, Hierarchical Refinement (HiRef), that dynamically constructs a multiscale partition of each dataset using low-rank OT subproblems, culminating in the bijective Monge map. Hierarchical Refinement runs in log-linear time and linear space, retaining the advantages of low-rank OT while overcoming its limited resolution. We demonstrate the advantages of Hierarchical Refinement on several datasets, including ones containing over a million points, scaling full-rank OT to problems previously beyond Sinkhorn's reach.

Section: Introduction
Optimal transport (OT) is a mathematical framework for comparing probability distributions µ and ν. Given a cost function c, the Monge problem is to find a mapping T transforming a distribution µ into ν (i.e. T ♯ µ = ν) with least-cost. A relaxation of this problem, called the Kantorovich prob-Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
lem, instead seeks a least-cost coupling γ between µ and ν. In the Kantorovich formulation, mass splitting is allowed and thus a solution always exists; in contrast, a Monge map between µ and ν may not exist. When a Monge map T does exist, the solution to the Kantorovich problem is a coupling γ = (id × T ) ♯ µ supported on its graph, and the Monge and Kantorovich problems coincide (Brenier, 1991).
When µ and ν are discrete uniform measures on n points the optimal transport problem reduces to an assignment problem. Classical algorithms such as the Hungarian algorithm and Network Simplex (Tarjan, 1997;Orlin, 1997), solve this in cubic time. The Sinkhorn algorithm (Cuturi, 2013) solves the entropy-regularized Kantorovich problem with quadratic runtime, greatly expanding the applicability of computational OT. However, the Sinkhorn algorihtm requires quadratic space to store the coupling γ.
In recent years, OT has found numerous applications in machine learning and across science, including: domain adaptation (Courty et al., 2014;Solomon et al., 2015), selfattention (Tay et al., 2020;Sander et al., 2022;Geshkovski et al., 2023), computational biology (Schiebinger et al., 2019;Yang et al., 2020;Zeira et al., 2022;Bunne et al., 2023;Halmos et al., 2025b;Klein et al., 2025), unpaired data translation (Korotin et al., 2021;De Bortoli et al., 2024;Tong et al., 2024;Klein et al., 2024), and alignment problems in transformers and large language models (Melnyk et al., 2024;Li et al., 2024). The least-cost principle of optimal transport is crucial for training high-quality generative models using Schrödinger bridges, flow-matching, diffusion models, or neural ordinary differential equations (Finlay et al., 2020;Tong et al., 2024;De Bortoli et al., 2024;Kornilov et al., 2024;Klein et al., 2024). These models typically require millions to hundreds of millions of data-points to achieve high-performance at scale (Ramesh et al., 2021), limiting the scope of OT for generative modeling.
As modern datasets grow to have tens of thousands or even millions of points, the quadratic space and time complexity of Sinkhorn becomes increasingly prohibitive. This limitation is widely recognized in the machine learning literature, with (De Bortoli et al., 2024) noting that the quadratic complexity of optimal transport renders its application to modern datasets on the order of millions of points impractical. A number of approaches have been proposed to address scal-ing OT to massive datasets which avoid instantiating a full coupling matrix. Mini-batch OT (Genevay et al., 2018) improves scalability, but incurs significant biases (Sommerfeld et al., 2019;Korotin et al., 2021;Fatras et al., 2021a) as each mini-batch alignment is a poor representation of the global coupling. Multiple works have investigated the theoretical properties of mini-batch estimators of the coupling (Fatras et al., 2020;2021b), while others have attempted to mitigate this bias using partial or unbalanced OT that allows mass variation between mini-batches (Nguyen et al., 2022a;Fatras et al., 2021a). However, these approaches introduce additional hyperparameters to control the degree of unbalancedness, and ultimately remain biased, local approximations of the global coupling.
Neural optimal transport methods (Makkuva et al., 2020;Bunne et al., 2023;Fan et al., 2023;Korotin et al., 2023;Buzun et al., 2024), parametrize the Monge map as a neural network instead of materializing a quadratic coupling matrix. However, these methods have noted limitations recovering faithful maps (Korotin et al., 2021).
Another approach to improve space complexity of OT is to introduce a low-rank constraint on the coupling matrix in the Kantorovich problem. This has been done by parameterizing the coupling through a set of low-rank factors (Scetbon et al., 2021;2022;Scetbon & Cuturi, 2022;Scetbon et al., 2023;Halmos et al., 2024) or by using a proxy objective for the low-rank problem, factoring the transport through a small number of anchor points (Forrow et al., 2019;Lin et al., 2021). For a given rank r these approaches have O(nr) space complexity, enabling linear time and space scaling. Low-rank OT has been used successfully on datasets on the order of 10 5 samples with ranks on the order of 10 1 (Scetbon et al., 2023;Halmos et al., 2024;2025a;Klein et al., 2025), but computing full-rank couplings between datasets of sizes on the order of 10 5 and greater has not yet been accomplished.
this section cite: ['b3', 'b75', 'b58', 'b14', 'b13', 'b72', 'b76', 'b63', 'b30', 'b68', 'b80', 'b83', 'b5', 'b42', 'b44', 'b16', 'b78', 'b41', 'b52', 'b48', 'b25', 'b78', 'b16', 'b43', 'b41', 'b61', 'b16', 'b28', 'b73', 'b44', 'b22', 'b51', 'b5', 'b21', 'b45', 'b6', 'b44', 'b65', 'b67', 'b32', 'b26', 'b49', 'b67', 'b32', 'b42']

Section: Contributions
We introduce Hierarchical Refinement (HiRef), an algorithm to scalably compute a full-rank alignment between two equally-sized input datasets X and Y by solving a hierarchy of low-rank OT sub-problems. The success of this refinement is driven by a theoretical result, Proposition 3.1, stating that factors of an optimal low-rank coupling between X and Y co-cluster points X with their image under the Monge map. We use Proposition 3.1 recursively to obtain increasingly fine partitions of X and Y. At each scale, the solutions to low-rank OT sub-problems are bijections between the partitions of X and Y. Iterating to the finest scale gives a bijection between X and Y.
Hierarchical Refinement constructs a multiscale partition of each dataset, and thus is related to (Gerber & Maggioni, 2017), which introduced a general framework for multiscale optimal transport using such partitions, and the earlier work of (Mérigot, 2011). Unlike (Mérigot, 2011;Gerber & Maggioni, 2017), Hierarchical Refinement (i) does not assume multiscale partitions for each dataset are given, instead constructing them on the fly; and (ii) operates intrinsically to the data, without a mesh or anchor points in the ambient space of the data, avoiding the curse of dimensionality.
We demonstrate that Hierarchical Refinement computes OT maps efficiently in high-dimensional spaces, often matching or even outperforming Sinkhorn in terms of primal cost. Moreover, HiRef has linear space complexity and time complexity scaling log-linearly in the dataset size. Unlike low-rank OT, Hierarchical Refinement places X and Y in bijective correspondence. Hierarchical Refinement scales to over a million points, enabling the use of OT on massive datasets without incurring the bias of mini-batching.
this section cite: ['b29', 'b53', 'b53', 'b29']

Section: Background and Related Work
Suppose X = {x i } n i=1 and Y = {y j } m j=1 are datasets in the same metric space (X , d X ). Let c : X × X → R + be a cost function. This cost c is often assumed to satisfy strict convexity or to be a metric. Datasets X and Y are represented as discretely supported probability measures µ = n i=1 a i δ xi and ν = m j=1 b j δ yj for probability vectors a ∈ ∆ n and b ∈ ∆ m . Throughout, ∆ k denotes the k-simplex {p ∈ R k + : i p i = 1}, the set of probability vectors of length k.
this section cite: []

Section: Monge Problem
Optimal transport has its origin in the Monge problem (Monge, 1781), concerned with finding an optimal map T : X → Y pushing µ forward to ν:
M c (µ, ν) = min T :T ♯ µ=ν E µ c(x, T (x)) .
(1) Above, T ♯ µ is the pushforward of µ under T , the measure on Y with T ♯ µ(B) := µ(T -1 (B)) for any (measurable) set B ⊂ Y. In general, a Monge map may not exist (e.g. if m > n). However, when |X| = |Y| = n and a, b are uniform then the Monge problem becomes the assignment problem and has a bijective solution (Thorpe, 2018).
this section cite: ['b54', 'b77']

Section: Kantorovich Problem
The Kantorovich problem (Kantorovich, 1942) was introduced as a relaxation of the Monge problem. In contrast to the Monge problem, the Kantorovich problem allows mass-splitting and a solution is always guaranteed to exist. Define the transport polytope Π a,b as the following set of coupling matrices
Π a,b := P ∈ R n×m + : P1 m = a, P ⊤ 1 n = b , (2)
respectively with left (or "source") marginal a and with right (or "target") marginal b. For the cost c(•, •), define the cost matrix C by C ij = c(x i , y j ). In this discrete setting, the Kantorovich problem seeks a least cost coupling matrix P ∈ Π a,b between the probability vectors a, b associated to each measure µ, ν:
W c (µ, ν) = min P∈Π a,b ⟨C, P⟩ F .(3)
The optimal value W c (µ, ν) of ( 3) is called the c-Wasserstein distance between µ and ν.
this section cite: []

Section: Sinkhorn Algorithm and the ϵ-schedule
The Sinkhorn algorithm (Cuturi, 2013) relaxes the classical linearprogramming formulation of optimal transport by solving an entropy regularized version of (3),
W ϵ (µ, ν) := min P∈Π a,b ⟨C, P⟩ F -ϵH(P),(4)
where H(P) :=ij P ij (log P ij -1) is the Shannon entropy, and the parameter ϵ > 0 is the regularization strength. The Sinkhorn algorithm improved the O(n 3 log n) time complexity of classical techniques used for OT such as the Hungarian algorithm (Kuhn, 1955) and Network Simplex (Orlin, 1997;Tarjan, 1997) to O(n 2 log n) (Luo et al., 2023). As ϵ ↓ 0, the optimal coupling P ⋆,ϵ for (4) converges to a sparse optimal coupling for (3) at an extremal point of the transport polytope (c.f. (Peyré & Cuturi, 2019)). However, the number of iterations required scales as poly(1/ϵ), diverging as ϵ decreases. A technique used to improve this scaling is the ϵ-schedule, an adaptive, monotone-decreasing and step-dependent set of entropy parameters ϵ 1 > ϵ 2 > • • • > ϵ t fin . This anneals Problem 4 from high-entropy to low-entropy, gradually driving a dense initial condition to a sparse solution with a log (1/ϵ) rate (Chen et al., 2023).
this section cite: ['b14', 'b46', 'b58', 'b75', 'b50', 'b59', 'b8']

Section: Low-rank Optimal Transport
The nonnegative rank rk + (M) of a nonnegative matrix M ≽ 0 is the smallest number of nonnegative rank-1 matrices summing to M; i.e. rk + (M) is the smallest integer z such that there exist nonnegative vectors q 1 , . . . , q z ≽ 0 and r 1 , . . . , r z ≽ 0 satisfying M = z i=1 q i r ⊤ i . Let Π a,b (r) := {P ∈ Π a,b : rk + (P) = r} be the set of rank-r couplings. The low-rank Wasserstein problem for general cost matrix C is:
P ⋆ = arg min P∈Π a,b (r) ⟨C, P⟩ F .(5)
From (Cohen & Rothblum, 1993), each P ∈ Π a,b (r) may be decomposed as
P = r i=1 (1/g i )Q •,i R ⊤ •,i := Qdiag(1/g)R ⊤ ,(6)
where g ∈ ∆ r , Q ∈ Π a,g and R ∈ Π b,g . This factorization was introduced to optimal transport by (Scetbon et al., 2021) in the context of the general low-rank problem (5). The factors Q and R constitute co-clusterings of datasets X and Y onto the same set of r components. Other factorizations have recently been proposed (Halmos et al., 2024), using Q, R and an intermediate latent coupling T to solve (5) where X and Y have r 1 and r 2 components, respectively.
Hierarchical and Multiscale Approaches to OT Hierarchical optimal transport (Schmitzer & Schnörr, 2013) is a variant of OT modeling data and transport at two scales, using Wasserstein distances as the coarse-scale ground costs. It has been applied to document representation (Yurochkin et al., 2019), domain adaptation (El Hamri et al., 2022), sliced Wasserstein distances (Bonneel et al., 2015;Nguyen et al., 2022b) and to give a discrete formulation of transport between Gaussian mixture models (Chen et al., 2018;Delon & Desolneux, 2020). These works build interpretable, coarse-grained structure into a single coupling, rather than solving for a sequence of couplings at progressively finer scales as in the present work.
Multiscale approaches to OT generalize hierarchical OT to a progression of scales. Building on the semidiscrete approach of (Aurenhammer et al., 1998), (Mérigot, 2011) uses Lloyd's algorithm to progressively coarse-grain the target measure. More recently, using a regular family of multiscale partitions (Definition C.3) on each dataset, (Gerber & Maggioni, 2017) formalize a general hierarchical approach to the Kantorovich problem (3). They propose: (i) solving a Kantorovich problem between the coarsest partitions of X and Y in their respective multiscale families; and (ii) propagation of the optimal coupling at scale t ∈ {1, . . . , κ -1} to initialize the optimization at scale t + 1. They take as input a chain of partitions and measures across scales (X
(1) , µ 1 ) → • • • → (X (κ) , µ κ ) and (Y (1) , ν 1 ) → • • • → (Y (κ) , ν κ )
where each dataset X, Y is identified with the trivial partitions X (κ) = {{x} : x ∈ X} and Y (κ) = {{y} : y ∈ Y}. At the finest scale κ, (Gerber & Maggioni, 2017) recover the original datasets and a near optimal coupling for (3).
A naive implementation of the above idea requires quadratic memory complexity, but (Gerber & Maggioni, 2017) propose several propagation strategies to mitigate this, following (Glimm & Henscheid, 2013;Oberman & Ruan, 2015;Schmitzer, 2016). These strategies use the optimal coupling at scale t to restrict the support of the coupling computed at the next scale using local optimality criteria. In the next section, we give our own such criterion, Proposition 3.1.
this section cite: ['b12', 'b65', 'b32', 'b70', 'b82', 'b20', 'b2', 'b10', 'b18', 'b0', 'b53', 'b29', 'b29', 'b29', 'b31', 'b57', 'b69']

Section: Hierarchical Refinement

this section cite: []

Section: Low-rank optimal transport co-clusters source-target pairs under the Monge map
We first show that under a few assumptions, the optimal lowrank factors (Q ⋆ , R ⋆ ) for a variant of the low-rank Wasserstein problem (5) have qualities suited to our refinement strategy. Specifically, we parameterize low-rank couplings P of rank-r using the factorization P = Qdiag(1/g)R ⊤ of (Scetbon et al., 2021), fixing g ∈ ∆ r to be uniform. Define the following variant of ( 5):
(Q ⋆ , R ⋆ ) = arg min (Q,R) C, Qdiag(1/g)R ⊤ F (7) s.t. Q ∈ Π a,g , R ∈ Π b,g , g = (1/r)1 r
Proposition 3.1 below is the main structural result behind Hierarchical Refinement. It says that when optimal Q ⋆ and R ⋆ for (7) correspond to hard-clusterings (partitions) of each dataset, given by clustering functions q ⋆ : X → Let a, b ∈ ∆ n be uniform so that a Monge map T ⋆ : X → Y exists. If (Q ⋆ , R ⋆ ) are minimizers of (7) and correspond to clustering functions
q ⋆ : X → [r], r ⋆ : Y → [r], then for all x ∈ X one has q ⋆ (x) = r ⋆ (T ⋆ (x)) .
The proof of Proposition 3.1 is in two steps. First, we use the existence of a Monge map and its coupling P † to permute the cost C to cost C † (Definition B.1) for which the identity matrix is a Monge map. Second, supposing that strict r-Monge separability (Definition B.2) holds, we show the solution to Problem 7 with cost C † is symmetric, so that
min Q,R∈Πa,g ⟨C † , QR ⊤ ⟩ F = min Q∈Πa,g ⟨C † , QQ ⊤ ⟩ F .
Returning to the coordinate frame of the original cost C, we find that Q = P † R, implying Proposition 3.1. We note that when r = 2, optimal Q, R are hard-partitions (Lemma B.5) automatically satisfying one of the assumptions of Proposition 3.1.
this section cite: ['b65']

Section: Hierarchical Refinement Algorithm
The Hierarchical Refinement algorithm (Algorithm 1) uses Proposition 3.1 to guarantee that each low-rank step co-clusters the datasets optimally, in that x and T ⋆ (x) are assigned the same label by q ⋆ and r ⋆ . Using the same label set to partition X and Y automatically places the blocks of each partition in bijective correspondence. One then recurses on each pair of corresponding blocks (which we call a co-cluster) at the previous scale, until all blocks have size one. This guarantee holds despite that optimal (Q ⋆ , R ⋆ ) for (7) may not constitute an optimal triple (Q ⋆ , R ⋆ , g ⋆ ) for the original low-rank problem (5) under the (Scetbon et al., 2021) factorization.
A hierarchy-depth κ denotes the total number of times Algorithm 1 refines the initial trivial partitions {X}, {Y}. The effective rank at scale t is ρ t := t s=1 r s , given rankannealing schedule (r 1 , r 2 , . . . , r κ ) for which ρ κ divides n. The base rank is r base = n ρκ . Note that n/ρ t is also the size of each partition at scale t: n/ρ t = |X (t) | = |Y (t) |, and that any sequence of any factorization of n corresponds to a rank-annealing schedule.
Proposition 3.2. For any n, there exists a rank-schedule (r 1 , • • • , r κ ) factorizing n such that all partitions of Algorithm 1 at level t ∈ [0 : κ -1] satisfy strict r t+1 -Monge separability (Definition B.2). Let LROT denote an optimal rank-r solver for (7) over hard-partitions. For any satisfying rank-schedule, the map returned by Algorithm 1 is optimal and supported on the graph of the Monge map T ⋆ .
Proof. Existence follows from the trivial (r 1 ) = (n) rankschedule. For any schedule (r 1 , • • • , r κ ) satisfying Monge separability, applying the invariant of Proposition 3.1 inductively on t to level κ yields n tuples {(x, T ⋆ (x))} containing each x ∈ X and its image T ⋆ (x) under the Monge map.
If the black-box subroutine LROT in Algorithm 1 solves (7) optimally, then Hierarchical Refinement is guaranteed to recover a Monge map. In practice, we implement LROT using the low-rank solver (Halmos et al., 2024) and enforce that inner marginal g is uniform.
Let Γ t,q denote the q-th co-cluster at scale t generated by Hierarchical Refinement:
Γ t,q := (x, y) : x ∈ X (t) q , y ∈ Y (t) q , (8
)
where
X (t) = {X (t) q } ρt q=1 , Y (t) = {Y (t)
q } ρt q=1 , and define the co-clustering Γ t at scale t by:
Γ t := (X (t) q , Y (t) q ) ρt q=1
.
At scale t ∈ [κ], Hierarchical Refinement refines Γ t to Γ t+1 by running a rank r t+1 low-rank optimal transport problem between uniform g t+1 = (1/r t+1 )1 rt+1 and measures supported on each pair (X
(t) q , Y(t)
q ) in Γ t for q ∈ [ρ t ], yielding Algorithm 1 Hierarchical Refinement Require: Data X , Y; Low-rank OT solver LROT(•); Rank schedule (r 1 , r 2 , . . . , r κ ); Base rank r base (=1).
Initialize:
1: t ← 0, Γ 0 ← { (X, Y)} 2: while ∃ (X (t) q , Y (t) q ) ∈ Γ t such that 3: min{|X (t) q |, |Y (t) q |} > r base do 4: Γ t+1 ← ∅ 5: for (X (t) q , Y (t) q ) ∈ Γ t do 6: if min{|X (t) q |, |Y (t) q |} ≤ r base then 7: Γ t+1 ← Γ t+1 ∪ {(X (t) q , Y (t) q )} 8: else 9: µ X (t) q = 1 |X (t) q | x∈X (t) q δ x 10: µ Y (t) q = 1 |Y (t) q | y∈Y (t) q δ y . 11: g t+1 ← (1/r t+1 )1 rt+1 12: (Q, R) ← LROT(µ X (t) q , µ Y (t) q , g t+1 ) 13: for z = 1 → r t+1 do 14: X (t+1) z ← Assign(X (t) , Q, z) 15: Y (t+1) z ← Assign(Y (t) , R, z) 16: Γ t+1 ← Γ t+1 ∪ { (X (t+1) z , Y(t+1) z )} 17:
end for 18:
▷ Assign(S, M, z) = {s ∈ S | arg max z ′ M sz ′ = z} 19:
end if 20:
end for 21:
t ← t + 1 22: end while 23: Output: Γ κ = {(x i , T (x i )) n i=1 } ▷ Mapped pairs.
factors specific to this q ∈ [ρ t ]:
(Q, R) ← LROT(µ X (t) q , µ Y (t) q , g t+1 ) .(9)
For each q ∈ [ρ t ] we use the Q, R from (9) to co-cluster X
q with Y
q using r t+1 labels. Within this pair, each
x i ∈ X (t)
q is assigned a label z ∈ [r t+1 ] by taking the argmax over the i-th row of Q, and likewise each y j ∈ Y (t) q is assigned the argmax over the j-th row of R. This corresponds to the Assign step in Algorithm 1, and coincides with the hard assignment of q ⋆ and r ⋆ for an optimal (Q * , R * ) (Lemma B.5).
The uniform constraint g = 1 rt+1 /r t+1 in (7) enforces an even split of the dataset, which by Lemma B.5 ensures a partition at optimality (for r t = 2). Repeating for all q ∈ [ρ t ], one obtains a co-clustering with r t+1 components within each co-cluster at the previous scale, leading to a total of ρ t+1 = r t+1 ρ t co-clusters at scale t + 1 (Fig. 1). If the base-case rank r base is one, Algorithm 1 returns a bijection between X and Y as a collection of n tuples.
Note that Hierarchical Refinement defines an implicit hierarchy of block-couplings at each scale t.
Definition 3.3 (Hierarchical block-coupling). For each scale t ∈ [κ], given the Hierarchical Refinement co-cluster partition Γ t , the hierarchical block-coupling at scale t is defined by the matrix
P (t) ij := ρ t n 2 ρt q=1 δ (xi,yj )∈Γt,q ,(10)
Without loss of generality, P (t) may be block diagonalized into ρ t square blocks, as discussed in Appendix B (see Equation ( S13)). By Proposition 3.1, for any rank-schedule (r j ) κ j=1 satisfying Monge separability, the final P (κ) corresponds to an optimal coupling supported on the graph of the Monge map T ⋆ , P (κ) := (id × T ⋆ ) ♯ µ X . While these intermediate couplings are never instantiated, one can still use them to define a transport cost ⟨C, P (t) ⟩ at each scale. In Appendix B.8, we show the following bounds on the cost difference across scales.
Proposition 3.4. Let c(•, •) be a strictly-convex and Lipschitz cost function, let (r 1 , r 2 , • • • , r κ ) be a rank-schedule, and let P (t) denote the coupling defined in (10), obtained from step t of Algorithm 1. Define ∆ t,t+1 = ⟨C, P (t) ⟩ F -⟨C, P (t+1) ⟩ F . Then,
0 ≤ ∆ t,t+1 ≤ ∥∇c∥ ∞ 1 ρ t ρt q=1 diam Γ t,q ,(11)
where q indexes co-clusters Γ t,q at scale t, defined in (8).
Thus, the lower-bound implies that each step of refinement improves the coarse partition, and the upper-bound implies that the difference in solution value is bounded above by a factor depending on the Lipschitz constant and the mean diameter of the coarse partitions at each level t. The proof of Proposition 3.4 roughly follows that of Proposition 1 of (Gerber & Maggioni, 2017). In Remark B.9, we discuss how Proposition 3.4 compares, noting that our result makes fewer geometric assumptions on our multiscale partitions (X (t) ) κ t=1 and (Y (t) ) κ t=1 and therefore does not quantify the rate of decay of diam Γ t,q .
this section cite: ['b65', 'b32', 'b29']

Section: On the Rank-Annealing Schedule
As observed by (Forrow et al., 2019;Scetbon et al., 2021), rank behaves like a temperature parameter, inverse to the strength ϵ of entropy regularization. The correspondence between small ϵ and large rank implies that annealing in the parameter ϵ is, from the perspective of rank, analogous to initializing the optimization at a low-rank coupling, and then gradually increasing the rank constraint from low to full. In Hierarchical Refinement, this gradual rank increase is accomplished implicitly. At each scale t = 1, . . . , κ the implicit coupling P (t) is made explicit in the hierarchical block coupling defined in equation ( 10). A rank-annealing schedule (r 1 , . . . , r κ ) describes the sequence of multiplicative factors by which the rank of this explicit coupling will increase at successive scales. The partial products of these, denoted (ρ 1 , . . . , ρ κ ), are the ranks of the couplings P (1) , . . . , P ( κ) . Note that small values of r i generate coarse partitions of the points at the next scale, while large values of r i generate finer partitions at the next scale.
We now turn to the question of how to efficiently choose such a schedule under given memory constraints. For an integer n, Algorithm 1 has log-linear complexity for depth κ = log r n (Section 3.4). However, the large constants required by low-rank OT in practice encourage minimizing the number of calls to LROT as a subroutine, so that if memory permits, it may be advantageous to decrease the depth by storing couplings of higher rank. If desired, memory constraints can be enforced by imposing a maximum rank r max ≥ r t for all t ∈ [κ] to ensure Hierarchical Refinement only requires O(nr max ) space at each step. Thus, we seek factorizations with minimal partial sums of ranks while remaining below a desired memory-capacity:
min (ri) κ i=1 κ j=1 ρ j s.t. ρ κ = n, r i ≤ r max .(12)
The above optimization assumes a base-rank r base of 1; we describe how to handle the general case in Appendix E.1. Importantly, the recursive structure min (ri 12), storing a table of factors up to r max to optimize (12) in O(r max κn) time. Assuming κ, r max are small constants chosen to ensure that all matrices can fit within memory, determining the optimal rank-schedule with respect to κ, n, r max is a simple lineartime procedure.
) κ i=1 κ j=1 ρ j = min (ri) κ i=1 r 1 + r 1 κ j=2 j i=2 r i enables a dynamic programming approach to (
this section cite: ['b26', 'b65']

Section: Complexity and Scaling of Hierarchical Refinement
For two datasets X, Y of size n, the space complexity of Hierarchical Refinement is Θ(n), since at each level, one must store Γ t which is a set of subsets of X and Y. To derive the time-complexity of Hierarchical Refinement, note that if n = r k , a rank-r schedule at each layer requires n r instances of LROT over rapidly decaying dataset sizes. The complexity of low-rank OT (Scetbon et al., 2021;2022;Halmos et al., 2024) is linear (Kn) for a constant K = O(BLrd) dependent on B the number of inner Sinkhorn (Halmos et al., 2024) or Dykstra (Scetbon et al., 2021) iterations, L the number of mirror-descent steps, r the rank of the coupling, and d the rank of the factorization of the cost matrix C. In this setting, for n a power of r, the runtime of Algorithm 1 is given by the sum r
0 Θ(n) + r 1 Θ( n r ) + ... + r i-1 Θ( n r i-1 ) = Θ(ndr log r n) for i = log r n, achieving linear space with log-linear time for constant ranks r, d.
In cases where the cost matrix does not admit a low-rank factorization C = UV ⊤ , i.e., when d = O(n), one requires Θ(n 2 ) space to store the cost matrix and Hierarchical Refinement exhibits time complexity Õ(n 2 ), as in Sinkhorn.
For kernel costs such as squared Euclidean cost, as noted in (Scetbon et al., 2021), one may efficiently compute a (d + 2) dimensional factorization where d is the ambient dimension, to achieve log-linear scaling with exact distances. We also use the sample-linear algorithm of (Indyk et al., 2019) to compute approximate factorizations for distances c(•, •) satisfying metric properties such as the triangle inequality (e.g. Euclidean distance, see Appendix E.1). At each level, pairing such sample-linear approximations with each lowrank step only requires O(n log d n) time. We observe this scaling empirically, as reported in Fig. S2.
this section cite: ['b65', 'b32', 'b32', 'b65', 'b65', 'b38']

Section: Experiments
We benchmark Hierarchical Refinement (HiRef) against the full-rank OT methods Sinkhorn (Cuturi, 2013), ProgOT (Kassraie et al., 2024), and mini-batch OT (Genevay et al., 2018;Fatras et al., 2020;2021b). We additionally benchmark against the low-rank OT methods LOT (Scetbon et al., 2021) and FRLC (Halmos et al., 2024). We use the default implementations of Sinkhorn, ProgOT, and LOT in the high-performance ott-jax library (Cuturi et al., 2022). In particular, Sinkhorn is run with the default entropy regularization parameter of ϵ = 0.05. We also benchmark against the multiscale method MOP (Gerber & Maggioni, 2017), which requires multiscale partitions of the input datasetsakin to a family of dyadic cubes across scales -to compute alignments. This leads to a transport cost that depends on the choice of this partition. For simplicity, we choose the default partitions of MOP which are computed from the GMRA (Geometric Multi-Resolution Analysis) R package.
Hierarchical-Refinement Map Sinkhorn Barycentric Map Optimal Map (Dual Revised Simplex) a. b. Checkerboard dataset (Makkuva et al., 2020).
this section cite: ['b14', 'b40', 'b28', 'b22', 'b65', 'b32', 'b15', 'b29', 'b51']

Section: Evaluation on Synthetic Datasets.
We first evaluate the performance of Hierarchical Refinement against optimal transport methods returning primal couplings, namely Sinkhorn (Cuturi, 2013) (as implemented in ott-jax (Cuturi et al., 2022)) and ProgOT (Kassraie et al., 2024). We evaluate the methods with respect to the Wasserstein-1 and Wasserstein-2 distance on an alignment of 1024 pairs of samples on the Checkerboard (Makkuva et al., 2020), MAFMoons and Rings (Buzun et al., 2024), and Half-Moon and S-Curve (Buzun et al., 2024) synthetic datasets (Fig. 3, Table S6).
All methods are similarly effective at minimizing the primal OT cost ⟨C, P⟩ F , with small absolute difference in cost between the final couplings. Hierarchical Refinement achieves slightly lower primal cost on 4 out of the 6 evaluations. Notably, there is a massive difference in the number of non-zero entries (defined as entries P ij > 10 -8 ) in the couplings output by HiRef, Sinkhorn, and ProgOT (Table S3). Specifically, across the experiments HiRef outputs a bijection with exactly 1024 non-zero elements in the coupling matrix, equal to the number of aligned samples. In constrast, Sinkhorn and ProgOT output couplings with 624733 to 678720 and 271087 to 337258 non-zero entries.
We evaluate the scalability of Hierarchical Refinement relative to other full-rank solvers on varying numbers of samples from the Half Moon & S-Curve (Buzun et al., 2024) synthetic dataset. We vary the rank from 2 5 = 32 (64 points aligned) up to 2 20 = 1048576 points (2097152 points aligned) in R 2 , the latter dataset of a size that is beyond the capabilities of current optimal transport solvers. We observe that Sinkhorn (Cuturi, 2013) and ProgOT -methods which produce dense mappings -require a coupling matrix with O(n 2 ) non-zero entries and thus run only up to 16384 points. HiRef yields solutions with comparable primal cost to ProgOT and Sinkhorn on the sample sizes where all methods run.
We also find that HiRef achieves an OT cost that is competitive with the dual revised simplex solver (Huangfu & Hall, 2018), a solver which only scales up to 512 points (Table S4). This solver computes an optimal coupling, unlike ProgOT and Sinkhorn which rely on entropic regularization. While we benchmark Sinkhorn in place of mini-batch OT on the synthetic datasets due to their limited complexity, we also evaluate the multi-scale method MOP on the 512 point instance (Table S4). Although MOP outputs a fast approximation to optimal transport, its primal cost on the Checkerboard (Makkuva et al., 2020) dataset is twice as high as that of the other methods, and it performs significantly worse on the MAF Moons & Rings and Half Moon & S-Curve datasets (Buzun et al., 2024).
Lastly, we observe that Hierarchical Refinement scales to over a million points, two orders of magnitude greater than ProgOT and Sinkhorn, two full-rank OT methods that compute global alignments. We find HiRef scales linearly with the size of the problem instance (Fig. S2a) in contrast to the quadratic scaling in time complexity of Sinkhorn (Fig. S2b).
this section cite: ['b14', 'b15', 'b40', 'b51', 'b6', 'b6', 'b6', 'b14', 'b36', 'b51', 'b6']

Section: Large-scale Matching Problems and Transcriptomics
Recently, optimal transport has been applied to single-cell and spatial transcriptomics datasets to compute couplings between cells taken from different timepoints from developmental processes or perturbations (Schiebinger et al., 2019;Lavenant et al., 2024;Bunne et al., 2022;Huizing et al., 2024;Halmos et al., 2025b;Klein et al., 2025). However, the size of current datasets (Chen et al., 2022) (>100k cells) has exceeded the capacity of existing full-rank solvers, requiring low-rank approximations of the coupling (Scetbon et al., 2023;Klein et al., 2025;Halmos et al., 2025a) to produce alignments.
We evaluate whether the full-rank solver of Hierarchical Refinement exhibits competitive alignments for such datasets. Specifically, we analyze the mouse organogenesis spatiotemporal transcriptomic atlas (MOSTA) datasets, which include spatial transcriptomics data from mouse embryos at successive 1-day time-intervals with increasing number n of cells at each stage: E9.5 (n = 5913), E10.5 (n = 18408), E11.5 (n = 30124), E12.5 (n = 51365), E13.5 (n = 77369), E14.5 (n = 102519), E15.5 (n = 113350), and E16.5 (n = 121767). For the cost we use the Euclidean distance
C ij = ∥x i -y j ∥ 2 in 60-dimensional PCA space of expres- sion vectors, so x i , y j ∈ R 60 .
Sinkhorn and ProgOT are unable to produce alignments for the stages beyond E10.5 (n = 18408 cells), whereas HiRef, the low-rank solvers, and mini-batch OT (batchsizes B = 128 to B = 2048) are able to continue scaling to > 10 5 (Table 1, Table S6). We observe that the Kantorovich cost of HiRef is consistently lower than all other methods for all timepoints (Table 1, Table S6).
HiRef achieves a substantially lower cost than the lowrank solvers FRLC and LOT for rank r = 40, even though HiRef relies on low-rank optimal transport (FRLC) as a subroutine. This result underscores the empirical trend observed in Fig. S3, where the refinement step of HiRef progressively decreases the primal cost of coarser low-rank couplings (Proposition 3.4). While the mini-batch solvers exhibit competitive scaling up to the last pair, the primal cost of mini-batch is higher for all tested batch-sizes (Table S6).
Unlike HiRef, mini-batch OT does not compute a global alignment and exhibits batch-size dependent error.
this section cite: ['b68', 'b47', 'b4', 'b37', 'b42', 'b7', 'b67', 'b42']

Section: MERFISH Brain Atlas Alignment
We ran HiRef on two slices of MERFISH Mouse Brain Receptor Map data from Vizgen to test whether HiRef can produce biologically valid alignments using the only spatial densities of each tissue. These spatial transcriptomics data consist of spatial and gene expression measurements at individual spots in three full coronal slices across three bio-logical replicates. Our "source" dataset (X 1 , S 1 ) is replicate 3 of slice 2, while our "target" dataset (X 2 , S 2 ) is replicate 2 of slice 2, following the expression transfer task described (Clifton et al., 2023) between these two slices. Each dataset has roughly 84k spots, where memory constraints prohibit instantiation a full-rank alignment as a matrix. Thus, solvers such as Sinkhorn (Cuturi, 2013) and ProgOT (Kassraie et al., 2024) are unable to run on the dataset.
We use only spatial information when building a map between the two slices, using the spatial Euclidean cost C ij := ∥s 1 i -s 2 j ∥ 2 , after registering spatial coordinates S 1 = {s 1 i } n i=1 and S 2 = {s 2 i } n i=1 with an affine transformation. We gauged the quality of the HiRef alignment (Fig. 4a), using gene expression abundances of five "spatially-varying" genes. Specifically, we observe that expression vector v 1 of gene Slc17a7 in the source slice ( Fig. 4b) when transferred to target slice through the bijective mapping output by HiRef, denoted as v (Fig. 4c), closely matches the observed expression vector v 2 of Slc17a7 in the target slice (Fig. 4d) with cosine similarity equal to 0.8098. For genes Slc17a7, Grm4, Olig1, Gad1, Peg10, the corresponding cosine similarities between the transferred and observed expression vectors are 0.8098, 0.7959, 0.7526, 0.4932, 0.6015, respectively.
For comparison, we also ran the low-rank methods FRLC (Halmos et al., 2024) and LOT (Scetbon et al., 2021) with and without subsampling, reporting their best scores, as discussed in Section D.3. For the gene Slc17a7, FRLC's cosine similarity was 0.2373, while LOT's cosine similarity was 0.3390. For all five genes Slc17a7, Grm4, Olig1, Gad1, Peg10, FRLC's scores were (0.2373, 0.2124, 0.1929, 0.0963, 0.1550, respectively, while LOT's scores were 0.3390, 0.2712, 0.3186, 0.1666, 0.1080. Across all five genes HiRef's scores were at least twice those of FRLC or LOT (Table S7) with gene abundances shown in Fig. S1. On the same task, we compared against MOP, the method of (Gerber & Maggioni, 2017), whose scores for the five genes were: (0.5211, 0.4714, 0.5972, 0.3571, 0.2719). Finally, we also benchmarked against mini-batch OT using batch sizes ranging from 128 to 2048 in powers of two, whose best scores (0.7434, 0.7822, 0.7056, 0.4912, 0.5683) were more comparable to that of the performance of HiRef. Across all methods and genes compared in Table S7, HiRef had greatest cosine similarity scores in the expression transfer task, while also having lowest transport cost. Further experimental details are in Section D.3.
this section cite: ['b11', 'b14', 'b40', 'b32', 'b65', 'b29']

Section: ImageNet Alignment
We demonstrate the scalability of Hierarchical Refinement on a large-scale and high-dimensional dataset by aligning 2048-dimensional embeddings of 1.281 million images from the ImageNet ILSVRC dataset (Deng et al., 2009;
a. b. c.    Russakovsky et al., 2015). Each image is embedded using using the ResNet50 architecture (He et al., 2016), and we construct two datasets, X and Y, by taking a random 50:50 split of the embedded images. We align X and Y using HiRef, FRLC, and mini-batch OT with batch-sizes ranging from B = 128 to B = 1024. ProgOT, Sinkhorn, and LOT could not be run on the datasets due to memory constraints. HiRef yielded a primal OT cost of 18.974, while FRLC (Halmos et al., 2024) solution had a primal OT cost of 24.119 for rank r = 40 and mini-batch OT has costs of 21.89 (B = 128) to 19.58 (B = 1024) (Table 2).
this section cite: ['b19', 'b62', 'b35', 'b32']

Section: Discussion
Hierarchical Refinement computes the Monge map between large-scale datasets in linear space, but has several limitations. First, we currently assume that the datasets X and Y have the same number of samples. In many machine learning applications, this is not a limiting factor, as one generally seeks to pair an equal number of source points x to target points y. Second, while Hierarchical Refinement scales linearly in space and log-linearly in time, it still involves a constant dependent on the low-rank OT subprocedure used -this underscores the need to accelerate and stabilize low-rank OT solvers further (Scetbon & Cuturi, 2022;Halmos et al., 2024). Finally, while Hierarchical Refinement guarantees an optimal solution given an optimal black-box low-rank solver (Proposition 3.1), the low-rank solvers (Scetbon et al., 2022;Halmos et al., 2024) used in practice are not necessarily optimal, owing to the nonconvexity of low-rank problems.
Optimal transport has been successfully applied in deep learning frameworks, such as OT flow-matching (Tong et al., 2024), computer vision and point cloud registration, (Yu et al., 2021;Qin et al., 2022), among many others. The mini-batch procedure used to train many of these methods involves sampling two datasets X B ∼ µ and Y B ∼ ν with batch-size B and aligning them with Sinkhorn at every training iteration. HiRef suggests an alternative approach: one can precompute millions of globally aligned pairs and then sample X B ∼ µ and the optimal mapping T (X B ) ∼ ν by indexing into these precomputed pairs. This approach applies to any loss function dependent on an OT alignment.
Hierarchical Refinement may also be useful in neural OT approaches which learn a continuous Monge map between the densities of two datasets. For example, (Seguy et al., 2018) minimize a loss min θ 1 2 E µ ∥T θ (x i ) -T (x i )∥ 2 2 between a neural network T θ with parameters θ and a Monge map T over samples x i ∼ µ (Remark B.11). Thus, the procedure outlined above may be used to directly regress a neural network T θ on the Monge map T without the bias of mini-batching or entropy.
this section cite: ['b32', 'b32', 'b78', 'b81', 'b60', 'b71']

Section: Conclusion
We introduce Hierarchical Refinement (HiRef), an algorithm to solve optimal transport with linear complexity in the number of points, making sparse, full-rank optimal transport feasible for large-scale datasets. Our algorithm leverages that low-rank optimal transport co-clusters points with their image under the Monge map, refining bijections between partitions of each dataset across a hierarchy of scales, down to a bijective Monge map between the datasets at the finest scale. Hierarchical Refinement couplings achieve comparable primal cost to couplings obtained through full-rank entropic solvers, and scales to datasets with over a million points, opening the door to applications previously infeasible for optimal transport.
this section cite: []

Section: References
Ref_id:b0 Title: Minkowski-type theorems and least-squares clustering Year: (1998)
Ref_id:b1 Title: Tres observaciones sobre el algebra lineal Year: (1946)
Ref_id:b2 Title: Sliced and Radon Wasserstein barycenters of measures Year: (2015)
Ref_id:b3 Title: Polar factorization and monotone rearrangement of vector-valued functions Year: (1991)
Ref_id:b4 Title: Proximal optimal transport modeling of population dynamics Year: (2022)
Ref_id:b5 Title: Learning single-cell perturbation responses using neural optimal transport Year: (2023)
Ref_id:b6 Title: Expectile regularization for fast and accurate training of neural optimal transport Year: (2024)
Ref_id:b7 Title: Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays Year: (2022)
Ref_id:b8 Title: Exponential convergence of Sinkhorn under regularization scheduling Year: (2023)
Ref_id:b9 Title: Condition number-free query and active learning of linear families Year: (2017)
Ref_id:b10 Title: Optimal transport for Gaussian mixture models Year: (2018)
Ref_id:b11 Title: Alignment of spatial transcriptomics data using diffeomorphic metric mapping Year: (2023)
Ref_id:b12 Title: Nonnegative ranks, decompositions, and factorizations of nonnegative matrices Year: (1993)
Ref_id:b13 Title: Domain adaptation with regularized optimal transport Year: (2014)
Ref_id:b14 Title: Sinkhorn distances: Lightspeed computation of optimal transport Year: (2013)
Ref_id:b15 Title: Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein Year: (2022)
Ref_id:b16 Title: Schrödinger bridge flow for unpaired data translation Year: (2024)
Ref_id:b17 Title: Combinatorics and geometry of transportation polytopes: An update. Discrete Geometry and Algebraic Combinatorics Year: (2013)
Ref_id:b18 Title: A Wasserstein-type distance in the space of Gaussian mixture models Year: (2020)
Ref_id:b19 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b20 Title: Hierarchical optimal transport for unsupervised domain adaptation Year: (2022)
Ref_id:b21 Title: Neural Monge map estimation and its applications Year: (2023)
Ref_id:b22 Title: Learning with minibatch Wasserstein: asymptotic and gradient properties Year: (2020)
Ref_id:b23 Title: Unbalanced minibatch optimal transport; applications to domain adaptation Year: ()
Ref_id:b24 Title: Minibatch optimal transport distances; analysis and applications Year: (2021)
Ref_id:b25 Title: How to train your neural ODE: the world of Jacobian and kinetic regularization Year: (2020)
Ref_id:b26 Title: Statistical optimal transport via factored couplings Year: (2019)
Ref_id:b27 Title: Fast Monte-Carlo Algorithms for Finding Low-rank Approximations Year: (2004-11)
Ref_id:b28 Title: Learning generative models with Sinkhorn divergences Year: (2018)
Ref_id:b29 Title: Multiscale strategies for computing optimal transport Year: (2017)
Ref_id:b30 Title: A mathematical perspective on Transformers Year: (2023)
Ref_id:b31 Title: Iterative scheme for solving optimal transportation problems arising in reflector design Year: (2013)
Ref_id:b32 Title: Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling Year: (2024)
Ref_id:b33 Title: Learning latent trajectories in developmental time series with Hidden-Markov optimal transport Year: (2025)
Ref_id:b34 Title: DeST-OT: Alignment of spatiotemporal transcriptomics data Year: ()
Ref_id:b35 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b36 Title: Parallelizing the dual revised simplex method Year: (2018)
Ref_id:b37 Title: Learning cell fate landscapes from spatial transcriptomics using Fused Gromov-Wasserstein Year: (2024)
Ref_id:b38 Title: Sample-optimal low-rank approximation of distance matrices Year: (2019)
Ref_id:b39 Title: On the translocation of masses Year: (1942)
Ref_id:b40 Title: Progressive entropic optimal transport solvers Year: (2024)
Ref_id:b41 Title: Generative entropic neural optimal transport to map within and across space Year: (2024)
Ref_id:b42 Title: Mapping cells through time and space with moscot Year: (2025)
Ref_id:b43 Title: Optimal flow matching: Learning straight trajectories in just one step Year: (2024)
Ref_id:b44 Title: Do neural optimal transport solvers work? A continuous Wasserstein-2 benchmark Year: (2021)
Ref_id:b45 Title: Neural optimal transport. International Conference on Learning Representations Year: (2023)
Ref_id:b46 Title: The Hungarian method for the assignment problem Year: (1955)
Ref_id:b47 Title: Toward a mathematical theory of trajectory inference Year: (2024)
Ref_id:b48 Title: Interpreting generative language models via optimal transport. International Conference on Machine Learning Year: (2024)
Ref_id:b49 Title: Making transport more robust and interpretable by moving data through a small number of anchor points Year: (2021)
Ref_id:b50 Title: Improved complexity analysis of the sinkhorn and greenkhorn algorithms for optimal transport Year: (2023)
Ref_id:b51 Title: Optimal transport mapping via input convex neural networks Year: (2020)
Ref_id:b52 Title: Distributional preference alignment of LLMs via optimal transport Year: (2024)
Ref_id:b53 Title: A multiscale approach to optimal transport Year: (2011)
Ref_id:b54 Title: Mémoire sur la théorie des déblais et des remblais Year: (1781)
Ref_id:b55 Title: Improving mini-batch optimal transport via partial transportation Year: (2022)
Ref_id:b56 Title: Hierarchical sliced Wasserstein distance. International Conference on Learning Representations Year: (2022)
Ref_id:b57 Title: An efficient linear programming method for optimal transportation Year: (2015)
Ref_id:b58 Title: A polynomial time primal network simplex algorithm for minimum cost flows Year: (1997)
Ref_id:b59 Title: Computational optimal transport: With applications to data science Year: (2019)
Ref_id:b60 Title: Geometric transformer for fast and robust point cloud registration Year: (2022)
Ref_id:b61 Title: Zero-shot text-toimage generation Year: (2021)
Ref_id:b62 Title: ImageNet large scale visual recognition challenge Year: (2015)
Ref_id:b63 Title: Sinkformers: Transformers with doubly stochastic attention Year: (2022)
Ref_id:b64 Title: Low-rank optimal transport: Approximation, statistics and debiasing Year: (2022)
Ref_id:b65 Title: Low-rank Sinkhorn factorization Year: (2021)
Ref_id:b66 Title: Linear-time Gromov Wasserstein distances using low rank couplings and costs Year: (2022)
Ref_id:b67 Title: Unbalanced low-rank optimal transport solvers Year: (2023)
Ref_id:b68 Title: Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming Year: (2019)
Ref_id:b69 Title: A sparse multiscale algorithm for dense optimal transport Year: (2016)
Ref_id:b70 Title: A hierarchical approach to optimal transport Year: (2013)
Ref_id:b71 Title: Large-scale optimal transport and mapping estimation Year: (2018)
Ref_id:b72 Title: Convolutional Wasserstein distances: Efficient optimal transportation on geometric domains Year: (2015)
Ref_id:b73 Title: Optimal transport: Fast probabilistic approximation with exact solvers Year: (2019)
Ref_id:b74 Title: Visualization and analysis of gene expression in tissue sections by spatial transcriptomics Year: (2016)
Ref_id:b75 Title: Dynamic trees as search trees via Euler tours, applied to the network simplex algorithm Year: (1997)
Ref_id:b76 Title: Sparse Sinkhorn attention. International Conference on Machine Learning Year: (2020)
Ref_id:b77 Title: Introduction to optimal transport. Notes of Course at University of Cambridge Year: (2018)
Ref_id:b78 Title: Improving and generalizing flow-based generative models with minibatch optimal transport Year: (2024)
Ref_id:b79 Title: Largescale single-cell gene expression data analysis Year: (2018)
Ref_id:b80 Title: Predicting cell lineages using autoencoders and optimal transport Year: (2020)
Ref_id:b81 Title: Reliable coarse-to-fine correspondences for robust pointcloud registration Year: (2021)
Ref_id:b82 Title: Hierarchical optimal transport for document representation Year: (2019)
Ref_id:b83 Title: Alignment and integration of spatial transcriptomics data Year: (2022)
