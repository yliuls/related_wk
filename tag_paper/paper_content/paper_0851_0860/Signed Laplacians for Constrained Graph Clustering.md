Title: Signed Laplacians for Constrained Graph Clustering
Abstract: Given two weighted graphs G = (V, E, w G ) and H = (V, F, w H ) defined on the same vertex set, the constrained clustering problem seeks to find a subset S ⊂ V that minimises the cut ratio between w G (S, V \ S) and w H (S, V \ S). In this work, we establish a Cheeger-type inequality that relates the solution of the constrained clustering problem to the spectral properties of G and H. To reduce computational complexity, we utilise the signed Laplacian of H, streamlining calculations while maintaining accuracy. By solving a generalised eigenvalue problem, our proposed algorithm achieves notable performance improvements, particularly in challenging scenarios where traditional spectral clustering methods struggle. We demonstrate its practical effectiveness through experiments on both synthetic and real-world datasets.

Section: Introduction
Clustering is a fundamental technique in machine learning, with extensive applications across computer science and various scientific disciplines. The primary goal of clustering is to partition data points into clusters, such that points within each cluster are more densely connected than those in other clusters. Traditional clustering, such as spectral clustering (von Luxburg, 2007) relies solely on the structure of the data. However, in many real-world scenarios, additional domain knowledge is available, introducing specific constraints that should be incorporated into the clustering process to achieve more accurate and meaningful results (Basu et al., 2008).
Constrained clustering focuses on developing algorithms that effectively incorporate this domain knowledge to enhance clustering performance (Wagstaff et al., 2001). The domain knowledge is represented by two types of pairwise constraints: (1) MUST-LINK constraints, requiring that a pair of data points must be assigned to the same cluster, and (2) CANNOT-LINK constraints, requiring that a pair of data points must be assigned to different clusters. In the context of graph clustering, the goal is to partition the vertices of a graph based on edge connectivity while satisfying the given constraints.
In this paper, we examine the constrained clustering problem by formalising these constraints using the graphs G = (V, E, w) and H = (V, E ′ , w ′ ), in which every data point corresponds to a graph vertex, every MUST-LINK (resp. CANNOT-LINK) constraint corresponds to an edge in G (resp. H), and the edge weights capture the strength of the user's preference for satisfying the corresponding constraint. For any set S ⊆ V , we define the cut ratio of S between G and H by
cut G H (S, V \ S) = w G (S, V \ S) w H (S, V \ S) ,(1)
and the objective is to find S that achieves
Φ G H = min ∅⊂S⊂V cut G H (S, V \ S).
We develop an efficient approximation algorithm for the constrained graph clustering problem. The key to our algorithm is a Cheeger-type inequality that upper bounds Φ G H with respect to λ 2 (∆ G H ) and λ 2 (∆ H ), where
λ 2 ∆ G H = min x⊥1 ⟨x, ∆ G x⟩ ⟨x, ∆ H x⟩ ,(2)
λ 2 ∆ H = min x⊥1 ⟨x, ∆ H x⟩ ⟨x, x⟩ .(3)
and ∆ G and ∆ H are the Laplacian operators of G and H respectively. By introducing several techniques to adjust G and H, we significantly reduce the time complexity for solving the generalised eigenvalue problem, while maintaining the one-to-one correspondence between the solution of the new reduced instance and the initial one. The empirical studies on both the synthetic and real-world data sets confirm that with the two sets of constraints our algorithm presents significantly better performance than the classical spectral clustering algorithm, and the running time of our algorithm is close to traditional spectral clustering methods.
Related work. Cheeger-type inequalities for constrained graph clustering are studied in the literature. For example, Cucuringu et al. (2016) proved that
Φ G H • Φ G K ≤ 4λ 2 (∆ G H )
; this inequality is based on a third graph K, which they call the demand graph. Koutis et al. (2023) showed that Φ G H ≤ 16λ 2 (∆ G H )/Φ(G), where Φ(G) is the standard conductance of G. These two results cannot be directly compared with ours, since both inequalities upper bound Φ G H with respect to λ 2 (∆ G H ) and parameters of H, i.e., Φ G K in (Cucuringu et al., 2016) and Φ(G) in (Koutis et al., 2023). In contrast, we upper bound Φ G H with respect to λ 2 (∆ G H ) and λ 2 (∆ H ). Trevisan (2013) studied the computational complexity of the problem and proved that under the Unique Games Conjecture, it's impossible to find a cut that achieves
O Φ G H approximation in polynomial time.
Our work also relates to the studies on constrained graph clustering from practical perspectives (Jia et al., 2021;Wang & Davidson, 2010;Wang et al., 2014) and signed cuts using the signed Laplacian (Knyazev, 2017). Most of these studies, however, lack the rigorous analysis of the quality of the resulting clusters compared to the optimal solution. Our work is further linked to Cheeger-type inequalities for different graph Laplacians (Lange et al., 2015;Li et al., 2019) including the signed Laplacian (Atay & Liu, 2020), and their higher-order generalisations (Lee et al., 2014).
Contribution. We propose a novel method for constrained graph clustering that establishes a Cheeger-type inequality explicitly linking the cut ratio objective to the spectral properties of two graphs G and H. Our approach introduces a signed Laplacian-based implementation, which avoids additional parameters while improving both numerical stability and computational efficiency. Finally, our experiments on synthetic and real-world datasets confirm the robustness and scalability of our algorithm, significantly outperforming standard spectral clustering under challenging scenarios.
this section cite: ['b17', 'b2', 'b18', 'b5', 'b10', 'b5', 'b10', 'b16', 'b8', 'b19', 'b20', 'b9', 'b11', 'b13', 'b0', 'b12']

Section: Background & Preliminaries
We consider a finite, undirected graph G = (V, E), where V is the set of vertices and E is the set of edges. Each edge uv ∈ E denotes an undirected connection between vertices u and v. A self-loop in this context is represented by uu, indicating an edge that starts and ends at the same vertex u. The notation u ∼ v means that u and v are connected by an edge. We define weights in the graph G through the function w : E → R + , where w uv = w vu specifies the weight of the edge between vertices u and v. By definition, each self-loop uu contributes twice to the degree of vertex u.
For E 0 ⊆ E, we interpret w as a discrete measure on the corresponding sets, using the notation w(E 0 ) = uv∈E0 w uv .
For V 0 , V 1 ⊆ V , we denote the set of unoriented edges between V 0 and V 1 as E(V 0 , V 1 ) = {uv ∈ E | u ∈ V 0 and v ∈ V 1 }.
We denote by E v the set of all edges connected to v, and N v the neighbourhood of v as the set of vertices adjacent to v, i.e.,
E v = E({v}, V ) and N v = {u ∈ V | v ∼ u}.
The edge weights on a graph determine a weighted degree of a vertex, defined by deg
(v) = w(E v ) = e∈Ev w e .
The Laplacian. We define some standard spaces related to a finite and weighted graph G = (V, E) with weight function w. We define the Hilbert spaces ℓ 2 (V, w) and ℓ 2 (E, w) as
ℓ 2 (V, w) = {φ : V → R}, ℓ 2 (E, w) = {η : E → R}.
Furthermore, we consider the natural inner product for these spaces. For ℓ 2 (V, w), the inner product between functions φ and ψ is defined as ⟨φ,
ψ⟩ V = v∈V φ(v)ψ(v) deg(v).
For ℓ 2 (E, w), the inner product between functions η and ξ is defined as ⟨η,
ξ⟩ E = e∈E η e ξ e w e . Let G = (V, W, w)
be a weighted graph. The derivative d is defined as
d : ℓ 2 (V, w) -→ ℓ 2 (E, w), (dφ) e=(u,v) = φ(u) -φ(v).
The adjoint
d * : ℓ 2 (E, w) -→ ℓ 2 (V, w) is given by (d * η)(v) = - 1 deg(v) e∈Ev w e η e .
The weighted Laplacian ∆ : ℓ 2 (V, w) -→ ℓ 2 (V, w) is defined as ∆ = d * d, and acts as
(∆φ)(v) = 1 deg(v) u∈Nv φ(v) -φ(u) w uv .
Let G = (V, E, w) be a weighted graph, and for all φ ∈ ℓ 2 (V, w) we have that
⟨φ, ∆ G φ⟩ = ⟨φ, d * dφ⟩ ℓ2(V ) = ⟨dφ, dφ⟩ ℓ2(E) = u∼v |dφ uv | 2 w uv = u∼v (φ(u) -φ(v)) 2 w uv . Graph Signature. A signature of a graph G = (V, E) is a map α : E → {+1, -1}
, which assigns a sign to each edge. Let G = (V, E, w) be a weighted graph with a signature α. The signed Laplacian, denoted as ∆ α , is a linear operator
∆ α : ℓ 2 (V, w) → ℓ 2 (V, w), defined by (∆ α φ)(v) = 1 deg(v) u∈Nv φ(v) -α vu φ(u) w vu ,
where w uv is the weight of the edge uv, and α uv indicates the sign of the edge as given by the signature α. Observe that the Laplacian is a particular case of the signed Laplacian by taking α uv = 1 for all edges and the signless Laplacian by taking α uv = -1 for all edges. The signed Laplacian can be viewed as a special case of the magnetic Laplacian with a discrete magnetic potential taking values in {0, π}. The operators ∆ α (and therefore ∆) are positive, semi-definite, and self-adjoint, hence all of their eigenvalues are real and non-negative.
this section cite: []

Section: Algorithm & Analysis
In this section, we present a constrained graph clustering algorithm called CC++. At a high level, our algorithm consists of the following: in the preprocessing step, we adjust the edge weights of G and add self-loops to the vertices of G, such that both of G and H have the same degree sequence. Then, we show that a desired cut can be found by a sweep-set algorithm when the eigenvector corresponding to a generalised eigenvalue problem is given as input. Taking the practical implementation into account, we introduce a negative self-loop in H for efficient computation, and justify its performance both in theory and in practice. Due to page limitations, proofs omitted from this section can be found in the appendix.
this section cite: []

Section: Preprocessing G and H
In the preprocessing step, our algorithm first scales the weights of all the edges in G by the same factor, and adds self-loops to the resulting graph G. These two operations ensure that the constructed graph G and H have the same degree sequence, while maintaining the optimal cut of the input instance.
Scaling the graph G. For any c ∈ R + , we scale the edge weights of G by a factor of c and define G(c) = (V, E, c•w), where (c • w) e = c • w e for each edge e. By (1), we have that
cut G(c) H (S, V \ S) = w G(c) (S, V \ S) w H (S, V \ S) = c • w G (S, V \ S) w H (S, V \ S) = c • cut G H (S, V \ S).
Thus, the minimum cut problem for the scaled graph can be expressed as
Φ G(c) H = c • Φ G H .
This equivalence indicates that choosing an appropriate scaling factor c is crucial for balancing the edge weights between G and H. To ensure that the degrees of the vertices in the scaled graph G(c 0 ) do not exceed those in H, we define the scaling factor c 0 by
c 0 = min v∈V deg H (v) deg G (v) .(4)
This choice of c 0 guarantees that for the scaled graph G(c 0 ), the degree of each vertex v ∈ V satisfies that deg G(c0) (v) ≤ deg H (v) for all v ∈ V . With this scaling factor c 0 established, we now proceed to study the properties of the graph G = (V, E, w) and its corresponding minimum cut value
Φ G H in comparison to H = (V, E ′ , w ′ ), under the assumption that deg G (v) ≤ deg H (v) for all v ∈ V .
Equalising the Degrees of G. To further refine this comparison, consider the subset of vertices
V 0 = v ∈ V | deg G (v) < deg H (v) .
We now construct a new graph G = (V, E, w), where E = E ∪ {(v, v)} v∈V0 , and the weight function w is defined by
w | E = w and w vv = deg H (v) -deg G (v) 2 ∀v ∈ V 0 .
Observe that the construction ensures that deg
G (v) = deg H (v) for all v ∈ V . Indeed, we have that deg G (v) = u : u∼v w uv = u : u∼v w uv + 2 (v,v) w(v, v) = deg G (v) + (deg H (v) -deg G (v)) = deg H (v).
This modification of G demonstrates that, despite the additional restriction imposed by c 0 , the problems remain equivalent. Specifically, we observe that
Φ G H = min S⊆V w G (S, V \ S) w H (S, V \ S) = min S⊆V w(S, V \ S) w H (S, V \ S) = Φ G H .
Therefore, the generalized cut problem for G and H remains equivalent to that of G and H, despite the degree adjustments made in G. The following remark will be used in our analysis. Remark 3.1. For the weighted graph G = (V, E, w), the previous G = (V, E, w) where E includes additional self-loops at some vertices, and any function
φ : V (G) = V ( G) → R,
the following holds: adding self-loops does not affect the quadratic form associated with the graph Laplacian, i.e., ⟨φ,
∆ G φ⟩ = ⟨φ, ∆ G φ⟩. Specifically, ⟨φ, ∆ G φ⟩ = u∼ G v |φ(u) -φ(v)| 2 w uv = u∼ G v |φ(u) -φ(v)| 2 w uv = ⟨φ, ∆ G φ⟩.
However, the addition of self-loops does affect the norm in the space of vertices:
⟨φ, φ⟩ ℓ2(V,w) = v∈V φ(v) 2 deg G (v) ≤ v∈V φ(v) 2 deg G (v) = ⟨φ, φ⟩ ℓ2(V, w) .
Therefore, while the first quadratic form remains unchanged, the vertex norm in the extended space is generally increased due to the additional self-loops.
this section cite: []

Section: A Cheeger-type Inequality for Constrained Clustering
Next we relate the constrained clustering problem to the generalised eigenvalue problem. Our key result is a Cheegertype inequality proving that the value of Φ G H can be upper bounded with respect to λ 2 (∆ G H ) and λ 2 (∆ H ), and the cut with the proven approximation guarantee can be found by a sweep-set algorithm. Our result is as follows:
Theorem 3.2. Let G = (V, E, w) and H = (V, E ′ , w ′ ) be graphs such that deg G (v) = deg H (v) for all v ∈ V . Then, it holds that Φ G H ≤ 4 λ 2 (∆ G H ) λ 2 (∆ H ) ,(5)
where ∆ H denotes the normalised Laplacian of the graph H, ∆ G H is the operator given by
∆ G H (x) = ⟨x, ∆ G x⟩ ⟨x, ∆ H x⟩ ,
and λ 2 is the smallest non-trivial eigenvalue of the corresponding operator defined in (2). Moreover, the cut achieving this approximation guarantee can be found with a sweepset algorithm.
Notice that we can assume that G is a connected graph.
If H has m connected components, we will work with λ m-1 (∆ H ), i.e. x ⊥ 1 C for each connected component, ensuring x ⊥ ker(∆ H ). Before presenting the proof, notice that we can assume without loss of generality that the graphs G and H have the same degree sequence due to the preprocessing step.
Proof Sketch of Theorem 3.2. At a high level, our proof is similar with the one of the Cheeger inequality for graphs. We present the proof sketch here, and the full proof can be found in Appendix A.
Step 1. Let x 1 ≤ x 2 ≤ . . . ≤ x n be the eigenvector associated with λ 2 (∆ G H ). It suffices to prove the existence of a non-empty, proper subset ∅ ⊂ S ⊂ V such that w G (S, V \ S) w H (S, V \ S) = 4 ⟨x, ∆ G x⟩ • ⟨x, x⟩ ⟨x, ∆ H x⟩ 2 . (6
)
This suffices because the above inequality implies
Φ G H ≤ 4 λ 2 (∆ G H ) ⟨x, x⟩ ⟨x, ∆ H x⟩ ≤ 4 λ 2 (∆ G H ) 1 λ 2 (∆ H ) .
Step 2. We reduce the problem to proving (6) for a scaled version of x, defined as z = cx, where z 2 n + z 2 1 = 1. The scaling ensures the invariance of the expression:
⟨z, ∆ G z⟩ ⟨z, ∆ H z⟩ ⟨z, z⟩ ⟨z, ∆ H z⟩ = c 2 ⟨x, ∆ G x⟩ c 2 ⟨x, ∆ H x⟩ c 2 ⟨x, x⟩ c 2 ⟨x, ∆ H x⟩ = ⟨x, ∆ G x⟩ ⟨x, ∆ H x⟩ ⟨x, x⟩ ⟨x, ∆ H x⟩ .
Thus, the next equation is sufficient to establish (6):
w G (S, V \ S) w H (S, V \ S) = 4 ⟨z, ∆ G z⟩ • ⟨z, z⟩ ⟨z, ∆ H z⟩ 2 .(7)
Step 3. We now consider sweep sets S t , defined as S t = {v ∈ V | z v ≤ t}. To establish (7), it suffices to prove that there exists a threshold t 0 ∈ [z 1 , z n ] and defining S = S t0 :
w G (S t0 , V \ S t0 ) w H (S t0 , V \ S t0 ) ≤ 4 ⟨z, ∆ G z⟩ ⟨z, ∆ H z⟩ • ⟨z, z⟩ ⟨z, ∆ H z⟩ .(8)
Step 4. To establish (8), we will find a probability density function over [z 1 , z n ], and prove
E (w G (S t , V \ S t )) E (w H (S t , V \ S t )) ≤ 4 ⟨z, ∆ G z⟩⟨z, z⟩ ⟨z, ∆ H z⟩ 2 . (9
)
Thus, (8) holds because the existence of a threshold t 0 is guaranteed by the next equation:
w G (S t0 , V \ S t0 ) w H (S t0 , V \ S t0 ) ≤ E (w G (S t , V \ S t )) E (w H (S t , V \ S t ))
, and this implies
P w G (S t0 , V \ S t0 ) w H (S t0 , V \ S t0 ) ≤ 4 ⟨z, ∆ G z⟩⟨z, z⟩ ⟨z, ∆ H z⟩ 2 > 0.
Step 5. To establish (9), it suffices to find a probability distribution over [z 1 , z n ] such that
E (w G (S t , V \ S t )) E (w H (S t , V \ S t )) ≤ u∼ G v |z u -z v |(|z u | + |z v |)w uv u∼ H v |zu-zv| 2 2 w uv .(10)
Using the Cauchy-Schwarz inequality, the definition of ∆, and the property deg G (v) = deg H (v), we can derive ( 9) from ( 10) as follows:
u∼ G v |z u -z v | (|z u | + |z v |) w uv u∼ H v |z u -z v | 2 2 w uv ≤ u∼ G v |z u -z v | 2 w uv u∼ G v (|z u | + |z v |) 2 w uv 1 2 ⟨z, ∆ H z⟩ ≤ 4 ⟨z, ∆ G z⟩ ⟨z, z⟩ ⟨z, ∆ H z⟩ 2 .
Thus, finding a distribution that satisfies (10) is sufficient to complete the proof.
Step 6. We define a distribution, and choose t according to the probability density function 2|t|. Specifically, the probability that a value between [a, b] is chosen is
P[t ∈ [z v , z u ]] = zu zv 2|t|dt = sgn(z u ) • z 2 u -sgn(z v ) • z 2 v . Since z 2 1 + z 2 n = 1, we have that P[t ∈ [z 1 , z n ]] = 1. Step 7.
For this distribution, and regardless of the sign of z u and z v , we have
E [w G (S t , V \ S t )] = u∼ G v P [z u ≤ t and t < z v ] w uv ≤ u∼ G v |z u -z v |(|z u | + |z v |)w uv ,and
E [w H (S t , V \ S t )] = u∼ H v P [z u ≤ t and t < z v ] w uv ≥ u∼ H v |z u -z v | 2 2 w uv .
This establishes (10) and concludes the proof. The complete details are in the appendix.
Based on the proof of Theorem 3.2, to find the cut with the guaranteed approximation, we only need to order the vertices based on the entries of the eigenvector for the generalised eigenvalue problem, and construct n sweep sets. See Algorithm 1 for the formal description of our algorithm. Remark 3.3. Theorem 3.2 can be viewed as a generalisation of the classical Cheeger inequality for graphs (Chung, 1997). Specifically, if we consider the graph H as the complete
this section cite: ['b4']

Section: Algorithm 1 The Constrained Clustering Algorithm
Input: Graph G and graph H Output: A bi-partition of the vertex sets Compute the scaling factor c 0 defined in (4). Scale all edge weights in G by multiplying them with c 0 .
for each vertex v ∈ V do if deg H (v) > deg G (v) then Add a loop with weight 1 2 (deg H (v) -deg G (v)
). end if end for Compute the Laplacians ∆ G and ∆ H for the graphs G and H. Solve the generalised eigenvalue problem
⟨f, ∆ G f ⟩ ⟨f, ∆ H f ⟩ subject to f ⊥ 1, (11
)
where f is the eigenvector that minimises the ratio. Apply a sweep-set algorithm on the eigenvector f to partition the vertices of G into two clusters.
Return: a bi-partition of the vertex set graph with w uv = 1 for all edges, then it is straightforward to show that
min ∅⊂S⊂V w G (S, V \ S) |S| • |V \ S| ≤ 4 λ 2 (∆ G ),
where λ 2 (∆ G ) is the second smallest eigenvalue of the normalised graph Laplacian of G. Similarly, if we consider the graph H = (V, E ′ , w H ) as the complete graph with self-loops where
w H uv = deg G (u) deg G (v) vol(G) , then min ∅⊆S⊆V w G (S, V \ S) min(vol(S), vol(V \ S)) ≤ min ∅⊆S⊆V vol(G) w G (S, V \ S) vol(S) vol(V \ S) ≤ 4 λ 2 (∆ G ).
this section cite: []

Section: Practical Considerations
The proof of Theorem 3.2 not only establishes the general cut bound but also provides a constructive method to find a subset S ⊆ V that is close to minimising the generalised cut problem. However, this approach can be computationally expensive, particularly because the Laplacian H is not invertible. To address this issue, we modify the graph H by adding a "negative" self-loop at any vertex, effectively making the Laplacian invertible. This modification leverages the signed Laplacian, which adjusts the operator to ensure invertibility, and the introduction of a negative selfloop has little impact on the overall results, which will be demonstrated in Section 4 through experiments.
Formally, we prove that adding a negative self-loop to H makes the signed Laplacian ∆
H ′ α invertible, ensuring λ 1 (∆ H ′ α ) > 0. Since λ 1 (∆ H ′ α ) ≤ λ 2 (∆ H ), we can replace λ 2 (∆ H ) with λ 1 (∆ H ′ α )
in Theorem 3.2, maintaining the theorem's validity.
Lemma 3.4. Let H = (V, E, w) be a weighted graph, and let H ′ = (V, E ′ , w ′ ) be another weighted graph such that E ′ = E ∪ {(v 0 , v 0 )}, where v 0 is a vertex with a self-loop. Assume that w ′ | E = w and consider the signature s = 1 for all E and s (v0,v0) = -1. Then,
0 < λ 1 (∆ H ′ α ) ≤ λ 2 (∆ H ),
where ∆ H is the Laplacian of graph H.
The proof of Lemma 3.4 shows that ⟨g, ∆ H ′ α g⟩ ≈ ⟨g, ∆ H g⟩ for a small weight in the self-loop, then we will solve (11) for the self-loop, because
⟨f, ∆ G f ⟩ ⟨f, ∆ H α f ⟩ ≈ ⟨f, ∆ G f ⟩ ⟨f, ∆ H ′ α f ⟩ .
The problem involves identifying the eigenfunction and eigenvalue of a linear operator using a Lagrangian-based framework. The Lagrangian L(φ, λ) is defined as
L(φ, λ) = ⟨∆ G φ, φ⟩ -λ(⟨∆ H ′ α φ, φ⟩ -1),
where φ is the function to be optimised, and λ is the Lagrange multiplier. To find the minimiser φ, we set the gradient of L with respect to φ to zero, i.e., ∇ φ L(φ, λ) = 0.
Expanding this condition yields that
2∆ G φ -2λ∆ H ′ α φ = 0, which simplifies to ∆ G φ = λ∆ H ′ α φ.
This formulation leads to a generalised eigenvalue problem where φ is the eigenfunction, and λ is the eigenvalue. If ∆ H ′ α is invertible, the equation can be reformulated as
(∆ H ′ α ) -1 ∆ G φ = λφ,
illustrating the relationship between the linear operators and providing a solution to the eigenvalue problem via the Lagrange multiplier method. This eigenvalue equation is crucial for extracting the optimal partitions of the graph based on the constraints encoded within ∆ H ′ α . We prove that solving the generalised eigenvalue problem for ∆ G and ∆ H ′ α produces all feasible solutions, as all eigenvalues are real and non-negative. This contrasts with the approach by Wang et al. (2014).
Lemma 3.5. Let ∆ H ′ α be the signed Laplacian of the weighted graph H ′ , and ∆ G the normalized Laplacian of the weighted graph G. The operator (∆ H ′ α ) -1 ∆ G is a positive, self-adjoint operator, with all its eigenvalues being real and non-negative.
We remark that solving (11) becomes significantly more efficient for ∆ H ′ α because it is symmetric positive definite and invertible. For dense matrices, this property allows for the use of the Cholesky decomposition, reducing the problem to a standard eigenvalue problem (Saad, 2011). This is an improvement over the general case for the QZ algorithm (the generalised Schur decomposition). Moreover, the Cholesky decomposition enhances numerical stability, leading to fewer round-off errors. For large, sparse graphs and positive definite operators, iterative methods such as the Lanczos algorithm could be used. The complexity of these solvers is O(nkm), where k is the number of eigenvalues to find and m related to the iterations required for convergence and the condition number.
this section cite: ['b20', 'b15']

Section: Experiments
We conducted experiments to compare the spectral clustering method with the constrained clustering approach, using both synthetic and real-world datasets. The clustering accuracy was evaluated using the Adjusted Rand Index (ARI).
All simulations were run on a PC equipped with an Intel® Core™ i7-10610U CPU running at 1.80 GHz and 32 GB of RAM, using MATLAB R2024a for computation. The three clustering algorithms compared in our experiments are as follows:
• SPECTRAL CLUSTERING (SC): We computed the normalized Laplacian ∆ G and used its second smallest eigenvector (Fiedler vector) for clustering the vertices.
• CONSTRAINED CLUSTERING (CC): We solved (11) and used the eigenvector corresponding to the smallest positive eigenvalue (excluding the trivial zero eigenvalue) for clustering.
• CONSTRAINED CLUSTERING WITH NEGATIVE SELF-LOOPS (CC++): Our algorithm consisted in adding a negative self-loop and solved (11) for the signed Laplacian.
this section cite: []

Section: Stochastic Block Model
We considered a binary Stochastic Block Model (SBM) with n = 1, 000 vertices divided into two equal-sized communities. Edges between vertices were generated based on intra-cluster probability p and inter-cluster probability q. Specifically, we fixed p = 0.2 and varied q from 0.12 to 0.2 in 30 equidistant steps. For each value of q, we generated two graphs generated from the SBM:
• G: a graph with intra-cluster edge probability p and intercluster edge probability q.
• H: a graph generated with intra-cluster edge probability q and inter-cluster edge probability p, effectively the complement of G in terms of edge probabilities.
The vertex labels were kept consistent between G and H.
This variation allows us to observe how the clustering performance changes as the distinction between communities becomes less pronounced (since higher q implies more intercluster edges). The experimental results, visualised in Figure 1, illustrate the superior performance of CC++ compared to traditional SC, particularly as the inter-cluster edge probability q increases.
At low values of q both methods perform well, achieving near-perfect ARI values. As seen in Figure 1, both methods maintain the ARI values close to 1.0 when q ≤ 0.14. This is expected, as the strong intra-cluster edge probability p = 0.2 dominates the inter-cluster connections, making the community structure relatively easy to detect.
However, as q increases, the performance of SC deteriorates rapidly. For instance, between q = 0.16 and q = 0.18, the ARI for SC drops sharply from approximately 0.7 to near 0.1. This decline occurs because higher values of q increase the number of inter-cluster edges, blurring the distinction between communities. SC relies solely on the structure of G, and struggles to correctly partition the vertices under these conditions.
In contrast, both of CC and CC++ are significantly more robust to increasing q. Even as q approaches 0.17, the ARI remains above 0.5, significantly outperforming SC in this regime. This robustness stems from the ability of the generalised eigenvalue method to leverage the structural information of both G and H, the method mitigates the negative impact of increased inter-cluster edges, thus maintaining better clustering accuracy even when the community structure is less pronounced.
This set of experimental results clearly demonstrates that the CC++ algorithm outperforms the traditional SC, particularly in challenging scenarios where the inter-cluster edge probability q is high.
Figure 2 compares the mean execution time of SC, CC, and CC++ as the number of vertices increases. For smaller graphs, CC++ has nearly the same runtime as standard spectral clustering (SC), indicating minimal overhead from incorporating the second graph's constraints. As the graph size grows, however, CC++ clearly scales more efficiently than the original CC: the runtime of CC++ increases only modestly with n, remaining close to SC's runtime, whereas CC's runtime rises sharply. This demonstrates that CC++ retains the computational efficiency of SC while handling larger graphs far more effectively than CC, underscoring its superior scalability.
this section cite: []

Section: Varying Cluster Distance
Based on the Geometric Random Graphs (RGG) (Avrachenkov et al., 2021;Dall & Christensen, 2002), we generated two clusters of vertices,
0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 Inter-cluster Edge Probability (q) each containing 500 points, randomly distributed within two-dimensional circular regions (disks) of radius 0.2. The separation between the clusters' centroids varied between -0.35 and 0.35, in 25 equidistant steps. This range allows us to simulate different levels of overlap between the clusters. Each configuration of cluster separation was repeated 10 times, and we generated G and H on the same set of vertices but with different connectivity structures:
• G: vertices within the same cluster were connected with a large radius r intra = 0.1, and vertices between clusters were connected with a smaller radius r inter = 0.05.
• H: the intra-cluster connection radius is reduced to r intra = 0.05, while the inter-cluster radius is increased to r inter = 0.1. Edges more likely connect vertices across the two clusters
Figure 3 shows the RGG used in the experiments. The cluster distance is varied to simulate different levels of overlap between clusters. Figure 4 shows the ARI scores for both methods across different cluster distances. Each point represents the mean ARI, with error bars indicating the standard error across 10 repetitions.
When the cluster distance is large (above 0.3), both methods achieve near-perfect ARI values. The separation between the clusters is clear, and both algorithms can detect the underlying structure accurately. As the clusters get closer, SC shows a significant drop in performance. For distances near zero, where clusters overlap, the ARI scores for SC drop to nearly zero, indicating its struggle to differentiate overlapping clusters. In contrast, the generalised eigenvalue method is more resilient to cluster overlap, maintaining significantly higher ARI values even when the clusters become indistinguishable by conventional means. This robustness  is due to the method's ability to leverage information from both graphs G and H, capturing both intra-and inter-cluster relationships.
this section cite: ['b1', 'b6']

Section: Experiments with Temperature Data
We evaluated SC and CC++ on real-world temperature data from ground stations in Brittany, January 2014 (Girault, 2015). The experiments considered three data types: temperature, maximal temperature, and minimal temperature.
Temperature values can be seen as graph signals (Ortega et al., 2018), with vertices representing readings. The objective is to cluster stations by proximity and similar temperature patterns, combining spatial and temperature data. This approach can be applied to identify micro climates (Cao et al., 2021) and segment regions for agricultural (Yao et al., 2022), where both location and temperature are important factors.
We construct a graph from the input data as follows: every station is represented as a vertex, and the edge weight is defined by spatial similarity via a Gaussian kernel:
W ij = exp -d(i,j) 2 2σ 2 1 if d(i, j) ≤ σ 2
, and 0 otherwise, where d(i, j) is the Euclidean distance. Parameters were set to σ 2 1 = 5 × 10 8 and σ 2 = 10 5 (Girault, 2015). SC primarily grouped stations by proximity (Figure 5a).
To construct the constraint graph H, we use the temperature values and an inverse Gaussian kernel, assigning edge weights close to 1 for large temperature differences and 0 for similar temperatures, analogous to how G is built using spatial proximity. This allows the clustering algorithm to incorporate both geographical and temperature constraints for a more nuanced partition. The output of CC++ is shown in Figure 5b, where only two cities differ from the clustering based on spatial proximity alone (SC). Finally, we compute the mean and standard error (SE) for every clustering method. For this specific hour, SC shows overlap between clusters' temperature values, whereas CC++ achieves no overlap, indicating better clustering of stations with similar temperatures (Figure 5c). We repeated this process for all available measurements, one per hour, across 24 hours over 31 days, totaling 744 observations. To evaluate clustering accuracy, we calculated the mean and SE for clustering using both methods. A separation was considered correct if the clusters' values did not overlap, as determined by their SE. Table 1 summarises the results, showing the percentage of correct separations for each temperature data type when comparing SC and CC++, and demonstrates that CC++ consistently outperformed CC across all temperature data types. This can be attributed to the method's ability to leverage both the spatial structure of the stations and the actual temperature data. In contrast, SC, which relies solely on the graph structure, struggled to accurately separate stations into distinct clusters when the temperature differences were subtle.
this section cite: ['b7', 'b14', 'b3', 'b21', 'b7']

Section: Conclusion
In this paper, we introduced a novel spectral method for the constrained clustering problem that incorporates MUST-LINK and CANNOT-LINK constraints on the same vertex set. We established a Cheeger-type inequality that provides a theoretical approximation guarantee, relating the optimal constrained cut value to the spectral properties of the graphs. Building on this insight, we proposed an efficient algorithm (CC++) that solves a generalised eigenvalue problem involving the signed Laplacian of the constraint graph, which significantly reduces computational complexity while maintaining accuracy.
Empirically, CC++ demonstrated superior clustering performance on both synthetic and real-world datasets, particularly in challenging scenarios with weak or noisy cluster structure. It also achieved runtime comparable to standard spectral clustering (SC) and scaled much better than the original constrained clustering method (CC). These results highlight the significance of our proposed CC++ method as a principled and practical solution for constrained graph clustering, bridging the gap between theoretical guarantees and scalable real-world performance.
this section cite: []

Section: References
Ref_id:b0 Title: Cheeger constants, structural balance, and spectral clustering analysis for signed graphs Year: (2020)
Ref_id:b1 Title: Higherorder spectral clustering for geometric graphs Year: (2021)
Ref_id:b2 Title: Constrained clustering: Advances in algorithms, theory, and applications Year: (2008)
Ref_id:b3 Title: Withincity spatial and temporal heterogeneity of air temperature and its relationship with land surface temperature. Landscape and Urban Planning Year: (2021)
Ref_id:b4 Title: Spectral Graph Theory Year: (1997)
Ref_id:b5 Title: Simple and scalable constrained clustering: a generalized spectral method Year: (2016)
Ref_id:b6 Title: Random geometric graphs Year: (2002)
Ref_id:b7 Title: Stationary graph signals using an isometric graph translation Year: (2015)
Ref_id:b8 Title: Constrained clustering with dissimilarity propagation-guided graph-laplacian pca Year: (2021)
Ref_id:b9 Title: Signed Laplacian for spectral clustering revisited Year: (2017)
Ref_id:b10 Title: A generalized Cheeger inequality Year: (2023)
Ref_id:b11 Title: Frustration index and Cheeger inequalities for discrete and continuous magnetic Laplacians Year: (2015)
Ref_id:b12 Title: Multiway spectral partitioning and higher-order cheeger inequalities Year: (2014)
Ref_id:b13 Title: Hermitian Laplacians and a Cheeger inequality for the Max-2-Lin problem Year: (2019)
Ref_id:b14 Title: Graph signal processing: Overview, challenges, and applications Year: (2018)
Ref_id:b15 Title: Numerical methods for large eigenvalue problems: revised edition Year: (2011)
Ref_id:b16 Title: Is Cheeger-type approximation possible for nonuniform sparsest cut? Year: (2013)
Ref_id:b17 Title: A tutorial on spectral clustering Year: (2007)
Ref_id:b18 Title: Constrained k-means clustering with background knowledge Year: (2001)
Ref_id:b19 Title: Flexible constrained spectral clustering Year: (2010)
Ref_id:b20 Title: On constrained spectral clustering and its applications Year: (2014)
Ref_id:b21 Title: Energy-efficient routing protocol based on multithreshold segmentation in wireless sensors networks for precision agriculture Year: (2022)
