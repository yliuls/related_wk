Title: Boundary-Value PDEs Meet Higher-Order Differential Topology-aware GNNs
Abstract: Recent advances in graph neural network (GNN)-based neural operators have demonstrated significant progress in solving partial differential equations (PDEs) by effectively representing computational meshes. However, most existing approaches overlook the intrinsic physical and topological meaning of higher-order elements in the mesh, which are closely tied to differential forms. In this paper, we propose a higher-order GNN framework that incorporates higher-order interactions based on discrete and finite element exterior calculus. The time-independent boundary value problems (BVPs) in electromagnetism are instantiated to illustrate the proposed framework. It can be easily generalized to other PDEs that admit differential form formulations. Moreover, the novel physics-informed loss terms, integrated form estimators, and theoretical support are derived correspondingly. Experiments show that our proposed method outperforms the existing neural operators by large margins on BVPs in electromagnetism. Our code is available at https://github.  com/Supradax/Higher-Order-Differential-Topology-aware-GNN.

Section: Introduction
Solving partial differential equations (PDEs) accurately is fundamental in scientific computations. Traditional numerical solvers [1] and the emerging physics-informed neural networks [2] rely on iterative computations. This is a significant bottleneck that precludes their application in timesensitive domains where slight inaccuracy is tolerable but speed matters, such as gaming engines and interactive simulations. To address these limitations, neural operators [3][4][5][6] propose to directly learn the mapping between initial/boundary conditions and complete PDE solutions, eliminating time-consuming iterations while maintaining the capability to handle the PDEs whose exact analytical solutions are unattainable.
Convolutional neural network-based solvers are inherently constrained to regular, grid-like domains, whereas numerical solvers typically employ meshes to represent irregular solving regions-an approach naturally aligned with graph neural networks (GNNs). This compatibility has spurred growing interest in GNNs for time-dependent physics simulations, demonstrated through applications from fabric dynamics in wind [7] and granular particle systems [8] to neural mesh refinement schemes [9]. However, solving time-independent PDEs, especially boundary value problems (BVPs), presents greater challenges due to the absence of temporal guidance (initial data) and limited feature representation. Recent work [10] attempts to apply GNNs to BVPs in electromagnetism and has shown promising results, but is still limited to relatively simple cases.
Conventional GNN-based BVP solvers primarily utilize vertex adjacency in meshes while neglecting higher-order topological elements (edges, faces, cells), despite their fundamental physics interpretations from the perspective of differential forms [11]. While vector analysis has long dominated physical modeling, modern physics recognizes that many vector fields are more naturally interpreted as differential forms on manifolds [12]. In electromagnetism, it reveals a hierarchical structure: potentials (φ) manifest as 0-forms, field intensities (E, H) as 1-forms, flux densities (D, B) as 2-forms and density distributions (ρ) as 3-forms [13]. This formalism, crystallized in the Maxwell's House representation of Maxwell's equations [14], treats traditional vector fields as mere proxies for underlying forms. Discrete exterior calculus [15] operationalizes this approach through De Rham mappings that represent k-forms as integrals over k-simplices [16]. Also, introducing geometric objects like differential forms allows us to naturally generalize this framework to PDEs on curved spaces [17] beyond merely incorporating topological structures [18,19].
In this paper, we propose a higher-order GNN-based PDE solver framework by exploring the ideas in discrete exterior calculus (DEC) and finite element exterior calculus (FEEC) in a principled manner. By encoding the integrals over k-simplices as k-simplex features within higher-order GNNs (HOGNNs), which explicitly model interactions between simplices of varying dimensions, we establish a principled framework for solving form-based PDEs while preserving the topological and physical structure inherent to the problem domain. The resulting framework is dubbed DEC-HOGNN and illustrated in Figure 1. DEC-HOGNN enjoys better physical interpretation from the differential form perspective and can be extended to higher-dimension cases naturally. Our main contributions are summarized as follows: 1) We design a differential topology-aware HOGNN, which naturally encodes and decodes PDE operators based on the principles of DEC and FEEC. 2) Various physics-informed loss terms are derived under DEC-HOGNN, including the boundary condition ones, which can enable solving the boundary-value PDEs more effectively. 3) The universal approximation property in solving Poisson problems (electrostatics and magnetostatics) is presented, and the performance excellence is demonstrated via empirical experiments.
2 Related Work Higher-Order GNNs. HOGNN is an extended learning framework on generalized graphs, i.e., hypergraphs [20]. A hypergraph G allows a hyperedge to contain more than two vertices [21]. HOGNN leverages the more abundant adjacencies on hypergraphs and mimics what GNN does on plain graphs via redefining various neighborhoods. If G has no further decorated structures, then one can define the k-node-tuple adjacency [22]; while boundary, co-boundary, upper, and lower adjacencies are available when G is a simplical complex [18] or a cell complex [23]. Mechanisms in GNN are mostly based on a special adjacency induced by edges and thus can be easily transplanted to hypergraphs. The graph-convolution [24], attention [25], and generalized message passing [26] of HOGNN all fall into this category. HOGNN has been well-studied in various regions, such as recommendation systems [27] and molecular predictions [28], where multi-body interactions are of significance, but beyond the expressive ability of plain graphs.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b17', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27']

Section: Neural Operators.
Neural operator [3] learns function-to-function mappings mainly using datadriven loss instead of PDE-based physics loss. Its original implementation is furnished with kernel convolution of O(n 2 ) complexity, which can be improved to O(n log n) in the spectral domain via discrete fast Fourier transform [3]. It is then extended to non-square-like regions [4] and spatialspectral neural operators realized by wavelet transform [29][30][31]. This field later gradually shifts towards Transformer architectures with PDE-compatible attentions, from GNOT [5] employing boundary-aware cross-attention to Transolver [6] using attentions among domain slices. GNN-based PDE Solvers. GNN-based neural operators have been studied in time-evolving meshbased and particle-based physics simulations [7,8]. The particles appear in the form of point clouds, and adjacency is built on local neighborhoods, in which equivariance is introduced to enhance model performance, such as subequivariance [32] and IsoGCN [33], an equivariant data-driven neural differential operator. To align with Neumann boundary in PDEs, NIsoGCN [34] further introduces a Neumann term into the differential kernel. This category requires data on fields and their differentials, but the latter is often intractable. It is found that message passing in GNNs can represent various numerical methods for time-dependent PDEs [35] and aligns with finite volume methods to achieve local mass conservation [36]. In addition, GNN-based approaches for BVPs [10] and inverse problems [37] are also explored.
this section cite: ['b2', 'b2', 'b3', 'b28', 'b29', 'b30', 'b4', 'b5', 'b6', 'b7', 'b31', 'b32', 'b33', 'b34', 'b35', 'b9', 'b36']

Section: Preliminary
De-Rham Mapping. In algebraic topology, a simplex chain complex C(X) on a topological space X is a graded vector space of k-order simplices C k (X), decorated with the boundary operator ∂. A k-cochain σ k ∈ C k (X) : C k (X) → R is a real-valued function of σ k ∈ C k (X). If there exists a diffeomorphism φ between X and a smooth manifold M, then for any k-form ω, we obtain a covariant functor mapping from C k (X) to C k (X), namely, De-Rham mapping [16]:
F : C k (X) → C k (X), σ k → φ(σ k ) ω.(1)
DEC and FEEC. Discrete Exterior Calculus (DEC) offers a comprehensive and differential topologypreserving toolkit to discretize operators on manifolds. In contrast to Graph Calculus [38], which treats a discretized manifold as a graph-at the cost of losing essential differential properties and thereby introducing inaccuracies-DEC maintains differential topology properties by working with integrations. Specifically, it characterizes a k-form ω ∈ Ω k (M) on a discrete manifold M through its integral over every k-simplex [16]. It yields high accuracy in applications where differential information matters, and is widely used in computational physics [39][40][41][42]. Another benefit of DEC is that it allows for processing manifolds from a dual perspective. For instance, the Hodge star * k : Ω k (M) → Ω n-k (M) on M gives the important constitution relation between B and H in electromagnetism. Even if * k ω is intractable in the discrete case, we can still estimate its integral on the dual manifold ⋆M (shown in Figure 2) without a priori on the metric [15]. Finite Element Exterior Calculus (FEEC) generalizes the Finite Element Method (FEM) and is a numerical implementation of Galerkin methods. FEM is in essence an interpolation method with nodal functions, a.k.a., Lagrangian element [1]. However, such node-wise interpolation is inherently incompatible with differential operators such as div and curl, whereas the adoption of higher-order finite elements in FEEC enables the exact representation of these operators in the integral sense via straightforward linear combinations [14].
this section cite: ['b15', 'b37', 'b15', 'b38', 'b39', 'b40', 'b41', 'b14', 'b0', 'b13']

Section: Whitney Element.
Whitney elements are a type of finite elements used in FEEC and DEC. They provide a way to approximate differential forms on a discretized manifold that respects the geometry
this section cite: []

Section: Primal Edge Coboundary Adjacency

this section cite: []

Section: Dual Edge Coboundary Adjacency

this section cite: []

Section: Field Enconding
Field Recovery/Decoding
this section cite: []

Section: Feature Forwarding
Primal Edge Lower Adjacency and topology of the manifold. Let w and w denote scalar and vector fields, respectively. Given a tetrahedralized mesh representation of a manifold M with dim M = 3, let [i, j] denote the edge containing v i , v j and so are [i, j, k], [i, j, k, l] for face and cell, respectively. The canonical node elements {w i : w i (v j ) = δ ij } are functions on M, in which δ ij is the Kronecker delta. Let W 0 (M), W 1 (M), W 2 (M), and W 3 (M) denote the function spaces spanned by node elements w i , edge elements w [i,j] , face elements w [i,j,k] , and cell elements w [i,j,k,l] (scalar fields), respectively. Then the integral of w [i,j] along [i, j], the flux of w [i,j,k] through face [i, j, k], and the volume integral of w [i,j,k,l] within tetrahedral [i, j, k, l] all equal to 1 [14]. The element definitions are as follows:
w [i,j] := w i ∇w j -w j ∇w i(2)
w [i,j,k] := 2(w i (∇w j × ∇w k ) + w j (∇w k × ∇w i ) + w k (∇w i × ∇w j ))(3)
w [i,j,k,l] := 6 cyc w i (∇w j × ∇w k ) • ∇w l = χ x∈[i,j,k,l] / vol([i, j, k, l])(4)
in which w i , w j , w k , w l ∈ W 0 (M). For ease of reference, the notation frequently used throughout the paper is summarized in Table 5 in Appendix A.
this section cite: ['b13']

Section: Methodology
Problem Setup and Motivation. In this work, we focus on PDEs that admit differential form formulation (many important PDEs in physics and engineering can be formulated using differential forms, such as Maxwell's, Navier-Stokes, Yang-Mills Equations, etc.). Given PDEs (often formulated in vector fields) on a discrete manifold M, we aim to learn a neural operator G Θ that takes as inputs the observed scalar fields {s i } and vector fields {v i }, and outputs the target vector field u of interest:
u = G Θ ({s 1 , ..., s m }, {v 1 , ..., v n }, M).(5)
In comparison to the vector field formulation, the differential-form characterization of PDEs is coordinate-independent and explicitly reveals the geometric and topological aspects of the spaces. Motivated by this, we propose to transform the vector fields into the differential form formulation of the PDEs, which enables us to develop the neural operators by combining the principles in DEC and FEEC. The benefits are threefold: 1) they permit exploring the higher-order topological elements to facilitate PDE solving, 2) developing various physics-consistent losses, especially for the boundaryvalue problems, and 3) differential operators can be interpreted as simple linear combinations in DEC and FEEC, aligning with the higher-order message passing framework. Once the PDEs are solved, we then translate the results into the vector field language for downstream analysis.
this section cite: []

Section: Method Overview.
Since differential k-forms can be represented as integrals on k-dimensional elements and further identified as higher-order element features, we introduce physics-informed higher-order interactions and aggregations into existing HOGNN frameworks in Section 4.1. Nevertheless, the initial input and expected output are often vectors instead of forms in practice. To circumvent this, we propose a proper encoder-decoder to enable consistent transformations in Section 4.2. In a nutshell, the input vectors are encoded into forms (higher-order features), processed by HOGNN, and then decoded back into vectors, as illustrated in Figure 1. The physics-constrained loss and the universal approximation property of the proposed method are presented in Section 4.3.
As an illustrative example, we instantiate our framework by solving the classical BVPs in electromagnetism, and the proposed method can be easily generalized to other PDEs that can be formulated using differential forms. Recall that a typical Neumann electrostatic BVP is
∇ • D = ρ, ∇ × E = 0 in Ω; E = E 0 , D = D 0 on ∂Ω; D = ε i • E in Ω i ,(6)
in which ρ, D, E are the charge density, displacement field, and electric field, respectively, whereas {Ω i } is a partition of the domain Ω. The permittivity tensor ε i can vary in different Ω i consisting of different materials. In this case, the learned operator G Θ will take as input ({ρ} , {E 0 , D 0 } , Ω) and produce the complete fields (E, D). To this end, we rewrite Eq. 6 in its differential form formulation.
dD = ρ, dE = 0 in Ω; E = E 0 , D = D 0 on ∂Ω; D = ε * 1 E in Ω i ,(7)
in which d• denotes the exterior derivative and D, E are the corresponding k-forms. The input scalar field ρ and masked vector fields E, D are first encoded into integrated forms in the form of vertex, edge and face features. Three types of edge adjacencies are used in the higher-order GNN, which give important physics interpretations (as shown in Figure 3): the curl-free property dE = 0 is implicitly included in the primal edge co-boundary adjacency, while Gauss's Law dD = ρ is involved in the dual edge co-boundary adjacency. These GNN layers can be stacked sequentially, and the aggregated features can be either decoded back as complete E, D or forwarded to other networks.
this section cite: []

Section: Physics-Informed Higher-Order Interactions
In light of DEC, we identify the potential on node, the circulation along an edge, the flux through a face, and the mass within a cell as the node, edge, face, and cell features, respectively. Given a discrete manifold M and its dual ⋆M, since both of them are cell complexes, an element c onward have four types of neighborhood: boundary B(c) := ∂c and co-boundary C(c) := {d : c ∈ B(d)}, upper adjacency N ↑ (c) := {d : ∃δ, {c, d} ⊂ B(δ)} and lower adjacency N ↓ (c) := {d : ∃δ, {c, d} ⊂ C(δ)}.
Following the paradigm of [26], the element feature h ′ c of c is updated by its original feature h c and the aggregated messages from four neighborhoods B(c), C(c), N ↑ (c), N ↓ (c) via:
h ′ c = φ h c , m B c , m C c , m N ↓ c , m N ↑ c , c ∈ M,(8)
m B c = Aggregate d∈B(c) φ B (h c , h d ),(9)
m C c = Aggregate d∈C(c) φ C (h c , h d ), (10
)
m N ↑ c = Aggregate d∈N ↑ (c), δ∈C(c)∩C(d) φ N ↑ (h c , h d , h δ ),(11)
m N ↓ c = Aggregate d∈N ↓ (c), δ∈B(c)∩B(d) φ N ↓ (h c , h d , h δ ).(12)
Since forms in a PDE can be defined on both M and ⋆M (e.g., E and D in Eq. 7), we also propose a higher-order MPNN on the dual manifold ⋆M for dual forms by the fact:
B(⋆c) = {⋆d : d ∈ C(c)}, C(⋆c) = {⋆d : d ∈ B(c)}, N ↑ (⋆c) = {⋆d : d ∈ N ↓ (c)}, N ↓ (⋆c) = {⋆d : d ∈ N ↑ (c)}.
Usually, not all neighborhoods and elements will be used in higher-order MPNN due to computational complexity and the absence of features on corresponding elements. For instance, 2D-electrostatics BVPs only involve E, D, ρ and thus only adjacencies about edges and faces are considered. In recent studies, there are various ways to implement the message passing on four different adjacencies, including generalized convolution, attention, and Transformer. We list several candidates of the adjacency layer backbones in Table 1 and will study their impact on model efficacy in Section 5.2.
In Table 1, L (k) := D -A = 2D -II ⊤ is the higher-order Laplacian [21], defined by indicator matrix I ij := χ σi∈N ↓ (σj ) and diagonal degree matrix D; L is the Laplacian of an extended graph G = (C k ∪ C k+1 , {(c, d) : c ∈ C k , d ∈ C(c)}); h c,d,δ is the concatenation of h c , h d , h δ and h c,d is likewise. In our implementation, the relative orientation sign(c, d) between two elements is further considered, via replacing h c,d by sign(c, d)h c,d .
Consistency with Conservation Law. It is beneficial for solving BVPs by introducing higher-order interactions, because the interactions have meaningful physics interpretations, e.g., the sum of edge
L (k) h (k) Θ (k) d + L (k+1) h (k+1) Θ (k+1) δ h ′(k) c = σ |C(c)| -1 2 d∈C(c) L(k) h (k) Θ (k) d Attention h ′ c = σ(WC hc + (d,δ)∈N ↓ (c) softmax N ↓ (c) (α dδ )WN h c,d,δ ), α dδ = WAh c,d,δ h ′ c = σ(WC hc + d∈C(c) softmax C(c) (α d )WN h c,d ), α d = WAh c,d Transformer h ′ c = σ(WC hc + (d,δ)∈N ↓ (c) softmax N ↓ (c) (|N ↓ (c)| -1 2 q ⊤ c k d )WN h d ) h ′ c = σ(WC hc + d∈C(c) softmax C(c) (|C(c)| -1 2 q ⊤ c k d )WN h d )
features around a face corresponds to the vorticity, while the sum of face features around a cell indicates the divergence at that location. This enables us to preserve structural conservation discretely by using integrated differential forms. Moreover, prior work also shows that adding conservation regularization can boost the performance of PDE solvers. For instance, a conservative GNN solver for 2D fluid dynamics [36] employs message passing based on face lower adjacency in the HOGNN framework and achieves local mass conservation (divergence-free) through asymmetric aggregation.
We can realize more conservation constraints with DEC, e.g., enforcing vorticity conservation (∇ × E = 0) and divergence conservation (∇
• B = 0, ∇ • D = ρ) in electromagnetism.
Consistency with Electromagnetic Constitution Law. As a side-product, we can estimate the electric and magnetic permeability ε and µ of the medium while solving BVPs. The constitution law B = µH, D = εE can be equivalently interpreted by Hodge star: B = µ * 1 H, D = ε * 1 E. It allows us to estimate the µ, ε along different directions via the definition in DEC:
⋆σ k * k ω := vol (⋆σ k ) vol (σ k ) σ k ω, ω ∈ Ω k (M). (13
)
this section cite: ['b25', 'b6', 'b20', 'b35']

Section: Encoder-Decoder between Vector Fields and Forms
Encoder. The encoder aims to transform vector fields into higher-order element features (integrated forms). In 3D-cases, a 1-form ω 1 and a 2-form ω 2 can be realized by vector proxies u 1 and u 2 , with the aid of a unit tangent vector t on an edge σ 1 and a unit normal vector n on a face σ 2 , respectively; while a 3-form ω 3 can be identified as a scalar field u 3 : R 3 → R, and the integrated form is the usual volume integral. More formally,
σ1 ω 1 := σ1 u 1 • tds, σ2 ω 2 := σ2 u 2 • ndS, σ3 ω 3 := σ3 u 3 dV.(14)
Given the samples {x j } observed on σ i (i = 1, 2, 3), the integrals on the right-hand sides of Eq. 14 can be estimated to yield the integrated forms numerically with either the Monte Carlo method or canonical quadrature rules on simplices (e.g., cubic quadrature on a triangle [43]). Similarly, the 2D case is given as follows and can be estimated similarly:
σ1 w 1 = σ1 u 1 • tds, σ2 w 2 := σ2 u 2 dxdy. (15
)
In DEC, the differential forms in the primal manifold and dual manifold are related by the Hodge star * 1 (e.g., E ∈ Ω 1 (M) and D ∈ Ω 2 (⋆M) in Eq. 7). Since we only have observations at primal nodes, it is simpler to estimate the integrated 2-forms on the primal manifold. However, DEC requires the integrals on the dual manifold, which poses a challenge for implementation. To sidestep this, we derive an approximation of the integrated dual forms based on the integrated primal forms, presented in Theorem 1. The details of the theorem can be found in Appendix C. Intuitively, the theorem offers us a simpler way to calculate the integrated dual forms.
this section cite: ['b42']

Section: Theorem 1 (Non-Cycle Forms Estimation).
For a smooth k-form ω that is not a cycle defined on a bounded region, i.e., dω does not vanish identically in any k-simplex σ k , one can estimate the integrated dual forms { ⋆σ n-k ω : σ n-k ∈ C n-k (X)} by using the integrated primal forms
{ σ k ω : σ k ∈ C k (X)} up to accuracy O(ε k+1 ) in which ε := sup σ k ∈C k (X) diam σ k .
Decoder. The decoder intends to recover the vector fields from the derived integrated forms. The principle is that the integral of a proper vector field proxy on a simplex should equal the integrated form onward. To find the proxy at v i for a 1-form (e.g., E in Eq.
7) along edge [v i , v j ], note that h [vi,vj ] -the final edge feature output by DEC-HOGNN-indicates the circulation of the proxy and the circulation of w [vi,vj ] in Eq. 2 along edge [v i , v j ] is 1, and thus the Whitney electric field contributed by edge [v i , v j ] at v i is set to h [vi,vj ] w [vi,vj ] . The contributions to v i from different edges can be aggregated by average pooling or attention-style weighted sum. Similarly, for the proxies of 2-forms (e.g., D in Eq. 7) with final face feature h [vi,vj ,v k ] and 3-forms (e.g., ρ in Eq. 7) with final cell feature h [vi,vj ,v k ,v l ] , the face and cell contribution are h
[vi,vj ,v k ] w [vi,vj ,v k ] and h [vi,vj ,v k ,v l ] w [vi,vj ,v k ,v l ] respectively, with w [vi,vj ,v k ] , w [vi,vj ,v k ,v l ]
defined by Eq. 3 and Eq. 4.Remark. The interpolation nature of the decoding approach offers a continuous field not only defined on the nodes but also in the cells. Though other GNN operators can also estimate the inner fields by multi-linear interpolation based on values at nodes, such methods can destroy certain important physical properties compared with Whitney elements. For instance, the reflection law in electrodynamics elucidates that the tangent component of E on the medium interface is continuous while the normal component is not. With prior knowledge of the interface positions, one can choose to average out contributions from each side respectively and obtain two different E's before and after reflection, which is consistent with physics on the interface, while a usual interpolation based on node-wise data cannot. Lastly, we present the theoretical analysis in Theorem 2, demonstrating that our encoding and decoding scheme preserves sufficient information for scalar and vector fields on a sufficiently fine mesh.
this section cite: []

Section: Theorem 2 (Proper Encoder-Decoder).
Given a uniform partition {[
x i , x i+1 ] : 1 ≤ i ≤ 2 N } on the unit interval, i.e., 0 = x 1 < x 2 ... < x 2 N +1 = 1, the encoding operator E N : W 1,2 ([0, 1]) → R 2 N is defined as the 2 N integrals on each [x i , x i+1 ]
and the decoding operator D N : R 2 N → C ∞ ([0, 1]) maps any 2 N -dimensional feature h to a function f (x) on [0, 1] such that xi+1 xi f (x)dx = h i . Then for any φ ∈ L 2 ([0, 1]) and ε > 0, there exists an integer M > 0 such that for all N > M ,
||φ -D N • E N (φ)|| L 2 < ε. (16
)
By definition, we have E N • D N = id .
this section cite: []

Section: Physics-Informed Loss and Universal Approximation Property
Physics-Informed Loss. Purely data-driven methods are likely to go against physics. Hence, PINN introduces physics-constrained loss as regularization to guide models to learn beyond the data resolution [44]. In PINN, the PDE residual is usually estimated by point-wise sampling and auto-differentiation. This is incompatible with GNNs since vertices are discrete, making it hard to enforce physics-consistency constraints, e.g., constraining a field to be curl-free [10]. But in integrated forms, one can interpret a differential version of PDEs into the integral version. For instance, magnetic flux density B is always divergence-free, inducing a DEC-version constraint:
dB = 0 ⇔ ∇ • B = 0 ⇒ Ω ∇ • B = ∂Ω B • n = σ2∈∂Ω σ2 B • n σ2 = σ2∈∂Ω σ2 B = 0. (17
)
In our proposed framework, σ2 B is indeed a feature of face σ 2 . Therefore, we can introduce physicsinformed loss without sampling and differentiation, but by simply summing up corresponding higher-order features. More physics-informed loss terms are derived in Appendix B, covering the divergence, vorticity, and boundary conditions.
this section cite: ['b43', 'b9']

Section: Universal Approximation Property.
Theorem 3 shows the universal approximation ability of DEC-HOGNN in solving the Poisson problems. The proof is available in Appendix D.
this section cite: []

Section: Theorem 3 (Universal Approximation Property).
Let H 2 (Ω) denote the Hilbert space on a bounded closed region Ω with C 1 -boundary, and P i := D i • E i be the encoder-decoder projection with  resolution i. Given a simplex-partition {X α } of Ω with measure-zero intersections, for any f ∈ V ∪ ∞ i=1 P i V, V := H 2 ([0, 1] n ) ∩ {f : ||∇f || ∞ < M }, there exists a network in the form of D • L N ... • L 1 • E with infinite neurons to solve the Dirichlet Poission problem on Ω, such that:
lim sup diamXα→0 ||D • L N ... • L 1 • E(f, g) -u|| L 2 = 0, (18
)
in which ∆u(x) = f (x), x ∈ Ω; u(x) = g(x), x ∈ ∂Ω.
As a corollary, note that E = ∇u and the experiment setting below is equivalent to solving a Neumann boundary condition in a linear medium, the universal approximation property also rings true by a similar proof in Appendix D via single-layer potential method.
this section cite: []

Section: Experiment

this section cite: []

Section: Experiments and Benchmarks
Experiment Tasks. We assess the performance of the proposed model on 2D electric and magnetic boundary value problems. In the electrostatic case, the system is governed by a potential φ: ∆φ(x) = ρ(x), x ∈ Ω; ∇φ(x) = u(x), x ∈ ∂Ω, which gives the electric intensity E and electric displacement D. In our settings, the domain Ω is partitioned into two regions Ω 1 , Ω 2 with measure-zero intersection. Ω 1 has isotropic permeability ε 1 I while Ω 2 has a non-isotropic linear one ε 2 , as shown in the blue and green-colored regions in Figure 4, and D(x) = ε i E(x), x ∈ Ω i , i = 1, 2. Let H 2 (Ω) be the function space on Ω with L 2 derivatives, i.e., H 2 (Ω) = f : ||f || 2  2 + ||∇f || 2 2 < ∞ and H 2 (Ω) the vector-valued function space on Ω with H 2 (Ω) components. The operator G :
H 2 (Ω) × H 2 (∂Ω) 2 → H 2 (Ω) 2 , (ρ, E • χ x∈∂Ω , D • χ x∈∂Ω ) → (E, D)
recovers the field E, D on Ω based on prior knowledge on the boundary field data (E, D) on ∂Ω and the charge density distribution ρ on Ω, as shown in Figure 4.
Dataset Generation. Both 2D and 3D meshes are adopted as illustrated in Figure 5. The electrostatic data is obtained by FEM-based electromagnetism PDE solvers. Ω is partitioned into many triangles X i with measure-zero intersections. The charge density ρ i in each X i is randomly assigned following a uniform distribution. The PDEs are then solved in a large enough vacuum region with Neumann boundary conditions. E, D on boundary vertices and ρ on all vertices are sampled as input while E, D on all vertices are the target output. Scalar fields are normalized, and vector fields are shrunk with respect to the average norm to eliminate magnitude differences.
Note that the 2D magnetostatic field is also governed by a potential A z (the vector potential A has only one non-vanishing component A z if the 2D space is identified as the xy-plane). Therefore, the data generation is similar. For more details, please refer to Appendix E.
this section cite: []

Section: Numerical Results
Baselines and Implementation Details. To showcase the necessity of introducing specifically devised solvers for BVPs and the efficacy of the proposed method, we evaluate our model against the following general time-dependent PDE neural solvers, in which the input fields are masked accordingly: DeepONet [45], MKGN [46], Galerkin-type Attention [47], GNOT [5], and Transolver [6]; in addition, various GNN-based solvers devised particularly for BVPs are also included (as in [10]): GCN [48], GAT [49], Graph U-Net [50], and Graph-Transformer-based BVP solvers [51].
this section cite: ['b44', 'b45', 'b46', 'b4', 'b5', 'b9', 'b47', 'b48', 'b49', 'b50']

Section: Main Results.
Table 2 presents the MSE (mean square error) of different methods on 2D/3D electrostatics/magnetostatics BVPs. The neural operators from DeepONet to Transolver, which are mostly designed for time-dependent PDEs, yield larger errors compared with GNN-based BVP solvers. It is because many time-evolving PDE solvers like DeepONet predict an increment based on previous observations, which is unfortunately intractable in static cases. The multi-scale design in MKGN and the heterogeneous cross-attention in GNOT both suffer from the lack of features, as time-independent PDEs can be determined by the boundary conditions on boundary nodes, which
takes up a minority. These make them no better than simple GNN operators in [10]. The proposed DEC-HOGNN considers implicit higher-order interactions in the governing systems and thereby outperforms usual GNN operators.
Ablation I: On Higher-Order Interactions. Different types of interactions can contribute variably to the performance of DEC-HOGNN. In this experiment, the full model incorporates four types of interactions, namely, primal and dual lower/co-boundary adjacencies, which are systematically ablated to evaluate their impact. The resulting variants are denoted as Primal-Dual (PD), Primal-only (P), Dual-only (D), and None. Table 3 presents the results in 2D electrostatic BVPs, which shows that message passing on edge lower adjacency brings minor enhancement while co-boundary adjacency matters much more. This result coincides with the governing PDE in electrostatics, dE = 0, dD = ρ, whose inducing loss terms are supposed to be computed based on both primal and dual co-boundary adjacency (Appendix B).  Ablation II: On Different GNN Backbones. In this experiment, we study the impact of different variants of higher-order GNNs on the efficacy of DEC-HOGNN by selecting different backbones, including GCN, GAT, and Graph Transformer (GT), on 2D electrostatic BVPs. Results in Table 4 manifest that higher-order interactions can enhance model performance (GAT and GT), which is consistent with the previous results. It further shows that adding more higher-order interactions does not necessarily imply more positive sides. For instance, the appearance of lower adjacency impedes GAT and GCN, while the co-boundary adjacency is much more beneficial for all backbones. Also, not every convolution can fit the DEC-HOGNN framework well just like the counterexample GCN.
Ablation III: On Performance Degeneration due to Mesh Quality. We analyze the negative impact out of mesh degradation by randomly dropping a certain amount of edges hierarchically. To measure the mesh quality quantitatively, three indicators are adopted with arrows implying degrading directions since good quality usually comes with uniform and regular elements. Further details on these tailored meshes are covered in Appendix E. The left panel of Figure 6 reflects that dropping edges from a triangularized mesh is followed by mesh degeneration. Also, minor degeneration would not affect the performance violently while a major one leads to salient performance drop. Note that it is also infeasible to adopt classical solvers using meshes with prominent quality issues. Thus these negative effects are tolerable.
Ablation IV: On Different Topological Characteristics. As shown in Figure 12, eight 2D magnetotastics benchmarks are used for the evaluation, which are named after their different topological properties by (Connected Component Amount, Hole Amount). The right panel of Figure 6 shows that the advantage of our approach persists as the underlying topology changes.
this section cite: ['b9']

Section: Conclusion
In this paper, we propose a BVP solver via integrating higher-order topological interactions, which aligns with the discrete representation of differential forms, and the resulting model is aware of differential topology. Several novel physics-informed loss terms and integrated form estimators are also developed. Both theoretical analysis and experimental results demonstrate the advantages of incorporating higher-order interactions via integrated differential forms. Similar to traditional mesh-based numerical solvers, the performance of DEC-HOGNN is influenced by mesh quality, as Whitney elements may degrade on poorly shaped triangulated or tetrahedral meshes, potentially leading to convergence issues. Extending this approach to more general time-dependent PDEs and developing methods to mitigate mesh quality dependence are left for our future work.
this section cite: []

Section: References
Ref_id:b0 Title: Finite element method Year: (2012)
Ref_id:b1 Title: Deep hidden physics models: Deep learning of nonlinear partial differential equations Year: (2018)
Ref_id:b2 Title: Fourier neural operator for parametric partial differential equations Year: (2020)
Ref_id:b3 Title: Fourier neural operator with learned deformations for pdes on general geometries Year: ()
Ref_id:b4 Title: GNOT: A general neural operator transformer for operator learning Year: ()
Ref_id:b5 Title: Transolver: A fast transformer solver for PDEs on general geometries Year: ()
Ref_id:b6 Title: Learning meshbased simulation with graph networks Year: ()
Ref_id:b7 Title: Learning to simulate complex physics with graph networks Year: (2020)
Ref_id:b8 Title: Graph element networks: Adaptive, structured computation and memory Year: (2019)
Ref_id:b9 Title: Learning the solution operator of boundary value problems using graph neural networks Year: (2022)
Ref_id:b10 Title: Vector Calculus, Linear Algebra and Differential Forms: A Unified Approach Year: (2001)
Ref_id:b11 Title: Differential forms and electromagnetic field theory Year: (2014)
Ref_id:b12 Title: On the geometry of electromagnetism Year: (1998)
Ref_id:b13 Title: Computational electromagnetism: variational formulations, complementarity, edge elements Year: (1998)
Ref_id:b14 Title: Discrete exterior calculus Year: (2003)
Ref_id:b15 Title: Discrete differential forms for computational modeling Year: (2006)
Ref_id:b16 Title: Curvature-aware graph attention for PDEs on manifolds Year: ()
Ref_id:b17 Title: Simplicial neural networks Year: (2020)
Ref_id:b18 Title: Gaussian processes on cellular complexes Year: (2024)
Ref_id:b19 Title: Demystifying higher-order graph neural networks Year: (2024)
Ref_id:b20 Title: Networks beyond pairwise interactions: Structure and dynamics Year: (2020)
Ref_id:b21 Title: Weisfeiler and leman go neural: Higher-order graph neural networks Year: (2019)
Ref_id:b22 Title: Weisfeiler and lehman go cellular: Cw networks Year: ()
Ref_id:b23 Title: Simplicial 2-complex convolutional neural networks Year: (2020)
Ref_id:b24 Title: Cell attention networks Year: ()
Ref_id:b25 Title: Weisfeiler and lehman go topological: Message passing simplicial networks Year: ()
Ref_id:b26 Title: Graph neural networks in recommender systems: a survey Year: (2022)
Ref_id:b27 Title: Few-shot molecular property prediction via hierarchically structured learning on relation graphs Year: (2023)
Ref_id:b28 Title: Multiwavelet-based operator learning for differential equations Year: ()
Ref_id:b29 Title: Wavelet neural operator for solving parametric partial differential equations in computational mechanics problems Year: (2023)
Ref_id:b30 Title: Coupled multiwavelet operator learning for coupled differential equations Year: ()
Ref_id:b31 Title: Learning physical dynamics with subequivariant graph neural networks Year: ()
Ref_id:b32 Title: Isometric transformation invariant and equivariant graph convolutional networks Year: ()
Ref_id:b33 Title: Physics-embedded neural networks: Graph neural pde solvers with mixed boundary conditions Year: ()
Ref_id:b34 Title: Message passing neural PDE solvers Year: ()
Ref_id:b35 Title: Graph neural PDE solvers with conservation and similarity-equivariance Year: (2024)
Ref_id:b36 Title: Learning to solve PDE-constrained inverse problems with graph networks Year: (2022)
Ref_id:b37 Title: Graph laplacians and their convergence on random neighborhood graphs Year: (2007)
Ref_id:b38 Title: Stable, circulationpreserving, simplicial fluids Year: (2007)
Ref_id:b39 Title: A primitive variable discrete exterior calculus discretization of incompressible navier-stokes equations over surface simplicial meshes Year: ()
Ref_id:b40 Title: Effects of rotation on vorticity dynamics on a sphere with discrete exterior calculus Year: ()
Ref_id:b41 Title: Energypreserving integrators for fluid animation Year: (2009)
Ref_id:b42 Title: Gaussian quadrature formulas for triangles Year: (1973)
Ref_id:b43 Title: Learning the solution operator of parametric partial differential equations with physics-informed deeponets Year: (2021)
Ref_id:b44 Title: Learning nonlinear operators via deeponet based on the universal approximation theorem of operators Year: (2021)
Ref_id:b45 Title: Multipole graph neural operator for parametric partial differential equations Year: (2020)
Ref_id:b46 Title: Choose a transformer: Fourier or galerkin Year: ()
Ref_id:b47 Title: A comprehensive review of graph convolutional networks: Approaches and applications Year: ()
Ref_id:b48 Title: Graph attention networks Year: (2018)
Ref_id:b49 Title: Graph u-nets Year: (2019)
Ref_id:b50 Title: Graph transformer networks Year: (2019)
Ref_id:b51 Title: Partial Differential Equations Year: (2010)
Ref_id:b52 Title: Differential Geometry: Connections, Curvature, and Characteristic Classes Year: (2017)
Ref_id:b53 Title: Approximation by superpositions of a sigmoidal function Year: (1989)
Ref_id:b54 Title: Delaunay refinement algorithms for triangular mesh generation Year: (2002)
Ref_id:b55 Title: Tetgen, a delaunay-based quality tetrahedral mesh generator Year: (2015)
