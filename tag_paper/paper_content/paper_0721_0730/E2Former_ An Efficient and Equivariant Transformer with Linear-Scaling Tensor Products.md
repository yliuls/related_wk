Title: E2Former: An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products
Abstract: Equivariant Graph Neural Networks (EGNNs) have demonstrated significant success in modeling microscale systems, including those in chemistry, biology and materials science. However, EGNNs face substantial computational challenges due to the high cost of constructing edge features via spherical tensor products, making them almost impractical for large-scale systems. To address this limitation, we introduce E2Former, an equivariant and efficient transformer architecture that incorporates a Wigner 6j convolution (Wigner 6j Conv). By shifting the computational burden from edges to nodes, Wigner 6j Conv reduces the complexity from O(|E|) to O(|V|) while preserving both the model's expressive power and rotational equivariance. We show that this approach achieves a 7x-30x speedup compared to conventional SO(3) convolutions. Furthermore, our empirical results demonstrate that the derived E2Former mitigates the computational challenges of existing approaches without compromising the ability to capture detailed geometric information. This development could suggest a promising direction for scalable molecular modeling.

Section: Introduction
Molecular simulations underpin critical computational tasks across chemistry [31,36,38,39], biology [9], and materials science [64], facilitating detailed exploration of microscopic processes. Although quantum mechanical approaches such as Density Functional Theory (DFT) provide highly accurate predictions [30,37], their computational complexity scales poorly with system size [56], thus limiting practical applicability to small-scale problems. Machine Learning (ML) techniques have emerged as promising alternatives, balancing computational efficiency and accuracy [5,4,16]. ML-based models, particularly Equivariant Graph Neural Networks (EGNNs), significantly reduce simulation times, enabling molecular property predictions and dynamic simulations within practical computational budgets [53,26,25,7,23,41]. EGNN architectures explicitly encode symmetry constraints-such as rotational and reflectional equivariances-through graph-based atomic representations. This symmetry-awareness leads to strong inductive biases and improved sample efficiency. EGNNs have evolved from rotationally invariant embedding methods like SchNet [53] to schemes incorporating bond and dihedral angles [26,25], scalarization techniques [52,63], and spherical tensor-product frameworks such as E(3) and SE(3)-Transformers [57,28,23,41]. Recent refinements, including Gaunt Tensor Product [44] eSCN convolutions [47,42], primarily focus on enhancing computational efficiency.
In this work, we specifically focus on spherical-equivariant EGNN architectures [57,23,41], which leverage spherical harmonics and Clebsch-Gordan tensor products. These models-commonly referred to as spherical EGNNs-have demonstrated state-of-the-art accuracy, especially for periodic systems where symmetry constraints are critical [59,10]. By encoding higher-order geometric correlations through irreducible representations (irreps) with angular momentum L > 1, spherical EGNNs offer expressive, data-efficient models capable of capturing complex geometric interactions [57,55]. Unfortunately, these gains come at a computational cost. The use of spherical tensor products for feature construction incurs complexity driven by two factors: (i) the number of tensor products required, which scales with the number of edges |E| in the molecular graph, and (ii) the computational cost of each tensor product, which grows with the angular momentum cutoff L. Together, these lead to runtime costs of O(|E|L 6 ) or O(|E|L 3 ) when implemented with the sparse eSCN convolution. This scaling presents a significant bottleneck, limiting the use of spherical EGNNs to small-or mediumscale systems, despite their improved performance in principle. While recent spherical-scalarization methods [52,63,2] offer efficient alternatives by bypassing tensor products, they sacrifice theoretical completeness [18]. Tensor-product formulations, in contrast, preserve the full space of equivariant functions between irreps. This trade-off motivates our effort to retain the expressive power of tensor products while eliminating their prohibitive complexity.
Here, we introduce the Wigner 6j convolution (Wigner 6j Conv, Figure 1), a spherical-equivariant method that uses Wigner 6j symbols [40,45,20], provably reducing tensor product complexity to O(|V|) while maintaining the exact expressive power and rotational equivariance.
this section cite: ['b30', 'b35', 'b37', 'b38', 'b8', 'b63', 'b29', 'b36', 'b55', 'b4', 'b3', 'b15', 'b52', 'b25', 'b24', 'b6', 'b22', 'b40', 'b52', 'b25', 'b24', 'b51', 'b62', 'b56', 'b27', 'b22', 'b40', 'b43', 'b46', 'b41', 'b56', 'b22', 'b40', 'b58', 'b9', 'b56', 'b54', 'b51', 'b62', 'b1', 'b17', 'b39', 'b44', 'b19']

Section: Contributions.
Our contributions can be summarized as follows: (1) We introduce the Wigner 6j convolution, a spherical-equivariant technique that reduces the computational complexity from O(|E|) to O(|V|) , enabling the modeling of larger molecular systems without compromising the network's expressive power or symmetry properties. As shown in Figure 2(b), our model demonstrates better scaling behavior than the SO(3) convolution, achieving 7x to 30x speed-up given the sparsity of the molecular graph. (2) We propose E2Former, an equivariant and efficient Transformer architecture specifically designed for scalable molecular modeling. E2Former leverages the Wigner 6j convolution to maintain rotational equivariance while significantly enhancing computational efficiency. (3) Extensive experiments on benchmark datasets like OC20, OC22, and SPICE show that E2Former achieves competitive accuracy in predicting molecular energies and forces, while offering improved efficiency and scalability over existing spherical-equivariant methods. (4) Finally, we pre-trained E2Former on a large-scale dataset and evaluated its performance in molecular dynamics simulations, where it achieves high accuracy with faster speed, outperforming state-of-the-art empirical potential methods and EGNNs. These results suggest its potential to advance large-scale molecular simulations and to serve as a foundational model for machine learning force fields.
this section cite: ['b0', 'b1']

Section: Background and Preliminaries
Notation. Throughout this paper, we use ℓ and m to denote angular momentum quantum numbers associated with spherical harmonics Y (ℓ) m , where ℓ ≥ 0 and -ℓ ≤ m ≤ ℓ. All spherical harmonics are considered real-valued functions on R 3 . Positions of nodes in R 3 are represented as r, while node-level irreducible features are denoted h i ∈ R s×c , where s represents the spherical dimension and c the feature dimension per spherical component (i.e. number of channels). The operation [...] (ℓ) is the projection operation which extracts only the ℓ-th order irreducible component from a representation or tensor product. Clebch-Gorden Tensor products of irreps are symbolized by ⊗ (without any superscripts), and its Wigner 6j counterpart is denoted by ⊗ 6j .
In this section, we establish the mathematical foundations necessary for constructing the Wigner-6j Convolution. These include real-space solid spherical harmonics, tensor products of irreducible representations (irreps), and Wigner 6j recoupling theory. We commence by defining the solid spherical harmonics in real basis, which is commonly used in modern ML applications: Definition 2.1 (Solid Spherical Harmonics in Real Basis). Let r = (x, y, z) ∈ R 3 , r = ∥r∥ =
x 2 + y 2 + z 2 , and (r, θ, ϕ) be the spherical coordinates with: θ = arccos z r , ϕ = atan2(y, x).
this section cite: []

Section: Binomial Local Expansion
(b) CG CG CG 6j CG CG Tensor Product Count Equivalent Rearrange, Factorization and Wigner 6j Recouping Wigner 6j Conv (a) Irreducible Representation Spherical Harmonics Edge-Level Edge-Level Node-Level Source Dependent Target Dependent Edge Dependent Overview of the Framework SO(3) Conv. Wigner 6j Recouping L = 2 L = 1 L = 1 L = 1 L = 1 c m l m1 m2 m3 m4 mi Good Scalability CG Solid Sph. Harmonics "Bionomial Theorem" in Tensor Space The (regular) solid spherical harmonics are homogeneous harmonic polynomials of degree ℓ defined by:
R (ℓ) m (r) = r ℓ Y (ℓ) m (r), r = r/r, ℓ ≥ 0, -ℓ ≤ m ≤ ℓ,where
Y (ℓ) m are real spherical harmonics on S 2 . Equivalently, in (r, θ, ϕ), R (ℓ) m (r, θ, ϕ) =        k (ℓ) m r ℓ P (ℓ) m (cos θ) cos(mϕ), m > 0, k (ℓ) 0 r ℓ P (ℓ) 0 (cos θ), m = 0, k (ℓ) |m| r ℓ P (ℓ) |m| (cos θ) sin(|m|ϕ), m < 0, with P (ℓ)
|m| the associated Legendre polynomials and k (ℓ) m normalization constants (chosen per the convention used in this work). For example, in e3nn implementation, the real spherical harmonics for ℓ = 2 take the following form in Cartesian coordinates: R (2)
-2 (x, y, z) = xy, R (2) -1 (x, y, z) = yz, R (2) 0 (x, y, z) = 3z 2 -(x 2 + y 2 + z 2 ), R (2) 1 (x, y, z) = xz, R (2) 2 (x, y, z) = x 2 -y 2 .
Under a rotation g ∈ SO(3), these functions transform according to: R (ℓ)
m (r) → (ℓ) m ′ =-ℓ D (ℓ) m,m ′ (g) R (ℓ) m ′ (r), where D (ℓ)
m,m ′ (g) are the Wigner D-matrices. This transformation rule ensures that spherical harmonics of fixed degree ℓ transform properly under the action of SO(3). One useful property of solid spherical harmonics that will come in useful later is R (1) (r ij ) = R (1) (r i ) -R (1) (r j ). It is also worth noting that this equality holds universally if and only if ℓ = 1.
this section cite: []

Section: Next, we introduce the behavior of irreducible representations (irreps).
A key principle is that the tensor product of two irreps is generally reducible, meaning it decomposes into a direct sum of other irreps. This decomposition mechanism is precisely what will allow us to relate the general irrep R (ℓ) (•) back to R (1) (•).
this section cite: []

Section: Definition 2.2 (Tensor Products of Irreps). Let U (ℓ1) and U (ℓ2) be irreducible representations (irreps) of SO(3).
Their tensor product U (ℓ1) ⊗U (ℓ2) decomposes into a direct sum of irreps: U (ℓ1) ⊗U (ℓ2) = ℓ1+ℓ2 ℓ=|ℓ1-ℓ2| U (ℓ) . The decomposition is governed by the Clebsch-Gordan coefficients. Specifically, the tensor product, projected onto a specific irreducible component U (ℓ3) , is denoted as:
U (ℓ1) ⊗ U (ℓ2) (ℓ3) = ℓ1m1,ℓ2m2 C ℓ3m3 ℓ1m1,ℓ2m2 U (ℓ1) m1 U (ℓ2) m2 .
(2.1)
Commutativity of the Clebsch-Gordan Tensor Product. The tensor product of two irreducible representations (irreps) U (a) and U (b) of SO(3) is not strictly commutative as a bilinear operation on vector spaces: a) . Nonetheless, this operation is effectively commutative at the level of irreducible decompositions. Interchanging the order of the factors does not change the set of irreps that appear, although it permutes the corresponding CG coefficients. In particular, we have:
U (a) ⊗ U (b) is not identical to U (b) ⊗ U (
U (a) ⊗ U (b) ∼ = U (b) ⊗ U (a) ∼ = a+b j=|a-b| U (j) . (2.2)
Associativity of the Clebsch-Gordan Tensor Product. For irreducible representations (irreps) of SO(3), the tensor product is associative up to a canonical isomorphism. Specifically, for any three irreps U (a) , U (b) , and U (c) , the following holds:
(U (a) ⊗ U (b) ) ⊗ U (c) ∼ = U (a) ⊗ (U (b) ⊗ U (c) ).
While the set of resulting irreps is independent of the association order, the CG coefficients that appear in the decomposition do depend on the chosen coupling scheme. Transitions between different coupling orders are governed by Wigner 6j symbols, which express changes of basis without modifying the underlying irreducible content.
Definition 2.3 (Wigner 6j Symbol). For three irreps U (a) , U (b) , and U (c) of SO(3), one can couple c) . The Wigner 6j symbol a b d c ℓ j relates these two coupling schemes through the identity: c) .
them either as U (a) ⊗ (U (b) ⊗ U (c) ) or as (U (a) ⊗ U (b) ) ⊗ U (
U (a) ⊗ U (b) ⊗ U (c) (j) (ℓ) = d (-1) a+b+c+d (2d + 1)(2j + 1) a b d c ℓ j U (a) ⊗ U (b) (d) ⊗U(
(2.3)
To simplify notation, we abstract the recoupling process as follows:
U (a) ⊗ (U (b) ⊗ U (c) ) = (U (a) ⊗ U (b) ) ⊗ 6j U (c) ,(2.4)
where ⊗ 6j denotes a CG tensor product accompanied by a re-indexing via Wigner 6j coefficients.
this section cite: []

Section: Wigner 6j Convolution
In this section, we introduce the SO(3)-Equivariant Node convolution and demonstrate how Wigner 6j recoupling facilitates an efficient node-wise computation.
this section cite: []

Section: Definition 3.1 (SO(3)-Equivariant Node Convolution).
Let h i ∈ R s×c denote the irreducible feature tensor of node i, where s indexes the irreducible representation (irrep) type and c indexes the channels within each irrep. Let R (ℓ) (r ij ) denote the degree-ℓ spherical harmonic evaluated at the relative direction. The SO(3)-equivariant node convolution is via the CG tensor product between the source irreps and the spherical harmonics:
h i := j∈N (i) h j ⊗ R (ℓ) (r ij ).
We clarify that our formulation of the SO(3) convolution employs R(•) rather than Y (•). The two formulations are related through a normalization factor. To realize the Y (•)-based variant, this normalization factor can be absorbed into the attention coefficients, as detailed in Alg. 1.
Wigner 6j convolution. Given the SO(3) convolution, we aim to demonstrate that the operation admits a node-wise factorization via Wigner 6j symbols. In particular, we show that the SO(3) convolution can be expressed as:
hi = j∈N (i) hj ⊗ R (ℓ) (rij) ij-dependent = ℓ u=0 (-1) ℓ-u ℓ u R (u) (ri) i-dependent ⊗ 6j j∈N (i) hj ⊗ R (ℓ-u) (rj) j-dependent .
The blue-boxed factors R (u) (⃗ r i ) aggregate all node-i-specific terms, whereas the red-boxed factors h j , R (ℓ-u) (⃗ r j ) isolate the node-j contribution. This separation removes explicit edge dependencies, resulting in the number of tensor products in the network scaling with O(|V |). To build further intuition, we draw an analogy to factorization techniques in kernelized attention mechanisms [13], which achieve linear scaling by decoupling query-key interactions.
To establish this result, we introduce the concept of the Binomial Local Expansion. The expansion is based on the key insight that any term R (ℓ) (•) of arbitrary order ℓ can be expressed through iterative tensor products of the first-order term, R (1) (•). This effectively reduces the problem to the first-order case, where we can apply the previously introduced relation R (1) (r ij ) = R (1) (r i ) -R (1) (r j ) to factor the edge-dependent expression into node-local terms.
this section cite: ['b12']

Section: Theorem 3.2 (Bionomial Local Expansion).
Let ℓ = u ≥ 1. Every ℓ = u spherical harmonic R (l) (r ij ) can be expressed as an irreducible subspace of the u-fold tensor product (R (1) (r ij )) ⊗u . When expanded in terms of node-local terms, this satisfies:
R (ℓ) (r ij ) = ℓ u=0 (-1) ℓ-u ℓ u R (u) (r i ) ⊗ R (ℓ-u) (r j )(ℓ)
, Proof Sketch. This spherical harmonic R (ℓ) (r ij ) could be constructed by projecting the ℓ-fold tensor product of the first-order harmonic R (1) (r ij ) onto the subspace transforming as the irreducible representation (irrep) ℓ of SO(3). Recall that the projection operator is denoted by [...] (ℓ) and using the identity R (1) (r ij ) = R (1) (r i ) -R (1) (r j ), the objective could be rewritten as R (ℓ) (r ij ) = [(R (1) (r i ) -R (1) (r j )) ⊗ℓ ] (ℓ) .
We begin by expanding the tensor power (R (1) (r i ) -R (1) (r j )) ⊗ℓ , which produces a sum of 2 ℓ tensor products. Each term corresponds to an ordered sequence P ∈ {i, j} ℓ , where each factor is either R (1) (r i ) or R (1) (r j ). Denote the corresponding tensor product as T P . For example, if P = (i, j, i), then T P = R (1) (r i ) ⊗ R (1) (r j ) ⊗ R (1) (r i ). Initially, each ordering ( * , * , • • • , * ) defines a distinct term. Later, we will show that the projection operator renders the result invariant to the ordering. We write the full expansion as: (R (1) (r i )-R (1) (r j )) ⊗ℓ = ℓ u=0 (-1) ℓ-u P ∈Πu T P , where Π u denotes the set of orderings containing exactly u factors of R (1) (r i ) and ℓ -u factors of R (1) (r j ). Applying the linear projection operator [...] (ℓ) to this sum distributes the operator yields [(R (1)  ℓ) . The key insight comes from angular momentum coupling theory. Combining ℓ systems with angular momentum 1 yields components with total angular momentum ranging up to ℓ. The subspace associated with the highest possible angular momentum, L = ℓ, is unique and corresponds to the fully symmetric combination of the individual factors. The projector [...] (ℓ) isolates precisely this unique, symmetric component. As a result, the projected tensor [T P ] (ℓ) remains identical for all orderings P ∈ Π u , indicating that the projection depends solely on the multiplicities of the factors R (1) (r i ) and R (1) (r j ) in T P , rather than their ordering. The inner sum over the ℓ u identical projected terms simplifies. Let T rep = (R (1) (r i )) ⊗u ⊗ (R (1) (r j )) ⊗(ℓ-u) serve as a representative tensor product for the class Π u . Then: ℓ) . Substituting this simplification back into the expression for the projected tensor power yields: [(R (1)
(r i ) -R (1) (r j )) ⊗ℓ ] (ℓ) = ℓ u=0 (-1) ℓ-u P ∈Πu [T P ] (
P ∈Πu [T P ] (ℓ) = |Π u |[T rep ] (ℓ) = ℓ u [(R (1) (r i )) ⊗u ⊗ (R (1) (r j )) ⊗(ℓ-u) ] (
(r i ) -R (1) (r j )) ⊗ℓ ] (ℓ) = ℓ u=0 (-1) ℓ-u ℓ u [(R (1) (r i )) ⊗u ⊗ (R (1) (r j )) ⊗(ℓ-u) ] (ℓ) .
this section cite: []

Section: Theorem 3.3 (Node-Based Factorization via Wigner 6j
). SO(3) convolutions admit a factorization that separates the dependence on the central node i from the aggregation over neighbors j, yielding the form:
j∈N (i) h j ⊗ R (ℓ) m (r ij ) = ℓ u=0 (-1) ℓ-u ℓ u R (u) (r i ) ⊗ 6j   j∈N (i) h j ⊗ R (ℓ-u) (r j )   ,
where ⊗ 6j denotes a CG tensor product where the path weight is parameterized by the corresponding Wigner 6j coefficients.
Proof Sketch. We begin by substituting the spherical harmonic R (ℓ) m (r ij ) using the binomial local expansion from into the original SO(3)-equivariant convolution expression, we obtain:
j∈N (i) h j ⊗ ℓ u=0 (-1) ℓ-u ℓ u R (u) (r i ) ⊗ R (ℓ-u) (r j )(ℓ)
.
(3.1) By linearity of the tensor product, this expression becomes:
ℓ u=0 (-1) ℓ-u ℓ u j∈N (i) h j ⊗ R (u) (r i ) ⊗ R (ℓ-u) (r j )(ℓ)
.
(3.2)
To reorganize this expression in terms of node-dependent features, we apply Wigner 6j recoupling.
Letting A = h j , B = R (ℓ-u) (r j ), C = R (u) (r i ),
The recoupling identity states: A ⊗ (B ⊗ C) = (A⊗B)⊗ 6j C. Since CG tensor products commute effectively, we can swap B and C before applying recoupling. This gives:
h j ⊗ R (u) (r i ) ⊗ R (ℓ-u) (r j ) = (h j ⊗ R (ℓ-u) (r j )) ⊗ 6j R (u) (r i ).(3.3)
Applying this recoupling within the sum, we arrive at the factorized form as claimed. Note that it is safe to apply the recoupling within a projection operator. This constraint can be implemented by fixing one of the intermediate coupling indices in the Wigner 6j symbol to ℓ.
We now formalize the key properties of the resulting Wigner 6j convolution. These results are stated in the following lemmas. Proofs are provided in Appendix E and Appendix F, respectively.
this section cite: []

Section: Lemma 3.4 (Equivariance of Wigner 6j Convolution).
The Wigner 6j convolution operator (denoted as F ), is equivariant under the Euclidean group SE(3). That is, for any rigid transformation g ∈ SE(3), the output satisfies
F[g • f ] = D(g) • F[f ].
this section cite: []

Section: Lemma 3.5 (Time Complexity of Wigner 6j Convolution).
The time complexity of Wigner 6j convolution is O((L 6 C + C 2 L 2 )|V|), where L is the degree cutoff and C is the number of channels.
this section cite: []

Section: Model Architecture.
Based on these, we propose E2Former, a modular architecture that alternates between E2Attention and feed-forward layers (Appendix Fig. 5(a); additional architectural details provided in the Appendix H). At its core lies a convolution layer based on the Wigner 6j convolution, which serves as a backend kernel to efficiently capture rotational symmetries. We highlight two key design considerations underlying E2Former. First, we observe that the attention computation constitutes only a small fraction of the overall runtime in an attention-based SO(3) convolution (Figure 2 (a)). Leveraging this, we integrate the attention mechanism directly into the Wigner 6j convolution (Algorithm 1). As a result, while the resulting model is not strictly linear, the number of tensor product operations scales linearly with input size, enabling efficient computation. Second, E2Former is not defined solely by its use of Wigner 6j convolutions. Rather, it represents a broader architectural principle that combines symmetry-aware design with practical engineering. The efficiency gains afforded by the Wigner 6j kernel allow us to reallocate the computational budget toward increased expressivity-e.g., by incorporating deeper layers, wider hidden dimensions, more attention heads, and MACE higher-order interactions [6]. This trade-off between symmetry-driven modeling and architectural scalability is central to the design philosophy of E2Former. 4 Results
this section cite: ['b5']

Section: Scaling Analysis of Wigner 6j Conv and SO(3) Conv
Here, we compare the runtime of Wigner 6j convolution (purple squares) and SO(3) convolution (blue circles). The two implementations are mathematically equivalent and, by construction, produce identical outputs given the same molecular graphs. Specifically, we compare different graph sizes N , maximum angular momenta L max , and connectivity patterns. For dense graphs, defined as graphs where every node is connected to all other nodes, at L max = 3 (Fig. 2 (b.i)), the quadratic scaling of SO(3) convolution introduces a noticeable performance gap. Additionally, for sparse graphs, defined here as graphs with k-nearest neighbor connectivity (k = 32), at L max = 2 and L max = 3 (Figs. 2 (b.ii) and (b.iii)), Wigner 6j convolution scales consistently better than the SO(3) convolution. Fig. 2c further shows the impact of increasing the angular momentum cutoff up to L max = 6 on 1000-node graphs, where our method consistently achieves approximately a 7× speed-up over the baseline. Finally, Fig. 2d demonstrates that as the number of neighbors per node increases from 64 to 512 (with L max = 3), the speed-up from Wigner 6j convolution becomes even more pronounced.
this section cite: []

Section: E2Former Results
We evaluate E2Former which heavily utilizes the Wigner 6j convolution on three standard benchmarks-two catalysis datasets (OC20, OC22) and a molecular conformer dataset (SPICE)-and find that it achieves strong accuracy while maintaining computational efficiency.
this section cite: []

Section: Performance on the OC20 Dataset
Dataset Description. The OC20 dataset [10] comprises 1.2 million DFT relaxations computed using the revised Perdew-Burke-Ernzerhof (RPBE) functional [29]. Each system, averaging 73 atoms, represents an adsorbate molecule on a catalyst surface and is designed for the Structure-to-Energyand-Forces (S2EF) task. This task involves predicting the system's energy and per-atom forces, with performance evaluated based on the mean absolute error (MAE) of these predictions. Following [27,41], we use the 2M subset for training, and evaluate on the validation split. This choice also reflects practical computational constraints, as training on the full dataset requires significant time and resources. All reported results are taken directly from previous publications [27,41]. We compare two model variants: the 33M-parameter version and the 67M-parameter version. A summary of the results is presented in Table 1. E2Former demonstrates strong performance across all model sizes, with 67M variant achieving results comparable to state-of-the-art methods. Notably, the Small variant (33M parameters) maintains competitive accuracy while offering significant computational advantages.
this section cite: ['b9', 'b28', 'b26', 'b40', 'b26', 'b40']

Section: Performance on the OC22 Dataset
The OC22 dataset [59] is specifically designed for studying oxide electrocatalysis. In contrast to OC20, OC22 features DFT total energies, which could serve as a general and versatile DFT surrogate, enabling investigations beyond adsorption energies. We train on the OC22 S2EF-Total task and measure energy and force MAE on the S2EF-Total validation splits. Table 2 summarizes our results on the OC22 S2EF task. E2Former achieves competitive energy and force MAEs while enabling rapid training and inference. Notably, it converges in just 1,500 GPU hours-only one-third of the runtime required by the SOTA model.
this section cite: ['b58']

Section: Performance on the SPICE Dataset
The SPICE dataset [19] comprises small organic molecules and encompasses a diverse array of chemical species with neutral formal charges. The geometries were generated through molecular dynamics simulations using classical force fields, followed by the sampling of various conformations. High-fidelity labeling was achieved at the ωB97M-D3(BJ)/def2-TZVPPD level of calculations. This dataset includes configurations of up to 50 atoms. It was further augmented with larger molecules, ranging from 50 to 90 atoms, derived from the QMugs dataset [32], as well as water clusters obtained from simulations of liquid water. Approximately 85% of the SPICE dataset was used for model training, while 15% was allocated for model testing. We evaluate E2Former 33M on the SPICE dataset and a summary of the results is provided in Table 3. E2Former achieved state-of-the-art performance in most subsets, particularly in datasets with ample data, such as PubChem and DEShaw370-Dimers. Furthermore, compared to the MACE-Large model, E2Former achieves approximately a fivefold increase in training speed, thereby further validating its efficiency.
this section cite: ['b18', 'b31']

Section: Meomory and Efficiency Scaling
To further probe efficiency, we evaluated computational performance in an even more extreme case: system scale. We benchmarked E2Former against MACE-Large and EquiformerV2 on systems containing up to 6,400 atoms (Table 4). E2Former consistently achieved the highest throughput, with its performance advantage becoming clearer as system size increased. At 3,200 atoms, E2Former processes data nearly three times faster than MACE-Large. Crucially, E2Former was the only model capable of handling simulations at the 6,400-atom scale, a size at which both MACE-Large and EquiformerV2 failed.  We began by pretraining E2Former on a large inhouse dataset derived from DeShaw (2M) [15] and GEMS [60] (2.7M), constituting a foundational model on machine-learning force field.
this section cite: ['b14', 'b59']

Section: Small-scale Amino Acid Systems
To evaluate the model, we first performed an NVT (T = 300 K) simulation of an amino acid wrapped by water molecules (The structure is shown in Fig. 4(a)), totaling 253 atoms. This evaluation was conducted on a system equipped with a single NVIDIA A100 GPU and an AMD EPYC 7V13 24-core CPU. Over the course of 10,000 simulation time steps (1fs per step), we compared the trajectory 's power spectrum obtained from E2Former, CUDA-accelerated DFT [34]-namely, the MADFT software and state-of-the-art empirical potential methods GFN2-xTB [3]. Fig. 4(a) illustrates the results. The evaluation demonstrates that E2Former exhibits long-term stability in molecular simulations. The power spectrum shows that E2Former predictions align closely with those of the DFT baseline. In contrast, empirical method shows significant devia-tions in high-frequency regions, particularly near 3,000 and 3,500 frequencies. These high-frequency components are critical as they provide insights into bond vibrations [14] and molecular stability [12], underscoring the ability of E2Former to effectively extrapolate across the molecular potential energy surface. To assess computational efficiency, we compare runtime across methods, which is depicted in Fig. 4(b). E2Former achieves a computational speed approximately 1,000 times faster than DFT, and around 2 times faster than GFN2-xTB.
this section cite: ['b33', 'b2', 'b13', 'b11']

Section: Large-Scale 6000-atom Water Cluster
0 500 1000 1500 2000 2500 3000 3500 4000 0 20 40 60 80 100 Power (cm -1 ) Frequency E2Former MACE MADFT Water Cluster Power Spectrum To evaluate accuracy and efficiency at larger scales, we tested our model on a 6,000-atom water cluster by analyzing atomic vibration patterns. E2Former closely reproduced the reference DFT results, accurately capturing key spectral features: low-frequency modes (below 1000 cm -1 ), the H-O-H bending mode (1650 cm -1 ), and O-H stretching vibrations. In contrast, MACE-Large exhibited larger deviations, particularly for high-frequency stretching modes. We further validated our approach on a more complex system-a Chignolin peptide solvated in water (approximately 2,000 atoms, Fig 6)-successfully optimizing its structure while maintaining high force prediction accuracy (0.484 kcal/mol/Å error).
this section cite: []

Section: Related Work
Invariant GNNs. Invariant geometric GNNs have driven state-of-the-art performance in predicting molecular and crystalline properties [53,50,11,26,43,25,62,48] and have been instrumental in advancing protein structure prediction [35].
this section cite: ['b52', 'b49', 'b10', 'b25', 'b42', 'b24', 'b61', 'b47', 'b34']

Section: Cartesian Equivariant GNNs.
Building on invariance, Cartesian equivariant GNNs explicitly model transformations in R 3 , offering greater flexibility. These models have shown strong empirical results in similar domains [33,51,17,54,2] and have recently evolved to include Cartesian equivariant transformer layers [22].
this section cite: ['b32', 'b50', 'b16', 'b53', 'b1', 'b21']

Section: Spherical Equivariant GNNs.
Complementing Cartesian approaches, spherical equivariant GNNs leverage spherical tensors to naturally handle rotational symmetries, relying on the representation theory of SO(3). Recent advancements include SO(3)and SE(3)-equivariant transformer layers [23,41], efficient interatomic potential calculations [7,6,46], and optimizations that reduce convolutions in SO(3) to SO(2) [47]. These improvements have enabled strong performance in diverse applications, including geometry, physics, and chemistry [57], dynamic molecular modeling [1], and fluid mechanical modeling [58].
this section cite: ['b22', 'b40', 'b6', 'b5', 'b45', 'b46', 'b56', 'b0', 'b57']

Section: Conclusion and Future Work
We introduced E2Former, an efficient and scalable Transformer architecture for molecular modeling. By leveraging the Wigner 6j convolution, E2Former shifts computation from edges to nodes, reducing complexity from O(|E|) to O(|V|) while maintaining rotational equivariance and expressive power. E2Former demonstrated competitive performance across OC20, OC22, and SPICE benchmarks with significantly improved computational efficiency. Its scalability makes it suited for large-scale applications in biology, drug discovery, and materials science.
Future work could focus on optimizing E2Former for hardware accelerators and integrating kernelized Euclidean attention [21]. Combining Wigner-6j Conv with SO(2) convolution could further bolster the model's efficiency. Scaling to the real-world all-atom protein systems will also be investigated. Maximum angular momentum (highest spherical degree). ℓ ≥ 0, m ∈ {-ℓ, . . . , ℓ} Angular momentum quantum numbers for spherical harmonics. R (ℓ) m (r)
this section cite: ['b20']

Section: References
Ref_id:b0 Title: Cormorant: Covariant molecular neural networks Year: (2019)
Ref_id:b1 Title: GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks Year: (2025)
Ref_id:b2 Title: Gfn2-xtb-an accurate and broadly parametrized self-consistent tight-binding quantum chemical method with multipole electrostatics and density-dependent dispersion contributions Year: (2019)
Ref_id:b3 Title: On representing chemical environments Year: (2013)
Ref_id:b4 Title: Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons Year: (2010)
Ref_id:b5 Title: MACE: Higher order equivariant message passing neural networks for fast and accurate force fields Year: (2022)
Ref_id:b6 Title: E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials Year: (2022)
Ref_id:b7 Title: Angular momentum in quantum physics: theory and application Year: (1984)
Ref_id:b8 Title: Making connections between ultrafast protein folding kinetics and molecular dynamics simulations Year: (2011)
Ref_id:b9 Title: Open catalyst 2020 (oc20) dataset and community challenges Year: (2021)
Ref_id:b10 Title: Graph networks as a universal machine learning framework for molecules and crystals Year: (2019)
Ref_id:b11 Title: Molecular dynamics simulations reliably identify vibrational modes in far-ir spectra of phospholipids Year: (2024)
Ref_id:b12 Title: Rethinking attention with performers Year: (2020)
Ref_id:b13 Title: Vibrational spectroscopy by means of first-principles molecular dynamics simulations Year: (2022)
Ref_id:b14 Title: Quantum chemical benchmark databases of gold-standard dimer interaction energies Year: (2021)
Ref_id:b15 Title: Atomic cluster expansion for accurate and transferable interatomic potentials Year: (2019)
Ref_id:b16 Title: Se (3) equivariant graph neural networks with complete local frames Year: (2022)
Ref_id:b17 Title: On the universality of rotation equivariant point cloud networks Year: ()
Ref_id:b18 Title: Spice, a dataset of drug-like molecules and peptides for training machine learning potentials Year: (2023)
Ref_id:b19 Title: Angular momentum in quantum mechanics Year: (1996)
Ref_id:b20 Title: Euclidean fast attention: Machine learning global atomic representations at linear cost Year: (2024)
Ref_id:b21 Title: So3krates-self-attention for higherorder geometric interactions on arbitrary length-scales Year: (2022)
Ref_id:b22 Title: Se (3)-transformers: 3d roto-translation equivariant attention networks Year: (1970)
Ref_id:b23 Title: Representation theory: a first course Year: (2013)
Ref_id:b24 Title: Gemnet: Universal directional graph neural networks for molecules Year: (2021)
Ref_id:b25 Title: Directional message passing for molecular graphs Year: (2020)
Ref_id:b26 Title: Gemnet-oc: developing graph neural networks for large and diverse molecular simulation datasets Year: (2022)
Ref_id:b27 Title:  Year: (2022-04)
Ref_id:b28 Title: Improved adsorption energetics within density-functional theory using revised perdew-burke-ernzerhof functionals Year: (1999)
Ref_id:b29 Title: Inhomogeneous electron gas Year: (1964)
Ref_id:b30 Title: Reaction mechanisms of aqueous monoethanolamine with carbon dioxide: a combined quantum chemical and molecular dynamics study Year: (2015)
Ref_id:b31 Title: Qmugs, quantum mechanical properties of drug-like molecules Year: (2022)
Ref_id:b32 Title: Learning from protein structure with geometric vector perceptrons Year: (2020)
Ref_id:b33 Title: Acceleration without disruption: Dft software as a service Year: (2024)
Ref_id:b34 Title: Highly accurate protein structure prediction with alphafold Year: (2021)
Ref_id:b35 Title: Electron-phonon coupling in semiconductors within the gw approximation Year: (2018)
Ref_id:b36 Title: Self-consistent equations including exchange and correlation effects Year: (1965)
Ref_id:b37 Title: Quantum vibronic effects on the electronic properties of solid and molecular carbon Year: (2021)
Ref_id:b38 Title: Influence of nuclear quantum effects on the electronic properties of amorphous carbon Year: (2022)
Ref_id:b39 Title: Exact computation of the 3-j and 6-j symbols Year: (1990)
Ref_id:b40 Title: Equiformer: Equivariant graph attention transformer for 3d atomistic graphs Year: (2023)
Ref_id:b41 Title: Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations Year: (2023)
Ref_id:b42 Title: Spherical message passing for 3d molecular graphs Year: ()
Ref_id:b43 Title: Enabling efficient equivariant operations in the fourier basis via gaunt tensor products Year: (2024)
Ref_id:b44 Title:  Year: (2010)
Ref_id:b45 Title: Learning local equivariant representations for large-scale atomistic dynamics Year: (2023)
Ref_id:b46 Title: Reducing so (3) convolutions to so (2) for efficient equivariant gnns Year: (2023)
Ref_id:b47 Title: The importance of being scalable: Improving the speed and accuracy of neural network interatomic potentials across chemical domains Year: (2024)
Ref_id:b48 Title: Theory of complex spectra. ii Year: (1942)
Ref_id:b49 Title: Mt-cgcnn: Integrating crystal graph convolutional neural network with multitask learning for material property prediction Year: (2018)
Ref_id:b50 Title: E (n) equivariant graph neural networks Year: (2021)
Ref_id:b51 Title: Equivariant message passing for the prediction of tensorial properties and molecular spectra Year: (2021)
Ref_id:b52 Title: Schnet-a deep learning architecture for molecules and materials Year: (2018)
Ref_id:b53 Title: Tensornet: Cartesian tensor representations for efficient learning of molecular potentials Year: (2024)
Ref_id:b54 Title: Finding symmetry breaking order parameters with euclidean neural networks Year: (2021-01)
Ref_id:b55 Title: Modern quantum chemistry: introduction to advanced electronic structure theory Year: (2012)
Ref_id:b56 Title: Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds Year: (2018)
Ref_id:b57 Title: E (3) equivariant graph neural networks for particle-based fluid mechanics Year: (2023)
Ref_id:b58 Title: The open catalyst 2022 (oc22) dataset and challenges for oxide electrocatalysts Year: (2023)
Ref_id:b59 Title: Biomolecular dynamics with machine-learned quantum-mechanical force fields trained on diverse chemical fragments Year: (2024)
Ref_id:b60 Title: Quantum theory of angular momentum Year: (1987)
Ref_id:b61 Title: ComENet: Towards complete and efficient message passing for 3d molecular graphs Year: (2022)
Ref_id:b62 Title: Visnet: a scalable and accurate geometric deep learning potential for molecular dynamics simulation Year: (2022)
Ref_id:b63 Title: Dislocation processes in the deformation of nanocrystalline aluminium by molecular-dynamics simulation Year: (2002)
