Title: On Universality Classes of Equivariant Networks
Abstract: Equivariant neural networks provide a principled framework for incorporating symmetry into learning architectures and have been extensively analyzed through the lens of their separation power, that is, the ability to distinguish inputs modulo symmetry. This notion plays a central role in settings such as graph learning, where it is often formalized via the Weisfeiler-Leman hierarchy. In contrast, the universality of equivariant models-their capacity to approximate target functions-remains comparatively underexplored. In this work, we investigate the approximation power of equivariant neural networks beyond separation constraints. We show that separation power does not fully capture expressivity: models with identical separation power may differ in their approximation ability. To demonstrate this, we characterize the universality classes of shallow invariant networks, providing a general framework for understanding which functions these architectures can approximate. Since equivariant models reduce to invariant ones under projection, this analysis yields sufficient conditions under which shallow equivariant networks fail to be universal. Conversely, we identify settings where shallow models do achieve separation-constrained universality. These positive results, however, depend critically on structural properties of the symmetry group, such as the existence of adequate normal subgroups, which may not hold in important cases like permutation symmetry.

Section: Introduction
Equivariant neural networks offer a principled framework to incorporate symmetry into learning architectures, attracting sustained attention for both their empirical successes and theoretical richness [1][2][3][4][5]. While their separation power-the capacity to distinguish inputs up to symmetry-has been extensively studied, comparatively less is understood about their approximation capabilities.
In classical approximation theory, expressivity is often characterized via universality-the capacity of a model class to approximate any target function within a given function space to arbitrary precision [6,7]. In the equivariant setting, however, this notion must be refined. This is because such models treat symmetric inputs as indistinguishable; they can only approximate functions compatible with the underlying symmetry, subject additionally to spurious constraints arising from the imperfect interactions between equivariance and linear inductive biases on neural network layers. In this context, universality becomes inherently relative-defined with respect to a particular separation relation that circumscribes the model's ability to distinguish inputs.
Graph learning has served as a primary testbed for studying invariant and equivariant architectures, where models are typically required to respect node permutation symmetries [4,8,9]. Within this setting, separation power is most commonly assessed using the Weisfeiler-Leman (WL) test [10] or homomorphism counting techniques [11]. A whole range of architectures-including Graph Neural Networks (GNNs) [12][13][14], Invariant Graph Networks (IGNs) [4,15], and subgraph-based models [16,9]-have been analyzed through this lens. More recently, investigations analyzing separation power have been extended beyond graph-structured data to broader classes of equivariant models [17,18]. While much of the literature in geometric deep learning has centered on separation as the primary metric of expressivity, recent work [19,20] has called for a more comprehensive view that includes approximation capabilities more generally. The role of equivariant layers in determining approximation power remains underexplored. Typically, these are composed to increase the model's separation capacity, while a universal component-such as a multilayer perceptron with adjustable width-is appended to approximate functions within the separation-constrained class. As a result, universality is achieved only relative to the distinctions introduced by the equivariant backbone. However, there is no general theory describing how equivariant layers themselves contribute to approximation.
To address this gap, we examine the approximation capabilities of equivariant neural networks beyond what is captured by separation constraints alone. For this purpose, it suffices to focus on invariant architectures, as the core phenomena extend to the equivariant setting. Indeed, projecting the output of an equivariant model onto the trivial representation yields an invariant network. Accordingly, our analysis of invariant networks provides insight into the approximation limits of a broad class of equivariant architectures. We begin by showing that invariant neural networks can be expressed as function that vanishes on certain differential operators (Section 5.1). This formulation allows us to derive sufficient conditions under which a shallow invariant network fails to be universal within the class of separation-constrained continuous functions (Section 5.2).
Our theory and analysis leads to three key insights. First, remarkably, we identify network families that possess identical separation power yet differ in their approximation capabilities-demonstrating that separation alone does not fully characterize expressivity. In particular, we show that shallow networks composed of commonly used equivariant layers-such as PointNets and CNNs with filter width 1-fail to be universal, despite matching the separation power of permutation-invariant continuous functions (Section 6.1). Second, this implies that the only two architectural choices that impact approximation power are depth and the type of hidden representations, the latter being strongly influenced by the structure of the symmetry group. Third, we show that a generalization of the results by [5] produces a broad family of shallow models that are universal within the separation-constrained function class (Section 6.2). However, these constructions fundamentally rely on the structure of the symmetry group. In particular, on the existence of normal subgroups of suitable size, a condition that is not always met, as is the case for key symmetry groups such as the permutation group.
We summarize the main contributions of this work as follows:
• We characterize the universality classes of shallow invariant networks (Theorem 13).
• We establish general sufficient conditions under which universality fails, even within function classes exhibiting maximal separation (Theorem 14 and Theorem 15).
• Leveraging these results, we construct explicit examples of invariant models that attain maximal separation yet fail to be universal, demonstrating that separation is not sufficient to guarantee universality (Proposition 16).
• We generalize the results by Ravanbakhsh [5] to a broader family of models (Theorem 18).
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b3', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b3', 'b14', 'b15', 'b8', 'b16', 'b17', 'b18', 'b19', 'b4', 'b4']

Section: Related Work
Classical approximation theory for neural networks has established foundational results for shallow architectures with sigmoidal activations [6,7,21]. Necessary and sufficient conditions on activation functions were later given by Leshno et al. [22], and further refinements appear in Pinkus [23]. For general treatments of approximation theory in modern neural networks, we refer to [24,25].
Moving beyond shallow networks, Yarotsky [26,27] proved fundamental results on the approximation rates of deep neural networks, while Siegel [28] derived sharp bounds for deep ReLU networks. These results establish that deep networks are not only universal under mild assumptions but also more parameter-efficient than their shallow counterparts, approximating complex functions with significantly fewer parameters.
Equivariant neural networks offer a principled way to encode symmetry into learning architectures [1][2][3], with early applications across physics [29], chemistry [30], biology [31], and computer vision [32].
Beyond the foundational work of Yarotsky [33], universality in equivariant and invariant settings has been studied from multiple perspectives. A number of works [5,[34][35][36] establish universality for certain shallow equivariant networks using unconstrained hidden representations. Keriven and Peyré [37] extended this analysis to equivariant graph neural networks. These proofs rely on two main techniques. The first is the application of the Stone-Weierstrass theorem, or one of its variants, for instance via invariant polynomials [38,8], to establish density results in spaces of continuous functions. The second is the use of a symmetrization operator, which enforces equivariance but causes the dimension of intermediate representations to grow exponentially. However, these approaches are often impractical: the Stone-Weierstrass theorem cannot be applied directly, since the network families of interest do not form function algebras, while symmetrization leads to prohibitive computational costs. Although canonicalization methods can improve the efficiency of models derived through symmetrization [39,40], in many cases such techniques cannot be applied [41] or remain computationally inefficient.
A complementary line of work examines permutation-equivariant networks over multisets. Zaheer et al. [42], Qi et al. [43], Segol and Lipman [44] prove universality for such models under constrained hidden representations, but their results are restricted to architectures of depth three. As a result, the universality of truly shallow networks-those with depth two or less-remained unresolved.
In this work, we address this gap and show that certain shallow equivariant networks are not universal in the space of equivariant functions. This stands in contrast to the fully connected case, where universality holds generically, and highlights that depth can play a qualitatively different role in equivariant architectures, extending beyond parameter efficiency to approximation capacity itself.
To capture practical models within a theoretical framework, recent work has shifted toward studying universality up to separation. In permutation-equivariant networks, expressivity has been analyzed through the Weisfeiler-Leman (WL) hierarchy [45][46][47][48][49], with refinements based on homomorphism counts and subgraph-aware techniques [50,51]. Joshi et al. [17] extended this approach to geometric domains, deriving depth-sensitive universality results under representation and orbit separation constraints. More generally, Pacini et al. [18] recently characterized the separation power of neural networks for arbitrary finite groups and permutation representations. While these works elucidate distinguishability, they do not fully account for approximation behavior.
The role of equivariant layers in approximation, beyond their contribution to separation, remains only partially understood. In practice, such layers are often composed to enhance separation power, followed by a universal component-typically an MLP-to approximate functions within the induced separation class. In the invariant case, this composition can yield universality. In the equivariant case, however, universality is not guaranteed and is only known to hold in specific instances [44]. Thus, the expressive power of the overall model remains fundamentally limited by the separation achieved by the equivariant stack. Yet equivariant layers may also contribute directly to approximation, and in some cases are known to suffice for universality within a fixed separation class [5].
Our work provides a detailed analysis of the universality classes of shallow equivariant networks. We show that equivariant layers are not always sufficient to guarantee universality up to separation, and that separation alone is not a complete proxy for approximation. We show explicit examples of models with identical separation power but differing approximation capacity. More broadly, we introduce general techniques for comparing the approximation power of equivariant models beyond separation, offering a more refined and complete understanding of expressivity in symmetryconstrained architectures.
this section cite: ['b5', 'b6', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b0', 'b1', 'b2', 'b28', 'b29', 'b30', 'b31', 'b32', 'b4', 'b33', 'b34', 'b35', 'b36', 'b37', 'b7', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b16', 'b17', 'b43', 'b4']

Section: Preliminaries

this section cite: []

Section: Groups and Equivariance
We are interested in functions that exhibit symmetry under specified transformations. Mathematically, such symmetries are described by groups: sets of transformations closed under composition, equipped with inverses and an identity element. While group theory offers a rigorous algebraic framework for analyzing symmetry, applying these ideas within neural networks requires their reformulation in linear-algebraic terms. This translation is achieved via representation theory, which associates abstract group elements with matrix actions on vector spaces. For a brief overview, see Appendix A; for a more detailed treatment, see [52].
Our focus will be on permutation representations, which naturally arise when a group G acts on a finite set X. Let R X denote the space of real-valued functions on X. For each x ∈ X, define e x ∈ R X as the function taking value 1 at x and 0 elsewhere. The set {e x } x∈X forms a canonical basis for R X . A permutation representation of G on V = R X is a linear action satisfying g(e x ) = e gx for all g ∈ G and x ∈ X. If V and W are permutation representations of G, a map ϕ : V → W is G-equivariant if ϕ(gv) = gϕ(v) for all g ∈ G and v ∈ V . We denote by Hom(V, W ) the space of linear maps from V to W , and by Hom G (V, W ) the subspace of G-equivariant linear maps. Similarly, let Aff(V, W ) denote the space of affine maps from V to W , and Aff G (V, W ) the subspace of G-equivariant affine maps. The spaces Hom(V, W ), Aff(V, W ), and their equivariant counterparts are real vector spaces under pointwise addition and scalar multiplication. A result from Pacini et al. [53] shows that any map f ∈ Aff(V, W ) admits a unique decomposition of the form f = τ v • ϕ for some v ∈ W and ϕ ∈ Hom(V, W ), where
τ v (w) = w + v. Such a map is G-equivariant iff ϕ is G-equivariant and v ∈ W G = {v ∈ W | gv = v; ∀g ∈ G}, the fixed-point subspace of W . In particular, there is a linear morphism λ : Aff G (V, W ) → Hom G (V, W )
that projects an affine map to its linear part.
this section cite: ['b51', 'b52']

Section: Equivariant Neural Networks
With all necessary definitions in place, we now introduce the notion of an equivariant neural network. Throughout this work, we consider networks that are equivariant under the action of a finite group, using arbitrary point-wise continuous activation functions, and with layers that transform according to permutation representations. We adopt the notation introduced by Pacini et al. [18].
Definition 1 (Point-wise Activation). Let σ : R → R be a nonlinear activation function, and let R X denote a permutation representation of a group G. We define the corresponding point-wise activation
σ : R X → R X by setting σ x∈X α x e x = x∈X σ(α x )e x .
When no confusion arises, we will denote both σ and σ by the same symbol.
Definition 2 (Neural Networks and Neural Spaces). Let G be a group, and let V 0 , . . . , V d be permutation representations of G. For each i = 1, . . . , d, let M i ⊆ Aff G (V i-1 , V i ) be a set of G-equivariant affine maps. For d ≥ 2, the neural space associated with the layers M 1 , . . . , M d and a point-wise activation function σ is defined recursively by
N σ (M 1 , . . . , M d ) = ϕ d • σ • η d-1 ϕ d ∈ M d , η d-1 ∈ N σ (M 1 , . . . , M d-1 ) ,
with the base case N σ (M 1 ) = M 1 . An element η d ∈ N σ (M 1 , . . . , M d ) is called a neural network with layers in M 1 , . . . , M d and activation σ. When each M i is taken to be the full space
Aff G (V i-1 , V i ), we write N σ (V 0 , . . . , V d ) as shorthand for N σ (M 1 , . . . , M d ).
To capture architectures commonly used in practice, we adopt a more structured form for the layer spaces M ⊆ Aff G (V, R X ), as proposed in Section 4.2 of Pacini et al. [18]. Specifically, we assume that M takes the form
M =    v → k i=1 x i ϕ i (v) + ℓ j=1 y j 1 Xj x 1 , . . . , x k , y 1 , . . . , y ℓ ∈ R    ,(1)
where ϕ 1 , . . . , ϕ k span a subspace of Hom G (V, R X ), X 1 , . . . , X ℓ are the orbits of X under the G-action, and 1 Xi := x∈Xi e x for i = 1, . . . , ℓ. This formulation, while notation-heavy, plays a central role in the development of our main results.
We now present two working examples of equivariant affine maps and their associated neural spaces. These examples both reflect architectures commonly used in geometric deep learning and illustrate how standard models naturally conform to the structure in (1). They will serve as recurring reference points throughout to highlight key phenomena in the universality landscape of equivariant networks. with the graph corresponding to a standard fully connected layer, following the convention of Ravanbakhsh et al. [54] where arrows of the same color denote identical weights applied to different input values.
Example 3 (PointNets). We focus on the sum-pooling variant of PointNet architectures [43], which are designed to process unordered collections, such as point clouds, by enforcing permutation equivariance. An input configuration of n elements with f -dimensional features is represented by a tensor A ∈ R n×f , where each row corresponds to the features of a single object. Permuting the elements corresponds to permuting the rows of A, i.e., the indices along its first axis. In our framework, the input tensor A is modeled as an element of R X ⊗ R f , where X = [n] and the symmetric group G = S n acts on X via its standard action and trivially on R f . PointNet architectures operate on such inputs using layers in the space Aff Sn (R X ⊗ R fi-1 , R X ⊗ R fi ), where each R fi corresponds to a space of S n -invariant hidden features. Accordingly, the neural spaces corresponding to these equivariant architectures and their invariant counterparts take the following forms, respectively:
N σ (R X0 ⊗ R f0 , . . . , R X d ⊗ R f d ) and N σ (R X0 ⊗ R f0 , . . . , R X d-1 ⊗ R f d-1 , R f d ).
Zaheer et al. [42] showed that understanding the structure of 1b shows the colored bipartite graph corresponding to the layer space Aff Sn (R n , R n ). In the invariant case, Aff Sn (R n , R) = v → x 1 ⊤ v + y x, y ∈ R , which is consistent with the notation introduced in (1). Figure 1a shows the colored bipartite graph corresponding to the layer space Aff Sn (R n , R). Example 4 (Convolutional Neural Networks). Circular convolutional filters can be naturally formulated within the framework of permutation representations. For simplicity, we focus on the one-dimensional case. Let X = [n] and let G = Z n act on X by modular shifts. Identifying R X with R n , the space Hom Zn (R n , R n ) corresponds to circulant matrices A(x), each determined by a generating vector x = (x 1 , . . . , x n ) ∈ R n , as shown below.
Aff Sn (R X ⊗ R fi-1 , R X ⊗ R fi ) reduces to understanding Aff Sn (R X , R X ). Identifying R X with R n , they established that Aff Sn (R n , R n ) = v → (x 1 id + x 2 11 ⊤ )v + y1 x 1 , x 2 , y ∈ R , where 1 = 1 [n] = [1, . . . , 1] ⊤ . Figure
Each map in Aff Zn (R n , R n ) consists of a linear part defined by a circulant matrix and a bias term in R n :
A(x) :=       x 1 x n x n-1 • • • x 2 x 2 x 1 x n • • • x 3 x 3 x 2 x 1 • • • x 4 . . . . . . . . . . . . . . . x n x n-1 x n-2 • • • x 1       and y1 X = y1 [n] = y    1 . . . 1    .
Observe that any circulant matrix A(x) can be written as a linear combination A(e 1 ), . . . , A(e n ), that is, A(x) = n i=1 x i A(e i ) where {e 1 , . . . , e n } denotes the standard basis of R n . Since limitedwidth convolutional filters are standard in practice, we restrict attention to the following maps:
C k = v → k i=1 x i A(e i )v + y1 [n] x 1 , . . . , x k , y ∈ R .(2)
This class can be seen as the one-dimensional analogue of the k × k convolutional kernels widely used in 2-D computer vision applications. The corresponding neural space is given by N σ (C k1 , . . . , C k d ), for a choice of filter sizes 1 ≤ k 1 , . . . , k d ≤ n. Circular invariant layers can be characterized as
I := Aff Zn (R n , R) = v → (x 1 ⊤ ) • v + y x, y ∈ R .(3)
In particular, we focus on the spaces C 1 , which correspond to convolutional filters of width one; see Figure 1c for the colored bipartite graph representing the space Aff Sn (R n , R n ).
Having detailed the structure of the layer spaces and their correspondence to practical architectures, we now turn to the study of universality in families of shallow neural spaces.
this section cite: ['b17', 'b17', 'b53', 'b42', 'b41']

Section: Universality in Shallow Neural Spaces
Universality Classes. To establish notation and introduce the notion of universality classes, we begin by reformulating the classical universality result for shallow neural networks [23] in terms of our framework. Observe that the full class of shallow neural networks with variable width can be written as h∈N N σ (R m , R h , R). We denote by U σ (R m , R, R) the associated universality class-namely, the set of continuous functions on R m approximable by such networks. Formally, U σ (R m , R, R) is defined as the closure of this union in C(R m ), equipped with the topology of uniform convergence on compact sets. Theorem 5. The universality class for shallow neural networks, U σ (R m , R, R), coincides with C(R m ) if and only if the activation function σ is not a polynomial.
An analogous result in the equivariant setting was established by Ravanbakhsh [5] for neural networks defined on representations V and W as input and output spaces, respectively, and with regular hidden representations of the form R G . We define the universality class
U σ (V, R G , W ) as the set of functions in C(V, W ) that can be approximated by elements of h∈N N σ (V, R G ⊗ R h , W ).
Note that, in analogy with classical networks, the role of width is played by the hyperparameter h, which determines the dimension of the invariant hidden representation. The results of Ravanbakhsh [5] can then be stated as follows.
Theorem 6. The universality class U σ (V, R G , W ) coincides with C G (V, W ), the space of continuous G-equivariant functions from V to W , if and only if the activation function σ is not a polynomial.
We aim to provide a definition of universality classes that encompasses the notions introduced in Theorem 5 and Theorem 6, while being general enough to cover a wider range of architectures, such as PointNets and CNNs with variable filter size. To this end, we introduce the following auxiliary notation. Let G be a finite group and let V , W and Z be permutation representations of G. Let M be a subspace of Aff G (V, W ) and N a subspace of Aff G (W, Z), as defined in (1). Then, for each h ∈ N, we define M h as the subspace of Aff G (V, W ⊗ R h ) given by
M h := {x → (f 1 (x), . . . , f h (x)) | f 1 , . . . , f h ∈ M } ,(4)
and define h N as the subspace of
Aff G (W ⊗ R h , Z) given by h N := {(x 1 , . . . , x h ) → g 1 (x 1 ) + • • • + g h (x h ) | g 1 , . . . , g h ∈ N } .(5)
Recalling the isomorphism W ⊗R h ∼ = (W ) ⊕h , note that in the special cases where
M = Aff G (V, W ) and N = Aff G (W, Z), we have M h ∼ = Aff G (V, W ⊗ R h ) and h N ∼ = Aff G (W ⊗ R h , Z).
With this notation in place, we can now provide a general definition of universality classes. Definition 7 (Universality Classes). The universality class U σ (M, N ) associated with a family of neural spaces N σ (M h , h N ) for h ∈ N is the set of continuous functions approximated by these neural networks. More formally, U σ (M, N ) is defined as the closure of h∈N N σ (M h , h N ) in C(V, Z), equipped with the topology of uniform convergence on compact sets. As in the case of neural spaces, when M = Aff G (V, W ) and N = Aff G (W, Z), we will simply write U σ (V, W, Z).
However, comparing different universality classes is particularly challenging, and in the literature, separation power has often been used as a proxy for this purpose. The next section revisits this notion and critically examines its adequacy as a surrogate for universality.
On Separation-Constrained Universality. Theorem 6 establishes that equivariant neural networks cannot approximate all continuous functions. In particular, invariant networks are inherently unable to distinguish between symmetric inputs-a limitation that naturally constrains the class of functions they can represent. To make this precise, we formally define the notion of separation and the concept of separation-constrained universality.
Definition 8 (Separation-Constrained Universality). A family of functions N ⊆ {f : X → Y } is said to separate α and β if there exists f ∈ N such that f (α) ̸ = f (β). The set of point pairs not separated by N defines an equivalence relation:
ρ(N ) = {(α, β) ∈ X × X | f (α) = f(
β) for all f ∈ N }. A family N is said to be separation-constrained universal if its relative universality class coincides with the entire set of continuous functions that respect the equivalence relation ρ(N ), that is,
C ρ (X, Y ) = {f ∈ C(X, Y ) | f (x) = f (y) for all (x, y) ∈ ρ(N )}.
It is a standard fact in approximation theory [55] that if a family of functions N fails to separate two points, then it cannot approximate any function that does. As such, separation-constrained universality captures the maximal expressivity achievable by N . Here, we aim to investigate whether separation alone suffices to characterize expressivity i.e., whether universality classes with the same separation power must necessarily coincide. To this end, we now present three network families that share the same separation relation, despite differing in their internal representations. Throughout the remainder of the paper, we assume that all activation functions σ : R → R are non-polynomial. Proposition 9. 5 Let C 1 be defined as in (2), representing convolutional filters of width 1, and let I be as defined as in (3), representing invariant circular layers. Let S n act on R n ∼ = R [n] via the standard permutation action. Then, the following universality classes have the same separation power:
ρ U σ (C 1 , I) = ρ (U σ (R n , R n , R)) = ρ U σ (R n , R Sn , R) .
This naturally raises the following question. Question 10. Are these universality classes equal as well? More generally, is separation a complete proxy for comparing universality classes?
We answer Question 10 in the negative via Proposition 16, after developing the necessary theory.
this section cite: ['b22', 'b4', 'b4', 'b54']

Section: Main Results
In this section, we characterize the universality classes of invariant shallow neural networks (Section 5.1) and compare them (Section 5.2). Although the characterization is restricted to the invariant case, the following remark shows that it can be used to demonstrate that non-approximation in the invariant setting implies failure in the equivariant case as well. Remark 11. Let U 1 ⊆ U 2 ⊆ C G (V, W ) be two universality classes with input space V and output space W . Let π : W → W G denote the projection onto the trivial component of W . Define the pullback map
π * : C G (V, W ) -→ C G (V, W G ) f -→ π • f, where C G (V, W ) denotes the space of continuous equivariant functions from V to W . Then, π * (U 1 ) ⊊ π * (U 2 ) implies U 1 ⊊ U 2 , since π * is a continuous linear operator.
This shows that a sufficient condition for strict inclusion between spaces of invariant networks also yields a sufficient condition for strict inclusion between the corresponding spaces of equivariant networks.
With this observation, we now restrict our attention to invariant networks without loss of generality.
this section cite: []

Section: Characterization of Universality Classes
To characterize the universality classes of invariant shallow networks, we begin by introducing the notion of a basis map. Definition 12 (Basis maps). As defined in (1), let M be a subspace of Aff G (V, R Y ), where V is a permutation representation and Y is a finite G-set of cardinality ℓ, which we identify with [ℓ]. Let ϕ 1 , . . . , ϕ m be a basis for the linear part of M , and for each i ∈ Y , define the linear maps
ϕ i : R X → R m x → (ϕ 1 i (x), . . . , ϕ m i (x)).(6)
We refer to the maps ϕ 1 , . . . , ϕ ℓ as the basis maps associated with M or its basis ϕ 1 , . . . , ϕ m .
We now state the central characterization theorem for universality classes in terms of differential constraints on invariant functions.
Theorem 13. Let M and N be, respectively, subspaces of Aff G (V, W ) and Aff G (W, R). Let f be an invariant function, then f ∈ U σ (M, N ) if and only if P (∂ 1 , . . . , ∂ d )f = 0 for every polynomial P that vanishes on the spaces spanned by the rows ϕ 1 i , . . . , ϕ m i of each basis map ϕ 1 , . . . , ϕ ℓ , see (6).
Here, we assume d = dim V , and let P (∂ 1 , . . . , ∂ d ) denote the constant-coefficient linear differential operator associated with the polynomial P . The derivatives ∂ i on V are interpreted in the distributional sense; see [56] for details.
Although Theorem 13 provides a complete characterization of the universality classes for arbitrary families of neural spaces, this generality may come at the cost of practicality. Indeed, computing the exact set of polynomials P can be particularly challenging, due to the combinatorial complexity arising from the intersections of the subspaces spanned by ϕ 1 i , . . . , ϕ m i . Nonetheless, the theorem is not merely of theoretical interest-it plays a central role in deriving sufficient conditions for universality failure. These conditions enable a principled comparison of the approximation power of distinct model families, as we explore in the following sections.
this section cite: ['b5', 'b55']

Section: Sufficient Conditions for Universality Failure
In this section, we present two sufficient conditions for the failure of separation-constrained universality. These results will be used to resolve Question 10 and to prove Proposition 16. We begin with Theorem 14, which provides a general-but more difficult to verify-criterion, followed by Theorem 15, a less general version that is simpler to apply, despite its more convoluted appearance.
First, we introduce the notion of a directional derivative. For each vector c = (c 1 , . . . , c n ) ∈ R n , the directional derivative is defined as the differential operator
D c = c 1 • ∂ 1 + • • • + c n • ∂ n . Theorem 14. A continuous function f does not belong to the class U σ (M, N ) if D c1 • • • D c ℓ f ̸ = 0 (7
)
for some choice of c α in ker(ϕ ⊤ α ) for each basis map ϕ 1 , . . . , ϕ ℓ .
In the case of equivariant networks where each affine layer is allowed to be an arbitrary equivariant affine map, Theorem 14 can be strengthened as follows.
Theorem 15. Let M = Aff G (V, W ) and N = Aff G (W, R), where V and W are permutation representations. Let ϕ 1 , . . . , ϕ ℓ denote the basis maps associated with M , see (6). Then, the universal class U σ (M, N ) fails to be separation-constrained universal if, for some choice of:
• integers s 1 , . . . , s ℓ ∈ {0, . . . , ℓ} satisfying s 1 + • • • + s ℓ = ℓ,
• integers a 1 > ℓ and a i + ℓ < a i+1 for each i = 1, . . . , ℓ,
• vectors c i ∈ ker(ϕ ⊤ i ) for each i = 1, . . . , ℓ, Let i 1 , . . . , i r be the indices such that s ij ̸ = 0. The following expression is nonzero:
σ∈S ℓ a i1 ! s i1 ! • • • a ir ! s ir ! (c σ(1),1 • • • c σ(s1),1 ) • (c σ(s1+1),2 • • • c σ(s1+s2),2 ) • • • (c σ(ℓ-s ℓ ),ℓ • • • c ℓ,ℓ ).
this section cite: ['b5']

Section: The Heterogeneous Landscape of Universality Classes
We now apply the tools developed in Section 5 to investigate the structure of universality classes and illustrate their heterogeneity. In Section 6.1, we address Question 10 by applying Theorems 14 and 15 to exhibit concrete examples of failure. In contrast, Section 6.2 presents Theorem 18, a generalization of Theorem 6, which provides sufficient conditions for achieving separation-constrained universalityhighlighting the diversity of behaviors even within fixed symmetry classes.
this section cite: []

Section: Examples of Failure
Proposition 16. As established in Proposition 9, the following spaces achieve the same separation power, yet differ in their approximation capabilities when n > 2:
U σ (C 1 , I) ⊊ U σ (R n , R n , R) ⊊ U σ (R n , R Sn , R).
By Remark 11, the corresponding equivariant models also have distinct approximation power.
We will prove the two strict inclusions of Proposition 16 in the following three paragraphs.
Failure for CNN with filter width 1: We now apply Theorem 14 to show that CNNs with filter width 1 cannot approximate the function (
x 1 + • • • + x n ) n for n > 1, namely (x 1 + • • • + x n ) n
/ ∈ U σ (C 1 , I). Indeed, for any α = 1, . . . , n, we have e α+1 ∈ ker(π ⊤ α ) = Span{e 1 , . . . , êα , . . . , e n }, where
α + 1 is modulo n. Moreover, note that D eα = ∂ α , thus ∂ n • • • ∂ 1 (x 1 + • • • + x n ) n = n! ̸ = 0, which violates (7) in Theorem 14.
Success for PointNet: We now show that shallow PointNets approximate the polynomial function
(x 1 +• • •+x n ) n . By Proposition 41 in Appendix D, f (x 1 , x 1 +• • •+x n )+• • •+f (x n , x 1 +• • •+x n ) belongs to U σ (R n , R n ,
R) for any f ∈ C(R 2 ). In particular, for f (x, y) := y n ∈ C(R 2 ), we see that
(x 1 + • • • + x n ) n ∈ U σ (R n , R n , R).
Together with the previous observation, this establishes the first strict inclusion in Proposition 16, namely
U σ (C 1 , I) ⊊ U σ (R n , R n , R).
Failure for PointNet: We now aim to show that shallow PointNets cannot approximate the polynomial function x 1 • • • x n , which is S n -invariant and therefore should, in principle, be approximable in a separation-constrained setting. We distinguish two cases: n > 3 and n = 3. Note that for n = 2, the symmetric group S 2 is abelian, and universality follows directly from Theorem 6. We start considering (n > 3). We again employ Theorem 14 to show that shallow invariant PointNets cannot approximate x 1 • • • x n , and hence neither CNNs with filter width 1. Indeed, note that the basis maps for Aff Sn (R n , R n ) in this case are given by ϕ
α (x 1 , . . . , x n ) = (x α , x 1 + • • • + x n ).
In matrix form, we write ϕ α = [e α , 1] ⊤ . We define K α := ker ϕ ⊤ α = Span(e i -e j ) i,j=1,..., α,...,n . Then, define the following direction vectors: In view of the universality results for PointNet with depth 3 and arbitrary widths in both hidden layers by Segol and Lipman [44], this example highlights how, in the case of permutation equivariance, depth is crucial for achieving separation-constrained universality. This contrasts with other settings where universality can be achieved without relying on depth, as we will describe in the next section.
this section cite: ['b43']

Section: Examples of Separation-Constrained Universality
We now present Theorem 18, a generalization of Theorem 6, which shows that a specific class of hidden representations can achieve separation-constrained universality. These representations arise from cosets of particular subgroups H of G, defined as follows:
Definition 17 (Normal subgroup). A subgroup H is normal if ghg -1 ∈ H for each h ∈ H, g ∈ G.
Theorem 18. Let V and Z be permutation representations of a finite group G, and let H be a normal subgroup of G. Therefore, U σ (V, R G/H , Z) is separation-constrained universal.
The converse does not always hold: representations arising from non-normal subgroups may nevertheless achieve separation-constrained universality, as illustrated in the following remark.
Remark 19. Let H be a non-normal subgroup of S n contained in A n . Then
U σ (R Sn/An , R Sn/An , R) = U σ (R Sn/An , R Sn/H , R) = C Sn (R Sn/An ).
All subgroups of an abelian group are normal, whereas S n has only one non-trivial normal subgroup, A n , with |S n /A n | = 2, yielding hidden representations that are too small to be effective. We summarize by noting that intermediate representations built from abelian groups, such as those in standard circular CNNs, achieve separation-constrained universality. In contrast, architectures based on permutation representations lack this guarantee, as shown in Proposition 16.
this section cite: []

Section: Limitations
This work represents a first step toward understanding the approximation capabilities of equivariant networks beyond separation. Several limitations, however, remain. In particular, our analysis is limited to shallow networks. While these serve as minimal and analytically tractable examples, they may not fully capture the behavior of deeper architectures. Extending this framework to deeper networks-particularly in settings where depth interacts nontrivially with separation, as in IGNs-poses a significant challenge.
this section cite: []

Section: Conclusions
We investigated the approximation capabilities of equivariant neural networks, moving beyond their well-studied separation properties. By formulating shallow invariant networks as generalized superpositions of ridge functions (see Proposition 41), we developed a novel characterization of their universality classes and examined how architectural choices influence approximation behavior. Our analysis reveals that even networks with maximal separation power may fail to approximate all functions within the corresponding symmetry-respecting class, a phenomenon we attribute to the structure of their hidden representations. These findings suggest that approximation power cannot be deduced from separation alone and should be treated as a distinct axis of expressivity. Our results thus call for a more nuanced understanding of equivariant architectures-one that takes both axes into account in theoretical analysis and model design.
As future directions, we aim to extend our framework to determine whether failures of separationconstrained universality, such as those established in Proposition 16, persist in deeper architectures. Another important avenue for investigation is how differences in expressivity affect generalization, particularly among models that share the same separation power.
this section cite: []

Section: References
Ref_id:b0 Title: Group Equivariant Convolutional Networks Year: (2016-06)
Ref_id:b1 Title: On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups Year: (2018-07)
Ref_id:b2 Title: Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges Year: (2021-05)
Ref_id:b3 Title: Invariant and Equivariant Graph Networks Year: (2018-09)
Ref_id:b4 Title: Universal Equivariant Multilayer Perceptrons Year: ()
Ref_id:b5 Title: Approximation by superpositions of a sigmoidal function Year: (1989-12)
Ref_id:b6 Title: Approximation capabilities of multilayer feedforward networks Year: (1991-01)
Ref_id:b7 Title: Equivariant Polynomials for Graph Neural Networks Year: (2023-06)
Ref_id:b8 Title: Equivariant Subgraph Aggregation Networks Year: (2022-03)
Ref_id:b9 Title: THE REDUCTION OF A GRAPH TO CANONICAL FORM AND THE ALGEBRA WHICH APPEARS THEREIN Year: (1968)
Ref_id:b10 Title: Large Networks and Graph Limits Year: (2012-12)
Ref_id:b11 Title: The graph neural network model. Faculty of Informatics -Papers (Archive) Year: (2009-01)
Ref_id:b12 Title: A new model for learning in graph domains Year: (2005-07)
Ref_id:b13 Title: Semi-Supervised Classification with Graph Convolutional Networks Year: (2017-02)
Ref_id:b14 Title: On Learning Sets of Symmetric Elements Year: (2020-11)
Ref_id:b15 Title: Subgraph Neural Networks Year: (2020-11)
Ref_id:b16 Title: On the Expressive Power of Geometric Graph Neural Networks Year: (2023)
Ref_id:b17 Title: Separation Power of Equivariant Neural Networks Year: (2024-12)
Ref_id:b18 Title: Future Directions in the Theory of Graph Machine Learning Year: (2024-06)
Ref_id:b19 Title: On the Hölder Stability of Multiset and Graph Neural Networks Year: (2025-04)
Ref_id:b20 Title: Universal approximation bounds for superpositions of a sigmoidal function Year: (1993-05)
Ref_id:b21 Title: Multilayer feedforward networks with a nonpolynomial activation function can approximate any function Year: (1993-01)
Ref_id:b22 Title: Approximation theory of the MLP model in neural networks Year: (1999-01)
Ref_id:b23 Title: Neural network approximation Year: (2021-05)
Ref_id:b24 Title: Mathematical theory of deep learning Year: (2025-04)
Ref_id:b25 Title: Error bounds for approximations with deep ReLU networks Year: (2017-10)
Ref_id:b26 Title: Optimal approximation of continuous functions by very deep ReLU networks Year: (2018-06)
Ref_id:b27 Title: Optimal Approximation Rates for Deep ReLU Neural Networks on Sobolev and Besov Spaces Year: (2024-04)
Ref_id:b28 Title: Tensor field networks: Rotation-and translation-equivariant neural networks for 3D point clouds Year: (2018-05)
Ref_id:b29 Title: SchNet -a deep learning architecture for molecules and materials Year: (2018-06)
Ref_id:b30 Title: Geometric Deep Learning for 3D RNA inverse design Year: (2024-04)
Ref_id:b31 Title: 3D steerable CNNs: learning rotationally equivariant features in volumetric data Year: (2018)
Ref_id:b32 Title: Universal approximations of invariant maps by neural networks Year: (2018-04)
Ref_id:b33 Title: Equivalence of approximation by convolutional neural networks and fully-connected networks Year: (2020-04)
Ref_id:b34 Title: On the Universality of Invariant Networks Year: (2019-05)
Ref_id:b35 Title: Universality of group convolutional neural networks based on ridgelet analysis on groups Year: (2022-11)
Ref_id:b36 Title: Universal Invariant and Equivariant Graph Neural Networks Year: (2019)
Ref_id:b37 Title: Machine learning and invariant theory. Notices of the Year: (2023-09)
Ref_id:b38 Title: Frame Averaging for Invariant and Equivariant Network Design Year: (2021-10)
Ref_id:b39 Title: Yoshua Bengio, and Siamak Ravanbakhsh. Equivariance with Learned Canonicalization Functions Year: (2023-07)
Ref_id:b40 Title: Equivariant Frames and the Impossibility of Continuous Canonicalization Year: (2024-06)
Ref_id:b41 Title: Deep Sets Year: (2017)
Ref_id:b42 Title: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation Year: (2017-07)
Ref_id:b43 Title: On Universal Equivariant Set Networks Year: (2020-01)
Ref_id:b44 Title: How Powerful are Graph Neural Networks? Year: (2019-02)
Ref_id:b45 Title: Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks Year: (2019-07)
Ref_id:b46 Title: Provably Powerful Graph Networks Year: (2019)
Ref_id:b47 Title: On the equivalence between graph isomorphism testing and function approximation with GNNs Year: (2019-05)
Ref_id:b48 Title: Expressive Power of Invariant and Equivariant Graph Neural Networks Year: (2021-06)
Ref_id:b49 Title: Beyond Weisfeiler-Lehman: A Quantitative Framework for GNN Expressiveness Year: (2024-01)
Ref_id:b50 Title: Understanding and Extending Subgraph GNNs by Rethinking Their Symmetries Year: (2022-10)
Ref_id:b51 Title: Representation Theory Year: (2004)
Ref_id:b52 Title: A Characterization Theorem for Equivariant Networks with Point-wise Activations Year: (2024-01)
Ref_id:b53 Title:  Year: ()
Ref_id:b54 Title: Constructive Approximation Year: (1993)
Ref_id:b55 Title:  Year: (2009)
Ref_id:b56 Title: Introduction To Commutative Algebra Year: (1994-02)
Ref_id:b57 Title: Ridge Functions. Cambridge Tracts in Mathematics Year: (2015)
