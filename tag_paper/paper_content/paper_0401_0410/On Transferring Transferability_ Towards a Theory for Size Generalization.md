Title: On Transferring Transferability: Towards a Theory for Size Generalization
Abstract: Many modern learning tasks require models that can take inputs of varying sizes. Consequently, dimension-independent architectures have been proposed for domains where the inputs are graphs, sets, and point clouds. Recent work on graph neural networks has explored whether a model trained on low-dimensional data can transfer its performance to higher-dimensional inputs. We extend this body of work by introducing a general framework for transferability across dimensions. We show that transferability corresponds precisely to continuity in a limit space formed by identifying small problem instances with equivalent large ones. This identification is driven by the data and the learning task. We instantiate our framework on existing architectures, and implement the necessary changes to ensure their transferability. Finally, we provide design principles for designing new transferable models. Numerical experiments support our findings.

Section: Introduction
Modern learning problems often involve inputs of arbitrary size: language has text sequences of arbitrary length, graphs encode networks with an arbitrary number of nodes, and particle systems might have an arbitrary number of points. Although traditional models are typically limited to the specific input dimensions they were trained on, dimension-independent architectures have been proposed for several domains. For instance, graph neural networks (GNN) for graphs [70], DeepSet for sets [91], and PointNet for point clouds [66]. These models have a desirable feature: they can be trained with low-dimensional samples and, then, deployed on higher-dimensional data.
A key question in this context is whether the performance of a model trained with low-dimensional samples transfers across dimensions. This has been thoroughly studied in the context of GNNs [70,58,59], where the term transferability was first coined. 1 In turn, the tools used to establish transferability of GNNs are, at first sight, problem-specific, including the concept of a graphon, and certain convergence of homomorphism densities. Thus, the analysis does not readily extend beyond GNNs, and so we ask What properties of the model, data, and learning task ensure that the learning performance transfers well across dimensions?
This question is relevant for the current pursuit of graph foundation models [53], GNN for combinatorial optimization [12], neural PDE solvers [67,51], and so on, where pre-trained models aim to be adapted to a wide range of tasks involving different-sized objects. In this paper, we answer this question by identifying assumptions on the learning problem that make a model generalize across dimensions, and give a theoretical framework to understand models with such properties. Our framework requires technical notions that we develop throughout the paper, so before diving into the details, we proceed to give the high-level intuition.
Our framework considers a machine learning model parametrized by θ ∈ R k , where the number of parameters k is fixed. For any fixed θ, the model can be evaluated on inputs of any size n. That is, the learned θ defines a sequence of functions, for example, (f n : V n → R) n with increasingly larger input sizes. 2 We think of V n ⊆ V ∞ as a subspace of an infinite-dimensional limit space. Transferability then amounts to asking whether for large enough n and m we have f n (x n ) ≈ f m (x m ) when the inputs x n and x m are close to each other in the limiting space V ∞ . For example, we consider graphs of different sizes to be close if their associated step graphons are close in the cut metric, following standard practice in the GNN literature. Similarly, we view point clouds of varying sizes as close if their corresponding empirical measures are close in the Wasserstein metric.
In a nutshell, we identify two properties that guarantee transferability: (i) the functions in the sequence are compatible with one another in a precise sense, and (ii) all of them are continuous with respect to a prescribed norm on V ∞ . Compatibility ensures that the sequence extends to a function taking infinite-dimensional inputs f ∞ : V ∞ → R, while continuity ensures that this extension is continuous. This allows us to relate model outputs on two differently-sized inputs via their extension
f n (x n ) = f ∞ (x n ) ≈ f ∞ (x m ) = f m (x m ).
Moreover, if we evaluate our models on a sequence of inputs x n converging in a suitable sense to an infinite-dimensional input x, then the evaluations f n (x n ) converge to the value of the limiting model f ∞ (x). Our framework parallels the ideas from the GNN literature, where graph convergence to a limiting graphon is measured in cut distancewhich is equivalent to homomorphism density convergence. However, this framework provides a transparent extension that allows us to analyze other models, e.g., DeepSets and PointNet.
Our contributions. Next, we summarize our three core contributions.
(A framework for transferability) We introduce a class of any-dimensional models and a formal definition of transferability. We show that transferability holds whenever certain notions of compatibility and continuity hold. We further leverage transferability to establish a generalization bound across dimensions. In our theory, the notion of transferability relies on identifying lowdimensional objects with higher-dimensional ones, and measuring similarity of such objects in an appropriate metric. We illustrate how these choices have to align with the data and learning tasks.
(Transferability of existing models) We instantiate our framework on several models, including DeepSet [91], PointNet [66], standard GNNs [70], Invariant Graph Networks (IGN) [56], and DS-CI [5]. Not all of these models satisfy our assumptions. For those that do, our framework yields new and existing transferability results in a unified fashion. For those that do not, we either modify them to ensure transferability or demonstrate numerically that transferability fails and impedes the performance of these models.
(New transferable models) We provide design principles for constructing transferable models. Leveraging these ideas, we develop a variant of IGN that is provably transferable, addressing challenges highlighted in previous work [56,35]. We also design a new transferable model for point cloud tasks that is more computationally efficient than existing methods [5].
Notation. The Kronecker product on matrices is denoted by ⊗. We write 1 n and I n for the all-ones vector and the identity matrix of size n respectively. Given two sequences (a n ), (b n ) ⊆ R, we say that a n ≲ b n if there is a constant C > 0 so that a n ≤ Cb n for all n. See also Appendix A.
Outline. Section 2 introduces the theoretical framework. Section 3 defines transferability, which is used in Section 4 to derive a generalization guarantee. Section 5 instantiates our framework on concrete neural network examples, and Section 6 provides supporting experiments. 3 Related work and limitations are discussed in the Appendix.
2 How to consistently grow: equivalence of differently sized objects
In this section, we formally introduce the setting for studying transferability. The definitions here build upon the framework introduced in [49,50] for defining any-dimensional neural networks and optimization problems. Central to our developments is the notion of compatibility between a sequence of functions. Intuitively, a sequence of functions is compatible when they respect the equivalence between low-dimensional objects and high-dimensional version of them. For example, sample statistics like the mean and standard deviation remain invariant when each element is repeated m times. Similarly, various graph parameters such as triangle densities and normalized max-cut values are preserved when each vertex is replaced by m identical copies of itself.
To formalize these notions of equivalence of objects and compatibility between functions, we will leverage the concept of a consistent sequence. Intuitively, they are nested sequences of vector spaces with growing groups acting on them, representing problem instances of different sizes. They are interconnected by embeddings that establish equivalence between smaller and larger instances.
Definition 2.1.
A consistent sequence V = {(V n ) n∈N , (φ N,n ) n⪯N , (G n ) n∈N }
is a sequence of finite-dimensional vectors spaces V n , maps φ N,n and groups G n acting linearly on V n , indexed by a directed poset 4 (N, ⪯), such that for all n ⪯ N , the group G n is embedded into G N , and φ N,n : V n ↩→ V N is a linear, G n -equivariant embedding.
Although the definition includes a sequence of groups, it applies to situations without symmetry by setting G n = {id} for all n. While all the examples in this work involve symmetries, we believe our framework might be relevant in symmetry-free settings. 5Example 2.2. Here are two prototypical examples of consistent sequences. In both cases, V n = R n , with the symmetric group S n acting via coordinate permutation. (i) the sequence indexed by N with the standard ordering ≤, paired with the zero-padding embedding R n ↩→ R N , which maps (x 1 , . . . , x n ) to (x 1 , . . . , x n , 0, . . . , 0); and (ii) the sequence indexed by N with the divisibility ordering (n ⪯ N whenever N is divisible by n) paired with the duplication embedding R n ↩→ R N , sending a vector x = (x 1 , . . . , x n ) to x ⊗ 1 N/n = (x 1 , . . . , x 1 N/n copies , . . . , x n , . . . , x n N/n copies ).
Next, we define a common space for objects of all sizes in a consistent sequence, in which we can compare objects of different sizes and take their limits. For example, if we identify vectors x with their duplication embeddings x ⊗ 1 N/n , we can view the resulting equivalence classes as step functions on [0, 1] taking value x i on consecutive intervals of length 1/n.
Definition 2.3. Let V ∞ = V n where we identify v ∈ V n with its image φ N,n (v) for all n ⪯ N to be equivalent. Analogously, we let G ∞ be the union of groups with equivalent identifications.
For simplicity, we will often write v ∈ V ∞ for any finite-dimensional object v ∈ V n to denote the equivalence class [v].
It is straightforward to check that G ∞ acts on V ∞ via g • [v] = [g • v].
Having established how consistent sequences define a desirable equivalence relation in the space of problem instances of varying sizes, we now introduce compatible functions that respect such equivalence.
Definition 2.4. Let V = {(V n ), (φ N,n ), (G n )} and U = {(U n ), (ψ N,n ), (G n )} be two consistent sequences indexed by (N, ⪯). A sequence of maps (f n : V n → U n ) is compatible with respect to V, U if f N • φ N,n = ψ N,n • f n for all n ⪯ N , and each f n is G n -equivariant.
It is instructive to visualize compatible maps using the following commutative diagram, which represents a mapping between sequences while ensuring that the diagram commutes:
. . .
(V n1 , G n1 ) (V n2 , G n2 ) (V n3 , G n3 ) . . . . . . (U n1 , G n1 ) (U n2 , G n2 ) (U n3 , G n3 ) . . . ... φn 2 ,n 1 fn 1 φn 3 ,n 2 fn 2 ... fn 3 ... ψn 2 ,n 1 ψn 3 ,n 2 ...
for n 1 ⪯ n 2 ⪯ . . .
Equivalently, compatible maps are precisely those sequences (f n ) that extend to equivalence classes:
A sequence of maps (f n ) is compatible if, and only if, there exists a G ∞ -equivariant map f ∞ : V ∞ → U ∞ such that f n = f ∞ | Vn for all n.
See Appendix C.2 for a proof of this equivalence.
this section cite: ['b69', 'b65', 'b69', 'b57', 'b58', 'b52', 'b11', 'b66', 'b50', 'b65', 'b69', 'b55', 'b4', 'b55', 'b34', 'b4', 'b48', 'b49']

Section: How to define distances across dimensions
Compatible maps are functions that respect the equivalence defined by consistent sequences. We now consider whether these functions further preserve proximity, mapping "nearby" objects to "nearby" outputs. This requires a well-defined notion of distance between objects of different sizes that is consistent with the earlier equivalence.
Definition 2.5. For a consistent sequence V, a sequence of norms (∥ • ∥ Vn ) on V n are compatible if all the embeddings φ N,n and the G n -actions are isometries. i.e., for all n ⪯ N, x ∈ V n , g ∈ G n , ∥φ N,n x∥ V N = ∥x∥ Vn and ∥g • x∥ Vn = ∥x∥ Vn .
Equivalently, compatible norms are those that extend to the limit; that is, there exists a norm ∥ • ∥ V∞ on V ∞ such that for any n and x ∈ V n , ∥x∥ Vn = ∥x∥ V∞ , and the G ∞ -action on V ∞ is an isometry with respect to ∥ • ∥ V∞ ; see Appendix C.3 for a proof of this statement.
this section cite: []

Section: Example 2.6 (Example 2.2 continued).
For our prototypical consistent sequences, the ℓ p norms ∥x∥ p := ( i |x i | p ) 1/p are compatible with the zero-padding embeddings, while the normalized ℓ p norms ∥x∥ p := 1 n i |x i | p 1/p are compatible with the duplication embeddings. With norms in place, we can define a limit space that includes not only equivalence classes of finite-dimensional objects, but also their limits. As we will see, this recovers meaningful spaces such as L p ([0, 1]) and the space of graphons with the cut norm. Moreover, the orbit spaces of these limits recover probability measures (with the Wasserstein distance) and equivalence classes of graphons (with the cut distance), respectively.
Definition 2.7. The limit space is the pair (V ∞ , G ∞ ) where V ∞ denotes the completion of V ∞ with respect to ∥ • ∥ V∞ , endowed with the symmetrized metric 6d(x, y) := inf g∈G∞ ∥g • x -y∥ V∞ for x, y ∈ V ∞ .
this section cite: []

Section: Transferability is just continuity
The notion of GNN transferability was established in [70,48]. It states that the output discrepancy ∥f n (A n , X n )f m (A m , X m )∥ between graph signals of sizes n, m sampled from the same graphon signal (W, f ) vanishes as n, m grow. Prior studies explore this property under various sampling schemes, norms, and GNN architectures [41,70,47,11,35].
We formalize the idea of functions "mapping close objects to close outputs" as (Lipschitz) continuity of compatible maps.
Definition 3.1. Let V, U be consistent sequences endowed with norms. A sequence of equivariant maps (f n : V n → U n ) is continuously (respectively, L-Lipschitz, L(r)-locally Lipschitz) transferable if there exists f ∞ : V ∞ → U ∞ that is continuous (respectively, L-Lipschitz, L(r)-Lipschitz on the ball B(0, r) = {v ∈ V ∞ : ∥v∥ V∞ < r} for all r > 0) with respect to ∥ • ∥ V∞ , ∥ • ∥ U∞ , such that f n = f ∞ | Vn for all n. Notice that if (f n ) is transferable, then it must be compatible.
In Appendix D.1 we show that Lipschitz transferability is equivalent to having a compatible sequence (f n ), each Lipschitz with the same constant. For linear maps, it suffices to verify that the operator norms are uniformly bounded. This simplifies the verification of transferability. Importantly, we also show that transferability with respect to the norm ∥ • ∥ V∞ implies transferability with respect to the symmetrized metric d.
The following proposition shows that the output discrepancy is controlled whenever a dimensionindependent model is continuously or Lipschitz transferable, justifying our terminology. Thus, our framework extends transferability beyond GNNs, highlighting that transferability is equivalent to continuity in the limit space. For a given function R : N → R + , we say that (
x n ) converges to x in V ∞ at a rate R(n) in d if d(x n , x) ≲ R(n) and R(n) → 0.
Proposition 3.2. Let V, U be consistent sequences and let (f n : V n → U n ) be maps between them.
this section cite: ['b69', 'b47', 'b40', 'b69', 'b46', 'b10', 'b34']

Section: (Transferability) For any sequence (x
n ∈ V n ) converging to a limiting object x ∈ V ∞ in d, if (f n ) is continuously transferable, then d(f n (x n ), f m (x m )) → 0 as n, m → ∞. Furthermore, if (x n )
converges to x at rate R(n) and (f n ) is locally Lipschitz-transferable, then
d(f n (x n ), f m (x m )) ≲ R(n) + R(m). (Stability) If (f n ) is L(r)
-locally Lipschitz transferable, then for any two inputs x n ∈ V n and x m ∈ V m of any two sizes n, m with
∥x n ∥ Vn , ∥x m ∥ Vm ≤ r, we have d(f n (x n ), f m (x m )) ≤ L(r) d(x n , x m ).
Several remarks are in order. First, the same results trivially hold when the symmetrized metric is replaced by the norms on V ∞ , U ∞ . However, the finite inputs (x n ) are often obtained from the limiting object via random sampling, and converge only in the symmetrized metric. This is the case for all the common sampling strategies used for sets, graphs, and point clouds, which we review in Appendix D.3. In the case of graphs, our framework recovers results from prior work, achieving the same or stronger rate of convergence. Second, some models are only locally Lipschitz outside of a measure-zero set. Transferability still holds for such models for almost all limit objects x ∈ V ∞ . This and other extensions are deferred to Appendix D.2. Third, Lipschitz continuity of neural networks has been extensively studied in the context of model stability and robustness to small input perturbations [25,85,43,27]. Our results generalize the connection between stability and transferability, first established in [72], by showing that Lipschitz continuity of an any-dimensional model implies not only stability, but also transferability across input sizes.
this section cite: ['b24', 'b84', 'b42', 'b26', 'b71']

Section: Transferability implies size generalization
We use transferability to derive a generalization bound for models trained on inputs of fixed-size n, and evaluate their performance as n → ∞. It can therefore be interpreted as a size generalization bound, accounting for distributional shifts induced by size variation. We study a supervised learning task with input and output spaces modeled by consistent sequences V and U satisfying the following.
Assumption 4.1. Let µ be a probability distribution supported on a product of subsets X × Y ⊆ V ∞ ×U ∞ whose orbit closures are compact in the symmetrized metrics. Let S n : V ∞ ×U ∞ → V n ×U n be a random sampling procedure such that S n (x, y) ∈ X × Y almost surely for (x, y) ∈ supp(µ). This sampling induces a distribution µ n on V n × U n by drawing (x, y) ∼ µ and then sampling (x n , y n ) ∼ S n (x, y). A dataset s = {(x i , y i )} N i=1 is drawn i.i.d. from µ n . Suppose the loss function ℓ : U ∞ × U ∞ → R is bounded by M and c ℓ -Lipschitz. Assume training is performed using a locally Lipschitz transferable neural network model, and
A s : V ∞ → U ∞ is the hypothesis learned (given dataset s), which is c s -Lipschitz on X × Y in d.
In Appendix E, we further break down these technical assumptions and provide detailed motivation to make them more accessible. We also discuss two concrete examples of sets and graphs to justify the practicality of the key assumptions. The following generalization bound follows by applying the results from [88] to our setting. Proposition 4.2. Consider a learning task with input and output spaces modeled by consistent sequences V and U satisfying Assumption 4.1. For any δ > 0, with probability at least 1δ,
1 N N i=1 ℓ(A s (x i ), y i ) -E (x,y)∼µ ℓ(A s (x), y) ≤ c l (c s ∨ 1) ξ -1 (N ) + W 1 (µ, µ n ) + M (2 log 2)ξ -1 (N ) 2 + 2 log(1/δ) N(1)
where ξ(r) := C X (r/4)C Y (r/4) r 2
, C X (ε) and C Y (ε) are the ε-covering numbers in d of X and Y , respectively, and W 1 denotes the Wasserstein-1 distance. Moreover, if the sampling S n has convergence rate R(n) in expectation, then W 1 (µ, µ n ) ≲ R(n) as n → ∞ and the bound (1) converges to 0 as n, N → ∞.
We defer the proof to Appendix E. This bound shows that generalization improves with greater model transferability (i.e., smaller Lipschitz constants) and higher-dimensional training data (i.e., large n), while it deteriorates with increasing complexity of the data space (i.e., larger covering numbers).
this section cite: ['b87']

Section: Transferable neural networks
We now apply our general framework to the settings of sets, graphs, and point clouds. In each case, we identify suitable consistent sequences, analyze the compatibility and transferability of existing neural network models with respect to these sequences, and propose a principled recipe for designing new transferable models. This analysis provides new insights into the tasks for which each model is best suited, while offering provable size-generalization guarantees.
A transferable neural network learns a fixed set of parameters θ that define transferable functions (f n ) for any n-dimensional normed space V n . The compatibility conditions on (f n ) and the sequence of norms ∥ • ∥ Vn describes the implicit inductive bias of the model.
Our analysis hinges on the following proposition, which observes that compatibility and transferability are preserved under composition and reduces the verification of Lipschitz continuity from the limit space to each finite-dimensional space. This significantly simplifies the task of establishing transferability in complex neural networks. The following result is proved in Appendix E.1.
Proposition 5.1. Let (V (i) n ) n , (U (i) n ) n be consistent sequences for i = 1, . . . , D. For each i, let (W (i) n : V (i) n → U (i) n ) be linear maps and (ρ (i) n : U (i) n → V (i+1) n
) be nonlinearities, and assume that (W
(i) n ), (ρ (i) n ) are compatible, that sup n,i ∥W (i) n ∥ op < ∞, and that ρ (i) n is L(r)-Lipschitz on balls B(0, r) in U (i)
n for all r > 0 and all n. Then the sequence of neural networks (
f n = W (D) n • ρ (D-1) n • . . . • ρ (1) n • W (1)
n ) is L(r)-locally Lipschitz transferable for explicit L(r).
this section cite: []

Section: Sets
Consistent sequence of sets. We have described in Examples 2.2 and 2.6 two consistent sequences that formalize equivalence across sets of varying sizes: the zero-padding consistent sequence V zero and the duplication consistent sequence V dup , along with norms on R n that are compatible with each sequence. The resulting limit spaces and symmetrized metrics recover interesting mathematical structures, as summarized in Table 1. Further details are provided in Appendix F.1. See Figure 1 for an illustration. Permutation-invariant neural networks on sets. Prominent invariant neural networks on multisets including DeepSet [91], normalized DeepSet [9], and PointNet [66] take the form f n (X) = σ (Agg n i=1 ρ(X i: )), where ρ : R d → R h , σ : R h → R are fully-connected neural networks, both independent of n, and Agg denotes a permutation-invariant aggregation. Sum (Agg n i=1 := n i=1 ), mean (Agg n i=1 := 1 n n i=1 ) and max (Agg n i=1 := max n i=1 ) aggregations yield DeepSet, normalized DeepSet and PointNet, respectively. Using Proposition 5.1, we examine the transferability of these models, demonstrating that they parameterize functions on different limit spaces, and hence are suitable for different tasks, see Corollary 5.2, Table 2, and Figure 2. This is proved in Appendix F.2. L p ∥f ∥ p = ( ∫ 1 0 f(x) p dx ) 1/p functions on with finite -norm V ∞ = [0,1] L p add limit points Empirical measures Probability measures with finite -th moment p Wasserstein -distance p Duplication Corollary 5.2. The transferability of the three models are summarized in Table 2. Particularly, they extend to Lipschitz functions
ℝ n ↪ ℝ kn (x 1 , …, x n ) ↦ x ⊗ 1 k ∥x∥ p = n -1/p ∥x∥ p Limit space V ∞
DeepSet ∞ : l 1 (R d ) → R, DeepSet ∞ : P 1 (R d ) → R, PointNet ∞ : P ∞ (R d ) → R given by DeepSet ∞ ((xi)) = σ ∞ i=1 ρ(xi) , DeepSet ∞ (µ) = σ ρdµ , PointNet∞(µ) = σ sup x∈supp(µ) ρ(x) . Model Vzero, ∥ • ∥1 V dup , ∥ • ∥p DeepSet Lipschitz transferable if ρ(0) = 0 Incompatible
this section cite: ['b8', 'b65']

Section: DeepSet

this section cite: []

Section: Incompatible Lipschitz transferable
PointNet Incompatible Compatible, Lipschitz transferable if p = ∞ Table 2: Transferability of invariant neural networks on sets, assuming the nonlinearities σ, ρ are Lipschitz.
this section cite: []

Section: Graphs Consistent sequences for graph signals.
To model graph signals, we consider the sequence of vector spaces V n = R n×n sym × R n×d , representing the adjacency matrix of a weighted graph with n nodes and node features of dimension d. The symmetric group S n acts on (A, X) ∈ V n by g • (A, X) = (gAg ⊤ , gX), capturing the invariance to node ordering. While other embeddings are possible, in this work we focus on the duplication-consistent sequence
V G
dup for graphs, illustrated in Figure 3. Specifically, for any n|N , the embedding is defined by
φ N,n (A, X) = A ⊗ 1 N/n 1 ⊤ N/n , X ⊗ 1 N/n
, which corresponds to replacing each node with N/n duplicated copies. These embeddings precisely identify graph signals that induce the same step graphon signal. We consider three compatible norms: the p-norm ∥(A, X)∥ p = ( 1
n 2 i,j |A ij | p ) 1/p + 1 n i ∥X i: ∥ p R d 1/p , the operator p-norm ∥(A, X)∥ op,p := 1 n ∥A∥ op,p + 1 n i ∥X i: ∥ p R d
1/p , and the cut norm ∥(A, X)∥ □ := ∥A∥ □ + ∥X∥ □ . In all cases, the limit space is the space of graphon signals and the symmetrized metrics recover extensively studied graphon distances in [54,47].
10 1 10 2 10 3 10 4 Set size n -60 -40 -20 0 fn(xn) (a) DeepSet [91] output (incompatible) 10 1 10 2 10 3 10 4 Set size n 0.0912 0.0914 0.0916 0.0918 0.0920 fn(xn) (b) Normalized DeepSet [9] output ( Lipschitz transferable) 10 1 10 2 10 3 10 4 Set size n 0.122 0.124 0.126 0.128 fn(xn) (c) PointNet [66] output (compatible, not transferable) 10 1 10 2 10 3 10 4 Set size n 10 -5 10 -4 | fn(xn) -f∞(µ)| n -1/2 (d) Normalized DeepSet [9] error step graphons with graphon operator/cut norm V ∞ = general graphons V ∞ = add limit points Duplication operator/cut norm
ℝ n×n sym ↪ ℝ kn×kn sym A ↦ A ⊗ 1 k 1 ⊤ k Limit space V ∞ distance = W A1 -W A2 "=" interval permutations 𝖦 ∞ = A 1 = A 2 = Orbit closures orbit of W A1 orbit of W A2 symmetrized distance = min σ∈𝖦∞ W A1 -σ ⋅ W A2
this section cite: ['b53', 'b46']

Section: Equivalence classes of step graphons
Equivalence classes of general graphons Message Passing Neural Networks (MPNNs) Message-passing-based GNNs form the most widely used and general paradigm, encompassing many existing models [31]. Instantiating Proposition 5.1, we analyze in Appendix G.2 the transferability of MPNNs under constraints on the message function, update function, and local aggregation, and compare our results with [70,47].
Making Invariant Graph Networks (IGN) Transferable IGNs [56] are an alternative approach to designing GNNs by alternating linear S n -equivariant layers with pointwise nonlinearities. IGN is incompatible with respect to V G dup because the linear layers used in [56] are incompatible. Nevertheless, the hypotheses in Proposition 5.1 provide a systematic approach for modifying IGN to achieve transferability, leading to two newly proposed models. We highlight this recipe for constructing transferable neural networks is general.
this section cite: ['b69', 'b46', 'b55', 'b55']

Section: Generalizable Graph Neural Network (GGNN):
We use compatible linear layers and messagepassing-like nonlinearities of the form (A, X) → A, σ S s=0 n -s A s X s , where σ acts entrywise. The GGNNs are compatible and at least as expressive as the GNNs in [70]. Continuous GGNN: We restrict the linear layers in GGNN to the subspace that has bounded operator norm on the limit space. This model is transferable in both operator 2-norm and the cut norm.
The complete description of these models and proof of their transferability are in Appendix G.3. The transferability of various GNN models with respect to (V G dup , ∥ • ∥ op,p ), p = 2 is illustrated in Figure 4 through a numerical experiment.
this section cite: ['b69']

Section: Point clouds
Consistent sequences for point clouds. For k-dimensional point clouds, we consider a sequence of vector spaces V n = R n×k , representing sets of n points in R k , where k is fixed (typically k = 2
10 1 10 2 Graph size n 0.099 0.100 0.101 0.102 0.103 fn(xn) 2 Erdos-Renyi p = 1/2 Fully connected w = 1/2 (a) GNN [70] (locally Lipschitz transferable) 10 1 10 2 Graph size n 0.01 0.02 0.03 0.04 0.05 fn(xn) 2 Erdos-Renyi p = 1/2 Fully connected w = 1/2 (b) Normalized 2-IGN [11] (incompatible) 10 1 10 2 Graph size n 8 10 12 fn(xn) 2 Erdos-Renyi p = 1/2 Fully connected w = 1/2 (c) GGNN (compatible, not transferable) 10 1 10 2 Graph size n 2.2 2.3 2.4 2.5 2.6 2.7 fn(xn) 2 , Xn = 1n. (solid lines) An is drawn i.i.d. from the Erdős-Rényi model G(n, 1/2), with Xn = 1n. These two sequences represent different samplings of the same underlying constant graphon signal, where W ≡ 1/2 and f ≡ 1. Error bars indicate one standard deviation above and below the mean over 100 random samples. (dashed lines): For the fully connected model each finite graph signal exactly induces the underlying graphon signal. The outputs of all compatible models ((a), (c), (d)) remain constant over n, whereas the incompatible model (b) does not. (solid lines): The outputs of all transferable models ((a), (d)) converge to the same limit as Sequence (1), while the discontinuous model (c) does not.
or 3). Unlike the sets models in Section 5.1, here we consider not only the permutation symmetry S n , which acts on the rows and ensures invariance to the ordering of points, but also the additional symmetry given by the orthogonal group O(k), which acts on the columns to capture rotational and reflectional symmetries in the Euclidean space R k . i.e. (g, h) • X = gXh ⊤ .
We consider the duplication consistent sequences of point clouds V P dup . Specifically, for any n | N , the embedding is defined as
φ N,n (X) = X ⊗ 1 N/n . The group embedding S n × O(k) ↩→ S N × O(k)
is given by (g, h) → (g ⊗ I N/n , h). We further endow each V n with the normalized ℓ p norm
∥X∥ p = 1 n n i=1 ∥X i: ∥ p 2 1/p .
The orbit closures of V ∞ can be identified with the orbit space under O(k) of probability measures on R k with finite pth moment.
Invariant neural networks on point clouds. First, we analyze the DeepSet for Conjugation Invariance (DS-CI) proposed in [5]. In Appendix H.2, we show that the normalized variants of DS-CI is "approximately" locally Lipschitz transferable with respect to (V P dup , ∥ • ∥ p ). We also introduce a more time and space-efficient model, normalized SVD-DeepSet, defined as: SVD-DS n (X) = DeepSet n (XV ), where for an input point cloud X ∈ R n×k , we compute its singular value decomposition X = U ΣV ⊤ with ordered singular values. This model effectively applies a canonicalization step with respect to the O(k) action. We prove that it is locally Lipschitz-transferable outside of a zero-measure set (which exists for any canonicalization [23]) in Appendix H.3. In Appendix H.3, we illustrate the transferability of these models when input point clouds are i.i.d. samples from an O(k)-invariant measure on R k through a numerical experiment.
this section cite: ['b4', 'b22']

Section: Size generalization experiments
Our theoretical framework emphasizes understanding how solutions to small problem instances inform solutions to larger ones, i.e., whether the target function is compatible with respect to a given consistent sequence, and continuous with respect to an associated norm. Under this view, size generalization can be provably achieved using a neural network model that is aligned with the target function: Compatibility alignment ensures that the model is compatible with respect to the same consistent sequence as the target function, and yields a function on the correct limit space. Continuity alignment further requires the model to be continuously transferable with respect to the same norm as the target function, providing stronger inductive bias and asymptotic guarantees.
We evaluate the effect of these alignments through experiments where models are trained on small inputs of fixed size n train and tested on larger inputs n test ≥ n train . In each experiment, all models have approximately the same number of parameters to ensure fair comparisons. In this Section we show a selected set of results. See Appendix I for a full discussion.
500 1500 2500 3500 4500 Test set size (n) 10 -4 10 -3 10 -2
10 -1 10 0 Test MSE DeepSet (incompatible) Normalized DeepSet (transferable) PointNet (compatible, not transferable) (a) Set: Population statistics 20 60 100 140 180 Test set size (n) 10 -5 10 -3 10 -1 10 1 Test MSE DeepSet (incompatible) Normalized DeepSet (compatible, not transferable) PointNet (transferable) (b) Set: Maximal distance from the origin 50 200 500 1000 2000 Test graph size (n) 10 -6 10 -5 Test MSE Normalized 2-IGN (incompatible) GNN (transferable) GGNN (compatible, not transferable) Continuous GGNN (transferable) (c) Graph: Weighted triangle densities 20 100 200 300 500 Test pointcloud sizes (n) 10 -2 10 -1 10 0 10 1 10 2 Test MSE Unnormalized SVD-DS (incompatible) Normalized SVD-DS (transferable) Normalized DS-CI (approximately transferable) (d) Point cloud: Gromov-Wasserstein lower bound Size generalization on sets. We consider two learning tasks on sets of arbitrary size, where the target functions exhibit distinct properties, leading to different models being more suitable.
this section cite: []

Section: Experiment 1: Population Statistics.
We tackle Task 3 from Section 4.1.1 of [91] (other tasks yield similar results). The dataset consists of N sample sets. We first randomly choose a unit vector v ∈ R 32 . For each set, we sample a parameter λ ∈ [0, 1] and then generate a set of n points from µ(λ) = N (0, I + λvv ⊤ ). The target function involves learning the mutual information, which depends on the underlying probability measure µ(λ) and is continuous with respect to the Wasserstein p-distance. Based on the transferability analysis in Table 2, normalized DeepSet is well-suited for this task, as it aligns with the target function at the continuity level, whereas PointNet aligns only at the compatibility level, and DeepSet lacks alignment at either level. Indeed, normalized DeepSet has the best in-distribution test performance and size generalization behavior as shown in Figure 5a.
this section cite: []

Section: Experiment 2: Maximal Distance from the Origin.
In this task, each dataset consists of N sets where each set contains n two-dimensional points sampled as follows. First, a center is sampled from N (0, I 2 ) and a radius is sampled from Unif([0, 1]), which together define a circle. The set then consists of n points sampled uniformly along the circumference. The goal is to learn the maximum Euclidean norm among the points in each set. The target function in this case only depends on a point cloud via its support, and is continuous with respect to the Hausdorff distance. Consequently, for this task, PointNet is well-suited, as it aligns with the task at the level of continuity. (It was proved in [9] that PointNet extends to a function that is continuous with respect to the Hausdorff distance.
We discuss the relationship between this result and ours in the PointNet part of Appendix F.
y i = 1 n 2 j,k∈[n] A ij A jk A ki x i x j x k
, which depends on the underlying graphon and is continuous with respect to the cut norm. Figure 5c shows that our proposed continuous GGNN, which aligns with the target function at the continuity level, achieves the best test performance on large graphs. Although the message-passing GNN is also provably transferable, it lacks sufficient expressiveness for this task [14], leading to poor performance.
Size Generalization on Point Clouds. We follow the setup in Section 7.2 of [5], using the Model-Net10 dataset [87]. From Class 2 (chair) and Class 7 (sofa), we randomly selected 80 point clouds each, splitting them into 40 for training and 40 for testing. This results in 40 × 40 cross-class pairs. The objective is to learn the third lower bound of the Gromov-Wasserstein distance [60], which is invariant and continuous with respect to the Wasserstein p-metric (proven in Appendix I.4). Figure 5d shows that normalized DS-CI and normalized SVD-DS, both aligned with the target at the continuity level, achieve good test performance and size generalization. While normalized SVD-DS underperforms compared to normalized DS-CI, it offers superior time and memory efficiency.
[91] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. Advances in neural information processing systems, 30, 2017.
[92] Aaron Zweig and Joan Bruna. A functional perspective on learning symmetric functions with neural networks. In International Conference on Machine Learning, pages 13023-13032. PMLR, 2021.
this section cite: ['b8', 'b13', 'b4', 'b86', 'b59']

Section: A Notation
We use R and N to denote the sets of real numbers and natural numbers. We use the pair (N, ⪯) to denote a directed poset on natural numbers, used for indexing. We write [n] to denote the set {1, 2, . . . , n}. The symbols V and U are used for consistent sequences, and V n , U n for the corresponding vector spaces at finite levels. General groups are denoted by G n ; we use S n for the symmetric group and O(k) for the orthogonal group. We denote embeddings of vector spaces from dimension n to N by φ N,n and ψ N,n . Embeddings of groups are denoted by θ N,n . We use "•" to denote the composition of functions, and id for the identity element in a group or the identity map, depending on the context. Binary operations in groups are denoted by " * ", with subscripts such as " * n " used when clarity is needed, e.g., to indicate the operation in the group G n . Group actions are denoted by "•", with subscripts such as "• n " when needed.
We write ∼ to denote equivalence relations, and use [x] to denote the equivalence class of x, i.e., [x] = {y : y ∼ x}. We write ∨ and ∧ for maximum and minimum respectively. In a metric space, we use B(x 0 , r) to denote the ball centered at x 0 with radius r, i.e. B(x 0 , r) := {x : d(x 0 , x) < r}.
Given a set S, the symbol S denotes either completion or closure, depending on the context. Given a function f, the symbol f denotes a normalized variant of that function. Given two sequences (a n ), (b n ) ⊆ R, we write a n ≲ b n if there exists a constant C > 0 such that a n ≤ Cb n for all n ∈ N.
For a given function R : N → R + , we say a sequence (x n ) converges to x at rate R(n) with respect to distance metric d if d(x n , x) ≲ R(n) and R(n) → 0. For a matrix X, we use X i: to denote the vector formed by its i-th row. This work employs several norms: ∥ • ∥ p refers to the standard ℓ p -norm (for vectors and infinite sequences) or L p -norm (for functions), while ∥ • ∥ p denotes their normalized counterparts which will be defined later. The operator norm of a linear map T : V → W between two normed spaces is denoted ∥T ∥ op := sup ∥v∥=1 ∥T v∥; its definition inherently depends on the specific norms chosen for V and W . Particularly, ∥ • ∥ op,p,q denotes the operator norm when the domain is equipped with the ℓ p -norm (or L p -norm) and the codomain with the ℓ q -norm (or L q -norm), we use ∥ • ∥ op,p as an abbreviation for ∥ • ∥ op,p,p .
this section cite: []

Section: B Related work
GNN transferability. The work on GNN transferability under the graphon framework was pioneered by [70], focusing on a variant of graph convolutional network (GCN) for deterministic graphs obtained from the same graphon. In parallel, [48] explores transferability with respect to an alternative limit space to graphon in the form of a topological space, and [41] examines an arguably equivalent notion-convergence and stability-by analyzing random graphs sampled from a graphon. This line of research has since been further developed [71,58], extending to more general message-passing networks (MPNNs) [18], more general notion of graph limits [46], and other models [73,81,11,35].
Our framework unifies and recovers several of the above-mentioned results, which we briefly discuss in Appendix G.2. Furthermore, we develop new insights into the transferability of Invariant Graph Networks (IGNs) [56], a topic previously examined in [11,35]. We leverage our framework to advance this line of inquiry and elaborate on the connections between our approach and prior work in Appendix G.3.
this section cite: ['b69', 'b47', 'b40', 'b70', 'b57', 'b17', 'b45', 'b72', 'b80', 'b10', 'b34', 'b55', 'b10', 'b34']

Section: GNN generalization.
A related line of research concerns the generalization theory of GNNs. Generalization bounds have been derived based on various frameworks, including Rademacher complexity [28,55], VC dimension [75,62,24], the PAC-Bayesian approach [52,39], the neural tangent kernel [22], and covering numbers [59,57,47,79,68]. Notions of size generalization without an explicit notion of convergence have been studied in [89,4]. For a comprehensive overview, we refer the reader to the recent survey [80]. While most research on GNN generalization bounds consider graphs of fixed or bounded size, [47,68] explore the "uniform regime," addressing unbounded graph sizes. This perspective is closely aligned with the transferability theory, as they both consider continuous extensions over suitable limit spaces. Our work formalizes this connection, showing that transferability implies generalization (Section 4, Appendix E). We derive a generalization bound via covering numbers using the same proof strategy as prior works, but it applies well beyond GNNs and offers a setting that directly connects to the notion of size generalization.
this section cite: ['b27', 'b54', 'b74', 'b61', 'b23', 'b51', 'b38', 'b21', 'b58', 'b56', 'b46', 'b78', 'b67', 'b88', 'b3', 'b79', 'b46', 'b67']

Section: Equivariant machine learning.
Our theory naturally applies to equivariant machine learning models, i.e., neural networks with symmetries imposed. We particularly focus on models from [91, 56,66,72,5]. There are many other equivariant machine learning models we haven't discussed here, such as the ones expressed in terms of group convolutions [17,45,33], representation theory [44,78,29], canonicalization [40], and invariant theory [83,6,84], among others.
Representation stability and any-dimensional learning. As noted by [49,50], the presence of symmetry allows functions operating on inputs of arbitrary dimension to be parameterized with a finite number of parameters, which may be partially explained by the theory of representation stability [15]. Leveraging techniques from this theory, [50] provides the first general theoretical framework for any-dimensional equivariant models. Recent work applies similar techniques to study the generalization properties across dimensions of any-dimensional regression models [20]. Our work builds on the theoretical foundation established by this line of research.
this section cite: ['b55', 'b65', 'b71', 'b4', 'b16', 'b44', 'b32', 'b43', 'b77', 'b28', 'b39', 'b82', 'b5', 'b83', 'b48', 'b49', 'b14', 'b49', 'b19']

Section: Any-dimensional expressivity and universality.
Our framework naturally prompts the question of expressivity for any-dimensional neural networks on their limit spaces. While we do not pursue this direction here, related questions have been independently studied. [9] shows that normalized DeepSets and PointNet are universal for uniformly continuous functions on suitable limit spaces, closely tied to our results (see Appendix F.2.1). In GNNs, the notion of "uniform expressivity"-expressing logical queries without parameter dependence on input size-has been explored in [34,2,42,69].
Our framework offers complementary insights despite differing foundations.
C Consistent sequences and compatible, transferable maps: details and missing proofs from Section 2 C.1 Consistent sequences and limit space
The concept of consistent sequences originated in the theory of representation stability [16]. We generalize this notion, originally considering a sequence of vector spaces, by allowing indexing over any directed poset on the natural numbers. Although we focus on natural numbers in this work, our theory generalizes to any directed poset.
Directed poset indexing. We require the indexing set (N, ⪯) to be a directed poset on natural numbers, meaning that N is the set of natural numbers equipped with a binary operation ⪯ satisfying:
(Partial order) The binary operation ⪯ is a partial order; that is, it satisfies: reflexivity (a ⪯ a for all a ∈ N), transitivity (if a ⪯ b and b ⪯ c, then a ⪯ c), and antisymmetry (if a ⪯ b and b ⪯ a, then a = b). (Upper bound condition) For every pair a, b ∈ N, there exists c ∈ N such that a ⪯ c and b ⪯ c.
The directed poset indexing generalizes the notion of sequences to allow a more complex and flexible way of defining how smaller problem instances are embedded into larger ones, permitting "branching" directions of growth. We will see that the upper bound condition is crucial, as it ensures that any two problem instances are comparable-meaning they can both be embedded into a third, larger problem dimension and compared there. In this work, we only consider two cases: the natural numbers with the standard ordering ≤, and with the divisibility ordering, where a ⪯ b if and only if a | b.
Definition C.1 (Consistent sequence: detailed version of Definition 2.1). A consistent sequence of group representations over directed poset (N, ⪯) is V = {(V n ) n∈N , (φ N,n ) n⪯N , (G n ) n∈N }, where 1. (Groups) (G n ) is a sequence of groups indexed by N such that whenever n ⪯ N , G n is embedded into G N via an injective group homomorphism θ N,n : G n → G N ,
where
θ i,i = id Gi for all i ∈ N, θ k,j • θ j,i = θ k,i whenever i ⪯ j ⪯ k in N. 2. (Vector spaces) (V n ) is a sequence of (finite-dimensional, real) vector spaces indexed by N, such that each V n is a G n -representation, and whenever n ⪯ N , V n is embedded into V N through a linear embedding φ N,n : V n ↩→ V N ,
where
φ i,i = id Vi for all i ∈ N, φ k,j • φ j,i = φ k,i whenever i ⪯ j ⪯ k in N.
this section cite: ['b8', 'b33', 'b1', 'b41', 'b68', 'b15']

Section: (Equivariance) Every
φ N,n is G n -equivariant, i.e., φ N,n (g • v) = θ N,n (g) • φ N,n (v) for all g ∈ G n , v ∈ V n .
Given a consistent sequence, we can first "summarize" the sequence of groups (G n ) into a single limit group G ∞ , and likewise "summarize" the sequence of vector spaces (V n ) into a single limit vector space V ∞ . We can then consider the action of G ∞ on V ∞ .
this section cite: []

Section: Definition C.2 (Limit group: detailed version of Definition 2.3).
The limit group G ∞ is defined as the disjoint union n G n modulo the equivalence relation that identifies each element g ∈ G n with its images under the transition maps θ N,n (g) for all N ≥ n, i.e.,
G ∞ := n G n / ∼,
where g ∼ θ N,n (g) whenever n ⪯ N and g ∈ G n . This construction is also known as the direct limit of groups, and is denoted by
G ∞ = lim -→ G n . For each g ∈ G n , its equivalence class in G ∞ is denoted by [g]
, representing the corresponding limiting object.
The group structure on G ∞ is inherited from the groups (G n ) as follows. For g n ∈ G n and g m ∈ G m , define the binary operation on equivalence classes by
[g n ] * [g m ] := [θ N,n (g n ) * N θ N,m (g m )] ,
where N ∈ N is a common upper bound of n and m in (N, ⪯), and * N denotes the group operation in G N . It is straightforward to check that this operation is well-defined.
this section cite: []

Section: Definition C.3 (Limit space of consistent sequence: detailed version of Definition 2.3).
Define V ∞ as the disjoint union V n modulo an equivalence relation identifying each element v ∈ V n with its images under the transition map φ N,n (v) for all n ⪯ N , i.e.,
V ∞ := n V n / ∼,
where v ∼ φ N,n (v) whenever n ⪯ N . This construction is also known as the direct limit of vector spaces, and is denoted as
V ∞ = lim -→ V n .
The vector space structure on V ∞ is inherited from the vector spaces (V n ) as follows. For v n ∈ V n , v m ∈ V m , the addition and scalar multiplication on the equivalent classes are defined by
[v n ] + [v m ] := [φ N,n (v n ) + φ N,m (v m )] where N is an upper bound for n, m in (N, ⪯), λ[v n ] := [λv n ]. It is straightforward to check that these operations are well-defined. The limit group G ∞ acts on V ∞ by [g] • [v] := [θ N,n (g) • N φ N,m (v)] for g ∈ G n , v ∈ V m , where N is an upper bound of n, m in (N, ⪯), and • N is the group action of G N on V N . It is also easy to check that this group action is well-defined, and V ∞ is a G ∞ -representation. The orbit space of V ∞ under the action of G ∞ is {G ∞ • x : x ∈ V ∞ } , where G ∞ • x := {g • x : g ∈ G ∞ } is the orbit of point x ∈ V ∞ under the G ∞ -action. The orbits form a partition of V ∞ into disjoint subsets.
Consistent sequences without symmetries. A special case of consistent sequences arises when G n = {id}, the trivial group, for all n ∈ N. In this case, the structure reduces to a directed system of vector spaces
V = {(V n ) n∈N , (φ N,n ) n⪯N } .
Hence, our theory of size generalization applies in scenarios where no intrinsic symmetries are present.
Trivial consistent sequence. Another special case arises when V n = V , a fixed vector space, for all n ∈ N, with embeddings given by φ N,n = id V for all n ⪯ N . This yields the trivial consistent sequence associated with V , which we denote by V V . This construction is useful for modelling non-size-dependent spaces, such as the output space in graph-level classification or regression tasks.
Direct sum and tensor product. Given a consistent sequence V = {(V n ), (φ N,n ), (G n )}, we can define its direct sum and tensor product. Both of them are also consistent sequences.
Definition C.4. The d-th direct sum of V is defined as
V ⊕d := {(V ⊕d n ), (φ ⊕d N,n ), (G n )}, where V ⊕d n denotes the direct sum of d copies of V n and φ ⊕d N,n : V ⊕d n → V ⊕d N is defined by applying φ N,n to each component. The group G n acts on V ⊕d n by simultaneously acting on every copy of V n , i.e. g • (v 1 , . . . , v d ) := (g • v 1 , . . . , g • v d ).
Definition C.5. The d-th tensor product of V is defined as
V ⊗d := {(V ⊗d n ), (φ ⊗d N,n ), (G n )}, where V ⊗d n denotes the d-fold tensor product of V n . φ ⊗d N,n : V ⊗d n → V ⊗d N is uniquely defined by φ ⊗d N,n (v 1 ⊗ • • • ⊗ v d ) := φ N,n (v 1 ) ⊗ . . . φ N,n (v d ), and the group action of G n on V ⊗d n is uniquely defined by g • (v 1 ⊗ • • • ⊗ v d ) := (g • v 1 ) ⊗ • • • ⊗ (g • v d ).
The universal property of tensor product guarantees that φ ⊗d N,n and the group action mentioned are well-defined. Similarly, we also consider the d-th symmetric tensors of V as
Sym d (V) := {(Sym d (V n )), (φ ⊗d N,n ), (G n )}, where Sym d (V n )
denotes the space of symmetric tensors of order d defined on V n , i.e. the subspace of V ⊗d n invariant under the action of the symmetric group S d .
The direct sum, V ⊕d , is particularly useful for incorporating hidden channels into our analysis, as it effectively adds the extra channel dimensions to our data. In contrast, the tensor product,V ⊗d , is helpful to extend a consistent sequence on vectors or sets to higher-order objects such as matrices or graphs. For example, the duplication consistent sequences for graphs exactly arise as the 2nd symmetric tensors of the duplication consistent sequences for sets.
this section cite: []

Section: C.2 Compatible maps
Recall from Definition 2.4 that a sequence of maps (f n : V n → U n ) is compatible with respect to the consistent sequences V, U if, for all n ⪯ N ,
f N • φ N,n = ψ N,n • f n ,
and each f n is G n -equivariant. This condition is equivalent to the existence of an extension to the limit map f ∞ .
Proposition C.6 (Compatible maps and extension to limit).
Let V = {(V n ), (φ N,n ), (G n )} and U = {(U n ), (ψ N,n ), (G n )} be two consistent sequences. A sequence of maps (f n : V n → U n ) is
compatible if and only if it extends to the limit; that is, there exists a G ∞ -equivariant map
f ∞ : V ∞ → U ∞ such that f n = f ∞ | Vn for all n. Proof. (⇐) Suppose there exists a G ∞ -equivariant map f ∞ such that f n = f ∞ | Vn for all n. Then for all n ⪯ N and x ∈ V n , [f n (x)] = f ∞ ([x]) = f ∞ ([φ N,n (x)]) = [f N (φ N,n (x))], which implies f N • φ N,n = ψ N,n • f n . Moreover, for all n ∈ N, x ∈ V n , and g ∈ G n , [f n (g • x)] = f ∞ ([g • x]) = f ∞ ([g] • [x]) = [g] • f ∞ ([x]) = [g] • [f n (x)] = [g • f n (x)], so each f n is G n -equivariant. (⇒) Conversely, suppose (f n ) are compatible. Define f ∞ : V ∞ → W ∞ , f ∞ ([x]) := [f n (x)] if x ∈ V n .
Compatibility ensures this is well-defined. To verify equivariance, let g ∈ G m and x ∈ V n , and let N be a common upper bound of n and m in (N, ⪯). Then,
f ∞ ([g] • [x]) = f ∞ ([θ N,m (g) • φ N,n (x)]) = [f N (θ N,m (g) • φ N,n (x))] = [θ N,m (g) • f N (φ N,n (x))] = [θ N,m (g) • ψ N,n (f n (x))] = [g] • f ∞ ([x]). Therefore, f ∞ is G ∞ -equivariant.
This proposition implies that learning a function on the infinite-dimensional space V ∞ , a task that may appear difficult, reduces to learning a compatible sequence of functions on the finite-dimensional vector spaces along the sequence, which is a more tractable problem.
this section cite: []

Section: C.3 Metrics on consistent sequences
In Section 2.1, we introduced a norm on V ∞ so as to define distance between objects of different dimensions. In this appendix, we take a more general perspective and examine metric structures on consistent sequences, and present detailed proofs. The same proofs carry over to the norm setting with minimal modification.
this section cite: []

Section: Definition C.7 (Compatible metrics: generalized version of Definition 2.5).
Let V = {(V n ), (φ N,n ), (G n )} be a consistent sequence. A sequence of metrics (d n ) on the vector spaces V n
is said to be compatible if all the embeddings φ N,n and the G n -actions are isometries. That is, for all n ⪯ N , x, y ∈ V n , and g ∈ G n , we have:
d N (φ N,n (x), φ N,n (y)) = d n (x, y), and d n (g • x, g • y) = d n (x, y).
Similar to compatible maps, this is equivalent to the existence of an extension to a metric d ∞ on V ∞ .
Proposition C.8 (Compatible metrics and extension to the limit). A sequence of metrics (d n ) on the spaces V n is compatible if and only if it extends to a metric on the limit space. That is, there exists a metric d ∞ on V ∞ such that
d n (x, y) = d ∞ ([x], [y]) for all n ∈ N, x, y ∈ V n ,
and the G ∞ -action on V ∞ is an isometry with respect to d ∞ , i.e.,
d ∞ (g • x, g • y) = d ∞ (x, y) for all x, y ∈ V ∞ , g ∈ G ∞ .
Proof. The proof is primarily a matter of bookkeeping, similar in spirit to Proposition C.6. For completeness, we present the full argument below.
(⇐) Suppose (d n ) extends to a metric d ∞ on V ∞ . Then for all n ⪯ N and x, y ∈ V n , we have
d n (x, y) = d ∞ ([x], [y]) = d ∞ ([φ N,n (x)], [φ N,n (y)]) = d N (φ N,n (x), φ N,n (y)).
Thus, the embeddings φ N,n are isometries. Moreover, for any g ∈ G n and x, y ∈ V n ,
d n (g • x, g • y) = d ∞ ([g • x], [g • y]) = d ∞ ([g] • [x], [g] • [y]) = d ∞ ([x], [y]) = d n (x, y),
so the group actions are isometries as well.
(⇒) Conversely, suppose the collection (d n ) is compatible. Define a metric d ∞ : V ∞ × V ∞ → R as follows: for x ∈ V n and y ∈ V m , let N be any common upper bound of n and m in (N, ⪯), and set
d ∞ ([x], [y]) := d N (φ N,n (x), φ N,m (y)).
Compatibility of the metrics ensures that d ∞ is well-defined. It is also easy to check that d ∞ is a metric. Moreover, by construction,
d n = d ∞ | Vn for all n.
To verify that the G ∞ -action is isometric, take x ∈ V n1 , y ∈ V n2 , and g ∈ G n for some n ∈ N. Let N be a common upper bound of n, n 1 , n 2 in (N, ⪯). Then,
d ∞ ([g] • [x], [g] • [y]) = d N (θ N,n (g) • φ N,n1 (x), θ N,n (g) • φ N,n2 (y)) = d N (φ N,n1 (x), φ N,n2 (y)) = d ∞ ([x], [y]
). This completes the proof.
With the metric structure in place, we define the limit space via the completion of the metric space. The completion of a metric space M is a complete metric space M -that is, a space in which every Cauchy sequence converges-that contains M as a dense subset (i.e., the smallest closed subset of M containing M is M itself). Definition C.9 (Limit space: detailed version of Definition 2.7). Let V be a consistent sequence, and let V ∞ be equipped with the metric d ∞ . Denote by V ∞ the completion of V ∞ with respect to d ∞ . The G ∞ -action on V ∞ extends to a well-defined action on V ∞ as follows: for any
x ∈ V ∞ and g ∈ G ∞ , choose a sequence (x n ) in V ∞ such that x n → x in V ∞ , and define g • x := lim n→∞ g • x n .
This limit exists because (g • x n ) is a Cauchy sequence, as the G ∞ -action on V ∞ is isometric. The resulting action on V ∞ is linear and isometric. We define the limit space of the consistent sequence
V to be the G ∞ -representation V ∞ . The set of orbit closures in V ∞ under the action of G ∞ is G ∞ • x : x ∈ V ∞ .
where G ∞ • x is the closure of the orbit G ∞ • x. Intuitively, V ∞ includes not only elements from finite-dimensional objects (elements in V ∞ ), but also additional points that are "reachable" as limits of finite-dimensional objects. We can further define a symmetrized metric on the limit space.
Proposition C.10 (Symmetrized metric). Let x, y ∈ V ∞ . Define d(x, y) := inf g∈G∞ d ∞ (g • x, y).
Then d is a pseudometric on V ∞ and induces a metric on the space of orbit closures in V ∞ under the G ∞ -action. We refer to d as the symmetrized metric.
Proof. The non-negativity of d follows directly from the non-negativity of d ∞ .
(Symmetry) Since d ∞ (g • x, y) = d ∞ (x, g -1 • y) for any g ∈ G ∞ by isometry of the G ∞ -action, taking the infimum over all g ∈ G ∞ (equivalently over g -1 ) yields d(x, y) = d(y, x). (Triangle inequality) Let ε > 0. Then there exist g, h ∈ G ∞ such that d(x, y) > d ∞ (g • x, y) -ε, d(y, z) > d ∞ (h • y, z) -ε.
Using the isometry of the group action:
d(x, y) + d(y, z) > d ∞ (g • x, y) + d ∞ (h • y, z) -2ε = d ∞ (hg • x, h • y) + d ∞ (h • y, z) -2ε ≥ d ∞ (hg • x, z) -2ε ≥ d(x, z) -2ε.
Since this holds for arbitrary ε > 0, the triangle inequality follows.
(Definiteness) We have d(x, y) = 0 if and only if there exists a sequence
(g n ) ⊆ G ∞ such that d ∞ (g n • x, y) → 0.
This is precisely the condition that y ∈ G ∞ • x, i.e., x and y lie in the same orbit closure.
Therefore, d is a pseudometric on V ∞ , and descends to a true metric on the space of orbit closures which completes the proof.
this section cite: []

Section: D Transferability: details and missing proofs from Section 3 D.1 Transferable maps
Following Definition 3.1 of continuously, L-Lipschitz, and L(r)-locally Lipschitz transferable, we further define the following notion: If (f n ) is a sequence of equivariant maps extending to f ∞ which is locally Lipschitz at x 0foot_6 , we say (f n ) is locally Lipschitz transferable at x 0 . Being locally Lipschitz transferable at x 0 is a weaker condition than being L(r)-locally Lipschitz transferable, which is itself weaker than L-Lipschitz transferable. This definition is useful when studying models which are discontinuous on negligible sets of inputs, and which are therefore not L(r)-locally Lipschitz transferable. These often come up when constructing architectures based on canonicalizations [23], as commonly done for point clouds for instance (see Section 5.3).
We first show a useful property that continuity/Lipschitz with respect to ∥ • ∥ V∞ , ∥ • ∥ U∞ implies the same property with respect to the symmetrized metrics, even though the converse does not hold. We will again state and prove the results for metrics instead.
Proposition D.1 (Continuity in d ∞ implies continuity in d). Let V, U be consistent sequences. If f ∞ : V ∞ → U ∞ is continuous (respectively, L(r)-Lipschitz on B(0, r) for all r > 0, L-Lipschitz, locally Lipschitz at x 0 ∈ V ∞ ) with respect to d V
∞ and d U ∞ , then f ∞ satisfies the same property with respect to the symmetrized metrics d V and d U .
Proof. (Continuity) Let x n → x in V ∞ with respect to the symmetrized metric d V , and ε > 0. By continuity of f ∞ with respect to d ∞ , there exists δ > 0 such that whenever
d ∞ (x, y) < δ, d ∞ (f ∞ (x), f ∞ (y)) < ε. Take N such that for all n ≥ N , d V (x n , x) < δ 2 . Moreover, for each n, choose g n ∈ G ∞ such that d ∞ (g n • x n , x) ≤ d V (x n , x) + δ 2 . Then for all n ≥ N , d ∞ (g n • x n , x) < δ and hence d U (f ∞ (x n ), f ∞ (x)) ≤ d ∞ (g n • f ∞ (x n ), f ∞ (x)) = d ∞ (f ∞ (g n • x n ), f ∞ (x)) < ε. Therefore f ∞ (x n ) → f ∞ (x) with respect to d U . (L(r)-Lipschitz on B(0, r)) Suppose f ∞ is L(r)-Lipschitz on B(0, r) for all r > 0 with respect to d ∞ . Consider x, y such that d V (0, x) = d V ∞ (0, x) < r, and d V (0, y) = d V ∞ (0, y) < r. Then for any g ∈ G ∞ , we have d V ∞ (0, g • x) = d V ∞ (0, x) < r. Hence, d U (f ∞ (x), f ∞ (y)) ≤ d ∞ (g • f ∞ (x), f ∞ (y)) = d ∞ (f ∞ (g • x), f ∞ (y)) ≤ L(r)d ∞ (g • x, y). Take infimum over g ∈ G ∞ , get d U (f ∞ (x), f ∞ (y)) ≤ L(r) d V (x, y). (Lipschitz) For any x, y ∈ V ∞ , g ∈ G ∞ , d U (f ∞ (x), f ∞ (y)) ≤ d ∞ (g • f ∞ (x), f ∞ (y)) = d ∞ (f ∞ (g • x), f ∞ (y)) ≤ Ld ∞ (g • x, y). Take infimum over g ∈ G ∞ , get d U (f ∞ (x), f ∞ (y)) ≤ L d V (x, y). (Locally Lipschitz at x 0 ) Suppose d ∞ (f (x), f (x 0 )) ≤ Ld ∞ (x, x 0 ) whenever d ∞ (x, x 0 ) < r. Then for any such x we have d V (x, x 0 ) = inf g∈G (r) ∞ d ∞ (x, x 0 ) where G (r) ∞ = {g ∈ G ∞ : d ∞ (x, x 0 ) ≤ r}. Moreover, for any g ∈ G (r) ∞ we have d U (f (x), f (x 0 )) ≤ d ∞ (f (g • x), f (x 0 )) ≤ Ld ∞ (g • x, x), so after taking infimum over such g we conclude that d U (f (x), f (x 0 )) ≤ L d V (x, x 0 ) whenever d V (x, x 0 ) <
r, as desired. This completes the proof of the proposition.
Finally, we state and prove a set of more concrete characterizations of Lipschitz transferable sequence of functions defined in Definition 3.1, which are straightforward to check.
Proposition D.2. Let V, U be consistent sequences endowed with metrics.
this section cite: ['b22']

Section: (General case) A compatible sequence of functions (f
n : V n → U n ) is L-Lipschitz (respectively,
L(r)-locally Lipschitz) transferable if and only if for all n, f n is L-Lipschitz (respectively, L(r)-Lipschitz on B n (0, r) := {v ∈ V n : d V n (0, v) < r}). 2. (Linear maps) When the metrics are induced by norms, a compatible sequence of linear maps (W n : V n → U n ) is continuously (respectively, L-Lipschitz) transferable if and only if sup n ∥W n ∥ op < ∞ (respectively, sup n ∥W n ∥ op ≤ L).
Proof. We begin by deriving the result for the general case.
this section cite: []

Section: General case.
The "⇒" direction again follows immediately from
d n = d ∞ | Vn , f n = f ∞ | Vn for all n.
We focus on proving "⇐". First, by Proposition C.6, compatibility implies that the sequence
(f n ) extends to a function f ∞ : V ∞ → U ∞ .
(Lipschitz) Suppose f n is L-Lipschitz for all n. For any x ∈ V n and y ∈ V m , let N be a common upper bound of n and m in (N, ⪯). Then:
d U ∞ (f ∞ ([x]), f ∞ ([y])) = d U N (f N (φ N,n (x)), f N (φ N,m (y))) ≤ L d V N (φ N,n (x), φ N,m (y)) = L d V ∞ ([x], [y]). Hence, f ∞ is L-Lipschitz on V ∞ . Since Lipschitz continuity implies Cauchy continuity, f ∞ extends uniquely to f ∞ : V ∞ → U ∞ .
For any Cauchy sequences (x n ) and (y n ) in V ∞ with limits x, y ∈ V ∞ , we have:
d U ∞ (f ∞ (x), f ∞ (y)) = lim n→∞ d U ∞ (f ∞ (x n ), f ∞ (y n )) ≤ lim n→∞ L d V ∞ (x n , y n ) = L d V ∞ (x, y),
which shows that f ∞ remains L-Lipschitz after extending to V ∞ .
(L(r)-locally Lipschitz) Suppose each f n is L(r)-Lipschitz on B n (0, r) for all r > 0. As above, for any x ∈ V n and y ∈ V m with d n (0, x), d m (0, y) < r, let N be a common upper bound of n, m in (N, ⪯). Then φ N,n (x), φ N,m (y) ∈ B N (0, r), and by the L(r)-Lipschitz property of f N on B N (0, r), we get
d U ∞ (f ∞ ([x]), f ∞ ([y])) ≤ L(r) • d V ∞ ([x], [y]). Thus, f ∞ : V ∞ → U ∞ is L(r)-Lipschitz on {v ∈ V ∞ : d V ∞ (0, v) < r}.
This implies f ∞ is Cauchy continuous: any Cauchy sequence (x n ) lies within some ball of radius R, and since f ∞ is Lipschitz continuous there, (f ∞ (x n )) is also Cauchy. Hence, f ∞ extends uniquely to V ∞ , and it is easy to check that it is L(r)-Lipschitz on B(0, r) for all r.
Linear maps. We leverage the argument for the general case to derive the two stated claims.
(Lipschitz) By our argument for normed spaces, (W n ) is L-Lipschitz transferable if and only if for all n, W n is L-Lipschitz. By linearity of each W n , this is equivalent to ∥W n ∥ op ≤ L for all n.
(Continuity) It is sufficient to prove that (W n ) is continuously transferable if and only if it is L-Lipschitz transferable for some L > 0. The "⇐" direction is immediate. To prove "⇒", suppose (W n ) extends to a continuous function W ∞ : V ∞ → U ∞ . Then W ∞ is linear on V ∞ because for any x ∈ V n , y ∈ V m , and any common upper bound N of n, m in (N, ⪯), we have
W ∞ (a[x] + b[y]) = [W N (aφ N,n (x) + bφ N,m (y))] = aW ∞ ([x]) + bW ∞ ([y]).
By continuity of W ∞ , it remains linear on V ∞ . The result follows since for linear operators, Lipschitz continuity and continuity are equivalent.
Thus, the proof is finished.
this section cite: []

Section: D.2 Convergence, transferability and stability
Stability. The following stability result states that small perturbations of the input (e.g., adding a small number of nodes to a graph) lead to small changes in the output. It resembles the stability considered in [70].
this section cite: ['b69']

Section: Proposition D.3 (Stability: detailed version of Proposition 3.2).
If the sequence of maps (f n : V n → U n ) is L(r)-locally Lipschitz transferable, then for any two inputs x n ∈ V n and x m ∈ V m of any two sizes n, m with d
V n (0, x n ), d V m (0, x m ) ≤ r, we have d U ∞ ([f n (x n )], [f m (x m )]) ≤ Ld V ∞ ([x n ], [x m ]
). Moreover, the same holds when replacing every d ∞ with the symmetrized metric d.
Convergence and transferability: deterministic sampling. For a sequence of inputs (x n ) sampled (deterministically) from the same underlying limiting object x, the outputs of a transferable function satisfy f n (x n ) ≈ f m (x m ) for big n, m, and converge as f n (x n ) → f ∞ (x). We provide examples of sampling procedures later in Appendix D.3.
this section cite: []

Section: Proposition D.4 (Convergence and transferability: detailed version of Proposition 3.2).
Let (x n ∈ V n ) n∈N be a sequence of inputs sampled from a limiting object x ∈ V ∞ , such that [x n ] → x at a rate R(n) with respect to d V ∞ . 1. (Asymptotic) If (f n : V n → U n ) n∈N is continuously transferable, then the following holds. (Convergence) The sequence [f n (x n )] → f ∞ (x) with respect to d U ∞ . (Transferability) The distance d U ∞ ([f n (x n )], [f m (x m )]) → 0 as n, m → ∞. 2. (Nonasymptotic) If (f n : V n → U n ) n∈N is locally Lipschitz transferable at x, then the following holds. (Convergence) The sequence [f n (x n )] → f ∞ (x) at a rate R(n) with respect to d U ∞ . (Transferability) The distance is bounded by d U ∞ ([f n (x n )], [f m (x m )]) ≲ R(n) + R(m).
That is, Lipschitzness provides quantitative guarantees for the convergence rate. We remark that by By Proposition D.1, the same holds when replacing every d ∞ with the symmetrized metric d.
Proof. We start by noticing that both transferability results directly follows from convergence, thanks to the triangle inequality
d ∞ ([f n (x n )], [f m (x m )]) ≤ d ∞ ([f n (x n )], f ∞ (x)) + d ∞ ([f m (x m )], f ∞ (x)). To show convergence in under continuous transferability, observe that if f ∞ is continuous, then [f n (x n )] = f ∞ ([x n ]) → f ∞ (x) immediately follows.
Next, we establish the guarantee under local Lipschitz transferability at x. Suppose f ∞ is locally Lipschitz at x. Then there exists r > 0 such that for all y ∈ B(x, r), we have d ∞ (f ∞ (x), f ∞ (y)) ≤ Ld ∞ (x, y). Let N be large enough so that x n ∈ B(x, r) for all n ≥ N . Then for all n ≥ N , we have
d ∞ (f ∞ (x), [f n (x n )]) = d ∞ (f ∞ (x), f ∞ ([x n ])) ≤ L • d ∞ (x, [x n ]) ≲ R(n)
, as claimed; finishing the proof of the proposition.
Convergence and transferability: random sampling. Under random sampling of inputs, we need to specify the mode of convergence. In the case where x n → x almost surely at rate R(n), the results are identical to the deterministic case: both convergence and transferability hold almost surely. We now consider a different mode of convergence-convergence in expectation. As we will see in Appendix D.3, many common sampling procedures satisfy this condition. Proposition D.5 (Convergence and transferability: Random sampling). Let (x n ∈ V n ) be a sequence of inputs randomly sampled from a limiting object
x ∈ V ∞ , such that [x n ] → x in expectation at rate R(n) with respect to d V ∞ , i.e. E[d V ∞ ([x n ], x)] ≲ R(n) and R(n) → 0. Suppose (f n : V n → U n ) is locally Lipschitz transferable at x, i.e., f ∞ is L-Lipschitz on B(x, r
). Further, assume that there exists M > 0 such that
E [d ∞ (f ∞ ([x n ]), f ∞ (x))1([x n ] / ∈ B(x, r))] ≤ M E [d ∞ ([x n ], x)] .
(2) (Convergence) The function values converge in expectation
E[d ∞ ([f n (x n )], f ∞ (x))] ≲ R(n).
(Transferability) The distance converges in expectation
E[d ∞ ([f n (x n )], [f m (x m )])] ≲ R(n) + R(m).
The assumptions in this proposition are rather mild. Indeed, (2) amounts to a localized version of uniform integrability. Simple arguments show that these assumptions are satisfied under any of the following scenarios: (i) the sequence (f n ) is globally Lipschitz transferable, (ii) the map f ∞ is bounded or (iii) the sequence (x n ) is supported on B(x, r). Furthermore, the same conclusion remains valid when replacing d ∞ with the symmetrized metric d.
Proof. Suppose for all y ∈ B(x, r), we have d ∞ (f ∞ (x), f ∞ (y)) ≤ Ld ∞ (x, y). Then,
E[d ∞ (f ∞ (x), [f n (x n )])] ≤ E d ∞ (f ∞ (x), f ∞ ([x n ])) • 1{[x n ] ∈ B(x, r)} + M E [d ∞ (x, [x n ])] ≤ L • E d ∞ (x, [x n ]) • 1{[x n ] ∈ B(x, r)} + M E [d ∞ (x, [x n ])]
≲ R(n). Transferability then follows by the triangle inequality.
this section cite: []

Section: D.3 Convergence rates under sampling
Propositions D.4 and D.5 show that for Lipschitz transferable models (f n ), the convergence of f n (x n ) to f ∞ (x) is at least as fast as the convergence of x n to x, characterized by the rate R(n). This rate depends on the specific application and sampling scheme. Below, we review common sampling schemes and their associated convergence rates from the literature.
this section cite: []

Section: D.3.1 Random sampling
Empirical distributions and signals. Suppose p ∈ [1, ∞] and µ ∈ P q (R d ) for some q > 2p. Suppose X ∈ R n×d has rows sampled i.i.d. from µ, and let µ X = 1 n n i=1 δ Xi: be the corresponding (empirical) distribution on R d . Then, we have [26]
E[W p (µ, µ X )] ≲    n -1/2p if p > d/2, n -1/2p log 1/p (1 + n) if p = d/2, n -1/d if p < d/2.(3)
Similarly, suppose f ∈ L ∞ ([0, 1]) is a bounded signal sampled by x i = f (t i ) where t 1 , . . . , t n are i.i.d. uniform [0, 1], and if f n be the step function corresponding to x. Noting that µ f has moments of all orders and that µ fn is the empirical measure obtained by sampling n iid points from µ f , we conclude that E d p (f, f n ) = EW p (µ f , µ fn ) converges at the rates (3) with d = 1, where d p is the symmetrized metric with respect to the L p norm on functions. Point clouds. Again let p ∈ [1, ∞] and µ ∈ P q (R k ) with q > 2p, and suppose X ∈ R n×k has rows sampled i.i.d. from µ. Let G ∈ O(k) be a random (or deterministic) rotation, sampled independently of X, and consider the rotated point cloud XG. Then the expected symmetrized metric between µ XG and µ can be bounded by (3) since E inf g∈O(k)
W p (µ, µ XGg -1 ) = E E inf g∈O(k) W p (µ, µ XGg -1 ) G ≤ E[W p (µ, µ X )].(4)
Graphons. Let W : [0, 1] 2 → [0, 1] and A n ∈ R n×n sym be sampled as (A n ) i,j ∼ Ber(W (x i , x j )) where x 1 , . . . , x n are i.i.d. Unif([0, 1]). Let W An be the step graphon associated to A n . Then, [54, §10.4] implies E[δ □ (W, W An )] ≲ 1 log(n) . 8 where δ □ is the cut distance of graphons; see [54, §8.2.2] for a formal definition. Moreover, we have W An → W in cut metric almost surely [54, Cor. 11.15]. Similarly, if T W : L 2 ([0, 1]) → L 2 ([0, 1]) is the integral operator associated with W , then by [37, Equation 4.4 and Lemma E.6],
E[d 2 (T W , T Wn )] = E inf σ∈G∞ ∥T W -T σ•Wn ∥ op,2 ≤ 2 3/2 E[δ □ (W, W Xn )] 1/2 ≲ (log n) -1/4 .
See Appendix G.1 for the definitions of the symmetrized metric and norm on graphons as integral operators.
this section cite: ['b25']

Section: D.3.2 Deterministic sampling
Uniform grid. Suppose f : [0, 1] k → R is L-Lipschitz with respect to ∥ • ∥ p , and consider its values on a uniform grid X i1,...,i k = f ((i 1 -1)/n, . . . ,
(i k -1)/n) ∈ (R n ) ⊗k . If we extend X to a step function as usual by f X (x 1 , . . . , x k ) = X ⌈x1n⌉,...,⌈x k n⌉ , then ∥f -f X ∥ q ≤ ∥f -f X ∥ ∞ ≤ L sup x1,...,x k ∈[0,1] ∥(x 1 , . . . , x k ) -((⌈x 1 n⌉ -1)/n, . . . (⌈x k n⌉ -1)/n)∥ p = L∥(1/n, . . . , 1/n)∥ p = Lk 1/p n , for all q ∈ [1, ∞].
Also note that if we evaluate an L-Lipschitz graphon W : [0, 1] 2 → [0, 1] on such a uniform grid, we have
∥T W -T Wn ∥ op ≤ ∥W -W n ∥ 2 ≤ L √ 2 n .
Local averaging. For any f ∈ L p ([0, 1] k ), we can locally average it over hypercubes of side length 1/n to produce values
X i1,...,i k = n k i1/n (i1-1)/n • • • i k /n (i k -1)/n f (x 1 , . . . , x k ) dx 1 • • • dx k ,
and again extend these values to a step function f X . In this case,
∥f -f X ∥ p ≤ 2dist(f, V n ),
so, we get the optimal rate of convergence.
this section cite: []

Section: E Generalization bounds: details and missing proofs from Section 4
We apply the framework connecting robustness and generalization established by [88], which is built on the idea that algorithmic robustness-that is, a model's stability to input perturbations-is fundamentally linked to its ability to generalize. We refer readers to [88] for the necessary background. This framework has also been recently employed to derive generalization bounds for GNNs in [79], though their analysis is restricted to graphs with bounded size. Similar techniques are used in [47,68].
Any-dimensional generalization bound from algorithmic robustness. We consider an anydimensional supervised learning task where consistent sequences model the input and output space
V = {(V n ), (φ N,n ), (G n )} and U = {(U n ), (ψ N,n ), (G n )},
with associated symmetrized metrics d V and d U . The dataset s consists of N input-output pairs
(x i , y i ) ∈ X × Y ⊆ V ∞ × U ∞ ,
A : (V ∞ × U ∞ ) N → H.
We write A s for the hypothesis learned from the dataset s.
Assume training is performed using a neural network model that is L(r)-locally Lipschitz transferable.
(Recall from Proposition D.1 that this implies A s is L(r)-locally Lipschitz on B(0, r) for all r > 0 with respect to the symmetrized metrics d V and d U .) Since X × Y is compact, and hence bounded, there exists a constant c s > 0 such that A s : V ∞ → U ∞ is c s -Lipschitz on X × Y with respect to the symmetrized metrics. Further, let the loss function ℓ : U ∞ × U ∞ → R be bounded by M , and c ℓ -Lipschitz with respect to the product metric d (x, y), (x ′ , y ′ ) := d U (x, x ′ ) + d U (y, y ′ ). By applying the framework of [88] to the limit space, we immediately obtain a generalization bound for learning tasks where the data consists of inputs of varying dimensions. We note that this is not the result stated in Proposition 4.2 of the main paper; the version claimed there will be established later in Proposition E.3.
this section cite: ['b87', 'b87', 'b78', 'b46', 'b67', 'b87']

Section: Proposition E.1 (Any-dimensional generalization bound).
Assume that the training data consists of
N i.i.d. samples s = (x i , y i ) ∼ μ from a measure μ supported on X × Y ⊆ V ∞ × U ∞ ,
where X and Y have finite ε-covering numbers C X (ε), C Y (ε) with respect to the symmetrized metrics for all ε > 0. Then, for any δ > 0, with probability at least 1δ, the generalization error satisfies
1 N N i=1 ℓ(A s (x i ), y i ) -E (x,y)∼μ ℓ(A s (x), y) ≤ inf γ>0 c ℓ (c s ∨ 1)γ + M 2C X (γ/4)C Y (γ/4) log 2 + 2 log(1/δ) N (5
) ≤ c ℓ (c s ∨ 1)ξ -1 (N ) + M (2 log 2)ξ -1 (N ) 2 + 2 log(1/δ) N ,(6)
where ξ(r) := C X (r/4)C Y (r/4) r 2
and we set γ = ξ -1 (N ) in the second line to obtain the third.
Remark E.2. We make the following observations.
1. The bound (6) converges to 0 as N → ∞. Indeed, ξ is strictly decreasing, and hence its inverse is well-defined and also strictly decreasing. Since ξ(x) → ∞ as x → 0 + , we get ξ -1 (x) → 0 as x → ∞.
2. The generalization bound reveals that the ability to generalize improves with greater model transferability/stability (i.e., smaller Lipschitz constants), and deteriorates with increasing geometric complexity of the data space (i.e., larger covering numbers).
3. We emphasize that μ is a distribution on V ∞ × U ∞ , ensuring that every sample drawn from μ admits a finite-dimensional representative, i.e., (x i , y i ) ∈ V n × U n for some n. This reflects the realistic setting in which data consists of finite-dimensional inputs. This stands in contrast to prior work on GNN generalization bounds [47,68], which considers data distributions on V ∞ -the space of graphon signals in [48], and the space of iterated degree measures in [68]. Such an assumption is somewhat unrealistic, as many elements in these spaces cannot be realized as finite-dimensional data. 4. Note that μ induces a distribution (μ(V n × U n )) n∈N over sample dimensions in N, which inherently places less weight on larger sizes. Consequently, the generalization bound does not offer guarantees on the asymptotic performance of the model as the input dimension n → ∞.
Next, we will derive the second generalization bound that addresses this problem.
Proof. For all (x 1 , y 1 ), (x 2 , y 2 ) ∈ X × Y ,
|ℓ (A s (x 1 ), y 1 ) -ℓ (A s (x 2 ), y 2 )| ≤ c ℓ (c s d V (x 1 , x 2 ) + d U (y 1 , y 2 )) ≤ c ℓ (c s ∨ 1)(d V (x 1 , x 2 ) + d U (y 1 , y 2 )).
Applying [88, Theorem 14] yields that the algorithm A is (C X (γ/4)C Y (γ/4), γc ℓ (c s ∨1))-robust [88, Definition 2] for all γ > 0. Further, applying [88,Theorem 3] gives the generalization bound (5). Finally, (6) follows by taking the γ as defined.
Size-generalization bound: train on finite sizes and test on the limit space. The previous generalization bound follows the classical statistical learning setup, where both training and test data are assumed to be sampled i.i.d. from the same distribution. However, in any-dimensional learning, we are typically concerned with a different scenario: training on data of smaller sizes and testing on data of larger sizes. This motivates the need for a new form of generalization bound that accounts for such settings.
We propose the following set-up (described in the main paper). Let µ be a probability distribution supported on X × Y ⊆ V ∞ × U ∞ , which are subsets whose sequence of orbit closures is compact in the symmetrized metrics. Consider a random sampling procedure
S n : V ∞ × U ∞ → V n × U n
, such that for all n and for all (x, y) ∈ supp(µ), we have supp(S n (x, y)) ⊆ X × Y . This sampling induces a distribution µ n on V n × U n via the sampling procedure; that is, µ n := Law(x n , y n ), where (x n , y n ) ∼ S n (x, y) and (x, y) ∼ µ.
this section cite: ['b46', 'b67', 'b47', 'b67', 'b87']

Section: Proposition E.3 (Size-generalization bound).
Suppose the training data consists of N i.i.d. samples s = (x i , y i ) ∼ µ n . Then, for any δ > 0, with probability at least 1δ, the generalization error satisfies
1 N N i=1 ℓ(A s (x i ), y i ) -E (x,y)∼µ ℓ(A s (x), y) ≤ c ℓ (c s ∨ 1) ξ -1 (N ) + W 1 (µ, µ n ) + M (2 log 2) ξ -1 (N ) 2 + 2 log(1/δ) N ,(7)
where W 1 denotes the Wasserstein-1 distance, and ξ(r) := C X (r/4) C Y (r/4) r 2
, with C X (ε) and C Y (ε) denoting the ε-covering numbers of X and Y , respectively, with respect to the symmetrized metrics. Moreover, assuming that the sampling procedure converges in expectation at a rate R(n), i.e.,
E (xn,yn)∼Sn(x,y) d V (x, [x n ]) + d U (y, [y n ]) ≲ R(n), we have that 1 N N i=1 ℓ(A s (x i ), y i ) -E (x,y)∼µ ℓ(A s (x), y) ≲ c ℓ (c s ∨ 1) ξ -1 (N ) + R(n) + M (2 log 2) ξ -1 (N ) 2 + 2 log(1/δ) N . (8
)
Remark E.4. We make the following observations.
1. The bound (8) converges to 0 if both the training input dimension n and the amount of data N goes to ∞. Indeed, we have justified in Remark E.2 that (6) converges to 0 as N → ∞. The only additional term in (8) is R(n) which converges to 0 as n → ∞.
2. This new generalization bound aligns with the setup where training is performed on inputs of fixed size n (also naturally extends to inputs of varying finite sizes), and testing evaluates the asymptotic performance as n → ∞. It can therefore be interpreted as a size generalization bound, accounts for distributional shifts induced by size variation. As a consequence, an additional term appears in the bound, reflecting the convergence rate of the sampling procedure.
Proof. By triangle inequality, .
1 N N i=1 ℓ(A s (x i ), y i ) -E (x,y)∼µ ℓ(A s (x), y) ≤ 1 N N i=1 ℓ(A s (x i ), y i ) -E (x,
We bound the two terms separately. By the Kantorovich-Rubinstein duality, we have almost surely that
T 2 ≤ c ℓ (c s ∨ 1) • W 1 (µ, µ n ).
To bound T 1 , recall that supp(S n (x, y)) ⊆ X × Y for all (x, y) ∈ supp(µ). It follows that supp(µ n ) ⊆ X × Y , which is therefore totally bounded. Its covering number is upper bounded by that of X × Y . Applying Proposition E.1 with μ = µ n yields (7).
Finally, to bound W 1 (µ, µ n ), we note that the sampling procedure induces a natural coupling between µ and µ n . Using this coupling, (8); completing the proof.
W 1 (µ, µ n ) ≤ E (x,y)∼µ, (xn,yn)∼Sn(x,y) d V (x, [x n ]) + d U (y, [y n ]) ≲ R(n), which yields
Practicality of the assumptions. Finally, we reflect on the key assumptions required for the bound (8) to hold and assess their practicality in more specific settings. First, we assumed that the loss function is bounded (and, consequently, Lipschitz continuous). We note that these assumptions are relatively standard [1,76,3]. When the predictions and target outputs are bounded, several widelyused loss functions, such as cross-entropy, L1-loss, L2-loss, and Huber loss, are all bounded and Lipschitz continuous. Moreover, many clipped loss functions also satisfy this assumption. Second, regarding the assumption of a sampling procedure that converges in expectation at a rate R(n), we refer readers to Appendix D.3 for examples involving sets, graphs, and point clouds. Most importantly, the bound critically depends on the compactness of X and Y in the symmetrized metrics, and on the sampling procedure generating finite-size samples that remain within this compact space, i.e., supp(S n (x, y)) ⊂ X × Y for all (x, y) ∈ supp(µ). Below, we provide two concrete examples involving sets and graphs where these assumptions hold, and where bounds for the covering numbers are explicitly known. In these specific settings, our generalization bound applies directly.
this section cite: ['b6', 'b7', 'b7', 'b0', 'b75', 'b2']

Section: Example E.5 (Probability measures supported on compact set). Consider V ⊕d
dup , the duplication consistent sequence for sets endowed with the normalized ℓ p metric. See Appendix F.1 for the precise definitions. The limit space is V ∞ = L p ([0, 1], R d ), and the space of orbit closures of V is P p (R d ), the space of probability measures on R d with finite p-th moment, endowed with the Wasserstein-p distance. Fix a compact set Ω ⊆ R d , and let
X := f ∈ L p ([0, 1], R d ) : µ f is supported on Ω .
Note that the sequence of orbit closures in X, namely {µ f : f ∈ X}, is compact with respect to W p . A bound on the covering number C X is given by [63, Theorem 2.2.11]. Consider the sampling procedure S n : L p ([0, 1], R d ) → R n×d defined by drawing z i i.i.d.
∼ Unif([0, 1]), and setting S n (f ) i: = f (z i ) for i = 1, . . . , n. Then, for all f ∈ X, each entry of S n (f ) lie in Ω. Hence, we have S n (f ) ∈ X. Our generalization bound therefore applies to this setting. Example E.6 (Graphon signals with cut distance). Consider V G dup , the duplication-consistent sequence for graph signals endowed with the cut metrics. See Appendix G.1 for the precise definitions. Define the space
X = W : [0, 1] 2 → [0, 1] measurable : W (x, y) = W (y, x) ×{f ∈ L ∞ ([0, 1], R) : ∥f ∥ ∞ ≤ r} .
By [47,Theorem 3], the sequence of orbit closures in X is compact with respect to the cut metric on graphon signals. Moreover, a bound on the covering number is also provided in the same result. Consider the sampling procedure S n : X → R n×n sym × R n defined as follows: draw z i i.i.d.
∼ Unif([0, 1]), and set S n (W, f ) = (A, X), where A ij ∼ Ber(W (z i , z j )) and X i = f (z i ) for i = 1, . . . , n. Then, for all (W, f ) ∈ X, the sampled pair S n (W, f ) belongs to X. This is the standard sampling procedure for graphon signals, and once again our generalization bound applies.
this section cite: ['b46']

Section: E.1 Transferable neural networks
To prove the transferability of a neural network, we first observe that compatibility and transferability are preserved under composition. Therefore, it suffices to verify these properties for each individual layer. Moreover, by Propositions C.6 and D.2, it is enough to prove the compatibility and Lipschitz continuity of each f n on the finite space, rather than analyzing the limiting function f ∞ directly. This idea is formalized in the following proposition, which serves as a key tool in our transferability analysis of neural networks in the later sections. Importantly, this provides a general and easy-to-apply proof strategy. In contrast, previous works often begin by characterizing a natural limiting function f ∞ (e.g., a graphon neural network), and then directly prove its Lipschitz continuity in the limit space-a process that typically requires case-specific proof techniques.
this section cite: []

Section: Proposition E.7 (Transferable networks: detailed version of Proposition 5.1). Let (V
(i) n ) n , (U (i) n ) n be consistent sequences for i = 1, . . . , D. For each i, let (W (i) n : V (i) n → U (i) n ) be linear maps and (ρ (i) n : U (i) n → V (i+1) n
) be nonlinearities. Assume the following three properties hold.
1. The maps W (i) n , ρ (i) n are compatible. 2. The linear maps are uniformly bounded sup n,i ∥W
(i) n ∥ op = L W < ∞.
this section cite: []

Section: The map ρ
(i) n is L i (r)-Lipschitz on u ∈ U (i) n : ∥u∥ < r for all n.
this section cite: []

Section: Then the composition
W (D) n • ρ (D-1) n • . . . • ρ (1) n • W (1) n is locally Lipschitz transferable, ex- tending to a function on V ∞ → U ∞ that is L NN (r)-Lipschitz on v ∈ V (1)
∞ : ∥v∥ < r , where we inductively define
ℓ 1 = L 1 (L W r), ℓ i+1 = L i+1 (L i+1 W ℓ i ), L NN (r) = L D W D-1 i=1 ℓ i .(9)
In particular, if ρ
n is L ρ -Lipschitz for all i, n then the composition is Lipschitz transferable, extending to a function on
V ∞ → U ∞ that is L NN -Lipschitz where L NN = L D W L D-1 ρ .
Remark E.8. By Proposition D.1, the composition is also L NN (r)-Lipschitz with respect to the symmetrized metrics on the same r-ball.
Proof. Note that if f 1 : V
∞ → V(1)
∞ and f 2 : V
∞ → V(2)
∞ are L 1 (r)-and L 2 (r)-locally Lipschitz, respectively, then f 2 • f 2 is L 2 (L 1 (r))L 1 (r)-locally Lipschitz. Our claim follows by an inductive application of this fact and Proposition D.2.
this section cite: []

Section: F Example 1 (sets): details and missing proofs from Section 5.1
In this section, we study the transferability of architectures taking sets as inputs. Section F.1 introduces the consistent sequences we consider and Section F.2 presents results for three architectures: DeepSets[91], normalized DeepSets [9], and PointNet [66].
this section cite: ['b8', 'b65']

Section: F.1 Consistent sequences on sets
We present two examples of consistent sequences on sets. See Figure 1 in the main text for a graphical illustration.
Zero-padding consistent sequence V zero with ℓ p norm
The zero-padding consistent sequence V zero = {(V n ), (φ N,n ), (G n )} is defined as follows: The index set N = (N, ≤) is the poset of natural numbers with the standard ordering. Let V n = R n for every n ∈ N, and the zero-padding embedding is given by, for n ≤ N ,
φ N,n : R n ↩→ R N (x 1 , . . . , x n ) → (x 1 , . . . , x n , 0, . . . , 0 (N -n) 0's ).
The group of permutations S n on n letters acts on R n by permuting coordinates: (g • x) i := x g -1 (i) for g ∈ S n . The embedding of groups is given by, for n ≤ N ,
θ N,n : S n ↩→ S N g → g 0 0 I N -n .
That is, view S n as the subgroup of S N which acts trivially on n + 1, . . . , N . In this case, V ∞ can be identified with ℓ 0 , i.e., the space of infinite scalar sequences with finitely many nonzero entries. The limit group G ∞ is the group of permutations of N fixing all but finitely many indices.
We associate every infinite sequence (x i ) ∞ i=1 ∈ V ∞ with the tuple of sequences (x + i ) ∞ i=1 , (x - i ) ∞ i=1 . The sequence (x + i ) ∞ i=1 comprises the positive entries of (x i ), ordered in descending order and extended with trailing zeros. The sequence (x - i ) ∞ i=1 comprises the negative entries of (x i ), ordered in ascending order and similarly extended with trailing zeros. Notice that different sequences in V ∞ are associated with the same tuple of sequences if and only if they belong to the same orbit under the action of G ∞ . Hence, the orbit space of V ∞ under the action of G ∞ can be identified with tuples of ordered infinite sequences. Specifically, one sequence consists of non-negative entries arranged in descending order, and the other sequence consists of non-positive entries arranged in ascending order, with both sequences having finitely many non-zero entries.
The ℓ p norm on V zero . We can endow each V n with the ℓ p -norms
∥x∥ p = ( n i=1 |x i | p ) 1/p if p ∈ [1, ∞), max n i=1 |x i | if p = ∞.
It is easy to check by Proposition C.8 that this induces a norm on V ∞ = R ∞ , which coincides with the ℓ p -norms on infinite sequences, which is also denoted as ∥ • ∥ p . The limit space is then
V ∞ = ℓ p = {(x i ) ∞ i=1 : ∞ i=1 |x i | p < ∞} if p ∈ [1, ∞), c 0 = {(x i ) ∞ i=1 : lim i→∞ |x i | = 0} if p = ∞.
Similarly, the space of orbit closures of V ∞ under the action of G ∞ can be identified with tuples of ordered infinite sequences. Specifically, one sequence consists of non-negative entries arranged in descending order, and the other sequence consists of non-positive entries arranged in ascending order, with both sequences in ℓ p (for p ∈ [1, ∞)) or c 0 (for p = ∞). This space is endowed with the symmetrized metric d p (x, y) = min σ∈G∞ ∥xσ • y∥ p .
The ℓ p norm on the direct sum V ⊕d zero . The direct sum V ⊕d zero = {(R n×d ), (φ ⊕d N,n ), (S n )} defined in Definition C.4 extends the above to the case of a set of vectors in R d . To endow it with a norm, we first fix an arbitrary norm ∥ • ∥ R d on R d . Then the ℓ p -norm on R n×d is defined analogously with respect to ∥ • ∥ R d , i.e., for X ∈ R n×d ,
∥X∥ p = n i=1 ∥X i: ∥ p R d 1/p if p ∈ [1, ∞), max n i=1 ∥X i: ∥ R d if p = ∞.
Analogously, in this case, V ∞ can be seen as the space of infinite sequences in R d with finitely many nonzero entries, and the limit space V ∞ is the corresponding ℓ p space (if p ∈ [1, ∞)) or c 0 space (if p = ∞). The space of orbit closures can be seen as tuples of infinite sequences in R d , ordered in lexicographic order.
this section cite: []

Section: Duplication consistent sequence V dup with normalized ℓ p -norms
The duplication consistent sequence
V dup = {(V n ), (φ N,n ), (G n )} is defined as follows. The index set (N, • | •)
is the set of natural numbers with divisibility partial order, where n ⪯ N if and only if n | N . Let V n = R n for all n ∈ N, and the duplication embeddings is given by
φ N,n : R n ↩→ R N (x 1 , . . . , x n ) → x ⊗ 1 N/n = (x 1 , . . . , x 1 N/n times , . . . , x n , . . . , x n ),
for n ⪯ N . The group embeddings are given by θ N,n : S n ↩→ S N g → g ⊗ I N/n , for n ⪯ N . That is, g ∈ S n acts on [N ] by sending (i -1)N/n + j to (g(i) -1)N/n + j for i = 1, . . . , n and j = 1, . . . , N/n.
In this case, V ∞ can be identified with step functions on [0, 1] whose discontinuity points are in Q: each x ∈ R n corresponds to f x : [0, 1] → R where f x (t) = x ⌈tn⌉ for t > 0 and f (0) = x 1 .
In other words, f x is the step function which takes value x i on I i,n = i-1 n , i n for i = 1, . . . , n. Indeed, all equivalent objects x and φ N,n x correspond to the same function in this way. Therefore, V ∞ can be seen as the union of step functions of this form for n ∈ N. Under this identification, permutations S n permute the n intervals I i,n and act on functions by g • f = f • g -1 . The limit group G ∞ is the union of such interval permutations. The orbit space of the G ∞ -action on V ∞ can be identified with monotonically increasing step functions on [0, 1] whose discontinuity points are in Q. Alternatively, the orbit space can be identified with the space of empirical measures on R: each x ∈ R n corresponds to the empirical measure µ x = 1 n n i=1 δ xi . Indeed, any equivalent objects x and φ N,n x are identified with the same measure; furthermore, this resulting measure is constant on orbits under the G ∞ -action.
Under the identification of V ∞ with step functions, a step function f ∈ V ∞ is identified with the probability measure µ f , defined as the distribution of f (T ) for T uniformly sampled from [0, 1]. Indeed, all elements in the orbit of f under the G ∞ -action correspond to this same measure µ f . Note that the generalized inverse CDF of f (T ) is precisely the 'sorted' version of f (called its increasing rearrangement), relating the above two perspectives on the orbit space. The latter view of the orbit space as a sequence of measures generalizes readily to other consistent sequences obtained from V dup , such as V ⊕d dup which we consider below, so we shall take this view from now. Normalized ℓ p norm on V dup . We can endow each space V n with the normalized ℓ p -norms
∥x∥ p = 1 n n i=1 |x i | p 1/p if p ∈ [1, ∞), max n i=1 |x i | if p = ∞.
Using Proposition C.8, it is straightforward to verify that this defines a norm on V ∞ . Under the identification of V ∞ with step functions, the induced norm on V ∞ coincides with the conventional L p norm on measurable functions, given by
∥f ∥ p =    1 0 |f (t)| p dt 1/p if p ∈ [1, ∞), sup t∈[0,1] |f (t)| if p = ∞.
That is, for any x ∈ R n , we have ∥x∥ p = ∥f x ∥ p , where f x is the corresponding step function. The limit space is then
V ∞ =      L p ([0, 1]) = f : [0, 1] → R measurable : 1 0 |f (t)| p dt < ∞ if p ∈ [1, ∞), f : [0, 1] → R
: f is bounded and continuous on [0, 1] \ Q, with left and right limits at every x ∈ [0, 1] ∩ Q if p = ∞, where the result for p = ∞ follows from [21, Chap. VII.6]. These are a subspace of so-called regulated functions, which have left and right limits at each x ∈ [0, 1].
When p ∈ [1, ∞), the space of orbit closures, equipped with the symmetric metric, can be identified with P p (R), the space of probability measures on R with finite p-th moment, endowed with the Wasserstein p-distance. In the case p = ∞, the space of orbit closures corresponds to a subset of P ∞ (R), the space of probability measures on R with bounded support, equipped with the Wasserstein ∞-distance. This is formalized and proved by the following propositions:
Proposition F.1. For any p ∈ [1, ∞] and all f, g ∈ V ∞ , the symmetrized metric d p (f, g) := inf σ∈G∞ ∥σ • f -g∥ p equals the Wasserstein p-distance between the associated measures:
d p (f, g) = W p (µ f , µ g ).
Proof. We first prove they match on V ∞ . Consider vectors x ∈ R n , y ∈ R m under the action of S n , S m respectively, and let N = lcm(n, m). Then, by standard results on the Wasserstein distance of empirical measures [65, §2.2],
d p (x, y) =    min σ∈S N 1 N N i=1 φ N,n (x) σ(i) -φ N,m (y) i p 1/p = W p (µ x , µ y ) if p ∈ [1, ∞), min σ∈S N max N i=1 φ N,n (x) σ(i) -φ N,m (y) i = W ∞ (µ x , µ y ) if p = ∞.
where µ x was defined above, and W p is the Wasserstein p-distance. Hence under the identification with step functions, for any f, g ∈ V ∞ we have d p (f, g) = W p (µ f , µ g ).
Now consider the limit points. Let (f n ), (g n ) be two Cauchy sequences in V ∞ with f n → f, g n → g in V ∞ with respect to the L p norm. Then
d p (f n , g n ) -d p (f, g) ≤ d p (f, f n ) + d p (g n , g) ≤ ∥f -f n ∥ p + ∥g n -g∥ p → 0.
Similarly, since for any f , g ∈ V ∞ ,
W p (µ f , µ g ) ≤ E T ∼Unif[0,1] | f (T ) -g(T )| p 1/p = ∥ f -g∥ p , we also get |W p (µ fn , µ gn ) -W p (µ f , µ g )| → 0. But for all n, d p (f n , g n ) = W p (µ fn , µ gn )
, so by the uniqueness of the limit,
d p (f, g) = W p (µ f , µ g ).
Proposition F.2. For p ∈ [1, ∞), the space of orbit closures {µ f : f ∈ V ∞ } coincides with P p (R). For p = ∞, this set is a subset of P ∞ (R).
Proof. For p ∈ [1, ∞), by definition, the space of orbit closures is the set of probability measures {µ f : f ∈ L p ([0, 1])}. We claim that this set is equal to P p (R). Observe that ((E X∼µ f |X| p ) 1/p = ∥f ∥ p . On the one hand, this implies that if f ∈ L p ([0, 1]), then µ f ∈ P p (R). Conversely, given any µ ∈ P p (R), let f be the generalized inverse of the CDF of µ, then µ f = µ and f ∈ L p ([0, 1]). Hence µ ∈ P p (R) implies that µ = µ f for f ∈ L p ([0, 1]).
For p = ∞, note that any f ∈ V ∞ is bounded, so the support of µ f is compact, implying that µ f ∈ P ∞ (R).
this section cite: []

Section: Norms on V ⊕d
dup . Similarly, we fix an arbitrary norm ∥ • ∥ R d on R d and define the norms on R n×d with respect to ∥ • ∥ R d :
∥X∥ p := 1 n n i=1 ∥X i: ∥ p R d 1/p if p ∈ [1, ∞), max n i=1 ∥X i: ∥ R d if p = ∞. (10
)
For p ∈ [1, ∞), the space of orbit closures can be identified with P p (R d ) endowed with the Wasserstein p-distance with respect to ∥ • ∥ R d . For p = ∞, the space of orbit closures can be seen as a subset of P ∞ (R d ) with the Wasserstein-∞ distance with respect to ∥ • ∥ R d . This is because R d is a standard Borel space [77, Thm. 3.3.13], so the same arguments as in Propositions F.1-F.2 apply.
this section cite: []

Section: F.2 Invariant networks on sets
We consider three prominent permutation-invariant neural network architectures for set-structured data: DeepSets [91], normalized DeepSets [9], and PointNet [66]. These models are defined as follows:
DeepSet n (X) = σ n i=1 ρ(X i: ) , DeepSet n (X) = σ 1 n n i=1 ρ(X i: ) ,and
PointNet n (X) = σ n max i=1 ρ(X i: ) ,
where ρ : R d → R h and σ : R h → R are multilayer perceptrons (MLPs). In the case of PointNet, the maximum is taken entrywise over vectors in R h .
They follow the same paradigm f n (X) = σ (Agg n i=1 ρ(X i: )) where the three models use different permutation-invariant aggregations Agg. We refer the reader to [9] for a comprehensive study of the expressive power of these models in the any-dimensional setting. In particular, they show that normalized DeepSets (respectively, PointNet) can uniformly approximate all set functions that are uniformly continuous with respect to the Wasserstein-1 distance (respectively, the Hausdorff distance). In contrast, our work focuses on transferability and size generalization, rather than expressive power.
this section cite: ['b8', 'b65', 'b8']

Section: F.2.1 Transferability analysis: proof of Corollary 5.2
We prove the Corollary by instantiating Proposition E.7. The invariant network is given by the following composition
R n×d ρ ⊕n -----→ row-wise R n×h Agg n i=1 ----→ R h σ -→ R,
where we use ρ ⊕n to denote the row-wise application of the same ρ : R d → R h . Thus, it suffices to analyze each term in this composition individually.
DeepSet. Notice that the sum aggregation is not compatible with the duplication embedding. Indeed, for any x ∈ R n such that i x i ̸ = 0, and for n | N, n ̸ = N ,
N i=1 (x ⊗ 1 N/n ) i = (N/n) n i=1 x i ̸ = n i=1 x i .
Therefore, DeepSet is not compatible with respect to the duplication consistent sequence in general. We now prove its compatibility and transferability with respect to zero-padding.
Corollary F.3. Fix arbitrary norms ∥•∥ R d on R d and ∥•∥ R h on R h . Let ρ : R d → R h be L ρ -
DeepSet ∞ : ℓ 1 (R d ) → R, DeepSet ∞ ((x i ) ∞ i=1 ) = σ ∞ i=1 ρ(x i ) ,
which is (L ρ L σ )-Lipschitz with respect to the ℓ 1 -norm on the infinite sequences.
Proof. We model each intermediate space with consistent sequences:
V ⊕d zero = (R n×d ) ρ ⊕n -----→ (row-wise) V ⊕h zero = (R n×h ) n i=1 ---→ V R h = (R h ) σ -→ V R = (R).
We first check the compatibility of each map.
• As long as ρ(0) = 0, the ρ-map is compatible because
ρ ⊕N X 0 = ρ ⊕n (X) 0 for all X ∈ R n×d , 0 ∈ R (N -n)×d , n ≤ N ,
and the row-wise application makes sure ρ is S n -equivariant.
• The sum aggregation Agg n i=1 = n i=1 is compatible because adding zeros does not change the sum, and the summation operation is S n -invariant.
• The map σ is between two trivial consistent sequences. Hence it is automatically compatible. Endow V ⊕d zero with the ℓ 1 norm induced by ∥ • ∥ R d , and V ⊕h zero with the ℓ 1 norm with induced by ∥ • ∥ R h . V R h , V R are endowed with ∥ • ∥ R h and | • | respectively. Next, we check the Lipschitz transferability of each map.
• The ρ-map is L ρ -Lispchitz transferable map because for all n, we can prove ρ ⊕n : R n×d → R n×h (applying the same ρ row-wise) is L ρ Lipschitz with respect to the ℓ 1 norms:
∥ρ ⊕n (X) -ρ ⊕n (Y )∥ 1 = n i=1 ∥ρ(X i: ) -ρ(Y i: )∥ R h ≤ n i=1 L ρ ∥X i: -Y i: ∥ R d = L ρ ∥X -Y ∥ 1 .
• The sum aggregation Agg n i=1 = n i=1 is 1-Lipschitz transferable because for all n, n i=1
X i: - n i=1 Y i: R h ≤ n i=1 ∥X i: -Y i: ∥ R h = ∥X -Y ∥ 1 .
We highlight that this does not necessarily hold for other ℓ p norms for p ̸ = 1.
• The map σ is L σ -Lipschitz transferable.
Thus, the result follows from Proposition 5.1; completing the proof.
Normalized DeepSet. The mean aggregation is not compatible with the zero-padding embedding. Consider a vector x = (x 1 , . . . , x n ) ∈ R n such that i x i ̸ = 0, and suppose n < N . When zero-padded to length N , we obtain
x = (x 1 , . . . , x n , 0, . . . , 0) ∈ R N . Then 1 N N i=1 xi = 1 N n i=1 x i ̸ = 1 n n i=1 x i .
Therefore, normalized DeepSet is not compatible with respect to the zero-padding consistent sequence in general.
We now prove its compatibility and transferability with respect to the duplication consistent sequence with normalized ℓ p norm.
DeepSet ∞ : P p (R d ) → R, DeepSet ∞ (µ) = σ ρdµ , which is (L ρ L σ )-Lipschitz with respect to the Wasserstein-p distance on ∥ • ∥ R d .
Proof. We model each intermediate space with consistent sequences:
V ⊕d dup = (R n×d ) ρ ⊕n -----→ (row-wise) V ⊕h dup = (R n×h ) 1 n n i=1 -----→ V R h = (R h ) σ -→ V R = (R).
We first consider the compatibility of each map.
• The ρ-map is compatible because ρ ⊕N (X ⊗ 1 N/n ) = ρ ⊕n (X) ⊗ 1 N/n for all n | N , and the row-wise application makes sure ρ is S n -equivariant.
• The mean aggregation
Agg n i=1 = 1 n n i=1 is compatible because for all n | N, X ∈ R n×d , 1 N N i=1 (X ⊗ 1 N/n ) i: = 1 N n i=1 (N/n)X i: = 1 n n i=1 X i: ,
and the mean operation is S n -invariant.
• The map σ is again automatically compatible.
Endow V ⊕d dup with the normalized ℓ p norm with respect to ∥ • ∥ R d , and V ⊕h dup with the normalized ℓ p norm with respect to ∥ • ∥ R h . The trivial consistent sequences V R h , V R are endowed with ∥ • ∥ R h and | • | respectively. Next, we check the Lipschitz transferability of each map in the composition.
• The ρ-map is L ρ -Lipschitz because for all n, we can prove ρ ⊕n : R n×d → R n×h is L ρ Lipschitz with respect to the normalized ℓ p norm:
∥ρ ⊕n (X) -ρ ⊕n (Y )∥ p = 1 n n i=1 ∥ρ(X i: ) -ρ(Y i: )∥ p R h 1/p ≤ 1 n n i=1 L p ρ ∥X i: -Y i: ∥ p R d 1/p = L ρ ∥X -Y ∥ p .
• The mean aggregation
Agg n i=1 = 1 n n i=1 is 1-Lipschitz transferable because 1 n n i=1 X i: - 1 n n i=1 Y i: R h ≤ 1 n n i=1 ∥X i: -Y i: ∥ R h = ∥X -Y ∥1 ≤ ∥X -Y ∥ p,
where the last inequality follows from Hölder's inequality.
• The map σ is L σ -Lipschitz transferable.
The result follows from an application of Proposition 5.1; completing the proof.
Remark F.5. The same result was also proved in [9, Theorem 3.7] by directly verifying the Lipschitz property of DeepSet ∞ : for all p ≥ 1,
|DeepSet ∞ (µ X ) -DeepSet ∞ (µ Y )| ≤ L σ ρ d(µ X -µ Y ) ≤ L σ L ρ W 1 (µ X , µ Y ) ≤ L σ L ρ W p (µ X , µ Y )
, where the second inequality follows from the Kantorovich-Rubinstein duality. Our methods provide an alternative proof, using a proof technique that applies more generally (Proposition 5.1).
Following this result, we can directly apply Propositions D.4 and D.5, along with the convergence rates described in Appendix D.3, which immediately yields the following transferability result.
this section cite: []

Section: Corollary F.6 (Transferability of normalized DeepSet).
We have the following transferability results for normalized DeepSet:
1. (Uniform grid sampling) Let (X n ) ∈ R n×d be a sequence of matrices sampled from the same signal f via the "uniform grid" sampling scheme, i.e., taking (X n ) i: Then,
= f i-1 n for all i ∈ [n]. Suppose f is Lipschitz. Then, DeepSet n (X n ) -DeepSet m (X m ) ≲ n -1 + m -1 . 2. (Random signal sampling) Let (X n ) ∈ R n×d
E DeepSet n (X n ) -DeepSet m (X m ) ≲    n -1/2 + m -1/2 if d = 1, n -1/2 log(1 + n) + m -1/2 log(1 + m) if d = 2, n -1/d + m -1/d if d > 2.
this section cite: []

Section: (Empirical distributions)
Let (X n ) ∈ R n×d be a sequence of matrices sampled from the same underlying distribution µ ∈ P 3 (R d ), i.e., each X n has rows sampled i.i.d. from µ.
Then,
E DeepSet n (X n ) -DeepSet m (X m ) ≲    n -1/2 + m -1/2 if d = 1, n -1/2 log(1 + n) + m -1/2 log(1 + m) if d = 2, n -1/d + m -1/d if d > 2.
PointNet. The max aggregation is not compatible with zero-padding. Consider a vector x = (x 1 , . . . , x n ) ∈ R n where all entries x i < 0, and suppose n < N . When zero-padded to length N , we obtain x = (x 1 , . . . , x n , 0, . . . , 0) ∈ R N . Then, we have
max 1≤i≤N xi = 0 ̸ = max 1≤i≤n x i .
Hence, unless we restrict the model to avoid all-negative entries, PointNet is not compatible with the zero-padding consistent sequence. We now prove its compatibility and transferability with respect to the duplication sequence with the ℓ ∞ norm.
PointNet ∞ : P ∞ (R d ) → R, PointNet ∞ (µ) = σ sup x∈supp(µ) ρ(x) ,
which is (L ρ L σ )-Lipschitz with respect to the Wasserstein-∞ distance on ∥ • ∥ R d .
Proof. We again consider consistent sequences
V ⊕d dup = (R n×d ) ρ ⊕n -----→ (row-wise) V ⊕h dup = (R n×h ) max n i=1 -----→ V R h = (R h ) σ -→ V R = (R).
For compatibility, it is left to check that Agg n i=1 = max n i=1 is compatible. Indeed, for any X ∈ R n×d , n | N , we have max N i=1 (X ⊗ 1 N/n ) i: = max n i=1 X i: , and the max operation is S n -invariant. Endow V ⊕d dup with the ℓ ∞ norm with respect to ∥ • ∥ R d , and V ⊕h dup with the ℓ ∞ norm with respect to ∥ • ∥ ∞ on R h . The trivial consistent sequences V R h , V R are endowed with ∥ • ∥ ∞ and | • | respectively. Next, we check Lipschitz transferability of each map.
• We have proved in the proof for normalized DeepSet that ρ, σ are L ρ , L σ Lipschitz transferable respectively.
• For any j ∈
[d], | max n i=1 X ij -max n i=1 Y ij | ≤ max n i=1 |X ij -Y ij |. Take max over j ∈ [d], we conclude n max i=1 X i: - n max i=1 Y i: ∞ = d max j=1 n max i=1 X ij - n max i=1 Y ij ≤ n max i=1 ∥X i: -Y i: ∥ ∞ . Hence, Agg n i=1 = max n i=1 is 1-Lipschitz transferable.
The result follows from Proposition 5.1; which completes the proof.
Remark F.8. PointNet ∞ produces identical outputs for probability measures with the same support. Thus, it can be viewed as a function
PointNet ∞ : K(R d ) → R, PointNet ∞ (X) = σ sup x∈X ρ(x) ,
where K(R d ) denotes the set of non-empty compact subsets of R d . The W ∞ distance on P ∞ (R d ) induces the quotient metric d K on K(R d ) via the equivalence relation µ ∼ ν if supp(µ) = supp(ν).
Our results imply that PointNet ∞ is (L ρ L σ )-Lipschitz with respect to d K .
A more commonly used metric on K(R d ) is the Hausdorff distance, defined by
d H (X, Y ) := max sup x∈X inf y∈Y ∥x -y∥, sup y∈Y inf x∈X ∥x -y∥ .(11)
[9, Theorem 3.7] shows that PointNet ∞ is (2L ρ L σ )-Lipschitz with respect to d H . It is easy to see that d H ≤ d K , but we leave exploring further relations between these two metrics to future work.
Finally, we show that the sequence of maps (PointNet n ) is, in general, not transferable with respect to the duplication-based consistent sequence V ⊕d dup when equipped with the normalized ℓ p norm for any p ∈ [1, ∞). Consider the sequence of matrices (X (n) ∈ R n×h ) n where the first row is the all-one vector 1 ⊤ h , and the remaining n -1 rows are zero vectors. Then,
∥X (n) -0∥ p → 0 as n → ∞, which implies that [X (n) ] → [0] in V ∞ . However, n max i=1 X (n) i: = 1 h for all n.
This demonstrates that the max aggregation Agg n i=1 = max n i=1 is not continuously transferable under the normalized ℓ p norm, and so neither is the sequence (PointNet n ).
Related work. Several previous works [92,64,9] studied (variants of) normalized DeepSets in the infinite-dimensional limit as operating in the space of probability measures, and investigated their approximation power and generalization behavior.
Moreover, while our analysis primarily focused on invariant neural networks on sets, it is also natural to design and analyze equivariant neural networks on sets with our theoretical framework. In particular, we may consider such neural networks that are transferable with respect to duplicationconsistent sequences, which parametrize measure-to-measure functions. For example, [19] proposed a general framework for designing measure-to-measure neural network architectures. Additionally, [86,30,74] analyzed transformers (without causal masking or positional encoding) as measure-tomeasure functions in the limit space. We leave the analysis of the transferability of these models within our framework as a future direction. F.2.2 Explanation of transferability plots (Figure 2) Our numerical experiment in Figure 2 illustrates the second column of Table 2 for p = 1: the Lipschitz transferability of normalized DeepSet, and non-transferability of DeepSet and PointNet, with respect to (V dup , ∥ • ∥ 1 ). First, applying Proposition D.5 to normalized DeepSet, and recalling the convergence rate of empirical distributions given in (3) for d = 1, p = 1, we get the following corollary.
Corollary F.9 (Convergence and transferability of normalized DeepSet).
Let (x n ∈ R n ) n∈N be a sequence of inputs with entries (x n ) i i.i.d.
∼ µ for i = 1, . . . , n, where µ is a probability measure on R with finite expectation. Define the empirical measure µ
x := 1 n n i=1 δ xi for x ∈ R n . Then E[W 1 (µ, µ xn )] ≲ n -1/2 ,
this section cite: ['b63', 'b8', 'b18', 'b85', 'b29', 'b73']

Section: and hence
E DeepSet n (x n ) -DeepSet ∞ (µ) ≲ n -1/2 , E DeepSet n (x n ) -DeepSet m (x m ) ≲ n -1/2 + m -1/2 .
Indeed, Figure 2(b) shows convergence of the model outputs, and Figure 2(d) confirms that the convergence rate is O(n -1/2 ), as predicted. Figure 2(a) illustrates the divergence of outputs from DeepSet. This occurs because the sum n i=1 ρ(x i ) = O(n). If the function σ in DeepSet is unbounded, this leads to unbounded (blow-up) outputs as n increases. Figure 2(c) shows divergent outputs from PointNet. When the input distribution µ has compact support, the output of PointNet will converge, although without guarantees on the rate. However, in our experiment where µ = N (0, 1) has non-compact support. If ρ in the PointNet is unbounded, the maximum value max n i=1 ρ(x i ) diverges almost surely as n → ∞. This again results in blow-up outputs.
this section cite: []

Section: G Example 2 (graphs): details and missing proofs from Section 5.2
In this section, we study transferability for GNN architectures. Section G.1 introduces the consistent sequences we consider. Section G.2 studies existing architectures and Section G.3 introduces a new class of GNNs with better transferability properties.
this section cite: []

Section: G.1 Duplication consistent sequence for graphs
We present an examples of consistent sequences on graphs, illustrated graphically in Figure 3. Start with the duplication consistent sequence for sets V dup defined in F.1, we define
V G dup := Sym 2 (V dup ) ⊕ V ⊕d
dup , following the definition of direct sum and tensor product in Definition C.4, C.5. This gives the duplication consistent sequence for graphs. Specifically,
V G dup = {(V n ), (φ N,n ), (G n )} where V n = R n×n
sym × R n×d for each n and the embedding for n | N is given by,
φ N,n : R n×n sym × R n×d ↩→ R N ×N sym × R N ×d (A, X) → (A ⊗ (1 N/n 1 ⊤ N/n ), X ⊗ 1 N/n ),
which can be interpreted as replacing each node in the graph with N/n duplicated copies. The symmetric group
S n acts on V n by g • (A, X) = (gAg ⊤ , gX).
The space V ∞ can be identified with the space with all step graphons (and signals) in a similar way: Given (A, X) ∈ R n×n sym × R n×d , define W A : [0, 1] 2 → R, f X : [0, 1] → R d such that W A takes value A i,j on the interval I i,n × I j,n ⊂ [0, 1] 2 for i, j = 1, . . . , n ∈ [n], and f X takes value X i on the interval I i,n ⊂ [0, 1] for i = 1, . . . , n . We call (W A , f X ) the induced step graphon from (A, X). Under this identification, permutations S n permute the n intervals I i,n and act on
(W, f ) by σ • (W, f ) = (σ • W, σ • f ) = (W σ -1 , f • σ -1 ), where W σ -1 is defined by W σ -1 (x, y) := W (σ -1 (x), σ -1 (y)). The limit group G ∞ is the union of such interval permutations. The p-norm on V G dup . Fix ∥ • ∥ R d a norm on R d . We equip V n with a p-norm given by ∥(A, X)∥ p := max(∥A∥ p , ∥X∥ p ) =    max 1 n 2 i,j∈[n] |A ij | p 1/p , 1 n n i=1 ∥X i: ∥ p R d 1/p p ∈ [1, ∞), max max i,j∈[n] |A ij |, max n i=1 ∥X i: ∥ R d p = ∞.(12)
It is easy to check by Proposition C.8 that this extends to a norm on V ∞ . Under the identification with step graphons, this norm on V ∞ coincides with the standard L p -norm given by
∥(W, f )∥ p := max (∥W ∥ p , ∥f ∥ p ) =    max |W (x, y)| p dx dy 1/p , |f (x)| p dx 1/p p ∈ [1, ∞), max sup x,y |W (x, y)|, sup x |f (x)| p = ∞.
this section cite: []

Section: That is, for any
(A, X) ∈ R n×n sym × R n×d , ∥(A, X)∥ p = ∥(W A , f X )∥ p . The symmetrized metric is d p ((W, f ), (W ′ , f ′ )) = inf σ∈G∞ {max (∥W -σ • W ′ ∥ p , ∥f -σ • f ′ ∥ p )}.(13)
Operator p-norm on V G dup . Fix ∥ • ∥ R d a norm on R d and p ∈ [1, ∞].
We equip V n with a norm given by
∥(A, X)∥ op,p := max 1 n ∥A∥ op,p , ∥X∥ p ,(14)
where ∥A∥ op,p is the operator norm of A with respect to the ℓ p norm, i.e.
∥A∥ op,p = max ∥x∥p=1 ∥Ax∥ p ,
and ∥X∥ p is the normalized ℓ p -norms with respect to ∥ • ∥ R d defined in (10). It is easy to check by Proposition C.8 that this extends to a norm on V ∞ .
Let T W be the shift operator of graphon W defined by T W (f )(u) :
= 1 0 W (u, v)f (v)dv.
Under the identification with step graphons, the norm on V ∞ coincides with ∥(W, f )∥ op,p := max (∥T W ∥ op,p , ∥f ∥ p ) , where ∥T W ∥ op,p := sup ∥f ∥p=1 ∥T W (f )∥ p , and the norm on ∥f ∥ p is the L p norm as before. That is, for any
(A, X) ∈ R n×n sym × R n×d , ∥(A, X)∥ op,p = ∥(W A , f X )∥ op,p . For p ∈ [1, ∞), the completion with respect to this metric is V ∞ = {W ∈ L p ([0, 1] 2 ) : W (x, y) = W (y, x)} × {f ∈ L p ([0, 1], R d )},
the space of L p -graphon signals. We do not characterize V ∞ for p = ∞ since it requires additional technicalities. However, it contains all bounded and continuous graphon signals. The symmetrized metric is
d p ((W, f ), (W ′ , f ′ )) = inf σ∈G∞ {max (∥T W -T σ•W ′ ∥ op,p , ∥f -σ • f ′ ∥ p )}.(15)
Cut norm on V G dup . We can also equip V n with the cut norm (on matrices and vectors),
∥(A, X)∥ □ := max (∥A∥ □ , ∥X∥ □ ) = max   1 n 2 max S,T ⊆[n] i∈S,j∈T A ij , 1 n max S⊆[n] i∈S X i: R d   .(16)
It is easy to check by Proposition C.8 that this extends to a norm on V ∞ . Under the identification with step graphons, this norm on V ∞ coincides with the cut norm on graphons and graphon signals
∥(W, f )∥ □ := max (∥W ∥ □ , ∥f ∥ □ ) = max sup S,T ⊆[0,1] S×T W (x, y)dxdy , sup S⊆[0,1] S f (x)dx R d
, where the supremum is taken over all measurable sets S, T . The cut norm on graphon is studied in-depth in [54]. Though hard to compute, it has strong combinatorial interpretations. Hence, it has played an important role in the work of GNN transferability, and has been extended to the graphon signals in [47], which we have adopted here.
The symmetrized metric is
d((W, f ), (W ′ , f ′ )) := inf σ∈G∞ {max (∥W -σ • W ′ ∥ □ , ∥f -σ • f ′ ∥ □ )}.
It can be proved that this exactly coincides with the cut distance below, defined on graphon signals (extending the original definition of graphon cut distance from [54], similarly to the version in [47]):
d((W, f ), (W ′ , f ′ )) = inf σ∈S [0,1] {max (∥W -σ • W ′ ∥ □ , ∥f -σ • f ′ ∥ □ )} ,(17)
where S [0,1] is the group of measure-preserving bijections σ : [0, 1] → [0, 1] with measurable inverse. The proof follows analogously to [8,Lemma 3.5]. Specifically, we first verify that the definitions agree on step graphons. Then, since both definitions are continuous with respect to the cut norm ∥ • ∥ □ , they must also agree on the limit points. While the cut norm is hard to work with directly, it is topologically equivalent to the operator 2-norms considered previously on a bounded domain. This means that any function continuous with respect to one of these norms is also continuous with respect to the other.
Proposition G.1. If ∥W ∥ ∞ < r and ∥f ∥ ∞ < r, then ∥(W, f )∥ □ ≤ ∥(W, f )∥ op,2 ≲ ∥(W, f )∥ 1/2
□ . Consequently, for p ∈ (1, ∞), ∥ • ∥ □ and ∥ • ∥ p are topologically equivalent on the space
W : [0, 1] 2 → [-r, r] measurable, W (x, y) = W (y, x) × {f : [0, 1] → [-r, r] measurable} . Proof. Without loss of generality, let r = 1. Consider the norm ∥T W ∥ op,∞,1 := sup ∥f ∥∞,∥g∥∞≤1 1 0 1 0 W (u, v)f (u)g(v)dudv . By [37, Equation 4.4], ∥W ∥ □ ≤ ∥T W ∥ op,∞,1 ≤ 4∥W ∥ □ . By [37, Lemma E.6], if ∥W ∥ ∞ ≤ 1, ∥T W ∥ op,∞,1 ≤ ∥T W ∥ op,2 ≤ √ 2∥T W ∥ 1/2 op,∞,1 . Combining the inequalities, ∥W ∥ □ ≤ ∥T W ∥ op,2 ≤ 2 3/2 ∥W ∥ 1/2 □ . Moreover, by [47, Appendix A.2], ∥f ∥ □ ≤ ∥f ∥ 1 ≤ 2∥f ∥ □ . If ∥f ∥ ∞ ≤ 1, then ∥f ∥ 2 2 ≤ ∥f ∥ 1 ≤ ∥f ∥ 2 . Combining the inequalities, ∥f ∥ □ ≤ ∥f ∥ 2 ≤ 2 1/2 ∥f ∥ 1/2 □ .
Therefore, we conclude that
∥(W, f )∥ □ ≤ ∥(W, f )∥ op,p ≤ 2 3/2 ∥(W, f )∥ 1/2 □ , as claimed.
this section cite: ['b9', 'b53', 'b46', 'b53', 'b46', 'b7']

Section: G.2 Message Passing Neural Networks (MPNNs)
Background. MPNN parametrizes a sequence of functions (MPNN n : R n×n sym × R n×d1 → R n×d L ) by composition of message passing layers. The l-th message passing layer
MP (l) n : R n×n sym × R n×d l → R n×n sym × R n×d l+1 , (A, X (l) ) → (A, X (l+1) )
is given by
X (l+1) i: = ϕ (l) X (l) i: , Agg j∈Ni ψ (l) X (l) i: , X (l) j: , A ij , i = 1, . . . , n,(18)
where Agg is a permutation-invariant aggregation function such as sum, mean, or max; N i := {j : A ij ̸ = 0} denotes the neighborhood of node i in the input graph; the message function
ψ (l) : R d l × R d l × R → R h l and the update function ϕ (l) : R d l × R h l → R d l+1 are independent of the graph size n. Composing L message-passing layers defines an MPNN, mapping (A, X (1) ) → X (L) .
Observe that MPNN is permutation-equivariant:
MPNN n (gAg ⊤ , gX) = gMPNN n (A, X) for all g ∈ S n .
If we want a permutation-invariant function, this is followed by a read-out operation taking the form of DeepSet. In this work, we focus on the equivariant case.
MPNN is a general framework for GNNs based on local message passing: [31] formulates multiple GNNs as MPNNs with specific choices of ϕ, ψ, Agg; other state-of-the-art GNNs can be simulated by MPNN on a transformed graph [38]. Moreover, ϕ and ψ can also be parameterized with MLPs to provide good flexibility.
Transferability analysis of MPNNs. The following corollary gives sufficient conditions on the layers of MPNNs to obtain Lipschitz transferability.
Corollary G.2. Consider one message passing layer, (MP (l) n ), as defined in (18), with the following properties.
1. The message function ψ (l) takes the form ψ (l) (x 1 , x 2 , w) := wξ(x 2 ), where
ξ : R d l → R h l is L ξ Lipschitz with respect to ∥ • ∥ R d l , ∥ • ∥ R h l .
2. The aggregation used is the normalized sum aggregation Agg j∈Ni := 1 n j∈Ni .
this section cite: ['b30', 'b37']

Section: The update function
ϕ (l) is L ϕ Lipschitz, i.e. for all (x, y), (x ′ , y ′ ) ∈ R d l × R h l , ∥ϕ (l) (x, y) -ϕ (l) (x ′ , y ′ )∥ R d l+1 ≤ L ϕ max (∥x -x ′ ∥ R d l , ∥y -y ′ ∥ R h l )
Endow the space of duplication-consistent sequences with the operator p-norm as defined in (14), where p ∈ [1, ∞). Then, the sequence of maps (MP
(l) n ) is locally Lipschitz transferable. Remark G.3. That is, (MPNN n ),
which is a composition of message-passing layers, is locally Lipschitz transferable. Consequently it extends to a function MPNN ∞ on the space of graphon signals, which is L(r)-Lipschitz on B(0, r) for all r > 0 with respect to the symmetrized operator p-metric defined in ( 15)). By Proposition G.1, the sequence of maps (MPNN n ) is continuously transferable with respect to the cut norm (16) on B(0, r). The GNN studied in [70] is a special case of our MPNN considered here; meanwhile, ours is a special case of [47], which directly establishes Lipschitzness with respect to the cut distance by analysis on the graphon space. While our results are not new, our proof technique-following Proposition 5.1-is new and generally applicable to various models.
Proof. We decompose MP (l)  n as a composition of the following maps, modelling each of the intermediate spaces using the duplication consistent sequences endowed with compatible norms. For the metric on the product spaces, we always use the L ∞ product metric, i.e., taking the maximum over the individual components. Consider the following three functions
f (1) n : R n×n sym × R n×d l → R n×n sym × R n×d l × R n×h l given by (A, X) →   A, X,    ξ(X 1: ) . . . ξ(X n: )       , f (2) n : R n×n sym × R n×d l × R n×h l → R n×n sym × R n×d l × R n×h l given by (A, X, Y 0 ) → (A, X, 1 n AY 0 ), f (3) n : R n×n sym × R n×d l × R n×h l → R n×n sym × R n×d l+1 given by (A, X, Y ) → (A, X),
where Xi: = ϕ (l) (X i: , Y i: ). It is straightforward to check that each of them is compatible with respect to the duplication embedding. We now check the Lipschitz transferability.
• The sequence (f
(1) n ) is Lipschitz transferable because f (1) n (A, X) -f (1) n (A ′ , X ′ ) op,p ≤ (1 ∨ L ξ ) ∥(A, X) -(A ′ , X ′ )∥ op,p .
• The sequence (f
n ) is locally Lipschitz transferable because we can bound its Jacobian. In particular, the Jacobian acts on
(H A , H X , H Y ) via Df (2) n (A, X, Y 0 )[H A , H X , H Y ] = H A , H X , 1 n (AH Y + H A Y ) . Hence, on {(A, X, Y 0 ) : ∥(A, X, Y 0 )∥ op,p < r}, Df (2) n (A, X, Y 0 )[H A , H X , H Y ] op,p ≤ max 1 n ∥H A ∥ op,p , ∥H X ∥ p , 1 n ∥A∥ op,p ∥H Y ∥ p + 1 n ∥H A ∥ op,p ∥Y ∥ p ≤ (1 ∨ 2r)∥(H A , H X , H Y )∥ op,p , i.e., f(2)
n is (1 ∨ 2r) Lipschitz on this space.
• The sequence (f
n ) is Lipschitz transferable because ∥f (3) n (A, X, Y ) -f (3) n (A ′ , X ′ , Y ′ )∥ p = max 1 n ∥A -A ′ ∥ op,p , ∥ X -X′ ∥ p where ∥ X -X′ ∥ p p ≤ 1 n n i=1 ∥ϕ (l) (X i: , Y i: ) -ϕ (l) (X ′ i: , Y ′ i: )∥ p R d l+1 ≤ L p ϕ 1 n n i=1 (∥X i: -X ′ i: ∥ R d l + ∥Y i: -Y ′ i: ∥ R h l ) p ≤ L p ϕ 2 p-1 1 n n i=1 ∥X i: -X ′ i: ∥ p R d l + ∥Y i: -Y ′ i: ∥ p R h l(3)
(by Jensen's inequality a+b 2 p ≤ a p +b p 2 for all a, b)
≤ L p ϕ 2 p ∥(X, Y ) -(X ′ , Y ′ )∥ p p . So f (3) n is (1 ∨ 2L ϕ )-Lipschitz.
Finally, apply Proposition 5.1, (MP (l)  n ) is locally Lipschitz transferable; completing the proof.
associated with higher-order tensors, in practice, only k-IGNs for k ≤ 2 have been implemented to the best of our knowledge. In this work, we focus exclusively on 2-IGNs.
The basis in ( 19) is inherently dimension-agnostic, allowing IGN to serve as an any-dimensional neural network that parameterizes functions on inputs of arbitrary size n using a fixed set of parameters. This feature fundamentally relies on representation stability, which is discussed in greater detail in [50].
this section cite: ['b69', 'b46', 'b49']

Section: Incompatibility of IGN.
2-IGN is incompatible with the subspace V G dup . First, its basis functions are not properly normalized, and therefore cannot be extended to functions on graphons. For instance, the fourth basis function ℓ 4 (A) = A11 ⊤ yields output entries of order O(n), and should thus be normalized by a factor of n -1 . To address this issue, [11] introduces a normalized version of 2-IGN, defined by
W (i) n (A) = α 1 A + α 2 A ⊤ + α 3 diag(diag * (A)) + α 4 1 n A11 ⊤ + α 5 1 n 11 ⊤ A + α 6 1 n diag(A1) + α 7 1 n A ⊤ 11 ⊤ + α 8 1 n 11 ⊤ A ⊤ + α 9 1 n diag(A ⊤ 1) + α 10 1 n 2 (1 ⊤ A1)11 ⊤ + α 11 1 n 2 (1 ⊤ A1)diag(1) + α 12 (1 ⊤ diag * (A))11 ⊤ + α 13 (1 ⊤ diag * (A))diag(1) + α 14 diag * (A)1 ⊤ + α 15 1diag * (A) ⊤ + β 1 11 ⊤ + β 2 diag(1).(20)
However, the normalized 2-IGN is still not compatible. Consider the third basis function ℓ 3 (A) := diag(diag * (A)). It fails to satisfy the compatibility condition:
ℓ 3 (A ⊗ 1 m ) ̸ = ℓ 3 (A) ⊗ 1 m , m ≥ 2,
as the left-hand side yields a diagonal matrix, while the right-hand side generally does not. In fact, all basis maps that output diagonal matrices share this incompatibility.
Nonetheless, our Proposition 5.1 immediately provides a constructive recipe for making 2-IGN transferable: we start from a basis for linear equivariant layers W (i) n -which is compatible under duplication-and then select only the basis elements which have a finite operator norm as n grows. Furthermore, we use nonlinearities ρ (i) n which are compatible and Lipschitz continuous. Following this recipe, we introduce two modified versions of 2-IGN: Generalizable Graph Neural Network (GGNN): Compatible with respect to V G dup , locally Lipschitz transferable under the ∞-norm.
this section cite: ['b10']

Section: Continuous GGNN: Compatible with respect to V G
dup , locally Lipschitz transferable under the operator 2-norm, and continuously transferable under the cut-norm.
We highlight that this is a general methodology for constructing transferable equivariant networks: the framework established in [50] yields bases for compatible equivariant linear layers. We can then select only those basis elements whose operator norms do not grow with dimension, which we have shown yields a transferable neural network. 1)  n , where for the following conditions hold for all i.
this section cite: ['b49']

Section: GGNN architecture. A D-layer GGNN parameterizes an S
n -equivariant function R n×n sym ×R n×d ′ 1 → R n×n sym × R n×d ′ D defined via the composition W (D) n • ρ (D-1) n • • • • • ρ (1) n • W (
• The linear map
W (i) n : (R n×n sym ) ⊕di ⊕ (R n ) ⊕d ′ i → (R n×n sym ) ⊕di ⊕ ((R n ) ⊕d ′ i ) ⊕S
is a is S nequivariant and compatible with the duplication embedding.
• The functions ρ
(i) n : (R n×n sym ) ⊕di ⊕ ((R n ) ⊕d ′ i ) ⊕S → (R n×n sym ) ⊕di ⊕ (R n ) ⊕d ′ i
that are compatible with respect with the duplication embedding.
Here, d and d ′ are feature channels of A and X, respectively-we fix d 1 = d D = 1.
For the ease of notation, we assume d i = d i+1 = 1 (The general case follows analogously). The maps W (i) n , ρ (i) n are given by
W (i) n (A, X) = (A ′ , (X ′ s ) S s=0 ) = α 1 A + α 2 1 ⊤ A1 n 2 11 ⊤ + α 3 Tr(A) n 11 ⊤ + α 4 1 n (A11 ⊤ + 11 ⊤ A) + α 5 (diag(A)1 ⊤ + 1diag(A) ⊤ ) + d ′ i j=1 α 6,j (X :,j 1 ⊤ + 1X ⊤ :,j ) + α 7,j 1 n (1 ⊤ X :,j )11 ⊤ + β 1 11 ⊤ , XΘ 1,s + 1 n 11 ⊤ XΘ 2,s + 1 n A1θ ⊤ 1,s + diag(A)θ ⊤ 2,s + Tr(A) n 1θ ⊤ 3,s + 1 ⊤ A1 n 2 1θ ⊤ 4,s + 1β ⊤ 2,s , ρ (i) n (A, (X s ) S s=0 ) = A, σ S s=0 n -s A s X s(21)
where α, θ, β are learnable parameters, and σ : R → R is an arbitrary L-Lipschitz entrywise nonlinearity.
Consider the input and output spaces as (variants of) V G dup , the duplication consistent sequences for graph signals. The linear layer W n in (21) parameterizes all linear S n -equivariant maps between these two spaces that are also compatible with the duplication embedding.
The GGNN model is a modification of the 2-IGN (19) with three key differences. Firstly, we treat the adjacency matrix and node features separately so that each layer has a graph and a signal component. Moreover, we explicitly require the matrix component to be symmetric. Secondly, we impose the compatibility with respect to the duplication embedding on the linear layers. This leads to both proper normalization of each basis function and a reduction in the total number of basis functions. In particular, all basis functions that output a diagonal matrix are removed. Thirdly, for the nonlinearity ρ n , instead of the entrywise nonlinearity used in IGN, we adopt a message-passing-like nonlinearity. The form of the nonlinearity mirrors the GNN model studied in [70]. Particularly, in (21), if we set all the α's to 0 except α 1 = 1, and all the θ's to 0 except Θ 1,s , then we exactly recover the transferable GNN in [70].Therefore, our model is at least as expressive as the GNN in [70].
Transferability analysis of GGNN. Even though we only impose compatibility by design, we can still prove that GGNN is Lipschitz transferable with respect to some norm. Albeit this norm is arguably too weak. (21), where the entrywise nonlinearity σ is L σ -Lipschitz. Endow the space of duplication-consistent sequences with the ∞-norm as defined in (12) (with respect to ∥ • ∥ ∞ on R d ′ i ). Then, the sequence of maps (GGNN n ) is locally Lipschitz transferable. Consequently, (GGNN n ) extends to a function GGNN ∞ on the space of graphon signals, which is L(r)-Lipschitz on B(0, r) with respect to the symmetrized metric defined in (13) with p = ∞.
Corollary G.5. Consider one layer of GGNN, (GGNN n (A, X) = ρ (i) n • W (i) n ), as defined in
Proof. The sequence of linear maps (W 21) is Lipschitz transferable because
(i) n ) in (
∥W (i) n ∥ op ≤ max |α 1 | + |α 2 | + |α 3 | + 2|α 4 | + 2|α 5 | + d ′ i j=1 (|α 6,j | + |α 7,j |), ∥Θ 1,s ∥ op,1,1 + ∥Θ 2,s ∥ op,1,1 + ∥θ 1,s ∥ ∞ + ∥θ 2,s ∥ ∞ + ∥θ 3,s ∥ ∞ + ∥θ 4,s ∥ ∞ ,
where ∥Θ∥ op,1,1 = max j k |θ k,j | is the max ℓ 1 norm of a column.
For the nonlinearity (ρ (i) n ), we consider its Fréchet derivative (since all norms are equivalent in finite-dimensional vector spaces, this is independent of the norm chosen):
Dρ (i) n (A, X 0 , . . . , X S )[H, H 0 , . . . , H S ] = H, S s=0 n -s s-1 k=0 A k HA s-1-k • X s + A s H s .
Hence,
∥Dρ (i) n (A, X 0 , . . . , X S )[H, H 0 , . . . , H S ]∥ ∞ ≤ max ∥H∥ ∞ , S s=0 s-1 k=0 ∥A∥ s-1 ∞ ∥H∥ ∞ ∥X s ∥ ∞ + ∥A∥ s ∞ ∥H s ∥ ∞ ≤ max ∥H∥ ∞ , S s=0 sr s ∥H∥ ∞ + r s ∥H s ∥ ∞ ≤ 1 ∨ S s=0 (sr s + r s ) =:L(r)
•∥(H, H 0 , . . . , H S )∥ ∞ .
Therefore, for all n, ρ
(i) n is L σ L(r)-Lipschitz on the set B n (0, r) = {(A, X 0 , . . . , X S ) : ∥A, X 0 , . . . , X S ∥ ∞ < r} . Applying Proposition 5.1, the sequence of maps (GGNN n ) is locally Lipschitz transferable, where the extension GGNN ∞ is (L σ L(∥W n ∥ op r)∥W n ∥ op )-Lipschitz on the set B(0, r) = {(W, f ) : ∥(W, f )∥ ∞ < r} .
Continuous GGNN architecture. We aim to further restrict GGNN to construct a model that is transferable with respect to the cut norm. By Proposition G.1, we consider endowing the consistent sequence with the operator 2-norm, which is easier to analyze. The Continuous GGNN is a variant of GGNN with an additional constraint on the linear layers W (i) n , requiring them to have bounded operator norm: ∥W n ∥ op < ∞ (with respect to the operator 2-norm). This constraint effectively leads to a further reduction in the set of basis functions:
W (i) n (A, X) = (A ′ , (X ′ s ) S s=0 ) = α 1 A + α 2 1 ⊤ A1 n 2 11 ⊤ + α 4 1 n (A11 ⊤ + 11 ⊤ A) + k j=1 α 6,j (X :,j 1 ⊤ + 1X ⊤ :,j ) + α 7,j 1 n (1 ⊤ X :,j )11 ⊤ , XΘ 1,s + 1 n 11 ⊤ XΘ 2,s + 1 n A1θ ⊤ 1,s + 1 ⊤ A1 n 2 1θ ⊤ 4,s ,(22)
Therefore, the hypothesis class of continuous GGNN forms a strict subset of that of GGNN, with the additional constraint enabling improved transferability. Meanwhile, for the same reasons outlined for GGNN, the continuous GGNN is also at least as expressive as the GNN proposed in [70]. We use (cGGNN n ) to denote the sequence of functions of continuous GGNN.
this section cite: ['b69', 'b20', 'b69', 'b69', 'b20', 'b69']

Section: Transferability analysis of Continuous GGNN.
Corollary G.6. Consider one layer of the continuous GGNN,
(cGGNN n (A, X) = ρ (i) n • W (i) n
), as defined in (22), where the entrywise nonlinearity σ is L σ -Lipschitz. Endow the space of duplicationconsistent sequences with the operator 2-norm as defined in (14) (with respect to ∥ • ∥ ∞ on R d ′ i ). Then, the sequence of maps (cGGNN n ) is locally Lipschitz transferable. Therefore, (cGGNN n ) extends to a function cGGNN ∞ on the space of graphon signals, which is L(r)-Lipschitz on B(0, r) with respect to the symmetrized operator 2-metric defined in (15) with p = 2. Remark G.7. By Proposition G.1, the sequence of maps (cGGNN n ) is continuously transferable with respect to the cut distance (17) on the space {(W, f ) : ∥(W, f )∥ op,2 < r, ∥W ∥ ∞ , ∥f ∥ ∞ < r}.
Moreover, for the convergence and transferability results stated in Proposition D.4,D.5, one can additionally obtain quantitative rates of convergence with respect to the cut distance.
Proof. First, by construction, the sequence of maps (W
(i) n ) is Lipschitz transferable because ∥W (i) n ∥ op ≤ max |α 1 | + |α 2 | + 2|α 4 | + d ′ i j=1 (|α 6,j | + |α 7,j |) , ∥Θ 1,s ∥ op,1 + ∥Θ 2,s ∥ op,1 + ∥θ 1,s ∥ ∞ + ∥θ 4,s ∥ ∞ < ∞.
For the nonlinearity (ρ
(i) n
), the action of its Jacobian yields
Dρ (i) n (A, X 0 , . . . , X S )[H, H 0 , . . . , H S ] = H, S s=0 n -s s-1 k=0 A k HA s-1-k • X s + A s H s .
Hence,
∥Dρ (i) n (A, X 0 , . . . , X S )[H, H 0 , . . . , H S ]∥ op,2 ≤ max n -1 ∥H∥ op,2 , S s=0 s-1 k=0 ∥A∥ s-1 op,2 n s-1 • ∥H∥ op,2 n • ∥X s ∥ 2 + ∥A∥ s op,2 n s • ∥H s ∥ 2 ≤ max n -1 ∥H∥ op,2 , S s=0 sr s • ∥H∥ op,2 n + r s ∥H s ∥ 2 ≤ 1 ∨ S s=0 (sr s + r s ) =:L(r)
•∥(H, H 0 , . . . , H S )∥ op,2 .
Therefore, for all n, ρ
n is L σ L(r)-Lipschitz on the set {(A, X 0 , . . . , X S ) : ∥A, X 0 , . . . , X S ∥ op,2 < r} .
this section cite: ['b21']

Section: Applying Proposition 5.1, the sequence of maps (f n ) is locally Lipschitz transferable, where the extension cGGNN
∞ is (L σ L(∥W n ∥ op r)∥W n ∥ op )-Lipschitz on the set B(0, r) = {(W, f ) : ∥(W, f )∥ op,2 < r} .
We can directly apply Propositions D.4 and D.5, together with the convergence rates established in Appendix D.3. This leads to transferability results for continuous GGNNs that are exactly the same as those for MPNNs, as stated in Corollary G.4.
Related work on IGN transferability. We discuss two closely related works, [11] and [35], that address the transferability of IGNs. Interpreting their results within our theoretical framework offers a better understanding of IGN transferability. As shown in our work, the normalized 2-IGN is not compatible with the duplication-consistent subspace V G dup , and thus fails to satisfy the convergence and transferability in Proposition 3.2. At first glance, this observation may seem to contradict [11,Theorem 2]. However, this is not the case. While [11] introduces cIGN, a "graphon analogue of IGN," and proves its continuity in the graphon space, it is crucial to note that the discrete IGN does not extend to cIGN in general:
IGN n (A n , X n ) ̸ = cIGN([A n ], [X n ]).
Therefore, the convergence of cIGN established in Theorem 2 of [11] does not imply the convergence or transferability of the finite-dimensional IGN model. Moreover, [11,Definition 6] introduces a constraint that resembles our compatibility condition, formulated through a restricted variant termed "IGN-small." Our definition of compatibility clarifies this notion and enables explicit constructions and practical implementations of compatible, transferable versions of IGNs.
In a more recent work, [35] adopts an approach similar to ours by imposing additional constraints on the linear layers of IGN, specifically requiring them to have bounded operator norm. This leads to the Invariant Graphon Network (IWN) model. Unlike our construction, IWN retains standard pointwise nonlinearities. As shown in [35,Proposition 5.5], it is precisely these point-wise nonlinearity layers that cause IWN to be discontinuous with respect to the cut norm, rendering it generally non-transferable in our setting. Interestingly, [35,Appendix G.4] observes that under suitable assumptions, IWN restricted to the space of simple-graph inputs (i.e., adjacency matrices with 0/1 entries) is Lipschitz continuous with respect to the cut norm and hence admit Lipschitz extensions. This implies convergence and transferability of IWN specifically under the "graphon sampling" scheme, where edges are Bernoulli-randomized. However, the limit of this convergence result does not align with the behavior on weighted graphs (i.e., adjacency matrices with non-binary entries). This discrepancy highlights the cost of lacking cut-norm continuity over the full space. This phenomenon may also explain Figure 4(c) in our GGNN experiments, where outputs on graphs sampled from the Erdős-Rényi model appear to converge (with diminishing error bars), yet to a different limit than those on the corresponding fully connected weighted graphs derived from the same graphon.
In the case of 2-IGN, our continuous cGGNN model provides a remedy for the lack of cut-norm continuity by replacing point-wise nonlinearities with message-passing-style operators, thereby ensuring Lipschitz continuity with respect to the cut norm and circumventing the issue. However, our construction is currently limited to 2-IGN and does not generalize to higher-order k-IGNs. As noted in [35, Section 5.1], cut-norm discontinuity is inherent to the k-WL hierarchy and is likely unavoidable for all higher-order GNNs with better expressivity.
Remarks on expressive power. The expressive power of GNNs has been widely studied in terms of their ability to distinguish non-isomorphic graphs, where a GNN is said to be k-WL expressive if it is as powerful as k-WL testing [36]. While this work does not explore expressivity, we note that the standard k-WL expressivity is not appropriate for studying the expressive power of graphoncompatible GNNs (i.e., GNNs that are compatible with the duplication-consistent sequence and thus extend to graphon space). This is because different-sized graphs corresponding to the same step graphon are always distinguishable by WL, but are considered equivalent by any graphon-compatible GNN. Instead, it is necessary to consider a variant of k-WL specifically designed for graphons [7,35].
this section cite: ['b10', 'b34', 'b10', 'b10', 'b10', 'b10', 'b34', 'b34', 'b34', 'b35', 'b6', 'b34']

Section: H Example 3 (point clouds): details and missing proofs from Section 5.3
In this appendix, we study the transferability of two any-dimensional architectures for point clouds. We start by presenting the consistent sequences we consider. Then, in Section H.2 we study the DS-CI model proposed in [5], which is known to be very expressive but computationally expensive. Finally, in Section H.3, we introduce a novel architecture that turns out to be much cheaper to compute.
this section cite: ['b4']

Section: H.1 Duplication consistent sequence for point clouds
The duplication consistent sequence for point clouds
V P dup = {(V n ), (φ N,n ), (G n )} is defined as follows. The index set is again N = (N, • | •). For each n, the vector spaces are V n = R n×k , with the group G n = S n × O(k) acting on V n by (g, h) • X = gXh ⊤ .
For any n | N , the embedding is given by,
φ N,n : R n×k ↩→ R N ×k X → X ⊗ 1 N/n ,
and the group embedding is
θ N,n : S n × O(k) ↩→ S N × O(k) (g, h) → (g ⊗ I N/n , h).
However, we can make some additional adjustments to ensure compatibility: we define the compatible DS-CI by modifying the inputs of DeepSet (2) to be {f a l (V V ⊤ )} l=1,...,n 2 , where f a l (X) denotes the l-th largest value among the entries X ij for 1 ≤ i, j ≤ n. Additionally, we replace f * (X) with f * (X) := 1 n 2 i,j
X ii X ij .
We denote the sequence of functions of compatible DS-CI by (C-DS-CI n ). We prove that this model is locally Lipschitz transferable.
Corollary H.1. Endow V P dup with the normalized ℓ p norm with p ∈ [1, ∞]. Assume that all activation functions in the MLPs used for DS-CI are Lipschitz. Then the sequence of maps (C-DS-CI n ) is Lipschitz transferable on the space {f : ∥f ∥ ∞ < r} for all r > 0.
Proof. By Proposition 5.1, it is sufficient to verify the compatibility and Lipschitz continuity of each individual layer.
• The sequence of maps
(R n×k , ∥ • ∥ p ) → (R n , ∥ • ∥ p ), V → diag * (V V ⊤ ) is (2r)-Lipschitz transferable. Indeed, it is S n -equivariant, O(k)-invariant, and diag * (V ⊗ 1 m )(V ⊗ 1 m ) ⊤ = diag * (V V ⊤ ) ⊗ (1 m 1 ⊤ m ) = diag * (V V ⊤ ) ⊗ 1 m , diag * (V V ⊤ ) -diag * (W W ⊤ ) p = 1 n n i=1 ∥V i: ∥ 2 2 -∥W i: ∥ 2 2 p 1/p = 1 n n i=1 |⟨V i: -W i: , V i: + W i: ⟩| p 1/p ≤ 1 n n i=1 ∥V i: -W i: ∥ p 2 • (2r) p 1/p (by Cauchy-Schwarz) = 2r∥V -W ∥ p , for V, W satisfying ∥V ∥ ∞ = max i ∥V i: ∥ 2 < r, ∥W ∥ ∞ = max i ∥W i: ∥ 2 < r.
• The sequence of maps
(R n×k , ∥ • ∥ p ) → (R n 2 , ∥ • ∥ p ), V → vec(V V ⊤ )
is (2r)-Lipschitz transferable, where the codomain is equipped with a consistent sequence structure as follows: for g ∈ S n , define π(g) = g ⊤ ⊗ g ∈ S n 2 , and let g act on R n 2 by g • x = π(g)x. The symmetric groups (S n ) are embedded into each other as usual, and the vector spaces are embedded by φ nm,n : R n 2 → R (nm) 2 where
φ nm,n (x) = vec(reshape n (x) ⊗ 1 m 1 ⊤ m ),
and reshape n : R n 2 → R n×n is the inverse of vec on n × n matrices. Since these are all linear maps, so is φ nm,n . We then have for all V ∈ R n×k , g ∈ S n , h ∈ O(k) that
vec (V ⊗ 1 m )(V ⊗ 1 m ) ⊤ = vec (V V ⊤ ) ⊗ (1 m 1 ⊤ m ) = φ nm,n (vec(V V ⊤ )), vec (gV h ⊤ )(gV h ⊤ ) ⊤ = vec(gV V ⊤ g ⊤ ) = π(g)vec(V V ⊤ ). Furthermore, ∥vec(V V ⊤ ) -vec(W W ⊤ )∥ p =   1 n 2 i,j |⟨V i: , V j: ⟩ -⟨W i: , W j: ⟩| p   1/p ≤   1 n 2 i,j (|⟨V i: , V j: -W j: ⟩| + |⟨V i: -W i: , W j: ⟩|) p   1/p ≤   1 n 2 i,j 2 p-1 r p (∥V j: -W j: ∥ p 2 + ∥V i: -W i: ∥ p 2 )   1/p (Using (a + b) p ≤ 2 p-1 (a p + b p ) and Cauchy-Schwarz) = 2r∥V -W ∥ p , for V, W satisfying ∥V ∥ ∞ = max i ∥V i: ∥ 2 < r, ∥W ∥ ∞ = max i ∥W i: ∥ 2 < r.
• The scalar maps
(R n×k , ∥ • ∥ p ) → (R, | • |), V → f * (V V ⊤ ) is (4r 3 )-Lipschitz transferable. Indeed, it is S n × O(k) invariant and f * ((V ⊗ 1 m )(V ⊗ 1 m ) ⊤ ) = f * (V V ⊤ ), f * (V V ⊤ ) -f * (W W ⊤ ) ≤ 1 n 2 i,
j ⟨V i: , V j: ⟩∥V i: ∥ 2 2 -⟨W i: , W j: ⟩∥W i: ∥ 2 2 ≤ 1 n 2 i,j |⟨V i: , V j: ⟩ -⟨W i: , W j: ⟩| ∥V i: ∥ 2 2 + |⟨W i: , W j: ⟩| ∥V i: ∥ 2 2 -∥W i: ∥ 2 2 ≤ 4r 3 ∥V -W ∥ 1 (by the previous two computations)
≤ 4r 3 ∥V -W ∥ p , for V, W satisfying ∥V ∥ ∞ = max i ∥V i: ∥ 2 < r, ∥W ∥ ∞ = max i ∥W i: ∥ 2 < r.
• If the activation functions used are Lipschitz, then MLPs are Lipschitz. By Corollary F.4, the normalized DeepSet is Lipschitz transferable, assuming the constituent MLPs are Lipschitz.
Thus, our compatible DS-CI architecture is a composition of Lipschitz layers.
Finally, we conclude that the normalized DS-CI is "approximately transferable" since it is asymptotically equivalent to the compatible DS-CI up to an error of O(n -1 ).
Lemma H.2. If the activations in all the MLPs used are Lipschitz, then for any sequence of inputs
V (n) ∈ R n×k , | C-DS-CI n (V (n) ) -DS-CI n (V (n) )| = O(n -1 ) Proof. Assume for x ∈ R n , DeepSet (2) (x) = σ( 1 n i ρ(x i ))
, and σ, ρ are L σ , L ρ Lipschitz respectively. Then, we have
f * (V V ⊤ ) -f * (V V ⊤ ) ≤ 1 n(n -1) - 1 n 2 i̸ =j (V V ⊤ ) ii (V V ⊤ ) ij + 1 n 2 i (V V ⊤ ) 2 ii = O(n -1 )
and, moreover,
DeepSet (2) f o ℓ (V V ⊤ ) ℓ=1,...,n(n-1)/2 -DeepSet (2) f a ℓ (V V ⊤ ) ℓ=1,...,n 2 ≤ L σ   1 n(n -1) - 1 n 2 i̸ =j ρ((V V ⊤ ) ij ) + 1 n 2 i ρ((V V ⊤ ) ii )   = O(n -1 ).
Since every layer is Lipschitz, this leads to an overall error of O(n -1 ).
Following the analysis above, we can directly apply Propositions D.4 and D.5, together with the convergence rates established in Appendix D.3. Moreover, observe that the O(n -1 ) discrepancy between normalized DS-CI and compatible DS-CI is dominated by the convergence rate of interest. This immediately yields the following transferability result.
this section cite: []

Section: Corollary H.3 (Transferability of normalized DS-CI).
Let (X n ) ∈ R n×k be a sequence of matrices sampled from the same underlying distribution µ ∈ P 3 (R k ) with bounded support in the following way: first, sample Y n ∈ R n×k with rows drawn i.i.d. from µ. Then, let G ∈ O(k) be a (random or deterministic) rotation matrix, sampled independently of Y n , and define
X n = Y n G. Then, E DS-CI n (X n ) -DS-CI m (X m ) ≲    n -1/2 + m -1/2 if k = 1, n -1/2 log(1 + n) + m -1/2 log(1 + m) if k = 2, n -1/k + m -1/k if k > 2.
this section cite: []

Section: H.3 Constructing new transferable models: SVD-DS
We propose the SVD-DS model defined as follows:
SVD-DS n : R n×k → R, X → DeepSet n (XV ),
where X = U DV ⊤ is the singular value decomposition (SVD) for X with ordered singular values. We proceed to show that it is locally transferable almost everywhere on its domain, and that its performance is competitive with DS-CI while being more computationally efficient.
Transferability analysis of SVD-DS. We extend the SVD to elements in the limit space V ∞ = L 2 ([0, 1], R k ) and analyze its continuity, yielding the following transferability result. Recall our definition of locally Lipschitz transferable at a point in Appendix D.1.
Corollary H.4. Endow the duplication consistent sequences with the normalized ℓ 2 norm induced by ∥ • ∥ 2 on R k . Observe that
∥X∥ 2 := 1 n n i=1 ∥X i: ∥ 2 2 1/2 = 1 √ n ∥X∥ F ,
where ∥ • ∥ F denotes the Frobenius norm of a matrix. Then, the sequence of maps (SVD-DS n ) is compatible and locally Lipschitz transferable at every point except for a set of measure zero, corresponding to points with non-distinct singular values.
Remark H.5. This transferability result is weaker than the "L(r)-locally Lipschitz transferability" defined in Definition 3.1, since our model may be discontinuous at points with non-distinct singular values. Therefore, in this case our transferability results in Propositions D.4 and D.5 only apply to sequences (x n ) converging to a limit x ∈ V ∞ with distinct singular values.
Proof. Decompose SVD-DS n as the composition
V P dup = {R n×k } X →XV -----→ U = {R n×k } DeepSet -----→ V R ,
where V R is the trivial consistent sequence over R, and the consistent sequence U consists of vector spaces U n = R n×k under the duplication embedding ⊗1. The group S n × O(k) acts on U n by (g, h) • X = gX, i.e., the action of O(k) is trivial. By Corollary F.4, the normalized DeepSet map is Lipschitz transferable, assuming the constituent MLPs are Lipschitz. It remains to show that the SVD-based map X → XV extends to a function that is locally Lipschitz at every point with distinct singular values. We show this in Proposition H.7 below after extending the SVD to all of V ∞ and considering its ambiguities.
Following the analysis above, we can directly apply Propositions D.4 and D.5, together with the convergence rates established in Appendix D.3. These results immediately yield a transferability result for SVD-DS.
this section cite: []

Section: Corollary H.6 (Transferability of SVD-DS).
Let (X n ) ∈ R n×k be a sequence of matrices sampled from the same underlying distribution µ ∈ P 4 (R k ) in the following way: First, sample Y n ∈ R n×k with rows drawn i.i.d. from µ. Then, let G ∈ O(k) be a (random or deterministic) rotation matrix, sampled independently of Y n , and define
X n = Y n G. Suppose the second moment E x∼µ xx ⊤ ∈ R k×k of µ has k distinct eigenvalues, then E SVD-DS n (X n ) -SVD-DS m (X m ) ≲    n -1/4 + m -1/4 if k < 4, n -1/4 log 1/2 (1 + n) + m -1/4 log 1/2 (1 + m) if k = 4, n -1/k + m -1/k if k > 4.
Note that the functional SVD is locally Lipschitz only at points where the singular values are distinct. This motivates our assumption on the distribution µ in the Corollary. To see this, observe that when n ≥ k, the singular values of X n are the same as those of Y n , and are the square roots of the eigenvalues of --→ Σ mn for all m, n ∈ [k], where Σ := E x∼µ [xx ⊤ ] is the second-moment of µ. It follows from Weyl's theorem that each eigenvalue converges almost surely to those of Σ:
Y ⊤ n Y n = n i=1 x i x ⊤ i ,
λ j 1 n n i=1 x i x ⊤ i -λ j (Σ) ≤ 1 n n i=1 x i x ⊤ i -Σ 2 a.s.
→ 0 for all j ∈ [k].
Therefore, if Σ has distinct eigenvalues, then with probability one, the empirical matrix has distinct eigenvalues for all sufficiently large n, ensuring that the functional SVD is locally Lipschitz at this point. Note that X vanishes identically on V k = span{f 1 , . . . , f k } ⊥ , while X : V k → R k is a linear map between finite-dimensional vector spaces and therefore admits a singular value decomposition. Thus, there exists positive numbers σ ∈ R k ≥0 , orthonormal v 1 , . . . , v k ∈ R k , and orthonormal functions
this section cite: []

Section: Functional
u 1 , . . . , u k ∈ L 2 ([0, 1]) satisfying X = k i=1 σ i ⟨u i , •⟩v i .(24)
If X ∈ R n×k and X = k i=1 σi ũi ṽ⊤ i , is the usual SVD of X, then σ i = σi / √ n, v i = ṽi , and (24). Conversely, if (24) is the functional SVD of such an X then X =
u i (t) = √ n[ũ i ] ⌈nt⌉ is the functional SVD of X as in
k i=1 (σ i √ n)([u i (j/n)] k j=1 / √ n)v ⊤ i is the usual SVD of X.
Note that the right singular vectors V are the same in both SVDs.
If for any X ∈ V n we let σ(X) be its (functional) singular values from (24) and σ(X) be its usual singular values, then
∥X∥ 2 2 = 1 n ∥X∥ 2 F = 1 n k i=1 σi (X) 2 = k i=1 σ i (X) 2 ,(25)
and by Mirsky's inequality [61],
∥σ(X) -σ(Y )∥ 2 = 1 √ n ∥σ(X) -σ(Y )∥ 2 ≤ 1 √ n ∥X -Y ∥ F = ∥X -Y ∥ 2 .(26)
Furthermore, whenever V ∈ R k×k and X ∈ V n we have
∥XV ∥ 2 = 1 √ n ∥XV ∥ F ≤ 1 √ n ∥V ∥ F σ1 (X) = ∥V ∥ F σ 1 (X).(27)
Note that the final bounds in all of the above inequalities are independent of n, continuous in ∥ • ∥, and hold for all X ∈ V ∞ . We therefore conclude that they also hold for all X ∈ V ∞ , with ∥ • ∥ 2 replaced with ∥ • ∥ HS .
There is an ambiguity in the above decomposition, since if (u i ), (v i ) satisfy (24) then so do (s i u i ), (s i v i ) for any choice of signs s i ∈ {±1}. Furthermore, if the singular values (σ i ) are distinct then this is the only ambiguity in (24), see [13]. To disambiguate the SVD, we therefore choose signs so that v i > -v i in lexicographic order. When the entries of the v i are all nonzero, this amounts to requiring the first row of V = [v 1 , . . . , v k ] to be positive. We proceed to prove that the map X → V (X) = [v 1 , . . . , v k ] with this choice of signs is locally Lipschitz continuous on a dense subset of V ∞ .
Proposition H.7. Fix X 0 ∈ V ∞ with distinct singular values and all-nonzero entries in its right singular vectors (v i ). Let gap p (X 0 ) = min 2≤i≤k {σ i-1 (X 0 ) pσ i (X 0 ) p } be the minimum gap between p-th powers of (functional) singular values of X 0 , and set
B(X 0 ) = √ 8(2σ 1 (X 0 ) + 1) gap 2 (X 0 ) .(28)
For any X ∈ V ∞ satisfying
∥X 0 -X∥ HS ≤ 1 ∧ gap 1 (X 0 ) 2 √ k ∧ 1 2B(X 0 ) min 1≤i,j≤k |[v i (X 0 )] j |,
the following two hold true.
1. The matrix X has distinct singular values, and all nonzero entries of v i ( X) have the same sign as those of v i (X 0 ).
this section cite: ['b23', 'b23', 'b23', 'b60', 'b23', 'b12']

Section: We have
∥V (X 0 ) -V ( X)∥ F ≤ kB(X 0 )∥X 0 -X∥ HS ,(29)
and
∥X 0 V (X 0 ) -XV ( X)∥ HS ≤ (kσ 1 (X 0 )B(X 0 ) + 1)∥X 0 -X∥ HS .(30)
Proof. We start by establishing the first claim. If ∥X 0 -X∥ HS ≤ gap 1 (X0) 2 √ k , then for any 2 ≤ i ≤ k we have by
σ i-1 ( X) -σ i ( X) = σ i-1 ( X) -σ i-1 (X 0 ) + σ i-1 (X 0 ) -σ i (X 0 ) + σ i (X 0 ) -σ i ( X) ≥ (σ i-1 (X 0 ) -σ i (X 0 )) - k i=1 |σ i ( X) -σ i (X 0 )| ≥ gap 1 (X 0 ) - √ k∥σ(X 0 ) -σ( X)∥ 2 (Cauchy-Schwarz) ≥ gap 1 (X 0 ) 2 > 0.
this section cite: []

Section: (Mirsky's inequality (26))
Thus, X has distinct singular values.
Let gap p (X 0 ) = min 2≤i≤k {σ i-1 (X 0 ) pσi (X 0 ) p } be the minimum gap between p-th powers of (usual) singular values of X 0 . For X 0 , X ∈ V n , the result [90,Thm. 4] shows that for each i ∈
[k] min s∈{±1} ∥v i (X 0 ) -s • v i ( X)∥ 2 ≤ √ 8(2σ 1 (X 0 ) + σ1 (X 0 -X))∥X 0 -X∥ F gap 2 (X 0 ) = √ 8n(2σ 1 (X 0 ) + σ 1 (X 0 -X))∥X 0 -X∥ F ngap 2 (X 0 ) (since σi (X) = σ i (X) √ n) = √ 8(2σ 1 (X 0 ) + σ 1 (X 0 -X))∥X 0 -X∥ 2 gap 2 (X 0 ) (since ∥X∥ 2 = ∥X∥ F / √ n) ≤ √ 8(2σ 1 (X 0 ) + ∥X 0 -X∥ 2 )∥X 0 -X∥ 2 gap 2 (X 0 ) (by (25), σ 1 (X) 2 ≤ ∥X∥ 2 2 = k i=1 σ i (X) 2 ) ≤ √ 8(2σ 1 (X 0 ) + 1)∥X 0 -X∥ 2 gap 2 (X 0 ) (since ∥ X -X 0 ∥ 2 ≤ 1) = B(X 0 )∥X 0 -X∥ 2 .
The final bound is independent of n and hence applies on all of V ∞ . It is also continuous in ∥ • ∥ 2 on the dense subset of V ∞ consisting of operators with distinct singular values, so taking closures we conclude that this bound applies for any X 0 , X ∈ V ∞ such that gap 2 (X 0 ) > 0. Combining the last line above with our bound on ∥ X -X 0 ∥ 2 , we get
min s∈{±1} max 1≤j≤k [v i (X 0 )] j -s • [v i ( X)] j ≤ min s∈{±1} v i (X 0 ) -s • v i ( X) 2 ≤ 1 2 min 1≤i,j≤k |[v i (X 0 )] j | .(31)
Thus, the sign s achieving the above minimum is the one making all entries of v i ( X) have the same sign as those of v i (X 0 ). Since the first entries of v i (X 0 ) and of v i ( X) are positive, we conclude that s = 1 achieves the above minimum. To prove the second claim, we combine the bounds above, which yields
∥V (X 0 ) -V ( X)∥ F ≤ k i=1 ∥v i (X 0 ) -v i ( X)∥ 2 = k i=1 min s∈{±1} ∥v i (X 0 ) -s • v i ( X)∥ 2 ≤ kB(X 0 )∥X 0 -X∥ HS ,
as claimed. Finally, applying the triangle inequality gives
∥X 0 V (X 0 ) -XV ( X)∥ HS ≤ ∥X 0 (V (X 0 ) -V ( X)) ⊤ ∥ HS + ∥(X 0 -X)V ( X)∥ HS ≤ kB(X 0 )σ 1 (X 0 )∥X 0 -X∥ HS + ∥X 0 -X∥ HS ,
(by ( 27)) yielding the last claim.
SVD-DS is computationally more efficient than DS-CI. When k ≪ n (for example, for us k = 2 or 3), to evaluate SVD-DS on a given input of size n × k, we need to compute its SVD-which takes O(nk 2 ) flops [32, Section 8.6]-and then evaluate DeepSet on the output, which takes O(nC(k)) where C(k) is the cost of evaluating the involved fixed-size MLPs taking inputs of size k. Moreover, during training, we can compute the SVD of the dataset once in advance. In contrast, for DS-CI we need to form V V ⊤ at a cost of O(n 2 k), and this needs to be differentiated through during training. After forming V V ⊤ , we evaluate normalized DeepSets on its entries at a cost of O(n 2 ). Thus, SVD-DS is much faster to train and deploy than DS-CI.
and hidden dimensions such that each model has approximately 2k parameters to ensure an arguably fair comparison. During training, we use a dataset of N = 5000 graphs, each with n train = 50 nodes. This dataset is randomly split into training, validation, and test with proportions 60%, 20%, and 20%, respectively. For evaluation, we test the trained models on datasets of graph sizes n test ∈ {50, 200, 500, 1000, 2000}, each containing N = 1000 graphs. Training is performed by minimizing the MSE loss using AdamW with an initial learning rate of 0.001 and weight decay of 0.1. Each model is trained for 500 epochs. Training a single run takes less than 3 minutes for the GNN model, and approximately 6-9 minutes for the IGN, GGNN, and continuous GGNN models, which are more computationally intensive. We note that evaluation on large graphs is particularly time-and memory-intensive for these models, taking up to several hours. Memory limitations restrict the maximum graph size we can evaluate to n = 2000.
The results of the size generalization experiments are summarized in Figure 9.
Let us elaborate on these results. Since the target function naturally extends to the graphon-level signal-weighted triangle density (W, f ) → g, given by g(x) = [0,1] 2 W (x, y)W (y, z)W (z, x)f (x)f (y)f (z) dy dz, which is continuous with respect to the cut norm, models that are continuous under this topology-such as the GNN and continuous GGNN-are aligned with the task at the level of continuity. Our proposed continuous GGNN, which is provably transferable and possibly more expressive than the GNN, achieves the best performance. Although GGNN is not transferable under the cut norm, it is transferable under a weaker topology (see Appendix G.3), enabling it to perform reasonably well.
In contrast, the 2-IGN model, even after proper normalization, exhibits divergent outputs for larger graph sizes, indicating a lack of compatibility with the task.
Finally, we remark that the expressive power of various GNN architectures with respect to homomorphism densities has been extensively studied. Prior work has shown that common GNNs-including those considered in this study-are generally unable to express high-degree homomorphism densities [14,35]. However, our results demonstrate that GNNs can still achieve strong performance on this task when evaluated over certain large parametric families of random graph models. This does not contradict prior theoretical findings, as our results pertain to an average-case evaluation, while the negative results in the literature are established in the worst-case setting.
this section cite: ['b89', 'b13', 'b34']

Section: I.3 Size generalization on 3D point clouds
For our final set of experiments, we follow the setup of Section 7.2 in [5], which uses ModelNet10. We select 80 point clouds from Class 2 (chair) and 80 from Class 7 (sofa), and split them into 40 training and 40 testing samples per class. Each dataset has 40 × 40 cross-class pairs. The objective is to learn the third lower bound of the Gromov-Wasserstein distance in [60] (Definition 6.3). We prove in Appendix I.4 that it is continuous with respect to the Wasserstein p-distance.
Unlike [5], which downsampled all point clouds to 100 points, we focus on size generalization: training is done on n train = 20, and testing is done on n test ∈ {20, 100, 200, 300, 500}. We compare the size generalization of 3 models: unnormalized SVD-DS, (normalized) SVD-DS and normalized DS-CI. For each pair of inputs V, V ′ , we predict the GW lower bound via:
GW(V, V ′ ) = a∥W (f (V ) -f (V ′ ))∥ 2 + b,
where f : R n×k → R t is the S n , O(k)-invariant model, t = 10, and W ∈ R t×t , a, b ∈ R are learnable. All DeepSet components (σ, ϕ) are fully connected ReLU networks. We choose the number of layers and hidden dimensions such that each model has approximately 2k parameters to ensure a fair comparison. Training is performed by minimizing the MSE loss using AdamW with an initial learning rate of 0.01 and weight decay of 0.1. Each model is trained for 3000 epochs. Training a single run takes less than 3 minutes.
The experiment results are summarized in Figure 10. Normalized DS-CI and normalized SVD-DS, both aligned with the target at the continuity level, achieve good performance. While normalized SVD-DS underperforms compared to normalized DS-CI, it offers superior time and memory efficiency.
50 200 500 1000 2000 Test graph size (n) 10 -6 10 -5 Test MSE Normalized 2-IGN (incompatible) GNN (transferable) GGNN (compatible, not transferable) Continuous GGNN (transferable) (a) 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Target 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Prediction Model output on graph size ntest = 50 Normalized 2-IGN (incompatible) GNN (transferable) GGNN (compatible, not transferable) Continuous GGNN (transferable) 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Target 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Prediction Model output on graph size ntest = 2000 Normalized 2-IGN (incompatible) GNN (transferable) GGNN (compatible, not transferable) Continuous GGNN (transferable) (b) 50 200 500 1000 2000 Test graph size (n) 10 -5 Test MSE Normalized 2-IGN (incompatible) GNN (transferable) GGNN (compatible, not transferable) Continuous GGNN (transferable) (c) 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Target 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Prediction Model output on graph size ntest = 50 Normalized 2-IGN (incompatible) GNN (transferable) GGNN (compatible, not transferable) Continuous GGNN (transferable) 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Target 0.00 0.01 0.02 0.03 0.04 0.05 0.06 Prediction Model output on graph size ntest = 2000 Normalized 2-IGN (incompatible) GNN (transferable) GGNN (compatible, not transferable) Continuous GGNN (transferable) (d) Figure 9: GNN size generalization results on weighted triangle density. Figures (a)-(b) show the results on fully connected weighted graphs (the first data generation procedure), and Figures (c)-(d) show the results on simple graphs sampled from SBM (the second data generation procedure). Figures (a) and (c) display Test MSE vs. test graph size. The solid line denotes the mean, and the error bar extend from the 20% to 80% percentile test MSE over 10 randomly initialized trainings. Continuous GGNN performs the best for both random graph models. Figures (b) and (d) display Test-set predictions vs. ground truth. The middle columns displays results for graph size ntest = ntrain = 50, while the right-most column displays results for ntest = 2000 and ntrain = 50. 20 100 200 300 500 Test pointcloud sizes (n) 10 -2 10 -1 10 0 10 1 10 2 Test MSE Unnormalized SVD-DS (incompatible) Normalized SVD-DS (transferable) Normalized DS-CI (approximately transferable) (a) 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 Target 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 Prediction Model output on pointcloud size ntest = 20 Unnormalized SVD-DS (incompatible) Normalized SVD-DS (transferable) Normalized DS-CI (approximately transferable) 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75
this section cite: ['b4', 'b59', 'b4']

Section: I.4 Continuity of Gromov-Wasserstein distance and its third lower bound
The following is based on [60], though these continuity results are not stated there. Let µ ∈ P p (R k ) be a probability measure, and associate with it the "metric measure space" (X , d X , µ) where X = supp(µ) and d X (x, y) = ∥x -y∥ p . Given two such measures µ, ν ∈ P p (R k ), the
this section cite: ['b59']

Section: J Limitations of this work
This work provides a theoretical framework for transferability based on consistent sequences. It applies to several machine learning models that use a fixed number of parameters to define functions on any-dimensional inputs. However, this theory does not capture all possible ways for inputs to grow in dimension. In particular, it does not capture settings where there is no limiting space containing all finite-sized inputs, and where such inputs can be compared. For example, how do we compare natural language inputs of different lengths to each other? Furthermore, while we believe our framework may extend to settings such as images (with varying resolutions and sizes), partial differential equations (across different scales), and sequences (of varying lengths), we did not explore these directions. We leave these investigations for future work. Finally, as discussed briefly in the related work, we do not consider the expressive power of neural networks on the limiting space. If the target function is not expressible, mere alignment in terms of compatibility and continuity-as discussed in Section 4-is insufficient to ensure good performance. Studying universal approximation theory on the limiting space is an important and promising direction for future research.
this section cite: []

Section: 
Following this result, we can directly apply Propositions D.4 and D.5, along with the convergence rates described in Appendix D.3, which immediately yields the following transferability result.
this section cite: []

Section: Corollary G.4 (Transferability of MPNN).
For MPNNs satisfying the assumptions in Corollary G.2, we have the following transferability results. 1. (Uniform grid sampling) Let (A n , X n ) ∈ R n×n sym × R n×d be a sequence of graph signals sampled from the same graphon signal (W, f ) via the "uniform grid" sampling scheme, i.e., taking (A n ) ij = W i-1 n , j-1 n , (X n ) i: = f i-1 n for all i, j ∈ [n]. Suppose W : [0, 1] 2 → [0, 1] and f : [0, 1] → R d are both Lipschitz. Then, ∥[MPNN n (A n , X n )] -[MPNN m (A m , X m )]∥ 2 ≲ n -1 + m -1 .
this section cite: []

Section: (Graphon sampling)
Let (A n , X n ) ∈ R n×n
sym × R n×d be a sequence of graph signals sampled from the same graphon signal (W, f ) via the "graphon" sampling scheme, i.e. taking (A n ) ij ∼ Ber(W (x i , x j )), (X n ) i: = f (x i ) for all i, j ∈ [n], where x 1 , . . . , x n are sampled i.i.d. from Unif([0, 1]). Suppose W : [0, 1] 2 → [0, 1] is symmetric and measurable, and f ∈ L ∞ ([0, 1], R d ). Then,
E d 2 ([MPNN n (A n , X n )], [MPNN m (A m , X m )]) ≲ (log n) -1/4 + (log m) -1/4 .
The first part of the previous corollary recovers the transferability results in [70], yielding an improved convergence rate of O(n -1 ) and thus strengthening the previously established bounds of O(n -1/2 ). The second part of the corollary resembles the setting considered in [41], although the random sampling scheme used there operates at a different sparsity level, with A ij ∼ Ber(α n W (x i , x j )) and α n ∼ log n n . As a result, our result is not directly comparable.
this section cite: ['b69', 'b40']

Section: G.3 Constructing new transferable GNNs: GGNN and continuous GGNN
Background: Invariant Graph Networks (IGN). Invariant Graph Networks (IGN) [56] are a class of GNN architectures that alternate between linear S n -equivariant layers and nonlinearities. They follow a design paradigm that differs fundamentally from MPNNs. Specifically, a D-layer 2-IGN parameterizes an S n -equivariant function (R n ) ⊗2 → (R n ) ⊗2 as a composition (1)  n • W (1)  n , where for each i we have the following.
W (D) n • ρ (D-1) n • • • • • ρ
• The linear maps
W (i) n : ((R n ) ⊗2 ) ⊕di ∼ = R n 2 ×di → ((R n ) ⊗2 ) ⊕di+1 ∼ = R n 2 ×di+1 arer S n - equivariant.
Here, d i denotes the number of feature channels. [56] provides a parameterization of W (i) n as a linear combination of basis maps: In the special case where d i = d i+1 = 1, the linear layer W (i) n can be written as a linear combination of 17 basis functions (two of them are biases), where the coefficients α, β are the learnable parameters: (19) For general d i , d i+1 , the number of basis terms becomes 17d i d i+1 .
W (i) n (A) = α 1 A + α 2 A ⊤ +
• The activations ρ
(i) n : ((R n ) ⊗2 ) ⊕di+1 ∼ = R n 2 ×di+1 → ((R n ) ⊗2 ) ⊕di+1 ∼ = R n 2 ×di+1
apply a nonlinearity (e.g., ReLU) entry-wise.
To improve expressivity, [56] proposed extending the architecture to use higher-order tensors in the intermediate layers. When the maximum tensor order is k, the architecture is referred to as a k-IGN. While this is theoretically tractable, due to the high memory cost and implementation challenges Analogous to the case of sets, we can identify each matrix X ∈ R n×k with a step function f X : [0, 1] → R k , thereby interpreting V ∞ as the space of step functions with discontinuities at rational points Q. We also view the orbit of X as an empirical probability measure 1 n n i=1 δ Xi: . Equivalently, this identifies the orbit of a step function f ∈ V ∞ with µ f = Law(f (T )) where T ∼ Unif[0, 1].
The orthogonal group O(k) acts on probability measures via push-forward: for g ∈ O(k) and a measure µ, the action is given by g • µ = g # µ, where g # µ(B) = µ(g -1 (B)) for all measurable sets B ⊆ R k . The orbit space of V ∞ can be identified with the orbit space of empirical probability measures on R k under the action of O(k).
this section cite: ['b55', 'b55', 'b55']

Section: Norm on V P
dup . Consider Euclidean norm ∥ • ∥ 2 on R k which corresponds to the inner product preserved by elements of O(k). We equip each V n with the normalized ℓ p norm:
∥X∥ p = 1 n n i=1 ∥X i: ∥ p 2 1/p p ∈ [1, ∞) max n i=1 ∥X i: ∥ 2 p = ∞
By Proposition C.8, it is straightforward to verify that this norm extends naturally to V ∞ , and that the limit space in this case can be identified with
V ∞ = L p ([0, 1]; R k ) of functions f : [0, 1] → R k with norm ∥f ∥ = 1 0 ∥f (t)∥ p 2 dt 1/p < ∞.
Analogous to the case of sets, the corresponding space of orbit closures can be identified with the space of orbit closures of probability measures on R k (with finite p-th moments) under the O(k)-actions. The symmetrized metric is given by:
d p (f, g) = inf g∈O(k) W p (g • µ f , µ g ) for f, g ∈ V ∞ ,(23)
where W p is the Wasserstein p-distance with respect to the ℓ 2 -norm on R k .
this section cite: []

Section: H.2 DeepSet for Conjugation Invariance (DS-CI)
The DS-CI model [5] is given by
DS-CI n : R n×k → R V → MLP c DeepSet (1) f d j (V V ⊤ ) j=1,...,n , DeepSet (2) f o ℓ (V V ⊤ ) ℓ=1,...,n(n-1)/2 , MLP (3) f * (V V ⊤ ) ,
where for symmetric matrix X ∈ R n×n sym , the invariant features are given by f d j (X) = the j-th largest of the numbers X 11 , . . . , X nn , by f o ℓ (X) = the ℓ-th largest of the numbers X ij , 1 ≤ i < j ≤ n, and by
f * (X) = i̸ =j X ii X ij .
We define normalized DS-CI with appropriate normalization: replacing DeepSet (1) , DeepSet (2) with their normalized version (i.e. replacing the sum aggregation with the mean aggregation), and
replacing f * (X) with f * (X) = 1 n(n-1) i̸ =j X ii X ij .
We denote the sequence of functions of normalized DS-CI by (DS-CI n ).
this section cite: ['b4']

Section: Transferability analysis of normalized DS-CI.
Normalized DS-CI is not compatible with respect to V P dup . To see this, observe that under duplication, we have
(V ⊗ 1 N/n )(V ⊗ 1 N/n ) ⊤ = (V V ⊤ ) ⊗ (1 N/n 1 ⊤ N/n ), Therefore, diagonal elements of V V ⊤ become off-diagonal elements in (V V ⊤ ) ⊗ (1 N/n 1 ⊤ N/n ), so DeepSet (2) f o ℓ (V ⊗ 1 N/n )(V ⊗ 1 N/n ) ⊤ N (N -1)/2 ℓ=1 ̸ = DeepSet (2) f o ℓ (V V ⊤ ) n(n-1)/2 ℓ=1 , f * (V ⊗ 1 N/n )(V ⊗ 1 N/n ) ⊤ ̸ = f * (V V ⊤ ).
this section cite: []

Section: Transferability plots.
The numerical experiments illustrating the transferability of SVD-DS and normalized and compatible DS-CI is shown in Figure 6.
10 1 10 2 10 3 Pointcloud size n -0.096 -0.094 -0.092 -0.090 fn(xn) (a) Normalized SVD-DS (transferable) 10 1 10 2 10 3 Pointcloud size n 0.25 0.26 0.27 0.28 0.29 fn(xn) Normalized DS-CI (approximately transferable) Compatible DS-CI (transferable) (b) Normalized DS-CI (approximately transferable) and
Compatible DS-CI (transferable) [5] Figure 6: Transferability of invariant models on point clouds with respect to (V P dup , ∥ • ∥p). The plot shows outputs of untrained, randomly initialized models for a sequence of point clouds Xn ∈ R n×k , where each point is sampled i.i.d. from N (0, I k ). The error bars extend from the mean to ± one standard deviation over 100 random samples. The figure shows that transferable models generate convergent outputs. Figure (b) shows the asymptotic equivalence between the normalized DS-CI and compatible DS-CI as proved in Lemma H.2.
this section cite: ['b4']

Section: I Size generalization experiments: details from Section 6
In this section, we provide details of our size generalization experiments. All experiments were implemented using the PyTorch framework and trained on a single NVIDIA A5000 GPU. Specific training and model configurations are provided in the descriptions of the individual experiments. For all experiments, the training dataset consists of inputs with a fixed, small dimension n train . For evaluation, we use a series of test datasets where the input dimension n test is progressively larger than n train .
this section cite: []

Section: I.1 Size generalization on sets
We consider two any-dimensional learning tasks on sets, where the target functions have different properties, so that different models are expected to perform better. In both experiments, we compare the size generalization of three models: DeepSet, normalized DeepSet, and PointNet as analyzed in Appendix F.2. The maps σ and ρ in the model are parametrized by three fully connected layers with a hidden dimension of 50 and ReLU activation functions. Training was performed by minimizing the MSE loss using AdamW with an initial learning rate of 0.001 and a weight decay of 0.1. The learning rate was halved if the validation loss did not improve for 50 consecutive epochs. Each model was trained for 1000 epochs, with each run taking less than three minutes to complete.
this section cite: []

Section: I.1.1 Experiment 1: Population statistics
We adopt the experimental setup from [91, Section 4.1.1], which comprises four distinct tasks on population statistics. In all four tasks, the datasets consist of sets where each set contains i.i.d. samples from a distribution µ, where µ itself is sampled from a parameterized distribution family. The objective is to learn either the entropy or the mutual information of the distribution µ.
While the original experiment in [91] focused on training and testing with set sizes n train = n test = [300, 500], we instead evaluate size generalization. During the training stage, the dataset consists of N = 2048 sets, each of size n train = 500. This dataset is randomly split into training, validation, and test data with proportions 50%, 25%, and 25%, respectively. During the evaluation stage, the trained model is tested on a sequence of datasets with set sizes n test ∈ {500, 1000, 1500, . . . , 4500}, each consisting of N = 512 sets. The descriptions of the four tasks, as originally presented in [91], are provided below: Rotation: Generate N datasets of size M from N (0, R(α)ΣR(α) T ) for random Σ and α ∈ [0, π]. The goal is to learn the marginal entropy of the first dimension.
Correlation: Generate sets from N (0, [Σ, αΣ; αΣ, Σ]) for random Σ and α ∈ (-1, 1). The goal is to learn the mutual information between the first 16 and last 16 dimensions.
Rank 1: Generate sets from N (0, I + λvv T ) for random v ∈ R 32 and λ ∈ (0, 1). The goal is to learn the mutual information.
Random: Generate sets from N (0, Σ) for random 32 × 32 covariance matrices Σ. The goal is to learn the mutual information.
For all tasks, the target functions are scalar functions on the underlying probability measure µ, and are continuous with respect to the Wasserstein p-distance. Based on Appendix F.2.1, normalized DeepSet is well-aligned with the task from a continuity perspective and PointNet aligns from a compatibility perspective, while unnormalized DeepSet lacks alignment altogether. The results are summarized in Figure 7, showing that stronger task-model alignment improves both in-distribution and size-generalization performance.
this section cite: []

Section: I.1.2 Experiment 2: Maximal distance from the origin
We consider the following data and task: each dataset consists of sets where each set contains n two-dimensional points sampled as follows. First, a center is sampled from N (0, I 2 ) and a radius is sampled from Unif([0, 1]), which together define a circle. The set then consists of n points sampled uniformly along the circumference. The goal is to learn the maximum Euclidean norm among the points in each set. Equivalently, our goal is to learn the so-called Hausdorff distance d H ({0}, X) = sup x∈X ∥x∥; see (11). Hence, the target function depends only on the support of the point cloud and is continuous with respect to the Hausdorff distance.
Once more, we evaluate size generalization. During the training stage, the dataset consists of N = 5000 sets, each of size n train = 20. The dataset is randomly split into training, validation, and test data with proportions 80%, 10%, and 10%, respectively. During the evaluation stage, the trained model is tested on a sequence of datasets with set sizes n test ∈ {20, 40, . . . , 200}, each consisting of N = 1000 sets. For this task, PointNet aligns from a continuity perspective, normalized DeepSet from a compatibility perspective, and DeepSet does not align with the learning task. The results are summarized in Figure 8, showing that the model performance improves with task-model alignment.
this section cite: ['b10']

Section: I.2 Size generalization on graphs
For graphs, the task is to learn a homomorphism density of degree three. To formally describe the task, we first describe its inputs. We consider a dataset consisting of N attributed graphs (A, x) generated according to the following two procedures (the first is described and reported in the main paper):
1. Each graph is a fully connected weighted graph whose adjacency matrix has entries A ij = A ji i.i.d. ∼ Unif([0, 1]) for i ≤ j, and node features x i i.i.d. ∼ Unif([0, 1]). 2. First, sample the number of clusters K uniformly from {10, . . . , 20}, and construct a K × K symmetric probability matrix P with entries sampled uniformly from [0, 1]. The resulting stochastic block model (SBM) is used to generate an undirected, simple graph with edges sampled as A ij = A ji i.i.d. ∼ Ber(P zi,zj ) for i ≤ j, where z i i.i.d.
∼ Unif({1, . . . , K}) are cluster assignments. Node features are given by x i = γ zi , where γ ∈ R K with i.i.d. entries Unif([0, 1]).
The task is to learn the rooted, signal-weighted homomorphism density of degree three sending
R n×n sym × R n → R n , (A, x) → y, via y i = 1 n 2 j,k∈[n] A ij A jk A ki x i x j x k .
Note that when x = 1, y i corresponds to a normalized count of triangles centered at node i. Thus, this can be interpreted as a signal-weighted triangle density. This formulation is related to the signalweighted homomorphism density studied in [35], which generalizes the notion of homomorphism density extensively studied in graphon theory [54].
We conduct experiments to evaluate the size generalization performance of the following models: the GNN from [70], the normalized 2-IGN [56], and our proposed GGNN and continuous GGNN.
ReLU is used as the entry-wise activation function in all models. We choose the number of layers Gromov-Wasserstein distance between their associated metric spaces is defined by
D p (µ, ν) = inf π∈M(µ,ν) ∥Γ X,Y ∥ L p (π⊗π) (32
) = inf π∈M(µ,ν) ∥x -x ′ ∥ p -∥y -y ′ ∥ p p dπ(x, y) dπ(x ′ , y ′ ) 1/p ,(33)
where Γ X,Y (x, y, x ′ , y ′ ) = ∥xx ′ ∥ p -∥yy ′ ∥ p and M(µ, ν) is the set of couplings between µ and ν. The G-W distance admits the following lower bound [60, Def. 6.3].
D p ≥ TLB p (µ, ν) = inf π∈M(µ,ν) ∥Ω µ,ν ∥ L p (π) ,
where Ω µ,ν (x, y) = inf
π ′ ∈M(µ,ν) ∥Γ(x, y, •, •)∥ L p (π ′ ) .(34)
We aim to show that both D p and TLB p are continuous with respect to the Wasserstein-p metric on P p . More precisely, the following Lipschitz bounds hold.
Proposition I.1. Let µ, ν, µ ′ , ν ′ ∈ P p (R k ).
Then
|D p (µ, ν) -D p (µ ′ , ν ′ )| ≤ W p (µ, µ ′ ) + W p (ν, ν ′ ).(35)
Proof. By the triangle inequality for D p , which holds by [60, Thm. 5.1(a)], we have
D p (µ, ν) ≤ D p (µ, µ ′ ) + D p (µ ′ , ν ′ ) + D p (ν, ν ′ ).
By [60, Thm. 5.1(d)], we further have D p (µ, µ ′ ) ≤ W p (µ, µ ′ ) and similarly for D p (ν, ν ′ ). Combining these inequalities and interchanging the roles of (µ, ν) and (µ ′ , ν ′ ), we get the claim.
The above proof only used the triangle inequality and the bound D p ≤ W p for the G-W distance.
Since TLB p ≤ D p ≤ W p , the latter property also holds for TLB. Thus, it suffices to prove TLB p satisfies the triangle inequality, hence is similarly Lipschitz in W p .
Lemma I.2 (Triangle inequality for Ω µ,ν ). Let µ, ν, ξ ∈ P p (R k ). We have Ω µ,ν (x, y) ≤ Ω µ,ξ (x, z)+ Ω ξ,ν (z, y) for all x, y, z in the relevant supports. Furthermore, we have TLB p (µ, ν) ≤ TLB p (µ, ξ)+ TLB p (ξ, ν).
Proof. Note that the usual triangle inequality for ∥ • ∥ p gives Γ(x, y, x ′ , y ′ ) ≤ Γ(x, z, x ′ , z ′ ) + Γ(z, y, z ′ , y ′ ) for any x, y, z, x ′ , y ′ , z ′ ∈ R k . For any couplings π 1 ∈ M(µ, ξ) and π 2 ∈ M(ξ, ν), the Gluing Lemma [82, Lemma 7.6] guarantees the existence of a coupling π ∈ M(µ, ξ, ν) whose corresponding marginals are π 1 , π 2 . Let π 3 ∈ M(µ, ν) be the marginal of π on its first and third coordinates. Then
Ω µ,ν (x, y) p ≤ ∥Γ(x, y, •, •)∥ p L p (π3) = ∥Γ(x, y, •, •)∥ p L p (π) ≤ ∥Γ(x, z, •, •) + Γ(z, y, •, •)∥ L p (π) ≤ ∥Γ(x, z, •, •)∥ L p (π1) + ∥Γ(z, y, •, •)∥ L p (π2) .
Since this holds for all couplings π 1 , π 2 as above, we obtain the first claim.
The second claim is proved analogously. For couplings π 1 , π 2 , π 3 , π as above, we have
TLB p (µ, ν) ≤ ∥Ω µ,ν ∥ L p (π3) = ∥Ω µ,ν ∥ L p (π) ≤ ∥Ω µ,ξ +Ω ξ,ν ∥ L p (π) ≤ ∥Ω µ,ξ ∥ L p (π1) +∥Ω ξ,ν ∥ L p (π2) ,
and taking infimums over π 1 , π 2 completes the proof.
Proposition I.3. Let µ, ν, µ ′ , ν ′ ∈ P p (R k ).
Then
|TLB p (µ, ν) -TLB p (µ ′ , ν ′ )| ≤ W p (µ, µ ′ ) + W p (ν, ν ′ ).(36)
Proof. The proof is now identical to that of Proposition I.1.
The above argument generalizes to measures on different abstract metric spaces, we did not use the fact that all measures involved were over R k .
this section cite: ['b34', 'b53', 'b69', 'b55']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: [NA] Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: The limitations of the work are discussed in Appendix J.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: All proofs are in the supplementary material due to space constraints. All the assumptions are explicitly stated.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: The main contribution is theoretical. Experiments are provided to illustrate the results. In the Appendix I, we provide all relevant details for reproduction, including the data, training and evaluation procedures. Also, the code is available via an anonymized GitHub repository.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: 5.
Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code with reproduction instructions is provided via an anonymized GitHub repository.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: All these experiment details are provided in Appendix I. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: The transferability experiments report the mean among 100 random inputs with error bars indicating one standard deviation. The size generalization experiments report min/max results over 10 initialized runs, and evaluations are conducted on datasets with 1000 samples. Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: This information is reported in Appendix I. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research conforms the code of ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10. Broader impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This is a theory paper. We cannot think of any direct societal impacts.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: The paper poses no such risk.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: All resources are properly cited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] Justification: We provide the code for reproducing our experiments through an anonymized GitHub repository.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification:[NA] Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Learning theory from first principles Year: (2024)
Ref_id:b1 Title: The logical expressiveness of graph neural networks Year: (2020-04)
Ref_id:b2 Title: Rademacher and gaussian complexities: Risk bounds and structural results Year: (2002-11)
Ref_id:b3 Title: Size-invariant graph representations for graph classification extrapolations Year: (2021)
Ref_id:b4 Title: Learning functions on symmetric matrices and point clouds via lightweight invariant features Year: (2024)
Ref_id:b5 Title: Machine learning and invariant theory Year: (2023)
Ref_id:b6 Title: Fine-grained expressivity of graph neural networks Year: (2024)
Ref_id:b7 Title: Convergent sequences of dense graphs I: Subgraph frequencies, metric properties and testing Year: (2008)
Ref_id:b8 Title: On the representation power of set pooling networks Year: (2021)
Ref_id:b9 Title: Towards a bilipschitz invariant theory Year: (2024)
Ref_id:b10 Title: Convergence of invariant graph networks Year: (2022)
Ref_id:b11 Title: Combinatorial optimization and reasoning with graph neural networks Year: (2023)
Ref_id:b12 Title: How unique are U and V in the singular value decomposition A = U ΣV † ? Mathematics Stack Exchange Year: ()
Ref_id:b13 Title: Can graph neural networks count substructures? Advances in neural information processing systems Year: (2020)
Ref_id:b14 Title: Representation stability in cohomology and asymptotics for families of varieties over finite fields Year: (2013)
Ref_id:b15 Title: Representation theory and homological stability Year: (2013)
Ref_id:b16 Title: Group equivariant convolutional networks Year: (2016)
Ref_id:b17 Title: Convergence of message passing graph neural networks with generic aggregation on random graphs Year: (2023)
Ref_id:b18 Title: Stochastic deep networks Year: (2019)
Ref_id:b19 Title: Invariant kernels: Rank stabilization and generalization across dimensions Year: (2025)
Ref_id:b20 Title: Foundations of Modern Analysis Year: (1969)
Ref_id:b21 Title: Graph neural tangent kernel: Fusing graph neural networks with graph kernels Year: (2019)
Ref_id:b22 Title: Equivariant frames and the impossibility of continuous canonicalization Year: (2024-07)
Ref_id:b23 Title: Vc dimension of graph neural networks with pfaffian activation functions Year: (2025)
Ref_id:b24 Title: Efficient and accurate estimation of lipschitz constants for deep neural networks Year: (2019)
Ref_id:b25 Title: Convergence of the empirical measure in expected wasserstein distance: non-asymptotic explicit bounds in R d . ESAIM: Probability and Statistics Year: (2023)
Ref_id:b26 Title: Stability properties of graph neural networks Year: (2020)
Ref_id:b27 Title: Generalization and representational limits of graph neural networks Year: (2020)
Ref_id:b28 Title: Euclidean neural networks Year: (2022)
Ref_id:b29 Title: A mathematical perspective on transformers Year: (2025)
Ref_id:b30 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b31 Title: Matrix computations Year: (2013)
Ref_id:b32 Title: Equivariant geometric convolutions for emulation of dynamical systems Year: (2023)
Ref_id:b33 Title: The logic of graph neural networks Year: (2021)
Ref_id:b34 Title: Higher-order graphon neural networks: Approximation and cut distance Year: (2025)
Ref_id:b35 Title: A short tutorial on the Weisfeiler-Lehman test and its variants Year: (2021)
Ref_id:b36 Title: Graphons, cut norm and distance, couplings and rearrangements Year: (2013)
Ref_id:b37 Title: Expressivity-preserving GNN simulation Year: (2024)
Ref_id:b38 Title: Generalization in graph neural networks: Improved pac-bayesian bounds on graph diffusion Year: (2023)
Ref_id:b39 Title: Yoshua Bengio, and Siamak Ravanbakhsh. Equivariance with learned canonicalization functions Year: (2023)
Ref_id:b40 Title: Convergence and stability of graph convolutional networks on large random graphs Year: (2020)
Ref_id:b41 Title: Is uniform expressivity too restrictive? towards efficient expressivity of GNNs Year: (2024)
Ref_id:b42 Title: The lipschitz constant of self-attention Year: (2021)
Ref_id:b43 Title: N-body networks: a covariant hierarchical neural network architecture for learning atomic potentials Year: (2018)
Ref_id:b44 Title: On the generalization of equivariance and convolution in neural networks to the action of compact groups Year: (2018)
Ref_id:b45 Title: Limits, approximation and size transferability for GNNs on sparse graphs via graphops Year: (2024)
Ref_id:b46 Title: A graphon-signal analysis of graph neural networks Year: (2024)
Ref_id:b47 Title: Transferability of spectral graph convolutional neural networks Year: (2021)
Ref_id:b48 Title: Free descriptions of convex sets Year: (2023)
Ref_id:b49 Title: Any-dimensional equivariant neural networks Year: (2024)
Ref_id:b50 Title: Fourier neural operator for parametric partial differential equations Year: (2020)
Ref_id:b51 Title: A PAC-bayesian approach to generalization bounds for graph neural networks Year: (2020)
Ref_id:b52 Title: Towards graph foundation models: A survey and beyond Year: (2023)
Ref_id:b53 Title: Large networks and graph limits Year: (2012)
Ref_id:b54 Title: Generalization bounds for graph convolutional neural networks via rademacher complexity Year: (2021)
Ref_id:b55 Title: Invariant and equivariant graph networks Year: (2018)
Ref_id:b56 Title: Generalization bounds for message passing networks on mixture of graphons Year: (2024)
Ref_id:b57 Title: Transferability of graph neural networks: an extended graphon approach Year: (2023)
Ref_id:b58 Title: Generalization analysis of message passing neural networks on large random graphs Year: (2022)
Ref_id:b59 Title: Gromov-Wasserstein distances and the metric approach to object matching Year: (2011)
Ref_id:b60 Title: Symmetric gauge functions and unitarily invariant norms Year: (1960)
Ref_id:b61 Title: Wl meet vc Year: (2023)
Ref_id:b62 Title: An invitation to statistics in Wasserstein space Year: (2020)
Ref_id:b63 Title: Approximation capability of neural networks on spaces of probability measures and tree-structured domains Year: (2019)
Ref_id:b64 Title: Computational optimal transport: With applications to data science Year: (2019)
Ref_id:b65 Title: Pointnet: Deep learning on point sets for 3d classification and segmentation Year: (2017)
Ref_id:b66 Title: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations Year: (2019)
Ref_id:b67 Title: Generalization, expressivity, and universality of graph neural networks on attributed graphs Year: (2025)
Ref_id:b68 Title: Some might say all you need is sum Year: (2023)
Ref_id:b69 Title: Graphon neural networks and the transferability of graph neural networks Year: (2020)
Ref_id:b70 Title: Transferability properties of graph neural networks Year: (2023)
Ref_id:b71 Title: Graph neural networks: Architectures, stability, and transferability Year: (2021)
Ref_id:b72 Title: A spectral analysis of graph neural networks on dense and sparse graphs Year: (2024)
Ref_id:b73 Title: Sinkformers: Transformers with doubly stochastic attention Year: (2022)
Ref_id:b74 Title: The Vapnik-Chervonenkis dimension of graph and recursive neural networks Year: (2018)
Ref_id:b75 Title: Understanding machine learning: From theory to algorithms Year: (2014)
Ref_id:b76 Title: A course on Borel sets Year: (1998)
Ref_id:b77 Title: Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds Year: (2018)
Ref_id:b78 Title: Covered forest: Fine-grained generalization analysis of graph neural networks Year: (2024)
Ref_id:b79 Title: Survey on generalization theory for graph neural networks Year: (2025)
Ref_id:b80 Title: Graph neural networks and non-commuting operators Year: (2024)
Ref_id:b81 Title: Topics in optimal transportation Year: (2003)
Ref_id:b82 Title: Scalars are universal: Equivariant machine learning, structured like classical physics Year: (2021)
Ref_id:b83 Title: Dimensionless machine learning: Imposing exact units equivariance Year: (2023)
Ref_id:b84 Title: Lipschitz regularity of deep neural networks: analysis and efficient estimation Year: (2018)
Ref_id:b85 Title: Aristide Baratin, and Remi Tachet des Combes. A mathematical theory of attention Year: (2020)
Ref_id:b86 Title: 3D ShapeNets: A deep representation for volumetric shapes Year: (2015)
Ref_id:b87 Title: Robustness and generalization Year: (2012)
Ref_id:b88 Title: From local structures to size generalization in graph neural networks Year: (2021)
Ref_id:b89 Title: A useful variant of the Davis-Kahan theorem for statisticians Year: (2014)
