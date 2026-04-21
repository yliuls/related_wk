Title: Spectral Graph Neural Networks are Incomplete on Graphs with a Simple Spectrum
Abstract: Spectral features are widely incorporated within Graph Neural Networks (GNNs) to improve their expressive power, or their ability to distinguish among nonisomorphic graphs. One popular example is the usage of graph Laplacian eigenvectors for positional encoding in MPNNs and Graph Transformers. The expressive power of such Spectrally-enhanced GNNs (SGNNs) is usually evaluated via the k-WL graph isomorphism test hierarchy and homomorphism counting. Yet, these frameworks align poorly with the graph spectra, yielding limited insight into SGNNs' expressive power. In this paper, we leverage a well-studied paradigm of classifying graphs by their largest eigenvalue multiplicity to introduce an expressivity hierarchy for SGNNs. We then prove that many SGNNs are incomplete even on graphs with distinct eigenvalues. To mitigate this deficiency, we adapt rotation equivariant neural networks to the graph spectra setting, yielding equiEPNN, a novel SGNN that provably improves upon contemporary SGNNs' expressivity on simple spectrum graphs. We then demonstrate that equiEPNN achieves perfect eigenvector canonicalization on ZINC, and performs favorably on image classification on MNIST-Superpixel and graph property regression on ZINC, compared to leading spectral methods.

Section: Introduction
Graph Neural Networks (GNNs) have become a ubiquitous paradigm for learning on graph-structured data. The core principle of GNNs is to maintain a representation of each graph vertex and leverage the graph structure to iteratively refine each representation by its vertex's graph neighborhood [41]. To enhance the purview of the vertex's neighborhood, it is common to incorporate spectral features, such as Random Walk matrices, positional encoding, and graph distances, into the refinement operation of GNNs [9,1,45,51]. Such GNNs, which systematically incorporate spectral features within their representation refinement procedure, or Spectrally-enhanced GNNs (SGNNs) [52], have gained significant traction in the graph learning community, due to their reasonable complexity and empirical benefits [18,53,52,13].
Understanding the expressive power of GNNs provides researchers with a framework for comparing different models and identifying their deficiencies, often leading to improvements [15,32,35,16,49]. These frameworks ought to characterize which graphs the GNN can distinguish among, based on the GNNs' inner workings. For instance, the Weisfeiler-Leman (WL) test, which maintains and refines vertex representations similarly to Message Passing Neural Networks, a subclass of GNNs, completely determines which graphs these models can distinguish among [49].
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
this section cite: ['b40', 'b8', 'b0', 'b44', 'b50', 'b51', 'b17', 'b52', 'b51', 'b12', 'b14', 'b31', 'b34', 'b15', 'b48', 'b48']

Section: 1-WL

this section cite: []

Section: EPNN equiEPNN
⊐ ⊐ Figure 1: Hierarchy of 1-WL test variants. The arrows with ⊐ indicate strict inclusion relationships, meaning each variant can distinguish all graphs that the previous one can, plus additional graphs. Standard 1-WL is the least discriminative, while equiEPNN achieves the highest discriminative power, by incorporating both spectral invariant and equivariant refinement.
To study the expressive power of SGNNs, recent papers [52,13] proposed a spectrally enhanced GNN, called Eigenspace Projection GNN (EPNN), which generalizes many popular spectral graph neural networks, and analyze its expressivity via WL tests and homomorphism counting. This comparison is valuable in comparing the expressivity of SGNNs to that of their combinatorial GNN counterparts. Yet, this analysis does not yield insight into the role of the graph spectra in the distinguishing ability of these GNNs.
To address this gap, we propose analyzing the expressive power of SGNNs via Spectral Graph Theory, and in particular via the maximal eigenvalue multiplicity of a graph. As isomorphism of graphs with bounded eigenvalue multiplicity can be determined in polynomial time, with the complexity depending exponentially on the eigenvalue multiplicity [2], this notion imposes a natural hierarchical classification of graphs, and SGNNs can potentially be complete on these graph classes, making this hierarchy a viable method for assessing their expressive power.
Our analysis centers around the expressivity of EPNN on graphs with distinct eigenvalues. This model is at least as expressive as many commonly used SGNNs [52], making an upper bound on the expressivity of EPNN applicable to these models. Surprisingly, we find that EPNN is incomplete even on the class of graphs with distinct eigenvalues. On the positive side, EPNN achieves completeness on simple spectrum graphs whose eigenvectors exhibit certain sparsity patterns. Based on these theoretical insights, we propose equiEPNN, inspired by equivariant neural networks for point clouds, which attains provably improved expressivity on graphs with distinct eigenvalues.
Our main contributions are summarized as follows:
1. We prove the incompleteness of EPNN (in Subsection 3.2) on graphs with a simple spectrum.
2. We formulate a guarantee on the completeness of EPNN on graphs with a simple spectrum based on sparsity patterns of the eigenvectors.
3. We introduce equiEPNN (in Section 3.3), a modified EPNN variant, which integrates Euclidean message passing into the feature refinement procedure.
4. We benchmark equiEPNN on the ZINC and MNIST-Superpixel datasets, yielding favorable performance in comparison with popular spectral methods. Furthermore, equiEPNN performs perfect eigenvector canonicalization on the ZINC dataset.
2 Related work
this section cite: ['b51', 'b12', 'b1', 'b51']

Section: Spectral invariant GNNs
An enhancement to MPNNs and Transformer-based models is to incorporate spectral distances such as Random Walk, resistance, and shortest-path distances within the message passing operation [26,50,33,12]. Zhang et al. [52] compare among spectral GNNs and the WL hierarchy by proving EPNN is strictly more powerful than 1-WL yet strictly less powerful than 3-WL. Despite the important result that 3-WL strictly bounds the expressive power of EPNN, the large expressivity gap between 1and 3-WL makes this determination difficult to conceptualize. Building on this work, Gai et al. [13] have characterized the expressive power of EPNN via graph homomorphism counting, showing spectral invariant GNNs can homomorphism-count a class of specific tree-like graphs. Despite providing a deeper understanding of EPNN's expressive power, it remains hard to conceptualize and propose more expressive models based on it.
this section cite: ['b25', 'b49', 'b32', 'b11', 'b51', 'b12']

Section: Spectral canonicalization methods
The eigenvectors of a graph are used as positional encoding to improve the expressive power of message-passing and as positional encoding for Transformer [38,21,31] based models. Yet, positional encoding has an inherent ambiguity problem. An eigenvector corresponding to a unique eigenvalue can be represented as itself or its negation [43]. Canonicalization methods [30,29] are used to address the ambiguity problems of eigenvectors, by choosing a unique representative for each eigenvector.
Ma et al. [29] have uncovered an inherent limitation of canonicalization methods that process each eigenspace separately, which is that they cannot canonicalize eigenvectors with nontrivial selfsymmetries. These models process each eigenbasis independently to obtain an orthogonal invariant and permutation-equivariant feature, and then use these features for downstream applications. Notable examples include SignNet and BasisNet [27], MAP [29] and OAP [30]. Ma et al. [30] have shown that these methods lose information when canonicalizing eigenvectors with self-symmetries, proving that the popular spectral invariant models SignNet and BasisNet are incomplete. In section 5.2, we provide a canonicalization scheme that bypasses this issue, and while not provable complete on all eigenvectors, empirically it canonicalizes all eigenvectors corresponding to distinct eigenvalues, in the ZINC [19] dataset.
this section cite: ['b37', 'b20', 'b30', 'b42', 'b29', 'b28', 'b28', 'b26', 'b28', 'b29', 'b29', 'b18']

Section: Expressivity on simple spectrum graphs
An early study on the connection between GNNs and spectral features of the underlying graph studied the expressive power of CGNs [47]. They have proven that linear graph convolutional neural networks (GCNs) can map a graph signal to any chosen target vector, if the graph has distinct eigenvalues. Yet, this graph signal is sampled randomly and thus is not equivariant to permutations of the graph nodes, which may lead to degraded generalization, see Bechler-Speicher et al. [5].
For more related work see Appendix C.
this section cite: ['b46', 'b4']

Section: Problem statement

this section cite: []

Section: Spectral graph decomposition
Graphs are typically represented by a matrix A ∈ R n×n , where the (i, j)-th entry of the matrix encodes the relationship between node i and j. This matrix could be the adjacency matrix, the normalized or un-normalized graph Laplacian, or a distance or Gram Matrix where the graph nodes have some underlying geometry.
A crucial principle in the design of graph neural networks is the notion of permutation invariance. Since graph nodes are not endowed with an intrinsic order, we would like to think of a matrix A and its conjugation P AP T by a permutation matrix P ∈ S n , as being equivalent. Graph neural networks respect this invariance constraint and produce a permutation-invariant function f satisfying f (A) = f (P AP T ). One popular method to design these functions exploits the eigendecomposition of the matrix A.
In the general case, we assume that A has an eigenbasis v (1) , . . . , v (n) of vectors of norm one, which corresponds to real eigenvalues λ 1 , . . . , λ n . This assumption holds when in the typical case where A is a symmetric matrix (e.g., adjacency and Laplacian matrices), and also often holds in other settings (e.g., Random Walk matrix). This endows an alternative representation of the matrix A with its own symmetries. Firstly, we note that each vector P v (q) will be an eigenvector of P AP T with the same eigenvalue λ q . Secondly, if v (q) is an eigenvector of norm one, then so is -v (q) . When the eigenvalues of f are pairwise distinct, then these are all the relevant ambiguities. This is referred to as the simple spectrum case. In the case of an eigenbasis of dimension k, the eigendecomposition ambiguity is defined by orthogonal transformations in O k . In this paper, we will focus on the simple spectrum case. In this case, we define sign-invariant functions as follows
this section cite: []

Section: Definition 1 (Sign Invariant functions).
For fixed natural n and K ≤ n, denote
V K simple = {(V, ⃗ λ) ∈ R n×K ⊕ R K | λ 1 > λ 2 > . . . > λ K }.
We say that F :
V K simple → R m is sign invariant if F (V, ⃗ λ) = F (P V S, ⃗ λ), ∀P ∈ S n , S ∈ {-1, 1} K
We note that in this definition, V represents a n × K matrix whose K columns represent the first K eigenvectors v (1) , . . . , v (K) of A, and the notation S ∈ {-1, 1} K means that S is a diagonal matrix whose diagonal is a vector in {-1, 1} K .
The notion of sign invariant function was first introduced in [27], and was later discussed in [29,30]. These papers discuss a collection of parametric functions F = {f θ (V, ⃗ λ) | θ ∈ Θ}, such that for all parameters θ the function f θ is sign invariant. To understand the expressiveness of these models, we formally define the notion of completeness on simple spectrum graphs.
this section cite: ['b26', 'b28', 'b29']

Section: Definition 2 (Sign Invariant Separation).
For K ≤ n, let F denote a collection of sign invariant functions defined on V K simple , and let D be a subset of V K simple . We say that F is complete on D if for any non-isomorphic pair (V, ⃗ λ) and (U, ⃗ η) in D, there exists a function f ∈ F such that
f (V, ⃗ λ) ̸ = f (U, ⃗ η).
Ideally, we would like F to be complete on all of the domain V K simple . If F is complete, then by applying it to eigendecompostions of graphs with simple spectrum, we will obtain models which can separate all graphs with simple spectrum, up to permutation equivalence. The goal of this paper is to understand whether existing sign-invariant functions are complete.
this section cite: []

Section: EPNN
We will focus on a large family of sign invariant functions named Eigenspace Projection GNNs (EPNN). This family of functions, introduced in Zhang et al. [52], was shown to generalize many spectral invariant methods such as Random Walk, resistance, and shortest-path distances [26,50,33,12]. This method is based on a message passing like mechanism, where the spectral information is encoded by using the projection onto eigenspaces as edge features. In the simple spectrum case, this method can be formulated as follows:
For a given eigendecomposition (V, ⃗ λ) ∈ V K simple , we we initialize a coloring for each 'node' i ∈ [n] by h
(0) i = V i ⊙ V i ,(1)
where
V i ≜ V i,: is the K dimensional vector [V i,:
(1), . . . , V i,: (K)] obtained by sampling all eigenvectors at the i-th node, and ⊙ denotes elementwise multiplication. Importantly, this initialization is sign-invariant: while the global sign of each eigenvector is ambiguous, the product of two elements of the same eigenvector is not.
We next iteratively refine the node features via the update rule:
h (t+1) i = UPDATE (t) h (t) i , ⃗ λ, (h (t) j , V i ⊙ V j ) | j = 1, . . . , n(2)
Here and throughout {•} denote multisets (multiplicities are allowed) and the multiset notation implies that UPDATE (t) is required to be invariant to the order of the elements in the multiset.
Finally, we apply a global pooling operation to obtain a final permutation invariant representation
h global = READOUT({h (T ) i | i = 1, . . . , n})(3)
Once UPDATE (t) and READOUT functions are determined, this procedure determines a function f (V, ⃗ λ) = h global which is sign-invariant as in Definition 1. The collection of all such functions obtained by all possible choices of UPDATE (t) and READOUT functions is denoted by F EPNN .
this section cite: ['b51', 'b25', 'b49', 'b32', 'b11']

Section: Equivariant EPNN
In [13], the authors suggest methods based on higher order WL tests to boost the expressive power of spectral message passing neural networks. The complexity of these methods is considerably higher than EPNN. In contrast, we will now suggest a method for increasing the expressive power of EPNN without significantly changing model complexity.
Our suggestions are based on constructions from neural networks for geometric point clouds. These neural networks operate on point clouds X ∈ R n×d (where in many applications d = 3) and each of the n points in R d represents a geometric coordinate. Models for such data are required to be invariant (or equivariant) to both permutations in S n and rotations in O(d). This equivariant structure is similar to, but not identical to, the situation we have for graph eigecomposition: under the simple spectrum assumption, the symmetry transformations we are interested in is a single global permutation, and K sign changes, which are rotations in O(1) K . In the more general setting, we will have a single permutation and multiple rotations, whose dimension is determined by the multiplicity of each eigenvalue.
Via this analogy, we can look at spectral models for graphs from the perspective of point cloud networks. From this perspective, EPNN resembles geometric invariant networks, such as Schnet [42], which are based on simple invariant features. In contrast, [20] and [44] showed that, at least for point clouds, expressivity can be increased by recursively updating a rotation equivariant (in our scenario, sign equivariant) feature v (t)
i in parallel with the invariant feature h (t)
i . Inspired by these observations, we suggest the following sign equivariant feature refinement procedure:
We use the same initialization h (0) i as in Equation 1, and we initialize the equivariant feature v
(0) i to v (0) i = V i . We then iteratively update these two features via h (t+1) i = UPDATE (t,1) h (t) i , ⃗ λ, (h (t) j , v (t) i ⊙ v (t) j ) | j = 1, . . . , n v (t+1) i = v (t) i + n j=1 v (t) j ⊙ UPDATE (t,2) (h (t) i , h (t) j , v (t) i ⊙ v (t) j )
where UPDATE (t,1) is a multiset function, and UPDATE (t,2) maps its input to R K so that the elementwise product in the equation above is well defined.
After running this procedure for T iterations, we obtain an invariant global feature h global by aggregating the invariant node features h (T ) i using a READOUT function, as in (3). This gives us a sign invariant function f (V, ⃗ λ) = h global . We name the class of all functions obtained by running this procedure with all different choices of update and readout functions equiEPNN.
We note that we can obtain EPNN models by setting UPDATE (t,2) to be the constant mapping to the zero vector. Accordingly, F equi is at least as expressive as EPNN. In Section 4.4 we will show that it is strictly more expressive.
this section cite: ['b12', 'b41', 'b19', 'b43']

Section: On the incompleteness of spectral graph neural networks
In this section, we analyze the expressive power of EPNN and equiEPNN on graphs with a simple spectrum. We first provide a counterexample to prove its incompleteness of EPNN on simple spectrum graphs. We then show that an equiEPNN can separate the counterexample, thus proving it is strictly more expressive than EPNN. Next we provide a subset of V K simple on which EPNN is complete. Finally, we discuss how our results imply the incompleteness of popular spectral GNNs even in the simple spectrum case.
this section cite: []

Section: EPNN is incomplete
We first introduce a pair of non-isomorphic eigendecompositions, (V, ⃗ λ) and (U, ⃗ λ) in V K simple , which EPNN cannot distinguish, that is, it assigns them the same final feature after any number of refinement steps. In this construction n = 12, K = 6, and we fix the same choice of distinct eigenvalues ⃗ λ for both examples. To define V, U , we denote
z 0 = 1 1 , z 1 = -1 1 , z 2 = 1 -1 , z 3 = -1 -1 , 0 2 = 0 0 ,
and note that z 0 , . . . , z 3 are the four elements of the abelian group {-1, 1} 2 . Using these, we define U, V via
U T = z 0 z 1 z 2 z 3 0 2 0 2 0 2 0 2 z 0 z 1 z 2 z 3 z 0 z 1 z 2 z 3 z 0 z 1 z 2 z 3 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 z 0 z 1 z 3 z 2 z 0 z 2 z 1 z 3 V T = z 0 z 1 z 2 z 3 0 2 0 2 0 2 0 2 z 0 z 1 z 2 z 3 z 0 z 1 z 2 z 3 z 0 z 1 z 2 z 3 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 z 1 z 0 z 2 z 3 z 2 z 0 z 3 z 1
We now show that U, V are not isomorphic and cannot be separated by EPNN: Theorem 1. (Incompleteness of EPNN) The following statements hold:
1. U and V are not isomorphic under the group action of S 12 × {-1, 1} 6 .
2. EPNN cannot separate U and V after any number of iterations.
3. U and V have no non-trivial automorphisms.
Therefore, EPNN is incomplete on simple spectrum graphs.
Proof Idea. To show U, V are not isomorphic, we note that for any pair of permutation-sign matrices taking U to V , the first four columns of U T must be mapped the first four columns of V T . The same is true for columns 5 -8 and 9 -12. Considering the first four columns, we see that any sign matrix mapping them from U to V will be of the form diag(z, z, z ′ ) for z, z ′ ∈ {-1, +1} 2 . The same argument for columns 5 -8 and 9 -12 gives sign patterns of the form diag(z ′ , z, z 1 • z) and diag(z, z ′ , z 2 • z), respectively. But there is no sign pattern satisfying these three constraints simultaneously.
We now explain the lack of separation of EPNN. We refer to the multiset of the multiplications of a column i with all the other columns, as the column i's purview. In the initial step, the purview of each column in the first 4-column block in V T and U T , is identical, as the first 4 columns exhibit a group structure with the multiplication operation. Thus, the hidden states of the first 4 indices of U T and V T will be identical. By similar arguments, this holds for the remaining two blocks. Thus, after a refinement step, the nodes in each block cannot distinguish among those from other blocks, both in U T and V T . Therefore, additional refinement procedures maintain identical representations for members of each index 'block' and corresponding blocks in U T and V T . This implies EPNN cannot separate U and V .
A full proof of the theorem is provided in the Appendix.
this section cite: []

Section: Remark:
In many cases we are interested in eigenvalue decompositions of symmetric matrices, in which case the columns of V, U (the rows of V T , U T ) should be orthonormal. While our V, U do not satisfy this condition, in the Appendix we show how they can be enlarged to yield a counterexample that has the same properties, and does have orthonormal columns.
this section cite: []

Section: When is EPNN complete?
The counterexample proves that there is an inherent limit to the expressive power of contemporary spectral invariant networks. We note that in this example U, V had a significant number of zero entries. We now show that when U, V each have at least one row without any zeros, EPNN will be complete (in particular, this condition always holds when the matrices U, V have less than n zero entries): Theorem 2 (EPNN Can Distinguish Dense Graphs with Distinct Eigenvalues). Let D ⊆ V K simple denote the set of (V, ⃗ λ) where V has a row without zero entries. Then EPNN is complete on D .
Proof. By assumption, an index i exists such that the i-th row of V has no zeros. The hidden state h
i after a signal iteration of EPNN (see ( 2)) can encode the eigenvalues ⃗ λ, the squared values of each coordinate of V i , and the multiset of pairwise products
V i ⊙ V j , as h (1) i = (V i ⊙ V i , V i ⊙ V j | j = 1, . . . , n )(4)
To recover V from h
i up to symmetries, we can fix the sign ambiguity by choosing all coordinates of V i to be positive. We can then recover the remaining V j from the multiset in Equation 4. This uncovers the inner workings of EPNN in processing simple spectrum graphs. Essentially, each entry can be normalized to represent a group element in O(1), which acts as a local frame of reference, see [10] for more background, allowing us to reconstruct the eigenvectors up to sign symmetries.
this section cite: ['b9']

Section: Unique node identification via EPNN
A well-known mechanism for circumventing the limited expressive power of GNNs is by injecting unique node identifiers (IDs), which break the symmetries that hinder GNNs' separation ability [28,14]. Popular approaches include random node initialization [5] and combinatorial methods [8], yet they are either limited by their discontinuity or break permutation equivariance. A natural question is whether the node features from EPNN are unique after finitely many iterations? If so, we have attained node IDs that do not break equivariance and change continuously with the eigendecomposition, alleviating the deficiencies of widely-used methods. We answer this question in the affirmative, provided the eigenvectors adhere to a sparsity pattern.
this section cite: ['b27', 'b13', 'b4', 'b7']

Section: Theorem 3. (EPNN for Unique Node Identifiers) Let D ⊆ V K
simple denote the set of (V, ⃗ λ) where V has no automorphisms, and has at most one zero per eigenvector. Then, one iteration of EPNN with injective UPDATE and READOUT functions assigns a unique identifier to each hidden node feature.
Proof. By contradiction, assume that there exist distinct indices i, j such that h (1) i = h
(1) j . By the definition of EPNN, we have that
{V i ⊙ V k } n k=1 = {V j ⊙ V k } n k=1 and V i ⊙ V i = V j ⊙ V j .(5)
We deduce for the second equality that |V
i | = |V (q)(q)
j | for all coordinates q = 1, . . . , K. If for some q we had V (q) i = 0, then also V (q) j = 0, in contradiction to the assumption that the q-th eigenvector has at most one zero entry. Thus all entries of V i and V j are non-zero.
Next, we deduce from Equation 5 and the fact that
|V (q) i | = |V (q) j | > 0 for all q, that {(s (q) i V (q) k ) K q=1 } n k=1 = {(s (q) j V (q) k ) K q=1 } n k=1
where s (q) i ∈ {±1} and is defined as
V (q) i |V (q) i |
and s (q) j is defined analogously. This means that there exists a permutation σ which swaps i with j, such that s
(q) i V (q) k = s (q) j V (q)
σ(k) for all k = 1, . . . , n and q = 1, . . . , K. Equivalently,
P V S 1 = V S 2 =⇒ P V S 1 S 2 = V(6)
where S 1 and S 2 are diagonal matrices with s
i and s (q) j , respectively, on the diagonals, and P is the permutation matrix corresponding to σ. Since P swaps i with j, this is a non-trivial automorphism, in contradiction to the assumption. Thus h
(1) i ̸ = h (1) j , as required. Proof Idea. We show that after a single iteration, the equivariant update step can yield new matrices U (t) , V (t) , t = 1 which have no zeros. From Theorem 2, we know that a single iteration of EPNN, and hence also equiEPNN, is complete for such U (t) , V (t) , and thus two iterations of equiEPNN are sufficient for separation.
While equiEPNN is stronger than EPNN, the following result (proven in the appendix) shows that equiEPNN is also incomplete over simple spectrum graphs: Theorem 4. (Incompleteness of Equivariant EPNN) There exist X, Y ∈ R 16×6 such that the following statements hold: 1. X and Y are not isomorphic under the group action of S 16 × {-1, 1} 6 .
this section cite: []

Section: Equivariant EPNN cannot separate X and Y after any number of iterations.
Therefore, Equivariant EPNN is incomplete on simple spectrum graphs.
In the appendix we also explain how this counterexample can be extended so that X, Y are orthogonal matrices which thus can form a full eigendecomposition of a real symmetric matrix.
this section cite: []

Section: Incompleteness of spectral GNNs
Theorem 1 proves that EPNN is incomplete on graphs with a simple spectrum. This spectral isomorphism test upper bounds the expressive power of many popular distance-based GNNs, which incorporate graph distances as edge features, such as Random Walk, PageRank, shortest path, or resistance distances [50,26,1,45,51]. Therefore, an immediate corollary of Theorem 1 follows:
Corollary 2. Graphormer-GD [50], PRD-WL [26], DiffWire [1], and Random-Walk based GNNs [45,51] are incomplete over graphs with a simple spectrum.
In addition to this result, in the appendix we prove that the model proposed by Zhou et al. [53] is not universal on simple spectrum graphs.
Proposition 3. Vanilla OGE-Aug [53] is incomplete over graphs with a simple spectrum.
this section cite: ['b49', 'b25', 'b0', 'b44', 'b50', 'b49', 'b25', 'b0', 'b44', 'b50', 'b52', 'b52']

Section: Experiments
Our goal in the experiments section is twofold: (a) statistically evaluate the validity of our bounded eigenmultiplicity approach for measuring expressivity and (b) empirically exemplify the utility of equiEPNNfoot_0 . To meet the first goal, we statistically analyze the eigenvalue multiplicity in realworld datasets, and the number of non-zero entries in the eigenvectors, to compare these with our theoretical conditions for EPNN completeness. We find that while the sparsity conditions for EPNN completeness are satisfied on some real-world datasets (MNIST-Superpixel), they are not satisfied on datasets with more intricate symmetries (ZINC). For the second goal, we evaluate the utility of the equivariant features derived from equiEPNN on the task of eigenvector canonicalization [30]. Finally, we benchmark equiEPNN against leading spectral methods on the popular ZINC and MNIST-Superpixel datasets.
this section cite: ['b29']

Section: Dataset statistics
We surveyed several popular graph datasets and documented their graph spectral properties. The results are shown in Table 1. We find that the MNIST Superpixel [34] dataset is almost homogeneously composed of graphs with a simple spectrum, and we find that (96.9%) of the graphs in this dataset have a full row without zeros, implying that EPNN is complete on almost all graphs.
Other datasets, such as MUTAG, ENZYMES, PROTEINS and ZINC [19,36], contain a substantial amount of graphs with eigenvalue multiplicity 2 and 3. Despite this, the number of eigenspaces of dimensions 2 and 3 is very low, averaging at around 1 per graph. On datasets with highly symmetric graphs, such as ENZYMES and PROTEINS, the graphs do not meet the sparsity condition of Theorem 2, thus EPNN will not necessarily faithfully learn the graph structure. This exemplifies the need for more expressive models that are complete on graphs with higher maximal eigenvalue multiplicity and sparse eigenvectors.
this section cite: ['b33', 'b18', 'b35']

Section: Eigenvector canonicalization
Positional encoding is a cornerstone of graph learning using Transformer architectures, yet they suffer from the sign ambiguity problem [9]. It can be resolved by eigenvector canonicalization, which involves choosing a unique representation of each eigenvector. Yet, an inherent limitation of current canonicalization methods is that they are unable to canonicalize eigenvectors with nontrivial self-symmetries, often called uncanonicalizable eigenvectors [29,30].
Table 2: Uncanonicalizable Graph Eigenvectors in ZINC (Subset) [19] as percentage of total eigenvectors of eigen-space dimension 1. Property Percentage (%) Sum to 0 11.15 % Uncanonicalizable 10.93 % equiEPNN output sum to 0 0.0 % Uncanonicalizable after equiEPNN 0.0 %
To overcome this limitation, we devise a method to choose a canonical representation of the original eigenvectors via the equivariant output of equiEPNN. The only requirement is that each vector in the equivariant output does not sum to 0.
We test our hypothesis on a popular benchmark ZINC [19], and find that all the vectors in the equivariant output are canonicalizable and sum to zero, in contrast to the vectors from the eigendecomposition, where 10% of them are uncanonicalizable. Furthermore, we devise a way to choose a canonical representation of the original eigenvectors via the equivariant output and describe this in the Appendix. The results are shown in Table 2.
this section cite: ['b8', 'b28', 'b29', 'b18']

Section: Benchmarks: ZINC and MNIST
We evaluated equiEPNN on the image classification task MNIST-Superpixel [34], in which clustering of images is performed according to regions with similar pixel values, an algorithm creates a graph based on these regions, and each node is assigned a region-induced feature. We compared equiEPNN to leading spectral methods, all with a comparable parameter budget of ≈ 35K (see Table 3). We observe that it outperforms PPGN [32], which has cubic complexity, and GNNML1, which also processes the eigendecomposition of the graph. ChebNet outperforms all other methods, perhaps due to its handcrafted polynomial features.
We further evaluate equiEPNN via the standard regression task on the ZINC dataset of molecular graphs (we also tested eigenvector canonicalization on this same dataset). ZINC (Subset) has 12000 graphs with an average of 23.16 nodes per graph. We compare ourselves to leading methods with the standard ≈ 500K parameter budget and find that, out of the spectral methods, our method attains the best results, see Table 4.
this section cite: ['b33', 'b31']

Section: Future Work
A key future goal is to devise spectral GNNs that achieve completeness on graphs with simple spectra, and higher eigenvalue multiplicities. One interesting direction is to use higher-order point cloud networks to process the eigenvectors [53]. We have shown that treating each eigenspace as a separate entity does not lead to universality (see Subsection 4.5). Thus, these high-order networks should process the eigenvectors as a single entity, but remain invariant only to the sign and basis symmetries.
this section cite: ['b52']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: Our claims are properly stated in the abstract within their scope. We diligently wrote the assumptions. The experiments are aligned with the claims.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: We claim that we have not fully determined when completeness on simple spectrum graphs is achieved.
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
Answer: [Yes] Justification: All assumptions are stated, we provide proof ideas and full proofs are provided in the appendix. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: All code is provided in the supplementary material and configurations and instructions as well.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: All code is reproducible and provided openly with instructions.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: All configurations are described in supplementary material.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: When statistical significance is clear, we mention; otherwise we claim it is only comparable.
Guidelines:
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: All computer resources needed are mentioned in the supplementary material.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We fully abide by the NeurIPS Code of Ethics and preserve anonymity.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This is theoretical research that does not affect society.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
this section cite: []

Section: Answer: [NA]
Justification: There is no risk, all datasets have no risk and are widely used.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All the results by other researchers are cited, stated and given credit fully.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: All code is reproducible with instructions.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: No human subjects are involved in this research. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: No research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: No usage of LLMs in core method. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
We first prove: 2. the inseparability of U and V by EPNN.
Observe the purview of node i of U after the first refinement step of EPNN:
h (1
) i (U ) = (U i ⊙ U i , {U i ⊙ U j | j ∈ [10]})(7)
We will show that point clouds can be partitioned into 'blocks' such that each point in the block obtains the same hidden state. This block structure is recognized by viewing each point as a group element and each block as a multiplicative group. We will then show that this multiplicative group structure allows us to prove the inseparability of EPNN.
Concretely, our proof proceeds as follows:
1. The column entries of U T and V T can be partitioned into 3 blocks : B 1 ≜ {1, 2, 3, 4}, B 2 ≜ {5, 6, 7, 8}, and B 3 ≜ {9, 10, 11, 12}, such h
i (U ) and h
(1) j (U ) are identical for every i, j ∈ B k , k = 1, 2, 3.
this section cite: []

Section: It holds that h

this section cite: []

Section: For any
t ∈ N, h (t) i (U ) = h (t) i (V ) for every i = 1, 2, . . . , 12.
1. We first focus on B 1 and then extend the argument to B 2 and B 3 .
Since the elements U j for j = 1, 2, 3, 4 admit a multiplicative group structure, then for every i = 1, 2, 3, 4, the respective entries U i ⊙ U j for j ∈ [4] are identical (closure of groups.) For j = 5, 6, 7, 8 and i = 1, 2, 3, 4, the entries of the products V i ⊙ V j , are are zeros in two row entries and the non-zero entries in the remaining row, each element of the group Z 2 2 ∼ = {z 0 , z 1 , z 2 , z 3 } appears exactly once in the non-zero entries of the products, as it holds that z i Z 2 2 = Z 2 2 . Analogously, we can extend this argument to j = 9, 10, 11, 12 and i = 1, 2, 3, 4.
this section cite: []

Section: This means that h
(1) i (U ) and h (1) j (U ) are identical for every i, j ∈ B 1 .
Since, by definition of U and symmetry, each four-index quadruple B 1 ≜ 1, 2, 3, 4, B 2 ≜ 5, 6, 7, 8, and B 3 ≜ 9, 10, 11, 12 is a multiplicative group, the analysis for the hidden states of the indices in B 1 holds for B 2 and B 3 . This concludes item 1.
2. Up to now, we proved for indices i, j ∈ B k for k = 1, 2, 3, it holds that h (1) i = h (1) j . It remains to be proven that these hidden states are equivalent in both point clouds to conclude step 2.
Since the point cloud V T is derived from U T by multiplying the columns in B 2 by diag(z 0 , z 0 , z 1 ) and the columns in B 3 by diag(z 0 , z 0 , z 2 ), the purview (see Equation 7) of each index is identical in both point clouds, since z 2 • z 2 = z 1 • z 1 = z 0 which is the identity element, thus by definition of EPNN, this modification that maps U T to V T doesn't affect the pairwise multiplications in Equation 7.
this section cite: []

Section: 3.
To prove this step, we only need to show that the hidden states remain identical within each block, since the fact that they are identical across the point clouds stems from the same justification of step 2.
In the second update step, the arguments of Step 1 remain identical. Still, now we have updated hidden node information, but the hidden node information is identical across nodes belonging to the same block. Therefore, the only information this refinement yields is the categorization of nodes into blocks. Yet this information is already known in the initialized hidden states,{h (0) i i = 1, . . . , n}, since the zero entries of multiplication h (0) i = V i ⊙ V i determine the block that i belongs to. Therefore, the hidden states don't supply the network with any supplementary information other than the initialization h (0) i = V i ⊙ V i . Thus, after a second refinement step, the hidden states remain identical within each block, as they have after the first refinement step. Moreover, the corresponding hidden states of the two point clouds also remain equivalent due to the arguments in Step 2, which remain analogous, as the hidden states after a refinement only assign each node its respective block membership, which is exactly the information given in the first update step. This argument can then be applied recursively to any number of update steps.
In conclusion, we have shown that for any t ∈ N, the hidden states of both point clouds are identical (in corresponding indices), therefore after a permutation invariant readout, we obtain the same output.
We now prove 3. U and V have no nontrivial automorphisms. To show U, V are not isomorphic, we note that for any pair of permutation-sign matrices taking U to V , the first four columns of U T must be mapped the first four columns of V T . The same is true for columns 5 -8 and 9 -12. Considering the first four columns, we see that any sign matrix mapping them from U to V will be of the form diag(z, z, z ′ ) for z, z ′ ∈ {-1, +1} 2 . The same argument for columns 5 -8 and 9 -12 gives sign patterns of the form diag(z ′ , z, z 1 • z) and diag(z, z ′ , z 2 • z), respectively. But there is no sign pattern satisfying these three constraints simultaneously.
The automorphism group of this extended eigendecomposition is contained within that of U and U , respectively, and thus is also only the trivial group.
The proof of 1. which states that U and V are not isomorphic, is analogous to the proof of 3, and yields that the only sign pattern taking each point cloud to itself is (z 0 , z 0 , z 0 ), which implies each point cloud has only a trivial automorphism..
this section cite: []

Section: A.2 Extension to orthonormal counterexamples
The rows of the above point clouds U, V are not orthonormal. Thus, they are not eigenvectors of an eigendecomposition of a symmetric matrix. We fix this misalignment via the following 'orthogonalization' matrices: Taking Ũ to be a concatenation of the previous U and Û defined by
Û T = 2z 0 2z 1 2z 2 2z 3 0 2 0 2 0 2 0 2 -2z 0 -2z 1 -2z 2 -2z 3 -z0 2 -z1 2 -z2 2 -z3 2 2z 0 2z 1 2z 2 2z 3 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 -z0 2 -z1 2 -z3 2 -z2 2 z0 2 z2 2 z1 2 z32
Then take Ṽ to be a concatenation of the previous V and V defined by
V T = 2z 0 2z 1 2z 2 2z 3 0 2 0 2 0 2 0 2 -2z 0 -2z 1 -2z 2 -2z 3 -z0 2 -z1 2 -z2 2 -z3 2 2z 0 2z 1 2z 2 2z 3 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 -z1 2 -z0 2 -z2 2 -z3 2 z2 2 z0 2 z3 2 z12
The columns of Ũ and Ṽ are now orthogonal, and they can be made to have unit norm by normalizing each column. As these extensions exhibit the same symmetries of U and V , respectively, analogous arguments to the proof of inseparability of U and V by EPNN (Theorem 2) will apply to this new pair Ũ , Ṽ . Therefore, EPNN cannot distinguish Ũ and Ṽ .
this section cite: []

Section: A.3 Proofs for implications for real-world GNNs
Proposition 3. Vanilla OGE-Aug [53] is incomplete over graphs with a simple spectrum.
Proof. The method proposed by Zhou et al. [53] consists of a permutation equivariant and orthogonal invariant function. We will show that a counterexample by [30] also applies to this network.
Vanilla PGE-Aug relies on a permutation-equivariant and orthogonal invariant set encoding to process each eigenspace separately. Werevisit their separate definitions and theorems:
Definition 4 (O(p)-invariant universal representation [53]). Let f :
∞ n=0 R n×p → ∞ n=0 R n . Given an input V ∈ R n×p , f outputs a vector f (V ) ∈ R n .
The function f is said to be an O(p)-invariant universal representation if given V, V ′ ∈ R n×p and P ∈ S n , the following two conditions are equivalent:
(i) f (V ) = P f (V ′ ); (ii) ∃Q ∈ O(p), such that V = P V ′ Q.
Definition 5 (Universal set representation [53]). Let X be a non-empty set. A function f : 2 X → R is said to be a universal set representation if ∀X 1 , X 2 ∈ 2 X , f (X 1 ) = f (X 2 ) if and only if the two sets X 1 and X 2 are equal. Proposition 3.5 (Zhou et al. [53]) For each p = 1, 2, . . ., let f p be an O(p)-invariant universal representation function. Further let g : 2 R 3 → R be a universal set representation. Then the following function
r(G, X G ) = GNN A G , concat X G , g {concat[µ j 1 n , λ j 1 n , f µj (V j )]} K j=1 (8
)
is a universal representation. Here n = |V (G)|, ((λ 1 , µ 1 ), . . . , (λ K , µ K )) is the spectrum of G, and V j ∈ R n×µj are the µ j mutually orthogonal normalized eigenvectors of L G corresponding to λ j . We denote 1 n an all-1 vector of shape n × 1. GNN is a maximally expressive MPNN.
Then Zhou et al. [53] propose the following graph neural network: Definition 3.6 (Vanilla OGE-Aug). Let f p be an O(p)-invariant universal representation, for each p = 1, 2, . . ., and g : 2 R 3 → R be a universal set representation. Define Z : G →
∞ n=1 R n as Z(G) = g concat µ j 1 |V (G)| , λ j 1 |V (G)| , f µj (V j ) K j=1 ,(5)
In which the notations follow Proposition 3.5. For G ∈ G, Z(G) is called a vanilla orthogonal group equivariant augmentation, or Vanilla OGE-Aug on G.
We will show that architectures of the form of Proposition 3.5 and specifically Vanilla OGE-Aug are incomplete on simple spectrum graphs, contradicting the claim in Proposition 3.5 that such a representation is universal.
Consider the point clouds proposed by Ma et al. [30]:
U 1 = [u 11 , u 12 ] = 1 -1 1 -1 2 3 4 5 ⊤ ,(9)
U 2 = [u 21 , u 22 ] = -1 1 1 -1 2 3 4 5 ⊤ . (10
)
Suppose the first column eigenvector of U 1 and U 2 corresponds to eigenvalue λ 1 = 1, the second column eigenvector of U 1 and U 2 corresponds to eigenvalue λ 2 = 2, and other eigenvectors not shown corresponds to eigenvalue 0 (so we safely ignore them). Then the Laplacian matrices corresponding to U 1 and U 2 are: 17 19  11 19 23 31  17 23 33 39  19 31 39 51
L 1 = λ 1 u 11 u ⊤ 11 + λ 2 u 12 u ⊤ 12 =    9 11
   ,(11)
L 2 = λ 1 u 21 u ⊤ 21 + λ 2 u 22 u ⊤ 22 =   9
11 15 21 11 19 25 29 15 25 33 39 21 29 39 51
   . (12
)
We will now demonstrate the model in Proposition 3.5 will be unable to distinguish U 1 and U @ , regardless of the choice of the GNN.
First, consider an arbitrary O(1)-invariant representation f : R n → R n . We will show that f (U 1 ) and f (U 2 ) are identical.
By the permutation equivariance and O(1) invariance:
f (u 11 ) = f (-u 11 ) = f (P 11 u 11 ) = P 11 f (u 11 )(13)
where P 11 is any permutation that satisfies P 11 u 11 = -u 11 . Therefore P 11 can be chosen to be σ 1 ≜ (1 2) (3 4) or σ 2 ≜ (1 4) (2 3) . This analysis naturally extends to a proper eigendecomposition (orthonormal eigenvectors of a graph as proposed by Ma et al. [30] in the proof of their Corollary 3.5 [30].
this section cite: ['b52', 'b52', 'b29', 'b52', 'b52', 'b52', 'b52', 'b29', 'b29', 'b29']

Section: By
Therefore, as any universal, invariant set representation is the same on both U 1 and U 2 , the input to the network will be identical per its definition, and thus for their corresponding graphs G 1 and G 2 and identical node features X G1 and
X G2 , respectively it holds that r(G 1 , X G1 ) = r(G 2 , X G2 ) yet G
1 and G 2 are non-isomorphic, thus Vanilla OGE-Aug is incomplete. A.4 Proof for equiEPNN strictly more expressive Corollary 1. equiEPNN (see Section 3.3) can separate U and V after 2 iterations. Thus equiEPNN is strictly stronger than EPNN.
Proof. We show that after a single iteration, the equivariant update step can yield new matrices U (t) , V (t) , t = 1 which have no zeros. We can choose the update function UPDATE (1,2) such that
UPDATE (1,2) (v 5 ⊙ v 5 , v 1 ⊙ v 1 , v 5 ⊙ v 1 ) ≜ (1
, 1, 0, 0, 0, 0, 0) and for all other values we define it as ⃗ 0.
After a single iteration U (1) and V (1) will be
U (1)T = z 0 z 1 z 2 z 3 z 0 0 2 0 2 0 2 z 0 z 1 z 2 z 3 z 0 z 1 z 2 z 3 z 0 z 1 z 2 z 3 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 z 0 z 1 z 3 z 2 z 0 z 2 z 1 z 3 V (1)T = z 0 z 1 z 2 z 3 z 0 0 2 0 2 0 2 z 0 z 1 z 2 z 3 z 0 z 1 z 2 z 3 z 0 z 1 z 2 z 3 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 z 1 z 0 z 2 z 3 z 2 z 0 z 3 z 1
Since there exists a column (the fifth column) such that all its entries are non-zero in both U (1)T and V (1)T , from Theorem 2, we know that a single iteration of EPNN, and hence also of equiEPNN, can separate U (1)T , V (1)T . In conclusion, two iterations of equiEPNN are sufficient for separation.
this section cite: ['b0', 'b1']

Section: A.5 Proof of Incompleteness of Equivariant EPNN
The purpose of this section is to show that equivariant EPNN is also not complete on simple spectrum graphs. To show this, we will construct a counter-example of a pair X, Y which are not isomorphic with respect to the joint action of permutations and sign multiplications, and yet cannot be distinguished by equiEPNN. We note that the columns of X, Y are not orthonormal, and they do have automorphisms.
For X, Y ∈ R n×K , we will say that X ≡ Y , if there is some permutation matrix P such that P X = Y . We will say that s ∈ {-1, 1} K is an isomorphism between X and Y , if Xdiag(s) ≡ Y . Here diag(s) is the K × K diagonal matrix with s on the diagonal. An automorphism of X is an isomorphism from X to X.
As a first step to construct our counter example, we consider the subgroup H ≤ {-1, 1} 3 defined by H = {s ∈ {-1,
1} 3 | s 1 • s 2 • s 3 = 1}.
Let T be the 4 × 3 matrix whose rows are the four elements of H, namely
T =    1 1 1 1 -1 -1 -1 -1 1 -1 1 -1   
Note that Aut(T ) = H due to H having a group structure.
Next, we build the matrix X to consist of four different copies of T . Each copy will be not a 4 × 3 but a 4 × 6 matrix, where three of the columns are the columns of T , and the rest are zero columns. Moreover, any two copies of T will only have one non-zero column in common.
To do this, we choose four index sets in {1, 2, . . . , 6}, who have this intersection pattern, namely I 1 = {1, 2, 3}, I 2 = {3, 4, 5}, I 3 = {2, 4, 6}, I 4 = {1, 5, 6}. One can verify that indeed |I j ∩ I k | = 1 for all j ̸ = k. We then define the matrix T [I j ] to be the 4 × 6 matrix as described previously. For example
T [I 2 ] =    0 0 1 1 1 0 0 0 1 -1 -1 0 0 0 -1 -1 1 0 0 0 -1 1 -1 0   
We define X to be the block matrix
X =    T [I 1 ] T [I 2 ] T [I 3 ] T [I 4 ]    ∈ R 16×6 or explicitly X =                          1 1 1 0 0 0 1 -1 -1 0 0 0 -1 -1 1 0 0 0 -1 1 -1 0 0 0 0 0 1 1 1 0 0 0 1 -1 -1 0 0 0 -1 -1 1 0 0 0 -1 1 -1 0 0 1 0 1 0 1 0 1 0 -1 0 -1 0 -1 0 -1 0 1 0 -1 0 1 0 -1 1 0 0 0 1 1 1 0 0 0 -1 -1 -1 0 0 0 -1 1 -1     
We define Y similarly, but we elementwise multiply the rows of T [I 1 ] by the sign vector
q = [-1, 1, 1, 1, 1, 1] to obtain Y =    T [I 1 ]diag(q) T [I 2 ] T [I 3 ] T [I 4 ]    ∈ R 16×6 .
This is our counterexample. We claim/ Theorem 4. (Incompleteness of Equivariant EPNN) There exist X, Y ∈ R 16×6 such that the following statements hold:
1. X and Y are not isomorphic under the group action of S 16 × {-1, 1} 6 .
this section cite: []

Section: Equivariant EPNN cannot separate X and Y after any number of iterations.
Therefore, Equivariant EPNN is incomplete on simple spectrum graphs.
Proof. Remark: X, Y are not isomorphic. By considering the zero patterns of X and Y , one sees that if s ∈ {-1, 1} 6 is an isomorphism mapping X to Y , then s satisfies P T T [I 1 ]diag(s) = T [I 1 ]diag(q), for some permutation matrix P (acting on the rows of T [I 1 ]), and s must also define an automorphism of T [I j ] for j = 2, 3, 4. Since each T [I j ]'s rows (padded with zeros) form a group, its only automorphisms are elementwise multiplications of its rows by its group elements, which implies i∈Ij s i = 1. From these three automorphism conditions, we deduce:
s 3 • s 4 • s 5 = 1 s 2 • s 4 • s 6 = 1 s 1 • s 5 • s 6 = 1
Multiplying these three equations with each other we deduce that
s 1 • s 2 • s 3 = 1.
Now, if this holds, then s cannot satisfy P T T [I 1 ]diag(s) = T [I 1 ]diag(q) because the product of the first three entries of any row of P T T [I 1 ]diag(s) is 1, while the product of the first three entries of any row of T [I 1 ]diag(q) is -1.
X and Y cannot be separated by equiEPNN
We prove by induction that for any number of layers in an equiEPNN, the hidden states for nodes within the same partition B k are identical, and this holds for both graph structures X and Y . This equivalence prevents the network from separating them.
We introduce useful definitions:
Definition 6 (Block Structure and Neighborhoods). We partition the n = 16 nodes (rows) into 4 disjoint blocks B k for k = 1, . . . , 4 (e.g., B 1 = {1, . . . , 4}, B 2 = {5, . . . , 8}, etc.). The 4 × 6 matrix of initial equivariant features for block B k is B
k ≜ X[B k , :] = T [I k ].(0)
The non-zero feature indices for this block are I k . For a node i ∈ B k , we define its neighbors: N intra (i) ≜ B k and N inter (i) ≜ {1, . . . , n} \ B k .
this section cite: []

Section: Definition 7 (Invariant Node Neighborhood).
The message from a neighbor j to a node i of X is a tuple containing the neighbor's invariant features and an invariant computed from their equivariant features, (h
(l) j , x (l) i ⊙ x (l) j ). The Invariant Node Neighborhood of a node i at layer l is the multiset of invariant features I (l) i ≜ I (l) i,intra ∪ I (l)
i,inter , where
• I (l) i,intra = {(h (l) j , x (l) i ⊙ x (l) j ) | j ∈ N intra (i)} • I (l) i,inter = {(h (l) j , x (l) i ⊙ x (l) j ) | j ∈ N inter (i)}
where {•} denotes a multi-set. The update rule combines the node's own invariant state h (l) i with aggregations of the messages from its neighborhoods:
h (l+1) i = ϕ h (h (l) i , AGG(I (l) i ))
where AGG is a permutation-invariant aggregation function (e.g., sum or mean).
this section cite: []

Section: Definition 8 (Equivariant Node Neighborhood). The Equivariant Node Neighborhood of a node
i at layer l is defined by E (l) i ≜ E (l) i,intra ∪ E (l)
i,inter , where
• Intra-block Neighborhood E (l) i,intra = {ϕ v (h (l) i , h (l) j , x (l) i , x (l) j ) ⊙ x (l) j | j ∈ N intra (i)} • Inter-block Neighborhood E (l) i,inter = {ϕ v (h (l) i , h (l) j , x (l) i , x (l) j ) ⊙ x (l) j | j ∈ N inter (i)}
where {•} denotes a multi-set. Also, define the messages arriving to a node i ∈ B p from a different block B k (k
̸ = p) at layer l by E (l) i,k = {ϕ v (h (l) i , h (l) j , x (l) i , x (l) j ) ⊙ x (l) j | j ∈ B k }.
The equivariant feature is updated by summing over both neighborhoods:
x (l+1) i = x (l) i + m∈E (l) i,intra m + m∈E (l) i,inter m Proof outline:
1. We show that the blocks of X and Y are a particular case of a generalized block structure.
this section cite: []

Section: 2.
We analyze the mechanics of equiEPNN when processing these generalized X and Y to prove that the invariant node neighborhoods of corresponding nodes in X and Y are equivalent. This is the base of our induction.
3. We show that the equivariant update step maintains this generalized block structure for both X and Y . The equivariant update maintaining the generalized block pattern of X and Y is the induction step of the proof.
4. Since an equivariant update maintains the generalized block structure of X and Y , and the subsequent invariant node neighborhoods of corresponding points in generalized X and Y are identical, by the base of induction, equiEPNN will output the same readout for both X and Y after arbitrarily many refinement iterations (the hidden states are equivalent as multisets for both point clouds).
Base Case (Generalized Block Pattern and Invariant Update)
Generalized Block Pattern The initial invariant features h
(0) i = x (0
) i ⊙ x (0
)
i are identical for all i ∈ B k , as they equal the indicator vector for the partition I k . We consider a generalized case, where the initial equivariant features for block B k (with non-zero columns I k ) form a matrix B
B (0) k [:, I k ] =       s (0) k,1 s (0) k,2 s (0) k,3 -s (0) k,1 -s (0) k,2 s (0) k,3 s (0) k,1 -s (0) k,2 -s (0) k,3 -s (0) k,1 s (0) k,2 -s (0) k,3      
In our counterexample X, the scalars s k,j ≡ 1 for all k, j. For Y , s 1,1 = -1 (from block k = 1, column j = 1) and all other s k,j ≡ 1. We consider this generalized case because we will show that after an equivariant aggregation, this will be the format of the blocks. These are called generalized X and Y with a single scalar choice defining them, as the generalized Y is equivalent to generalized X up to a negation first row of the first block of X. We refer to these generalized X and Y as simply X and Y in the remainder of the proof. These initial hidden states h (0) i are identical for both point clouds X and Y , due to the invariance of squaring to sign changes. Additionally, h
(0) i = h (0) j for i, j ∈ B k and h (0) i ̸ = h (0) j for j /
∈ B k , due to the unique sparsity pattern of each block. This completes the first step of the outline. We now proceed to the second step of the outline, where we prove that an invariant update maintains the equivalence of the hidden states within each block.
Invariant Update We formally define the aggregation steps for a node i ∈ B k at layer l by splitting our analysis of its neighborhood into intra-block neighbors N intra (i) and inter-block neighbors N inter (i). For any two nodes i, j ∈ B k , we show their invariant neighborhoods yield identical aggregations.
• Intra-block: A message from a neighbor m ∈ B k is (h • Inter-block: The graph is constructed such that for any k ̸ = p, |I k ∩I p | = 1. The inter-block neighborhood for a node in B k consists of nodes from the other three blocks. Consider a neighbor m ∈ B p . The product
x (l) i ⊙ x (l)
m is non-zero only at the single index j = I k ∩ I p . Due to this structure, the resulting multiset of invariants from block B p is of the form {α j e j , α j e j , -α j e j , -α j e j } (where e j is the standard basis vector and α j is some scalar), which is identical for all i ∈ B k .
Since both neighborhood aggregations are identical, and h
(l) i = h (l) j for i, j ∈ B k , the update yields h (l+1) i = h (l+1) j
for both X and Y .
this section cite: []

Section: Inductive Step
Assume at layer l, for any partition B k , h (l) i = h (l) j for all i, j ∈ B k , and the equivariant feature matrix B (l) k ≜ X[B k , :] (l) maintains the scaled pattern structure.
Equivariant Update The update for the equivariant features x (l+1) i combines the original features x (l) i with aggregations from intra-block and inter-block neighbors.
this section cite: []

Section: Intra-block Aggregation:
The aggregation of messages within a block B k can be compactly expressed via summation and Hadamard products. The message function ϕ v produces scalar weights for each interaction. Since the invariant features h (l) are constant within the block, these weights depend only on the structural relationship between nodes i and j. Due to the graph's symmetries, there are only four unique interaction types within a block, resulting in four learned scalar vectors, with scalar dimension weights in each feature dimension in R K . Since only I k are the indices with non-zero features, we focus on their aggregation, and the rest of the inputs along other feature dimensions will be aggregated to 0, therefore we denote by a, b, c, d ∈ R 3 the reduction into the feature indices in I k . These form a symmetric weight matrix (in the node dimension) we denote by
Φ (l) k =    a b c d b a d c c d a b d c b a    ∈ R 4×4×3
This operation, which we denote by ⋆, scales the columns of the feature matrix B (l) k while preserving their sign-pattern structure:
Φ (l) k ⋆ B (l) k [:, I k ] =      a ⊙ B (l) k [1, I k ] + b ⊙ B (l) k [2, I k ] + c ⊙ B (l) k [3, I k ] + d ⊙ B (l) k [4, I k ] b ⊙ B (l) k [1, I k ] + a ⊙ B (l) k [2, I k ] + d ⊙ B (l) k [3, I k ] + c ⊙ B (l) k [4, I k ] c ⊙ B (l) k [1, I k ] + d ⊙ B (l) k [2, I k ] + a ⊙ B (l) k [3, I k ] + b ⊙ B (l) k [4, I k ] d ⊙ B (l) k [1, I k ] + c ⊙ B (l) k [2, I k ] + b ⊙ B (l) k [3, I k ] + a ⊙ B (l) k [4, I k ]      ∈ R 4×3 (14) =    α β γ -α -β γ α -β -γ -α β -γ    (15
)
where the entries of the resulting matrix (in the I k columns) are denoted by the scalars α, β, γ. These scalars are the result of applying the learned weights a, b, c, d (which are vectors) to the corresponding columns of B (l) k . Specifically, they are defined as:
α = s (l) k,1 (a i1 -b i1 + c i1 -d i1 ) β = s (l) k,2 (a i2 -b i2 -c i2 + d i2 ) γ = s (l) k,3 (a i3 + b i3 -c i3 -d i3 )
where a j is the j-th component of a, etc. and i 1 , i 2 , i 3 ∈ I k . This operation preserves the fundamen tal sign-pattern structure of each column, merely updating its overall scaling factor.
Inter-block Aggregation: Let i ∈ B p be a node index in block p, and let k ̸ = p be a different block index. Consider the equivariant node neighborhood E (l) i,k . We first focus on the inter-block aggregation of X and proceed to discuss that of Y . Consider the contribution of the equivariant message passing to the features of node i from block B k . There are 3 possible cases:
E (l) i,k (X) = {⃗ a ⊙ x j1 , ⃗ a ⊙ x j2 , ⃗ b ⊙ x j3 , ⃗ b ⊙ x j4 } (16
)
where x j1 , x j2 are (w.l.o.g) the points with a positive scalar product with x i in dimension d, and x j3 , x j4 are those with a negative product. By construction of T , the points x j1 , x j2 satisfy x j1 (m) = -x j2 (m) for each m ∈ I k \ I p . An analogous result holds for x j3 , x j4 . Therefore, summing all points in E (l) i,k yields zeros in feature entries I k \ I p .
this section cite: []

Section: Case 3: Aggregation of E
i,k along remaining indices. In all other indices, {1, 2, . . . , 6}\(I p ∪I k ), the features of x j (for j ∈ B k ) are 0. Thus, it trivially holds that after aggregating E (l) i,k , the resulting vector entries in those dimensions will also be 0.
We now address the inter-block update of Y in comparison with that of X. The only structural difference is the negated first column in block B 1 of Y . This affects aggregation for i ∈ B 1 and for i ∈ B 4 (since I 1 ∩ I 4 = {1}). The sign of [1] is flipped. This means the roles of ⃗ a and ⃗ b are swapped. For i ∈ B 1 :
x (l) i [1] • x (l) j
E (l) i,k (Y ) = { ⃗ b ⊙ y j1 , ⃗ b ⊙ y j2 , ⃗ a ⊙ y j3 , ⃗ a ⊙ y j4 } (17
)
The sum is thus negated. This occurs only along the first column of the first block. For i ∈ B 4 , the aggregation from B 1 is:
AGG(E i,1 (X)) = AGG({⃗ a ⊙ x j1 , ⃗ a ⊙ x j2 , ⃗ b ⊙ x j3 , ⃗ b ⊙ x j4 }) (18
) = AGG({⃗ a ⊙ e 1 ⊙ x j1 , ⃗ a ⊙ e 1 ⊙ x j2 , ⃗ b ⊙ -e 1 ⊙ x j3 , ⃗ b ⊙ -e 1 ⊙ x j4 }) (19) = AGG({ ⃗ b ⊙ -e 1 ⊙ y j1 , ⃗ b ⊙ -e 1 ⊙ y j2 , ⃗ a ⊙ e 1 ⊙ y j3 , ⃗ a ⊙ e 1 ⊙ y j4 }) (20
) = AGG({ ⃗ b ⊙ y j1 , ⃗ b ⊙ y j2 , ⃗ a ⊙ y j3 , ⃗ a ⊙ y j4 }) = AGG(E i,1 (Y ))(21)
for some ⃗ a, ⃗ b ∈ R 6 . The equality holds. Therefore, the equivariant aggregation of Y is equivalent to that of X, except in the first column of the first block, where it is negated. The aggregations for both X and Y maintain the generalized block pattern.
In conclusion of the inter-block aggregation, each x i ∈ B k will be added with an equivariant feature of the form c (l) ⊙ x i (where c (l) is a shared column vector), and y i will be added with c (l) ⊙ y i .
this section cite: ['b0']

Section: Full Update:
The new feature matrix B (l+1) k is the sum of the original features and the intra-and inter-block aggregations. This process preserves the essential column structure. The update can be expressed as:
B (l+1) k = (I + Φ (l) k )B (l) k + c (l) ⊙ B (l) k (22
)
where c (l) is a column vector. This operation simply updates the scalar multiples of each column. For example, the sum of the original features and the intra-block aggregation (for the I k columns) results in:
(I + Φ (l) k )B (l) k [:, I k ] =       s (l) k,1 + α s (l) k,2 + β s (l) k,3 + γ -(s (l) k,1 + α) -(s (l) k,2 + β) s (l) k,3 + γ s (l) k,1 + α -(s (l) k,2 + β) -(s (l) k,3 + γ) -(s (l) k,1 + α) s (l) k,2 + β -(s (l) k,3 + γ)      
By Equation 22, the equivariant features remain in the generalized block form.
In conclusion, at each layer, invariant features remain uniform within partitions, and equivariant features update symmetrically. Since the representations are structurally identical (up to the s 1,1 sign flip, which is preserved) for both graphs, they are indistinguishable.
this section cite: []

Section: X and Y can be extended to a proper eigendecomposition
To form a complete basis of 16 eigenvectors, we construct the remaining 10 orthogonal vectors, X ∈ R 16×10 . Define the local orthogonal basis for R 4 (along the rows of the matrix):
   a b c 1    ≜    1 -1 1 -1 1 -1 -1 1 1 1 -1 -1 1 1 1 1    (23
)
Let 0 ∈ R 4 be the zero vector. We construct the matrix XT ∈ R 10×16 as a block matrix (where each block a, b, . . . is a 1 × 4 row vector):
XT ≜               a -a 0 0 b 0 -a 0 c 0 0 -a 0 b -b 0 0 c 0 -b 0 0 c -c 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1              (24)
The rows of XT (columns of X) are orthogonal to each other and to the columns of X. Thus, after scaling, X f ull = [X, X] ∈ R 16×16 forms an orthonormal basis. The block structure is maintained within the first 6 rows of XT . An analogous proof to Part 2 shows that these do not contribute new information to the hidden states, other than their block membership. The "all-ones" vectors (last 4 rows) are constant on each block and do not pass messages between blocks. This means the invariant and equivariant aggregation remain analogous to Part 2 when processing the full matrix X f ull = [X, X]. The hidden states only depend on the structural relations defined by X. Therefore, for an analogous matrix Ỹ (with its first row a replaced by -a to maintain orthogonality with Y ), equiEPNN will yield the same output on [Y, Ỹ ] and [X, X], which form valid eigendecompositions for an equal simple spectrum. In conclusion, equiEPNN cannot separate X and Y .
this section cite: []

Section: B Experiments

this section cite: []

Section: B.1 Dataset statistics
We surveyed popular graph datasets and documented their graph spectral properties. The results are shown in Table 5. We find that the MNIST Superpixel [34] dataset is almost homogeneously composed of graphs with a simple spectrum, and we find that (96.9%) of the graphs in this dataset have a full row without zeros, implying that EPNN is complete on almost all graphs.
Other datasets, such as MUTAG, ENZYMES, PROTEINS and ZINC [19,36], contain a substantial amount of graphs with eigenvalue multiplicity 2 and 3. Despite this, the number of eigenspaces of dimensions 2 and 3 is very few per graph, averaging at around 1 per graph. On datasets with highly symmetric graphs, such as ENZYMES and PROTEINS, the graphs do not meet the sparsity condition of Theorem 2, thus EPNN will not necessarily faithfully learn the graph structure. This exemplifies the need for more expressive models that are complete on graphs with higher maximal eigenvalue multiplicity and sparse eigenvectors. We surveyed the graph spectra of popular datasets to verify the need for more expressive architectures based on graph properties. We now further specify the meaning of each row of Table 5 in Table B.1.
this section cite: ['b33', 'b18', 'b35']

Section: B.2 MNIST Superpixel
Below, in Tables 7, 8 and B.2, we list the experiment configurations and hyperparameters of the MNIST experiment.
As a toy experiment to examine the potential benefit of using equiEPNN, We implemented equiEPNN via a modification of the EGNN architecture [40] and EPNN with the same architecture, but without the eigenvector update step. For precise hyperparameter configuration, see the Appendix.
In our first experiment, we applied the proposed method on a classical task of handwritten digit classification in the MNIST dataset [24]. While almost trivial by today's standards, we use this example to verify the theoretical claims regarding expressivity on simple spectrum graphs. Our experimental setup employed both EPNN (coordinate updates disabled) and equiEPNN (coordinate updates enabled) as our models exclusively on the superpixel-based graph representation from the MNISTSuperpixels dataset. In this approach, each 28 × 28 image was converted into a graph where vertices correspond to superpixels and edges represent their spatial adjacency relations, each image was represented as a different graph. We tested our models with different positional encoding dimensions of k = 3, 8, 16 to evaluate performance across varying levels of spectral information.
For details configutions see Tanbes 7, 8, and 9.
this section cite: ['b39', 'b23']

Section: B.2.1 Ablation
We examined the performance of both methods on the MNIST Superpixel datasets, where the task is classification of handwritten digits. We found that equiEPNN outperforms EPNN, with the same
this section cite: []

Section: Definition 9 (Eigenvector Canonicalization).
A canonicalization of an eigenvector v ∈ R n is a map ϕ : R n → R n such that for every s ∈ O(1) ≃ {-1, 1}, it holds that ϕ(sv) = ϕ(v) and is permutation equivariant, that is for every permutation σ, ϕ(σv) = σϕ(v).
We now define the following eigenvector canonicalization map via the steps 1. For given eigenvectors V ∈ R n×k corresponding to distinct eigenvalues, we run equiEPNN for T iterations, to obtain the equivariant output V (T ) 2. We sum over the columns to obtain a matrix S = diag(s 1 , s 2 , . . . , s k ) where s i ≜ sign( n j=1 V (T ) (i, j)) ∈ {-1, +1}.
this section cite: []

Section: Canonicalize the eigenvectors via SV.
This defines an eigenvector canonicalization map ψ : R n×k → R n×k where ψ(V ) = SV for the S(V ) defined above. This map is naturally permutation equivariant, and it is easy to check that it is sign invariant.
As this maps canonicalized the original eigenvectors via aggregating global graph information that depends on the entire graph eigendecomposition and not each eigenvector separately, we obtain a map that practically achieves perfect canonicalization on ZINC [19].
See Tables 11 and 13 for experiment configurations. The expressive power of GNNs is commonly evaluated via the Weisfeiler-Lehman (WL) test, with standard Message Passing Neural Networks (MPNNs) being upper-bounded by the 1-WL test [49,35]. This has motivated the development of more powerful models aligned with higher-order k-WL tests [32]. The WL hierarchy and its variants have been clarified in tutorials by [17,37]. Other works have moved beyond the binary isomorphism objective to develop more continuous, fine-grained measures of expressivity based on graphons and tree distances [7]. Our work diverges from these combinatorial frameworks by proposing a hierarchy based on eigenvalue multiplicity, a natural concept in spectral graph theory. We demonstrate that even SGNNs considered powerful in the WL hierarchy (EPNN) can fail on spectrally-defined graph classes, revealing limitations not captured by combinatorial tests.
this section cite: ['b18', 'b48', 'b34', 'b31', 'b16', 'b36', 'b6']

Section: C.2 Higher-Order and Subgraph GNNs
To overcome the 1-WL barrier, a prominent line of research has focused on architectures that process higher-order structures. Subgraph GNNs, which represent a graph as an equivariant collection of its subgraphs, have proven to be a particularly powerful paradigm [6]. A significant challenge has been the computational complexity of these models. Recent work by [3] introduces a flexible and scalable framework for Subgraph GNNs using graph products and coarsening to manage complexity. This line of research, including work by [11], has also explored novel methods to boost expressivity by leveraging high-order derivatives of a base GNN model, drawing deep connections between this calculus-based approach and the WL hierarchy. Our work provides a complementary perspective by showing that even highly expressive architectural paradigms can have fundamental blind spots, such as the inability to distinguish certain graphs with simple spectra.
this section cite: ['b5', 'b2', 'b10']

Section: C.3 Spectral GNNs and Universality
Spectral GNNs define graph convolutions via spectral filters. Early work improved filter expressivity by moving from polynomials to complex rational functions, as in CayleyNets [25]. A key theoretical result from [47] established that linear spectral GNNs can achieve universal approximation on graphs with a simple spectrum. However, this universality relies on a crucial assumption: the use of a randomly sampled, non-equivariant node signal. This setting is distinct from the standard GNN expressivity analysis, which assumes permutation-equivariant operations on graph structure. Our work investigates the expressivity of permutation-equivariant SGNNs, such as EPNN, under the same simple spectrum condition. We prove that, in this more standard setting, these models are fundamentally incomplete. We construct explicit counterexamples of non-isomorphic graphs with simple spectra that EPNN cannot distinguish, revealing a critical limitation that was not apparent from prior analyses.
this section cite: ['b24', 'b46']

Section: C.4 Equivariant Design and Generalization
A core principle in modern GNN theory is designing architectures that respect the symmetries of graph data, i.e., permutation invariance and equivariance [46]. This has led to principled methods for handling spectral features, such as the sign and basis ambiguities of eigenvectors. Models like SignNet and BasisNet are designed to be invariant to these symmetries by processing eigenspaces independently [27]. Work by [22] has analyzed the implicit bias of such equivariant networks, showing that gradient descent favors solutions with specific structural properties in the Fourier domain. Other work has explored probabilistic frameworks for breaking symmetries when necessary [23]. Our work builds on these principles; we show that even a principled equivariant architecture like EPNN is incomplete, and our proposed solution, equiEPNN, is directly inspired by equivariant network designs.
this section cite: ['b45', 'b26', 'b21', 'b22']

Section: C.5 Unified Theories and GNN Limitations
One recent research direction is to move towards a more holistic understanding of GNNs by connecting expressivity, generalization, and universality. Work by [39] proposes a unified framework using pseudometrics based on optimal transport to derive both universal approximation theorems and generalization bounds for MPNNs on attributed graphs. Concurrently, critical work has highlighted the practical limitations of GNNs. For instance, [4] demonstrated that GNNs can 'overfit' the graph structure, using it even when it is detrimental to the task. This suggests that theoretical expressivity does not automatically translate to better performance. Our paper contributes to this line of inquiry by identifying a novel and unexpected failure mode for a class of GNNs that are already considered highly expressive. This reinforces the notion that expressivity is not monolithic and that different architectures have distinct failure modes.
this section cite: ['b38', 'b3']

Section: References
Ref_id:b0 Title: DiffWire: Inductive Graph Rewiring via the Lovász Bound Year: (2022)
Ref_id:b1 Title: Isomorphism of graphs with bounded eigenvalue multiplicity Year: (1982)
Ref_id:b2 Title: A flexible, equivariant framework for subgraph GNNs via graph products and graph coarsening Year: (2024)
Ref_id:b3 Title: Graph neural networks use graphs when they shouldn't Year: (2024)
Ref_id:b4 Title: Towards invariance to node identifiers in graph neural networks Year: (2025)
Ref_id:b5 Title: Equivariant subgraph aggregation networks Year: (2022)
Ref_id:b6 Title: Fine-grained expressivity of graph neural networks Year: (2023)
Ref_id:b7 Title: Rethinking the power of graph canonization in graph representation learning with stability Year: (2024)
Ref_id:b8 Title: Benchmarking graph neural networks Year: ()
Ref_id:b9 Title: Equivariant frames and the impossibility of continuous canonicalization Year: (2024)
Ref_id:b10 Title: On the expressive power of GNN derivatives Year: (2025)
Ref_id:b11 Title: Weisfeiler and leman go infinite: Spectral and combinatorial pre-colorings Year: (2022)
Ref_id:b12 Title: Homomorphism expressivity of spectral invariant graph neural networks Year: (2025)
Ref_id:b13 Title: Generalization and representational limits of graph neural networks Year: (2020)
Ref_id:b14 Title: Weisfeiler leman for Euclidean equivariant machine learning Year: (2024)
Ref_id:b15 Title: Complete neural networks for complete euclidean graphs Year: (2024)
Ref_id:b16 Title: A short tutorial on the weisfeiler-lehman test and its variants Year: ()
Ref_id:b17 Title: On the stability of expressive positional encodings for graphs Year: (2024)
Ref_id:b18 Title: ZINC: A free tool to discover chemistry for biology Year: (2012)
Ref_id:b19 Title: On the expressive power of geometric graph neural networks Year: (2023)
Ref_id:b20 Title: Rethinking graph transformers with spectral attention Year: (2021)
Ref_id:b21 Title: Implicit bias of linear equivariant networks Year: (2022)
Ref_id:b22 Title: Improving Equivariant Networks with Probabilistic Symmetry Breaking Year: ()
Ref_id:b23 Title: Gradient-based learning applied to document recognition Year: (2002)
Ref_id:b24 Title: Cayleynets: Graph convolutional neural networks with complex rational spectral filters Year: (2019)
Ref_id:b25 Title: Distance encoding: Design provably more powerful neural networks for graph representation learning Year: (2020)
Ref_id:b26 Title: Sign and basis invariant networks for spectral graph representation learning Year: (2023)
Ref_id:b27 Title: What graph neural networks cannot learn: depth vs width Year: (2020)
Ref_id:b28 Title: Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding Year: (2023)
Ref_id:b29 Title: A canonicalization perspective on invariant and equivariant learning Year: (2024)
Ref_id:b30 Title: Graph inductive biases in transformers without message passing Year: (2023)
Ref_id:b31 Title: Provably powerful graph networks Year: (2019)
Ref_id:b32 Title: Graphit: Encoding graph structure in transformers Year: (2021)
Ref_id:b33 Title: Geometric deep learning on graphs and manifolds using mixture model CNNs Year: (2017)
Ref_id:b34 Title: Weisfeiler and leman go neural: Higher-order graph neural networks Year: (2019)
Ref_id:b35 Title: TUDataset: A collection of benchmark datasets for learning with graphs Year: (2020)
Ref_id:b36 Title: Weisfeiler and leman go machine learning: The story so far Year: (2023)
Ref_id:b37 Title: Recipe for a general, powerful, scalable graph transformer Year: (2022)
Ref_id:b38 Title: Generalization, expressivity, and universality of graph neural networks on attributed graphs Year: (2025)
Ref_id:b39 Title: E(n) equivariant graph neural networks Year: (2021)
Ref_id:b40 Title: The graph neural network model Year: (2009)
Ref_id:b41 Title: SchNet: A continuous-filter convolutional neural network for modeling quantum interactions Year: (2017)
Ref_id:b42 Title: Spectral graph theory Year: (2012)
Ref_id:b43 Title: On the expressive power of sparse geometric MPNNs Year: (2025)
Ref_id:b44 Title: Affinity-aware graph networks Year: (2023)
Ref_id:b45 Title: Scalars are universal: equivariant machine learning, structured like classical physics Year: (2021)
Ref_id:b46 Title: How powerful are spectral graph neural networks Year: (2022)
Ref_id:b47 Title: An empirical study of realized GNN expressiveness Year: (2024)
Ref_id:b48 Title: How powerful are graph neural networks? Year: (2019)
Ref_id:b49 Title: Rethinking the expressive power of GNNs via graph biconnectivity Year: (2023)
Ref_id:b50 Title: A complete expressiveness hierarchy for subgraph GNNs via subgraph weisfeiler-lehman tests Year: (2023)
Ref_id:b51 Title: On the expressive power of spectral invariant graph neural networks Year: (2024)
Ref_id:b52 Title: Towards stable, globally expressive graph representations with laplacian eigenvectors Year: (2024)
