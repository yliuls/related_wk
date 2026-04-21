Title: Joint Hierarchical Representation Learning of Samples and Features via Informed Tree-Wasserstein Distance
Abstract: High-dimensional data often exhibit hierarchical structures in both modes: samples and features. Yet, most existing approaches for hierarchical representation learning consider only one mode at a time. In this work, we propose an unsupervised method for jointly learning hierarchical representations of samples and features via Tree-Wasserstein Distance (TWD). Our method alternates between the two data modes. It first constructs a tree for one mode, then computes a TWD for the other mode based on that tree, and finally uses the resulting TWD to build the second mode's tree. By repeatedly alternating through these steps, the method gradually refines both trees and the corresponding TWDs, capturing meaningful hierarchical representations of the data. We provide a theoretical analysis showing that our method converges. We show that our method can be integrated into hyperbolic graph convolutional networks as a pre-processing technique, improving performance in link prediction and node classification tasks. In addition, our method outperforms baselines in sparse approximation and unsupervised Wasserstein distance learning tasks on word-document and single-cell RNA-sequencing datasets.

Section: Introduction
High-dimensional data with hierarchical structures are prevalent in numerous fields, e.g., gene expression [1][2][3], image analysis [4][5][6], neuroscience [7,8], and citation networks [9][10][11]. Therefore, finding meaningful hierarchical representations for these data is an important task that has attracted considerable attention [12][13][14][15][16][17][18][19][20]. While hierarchical structure is often assumed to exist in either the samples or the features [13,[21][22][23] (i.e., one of the data modes of a matrix), many real-world datasets exhibit hierarchical structure in both modes. For example, word-document data [24][25][26] commonly contain hierarchies within features (e.g., related keywords) and within samples (e.g., document topics). Another example is recommendation systems [27,28], where items (features) could be arranged into taxonomies or product categories, and users (samples) can be represented by multiple levels of behavioral or demographic relations [29,30].
An emerging approach to represent hierarchical data is based on embedding in hyperbolic space [13,14,[21][22][23]31]. However, most existing hyperbolic representation learning methods focus only on one mode of a data matrix [13,21,22], either the rows (samples) or the columns (features). A straightforward way to handle hierarchies in both modes is to learn a separate hierarchical representation for each. We postulate that a joint approach facilitates significant advantages and investigate how the hierarchical structure of features can improve the discovery of the sample hierarchy, and, conversely, how the sample hierarchy can improve the discovery of the feature hierarchy.
To this end, we propose to jointly learn the hierarchical representation of features and samples using Tree-Wasserstein Distance (TWD) [32] in an iterative manner. Our approach departs from Consider a word-document data matrix. We construct an initial tree for one data mode (e.g., words). This tree is then used to compute the TWD in the other mode (e.g., documents). The newly computed TWD informs a tree update of that mode, and the updated tree is subsequently used to compute the TWD in the cross-mode. This alternating procedure continues iteratively, refining both trees.
standard uses of TWD, where trees are typically built for computational efficiency in approximating Wasserstein distances with Euclidean ground metrics [32][33][34][35][36]. Instead, we use trees learned from hierarchical data through diffusion geometry and hyperbolic embedding [37], allowing the hierarchical structure of one mode to be incorporated in the computation of a distance of the other. We start by learning an initial tree of one mode (either features or samples). Then, this learned tree is incorporated into the computation of the TWD of the other mode. The computed, informed TWD is then used to infer a tree of that mode, i.e., sample TWD is used to update the sample tree and feature TWD is used to update the feature tree, as shown in Fig. 1. This procedure is repeated, alternating between the two modes, iteratively refining both trees and recomputing the corresponding TWDs. We provide theoretical guarantees that this iterative process converges and that a fixed point exists.
Based on the learned trees that represent the hierarchical structures of the data, we extend the iterative algorithm by incorporating an additional adaptive filtering step. This filtering is implemented using Haar bases [38][39][40][41], which are constructed based on the learned trees. More concretely, at each iteration, we view each sample (resp. feature) as a signal supported on the feature tree (resp. sample tree) [42]. We then apply Haar wavelet filters [38,43] induced by the feature tree (resp. sample tree) to the samples (resp. features), as illustrated in Fig. 1. Assuming the trees reflect the intrinsic hierarchical structure, applying the resulting data-driven wavelet filters can remove noise and other nuisance components [44]. We show that integrating the filters into the iterative process also leads to convergence. Empirically, we demonstrate that this approach improves hierarchical representations in terms of sparse approximation, and the resulting TWD leads to superior performance in document and single-cell classification.
We further demonstrate the practical benefit of our hierarchical representations by using them to initialize hyperbolic graph convolutional networks (HGCNs) [21,45]. By incorporating our datadriven hierarchical representation as a preprocessing step, we observe improved performance in link prediction and node classification tasks compared to standard HGCNs. This highlights both the compatibility and effectiveness of our approach for hyperbolic graph-based models.
Our contribution. (1) We present an iterative framework for jointly learning hierarchical representations of features and samples using TWD. (2) We further enhance the hierarchical representations using Haar wavelet filters constructed from the learned trees. (3) We show that our method achieves superior performance in sparse approximation, as well as document and single-cell (scRNA-seq) classification. (4) We also show that the proposed hierarchical representation can be used to initialize HGCNs, improving link prediction and node classification on hierarchical graph data.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b12', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b12', 'b13', 'b20', 'b21', 'b22', 'b30', 'b12', 'b20', 'b21', 'b31', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b37', 'b42', 'b43', 'b20', 'b44', 'b0', 'b1', 'b2']

Section: Related work
Tree-Wasserstein distance. Tree-Wasserstein distance (TWD) [32][33][34][35][36] was introduced to mitigate the high computational cost of the Wasserstein distance [46], which is a powerful tool to compare sample distributions while taking into account feature relationship [47][48][49]. Most TWD methods involve two steps: (i) constructing a feature tree and (ii) using this tree to compute a sample TWD matrix. While the tree is typically built to approximate the Euclidean ground metric in TWD literature, [37] recently proposed tree construction via hyperbolic embeddings [50] and diffusion geometry [51], enabling the tree metric to reflect geodesic distances on a latent unobserved tree underlying features.
Co-Manifold learning. Co-manifold learning aims to jointly recover the geometry of samples and features by treating their relationships as mutually informative, i.e., the feature manifold informs the sample manifold, and vice versa. It has been applied in joint embedding [52][53][54][55] and dimensionality reduction [7,56,57], assuming that both rows and columns of the data matrix lie on smooth, lowdimensional manifolds. While effective in capturing geometric structures, these methods often rely on Riemannian manifolds with non-negative curvature assumptions [55,58], which may not be suited well with the negative curvature often associated with hierarchical data.
Unsupervised ground metric learning. Wasserstein singular vectors (WSV) [59] were introduced to jointly compute Wasserstein distances across samples and features, where the Wasserstein distance in one mode (e.g., samples) acts as the ground metric for the other (e.g., features). However, WSV is computationally expensive. To address this, Tree-WSV [60] was recently proposed, using tree-based approximations of the Wasserstein distance to reduce computational complexity. While these methods alternate between computing distances in a manner similar to ours, neither WSV nor Tree-WSV was designed to learn the hierarchical representations of the data. In contrast, our method explicitly learns the hierarchical representation of samples and features, further enhanced by wavelet filtering and its integration into HGCNs. These aspects were not explored in the WSV or Tree-WSV frameworks.
this section cite: ['b31', 'b32', 'b33', 'b34', 'b35', 'b45', 'b46', 'b47', 'b48', 'b36', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b6', 'b55', 'b56', 'b54', 'b57', 'b58', 'b59']

Section: Background
Tree construction using hyperbolic and diffusion geometry. Consider a set of high-dimensional points Z = {z j ∈ R n } m j=1 , and let M ∈ R m×m be a suitable pairwise distance matrix between these m points. In a recent work [37], a binary tree was derived from a hyperbolic embedding obtained through multiscale diffusion densities built from M [50], employing the concept of lowest common ancestors [61] within this embedding space. The resulting tree, denoted T (M), has m leaves corresponding to the m points. Details on the embedding and tree construction are in App. A.
Tree-Wasserstein distance. Consider a tree T = (V, E, A) with N leaf leaves, where V is the vertex set with N nodes, E is the edge set, and A ∈ R N ×N is the edge weight matrix. The tree distance d T is the sum of weights of the edges on the shortest path between any two nodes on T . Let Υ T (v) be the set of nodes in the subtree rooted at v ∈ V . Each u ∈ V has a unique parent v, with edge weight w u = d T (u, v) [62]. Given distributions ρ 1 , ρ 2 ∈ R Nleaf supported on T , the TWD [32] is defined by
TW(ρ 1 , ρ 2 , T ) = v∈V w v u∈Υ T (v) (ρ 1 (u) -ρ 2 (u)) .(1)
Haar wavelet. Consider a complete binary tree B with m leaves. Let ℓ = 1, . . . , L denote the levels in the tree, where ℓ = 1 is the root level and ℓ = L is the leaf level. Let Υ(ℓ, s) be the set of all leaves in the subtree, whose root is the s-th node of the tree at level ℓ, where s = 1, . . . , N ℓ and N ℓ is the number of nodes in the ℓ-level. At level ℓ, a subtree Υ(ℓ, s) splits into two sub-subtrees Υ(ℓ + 1, s 1 ) and Υ(ℓ + 1, s 2 ). A zero-mean Haar wavelet β ℓ,s ∈ R m has non-zero values only at the indices corresponding to leaves in the sub-subtrees and is piecewise constant on each of them (see App. A for an illustration and further details) [40,39,43]. The set of these Haar wavelets, along with a constant vector, is complete and forms an orthonormal Haar basis, denoted by B ∈ R m×m with each column corresponding to a Haar wavelet (basis vector). Any vector a ∈ R m can be expanded in this Haar basis as a = ℓ,ε α i,ℓ,ε β ℓ,ε , where α i,ℓ,s = ⟨a, β ℓ,s ⟩ is the expansion coefficient.
this section cite: ['b36', 'b49', 'b60', 'b61', 'b31', 'b39', 'b38', 'b42']

Section: Proposed method
Problem setting. Given a data matrix X ∈ R n×m + with n rows (samples) and m columns (features), we denote X i,: and X :,j as the i-th sample and the j-th feature, respectively. Our goal is to learn hierarchical representations for both samples and features. We model the hierarchical structure of the data by constructing rooted weighted trees as follows: a sample tree T r with n leaves, where each leaf represents one sample, and a feature tree T c with m leaves, where each leaf represents one feature. ▷ Iterative update l ← l + 1 until convergence Placing data points as leaves follows the established line of works in statistics [63][64][65], manifold learning [52-55, 7, 43], and TWD literature [32][33][34][35][36]. We propose an iterative scheme that alternates between learning the sample tree and learning the feature tree.
this section cite: ['b62', 'b63', 'b64', 'b31', 'b32', 'b33', 'b34', 'b35']

Section: Iterative scheme for joint hierarchical representation learning
To jointly learn hierarchical representations for samples and features, we use TWD as a means to facilitate the relationships between samples and features. We begin by constructing a tree for one data mode; without loss of generality, we first construct an initial feature tree. Given a pairwise distance matrix M c ∈ R m×m over the m features, we build a complete binary tree T (M c ) with m leaves [37], where each leaf corresponds to a feature (see App. A for details). This feature tree then serves as the hierarchical ground metric for computing TWD between the samples:
W (0) r (i, i ′ ) = TW(r i , r i ′ , T (M c )) + γ r ζ(r i -r i ′ ),(2)
where r i = X ⊤ i,: / ∥X i,: ∥ 1 is the i-th normalized sample, ζ is a norm regularize based on the snowflake penalty [57], and γ r > 0. For brevity, we write the resulting sample pairwise TWD matrix as
W (0) r = Φ( X r ; T (M c )) ∈ R n×n ,(3)
where X r = [r 1 , . . . , r n ] ⊤ ∈ R n×m , and Φ denotes a function that computes the pairwise TWD between rows of the matrix X r using the tree T (M c ) defined over features. A similar analogous construction is applied to the other mode (the samples). Given an initial pairwise distance matrix M r ∈ R n×n over the n samples, we construct an initial sample tree T (M r ) with n leaves and use it to compute TWD between the features:
W (0) c (j, j ′ ) = TW(c j , c j ′ , T (M r )) + γ c ζ(c j -c j ′ ),(4)
where c j = X :,j / ∥X :,j ∥ 1 and γ c > 0. The resulting feature pairwise TWD is then written as
W (0) c = Φ( X c ; T (M r )) ∈ R m×m ,(5)
where X c = [c 1 , . . . , c m ] ⊤ ∈ R m×n . These matrices of the initial sample and feature pairwise TWDs in Eq. ( 3) and Eq. ( 5) serve as the starting point of the proposed iterative scheme.
Our iterative scheme alternates between the two data modes in a coordinate descent manner [66,67]. At iteration l = 0, the initial pairwise distance matrices M r and M c are used to construct the corresponding sample and feature trees, respectively. These trees are then used to compute the initial TWDs in Eq. ( 3) and Eq. ( 5). For all subsequent iterations l ≥ 1, the learned feature TWD from the previous step is used to construct a new feature tree, which is then used to compute the updated sample TWD. The same process is applied to the other mode: the sample TWD is used to construct a new sample tree, which in turn is used to compute the subsequent feature TWD. Formally, using the notation introduced above, the update steps at l + 1 iteration are given by
W (l+1) r = Φ( X r ; T (W (l) c )) ∈ R n×n , W (l+1) c = Φ( X c ; T (W (l) r )) ∈ R m×m .(6)
This alternating scheme, outlined in Alg. 1, allows the hierarchical representation in one mode to iteratively inform and refine the representation in the other mode.
Theorem 1. The sequences W (l) r and W (l) c generated by Alg. 1 have at least one limit point, and all limit points are fixed points if γ r , γ c > 0.
The proof is in App. C. Thm. 1 implies the existence of a limit point to which the proposed iterative scheme converges. Alg. 1 can be implemented in practice without regularization, i.e., γ r = γ c = 0. However, the conditions γ r , γ c > 0 are necessary for Thm. 1.
We remark that while other TWD methods [32][33][34][35][36] could in principle be incorporated into our iterative framework, we chose the tree construction method [37] for two main reasons. First, we empirically observe that computing sample and feature TWDs with these alternative TWD methods often fails to converge. Second, their use as cross-mode tree references would yield trees whose tree distances approximate the Wasserstein distance. However, the Wasserstein metric is not inherently a hierarchical metric. As a result, the trees derived from such approximations do not represent the hierarchy present in the data. In contrast, constructing a hierarchical representation using [37] yields a tree whose geodesic (shortest path) distances reflect the hierarchical relationship underlying the data [68,51]. As shown empirically in Sec. 6, the trees obtained by Alg. 1 yield meaningful hierarchical representations of features and samples across benchmarks from multiple domains, leading to improved performance compared to using other TWD methods within our iterative scheme. In addition, we show that the trees and the corresponding TWDs are progressively refined throughout the iterations.
this section cite: ['b36', 'b56', 'b65', 'b66', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b36', 'b67', 'b50']

Section: Haar wavelet filtering
To improve the refinement of trees across iterations, we apply a filtering step at every iteration. Specifically, our filters are constructed from Haar wavelets [38][39][40][41]. It was shown that Haar wavelets can be derived adaptively from trees [43]. Here, we propose to build the Haar wavelets from the trees inferred through the iterative process. Viewing each sample (resp. feature) as a signal supported on the feature tree (resp. sample tree) [42] allows us to apply data-driven Haar wavelet filters. Since by construction the filters reflect the intrinsic hierarchical structure of the data, they enhance this meaningful representation while suppressing noise and other nuisance components.
Given a feature tree T (M c ), we construct a Haar basis B c ∈ R m×m associated with the tree (see Sec. 3 and App. A for details). Subsequently, each sample can be expanded in this Haar basis, and we denote α i = (X i,: B c ) ⊤ ∈ R m as the vector of the expansion coefficients of the i-th sample. To define a wavelet filter, we select a subset of the Haar basis vectors in B c as follows. For each coefficient index j, we compute the aggregate L 1 norm n i=1 |α i (j)| and sort these values in descending order. Then, we sequentially add the corresponding indices to Ω until the cumulative contribution η Ω = q∈Ω n i=1 |α i (q)| exceeds a threshold ϑ c > 0. Let B c ∈ R m×d denote the matrix consisting of the d basis vectors in Ω. The filtering step, which yields the filtered samples, is defined as
Ψ(X; T (M c )) = (X B c ) B ⊤ c ∈ R n×m ,(7)
where Ψ denotes a wavelet filtering operator applied to the rows of the matrix X, using the Haar wavelet induced by the tree T (M c ) defined over features. An analogous wavelet filter can be constructed from a sample tree T (M r ) and applied to the features:
Ψ(Z; T (M r )) = (Z B r ) B ⊤ r ∈ R m×n ,(8)
where Z = X ⊤ , and B r ∈ R n×d ′ consists of the top d ′ basis vectors selected using a threshold ϑ r > 0 on cumulative coefficient magnitude.
We can apply this Tree Haar Wavelet filtering in our joint iterative scheme to update the trees as follows. Given initial inputs X (0) = X, Z (0) = X ⊤ and using the notation introduced above, at each iteration l ≥ 0, we filter the data using the Haar basis vectors derived from the trees:
X (l+1) = Ψ(X (l) ; T ( W (l) c )) ∈ R n×m , Z (l+1) = Ψ(Z (l) ; T ( W (l) r )) ∈ R m×n , (9
)
where
W (0) r = M r and W (0) c = M c for iteration l = 0.
The filtered data is normalized into discrete histograms 1 . Then we refine the trees and TWDs based on the normalized Haar-filtered data:
W (l+1) r = Φ( X (l+1) r ; T ( W (l) c )) ∈ R n×n , W (l+1) c = Φ( X (l+1) c ; T ( W (l) r )) ∈ R m×m , (10
) where X (l+1) r
and X (l+1) c are the column-normalized matrices of X (l+1) and Z (l+1) , respectively. We summarize this iterative learning scheme with the Haar wavelet filters in Alg. 2.
(l) c l ← 0, X (0) ← X, Z (0) ← X ⊤ , W (0) r ← M r and W (0) c ← M c ▷ Initialization repeat X (l+1) ← Ψ(X (l) ; T ( W (l) c )) and Z (l+1) ← Ψ(Z (l) ; T ( W (l) r )) ▷ Tree haar wavelet filtering X (l+1) r ← r (l+1) i = X (l+1) i,: ⊤ /∥X (l+1) i,: ∥ 1 X (l+1) c ← c (l+1) j = Z (l+1) j,: ⊤ /∥Z (l+1) j,: ∥ 1 W (l+1) r ← Φ( X (l+1) r ; T ( W (l) c )) and W (l+1) c ← Φ( X (l+1) c ; T ( W (l) r )) ▷ Iterative update l ← l + 1 until convergence Theorem 2. The sequences W (l)
r and W (l) c generated by Alg. 2 have at least one limit point, and all limit points are fixed points if γ r , γ c > 0.
The proof of Thm. 2 is in App. C. Same as Thm. 1, Thm. 2 implies the existence of a limit point to which the proposed iterative scheme with the addition of the Haar wavelet filters converges. We argue that the resulting data-driven wavelet filters attenuate noise and other nuisance components [43], and therefore improve the quality of the learned trees. As shown in Sec. 6, this filtering step contributes to more informative hierarchical representations underlying the high-dimensional data, resulting in superior performance on various tasks. We note that while constructing the filters using L 1 -based selection criterion [69] is effective, alternative filtering strategies [70][71][72][73][74][75] can be considered, depending on the downstream tasks. We leave this extension for future work.
We conclude this section with a few remarks. First, while our iterative procedure is broadly applicable to various TWD methods [32][33][34][35][36], the theoretical results and the ability to obtain meaningful hierarchical representations rely on our choice of the specific tree construction method [37]. Second, in each iteration, our method can be computed in O(n 1.2 + m 1.2 ) [37,76] (see App. E). In contrast, a naïve computation requires O(mn 3 + m 3 log m + nm 3 + n 3 log n), making our method more efficient. Third, similar to WSV [59] and Tree-WSV [60], our approach can be viewed as an unsupervised ground metric learning technique [77]. However, its specific focus on learning hierarchical representations for both samples and features distinguishes it from these methods and leads to superior empirical performance for hierarchical data (see Sec. 6). For further comparison with Tree-WSV, we refer to App. B and App. G.3. Finally, we remark that either the sample or the feature hierarchical structure can be provided as a prior and used for initialization of our method (see Sec. 5).
this section cite: ['b37', 'b38', 'b39', 'b40', 'b42', 'b41', 'b42', 'b68', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b36', 'b75', 'b58', 'b59', 'b76']

Section: Incorporating learned hierarchical representation within HGCNs
In Sec. 4, we introduced a joint hierarchical representation learning framework using TWD. Typically, the initial pairwise distance matrices M r and M c are data-driven, e.g., using standard metrics such as the Euclidean distance or the cosine similarity. An advantage of our method is that it can incorporate prior knowledge when a hierarchical structure is available for one of the modes. Without loss of generality, we consider scenarios where a hierarchical structure over the n samples is known and represented by a graph H = ([n], E, A), with [n] = {1, . . . , n}. To integrate this prior, we initialize M r using the shortest-path distances d H induced by H. This initialization introduces a structured prior in one mode while allowing the hierarchy in the other mode to be learned jointly via TWD.
Incorporating such a prior is particularly relevant for hierarchical graph data [9,21,78], where the node hierarchy is provided, while the feature structure remains implicit. We demonstrate the compatibility of our approach with hyperbolic graph convolutional networks (HGCNs) [21,45]. Specifically, during the neighborhood aggregation step in HGCNs, we replace the predefined hierarchical graph with the sample tree inferred by our method after convergence. This learned tree guides the aggregation process, enabling hierarchy-aware message passing that reflects structured relations among samples and across features. As shown in Sec. 6, incorporating our method as a pre-processing step improves performance on link prediction (LP) and node classification (NC) for hierarchical graph datasets. While our focus here is on HGCNs due to their improved performance on NC and LP, the same initialization strategy can be adopted in other hyperbolic architectures, such as hyperbolic neural networks (HNNs) [79,80]. Further technical details on this integration are provided in App. D.
this section cite: ['b20', 'b77', 'b20', 'b44', 'b78', 'b79']

Section: Experimental results
We evaluate our methods on sparse approximation and unsupervised Wasserstein distance learning with word-document and scRNA-seq benchmarks. Additionally, we examine the integration of our methods into HGCNs for hierarchical graph data on LP and NC tasks. The implementation details, including hyperparameters, are reported in App. E. Additional experiments, e.g., empirical convergence, ablation study, runtime analysis, and co-clustering performance, are presented in App. F.
this section cite: []

Section: Evaluating hierarchical representations via sparse approximation
We first demonstrate the advantages of the learned trees from Alg. 1 and Alg. 2 for sparse approximation tasks on the data matrix. The quality of the feature tree (and similarly, the sample tree) is evaluated by the L 1 norm of their expansion coefficients across all samples (and the features, respectively) [81][82][83]. A lower L 1 norm indicates a more efficient (sparser) representation of the data using fewer significant Haar coefficients, thus indicating the learned tree structures better reflect the hierarchical information of the data [43,53]. We test four word-document datasets [25]: BBCSPORT, TWITTER, CLASSIC, and AMAZON, and two scRNA-seq datasets [84]: ZEISEL and CBMC. Both types of data exhibit hierarchical structures in their features and samples [85,86]. In document data, words (features) form semantic hierarchies (e.g., animal → mammal → dog), while documents (samples) follow topic-subtopic structures (e.g., science → biology → genomics). In scRNA-seq data, genes (features) are organized by functional relationships such as biological pathways and gene ontologies (e.g., immune response genes → cytokine genes → specific interleukins), and cells (samples) follow developmental or taxonomic hierarchies (e.g., hematopoietic stem cell → progenitor cell → mature blood cell types), which are widely modeled as hierarchical in computational biology [87]. Additional information about these datasets can be found in App. E.
We propose to learn hierarchical representations for both samples and features, where each informs the other through TWD. To the best of our knowledge, this approach to hierarchical representation learning has not been previously explored. To demonstrate its effectiveness, we compare Alg. 1 and Alg. 2 against existing TWD-based methods as follows. For a fair comparison, we adapt each baseline TWD method to our iterative setting. Specifically, we begin by using the competing method to compute the sample TWD matrix, from which the sample tree is constructed to approximate the corresponding tree-based Wasserstein ground metric. This sample tree is then used to compute the feature TWD matrix, which in turn defines the feature tree. At each iteration, the same baseline method is used to compute the TWDs for samples and features. The competing TWD methods include: Quadtree [32], Flowtree [88], TSWD [35], UltraTree [36], weighted cluster TWD (WCTWD), weighted Quadtree TWD (WQTWD) [34], their sliced variants SWCTWD and SWQTWD, MST [89], Tree Representation (TR) [22], gradient-based hierarchical clustering (HC) in hyperbolic space (gHHC) [90], gradient-based Ultrametric Fitting (UltraFit) [91], and HC by hyperbolic Dasgupta's cost (HHC) [92]. See App. B for details on these methods. We use the prefix "co-" to denote the adaptation of each method to our iterative framework. In addition, we include comparisons with co-manifold learning that involves trees induced by a diffusion embedding (QUE) [52], and Tree-WSV [60], which learns unsupervised ground metrics based on tree approximation of WSV [59]. Tab. 1 reports the L 1 norm of the Haar coefficients across all samples and all features, respectively. We report the value after convergence for our methods, and for baselines, we either report it after convergence or, if convergence is not achieved, the obtained value after 25 iterations. We see that our methods provide a more efficient representation, showing that the sparsity and quality of the trees produced by our methods are superior and outperform baselines. Fig. 2 shows the L 1 norm of the Haar expansion coefficients obtained by the proposed methods across iterations on ZEISEL dataset, where we observe that the L 1 norm is iteratively reduced and reaches convergence. Note that this is not the objective we are minimizing but a consequence of our method of learning well hierarchical representations of the data. Notably, Alg. 2 consistently achieves better sparse approximation performance than Alg. 1. We attribute this improvement to the wavelet filtering step, which jointly considers the data and the structure and further improves the quality of the learned hierarchies at each iteration. One might ask whether wavelet filters could similarly benefit other TWD-based baselines. In App. F, we test this by applying wavelets using trees constructed from various TWD baselines. The results show that it does not consistently improve the quality of the tree representations. We argue that this is because the trees in these methods are primarily designed to approximate the Wasserstein distance as the ground metric, rather than to represent the hierarchical structure of the data. Therefore, these trees are not faithful hierarchical representations of the data, and thus, applying wavelet filters that depend on both data and tree structure fails to improve hierarchical representation learning.
this section cite: ['b80', 'b81', 'b82', 'b42', 'b52', 'b24', 'b83', 'b84', 'b85', 'b86', 'b31', 'b87', 'b34', 'b35', 'b33', 'b88', 'b21', 'b89', 'b90', 'b91', 'b51', 'b59', 'b58']

Section: Document and single-cell classification using learned TWD
We further demonstrate the effectiveness of the TWDs obtained by our methods through document and cell classification tasks. We compare our results with the same competing methods used in Sec. 6.1, and additionally include WSV [59] that learns unsupervised ground metrics as a baseline. Classification is performed using kNN based on the obtained distances, with cross-validation over five trials. Each trial randomly splits the dataset into 70% training and 30% testing sets.
Tab. 2 shows the document and single-cell classification accuracy. The accuracy of our methods is based on the TWDs after convergence. For the baselines, the accuracy is reported either based on the distance obtained after convergence or the distance after 25 iterations if convergence is not achieved. We see that our methods outperform the baselines by a large margin. This indicates that the TWDs, learned through our iterative scheme, effectively capture the interplay of the hierarchical structures between rows and columns. In addition, we observe that the classification accuracy improves with each iteration of our methods (see Fig. 9 in App. F). While the competing methods also show marginal improvement through the iterative procedure, our methods consistently achieve better performance. Our approaches exhibit fast convergence, typically within 10-14 iterations in all the tested datasets.
this section cite: ['b58']

Section: Link prediction and node classification for hierarchical graph data
Finally, we show the utility of our methods as a pre-processing step for HGCNs [21,45], evaluated on LP and NC tasks. We adhere to the experimental setups and baselines used in these works to maintain consistency. For LP task, we use a Fermi-Dirac decoder [93,13] to compute probability scores for edges. Then, the networks are trained by minimizing cross-entropy loss with negative sampling. The performance of LP is assessed by measuring the area under the ROC curve (AUC). For NC task, we employ a centroid-based classification method [94], where softmax classifiers and cross-entropy loss functions are utilized. Additionally, an LP regularization objective is integrated into the NC task [21,45]. The NC task is evaluated using the F1 score for binary-class datasets and accuracy for multi-class datasets. We test four datasets [9,95,21], including CORA, PUBMED, DISEASE and AIRPOT. Descriptions of these datasets and their splits are included in App. E. We compare our methods with two shallow methods: Euclidean embedding (EUC) and Poincaré embedding (HYP) [13]. We also include comparisons with the concatenation of shallow embeddings and node features, denoted as EUC-MIXED and HYP-MIXED. Furthermore, we include multi-layer perceptron (MLP) and its hyperbolic extension, HNNs [79], as well as four GNNs: GCN [96], GAT [97], GRAPHSAGE [98], and SGC [99]. Lastly, we include HGCNs [21] and H2H-GCN [45].
Tab. 3 shows the performance of integrating our methods with HGCNs on the LP and NC tasks compared to the competing methods. We repeat the random split process 10 times and report the average performance and standard deviation. Our methods consistently outperform the competing baselines across both tasks. Similar to Tab. 1 and Tab. 2, we observe in Tab. 3 that using wavelet filters demonstrates superior performance by a significant margin. This indicates that it effectively represents the hierarchical structure of the graph data, improving the expressiveness of both GNNs and hyperbolic embeddings. A natural question is whether wavelet filters alone (i.e., without our iterative scheme) could benefit HGCNs. We investigate it in App. F and find that such integration does not yield comparable outcomes. To our knowledge, our work is the first to incorporate wavelets with TWD and apply them to HGCNs. We note that our methods could have been integrated within HGCNs in other ways. However, as shown in Tab. 3, the straightforward integration already delivers favorable results. Thus, we opt to explore more complex integration techniques for future work.
this section cite: ['b20', 'b44', 'b92', 'b12', 'b93', 'b20', 'b44', 'b94', 'b20', 'b12', 'b78', 'b95', 'b96', 'b97', 'b98', 'b20', 'b44']

Section: Conclusions
This work introduces an iterative framework for jointly learning hierarchical representations of both samples and features using TWD. The proposed method begins by constructing a tree for one mode (either features or samples), which is then used to compute TWD and infer the tree construction for the other mode. The process alternates between modes, with each tree informing the inference of the other through pairwise TWD computations. To further improve the quality of the tree representations, we apply wavelet filters derived from the learned trees to the data at each iteration, which effectively suppress noise and filter out nuisance components. We show theoretically that the procedure converges and empirically that the trees and TWDs are refined across the iterations. Specifically, empirical evaluations on word-document and scRNA-seq datasets show that the resulting tree representations and TWDs lead to meaningful hierarchical representations. We further demonstrate that the proposed method can serve as a preprocessing step for HGCNs applied to hierarchical graph data, improving performance in hierarchical graph-based learning problems on link prediction and node classification.
Limitations and future work. One limitation of our approach lies in the use of trees to represent data geometry. While this representation aligns with our assumption that the data exhibits underlying hierarchical structures, it may not generalize well to data supported on more complex geometries, e.g., spherical manifolds [100], spaces with mixed curvatures [101], asymmetric data [102][103][104], or general graphs [105]. In future work, we plan to explore more flexible geometric representation methods that can accommodate a broader class of data geometries, while utilizing the proposed iterative procedure between samples and features. We also plan to explore a differentiable variant of TWD, such as the soft TWD [106], which could enable integration of our iterative process into neural architectures. We will further incorporate supervision or task-specific signals into the learning process. Finally, we plan to extend the approach to multi-way data (e.g., tensor-valued inputs).
this section cite: ['b99', 'b100', 'b101', 'b102', 'b103', 'b104', 'b105']

Section: References
Ref_id:b0 Title: Scaling single-cell genomics from phenomenology to mechanism Year: (2017)
Ref_id:b1 Title: The gene mover's distance: Single-cell similarity via optimal transport Year: (2021)
Ref_id:b2 Title: Current best practices in single-cell RNA-seq analysis: a tutorial Year: (2019)
Ref_id:b3 Title: Image denoising by sparse 3-d transform-domain collaborative filtering Year: (2007)
Ref_id:b4 Title: Deep image prior Year: (2018)
Ref_id:b5 Title: Hyperbolic deep learning in computer vision: A survey Year: (2024)
Ref_id:b6 Title: Hierarchical coupled-geometry analysis for neuronal structure and activity pattern discovery Year: (2016)
Ref_id:b7 Title: Poincaré embedding reveals edge-based functional networks of the brain Year: (2020)
Ref_id:b8 Title: Collective classification in network data Year: (2008)
Ref_id:b9 Title: Community structure in directed networks Year: (2008)
Ref_id:b10 Title: Hierarchical structure and the prediction of missing links in networks Year: (2008)
Ref_id:b11 Title: Learning structure from the ground up-hierarchical representation learning by chunking Year: (2022)
Ref_id:b12 Title: Poincaré embeddings for learning hierarchical representations Year: (2017)
Ref_id:b13 Title: Learning continuous hierarchies in the Lorentz model of hyperbolic geometry Year: (2018)
Ref_id:b14 Title: Hyperbolic image-text representations Year: (2023)
Ref_id:b15 Title: Hyperbolic vision transformers: Combining improvements in metric learning Year: (2022)
Ref_id:b16 Title: Hyperbolic deep neural networks: A survey Year: (2021)
Ref_id:b17 Title: Rdesign: hierarchical data-efficient representation learning for tertiary structure-based rna design Year: (2023)
Ref_id:b18 Title: Harp: Hierarchical representation learning for networks Year: (2018)
Ref_id:b19 Title: The numerical stability of hyperbolic representation learning Year: (2023)
Ref_id:b20 Title: Hyperbolic graph convolutional neural networks Year: (2019)
Ref_id:b21 Title: Tree! I am no Tree! I am a low dimensional hyperbolic embedding Year: (2020)
Ref_id:b22 Title: Hyperbolic representation learning: Revisiting and advancing Year: (2023)
Ref_id:b23 Title: Hierarchical relational models for document networks Year: (2010)
Ref_id:b24 Title: From word embeddings to document distances Year: (2015)
Ref_id:b25 Title: Supervised word mover's distance Year: (2016)
Ref_id:b26 Title: Matrix factorization techniques for recommender systems Year: (2009)
Ref_id:b27 Title: Introduction to recommender systems handbook Year: (2010)
Ref_id:b28 Title: Billionscale commodity embedding for e-commerce recommendation in alibaba Year: (2018)
Ref_id:b29 Title: Hierarchical user profiling for e-commerce recommender systems Year: (2020)
Ref_id:b30 Title: Flattening the parent bias: Hierarchical semantic segmentation in the Poincaré ball Year: (2024)
Ref_id:b31 Title: Fast image retrieval via embeddings Year: (2003)
Ref_id:b32 Title: The phylogenetic Kantorovich-Rubinstein metric for environmental sequence samples Year: (2012)
Ref_id:b33 Title: Approximating 1-Wasserstein distance with trees Year: (2022)
Ref_id:b34 Title: Tree-sliced variants of Wasserstein distances Year: (2019)
Ref_id:b35 Title: Learning ultrametric trees for optimal transport regression Year: (2024)
Ref_id:b36 Title: Tree-Wasserstein distance for high dimensional data with a latent feature hierarchy Year: (2025)
Ref_id:b37 Title: Multiresolution approximations and wavelet orthonormal bases of L 2 (R) Year: (1989)
Ref_id:b38 Title: A wavelet tour of signal processing Year: (1999)
Ref_id:b39 Title: Zur theorie der orthogonalen funktionensysteme Year: (1911)
Ref_id:b40 Title: The wavelet transform, time-frequency localization and signal analysis Year: (1990)
Ref_id:b41 Title: Graph signal processing: Overview, challenges, and applications Year: (2018)
Ref_id:b42 Title: Multiscale wavelets on trees, graphs and high dimensional data: Theory and applications to semi supervised learning Year: (2010)
Ref_id:b43 Title: The contourlet transform: an efficient directional multiresolution image representation Year: (2005)
Ref_id:b44 Title: A hyperbolic-to-hyperbolic graph convolutional network Year: (2021)
Ref_id:b45 Title: Optimal transport: old and new Year: (2009)
Ref_id:b46 Title: Mémoire sur la théorie des déblais et des remblais Year: (1781)
Ref_id:b47 Title: On the translocation of masses Year: (1942)
Ref_id:b48 Title: Computational optimal transport: With applications to data science Year: (2019)
Ref_id:b49 Title: Hyperbolic diffusion embedding and distance for hierarchical representation learning Year: (2023)
Ref_id:b50 Title: Diffusion maps Year: (2006)
Ref_id:b51 Title: Geometry and analysis of dual networks on questionnaires Year: (2014)
Ref_id:b52 Title: Sampling, denoising and compression of matrices by coherent matrix organization Year: (2012)
Ref_id:b53 Title: Reconstruction of normal forms by learning informed observation geometries from data Year: (2017)
Ref_id:b54 Title: Hölder-Lipschitz norms and their duals on spaces with semigroups, with applications to earth mover's distance Year: (2016)
Ref_id:b55 Title: Fast robust pca on graphs Year: (2016)
Ref_id:b56 Title: Co-manifold learning with missing data Year: (2019)
Ref_id:b57 Title: Testing the manifold hypothesis Year: (2016)
Ref_id:b58 Title: Unsupervised ground metric learning using Wasserstein singular vectors Year: (2022)
Ref_id:b59 Title: Fast unsupervised ground metric learning with tree-Wasserstein distance Year: (2025)
Ref_id:b60 Title: An improved cost function for hierarchical cluster trees Year: (2018)
Ref_id:b61 Title:  Year: (2008)
Ref_id:b62 Title: Convex biclustering Year: (2017)
Ref_id:b63 Title: Splitting methods for convex clustering Year: (2015)
Ref_id:b64 Title: Recovering trees with convex clustering Year: (2019)
Ref_id:b65 Title: Coordinate descent algorithms Year: (2015)
Ref_id:b66 Title: Regularization paths for generalized linear models via coordinate descent Year: (2010)
Ref_id:b67 Title: Manifold learning with arbitrary norms Year: (2021)
Ref_id:b68 Title: Entropy-based algorithms for best basis selection Year: (1992)
Ref_id:b69 Title: Biorthogonal bases of compactly supported wavelets Year: (1992)
Ref_id:b70 Title: Wavelets on graphs via spectral graph theory Year: (2011)
Ref_id:b71 Title: The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains Year: (2013)
Ref_id:b72 Title: Graph wavelets for multiscale community mining Year: (2014)
Ref_id:b73 Title: Tight wavelet frames on multislice graphs Year: (2013)
Ref_id:b74 Title: Learning laplacian matrix in smooth graph signal representations Year: (2016)
Ref_id:b75 Title: Scalability and robustness of spectral embedding: Landmark diffusion is all you need. Information and Inference: A Year: (2022)
Ref_id:b76 Title: Ground metric learning Year: (2014)
Ref_id:b77 Title: Infectious diseases of humans: dynamics and control Year: (1991)
Ref_id:b78 Title: Hyperbolic neural networks. Advances in Neural Information Processing Systems Year: (2018)
Ref_id:b79 Title: Hyperbolic neural networks++ Year: (2020)
Ref_id:b80 Title: Sparse image and signal processing: wavelets, curvelets, morphological diversity Year: (2010)
Ref_id:b81 Title: Tree approximation and optimal encoding Year: (2001)
Ref_id:b82 Title: Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information Year: (2006)
Ref_id:b83 Title: Optimal marker gene selection for cell type discrimination in single cell analyses Year: (2021)
Ref_id:b84 Title: The nested chinese restaurant process and bayesian nonparametric inference of topic hierarchies Year: (2010)
Ref_id:b85 Title: The single-cell transcriptional landscape of mammalian organogenesis Year: (2019)
Ref_id:b86 Title: Single-cell rna-seq clustering: datasets, models, and algorithms Year: (2020)
Ref_id:b87 Title: Scalable nearest neighbor search for optimal transport Year: (2020)
Ref_id:b88 Title: Shortest connection networks and some generalizations. The Bell System Technical Year: (1957)
Ref_id:b89 Title: Gradientbased hierarchical clustering using continuous representations of trees in hyperbolic space Year: (2019)
Ref_id:b90 Title: Advances in neural information processing systems Year: (2019)
Ref_id:b91 Title: From trees to continuous embeddings and back: Hyperbolic hierarchical clustering Year: (2020)
Ref_id:b92 Title: Hyperbolic geometry of complex networks Year: (2010)
Ref_id:b93 Title: Hyperbolic graph neural networks Year: (2019)
Ref_id:b94 Title: Revisiting semi-supervised learning with graph embeddings Year: (2016)
Ref_id:b95 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b96 Title: Graph attention networks Year: (2017)
Ref_id:b97 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b98 Title: Simplifying graph convolutional networks Year: (2019)
Ref_id:b99 Title: Spherical sliced-wasserstein Year: (2022)
Ref_id:b100 Title: Learning mixed-curvature representations in product spaces Year: (2018)
Ref_id:b101 Title: Finsler multi-dimensional scaling: Manifold learning for asymmetric dimensionality reduction and embedding Year: (2025)
Ref_id:b102 Title: Metric convolutions: A unifying theory to adaptive convolutions Year: (2024)
Ref_id:b103 Title: Finsler-Laplace-Beltrami operators with application to shape analysis Year: (2024)
Ref_id:b104 Title: Geometric deep learning: going beyond euclidean data Year: (2017)
Ref_id:b105 Title: Supervised tree-Wasserstein distance Year: (2021)
Ref_id:b106 Title: Hyperbolic groups Year: (1987)
Ref_id:b107 Title: Introduction to algorithms Year: (2022)
Ref_id:b108 Title: Metric spaces of non-positive curvature Year: (2013)
Ref_id:b109 Title: Foundations of hyperbolic manifolds Year: (1994)
Ref_id:b110 Title: Hyperbolic Procrustes analysis using Riemannian geometry Year: (2021)
Ref_id:b111 Title: The geometry of discrete groups Year: (2012)
Ref_id:b112 Title: From graph to manifold Laplacian: The convergence rate Year: (2006)
Ref_id:b113 Title: Towards a theoretical foundation for Laplacian-based manifold methods Year: (2008)
Ref_id:b114 Title: Hyperbolic distance based on emd and diffusion for hyperspectral imaging Year: (2025)
Ref_id:b115 Title: Les éléments aléatoires de nature quelconque dans un espace distancié Year: (1948)
Ref_id:b116 Title: Fast wavelet transforms and numerical algorithms i Year: (1991)
Ref_id:b117 Title: Displacement interpolation using Lagrangian mass transport Year: (2011)
Ref_id:b118 Title: Sinkhorn distances: Lightspeed computation of optimal transport Year: (2013)
Ref_id:b119 Title: Faster Wasserstein distance estimation with the Sinkhorn divergence Year: (2020)
Ref_id:b120 Title: Scalable optimal transport in high dimensions for graph distances, embedding alignment, and more Year: (2021)
Ref_id:b121 Title: Optimal transport: Fast probabilistic approximation with exact solvers Year: (2019)
Ref_id:b122 Title: Near-linear time approximation algorithms for optimal transport via sinkhorn iteration Year: (2017)
Ref_id:b123 Title: Convolutional networks on graphs for learning molecular fingerprints Year: (2015)
Ref_id:b124 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b125 Title: Heterogeneous graph attention network Year: (2019)
Ref_id:b126 Title: Graph convolutional neural networks for web-scale recommender systems Year: (2018)
Ref_id:b127 Title: Moleculenet: a benchmark for molecular machine learning Year: (2018)
Ref_id:b128 Title: Data-driven tree transforms and metrics Year: (2017)
Ref_id:b129 Title: Wasserstein barycenter and its application to texture mixing Year: (2011-06-02)
Ref_id:b130 Title: Sliced and radon wasserstein barycenters of measures Year: (2015)
Ref_id:b131 Title: Generalized sliced wasserstein distances Year: (2019)
Ref_id:b132 Title: Sliced-wasserstein on symmetric positive definite matrices for m/eeg signals Year: (2023)
Ref_id:b133 Title: Hyperbolic sliced-Wasserstein via geodesic and horospherical projections Year: (2023)
Ref_id:b134 Title: Ground metric learning on graphs Year: (2021)
Ref_id:b135 Title: Simultaneous ground metric learning and matrix factorization with earth mover's distance Year: (2014)
Ref_id:b136 Title: Supervised earth mover's distance learning and its computer vision applications Year: (2012)
Ref_id:b137 Title: Learning to match via inverse optimal transport Year: (2019)
Ref_id:b138 Title: Inverse optimal transport Year: (2020)
Ref_id:b139 Title: Correction to: convex analysis and monotone operator theory in Hilbert spaces Year: (2017)
Ref_id:b140 Title: Heat kernel and analysis on manifolds Year: (2009)
Ref_id:b141 Title: Hyperbolic graph attention network Year: (2021)
Ref_id:b142 Title: Hyperbolic diffusion Procrustes analysis for intrinsic representation of hierarchical data sets Year: (2024)
Ref_id:b143 Title: Distributed representations of words and phrases and their compositionality Year: (2013)
Ref_id:b144 Title: Cell types in the mouse cortex and hippocampus revealed by single-cell RNA-seq Year: (2015)
Ref_id:b145 Title: Simultaneous epitope and transcriptome measurement in single cells Year: (2017)
Ref_id:b146 Title: Gene2vec: distributed representation of genes based on co-expression Year: (2019)
Ref_id:b147 Title: Gaussian bandwidth selection for manifold learning and classification Year: (2020)
Ref_id:b148 Title: On the selection of appropriate distances for gene expression data clustering Year: (2014)
Ref_id:b149 Title: Short text similarity with word embeddings Year: (2015)
Ref_id:b150 Title: Optuna: A next-generation hyperparameter optimization framework Year: (2019)
Ref_id:b151 Title: Self-tuning spectral clustering Year: (2004)
Ref_id:b152 Title: Impact of signal-to-noise ratio and bandwidth on graph Laplacian spectrum from high-dimensional noisy point cloud Year: (2020)
Ref_id:b153 Title: Harmonic analysis of digital data bases Year: (2011)
Ref_id:b154 Title: Pot: Python optimal transport Year: (2021)
Ref_id:b155 Title: Efficient and effective optimal transportbased biclustering Year: (2022)
Ref_id:b156 Title: Cumida: an extensively curated microarray database for benchmarking and testing of machine learning approaches in cancer research Year: (2019)
Ref_id:b157 Title: Algorithm AS 136: A k-means clustering algorithm Year: (1979)
Ref_id:b158 Title: Coclustering through optimal transport Year: (2017)
Ref_id:b159 Title: Co-optimal transport Year: (2020)
Ref_id:b160 Title: Non-negative matrix factorization on manifold Year: (2008)
Ref_id:b161 Title: Comparing partitions Year: (1985)
Ref_id:b162 Title: A graph-theoretic game and its application to the k-server problem Year: (1995)
Ref_id:b163 Title: Gromov-wasserstein distances and the metric approach to object matching Year: (2011)
Ref_id:b164 Title: Tree mover's distance: Bridging graph metrics and stability of graph neural networks Year: (2022)
Ref_id:b165 Title: Decomposition of hardy functions into square integrable wavelets of constant shape Year: (1984)
Ref_id:b166 Title: Multiresolution graph transformers and wavelet positional encoding for learning long-range and hierarchical structures Year: ()
Ref_id:b167 Title: Graph signal processing for machine learning: A review and new perspectives Year: (2020)
Ref_id:b168 Title: The laplacian spectrum of a graph Year: (1990)
Ref_id:b169 Title: Spectral graph theory Year: (1997)
Ref_id:b170 Title: Spectral analysis of signals Year: (2005)
Ref_id:b171 Title: Discrete signal processing on graphs: Graph fourier transform Year: (2013)
Ref_id:b172 Title: Graph signal processing for geometric data and beyond: Theory and applications Year: (2021)
Ref_id:b173 Title: The L 1 norm of the Haar coefficients across all samples and features when applying wavelet filtering induced by trees constructed from various TWD-based baselines. Lower values indicate sparser representations. Arrows (↑) denote cases where wavelet filtering leads to an improvement (i.e., reduction in L 1 norm) compared to the unfiltered baseline Year: ()
