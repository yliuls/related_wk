Title: T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning
Abstract: Self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations without labeled data, often by enforcing invariance to input transformations such as rotations or blurring. Recent studies have highlighted two pivotal properties for effective representations: (i) avoiding dimensional collapse-where the learned features occupy only a low-dimensional subspace, and (ii) enhancing uniformity of the induced distribution. In this work, we introduce T-REGS, a simple regularization framework for SSL based on the length of the Minimum Spanning Tree (MST) over the learned representation. We provide theoretical analysis demonstrating that T-REGS simultaneously mitigates dimensional collapse and promotes distribution uniformity on arbitrary compact Riemannian manifolds. Several experiments on synthetic data and on classical SSL benchmarks validate the effectiveness of our approach at enhancing representation quality. Code is available here. z 1 ' z 2 ' z 2 z 4 ' z 3 z 3 ' z 4 (0,0) x S d-1 z 1 MST(Z')

Section: Introduction
Self-supervised learning (SSL) has emerged as a powerful paradigm for learning meaningful data representations without relying on human annotations. Recent advances, particularly in visual domains [4,31,17,55,60], have demonstrated that self-supervised representations can rival or even surpass those learned through supervised methods. A dominant approach in this field is joint embedding self-supervised learning (JE-SSL) [12,4,56], where two networks are trained to produce similar embeddings for different views of the same image (see Figure 1). The fundamental challenge in JE-SSL is to prevent representation collapse, where networks output identical and non-informative vectors regardless of the input. To address this challenge, researchers have developed various strategies. Contrastive approaches [12,30] encourage embeddings of different views of the same image to be similar while pushing away embeddings of different images. Non-contrastive methods bypass the need of negative pairs, often employing asymmetric architectures [13,28,10] or enforcing decorrelation among embeddings through redundancy reduction [4,65,63].
Recent studies have identified a more subtle form of collapse known as dimensional collapse [33,35,29,41]. This phenomenon occurs when the embeddings span only a lower-dimensional subspace of the representation space, leading to high feature correlations and reduced representational diversity. Such a collapse can significantly impair the model's ability to capture the full complexity of the data, limiting performance on downstream tasks [26]. Another crucial aspect of representation quality is uniformity, which measures how evenly the embeddings are distributed across the representation space. It ensures that the learned representations preserve the maximum amount of information from the input data and avoid clustering in specific regions of the space. This property is fundamental because it helps maintain the discriminative power of the representations, and allows for better generalization to downstream tasks [23,61,52,24]. While existing methods have made progress in mitigating dimensional collapse and enforcing uniformity, they present limitations: contrastive methods are sensitive to the number of negative samples [26,4], and require large batch sizes, which can be computationally expensive; redundancy reduction methods, which enforce the covariance matrix to be close to the identity matrix, only leverage the second moment of the data distribution and are blind to, e.g., concentration points of the density, which can prevent convergence to the uniform density (Figure 6); and asymmetric methods lack theoretical grounding to explain how the asymmetric network helps prevent collapse [63].
Given these limitations, Fang et al. [23] suggested rethinking the notion of good SSL regularization, and proposed an Optimal Transport-based metric that satisfies a set of four principled properties (instance permutation, instance cloning, feature cloning, and feature baby constraints) that prevent dimensional collapse and promote sample uniformity. Their approach has its drawbacks: the optimal transport distances are costly to compute in general, and the proposed closed formula for accelerating the computation holds only on the sphere and requires square roots over SVD computations, which may lead to numerical instabilities.
To address these limitations, we propose T-REG, a novel regularization approach that is conceptually simple, easy to implement, and computationally efficient. T-REG naturally satisfies the four principled properties of Fang et al. [23] (Appendix G) and provably prevents dimensional collapse while promoting sample uniformity (Figure 2a). These properties make T-REG suitable for joint-embedding self-supervised learning, where it can be applied independently to each branch, yielding T-REGS (see Figure 1).
The central idea of T-REG is to maximize the length of the minimum spanning tree (MST) of the samples in the embedding space. It has strong theoretical connections to the line of work on statistical dimension estimation via entropy maximization [57] (Section 4.1). More explicitly, given a point cloud Z in Euclidean space, a spanning tree (ST) of Z is an undirected graph G = (V, E) with vertex set V = Z and edge set E ⊂ V × V such that G is connected without cycle. We define the length of G as:
E(G) := (z,z ′ )∈E ∥z -z ′ ∥ 2 .(1)
A minimum spanning tree of Z, denoted by MST(Z), is an ST of Z that minimizes length E; it is unique under a genericity condition on Z. Since the length of MST(Z) scales under rescaling of Z, maximizing it alone leads the points to diverge (see Figure 2b). To prevent trivial scaling, T-REG constrains embeddings to a compact manifold, encouraging full use of the representation dimension and a uniform distribution.
Our main contributions can be summarized as follows:
i) We introduce T-REG (Equation ( 6)), a regularization technique that maximizes the length of the minimum spanning tree (MST) while constraining embeddings to lie on a hypersphere (Section 4). ii) We show both theoretically and empirically that T-REG naturally prevents dimensional collapse while enforcing sample uniformity (Sections 4.1 and 4.2). iii) We apply T-REG to SSL either as standalone regularization, combining directly with viewinvariance, or as an auxiliary loss to existing methods, yielding the T-REGS frameworkwhose effectiveness is evaluated through experiments on standard JE-SSL benchmarks (Section 5).
this section cite: ['b3', 'b30', 'b16', 'b54', 'b59', 'b11', 'b3', 'b55', 'b11', 'b29', 'b12', 'b27', 'b9', 'b3', 'b64', 'b62', 'b32', 'b34', 'b28', 'b40', 'b25', 'b22', 'b60', 'b51', 'b23', 'b25', 'b3', 'b62', 'b22', 'b22', 'b56']

Section: Related Work
Our work builds upon recent advances in Joint-Embedding Self-Supervised Learning (JE-SSL), which can be broadly categorized into two main approaches: contrastive and non-contrastive methods [26].
(i) Contrastive methods [32,44,12,30,14,9] are commonly based on the InfoNCE loss [46]. These methods encourage the embeddings of augmented views of the same image to be similar, while ensuring that embeddings from different images remain distinct. Contrastive pairs can either be sampled from a memory bank, as in MoCo [30,14], or generated within the current batch, as in SimCLR [12]. Furthermore, clustering-based methods [8,9,27] can also be seen as contrastive methods between prototypes, or clusters, instead of samples. SwAV [9], for instance, learns online clusters using the Sinkhorn-Knopp transform. However, despite their effectiveness, both approaches require numerous negative comparisons to work well, which can lead to high memory consumption. This limitation has spurred the exploration of alternative methods.
(ii) Non-contrastive methods bypass the reliance on explicit negative samples. Distillation-based methods incorporate architectural strategies inspired by knowledge distillation to avoid representation collapse, such as an additional predictor [13], self-distillation [10], or a moving average branch as in BYOL [28]. Meanwhile, redundancy reduction methods [65,20,4,68,63,56,62] attempt to produce embedding variables that are decorrelated from each other, thus avoiding collapse. These methods can be broadly categorized into two groups: those that enforce soft whitening through regularization and those that perform hard whitening through explicit transformations. BarlowTwins [65] and VICReg [4] regularize the off-diagonal terms of the covariance matrix of the embedding to have a covariance matrix that is close to the identity. W-MSE [20] transforms embeddings into the eigenspace of their covariance matrix (batch whitening) and enforces decorrelation among the resulting vectors. CW-REG [62] introduces channel whitening, while Zero-CL [68] combines both batch and channel whitening techniques. Building upon these methods, INTL [63] proposed to modulate the embedding spectrum and explore functions beyond whitening to prevent dimensional collapse.
In this paper, we introduce a self-supervised learning (SSL) approach that uses a novel regularization criterion: the maximization of the embeddings' minimum spanning tree (MST) length. Our approach aligns with the framework proposed by Fang et al. [23], satisfying the same four principled properties: instance permutation, instance cloning, feature cloning, and feature baby constraints.
this section cite: ['b25', 'b31', 'b43', 'b11', 'b29', 'b13', 'b8', 'b45', 'b29', 'b13', 'b11', 'b7', 'b8', 'b26', 'b8', 'b12', 'b9', 'b27', 'b64', 'b19', 'b3', 'b67', 'b62', 'b55', 'b61', 'b64', 'b3', 'b19', 'b61', 'b67', 'b62', 'b22']

Section: MST and dimension estimation
Steele [57] studies the total length of a minimal spanning tree (MST) for random subsets of Euclidean spaces. Let X n be an i.i.d. 1 n-sample drawn from a probability measure P X with compact support on R d . For d ≥ 2, Theorem 1 of [57] controls the growth rate of the length of MST(X n ) as follows:
E (MST(X n )) ∼ Cn (d-1)/d almost surely, as n → ∞,(2)
where ∼ denotes asymptotic convergence, and where C is a constant depending only on P X and d.
This asymptotic rate makes it possible to derive several estimators of the intrinsic dimension of the support of a measure of its samples [53,2,15]. Among these estimators, the following one theoretically coincides with the usual dimension in non-singular cases, and it empirically coincides with the real-valued Hausdorff dimension [21] in classical manifold examples or even fractal examples, such as the Cantor set or the Sierpiński triangle. Definition 3.1. Given a bounded metric space M , the MST dimension of M , denoted by dim MST (M ), is the infimal exponent
d ∈ N such that E (MST(X)) /|X| d-1 d
is uniformly bounded for all finite subsets X ⊆ M : dim MST (M ) := inf{d : ∃C such that E (MST(X)) /|X| d-1 d ≤ C for every finite subset X of M }.
this section cite: ['b56', 'b56', 'b52', 'b1', 'b14', 'b20']

Section: Persistent Homology Dimension.
The MST also appears in Topological Data Analysis (TDA) [47], where it relates to the total persistence in degree 0 of the Rips filtration [47]. Moreover, Persistent Homology (PH) has been used to define a family of fractal dimensions [1], dim i PH (M ), for each homological degree i ≥ 0. In particular, for i = 0 this coincides with the MST-based dimension, i.e., dim 0 PH (M ) = dim MST (M ). The PH dimension can be derived from entropy computations and has already been used in several dimension-estimation applications [58,5,19]. In this work, we directly leverage the connection to the entropy to obtain uniformity properties on compact Riemannian manifolds (see Section 4.1.2).
this section cite: ['b46', 'b46', 'b0', 'b57', 'b4', 'b18']

Section: MST optimization.
TDA further provides a mathematical framework for optimizing the length of MST(X) with respect to the point positions of X [11,40]. Within this framework, E (MST(X)) is differentiable almost everywhere, with derivatives given by the following simple formula:
∀x ∈ X, ∇ x E (MST(X)) = (x,z) edge of MST(X) ∇ x ∥x -z∥ 2 = (x,z) edge of MST(X) ∥x -z∥ -1 2 (x -z). (3)
Furthermore, under standard assumptions on the learning rate, stochastic gradient descent is guaranteed to converge almost surely to critical points of the functional. In particular, Equation (3) shows that each pair of points forming an edge in the MST exerts a repulsive force on the other during optimization.
MST computation. Given a finite point set X ⊂ R d , several classic sequential procedures exist to compute MST(X), notably Kruskal's, Prim's, and Boruvka's algorithms, which all have at least quadratic running time in the size of X. However, fast GPU-based parallelized implementations exist, for instance [22], which unifies Kruskal's and Boruvka's algorithms and incorporates optimizations such as path compression and edge-centric operations.
this section cite: ['b10', 'b39', 'b21']

Section: T-REG: Minimum Spanning Tree based Regularization
Our regularization T-REG has two terms: a length-maximization loss L E that decreases with the length of the minimum spanning tree, and a soft sphere-constraint L S that increases with the distance to a fixed sphere S. These two terms combined force the embeddings to lie on S (or close to it), while spreading them out along S.
Formally, given Z = {z 1 , ..., z n } ⊆ R d , the MST length maximization loss is defined as:
L E (Z) = - 1 n E (MST(Z)) ,(4)
where E (MST(Z)) denotes the length of the MST of Z. The soft sphere-constraint is given by:
L S (Z) = 1 n i (∥z i ∥ 2 -1) 2 .(5)
It penalizes points that move away from the unit sphere. Maximizing the MST length alone would cause the points to diverge to infinity; the sphere constraint prevents this by keeping the embeddings within a fixed region around S (Figures 2b and 2c). The overall T-REG loss combines these two terms:
L T-REG (Z) = γ L E (Z) + λ L S (Z),(6)
where γ and λ are hyperparameters controlling the trade-off between spreading out the embeddings and maintaining them on the sphere. The remainder of the section provides a theoretical analysis (Section 4.1) and empirical evaluation (Section 4.2) of T-REG.
this section cite: []

Section: Theoretical analysis

this section cite: []

Section: Behavior on small samples
We begin by considering the case where n ≤ d + 1. It is particularly relevant since, in SSL, batch sizes are often smaller than or comparable to the ambient dimension. In order to account for the effect of the soft sphere constraint, we assume the points of X lie inside some fixed closed Euclidean d-ball B of radius r centered at the origin (see below for the explanation). Theorem 4.1. Under the above conditions, the maximum of E (MST(X)) over the point sets X ⊂ B of fixed cardinality n is attained when the points of X lie on the sphere S = ∂B, at the vertices of a regular (n -1)-simplex that has S as its smallest circumscribing sphere.
Recall that a k-simplex is the convex hull of a set of k + 1 points that are affinely independent in R d -which is possible only for k ≤ d. The simplex is regular if all its edges have the same length, i.e., all the pairwise distances between its vertices are equal. In such a case, we have the following relation between its edge length a and the radius r of its smallest circumscribing sphere:
a = r 2(k + 1) k .(7)
Theorem 4.1 explains the behavior of T-REG as follows: first, minimizing the term L E in Equation (6) expands the point cloud until the sphere constraint term L S becomes the dominating term (which happens eventually since L S grows quadratically with the scaling factor, versus linearly for L E ); at that stage, the points stop expanding and start spreading themselves out uniformly along the sphere of directions. The amount of expansion before spreading is prescribed by the strength of the sphere constraint term versus the term in the loss, which is driven by the ratio between their respective mixing parameters λ and γ.
The proof of Theorem 4.1 relies on the following two ingredients: a standard result in convex geometry (Proposition 4.2), and a technical lemma-proved in Appendix A-relating the length of the MST to the sum of pairwise distances (Lemma 4.3). Proposition 4.2 (Eq. (14.25) in Apostol and Mnatsakanian [3]). Under the conditions of Theorem 4.1, and assuming n = d + 1, the sum of pairwise distances 1≤i<j≤n ∥z iz j ∥ 2 is maximal when the points of X lie on the bounding sphere S, at the vertices of a regular d-simplex. Lemma 4.3. For any points z 1 , . . . , z n ∈ R d :
E (MST ({z 1 , . . . , z n })) ≤ 2 n 1≤i<j≤n ∥z i -z j ∥ 2 .
Proof of Theorem 4.1. We prove the result in the case n = d + 1. The case n < d + 1 is the same modulo some extra technicalities and can be found in Appendix A.
Let z * 1 , . . . , z * n ∈ S lie at the vertices of a regular d-simplex. Then, for any points z 1 , . . . , z n ∈ B:
E (MST ({z 1 , . . . , z n })) Lemma 4.3 ≤ 2 n 1≤i<j≤n ∥z i -z j ∥ 2 Proposition 4.2 ≤ 2 n 1≤i<j≤n z * i -z * j 2 Eq. (7) = 2 n n(n-1) 2 r 2(d+1) d = (n -1) r 2(d+1) d = E (MST ({z * 1 , . . . , z * n })) .
this section cite: ['b2']

Section: Asymptotic behavior on large samples
We now consider the case where n > d + 1, focusing specifically on the asymptotic behavior as n → ∞. We analyze the constant C in Equation ( 2), which can be made independent of the density of the sampling X. This, in particular, allows us to show that uniform and dimension-maximizing densities are asymptotically optimal for E(MST(•)). We fix a compact Riemannian d-manifold, M, equipped with the d-dimensional Hausdorff measure µ.
this section cite: []

Section: Theorem 4.4 ([15, Corollary 5]).
Let X n be an iid n-sample of a probability measure on M with density f X w.r.t. µ. Then, there exists a constant C ′ independent of f X and of M such that:
n -d-1 d • E (MST(X n )) ----→ n→∞ C ′ f d-1 d X dµ almost surely.(8)
As pointed out by Costa and Hero [15], the limit in Equation ( 8) is related to the intrinsic Rényi
d-1 d -entropy: φ d-1 d (f ) = 1 1 -d-1 d log f d-1 d dµ,(9)
which is known to converge to the Shannon entropy as d-1 d → 1 [6]. The Shannon entropy, in turn, achieves its maximum at the uniform distribution on compact sets [48]. This result can be shown by directly studying the map ϕ : f → f p dµ, which shows that an optimal density function f X maximizes the dimensionality of the sampling X n . Given a compact set K ⊆ M, we consider the space D K of positive, continuous probability densities f on K. Proposition 4.5. For any 0 < p < 1 and any compact set K ⊆ M, the map ϕ| D K admits a unique maximum at the uniform distribution U K on K. Furthermore, we have ϕ(U A ) < ϕ(U B ) for all sets A, B ⊆ M such that µ(A) < µ(B).
Proof. The map ϕ is strictly concave, as the composition of the strictly concave function x → x p with the linear map x → x dµ. Since density functions integrate to 1 over K, maximizing ϕ corresponds to an optimization problem under constraint, which can be solved using the Lagrangian: Now, recalling that 0 < p < 1, we have:
L λ (f ) := K f p dµ -λ K f dµ -1 , with differential d (L λ ) f (h) = K pf p-1 -λ h dµ.
ϕ(U K ) = dU K dµ p dµ = K 1 µ(K) p dµ = µ(K) (µ(K)) p dU K = µ(K) 1-p ,
this section cite: ['b14', 'b5', 'b47']

Section: which proves the second part of the statement.
A direct consequence of Proposition 4.5 and the fact that D M is dense in the set of probability densities of L p (M) is the following corollary. Corollary 4.6. Let D be the set of probability densities over M, and p ∈ (0, 1). Then, the map f ∈ D → f p dµ reaches its maximum at the density f := dU M dµ . We conduct an empirical study on synthetic data to validate T-REG's ability to prevent dimensional collapse and promote sample uniformity.
this section cite: []

Section: Empirical evaluation
Preventing dimensional collapse. Following Fang et al. [23], we assess T-REG's effectiveness against dimensional collapse by measuring how sensitive its loss L E is to simulated collapse. Specifically, we generate 10, 000 data points in dimension 1024 from an isotropic Gaussian distribution, then zero out a fraction η of their coordinates to control the collapse level. As shown in Figure 3, the sensitivity of the T-REG loss to η is similar to that of the W 2 loss from Fang et al. [23], indicating that T-REG effectively penalizes dimensional collapse.
this section cite: ['b22', 'b22']

Section: Promoting sample uniformity.
We apply T-REG alone to optimize a given point cloud and analyze its behavior in both low-dimensional and high-dimensional scenarios (Figure 2). For this, we use two different input point clouds: (i) a degenerate set of 256 points on a 1-d curve (corresponding to the orange dots in Figures 2a and 2b); (ii) a set of 256 points in R 256 , initially concentrated around a specific point on the unit sphere, where each point x i is sampled as x i = e 1 + ε i , with ε i drawn uniformly from a ball of radius 0.001.
As illustrated in Figure 2a, optimization with T-REG successfully transforms the initial point cloud in a 3-d space into a uniformly distributed point cloud on the sphere, as per Corollary 4.6. This is achieved through the combination of MST length maximization and sphere constraint. The sphere constraint is crucial here: when optimizing only the MST length, L E , (see Figure 2b), the optimization fails to converge.
In high dimensions (Figure 2d), we analyze the distribution of cosine similarities between the embeddings. The initial distribution shows a sharp peak near 1, indicating highly correlated samples on the sphere. After optimization with T-REG, the distribution becomes almost a Dirac slightly below 0, indicating that the configuration of the points is close to that of the vertices of the regular simplex, as per Theorem 4.1.
this section cite: []

Section: T-REGS: T-REG for Self-supervised learning
T-REGS extends T-REG to Joint-Embedding Self-Supervised Learning. For an input image i, two transformations t, t ′ are sampled from a distribution T to produce two augmented views x = t(i) and x ′ = t ′ (i). These transformations are typically random crops and color distortions. We compute Method CIFAR-10 [37] CIFAR-100 [37] Zero-CL [68] 91. 3 68. [23], and we follow the same protocol: ResNet-18 models are pre-trained for 500 epochs on CIFAR-10/100, with a batch size of 256, followed by linear probing. We report Top-1 accuracy (%). Boldface indicates best performance.
embeddings z = h ϕ (f θ (x)) and z ′ = h ϕ (f θ (x ′
)) using a backbone f θ and projector h ϕ . T-REGS acts as a regularization applied separately to the embedding batches Z = [z 1 , ..., z n ] and Z ′ = [z ′ 1 , ..., z ′ n ]. Specifically, embeddings from each view batch, Z and Z ′ , are treated as points in a high-dimensional space, and Kruskal's algorithm [38] is used to construct two Minimum Spanning Tree (MST), one for each view batch. These MSTs yield two T-REG regularization terms, which are combined into the T-REGS objective as follows:
L T-REGS (Z, Z ′ ) = γ L E (Z) + λ L S (Z) LT-REG(Z) + γ L E (Z ′ ) + λ L S (Z ′ ) LT-REG(Z ′ ) .(10)
where γ, λ control the contribution of each term.
In practice, T-REGS can be used as (i) a standalone regularization, combined directly with an invariance term such as the Mean Squared Error:
L(Z, Z ′ ) = β L MSE (Z, Z ′ ) + L T-REGS (Z, Z ′ ), where L MSE (Z, Z ′ ) = 1 n i ∥z i -z ′ i ∥ 2 2
and β is a mixing parameter; or (ii) as an auxiliary loss to existing SSL methods: L(Z, Z ′ ) = β L SSL (Z, Z ′ ) + L T-REGS (Z, Z ′ ), where L SSL (Z, Z ′ ) denotes the objective function of a given SSL method, and β is a mixing parameter. An overview of T-REGS is presented in Figure 1.
The remainder of the section provides evaluations of T-REGS on standard SSL benchmarks (Section 5.1) and on a multi-modal application (Section 5.2); as well as loss coefficients and computational analyses (Section 5.3). Implementation details and further analyses are in Appendices C and E.
this section cite: ['b36', 'b67', 'b2', 'b37']

Section: Evaluation on standard SSL benchmark
We evaluate the representations obtained after training with T-REGS, either directly combined with view invariance or integrated with existing methods (i.e., BYOL, and Barlow Twins) on CIFAR-10/100 [37], ImageNet-100 [59], and ImageNet [18]. Our implementation is based on solo-learn [16], and we use torchph [7] for the MST computations. For T-REGS as a standalone regularizer, we use β = 10, γ = 0.2, λ = 8e -4.
Evaluation on CIFAR-10/100. We first focus on comparisons with Fang et al. [23] (W 2 -regularized methods), following the same protocol on CIFAR-10/100 [37] with ResNet-18. As shown in Table 1, T-REGS demonstrates strong standalone performance, achieving results within 0.1% of the best W 2 -regularized approach on CIFAR-10. Additionally, using T-REGS as an auxiliary loss consistently improves performance over the respective baselines, and over variants that use L u or W 2 as additional regularization terms.
Evaluation on ImageNet-100/1k. To assess the scalability of T-REGS, we evaluate our model on ImageNet-100 and ImageNet-1k using ResNet-18 and ResNet-50, respectively, following the standard linear evaluation protocol on ImageNet and comparing with the state of the art. We report Top-1 accuracy. As shown in Table 2, T-REGS is competitive with methods that use the same number of views (e.g., INTL), and improves existing methods when used as an auxiliary loss.
this section cite: ['b36', 'b58', 'b17', 'b15', 'b6', 'b22', 'b36']

Section: # views
Imagenet-100 [59] ImageNet-1k [18] Method Top-1 Batch Size Top-1 8 SwAV [9] 74.3 4096 66.5 FroSSL [56] 79.8 --SSOLE [34] 82.5 256 73.9 2 SimCLR [12] 77.0 4096 66.5 MoCo v2 [14] 79.3 256 67.4 SimSiam [13] 78.7 256 68.1 W-MSE [20] 69.1 512 65.1 Zero-CL [68] 79.3 1024 68.9 VICReg [4] 79.4 1024 68.3 CW-RGP [62] 77.0 512 67.1 INTL [63] 81 Table 3: Cross-Modal Retrieval after finetuning CLIP. Image-to-text (i → t) and text-to-image (t → i) retrieval results (top 1/5 Recall: R@1, R@5). The table is mostly inherited from Oh et al. [45]. Boldface indicates the best performance.
T-REGS can also be applied when branches differ in architecture and data modalities, as it regularizes each branch independently. Accordingly, we demonstrate its capabilities in a joint-embedding multi-modal setting.
Pre-trained multi-modal models, such as CLIP [51], provide broadly transferable embeddings. However, several works have shown that CLIP preserves distinct subspaces for text and image-the modality gap [42,45,36]. Prior analyses [45,64] relate this gap to low embedding uniformity; notably, CLIP's embedding space often remains non-uniform even after fine-tuning, which can hinder transferability. Given that T-REGS improves embedding uniformity when used as an auxiliary loss, we evaluate its impact on CLIP fine-tuning. We fine-tune CLIP using T-REGS as an auxiliary regularizer; more precisely, L T-REGS is applied independently to the image and text branches and combined with the standard L CLIP objective [51] to encourage more robust and uniformly distributed representations. We follow the protocol of m 3 -Mix [45]. We report R@1 and R@5 for image-to-text and text-to-image retrieval on Flickr30k and MS-COCO in Table 3, which shows that T-REGS improves performance over prior methods.
this section cite: ['b58', 'b8', 'b55', 'b33', 'b11', 'b13', 'b12', 'b19', 'b67', 'b3', 'b61', 'b62', 'b44', 'b50', 'b41', 'b44', 'b35', 'b44', 'b63', 'b50', 'b44']

Section: Analysis
Loss coefficients. We determine the final coefficients for T-REGS as a standalone regularizer on ImageNet-1k as follows (the same approach was applied when combining T-REGS with existing methods). Initial experiments revealed that maintaining β ≥ γ ≥ λ was essential to prevent representation collapse. To efficiently explore the parameter space while managing computational Table 5: Complexity and computational cost. Comparison between different methods is performed, with training on ImageNet-1k distributed across 4 Tesla H100 GPUs. The wallclock time (sec/step) is averaged over 500 steps. B, D ranges are reported from Bardes et al. [4], Garrido et al. [25].
costs, we fixed β (the largest coefficient) and systematically varied the ratios β γ and γ λ using 50epoch online probing [25]. As shown in Table 4, both L E and L S contribute to performance, with L S requiring a smaller weight. This suggests that the actual radius of the sphere is not critical, thereby validating our choice of a soft sphere constraint instead of a hard one.
Computational cost. We evaluate the computational cost of T-REGS. The MSTs are computed with Kruskal's algorithm [38], whose worst-case time is O(B 2 (D • logB)), with B the batch size and D the embedding dimension. Although Kruskal's main loop is sequential, preprocessing (computing the distance matrix and sorting its entries) dominates in practice and can be efficiently parallelized on GPUs (as in torchph [7], used in our implementation). Empirically, T-REGS matches the per-step wall-clock of VICReg and SimCLR (averaged over 500 steps during training on ImageNet-1k with B = 512, D = 1024; see Table 5).
this section cite: ['b24', 'b37', 'b6']

Section: Conclusion
We introduced T-REG, a regularization approach that prevents dimensional collapse and promotes sample uniformity. Our method maximizes the length of the minimum spanning tree (MST), coupled with a sphere constraint. Our analysis connects MST optimization to entropy maximization and uniformity on compact manifolds, providing theoretical guarantees corroborated by empirical results. We extend T-REG to Self-supervised learning, yielding T-REGS. On CIFAR-10/100 and ImageNet-100/1k, T-REGS is competitive with W 2 -regularized and state-of-the-art methods, both as a standalone regularizer and as an auxiliary term, underscoring its effectiveness.
this section cite: []

Section: References
Ref_id:b0 Title: A fractal dimension for measures via persistent homology Year: (2018)
Ref_id:b1 Title: A Fractal Dimension for Measures via Persistent Homology Year: (2020)
Ref_id:b2 Title: New horizons in geometry Year: (2017)
Ref_id:b3 Title: Vicreg: Variance-invariance-covariance regularization for self-supervised learning Year: (2022)
Ref_id:b4 Title: Intrinsic dimension, persistent homology and generalization in neural networks Year: (2021)
Ref_id:b5 Title:  Year: (2004)
Ref_id:b6 Title: Connectivity-optimized representation learning via persistent homology Year: (2019)
Ref_id:b7 Title: Deep clustering for unsupervised learning of visual features Year: (2018)
Ref_id:b8 Title: Unsupervised learning of visual features by contrasting cluster assignments Year: (2020)
Ref_id:b9 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b10 Title: Optimizing persistent homology-based functions Year: (2021)
Ref_id:b11 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b12 Title: Exploring simple siamese representation learning Year: (2021)
Ref_id:b13 Title: Improved baselines with momentum contrastive learning Year: (2020)
Ref_id:b14 Title: Determining Intrinsic Dimension and Entropy of High-Dimensional Shape Spaces Year: (2006)
Ref_id:b15 Title: solo-learn: A library of selfsupervised methods for visual representation learning Year: (2022)
Ref_id:b16 Title: Cluster and predict latent patches for improved masked image modeling Year: (2025)
Ref_id:b17 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b18 Title: Generalization bounds using data-dependent fractal dimensions Year: (2023)
Ref_id:b19 Title: Whitening for self-supervised representation learning Year: (2021)
Ref_id:b20 Title: Fractal Geometry: Mathematical Foundations and Applications Year: (2013)
Ref_id:b21 Title: A High-Performance MST Implementation for GPUs Year: (2023)
Ref_id:b22 Title: Rethinking the uniformity metric in self-supervised learning Year: (2024)
Ref_id:b23 Title: SimCSE: Simple contrastive learning of sentence embeddings Year: (2021)
Ref_id:b24 Title: Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank Year: (2023)
Ref_id:b25 Title: On the duality between contrastive and non-contrastive self-supervised learning Year: (2023)
Ref_id:b26 Title: Soft neighbors are positive supporters in contrastive visual representation learning Year: (2023)
Ref_id:b27 Title: Bootstrap your own latent-a new approach to self-supervised learning Year: (2020)
Ref_id:b28 Title: Exploring the gap between collapsed & whitened features in self-supervised learning Year: (2022)
Ref_id:b29 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b30 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b31 Title: Learning deep representations by mutual information estimation and maximization Year: (2018)
Ref_id:b32 Title: On feature decorrelation in selfsupervised learning Year: (2021)
Ref_id:b33 Title: Ssole: Rethinking orthogonal low-rank embedding for self-supervised learning Year: (2025)
Ref_id:b34 Title: Understanding dimensional collapse in contrastive self-supervised learning Year: (2022)
Ref_id:b35 Title: Enhanced ood detection through cross-modal alignment of multi-modal representations Year: (2025)
Ref_id:b36 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b37 Title: On the shortest spanning subtree of a graph and the traveling salesman problem Year: (1956)
Ref_id:b38 Title: i-mix: A domain-agnostic strategy for contrastive representation learning Year: (2021)
Ref_id:b39 Title: A framework for differential calculus on persistence barcodes Year: (2022)
Ref_id:b40 Title: Understanding collapse in non-contrastive siamese representation learning Year: (2022)
Ref_id:b41 Title: Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning Year: (2022)
Ref_id:b42 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b43 Title: Self-supervised learning of pretext-invariant representations Year: (2020)
Ref_id:b44 Title: Geodesic multi-modal mixup for robust fine-tuning Year: (2023)
Ref_id:b45 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b46 Title: Persistence theory: from quiver representations to data analysis Year: (2017)
Ref_id:b47 Title: Maximum entropy autoregressive conditional heteroskedasticity model Year: (2009-06)
Ref_id:b48 Title: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models Year: (2015)
Ref_id:b49 Title: GUDHI User and Reference Manual Year: (2025)
Ref_id:b50 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b51 Title: A theoretical analysis of contrastive unsupervised representation learning Year: (2019)
Ref_id:b52 Title: Persistent homology and the upper box dimension Year: (2021)
Ref_id:b53 Title: Un-mix: Rethinking image mixtures for unsupervised visual representation learning Year: (2022)
Ref_id:b54 Title:  Year: (2025)
Ref_id:b55 Title: Frossl: Frobenius norm minimization for efficient multiview self-supervised learning Year: (2024)
Ref_id:b56 Title: Growth rates of euclidean minimal spanning trees with power weighted edges Year: (1988)
Ref_id:b57 Title: On the limitations of fractal dimension as a measure of generalization Year: (2024)
Ref_id:b58 Title: Contrastive multiview coding Year: (2020)
Ref_id:b59 Title: Franca: Nested matryoshka clustering for scalable visual representation learning Year: (2025)
Ref_id:b60 Title: Understanding contrastive representation learning through alignment and uniformity on the hypersphere Year: (2020)
Ref_id:b61 Title: An investigation into whitening loss for self-supervised learning Year: (2022)
Ref_id:b62 Title: Modulate your spectrum in self-supervised learning Year: (2024)
Ref_id:b63 Title: Post-pre-training for modality alignment in vision-language foundation models Year: (2025)
Ref_id:b64 Title: Barlow twins: Self-supervised learning via redundancy reduction Year: (2021)
Ref_id:b65 Title: Deep metric learning with spherical embedding Year: (2020)
Ref_id:b66 Title: Hyperspherical embedding for point cloud completion Year: (2023)
Ref_id:b67 Title: Zero-cl: Instance and feature decorrelation for negative-free symmetric contrastive learning Year: (2021)
