Title: Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few
Abstract: Attention mechanisms have achieved significant empirical success in multiple fields, but their underlying optimization objectives remain unclear yet. Moreover, the quadratic complexity of self-attention has become increasingly prohibitive. Although interpretability and efficiency are two mutually reinforcing pursuits, prior work typically investigates them separately. In this paper, we propose a unified optimization objective that derives inherently interpretable and efficient attention mechanisms through algorithm unrolling. Precisely, we construct a gradient step of the proposed objective with a set of forward-pass operations of our Contractand-Broadcast Self-Attention (CBSA), which compresses input tokens towards low-dimensional structures by contracting a few representatives of them. This novel mechanism can not only scale linearly by fixing the number of representatives, but also covers the instantiations of varied attention mechanisms when using different sets of representatives. We conduct extensive experiments to demonstrate comparable performance and superior advantages over black-box attention mechanisms on visual tasks. Our work sheds light on the integration of interpretability and efficiency, as well as the unified formula of attention mechanisms. Code is available at this https URL.

Section: Introduction
Attention mechanisms have been widely applied across diverse areas, including computer vision [1,2], natural language processing [3,4], and scientific discovery [5]. Nonetheless, a series of puzzling phenomena-such as emergent segmentation properties [6], in-context learning ability [7], attention collapse [8,9] and extreme-token phenomena [10]-have been uncovered in them, hindering the principled and trustworthy development. At the same time, the quadratic computational and memory complexity of self-attention with respect to the sequence length impedes its broader applications in real-time systems [11], as well as the processing of long documents [12] and high-resolution images [13].
In light of these challenges, it has been more crucial to mathematically demystify attention mechanisms, which offers deeper insights into their simplification and acceleration. Over the past few years, remarkable advances have been made in addressing the interpretability or efficiency issue separately. On the one hand, in an ante-hoc manner, attention mechanisms can be interpreted by optimization objectives grounded in clustering [14], denoising [15], energy minimization [16], matrix decomposition [17], and contrastive learning [18]. These inherently interpretable approaches are more rigorous than post-hoc explanations [19]. On the other hand, numerous techniques have been developed to alleviate the quadratic complexity of self-attention, including sparse attention [20] and linear attention [21].
However, the joint development of interpretability and efficiency in attention mechanisms remains a largely unexplored area of research. This leaves the design of efficient attention mostly heuristic, and the interpretations and explanations for attention mechanisms less instructive. To bridge this gap, we formulate a unified optimization objective by mildly modifying a compression-driven optimization objective called MCR 2 [22]. Indeed, this objective has been utilized for designing an interpretable softmax attention, MSSA [23], and a linear-time attention, TSSA [24]. But MSSA also scales quadratically, and TSSA is effectively a channel attention mechanism, which contrasts sharply with both softmax attention (token mixer) and linear attention (channel mixer). 1 Therefore, instead of an isolated mechanism, we aim to develop a framework that unifies these varied attention mechanisms in an interpretable way, revealing how they are fundamentally connected yet distinctly presented, as well as the trade-off between expressive capacity and efficiency.
In this paper, we adopt two ante-hoc interpretations to constitute our proposed optimization objective: a) input tokens are compressed towards low-dimensional structures for compact and structured representation; and b) the geometry and information-theoretic essence of input tokens can be captured by a small number of representatives [25,26] of them. Since the former has been formulated as the MCR 2 objective [22,23] (see Section 2), the remaining task is to leverage the representatives to optimize it, thereby efficiently compressing all by contracting a few (see Section 3.1). By unrolling the resulting optimization objective, we derive our Contract-and-Broadcast Self-Attention (CBSA), which contracts the representatives and broadcasts the contractions back to input tokens (see Section 3.2).
Given a fixed number of representatives, the computational and memory complexity of CBSA scales linearly with the number of input tokens. Moreover, CBSA covers the instantiations of varied attention mechanisms, including softmax attention, linear attention, and channel attention, by taking different sets of representatives (see Section 3.3). As a result, CBSA serves as a unified formula for these attention mechanisms, and attributes their differences to their distinct information propagation (more precisely, compression) patterns induced by the different number and structure of representatives.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b21', 'b22']

Section: Matmul Projection

this section cite: []

Section: Contraction

this section cite: []

Section: Paper contributions
The contributions of the paper are summarized as follows.
1. We formulate an optimization objective that unifies the interpretability and efficiency of attention mechanisms through the idea of compressing all by contracting a few. 2. We derive an inherently interpretable and efficient attention mechanism, CBSA, which is a potential unified formula for different attention mechanisms. 3. We validate the interpretability and efficiency of CBSA through extensive experiments on visual tasks.
this section cite: []

Section: Notations and preliminaries
Notations. Given a positive integer n, let [n] . = {1, 2, . . . , n}. For s ≥ n, let O(s, n) ⊆ R s×n denote the set of s × n matrices with orthonormal columns, and O(s) . = O(s, s) denote the set of s × s orthogonal matrices. Let I n denote an identity matrix of size n, and O n denote a zero square matrix of size n. Given a vector v ∈ R n , let Diag(v) ∈ R n×n be a diagonal matrix with the entries of v along its diagonal. Let Z ∈ R d×N denote N input tokens represented in the ambient space R d . Specially, let Z ℓ denote these tokens feeding into the ℓ-th attention layer. The same holds for the representatives of input tokens, Q ∈ R d×m , where m can be much smaller than N .
this section cite: []

Section: Union of subspaces.
Although the union of nonlinear manifolds provides a better approximation [27], we adopt a much simpler structure: the union of (low-dimensional) linear subspaces. 2 Specifically, it is parameterized as K incoherent p-dimensional subspaces spanned by orthonormal bases U [K] . = {U k ∈ O(d, p)} K k=1 , where pK = d and
U ⊤ i U j = O p , ∀ i ̸ = j.
We refer to the i-th basis vector of U k as u ki . Similar to [23], we implement U [K] as learnable parameters in each attention layer and thus denote the U [K] implemented by the ℓ-th layer as U ℓ
[K] = {U ℓ k } K k=1 . Coding rate. To quantify the compactness, i.e., the extent to which tokens are compressed towards subspaces, we adopt the (lossy) coding rate [28], which measures how efficiently the token distribution can be covered by ϵ-balls under a given quantization precision ϵ > 0 (as illustrated in Fig. 1(a)). The coding rate of input tokens in the ambient space R d is defined as: R(Z) . = 1 2 log det
I N + d N ϵ 2 Z ⊤ Z .(1)
MCR 2 objective. The Maximal Coding Rate Reduction (MCR 2 ) [22] objective adopted in [23] is defined on the coding rate as follows:
max Z ∆R(Z) . = R(Z) expansion -R c (Z | U [K] ) compression -λ∥Z∥ 0 sparsity . = R(Z) - K k=1 R(U ⊤ k Z) -λ∥Z∥ 0 . (2
)
The input tokens are compressed towards K subspaces by the compression term, while expanded in the ambient space by the expansion term to avoid collapse, yielding compact and structured representation [22]. Yu et al. [23] have demonstrated that the approximated gradient step of the compression term in (2) corresponds an interpretable softmax attention mechanism.
this section cite: ['b26', 'b22', 'b27', 'b21', 'b22', 'b21', 'b22']

Section: Methods

this section cite: []

Section: Compressing all by contracting a few
Due to the existence of Gram matrix in (1), the attention mechanism derived from (2), which is called Multi-head Subspace Self-Attention (MSSA), inevitably scales quadratically with the number of input tokens. While previous work has bypassed this issue by replacing the Gram matrix with the covariance matrix [29] and further introduced a variational formulation [24], these strategies degenerate the token mixer into a channel mixer and channel attention, respectively.
In this paper, inspired by the concept of landmarks [30,31], we propose a simple but flexible approach to streamline the optimization of MCR 2 : compressing all input tokens by contracting a small number of representatives of them. Before demonstrating that this achieves linear complexity in N and prevents the aforementioned degeneration, we first formulate it as a new optimization objective.
An initial attempt is to impose a set of equality constraints on the coding rates as follows:
max Z R(Z) - K k=1 R(U ⊤ k Q) -λ∥Z∥ 0 s.t. R(U ⊤ k Q) = R(U ⊤ k Z), ∀ k ∈ [K],(3)
where representatives Q . = q(Z) are extracted from input tokens Z by a differential function q(•) : R d×N → R d×m . 3 This new objective in (3) is equivalent to the original objective in (2) but is more efficient to handle because the number of representatives (e.g., m = p = d /K) can be far smaller.
Since that the equality constraints in (3) is overly restrictive in practice, we attempt to relax them by introducing a tolerance τ , which uniformly bounds the absolute difference of the two coding rates within each subspace, i.e., |R(U ⊤ k Q) -R(U ⊤ k Z)| ≤ τ . Therefore, contracting the representatives will correspondingly compress the input tokens as well up to the tolerance τ .
Consequently, we have a relaxed optimization problem for our subsequent derivations as follows:
max Z R(Z) - K k=1 R(U ⊤ k Q) -λ∥Z∥ 0 s.t. |R(U ⊤ k Q) -R(U ⊤ k Z)| ≤ τ, ∀ k ∈ [K].(4)
In this paper, we employ an arguably simplest way to extract Q in each subspace: U ⊤ k Q = U ⊤ k ZA k , where A k ∈ R N ×m is the coefficient matrix over dictionary U ⊤ k Z ∈ R p×N , i.e., the projected representatives in each subspace are linear combinations of the projected input tokens.
this section cite: ['b28', 'b23', 'b29', 'b30']

Section: Contract-and-Broadcast Self-Attention
We now are ready to derive an attention mechanism in an ante-hoc interpretable manner, by implementing a gradient step of the compression term in the proposed objective as its forward pass. This methodology dates back its origins to the pioneering work [32], and is referred to as algorithm unrolling or unfolding [33].
Representative initialization and extraction. Inspired by the fact that cross-attention can be interpreted as approximating the coding rate [34], we employ this idea to extract representatives that satisfy the inequality constraints in (4), thus capturing the information-theoretic essence of input tokens. Specifically, we take an initial guess of the representatives as the query, and the input tokens as the key and value matrices, i.e.,
U ⊤ k Q = U ⊤ k Z softmax (U ⊤ k Z) ⊤ (U ⊤ k Q ini ) A k , ∀ k ∈ [K],(5)
where the initial guess Q ini is treated as a constant with respect to Z such that its strategy (whether input-dependent or not) does not affect the subsequent derivation. To be more specific, following [35], we initialize Q ini via an average pooling over the input tokens; see Appendix A for discussion. More importantly, the representatives extracted by this cross-attention operation are linear combinations of the input tokens, thus naturally leading to the form we desire. Therefore, the attention matrix in (5) effectively is the coefficient matrix A k .
this section cite: ['b31', 'b32', 'b33', 'b34']

Section: Representative contraction and contraction broadcast.
To derive the attention mechanism, following [23], we focus on optimizing the compression term via a gradient descent step. Having the extracted representatives satisfying the inequality constraints, we take a gradient descent step on the compression term of the objective function in (4) with respect to input tokens as follows:
Z ← Z -κ CBSA(Z | U [K]
) where ( 6)
CBSA(Z | U [K] ) . = K k=1 U k U ⊤ k Q I m + p mϵ 2 (U ⊤ k Q) ⊤ (U ⊤ k Q) -1 Contraction A ⊤ k Broadcast , (7
)
in which the step size parameter κ is learnable in our implementation. One can verify that (7) is proportional to the gradient of the compression term in (4) with respect to input tokens, while the contraction term in ( 7) is proportional to the gradient with respect to representatives. Hence, we refer to this formula as a Contract-and-Broadcast Self-Attention (CBSA), reflecting that: a) the contraction term gives the contracting directions of the representatives (abbreviated as contractions); b) the broadcast term, which reuses the attention matrix in (5), broadcasts the contractions back to all input tokens.
this section cite: ['b22']

Section: Contraction via self-attention.
To avoid computing the expensive matrix inverse in (7), similar to [23], we approximate it by a Gram matrix and a softmax function (i.e., an attention matrix):foot_3
CBSA(Z | U [K] ) ≈ U ⊤ k Q softmax (U ⊤ k Q) ⊤ (U ⊤ k Q) Contraction via self-attention A ⊤ k Broadcast .(8)
Note that the contraction term now effectively constitutes a self-attention operation in which the linear projections for the query, key, and value are all identical to the subspace basis, i.e., W query = W key = W value = U ⊤ k . By default, we implement CBSA via (8) rather than (7) in our experiments.
this section cite: ['b22', 'b6']

Section: Overview of CBSA.
The workflow of CBSA is illustrated in the left panel of Fig. 1. We also construct an inherently interpretable Contract-and-Broadcast Transformer (CBT) by stacking CBSA with the ISTA module [23], which is also derived via algorithm unrolling. 5 We report the computational complexities of CBSA and its sub-operations in Table 1 and compare the FLOPs of different attention mechanisms in Fig. 2 where d = 384, H = 6, and a patch size of 16 × 16. Provided N > 2d /H = 2p (typically 128 in Transformers), the FLOPs of CBSA are lower than those of MSSA. Further comparisons with other modules, including MHSA and MLP, are provided in Fig. 9.
this section cite: ['b22']

Section: Table 1:
Computational complexities. By default, we set m = p = d /H, where H = K denotes the number of attention heads (interpreted as subspaces in our case). The complexity of each sub-operation is computed by summing the costs across all heads. It is worth noting that the projection operations, which are essential to almost all attention mechanisms, confine the overall complexity at least O(N d 2 ).
Ω(MSSA) Ω(CBSA) sub-operations of CBSA Ω(extraction) / Ω(broadcast) Ω(contraction) Ω(projection) Attention [35] are omitted for simplicity, and ⊤ stands for the matrix transpose.
2N d 2 + 2N 2 d 2N d 2 + 3N md + 2m 2 d N md m 2 d N d 2
this section cite: ['b34']

Section: CBSA as a unified attention formula
In this subsection, we explore the potential of CBSA to serve as a unified formula for different attention mechanisms. Unlike recent work [37,38], CBSA encompasses a broader spectrum of mechanisms (see Fig. 3) in an interpretable and mathematically grounded manner.
Our analysis reveals that, CBSA (7) can derive multiple variants corresponding to existing attention mechanisms by varying the choice of representatives. In these variants, the distinct initialization, extraction, contraction, and broadcast steps of CBSA may not be explicitly observed, as some of them are simplified or fused together due to the specific number and structure of representatives, or engineering concerns.
this section cite: ['b36', 'b37']

Section: Softmax attention variant.
Obviously, the input tokens themselves satisfy the constraints in (4), and thus can be directly used as the representatives, i.e., Q = Z and m = N . In this case, we call the input tokens are self-expressed [39], where data samples are linearly represented over a dictionary composed of themselves. Then we can take a trivial solution for the regularization term where all coefficient matrices are identity matrices. Substituting them into (8) yields the following operator, known as MSSA in the white-box transformer [23]:
MSSA(Z | U [K] ) . = K k=1 U k U ⊤ k Z v.s. WvalueZ softmax (U ⊤ k Z) ⊤ (U ⊤ k Z) v.s. softmax((WkeyZ) ⊤ (WqueryZ)) .(9)
Linear attention variant. To analyze the case of orthogonal representatives, we start with a canonical choice: the principal directions of input tokens. We thus perform singular value decomposition (SVD) within each subspace:
U ⊤ k Z = L k Σ k R ⊤ k , ∀ k ∈ [K],(10)
where L k ∈ O(p), R k ∈ O(N, p), Σ k is a p × p diagonal matrix of singular values, and the columns of L k are known as the principal directions. Then, by right multiplying both sides by R k , we have:
U ⊤ k ZR k = L k Σ k R ⊤ k R k = L k Σ k , ∀ k ∈ [K].(11)
By comparing (11) with the way to form the representatives, i.e.,
U ⊤ k Q = U ⊤ k ZA k , we let U ⊤ k Q = L k Σ k and A k = R k .
Substituting them into (7) leads to the following operator:
K k=1 U k F (U ⊤ k Z)(U ⊤ k Z) ⊤ v.s. WvalueZϕ(WkeyZ) ⊤ U ⊤ k Z v.s. ϕ(WqueryZ) . = K k=1 U k L k I m + 1 ϵ 2 Σ 2 k -1 L ⊤ k U ⊤ k Z, (12
)
where F is a function defined on the spectrum of a positive semi-definite matrix and applies
f (λ i ) = ϵ 2 /(ϵ 2 +λi)
to each eigenvalues {λ i } p i=1 of the covariance matrix. 6 This operator highly resembles the linear attention [21], due to that it also factorizes the N × N attention matrix and multiplies the key and value first to linearize the computational complexity. 7 In Appendix B, we prove that a similar result holds for any set of orthogonal representatives.
this section cite: ['b38', 'b22', 'b10', 'b20']

Section: Channel attention variant.
Assuming that the basis vectors of U k are the principal directions for any set of input tokens (which is impossible but simplifies the computation), the directions of the representatives can be fixed along these basis vectors, i.e., U k = L k , thereby being orthogonal and input-agnostic (fixed). Then, (12) is simplified to:
K k=1 U k D k U ⊤ k Z, where D k . = Diag [f (u ⊤ ki Z)(u ⊤ ki Z) ⊤ ] p i=1 ,(13)
which basically recovers TSSA [24]. In (13), the feature channels are adaptively scaled according to their second moments of token projections, while channel attention typical employs an MLP to predict the channel-wise scaling factors [40,41].
this section cite: ['b23', 'b39', 'b40']

Section: Agent attention variant.
Agent Attention is basically a variant of CBSA with the contraction step removed 8 which can be perceived in Fig. 3. Although this removal appears to confine the token-mixing ability, it is compensated by the pooling-based initialization, which is also a token mixer [42].
To gain some intuition of the gap in expressive capacity among the aforementioned mechanisms, we illustrate their compression patterns in the right panel of Fig. 1. The channel attention variant is restricted to compressing input tokens along fixed axes parameterized by U [K] , whereas the linear attention variant compresses them along principal directions that are dynamically determined by the input. We argue that such a dynamism is crucial for in-context learning [7] and for mitigating superposition [43]. In contrast, softmax attention exhibits much greater flexibility, as it manipulates each token independently. Actually, its compression can be viewed as operating in an N -dimensional space, rather than in the d-dimensional feature space. Our proposed CBSA aims to approximate the behavior of softmax attention while significantly reducing computational cost.
The above findings can also be interpreted from a dictionary learning perspective, where the representatives correspond to the atoms of a dictionary. When the representatives are orthogonal and fixed, they form a complete dictionary; when they are orthogonal yet input-dependent, they resemble a submatrix of an overcomplete dictionary as in compressed sensing [44]. When the input tokens themselves serve as representatives, they constitute a self-expressive dictionary [39].
this section cite: ['b41', 'b6', 'b42', 'b43', 'b38']

Section: Experiments
In this section, we evaluate the interpretability and the efficiency of the proposed CBSA and the CBTs built upon CBSA. As natural images often lie on low-dimensional subspaces [45,46], we focus on classical visual tasks such as image classification and semantic segmentation, where higher resolutions generally lead to better accuracy [47,48].
Baseline and training configuration. We compare our CBSA to the vanilla softmax attention [49,1], and interpretable attention mechanisms based on MCR 2 , e.g., CRATE [23], ToST [24] and DEPICT [34]. Table 2 summarizes the baselines with brief descriptions. The results in gray are cited directly from the corresponding papers; whereas the others are reproduced under varied settings for fair comparisons. By default, the training configuration follows the baselines, with detailed information provided in Appendix C.
this section cite: ['b44', 'b45', 'b46', 'b47', 'b48', 'b0', 'b22', 'b23', 'b33']

Section: Implementation detail.
The projection back to the ambient space, which should theoretically be a left multiplication by U k , is over-parameterized with an independently learnable matrix. This strategy is also adopted in MSSA [23] and TSSA [24], and its effect has been analyzed in [36]. In short, although this relaxation compromises the theoretical rigour, it is crucial for achieving better accuracy.
In addition, the step size κ in ( 6) is implemented as a learnable parameter without constraining its sign. This allows the model to flexibly choose between compression and decompression. The PyTorch implementation is provided in Appendix D.
Table 2: Summary of baselines. Note that these methods are not limited to the tasks listed here, our descriptions only indicate their usages in the experiments of this paper.
this section cite: ['b22', 'b23', 'b35']

Section: Methods Attention Mechanism Complexity Interpretable Tasks
ViT [1] MHSA (softmax attention) quadratic ✕ image classification CRATE [23] MSSA (softmax attention) quadratic ✓ image classification ToST [24] TSSA (channel attention) linear ✓ image classification Agent Attention [35] Agent Attention (linear attention) linear ✕ image classification Segmenter [50] MHSA quadratic ✕ semantic segmentation DEPICT [34] MSSA quadratic ✓ semantic segmentation
this section cite: ['b0', 'b22', 'b23', 'b34', 'b49', 'b33']

Section: Advantages enabled by interpretability
In this subsection, we show superior advantages of CBSA over black-box attention mechanisms. The most essential aspect of our CBSA is its interpretability, which induces other desirable properties such as robustness and emergent segmentation. Compact and structured representation. Similar to MCR 2 [22], our optimization objective (4) aims to learn compact and structured representation by compressing input tokens towards low-dimensional subspaces. To confirm whether its iterative gradient steps can actually achieve this goal, we iterates the linear attention variant of CBSA (12) on synthetic data, where image tokens are modeled as the points in a three-dimensional space R 3 . Specifically, it is conducted on each class in the ambient space (thus being parameter-free) with a forward-only manner. As shown in Fig. 4, the representation ultimately admits a union of well-separated one-dimensional subspaces after 1024 iterations.
1 2 3 4 5 6 7 8 9 10 11 12 Layer index -300 400 500 600 700 800 Rc(Z U[K]) Coding rate (tiny models) CRATE TOST CBT (ours) 1 2 3 4 5 6 7 8 9 10 11 12 Layer index -0 500 1000 1500 2000 Rc(Z U[K]) Coding rate (small models) CRATE (w/o conv) 1 2 3 4 5 6 7 8 9 10 11 12 Layer index -220 240 260 280 300 320 Rc(Z U[K]) Normalized Coding rate (tiny models) 1 2 3 4 5 6 7 8 9 10 11 12 Layer index -400 600 800 1000 Rc(Z U[K]) Normalized Coding rate (small models) Compressing all by contracting a few. To confirm our interpretation that CBSA compresses all input tokens by contracting a few representatives, we check that: whether the input tokens are indeed compressed; if so, whether the compression is driven by contracting the representatives. We measure the compression term of (2) as well as its normalized variant 9 in Fig. 5. This normalized coding rate is invariant to in-place scaling and depends on the angles between input tokens. We observe two pieces of supporting evidences: a) the more compact the ultimate representation is, i.e., the compression term measured in the last layer is lower, the better the model performs on ImageNet-1K (see Table 3);foot_9 b) the latter half of the layers exhibit consecutive compression, in nearly all models. 11 Then, in Fig. 6, we measure the reduced coding rate of the input tokens and representatives, respectively, after they are processed by CBSA within each subspace. We observe that the two kinds of reduced coding rates show highly similar trends across most subspaces.
foot_11 L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 0 2 4 6 8 Reduction w.r.t. input tokens w.r.t. input tokens w.r.t. representatives 20 0 20 40 60 80 100 120 Reduction w.r.t. representatives Reduced coding rate across all layers and heads L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 4 2 0 2 4 6 8 10 Reduction w.r.t. input tokens w.r.t. input tokens w.r.t. representatives 50 0 50 100 150 Reduction w.r.t. representatives Reduced coding rate across all layers and heads
this section cite: ['b21', 'b11']

Section: Figure 7: Visualization of [CLS] attention map.
We empirically estimate the full attention matrix by
A k A ⊤ k ∈ R N ×N
and visualize the attention maps of the [CLS] token from the early, middle, and late layers of CBT, CRATE, and their hybrid model, respectively.
this section cite: []

Section: Emergent segmentation properties.
It has been reported that segmentation properties emerge in CRATE with merely standard supervised classification training owing to MSSA [51]. Compared to CRATE, our CBT attends to more semantically meaningful regions in the early layers, but the segmentation properties fail to persist in subsequent layers. as shown in Fig. 7. To address this phenomenon, we construct a hybrid model, termed CRATE+CBT, where the first half of the attention layers employ MSSA and the latter half employs CBSA. In this hybrid model, we observed that the segmentation properties not only emerged in the very first layer, but are also progressively enhanced in the following layers, rather than fading as in CBT. Qualitative results supporting this conclusion can be found in Appendix C.
0 20 40 60 80 100 GFLOPs (decoder only) 38.0 42.0 46.0 50.0 53.3 mIoU (%) Semantic segmentation CBT (ours) DEPICT CBT (ours) Segmenter 10 3 10 2 10 1 10 0 Noise level -0 10 20 30 40 mIoU Parameter robustness -tiny models ToST DEPICT CBT (ours) Segmenter 10 3 10 2 10 1 10 0 Noise level -0 10 20 30 40 mIoU Parameter robustness -small models Robustness against parameter perturbation. As the attention heads of CBSA are modeled as low-dimensional subspaces, perturbing their projection matrices (i.e., subspace bases) with relatively small noise does not significantly alter the subspaces they span [34]. Consequently, as shown in the right two panels of Fig. 8, CBSA is extremely robust against parameter perturbation, whereas the black-box method (i.e., Segmenter [50]) collapses under the same perturbation. Figure 9: Adapting pre-trained ViTs into CBSA style. We finetune both the adapted models and the original ViTs on ImageNet-1k for 50 epochs, and report the top-1 accuracy on the validation set. CBSA * leverages the three distinct projection matrices inherited from the pretrained MHSA to calculate query, key, and value, instead of using a single projection matrix as in (8). CBSA ∨ refers to the CBSA * without the contraction step, which is essentially an Agent attention.
this section cite: ['b50', 'b33', 'b49', 'b7']

Section: Evaluations on real-world visual tasks
We pretrain CBT models on the ImageNet-1k dataset, and finetune them on several downstream datasets. The top-1 accuracy on validation sets is reported in Table 3. In particular, our CBT-Small achieves comparable top-1 accuracy to ViT-S using only 30% of the parameters and 40% of the FLOPs. Compared to CRATE models, our CBTs (with convolutional embedding layers) perform remarkably better while using fewer parameters and FLOPs.
We also conduct a set of fair comparisons across different attention mechanisms, which can be regarded as variants of CBSA, and report the results in Table 4. In this setting, our CBT models remain competitive with CRATE while computing significantly fewer pairwise similarities. These results confirm that, from MSSA to CBSA and then to TSSA, the number of pairwise similarity computations decreases at the cost of performance sacrifice. In addition, we present the throughput comparisons on high-resolution images (e.g., 512 × 512) in Table 7 (Appendix C), showing that our methods consistently achieves superior training and inference efficiency.
To investigate the potential of applying CBSA to large-scale pretraining, we preliminarily finetune ViT models pretrained on ImageNet-21kfoot_12 by adapting their attention blocks into the CBSA style.
For comparison, we also adapt them into linear attention. Experimental results are shown in Fig. 9.
Although CBSA deviates more from MHSA than linear attention and is consequently harder to adapt from pretrained ViTs, CBSA achieves comparable performance to that of linear attention with nearly the same FLOPs. Interestingly, when the contraction step is removed, CBSA surpasses linear attention, which has been extensively investigated in [35].
For semantic segmentation, following the design of DEPICT [34], we build CBT decoders by stacking CBSA layers without the feed-forward modules on the top of ViT encoders. We evaluate the performance of them on the ADE20K dataset [53] and show the results in the left panel of Fig. 8. Clearly, our CBT decoder consistently surpasses both white-box (DEPICT) and black-box (Segmenter) counterparts that rely on softmax attention. In particular, the best-performing CBT decoder improves upon Segmenter by 1.5% mIoU while using merely 20% of the FLOPs and 0.06% of the pairwise similarities in the decoder.
this section cite: ['b34', 'b33', 'b52']

Section: Related work
Efficient attention mechanisms can be roughly divided into two categories: sparse attention and linear attention [54]. Approaches in sparse attention sparsify the attention matrix proactively by restricting the attention span to either random, or fixed [55,20], or learnable [56,57] patterns, or their combinations. Approaches in linear attention [21,58] decompose the attention matrix into a product of two low-rank matrices and thus avoids its explicit computation via the associative property of multiplication. The idea of using representative tokens has been applied to both. Global tokens or memory can be introduced in sparse attention to maintain global connectivity, thereby further shrinking the attention span [59,60]. Meanwhile, Agent tokens [35] and landmarks [30] can also be incorporated into linear attention from different perspectives.
Our CBSA distinguishes itself from previous efficient attention mechanisms by being inherently interpretable and derived from an optimization objective which efficiently compresses input tokens towards low-dimensional structures. Moreover, it can not only be viewed as sparse attention or linear attention for leveraging the representatives, but also mathematically generalizes softmax attention, linear attention, and channel attention as its special cases.
this section cite: ['b53', 'b54', 'b19', 'b55', 'b56', 'b20', 'b57', 'b58', 'b59', 'b34', 'b29']

Section: Conclusion
We have proposed an optimization objective for deriving attention mechanism and unifying the investigation towards the interpretability and the efficiency. By unrolling the gradient optimization steps of this objective, we derived an inherently interpretable and efficient attention mechanism, called Contract-and-Broadcast Self-Attention (CBSA). We found that our CBSA covers the instantiations of softmax attention, linear attention, and channel attention by changing the number and structure of representatives, thus revealing their fundamental connections. We validated the effectiveness of our CBSA through extensive experiments on visual tasks. We believe that the preliminary framework established in this work offers a promising direction for exploring a unified formula for existing attention mechanisms as well as new attention mechanisms in an inherently interpretable way.
this section cite: []

Section: References
Ref_id:b0 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b1 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b2 Title: BERT: pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b3 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b4 Title: Highly accurate protein structure prediction with AlphaFold Year: (2021)
Ref_id:b5 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b6 Title: Language models are few-shot learners Year: (2020)
Ref_id:b7 Title: Towards deeper vision transformer Year: (2021)
Ref_id:b8 Title: Attention is not all you need: pure attention loses rank doubly exponentially with depth Year: (2021)
Ref_id:b9 Title: Active-dormant attention heads: Mechanistically demystifying extreme-token phenomena in llms Year: (2024)
Ref_id:b10 Title: A survey of vision transformers in autonomous driving: Current trends and future directions Year: (2024)
Ref_id:b11 Title: The long-document transformer Year: (2020)
Ref_id:b12 Title: Swin transformer V2: scaling up capacity and resolution Year: (2022)
Ref_id:b13 Title: Centroid transformers: Learning to abstract with attention Year: (2021)
Ref_id:b14 Title: Attention-only transformers via unrolled subspace denoising Year: (2025)
Ref_id:b15 Title: Hopfield networks is all you need Year: (2021)
Ref_id:b16 Title: Is attention better than matrix decomposition Year: (2021)
Ref_id:b17 Title: In-context learning with transformer is really equivalent to a contrastive learning pattern Year: (2023)
Ref_id:b18 Title: Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead Year: (2019)
Ref_id:b19 Title: Generating long sequences with sparse transformers Year: (2019)
Ref_id:b20 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b21 Title: Learning diverse and discriminative representations via the principle of maximal coding rate reduction Year: (2020)
Ref_id:b22 Title: White-box transformers via sparse rate reduction Year: (2023)
Ref_id:b23 Title: Token statistics transformer: Linear-time attention via variational rate reduction Year: (2025)
Ref_id:b24 Title: See all by looking at a few: Sparse modeling for finding representative objects Year: (2012)
Ref_id:b25 Title: Self-representation based unsupervised exemplar selection in a union of subspaces Year: (2020)
Ref_id:b26 Title: Unsupervised manifold linearizing and clustering Year: (2023)
Ref_id:b27 Title: Segmentation of multivariate mixed data via lossy data coding and compression Year: (2007)
Ref_id:b28 Title: Redunet: A white-box deep network from the principle of maximizing rate reduction Year: (2022)
Ref_id:b29 Title: Nyströmformer: A nyström-based algorithm for approximating self-attention Year: (2021)
Ref_id:b30 Title: Landmark attention: Random-access infinite context length for transformers Year: (2023)
Ref_id:b31 Title: Learning fast approximations of sparse coding Year: (2010)
Ref_id:b32 Title: Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing Year: (2021)
Ref_id:b33 Title: Rethinking decoders for transformer-based semantic segmentation: A compression perspective Year: (2024)
Ref_id:b34 Title: Agent attention: On the integration of softmax and linear attention Year: (2024)
Ref_id:b35 Title: An in-depth investigation of sparse rate reduction in transformer-like models Year: (2024)
Ref_id:b36 Title: Native hybrid attention for efficient sequence modeling Year: (2025)
Ref_id:b37 Title: Log-linear attention Year: (2025)
Ref_id:b38 Title: Sparse subspace clustering Year: (2009)
Ref_id:b39 Title: Squeeze-and-excitation networks Year: (2018)
Ref_id:b40 Title: CBAM: convolutional block attention module Year: (2018)
Ref_id:b41 Title: Metaformer is actually what you need for vision Year: (2022)
Ref_id:b42 Title: Toy models of superposition Year: (2022)
Ref_id:b43 Title: Decoding by linear programming Year: (2005)
Ref_id:b44 Title: Generalized Principal Component Analysis Year: (2016)
Ref_id:b45 Title: The intrinsic dimension of images and its impact on learning Year: (2021)
Ref_id:b46 Title: An image is worth more than 16x16 patches: Exploring transformers on individual pixels Year: (2025)
Ref_id:b47 Title: Scaling laws in patchification: An image is worth 50 Year: (2025)
Ref_id:b48 Title: Attention is all you need Year: (2017)
Ref_id:b49 Title: Transformer for semantic segmentation Year: (2021)
Ref_id:b50 Title: Emergence of segmentation with minimalistic white-box transformers Year: (2024)
Ref_id:b51 Title: Pytorch image models Year: (2019)
Ref_id:b52 Title: Semantic understanding of scenes through the ade20k dataset Year: (2019)
Ref_id:b53 Title: Efficient attention mechanisms for large language models: A survey Year: (2025)
Ref_id:b54 Title: Image transformer Year: (2018)
Ref_id:b55 Title: Sparse sinkhorn attention Year: (2020)
Ref_id:b56 Title: Efficient content-based sparse attention with routing transformers Year: (2021)
Ref_id:b57 Title: Rethinking attention with performers Year: (2021)
Ref_id:b58 Title: Star-transformer Year: (2019)
Ref_id:b59 Title: Big bird: Transformers for longer sequences Year: (2020)
Ref_id:b60 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b61 Title: Vision transformers need registers Year: (2024)
Ref_id:b62 Title: Flatten transformer: Vision transformer using focused linear attention Year: (2023)
Ref_id:b63 Title: Symbolic discovery of optimization algorithms Year: (2023)
Ref_id:b64 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b65 Title: The pascal visual object classes challenge Year: (2011)
