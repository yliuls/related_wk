Title: Scalable Cross-View Sample Alignment for Multi-View Clustering with View Structure Similarity
Abstract: Most existing multi-view clustering methods aim to generate a consensus partition across all views, based on the assumption that all views share the same sample arrangement. However, in real-world scenarios, the collected data across different views is often unsynchronized, making it difficult to ensure consistent sample correspondence between views. To address this issue, we propose a scalable sample-alignment-based multi-view clustering method, referred to as SSA-MVC. Specifically, we first employ a cluster-label matching (CLM) algorithm to select the view whose clustering labels best match those of the others as the benchmark view. Then, for each of the remaining views, we construct representations of nonaligned samples by computing their similarities with aligned samples. Based on these representations, we build a similarity graph between the non-aligned samples of each view and those in the benchmark view, which serves as the alignment criterion. This alignment criterion is then integrated into a late-fusion framework to enable clustering without requiring aligned samples. Notably, the learned sample alignment matrix can be used to enhance existing multi-view clustering methods in scenarios where sample correspondence is unavailable. The effectiveness of the proposed SSA-MVC algorithm is validated through extensive experiments conducted on eight real-world multi-view datasets.

Section: Introduction
Clustering aims to assign each sample to its corresponding class by leveraging the intrinsic similarities within the original data [1]. With the rapid advancement of science and technology, data have become increasingly diverse in their forms of representation. The same object can often be described from multiple perspectives. For instance, video content can be represented through audio, visual, and textual modalities. Such heterogeneous but complementary data representations are collectively referred to as multi-view data [2,3,4,5]. To fully exploit the rich semantic information embedded in multi-view data, a variety of advanced multi-view clustering algorithms have been developed in recent years [6,7,8,9]. These methods have demonstrated promising performance across a wide range of real-world applications by effectively integrating complementary information from multiple views.
Despite the effectiveness of these approaches in integrating multi-view information, they typically rely on the assumption of strict one-to-one correspondence among samples across different views, which is an idealized condition in practical applications [10,11,12]. In real-world scenarios, variations in Figure 1: The flowchart of the proposed method. First, the baseline view is selected based on the CLM criterion. Next, feature representations of the unaligned samples, denoted as {W v } V v=1 , are constructed. Subsequently, the cross-view similarity graphs {S v } V v=1 between the baseline view and the other views are established. Finally, these cross-view similarity graphs serve as alignment constraints within a late fusion multi-view clustering framework to obtain a unified partition matrix F * . The final clustering results are then derived by applying k-means clustering on F * . sample organization or ordering across views commonly lead to inconsistent or misaligned sample correspondences. To this end, some studies attempt to achieve sample alignment jointly with the learning of data representations. Given the effectiveness of the Hungarian algorithm in assignment problems [13], Huang et al. [14] integrated it into their clustering framework to facilitate sample alignment. However, due to semantic discrepancies between views and high intra-class similarity within views, establishing strict one-to-one alignment based solely on sample similarity limits tolerance to noise and misalignment. To address this, Yang et al. [15] proposed alignment at the class level, reformulating it as a class identification problem and introducing a noise-robust contrastive loss to improve robustness. Furthermore, Ren et al. [16] leveraged sample commonality and view diversity to adaptively construct alignment matrices and designed an unsupervised data completion mechanism to handle incomplete or unaligned data.
Although the aforementioned algorithms have shown promising performance in multi-view clustering with sample alignment, they still face several limitations. (1) Due to semantic discrepancies across views and the absence of supervision, establishing strict one-to-one correspondences is often difficult. In real-world scenarios, the sample relationships of different views are typically many-to-many, and enforcing strict matching may introduce noise and lead to sub-optimal alignment [17]. (2) While some recent methods employ joint learning frameworks that integrate alignment with feature representation to enhance performance, they often fail to model explicit alignment relationships, thus limiting their scalability to other multi-view clustering methods that are not applicable in sample non-alignment scenarios [18,19]. (3) Clustering performance is heavily influenced by the choice of a benchmark view, yet selecting an appropriate one remains an open challenge in current approaches [20].
Therefore, we propose a scalable multi-view clustering algorithm that integrates sample alignment into a unified clustering framework. To mitigate the impact of structurally noisy or disordered views, we first employ the CLM algorithm [21] to select the view that exhibits the highest structural consistency with the underlying semantic labels, designating it as the baseline. Considering that samples within the same subspace can be linearly reconstructed by their peers [22], we reformulate the alignment task as a similarity-based reconstruction problem rather than relying on rigid one-to-one index matching. Specifically, each view is structurally characterized by computing the similarity between unaligned samples and the rest of the view. Then, an alignment relationship is established by comparing these structural representations to that of the baseline view. Finally, the resulting alignment matrices are incorporated into a late fusion clustering framework, enabling effective alignment without the need for direct correspondence. Furthermore, the learned alignment relationship can be reused as auxiliary information to enhance the performance of existing multi-view clustering methods under misaligned conditions.
Overall, the main contributions of this paper are listed as follows:
• We propose to select the baseline view by measuring the similarity between sample cluster distributions and their corresponding labels within each view, effectively minimizing the impact of irrelevant or noisy structural information on the alignment process.
• We propose a structural representation for each view based on the correlation between nonaligned and aligned samples. This representation guides cross-view alignment by integrating sample-level features with intrinsic structural information.
• An alternating optimization algorithm is proposed to efficiently solve the model. Its effectiveness is validated through extensive experiments on eight multi-view datasets.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21']

Section: Preliminaries

this section cite: []

Section: Adaptive Neighbor Graph Learning
Graph-based multi-view clustering algorithms have attracted considerable attention in recent years due to their strong capability in capturing the intrinsic structural information embedded in the original data [23,24,25,26]. Based on the fact that samples within the same cluster or samples with smaller pairwise distances tend to exhibit higher similarity than those from different clusters, Nie et al. [27] proposed a clustering algorithm that constructs a nearest-neighbor graph to capture local structural relationships. The objective function of this method is formulated as follows:
min S n i=1 n j=1 ∥x i -x j ∥ 2 2 • s ij + βs 2 ij s.t. s ⊤ i 1 = 1, 0 ≤ s ij ≤ 1,(1)
where x i and x j denotes the i-th and j-th samples of the original data matrix X ∈ R n×d , where n is the total number of samples and d is the feature dimension. The variable s ij indicates the similarity between samples x i and x j . The parameter β is a regularization coefficient that balances the trade-off between the similarity graph learning and the sparsity of the graph, and it can be adaptively tuned during the optimization process. For a more detailed description of Eq. ( 1), please refer to [27].
this section cite: ['b22', 'b23', 'b24', 'b25', 'b26', 'b26']

Section: Late Fusion based Multi-view Clustering
Given multi-view datasets X v ∈ R n×dv , where d v denotes the feature dimensionality of the v-th view, late fusion-based multi-view clustering methods aim to extract partition-level clustering information from each view. A unified partition matrix is then obtained by integrating the partition information from all views. Specifically, assuming that the base partition matrices F v ∈ R n×du are obtained from X v via eigen-decomposition or other representation learning techniques, where d u denotes the latent feature dimension, the typical mathematical formulation can be expressed as follows [28]:
max F * Φ(F * , F v ) + λ Ψ(F * ),(2)
where Φ(•) denotes the partition fusion module, which integrates the base partitions into a unified one, and Ψ(•) represents a regularization term designed to preserve desirable properties such as smoothness [29], sparsity [30], or consistency across views [31,32].
this section cite: ['b27', 'b28', 'b29', 'b30', 'b31']

Section: Proposed Method

this section cite: []

Section: Cross Sample Similarity Learning
Late fusion-based multi-view clustering algorithms have attracted substantial attention due to their demonstrated effectiveness, and a variety of advanced methods have been proposed within this framework [33,34,35]. However, a common assumption in these algorithms is the existence of a strict one-to-one correspondence between samples across all views during the fusion of partition information. In practice, this assumption is often idealized. Due to temporal misalignment during data acquisition or storage constraints, mismatches between samples across different views frequently occur. Under such circumstances, directly fusion partition-level information without addressing the cross-view sample alignment may introduce irrelevant or inconsistent structural information, thereby degrading the quality of the unified partition and affecting the final clustering performance.
To overcome the limitations of traditional late fusion strategies, several algorithms have introduced a sample alignment matrix jointly with view-specific representation learning, integrating the two processes into a unified framework to enable mutual reinforcement [36,37,38]. In the context of unsupervised learning, sample alignment is typically inferred by exploiting the intrinsic feature similarities among data samples. However, due to the presence of substantial cross-view heterogeneity, the assumption of semantic consistency across views is often difficult to capture using rigid one-toone matching strategies. Such hard alignment approaches fail to model potential one-to-many or many-to-one semantic relationships across views, thereby overlooking alternative and potentially meaningful alignment relationships. Moreover, although some methods attempt to embed the sample alignment process into neural network-based representation learning, they often do not explicitly model the alignment relationships themselves, which limits the scalability of these approaches.
In light of the above challenges, we propose a novel strategy that utilizes the correlation between unaligned and aligned samples within each view as a view-specific structural representation. Specifically, given an unaligned multi-view dataset
X v = [X v 1 ; X v 2 ]
, where X v 1 ∈ R n1×dv and X v 2 ∈ R n2×dv denote the aligned and unaligned samples in the v-th view, respectively, and where n 1 and n 2 are the corresponding numbers of aligned and unaligned samples, we construct the structural representations of the unaligned samples for each view based on Eq. ( 1), i.e., min
{W v } V v=1 V v=1 n2 i=1 n1 j=1 X v 1[i,:] -X v 2[j,:] 2 2 w v ij + β (w v ij ) 2 s.t. w v⊤ i 1 = 1, 0 ≤ w v ij ≤ 1,(3)
where V denotes the total number of views. The matrix W v ∈ R n2×n1 represents the constructed feature representation for the v-th view in the presence of sample non-alignment. It is worth noting that the above construction of the feature representation for each view is not unique. Here we adopt the formulation given in Eq. (1) for simplicity. Nevertheless, alternative learning mechanisms could also be employed within our framework.
this section cite: ['b32', 'b33', 'b34', 'b35', 'b36', 'b37']

Section: Cross View Similarity Learning
After obtaining the feature representations for all views, a key challenge lies in constructing a reliable sample alignment across views. A straightforward approach is to randomly select one view as the baseline and align the remaining views to it. However, due to inevitable noise introduced during data collection, some views may contain structural information that does not reflect the true underlying cluster distribution. To mitigate the impact of such irrelevant or misleading information on the alignment process, we adopt the CLM algorithm to identify the most reliable baseline view. Specifically, the view that exhibits the highest consistency between its sample distribution structure and the semantic labels is selected as the baseline. The detailed selection process is defined as:
H(Y, X, d 2 ) = exp 1 σ d 2 n x∈X d 2 (x, y) exp 1 σ d 2 n k i=1 x∈Yi d 2 (x, y i ) × k i=1 |Y i |d 2 (y i , y) σ d 2 n (k -1)(4)
CLM (X) = 1 2 k 2 G⊆Y |G|=2 1 1 + exp -δ • H(G, X, d 2 )(5)
where Y = {Y 1 , Y 2 , • • • , Y k } denotes the ground-truth cluster assignment of the dataset X, and k is the total number of clusters. Let y i = Y i denote the mean of the samples in the i-th cluster, and c = X denote the mean of all samples. The function d 2 (•) represents the squared Euclidean distance, and σ d 2 = std(d 2 (x, c)|x ∈ X) denotes the standard deviation of the distances between the original data samples and the global centroid. The parameter δ is a pre-defined scaling factor. Based on the above formulation, we compute a matching score that quantifies the consistency between the structural distribution of samples and the corresponding semantic clusters in each view. The view with the highest matching score is then selected as the baseline. Accordingly, the cross-view structural similarity graph S v is constructed as follows:
min S v V v=1 v̸ =t ∥w t i -w v j ∥ 2 2 s v ij + β(s v ij ) 2 s.t. t = arg max v CLM (X v ), s ⊤ i 1 = 1, 0 ≤ s v ij ≤ 1, (6
)
where S v denotes the similarity graph that captures the structural correspondence between the unaligned samples in the v-th view and those in the baseline view, denoted by t. As a special case, when v = t, we define the similarity graph as the identity matrix, i.e., S t = I.
this section cite: []

Section: Sample-Aligned Late Fusion Strategy
In general, a higher similarity between samples implies a greater likelihood of a semantic match.
Based on this intuition, we propose to capture the matching criterion between unaligned samples across views by leveraging cross-view sample similarity. As discussed earlier, the ideal scenario assumes a strict one-to-one correspondence between samples across views. However, in practice, such hard 0-1 alignments are difficult to establish in the absence of external supervision, due to the presence of noise and structurally irrelevant information in certain views.
To address this challenge, we reformulate the alignment problem by reconstructing unaligned samples using samples within their corresponding subspace, rather than explicitly matching index positions across views. In this way, the alignment is achieved in a soft and structure-preserving manner. By integrating this alignment strategy with the late fusion-based multi-view clustering framework, we formulate the final objective function as follows:
max R v ,F * ,M v ,αv Tr F * ⊤ α t F t R t + V v=1 v̸ =t α v I 0 0 M v F v R v + λ V v=1 Tr(M v⊤ S v ) s.t. t = arg max v CLM (X v ), F * ⊤ F * = I, R v⊤ R v = I, V v=1 α 2 v = 1, M v⊤ M v = I,(7)
where M v denotes the sample realignment matrix that maps the unaligned samples in the v-th view to those in the baseline view, while R v represents the feature rotation matrix used to align the feature space. The scalar α v indicates the weight assigned to the v-th view, and λ is a hyperparameter that controls the trade-off between feature information and structural information.
this section cite: []

Section: Optimization

this section cite: []

Section: Optimization Algorithms
In this section, we develop an iterative optimization algorithm to solve the objective function in Eq. ( 7) with respect to the variables R v , F * , M v , and α v . The detailed optimization procedure is described as follows:
Update F * : When optimizing F * while keeping all other variables fixed, the objective function in Eq. ( 7) can be equivalently reformulated as:
max F * V v=1 Tr α v F * ⊤ 1 F v 1 R v + α v F * ⊤ 2 M v F v 2 R v s.t. F * ⊤ F * = I, F * = F * 1 F * 2 . (8
)
Since F * 1 and F * 2 are independent of each other, they can be optimized separately to obtain the complete solution for F * . Specifically, when optimizing the variable F * 1 , the objective function in Eq. ( 8) can be equivalently rewritten as:
max F * 1 V v=1 Tr(F * ⊤ 1 α v F v 1 R v ) s.t. F * ⊤ 1 F * 1 = I.(9)
The optimal solution to Eq. ( 9) can be obtained by performing singular value decomposition (SVD) on the matrix α v F v 1 R v . Since the optimization of F v 2 follows a procedure analogous to that of F v 1 , we omit the details here for brevity. Once the optimal solutions for both F v 1 and F v 2 are obtained, the final solution for F * is constructed by concatenating the two parts. Update R v : When all other variables are fixed, the optimization of R v in Eq. ( 7) can be equivalently reformulated as:
max R v α v Tr R v⊤ F v⊤ I 0 0 M v F * s.t. R v⊤ R v = I.(10)
Let Q v = α v F v⊤ I 0 0 M v F * , the optimal solution for the variable R v can then be obtained like that of F * 1 , specifically by performing singular value decomposition on Q v . Update α v : By fixing other variables, the Eq. ( 7) can be formulated as:
max α V v=1 α v γ v s.t. V v=1 α 2 v = 1,(11)
where
γ v = Tr(F * ⊤ C v F v R v ).
According to the Cauchy inequality, the optimal solution to the above optimization problem can be derived in closed form as:
α v = γ v V v=1 γ 2 v . (12
)
Update M v : By fixing F * , R v , and α v , M v can be optimized by solving the following subproblem:
max M v α v Tr M v⊤ F * 2 R v⊤ F v⊤ 2 + λTr M v⊤ S v s.t. M v⊤ M v = I.(13)
Let T = α v F * 2 R v⊤ F v⊤ 2 + λS v .
Following an optimization procedure similar to that for the variable M v , the optimal solution can be obtained by performing SVD on the matrix T.
In summary, the detailed procedure of the proposed method is described in the Appendix A.2.
this section cite: []

Section: Convergence Property
In the above optimization process, each subproblem is independent, and its corresponding optimal solution can be obtained. Consequently, the proposed algorithm converges within a few iterations according to Theorem 1. A detailed convergence proof is provided in the Appendix A.3. Theorem 1. The proposed optimization algorithm is guaranteed to converge to a local optimum of the SSA-MVC method.
this section cite: []

Section: Computational Complexity Analysis
In the proposed method, the primary computational complexity arises from three components: cross-sample similarity learning, cross-view similarity learning, and sample-aligned late fusion. Specifically, the computational cost for obtaining the feature representations
{W v } V v=1 ∈ R n2×n1 is O(V n 2 Kd max ),
where K denotes the number of neighbors and
d max = max{d 1 , d 2 , • • • , d V }
represents the maximum feature dimension across all views. For the CLM algorithm, the complexity is O(nd max ), while the construction of the cross-view similarity graph requires O(V n 2 n 1 K) operations. Finally, the computational cost of the late fusion step is O(n 2 ), mainly due to the generation of the partition matrix F v . Consequently, the overall computational complexity is O(n 2 ).
this section cite: []

Section: Experiments

this section cite: []

Section: Datasets
To further validate the effectiveness of the proposed method, we conduct experiments on eight real-world multi-view datasets, including Yale, 3sources, MSRCV, 100leaves, HW, Scene, EMNIST, and Hdigit. The detailed summary of them is provided in Appendix A.4.
this section cite: []

Section: Compared Methods
Ten state-of-the-art MVC methods are selected as baselines for comparison, including EEOMVC [39], DealMVC [40], MVCAN [41], EBMGC [42], Vsc_mH [43], OpVuC [44], DCMVC [45], LMTC [46], TMSL [47], DSTL [48]. The detailed introductions of them are presented in Appendix A.5.
this section cite: ['b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47']

Section: Experiments Setup
In the experiments, four widely used evaluation metrics are employed to assess the clustering performance of all compared methods: Accuracy (ACC), Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), and F1score. For the proposed method, we conducted a grid search over the set {0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000} to determine the best value for each dataset. The unified latent feature dimension d is set to the number of clusters. Regarding the baseline methods, parameters were tuned according to the ranges provided in their respective publicly available source codes, and the best results were selected in the experiments. To mitigate the influence of randomness on the experimental results, each experiment was repeated 20 times, and the mean and variance of the results are reported. All experiments are conducted on a Windows 11 PC equipped with an Intel Core i7-13700F CPU and 64GB RAM.
this section cite: []

Section: Results Analysis
To facilitate a fair comparison between the proposed method and existing approaches under the sample non-alignment setting, we fix the sample alignment ratio ρ to 50% in the main experiments.  Due to space constraints, results under other alignment ratios are provided in the Appendix and can be found in Tables 7-8 for reference. Notably, some baselines are not directly applicable to the non-aligned scenario. For fair evaluation, we apply the Hungarian algorithm to align the data before using these methods. Clustering results under the four evaluation metrics are shown in Tables 1234with the best and the second results highlighted in bold and underlined respectively. Methods that encounter memory overflow are marked as OOM. Based on the results reported in the Tables, several key observations can be obtained:
(1) The proposed algorithm consistently outperforms most baseline methods, including those using Hungarian-based sample alignment. For example, on the EMNIST and MSRCV datasets, it achieves ACC improvements of 10.66% and 17.83% over the second-best methods, EEOMVC and DCMVC, respectively. Similar gains are observed across other datasets, highlighting the method's effectiveness in capturing true cross-view sample correspondences and enhancing clustering performance.
(2) Our method is superior to existing methods such as Vsc_mH and OpVuC, which are designed for non-aligned sample clustering. These two kind of methods rely on mining alignment relationships directly from raw features without explicitly modeling the structural hierarchy within each view. Moreover, they employ a hard matching strategy, determining class correspondences based on pairwise sample similarity. Due to high intra-class similarity, this often results in unstable alignment matrices, adversely affecting algorithm convergence and leading to performance fluctuations.
(3) Compared with deep clustering methods such as DealMVC, MVCAN, and DCMVC, the proposed method demonstrates notable advantages. Although deep neural networks possess strong representation capabilities, they often rely on the assumption of consistent semantic information (a) Yale (b) 3sources (c) MSRCV (d) 100leaves (e) HW (f) Scene (g) EMNIST (h) Hdigit Figure 3: Clustering performance of the proposed method with varying values of the parameter λ.
across views. This assumption breaks down in the presence of sample misalignment, resulting in inconsistent feature learning and diminished clustering performance. Moreover, the use of the Hungarian algorithm for late fusion alignment does not consistently lead to performance gains and can even degrade results. This may be due to incorrect alignments introducing noisy or misleading information, ultimately impairing the effectiveness of the model.
this section cite: []

Section: Convergence and Parameter Sensitivity Analysis
In the previous section, we theoretically established that the proposed algorithm converges within a finite number of iterations. In this section, we further verify the convergence behavior empirically. The corresponding experimental results are illustrated in Fig. 2. As shown in the figure, the proposed method typically converges within approximately 10 iterations across all datasets, which empirically confirms its favorable convergence properties.
The results of our proposed method across varying λ values are presented in Fig. 3. Overall, the method demonstrates strong robustness to λ, with stable performance on most datasets. Notably, fluctuations on datasets like 3sources suggest higher sensitivity, may be can attributed to significant semantic divergence among views, which underscores the importance of appropriately weighting structural similarity.
this section cite: []

Section: Ablation Studies
We conduct ablation studies to assess the contribution of the proposed cross-view structural similarity module to clustering performance. Specifically, we denote the model without this module as SSA-MVC w/o CVS. The results, shown in Fig. 4, indicate that incorporating the module consistently improves sample alignment and clustering performance across most datasets. These findings highlight the effectiveness of the module and its integral role in the overall framework.
this section cite: []

Section: Effectiveness of the Alignment Strategy
To evaluate the scalability of our proposed method, we assess its effectiveness on the clustering algorithms that do not inherently handle sample misalignment. Specifically, under an alignment ratio of ρ = 50%, we use M to realign the originally non-aligned 100leaves multi-view data and compare the performance of several baseline algorithms on both the original and the realigned 100leaves. As shown in Table 5, our method can benefit the clustering performance of these algorithms in the non-aligned setting, demonstrating its effectiveness and potential for generalization to other methods.
this section cite: []

Section: Conclusion
This paper proposes a scalable multi-view clustering algorithm to tackle sample non-alignment. By selecting a baseline view via the CLM algorithm and leveraging structural similarities between aligned and non-aligned samples, the method guides cross-view alignment and integrates the resulting alignment matrix into a late fusion clustering framework. Experiments on eight benchmark datasets validate the effectiveness of the proposed method.
this section cite: []

Section: References
Ref_id:b0 Title: A survey of multi-view representation learning Year: (2018)
Ref_id:b1 Title: A comprehensive survey on multi-view clustering Year: (2023)
Ref_id:b2 Title: Reliable conflictive multiview learning Year: (2024)
Ref_id:b3 Title: Fast continual multi-view clustering with incomplete views Year: (2024)
Ref_id:b4 Title: On the adversarial robustness of multi-kernel clustering Year: ()
Ref_id:b5 Title: Multi-view clustering Year: (2004)
Ref_id:b6 Title: Robust multi-view clustering with incomplete information Year: (2022)
Ref_id:b7 Title: Multiple kernel clustering with adaptive multi-scale partition selection Year: (2024)
Ref_id:b8 Title: One-step multi-view clustering with diverse representation Year: (2024)
Ref_id:b9 Title: Gcfagg: Global and cross-view feature aggregation for multi-view clustering Year: (2023)
Ref_id:b10 Title: Mask-informed deep contrastive incomplete multi-view clustering Year: (2025)
Ref_id:b11 Title: Sparse low-rank multi-view subspace clustering with consensus anchors and unified bipartite graph Year: (2025)
Ref_id:b12 Title: The dynamic hungarian algorithm for the assignment problem with changing costs Year: (2007)
Ref_id:b13 Title: Partial multi-view clustering Year: (2014)
Ref_id:b14 Title: Partially view-aligned representation learning with noise-robust contrastive loss Year: (2021)
Ref_id:b15 Title: A novel federated multi-view clustering method for unaligned and incomplete data fusion Year: (2024)
Ref_id:b16 Title: Multi-view clustering: A survey Year: (2018)
Ref_id:b17 Title: Scalable incomplete multi-view clustering with structure alignment Year: (2023)
Ref_id:b18 Title: Multi-view clustering via high-order bipartite graph learning and tensor low-rank representation Year: (2025)
Ref_id:b19 Title: Deep safe multi-view clustering: Reducing the risk of clustering performance degradation caused by view increase Year: (2022)
Ref_id:b20 Title: Measuring the validity of clustering validation datasets Year: (2025)
Ref_id:b21 Title: Robust recovery of subspace structures by low-rank representation Year: (2012)
Ref_id:b22 Title: Gmc: Graph-based multi-view clustering Year: (2019)
Ref_id:b23 Title: Robust graphbased multi-view clustering Year: (2022)
Ref_id:b24 Title: Embedded feature selection on graph-based multi-view clustering Year: (2024)
Ref_id:b25 Title: From concrete to abstract: Multi-view clustering on relational knowledge Year: (2025)
Ref_id:b26 Title: Clustering and projected clustering with adaptive neighbors Year: (2014)
Ref_id:b27 Title: Multi-view clustering via late fusion alignment maximization Year: (2019)
Ref_id:b28 Title: Learning smooth representation for multi-view subspace clustering Year: (2022)
Ref_id:b29 Title: Multi-view clustering and feature learning via structured sparsity Year: (2013)
Ref_id:b30 Title: Consistent and specific multiview subspace clustering Year: (2018)
Ref_id:b31 Title: Towards resource-friendly, extensible and stable incomplete multi-view clustering Year: (2024)
Ref_id:b32 Title: Late fusion multi-view clustering via global and local alignment maximization Year: (2022)
Ref_id:b33 Title: One pass late fusion multi-view clustering Year: (2021)
Ref_id:b34 Title: Late fusion incomplete multi-view clustering Year: (2018)
Ref_id:b35 Title: Align then fusion: Generalized large-scale multi-view clustering with anchor matching correspondences Year: (2022)
Ref_id:b36 Title: Deep incomplete multi-view clustering with cross-view partial sample and prototype alignment Year: (2023)
Ref_id:b37 Title: How to construct corresponding anchors for incomplete multiview clustering Year: (2024)
Ref_id:b38 Title: Efficient and effective one-step multiview clustering Year: (2023)
Ref_id:b39 Title: Dealmvc: Dual contrastive calibration for multi-view clustering Year: (2023)
Ref_id:b40 Title: Investigating and mitigating the side effects of noisy views for self-supervised clustering algorithms in practical multi-view scenarios Year: (2024)
Ref_id:b41 Title: Ebmgc-gnf: Efficient balanced multi-view graph clustering via good neighbor fusion Year: (2024)
Ref_id:b42 Title: View-shuffled clustering via the modified hungarian algorithm Year: (2024)
Ref_id:b43 Title: One-pass view-unaligned clustering Year: (2024)
Ref_id:b44 Title: Dual contrast-driven deep multi-view clustering Year: (2024)
Ref_id:b45 Title: Large-scale multi-view tensor clustering with implicit linear kernels Year: (2025)
Ref_id:b46 Title: Tensor multi-subspace learning for robust tensor-based multi-view clustering. Knowledge-Based Systems Year: (2025)
Ref_id:b47 Title: Fast disentangled slim tensor learning for multi-view clustering Year: (2025)
Ref_id:b48 Title: Completer: Incomplete multi-view clustering via contrastive prediction Year: (2021)
