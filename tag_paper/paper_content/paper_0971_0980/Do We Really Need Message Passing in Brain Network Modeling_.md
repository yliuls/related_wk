Title: Do We Really Need Message Passing in Brain Network Modeling?
Abstract: Brain network analysis plays a critical role in brain disease prediction and diagnosis. Graph mining tools have made remarkable progress. Graph neural networks (GNNs) and Transformers, which rely on the message-passing scheme, recently dominated this field due to their powerful expressive ability on graph data. Unfortunately, by considering brain network construction using pairwise Pearson's coefficients between any pairs of ROIs, model analysis and experimental verification reveal that the message-passing under both GNNs and Transformers can NOT be fully explored and exploited. Surprisingly, this paper observes the significant performance and efficiency enhancements of the Hadamard product compared to the matrix product, which is the matrix form of message passing, in processing the brain network. Inspired by this finding, a novel Brain Quadratic Network (BQN) is proposed by incorporating quadratic networks, which possess better universal approximation properties. Moreover, theoretical analysis demonstrates that BQN implicitly performs community detection along with representation learning. Extensive evaluations verify the superiority of the proposed BQN compared to the message-passing-based brain network modeling. Source code is available at https:  //github.com/LYWJUN/BQN-demo.

Section: Introductions
Brain network analysis provides a deep understanding of human brain organizations and assists in neurological disease diagnosis (Fornito et al., 2016;Cui et al., 2022). As a general-purpose language to model complex relationships, the research on graphs has a long history, ranging from graph theory (Bondy & Murty, 2008), to network science (Barabási, 2013), to network embedding (Cui et al., 2019). Along with the concept of the brain network, graph mining is widely used in brain network analysis, including concepts, characteristics, and models (Fornito et al., 2016;Chung, 2019). Network motif (Sporns & Kötter, 2004), small-world property (Bassett & Bullmore, 2006), and modularity (Meunier et al., 2009) are identified from the perspective of characteristics, while the persistent homology (Bendich et al., 2016), community detection (Garcia et al., 2018) and null model (Váša & Mišić, 2022) are employed to analyze brain network from the perspective of the model.
Graph Learning (Xia et al., 2021) plays an important role in diverse machine learning fields. (1) For tasks on graph data, Graph Neural Networks (GNNs), which combine graph topology and node attribute for representation, recently dominated this field (Wu et al., 2021b). They follow the message-passing scheme (Gilmer et al., 2017) by propagating node representation as a message over the graph. (2) For the domains without explicit graph structure, such as natural language processing (NLP) and computer vision (CV), implicit graph learning also shows amazing performance.
The key component Multi-Heads Self-Attention (MHSA) in Transformer (Vaswani et al., 2017) and its variant ViT (Dosovitskiy et al., 2021) constructs a fully connected graph between all token pairs and performs a holistic aggregation (Section 3.3). In conclusion, the message-passing scheme dominates many machine learning fields.
Interactions between brain regions are regarded as the key factors for neural development and disorder analysis. Functional MRI (fMRI) provides valuable information for exploring connectivity by capturing correlations between signal sequences of brain regions. Thus, it is direct to analyze brain networks, which are often constructed using pairwise Pearson correlation coefficients between any pairs of ROIs, using powerful graph learning tools, such as GNNs (Li et al., 2021;Cui et al., 2022;Bessadok et al., 2023) and Transformers (Kan et al., 2022b;Yu et al., 2024). Unfortunately, following model analysis and experiments motivate questioning this direct treatment in the brain network.
• GNN-based methods constructed node attributes from the brain network topology, which goes against GNNs' need for different types of information, and thus can't be fully exploited GNNs' characteristics (Fig. 1).
• Transformers employ the constructed brain network containing a holistic relationship between all ROIs, as initial token embedding. Thus the necessity to explore holistic relations with Transformers is weak (Fig. 1).
• Simple classifier with brain functional connectivity matrix as feature outperforms basic GNNs and Transformers (Fig. 2).
To enhance the performance of brain network analysis, the fundamental brain network operator should be updated. The recent breakthrough points out that the quadratic function can implement XOR logic operation and possesses better universal approximation properties compared to the linear function (Fan et al., 2018;2020). Inspired by this, the quadratic network is employed in brain network analysis to obtain the Brain Quadratic Network (BQN). BQN iteratively performs Hadamard product/element-wise product between the representation in the previous layer and the initial representation to get a new representation. Theoretical analysis reveals that BQN is equivalent to the updating rule of a community detection objective function based on nonnegative matrix factorization (NMF) of the adjacency matrix. Since the NMF of the adjacency matrix is a widelyused community detection paradigm, the proposed BQN seeks representation, which can capture mesoscopic community structure to reflect brain functional modules. The main contributions of this paper are summarized as follows.
• We investigate the rationality of the widely adopted message-passing in the brain network analysis.
• We propose a simple and effective Brain Quadratic Network (BQN) as the fundamental operator, with superior computational efficiency.
• We provide rigorous theoretical analysis connecting the proposed BQN with community detection.
• The proposed BQN achieves new SOTA on widelyused brain datasets.
this section cite: ['b18', 'b10', 'b7', 'b1', 'b11', 'b18', 'b9', 'b35', 'b2', 'b31', 'b3', 'b19', 'b37', 'b45', 'b20', 'b38', 'b14', 'b28', 'b10', 'b5', 'b52', 'b15', 'b16']

Section: Related Work
Recent advancements in brain network analysis have led to a surge in the application of graph-based learning methods, especially Graph Neural Networks and Graph Transformers.
this section cite: []

Section: Graph Neural Networks.
In recent studies, Graph Neural Networks (GNNs) have been leveraged to learn representations of brain regions, as well as the intricate patterns of functional connectivity within the brain. For example, BrainGNN (Li et al., 2021) utilizes ROI-aware GNNs and specialized pooling operators to identify critical nodes, improving the model's interpretability. BrainGB (Cui et al., 2022) systematically assesses various GNN designs on brain network data and introduces a practical pipeline that enhances GNN applications in brain research. FBNetGen (Kan et al., 2022a) constructs a learned task-oriented brain graph for downstream tasks. A-GCL (Zhang et al., 2023) proposes an adversarial self-supervised brain graph neural network by integrating graph contrastive learning with ROIs multiband information. However, these approaches often concentrate on local neighborhood aggregation, potentially neglecting the significance of interactions between non-adjacent brain regions.
this section cite: ['b28', 'b10', 'b54']

Section: Graph Transformers.
Graph Transformers are utilized in the analysis of brain networks, with the objective of capturing the holistic interaction of brain regions. Brain-NETTF (Kan et al., 2022b) employs classical Transformer encoders to generate ROI embeddings based on Pearson correlation matrices, along with an orthogonal clustered readout. ALTER (Yu et al., 2024) has specifically designed a brain graph Transformer to capture long-range dependencies among brain regions. ContrastPool (Xu et al., 2024) emphasizes attention on ROIs and subjects to learn a contrast graph, guiding the generation of brain representations through graph pooling. BioBGT (Peng et al., 2025) captures the small-world architecture in brain graphs through a network entanglement-based technique, emphasizing the biological characteristics of brain structure.
However, many of these methods often conflate the concepts of brain functional connectivity matrices and region of interest (ROI) features. The correlation coefficient matrix serves a dual purpose: it acts as both the adjacency matrix of the brain graph and the feature matrix. This duality contradicts established principles in graph research, where topology and features are typically regarded as two distinct types of information. This observation prompts us to design a simple yet effective network architecture specifically tailored for brain functional connectivity.
this section cite: ['b52', 'b47', 'b32']

Section: Preliminaries
This section provides notations, problem definition, and the concepts of GNNs and Transformers.
this section cite: []

Section: Notations and Problem Definition
Brain network, which models the connectivity between ROIs, can be represented as a graph G = (V, E, X), where V = {v 1 , v 2 , ..., v N } stands for the collection of N nodes, i.e., ROIs, E denotes the collection of edges between nodes, and X ∈ R N ×D represents the feature matrix with the i-th row x i ∈ R D as the feature of node/ROI v i . The graph topology is represented as the adjacency matrix A = [a ij ] ∈ R N ×N , where a i,j is the weight between nodes v i and v j . N (v) denotes the neighbourhoods of node v. D ∈ R N ×N stands for the degree matrix with diagonal element d vv = u∈N (v) a uv as the degree of node v.
this section cite: []

Section: Graph Neural Networks
Graph Neural Networks (GNNs) often follow the messagepassing paradigm (Gilmer et al., 2017;Yang et al., 2022;Zhuo et al., 2023), by iteratively aggregating node representations from neighborhoods and combining them with the representation of itself. By denoting H l as the collection of representations in the l-th layer, its v-th row h l v , i.e., the representation of node v in the l-th layer, can be obtained as
ĥl v := Aggregation l ({h l-1 u |u ∈ N (v)}),(1)
h l v := Combination l (h l-1 v , ĥl v ),(2)
where Aggregation(•) and Combination(, ) denote the aggregation and combination modules, respectively. Following the above paradigm, classical GNNs, e.g., GCN (Kipf & Welling, 2017)-one of representative GNNs employ the weighted average function ÃH l-1 to implement the above operations as
GCN : H l = σ( ÃH l-1 W l ),(3)
where σ denotes the non-linear activation function and H 0 = X denotes the initial node attributes. Ã = (D +
I n ) -1 2 (A + I n )(D + I n ) -1 2
represents the normalized adjacency matrix with self-loop.
this section cite: ['b20', 'b49', 'b55', 'b24']

Section: Transformers
Unlike GNNs that aggregate information from local neighborhoods, Transformers facilitate a holistic aggregation across all token pairs through the self-attention mechanism. The main component of Transformer (Vaswani et al., 2017) is Multi-Heads Self-Attention (MHSA). Given N tokens input Z = [z i ] N -1 i=0 ∈ R N ×F , which is the concatenation of the initial token embedding X and the position encoding P as Z = [X||P], the self-attention first maps the input features Z to query (Q), key (K), and value (V) vectors. Then, attention scores from query-key pairs are employed to aggregate the value vectors in a weighted manner, operating as a global message-passing mechanism. Specifically, a Self-Attention (SA) module can be formulated as
Q = ZW Q , K = ZW K , V = ZW V , Ẑ = Attention(Q, K, V) = softmax QK ⊤ √ F V,(4)
where W Q , W K , and W V ∈ R F ×F denotes trainable projection matrices and F stands for the feature dimensions.Further details of GNNs and Transformers are provided in the appendix A.
this section cite: ['b38']

Section: Graph Transformer.
To employ a Transformer to graph structure data, it is critical to incorporate graph topology appropriately (Zhuo et al., 2025). There are two kinds of strategies (Ying et al., 2021;Rampásek et al., 2022). Some methods directly employ adjacency matrix A to regularize the Transformer, e.g., element-wise product A ⊙ Sim QK . The other methods encode topology structure into position encoding P, e.g., the eigenvector of its Laplacian matrix. Thus, it is evident that the utilization of the Transformer for graph-data processing calls for a careful handling of graph topology and node representation, a consideration that stands out as especially striking in the context of brain network analysis.
this section cite: ['b56', 'b51', 'b33']

Section: Issues with Message Passing
This section begins with the method of brain network construction, followed by the analysis of the GNNs and Transformers on the constructed brain network. Finally, experiments are conducted to verify the above analysis.
this section cite: []

Section: Brain Network Construction
Refer to (Cui et al., 2022) for the detailed construction process of the brain network from raw data. Here, the final two steps, which are closely related to the brain network G, are considered. Firstly, the Brain Region Parcellation segments each subject into ROIs. Secondly, the weights of edges between ROIs are calculated using pairwise Pearson correlation coefficient as
r xy = n i=1 (x i -x)(y i -ȳ) n i=1 (x i -x) 2 n i=1 (y i -ȳ) 2 ,(5)
where {x i , ..., x n } and {y i , ..., y n } are two sequences of response values of two ROIs with the same length n, and x and ȳ are the means of sequences. To obtain a robust graph, a threshold is often used to sparsify the edge weights as
a xy = r xy , if r xy > threshold, 0, otherwise. (6
)
The constructed symmetric A = [a xy ] ∈ R N ×N is seen as the adjacency matrix of the brain network. Note that the sequence {x i , ..., x n } is infrequently characterized as the attribute of ROI, since its content is the response values at some moments instead of the essential characteristics of the ROI. Section 4.2 elaborates on how GNNs and Transformers construct node attributes.
this section cite: ['b10']

Section: Model Analysis
This section analyzes how GNNs and Transformers process brain networks constructed in Section 4.1.
GNNs for Brain Network. As shown in Section 3.2, GNNs learn representation by combining graph topology and node attributes, which are two different types of information.
Unfortunately, the constructed brain network often lacks essential node attributes. To alleviate this issue, existing GNN-based methods often constructed node attributes from the brain network topology, such as (1) identity matrix, (2) Eigenvectors of the adjacency matrix, (3) node degree, (4) local statistic of degree, (5) adjacency matrix itself or from another correlation measurement (Fig. 1). This strategy goes against GNNs' need for different types of information. Therefore, GNNs' characteristics, especially message passing, can't be fully exploited in brain network modeling.
Transformers for Brain Network. As shown in Section 3.3, Transformers learn holistic relations between all tokens with the self-attention mechanism. To employ Transformers in the brain network modeling, the initial token embedding is often set as the adjacency matrix, i.e., X = A. Since A is obtained by calculating Pearson correlation coefficients between all ROI pairs, the holistic relationship between all tokens is present in the initial token embedding X. Thus, the necessity of employing Transformers, especially the implicit message passing in the attention mechanism, to explore holistic relations in brain networks is weak (Fig. 1).
In summary, the message-passing under both GNNs and Transformers can't be fully explored and exploited by considering the characteristics of the brain networks.
this section cite: []

Section: Experimental Investigation
To verify the above analysis of the ability of GNNs and Transformers in brain networks, this section conducts basic experiments. The performances are measured according to AUC and ACC metrics on ABDIE and ADNI datasets, results of more metrics are placed in Appendix B. Three baselines are involved as follows:
• The simple classifier f (A) = AW, with A as the feature matrix and W as learnable matrix;
• The GNN as in Section 3.2 with A as topology and node attribute;
• The Transformer as in Section 3.3 with A as node attribute.
The performances are shown in Fig. 2. It can be observed that the basic and simple classifier consistently outperforms the other two models, i.e., GNN and Transformer. This result meets our analysis in Section 4.2 that message passing under both GNNs and Transformers can't be fully explored and exploited in brain network modeling. Note that some GNN-based and Transformer-based models, such as Brain 7KHQXPEHURIOD\HUV $8& QNN GCN 7KHQXPEHURIOD\HUV $&& QNN GCN 7KHQXPEHURIOD\HUV 6(1 QNN GCN 7KHQXPEHURIOD\HUV 63( QNN GCN Figure 3. Performance comparison of QNN (Hadamard product) and GCN (matrix product) on ABIDE dataset according to AUC, ACC, SEN, and SPE. Note that QNN consistently outperforms GCN, both the AUC and the ACC increase as the number of layers increases.
Network Transformer (Kan et al., 2022b), achieve much better performance compared to the basic GNN and Transformer. However, their high performances mainly rely on the modification to GNN and Transformer instead of the message-passing mechanism behind them.
this section cite: []

Section: Methodology
Brain Quadratic Network (BQN), which breaks the messagepassing scheme, is proposed here.
this section cite: []

Section: Motivations
The model analysis and experimental investigation in Sections 4.2 and 4.3 demonstrate that message passing under both GNNs and Transformers can't be fully explored and exploited in brain network modeling. Thus, to enhance the performance of brain network analysis, the fundamental brain network operator should be updated.
Here, the simple classifier f (A) = AW is employed as the base of the investigation for its satisfactory performance. For one-dimension output, f (•) acts as the linear function as f (a) = N i=1 w i a i = wa, where a = [a 1 , ..., a N ] and w = [w 1 , ..., w N ] denote one-row of A and one-column of W, respectively. It is well-known that linear functions can't implement XOR logic operation (Bishop & Nasrabadi, 2006). Recent attempts (Fan et al., 2018;2020) demonstrate the expressive ability of the Quadratic/Second-order function of the following forms a ⊙ a denotes the element-wise square operator, and ⊙ stands for Hadamard product product. It is proved that quadratic function can implement XOR logic operation and possesses some better universal approximation properties compared to linear function (Fan et al., 2020;2025). This motivates us to employ the quadratic function to enhance the performance of brain network analysis. Moreover, we provide further elaboration of Quadratic function and its representative applications in Appendix C.
h(a) = N i=1 w ri a i + b r N i=1 w gi a i + b g + N i=1 w oi a 2 i + c = (w r a + b r )(w g a + b g ) + w o (a 2 ) + c,(7)
To verify the effectiveness of the quadratic function with the Hadamard product, the following two brain network encoding methods are compared with H 0 = AW 0 as the initial embedding (0-th layer).
• Graph Convolution Network:
H l = σ(AH l-1 W l ),
• Quadratic Neural Network:
H l = H l-1 ⊙ (AW l ),
where ⊙ stands for Hadamard product/element-wise product. The performances of the Graph Convolution Network (GCN) and the Quadratic Neural Network (QNN) are shown in Fig. 3. It can be observed that the QNN significantly and consistently outperforms the GCN. Besides, both the AUC and the ACC increase as the number of layers increases.
In summary, both the expressive ability and universal approximation from theory and experimental results suggest the Quadratic Neural Network (QNN) as an alternative for brain network modeling.
this section cite: ['b6', 'b15', 'b16', 'b16']

Section: Brain Quadratic Network
According to Eq. ( 7), the Quadratic Network for Brain is formulated by using the Hadamard product as follows.
H l = H l-1 W l A ⊙ H l-1 W l B + H l-1 ⊙ H l-1 W l C ,(8)
where W l A and W l B are the learnable parameters for the l-th layer. To stabilize the learning process, the first term of Eq. ( 8) is replaced with H l-1 ⊙ AW l A , which meets the QNN formula in Section 5.1, and the Brain Quadratic Network(BQN) is as follows:
H l = H l-1 ⊙ AW l A + H l-1 ⊙ H l-1 W l H , (9
)
where learnable mappings W l A and W l H are for original brain network A and the embedding H. As shown in the experiments in Section 5.1, the first term in Eq. ( 9) plays a key role in performance enhancement, while the second one mainly tends to reduce the variance.
this section cite: []

Section: Theoretical Analysis
This section provides deep insight into the Brain Quadratic Network, especially the first term in Eq. ( 9), i.e.,
H l = H l-1 ⊙ AW l A .(10)
The following theorem provides an intuitive understanding from the concept of community detection.
Theorem 5.1. The iterative formula Eq. ( 10) of the proposed BQN is equivalent to the updating rule of the following community detection objective function, which is based on nonnegative matrix factorization (NMF) of the adjacency matrix.
min H≥0,W A ≥0 L(H) = ∥HW ⊤ A -A∥ 2 F ,(11)
where H ≥ 0 denotes all elements in H are non-negative.
Proof. The constrained optimization in Eq. ( 11) can be converted to an unconstrained one by employing a Lagrange multiplier (Bertsekas, 2014). To this end, a Lagrange multiplier matrix Θ is introduced corresponding to the nonnegative constraints H ≥ 0. Thus, the equivalent objective function is
min H L(H) = HW ⊤ A -A 2 F + tr(ΘH ⊤ ).
Set derivative of L(H) with respect to H to 0, it obtains
Θ = 2AW A -2HW ⊤ A W A .(12)
Following the KKT condition for the non-negativity of H, Eq. ( 12) can be further reformulated as element-wise form
2AW A -2HW ⊤ A W A ij H ij = Θ ij H ij = 0, (13
)
which is the fixed point equation that the solution must satisfy at convergence. Given an initial value of H, the updating rule of H can be formulated as
H ij ← H ij (AW A ) ij (HW ⊤ A W A ) ij ,(14)
where the denominator is to normalize the iterations, and ← denotes the assignment operation. Therefore, the iteration can be written as in matrix form
H = H ⊙ (AW A ),(15)
which is the same as the iterative formula of BQN.
The term ∥HW ⊤ A -A∥foot_1 F in Eq. ( 11) is the nonnegative matrix factorization of the adjacency matrix A. Nonnegative matrix factorization of the adjacency matrix is a widelyused community detection method (Yang & Leskovec, 2013;Wang et al., 2016;Luo et al., 2022), since its superior performance and outstanding interpretability. Therefore, the formula Eq. ( 10) leads the proposed BQN to seek representation, which can capture mesoscopic community structure to reflect brain functional modules. This provides an interpretation of the success of the proposed BQN. In addition to the theoretical analysis, experiments reflecting the clustering characteristics of brain region representations obtained through stacking multiple layers of Eq. ( 9) are presented in Appendix D.
this section cite: ['b4', 'b48', 'b40', 'b30']

Section: Evaluations

this section cite: []

Section: Experiments Setup
Datasets. In the experiments, two real-world fMRI datasets are employed: as follows.
• Autism Brain Imaging Data Exchange (ABIDE): This dataset is primarily utilized to investigate brain functional connectivity variation and structural differences associated with Autism Spectrum Disorder. The preprocessed data version can be accessed from the official websitefoot_0 . It collects resting-state functional magnetic resonance imaging data from 17 international sites, as well as anatomical and phenotypic data. The dataset we used contains 516 Autism Spectrum Disorder patients (ASD) and 493 normal controls (NC).
• Alzheimer's Disease Neuroimaging Initiative (ADNI): ADNI is a widely utilized multimodal neuroimaging repository focused on Alzheimer's disease. The raw images can be obtained from ADNI official website 2 . Access is limited and requires adherence to a request procedure to acquire the data. The dataset for this paper contains 53 Alzheimer's disease (AD) samples and 71 normal controls.
The construction of brain functional connectivity matrix for ABIDE is based on Craddock 200 atlas. The Pearson correlation coefficient between the region-averaged BOLD signals from pairs of ROIs (Regions of Interest) is adopted as the measure of functional connectivity strength between ROIs, namely as brain graph adjacency matrix (Cui et al., 2022;Bessadok et al., 2023;Xu et al., 2024). For ADNI, the fMRI data are first preprocessed using the Data Processing Assistant for Resting-State fMRI (DPARSF) toolkit. Next brain ROIs are defined based on AAL 90 atlas and the average time-series feature is calculated for each individual brain ROI. Pearson correlation coefficient between ROIs is then calculated, which serves as the functional connectivity matrix. Specifically, thresholds are set to keep edges with positive weights and drop those with negative weights. Baselines. Thirteen Baselines are compared in the experiments, which can be divided into two categories.
• Graph Neural Network (GNN)-based models, including two typical GNNs: GCN (Kipf & Welling, 2017) and GAT (Veličković et al., 2018)), and four brainspecific GNNs: BrainGNN (Li et al., 2021), BrainGB (Cui et al., 2022), FBNETGEN (Kan et al., 2022a) and A-GCL (Zhang et al., 2023).
• Graph Transformer (GT)-based models, including three typical GTs: SAN (Kreuzer et al., 2021), Graphormer (Ying et al., 2021) and GraphTrans (Wu et al., 2021a)), and four brain-specific GTs: Brain-NETTF (Kan et al., 2022b), ContrastPool (Xu et al., 2024), ALTER (Yu et al., 2024) and BioBGT (Peng et al., 2025).
this section cite: ['b10', 'b5', 'b47', 'b24', 'b39', 'b28', 'b10', 'b54', 'b26', 'b51', 'b47', 'b52', 'b32']

Section: Additional comparative experiments of BQN are conducted on supplemental datasets, please refer to Appendix E.
Metrics. To conduct comprehensive evaluations of performance, a combination of machine learning and medical diagnostic-specific metrics are employed, including Area Under the Receiver Operating Characteristic Curve (AUC), Accuracy (ACC), Sensitivity (SEN), and Specificity (SPE). AUC provides a threshold-independent measure of the model's ability to distinguish between classes. ACC is a straightforward metric to assess the model's classification performance. SEN, also known as the true positive rate, is a critical metric for medical diagnostics. In contrast, SPE represents the true negative rate.
this section cite: []

Section: Implementation Details.
The experiments are performed on a GeForce RTX3090 GPU. For all datasets, we employ random splits with the ratio 7:1:2 to get the training set, validation set and test set. Furthermore, the number of training epochs is set to 200 with batch size 16. An adam optimizer is adopted with initial learning rate of 10 -4 and weight decay as 10 -4 while training, target learning rate is from 10 -5 to 10 -4 . The activation function we selected is LeakyReLU. The number of layers, i.e., k is selected from 1, 2, 3, 4, 5 and the dropout rate is chosen from 0., 0.1, 0.2, 0.3. The results were averaged over 5 random runs. Notably, existing studies of GNNs and Transformers for brain networks show marked differences in data processing and model selection, which poses challenges in fairly evaluating their performance. To address this, this paper unifies these settings and compares methods in a fair manner. The details of corresponding unified settings are elaborated in the Appendix F.
this section cite: []

Section: Result Analysis
Brain Disorder Disease Classification. The performance comparison between BQN and the baseline models on the ABIDE and ADNI datasets are presented in Tab. 1. Upon observation, it is evident that compared to GNN-based and GT-based baselines, BQN exhibits superior performance in terms of AUC and ACC metrics. In particular, on the ABIDE dataset, BQN achieves an accuracy that is 2.43% higher than that of the runner-up baseline ALTER. This advantage can be attributed to the fact that BQN directly leverages the essential information from brain functional connectivity matrices. Therefore, it reduces model complexity and inference uncertainty that may arise from extraneous factors. Moreover, during training, BQN implicitly learns characteristic representations of brain functional modules, as evidenced in Sec. 5.3. These learned representations of functional regions enhance the accuracy of brain disorder classification. By directly modeling the intrinsic properties of brain functional connectivity, BQN is particularly well-suited for brain disorder classification tasks, especially ), where n denotes ROIs number. In contrast, both GNN-based and GT-based models require the computation of matrix multiplication between two square matrices (the propagation matrix and the feature matrix, e.g., Pearson matrices), which typically involves a cubic complexity of O(n 3 ).
this section cite: []

Section: Additional Experiments w/o Hadamard residual.
To validate the rationality and effectiveness of residuals, i.e., the second term proposed by Eq. ( 9). An ablation study is conducted with an architectural theme defined as H (l) = H (l-1) ⊙ AW l A with or without (H l-1 ⊙ H l-1 )W l H moduel, where H 0 = AW, W represents a linear layer. The comparison results are shown in Fig. 4. It can be seen that for the AUC, ACC, and SEN metrics, the addition of the Hadamard residual provides consistent performance improvements on both ABIDE and ADNI datasets. This illustrates the validity of the Hadamard residual. Besides, for AUC and ACC metrics, the standard deviations of the model results are smaller, indicating that the Hadamard residual also contributes to the stability of the model.
this section cite: []

Section: Hyperparameter Sensitivity Analysis.
This experiment is designed to offer an intuitive understanding of how to select the optimal number of layers k. The effect of this hyperparameter on the model performance is reported in D 7KHQXPEHURIOD\HUV $8& $%,'( $'1, E 7KHQXPEHURIOD\HUV $&& $%,'( $'1, Figure 5. Performance variations for varying k on two datasets.
Fig. 5. From this figure, it is evident that the proposed BQN achieves stable performance on these two datasets when the parameter k is within the range {1, 2, 3}. This demonstrates the stability of BQN. Moreover, the experimental results indicate that the model performance exhibits a gradual decline as the number of layers increases. Considering that brain datasets are generally of limited size (on the order of hundreds of samples), the observed performance degradation can be attributed to overfitting, which is likely induced by the excessive number of parameters and the heightened model complexity associated with deeper architectures.
this section cite: []

Section: Case Study.
To interpret the rationality of BQN on the ABIDE dataset, we construct two contrast brain graphs that reflect brain network differences among distinct groups. The first contrast brain graph is derived from the initial brain connectivity matrices, while another contrast brain graph is from the brain network generated by BQN. Specifically, the brain connectivity matrix template of ASD is obtained by averaging functional connectivity matrices within ASD groups,
A ASD Template = 1 n n i=1 A ASD i . Brain template of NC is obtained similarly, A N C Template = 1 m m i=1 A N C i
, where m denotes the number of normal subjects, and n represents the number of ASD patients in the ABIDE dataset. The contrast brain graph is then calculated as:
A contrast = A N C template ⊖ A ASD
Template , where ⊖ performs element-wise subtraction on the two input matrices. The learned contrast brain graph is then obtained by performing the same process as the equations above. The only difference is that the output of the last layer of a well-trained BQN is as brain graphs to construct the learnable contrast graph. Notably, the input of the well-trained BQN is a test set, and the phenotype of the subject is based on the result of the classification head of the model.  The visualization of the two contrast graphs, the top-20 edges with the largest weights are selected, is shown in Fig. 6. Firstly, it can be observed that compared to the initial contrast brain graph (subfigure a), the contrast graph learned by BQN is sparser and retains the core focus of abnormal connectivity, such as connectivity between paracentral lobule region and cingulum region. This finding indicates that the BQN learn a more holistic brain functional connectivity and gain ability of learning characteristic representations of brain functional modules, consistent with the theoretical analysis.
Moreover, our model focuses on the abnormal connections of the prefrontal cortex, cingulate, corpus callosum, and parietal as is shown in Fig. 6. Disruption of functional connectivity in these brain regions has been shown to be strongly associated with ASD patients (Assaf et al., 2010;Weng et al., 2010). This indicates that our model possesses a high degree of biological interpretability, rather than merely an improvement in performance.
this section cite: ['b0', 'b41']

Section: Conclusions
This paper observed that the message-passing mechanism, commonly used in brain network analysis, is redundant from both model analysis and experimental investigation. Based on this finding, BQN is introduced by employing Quadratic Network. BQN is a simple but novel model that adaptively learns brain functional networks while implicitly clustering connectivity between brain regions. Comprehensive experiments on the ABIDE and ADNI datasets demonstrate that BQN consistently outperforms models based on GNNs and Transformers. We hope this study could offer valuable insights into simpler models that can also achieve remarkable performance in brain disorder disease classification. 7KHQXPEHURIOD\HUV PLFUR) QNN GCN (a) QNN vs. GCN 7KHQXPEHURIOD\HUV PLFUR) $%,'( $'1, (b) Sensitivity of layers.
this section cite: []

Section: References
Ref_id:b0 Title: Abnormal functional connectivity of default mode sub-networks in autism spectrum disorder patients Year: (2010)
Ref_id:b1 Title:  Year: (1987)
Ref_id:b2 Title: Small-world brain networks Year: (2006)
Ref_id:b3 Title: Persistent homology analysis of brain artery trees Year: (2016)
Ref_id:b4 Title: Constrained optimization and Lagrange multiplier methods Year: (2014)
Ref_id:b5 Title: Graph neural networks in network neuroscience Year: (2023)
Ref_id:b6 Title: Pattern recognition and machine learning Year: (2006)
Ref_id:b7 Title: Graph theory Year: (2008)
Ref_id:b8 Title: Neurodegenerative brain network classification via adaptive diffusion with temporal regularization Year: (2024)
Ref_id:b9 Title: Brain network analysis Year: (2019)
Ref_id:b10 Title: Braingb: a benchmark for brain network analysis with graph neural networks Year: (2022)
Ref_id:b11 Title: A survey on network embedding Year: (2019)
Ref_id:b12 Title: Distinct brain networks for adaptive and stable task control in humans Year: (2007)
Ref_id:b13 Title: Prediction of individual brain maturity using fmri Year: (2010)
Ref_id:b14 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b15 Title: A new type of neurons for machine learning Year: (2018)
Ref_id:b16 Title: Universal approximation with quadratic deep networks Year: (2020)
Ref_id:b17 Title: On expressivity and trainability of quadratic networks Year: (2025)
Ref_id:b18 Title: Fundamentals of brain network analysis Year: (2016)
Ref_id:b19 Title: Applications of community detection techniques to brain graphs: Algorithmic considerations and implications for neural function Year: (2018)
Ref_id:b20 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b21 Title: FB-NETGEN: task-aware gnn-based fmri analysis via functional brain network generation Year: (2022)
Ref_id:b22 Title: Brain network transformer Year: (2022)
Ref_id:b23 Title: Towards efficient and interpretative rolling bearing fault diagnosis via quadratic neural network with bi-lstm Year: (2024)
Ref_id:b24 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b25 Title: Predict then propagate: Graph neural networks meet personalized pagerank Year: (2019)
Ref_id:b26 Title: Rethinking graph transformers with spectral attention Year: (2021)
Ref_id:b27 Title: Deeper insights into graph convolutional networks for semi-supervised learning Year: (2018)
Ref_id:b28 Title: Interpretable brain graph neural network for fmri analysis Year: (2021)
Ref_id:b29 Title: Attention-embedded quadratic network (qttention) for effective and interpretable bearing fault diagnosis Year: (2023)
Ref_id:b30 Title: Symmetric nonnegative matrix factorization-based community detection models and their convergence analysis Year: (2022)
Ref_id:b31 Title: Hierarchical modularity in human brain functional networks Year: (2009)
Ref_id:b32 Title: Biologically plausible brain graph transformer Year: (2025)
Ref_id:b33 Title: Recipe for a general, powerful, scalable graph transformer Year: (2022)
Ref_id:b34 Title: Multiscale graph transformer for brain disorder diagnosis Year: (2025)
Ref_id:b35 Title: Motifs in brain networks Year: (2004)
Ref_id:b36 Title: Understanding over-squashing and bottlenecks on graphs via curvature Year: (2022)
Ref_id:b37 Title: Null models in network neuroscience Year: (2022)
Ref_id:b38 Title: Attention is all you need Year: (2017)
Ref_id:b39 Title: Graph attention networks Year: (2018)
Ref_id:b40 Title: Semantic community identification in large attribute networks Year: (2016)
Ref_id:b41 Title: Alterations of resting state functional connectivity in the default network in adolescents with autism spectrum disorders Year: (2010)
Ref_id:b42 Title: Simplifying graph convolutional networks Year: (2019)
Ref_id:b43 Title: Representing long-range context for graph neural networks with global attention Year: (2021)
Ref_id:b44 Title: A comprehensive survey on graph neural networks Year: (2021)
Ref_id:b45 Title: Graph learning: A survey Year: (2021)
Ref_id:b46 Title: Data-driven network neuroscience: On data collection and benchmark Year: (2023)
Ref_id:b47 Title: Contrastive graph pooling for explainable classification of brain networks Year: (2024)
Ref_id:b48 Title: Overlapping community detection at scale: a nonnegative matrix factorization approach Year: (2013)
Ref_id:b49 Title: OPEN: orthogonal propagation with ego-network modeling Year: (2022)
Ref_id:b50 Title: Graph reciprocal neural networks by abstracting node as attribute Year: (2023)
Ref_id:b51 Title: Do transformers really perform badly for graph representation Year: (2021)
Ref_id:b52 Title: Long-range brain graph transformer Year: (2024)
Ref_id:b53 Title: An attention free transformer Year: (2021)
Ref_id:b54 Title: A-gcl: Adversarial graph contrastive learning for fmri analysis to diagnose neurodevelopmental disorders Year: (2023)
Ref_id:b55 Title: Propagation is all you need: A new framework for representation learning and classifier training on graphs Year: (2023)
Ref_id:b56 Title: Dualformer: Dual graph transformer Year: (2025)
