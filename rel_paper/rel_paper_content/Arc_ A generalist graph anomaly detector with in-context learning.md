Title: ARC: A Generalist Graph Anomaly Detector with In-Context Learning
Abstract: Graph anomaly detection (GAD), which aims to identify abnormal nodes that differ from the majority within a graph, has garnered significant attention. However, current GAD methods necessitate training specific to each dataset, resulting in high training costs, substantial data requirements, and limited generalizability when being applied to new datasets and domains. To address these limitations, this paper proposes ARC, a generalist GAD approach that enables a "one-for-all" GAD model to detect anomalies across various graph datasets on-the-fly. Equipped with incontext learning, ARC can directly extract dataset-specific patterns from the target dataset using few-shot normal samples at the inference stage, without the need for retraining or fine-tuning on the target dataset. ARC comprises three components that are well-crafted for capturing universal graph anomaly patterns: 1) smoothnessbased feature Alignment module that unifies the features of different datasets into a common and anomaly-sensitive space; 2) ego-neighbor Residual graph encoder that learns abnormality-related node embeddings; and 3) cross-attentive in-Context anomaly scoring module that predicts node abnormality by leveraging few-shot normal samples. Extensive experiments on multiple benchmark datasets from various domains demonstrate the superior anomaly detection performance, efficiency, and generalizability of ARC. The source code of ARC is available at https://github.com/yixinliu233/ARC.

Section: Introduction
Graph anomaly detection (GAD) aims to distinguish abnormal nodes that show significant dissimilarity from the majority of nodes in a graph. GAD has broad applications across various real-world scenarios, such as fraud detection in financial transaction networks [1] and rumor detection in social networks [2]. As a result, GAD has attracted increasing research attention in recent years [3,4,5,6,7,8]. Conventional GAD methods employ shallow mechanisms to model node-level abnormality [9,10,11]; however, they face limitations in handling high-dimensional features and complex interdependent relations on graphs. Recently, graph neural network (GNN)-based approaches have emerged as the go-to solution for the GAD problem due to their superior performance [4,6]. Some GNN-based GAD approaches regard GAD as a supervised binary classification problem and use specifically designed GNN architectures to capture anomaly patterns [6,12,13,14]. Another line of approaches targets the more challenging unsupervised paradigm, employing various unsupervised learning objectives and frameworks to identify anomalies without relying on labels [4,15,16,17]. Despite their remarkable detection performance, the existing GAD approaches follow a "one model for one dataset" learning paradigm (as shown in Fig. 1 (a) and (b)), necessitating dataset-specific training and ample training data to construct a detection model for each dataset. This learning paradigm inherently comes with the following limitations: ❶ Expensive training cost. For each dataset, we need to train a specialized GAD model from scratch, which incurs significant costs for model training, especially when dealing with large-scale graphs. ❷ Data requirements. Training a reliable GAD model typically needs sufficient in-domain data, sometimes requiring labels as well. The data requirements pose a challenge when applying GAD to scenarios with sparse data, data privacy concerns, or high label annotation costs. ❸ Poor generalizability. On a new-coming dataset, existing GAD methods require hyperparameter tuning or even model architecture modifications to achieve optimal performance, which increases the cost of applying them to new data and domains.
Given the above limitations, a natural question arises: Can we train a "one-for-all" GAD model that can generalize to detect anomalies across various graph datasets from different application domains, without any training on the target data? Following the trend of artificial general intelligence and foundation models, a new paradigm termed "generalist anomaly detection", originating from image anomaly detection, is a potential answer to this question [18]. As shown in Fig. 1 (c), in the generalist paradigm, we only need to train the GAD model once; afterward, the well-trained generalist GAD model can directly identify anomalies on diverse datasets, without any re-training or fine-tuning. Considering the diversity of graph data across different domains and datasets, the labels of few-shot normal nodes are required during the inference stage to enable the model to grasp the fundamental characteristics of the target dataset. Compared to conventional paradigms, the generalist paradigm eliminates the need for dataset-specific training, resulting in fewer computations, lower data costs, and stronger generalizability when applying GAD models to new datasets.
Nevertheless, due to the unique characteristics of graph data and GAD problem, it is non-trivial to design a generalist GAD approach. The challenge is three-fold: C1 -Feature alignment. Unlike image data, which are typically represented in a consistent RGB feature space, the feature dimensionality and semantic space can vary significantly across different graph datasets. Substituting features with unified representations generated by large language models may be a potential solution [19]; however, this approach is limited to specific feature semantics and cannot address more general cases [20]. C2 -Representation encoding. As the core of a generalist GAD model, a GNN-based encoder is expected to learn dataset-agnostic and abnormality-aware node embeddings for anomaly detection. However, in the absence of universal pre-trained foundation models [18] for graph data, crafting a potent encoder for a generalist GAD model presents a challenge. C3 -Few-shot sample-guided prediction. Existing GAD methods typically focus on single dataset settings, where dataset-specific knowledge is embedded in the model through training, enabling it to predict abnormality for each node independently. In contrast, a generalist GAD model should derive such knowledge from a small number of normal nodes. In this case, how to effectively utilize the few-shot normal samples during inference remains an open question.
To tackle these challenges, we introduce ARC, a generalist GAD approach based on in-context learning. ARC comprises three meticulously designed modules, each targeting a specific challenge. To address C1, we introduce a smoothness-based feature Alignment module, which not only standardizes features across diverse datasets to a common dimensionality but also arranges them in an anomalysensitive order. To deal with C2, we design an ego-neighbor Residual graph encoder. Equipped with a multi-hop residual-based aggregation scheme, the graph encoder learns attributes that indicate high-order affinity and heterophily, capturing informative and abnormality-aware embeddings across different datasets. Last but not least, to solve C3, we propose a cross-attentive in-Context anomaly scoring module. Following the in-context learning schema, we treat the few-shot normal nodes as context samples and utilize a cross-attention block to reconstruct the embeddings of unlabeled samples based on the context samples. Then, the reconstruction distance can serve as the anomaly score for each unlabeled node. In summary, this paper makes the following contributions:
• Problem. We, for the first time, propose to investigate the generalist GAD problem, aiming to detect anomalies from various datasets with a single GAD model, without dataset-specific fine-tuning.
• Methodology. We propose a novel generalist GAD method ARC, which can detect anomalies in new graph datasets on-the-fly via in-context learning based on few-shot normal samples.
• Experiments. We conduct extensive experiments to validate the anomaly detection capability, generalizability, and efficiency of ARC across multiple benchmark datasets from various domains.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b3', 'b5', 'b5', 'b11', 'b12', 'b13', 'b3', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b17']

Section: Related Work
In this section, we offer a brief review of pertinent related works, with a more extensive literature review available in Appendix A.
this section cite: []

Section: Anomaly Detection.
Anomaly detection (AD) aims to identify anomalous samples that deviate from the majority of samples [21]. Mainstream AD methods focus on unsupervised settings and employ various unsupervised techniques to build the models [22,23,24,25,26,27,28,29]. To enhance the generalizability of AD methods across diverse datasets, RegAD [30] considers few-shot setting and trains a single generalizable model capable of being applied to new in-domain data without re-training or fine-tuning. WinCLIP [31] utilizes visual-language models (VLMs, e.g., CLIP [32]) with well-crafted text prompts to perform zero/few-shot AD for image data. InCTRL [18], as the first generalist AD approach, integrates in-context learning and VLMs to achieve domain-agnostic image AD with a single model. However, due to their heavy reliance on pre-trained vision encoders/VLMs and image-specific designs, these approaches excel in AD for image data but face challenges when applied to graph data.
this section cite: ['b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b17']

Section: Graph Anomaly Detection (GAD).
In this paper, we focus on the node-level AD on graphs and refer to it as "graph anomaly detection (GAD)" following [6,33,34]. While shallow methods [9,10,11] show limitations in handling complex real-world graphs [4], the advanced approaches are mainly based on GNNs [35]. The GNN-based approaches can be divided into supervised and unsupervised approaches [3,5,7]. Supervised GAD approaches assume that the labels of both normal and anomalous nodes are available for model training [7]. Hence, related studies mainly introduce GAD methods in a binary classification paradigm [6,12,13,14,36,37]. In contrast, unsupervised GAD approaches do not require any labels for model training. They employ several unsupervised learning techniques to learn anomaly patterns on graph data, including data reconstruction [4,34,38], contrastive learning [15,39,40], and other auxiliary objectives [16,17,41,42]. Nevertheless, all the above methods adhere to the conventional paradigm of "one model for one dataset". Although some GAD approaches [43,44] can handle cross-domain scenarios, their requirement for high correlation (e.g., aligned node features) between source and target datasets limits their generalizability. Differing from existing methods, our proposed ARC is a "one-for-all" GAD model capable of identifying anomalies across target datasets from diverse domains, without the need for re-training or fine-tuning.
In-Context Learning (ICL). ICL enables a well-trained model to be effectively (fine-tuning-free) adapted to new domains, datasets, and tasks based on minimal in-context examples (a.k.a. context samples), providing powerful generalization capability of large language models (LLMs) [45,46,47] and computer vision (CV) models [18,48,49,50]. Two recent approaches, PRODIGY [51] and UniLP [52] attempt to use ICL for GNNs to solve the node classification and link prediction tasks, respectively. However, how to use ICL to deal with the generalist GAD problem where only normal context samples are available still remains open.
this section cite: ['b5', 'b32', 'b33', 'b8', 'b9', 'b10', 'b3', 'b34', 'b2', 'b4', 'b6', 'b6', 'b5', 'b11', 'b12', 'b13', 'b35', 'b36', 'b3', 'b33', 'b37', 'b14', 'b38', 'b39', 'b15', 'b16', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b17', 'b47', 'b48', 'b49', 'b50', 'b51']

Section: Problem Statement
Notations. Let G = (V, E, X) be an attributed graph with n nodes and m edges, where V = {v 1 , • • • , v n } and E are the set of nodes and edges, respectively. The node-level attributes are included by feature matrix X ∈ R n×d , where each row X i indicates the feature vector for node v i . The inter-node connectivity is represented by an adjacency matrix A ∈ {0, 1} n×n , where the i, j-th entry A ij = 1 means v i and v j are connected and vice versa.
this section cite: []

Section: … …
Smoothness-Based Feature Alignment
this section cite: []

Section: Ego-Neighbor Residual

this section cite: []

Section: Graph Encoder
Prop.
Prop.
MLP MLP MLP Shared Weight Shared Weight Cross-Attentive In-Context Anomaly Scoring 𝐖 𝑘 0.82 0.30 0.28 0.23 0.21 Feature Projection Smoothness-Based Feature Sorting 𝐙 [𝟎]
𝐙 [1]   𝐙 [2]   𝐑 [1]   𝐑 [2]   𝐇 Conventional GAD Problem. GAD aims to differentiate abnormal nodes V a from normal nodes V n within a given graph G = (V, E, X), where V a and
V n satisfy V a ∪ V n = V, V a ∩ V n = ∅,and
|V a | ≪ |V n |.
An anomaly label vector y ∈ {0, 1} n can be used to denote the abnormality of each node, where the i-th entry y i = 1 iff v ∈ V a and y i = 0 iff v ∈ V n . Formally, the goal of GAD is to learn an anomaly scoring function (i.e., GAD model) f : V → R such that f (v ′ ) > f (v) for ∀v ′ ∈ V a and ∀v ∈ V n . In the conventional GAD setting of "one model for one dataset", the GAD model f is optimized on the target graph dataset D = (G, y) with a subset of anomaly labels (in supervised setting) or without labels (in unsupervised setting). After sufficient training, the model f can identify anomalies within the target graph G during the inference phase.
this section cite: []

Section: Generalist GAD Problem.
In this paper, we investigate the generalist GAD problem, wherein we aim to develop a generalist GAD model capable of detecting abnormal nodes across diverse graph datasets from various application domains without any training on the specific target data. Formally, we define the generalist GAD setting, aligning it with its counterpart in image AD as introduced by Zhu et al. [18]. Specifically, let
T train = {D (1
)
train , • • • , D (N )
train } be a collection of training datasets, where each
D (i) train = (G (i) train , y (i)
train ) is a labeled dataset from an arbitrary domain. We aim to train a generalist GAD model f on T train , and f is able to identify anomalies within any test graph dataset D (i) test ∈ T test , where
T test = {D (1) test , • • • , D (N ′ )
test } is a collection of testing datasets. Note that T train ∩ T test = ∅ and the datasets in T train and T test can be drawn from different distributions and domains. Following [18], we adopt a "normal few-shot" setting during inference: for each D (i) test , only a handful of n k normal nodes (n k ≪ n) are available, and the model f is expected to predict the abnormality of the rest nodes without re-training and fine-tuning.
this section cite: ['b17', 'b17']

Section: ARC: A generalist GAD approach
In this section, we introduce ARC, a generalist GAD approach capable of identifying anomalies across diverse graph datasets without the need for specific fine-tuning. The overall pipeline of ARC is demonstrated in Fig. 2. Firstly, to align the features of different datasets, we introduce a smoothnessbased feature alignment module (Sec. 4.1), which not only projects features onto a common plane but also sorts the dimensions in an anomaly-sensitive order. Next, to capture abnormality-aware node embeddings, we propose a simple yet effective GNN model termed ego-neighbor residual graph encoder (Sec. 4.2), which constructs node embeddings by combining residual information between an ego node and its neighbors. Finally, to leverage knowledge from few-shot context samples for predicting node-level abnormality, we introduce a cross-attentive in-context anomaly scoring module (Sec. 4.3). Using the cross-attention block, the model learns to reconstruct query node embeddings based on context node embeddings. Ultimately, the drift distance between the original and reconstructed query embeddings can quantify the abnormality of each node.
this section cite: []

Section: Smoothness-Based Feature Alignment
Graph data from diverse domains typically have different features, characterized by differences in dimensionality and unique meanings for each dimension. For example, features in a citation network usually consist of textual and meta-information associated with each paper, whereas in a social network, the features may be the profile of each user. Therefore, in the first step, we need to align the features into a shared feature space. To achieve this, we introduce the feature alignment module in ARC, consisting of two phases: feature projection, which aligns dimensionality, and smoothnessbased feature sorting, which reorders features according to their smoothness characteristics.
this section cite: []

Section: Feature Projection.
At the first step of ARC, we employ a feature projection block to unify the feature dimensionality of multiple graph datasets [20]. Specifically, given a feature matrix X (i) ∈ R n (i) ×d (i) of D (i) ∈ T train ∪ T test , the feature projection is defined by a linear mapping:
X(i) ∈ R n (i) ×du = Proj X (i) = X (i) W (i) ,(1)
where X(i) is the projected feature matrix for D (i) , d u is a predefined projected dimension shared across all datasets, and W (i) ∈ R d (i) ×du is a dataset-specific linear projection weight matrix. To maintain generality, W (i) can be defined using commonly used dimensionality reduction approaches such as singular value decomposition [53] (SVD) and principal component analysis [54] (PCA).
this section cite: ['b19', 'b52', 'b53']

Section: Smoothness-Based Feature Sorting.
Although feature projection can align dimensionality, the semantic meaning of each projected feature across different datasets remains distinct. Considering the difficulty of semantic-level matching without prior knowledge and specific fine-tuning [19,20], in this paper, we explore an alternative pathway: aligning features based on their contribution to anomaly detection tasks. Through analytical and empirical studies, we pinpoint that the smoothness of each feature is strongly correlated with its contribution to GAD. Building on this insight, in ARC, we propose to sort the features according to their contribution as our alignment strategy. From the perspective of graph signal processing, Tang et al. [6] have demonstrated that the inverse of the lowfrequency energy ratio monotonically increases with the anomaly degree. In other words, high-frequency graph signals tend to play a more significant role in detecting anomalies. Similar findings have also been observed from the standpoint of spatial GNNs [37,55], where heterophily information has been shown to be crucial in discriminating anomalies. Motivated by these findings, can we develop a metric to gauge the contribution of each feature to GAD based on its frequency/heterophily? Considering its correlation to frequency [56] and heterophily [57,58,59,60], in this paper, we select featurelevel smoothness as the measure for contribution. Formally, given a graph G = (V, E, X) with a normalized feature matrix X, the smoothness of the k-th feature dimension is defined as:
s k (X) = - 1 |E| (vi,vj )∈E (X ik -X jk ) 2 ,(2)
where a lower s k indicates a significant change in the k-th feature between connected nodes, implying that this feature corresponds to a high-frequency graph signal and exhibits strong heterophily.
To verify whether smoothness can indicate the contribution of features in GAD, we further conduct empirical analysis (experimental setup and more results can be found in Appendix B). Concretely, we sort the raw features of each dataset based on the smoothness s k and divide them into 5 groups according to the percentile of s k . Then, we train different GAD models using each group of features separately, and the performance is shown in Fig. 3 and 8. On both datasets, a model-agnostic observation is that the features with lower s k are more helpful in discriminating anomalies. The consistent trend demonstrates the effectiveness of s k as an indicator of the role of features in GAD.
In light of this, given the projected features of different datasets, we can align their feature spaces by rearranging the permutation of features based on the descending order of s k w.r.t. each projected feature. For all datasets, the feature in the first column is the one with the lowest s k , which deserves more attention by ARC; conversely, features with less contribution (i.e. higher s k ) are placed at the end. In this way, the GNN-based model can learn to filter graph signals with different smoothness levels automatically and predict anomalies accordingly. During inference, the smoothness-related information remains transferable because we adhere to the same alignment strategy.
this section cite: ['b18', 'b19', 'b5', 'b36', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59']

Section: Ego-Neighbor Residual Graph Encoder
Once the features are aligned, we employ a GNN-based graph encoder to learn node embeddings that capture both semantic and structural information for each node. The learned embedding can be utilized to predict the abnormality of the corresponding node with the downstream anomaly scoring module. A naive solution is directly employing commonly used GNNs, such as GCN [61] or GAT [62], as the graph encoder. However, due to their low-pass filtering characteristic, these GNNs face difficulty in capturing abnormality-related patterns that are high-frequency and heterophilic [6,37]. Moreover, most GNNs, including those tailored for GAD, tend to prioritize capturing node-level semantic information while disregarding the affinity patterns of local subgraphs [17]. Consequently, employing existing GNN models as the encoder may overemphasize dataset-specific semantic knowledge, but overlook the shared anomaly patterns (i.e. local node affinity) across different datasets.
To address the above issues, we design an ego-neighbor residual graph encoder for ARC. Equipped with a residual operation, the encoder can capture multi-hop affinity patterns of each node, providing valuable and comprehensive information for anomaly identification. Similar to the "propagation then transformation" GNN architecture in SGC [63], our graph encoder consists of three steps: multi-hop propagation, shared MLP-based transformation, and ego-neighbor residual operation. In the first two steps, we perform propagation on the aligned feature matrix X ′ = X [0] for L iterations, and then conduct transformation on the initial and propagated features with a shared MLP network:
X [l] = ÃX [l-1] , Z [l] = MLP X [l] ,(3)
where l ∈ {0, • • • , L}, X [l] is the propagated feature matrix at the l-th iteration, Z [l] is the transformed representation matrix at the l-th iteration, and Ã is the normalized adjacency matrix [61,63]. Note that, unlike most GNNs that only consider the features/representations after L-iter propagation, here we incorporate both the initial features and intermediate propagated features and transform them into the same representation space. After obtaining Z [0] , • • • , Z [L] , we calculate the residual representations by taking the difference between Z [l] (1 ≤ l ≤ L) and Z [0] , and then concatenate the multi-hop residual representations to form the final embeddings:
R [l] = Z [l] -Z [0] , H = [R [1] || • • • ||R [L] ],(4)
where R [l] is the residual matrix at the l-th iteration, H ∈ R n×de is the output embedding matrix, and || denotes the concatenation operator.
Discussion. Compared to existing GNNs, our graph encoder offers the following advantages. Firstly, with the residual operation, the proposed encoder emphasizes the difference between the ego node and its neighbors rather than ego semantic information. This approach allows for the explicit modeling of local affinity through the learned embeddings. Since local affinity is a crucial indicator of abnormality and this characteristic can be shared across diverse datasets [17], the learned embeddings can offer valuable discriminative insights for downstream prediction. Second, the residual operation performs as a high-pass filter on the graph data, aiding ARC in capturing more abnormality-related attributes, i.e., high-frequency signals and local heterophily. Moreover, unlike existing approaches [15,17] that only consider 1-hop affinity, our encoder also incorporates higher-order affinity through the multi-hop residual design, which enables ARC to capture more complex graph anomaly patterns. More discussion and comparison to the existing GNNs/GAD methods are conducted in Appendix C.
this section cite: ['b60', 'b61', 'b5', 'b36', 'b16', 'b62', 'b60', 'b62', 'b16', 'b14', 'b16']

Section: Cross-Attentive In-Context Anomaly Scoring
To utilize the few-shot normal samples (denoted by context nodes) to predict the abnormality of the remaining nodes (denoted by query nodes), in ARC, we devise an in-context learning module with a cross-attention mechanism for anomaly scoring. The core idea of our in-context learning module is to reconstruct the node embedding of each query node using a cross-attention block to blend the embeddings of context nodes. Then, the drift distance between the original and reconstructed embeddings of a query node can serve as the indicator of its abnormality.
Specifically, we partition the embedding matrix H into two parts by indexing the corresponding row vectors: the embeddings of context nodes H k ∈ R n k ×de and the embeddings of query nodes H q ∈ R nq×de . Then, a cross-attention block is utilized to reconstruct each row of H q through a linear combination of H k :
Q = H q W q , K = H k W k , Hq = Softmax QK ⊤ √ d e H k ,(5)
where Q ∈ R nq×de and K ∈ R n k ×de are the query and key matrices respectively, W q and W k are learnable parameters, and Hq is the reconstructed query embedding matrix. Note that, unlike the conventional cross-attention blocks [64,65,66] that further introduce a value matrix V, our block directly multiplies the attention matrix with H k . This design ensures that Hq is in the same embedding space as H q and H k . Thanks to this property, given a query node v i , we can calculate its anomaly score f (v i ) by computing the L2 distance between its query embedding vector Hq i and the corresponding reconstructed query embedding vector Hq i , i.e., f
(v i ) = d(Hq i , Hq i ) = de j=1 Hq ij -Hq ij 2 .
Context Node Embedding Query Node Embedding Rec. Query Node Embedding Discussion. The design of cross-attentive in-context anomaly scoring follows a basic assumption: normal query nodes have similar patterns to several context nodes, and hence their embeddings can be easily represented by the linear combination of context node embeddings. Consequently, given a normal node, its original and reconstructed embeddings can be close to each other in the embedding space. In contrast, abnormal nodes may display distinct patterns compared to normal ones, making it difficult to depict their corresponding abnormal query embeddings using context embeddings. As a result, their drift distance s i can be significantly larger. Fig. 4 provides examples for the scenarios of (a) single-class normal and (b) multi-class normal 3 . In both cases, the drift distance (➞) can be a significant indicator for distinguishing anomaly (5) from normal nodes (1~4). Interestingly, if the attention matrix assigns uniform weights to all context nodes, then our scoring module becomes a one-class classification model [22]. This property ensures the anomaly detection capability of ARC even without extensive training. A detailed discussion is conducted in Appendix D.2.
this section cite: ['b63', 'b64', 'b65', 'b21']

Section: Model Training.
To optimize ARC on training datasets T train , we employ a marginal cosine similarity loss to minimize the drift distance of normal nodes while maximizing the drift distance of abnormal nodes. Specifically, given graph data with anomaly labels, we randomly select n k normal nodes as context nodes and sample an equal number of normal and abnormal nodes as query nodes. Then, given a query node v i with embedding Hq i , reconstructed embedding Hq i , and anomaly label y i , the sample-level loss function can be written by:
L =    1 -cos Hq i , Hq i , if y i = 0 max 0, cos Hq i , Hq i -ϵ , if y i = 1(6)
where cos(•, •) and max(•, •) denote the cosine similarity and maximum operation, respectively, and ϵ is a margin hyperparameter. Detailed algorithmic description and complexity analysis of ARC can be found in Appendix E.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets. To learn generalist GAD models, we train the baseline methods and ARC on a group of graph datasets and test on another group of datasets. For comprehensive evaluations, we consider graph datasets spanning a variety of domains, including social networks, citation networks, and e-commerce co-review networks, each of them with either injected anomalies or real anomalies [7,15,17]. Inspired by [52], we train the models on the largest dataset of each type and conduct testing on the remaining datasets. Specifically, the training datasets T train comprise PubMed, Flickr, Questions, and YelpChi, while the testing datasets T test consist of Cora, CiteSeer, ACM, BlogCatalog, Facebook, Weibo, Reddit, and Amazon. For detailed information, please refer to Appendix F.1.
Baselines. We compare ARC with both supervised and unsupervised methods. Supervised methods include two conventional GNNs, i.e., GCN [61] and GAT [62], and three state-of-the-art GNNs specifically designed for GAD, i.e., BGNN [67], BWGNN [6], and GHRN [37]. Unsupervised methods include four representative approaches with distinct designs, including the generative method DOMINANT [4], the contrastive method CoLA [15], the hop predictive method HCM-A [16], and the affinity-based method TAM [17]. For detailed information, refer to Appendix F.2.
this section cite: ['b6', 'b14', 'b16', 'b51', 'b60', 'b61', 'b66', 'b5', 'b36', 'b3', 'b14', 'b15', 'b16']

Section: Evaluation and Implementation.
Following [7,17,68], we employ AUROC and AUPRC as our evaluation metrics for GAD. We report the average AUROC/AUPRC with standard deviations across 5 trials. We train ARC on all the datasets in T train jointly, and evaluate the model on each dataset in T test in an in-context learning manner (n k = 10 as default). For the supervised baselines, we follow the same training and testing procedure (denoted as "pre-train only"), since no labeled anomaly is available for fine-tuning. For the unsupervised baselines, we consider two settings: "pre-train only" and "pre-train & fine-tune". In the latter, we additionally conduct dataset-specific fine-tuning with a few epochs. To standardize the feature space in baseline methods, we utilize either learnable projection or random projection as an adapter between the raw feature and the model input layer.
We utilize random search to determine the optimal hyperparameters for both the baselines and ARC. Since our goal is to train generalist GAD models, we do not perform dataset-specific hyperparameter search, but instead use the same set of hyperparameters for all testing datasets. More implementation details can be found in Appendix F.3.
this section cite: ['b6', 'b16', 'b67']

Section: Experimental Results
Performance Comparison. Table 1 shows the comparison results of ARC with baseline methods in terms of AUROC. Results in AUPRC are provided in Appendix G.1. We have the following observations. ❶ ARC demonstrates strong anomaly detection capability in the generalist GAD scenario, without any fine-tuning. Specifically, ARC achieves state-ofthe-art performance on 5 out of 8 datasets and demonstrates competitive performance on the remainder. On several datasets, ARC demonstrates significant improvement compared to the best baseline (e.g., ↑21.1% on Cora, ↑18.8% on CiteSeer, and ↑36.6% on Amazon). ❷ Simply pre-training the dataset-specific GAD methods typically results in poor generalization capability to new datasets. Specifically, the AUROC of the majority of "pre-train only" approaches is close to random guessing (50%) or even lower. ❸ With dataset-specific fine-tuning, the baseline methods achieve better performance in the majority of cases. However, the improvement can be minor or even negative in some cases, demonstrating the limitations of fine-tuning. Additionally, we observe that their performance is sometimes lower than the reported results from training models from scratch [4,15,17], indicating potential risk of negative transfer within the "pre-train & fine-tune" paradigm. ❹ Unsupervised baselines (except HCM-A) generally outperform the supervised ones, highlighting the difficulty of training a generalist GAD model using the binary classification paradigm.
this section cite: ['b3', 'b14', 'b16']

Section: Effectiveness of #Context Nodes.
To investigate how the number of context nodes n k affects the performance of ARC during inference, we vary n k within the range of 2 to 100. The results are shown in Fig. 5 (more results are in Appendix G.2). From the figure, we observe that the performance of ARC increases as more context nodes are involved, demonstrating its capability to leverage these labeled normal nodes with in-context learning. Furthermore, we can conclude that ARC is also label-efficient: when n k ≥ 10, the performance gain from using more context nodes becomes minor; moreover, even when n k is extremely small, ARC can still perform well on the majority of datasets.
this section cite: []

Section: Ablation Study.
To verify the effectiveness of each component of ARC, we make corresponding modifications to ARC and designed three variants: 1) w/o A: using random projection to replace smoothnessbased feature alignment; 2) w/o R: using GCN to replace ego-neighbor residual graph encoder; and 3) w/o C: using binary classification-based predictor and loss to replace crossattentive in-context anomaly scoring. The results are demonstrated in Table 2 (full results are in Appendix G.3). From the results, we can conclude that all three components significantly contribute to the performance. Among them, the in-context anomaly scoring module has a significant impact, as the performance of w/o C is close to random guessing on most datasets. The residual graph encoder also has a significant impact on the final performance. Notably, Weibo dataset is an exception where the GCN encoder performs better. A possible reason is that the Weibo dataset exhibits different anomaly patterns compared to the others.
G C N G A T B G N N B W G N N G H R N D O M I . C o L A H C M -A T A M A R C D O M I . C o L A H C M -A T A M10
this section cite: []

Section: Efficiency Analysis.
To assess the runtime efficiency of ARC, we compare the inference and fine-tuning time on the ACM dataset.
As depicted in Fig. 6, ARC demonstrates comparable runtime performance with the fastest GNNs (e.g., GCN and BWGNN), and significantly outperforms the unsupervised methods in terms of efficiency. Additionally, we observe that dataset-specific finetuning consumes more time compared to inference. Visualization. To investigate the weight allocation mechanism of the cross-attention module in ARC, we visualize the attention weights between context nodes and query nodes in Fig. 7 (additional results are in Appendix G.4). From Fig. 7(a), it is evident that ARC tends to assign uniform attention weights to normal nodes, leading to reconstructed embeddings that closely resemble the average embedding of the context nodes. Conversely, anomalies are reconstructed using a combination of 1 or 2 context nodes, suggesting that their embeddings are farther from the center. This allocation aligns with the case of "single-class normal" in Fig. 4(a). Differently, in Fig. 7(b), we observe that each normal query node is assigned to several context nodes following two fixed patterns, corresponding to the case of "multi-class normal" in Fig. 4(b). In summary, the cross-attention module enables ARC to adapt to various normal/anomaly distribution patterns, enhancing its generalizability.
this section cite: []

Section: Conclusion
In this paper, we take the first step towards addressing the generalist GAD problem, aiming to detect anomalies across diverse graph datasets with a "one-for-all" GAD model, without requiring dataset-specific fine-tuning. We introduce ARC, a novel and well-crafted in-context learning-based generalist GAD approach, capable of identifying anomalies on-the-fly using only few-shot normal nodes. Extensive experiments on real-world datasets from various domains demonstrate the detection prowess, generalizability, and efficiency of ARC compared to existing approaches. One limitation is that ARC can only use normal context samples during inference but cannot directly utilize abnormal context samples, even when they are available. A potential future direction could involve developing generalist GAD methods that utilize context samples containing both anomalies and normal instances.
this section cite: []

Section: References
Ref_id:b0 Title: Internet financial fraud detection based on graph learning Year: (2022)
Ref_id:b1 Title: Rumor detection on social media with bi-directional graph convolutional networks Year: (2020)
Ref_id:b2 Title: A comprehensive survey on graph anomaly detection with deep learning Year: (2021)
Ref_id:b3 Title: Deep anomaly detection on attributed networks Year: (2019)
Ref_id:b4 Title: Benchmarking unsupervised outlier node detection on static attributed graphs Year: (2022)
Ref_id:b5 Title: Rethinking graph neural networks for anomaly detection Year: (2022)
Ref_id:b6 Title: Gadbench: Revisiting and benchmarking supervised graph anomaly detection Year: (2024)
Ref_id:b7 Title: Wenzhong Guo, and See-kiong Ng. Towards effective federated graph anomaly detection via self-boosted knowledge distillation Year: (2024)
Ref_id:b8 Title: Scalable anomaly ranking of attributed neighborhoods Year: (2016)
Ref_id:b9 Title: Radar: Residual analysis for anomaly detection in attributed networks Year: (2017)
Ref_id:b10 Title: Anomalous: A joint modeling approach for anomaly detection on attributed networks Year: (2018)
Ref_id:b11 Title: Enhancing graph neural network-based fraud detectors against camouflaged fraudsters Year: (2020)
Ref_id:b12 Title: Spam review detection with graph convolutional networks Year: (2019)
Ref_id:b13 Title: Pick and choose: a gnn-based imbalanced learning approach for fraud detection Year: (2021)
Ref_id:b14 Title: Anomaly detection on attributed networks via contrastive self-supervised learning Year: (2021)
Ref_id:b15 Title: Hop-count based self-supervised anomaly detection on attributed networks Year: (2022)
Ref_id:b16 Title: Truncated affinity maximization: One-class homophily modeling for graph anomaly detection Year: (2023)
Ref_id:b17 Title: Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts Year: (2024)
Ref_id:b18 Title: One for all: Towards training one graph model for all classification tasks Year: (2024)
Ref_id:b19 Title: All in one and one for all: A simple yet effective method towards cross-domain graph pretraining Year: (2024)
Ref_id:b20 Title: Longbing Cao, and Anton Van Den Hengel. Deep learning for anomaly detection: A review Year: (2021)
Ref_id:b21 Title: Deep one-class classification Year: (2018)
Ref_id:b22 Title: Drocc: Deep robust one-class classification Year: (2020)
Ref_id:b23 Title: Towards total recall in industrial anomaly detection Year: (2022)
Ref_id:b24 Title: Anomaly detection with robust deep autoencoders Year: (2017)
Ref_id:b25 Title: Unsupervised anomaly detection with generative adversarial networks to guide marker discovery Year: (2017)
Ref_id:b26 Title: SSD: A unified framework for self-supervised outlier detection Year: (2021)
Ref_id:b27 Title: Perturbation learning based anomaly detection Year: (2022)
Ref_id:b28 Title: Deep orthogonal hypersphere compression for anomaly detection Year: (2024)
Ref_id:b29 Title: Registration based few-shot anomaly detection Year: (2022)
Ref_id:b30 Title: Winclip: Zero-/few-shot anomaly classification and segmentation Year: (2023)
Ref_id:b31 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b32 Title: Generative and contrastive self-supervised learning for graph anomaly detection Year: (2021)
Ref_id:b33 Title: Comga: Community-aware attributed graph anomaly detection Year: (2022)
Ref_id:b34 Title: A comprehensive survey on graph neural networks Year: (2020)
Ref_id:b35 Title: Bernnet: Learning arbitrary graph spectral filters via bernstein approximation Year: (2021)
Ref_id:b36 Title: Addressing heterophily in graph anomaly detection: A perspective of graph spectrum Year: (2023)
Ref_id:b37 Title: Anomalydae: Dual autoencoder for anomaly detection on attributed networks Year: (2020)
Ref_id:b38 Title: Graph anomaly detection via multi-scale contrastive learning networks with augmented view Year: (2023)
Ref_id:b39 Title: Gccad: Graph contrastive learning for anomaly detection Year: (2022)
Ref_id:b40 Title: Error-bounded graph anomaly loss for gnns Year: (2020)
Ref_id:b41 Title: Prem: A simple yet effective approach for node-level graph anomaly detection Year: (2023)
Ref_id:b42 Title: Cross-domain graph anomaly detection Year: (2021)
Ref_id:b43 Title: Cross-domain graph anomaly detection via anomaly-aware contrastive alignment Year: (2023)
Ref_id:b44 Title: Language models are few-shot learners Year: (2020)
Ref_id:b45 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b46 Title: Language models are general-purpose interfaces Year: (2022)
Ref_id:b47 Title: A unified sequence interface for vision tasks Year: (2022)
Ref_id:b48 Title: Ofa: Unifying architectures, tasks, and modalities through a simple sequenceto-sequence learning framework Year: (2022)
Ref_id:b49 Title: Uvim: A unified modeling approach for vision with learned guiding codes Year: (2022)
Ref_id:b50 Title: Prodigy: Enabling in-context learning over graphs Year: (2024)
Ref_id:b51 Title: Universal link predictor by in-context learning Year: (2024)
Ref_id:b52 Title: On the early history of the singular value decomposition Year: (1993)
Ref_id:b53 Title: Principal component analysis Year: (2010)
Ref_id:b54 Title: Alleviating structural distribution shift in graph anomaly detection Year: (2023)
Ref_id:b55 Title: Adagnn: Graph neural networks with adaptive frequency response filter Year: (2021)
Ref_id:b56 Title: Revisiting heterophily for graph neural networks Year: (2022)
Ref_id:b57 Title: Finding the missing-half: Graph complementary learning for homophily-prone and heterophily-prone graphs Year: (2023)
Ref_id:b58 Title: Beyond smoothing: Unsupervised graph representation learning with edge heterophily discriminating Year: (2023)
Ref_id:b59 Title: Graph neural networks for graphs with heterophily: A survey Year: (2022)
Ref_id:b60 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b61 Title: Graph attention networks Year: (2018)
Ref_id:b62 Title: Simplifying graph convolutional networks Year: (2019)
Ref_id:b63 Title: Attention is all you need Year: (2017)
Ref_id:b64 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b65 Title: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models Year: (2023)
Ref_id:b66 Title: Boost then convolve: Gradient boosting meets graph neural networks Year: (2021)
Ref_id:b67 Title: Toward deep supervised anomaly detection: Reinforcement learning from partially labeled anomaly data Year: (2021)
Ref_id:b68 Title: Padim: a patch distribution modeling framework for anomaly detection and localization Year: (2021)
Ref_id:b69 Title: Reconstruction by inpainting for visual anomaly detection Year: (2021)
Ref_id:b70 Title: Spatio-temporal autoencoder for video anomaly detection Year: (2017)
Ref_id:b71 Title: Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise Year: (2022)
Ref_id:b72 Title: Fast unsupervised anomaly detection with generative adversarial networks Year: (2019)
Ref_id:b73 Title: Cutpaste: Self-supervised learning for anomaly detection and localization Year: (2021)
Ref_id:b74 Title: Self-supervision improves diffusion models for tabular data imputation Year: (2024)
Ref_id:b75 Title: Catching both gray and black swans: Open-set supervised anomaly detection Year: (2022)
Ref_id:b76 Title: Few-shot scene-adaptive anomaly detection Year: (2020)
Ref_id:b77 Title: Anomaly detection under distribution shift Year: (2023)
Ref_id:b78 Title: Cross-domain video anomaly detection without target domain adaptation Year: (2023)
Ref_id:b79 Title: Netwalk: A flexible deep embedding approach for anomaly detection in dynamic networks Year: (2018)
Ref_id:b80 Title: Addgraph: Anomaly detection in dynamic graph using attention-based temporal gcn Year: (2019)
Ref_id:b81 Title: Anomaly detection in dynamic graphs via transformer Year: (2021)
Ref_id:b82 Title: Deep graph-level anomaly detection by glocal knowledge distillation Year: (2022)
Ref_id:b83 Title: Towards self-interpretable graph-level anomaly detection Year: (2024)
Ref_id:b84 Title: Unifying unsupervised graph-level anomaly detection and out-of-distribution detection: A benchmark Year: (2024)
Ref_id:b85 Title: On unsupervised graph out-of-distribution detection Year: (2023)
Ref_id:b86 Title: Towards test-time graph out-of-distribution detection Year: (2024)
Ref_id:b87 Title: Lg-fgad: An effective federated graph anomaly detection framework Year: (2024)
Ref_id:b88 Title: Towards data-centric graph machine learning: Review and outlook Year: (2023)
Ref_id:b89 Title: Learning strong graph neural networks with weak information Year: (2023)
Ref_id:b90 Title: Noise-resilient unsupervised graph representation learning via multi-hop feature quality estimation Year: (2024)
Ref_id:b91 Title: Cyclic label propagation for graph semi-supervised learning Year: (2022)
Ref_id:b92 Title: Rethinking and scaling up graph contrastive learning: An extremely efficient approach with group discrimination Year: (2022)
Ref_id:b93 Title: Improving in-context few-shot learning via self-supervised training Year: (2022)
Ref_id:b94 Title: Learning to learn in context Year: (2021)
Ref_id:b95 Title: Visual prompting via image inpainting Year: (2022)
Ref_id:b96 Title: Resgcn: Attention-based deep residual modeling for anomaly detection on attributed networks Year: (2022)
Ref_id:b97 Title: Collective classification in network data Year: (2008)
Ref_id:b98 Title: Arnetminer: extraction and mining of academic social networks Year: (2008)
Ref_id:b99 Title: Relational learning via latent social dimensions Year: (2009)
Ref_id:b100 Title: Collective opinion spam detection: Bridging review networks and metadata Year: (2015)
Ref_id:b101 Title: From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews Year: (2013)
Ref_id:b102 Title: Gcn-based user representation learning for unifying robust recommendation and fraudster detection Year: (2020)
Ref_id:b103 Title: What yelp fake review filter might be doing? Year: (2013)
Ref_id:b104 Title: Contrastive attributed network anomaly detection with data augmentation Year: (2022)
Ref_id:b105 Title: Predicting dynamic embedding trajectory in temporal interaction networks Year: (2019)
Ref_id:b106 Title: A critical look at the evaluation of gnns under heterophily Year: (2023)
Ref_id:b107 Title: Interactive anomaly detection on attributed networks Year: (2019)
Ref_id:b108 Title: Detecting anomalies in graphs Year: (2007)
Ref_id:b109 Title: Conditional anomaly detection Year: (2007)
