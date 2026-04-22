Title: Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection
Abstract: We reveal a one-class homophily phenomenon, which is one prevalent property we find empirically in real-world graph anomaly detection (GAD) datasets, i.e., normal nodes tend to have strong connection/affinity with each other, while the homophily in abnormal nodes is significantly weaker than normal nodes. However, this anomaly-discriminative property is ignored by existing GAD methods that are typically built using a conventional anomaly detection objective, such as data reconstruction. In this work, we explore this property to introduce a novel unsupervised anomaly scoring measure for GAD -local node affinity -that assigns a larger anomaly score to nodes that are less affiliated with their neighbors, with the affinity defined as similarity on node attributes/representations. We further propose Truncated Affinity Maximization (TAM) that learns tailored node representations for our anomaly measure by maximizing the local affinity of nodes to their neighbors. Optimizing on the original graph structure can be biased by nonhomophily edges (i.e., edges connecting normal and abnormal nodes). Thus, TAM is instead optimized on truncated graphs where non-homophily edges are removed iteratively to mitigate this bias. The learned representations result in significantly stronger local affinity for normal nodes than abnormal nodes. Extensive empirical results on 10 real-world GAD datasets show that TAM substantially outperforms seven competing models, achieving over 10% increase in AUROC/AUPRC compared to the best contenders on challenging datasets. Our code is available at https://github.com/mala-lab/TAM-master/.

Section: Introduction
Graph anomaly detection (GAD) aims to identify abnormal nodes that are different from the majority of the nodes in a graph. It has attracted great research interest in recent years due to its broad realworld applications, e.g., detection of abusive reviews or malicious/fraudulent users [9,16,32,37,57]. Since graph data is non-Euclidean with diverse graph structure and node attributes, it is challenging to effectively model the underlying normal patterns and detect abnormal nodes in different graphs. To address this challenge, graph neural networks (GNNs) have been widely used for GAD. The GNN-based methods are often built using a data reconstruction [8,11,31,59,65] or self-supervised learning [15,17,28,54,63] objective. The data reconstruction approaches focus on learning node representations for GAD by minimizing the errors of reconstructing both node attributes and graph structure, while the self-supervised approaches focus on designing a proxy task that is related to anomaly detection, such as prediction of neighbor hops [17] and prediction of the relationship between a node and a subgraph [28], to learn the node representations for anomaly detection.
These approaches, however, ignore one prevalent anomaly-discriminative property we find empirically in real-world GAD datasets, namely one-class homophily, i.e., normal nodes tend to have strong connection/affinity with each other, while the homophily in abnormal nodes is significantly weaker than normal nodes. This phenomenon can be observed in datasets with synthetic/real anomalies, as shown in Fig. 1(a) (see App. A for results on more datasets). The abnormal nodes do not exhibit homophily relations to each other mainly because abnormal behaviors are unbounded and can be drawn from different distributions. and abnormal nodes on two popular benchmarks, BlogCatalog [49] and Amazon [10]. The homophily of a given node is calculated using the number of nodes that have the same class label as the given node [13]. The local affinity is calculated on raw attributes (RA) and node representations learned by DGI [51] and TAM, respectively.
Motivated by the one-class homophily property, we introduce a novel unsupervised anomaly scoring measure for GAD -local node affinity -that assigns a larger anomaly score to nodes that are less affiliated with their neighbors.
Since we do not have access to class labels, we define the affinity in terms of similarity on node attributes/representations to capture the homophily relations within the normal class. Nodes having strong local affinity are the nodes that are connected to many nodes of similar attributes, and those nodes are considered more likely to be normal nodes. One challenge of using this anomaly measure is that some abnormal nodes can also be connected to nodes of similar abnormal behaviors. A straightforward solution to this problem is to apply our anomaly measure in a node representation space learned by off-the-shelf popular representation learning objectives like DGI [51], but the representations of normal and abnormal nodes can become similar due to the presence of non-homophily edges (i.e., edges connecting normal and abnormal nodes) that homogenize the normal and abnormal node representations in GNN message passing. To address this issue, we further propose Truncated Affinity Maximization (TAM) that learns tailored node representations for our anomaly scoring measure by maximizing the local node affinity to their neighbors, with the non-homophily edges being truncated to avoid the over-smooth representation issue. The learned representations result in significantly stronger local affinity for normal nodes than abnormal nodes. As shown in Fig. 1(b), it is difficult for using the local node affinity to distinguish normal and abnormal nodes on raw node attributes and DGI-based node representations, whereas the TAM-based node representation space offers well-separable local affinity results between normal and abnormal nodes. In summary, this work makes the following main contributions:
• We, for the first time, empirically reveal the one-class homophily phenomenon that provides an anomaly-discriminative property for GAD. Motivated by this property, we introduce a novel unsupervised anomaly scoring measure, local node affinity (Sec. 3.2).
• We then introduce Truncated Affinity Maximization (TAM) that learns tailored node representations for the proposed anomaly measure. TAM makes full use of the one-class homophily to learn expressive normal representations by maximizing local node affinity on truncated graphs, offering discriminative local affinity scores for accurate GAD.
and ANOMALOUS [41]. Matrix decomposition and residual analysis are commonly used in these methods, whose performance is often bottlenecked due to the lack of representation power to capture the rich semantics of the graph data and to handle high-dimensional node attributes and/or sparse graph structures.
GNN-based GAD methods have shown substantially better detection performance in recent years [32].
Although some methods are focused on a supervised setting, such as CARE-GNN [10], PCGNN [27], Fraudre [58], BWGNN [47], and GHRN [12], most of them are unsupervised methods. These unsupervised methods can be generally categorized into two groups: data reconstruction-based and self-supervised-based approach. Below we discuss these most related methods in detail.
this section cite: ['b8', 'b15', 'b31', 'b36', 'b56', 'b7', 'b10', 'b30', 'b58', 'b64', 'b14', 'b16', 'b27', 'b53', 'b62', 'b16', 'b27', 'b48', 'b9', 'b12', 'b50', 'b50', 'b40', 'b31', 'b9', 'b26', 'b57', 'b46', 'b11']

Section: Data Reconstruction-based Approach.
As one of the most popular methods, graph auto-encoder (GAE) is employed by many researchers to learn the distribution of normal samples for GAD. Ding et al. [8] propose DOMINANT, which reconstructs the graph structure and node attributes by GAE. The anomaly score is defined as the reconstruction error from both node attributes and its structure. It is often difficult for GAE to learn discriminative representations because it can overfit the given graph data. Several mechanisms, including attention, sampling, edge removing, and meta-edge choosing [9,27,30,45], are designed to alleviate this issue during neighborhood aggregation. Some recent variants like ComGA [31] and AnomalyDAE [11] incorporate an attention mechanism to improve the reconstruction. Other methods like SpaceAE [24] and ResGCN [39] aim to differentiate normal and abnormal nodes that do not have significant deviation, e.g., by exploring residual information between nodes. In general, reconstructing the structure of a node based on the similarities to its neighbors is related to but different from local node affinity in TAM (see Sec. 3.2), and GAE and TAM are also learned by a very different objective (see Sec. 3.3). Being GNN-based approaches, both the reconstruction-based approaches and our approach TAM rely on the node's neighborhood information to obtain the anomaly scores, but we explicitly define an anomaly measure from a new perspective, i.e., local node affinity. This offers a fundamentally different approach for GAD.
this section cite: ['b7', 'b8', 'b26', 'b29', 'b44', 'b30', 'b10', 'b23', 'b38']

Section: Self-supervised Approach.
Although data reconstruction methods can also be considered as a self-supervised approach, here we focus on non-reconstruction pre-text tasks for GAD. One such popular method is a proxy classification or contrastive learning task [18,53,65]. Liu et al. [28] propose CoLA, which combines contrastive learning and sub-graph extraction to perform selfsupervised GAD. Based on CoLA, SL-GAD is proposed to develop generative attribute regression and multi-view contrastive learning [63]. There are some methods that leverage some auxiliary information such degree, symmetric and hop to design the self-supervised task [6,17,23,42,56,61].
this section cite: ['b17', 'b52', 'b64', 'b27', 'b62', 'b5', 'b16', 'b22', 'b41', 'b55', 'b60']

Section: For example, Huang et al.
propose the method HCM-A [17] to utilize hop count prediction for GAD. Although some self-supervised methods construct the classification model on the relationship between the node and the contextual subgraph, this group of methods is not related to local node affinity directly, and its performance heavily depends on how the pre-text task is related to anomaly detection. Similar to the self-supervised approaches, the optimization of TAM also relies on an unsupervised objective. However, the self-supervised approaches require the use of some pre-text tasks like surrogate contrastive learning or classification tasks to learn the feature representations for anomaly detection. By contrast, the optimization of TAM is directly driven by a new, plausible anomaly measure, which enables end-to-end optimization of an explicitly defined anomaly measure.
this section cite: ['b16']

Section: Method

this section cite: []

Section: The Proposed TAM Approach
Problem Statement. We tackle unsupervised anomaly detection on an attributed graph. Specifically, let G = (V, E, X) be an attributed graph, where
V = {v 1 , • • • , v N } denotes its node set, E ⊆ V × V
with e ∈ E is the edge set, e ij = 1 represents there is a connection between node v i and v j , X ∈ R N ×M denotes the matrix of node attributes and x i ∈ R M is the attribute vector of v i , and A ∈ {0, 1} N ×N is the adjacency matrix of G with A ij = 1 iff (v i , v j ) ∈ E, then GAD aims to learn an anomaly scoring function
f : V → R, such that f (v) < f (v ′ ), ∀v ∈ V n , v ′ ∈ V
a , where V n and V a denotes the set of normal and abnormal nodes, respectively. Per the nature of anomaly, it is typically assumed that |V n | ≫ |V a |. But in unsupervised GAD we do not have access to the class labels of the nodes during training. In this work, our problem is to learn an unsupervised local node affinity-based anomaly scoring function f . (a) TAM leverages the observation that normal nodes have stronger affinity relations to their neighbors than anomalies to learn an unsupervised GAD model. It learns a set of affinity maximization GNNs (i.e., LAMNet) on a set of sequentially truncated graphs yielded by our probabilistic graph truncation method NSGT. We build an ensemble of TAM models to make use of the randomness in NSGT for more effective GAD. (b) NSGT iteratively removes edges with a probability proportional to the distance between the connected nodes.
this section cite: []

Section: Our Proposed Framework.
Motivated by the one-class homophily phenomenon, we introduce a local node affinity-based anomaly score measure. Since it is difficult to capture and quantify the one-class homophily in the raw attribute space and the generic node representation space, we introduce a truncated affinity maximization (TAM) approach to learn a tailored representation space where the local node affinity can well distinguish normal and abnormal nodes based on the one-class homophily property. As shown in Fig. 2, TAM consists of two novel components, namely local affinity maximization-based graph neural networks (LAMNet) and normal structure-preserved graph truncation (NSGT). LAMNet trains a graph neural network using a local affinity maximization objective on an iteratively truncated graph structure yielded by NSGT.
NSGT is designed in a way through which we preserve the homophily edges while eliminating non-homophily edges iteratively. The message passing in LAMNet is then performed using the truncated adjacency matrix rather than the original one. In doing so, we reinforce the strong affinity among nodes with homophily relations to their neighbors (e.g., normal nodes), avoiding the potential bias caused by non-homophily edges (e.g., connections to abnormal nodes). The output node affinity in LAMNet is used to define anomaly score, i.e., the weaker the local affinity is, the more likely the node is an abnormal node. There exists some randomness in our NSGT-based graph truncation. We utilize those randomness for more effective GAD by building a bagging ensemble of TAM.
this section cite: []

Section: Local Node Affinity as Anomaly Measure
As shown in Fig. 1, normal nodes have significantly stronger homophily relations with each other than the abnormal nodes. However, we do not have class label information to calculate the homophily of each node in unsupervised GAD. We instead utilize the local affinity of each node to its neighbors to exploit this one-class homophily property for unsupervised GAD. The local affinity can be defined as an averaged similarity to the neighboring nodes, and the anomaly score f is opposite to the affinity:
h(v i ) = 1 |N (v i )| vj ∈N(vi) sim (x i , x j ) ; f (v i ) = -h(v i ),(1)
where N (v i ) is the neighbor set of node v i and sim(x i , x j ) =
x T i xj ∥xi∥∥xj ∥ measures the similarity of a node pair (v i , v j ). The local node affinity h(v i ) is a normal score: the larger the affinity is, the stronger homophily the node has w.r.t. its neighboring nodes based on node attributes, and thus, the more likely the node is a normal node.
The measure in Eq. ( 1) provides a new perspective to quantify the normality/abnormality of nodes, enabling a much simpler anomaly scoring than existing popular measures such as the reconstruction error f (v i ) = (1 -α)∥a i -a i ∥ 2 + α∥x i -x i ∥ 2 , where a i denotes the neighborhood structure of v i , a i and x i are the reconstructed structure and attributes for node v i , and α is a hyperparameter.
Our anomaly measure also provides a new perspective to learn tailored node representations for unsupervised GAD. Instead of minimizing the commonly used data reconstruction errors, based on the one-class homophily, we can learn anomaly-discriminative node representations by maximizing the local node affinity. Our TAM approach is designed to achieve this goal.
this section cite: []

Section: TAM: Truncated Affinity Maximization
The local node affinity may not work well in the raw node attribute space since (i) there can be many irrelevant attributes in the original data space and (ii) some abnormal nodes can also be connected to nodes of similar attributes. To tackle this issue, TAM is proposed to learn optimal node representations that maximize the local affinity of nodes that have strong homophily relations with their neighbors in terms of node attributes. Due to the overwhelming presence of normal nodes in a graph, the TAM-based learned node representation space is optimized for normal nodes, enabling stronger local affinity for the normal nodes than the abnormal ones. The TAM-based anomaly scoring using local node affinity can be defined as:
f TAM (v i ; Θ, A, X) = - 1 |N (v i )| vj ∈N(vi) sim (h i , h j ) ,(2)
where h i = ψ(v i ; Θ, A, X) is a GNN-based node representation of v i learned by a mapping function ψ parameterized by Θ in TAM. Below we introduce how we learn the ψ function via the two components of TAM, LAMNet and NSGT.
this section cite: []

Section: Local Affinity Maximization Networks (LAMNet).
LAMNet aims to learn a GNN-based mapping function ψ that maximizes the affinity of nodes with homophily relations to their neighbors, while keeping the affinity of nodes with non-homophily edges are weak. Specifically, the projection from the graph nodes onto new representations using ℓ GNN layers can be generally written as
H (ℓ) = GNN A, H (ℓ-1) ; W (ℓ) ,(3)
where
H (ℓ) ∈ R N ×h (l)
and H (ℓ-1) ∈ R N ×h (l-1) are the h (l) -dimensional and h (l-1) -dimensional representations of node v i in the (ℓ)-th layer and (ℓ -1)-th layer, respectively. In the first GNN layer, i.e., when ℓ = 1, the input H (0) is set to the raw attribute matrix X. W (ℓ) are the weight parameters of (ℓ)-th layer. For GNN(•), multiple types of GNNs can be used [50,55]. In this work, we employ a GCN (graph convolutional network) [20] due to its high efficiency. Then H (ℓ) can be obtained via
H (ℓ) = ϕ D -1 2 AD -1 2 H (ℓ-1) W (ℓ-1) ,(4)
where
D = diag (D i ) = j
A ij is the degree matrix for the graph G, and ϕ(•) is an activation function. Let H (ℓ) = {h 1 , h 2 , ..., h N } be the node representations of the last GCN layer, then the mapping function ψ is a sequential mapping of graph convolutions as in Eq. ( 4), with Θ = {W 1 , W 2 , • • • , W (ℓ) } be the parameter set in our LAMNet. The following objective can then be used to optimize ψ:
min Θ vi∈V f TAM (v i ; Θ, A, X) + λ 1 |V\N(vi)| v k ∈V\N(vi) sim (h i , h k ) ,(5)
where the first term is equivalent to maximizing the local affinity of each node based on the learned node representations, while the second term is a regularization term, and λ is a regularization hyperparameter. The regularization term adds a constraint that the representation of each node should be dissimilar from that of non-adjacent nodes to enforce that the representations of non-local nodes are distinguishable, while maximizing the similarity of the representations of local nodes.
The optimization using Eq. ( 5) can be largely biased by non-homophily edges. Below we introduce the NSGT component that helps overcome this issue.
this section cite: ['b49', 'b54', 'b19']

Section: Normal Structure-preserved Graph Truncation (NSGT).
LAMNet is driven by the one-class homophily property, but its objective and graph convolution operations can be biased by the presence non-homophily edges, i.e., edges that connect normal and abnormal nodes. The NSGT component is designed to remove these non-homophily edges, yielding a truncated adjacency matrix Ã with homophily edge-based normal graph structure. LAMNet is then performed using the truncated adjacency matrix Ã rather than the original adjacency matrix A.
Since homophily edges (i.e., edges connecting normal nodes) connect nodes of similar attributes, the distance between the nodes of homophily edges is often substantially smaller than that of nonhomophily edges. But there can also exist homophily edges that connect dissimilar normal nodes. These can be observed in GAD datasets, as shown in Fig. 3(a-b) for datasets with synthetic/real anomalies. Motivated by this, NSGT takes a probabilistic approach and performs the graph truncation as follows: for a given edge e ij = 1, it is considered as a non-homophily edge and removed (i. e. e ij = 0) if and only if the distance between node v i and node v j is sufficiently large w.r.t. the neighbor sets of both v i and v j , N(v i ) and N(v j ). Formally, NSGT truncates the graph by
e ij ← 0 iff d ij > r i & d ij > r j , ∀e ij = 1,(6)
where d ij is a Euclidean distance between v i and v j based on node attributes, r i is a randomly selected value from the range [d mean , d i,max ] where d mean = 1 m (vi,vj )∈ε d ij is the mean distance of graph, where m is the number of non-zero elements in the adjacent matrix, and d i,max is the maximum distances in {d ik , v k ∈ N(v i )}. Similarly, r j is randomly sampled from the range [d mean , d j,max ].
Theoretically, the probability of r i < d ij can be defined as
p (r i < d ij ) = max(d ij -d mean , 0) d i,max -d mean .(7)
Note that this probability is premised on d i,max > d mean , and we set p (r i < d ij ) = 0 if d i,max ≤ d mean . Then the probability of e ij being removed during the truncation process is as follows
p(E \ e ij ) = p (r i < d ij ) p (r j < d ij ) , (8
)
As shown in Fig. 2(b), if e ij is a homophily edge, we would have a small d ij , which results in small p (r i < d ij ) and p (r j < d ij ), and thus, p(E \ e ij ) is also small. By contrast, a non-homophily edge would result in a large d ij , and ultimately a large p(E \ e ij ). Therefore, NSGT can help eliminate non-homophily edges with a high probability, while preserving the genuine homophily graph structure. Particularly, for each LAMNet, instead of using A, its graph convolutions across all GCN layers are performed on the truncated adjacency matrix Ãk as follows to mitigate the biases caused by the non-homophily edges in message passing:
H (ℓ) = ϕ D -1 2 Ãk D -1 2 H (ℓ-1) W (ℓ-1) .(9)
In doing so, we complete the training of a TAM-based base model, consisting of K LAMNets.
Further, being a probabilistic approach, NSGT has some randomnesses in the graph truncation, and so does the LAMNets. To make use of those randomness, we build an ensemble of TAM models with a size of T , as shown in Fig. 2(a). That is, for a graph G, we perform NSGT T times independently, resulting in T sets of the truncated adjacency matrix set
{A 1 , A 2 , • • • , A T }, with each A = { Ã1 , Ã2 , • • • , ÃK }.
We then train a LAMNet on each of these T × K adjacency matrices, obtaining an ensemble of T TAM models (i.e., T × K LAMNets).
Note that in training the LAMNets, the local affinity in the first term in Eq. ( 5) is computed using the original adjacency matrix A. This is because we aim to utilize the primary graph structure for local affinity-based GAD; the graph truncation is designed to mitigate the graph convolution biases only.
Inference. During inference, we can obtain one local node affinity-based anomaly score per node from each LAMNet, and we aggregate the local affinity scores from all T × K LAMNets to compute an overall anomaly score as
score (v i ) = 1 T × K T t=1 K k=1 f TAM (v i ; Θ * t,k , A, X),(10)
where Θ * t,k is the learned weight parameters for the LAMNet using the k-th truncated adjacency matrix in the t-th TAM model. The weaker the local node affinity in the learned representation spaces under various graph truncation scales, the larger the anomaly score the node v i has.
this section cite: []

Section: Experiments
Datasets. We conduct the experiments on six commonly-used publicly-available real-world GAD datasets from diverse online shopping services and social networks, and citation networks, including BlogCatalog [49], ACM [48], Amazon [10], Facebook [56], Reddit, and YelpChi [21]. The first two datasets contain two types of injected anomalies -contextual and structural anomalies [8,34] -that are nodes with significantly deviated graph structure and node attributes respectively. The other four datasets contain real anomalies. Detailed information about the datasets can be found in App. B.
this section cite: ['b48', 'b47', 'b9', 'b55', 'b20', 'b7', 'b33']

Section: Competing Methods and Performance Metrics.
TAM is compared with two state-of-the-art (SOTA) shallow methods -iForest [25] and ANOMALOUS [41] -and five SOTA GNN-based deep methods, including three self-supervised learning based methods -CoLA [28], SL-GAD [63], and HCM-A [17] -and two reconstruction-based methods -DOMINANT [8] and ComGA [31]. iForest works on the raw node attributes, while ANOMALOUS works on the raw node attributes and graph structure. The other methods learn new representation space for GAD.
Following [4,38,52,64], two popular and complementary evaluation metrics for anomaly detection, Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the precisionrecall curve (AUPRC), are used. Higher AUROC/AUPRC indicates better performance. The reported AUROC and AUPRC results are averaged over 5 runs with different random seeds.
Implementation Details. TAM is implemented in Pytorch 1.6.0 with python 3.7 and all the experiments are run on an NVIDIA GeForce RTX 3090 24GB GPU. In TAM, each LAMNet is implemented by a two-layer GCN, and its weight parameters are optimized using Adam [19] optimizer with 500 epochs and a learning rate of 1e -5 by default. T = 3 and K = 4 are used for all datasets. Datasets with injected anomalies, such as BlogCatalog and ACM, require strong regularization, so λ = 1 is used by default; whereas λ = 0 is used for the four real-world datasets. Hyperparameter analysis w.r.t. K is presented in Sec. 4.2. TAM can perform stably within a range of T and λ (see App. C.1 for detail). All the competing methods are based on their publicly-available official source code, and they are trained using their recommended optimization and hyperparameter settings in the original papers.
this section cite: ['b24', 'b40', 'b27', 'b62', 'b16', 'b7', 'b30', 'b3', 'b37', 'b51', 'b63', 'b18']

Section: Main Results
Effectiveness on Diverse Real-world Datasets. The AUROC and AUPRC results on six real-world GAD datasets are reported in Tab. 1. TAM substantially outperforms all seven competing methods on all datasets except Reddit in both metrics, having maximally 9% AUROC and 12% AUPRC improvement over the best-competing methods on the challenge dataset Amazon; on Reddit, it ranks second and performs similarly well to the best contender CoLA. Further, existing methods perform very unstably across different datasets. For example, DOMINANT performs well on ACM but badly on the other datasets; CoLA works well on Reddit but it fails in the other datasets. Similar observations are found in a recent comprehensive comparative study [26]. By contrast, TAM can perform consistently well on these diverse datasets. This is mainly because i) the proposed one-class homophily is more pervasive than the GAD intuitions used by existing methods in different datasets, and ii) TAM offers an effective anomaly scoring function to utilize this anomaly-discriminative property for accurate GAD.
this section cite: ['b25']

Section: On Detecting Structural and Contextual Anomalies.
We further examine the effectiveness of TAM on detecting two commonly-studied anomaly types, structural and contextual anomalies, with the results on BlogCatalog and ACM reported in Tab. 2, where the three best competing methods on the two datasets in Tab. 1 are used as baselines. Compared to contextual anomalies, it is significantly more challenging to detect structural anomalies, for which TAM outperforms all three methods in both AU-ROC and AUPRC. As for contextual anomalies, although TAM underperforms SL-GAD and ranks in second in AUROC, it obtains substantially better AUPRC than SL-GAD. Note that compared to AUROC that can be biased by low false positives and indicates overoptimistic performance, AUPRC is a more indicative measure focusing on the performance on the anomaly class exclusively [3,38]. So, achieving the best AUPRC on all four cases of the two datasets demonstrates the superiority of TAM in precision and recall rates for both types of anomaly. These results also indicate that structural anomalies may have stronger local affinity than contextual anomalies, but both of which often have weaker local affinity than normal nodes. TAM vs. Raw/Generic Node Representation Space. As discussed in Sec. 1, local node affinity requires a new node representation space that is learned for the affinity without being biased by non-homophily edges. We provide quantitative supporting results in Tab. 3, where TAM is compared with Raw Attribute (RA), and the spaces learned by DGI [51] and GMI [40]. It is clear that the representations learned by TAM significantly outperforms all three competing representation spaces on all six datasets. Computational Efficiency. The runtime (including both training and inference time) results are shown in Tab. 4. DOMINANT and ComGA are the simplest GNN-based methods, achieving the most efficient methods. Our method needs to perform multiple graph truncation and train multiple LAMNets, so it takes more time than these two reconstruction-based methods, but it runs much faster than the three recent SOTA models, HCM-A, CoLA, and SL-GAD, on most of the datasets. A detailed analysis is provided in App. C.2.
this section cite: ['b2', 'b37', 'b50', 'b39']

Section: Ablation Study
Graph Truncation NSGT. Three alternative approaches to our graph truncation NSGT include:
this section cite: []

Section: Affinity Maximization Network LAMNet.
The importance of LAMNet is examined by comparing it to its two variants, including Raw Truncated Affinity (RTA) that directly calculates the local affinity-based anomaly scores after NSGT (i.e., without involving LAMNet at all), and DOM that performs LAMNet but with our affinity maximization objective replaced by the popular graph reconstruction loss used in DOMINANT [8]. As shown by the comparison results reported in Tab. 6, LAMNet consistently and significantly outperforms both RTA and DOM, showing that LAMNet can make much better use of the truncated graphs.
In addition, by working on our truncated graphs, DOM can substantially outperform its original version DOMINANT in Tab. 1 on most of the datasets. This indicates that the proposed one-class homophily property may be also exploited to improve existing GAD methods.
this section cite: ['b7']

Section: Anomaly Scoring.
We compare the TAM anomaly scoring to its two variants: i) TAM-T that calculates the node affinity in Eq. ( 5) on the the truncated graph structure rather than the primary graph structure as in TAM, and ii) Degree that directly uses the node degree after our graph truncation as anomaly score. As illustrated by the AUPRC results in Fig. 4(a), TAM consistently and significantly outperforms both variants. Degree obtains fairly good performance, which shows that our graph truncation results in structural changes that are beneficial to GAD. TAM-T underperforms TAM since the local affinity based on the truncated graph structure is affected by randomness and uncertainty in the truncation, leading to unstable and suboptimal optimization. We also show in Fig. 4 (b) that aggregating the anomaly scores obtained from different truncation scales/depths helps largely improve the detection performance. Similar observations are found in the corresponding AUROC results (see App. C.3). In order to demonstrate the effectiveness of TAM on the large-scale datasets, we conduct the experiments on the four large-scale datasets with a large set of nodes and edges, Amazon-all and YelpChi-all by treating the different relations as a single relation following [6], T-Finance [47] and OGB-Proteins [14]. The experimental results are shown in Tab. 7. Due to the increased number of nodes and edges, we set K = 7 for these datasets. TAM can perform consistently well on these large-scale datasets and outperforms four comparing methods, which provides further evidence about the effectiveness of our proposed one-class homophily and anomaly measure. Fraudsters may adjust their behaviors to camouflage their malicious activities, which could hamper the performance of GNN-based methods. We evaluate the performance of TAM when there are camouflages in the raw attributes. Particularly, we replace 10%, 20%, 30% randomly sampled original attributes with camouflaged attributes, in which the feature/attribute value of the abnormal nodes is replaced (camouflaged) with the mean feature value of the normal nodes. The results are shown in the Tab. 8, which demonstrates that TAM is robust to a high level of camouflaged features and maintains its superiority over the SOTA models that work on the original features. The reason is that NSGT can still successfully remove some non-homophily edges in the local domain, allowing LAMNet to make full use of the one-class homophily and achieve good performance under the camouflage.
this section cite: ['b5', 'b46', 'b13']

Section: Performance on Large-scale Graphs

this section cite: []

Section: Handling Camouflage Attributes

this section cite: []

Section: Conclusion and Future Work
This paper reveals an important anomaly-discriminative property, the one-class homophily, in GAD datasets with either injected or real anomalies. We utilize this property to introduce a novel unsupervised GAD measure, local node affinity, and further introduce a truncated affinity maximization (TAM) approach that end-to-end optimizes the proposed anomaly measure on truncated adjacency matrix. Extensive experiments on 10 real-world GAD datasets show the superiority of TAM over seven SOTA detectors. We also show that the one-class homophily can be exploited to enhance the existing GAD methods.
Limitation and Future Work. TAM cannot directly handle primarily isolated nodes in a graph, though those isolated nodes are clearly abnormal. Additionally, like many GNN-based approaches, including GAD methods, TAM also requires a large memory to perform on graphs with a very large node/edge set. The one-class homophily may not hold for some datasets, such as datasets with strong heterophily relations/subgraphs of normal nodes [1,29,36,46,62,66]. Our method would require some adaptations to work well on the dataset with strong heterophily or very large graphs, which are also left for future work.
this section cite: ['b0', 'b28', 'b35', 'b45', 'b61', 'b65']

Section: References
Ref_id:b0 Title: Oddball: Spotting anomalies in weighted graphs Year: (2010)
Ref_id:b1 Title: Graph based anomaly detection and description: a survey Year: (2015)
Ref_id:b2 Title: Area under the precision-recall curve: point estimates and confidence intervals Year: (2013)
Ref_id:b3 Title: Can abnormality be detected by graph neural networks? Year: (2022)
Ref_id:b4 Title: Anomaly detection: A survey Year: (2009)
Ref_id:b5 Title: Gccad: Graph contrastive learning for anomaly detection Year: (2022)
Ref_id:b6 Title: Fast gradient attack on network embedding Year: (2018)
Ref_id:b7 Title: Deep anomaly detection on attributed networks Year: (1920)
Ref_id:b8 Title: Bi-level selection via meta gradient for graph-based fraud detection Year: (2022)
Ref_id:b9 Title: Enhancing graph neural network-based fraud detectors against camouflaged fraudsters Year: (2020)
Ref_id:b10 Title: Anomalydae: Dual autoencoder for anomaly detection on attributed networks Year: (2020)
Ref_id:b11 Title: Addressing heterophily in graph anomaly detection: A perspective of graph spectrum Year: (2023)
Ref_id:b12 Title: Alleviating structural distribution shift in graph anomaly detection Year: (2023)
Ref_id:b13 Title: Open graph benchmark: Datasets for machine learning on graphs Year: (2020)
Ref_id:b14 Title:  Year: (2021)
Ref_id:b15 Title: Auc-oriented graph neural network for fraud detection Year: (2022)
Ref_id:b16 Title: Hop-count based self-supervised anomaly detection on attributed networks Year: (1920)
Ref_id:b17 Title: Anemone: graph anomaly detection with multi-scale contrastive learning Year: (2021)
Ref_id:b18 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b19 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b20 Title: Predicting dynamic embedding trajectory in temporal interaction networks Year: (2019)
Ref_id:b21 Title: Radar: Residual analysis for anomaly detection in attributed networks Year: (2017)
Ref_id:b22 Title: Dual-augment graph neural network for fraud detection Year: (2022)
Ref_id:b23 Title: Specae: Spectral autoencoder for anomaly detection in attributed networks Year: (2019)
Ref_id:b24 Title: Isolation-based anomaly detection Year: (1920)
Ref_id:b25 Title: Benchmarking unsupervised outlier node detection on static attributed graphs Year: (2022)
Ref_id:b26 Title: Pick and choose: a gnn-based imbalanced learning approach for fraud detection Year: (2021)
Ref_id:b27 Title: Anomaly detection on attributed networks via contrastive self-supervised learning Year: (1920)
Ref_id:b28 Title: Beyond smoothing: Unsupervised graph representation learning with edge heterophily discriminating Year: (2023)
Ref_id:b29 Title: Alleviating the inconsistency problem of applying graph neural network to fraud detection Year: (2020)
Ref_id:b30 Title: Comga: Community-aware attributed graph anomaly detection Year: (1920)
Ref_id:b31 Title: A comprehensive survey on graph anomaly detection with deep learning Year: (2021)
Ref_id:b32 Title: Is homophily a necessity for graph neural networks Year: (2021)
Ref_id:b33 Title: From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews Year: (2013)
Ref_id:b34 Title: What yelp fake review filter might be doing? Year: (2013)
Ref_id:b35 Title: Graph-based anomaly detection Year: (2003)
Ref_id:b36 Title: Longbing Cao, and Anton Van Den Hengel. Deep learning for anomaly detection: A review Year: (2021)
Ref_id:b37 Title: Toward deep supervised anomaly detection: Reinforcement learning from partially labeled anomaly data Year: (2021)
Ref_id:b38 Title: Resgcn: attentionbased deep residual modeling for anomaly detection on attributed networks Year: (2022)
Ref_id:b39 Title: Graph representation learning via graphical mutual information maximization Year: (2020)
Ref_id:b40 Title: Anomalous: A joint modeling approach for anomaly detection on attributed networks Year: (1920)
Ref_id:b41 Title: A deep multi-view framework for anomaly detection on attributed networks Year: (2020)
Ref_id:b42 Title: Scalable anomaly ranking of attributed neighborhoods Year: (2016)
Ref_id:b43 Title: Collective opinion spam detection: Bridging review networks and metadata Year: (2015)
Ref_id:b44 Title: H2-fdetector: a gnn-based fraud detector with homophilic and heterophilic connections Year: (2022)
Ref_id:b45 Title: Neighborhood formation and anomaly detection in bipartite graphs Year: (2005)
Ref_id:b46 Title: Rethinking graph neural networks for anomaly detection Year: (2022)
Ref_id:b47 Title: Arnetminer: extraction and mining of academic social networks Year: (2008)
Ref_id:b48 Title: Relational learning via latent social dimensions Year: (2009)
Ref_id:b49 Title: Graph attention networks Year: (2017)
Ref_id:b50 Title:  Year: (2008)
Ref_id:b51 Title: Crossdomain graph anomaly detection via anomaly-aware contrastive alignment Year: (2022)
Ref_id:b52 Title: One-class graph neural networks for anomaly detection in attributed networks Year: (2021)
Ref_id:b53 Title: Decoupling representation learning and classification for gnn-based anomaly detection Year: (2021)
Ref_id:b54 Title: How powerful are graph neural networks? arXiv preprint Year: (2018)
Ref_id:b55 Title: Contrastive attributed network anomaly detection with data augmentation Year: (2022)
Ref_id:b56 Title: Mining fraudsters and fraudulent strategies in large-scale mobile social networks Year: (2019)
Ref_id:b57 Title: Fraudre: Fraud detection dual-resistant to graph inconsistency and imbalance Year: (2021)
Ref_id:b58 Title: Reconstruction enhanced multi-view contrastive learning for anomaly detection on attributed networks Year: (2022)
Ref_id:b59 Title: Gcn-based user representation learning for unifying robust recommendation and fraudster detection Year: (2020)
Ref_id:b60 Title: Error-bounded graph anomaly loss for gnns Year: (2020)
Ref_id:b61 Title: Graph neural networks for graphs with heterophily: A survey Year: (2022)
Ref_id:b62 Title: Generative and contrastive self-supervised learning for graph anomaly detection Year: (1920)
Ref_id:b63 Title: Unseen anomaly detection on networks via multi-hypersphere learning Year: (2022)
Ref_id:b64 Title: Subtractive aggregation for attributed network anomaly detection Year: (2021)
Ref_id:b65 Title: Graph neural networks with heterophily Year: (2021)
