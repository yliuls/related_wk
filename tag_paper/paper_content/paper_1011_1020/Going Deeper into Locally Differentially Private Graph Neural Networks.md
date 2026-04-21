Title: Going Deeper into Locally Differentially Private Graph Neural Networks
Abstract: Graph Neural Networks (GNNs) have demonstrated superior performance in a variety of graph mining and learning tasks. However, when node representations involve sensitive personal information or variables related to individuals, learning from graph data can raise significant privacy concerns. Although recent studies have explored local differential privacy (LDP) to address these concerns, they often introduce significant distortions to graph data, severely degrading private learning utility (e.g., node classification accuracy). In this paper, we present UPGNET, an LDP-based privacy-preserving graph learning framework that enhances utility while protecting user data privacy.Specifically, we propose a three-stage pipeline that generalizes the LDP protocols for node features, targeting privacy-sensitive scenarios. Our analysis identifies two key factors that affect the utility of privacy-preserving graph learning: feature dimension and neighborhood size. Based on the above analysis, UPGNET enhances utility by introducing two core layers: High-Order Aggregator (HOA) layer and the Node Feature Regularization (NFR) layer. Extensive experiments on real-world datasets indicate that UPGNET significantly outperforms existing methods in terms of both privacy protection and learning utility.

Section: Introduction
In recent years, Graph Neural Networks (GNNs) have shown superior performance in various domains, including social sciences (Hamilton et al., 2017), graph mining (Li et al., 2019), and bioinformatics (Fout et al., 2017). GNNs have also achieved state-of-the-art performance on a range of downstream graph learning tasks, such as node classification (Kipf & Welling, 2017), link prediction (Zhang & Chen, Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). Users' sensitive node features x are perturbed to x ′ using LDP before uploading to the cloud server for graph learning. Our approach achieves higher utility by integrating than prior works.
2018), and community detection (Chen et al., 2019). In real-world scenarios, graphs frequently contain significant amounts of sensitive personal information, such as user profiles on social networks. However, recent studies (Wang & Wang, 2022;Shen et al., 2022;Zhang et al., 2022;Meng et al., 2023) have proposed various privacy attack methods targeting GNN models, which pose serious security and privacy challenges during their training. Therefore, designing an efficient privacy-preserving GNN framework to protect users' private information is of paramount importance.
To collect and analyze private data from decentralized data owners, local differential privacy (LDP) (Kasiviswanathan et al., 2011) has been increasingly accepted as the de facto standard for data privacy in the research community (Erlingsson et al., 2014;Ding et al., 2017;Wang et al., 2019a;b).
In the LDP protocol, multiple users must interact with an untrustworthy server that may exploit their private data. Each user perturbs their data locally, typically through noise injection (Dwork et al., 2006), to ensure privacy. The perturbed data is then transmitted to the server, which conducts data analysis and learning based on this information. Locally private graph learning have recently gained significant attention from researchers (Sajadmanesh & Gatica-Perez, 2021;Lin et al., 2022;Jin & Chen, 2022;Pei et al., 2023). Fig. 1 illustrates this scenario, where a cloud server interacts with decentralized users. Each user perturbs their sensitive node features x to x ′ under LDP before submitting them to the server for private graph learning (e.g., node classification).
However, in locally private graph learning, existing LDP protocols for perturbing node features cause significant performance degradation in GNNs (Sajadmanesh & Gatica-Perez, 2021;Lin et al., 2022). Specifically, LDP regulates noise injection through the privacy budget ϵ, where a smaller ϵ introduces more noise. In practice, ϵ is highly limited and node features are typically multidimensional with multiple attributes. As a result, each attribute receives only a minimal fraction of the budget, leading to substantial information loss. While server-side aggregation mitigates noise to some extent, it also introduces excessive estimation errors, further reducing utility (e.g., node classification accuracy). Thus, the key challenge is how to maximize the utility of privacypreserving graph learning while ensuring user privacy.
Contributions. To address this challenge, we present a threestage pipeline that generalizes the current LDP protocols for perturbing node features. Our analysis of the pipeline reveals two key factors that influence the estimation error in feature aggregation: feature dimension and neighborhood size. We conclude that reducing the effective feature dimension and expanding the effective neighborhood size help minimize the estimation error and thus enhance the utility.
Based on these findings, we propose UPGNET, a utilityenhanced framework for locally privacy-preserving graph learning that minimizes estimation error and maximizes utility from two key perspectives (Comparison with prior works is presented in Fig. 1). First, to reduce the effective feature dimensions, we introduce a Node Feature Regularization (NFR) layer based on L 1 -regularization (Bühlmann & Van De Geer, 2011), a classical optimization technique that promotes sparsity in solutions. We conduct a theoretical analysis using proximal gradient descent (Nitanda, 2014;Li & Lin, 2015) to derive a one-off, non-iterative solution. This enables the NFR to obtain sparse embeddings through feature selection, thereby reducing effective feature dimensions during aggregation. Second, to expand the effective neighborhood size, we propose a multi-hop aggregation method, the High-Order Aggregator (HOA) layer. Although our theoretical analysis indicates that increasing neighborhood size improves utility, real-world graphs often feature small neighborhoods. A straightforward solution, such as aggregating multi-layer node features (Abu-El-Haija et al., 2019;Gasteiger et al., 2019;Chen et al., 2020), suffers from over-smoothing (Rusch et al., 2023). As the number of layers increases, node embeddings tend to converge, diminishing the effectiveness of higher-order neighbors in correcting errors. To address this, our HOA layer leverages personalized aggregation and Dirichlet energy (Zhou et al., 2021;Rusch et al., 2023) analysis, effectively mitigating oversmoothing and reducing noise bias injection. UPGNET comprises two architectures: H-N (HOA followed by NFR) and N-H (NFR followed by HOA), both of which can be independently integrated with any GNN architecture. We evaluate overall performance of UPGNET and the contributions of each component under different parameters through theoretical analysis and extensive experiments.
Our contributions are summarized as follows. ① We propose a three-stage pipeline to systematically generalize the LDP protocols for perturbing node features. By analyzing the pipeline, we identify two key factors influencing the estimation error of feature aggregation. ② Based on the above analysis, we propose NFR and HOA layers to reduce the estimation error and integrate them with LDP protocols, introducing UPGNET, a utility-enhanced privacy-preserving graph learning framework. ③ Extensive experiments on real datasets demonstrate that UPGNET excels in achieving privacy preservation and superior graph learning utility.
this section cite: ['b15', 'b27', 'b12', 'b23', 'b4', 'b42', 'b57', 'b32', 'b21', 'b11', 'b5', 'b9', 'b41', 'b28', 'b20', 'b35', 'b41', 'b28', 'b2', 'b34', 'b26', 'b0', 'b13', 'b3', 'b40', 'b58', 'b40']

Section: Preliminaries
This section first defines the problem (Sec. 2.1), then provides the essential background on LDP (Sec. 2.2) and GNN (Sec. 2.3), and finally describes the threat model (Sec. 2.4).
this section cite: []

Section: Problem Definition
Consider a graph G = (V, E), where V = {v 1 , v 2 , . . . , v |V| }
represents the set of nodes and E is the set of edges. Each decentralized user v ∈ V locally possesses a d-dimensional feature vector x v ∈ R d , and the feature matrix is defined as X ∈ R |V|×d . Following previous work (Sajadmanesh & Gatica-Perez, 2021;Lin et al., 2022;Pei et al., 2023), we assume that the server has access to V and E, but X remains private and inaccessible to the server.foot_2 Consequently, we face the challenge of performing graph learning on G while protecting the privacy of node features. Consistent with previous work (Sajadmanesh & Gatica-Perez, 2021;Lin et al., 2022), this paper focuses on the node classification task in graph learning. Specifically, the node set V = V l ∪ V u is the union of the set of labeled nodes V l and unlabeled ones V u . The label y v for each node v is derived from a set of possible labels, denoted Y = {y 1 , y 2 , . . . , y c }. The objective of node classification (Kipf & Welling, 2017) is to learn a function f : V → Y that assigns labels to unlabeled nodes based on the graph structure and available node features.
this section cite: ['b41', 'b28', 'b35', 'b41', 'b28', 'b23']

Section: Local Differential Privacy
LDP (Kasiviswanathan et al., 2011;Yang et al., 2024;He et al., 2025) has been extensively studied and widely deployed in decentralized data collection and analysis scenarios. In particular, major companies such as Apple (Thakurta et al., 2017), Google (Erlingsson et al., 2014), and Microsoft (Ding et al., 2017) have adopted LDP. In an LDP setting, there exists a server and multiple users, each possessing sensitive data. Users are not required to transmit their private data to an untrustworthy server. Instead, each user initially perturbs their data using a perturbation mechanism M and then transmits the perturbed data to the server. Following the collection of perturbed data from each user, the server performs data analysis and learning based on these data, ensuring that user privacy remains uncompromised.
The formal definition of ϵ-LDP is provided below.
Definition 1 (ϵ-LDP). A local perturbation mechanism M satisfies ϵ-local differential privacy (ϵ-LDP), where ϵ > 0, if and only if for any user's private data x and x ′ , we have:
∀y ∈Range(M): Pr[M(x)= y]≤e ϵ •Pr[M(x ′ )= y], (1)
where Range(M) denotes the set of all possible outputs of the perturbation mechanism M. In essence, LDP guarantees that the data aggregator on the server side can't reconstruct the data source, regardless of any prior knowledge. The parameter ϵ, called the privacy budget, plays a pivotal role in balancing privacy and utility. A smaller (resp. larger) ϵ provides stronger (resp. weaker) privacy preservation but also results in lower (resp. higher) utility. LDP has several important properties, such as immunity to post-processing and sequential composition (Dwork, 2008).
this section cite: ['b21', 'b52', 'b16', 'b46', 'b11', 'b5', 'b8']

Section: Graph Nerual Networks
In recent years, GNNs have gained popularity for graph mining and learning. The primary goal of GNNs is to learn embeddings for each node in a graph by combining initial node features with the graph's topology. These learned node embeddings can then be applied to various downstream tasks such as node classification (Kipf & Welling, 2017;Sun et al., 2024a;b). A typical K-layer GNN consists of K graph convolutional layers. Each layer aggregates information from neighboring nodes and updates the node's embedding. Following K aggregation iterations, the embedding of a node captures information from its neighbors within K hops. The formal definition of the k-th layer is as follows:
h k N (v) = AGGREGATE k ({h k-1 u , ∀u ∈ N (v)}), (2) h k v = UPDATE k (h k N (v) ),(3)
where N (v) represents the set of neighbors of node v (which could include v itself). For any node u ∈ N (v), h k-1 u denotes the embedding of node u at layer k -1. The aggregation functions at layer k, such as mean, sum, and max, are denoted as
AGGREGATE k (•) functions. h k N (v)
represents the output of the aggregation function on N (v). UPDATE k (•) denotes a learnable non-linear function at layer k, such as a neural network. Initially, h 0 v = x v , indicating that the initial embedding of node v is its feature vector x v .
this section cite: ['b23']

Section: Threat Model
As shown in Fig. 1, different users upload graph data to a third-party untrustworthy server. On the one hand, under a semi-honest adversary setup, although the server follows our LDP protocol honestly, it may attempt to individually learn the private information of the data owner. On the other hand, an attacker (Wang & Wang, 2022;Shen et al., 2022;Zhang et al., 2022;Meng et al., 2023) can target the GNN model to extract private information from the victim node, leading to disclosure of the user's privacy.
this section cite: ['b42', 'b57', 'b32']

Section: Methodology
In this section, we begin with a theoretical analysis of prior work on locally differentially private graph neural networks (LDPGNN) in Sec. 3.1, identifying the key factors limiting their utility. Following this, we introduce UPGNET, a utility-enhanced private graph learning model, in Sec. 3.2.
this section cite: []

Section: Theoretical Analysis of LDPGNN
A series of studies (Sajadmanesh & Gatica-Perez, 2021;Du et al., 2021;Lin et al., 2022;Jin & Chen, 2022;Qi et al., 2024;Pei et al., 2023) have been conducted on private graph learning based on LDP. However, no effort has been made to establish a unified framework for revisiting existing work to further enhance the utility of private graph learning. In this subsection, our objective is to identify the key factors influencing the aggregation estimation error by constructing and analyzing a unified node feature LDP pipeline.
this section cite: ['b41', 'b28', 'b20', 'b36', 'b35']

Section: NODE FEATURE LDP PIPELINE
Currently, two state-of-the-art LDP mechanisms are applied to node features: the piecewise mechanism (PM) (Wang et al., 2019a;Pei et al., 2023) and the multi-bit mechanism (MBM) (Du et al., 2021;Sajadmanesh & Gatica-Perez, 2021;Lin et al., 2022;Jin & Chen, 2022) (for more details on PM and MBM see App. A). We propose a unified node-feature LDP pipeline that generalizes these two approaches. Assuming that the node-feature LDP mechanism is denoted as M and the total privacy budget employed is ϵ, the node-feature LDP mechanism can be outlined in three steps: Perturbation. In total, there are |V| users, and each user v possesses a d-dimensional node feature x v containing their sensitive information. Users employ an LDP mechanism M to protect their privacy. Initially, the mechanism M randomly selects m dimensions from d dimensions without replacement, with m being a configurable parameter controlling the number of perturbed dimensions. Subsequently, each sampled dimension undergoes random perturbation with a privacy budget denoted as ϵ/m, while the remaining d -m dimensions are set to 0. Ultimately, x v undergoes the M mechanism to yield
x ′ v , denoted as x ′ v = M(x v ). Calibration. Following perturbation, x ′ v remains biased, i.e., E [x ′ v ] ̸ = x v . Let σ = E [x ′ v ] -x v
, representing the expected bias shift. The server calibrates the perturbed values with σ. Note that σ = 0 indicates an unbiased estimate.
Aggregation. After receiving the feature vectors x ′ v for all users v ∈ V, the server aggregates these vectors as follows:
h N (v) = AGGREGATE ({x ′ u , ∀u ∈ N (v)}) ,(4)
where h N (v) represents the estimated embedding for any given node v after undergoing the AGGREGATE(•). This estimate is derived by aggregating the perturbed node feature vectors x ′ u from all nodes u adjacent to the target node v. Theorem 2. Assuming σ = 0, the aggregator function defined by Eq. ( 4) is an unbiased estimate, i.e., for any v,
E[ h N (v) ] = h N (v) .(5)
As demonstrated in Thm. 2, the aggregation process is an unbiased estimate when σ = 0 and the AGGREGATE function is linear, meaning the output is a weighted summation of the inputs. For the proof, please refer to App. B.1.
this section cite: ['b35', 'b41', 'b28', 'b20']

Section: KEY FACTOR ANALYSIS
In Sec. 3.1.1, we establish an LDP analytical pipeline for the node features, consisting of three stages: perturbation, calibration, and aggregation. In the context of privacypreserving graph learning, the aggregation stage significantly impacts the overall utility of graph learning. Highquality aggregation helps mitigate the injected noise to a greater extent. Therefore, our objective is to explore the key factors that directly influence the estimation error in the aggregation stage. The estimation error is defined as the discrepancy between h N (v) , obtained by aggregating the perturbed node features x ′ , and h N (v) , obtained by aggregating the original node features x. This discrepancy is represented as (Giroux et al., 1979), Thm. 3 provides an analysis of the various factors influencing the discrepancy. Theorem 3. Given the aggregator for the first layer and δ>0, with probability at least 1-δ, for any node v, we have:
ξ i = |( h N (v) ) i -(h N (v) ) i |, i ∈ {1, . . . , d}. Based on Bern- stein's inequality
max ξi = O( d log(d/δ)/(ϵ |N (v)|)), i ∈ {1, . . . , d}. (6)
According to Thm. 3, after eliminating the known privacy budget parameter ϵ and the analysis parameter δ, two key factors that impact the estimation error are the feature dimension d and the neighborhood size |N (v)|. From Eq. ( 6), we infer that a smaller effective d is more conducive to reducing the estimation error, while a larger effective |N (v)| is also advantageous to minimizing the estimation error. Therefore, in Sec. 3.2, our objective is enhance the utility of privacy-preserving graph learning by influencing d and |N (v)|. Please see App. B.2 for the proof of Thm. 3.
this section cite: ['b14']

Section: UPGNET: Utility-Enhanced Private GNNs
In this section, we propose a utility-enhanced private graph learning framework called UPGNET. Designed for various node feature LDP protocols, UPGNET incorporates plugand-play NFR and HOA layers that significantly boost utility.
this section cite: []

Section: OVERVIEW
This subsection introduces the foundational design principles of UPGNET. According to Thm. 3, the key factors influencing the estimation error during the aggregation process are the feature dimension and the neighborhood size. Naturally, our objective is to minimize the estimation error during aggregation by targeting these two crucial factors, thereby enhancing the practicality of private graph learning.
To this end, we design from the following two perspectives:
Expanding the Effective Neighborhood Size. In order to extend the effective neighborhood size, direct multi-layer aggregation is a potential approach. However, our analysis reveals that this method is significantly limited by oversmoothing (Rusch et al., 2023), which adversely affected the denoising performance. To address this, we propose a Higher-Order Aggregator (HOA) layer, leveraging personalized aggregation and Dirichlet energy analysis to effectively mitigate over-smoothing and reduce noise injection.
Reducing the Effective Feature Dimensions. In order to minimize estimation error by reducing the effective feature dimensions, we focus on the aggregation stagefoot_4 . Based on L 1 -regularization and proximal gradient descent (PGD) (Nitanda, 2014;Li & Lin, 2015;Duan et al., 2022), we introduce the Node Feature Regularizer (NFR) layer.
Through theoretical analysis and experimental validation, UPGNET, in both the H-N (see Sec. 3.2.2) and the N-H architecture (see Sec. 3.2.3), effectively integrates the HOA layer and the NFR layer to reduce estimation error and enhance learning utility across various LDP mechanisms.
this section cite: ['b40', 'b34', 'b26', 'b7']

Section: UPGNET IN H-N ARCHITECTURE
In this part, we introduce the H-N architecture of UPGNET, where the perturbed node features x ′ are successively enhanced by the HOA layer followed by the NFR layer. smaller estimation errors. However, in practical scenarios, |N (v)| is usually quite small. One approach to address this issue is to expand N (v) by directly aggregating multiple layers of node features across K hops (Abu-El-Haija et al., 2019; Gasteiger et al., 2019; Chen et al., 2020 . However, the SKA scheme encounters the oversmoothing (Rusch et al., 2023), where increasing the value of K causes node embeddings to converge, reducing the effectiveness of information aggregation from higher-order neighbors and skewing the calibration of aggregation errors. To better understand the over-smoothing issue, we examine the Dirichlet energy (Rusch et al., 2023) of the estimated node embeddings on the graph, a primary measure of over-smoothing in deep GNNs. Specifically, the estimated embedding h is modeled as a combination of the original embedding h and the noise signal η, i.e., h = h + η. The Dirichlet energy Υ(•) of h is then defined as:
Υ( h) = 1 |V| i∈V j∈N (v i ) ( h k i -h k j 2 2 + η k i -η k j 2 2 noise signal ),(7)
where k ∈ {1, 2, • • • , K} represents the step parameter. Eq. ( 7) highlights two key observations: ① Over-smoothing exists. As k increases, the difference between the embedding h i and its neighbor embedding h j diminishes. After several rounds of propagation, all node features converge, causing a sharp decline in the first term of Υ( h), resulting in the typical oversmoothing phenomenon. ② Noise exacerbates over-smoothing. Under the LDP, when ϵ approaches 0, the noise term ∥η i -η j ∥ 2 2 dominates the Dirichlet energy. This accelerates the homogenization of node features, leading to faster energy decay and thus intensifying the oversmoothing effect. Therefore, mitigating the over-smoothing problem to extend more effective neighborhood sizes is crucial.
To address the aforementioned issues, we propose a utilityenhanced graph convolution layer called the High-Order Aggregator (HOA), as shown in Alg. 1. Compared to SKA, it has two advantages in reducing estimation error for noisy data: ① mitigating over-smoothing and ② reducing noise
Algorithm 1 High-Order Aggregator (HOA) Layer Input: G = (V, E), input vector x v , ∀v ∈ V, step param- eter K ≥ 0, linear aggregator function AGGREGATE(•) for v ∈ V do h 0 N (v) = 0, x 0 N (v) = x v for k = 1 to K do x k N (v) = AGGREGATE {x k-1 N (u) , ∀u ∈ N (v)\{v}} h k N (v) = h k-1 N (u) + x k N (v) end for h v = 1 K h K N (v)
end for Return: Aggregated embedding vector h v , ∀v ∈ V bias injection. First, as shown in Thm. 4, the energy ratio Φ K between HOA(•) and SKA(•) approaches 0 as K → ∞. This indicates that HOA is less influenced by information from infinite-hop receptive fields, allowing it to mitigate over-smoothing and expand into larger neighborhoods while still aggregating data from smaller ones. Second, HOA employs personalized weightings during neighbor information aggregation, assigning the highest weight to the nearest neighbor. The insight behind our approach is that information from the closest neighbor is more effective in calibrating noise for that specific node, justifying its higher weight, while more distant neighbors receive lower weights. This design further alleviates noise bias injection. See App. B.3 for the proof of Thm. 4 and more details.
Theorem 4. Let Υ k HOA and Υ k SKA represent the Dirichlet energies of HOA(•) and SKA(•) at layer k, respectively. The energy ratio of HOA(•) to SKA(•) across K layers satisfies:
Φ K = lim K→∞ K k=1 Υ k HOA / K k=1 Υ k SKA = 0. (8)
As in (Sajadmanesh & Gatica-Perez, 2021), we also employ the GCN aggregator function (Kipf & Welling, 2017) and perform the aggregations without including the self-loop, which will facilitate the reduction of the total noise.
this section cite: ['b40', 'b40', 'b41', 'b23']

Section: Node Feature Regularization.
Regularization is a clas-sical method for optimizing minimization tasks (Hoyer, 2004;Bühlmann & Van De Geer, 2011;Negahban et al., 2017;Duan et al., 2022). Among various regularization techniques, L 1 -regularization tends to produce sparse solutions. In other words, parameters obtained through L 1regularization are more likely to have fewer non-zero components. This property facilitates the implementation of embedded feature selection, aligning with our objective of reducing the effective feature dimension. Next, we formalize the L 1 -regularization problem within UPGNET under the H-N architecture. We employ the mean aggregation function and define the embedding of node v after the server aggregates the perturbed feature vectors as follows:
hv=AGGREGATE {x ′ u , ∀u ∈ N (v)} = 1 |N (v)| u∈N (v) x ′ u , (9
)
where x ′ u represents the noisy node features of node u. We define the loss function L 1 (w) as follows:
L 1 (w) = 1 2 |N (v)| • u∈N (v) ∥x ′ u -w∥ 2 2 . (10
)
Based on this, we add the L 1 -regularization terms ∥w∥ 1 to L 1 (w) to obtain the enhanced node embedding h v for any node v as h v = arg min w∈R d L 1 (w) + µ 1 ∥w∥ 1 . Thm. 5 derives the above L 1 -regularization problem by employing the proximal gradient descent (PGD) (Nitanda, 2014;Li & Lin, 2015;Duan et al., 2022) method.
Theorem 5. For any node v ∈ V and any feature dimension i ∈ {1, . . . , d}, ( h v ) i in the following equation can efficiently achieve feature selection for ( h v ) i :
( h v ) i = sign ( h v ) i • max |( h v ) i | -µ 1 , 0 , (11
)
where sign(•) denotes the sign function, which takes 1 if
( h v ) i > 0, 0 if ( h v ) i = 0, and -1 if ( h v ) i < 0.
The optimal value for µ 1 is τ 1 B/ dK , where τ 1 ∈ (0, 1), with B as the boundary of the perturbed node features, d as the approximate average degree of the graph, and K as the step parameter of the HOA layer. By Thm. 5, we conclude that Eq. ( 11) can efficiently achieve feature selection for h v , thereby contributing to enhancing the utility of private graph learning. See App. B.4 for the proof.
this section cite: ['b18', 'b2', 'b33', 'b7', 'b34', 'b26', 'b7']

Section: UPGNET IN N-H ARCHITECTURE
Under the N-H architecture of UPGNET, the HOA is consistent with Alg. 1. The NFR specifically aims to enhance utility through efficient feature selection of the perturbed node features x ′ directly using L 1 -regularization. The objective function L 2 of HOA is formalized as follows: Theorem 6. For any node v and any i ∈ {1, . . . , d}, ( x v ) i in Eq. ( 13) can efficiently achieve feature selection:
L 2 (x) = 1 2 ∥x ′ -x∥ 2 2 + µ 2 ∥x∥ 1 . (12
)
( x v ) i = sign ((x ′ v ) i ) • max (|(x ′ v ) i | -µ 2 , 0) ,(13)
where the optimal value for µ 1 is τ 2 B, where τ 2 ∈ (0, 1), with B as the boundary of the perturbed node features. According to Thm. 6, we conclude that Eq. ( 13) can efficiently achieve feature selection for x ′ v , thus enhancing the utility of graph learning. See Sec. C and App. B.5 for more details.
this section cite: []

Section: PRIVACY AND COMPLEXITY ANALYSIS
Privacy Analysis. The PM and MBM satisfy ϵ-LDP for each node. The entire training process remains LDP compliant due to the robustness of DP against the post-processing theorem (Dwork et al., 2014). Moreover, any subsequent prediction is bounded by the post-processing theorem (Dwork et al., 2014), since the LDP protocol is applied only once to the private data. This ensures that LDP holds for all nodes throughout the process. For more details, see App. D.
this section cite: ['b10', 'b10']

Section: Complexity Analysis.
The computational complexity of UPGNET mainly arises from its two key components: the HOA and the NFR. Through analysis, the overall complexity of UPGNET is O(K • |E| • d + |V| • d), scaling linearly with graph size and feature dimensionality. This ensures that UPGNET remains both practical and scalable for largescale graphs with high-dimensional data. See App. E for more detailed analysis and comparisons with baselines.
this section cite: []

Section: Experiments
In this section, we conduct a series of experiments to validate the performance of UPGNET and its core components. More experimental results can be found in App. F.
this section cite: []

Section: Experimental Setting
Datasets. We conduct experiments on four representative graph datasets: Cora (Yang et al., 2016), Citeseer (Yang et al., 2016), LastFM (Rozemberczki & Sarkar, 2020), and Facebook (Rozemberczki et al., 2021). These datasets are commonly used in graph machine learning (Wu et al., 2020;Zhang et al., 2020). Table 1 provides the statistics for these datasets, with specific descriptions as follows:
• Cora and CiteSeer. They are well-known citation networks, where each node represents a scientific paper, and
$FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW (a) Cora (GCN) (b) CiteSeer (GCN) (c) LastFM (GCN) (d) Facebook (GCN) 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW (e) Cora (GraphSAGE) (f) CiteSeer (GraphSAGE) (g) LastFM (GraphSAGE) (h) Facebook (GraphSAGE) 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW 3ULYDF\%XGJHW $FFXUDF\ %$6( 6ROLWXGH /3*11 83*1HW the edges denote citation links. Each node contains a bag-of-words feature vector and a label for each category.
• Facebook. This social network consists of nodes as official Facebook pages, with edges representing mutual liking relationships. Each node has a feature extracted from the site description and a label indicating the category.
• LastFM. Nodes in this dataset represent users of the music streaming service LastFM and links represent friendships between them. The classification task is to predict the users' home country given the artists liked them.
Baselines. To comprehensively assess the performance of UPGNET, we compare it with the following baselines: The NonPriv sets ϵ = ∞ and inputs clean (non-perturbed) node features directly into the GNN for graph learning. In contrast, BASE utilizes the GNN for graph learning directly after using node feature LDP protocols to perturb feature vectors, without incorporating additional utility enhancement strategies. LPGNN (Sajadmanesh & Gatica-Perez, 2021) and Solitude (Lin et al., 2022) apply different strategies to achieve locally differentially private graph learning.
In addition, we consider the multi-bit mechanism (MBM) (Sajadmanesh & Gatica-Perez, 2021) and the piecewise mechanism (PM) (Wang et al., 2019a) independently.
Parameter Settings. All datasets are randomly divided into 50/25/25% for training, validation, and test sets, respectively. To evaluate the performance of UPGNET, we use three representative GNN architectures, graph convolutional networks (GCN) (Kipf & Welling, 2017), Graph-SAGE (Hamilton et al., 2017), and graph attention networks (GAT) (Velickovic et al., 2018) as backbone models. By default, the dataset used is Cora, the LDP protocol applied is the MBM, the GNN model is the GCN, and UPGNET adopts the N-H architecture. For more details, see App. F.1 and F.2.
Evaluation Metrics. Consistent with prior work (Sajadmanesh & Gatica-Perez, 2021;Lin et al., 2022), we conduct experiments on the node classification task, using classification accuracy as the primary metric to evaluate the per- formance of UPGNET. All models undergo 500 training iterations and the best model is chosen for testing based on validation loss. Accuracy is measured over 10 consecutive runs, and we report the average along with 95% confidence intervals calculated by bootstrapping over 1000 samples.
$FFXUDF\ *&1 *UDSK6$*( *$7 $FFXUDF\ 1+ +1(
this section cite: ['b53', 'b53', 'b38', 'b39', 'b51', 'b56', 'b41', 'b28', 'b23', 'b15', 'b47', 'b41', 'b28']

Section: Evaluating the Performance of UPGNET
In this experiment, we vary ϵ in {0.01, 0.1, 1.0, 2.0, 3.0} to thoroughly validate the performance under different noise scales. The experimental results across four datasets and various GNN backbone models are presented in Fig. 3. It shows that, in all cases, UPGNET consistently achieves higher accuracy than BASE, LPGNN and Solitude, and in some instances, it even approaches the accuracy of NonPriv. For example, in the case of
this section cite: []

Section: Comparison of Different GNN Models
Fig. 4(a) intuitively compares the accuracy of UPGNET under different privacy budgets ϵ and across various GNN models. The results reveal that the GAT slightly worse than GCN and GraphSAGE. Specifically, under high privacy settings (e.g., when ϵ = 0.01), the accuracy gap between GAT and the other two models becomes more pronounced. The GAT model introduces an attention mechanism that learns attention coefficients to weight the neighbors in neighborhood aggregation. Consequently, this makes GAT more sensitive to feature perturbations, resulting in a greater degradation of utility in high-noise settings. On the other hand, the utility of GAT remains comparable when ϵ ≥ 0.1.
this section cite: []

Section: Ablation Study on the Performance of NFR
In this experiment, we investigate the utility enhancement of Node Feature Regularizer (NFR) layer for two node feature LDP mechanisms: piecewise mechanism (PM) and multi-bit mechanism (MBM). Table 2 shows the accuracy comparison  The results in Table 2 clearly demonstrate that applying the NFR layer improves graph learning accuracy in all cases. Moreover, Table 2 and Fig. 5 indicates that NFR layer is more effective in improving graph learning accuracy when the privacy budget is small compared to when the privacy budget is large. For instance, in the case of the Cora dataset with MBM perturbing the node features, when ϵ = 0.01, MBM ⋆ improves accuracy by approximately 7% over MBM.
When ϵ = 1.0, MBM ⋆ enhances accuracy by 2.6% over MBM. This is attributed to the fact that when ϵ is small, more noise is injected into the node features, so our NFR layer calibrates the noise and improves the accuracy more significantly.
this section cite: []

Section: Ablation Study on the Performance of HOA
In this experiment, we investigate two aspects: first, whether Higher-Order Aggregator (HOA) layer can mitigate the oversmoothing in multilayer aggregation; and second, whether the HOA can effectively enhance the performance of private graph learning. To achieve these objectives, we compare HOA with SKA and set K = {0, 2, 4, 8, 16, 32, 64} for both HOA and SKA, consider the privacy budget ϵ = 0.01. Fig. 6 shows the performance of HOA compared to SKA on accuracy for different values of K. As illustrated in Fig. 6, for all datasets, the accuracy trend under SKA initially rises with increasing K, but after a certain point, it sharply declines, while the accuracy under HOA continues to improve steadily. For example, in the LastFM dataset (Fig. 6 (c)), SKA's accuracy peaks at K = 4 but then drops rapidly, falling below 40% accuracy at K = 64. This sharp decline is due to the oversmoothing effect in SKA, where larger K values cause node embeddings to become overly similar, diminishing graph learning performance. In contrast, the HOA algorithm shows a steady increase in accuracy as K grows, indicating that the proposed HOA layer successfully mitigates oversmoothing and effectively aggregates useful information from expanded neighborhoods. Furthermore, for K ∈ {2, 4, 8, 16, 32, 64}, HOA consistently outperforms SKA, demonstrating that the HOA can significantly enhance the learning utility. See App. F.3 and F.5 for more details.
K $FFXUDF\ HOA SKA K $FFXUDF\ HOA SKA K $FFXUDF\ HOA SKA K $FFXUDF\ HOA SKA (a) Cora (b) Citeseer (c) LastFM (d) Facebook
this section cite: []

Section: Comparison of Different Architectures
We evaluate the effect of both the N-H and H-N architectures on utility by varying ϵ ∈ {0.01, 0.1, 1.0, 2.0, 3.0} and conducting comparisons using the GCN. As shown in Fig. 4(b), for the Cora, the N-H architecture outperforms the H-N slightly in terms of accuracy. Notably, with smaller ϵ, the N-H architecture excels in feature dimension optimization, which allows it to better handle the information loss from noise injection. However, as ϵ increases, the performance gap between the N-H and H-N architectures narrows, indicating that the early application of the NFR layer in the N-H architecture is more effective in expanding the neighborhood range and enhancing utility when the noise is higher.
this section cite: []

Section: Related work
Recently, a series of works related to locally differentially private GNNs have been proposed. Sajadmanesh & Gatica-Perez (2021) propose a privacy-preserving graph learning framework called LPGNN, which assumes that node features are private and the server has access to the graph topol-ogy, aligning with the scenario of this paper. In LPGNN, each user perturbs their features using the multi-bit mechanism (MBM). However, this paper demonstrates that MBM introduces excessive noise to the feature vector, thereby reducing the utility of the final private graph learning process. Similarly, Du et al. (2021), Lin et al. (2022) and Jin & Chen (2022) utilize the MBM to perturb node features or employ the SKA (Sajadmanesh & Gatica-Perez, 2021;Lin et al., 2022) to calibrate noisy features. Besides MBM, the PM (Wang et al., 2019a;Pei et al., 2023) has also been explored for privacy-preserving graph learning. Yet, it faces similar utility challenges. In contrast to these approaches, our paper introduces a utility-enhanced locally private graph learning framework applicable to various node feature perturbation mechanisms, including MBM and PM, to further enhance the utility of private graph learning.
In addition to node feature perturbation, other efforts have addressed link privacy under LDP. Lin et al. (2022) adopted a naive randomized response mechanism (Qin et al., 2017) to protect adjacency lists. Hidano & Murakami (2024) propose a link LDP mechanism called DPRR, which injects noise into adjacency lists and node degrees respectively and calibrates the noisy links by a degree-sampling method. Zhu et al. (2023) propose a Bayesian estimation-based link LDP mechanism called BLINK. Similar to DPRR, BLINK injects noise into adjacency lists and node degrees, and then estimates the ground truth graph using the adjacency lists as prior and the node degrees as evidence. This work, in contrast, focuses on protecting node features, and is orthogonal to these approaches. It can be seamlessly integrated with existing methods designed to protect neighbor lists.
this section cite: ['b41', 'b28', 'b41', 'b28', 'b35', 'b28', 'b37', 'b17', 'b59']

Section: Conclusion
In this paper, we initially establish a pipeline to generalize the LDP protocols for perturbing node features. Through our analysis, we identify two key factors that affect the estimation error. Building on these insights, we propose UPGNET, which incorporates NFR and HOA layers. The generalization and effectiveness of UPGNET and its components are validated through theoretical analysis and extensive experiments in various datasets and parameter settings.
this section cite: []

Section: References
Ref_id:b0 Title: Higher-order graph convolutional architectures via sparsified neighborhood mixing Year: (2019)
Ref_id:b1 Title: Understanding dropout Year: (2013)
Ref_id:b2 Title: Statistics for highdimensional data: methods, theory and applications Year: (2011)
Ref_id:b3 Title: Simple and deep graph convolutional networks Year: (2020)
Ref_id:b4 Title: Supervised community detection with line graph neural networks Year: (2019)
Ref_id:b5 Title: Collecting telemetry data privately Year: (2017)
Ref_id:b6 Title: Calibrating privacy budgets for locally private graph neural networks Year: ()
Ref_id:b7 Title: Utility analysis and enhancement of ldp mechanisms in high-dimensional space Year: (2022)
Ref_id:b8 Title: Differential privacy: A survey of results Year: (2008)
Ref_id:b9 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b10 Title: The algorithmic foundations of differential privacy Year: (2014)
Ref_id:b11 Title: Randomized aggregatable privacy-preserving ordinal response Year: (2014)
Ref_id:b12 Title: Protein interface prediction using graph convolutional networks Year: (2017)
Ref_id:b13 Title: Diffusion improves graph learning Year: (2019)
Ref_id:b14 Title: On bernstein's inequality Year: (1979)
Ref_id:b15 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b16 Title: Mitigating privacy risks in retrieval-augmented generation via locally private entity perturbation Year: (2025)
Ref_id:b17 Title: Degree-preserving randomized response for graph neural networks under local differential privacy Year: (2024)
Ref_id:b18 Title: Non-negative matrix factorization with sparseness constraints Year: (2004)
Ref_id:b19 Title: Label informed attributed network embedding Year: (2017)
Ref_id:b20 Title: Gromov-wasserstein discrepancy with local differential privacy for distributed structural graphs Year: (2022)
Ref_id:b21 Title: What can we learn privately? Year: (2011)
Ref_id:b22 Title: A method for stochastic optimization Year: (2014)
Ref_id:b23 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b24 Title: Self-normalizing neural networks Year: (2017)
Ref_id:b25 Title: Optimization of gpu-based sparse matrix multiplication for large sparse networks Year: (2020)
Ref_id:b26 Title: Accelerated proximal gradient methods for nonconvex programming Year: (2015)
Ref_id:b27 Title: Graph matching networks for learning the similarity of graph structured objects Year: (2019)
Ref_id:b28 Title: Towards private learning on decentralized graphs with local differential privacy Year: (2022)
Ref_id:b29 Title: Comprehensive graph gradual pruning for sparse training in graph neural networks Year: (2023)
Ref_id:b30 Title: Graph neural networks with adaptive residual Year: (2021)
Ref_id:b31 Title: The group lasso for logistic regression Year: (2008)
Ref_id:b32 Title: Devil in disguise: Breaching graph neural networks privacy through infiltration Year: (2023)
Ref_id:b33 Title: A unified framework for high-dimensional analysis of m-estimators with decomposable regularizers Year: (2017)
Ref_id:b34 Title: Stochastic proximal gradient descent with acceleration techniques Year: (2014)
Ref_id:b35 Title: Privacyenhanced graph neural network for decentralized local graphs Year: (2023)
Ref_id:b36 Title: Linkguard: Link locally privacy-preserving graph neural networks with integrated denoising and private learning Year: (2024)
Ref_id:b37 Title: Generating synthetic decentralized social graphs with local differential privacy Year: (2017)
Ref_id:b38 Title: Characteristic functions on graphs: Birds of a feather, from statistical descriptors to parametric models Year: (2020)
Ref_id:b39 Title: Multi-scale attributed node embedding Year: (2021)
Ref_id:b40 Title: A survey on oversmoothing in graph neural networks Year: (2023)
Ref_id:b41 Title: Locally private graph neural networks Year: (2021)
Ref_id:b42 Title: Model stealing attacks against inductive graph neural networks Year: (2022)
Ref_id:b43 Title: Spiking graph neural network on riemannian manifolds Year: ()
Ref_id:b44 Title: Advances in Neural Information Processing Systems (NeurIPS) Year: (2024)
Ref_id:b45 Title: Motif-aware riemannian graph neural network with generative-contrastive learning Year: (2024)
Ref_id:b46 Title: Learning new words, us patent 9 Year: (2017)
Ref_id:b47 Title: Graph attention networks Year: (2018)
Ref_id:b48 Title: Collecting and analyzing multidimensional data with local differential privacy Year: (2019)
Ref_id:b49 Title: Answering multi-dimensional analytical queries under local differential privacy Year: (2019)
Ref_id:b50 Title: Group property inference attacks against graph neural networks Year: (2022)
Ref_id:b51 Title: A comprehensive survey on graph neural networks Year: (2020)
Ref_id:b52 Title: Local differential privacy and its applications: A comprehensive survey. Computer Standards & Interfaces Year: (2024)
Ref_id:b53 Title: Revisiting semi-supervised learning with graph embeddings Year: (2016)
Ref_id:b54 Title: Topology-aware network pruning using multi-stage graph embedding and reinforcement learning Year: (2022)
Ref_id:b55 Title: Link prediction based on graph neural networks Year: (2018)
Ref_id:b56 Title: Deep learning on graphs: A survey Year: (2020)
Ref_id:b57 Title: Inference attacks against graph neural networks Year: (2022)
Ref_id:b58 Title: Dirichlet energy constrained learning for deep graph neural networks Year: (2021)
Ref_id:b59 Title: Blink: Link local differential privacy in graph neural networks via bayesian estimation Year: (2023)
Ref_id:b60 Title: 2022) and LPGNN (Sajadmanesh & Gatica-Perez, 2021)), the computational complexity of UPGNET introduces only an additional factor |V| • d. This factor is linear with respect to the number of nodes and the feature dimension, making it highly efficient in practice. Moreover, graph pruning Year: (2022)
