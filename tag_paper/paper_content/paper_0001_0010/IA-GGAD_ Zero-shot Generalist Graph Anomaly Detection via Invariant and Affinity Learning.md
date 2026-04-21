Title: IA-GGAD: Zero-shot Generalist Graph Anomaly Detection via Invariant and Affinity Learning
Abstract: Generalist Graph Anomaly Detection (GGAD) extends traditional Graph Anomaly Detection (GAD) from one-for-one to one-for-all scenarios, posing significant challenges due to Feature Space Shift (FSS) and Graph Structure Shift (GSS). This paper first formalizes these challenges and proposes quantitative metrics to measure their severity. To tackle FSS, we develop an anomaly-driven graph invariant learning module that learns domain-invariant node representations. To address GSS, a novel structure-insensitive affinity learning module is introduced, capturing cross-domain structural correspondences via affinity-based features. Our unified framework, IA-GGAD, integrates these modules, enabling anomaly prediction on unseen graphs without target-domain retraining or fine-tuning. Extensive experiments on benchmark datasets from varied domains demonstrate IA-GGAD 's superior performance, significantly outperforming state-of-the-art methods (e.g., achieving up to +12.28 % AUROC over ARC on ACM). Ablation studies further confirm the effectiveness of each proposed module. The code is available at https://github.com/kg-cc/IA-GGAD/.

Section: Introduction
Node-level Graph Anomaly Detection (GAD) has become an essential tool for identifying suspicious entities in complex relational data. Its applications span many critical domains [1,2,3,4,5,6,7]. For example, in finance, anomalous transaction networks can reveal fraud or money laundering [1,4,8]; in social media, detecting abnormal user behavior or bot accounts is essential for maintaining platform integrity [9,10,11]; and in e-commerce, uncovering unusual purchase or review patterns helps identify fraudulent activity [12,13]. Modern GAD techniques, often based on graph neural networks [14,15,16] or statistical models [17,18,19], have achieved great success on individual tasks, but training separate detectors for each graph or domain is time-consuming and costly, limiting the applicability of GAD in unseen anomaly detection environments.
In practice, organizations often manage multiple graph data sources and seek a unified anomaly detector. This motivates Generalist Graph Anomaly Detection (GGAD), a new and challenging setting in which a single model must detect anomalies across diverse graph domains [20,21]. Unlike traditional GAD, which trains a dedicated detector per dataset, a generalist GAD model aims for one model for all domains. Such cross-domain detection is highly significant in real-world deployments;
e.g., Assortative structure e.g., Disassortative structure Graph Structu re Shift! CiteSeer Graph (a citation network) Flicker Graph (a social network) anomaly node anomaly node (b) GSS problem in GGAD tasks Training Domain Target Domain (a) FSS problem in GGAD tasks personal descriptions ... ... Glove, BERT, GPT, etc., embedding • High information density • Rich expressiveness Node Feature: Feat ure Spac e Shift ! Training Domain Target Domain 0 1 0 0 1 0 0 1 0 1 1 1 0 0 1 ... ... bag-of-words embedding Node Feature: • Low information density • Lack expressiveness Citeseer Graph (a citation network) Flicker Graph (a social network) ... ... article title words for example, a single system could detect fraud in a financial transaction network and malicious accounts in a social network using the same model, reducing maintenance burden. However, this generalization requirement introduces substantial domain-shift challenges between different graphs.
Existing GAD approaches have not fully addressed GGAD. Traditional methods typically assume that data comes from a single graph and must be retrained or fine-tuned on each new dataset. Recently, a few works have attempted to develop generalist GAD models. For instance, ARC [20] employs an in-context learning strategy with a feature alignment module to extract cross-dataset anomaly patterns, and Unprompt [22] uses a prompt-based mechanism to unify node representations across graphs. While these methods make strides toward generalization, they still struggle to handle fundamental cross-domain shifts. Specifically, we identify two core domain-shift challenges:
• Feature Space Shift (FSS): Differences in node feature distributions or semantics across domains. Nodes from different domains may differ in feature scale, dimensionality, or meaning, causing models trained on one domain to misinterpret features from another, as illustrated in Figure 1 (a).
• Graph Structure Shift (GSS): Variability in graph structure across different domains.
Graphs can exhibit significantly different connectivity patterns (e.g., community structures, average degrees), causing structural patterns normal in one domain to appear anomalous in another, as illustrated in Figure 1 (b).
These shifts cause existing models to misinterpret node features or structural cues when applied to new graphs. To tackle these challenges, we propose IA-GGAD (Invariance and Affinity Combined Graph Anomaly Detection), a novel framework that explicitly addresses both FSS and GSS. IA-GGAD combines two key components: invariant feature learning and structure-insensitive affinity learning. The invariant feature component learns node representations that capture essential anomalyrelated patterns while being insensitive to domain-specific feature shifts, effectively aligning feature spaces across domains. The graph affinity component learns cross-graph structural correspondences, enabling the model to transfer anomaly cues across heterogeneous structures. These components are trained jointly in a unified architecture. Notably, IA-GGAD requires no access to any target-domain data or labels at test time for retraining or fine-tuning. Once trained on source graphs, the model can be directly applied to an unseen target graph without additional adaptation.
We evaluate IA-GGAD on benchmark datasets from diverse domains, including social networks, citation networks, and e-commerce graphs. Empirical results demonstrate that IA-GGAD achieves state-of-the-art anomaly detection performance across all tasks. It consistently outperforms existing GGAD baselines, achieving substantial improvements in AUROC. For example, on the ACM citation network, IA-GGAD improves AUROC by +12.28% compared with ARC [20]. Importantly, these gains are obtained without any target-domain retraining or labeled data, underscoring the practical utility of our zero-shot approach.
this section cite: ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b3', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b19', 'b21', 'b19']

Section: Related Work
Recent advances in anomaly detection have moved beyond task-specific models toward generalist approaches that apply to diverse domains.
this section cite: []

Section: Graph Anomaly Detection.
Graph anomaly detection (GAD) aims to identify nodes or substructures in a graph that deviate from normal patterns [3,23,24]. Traditional GAD methods are typically trained on a single dataset and can be categorized by supervision level. Supervised GAD uses labeled anomalies (or normal nodes) to train a classifier, but acquiring anomaly labels is often infeasible in practice, such as BGNN [25], BWGNN [3], GHRN [26], and CAGAD [27]. Semi-supervised GAD [28] assumes a subset of nodes are labeled as normal. For example, Qiao et al. [28] propose S-GAD, a generative semi-supervised method that synthesizes artificial outliers from known normal nodes to train a one-class classifier. Unsupervised GAD operates without any labels, often using graph autoencoders or one-class objectives. DOMINANT [15] employs GCNs [14] as an autoencoder to reconstruct graph structure and node attributes and identifies anomalies via the reconstruction error. CoLA [16], based on contrastive learning, uses a discriminator to detect inconsistencies between the target node and neighbor subgraph embeddings. Similarly, TAM [29] and GCTAM [30] extend one-class deep learning to graphs by optimizing an affinity objective over GNN embeddings. These unsupervised methods achieve strong performance on individual graphs but must be retrained for each new graph, limiting their cross-domain generalization.
this section cite: ['b2', 'b22', 'b23', 'b24', 'b2', 'b25', 'b26', 'b27', 'b27', 'b14', 'b13', 'b15', 'b28', 'b29']

Section: Generalist Graph Anomaly Detection.
To overcome the one-model-per-dataset limitation, recent work proposes generalist graph anomaly detection frameworks. Although some GAD approaches [31,32] can handle cross-domain scenarios, their requirement for high correlation (e.g., aligned node features) between source and target datasets limits their generalizability. Differing from those, Liu et al. [20] introduce ARC, a "one-for-all" GAD framework based on in-context learning. ARC aligns node features from different graphs using a learned feature-space projection, encodes residual neighborhood patterns via an ego-neighbor graph encoder, and employs a cross-attentive scoring module that compares nodes to a few-shot set of normal examples. This design allows ARC to adapt to new graphs with minimal additional data. Niu et al. [22] present UNPrompt, a zero-shot generalist GAD model trained on a single source graph. UNPrompt aligns the dimensionality and semantics of node attributes across graphs through coordinate-wise normalization and learns generalized neighborhood prompts so that the predictability of latent node attributes serves as a universal anomaly score. Despite their contributions, ARC and UNPrompt have limitations. ARC still requires a few target-domain normal examples at inference, and its learned alignment may not eliminate all domain shifts. UNPrompt relies on consistent attribute semantics and a single-source training setup, which can limit its applicability when graphs vary widely in feature space or structure.
this section cite: ['b30', 'b31', 'b19', 'b21']

Section: Problem Statement
This section presents the formal problem definition and highlights the key challenges that our framework aims to address.
Preliminaries. Let us define an attributed graph as G = (V, A, X), where
V = {v 1 , . . . , v N } is the set of N nodes, X ∈ R N ×d = {x 1 , • • • , x N }
is the node feature matrix, and A ∈ {0, 1} N ×N is the adjacency matrix such that A ij = 1 indicates an edge between v i and v j . Each node v i is associated with a feature vector x i ∈ R d .
this section cite: []

Section: Task Definition.
In the GGAD setting, we aim to learn a generalist graph anomaly detection model from a collection of labeled source-domain graphs T train = {D test } without access to any target-domain labels or retraining, where n o and n t denote the number of source-domain training datasets and target-domain testing datasets, respectively. Each D (i) = (G (i) , y (i) ) is a labeled dataset from an arbitrary domain. Unlike traditional GAD methods that build graph-specific detectors, our goal is to develop a single generalist graph anomaly detector that can effectively identify anomalies across diverse graph domains with varying semantics and structures.
this section cite: []

Section: Graph Structure Shift (GSS).
To rigorously characterize and quantify the severity of these shifts, we introduce two mathematical formulations:
(1) Feature Space Shift (FSS). Given the node feature matrices X o and X t from source and target domains, respectively, we use Maximum Mean Discrepancy (MMD) to measure the distance between their feature distributions in a reproducing kernel Hilbert space (RKHS). Specifically,
FSS(X o , X t ) = 1 N 2 o i,j ⟨ϕ(x o i ), ϕ(x o j )⟩ + 1 N 2 t i,j ⟨ϕ(x t i ), ϕ(x t j )⟩ - 2 N o N t i,j ⟨ϕ(x o i ), ϕ(x t j )⟩,(1)
where ϕ(•) is an RKHS mapping function, referring to Eq.( 3), and N o , N t denote the number of nodes in source and target graphs, respectively. A larger FSS score reflects greater feature misalignment, leading to unreliable inference across domains.
(2) Graph Structure Shift (GSS). Similarly, we quantify structural discrepancies using degree-derived structural vectors and a Gaussian kernel f (x, y) = exp(-∥x -y∥ 2 /2σ 2 ). Given adjacency matrices A o and A t from source and target graphs, GSS is computed as:
GSS(A o , A t ) = 1 N 2 o i,j f ([A o • A o ] i , [A o • A o ] j ) + 1 N 2 t i,j f ([A t • A t ] i , [A t • A t ] j ) - 2 N o N t i,j f ([A o • A o ] i , [A t • A t ] j ).(2)
This metric reflects the misalignment in higher-order structural properties such as connectivity profiles and neighborhood distributions. A high GSS score indicates a greater structure divergence between source and target graphs.
this section cite: []

Section: Empirical Evidence.
To further investigate the FSS and GSS challenges, we conduct a quantitative analysis of both FSS and GSS, with detailed results provided in Appendix B.
this section cite: []

Section: Method
To address the twin challenges of FSS and GSS identified in Section 3, we introduce IA-GGAD, an end-to-end, zero-shot framework that learns to detect anomalies on unseen graphs after training on multiple, diverse source domains. Figure 2 gives an overview of IA-GGAD. IA-GGAD is composed of four tightly coupled modules: (1) Invariant feature pool construction (Section 4.
1): align various node representations and extract a shared invariant feature pool composed of domaininvariant prototypes; (2) Graph-invariant representation learning (Section4.2): embeds node representations using the shared invariant feature pool ensuring that normal and abnormal patterns are consistently separated across domains; (3) Structure-insensitive affinity learning (Section 4.3): learns affinity-based node representations to enable structure-insensitive cross-graph affinity scoring;
(4) Joint anomaly scoring and prediction (Section 4.4): fuses semantic and structural evidence to produce a final, domain-agnostic anomaly score and predictions.
this section cite: []

Section: Invariant Feature Pool Construction
The first step toward cross-domain generalization is to eliminate feature divergence while retaining critical anomaly-related and normal patterns.
Feature alignment. Empirical evidence in Table 5 shows that node attributes differ widely in scale, dimensionality, and semantics across datasets. We therefore project every raw feature matrix
X ∈ R N ×di to a shared latent dimension d u : X = proj X ∈ R N ×du ,
where proj(•) is a learnable linear layer that can be initialized with classical dimensionality-reduction techniques such as PCA [33]. This simple yet effective step standardizes the input space for the subsequent graph encoder.
this section cite: ['b32']

Section: Shared graph encoder.
Following alignment, we feed X into a stack of parameter-shared Graph Convolutional Networks (GCNs) [14] to obtain multihop node embeddings:
Z [1] = σ D -1 2 AD -1 2 XW [1] , . . . , Z [ℓ] = σ D -1 2 AD -1 2 Z [ℓ-1] W [ℓ] = z [ℓ]
1 , . . . , z
[ℓ] where A is the adjacency matrix with self-loops, D its degree matrix, W [ℓ] are learnable weights, σ(•) is an activation function, and Z [ℓ] is the transformed representation matrix at the ℓ-th layer.
this section cite: ['b13']

Section: Invariant feature extraction.
To make the representations immune to feature-space shift, we adopt a modified Vector-Quantized VAE (VQ-VAE) [34]. Each source graph contributes to a shared invariant features pool E = {e 1 , . . . , e M } ⊂ R M ×d I whose entries serve as domain-invariant prototypes.
The invariant features pool is initialized from sampled embeddings and refined by the standard assignment-update routine:
e (l-1) i = 1 |N i | Ni j z [ℓ](l) j , N i = argmin-k j∈{1,••• ,N } (||z [ℓ](l) j -e (l-1) i || 2 ), e (0) i = z [ℓ](1) i ,(4)
where e (l-1) i denotes the result of the i-th invariant feature prototype at the (l -1)-th iteration. In each iteration, e i is updated as the mean of its k nearest node embeddings z i . After convergence, a lightweight decoder reconstructs the aligned features xi from the selected prototype, and the entire module is trained with the composite loss:
L inv = 1 N N i=1 1 - xi • xi ∥x i ∥ • ∥x i ∥ γ + 1 N N i=1 ∥sg[z i ] -e i ∥ 2 2 + η N N i=1 ∥sg[e i ] -z i ∥ 2 2 . (5
)
The composite loss comprises three intuitive terms. (i) Reconstruction. The scaled cosine error 1 -xi
• xi ∥xi∥∥xi∥ γ (γ ≥ 1) enforces faithful recovery of aligned attributes, encouraging the encoder to retain informative semantics. (ii) VQ update. The vector-quantization loss ∥ sg[z i ] -e i ∥ 2 2 pushes each invariant features pool entry e i toward its assigned encoder output z i , progressively refining the invariant prototypes. (iii) Commitment. The commitment term ∥ sg[e i ] -z i ∥ 2 2 penalizes large deviations of the encoder from the selected prototype, stabilizing training. A single hyper-parameter η (empirically 0.25) balances invariant features pool adaptation and encoder commitment, while the stop-gradient operator sg[•] prevents back-propagation through the detached branch.
this section cite: ['b33']

Section: Graph-Invariant Representation Learning
While conventional GNNs consume only the ℓ-layer output embeddings, IA-GGAD enriches every node representation with explicit invariant semantics extracted from the invariant features pool E. This fusion yields embeddings that are simultaneously sensitive to local graph context and robust to cross-domain feature shifts. +Invariant-feature guided embedding fusion. For each GCN layer ℓ, we retrieve the k invariant features pool vectors whose cosine similarity with z
[ℓ]
i is highest. Averaging these invariant features and concatenating with z
[ℓ]
i gives an invariant-aware embedding:
h [ℓ] i = concat(z [ℓ] i , 1 |T i | j∈Ti e j ), T i = argmax-k j∈{1,...,N } ( z [ℓ] i • e j |z [ℓ] i | • |e j | ),(6)
where h
[ℓ]
i ∈ R de is the output embedding matrix with invirant code emebding, T i is a set of k invariant feature emebding that are closest to the current node representation z i .
this section cite: []

Section: Multi-hop residual encoding.
To capture how a node's representation drifts across message-passing layers, we take successive differences with the 1-hop baseline h [1] i and concatenate them:
r i = (h [2] i -h [1] i ) || (h [3] i -h [1] i ) || • • • || (h [ℓ] i -h [1] i ),(7)
where || denotes vector concatenation. The resulting vector r i encodes multi-scale deviations that are highly informative for anomaly assessment.
Residual similarity loss. Given source-domain labels, we encourage residuals of normal nodes to cluster while pushing those of abnormal nodes away beyond a margin ϵ:
L res = N + t t N + i i (1 - r + t • r + i ||r + t || • ||r + i || ) + N + t t N - j j max(0, r + t • r - j ||r + t || • ||r - j || -ϵ),(8)
where r + t and r - j denote residuals of normal and anomalous nodes, respectively. Joint optimization. The invariant feature pool, GCN encoder, and residual module are optimized jointly with L = L inv + L res , such that semantic invariance (Section 4.1) and residual discriminability reinforce each other throughout training. As a result, the learned node embeddings exhibit strong generalization ability to unseen graphs, even in the presence of severe feature space shift (FSS).
this section cite: []

Section: Structure-Insensitive Affinity Learning
FSS-robust embeddings alone are insufficient when two graphs differ markedly in structure. To explicitly cope with GSS, we introduce a Graph Affinity Encoder (GAE) that learns a homophily-driven local affinity score. Normal nodes are expected to exhibit high affinity with their neighbours, whereas anomalous nodes break this pattern and thus receive low affinity. Optimising this contrast yields a structure-aware signal that complements the invariant semantics learned in Section 4.2.
this section cite: []

Section: GNN-based node projection.
GAE first maps each node to a latent space that captures neighbourhood context. For computational efficiency and fair comparison with prior work, we employ a single Graph Convolutional Network (GCN) layer [14]:
H = σ(D -1 2 AD -1 2 XW) = { h1 , • • • , hN }, (9
)
where X is the feature-aligned input from Section 4.1, A the adjacency matrix with self-loops, and σ(•) an activation function.
this section cite: ['b13']

Section: Local affinity score.
Given H, we quantify how well node v i conforms to its immediate neighbourhood N (v i ) via the average cosine similarity:
AS(v i ) = 1 |N (v i )| vj ∈N (vi) hi • hj | hi || hj | .(10)
A high value indicates strong homophily-typical for normal nodes; low values signal structural irregularities, hinting at anomalies.
this section cite: []

Section: Unsupervised affinity maximisation.
We train GAE in an unsupervised manner by maximising each node's local affinity:
L aff = min Θ - N i=1 AS(v i ) .(11)
Optimizing (11) encourages neighbourhood-coherent representations for the vast majority of normal nodes, implicitly relegating anomalous nodes to the low-affinity tail.
this section cite: []

Section: Joint Anomaly Scoring and Prediction
Having trained the three upstream modules on the source-domain set T train , IA-GGAD performs inference on any unseen graph by issuing two complementary anomaly signals that mirror the twin challenges of FSS and GSS.
this section cite: []

Section: Residual Score RS.
The residual embeddings r i defined in Eq. ( 7) faithfully encode feature-space deviations that may arise from FSS. We measure how isolated a node is within this residual space via the mean-squared distance to n k random sample residual spaces:
RS(v i ) = 1 n k n k j ∥r i -r j ∥ 2 ,(12)
where r j are residuals from the test graph obtained with the frozen encoder.
Local Affinity Score AS. Complementing RS, the affinity score in Eq. ( 10) probes structural homophily and is thus sensitive to GSS-induced anomalies.
this section cite: []

Section: Weighted Fusion.
Because the two scores live on different scales, we blend them with a weighting factor λ ∈ [0, 1]:
S(v i ) = (1 -λ) RS(v i ) + λ 1 -AS(v i ) .(13)
Intuitively, a node is deemed suspicious if it shows either a large residual dispersion (semantic oddity) or a weak local affinity (structural oddity).
this section cite: []

Section: Adaptive thresholding.
To convert the anomaly score S(v i ) into a binary prediction, we adopt the data-driven rule in Eq. ( 14). The optimal threshold τ * maximises the separation between normal and anomaly sets, after which nodes with S(v i ) ≥ τ * are labelled as anomalies.
             τ * = arg max τ ∈{S(v)|v∈N }   1 |N + | vi∈N + I (S(v i ) ≥ τ ) - 1 |N -| vj ∈N - I (S(v j ) ≥ τ )   ŷi = 1, ifS(v i ) ≥ τ * 0, ifS(v i ) < τ *(14)
Discussion. The weighted fusion couples the strengths of semantic (FSS-oriented) and structural (GSS-oriented) cues, while the adaptive threshold obviates manual calibration on each new graph. Together, these choices complete an end-to-end zero-shot pipeline whose predictions remain reliable across dramatic domain shifts. Detailed algorithmic description and complexity analysis of IA-GGAD can be found in Appendix C.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Dataset configuration. Following ARC [20], we create a deliberately shifted source/target split. The training set is T train = PubMed, Flickr, Questions, YelpChi , while the zero-shot test set is T test = ACM, Facebook, Amazon, Cora, CiteSeer, BlogCatalog, Reddit, Weibo . All graphs contain injected or naturally occurring anomalies [35,16,29], providing a realistic benchmark for generalization.
Baselines. We compare IA-GGAD with sixteen strong competitors: six supervised GNNs (GCN [14], GAT [36], BGNN [25], BWGNN [3], GHRN [26], CAGAD [27]), one semi-supervised model (S-GAD [28]), five unsupervised methods (DOMINANT [15], CoLA [16], HCM-A [37], TAM [29], GCTAM [30]), and the two state-of-the-art generalist approaches ARC [20] and UNPrompt [22]. Implementation details for all baselines are provided in Appendix F.
this section cite: ['b19', 'b34', 'b15', 'b28', 'b13', 'b35', 'b24', 'b2', 'b25', 'b26', 'b27', 'b14', 'b15', 'b36', 'b28', 'b29', 'b19', 'b21']

Section: Evaluation protocol.
We report AUROC and AUPRC averaged over five random seeds, together with standard deviations [35,38]. Each method, baselines and IA-GGAD alike, is trained once on T train and then evaluated on every target graph in a pretrain only manner. Feature dimensions are aligned by inserting either a learnable or a random projection adapter before the input layer. Hyperparameters are selected via random search on the training split and remain fixed across target graphs, with no dataset-specific tuning. For IA-GGAD, the sample count is set to n k = 10 unless specified. Further implementation details can be found in Appendix G. ∆ represents the improvement (↑) or degradation (↓) compared to the current best baseline method. Rank indicates the average ranking over 8 datasets. * represents reproduced results, others are reported in ARC [20].
this section cite: ['b34', 'b37', 'b19']

Section: Main Results
Table 1 summarises AUROC performance across fourteen benchmarks. IA-GGAD achieves the top rank on 7/8 datasets, with especially large gains of +12.28% on ACM and +10.46% on Facebook. These results confirm that our joint handling of FSS and GSS yields robust zero-shot generalisation. Most strikingly, IA-GGAD outperforms the strongest baselines on Amazon (+3.11%), Cora (+1.23%), and CiteSeer (+0.88%), demonstrating consistent improvements even on highly diverse graphs. Traditional supervised GAD methods (e.g. GCN, GAT, BGNN, GHRN) fall near chance without retraining, and unsupervised approaches like DOMINANT and TAM perform well only when their implicit homophily assumptions hold. The sole exception is Weibo, where DOMINANT's reconstruction-based detector attains 92.88% versus our 91.18% (-1.70%), which remains a very high score. We attribute this to Weibo's extremely strong local homophily and low attribute variance, which favour autoencoder reconstruction errors over invariant-feature alignment. In contrast, IA-GGAD 's invariant pool and affinity encoder balance semantic and structural cues, leading to more stable performance across weaker-homophily graphs. Finally, IA-GGAD 's mean rank of 1.12 far surpasses ARC (2.38) and GCTAM (4.37), and its low standard deviations (all <3%) underscore stable behaviour across random seeds. Full AUPRC results appear in Appendix H.1. To disentangle the impact of each module, we evaluate three variants: (i) w/o I&A-backbone only;
this section cite: []

Section: Ablation Study
(ii) w/I-backbone plus invariant feature pool; (iii) w/A-backbone plus graph affinity encoder. Table 2 shows that invariant features chiefly improve FSS-dominated graphs (e.g. CiteSeer, Cora), while affinity modelling is crucial for GSS-heavy graphs (ACM, Facebook, Amazon). Combining both yields the full model, outperforming every variant on all datasets. Figure 3 studies the weighting factor λ in Eq. ( 13). On both ACM and Facebook, performance peaks at λ = 0.9, confirming that a modest contribution from affinity scores complements the residual signal without amplifying noise. Beyond this point, AUROC and AUPRC decline, underscoring the need for balanced fusion. Additional sensitivity plots are provided in Appendix 4. Table 3 pairs four targets with a common source (PubMed) and reports the quantitative shift measures from Eq. ( 1)-( 2). ACM and Facebook exhibit high GSS but negligible FSS; CiteSeer and Cora show the opposite pattern. GCTAM excels under GSS but degrades under FSS; ARC shows the reverse. By contrast, IA-GGAD outperforms both baselines in all scenarios, confirming that the dual design-semantic invariance for FSS, affinity learning for GSS-generalises robustly across heterogeneous graphs. To evaluate the scalability of IA-GGAD, we test it on large-scale financial networks with real anomalies, including DGraph [39], T-Finance [3], and Elliptic [40]. We compare IA-GGAD with recent state-of-the-art methods such as CONSISGAD [41], SmoothGNN [42], AnomalyGFM [43], and SpaceGNN [44]. IA-GGAD consistently achieves the best performance. On Elliptic, it reaches an AU-ROC of 74.24, outperforming the secondbest (SpaceGNN, 57.43) by 16.81 %. On T-Finance, it scores 75.41, surpassing the nextbest (AnomalyGFM, 67.57) by 11.75 %. On DGraph-Fin, where many baselines fail due to memory constraints, IA-GGAD still attains 54.39. These results demonstrate its robustness and scalability on large and complex financial graphs.
this section cite: ['b38', 'b2', 'b39', 'b40', 'b41', 'b42', 'b43']

Section: Hyper-parameter Sensitivity

this section cite: []

Section: Impact of FSS and GSS

this section cite: []

Section: Performance on Large Datasets

this section cite: []

Section: Limitations
Although IA-GGAD substantially advances zero-shot graph anomaly detection, several limitations remain. (1) Dependence on informative node attributes. The invariant-feature module assumes moderately descriptive features; on purely structural graphs or graphs with sparse or noisy attributes, its FSS-mitigation effect can disappear and performance degrades. (2) Homogeneity assumption in affinity learning. The affinity encoder is based on the homogeneity assumption, which does not directly apply to heterogeneous graphs; however, by employing meta-path, IA-GGAD can be extended to heterogeneous graph scenarios. (3) Static structures. Our evaluation covers only static graphs; dynamic (time-varying) graphs remain out of scope and likely require substantive extensions. (4) Limited scale analysis. Scalability to million-scale graphs is untested, and both the k-NN prototype retrieval and global affinity objective may need approximate or mini-batch variants.
(5) Hyper-parameter sensitivity. Results depend on weighting factor λ; a comprehensive robustness study is left to future work.
this section cite: []

Section: Conclusion
We presented IA-GGAD, an end-to-end zero-shot framework for GGAD that detects anomalies on unseen graphs without retraining. By formalising and quantifying Feature Space Shift (FSS) and Graph Structure Shift (GSS), we pinpointed the key barriers to cross-domain generalisation. IA-GGAD counters them with two lightweight modules: an anomaly-driven invariant feature pool to mitigate FSS and a graph affinity encoder to withstand GSS. Across eight real-world datasets, IA-GGAD outperforms fourteen strong baselines-including ARC and UNPrompt-achieving up to +12.28% AUROC on ACM and +10.46% AUROC on Facebook, thereby setting a new benchmark for zero-shot graph anomaly detection.
this section cite: []

Section: Future work.
Although treating FSS and GSS separately proves effective, a unified representation that jointly normalizes feature semantics and structural patterns could further improve robustness. In addition, we plan to extend IA-GGAD to dynamic graphs and to graphs with heterogeneous node and edge types, which would broaden its applicability in real-world scenarios.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: The abstract and introduction clearly summarize the key contributions. These statements align with the theoretical analysis and experimental results and accurately reflect the scope of the paper.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: The limitations of this paper have been discussed in Section 6 "Limitations".
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
Answer: [Yes] Justification: All theorems clearly state their assumptions and the full proofs are provided in Appendix A, ensuring correctness and completeness.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: We include experimental instructions in the paper and provide detailed dataset configurations (Appendix E), baseline implementations and hyperparameters (Appendix G), and a public code repository for our framework (https://anonymous.4open.science/  r/GGAD-8F0F/), ensuring that all information needed to reproduce the main results is available.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: Yes, we provide the computing infrastructures in Appendix G.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: This research conforms with the Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: There is no societal impact of the work performed.
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
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We cite all original publications for the baseline code packages. Note that these are CC-BY 4.0 licensed. We also specify the sources of public datasets in our public repository.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: No LLMs were used in the design, implementation, or evaluation of our framework. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: A Theoretical Foundations of FSS and GSS
This appendix provides a rigorous foundation for FSS and GSS as defined in the main text. We formalize the kernels underlying Eqs. (1)-( 2), address well-definedness across domains of different sizes, establish an additive-kernel decomposition, and present a domain-adaptation bound under explicit assumptions.
this section cite: []

Section: A.1 Notation and Background
RKHS and kernels. A Hilbert space (H, ⟨•, •⟩) is an RKHS for a positive-definite kernel k if there exists a feature map ϕ with k(u, v) = ⟨ϕ(u), ϕ(v)⟩.
Population and empirical MMD. For distributions P, Q over a common space Z, the (population) MMD is
MMD k (P, Q) = ∥µ k (P ) -µ k (Q)∥ H k , µ k (P ) = E z∼P [ϕ(z)].
Given i.i.d. samples S = {s i } n S i=1 ∼ P and T = {t j } n T j=1 ∼ Q, the biased empirical estimator is
MMD 2 k (S, T ) = 1 n 2 S i,i ′ k(s i , s i ′ ) + 1 n 2 T j,j ′ k(t j , t j ′ ) -2 n S n T i,j k(s i , t j ).(
A.1) Assume throughout that the kernel is bounded, sup z k(z, z) ≤ κ 2 , and that samples are drawn i.i.d. when invoking concentration bounds. Then MMD k (P, Q) -MMD k (P, Q) = O p 1/n S + 1/n T [45]. A.2 Formalizing the Kernels Behind Eqs. (1)-(2) FSS kernel consistent with Eq. (1). Let G denote a graph domain and let ϕ G : R d → R h be the shared graph encoder with a single set of shared parameters held fixed when computing MMD. Define a single kernel on graph-tagged inputs
k f (x, G), (x ′ , G ′ ) := ϕ G (x), ϕ G ′ (x ′ ) .
This is positive-definite since it is an inner product in a common feature space via the unified feature map Φ : (x, G) → ϕ G (x) ∈ R h . With this kernel, the empirical MMD 2 k f between source and target node-feature pairs reproduces Eq. ( 1).
this section cite: []

Section: GSS kernel consistent with Eq. (2). For each node i in graph G, the vector [A
G •A G ] i has length N G .
To ensure well-defined comparisons when N o ̸ = N t , fix a size-harmonization operator ρ : R N• → R p (e.g., truncation/aggregation of walk-counts, heat signatures at p scales, or any domain-agnostic deterministic map), applied identically across domains and independent of the data. Define
s G i := ρ [A G •A G ] i ∈ R p , k g (i, G), (j, G ′ ) := exp - ∥s G i -s G ′ j ∥ 2 2σ 2 .
When N o = N t and ρ is the identity, Eq. ( 2) is recovered verbatim; otherwise, Eq. ( 2) is interpreted with the implicit ρ for well-definedness.
this section cite: []

Section: A.3 Definitions of FSS and GSS (as MMDs)
Let X o = {(x o i , G o )} No i=1 and X t = {(x t j , G t )} Nt j=1 be source/target node-feature pairs; let S o = {(i, G o )} No
i=1 and S t = {(j, G t )} Nt j=1 be the node-index pairs for structure. We define
FSS := MMD 2 k f X o , X t ,(
A.2) GSS := MMD 2 kg S o , S t . (A.3) By construction, (A.2) reproduces Eq. (1); (A.3) coincides with Eq. ( 2) when N o = N t and ρ is identity, and otherwise implements its size-harmonized counterpart.
this section cite: []

Section: A.4 Additive Kernel and Decomposition
For the joint representation U = ((x, G), (i, G)) with feature part Φ(x, G) = ϕ G (x) and structure part
s G i = ρ([A G A G ] i ), define the additive kernel k U, U ′ = k f (x, G), (x ′ , G ′ ) + k g (i, G), (j, G ′ ) .(
A.4) Its RKHS is the orthogonal direct sum H k = H k f ⊕ H kg , hence the mean embeddings decompose. Let P U s , P U t be the joint distributions of U on source/target. Then MMD 2 k (P U s , P U t ) = MMD 2 k f (P Φ s , P Φ t ) + MMD 2 kg (P S s , P S t ) = FSS + GSS. (A.5)
this section cite: []

Section: A.5 Key Properties
Theorem 1 (Non-negativity and identity of indiscernibles). For positive-definite kernels k f , k g , FSS ≥ 0 and GSS ≥ 0. If k f (resp. k g ) is characteristic on its input space, then FSS = 0 (resp. GSS = 0) iff the corresponding source and target distributions coincide.
Proof. Immediate from MMD as the RKHS distance between mean embeddings; characteristic kernels yield injective mean embeddings [45].
this section cite: ['b44']

Section: A.6 Domain-Adaptation Bound via MMD
We bound the generalization gap using the IPM property of MMD under explicit assumptions.
Assumption 1 (Covariate shift). P s (Y | U = u) = P t (Y | U = u) for all u in the support of U .
this section cite: []

Section: Assumption 2 (RKHS capacity on conditional risk). For each hypothesis h,
define g h (u) = E Y |U =u [ℓ(h(u), Y )]
and assume g h ∈ H k with ∥g h ∥ H k ≤ B. Assume also that the loss is bounded, 0 ≤ ℓ ≤ M (used when relating empirical and population risks).
this section cite: []

Section: Theorem 2 (Generalization gap under additive-kernel discrepancy).
Under Assumptions 1-2,
ϵ t (h) -ϵ s (h) ≤ B MMD k P U s , P U t = B √ FSS + GSS. (A.6)
Moreover, with probability at least 1 -δ over i.i.d. draws of N o source and N t target samples,
ϵ t (h) ≤ ϵ s (h) + B MMD k S U , T U + C ln(2/δ) No+Nt , (A.7)
for a constant C = C(B, κ) depending only on the RKHS radius B and the kernel bound κ.
Proof sketch. Under Assumption 1, ϵ d (h) = E U ∼P U d [g h (U )] for d ∈ {s, t} with the same g h . By the IPM characterization of MMD, sup ∥f ∥ H k ≤1 E s f (U ) -E t f (U ) = MMD k (P U
s , P U t ). Since ∥g h ∥ H k ≤ B, we obtain (A.6); (A.7) follows by replacing the population MMD with its empirical counterpart and applying concentration for bounded kernels, yielding a deviation term that depends on B and κ.
this section cite: []

Section: A.7 Conclusion
This appendix formalized FSS and GSS as squared MMDs computed with well-defined kernels on shared, fixed-dimension spaces. Concretely, we: (1) specified a single positive-definite feature kernel k f induced by the shared encoder (parameters fixed when computing MMD), yielding a valid empirical MMD 2 k f ; (2) introduced a size-harmonization operator to define a structure kernel k g consistently across graphs of different sizes; and (3) established the additive-kernel identity
MMD 2 k = FSS + GSS for k = k f + k g .
Under covariate shift and an RKHS capacity assumption on the conditional risk, we derived the domain-adaptation bound |ϵ t (h) -ϵ s (h)| ≤ B √ FSS + GSS, with an empirical counterpart including a standard concentration term depending only on B and κ. These results justify minimizing both FSS and GSS in our algorithm design-respectively via invariant feature learning and structure-insensitive affinity learning-to reduce the generalization gap on unseen graphs.
this section cite: []

Section: B Experimental Analysis of FSS and GSS
To further understand how IA-GGAD tackles domain shifts, we analyze the quantitative behaviors of FSS and GSS as defined in Eq. ( 1) and Eq. ( 2), and examine the effect of the final score fusion strategy defined in Eq. ( 13). As shown in Fig. 4, our anomaly scoring mechanism effectively mitigates the domain shift challenges posed by FSS and GSS. By adaptively weighting the residual-based semantic score and the affinity-based structural score, IA-GGAD balances two orthogonal cues, resulting in robust performance across both FSS-dominated and GSS-dominated datasets. Feature Space Shift (FSS). Table 5 reports the average FSS scores between each source and target domain, providing a quantitative measure of the distributional misalignment in the node feature space. Based on the average FSS values, we categorize the target datasets into three types of FSS domains:
this section cite: []

Section: B.1 FSS Scores Analysis
• Low-FSS domains (FSS < 0.05): Reddit (0.0193), ACM (0.0214), BlogCatalog (0.0221), and Amazon (0.0321) exhibit minimal feature space shift from the source domains. Their node attributes are highly compatible with those in the training graphs, likely due to shared semantics such as user interactions, co-occurrence structures, or platform-generated tags. As such, these domains allow for direct knowledge transfer with negligible adaptation cost.
• Moderate-FSS domains (0.05 ≤ FSS < 0.4): Facebook (0.3494) demonstrates moderate misalignment in its node feature space. While not as challenging as high-FSS domains, the shift indicates partial semantic divergence, potentially stemming from demographic-specific behaviors or inconsistent attribute ontologies. Alignment strategies are still necessary to ensure effective transfer.
• High-FSS domains (FSS ≥ 0.4): Cora (0.4937), Weibo (0.5482), and Citeseer (0.7032) represent severely misaligned domains with pronounced semantic drift. Such high FSS scores suggest substantial differences in feature distributions, likely caused by sparse vocabulary, heterogeneous encodings, or conflicting representation schemes (e.g., bag-of-words vs. contextual embeddings). These domains demand robust invariant encoding mechanisms to support generalization under extreme domain shifts.
this section cite: []

Section: B.2 GSS Scores Analysis

this section cite: []

Section: Graph Structure Shift (GSS).
Table 6 reports the average GSS scores from each source domain to each target domain, quantifying the degree of structural distributional shift. Based on these scores, we classify the target domains into three levels of structural misalignment:
• Low-GSS domains (GSS < 0.35): Amazon (0.3095) and Citeseer (0.2970) exhibit minimal structural deviation from the source graphs. Their graph topologies-such as degree distributions, community structures, and connectivity statistics-align closely with those seen in training domains. These domains are structurally compatible and require little to no adaptation for generalization.
• Moderate-GSS domains (0.35 ≤ GSS < 0.5): Cora (0.3608), Weibo (0.3996), Facebook (0.4583), and Reddit (0.4996) fall into the moderate shift category. These domains show partial topological divergence, possibly due to differences in local density, edge formation policies, or subgraph structures. Moderate adaptation via structure-aware encoders remains beneficial here.
• High-GSS domains (GSS ≥ 0.5): BlogCatalog (0.5200) and ACM (0.5719) exhibit strong structural misalignment. These domains likely differ in both macro-scale topology (e.g., degree skewness, small-worldness) and micro-scale motifs. As a result, traditional structural priors become unreliable, necessitating robust affinity modeling or structureinvariant mechanisms for effective transfer.
this section cite: []

Section: B.3 Empirical Evidence
The Solution of FSS and GSS Challenges. To balance feature space shift and graph structure shift across domains, we adopt a weighted fusion scheme (Eq. 13) controlled by a weighting factor λ ∈ [0, 1]. Our empirical study (Fig. 4) reveals a clear correspondence between the optimal choice of λ and the underlying FSS/GSS characteristics of each dataset. Specifically, datasets suffering from substantial semantic shift, such as Citeseer (FSS = 0.7032, GSS = 0.2970) and Cora (FSS = 0.4937, GSS = 0.3608)-achieve peak performance at lower values of λ (e.g., 0.1-0.3). This indicates that the residual-based semantic score RS(v i ) plays a dominant role in these scenarios, where invariant feature alignment is crucial for mitigating cross-domain semantic discrepancies. In contrast, structurally misaligned domains like ACM (FSS = 0.0214, GSS = 0.5719) and Facebook (FSS = 0.3494, GSS = 0.4583) require larger λ values (e.g., 0.7-0.9), reflecting a stronger dependence on the affinity-based structural score AS(v i ). In these cases, topological deviations from the source domains dominate, making structural modeling the primary means of anomaly detection. These findings confirm that our fusion mechanism flexibly adapts to the dominant domain shift type-semantic or structural-thereby enabling robust zero-shot generalization without manual tuning.
this section cite: []

Section: C Algorithm and Complexity

this section cite: []

Section: C.1 Algorithmic description
The algorithmic description of the training and inference process of IA-GGAD is summarized in Algorithm. 1, and Algorithm. 2, respectively.
(a) Amazon 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
The impact of Algorithm 1: The Training algorithm of IA-GGAD Input: Training datasets T train . Parameters :Number of epoch T ; GCNs layers: ℓ . 1 Generate initial parameters for all learnable parameters. 2 for D ∈ T train do 3 Obtain aligned features X from X via feature alignment. 4 end 5 for each epoch t = 0, 1, 2, ..., T do 6
for D ∈ T train do 7 Obtain X, A, V, y from D 8 Z [1] , • • • , Z [ℓ] ←
GCNs transform embedding via Eq. (3) 9 Initial and update E = {e 1 , e 2 , • • • , e M } with Z [l] via Eq. (4) 10 for ℓ = 1 : L do 11 h [ℓ] i ← concat Z [l] and E via Eq. (6) 12 r i ← Calculate residual of h ℓ i and h [1] i via Eq. (7) 13 end 14 Calculate loss L inv and L res via Eq. (5) and Eq. (8) Update model parameters via gradient descent. 19 end 20 end Algorithm 2: The Inference algorithm of IA-GGAD Input: Test dataset T test Parameters :Well-trained IA-GGAD model weight parameters. 1 for D ∈ T test do 2 Obtain aligned features X from X via feature alignment. 3 end 4 Obtain X, A, V from D 5 Z [1] , • • • , Z [ℓ] ← GCNs transform embedding via Eq. (3) 6 for l = 1 : L do 7 h [ℓ] i ← concat Z [l] and pretrianed E via Eq. (6) 8 r i ← Calculate residual of h ℓ i and h [1] i via Eq. (7) 9 end 10 Obatain node affinity emebding H via Eq. (9) 11 Calculate residual dispersion score RS(v i ) by Eq. (12) 12 Calculate nodes affinity score AS(v i ) by Eq. (10) 13 Joint anomaly scoring and predictionS(v i ) by Eq. (13)
this section cite: []

Section: C.2 Complexity Analysis
Training Phase. The total time complexity in the training phase of IA-GGAD consists of four main components:
• Feature alignment: projecting raw node features X ∈ R n×d into a unified latent space with dimension d u via linear projection (e.g., PCA), resulting in complexity O(ndd u ).
• Graph representation learning: using L-layer GCNs for message passing and residual encoding. This includes feature propagation O(md u ) and residual encoding O(nd u h+nh 2 ), leading to an overall complexity of O(L(md u + nd u h + nh 2 )).
• Invariant prototype updating: constructing and updating a invariant features pool E of M domain-invariant features using nearest neighbor assignment: O(nM d u ).
• Affinity learning: calculating affinity scores between each node and its neighbors using cosine similarity: O(n d h), where d is the average node degree, h is the affinity emebding.
Thus, the total training complexity is:
O ndd u + L(md u + nd u h + nh 2 ) + nM d u + n d h
In our experiments, the total training time across all datasets was approximately 9 minutes for five independent runs (with different random seeds), conducted on an NVIDIA GeForce RTX A6000 GPU.
Testing Phase. The inference process involves feature alignment, embedding generation, and anomaly scoring:
• Feature alignment: similar to training, O(ndd u ) for projection, smoothness computation, and feature reordering.
• Embedding generation: O(L(md u + nd u h + nh 2 )), as in training.
• Anomaly scoring:
-Residual dispersion scoring via k sample in residual space: O(nkh).
-Affinity scoring via neighbor cosine similarity: O(n dh).
-Weighted fusion and thresholding: O(n).
Therefore, the total inference complexity is:
O ndd u + L(md u + nd u h + nh 2 ) + nkh + n d h
In our experiments, the total inference time across all datasets was approximately 3-5 seconds when performed on an NVIDIA GeForce RTX A6000 GPU.
this section cite: []

Section: D Detailing Related Work
Anomaly Detection. Anomaly detection (AD) aims to identify samples that deviate significantly from the norm [5]. Due to the scarcity of labeled anomalies, most AD methods operate in an unsupervised manner. To extract anomaly patterns without supervision, existing approaches leverage techniques such as one-class classification [46], distance-based scoring [47], reconstruction [15], generative modeling [48], and self-supervised learning [16]. For instance, DeepSVDD [46] formulates an objective to learn compact representations within a hypersphere, capturing the common modes of normal data. AnoDDPM [48] adopts a partial diffusion process with simplex noise to detect anomalies effectively, particularly in high-resolution image settings. While effective, these methods are typically specialized for the domain on which they are trained, limiting their ability to generalize to unseen datasets or cross-domain scenarios.
this section cite: ['b4', 'b45', 'b46', 'b14', 'b47', 'b15', 'b45', 'b47']

Section: Cross-Dataset Anomaly Detection.
To overcome dataset-specific limitations, recent anomaly detection (AD) research has explored generalization across domains. Some methods tackle domain shifts by leveraging distribution alignment or domain adaptation [49,50], yet they often assume semantic similarity between source and target domains, which limits broader applicability. A more flexible strategy is the few-shot setting, where a small number of normal samples from the target domain are available to guide detection. Under this paradigm, RegAD [51] learns a transferable model that generalizes to new domains without retraining. WinCLIP [52] and InCTRL [53] exploit vision-language models (VLMs) such as CLIP to achieve zero-or few-shot image anomaly detection.
Despite their impressive performance on image data, these methods rely heavily on pre-trained vision encoders and tailored architectures, limiting their transferability to graph-based domains.
this section cite: ['b48', 'b49', 'b50', 'b51', 'b52']

Section: Anomaly Detection on Graph Data.
Depending on the granularity of anomalies, graph-based anomaly detection (AD) methods can be broadly classified into three categories: node-level [15,3], edge-level [54], and graph-level [55,56]. Among these, node-level AD has attracted the most attention due to its wide applicability in real-world scenarios [6]. In this work, we focus specifically on node-level anomaly detection and adopt the term Graph Anomaly Detection (GAD), in line with common usage in prior literature [29,3,23].
this section cite: ['b14', 'b2', 'b53', 'b54', 'b55', 'b5', 'b28', 'b2', 'b22']

Section: Graph Anomaly Detection.
Early graph anomaly detection (GAD) methods primarily relied on shallow techniques. For instance, AMEN [17] detects anomalies by modeling attribute correlations within each node's ego-network. Residual analysis was also commonly employed, such as in Radar [18], which evaluates inconsistencies between node attributes and structural information. ANOMALOUS [19] further integrates attribute selection with anomaly scoring via CUR decomposition. Although these methods achieved reasonable performance on low-dimensional graphs, they struggle with complex structures and high-dimensional attributes due to their limited representational capacity [57,58]. To overcome the limitations of shallow methods, graph neural networks (GNNs) have become the dominant paradigm for graph anomaly detection (GAD). Existing GNN-based approaches can be broadly categorized into supervised and unsupervised settings [6,35]. Supervised GAD assumes access to labeled normal and anomalous nodes, and research in this direction primarily focuses on improving convolutional architectures and task-specific objective functions [59,3]. For example, CARE-GNN [59] enhances fraud detection by incorporating label-aware neighbor aggregation via reinforcement learning to combat disguise attacks. Spectral methods provide another perspective, linking anomalies to high-frequency signals in the graph spectrum. BWGNN [3] introduces band-pass filters to capture localized spectral patterns, while GHRN [26] further emphasizes high-frequency components by pruning inter-class edges, isolating anomalous nodes more effectively. Unlike supervised methods, unsupervised graph anomaly detection (GAD) does not rely on labeled data. Inspired by unsupervised anomaly detection in images, these approaches adopt various learning paradigms-such as reconstruction, contrastive learning, and auxiliary pretext tasks-to uncover node-level anomalies. For example, DOMINANT [15] leverages a graph autoencoder to jointly reconstruct adjacency and attribute matrices, detecting anomalies via reconstruction errors. ComGA [23] enhances detection by integrating community structure and tailored GCNs to capture local and structural anomalies. CoLA [16] introduces contrastive self-supervised learning into GAD, constructing instance pairs without labels to learn discriminative node representations. HCM-A [37] incorporates hop prediction and Bayesian learning to integrate multiscale context. More recently, TAM [29] proposes a homophily-aware affinity score, optimized end-to-end on a truncated graph structure to better isolate anomalous nodes. GCTAM [30] extends one-class deep learning to graphs by optimizing an affinity objective over global GNN embeddings.
this section cite: ['b16', 'b17', 'b18', 'b56', 'b57', 'b5', 'b34', 'b58', 'b2', 'b58', 'b2', 'b25', 'b14', 'b22', 'b15', 'b36', 'b28', 'b29']

Section: Generalist Graph Anomaly Detection.
Nevertheless, all the above methods adhere to the conventional paradigm of "one-for-one" dataset. Recent work proposes generalist graph anomaly detection frameworks. ARC [20], a "one-for-all" GAD framework based on in-context learning, which aligns node features from different graphs using a learned feature-space projection, encodes residual neighborhood patterns via an ego-neighbor graph encoder, and employs a cross-attentive scoring module that compares nodes to a few-shot set of normal examples. UNPrompt [22], a zero-shot generalist GAD model trained on a single source graph, which aligns the dimensionality and semantics of node attributes across graphs through coordinate-wise normalization and learns generalized neighborhood prompts so that the predictability of latent node attributes serves as a universal anomaly score. Despite their contributions, ARC and UNPrompt have limitations. ARC still requires a few target-domain normal examples at inference, and its learned alignment may not eliminate all domain shifts. UNPrompt relies on consistent attribute semantics and a single-source training setup, which can limit its applicability when graphs vary widely in feature space or structure. In contrast, our IA-GGAD directly addresses these gaps. IA-GGAD is a zero-shot generalist GAD framework requiring no target-specific data or fine-tuning.
this section cite: ['b19', 'b21']

Section: E Description of Datasets
Following ARC [20], we evaluate our model on 12 benchmark datasets, categorized into four groups:
(1) citation networks with injected anomalies, (2) social networks with injected anomalies, (3) social networks with real anomalies, and (4) co-review networks with real anomalies. For each category, we designate the largest dataset as the training source, while the remaining datasets serve as testing targets. This setting enables a comprehensive evaluation of the generalization capability of our proposed IA-GGAD model.
Table 7 summarizes the statistics of all datasets. The selected datasets span diverse domains and include both synthetic and real-world anomalies, ensuring the model is exposed to a wide range of anomaly types. This diversity is essential for equipping IA-GGAD with the ability to generalize effectively to unseen graphs. Detailed descriptions of each dataset are provided below. • Cora, CiteSeer, PubMed [60] and ACM [61] are four widely-used citation network datasets. In these datasets, nodes correspond to scientific publications, and edges represent citation relationships between them. Each node is described by a bag-of-words feature vector, where the dimensionality is determined by the size of the vocabulary specific to each dataset.
• BlogCatalog and Flickr [15] are representative social network datasets, where users are connected via mutual following relationships. Each user is represented as a node, and edges denote social connections. Node attributes are derived from user-generated textual content within the platform, including blog posts, photo tags, and other descriptive metadata.
• Amazon and YelpChi [62,63] are datasets that capture user-review interactions to identify opinion fraud. The Amazon dataset is constructed to detect users who were incentivized to post fake product reviews. Following prior work [29], three graph variants are derived from Amazon using different relational schemes to form adjacency matrices. YelpChi, on the other hand, focuses on detecting deceptive reviews on Yelp.com that unfairly promote or defame businesses. Based on [62,64], three graph variants are also constructed for YelpChi, incorporating relationships among users, review content, and timestamps. In this study, we specifically adopt the Amazon-UPU variant (where edges connect users who reviewed at least one common product) and the YelpChi-RUR variant (where edges connect reviews posted by the same user).
• Facebook [65] is a social network in which users can build relationships with others and share their friends.
• Reddit [66] is a forum-based social network dataset collected from the Reddit platform. In this dataset, users who have been banned are labeled as anomalies. Each node represents a user, and edges reflect interactions such as replies or shared threads. The textual content of user posts is encoded into vector representations and used as node attributes.
• Weibo [66] is a social media dataset derived from the Tencent Weibo platform, comprising a graph of users and their associated hashtags. Within a defined temporal window (e.g., 60 seconds), if a user posts consecutively, the behavior is considered potentially suspicious. Users who exhibit at least five such instances are labeled as "suspicious" and treated as anomalies. Node features include geolocation data of microblog posts and bag-of-words representations of the textual content.
• Questions [67] dataset originates from Yandex Q, a platform dedicated to questionanswering. Users represent the nodes, while the connections between them signify the presence or absence of a question-and-answer interaction within a one-year timeframe. Node features are constructed by averaging the FastText embeddings of the words in each user's profile description. An additional binary feature is included to denote users with missing descriptions.
this section cite: ['b19', 'b59', 'b60', 'b14', 'b61', 'b62', 'b28', 'b61', 'b63', 'b64', 'b65', 'b65', 'b66']

Section: Anomaly Injection.
For datasets with injected anomalies, we follow the injection strategy introduced in [15,16]. Specifically, we directly adopt the publicly available datasets from ARC [20], in which anomalies have already been injected using standardized procedures. In summary, the injection process perturbs both graph structure and node attributes. Structurally, anomalous cliques are created by densely connecting randomly selected nodes, simulating unnatural substructures. For attribute perturbations, features of selected nodes are replaced with those from the most dissimilar nodes to ensure a significant semantic shift. The total number of anomalies is controlled proportionally to the dataset size. More detailed statistics of all datasets are summarized in Table 7.
this section cite: ['b14', 'b15', 'b19']

Section: F Description of Baselines
In our evaluation, we present a comprehensive comparison of IA-GGAD against a variety of graph anomaly detection (GAD) methods, including supervised, semi-supervised, and unsupervised paradigms. We also include comparisons with recent SOTA generalist GAD (GGAD) approaches, which aim to perform anomaly detection across diverse datasets using a single unified model.
this section cite: []

Section: Supervised Method.
For the supervised setting, we consider two classical GNN architectures as well as four state-of-the-art (SOTA) models specifically developed for the GAD task. These methods assume access to labels for both normal and anomalous nodes during training. Accordingly, the problem is framed as a binary node classification task, where the goal is to accurately distinguish anomalous nodes from normal ones.
• GCN [14] is a seminal model in the development of graph neural networks (GNNs). It leverages neighborhood aggregation to effectively capture local graph structure, enabling efficient node feature extraction and representation learning for graph-structured data.
• GAT [36] introduces an attention mechanism into the GNN framework, allowing the model to dynamically assign weights to neighboring nodes. This enhances its adaptability across downstream tasks by producing context-aware node representations.
• BGNN [25] integrates gradient boosted decision trees (GBDTs) with GNNs to effectively handle graphs with tabular node features. While GBDTs manage feature heterogeneity, the GNN component captures structural dependencies, leading to superior performance on mixed-type data.
• BWGNN [3] utilizes spectrally and spatially localized band-pass filters to address the "rightshift" phenomenon in graph anomalies, where abnormal nodes tend to exhibit high-frequency spectral energy concentrations.
• GHRN [26] is a heterophily-aware supervised GAD model based on spectral analysis. By enhancing high-frequency signals and pruning inter-class edges, GHRN effectively isolates anomalous nodes and improves detection performance.
• CAGAD [27] employs a graph pointer network to identify heterophilic anomalies-nodes embedded in neighborhoods dominated by normal nodes. It generates counterfactual representations by aggregating information from unseen neighbors, enhancing anomaly detection in an unsupervised manner.
Semi-supervised Method. For the semi-supervised method, S-GAD [28], a recently proposed method specifically designed for graph anomaly detection with access to only a small subset of labeled normal nodes.
• S-GAD [68] is a semi-supervised GAD method that uses a few labeled normal nodes to train a one-class classifier. It generates learnable pseudo-anomalies based on asymmetric local affinity and egocentric closeness, which serve as negative samples to enhance anomaly detection without requiring labeled anomalies.
Unsupervised Methods.For the unsupervised methods, we consider 5 representative SOTA GAD methods, each of them belonging to a sub-type: data reconstruction, contrastive learning, hop-based auxiliary goal, or affinity-based auxiliary goal:
• DOMINANT [15] combines GCN and deep auto-encoder, and its learning objective is to reconstruct the adjacency matrix and node features jointly. It aims to identify structural and attribute anomalies based on reconstruction errors.
• CoLA [16] is a contrastive self-supervised learning for anomaly detection on graphs with node attributes. The framework captures the relationship between each node and its neighborhood substructure in an unsupervised manner by sampling novel pairs of contrasting instances and leveraging the local information of the graph.
• HCM-A [37] uses hop-count prediction as a self-supervised task to better identify anomalies by modeling both local and global context information. In addition, HCM-A designs two new anomaly scores and introduces Bayesian learning to train the model to capture anomalies. • 1 TAM [29] is designed based on one-class homophily and local affinity. The learning target of TAM is to optimize the proposed anomaly metric (i.e., affinity) end-to-end on the truncated adjacency matrix.
• 2 GCTAM [30] is an unsupervised GAD method that enhances truncated affinity maximization by combining contextual and global affinity truncation. It introduces two key modules: contextual affinity truncation (CAT), which reduces the influence of anomalous nodes by cutting weak contextual links, and global affinity truncation (GAT), which enhances affinity among normal nodes. By integrating both modules through shared GCNs, GCTAM generates node representations that better reflect homophily and irregularity, significantly boosting anomaly detection performance across real-world datasets.
this section cite: ['b13', 'b35', 'b24', 'b25', 'b26', 'b27', 'b67', 'b14', 'b15', 'b36', 'b28', 'b29']

Section: Generalist GAD methods.
For the generalist setting, we consider 2 state-of-the-art GGAD methods that aim to detect anomalies across diverse domains using a unified model. These methods do not rely on dataset-specific training or adaptation and are designed to generalize to unseen graphs via in-context learning, prototype alignment:
• 3 ARC [20] is a few-shot generalist GAD method based on in-context learning. It uses a residual graph encoder to extract anomaly-aware node features and a cross-attention module to reconstruct query nodes from a few labeled normal context nodes. Anomaly scores are computed by measuring residual distance between original and reconstructed embeddings, allowing ARC to detect anomalies across unseen graphs without fine-tuning.
• 4 UNPrompt [22] is a zero-shot generalist GAD method that unifies node attributes across graphs via coordinate-wise normalization and learns transferable normal/abnormal patterns through neighborhood prompt learning. It performs anomaly detection by measuring latent attribute predictability without any training or labels on the target graphs.
this section cite: ['b19', 'b21']

Section: G Details of Implementation
Hyper-parameters. We select some key hyperparameters of IA-GGAD through random search within specified grids. Specifically, the random search was performed within the following search space:
• Hidden layer dimension: {64, 128, 256, 512, 1024}
• Number of invariant and affinity encoder layers: {1, 2, 3}
• Dropout rate: {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
• Learning rate: floats between 10 -5 and 10 -2
• Weight decay: floats between 10 -6 and 10 -3
• Number of invariant features: {1024, 2048, 4096, 8192}
Baseline Implementation. We adopt a unified set of hyperparameters to construct a generalist GAD model applicable across all datasets. All methods, including the proposed IA-GGAD and baseline models, are first trained on the source training set T train using full anomaly labels. Subsequently, each model is evaluated independently on every dataset in the target test datasets T test, without any retraining or fine-tuning. For feature projection, we apply Principal Component Analysis (PCA) to map the raw node attributes into a fixed-dimensional latent space with d u = 64. In cases where the original feature dimension is smaller than d u , we first apply a random projection (e.g., Gaussian random projection) to upscale the features, followed by PCA to ensure uniform dimensionality alignment at d u . For CAGAD [27], S-GAD [28], GCTAM [30], and UNPrompt [22], we reproduce their results using their official implementations and conduct optimal hyperparameter tuning. For all other baselines, we follow the reproduction settings reported in ARC [20]. It is worth noting that all methods are trained and evaluated under the same standardized experimental pipeline to ensure fair comparison.
this section cite: ['b26', 'b27', 'b29', 'b21', 'b19']

Section: Metrics.
Following [35,29,38], we employ two popular and complementary evaluation metrics for evaluation, including area under the receiver operating characteristic Curve (AUROC) and area under the precision-recall curve (AUPRC). A higher AUROC/AUPRC value indicates better performance. We report the average AUROC/AUPRC with standard deviations across 5 runs.
this section cite: ['b34', 'b28', 'b37']

Section: Implementation Details.
The experiments in this study were conducted on a Linux server running Ubuntu 20.04. The server was equipped with a 13th Gen Intel(R) Core(TM) i7-12700 CPU, 64GB of RAM, and an NVIDIA GeForce RTX A6000 GPU (48GB memory). For software, we used Anaconda3 to manage the Python environment and PyCharm as the development IDE. The specific software versions were Python 3.8.14, CUDA 11.7, DGL 0.9.1, and PyTorch 2.0.1 [69].
this section cite: ['b68']

Section: H Supplementary Experiments

this section cite: []

Section: H.1 Performance Comparison of AUPRC
Table 8 presents the anomaly detection performance in terms of AUPRC across eight target datasets. IA-GGAD achieves the best overall ranking (1.87), consistently outperforming both conventional GAD baselines and recent generalist methods. It ranks first on five datasets (ACM, Amazon, Cora, CiteSeer, Weibo) and second on two (BlogCatalog, Reddit), demonstrating strong adaptability to diverse semantic and structural shifts. Notable gains over the strongest baseline, ARC, include +17.83% on ACM, +12.29% on Amazon, and +6.77% on Weibo. On Facebook-ARC's strongest domain-IA-GGAD remains competitive (6.55% vs. 8.38%, -1.83% gap). While ARC ranks second overall, its performance varies across datasets. GCTAM performs well on ACM and Facebook but fails to generalize under structural shift. Traditional models like GCN, GAT, and BGNN perform poorly, particularly on complex graphs such as Reddit and Weibo, highlighting their limited transferability. These results validate the effectiveness of IA-GGAD's fusion of invariant semantic features and structure-aware affinity encoding, enabling robust zero-shot anomaly detection across heterogeneous domains.
this section cite: []

Section: 
some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: All the datasets are included along with the uploaded source code.
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
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: All the experimental details are given in Section 5) and Appendix G). Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: All the experimental results are acquired by multiple trails of experiments, and we report the average and standard deviation results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
this section cite: []

Section: References
Ref_id:b0 Title: Fraud detection: A systematic literature review of graph-based anomaly detection approaches Year: (2020)
Ref_id:b1 Title: Charu Aggarwal, and Guansong Pang. Deep graph anomaly detection: A survey and new perspectives Year: (2024)
Ref_id:b2 Title: Rethinking graph neural networks for anomaly detection Year: (2022)
Ref_id:b3 Title: Internet financial fraud detection based on graph learning Year: (2022)
Ref_id:b4 Title: Longbing Cao, and Anton Van Den Hengel. Deep learning for anomaly detection: A review Year: (2021)
Ref_id:b5 Title: A comprehensive survey on graph anomaly detection with deep learning Year: (2021)
Ref_id:b6 Title: Integrating visualised automatic temporal relation graph into multi-task learning for alzheimer's disease progression prediction Year: (2024)
Ref_id:b7 Title: Remember: Ranking metric embedding-based multicontextual behavior profiling for online banking fraud detection Year: (2021)
Ref_id:b8 Title: Rumor detection on social media with bi-directional graph convolutional networks Year: (2020)
Ref_id:b9 Title: Trust-aware detection of malicious users in dating social networks Year: (2022)
Ref_id:b10 Title: Noisehgnn: Synthesized similarity graph-based neural network for noised heterogeneous graph representation learning Year: (2025)
Ref_id:b11 Title: Understanding electricity-theft behavior via multi-source data Year: (2020)
Ref_id:b12 Title: Mining fraudsters and fraudulent strategies in large-scale mobile social networks Year: (2019)
Ref_id:b13 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b14 Title: Deep anomaly detection on attributed networks Year: (2019)
Ref_id:b15 Title: Anomaly detection on attributed networks via contrastive self-supervised learning Year: (2021)
Ref_id:b16 Title: Scalable anomaly ranking of attributed neighborhoods Year: (2016)
Ref_id:b17 Title: Radar: Residual analysis for anomaly detection in attributed networks Year: (2017)
Ref_id:b18 Title: Anomalous: A joint modeling approach for anomaly detection on attributed networks Year: (2018)
Ref_id:b19 Title: Arc: A generalist graph anomaly detector with in-context learning Year: (2024)
Ref_id:b20 Title: Information maximizing adaptation network with label distribution priors for unsupervised domain adaptation Year: (2022)
Ref_id:b21 Title: Zero-shot generalist graph anomaly detection with unified neighborhood prompts Year: (2024)
Ref_id:b22 Title: Comga: Community-aware attributed graph anomaly detection Year: (2022)
Ref_id:b23 Title: Alleviating structural distribution shift in graph anomaly detection Year: (2023)
Ref_id:b24 Title: Boost then convolve: Gradient boosting meets graph neural networks Year: (2021)
Ref_id:b25 Title: Addressing heterophily in graph anomaly detection: A perspective of graph spectrum Year: (2023)
Ref_id:b26 Title: Counterfactual data augmentation with denoising diffusion for graph anomaly detection Year: (2024)
Ref_id:b27 Title: Ee-Peng Lim, and Guansong Pang. Generative semi-supervised graph anomaly detection Year: (2024)
Ref_id:b28 Title: Truncated affinity maximization: One-class homophily modeling for graph anomaly detection Year: (2023)
Ref_id:b29 Title: Gctam: Global and contextual combined truncated affinity maximization model for unsupervised graph anomaly detection Year: (2025)
Ref_id:b30 Title: Cross-domain graph anomaly detection Year: (2021)
Ref_id:b31 Title: Cross-domain graph anomaly detection via anomaly-aware contrastive alignment Year: (2023)
Ref_id:b32 Title: Principal component analysis Year: (2010)
Ref_id:b33 Title: Neural discrete representation learning Year: (2017)
Ref_id:b34 Title: Gadbench: Revisiting and benchmarking supervised graph anomaly detection Year: (2024)
Ref_id:b35 Title: Graph attention networks Year: (2018)
Ref_id:b36 Title: Hop-count based self-supervised anomaly detection on attributed networks Year: (2022)
Ref_id:b37 Title: Toward deep supervised anomaly detection: Reinforcement learning from partially labeled anomaly data Year: (2021)
Ref_id:b38 Title: Dgraph: A large-scale financial dataset for graph anomaly detection Year: (2022)
Ref_id:b39 Title: Can abnormality be detected by graph neural networks? Year: (2022)
Ref_id:b40 Title: Consistency training with learnable data augmentation for graph anomaly detection with limited supervision Year: (2024)
Ref_id:b41 Title: Smoothgnn: Smoothing-aware gnn for unsupervised node anomaly detection Year: (2025)
Ref_id:b42 Title: Anomalygfm: Graph foundation model for zero/few-shot anomaly detection Year: (2025)
Ref_id:b43 Title: Spacegnn: Multi-space graph neural network for node anomaly detection with extremely limited labels Year: (2025)
Ref_id:b44 Title: A kernel two-sample test Year: (2012)
Ref_id:b45 Title: Deep one-class classification Year: (2018)
Ref_id:b46 Title: Towards total recall in industrial anomaly detection Year: (2022)
Ref_id:b47 Title: Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise Year: (2022)
Ref_id:b48 Title: Catching both gray and black swans: Open-set supervised anomaly detection Year: (2022)
Ref_id:b49 Title: Cross-domain video anomaly detection without target domain adaptation Year: (2023)
Ref_id:b50 Title: Registration based few-shot anomaly detection Year: (2022)
Ref_id:b51 Title: Winclip: Zero-/few-shot anomaly classification and segmentation Year: (2023)
Ref_id:b52 Title: Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts Year: (2024)
Ref_id:b53 Title: Anomaly detection in dynamic graphs via transformer Year: (2021)
Ref_id:b54 Title: Towards self-interpretable graph-level anomaly detection Year: (2024)
Ref_id:b55 Title: Unifying unsupervised graph-level anomaly detection and out-of-distribution detection: A benchmark Year: (2024)
Ref_id:b56 Title: Towards data-centric graph machine learning: Review and outlook Year: (2023)
Ref_id:b57 Title: Learning strong graph neural networks with weak information Year: (2023)
Ref_id:b58 Title: Enhancing graph neural network-based fraud detectors against camouflaged fraudsters Year: (2020)
Ref_id:b59 Title: Collective classification in network data Year: (2008)
Ref_id:b60 Title: Arnetminer: extraction and mining of academic social networks Year: (2008)
Ref_id:b61 Title: Collective opinion spam detection: Bridging review networks and metadata Year: (2015)
Ref_id:b62 Title: From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews Year: (2013)
Ref_id:b63 Title: What yelp fake review filter might be doing? Year: (2013)
Ref_id:b64 Title: Contrastive attributed network anomaly detection with data augmentation Year: (2022)
Ref_id:b65 Title: Predicting dynamic embedding trajectory in temporal interaction networks Year: (2019)
Ref_id:b66 Title: A critical look at the evaluation of gnns under heterophily Year: (2023)
Ref_id:b67 Title: Ee-Peng Lim, and Guansong Pang. Generative semi-supervised graph anomaly detection Year: (2024)
Ref_id:b68 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b69 Title:  Year: ()
