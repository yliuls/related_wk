Title: MODALITY-FREE GRAPH IN-CONTEXT ALIGNMENT
Abstract: In-context learning (ICL) converts static encoders into task-conditioned reasoners, enabling adaptation to new data from just a few examples without updating pretrained parameters. This capability is essential for graph foundation models (GFMs) to approach LLM-level generality. Yet current GFMs struggle with crossdomain alignment, typically relying on modality-specific encoders that fail when graphs are pre-vectorized or raw data is inaccessible. In this paper, we introduce Modality-Free Graph In-context Alignment (MF-GIA), a framework that makes a pretrained graph encoder promptable for few-shot prediction across heterogeneous domains without modality assumptions. MF-GIA captures domain characteristics through gradient fingerprints, which parameterize lightweight transformations that align pre-encoded features and indexed labels into unified semantic spaces. During pretraining, a dual prompt-aware attention mechanism with episodic objective learns to match queries against aligned support examples to establish prompt-based reasoning capabilities. At inference, MF-GIA performs parameter-update-free adaptation using only a few-shot support set to trigger cross-domain alignment and enable immediate prediction on unseen domains. Experiments demonstrate that MF-GIA achieves superior few-shot performance across diverse graph domains and strong generalization to unseen domains. The code is available at https://github.com/JhuoW/MF-GIA. Published as a conference paper at ICLR 2026 2024), and GFT (Wang et al., 2024b) compromise ICL through required post-training on downstream graphs, where All in One trains task-specific prompts while GCOPE and GFT require fine-tuning; GPF (Fang et al., 2023) and Prodigy (Huang et al., 2023) lack cross-domain alignment, limiting their generalization on graphs from unseen domains; recent advances like UniGraph (He et al., 2025a), GraphAlign (Hou et al., 2024), and OFA (Liu et al., 2024a) achieve alignment and post-training-free inference, yet sacrifice modality freedom by requiring conversion to a single unified modality (e.g., text-attributed graphs) for alignment, making them inapplicable when raw data are inaccessible or when graphs are already pre-encoded by domain-specific pipelines. More related work is discussed in Appendix A.In this work, we present Modality-free Graph In-context Alignment (MF-GIA), the first GFM to achieve all three criteria for true in-context learning on graphs. Our key insight is that the interaction between a graph and a shared frozen encoder reveals its domain characteristics, which can be captured by a gradient fingerprint: a single-step parameter update that encodes how features, labels, and structure jointly influence the model. This fingerprint drives lightweight domain-conditioned transformations that align pre-encoded features and graph-local label IDs into unified semantic spaces, where related domains occupy neighboring subspaces while preserving intra-domain geometry, thereby achieving modality-free domain alignment. The aligned features and labels are then processed by Dual Prompt-Aware Attention (DPAA) optimized with an episodic objective that learns to match queries against support examples. This approach establishes prompt-based in-context reasoning that simulates the few-shot scenarios faced at test time. At inference, given a few labeled examples as prompts, MF-GIA computes the fingerprint, instantiates the aligners, and performs parameterupdate-free prediction on unseen domains. Experiments across diverse benchmarks demonstrate that MF-GIA excels at few-shot node-level tasks, generalizes to entirely unseen domains without additional training, and transfers seamlessly to edge-level tasks. These results bring GFMs closer to the universal in-context learning capabilities exhibited by LLMs.

Section: INTRODUCTION
Table 1: Comparison of methods with respect to the three main criteria of true ICL.
this section cite: []

Section: Method
Post-Training Free Domain Alignment Modality-Free SSL-GNN ✗ ✗ ✓ All in One (Sun et al., 2023) ✗ ✗ ✓ GPF (Fang et al., 2023) ✓ ✗ ✓ GCOPE (Zhao et al., 2024) ✗ ✓ ✓ GFT (Wang et al., 2024b) ✗ ✓ ✗ Prodigy (Huang et al., 2023) ✓ ✗ ✓ Unigraph (He et al., 2025a) ✓ ✓ ✗ AutoGFM (Chen et al., 2025) ✗ ✓ ✗ GraphAlign (Hou et al., 2024) ✓ ✓ ✗ OFA (Liu et al., 2024a) ✓ ✓ ✗ GOFA (Kong et al., 2025
) ✓ ✓ ✗ MF-GIA ✓ ✓ ✓
The remarkable success of Large Language Models (LLMs) has fundamentally revolutionized AI, with in-context learning (Brown et al., 2020;Zhang et al., 2023;Lu et al., 2022) emerging as a pivotal capability that enables these models to adapt to new tasks through mere exposure to a few demonstration examples, without any parameter updates like fine-tuning. This paradigm shift, from task-specific fine-tuning to prompt-based adaptation, naturally sparks a profound question for the graph learning community: Can we achieve similar foundation-level generality for graph-structured data? Unlike sequential text where context flows naturally, graphs encode complex topological patterns, multihop dependencies, and heterogeneous node and edge attributions that demand fundamentally new approaches to demonstration selection, prompt design, and reasoning.
Achieving true graph in-context learning demands three fundamental criteria that remain elusive in existing methods. First, post-training-free inference is essential for genuine ICL, where models must adapt to new tasks through demonstrations alone, without fine-tuning or learnable prompt engineering. Second, cross-domain alignment enables a single model to reason across diverse graph types within a unified semantic space, mirroring LLMs' domain-agnostic capabilities. Third, modality-free operation ensures that the model can process arbitrary pre-encoded graphs without requesting raw data, crucial for the heterogeneous domains of real-world graphs. As shown in Table 1, prior approaches fall short of meeting all three criteria at once: self-supervised GNNs (You et al., 2020;Qiu et al., 2020) and GFMs like All in One (Sun et al., 2023), GCOPE (Zhao et al.,
this section cite: ['b50', 'b7', 'b66', 'b21', 'b2', 'b18', 'b25', 'b1', 'b65', 'b34', 'b60', 'b43', 'b50']

Section: PRELIMINARIES
Following the ICL setup of the pioneering work Prodigy (Huang et al., 2023), we study few-shot, prompt-based node and edge classification. In this section, we formalize graph ICL as episodic classification over graphs and introduce a modality-free alignment perspective that standardizes features and labels across domains.
this section cite: ['b21']

Section: GRAPH IN-CONTEXT LEARNING
Let G = {G 1 , G 2 , • • • , G M } denote a collection of M graphs drawn from heterogeneous domains, where each graph G i = (V i , E i , X i , Y i ) comprises a node set V i , an edge set E i ⊆ V i × V i , node features X i = {x i,1 , • • • , x i,|Vi| } ∈ R |Vi|×di
, and labels Y i . The node features X i may exist in domain-specific formats (e.g., dense vectors, categorical attributes, IDs), with potentially different dimensions d i . To pretrain a universal model across these graphs with a common input width, following (Yu et al., 2025;Zhao et al., 2024)
v ∈ V i is h i,v = f θ (v; G i ) ∈ R d .
For edge classification, we analogously obtain h i,e = f θ (e; G i ) for e ∈ E i using endpoint features and structure as needed. We use the generic symbol w to denote an item (w = v or w = e).
Given G as a pretraining corpus with M graphs and a target graph
G new = (V new , E new , X new , Y new )
from an unseen domain with C new classes, graph in-context learning aims to classify graph items in G new using a few labeled examples per class as in-context demonstrations, without updating model parameters. Formally, the graph ICL operates in two phases. During pretraining, we learn a unified model M Φ : G → Y on the corpus G. At test time, given a support set S = {(w j , y j )} k•Cnew j=1 containing k labeled graph items per class from G new as prompts, the model predicts labels for query items Q = {q : q ∈ G new \S} as: ŷq = M Φ (q, G new , S) , ∀q ∈ Q, where the pretrained model M Φ is parameterized by Φ. Crucially, Φ remains frozen during inference, so the model leverages the in-context demonstrations in S to adapt to the new domain without fine-tuning. For example, consider M Φ pretrained on citation and E-commerce networks. When tested on a social network G new from an unseen domain, the model can classify users in G new without fine-tuning. Instead, we provide a support set containing a few labeled users from each class. By leveraging these in-context demonstrations as prompts, M Φ identifies patterns between the support examples and query users to classify the remaining users, all while keeping its parameters frozen.
this section cite: ['b61', 'b66']

Section: Episodic Meta Learning.
To enable in-context adaptation, we adopt an episodic training paradigm (Vinyals et al., 2016;Li et al., 2019). Specifically, for each pretraining graph G i ∈ G, we construct m-way k-shot episodes by sampling m classes and k examples per class as a support set S, with additional samples as queries Q. The model M Φ consumes (G i , S) as the prompt and is optimized to maximize the likelihood of the ground-truth labels on Q:
min Φ E[- 1 |Q| q∈Q log p (y q | q, S, G i )].(2)
This episodic formulation teaches the model to recognize patterns from limited examples. By pretraining on numerous episodes that simulate the few-shot scenarios encountered at test time, the model acquires the capacity to perform in-context reasoning. At inference, this enables adaptation to new domains through few-shot prompts alone, with all pretrained parameters Φ remaining frozen.
this section cite: ['b54', 'b28']

Section: MODALITY-FREE ALIGNMENT
The pretraining graphs in G often differ in both input modalities and label systems. Features range from dense vectors to categorical attributes or domain-specific identifiers. Likewise, label spaces are graph-local and vary in cardinality, with no global alignment across domains. These heterogeneities make direct in-context transfer across graphs challenging.
Recent GFMs (Wang et al., 2024b;He et al., 2025a;Liu et al., 2024a) attempt to unify graphs from heterogeneous domains with Text-Attributed Graphs (TAGs), which convert all features and labels to natural language, then map them with language models into shared semantic spaces. However, this approach has fundamental limitations. Real-world graph data is typically already vectorized through domain-specific methods, such as word2vec (Mikolov et al., 2013) for documents, molecular fingerprints (Rogers & Hahn, 2010) for compounds, and user behavior embeddings (Pan & Ding, 2019). Converting these optimized representations to text and back introduces information loss and computational overhead. Furthermore, privacy constraints often restrict access to raw data, and data providers usually release only pre-encoded datasets, making modality-aware conversions infeasible in sensitive domains. Instead, we adopt a modality-free alignment perspective, which aligns graphs directly in their existing representations without modality-aware conversion.
Definition 1. (Modality-free Alignment) Let {(G i , L i )} M i=1
be graphs from M domains, whose item features are already pre-encoded by (unknown) domain-specific pipelines, X i ∈ R do , and whose labels have been indexed by
L i = {0, • • • , C i -1}. A modality-free alignment is a domain- conditioned transformation system T = (K feat i , K label i ) M i=1
with:
K feat i : R do → R d (feature alignment) and K label i : L i → R d (label alignment)(3)
that maps domain-specific features and label IDs directly into a unified d-dimensional feature space and label space, respectively, without reconstructing or converting to any intermediate modality. The transformations should be conditioned on the domain descriptors e i ∈ R de that capture domain characteristics of G i , such that for any two domain i, j,
∥K feat i -K feat j ∥ ∝ ∥e i -e j ∥ and ∥K label i -K label j ∥ ∝ ∥e i -e j ∥.(4)
This ensures that similar domains with close descriptors e i ≈ e j produce similar transformations, causing their aligned features and labels to occupy neighboring subspaces in the unified space.
Fig. 1-left illustrates the idea intuitively. Modality-free alignment maps every graph into a unified semantic space according to domain relationships: graphs from related domains (e.g., two citation networks G 1 and G 2 ) have similar domain descriptors and thus map to neighboring subspaces, whereas unrelated domains (social network G 3 ) sit far away. The domain-conditioned transformations K feat i and K label i project each graph's pre-encoded features and indexed labels into unified feature and label spaces, preserving intra-domain semantics while enabling cross-domain transfer. This is essential because numerically similar feature vectors from different domains can carry entirely different meanings (each domain's encoder defines its own coordinate system), and indexed label IDs [0, 1, 2, • • • ] are reused with domain-specific semantics. Modality-free alignment reconciles these differences by calibrating features and labels via the domain descriptor, unifying them in shared spaces without requiring any knowledge of the original data modality like TAGs.
this section cite: ['b37', 'b46']

Section: MF-GIA: MODALITY-FREE GRAPH IN-CONTEXT ALIGNMENT
In this section, we present MF-GIA for enabling in-context learning across heterogeneous graph domains without modality-specific priors. MF-GIA addresses the fundamental challenge of aligning graphs with incompatible feature spaces and label systems through three key components: (1) domain embedder encodes domain characteristics, (2) domain-conditioned alignment maps pre-encoded features and indexed labels to unified spaces, and (3) episodic pretraining realizes few-shot adaptation during pretraining. We then describe in-context inference on graphs from unseen domains.
this section cite: []

Section: DOMAIN EMBEDDER
one gradient step Reliable domain embeddings are the pivot of MF-GIA: they summarize graphs' domain characteristics, parameterize the domain-conditioned aligners (K feat i , K label i ), and ensure that graphs with related domains are mapped to neighboring subspaces while preserving intra-/cross-domain semantics. Prior work represents graph domains using learnable tokens, but depends on external signals, such as domain labels (Yu et al., 2025) or modality metadata (He et al., 2025a), which are often unavailable in practice. We instead induce the domain embeddings {e i } M i=1 directly from each graph's intrinsic properties by capturing the interactions between the graph and a shared encoder, without external signals. Starting from a single shared weight initialization θ 0 ∈ R do×d for a one-layer GNN encoder followed by a fixed all-ones projection matrix 1 d×Ci from embedding to label space as shown in Fig. 2, we take a single gradient step on each pretraining graph
G i ∈ G as θ i = θ 0 -η∇ θ L i (θ 0 ),
where η is a small learning rate uniformly set to 0.01 and L i is the task loss w.r.t. the available labels on G i . The resulting single-step displacement ∆θ i = θ i -θ 0 serves as a gradient fingerprint that captures how the graph's features, labels, and structure jointly influence the shared encoder. Intuitively, graphs with similar gradient patterns are likely to come from related domains, making ∆θ i a natural descriptor of domain-level information. To obtain compact domain embeddings, we project these fingerprints through a learnable domain embedder f ϕde : R do×d → R de :
e i = f ϕde (∆θ i ) = MLP (Flatten (Conv2D (∆θ i ))) ∈ R de ,(5)
where the fingerprint ∆θ i ∈ R do×d is treated as a single-channel image to be embedded. The embedder f ϕde is trained to preserve domain relationships by minimizing:
L de = Gi,Gj ∈G ∥∆θ i -∆θ j ∥ F -∥e i -e j ∥ 2 2 ,(6)
so that pairwise relationships among graphs are retained in the embedding space. This approach naturally captures domain characteristics without domain labels or modality priors, as the gradient pattern inherently reflects the unique way each domain's data distribution interacts with the shared model initialization.
The domain embedding induced by the gradient fingerprint is central to MF-GIA, and the subsequent alignment operations are established on it. To justify its effectiveness, we provide a theoretical analysis showing that this embedding faithfully preserves domain characteristics (Proof in Appendix B.1).
Theorem 3.1. Let G i and G j be graphs sampled from domains D i and D j respectively, with corresponding gradient fingerprints ∆θ i , ∆θ j ∈ R do×d computed using task loss L i and L j (e.g., cross-entropy). The domain embedder f ϕ de produces domain embeddings e i and e j . Assuming every task loss L is L task -smooth with respect to model parameters, and f ϕ de has Lipschitz constant L de , the domain embeddings preserve domain relationships:
∥e i -e j ∥ 2 ≤ C • W 2 (D i , D j ) (7
)
where W 2 (•, •) measures inherent distance between two domains, and C is a constant.
This upper bound ensures that if two domains are inherently similar, their embeddings learned by f ϕde will be close in the embedding space, while dissimilar domains produce distant embeddings.
In-context Domain Embedding. For a downstream graph G new from an unseen domain, we compute its domain embedding using the same fingerprinting process. Given a few labeled items S = {(w i , y i )} k•Cnew i=1 as a C new -way k-shot prompt from G new , we perform a single gradient step from the same initialization θ 0 , which is a component of the pretraining model M (θ 0 ∈ Φ), as θ new = θ 0 -η∇ θ L new (θ 0 , S), where L new is computed using only the prompt S. The in-context domain embedding of G new is then computed by passing the gradient fingerprint through the pretrained domain embedder:
e new = f ϕde (θ new -θ 0 ) .(8)
This process automatically captures G new 's characteristics and positions it within the learned domain space. Since the domain embedder f ϕde has been trained to preserve domain relationships during pretraining, it naturally maps the new graph's fingerprint to an appropriate location based on the knowledge learned from existing domains.
this section cite: ['b61']

Section: DOMAIN-CONDITIONED ALIGNMENT
With the domain embedding e i for G i ∈ G, MF-GIA instantiates two lightweight transformations (K feat i , K label i ) as aligners that respectively align G i 's item features and graph-local label IDs into unified semantic spaces. Because the transformations are conditioned on e i , related domains with nearby e i induce similar transformations and occupy neighboring subspaces after alignment, while dissimilar domains remain separated. Here, we detail the feature and label alignment mechanisms and their application during in-context inference.
this section cite: []

Section: FEATURE ALIGNMENT
For each pretraining graph G i , we learn a domain-conditioned feature transformation K feat i mapping pre-encoded item features to a unified feature space. Given an item w ∈ G i with its feature x w ∈ R do , we first obtain its base representation via a shared GNN encoder f θ , whose first-layer weight matrix is initialized from the stored θ 0 :
h
i,w = f θ (w, G i ) ∈ R d .(9)
Then we apply Feature-wise Linear Modulation (FiLM) (Perez et al., 2018) to generate domainconditioned transformations from the domain embedding:
γ feat i , β feat i = f ϕfeat (e i ) , γ feat i , β feat i ∈ R d , z i,w = K feat i (h i,w ) = γ feat i ⊙ h i,w + β feat i ,(10)
where f ϕfeat : R de → R 2d is a two-layer MLP that outputs scale γ feat i (with SoftPlus head for positivity) and shift β feat i parameters, ⊙ denotes element-wise product, and z i,w is the aligned feature for w in G i . The FiLM-based transformation K feat i is affine, so it calibrates scales and offset across domains to map features to a domain-specific subspace determined by e i , while preserving the intradomain geometry already present in h i,w . Formally, the alignment satisfies (Proof in Appendix B.2):
Property 1. If f ϕ feat is L -Lipschitz continuous, then ∥γ feat i -γ feat j ∥ 2 +∥β feat i -β feat j ∥ 2 ≤ L ∥e i -e j ∥ 2
for two graph G i and G j , so nearby domains yield similar feature transforms and thus neighboring subspaces in the unified feature space.
this section cite: ['b40']

Section: The domain-conditioned transformations {K feat
i } M i=1 parameterized by shared ϕ feat , are learned jointly over all pre-training graphs G using their domain embeddings {e i } M i=1 to form a unified and general feature space. Together with the GNN encoder f θ , they constitute part of the pretrained model M (i.e., θ, ϕ feat ∈ Φ).
In-context Feature Alignment. At test time, for a downstream graph G new with domain embedding e new computed via the pretrained domain embedder, we generate its alignment parameters (γ feat new , β feat new ) = f ϕfeat (e new ). For any item w ∈ G new , its aligned feature is computed as: z new,w = γ feat new ⊙ f θ (w, G new ) + β feat new . (11) 3.2.2 LABEL ALIGNMENT Domain-agnostic Domain-specific Label Distributions
γ label i , β label i = f ϕlabel (e i ) , γ label i , β label i ∈ R d , u i,l = K label i (E l ) = γ label i ⊙ E label l + β label i , l ∈ L i = {0, • • • , C i -1},(12)
where f ϕlabel : R d → R 2d is a two-layer MLP with a SoftPlus for γ label i . u i,l is the aligned label embeddings conditioned on e i for label l. As illustrated in Fig. 3, this mechanism transforms a single domain-agnostic distribution into domain-specific label distributions. The shared base E label provides a common reference, while FiLM parameters shift and scale these prototypes based on domain characteristics, ensuring semantically distinct labels occupy different subspaces in a unified label space even when sharing the same ID. f ϕlabel is a component of the pretrained model M (ϕ label ∈ Φ).
In-context Label Alignment. For a new graph G new with C new classes, we compute label alignments using the pretrained transformation (γ label new , β label new ) = f ϕlabel (e new ). The aligned label embeddings for G new are:
u new,l = γ label new ⊙ E label l + β label new , l ∈ {0, . . . , C new -1}(13)
which yields domain-aware label prototypes that are compatible with the unified feature space and ready for few-shot matching.
3.3 EPISODIC PRETRAINING MF-GIA is pretrained with an episodic, prompt-based objective that teaches the model to match aligned item features to aligned label prototypes, mimicking the few-shot scenarios encountered during inference.
For each pretraining graph G i ∈ G, we construct m-way k-shot episodes to simulate in-context learning scenarios. Specifically, in each episode, we select m classes and sample k labeled items
per class to form a support set S = m c=1 w (c) j , l (c) k j=1
, where
w (c) j
is the j-th item of the c-th selected class and l (c) = y j is its label ID. We also sample T items per class for the query set
Q = m c=1 q (c) t , l (c) T t=1
. Using the domain embedding e i , we compute aligned item features with Eq. ( 10) and aligned label prototypes with Eq. ( 12), yielding z i,w (c
) j = K feat i h i,w (c) j , z i,q (c) t = K feat i h i,q (c) t
, and
u i,l (c) = K label i E label l (c) .
The prompt-query pairs become:
Prompt S : z i,w (c) j , u i,l (c) c∈[m],j∈[k] , Query Q : z i,q (c) t c∈[m],t∈[T ] .(14)
Recalling the episodic meta learning objective in Eq. ( 2), which requires matching queries to classes using only the prompt, we propose a Dual Prompt-Aware Attention (DPAA) mechanism. It allows queries to attend to prompt examples but prevents prompts from interacting with each other, strictly following the principle of in-context learning. Specifically, let
Z pmt = z i,w (1) 1 , • • • , z i,w (m) k ∈
R mk×d be the matrix of row-stacked support features and U pmt = u i,l (1) , • • • , u i,l (m) ∈ R m×d be the label prototype matrix. DPAA consists of two single-query attention layers, one feature-side and one label-side, both sharing the same projection matrices W K , W V ∈ R d×d . For an aligned query feature z i,q ∈ Q from G i , the feature-side attention computes:
K feat = Z pmt W K , V feat = Z pmt W V , Q feat = z i,q W Q , z out i,q = softmax Q feat (K feat ) ⊤ √ d V feat ,(15)
where the attended representation z out i,q aggregates features from prompt examples relevant to the query. In other words, Eq. ( 15) aims to use the support features Z pmt to prompt the query feature z i,q , producing the prompt-conditioned feature z out i,q for the query. z out i,q is then projected to label space via a learnable function f Ω : R d → R d , which is also prompted by the support set. Thus, the label-side attention lets the query interact with label prototypes:
K label = U pmt W K , V label = U pmt W V , Q label = f Ω (z out i,q ), u out i,q = softmax Q label (K label ) ⊤ √ d V label ,(16)
Analogous to LLMs, where prompt examples guide task completion, here (Z pmt , U pmt ) serves as the few-shot demonstrations, z i,q as the query to be answered, and z out i,q as the prompt-conditioned intermediate, and u out i,q as the answer produced from the prompt for the task objective z i,q . Thus, the pretraining objective is to build the matching between u out i,q and the ground-truth label. The final prediction is obtained by scoring the query's prompted representation against all label prototypes:
s i,q = u out i,q (U pmt ) ⊤ ∈ R m ,(17)
where s i,q contains the per-class scores for the query item q. For each episode from G i , we minimize the cross-entropy loss over all queries in Q:
L episode (G i ) = - 1 mT m c=1 T t=1 log exp s i,q (c) t [c]/τ m j=1 exp s i,q (c) t [j]/τ ,(18)
where τ > 0 is a temperature that controls the sharpness of the softmax, s i,q (c) t
[c] denotes the score of the ground-truth class for the query q (c) t . The complete pretraining loss aggregates episodes across all pretraining graphs:
L pretrain = E Gi∼G E episode∼Gi [L episode (G i )] .(19)
Note that the domain embedder f ϕde is optimized with L de prior to episodic pretraining and then kept fixed. Overall, the pretraining model M Φ comprises the frozen encoder initialization f θ0 , the domain embedder f ϕde , the domain-conditioned transformation f ϕfeat and f ϕlabel , the DPAA projection matrices and the projection head f Ω . This episodic regime trains the model to leverage prompt examples for prediction, establishing the feature-label matching capability essential for ICL on unseen domains.
this section cite: []

Section: IN-CONTEXT PREDICTION ON UNSEEN DOMAINS
At test time, MF-GIA freezes all pretrained parameters. Given an unseen graph G new together with a
C new -way k-shot support set S = {(w i , y i )} k•Cnew i=1
as prompts, we compute the in-context domain embedding e new using the gradient fingerprint and the pretrained domain embedder described in Section 3.1. With the pretrained domain-conditioned transformations, any item w ∈ G new is mapped to its aligned feature z new,w via Eq. ( 11). For label IDs l ∈ L new = {0, • • • , C new -1}, the aligned label prototypes are {u new,l } l∈Lnew obtained by Eq. ( 13). We then form the prompt matrices
Z pmt new = z new,w (1) 1 , . . . , z new,w (Cnew) k ∈ R (kCnew)×d and U pmt new = u new,l (1) , . . . , u new,l (Cnew) ∈ R Cnew×d .
To make a prediction on a query item q, we apply the pretrained DPAA on it. Specifically, the featureside attention produces the prompt-conditioned feature z out new,q by Eq. ( 15), which is then fed to the label-side attention Eq. ( 16) to yield u out new,q . The final scores and prediction are:
s new,q = u out new,q U pmt new ⊤ ∈ R Cnew , ŷq = arg max j∈[Cnew] s new,q [j].(20)
This inference procedure is parameter-update-free w.r.t. the pretrained model M, and the same pipeline applies to node or edge items by letting w range over V new or E new . The detailed algorithms and complexity analysis of MF-GIA are provided in Appendix C.
To fully exploit the sparse few-shot support labels and the topology of G new during inference, we further enhance prototype construction and prediction refinement beyond DPAA. Specifically, we construct graph-aware class prototypes by propagating soft label distributions initialized from the support set through the graph structure using label propagation, which enriches the prototypes with neighborhood context to compensate for the sparsity of the k-shot support. For query classification, we employ an adaptive distance metric that combines cosine similarity and inverse Euclidean distance, weighted per query by σ(Var(z new,q )), enabling the metric to adapt to local feature geometry. Finally, predictions are iteratively refined through semi-supervised label propagation, where query pseudolabels are updated by blending propagated and current distributions while keeping support labels fixed, enforcing graph-level consistency in the final output.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETUP
We employ cross-domain graph datasets to evaluate MF-GIA, which is pretrained exclusively on node classification tasks using four datasets: WikiCS (web link), PubMed and ogbn-Arxiv (citation), and Amazon-ratings (e-commerce rating). These datasets are pre-encoded with heterogeneous feature spaces and label systems, enabling us to learn domain alignment without modality priors.  For downstream evaluation, we test on both node-level and edge-level classification tasks across seen and unseen domains. For node classification, we evaluate on Cora (citation), and unseen domains including ogbn-Products and Computers (e-commerce product), Physics (co-authorship), and BlogCatalog (social media). For link classification, we assess performance on two knowledge graphs (KGs) from different domains to predict the relation types: FB15K237 (encyclopedic) and WN18RR (lexical), which represent entirely new tasks not encountered in pretraining. To unify the task formulation, we transform the edge-level task into the node-level task by converting edges to nodes in a line graph, as detailed in Appendix D. This task allows us to evaluate our model's generalization capability on an entirely new task and domains not seen during pretraining. More information about baseline configurations, datasets, and implementation details is provided in Appendix E.
this section cite: []

Section: IN-CONTEXT LEARNING RESULTS
Table 2 demonstrates that MF-GIA achieves state-of-the-art node classification results across diverse graph domains. Remarkably, MF-GIA reaches 63.98% on Cora with 5-shot prompting, which is an 11.34% absolute improvement over the second-best baseline. Across all 15 configurations, MF-GIA consistently outperforms existing methods with an average margin of 4.2%, despite using pure promptbased inference without any parameter updates. In contrast, methods like GFT and AutoGFM require extensive fine-tuning on target domains yet still achieve inferior results. This superiority reveals a fundamental insight: when equipped with proper cross-domain alignment, ICL beats fine-tuning. The critical role of alignment is also evident when comparing MF-GIA with Prodigy, which is a true ICL model without domain alignment. MF-GIA consistently outperforms Prodigy on unseen domains, demonstrating that domain embeddings capture domain characteristics for successful cross-domain transfer. Recent modality-dependent GFMs fail on graphs without raw text data (marked "-"), while MF-GIA operates universally on any pre-encoded graphs. Moreover, as shown in Table 3, MF-GIA excels at edge-level tasks, an entirely new task formulation never encountered during pretraining. It demonstrates that MF-GIA captures generalizable patterns for in-context reasoning rather than memorizing dataset/task-specific features. On WN18RR dataset with a 10-way setting, our MF-GIA does not surpass the state-of-the-art baselines GFT and AutoGFM, achieving third-best performance across all shot settings. It is because MF-GIA is pretrained exclusively on node classification tasks, while WN18RR is a dataset for edge-level tasks, which is an entirely different task formulation never encountered during pretraining. We deliberately evaluate on this dataset to assess our model's generalization capacity to unseen tasks, as we believe a genuine graph foundation model should generalize not only to unseen domains but also to unseen task types. While GFT and AutoGFM achieve superior performance on WN18RR-10way, they are pretrained on both node-level and  edge-level tasks. Therefore, edge classification is not an unseen task for these baselines, so their performance advantage does not necessarily demonstrate stronger cross-task generalization.
this section cite: []

Section: MODEL ANALYSIS

this section cite: []

Section: Effect of Core Components.
We analyze the contribution of each component in MF-GIA, starting from its GraphSAGE backbone. GraphSAGE+FT is pretrained on the same datasets and fine-tuned on support sets of test graphs. Adding a domain embedder with FiLM-based feature alignment (+Feat. Align.) improves cross-domain adaptability. Extending alignment to the label space (++Label Align.) further boosts performance by unifying class indices across graphs. Finally, incorporating DPAA with an episodic objective yields the full MF-GIA, which achieves the largest gains across datasets and shots. Table 4 shows a clear step-wise improvement, underscoring that both domain-conditioned alignment and prompt-aware reasoning are crucial for effective graph ICL. Effect of Episodic Inference. ICL can be achieved through two paradigms: episodic meta-learning, which unifies pretraining and inference by training the model to perform inference episodes (MF-GIA and Prodigy), and supervised pretraining with test-time prototype construction, where class prototypes are built from support sets and queries are classified by proximity (GraphAlign). As shown in Table 5, episodic inference (MF-GIA) consistently outperforms the supervised variant (MF-GIA sup ).
Effect of Backbone GNNs. In MF-GIA, we adopt GraphSAGE as the default backbone. Fig. 4 shows MF-GIA exhibits minor accuracy fluctuations across different GNN backbones under 1-shot settings, demonstrating that MF-GIA is robust to backbone selections. More analytical results are provided in Appendix F.
this section cite: []

Section: CONCLUSION
We introduced MF-GIA, a pretraining framework for graph neural networks that enables in-context learning across heterogeneous domains without relying on modality assumptions. By capturing domain characteristics via gradient fingerprints and aligning pre-encoded features and graph-local labels through domain-conditioned transformations, MF-GIA supports parameter-update-free adaptation from few-shot prompts. This design overcomes key limitations of existing GFMs by removing the need for post-training fine-tuning and modality-specific conversions. Experiments demonstrate strong performance on both seen and unseen domains, with seamless transfer to new tasks.
Future Directions. Beyond our current design, MF-GIA opens up several promising avenues for future work. One direction is to couple our gradient fingerprints with LLMs to generate semantic domain descriptions, enabling human-interpretable summaries of latent domain characteristics and more transparent cross-domain reasoning. Another is to leverage these fingerprints to automatically discover latent domain structure from large unlabeled graph collections, moving from manually curated domains to data-driven domain decomposition. We believe these extensions will further enhance the interpretability, automation, and scalability of modality-free graph foundation models.
this section cite: []

Section: References
Ref_id:b0 Title: Translating embeddings for modeling multi-relational data Year: (2013)
Ref_id:b1 Title: Language models are few-shot learners Year: (2020)
Ref_id:b2 Title: AutoGFM: Automated graph foundation model with adaptive architecture customization Year: (2025)
Ref_id:b3 Title: Exploring the potential of large language models (llms) in learning on graphs Year: (2024)
Ref_id:b4 Title: Text-space graph foundation models: Comprehensive benchmarks and new insights Year: (2024)
Ref_id:b5 Title: A prompt-based knowledge graph foundation model for universal in-context reasoning Year: (2024)
Ref_id:b6 Title: Convolutional 2d knowledge graph embeddings Year: (2018)
Ref_id:b7 Title: Universal prompt tuning for graph neural networks Year: (2023)
Ref_id:b8 Title: Uniglm: Training one unified language model for text-attributed graphs embedding Year: (2025)
Ref_id:b9 Title: Graph world model Year: (2025)
Ref_id:b10 Title: Model-agnostic meta-learning for fast adaptation of deep networks Year: (2017)
Ref_id:b11 Title: Towards foundation models for knowledge graph reasoning Year: (2024)
Ref_id:b12 Title: What can transformers learn in-context? a case study of simple function classes Year: (2022)
Ref_id:b13 Title: What makes a good order of examples in in-context learning Year: (2024)
Ref_id:b14 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b15 Title: Unigraph: Learning a unified cross-domain foundation model for text-attributed graphs Year: (2025)
Ref_id:b16 Title: Unigraph2: Learning a unified embedding space to bind multimodal graphs Year: (2025)
Ref_id:b17 Title: Graphmae: Self-supervised masked graph autoencoders Year: (2022)
Ref_id:b18 Title: Graphalign: Pretraining one graph neural network on multiple graphs via feature alignment Year: (2024)
Ref_id:b19 Title: Open graph benchmark: Datasets for machine learning on graphs Year: (2020)
Ref_id:b20 Title: OGB-LSC: A large-scale challenge for machine learning on graphs Year: (2021)
Ref_id:b21 Title: Prodigy: Enabling in-context learning over graphs Year: (2023)
Ref_id:b22 Title: How expressive are knowledge graph foundation models? Year: (2025)
Ref_id:b23 Title: Hgmp: Heterogeneous graph multi-task prompt learning Year: (2025)
Ref_id:b24 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b25 Title: GOFA: A generative one-for-all model for joint graph language modeling Year: (2025)
Ref_id:b26 Title: Snap: A general-purpose network analysis and graph-mining library Year: (2016)
Ref_id:b27 Title: The power of scale for parameter-efficient prompt tuning Year: (2021)
Ref_id:b28 Title: Episodic training for domain generalization Year: (2019)
Ref_id:b29 Title: One for all: Towards training one graph model for all classification tasks Year: (2024)
Ref_id:b30 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b31 Title: Arc: A generalist graph anomaly detector with in-context learning Year: (2024)
Ref_id:b32 Title: Graphprompt: Unifying pre-training and downstream tasks for graph neural networks Year: (2023)
Ref_id:b33 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b34 Title: Frozen pretrained transformers as universal computation engines Year: (2022)
Ref_id:b35 Title: In-context learning with retrieved demonstrations for language models: A survey Year: (2024)
Ref_id:b36 Title: Wiki-cs: A wikipedia-based benchmark for graph neural networks Year: (2020)
Ref_id:b37 Title: Efficient estimation of word representations in vector space Year: (2013)
Ref_id:b38 Title: MetaICL: Learning to learn in context Year: (2022)
Ref_id:b39 Title: Shimei Pan and Tao Ding. Social media-based user embedding: A literature review Year: (2019)
Ref_id:b40 Title: Film: Visual reasoning with a general conditioning layer Year: (2018)
Ref_id:b41 Title: A critical look at the evaluation of GNNs under heterophily: Are we really making progress? Year: (2023)
Ref_id:b42 Title: Anirudh Dagar, and Wenming Ye. In-context learning with iterative demonstration selection Year: (2024)
Ref_id:b43 Title: Gcc: Graph contrastive coding for graph neural network pre-training Year: (2020)
Ref_id:b44 Title: Sentence-bert: Sentence embeddings using siamese bert-networks Year: (2019)
Ref_id:b45 Title: Towards understanding how transformers learn in-context through a representation learning lens Year: (2024)
Ref_id:b46 Title: Extended-connectivity fingerprints Year: (2010)
Ref_id:b47 Title: Learning to retrieve prompts for in-context learning Year: (2022)
Ref_id:b48 Title: Pitfalls of graph neural network evaluation Year: (2018)
Ref_id:b49 Title: Prototypical networks for few-shot learning Year: (2017)
Ref_id:b50 Title: All in one: Multi-task prompting for graph neural networks Year: (2023)
Ref_id:b51 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b52 Title: Graph Attention Networks. International Conference on Learning Representations Year: (2018)
Ref_id:b53 Title: Deep Graph Infomax Year: (2019)
Ref_id:b54 Title: Matching networks for one shot learning Year: (2016)
Ref_id:b55 Title: Transformers learn in-context by gradient descent Year: (2023)
Ref_id:b56 Title: Towards graph foundation models: Training on knowledge graphs enables transferability to general graphs Year: (2024)
Ref_id:b57 Title: Gft: Graph foundation model with transferable tree vocabulary Year: (2024)
Ref_id:b58 Title: Pane: scalable and effective attributed network embedding Year: (2023)
Ref_id:b59 Title: Revisiting semi-supervised learning with graph embeddings Year: (2016)
Ref_id:b60 Title: Graph contrastive learning with augmentations Year: (2020)
Ref_id:b61 Title: Samgpt: Text-free graph foundation model for multi-domain pre-training and cross-domain adaptation Year: (2025)
Ref_id:b62 Title: How much can transfer? BRIDGE: Bounded multi-domain graph foundation model with generalization guarantees Year: (2025)
Ref_id:b63 Title: Trained transformers learn linear models in-context Year: (2024)
Ref_id:b64 Title: Active example selection for in-context learning Year: (2022)
Ref_id:b65 Title: Automatic chain of thought prompting in large language models Year: (2023)
Ref_id:b66 Title: All in one and one for all: A simple yet effective method towards cross-domain graph pretraining Year: (2024)
Ref_id:b67 Title: Calibrate before use: Improving few-shot performance of language models Year: (2021)
Ref_id:b68 Title: Graphclip: Enhancing transferability in graph foundation models for text-attributed graphs Year: (2025)
