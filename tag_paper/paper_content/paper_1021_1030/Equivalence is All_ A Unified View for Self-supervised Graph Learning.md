Title: Equivalence is All: A Unified View for Self-supervised Graph Learning
Abstract: Node equivalence is common in graphs, such as computing networks, encompassing automorphic equivalence (preserving adjacency under node permutations) and attribute equivalence (nodes with identical attributes). Despite their importance for learning node representations, these equivalences are largely ignored by existing graph models. To bridge this gap, we propose a GrAph selfsupervised Learning framework with Equivalence (GALE) and analyze its connections to existing techniques. Specifically, we: 1) unify automorphic and attribute equivalence into a single equivalence class; 2) enforce the equivalence principle to make representations within the same class more similar while separating those across classes; 3) introduce approximate equivalence classes with linear time complexity to address the NP-hardness of exact automorphism detection and handle nodefeature variation; 4) analyze existing graph encoders, noting limitations in message passing neural networks and graph transformers regarding equivalence constraints; 5) show that graph contrastive learning are a degenerate form of equivalence constraint; and 6) demonstrate that GALE achieves superior performance over baselines.

Section: Introduction
Node equivalence (Lorrain & White, 1971;Bafai, 1977) is common in real-world graphs such as computing networks, where resources form nodes, and their dependencies form 1. An example of automorphic equivalence. Hollow double arrows indicate node permutations in graph G, shown in cycle notation (e.g., (1, 2, 3) means 2→1, 3→2, 1→3).
edges. In terms of graph structure, automorphic equivalence represents a strict form of equivalence: there exists a global permutation that maps nodes within the same equivalence class to each other (Cameron & Mary, 2004). As shown in Figure 1, we perform permutations on the nodes of the original graph G. Here, G (i,j,...,k) represents a permuted graph under the cycle notation of permutation, For instance, G (1,2,3) means nodes 1, 2, and 3 are cyclically permuted (node 2 is mapped to node 1, node 3 to node 2, and node 1 to node 3). We can observe that G = G (6,7) = G (1,2,3) , indicating that these nodes are structurally equivalent with respect to the graph topology. In contrast, nodes 4 and 5 are not equivalent, as swapping them would not preserve their adjacency relations.
Automorphic equivalence (AE) is a fundamental concept in various scientific domains such as chemistry and network analysis (Faulon, 1998;Friedkin & Johnsen, 1997). For instance, AE is a key indicator of similarity in social status and behavior in social networks (Everett, 1985). Furthermore, for graphs with node attributes, another type of equivalence can be defined based on attribute similarity. Specifically, a new equivalence class can be constructed by measuring the distance between attribute vectors: nodes with a close distance are grouped into the same equivalence class.
Naturally, we would expect graph representation models to preserve and reflect these node equivalences in their encoded node representations. However, we find that few studies ex-plicitly consider these equivalences in graph representation learning, particularly in the widely studied paradigm of self-supervised graph learning, such as graph contrastive learning (Thakoor et al., 2022;Liu et al., 2024a;Wan et al., 2024). This method typically augments the original graph (e.g., by randomly removing edges) to generate two contrasting views. Contrastive constraints are then applied to encourage the representations of the same node (positive views) in both augmented graphs to be as similar as possible, while ensuring that representations of different nodes (negative views) become dissimilar. However, this paradigm overlooks the equivalence relations in terms of graph structure and node attributes. Existing methods focus solely on aligning the representations of the same node across augmented views, pushing apart those of different nodes, while often neglecting the fact that nodes similar to a given node are not limited to that node alone. Consequently, certain similarities between different nodes are frequently ignored.
To address these issues, we propose a novel equivalencebased self-supervised graph representation learning framework (GALE) grounded in the equivalence principle: node representations within the same equivalence class should be more similar, while those in different equivalence classes should be as dissimilar as possible. Specifically, 1) we define two equivalence relations-automorphic equivalence based on graph structure and attribute equivalence based on node attributes-and combine them into a unified equivalence class; 2) Then we enforce the equivalence principle, ensuring the node representations within the same equivalence class are similar, while those in different classes are dissimilar; 3) We introduce approximate equivalence classes with linear time complexity to address the NP-hard nature of exact automorphism detection and handle practical challenges such as attribute noise or slight variations that still indicate similarity, which make strict equivalence impractical in real-world graphs; 4) We note that message passing neural networks (MPNNs) make equivalent nodes similar but risk over-similarity for non-equivalent nodes, while most position encoding methods in graph transformers fail to incorporate the automorphic equivalence constraint; 5) From the equivalence perspective, we show that the current graph contrastive learning paradigm is a degenerate form of equivalence constraint, where each equivalence class is limited to a single node; 6) We demonstrate that GALE surpasses SOTA algorithms through experiments on benchmark datasets.
this section cite: ['b28', 'b1', 'b5', 'b9', 'b11', 'b8', 'b45', 'b48']

Section: Related Work
Equivalence in Graphs. The concept of node equivalence traces its roots to structural sociology and algebraic graph theory (Wasserman & Faust, 1994). In social network analysis, Lorrain and White's structural equivalence (Lorrain & White, 1971;Fortunato, 2010) formalized the idea that nodes sharing identical connectivity patterns occupy equivalent roles, forming the basis for role discovery in networks. However, its strict definition limits its applicability in complex networks. A broader concept of equivalence, automorphic equivalence (AE) (Lauri & Scapellato, 2016), emerged from graph automorphism theory, where two nodes are equivalent if there exists a permutation of the graph's vertices such that adjacency relations are preserved (Bafai, 1977). AE has proven fundamental across scientific domains, e.g., in social networks, AE identifies individuals with identical social statuses or influence patterns (Everett, 1985;Friedkin & Johnsen, 1997). In chemistry, symmetric atoms in molecules (e.g., carbon atoms in benzene rings) are automorphically equivalent, explaining their identical chemical properties (Faulon, 1998).
this section cite: ['b53', 'b28', 'b10', 'b23', 'b1', 'b8', 'b11', 'b9']

Section: Self-supervised Graph Learning.
Graph learning has attracted significant attention in recent years (Koh et al., 2024;Wang et al., 2024a;b;Liu et al., 2024b;Wang et al., 2024c). Graph self-supervised learning has emerged as a dominant paradigm for learning representations from unlabeled graph data (Wang et al., 2023;Zhao et al., 2025). Early approaches focused on preserving structural proximities through matrix factorization (Belkin & Niyogi, 2001), random walk objectives (Grover & Leskovec, 2016), or deep autoencoders (Cao et al., 2016). The recent surge in graph contrastive learning (GCL) methods (Cai et al., 2023;Li et al., 2022;Thakoor et al., 2022) has centered around maximizing agreement between positive views of the same node while repelling negative samples. Contrastive loss encourages similar representations for the same node across augmented views, while separating different nodes. However, it treats each node as only equivalent to itself, overlooking structural or attribute-based equivalence.
this section cite: ['b18', 'b52', 'b64', 'b2', 'b13', 'b6', 'b4', 'b24', 'b45']

Section: Background
Notations. Let G = (V, E) be a graph with n nodes and m edges, where V denotes the set of nodes and E represents the set of edges. If node attributes are available, we denote the node feature matrix as X ∈ R n×d , where d represents the dimension of the node attributes.
Problem Formulation. In this work, we focus on learning the representation z v ∈ R q for each node v in an unsupervised manner, where q denotes the output dimension.
Equivalence Relation and Equivalence Class. Let S be a non-empty set (in this paper, the set of interest is the node set V ), and let R be a relation on S. We say that R is an equivalence relation on S if and only if R satisfies the following properties: 1) Reflexivity: Every element is related to itself under R (i.e., x R x for all x ∈ S). 2) Symmetry: If x ∈ S is related to y ∈ S under R, then y is related to x under R (i.e., x R y ⇒ y R x). 3) Transitivity: If x ∈ S is related to y ∈ S and y is related to z ∈ S under R, then x is related to z under R (i.e., x R y ∧ y R z ⇒ x R z). For example, the identity relation (=) on a set S is an equivalence relation. For each x ∈ S, the equivalence class of x determined by R, denoted as [x], is defined as [x] = {y ∈ S : x R y}.
Partition and Induced Partition. Let P be a family of subsets of S. We say that P is a partition of S if and only if the following conditions are satisfied: 1) If X ∈ P, then X = ∅ (i.e., no subset in the partition is empty); 2) If X ∈ P and Y ∈ P, then either X = Y or X ∩ Y = ∅ (i.e., subsets in the partition are pairwise disjoint); 3) X∈P X = S (i.e., the union of all subsets in the partition covers the entire set S). It can be proven that the equivalence classes determined by any equivalence relation on a set form a partition of that set. This partition is referred to as the induced partition from the equivalence relation.
this section cite: []

Section: Automorphism and Orbit.
A permutation π on a set V rearranges its elements as V π = {v π }. We use cycle notation to represent permutations. All permutations of V form a symmetric group S n . A graph automorphism of a graph G is a permutation π ∈ S n that preserves the edge relations of G, i.e., E = E π {(u π , v π ) | (u, v) ∈ E}. Every graph has a trivial automorphism, the so-called identity automorphism, which maps each vertex to itself. The set of all automorphisms of a graph forms a group, called the automorphism group of the graph, denoted as Aut(G). The elements (permutations) in the automorphism group of a graph define an equivalence relation on the node set V :
u auto v ⇐⇒ ∃π ∈ Aut(G), u π = v (1
)
This equivalence relation, also known as a symmetry relation, partitions the node set V into non-overlapping equivalence classes called orbits.
this section cite: []

Section: Learning with Equivalence Classes
In this Section, we introduce a novel equivalence-based selfsupervised graph learning framework (GALE), as shown in Figure 2. GALE consists of a dual-branch structure. One branch involves a graph encoder, which encodes the graph to obtain node representations; we use a graph neural network (GNN) for this purpose. The other branch focuses on inducing equivalence-class partitions of the node set. Based on the equivalence principle-nodes within the same equivalence class are considered similar, while nodes in different classes are considered dissimilar-we impose regularization constraints on the node representations and optimize the encoder with an equivalence loss. In our work, we construct two types of equivalence-class partitions: 1) Group I: Calculated based on the graph's adjacency information, capturing the graph's structural automorphism. 2) Group II: Calculated based on the attribute information of nodes, capturing feature equality. We fuse these partitions into a unified node constraint network, ensuring the learned representations capture both structural and attribute-based equivalences.
this section cite: []

Section: Automorphic Equivalence from Structure
A graph typically encompasses two components: the relationships between nodes (edges) and the properties associated with each node. We establish equivalence relations and their corresponding partitions on both of these aspects. First, we focus on adjacency. As discussed earlier, automorphic equivalence represents a strict node equivalence relation based on the graph's structure. Indeed, real-world graph data often exhibits a substantial number of nodes with automorphic equivalence.
To verify this, we present statistics on the proportion of nodes/graphs with automorphic equivalence in 3 benchmark network datasets (Kipf & Welling, 2017) and 3 graph datasets (Morris et al., 2020). For network data (single graph), we calculate the ratio of nodes in non-singleton orbits (the number of nodes is greater than one) to the total number of nodes. For graph datasets (multiple graphs), we compute the ratio of graphs with non-trivial automorphisms. The results, shown in Table 1, demonstrate that a significant proportion of nodes/graphs exhibit non-trivial automorphic equivalence, highlighting its prevalence in realworld datasets. To compute the exact automorphisms of a graph, we adopt classical algorithms such as bliss or nauty (Junttila & Kaski, 2011;McKay & Piperno, 2014). For example, bliss is an efficient method for computing the automorphism group of real-world graphs by leveraging symmetry-breaking and backtracking search. Once the automorphism orbits are obtained, we construct the automorphic equivalence partition P auto = {C i }, where C i represents an orbit.
this section cite: ['b17', 'b33', 'b15', 'b29']

Section: Attribute Equivalence from Nodes
To construct a partition of the graph based on node attributes, we define an equivalence relation among nodes such that nodes with identical attributes are grouped into the same equivalence class:
u attr v ⇐⇒ x u = x v , ∀u, v ∈ V(2)
which ensures that the partition reflects the inherent equation of nodes in terms of their attributes. This equivalence relation induces a partition P attr = {D i }, where each subset D i ⊆ V contains all nodes with the same attribute values.
this section cite: []

Section: Fusion of Equivalences
With the two partitions established, we aim to create a refined node partition by fusion. Generally, both the structural role and the attributes of a node are equally important for its characterization. For example, consider a citation network (where nodes represent papers and edges represent citations): two papers might be cited by similar papers (graph structure), yet their topics (attributes) could be different. Therefore, we seek a refined partition that preserves the equivalence relationships from both perspectives. To accomplish this, we construct the fused partition by intersecting the structure-based and attribute-based partition:
P fuse = {C i ∩ D j | C i ∈ P auto , D j ∈ P attr , C i ∩ D j = ∅}
(3) It inherits characteristics from both partitions, ensuring consistency with both graph structure and node attributes. Each fused equivalence class F = C i ∩ D j contains nodes that are equivalent under both graph automorphisms and node attribute equivalence. The resulting partition satisfies: 1) F ∈Pfuse F = V ; 2) F k ∩ F l = ∅ for k = l. We now prove that the relation fuse , which induces the fused partition P fusion , satisfies the properties of an equivalence relation. Theorem 4.1. The fusion relation defined as u f use v is an equivalence relation.
Proof. To show that fuse is an equivalence relation, we verify the three properties: 1) Reflexivity: For any node u ∈ V , we have u auto u (since automorphic equivalence is reflexive) and u attr u (since attribute equivalence is reflexive), Therefore, u fuse u, satisfying reflexivity. 2) Symmetry: If u fuse v, then u auto v and u attr v. By the symmetry of auto and attr , we have v auto u and v attr u, proving symmetry. 3) Transitivity: If u fuse v and v fuse w, then u auto v, v auto w, and u attr v, v attr w. By the transitivity of auto and attr , we have u auto w and u attr w. Hence, transitivity is established.
this section cite: []

Section: Equivalence Loss
To encourage nodes within the same fused equivalence class to have similar representations and nodes from different equivalence classes to have dissimilar representations, we define a loss function based on the similarity between node embeddings.
Intra-class Loss. Let P fuse = {F i } represent the fused equivalence partition, and let z u denote the embedding of node u. The intra-class loss is defined to encourage the embeddings of nodes within the same equivalence class F i to be similar:
L intra = i u,v∈Fi -D(z u , z v )(4)
where D : R q × R q → R is a discriminator that maps two views to an agreement score. In this work, we use the standard inner product D (z i , z j ) = z T i z j . Inter-class Loss. The inter-class loss is designed to ensure that embeddings of nodes from different equivalence classes are dissimilar. For nodes s ∈ F i and t ∈ F j with i = j, we penalize high similarity between their embeddings. The inter-class loss is defined as:
L inter = i =j s∈Fi,t∈Fj D(z s , z t )(5)
Finally, the equivalence loss is defined as:
L equiv = L intra + L inter (6
)
this section cite: []

Section: Approximate Equivalence Classes
In real-world graphs, strict equivalence relationships, such as automorphic or exact attribute-based equivalence, are often impractical due to the computational cost of determining exact automorphisms and the rarity of complete attribute equality caused by noise or data variations. To overcome the challenges, we introduce approximate equivalence classes, which relax the strict requirements of exact equivalence.
this section cite: []

Section: Approximate Automorphic Equivalence
Determining the exact automorphism group of a graph is computationally intractable, as it is the NP-hard problem. To overcome this, we propose an efficient approximation using PageRank vectors, which leverage their inherent relationship to approximate automorphic equivalence effectively.
The PageRank algorithm (Langville & Meyer, 2004) assigns a score to each node in a graph, measuring its relative importance based on the graph structure. Let P ∈ R n×n denote the Markov transition matrix of the graph, where
P ij = 1/deg(i) if (i, j
) ∈ E, and P ij = 0 otherwise, with deg(i) representing the degree of node i. The PageRank vector r ∈ R n is the stationary distribution of a random walk with teleportation, based on the transition probabilities P, computed as:
r = αP r + (1 -α)v(7)
where α ∈ (0, 1) is the teleportation probability, v ∈ R n is a personalization vector (typically uniform). The PageRank vector assigns a score r i to each node i ∈ V , reflecting its importance based on local connectivity and global graph structure. The following lemma establishes the relationship between automorphisms and PageRank.
Lemma 5.1 (Ghorbani et al. (2021)). If two vertices u, v ∈ V are automorphically equivalent in G, then u and v have the same PageRank score:
u auto v =⇒ r u = r v(8)
We leverage this property to approximate automorphic equivalence by grouping nodes with similar PageRank scores. While nodes with equal PageRank scores are not necessarily automorphically equivalent, they often exhibit structural similarity in terms of their importance within the graph, providing an efficient approximation. To verify this, we conduct experiments on 8 benchmark datasets: Cora, Citeseer, Pubmed (Kipf & Welling, 2017), Wiki-CS, Amazon-Computers, Amazon-Photo, Coauthor-CS, and Coauthor-Physics (Shchur et al., 2018). Empirical results show the alignment between the true and approximate equivalence partitions, measured by the Variation of Information (Meilȃ, 2007) (with values closer to 0 indicating better alignment and 0 indicating perfect alignment). The detailed results are presented in Table 2, where we find that for all the data, the approximate partitions closely match the true ones, validating the effectiveness of our approximation. Here, we use α = 0.85; for an analysis of different parameter settings, please refer to the Appendix. Computing PageRank has a time complexity of O(m) for graphs, where m denotes the number of edges, which is significantly more efficient than exact automorphism computation (Kondaveeti et al., 2024).
this section cite: ['b22', 'b12', 'b17', 'b40', 'b30', 'b19']

Section: Approximate Attribute Equivalence
In real-world networks, nodes often exhibit attributes that are not perfectly identical but are instead only slightly different. Strictly enforcing exact equality of node attributes when defining equivalence classes can lead to overly fragmented partitions, as even minor differences in attributes result in nodes being classified into different equivalence classes.
To address this issue, we propose relaxing the equivalence constraint by using a similarity threshold based on the Euclidean distance between node attribute vectors. This allows nodes with sufficiently similar attributes to be grouped into the same equivalence class. For nodes u, v ∈ V , we define the approximate attribute equivalence relation attr as:
u attr v ⇐⇒ x u -x v 2 ≤ (9
)
where ≥ 0 is a small positive threshold that controls the degree of similarity required for nodes to be considered equivalent. The relaxed partition allows nodes with similar attributes (within the threshold ) to be grouped together, even if their attributes are not exactly identical. Clearly, when = 0, it reduces to strict equivalence. When > 0, it is not a strict equivalence, though the resulting node sets may be called similarity groups.
this section cite: []

Section: Complexity Analysis
The time complexity of the GALE involves four components: 1) Automorphic equivalence can be approximated using PageRank, which runs in O(m); 2) Attribute-based equivalence requires grouping nodes by attributes, with a complexity of O(n 2 ); 3) Fusion of equivalences involves intersecting equivalence classes, scaling as O(n) in practice due to the typically small number and size of partitions. 4) Equivalence loss computation with a worst-case complexity of O(n 2 ). Overall, the dominant term is O(n 2 + m).
this section cite: []

Section: Equivalence and Graph Encoders
In this section, we discuss how mainstream graph encoders, message passing neural networks (MPNNs) and graph transformers, interact with the concept of equivalence.
this section cite: []

Section: MPNNs and Automorphic Equivalence
MPNNs (Wu et al., 2020) and their variants, such as Graph Convolutional Networks (GCNs), rely on the principle of neighborhood aggregation to learn node representations. The neighborhood aggregation mechanism of MPNNs inherently aligns with the goal of making automorphically equivalent nodes have similar representations. For example, consider two automorphically equivalent nodes u and v and their adjacency structures are identical. During the messagepassing process, u and v will aggregate similar information from their neighbors. This property ensures that MPNNs can naturally preserve the similarity of node representations within the same equivalence class.
However, MPNNs cannot guarantee that dissimilarity between different equivalence classes remains valid. As the number of layers increases, the representations of all nodes in the graph tend to converge, leading to the over-smoothing problem. This phenomenon reduces the discriminative power of the learned node embeddings. Common oversmoothing mitigations, like residual connections or dynamic neighborhoods, typically overlook node equivalence.
To address this, explicit constraints based on equivalence classes can be incorporated into MPNN models. These constraints enforce inter-class dissimilarity while preserving intra-class similarity, helping to alleviate the over-smoothing problem. To demonstrate this, Figure 3 shows the classification results of GCN and GALE with increasing depth on the Cora and CiteSeer dataset. GCN's performance declines sharply with more layers, but GALE with equivalence constraints significantly mitigates the over-smoothing issue.
this section cite: ['b55']

Section: Graph Transformers and Equivalence
Graph transformers (Min et al., 2022) take a fundamentally different approach to learning node representations compared to MPNNs. Instead of relying on neighborhood aggregation based on adjacency structure, graph transformers use node positional embeddings to inject structural information and infer relationships between nodes through their positions and attributes. The node representations are then updated using a fully-connected self-attention mechanism.
While graph transformers can capture global relationships, they do not inherently ensure that automorphically equivalent nodes are represented similarly. The effectiveness of capturing automorphic equivalence depends on the quality of the positional embeddings, which may not always align with the graph's symmetries. Moreover, the global nature of the attention mechanism, which lacks local constraints, means that nodes within the same equivalence class might not remain sufficiently similar, and there's no inherent mechanism to ensure dissimilarity between nodes in different equivalence classes. As a result, while graph transformers excel at encoding global structures, they may struggle to consistently respect automorphic equivalence without additional constraints or interventions.
Positional Encoding and AE. Therefore, node positional embeddings, which are solely based on the graph's structure, are essential for reflecting the equivalence relationships within that structure. To evaluate whether common positional encoding (PE) methods align with automorphic equivalence, we propose a validation procedure that compares the ground truth automorphic equivalence with those derived from positional embeddings, using a distance threshold pe = 10 -4 to construct equivalence classes. Specifically, we conduct experiments on 8 benchmark graph datasets using 5 PE methods: Laplace positional encoding (LapPE) (Kreuzer et al., 2021), Random walk positional encoding (RWSE) (Dwivedi et al., 2022), SignNet (Lim et al., 2023), ElstaticSE, and HKdiagSE (Rampášek et al., 2022). We compute the Variation of Information (VI) (Meilȃ, 2007), and the results are shown in Table 3. Our findings indicate that most existing PE methods do not truly adhere to automorphic equivalence, and in some cases, deviate significantly. This may be because these methods focus primarily on ensuring that neighboring nodes are assigned similar positions (locally), while neglecting the global equivalences within the graph.
this section cite: ['b31', 'b21', 'b7', 'b25', 'b38', 'b30']

Section: Equivalence in Graph Contrastive
In this section, we analyze how equivalence classes play a role in graph contrastive learning (GCL).
this section cite: []

Section: Analysis of Graph Contrastive Learning
GCL starts by creating two views of the original graph G 1 through small perturbations, such as edge deletion, resulting in an augmented graph G 2 . These two views, G 1 and G 2 , are assumed to have minimal differences and preserve label invariance. For each node i, the goal of the contrastive loss is to make the node's representations across the two views z 1 i (from G 1 ) and z 2 i (from G 2 ) more similar (positive pairs), while ensuring that the representations of different nodes z 1 j (from G 1 , j = i) and z 2 i (from G 2 ) are as dissimilar as possible (negative pairs). The contrastive loss can be formulated as follows:
L con = - 1 n n i=1 log exp(sim(z 1 i , z 2 i )/τ ) n j=1 exp(sim(z 1 i , z 2 j )/τ )(10)
where sim(•, •) is a similarity function, τ is a temperature.
From the perspective of equivalence classes, current GCL methods implicitly assume a trivial equivalence class structure, where each node is treated as its own equivalence class. Specifically, the contrastive loss enforces that each node's representation remains consistent across views (within the same trivial equivalence class) while being distinct from the representations of all other nodes (across different classes).
Clearly, the assumption of trivial equivalence classes limits the ability of GCL to capture richer graph structures. Treat-ing each node as a separate equivalence class ignores similarities between nodes, such as automorphic equivalence or attribute-based similarity.
this section cite: []

Section: Role of Augmentation and Contrastive
GCL primarily relies on two mechanisms: graph augmentation and a contrastive loss that implicitly enforces equivalence class constraints. To disentangle their respective contributions, we conduct ablation studies on the loss in Eq. ( 10) using five benchmark datasets (Kipf & Welling, 2017). In the first ablation (w/o aug), we remove graph augmentations and perform contrastive learning on two identical views of the graph (i.e., G 1 = G 2 ), using a single shared encoder (instead of two). This design isolates the effect of augmentation by ensuring there is no variation in the contrastive loss other than the inherent randomness in training. In the second ablation (w/o Pos), we modify the contrastive loss by removing positive pairs-i.e., the term encouraging the similarity of z 1 i and z 2 i -and retain only the negative sample term. This effectively disrupts the equivalence class constraint, as nodes are no longer required to maintain consistent representations across views. Table 4 presents the corresponding results. The results in the table show that removing graph augmentations has minor impact on performance, indicating that augmentations are not the primary driver of GCL's effectiveness. In contrast, removing positive pairs significantly reduces performance across most data, highlighting the importance of equivalence constraints in the contrastive loss.
this section cite: ['b17']

Section: Comparison with the SOTA Methods

this section cite: []

Section: Experimental Setup
Datasets. We evaluate the proposed model on both node classification and graph classification tasks. For node classification, we use 8 benchmark datasets: Cora, Citeseer, Pubmed (Kipf & Welling, 2017), Wiki-CS, Amazon-Computers, Amazon-Photo, Coauthor-CS, and Coauthor-Physics (Shchur et al., 2018). For graph classification, we evaluate on 8 datasets from the TUDataset benchmark (Morris et al., 2020), including NCI1, PROTEINS, DD, MUTAG, COLLAB, RDT-B, RDT-M5K, and IMDB-B.
Baselines. We compare our method against a wide range of baseline methods across both node-level and graphlevel tasks. For graph-level task: 1) two supervised learn-ing methods, including GCN (Kipf & Welling, 2017) and GIN (Xu et al., 2018); 2) four kernel-based methods, including SP (Borgwardt & Kriegel, 2005), GK (Shervashidze et al., 2009), WL (Shervashidze et al., 2011), DGK (Yanardag & Vishwanathan, 2015); 3) three unsupervised methods, including NODE2VEC (Grover & Leskovec, 2016), SUB2VEC (Adhikari et al., 2018), GRAPH2VEC (Narayanan et al., 2017); 4) five self-supervised graph contrastive learning, including INFOGRAPH (Sun et al., 2020), GRAPHCL (You et al., 2020), AD-GCL (Suresh et al., 2021), JOAOv2 (You et al., 2021), RGCL (Li et al., 2022), SIM-GRACE (Xia et al., 2022), SEGA (Wu et al., 2023), and AUTOGCL (Yin et al., 2022); For node-level tasks: 1) Supervised learning methods, including MLP and GCN (Kipf & Welling, 2017); 2) Graph embedding methods, including DEEPWALK (Perozzi et al., 2014) and NODE2VEC (Grover & Leskovec, 2016); 3) Graph contrastive learning methods, including VGAE (Kipf & Welling, 2016) , DGI (Velickovic et al., 2019), GMI (Peng et al., 2020), MVGRL (Hassani & Khasahmadi, 2020), GRACE (Zhu et al., 2020), GCA (Zhu et al., 2021), BGRL (Thakoor et al., 2022), CCA-SSG (Zhang et al., 2021) and SUGRL (Mo et al., 2022).
Protocol. We follow the standard evaluation protocol of previous state-of-the-art self-supervised learning methods. For node classification, we report the mean accuracy on the test set after 50 runs of training. Pretrained node embeddings are used to train a linear neural network for classification. The dataset is split into 10%/10%/80% for training, validation, and testing, respectively. For graph classification, we evaluate the learned graph representations using a linear SVM classifier. We report the mean 10-fold cross-validation accuracy across 5 runs. For each training fold, the linear SVM is tuned using cross-validation, and the best mean accuracy is reported. The dataset is split into 80%/10%/10% for training, validation, and testing, respectively.
Implementation Details. We implement both GALE and its variant GALE-APR using PyTorch Geometric. The key difference is that GALE uses Nauty (McKay & Piperno, 2014) for exact automorphisms, while GALE-APR employs PageRank equivalence with α = 0.85. For node attributes, both rely on attr with ∈{0, 10 -7 , . . . , 10 -1 }.
We adopt the Adam optimizer, tuning learning rates {0.0001, 0.001, 0.01}, batch sizes {16, 64, 128, 256, 512}.
As iterations increase, Intra-class loss (4) dominates, causing the total loss (6) to become imbalanced and too small. We can add Softplus after the discriminator to mitigate this.
this section cite: ['b17', 'b40', 'b17', 'b58', 'b3', 'b41', 'b42', 'b59', 'b13', 'b0', 'b34', 'b43', 'b61', 'b44', 'b62', 'b24', 'b56', 'b54', 'b60', 'b17', 'b37', 'b13', 'b16', 'b47', 'b36', 'b14', 'b65', 'b66', 'b45', 'b63', 'b32', 'b29']

Section: Results and Analysis
Performance on Graph-level Tasks. We evaluate whether GALE and its variant GALE-APR can outperform state-ofthe-art methods on multiple graph-level benchmarks. Table 5 summarizes the results for both supervised and unsupervised baselines. Overall, GALE achieves top-tier perfor-
this section cite: []

Section: Conclusion
This paper proposes a unified self-supervised graph representation framework, GALE, which integrates structural automorphism and attribute equivalence to construct joint equivalence classes. Such combined equivalences are frequently encountered in real-world scenarios, including computing networks. To address computational bottlenecks and noise interference, we design an efficient approximation strategy with linear complexity. GALE preserves the similarity of equivalent nodes while enforcing separation between representations of non-equivalent nodes. Analysis reveals the inherent limitations of mainstream graph models, including over-smoothing in MPNNs, failure to respect automorphism constraints in Transformer positional encodings, and the degeneration of graph contrastive learning into single-node equivalence. Experimental results demonstrate the superiority of GALE.
this section cite: []

Section: References
Ref_id:b0 Title: Sub2vec: Feature learning for subgraphs Year: (2018)
Ref_id:b1 Title: Isomorphism problem for a class of pointsymmetric structures Year: (1977)
Ref_id:b2 Title: Laplacian eigenmaps and spectral techniques for embedding and clustering Year: (2001)
Ref_id:b3 Title: Shortest-path kernels on graphs Year: (2005)
Ref_id:b4 Title: Simple yet effective graph contrastive learning for recommendation Year: (2023)
Ref_id:b5 Title: Automorphisms of graphs. Topics in algebraic graph theory Year: (2004)
Ref_id:b6 Title: Deep neural networks for learning graph representations Year: (2016)
Ref_id:b7 Title: Graph neural networks with learnable structural and positional representations Year: (2022)
Ref_id:b8 Title: Role similarity and complexity in social networks Year: (1985)
Ref_id:b9 Title: automorphism partitioning, and canonical labeling can be solved in polynomial-time for molecular graphs Year: (1998)
Ref_id:b10 Title: Community detection in graphs Year: (2010)
Ref_id:b11 Title: Social positions in influence networks Year: (1997)
Ref_id:b12 Title: On the relationship between pagerank and automorphisms of a graph Year: (2021)
Ref_id:b13 Title: Node2vec: Scalable feature learning for networks Year: (2016)
Ref_id:b14 Title: Contrastive multi-view representation learning on graphs Year: (2020)
Ref_id:b15 Title: Conflict propagation and component recursion for canonical labeling Year: (2011)
Ref_id:b16 Title: Variational graph auto-encoders Year: (2016)
Ref_id:b17 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b18 Title: Physicochemical graph neural network for learning protein-ligand interaction fingerprints from sequence data Year: (2024)
Ref_id:b19 Title: Time complexity analysis of graph algorithms in big data: Evaluating the performance of pagerank and shortest path algorithms for large-scale networks Year: (2024-08)
Ref_id:b20 Title: On the generalization of equivariance and convolution in neural networks to the action of compact groups Year: (2018)
Ref_id:b21 Title: Rethinking graph transformers with spectral attention Year: (2021)
Ref_id:b22 Title: Deeper inside pagerank Year: (2004)
Ref_id:b23 Title: Topics in graph automorphisms and reconstruction Year: (2016)
Ref_id:b24 Title: Let invariant rationale discovery inspire graph contrastive learning Year: (2022)
Ref_id:b25 Title: Sign and basis invariant networks for spectral graph representation learning Year: (2023)
Ref_id:b26 Title: Perfect alignment may be poisonous to graph contrastive learning Year: (2024)
Ref_id:b27 Title: Arc: A generalist graph anomaly detector with in-context learning Year: (2024)
Ref_id:b28 Title: Structural equivalence of individuals in social networks Year: (1971)
Ref_id:b29 Title: Practical graph isomorphism, ii Year: (2014)
Ref_id:b30 Title: Comparing clusterings-an information based distance Year: (2007)
Ref_id:b31 Title: Transformer for graphs: An overview from architecture perspective Year: (2022)
Ref_id:b32 Title: Simple unsupervised graph representation learning Year: (2022)
Ref_id:b33 Title: Tudataset: A collection of benchmark datasets for learning with graphs Year: (2020)
Ref_id:b34 Title: Graph2vec: Learning distributed representations of graphs Year: (2017)
Ref_id:b35 Title: Graph automorphism group equivariant neural networks Year: (2024)
Ref_id:b36 Title: Graph representation learning via graphical mutual information maximization Year: (2020)
Ref_id:b37 Title: Online learning of social representations Year: (2014)
Ref_id:b38 Title: Recipe for a general, powerful, scalable graph transformer Year: (2022)
Ref_id:b39 Title: Objective criteria for the evaluation of clustering methods Year: (1971)
Ref_id:b40 Title: Pitfalls of graph neural network evaluation Year: (2018)
Ref_id:b41 Title: Efficient graphlet kernels for large graph comparison Year: (2009)
Ref_id:b42 Title: Weisfeiler-lehman graph kernels Year: (2011)
Ref_id:b43 Title: Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization Year: (2020)
Ref_id:b44 Title: Adversarial graph augmentation to improve graph contrastive learning Year: (2021)
Ref_id:b45 Title: Largescale representation learning on graphs via bootstrapping Year: (2022)
Ref_id:b46 Title: Autobahn: Automorphism-based graph neural nets Year: (2021)
Ref_id:b47 Title: Deep graph infomax Year: (2019)
Ref_id:b48 Title: S3GCL: Spectral, swift, spatial graph contrastive learning Year: (2024)
Ref_id:b49 Title: Ime: Integrating multi-curvature shared and specific embedding for temporal knowledge graph completion Year: (2024)
Ref_id:b50 Title: Large language models-guided dynamic adaptation for temporal knowledge graph reasoning Year: (2024)
Ref_id:b51 Title: Made: Multicurvature adaptive embedding for temporal knowledge graph completion Year: (2024)
Ref_id:b52 Title: Graph transport learning with optimal plan alignment Year: (2023)
Ref_id:b53 Title: Social network analysis: Methods and applications Year: (1994)
Ref_id:b54 Title: SEGA: Structural entropy guided anchor view for graph contrastive learning Year: (2023)
Ref_id:b55 Title: A comprehensive survey on graph neural networks Year: (2020)
Ref_id:b56 Title: Simgrace: A simple framework for graph contrastive learning without data augmentation Year: (2022)
Ref_id:b57 Title: Automorphic equivalenceaware graph neural network Year: (2021)
Ref_id:b58 Title: How powerful are graph neural networks? Year: (2018)
Ref_id:b59 Title: Deep graph kernels Year: (2015)
Ref_id:b60 Title: Autogcl: Automated graph contrastive learning via learnable view generators Year: (2022)
Ref_id:b61 Title: Graph contrastive learning with augmentations. Advances in Neural Information Processing Systems Year: (2020)
Ref_id:b62 Title: Graph contrastive learning automated Year: (2021)
Ref_id:b63 Title: From canonical correlation analysis to self-supervised graph neural networks Year: (2021)
Ref_id:b64 Title: Graph contrastive learning with progressive augmentations Year: (2025)
Ref_id:b65 Title: Deep graph contrastive representation learning Year: (2020)
Ref_id:b66 Title: Graph contrastive learning with adaptive augmentation Year: (2021)
