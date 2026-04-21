Title: Local Identifying Causal Relations in the Presence of Latent Variables
Abstract: We tackle the problem of identifying whether a variable is the cause of a specified target using observational data. State-of-the-art causal learning algorithms that handle latent variables typically rely on identifying the global causal structure, often represented as a partial ancestral graph (PAG), to infer causal relationships. Although effective, these approaches are often redundant and computationally expensive when the focus is limited to a specific causal relationship. In this work, we introduce novel local characterizations that are necessary and sufficient for various types of causal relationships between two variables, enabling us to bypass the need for global structure learning. Leveraging these local insights, we develop efficient and fully localized algorithms that accurately identify causal relationships from observational data. We theoretically demonstrate the soundness and completeness of our approach. Extensive experiments on benchmark networks and two real-world datasets further validate the effectiveness and efficiency of our method.

Section: Introduction
Identifying causal relationships, known as causal discovery, plays a crucial role in various fields, including computer science (Jonas et al., 2017;Pearl, 2018;Schölkopf, 2022), sociology (Spirtes et al., 2000), epidemiology (Hernán & Robins, 2010), and neuroscience (Smith et al., 2011;Sanchez-Romero et al., 2019). A key challenge in causal discovery is determining whether one variable causes another (Pearl, 2009). For instance, in medical diagnosis, understanding the causal relationships between symptoms and diseases is crucial for accurate diagnoses and the devel- The causal graphical model is one of the most widely used models for graphically representing causal relations among observed variables, which consists of vertices denoting variables and edges denoting causal relations (Pearl, 2009;Spirtes et al., 2000). Directed acyclic graphs (DAGs) are widely used to represent causal relationships among observed variables under the assumption of causal sufficiency (Pearl, 2009;Spirtes et al., 2000), i.e. , the absence of unobserved latent variables between observed variables. However, when latent variables are present, DAGs may fail to represent the causal relationships among observed variables (Richardson & Spirtes, 2002). As shown in Figure 1 (b), the DAG incorrectly depicts Hypoxia in O 2 as a cause of Hypoxia distribution, but there is no directed path from Hypoxia in O 2 to Hypoxia distribution in the underlying graph. Therefore, directly employing methods that do not account for the influence of latent variables (when the system contains latent variables), such as those in Fang et al. (2022); Zuo et al. (2022); Zheng et al. (2024), may result in incorrect inferences about the causal relationships among the observed variables.
Maximal Ancestral Graphs (MAGs), whose main advantage is that, without explicitly including latent variables, they can represent conditional independence and causal relationships among observed variables (Richardson & Spirtes, 2002;Zhang, 2008). In a MAG, a vertex X is an ancestor (cause) of a vertex Y and Y is a descendant (effect) of X if there is a directed path from X to Y (Zhang, 2006). As shown in Figure 1 (c), it correctly represents Hypoxia in O 2 does not cause Hypoxia distribution, and vice versa. However, from the observational data, without additional distributional assumptions or background knowledge, we generally learn a Markov equivalence class (MEC) of MAGs that encodes the invariant features of the underlying MAG, which can be represented by a partial ancestral graph (PAG) (Spirtes & Richardson, 1996;Zhang & Spirtes, 2005;Ali et al., 2005;Zhao et al., 2005) 1 . The undirected edges (or marks) in a PAG imply that some causal relations among variables cannot be read from the graph directly. Hence, given a Markov equivalent class of MAGs, there are three possible types of causal relationships 2 :
1. A variable X is an invariant ancestor of a variable Y if and only if there is a directed path from X to Y in every equivalent MAG. 2. A variable X is an invariant non-ancestor of a variable Y if and only if there is no directed path from X to Y in any equivalent MAG. 3. A variable X is a possible ancestor of variable Y if X is neither an invariant ancestor nor an invariant nonancestor of Y .
A direct approach to identifying the causal relationship between a pair of variables (X, Y ) is to first use methods like FCI (Spirtes et al., 2000) or RFCI (Colombo et al., 2012) to learn a PAG from observational data, and then enumerate all MAGs within this class to determine whether X is an invariant ancestor or non-ancestor of Y across all equivalent MAGs. However, this approach becomes computationally inefficient when the number of MAGs in the MEC is large (Malinsky & Spirtes, 2016). In addition, these approaches are often redundant and computationally expensive when the focus is limited to a specific causal relationship. In this paper, we address the challenge of locally identifying the causal relationship between a pair of variables without requiring the learning of a full PAG, the enumeration of MAGs, or the assumption of causal sufficiency. Our primary contributions are summarized as follows:
• We provide both sufficient and necessary local characterizations for the invariant ancestor, invariant non-ancestor, and possible ancestor relationships, relying solely on local structure rather than the entire graph, even in the presence of latent variables.
• We propose a novel algorithm, LocICR, that locally identifies the causal relationship between a pair of variables. We provide theoretical proof of its completeness, demonstrating that it can identify the same causal relationships 1 Under the assumption of causal sufficiency, the Markov equivalence class (MEC) of the underlying DAG is typically represented by a completed partially directed acyclic graph (CPDAG). 2 One may refer to Appendix A.3 for further clarification of these causal relationships.
for a target pair of variables as state-of-the-art global learning approaches.
• We conduct extensive experiments on benchmark network structures and real-world datasets, showcasing the effectiveness and efficiency of our method.
this section cite: ['b17', 'b32', 'b42', 'b48', 'b16', 'b44', 'b41', 'b31', 'b31', 'b48', 'b31', 'b48', 'b38', 'b11', 'b63', 'b61', 'b38', 'b58', 'b57', 'b47', 'b1', 'b60', 'b48', 'b7', 'b25']

Section: Related Works
This paper focuses on identifying causal relationships between variable pairs in the presence of latent variables. Existing methods generally fall into two categories: global structure-based learning and local structure-based learning.
Global Structure-Based Learning. This category begins by using FCI (Spirtes et al., 2000) and its variants (Colombo et al., 2012;Claassen et al., 2013;Claassen & Heskes, 2011;Ogarrio et al., 2016;Raghu et al., 2018;Tsirlis et al., 2018;Akbari et al., 2021;Rohekar et al., 2021;Bhattacharya et al., 2021;Claassen & Bucur, 2022) to learn the global causal PAG. Then, it infers the causal relationships based on criteria proposed by Zhang (2006) and Roumpelaki et al. (2016); Mooij & Claassen (2020) or through causal effect estimation methods, such as LV-IDA (Malinsky & Spirtes, 2016) and its extensions (Maathuis et al., 2009;Nandy et al., 2017;Liu et al., 2020a;Fang & He, 2020;Pensar et al., 2020;Wang et al., 2023a). However, these methods rely on global causal graph, which can be computationally expensive and restrictive (Guo & Perkovic, 2021;Fang et al., 2022).
Local Structure-Based Learning. This category primarily focuses on identifying relationships between a target variable and its adjacent variables. Well-known algorithms in this line include (Yin et al., 2008;Zhou et al., 2010;Wang et al., 2014;Gao & Ji, 2015;Liu et al., 2020b;Yang et al., 2021;Liang et al., 2023). Recently, Fang et al. (2022) and its variants (Zheng et al., 2024) introduced local approaches for identifying causal relationships among arbitrary pairs of variables within a system. However, these methods require the assumption of causal sufficiency. More recently, Xie et al. (2024) proposed a method for identifying local causal structures in the presence of latent variables. Nevertheless, their approach focuses on relationships between a target variable and its adjacent variables, without generalizing to arbitrary pairs of variables.
To our knowledge, no method locally identifies the causal relationship between an arbitrary pair of variables, without assuming causal sufficiency and learning the full causal PAG.
this section cite: ['b48', 'b7', 'b6', 'b5', 'b28', 'b36', 'b49', 'b0', 'b39', 'b3', 'b4', 'b57', 'b40', 'b26', 'b25', 'b24', 'b27', 'b35', 'b14', 'b11', 'b56', 'b62', 'b50', 'b12', 'b55', 'b20', 'b11', 'b61', 'b54']

Section: Preliminaries

this section cite: []

Section: Terminology
A univariate variable (or vertex) is denoted by an uppercase letter (e.g., V ), while sets of variables (or vertices) are denoted by bold uppercase letters (e.g., V).
Graphs. A graph G = (V, E) consists of a set of vertices V = {V 1 , . . . , V n }
and a set of edges E. The two ends of an edge are called marks. A graph is directed mixed if the edges in the graph are directed (→), or bi-directed (↔). A directed mixed graph is ancestral if it doesn't contain a directed or almost directed cycle. An ancestral graph is a maximal ancestral graph (MAG, denoted by M) if for any two non-adjacent vertices, there exists a set of vertices that m-separates them. Two MAGs are Markov equivalent if they share the same m-separations. A class of Markov equivalent MAGs, denoted as [M], can be represented as a Partially Ancestral Graph (PAG, denoted by P), where a tail '-' or arrowhead '>' occurs if the corresponding mark is tail or arrowhead in all the Markov equivalent MAGs, and a circle '•' occurs otherwise. For convenience, we use an asterisk (*) to denote any possible mark of a PAG (•, >, -) or a MAG (>, -). For two vertices
V i and V j in P, V i is a possibly parent/possibly child/neighbor of V j if there is V i •→ V j /V i ←•V j /V i •-• V j in P. A path is a collider path if every non-endpoint vertex on it is a collider along the path. A directed path from V i to V j is a path composed of directed edges pointing towards V j . A partially directed path from V i to V j is a path where every edge without an arrowhead at the mark near V i . A path from V i to V j that is not possibly causal is called a non-causal path from V i to V j .
The detailed graph-related definitions are provided in Appendix A.1.
this section cite: []

Section: Markov Blanket.
The Markov blanket(MB) of a variable X is the smallest set conditioned on which all other variables are statistically independent of X 3 . Graphically, assuming faithfulness, in a DAG, this is the set of parents, children, and children's parents of vertex X. In a MAG, the Markova blanket of a vertex X, noted as MB (X, M), consists of the set of parents, children, children's parents of X, as well as the district of X and of the children of X, and the parents of each vertex of these districts, where the district of a vertex V is the set of all vertices reachable from V using only bidirected edges. Figure 2 specifically illustrates the Markov blanket of vertex X in a MAG. The vertices shaded in blue belong to MB (X, M).
Let P represent the MEC of M, the MB remains invariant across M and P, i.e., MB (X, M) = MB (X, P). For simplicity, MB (X) denotes MB (X, P) when unambiguous, and MB + (X ) denotes {MB (X ) ∪ X}.
Notations. 4 We denote:
Pa(V i , G), De(V i , G), Adj (V i , G)
as the parent, descendant, and adjacent vertex sets of V i in G;
3 Some authors use the term "Markov blanket" without the notion of minimality, and use "Markov boundary" to denote the smallest Markov blanket. For clarity, we adopt the convention that the Markov blanket refers to the minimal Markov blanket. 4 The main symbols are summarized in Table 1 in Appendix A. Ne(V i , P) and PossCh(V i , P) as the neighbor and possible child sets of V i in a PAG P; [P] as the Markov equivalence class represented by PAG P. We use the notation X ⊥ ⊥ Y|Z for "X is statistically independent of Y given Z", and X ̸⊥ ⊥ Y|Z for the negation of the same sentence. We use the (X, Y ) to denote the target pair of variables, identifying whether X is an invariant non-ancestor/invariant ancestor/possible ancestor of Y .
this section cite: []

Section: Problem Definition
Our work is in the framework of causal graphical models ⟨G, Θ G ⟩, where G represents the causal structure and Θ G refers to the associated parameters with G (Pearl, 2009). The causal structure G = ⟨V, E⟩ is a directed acyclic graph (DAG) where V represents a set of vertices and E represents a set of edges. The parameters Θ G specify a functional relationship for each
V i ∈ V, in the form V i = f i (P a(V i ), u i ),
where u i represents independent errors due to omitted factors, and all error terms u i are assumed to be mutually independent. The variable set V consists of the observed variables O and the latent variables L.
Task. Under the standard assumptions of the causal Markov condition and the causal Faithfulness condition, our objective is to characterize the local graphical features of different types of causal relationships between a pair of variables X and Y , where X, Y ∈ O. Subsequently, we aim to develop a fully local algorithm to determine the causal relationship between X and Y . Note that we do not assume causal sufficiency, allowing for latent variables between observed variables in the system.
this section cite: ['b31']

Section: Local Characterization of Causal Relations

this section cite: []

Section: Foundations of Local Characterization
In this section, we build on the well-established local Markov property for DAGs to introduce the local Markov property for MAGs, following the ordered local Markov property described in Richardson & Spirtes (2002). Specifically, the local Markov property states that a target variable is independent of its non-descendants, given a particular set of variables.
D E F H J A B L2 L3 C G L1 (a) Underlying Graph D E F H J B G A C (b) MAG D E F H J B G A C (c) PAG
To define this particular set, we first introduce a particular type of collider path, as outlined in Definition 1.
Definition 1 (Arrow-Collider Path). In a PAG or a MAG, a path π = ⟨V 0 , . . . , V n ⟩ is called an arrow-collider path from V 0 to V n if every non-endpoint vertex is a collider on π, and the edge between
V 0 and V 1 points into V 0 , i.e. , V 0 ↔ V 1 • • • ← * V n . If n = 1, π simplifies to V 0 ← * V 1 .
Building on Definition 1, we define the particular set graphically in Definition 2.
Definition 2 (Augmented Parent Set). Let G be a PAG or a MAG. The augmented parent set of a vertex X, denoted as Pa * (X, G), is defined as follows: for any vertex V ∈ O, V ∈ Pa * (X, G) if and only if there exists an arrow-collider path π from X to V such that:
(1) in a MAG, X is a non-ancestor of every vertex on π, including V . (2) in a PAG, X is an invariant non-ancestor of all vertices on π, including V .
In general, the augmented parent set of a target vertex includes not only its direct parents but also vertices connected via arrow-collider paths. See Example 1 for illustrations of Definitions 1 and 2.
Below, we introduce the local Markov property for MAGs using the concept of the augmented parent set. Notably, this result is theoretically equivalent to the ordered local Markov property for MAGs proposed by Richardson & Spirtes (2002). For details, see Theorem 7 in Appendix B.
Definition 3 (Local Markov Property for MAGs). Let M be the MAG over O, and let Pre(X, M) denote the pretreatment vertices of X in M, i.e. , the vertices for which X is not an ancestor. The local Markov property for the MAG states that for every variable X ∈ O, the following property holds:
X ⊥ ⊥ Pre(X, M) \ Pa * (X, M) | Pa * (X, M) (1)
If no latent variables exist in the system, the local Markov property for MAGs reduces to the local Markov property for DAGs, i.e., each variable in the DAG is independent of its non-descendants given its parents.
Example 1. Consider the MAG M shown in Figure 3 (b), with J as the target vertex of interest. Each vertex V ∈ {A, B, C, F, G} is connected to J by an arrowcollider path from J to V . Additionally, the descendant set of J in M is De(J, M) = {D, E, F, B}. As a result, the augmented parent set of J is Pa * (J, M) = {A, G} and Pre(J, M) = {A, C, G, H}. Therefore, the local Markov property is J ⊥ ⊥ {C, H} | {A, G}.
this section cite: ['b38', 'b38']

Section: Local Characterization
As outlined in the introduction, causal relationships between pairs of variables, inferred from non-experimental observational data can be classified into three types: invariant non-ancestors, invariant ancestors, and possible ancestors.
In this section, we present local characterizations of these causal relationships, which rely on the induced subgraph of the PAG over MB + (X).
We begin by presenting the local characterization of invariant non-ancestor relations, as shown in Theorem 1 below.
Theorem 1. Let P be the PAG over O. For any pair of vertices (X, Y ) in P, X is an invariant non-ancestor of Y if and only if X ⊥ ⊥ Y | Pa * (X, P). Intuitively, according to the local Markov property, given a PAG (the MEC of MAGs), Pa * (X, P) blocks all noncausal paths from X to Y in P. If X is not an invariant non-ancestor of Y (i.e. , there exists a directed path from
X to Y in a MAG M ∈ [P]), this implies X ̸⊥ ⊥ Y | Pa * (X, P). Conversely, if X ⊥ ⊥ Y | Pa * (X, P),
then there is no directed path from X to Y in any M ∈ [P], and X is an invariant non-ancestor of Y . Example 2. Consider the PAG shown in Figure 3 (c), with (J, C) as the target pair of interest. Each vertex in {A, B, C, F, G} is connected to J by an arrow-collider path from J to it. By Definition 2, we have Pa * (J, P) = {A, G}. Furthermore, there is no active path between J and C given {A, G}, which implies J ⊥ ⊥ C | {A, G}. Consequently, J is an invariant non-ancestor of C.
We now turn to the concept of the invariant ancestor relation.
To clarify its local characterization, we define the local features of the invariant ancestor relation, categorized into two types: explicit invariant ancestor and implicit invariant ancestor, as detailed in Definitions 4 and 5, respectively. Definition 4 (Explicit Invariant Ancestor). Given a PAG P, a vertex X is an explicit invariant ancestor of another vertex Y if and only if a common directed path exists from X to Y in every MAG within [P]. Definition 5 (Implicit Invariant Ancestor). Given a PAG P, a vertex X is an implicit invariant ancestor of another vertex Y if and only if X is an invariant ancestor of Y , but there is no directed path from X to Y common to every MAG within [P]. Remark 1. We define X as an invariant ancestor of Y if there exists a directed path from X to Y in every MAG in [P]. If these directed paths are identical across all MAGs in [P], the invariant ancestor relation is explicit; otherwise, it is implicit. Furthermore, if a common directed path from X to Y exists in every MAG in [P], then a directed path from X to Y must also exist in P (Zhang, 2006).
To graphically characterize these two types of invariant ancestor relations, we define a specific type of collider path in Definition 6, analogous to the arrow-collider path, and a particular set in Definition 7, similar to the augmented parent set.
Definition 6 (Circle-Collider Path). In a PAG, a path π = ⟨V 0 , . . . , V n ⟩ is called a circle-collider path from V 0 to V n if every non-endpoint vertex is a collider on π, and the edge between V 0 and V 1 is undirected relative to V 0 , i.e. ,
V 0 •→ V 1 • • • ← * V n . If n = 1, π simplifies to V 0 •- * V 1 .
Definition 7. Let P be a PAG, the augmented undirected neighbor set of a vertex X, denoted as Ne * (X, P), is defined as follows: For any vertex V ∈ O, V ∈ Ne * (X, P) if and only if there exists a circle-collider path
π = ⟨X = V 0 , V 1 , . . . , V n = V ⟩ from X to V such that for every 2 ≤ i ≤ n, X is an invariant non-ancestor of V i 5 .
Remark 2. Note that in a PAG, Ne(X, P) consists of vertices connected to X by edges of the form •-•, Ne * (X, P) includes not only Ne(X, P) but also vertices connected by circle-collider paths.
Next, we present the local characterization of explicit invariant ancestor relations, as stated in Theorem 2.
Theorem 2. Let P be the PAG over O. For any pair of vertices (X, Y ) in P, X is an explicit invariant ancestorof Y if and only if X ̸⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X, P). Intuitively, given a PAG, Ne * (X, P) blocks all partially directed paths(except directed paths) from X to Y in P, while Pa * (X, P) blocks all non-causal paths that from X to Y in P. If X is not an explicit invariant ancestor of Y (i.e. , no directed path from X to Y exists in P), then X ⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X, P). Conversely, if X ̸⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X, P), a directed path from X to Y exists in P, making X an explicit invariant ancestor of Y .
Example 3. Consider the PAG shown in Figure 3 (c), with (J, B) be the target pair of interest. From Example 2, we know Pa * (J, P) = {A, G}. Additionally, Ne * (J, P) = ∅ , as no vertex is connected to J by a circle-collider path. Furthermore, there is an active path ⟨J, D, E, . . . , F, B⟩ between J and B, conditioned on {A, G}, implying J ̸⊥ ⊥ B | {A, G}. Thus, J is an invariant explicit ancestor of B.
We now characterize implicit invariant ancestor relations locally based on Definition 8, as detailed in Theorem 3.
Definition 8. Let P be a PAG and let M represent the set of maximal cliquesfoot_1 of the induced subgraph of P over PossCh(X, P) ∪ Ne(X, P). The set of augmented undirected neighbor of a vertex X relative to a maximal clique M ∈ M, denoted as Ne * (X M , P). For any vertex V ∈ O, V ∈ Ne * (X M , P) if and only if there exists a circle-collider path
π = ⟨X = V 0 , V 1 , . . . , V n = V ⟩ from X to V such that (1) for every 2 ≤ i ≤ n, X is an invariant non-ancestor of V i and (2) V 1 ∈ M. Theorem 3. Let P be the PAG over O. For any pair of vertices (X, Y ) in P, X is an implicit invariant ancestor of Y if and only if (1) X ⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X, P), but (2) X ̸⊥ ⊥ Y | Pa * (X, P)∪Ne * (X M , P) for every maximal clique M ∈ M.
The first condition implies that there is no common directed path from X to Y in any MAG within [P]. Intuitively, similar to the role of Ne
* (X, P) in P, Ne * (X M , P) can block all partially directed paths (excluding directed paths) from X to Y that pass through M in P. If X ̸⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X M , P) for a maximal clique M ∈ M, this indicates the presence of partially directed paths from X to Y in P, given Pa * (X, P) ∪ Ne * (X M , P). That is, in some MAGs within [P] where a directed path from X to Y exists, given Pa * (X, P) ∪ Ne * (X M , P). Moreover, if X ̸⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X M , P) holds for every M ∈ M, then all MAGs within [P] contain a directed path from X to Y , meaning X is an implicit invariant ancestor of Y . Conversely, if X ⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X M , P)
for a maximal clique M ∈ M, this implies that some MAGs within
(G, P) = {H, J, A}. For M = {H}, we have Ne * (G M , P) = {H}. Similarly, for M = {J}, we have Ne * (G M , P) = {J, A}. Observing that G ⊥ ⊥ D | {H, J, A}, G ̸⊥ ⊥ D | {H} and G ̸⊥ ⊥ D | {J, A}.
Consequently, G is an invariant implicit ancestor of D.
Based on Theorems 2 and 3, we provide a sound and complete local characterization of invariant ancestor relations. Corollary 1. Let P be the PAG over O, and let M denote the set of maximal cliques of the induced subgraph of P over PossCh(X, P) ∪ Ne(X, P). For any pair of vertices
(X, Y ) in P, X is an invariant ancestor of Y if and only if (1) X ̸⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X, P), or (2) X ̸⊥ ⊥ Y | Pa * (X, P)∪Ne * (X M , P) for every M ∈ M
By Theorem 1 and Corollary 1, we can identify all stable causal relationships (invariant non-ancestor and invariant ancestor). Naturally, the remaining ones (possible ancestors) are subject to change, as stated below.
Theorem 4. Let P be the PAG over O. For any pair of vertices (X, Y ) in P, X is an possible ancestor of Y if and only if neither Theorem 1 nor Corollary 1 applies.
this section cite: ['b57']

Section: The Local Identifying Algorithm
This section discusses how to locally learn the Conditional Sets involved in the aforementioned theoretical results and present the algorithm for locally identifying causal relations.
this section cite: []

Section: Locally Learning Conditional Sets
We incorporate the properties of Markov Blanket (MB) to locally learn the conditional set used in the above results of local characterizations, i.e. , Pa * (X, P), Ne * (X, P), and Ne * (X M , P). Specifically, we answer the following questions:
• How to discover which vertices in MB (X) are connected to X via arrow-collider paths or circle-collider paths? This involves locally learning the induced subgraph of P over MB + (X), denoted as P MB + (X) .
• How to determine whether X is an invariant nonancestor of the vertices on these paths? This requires identifying all vertices in MB (X) for which X is an invariant non-ancestor, denoted as IPre MB (X).
To address the first question, we extend the MMB-by-MMB algorithm proposed by Xie et al. (2024) to learn P MB + (X) .
Note that the original MMB-by-MMB algorithm focuses solely on learning the structure involving the target variable and its adjacent variables, whereas we generalize it to apply to any variable, not limited to the target variable's adjacent variables. Specially, the P MB + (X) learning process is iterative, with each step focusing on the local structure of a variable V i , denoted as L Vi , derived from the observed data of MB + (V i ) and is relevant for constructing P MB + (X) . Let Waitlist store variables that are potentially relevant for learning P MB + (X) , and Donelist store variables removed from Waitlist and P store true causal information, the basic idea is as follows:
Learning P MB + (X) algorithm1
. Given a target variable X, observed data O. Initialize Waitlist = {X}, Donelist = ∅, and P = ∅. 2. During each step, perform sequential iterations, focusing on the first variable V i in Waitlist. First, learn MB (V i ), followed by the local structure L Vi over the observed data of MB + (V i ). Next, extract true edges and direction information from L Vi (by Proposition 1 and Proposition 2) and update P. Orient P using standard orientation criteria. Then, update Waitlist and Donlist. 3. The process ends when the stopping criteria (as defined in Proposition 3) are met. We outline the technical details of the algorithm below. Proposition 1. (Theorem 1 in Xie et al. ( 2024)) Let X be any vertex in O, and V be a vertex in MB (X ). Then X and V are m-separated by a subset of O \ {X, V } if and only if they are m-separated by a subset of MB (X ) \ {V }.
Proposition 1 ensures that the existence of an edge between X and another vertex V ∈ MB (X ), as identified in L X , is consistent with the edge identified in the PAG learned from the observed data O.
Proposition 2. Let L X be the inferred PAG over MB + (X ).
Let V i (1 ≤ i ≤ |MB (X )|) represent the vertices in MB (X ).
The following statements hold:
S1. The unshielded collider triples (V-structures) V 1 * → X ← * V 2 identified in L X are consistent with those in the ground-truth PAG. S2. The uncovered collider paths X * → V 1 ↔ • • • ← * V i
identified in L X are consistent with those in the groundtruth PAG.
Proposition 2 specifies the locally identified colliders that are correct, representing the true directional information that can be retained.
Proposition 3 (Stop Rules). Let X be the target variable of interestand Waitlist represent the collection of variables whose L will be learned. If any of the following rules are satisfied, the learned P MB + (X) is equivalent to the structure identified by global learning methods.
this section cite: ['b54']

Section: R1.
The edges among the variables of MB + (X) are all determined, i.e. , no circle present in the marks. R2. The Waitlist is empty.
this section cite: []

Section: R3.
All paths from each vertex in MB + (X), which include undirected edges (connected two vertices in MB + (X)), are blocked by edges * →.
R1 and R2 indicate that all causal information of interest has been identified, or L for all variables in O has been learned. Broadly, R3 states that if all paths connecting the undirected edges between two vertices in MB + (X) are all blocked by the edge * →, further exploration of L for the remaining variables will not contribute to determining the direction of these undirected edges.
We address the second question. The basic idea is as follows:
Learning IPre MB (X) algorithm 1. Induced the subgraph P MB + (X) from P. Initialize IPre MB (X) with vertices adjacent to X where the edge is X ← * , and set
CandSet = MB (X) \ Adj (X, P MB + (X) ). 2. Perform sequential iterations, focusing on the first vari- able V i in CandSet during each step. If there exists a set Z ⊆ IPre MB (X) such that X ⊥ ⊥ V i | Z, add V i to
IPre MB (X) and remove it from CandSet. 3. The process ends when no variable in CandSet can be added in IPre MB (X).
Proposition 4 ensures the correctness of the above algorithm.
Proposition 4. Let P be the ground-truth PAG over O, and let P MB + (X) denote the induced subgraph of P over MB + (X). Let IPre MB (X) be the set of invariant non-descendant of X. For a vertex V ∈ {MB (X) \ Adj (X, P MB + (X) )}, X is an invariant non-ancestor of V if and only if there exists a set Z ⊆ IPre MB (X) that mseparates X and V .
After addressing the two main questions, we present the local learning method for conditional sets in Algorithm 1. The detailed pseudocode is provided in Algorithm 1 in Appendix D, with a complete example included in Appendix E.
Theorem 5. Assuming oracle tests for conditional independence, the outputs of Algorithm 1 are identical to those obtained from the ground-truth PAG.
this section cite: []

Section: Local Identifying Causal Relations Algorithm
In this section, we introduce the Local Identifying Causal Relations algorithm, referred to as the LocICR algorithm.
Theorem 6. Assuming the oracle tests for conditional independence. Algorithm 2 is both sound and complete for identifying invariant non-ancestor, explicit invariant ancestor, implicit invariant ancestor, and possible ancestor causal relationships between any pair of variables in O.
Complexity of Algorithm 2. Let r denote the number of local structures to be learned sequentially in Algorithm 1, and let n denote the size of the observed set O. The worst-case complexity is O r(2n-r-1)
2 + r|M B + | 2 2 |M B + | + |M| ,
where |M| is the number of maximal cliques of PossCh(X, P)∪Ne(X, P), and |MB + | is the size of MB + . The first two terms correspond to Line 1 (i.e., Algorithm 1), while the last term corresponds to Lines 2 ∼ 12. The detailed calculation process is provided in Appendix F.
this section cite: []

Section: Algorithm 1 Local Learning Conditional Sets
Input: Target X, observed data O 1: The local structure P MB + (X) is obtained by invoking the Learning P MB + (X) and Learning IPre MB (X) algorithms. 2: Based on P MB + (X) and IPre MB (X), we derive Pa * (X, P), Ne * (X, P), and Ne * (X M , P) for each M ∈ M. Output: Pa * (X, P), Ne * (X, P) and Ne * (X M , P) for each M ∈ M.
this section cite: []

Section: Algorithm 2 LocICR
Input: Target variable pair (X, Y ) , observed data O 1: Pa * (X, P), Ne * (X, P), and Ne * (X M , P) for each M ∈ M are obtained by invoking the Algorithm 1
2: if X ⊥ ⊥ Y | Pa * (X, P) then 3: return X is an invariant non-ancestor of Y 4: end if 5: if X ̸⊥ ⊥ Y | {Pa * (X, P) ∪ Ne * (X, P)} then 6: return X is an explicit invariant ancestor of Y 7: end if 8: M = the set of maximal cliques of {PossCh(X, P) ∪ Ne(X, P)} 9: if exist M ∈ M such that X ⊥ ⊥ Y | Pa * (X, P) ∪ Ne * (X M , P) then 10:
return X is an possible ancestor of Y 11: end if 12: return X is a implicit invariant ancestor of Y Output: The causal relation between X and Y .
this section cite: []

Section: Experimental Results
To evaluate the accuracy and efficiency of our algorithm, we apply it to synthetic datasets generated from benchmark networks, as well as to two real-world datasets. We utilize the existing implementation of the Total Conditioning (TC) discovery algorithm (Pellet & Elisseeff, 2008b) to identify the Markov Blanket (MB) of a variable. Our source code is available at https://github.com/zhengli0060/  LocICR. 6.1. Synthetic Data with Benchmark Networks
In this section, we compare the proposed LocICR algorithm with global structure-based learning methods, including PC-ITC (Spirtes & Glymour, 1991;Fang et al., 2022), RFCI-Zhang (Colombo et al., 2012;Zhang, 2006), M3HC-Zhang (Tsirlis et al., 2018), ICD-Zhang (Rohekar et al., 2021) (based on criteria), and PC-IDA (Maathuis et al., 2009) as well as RFCI-LVIDA (Malinsky & Spirtes, 2016)(using causal effect estimation methods). We also compared it with the local structure-based learning method, Local-ITC, which does not account for latent variables (Fang et al., 2022). Detailed descriptions of these methods are provided in Appendix G.1.
this section cite: ['b46', 'b11', 'b7', 'b57', 'b49', 'b39', 'b24', 'b25', 'b11']

Section: Experimental setup:
We use four benchmark networks varying dimensionality: MILDEW, ALARM, WIN95PTS, and ANDES, containing 35, 37, 76, and 223 vertices, respectively. 7 . Following the convention in (Colombo et al., 2012;Malinsky & Spirtes, 2016;Fang et al., 2022), the benchmark networks are parameterized as linear Gaussian structural causal models, with the causal strengths chosen uniformly at random from the range ±(0.5, 1), and noises drawn from the standard Gaussian distribution. The number of latent variables is set to 4, 4, 6, and 10 for the respective network. For each network, 100 datasets were randomly generated, with latent variables randomly selected for each dataset. Two observed variables were then randomly chosen as the target pair (X, Y ) for each dataset. The reported results are averaged across the 100 datasets.
Metrics: We use the following metrics:
• Weighted Precision (WP): the weighted average of perclass precision, where each precision is the ratio of true positives for the i-th class in the output to the total number of predictions made by the algorithm for the i-th class.
• Weighted Recall (WR): the weighted average of perclass recall, where each recall is the ratio of true positives for the i-th class in the output to the total number of true instances of the i-th class in the ground truth.
• Weighted F1 (WF1): the harmonic mean of WP and WR, calculated as
WF1 = 2 • WP • WR WP + WR .
• nTest: the number of (conditional) independence tests implemented by an algorithm.
Results: Due to space limitations, we present results for two metrics in Figure 4, with complete results provided in Appendix G. As shown, our proposed LocICR algorithm outperforms other methods in Weighted F1 score across all networks and sample sizes, demonstrating its effectiveness. The number of conditional independence tests required by our method is significantly lower than that of global structurebased methods, such as PC-ITC, PC-IDA, RFCI-LVIDA, RFCI-ZHANG, and ICD-Zhang. Moreover, since M3HC is a hybrid method rather than a purely constraint-based one, we excluded it from the comparison regarding the nTest. Notably, although the nTest value of the Local-ITC method is lower for most networks and sample sizes, our method still outperforms Local-ITC on the other three metrics. This is likely because Local-ITC only learns the local structure of X's adjacencies under the assumption of no latent variables. Additionally, methods assuming causal sufficiency, such as PC-ITC, PC-IDA, and Local-ITC, show less satisfactory results, highlighting their inability to handle situations involving latent variables.
6.2. Application to Real-World Datasets General Social Survey Data. We applied our method to a dataset from the General Social Survey, a sociological data repository available online https://gss.norc.org/  us/en/gss.html. The dataset contains six observed variables: father's occupation, son's income, father's education, son's occupation, son's education, and number of siblings, with a sample size of 1380. We use the hypothesized model from Duncan et al. (1972) as a baseline. Their graph, determined using domain knowledge and temporal orders, is shown in Figure 5. Father's Education (V3) Father's Occupation (V1) Number of Siblings (V6) Son's Education (V5) Son's Occupation (V4) Son's Income (V2)
Figure 5. Status attainment model based on domain knowledge (Duncan et al., 1972;Shimizu et al., 2011). A directed edge between two vertices in the figure means that there could be a directed edge between the two variables. A bi-directed edge between two vertices means that the relation is not modeled.
this section cite: ['b7', 'b25', 'b11', 'b9', 'b43']

Section: Results:
We selected four pairs of variables as target pairs, with direct relationships and directed paths, as well as pairs with no direct relationships or directed paths.
• We first selected the father's occupation (X) and the son's education (Y ) as the target variable pair, connected by a direct edge. Our method identifies the father's occupation as an invariant ancestor of the son's education.
• Next, we selected father's occupation (X) and son's income (Y ), as well as father's education (X) and son's income (Y ) as the target variable pairs. In both cases, X and Y are connected by directed paths. Our method identifies both father's occupation and father's education as invariant ancestors of the son's income.
• Finally, for son's income (X) and number of siblings (Y ), which are neither connected by a direct edge nor a directed path from X to Y . Our method finds the son's income as an invariant non-ancestor of the number of siblings.
These findings align with the domain knowledge in Duncan et al. (1972).
this section cite: ['b9']

Section: Gene Expression Data.
We applied our proposed method to the gene expression dataset from Wille et al. (2004), which contains measurements from Arabidopsis thaliana under 118 different experimental conditions, including variations in light and darkness and exposure to growth hormones. The dataset includes expression data for 33 genes. We here adopt the model presented in Wille et al. (2004) (see Figure 3 of Wille et al. (2004)) as a baseline.
Results: We selected six pairs of genes as target pairs, with direct relationships and directed paths, as well as pairs with no direct relationships or directed paths.
• We first selected DXR (X) and MCT (Y ), as well as HMGS (X) and HMGR1 (Y ), as the target pairs, both of which are connected by a direct edge. Our method identifies DXR as an invariant ancestor of MCT, and similarly, HMGS as an invariant ancestor of HMGR1.
• Next, we selected AACT1 (X) and FPPS1 (Y ), as well as DXPS3 (X) and CMK (Y ), where each pair is connected by a directed path from X to Y . Our method identifies AACT1 as an invariant ancestor of FPPS1, and likewise, DXPS3 as an invariant ancestor of CMK.
• Finally, we considered PPDS1 (X) and DXPS1 (Y ), as well as DXPS1 (X) and DXPS3 (Y ), neither of which is connected by a direct edge or a directed path from X to Y . Our method finds that PPDS1 is an invariant nonancestor of DXPS1, and similarly, DXPS1 is an invariant non-ancestor of DXPS3.
These findings align with Wille et al. (2004).
this section cite: ['b53', 'b53', 'b53', 'b53']

Section: Conclusion
We addressed the problem of locally learning causal relations from observational data without assuming causal sufficiency. First, we provided sufficient and necessary local characterizations for identifying invariant ancestors, invariant non-ancestors, and possible ancestors. Then, we introduced LocICR, a novel algorithm for local causal discovery. We proved that LocICR is complete, matching the accuracy of existing methods. Experiments demonstrate its effectiveness, efficiency, and robustness in handling latent variables in complex environments. Future work could explore incorporating background knowledge, such as data generation mechanisms (Kaltenpoth & Vreeken, 2023) or expert insights (Wang et al., 2023b), to enhance causal discovery within local structures in LocICR.
this section cite: ['b19']

Section: References
Ref_id:b0 Title: Recursive causal structure learning in the presence of latent variables and selection bias Year: (2021)
Ref_id:b1 Title: Orientation rules for constructing markov equivalence classes of maximal ancestral graphs Year: (2005)
Ref_id:b2 Title: Local causal and markov blanket induction for causal discovery and feature selection for classification part i: algorithms and empirical evaluation Year: (2010)
Ref_id:b3 Title: Differentiable causal discovery under unmeasured confounding Year: (2021)
Ref_id:b4 Title: Greedy equivalence search in the presence of latent confounders Year: (2022)
Ref_id:b5 Title: A logical characterization of constraint-based causal discovery Year: (2011)
Ref_id:b6 Title: Learning sparse causal models is not np-hard Year: (2013)
Ref_id:b7 Title: Learning high-dimensional directed acyclic graphs with latent and selection variables Year: (2012)
Ref_id:b8 Title: Order-independent constraint-based causal structure learning Year: (2014)
Ref_id:b9 Title:  Year: (1972)
Ref_id:b10 Title: Ida with background knowledge Year: (2020)
Ref_id:b11 Title: A local method for identifying causal relations under markov equivalence Year: (2022)
Ref_id:b12 Title: Local causal discovery of direct causes and effects Year: (2015)
Ref_id:b13 Title: Efficient markov blanket discovery and its application Year: (2016)
Ref_id:b14 Title: Minimal enumeration of all possible total effects in a markov equivalence class Year: (2021)
Ref_id:b15 Title: An introduction to variable and feature selection Year: (2003-03)
Ref_id:b16 Title:  Year: (2010)
Ref_id:b17 Title: Elements of Causal Inference Year: (2017)
Ref_id:b18 Title: Causal inference using graphical models with the r package pcalg Year: (2012)
Ref_id:b19 Title: Nonlinear causal discovery with latent confounders Year: (2023)
Ref_id:b20 Title: Gradient-based local causal structure learning Year: (2023)
Ref_id:b21 Title: Collapsible ida: Collapsing parental sets for locally estimating possible causal effects Year: (2020)
Ref_id:b22 Title: Local causal network learning for finding pairs of total and direct effects Year: (2020)
Ref_id:b23 Title: A generalized backdoor criterion Year: (2015)
Ref_id:b24 Title: Estimating high-dimensional intervention effects from observational data Year: (2009)
Ref_id:b25 Title: Estimating causal effects with ancestral graph markov models Year: (2016)
Ref_id:b26 Title: Constraint-based causal discovery using partial ancestral graphs in the presence of cycles Year: (2020)
Ref_id:b27 Title: Estimating the effect of joint interventions from observational data in sparse high-dimensional settings Year: (2017)
Ref_id:b28 Title: A hybrid causal search algorithm for latent variable models Year: (2016)
Ref_id:b29 Title: Probabilistic reasoning in intelligent systems: networks of plausible inference Year: (1988)
Ref_id:b30 Title: Causality: Models, Reasoning, and Inference Year: (2000)
Ref_id:b31 Title:  Year: (2009)
Ref_id:b32 Title: Theoretical impediments to machine learning with seven sparks from the causal revolution Year: (2018)
Ref_id:b33 Title: Finding latent causes in causal networks: an efficient approach based on markov blankets Year: (2008)
Ref_id:b34 Title: Using markov blankets for causal structure learning Year: (2008)
Ref_id:b35 Title: A bayesian approach for estimating causal effects from observational data Year: (2020)
Ref_id:b36 Title: Comparison of strategies for scalable causal discovery of latent variable models from mixed data. International journal of data science and analytics Year: (2018)
Ref_id:b37 Title: Markov properties for acyclic directed mixed graphs Year: (2003)
Ref_id:b38 Title: Ancestral graph markov models Year: (2002)
Ref_id:b39 Title: Iterative causal discovery in the possible presence of latent confounders and selection bias Year: (2021)
Ref_id:b40 Title: Marginal causal consistency in constraintbased causal learning Year: (2016)
Ref_id:b41 Title: Estimating feedforward and feedback effective connections from fmri time series: Assessments of statistical methods Year: (2019)
Ref_id:b42 Title: Causality for machine learning Year: (2022)
Ref_id:b43 Title: Directlingam: A direct method for learning a linear non-gaussian structural equation model Year: (2011-04)
Ref_id:b44 Title: Network modelling methods for fmri Year: (2011)
Ref_id:b45 Title: Bayesian analysis in expert systems Year: (1993)
Ref_id:b46 Title: An algorithm for fast recovery of sparse causal graphs Year: (1991)
Ref_id:b47 Title: A polynomial time algorithm for determining dag equivalence in the presence of latent variables and selection bias Year: (1996)
Ref_id:b48 Title: Causation, Prediction, and Search Year: (2000)
Ref_id:b49 Title: On scoring maximal ancestral graphs with the maxmin hill climbing algorithm Year: (2018)
Ref_id:b50 Title: Discovering and orienting the edges connected to a target variable in a dag via a sequential local learning approach Year: (2014)
Ref_id:b51 Title: Estimating possible causal effects with latent variables via adjustment Year: (2023)
Ref_id:b52 Title: Sound and complete causal identification with latent variables given local background knowledge Year: (2023)
Ref_id:b53 Title: Sparse graphical gaussian modeling of the isoprenoid gene network in arabidopsis thaliana Year: (2004)
Ref_id:b54 Title: Local causal structure learning in the presence of latent variables Year: (2024)
Ref_id:b55 Title: Towards efficient local causal structure learning Year: (2021)
Ref_id:b56 Title: Partial orientation and local structural learning of causal networks for prediction Year: (2008)
Ref_id:b57 Title: Causal inference and reasoning in causally insufficient systems Year: (2006)
Ref_id:b58 Title: On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias Year: (2008)
Ref_id:b59 Title: A transformational characterization of markov equivalence for directed acyclic graphs with latent variables Year: (2005)
Ref_id:b60 Title: On the markov equivalence of maximal ancestral graphs Year: (2005)
Ref_id:b61 Title: Local causal discovery with background knowledge Year: (2024)
Ref_id:b62 Title: Discover local causal network around a target to a given depth Year: (2010)
Ref_id:b63 Title: Counterfactual fairness with partially known causal graph Year: (2022)
