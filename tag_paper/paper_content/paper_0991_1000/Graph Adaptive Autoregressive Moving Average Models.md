Title: Graph Adaptive Autoregressive Moving Average Models
Abstract: Graph State Space Models (SSMs) have recently been introduced to enhance Graph Neural Networks (GNNs) in modeling long-range interactions. Despite their success, existing methods either compromise on permutation equivariance or limit their focus to pairwise interactions rather than sequences. Building on the connection between Autoregressive Moving Average (ARMA) and SSM, in this paper, we introduce GRAMA, a Graph Adaptive method based on a learnable ARMA framework that addresses these limitations. By transforming from static to sequential graph data, GRAMA leverages the strengths of the ARMA framework, while preserving permutation equivariance. Moreover, GRAMA incorporates a selective attention mechanism for dynamic learning of ARMA coefficients, enabling efficient and flexible long-range information propagation. We also establish theoretical connections between GRAMA and Selective SSMs, providing insights into its ability to capture long-range dependencies. Experiments on 26 synthetic and real-world datasets demonstrate that GRAMA consistently outperforms backbone models and performs competitively with state-of-the-art methods.

Section: Introduction
Graph learning (Scarselli et al., 2008;Micheli, 2009;Bruna et al., 2013;Defferrard et al., 2016;Kipf & Welling, 2016;Veličković et al., 2018) has become crucial in handling graph-structured data across various domains (Gravina & Bacciu, 2024), such as social networks (Kipf & Welling, Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). 2016; Hamilton et al., 2017), molecular interactions (Xu et al., 2019;Bouritsas et al., 2022), and more (Khemani et al., 2024). The most popular framework of neural graph learning is that of Message Passing Neural Networks (MPNNs). Some prominent examples are GCN (Kipf & Welling, 2016), GAT (Veličković et al., 2018), GIN (Xu et al., 2019), and GraphConv (Morris et al., 2019). However, many MPNNs suffer from a critical shortcoming of oversquashing (Alon & Yahav, 2021;Di Giovanni et al., 2023), that hinders their ability to model long-range interactions. To address this limitation, several proposals were made, from graph rewiring (Topping et al., 2022;Di Giovanni et al., 2023;Karhadkar et al., 2023), to multi-hop MPNNs (Gutteridge et al., 2023), weight space regularization (Gravina et al., 2023;2025), as well as Graph Transformers (GTs) (Yun et al., 2019;Dwivedi & Bresson, 2022;Kreuzer et al., 2021b). Specifically, GTs became popular because of their theoretical and often practical ability to capture long-range node interactions through the attention mechanism. However, the quadratic computational cost of full attention limits their scalability, and in some cases, they were found to underperform on long-range benchmarks when compared to standard MPNNs (Tönshoff et al., 2023).
At the same time, State Space Models (SSMs), such as S4 (Gu et al., 2021c) and Mamba (Gu et al., 2023), have emerged as promising, linear-complexity alternatives to Transformers. SSMs leverage a recurrent and convolutional structure to efficiently capture long-range dependencies while maintaining linear time complexity (Nguyen et al., 2023). Contemporary models like Mamba develop selective filters that prioritize context through input-dependent selection, offering compelling advantages in processing long sequences with reduced computational demands compared to transformers (Gu et al., 2023). Despite these benefits, adapting SSMs to the non-sequential structure of graphs remains a significant challenge. Perhaps the biggest challenge in applying SSMs to graph learning tasks, lies in the fundamental question of "how to transform a graph into a sequence?". To this end, several graph SSM approaches were proposed, from a graph-to-sequence heuristic in Wang et al. (2024a), to studying the relationship between SSMs and spectral GNNs by pairwise interactions (Huang et al., 2024b), as well as sample-based random walk sequencing of the graph (Behrouz & Hashemi, 2024). More broadly, this question has also been studied in other, non-SSM related works, discussed in Appendix A. However, as we discuss later, some of them lose the permutation-equivariance property desired in GNNs, while others do not take advantage of the sequence processing ability of SSMs. These limitations hinder their ability to fully leverage sequence-processing capabilities, especially for addressing oversquashing in GNNs. To resolve these issues, we propose, instead, a complementary approach -transforming a static input graph into a sequence of graphs, combined with an adaptive neural autoregressive moving-average (ARMA) mechanism, called GRAMA. We show that GRAMA is theoretically equivalent to an SSM on graphs. Our GRAMA allows us to enjoy the benefits of sequential processing mechanisms like SSMs, coupled with any GNN backbone, from MPNNs to graph transformers, while maintaining backbone properties, such as permutation-equivariance.
this section cite: ['b120', 'b98', 'b19', 'b31', 'b85', 'b130', 'b68', 'b139', 'b15', 'b84', 'b85', 'b130', 'b139', 'b99', 'b2', 'b33', 'b33', 'b82', 'b65', 'b143', 'b36', 'b127', 'b62', 'b103', 'b62', 'b9']

Section: Main Contributions.
Our Adaptive Graph Autoregressive Moving Average (GRAMA) model offers several advancements in the conjoining of dynamical systems theory into GNNs:
• Principled Integration of SSMs in GNNs. We enable the use of sequence-based models (like ARMA) coupled with virtually any GNN backbone, by transforming graph inputs into temporal sequences without sacrificing permutation invariance.
• Theoretical Understanding of the coupling of SSMs and GNNs. We demonstrate that augmenting GNNs with ARMA via our GRAMA has an equivalent SSM model.
• Mitigation of the oversquashing problem. We provide the theoretical foundation that our GRAMA effectively addresses the oversquashing phenomenon in GNNs and improves the long-range interaction modeling capabilities.
• Strong Practical Performance. We demonstrate our GRAMA on three popular backbones (GCN (Kipf & Welling, 2016), GatedGCN (Bresson & Laurent, 2018), and GPS (Rampášek et al., 2022)) and show the compelling performance by GRAMA on 26 synthetic and real-world datasets.
this section cite: ['b85', 'b17', 'b114']

Section: Related Work
We now provide an overview and discussion of related topics and works to our GRAMA. In Appendix A, we discuss additional related works.
this section cite: []

Section: Long-Range Interactions on Graphs.
GNNs rely on message-passing mechanisms to aggregate information from neighboring nodes, which limits their ability to capture long-range dependencies, as highlighted by Alon & Yahav (2021); Di Giovanni et al. (2023). Models like GCN (Kipf & Welling, 2016), GraphSAGE (Hamilton et al., 2017), and GIN (Xu et al., 2019) face challenges such as oversmoothing (Nt & Maehara, 2019;Oono & Suzuki, 2020;Cai & Wang, 2020;Rusch et al., 2023), over-squashing (Alon & Yahav, 2021;Topping et al., 2022;Di Giovanni et al., 2023) and more generally vanishing gradients (Arroyo et al., 2025), which hinder long-range information propagation-critical in applications like bioinformatics (Baek et al., 2021;Dwivedi et al., 2022b) and heterophilic settings (Luan et al., 2024;Wang et al., 2024b). To address these limitations, various methods have emerged, including graph rewiring (Topping et al., 2022;Karhadkar et al., 2023), adaptive message passing (Errica et al., 2024;Finkelshtein et al., 2024), weight space regularization (Gravina et al., 2023;2024b;2025), exploitation of port-Hamiltonian dynamics (Heilig et al., 2025), and Graph Transformers (GTs). GTs, which capture both local and global interactions, have been particularly promising, as demonstrated by models like SAN (Kreuzer et al., 2021c), Graphormer (Ying & Leskovec, 2021), and GPS (Rampášek et al., 2022). These models often incorporate positional encodings, such as Laplacian eigenvectors (Dwivedi et al., 2021) or random-walk structural encodings (RWSE) (Dwivedi et al., 2022a), to encode graph structure. However, the quadratic complexity of full attention in GTs presents scalability challenges. Recent innovations like sparse attention mechanisms (Zaheer et al., 2020;Choromanski et al., 2020), Exphormer (Shirzad et al., 2023), and linear graph transformers (Wu et al., 2023;Deng et al., 2024) address these bottlenecks, improving efficiency and scalability for long-range propagation.
State Space Models (SSMs). SSMs, traditionally used for time series analysis (Hamilton, 1994b;Aoki, 2013), process sequences through latent states. However, classic SSMs struggle with long-range dependencies and lack parallelism, limiting their computational efficiency. Recent advances, such as the Structured State Space Sequence model (S4) (Gu et al., 2021c;Fu et al., 2023), mitigate these issues by employing linear recurrence as a structured convolutional kernel, enabling parallelization on GPUs. Despite this, simple SSMs still underperform compared to attention models in natural language tasks. Mamba (Gu et al., 2023) improves the ability of SSMs to capture longrange dependencies by selectively controlling which sequence parts influence model states. Mamba has shown promising results, outperforming Transformers in several benchmarks (Gu et al., 2023;Liu et al., 2024) while being more computationally efficient. The combination of SSMs with graph models presents challenges, particularly in transforming the articulated connectivity of graphs into sequences. For instance, Graph-Mamba (Wang et al., 2024a) orders nodes by degree, but this heuristic approach sacrifices permutation-equivariance, a desirable property in GNNs. Similarly, Behrouz & Hashemi (2024) propose generating sequences via random walks, which improves performance but also sacrifices permutation-equivariance while adding non-determinism to the model. Also, turning a graph into a sequence based on a policy, such as sorting nodes by degree, limits direct use of the input graph, as multiple graphs can share the same node degrees and thus be indistinguishable. Huang et al. (2024b) explored links between spectral GNNs and graph SSMs, focusing on pairwise interactions; however, this design choice may not fully exploit the sequence-handling capacity of SSMs and may reach the state of oversquahsing earlier because of the use of powers of the adjacency matrix (Di Giovanni et al., 2023).
In this work, we harness the potential of SSMs by adopting a structure inspired by the connection between SSMs and ARMA models. By transforming static graphs into sequences, GRAMA maintains permutation-equivariance, a desired property in GNNs (Bronstein et al., 2021), also useful for long-propagation (Pan & Kondor, 2022;Schatzki et al., 2024), while enabling effective learnable and selective long-range propagation.
this section cite: ['b2', 'b85', 'b68', 'b139', 'b105', 'b106', 'b20', 'b117', 'b2', 'b33', 'b5', 'b6', 'b40', 'b93', 'b82', 'b49', 'b50', 'b70', 'b141', 'b114', 'b38', 'b144', 'b27', 'b123', 'b137', 'b32', 'b3', 'b62', 'b62', 'b62', 'b9', 'b33', 'b18', 'b109', 'b121']

Section: Autoregressive Moving Average Models (ARMA).
ARMA models, introduced by Whittle (1951), combine an autoregressive (AR) component, modeling dependencies on previous time steps, with a moving average (MA) component, considering residuals. Widely applied in stationary time series analysis (Box et al., 1970), ARMA models are equivalent to state space models (SSMs) (Hamilton, 1994a). An ARMA(p, q) model considers previous p states and q residuals δ(•), and is governed by the following equation:
f (t) = p i=1 ϕ i f (t -i) + q j=1 θ j δ(t -j) + δ(t), (1
)
where {ϕ i } p i=1 , {θ i } q j=1 are the autoregressive and moving average coefficients, respectively.
Although ARMA models are traditionally used for processing sequences, they have also been studied for classical graph filtering (Isufi et al., 2016) and more recently formulated as an MPNN in Bianchi et al. (2021). The Graph ARMA model (Bianchi et al., 2021) introduced a learnable ARMA version for GCNs, using recursive 1-hop filters to create a structure resembling ARMA methods. In this paper, we introduce GRAMA, a method that leverages neural ARMA models by transforming a static input graph into a graph sequence. Different than Bianchi et al. (2021), which uses the static input graph and formulates a recursive ARMA model through a spectral convolution perspective, our GRAMA incorporates a selective and graph adaptive mechanism that learns ARMA coefficients along the graph sequence. This dynamic adjustment of coefficients directly addresses oversquashing by preserving long-range dependencies and enabling adaptive control over feature propagation. Additionally, Bianchi et al. (2021) uses an ARMA(1, 1) model with non-linearities between steps, hin-dering its direct conversion into an SSM, while we show that our GRAMA has an equivalent SSM, providing deeper theoretical understandings.
this section cite: ['b136', 'b16', 'b80', 'b13', 'b13', 'b13', 'b13']

Section: GRAMA
Although a graph is a static structure, the process of message passing introduces a dynamic element. In message passing, information is propagated through the graph, allowing nodes to update their states based on the states of their neighbors. This dynamic behavior can be viewed through the lens of dynamical systems, where the state of each node evolves according to certain aggregating rules, as discussed in Section 2. This perspective is instrumental in Recurrent Neural Networks (RNNs), which are designed to handle sequential data and capture temporal dependencies. By treating the message-passing process as a dynamical system, we can leverage the strengths of RNNs to model the evolution of node states over time. The model we propose, GRAMA, takes inspiration from the architectural structure of the latest generation of sequential models, like S4 (Gu et al., 2021a), Mamba (Gu et al., 2023), LRU (Orvieto et al., 2023b), and xLSTM (Beck et al., 2024). To import these powerful sequential models to graph learning, we first translate static input graphs into sequences of graphs. Then, the GRAMA block transforms such graph sequence into another graph sequence, while considering the structure of the graph. Each GRAMA block is linear, and non-linear activations are applied between GRAMA blocks to increase the flexibility of the overall model. Below, we discuss in detail the different aspects of our GRAMA -from its initialization to the graph sequence processing blueprint by ARMA, to the learning of ARMA coefficients in a graph adaptive manner. The overall design of GRAMA is illustrated in Figure 1.
Notations. We denote a graph by G = (V, E), with |V | = n nodes and |E| = m edges. A node v is associated with input node features f v ∈ R c . The node features are then denoted by f ∈ R n×c .
this section cite: ['b62', 'b8']

Section: Initialization.
Processing information with ARMA or SSM frameworks, by design, requires a sequence. As discussed in Section 2, previous studies on graph SSMs have chosen to transform the graph into a sequence by means of heuristic node ordering, random walk sampling, or by considering pairwise interactions (edges) as sequences of length 2. While these choices are valid, and show strong performance in practice, they also introduce challenges compared to common graph learning approaches, or may not fully utilize the underlying sequence processing framework. Specifically, the first two approaches (node ordering and walk sampling) do not maintain the permutation equivariance desired in GNNs, and the third (pairwise interactions) considers only very short sequences, while one key benefit of the ARMA and SSM frameworks is their ability to cap-Figure 1: An illustration of the GRAMA framework with L recurrences. We embed a static input graph into a sequence of graphs. This sequence is the input for the first GRAMA block. Here, a GRAMA block is composed of a neural ARMA(L, L) layer with adaptive autoregressive ϕ = {ϕ i } L i=1 and moving average θ = {θ j } L j=1 coefficients, that weigh previous states {f l } L-1 l=0 and residuals {δ l } L-1 l=0 , and a graph-informed residual update via a GNN backbone. A GRAMA block yields two updated state and residual sequences F (s) , ∆ (s) for the s = 1, . . . , S block. Each GRAMA block is a linear system, and non-linearities are applied between GRAMA blocks, as in Equation ( 8).
ture long-range dependencies in long sequences (Gu et al., 2021b). To address these challenges, we propose to transform a static graph into a sequence of graphs, such that each node is equipped with a sequence of input node feature vectors rather than a single input node feature vector. By following this idea, we can employ sequence processing frameworks such as ARMA on data beyond pairwise interactions, while maintaining permutation-equivariance, as we discuss later. Specifically, we first stack the input node features f for L times, where L > 0 is a hyperparameter that determines the length of the sequence to process, followed by the application of a set of MLPs, {g k } L-1 k=0 , one for each k = 0, . . . , L -1, that embed the original c node features into d channels:
F (0) = f (0) , . . . , f (L-1) = g 0 (f ), . . . , g L-1 (f ) , (2)
where F (0) ∈ R L×n×d . We refer to the sequence encoded by F (0) as the initial input sequence, and to work with an ARMA model, we also define the residuals sequence as follows:
∆ (0) = δ (0) , . . . , δ (L-1) , ∆ (0) ∈ R L×n×d , (3)
where δ (ℓ) = f (ℓ+1) -f (ℓ) for ℓ = 0, . . . , L -2. Note that by subtracting subsequent elements in the input sequence F (0) , we are left with L -1 elements. Therefore, we choose the last residual term in ∆ (0) (that is δ (L-1) ) to be a matrix of zeros at the initialization step. We note that via this approach, we can perform sequence modeling using ARMA on the sequence dimension (L) while retaining the ability to use any desired backbone GNN to exchange information between nodes, as shown in Section 3.1, thus rendering our GRAMA a drop-in mechanism.
this section cite: []

Section: Graph Neural ARMA Autoregressive (AR) Layers.
An AR p captures the relationship between current node features and their p > 0 previous historical values, through the learnable coefficients {ϕ i } p i=1 discussed in Section 3.2. Formally, given a sequence of node features of length L f (ℓ) , . . . , f (ℓ+L-1) , assuming p ≤ L, the node features at step ℓ + L read: (ℓ+L-i) .
f (ℓ+L) ARp = AR p (f (ℓ) , . . . , f (ℓ+L-1) ) = p i=1 ϕ i f
(4) Moving Average (MA) Layers. Given a residuals sequence δ (ℓ) , . . . , δ (ℓ+L-1) , a MA q layer with {θ j } q j=1 learnable coefficients, captures the dependency of the latest 0 < q ≤ L residuals:
f (ℓ+L) MAq = MA q (δ (ℓ) , . . . , δ (ℓ+L-1) ) = q j=1
θ j δ (ℓ+L-j) .
(5) GRAMA Recurrence. Combining AR p and MA q layers, leads to the ARMA(p, q) recurrence:
f (ℓ+L) = f (ℓ+L) ARp + f (ℓ+L) MAq + δ (ℓ+L) ,(6)
where δ (ℓ+L) is the residual of the last step, which is given by a GNN backbone that is optimized jointly with the ARMA coefficients, that is, δ (ℓ+L) = GNN(f (ℓ+L-1) ; G).
Here, we apply the GNN backbone without non-linearity so that each recurrence step within a GRAMA block is a linear function. In particular, note that the GNN can be any graph neural network, because at each recurrence, GRAMA processes a sequence of graphs by updating each node feature based on its sequence via the terms f (ℓ+L)
ARp , f(ℓ+L)
MAq , coupled a with a GNN in the term δ (ℓ+L) . Moreover, the structure of the terms f (ℓ+L) ARp , f (ℓ+L) MAq includes multiple residual connections, which can implement standard skip-connections, retaining the expressiveness of the backbone GNN. Section 5 showcases GRAMA with various GNN backbones, from MPNNs to graph transformers.
GRAMA Block. Equation ( 6) describes a single recurrence step within a GRAMA block. Similar to other recurrent mechanisms, we apply R recurrences, where R > 1 is a hyperparameter. Thus, given the initial states F (0) and residuals ∆ (0) , after R recurrences according to Equation (6), we obtain updated states f (L) , . . . , f (L+R-1) and residuals δ (L) , . . . , δ (L+R-1) sequences, followed by an elementwise application of non-linearity σ:
F (1) = σ(f (L) ), . . . , σ(f (L+R-1) ) , ∆ (1) = σ(δ (L) ), . . . , σ(δ (L+R-1) ) .(7)
In practice, as discussed in Appendix D.6, R is chosen such that p = q = R = L, and the obtained updated sequences are
F (1) = σ(f (L) ), . . . , σ(f (2L-1) ) , ∆ (1) = σ(δ (L) ), . . . , σ(δ (2L-1) ) .
Deep GRAMA. In Equation (7), we describe the action of a single, first GRAMA block. Overall, each block performs R recurrence steps. As such, the first GRAMA block yields R new states and residuals encoded in F (1) and ∆ (1) , respectively, that can then be processed by subsequent GRAMA blocks. That is, we can stack S ≥ 1 GRAMA blocks, each block with its own parameters, forming a deep GRAMA network, where the updated sequences at the s-th GRAMA block are:
F (s) = σ(f (L+(s-1)R ), . . . , σ(f (L+sR-1) ) , ∆ (s) = σ(δ (L+(s-1)R) ), . . . , σ(δ (L+sR-1) ) ,(8)
for s = 1, . . . , S. Note that the depth of a GRAMA network is therefore equivalent to the number of systems S to be learned, multiplied by the number of recurrent steps R. The outputs of the GRAMA network are then the final state and residual sequences F (S) , ∆ (S) . We illustrate this process in Figure 1. Because in our experiments we are interested in static graph learning problems, we feed the latest state matrix within the sequence F (S) to a readout layer to obtain the final prediction, as elaborated in Appendix C.2. The additional processing in GRAMA introduces some computational overhead, as detailed in Section 3.3. However, this cost remains reasonable compared to other methods and yields significant performance improvements, as detailed in Section 5.
this section cite: []

Section: Learning Adaptive Graph ARMA Coefficients
We now introduce our graph adaptive approach for learning the ARMA coefficients, which is a key component in our approach to allow a flexible and selective GRAMA.
Naive ARMA Learning. The most straightforward way to learn the AR and MA coefficients, {ϕ i } p i=1 and {θ j } q j=1 , is to consider them as parameters of the neural network and learn them via gradient descent. However, this yields coefficients that are identical for all inputs, thereby not adaptive. This approach is directly linked to non-selective weights in SSM models (Gu et al., 2021c), which were shown to be less effective compared to selective coefficients (Gu et al., 2023). Selective ARMA Learning. To allow selective ARMA coefficient learning similarly to Mamba (Gu et al., 2023), we use an attention mechanism (Vaswani et al., 2017) applied over the state and residual sequences F (s) , ∆ (s) at each GRAMA block s = 1, . . . , S. The rationale behind this construction is that an attention layer assigns scores between elements within the sequence. Formally, we obtain two scores matrices A F (s) , A ∆ (s) ∈ [0, 1] L×L . The last row in each matrix represents the predicted coefficients for our GRAMA, {ϕ i } p i=1 and {θ j } q j=1 , respectively. However, the SoftMax normalization in standard attention layers yields non-negative pairwise values, which is not consistent with the usual choice of ARMA coefficients. Therefore, we follow the self-attention implementation (Vaswani et al., 2017) up to the SoftMax step, and we normalize the scores to be in [-1, 1] while complying with a sum-to-one constraint. We note that, this procedure facilitates learning stability, such that ARMA coefficients do not explode or vanish, and its design is guided by the insights from Theorems 4.3 and 4.4. We also note that this overall construction yields twofold adaptivity in the predicted ARMA coefficients: First, the attention mechanism allows selectivity with respect to inputs, which are the sequences F (s) , ∆ (s) . Second, because these sequences are coupled with a GNN backbone, as shown in Equation (6), it implies that the input node features and the graph structure influence the coefficients. We provide further implementation details in Appendix C, and a comparison between naive and selective ARMA learning in Appendix E.4.
this section cite: ['b62', 'b62', 'b129', 'b129']

Section: Time and Space Complexity of GRAMA
We discuss the theoretical complexity of our method, showing its reduced computational complexity compared with other models, e.g., transformers. We note that, overall, our GRAMA retains the asymptotic complexity of the underlying GNN backbone, assuming that the number of recurrences is a constant, or smaller than the number of nodes and edges within the graph. We report empirical runtimes in Appendix E.3, demonstrating that GRAMA offers better scalability and performance compared to other approaches such as graph transformers.
this section cite: []

Section: Time Complexity.
We analyze the case where we use an MPNN that is linear in graph size (nodes and edges) such as GCN is used within GRAMA. Our GRAMA is comprised of L initial MLPs, S GRAMA blocks, each with L recurrent steps, and a final readout layer. Note that L is the sequence length, which is a hyperparameter and does not exceed the value of 50 in our experiments. The initial MLPs operate on the input f ∈ R n×c and embed them to a hidden dimension d. Therefore, their time complexity is O(L • n • c • d). Each GRAMA block is comprised of two attention layers -one for the pooled states sequence and the other for the residual pooled sequence, and a GNN layer for predicting the current step residual, which operates on graph node features. The attention layer time complexity is O(L 2 d 2 ) because the pooled (across the graph nodes) sequence is of shape L × d, and the GNN layer complexity is O(n + m), where n is the number of nodes and m is the number of edges in the graph. Note that usually m ≫ n, so the GNN complexity can be rewritten as O(m). We note that, in the case of a graph-transformer based GNN, like GPS, we have that m = n 2 . In the following, we consider the more general case. In total, we have S GRAMA blocks, where S is a hyperparameter, and is typically small, up to 4. The final readout layer is a standard MLP and, therefore, has the time complexity of O(n•d•o) for node-wise tasks, and O(d•o) for graph-level tasks. Therefore, the overall time complexity (including initial MLPs and readout) of our GRAMA is
O L • n • c • d + SL • (n + m + L 2 • d 2 ) + n • d • o .
this section cite: []

Section: Space Complexity.
We analyze the case where linear in graph size (nodes and edges) complexity MPNN (such as GCN) is used within GRAMA. The space complexity of the initial MLPs is O(L • c • d). The space complexity for each GRAMA block is O(d 2 ) for the GNN layer, and similarly O(d 2 ) for the two attention layers. Overall, we have S such blocks. The readout layer space complexity is O(d • o). Thus, the overall space complexity (including initial MLPs and readout
) of GRAMA is O(L • c • d + S • d 2 + d • o).
this section cite: []

Section: Theoretical Properties of GRAMA
We now formally cast common knowledge formulated in the context of RNNs, control theory, and SSMs (Yu et al., 2019;Slotine et al., 1991;Khalil, 2002;Aoki, 2013) to the realm of GNNs, aiming to adapt foundational results from non-graph settings of SSMs and ARMA models into a graph-learning framework. We discuss the main theoretical properties of our GRAMA: (i) its representation as an SSM model, (ii) its stability, and (iii) its ability to model longrange interactions in graphs. All the proofs are provided in Appendix B. Connection to SSM. As discussed in Section 3, each GRAMA block is fundamentally an ARMA model. In Theorem 4.1, we formalize the equivalence between ARMA models and linear SSMs. This allows us to interpret our GRAMA model as a stack of graph-informed SSMs through the backbone GNN encoded in Equation ( 6).
Theorem 4.1 (Equivalence between ARMA models and State Space Models). For every ARMA model, there exists an equivalent State Space Model (SSM) representation, and conversely, for every linear SSM, there exists an equivalent ARMA model representation.
Stability. Representing an ARMA system as an SSM involves the description of a linear recurrence equation as
f (L) = p i=1 ϕ i f (L-i) + q j=1 θ j δ (L-j) + δ (L) , or, alter- natively, in matrix form as X (L) = AX (L-1) + Bδ (L) , with X (L-1) = f (L-1) , . . . , f (0) , δ (L-1) , . . . , δ (0) , see Appendix B for more details. 1
In the SSM literature, the A matrix is called the state matrix. The state matrix corresponding to a GRAMA block is entirely determined by the set of autoregressive and moving average coefficients. Thus, each GRAMA block is characterized by an adaptive state matrix, which is especially important since it directly governs the evolution of the node features f . In particular, the stability of this evolution can be established by analyzing the powers of the state matrix, as widely studied in the context of RNN and SSM theory (Pascanu, 2013;Gu et al., 2021b). Hence, the stability of a GRAMA block can be characterized by the following Lemma 4.2.
Lemma 4.2 (Stability of GRAMA). The linear SSM corresponding to a GRAMA block with autoregressive coefficients {ϕ i } p i=1 is stable if and only if the spectral radius of its state matrix is less than (or at most equal to) 1. In particular, this happens if and only if the polynomial P (λ) = λ pp j=1 ϕ j λ p-j has all its roots inside (or at most on) the unit circle.
We now give a sufficient condition for the stability of the SSM corresponding to a GRAMA block.
Theorem 4.3 (Sufficient condition for GRAMA stability). If p j=1 |ϕ j | ≤ 1, then the GRAMA block with autoregressive coefficients {ϕ i } p i=1 corresponds to a stable linear SSM.
this section cite: ['b142', 'b124', 'b83', 'b3', 'b110']

Section: Long-Range Interactions.
A key distinction between standard MPNNs and our GRAMA lies in its neural selective sequential mechanism, which uses learned ARMA coefficients to operate across two domains: the spatial graph domain via a GNN backbone, and the sequence domain via the ARMA mechanism, enabling selective state updates. Remarkably, the state matrices of each GRAMA block play a significant role in the propagation of the information from the first sequence of node features, F (0) = f (0) , . . . , f (L-1) , to the last sequence of node features after S GRAMA blocks, F (S) = f (LS) , . . . , f (L(S+1)-1) , especially for large L and S. In fact, if the entries of the k-th power of the state matrix of a GRAMA block vanish, then for a stable GRAMA it is impossible to model long-range dependencies of k hops, in the sequence, as we show in Lemma B.1. This fact relates to a broadly acknowledged problem in the RNN literature, the vanishing gradient issue (Hochreiter et al., 2001;Bengio et al., 1994;Orvieto et al., 2023a): the entries of the powers of a matrix with a spectral radius less than 1 can quickly vanish, making it challenging for gradient-based algorithms to effectively long-range patterns. Therefore, to bias the long-term propagation of the information of a GRAMA block, we can initialize the state matrix to have its eigenvalues close enough to the unitary circle, following the footsteps of recent RNN methodologies (Orvieto et al., 2023b;Arjovsky et al., 2016;De et al., 2024). In fact, the closer the eigenvalues are to the unitary circle, the slower the powers of the state matrix vanish (Horn & Johnson, 2012). The following Theorem 4.4 provides a criterion to control the long-range interaction of GRAMA.
this section cite: ['b72', 'b10', 'b4', 'b29', 'b73']

Section: Theorem 4.4 (GRAMA allows long-range interactions).
Let us be given a GRAMA block with autoregressive coefficients {ϕ i } p i=1 . Assume the roots of the polynomial P (λ) = λ pp j=1 ϕ j λ p-j are all inside the unit circle. Then, the closer the roots P (λ) are to the unit circle, the longer the range propagation of the linear SSM corresponding to such a GRAMA block.
The results derived in this section provide the theoretical foundation and motivation for the employment of GRAMA as a method to address the oversquashing phenomenon in GNNs, and to enhance long-range interaction modeling capabilities, as we show in our experiments in Section 5.
this section cite: []

Section: Experiments
We present the empirical performance of our GRAMA on a suite of benchmarks similar to previous graph SSM studies. Specifically, we show the efficacy in performing long-range propagation, thereby mitigating oversquashing. To this end, we evaluate GRAMA on a graph transfer task (Gravina et al., 2025) in Section 5.1. In a similar spirit, we assess GRAMA on synthetic benchmarks that require the exchange of messages at large distances over the graph, called graph property prediction from Gravina et al. (2023), in Section 5.2. We also verify GRAMA on real-world datasets, including the long-range graph benchmark (Dwivedi et al., 2022b) in Section 5.3, and additional GNN benchmarks in Appendix E.1, where we consider MalNet-Tiny (Freitas et al., 2021), the heterophilic node classification datasets from Platonov et al. (2023), ZINC-12k, OGBG-MOLHIV, Cora, CiteSeer, Pubmed, MNIST CIFAR10, PATTERN, and CLUSTER. In Appendix E.3, we discuss the runtimes of GRAMA, and compare with other methods. In Appendix E.4, we report ablation studies and additional comparisons to provide a comprehensive understanding of our GRAMA, while in Appendix E.6 we include an evaluation on temporal setting. Notably, the performance of GRAMA is compared with popular and state-of-the-art methods, such as MPNN-based models, DE-GNNs, higher-order GNNs, and graph transformers, and shows consistent improvements over its baseline models, with competitive results to state-of-the-art methods (see Appendix F). We note that, in the main text, we report models and variants that are state-of-the-art on the individual benchmarks, which may lead to differences between the tables, while more variants are explored in the appendix. Additional details on baseline methods are presented in Appendix D.1, and the explored grid of hyperparameters in Appendix D.6. We demonstrate GRAMA on three widely used backbones-GCN (Kipf & Welling, 2016), GatedGCN (Bresson & Laurent, 2018), and GPS (Rampášek et al., 2022), highlighting its versatility across different backbone types, including linear MPNNs and graph transformers, and its consistently strong performance regardless of the underlying backbone architecture. We release our code at https://github.com/MosheEliasof/GRAMA.
this section cite: ['b59', 'b40', 'b51', 'b112', 'b85', 'b17', 'b114']

Section: Graph Feature Transfer
Setup. We consider three graph feature transfer tasks based on (Gravina et al., 2025). The objective is to transfer a label from a source to a target node, placed at a distance ℓ in the graph. By increasing ℓ, we increase the complexity of the task and require longer-range information. Moreover, due to oversquashing, the performance is expected to degrade as ℓ increases. We initialize nodes with a random valued feature, and we assign values "1" and "0" to source and target nodes, respectively. We consider three graph distributions, i.e., line, ring, crossed-ring, and four different distances ℓ = {3, 5, 10, 50}. Appendix D.2 provides additional details about the dataset and the task.
Results. Figure 2 reports the test mean-squared error (and standard deviation) of GRAMA compared to well-known models from the literature. Results show that traditional MPNNs (GCN, GAT, GraphSAGE, and GIN) struggle to propagate information effectively over long distances, with their performance deteriorating significantly as the sourcetarget distance ℓ increases. This is evident across all graph types. In contrast, GRAMA coupled with GCN achieves a low error even when the source-target distance is 50. Among the models, A-DGN, SWAN, and GPS come closest to GRAMA performance, as they are a non-dissipative approach and a transformer-based model, respectively. How-
(a) Line (b) Ring (c) Crossed-Ring ever, GRAMA still outperforms all baselines across all graph structures, especially as the propagation distance increases, thereby offering solid empirical evidence of its ability to transfer information across long distances, as supported by our theoretical understanding from Section 4.
this section cite: ['b59']

Section: Graph Property Prediction
Setup. We consider the three graph property prediction tasks presented in (Gravina et al., 2023), investigating the performance of GRAMA in predicting graph diameters, single source shortest paths (SSSP), and node eccentricity on synthetic graphs. To effectively address these tasks, it is essential to propagate information not only from direct neighbors but also from distant nodes within the graph. As a result, strong performance in these tasks mirrors the ability to facilitate long-range interactions. We provide more details on the setup and task in Appendix D.3. For the GPS results, we use a basic GPS with no additional components (e.g., encodings), to quantify the contribution of GRAMA.
Results. Table 1 reports the mean test log 10 (MSE), comparing our GRAMA with various MPNNs, DE-GNNs, and transformer-based models. The results highlight that GRAMA GPS consistently achieves the best performance across all tasks, demonstrating significant improvements over baseline models. For example, in the Eccentricity task, GRAMA GPS reduces the error score by over 1.2 points compared to SWAN and by over 1.7 points compared to A-DGN, which are models designed to propagate information over long radii effectively. Compared to ARMA (Bianchi et al., 2021), our method demonstrates an average improvement of 2.4 points, highlighting the empirical difference between the methods, besides their major qualitative differences.
Overall, these results further validate the effectiveness of our GRAMA in modeling long-range interactions and mitigating oversquashing. Furthermore, GRAMA not only surpasses strong models like GPS, but also strengthens the performance of simple MPNN backbones like GCN. For example, GCN augmented with our GRAMA consistently delivers better results than the baseline GCN, highlighting its ability to enhance traditional message-passing frame- works. This demonstrates that our method can effectively leverage the strengths of simple models while overcoming their limitations in long-range propagation.
this section cite: ['b13']

Section: Long-Range Benchmark
Setup. We assess the performance of our method on the realworld long-range graph benchmark (LRGB) from (Dwivedi et al., 2022b), focusing on the Peptides-func and Peptidesstruct datasets. We follow the experimental setting in (Dwivedi et al., 2022b), including the 500K parameter budget. All transformer baselines include Laplacian positional encodings, for a fair evaluation. Our GRAMA does not use additional encodings. The datasets consist of large molecular graphs derived from peptides, where the structure and function of a peptide depend on interactions between distant parts of the graph. Therefore, relying on short-range interactions, such as those captured by local message passing in GNNs, may not be insufficient to excel at this task. More Table 2: Results for Peptides-func and Peptides-struct averaged over 3 training seeds. Baselines are taken from (Dwivedi et al., 2022b) and (Gutteridge et al., 2023). All MPNN-based methods include structural and positional encoding. The first, second, and third best scores are colored, and we color only the best configuration of GRAMA.
Model Peptides-func Peptides-struct AP ↑ MAE ↓ MPNNs GCN 59.30±0.23 0.3496±0.0013 GatedGCN 58.64±0.77 0.3420±0.0013 ARMA 64.08±0.62 0.2709±0.0016 Multi-hop GNNs DIGL+MPNN+LapPE 68.30±0.26 0.2616±0.0018 MixHop-GCN+LapPE 68.43±0.49 0.2614±0.0023 DRew-GCN+LapPE 71.50±0.44 0.2536±0.0015 DRew-GatedGCN+LapPE 69.77±0.26 0.2539±0.0007 Graph Transformers Transformer+LapPE 63.26±1.26 0.2529±0.0016 SAN+LapPE 63.84±1.21 0.2683±0.0043 GraphGPS+LapPE 65.35±0.41 0.2500±0.0005 DE-GNNs GRAND 57.89±0.62 0.3418±0.0015 GraphCON 60.22±0.68 0.2778±0.0018 A-DGN 59.75±0.44 0.2874±0.0021 SWAN 67.51±0.39 0.2485±0.0009 Graph SSMs Graph-Mamba 67.39±0.87 0.2478±0.0016 GMN 70.71±0.83 0.2473±0.0025 Ours GRAMAGCN 70.93±0.78 0.2439±0.0017 GRAMAGATEDGCN 70.49±0.51 0.2459±0.0020 GRAMAGPS 69.83±0.83 0.2436±0.0022 details on the setup and tasks can be found in Appendix D.4. Results. Table 2 provides a comparison of our GRAMA model with a wide range of baselines. A broader comparison is presented in Table 8. The results indicate that GRAMA outperforms standard MPNNs, transformer-based GNNs, DE-GNNs, SSM-based GNNs, and most Multi-hop GNNs. Such a result highlights the competitiveness of our method and its ability to propagate information effectively. Moreover, its empirical advantage over existing Graph SSMs emphasizes the strength of GRAMA in modeling long-range interactions while maintaining permutation equivariance and processing sequences that go beyond pairwise interactions. Similarly to Section 5.2, our results show that GRAMA strengthens the abilities of simple GNN backbones. Specifically, our method boosts GCN and GatedGCN by more than 11 AP points on the Peptide-func task.
this section cite: ['b40', 'b40', 'b40', 'b65']

Section: Conclusion
We introduced GRAMA, a novel sequence-based framework that enhances the long-range interaction modeling ability and feature update selectivity of Graph Neural Networks (GNNs) through the integration of adaptive neural Autoregressive Moving Average (ARMA) models with potentially any GNN backbone. We draw a theoretical link between SSM models and GRAMA, to build solid groundwork and understanding of the qualitative behavior of GRAMA. Compared with several existing Graph SSMs, our GRAMA allows to benefit from long-range interaction modeling abilities, while maintaining permutation equivariance. Through a series of extensive experiments on 26 synthetic and realworld datasets, we demonstrated that GRAMA consistently offers competitive performance with well-established baseline models, from classical MPNNs to more complex approaches such as Graph Transformers and Graph SSMs.
Overall, GRAMA offers a theoretically grounded, powerful, and flexible solution that bridges the gap between contemporary sequential models and existing graph learning methods, stepping forward towards a new family of graph machine learning models.
this section cite: []

Section: References
Ref_id:b0 Title: The surprising power of graph neural networks with random node initialization Year: ()
Ref_id:b1 Title: Higher-order graph convolutional architectures via sparsified neighborhood mixing Year: (2019)
Ref_id:b2 Title: On the bottleneck of graph neural networks and its practical implications Year: (2021)
Ref_id:b3 Title: State Space Models: A Unifying Framework Year: (2013)
Ref_id:b4 Title: Unitary evolution recurrent neural networks Year: (2016)
Ref_id:b5 Title: On vanishing gradients, over-smoothing, and oversquashing in gnns: Bridging recurrent and graph learning Year: (2025)
Ref_id:b6 Title: Accurate prediction of protein structures and interactions using a three-track neural network Year: (2021)
Ref_id:b7 Title: A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting Year: (2021)
Ref_id:b8 Title: Extended long short-term memory Year: (2024)
Ref_id:b9 Title: Graph Mamba: Towards Learning on Graphs with State Space Models Year: (2024)
Ref_id:b10 Title: Learning long-term dependencies with gradient descent is difficult Year: (1994)
Ref_id:b11 Title: Equivariant subgraph aggregation networks Year: (2022)
Ref_id:b12 Title: Efficient subgraph GNNs by learning effective selection policies Year: (2024)
Ref_id:b13 Title: Graph neural networks with convolutional arma filters Year: (2021)
Ref_id:b14 Title: Beyond lowfrequency information in graph convolutional networks Year: (2021-05)
Ref_id:b15 Title: Improving graph neural network expressivity via subgraph isomorphism counting Year: (2022)
Ref_id:b16 Title: Time Series Analysis: Forecasting and Control Year: (1970)
Ref_id:b17 Title: Residual Gated Graph Con-vNets Year: (2018)
Ref_id:b18 Title: Geometric deep learning: Grids, groups, graphs, geodesics, and gauges Year: (2021)
Ref_id:b19 Title: Spectral networks and locally connected networks on graphs Year: (2013)
Ref_id:b20 Title: A note on over-smoothing for graph neural networks Year: (2020)
Ref_id:b21 Title: GRAND: Graph neural diffusion Year: (2021)
Ref_id:b22 Title: Simple and Deep Graph Convolutional Networks Year: (2020-07)
Ref_id:b23 Title: Optimization-induced graph implicit nonlinear diffusion Year: (2022)
Ref_id:b24 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b25 Title: Adaptive universal generalized pagerank graph neural network Year: (2021)
Ref_id:b26 Title: Graph neural reaction-diffusion networks Year: (2023)
Ref_id:b27 Title: Performers: A new approach to scaling transformers Year: (2020)
Ref_id:b28 Title: From block-toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked transformers Year: (2022)
Ref_id:b29 Title: Mixing gated linear recurrences with local attention for efficient language models Year: (2024)
Ref_id:b30 Title: The arma model in state space form Year: (2004)
Ref_id:b31 Title: Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems Year: (2016)
Ref_id:b32 Title: Polynormer: Polynomialexpressive graph transformer in linear time Year: (2024)
Ref_id:b33 Title: On over-squashing in message passing neural networks: the impact of width, depth, and topology Year: (2023)
Ref_id:b34 Title: Recurrent distance filtering for graph representation learning Year: (2024)
Ref_id:b35 Title: Gbk-gnn: Gated bi-kernel graph neural networks for modeling both homophily and heterophily Year: (2022)
Ref_id:b36 Title: Benchmarking graph transformers Year: (2022)
Ref_id:b37 Title: Benchmarking graph neural networks Year: ()
Ref_id:b38 Title: A Generalization of Transformer Networks to Graphs Year: (2021)
Ref_id:b39 Title: Graph neural networks with learnable structural and positional representations Year: ()
Ref_id:b40 Title: Long Range Graph Benchmark Year: (2022)
Ref_id:b41 Title: Benchmarking graph neural networks Year: (2023)
Ref_id:b42 Title: Novel architectures for graph neural networks motivated by partial differential equations Year: (2021)
Ref_id:b43 Title: Learning general graph spatial operators from paths Year: (2022)
Ref_id:b44 Title: Graph positional encoding via random feature propagation Year: (2023)
Ref_id:b45 Title: Adr-gnn: advectiondiffusion-reaction graph neural networks Year: (2023)
Ref_id:b46 Title: GRANOLA: Adaptive normalization for graph neural networks Year: ()
Ref_id:b47 Title: Graph neural reaction diffusion models Year: (2024)
Ref_id:b48 Title: On the temporal domain of differential equation inspired graph neural networks Year: (2024)
Ref_id:b49 Title: Adaptive message passing: A general framework to mitigate oversmoothing, oversquashing, and underreaching Year: (2024)
Ref_id:b50 Title: Cooperative Graph Neural Networks Year: (2024)
Ref_id:b51 Title: A large-scale database for graph representation learning Year: (2021)
Ref_id:b52 Title: Structured state space for scalable and efficient sequence modeling Year: ()
Ref_id:b53 Title: Diffusion Improves Graph Learning Year: (2019)
Ref_id:b54 Title: Deep Learning for Dynamic Graphs: Models and Benchmarks Year: (2024)
Ref_id:b55 Title: Anti-Symmetric DGN: a stable architecture for Deep Graph Networks Year: (2023)
Ref_id:b56 Title: Non-Dissipative Propagation by Randomized Anti-Symmetric Deep Graph Networks Year: (2023)
Ref_id:b57 Title: Long Range Propagation on Continuous-Time Dynamic Graphs Year: (2024-07)
Ref_id:b58 Title: Temporal graph odes for irregularly-sampled time series Year: (2024)
Ref_id:b59 Title: On Oversquashing in Graph Neural Networks Through The Lens of Dynamical Systems Year: (2025-04)
Ref_id:b60 Title: Efficiently modeling long sequences with structured state spaces Year: (2021)
Ref_id:b61 Title: Combining recurrent, convolutional, and continuous-time models with linear state space layers Year: (2021)
Ref_id:b62 Title: A flexible mechanism for long-range dependencies in state space models Year: (2023)
Ref_id:b63 Title: Implicit graph neural networks Year: (2020)
Ref_id:b64 Title: Structured state space models for efficient sequence modeling Year: ()
Ref_id:b65 Title: Dynamically rewired message passing with delay Year: (2023)
Ref_id:b66 Title: Time Series Analysis Year: (1994)
Ref_id:b67 Title: State-space models. Handbook of econometrics Year: (1994)
Ref_id:b68 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b69 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b70 Title: Port-Hamiltonian Architectural Bias for Long-Range Propagation in Deep Graph Networks Year: (2025)
Ref_id:b71 Title: Bounding the roots of polynomials Year: (1997)
Ref_id:b72 Title: Gradient flow in recurrent nets: the difficulty of learning long-term dependencies Year: (2001)
Ref_id:b73 Title: Matrix analysis Year: (2012)
Ref_id:b74 Title: Open graph benchmark: Datasets for machine learning on graphs Year: (2020)
Ref_id:b75 Title: Densely connected convolutional networks Year: (2017)
Ref_id:b76 Title: On the stability of expressive positional encodings for graphs Year: ()
Ref_id:b77 Title: What can we learn from state space models for machine learning on graphs Year: (2024)
Ref_id:b78 Title: Going deeper into permutation-sensitive graph neural networks Year: (2022)
Ref_id:b79 Title: Global self-attention as a replacement for graph convolution Year: (2021)
Ref_id:b80 Title: Autoregressive moving average graph filtering Year: (2016)
Ref_id:b81 Title: Unleashing the potential of fractional calculus in graph neural networks with FROND Year: (2024)
Ref_id:b82 Title: FoSR: First-order spectral rewiring for addressing oversquashing in GNNs Year: (2023)
Ref_id:b83 Title: Nonlinear Systems Year: (2002)
Ref_id:b84 Title: A review of graph neural networks: concepts, architectures, techniques, challenges, datasets, applications, and future directions Year: (2024)
Ref_id:b85 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b86 Title: GOAT: A global transformer on largescale graphs Year: (2023-07)
Ref_id:b87 Title: Rethinking graph transformers with spectral attention Year: (2021)
Ref_id:b88 Title: Positional encodings in graph transformers Year: (2021)
Ref_id:b89 Title: Structural attention networks for graphs Year: ()
Ref_id:b90 Title: Finding global homophily in graph neural networks when meeting heterophily Year: (2022-07)
Ref_id:b91 Title: Interaction of Crowd Annotators, February 2023 Year: ()
Ref_id:b92 Title: Beyond long sequences Year: ()
Ref_id:b93 Title: The heterophilic graph learning handbook: Benchmarks, models, theoretical analysis, applications and challenges Year: (2024)
Ref_id:b94 Title: Graph inductive biases in transformers without message passing Year: (2023-07)
Ref_id:b95 Title: Diffeomorphic graph-adaptive activation function Year: (2024)
Ref_id:b96 Title: A fractional graph laplacian approach to oversmoothing Year: (2023)
Ref_id:b97 Title: Simplifying approach to node classification in graph neural networks Year: (2022)
Ref_id:b98 Title: Neural Network for Graphs: A Contextual Constructive Approach Year: (2009)
Ref_id:b99 Title: Weisfeiler and leman go neural: Higher-order graph neural networks Year: (2019)
Ref_id:b100 Title: Attending to graph transformers Year: (2024)
Ref_id:b101 Title: Relational pooling for graph representations Year: (2019)
Ref_id:b102 Title: Janossy pooling: Learning deep permutation-invariant functions for variable-size inputs Year: (2019)
Ref_id:b103 Title: Efficiency in state space models for sequence learning Year: (2023)
Ref_id:b104 Title: Learning convolutional neural networks for graphs Year: (2016)
Ref_id:b105 Title: Revisiting graph neural networks: All we have is low-pass filters Year: (2019)
Ref_id:b106 Title: Graph neural networks exponentially lose expressive power for node classification Year: (2020)
Ref_id:b107 Title: On the universality of linear recurrences followed by nonlinear projections Year: (2023)
Ref_id:b108 Title: Resurrecting recurrent neural networks for long sequences Year: (2023)
Ref_id:b109 Title: Permutation equivariant layers for higher order interactions Year: (2022)
Ref_id:b110 Title: On the difficulty of training recurrent neural networks Year: (2013)
Ref_id:b111 Title: Geom-gcn: Geometric graph convolutional networks Year: (2020)
Ref_id:b112 Title: A critical look at the evaluation of GNNs under heterophily: Are we really making progress? Year: (2023)
Ref_id:b113 Title: Graph neural ordinary differential equations Year: (2019)
Ref_id:b114 Title: Recipe for a General, Powerful, Scalable Graph Transformer Year: (2022)
Ref_id:b115 Title: Pytorch geometric temporal: Spatiotemporal signal processing with neural machine learning models Year: (2021)
Ref_id:b116 Title: Graph-coupled oscillator networks Year: (2022)
Ref_id:b117 Title: Survey on Oversmoothing in Graph Neural Networks Year: (2023)
Ref_id:b118 Title: Deep neural networks motivated by partial differential equations Year: (2020)
Ref_id:b119 Title: Random features strengthen graph neural networks Year: (2021)
Ref_id:b120 Title: The graph neural network model Year: (2008)
Ref_id:b121 Title: Theoretical guarantees for permutationequivariant quantum neural networks Year: (2024)
Ref_id:b122 Title: Masked label prediction: Unified message passing model for semi-supervised classification Year: ()
Ref_id:b123 Title: Exphormer: Sparse transformers for graphs Year: (2023)
Ref_id:b124 Title: Applied nonlinear control Year: (1991)
Ref_id:b125 Title: All in a row: Compressed convolution networks for graphs Year: (2023)
Ref_id:b126 Title: Towards dynamic message passing on graphs Year: (2024)
Ref_id:b127 Title: Where did the gap go? reassessing the long-range graph benchmark Year: (2023)
Ref_id:b128 Title: Understanding oversmoothing in graph neural networks Year: ()
Ref_id:b129 Title: Attention is all you need Year: (2017)
Ref_id:b130 Title: Graph attention networks Year: (2018)
Ref_id:b131 Title: Graph-mamba: Towards long-range graph sequence modeling with selective state spaces Year: (2024)
Ref_id:b132 Title: The heterophilic snowflake hypothesis: Training and empowering gnns for heterophilic graphs Year: (2024)
Ref_id:b133 Title: How powerful are spectral graph neural networks Year: (2022-07)
Ref_id:b134 Title: Dissecting the Diffusion Process in Linear Graph Convolutional Networks Year: (2021)
Ref_id:b135 Title: ACMP: Allen-cahn message passing with attractive and repulsive forces for graph neural networks Year: (2023)
Ref_id:b136 Title: Hypothesis Testing in Time Series Analysis. Statistics / Uppsala universitet. Almqvist & Wiksells boktr Year: (1951)
Ref_id:b137 Title: A linear graph transformer framework via kernel decomposition approach Year: (2023)
Ref_id:b138 Title: Representation learning on graphs with jumping knowledge networks Year: (2018-07)
Ref_id:b139 Title: How powerful are graph neural networks? Year: (2019)
Ref_id:b140 Title: Revisiting semi-supervised learning with graph embeddings Year: (2016)
Ref_id:b141 Title: Graphormer: A transformer for graphs Year: (2021)
Ref_id:b142 Title: A review of recurrent neural networks: Lstm cells and network architectures Year: (2019)
Ref_id:b143 Title: Graph transformers for long-range dependencies Year: (2019)
Ref_id:b144 Title: Bigbird: Transformers for longer sequences Year: (2020)
Ref_id:b145 Title: Linked dynamic graph cnn: Learning on point cloud via linking hierarchical features Year: (2019)
Ref_id:b146 Title: Graph neural convection-diffusion with heterophily Year: (2023-08)
Ref_id:b147 Title: A Temporal Graph Convolutional Network for Traffic Prediction Year: (2020)
Ref_id:b148 Title: Beyond homophily in graph neural networks: Current limitations and effective designs Year: (2020)
Ref_id:b149 Title: Graph neural networks with heterophily Year: (2021-05)
Ref_id:b150 Title: Ordinary differential equations on graph networks Year: (2020)
Ref_id:b151 Title:  Year: ()
Ref_id:b152 Title: Amazon-ratings Year: ()
Ref_id:b153 Title:  Year: ()
