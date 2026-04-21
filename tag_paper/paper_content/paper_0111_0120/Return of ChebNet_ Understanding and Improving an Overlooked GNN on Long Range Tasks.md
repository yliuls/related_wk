Title: Return of ChebNet: Understanding and Improving an Overlooked GNN on Long-Range Tasks
Abstract: ChebNet, one of the earliest spectral GNNs, has largely been overshadowed by Message Passing Neural Networks (MPNNs), which gained popularity for their simplicity and effectiveness in capturing local graph structure. Despite their success, MPNNs are limited in their ability to capture long-range dependencies between nodes. This has led researchers to adapt MPNNs through rewiring or make use of Graph Transformers, which compromises the computational efficiency that characterized early spatial message-passing architectures, and typically disregards the graph structure. Almost a decade after its original introduction, we revisit ChebNet to shed light on its ability to model distant node interactions. We find that out-of-box, ChebNet already shows competitive advantages relative to classical MPNNs and GTs on long-range benchmarks, while maintaining good scalability properties for high-order polynomials. However, we uncover that this polynomial expansion leads ChebNet to an unstable regime during training. To address this limitation, we cast ChebNet as a stable and non-dissipative dynamical system, which we coin Stable-ChebNet. Our Stable-ChebNet model allows for stable information propagation, and has controllable dynamics which do not require the use of eigendecompositions, positional encodings, or graph rewiring. Across several benchmarks, Stable-ChebNet achieves near state-of-the-art performance.

Section: Introduction
Graph Neural Networks (GNNs) [81,43,74,67,14,25,44] have emerged as a prevalent framework for handling data defined on graphs. Graph convolutional networks have their roots in spectral approaches that extend convolutional filters to non-Euclidean domains. The first practical instantiation of a GNN was proposed by [14], which leveraged the eigenbasis of the graph Laplacian to perform spectral filtering, as an attempt to generalize image convolutions to non-euclidean structures. However, this formulation required costly eigen-decompositions at each layer. Defferrard et al. [25] addressed this inefficiency by approximating spectral filters with truncated Chebyshev polynomials, giving rise to ChebNet, the first tractable and localized spectral GNN. By parameterizing filters as K-order polynomials of the Laplacian, ChebNet could aggregate information from K-hop neighborhoods without repeated eigendecompositions, thereby enabling scalable spectral convolution on large graphs.
In 2017, Kipf and Welling distilled ChebNet into a simpler, first-order approximation now known as the Graph Convolutional Network (GCN) by (i) restricting the polynomial order to one and (ii) tying filter coefficients across hops [57]. This yielded an architecture that was both lightweight and effective: a GCN with few layers would achieve strong node-classification performance on standard homophilic benchmarks, and its O(|E|) complexity made it practical for large-scale graphs. GCN's efficiency and strong locality bias quickly made it the default baseline, and subsequent message-passing neural networks (MPNNs) adopted a similar paradigm of iterative neighborhood aggregation [41]. While the original ChebNet's high-order Chebyshev filters induce unstable dynamics resulting in dissipative behavior, our Stable-ChebNet yields bounded propagation through layers.
Despite their popularity, MPNNs exhibit pronounced shortcomings when made deeper and when capturing long-range dependencies [33]. Repeated neighborhood aggregations tend to cause representational collapse, where node features become indistinguishable (often referred to as "oversmoothing") [15,69], and information from distant nodes is "squashed" through narrow bottlenecks, limiting the ability to model global context [1]. In recent years, researchers have proposed various strategies to overcome these limitations. To mitigate oversmoothing, several models have drawn on principles from physics [28,12] to preserve feature diversity across layers. At the same time, efforts to capture long-range dependencies have led to graph-rewiring techniques [82,49,7] that add or reweight edges to shorten information pathways, as well as the emergence of graph transformers [30], which replace purely local aggregation with global self-attention mechanisms. Although these advances can alleviate depth-related pathologies, they often trade off numerous benefits that made MPNNs appealing: scalability, parameter efficiency, and to only process information along the graph's edges. In this context, ChebNet and other spectral GNNs are usually relegated to a footnotementioned only as a predecessor to GCN, which is typically not revisited as a competitive baseline.
In this work, we revisit ChebNet from first principles. We demonstrate that the original Cheb-Net (without any rewiring or attention mechanisms) already delivers state-of-the-art performance or is close on long-range graph tasks while scaling gracefully to large graphs. By deriving and analyzing ChebNet's linearized dynamics, we prove that enlarging its receptive field introduces signal-propagation instabilities. To overcome this, we propose Stable-ChebNet, a minimal set of architectural modifications that restore stable propagation for arbitrarily large receptive fields, supported by both theoretical guarantees and empirical validation. To build intuition for our Stable-ChebNet framework, Figure 1 illustrates how classical ChebNet filters (top) can exhibit unbounded dynamics, whereas our antisymmetric, forward-Euler discretization yields smooth, stable propagation (bottom). An intuitive heat-transfer analogy helps explain the difference: if we inject 'heat' at seed nodes, the high-order filters of vanilla ChebNet diffuse and dissipate this heat so that distant nodes cool rapidly. In contrast, the non-dissipative dynamics induced by the antisymmetric, forward-Euler step in Stable-ChebNet preserve energy, keeping temperatures higher at nodes many hops away.
Across a suite of challenging long-range node-and graph-level benchmarks, Stable-ChebNet matches or outperforms state-of-the-art message-passing neural networks and graph transformers, while retaining the ChebNet backbone. We hope this work will reignite interest in spectral GNNs as a scalable, theoretically grounded alternative for long-range graph modeling.
this section cite: ['b80', 'b42', 'b73', 'b66', 'b13', 'b24', 'b43', 'b13', 'b24', 'b56', 'b40', 'b32', 'b14', 'b68', 'b0', 'b27', 'b11', 'b81', 'b48', 'b6', 'b29']

Section: Contributions and Outline
• In Section 3.1, we empirically demonstrate that vanilla ChebNet can achieve very strong performance on long-range benchmarks without incurring prohibitive computational cost.
• In Section 3.2, we analyze ChebNet's signal-propagation dynamics, providing exact sensitivity analysis, and theoretically and empirically prove the emergence of instability for large filter order.
• In Section 3.3, we introduce Stable-ChebNet, which enforces layer-wise stability.
• In Section 4, we empirically validate that Stable-ChebNet consistently outperforms MPNNs, rewiring methods, and graph transformers across a number of tasks.
this section cite: []

Section: Background

this section cite: []

Section: Background on Spectral Graph Neural Networks
Spectral GNNs extend the notion of convolution to graphs by leveraging the eigen-decomposition of the graph Laplacian. Given an undirected graph G = (V, E) with normalized Laplacian L = I -D -1/2 AD -1/2 , any graph signal X ∈ R n can be filtered in the spectral domain via Y = Ug θ (Λ)U ⊤ X, where L = UΛU ⊤ diagonalizes the Laplacian, Λ = diag(λ 1 , . . . , λ n ) its eigenvalues, and g θ is a learnable spectral response. Early methods directly parameterize g θ (Λ), but require an expensive eigendecomposition of the Laplacian, which can be computationally and memory intensive [14]. ChebNet alleviates the cost of an explicit eigendecomposition by approximating g θ using the recurrence relation for a K-th order Chebyshev polynomial in L [25]. The latter defines g θ as g θ (Λ) ≈ K k=0 Θ k T k ( Λ), where T k ( Λ) is the k-th polynomial of Λ with Λ = 2Λ λmax -I n . The spectral convolution can then be written without any eigendecomposition as the truncated expansion:
Y = K k=0 Θ k T k ( L) X(1)
where L = 2L λmax -I n enabling efficient, localized filtering in O(K|E|) time.
this section cite: ['b13', 'b24']

Section: MPNNs and their Limitations
Message-Passing Neural Networks (MPNNs) define a general framework in which node features are iteratively updated by exchanging "messages" along edges. At each layer l, every node v aggregates information from its neighbors u ∈ N (v) and combines it with its own representation. This formulation unifies many graph models, including graph convolutional networks (GCNs) [57] and graph attention networks (GATs) [83]. While this local neighborhood aggregation captures structural information effectively, it has a limited capacity to model long-range interactions within the graph. This is due to the phenomenon of over-squashing, an information bottleneck that impedes effective information flow among distant nodes [1,82,27]. Numerous techniques have emerged to address this limitation such as graph rewiring [49,7], Graph Transformers [71,80,79] in addition to some enhanced spatial methods that tackle over-squashing through combined local and global information [75,40], or through non-dissipativity achieved by antisymmetric weight parameterization [45,46] or port-Hamiltonian systems [54].
0 10 20 Epoch Time (sec.) M⋅K = 12 ChebNet K=12, M=1 ChebNet K=6, M=2 ChebNet K=4, M=3 DRew 0 25 50 Epoch Time (sec.) M⋅K = 20 ChebNet K=20, M=1 ChebNet K=10, M=2 ChebNet K=5, M=4 DRew 175 200 0 25 Epoch Time (sec.) M⋅K = 40 ChebNet K=40, M=1 ChebNet K=20, M=2 ChebNet K=10, M=4 DRew However, some of the aforementioned methods suffer from substantial overhead due to denser graph shift operators or the use of all-pairs interactions. Specifically, [79] and other graph transformers increase computational complexity through dense attention-maps; [40] relies on costly full eigendecomposition operations; and graph rewiring techniques heavily pre-process the graph topology, incurring O(n 3 ) time in the case of [49].For a detailed discussion on the relevant literature, we point the reader to the Appendix A.
this section cite: ['b56', 'b82', 'b0', 'b81', 'b26', 'b48', 'b6', 'b70', 'b79', 'b78', 'b74', 'b39', 'b44', 'b45', 'b53', 'b78', 'b39', 'b48']

Section: The Connection between ChebNet and GCN
Although often interpreted as disparate models, GCN [57] can be derived as a special case of ChebNet [25] by truncating the Chebyshev expansion to K = 1 and making additional simplifications such as approximating L ≈ I -D -1/2 AD -1/2 and adding self-loops to improve numerical stability. These choices introduced a strong locality bias, which aligned with widely used homophilic benchmarks and significantly reduced computational costs. Due to some of these strengths, GCN has become the de facto backbone for many modern GNNs, and has led to ChebNet being somewhat forgotten, under the assumption that it does not perform well and will not scale well in mid-to large-size graphs due to its spectral nature. For instance, until recently, ChebNet was almost never included in popular GNN benchmarks, such as [31], or in those designed to evaluate long-range dependencies, such as those in the Long Range Graph Benchmark (LRGB) [33] and the numerous studies that leveraged this dataset to assess the effectiveness of graph rewiring, positional encodings, or Transformer-based architectures. ChebNet was similarly disregarded from large-scale graph evaluations such as the OGB benchmarks [55], presumably due to prevailing assumptions about its scalability.
this section cite: ['b56', 'b24', 'b30', 'b32', 'b54']

Section: Analyzing and Improving ChebNet from First Principles

this section cite: []

Section: The Effectiveness and Scalability of Vanilla ChebNet
In this subsection, we perform two high-level empirical tests to challenge the commonplace assumption that spectral GNNs inherently suffer from poor performance and limited scalability. We do so by firstly testing ChebNet [25] on a long-range test on the Ring Transfer dataset from [27], which has become a de facto benchmark for state-of-the-art methods seeking to model long-distance dependencies on graphs. Furthermore, to test scalability, we compare epoch training times on the peptides-func dataset from [33] with respect to state-of-the-art rewired MPNNs [49] based on a GCN backbone. We compare ChebNet with different numbers of filters K and layers M Cheb with an M MPNN -layer MPNN. To ensure a fair comparison, we ensure that M Cheb K = M MPNN so that the two methods would have the same receptive field. The results are shown in Figures 2 and 3.
As seen from Figure 3, ChebNet is capable of performing long-range retrieval on the ring transfer dataset for rings of up to 50 nodes, which is a substantial improvement over a regular GCN. While this finding aligns with intuition, as we will formalize in the following sections, it may nonetheless surprise practitioners, since ChebNet is not typically included as a benchmark in long-range studies such as that in [30]. On the other hand, as seen in Figure 2, ChebNet scales gracefully on standard tasks such as the peptides-func dataset. When compared to DRew, a stateof-the-art method that leverages both dynamic rewiring and delay to propagate information through a GCN (or other MPNN) backbone, we observe that DRew's epoch times are almost up to two orders of magnitude larger than ChebNet's. We note further that this inefficiency is not unique to this particular baseline: Graph Transformers incur quadratic scaling in the number of nodes and, by effectively discarding the underlying graph structure, trade-off inductive bias for increased compute, while many rewiring approaches depend on cubic-time algorithms (e.g., Floyd-Warshall or eigendecompositions). Together, these examples underscore how numerous contemporary techniques intended to overcome traditional message-passing limitations actually erode computational benefits, whereas ChebNet, a more natural spectral baseline that generalizes GCN, delivers both training speed and strong performance out-of-the-box.
this section cite: ['b24', 'b26', 'b32', 'b48', 'b29']

Section: Signal Propagation Analysis of ChebNet
In this subsection, we conduct a sensitivity analysis of ChebNet via the spectral norm of the Jacobian of node features, providing an exact characterization of ChebNet's information flow through different layers and between pairs of nodes, in the spirit of [3]. Specifically, we begin by analyzing the layer-wise Jacobian for Spectral GNNs that use polynomial filters. In this setting, we demonstrate that the layer-wise Jacobian becomes unstable as the polynomial order K increases. Lastly, we investigate the sensitivity of node pairs when using ChebNet. We provide the proofs for the statements in Appendix B.
Lemma 3.1 (Layer-Wise Jacobian for a Spectral GNN). Consider a linear spectral GNN whose layer-wise update is performed through the following polynomial filter f (X) = K k=1 T k (L) X Θ k , where X ∈ R n×d is the node feature matrix, T k (L) ∈ R n×n is the k-th polynomial of the Laplacian L ∈ R n×n , and Θ k ∈ R d×d ′ are learnable weight matrices. Then, the vectorized Jacobian
J = ∂ vec(f (X))/∂ vec(X) is J = K k=1 Θ ⊤ k ⊗ T k (L).(2)
Given the layer-wise Jacobian from Lemma 3.1, we proceed to analyze the dynamics of a Spectral GNN in Theorem 1 below. Specifically, we focus on the case where the polynomial filter can be approximated by powers of the Laplacian, i.e., T k (L) = L k .
-1 0 1 Real -1 0 1 Imaginary K = 1 -1 0 1 Real -1 0 1 K = 2 -1 0 1 Real -1 0 1 K = 3 -1 0 1 Real -1 0 1 K = 5 -1 0 1 Real -1 0 1 K = 10
Figure 4: Singular-value spectra of the graph-wise Jacobian in the complex plane for vanilla ChebNet with increasing polynomial order K.
Theorem 1 (Layer-Wise Jacobian singular-value distribution). Assume the setting of Lemma 3.1, with T k (L) = L k and L the symmetric normalized Laplacian, and let all Θ k ∈ R d×d be initialized with i.i.d. N (0, σ 2 ) entries. Denote the eigenvalues of L as {λ1, . . . , λn}, the squared singular values of Θ k Θ T k as {µ 1,k , . . . , µ d,k }, and the squared singular values of the Jacobian by γi,j. Then, for sufficiently large d the empirical eigenvalue distribution of Θ k Θ T k converges to the Marchenko-Pastur distribution ∀k. Then, the mean and variance of each γi,j are
E γi,j = σ 2 K k=1 λ 2k i ,(3)
Var γi,j = σ 4 K k=1 λ 2k i 2 .(4)
Theorem 1 shows that the singular values spectrum of the layer-wise Jacobian depends on the sum over the powers of the normalized Laplacian's eigenvalues. This indicates that larger polynomial orders K push the singular value-spectrum towards unstable dynamics, as empirically demonstrated in Figure 4. Stacking several layers of large filter orders will therefore severely hinder the trainability of a Spectral GNN of this form.
Beyond analyzing the information propagation dynamics between different layers, we are interested in quantifying the communication ability between distant nodes in the graph. Recent literature has proposed to measure information flow in the graph by evaluating the sensitivity of a node embedding after l layers (i.e., hops of propagation) with respect to the input of another node using the node-wise Jacobian [82,27,3], i.e., ∂x
(l) u /∂x (0) v .
Following this approach, we measure how sensitive a node embedding of ChebNet at an arbitrary layer l with respect to the initial features of another node, to better illustrate the long-range propagation capabilities of spectral GNNs.
Theorem 2 (ChebNet Sensitivity). Consider a Chebyshev-based Graph Neural Network (ChebNet) defined as:
X (l+1) = K k=0 T k (L)X (l) W (l) k ,(5)
where L ∈ R n×n is the graph Laplacian, T k (L) is the k-th Chebyshev polynomial of the Laplacian, and W (l) k are learnable weight matrices. Assume activation function σ is identity and let X (0) be the input features.
Then, the sensitivity of node v with respect to node u after l layers is given by:
∂x (l) v ∂x (0) u = l-1 l=0 K k=0 T k (L)W (l) k v,u .(6)
This result indicates that the sensitivity of ChebNet is closely tied to the polynomial order K. We note that, in the case of K = 1, the sensitivity aligns with that of standard MPNNsfoot_0 (such as GCN), exhibiting reduced long-range communication. However, for K > 1, the higher polynomial orders significantly enhance the ChebNet's sensitivity, enabling more effective long-range propagation and improving the overall capacity to capture distant dependencies in the graph. Therefore, we conclude that a large filter order K is needed to enable long-range communication between nodes in the graph, but this will result in unstable training dynamics. This serves to explain the decay in performance in Figure 3. In the next subsection, we will propose a remedy to this issue.
this section cite: ['b2', 'b81', 'b26', 'b2']

Section: Stable-ChebNet: Stability with Antisymmetric Parameterization
As discussed in Section 3.2, although ChebNet demonstrates strong long-range propagation capabilities, increasing the polynomial order can introduce significant instability into the model dynamics. To address this challenge, we propose Stable-ChebNet, a simple yet effective modification of classical ChebNet aimed at improving its stability. Recent literature has demonstrated that the effectiveness of neural architectures can be significantly improved by framing them as stable, non-dissipative dynamical systems [50,18,45,46,54]. The core idea behind these approaches is to carefully regulate the spectrum of the Jacobian matrix to ensure the network operates within a stable regime. Specifically, this behavior can be achieved by constraining the eigenvalues of the Jacobian to be purely imaginary. Under this constraint, the input graph information is effectively propagated through the successive transformations into the final nodes' representation. Motivated by this line of work, we begin by reformulating ChebNet as a continuous-time differential equation. Specifically, we consider the following ordinary differential equation (ODE):
dX(t) dt = K k=0 T k (L)X(t)W k(7)
for time t ∈ [0, T ] and subject to the initial condition (i.e., the input features) X(0) = X (0) . In other words, the dynamics of the system (i.e., the continuous flow of information over the graph) is now described as the ChebNet update rule. To ensure the Jacobian of this system has purely imaginary eigenvalues, a straightforward approach is to use antisymmetric weight matricesfoot_1 and the symmetrically normalized Laplacian. This choice, as formalized in the following theorem, directly enforces the desired spectral property, leading to inherently stable dynamics.
this section cite: ['b49', 'b17', 'b44', 'b45', 'b53']

Section: Theorem 3 (Purely Imaginary Eigenvalues).
Let L be the symmetric normalized Laplacian and -W k = W ⊤ k ∀k = 0, . . . , K, then the graph-wise Jacobian of the ODE in Equation (7) has purely imaginary eigenvalues, i.e., Re(λi(J)) = 0, ∀i.
Even in this section, we provide the proofs for the statements in Appendix B. Theorem 3 shows that by enforcing antisymmetry in the weight matrices and leveraging the symmetric structure of the Laplacian, we guarantee that the Jacobian has purely imaginary eigenvalues, ensuring that node representations remain sensitive to input features of far away nodes without suffering from the instability of standard ChebNet.
As for standard differential-equation-inspired neural architectures, a numerical discretization method is needed to solve Equation (7). We solve the equation with a simple finite difference scheme, i.e., forward Euler's method, yielding the following node update equation
X (l+1) = X (l) + ϵ K k=0 T k (L)X (l) (W k -W ⊤ k -γI) (9
)
where I is the identity matrix, γ ∈ R is a hyper-parameter that maintains the stability of the forward Euler method, and ϵ ∈ R + is the discretization step.
We refer to the ODE in Equation ( 9) as Stable-ChebNet, and in the following, we show that this new formulation achieves second-order stability. This is a critical improvement, as a naive Euler discretization of the original ChebNet (without imposing constraints on the Jacobian eigenvalues) would result in only first-order stability, which is considerably more prone to numerical instability. Specifically, without these constraints, the model can exhibit exponential growth or decay in the gradients, significantly limiting its ability to capture long-range dependencies [3]. In contrast, second-order stable systems, like Stable-ChebNet, maintain controlled gradient dynamics over longer timescales, allowing for effective long-range information propagation.
X (l+1) = X (l) + ϵ K k=0 T k (L)X (l) W (l) k ,(10)
with small step size ϵ > 0 and antisymmetric weight matrices:
(W (l) k ) ⊤ = -W (l) k , ∀k, l.(11)
Then the Jacobian J (l) of the layer does not lead to exponential growth or decay across layers. Specifically, we have:
∥J (l) ∥2 = 1 + O(ϵ 2 ).(12)
Conversely, for general weights without the antisymmetric property, exponential growth or decay of the Jacobian norm typically occurs.
this section cite: ['b6', 'b2']

Section: Experiments
We evaluate ChebNet and its stable formulation (Stable-ChebNet) across a variety of settings to thoroughly assess long-range capabilities. In this section, we report and discuss the performance of both models on these benchmarks. We report additional experiments on heterophilic node classification tasks from [70] in Appendix F. We run our experiments on a single A100 GPU and provide the full details on the hyperparameter search for all datasets in Appendix D, and baseline and datasets details in Appendix C.
this section cite: ['b69']

Section: Graph Property Prediction Dataset.
We evaluate our model's ability to predict long-range graph properties using a synthetic dataset developed by Corso et.al in [24] under the experimental setup of [45]. The dataset consists of undirected graphs drawn from a diverse set of random and structured families (Erdős-Rényi, Barabási-Albert, caterpillar, etc), ensuring a broad coverage of topological properties. Each graph contains between 25 and 35 nodes as per the setup of Gravina et.al in [45] (in contrast to the 15-25 node range originally used by [24]), thus increasing task complexity and raising the need for long-range information propagation. Finally, each node is assigned a single scalar feature sampled uniformly at random from the interval [0, 1]. We provide a detailed comparison in Table 1.
Results. Compared to classical ChebNet, Stable-ChebNet yields consistent and significant gains across all three tasks. On the Diameter task, classical ChebNet achieves a log 10 (MSE) of -0.15, whereas Stable-ChebNet improves it to -0.25. On the Single Source Shortest Path (SSSP) task, the gain is larger and goes from -1.85 using ChebNet to -2.21 with our Stable-ChebNet form. Finally, on Eccentricity, where the difficulty is highest due to the necessity to propagate information about the most distant nodes individually, classical ChebNet achieves a log 10 (MSE) of -1.22 while Stable-ChebNet reaches -2.10, reducing the average prediction error by more than an order of magnitude relative to the baseline. Relative to other baselines, Stable-ChebNet dominates most models based on standard message-passing or modified diffusion mechanisms. Methods like GCN and GCNII are badly over-squashed, barely reaching negative log 10 (MSE) values.
this section cite: ['b23', 'b44', 'b44', 'b23']

Section: Over-Squashing Analysis on Barbell Graphs.
To further investigate the robustness of our model to oversquashing, we use as a benchmark the barbell regression tasks introduced in [5]. In this task, a model that fails to transfer any information across the single bridge edge will produce an essentially random constant and obtain a mean-squared error (MSE) close to 1; an error in the 0.4-0.6 band indicates that only a partial amount of information has overcome the bottleneck. Errors around ≈ 0.25 and below suggest that the oversquashing has been effectively overcome. In this work, we compare Stable-ChebNet's performance on barbell graphs of varying sizes (N = 10, 25, 50, 100) against numerous baselines, mainly an MLP and variants of MPNNs such as GCN [57], GAT [83], and SAGE [51]. Further description of the task is found in Appendix C.2.
10 25 50 Number of nodes N 0.00 0.25 0.50 0.75 1.00 MSE MLP GCN SAGE GAT ChebNet Stable-ChebNet 10 0.00 0.02 0.04 Results. We observe in Figure 5 that both a classical ChebNet and Stable-ChebNet successfully learn the small N = 10 case with negligible error. However, for moderate graph sizes with N = 50, a classical ChebNet with fixed K = 8 already sits in the "partial collapse" regime as its MSE increases to around 0.90 and slides towards the random-guess area as N keeps growing (Table 2). For the same range of hops K, replacing the standard update with our stable Eulerbased formulation keeps the error almost two orders of magnitude smaller with an MSE below 0.20, confirming that the non-dissipative timestepping effectively prevents the over-squashing phenomenon.
Table 2: Mean squared error (MSE) of ChebNet and Stable-ChebNet on the over-squashing experiment for barbell graphs. Left: sizes N = 50, 70 for K = 9 and 10. Right: size N = 100 for K = 20.
this section cite: ['b4', 'b56', 'b82', 'b50']

Section: Method K 50 70
ChebNet K = 9 0.32 ± 0.39 1.08 ± 0.05 K = 10 0.05 ± 0.00 1.08 ± 0.01 Stable-ChebNet K = 9 0.17 ± 0.11 0.47 ± 0.49 K = 10 0.05 ± 0.00 0.06 ± 0.03
this section cite: []

Section: Method K 100
ChebNet K = 20 0.87 ± 0.05 Stable-ChebNet (ours) K = 20 0.21 ± 0.27 Open-Graph Benchmark. To evaluate real-world applicability on large-scale graphs, we run experiments for node-level tasks on two large-scale graph datasets from the Open-Graph Benchmark (OGB) [55]. ogbn-arxiv is a citation network in which each node corresponds to an academic paper, and the task is node classification by predicting the subject area of unseen papers. The other dataset we use, ogbn-proteins, is a protein-protein interaction network aimed at inferring protein functions. To ensure a fair comparison and emphasize efficiency, we limit the number of parameters in our models to be within the same range as those used in existing and most recent OGB benchmarks.
Results. Table 3 reports the performance on the ogbn-arxiv citation network, where ChebNet achieves around 73% test accuracy, while Stable-ChebNet boosts the performance further to 75.7%, outperforming all other methods, including a variety of MPNNs and Graph Transformers such as GraphGPS [71] and Exphormer [80]. Similarly, on the ogbn-proteins interaction network, ChebNet attains about 77.6% accuracy compared to Stable-ChebNet's 79.5% (Table 4). Competing approaches achieve nearly 72% for MPNN-based methods, while Transformer-based models' performances range from 77.4% for NodeFormer [88] to 79.5% for SGFormer [89]. Hence, Stable-ChebNet remarkably competes and often outperforms state-of-the-art models on this benchmark, demonstrating that the Euler formulation consistently narrows the gap with and in some cases overtakes Transformer baselines such as SGFormer [89] and Spexphormer [79]. Together, these results demonstrate that augmenting ChebNet with an Euler step not only addresses the classical ChebNet's shortcomings on long-range information propagation but also performs effectively well on graphs with hundreds of thousands of nodes, in contrast to regular-sized graphs seen in previous experiments. Long-Range Graph Benchmark (LRGB). LRGB [33] is a collection of GNN benchmarks that evaluate models on tasks involving long-range interactions. We use two of its molecular-property datasets: Peptides-func for graph classification and Peptides-struct for graph regression.
this section cite: ['b54', 'b70', 'b79', 'b87', 'b88', 'b88', 'b78', 'b32']

Section: Results.
A detailed leaderboard is shown in Table 5. It can be seen that Stable-ChebNet improves upon its vanilla counterpart. Together with S2GCN, it reaches an average precision (AP) above 70 on Peptide-func and a Mean Absolute Error (MAE) below 0.26 on the regression task, bearing in mind that S2GCN requires a more expensive full Laplacian eigendecomposition. Overall, our model achieves competitive performance on peptide structures with results competing with and often outperforming some well-known graph-based models including graph transformers such as Exphormer [80] and GraphViT [53], state space models including Graph Mamba and GMN [84], and rewiring methods like DRew [49]. It is worth noting that the gain in AP for DRew comes at the cost of computing positional encodings (Laplacian eigenvectors) for every graph before training, while Stable-ChebNet does not use any positional encodings.
this section cite: ['b79', 'b52', 'b83', 'b48']

Section: Heterophilic benchmarks
We further assess Stable-ChebNet on node-classification tasks explicitly designed to stress performance under heterophily, following the standardized protocol of Platonov et al. ("Roman-empire", "Amazon-ratings", "Minesweeper" and "Tolokers") [70]. We keep the exact data processing, splits, and metrics recommended therein. Concretely, we report accuracy on Roman-empire and Amazon-ratings, and ROC-AUC on Minesweeper and Tolokers averaging over four random initializations as in the protocol.
Link to oversmoothing, heterophily, and long-rangeness. Our heterophilic results (Table 9) should not be over-interpreted as "evidence of long-range propagation" or as a direct antidote to oversmoothing. The recent position paper by Arnaiz-Rodríguez & Errica [2] argues that several widespread assumptions in the literature are often conflated: (i) that heterophily is inherently detrimental while homophily is beneficial, (ii) that long-range propagation is best evaluated on heterophilic graphs, and (iii) that performance degradation mainly arises from oversmoothing. They show that heterophily, long-range interactions, and oversmoothing are orthogonal factors: a graph may be heterophilic yet dominated by local dependencies, or homophilic yet require long-range reasoning. Hence, evaluations should focus on the nature of the learning task, not merely on global homophily ratios. In this light, our Stable-ChebNet scores on Roman-empire, Minesweeper, and Tolokers demonstrate that a stable spectral propagator has competitive performance on standardized heterophily benchmarks, but they do not by themselves certify long-rangeness. Those conclusions are better supported by our dedicated long-range tests and stability analysis.
this section cite: ['b69', 'b1']

Section: Conclusion
In this work, we have re-examined ChebNet, one of the earliest spectral GNNs, from first principles, uncovering its innate ability to capture long-range dependencies via higher-order polynomial filters, but also its susceptibility to unstable propagation dynamics as the polynomial order grows. By casting ChebNet as a continuous-time ODE and imposing antisymmetric weight constraints, we introduced Impact Statement. This work aims to advance the field of machine learning on graph-structured data which are abundant in the real world. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.
this section cite: []

Section: References
Ref_id:b0 Title: On the bottleneck of graph neural networks and its practical implications Year: (2021)
Ref_id:b1 Title: oversquashing Year: (2025)
Ref_id:b2 Title: On vanishing gradients, oversmoothing, and over-squashing in gnns: Bridging recurrent and graph learning Year: (2025)
Ref_id:b3 Title: Rewiring techniques to mitigate oversquashing and oversmoothing in gnns: A survey Year: (2024)
Ref_id:b4 Title: Bundle neural networks for message diffusion on graphs Year: (2024)
Ref_id:b5 Title: On measuring long-range interactions in graph neural networks Year: (2025)
Ref_id:b6 Title: Locality-aware graph rewiring in GNNs Year: (2024)
Ref_id:b7 Title: Graph mamba: Towards learning on graphs with state space models Year: (2024)
Ref_id:b8 Title: Understanding oversquashing in gnns through the lens of effective resistance Year: (2023)
Ref_id:b9 Title: Spectral graph neural networks meet transformers Year: (2023)
Ref_id:b10 Title: Beyond low-frequency information in graph convolutional networks Year: (2021-05)
Ref_id:b11 Title: Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in GNNs Year: (2022)
Ref_id:b12 Title:  Year: (2018)
Ref_id:b13 Title: Spectral networks and locally connected networks on graphs Year: (2013)
Ref_id:b14 Title: A note on over-smoothing for graph neural networks Year: (2020)
Ref_id:b15 Title: Message-Passing State-Space Models: Improving Graph Learning with Modern Sequence Modeling Year: (2025)
Ref_id:b16 Title: GRAND: Graph neural diffusion Year: (2021)
Ref_id:b17 Title: AntisymmetricRNN: A dynamical system view on recurrent neural networks Year: (2019)
Ref_id:b18 Title: Nagphormer: A tokenized graph transformer for node classification in large graphs Year: (2023)
Ref_id:b19 Title: Simple and Deep Graph Convolutional Networks Year: (2020-07)
Ref_id:b20 Title: Adaptive universal generalized pagerank graph neural network Year: (2021)
Ref_id:b21 Title: Topology-informed graph transformer Year: (2025)
Ref_id:b22 Title: Performers: A new approach to scaling transformers Year: (2020)
Ref_id:b23 Title: Principal neighbourhood aggregation for graph nets Year: (2020)
Ref_id:b24 Title: Convolutional neural networks on graphs with fast localized spectral filtering Year: (2016)
Ref_id:b25 Title: Polynormer: Polynomial-expressive graph transformer in linear time Year: (2024)
Ref_id:b26 Title: On over-squashing in message passing neural networks: The impact of width, depth, and topology Year: (2023)
Ref_id:b27 Title: Graph neural networks as gradient flows: understanding graph convolutions via energy Year: (2022)
Ref_id:b28 Title: Gbk-gnn: Gated bi-kernel graph neural networks for modeling both homophily and heterophily Year: (2022)
Ref_id:b29 Title: A generalization of transformer networks to graphs Year: (2020)
Ref_id:b30 Title: Benchmarking graph neural networks Year: (2023-01)
Ref_id:b31 Title: Graph neural networks with learnable structural and positional representations Year: (2022)
Ref_id:b32 Title: Long range graph benchmark Year: (2022)
Ref_id:b33 Title: Graph Adaptive Autoregressive Moving Average Models. In Forty-second International Conference on Machine Learning Year: (2025)
Ref_id:b34 Title: Adaptive message passing: A general framework to mitigate oversmoothing, oversquashing, and underreaching Year: (2025-07)
Ref_id:b35 Title: Mitigating over-smoothing and over-squashing using augmentations of forman-ricci curvature Year: (2024-11)
Ref_id:b36 Title: Cooperative Graph Neural Networks Year: (2024)
Ref_id:b37 Title: Sign: Scalable inception graph neural networks Year: (2020)
Ref_id:b38 Title: Diffusion Improves Graph Learning Year: (2019)
Ref_id:b39 Title: Spatiospectral graph neural networks Year: (2024)
Ref_id:b40 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b41 Title: Cin++: Enhancing topological message passing Year: (2023)
Ref_id:b42 Title: A new model for learning in graph domains Year: (2005)
Ref_id:b43 Title: Deep Learning for Dynamic Graphs: Models and Benchmarks Year: (2024)
Ref_id:b44 Title: Anti-Symmetric DGN: a stable architecture for Deep Graph Networks Year: (2023)
Ref_id:b45 Title: On oversquashing in graph neural networks through the lens of dynamical systems Year: (2025)
Ref_id:b46 Title: Non-dissipative Propagation by Randomized Anti-symmetric Deep Graph Networks Year: (2025)
Ref_id:b47 Title: Long Range Propagation on Continuous-Time Dynamic Graphs Year: (2024-07)
Ref_id:b48 Title: Drew: Dynamically rewired message passing with delay Year: (2023)
Ref_id:b49 Title: Stable architectures for deep neural networks Year: (2017)
Ref_id:b50 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b51 Title: Convolutional neural networks on graphs with chebyshev approximation, revisited Year: (2022)
Ref_id:b52 Title: A generalization of vit/mlp-mixer to graphs Year: (2023)
Ref_id:b53 Title: Port-Hamiltonian Architectural Bias for Long-Range Propagation in Deep Graph Networks Year: (2025)
Ref_id:b54 Title: Open graph benchmark: datasets for machine learning on graphs Year: (2020)
Ref_id:b55 Title: Fosr: First-order spectral rewiring for addressing oversquashing in gnns Year: (2023)
Ref_id:b56 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b57 Title: GOAT: A global transformer on large-scale graphs Year: (2023-07)
Ref_id:b58 Title: Rethinking graph transformers with spectral attention Year: (2021)
Ref_id:b59 Title: Finding global homophily in graph neural networks when meeting heterophily Year: (2022-07)
Ref_id:b60 Title: Towards quantifying long-range interactions in graph machine learning: a large graph dataset and a measurement Year: (2025)
Ref_id:b61 Title: The heterophilic graph learning handbook: Benchmarks, models, theoretical analysis, applications and challenges Year: (2024)
Ref_id:b62 Title: Graph inductive biases in transformers without message passing Year: (2023-07)
Ref_id:b63 Title: A fractional graph laplacian approach to oversmoothing Year: (2024)
Ref_id:b64 Title: Simplifying approach to node classification in graph neural networks Year: (2022)
Ref_id:b65 Title: Path neural networks: expressive and accurate graph neural networks Year: (2023)
Ref_id:b66 Title: Neural Network for Graphs: A Contextual Constructive Approach Year: (2009)
Ref_id:b67 Title: Revisiting over-smoothing and over-squashing using ollivier-ricci curvature Year: (2023)
Ref_id:b68 Title: Graph neural networks exponentially lose expressive power for node classification Year: (2020)
Ref_id:b69 Title: A critical look at the evaluation of GNNs under heterophily: Are we really making progress? Year: (2023)
Ref_id:b70 Title: Recipe for a general, powerful, scalable graph transformer Year: (2022)
Ref_id:b71 Title: A Survey on Oversmoothing in Graph Neural Networks Year: (2023)
Ref_id:b72 Title: Graph-coupled oscillator networks Year: (2022)
Ref_id:b73 Title: The graph neural network model Year: (2008)
Ref_id:b74 Title: Vn-egnn: E (3)-equivariant graph neural networks with virtual nodes enhance protein binding site identification Year: (2024)
Ref_id:b75 Title:  Year: (2024)
Ref_id:b76 Title: Exposition on over-squashing problem on GNNs: Current Methods, Benchmarks and Challenges Year: (2023)
Ref_id:b77 Title: Masked label prediction: Unified message passing model for semi-supervised classification Year: ()
Ref_id:b78 Title: Even sparser graph transformers Year: (2024)
Ref_id:b79 Title: Exphormer: Sparse transformers for graphs Year: (2023)
Ref_id:b80 Title: Encoding labeled graphs by labeling raam Year: (1993)
Ref_id:b81 Title: Understanding over-squashing and bottlenecks on graphs via curvature Year: (2022)
Ref_id:b82 Title: Graph attention networks Year: (2018)
Ref_id:b83 Title: Graph-mamba: Towards long-range graph sequence modeling with selective state spaces Year: (2024)
Ref_id:b84 Title: How powerful are spectral graph neural networks Year: (2022-07)
Ref_id:b85 Title: Dissecting the Diffusion Process in Linear Graph Convolutional Networks Year: (2021)
Ref_id:b86 Title: Simplifying Graph Convolutional Networks Year: (2019-06)
Ref_id:b87 Title: Nodeformer: A scalable graph structure learning transformer for node classification Year: (2022)
Ref_id:b88 Title: Sgformer: Simplifying and empowering transformers for large-graph representations Year: (2023)
Ref_id:b89 Title: How powerful are graph neural networks? Year: (2019)
Ref_id:b90 Title: Do transformers really perform bad for graph representation? Year: (2021)
Ref_id:b91 Title: Bigbird: Transformers for longer sequences Year: (2020)
Ref_id:b92 Title: Graphsaint: Graph sampling based inductive learning method Year: (2019)
Ref_id:b93 Title: Graph neural networks with heterophily Year: (2021-05)
Ref_id:b94 Title: Beyond homophily in graph neural networks: Current limitations and effective designs Year: (2020)
Ref_id:b95 Title: Mean test set score and std averaged over 4 random weight initializations on heterophilic datasets. The higher, the better. Model Roman-empire Amazon-ratings Minesweeper Tolokers Acc ↑ Acc ↑ AUC ↑ AUC ↑ Year: ()
