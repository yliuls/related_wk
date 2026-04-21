Title: Leveraging Diffusion Model as Pseudo-Anomalous Graph Generator for Graph-Level Anomaly Detection
Abstract: A fundamental challenge in graph-level anomaly detection (GLAD) is the scarcity of anomalous graph data, as the training dataset typically contains only normal graphs or very few anomalies. This imbalance hinders the development of robust detection models. In this paper, we propose Anomalous Graph Diffusion (AGDiff), a framework that explores the potential of diffusion models in generating pseudo-anomalous graphs for GLAD. Unlike existing diffusionbased methods that focus on modeling data normality, AGDiff leverages the latent diffusion framework to incorporate subtle perturbations into graph representations, thereby generating pseudoanomalous graphs that closely resemble normal ones. By jointly training a classifier to distinguish these generated graph anomalies from normal graphs, AGDiff learns more discriminative decision boundaries. The shift from solely modeling normality to explicitly generating and learning from pseudo graph anomalies enables AGDiff to effectively identify complex anomalous patterns that other approaches might overlook. Comprehensive experimental results demonstrate that the proposed AGDiff significantly outperforms several state-of-the-art GLAD baselines.

Section: Introduction
Graph-level anomaly detection (GLAD) (Akoglu et al., 2015;Qiao et al., 2024a;Liu et al., 2024b;Cai et al., 2024d) focuses on the fundamental challenge of identifying irregularities in graph-level data, where anomalous graphs significantly deviate from the normal graph distribution. GLAD is crucial across numerous domains, from detecting abnormal patterns in social networks (Yu et al., 2016;Zhang et al., 2025;Qiao & Pang, 2023) to identifying anomalous proteins in biological systems (Li et al., 2022). Unlike node-or edge-level anomaly detection (Duan et al., 2023;Qiao et al., 2024a;b;Pan et al., 2025), GLAD poses unique challenges as it requires modeling complex topological and geometric structures at the entire graph level.
The evolution of GLAD methods has witnessed several key developments. Classical graph kernel methods, such as the Weisfeiler-Lehman kernel (Shervashidze et al., 2011), random walk kernel (Vishwanathan et al., 2010), have established a foundation for GLAD by computing graph similarity matrices based on structural features. These approaches excel at capturing local topological patterns, but typically struggle with computational complexity. Recent deep learning based GLAD methods have emerged in both unsupervised and semi-supervised paradigms. Unsupervised methods typically employ graph neural networks (GNNs) (Kipf & Welling, 2017;Xu et al., 2019;Huang et al., 2023;Wan et al., 2024;Tu et al., 2025) to learn graph-level features for anomaly detection, with techniques such as one-class classification (Ruff et al., 2018;Qiu et al., 2022;Zhang et al., 2024), information bottleneck (Liu et al., 2023a), knowledge distillation (Ma et al., 2022), and graph reconstruction (Kim et al., 2024) to learn normality patterns without labeled anomalies. Semi-supervised approaches (Zhang et al., 2022;Xu et al., 2024) generally leverage limited graph anomalies to train a classifier as the anomaly detector. Even though only a small fraction of labeled anomalies are available, semi-supervised approaches have demonstrated remarkable performance improvement in detecting graph anomalies.
Despite these advancements, several critical limitations persist in existing approaches. Unsupervised methods focus on modeling normal graph distributions, they generally struggle to distinguish intricate or subtle anomalies, especially those near the boundaries of normal graphs, due to the lack of explicit supervised information. On the other hand, semi-supervised approaches can leverage limited labeled anomalies to enhance decision boundary learning. However, their effectiveness is constrained by the scarcity and diversity of labeled anomalous graphs, which limits their Latent diffusion-based pseudo-anomalous graph generation, and (3) Anomaly detector. The pre-training phase learns a structured latent space via a reconstruction model. The latent diffusion process generates pseudo-anomalous graphs by perturbing latent embeddings through a forward diffusion process and a reverse denoising process. Finally, the anomaly detector distinguishes between normal and generated pseudo-anomalous graphs through joint training.
generalizability to rare or unseen anomaly types.
To address these challenges, we propose Anomalous Graph Diffusion (AGDiff), a novel framework (see Figure 1) that leverages the generative capabilities of diffusion models to generate diverse pseudo-anomalous graphs for GLAD. By introducing a conditioned latent diffusion process within a well-trained graph representation learning model, AGDiff ensures the preservation of essential graph properties while imposing controllable perturbations in the latent space. Additionally, AGDiff employs a joint training paradigm that simultaneously optimizes pseudo-anomalous graph generation and anomaly detection. Our algorithm analysis (in Appendix A) highlights the advantages of AGDiff over traditional reconstruction-based methods by demonstrating that leveraging diverse pseudo-anomalous graphs enables a more robust and refined decision boundary for anomaly detection. By bridging the gap between generative modeling and discriminative learning, AGDiff offers a generalizable solution to the fundamental challenges of GLAD.
The primary contributions of this work are as follows:
1. We propose AGDiff, the first framework that explores the potential of diffusion models to mitigate the anomaly scarcity challenge in GLAD.
2. We propose a latent diffusion process with perturbation conditions to generate pseudo-anomalous graphs without relying on any labeled anomalies for improving decision boundary learning.
this section cite: ['b0', 'b52', 'b53', 'b31', 'b22', 'b9', 'b29', 'b36', 'b42', 'b19', 'b49', 'b14', 'b43', 'b40', 'b35', 'b34', 'b56', 'b28', 'b15', 'b54', 'b48']

Section: We demonstrate the superiority of AGDiff across extensive comparisons with state-of-the-art GLAD baselines on diverse graph benchmarks.

this section cite: []

Section: Related Work

this section cite: []

Section: Graph-level Anomaly Detection
Graph-level anomaly detection (GLAD) (Akoglu et al., 2015;Qiao et al., 2024a;Cai et al., 2024b) aims to identify graphs whose structural or attribute patterns deviate from the majority, with widespread real-world applications in fraud detection, bioinformatics, and cybersecurity. Traditional approaches, such as graph kernels (Borgwardt & Kriegel, 2005;Vishwanathan et al., 2010;Shervashidze et al., 2011), struggle to model the intricate dependencies within graph data due to their reliance on hand-crafted features. In recent years, the emergence of GNNs (Kipf & Welling, 2017;Xu et al., 2019;Li et al., 2023;Liu et al., 2023b;Cai et al., 2024a;c) has substantially advanced GLAD via their expressive graph representation learning capabilities. Latest GLAD approaches have explored knowledge distillation for global-local anomaly identification (Ma et al., 2022), graph transformation learning to mitigate representation collapse (Qiu et al., 2022), counterfactual graph generation for robust detection (Xiao et al., 2024a), spectral analysis to capture global graph anomaly properties (Dong et al., 2024), and explainable learning for interpretable anomaly detection (Liu et al., 2023a), etc. Additionally, semi-supervised learning (Zhang et al., 2022) is also studied to alleviate the data imbalance problem in GLAD.
Despite these advancements, existing GLAD methods still face significant limitations. For example, the reconstruction flip phenomenon identified by Kim et al. (2024) reveals that the common assumption in existing approaches, i.e., anomalies necessarily exhibit higher reconstruction errors, is not universally applicable. Moreover, existing approaches are trained with only normal graph data or just a small set of anomalies, which limits their generalizability to unseen anomalies. Consequently, in the proposed AGDiff framework, we aim to adaptively generate diverse pseudoanomalous graphs to enhance the capacity to detect subtle and complex anomalies while mitigating data imbalance issues. Furthermore, the generative process inherently improves interpretability by revealing how pseudo-anomalous graphs are generated through controlled perturbations. Compared to other approaches, our method delivers a robust and interpretable solution for GLAD problems.
this section cite: ['b0', 'b2', 'b42', 'b36', 'b19', 'b49', 'b23', 'b28', 'b34', 'b8', 'b54', 'b15']

Section: Diffusion Model
Diffusion models (Ho et al., 2020;Song et al., 2021;Yang et al., 2023) have emerged as a new paradigm for generative modeling, wherein data are gradually corrupted through a diffusion forward process and subsequently denoised through a learned reverse process. Early work by Ho et al. (Ho et al., 2020) demonstrated that diffusion processes could capture high-dimensional densities more effectively than traditional architectures such as GANs (Goodfellow et al., 2020;Wang et al., 2021;Cai et al., 2024e) and VAEs (Kingma, 2013). In the context of image anomaly detection, diffusion-based methods have demonstrated impressive capabilities in domains such as industrial defect inspection (Zhang et al., 2023;Tebbe & Tayyub, 2024) and medical anomaly detection (Wolleb et al., 2022;Bercea et al., 2024), as they excel by modeling intricate pixel-level dependencies, thereby enabling the identification of subtle deviations through the pixel-level reconstruction of images. Recent advances have investigated the diffusion frameworks for graph-structured data, demonstrating notable potential for tasks such as graph generation (Kong et al., 2023), node classification (Yang et al., 2024), and graph anomaly detection (Li et al., 2024;Liu et al., 2024a;Xiao et al., 2024b).
However, existing diffusion-based anomaly detection meth-ods (Wolleb et al., 2022;Tebbe & Tayyub, 2024;Li et al., 2024) primarily focus on modeling normality and expect anomalies to emerge naturally as reconstruction outliers, presupposing that anomalies deviate significantly in topology or node features. Such an assumption falters when anomalies are inherently subtle, especially in graphs where localized irregularities can be easily "diluted" by global reconstruction. Different from the existing approaches, we propose to leverage diffusion models as a pseudo-anomalous graph generator through controlled perturbations introduced during the diffusion process. We expect the generated pseudo graphs to resemble the normal graphs, so that they can provide explicit supervision for learning a robust anomaly detection model.
this section cite: ['b13', 'b38', 'b50', 'b13', 'b11', 'b44', 'b16', 'b55', 'b39', 'b45', 'b1', 'b20', 'b51', 'b21', 'b45', 'b39', 'b21']

Section: Preliminary
The diffusion model, e.g., denoising diffusion probabilistic model (DDPM) (Ho et al., 2020), is one of the generative model families, which typically contains a forward diffusion process that perturbs the original data x 0 into a noisy sample x t by progressively adding Gaussian noise over T time steps, and a reverse denoising process that attempts to recover x 0 from x t by removing the noise step-by-step. Specifically, the forward diffusion process is defined as:
x t = √ ᾱt x 0 + √ 1 -ᾱt ϵ t , ϵ t ∼ N (0, I).(1)
Here, ᾱt = t i=1 α i = t i=1 (1 -β i ), where β i denotes the noise variance schedule that imposed at each time step. The reverse denoising process can be understood as a series of learned denoising steps to recover the original data x 0 from the noise sample x t . At each time step t, x t-1 is reconstructed by:
x t-1 = 1 √ α x t - 1 -α t √ 1 -ᾱt ϵ θ (x t , t) + βz, (2
)
where ϵ θ (x t , t) is a learnable noise predict function, β = 1-ᾱt-1 ᾱt , and z ∼ N (0, I) is sampled from a normal Gaussian distribution. The training of diffusion models revolves around learning the reverse process that can accurately recover data from the noisy representation, where the objective can be expressed as:
min θ E t∼[1,T ],x0∼q(x0),ϵ∼N (0,I) ∥ϵ -ϵ θ (x t , t)∥ 2 2 . (3)
After training the model, the original data x 0 can be reconstructed by iteratively removing predicted noise ϵ θ (x t , t) from x t .
this section cite: ['b13']

Section: Methodology

this section cite: []

Section: Problem Formulation
Let G = (V, E) denote a graph where V represents the set of vertices and E ⊆ V × V represents the set of edges.
Each graph G is characterized by its adjacency matrix A ∈ {0, 1} n×n and node feature matrix X ∈ R n×d , where n = |V| is the number of nodes and d is the dimension of node features. In an unsupervised GLAD problem (Akoglu et al., 2015;Qiao et al., 2024a), given a normal graph set G = {G 1 , G 2 , . . . , G N } is given by sampling from an unknown normal graph distribution P normal . Our goal is to learn an anomaly detection function f : G → R that identifies graphs deviating from P normal . Specifically, for a test graph G test , we aim to estimate:
f (G test ) = P (G test / ∈ P normal |G test ).(4)
While some semi-supervised methods (Zhang et al., 2022) attempt to introduce a small fraction of real anomalies into training a classifier as the anomaly detector, the scarcity and limited diversity of anomalies make it challenging to generalize to unseen anomalous patterns. Therefore, we are curious whether the normality induced by normal graph distribution can be leveraged as a special kind of "supervised information" to facilitate the training of a more robust graph anomaly detector. In this work, we aim to generate a set of pseudo-anomaly graphs by introducing controlled perturbations for normal graphs under a latent diffusion framework. Then, we jointly train a graph anomaly detector with the generation framework so that the decision boundary learning and the pseudo-anomalous graph generation can be mutually refined. We will describe the details of our framework in the following sections.
this section cite: ['b0', 'b54']

Section: Modeling Normality via Variational Inference
In the context of graph-level anomaly detection, one of the foundational challenges is the modeling of the normal graph distribution, as anomalies are typically defined as deviations from the normality learned from a given normal graph set G normal . To achieve this, we first pre-train a graph representation learning model aiming at capturing the normality of graphs. Particularly, we leverage variational inference to learn a probabilistic mapping from graph space to continuous latent space, as it naturally provides a well-regularized manifold for subsequent perturbation-based anomaly generation. For a given graph set G = {A, X}, we approximate the posterior distribution over the latent representation Z ∈ R n×dz as:
q(Z|X, A) = n i=1 q(z i |X, A), w.r.t q(z i |X, A) = N (z i |µ i , diag(σ 2 i )),(5)
where µ and σ are parameterized by µ = GNN µ (X, A) and log σ = GNN σ (X, A), respectively. By using the reparameterization trick to enable gradient propagation, we sample Z = µ+σ ⊙τ , where τ ∼ N (0, I). The reconstruction process involves recovering both structural (denoted by Â) and attribute (denoted by X) information for graphs:
Â = T (ZZ ⊤ ), X = D(Z),(6)
where T (•) denotes a Sigmoid transformation function, and D(•) represents an MLP-based decoder for attribute reconstruction. The model is optimized by minimizing:
L pretrain =ℓ attr r + ℓ edge r + ℓ KL = N i=1 (∥X i -Xi ∥ 2 F + H(A i , Âi ) -KL(q(Z i |X i , A i )|P(Z))),(7)
where H(•) denotes the binary cross-entropy loss for edge reconstruction, and the KL divergence term regularizes the learned distribution towards a prior P(Z) = i N (z i |0, I), which encourages a smooth and well-structured latent space and prevents the model from overfitting to a narrow or degenerate latent manifold (Kipf & Welling, 2016).
this section cite: ['b18']

Section: Generating Anomalous Graphs via Latent Diffusion
Building on a well-structured latent space that effectively captures normal graph patterns, we propose a novel approach that utilizes latent diffusion models to generate pseudo-anomalous graphs. This approach represents a substantial departure from conventional diffusion-based anomaly detection methods, which typically aim to model the normality of data via diffusion models. Instead, we exploit the generative power of diffusion models to generate diverse pseudo-anomalous graphs by introducing controlled perturbations during the latent diffusion process. The smooth and continuous geometry of the latent space provides a foundation for this process, enabling the preservation of critical structural properties and meaningful deviations from normality. Our key insights are: (1) the continuous nature of the latent space makes it particularly suitable for the diffusion process, and (2) the asymptotic nature of the diffusion process allows fine-grained control over the deviation from normality.
Given a normal graph set G = {X, A}, we first obtain its latent representation Z = E(X, A) through the pre-trained model. To generate controlled perturbations, we design a conditional latent diffusion process that consists of two key components: a forward process that gradually adds noise to the latent representation, and a reverse process that learns to denoise while preserving essential graph properties. The forward process progressively corrupts the latent representation by injecting Gaussian noise over T time steps:
z t = √ ᾱt z 0 + √ 1 -ᾱt ϵ t , ϵ t ∼ N (0, I),(8)
where z 0 is initialized by Z, and ᾱt =
t i=1 α i = t i=1 (1 -β i ) determines the noise schedule through β i .
This schedule is crucial as it determines the degree of perturbation at each step.
A key challenge in generating pseudo-anomalous graphs is maintaining the balance between structural preservation and anomaly injection. To address this, we design a conditional reverse process:
z t-1 = 1 √ α z t - 1 -α t √ 1 -ᾱt ϵ θ (z t , t, c) + βv, (9
)
where ϵ θ (z t , t, c) is our conditional noise prediction network, β = 1-ᾱt-1 ᾱt , and v ∼ N (0, I). The condition vector c is obtained via a perturbation condition model τ ω to add auxiliary noise information to the generation process:
c = τ ω (z 0 ) = σ(W c (z 0 + η) + b c ),(10)
where η ∼ N (0, I) is a Gaussian noise vector that introduces perturbations to the initial latent representation, and the learnable weight matrix W c and bias vector b c transform the perturbed representation to a more expressive feature space through a non-linear activation function σ(•). The condition vector c is concatenated with the latent variable z t at each denoising step to introduce controlled variations during the diffusion process.
The condition vector c is crucial for guaranteeing the generation quality of pseudo graph anomalies. If we directly use the perturbed normal data (e.g., via random noise η) as pseudo anomalies, it would constrain the diversity of anomalous patterns the model encounters, as such perturbations are static and lack adaptability during training. In contrast, we propose to perturb the initial latent embedding through a learnable perturbation transformation c, which injects additional variability into the latent diffusion process. The perturbations are learnable and dynamically adjusted during joint training with the anomaly detector. As the anomaly detector improves, the perturbation mechanism evolves and generates increasingly sophisticated and diverse pseudo anomalies to refine the decision boundary. This ensures that the denoising network is conditioned to deviate from purely "normal" reconstructions and allows the model to be exposed to a broader spectrum of potential anomalies, thereby enhancing its robustness and generalization. The latent diffusion model is then optimized by minimizing:
L diff = E z0,ϵ,t,c ∥ϵ -ϵ θ (z t , t, c)∥ 2 2 . (11
)
Through the conditional diffusion process, we can iteratively generate perturbed latent features z0 , i.e., z t → z t-1 → • • • → z0 for each graph. Therefore, the pseudo-anomalous graphs can be obtained through the pre-trained decoder via: Ã = T ( Z Z⊤ ) and X = D( Z), where Z = {z
(i) 0 } N i=1 .
this section cite: []

Section: Detecting Anomalies from Subtle Deviations
Conventional semi-supervised GLAD approaches generally struggle in scenarios where real anomalous samples are very
Algorithm 1 AGDiff Input: Input graph set G, number of GIN layers K, learning rate ρ, diffusion time steps T , training epochs Epochs. Output: The anomaly detection scores. 1: Initialize the network parameters; 2: Pre-train the representation learning model by minimizing Eq. (7); 3: Freeze the network parameters of the pre-trained model; 4: for epoch = 1 to Epochs do 5: Sample a mini-batch of graphs from graph dataset G; 6: Obtain latent embeddings Z from the pre-trained encoder; 7: Perform forward diffusion process on Z via Eq. (8); 8: Obtain the condition vector via Eq. (10); 9: Generate pseudo-anomalous embeddings Z through the reverse diffusion process via Eq. (9); 10: Decode Z via pre-trained decoder to generate pseudoanomalous graphs set G; 11: Predict scores for the normal and generated pseudoanomalous graphs via Eq. (12); 12:
Jointly update the parameters of the latent diffusion model and anomaly detector by minimizing Eq. ( 14); 13: end for 14: Compute anomaly detection scores for test graphs via the trained classifier f θ ; 15: Return: The anomaly detection scores. rare or even unavailable, as their reliance on labeled anomalies limits their capacity to generalize to unseen abnormal patterns. This challenge becomes particularly pronounced in real-world scenarios, where anomalies are inherently rare and diverse. To overcome this limitation, we propose a joint learning framework that leverages generated pseudoanomalous graphs to facilitate the training of the anomaly detector. These pseudo-anomalous graphs serve as proxies for real anomalies, which enables the model to learn more robust, adaptive decision boundaries that capture subtle deviations from normality.
In practice, we employ a GIN-based anomaly detector h ϕ (•) to distinguish between normal graphs and pseudoanomalous graphs, which is defined by:
h ϕ (G) = MLP(GIN(X, A)),(12)
where h ϕ (•) parameterized by ϕ is comprised of a GINbased backbone network GIN(•) and an MLP-based projector MLP(•). We employ a following binary cross-entropy loss L cls to train the anomaly detector:
L cls = - 1 |G ∪ G| G∈G∪ G(y G log h ϕ (G) + (1 -y G ) log(1 -h ϕ (G))),(13)
where G and G denote the normal and pseudo-anomalous graph sets, respectively. Note that we set y G = 1, ∀G ∈ G, and y G = 0, ∀G ∈ G to train the anomaly detector. In particular, we jointly train the latent diffusion model and the anomaly detector by minimizing:
L = L cls + λL diff (14
)
where λ is the hyper-parameter that controls the trade-off between two objectives. L diff guides the generation of pseudo graph anomalies, and L cls enhances the discriminative power of the anomaly detector. In the inference phase, we can obtain the anomaly score of each test graph sample via the trained anomaly detector. Algorithm 1 summarizes the training procedure of AGDiff. The joint learning framework offers several unique advantages:
1. The latent diffusion model learns to generate increasingly challenging pseudo-anomalies that explore the decision boundary of the anomaly detector.
2. The gradient of the detector directs the diffusion process toward generating more informative pseudo-anomalous samples.
3. The iterative refinement between generation and detection leads to a more robust anomaly detector.
this section cite: []

Section: Computational Complexity Analysis
For a dataset of N graphs, each with an average of m nodes (feature dimension d), |E| edges, and latent dimension d z , the AGDiff framework operates in three phases:
1. Pre-training: An L-layer GIN is employed as the backbone network in the pre-training, where the overall complexity is O(N L(|E|d+md 2 )) due to the message aggregation (O(|E|d)) and feature transformation (O(md 2 )).
2. Pseudo Anomaly Generation: This phase involves the computation of conditional vector (O(N d 2 z )) and a Tstep latent diffusion process (O(N T md 2 z )) across all graphs.
3. Decoding and Anomaly Detection: Decoding involves the computation of node attributes and adjacency matrices from latent embeddings, which results in time complexity of O(N |E|d z ) when we apply a negative sampling strategy in practice. The computational complexity of the anomaly detector is similar to the pre-training stage, i.e., (O(N L(|E|d + md 2 ))), due to their similar network structure.
Therefore, the overall computational complexity of AGDiff is approximately O(N L(|E|d + md 2 ) + N (T m + 1)d 2 z + N |E|d z ), which is comparable with other state-of-the-art baselines such as SIGNET, MUSE, DO2HSC. In addition, potential optimizations such as using parallel computation further enhance efficiency.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets. We conduct experiments with two types of graph benchmarks, including
• Moderate-Scale Datasets: MUTAG, DD, COX2, and ER MD. These datasets primarily consist of molecular graphs and bioinformatics networks, where nodes and edges represent molecular structures or protein interactions.
• Large-Scale Imbalanced Datasets: SW-620, MOLT-4, PC-3, and MCF-7. Each graph in these datasets represents a chemical compound, with labels indicating whether it exhibits anti-cancer activity. The datasets are highly imbalanced as active compounds form only a small fraction of the total samples, which makes them well-suited for evaluating the robustness of anomaly detection methods in real-world, unbalanced settings.
We describe the details about the datasets and their characteristics in Appendix B. Note that we follow the settings in existing works (Ma et al., 2022;Zhang et al., 2022;Liu et al., 2023a) to treat the minority class as the anomalous class to better align with real-world scenarios where anomalies often correspond to rare events or unusual patterns.
Implementation Details. Due to the length limitation of the paper, please refer to Appendix C for the implementation details of the experiment, including the data split, network architecture, hyper-parameter setting, baseline setting, and computing resources.
this section cite: ['b28', 'b54']

Section: Compared Baselines.
To evaluate the effectiveness of the proposed AGDiff method, we compared it with two types of GLAD baselines, including: (1) Graph Kernel Methods: Short-Path (SP) kernel (Borgwardt & Kriegel, 2005), Weisfeiler-Lehman (WL) kernel (Shervashidze et al., 2011), NH (Hido & Kashima, 2009), Random Walk (RW) kernel (Vishwanathan et al., 2010), and (2) GNN-based GLAD Methods: OCGIN (Zhao & Akoglu, 2023), OCGTL (Qiu et al., 2022), GLocalKD (Ma et al., 2022), iGAD (Zhang et al., 2022), SIGNET (Liu et al., 2023a), MUSE (Kim et al., 2024), DO2HSC (Zhang et al., 2024).
Evaluation Metrics. To evaluate the anomaly detection performance of each method, we utilize two widely adopted metrics: Area Under the Curve (AUC) and F1-Score. The experimental results are reported as the mean values and  (Borgwardt & Kriegel, 2005) 67.52±0.00 60.00±0.00 82.73±0.00 76.09±0.00 54.08±0.00 49.32±0.00 40.92±0.00 37.74±0.00 WL (Shervashidze et al., 2011) 60.00±0.00 89.12±0.00 81.57±0.00 74.64±0.00 49.32±0.00 50.19±0.00 37.74±0.00 45.71±0.00 NH (Hido & Kashima, 2009) 79.97±0.40 76.00±0.00 81.61±0.32 73.91±0.65 61.41±0.82 56.44±1.03 51.55±2.00 50.19±0.92 RW (Vishwanathan et al., 2010) 86.98±0.00 83.33±0.00 OM OM 52.43±0.00 30.00±0.00 78.94±0.00 65.96±0.00 OCGIN (Zhao & Akoglu, 2023) 74.66±1.68 62.95±0.00 66.59±4.44 56.12±0.00 59.64±5.78 47.95±0.00 47.63±3.59 50.94±1.89 OCGTL (Qiu et al., 2022) 87.04±1.74 80.00±0.00 77.52±0.43 71.65±0.73 60.42±0.90 55.62±5.24 72.67±0.20 67.17±0.92 GLocalKD (Ma et al., 2022) 90.59±0.61 86.17±0.91 80.59±0.00 73.48±0.57 51.42±0.66 51.24±0.60 78.94±0.00 70.21±0.00 iGAD (Zhang et al., 2022) 92.58±1.25 85.20±2.30 74.83±2.30 70.39±2.60 72.09±2.29 61.94±1.09 80.56±2.57 74.57±2.45 SIGNET (Liu et al., 2023a) 87.73±2.45 73.07±4.11 59.53±3.45 56.76±3.47 52.80±2.53 20.24±4.92 77.02±1.07 77.06±1.70 MUSE (Kim et al., 2024) 83  across all datasets. Additionally, we observe that the semisupervised method, iGAD, significantly outperforms other baselines (e.g., 86.04% AUC on PC-3) as it leverages partial real anomalies to refine its decision boundary. Nevertheless, AGDiff still surpasses iGAD despite being an unsupervised method across all datasets. For example, AGDiff achieves AUCs of 91.60% and 94.32% on SW-620 and PC-3, respectively, which improves by 5.78% and 8.28% over iGAD. The reason for this improvement is that although iGAD utilizes a certain amount of labeled anomalies to enhance the decision boundary learning, they may not be sufficiently representative of real anomalies due to the limited availability and diversity of anomalies in a large-scale imbalance scenario. Different from semi-supervised iGAD, the diverse pseudo-anomalous graphs generated from AGDiff provide rich self-supervised signals for training. This enables the anomaly detector to distinguish between normal and anomalous graphs with subtle deviations, which effectively overcomes the limited diversity of available anomalies in imbalanced scenarios.
this section cite: ['b2', 'b36', 'b12', 'b42', 'b57', 'b34', 'b28', 'b54', 'b15', 'b56', 'b2', 'b36', 'b12', 'b42', 'b57', 'b34', 'b28', 'b54', 'b15']

Section: Scoring Distribution Analysis
To verify the effectiveness of the pseudo-anomalous graphs generated by AGDiff in facilitating anomaly detection, we analyze the scoring distributions of normal, pseudoanomalous, and real anomalous graphs on the ER MD and PC-3 datasets, as shown in Figure 2. In ER MD, pseudoanomalous graphs exhibit a certain overlap with normal graphs, which indicates that generated pseudo-graph anoma- lies maintain structural similarities to normal graphs while introducing subtle deviations via the latent diffusion process. In contrast, real anomalous graphs are more distinctly separated from normal graphs, which implies that incorporating pseudo-anomalous graphs during training improves the discriminative ability of the anomaly detector to identify real anomalies. Moreover, in PC-3, we observe that the scoring distribution of pseudo-anomalous graphs primarily lies in a low-score region, whereas real anomalous graphs remain well-separated from normal graphs in the test stage. This separation highlights the role of AGDiff in modeling the diversity of potential anomalies, which generates pseudo-graph anomalies to provide rich anomalous signals to facilitate robust decision boundary learning, particularly in large-scale imbalanced scenarios.
this section cite: []

Section: Parameter Analysis
To evaluate the impact of key hyper-parameters on the anomaly detection performance of AGDiff, we conduct a sensitivity analysis on hyper-parameter λ, which balances the contribution of the diffusion loss. Figure 3 shows the trend of AUC and F1-Score under different values of λ on MUTAG and ER MD. As shown in the figure, a moderate λ generally improves performance as it ensures AGDiff generates informative pseudo-anomalous graphs, which in turn enhances the discriminative ability of the model. However, excessively large λ prioritizes diffusion modeling over classification, which tends to make the generated pseudoanomalous graphs overly resemble normal graphs, thereby increasing the difficulty of training the anomaly detector. Conversely, a very small λ can also degrade the anomaly detection performance because it limits the diversity of the generated pseudo anomalies.
this section cite: []

Section: Ablation Study
We conduct an ablation study to analyze the impact of each component in AGDiff, including the pre-training strategy, condition embedding, and the latent diffusion module. Particularly, we utilize the reconstruction error to detect anomalies when we remove the conditioned latent diffusion module, as pseudo-anomalous graphs are not available in this variant. Table 3 presents the experimental results on MU-TAG and ER MD, where we can have the following observations. First, the significant performance degradation when the pertaining strategy indicates the importance of pretraining to our framework, as it provides a well-structured latent space for modeling the normality, so that the model is able to generate informative pseudo-anomalous graphs. Second, we can observe that removing the condition embedding also significantly degrades the anomaly detection performance. This is because the condition embedding injects additional variability via introducing a noise-augmented condition vector, which encourages the model to generate diverse pseudo-anomalous samples that better capture the nuances of anomalous patterns. Otherwise, the generated graphs may closely resemble normal graphs, which leads to the difficulty of distinguishing them. Lastly, we observe the most severe performance decline in the reconstruction-based variants, which highlights the limitations of reconstructionbased methods, which generally struggle to differentiate between normal and anomalous graphs when structural deviations are subtle. In contrast, our approach leverages the latent diffusion process to explicitly model pseudo-anomalous variations, providing more robust and discriminative learning signals for training the anomaly detector.
this section cite: []

Section: More Experimental Analysis
We provide a more experimental analysis of the proposed AGDiff method in the Appendix, such as the algorithm analysis (Appendix A), visualization results (Appendix D), more parameter analysis (Appendix E), and more ablation study (Appendix F), etc.
this section cite: []

Section: Conclusion
In this work, we introduced Anomalous Graph Diffusion (AGDiff), a novel framework that addresses the scarcity of anomalous data in graph-level anomaly detection. By introducing a latent diffusion module to inject controlled perturbations into graph representations, AGDiff is able to generate diverse pseudo-anomalous graphs that closely resemble normal ones while exhibiting subtle deviations. A GNN-based anomaly detector is then jointly trained with the latent diffusion module to distinguish the pseudo-anomalous and normal graphs. Moreover, we theoretically demonstrated the effectiveness of these perturbed graphs generated via AGDiff for facilitating the learning of a more robust decision boundary.
Empirical evaluations on multiple graph benchmarks validate the superiority of AGDiff against stateof-the-art GLAD baselines. Two limitations of this work are: (1) It assumes a sufficiently representative distribution of normal graphs, which may not hold in shifting or highly heterogeneous environments. (2) While AGDiff can generate pseudo graph anomalies to enhance decision boundary training, it is currently limited to static graphs. Future research can focus on exploring more flexible noise scheduling diffusion approaches, as well as investigating the challenging GLAD tasks in heterogeneous environments or dynamic graph settings.
this section cite: []

Section: References
Ref_id:b0 Title: Graph based anomaly detection and description: a survey Year: (2015)
Ref_id:b1 Title: Diffusion models with implicit guidance for medical anomaly detection Year: (2024)
Ref_id:b2 Title: Shortest-path kernels on graphs Year: (2005)
Ref_id:b3 Title: Deep graph-level clustering using pseudo-label-guided mutual information maximization network. Neural Computing and Applications Year: (2024)
Ref_id:b4 Title: Towards effective federated graph anomaly detection via self-boosted knowledge distillation Year: (2024)
Ref_id:b5 Title: Dual contrastive graph-level clustering with multiple cluster perspectives alignment Year: (2024)
Ref_id:b6 Title: Lg-fgad: An effective federated graph anomaly detection framework Year: (2024)
Ref_id:b7 Title: Wasserstein embedding learning for deep clustering: A generative approach Year: (2024)
Ref_id:b8 Title: Rayleigh quotient graph neural networks for graph-level anomaly detection Year: (2024)
Ref_id:b9 Title: Graph anomaly detection via multi-scale contrastive learning networks with augmented view Year: (2023)
Ref_id:b10 Title: Your data is not perfect: Towards cross-domain out-ofdistribution detection in class-imbalanced data Year: (2025)
Ref_id:b11 Title: Generative adversarial networks Year: (2020)
Ref_id:b12 Title: A linear-time graph kernel Year: (2009)
Ref_id:b13 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b14 Title: Federated graph semantic and structural learning Year: (2023)
Ref_id:b15 Title: Rethinking reconstruction-based graph-level anomaly detection: Limitations and a simple remedy Year: (2024)
Ref_id:b16 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b17 Title: A method for stochastic optimization Year: (2014)
Ref_id:b18 Title: Variational graph auto-encoders Year: (2016)
Ref_id:b19 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b20 Title: Autoregressive diffusion model for graph generation Year: (2023)
Ref_id:b21 Title: Diffgad: A diffusion-based unsupervised graph anomaly detector Year: (2024)
Ref_id:b22 Title: Graph representation learning in biomedicine and healthcare Year: (2022)
Ref_id:b23 Title: Cross-view graph matching guided anchor alignment for incomplete multi-view clustering Year: (2023)
Ref_id:b24 Title: Data augmentation for supervised graph outlier detection via latent diffusion models Year: (2024)
Ref_id:b25 Title: Towards self-interpretable graph-level anomaly detection Year: (2023)
Ref_id:b26 Title: Dink-net: Neural clustering on large graphs Year: (2023)
Ref_id:b27 Title: Arc: A generalist graph anomaly detector with in-context learning Year: (2024)
Ref_id:b28 Title: Deep graph-level anomaly detection by glocal knowledge distillation Year: (2022)
Ref_id:b29 Title: A label-free heterophily-guided approach for unsupervised graph fraud detection Year: (2025)
Ref_id:b30 Title: Scikit-learn: Machine learning in Python Year: (2011)
Ref_id:b31 Title: Truncated affinity maximization: One-class homophily modeling for graph anomaly detection Year: (2023)
Ref_id:b32 Title: Deep graph anomaly detection: A survey and new perspectives Year: (2024)
Ref_id:b33 Title: Generative semi-supervised graph anomaly detection Year: (2024)
Ref_id:b34 Title: Raising the bar in graph-level anomaly detection Year: (2022)
Ref_id:b35 Title: Deep one-class classification Year: (2018)
Ref_id:b36 Title: Weisfeiler-lehman graph kernels Year: (2011)
Ref_id:b37 Title: Grakel: A graph kernel library in Python Year: (2020)
Ref_id:b38 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b39 Title: Dynamic addition of noise in a diffusion model for anomaly detection Year: (2024)
Ref_id:b40 Title: Wage: Weight-sharing attribute-missing graph autoencoder Year: (2025)
Ref_id:b41 Title: Visualizing data using t-sne Year: (2008)
Ref_id:b42 Title: Graph kernels Year: (2010)
Ref_id:b43 Title: S3gcl: Spectral, swift, spatial graph contrastive learning Year: (2024)
Ref_id:b44 Title: Generative partial multi-view clustering with adaptive fusion and cycle consistency Year: (2021)
Ref_id:b45 Title: Diffusion models for medical anomaly detection Year: (2022)
Ref_id:b46 Title: Motif-consistent counterfactuals with adversarial refinement for graph-level anomaly detection Year: (2024)
Ref_id:b47 Title: Counterfactual data augmentation with denoising diffusion for graph anomaly detection Year: (2024)
Ref_id:b48 Title: Gladformer: A mixed perspective for graph-level anomaly detection Year: (2024)
Ref_id:b49 Title: How powerful are graph neural networks? Year: (2019)
Ref_id:b50 Title: Diffusion models: A comprehensive survey of methods and applications Year: (2023)
Ref_id:b51 Title: Directional diffusion models for graph representation learning Year: (2024)
Ref_id:b52 Title: A survey on social media anomaly detection Year: (2016)
Ref_id:b53 Title: Semi-supervised multi-view clustering with active constraints Year: (2025)
Ref_id:b54 Title: Dual-discriminative graph neural network for imbalanced graph-level anomaly detection Year: (2022)
Ref_id:b55 Title: Unsupervised surface anomaly detection with diffusion probabilistic model Year: (2023)
Ref_id:b56 Title: Deep orthogonal hypersphere compression for anomaly detection Year: (2024)
Ref_id:b57 Title: On using classification datasets to evaluate graph outlier detection: Peculiar observations and new insights Year: (2023)
