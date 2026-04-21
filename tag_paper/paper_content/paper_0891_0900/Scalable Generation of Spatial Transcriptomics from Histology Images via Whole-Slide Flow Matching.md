Title: Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching
Abstract: Spatial transcriptomics (ST) has emerged as a powerful technology for bridging histology imaging with gene expression profiling. However, its application has been limited by low throughput and the need for specialized experimental facilities. Prior works sought to predict ST from wholeslide histology images to accelerate this process, but they suffer from two major limitations. First, they do not explicitly model cell-cell interaction as they factorize the joint distribution of wholeslide ST data and predict the gene expression of each spot independently. Second, their encoders struggle with memory constraints due to the large number of spots (often exceeding 10,000) in typical ST datasets. Herein, we propose STFlow, a flow matching generative model that considers cell-cell interaction by modeling the joint distribution of gene expression of an entire slide. It also employs an efficient slide-level encoder with local spatial attention, enabling whole-slide processing without excessive memory overhead. On the recently curated HEST-1k and STImage-1K4M benchmarks, STFlow substantially outperforms state-of-the-art baselines and achieves over 18% relative improvements over the pathology foundation models.

Section: Introduction
Compared to the early days of bulk RNA sequencing, recent advancements in spatial transcriptomics (ST) technology offer a novel approach to molecular profiling within the spatial context of tissues, providing insights into cellular interactions and the microenvironment (Ståhl et al., 2016;Xiao & Yu, 2021). One of the promising clinical applications of ST is the prediction of biomarkers in digital pathology, often visualized in hematoxylin and eosin (H&E)-stained wholeslide images (WSIs), by analyzing the gene expression levels in relation to the tissue morphology (Levy-Jurgenson et al., 2020;Zhang et al., 2022). However, the conventional ST methods (Moffitt et al., 2018;Eng et al., 2019;Ståhl et al., 2016) suffer from low throughput and need of specialized equipment, limiting their application compared to standard histology imaging.
To address this, recent works resort to deep learning to predict spatially-resolved gene expression from H&E images. As illustrated in Figure 1(a), a histology image is segmented into small spots, with the objective of predicting the gene expression with the spot image and the coordinate. This line of research has achieved promising results using an image foundation model to encode local spot-level features (Chen et al., 2024b;He et al., 2020;Ciga et al., 2022;He et al., 2016), but they neglect the utilization of spatial dependencies between spots. To bridge this gap, some studies introduce an additional slide-level encoder to incorporate global spatial context (Xu et al., 2024;Chung et al., 2024;Pang et al., 2021;Zeng et al., 2021) with an vision transformer.
Despite their initial success, these methods are limited by (1) high computational complexity: the exhaustive attention mechanism among all spots leads to significant computational overhead, making it impractical for standard gigapixel slides, which contain tens of thousands of spots (Campanella et al., 2019;Lu et al., 2021); (2) weak utilization of spatial dependencies: these methods typically encode coordinates as positional embeddings, making them sensitive to numerical noise and variations in coordinate scales caused by batch effects. (3) overlooking cell-cell interaction, i.e., certain genes regulating the expression of genes in other cells (Figure 1(b)) (Li et al., 2022;Biancalani et al., 2021).
In light of this, we propose STFlowfoot_0 , a flow matching-based model that reformulates the original regression task as a generative modeling problem. Instead of performing onestep regression, we model the joint distribution over the whole-slide gene expression through an iterative refinement  process, as depicted in Figure 1(c). Each refinement step is guided by the flow matching framework, where the predicted gene expression serves as context for the subsequent step. This enables explicit modeling of cell-cell interactions, leading to more biologically meaningful predictions. The denoising network employs a frame averaging (FA)based (Puny et al., 2021) spatial model with E(2)-invariant spatial attention and achieves great efficiency by modeling the local context of each spot.
T P 5 3 B R C A 1 E G F R Y E G F A A D H 1 B A Q P 1 G S T Denoiser Denoiser 𝒀 𝟏 … 𝒀 𝑻 … … … T P 5 3 B R C A 1 E G F R Y E G F A A D H 1 B A Q P 1 G S T T P 5 3 B R C A 1 E G F R Y E G F A A D H 1 B A Q P 1 G S T 𝒀 !,! 𝒀 !," 𝒀 !,# 𝒀 !,$ 𝒀 ",! 𝒀 "," 𝒀 ",# 𝒀 ",$ 𝒀 &,! 𝒀 &," 𝒀 &,# 𝒀 &,$ T P 5 3 B R C A 1 Y E G F A A D H 1 B A Q P 1 G S T E G F R
We evaluate STFlow on HEST-1k (Jaume et al., 2024) and STImage-1K4M (Chen et al., 2024a), two large-scale ST-WSI collections comprising a total of 17 benchmark datasets. Compared to five spot-based and three slide-based methods, STFlow consistently outperforms all baselines and achieves an 18% average relative improvement over the pathology foundation models. It also excels in the prediction of 4 biomarker genes, highlighting its clinical potential. Moreover, our proposed architecture offers orders-of-magnitude faster runtime and lower memory cost than existing slidebased approaches.
this section cite: ['b42', 'b48', 'b25', 'b52', 'b35', 'b12', 'b42', 'b17', 'b8', 'b18', 'b50', 'b7', 'b37', 'b51', 'b4', 'b32', 'b26', 'b1', 'b20']

Section: Related Work WSI-based spatial gene expression prediction
Rapid advances in spatial transcriptomics (ST) (Li & Wang, 2021) have enabled the detection of RNA transcript spatial distribution at sub-cellular resolution (Moffitt et al., 2018;Codeluppi et al., 2018;Eng et al., 2019;Ståhl et al., 2016;Stickels et al., 2021). Machine learning-based approaches have recently shown promising results in predicting expression from histology image (Lee et al., 2023). The previous studies fall into two categories: (1) spot-based approaches which solely encode the spot and predict the gene expression individually, i.e., modeling p(Y i |I i ) (He et al., 2020;Pang et al., 2021;Chen et al., 2024b;Ciga et al., 2022;Xie et al., 2023). Some of these methods leverage foundation models pretrained on large-scale digital pathology datasets, achieving promising results in gene expression prediction (Jaume et al., 2024). One concurrent work (Zhu et al., 2025) leverages diffusion models for ST gene expression generation, but they still treat each spot independently. In contrast, our generative model operates at the whole-slide level, explicitly modeling the joint distribution of genes across spots. (2) slide-based approaches which incorporate the slide-level context and predict the gene expression of each spot individually, i.e., modeling p(Y i |I 0 , • • • , I N ) (Pang et al., 2021;Zeng et al., 2021;Jia et al., 2024;Xu et al., 2024;Chung et al., 2024). The main idea of these methods is to aggregate the representations of other spots after the image encoders extract each spot's features. The key difference between our proposed STFlow and previous methods is that STFlow explicitly utilizes gene-gene dependency between cells for prediction using a generative model, i.e., modeling the joint distribution p
(Y 0 , • • • , Y N |I 0 , • • • , I N ).
Flow matching Flow matching is a generative modeling paradigm (Lipman et al., 2022;Albergo & Vanden-Eijnden, 2022;Liu et al., 2022;Jing et al., 2024;Nori & Jin, 2024) that has shown impressive results across various modalities, including images and biomolecules. The objective is to approximate the marginal vector field of the time-dependent probability path using a neural network. In this work, we reformulate the gene expression regression as a generative task and apply the flow matching since (1) its iterative denoising scheme allows us to incorporate the gene expression within the modeling, and (2) it offers flexibility in selecting a gene expression-specific prior distribution, e.g., zero-inflated negative binomial distribution.
this section cite: ['b27', 'b35', 'b9', 'b12', 'b42', 'b43', 'b24', 'b17', 'b37', 'b8', 'b49', 'b20', 'b54', 'b37', 'b51', 'b21', 'b50', 'b7', 'b29', 'b0', 'b31', 'b22', 'b36']

Section: Geometric deep learning
Geometric deep learning has achieved significant success in chemistry, physics, and biology (Bronstein et al., 2021;Zhang et al., 2023;Liu et al., 2023). The key to this success lies in generating invariant representations for 3D structures, such as molecule conformation, that remain consistent under E(n) transformations, where n represents the dimension of the Euclidean space. E(n) transformations include translations, rotations, and reflections. Previous methods achieve invariance by leveraging invariant features (Satorras et al., 2021;Schütt et al., 2018;Gasteiger et al., 2021) or employing equivariant transformations, such as irreducible representations (Fuchs et al., 2020;Liao & Smidt, 2022;Weiler & Cesa, 2019) and frame averaging (FA) (Puny et al., 2021;Huang et al., 2024). The architecture of the denoiser encodes the spatial context of WSIs using an FA-based Transformer architecture, designed to produce invariant representations for each spot, regardless of any E(2) transformations.
this section cite: ['b3', 'b53', 'b30', 'b40', 'b41', 'b15', 'b14', 'b28', 'b47', 'b38', 'b19']

Section: Method
In this section, we introduce STFlow, a flow matching framework for modeling the joint distribution of gene expression across spots, along with an E(2)-invariant denoiser for capturing spatial dependencies. We first introduce the necessary background in Section 3.1 and elaborate on the learning framework in Section 3.2. The introduction of architecture is provided in Section 3.3.
this section cite: []

Section: Preliminaries
Problem Formulation An H&E-stained WSI is segmented into a set of patches, which can be represented as (C, I, Y ), with coordinates C ∈ R N ×2 , spot images I ∈ R N ×3×H×W , and gene expression levels Y ∈ R N ×G , where N is the number of spots, G is the number of genes, and H, W indicate the image dimensions. Each element in Algorithm 1 STFlow: Training Require: Training WSIs (C,
I, Y ) Sample prior Y 0 ∼ Z(µ, ϕ, π) Sample timestep t ∼ Uniform[0, 1] Interpolate Y t ← t • Y + (1 -t) • Y 0 Predict Ŷ ← f θ (C, I, Y t , t) Minimize objective MSE(Y , Ŷ ) Algorithm 2 STFlow: Inference Require: Testing WSIs (C, I) Sample prior Y 0 ∼ Z(µ, ϕ, π) for s ← 0 to S -1 Let t 1 ← s/S and t 2 ← (s + 1)/S Predict Ŷ ← f θ (C, I, Y t1 , t 1 ) if s = S -1 then return Ŷ end if Interpolate Y t2 ← Y t1 + ( Ŷ -Y t 1 ) (1-t1) * (t 2 -t 1 ) end for
Y is the count of detected RNA transcripts for a particular gene (starting from 0), representing the gene's expression level. In this study, the goal of STFlow is to predict the gene expression Y among spots with the input of (C, I).
this section cite: []

Section: Pathology Foundation Model
We define f PFM (•) as a pathology foundation model, which aims to extract generalpurpose embeddings for digital pathology after being pretrained on large-scale histology slides (Ciga et al., 2022;Chen et al., 2024b;Xu et al., 2024). They receive a patch from the slide as input and produce the embedding for downstream tasks:
{Z 0 , • • • , Z N } = f PFM {I 0 , • • • , I N }(1)
where Z i , I i represent the i-th spot's encoded representation and H&E image.
In our study, we leverage these foundation models to extract visual features for each spot image instead of training an individual image encoder. The motivation is that, after being pretrained on large-scale histology slides, these foundation models exhibit strong generalization abilities and help mitigate batch effect (Jaume et al., 2024).
this section cite: ['b8', 'b50', 'b20']

Section: Learning with Flow Matching
Modeling cell-cell interaction is essential for predicting the gene expression of each spot. Our key hypothesis is that the expression levels of certain genes in neighboring regions can strongly indicate the target spot's expression ( Li et al., 2022;Biancalani et al., 2021;Cordell, 2009). However, the standard regression objective cannot model cell-cell interaction as it predicts gene expression in one go.
To address this issue, we reformulate the gene expression regression model into a generative model, using samples from a prior distribution as input, which is then iteratively optimized instead of performing a one-step prediction.
Specifically, we apply flow matching (Lipman et al., 2022;Albergo & Vanden-Eijnden, 2022;Jing et al., 2024) as the optimization framework, which aims to learn a denoiser model f θ (•):
min θ MSE Y , f θ (Y t , I, C, t) (2
)
where t is a time step sampled uniformly from the uniform distribution Prior Distribution One key advantage of flow matching over diffusion models is its compatibility with different prior distributions. To explore gene expression patterns in ST samples, we analyze certain datasets from HEST-1k (Jaume et al., 2024). Figure 2 shows the distribution of gene expression frequencies across four datasets, revealing two key observations: (1) non-activated genes dominate the dataset, and (2) the data exhibits an overdispersion pattern (variance > mean).
In light of this, we explore zero-inflated negative binomial (ZINB) distribution Z(µ, ϕ, π), defined by the following probability mass function:
p(y | µ, ϕ, π) =      π + (1 -π) Γ(y+ϕ) Γ(ϕ) y! ϕ ϕ+µ ϕ µ ϕ+µ y if y = 0, (1 -π) Γ(y+ϕ) Γ(ϕ) y! ϕ ϕ+µ ϕ µ ϕ+µ y if y > 0, (3
)
where y is the count outcome, µ is the mean of the distribution, ϕ denotes the number of failures until stopped, and π is the zero-inflation probability. The negative binomial component introduces ϕ to explicitly account for variability beyond what is expected under a Poisson distribution, therefore modeling the overdispersion. Besides, the zero-inflation component represents the sparsity with a dropout propability (Virshup et al., 2023;Gayoso et al., 2022;Eraslan et al., 2019). Besides ZINB, Gaussian and zero distributions can also serve as priors, as further explored in Appendix C.
Training As shown in Algo.1, during training, we sample a time step t from the uniform distribution and interpolate the ground-truth gene expression Y with the sampled noise Y 0 to obtain noisy sample Y t . The denoiser predicts the denoised gene expression with the inputs of image features, coordinates, noisy samples, and time steps. The model is then optimized by minimizing the difference between the prediction and the ground-truth expression.
this section cite: ['b26', 'b1', 'b10', 'b29', 'b0', 'b22', 'b20', 'b46', 'b16', 'b13']

Section: Sampling
As shown in Algo.2, we begin with an initial "expression guess" Y 0 sampled from the ZINB distribution and iteratively refine it using the trained denoiser. The model interpolates between the noisy input Y t and the predicted denoised expression Ŷ over multiple steps, with a decay coefficient that gradually increases as the time steps increase. This process ultimately converges to the optimal gene expression in the final step.
this section cite: []

Section: Denoiser Architecture f θ
The STFlow's denoiser receives visual features Z, coordinates I, and gene expression Y t at time step t as input. The backbone is based on the Transformer architecture (Vaswani, 2017), achieving E(2)-invariance to the coordinates by incorporating frame averaging (FA) within each layer and explicitly encoding spatial dependencies by conducting attention to each spot's local neighbors. The overall architecture is shown in Figure 3(a).
Local Spatial Context Cells within the tissues can interact and influence each other's gene expression, thereby forming a spatial context with spot-to-spot dependencies. To efficiently leverage such dependencies, we encode the local spatial context around each spot i and limit the attention to its k-nearest neighbors, i.e., N (i), in the WSI. Long-range context information can be captured through multi-layer attention within the local neighbors of every spot.
this section cite: ['b45']

Section: E(2)-Invariant Spatial Attention
We introduce a spatial attention mechanism that generates spot representations invariant to E(2) operations, i.e., rotation, translation, and reflection, of the coordinates. To achieve this, we adapt frame averaging (FA), an E(2)-invariant transformation for point clouds (Puny et al., 2021), to the attention scheme.
The flexibility of FA provides a recipe for encoding the coordinates with minimal modification to the Transformer. Specifically, for i-th spot, we first construct the local context with the direction vectors from it to its neighbors:
C i = {C i→j | j ∈ N (i)} (4
)
where C i→j = C i -C j is the direction vector and represents the orientation between spots. Such a geometric context is then projected into frames extracted by PCA:
F(C i ) := {(U , ĉ) | U = [α 1 u 1 , α 2 u 2 ], α 1,2 ∈ {-1, 1}},(5)
where F(•) denotes four extracted frames with the two principal components (u 1 , u 2 ) and centroid c. We use C (g) i→j = (C i→j -ĉ)U to denote the projected direction vector from i-th to j-th spot using g-th frames. Building on top of them, we embed these spatial spot-spot dependencies with linear layers and achieve invariance by averaging the representations in different frames:
C ′ i→j = 1 |F(C i )| g MLP(C (g) i→j )(6)
where C ′ i→j ∈ R d is the encoded representation of the spatial relationship between i-th spot and its neighbor j at l-th layer. With such pairwise encoding, the spatial information sent from one source spot depends on the target spot, which is compatible with the attention mechanism.
After transforming the image features Z i into query, key, and value representations: Z Q,i , Z K,i , Z V,i , we adopt MLP attention (Brody et al., 2021) to derive the attention weight between spots, which incorporates the spatial information and the gene expression difference between spots within the calculation:
A ij = Softmax i MLP Z Q,i || Z K,j || C ′ i→j || (Y t,i -Y t,j ) (7
)
where A ij denotes the attention score between i-th and j-th spots, and Softmax i (•) is the softmax function operated on the attention scores of spot i's neighbors.
The spatial representation is then aggregated as the context for updating the spot representation, and the gene expression is iteratively updated at each layer, which progressively denoises the gene expression data across different receptive fields:
Z ′ i = MLP   j∈N (i) A ij Z V,j || j∈N (i) A ij C ′ i→j   + Z i(8)
Y ′ t,i = MLP Z ′ i (9
)
where Z ′ i and Y ′ t,i represent the updated i-th spot's representation and gene expression from the spatial attention module. This process is repeated across each layer, with the gene expression updates from each layer averaged to produce the final gene expression prediction.
this section cite: ['b38', 'b2']

Section: Discussion
Notes on invariance For the spot-level, Equ.6 demonstrates E(2)-invariance to the coordinates as it encodes and averages the coordinates across different frames, which is guaranteed by the frame averaging framework. Consequently, the spatial attention mechanism (Equ.7 and Equ.9) that relies on the output of Equ.6 is E(2)-invariant. For the pixel level, we apply pathology foundation models, which are pretrained with extensive image augmentations, making the extracted spot features robust to any E(2) transformations.
Computational Complexity For spatial attention, FA is efficient due to the low dimensionality of the coordinates (only 2) and the accelerated PCA algorithm, thus we ignore its complexity. The attention calculation involves neighboring spots and linear transformations, resulting in a complexity of O(N kd + N kdfoot_5 ), where d is the embedding size, and is efficient since k ≪ N . With flow matching, the computation scales linearly with the number of refinement steps S.
In practice, this remains efficient as flow matching requires relatively few steps, a key advantage over diffusion models.
In our experiments, we set S to 5.
this section cite: []

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2022)
Ref_id:b1 Title: Deep learning and alignment of spatially resolved single-cell transcriptomes with tangram Year: (2021)
Ref_id:b2 Title: How attentive are graph attention networks Year: (2021)
Ref_id:b3 Title: Geometric deep learning: Grids, groups, graphs, geodesics, and gauges Year: (2021)
Ref_id:b4 Title: Clinical-grade computational pathology using weakly supervised deep learning on whole slide images Year: (2019)
Ref_id:b5 Title: Stimage-1k4m: A histopathology image-gene expression dataset for spatial transcriptomics Year: (2024)
Ref_id:b6 Title: Towards a general-purpose foundation model for computational pathology Year: (2024)
Ref_id:b7 Title: Accurate spatial gene expression prediction by integrating multiresolution features Year: (2024)
Ref_id:b8 Title: Self supervised contrastive learning for digital histopathology Year: (2022)
Ref_id:b9 Title: Spatial organization of the somatosensory cortex revealed by osmfish Year: (2018)
Ref_id:b10 Title: Detecting gene-gene interactions that underlie human diseases Year: (2009)
Ref_id:b11 Title: Von willebrand factor as a potential predictive biomarker of early complications of endothelial origin after allogeneic hematopoietic stem cell transplantation Year: (2024)
Ref_id:b12 Title: Transcriptome-scale super-resolved imaging in tissues by rna seqfish+ Year: (2019)
Ref_id:b13 Title: Single-cell rna-seq denoising using a deep count autoencoder Year: (2019)
Ref_id:b14 Title: )-transformers: 3d roto-translation equivariant attention networks Year: (1970)
Ref_id:b15 Title: Gemnet: Universal directional graph neural networks for molecules Year: (2021)
Ref_id:b16 Title: A python library for probabilistic analysis of single-cell omics data Year: (2022)
Ref_id:b17 Title: Integrating spatial gene expression and breast tumour morphology via deep learning Year: (2020)
Ref_id:b18 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b19 Title: Protein-nucleic acid complex modeling with frame averaging transformer Year: (2024)
Ref_id:b20 Title: HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis Year: (2024-06)
Ref_id:b21 Title: Thitogene: a deep learning method for predicting spatial transcriptomics from histological images Year: (2024)
Ref_id:b22 Title: Alphafold meets flow matching for generating protein ensembles Year: (2024)
Ref_id:b23 Title: A method for stochastic optimization Year: (2014)
Ref_id:b24 Title: Machine learning for uncovering biological insights in spatial transcriptomics data Year: (2023)
Ref_id:b25 Title: Spatial transcriptomics inferred from pathology whole-slide images links tumor heterogeneity to survival in breast and lung cancer Year: (2020)
Ref_id:b26 Title: Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution Year: (2022)
Ref_id:b27 Title: From bulk, single-cell to spatial rna sequencing Year: (2021)
Ref_id:b28 Title: Equiformer: Equivariant graph attention transformer for 3d atomistic graphs Year: (2022)
Ref_id:b29 Title: Flow matching for generative modeling Year: (2022)
Ref_id:b30 Title: Symmetry-informed geometric representation for molecules, proteins, and crystalline materials Year: (2023)
Ref_id:b31 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b32 Title: Data-efficient and weakly supervised computational pathology on whole-slide images Year: (2021)
Ref_id:b33 Title: Ube2c serves as a prognosis biomarker of uterine corpus endometrial carcinoma via promoting tumor migration and invasion Year: (2023)
Ref_id:b34 Title: Identification of gata3 as a breast cancer prognostic marker by global gene expression meta-analysis Year: (2005)
Ref_id:b35 Title: Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region Year: (2018)
Ref_id:b36 Title: Rna structure & sequence design via inverse folding-based flow matching Year: (2024)
Ref_id:b37 Title: Leveraging information in spatial transcriptomics to predict super-resolution gene expression from histology images in tumors Year: (2021)
Ref_id:b38 Title: Frame averaging for invariant and equivariant network design Year: (2021)
Ref_id:b39 Title: Erbb2 oncogene in human breast cancer and its clinical significance Year: (1998)
Ref_id:b40 Title: E (n) equivariant graph neural networks Year: (2021)
Ref_id:b41 Title: Schnet-a deep learning architecture for molecules and materials Year: (2018)
Ref_id:b42 Title: Visualization and analysis of gene expression in tissue sections by spatial transcriptomics Year: (2016)
Ref_id:b43 Title: Highly sensitive spatial transcriptomics at near-cellular resolution with slide-seqv2 Year: (2021)
Ref_id:b44 Title: von willebrand factor is a useful biomarker for liver fibrosis and prediction of hepatocellular carcinoma development in patients with hepatitis b and c. United Year: (2018)
Ref_id:b45 Title: Attention is all you need Year: (2017)
Ref_id:b46 Title: The scverse project provides a computational ecosystem for single-cell omics data analysis Year: (2023)
Ref_id:b47 Title: General e (2)-equivariant steerable cnns Year: (2019)
Ref_id:b48 Title: Tumor microenvironment as a therapeutic target in cancer Year: (2021)
Ref_id:b49 Title: Spatially resolved gene expression prediction from histology images via bi-modal contrastive learning Year: (2023)
Ref_id:b50 Title: A whole-slide foundation model for digital pathology from real-world data Year: (2024)
Ref_id:b51 Title: Spatial transcriptomics prediction from histology jointly through transformer and graph neural networks Year: (2021)
Ref_id:b52 Title: Clinical and translational values of spatial transcriptomics Year: (2022)
Ref_id:b53 Title: Artificial intelligence for science in quantum, atomistic, and continuum systems Year: (2023)
Ref_id:b54 Title: Diffusion generative modeling for spatially resolved gene expression inference from histology images Year: (2025)
