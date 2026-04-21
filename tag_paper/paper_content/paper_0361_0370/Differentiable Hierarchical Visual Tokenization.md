Title: Differentiable Hierarchical Visual Tokenization
Abstract: Vision Transformers rely on fixed patch tokens that ignore the spatial and semantic structure of images. In this work, we introduce an end-to-end differentiable tokenizer that adapts to image content with pixel-level granularity while remaining backward-compatible with existing architectures for retrofitting pretrained models.Our method uses hierarchical model selection with information criteria to provide competitive performance in both image-level classification and dense-prediction tasks, and even supports out-of-the-box raster-to-vector conversion.

Section: Introduction
Transformers [1] have become the de facto architecture for all but a few data modalities [2][3][4][5][6][7][8]. The architecture is, however, contingent on the process of tokenization. Tokenizers for natural language [9,10] are designed to compress text into morphemes and semantic subwords-minimal units aligned with meaning. Yet in vision, tokenization is comprised of partitioning images into uniform square patches, ignoring semantic content and object boundaries in favor of computational convenience. This highlights a key incongruity; text tokenizers align with semantic units while patch-based vision tokenizers fragment objects without regard for their structure. Figure 1 illustrates how patch tokens lack the semantic coherence and granularity necessary for dense predictions.
Efforts to move beyond the grid paradigm include clustering or merging [11,12] for grouping features dynamically, or deformable patches [13,14] to improve spatial adaptivity. Recent work propose subobject tokenizers [15][16][17] to partition images into semantically coherent regions, providing gains in classification, segmentation accuracy, and interpretability. However, each method tackles only one or two facets of the larger tokenization problem, and none are fully end-to-end learnable.
An effective visual tokenizer must unify precise semantic alignment, differentiability, and adaptive granularity. Our key insight is that hierarchical pixel-level partitioning can be formulated as a multiscale model selection problem, and can be combined with differentiable mechanisms for end-to-end learning. We propose differentiable hierarchical tokenization (∂HT) for Vision Transformers (ViTs), emphasizing modularity [18,19] for reuse and backward compatibility.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18']

Section: Original
ViT [2] Quadtree [20] SLIC [16] ∂HT (Ours) Code and model weights: https://github.com/dsb-ifi/dHT 39th Conference on Neural Information Processing Systems (NeurIPS 2025). From an input image we produce a hierarchy of superpixel representations. An optimal segmentation is then selected from the hierarchy using information criteria, and features are extracted for each superpixel. Features can then be used in any ViT backbone. (Right) We also depict the feature extraction process of a superpixel S where its features are mixed based on foreground, M + , background, M -, and shared background features, β. Details in Section 2.4.
Our contributions can be summarized as follows:
• Learnable tokens: A novel end-to-end differentiable tokenization method which adapts to training data, and ensures effective tokenization for classification and segmentation tasks.
• Retrofitted tokenizers: A flexible fine-tuning strategy to adapt pretrained ViTs for superpixel tokens with pixel-level granularity, enhancing versatility across tasks.
• Multiscale model selection: A lattice theoretic extension of information criteria to multilevel hierarchical partitioning for selecting the most informative image partitions.
• Image vectorization: A out-of-the-box method for adapting hierarchical superpixel tokenizers to raster-to-vector graphics conversion, without being specifically trained for this task.
this section cite: ['b1', 'b19', 'b15']

Section: Motivation
Visual tokenization means discovering a discrete set of regions from a continuous, high-dimensional image, under strict compute and memory budgets-effectively solving segmentation, compression and representation learning all at once. Unlike 1D sequences, where you have a natural ordering and can pick breakpoints by cues such as whitespace and morpheme statistics, spatial data offer no canonical order. The space of possible region shapes, sizes and connectivity explodes, and tokens must respect multi-scale structure and spatial invariances. Moreover, full integration with ViTs require differentiable tokenization-adaptively choosing token count, placement, and shape to adhere to semantic boundaries to extract meaningful atomistic units from a scene with end-to-end learning.
this section cite: []

Section: Related Work
Existing approaches to adaptive visual tokenization can be organized into three broad categories.
Dynamic Grouping and Merging: Several works [11,[20][21][22] propose to dynamically cluster or merge grid tokens within the transformer's layers. These approaches reduce redundancy and adapt token budgets per image, but remain tethered to the original grid primitives and do not introduce truly off-grid partitions. This prevents discovery of true subobject boundaries for semantic coherence.
Deformable Sampling: A different approach stems from replacing fixed patches with learned sampling with deformation and dynamic positions. DPT [13] and Deformable Attention Transformer (DAT) [14] learn per-token centers and scales via differentiable bilinear sampling, granting geometric flexibility. While fully end-to-end learnable, these are still fundamentally patch based, leaving semantic alignment and object boundaries implicit.
Subobject and Superpixel Tokenizers: Superpixel-based methods partition images into semantically coherent regions prior to encoding. SuiT [16] pools CNN features over SLIC superpixels; EPOC [17] combines learned boundary cues with watershed grouping to form panoptic tokens; SP-former [23] uses hybrid cross-attention pixel-to-token clustering between attention blocks; SPiT [15] performs parallel hierarchical graph merging.
While SuiT works on a downsampled version of the input image, other methods provide true pixel-level granularity for extracting subobject tokens. These yield clear gains in classification, segmentation, and interpretability, but the non-differentiable grouping step prevents the tokenizer itself from being trained end-to-end.
this section cite: ['b10', 'b19', 'b20', 'b21', 'b12', 'b13', 'b15', 'b16', 'b22', 'b14']

Section: Method: Differentiable Hierarchical Tokenization
Our proposed method builds on previous subobject level approaches to provide the first fully endto-end learnable tokenizer for ViTs. We emphasize that ∂HT is not another ViT variant, but a fully modular tokenizer that can serve as a plug-and-play extension for pretrained models. Our design emphasizes precise semantic alignment with pixel-level granularity, multi-scale awareness via hierarchical pruning with information criteria, end-to-end differentiability, and modularity.
High Level Overview: ∂HT produces adaptive tokens through four stages, illustrated in Figure 2.
• Feature projection (Section 2.1): We embed each pixel into a learned d-dimensional feature space using a lightweight CNN, establishing a similarity metric for subsequent grouping.
• Hierarchical partitioning (Section 2.2): Starting from individual pixels, we iteratively merge similar adjacent regions to construct a complete hierarchy, capturing structure at multiple scales.
• Optimal selection (Section 2.3): Information criteria identify the partition that best balances model fit against complexity, eliminating manual threshold tuning.
• Differentiable extraction (Section 2.4): A mean-injection mechanism produces token features compatible with standard ViTs while enabling end-to-end gradient flow.
Preliminaries: Consider a graph G = (V, E) where the vertices represent pixel positions in a grid of width w and height h, and edges E are connections of horizontally and vertically adjacent vertices.
We view an image as a signal function x : V → R c , mapping each vertex v ∈ V to a c-channel pixel feature. A connected partition of V is a set π = {S : S ⊆ V } which satisfies:
(i) Non-overlapping: For any pair S, S ′ ∈ π, their intersection is empty, i.e., S ∩ S ′ = ∅.
(ii) Covering: The union of all S ∈ π is the full vertex set, i.e., S∈π S = V . (iii) Connected: For any vertices u, v ∈ S, there exists a path of adjacent vertices in S starting at u and ending at v, where each pair of consecutive vertices is connected by an edge in E.
Let Π(V ) be the space of all partitions of V and consider two partitions π 1 , π 2 ∈ Π(V ). If for every region S 1 ∈ π 1 there exists a region S 2 ∈ π 2 such that S 1 ⊆ S 2 , we say that the two partitions form a hierarchy H = (π 1 , π 2 ) ordered by refinement.
this section cite: []

Section: Subobject Feature Projection
We use a lightweight CNN encoder f : R c → R d to embed each pixel to an initial feature space f 0 (v) = f (x(v)). We design f with a residual branch that applies a 1 × 1 convolution to lift the c-channel input to d dimensions, and a main branch that downsamples via two successive convolutions (3 × 3 or 2 × 2 strided) followed by bilinear upsampling.
this section cite: []

Section: Hierarchical Vertex Merging
We expand on existing methods [15] to construct hierarchical partitions by iterative vertex merging. On a high level, the procedure pairs vertices with their most similar neighbor and merges them via parallel connected components, ensuring that each region is a connected superpixel.
More formally, given a positive semi-definite kernel κ : V × V → R ≥0 , we define the maximally similar vertex v max = arg max u∈N(v) κ(u, v) such that
E max = {(v, v max ) : v ∈ V } ⊂ E,(1)
pairs each vertex with its most similar neighbor within the neighboring vertices N(v) defined by E.
Let C 1 : V → V 1 denote a mapping of vertices to the connected components of the spanning subgraphfoot_0 G[E max ]. Then, C 1 (v) denotes a connected superpixel region S ∈ V 1 such that vertices within the same component of G[E max ] are assigned to S. In particular, S contains v, v max , and potentially other vertices based on similarity. A natural choice for updating the vertex features for V 1 is to simply take the average feature values in each component
f1 (v) = 1 |S| u∈S f 0 (u), S = C 1 (v).(2)
For a given level t, the mapping C t induces a graph contraction G t-1 → G t . Each vertex v ∈ V t-1 is assigned to a connected component C t (v) = S t ∈ π t , which yields a hierarchy of partitions H = (π 0 , . . . , π T ). Importantly, each superpixel S t at any level t in the hierarchy is always grounded in the original pixel positions in V . We formalize the construction in Appendix A.2.
this section cite: ['b14']

Section: Hierarchical Pruning via Information Criteria
Having constructed a full hierarchy, we now look to select an optimal partition. The construction of H is designed to tackle two objectives; multi-scale adaptability and redundancy management.
We compute the hierarchy up to a singleton such that all regions are connected up to a root region representing the full image. This allows us to frame the search for an optimal partition π * ∈ Π(V ) as a model selection problem using information criteria (IC) [24], removing the need for threshold calibration or gating [20,25,26]. Recall that an IC has the generalized form
IC(θ) = -2 log L(θ) + g(df θ ),(3)
where L(θ) is the likelihood of the data under the parameter θ, and g(df θ ) is a penalty function based on the statistical degrees of freedom df θ that discourages overly complex models. A choice of IC typically determines the form of g. Our goal is then to derive an estimate for IC(π * ).
Likelihood: A superpixel representation v ∈ S ∈ π * can be considered a piecewise constant model of the image for which each region S is assigned a constant value µ S . We model each pixel feature f 0 as i.i.d. samples from a Gaussian distribution, i.e., f 0 | S ∼ N (µ S , Σ S ). Under this assumption, the log-likelihood log L(π * ) simplifies to terms only involving Σ S . This aligns with variance reduction criteria [25], but augmented with additional parsimony constraints. See Appendix B.1 for details.
Degrees of freedom: Traditionally, degrees of freedom df θ correspond to the number of parameters estimated in a model. For each superpixel S, µ S is a parameter in a piecewise constant model of image x, making df θ proportional to the total number of regions in the optimal partition π * . However, this cannot be determined exactly without search of all possible combinations in H. Instead, we leverage the lattice structure [27] of partition space Π(V ), and find that df θ are inversely proportional to the total number of connected edges within each S. This gives a proportional estimate
df π * = S∈π * df S ∝ S∈π * Vol G (S) -1 ,(4)
where Vol G (S) represents the total number of edges (u, v) ∈ E such that u, v ∈ S. This formulation effectively penalizes partitions that consist of smaller regions (with fewer internal edges), encouraging larger, more informative regions while still capturing essential structural information. We provide a formal derivation of this result in Appendix B.2.
In turn, ∂HT prunes a full hierarchy H according to the selected IC, balancing model fit and complexity to select optimal partitions. We explore the effects of different information criteria on our method's performance in Section 3.3.
this section cite: ['b23', 'b19', 'b24', 'b25', 'b24', 'b26']

Section: Differentiable Token Embeddings
There are two main approaches to feature extraction with superpixel tokenization. The most common approach is to perform aggregation of regions through a separate encoder [16,17]. However, this prevents drop-in replacement in a ViT model, making retrofitting tokenizers to pre-trained models non-trivial. In contrast, SPiT [15] generalizes the ViT feature extraction process to irregular regions. Each region's bounding box is interpolated to a fixed patch size while masking out the background features of the surrounding vertices. This is backward compatible with standard ViTs, but suffers from being inherently non-differentiable. ∂HT makes generalized ViT features fully differentiable by introducing weighted aggregation and a novel mean-injection trick reminiscent of straight-through estimation [28,29].
At each level t, we collect learning signals from the contraction process by computing features via
f t+1 (v) = |S t | |S t+1 | u∈St+1 κ(u, u max ) • f t (u),(5)
where |S t | and |S t+1 | denote the number of pixels in the respective superpixel regions at two consecutive levels in the hierarchy. This update is a weighted variant of (2) with each vertex's contribution weighted by the kernel score of their most similar neighbor at step t.
Additionally, let f * (v) denote the feature of vertex v ∈ V under some optimal partition π * . To extract features from interpolated regions of the original image x, the idea is to inject the pruned vertex features into the original image channels without altering its local texture properties. For each pixel v ∈ S ∈ π * , we adjust the original pixel feature x(v) by computing
x(v) = x(v) + W f * (v) -x * (v),(6)
where W : R d → R c is a learnable linear mapping. This effectively replaces the mean of each superpixel region x * (v) = 1 |S| u∈S x(u) with a learnable estimate from the tokenization process, enabling full gradient flow from tokenization to the ViT backbone.
Positional Embedding: Previous work has shown that positional embedding computed as a linear combination of kernelized joint-histogram positions for each region generalize standard learnable positional embeddings [15]. We extend previous work by allowing for higher resolution, which provides more fine grained detail to token embeddings, and ablate the effect in Table 8.
Modularity and Retrofitting: Modularity is central to developing complex systems [18,30], and allows architectures to be broken down into reusable components. Just as transfer learning lets you fine-tune pretrained task heads, a modular tokenizer enables you to transfer pretrained models across distinct tokenization schemes without retraining from scratch. To apply this principle to a pre-trained patch-based ViT, we initialize ∂HT as follows. From (6), we observe that having W f * (v) = x * (v) results in x = x; a perfect reconstruction of the original image. We find that pretraining the encoder f and the linear mapping W jointly with
L rec = 1 |V | v∈V x(v) -W f * (v) 2 2(7)
provides maximally aligned features for fine-tuning a ViT backbone with ∂HT in place of the canonical tokenizer, allowing for fully differentiable tokenizer retrofitting.
Background Masking: During our experiments on retrofitting, we observe that masking out background features leads to sparsity when highly irregular regions are interpolated to a fixed size. Intuitively, masking leads to a minor domain shift in token embeddings, since the original features are dense within a patch and masking naturally leads to sparser representations. This can result in slower convergence and loss in performance.
In response, we introduce dynamic adaption of background masks during training. Let q denote the feature patch size, and let λ ∈ [0, 1], β ∈ R c×q×q be learnable parameters in a feature extractor. Let M + S ∈ {0, 1} q×q be the interpolated foreground mask of a superpixel S, with M - S = 1 -M + S denoting the background mask. We extract token features F (S) ∈ R c×q×q via
F (S) = (M + S + λM - S ) ⊙ x(S) + (1 -λ)M - S ⊙ β, (8
)
where ⊙ is element-wise product. This tweak allows the model to blend the foreground and background elements within each region through λ. The parameter β serves as a shared background feature, which mitigates sparsity by preventing zeroing out background regions completely. An illustration is provided in the right section of Fig. 2.
We ablate the effect of masking, encoder f , choice of kernel κ and information criteria IC, as well as other hyperparameters and architectural specifics of ∂HT in Tables 6 and 7.
this section cite: ['b15', 'b16', 'b14', 'b27', 'b28', 'b14', 'b17', 'b29']

Section: Experiments and Results
We design our experiments to investigate the representation capabilities of our method's extracted tokens in multiple settings; including end-to-end learning with classification on ImageNet1k [31], transfer learning as a drop-in tokenizer replacement for pretrained ViTs (cf. Section 3.1), and demonstrating decoder-free segmentation models with learnable tokenization (cf. Section 3.2). Moreover, ∂HT can also be evaluated on learnable image vectorization, and we compare our method to learnable image vectorization models (cf. Section 3.2). Training setup is detailed in Appendix D.
this section cite: ['b30']

Section: Image Level Predictions
We focus on transformer baselines trained exclusively on ImageNet1k [31], and validate on various downstream tasks [32][33][34][35][36]. In addition to reporting top-1 accuracy scores, we perform a kNN evaluation to assess the quality of the representation space. kNN scores are computed by taking the max score over k ∈ {10, 20, 50, 100, 150, 200} [39].
Retrofitting: We evaluate the effect of retrofitting ∂HT to pretrained models [37], including the less common B32 capacity [38] for completeness. We align the ∂HT tokenizer by pretraining using (7). Then, we finetune pretrained models to replace the canonical tokenizer in the ViT that will be retrofitted with ∂HT tokenizer. All results are uniformly re-evaluated for fair comparison, explaining minor differences from previously reported results [37].
Table 1 (top) shows that tokenizer retrofitting has a generally positive effect on linear evaluation models with base capacity, while maintaining competitive performance for small capacity models. Interestingly, our kNN results for ImageNet1k indicate that DEiT3 [37] models generally perform better with kNN than the linear evaluation, which is surprising as one typically expects the opposite. Contrarily, our retrofitted counterparts yield an expected result; the linear head produces better results than kNN. On the other hand, our retrofitted models are better aligned with both linear probing and kNN on Caltech256 [34] and CUB200 [35], indicating that ∂HT provides token representations that are useful for generalizing beyond the training data.
In general, we observe that retrofitting models with ∂HT enables models to adapt tokens to individual images, maintaining or slightly improving overall classification performance compared to baselines. Training from Scratch: We compare transformers using ∂HT to standard ViT patch tokenization [2] by training models from scratch under the same training regime, such that training is guaranteed to be equivalent. This eliminates confounding factors such as hardware and minor implementation differences. Contrary to the retrofitted setup, models trained from scratch use local gradient features, which have been shown to improve performance for both canonical and superpixel tokenization [15] with minimal computational overhead (+0.07 GFLOPs, +256 pa- Table 4: Zero-shot segmentation results on salient detection datasets using token-cut. ∂HT outperforms existing approaches, including other adaptive tokenization frameworks. ECSSD [55] DUTS [56] DUT-OMRON [57] Backbone Fmax IoU Acc@1 Fmax IoU Acc@1 Fmax IoU Acc@1 DINO-B [58] 80.3 71.2 91.8 67.2 57.6 90.3 60.0 53.3 88.0 DINO-B † [58] 87.4 77.2 93.4 75.5 62.4 91.4 69.7 61.8 89.7 SPiT-B [15] 90.3 77.3 93.4 77.1 63.9 89.4 71.1 56.4 86.8 SuiT-B [16] 87.0 80.5 93.8 68.0 60.0 88.8 63.4 57.5 87.2 ∂HT-B 92.4 79.9 94.2 77.9 64.4 90.5 71.9 58.7 89.8 † Applies post-processing via bilateral upsampling.    rameters). Our results show that superpixel tokenization generally provides stronger classification results than canonical tokenization with patches, cf. Table 1 (bottom) in a strict apples-to-apples comparison. We also perform a comparison with existing superpixel-based tokenization approaches in Table 2. Our results show that ∂HT for ViT-B outperforms other methods while preserving modular compatibility with ViT architectures.
■ Bed ■ Grass ■ Wall ■ Rug ■ Brick ■ Cat ■ Wood Wall ■ Window
this section cite: ['b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b38', 'b36', 'b37', 'b6', 'b36', 'b36', 'b33', 'b34', 'b1', 'b14']

Section: Dense Predictions
We assess our method on dense tasks by evaluating the fully trained ∂HT models for semantic segmentation, zero-shot segmentation performance on selected benchmarks, as well as the nonstandard task of image autovectorization-converting rasterized images to vector images.
this section cite: []

Section: Semantic Segmentation:
We fine tune our fully trained ∂HT models on semantic segmentation baselines [41,42], without a dense decoder for upscaling. Instead, each superpixel is individually classified as a single segment using a simple MLP head for each token. The results in Table 3 show that ∂HT provides raw token embeddings that are well suited for image segmentation. A qualitative analysis with examples can be found in Appendix E, with extended results in Appendix C.
Zero-Shot Segmentation: Table 4 shows results for fully trained ∂HT models using the Token-Cut [58] method over three selected salient segmentation tasks [55][56][57]. The results demonstrate that ∂HT provides strong results in zero-shot salient segmentation, and shows that tokens can be used out-of-the-box with no post-processing or specialized training required.   Autovectorization and Image Tracing: ∂HT provides out-of-the-box superpixel partitions that are able to represent images with very high levels of detail. By converting these regions into vectorized representations, our pretrained models can serve as a high quality raster-to-vector graphics pipeline. We extract the optimal superpixel partition, and convert each region into a vectorized path using potrace [61]. Since previous work on learnable image vectorization [59,62] provide few quantitative baselines, we compare our method to the five examples provided by Li et al. [59] in Table 5. This shows that ∂HT yields high-quality raster-to-vector conversion, illustrated in Figures 4, 5 and E.3.
this section cite: ['b40', 'b41', 'b57', 'b54', 'b55', 'b56', 'b60', 'b58', 'b61', 'b58']

Section: Ablations and Hyperparameters
In our ablative study, we by measuring the effect of architectural mechanisms in ∂HT-S for both training paradigms. For ablating f as a CNN, we note that f is the foundational source of gradients from the image, so it cannot simply be dropped. We therefore replace the CNN with a linear projection (i.e., a 1 × 1 convolution) over the input channels.
Table 7 shows that each component contributes to a drop in accuracy compared to the baseline, with the exception of model selection. This can be attributed to the fact that model selection represents a constraint on deduplication. When such constraints are removed, the number of tokens increases during training, and the model ends up using much more tokens without increase in performance.
Tokenizer Hyperparameters: We ablate the effect of tokenizer hyperparameters by evaluating image reconstruction quality after tokenization, using mean squared error (MSE) and structural similarity index measure (SSIM) [60]. We focus on the reconstruction of the tokens (which only requires training the tokenizer) instead of their predictive properties (which requires training the full ViT) due to our computational limitations. The ablations compare the cosine, Tanimoto [63], and Gaussian kernels, as well as the Akaike, corrected Akaike, Bayesian, and correlation information criteria (AIC, AICC, BIC, and CIC, respectively) [24].
The results in Table 6 show that a Gaussian kernel with d = 8 and AICC produces the best scores. Moreover, we evaluate the effect of different resolutions for the positional embeddings in Table 8.
These results indicate that increasing the resolution of positional embeddings generally improve results. However, the effect saturates slightly at 24 × 24 for classification, and at 48 × 48 for segmentation. Given that increasing the resolution adds to the computational complexity (GFLOPs), we use these resolutions for our final models.
Scale Invariance: A central feature of ∂HT is that the model can select a subset of tokens that is most informative to represent each individual image. As a result, the number of tokens differs from image to image, adapting to variations in information and scale. To evaluate the effect of this behavior, we perform an experiment where we compare results over various image scales while keeping the number of tokens equivalent between models. We do this by performing an extra step of merging after the model selection step, merging regions with high similarity to produce similar numbers of tokens for each model. This test is performed comparing the retrofitted models to the baselines.
6 4 × 6 4 9 6 × 9 6 1 2 8 × 1 2 8 2 2 4 × 2 2 4 2 5 6 × 2 5 6 3 8 4 × 3 8 4 5 1 2 × 5 1 2 7 6 8 × 7 6 8 20 40 60 80 Resolution Acc@1 6 4 × 6 4 9 6 × 9 6 1 2 8 × 1 2 8 2 2 4 × 2 2 4 2 5 6 × 2 5 6 3 8 4 × 3 8 4 5 1 2 × 5 1 2 7 6 8 × 7 6 8 10 1 10 2 10 3 Resolution # Tokens Patch-S16 Patch-B16 Patch-B32 ∂HT-S16 ∂HT-B16
∂HT-B32
Figure 6: Scale invariance and token granularity for retrofitted models over ImageNet [31] with extended lowresolution evaluations (64-768px). The additional points highlight how both patch and adaptive tokenizations degrade gracefully with coarser sampling, showing stronger invariance for ∂HT.
Our results in Figure 6 show that ∂HT scales considerably better to higher resolutions by adapting tokens to image content, notably without modification of the resolution of positional embeddings.
In very low resolution settings (e.g. CIFAR's 32 × 32), canonical ViT tokenization may outperform ∂HT. In these settings, a few of pixels can end up covering an entire region and ∂HT may yield block-like regions similar to a square-patch tokenizer, as there are no clear inter-pixel edges to delineate. The uniform grid of patches aligns well with the reduced information content, allowing the standard tokenizer to perform adequately. ∂HT's adaptive token selection offers less advantage in this scenario as there is less fine-grained information to exploit. However, we note that at moderately low resolutions, ∂HT generally performs quite well out-of-the-box compared to patch-based ViTs.
At higher resolutions, images contain more detailed and fine-grained features, and ∂HT's ability to select and adapt tokens based on the most informative regions becomes highly beneficial. It can effectively capture critical details without being constrained by a fixed grid, leading to improved performance over the canonical tokenizer. ∂HT achieves this advantage without modifying the positional embeddings, demonstrating its inherent scalability to higher resolutions.
this section cite: ['b59', 'b62', 'b23', 'b30']

Section: Discussion and Conclusion
We propose ∂HT, a differentiable tokenizer with pixel-level granularity that uses information criteria to dynamically select optimal partitions from hierarchical representations. Our experiments demonstrate that ∂HT achieves strong performance on both image-level classification and dense prediction tasks while maintaining modularity when retrofitting pretrained models.
Our work establishes tokenization as an adaptive, learnable component in ViT architectures. As models and datasets scale, this modularity becomes increasingly valuable for adapting representations to specific tasks and domains. Given the broad applicability of superpixels in vision modeling [64][65][66][67], integrating adaptive tokenization could unlock performance gains across various applications, from medical imaging to video understanding where redundancy management is critical.
this section cite: ['b63', 'b64', 'b65', 'b66']

Section: Limitations

this section cite: []

Section: Further Work
As mentioned in the limitations, architectural design choices for edge contraction and feature aggregation are promising avenues for optimal design of adaptive tokenization in vision transformers. In this work, we gather implicit learning signals weighted by the similarity kernel in ( 5), but extending this to explicit learnable aggregation with graph neural networks could provide more expressive modeling. However, such extensions should preserve symmetry and positive semi-definiteness to ensure consistency and well defined edge contractions in the hierarchy (Appendix A.2).
Extending differentiable tokenization to self-supervised learning represents a natural extension. This was previously explored by Lew et al. [16], which trains a DINO variant with strong results. In self-supervised settings, masked image modeling (MIM) paradigms have potential for synergy with differentiable tokenization mechanisms, providing more direct learning signals via invariants such as translation, scale, and rotation.
Vision-language models represent another promising research direction, where alignment with language could help inform the edge contraction process. Adaptive tokenization can be beneficial for document-focused tasks where fixed patches poorly align with heterogeneous text and layout structures. Preliminary investigations suggest ∂HT could address key limitations in current docVLM approaches by generating tokens that better capture semantic boundaries on a per-sample basis.
Additionally, video transformers present a compelling application domain. The quadratic attention complexity makes redundancy management crucial, and spatiotemporal superpixel tokenization could significantly improve efficiency while preserving semantic coherence across frames. More broadly, our findings suggest that for tasks where high dimensionality is a bottleneck, better adaptive tokenization such as ∂HT can provide tractable dimensionality reduction by exploiting inherent local redundancies. Such redundancies cannot be meaningfully exploited if tokens are invariant to image content, such as in the case of square patches.
this section cite: ['b15']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: We briefly outline the main contributions of the work, shown in the main paper.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: Our conclusion has a section on the general limitations. We also point out and discuss the independence assumptions on residuals within individual regions in the supplementary.
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
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We provide a GitHub repository with code and model weights.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: We provide a list of hyperparameters and architecture parameters in the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [No] Justification: Due to the extent of experiments and computational resources of the models trained, we have not added confidence intervals for the main results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: [NA]
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: The code has a permissive MIT license.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: The code includes docstrings adhering to the NumPy format. This will be improved for the camera-ready version, which will include more extensive documentation and example notebooks.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Theoretical Results
In this section, we provide a formal construction for monotonic edge coloring with respect to graph connectedness, and show that this construction yields a hierarchical partitioning of V in Proposition A.13. We leverage this result to define a hierarchical graph partitioning in Definition A.14.
this section cite: []

Section: A.1 Notation and Preliminaries
[[n]] Discrete sequence (0, . . . , n). 2 X Power set of set X. G An undirected graph G = (V, E). x c-channel image x : V → R c . N(v) Neigborhood of v ∈ G. C(v) Connected components of v ∈ G. G[F ] Spanning subgraph of G under F ⊆ E. Π(V ) Set of partitions over V . π Partition in Π(V ). H Hierarchical partition (πt : πt ⊑ πt+1) L t=1 . Gt Graph Gt = (Vt, Et) in hier. graph seq. IC An information criteria. L Likelihood function. df Degrees of freedom. VolG(X) Number of (u, v) ∈ E s.t. u, v ∈ X. κ(u, v) Pos. def. kernel κ : R d × R d → R ≥0 .
Definition A.1 (Neighborhood). Let G = (V, E) be a graph. The neighborhood of a vertex v ∈ V is defined by N(v) = {u : {u, v} ∈ E}.
Definition A.2 (Subgraphs). Let G = (V, E) be a graph, and let
S ⊆ V . Let G[S] = (S, E[S]) be a graph such that E[S] = {{u, v} ∈ E : u, v ∈ S}. Then G[S] is a subgraph induced by S. Symetrically, for F ⊆ E, a subgraph G[F ] = (V, F ) is a spanning subgraph under F . Definition A.3 (Graph Connectivity). Let G = (V, E) be a graph. For v ∈ V , let N 0 (v) = {v} and N 1 (v) = N(v). By recursion, define N i+1 (v) = N i (v) ∪ u∈Ni(v) N(u). (A.1) Then C(v) = lim i→∞ N i (v) is called the connected component of v. If C(v) = V for any v ∈ V then G is a connected graph.
Definition A.4 (Reachability). Let G = (V, E) be a graph. We say that two nodes u, v ∈ V are reachable in G if and only if u ∈ C(v).
Definition A.5 (Equivalence Relations and Classes). Let V be a set. Let ∼ ⊆ V × V be a binary relation that is reflexive, transitive, and symmetric. Then ∼ is an equivalence relation on V . Furthermore, for some fixed element v ∈ V , let
[v] = {v ′ ∈ V : v ∼ v ′ }. (A.2)
Then [v] is an equivalence class of v under ∼.
Definition A.6 (Quotient Set). Let V be a set, and let ∼ be an equivalance relation on V . Then
V /∼ = {[v] : v ∈ V } (A.3)
is the quotient set of V induced by ∼.
Definition A.7 (Partition of Sets). Let V be a set. Let Π(V ) ⊂ 2 2 V such that for all π ∈ Π(V ) (i) ∅ / ∈ π, (ii) π covers V , i.e., S∈π S = V , (iii) for all S, S ′ ∈ π, if S ̸ = S ′ , then S ∩ S ′ = ∅.
Then π is a partition of V , and we call Π(V ) the set of all partitions of V .
Definition A.8 (Refinement of Partitions). Let V be a set, and let π, π ′ ∈ Π(V ). If for all S ∈ π there exists S ′ ∈ π ′ such that S ⊆ S ′ , then we say that π is a refinement of π ′ , denoted by π ⊑ π ′ . Furthermore, we say that π is finer than π ′ , and equivalently that π ′ is coarser than π.
Definition A.9 (Hierarchy of Partitions). Let V be a set, and let H = π t ∈ Π(V ) : π t ⊑ π t+1 T t=0 . Then H is a hierarchy of partitions ordered by refinement.
Theorem A.10 (Fundamental Theorem on Equivalence Relations). Let V be a set, and let ∼ be an equivalence relation on V . Then V /∼ ∈ Π(V ). Definition A.11 (Weighted Graph). Let G = (V, E) be a graph. Let χ : E → R be a function on the edges of G. Then G ′ = (V, E, χ) is called a weighted graph.
this section cite: []

Section: A.2 Hierarchical Graph Partitions
In this section, we construct the main result Definition A.14, which formalizes the construction of hierarchical partitions via monotonic binary edge coloring.
Proposition A.12 (Partition by Edge Coloring). Let G = (V, E, χ) be a weighted graph, and let χ : E → {0, 1} be a binary coloring of the edges such that the edge set
E χ = {{u, v} ∈ E : χ(u, v) = 1} (A.4)
is invariant under transitive closure; i.e., E + χ = E χ . Then, the coloring χ induces a partition of V into connected components, where each component is connected via E χ .
Proof. From E χ , we construct a relation ∼ on V such that u ∼ v if and only if u is reachable by v in the subgraph G[E χ ]. Note that ∼ is symmetric since G is undirected,
this section cite: []

Section: and transitive due to E +
χ = E χ . Since v ∈ C(v) for all v ∈ V , then v ∼ v so ∼ is necessarily also reflexive. Hence, ∼ is an equivalence relation on V , and by the fundamental theorem of equivalence relations, the quotient set V / ∼ is a partition corresponding to the connected components of G[E χ ].
Proposition A.13 (Hierarchical Partitioning by Monotonic Edge Coloring). Let G = (V, E, χ) be a weighted graph, and let χ : E × {0, . . . , T } → {0, 1} be a binary coloring of the edges satisfying (i) Monotonicity, χ(u, v, t) ≤ χ(u, v, t + 1) for all {u, v} ∈ E and t = 0, . . . , T -1.
(ii) For each step t = 0, . . . , T , the edge set E χ (t) = {{u, v} ∈ E : χ(u, v, t) = 1} is invariant under transitive closure, i.e., E + χ (t) = E χ (t). Then, χ induces a hierarchical partition of V ordered by refinement; i.e., the partition induced by χ t is a refinement of the partition induced by χ t ′ for all t ≤ t ′ , where we have that χ t (u, v) = χ(u, v, t) for all {u, v} ∈ E.
Proof. By Proposition A.12, we have that each χ t induces an equivalence relation ∼ t , partitioning V into equivalence classes at level t. Denote this partition by π t . We will show that for all t ≤ t ′ , the partition π t is a refinement of π t ′ . The monotonicity criteria (i) χ(u, v, t) ≤ χ(u, v, t + 1
) implies E χ (t) ⊆ E χ (t + 1). By induction, E χ (t) ⊆ E χ (t ′ ) for all t ≤ t ′ , Since E χ (t) ⊆ E χ (t ′ ), any path using edges in E χ (t) is also a path in E χ (t ′
). Therefore, if u ∼ t v, then u ∼ t ′ v. Now, let [u] t denote the equivalence class of u under ∼ t . Then [u] t ⊆ [u] t ′ for all u ∈ V and all t ≤ t ′ . Since π t = V / ∼ t = {[u] t : u ∈ V }, we have that π t ⊑ π t ′ , as we wanted to show.
Definition A.14 (Hierarchical Graph Partition). Let G = (V, E, χ) be an weighted graph where χ : E ×{0, . . . , T } → {0, 1} is a binary coloring as in Proposition A.13, i.e., monotonic and invariant under transitive closure. Let χ(u, v, 0) = 0 for all {u, v} ∈ E and let ∼ t denote the equivalence relation induced by E χ (t) for t = 0, . . . , T . Then the sequence
G[E χ (t)] = (V / ∼ t , E χ (t)), t = 0, . . . , T (A.5)
is called a hierarchical graph partition, where for each v ∈ V we have that each equivalence class [v] t = S ∈ π t denotes a connected region for π t ∈ H. For notional convenience, we write
G[E χ (t)] = G t = (V t , E t ).
this section cite: []

Section: B Modeling Assumptions and Estimators
In this section, we discuss details regarding methodological assumptions of ∂HT from Section 2.4. We cover the i.i.d. Gaussian assumption on the distribution of pixels, show that this approximation has precedence, and empirically verify that this it is a reasonable modeling choice, and derive the estimator for df π * via atomistic properties of the partition lattice.
Table C.1: Additional Single Scale Semantic Segmentation mIoU on COCO-Stuff [42] for the original 10k fold. Note the lack of comparative baselines for base capacity models. Dataset Backbone Method Size (↓) mIoU (↑) COCO10k Swin-L SeMask [69] 640 47.4 ConvNext-L CAR [70] 640 49.0 Swin-L Senformer [71] 640 49.8 ViT-L Segmenter [54] 512 47.1 ViT-B ∂HT + MLP 512 49.2
We also assess the robustness to the Gaussian assumption. We trained tokenizers under alternative Generalized Normal distributions with shape parameters b ∈ {2, 1, 0.5}, where b = 2 corresponds to the Gaussian baseline. Table B.1 shows that the Gaussian assumption yields the best reconstruction quality, despite final residuals being closer to b ≈ 0.551. This occurs because the distribution is closer to Gaussian in early training iterations, providing initial stability for model fitting.
The Gaussian assumption primarily affects pruning through residual variance estimation; heaviertailed residuals inflate variance, making the criterion more conservative. Importantly, the complexity penalty remains invariant to distributional misspecification as it depends only on graph topology, and ∂HT demonstrates robustness consistent with theoretical results on information criteria under model misspecification [24].
this section cite: ['b23']

Section: B.2 Atomistic Properties of Π(G)
We informally outline some fundamental lattice theory [27] and describe how we can derive an estimate of the degrees of freedom for a partitioned graph under the constraint of connectivity.
The set Π(V ) is a partially ordered set (poset) under refinement (Definition A.8), and contains two seemingly trivial elements; one is the minimal partition ⊥ = {{v} : v ∈ V }, called the bottom where all elements of V are individual blocks. Dually, the top ⊤ = {V }, is a partition in which all elements are grouped in a single block. Any finite nonempty subset V will necessarily satisfy ⊥, ⊤ ∈ Π(V ).
In the partition lattice Π(V ), atoms are defined as the minimal non-trivial partitions that cover the bottom element ⊥, where each vertex are isolated singletons. Dually, the co-atoms are the elements covered by ⊤ where all vertices comprise a single set. Independent sets of atoms form what is equivalent to a basis (independent sets) in constructing more complex partitions that define each superpixel. Under connectivity in G, the atoms of Π(V ) are precisely the edges E of G.
By assumption of a piecewise constant model, the complexity of the model decreases for courser partitions such that the level of complexity is maximal at ⊥. Then df π * is necessarily inversely proportional to the number of independent atoms each superpixel encompasses within Π(V ). Dually, it is necessarily also proportional to the number of possible bipartitions required to form the partition v ′ ∪ {v : v ∈ V, v / ∈ v ′ }. Unfortunately, this quantity can be considered more or less intractable, however, an estimate can be derived by considering the dualistic nature of the partition lattice.
Recall that Vol G (S) for some S ⊆ V is defined as |{{u, v} ∈ E : u, v ∈ S}|. Then Vol G (S) is the maximal number of atoms required to form S, which yields a direct measure of the number of steps between S and ⊥. This is typically formalized via a rank function r : Π(V ) → Z ≥0 which turns out to be equivalent to the number of atoms in a partition. However, we are instead interested in the number of steps between S and ⊤. Noting that for our construction, we have that r(⊤) = Vol G (V ) = |E| is a maxima, we can estimate degrees of freedom of S ∈ π * (V ) by letting
df π * (S) ≈ |E| • Vol G (S) -1 (B.5)
This serves to penalize partitions S that has lower volume and are closer to ⊥. In effect, the estimate penalizes higher parameter complexity w.r.t. the piecewise constant model of the image by inducing a preference for larger connected regions in the superpixel partition.
this section cite: ['b26']

Section: C Extended Results
In the interest of completeness, we include results for COCO-Stuff on the 10k fold to complement
κ(u, v) end for St+1 ← CONNECTEDCOMPONENTS(Emax) ft+1 ← ZEROS(N ′ × C) for u ∈ St+1 do for v ∈ {v | St+1(v) = u} do vmax ← Emax[v] w ← |St(v)|/|St+1(u)| • κ(v, vmax) ft+1[u] ← ft+1[u] + w • ft[v] end for end for Et+1 ← UPDATEEDGES(St+1) for u ∈ St+1 do LIC[u] ← IC(ft+1[u], Et+1) end for return ft+1, St+1, Et+1, LIC
this section cite: []

Section: Algorithm D.2 Feature Extraction
Require: Image tensor x ∈ R B×C×H×W Require: Region features f ∈ R N ×C ′ Require: Region map S Require: Grid resolution q ∈ N Require: Positional resolution p ∈ N Require: Projection matrix W ∈ R C×C ′ Require: Background token β ∈ R C×q×q Require: Mixing weight λ ∈ [0, 1] Ensure: Tokenized features F ∈ R N ×C×q×q Ensure: Kernel pos. features P ∈ [0, 1] N ×p×p for u ∈ S do µu ← meanv∈u(x[v]) μu ← W • f [u] for each pixel p ∈ u do x[p] ← x[p] + μuµu end for end for F ← ZEROS(N × C × q × q) P ← ZEROS(N × p × p) for u ∈ S do M ← DOWNSAMPLEMASK(S = u, q × q) for (i, j) ∈ q × q do s ← BILINEARSAMPLE(x, bbox(u), i, j) m ← M [i, j] smix ← λ • s + (1λ) • β[:, i, j] F [u, :, i, j] ← m • s + (1m) • smix end for P [u] ← KERNELPOSEMBED(S = u, p × p) end for return F, P which were updated to include the full COCO164k fold at a later time. Consequently, there are fewer baselines available. To the best of our knowledge, out of the works reporting results on COCO10k, there are no instances of base-or small capacity models available. Nevertheless, we include relevant results for COCO10k in Table C.1, which illustrate that ∂HT performs relatively well compared to larger models with larger capacity (305M parameters for large (L) compared to 87M for base (B) capacity models).
this section cite: []

Section: D Implementation and Training Details
We provide a full overview of our experimental setup and training configuration. Training and inference was performed on AMD MI250x and Nvidia A100. We detail central algorithms in Fig. D.1, and provide code and checkpoints in our GitHub repo. Our experiments were conducted as follows:
• Tokenizer Pretraining: ∂HT modules were pretrained to optimally reconstruct images from ImageNet1k, using (7). We train for 10 epochs using AdamW with 1e-4 learning rate and 1e-2 weight decay, but find that performance saturates between epoch 5-6. We test the performance with different hyperparameter settings-cf. Table 6.
• Retrofitting: We select three baseline models trained exclusively on ImageNet1k. Each model is then retrofitted with our pretrained tokenizer, using the configuration in Table D. 1(b). Models are fine tuned with layer-wise learning rate decay of 0.65 [72], which improves learning for earlier layers. We evaluate the models over various downstream tasks, yielding the results in Table 1.
• Scale Invariance: Given how the baseline ViT-B32 model produces very few regions, we perform a comparative evaluation by adding more fine-grained control over the number of tokens over different image resolutions. We add merging mechanisms which serves to limit the total number of tokens in a model, and evaluate baselines and retrofitted models over various image sizes. The results are provided in Figure 6.
this section cite: ['b6', 'b71']

Section: E Qualitative Results and Visualizations
In this section, we extend the visualizations and results from dense predictions and image vectorization. In Figure E.3, we show more raster-to-vector graphics conversions on example images from COCO-Val. We emphasize that our method produces high quality results, despite a comparatively simpler approach to image vectorization. Unlike other approaches [59,62], our method does not yield differentiable paths, and does not optimize the vector graphics for each individual image. Instead, we simply use the results of our tokenizer, trained with unrelated downstream tasks, to produce vectorized images.    Original ViT [2] Quadtree [20] SLIC [16] ∂HT (Ours) Figure E.6: Comparison of spatial granularity in tokenization methods. Our proposed ∂HT (right) provides an end-to-end learnable framework for tokenization.
this section cite: ['b58', 'b61']

Section: 
Justification: The theoretical results are clearly outlined in the supplementary.
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
Answer: [Yes] Justification: We detail the methodology to promote reproducibility. We include algorithms and a section on implementation and training details, including hyperparameter settings.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: 5.
Open access to data and code
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We estimate the computational resources in the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We have read the code of ethics, and adhere to these. Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. Table B.1: Ablation on sensitivity to Gaussian Assumptions. Shape parameter Metric b = 2.0 b = 1.0 b = 0.5 MSE 0.06 0.07 0.09 SSIM 0.60 0.57 0.55
this section cite: []

Section: B.1 Distrubutional Assumptions
A superpixel model can be viewed as a spatially piecewise-constant approximation to image data, constrained by connectivity requirements [68]. Specifically, for an image x a superpixel model M π is on the form
M π (v) = 1[v ∈ S] µ S . (B.1)
Given an image defined on pixels v ∈ V , we assume the existence of a partition π * into connected regions that minimize
L MSE = 1 |V | v∈V ∥x(v) -M π * (v)∥ 2 , (B.2)
where M π * (v) takes constant value µ S for all pixels v ∈ S, and the optimal µ S is given by the arithmetic mean of pixel intensities within S. Given a fixed partition, this choice of estimator is optimal under the Gauss-Markov theorem as the best linear unbiased estimator (BLUE).
Within each region S, the estimator µ S is BLUE under the assumption that the noise affecting pixels is independent, identically distributed (i.i.d.) with finite variance and zero mean. Further, adopting a Gaussian distribution for pixels conditioned on region membership x v | S ∼ N (µ S , Σ S ) aligns the squared-error minimization directly with maximum likelihood estimation. Specifically, for diagonal Σ S , maximizing the Gaussian log-likelihood corresponds exactly to minimizing within-region variance, connecting clearly with variance-reduction criteria in regression trees and quadtrees [20,25], supporting our argument in Section 2.4. This is clear from the log-likelihood for a multivariate Gaussian of dimension d for a region S, which is given by
log L(θ S ) = - |S| 2 (d log(2π) + log det Σ S ) - 1 2 v∈S (x v -µ S ) ⊺ Σ -1 S (x v -µ S ) (B.3) = - |S| 2 d log(2πe) + log det Σ S (B.4)
where (B.4) follows by the MLE for Σ S given µ S .
We emphasize that these assumptions are made explicitly and are not automatically justified by the existence of an optimal partition under variance minimization. The choice of empirical variance as a criterion to evaluate model fit is commonly employed in nonparametric contexts without distributional assumptions, hence the assumption of Gaussianity is not strictly necessary but implicit by the choice of risk minimizer. Claeskens and Hjort [24] show that, even if the candidate models are not parametric distributions, IC approaches remain asymptotically valid in selecting a model that minimizes expected prediction error.
To empirically examine the appropriateness of these assumptions, we evaluated the pixel intensity distributions within representative superpixels over ImageNet1k. Table D.1: Configuration parameters for different stages (a) Pretraining config value batch size 2048 epochs 400 dataset ImageNet1k img.size 192 × 192 pos.emb. 16 × 16 loss fn. CE (0.1 smooth.) optimizer LAMB lr.sched. cos.decay (5 w.u.) lr (start / base / stop) 3e-3 / 3e-7 / 1e-6 momentum 0.9 dropout path 0.1 (S) / 0.2 (B) opt. ϵ 1e-7 cutmix α 1.0 augment rand.aug. / aug3 (b) Tokenizer Retrofitting config value batch size 2048 epochs 100 dataset ImageNet1k img.size 192 × 192 pos.emb. 16 × 16 loss fn. CE (0.1 smooth.) optimizer LAMB lr.sched. cos.decay (5 w.u.) lr (start / base / stop) 1e-7 / 6e-5 / 1e-6 momentum 0.9 dropout path 0.1 (S) / 0.2 (B) opt. ϵ 1e-8 augment rand.aug. / aug3 llrd 0.65 (c) Finetuning config value batch size 512 epochs 100 dataset ImageNet1k img.size 224 × 224 pos.emb. 24 × 24 loss fn. CE (0.1 smooth.) optimizer AdamW lr.sched. cos.decay (5 w.u.) lr (start / base / stop) 1e-6 / 1e-5 / 1e-5 dropout path 0.1 (S) / 0.2 (B) opt. ϵ 1e-8 augment rand.aug. / aug3 llrd 0.9 (d) Segmentation Finetuning config value batch size 512 epochs 400 dataset COCO-Stuff, ADE20k img.size 512 × 512 pos.emb. 48 × 48 loss fn. BCE + Focal optimizer AdamW lr.sched. cos.decay (5 w.u.) lr (start / base / stop) 1e-6 / 1e-5 / 1e-5 dropout path 0.1 (S) / 0.2 (B) opt. ϵ 1e-8 augment rand.aug. / aug3 crop scale / ratio (0.5, 1.0) / (0.8, 1.2) llrd 0.85
• Full Training: We extend these experiments by evaluating a full training procedure, following the training process outlined by Touvron et al. [37], Steiner et al. [38], notably without the use of MixUp [73] augmentation, as blended images produces inaccurate boundaries for learning coherent regions. Following previous works [15], models trained from scratch apply gradient histogram features. As is generally recommended, the training was performed in two steps, outlined in
Table D.1(a) and Table D.1(c) respectively. The results are featured in the lower half of Table 1.
• Segmentation Fine Tuning: Given our fully trained ∂HT models, we perform fine tuning for semantic segmentation. We replace each head with a single hidden-layer MLP with a hidden ratio of 4×. The fine tuning is performed using the configuration in Table D.1(d), and results are reported in Table 3.
• Zero-shot Salient Segmentation: We evaluate our fine-tuned classification model on zeroshot salient segmentation. We emphasize that the model has not been trained for this task. Following Wang et al. [58], we compute the graph Laplacian of the token representations, and compute a bipartition using the Fiedler vector. Foreground masks are selected by passing masked tokens through the transformer, and selecting the mask that sees the least drop in performance under occlusion. Results are featured in Table 4.
• Image Vectorization: Our ∂HT tokenizer provide high fidelity superpixels, which can be directly applied for image vectorization out-of-the-box. From our pretrained tokenizer we extract both an optimal, as well as a low granularity partition, noting that lower granularity partitions has much fewer superpixels. We then extract paths for each superpixel in the lower granularity region, and layer high granularity paths on top using potrace [61], resulting in an SVG image. We compare results with quantitative baselines in Table 5.
this section cite: ['b67', 'b19', 'b24', 'b23', 'b36', 'b37', 'b72', 'b14', 'b57', 'b60']

Section: References
Ref_id:b0 Title: Attention is all you need Year: (2017)
Ref_id:b1 Title: An image is worth 16 × 16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b2 Title: AST: Audio spectrogram transformer Year: ()
Ref_id:b3 Title: Do transformers really perform badly for graph representation Year: (2021)
Ref_id:b4 Title: A comprehensive survey on applications of transformers for deep learning tasks Year: (2024)
Ref_id:b5 Title: Set transformer: A framework for attention-based permutation-invariant neural networks Year: (2019)
Ref_id:b6 Title: Point transformer Year: ()
Ref_id:b7 Title: Temporal fusion transformers for interpretable multi-horizon time series forecasting Year: (2021)
Ref_id:b8 Title: Neural machine translation of rare words with subword units Year: (2016)
Ref_id:b9 Title: Google's multilingual neural machine translation system: Enabling zero-shot translation Year: (2017)
Ref_id:b10 Title: Object-centric learning with slot attention Year: (2020)
Ref_id:b11 Title: SPFormer: Enhancing vision transformer with superpixel representation Year: (2025)
Ref_id:b12 Title: DPT: Deformable patch-based transformer for visual recognition Year: ()
Ref_id:b13 Title: Vision transformer with deformable attention Year: (2022)
Ref_id:b14 Title: A spitting image: Modular superpixel tokenization in vision transformers Year: (2024)
Ref_id:b15 Title: Superpixel tokenization for vision transformers: Preserving semantic integrity in visual tokens Year: (2024)
Ref_id:b16 Title: Subobject-level Image Tokenization Year: (2025)
Ref_id:b17 Title: Representation Learning: A Review and New Perspectives Year: (2013)
Ref_id:b18 Title: Design and Evolution of Modular Neural Network Architectures Year: (1994)
Ref_id:b19 Title: Vision Transformers with Mixed-Resolution Tokenization Year: (2023)
Ref_id:b20 Title: Tokens-to-Token ViT: Training vision transformers from scratch on imagenet Year: (2021)
Ref_id:b21 Title: Token Merging: Your ViT But Faster," in Inter. Conf. Learn. Represent. (ICLR) Year: (2023)
Ref_id:b22 Title: SegFormer: Simple and efficient design for semantic segmentation with transformers Year: (2021)
Ref_id:b23 Title: Model selection and model averaging Year: (2008)
Ref_id:b24 Title: Classification and regression trees Year: (1984)
Ref_id:b25 Title: MSViT: Dynamic mixed-scale tokenization for vision transformers Year: (2023)
Ref_id:b26 Title: General lattice theory Year: (2002)
Ref_id:b27 Title: Estimating or propagating gradients through stochastic neurons Year: (2013)
Ref_id:b28 Title: Categorical reparameterization with gumbel-softmax Year: (2017)
Ref_id:b29 Title: On the criteria to be used in decomposing systems into modules Year: (1972)
Ref_id:b30 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b31 Title: Are we done with ImageNet? Year: (2006)
Ref_id:b32 Title: Do ImageNet Classifiers Generalize to ImageNet?" in Inter. Conf. Mach. Learn. (ICML) Year: (2019)
Ref_id:b33 Title: Caltech 256 Year: (2022-04)
Ref_id:b34 Title: Caltech-UCSD Birds 200 Year: (2010)
Ref_id:b35 Title: 3D Object Representations for Fine-Grained Categorization Year: (2013)
Ref_id:b36 Title: DeiT III: Revenge of the ViT Year: (2022)
Ref_id:b37 Title: How to train your ViT? data, augmentation, and regularization in vision transformers Year: (2021)
Ref_id:b38 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b39 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b40 Title: Semantic Understanding of Scenes Through the ADE20k Dataset Year: (2019)
Ref_id:b41 Title: Microsoft COCO: Common Objects in Context Year: (2014)
Ref_id:b42 Title: Deep Residual Learning for Image Recognition Year: (2016)
Ref_id:b43 Title: Unified Perceptual Parsing for Scene Understanding Year: (2018)
Ref_id:b44 Title: Swin Transformer V2: Scaling Up Capacity and Resolution Year: (2022)
Ref_id:b45 Title: Masked-Attention Mask Transformer for Universal Image Segmentation Year: (2022)
Ref_id:b46 Title: SegViT: Semantic Segmentation with Plain Vision Transformers Year: (2022)
Ref_id:b47 Title: Biformer: Vision Transformer with Bi-Level Routing Attention Year: (2023)
Ref_id:b48 Title: Superpixel Transformers for Efficient Semantic Segmentation Year: (2023)
Ref_id:b49 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b50 Title: DINOv2: Learning Robust Visual Features without Supervision Year: (2023)
Ref_id:b51 Title: Multi-scale representations by varying window attention for semantic segmentation Year: (2024)
Ref_id:b52 Title: Region rebalance for long-tailed semantic segmentation Year: (2022)
Ref_id:b53 Title: Segmenter: Transformer for semantic segmentation Year: (2021)
Ref_id:b54 Title: Hierarchical Saliency Detection Year: (2013)
Ref_id:b55 Title: Learning to Detect Salient Objects with Image-level Supervision Year: (2017)
Ref_id:b56 Title: Saliency detection via graph-based manifold ranking Year: (2013)
Ref_id:b57 Title: Self-supervised Transformers for Unsupervised Object Discovery using Normalized Cut Year: (2022)
Ref_id:b58 Title: Differentiable Vector Graphics Rasterization for Editing and Learning Year: (2020)
Ref_id:b59 Title: Image Quality Assessment: From Error Visibility to Structural Similarity Year: (2004)
Ref_id:b60 Title: Potrace : A Polygon-Based Tracing Algorithm Year: (2003)
Ref_id:b61 Title: Towards Layer-wise Image Vectorization Year: (2022)
Ref_id:b62 Title: Elementary mathematical theory of classification and prediction Year: (1958)
Ref_id:b63 Title: A Region-Based Approach to Diabetic Retinopathy Classification with Superpixel Tokenization Year: (2024)
Ref_id:b64 Title: SPSN: Superpixel Prototype Sampling Network for RGB-D Salient Object Detection Year: (2022)
Ref_id:b65 Title: Efficient active domain adaptation for semantic segmentation by selecting information-rich superpixels Year: (2024)
Ref_id:b66 Title: Lightweight Image Super-Resolution with Superpixel Token Interaction Year: (2023)
Ref_id:b67 Title: Superpixel Lattices Year: (2008)
Ref_id:b68 Title: SeMask: Semantically Masked Transformers for Semantic Segmentation Year: (2023)
Ref_id:b69 Title: CAR: Class-aware regularizations for semantic segmentation Year: (2022)
Ref_id:b70 Title: Efficient self-ensemble for semantic segmentation Year: (2022)
Ref_id:b71 Title: CLIP itself is a strong fine-tuner: Achieving 85.7% and 88.0% top-1 accuracy with ViT-B and ViT-L on imagenet Year: (2022)
Ref_id:b72 Title: MixUp: Beyond Empirical Risk Minimization Year: (2018)
