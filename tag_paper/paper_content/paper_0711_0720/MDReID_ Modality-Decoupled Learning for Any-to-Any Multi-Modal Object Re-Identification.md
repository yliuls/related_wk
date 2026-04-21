Title: MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification
Abstract: Real-world object re-identification (ReID) systems often face modality inconsistencies, where query and gallery images come from different sensors (e.g., RGB, NIR, TIR). However, most existing methods assume modality-matched conditions, which limits their robustness and scalability in practical applications. To address this challenge, we propose MDReID, a flexible any-to-any image-level ReID framework designed to operate under both modality-matched and modality-mismatched scenarios. MDReID builds on the insight that modality information can be decomposed into two components: modality-shared features that are predictable and transferable, and modality-specific features that capture unique, modalitydependent characteristics. To effectively leverage this, MDReID introduces two key components: the Modality Decoupling Learning (MDL) and Modality-aware Metric Learning (MML). Specifically, MDL explicitly decomposes modality features into modality-shared and modality-specific representations, enabling effective retrieval in both modality-aligned and mismatched scenarios. MML, a tailored metric learning strategy, further enforces orthogonality and complementarity between the two components to enhance discriminative power across modalities. Extensive experiments conducted on three challenging multi-modality ReID benchmarks (RGBNT201, RGBNT100, MSVR310) consistently demonstrate the superiority of MDReID. Notably, MDReID achieves significant mAP improvements of 9.8%, 3.0%, and 11.5% in general modality-matched scenarios, and average gains of 3.4%, 11.8%, and 10.9% in modality-mismatched scenarios, respectively. The code is available at: https://github.com/stone96123/MDReID.

Section: Introduction
Object Re-Identification (ReID) focuses on identifying and retrieving specific objects across nonoverlapping camera views. Conventional object re-identification (ReID) methods [1,2,3,4,5,6,7] primarily rely on RGB images. However, RGB-based approaches often face significant limitations under challenging environmental conditions, such as poor illumination, shadows, and low image resolution. These adverse conditions frequently lead to the extraction of misleading features, thereby diminishing discriminative capability. To overcome these limitations, multi-modal ReID [8,9,10,11,12,13], which integrates complementary information from multiple spectrums, has emerged as a promising approach. By leveraging the inherent strengths and complementary features of different modalities, multi-modal ReID significantly enhances feature representations, enabling more robust and accurate identification in complex real-world scenarios.
[RGB] [RGB, NIR, TIR] [RGB, TIR] Modality-matched Modality-mismatched [RGB, TIR] [NIR, TIR] [RGB, TIR]
[NIR]
[RGB]
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12']

Section: Modality-specific Feature

this section cite: []

Section: MDReID
Uncertain Gallery
[RGB] [RGB, NIR, TIR] [RGB, TIR] The introduction of multi-modal data has significantly enhanced the performance of ReID in challenging scenarios, thereby stimulating extensive research efforts in this field [8,9,10,11,12]. EDITOR [11] selects diverse tokens from Vision Transformer (ViT) to mitigate the impact of irrelevant backgrounds and narrow the gap between modalities, achieving strong detection performance. TOP-ReID [12] incorporates a cyclic token permutation module to aggregate multi-spectral features and a complementary reconstruction module to minimize the distribution gap across different image spectra. These innovations enable TOP-ReID to achieve robust performance under both modality-complete and modality-missing scenarios. Although these methods effectively leverage the characteristics of multi-modal data, they are developed based on a fundamental assumption that all modalities within the dataset are strictly aligned (as illustrated in Figure 1 (a) Modality-matched). Ideally, every camera would simultaneously provide RGB, NIR, and TIR modalities. However, due to practical constraints such as device differences and deployment diversity, achieving fully matched retrieval conditions is challenging in real-world scenarios. Therefore, it is crucial to develop a flexible image-level any-to-any ReID framework capable of effectively operating under both modality-matched and modality-mismatched scenarios (as illustrated in Figure 1 (a)).
To address this limitation, we propose MDReID, a flexible any-to-any ReID framework designed to support retrieval tasks involving arbitrary combinations of query and gallery modalities. The core challenge of this problem lies in learning effective representations that remain robust under both modality-matched and modality-mismatched conditions. A straightforward solution, as adopted by TOP-ReID [12], attempts to predict the missing modality representation from the available one, thereby enabling modality-aligned retrieval. However, as highlighted by RLE [14], predicting crossspectrum features solely from visual inputs constitutes an ill-posed problem. The modality-specific characteristics that are inherently unpredictable often lead to suboptimal learning. To this end, MDReID introduces a Modality-Decoupled Learning (MDL), which introduces a modality-shared and modality-specific token into the ViT architecture, leveraging multi-layer attention within the transformer to extract shared and specific features from multi-modal inputs. MDL explicitly disentangles the modality representations into two complementary components: a modality-shared component that captures the predictable cross-modal representation and a modality-specific component that preserves the unpredictable modality characteristics. This decoupling enables the model to better generalize across diverse modality combinations while maintaining the modality-specific cues. Furthermore, to enhance the disentanglement between the two components, we propose a Modalityaware Metric Learning (MML) strategy. MML comprises two complementary objectives: a representation orthogonality loss (ROL) and a knowledge discrepancy loss (KDL). ROL is applied at the channel level to both promote the aggregation of modality-shared features across modalities and enforce orthogonality between shared and specific components, ensuring they capture distinct and non-overlapping information. KDL, on the other hand, encourages representational complementarity between shared and specific components by ensuring that the combined representation is more discriminative than either component alone.
To sum up, the main contributions of this paper are as follows:
• We propose MDReID, a flexible any-to-any object re-identification framework that supports retrieval across arbitrary query-gallery modality combinations, addressing the practical limitations of strictly aligned multi-modal datasets.
• We introduce a Modality-Decoupled Learning (MDL) strategy, coupled with a Modalityaware Metric Learning (MML) strategy. MDL explicitly disentangles modality-shared and modality-specific representations, while MML further strengthens this disentanglement by introducing a representation orthogonality loss and a knowledge discrepancy loss, encouraging the two components to encode distinct and complementary information.
• Extensive experiments on multi-spectral object ReID datasets (RGBNT201, RGBNT100, MSVR310) demonstrate the superior adaptability and performance of the MDReID across diverse scenarios. MDReID achieves significant mAP improvements of 9.8%, 3.0%, and 11.5% in general modality-matched scenarios, and average gains of 3.4%, 11.8%, and 10.9% in modality-mismatched scenarios, respectively.
this section cite: ['b7', 'b8', 'b9', 'b10', 'b11', 'b10', 'b11', 'b11', 'b13']

Section: Related Work
Compared to the general RGB-to-RGB object re-identification [15,16,17,18,19,20,21], multimodal object re-identification (ReID) has demonstrated significant practical value, achieving widespread applications in cross-scenario systems [22,23] (e.g., security surveillance, intelligent transportation, etc.). Current methods [24,9,25] primarily enhance re-identification accuracy by effectively leveraging the complementarity and integration of multi-modal features. For example, Zheng et al. [26] proposed the cross-directional consistency network (CCNet), which overcomes discrepancies in both modality and sample aspects, thereby achieving more effective multi-modal data collaboration. To address the modal-missing problem, Zheng et al. [27] proposed the DENet model, which incorporates a feature transformation module to recover information from missing modalities and a dynamic enhancement module to improve multi-modality representation. Li et al. [8] proposed HAMNet, which automatically fuses different spectral features using a specially designed heterogeneous score coherence loss. He et al. [28] proposed the GPFNet model, which utilizes graph learning to fuse multi-modal features.
Notably, models based on Vision Transformer (ViT) [29,30,10,31,32,33] for object re-identification have achieved excellent results in recent years and gained widespread attention. Pan et al. [34] proposed the H-ViT model, which alleviates challenges caused by heterogeneous multi-modalities and reduces feature deviations arising from modal variations. Additionally, they introduced a progressively hybrid transformer (PHT [35]) that effectively fuses multi-modal complementary information through random hybrid augmentation and a feature hybrid mechanism. Crawford et al. [36] addressed the issue of modality laziness in multi-modal fusion by proposing the UniCat model based on a ViT architecture. By selecting diverse tokens from ViT to mitigate the impact of irrelevant backgrounds and narrow the gap between modalities, Zhang et al. [11] developed the EDITOR model. Wang et al. [12] designed the Top-ReID model, which achieves state-of-the-art accuracy by incorporating a cyclic token permutation module to aggregate multi-spectral features and a complementary reconstruction module to minimize the distribution gap across different image spectra.
Despite advancements in multi-modal object ReID, existing methods rely on modality-aligned conditions, limiting their adaptability to real-world scenarios. To address this, we introduce MDReID, which disentangles modality-shared and modality-specific features for effective retrieval across both matched and mismatched conditions. It incorporates Modality-Decoupled Learning (MDL) to refine feature separation and Modality-aware Metric Learning (MML) to enhance discrimination through orthogonality and knowledge discrepancy losses. As a result, MDReID enables robust object re-identification across diverse modality configurations.
this section cite: ['b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b8', 'b24', 'b25', 'b26', 'b7', 'b27', 'b28', 'b29', 'b9', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b10', 'b11']

Section: Methodology

this section cite: []

Section: MDReID: Any-to-any Object ReID
In ideal multispectral person re-identification (ReID) settings, complete RGB, NIR, and TIR modalities are available for both query and gallery (RNT-to-RNT). However, real-world deployments often suffer from different modalities due to heterogeneous sensor availability and deployment
RGB NIR TIR Linear Projection [𝐼 𝑠𝑝 𝑅 , 𝐼 𝑖𝑚𝑔 𝑅 ] [𝐼 𝑠𝑝 𝑁 , 𝐼 𝑖𝑚𝑔 𝑁 ] [𝐼 𝑠𝑝 𝑇 , 𝐼 𝑖𝑚𝑔 𝑇 ] [𝐼 𝑠𝑝 𝑅 , 𝐼 𝑠ℎ 𝑅 , 𝐼 𝑖𝑚𝑔 𝑅 ] Shared token [𝐼 𝑠𝑝 𝑁 , 𝐼 𝑠ℎ 𝑁 , 𝐼 𝑖𝑚𝑔 𝑁 ] [𝐼 𝑠𝑝 𝑇 , 𝐼 𝑠ℎ 𝑇 , 𝐼 𝑖𝑚𝑔 𝑇 ] Vision Transformer Block Vision Transformer Block … [𝐼 𝑠𝑝 𝑅 , 𝐼 𝑠ℎ 𝑅 ] [𝐼 𝑠𝑝 𝑁 , 𝐼 𝑠ℎ 𝑁 ] [𝐼 𝑠𝑝 𝑇 , 𝐼 𝑠ℎ 𝑇 ] 𝐼 𝑠𝑝 𝑇 𝐼 𝑠𝑝 𝑅 𝐼 𝑠𝑝 𝑁 [𝐼 𝑠ℎ 𝑅 ,𝐼 𝑠ℎ 𝑁 , 𝐼 𝑠ℎ 𝑇 ] 𝑳 𝒄𝒍𝒔 + 𝑳 𝒕𝒓𝒊 Target cross-correlation 𝒅 𝒔𝒉 𝒑 𝒅 𝒔𝒉 𝒏 [𝐼 𝑠𝑝 𝑅 , 𝐼 𝑠𝑝 𝑁 , 𝐼 𝑠𝑝 𝑇 , 𝐼 𝑠ℎ 𝑅 , 𝐼 𝑠ℎ 𝑁 , 𝐼 𝑠ℎ 𝑇 ] Modality-aware Metric Learning Representation Orthogonality Loss (ROL) Knowledge Discrepancy Loss (KDL) Anchor Positive Negative Target: 𝒅 𝒔𝒑+𝒔𝒉 𝒏 > 𝒅 𝒔𝒑 𝒏 , 𝒅 𝒔𝒑 𝒏 𝒅 𝒔𝒑+𝒔𝒉 𝒑 < 𝒅 𝒔𝒑 𝒑 , 𝒅 𝒔𝒉 𝒑 (a) Shared Feature (𝐼 𝑠ℎ ) 𝒅𝒔𝒑 𝒏 𝒅 𝒔𝒑 𝒑 (b) Specific Feature (𝐼 𝑠𝑝 ) 𝒅 𝒔𝒑+𝒔𝒉 𝒑 𝒅 𝒔𝒑+𝒔𝒉 𝒏 (c) Combined Feature ([𝐼 𝑠𝑝 , 𝐼 𝑠ℎ ]) constraints. To address this fundamental challenge, we propose MDReID, a flexible, any-to-any ReID framework that supports arbitrary combinations of input and output modalities. As illustrated in Fig. 2, MDReID adopts a Vision Transformer (ViT) backbone and integrates two core components: Modality-Decoupled Learning (MDL) and Modality-Aware Metric Learning (MML). MDL explicitly disentangles each modality's representation into modality-shared features (crucial for cross-modality retrieval) and modality-specific features (essential for preserving discriminative cues in matched scenarios). MML further enhances this decoupling by enforcing metric-level consistency across modalities. This design enables robust and flexible object re-identification in diverse and modality-incomplete environments, significantly advancing the applicability of ReID in practical settings.
this section cite: []

Section: Modality Decoupled Learning
Generally, with a ViT as the backbone, we process input images from different modalities (M ∈ {R, N, T }, corresponding to RGB, NIR, and TIR). Standard ViT procedure involves segmenting the image into non-overlapping patches, which are then linearly projected into patch embeddings as:
I M = {[I M cls ], I M 1 , I M 2 , . . . , I M n }, where M ∈ {R, N, T }.(1)
To explicitly model and decouple modality characteristics crucial for cross-modal ReID, we deviate from the standard use of a single [CLS] token. Instead, we prepend two distinct learnable tokens to the patch embedding sequence for each modality: a modality-specific token and a modality-shared token. This augmented sequence is then processed by the ViT encoder to get:
I M e = {[I M
sp , I M sh ], I M 1 , I M 2 , . . . , I M n }, where M ∈ {R, N, T }.
where I M sp denotes the modality-specific features, I M sh denotes the modality-shared features, and I M e represents the encoded features.
Leveraging these decoupled features from different modalities (I R sp ,I N sp ,I T sp ,I R sh ,I N sh ,I T sh ), we construct a unified representation for each sample using a fixed-dimensional structure and an associated availability mask. This allows consistent handling across different modality presence scenarios. Specifically, we define a potential full feature vector, v f ull , by concatenating all possible specific and shared components, assuming all three modalities (RGB, NIR, TIR) are present:
v f ull = I R
sp , I N sp , I T sp , I R sh , I N sh , I T sh .
Correspondingly, a binary availability mask, mask f ull , indicates the presence of each component:
mask f ull = [1, 1, 1, 1, 1, 1],(4)
where '1' signifies that the feature component at that position is present and valid, and '0' (as introduced below) signifies absence or invalidity.
The actual feature vector v and mask mask used for any given input sample are derived from this full structure based on the modalities available for that specific sample. If a modality M ∈ {R, N, T } is missing for a sample, the corresponding modality-specific feature I M sp and the modality-shared feature I M sh in the vector v are replaced with a zero vector 0. Also, the entries in the mask mask corresponding to these zeroed-out features (I M sp and I M sh ) are set to 0. For example, if the TIR modality (T) is missing for a sample, its feature vector v and mask mask would be:
v = I R sp , I N sp , 0, I R sh , I N sh , 0 , mask = [1, 1, 0, 1, 1, 0].(5)
This structured representation (v and mask) is used consistently across all scenarios, including both modality-matched and modality-mismatched retrieval. In modality-matched scenarios (e.g., RN-to-RN), both the query and gallery samples will have the same modalities present (or absent), resulting in identical structures for their masks (e.g., [1, 1, 0, 1, 1, 0] for both if T is missing). In modality-mismatched scenarios (e.g., R-to-N), the query and gallery samples will inherently have different available modalities. Their respective v and mask representations are generated according to the rules above based on their own available data. For instance, the RGB query sample would have v q = [I R sp , 0, 0, I R sh , 0, 0] and mask q = [1, 0, 0, 1, 0, 0]. And the NIR gallery sample would have v g = [0, I N sp , 0, 0, I N sh , 0] and mask g = [0, 1, 0, 0, 1, 0]. The subsequent retrieval process utilizes both the feature vectors (v q , v g ) and their corresponding availability masks (mask q , mask g ) to compute a similarity score that respects the decoupled nature of the features and handles missing modalities robustly. We define the total similarity, Sim(v q , v g ), as the combine of two components, i.e., the modality-specific similarity Sim sp and the modalityshared similarity Sim sh . The modality-specific similarity compares modality-specific features between identical modalities only. The similarity is calculated as the sum of dot products between corresponding specific features, masked for joint availability:
Sim sp = 1 sum(mask sp,q ) * sum(mask sp,g ) M ∈{R,N,T } (I M sp,q ) T (I M sp,g ) • mask M sp,q • mask M sp,g .
(6) The modality-shared similarity computes the similarity between all pairs of available shared features across modalities. Define V sh,q be a matrix with shared features I R sh,q , I N sh,q , I T sh,q (or zero vectors if unavailable) as columns, and similarly for V sh,g , mask sh,q and mask sh,g . We first compute the matrix of all pairwise shared feature similarities:
S sh = V T sh,q V sh,g .(7)
This results in a 3 × 3 matrix where (S sh ) ij is the dot product (I i sh,q ) T (I j sh,g ) for i, j ∈ {R, N, T }. To account for missing modalities, we define a pairwise availability mask matrix with the outer product of the shared masks:
M sh_pair = mask T sh,q • mask sh,g ,(8)
where (M sh_pair ) ij is 1 if and only if both the i-th shared feature of the query and the j-th shared feature of the gallery are present. The total shared similarity is the sum of all valid pairwise similarities, obtained by Hadamard product of S sh and M sh_pair :
Sim sh = 1 sum(M sh_pair ) i,j (S sh ⊙ M sh_pair )ij.(9)
Finally, the total similarity score is:
Sim total (v q , v g ) = (Sim sp + Sim sh )/2.(10)
This approach, based on a fixed-size zero-padded feature vector and an explicit availability mask, provides a flexible and robust mechanism for handling arbitrary combinations of modality presence and absence, effectively adapting to all modality-matched and mismatched object ReID scenarios addressed by our method.
this section cite: []

Section: Modality-aware Metric Learning
After MDL, the features are separated into modality-specific and modality-shared components. To achieve more complete and thorough feature decoupling, we propose a modality-aware metric learning approach. Note that, during the training phase, we assume all the modalities for the samples are available. The core objective of this metric learning is twofold: 1) Enhance Shared Feature Consistency: Promote high similarity between modality-shared features irrespective of their originating modality, which is crucial to bridge the modality gap in mismatched retrieval scenarios. 2) Reinforce Specific Feature Purity: Ensure modality-specific features are distinct from each other and also orthogonal to all modality-shared features. This preserves unique modality characteristics and prevents leakage between specific and shared representations.
To achieve these goals simultaneously, we define a representation orthogonality loss, L ROL , that minimizes the discrepancy between the computed pairwise similarities of the decoupled features and a predefined target similarity structure. Within our framework, we operate on the feature vector v and its availability mask mask generated for each sample, as defined previously. We compute the 6 × 6 pairwise similarity matrix V sim based on the L 2 -normalized vector v:
V sim (i, j) = v T i v j , for i, j ∈ {1, . . . , 6}.(11)
We then define the ideal target similarity matrix as:
A =        1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1        .(12)
Finally, the loss L ROL is then formulated as the sum of squared errors between the computed similarities V sim and the target similarities A, masked by M pair :
L ROL = 6 i=1 6 j=1 (V sim (i, j) -A(i, j)) 2 . (13
)
Additionally, the concept of triplet loss is incorporated to further enhance the model's feature decoupling capability. In other words, the goal of feature decoupling is to achieve a synergistic effect and prevent information from collapsing into modality-specific representations. To this end, as shown in Fig. 2, we introduce a knowledge discrepancy loss (KDL) to enforce the complementarity between shared and modality-specific features, ensuring that their combination yields better retrieval performance than using either alone. Specifically, within a batch of samples, given an anchor a, the positive samples P a denotes the samples with the same label as a while the negative samples N a denotes the opposite samples. The distances using specific feature from the anchor a to the batch of positive or negative samples are computed as follows:
d S (a, T ) = {∥I S (a) -I S (t)∥ 2 | t ∈ T },
where T ∈ {P a , N a }, S ∈ {sp + sh, sp, sh},
where S indicates the features used (modality-specific one, modality-shared one, or the combined one). Note that we omit the superscript representing the modality for simplicity, considering the modalities are completed, and the features of RGB, NIR, TIR modalities are concatenated to calculate the distance. Then, for the combination of shared and modality-specific features, the maximum positive sample distance should be as small as possible compared to using a single type of feature, and the minimum negative sample distance should be as large as possible compared to using a single type of feature. As shown in Fig. 2, this formulation can be described as: max(d sp+sh (a, p)) > max(d sp (a, p)), max(d sh (a, p)) and min(d sp+sh (a, n)) < min(d sp (a, n)), min(d sh (a, n)). Therefore, we define the knowledge discrepancy loss L KDL as:
L KDL = ∥D p -0∥ 1 + ∥D n -1∥ 1 ,
where D p = max(d sp+sh (a, p)) max (d sp+sh (a, p)) + max (d sp (a, p)) + max(d sh (a, p)) ,
D n = min(d sp+sh (a, n)) min (d sp+sh (a, n)) + min (d sp (a, n)) + min(d sh (a, n)) .(15)
It is worth noting that detach operations are applied to d sp (a, n) and d RN T sh (a, n) to avoid gradient computation. Thus, minimizing D p reduces the farthest positive sample distance for the combined shared and modality-specific features, while minimizing ∥D n -1∥ 1 increases the nearest negative sample distance. Consequently, improving the retrieval capability of the combined features enhances the performance of model. In summary, the loss function for modality-aware metric learning is defined as follows:
L M M L = w 1 × L ROL + w 2 × L KDL .(16)
this section cite: []

Section: Objective Function
As illustrated in Fig. 2, our model incorporates two primary loss functions: one for the ViT backbone and another for modality-aware metric learning. For the ViT backbone, we adopt label smoothing cross-entropy loss L ce and triplet loss L tri following previous research to optimize the representation. In summary, the total loss function is as follows:
L = L ce + L tri + L M M L .(17)
4 Experiments
this section cite: []

Section: Datasets and Evaluation Protocols
We evaluate the proposed MDReID framework on three publicly available datasets spanning both person (RGBNT201 [24]) and vehicle re-identification (RGBNT100 [8] and MSVR310 [26]) tasks. For evaluating modality mismatch scenarios, we focus on representative cross-modal configurations, including RT-to-NT, RT-to-N, R-to-N, and R-to-NT. Here, NIR is the second most widely used modality in surveillance systems after RGB, and RGB-to-NIR research continues to attract significant interest. Therefore, we evaluate our method in four R-to-N-based application scenarios. Performance is measured using mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) at ranks 1, 5, and 10 (R-1, R-5, R-10). We also report key complexity indicators, including the number of trainable parameters and floating point operations (FLOPs), to quantify the efficiency of the model.
RGBNT201 [24] is a multi-modality person re-identification dataset designed to overcome the limitations of single-modal imaging in challenging surveillance scenarios. It was captured on a university campus using four non-overlapping camera views with a synchronized triple-camera system that simultaneously records RGB, near-infrared (NIR), and thermal-infrared (TIR) images. The dataset contains 201 identities, each represented by at least 20 non-adjacent image triplets, resulting in 4,787 images per modality. The collection covers a range of real-world conditions, including severe illumination changes, occlusion, viewpoint variations, and background clutter. The dataset is split into 141 identities for training, 30 for validation, and 30 for testing, with the entire test set serving as the gallery and 10 records per identity sampled as probes. RGBNT201 thus provides a robust benchmark for evaluating multi-modal fusion strategies and addressing missing-modality issues in person re-identification.
RGBNT100 [8] contains 17,250 spatially aligned image triples from 100 vehicles captured under uniform conditions. Each triple includes RGB, NIR, and TIR images. Fifty vehicles (8,675 triples) are used for training, and the remaining 50 vehicles (8,575 triples) form the testing/gallery set, with 1,715 triples randomly selected as queries. The RGB and NIR images are recorded at a resolution of 1920×1080, while the TIR images are captured at 640×480, all at a consistent frame rate of 25 fps. TIR images provide thermal information that is robust to illumination changes, further enhancing the multi-spectral data for challenging vehicle re-identification scenarios.
MSVR310 [26] is a high-quality multi-spectral vehicle re-identification benchmark captured under diverse, challenging conditions. It contains 2,087 triplet samples (6,261 images in total) from 310 vehicles. Each sample includes spatially aligned images from three modalities: RGB, NIR, and TIR. RGB images are captured by a 360 D866 camera during the day and by a Mi8 mobile phone at night, NIR images are obtained with the 360 D866 in near-infrared mode, and TIR images are recorded with a FLIR SC620 camera at 640×480 resolution. The number of samples per vehicle ranges from 2 to 20, and time labels are provided for cross-time matching. The dataset is divided into 1,032 samples from 155 vehicles for training and 1,055 samples from 155 vehicles for testing/gallery, with 591 samples (from 52 vehicles) used as queries. MSVR310 provides a comprehensive platform for addressing intra-class appearance variations and modality differences to support robust vehicle re-identification.
this section cite: ['b23', 'b7', 'b25', 'b23', 'b7', 'b25']

Section: Implementation Details
All experiments are implemented in PyTorch and conducted on a single NVIDIA RTX 4090 GPU with CUDA 12.5 and Python 3.8. We adopt the CLIP-Base [37] visual encoder as the backbone. Input images are resized to 256 × 128 for RGBNT201 and 128 × 256 for RGBNT100 and MSVR310. We employ random horizontal flipping, cropping, and erasing [38] for data augmentation. The model is optimized using Adam with a batch size of 64. The base learning rate is initialized at 3.5 × 10 -4 , while the visual encoder is fine-tuned with a reduced rate of 5 × 10 -6 . Training is performed for 50 epochs.
this section cite: ['b36', 'b37']

Section: Comparison with State-of-the-Art Methods
Multi-spectral Object ReID (RNT-to-RNT) We compare our MDReID against a range of state-ofthe-art single-spectral and multi-spectral approaches on RGBNT201, RGBNT100, and MSVR310. For person ReID on RGBNT201, as shown in Table 1, the TOP-ReID [12] achieves 72.3% mAP, 76.6% R-1, 84.7% R-5, and 89.4% R-10. In comparison, our MDReID outperforms these results by 9.8%, 8.6%, 5.6%, and 3.2%, respectively, highlighting the effectiveness of our proposed MDReID. For vehicle re-ID, MDReID achieves the highest mAP on RGBNT100 (85.3%) and surpasses the nextbest approach on MSVR310 by a significant margin of 11.5% in mAP and 13.7% in R-1. These results clearly demonstrate the superior discriminative power and robustness of our modality-decoupled design in multi-spectral re-ID scenarios where all modalities are available.
Evaluation on Missing-spectral Scenarios. In practical applications, sensor limitations or environmental factors often lead to incomplete modality inputs. Following the setting of TOP-ReID [12], we evaluate the robustness of our method under all six possible missing-modality configurations across the RGB, NIR, and TIR channels using the RGBNT201 dataset. As shown in Table 2, our MDReID consistently outperforms the TOP-ReID, achieving an average improvement of 10.0% in mAP and 9.4% in Rank-1 accuracy across all missing-modality scenarios. These results demonstrate that our modality-decoupled framework remains highly effective even when partial modality information is absent, highlighting its strong generalization ability in incomplete multispectral settings.
this section cite: ['b11', 'b11']

Section: Evaluation on Modality-mismatched Scenarios.
To evaluate the generalizability of our approach under realistic deployment conditions, we benchmark MDReID across four representative modalitymismatched scenarios on three public datasets. We compare against two strong baselines, TOP-ReID [12] and EDITOR [11], by reproducing their results using official open-source implementations. As shown in Table 3, EDITOR is designed exclusively for modality-matched settings, which exhibits limited effectiveness when modalities differ across query and gallery. Although TOP-ReID attempts Table 2: Performance of missing-modality settings on RGBNT201. "M (X)" means missing the X image modality. The best and second results are in bold and underlined, respectively.
Methods M (RGB) M (NIR) M (TIR) M (RGB+NIR) M (RGB+TIR) M (NIR+TIR) Average mAP R-1 mAP R-1 mAP R-1 mAP R-1 mAP R-1 mAP R-1 mAP R
-1 Single MUDeep [39] 19.2 16.4 20.0 17.2 18.4 14.2 13.7 11.8 11.5 6.5 12.7 8.5 15.9 12.9 HACNN [41] 12.5 11.1 20.5 19.4 16.7 13.3 9.2 6.2 6.3 2.2 14.8 12.0 13.3 10.7 MLFN [43] 20.2 18.9 21.1 19.7 17.6 11.1 13.2 12.1 8.3 3.5 13.1 9.1 15.6 12.4 PCB [45] 23.6 24.2 24.4 25.1 19.9 14.7 20.6 23.6 11.0 6.8 18.6 14.4 19.7 18.1 OSNet [47] 19.8 17.3 21.0 19.0 18.7 14.6 12.3 10.9 9.4 5.4 13.0 10.2 15.7 12.9 Multi PFNet [24] --31.9 29.8 25.5 25.8 ----26.4 23.4 --DENet [27] --35.4 36.8 33.0 35.4 ----32.4 29.2 --TOP-ReID [12] 54.4 57.5 64.3 67.6 51.9 54.5 35.3 35.4 26.2 26.0 34.1 31.7 44.4 45.4 MDReID (Ours) 67.0 67.9 75.8 80.9 61.7 59.8 48.6 51.1 31.4 29.2 42.1 40.0 54.4 54.8
Table 3: Performance of modality-mismatched Scenarios. We show the best average score in bold.
this section cite: ['b11', 'b10']

Section: Methods

this section cite: []

Section: RT-to-NT RT-to-N R-to-N R-to-NT Average
mAP R-1 mAP R-1 mAP R-1 mAP R-1 mAP R-1 RGBNT201 EDITOR [11]
27.3 27.9 2.80 0.0 4.0 2.3 4.3 4.1 8.5 7.5 TOP-ReID [12] 43.0 44.6 14.4 13.3 15.4 14.0 11.9 8.7 18.2 18.0 MDReID (Ours) 53.1 51.7 16.7 13.8 16.6 11.1 15.1 10.9 21.6 19.1 RGBNT100 EDITOR [11] 42.1 59.9 2.6 0.8 2.8 1.5 2.7 1.5 11.9 15.5 TOP-ReID [12] 59.0 81.8 21.9 28.5 26.2 34.0 21.7 25.4 26.8 36.1 MDReID (Ours) 69.4 85.5 39.6 47.9 45.4 56.4 37.6 41.8 38.6 47.4 MSVR310 EDITOR [11] 6.4 11.0 2.2 2.5 1.6 0.2 1.7 1.5 2.5 3.4 TOP-ReID [12] 18.4 30.5 12.9 19.5 13.7 21.3 12.6 18.8 11.2 17.8 MDReID (Ours) 35.1 52.5 24.7 34.5 28.6 39.8 27.0 36.0 22. 1 31.7 to address missing modalities via reconstruction, its performance remains constrained due to the ill-posed nature of the reconstruction problem. In contrast, our MDReID framework consistently improves the average mAP by 3.4%, 11.8%, and 10.9% on the RGBNT201, RGBNT100, and MSVR310 datasets, respectively. In addition, we train four expert models on the RGBNT201 dataset for four specific scenarios. Results show that our method outperforms these experts by 9.2% on average in mAP and by 8.9% in R-1. Importantly, our single-model approach adapts to all four scenarios and offers greater flexibility. In summary, these results highlight the robustness and adaptability of MDReID in handling modality-mismatched ReID, affirming its effectiveness capable of supporting arbitrary modality combinations in both query and gallery inputs.
this section cite: ['b11', 'b10', 'b11', 'b10', 'b11']

Section: Ablation Study
To evaluate the effectiveness of each component in our proposed MDReID framework, we conduct comprehensive ablation studies on the RGBNT201 dataset. To ensure a thorough evaluation across both modality-matched and modality-mismatched scenarios, we report the average performance over 8 evaluation settings, including RNT-to-RNT, RT-to-RT, RT-to-NT, RT-to-N, R-to-N, and R-to-NT. The detailed results are presented in Table 4 (a). Note that configuration "1" corresponds to using a single classifier for all modalities, while configuration "3" corresponds to assigning a separate classifier to each modality. Ablation results show that, compared with one classifier, modality-specific classifiers effectively improve performance, increasing mean mAP by 13.4% and mean R-1 by 13.7%.
this section cite: []

Section: Modality Decoupled Learning
Our base framework integrates a shared vision transformer, utilizing label smoothing cross-entropy along with triplet loss for optimization. To better address modality-mismatched object re-identification, we design a modality decoupling learning (MDL) that separates features into shared and specific representations. Experimental results demonstrate that this module boosts performance, with an average increase of 11.5% in mAP and 11.1% in Rank-1. This demonstrates that modality decoupling not only benefits re-identification in mismatched settings but also enhances retrieval performance in matched scenarios.
this section cite: []

Section: Modality-aware Metric Learning
To further decouple shared and specific features, we design a modality-aware metric learning (MML) loss function. This loss consists of two components: the first aligns shared features across multiple modalities, while the second enhances the decoupling of features based on the aligned representation. Experimental results show that introducing ROL improves average mAP and Rank-1 by 1.8% and 2.7%, respectively. Similarly, introducing KDL increases mAP by 0.5% and Rank-1 by 1.7%. When both components are combined, the average mAP and Rank-1 improve by 3.8% and 4.1%, respectively. These results indicate that ROL significantly enhances re-identification accuracy in mismatch scenarios. Moreover, while KDL alone provides limited improvement, its combination with ROL effectively boosts overall model performance.
this section cite: []

Section: Modality Decoupled Learning
We design the KDL objective to enforce a knowledge gap between specific and shared features. Without KDL, the model may degrade into relying only on shared features. Therefore, we ensure that combining specific and shared features yields higher retrieval performance than using either feature alone. To achieve this, we apply formulation 15 to impose distance constraints in the feature space based on the described relations. As shown in Table 4 (a), index 3 and 5, adding KDL improves mAP by 2.0% and Rank-1 by 1.5% across multiple scenarios. In summary, the MDReID framework, comprising the modality decoupled learning and modality-aware metric learning, achieves an average performance of 43.2% mAP, 42.3% Rank-1, 50.2% Rank-5, and 54.9% Rank-10 across various scenarios, including both modality-matched and modality-mismatched cases. Its modular design effectively decouples shared and specific features, enhancing object reidentification accuracy under multi-spectral conditions while maintaining robust adaptability across diverse modalities.
this section cite: []

Section: Discussions 4
Hyper-parameter settings of MDReID In Eq. ( 16), we introduce two hyperparameters w 1 and w 2 to balance the importance of ROL and KDL. Therefore, this part evaluates the performance under different hyperparameter settings and shows the results in Table 4. We first vary w 1 without KDL, observing that the model achieves optimal performance at w 1 = 1.5, highlighting the importance of enforcing orthogonality between shared and specific components. Subsequently, with w 1 = 1.5, we tune w 2 and find that the best results are obtained at w 2 = 5.25, confirming the benefit of encouraging representational synergy and complementarity. These observations validate that a contribution from both ROL and KDL is crucial for maximizing retrieval accuracy.
this section cite: []

Section: Conclusion
In this paper, we propose MDReID, an image-level any-to-any object re-identification framework that supports retrieval across arbitrary query-gallery modality combinations. Our approach leverages Modality-Decoupled Learning (MDL) to explicitly disentangle modality-shared and modality-specific features, while Modality-aware Metric Learning (MML) refines the feature space to enhance discrimination. Extensive evaluations on the RGBNT201, RGBNT100, and MSVR310 datasets validate MDReID's superior performance in both modality-matched and modality-mismatched scenarios, demonstrating its practical potential for any-to-any ReID tasks. Specifically, in (a), with only MDL applied, the features exhibit noticeable overlap, and the separation between modality-specific and shared components remains unclear. In (c), the addition of KDL slightly improves the clustering and separation of shared and specific features. In contrast, (b) shows that introducing ROL significantly enhances feature orthogonality, forming clearer boundaries between shared and modality-specific clusters. Finally, (d) demonstrates that the combination of both ROL and KDL yields the most structured and disentangled feature space, with shared features tightly clustered and well-separated from their modality-specific counterparts. These results confirm the complementary roles of ROL and KDL in refining the representation space.
this section cite: []

Section: A.5 Limitations and Broader Impact
Although our work is the first to explore object re-identification under the uncertain query-gallery modality setting and achieves significant performance improvements, the current accuracy is still insufficient for deployment in real-world applications. Furthermore, existing datasets are limited in both scale and modality diversity, making it unclear how well MDReID would generalize to larger and more complex datasets involving a broader range of modality types. Nevertheless, MDReID represents the first effort to address the any-to-any object re-identification task, marking a significant step toward this emerging direction. We believe our work will inspire new perspectives and challenges in the re-identification community.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: This paper poses no such risks since it does not release data or models that have risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: This paper has properly cited the original paper that produced the code package or dataset.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
this section cite: []

Section: A Supplemental Material

this section cite: []

Section: A.1 Analysis of model computational complexity
Table 5 compares the computational cost of our method with recent state-of-the-art approaches. MDReID achieves the most efficient design, requiring only 57.2M parameters and 22.3G FLOPs, which is significantly lower than HTT (85.6M / 33.1G), EDITOR (117.5M / 38.6G), and TOP-ReID (278.2M / 34.5G). Despite its efficient architecture, MDReID consistently outperforms these heavier models in accuracy, demonstrating a superior trade-off between performance and efficiency. This highlights the effectiveness of our MDL and MML designs, which provide robust alignment in all kinds of query-gallery combinations without incurring large computational overhead. Furthermore, we evaluate the total feature generation and retrieval times of different models on the RGBNT201 test set under the RNT-to-RNT, RT-to-NT, and R-to-N scenarios. Theoretically, our method reduces retrieval cost compared to Top-ReID. For example, in the RT-to-NT scenario, we only perform T-to-T (specific features) and R-to-T (shared features) matching, whereas Top-ReID first generates missing-modality features before conducting a full RNT-to-RNT matching. Experimental results confirm this, showing that our method incurs no additional time and generates features faster.
Table 5: Comparison of computational cost with recent methods. We show the best result in bold.
this section cite: []

Section: Methods

this section cite: []

Section: Params(M) Flops(G) RNT-to-RNT(s) RT-to-NT(s) R-to-N(s)
Extract Retrieval Extract Retrieval Extract Retrieval HTT [10] 85.6 33.1 8.32 0.12 ----EDITOR [11] 117.5 38.6 31.12 0.12 30.99 0.12 30.98 0.12 TOP-ReID [12] 278.
2 34.5 71.78 0.12 73.32 0.11 73.71 0.12 MDReID (Ours) 57.2 22.3 3.27 0.14 2.90 0.11 3.14 0.12
this section cite: ['b9', 'b10', 'b11']

Section: A.2 Analysis of specific and shared features
In MDReID, both shared and modality-specific features are essential. For the RT-to-NT task, using only shared features overlooks the unique characteristics of the T modality, while using only modalityspecific features prevents valid R-to-N matching. To quantify this, we evaluate RT-to-NT performance under three settings: shared features only, modality-specific features only, and both combined. As shown in Table 6, combining shared and modality-specific features yields the best performance. We compare our method with the cross-modality object re-identification model DEEN [50] on the RGBNT201 dataset under the R-to-N scenario. Table 7 shows that even when DEEN is specially trained for R-to-N, its mean precision is 3.5% lower than ours. Notably, our model demonstrates stronger generalization across diverse application scenarios.
this section cite: ['b49']

Section: A.4 Visualization
To better illustrate the impact of different components on feature disentanglement, we visualize the learned modality-specific and modality-shared features across RGB, NIR, and TIR using t-SNE, as shown in Figure 3. Notably, the baseline method does not use MDL, so it does not separate specific and shared features and only outputs fused features, hence no corresponding visualization.
this section cite: []

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction of this paper accurately reflect the paper's contributions and scope.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We have discussed the limitations of this work in Section A.5.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We have introduced all the details in our theoretical result. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: This paper has provided detailed information for reproducing the experimental results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code is available at: https://github.com/stone96123/MDReID. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental Setting/Details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: This paper has specified all the training and test details, such as dataset and hyperparameters.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7.
this section cite: []

Section: Experiment Statistical Significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: This paper does not provide error bars due to the limited resources. Also, previous methods do not provide error bars either.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments Compute Resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes]
Justification: This paper has provided sufficient information on the computer resources.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code Of Ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: This paper meets the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader Impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: This paper has no societal impact to the best of our knowledge.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New Assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: This paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: This paper only uses LLMs for writing and editing. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Transreid: Transformerbased object re-identification Year: (2021-10)
Ref_id:b1 Title: Dc-former: Diverse and compact transformer for person re-identification Year: (2023)
Ref_id:b2 Title: Pha: Patch-wise highfrequency augmentation for transformer-based person re-identification Year: (2023)
Ref_id:b3 Title: Clip-reid: exploiting vision-language model for image reidentification without concrete text labels Year: (2023)
Ref_id:b4 Title: Denoising model for representation learning Year: (2024)
Ref_id:b5 Title: Dynamic prototype mask for occluded person re-identification Year: (2022)
Ref_id:b6 Title: Partformer: Awakening latent diverse representation from vision transformer for object re-identification Year: (2024)
Ref_id:b7 Title: Multi-spectral vehicle re-identification: A challenge Year: (2020-04)
Ref_id:b8 Title: Ran He, and Jin Tang. Interact, embed, and enlarge: Boosting modality-specific representations for multi-modal person re-identification Year: (2022-06)
Ref_id:b9 Title: Heterogeneous test-time training for multimodal person re-identification Year: (2024-03)
Ref_id:b10 Title: Magic tokens: Select diverse tokens for multi-modal object re-identification Year: (2024-06)
Ref_id:b11 Title: Topreid: Multi-spectral object re-identification with token permutation Year: (2024-03)
Ref_id:b12 Title: Multi-modal object re-identification via sparse mixture-of-experts Year: ()
Ref_id:b13 Title: Rle: A unified perspective of data augmentation for cross-spectral re-identification Year: (2024)
Ref_id:b14 Title: Diverse part discovery: Occluded person re-identification with part-aware transformer Year: (2021)
Ref_id:b15 Title: Deep learning for person re-identification: A survey and outlook Year: (2022)
Ref_id:b16 Title: Towards efficient object re-identification with a novel cloud-edge collaborative framework Year: (2025)
Ref_id:b17 Title: Occluded person re-identification via saliency-guided patch transfer Year: (2024)
Ref_id:b18 Title: Doubly contrastive learning for source-free domain adaptive person search Year: (2025)
Ref_id:b19 Title: Roda: Robust domain alignment for cross-domain retrieval against label noise Year: (2025)
Ref_id:b20 Title: Dida: Disambiguated domain alignment for cross-domain retrieval with partial labels Year: (2024)
Ref_id:b21 Title: Dica: Disambiguated contrastive alignment for cross-modal retrieval with partial labels Year: (2025)
Ref_id:b22 Title: Diverse embedding expansion network and low-light crossmodality benchmark for visible-infrared person re-identification Year: (2023)
Ref_id:b23 Title: Robust multi-modality person re-identification Year: (2021-05)
Ref_id:b24 Title: Generative and attentive fusion for multi-spectral vehicle re-identification Year: (2022)
Ref_id:b25 Title: Multi-spectral vehicle re-identification with cross-directional consistency network and a high-quality benchmark Year: (2023)
Ref_id:b26 Title: Dynamic enhancement network for partial multi-modality person re-identification Year: (2023)
Ref_id:b27 Title: Graph-based progressive fusion network for multi-modality vehicle re-identification Year: (2023)
Ref_id:b28 Title: Transformer for object re-identification: A survey Year: (2024)
Ref_id:b29 Title: Other tokens matter: Exploring global and local features of vision transformers for object re-identification Year: (2024)
Ref_id:b30 Title: Representation selective coupling via token sparsification for multi-spectral object reidentification Year: (2024)
Ref_id:b31 Title: Occlusion-aware transformer with second-order attention for person re-identification Year: (2024)
Ref_id:b32 Title: Aaformer: Auto-aligned transformer for person re-identification Year: (2023)
Ref_id:b33 Title: H-vit: Hybrid vision transformer for multi-modal vehicle re-identification Year: (2022)
Ref_id:b34 Title: Progressively hybrid transformer for multi-modal vehicle re-identification Year: ()
Ref_id:b35 Title: Unicat: Crafting a stronger fusion baseline for multimodal re-identification Year: (2023)
Ref_id:b36 Title: Learning transferable visual models from natural language supervision Year: (2021-07)
Ref_id:b37 Title: Random erasing data augmentation Year: (2020-04)
Ref_id:b38 Title: Multi-scale deep learning architectures for person re-identification Year: (2017-10)
Ref_id:b39 Title: Deep meta metric learning Year: (2019-10)
Ref_id:b40 Title: Harmonious attention network for person reidentification Year: (2018-06)
Ref_id:b41 Title: Bag of tricks and a strong baseline for deep person re-identification Year: (2019-06)
Ref_id:b42 Title: Multi-level factorisation net for person re-identification Year: (2018-06)
Ref_id:b43 Title: Circle loss: A unified perspective of pair similarity optimization Year: (2020-06)
Ref_id:b44 Title: Beyond part models: Person retrieval with refined part pooling (and a strong convolutional baseline) Year: (2018-09)
Ref_id:b45 Title: Heterogeneous relational complement for vehicle re-identification Year: (2021-10)
Ref_id:b46 Title: Omni-scale feature learning for person re-identification Year: (2019-10)
Ref_id:b47 Title: Counterfactual attention learning for fine-grained visual categorization and re-identification Year: (2021-10)
Ref_id:b48 Title: Graft: Gradual fusion transformer for multimodal re-identification Year: (2023)
Ref_id:b49 Title: Diverse embedding expansion network and low-light crossmodality benchmark for visible-infrared person re-identification Year: (2023-06)
