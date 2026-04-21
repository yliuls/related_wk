Title: SHF: Symmetrical Hierarchical Forest with Pretrained Vision Transformer Encoder for High-Resolution Medical Segmentation
Abstract: This paper presents a novel approach to addressing the long-sequence problem in high-resolution medical images for Vision Transformers (ViTs). Using smaller patches as tokens can enhance ViT performance, but quadratically increases computation and memory requirements. Therefore, the common practice for applying ViTs to high-resolution images is either to: (a) employ complex sub-quadratic attention schemes or (b) use large to medium-sized patches and rely on additional mechanisms within the model to capture the spatial hierarchy of details. We propose Symmetrical Hierarchical Forest (SHF), a lightweight approach that adaptively patches the input image to increase token information density and encode hierarchical spatial structures into the input embedding. We then apply a reverse depatching scheme to the output embeddings of the transformer encoder, eliminating the need for convolution-based decoders. Unlike previous methods that modify attention mechanisms or use a complex hierarchy of interacting models, SHF can be retrofitted to any ViT model to allow it to learn the hierarchical structure of details in high-resolution images without requiring architectural changes. Experimental results demonstrate significant gains in computational efficiency and performance: on the PAIP WSI dataset, we achieved a 3∼32× speedup or a 2.95%∼7.03% increase in accuracy (measured by Dice score) at a 64K 2 resolution with the same computational budget, compared to state-of-the-art production models. On the 3D medical datasets BTCV and KiTS, training was 6× faster, with accuracy gains of 6.93% and 5.9%, respectively, compared to models without SHF.

Section: Introduction
Recently, Transformers have been rapidly adopted in the field of computer vision [1,2]. Building on the self-attention mechanism, Vision Transformers (ViTs) and their variants have achieved significant advancements in various image classification and downstream visual tasks [3][4][5][6][7]. Text tokens are atomic, semantically distinct, and rich in information, whereas visual tokens are geometrically related and sparse in semantics. In other words, feeding a sequence of image patches to a transformer encoder deprives the self-attention mechanism of direct information about spatial hierarchy.
The loss of spatial hierarchy information becomes more pronounced when working with highresolution or multi-dimensional medical images, as the spatial hierarchy becomes more detailed and intricate [8]. This requires the use of very small patches so that the self-attention mechanism can capture local features [9]. However, using smaller patches quadratically increases the computational and memory costs of self-attention, prompting several approaches to address this issue by modifying the model architecture to help it learn the hierarchical structure in images. Most of these approaches fall into two broad categories: model-hierarchical and attention-hierarchical. Model-hierarchical solutions involve training ViTs hierarchically, with multiple transformers operating at different resolution levels [10,8,11,12]. While this approach can improve model performance, it also increases training time and memory usage. Additionally, managing multiple interacting transformers adds complexity, requiring extensive hyperparameter tuning at each resolution level. Attentionhierarchical solutions alter the patching scheme at the self-attention stage to represent hierarchical features, as seen in Swin [13,14], MViT [15], and MViTv2 [16]. Although these methods are more parameter-efficient than standard ViTs [4], they introduce additional spatial operations, increasing model complexity and reducing multi-modal capabilities.
From the above summary, two questions arise: (1) How can we design a patching strategy that uses the fewest possible patches to represent the original image, thus increasing the information density of each patch to maintain model performance, without altering the structure of the ViT model? (2) If an effective patching strategy could be developed to relay spatial hierarchical information to the model, could we leverage a post-encoder adaptive dispatching strategy during inference to eliminate the need for post-encoder convolution decoding?
To address the two questions above, this paper proposes Symmetrical Hierarchical Forest (SHF), which adapts the patching strategy to the hierarchical details of each training example. This approach enables the ViT to capture hierarchical local spatial information typically derived either from postencoder convolutions (e.g. the convolution decoder used in the SAM 2 model [17]) or hierarchical architecture features (i.e. Swin [14] and HIPT [8]). We downscale patches in regions with fewer details, aligning them to the patch size used for regions with more details. However, without expert knowledge of hyperparameters, we rely solely on hierarchical forests to extract/represent the spatial hierarchy information. Using the SHF scheme, we demonstrate that the model receives sufficient information about the spatial hierarchy, allowing us to eliminate additional model components (such as U-Net [18] or convolution decoding blocks) that would otherwise be required. Notably, the use of additional components (e.g. the convolution decoder in SAM 2 [17]) can lead to high memory demands to store activations for high-resolution masks (e.g. over 20GB of memory for storing mask activations when using SAM 2 on 64K 2 images). By employing a transformer encoder-only design with hierarchical forest and post-depatching, we can use smaller patch sizes, enabling self-attention to capture the spatial hierarchy more effectively than larger patches and additional mechanisms (such as U-Net or convolution decoding blocks). As an added benefit, our method simplifies model design and allows for swapping in different encoders, as it is a data-based approach that operates on the input and output of the transformer encoder without modifying the encoder itself.
The contributions of this work are as follows:
• Symmetrical Hierarchical Forest. By applying the Symmetrical Hierarchical Forest (SHF), we can extract the hierarchical information directly and eliminate the expert knowledge required for tuning hyperparameters. Additionally, we completely discard the convolution decoder, significantly reducing the computational and memory overhead (∼75% GPUs) of mask processing in high-resolution segmentation tasks. Finally, by downscaling redundant regions in the input image space, we achieve a quadratic reduction in the computational cost of the ViT encoder.
• Long Context Segmentation. To demonstrate the efficiency of SHF, we conducted experiments on high-resolution medical imaging datasets, retrofitting state-of-the-art (SoTA) models such as SAM 1 [19] and SAM 2 [17] with our SHF scheme in place of their convolution decoders. When comparing models retrofitted with SHF to SoTA models without SHF, high-resolution pathology datasets (e.g. the PAIP dataset, ranging from 512 2 to 64K 2 pixels) and 3D MRI datasets (BTCV, KiTS) benefit from the efficiency of SHF, allowing patch sizes as small as 2×2 pixels and 2×2×2 voxels. At the same performance level, we achieve a 3× to 32× speedup on the PAIP dataset, or, with the same computational budget, a 7.03% increase in Dice score at 64K 2 resolution. On 3D medical imaging datasets, such as BTCV and KiTS, we see a ∼6× training speed improvement along with performance increases of 6.93% and 5.9%, respectively, compared to SoTA models without SHF.
Figure 1: We compared the segmentation difference between SAM [19] and SAM retrofitted with our SHF scheme (SHF-SAM) instead of the original convolution decoder for PAIP [20] at 16, 384 2 , 32, 768 2 , and 65, 536 2 resolutions. At the same GPU budgets, SHF-SAM can go down to batch size of 8x8 (vs. 1, 024x1, 024 at best for SAM before going OOM). As a result, SHF-SAM can extract and express mask details better than SAM, with the gap in accuracy favoring SHF-SAM as the resolution gets higher.
• Simplicity and Low-Overhead. Unlike existing methods that modify the self-attention or transformer encoder mechanisms, our solution preserves the original self-attention mechanism. This ensures seamless retrofitting of SHF into any vision transformer. SHF is a low-overhead pre-processing and post-processing solution that is further amortized over epochs, making the overhead effectively negligible.
The rest of this paper is organized as follows: Section 2 reviews related work. Section 3 presents the methodology. Section 4 describes the experimental setup. Section 5 reports the evaluation results. Finally, Section 6 concludes the paper and outlines future work.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b7', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b3', 'b16', 'b13', 'b7', 'b17', 'b16', 'b18', 'b16', 'b18', 'b19']

Section: Related Work
The computational cost of self-attention increases as the patch size decreases. To mitigate this, several strategies have been developed. Sequence parallel methods, including Deep-Speed Ulysses [21], LightSeq [22], RingAttention [23], LLS [24], FlashAttention [25], and [26]. Linear approximation methods, such as spectral attention [27][28][29], low-rank approximation [30,31], sparse attention matrix sampling [32][33][34][35][36], infrequent self-attention updates [37,38], or combinations of these [39]. These methods reduce the computational load of the attention mechanism; however, excessive reduction SHF begins with the original image and ends with feeding the extracted patches (tokens) into an intact transformer-based model. In a real training example on the SAM [19] model using 512×512 images from the PAIP [20] liver cancer dataset, SHF reduces the number of patches from 4,096 to 512 (each of size 4×4) while maintaining the same Dice score. This results in an ∼8× reduction in sequence length and a ∼7.53× speedup in end-to-end training. can lead to performance loss, as reported in the literature [40]. Hierarchical training of ViTs, where multiple transformers are trained at different resolution levels [10,8,11,12]. However, using multiple transformers increases training time and memory usage, and managing multiple interacting transformers is complex. Recently, quadtrees have been used in image segmentation to reduce attention cost, e.g. quadtree/octree attention or patch pre-processing [41][42][43]. Both of those approaches employ quadtrees, but involve additional model complexity or need expert knowledge in the processing stage for hyperparameters.
this section cite: ['b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b18', 'b19', 'b39', 'b9', 'b7', 'b10', 'b11', 'b40', 'b41', 'b42']

Section: 𝑥: Image

this section cite: []

Section: 𝑦: Mask

this section cite: []

Section: Methodology

this section cite: []

Section: Vision Transformers and Attention
The self-attention mechanism in transformers computes attention scores A between input tokens, forming the attention matrix. Let x ∈ R N ×F denote a sequence of N feature vectors of dimensions F . A transformer is a function T : R N ×F → R N ×F defined by the composition of L transformer layers T 1 (•), ..., T L (•) as follows:
T r l (x) = f l (A l (x) + x)(1)
A l (•) is the self-attention function. The function f l (•) transforms each feature independently of the others and is usually implemented with a small two-layer feedforward network. Formally, the input sequence x is projected by three matrices W Q ∈ R F ×D , W K ∈ R F ×D , and W V ∈ R F ×D , to corresponding representations Q, K and V . Thus, the attention scores are calculated as follows:
Q = xWQ, K = xWK , V = xWV , Aij = Softmax (QiK T j )/ d k(2)
where Q i and K j are query and key vectors for tokens i and j, and d k is the dimension of the key vectors. The complexity of the attention matrix is O(N 2 ), where N is the sequence length. We further assume that the input is the content of a square image x with a resolution of Z, that is, let x ∈ R Z×Z , and by assuming that patches arise from the uniform grid patch method of patch size p.
Thus the sequence N = (Z/P ) 2 . The total computation and memory cost of attention scores defined in Eqn 2 according to resolution and patch size is O([Z/P ] 4 ). This complexity shows the difficulties of increasing the resolution while decreasing the patch size P with the uniform grid patch strategy.
this section cite: []

Section: Symmetrical Hierarchical Forest (SHF)
In the following paragraphs, we describe how SHF works, following the steps outlined in Fig. 2:
Hiera-Edges Detection. We aim to use different methods to extract hierarchical details, as this would allow us to augment the dataset. This, however, is different from traditional augmentation applied at the image level; we augment by providing the model with different views of the spatial hierarchy for each image, thereby giving the model a better opportunity to learn the hierarchy. To use different ways to extract the hierarchical details in the image x ∈ X, as Fig. 2-1, we use a Gaussian blur with kernel k and Canny [44] edge detection with threshold l to the original input images x ∈ X. The Gaussian blur with kernel k smooths the irrelevant details, and the Canny edge detection with threshold l extracts the grayscale edges x e of the image. To generate different spatial structures that can also recover the original inputs, during our experiments, we randomly choose the threshold to be in the range [100,200], and the kernel size is randomly set to one of [3,5,7,9,11,13].
this section cite: ['b43', 'b2', 'b4', 'b6', 'b8', 'b10', 'b12']

Section: Hierarchical Forest.
In the Hierarchical Forest stage (Fig. 2-2), we build several quadtrees (octrees in 3D) from each x e ; x e undergoes a recursive tree partitioning. To construct the tree T , we create tree nodes T n representing specific regions where n is the number of leaf nodes. The leaf nodes T n+s-1 is defined recursively as follows:
T n+s-1 = Tn, if n ≥ N Tn[i] = {T 1 n , T 2 n , ..., T s n }, arg max i V (Tn[i])(3)
where V is the criterion we use to differentiate the different levels of detail. We choose the maximum sum of pixel values among the tree nodes by i = arg max i V (T n [i]), {T 1 n , T 2 n , ..., T s n } are the s new child nodes after the subdivision of T n [i], s = 2 d is the number of subdivisions and d is the number of dimensions. T can be used for both 2D and 3D tasks, for example s = 4, 8 which means quadtree and octree, respectively [45,46]. In our implementation, to avoid unnecessary padding or dropping due to varying sample lengths, we control the number of splits at the leaf nodes to keep each sample at the same sequence length N . This approach fully utilizes the GPU resources without discarding sample information. The sequence length N is set to [1024, 4096, 8192, 16384, 16384, 16384] with respect to resolutions, which practically allows the input x e to be subdivided all the way down to the 2 × 2 patch size level.
After building the tree, as a property of quad/octrees, visiting all the tokens (appearing as leaves in the tree) from left to right gives the token sequence as a Z-order space-filling curve in the image space [47]. This operation( ti = ⇒) results in a sequence of image patches shown in step Fig. 23. We not only use spatial trees to encode images x, but we also encode masks y. Using the same tree and z-order of the mask will also result in a sequence of mask patches y i . Since the spatial trees are built from the image, the sequence length of the image and mask patches (x i , y i ) should be the same N .
Image Encoder. After completing hierarchical forest patching, we obtain an image patch sequence and a mask patch sequence of the same length, allowing us to (continuously) train the ViT model. We use the well-known pre-trained image encoder from SAM in steps Fig. 234and then continue training the model on our dataset after retrofitting our SHF scheme into the SAM model. SAM uses an MAE [2] pre-trained ViT [4], minimally adapted to process high-resolution inputs. For 3D MRI data tasks, we use the SAM 2 [17] encoder and similarly retrofit SHF into SAM 2. When retrofitting SHF into both SAM and SAM 2, we remove the convolution decoder that is part of the original SAM and SAM 2 models. It should be noted that during the training phase of the model, we do not directly generate masks y ′ but instead generate encoded masks y ′ i . The advantage of this approach is that it saves memory, which would otherwise be needed for high-resolution masks, and reduces the backpropagation compute costs associated with the heavy decoders. We summarized the objective function as follows:
min θ E (x,y)∼D k∼K,l∼L M i=1 L hiera f θ T tree (x, k, l) i , T sym (y, k, l) i (4
)
Where M is total number of samples, θ represents the trainable weights, L hiera denotes the Dice loss, (k, l) specify the Gaussian filter size and Canny threshold, and T tree = T sym are the recursive tree operators applied to both the input x and the mask y.
this section cite: ['b44', 'b45', 'b46', 'b1', 'b3', 'b16']

Section: Symmetrical Depatching.
The function of depatching (
t ′ i ⇐ =
) is to upscale the sequence obtained from the ViT into a mask during the inference stage. By depatching, we can eliminate the need for general U-Net or lightweight decoders and calculate the backpropagation gradient directly at the output of the encoder (in Fig. 2345). This is because we have observed that the information in the mask itself is sparser compared to the input image (see the Fig. 4). Therefore, when reconstructing the mask at the evaluation stage, we simply use the same quad/octree structure derived from the image to linearly upscale the patches at the corresponding positions in the sequence.
this section cite: []

Section: Experimental Setup

this section cite: []

Section: High-Resolution Medical Image Datasets: PAIP, BTCV, & KiTS
PAIP: [20] is a high-resolution, real-world liver cancer pathology dataset, with sample resolutions up to 64K 2 , significantly surpassing those of conventional image datasets. PAIP contains 2, 457 Whole-Slide Images (WSIs). When lower resolutions are needed, we downscale the images to uniform sizes of [512, 1024, 4096, 8192, 16384, 32768, 65536] square pixels. For training, we randomly select 70% of samples, 10% for validation, and 20% for testing. All datasets are shuffled and normalized to [0.0, 1.0] as input for the model. The BTCV: challenge [48] for 3D multi-organ segmentation includes 30 subjects with abdominal Computed Tomography (CT) [49] scans, with 13 organs annotated by experts. Each CT scan consists of 80 to 225 slices, each with 512 2 pixels. The multi-organ segmentation task is defined as a 13-class segmentation problem, where the average dice score across all classes is typically reported. Although BTCV has a lower resolution compared to the PAIP dataset (512 2 vs. 64K 2 ), it remains a widely used benchmark in the high-resolution medical segmentation community. KiTS: the 2019 Kidney and Kidney Tumor Segmentation Challenge (KiTS19 [50]) aimed to develop algorithms for segmenting kidneys and kidney tumors in contrast-enhanced 3D CT scans. The dataset consists of 300 anonymized scans in NIfTI (.nii.gz) format, each manually annotated with kidney and tumor labels. Each scan has a typical resolution of 512×512 pixels per slice, with the number of slices varying based on patient anatomy. KiTS is a common benchmark for 3D MRI segmentation [51].
this section cite: ['b19', 'b47', 'b48', 'b49', 'b50']

Section: Evaluating Models: Baselines & Proposed
Baseline Model: We use the well-known segmentation model SAM [19], which employs a Vision Transformer (ViT) encoder as its backbone for segmentation tasks. SAM offers ViT variants (ViT-Base (b), ViT-Large (l), and ViT-Huge (h)), each with 12, 24, or 32 transformer layers, respectively.  These weight configurations -b, l, and h-are pre-trained using Masked Autoencoders (MAE) [2]. Unlike Unet-style models [52,18,53,13], the SAM mask decoder is lightweight, containing only two convolutional layers. However, reconstructing high-resolution masks from the latent space requires additional upscaling layers, leading to increased memory usage.
Proposed Model: As defined in Eqn 4, our approach utilizes the tree structure from the patching stage in SHF, reducing the need for decoder training. We also experimented with the updated SAM2 [17] encoder ViT model, adapting it for 3D MRI data tasks, such as BTCV and KiTS19. Unlike SAM2 original video frame processing, we use 3D convolutional layers for voxel processing in the patch embedding stage. Unet-shaped models, such as UNETR [18], TransUnet [53], and SWIN Unet [13], follow a contraction-expansion pattern with transformer or convolutional layers as encoders to extract image details, connected to decoders via skip connections. These decoders upscale the representation vectors from the dense latent space to match the mask's size. For comparison with SHF, we test Unet [52] with pre-trained weights from the timm package, along with the SAM-pretrained UNETR [18] and TransUnet [53] models.
Performance Metrics: Computational performance is reported in seconds per image for end-to-end training. Segmentation accuracy is evaluated using the Dice score, which quantifies the overlap between predicted and ground truth segmentation masks. It is defined as:
Dice(X,Y) = 2 • |X∩Y |/(|X| + |Y |)(5)
where X and Y are the sets being compared, and |X∩Y | is the size of their intersection.
this section cite: ['b18', 'b1', 'b51', 'b17', 'b52', 'b12', 'b16', 'b17', 'b52', 'b12', 'b51', 'b17', 'b52']

Section: Evaluation

this section cite: []

Section: Speedup: SHF vs w/o SHF on the same models, datasets, and comparable performance
Training with SHF is faster due to its ability to significantly compress the sequence length , shown in Figure . 3, and eliminate the decoder component. As shown in Table 1, SHF, which combines preprocessing and post-processing steps on top of the pre-trained SAM baseline, achieves a geometric mean speedup ranging from 3.86× to 32.3× while maintaining comparable dice scores. This speedup is measured when both SHF and the baseline are trained for the same number of epochs. At the highest resolution of 64K 2 with training on 256/1024 GPUs, SHF delivers approximately a 32× speedup compared to naive SAM encode and a reduction of 75% usage of GPUs. Such a speedup  benefited from eliminating the heavy decoder, allowing more parallelization and reducing inference and back-propagation.
this section cite: []

Section: Segmentation Performance: SHF vs w/o SHF on the different models and datasets
Qualitative Results: Table 2 demonstrates the segmentation improvements across different models and PAIP resolutions. At comparable resolutions, SHF achieves a nearly 8× reduction in patch size while maintaining the same computational complexity. This results in an average 5.5% improvement in dice score over the original model. Additionally, these improvements come with training time speedups of up to 4.6×. At high resolution (64K 2 ), SHF lacks a decoder during training, offering substantial computational and GPU memory savings compared to SAM, which relies on an upsampling restoration mask. Consequently, SAM requires a shorter sequence length at high resolutions, as it performs more upsampling to generate masks. For instance, as shown in Table 1, moving from 16K 2 to 64K 2 resolution results in a slight 2% performance drop for SAM. Table 3 presents 3D MRI Table 3: Segmentation of BTCV [48] for multi-organ segmentation and KiTS19 [50] for Kidney Tumor Segmentation on a single GPU. Time indicates the end-to-end runtime to achieve the corresponding Dice Score.
Datset Model Patch Size Time Speedup (×) Dice Score (%) BTCV [48] U-Net [52] N/A 843.90 Seconds 9.04× 80.2 U-Mamba [52, 54] N/A 8,016.24 Seconds 0.95× 83.51 TransUNet [55] N/A 3115.25 Seconds 2.45× 83.8 UNETR [18] 4 3 8386.56 Seconds 0.91× 89.1 Swin UNETR [56] 4 3 5861.93 Seconds 1.30× 89.5 SAM2[17] 4 3 7637.28 Seconds 1.0× 82.77 SHF-SAM2 2 3 1067.88 Seconds 7.15× 89.71 KiTS [50] U-Net [52] N/A 243.7 Minutes 1.99× 83.23 U-Mamba [52, 54] N/A 969.0 Minutes 0.5× 86.22 CoTr [55] N/A 488.7 Minutes 0.99× 84.59 UNETR [18] 8 3 513.6 Minutes 0.94× 86.45 nnFormer [51] 8 3 876.5 Minutes 0.55× 75.85 Swin UNETR [51] 8 3 748.3 Minutes 0.65× 81.27 Swin UNETR-V2 [51] 8 3 766.4 Minutes 0.63× 84.14 SAM2[17] 8 3 483.1 Minutes 1.0× 81.35 SHF-SAM2 2 3 187.3 Minutes 2.58× 87.25 segmentation results for BTCV and KiTS datasets at a 512 3 resolution. Instead of processing each 2D slice independently and reconstructing the final 3D prediction as in previous works [57,53], we replace the 2D convolution patch embedding with 3D convolution for voxel processing. As shown in the tables, SHF outperforms the prestrained SAM2 Hiera image encoder, yielding a 5.9% higher dice score and requiring 4× less computational resources.
this section cite: ['b47', 'b49', 'b56', 'b52']

Section: Visual Results:
We compare the segmentation quality across different high resolutions ([16K 2 , 32K 2 , 64K 2 ]) for the baseline models SAM, and our proposed SHF. The segmentation results are summarized in Fig. 1. The first column shows the original input, with the label indicating the resolution. The red square highlights a small portion of the image, approximately 3.125% of the entire slide. The second column presents the ground truth, followed by the predictions from different models. At higher resolutions, all models can capture the general segmentation areas. However, due to the limitations of heavy convolution-based decoders and uniform grid patching, only large patch sizes are feasible, such as the 16K 2 patch size used by SAM and UNETR at a 64K 2 input resolution.
In contrast, at the same 64K 2 resolution, SHF can use smaller patch sizes, as small as 8 2 , significantly improving the quality of the detailed masks.
this section cite: []

Section: SHF vs HIPT hierarchical model: training from scratch on the same datasets
Table 4: Classification (Top-1 accuracy) of vanilla ViT, HIPT [8], and SHF-ViT on PAIP dataset (16, 384 2 res.) Model GPUs Patch Size Accuracy ViT [4] 128 4, 096 2 68.97 HIPT [8] 128 [16, 256 2 , 4, 096 2 ] 72.69 SHF-ViT-4096 8 4, 096 2 69.11 SHF-ViT-2 128 2 2 80.14
To demonstrate the versatility of SHF, we compare its classification performance on the PAIP dataset with HIPT [8], a SoTA highly advanced hierarchical multi-resolution model specifically designed for microscopic pathology classification. For this experiment, we restructured the PAIP dataset, originally intended for segmentation, into six organ-based categories. Each category consists of 40 samples, with 28 for training, 8 for testing, and 4 for validation. For HIPT, we resized all samples to three resolution scales ([256, 1024, 16384]) and set the patch sizes for each scale to [16,256,4096], adhering to the original settings. For SHF, we used only the 16K 2 resolution images for classification. Instead of relying on a decoder for segmentation, we added an output channel dedicated to class prediction. In Table 4, SHF achieves significant accuracy improvement (>8%) over HIPT, even when using a vanilla ViT model and the same computational budget. At high resolution (16K 2 ), HIPT is limited to a patch size of 4096 2 before running OOM, whereas SHF can use patch sizes as small as 2 2 in the highest-resolution regions. This substantial accuracy boost, despite SHF using a basic ViT, indicates that: a) SHF avoids random patch dropping or padding, and b) smaller patch sizes play a more crucial role in improving performance than model complexity.
this section cite: ['b7', 'b15']

Section: Conclusion and Future Work
This paper presents SHF, a lightweight and efficient method for enhancing ViT performance on high-resolution images. By increasing token information density and encoding hierarchical spatial structures through a hierarchical patching strategy and reverse depatching, SHF enables standard ViTs to handle long sequences effectively without requiring architectural changes. Experimental results on both 2D and 3D medical imaging tasks demonstrate significant improvements in computational efficiency and segmentation accuracy, establishing SHF as a practical solution for high-resolution medical image analysis. In future work, we aim to extend SHF to broader scientific domains, including materials discovery [58,59], brain neuron structure detection [60,61], and non-medical CT applications [62][63][64]. Additionally, we plan to enhance SHF's performance on multimodal problems [65] to support more complex and diverse applications.
this section cite: ['b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b2 Title: Attention is all you need Year: (2017)
Ref_id:b3 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b4 Title: Squeeze-and-excitation networks Year: (2018)
Ref_id:b5 Title: Non-local neural networks Year: (2018)
Ref_id:b6 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b7 Title: Scaling vision transformers to gigapixel images via hierarchical self-supervised learning Year: (2022)
Ref_id:b8 Title: Transformer for semantic segmentation Year: (2021)
Ref_id:b9 Title: Three-level hierarchical transformer networks for long-sequence and multiple clinical documents classification Year: (2021)
Ref_id:b10 Title: Crossvit: Cross-attention multiscale vision transformer for image classification Year: (2021)
Ref_id:b11 Title: Megabyte: Predicting million-byte sequences with multiscale transformers Year: (2023)
Ref_id:b12 Title: Swin-unet: Unet-like pure transformer for medical image segmentation Year: (2022)
Ref_id:b13 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b14 Title: Multiscale vision transformers Year: (2021)
Ref_id:b15 Title: Mvitv2: Improved multiscale vision transformers for classification and detection Year: (2022)
Ref_id:b16 Title: Segment anything in images and videos Year: (2024)
Ref_id:b17 Title: Unetr: Transformers for 3d medical image segmentation Year: (2022)
Ref_id:b18 Title: Segment anything Year: (2023)
Ref_id:b19 Title: Liver cancer segmentation challenge Year: (2019)
Ref_id:b20 Title: Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models Year: (2023)
Ref_id:b21 Title: Lightseq: Sequence level parallelism for distributed training of long context transformers Year: (2023)
Ref_id:b22 Title: Ring attention with blockwise transformers for near-infinite context Year: (2023)
Ref_id:b23 Title: Ultra-long sequence distributed transformer Year: (2023)
Ref_id:b24 Title: Flashattention: Fast and memory-efficient exact attention with io-awareness Year: (2022)
Ref_id:b25 Title: Flashattention-2: Faster attention with better parallelism and work partitioning Year: (2023)
Ref_id:b26 Title: Monarch: Expressive structured matrices for efficient and accurate training Year: (2022-07)
Ref_id:b27 Title: Specformer: Spectral graph neural networks meet transformers Year: (2023)
Ref_id:b28 Title: Rethinking graph transformers with spectral attention Year: (2021)
Ref_id:b29 Title: Rethinking attention with performers Year: (2020)
Ref_id:b30 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b31 Title: Generating long sequences with sparse transformers Year: (2019)
Ref_id:b32 Title: Reformer: The efficient transformer Year: (2020)
Ref_id:b33 Title: Efficient Content-Based Sparse Attention with Routing Transformers Year: (2021-02)
Ref_id:b34 Title: The long-document transformer Year: (2020)
Ref_id:b35 Title: Big bird: Transformers for longer sequences Year: (2020)
Ref_id:b36 Title: Lazyformer: Self attention with lazy update Year: (2021)
Ref_id:b37 Title: Self-attention does not need o(n 2 ) memory Year: (2022)
Ref_id:b38 Title: Scatterbrain: Unifying sparse and low-rank attention Year: (2021)
Ref_id:b39 Title: Sparsebert: Rethinking the importance analysis in self-attention Year: (2021)
Ref_id:b40 Title: Quadtree attention for vision transformers Year: (2022)
Ref_id:b41 Title: Octree transformer: Autoregressive 3d shape generation on hierarchically structured sequences Year: (2023)
Ref_id:b42 Title: Adaptive patching for high-resolution image segmentation with transformers Year: (2024)
Ref_id:b43 Title: A computational approach to edge detection Year: (1986)
Ref_id:b44 Title: The quadtree and related hierarchical data structures Year: (1984)
Ref_id:b45 Title: Quad trees: a data structure for retrieval on composite keys Year: (1974)
Ref_id:b46 Title: On locality-sensitive orderings and their applications Year: (2020)
Ref_id:b47 Title: Miccai multi-atlas labeling beyond the cranial vault-workshop and challenge Year: (2015)
Ref_id:b48 Title: Real-time high-resolution x-ray computed tomography Year: (2024)
Ref_id:b49 Title: The kits19 challenge data: 300 kidney tumor cases with clinical context, ct semantic segmentations Year: ()
Ref_id:b50 Title: nnu-net revisited: A call for rigorous validation in 3d medical image segmentation Year: (2024)
Ref_id:b51 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b52 Title: Transunet: Transformers make strong encoders for medical image segmentation Year: (2021)
Ref_id:b53 Title: U-mamba: Enhancing long-range dependency for biomedical image segmentation Year: (2024)
Ref_id:b54 Title: Transunet: Transformers make strong encoders for medical image segmentation Year: (2021)
Ref_id:b55 Title: Self-supervised pre-training of swin transformers for 3d medical image analysis Year: (2022)
Ref_id:b56 Title: A fixedpoint model for pancreas segmentation in abdominal ct scans Year: (2017)
Ref_id:b57 Title: Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties Year: (2018)
Ref_id:b58 Title: Materials discovery and design using machine learning Year: (2017)
Ref_id:b59 Title: Dense connectomic reconstruction in layer 4 of the somatosensory cortex Year: (2019)
Ref_id:b60 Title: High-precision automated reconstruction of neurons with flood-filling networks Year: (2018)
Ref_id:b61 Title: Paradigm shift in infrastructure inspection technology: Leveraging high-performance imaging and advanced ai analytics to inspect road infrastructure Year: (2025)
Ref_id:b62 Title: Scalable fbp decomposition for cone-beam ct reconstruction Year: (2021)
Ref_id:b63 Title: Ifdk: A scalable framework for instant high-resolution image reconstruction Year: (2019)
Ref_id:b64 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b65 Title: Etc: Encoding long and structured inputs in transformers Year: (2020)
Ref_id:b66 Title: Big bird: Transformers for longer sequences Year: (2020)
Ref_id:b67 Title: Reformer: The efficient transformer Year: (2020)
Ref_id:b68 Title: Generating long sequences with sparse transformers Year: (2019)
Ref_id:b69 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b70 Title: Spformer: Enhancing vision transformer with superpixel representation Year: (2024)
Ref_id:b71 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b72 Title: The Frontier supercomputer Year: ()
Ref_id:b73 Title: Three things everyone should know about vision transformers Year: (2022)
