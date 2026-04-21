Title: Orochi: Versatile Biomedical Image Processor
Abstract: Deep learning has emerged as a pivotal tool for accelerating research in the life sciences, with the low-level processing of biomedical images (e.g., registration, fusion, restoration, super-resolution) being one of its most critical applications. Platforms such as ImageJ (Fiji) and napari have enabled the development of customized plugins for various models. However, these plugins are typically based on models that are limited to specific tasks and datasets, making them less practical for biologists. To address this challenge, we introduce Orochi, the first application-oriented, efficient, and versatile image processor designed to overcome these limitations. Orochi is pre-trained on patches/volumes extracted from the raw data of over 100 publicly available studies using our Random Multi-scale Sampling strategy. We further propose Task-related Joint-embedding Pre-Training (TJP), which employs biomedical task-related degradation for self-supervision rather than relying on Masked Image Modelling (MIM), which performs poorly in downstream tasks such as registration. To ensure computational efficiency, we leverage Mamba's linear computational complexity and construct Multi-head Hierarchy Mamba. Additionally, we provide a three-tier fine-tuning framework (Full, Normal, and Light) and demonstrate that Orochi achieves comparable or superior performance to current state-of-the-art specialist models, even with lightweight parameter-efficient options. We hope that our study contributes to the development of an all-in-one workflow, thereby relieving biologists from the overwhelming task of selecting among numerous models. Our pre-trained weights and code will be released.

Section: Introduction
With the rapid advancement of deep learning, modern neural networks have demonstrated remarkable scalability and spawned a wide array of downstream applications in AI for Life Science [1,2,3]. Among these, biomedical image processing is a pivotal topic. Its significance arises from the inherent Figure 1: Trend of Versatile Biomedical Image Precessor. We listed the recent advancements in biomedical image processing, where matched row-to-column colour coding highlights the main task of each model. Stickers display the reported scores from the respective papers. Orochi extends the versatile bandwidth and exhibits exceptional performance across tasks and tuning modes. constraints in acquiring biomedical images compared to natural images, which often compromise source image quality. Specifically, the most common limitations stem from imaging device operational trade-offs. For instance, in optical microscopy, excessive laser intensity can damage target tissues, while insufficient laser power introduces low signal-to-noise ratios [4]. Similarly, in computed tomography (CT), thinner slice scans subject patients to prolonged high radiation exposure, posing health risks, whereas sparse slicing results in low-resolution data [5]. These challenges drive the demand for biomedical image restoration [6,4,7,8,9] and super-resolution [10,11,12,13,5] tasks. Another class of limitations originates from the intrinsic shortcomings of imaging modalities. For example, CT imaging is efficient and provides clear hierarchical information but suffers from poor soft-tissue contrast, in contrast, magnetic resonance imaging (MRI) excels in soft-tissue resolution but requires longer acquisition times and is susceptible to motion artifacts. Such modality-specific weaknesses necessitate biomedical image fusion tasks [14,15,16,17,18,19,20,21,22,23,24,25]. Furthermore, acquiring synchronous multi-modal data imposes demands on equipment and environments, making asynchronous data more prevalent. However, misalignment exists between tissues or cells due to temporal/specimen variability, motivating image registration [26,27,28,29,30,31,32,33] tasks to align asynchronous or even heterogeneous datasets of the same specimen.
With the emergence of long-range dependency models [34,35,36] and self-supervised pre-training methods [37,38,39,40], models designed for the aforementioned issues have advanced rapidly, giving rise to powerful specialist models (see Figure 1). However, we argue that in practical applications, these specialist models neglect three critical factors: (1) Task Perspective: Real-world biomedical imaging tasks often require multiple sequential steps (e.g., registration followed by fusion, as discussed earlier). (2) Degradation Perspective: Since the underlying causes of degradation share similarities, these degradations are interrelated-for example, both low signal-to-noise ratio and low resolution result in information loss. (3) Data Perspective: Due to their characteristics of being multi-channel, large-scale, and high-throughput, biomedical images are considerably larger than natural images, making the training and inference of multiple specialist models highly inefficient. From both efficiency and effectiveness standpoints, these issues collectively motivate the development of a universal foundational model. We aim for such a generalist model to optimize the aforementioned challenges by: (1) handling diverse low-level tasks within a unified framework, thereby avoiding the difficulties of selecting and integrating several specialist models; (2) capturing more generalized and robust features via cross-task learning during the pre-training phase; and (3) addresses real-world biomedical data processing costs to reduce redundant training and inference. Therefore, we introduce Orochi (named after the legendary multi-headed serpent). To fulfill the envisioned goals, our design emphasizes four aspects (see Figure 2): (1) Dataset Level: We extensively employ unlabeled raw data from over 100 publicly available studies (see Appendix A.2) and perform our Random Multi-scale Sampling, which considers the different scales of Regionof-Interest (ROI). (2) Pre-training Level: Inspired by Joint-embedding Prediction Architecture (JEPA) [40], where different degradations serve as context for each others. Our Task-related Joint-Figure 2: Overview of the Construction of Orochi. The upper panel illustrates the data conversion pipeline, taking into account patches/volumes at multiple scales. The lower panel presents the selfsupervised strategies utilized during pre-training. Additionally, we provide supplementary images (e.g., Visible A+B, Error Map) to facilitate the comparison between inputs and outputs.
embedding Pre-training (TJP) applies various forms of task-specific degradation, and the model learns from reconstructing them jointly. (3) Model Level: On one hand, we employ Mamba [41] as the building blocks to leverages its linear complexity [42,41,43,44]. On the other hand, the overall structure draws inspiration from the hierarchical design of the Swin-Transformer [36] by incorporating patch merging to enhance model efficiency further. (4) Post-training Level: We propose a three-tier fine-tuning framework to reduce the tuning cost. Ranging from full fine-tuning (Full), to fine-tuning only the replaced dense convolution head (Normal), and finally to the most lightweight variant using depth-wise separable convolution [45] (Light), thereby achieving Parameter-Efficient Fine-Tuning (PEFT) [46]. To this end, we hope that Orochi will distinguish itself as an exceptional tool among the extensive array of plugins available on platforms such as ImageJ (Fiji) [47] and napari [48], further advancing towards a user-friendly workflow with unified functionalities. In summary, our main contributions are as follows:
1. We systematically review the significance of low-level biomedical image processing and highlight that even in this era of powerful foundational models, the paradigm centred on specialist models still exhibits inherent deficiencies. Limiting both effectiveness and efficiency from the perspectives of task, degradation, and data. To the best of our knowledge, Orochi is the first versatile foundational model addressing these issues.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b3', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b4', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b39', 'b40', 'b41', 'b40', 'b42', 'b43', 'b35', 'b44', 'b45', 'b46', 'b47']

Section: 2.
We curated raw-level datasets from over 100 studies [49,50,51], covering a wide range of imaging modalities from 2-5D -with a total data size over 100 terabytes. During training, we introduce Random Multi-scale Sampling to achieve a unitedly raw data conversion into training patches/volumes. These converted data are used for both local and stream training, alleviating the challenges with the transmission and storage of extremely large datasets.
3. We propose Task-related Joint-embedding Pre-training (TJP), which directly learns the interrelations among various task-specific degradations rather than relying on common Masked Image Modelling (MIM). For the model architecture, we leverage the linear complexity of Mamba and design a multi-head hierarchical structure to minimize the costs of training and inference. Finally, for post-training, we introduce a three-tier fine-tuning framework and demonstrate that even the most lightweight depth-separable convolution tuning can achieve performance comparable to existing state-of-the-art specialist models.
this section cite: ['b48', 'b49', 'b50']

Section: Related Works
Self-supervised Learning Self-supervised Learning (SSL) extracts inherent data properties. Masked Image Modelling (MIM) predicts masked image regions from original pixel values using an encoder-decoder architecture, with loss in image space [37,38]. Contrastive Learning (CL) aligns representations of augmented views of the same image in an embedding space via specialized objectives [52,39]. Combining these, the Joint-Embedding Predictive Architecture (JEPA) [40] predicts full latent representations from context to learn robust image representations.
Restoration To address low image quality in fluorescence microscopy, Content-aware image restoration (CARE) [4] uses CNNs. Li et al. [8] improved axial resolution using a CARE-based model with physically acquired ground truth. Subsequent works integrated Swin-Transformers [36] for efficiency (SwinIR [9]) or Mamba blocks [42] for long-range dependency modeling (MambaIR [7]). UniFMIR [6] demonstrated that pre-trained foundation models generalize well for this task.
Super-resolution Super-resolution aims to overcome optical limits. DeepLP [53] employs pointscanning for reconstruction. Diffusion-based models, including volumetric conditioning modules [5] and latent diffusion in InverseSR [11], show promise for 3D brain MRI. Other approaches include local implicit image functions for flexible resolution enhancement [13], joint super-resolution and synthesis frameworks for isotropic volumes [12], and methods for multimodal image super-resolution [10].
Registration Image registration aligns images by optimizing a deformation field. VoxelMorph [54] provides a learning-based 3D framework. Dual-encoder U-Nets [26], Swin-Transformers for longdistance correspondences (TransMorph [32]), and Mamba blocks [42] for efficient long-range modeling (MambaMorph [33]). Fast 3D registration methods have been proposed by Siebert et al. [27,29], while Mok et al. [30,28] address large deformations with Laplacian Pyramid Networks.
this section cite: ['b36', 'b37', 'b51', 'b38', 'b39', 'b3', 'b7', 'b35', 'b8', 'b41', 'b6', 'b5', 'b52', 'b4', 'b10', 'b12', 'b11', 'b9', 'b53', 'b25', 'b31', 'b41', 'b32', 'b26', 'b28', 'b29', 'b27']

Section: Fusion Multi-modality image fusion integrates complementary information.
Techniques include bidirectional stepwise feature alignment for unaligned images (BSAFusion [25]), mutual enhancement for PAT/MRI fusion [24], and diffusion-based methods incorporating fusion priors (Diff-IF [55]) or denoising diffusion models [16]. Semantic-aware strategies with registration are found in SuperFusion [21] and MURF [22]. Other notable methods encompass one-stage progressive dense registration [23], U2Fusion [14], and Equivariant fusion [15]. Diverse strategies also include lightweight and semantic-guided approaches (ALMFNET [17], MSGFUSION [18]), dictionary-based and GAN-driven frameworks [19,56], and unsupervised methods [20].
this section cite: ['b24', 'b23', 'b54', 'b15', 'b20', 'b21', 'b22', 'b13', 'b14', 'b16', 'b17', 'b18', 'b55', 'b19']

Section: Methods
Due to page limitations, this section primarily emphasizes our comprehensive degradation designs used for self-supervision. The Appendix provided detailed architecture of the Multi-head Hierarchy Mamba model along with the three-tier fine-tuning framework B.
this section cite: []

Section: Preliminary: Self-supervised Degradation
Self-supervised image learning can be generally formulated as learning a reconstruction function f θ that recovers the original image x from its degraded D(x). Formally, this objective is defined as:
min θ E x ℓ x, f θ D(x) , (1
)
where x is the sampled data, p data , D(•) denotes a degradation function applied to x, f θ is the parameterized model, and ℓ is a loss function (e.g., the L2 loss or perceptual loss).
this section cite: []

Section: Masked Image
For masked image degradation, the degradation function is defined as: D mask (x) = x ⊙ M, where M ∈ {0, 1} H×W is a binary mask with height H and width W that selectively occludes regions of x. This degradation helps the model learn to infer missing information.
Deformed Image For deformed image degradation, the degradation function takes the form: D def (x) = T(x), where T(•) represents a spatial transformation (such as rotation, scaling, or warping). This degradation introduces geometric distortions that mimic real-world variations.
Nosiy Image For noisy image degradation, the degradation function is defined as: D noise (x) = x + η where η denotes additive noise (typically Gaussian noise), simulating sensor imperfections or environmental interference.
this section cite: []

Section: Low-resolution Image
For low-resolution image degradation, the degradation function is given by: D LR (x) =↓ s (x), where ↓ s is a down-sampling operator with scale factor s, reducing the resolution of x to simulate the effects of low-resolution imaging.
this section cite: []

Section: Orochi: Random Multi-scale Sampling
Random Multi-scale Sampling aims to extract patches/volumes with diverse scales from raw images. Given a raw image I, the procedure consists of two main steps: (1) Multi-scale Resizing: We first generate scaled versions of the raw image I to capture features at different resolutions. In particular, we resize I to scales 1/2 and 1/4 of its original size. Formally, let:
I s =↓ s (I), s ∈ {1, 1 2 , 1 4 },(2)
where ↓ s (•) denotes down-sampling with factor s. (2) Random Window Sampling: For each scaled image I s , we define a fixed-size window K (compatible with the pre-training requirements in either 2D or 3D) and perform random sampling to extract sub-patches. Let the window K have dimensions W × H (or W × H × D for 3D data). A randomly sampled 2D patch x s at scale s is given by:
x s = I s i : i + W -1, j : j + H -1 ,(3)
where (i, j) is a randomly chosen starting coordinate in I s .
Collectively, the set of patches extracted across scales is represented as:
x = {x s,n | s ∈ {1, 1 2 , 1 4 }, n = 1, . . . , N s },(4)
where N s denotes the number of patches sampled from the image at scale s. These multi-scale patches are then passed to subsequent degradation processes (e.g., masking, deformation, noise addition, and low-resolution conversion). By performing random sampling across multiple scales, our method extended the data diversity and enabled more robust feature learning across various datasets.
this section cite: []

Section: Orochi: Task-related Joint-embedding Pre-training Dual-Masking Reconstructive Fusion
To better address the biomedical image fusion task, where the combination of existing contexts is crucial, we modified the conventional Masked Image Modelling approaches [37,38], which typically employ a single masking strategy. Specifically, we applied two distinct masking operations to the training data x, thereby generating two independent masks:
x A = x ⊙ M A , x B = x ⊙ M B ,(5)
where M A , M B ∈ {0, 1} H×W are binary masks with only partial overlap and ensure invisible information retention even after fusion. The masking probabilities are generated by:
M k [i, j] = 1[ξ k i,j < τ ], k ∈ A, B,(6)
where ξ k i,j ∼ U(0, 1) represents a random value extracted from a uniform distribution for grid coordinates i, j, and τ is the masking threshold. The key innovation is that our model is exposed to process both masked inputs (x A , x B ) simultaneously to recover the original image: x = f θ (x A , x B ),. This guides the model to develop robust feature extraction capabilities that can identify complementary information across different masked views, and then fuse these partial observations coherently to reconstruct missing regions in both inputs.
this section cite: ['b36', 'b37']

Section: Spatially-varying Gaussian down-sample
For down-sampling, we adapt similar principles from DeepLP [53], which tested noisy down-sampling beyond uniform down-sampling in self-supervised microscopy restoration. We enhance this noisy down-sampling with spatially varying characteristics:
D LR (x) = Gσvar(↑ 1 s (↓ s (x + η))),(7)
where ↓ s represents down-sampling with a random scale factor s, ↑ 1 s denotes upsampling back to the original resolution, η ∼ N(0, σ 2 down ) is normal distributed noise added during the downsampling process with σ down ∼ U(0.01, 0.1), U represent uniform distribution, and Gσvar denotes spatially-varying Gaussian filtering. It can be defined as:
Gσvar(x)[i, j] = u,v g σ(i,j) (u, v) • x[i -u, j -v],(8)
where g σ represents a Gaussian kernel (2/3D) with standard deviation σ(i, j) ∼ U(σ min , σ max ) that varies across grid coordinates i, j. This mimics the heterogeneous blurring found in optical systems.
this section cite: ['b52']

Section: Multi-scale Smoothed Perlin Noise Deformation
For the self-supervised registration task, constructing a realistic deformation field is important. We conducted multi-scale Perlin noise fields that simulate the hierarchy variations in natural anatomical structures. Given an image x, we generate a deformation field Φ and its corresponding deformed image D def (x) as follows:
D def (x) = T(x, Φ), Φ = G σ (Per(f , p)),(9)
T(•, •) is a spatial transformation operator, G σ (•) denotes spatially-varying Gaussian smoothing with parameter σ, and Per(f , p) represents multi-octave Perlin noise with frequency f and persistence p.
The multi-octave Perlin noise is specifically defined as:
Per(f , p) = N n=1 p n-1 • S(f n-1 • (i, j)),(10)
where S(•) is the simplex noise function, N is the number of octaves and coords represents the grid coordinates. This multi-scale approach generates deformation fields with varying levels of detail.
To enhance the anatomical plausibility of the deformations, we apply normalization and bound it using a tanh function: Φ final = α • tanh(Φ), where α controls the maximum displacement magnitude.
this section cite: []

Section: Multi-stage Noise Simulation
To simulate realistic noise, we adopted a multi-stage process:
D noise (x) = Bi p (Poi(max(0, x + η))),(11)
where η ∼ N(0, σ 2 noise ) with σ noise ∼ U(0.075, 0.15) represents Gaussian noise, Poi(λ) denotes Poisson noise with intensity parameter λ (modeling photon-counting statistics), and Bi p represents binary (salt-and-pepper) noise that affects a proportion of pixels with probability p.
These sophisticated degradation designs enable our framework to simulate a wide spectrum of realworld imaging artifacts, encouraging the model to handle diverse image quality issues encountered.
this section cite: []

Section: Experiments
We conducted comprehensive comparisons strictly following the setups in published specialist models (UniFMIR [6], VCM [5], Transmorph [32], and BSAFusion [25], see Appendix A.4 for details). Resulting in more than 30 state-of-the-art baselines across multiple benchmarks for various biomedical image-processing tasks to demonstrate the effectiveness and versatility of Orochi. We color-coded the performance in Table 1,
this section cite: ['b5', 'b4', 'b31', 'b24']

Section: Generalization Capability on In-Domain Data
Given that our model, Orochi, is extensively pretrained, we expect it to exhibit strong generalization capabilities on in-domain data. Accordingly, in Figure 3 we demonstrate Orochi's zero-shot performance on various stained microscopy images [51] (results on clinical images [50] are detailed in the Appendix C.1). Panels (A)-(D) illustrate Orochi's robust processing capabilities. In Panel (E), we further examine whether these outcomes align with our algorithmic expectations. For example, our Dual-Masking Reconstructive Fusion anticipates that the model learns an effective fusion strategy and leverages the existing information from both  (A)-(D) illustrate Orochi's robust performance across various low-level processing tasks when applied to unseen testing images after pre-training. Supplementary images include the dual-masking images and naive merge results for the fusion task. Error maps for the registration task. (E) provides in-depth case studies: for the fusion task, the centromere count is emphasized in both the reconstructed image and the original image (highlighted with circles); for the registration task, subtle deformations of the cell membrane are accentuated; and for the restoration and super-resolution tasks, the fine details of bright-field images and the internal structures of DNA-stained cell nuclei are emphasized  Table 3: Inter-patient Brain Registration Task.
During training, the model goal is to input paired MRI data from distinct patients and output prediction of the registration flow. This flow is applied to the corresponding segmentation data to calculate the dice loss. Thereby, regional deformation can be learned with supervision. Image Super-resolution Task We next evaluated the image super-resolution capabilities of Orochi (see Table 2). Early super-resolution models typically rely on CNN-based architectures such as UniRes [10] and SynthSR [12], which are efficient yet often lack sufficient expressiveness and generalization ability. LIIF [13] leverages the power of Implicit Neural Representations (INR) to perform implicit interpolation; however, the high training cost associated with INR limits its adaptability to real-world scenarios. More recent approaches, including InverseSR [11] and VCM [5], based on powerful pre-trained Brain-Latent Diffusion Models (LDM) [59] to overcome these shortcomings. In this setting, Orochi significantly outperforms all the aforementioned architectures. At an 8mm slice thickness, Orochi achieves a PSNR that is 4.01 points higher than InverseSR and 2.76 points higher than VCM. These gains demonstrate that among pre-trained models, Orochi's pre-training is markedly superior to that of Brain-LDM, both in terms of the pre-training data and purpose.
this section cite: ['b50', 'b49', 'b9', 'b11', 'b12', 'b10', 'b4', 'b58']

Section: Method
Image Registration Task We further evaluated the registration task using the dataset from Learn2Reg [60] (see Table 3). In this task, brain MRI images from different patients (i.e., interpatients) are registered (see Appendix C.2 for patient-to-atlas brain registration test), and the model's ability to handle subtle deformations is assessed by measuring the similarity of the segmented brain regions after registration (e.g. Dice). Biomedical image registration has evolved from CNNbased [29,30,27] to Transformer-based architectures [32,54], with even linear-complexity models such as Mamba [33] emerging in recent work. In comparison to these methods, our approach achieves Dice scores that are 2.42 points higher than ConvexAdam, 2.0 points higher than Transmorph, and 1.81 points higher than Mambamorph.
Image Fusion Task Finally, as illustrated in Table 4, we evaluated Orochi's performance on the image fusion task. Recent trends in this domain have integrated image registration as an auxiliary task to facilitate fusion, as demonstrated by methods such as BSAFusion [25], UMF-CMGR [20], MURF [22], and SuperFusion [21]. Although these models typically exhibit limited registration capabilities (see Appendix C.2), this aligns with our pursuit of developing a versatile, comprehensive model. Compared with the recent advanced model BSAFusion, Orochi outperforms on all evaluated metrics, achieving improvements of +0.02 in Q abf , -1803.53 in Q cv , and +0.07 in SSIM. Combined with our state-of-the-art performance on the registration task, these results establish Orochi as the first model in this domain to achieve such performance.  Visualizations In Figure 4, we provide qualitative results of Orochi. Specifically, Orochi demonstrates a superior capability in handling subtle degradations. (see Appendix C.1 C.2 for more)
this section cite: ['b59', 'b28', 'b29', 'b26', 'b31', 'b53', 'b32', 'b24', 'b19', 'b21', 'b20']

Section: Ablation Study -Comparison to Other Pre-train Strategies
In Table 5, we demonstrate the limitations of relying solely on Masked-image-Modelling (MIM), particularly in registration tasks. Additionally, we observe that the dual-masking approach employed in I-JEPA [40] underperforms compared to Orochi. We hypothesize that this is because chunk masking is more advantageous for high-level tasks rather than the low-level focus of our study.
Ablation Study -Larger ̸ = Better, Fine-Tuning Efficiency V.S Performance As shown in Figure 5, the number of trainable parameters is not the decisive factor for downstream tasks-particularly in data-limited scenarios such as biomedical imaging. In many cases, opting for Parameter-Efficient Fine-Tuning (using only 1-2% of the total parameter count) prevents overfitting and achieves both efficient and effective results.
this section cite: ['b39']

Section: Conclusion
We introduce Orochi, the first versatile biomedical image processor designed for low-level tasks.
To enhance effectiveness, we propose Random Multi-scale Sampling, which is a scalable way to leverage raw data from a wide range of studies. The extracted data is then processed through our Task-related Joint-embedding Pre-training (TJP), where a unified and robust embedding is learned from various task-related degradations. For efficiency, we developed Multi-head Hierarchy Mamba and provide a three-tier fine-tuning framework (Full, Normal, and Light). These design choices ensure high efficiency during pre-training, post-tuning, and test inference. Our experiments demonstrate that Orochi exhibits in-domain generalization capability across multiple tasks and achieves stateof-the-art performance compared to specialist models with efficient fine-tuning (less than 5% of total parameters). This suggests that constructing a generalist image processor may lie more in the diversity of the dataset and the pre-training strategy than in increasing the model size naively.
NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The contributions are well justified with comprehensive theoretical and experimental results.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss the limitations in the Appendix.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We provide the full set of assumptions and a complete (and correct) proof Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We provide a condensed implementation in the experiment section and a detailed description in the Appendix, with code submitted in the supplemental materials. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described. A.2 Datasets Pre-train
• A combined multi-modal biomedical image dataset aggregated from over 100 public studies, encompassing various imaging modalities and degradation types [51,50,49]. In Figure 6, we provide a preview of the metadata of the studies we covered (Excel Form would be included in the Zip file). Since our RMS method is highly scalable, we plan to further update this list in the future and explore the borderline.
this section cite: ['b50', 'b49', 'b48']

Section: Registration
• The OASIS brain MRI dataset from the Learn2Reg 2021 challenge, used to evaluate the overlap of segmented regions and the smoothness of the deformation fields [60,58].
this section cite: ['b59', 'b57']

Section: Fuison
• A CT-MRI paired fusion dataset (VIFB), which assesses the integration of complementary information across modalities [57].
this section cite: ['b56']

Section: Super-Resolution
• The Harvard Whole Brain Atlas (HBA), providing high-quality MRI images for evaluating low-resolution image reconstruction [57].
this section cite: ['b56']

Section: Restoration
• The CARE microscopy image dataset, used to evaluate the enhancement of low signal-tonoise ratio fluorescence microscopy images [4].
this section cite: ['b3']

Section: A.3 Metrics Registration
• Dice similarity coefficient: Computed as
Dice = 2|A ∩ B| |A| + |B|
which measures the overlap between the segmented regions.
• 95 th percentile Hausdorff Distance (HD95): Defined as the 95 th percentile of the distances between boundary points of the segmented regions.
• Standard deviation of the log-Jacobian determinant (SDlogJ): Calculated as the standard deviation of log(det(J)), where J is the Jacobian matrix of the deformation field. This metric reflects the smoothness of the deformation field [60].
this section cite: ['b59']

Section: Fuison
• Q AB/F (Q abf ): Measures the quality of the fusion by evaluating the consistency between the fused image and the input modalities.
• Q CV (Q cv ): Assesses the contrast consistency across the fused image.
• Structural Similarity Index (SSIM): Computed based on comparisons of luminance, contrast, and structure between 2 source images [25].
this section cite: ['b24']

Section: Super-Resolution
• Peak Signal-to-Noise Ratio (PSNR): Calculated as
P SN R = 10 log 10 MAX 2 I M SE
where MAX I is the maximum possible pixel value and MSE is the mean squared error between the reconstructed and reference images.
• SSIM: Evaluates perceptual similarity between the super-resolved and reference images [9].
this section cite: ['b8']

Section: Restoration
• PSNR: As above, it measures the pixel-level fidelity between the restored image and the high-quality reference.
• SSIM: Measures the structural similarity between the restored and reference images [4].
this section cite: ['b3']

Section: A.4 Code-base
Pre-train
this section cite: []

Section: • Adapted from public GitHub implementations of the Swin-Transformer and Transmorph.
Swin-Transformer: https://github.com/microsoft/Swin-Transformer [36]; Transmorph: https://github.com/junyuchen245/TransMorph_Transformer_  for_Medical_Image_Registration [32].
this section cite: ['b35', 'b31']

Section: Registration
• Implemented based on the Transmorph GitHub code. Link: https://github.com/  junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration [32].
this section cite: ['b31']

Section: Fuison
• Built with reference to the BSAFusion GitHub code. Link: https://github.com/  slrl123/BSAFusion [25].
this section cite: ['b24']

Section: Super-Resolution
• Implemented based on GitHub codes of InverseSR and VCM. InverseSR: https:  //github.com/BioMedAI-UCSC/InverseSR [11]; VCM: https://github.com/  Ahn-Ssu/VCM [5].
this section cite: ['b10', 'b4']

Section: B Experiment Configurations
Multi-head Hierarchy Mamba & Three-Tier Fine-Tuning Framework Figure 7, presents a comprehensive diagram of Orochi's backbone architecture. Post-tuning, the interchangeable decoder can be replaced as required. We evaluated Orochi's performance using our Three-Tier Fine-Tuning Framework, which includes full fine-tuning (Full, 100% parameters), regular convolution head with the encoder frozen (Normal, 10-30% parameters), and depth-wise separable convolution head [45] with the encoder frozen (Light, less than 5% parameters). The optimal results were achieved across all three tiers, underscoring the significance of selecting an appropriate tuning method based on specific requirements.
Pre-train We pre-trained Orochi-B (3D version) with the configuration listed in Table 6. The 2D version has a similar configuration, with slight differences on some setups (e.g. batch size). We have 2 two sets of pre-training devices. The A800 80Gx8 device is used for local pre-train and the H100 40Gx8 device is for streaming pre-train.
this section cite: ['b44']

Section: Fine-tuning
We followed the same setups as our code base for each task (see Section ??), including the tuning resolution, epoch number, optimizer configurations and loss designs. The device we use for fine-tuning is NVIDIA 4090 24Gx4  Patient to Atlas Brain Image Registration The regional deformation is learned unsupervised in Table 7. Only the raw image of the atlas and the patient's brain would be used for loss calculation while training. Then we evaluate the dice score between the segmentation maps of these two brains. Since Orochi is pre-trained in this unsupervised fashion, it shows excellent adaptation to this task, similar to the case with supervision.
this section cite: []

Section: SPECT-MRI & PET-MRI Image Fusion
In Figure 10, we performed comparative evaluations using state-of-the-art fusion techniques on two additional Harvard Whole Brain datasets obtained from https://www.med.harvard.edu/aanlib/. These datasets specifically focus on the fusion of SPECT and PET imaging with MRI. The results demonstrate that Orochi outperforms recent advancements such as BSAFusion and maintains superior efficiency.
Table 7: Patient to Atlas Brain Registration Task. During the training phase, the model aims to input paired MRI data from both the standard brain atlas and patient scans, to output a predicted registration flow. This flow is subsequently applied to the atlas data to compute the similarity between the registered atlas and the patient's scan. During the testing phase, the predicted flow is applied to the atlas brain segmentation map, and the Dice coefficient is evaluated against the patient's brain segmentation map.
this section cite: []

Section: Method Dice
↑ % of |J Φ | ≤ 0 Dataset: IXI [32]
Affine 0.386 ± 0.195 -SyN [61] 0.645 ± 0.152 ≤ 0.0001 NiftyReg [62] 0.645 ± 0.167 0.020 ± 0.046 LDDMM [63] 0.680 ± 0.135 ≤ 0.0001 deedsBCV [64] 0.733 ± 0.126 0.147 ± 0.050 VoxelMorph-1 [54] 0.729 ± 0.129 1.590 ± 0.339 VoxelMorph-2 [54] 0.732 ± 0.123 1.522 ± 0.336 VoxelMorph-diff [54] 0.580 ± 0.165 ≤ 0.0001 CycleMorph [65] 0.737 ± 0.123 1.719 ± 0.382 MIDIR [66] 0.742 ± 0.128 ≤ 0.0001 ViT-V-Net [67] 0.734 ± 0.124 1.609 ± 0.319 PVT [68] 0.727 ± 0.128 1.858 ± 0.314 CoTr [69] 0.735 ± 0.135 1.292 ± 0.342 nnFormer [70] 0.747 ± 0.135 1.595 ± 0.358 TransMorph-Bayes [32] 0.753 ± 0.123 1.560 ± 0.333 TransMorph-diff [32] 0.594 ± 0.163 ≤ 0.0001 TransMorph-bspl [32] 0.761 ± 0.122 ≤ 0.0001 TransMorph [32] 0.754 ± 0.124 1.579 ± 0.328 Orochi (Full) 0.770 ± 0.120 1.592 ± 0.334 Orochi (Normal) 0.765 ± 0.121 1.571 ± 0.323 Orochi (Light) 0.752 ± 0.126 1.499 ± 0.301
this section cite: ['b60', 'b61', 'b62', 'b63', 'b53', 'b53', 'b53', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b31', 'b31', 'b31', 'b31']

Section: C.3 Super-Resolution & Restoration
Stress test on joint multi-modal data image repairing In this additional validation, we aim to evaluate Orochi's performance under stress using an extended benchmark. The BioSR [71] benchmark comprises four distinct categories of microscopy image pairs (x2 low/high imaging quality), captured by a multimodal structured illumination microscopy (SIM) system, encompassing Clathrin-Coated Pits (CCPs), Endoplasmic Reticula (ERs), Microtubules (MTs), and F-actin Filaments. Specifically, Orochi was trained on all four datasets concurrently, whereas the baseline [72,73,74,6] models were trained separately on each dataset. This deliberate approach highlights Orochi's capability in resource-constrained environments, where conducting hyperparameter searches for each subset is not feasible. As illustrated in Figure 11, despite the training constraints imposed on Orochi, an absolute improvement is still observed, further demonstrating its capability and efficiency.
this section cite: ['b70', 'b71', 'b72', 'b73', 'b5']

Section: D Limitations
Two limitations in our paper remain unaddressed at present. First, due to constraints on computational resources and group size, we were unable to further investigate the scaling law of our method during pre-training. This limitation also indicates that our focus was restricted to low-level tasks as presented in the paper. However, we firmly believe that a unified model for life sciences, capable of excelling in both high-level understanding tasks and low-level generation tasks, will emerge in the future. This is also an emerging trend that has already demonstrated progress in general applications.
this section cite: []

Section: A Experiment Setups
A.1 Baselines Registration • Lv et al. [26]: Uses a dual-encoder U-Net for coarse-to-fine registration.
• Siebert et al. [27]: Proposes a fast 3D registration approach.
• Mok et al. [28]: Employs conditional deformable convolutions.
• PIMed [31]: From the Learn2Reg challenge.
• LapIRN [30]: Uses a Laplacian pyramid network for large deformations.
• ConvexAdam [29]: Adopts a dual-optimization strategy.
• TransMorph [32]: Based on a Transformer architecture for capturing long-range correspondences.
• MambaMorph [33]: Utilizes mamba blocks for efficient long-range dependency modeling.
this section cite: ['b25', 'b26', 'b27', 'b30', 'b29', 'b28', 'b31', 'b32']

Section: Fuison
• U2Fusion [14]: Provides a unified unsupervised fusion approach.
• EMMA [15]: Employs equivariant learning for fusion.
• ALMFNet [17]: Searches for a lightweight generalized fusion network.
• MsgFusion [18]: Uses a semantic-guided two-branch network.
• MDHU [19]: Uses multi-dictionary learning with truncated Huber filtering.
• UMF-CMGR [20]: Adopts cross-modality generation and registration.
• SuperFusion [21]: Combines registration and fusion with semantic awareness.
• MURF [22]: Reinforces multi-modal registration and fusion mutually.
• IMF [23]: Improves fusion with a progressive dense registration strategy.
• PAMRFuse [24]: Focuses on feature alignment.
• BSAFusion [25]: Adopts bidirectional stepwise feature alignment.
• DDFM [16]: Utilizes a denoising diffusion model for fusion.
this section cite: ['b13', 'b14', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b15']

Section: Super-Resolution
• Cubic: Bicubic interpolation as a traditional baseline.
• UniRes [10]: Designed for super-resolving multimodal clinical MRI.
• SynthSR [12]: Performs joint super-resolution and synthesis.
• LIIF [13]: Learns continuous image representations for implicit interpolation.
• InverseSR [11]: Uses a latent diffusion model for 3D brain MRI super-resolution.
• VCM [5]: Applies a volumetric conditioning module.
this section cite: ['b9', 'b11', 'b12', 'b10', 'b4']

Section: Restoration
• Li et al. [8]: Improves axial resolution.
• CARE [4]: Uses a content-aware network for fluorescence microscopy image restoration.
• SwinIR [9]: Employs a Swin-Transformer for efficient image restoration.
• MambaIR [7]: Utilizes mamba blocks for modeling long-range dependencies.
• UniFMIR [6]: Fine-tunes a pre-trained foundation model for generalizable fluorescence microscopy-based restoration (with pruned FP16/FP32 variants).
this section cite: ['b7', 'b3', 'b8', 'b6', 'b5']

Section: 
Answer: [Yes] Justification: A Readme.md file is attached along with the code submitted in supplemental material.
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
Answer: [Yes] Justification: All the implementation details are included in the Appendix section.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: Our experiments are conducted with a set random seed 42.
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

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes]
Justification: We provide sufficient information on the computer resources.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We make sure to preserve anonymity.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
this section cite: []

Section: Answer: [NA]
Justification: There is no societal impact of the work performed.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All the assets are properly cited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper currently does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Restoration • Based on the UniFMIR GitHub implementation. Link: https://github.com/cxm12/  UNiFMIR [6].  In Figure 8, we present additional results demonstrating Orochi's zero-shot performance on both microscopy and medical images. Notably, Orochi yields satisfactory outcomes even when faced with extremely severe degradation, as exemplified in panel (B), row 2, and panel (C), rows 2 and 3.
this section cite: ['b5']

Section: C Extra Results

this section cite: []

Section: C.1 Zero-shot Processing on Biomedical Images

this section cite: []

Section: C.2 Registration & Fuison
Fusion Model on Registration Task In Figure 9, we demonstrate that, despite the recent trend of pre-registration before fusion, these methods remain predominantly fusion-oriented and are not wellsuited for addressing real-world registration tasks within the medical image registration community. However, Orochi represents a significant advancement in versatility, as it is designed not only for this specific scenario but also to achieve superior performance across all registration tasks.
this section cite: []

Section: References
Ref_id:b0 Title: Multimodal large language models for bioimage analysis Year: (2024)
Ref_id:b1 Title: Foundation models for generalist medical artificial intelligence Year: (2023)
Ref_id:b2 Title: Segment anything in medical images Year: (2024)
Ref_id:b3 Title: Content-aware image restoration: pushing the limits of fluorescence microscopy Year: (2018)
Ref_id:b4 Title: Volumetric conditioning module to control pretrained diffusion models for 3d medical images Year: (2024)
Ref_id:b5 Title: Pretraining a foundation model for generalizable fluorescence microscopy-based image restoration Year: (2024)
Ref_id:b6 Title: Mambair: A simple baseline for image restoration with state-space model Year: (2024)
Ref_id:b7 Title: Three-dimensional structured illumination microscopy with enhanced axial resolution Year: (2023)
Ref_id:b8 Title: Swinir: Image restoration using swin transformer Year: (2021)
Ref_id:b9 Title: A tool for super-resolving multimodal clinical mri Year: (2019)
Ref_id:b10 Title: Inversesr: 3d brain mri super-resolution using a latent diffusion model Year: (2023)
Ref_id:b11 Title: Joint super-resolution and synthesis of 1 mm isotropic mp-rage volumes from clinical mri exams with scans of different orientation, resolution and contrast Year: (2021)
Ref_id:b12 Title: Learning continuous image representation with local implicit image function Year: (2021)
Ref_id:b13 Title: U2fusion: A unified unsupervised image fusion network Year: (2020)
Ref_id:b14 Title: Equivariant multi-modality image fusion Year: (2024)
Ref_id:b15 Title: Ddfm: denoising diffusion model for multi-modality image fusion Year: (2023)
Ref_id:b16 Title: Learning to search a lightweight generalized network for medical image fusion Year: (2023)
Ref_id:b17 Title: Msgfusion: Medical semantic guided two-branch network for multimodal brain image fusion Year: (2023)
Ref_id:b18 Title: Multi-modal medical image fusion via multi-dictionary and truncated huber filtering Year: (2024)
Ref_id:b19 Title: Xin Fan, and Risheng Liu. Unsupervised misaligned infrared and visible image fusion via cross-modality image generation and registration Year: (2022)
Ref_id:b20 Title: Superfusion: A versatile image registration and fusion network with semantic awareness Year: (2022)
Ref_id:b21 Title: Murf: Mutually reinforcing multi-modal image registration and fusion Year: (2023)
Ref_id:b22 Title: Improving misaligned multi-modality image fusion with one-stage progressive dense registration Year: (2024)
Ref_id:b23 Title: Performance of medical image fusion in high-level analysis tasks: A mutual enhancement framework for unaligned pat and mri image fusion Year: (2024)
Ref_id:b24 Title: Bsafusion: A bidirectional stepwise feature alignment network for unaligned medical image fusion Year: (2024)
Ref_id:b25 Title: Joint progressive and coarse-to-fine registration of brain mri via deformation field integration and non-rigid feature fusion Year: (2022)
Ref_id:b26 Title: Fast 3d registration with accurate optimisation and little learning for learn Year: (2021)
Ref_id:b27 Title: Conditional deformable image registration with convolutional neural network Year: (2021-10-01)
Ref_id:b28 Title: Convexadam: Selfconfiguring dual-optimisation-based 3d multitask medical image registration Year: (2024)
Ref_id:b29 Title: Large deformation diffeomorphic image registration with laplacian pyramid networks Year: (2020)
Ref_id:b30 Title: Biomedical Image Registration, Domain Generalisation and Out-of-Distribution Analysis: MICCAI 2021 Challenges: MIDOG 2021, MOOD 2021, and Learn2Reg 2021, Held in Conjunction with MICCAI 2021 Year: (2021-10-01)
Ref_id:b31 Title: Transmorph: Transformer for unsupervised medical image registration Year: (2022)
Ref_id:b32 Title: Mambamorph: a mamba-based backbone with contrastive feature learning for deformable mr-ct registration Year: (2024)
Ref_id:b33 Title: Attention is all you need Year: (2017)
Ref_id:b34 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b35 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b36 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b37 Title: Simmim: A simple framework for masked image modeling Year: (2022)
Ref_id:b38 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b39 Title: Self-supervised learning from images with a joint-embedding predictive architecture Year: (2023)
Ref_id:b40 Title: Transformers are ssms: Generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b41 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b42 Title: Reinventing rnns for the transformer era Year: (2023)
Ref_id:b43 Title: Efficient attention: Attention with linear complexities Year: (2021)
Ref_id:b44 Title: Xception: Deep learning with depthwise separable convolutions Year: (2017)
Ref_id:b45 Title: Discovering long-term effects on parameter efficient fine-tuning Year: (2024)
Ref_id:b46 Title: Nih image to imagej: 25 years of image analysis Year: (2012)
Ref_id:b47 Title: napari: a multi-dimensional image viewer for python Year: (2019)
Ref_id:b48 Title: Image data resource: a bioimage data integration and publication platform Year: (2017)
Ref_id:b49 Title: Imaging intact human organs with local resolution of cellular structures using hierarchical phase-contrast tomography Year: (2021)
Ref_id:b50 Title: Integrated intracellular organization and its variations in human ips cells Year: (2023)
Ref_id:b51 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b52 Title: Deep learning-based point-scanning super-resolution imaging Year: (2019)
Ref_id:b53 Title: Voxelmorph: A learning framework for deformable medical image registration Year: (2019-08)
Ref_id:b54 Title: Diff-if: Multi-modality image fusion via diffusion model with fusion knowledge prior Year: (2024)
Ref_id:b55 Title: Generative adversarial network for trimodal medical image fusion using primitive relationship reasoning Year: (2024)
Ref_id:b56 Title: Harvard whole brain atlas Year: (2003)
Ref_id:b57 Title: Oasis is automated statistical inference for segmentation, with applications to multiple sclerosis lesion segmentation in mri Year: (2013)
Ref_id:b58 Title: Brain imaging generation with latent diffusion models Year: (2022)
Ref_id:b59 Title: Learn2reg: comprehensive multi-task medical image registration challenge, dataset and evaluation in the era of deep learning Year: (2022)
Ref_id:b60 Title: Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain Year: (2008)
Ref_id:b61 Title: Fast free-form deformation using graphics processing units Year: (2010)
Ref_id:b62 Title: Computing large deformation metric mappings via geodesic flows of diffeomorphisms Year: (2005)
Ref_id:b63 Title: Multi-modal multi-atlas segmentation using discrete optimisation and self-similarities Year: (1390)
Ref_id:b64 Title: Cyclemorph: cycle consistent unsupervised deformable image registration Year: (2021)
Ref_id:b65 Title: Learning diffeomorphic and modality-invariant registration using b-splines Year: (2021)
Ref_id:b66 Title: Vit-v-net: Vision transformer for unsupervised volumetric medical image registration Year: (2021)
Ref_id:b67 Title: Pyramid vision transformer: A versatile backbone for dense prediction without convolutions Year: (2021)
Ref_id:b68 Title: Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation Year: (2021-10-01)
Ref_id:b69 Title: nnformer: volumetric medical image segmentation via a 3d transformer Year: (2023)
Ref_id:b70 Title: Evaluation and development of deep neural networks for image super-resolution in optical microscopy Year: (2021)
Ref_id:b71 Title: Efficient non-local contrastive attention for image super-resolution Year: (2022)
Ref_id:b72 Title: Cross-modality supervised image restoration enables nanoscale tracking of synaptic plasticity in living mice Year: (2023)
Ref_id:b73 Title: Evaluation and development of deep neural networks for image super-resolution in optical microscopy Year: (2021)
