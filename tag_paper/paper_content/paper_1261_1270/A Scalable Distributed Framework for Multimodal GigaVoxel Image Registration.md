Title: A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration
Abstract: In this work, we propose FFDP, a set of IO-aware non-GEMM fused kernels supplemented with a distributed framework for image registration at unprecedented scales. Image registration is an inverse problem fundamental to biomedical and life sciences, but algorithms have not scaled in tandem with image acquisition capabilities. Our framework complements existing model parallelism techniques proposed for large-scale transformer training by optimizing non-GEMM bottlenecks and enabling convolution-aware tensor sharding. We demonstrate unprecedented capabilities by performing multimodal registration of a 100µm ex-vivo human brain MRI volume at native resolution -an inverse problem more than 570× larger than a standard clinical datum in about a minute using only 8 A6000 GPUs. FFDP accelerates existing state-of-the-art optimization and deep learning registration pipelines by upto 6 -7× while reducing peak memory consumption by 20 -59%. Comparative analysis on a 250µm dataset shows that FFDP can fit upto 64× larger problems than existing SOTA on a single GPU, and highlights both the performance and efficiency gains of FFDP compared to SOTA image registration methods.

Section: Introduction
Image registration (also called 'image alignment' or 'image matching') is a non-linear inverse problem ubiquitous in biomedical and life sciences. Given d-dimensional images F : Ω → R d and M : Ω → R d defined on domain Ω (usually a compact subset of R d ), image registration seeks to find a coordinate transform φ : Ω → Ω that deforms the moving image M to look similar to the fixed image F . Mathematically, we minimize the following objective (Fig. 1):
φ * = arg min φ∈G L(φ) . = C(F, M • φ) + R(φ)(1)
where C is a cost or dissimilarity function, and • is the interpolation operator, i.e. (I •g)(x) = I(g(x)) for all x ∈ Ω. Popular choices of φ are affine and deformable transforms, i.e. φ(x) = Ax + t, and φ(x) = x + u(x). Modern registration pipelines (Hoffmann et al., 2021;Jena et al., 2024a) consider an affine matching followed by a deformable matching step, resulting in a composite transform φ(x) = Ax + t + u(x). u is called the displacement field, modeled as a grid of per-voxel vectors u(x) ∈ R d . For an image of size N , the displacement field is a tensor of size dN . We use [x] Ω , A[x] Ω + t, and [u] Ω to denote the identity grid, grid of affine transformed coordinates, and deformation grid defined on Ω respectively. Common choices of C are mean squared error, Localized Normalized Cross Correlation (Avants et al., 2008a), and Mattes Mutual Information (Mattes et al., 2001). Common choices of R include Sobolev norm of the gradient or warp fields (Beg et al., 2005;Mang et al., 2019;Avants et al., 2008b), total variation, and inverse-consistency (Christensen & Johnson, 2001). To optimize Eq. ( 1), iterative methods optimize φ * directly using gradient descent, and deep learning methods learn a deep neural network φ = f θ (F, M ). Image registration establishes a common coordinate system, aligning scans across individuals and atlases (Hering et al., 2022; Green denotes the optimizable warp, red denotes the primary bottlenecks that we optimize in this paper. Marcus et al., 2007;Murphy et al., 2011). This alignment is a prerequisite for multimodal data fusion, cross-subject comparison, morphometric analysis (Das et al., 2009), and construction of large-scale atlases (Wang et al., 2020b). Establishing such voxelwise correspondence is fundamental for studying anatomical variability, detecting pathological signatures (Ravikumar et al., 2021), and advancing precision medicine (Börner et al., 2022;Jonsson et al., 2022). The saliency and centrality of the task across various biomedical and life science applications has spurred numerous methodological advances in the field, spanning more than three decades of research (Gee et al., 1993;Tian et al., 2024).
Over the past decade, advances in MRI, CT, PET, STPT, and microscopy have enabled ultra-highresolution imaging, often more than three orders of magnitude larger than macroscopic biomedical domains (Balchandani & Naidich, 2015;Esquivel et al., 2022;Badawi et al., 2019;Gambarotto et al., 2019;Wassie et al., 2019;Kleven et al., 2023;Wang et al., 2020b;Mansour et al., 2025;Kleinfeld et al., 2011). While a typical clinical registration problem involves ∼20M parameters, high-resolution ex-vivo human brain scans can require solving up to 11B parameters, far beyond the ∼50M-parameter scale at which current registration methods remain reliable. As a result, state-of-the-art deformable image alignment struggles to scale to the resolutions demanded in modern neuroimaging, computational pathology, developmental biology, and connectomics, creating a substantial performance gap. In parallel, innovations in large-scale transformer training such as IO-aware fused operations (Dao et al., 2022;Dao, 2023;Spector et al., 2025) and 5D parallelism for distributing larger-than-memory workloads (Shoeybi et al., 2019;Li et al., 2023;Jacobs et al., 2024;Li et al., 2024;Zhao et al., 2023;Ansel et al., 2024) optimize GEMM-like workflows. However, the fundamental concepts utilized by these methods (IO-awareness, recomputing and aggregating intermediates on shared memory to minimize high bandwidth memory (HBM) storage, identifying partial aggregates across hosts to minimize communication overheads for distributed optimization) are broadly applicable to a wide class of problems of the non-GEMM nature.
In this paper, we apply these concepts to scale image registration algorithms to match parity with the developments in both increasing resolution of image acquisition and compute capabilities. To that end, our contributions are twofold. First, we identify key compute and memory bottlenecks in image registration algorithms, and propose novel components that fit problems upto 64× larger than existing algorithms on a single GPU. Second, we propose Flash Fused Distributed Primitives (FFDP), a distributed framework to scale registration to an arbitrary number of GPUs, thereby scaling to ultra high-resolution problems. We present a first-of-its-kind demonstration: aligning a 250µm in-vivo MRI (Lüsebrink et al., 2017) to a 100µm ex-vivo human brain FLASH volume (Edlow et al., 2019)a multimodal registration problem more than 570× larger than a standard clinical datum (Marcus et al., 2007), with over 11.8B transform parameters -completed in one minute using only 8 A6000 GPUs. FFDP accelerates existing traditional registration pipelines by upto 7.48× while reducing memory consumption by upto 59%, and deep learning pipelines by upto 6.14× while consuming upto 24% less memory. We highlight the necessity of performing high-resolution registration by comparing our method with various SOTA optimization and deep learning baselines on a 250µm T1-weighted MRI dataset, showing unprecedented performance and gains in efficiency.
2 Related Work
this section cite: ['b49', 'b90', 'b15', 'b86', 'b25', 'b48', 'b88', 'b94', 'b29', 'b104', 'b19', 'b59', 'b42', 'b119', 'b13', 'b36', 'b11', 'b41', 'b128', 'b65', 'b87', 'b64', 'b28', 'b27', 'b114', 'b111', 'b72', 'b52', 'b71', 'b139', 'b4', 'b83', 'b88']

Section: Memory Efficient and Large Scale Optimization
Recent years have also witnessed tremendous innovations in large-scale transformer model training. IO-aware implementations typically include individual fused kernels (Dao et al., 2022;Dao, 2023) and domain-specific languages (Spector et al., 2025;PyTorch, 2025) to minimize launch latency and large memory overheads. To distribute larger-than-memory model training workloads across multiple GPUs, 5D parallelism techniques (Shoeybi et al., 2019;Li et al., 2023;Jacobs et al., 2024;Li et al., 2024;Zhao et al., 2023;Ansel et al., 2024) have been proposed. Many of these techniques leverage a divide-and-conquer approach to break down a larger GEMM-like operation like matrix multiplication or attention into smaller sub-problems that can be executed on multiple GPUs and synchronized to compute the final result. To our knowledge, most of these techniques are tailored to transformer-specific architectures and GEMM-like operations (self attention, FeedForward, LayerNorm, etc.) only, and a Model Parallel variant for convolution-aware tensor sharding and synchronization is not available.
this section cite: ['b28', 'b27', 'b114', 'b99', 'b111', 'b72', 'b52', 'b71', 'b139', 'b4']

Section: Large Scale Registration in Life Sciences and Biomedical Imaging
Ex-vivo neuroimaging. Neuroanatomical studies often integrate high-resolution ex-vivo MRI, blockface imaging, and histology to bridge the gap between in-vivo imaging and microscopic "gold standard" pathology (Casamitjana et al., 2025;Ravikumar et al., 2024). While large-scale consortia like SEA-AD and HMBA, along with submillimeter whole-brain datasets (Edlow et al., 2019;Lüsebrink et al., 2017), aim to map cellular and molecular organization across species, computational costs often limit analysis to local effects. Specifically, existing tools cannot register these datasets at native resolution due to excessive memory requirements. We overcome this limitation, demonstrating native-resolution registration of whole-brain datasets in one minute using eight A6000 GPUs (see Section 5.2), thereby preserving fine anatomical details typically lost to downsampling.
this section cite: ['b22', 'b105', 'b35', 'b83']

Section: Large-scale registration in model organisms.
Over the past decade, imaging across the life sciences and biomedical domains has progressed from mesoscale surveys to organ-and organism-wide acquisitions at cellular or even subcellular resolution. These span transparent organisms and small animal models (e.g., C. elegans, zebrafish, adult Drosophila) (Varol et al., 2020;Venkatachalam et al., 2016;Marquart et al., 2017;Gupta et al., 2018;Peng et al., 2011;Brezovec et al., 2024), adult mouse and rat brains imaged at sub-micron resolutions (Gong et al., 2016;Wang et al., 2020a;Kleven et al., 2023) using Light Sheet Fluorescence Microscopy (LSFM) and Serial Two-Photon Microscopy (STPT) imaging. Such modalities routinely generate gigavoxel to teravoxel volumes (Kutten et al., 2016;Nazib et al., 2018). Their scientific utility, however, hinges on the ability to perform registration at the native resolution of acquisition, i.e. aligning specimens (or modalities) in a common coordinate system without sacrificing the fine-scale morphologies including cell bodies, layers, axon bundles, synaptic neighborhoods, etc. that motivate high-resolution acquisition in the first place (Nazib et al., 2018;Goubran et al., 2013).
Across these diverse domains, the unifying requirement demands access to scalable multimodal registration algorithms -a challenge we address in this work. We provide an extended discussion of more related work and the necessity of our approach in Section A.
this section cite: ['b123', 'b124', 'b89', 'b46', 'b97', 'b20', 'b43', 'b65', 'b68', 'b95', 'b95', 'b44']

Section: Fused Kernels for Memory Efficient Registration on a Single GPU

this section cite: []

Section: Bottlenecks of a deformable image registration pipeline
Our primary objective is to identify compute and memory bottlenecks in large-scale image matching tasks. In identifying these bottlenecks, training-free optimization methods are better suited than deep networks since the latter has a much larger activation memory footprint, which forms the primary memory bottleneck (Tazi et al., 2024).
For instance, for a 250µm image pair, a standard deep learning method (Hoffmann et al., 2021) generates an activation map of size 27GB only after the first layer. Extrapolating memory usage for clinical data, existing deep networks will require upto 1.2TB of GPU memory at inference to process these image volumes at native resolution. In contrast, a training-free optimizer can fit this problem in less than 45GB of GPU memory. We use FireANTs (Jena et al., 2024a) as our base framework to identify compute and memory bottlenecks in a typical image registration problem. We analyze the flamegraph of a typical clinical MRI registration task from the OASIS brain dataset (Marcus et al., 2007) in Fig. 20. We identify three key memory bottlenecks in image matching pipelines (1) deformable interpolation and warp composition (2) cross-correlation loss, and (3) mutual information loss (see Fig. 2(right)).foot_0
We first propose efficient designs to fit larger problems on a single GPU, and then extend the framework to distributed registration.
this section cite: ['b117', 'b49']

Section: Composite Implicit Grid Sampler
A fundamental operation used in image registration is the grid sampler. This operator allows us to warp an image M using a deformation field φ : Ω → Ω and computes the image
M ′ : M ′ (x) = M (φ(x)).
Virtually every image registration pipeline uses this operation to warp the moving image using an affine, deformable, or composite transform. For affine and composite transforms, the operator initializes a regular grid [x] Ω , a grid of size 3N . The affine grid A[x] Ω + t is another grid of size 3N . If a deformable grid [u] Ω is optimized, then a third grid A[x] Ω + t + [u] Ω is materialized, costing a total of 9N overhead for an image of size N . To consolidate these memory overheads, we propose a composite implicit grid sampler. This is a fused CUDA kernel that performs the following operation:
fused grid sampler(I; A, t, [u], S, x bounds )(x) = I(Ax + t + Su(x))
where A, S ∈ GL(d, R) are affine matrices, t is a translation vector, [u] is the deformation grid, and x bounds are the bounds of the (implicit) identity grid [x] Ω . There are three benefits of this approach. First, the kernel avoids materializing any additional grids in HBM, reducing the memory overhead of the kernel from O(n) to O(1) with no loss in runtime or accuracy. Second, when the warp [u] Ω is sharded across hosts in a distributed setting, the identity grid [x] Ω needs to be sharded correctly too. Since the identity grid is implicitly defined by its bounds x bounds = (x min , x max ) ∈ R 2d , our implementation can be easily used in a distributed optimization setting without instantiating partial shards [x] Ω h . Finally, the matrix S is used to rescale the deformation field to sample from the coordinates of the sharded images I h which lie on the grid Ω h instead of Ω (see Section I.2) without initializing additional memory. The backward pass is very similar to the existing PyTorch implementation, with the exception of the gradient of the affine matrix. We discuss the derivation and pseudocode of the forward and backward pass in the Section H.
this section cite: []

Section: Implicit Parzen Windowing for Mutual Information
Mattes Mutual Information (MI) is one of the most commonly used loss functions for multimodal image matching (Chen et al., 2022;Avants et al., 2009;Mattes et al., 2001). For random variables X and Y , MI is the KL divergence between the joint distribution P (X, Y ) and product of marginal distributions P (X)P (Y ) of the intensities of the two images. For image matching, X and Y are the pixel intensities for the images I, J. The distributions are estimated using a kernel density estimator:
P I (v) = 1 N k κ(v -I k ), P (I,J) (v, w) = 1 N k κ(v -I k )κ(w -J k )(2)
where κ is a kernel function of choice. Common choices of κ are the Gaussian (Guo, 2019) and 3rd order B-Spline kernels (Thévenaz & Unser, 2000). To empirically compute the KL divergence, the distributions Eq. ( 2) are discretized over B equally spaced bins on the domain of u ∈ I, v ∈ J. However, to compute the joint histogram of size B 2 , this method requires materializing the entire Parzen Block Ψ I (j, k) = κ(b j -I k ) of size 2k P BN , where k P is a kernel-dependent constant. Since N >> B (B is typically chosen to be 32), this operation becomes a significant memory bottleneck for large N . For instance, a typical clinical image volume (N ≈ 30MB) with 32 bins will consume 7.5GB of HBM -a significantly huge cost that grows much faster for larger problems.
Our efficient implementation leverages the fact that B is small to avoid materializing the tensors Ψ I , Ψ J ∈ R B×N altogether and use high-throughput shared memory to compute and accumulate the histogram entries and partial gradients for each image pixel. We provide the detailed derivation in Section G. This leads to an efficient implementation that consumes O(1) additional HBM instead of O(N ) (holding B constant). This leads to upto 98% lesser HBM usage for images considered in our experiments, and an asymptotic 100% reduction in HBM usage for large images (Fig. 7(top-right)).
this section cite: ['b23', 'b90', 'b118']

Section: Efficient Implicit Fused Cross-Correlation
Local Normalized Cross-Correlation (LNCC) is used ubiquitously in signal and image processing as a similarity metric. In deformable image registration, it is used as a robust similarity function to compare anatomical similarities (Chen et al., 2022;Hoffmann et al., 2021;Avants et al., 2008b;Wu et al., 2024). Most LNCC implementations are memory-bound due to the large number of intermediate variables. Our analysis in Section F shows that the computational graph adds 16× HBM overhead, and upto another 16× HBM overhead for computing gradients with respect to all intermediates.
To avoid these huge memory overheads, we fuse all the intermediate computation in a fused kernel.
Our fused forward pass requires only 5× memory for storing all intermediates (I, J, I 2 , J 2 , IJ convolved with matrix w). In Section F we analytically derive the gradient and show that the input gradients can be computed by modifying the saved intermediates in-place. This leads upto a 76.5% reduction in memory (see Table 3) and outperforms even torch.compile implementations.
this section cite: ['b23', 'b49', 'b132']

Section: Extending image registration to multiple GPUs
Our composite implicit grid sampler and improved loss functions allows optimizing problems with image sizes that are upto two magnitudes larger than other baselines on a single A6000 GPU (Fig. 5a). However, many applications using mesoscopic and microscopic data require registration of images that do not fit on a single GPU. Inspired by distributed frameworks for LLM training (Shoeybi et al., 2019;Rajbhandari et al., 2020) and initial work on distributed image registration (Mang et al., 2019), we propose a distributed framework that allows sharding large images across multiple GPUs to efficiently scale to arbitrarily large problem sizes with any similarity loss function.
this section cite: ['b111', 'b103', 'b86']

Section: Distributed Setting.
For distributed registration with H hosts or GPUs, we partition the domain P (Ω) = {Ω 1 , Ω 2 , . . . Ω H } such that |Ω i | = N/H, Ω i ∩ Ω j = ϕ ∀i ̸ = j and ∪ i Ω i = Ω. We use [x] Ω h , A[x] Ω h + t, and [u] Ω h to denote the sharded tensors defined on domain Ω h .
this section cite: []

Section: Grid Parallel for Boundary-Synchronized Image Sharding
Techniques like Tensor/Sequence/Expert/Context Parallel have been tremendously successful in distributed optimization by sharding large models and sequences across multiple GPUs (Shoeybi et al., 2019;Li et al., 2023;Liu et al., 2024b;a). However, these techniques work for transformerlike architectures and input sequences where the model parameters and activations do not require boundary synchronization. In contrast, image registration contains operations that require boundary synchronization between image and grid shards to perform mathematically correct convolutions. Examples of such operations include convolutions for calculating LNCC, total variation loss, Sobolev norm of the gradient and warp fields (Mang et al., 2019;Avants et al., 2008b;Beg et al., 2005).
To enable these functionalities and complement existing parallelism techniques, we propose 'Grid Parallel' (GP) as an abstraction on a tensor. GP shards a tensor across hosts, stores the sharded dimension and bounds as metadata, and provides synchronization operations to augment the tensor with sufficient boundary padding from neighboring shards prior to performing a convolution operation. GP allows us to partition the fixed images, [u], and the optimizer state [m 1 ], [m 2 ] -essentially sharding the entire problem across H hosts while allowing the user to apply convolutional operations seamlessly. We compare the performance of GP with naive DTensor sharding in Section D.
this section cite: ['b111', 'b72', 'b86', 'b15']

Section: Distributed Ring Sampler
Despite the sharding in GP, the moving image M cannot be sharded across GPUs due to the random-access nature of the grid sample operation applied on M . In general, the warp vector φ(x) residing on GPU i can point to coordinates that reside on the sharded image on GPU j for any j ̸ = i. Even for neighboring coordinates x s , x u ∈ [x] i , the coordinates φ(x s ) and φ(x u ) can point to different shards j 1 ̸ = j 2 ̸ = i. This is illustrated in Fig. 4(a). Keeping the entire moving image in memory limits the maximum problem size to N ≤ V , where V is the memory per GPU, regardless of the number of hosts H. However, we want the maximum problem size to scale with H. Therefore, we propose a distributed grid sampler that allows us to correctly interpolate the moving image with sharded images scattered across multiple hosts without performing an allgather operation on the moving image.
Our approach leverages the key observation that (bi/tri)linear interpolation can be decomposed as an aggregate of partial sums of interpolated values on individual image shards.  Resolution Method AvgDice Score ↑ InvDice Score ↑ AvgHD90 cum (mm) ↓ 1 mm Baseline 0.579 ± 0.055 0.141 ± 0.142 1.587 ± 0.908 Anatomix 0.796 ± 0.035 0.386 ± 0.138 0.468 ± 0.137 CLAIRE 0.776 ± 0.044 0.344 ± 0.120 0.554 ± 0.150 FireANTs 0.822 ± 0.032 0.435 ± 0.147 0.393 ± 0.126 ITK-dreg 0.662 ± 0.055 0.199 ± 0.125 1.002 ± 0.277 SynthMorph 0.801 ± 0.022 0.378 ± 0.133 0.455 ± 0.098 TransMorph 0.851 ± 0.016 0.468 ± 0.161 0.310 ± 0.064 UniGradICON (IO) 0.826 ± 0.022 0.391 ± 0.155 0.384 ± 0.095 UniGradICON 0.815 ± 0.026 0.393 ± 0.156 0.419 ± 0.113 VFA 0.851 ± 0.023 0.494 ± 0.169 0.323 ± 0.096 Ours 0.838 ± 0.028 0.436 ± 0.148 0.341 ± 0.109 500 µm Baseline 0.580 ± 0.055 0.138 ± 0.143 1.357 ± 0.326 Anatomix † 0.758 ± 0.040 0.325 ± 0.159 0.619 ± 0.169 CLAIRE 0.779 ± 0.051 0.275 ± 0.210 0.570 ± 0.211 FireANTs 0.841 ± 0.033 0.489 ± 0.163 0.340 ± 0.127 ITK-dreg 0.699 ± 0.056 0.240 ± 0.130 0.834 ± 0.254 SynthMorph † 0.771 ± 0.035 0.337 ± 0.133 0.557 ± 0.144 TransMorph † 0.759 ± 0.028 0.300 ± 0.175 0.624 ± 0.127 UniGradICON † 0.610 ± 0.044 0.133 ± 0.122 1.231 ± 0.262 UniGradICON (IO) † 0.615 ± 0.047 0.149 ± 0.136 1.527 ± 1.495 VFA † 0.805 ± 0.044 0.419 ± 0.181 0.462 ± 0.163 Ours 0.872 ± 0.028 0.528 ± 0.180 0.258 ± 0.099 250 µm Baseline 0.580 ± 0.055 0.136 ± 0.141 1.409 ± 0.322 Anatomix † 0.620 ± 0.031 0.161 ± 0.115 1.179 ± 0.190 CLAIRE 0.809 ± 0.054 0.378 ± 0.133 0.570 ± 0.211 FireANTs † 0.777 ± 0.064 0.341 ± 0.199 0.629 ± 0.295 ITK-dreg 0.758 ± 0.048 0.299 ± 0.125 0.613 ± 0.191 SynthMorph † 0.690 ± 0.052 0.243 ± 0.164 0.882 ± 0.239 TransMorph † 0.689 ± 0.044 0.191 ± 0.132 0.973 ± 0.245 UniGradICON (IO) † 0.398 ± 0.062 0.063 ± 0.071 3.491 ± 3.198 UniGradICON † 0.359 ± 0.044 0.045 ± 0.056 2.992 ± 0.670 VFA † 0.714 ± 0.066 0.281 ± 0.216 0.821 ± 0.300 Ours 0.895 ± 0.029 0.597 ± 0.204 0.216 ± 0.098  problem sizes for sufficiently large H. The detailed derivation and correctness of this operation is shown in Section I.
this section cite: []

Section: Distributed Loss Functions
Since the moved image and fixed image are sharded cross H hosts, the loss function must take this into account to compute the loss function correctly.
this section cite: []

Section: Mean Squared Error (MSE).
Since MSE is a per-pixel loss, we compute the individual MSE on host h and perform an allreduce operation.
this section cite: []

Section: Localized Normalized Cross Correlation (LNCC).
The LNCC computes per-pixel patch similarities for each pixel, using a convolution over its neighbors. For sharded images, the patch statistics at the boundary requires a boundary synchronization with its neighboring shards which is provided by our GP implementation. After computing the LNCC for all pixels in each shard, we perform another allreduce to compute the LNCC over the entire image.
this section cite: []

Section: Mutual Information (MI).
The MI loss computes the joint histograms p (I,J) (x, y) and marginals p I (x), p J (y). However, these distributions are partial aggregates from the sharded images on each GPU. Eq. ( 2) can be rewritten as
p I (v) = h N h N 1 N h k∈Ω h κ(v -I k ) , p IJ (v, w) = h N h N 1 N h k∈Ω h κ(v -I k )κ(w -J k ) ,
where the red terms correspond to the per-host histogram computation. Performing an allreduce to compute the weighted average of these histograms (with weights N h /N ) results in a valid and correct joint and marginal distributions over all hosts. This also leads to only a B 2 + 2B communication overhead regardless of N , making a distributed implementation highly practical. 5 Experiments Our primary goals are to (a) accelerate both optimization and neural network based registration workflows, and (b) solve significantly larger image registration problems. We show the efficacy of our method by accelerating existing registration workflows on standard clinical data. This is followed by optimizing a multimodal registration task with more than 11.8B optimizable parameters, an unprecedented result in large-scale registration. We compare the performance and computational efficiency of our method with various state-of-the-art baselines on a simulated 250µm ex-vivo brain MRI dataset, followed by ablations on various components of our framework.
Baselines. To accelerate existing registration workflows, we compare against TransMorph (Chen et al., 2022) and FireANTs (Jena et al., 2024a), which are state-of-the-art deep learning and optimization based registration frameworks respectively. In addition, we perform comparative evaluation with two methods explicitly designed for large-scale registration: ITK-DReg (itk) (CPU-based) and CLAIRE (Mang et al., 2019) (multi-GPU), and several SOTA learning-based approaches for clinical data -SynthMorph (Hoffmann et al., 2021), Vector-Field Attention (Liu et al., 2024c), unigradICON (Tian et al., 2024) (with/without instance optimization), anatomix+ConvexAdam (Dey et al., 2025).
this section cite: ['b23', 'b86', 'b49', 'b119', 'b32']

Section: Accelerating existing registration workflows and ablations
For deep networks, we train TransMorph-large under three loss configurations: (a) LNCC+Dice, (b) MI+Dice, and (c) LNCC+scaling-and-squaring (Ashburner, 2007) +Dice. For each configuration shown in Table 1, we either use the vanilla PyTorch implementation (Baseline) or our kernels (Ours). For classical optimization, we benchmark runtime and memory against multiple LNCC backends (FireANTs, VoxelMorph/TransMorph, Fast LNCC, torch.compile, and Ours) and MI backends (PyTorch and Ours with and without torch.compile). Tables 1 and 4 and Fig. 12 show that during network training our kernels converge 6.1× faster with LNCC while using 16.5% less memory, and reduce MI memory usage by 24.7%. Despite being designed for very large images, the runtime and memory benefits are significant for clinical-scale data (i.e., 30MB for OASIS). Optimization frameworks see larger gains: FireANTs achieves up to 95.2% memory savings and 2.6× speedup with MI, and a 7.5× speedup over FastLNCC (Jia et al., 2025) (and 2.9× over FireANTs' LNCC backend which applies separable convolutions on FastLNCC), with 44-59% lower memory usage overall.
this section cite: ['b6', 'b57']

Section: Registration to a 100 micron ex-vivo brain MRI volume
To showcase the efficacy of our method on real large scale images, we register a 250µm in-vivo MRI image (Lüsebrink et al., 2017) to a 100 µm ex-vivo FLASH human brain volume (Edlow et al., 2019). This represents an inverse problem with more than 11.2B optimizable parameters (compared to ∼20M for clinical datasets), or 44.8GB of GPU memory. The entire problem does not fit on most GPUs, necessitating distributed multimodal registration. We optimize a composite transform -affine followed by a diffeomorphic mapping; details can be found in Section E.1. Multimodal deformable registration took ∼58 seconds on 8 NVIDIA A6000 GPUs, which is unprecedented at this resolution. Fig. 6 shows qualitative results, highlighting the ability to register highly detailed structures such as cerebellar white matter; these structures are not visible at macroscopic scales. The resultant advantages of performing registration at this scale can allow researchers to characterize the neuroanatomy at microscopic resolutions and allow morphometric analysis of cortical layers and subcortical nuclei among other structures.
Registration accuracy in these studies is measured using privately annotated fiducial markers, hindering reproducibility and comparability of methodological advances. Due to lack of scalable frameworks, most high-resolution studies simply run ANTs at a significantly downsampled resolution (Kleven et al., 2023;Mansour et al., 2025;Wang et al., 2020b;Kronman et al., 2024;Bogovic et al., 2020;Edlow et al., 2019) and upsample the warp field to the native resolution.  (Wang et al., 2020b;Mansour et al., 2025;Edlow et al., 2019), we register the images at a downsampled resolution, and then upsample the deformation field (b) inspired by several histology registration methods (Wodzinski et al., 2024;Lotz et al., 2015;Liang et al., 2021), we perform patchwise registration and mosaicing of the final deformation. We compare the methods at three resolutions: 1mm, 500µm, and 250µm. At 1mm, the full image fits within a patch, providing a baseline reference comparable to reported OASIS performance. At higher resolutions, patches are defined by each method's default input size with stride equal to 50% of the patch size. FireANTs augmented with FFDP is denoted as Ours. We report Dice, inverse-weighted Dice (InvDice; Mang et al. (2019)), and average Haussdorf distance capped at 90 percentile (AvgHD90). To compare efficiency, we measure both wall-clock time and GPU-hours.
this section cite: ['b83', 'b35', 'b65', 'b87', 'b18', 'b35', 'b87', 'b35', 'b130', 'b81', 'b73', 'b86']

Section: Comparative Analysis on a Simulated ex-vivo Brain MRI Dataset
Results. Fig. 5a summarizes performance metrics. At 1mm, most methods achieve performance consistent with their reported performance on OASIS, including VFA and TransMorph which were trained on the OASIS dataset with label supervision. At higher resolutions, nearly all methods degrade, especially for InvDice and HD90, which emphasize alignment of fine structures. In contrast, our method improves in accuracy: at 250µm, we improve Dice by 18.1 points, InvDice by 31.6 points, and reduce AvgHD90 by 62.1%. The correlation between resolution and performance is also observed in (Mang et al., 2019;Mang & Ruthotto, 2017;Nazib et al., 2018); in addition we verify that patch-based methods degrade in performance at higher resolutions. (a) Weak scaling and Per-GPU memory consumption of FFDP. (b) Qualitative ablation of GP synchronization in FFDP on the fMOST mouse brain dataset (Tustison et al., 2024). Red arrows highlight regions affected by incorrect boundary effects due to no GP. See Fig. 10 for more examples.
This degradation among patchwise methods is expected; histology-style pipelines typically register consecutive slides with small deformations after affine alignment. At high resolution, patching reduces anatomical context and the patches become progressively more out-of-distribution (see Fig. 19). Patchwise or downsampling strategies are therefore insufficient for ultra-high resolution large-scale registration, and existing deep methods cannot be repurposed to work at higher resolutions efficiently. Accuracy-efficiency tradeoffs in Figs. 5b and 5c show that our method is Pareto-efficient compared to all other methods (CPU, deep learning, and distributed GPU methods), requiring up to 500× fewer GPU-hours compared to alternatives at 250µm.
this section cite: ['b86', 'b85', 'b95']

Section: Ablation Studies
We ablate on the efficiency of various workhorse operations used in image registration in Fig. 7 and Table 3. We compare our implementations to community-standard Py-Torch implementation (Jia et al., 2025;Chen et al., 2022) and torch.compile versions. For grid sampler and MI kernels, our kernels have O(1) extra HBM overhead instead of O(N ) in the PyTorch implementation. For LNCC, our implementation achieves an average speedup in the forward pass by 5.22× and 56.98× in the backward pass.
Our grid sampler also leads to an efficient scaling-andsquaring operation, commonly used in deep learning registration pipelines (Chen et al., 2022), with a memory reduction of 50% compared to the baseline implementation. Scalability Analysis. We test the weak scaling of our distributed framework by registering synthetic images with increasing voxel sizes. For H GPUs, we instantiate an image pair of size 700 × 700 × 700H and shard the images, warp, and optimizer state across H GPUs. Fig. 8a shows weak scaling of FFDP with and without ring sampler. Without the ring sampler, the grid sample operation requires storing the moving image of size 700 × 700 × 700H on each GPU, leading to peak HBM memory increasing linearly with H. This implies the framework would not scale to arbitrarily large problem sizes, regardless of cluster size H. Peak Memory consumption is independent of H with the Ring Sampler, and scaling efficiency is only minimally affected. Ablation on GP. We ablate the effect of GP by replacing it with DTensor sharding (no boundary sync). Figs. 8b, 9 and 10 show that incorrect boundary synchronization leads to undesirable artifacts in the moved images, and reduces labelmap overlap.
this section cite: ['b57', 'b23', 'b23']

Section: Conclusion
We propose a novel distributed framework for arbitrarily large image registration problems. Our work identifies and proposes IO-aware and distributed-friendly implementations of workhorse operations in image registration algorithms, enabling registration of images at arbitrarily large resolutions on a single GPU. Our fused primitives demonstrate compelling results in both improving existing registration pipelines and scaling to arbitrarily large, multimodal problems pertinent in modern life science applications, that were previously infeasible without approximations. FFDP shows unprecedented registration capabilities that will enable researchers to leverage and effectively work with large-scale image volumes and unearth new insights leveraging the large resolution images.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: ()
Ref_id:b1 Title: A framework for distributed, large-scale image registration Year: ()
Ref_id:b2 Title: Tensorflow: Large-scale machine learning on heterogeneous distributed systems Year: (2016)
Ref_id:b3 Title: Non-linear registration, aka spatial normalisation fmrib technical report tr07ja2 Year: (2007)
Ref_id:b4 Title: Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation Year: (2024)
Ref_id:b5 Title: Advanced normalization tools (ants) Year: ()
Ref_id:b6 Title: A fast diffeomorphic image registration algorithm Year: (2007)
Ref_id:b7 Title: Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain Year: (2008-02)
Ref_id:b8 Title: Symmetric diffeomorphic image registration with cross-correlation: Evaluating automated labeling of elderly and neurodegenerative brain Year: (2008-02)
Ref_id:b9 Title: Lagrangian frame diffeomorphic image registration: Morphometric comparison of human and chimpanzee cortex Year: (2006-06)
Ref_id:b10 Title: Advanced normalization tools (ants) Year: (2009)
Ref_id:b11 Title: First human imaging studies with the explorer total-body pet scanner Year: (2019)
Ref_id:b12 Title: VoxelMorph: A Learning Framework for Deformable Medical Image Registration Year: (2019-08)
Ref_id:b13 Title: Ultra-high-field mr neuroimaging Year: (2015)
Ref_id:b14 Title: Improved visualization of cortical lesions in multiple sclerosis using 7t mp2rage Year: (2018)
Ref_id:b15 Title: Computing large deformation metric mappings via geodesic flows of diffeomorphisms Year: (2005)
Ref_id:b16 Title: A case study in cuda kernel fusion: Implementing flashattention-2 on nvidia hopper architecture using the cutlass library Year: (2023)
Ref_id:b17 Title: Synthseg: Segmentation of brain mri scans of any contrast and resolution without retraining Year: (2023)
Ref_id:b18 Title: An unbiased template of the drosophila brain and ventral nerve cord Year: (2020)
Ref_id:b19 Title: Tissue registration and exploration user interfaces in support of a human reference atlas Year: (2022)
Ref_id:b20 Title: Mapping the neural dynamics of locomotion across the drosophila brain Year: (2024)
Ref_id:b21 Title: Deformable image registration based on similarity-steered cnn regression Year: (2017)
Ref_id:b22 Title: A probabilistic histological atlas of the human brain for mri segmentation Year: (2025)
Ref_id:b23 Title: TransMorph: Transformer for unsupervised medical image registration Year: (2022-11)
Ref_id:b24 Title: TVM: An automated End-to-End optimizing compiler for deep learning Year: (2018-10)
Ref_id:b25 Title: Consistent image registration Year: (2001)
Ref_id:b26 Title: Mutual information as a general measure of structure in interaction networks Year: (2020)
Ref_id:b27 Title: Flashattention-2: Faster attention with better parallelism and work partitioning Year: (2023)
Ref_id:b28 Title: Flashattention: Fast and memoryefficient exact attention with io-awareness Year: (2022)
Ref_id:b29 Title: Registration based cortical thickness measurement Year: (2009)
Ref_id:b30 Title: Image registration and template based annotation of great ape skulls Year: (2018)
Ref_id:b31 Title: A deep learning framework for unsupervised affine and deformable image registration Year: (2019)
Ref_id:b32 Title: Learning general-purpose biomedical volume representations using randomized synthesis Year: (2025)
Ref_id:b33 Title: Flex attention: A programming model for generating optimized attention kernels Year: (2024)
Ref_id:b34 Title: Atrophy in the parahippocampal gyrus as an early biomarker of alzheimer's disease Year: (2011)
Ref_id:b35 Title: 7 tesla mri of the ex vivo human brain at 100 micron resolution Year: (2019)
Ref_id:b36 Title: Photon-counting detector ct: key points radiologists should know Year: (2022)
Ref_id:b37 Title: Lung250m-4b: A combined 3d dataset for CT-and point cloud-based intra-patient lung registration Year: (2023)
Ref_id:b38 Title: Pydpiper: a flexible toolkit for constructing novel registration pipelines Year: (2014)
Ref_id:b39 Title: Surfacenets for multi-label segmentations with preservation of sharp boundaries Year: (2022)
Ref_id:b40 Title: Early alzheimer's disease-like reductions in gray matter and cognitive function with aging in nonhuman primates Year: (2022)
Ref_id:b41 Title: Imaging cellular ultrastructures using expansion microscopy (u-exm) Year: (2019)
Ref_id:b42 Title: Elastically deforming a three-dimensional atlas to match anatomical brain images Year: (1993)
Ref_id:b43 Title: High-throughput dual-colour precision imaging for brain-wide connectome with cytoarchitectonic landmarks at the cellular level Year: (2016)
Ref_id:b44 Title: Image registration of ex-vivo mri to sparsely sectioned histology of hippocampal and neocortical temporal lobe specimens Year: (2013)
Ref_id:b45 Title: Multi-modal image registration with unsupervised deep learning Year: (2019)
Ref_id:b46 Title: Morphometric analysis and neuroanatomical mapping of the zebrafish brain Year: (2018)
Ref_id:b47 Title: Learn2reg challenge: Ct lung registrationtraining data Year: (2020)
Ref_id:b48 Title: Learn2reg: comprehensive multi-task medical image registration challenge, dataset and evaluation in the era of deep learning Year: (2022)
Ref_id:b49 Title: Synthmorph: learning contrast-invariant registration without acquired images Year: (2021)
Ref_id:b50 Title: A Plug-and-Play Image Registration Network Year: (2024-03)
Ref_id:b51 Title: Crisp boundary detection using pointwise mutual information Year: (2014)
Ref_id:b52 Title: System optimizations for enabling training of extreme long sequence transformer models Year: (2024)
Ref_id:b53 Title: Fireants: Adaptive riemannian optimization for multi-scale diffeomorphic registration Year: (2024)
Ref_id:b54 Title: Deep learning in medical image registration: Magic or mirage? Year: (2024)
Ref_id:b55 Title: Deep implicit optimization enables robust learnable features for deformable image registration Year: (2025)
Ref_id:b56 Title: Zhaowen Qiu, and Jinming Duan. U-net vs transformer: Is u-net outdated in medical image registration? arXiv preprint Year: (2022)
Ref_id:b57 Title: A naive trick to accelerate training of lncc-based deep image registration models Year: (2025-02)
Ref_id:b58 Title: Mamba? catch the hype or rethink what really helps for image registration Year: (2024)
Ref_id:b59 Title: An image registration method for voxel-wise analysis of whole-body oncological pet-ct Year: (2022)
Ref_id:b60 Title: Hierarchical mixtures of experts and the em algorithm Year: (1994)
Ref_id:b61 Title: Diffeomorphic Image Registration using Lipschitz Continuous Residual Networks Year: ()
Ref_id:b62 Title: A 3d adult zebrafish brain atlas (azba) for the digital age Year: (2021)
Ref_id:b63 Title: Elastix: a toolbox for intensity-based medical image registration Year: (2009)
Ref_id:b64 Title: Large-scale automated histology in the pursuit of connectomes Year: (2011)
Ref_id:b65 Title: Waxholm space atlas of the rat brain: a 3d atlas supporting data analysis and integration Year: (2023)
Ref_id:b66 Title: Robust non-rigid registration through agent-based action learning Year: (2017)
Ref_id:b67 Title: Developmental mouse brain common coordinate framework Year: (2024)
Ref_id:b68 Title: Deformably registering and annotating whole CLARITY brains to an atlas via masked LDDMM Year: (2016)
Ref_id:b69 Title: Breadth-first pipeline parallelism Year: (2023)
Ref_id:b70 Title: CorticalFlow: A Diffeomorphic Mesh Transformer Network for Cortical Surface Reconstruction Year: (2021)
Ref_id:b71 Title: DISTFLASHATTN: Distributed memory-efficient attention for long-context LLMs training Year: (2024)
Ref_id:b72 Title: Sequence parallelism: Long sequence training from system perspective Year: (2023-07)
Ref_id:b73 Title: Improving algorithm for the alignment of consecutive, whole-slide, immunohistochemical section images Year: (2021)
Ref_id:b74 Title: Automatic registration of multisensor images using an integrated spatial and mutual information (smi) metric Year: (2013)
Ref_id:b75 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b76 Title: Ringattention with blockwise transformers for near-infinite context Year: ()
Ref_id:b77 Title: Vector field attention for deformable image registration Year: (2024)
Ref_id:b78 Title: Protocol for using serial two-photon tomography to map cell types and cerebrovasculature at single-cell resolution in the whole adult mouse brain Year: (2023)
Ref_id:b79 Title: epdevatlas: mapping gabaergic cells and microglia in the early postnatal mouse brain Year: (2025)
Ref_id:b80 Title: Marching cubes: A high resolution 3d surface construction algorithm Year: (1998)
Ref_id:b81 Title: Patch-based nonlinear image registration for gigapixel whole slide images Year: (2015)
Ref_id:b82 Title: Highly-accurate community detection via pointwise mutual information-incorporated symmetric non-negative matrix factorization Year: (2021)
Ref_id:b83 Title: T1-weighted in vivo human whole brain mri dataset with an ultrahigh isotropic resolution of 250 µm Year: (2017)
Ref_id:b84 Title: Imaging cortical multiple sclerosis lesions with ultra-high field mri Year: (2021)
Ref_id:b85 Title: A lagrangian gauss-newton-krylov solver for mass-and intensitypreserving diffeomorphic image registration Year: (2017)
Ref_id:b86 Title: CLAIRE: A distributedmemory solver for constrained large deformation diffeomorphic image registration Year: (2019-01)
Ref_id:b87 Title: The duke mouse brain atlas: Mri and light sheet microscopy stereotaxic atlas of the mouse brain Year: (2025)
Ref_id:b88 Title: Open access series of imaging studies (oasis): cross-sectional mri data in young, middle aged, nondemented, and demented older adults Year: (2007)
Ref_id:b89 Title: High-precision registration between zebrafish brain atlases using symmetric diffeomorphic normalization Year: (2017)
Ref_id:b90 Title: Nonrigid multimodality image registration Year: (2001)
Ref_id:b91 Title: An open resource for non-human primate imaging Year: (2018)
Ref_id:b92 Title: Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks Year: (2020-06)
Ref_id:b93 Title: Affine medical image registration with coarse-to-fine vision transformer Year: (2022)
Ref_id:b94 Title: Evaluation of registration methods on thoracic ct: the empire10 challenge Year: (2011)
Ref_id:b95 Title: Performance of registration tools on high-resolution 3d brain images Year: (2018)
Ref_id:b96 Title:  Year: (2021)
Ref_id:b97 Title: Brainaligner: 3d registration atlases of drosophila brains Year: (2011)
Ref_id:b98 Title: Learning representations by graphical mutual information estimation and maximization Year: (2023)
Ref_id:b99 Title: Fusing convolution and batch norm using custom function Year: (2021-04-18)
Ref_id:b100 Title: Zero bubble pipeline parallelism Year: (2023)
Ref_id:b101 Title: Learning diffeomorphic and modality-invariant registration using b-splines Year: (2021)
Ref_id:b102 Title: Igu-aug: Information-guided unsupervised augmentation and pixel-wise contrastive learning for medical image analysis Year: (2024)
Ref_id:b103 Title: Zero: Memory optimizations toward training trillion parameter models Year: (2020)
Ref_id:b104 Title: Ex vivo mri atlas of the human medial temporal lobe: characterizing neurodegeneration due to tau pathology Year: (2021)
Ref_id:b105 Title: Postmortem imaging reveals patterns of medial temporal lobe vulnerability to tau pathology in alzheimer's disease Year: (2024)
Ref_id:b106 Title: Svf-net: learning deformable image registration using shape matching Year: (2017)
Ref_id:b107 Title: A study on the statistical significance of mutual information between morphology of a galaxy and its large-scale environment Year: (2020)
Ref_id:b108 Title: Really fast isocontouring Year: (2023-06-13)
Ref_id:b109 Title: Flashattention-3: Fast and accurate attention with asynchrony and low-precision Year: (2024)
Ref_id:b110 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017)
Ref_id:b111 Title: Megatron-lm: Training multi-billion parameter language models using model parallelism Year: (2019)
Ref_id:b112 Title: The brain/minds marmoset connectivity resource: An open-access platform for cellular-level tracing and tractography in the primate brain Year: (2023)
Ref_id:b113 Title: Nonrigid image registration using multi-scale 3d convolutional neural networks Year: (2017)
Ref_id:b114 Title: Thunderkittens: Simple, fast, and $\textit{Adorable}$ kernels Year: (2025)
Ref_id:b115 Title: PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK) Year: (2019-05)
Ref_id:b116 Title: Brain-wide cellular resolution imaging of cre transgenic zebrafish lines for functional circuit-mapping Year: (2019)
Ref_id:b117 Title: The ultra-scale playbook: Training LLMs on GPU clusters Year: (2024)
Ref_id:b118 Title: Optimization of mutual information for multiresolution image registration Year: (2000)
Ref_id:b119 Title: Franc ¸ois-Xavier Vialard, Raúl San José Estépar, Sylvain Bouix, Richard Rushmore, and Marc Niethammer. unigradicon: A foundation model for medical image registration Year: (2024)
Ref_id:b120 Title: Dopamine and parkinson's disease Year: ()
Ref_id:b121 Title: Explicit b-spline regularization in diffeomorphic image registration Year: (2013)
Ref_id:b122 Title: The antsx ecosystem for mapping the mouse brain Year: (2024)
Ref_id:b123 Title: Statistical atlas of c. elegans neurons Year: (2020)
Ref_id:b124 Title: Pan-neuronal imaging in roaming caenorhabditis elegans Year: (2016)
Ref_id:b125 Title: Flashmask: Efficient and rich mask extension of flashattention Year: (2024)
Ref_id:b126 Title: The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas Year: (2020-05)
Ref_id:b127 Title: The allen mouse brain common coordinate framework: a 3d reference atlas Year: (2020)
Ref_id:b128 Title: Expansion microscopy: principles and uses in biological research Year: (2019)
Ref_id:b129 Title: Ultra-high-field 7t mri in parkinson's disease: ready for clinical use?-a narrative review Year: (2023)
Ref_id:b130 Title: Deeperhistreg: robust whole slide images registration framework Year: (2024)
Ref_id:b131 Title: NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration Year: (2022-02)
Ref_id:b132 Title: Neural ordinary differential equation based sequential image registration for dynamic characterization Year: (2024)
Ref_id:b133 Title: Quicksilver: Fast predictive image registration-a deep learning approach Year: (2017)
Ref_id:b134 Title: Native sparse attention: Hardware-aligned and natively trainable sparse attention Year: (2025)
Ref_id:b135 Title: Cascaded feature warping network for unsupervised medical image registration Year: (2021)
Ref_id:b136 Title: Recursive cascaded networks for unsupervised medical image registration Year: (2019-10)
Ref_id:b137 Title: Unsupervised 3d end-to-end medical image registration with volume tweening network Year: (2019)
Ref_id:b138 Title: Region mutual information loss for semantic segmentation Year: (2019)
Ref_id:b139 Title: Pytorch fsdp: experiences on scaling fully sharded data parallel Year: (2023)
