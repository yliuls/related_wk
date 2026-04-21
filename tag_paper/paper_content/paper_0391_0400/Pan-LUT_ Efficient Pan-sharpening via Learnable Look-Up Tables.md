Title: Pan-LUT: Efficient Pan-sharpening via Learnable Look-Up Tables
Abstract: Recently, deep learning-based pan-sharpening algorithms have achieved notable advancements over traditional methods. However, deep learning-based methods incur substantial computational overhead during inference, especially with large images. This excessive computational demand limits the applicability of these methods in real-world scenarios, particularly in the absence of dedicated computing devices such as GPUs and TPUs. To address these challenges, we propose Pan-LUT, a novel learnable look-up table (LUT) framework for pan-sharpening that strikes a balance between performance and computational efficiency for large remote sensing images. Our method makes it possible to process 15K×15K remote sensing images on a 24GB GPU. To finely control the spectral transformation, we devise the PAN-guided look-up table (PGLUT) for channel-wise spectral mapping. To effectively capture fine-grained spatial details, we introduce the spatial details look-up table (SDLUT). Furthermore, to adaptively aggregate channel information for generating high-resolution multispectral images, we design an adaptive output look-up table (AOLUT). Our model contains fewer than 700K parameters and processes a 9K×9K image in under 1 ms using one RTX 2080 Ti GPU, demonstrating significantly faster performance compared to other methods. Experiments reveal that Pan-LUT efficiently processes large remote sensing images in a lightweight manner, bridging the gap to real-world applications. Furthermore, our model surpasses SOTA methods in full-resolution scenes under real-world conditions, highlighting its effectiveness and efficiency. We also extend our method to general image fusion tasks. The source code is available at https://github.com/CZhongnan/Pan-LUT.

Section: Introduction
High-resolution multispectral (HRMS) images are widely used in applications such as military operations, environmental monitoring, and mapping. However, due to the limitations of physical sensors, these images are challenging to obtain. Pan-sharpening addresses this issue by fusing high-resolution panchromatic (PAN) images with low-resolution multispectral (LRMS) images, producing high-quality HRMS images through complementary integration [55] [61] [42]. Recently, numerous pan-sharpening methods have been proposed, which can be generally categorized into two main groups: traditional methods and deep learning-based methods. Traditional methods, such as component substitution (CS) [2] [3] [11], multi-resolution analysis (MRA) [7] [19], and variational optimization (VO) [10] [35], often struggle to restore precise spatial or spectral details in HRMS images. In contrast, deep learning-based pan-sharpening methods have demonstrated exceptional fusion capabilities due to the powerful feature extraction ability of deep neural networks (DNNs) [23] [24] [27] [28]. Masi et al. [33] built the Pan-sharpening Neural Network (PNN) model, which first applied CNN to pan-sharpening field, achieving a significant improvement over traditional methods. Following this, researchers have explored more complicated and deeper networks to further promote the performance of pan-sharpening [13] [25] [34] [56]. However, they overlook a critical practical issue: the need for real-time processing of large remote sensing images in real-world applications. As illustrated in Figure 1, we observe two major limitations in DNN-based approaches: (1) they are highly sensitive to the size of the input images, and (2) they rely heavily on dedicated computing devices such as GPUs and TPUs. Increasing GPU memory does not significantly improve the image size these methods can handle and several of these methods demand a significant amount of time to process images in CPU-only environments. And we will demonstrate in the experiments sections that, even with GPU acceleration, most methods still fail to process large remote sensing images in real time. In practical applications, remote sensing images typically exhibit even higher resolutions, posing additional challenges to existing methods in terms of efficiency and scalability. Moreover, simply increasing network depth does not necessarily lead to better performance, as deeper models are harder to train and often suffer from overfitting due to redundant parameters.
To overcome the aforementioned challenges, we propose a novel learnable Look-Up Table (LUT) framework, called Pan-LUT, which achieves a good balance between performance and computational efficiency in pan-sharpening. Specifically, we replace complex DNN operations with learnable LUTs to enable lightweight deployment in practical applications. To finely control the spectral transformation, we devise the PAN-guided look-up table (PGLUT) for channel-wise spectral mapping. To effectively capture fine-grained spatial details, we introduce the spatial details look-up table (SDLUT). To further enable adaptive channel aggregation for high-resolution multispectral image generation, we design the adaptive output look-up table (AOLUT). The Pan-LUT consists of fewer than 700K parameters and can process 9K×9K images in under 1 ms using a single RTX 2080 Ti GPU. Furthermore, our approach outperforms traditional methods by 7 dB, while maintaining a speed comparable to that of conventional techniques, demonstrating superior speed and efficiency compared to existing methods. Our contributions can be summarized as follows:
• We present Pan-LUT, a novel learnable LUT framework that does not incorporate any network structure. This framework is designed to achieve a strong balance between performance and computational efficiency in pan-sharpening high-resolution remote sensing images. Our method makes it possible to process 15K×15K remote sensing images on one 24GB GPU.
• To finely control the spectral transformation, we devise the PAN-guided look-up table (PGLUT) for channel-wise spectral mapping. To effectively capture fine-grained spatial details and adaptively learn local contexts, we introduce the spatial details look-up table (SDLUT). To further enable adaptive channel aggregation for high-resolution multispectral image generation, we design the adaptive output look-up table (AOLUT).
• To the best of our knowledge, this is the first attempt to introduce LUTs for efficient pan-sharpening. Extensive experiments on different satellite datasets demonstrate the effectiveness and efficiency of Pan-LUT.
2 Related Work
this section cite: ['b54', 'b41', 'b1', 'b10', 'b27', 'b32', 'b55']

Section: Look-Up Table
Look-up Tables (LUT) are particularly useful for functions of multiple variables, as they store precomputed outputs for all possible input combinations. For example, in a 1D LUT, a single input index is mapped to an output value, often using linear interpolation for indices that fall between pre-stored values. More complex LUTs, such as 3D LUTs, use three independent input variables, which may require advanced interpolation methods like trilinear or tetrahedral interpolation. Due to its portability, various LUT based solutions have been proposed for image enhancement [6] [22] [26] [39] [52].
For instance, Zeng et al. [52] and Wang et al. [39] propose image-adaptive 3D LUTs for efficient single-image enhancement. These approaches rely on a network weight predictor to fuse different 3D LUTs, which may pose a limitation on platforms under resource-constrained conditions. Additionally, LUT-based methods have been explored in the area of super-resolution [18] [21] [31] [29]. SRLUT [18] trains a deep super-resolution (SR) network with a restricted receptive field and then caches the output values from the learned SR network in LUTs. However, issues such as performance degradation arise when large patches are cached in LUTs, prompting the development of strategies like MuLUT [21], which introduces multiple LUT variants and a fine-tuning strategy to improve performance. To further enhance the functionality of LUTs, architectures like SPLUT [31] and RCLUT [29] have been proposed. SPLUT processes different image information separately using multiple LUTs, while RCLUT introduces a plugin module to improve LUT-based models with minimal additional computational cost.
this section cite: ['b51', 'b51', 'b38', 'b17', 'b20', 'b30', 'b28']

Section: Traditional Pan-sharpening Methods

this section cite: []

Section: Traditional fusion techniques encompass component substitution (CS), multi-resolution analysis (MRA), and variational optimization (VO).
CS methods, such as IHS [2], Brovey [11], and PCA [3], utilize spatial details from high-resolution panchromatic (PAN) images to replace corresponding details in low-resolution multispectral (LRMS) images, which can lead to spectral distortion due to the incomplete incorporation of spectral information. MRA techniques, including DWT [19] and ATWT [7], apply multi-resolution decomposition to merge PAN and LRMS images, which enables better preservation of spectral information and reduces spectral distortion. VO methods, such as Bayesian [10] and Total Variation [35], formulate the fusion process as an optimization problem by iteratively minimizing the loss function. While VO methods show promising results, they encounter challenges in optimizing model design and loss functions. Although these approaches have yielded certain improvements, their performance remains constrained by the inadequate modeling, which restricts further advancements in pan-sharpening accuracy and quality.
this section cite: ['b1', 'b10', 'b2', 'b18', 'b6', 'b9', 'b34']

Section: Deep Learning-based Methods
Deep learning-based methods have emerged as the dominant approach for pan-sharpening in recent years [40] [14] [43]. The PNN [33] model, inspired by SRCNN [8], is the first to introduce CNNs into this domain, surpassing traditional methods. Models like PanNet [49] and MSDCNN [50] further enhance performance by leveraging residual connections and multi-scale convolutions, effectively capturing high-frequency details and supporting a wide range of remote sensing applications. Since then, more complex CNN-based architectures [4] [15] [62] have been proposed in this field to improve the mapping ability of pan-sharpening. Models like GPPNN [45], MMNet [47], and ARFNet [46], which enhance interpretability through deep unfolding techniques and accelerate model convergence. Spatial adaptive convolution methods, such as LAGConv [17] and CANConv [9], can adaptively generate different convolution kernel parameters based on various spatial locations, enabling them to accommodate different spatial regions. Other approaches, including SFDI [60] and MSDDN [50], utilize Fourier transforms to capture high-frequency features. Transformer-based architectures, such as INN-former [59], Panformer [57] and DRFormer [53] combine CNNs and Transformers to capture both local and global features. Instead of learning a deterministic mapping, generative models such as UCGAN [58], PanFlow [48] and PSCINN [38] generate a distribution of possible outputs for the given inputs. In addition, several studies have investigated lightweight pan-sharpening methods, including SpanConv [5] and LGPConv [54]. However, we show that current lightweight methods are only lightweight in the context of low-resolution images. Despite their promising results, these advanced methods come with high computational costs, limiting their practical applicability.
this section cite: ['b39', 'b42', 'b32', 'b7', 'b48', 'b49', 'b3', 'b61', 'b44', 'b46', 'b45', 'b16', 'b8', 'b59', 'b49', 'b58', 'b56', 'b52', 'b57', 'b47', 'b37', 'b4', 'b53']

Section: Method
Given the PAN image (P ∈ R H×W ×1 ) and the MS image (M S ∈ R H/r×W/r×C ), pan-sharpening aims to fuse the complementary information to generate the desirable high spatial resolution MS image (HRM S ∈ R H×W ×C ). Here, H and W denote the height and width of the images, r represents the spatial resolution ratio, with a value of 4, and C denotes the number of spectral bands.
this section cite: []

Section: Framework
The overall framework of our proposed Pan-LUT is illustrated in Figure 2, which consists of three specifically designed LUTs.
(1) To finely control the spectral transformation, we devise the PANguided look-up table (PGLUT) for channel-wise spectral mapping, which incorporates a PAN-guided indexing strategy and a pentalinear interpolation technique. (2) To effectively capture fine-grained spatial details and adaptively learn local contexts, we introduce the spatial details look-up table (SDLUT), which incorporates a rotation-enhanced indexing strategy and a quadrilinear interpolation technique. (3) To further enable adaptive channel aggregation for high-resolution multispectral image generation, we design the adaptive output look-up table (AOLUT), which incorporates a PAN-guided indexing strategy and a pentalinear interpolation technique. Specifically, given the PAN image (P ∈ R H×W ×1
) and the upsampled MS image (M S ∈ R H×W ×C ), they are concatenated into P M ∈ R H×W ×(C+1) and passed through the PGLUT for channel-wise spectral mapping, producing the output V pg ∈ R H×W ×(C+1) . SDLUT then takes V pg as input to generate local spatial details, yielding V sd ∈ R H×W ×(C+1) . Finally, AOLUT takes V sd as input to generate the final HRM S ∈ R H×W ×C result:
V pg = P GLU T (P M ), V sd = SDLU T (V pg ), HRM S = AOLU T (V sd ).(1)
this section cite: []

Section: Spectral Transformation
To preserve the rich spectral information of the MS image, we propose the PAN-guided look-up table (PGLUT), which leverages the PAN image as guidance to finely control the spectral transformation. Specifically, PGLUT is represented as a 5-dimensional matrix containing N 5 elements, where N denotes the number of bins per dimension. Each element corresponds to a sampling point, defining a set of indexed input pixels {I (i,j,k,m,n) } i,j,k,m,n=0,...,N -1 and their corresponding output pixels {O (i,j,k,m,n) } i,j,k,m,n=0,...,N -1 . Here, I ∈ {pa, r, g, b, nir} represents the pixel values from the PAN and MS images, while O ∈ {R, G, B, N IR} denotes the corresponding cached output pixels. Since the LUT elements are discretely distributed in space, the output value cannot be directly retrieved from the LUT. For an input value {pa I (w,h) , r I (w,h) , g I (w,h) , b I (w,h) , nir I (w,h) }, where (w, h) denotes the spatial position of a pixel in the image.
this section cite: []

Section: PAN-guided indexing strategy.
As illustrated in Figure 2, we introduce an indexing strategy for precise spectral transformation, referred to as the PAN-guided indexing strategy. In a MS image, pixels from different spatial locations may have identical values (e.g., r i = r j , g i = g j , b i = b j , nir i = nir j , where i ̸ = j). The LUT maps these identical inputs to the same output. The PAN-guided indexing strategy provides a more flexible indexing mechanism for the LUT. Specifically, it additionally considers the pixels at corresponding spatial positions in the PAN image (e.g., r i = r j , g i = g j , b i = b j , nir i = nir j , pa i ̸ = pa j , where i ̸ = j), thereby achieving finer-grained mapping. Specifically, PGLUT first performs a lookup operation to locate the corresponding input pixel in the LUT:
x = pa I (w,h) V max • N, y = r I (w,h) V max • N, z = g I (w,h) V max • N, s = b I (w,h) V max • N, e = nir I (w,h) V max • N,(2)
where V max denotes the maximum value (e.g., 255, 1023 or 2047). The coordinates of the sampling points, L = {(i + c, j + c, k + c, m + c, n + c)}, with c ∈ {0, 1}, can be derived as follows:
i = ⌊x⌋, j = ⌊y⌋, k = ⌊z⌋, m = ⌊s⌋, n = ⌊e⌋,(3)
where ⌊•⌋ denotes the floor function. {d l } l=x,y,z,s,e represents the offset of the input index (x, y, z, s, e) relative to the defined sampling point (i, j, k, m, n), e.g., d x = x -i.
this section cite: []

Section: Pentalinear Interpolation.
After locating 32 adjacent points, an appropriate interpolation technique is applied to these sampled values to generate the output:
O (x,y,z,s,e) = P Interpolation(LU T [L], {d l }),(4)
where P Interpolation(•) denotes the pentalinear interpolation. More details about PGLUT and the pentalinear interpolation can be found in Section A.2.
this section cite: []

Section: Spatial Details Transformation
PGLUT is essentially a channel-wise 5D LUT that operates globally, which limits its ability to capture local spatial information. To effectively capture fine-grained spatial details and adaptively learn local contexts, we propose the Spatial Details Lookup Table (SDLUT).
Rotation-indexing strategy. As illustrated in Figure 2, given a pixel p (w,h) , SDLUT processes this pixel along with its neighboring pixels as input. During the training phase, we employ a Rotation-indexing strategy to further expand the receptive field, which can be formulated as: where f SDLU T (•) denotes the lookup and interpolation process in the LUT retrieval. More details about SDLUT and the quadrilinear interpolation can be found in Section A.3.
p 1 (w,h) = f SDLU T (p (w,h) , p (w+1,h) , p (w,h+1) , p (w+1,h+1) ), p 2 (w,h) = f SDLU T (p 1 (w,h) , p 1 (w+1,h) , p 1 (w+1,h-1) , p 1 (w,h-1) ), p 3 (w,h) = f SDLU T (p 2 (w,h) , p 2 (w+1,h) , p 2 (w,h+1) , p 2 (w+1,h+1) ), V (w,h) = f SDLU T (p 3 (w,h) , p 3 (w+1,h) , p 3 (w,h+1) , p 3 (w+1,h+1) ),(5)
this section cite: []

Section: Adaptive Output
For the feature channel pixels from the SDLUT (
V 1 (w,h) , V 2 (w,h) , V 3 (w,h) , V 4 (w,h) , V 5 (w,h) ), AOLUT adaptively aggregates channel information to generate high-resolution multispectral channel pixels {R O (w,h) , G O (w,h) , B O (w,h) , N IR O (w,h) }.
Pixel-level Transformation can be formulated as:
{R O (w,h) , G O (w,h) , B O (w,h) , N IR O (w,h) } = f AOLU T (V 1 (w,h) , V 2 (w,h) , V 3 (w,h) , V 4 (w,h) , V 5 (w,h) ), (6
) where f AOLU T (•) denotes the lookup and interpolation process in the LUT retrieval. More details about AOLUT and the pentalinear interpolation can be found in Section A.4.
this section cite: []

Section: Loss Function
To achieve satisfying pan-sharpening results, we propose a joint loss for network training. Suppose the batch size is T . We first utilize the MSE loss:
L mse = 1 T T t=1 ∥HRM S t -GT t ∥ 2 ,(7)
where HRM S and GT denote the network output and the corresponding ground truth, respectively.
To enhance the stability and robustness of the learned LUTs, we incorporate smoothness regularization L s and monotonicity regularization L m :
L s = L P G s + L SD s + L AO s , L m = L P G m + L SD m + L AO m ,(8)
where L P G s , L SD s , and L AO s denote the smoothness regularizations for PGLUT, SDLUT, and AOLUT, while L P G m , L SD m , and L AO m represent the monotonicity regularizations for PGLUT, SDLUT, and AOLUT, respectively. Taking SDLUT as an example, the smoothness regularization can be defined as:
L SD s = O∈{l,o,c,a} N -1 i,j,k,m=0
( O (i+1,j,k,m) -O (i,j,k,m) 2 + O (i,j+1,k,m) -O (i,j,k,m) 2 + O (i,j,k+1,m) -O (i,j,k,m) 2 + O (i,j,k,m+1) -O (i,j,k,m) 2 ), where N represents the number of bins in each dimension of the LUT. O (i,j,k,m) is the corresponding output for the defined sampling point (i, j, k, m) in LUT. The definitions of Ls P G and Ls AO are similar to those in Equation 9.
The monotonicity regularization in AOLUT can be defined as:
L SD m = O∈{l,o,c,a} N -1 i,j,k,m=0 [g(O (i,j,k,m) -O (i+1,j,k,m) )
+g(O (i,j,k,m) -O (i,j+1,k,m) ) +g(O (i,j,k,m) -O (i,j,k+1,m) ) +g(O (i,j,k,m) -O (i,j,k,m+1) )],
where g(•) denotes the ReLU activation function. Similarly, L P G m and L AO m are defined in the same way as in Equation 10.
The final loss functions are as follows:
L = L 1 + λ s L s + λ m L m ,(11)
where the two constant parameters λ s and λ m are used to control the effects of the smoothness and monotonicity regularization terms, respectively. In our experiments, we empirically set λ s = 0.0001 and λ m = 10. More details about the loss functions can be found in Section A. 5 4 Experiments
this section cite: ['b4']

Section: Datasets
Remote sensing datasets from three satellites are used in our experiments, including WorldView-II (WV2), GaoFen2 (GF2) and WorldView-III (WV3). Due to the absence of high-resolution multispectral ground truth images in these datasets, we generate the training set using the Wald protocol tool [37]. Specifically, given the original MS image and its corresponding high-resolution PAN image, they are downsampled by a factor of r to obtain image pairs of MS and PAN, with r set to 4. During training, the original high-resolution MS image is treated as the ground truth, while the MS and PAN images serve as the input image pairs.
this section cite: ['b36']

Section: Implementation Details
We compare the proposed Pan-LUT model against several pan-sharpening methods on reducedresolution scenes from WV2, WV3, and GF2 datasets. Specifically, we choose four traditional  pan-sharpening techniques: Brovey [11], IHS [2], SFIM [30] and GS [20], along with ten deep learning-based approaches: PNN [33], PanNet [49], MSDCNN [50], Pan-GAN [32], SFDI [60], UCGAN [58], PanFlow [48], PSCINN [38], Pan-Mamba [12] and TA-DiffHQP [41]. Several widely used image quality assessment metrics are employed to evaluate the performance of the algorithm, including peak signal-to-noise ratio (PSNR) [16], structural similarity index (SSIM) [44], spectral angle mapper (SAM) [51], relative dimensionless global error in synthesis (ERGAS) [36], spectral distortion index(D λ ), spatial distortion index (D S ) and the quality with no reference (QNR) [1].
The PyTorch framework is implemented in our experiment. During the training phase, we employ an ADAM optimizer with β 1 = 0.9 and β 2 = 0.999, to update the network parameters for 1000 epochs with a batch size of 1. The learning rate is initialized with 5 × 10 -4 . In parallel, a StepLR learning rate adjustment strategy is employed to reduce the learning rate by half after every 200 iterations. The sizes of PGLUT, SDLUT and AOLUT are set to 9, 9 and 9, respectively.
this section cite: ['b10', 'b1', 'b29', 'b19', 'b32', 'b48', 'b49', 'b31', 'b59', 'b57', 'b47', 'b37', 'b11', 'b40', 'b15', 'b43', 'b50', 'b35', 'b0']

Section: Comparison with Other Methods
Evaluation on Reduced-resolution Scene. The quantitative results across three datasets are presented in Table 1, with the best results highlighted in red. Compared to traditional methods, Pan-LUT achieves an average PSNR improvement of 5dB, 7dB, and 7dB across the three datasets, while maintaining inference speeds comparable to those of traditional methods. It is worth noting that the proposed Pan-LUT does not incorporate any network structure, yet it outperforms some DNN-based methods, such as PanNet and PNN, in terms of both performance and inference time. We also provide visual comparisons for the WV3 datasets, as shown in Figure 3.
this section cite: []

Section: Evaluation on Full-resolution Scene.
To assess the performance and generalization capability of our method on full-resolution scenes under real-world conditions, we first trained Pan-LUT on the reduced-resolution WorldView-II data and then tested it on unseen full-resolution WorldView-II satellite datasets. The real-world dataset consists of 200 newly collected samples from the WorldView-II satellite for evaluation. The results are presented in Table 2. On reduced-resolution scenes, our method falls short of most DNN-based approaches in terms of performance. However, on fullresolution scenes, it outperforms all of them when considering the metrics of D λ , D S , and QNR. This demonstrates its strong generalization ability in real-world situations. Additionally, we provide a visual comparison against both traditional and DNN-based methods, as shown in Figure 4.
this section cite: []

Section: Computation Efficiency Comparison.
We conduct three experiments to comprehensively evaluate the computational efficiency of all methods: (1) testing the maximum image size that each method can handle on 11GB and 24GB GPUs; (2) measuring their inference time on the CPU; and (3) evaluating their inference time on an RTX 2080 Ti GPU with 2K×2K and 4K×4K images. For each method, we record the average inference time on 100 images. As shown in Figure 1, even  with a 24GB GPU, existing methods fail to process 8K×8K images, while our method can handle 9K×9K images on an 11GB GPU. In environments without GPU acceleration, most methods exhibit unsatisfactory inference speed. As shown in Table 1, Pan-LUT efficiently processes images at all resolutions. Compared to DNN-based methods, it achieves significantly faster inference speeds, while maintaining comparable speed to traditional methods. Our method easily meets the realtime processing requirements on GPUs, outperforming all other methods by a substantial margin. Notably, only PNN is capable of handling remote sensing satellite images at the 4K×4K resolution, highlighting the superior efficiency of our approach.
this section cite: []

Section: Ablation Study
Size of Look-Up Tables. As shown in Figure 5, changing the LUT size does not lead to a significant drop in performance. This observation suggests that the effectiveness of our proposed method is not dependent on consuming extensive storage resources to increase the LUT size. First, we examine the effect of PGLUT size, denoted N P . Performance improves with larger N P , reaching an optimal point at values N P = 9. Beyond this (from 9 to 17), only a minor gain of 0.01 dB is observed, while the number of parameters increases substantially from 236K to 5M, indicating capacity redundancy. Therefore, we set N P = 9 as the default to balance performance with storage requirements. For SDLUT, denoted N S , increasing N S from 3 to 9 improves performance, but values above 9 cause a slight performance drop. Similarly, enlarging the AOLUT size N A yields only minor gains but substantially increases parameters, especially beyond N A = 9. We thus set N A = 9 to balance performance with computational efficiency. We provide the parameter count for each LUT of different sizes in Table 3, which can be calculated as follows:
P aram P GLU T = 5N 5 , P aram SDLU T = N 4 , P aram AOLU T = 4N 5 .
Effectiveness of Each LUT. We further conduct ablation studies to verify the effectiveness of each LUT. Results are listed in Table 4. Our observations are as follows: 1) Comparing (i) with (iv) and (iii) with (v), SDLUT effectively captures fine-grained spatial details from the PAN image, thereby enhancing overall performance. 2) Comparing (ii) with (iv) and (iii) with (iv), PGLUT provides finer control over spectral transformation, resulting in improved performance. 3) Comparing (ii) with (v) and (i) with (vi), AOLUT demonstrates adaptive aggregation capabilities.
this section cite: []

Section: Conclusion
In this paper, we propose a novel learnable LUT framework, called Pan-LUT, which strikes an optimal balance between performance and computational efficiency for high-resolution remote sensing images in pan-sharpening. The proposed method makes it possible to process 15K × 15K remote sensing images on a 24GB GPU and processes a 9K×9K image in under 1 ms using one RTX 2080 Ti GPU.
Extensive experiments on various satellite datasets demonstrate the effectiveness and efficiency of Pan-LUT.
Pentalinear Interpolation. After locating 32 adjacent points, an appropriate interpolation technique is applied to generate the output value using the values of these sampled points: O P G(x,y,z,s,e) = d-xd-yd-zd-sd-eO (i,j,k,m,n) + dxd-yd-zd-sd-eO (i+1,j,k,m,n) +d-xdyd-zd-sd-eO (i,j+1,k,m,n) + d-xd-ydzd-sd-eO (i,j,k+1,m,n) +d-xd-yd-zdsd-eO (i,j,k,m+1,n) + d-xd-yd-zd-sdeO (i,j,k,m,n+1) +dxdyd-zd-sd-eO (i+1,j+1,k,m,n) + dxd-ydzd-sd-eO (i+1,j,k+1,m,n) +dxd-yd-zdsd-eO (i+1,j,k,m+1,n) + dxd-yd-zd-sdeO (i+1,j,k,m,n+1) +d-xdydzd-sd-eO (i,j+1,k+1,m,n) + d-xdyd-zdsd-eO (i,j+1,k,m+1,n) +d-xdyd-zd-sdeO (i,j+1,k,m,n+1) + d-xd-ydzdsd-eO (i,j,k+1,m+1,n) +d-xd-ydzd-sdeO (i,j,k+1,m,n+1) + d-xd-yd-zdsdeO (i,j,k,m+1,n+1) +dxdydzd-sd-eO (i+1,j+1,k+1,m,n) + dxdyd-zdsd-eO (i+1,j+1,k,m+1,n) +dxdyd-zd-sdeO (i+1,j+1,k,m,n+1) + dxd-ydzdsd-eO (i+1,j,k+1,m+1,n) +dxd-ydzd-sdeO (i+1,j,k+1,m,n+1) + dxd-yd-zdsdeO (i+1,j,k,m+1,n+1) +d-xdydzdsd-eO (i,j+1,k+1,m+1,n) + d-xdydzd-sdeO (i,j+1,k+1,m,n+1) +d-xdyd-zdsdeO (i,j+1,k,m+1,n+1) + d-xd-ydzdsdeO (i,j,k+1,m+1,n+1) +dxdydzdsd-eO (i+1,j+1,k+1,m+1,n) + dxdydzd-sdeO (i+1,j+1,k+1,m,n+1) +dxdyd-zdsdeO (i+1,j+1,k,m+1,n+1) + dxd-ydzdsdeO (i+1,j,k+1,m+1,n+1) +d-xdydzdsdeO (i,j+1,k+1,m+1,n+1) + dxdydzdsdeO (i+1,j+1,k+1,m+1,n+1) ,
where O (i,j,k,m,n) represents the value of the LUT at the coordinate (i, j, k, m, n).
this section cite: []

Section: References
Ref_id:b0 Title: Multispectral and panchromatic data fusion assessment without reference Year: (2008)
Ref_id:b1 Title: The use of intensity-hue-saturation transformations for merging spot panchromatic and multispectral image data Year: (1990)
Ref_id:b2 Title: Extracting spectral contrast in landsat thematic mapper image data using selective principal component analysis Year: (1989-02)
Ref_id:b3 Title: A novel pansharpening method based on cross stage partial network and transformer Year: (2024)
Ref_id:b4 Title: Spanconv: A new convolution via spanning kernel space for lightweight pansharpening Year: (2022)
Ref_id:b5 Title: Nilut: Conditional neural implicit 3d lookup tables for image enhancement Year: (2024)
Ref_id:b6 Title: Spectral and spatial quality assessment of ihs and wavelet based pan-sharpening techniques for high resolution satellite imagery Year: (2018)
Ref_id:b7 Title: Image super-resolution using deep convolutional networks Year: (2015)
Ref_id:b8 Title: Content-adaptive non-local convolution for remote sensing pansharpening Year: (2024)
Ref_id:b9 Title: Bayesian data fusion for adaptable image pansharpening Year: (2008)
Ref_id:b10 Title: Color enhancement of highly correlated images. ii. channel ratio and "chromaticity" transformation techniques Year: (1987-07)
Ref_id:b11 Title: Pan-mamba: Effective pan-sharpening with state space model Year: (2025)
Ref_id:b12 Title: Pyramid dual domain injection network for pansharpening Year: (2023-10)
Ref_id:b13 Title: Frequency-adaptive pan-sharpening with mixture of experts Year: (2024)
Ref_id:b14 Title: Bidomain modeling paradigm for pansharpening Year: (2023)
Ref_id:b15 Title: Scope of validity of psnr in image/video quality assessment Year: (2008)
Ref_id:b16 Title: Lagconv: Local-context adaptive convolution kernels with global harmonic bias for pansharpening Year: (2022)
Ref_id:b17 Title: Practical single-image super-resolution using look-up table Year: (2021-06)
Ref_id:b18 Title: Indusion: Fusion of multispectral and panchromatic images using the induction scaling technique Year: (2008)
Ref_id:b19 Title: Process for enhancing the spatial resolution of multispectral imagery using pan-sharpening Year: (2000-04)
Ref_id:b20 Title: Cooperating multiple look-up tables for efficient image super-resolution Year: (2022)
Ref_id:b21 Title: Fastllve: Real-time low-light video enhancement with intensity-aware look-up table Year: (2023)
Ref_id:b22 Title: Pair-id: A dual modal framework for identity preserving image generation Year: (2024)
Ref_id:b23 Title: Difftv: Identity-preserved thermal-to-visible face translation via feature alignment and dual-stage conditions Year: (2024)
Ref_id:b24 Title: Domain-irrelevant feature learning for generalizable pan-sharpening Year: (2023)
Ref_id:b25 Title: Unsupervised low-light image enhancement with lookup tables and diffusion priors Year: (2024)
Ref_id:b26 Title: Elevating autonomous driving perception with intelligent image restoration Year: (2025)
Ref_id:b27 Title: Aglldiff: Guiding diffusion models towards unsupervised training-free real-world low-light image enhancement Year: (2024)
Ref_id:b28 Title: Reconstructed convolution module based look-up tables for efficient image super-resolution Year: (2023)
Ref_id:b29 Title: Smoothing filter-based intensity modulation: A spectral preserve image fusion technique for improving spatial details Year: (2000)
Ref_id:b30 Title: Learning series-parallel lookup tables for efficient image superresolution Year: (2022)
Ref_id:b31 Title: Pan-gan: An unsupervised pan-sharpening method for remote sensing image fusion Year: (2020)
Ref_id:b32 Title: Pansharpening by convolutional neural networks Year: (2016)
Ref_id:b33 Title: Progressive high-frequency reconstruction for pan-sharpening with implicit neural representation Year: (2024)
Ref_id:b34 Title: A new pansharpening algorithm based on total variation Year: (2013)
Ref_id:b35 Title: Data fusion: definitions and architectures: fusion of images of different spatial resolutions Year: (2002)
Ref_id:b36 Title: Fusion of satellite images of different spatial resolutions: Assessing the quality of resulting images Year: (1997)
Ref_id:b37 Title: Pan-sharpening via conditional invertible neural network Year: (2024)
Ref_id:b38 Title: Real-time image enhancer via learnable spatial-aware 3d lookup tables Year: (2021)
Ref_id:b39 Title: Cross-modality interaction network for pan-sharpening Year: (2024)
Ref_id:b40 Title: Learning diffusion high-quality priors for pan-sharpening: A two-stage approach with time-aware adapter fine-tuning Year: (2025)
Ref_id:b41 Title: Learning high-frequency feature enhancement and alignment for pan-sharpening Year: (2023)
Ref_id:b42 Title: Towards generalizable pansharpening: Conditional flow-based learning guided by implicit high-frequency priors Year: (2025)
Ref_id:b43 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b44 Title: Deep gradient projection networks for pansharpening Year: (2021)
Ref_id:b45 Title: Panchromatic and multispectral image fusion via alternating reverse filtering network Year: (2022)
Ref_id:b46 Title: Memory-augmented model-driven network for pansharpening Year: (2022)
Ref_id:b47 Title: Panflownet: A flow-based deep network for pan-sharpening Year: (2023)
Ref_id:b48 Title: Pannet: A deep network architecture for pansharpening Year: (2017)
Ref_id:b49 Title: A multiscale and multidepth convolutional neural network for remote sensing imagery pan-sharpening Year: (2018)
Ref_id:b50 Title: Discrimination among semi-arid landscape endmembers using the spectral angle mapper (sam) algorithm Year: (1992)
Ref_id:b51 Title: Learning image-adaptive 3d lookup tables for high performance photo enhancement in real-time Year: (2020-01)
Ref_id:b52 Title: Drformer: Learning disentangled representation for pan-sharpening via mutual information-based transformer Year: (2023)
Ref_id:b53 Title: Lgpconv: Learnable gaussian perturbation convolution for lightweight pansharpening Year: (2023)
Ref_id:b54 Title: Deep adaptive pansharpening via uncertainty-aware image fusion Year: (2023)
Ref_id:b55 Title: Ssdiff: Spatial-spectral integrated diffusion model for remote sensing pansharpening Year: (2025)
Ref_id:b56 Title: Panformer: A transformer based model for pan-sharpening Year: (2022)
Ref_id:b57 Title: Unsupervised cycle-consistent generative adversarial networks for pan sharpening Year: (2022)
Ref_id:b58 Title: Pan-sharpening with customized transformer and invertible neural network Year: (2022)
Ref_id:b59 Title: Spatial-frequency domain information integration for pan-sharpening Year: (2022)
Ref_id:b60 Title: Pan-guided band-aware multi-spectral feature enhancement for pan-sharpening Year: (2023)
Ref_id:b61 Title: Probability-based global cross-modal upsampling for pansharpening Year: (2023)
