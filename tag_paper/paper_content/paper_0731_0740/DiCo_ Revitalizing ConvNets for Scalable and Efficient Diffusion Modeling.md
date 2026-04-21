Title: DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling
Abstract: Diffusion ConvNet achieves superior image quality with high efficiency. We show samples from two of our DiCo-XL models trained on ImageNet at 512×512 and 256×256 resolution.

Section: Introduction
Diffusion models [73,75,33,74,76] have sparked a transformative advancement in generative learning, demonstrating remarkable capabilities in synthesizing highly photorealistic visual content. Their versatility and effectiveness have led to widespread adoption across a broad spectrum of realworld applications, including text-to-image generation [66,69,67], image editing [59,46,10], image restoration [45,3,4], video generation [36,88,7], and 3D content creation [64,87,84].
Early diffusion models (e.g., ADM [14] and Stable Diffusion [67]) primarily employed hybrid U-Net [68] architectures that integrate convolutional layers with self-attention. More recently, Transformers [83] have emerged as a more powerful and scalable backbone [62,6], prompting a shift toward fully Transformer-based designs. As a result, Diffusion Transformers (DiTs) are gradually supplanting traditional U-Nets, as seen in leading diffusion models such as Stable Diffusion 3 [20], FLUX [49], and Sora [9]. However, the quadratic computational complexity of self-attention presents substantial challenges, especially for high-resolution image synthesis. Recent efforts [100,79,25,63,91,2] have explored more efficient alternatives, focusing on linear-complexity RNNlike architectures, such as Mamba [26] and Gated Linear Attention [92]. While these models improve efficiency, their causal design inherently conflicts with the bidirectional nature of visual generation [30,55], limiting their effectiveness. Furthermore, as illustrated in Fig. 3, even with highly optimized CUDA implementations, their runtime advantage over conventional DiTs remains modest in high-resolution settings. This leads us to a key question: Is it possible to design a hardware-efficient diffusion backbone that also preserves strong generative capabilities like DiTs?
To approach this question, we begin by examining the characteristics that underlie the generative power of DiTs. In visual recognition tasks, the success of Vision Transformers [18] is often credited to the self-attention's ability to capture long-range dependencies [42,23,22]. However, in generative tasks, we observe a different dynamic. As depicted in Fig. 4, for both pre-trained class-conditional (DiT-XL/2 [62]) and text-to-image (PixArt-α [12] and FLUX [49]) DiT models, when queried with an anchor token, attention predominantly concentrates on nearby spatial tokens, largely disregarding distant ones. This finding suggests that computing global attention may be redundant for generation, underscoring the significance of local spatial modeling. Unlike recognition tasks, where long-range interactions are critical for global semantic reasoning, generative tasks appear to emphasize finegrained texture and local structural fidelity. These observations reveal the inherently localized nature of attention in DiTs and motivate the pursuit of more efficient architectures.
In this work, we revisit convolutional neural networks (ConvNets) and propose Diffusion ConvNet (DiCo), a simple yet highly efficient convolutional backbone tailored for diffusion models. Compared to self-attention, convolutional operations are more hardware-friendly, offering significant advantages for large-scale and resource-constrained deployment. While substituting self-attention with convolution substantially improves efficiency, it typically results in degraded performance. As illustrated in Fig. 5, this naive replacement introduces pronounced channel redundancy, with many channels remaining inactive during generation. We hypothesize that this stems from the inherently stronger representational capacity of self-attention compared to convolution. To address this, we introduce a compact channel attention (CCA) mechanism, which dynamically activates informative channels with lightweight linear projections. As a channel-wise global modeling approach, CCA enhances the model's representational capacity and feature diversity while maintaining low computational overhead. Unlike modern recognition ConvNets that rely on large, costly kernels [15,28], DiCo adopts a streamlined design based entirely on efficient 1×1 pointwise convolutions and 3×3 depthwise convolutions. Despite its architectural simplicity, DiCo delivers strong generative performance.
As shown in Fig. 2 and Fig. 3, DiCo models outperform recent diffusion models on both the ImageNet 256×256 and 512×512 benchmarks. Notably, our DiCo-XL models achieve impressive FID scores of 2.05 and 2.53 at 256×256 and 512×512 resolution, respectively. In addition to performance gains, Model Variants 0 30 60 90 Small Base Large XLarge Runtime (ms) DiT (ICCV23) DiS (arXiv24) DiG (CVPR25) DiCo (ours) 6.8x faster 2 4.4 2.8 0 35 DiCo-XL (ours) DiS-H/2 DiT-XL/2 DiM-H U-ViT-H/4 U-ViT-L/4 7 .8 x fa s te r, 1 .5 x g a in 70 105 140 Runtime (ms) FID-50K (w/ cfg) 3.6 (a) Runtime Comparison w/ (b) FID vs. Runtime w/ img_size = 512 img_size = 512 Figure 3: (a) Runtime comparison between DiT [62], DiS [25] (with Mamba [26]), DiG [100] (with Gated Linear Attention [92]), and our DiCo at 512×512 resolution. DiCo is 3.3× faster than DiS at the small model scale and 6.8× faster at the XL scale. (b) FID vs. runtime of various methods on ImageNet 512×512. DiCo-XL achieves an FID of 2.53 while maintaining high efficiency.
DiCo models exhibit considerable efficiency advantages over attention-based [83], Mamba-based [26], and linear attention-based [44] diffusion models. Specifically, at 256×256 resolution, DiCo-XL achieves a 26.4% reduction in Gflops and is 2.7× faster than DiT-XL/2 [62]. At 512×512 resolution, DiCo-XL operates 7.8× and 6.7× faster than the Mamba-based DiM-H [79] and DiS-H/2 [25] models, respectively. Our largest model, DiCo-H with 1 billion parameters, further reduces the FID on ImageNet 256×256 to 1.90. In addition, we validate the applicability of DiCo for text-to-image generation on the MS-COCO dataset. These results collectively highlight the strong potential of DiCo in diffusion-based generative modeling.
Overall, the main contributions of this work can be summarized as follows:
• We analyze pre-trained DiT models and reveal significant redundancy and locality within their global attention mechanisms. These findings may inspire researchers to develop more efficient strategies for constructing high-performing diffusion models.
• We propose DiCo, a simple, efficient, and powerful ConvNet backbone for diffusion models.
By incorporating compact channel attention, DiCo significantly improves representational capacity and feature diversity without sacrificing efficiency.
• We conduct extensive experiments on class-conditional ImageNet benchmarks. DiCo outperforms recent diffusion models in both generation quality and speed. Furthermore, the purely convolutional DiCo demonstrates strong potential in text-to-image generation.  0 20 40 60 80 100 120 Channel Index 0.00 0.05 0.10 0.15 0.20 0.25 0.30 Activation Scores (a) DiT [62] Channel activation scores are computed using ReLU followed by global average pooling on the final layer's self-attention or convolution outputs [102]. Directly replacing self-attention in DiT with convolution introduces significant channel redundancy, as most channel activation scores remain at low levels.
this section cite: ['b72', 'b74', 'b32', 'b73', 'b75', 'b65', 'b68', 'b66', 'b58', 'b45', 'b9', 'b44', 'b2', 'b3', 'b35', 'b87', 'b6', 'b63', 'b86', 'b83', 'b13', 'b66', 'b67', 'b82', 'b61', 'b5', 'b19', 'b48', 'b8', 'b99', 'b78', 'b24', 'b62', 'b90', 'b1', 'b25', 'b91', 'b29', 'b54', 'b17', 'b41', 'b22', 'b21', 'b61', 'b11', 'b48', 'b14', 'b27', 'b61', 'b24', 'b25', 'b99', 'b91', 'b82', 'b25', 'b43', 'b61', 'b78', 'b24', 'b101']

Section: Related Work
Architecture of Diffusion Models. Early diffusion models commonly employ U-Net [68] as the foundational architecture [14,34,67]. More recently, a growing body of research has explored Vision Transformers (ViTs) [18] as alternative backbones for diffusion models, yielding remarkable results [62,6,58,96,65,54]. Notably, DiT [62] has demonstrated the excellent performance of transformer-based architectures, achieving SOTA performance on ImageNet generation. However, the quadratic computational complexity inherent in ViTs presents significant challenges in terms of efficiency for long sequence modeling. To mitigate this, recent studies have explored the use of RNN-like architectures with linear complexity, such as Mamba [26] and linear attention [44], as backbones for diffusion models [25,100,79,91,63]. DiS [25] and DiM [79] employ Mamba to reduce computational overhead, while DiG [100] leverages Gated Linear Attention [92] to achieve competitive performance with improved efficiency. In this work, we revisit ConvNets as backbones for diffusion models. We show that, with proper design, pure convolutional architectures can achieve superior generative performance, providing an efficient and powerful alternative to DiTs.
this section cite: ['b67', 'b13', 'b33', 'b66', 'b17', 'b61', 'b5', 'b57', 'b95', 'b64', 'b53', 'b61', 'b25', 'b43', 'b24', 'b99', 'b78', 'b90', 'b62', 'b24', 'b78', 'b99', 'b91']

Section: ConvNet Designs.
Over the past decade, convolutional neural networks (ConvNets) have achieved remarkable success in computer vision [31,41,89,5,19]. Numerous lightweight ConvNets have been developed for real-world deployment [39,71,38,16]. Although Transformers have gradually become the dominant architecture across a wide range of tasks, their substantial computational overhead remains a significant challenge. Many modern ConvNet designs achieve competitive performance while maintaining high efficiency. ConvNeXt [57] explores the modernization of standard ConvNets and achieves superior results compared to transformer-based models. RepLKNet [15] investigates the use of large-kernel convolutions, expanding kernel sizes up to 31×31. UniRepLKNet [17] further generalizes large-kernel ConvNets to domains such as audio, point clouds, and time-series forecasting.
In this work, we explore the potential of pure ConvNets for diffusion-based image generation, and show that simple, efficient ConvNet designs can also deliver excellent performance.
this section cite: ['b30', 'b40', 'b88', 'b4', 'b18', 'b38', 'b70', 'b37', 'b15', 'b56', 'b14', 'b16']

Section: Method

this section cite: []

Section: Preliminaries
Diffusion formulation. We first revisit essential concepts underpinning diffusion models [33,76]. Diffusion models are characterized by a forward noising procedure that progressively injects noise into a data sample x 0 . Specifically, this forward process can be expressed as:
q(x 1:T |x 0 ) = T t=1 q(x t |x t-1 ), q(x t |x 0 ) = N (x t ; √ ᾱt x 0 , (1 -ᾱt )I),(1)
where ᾱt are predefined hyperparameters. The objective of a diffusion model is to learn the reverse process: p θ (x t-1 |x t ) = N (µ θ (x t ), Σ θ (x t )), where neural networks parameterize the mean and covariance of the process. The training involves optimizing a variational lower bound on the loglikelihood of x 0 , which simplifies to:
L(θ) = -p(x 0 |x 1 ) + t D KL (q * (x t-1 |x t , x 0 )||p θ (x t-1 |x t )).(2)
To simplify training, the model's predicted mean µ θ can be reparameterized as a noise predictor ϵ θ . The objective then reduces to a straightforward mean-squared error between the predicted noise and the true noise ϵ t : L simple (θ) = ||ϵ θ (x t ) -ϵ t || 2 2 . Following DiT [62], we train the noise predictor ϵ θ using the simplified loss L simple , while the covariance Σ θ is optimized using the full loss L.
Classifier-free guidance. Classifier-free guidance (CFG) [35] is an effective method to enhance sample quality in conditional diffusion models. It achieves such enhancement by guiding the sampling process toward outputs strongly associated with a given condition c. Specifically, it modifies the predicted noise to obtain high p(x|c) as: εθ (x t , c) = ϵ θ (x t , ∅) + s • ∇ x log p(x|c) ∝ ϵ θ (x t , ∅) + s • (ϵ θ (x t , c) -ϵ θ (x t , ∅)). where s ≥ 1 controls the guidance strength, and ϵ θ (x t , ∅) is an unconditional prediction obtained by randomly omitting the conditioning information during training. Following prior works [62,100], we adopt this technique to enhance the quality of generated samples.
this section cite: ['b32', 'b75', 'b61', 'b34', 'b61', 'b99']

Section: Network Architecture
Currently, diffusion models are primarily categorized into three architectural types: (1) Isotropic architectures without any downsampling layers, as seen in DiT [62]; (2) Isotropic architectures with long skip connections, exemplified by U-ViT [6]; and (3) U-shaped architectures, such as U-DiT [82]. Motivated by the crucial role of multi-scale features in image denoising [97,1], we adopt a U-shaped design to construct a hierarchical model. We also conduct an extensive ablation study to systematically compare the performance of these different architectural choices in Table 4.
As illustrated in Fig. 6 (a), DiCo employs a three-stage U-shaped architecture composed of stacked DiCo blocks. The model takes the spatial representation z generated by the VAE encoder as input. For an image of size 256 × 256 × 3, the corresponding z has dimensions 32 × 32 × 4. To process this input, DiCo applies a 3 × 3 convolution that transforms z into an initial feature map z 0 with D channels. For conditional information-specifically, the timestep t and class label y-we employ a multi-layer perceptron (MLP) and an embedding layer, serving as the timestep and label embedders, respectively. At each block l within DiCo, the feature map z l-1 is passed through the l-th DiCo block to produce the output z l .
Within each stage, skip connections between the encoder and decoder facilitate efficient information flow across intermediate features. After concatenation, a 1 × 1 convolution is applied to reduce the channel dimensionality. To enable multi-scale processing across stages, we utilize pixel-unshuffle operations for downsampling and pixel-shuffle operations for upsampling. Finally, the output feature z L is normalized and passed through a 3 × 3 convolutional head to predict both noise and covariance.
this section cite: ['b61', 'b5', 'b81', 'b96', 'b0']

Section: DiCo Block
Motivation. As shown in Fig. 4, the self-attention computation in DiT models-whether for class-conditional or text-to-image generation-exhibits a distinctly local structure and significant redundancy. This observation motivates us to replace the global self-attention in DiT with more hardware-efficient operations. A natural alternative is convolution, which is well-known for its ability to efficiently model local patterns. We first attempt to substitute self-attention with a combination of 1 × 1 pointwise convolutions and 3 × 3 depthwise convolutions.
However, the direct replacement leads to a degradation in generation performance. As shown in Fig. 5, compared to DiT, many channels in the modified model remain inactive, indicating substantial channel redundancy. We hypothesize that this performance drop stems from the fact that self-attention, being dynamic and content-dependent, provides greater representational power than convolution, which relies on static weights. To address this limitation, we introduce a compact channel attention mechanism to dynamically activate informative channels. We describe the full design in detail below.
this section cite: []

Section: Block designs.
The core design of DiCo is centered around the Conv Module, as shown in Fig. 6  (c). We first apply a 1 × 1 convolution to aggregate pixel-wise cross-channel information, followed by a 3 × 3 depthwise convolution to capture channel-wise spatial context. A GELU activation is employed for non-linear transformation. To further address channel redundancy, we introduce a compact channel attention (CCA) mechanism to activate more informative channels. As illustrated in Fig. 6 (d), CCA first aggregates features via global average pooling (GAP) across the spatial dimensions, then applies a learnable 1 × 1 convolution followed by a sigmoid activation to generate channel-wise attention weights. Generally, the whole process of Conv Module can be described as:
Y = W p2 CCA(GELU(W d W p1 X)), CCA(X) = X ⊙ Sigmoid(W p GAP(X)),(3)
where W p (•) is the 1 × 1 point-wise conv, W d is the depthwise conv, and ⊙ denotes the channel-wise multiplication. As shown in Fig. 5 (c), this simple and efficient design effectively reduces feature redundancy and enhances the representational capacity of the model. To incorporate conditional information from the timestep and label, we follow DiT by adding the input timestep embedding t and label embedding y, and using them to predict the scale parameters α, γ and the shift parameter β.
Modification for text-to-image. We investigate two different approaches for incorporating textual features into DiCo. The first uses the widely adopted cross-attention [12] mechanism, integrated into the DiCo architecture to fuse text and visual features. The second transforms CLIP text embeddings into dynamic depthwise convolution (DWC) kernels. We pad the text embeddings to a length of 81, feed them through a learnable MLP, and reshape the output into a 9 × 9 kernel. This kernel dynamically modulates DiCo's features via depthwise convolution. In this way, we can construct a Algorithm 1 PyTorch code of text conditional depthwise convolution import torch import torch.nn.functional as F def text_conditional_dwconv(x, context): # x: (B, C, H, W) input feature maps # context: (B, 77, C) CLIP text embeddings after an MLP # output: (B, C, H, W) output after depthwise convolution B, C, H, W = x.shape context_pad = torch.cat([context, context[:, -1:].expand(-1, 4, -1)], dim=1) # (B, 81, C) kernels = context_pad.reshape(B, 9, 9, C).permute(0, 3, 1, 2).reshape(B * C, 1, 9, 9) x_flat = x.view(1, B * C, H, W) output = F.conv2d(x_flat, kernels, padding=4, groups=B * C).view(B, C, H, W) return output
fully convolutional text-to-image DiCo model without relying on any self-attention or cross-attention operations. We provide its detailed PyTorch implementation in Algorithm 1. Both feature fusion modules are inserted after the Conv Module within each DiCo block.
this section cite: ['b11']

Section: Architecture Variants
We establish four model variants-DiCo-S, DiCo-B, DiCo-L, and DiCo-XL-whose parameter counts are aligned with those of DiT-S/2, DiT-B/2, DiT-L/2, and DiT-XL/2, respectively. Compared to their DiT counterparts, our DiCo models achieve a significant reduction in computational cost, with Gflops ranging from only 70.1% to 74.6% of those of DiT. Furthermore, to explore the potential of our design, we scale up DiCo to 1 billion parameters, resulting in DiCo-H. The architectural configurations of these models are detailed in Appendix Table 6.
4 Experiments
this section cite: []

Section: Experimental Setup
Datasets and Metrics. Following previous works [62,100,81], we conduct experiments on classconditional ImageNet-1K [13] generation benchmark at 256×256 and 512×512 resolutions. We use the Fréchet Inception Distance (FID) [32] as the primary metric to evaluate model performance. In addition, we report the Inception Score (IS) [70], Precision, and Recall [48] as secondary metrics. All these metrics are computed using OpenAI's TensorFlow evaluation toolkit [14].
Implementation Details. For DiCo-S/B/L/XL, we adopt exactly the same experimental settings as used for DiT. Specifically, we employ a constant learning rate of 1 × 10 -4 , no weight decay, and a batch size of 256. The only data augmentation applied is random horizontal flipping. We maintain an exponential moving average (EMA) of the DiCo weights during training, with a decay rate of 0.9999. The pre-trained VAE [67] is used to extract latent features. For our largest model, DiCo-H, we follow the training settings of U-ViT [6], increasing the learning rate to 2 × 10 -4 and scaling the batch size to 1024 to accelerate training. Additional details are provided in Appendix Sec. B.
this section cite: ['b61', 'b99', 'b80', 'b12', 'b31', 'b69', 'b47', 'b13', 'b66', 'b5']

Section: Main Results
Comparison under the DiT Setting. In addition to DiT [62], we also select recent diffusion models, DiG [100] and DiC [81], as baselines, since they similarly follow the experimental setup of DiT.Table 1 presents the comparison results on ImageNet 256×256. Across different model scales trained for 400K iterations, our DiCo consistently achieves the best or second-best performance across all metrics. Furthermore, when using classifier-free guidance (CFG), our DiCo-XL achieves an FID of 2.05 and an IS of 282.17. Beyond performance improvements, DiCo also demonstrates significant efficiency gains compared to both the baselines and Mamba-based models.
Table 2 presents the results on ImageNet 512×512. At higher resolutions, our model demonstrates greater improvements in both performance and efficiency. Specifically, DiCo-XL achieves an FID of 2.53 and an IS of 275.74, while reducing Gflops by 33.3% and achieving a 3.1× speedup compared to DiT-XL/2. These results highlight that our convolutional architecture remains highly efficient and effective for high-resolution image generation.
this section cite: ['b61', 'b99', 'b80']

Section: Text-to-Image Generation.
We follow [6] to conduct small-scale text-to-image generation experiments. Specifically, we adopt the same experimental setup as [96]: training and evaluating models from scratch on MS-COCO [53], using CLIP as the text encoder with a token length of 77.
As shown in Table 3, our DiCo achieves superior generation quality for text-to-image generation. Notably, using text conditional DWC in place of cross-attention further improves throughput while maintaining competitive performance. This suggests that the fully convolutional DiCo has the potential to serve as the backbone for large-scale text-to-image diffusion models.
this section cite: ['b5', 'b95', 'b52']

Section: Ablation Study
For the ablation study, we use the small-scale model and evaluate performance on the ImageNet 256×256 benchmark to enable fast training speed. All models are trained for 400K iterations and evaluated without CFG. Notably, in this section, self-attention in DiT is not accelerated using FlashAttention-2 to ensure a fair speed comparison with other efficient attention mechanisms.  We analyze both the overall architecture and the contributions of individual components within DiCo to better understand their impact on model performance.
this section cite: []

Section: Architecture Ablation.
We evaluate the performance of DiCo under various architectural designs and conduct a fair comparison with DiT. As shown in Table 4, DiCo consistently outperforms DiT across all structures while also delivering significant efficiency gains. These results highlight the potential of DiCo as a strong and efficient alternative to DiT.
this section cite: []

Section: Component-wise Ablation.
We conduct a component-wise analysis of DiCo, examining the effects of the activation function, convolutional kernel size, compact channel attention (CCA), and the conv module (CM). The overall ablation results are summarized in Table 5. Increasing the convolutional kernel size leads to further perfor-   mance gains but at the expense of reduced efficiency, highlighting a trade-off between performance and computational cost.
The introduction of CCA results in a 4.81-point improvement in FID. As illustrated in Fig. 7, CCA significantly enhances feature diversity, demonstrating its effectiveness in improving the model's representational capacity. We also compare CCA with SE module [40] and Channel-wise Self-Attention [97]; despite its simplicity, CCA achieves superior performance and higher efficiency.
For the Conv Module, we benchmark it against several advanced efficient attention mechanisms (Window Attention [56], Focused Linear Attention [29], Agent Attention [65]). The results show that our CM offers both better performance and computational efficiency.
this section cite: ['b39', 'b96', 'b55', 'b28', 'b64']

Section: Conclusion
We propose a new backbone for diffusion models, Diffusion ConvNet (DiCo), as a compelling alternative to the Diffusion Transformer (DiT). DiCo replaces self-attention with a combination of 1 × 1 pointwise convolutions and 3 × 3 depthwise convolutions, and incorporates a compact channel attention mechanism to reduce channel redundancy and enhance feature diversity. As a fully convolutional network, DiCo surpasses recent diffusion models on the ImageNet 256×256 and 512×512 benchmarks, while achieving significant efficiency gains. Furthermore, the purely convolutional DiCo demonstrates strong potential in text-to-image generation. We look forward to further scaling up DiCo and extending it to broader generative tasks.
this section cite: []

Section: References
Ref_id:b0 Title: Lora-ir: taming low-rank experts for efficient all-in-one image restoration Year: (2024)
Ref_id:b1 Title: Breaking complexity barriers: Highresolution image restoration with rank enhanced linear attention Year: (2025)
Ref_id:b2 Title: Multimodal prompt perceiver: Empower adaptiveness generalizability and fidelity for all-in-one image restoration Year: (2024)
Ref_id:b3 Title: Dreamclear: High-capacity real-world image restoration with privacy-safe dataset curation Year: (2024)
Ref_id:b4 Title: Uncertainty-aware source-free adaptive image super-resolution with wavelet augmentation transformer Year: (2024)
Ref_id:b5 Title: All are worth words: A vit backbone for diffusion models Year: (2023)
Ref_id:b6 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b7 Title: Large scale gan training for high fidelity natural image synthesis Year: (2018)
Ref_id:b8 Title: Video generation models as world simulators Year: (2024)
Ref_id:b9 Title: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing Year: (2023)
Ref_id:b10 Title: Masked generative image transformer Year: (2022)
Ref_id:b11 Title: Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: ()
Ref_id:b12 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b13 Title: Diffusion models beat gans on image synthesis Year: (2009)
Ref_id:b14 Title: Scaling up your kernels to 31x31: Revisiting large kernel design in cnns Year: (2022)
Ref_id:b15 Title: Repvgg: Making vgg-style convnets great again Year: (2021)
Ref_id:b16 Title: Unireplknet: A universal perception large-kernel convnet for audio video point cloud time-series and image recognition Year: (2024)
Ref_id:b17 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b18 Title: Jie Cao, and Ran He. Test-time forgery detection with spatial-frequency prompt learning Year: (2025)
Ref_id:b19 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b20 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b21 Title: Rectifying magnitude neglect in linear attention Year: (2025)
Ref_id:b22 Title: Breaking the low-rank dilemma of linear attention Year: (2025)
Ref_id:b23 Title: Frido: Feature pyramid diffusion for complex scene image synthesis Year: (2023)
Ref_id:b24 Title: Scalable diffusion models with state space backbone Year: (2009)
Ref_id:b25 Title: Linear-time sequence modeling with selective state spaces Year: (2004)
Ref_id:b26 Title: Vector quantized diffusion model for text-to-image synthesis Year: (2022)
Ref_id:b27 Title: Segnext: Rethinking convolutional attention design for semantic segmentation Year: (2022)
Ref_id:b28 Title: Flatten transformer: Vision transformer using focused linear attention Year: (2023)
Ref_id:b29 Title: Demystify mamba in vision: A linear attention perspective Year: (2024)
Ref_id:b30 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b31 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b32 Title: Denoising diffusion probabilistic models Year: (2005)
Ref_id:b33 Title: Cascaded diffusion models for high fidelity image generation Year: (2022)
Ref_id:b34 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b35 Title: Video diffusion models Year: (2022)
Ref_id:b36 Title: simple diffusion: End-to-end diffusion for high resolution images Year: (2023)
Ref_id:b37 Title: Searching for mobilenetv3 Year: (2019)
Ref_id:b38 Title: Mobilenets: Efficient convolutional neural networks for mobile vision applications Year: (2017)
Ref_id:b39 Title: Squeeze-and-excitation networks Year: (2018)
Ref_id:b40 Title: Densely connected convolutional networks Year: (2017)
Ref_id:b41 Title: Ran He, and Tieniu Tan. Vision transformer with super token sampling Year: (2023)
Ref_id:b42 Title: Scaling up gans for text-to-image synthesis Year: (2023)
Ref_id:b43 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b44 Title: Denoising diffusion restoration models Year: (2022)
Ref_id:b45 Title: Imagic: Text-based real image editing with diffusion models Year: (2023)
Ref_id:b46 Title: Understanding diffusion objectives as the elbo with simple data augmentation Year: (2023)
Ref_id:b47 Title: Improved precision and recall metric for assessing generative models Year: (2019)
Ref_id:b48 Title: Flux: Official inference repository for flux.1 models Year: (2004)
Ref_id:b49 Title: Autoregressive image generation using residual quantization Year: (2022)
Ref_id:b50 Title: Return of unconditional generation: A self-supervised representation generation method Year: (2024)
Ref_id:b51 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b52 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b53 Title: Alleviating distortion in image generation via multi-resolution diffusion models and time-dependent layer normalization Year: (2024)
Ref_id:b54 Title: Linfusion: 1 gpu, 1 minute, 16k image Year: (2024)
Ref_id:b55 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b56 Title: A convnet for the 2020s Year: (2022)
Ref_id:b57 Title: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024-04-25)
Ref_id:b58 Title: Sdedit: Guided image synthesis and editing with stochastic differential equations Year: (2021)
Ref_id:b59 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b60 Title: Randar: Decoder-only autoregressive visual generation in random orders Year: (2025)
Ref_id:b61 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b62 Title: DiMSUM: Diffusion mamba -a scalable and unified spatial-frequency method for image generation Year: (2024)
Ref_id:b63 Title: Dreamfusion: Text-to-3d using 2d diffusion Year: (2022)
Ref_id:b64 Title: Efficient diffusion transformer with step-wise dynamic attention mediators Year: (2024)
Ref_id:b65 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b66 Title: High-resolution image synthesis with latent diffusion models Year: (2008)
Ref_id:b67 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b68 Title: Photorealistic text-toimage diffusion models with deep language understanding Year: (2022)
Ref_id:b69 Title: Improved techniques for training gans Year: (2016)
Ref_id:b70 Title: Inverted residuals and linear bottlenecks Year: (2018)
Ref_id:b71 Title: Stylegan-xl: Scaling stylegan to large diverse datasets Year: (2022)
Ref_id:b72 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b73 Title: Denoising diffusion implicit models Year: (2021)
Ref_id:b74 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b75 Title: Score-based generative modeling through stochastic differential equations Year: ()
Ref_id:b76 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b77 Title: Df-gan: A simple and effective baseline for text-to-image synthesis Year: (2022)
Ref_id:b78 Title: Dim: Diffusion mamba for efficient high-resolution image synthesis Year: (2009)
Ref_id:b79 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b80 Title: Dic: Rethinking conv3x3 designs in diffusion models Year: (2025)
Ref_id:b81 Title: U-dits: Downsample tokens in u-shaped diffusion transformers Year: (2024)
Ref_id:b82 Title: Attention is all you need Year: (2017)
Ref_id:b83 Title: Hallo3d: Multi-modal hallucination detection and mitigation for consistent 3d content generation Year: (2024)
Ref_id:b84 Title: Flowdcn: Exploring dcn-like architectures for fast image generation with arbitrary resolution Year: (2024)
Ref_id:b85 Title: Parallelized autoregressive visual generation Year: (2025)
Ref_id:b86 Title: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation Year: (2023)
Ref_id:b87 Title: Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation Year: (2023)
Ref_id:b88 Title: Aggregated residual transformations for deep neural networks Year: (2017)
Ref_id:b89 Title: Attngan: Fine-grained text to image generation with attentional generative adversarial networks Year: (2018)
Ref_id:b90 Title: Diffusion models without attention Year: (2009)
Ref_id:b91 Title: Gated linear attention transformers with hardware-efficient training Year: (2024)
Ref_id:b92 Title: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models Year: (2025)
Ref_id:b93 Title: Vector-quantized image modeling with improved vqgan Year: (2021)
Ref_id:b94 Title: An image is worth 32 tokens for reconstruction and generation Year: (2024)
Ref_id:b95 Title: Representation alignment for generation: Training diffusion transformers is easier than you think Year: (2009)
Ref_id:b96 Title: Restormer: Efficient transformer for high-resolution image restoration Year: (2022)
Ref_id:b97 Title: Cross-modal contrastive learning for text-to-image generation Year: (2021)
Ref_id:b98 Title: Towards language-free training for text-to-image generation Year: (2022)
Ref_id:b99 Title: Dig: Scalable and efficient diffusion models with gated linear attention Year: (2025)
Ref_id:b100 Title: Dm-gan: Dynamic memory generative adversarial networks for text-to-image synthesis Year: (2019)
Ref_id:b101 Title: Discrimination-aware channel pruning for deep neural networks Year: (2018)
