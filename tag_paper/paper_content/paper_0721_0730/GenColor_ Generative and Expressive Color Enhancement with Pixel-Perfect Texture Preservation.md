Title: GenColor: Generative and Expressive Color Enhancement with Pixel-Perfect Texture Preservation
Abstract: Supervision from Adobe5K Dataset Unsupervised Learning Generative Methods Difference Map VS Input Human Expert Input Texture Fidelity ✓ ✓ ✓ ✓ ✗ ✓ ✓ Exposure [14] D&R [30] 3D-LUT [37] RSFNet [28] Midjourney GenColor (Ours) Expert C Figure 1: GenColor achieves expressive color enhancement with superior texture preservation. While unsupervised methods (Exposure [14], Distort-and-Recover [30]) lack expressiveness, supervised approaches (3D-LUT [37], RSFNet [28]) are constrained by global adjustments in their training data. Generative methods (e.g., Midjourney), on the other hand, create dramatic transformations but alter textures, compromising authenticity. The difference maps (2 nd row) show GenColor making selective region-specific adjustments similar to Midjourney, but with better results than Expert C (the best retoucher in Adobe5K [2]), which is limited to global filter adjustments.

Section: Introduction
Color enhancement turns ordinary photographs into compelling visual narratives through the finegrained adjustments professional colorists apply to evoke emotion and aesthetic quality. Despite considerable progress, automated algorithms still struggle to match human expertise. As shown in Figure 1, three fundamental challenges impede progress in this domain: (1) achieving fine-grained expressiveness comparable to human colorists; (2) adapting to diverse input conditions involving varied lighting and content; and (3) preserving the textural fidelity of the input images while making color adjustments.
Existing approaches face significant limitations in addressing these challenges. Unsupervised methods like Exposure [14] and Distort-and-Recover [30] preserve texture but lack expressiveness, particularly in difficult lighting conditions. Supervised methods like DeepLPF [25] and RSFNet [28] rely on datasets, such as Adobe FiveK [2] and PPR10K [22] that contain only global adjustments, fail to capture the complex local modifications characteristic of expert retouching. These approaches produce less vibrant results, as visible in Figure 1. Meanwhile, generative methods like Midjourney create dramatic transformations but significantly alter textures, distorting details and introducing unnatural patterns that compromise the authenticity of the original scene. These collective limitations underscore the need for a color enhancement approach that combines expressive adjustments with texture preservation capabilities across diverse real-world conditions.
In this paper, we present GenColor, a novel framework reinterpreting color enhancement as a texturepreserving conditional image generation task. Our key insight is to leverage the expressive generation capabilities of diffusion models while decoupling texture preservation to a specialized transfer network, ensuring fine-grained adjustments without compromising texture fidelity.
For expressive color transformations, we curate ARTISAN-believed to be the largest dataset specifically designed for image enhancement with 1.2 million high-quality photographs-and develop a diffusion-based color generation approach with three distinctive technical advances. A central observation in our work is that diffusion models, traditionally known for generating new content, can be repurposed for color enhancement through targeted conditioning and careful training regimes. First, we reframe color enhancement as conditional generation by leveraging ControlNet [38] conditioned directly on input images, unlocking superior color expressiveness compared to traditional methods that rely on carefully designed filters with limited creative range. Second, we adapt distort-and-recover principles to diffusion models through a self-supervised training strategy with a wider range of color adjustments, enhancing compatibility with challenging lighting conditions and diverse content. Third, we discover that strategic blending of weights from different training stages yields a good balance between creative expressiveness and artifact control. These three technical advances enable highly expressive color generation that adapts to diverse input conditions, achieving sophisticated aesthetic enhancements that traditional methods cannot attain.
Although diffusion models excel at generating aesthetically pleasing colors, their intrinsic iterative denoising process inevitably alters textural elements-a fundamental limitation that can be alleviated but never fully eliminated. To address this limitation, we develop a texture-preserving network that achieves three critical objectives simultaneously: (1) transferring semantically consistent color styles from the diffusion-generated reference images to the original input, (2) restoring artifacts that typically appear as random noise or irregular color strokes in the diffusion-generated color reference image, and (3) performing implicit super-resolution to handle the resolution discrepancy between diffusion models (typically 512×512 pixels) and high-resolution input images. While this task resembles traditional color matching or transfer, a key technical observation guided our approach-the diffusion model's output and the input image are near-duplicates at the texture level, with relevant pixels for color transfer spatially placed in nearby neighborhoods. To leverage these characteristics, we design a self-supervised learning approach with a novel degradation scheme that simulates all three objectives: circular shift to model the nearby spatial misalignment, random stroke degradation to simulate or restore artifacts, and random resolution rescaling to simulate the
1. Color Generation ControlNet (Diffusion) 2. Texture Preservation Color Transfer Network 3. Global Adjustment 5 Image Processing Filters Enhanced Output I e ARTISAN-1M Dataset 1.2M high-quality photos
Target Images Aesthetic photos Color Distortions Random adjustments ControlNet Architecture Based on Stable
this section cite: ['b13', 'b29', 'b24', 'b27', 'b1', 'b21', 'b37']

Section: F
Texture Recovered < l a t e x i t s h a 1 _ b a s e 6 4 = " e o c z v M q Y / z g o s 1 J i a Z q B b 4 d G V P s = " > A A A B + X i c b V B P S 8 M w H E 3 n v z n / V T 1 6 K Q 7 B 0 2 h F p s e h F 7 1 N c G 6 w l Z K m 6 R a W J i X 5 d T D K v o k X D 4 p 4 9 Z t 4 8 9 u Y b j 3 o 5 o O Q x 3 u / H 3 l 5 Y c q Z B t f 9 t i p r 6 x u b W 9 X t 2 s 7 u 3 v 6 B f X j 0 p G W m C O 0 Q y a X q h V h T z g T t A A N O e 6 m i O A k 5 7 Y b j 2 8 L v T q j S T I p H m K b U T
/ B Q s J g R D E Y K b H s Q S h 7 p a W K u / H 4 W Q G D X 3 Y Y 7 h 7 N K v J L U U Y l 2 Y H 8 N I k m y h A o g H G v d 9 9 w U / B w r Y I T T W W 2 Q a Z p i M s Z D 2 j d U 4 I R q P 5 8 n n z l n R o m c W C p z B D h z 9 f d G j h N d h D O T C Y a R X v Y K 8 T + v n 0 F 8 7 e d M p B l Q Q R Y P x R l 3 Q D p F D U 7 E F C X A p 4 Z g o p j J 6 p A R V p i A K a t m S v C W v 7 x K n i 4 a X r P R f L i s t 2 7 K O q r o B J 2 i c + S h K 9 R C d 6 i N O o i g C X p G r + j N y q 0 X 6
9 3 6 W I x W r H L n G P 2 B 9 f k D H T a T / Q = = < / l a t e x i t > I t 3. Global Adjustment Lightweight Filters super-resolution challenge. This design, based on accurate observations of underlying data patterns, achieves superior texture preservation and significantly better color matching performance than state-of-the-art methods.
Extensive experiments demonstrate that GenColor outperforms existing methods in both subjective evaluations and objective metrics for color enhancement and texture-preserving color transfer. As illustrated in Figure 1, our approach outperforms existing methods while maintaining texture fidelity that generative methods sacrifice. This example image showcases GenColor's ability to create beautiful contrast between the golden building tones and clear blue sky, intelligently lightening shadowed areas while preserving architectural details. It demonstrates how our approach achieves more expressive results than Human Expert C-a limitation inherent to the supervised learning methods, which have relied on Adobe5K due to the absence of fine-grained paired datasets, for color enhancement-particularly in scenes with complex lighting and content that require different color treatments for specific elements. Our key contributions include:
1. A diffusion-based color enhancement framework capable of expert-level fine-grained adjustments, supported by a training scheme that enables sophisticated color generation with selective manipulations across varying lighting and content conditions. 2. A texture-preserving color transfer network that maintains pixel-perfect textural fidelity while achieving precise color matching with the diffusion-generated reference images, resulting in outputs that are both visually compelling and faithful to the original content. 3. ARTISAN, a large-scale dataset of 1.2 million high-aesthetic-quality photos specifically curated for color enhancement, enabling robust training of advanced models.
2 Related Works
this section cite: []

Section: Image Color Enhancement Methods
Traditional color enhancement relies on global adjustments using predefined filters or lookup tables (LUTs), which lack expressiveness for fine-grained local adjustments. To address this limitation, color enhancement methods that operate at a fine-grained level have been proposed. Supervised learning approaches such as DeepLPF [25], 3D-LUT [37], and RSFNet [28] learn enhancement mappings from paired datasets such as Adobe FiveK [2] and PPR10K [22] but are constrained by these datasets' focus on global adjustments rather than sophisticated local edits. Recent work like ICELUT [36] improves interpretability while still limited by training data expressiveness. Unsupervised methods, including Exposure [1], EnhanceGAN [4], and Distort-and-Recover [29], overcome paired data limitations but struggle to achieve fine-grained control comparable to professional retouching. Our approach addresses these limitations through diffusion models' generative capabilities combined with texture preservation mechanisms.
this section cite: ['b24', 'b36', 'b27', 'b1', 'b21', 'b35', 'b0', 'b3', 'b28']

Section: Diffusion Models for Image Color Editing
Diffusion models [12] have revolutionized image synthesis, with latent diffusion models [31] improving efficiency through compressed latent space operations. ControlNet [38] enables structure-guided generation through conditional signals, ControlColor [23] successfully applied this approach to image colorization and achieve strong performance in comparison to existing methods [17]. In a related vein, DiffRetouch [8] employs Stable Diffusion for multi-style photo retouching. Nevertheless, its focus remains on user-adjustable global attributes, whereas our work enables expert-level fine-grained color generation with adaptability across diverse lighting and content conditions.
this section cite: ['b11', 'b30', 'b37', 'b22', 'b16', 'b7']

Section: Texture Preservation in Image Enhancement
Diffusion-based methods struggle with texture preservation due to latent space compression (1/8 or 1/16 downsampling). While DiffRetouch [8] mitigates distortion using Affine Bilateral Grid for pixel-wise adjustments, two fundamental limitations remain: it operates in latent space where high-frequency details are already lost during downsampling, and its grid-based approach introduces interpolation errors in complex scenes like text. Several established approaches for texture preservation also face challenges in our setting. Existing approaches include: (1) hint-based color propagation methods (ControlColor [23], UniColor [15]) that preserve lightness but are unsuitable for enhancements requiring luminance adjustments; (2) style transfer methods (CAP-VSTNet [33]) that sacrifice texture fidelity; and (3) color matching/transfer techniques (Color Matcher [10], NeuralPreset [18]) that achieve pixel-perfect texture preservation but aren't optimized for scenarios where relevant color pixels for transfer exist in nearby spatial neighborhoods between input and reference images. (4) color fusion methods [7,6] that only focus on the edge area of the editing region.
Our approach uses a CNN-transformer hybrid operating directly in image space, leveraging the observation that diffusion outputs and inputs are texture-level near-duplicates with correspondences in nearby spatial neighborhoods. This insight enables self-supervised learning with a novel degradation scheme, significantly outperforming existing methods in accuracy and texture preservation.
this section cite: ['b7', 'b22', 'b14', 'b32', 'b9', 'b17', 'b6', 'b5']

Section: ARTISAN-1M Dataset
Deep learning requires large, high-quality datasets with diverse content and styles. Existing datasets fall short -Adobe5K [2] and PPR10K [22] are too small, while LAION-Aesthetics V2 [32] lacks real-world texture details as the majority of its aesthetically high subset consists of non-photographic artworks. We introduce ARTISAN-1M, containing 1.2 million carefully curated photographs from online platforms (primarily Flickr), selected based on visual quality (Q-Align score [34]), content diversity, and color representation. It includes 1.0M daytime and 240.9K nighttime images, with balanced distribution between images with people (617.0K) and without (653.0K).
ARTISAN-1M enables our color generation module to learn complex enhancement patterns. As shown in Figure 3, models trained on the smaller ARTISAN-100K subset exhibit artifacts with 11.9% of images having undesirable strong hue shifts, while the full dataset reduces this to just 1.3%. When strong hue changes occur with the full dataset, changing the generator's seed typically resolves the issue. ARTISAN-1M's unprecedented scale and quality provides a solid foundation for GenColor to learn expressive yet faithful color enhancement from real-world high-quality photography. Due to space limitations, the details of the datasets are discussed in the appendix.
this section cite: ['b1', 'b21', 'b31', 'b33']

Section: Method

this section cite: []

Section: Method Overview
As shown in Figure 2, we present GenColor, a novel framework that addresses the fundamental challenges of image color enhancement through a three-phase approach. Given an input image I, our goal is to produce an aesthetically enhanced output I e while preserving original texture details. First, our color generation module employs a diffusion model conditioned on the input image to create a color reference I r = D(I). This module, trained on our ARTISAN-1M dataset, leverages ControlNet [38] to learn sophisticated color transformations while carefully balancing expressiveness and artifact control through strategic blending of weights from different training stages. To address the inherent texture limitations of diffusion models, our texture preservation module I t = T (I, I r ) transfers color characteristics from the reference while maintaining the structural details of the input image. Finally, to counteract diffusion models' tendency to produce images with reduced contrast and vibrancy [24], a global adjustment module I e = F(I t ) applies five essential filters (brightness, contrast, highlight, shadow, and saturation) to enhance contrast and vibrancy while preserving the sophisticated color transformations.
Our complete enhancement pipeline can be expressed as: I e = F(T (I, D(I))). This formulation effectively combines the expressive power of diffusion models with precise texture preservation, enabling high-quality color enhancement that rivals professional human retouching.
this section cite: ['b37', 'b23']

Section: Color Generation

this section cite: []

Section: ControlNet-based Color Generation
As shown in Figure 4, we formulate color enhancement as conditional generation using the ControlNet architecture [38]. This approach shows superior color expressiveness compared to traditional methods that rely on carefully designed filters with limited creative range. ControlNet adds an auxiliary network branch that conditions on a control signal derived from a color-distorted version of the input image A(I), where A is a random color adjustment function. We train ControlNet on our high-quality ARTISAN-1M dataset, using the original images I as targets and color-distorted versions as conditioning inputs. The distortions range from strong deviations (extreme lighting conditions, restoration/colorization scenarios) to slight deviations (suboptimal color, fine color edits), creating a comprehensive training regime. This approach enables our method to tackle challenging inputs with high contrast lighting, difficult shadows, and many other complex scenarios that traditional methods struggle with. Although each image is adjusted globally, training over our large dataset enables the model to capture fine-grained color enhancement nuances that apply to different image regions contextually. In addition, we have found a weight blending strategy that can improve the performance of the color generation module, please refer to the appendix §B for details. We utilize a null text prompt by default, as textual information provides a negligible contribution to this network, as validated in Appendix §B.3.
this section cite: ['b37']

Section: Texture Preservation Module

this section cite: []

Section: Problem Formulation
The texture preservation network T aims to transfer the color characteristics from the reference image I r generated by the color generation module D while preserving the high-frequency structural details from the input image I. This can be formulated as a conditional generation problem: p(I e |I, I r ) = T (I, U(I r ); θ) where I e is the enhanced output image, U represents the upsampling operation to match the input resolution, and θ denotes the learnable parameters of T .
this section cite: []

Section: Degradation Process for Training Data Generation
In the absence of ground truth pairs, we design a degradation model to generate synthetic training data that mimics the characteristics of our unique problem, including color differences, texture and structure similarities, and artifacts. As shown in Figure 5, the degradation process consists of:
1. A random circular shift operation that maintains similar overall texture and structure between images while avoiding exact pixel-level correspondence. 2. Global color distortion applied randomly across the entire image to simulate the diverse range of color transformations that may occur during enhancement. 3. Localized color deviation strokes added in randomly selected sparse regions to replicate potential artifacts that could emerge during the color generation process. 4. Resolution adjustment through image resizing operations to account for potential differences in resolution between the input and reference images.
this section cite: []

Section: Network Architecture and Training
We propose a Transformer-based network (Figure 5, right panel) with residually connected feature extraction blocks integrating CNN and Swin-Transformer modules for effective local-global feature capture. The network is trained self-supervisedly using the color distorted image as ground truth and the degraded image paired with the original input as training data, optimized with a combination of Huber, perceptual, and adversarial losses. During inference, the degraded image is replaced by the output from the color generation network D. This unified framework effectively addresses color transfer, artifact reduction, and resolution enhancement while preserving the input image's structural details. Please refer to the appendix for more details.
this section cite: []

Section: Experiments
We present only key results here; see appendix for comprehensive analyses.
this section cite: []

Section: Datasets
We evaluate on three datasets: Adobe-FiveK [2] (5,000 raw images with expert retouches), PPR10K [22] (11,161 portrait photos with edits and masks), and FreeRaw (142 high-quality raw photos). Results for Adobe-FiveK and PPR10K are in the main paper, with FreeRaw in the appendix.
this section cite: ['b1', 'b21']

Section: Metrics
We evaluate using three categories of metrics: (1) Color Enhancement metrics including Q-Align [34] (emulates human judgment), NoR-HDR [9] (assesses dynamic range), and LIQE [39] (blind image quality evaluation); (2) Texture Preservation metrics including TD [7] (quantifies texture distortion), DISTS [5] (assesses structural/textural similarity), and GMSD [35] (compares gradient magnitude maps); and (3) Color Similarity metrics including W 1 (measures color distribution similarity), Semantic W 1 (captures fine-grained color characteristics), and MS-SWD [11] (latest metric aligned with human judgment).
this section cite: ['b33', 'b8', 'b38', 'b6', 'b4', 'b34', 'b10']

Section: Baseline Methods
We compare GenColor with state-of-the-art color enhancement methods: Exposure [14], Distort-and-Recover [30], 3D-LUT [37], DeepLPF [25], RSFNet [28], and ICELUT [36]. For texture-preserving color transfer, we compare with ColorMatcher [10], NLUT [37], Deep Preset [13], and a hint color propagation method from Colorization via Imagination [3], originally proposed in UniColor [15]. We exclude style transfer methods like CAP-VSTNet [33] (poor texture preservation) and NeuralPreset [18] (unavailable code). Appendix §I shows Tinge, an app, which uses NeuralPreset's algorithm, performs slightly better than ColorMatcher but significantly worse than GenColor.
this section cite: ['b13', 'b29', 'b36', 'b24', 'b27', 'b35', 'b9', 'b36', 'b12', 'b2', 'b14', 'b32', 'b17']

Section: Image Enhancement Quantitative Results
Table 1 presents a quantitative comparison of color enhancement quality on the Adobe-FiveK and PPR10K datasets. Our GenColor method achieves the best results on most metrics across all datasets, outperforming state-of-the-art methods and approaching or surpassing the performance of a human expert retoucher on the FiveK and PPR10K datasets. GenColor obtains the highest scores on Q-Align and NoR-VDPNet for all datasets, and the best LIQE on FiveK and FreeRaw, with its LIQE on PPR10K being second only to the human expert. Please note that all supervised methods are trained on their respective dataset, while we provide additional results for unsupervised methods trained on our ARTISAN-1M dataset, which indicated by (ARTISAN).
Table 1: Evaluation of GenColor's color-enhancement quality against state-of-the-art methods on various datasets using five perceptual quality metrics: Q-Align [34], LAION [32], LIQE [39], NoR-VDP [9], and C-VAR (↑). Higher values indicate better performance, with best results in bold.
Method Adobe5K PPR10K Aesthetic QualityDyn. RangeExpress. Aesthetic QualityDyn. RangeExpress. Q-Align↑LAION↑ LIQE↑ NoR-VDP↑C-VAR↑Q-Align↑LAION↑ LIQE↑ NoR-VDP↑C-VAR↑ 3D-LUT [37] 4.22 5.73 3.66 67.08 11.48 4.22 6.03 2.83 69.83 8.09 RSFNet [28] 4.23 5.74 3.65 67.85 10.09 4.29 6.05 2.97 69.69 6.68 DeepLPF [25] 4.16 5.79 3.53 69.08 10.97 3.92 6.08 2.55 71.07 11.33 ICELUT [36] 4.17 5.70 3.55 66.40 10.44 3.97 5.95 2.49 68.18 7.21 D&R [30] 2.39 5.33 1.35 64.43 10.29 2.44 5.30 1.21 67.78 5.68 D&R (ARTISAN) 2.31 5.24 1.31 63.92 9.92 2.34 5.23 1.18 67.21 5.41 Exposure [14] 4.00 5.72 3.29 66.02 13.96 3.90 5.96 2.37 70.73 12.44 Exposure (ARTISAN) 3.93 5.66 3.25 65.69 13.48 3.86 5.88 2.29 70.32 12.00 GenColor 4.29 5.83 3.76 69.16 16.96 4.38 6.26 3.21 71.11 13.57
this section cite: ['b33', 'b31', 'b38', 'b8']

Section: Analysis of Human Expert and LLM Preference Evaluation
Table 2 compares GenColor against human expert performance and LLM preferences. GenColor consistently outperforms Human Expert C across all five quantitative metrics. Notably, GenColor achieves a substantial 55.5% improvement in C-VAR, indicating significantly better expressiveness.
The LLM preference evaluation provides additional validation. When ChatGPT o1 evaluated 100 Adobe5K images, GenColor was preferred in 62% of cases compared to Expert C's 38%. This 24 percentage point preference gap demonstrates that GenColor exceeds professional human retouching quality as perceived by advanced AI systems.
this section cite: []

Section: Texture Preservation Quantitative Results
As shown in Table 3, GenColor achieves strong overall performance, with Color Matcher achieving the lowest TD score (0.42) while GenColor obtains competitive texture preservation metrics including tied for best GMSD (0.12) and best DISTS scores (0.13). GenColor excels particularly in color similarity with the lowest Semantic W 1 (3.59), W 1 (3.22), and MS-SWD (0.70) values. The ablation study reveals the importance of key components in GenColor's texture preservation module training, such as the circular shift, which contribute to its balanced performance. When compared to existing methods like Color Matcher [10] and hint color propagation [15,3], GenColor demonstrates significant improvements across most metrics, highlighting its effectiveness in texture-preserving color transfer.
this section cite: ['b9', 'b14', 'b2']

Section: Visual Results

this section cite: []

Section: Figure 6 presents visual comparisons of GenColor with other methods on representative images from the Adobe FiveK dataset.
The qualitative results align with our quantitative findings, showing that GenColor generates more appealing and natural-looking enhancements compared to other methods. For texture preserving color transfer, Figure 7 shows that our method can better preserve the texture while achieving accurate color transfer. More results can be found in the supplementary website.
this section cite: []

Section: Robustness Evaluation
As shown in Figure 8, our method achieves superior performance on challenging cases, where prior methods often struggle. Additionally, Figure 9 evaluates the robustness of our method across different Input D&R [30] Exposure [14] 3D-LUT [37] DeepLPF [26] RSFNet [27] ICELUT [36] Expert C [2] Ours Input Diffusion Ref NLUT [37] ColorMatcher [10] UniColor [15] DeepPreset [13] Ours image resolutions. We have tested on various resolutions and found that GenColor, which incorporates a resizing operation during training data generation, maintains consistently high performance.
this section cite: ['b29', 'b13', 'b36', 'b25', 'b26', 'b35', 'b1', 'b36', 'b9', 'b14', 'b12']

Section: User Study
To evaluate the perceptual quality of GenColor, we conducted a comprehensive user study involving 58 participants with varying levels of expertise in image processing and color enhancement, ranging from amateur to professional. Each participant was presented with 5 test images and their corresponding enhanced versions generated by all baseline methods. Participants rated the color enhancement results on a 5-point Likert scale based on color aesthetics. The results are presented in Figure 10. Statistical analysis of the ratings demonstrates that GenColor is the highest-rated method among users, confirming its ability to generate visually appealing and natural-looking enhancements that align with human perception. In addition, we conducted a pairwise comparison study to further validate the superiority of GenColor, which is presented in the appendix §G.
this section cite: []

Section: Limitations and Future Work
Despite GenColor's overall effectiveness, it exhibits limitations in specific scenarios. Notably, performance is reduced when processing images with extensive regions of pure black or white due to the inherent difficulty of enhancing extreme luminance values. Additionally, the model may occasionally produce unintended hue transformations under extreme illumination or with ambiguously 480P 720P 1080P 1440P 0.6 0.7 0.8 0.9 TD With Resize W/O Resize (a) Texture 480P 720P 1080P1440P 4.28 4.30 4.32 4.34 4.36 4.38 Q-Align With Resize W/O Resize (b) Aesthetic 480P 720P 1080P1440P 0.72 0.74 0.76 0.78 0.80 0.82 0.84 MS-SWD With Resize W/O Resize (c) Color Figure 9: Robustness on Various Resolutions. 0 1 2 3 4 5 GenColor (Ours) ICELUT DeepLPF RSFNet 3D-LUT Exposure D&R User Rating Figure 10: User study results. GenColor receives the highest rating among the compared methods.
colored objects. Furthermore, while trained on the diverse ARTISAN dataset, the current optimization dynamics cause the model to prioritize a dominant aesthetic mode, resulting in nearly deterministic outputs rather than reflecting the full aesthetic variety of the data. Future work will focus on enhancing control and unlocking this diversity. We plan to incorporate hue preservation masks to mitigate undesired color shifts and investigate specialized loss functions as an alternative to inference-time weight blending for balancing fidelity and aesthetics. Crucially, we aim to explore conditional generation and style disentanglement techniques to enable diverse outputs and facilitate data-efficient steering towards specific, user-desired styles using methods like LoRA.
this section cite: []

Section: Conclusion
In this paper, we have proposed GenColor, a novel framework for high-quality image color enhancement. Our method leverages a diffusion model to generate expressive color transformations and employs a dedicated texture preservation network to maintain structural integrity while achieving fine-grained color adjustments. Extensive experiments demonstrate that GenColor outperforms state-of-the-art methods on both color enhancement quality and texture preservation metrics, and it generates visually appealing results that align with human perception. The introduction of the large-scale, high-quality ARTISAN-1M dataset further contributes to the advancement of color enhancement research. We have released the code and dataset to the public to facilitate further research. Please refer to the project page: https://yidong.pro/projects/gencolor.
this section cite: []

Section: References
Ref_id:b0 Title: When color constancy goes wrong: Correcting improperly white-balanced images Year: (2019)
Ref_id:b1 Title: Learning photographic global tonal adjustment with a database of input / output image pairs Year: (2011)
Ref_id:b2 Title: Automatic controllable colorization via imagination Year: (2024-06)
Ref_id:b3 Title: Aesthetic-driven image enhancement by adversarial learning Year: (2018)
Ref_id:b4 Title: Image quality assessment: Unifying structure and texture similarity Year: (2022)
Ref_id:b5 Title: Chromafusionnet (cfnet): Natural fusion of fine-grained color editing Year: (2024)
Ref_id:b6 Title: Movingcolor: Seamless fusion of fine-grained video color enhancement Year: (2024)
Ref_id:b7 Title:  Year: (2024)
Ref_id:b8 Title: Deep bilateral learning for real-time image enhancement Year: (2017-07)
Ref_id:b9 Title: Plenopticam v1.0: A light-field imaging framework Year: (2021)
Ref_id:b10 Title: Multiscale sliced Wasserstein distances as perceptual color difference measures Year: (2024)
Ref_id:b11 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b12 Title: Deep preset: Blending and retouching photos with color style transfer Year: (2021-01)
Ref_id:b13 Title: Exposure: A white-box photo post-processing framework Year: (2018-05)
Ref_id:b14 Title: Unicolor: A unified framework for multi-modal colorization with transformer Year: (2022)
Ref_id:b15 Title: Robust Estimation of a Location Parameter Year: (1964)
Ref_id:b16 Title: Ddcolor: Towards photo-realistic image colorization via dual decoders Year: (2023)
Ref_id:b17 Title: Neural preset for color style transfer Year: (2023-06)
Ref_id:b18 Title: Harmonizer: Learning to perform white-box image and video harmonization Year: ()
Ref_id:b19 Title: Aesthetic subset of LAION-5b Year: (2022)
Ref_id:b20 Title: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation Year: (2022)
Ref_id:b21 Title: Ppr10k: A large-scale portrait photo retouching dataset with human-region mask and group-level consistency Year: (2021)
Ref_id:b22 Title: Control color: Multimodal diffusion-based interactive image colorization Year: (2024)
Ref_id:b23 Title: Common diffusion noise schedules and sample steps are flawed Year: (2024)
Ref_id:b24 Title: Deeplpf: Deep local parametric filters for image enhancement Year: (2020-06)
Ref_id:b25 Title: Deeplpf: Deep local parametric filters for image enhancement Year: (2020)
Ref_id:b26 Title: Rsfnet: Rich feature attention network for image enhancement Year: (2023)
Ref_id:b27 Title: Rsfnet: A whitebox image retouching approach using region-specific color filters Year: (2023-10)
Ref_id:b28 Title: Distort-and-recover: Color enhancement using deep reinforcement learning Year: (2018)
Ref_id:b29 Title: Distort-and-Recover: Color Enhancement Using Deep Reinforcement Learning Year: (2018-06)
Ref_id:b30 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b31 Title: Technical report and blog post Year: (2022)
Ref_id:b32 Title: Cap-vstnet: Content affinity preserved versatile style transfer Year: (2023-06)
Ref_id:b33 Title: Qalign: Teaching LMMs for visual scoring via discrete text-defined levels Year: (2024-07)
Ref_id:b34 Title: Gradient magnitude similarity deviation: A highly efficient perceptual image quality index Year: (2014)
Ref_id:b35 Title: Taming lookup tables for efficient image retouching Year: (2024)
Ref_id:b36 Title: Learning image-adaptive 3d lookup tables for high performance photo enhancement in real-time Year: (2022)
Ref_id:b37 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b38 Title: Blind image quality assessment via vision-language correspondence: A multitask learning perspective Year: (2023)
Ref_id:b39 Title: All-to-key attention for arbitrary style transfer Year: (2023-10)
