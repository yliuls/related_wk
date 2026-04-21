Title: ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection
Abstract: The scale and quality of datasets are crucial for training robust perception models. However, obtaining large-scale annotated data is both costly and time-consuming. Generative models have emerged as a powerful tool for data augmentation by synthesizing samples that adhere to desired distributions. However, current generative approaches often rely on complex post-processing or extensive fine-tuning on massive datasets to achieve satisfactory results, and they remain prone to content-position mismatches and semantic leakage. To overcome these limitations, we introduce ReCon, a novel augmentation framework that enhances the capacity of structure-controllable generative models for object detection. ReCon integrates region-guided rectification into the diffusion sampling process, using feedback from a pre-trained perception model to rectify misgenerated regions within diffusion sampling process. We further propose region-aligned cross-attention to enforce spatial-semantic alignment between image regions and their textual cues, thereby improving both semantic consistency and overall image fidelity. Extensive experiments demonstrate that ReCon substantially improve the quality and trainability of generated data, achieving consistent performance gains across various datasets, backbone architectures, and data scales. Our code is available at https://github.com/haoweiz23/ReCon.

Section: Introduction
Robust object detection and instance segmentation models are essential in modern computer vision (Bochkovskiy et al., 2020;Carion et al., 2020;Zhu et al., 2020;Liu et al., 2024). However, these models are highly dependent on large-scale, meticulously annotated datasets whose creation is expensive and time consuming (Barkai et al., 1993;Cherti et al., 2023). For instance, annotating a single image in the Cityscapes dataset can take up to 60 minutes (Cordts et al., 2016). Consequently, there is a pressing need for efficient and automated methods to synthesize high-quality annotated training data.
Data augmentation has emerged as a vital strategy to alleviate data scarcity by increasing sample diversity and improving model generalization. Traditional augmentation methods (Zhong et al., 2020;DeVries & Taylor, 2017;Yun et al., 2019;Cubuk et al., 2018;Dvornik et al., 2018) typically introduce only minor local variations, falling short of generating truly novel content. Recent advances in generative modeling, especially structurally controllable frameworks, offer a promising alternative by leveraging Canny edges (Zhang et al., 2023;Zavadski et al., 2024), spatial layouts (Chen et al., 2023;Wang et al., 2024b), or instance masks (Wang et al., 2024a;Wu et al., 2023) to maintain fine-grained control during image synthesis. Xu et al., 2023). However, GANs are often plagued by training instability, mode collapse, and limited controllability, particularly in low-data regimes.
Diffusion models have recently emerged as a robust alternative, offering enhanced controllability and adaptability. These models implement a reverse denoising process that gradually removes noise from an initial Gaussian distribution to approximate the real data distribution (Yang et al., 2023a). Moreover, diffusion models can effectively handle a variety of conditioning inputs, including text, images, layouts, edges, depth maps, points, and masks. This flexibility has enabled their application to a wide range of tasks such as text-to-image synthesis (Podell et al., 2023;Esser et al., 2024), image editing (Meng et al., 2021;Rombach et al., 2022), image inpainting (Lugmayr et al., 2022;Saharia et al., 2022), and data augmentation (Fang et al., 2024;He et al., 2022). For instance, LAMA (Li et al., 2021) proposed a large mask inpainting strategy to enhance image quality, while Taming Transformers (Esser et al., 2020) demonstrated that training in a latent space can outperform more complex baselines. Further innovations include GLIGEN (Li et al., 2023c), which incorporates gated self-attention for improved layout control, and LayoutDiffuse (Cheng et al., 2023), which employs layout-specific attention modules tailored for bounding box guidance. Additionally, methods like GeoDiffusion (Chen et al., 2023) and Instance Diffusion (Wang et al., 2024a) integrate geometry-aware modules to encode spatial features, leading to superior generation outcomes. DetDiffusion (Wang et al., 2024b) introduces a perception-aware loss to effectively bridge the gap between generation and perception.
In this paper, we exploit these advanced, controllable generative models to produce high-quality synthetic data without extra training, with the goal of enhancing downstream detection tasks.
this section cite: ['b3', 'b5', 'b82', 'b40', 'b2', 'b10', 'b12', 'b80', 'b15', 'b71', 'b13', 'b16', 'b73', 'b72', 'b7', 'b64', 'b67', 'b50', 'b18', 'b47', 'b55', 'b42', 'b57', 'b19', 'b27', 'b38', 'b17', 'b9', 'b7']

Section: Generative Data Augmentation.
Recent advancements in generative models (Rombach et al., 2022;Esser et al., 2024;Tian et al., 2025) have paved the way for synthesizing high-fidelity images that introduce novel content beyond the capabilities of traditional augmentation techniques (Cubuk et al., 2020(Cubuk et al., , 2018;;Yun et al., 2019;Chen et al., 2020). This enhanced data diversity is instrumental in improving the training of perceptual networks for tasks such as object detection.
Initial studies employed GANs (Goodfellow et al., 2020) for data augmentation. However, subsequent research has revealed several limitations of GAN-based approaches. For example, training networks like ResNet50 (He et al., 2016) on data synthesized by models such as BigGAN (Brock et al., 2018) often results in suboptimal performance compared to training with real images. Moreover, the inherent instability in GAN training and the difficulty of generating data under complex conditions (Bansal & Grover, 2023;Gowal et al., 2021;Ravuri & Vinyals, 2019) further constrain their effectiveness.
In contrast, diffusion models offer superior controllability and have gained widespread application in data generation. For image classification, methods such as LECF (He et al., 2022) use GLIDE (Nichol et al., 2021) to generate images and subsequently filter out low-confidence samples to enhance zeroshot and few-shot performance. Similarly, SGID (Li et al., 2023a) leverages BLIP (Li et al., 2022b) to ensure semantic consistency in generated outputs. Feng et al. (2023) filters samples based on feature similarity, while techniques like GIF (Zhang et al., 2022) and DistDiff (Zhu et al., 2024) incorporate additional guidance during the sampling process to refine the quality of generated samples. For object detection, recent methods such as GeoDiffusion (Chen et al., 2023) and DetDiffusion (Wang et al., 2024b) have demonstrated the ability to synthesize high-quality images with precise layout control, specifically designed for training detection models. Additional strategies include using diffusion models with post-filtering based on category-calibrated CLIP scores (Fang et al., 2024) and applying background inpainting to augment training data without extra annotations (Li et al., 2025).
Moreover, synthetic data has shown promise in other domains as well. For instance, MagicDrive (Gao et al., 2023) highlights the benefits of synthetic samples for 3D perception tasks, while TrackDiffusion (Li et al., 2023b) focuses on data generation for multi-object tracking. X-Paste (Zhao et al., 2023) and MosaicFusion (Xie et al., 2024) further contribute by producing samples with clear segmentation boundaries to boost instance segmentation performance.
Despite these advances, most current methods either require additional training of generative models or struggle to balance fidelity and diversity. In this work, we develop the generative data augmentation framework for object detection by leveraging emerging zero-shot recognition models (e.g., GroundedSAM (Ren et al., 2024)) alongside versatile conditional generation models (e.g., Stable Diffusion (Rombach et al., 2022) and ControlNet (Zhang et al., 2023)). Our approach eliminates the
Grounded SAM Class Prompt: Person; Snowboard; boat Context Prompt: An image with a person, a snow board and a boat. Text Encoder 𝒗 𝒕 Step1: Prepare text embedding and instance masks Step2: Diffusion sampling with region control 𝒛 𝒕+𝟏 Structural Control Diffusion Model 𝒛 𝒕 Region Aligned Cross Attention Region-Guided Rectification 𝒗 𝒕 𝑹 gt 𝒛 𝒕 ′ 𝒙 𝟎 𝑩 gt
this section cite: ['b55', 'b18', 'b59', 'b14', 'b13', 'b71', 'b8', 'b23', 'b26', 'b4', 'b1', 'b24', 'b52', 'b27', 'b49', 'b20', 'b75', 'b81', 'b7', 'b19', 'b36', 'b21', 'b78', 'b65', 'b54', 'b55', 'b73']

Section: Frozon

this section cite: []

Section: 𝑹 gt
Figure 1: Overview of the ReCon Pipeline. ReCon enhances object-detection data generation by integrating region control into frozen, off-the-shelf models. It first compute the text embedding and instance masks, and then leverages a structural controllable diffusion model as the data generator and introduces region-guided rectification to refine generated results during the sampling process. Additionally, region-aligned cross-attention is incorporated to mitigate semantic leakage. Our method is plug-and-play and can be integrated with existing structure-controllable models.
need to retrain generative models, which is often impractical in data-scarce scenarios, and instead focuses on simplicity and effectiveness in generating task-specific data for target detectors.
3 Method Task Definition. This study enhances a downstream object detector through data augmentation. By leveraging a pre-existing generator with the original image x, bounding boxes B, and class labels y, we aim to generate a high-fidelity augmented dataset where objects appear within the specified B and carry the correct labels y. The primary challenge is to preserve fidelity to the source while introducing useful novel variations (e.g., new colors, styles, or object poses) to increase content diversity and thereby improve downstream model performance.
Overview. Existing methods often face a trade-off between diversity and fidelity in generating downstream data. Recent works improve data diversity by using in-painting techniques to preserve certain image regions while redrawing others (Li et al., 2025;Ma et al., 2024a). Others improve fidelity by using perceptual models like CLIP (Radford et al., 2021) to filter out low-confidence samples (Fang et al., 2024;Zhao et al., 2023). In this paper, we propose a novel approach that utilizes an off-the-shelf perceptual model to adaptively calibrate image content during sampling, achieving a better balance between diversity and fidelity.
As illustrated in Figure 1, our method builds upon existing structural control models (e.g., ControlNet) to establish an initial layout control. During the sampling process, we employ a region-guided rectification strategy that refines instance-level content by automatically filtering out erroneous or low-confidence samples. Additionally, we introduce a region-aligned cross-attention mechanism to facilitate effective interaction between image content and the corresponding textual features.
this section cite: ['b36', 'b51', 'b19', 'b78']

Section: Preliminaries
Stable Diffusion is a generative model that synthesizes high-quality images from textual prompts by operating within a compressed latent space. It comprises forward and reverse diffusion stages.
Forward Process. In the forward process, noise is gradually added to the latent representation z 0 of an image x, turning it into pure Gaussian noise z T after T timesteps. This diffusion process is modeled by:
q(z t | z t-1 ) = N z t ; √ α t z t-1 , (1 -α t ) I ,(1)
where α t controls the balance between the previous latent state and the injected noise.
Denoising Process. Starting with z T (pure Gaussian noise), the model iteratively predicts and removes noise to generate the clean latent z 0 . This reverse sampling process is modeled by:
p θ (z t-1 | z t ) = N z t-1 ; µ θ (z t , t), Σ θ (t) ,(2)
where θ is the denoising network, µ θ (z t , t) and Σ θ (t) denote the predicted mean and covariance. Sequentially applying this reverse process from t = T down to t = 1 effectively removes the noise, recovering z 0 for final image generation.
this section cite: []

Section: Cross-Attention Mechanism.
In text-to-image generation, the text condition v t (encoded by a text encoder like CLIP) is integrated into the latent space via cross-attention:
Attention(Q, K, V) = softmax QK ⊤ √ d k V,(3)
where the query Q is derived from image features, and K and V derived from text embeddings.
Here, d k denotes the key dimension. This mechanism injects semantic text information into the latent features at each denoising step, guiding the image generation to reflect the text prompt.
During sampling, noisy latent representations are progressively denoised while being continuously influenced by the text embeddings. Starting the denoising from different timesteps allows control over the editing intensity, balancing adherence to the original image content. However, since the Stable Diffusion model lacks inherent structural control, additional structure-controllable models are required for generating object detection data.
this section cite: []

Section: Region-Controllable Data Augmentation
Structural Control with ControlNet. ControlNet (Zhang et al., 2023) enhances diffusion models (e.g., Stable Diffusion) by conditioning them on structural cues such as edge, depth, or pose maps. It integrates trainable control layers as follows:
ẑl = z l + γ • ControlBlock(c m , θ c )(4)
where z l are the latent features at layer l, c m is the structural conditioning map, θ c are learnable parameters of control blocks, and γ scales the control signal.
In our work, we use ControlNet with an edge canny map to enforce structural constraints during image generation, and we demonstrate that our approach can generalize to other layout-to-image models for diverse guided generation.
Region-Guided Rectification. Existing generative models often encounter issues such as generating an incorrect number of target objects or unintended ones. These challenges significantly affect the quality of the generated data. To address these problems, we propose a region-guided rectification method aimed at perceiving image content during sampling and applying region adjustments. This approach ensures consistency of the content and the layout.
𝒛 𝒕 Fast Sampling × 𝑵 Dec. 𝒛 𝟎|(𝒕-𝑵) Grounded SAM 𝒛 𝒕 𝐨𝐫𝐢𝐠 DDIM Forward Enc. 𝑴 𝒙 𝟎|(𝒕-𝑵) 𝒙 𝟎 𝒛 𝒕 ′ = 𝑴 ⊙ 𝒛 𝒕 𝒐𝒓𝒊𝒈 + (𝟏 -𝑴) ⊙ 𝒛 𝒕 Eq. (5) Rectification 𝒕 𝑩 gt 𝑹 gt 𝑹 𝒕 𝟏 𝒛 𝒕 𝐢𝐧 Crop 𝑹 𝒕 𝟐 𝑹 𝒕 𝟑 𝑹 𝒕 𝒃 CA CA CA CA 𝒗 𝟎 𝒕 𝒗 𝟏 𝒕 𝒗 𝟐 𝒕 𝒗 𝟑 𝒕 Paste 𝒛 𝒕 𝐨𝐮𝐭 𝐐 𝑲𝑽 𝐐 𝑲𝑽 𝐐 𝑲𝑽 𝐐 𝑲𝑽 𝑹 𝒈𝒕 Figure 3: Pipeline of our Region-Aligned Cross-Attention.
We first crop region-specific features from z in t using predefined regions R gt . For regions belonging to the same category, we perform cross-attention with the corresponding text features. Finally, the interacted region image features are concatenated to produce z out t .
As shown in Figure 2, given an image annotation comprising multiple bounding boxes B and their corresponding labels y, we employ the Grounded-SAM model (Ren et al., 2024) to detect potential objects in the data point z t during the sampling process. We then apply IoU-based matching to identify false positives and false negatives, allowing us to segment regions that are potentially misgenerated.
These out-of-control regions are defined by a binary mask M. The identified regions are then replaced with their corresponding noised versions z orig t sampled from the original image, while the remaining parts of z t are preserved. This region-guided rectification process can be formulated as:
z ′ t = M ⊙ z orig t + (1 -M) ⊙ z t ,(5)
where ⊙ denotes the element-wise multiplication, z ′ t represents the rectified latent data point, and z orig t is the latent point obtained after applying t steps of noise addition to the original image using Equation 1. This method leverages the intrinsic overridability property of diffusion sampling (Levin & Fried, 2023), allowing regions in intermediate images to be replaced with external content drawn from the same distribution. Such rectifications can influence the final generated image without disrupting the overall inference process.
However, directly detecting the intermediate latent point z t with a perception model is impractical due to the challenge of finding a pre-trained detector that provides meaningful guidance when the input is noisy. To address this issue, we leverage recent cache-based diffusion acceleration method (Ma et al., 2024b) to speed up sampling over N steps (with N = 5 by default). Furthermore, we utilize the capability of the diffusion model to predict the noise added to z t-N , enabling the prediction of a clean data point z 0|t-N . This process is formulated in Equation 6.
z 0|t-N = z t-N - √ 1 -α t θ(z t-N , t) √ α t ,(6)
Then we apply region rectification at T r timesteps (T r = 4), corresponding to the early (0.75 T ), middle (0.50 T ), latter (0.25 T ) and final (0.10 T ) stages of the diffusion process. At the early stage, when the overall object layout has begun to emerge, we can correct inaccuracies in the spatial distribution of objects. During the middle stage, as the object starts to take shape, we rectify any incorrect semantic content in the objects. Finally, in the latter and final stages, we refine regions with suboptimal generation quality. More details of the sampling process are presented in Algorithm 1.
this section cite: ['b73', 'b54', 'b31']

Section: Region-Aligned Cross Attention.
In text-to-image generation, semantic leakage often occurs, where the content of target regions does not align with the actual textual descriptions. To address this issue, we introduce region-aligned cross attention to mitigate information leakage across regions.
Since attention within the text encoder operates on all prompt tokens, and these tokens may belong to different categories, interference between category-specific features can occur (see the appendix Figure 8). To address this, we individually encode C textual features for C target categories using prompts in the format: [CLASS], as shown in Figure 1. Additionally, we employ a global context description to represent the overall scene, which interacts with the background region. For datasets like COCO, we can directly utilize the provided caption annotations or simply use a custom prompt, such as: "An image with two cars and three persons".
Next, as presented in Figure 3, we perform cross-attention interactions between the corresponding object regions and their associated textual features. This step alleviates information leakage in prompt descriptions by ensuring that region-specific textual features influence only their respective regions.
An alternative approach to implementing region-aware cross-attention is to use an cross attention mask to suppress text features from unrelated categories, as proposed in (Xue et al., 2023). However, we have observed that, due to the lack of disentanglement during the encoding of textual features from different categories, the masked attention mechanism still suffers from semantic leakage. Besides, Instance Diffusion introduces an instance-masked attention and fusion mechanism to integrate regionspecific conditions with corresponding visual tokens. However, it relies on additional region-specific modules and requires retraining to achieve satisfactory performance. In contrast, our approach mitigates the problem of semantic leakage and enhances the fidelity of generated images without the need for additional fine-tuning. Furthermore, we demonstrate that our method can be effectively combined with Instance Diffusion to further boost its performance, as shown in Table 1 and Figure 8.
this section cite: ['b68']

Section: Experiments

this section cite: []

Section: Experiment Settings
We evaluate our method by augmenting downstream object detectors with synthetic samples. We use Stable Diffusion v1.5 (Rombach et al., 2022) with a 25-step DDIM sampler (Song et al., 2020) and edge-conditioned ControlNet (Zhang et al., 2023) to generate training images. These samples are combined with the original trainset and used to jointly train object detectors. We implement training and evaluation code based on the MMDetection framework (Chen et al., 2019). For consistency with prior work (Wang et al., 2024b), our default detector is Faster R-CNN (Ren et al., 2015) with an R-50-FPN backbone trained for six epochs. We also evaluate our method with diverse detectors including RetinaNet (Lin et al., 2017), ATSS (Zhang et al., 2020), FCOS (Tian et al., 2019), YOLO-X (Ge et al., 2021), and DEIM (Huang et al., 2024). Following previous works, we select images containing 3 to 8 objects for data generation, resulting data set comprising 47,200 images with 227,406 objects. For VOC benchmark, we combine the training sets of VOC 2007 and VOC 2012 for model training, with evaluation performed on the VOC 2007 test set (4,952 images). We use mAP (mean Average Precision), mAR (mean Average Recall), FID to evaluate the performance. Extensive experiments are conducted across various datasets, backbone architectures, and data scales.
this section cite: ['b55', 'b58', 'b73', 'b6', 'b53', 'b39', 'b74', 'b60', 'b22', 'b28']

Section: Main Results

this section cite: []

Section: Compared with State-of-the-art Methods.
We compare our approach with state-of-the-art structurecontrollable generative diffusion models, as summarized in Table 1. The results demonstrate that our method significantly enhances the effectiveness of structure-guided techniques for object detection data augmentation. Specifically, we evaluate both general-purpose control methods (e.g., ControlNet) and models fine-tuned on COCO (e.g., DetDiffusion). When combined with these methods, our approach further improves their performance and establishes a new state of the art. For instance, integrating ReCon with ControlNet yields a mAP of 35.5, surpassing GeoDiffusion's 34.8. Moreover, our method can act as a plug-and-play enhancement for region-controlled diffusion models without requiring additional training. This is exemplified by the improvement in GLIGEN's mAP from 34.6 to 35.5. These findings validate that our approach enables the generation of higher-quality training samples, resulting in a substantial boost in object detection performance.
Table 1: Comparison with existing generative models on the COCO dataset. ReCon enhances the detector performance by integrating existing methods in a training-free manner. The best results are highlighted in bold, while the second-best outcomes are denoted by underlined italic.
this section cite: []

Section: Method mAP AP50 AP75 AP m AP l
Real only 34.5 55.5 37.1 37.9 44.3 ▷ General Control Layout Diffusion (Zheng et al., 2023) [CVPR23] 34.0 54.5 36.5 37.2 43.6 ControlNet (Zhang et al., 2023) [ICCV23] 34.9 55.5 37.7 38.2 45.5 Background-inpainting (Li et al., 2025) [ECCV24] 35.1 55.1 37.7 38.2 45.8 ControlNet-XS (Zavadski et al., 2024) [ECCV24] 35.1 55.8 37.6 38.6 45.0 ▷ Fine tuned on COCO ReCo (Yang et al., 2023b) [CVPR23] 33.6 53.2 36.2 36.7 44.0 GLIGEN (Li et al., 2023c) [CVPR23] 34.6 55.1 37.2 38.1 44.7 GeoDiffusion (Chen et al., 2023) [ICLR24] 34.8 55.3 37.4 38.2 45.4 DetDiffusion (Wang et al., 2024b) [CVPR24] 35.4 55.8 38.3 38.5 46.6 Instance Diffusion (Wang et al., 2024a) [CVPR24] 35.0 55.4 37.6 38.4 45.7 ControlNet + ReCon 35.5 56.2 38.4 39.0 46.0 GLIGEN + ReCon 35.3 56.0 38.1 38.7 45.8 Instance Diffusion + ReCon 35.6 56.0 38.4 39.0 46.4
Data-Scarce Scenarios. Data augmentation is crucial when training data is limited. To evaluate our approach under such conditions, we conduct experiments in three data-scarce regimes by randomly sampling 1%, 5%, and 10% of the COCO training set and then doubling each subset through augmentation. Our method delivers consistent gains over baseline approaches in all regimes. As shown in Table 2, with only 10% of the data, mAP rises from 18.5% to 21.7%. Training-based generative models often struggle in data-scarce settings due to their dependence on large datasets. In contrast, we employ a generic structure-controlled diffusion model (ControlNet) to produce highquality object detection samples. We further compare our method to traditional augmentation method RandAugment (Cubuk et al., 2020). Although RandAugment shows noticeable improvement, it remains inferior to our approach. Moreover, combining our method with RandAugment produces additional improvements, demonstrating compatibility with standard augmentation pipelines.
Few-Shot Scenarios. We also evaluated our method in a 30-shot training setting on YOLOX-S (Ge et al., 2021) using COCO dataset, following the few-shot split protocol of previous work (Wang et al., 2020). Our method performs well even under the few-shot setting, increasing the mAP from 5.4 to 6.7 and AP 50 from 10.3 to 12.3. More few-shot results are presented in Table 11 in Appendix. Comparison on Different Dataset. To validate the generalization capability of our method, we conducted additional experiments on PASCAL VOC datasets. We compared our approach against a baseline Faster R-CNN detector trained with 1× schedule. As demonstrated in Table 3, traditional data augmentation methods like RandAugment (Cubuk et al., 2020) show limited effectiveness, while simply duplicate the original dataset leads to overfitting (76.2 vs. 77.1 mAP). Our method achieves superior performance (78.5 mAP) through synthetic generation of diverse high-fidelity images that maintain crucial semantic features.
this section cite: ['b79', 'b73', 'b36', 'b72', 'b7', 'b14', 'b22', 'b61', 'b14']

Section: Data Scaling.
To assess scalability we measure detection accuracy in low-data regimes (5% and 10%), summarized in Figure 4. Repeating training data (real expansion) provides consistent improvements up to 3×: mAP increases from 13.0 to 17.1 on the 5% subset and from 18.5 to 21.1 on the 10% subset. However, further duplication (5×, 7×) leads to saturated performance and noticeable degradation, indicating overfitting under extended training. By contrast, our method generates diverse, annotationconsistent samples and continues to yield performance gains without overfitting. As the expansion scale grows, the relative accuracy improvements from our augmented data become more pronounced. Overall, our approach attains comparable or better performance than the ControlNet baseline using substantially fewer augmented examples, highlighting its efficiency for data augmentation.
this section cite: []

Section: Ablation Studies
Effectiveness of Each Component. We conducted ablation experiments to assess the contributions of each component in our proposed framework, as presented in Table 4. The results clearly indicate that each component enhances the performance of the downstream model. In particular, the integration of region-guided rectification (RGR) and region-aligned cross attention (RACA) significantly improves the consistency between the generated samples and their corresponding annotations, thereby elevating the quality of the synthesized data. Consequently, our approach increases the baseline mAP from 34.9 to 35.5 and improves the FID from 13.82 to 12.85, demonstrating enhanced trainability and fidelity of the generated samples. Perception Target. Our method leverages cache-based fast sampling (Ma et al., 2024b) to recover a clean x 0 , providing a more accurate control signal for region-guided rectification. we compare different perception targets: x t , x 0|t , and x 0|(t-N ) . As shown in Table 5. While x t yields modest gains due to low recall which in turn causes the model to favor a lower overall editing strength. In contrast, employing x 0|t further enhances performance, and the best results are achieved when using x 0|(t-N ) obtained via the fast sampling method.
this section cite: []

Section: Different Detection Backbone.
We evaluate multiple object detectors and report results on the state-of-the-art DEIM (CVPR25) method in Table 6. Additional detector comparisons are provided in Table 10 in the Appendix. Our experiments show that our method consistently improve performance across different detectors, demonstrating its robustness and effectiveness.
Original Ground Truth ControlNet GLIGEN Inst. Diffusion GLIGEN -ReCon ControlNet -ReCon Perception Model. Our approach employs the Grounded-SAM (Ren et al., 2024) as region-guided model to provide region-aware guidance. Additionally, we compare different backbone models for the detector within Grounded-SAM, as detailed in Table 7. The experimental results indicate that better perception leads to improved performance, suggesting that our method stands to benefit from stronger foundation models.
this section cite: ['b54']

Section: Qualitative Results
Figure 5 shows that our method substantially improves both the fidelity and the localization accuracy of generated samples. Unlike prior structure-control methods such as GLIGEN and ControlNet, which lack mechanisms for fine-grained region rectification and hence can exhibit imprecise localization and semantic leakage, our approach enforces strict consistency with the provided annotations. For example, it removes an extraneous zebra produced by GLIGEN outside the target bounding box (row 1) and a superfluous sheep outside the region of interest (row 3), and it correctly restores a person that ControlNet fails to generate (row 2). By aligning generated content with the original annotations, our method improves overall generation quality while maintaining high fidelity and sample diversity. More visualization results are provided in the Appendix.
this section cite: []

Section: Conclusion
This paper presents Region-Controllable data augmentation (ReCon), a training-free, diffusion-based method developed to generate high-quality, content-label-aligned synthetic data for enhancing object detection models. Extensive experimental evaluations demonstrate that ReCon outperforms traditional augmentation and generative methods, ultimately leading to superior detection performance.
this section cite: []

Section: References
Ref_id:b0 Title: Data augmentation generative adversarial networks Year: (2017)
Ref_id:b1 Title: Leaving reality to imagination: Robust classification via generated datasets Year: (2023)
Ref_id:b2 Title: Scaling laws in learning of classification tasks Year: (1993)
Ref_id:b3 Title: Optimal speed and accuracy of object detection Year: (2020)
Ref_id:b4 Title: Large scale gan training for high fidelity natural image synthesis Year: (2018)
Ref_id:b5 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b6 Title: MMDetection: Open mmlab detection toolbox and benchmark Year: (2019)
Ref_id:b7 Title: Geodiffusion: Text-prompted geometric control for object detection data generation Year: (2023)
Ref_id:b8 Title: Gridmask data augmentation Year: (2020)
Ref_id:b9 Title: Layoutdiffuse: Adapting foundational diffusion models for layout-to-image generation Year: (2023)
Ref_id:b10 Title: Reproducible scaling laws for contrastive language-image learning Year: (2023-06)
Ref_id:b11 Title: Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models Year: (2023)
Ref_id:b12 Title: The cityscapes dataset for semantic urban scene understanding Year: (2016)
Ref_id:b13 Title: Learning augmentation policies from data Year: (2018)
Ref_id:b14 Title: Randaugment: Practical automated data augmentation with a reduced search space Year: (2020)
Ref_id:b15 Title: Improved regularization of convolutional neural networks with cutout Year: (2017)
Ref_id:b16 Title: Modeling visual context is key to augmenting object detection datasets Year: (2018)
Ref_id:b17 Title: Taming transformers for high-resolution image synthesis Year: (2020)
Ref_id:b18 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b19 Title: Data augmentation for object detection via controllable diffusion models Year: (2024)
Ref_id:b20 Title: Diverse data augmentation with diffusions for effective test-time prompt tuning Year: (2023)
Ref_id:b21 Title: Magicdrive: Street view generation with diverse 3d geometry control Year: (2023)
Ref_id:b22 Title: Exceeding yolo series in 2021 Year: (2021)
Ref_id:b23 Title: Generative adversarial networks Year: (2020)
Ref_id:b24 Title: Improving robustness using generated data Year: (2021)
Ref_id:b25 Title: Deligan: Generative adversarial networks for diverse and limited data Year: (2017)
Ref_id:b26 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b27 Title: Is synthetic data from generative models ready for image recognition Year: (2022)
Ref_id:b28 Title: Detr with improved matching for fast convergence Year: (2024)
Ref_id:b29 Title: Dataset enhancement with instance-level augmentations Year: (2024)
Ref_id:b30 Title: xformers: A modular and hackable transformer modelling library Year: (2022)
Ref_id:b31 Title: Differential diffusion: Giving each pixel its strength Year: (2023)
Ref_id:b32 Title: Semantic-guided image augmentation with pre-trained models Year: (2023)
Ref_id:b33 Title: Bigdatasetgan: Synthesizing imagenet with pixel-wise annotations Year: (2022)
Ref_id:b34 Title: Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation Year: (2022)
Ref_id:b35 Title: Trackdiffusion: Multi-object tracking data generation via diffusion models Year: (2023)
Ref_id:b36 Title: A simple background augmentation method for object detection with diffusion model Year: (2025)
Ref_id:b37 Title: Gligen: Open-set grounded text-to-image generation Year: (2023)
Ref_id:b38 Title: Image synthesis from layout with locality-aware mask adaption Year: (2021)
Ref_id:b39 Title: Focal loss for dense object detection Year: (2017)
Ref_id:b40 Title: Grounding dino: Marrying dino with grounded pre-training for open-set object detection Year: (2024)
Ref_id:b41 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b42 Title: Repaint: Inpainting using denoising diffusion probabilistic models Year: (2022)
Ref_id:b43 Title: Latent consistency models: Synthesizing high-resolution images with few-step inference Year: (2020)
Ref_id:b44 Title: Erase, then redraw: A novel data augmentation approach for free space detection using diffusion model Year: (2024)
Ref_id:b45 Title: Deepcache: Accelerating diffusion models for free Year: (2024)
Ref_id:b46 Title: Data augmentation with balancing gan Year: (2018)
Ref_id:b47 Title: Sdedit: Guided image synthesis and editing with stochastic differential equations Year: (2021)
Ref_id:b48 Title: Social biases through the text-to-image generation lens Year: (2023)
Ref_id:b49 Title: Glide: Towards photorealistic image generation and editing with text-guided diffusion models Year: (2021)
Ref_id:b50 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b51 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b52 Title: Classification accuracy score for conditional generative models Year: (2019)
Ref_id:b53 Title: Faster r-cnn: Towards real-time object detection with region proposal networks Year: (2015)
Ref_id:b54 Title: Grounded sam: Assembling open-world models for diverse visual tasks Year: (2024)
Ref_id:b55 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b56 Title: Measuring social biases in grounded vision and language embeddings Year: (2020)
Ref_id:b57 Title: Palette: Image-to-image diffusion models Year: (2022)
Ref_id:b58 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b59 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2025)
Ref_id:b60 Title: Fcos: Fully convolutional one-stage object detection Year: (2019)
Ref_id:b61 Title: Frustratingly simple few-shot object detection Year: (2020)
Ref_id:b62 Title: Instancediffusion: Instance-level control for image generation Year: (2024)
Ref_id:b63 Title: Detdiffusion: Synergizing generative and perceptive models for enhanced data generation and perception Year: (2024)
Ref_id:b64 Title: Diffumask: Synthesizing images with pixel-level annotations for semantic segmentation using diffusion models Year: (2023)
Ref_id:b65 Title: Yew Soon Ong, and Chen Change Loy. Mosaicfusion: Diffusion models as data augmenters for large vocabulary instance segmentation Year: (2024)
Ref_id:b66 Title: Efficientsam: Leveraged masked image pretraining for efficient segment anything Year: (2024)
Ref_id:b67 Title: Handsoff: Labeled dataset generation with no additional human annotations Year: (2023)
Ref_id:b68 Title: Freestyle layout-to-image synthesis Year: (2023)
Ref_id:b69 Title: Diffusion models: A comprehensive survey of methods and applications Year: (2023)
Ref_id:b70 Title: Reco: Region-controlled text-to-image generation Year: (2023)
Ref_id:b71 Title: Cutmix: Regularization strategy to train strong classifiers with localizable features Year: (2019)
Ref_id:b72 Title: Controlnet-xs: Rethinking the control of text-to-image diffusion models as feedback-control systems Year: (2024)
Ref_id:b73 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b74 Title: Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection Year: (2020)
Ref_id:b75 Title: Expanding small-scale datasets with guided imagination Year: (2022)
Ref_id:b76 Title: Datasetgan: Efficient labeled data factory with minimal human effort Year: (2021)
Ref_id:b77 Title: Synthesizing informative training samples with gan Year: (2022)
Ref_id:b78 Title: X-paste: Revisiting scalable copy-paste for instance segmentation using clip and stablediffusion Year: (2023)
Ref_id:b79 Title: Layoutdiffusion: Controllable diffusion model for layout-to-image generation Year: (2023)
Ref_id:b80 Title: Random erasing data augmentation Year: (2020)
Ref_id:b81 Title: Distribution-aware data expansion with diffusion models Year: (2024)
Ref_id:b82 Title: Deformable detr: Deformable transformers for end-to-end object detection Year: (2020)
