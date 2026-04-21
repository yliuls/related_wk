Title: ControlFusion: A Controllable Image Fusion Network with Language-Vision Degradation Prompts
Abstract: Current image fusion methods struggle with real-world composite degradations and lack the flexibility to accommodate user-specific needs. To address this, we propose ControlFusion, a controllable fusion network guided by language-vision prompts that adaptively mitigates composite degradations. On the one hand, we construct a degraded imaging model based on physical mechanisms, such as the Retinex theory and atmospheric scattering principle, to simulate composite degradations and provide a data foundation for addressing realistic degradations. On the other hand, we devise a prompt-modulated restoration and fusion network that dynamically enhances features according to degradation prompts, enabling adaptability to varying degradation levels. To support user-specific preferences in visual quality, a text encoder is incorporated to embed user-defined degradation types and levels as degradation prompts. Moreover, a spatial-frequency collaborative visual adapter is designed to autonomously perceive degradations from source images, thereby reducing complete reliance on user instructions. Extensive experiments demonstrate that Control-Fusion outperforms SOTA fusion methods in fusion quality and degradation handling, particularly under real-world and compound degradations. The source code is publicly available at https://github.com/Linfeng-Tang/ControlFusion.

Section: Introduction
Image fusion is a crucial technique in image processing. By effectively leveraging complementary information from multiple sources, the limitations associated with information collection by singlemodal sensors can be significantly mitigated [41]. Among the research areas in this field, infraredvisible image fusion (IVIF) has garnered considerable attention. IVIF integrates essential thermal information from infrared (IR) images with the intricate texture details from visible (VI) images, offering a richer and more complete depiction of the scene [46]. By harmonizing diverse information and producing visually striking results, IVIF has found extensive applications in areas such as security surveillance [47], military detection [22], scene understanding [44], and assisted driving [1], etc.
Recently, IVIF has emerged as a focal point of research, leading to rapid advancements in related methods. Based on the adopted network architectures, these methods can be categorized into convolutional neural network-based [29,51], autoencoder-based [10,11], generative adversarial network-based [19,15], Transformer-based [20,45], and diffusion model-based [50,32] methods. In addition, from a functional perspective, they can be grouped into visual-oriented [29,51], joint registration-fusion [27,37], semantic-driven [28,16], and degradation-robust [31,42,43] schemes. Notably, although degradation-robust schemes can mitigate degradation to some extent, certain limitations exist. First, existing restoration-fusion methods often adopt simplistic strategies for training data construction, overlooking the domain gap between simulated data and realistic images, which hampers their generalizability in practical scenarios, as illustrated in Fig. 1 (I). Second, they are tailored for specific or single types of degradation, making them ineffective in handling more complex composite degradations, as illustrated in Fig. 1 (II). Finally, as shown in Fig. 1 (III), existing methods lack degradation level modeling, causing a sharp decline in performance as degradation intensifies. Moreover, they lack flexibility to adapt fusion results to diverse user preferences.
To overcome these limitations, we propose a versatile and controllable fusion model based on language-vision prompts, termed ControlFusion. On the one hand, we introduced a physics-driven degradation imaging model that differs from existing approaches by simultaneously simulating degradation processes for both visible and infrared modalities with high precision. This model not only effectively narrows the gap between synthetic data and real-world images but also provides crucial data support for addressing complex multi-modal composite degradation challenges. On the other hand, we develop a prompt-modulated image restoration and fusion network to generate high-quality fusion results. The prompt-modulated module enables our network to dynamically adjust feature distribution based on degradation characteristics (which can be specified by users), achieving robust feature enhancement. This also allows our method to respond to diverse user requirements. Furthermore, we devise a spatial-frequency visual adapter that combines frequencydomain degradation priors to directly extract degradation cues from degraded inputs, eliminating the heavy reliance on user instructions, as customizing prompts for each scene is time-consuming and labor-intensive. As shown in Fig. 1, benefiting from the aforementioned designs, our method excels in handling both real-world and composite degradation scenarios and can flexibly respond to user needs. In summary, our main contributions are as follows:
• We propose ControlFusion, a versatile image restoration and fusion framework that uniformly models diverse degradation types and degrees, using textual and visual prompts as a medium. Specifically, its controllability enables it to respond to user-specific customization needs.
• A spatial-frequency visual adapter is devised to integrate frequency characteristics and directly extract text-aligned degradation prompts from visual images, enabling automated deployment.
• A physics-driven imaging model is developed, integrating physical mechanisms such as the Retinex theory and atmospheric scattering principle to bridge the gap between synthetic data and real-world images, while taking into account the degradation simulation of infrared-visible dual modalities.
this section cite: ['b40', 'b45', 'b46', 'b21', 'b43', 'b0', 'b28', 'b50', 'b9', 'b10', 'b18', 'b14', 'b19', 'b44', 'b49', 'b31', 'b28', 'b50', 'b26', 'b36', 'b27', 'b15', 'b30', 'b41', 'b42']

Section: Related Work
Image Fusion. Earlier visual-oriented fusion approaches primarily concentrated on merging complementary information from multiple modalities and improving visual fidelity. The mainstream network architectures primarily include CNN-based [14,48], AE-based [11,49], GAN-based [19,15], Transformers [20] and diffusion models [50]. Furthermore, several schemes including joint registration and fusion [33,37], semantic-driven [28,15], and degradation-robust [39,42,38] methods, are proposed to broaden the practical applications of image fusion. For instance, Tang et al. [30] and Liu et al. [17] developed corresponding solutions for illumination distortions and noise interference, respectively. In addition, [39] proposed a image restoration and fusion network to address various degradations.
However, it fails to handle mixed degradations and struggles to generalize to real-world scenarios.
In particular, it relies on tedious manual efforts to customize text prompts for each scene, hindering large-scale automated deployment.
this section cite: ['b13', 'b47', 'b10', 'b48', 'b18', 'b14', 'b19', 'b49', 'b32', 'b36', 'b27', 'b14', 'b38', 'b41', 'b37', 'b29', 'b16', 'b38']

Section: Image Restoration with Vision-Language Models.
With the advancement of deep learning toward multimodal integration, the image restoration field has progressively evolved into a new paradigm of text-driven schemes. CLIP [25] establishes visual-textual semantic representation alignment through dual-stream Transformer encoders pre-trained on 400 million image-text pairs. This framework lays the foundation for the widespread adoption of Prompt Engineering in computer vision. Recent studies integrate CLIP encoders with textual user instructions, proposing generalized restoration frameworks including PromptIR [23], AutoDIR [8], and InstructIR [2], which effectively handle diverse degradation types. Moreover, SPIRE [24] further introduces fine-grained textual restoration cues by quantifying degradation severity levels and supplementing semantic information for precision restoration. To eliminate reliance on manual guidance, Luo et al. [18] developed DA-CLIP by fine-tuning CLIP on mixed-degradation datasets, enabling degradation-aware embeddings to facilitate distortion handling. However, existing unified restoration methods are primarily designed for natural images, which exhibit notable limitations when processing multimodal images (e.g., infrared-visible).
this section cite: ['b24', 'b22', 'b7', 'b1', 'b23', 'b17']

Section: Physics-driven Degraded Imaging Model
Due to the differences in the imaging mechanisms of infrared and visible, the types of degradation they face also vary. To tackle the complex and variable degradation challenges, we propose a physics-driven imaging model for infrared and visible images aimed at reducing the domain gap between simulated data and real-world imagery. Infrared (IR) images usually suffer from sensorrelated interference, such as stripe noise and low contrast. While Visible (VI) images are typically degraded by illumination conditions (low light, over-exposure), weather (rain, haze), and sensorrelated issues (noise, blur). Given a clear image I m (m ∈ {ir, vi}), the proposed imaging model can be mathematically represented as:
D m = P s (P w (P i (I m ))) ,(1)
where D m is the corresponding degraded image, and P i , P w , and P s represent illumination-, weather-, and sensor-related distortions, respectively.
this section cite: []

Section: Sensor-related Distortions.
Sensor-related distortions encompass various types of noise, motion blur, and contrast degradation. Contrast degradation and stripe noise of infrared images can be modeled as:
D s ir = P s (I ir ) = α • I ir + 1 H n ⊤ , (2
)
where α is a constant less than 1 for contrast reduction, 1 H ∈ R H is an all-ones column vector and n ∈ R W represents the column-wise Gaussian noise vector sampled from N (0, ϵ 2 ) with ϵ ∈ [1,15]. Moreover, Gaussian noise and motion blur in source images can be modeled as:
D s m = P s (I m ) = I m * K(N, θ) + N (0, σ 2 ),(3)
where N (0, σ 2 ) represents Gaussian noise with σ ∈ [5,20]. * denotes convolution with a blur kernel K(N, θ) = 1 N R θ (δ c ⊗ 1 N ) constructed by rotating an impulse δ c ⊗ 1 N with random angle R θ (θ ∈ [10, 80]), using 1  N to normalize the kernel energy. Here, N ∈ [3, 12] controls the blur level. Illumination-related Distortions. Following the theoretical framework of Retinex, the formation of illumination-degraded images D i vi is mathematically modeled as:
D i vi = P i (I vi ) = Ivi L • L γ , (4
)
where L is the illumination map estimated by LIME [5]. γ ∈ [0.5, 3] controls the illumination level.
Weather-related Distortions. According to the methodologies in [21,12], we employ the following formula to simulate weather-related degradations (i.e. rain and haze):
D w vi = P w (I vi ) = I vi • t + A(1 -t) + R,(5)
where t and A denote the transmission map and atmospheric light, respectively. And t is defined by the exponential decay of light, expressed as t = e -βd
(x) , where the haze density coefficient β ∈ [0.5, 2.0]. d(x) refers to the scene depth, estimated by DepthAnything-V2 [40]. Besides, A is constrained within [0.3, 0.9] for realistic simulation. Encoder and Fusion Layer CR-ATT Transformerbased Feature Extraction Intradomain Fusion Unit + + Up Sample + Prompt-Modulated Module Prompt-Modulated Module Transformer-based Decoder Transformer-based Decoder Prompt-Modulated Module Prompt-Modulated Module Transformer-based Decoder Q K V FFN SE-ATT Encoder and Fusion Layer Encoder and Fusion Layer Encoder and Fusion Layer Down Sample Slide 𝑭𝑭 𝒇𝒇𝒇𝒇𝒇𝒇 𝒎𝒎 Pool Visual Embedding Text Embedding Cosine Similarity × 𝟐𝟐 Linear GELU Text Encoder Various User Instructions MLP Prompt-guided Feature Enhancement Fast Fourier Transform Up Sample Up Sample 𝑭𝑭 Visual Encoder 𝑫𝑫 𝒊𝒊𝒇𝒇 & 𝑫𝑫 𝒗𝒗𝒊𝒊 𝑫𝑫 𝒊𝒊𝒇𝒇 Loss Function 𝑰𝑰 𝒊𝒊𝒇𝒇 𝒉𝒉𝒉𝒉 & 𝑰𝑰 𝒗𝒗𝒊𝒊 𝒉𝒉𝒉𝒉 Down Sample Down Sample Down Sample Conv Linear Based on Eq. ( 1), we construct a multi-modal composite Degradation Dataset with four Levels (DDL-12), comprising 12 distinct degradations. We selecte 2, 050 high-quality clear images from RoadScene [36], LLVIP [7], and MSRS [29] datasets. Subsequently, the degraded imaging model is employed to synthesize degraded images with 12 types and 4 levels of degradation. Finally, the DDL-12 dataset contains approximately 48, 000 training image pairs and 4, 800 test image pairs.
this section cite: ['b0', 'b14', 'b4', 'b19', 'b4', 'b20', 'b11', 'b35', 'b6', 'b28']

Section: Methodology

this section cite: []

Section: Problem Formulation
Given two source images I ir ∈ R H×W ×1 and I vi ∈ R H×W ×3 , the typical fusion paradigm employs a network N f to synthesize the fused image I f , expressed as: I f = N f (I ir , I vi ). However, in complex scenarios, source images usually suffer from degradation interference, thus the advanced fusion paradigm must take both image restoration and fusion into account. While the concatenation approach is straightforward, it does not model recovery and fusion as an end-to-end optimization process, yielding suboptimal outcomes. As illustrated in Fig. 2, we propose a controllable paradigm that couples restoration and fusion with degradation prompts, termed ControlFusion, to overcome this limitation. On the one hand, the coupled approach enhances task synergy. On the other hand, leveraging degradation prompts to modulate the restoration and fusion process enables the network to adapt to diverse degradation distributions while meeting user-specific customization requirements. The proposed restoration and fusion paradigm can be defined as: I f = N rf (I ir , I vi , p | Ω), where N rf is a restoration and fusion network, P denotes degradation prompts, and Ω indicates the degradation set. Our ControlFusion employs a two-stage optimization protocol. Stage I aligns textual prompts with visual embedding, while Stage II optimizes the overall framework.
this section cite: []

Section: Stage I: Textual-Visual Prompts Alignment Spatial-Frequency Collaborative Visual Adapter.
To achieve unified multimodal degradation representation, we first employ text semantics for explicit degradation modeling. Using a pretrained CLIP [25] text encoder E t , the user instruction T instr is converted into text embedding:
p text = E t (T instr ).
However, text-dependent models limit deployment flexibility. We therefore develop a spatial-frequency collaborative visual adapter (SFVA) to extract visual embeddings directly from images while maintaining semantic alignment with p text .
As shown in Fig. 3, degraded images exhibit distinct spectral characteristics, indicating frequency features contain rich degradation priors. Our SFVA contains dual branches: In the frequency branch, Fast Fourier Transform (FFT) and CNN extract frequency features:
F m f re = W -1 x=0 H-1 y=0 D m (x, y)e -j2π( ux W + vy H ) , F f re = Linear Conv [F ir f re , F vi f re ] , (6
)
where [•, •] denotes channel-wise concatenation. The spatial branch employs cropping/downsampling augmentation and CNN to extract spatial features F spa .
The fused visual embedding p vis is obtained through concatenation and linear projection of F f re and F spa . To ensure semantic consistency between visual/text embeddings, we apply MSE loss L mse and cosine similarity loss L cos :
L I = λ 1 ∥p vis -p text ∥ 2 Lmse +λ 2 (1 - p vis • p text ∥p vis ∥∥p text ∥ ) Lcos ,(7)
where λ 1 and λ 2 balance loss components. This design enables automatic degradation-aware adaptation while preserving text-aligned semantics.
this section cite: ['b24']

Section: Stage II: Prompt-modulated Restoration and Fusion

this section cite: []

Section: Network Architectures
Feature Encoding and Fusion Layer. As shown Fig. 2, we devise the hierarchical transformer-based encoders to extract multi-scale feature representations from degraded infrared (D ir ) and visible (D vi ) images separately, which is formulated as:
{F ir , F vi } = E ir (D ir ), E vi (D vi )
, where E ir and E vi are infrared and visible image encoders.
Furthermore, to achieve comprehensive cross-modal feature integration, we design the intra-domain fusion unit, where the cross-attention (CR-ATT) mechanism is employed to facilitate interaction between features across modalities. Specifically, the linear transformation functions F qkv ir and F qkv vi project F ir and F vi into their corresponding Q, K, and V , expressed as:
{Q ir , K ir , V ir } = F qkv ir (F ir ), {Q vi , K vi , V vi } = F qkv vi (F vi ).(8)
Subsequently, we swap the queries Q of complementary modalities to promote spatial interaction:
F ir f = softmax Q vi K ir √ d k V ir , F vi f = softmax Q ir K vi √ d k V vi ,(9)
where d k is scaling factor. We concatenate F ir f and F vi f to get fusion features:
F f = [F ir f , F vi f ].
this section cite: []

Section: Prompt-modulated Module and Image Decoder.
To dynamically adapt fusion feature distributions to degradation patterns and degrees, we propose a Prompt-Modulated Module (PMM) for robust feature enhancement. First, sequential MLPs (Φ p ) derive distributional optimization parameters:[γ p , β p ] = Φ p (p), where p ∈ {p vis , p text }. Then, γ p and β p are applied for feature scaling and bias shifting in a residual manner: Ff = (1 + γ p ) ⊙ F f + β p , where ⊙ denotes the Hadamard product, and Ff indicates enhanced features incorporating degradation prompts. We further deploy a series of Transformer-based decoder D f with the self-attention to progressively reconstruct the fused image I f = D f ( Ff ). In particular, PMM and D f are tightly coupled in a multi-stage process, effectively incorporating fusion features and degradation prompts to synthesize the desired image.
this section cite: []

Section: Loss Functions
Following the typical fusion paradigm [20], we introduce the intensity loss, structural similarity (SSIM) loss, maximum gradient loss, and color consistency loss to constrain the training of Stage II.
The intensity loss L int maximizes the target prominence of fusion results, defined as:
L int = 1 HW ∥I f -max(I hq ir , I hq vi )∥ 1 ,(10)
where I hq ir and I hq vi are the high-quality source images. The structural similarity loss L ssim ensures the fused image maintains structural consistency with the high-quality source images, preserving essential structural information, and is formulated as:
L ssim = 2 -(SSIM(I f , I hq ir ) + SSIM(I f , I hq vi )).(11)
The maximum gradient loss L grad maximizes the retention of key edge information from both source images, generating fusion results with clearer textures, formulated as:
L grad = 1 HW ∥∇I f -max(∇I hq ir , ∇I hq vi )∥ 1 ,(12)
where ∇ denotes the Sobel operator. Moreover, the color consistency loss L color ensures that the fusion results maintain color consistency with the visible image. We convert the image to YCbCr space and minimize the distance between the Cb and Cr channels, expressed as:
L color = 1 HW ∥F CbCr (I f ) -F CbCr (I hq vi )∥ 1 ,(13)
where F CbCr denotes the transfer function of RGB to CbCr.
Finally, the total loss L II for Stage II is the weighted sum of the aforementioned losses:
L II = α int • L int + α ssim • L ssim + α grad • L grad + α color • L color ,(14)
where α int , α ssim , α grad , and α color are hyper-parameters.
this section cite: ['b19']

Section: Experiments

this section cite: []

Section: Implementation and Experimental Configurations
Our image restoration and fusion network is built on a four-stage encoder-decoder architecture, with channel dimensions increasing from 48 to 384, specifically configured as [48,96,192,384]. The model is trained on the proposed DDL-12 dataset. During training, 224 × 224 patches are randomly cropped as inputs, with a batch size of 12 over 100 epochs. Optimization is performed using AdamW, starting with a learning rate of 1 × 10 -3 and decayed to 1 × 10 -5 via a cosine annealing schedule. For the loss configuration, λ 1 and λ 2 are set with a weight ratio of 1 : 3, and α int , α ssim , α grad , and α color are assigned values of 8 : 1 : 10 : 12, respectively.
We introduce text prompts to enable the unified network to effectively handle diverse and complex degradations. Each prompt specifies the affected modality, the degradation type, and its severity, enabling user-controllable flexibility. For example, a typical prompt for a single degradation is: We are performing infrared and visible image fusion, where the modality suffers from a gradeseverity degradation type. For composite degradations, we extend this template to specify multiple modality-degradation pairs, e.g., We are performing infrared and visible image fusion. Please handle a grade-severity-A degradation type-A in the modality-A, and a grade-severity-B degradation type-B in the modality-B. Details of the prompt construction paradigm are provided in the Appendix.
We compare our method with seven SOTA fusion methods, i.e., DDFM [50], DRMF [31], EMMA [51], LRRNet [11], SegMiF [16], Text-IF [39], and Text-DiFuse [42]. Firstly, we evaluate the fusion performance on four widely used datasets, i.e., MSRS [29], LLVIP [7], RoadScene [35], and FMB [16]. The test set sizes for MSRS, LLVIP, RoadScene, and FMB are 361, 50, 50, and 50, respectively. Four metrics, i.e., EN, SD, VIF, and Q abf , are used to quantify fusion performance. Additionally, we evaluate the restoration and fusion performance across various degradations, including low-contrast, random noise, and stripe noise in infrared images (IR), as well as blur, rain, over-exposure, and low light in visible images (VI). Additionally, we evaluate its effectiveness in handling composite degradation scenarios. Each degradation scenario includes 100 image pairs from DDL-12 dataset. We use CLIP-IQA [34], MUSIQ [9], TReS
this section cite: ['b47', 'b49', 'b30', 'b50', 'b10', 'b15', 'b38', 'b41', 'b28', 'b6', 'b34', 'b15', 'b33', 'b8']

Section: Fusion Performance Comparison
Comparison without Pre-enhancement. Table 1 summarizes the quantitative assessment across benchmark datasets. Notably, our method attains SOTA performance in SD, VIF, and Q abf metrics, showcasing its exceptional ability to maintain structural integrity and edge details. The EN metric remains competitive, indicating that our fusion results encapsulate abundant information.
Comparison with Pre-enhancement. For a fair comparison, we exclusively use the visual prompts to characterize degradation in the quantitative comparison. Meanwhile, we employ several SOTA image restoration algorithms as preprocessing steps to address specific degradations. Specifically, OneRestore [6] is used for weather-related degradations, Instruct-IR [2] for illumination-related degradations, and DA-CLIP [18] and WDNN [4] for sensor-related degradations. For composite degradations, we select the best-performing method from these algorithms.
The quantitative results in Tab. 2 show that ControlFusion achieves superior CLIP-IQA, MUSIQ, and TReS scores across most degradation scenarios, demonstrating a strong capability in degradation mitigation and complementary context aggregation. Additionally, no-reference fusion metrics (SD, EN) show that ControlFusion performs on par with SOTA image fusion methods. Qualitative comparisons in Fig. 4 indicate that ControlFusion not only excels in addressing single-modal degradation but also effectively tackles more challenging single-and multi-modal composite degradations. In particular, when rain and haze coexist in the visible image, ControlFusion successfully perceives these degradations and eliminates their effects. In multi-modal composite degradation scenarios, it leverages degradation prompts to adjust the distribution of fusion features, achieving high-quality restoration and fusion. Extensive experiments demonstrate that ControlFusion exhibits significant advantages in complex degradation scenarios.   Real-world Generalization. As shown in Tab. 3, we report comparative results on nighttime scenes from MSRS and the real weather-degraded benchmark AWMM-100k [13]. Our method achieves the best performance on CLIP-IQA, TRes, and SD, while slightly trailing Text-DiFuse on MUSIQ . On AWMM-100k, our approach consistently outperforms competing methods across all evaluation metrics under real-world weather degradation conditions.
this section cite: ['b5', 'b1', 'b17', 'b3', 'b12']

Section: Extended Experiments
In addition to quantitative comparisons, Fig. 5 presents visual examples demonstrating our method's strong generalization ability in removing composite degradations in real scenes. The top image shows typical data collected by our multimodal sensors under challenging low-light and noisy conditions, pixel brigntness value level 9 level 8 level 7 Input IR/VI Text-DiFuse ControlFusion with various levels for prompting level =7 level =8 level =9 level =3 level =4 level =5 level =5 level =4 level =3 level =9 level =8 level =7 Prompt: … merge infrared and visible images, handling grade #level low-light/over exposure problem … Low-light Low-light Over-exposure Over-exposure pixel brightness value level 9 level 8 level 7 level 5 level 4 level 3 pixel brightness value level 5 level 4 level 3 Brightness distribution pixel brightness value  while the bottom image illustrates the robustness of our method in handling extremely low-light scenarios. Additional qualitative results are provided in the Appendix.
this section cite: []

Section: Flexible Degree Control.
We employ numerical values to quantify the severity of degradation in design, where higher numbers indicate more severe degradation, enabling precise continuous control.
It should be noted that while only four anchor points (1, 4, 7, and 10) were used during training, the model demonstrates remarkable generalization capability to intermediate degrees during testing, owing to the inherent encoding characteristics of CLIP. As shown in Fig. 6, with degradation level prompts, our ControlFusion can adapt to varying degrees of degradation and deliver satisfactory fusion results. Moreover, fusion results generated using adjacent degradation-level prompts exhibit subtle yet perceptible differences, allowing users to obtain customized outcomes through their specific prompts. More visualization results are presented in the Appendix. Object Detection. We evaluate object detection performance on LLVIP to assess the fusion quality, using the retrained YOLOv8 [26]. Quantitative results are presented in Tab. 5. Our results enable the detector to identify all pedestrians with higher confidence, achieving the highest mAP@0.5-0.95. Visualizations are in the Appendix.   7, removing any loss or module significantly impacts the fusion quality. Specifically, without L color , color distortion occurs, while removing L grad leads to the loss of essential texture information. Excluding L int fails to highlight thermal targets, and removing PMM diminishes the ability to address composite degradation. Additionally, removing the frequency branch from SFVA causes visual prompt confusion. The t-SNE results further validate the importance of frequency priors. The quantitative results in Tab. 6 also confirm that each design element is crucial for enhancing fusion performance.
this section cite: ['b25']

Section: Computational Efficiency.

this section cite: []

Section: Discussions and Limitations.
To mitigate the gap between simulated and real-world data, our method constructs training samples using the degradation imaging model. An alternative strategy is test-time adaptation, in which part of the model is fine-tuned on the test set to better accommodate new data distributions. It should be noted that the degradation imaging model is specifically designed for infrared and visible image fusion and is difficult to generalize to other fusion tasks, such as medical image fusion and multi-focus image fusion.
this section cite: []

Section: Conclusion
This work proposes a controllable framework for image restoration and fusion leveraging languagevisual prompts. Initially, we develop a physics-driven degraded imaging model to bridge the domain gap between synthetic data and real-world images, providing a solid foundation for addressing composite degradations. Moreover, we devise a prompt-modulated network that adaptively adjusts the fusion feature distribution, enabling robust feature enhancement based on degradation prompts. Prompts can either come from text instructions to support user-defined control or be extracted from source images using a spatial-frequency visual adapter embedded with frequency priors, facilitating automated deployment. Extensive experiments demonstrate that our method excels in handling real-world and composite degradations, showing strong robustness across various degradation levels.
this section cite: []

Section: References
Ref_id:b0 Title: Heat-assisted detection and ranging Year: (2023)
Ref_id:b1 Title: Instructir: High-quality image restoration following human instructions Year: (2024)
Ref_id:b2 Title: No-reference image quality assessment via transformers, relative ranking, and self-consistency Year: (2022)
Ref_id:b3 Title: Wavelet deep neural network for stripe noise removal Year: (2019)
Ref_id:b4 Title: Lime: Low-light image enhancement via illumination map estimation Year: (2016)
Ref_id:b5 Title: Onerestore: A universal restoration framework for composite degradation Year: (2024)
Ref_id:b6 Title: Llvip: A visible-infrared paired dataset for low-light vision Year: (2021)
Ref_id:b7 Title: Autodir: Automatic all-in-one image restoration with latent diffusion Year: (2024)
Ref_id:b8 Title: Musiq: Multi-scale image quality transformer Year: (2021)
Ref_id:b9 Title: Densefuse: A fusion approach to infrared and visible images Year: (2019)
Ref_id:b10 Title: Lrrnet: A novel representation learning guided fusion network for infrared and visible images Year: (2023)
Ref_id:b11 Title: Heavy rain image restoration: Integrating physics model and conditional adversarial learning Year: (2019)
Ref_id:b12 Title: All-weather multimodality image fusion: Unified framework and 100k benchmark Year: (2024)
Ref_id:b13 Title: Fusion from decomposition: A self-supervised decomposition approach for image fusion Year: (2022)
Ref_id:b14 Title: Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection Year: (2022)
Ref_id:b15 Title: Multi-interactive feature learning and a full-time multi-modality benchmark for image fusion and segmentation Year: (2023)
Ref_id:b16 Title: Paif: Perception-aware infrared-visible image fusion for attack-tolerant semantic segmentation Year: (2023)
Ref_id:b17 Title: Controlling visionlanguage models for multi-task image restoration Year: (2024)
Ref_id:b18 Title: Fusiongan: A generative adversarial network for infrared and visible image fusion Year: (2019)
Ref_id:b19 Title: Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer Year: (2022)
Ref_id:b20 Title: Optics of the atmosphere: scattering by molecules and particles Year: (1976)
Ref_id:b21 Title: Cognitively-engineered multisensor image fusion for military applications Year: (2009)
Ref_id:b22 Title: Promptir: Prompting for all-in-one image restoration Year: (2023)
Ref_id:b23 Title: Spire: Semantic prompt-driven image restoration Year: (2024)
Ref_id:b24 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b25 Title: You only look once: Unified, real-time object detection Year: (2016)
Ref_id:b26 Title: Superfusion: A versatile image registration and fusion network with semantic awareness Year: (2022)
Ref_id:b27 Title: Image fusion in the loop of high-level vision tasks: A semanticaware real-time infrared and visible image fusion network Year: (2022)
Ref_id:b28 Title: Piafusion: A progressive infrared and visible image fusion network based on illumination aware Year: (2022)
Ref_id:b29 Title: Divfusion: Darkness-free infrared and visible image fusion Year: (2023)
Ref_id:b30 Title: Drmf: Degradationrobust multi-modal image fusion via composable diffusion prior Year: (2024)
Ref_id:b31 Title: Mask-difuser: A masked diffusion model for unified unsupervised image fusion Year: (2025)
Ref_id:b32 Title: Unsupervised misaligned infrared and visible image fusion via cross-modality image generation and registration Year: ()
Ref_id:b33 Title: Exploring clip for assessing the look and feel of images Year: (2023)
Ref_id:b34 Title: Fusiondn: A unified densely connected network for image fusion Year: (2020)
Ref_id:b35 Title: U2fusion: A unified unsupervised image fusion network Year: (2022)
Ref_id:b36 Title: Murf: Mutually reinforcing multi-modal image registration and fusion Year: (2023)
Ref_id:b37 Title: Urfusion: Unsupervised unified degradationrobust image fusion network Year: (2025)
Ref_id:b38 Title: Text-if: Leveraging semantic text guidance for degradation-aware and interactive image fusion Year: (2024)
Ref_id:b39 Title: Restormer: Efficient transformer for high-resolution image restoration Year: (2022)
Ref_id:b40 Title: Image fusion meets deep learning: A survey and perspective Year: (2021)
Ref_id:b41 Title: Text-difuse: An interactive multi-modal image fusion framework based on text-modulated diffusion model Year: (2024)
Ref_id:b42 Title: Omnifuse: Composite degradation-robust image fusion with language-driven semantics Year: (2025)
Ref_id:b43 Title: Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers Year: (2023)
Ref_id:b44 Title: Transformer-based end-to-end anatomical and functional image fusion Year: (2022)
Ref_id:b45 Title: Visible and infrared image fusion using deep learning Year: (2023)
Ref_id:b46 Title: Vehicle tracking using surveillance with multimodal data fusion Year: (2018)
Ref_id:b47 Title: Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection Year: (2023)
Ref_id:b48 Title: Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion Year: (2023)
Ref_id:b49 Title: Ddfm: Denoising diffusion model for multi-modality image fusion Year: (2023)
Ref_id:b50 Title: Equivariant multi-modality image fusion Year: (2024)
