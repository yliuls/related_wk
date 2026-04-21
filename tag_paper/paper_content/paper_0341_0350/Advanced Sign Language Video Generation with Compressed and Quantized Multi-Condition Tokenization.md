Title: Advanced Sign Language Video Generation with Compressed and Quantized Multi-Condition Tokenization
Abstract: Sign Language Video Generation (SLVG) seeks to generate identity-preserving sign language videos from spoken language texts. Existing methods primarily rely on the single coarse condition (e.g., skeleton sequences) as the intermediary to bridge the translation model and the video generation model, which limits both the naturalness and expressiveness of the generated videos. To overcome these limitations, we propose SignViP, a novel SLVG framework that incorporates multiple fine-grained conditions for improved generation fidelity. Rather than directly translating error-prone high-dimensional conditions, SignViP adopts a discrete tokenization paradigm to integrate and represent fine-grained conditions (i.e., fine-grained poses and 3D hands). SignViP contains three core components.(1) Sign Video Diffusion Model is jointly trained with a multi-condition encoder to learn continuous embeddings that encapsulate fine-grained motion and appearance.(2) Finite Scalar Quantization (FSQ) Autoencoder is further trained to compress and quantize these embeddings into discrete tokens for compact representation of the conditions. (3) Multi-Condition Token Translator is trained to translate spoken language text to discrete multi-condition tokens. During inference, Multi-Condition Token Translator first translates the spoken language text into discrete multi-condition tokens. These tokens are then decoded to continuous embeddings by FSQ Autoencoder, which are subsequently injected into Sign Video Diffusion Model to guide video generation. Experimental results show that SignViP achieves state-of-the-art performance across metrics, including video quality, temporal coherence, and semantic fidelity. The code is available at https://github.com/  umnooob/signvip/.

Section: Introduction
Sign language, as a visual language, serves as the primary communication medium for deaf individuals. Early research focused on Sign Language Recognition (SLR) [32,45,90], Translation (SLT) [5,20,42,14], or Production (SLP) [58,56,60,86,83]. More recently, Sign Language Video Generation (SLVG) [57,61,47,69] has gained increasing attention, which aims to generate realistic and expressive sign language videos from spoken language texts, preserving the unique identity of a target signer, as shown in Figure 1(1). This growing interest is driven by its potential applications in accessibility technologies, educational tools, immersive communication systems, etc.
SLVG presents a challenging problem due to the lack of explicit spatial or temporal alignment between the input (i.e., the spoken language texts) and output (i.e., the sign language videos) modalities. To address this, current SVLG methods focus on leveraging synchronized auxiliary conditions as an intermediary to align these two modalities. As shown in Figure 1(2a), most of the existing SLVG methods [57,61] leverage skeletal sequences as an intermediary to bridge a text-to-skeleton translation model (i.e., an SLP model) and a skeleton-to-video generation model. Because the generative model is only guided by a single coarse condition (e.g., skeleton), such single-conditional methods struggle with the naturalness and expressiveness of the generated videos, particularly in capturing facial expressions and figure movements, as illustrated in Figure 1(3).
Recent advancements in human video generation have shown that incorporating fine-grained conditions (e.g., dense pose [84] or 3D models [9,82]) or leveraging multiple conditions (e.g., depth combined with optical flow [85]) can substantially improve generative fidelity. Inspired by these, we consider whether multiple fine-grained conditions can be introduced to enhance the quality and expressiveness of generated sign language videos. As shown in Figure 1(2b), one intuitive solution is to extend the single-conditional methods by considering multiple fine-grained conditions as intermediaries, where a multi-condition translation model can directly predict capable of achieving multiple fine-grained conditions. However, as shown in Figure 1(4), we observe that directly translating such attributes is challenging due to their high-dimensional nature and susceptibility to errors. This raises an important question: How can we overcome the challenges in the translation of multiple fine-grained conditions to further advance SLVG?
To address the challenges, we propose SignViP, a novel framework designed to advance SLVG by incorporating multiple fine-grained conditions for enhanced generation fidelity. As shown in Figure 1(2c), instead of directly translating high-dimensional conditions from spoken language texts, SignViP adopts a discrete tokenization paradigm to effectively integrate and represent these fine-grained conditions. Central to this framework is the construction of a discrete multi-condition token space, which bridges fine-grained conditions (e.g., fine-grained poses and 3D hands) with the dynamics of sign language video frames. The framework consists of three key components: (1) Sign Video Diffusion Model is jointly trained with a multi-condition encoder using denoising loss to generate continuous embeddings that encapsulate fine-grained motion and appearance details; (2) Finite Scalar Quantization (FSQ) Autoencoder is trained with reconstruction loss to compress and quantize the continuous embeddings into discrete tokens, enabling highly dense representation for the conditions;
(3) Multi-Condition Token Translator is built upon an autoregressive model to translate spoken language text to discrete multi-condition tokens. During inference, the spoken language text is first translated into discrete multi-condition tokens by Multi-Condition Token Translator. These tokens are then decoded back into continuous embeddings by FSQ Autoencoder, which are subsequently injected into Sign Video Diffusion Model to guide sign language video generation. Our experimental results demonstrate that SignViP achieves state-of-the-art performance across multiple evaluation metrics, including video quality, temporal coherence, and semantic fidelity.
Our main contributions are summarized as follows:
• We introduce SignViP, a novel framework for Sign Language Video Generation (SLVG) that incorporates multiple fine-grained conditions for improved video quality and expressiveness.
• We propose a discrete tokenization paradigm through the construction of a discrete multi-condition token space to bridges fine-grained conditions with the dynamics of sign language video frames.
• The experiments validate the effectiveness of SignViP, demonstrating state-of-the-art performance across diverse metrics.
this section cite: ['b31', 'b44', 'b89', 'b4', 'b19', 'b41', 'b13', 'b57', 'b55', 'b59', 'b85', 'b82', 'b56', 'b60', 'b46', 'b68', 'b56', 'b60', 'b83', 'b8', 'b81', 'b84']

Section: Related Works
Sign Language Video Generation. Sign Language Video Generation (SLVG) aims to generate identity-preserving sign language videos from spoken language texts. Early methods decompose the task into two consecutive sub-tasks [57,61], which are the text-to-skeleton translation (i.e., SLP) and the skeleton-to-video generation. SignGAN [57] first employs a transformer [76] with a mixture density formulation to translate spoken language text to skeletal sequence. Then, a GAN-based [17] skeleton-conditioned human synthesis model is introduced to generate sign language videos. FS-Net [61] extends SignGAN by predicting the temporal alignment to a continuous signing sequence.
Because the single condition focuses solely on capturing basic pose structures while neglecting fine-grained details, the generated videos tend to appear less natural and expressive. SignGen [47] seeks to address this limitation through a novel end-to-end pipeline that integrates multi-condition guidance, including optical flow, pose, and depth. However, SignGen suffers from training-inference inconsistency, which leads to suboptimal results. In this paper, we aim to develop a framework that leverages multiple fine-grained conditions to enhance the quality and expressiveness of generated sign language videos.
this section cite: ['b56', 'b60', 'b56', 'b75', 'b16', 'b60', 'b46']

Section: Human Video Generation.
Human video generation has advanced significantly with the deep generative models. Early approaches, such as Pix2PixHD [80] and vid2vid [79], leveraged GANs [17] to generate realistic images and videos from the structured inputs. Several works have also explored the human pose generation, conditioning on the whole body [2,38,40,63], face [10,33], or hands [73,35]. However, GAN-based methods often suffer from mode collapse and optimization challenges. More recently, diffusion models [70,28,67,78,64,68] have emerged as a robust alternative, producing high-quality images or videos with greater stability. Most prior diffusion-based approaches rely on ControlNet [88] and OpenPose [7] to process each video frame independently, neglecting the temporal consistency and leading to the inevitable flickering artifacts. Pose-guided diffusion models [62,29,77,91,65,66] addresses this issue by generating temporally consistent human videos while preserving appearance fidelity. Furthermore, recent research shows that incorporating fine-grained conditions, such as dense pose [84] or 3D models [9,82], or leveraging multiple complementary conditions, such as depth and optical flow [85], can significantly enhance generative fidelity. Building on these advancements, we aim to harness state-of-the-art diffusion-based methods alongside multiple fine-grained conditions to advance SLVG further.
this section cite: ['b79', 'b78', 'b16', 'b1', 'b37', 'b39', 'b62', 'b9', 'b32', 'b72', 'b34', 'b69', 'b27', 'b66', 'b77', 'b63', 'b67', 'b87', 'b6', 'b61', 'b28', 'b76', 'b90', 'b64', 'b65', 'b83', 'b8', 'b81', 'b84']

Section: Methodology

this section cite: []

Section: Preliminary
Diffusion Models. As a class of the generative models, the diffusion models [70,28] consists of two processes, which are the diffusion process and the denoising process, respectively.
this section cite: ['b69', 'b27']

Section: In the diffusion
Spoken Language Text da können wir mit unserem osterwetter eigentlich ... process, the Gaussian noise is iteratively added to degrade the input sample over T steps until the sample becomes completely random noise. In the denoising process, a denoising model is used to iteratively generate a sample from the sampled Gaussian noise. When training, given an input sample x 0 and condition c, the denoising loss is defined as
this section cite: []

Section: Multi-Condition Tokens
L denoise = E x0,ϵ∼N (0,I),c,t ∥ϵ θ (x t , c, t) -ϵ∥ 2 .(1)
Among them, x t = √ α t x 0 + √ 1 -α t ϵ is the noisy sample at timestep t ∈ [1, T ], where α t is a predefined scalar from the noise scheduler. ϵ is the added noise. ϵ θ is the denoiser with the learnable parameters θ, which predicts the noise to be removed from the noisy sample. Latent Diffusion Models (LDMs) [55] stands out as one of the most popular diffusion models. It performs the two processes in the latent space, which is encoded by a Variational Auto-Encoder (VAE) [31,54].
Finite Scalar Quantization. Finite Scalar Quantization (FSQ) [41] is a concise quantization technique to compress continuous values into discrete values. FSQ can be an alternative of Vector Quantization (VQ) [18] in VQ-VAE [75]. Compared with VQ, FSQ does not suffer from codebook collapse and does not need complex machinery to learn expressive discrete representations. Specifically, given a scalar z ∈ R from the encoded latent, the quantized discrete value by FSQ is
f (z; L) = (L -1)σ(z) ; FSQ(z) ≜ rnd(f (z; L)) ∈ [0, 1, • • • , L -1] .(2)
Among them, rnd(•) is round function. σ(•) is sigmoid function. L is the predefined quantization level. Through this process, each value z is enumerated, leading to a bijection from z to an integer in {0, 1, . . . , L -1}. For a d-dimensional latent vector, the total codebook size is the product of L i across all dimensions, resulting in d i=1 L i possible discrete representations.
this section cite: ['b54', 'b30', 'b53', 'b40', 'b17', 'b74']

Section: Overview
Given a reference signer image x ⋆ and a spoken language text T , SLVG aims to generate a sign language video X with F frames, where the signer performs sign language accurately aligned with the semantics of the spoken language text. Specifically, the SLVG task can be formulated as p θ (X|x ⋆ , T ), where θ is parameters of the SLVG model.
Due to the lack of explicit spatial or temporal alignment between T and X, current SVLG methods focus on leveraging synchronized auxiliary conditions (e.g., skeletal sequence) as an intermediary to align them. Such pipeline paradigm can be formulated as
p θ (X|x ⋆ , T ) = p θgen (X|x ⋆ , C) • p θtran (C|T ) .(3)
Among them, C is the synchronized auxiliary conditions which spatially and temporally aligned with the target video frames. θ tran is parameters of the text-to-condition translation model, while θ gen is parameters of condition-to-video generation model.
To address the generation quality issues caused by relying on a single coarse condition, we introduce multiple fine-grained conditions as intermediaries. Specifically, we utilize fine-grained poses and 3D hands. Fine-grained poses capture the signer's body posture and facial expressions, while 3D hands provide detailed and accurate descriptions of hand movements, even in the presence of occlusions.
To avoid directly translating error-prone high-dimensional conditions, SignViP employs a discrete tokenization paradigm with effective integration and representation of these fine-grained conditions.
As shown in Figure 2, the spoken language text T is first translated into the discrete multi-condition tokens d 0:F -1 by the Multi-Condition Token Translator. The multi-condition tokens d 0:F -1 are then decoded by the FSQ Autoencoder to continuous multi-condition embeddings e 0:F -1 , which are equivalent to the embeddings obtained from a multi-condition encoder that encodes multiple fine-grained conditions (i.e., fine-grained poses and 3D hands). Finally, e 0:F -1 are injected into Sign Video Diffusion Model to guide the sign language video generation (i.e., animating the signer in reference image x ⋆ ). The overall pipeline of SignViP can be formulated as
p θ (X|x ⋆ , T ) = p θgen (X|x ⋆ , e 0:F -1 ) • p θAE (e 0:F -1 |d 0:F -1 ) • p θtran (d 0:F -1 |T ) ,(4)
where θ gen , θ AE , and θ tran denote parameters of Sign Video Diffusion Model, FSQ Autoencoder, and Multi-Condition Token Translator, respectively.
this section cite: []

Section: Construction of Multi-Condition Token Space
SignViP is trained with three steps to construct the multi-condition token space for the discrete tokenization paradigm.
Step I. We train Sign Video Diffusion Model with a multi-condition encoder (i.e., a multi-layer convolution network) using a denoising loss to establish a connection between the conditions and the sign language videos. Specifically, multiple fine-grained conditions (e.g., fine-grained poses and 3D hands) are encoded by the multi-condition encoder into the continuous multi-condition embeddings e 0:F -1 . These embeddings, along with the reference image x ⋆ , serve as the guidance signals for Sign Video Diffusion Model to perform diffusion process. More details can be found in Section 3.4.
this section cite: []

Section: Step II.
We train FSQ Autoencoder using a reconstruction loss to learn the compression and quantization of multi-condition embeddings e 0:F -1 . The encoder E FSQ of FSQ Autoencoder compresses and quantizes the embeddings e 0:F -1 into discrete multi-condition tokens d 0:F -1 , while the its decoder D FSQ reconstructs e 0:F -1 from d 0:F -1 . More details can be found in Section 3.5.
this section cite: []

Section: Step III.
We train the Multi-Condition Token Translator to autoregressively translate spoken language text T to the multi-condition tokens d 0:F -1 . More details can be found in Section 3.6.
this section cite: []

Section: Sign Video Diffusion Model
Sign Video Diffusion Model aims to generate sign language videos in a diffusion-based manner [70,28] under the guidance of the reference image x ⋆ and the continuous multi-condition embeddings e 0:F -1 . Inspired by the previous works [29,91], as shown in Figure 2(3), Sign Video Diffusion Model consists of three modules: Condition Guider, Denoising U-Net, and Reference Net. Condition Guider and Reference Net respectively encode the multi-condition embeddings e 0:F -1 and the reference image x ⋆ to guide the Denoising U-Net.
Denoising U-Net is the backbone of the Sign Video Diffusion Model, which mirrors the architecture of Stable Diffusion (SD) v1.5 [55]. Each U-Net block includes a ResNet layer [22], a self-attention layer, and a temporal-attention layer [21]. The self-attention layer and the temporal-attention layer perform attention operation [76] along the spatial axes and the temporal axis, respectively.
Condition Guider is a lightweight guidance network that encodes e 0:F -1 , whose each block consists of convolution layers and a temporal attention layer. The output feature of each block is added to the corresponding block's feature in the downsampling part of the Denoising U-Net.
Reference Net [29] shares the same architecture of SDv1.5 and operates in parallel with the Denoising U-Net. The reference image x ⋆ is first encoded into the latent space by the VAE encoder E VAE ,
z ⋆ = E VAE (x ⋆ ).
The encoded reference latent z ⋆ is then fed into the Reference Net. The output feature of the self-attention layer in each block of the Reference Net is spatially concatenated with the input feature of the self-attention layer in the corresponding block of the Denoising U-Net.
During training, the loss function is the extended denoising loss of Equation 1,
L denoise = E Z0,ϵ∼N (0,I),t ∥ϵ θ (Z t ; r ⋆ , E, t) -ϵ t ∥ 2 .(5)
Among them,
Z 0 = E VAE (X 0 ) is the target latent which is encoded from the target video X 0 . ϵ θ is the Denoising U-Net. r ⋆ = R(z ⋆ )
is the reference features, which are encoded by the Reference Net R. E = C(e 0:F -1 ) is the conditional features, which are encoded by the Condition Guider C.
Considering that subtle pose variations in sign language videos carry important semantic meaning, the model needs to be robust to potential anomalies in the generated condition sequences. To address this, we propose condition augmentation. Specifically, each condition frame has a probability p of being randomly replaced with frames from other videos, deliberately introducing controlled disruptions in the temporal continuity. By exposing the model to these artificial discontinuities during training, we effectively enhance its robustness to unexpected conditional transitions.
The inference starts from the sampled Gaussian noise. Then, the diffusion scheduler (e.g., DDIM [71]) is applied to generate images with multiple denoising steps. For each inference step, the noise prediction relies on Classifier-Free Guidance (CFG) [27]. Finally, the generated video is achieved from the latent by a VAE decoder D VAE .
this section cite: ['b69', 'b27', 'b28', 'b90', 'b54', 'b21', 'b20', 'b75', 'b28', 'b70', 'b26']

Section: FSQ Autoencoder
The FSQ Autoencoder is designed to establish a connection between the multi-condition embeddings e 0:F -1 and the corresponding discrete tokens d 0:F -1 . The pre-trained multi-condition encoder first encodes multiple conditions to the continuous embeddings e 0:F -1 . These embeddings are subsequently compressed and quantized into d 0:F -1 by the FSQ Autoencoder encoder E FSQ , which provides a compact representation for the Multi-Condition Token Translator. Finally, the FSQ Autoencoder decoder D FSQ dequantizes d 0:F -1 and reconstructs e 0:F -1 . The training objective of the FSQ Autoencoder is an L2 reconstruction loss.
L FSQ-AE = E e 0:F -1 ∥D FSQ (E FSQ (e 0:F -1 )) -e 0:F -1 ∥ 2 . (6
)
The architecture of the FSQ Autoencoder follows that of the VAE. Instead of applying variational Bayesian inference in the latent space, it performs the FSQ operation, as illustrated in Equation 2.
this section cite: []

Section: Multi-Condition Token Translator
Multi-Condition Token Translator is designed to translate the spoken language text T into the discrete multi-condition tokens d 0:F -1 . Since sign language videos often exceed 100 frames, and each frame should maintain coherent temporal relationships without a strict internal order, we design a frame-level autoregressive model.
As shown in Figure 2(1), following previous works [86,3], the spoken language text T is firstly encoded by CLIP text encoder [49] to obtain semantic embeddings, which serve as the initial input hidden states of the GPT-2 model [48]. Each output hidden state of the GPT model is decoded through multiple parallel prediction heads to generate all tokens of the corresponding frame simultaneously. On the input side, tokens belonging to the same frame are mixed to obtain unified input hidden states. Unlike methods that require dedicated modules for video length prediction [83] or rely on real video lengths [58], Multi-Condition Token Translator naturally determines the video generation endpoint by producing an "[EOS]" token.
During training, given the pre-trained multi-condition encoder and FSQ Autoencoder encoder, the produced tokens are considered the ground-truth. The cross-entropy loss is computed between the predicted tokens b0:F and the ground-truth tokens b 0:Ffoot_0 ,
L AR = 1 F + 1 F i=0 CrossEntropy( bi ; b i ) .(7)
To mitigate the exposure bias issue between training and inference, we employ a scheduled sampling strategy, wherein 40% of the input tokens are randomly replaced with arbitrary indices from the vocabulary during training. This approach improves the model's robustness and generalization performance during inference.
this section cite: ['b85', 'b2', 'b48', 'b47', 'b82', 'b57']

Section: Experiments

this section cite: []

Section: Experimental Settings
Datasets. We employ two sign language datasets for experiments. (1) RWTH-2014T [5] is a German sign language dataset. It comprises 8,257 sign language videos. The dataset is divided into 7,096 training samples, 519 validation samples, and 642 test samples. To align with the 8× downsampling rate of VAE, the frame size was resized from 260×210 to 272×224. (2) How2Sign [11] is an American sign language dataset. It includes 2,456 sign language videos. Using the provided timestamps, we segmented the videos to create a sentence-level dataset. This dataset consists of 31,128 training samples, 2,322 test samples, and 1,741 validation samples. The frame size is set to 512×512.
this section cite: ['b4', 'b10']

Section: Evaluation Metrics.
To evaluate semantic consistency, we utilize the back-translation metrics following ProTran [58]. Specifically, we train an SLT model [6] to translate sign language videos or poses back into texts. The back-translated text is then compared with the ground-truth text with metrics of BLEU [44] and ROUGE-L [34]. To provide a more comprehensive evaluation of back-translated texts, we further employ the COMET [52, 51] metric, which is specifically designed to predict human judgments of machine translation quality. COMET is widely used for machine translation tasks [19,1,24] and is considered more suitable than BLEU and ROUGE. To evaluate the video quality, we employ FID [26], CLIP-FID, FVD [74], and Identity Similarity (IDS). Among them, CLIP-FID is a variation of FID that utilizes CLIP [49] embedding as the frame's embedding. IDS measures the identity consistency between generated and ground-truth videos. It calculates the cosine similarity of face embeddings extracted using YOLO5Face [46] and Arc2Face [43]. To further investigate the generative capability of video diffusion models in addition to FVD [74], we employ frame-level metrics including PSNR [13], SSIM [81], and LPIPS [89], leveraging their suitability in scenarios where ground-truth videos are temporally aligned with the generated videos. Additionally, we introduce Hand SSIM, which measures SSIM specifically in the hand region for a more precise evaluation of the hand quality.
Implementation Details. In Multi-Condition Token Translator, we utilize a multilingual version of the CLIP model 4 to enable handling of multiple spoken language texts effectively. In FSQ Autoencoder, the encoder and decoder follow the architecture of their counterparts in VAE. Specifically, FSQ Autoencoder applies 4 latent channels, with each channel having a quantization level of 5. Together, this results in a total vocabulary size of 625, computed as 5 4 = 625 due to the combination of levels across all channels. In Sign Video Diffusion Model, both the Denoising U-Net and the Reference Net are initialized with Stable Diffusion v1.5 5 . The temporal-attention layers in the Denoising U-Net are initialized from AnimateDiff [21]. The condition augmentation rate is set to 0.001. During inference, Sign Video Diffusion Model utilizes a guidance scale of 3.5 for CFG. Additionally, the number of inference steps is configured to 50.
Training Details. The training of the three stages are conducted on 4 NVIDIA RTX A6000 GPUs using Adam optimizer [30], with each stage consisting of 50,000 training steps. The batch sizes of stage I, II, and III are 2, 16, and 16. Their learning rates are 1e-5, 5e-5, and 1e-6.
this section cite: ['b57', 'b5', 'b43', 'b33', 'b18', 'b0', 'b23', 'b25', 'b73', 'b48', 'b45', 'b42', 'b73', 'b12', 'b80', 'b88', 'b20', 'b29']

Section: Comparison Back-Translation Comparison.
To quantitatively evaluate the semantic accuracy of the generated sign language videos, we perform two types of back-translation comparisons. Specifically, we respectively train a video-to-text translation model and a pose-to-text translation model to compare with SLVG methods and SLP methods [6]. For pose back-translation comparison, we extract pose sequences from the generated videos using OpenPose [7] and a 2D-to-3D mapping method [87]. As shown in Table 1, in video back-translation comparison, SignViP outperforms all competing methods, including SignGAN [61], its enhanced version using AnimateAnyone [29], and SignGen [47]. These results validate SignViP as a more reliable solution for SLVG by effectively preserving semantic consistency. As shown in Table 2, in pose back-translation comparison, SignViP consistently outperforms previous SLVG methods and SLP methods (i.e., ProTran [58], Adversarial Training [56], and MDN [60]) across most evaluation metrics. Although MoMP [59] achieves slightly higher scores than our SignViP on certain metrics, our method remains highly competitive overall. It is worth noting that these SLP baselines translate text directly into pose sequences, which aligns with our pose back-translation evaluation pipeline. In contrast, our SLVG method requires detecting poses from the generated videos, potentially introducing additional errors that could impact evaluation results. To enable a more fair comparison under the SLVG setting, we further combine the state-of-the-art SLP method, MoMP, with a pose-to-video generation approach (i.e., ControlNet [88] or AnimateAnyone [29]). As shown in the first two rows of Table 1, the video back-translation results demonstrate that our method is better suited for the SLVG task compared to MoMP-based methods. These results further underscore SignViP's effectiveness in preserving semantic accuracy.
Video Quality Comparison. Table 3 summarizes the video quality comparison of the generated sign language videos. Our proposed SignViP method significantly outperforms prior SLVG approaches across all evaluated metrics. Specifically, the lowest FID, CLIP-FID, and FVD achieved by our model demonstrate its superior ability to generate sign language videos that are not only visually realistic but also exhibit high temporal coherence and natural motion consistency. Furthermore, the highest IDS scores achieved by our method highlight its effectiveness in accurately preserving the identity of the signer. These results collectively validate the efficacy of SignViP in producing high-fidelity, visually coherent, and perceptually realistic sign language videos.
this section cite: ['b5', 'b6', 'b86', 'b60', 'b28', 'b46', 'b57', 'b55', 'b59', 'b58', 'b87', 'b28']

Section: Generative Capability Comparison for Video Diffusion Models.
To compare the generative capabilities of different video diffusion models, we evaluate three methods, which are ControlNet [88], AnimateAnyone [29], and our Sign Video Diffusion Model. As detailed in Table 4, our Sign Video Diffusion Model consistently outperforms other methods across all metrics. Specifically, our Hand SSIM outperforms others, highlighting our model's ability to preserve hand details. The results clearly highlight the superiority of our method.
Qualitative Comparison. We present the qualitative results in Figure 3(a) of the previous SLVG methods and our SignViP. Compared to the previous methods, SignViP generates higher-quality sign language videos while maintaining greater semantic accuracy with the spoken language text.
this section cite: ['b87', 'b28']

Section: Model Study
Identity Generalization. Figure 3(b) showcases how our SignViP generalizes signer identities by adapting appearance guidance from distinct reference images. This demonstrates the robustness of our method in preserving signer-specific appearances while ensuring accurate sign language translation.
this section cite: []

Section: Effect of Multiple Conditions.
To evaluate whether incorporating multiple fine-grained conditions improves video quality, we ablate the fine-grained poses and 3D hands from our pipeline, respectively. As shown in Table 5, removing one of the conditions leads to a substantial performance degradation across all metrics. These results demonstrate that incorporating multiple fine-grained conditions is essential for enhancing both the semantic accuracy and visual quality of the generated videos.
this section cite: []

Section: Effect of Compression.
To investigate the necessity of compression for SignViP, we conduct experiments to assess the impact of the compression/downsampling rate in the FSQ Autoencoder. As illustrated in Figure 4(a), the performance of back-translation improves notably as the compression rate increases. Notably, when the compression rate is set to 1 (i.e., no compression is applied), the model demonstrates significantly poor performance. These results underscore the critical role of compression in enhancing the effectiveness of SignViP.
this section cite: []

Section: Effect of Quantization.
To investigate the necessity of quantization for SignViP, we conduct an experiment where FSQ is not performed during FSQ Autoencoder, while Multi-Condition Token Translator is trained with continuous embedding prediction. As illustrated in Figure 3(c), we observed that continuous embedding prediction poses significant challenges for the translator, resulting in weak semantic alignment and low video quality. When incorporating FSQ, we achieve substantial improved performance. These findings highlight the importance of quantization for our SignViP.
this section cite: []

Section: Effect of Condition Augmentation.
To evaluate the impact of condition augmentation (Section 3.4) on generation quality, we conducted experiments by varying the augmentation probability p. Figure 4(b) presents the results of condition augmentation with varying values of p. Specifically, introducing a small probability of augmentation (i.e., p = 10 -3 ) slightly improves FVD and ROUGE-L scores, suggesting enhanced video quality and linguistic consistency. However, as p increases further, the effectiveness of condition augmentation diminishes. These results indicate that excessive augmentation introduces too much randomness, impacting both video quality and textual coherence.
this section cite: []

Section: Effect of Scheduled Sampling Strategy.
To evaluate the effect of varying the sampling ratio r of the scheduled sampling strategy (Section 3.6) on generation quality, we conducted experiments by varying r. The experimental results, as shown in Figure 4(c), reveal that the scheduled sampling strategy significantly impacts the quality of generated outputs. When r = 1, meaning all input tokens are replaced with random indices, the results indicate that excessive randomness severely hurts the model's consistency and coherence. As r decreases, generation quality improves steadily. The best performance is observed at r = 0.4. However, as r is further reduced to 0.2, performance begins to  1 and Table 2, respectively. "FVD" evaluates the protocol described in Table 3. decline slightly. This suggests that a very low replacement ratio is insufficient to simulate the diverse distributions encountered during inference, leading to suboptimal performance.
FSQ vs. VQ. To compare the performance of FSQ with traditional Vector Quantization (VQ), we evaluate both methods in terms of codebook usage efficiency and multi-conditional reconstruction loss.
(1) Codebook Usage. To assess the efficiency of codebook utilization, we conducted experiments with varying codebook sizes ranging from 2 7 to 2 12 , and the results are summarized in Figure 4(d). The results demonstrate that FSQ consistently achieves high codebook usage rates, remaining above 97% even with larger codebook. In contrast, VQ experiences a sharp decrease in usage as the codebook size increases. These findings highlight the stability and scalability of FSQ. (2) Reconstruction Loss.
To evaluate the ability of conditional preservation, we measured the reconstruction loss of both methods under different codebook sizes, as detailed in Figure 4(e). The results show that FSQ achieves consistently lower reconstruction loss compared to VQ, demonstrating its superior capability in preserving original conditional structure.
this section cite: []

Section: Conclusion
In this work, we propose SignViP, a novel Sign Language Video Generation (SLVG) framework that incorporates multiple fine-grained conditions to enhance generation fidelity by adopting a discrete tokenization paradigm. SignViP consists of three components: (1) Sign Video Diffusion Model, which learns continuous embeddings encapsulating fine-grained motion and appearance details, (2) FSQ Autoencoder, which compresses and quantizes these embeddings into discrete tokens for compact representation, and (3) Multi-Condition Token Translator, which translates spoken language text to discrete multi-condition tokens. Experimental results demonstrate that SignViP achieves state-of-the-art performance in video quality, temporal coherence, and semantic fidelity.
this section cite: []

Section: References
Ref_id:b0 Title: An open multilingual large language model for translation-related tasks Year: (2024)
Ref_id:b1 Title: Synthesizing images of humans in unseen poses Year: (2018)
Ref_id:b2 Title: Neural sign actors: a diffusion model for 3d sign language production from text Year: (2024)
Ref_id:b3 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b4 Title: Neural sign language translation Year: (2018)
Ref_id:b5 Title: Sign language transformers: Joint end-to-end sign language recognition and translation Year: (2020)
Ref_id:b6 Title: Realtime multi-person 2d pose estimation using part affinity fields Year: (2017)
Ref_id:b7 Title: Contrastive prompting enhances sentence embeddings in llms through inference-time steering Year: (2025)
Ref_id:b8 Title: Multimodal diffusion for embodied avatar synthesis Year: (2024)
Ref_id:b9 Title: Disentangled and controllable face image generation via 3d imitative-contrastive learning Year: (2020)
Ref_id:b10 Title: How2sign: a large-scale multimodal dataset for continuous american sign language Year: (2021)
Ref_id:b11 Title: Sigmoid-weighted linear units for neural network function approximation in reinforcement learning Year: (2018)
Ref_id:b12 Title: A formal evaluation of psnr as quality measurement parameter for image segmentation algorithms Year: (2016)
Ref_id:b13 Title: Skeleton-aware neural sign language translation Year: (2021)
Ref_id:b14 Title: Implicit locationcaption alignment via complementary masking for weakly-supervised dense video captioning Year: (2025)
Ref_id:b15 Title: Fine-grained alignment network for zero-shot cross-modal retrieval Year: (2025)
Ref_id:b16 Title:  Year: (2014)
Ref_id:b17 Title: Vector quantization Year: (1984)
Ref_id:b18 Title: Hallucinations in large multilingual translation models Year: (2023)
Ref_id:b19 Title: Hierarchical lstm for sign language translation Year: (2018)
Ref_id:b20 Title: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning Year: (2023)
Ref_id:b21 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b22 Title: Latent video diffusion models for high-fidelity long video generation Year: (2022)
Ref_id:b23 Title: Exploring human-like translation strategy with large language models Year: (2024)
Ref_id:b24 Title: Gaussian error linear units (gelus) Year: (2016)
Ref_id:b25 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b26 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b27 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b28 Title: Animate anyone: Consistent and controllable image-to-video synthesis for character animation Year: (2024)
Ref_id:b29 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b30 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b31 Title: Re-sign: Re-aligned end-to-end sequence modelling with deep recurrent cnn-hmms Year: (2017)
Ref_id:b32 Title: Config: Controllable neural face image generation Year: (2020)
Ref_id:b33 Title: Automatic evaluation of machine translation quality using longest common subsequence and skip-bigram statistics Year: (2004)
Ref_id:b34 Title: Gesture-to-gesture translation in the wild via category-independent conditional maps Year: (2019)
Ref_id:b35 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b36 Title: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models Year: (2022)
Ref_id:b37 Title: Pose guided person image generation Year: (2017)
Ref_id:b38 Title: Follow your pose: Pose-guided text-to-video generation using pose-free videos Year: (2024)
Ref_id:b39 Title: Controllable person image synthesis with attribute-decomposed gan Year: (2020)
Ref_id:b40 Title: Finite scalar quantization: Vq-vae made simple Year: (2023)
Ref_id:b41 Title: Neural sign language translation by learning tokenization Year: (2020)
Ref_id:b42 Title: Arc2face: A foundation model for id-consistent human faces Year: (2024)
Ref_id:b43 Title: Bleu: a method for automatic evaluation of machine translation Year: (2002)
Ref_id:b44 Title: Iterative alignment network for continuous sign language recognition Year: (2019)
Ref_id:b45 Title: Yolo5face: Why reinventing a face detector Year: (2022)
Ref_id:b46 Title: Signgen: End-to-end sign language video generation with latent diffusion Year: (2024)
Ref_id:b47 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b48 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b49 Title: Swish: a self-gated activation function Year: (2017)
Ref_id:b50 Title: Comet: A neural framework for mt evaluation Year: (2020)
Ref_id:b51 Title: Comet-22: Unbabel-ist 2022 submission for the metrics shared task Year: (2022)
Ref_id:b52 Title: Customize-a-video: One-shot motion customization of text-to-video diffusion models Year: (2024)
Ref_id:b53 Title: Stochastic backpropagation and approximate inference in deep generative models Year: (2014)
Ref_id:b54 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b55 Title: Adversarial training for multi-channel sign language production Year: (2020)
Ref_id:b56 Title: Everybody sign now: Translating spoken language to photo realistic sign language video Year: (2020)
Ref_id:b57 Title: Progressive transformers for end-to-end sign language production Year: (2020)
Ref_id:b58 Title: Mixed signals: Sign language production via a mixture of motion primitives Year: (2021)
Ref_id:b59 Title: Continuous 3d multi-channel sign language production via progressive transformers and mixture density networks Year: (2021)
Ref_id:b60 Title: Signing at scale: Learning to co-articulate signs for large-scale photo-realistic sign language production Year: (2022)
Ref_id:b61 Title: Imagpose: A unified conditional framework for pose-guided person generation Year: (2024)
Ref_id:b62 Title: Advancing pose-guided image synthesis with progressive conditional diffusion models Year: (2023)
Ref_id:b63 Title: Controllable image editing with consistent object quantity and layout Year: (2025)
Ref_id:b64 Title: Imagdressing-v1: Customizable virtual dressing Year: (2025)
Ref_id:b65 Title: Long-term talkingface generation via motion-prior conditional diffusion model Year: (2025)
Ref_id:b66 Title: Boosting consistency in story visualization with rich-contextual conditional diffusion models Year: (2025)
Ref_id:b67 Title: Imaggarment-1: Fine-grained garment generation for controllable fashion design Year: (2025)
Ref_id:b68 Title: Pose-guided fine-grained sign language video generation Year: (2024)
Ref_id:b69 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b70 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b71 Title: Consistency models Year: (2023)
Ref_id:b72 Title: Gesturegan for hand gesture-to-gesture translation in the wild Year: (2018)
Ref_id:b73 Title: Towards accurate generative models of video: A new metric & challenges Year: (2018)
Ref_id:b74 Title: Neural discrete representation learning Year: (2017)
Ref_id:b75 Title: Attention is all you need Year: (2017)
Ref_id:b76 Title: Conditional dropout for progressive training of portrait video generation Year: (2024)
Ref_id:b77 Title: Ensembling diffusion models via adaptive feature aggregation Year: (2025)
Ref_id:b78 Title:  Year: (2018)
Ref_id:b79 Title: Highresolution image synthesis and semantic manipulation with conditional gans Year: (2018)
Ref_id:b80 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b81 Title: Temporally consistent face reenactment with 3d geometric guidance Year: (2025)
Ref_id:b82 Title: G2p-ddm: Generating sign pose sequence from gloss sequence with discrete diffusion model Year: (2024)
Ref_id:b83 Title: Magicanimate: Temporally consistent human image animation using diffusion model Year: (2024)
Ref_id:b84 Title: Follow-your-pose v2: Multiple-condition guided character image animation for stable pose control Year: (2024)
Ref_id:b85 Title: T2s-gpt: Dynamic vector quantization for autoregressive sign language production from text Year: (2024)
Ref_id:b86 Title: Neural sign language synthesis: Words are our glosses Year: (2020)
Ref_id:b87 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b88 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b89 Title: Spatial-temporal multi-cue network for continuous sign language recognition Year: (2020)
Ref_id:b90 Title: Equip controllable character animation with realistic hands Year: (2024)
