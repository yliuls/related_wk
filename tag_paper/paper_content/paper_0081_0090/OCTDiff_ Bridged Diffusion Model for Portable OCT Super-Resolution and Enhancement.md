Title: OCTDiff: Bridged Diffusion Model for Portable OCT Super-Resolution and Enhancement
Abstract: Medical imaging super-resolution is critical for improving diagnostic utility and reducing costs, particularly for low-cost modalities such as portable Optical Coherence Tomography (OCT). We propose OCTDiff, a bridged diffusion model designed to enhance image resolution and quality from portable OCT devices. Our image-to-image diffusion framework addresses key challenges in the conditional generation process of denoising diffusion probabilistic models (DDPMs). We introduce Adaptive Noise Aggregation (ANA), a novel module to improve denoising dynamics within the reverse diffusion process. Additionally, we integrate Multi-Scale Cross-Attention (MSCA) into the U-Net backbone to capture local dependencies across spatial resolutions. To address overfitting on small clinical datasets and to preserve fine structural details essential for retinal diagnostics, we design a customized loss function guided by clinical quality scores. OCTDiff outperforms convolutional baselines and standard DDPMs, achieving state-of-the-art performance on clinical portable OCT datasets. Our model and its downstream applications have the potential to generalize to other medical imaging modalities and revolutionize the current workflow of ophthalmic diagnostics. The code is

Section: Introduction
Medical image analysis has been transformed by recent deep learning advances, including disease classification [1,2], tissue and cellular segmentation [3], and contrast enhancement [4]. However, most existing models are trained on data from high-end imaging systems, making them less effective and even unusable in low-resource or point-of-care settings. One critical domain with this limitation is ophthalmology, where high-quality Optical Coherence Tomography (OCT) systems can cost over 50,000 dollars and weigh more than 50 pounds, significantly reducing their accessibility in underdeveloped regions.
Convolutional neural networks (CNNs) and generative adversarial networks (GANs) have been widely applied for OCT image enhancement tasks including super-resolution (SR) and denoising. For instance, an early approach [5] built on ESRGAN [6] and MedGAN [7] demonstrated promising visual improvements on simulated low-resolution OCT images. However, it suffered from mode collapse and generated unwanted artifacts; hence, its generalizability to real-world clinical scans remains untested. CNN-based super-resolution methods have also been applied on other medical imaging modalities such as computed tomographpy (CT) and magnetic resonance imaging (MRI) [8,9]. Though quantitative performance improves, these models tend to hallucinate fine details and 39th Conference on Neural Information Processing Systems (NeurIPS 2025). fail to preserve delicate anatomical structures, which are essential for accurate diagnosis. Medical visual tasks for diagnostic assistance such as anatomical segmentation and motion correction have achieved clinically-usable performance [10,11], but SR remains challenging due to the inherent low contrast of low-resolution images and the scarcity of large-scale patient datasets with low-resolution and high-resolution pairs, despite well-established SR benchmarks on natural images [12].
Figure 1: Pipeline of our study. The clinical low-resolution dataset was captured with a portable OCT device and then underwent necessary filtering, preprocessing, and registration. The synthetic dataset [5] was obtained by matching the histogram of the commercial OCT data to that of the portable OCT data, convolving the resulting data with the point spread function (PSF) of the portable OCT, and then downsampling the data by a factor of 4 [6]. OCTDiff super-resolves both datasets and generates high-resolution outputs for downstream classification of ophthalmic diseases such as glaucoma and age related macular degeneration (AMD).
Diffusion-based generative modeling, i.e. denoising diffusion probabilistic models (DDPMs) [13], has demonstrated superior performance over convolutional models and GANs in terms of both generative quality and training stability [14,15]. DDPM variants such as latent diffusion models (LDMs) [16] and conditional DDPMs (CDMs) [17] extend the power of DDPMs by incorporating guided conditioning to generate more controlled outputs. However, they are still not guaranteed to reliably translate images between different domains when conditioning on a target image. A more recent approach, the bridged Brownian diffusion model (BBDM) [18], emphasizes conditioning by directly using the reference image as the initial point in the reverse diffusion process. This architecture is particularly well-suited for image-to-image tasks such as style transfer and semantic synthesis. Nevertheless, its potential for super-resolution and its effectiveness when training on medical images, which are small in dataset size and large in spatial resolution [19], remains relatively unexplored.
To address these limitations, we propose OCTDiff, an image-to-image super-resolution conditional diffusion model for portable OCT enhancement. OCTDiff builds upon the bridged framework originated from BBDM. We propose two novel components: (1) an Adaptive Noise Aggregation (ANA) algorithm that stabilizes the reverse denoising trajectory by aggregating noise predictions from previous time steps in the reverse process; and (2) a Multi-scale Cross-Attention (MSCA) UNet backbone that enables cross-resolution feature interaction between encoder and decoder to better capture both global anatomy and fine retinal details. (3) During model training, we also introduce a custom loss function with modulation from a clinical quality score to guide the model toward perceptually and diagnostically meaningful outputs.
In summary, our main contributions are:
• We propose OCTDiff, which significantly outperforms baseline methods in both quantitative metrics and qualitative structural fidelity on real-world OCT datasets. The computational efficiency and training speed are also preserved, as demonstrated in Section 4.1.
• We are the first to address the semantic misalignment issue [20] in conditional DDPMs through temporal fusion across denoising steps, enabling stronger conditioning guidance throughout the generation process. This strategy is uniquely feasible within the bridged diffusion framework, where the conditioning image directly anchors the reverse process.
• Clinically, OCTDiff pioneers a new direction at the intersection of AI and healthcare, tailored for ultra-low-resolution images from portable devices. Its utility in downstream classification tasks (Section 4.3) shows OCTDiff's potential to deliver affordable and reliable vision care to under-served and remote populations, and to generalize to other resource-constrained medical imaging settings.
2 Related Work
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b4', 'b5', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19']

Section: Heuristic Optimization in Diffusion Models
Early improvements in diffusion models relied heavily on heuristic strategies for loss weighting, noise scheduling, and sampling. Hybrid loss designs [21,22,23,24] combine pixel-wise noise prediction with latent-space objectives to balance reconstruction accuracy and perceptual realism. For noise scheduling, linear [25] or cosine β t schedules are commonly used, while more recent works [26,27] propose handcrafted tuning based on empirical settings that better align with DDPM's denoising capacity across timesteps. Additionally, timestep reweighting prioritizes intermediate steps where gradients are more stable [21,28,29,30]. These heuristic strategies improve sample diversity and convergence without modifying model architecture, but they are inherently static and task-agnostic, often relying on fixed schedules and an assumption that certain timesteps are more important than others. In contrast, our Adaptive Noise Aggregation (ANA) module in OCTDiff dynamically aggregates multiple denoising predictions across timesteps, assigning different noise schedules to different images. This temporal fusion captures complementary information from various noise levels that represent multi-scale features, which is conceptually similar to ensemble learning. ANA is only realizable within a bridged diffusion framework, being particularly effective for conditional tasks such as super-resolution and reconstruction for degraded OCT images.
Learnable noise modulation MuLAN [31] learns per-sample noise schedules by predicting the optimal noise scale σ for each input, while other methods [32,33] use auxiliary networks or learned noise embeddings to refine denoising behavior. ANA also aims to improve signal-noise alignment but achieves this through temporal fusion rather than direct modulation. Unlike MuLAN's learned noise schedules, ANA adapts noise dynamically per image without introducing additional supervision, making it well-suited for resource-constrained applications. While ANA may sacrifice flexibility in handling highly variable noise patterns compared to MuLAN, it offers a simpler and more efficient solution, particularly for OCT scans that require consistent, low-complexity adaptations.
this section cite: ['b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b20', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32']

Section: Attention and Diffusion
Recent works increasingly integrate attention mechanisms into diffusion models. Transformer-based architectures such as latent diffusion [16], ImageCraft [34], and CDM [35] incorporate self-and cross-attention to enhance spatial coherence. Other designs introduce hierarchical [36], spatially-aware [37], or multi-scale attention [38,39] to improve structure preservation while maintaining efficiency. Some models further combine attention with guidance signals (e.g., text, edge maps) for stronger control [40,41]. While prior multi-scale works stack features within encoder or decoder branches or process them in parallel at a single resolution, our MSCA introduces explicit cross-attention between encoder and decoder features at different scales, enabling context-aware guidance across resolutions.
this section cite: ['b15', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40']

Section: Temporal Ensembling and Recurrent Denoising
We introduce ANA as a regularizing prior across timesteps, inspired by previous aggregation algorithms. For example, temporal ensembling reduces prediction variance and enhances consistency in semi-supervised learning [42], and recurrent denoising benefits from multi-step integration to avoid error explosion [26]. The idea of combining global and local features in medical images also resembles multi-scale fusion of frequency components via weighted aggregation [43]. Our work is also distinctive from differential-equation-based methods [44] because ANA is discrete-time, controllable (with attenuation α) and operates during inference (not training). Also rather than modifying the noise sampler like DPM-Solver [45], ANA assumes a fixed noise schedule (e.g., linear or cosine) and aggregates the previously-sampled noise vectors.
this section cite: ['b41', 'b25', 'b42', 'b43', 'b44']

Section: Method

this section cite: []

Section: Adaptive Noise Aggregation
Our OCTDiff builds upon bridged diffusion [18] that learns the translation between two image domains directly through a bidirectional diffusion process. We propose Adaptive Noise Aggregation (ANA) strategy to improve the stability of the reverse denoising process. Noise predictions at different time steps t encode complementary information particularly for high-frequency retinal details. Aggregating different noise levels basically leverages different image scales' information. ANA with adaptive weights enhances fine structure reconstruction, especially when the input is severely degraded as in portable OCT scans.
The reverse process starts from the high-resolution condition xT = y, and iteratively estimates intermediate latent states xt to recover a clean image x0 over T denoising steps. At each step t, the model defines the reverse transition as:
p θ (x t | x t+1 , y) ∼ N (µ θ (x t+1 , t, y), Σ θ (x t+1 , t, y))(1)
where p θ is a learned Gaussian distribution with parameters predicted by the U-Net backbone. The noise component is estimated as εt = ε θ (x t+1 , t, y), where ε θ is the noise prediction network. This predicted noise εt is then used to reconstruct an intermediate clean estimate xt 0 , which subsequently informs the mean function µ θ (x t+1 , t, y) in the reverse transition.
To improve the robustness of the reverse process, our ANA algorithm does not rely solely on the current noise prediction. Instead, it adaptively aggregates noise predictions from all previous time steps in the reverse process {ε τ } T -1 τ =t using exponential decay. This temporal fusion yields a more stable and informative estimate εt . The aggregation is formally defined as:
εt = 1 Z t T -1 τ =t exp(-α(τ -t)) • ετ ,(2)
where the weight function w(τ, t) = exp(-α(τ -t)) introduces a time-decay prior, emphasizing predictions closer to step t while still leveraging long-range information. The scalar α > 0 controls how fast the weight decays over time. We discuss the impact of different α values in an ablation study (Section 4.2). The normalization term Z t ensures the aggregated weights sum to 1:
T -1 τ =t w(τ,t) Zt
= 1. This updated ϵ t forms soft temporal fusion of noise estimates. An intuitive explanation for ANA is the exponential moving average (EMA) [46]. We adapt the EMA principle to the spatially-structured noise tensor space, making ANA a temporally-aware ensemble over latent signals within diffusion. In OCTDiff, the forward process generates a sequence of interpolated images ("bridges") between the low-resolution input and high-resolution target. This bridging mechanism ensures all intermediate steps are structurally meaningful and semantically anchored. Therefore, aggregating noise predictions across time is more stable and informative here. On the contrary, in standard DDPMs, reverse steps are inherently noisy in early steps and often sensitive to prediction errors, which makes temporal aggregation less stable.
The proposed ANA strategy enables the model to leverage not only the current noise prediction but also aggregated predictions across future steps. This stabilizes the reverse trajectory and enhances final image quality. The ANA algorithm is summarized in Algorithm 1.
this section cite: ['b17', 'b45']

Section: Algorithm 1 Adaptive Noise Aggregation (ANA)
Require: High-resolution input xT = y, total steps T Ensure: Super-resolved output x0
1: for t = T -1 to 0 do 2:
εt ← ε θ (x t+1 , t) 3:
if t > T /2 then 4:
εt ← Refine(ε t , ∇L denoise )
5:
end if 6:
εt ← 1 Zt T -1 τ =t exp(-α(τ -t)) • ετ 7: xt 0 ← Reconstruct(x t+1 , εt , t) 8:
xt ← µ θ (x t+1 , xt 0 , t) 9: end for 10: return x0 Refinement Module: To improve robustness at early stages (t > T /2), we apply a gradientbased refinement:
εt ← εt -η • ∇ εt L denoise ,
where η is a small step size and L denoise denotes a pixel-level loss against a pseudo-ground truth.
Reconstruction Module: Given xt+1 and the aggregated noise εt , we use the DDIM [47] inversion rule:
xt 0 = 1 √ ᾱt xt+1 - √ 1 -ᾱt • εt ,to
this section cite: ['b46']

Section: Multi-Scale Cross Attention
We implement multi-scale cross-attention (MSCA) in the UNet backbone of our OCTDiff model. The MSCA enables encoder features at each scale to attend to decoder features at different scales. This cross-scale interaction is particularly important for our super-resolution task on OCT images. Medical image scans such as OCT often contain crucial diagnostic patterns in small, localized patches observed at different scales. For example, at a scale of 32 × 32, only the vessels may be visible, while at 128 × 128, the scan may show larger structures like the retina and tissue. The MSCA helps preserve fine retinal structures while capturing global context in OCT.
For each encoder query Q s enc at scale s ∈ S enc , the attention is computed with key-value pairs
(K s ′ dec , V s ′ dec ) from different scales s ′ ∈ S dec where s ′ ̸ = s.
The attention is computed by the softmax of the scaled dot product between the query and the key. The output of the attention is then added to the original query Q s enc to preserve the residual connection. (Eq. ( 3)) depicts the MSCA mechanism:
Qenc = s∈Senc s ′ ∈Sdec s ′ ̸ =s Q s enc + Softmax Q s enc (K s ′ dec ) ⊤ √ C V s ′ dec (3
)
this section cite: []

Section: Loss Function with Clinical Quality Score
Due to the physical constraints of portable OCT devices, some acquired OCT scans are highly degraded. We thus incorporate clinical expert knowledge into model training by introducing a quality-aware loss function. Each high-resolution training image is assigned a perceptual quality score S (i) quality derived from subjective ratings provided by ophthalmologists. These ratings (e.g., from 1 to 10) are aggregated via voting and normalized to form a continuous score, reflecting the perceived clinical value of each target OCT image. We design a focal-style loss that enables high-quality scans to have a larger impact during model training. This is formulated as a perceptual quality modulated mean squared error:
MSE focal = 1 N N i=1 1 -S (i) quality γ • (x i -xi ) 2(4)
Here, γ < 0 is a focusing parameter that controls the degree to which high-quality samples are prioritized, usually being set in [-2, -1]. The modulation term (1 -S (i) quality ) γ prevents the model from being confused by relatively suboptimal OCT data.
this section cite: []

Section: Datasets and Experiments
Synthetic 500 Dataset We use two medical OCT datasets in our experiments: one synthetic and one clinical. The synthetic dataset is referred to as the Synthetic 500 dataset and contains 524 pairs of high-resolution / simulated low-resolution OCT B-scans; this dataset was designed to simulate the imaging characteristics of portable OCT devices. The high-resolution images were acquired from a Carl Zeiss © Cirrus HD-OCT 5000 machine and serve as ground truth. The low-resolution scans were simulated by first matching the high-resolution scans with the intensity histogram of low-resolution scans, then convolving the histogram-matched scans with the point spread function (PSF) of a portable OCT device, and finally downsampling the scans by a factor of 4 as done in ESRGAN [6] experiments. The Synthetic 500 dataset serves as a controlled setting for proof-of-concept evaluation and comparison with baselines to demonstrate the superiority of OCTDiff.
Philophos 84 Dataset Due to the lack of publicly available low-resolution OCT datasets, we collected our own real-world dataset with the Philophos © KUOS-O100 portable OCT device. OCT B-scans were captured from patients who visited Columbia Ophthalmology from May through July 2024. To obtain paired high-resolution scans under consistent physical and clinical conditions, each participant also underwent an additional OCT scan using a commercial device (Zeiss Cirrus 5000) during the same visit. The portable device has significantly limited imaging capabilities: shallower imaging depth of 2.0 mm (commercial: 2.9 mm), smaller field of view 6.5 × 6.5 mm (commercial: >8 mm), and reduced spatial resolution of 562 × 1286 (commercial: >3k). These hardware limitations result in low-resolution, high-noise, low-contrast images, often with offcentered or partially missing retinal structures. We first filtered out technically corrupted or clinically irrelevant scans, then applied an unsupervised denoising approach [48] to all images. Next, affine registration was performed using ImFusion© software [49] to align remaining scans to a standard OCT template [50] and resizing to 256 for model training. Then we conducted augmentation including flipping, scaling, rotation, elastic deformation, and contrast enhancement. After preprocessing, we obtained 504 paired B-scans from 84 patients, forming the Philophos 84 Dataset that presents realistic challenges for low-quality portable OCT enhancement. OCTDiff is proposed to faithfully superresolve these degraded scans from "unusable" to "usable", thereby making portable OCT devices clinically valuable.
this section cite: ['b5', 'b47', 'b48', 'b49']

Section: Results
We present quantitative and qualitative results of our OCTDiff against baseline models. We conduct ablation studies to analyze the effects of the ANA and MSCA modules and the quality-score informed loss on model performance, complexity, and training efficiency, including different cross-attention types and the exponential decay rate α in ANA. To demonstrate real-world applicability, we perform downstream disease classification using images generated by OCTDiff, comparing results with those from original low-resolution and target high-resolution scans. Finally, we discuss the limitations of our approach and directions for future work.  [51], VDSR [52], CycleGAN [53], Swin2SR [54], CDM [35], and BBDM [18]. Red boxes highlight regions with notable degradation or artifacts compared to GT. Each image is annotated with its BRISQUE score [55] to quantify perceptual quality. OCTDiff is trained and tested separately on the Philophos 84 and Synthetic 500 dataset. Training is conducted on a Lambda Labs Vector server equipped with two NVIDIA©A6000 GPUs, requiring approximately 48 hours to complete from scratch with input images resized to 256×256 pixels and a total diffusion time step T = 1000. For quantitative evaluation, we employ structural similarity index measure (SSIM) [56], peak signal-to-noise ratio (PSNR), and Learned Perceptual Image Patch Similarity (LPIPS) [57] to comprehensively assess reconstruction fidelity, pixel-level accuracy, and perceptual quality, respectively.
The baseline convolutional models include SRCNN [51] and VDSR [52]. Since previous work [5] on the Synthetic 500 dataset implemented ESRGAN and MedGAN, we include CycleGAN [53] for comparison. Swin2SR is chosen as a state-of-the-art transformer-based super-resolution model leveraging hierarchical self-attention and shifted windows to compare with our MSCA strategy. Regarding diffusion models, we implement CDM and the bridged model BBDM [18] as a foundation for our work. All models are trained from scratch under the same training strategy without pretraining to ensure a fair comparison.
Quantitative Results Our OCTDiff achieves the best performance in terms of SSIM (0.936 and 0.989 on the Philophos 84 and Synthetic 500 datasets, respectively) and attains the highest PSNR of 38.8 on the Philophos 84 dataset, as shown in Table 1. Most baseline models perform well on the Synthetic 500 dataset and achieve over 0.95 SSIM, which is comparable to OCTDiff, with BBDM reaching the highest PSNR of 42.7. This is likely because the Synthetic 500 dataset preserves scaling and local structural information during the synthesis process from high-resolution to low-resolution images, making the super-resolution task relatively easier for models employing local upsampling techniques such as convolutional kernels. Model performances drop on the Philophos 84 dataset, while our OCTDiff outperforms all baselines.
this section cite: ['b50', 'b51', 'b52', 'b53', 'b34', 'b17', 'b54', 'b55', 'b56', 'b50', 'b51', 'b4', 'b52', 'b17']

Section: Ablation Study
Impact of ANA and MSCA Modules To evaluate the contribution of the ANA and MSCA modules, we tested on different combinations of them and measured their impact on model performance, size (number of trainable parameters), and training cost (in FLOPs) [58], as summarized in Table 2.
Another example of output image and corresponding residual maps compared to the ground truth are shown in Figure 5. While ANA does not visibly alter the residual ratio, it effectively suppresses fundamental errors (i.e., the discontinuities in retinal layers) that occur when only MSCA is present.
ANA introduces no additional trainable parameters but increases FLOPs, offering a trade-off that results in SSIM increase. In contrast, MSCA yields more modest performance improvements but is crucial for maintaining spatial consistency, particularly for medical images that require structural integrity for diagnosis. In summary, ANA serves as the primary driver of performance gain and MSCA provides structural regularization. Both components together give rise to the superior performance of OCTDiff.
this section cite: ['b57']

Section: Weight Decay Factor
Another key hyperparameter in ANA is the exponential decay rate α, which controls how quickly the adaptive noise modulation weights diminish as introduced in Section 3.1. Table 3 reports the results with varying α values. When α is low (e.g., 0.1), the model keeps dependency on noise from earlier time steps over a longer duration and yields relatively high SSIM but slightly lower PSNR, reflecting good structural preservation but less sharpness. Increasing α further (e.g., 1.0) makes the model focus on the most recent two to three steps. This over-smoothing causes loss of fine structural details. This trade-off suggests that a moderate α achieves the best balance, and thus we selected α = 0.3 for our experiments.
this section cite: []

Section: Choices of Cross Attention
To justify our MSCA design of using encoder features as queries to attend to decoder features, we provide an empirical analysis to compare different cross attentions (CA), including unidirectional and bidirectional CA. The results are reported in Table 4.
Reverse CA (Decoder→Encoder) performed marginally worse though it converges faster, possibly due to the decoder lacking detailed structural localization at early stages that makes the query less efficient. Bidirectional CA is computationally expensive (twice as many attention layers per scale), with limited benefits. Unidirectional CA (Encoder→Decoder) yields the most significant improvement. The cost and scalability are also key impact factors for our decision when connecting two scales, as our ultimate goal is to deploy OCTDiff onto clinical OCT devices to achieve real-time image processing. We thus prioritized encoder-to-decoder attention in our task.  Quality Score in Loss Function To quantify how much the clinical input contributed to the overall performance, we compared all metrics with and without quality-aware loss as illustrated in 3.3. The results are in Table 5; the quality score significantly improved performance. This highlights how domain knowledge is required for the perceptual and structural understanding of OCT images.
this section cite: []

Section: Downstream Disease Classification
Table 6: Accuracy (%) on downstream disease classification using high-resolution images from commercial OCT, low-resolution images from portable OCT, images generated from our OCTDiff and BBDM models, in columns 1, 2, 3 and 4, respectively. The * indicates p-value ≤ 0.05 compared to the OCTDiff-generated input. The numbers are in the format of mean ± standard deviation. Disease Class Model High Res. Low Res. OCTDiff BBDM Glaucoma ViT 74.4 ±1.9 50.1 ±1.8* 74.5 ±3.1 62.0 ±2.5* CNN2D 93.4 ±5.0 83.3 ±2.2* 93.5 ±0.5 81.0 ±1.0* SwinT 75.5 ±2.6* 49.5 ±3.1* 55.6 ±4.3 57.1 ±3.2 AMD ViT 86.5 ±1.9 48.9 ±2.7* 85.8 ±1.6 79.5 ±2.1 CNN2D 94.6 ±0.9* 83.0 ±1.4* 96.4 ±0.4 91.8 ±0.7* SwinT 82.3 ±1.4* 49.4 ±2.4* 66.3 ±1.8 64.9 ±2.6
We perform downstream disease classification using images generated by OCTDiff to directly demonstrate its real-world utility in clinical applications. To reduce model bias, we trained three classification architectures: ViT [59], vanilla CNN, and SwinT [60] on three types of inputs: (1) high-resolution images from commercial OCT devices, (2) low-resolution images from portable OCT devices, (3) OCTDiff super-resolved images, and (4) BBDM super-resolved images. The pretrained weights on ImageNet1K [61] are imported. We performed 5-fold cross-validation for each model-dataset pair, with the results shown in Table 6, and we conducted Mann-Whitney U tests [62] to assess statistical significance. This evaluation was conducted independently for two representative ophthalmic tasks: glaucoma diagnosis and age-related macular degeneration (AMD) classification, both of which are widely studied in AI ophthalmology [63,64,65,66,67], as they are the top two causes of blindness worldwide.
Among the six models trained on high-resolution images, three showed no statistically significant difference, with the simplest vanilla CNN performing equally well in accuracy compared to counterparts trained on OCTDiff-generated images. In contrast, models trained on low-resolution portable OCT images exhibited a statistically significant drop. BBDM is selected as a representative of SR baselines, which is outperformed by OCTDiff in 5 out of 6 rows. These comparisons highlight OCTDiff's ability to transform previously suboptimal portable OCT scans into diagnostically reliable images, indicating that our method can closely match gold-standard OCT-image quality, or even exceed the performance achieved by commercial high-resolution scans.
this section cite: ['b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66']

Section: Limitations and Future Work
While OCTDiff demonstrates SOTA performance within each clinical dataset, the cross-dataset generalization, such as training on Synthetic 500 and testing on Philophos 84 and vice versa, is still under investigation due to the limited amount of data in each set. While OCTDiff is not positioned to be a general-purpose SR benchmark, further experiments on widely used natural image SR datasets are still needed to more comprehensively prove the advantages of the OCTDiff algorithm. Additionally, although OCTDiff is effective, the model remains relatively large and computationally demanding, leading to longer training times. The cross-scale attention mechanism shows promising benefits, but its performance is sensitive to the choice of scale combinations. Currently, only scales of 128, 64, and 32 have been explored.
Future work will focus on overcoming current limitations by exploring latent-space methods [16] such as LDM, pre-training the model using natural images, and cross-dataset evaluations to further validate OCTDiff's generalizability. Toward system-level advancement, future work will include integrating OCTDiff into physical portable OCT devices using edge computing platforms like NVIDIA©Jetson Orin Nano [68] to enable real-time image enhancement for point-of-care clinical applications. Safeguarding against hallucination in generated OCT images will be addressed in future work via posthoc clinical quality assessment.
We plan to extend OCTDiff to other medical imaging and broader fields. OCT shares core signal degradation characteristics, particularly speckle noise, with radiological imaging modalities such as ultrasound and low-dose CT. Also our algorithm's novel modules (ANA and MSCA) are not restricted to a specific image type. For these reasons, OCTDiff has potential for cross-modal generalization. There are no publicly available portable OCT datasets to our knowledge, and portable medical datasets are very rare in general, indicating the paradigm-shifting role of AI's entry into this field. Our work paves the way for the growing trend of developing tandem AI + imaging technology: making AI algorithms more scalable, interoperable, and easier to deploy within a portable form factor, enabling accessibility for the broadest populations at point-of-care.
this section cite: ['b15', 'b67']

Section: Conclusion
We propose OCTDiff, a bridged diffusion framework specifically designed for enhancing portable OCT images. We further propose Adaptive Noise Aggregation (ANA) to improve noise scheduling within the diffusion process, allowing the model to adaptively leverage information from multiple time steps.We incorporate Multi-Scale Cross-Attention (MSCA) to effectively capture spatial dependencies at multiple resolutions. To mitigate overfitting on limited clinical datasets and preserve diagnostically critical structures, we introduce a customized loss function guided by clinical quality scores. Downstream disease classification demonstrates that OCTDiff significantly improves image quality and makes low-cost OCT scans clinically usable. Our work lays the groundwork to revolutionize the current workflow of ophthalmic diagnostics with AI's power, making it more accessible and cost-effective, ultimately improving healthcare outcomes.
NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We clearly state our proposal of OCTDiff, a bridged diffusion model for superresolution and enhancement of portable OCT images, along with the introduction of key components such as Adaptive Noise Aggregation (ANA) 3.1, Multi-Scale Cross-Attention (MSCA) 3.2 and a customized loss function 3.3 with clinical quality score. The results include improved image quality, quantitative metrics 4.1, downstream clinical applications 4.3 and ablation studies 4.2. The claims in the abstract are well supported by our results and align with the content presented throughout the paper. Guidelines: • The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We explicitly acknowledge several limitations in Section 4.4.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [NA] Justification: Our work is primarily empirical and methodological, focusing on developing and evaluating the OCTDiff model for OCT image super-resolution and enhancement. It does not present new theoretical results that require formal proofs.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: We provide all necessary details on the model architecture in Figures 2 and 3, training procedure including dataset splits, hyperparameters (including the weight decay factor α), and evaluation protocols. This will enable reproduction of the main experimental results supporting the paper's claims. Additionally, the code and sample data (through figures in this publication) are now being made publicly available in this camera-ready version.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: We emphasized and evaluated the significant societal impacts of our work in AI healcare (Section 1, 4.3, 4.4 and 5). OCTDiff has the potential to revolutionize the current workflow of ophthalmic diagnostics with AI's power, making it more accessible and cost-effective, ultimately improving industrial healthcare outcomes.
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
Answer: [Yes] Justification: Our work focuses on medical image enhancement and does not involve highrisk generative models such as large language models or general-purpose image generators. The model is intended solely for clinical research use. The data used are de-identified and subject to IRB protocols, and the release of code will be accompanied by clear usage disclaimers to prevent misuse outside intended medical and academic contexts.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
this section cite: []

Section: 
(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [Yes] Justification: The code URL is being provided in this camera-ready version of the paper, with clear instructions and documentation. The complete clinical dataset is currently under IRB restrictions and ongoing collection, and will be made publicly available in a separate publication.
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
Answer: [Yes] Justification: We clearly specify the training details including the data splits, α hyperparameter settings in Section 4.2, GPU usage and rationale behind their selection. These details provide sufficient transparency to understand and reproduce the results. Other details such as batch size and optimizer option will be upon each user's preference.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: We report statistical significance in Section 4.3, and we have updated the results using the Mann-Whitney U test in this camera-ready version.
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
Answer: [Yes] Justification: We specify computational cost of models (Section 4.2) and hardware specifications such as GPU (Section 4.2) and training time (Section 4.2).
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: Our work complies fully with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All third-party assets used in the paper, including OCT imaging devices (Philophos and Zeiss) pretrained models (e.g., ViT, SwinT) and datasets (e.g., ImageNet1K), are properly cited with corresponding references. Their usage complies with the original licenses.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] Justification: The new assets introduced (e.g. the OCTDiff model, the Adaptive Noise Aggregation (ANA) module, and a clinical-quality-guided loss function) are thoroughly documented in the main text. In this camera-ready paper, we are now releasing the full source code with comprehensive instructions. A sample dataset will also be provided, while the full clinical dataset will be released separately due to IRB constraints Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: Our work does not involve crowdsourcing or direct research with human subjects. All human-related data (retinal OCT scans) were retrospectively collected under IRB-approved clinical protocols, with appropriate anonymization and ethical safeguards. No compensation was provided, and no participants were recruited specifically for this study.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [Yes] Justification: All clinical OCT data used in our study were collected under Institutional Review Board (IRB) approval number IRB-AAAV1311, in accordance with ethical standards for research involving human subjects. The data were fully deidentified before AI training. No additional risks were posed to participants beyond standard OCT scanning, and informed consent was obtained as part of the data collection protocol. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: Our work does not employ large language models (LLMs) as part of its core methodology or experimental process. Any use of LLMs was limited to writing and editing assistance that does not affect the scientific novelty or originality of our research. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Deep convolutional neural network based medical image classification for disease diagnosis Year: (2019)
Ref_id:b1 Title: Medical image classification with convolutional neural network Year: (2014)
Ref_id:b2 Title: Medical image segmentation using deep learning: A survey Year: (2022)
Ref_id:b3 Title: Medical image contrast enhancement using range limited weighted histogram equalization Year: (2018)
Ref_id:b4 Title: Enhancing portable OCT image quality via GANs for AI-based eye disease detection Year: (2022)
Ref_id:b5 Title: Esrgan: Enhanced super-resolution generative adversarial networks Year: (2018)
Ref_id:b6 Title: MedGAN: Medical image translation using GANs Year: (2020)
Ref_id:b7 Title: 3D-MRI super-resolution reconstruction using multi-modality based on multiresolution CNN Year: (2024)
Ref_id:b8 Title: Computed tomography super-resolution using deep convolutional neural network Year: (2018)
Ref_id:b9 Title: Low-dose CT denoising with convolutional neural network Year: (2017)
Ref_id:b10 Title: Motionttt: 2d test-time-training motion estimation for 3d motion corrected mri Year: (2024)
Ref_id:b11 Title: Binarized diffusion model for image super-resolution Year: (2024)
Ref_id:b12 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b13 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b14 Title: Diffusion model as representation learner Year: (2023)
Ref_id:b15 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b16 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b17 Title: Bbdm: Image-to-image translation with brownian bridge diffusion models Year: (2023)
Ref_id:b18 Title: Super-resolution in medical imaging Year: (2009)
Ref_id:b19 Title: Sdedit: Guided image synthesis and editing with stochastic differential equations Year: (2021)
Ref_id:b20 Title: Improved denoising diffusion probabilistic models Year: ()
Ref_id:b21 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b22 Title: Improved Image Denoising: A Combination Method Using Multiscale Contextual Fusion and Recursive Learning Year: (2025)
Ref_id:b23 Title: Diffusion model with perceptual loss Year: (2023)
Ref_id:b24 Title: Denoising Diffusion Probabilistic Models Year: (2020)
Ref_id:b25 Title: Elucidating the Design Space of Diffusion-Based Generative Models Year: (2022)
Ref_id:b26 Title: Efficient diffusion models: A survey Year: (2025)
Ref_id:b27 Title: Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation Year: (2024)
Ref_id:b28 Title: Pruning then reweighting: Towards data-efficient training of diffusion models Year: (2025)
Ref_id:b29 Title: Towards more accurate diffusion model acceleration with a timestep tuner Year: (2024)
Ref_id:b30 Title: Diffusion models with learned adaptive noise Year: (2024)
Ref_id:b31 Title: Denoising diffusion path: Attribution noise reduction with an auxiliary diffusion model Year: (2024)
Ref_id:b32 Title: Sifting through the noise: A survey of diffusion probabilistic models and their applications to biomolecules Year: (2025)
Ref_id:b33 Title: ImageCraft-Text To Image Synthesis Year: (2024)
Ref_id:b34 Title: Palette: Image-to-image diffusion models Year: (2022)
Ref_id:b35 Title: Cascaded diffusion models for high fidelity image generation Year: (2022)
Ref_id:b36 Title: Spatial-aware latent initialization for controllable image generation Year: (2024)
Ref_id:b37 Title: Diffusioninst: Diffusion model for instance segmentation Year: (2024)
Ref_id:b38 Title: Control-a-video: Controllable text-to-video diffusion models with motion prior and reward feedback learning Year: (2023)
Ref_id:b39 Title: eDiffi: Text-to-Image Diffusion Models with Editable Outputs Year: (2022)
Ref_id:b40 Title: Masked-attention diffusion guidance for spatially controlling text-to-image generation Year: (2024)
Ref_id:b41 Title: Temporal ensembling for semi-supervised learning Year: (2016)
Ref_id:b42 Title: HiFuse: Hierarchical multi-scale feature fusion network for medical image classification Year: (2024)
Ref_id:b43 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b44 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b45 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b46 Title: Denoising diffusion implicit models Year: ()
Ref_id:b47 Title: Unsupervised denoising of retinal OCT with diffusion probabilistic model Year: (2022)
Ref_id:b48 Title: Recent advances in point-of-care ultrasound using the ImFusion Suite for real-time image analysis Year: (2018)
Ref_id:b49 Title: Retinal OCT image registration: methods and applications Year: (2021)
Ref_id:b50 Title: Learning a deep convolutional network for image super-resolution Year: (2014)
Ref_id:b51 Title: Accurate image super-resolution using very deep convolutional networks Year: (2016)
Ref_id:b52 Title: Unpaired image-to-image translation using cycle-consistent adversarial networks Year: (2017)
Ref_id:b53 Title: Swin2sr: Swinv2 transformer for compressed image super-resolution and restoration Year: (2022)
Ref_id:b54 Title: No-reference image quality assessment in the spatial domain Year: (2012)
Ref_id:b55 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b56 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b57 Title: Flops as a direct optimization objective for learning sparse neural networks Year: (2018)
Ref_id:b58 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b59 Title: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows Year: (2021)
Ref_id:b60 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b61 Title: On a test of whether one of two random variables is stochastically larger than the other Year: (1947)
Ref_id:b62 Title: Deep learning in ophthalmology: The technical and clinical considerations Year: (2019)
Ref_id:b63 Title: A deep learning algorithm for prediction of age-related eye disease study severity scale for age-related macular degeneration from color fundus photography Year: (2018)
Ref_id:b64 Title: Identifying medical diagnoses and treatable diseases by image-based deep learning Year: (2018)
Ref_id:b65 Title: Deep learning for prediction of AMD progression: a pilot study Year: (2019)
Ref_id:b66 Title: Attention based glaucoma detection: A large-scale database and CNN model Year: (2019)
Ref_id:b67 Title: Introduction to nvidia jetson nano Year: (2020)
Ref_id:b68 Title: A Technical Appendices and Supplementary Material Supplementaty materials will be uploaded in a different PDF file Year: ()
