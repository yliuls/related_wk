Title: Boosting Generative Image Modeling via Joint Image-Feature Synthesis
Abstract: Our generative image modeling framework bridges the gap between generative modeling and representation learning by leveraging a diffusion model that jointly captures low-level image details (via VAE latents) and high-level semantic features (via DINOv2). Trained to generate coherent image-feature pairs from pure noise, this unified latent-semantic dual-space diffusion approach significantly boosts both generative quality and training convergence speed.

Section: Introduction
Latent diffusion models (LDMs) (Rombach et al., 2022) have emerged as a leading approach for high-quality image synthesis, achieving state-of-the-art results (Rombach et al., 2022;Peebles & Xie, 2023;Ma et al., 2024). These models operate in two stages: first, a variational autoencoder (VAE) compresses images into a compact latent representation (Rombach et al., 2022); second, a diffusion model learns the distribution of these latents, capturing their underlying structure.
Leveraging their intermediate features, pretrained LDMs have shown promise for various scene understanding tasks, including classification (Mukhopadhyay et al., 2023), pose estimation (Gong et al., 2023), and segmentation (Li et al., 2023b;Liu et al., 2023;Delatolas et al., 2025). However, their discriminative capabilities typically underperform specialized (self-supervised) representation learning approaches like masking-based (He et al., 2022), contrastive (Chen et al., 2020), selfdistillation (Caron et al., 2021), or vision-language contrastive (Radford et al., 2021a) methods. This limitation stems from the inherent tension in LDM training -the need to maintain precise low-level reconstruction while simultaneously developing semantically meaningful representations.
This observation raises a fundamental question: How can we leverage representation learning to enhance generative modeling? Recent work by Yu et al. (2025) (REPA) demonstrates that improving the semantic quality of diffusion features through distillation of pretrained self-supervised representations leads to better generation quality and faster convergence. Their results establish a clear connection between representation learning and generative performance.
Motivated by these insights, we investigate whether a more effective approach to leveraging representation learning can further enhance image generation performance. In this work, we contend that the answer is yes: rather than aligning diffusion features with external representations via distillation, we propose to jointly model both images (specifically their VAE latents) and their high-level semantic features extracted from a pretrained vision encoder (e.g., DINOv2 (Oquab et al., 2024)) within the same diffusion process. Formally, as shown in Figure 1, we define the forward diffusion process as q(x t , z t |x t-1 , z t-1 ) for t = 1, ..., T , where x 0 = x and z 0 = z are the clean VAE latents and semantic features, respectively. The reverse process p θ (x t-1 , z t-1 |x t , z t ) learns to gradually denoise both modalities from Gaussian noise. This joint modeling approach forces the diffusion model to explicitly learn the joint distribution of both precise low-level (VAE) and high-level semantic (DINOv2) features. We implement this approach, called ReDi (Representation Diffusion), within the DiT (Peebles & Xie, 2023) and SiT (Ma et al., 2024) frameworks with minimal modifications to their transformer architecture: we apply standard diffusion noise to both representations, combine them into a single set of tokens, and train the standard diffusion transformer architecture to denoise both components simultaneously.
Compared to REPA, our joint modeling approach offers three key advantages. First, the diffusion process explicitly models both low-level and semantic features, enabling direct integration of these complementary representations. Second, our method simplifies training by eliminating the need for additional distillation objectives. Finally, during inference, our unified approach enables Representation Guidance -where the model uses its learned semantic understanding to iteratively refine generated images, improving quality in both conditional and unconditional generation.
Our contributions can be summarized as follows:
1. We propose ReDi, a novel and effective method that jointly models image-compressed latents and semantically rich representations within the diffusion process, significantly improving image synthesis performance. 2. We provide a concrete implementation of our approach for both diffusion (DiT) and flowmatching (SiT) frameworks, leveraging DINOv2 (Oquab et al., 2024) as the source of highquality semantic representations. 3. We also introduce Representation Guidance, which leverages the model's semantic predictions during inference to refine outputs, further enhancing image generation quality. 4. We demonstrate that our approach boosts performance in both conditional and unconditional generation, while significantly accelerating convergence (see Figure 2).
this section cite: ['b50', 'b50', 'b47', 'b39', 'b50', 'b42', 'b21', 'b37', 'b12', 'b24', 'b9', 'b6', 'b64', 'b46', 'b47', 'b39', 'b46']

Section: Related work
Representation Learning. Various approaches aim to learn meaningful representations for downstream tasks, with self-supervised learning emerging as one of the most promising directions. Early approaches employed pretext tasks such as predicting image patch permutations (Noroozi & Favaro, 2016) or rotation angles (Gidaris et al., 2018), while more recent methods utilize contrastive learning (Chen et al., 2020;Van den Oord et al., 2018;Misra & Maaten, 2020), clustering-based objectives (Caron et al., 2020(Caron et al., , 2018(Caron et al., , 2019)), and self-distillation techniques (Grill et al., 2020;Chen & He, 2021;Caron et al., 2021;Gidaris et al., 2021). The introduction of transformers enabled Masked Image Modeling (MIM), introduced by BEiT (Bao et al., 2022) and evolved through SimMIM (Xie et al., 2022), MAE He et al. (2022), AttMask (Kakogeorgiou et al., 2022), iBOT (Zhou et al., 2022), and MOCA (Gidaris et al., 2024), with DINOv2 (Oquab et al., 2024) achieving state-of-the-art performance through scaled models and datasets. Separately, contrastive vision-language pretraining, initiated by CLIP (Radford et al., 2021a), established powerful joint image-text representations. Subsequent models like SigLIP Zhai et al. (2023) and SigLIPv2 (Tschannen et al., 2025) refined this framework through enhanced training techniques, excelling in zero-shot settings and image retrieval (Kordopatis-Zilos et al., 2025). Building on these advances, we leverage pretrained DINOv2 visual representations to enhance image generative modeling performance.
this section cite: ['b45', 'b18', 'b9', 'b57', 'b40', 'b5', 'b3', 'b4', 'b22', 'b10', 'b6', 'b19', 'b1', 'b60', 'b29', 'b20', 'b46', 'b55', 'b31']

Section: Diffusion Models and Representation Learning
Due to the success of diffusion models, many recent works leverage representations learned from pre-trained diffusion models for downstream tasks (Fuest et al., 2024). In particular, intermediate U-Net (Ronneberger et al., 2015) features have been shown to capture rich semantic information, enabling tasks such as semantic segmentation (Baranchuk et al., 2022;Zhao et al., 2023), semantic correspondence (Luo et al., 2023;Zhang et al., 2023;Hedlin et al., 2023), depth estimation (Zhao et al., 2023), and image editing (Tumanyan et al., 2023). Furthermore, diffusion models have been used for knowledge transfer by distilling learned representations through teacher-student frameworks (Li et al., 2023a) or refining them via reinforcement learning (Yang & Wang, 2023). Other works have shown that diffusion models learn strong discriminative features that can be leveraged for classification (Mukhopadhyay et al., 2023;Xiang et al., 2023). In a complementary direction, REPA (Yu et al., 2025) recently demonstrated that aligning the internal representations of DiT (Peebles & Xie, 2023) with a powerful pre-trained visual encoder during training significantly improves generative performance. Motivated by this observation, we propose to integrate images and semantic representations into a joint learning process.
Multi-modal Generative Modeling Unifying the generation across diverse modalities has recently attracted widespread interest. Notably, CoDi (Tang et al., 2023) leverages a diffusion model that enables generation across text, image, video, and audio in an aligned latent space. A joint representation for different modalities has been shown to have great scalability properties (Mizrahi et al., 2023). For video generation, WVD (Zhang et al., 2024) incorporates explicit 3D supervision by learning the joint distribution of RGB and XYZ frames. To capture richer spatial semantics, GEM (Hassan et al., 2024) generates paired images and depth maps. MT-Diffusion (Chen et al., 2024) learns to incorporate various multi-modal data types with a multitask loss including CLIP (Radford et al., 2021b) image representations. However, they do not quantitatively assess how this impacts the generative performance. VideoJam (Chefer et al., 2025) models a joint image-motion representation that boosts temporal coherence and introduces a theoretically motivated Classifier-Free Guidance (CFG) Ho & Salimans (2022) variant to condition on both motion and text. Inspired by this approach and building on the standard CFG framework, we propose Representation Guidance, incorporating the visual representations as an additional guidance signal during inference.
3 Method
this section cite: ['b16', 'b51', 'b2', 'b69', 'b38', 'b67', 'b25', 'b69', 'b56', 'b42', 'b59', 'b64', 'b47', 'b53', 'b41', 'b68', 'b23', 'b8', 'b7', 'b27']

Section: Preliminaries
Denoising Diffusion Probabilistic Models (DDPM) Diffusion models (Ho et al., 2020) generate data by gradually denoising a noisy input. The forward process corrupts an input x 0 (e.g., an image or its VAE latent) over T steps by adding Gaussian noise:
x t = √ ᾱt x 0 + √ 1 -ᾱt ϵ,(1)
where x t is the noisy input at step t, ᾱt are constants that define the noise schedule, and ϵ ∼ N (0, I) is the Gaussian noise term. Following Ho et al. (2020), the reverse process learns to denoise x t by predicting the added noise ϵ using a network ϵ θ (•) with parameters θ. The training objective is:
L simple = E x0,ϵ,t ∥ϵ θ (x t , t) -ϵ∥ 2 .(2)
Although we also include the variational lower bound loss from Nichol & Dhariwal (2021) to learn the variance of the reverse process, we omit it hereafter for brevity.
Unless otherwise specified, we focus on class-conditional image generation throughout this work.
For notational simplicity, we omit explicit class conditioning variables from all mathematical formulations.
this section cite: ['b28', 'b28']

Section: Diffusion Transformers (DiT)
The DiT Peebles & Xie (2023) implements ϵ θ using a Vision Transformer Dosovitskiy et al. (2021). Given the "patchified" input x t ∈ R L×Cx (L tokens of dimension C x ), the model first computes embeddings:
h t = x t W emb , W emb ∈ R Cx×C d .(3)
The transformer processes h t ∈ R L×C d to produce o t ∈ R L×C d . The final noise prediction is computed as:
ϵ θ (x t , t) = o t W dec , W dec ∈ R C d ×Cx .(4)
this section cite: ['b14']

Section: Joint Image-Representation Generation
Our goal is to train a single model to jointly generate images and their semantic-aware visual representations by modeling their shared probability distribution. This approach captures the interdependent structures and features of both modalities. While we frame our approach using DDPM, it is also applicable to models trained with flow-matching objectives Ma et al. (2024
this section cite: []

Section: ) (see Appendix A).
A high-level overview of our method is depicted in Figure During training, given x 0 and z 0 , we define a joint forward diffusion processes:
x t = √ ᾱt x 0 + √ 1 -ᾱt ϵ x , z t = √ ᾱt z 0 + √ 1 -ᾱt ϵ z ,(5)
where ᾱt controls the noise schedule and ϵ x ∼ N (0, I), ϵ z ∼ N (0, I) are Gaussian noise terms of dimensions R L×Cx and R L×Cz , respectively.
The diffusion model ϵ θ (x t , z t , t) takes as input x t and z t , along with timestep t, and jointly predicts the noise for both inputs. Specifically, it produces two separate predictions: ϵ x θ (x t , z t , t) for the image latent noise ϵ x , and ϵ z θ (x t , z t , t) for the visual representation noise ϵ z . The training objective combines both predictions:
L joint = E x0,z0,t ∥ϵ x θ (x t , z t , t) -ϵ x ∥ 2 + λ z ∥ϵ z θ (x t , z t , t) -ϵ z ∥ 2 , (6
)
where λ z balances the denoising loss for z t . By default, we use λ z = 1 in our experiments. We explore two approaches to combine and jointly process x t and z t in the diffusion transformer architecture: (1) merging tokens along the embedding dimension, and (2) maintaining separate tokens for each modality (see Fig. 4). Both methods require only minimal modifications to the DiT architecture, specifically defining modality-specific embedding matrices W x emb ∈ R Cx×C d and W z emb ∈ R Cz×C d , along with prediction heads W x dec ∈ R C d ×Cx and W z dec ∈ R C d ×Cz for x t and z t respectively.
this section cite: []

Section: Fusion of Image and Representation Tokens
Merged Tokens The tokens are embedded separately and summed channel-wise:
h t = x t W x emb + z t W z emb ∈ R L×C d . (7
)
The transformer processes h t to produce o t , with predictions:
ϵ x θ = o t W x dec , ϵ z θ = o t W z dec .(8)
This approach enables early fusion while maintaining computational efficiency, as the token count remains unchanged.
Separate Tokens Tokens are embedded separately and concatenated along the sequence dimension:
h t = [x t W x emb , z t W z emb ] ∈ R 2L×C d ,(9)
where [• , •] denotes sequence-wise concatenation. The transformer outputs separate representations o t = [o x t , o z t ], with predictions:
ϵ x θ = o x t W x dec , ϵ z θ = o z t W z dec .(10)
This method provides greater expressive power by preserving modality-specific information throughout processing, at the cost of increased computation due to increased token count.
Unless stated otherwise, we use the merged tokens approach for computational efficiency.
this section cite: []

Section: Dimensionality-Reduced Visual Representation
In practice, the channel dimension of visual representations (C z ) significantly exceeds that of image latents (C x ), i.e., C z ≫ C x . We empirically observe that this imbalance degrades performance, as the model disproportionately allocates capacity to visual representations at the expense of image latents.
To address this, we apply Principal Component Analysis (PCA) to reduce the dimensionality of z 0 from C z to C ′ z (where C ′ z ≪ C z ), preserving essential information while simplifying the prediction task. The PCA projection matrix is precomputed using visual representations sampled from the training set. All visual representations in Sections 3.2 and 3.3 refer to these PCA-reduced versions.
this section cite: []

Section: Representation Guidance
To ensure the generated images remain strongly influenced by the visual representations during inference, we introduce Representation Guidance. This technique during inference modifies the posterior distribution to: pθ (x t , z t ) ∝ p θ (x t )p(z t |x t ) wr , where w r controls how strongly samples are pushed toward higher likelihoods of the conditional distribution p θ (z t |x t ). Taking the log derivative yields the guided score function:
∇ xt log pθ (x t , z t ) =∇ xt log p θ (x t ) + w r ∇ xt log p θ (z t |x t )(11)
=∇ xt log p θ (x t ) + w r ∇ xt log p θ (x t , z t ) -∇ xt log p θ (x t ) .(12)
By recalling the equivalence of denoisers and scores (Vincent, 2011), we implement this representation-guided prediction êθ (x t , z t , t) at each denoising step as follows:
εθ (x t , z t , t) = ϵ θ (x t , t) + w r (ϵ θ (x t , z t , t) -ϵ θ (x t , t)) .(13)
Following Ho & Salimans (2022), we train both e θ (x t , z t , t) and e θ (x t , t) jointly. Specifically, during training, with probability p drop , we zero out z t (setting ϵ θ (x t , t) = ϵ θ (x t , 0, t)) and disable the visual representation denoising loss by setting λ z = 0 in Equation 6.
this section cite: ['b58', 'b27']

Section: Experiments

this section cite: []

Section: Setup
Implementation details. We follow the standard training setup of DiT (Peebles & Xie, 2023) and SiT (Ma et al., 2024), training on ImageNet at 256 × 256 resolution with a batch size of 256. Following ADM's preprocessing pipeline (Dhariwal & Nichol, 2021), we center-crop and resize all images to 256 × 256. Our experiments utilize transformer architectures B/2, L/2, and XL/2 all using a 2×2 patch size. For unconditional generation, we simply set the number of classes to 1, maintaining the original architecture. Images are encoded into VAE latent representations using SD-VAE-FT-EMA (Rombach et al., 2022) that produces outputs with ×8 spatial downsampling factor and 4 output channels. For 256 × 256 images, this results in 32 × 32 × 4 latent features. Through patchification with 2 × 2 patches, the VAE encoder E x (•) yields L = 256 tokens, each with C x = 16 channels (4 channels × 2×2 patch size). For semantic representation extraction, we employ DINOv2-B with registers (Darcet et al., 2023;Oquab et al., 2024). The 768-dimensional embeddings are reduced to 8 dimensions via PCA (trained on 76,800 randomly sampled ImageNet images). After bilinear interpolation to match the VAE's 32 × 32 × 4 spatial resolution and 2 × 2 patchification, the encoder E z (•) produces L = 256 tokens with C z = 32 channels each (8 channels × 2×2 patch size).
this section cite: ['b47', 'b39', 'b50', 'b11', 'b46']

Section: Sampling.
For DiT models, we adopt DDPM sampling, while for SiT models, we employ the SDE Euler-Maruyama sampler. The number of sampling steps is fixed at 250 across all experiments. When using Classifier-Free Guidance (CFG) (Ho & Salimans, 2022), we apply it only to the VAE channels, with a guidance scale of w = 2.4 (see Figure 6). For Representation Guidance, we set p drop = 0.2, the guidance scale to w r = 1.5 for B models and w r = 1.1 for XL models.
Evaluation. To benchmark generative performance, we report Frechet Inception Distance (FID) (Heusel et al., 2017), sFID (Nash et al., 2021), Inception Score (IS) (Salimans et al., 2016), Precision (Pre.) and Recall (Rec.) (Kynkäänniemi et al., 2019) using 50k samples and the ADM's TensorFlow evaluation suite (Dhariwal & Nichol, 2021).  & Salimans (2022). Once again, ReDi yields significant improvements, achieving an FID of 1.72 in just 350 epochs, outperforming the baseline trained to convergence over 1400 epochs.
Comparison with REPA. We further compare our results with REPA, which also leverages DINOv2 features to enhance generative performance. Our approach, ReDi, consistently achieves superior generative performance with both DiT and SiT as the base models. As shown in Table 1, DiT-L/2 with ReDi achives an FID of 10.5 significantly outperforming DiT-L/2 with REPA. Notably, it even surpasses REPA trained for the same number of iterations with the larger DiT-XL/2, which achieves a higher FID of 12.3. Further for SiT-XL models, ReDi attains an FID of 5.6 in just 700k iterations, while REPA requires 4M iterations to reach an FID of 5.9. These results highlight the effectiveness of our method in leveraging visual representations to significantly boost generative performance.
ReDi is complementary to REPA. Interestingly, we observe that the joint modeling objective of our ReDi and the alignment objective of REPA are complementary. As presented in Table 5 REPA + ReDi matches the FID of the fully-converged REPA after only 350K iterations, and at 1M iterations reaches an FID of 3.6. For the implementation details, see Appendix B.3.
this section cite: ['b27', 'b26', 'b43', 'b52', 'b32', 'b27']

Section: Accelerating convergence.
The aforementioned results indicate that ReDi significantly accelerates the convergence of latent diffusion models. As illustrated in Figure 2, ReDi speeds up the convergence of DiT-XL/2 and SiT-XL/2 by approximately ×23, respectively. Even when compared with REPA, ReDi demonstrated a ×6 faster convergence. When ReDi is applied on top of REPA, the convergence is ×11 faster.
Comparison with state-of-the-art generative models. Ultimately, we provide a quantitative comparison between ReDi and other recent generative models using Classifier-Free Guidance (CFG)  5.
this section cite: []

Section: Improving Unconditional Generation.
To establish the effectiveness of our method in improving generative models, we further present experiments for unconditional generation using DiT. As shown in Table 3, our ReDi significantly improves generative performance for various model sizes. Specifically, with our ReDi FID drops from 69.3 to 51.7 for B and from 44.6 to 25.1 for XL models.
this section cite: []

Section: Impact of Representation Guidance on generative performance.

this section cite: []

Section: Class Conditional Generation.
In Table 4 we present the impact of Representation Guidance (RG) on generative performance. We observe that for both B and XL models, Representation Guidance unlocks further performance enhancements by guiding the generated image to closely follow the semantic features of DINOv2. Particularly for DiT-XL w/ ReDi the FID drops from 8.7 to 5.9. We also present qualitative results in Figure 8.
this section cite: []

Section: Unconditional Generation.
Representation Guidance is especially useful in unconditional generation scenarios, where the absence of class or text conditioning prevents the use of Classifier-Free Guidance to enhance performance. As demonstrated in Table 3, Representation Guidance enhances the performance of ReDi with both B and XL models, further closing the performance gap between unconditional and conditional generation. Notably, ReDi with Representation Guidance achieves an FID of 22.6, approaching the performance of the class-conditioned DiT-XL/2 (FID of 19.5). w/o DINOv2 1 2 4 8 12 16 32 25 30 35 40 45 w/o DINOv2 31.9 29.2 27 25.7 27.5 29.1 36.9 43 # Principal Components FID Figure 7: Effect of number of principal components. FID of DiT-B/2 w/ ReDi with different number of DINOv2 Principal Components. The vanilla DiT-B/2 is illustrated with gray. No Classifier-Free Guidance is used.
this section cite: []

Section: Dimensionality reduction ablation.
We begin the analysis of our method by ablating the impact of dimensionality reduction on the visual representations, as shown in Figure 7. Initially, we observe that jointly learning as little as one principal component yields significant improvements in generative performance. Increasing the component count continues to improve performance, up to r = 8, beyond which further components begin to degrade the quality of generation. This suggests an optimal intermediate subspace where compressed visual features retain sufficient expressivity to guide generation without dominating model capacity.
this section cite: []

Section: Merged Tokens vs. Separate Tokens.
In Table 6, we evaluate the effectiveness of the two explored integration strategies, Merged Tokens (MR) and Separate Tokens (SP), for joint learning of image VAE latents and visual representations, using DiT-B/2 as our base model. While both approaches achieve comparable performance gains, SP demonstrates slightly better results. This advantage comes at a significant computational cost: SP doubles the transformer's input sequence length by introducing 256 additional DINOv2 tokens, resulting in approximately 2× greater compute demands during both training and inference (Kaplan et al., 2020). The MR strategy, by contrast, maintains the original sequence length while delivering similar performance improvements, thereby preserving computational efficiency as measured by throughput.
this section cite: ['b30']

Section: VAE-only Classifier-Free Guidance.
As ReDi jointly models both VAE latents and visual representations, we investigate two Classifier-Free Guidance (CFG) strategies: applying CFG exclusively to VAE latents (VAE-only CFG) versus applying it to both modalities simultaneously (VAE & DINOv2 CFG). Our experiments in Figure 6 demonstrate that VAE-only CFG achieves superior results, yielding an FID of 2.39 compared to 2.86 for the VAE & DINOv2 CFG approach. Notably, VAE-only CFG also shows greater robustness to variations in the CFG weight parameter.
this section cite: []

Section: Conclusion
In this work, we explore the relationship between semantic representation learning and generative performance in latent diffusion models. Building on recent insights, we introduced ReDi, a novel framework that integrates high-level semantic features with low-level latent representations within the diffusion process. Unlike prior approaches that rely on auxiliary objectives, ReDi jointly models the two distributions. We demonstrate that this simple approach is more effective at leveraging the semantic features and leads to drastic improvements in generative performance. We further proposed Representation Guidance, a novel guidance method that leverages the jointly learned semantic features to enhance image quality. Across both conditional and unconditional settings, ReDi consistently improves generation quality and accelerates convergence, highlighting the benefits of our approach.
this section cite: []

Section: References
Ref_id:b0 Title: All are worth words: A vit backbone for diffusion models Year: (2023)
Ref_id:b1 Title: BEit: BERT pre-training of image transformers Year: (2022)
Ref_id:b2 Title: Label-efficient semantic segmentation with diffusion models Year: (2022)
Ref_id:b3 Title: Deep clustering for unsupervised learning of visual features Year: (2018)
Ref_id:b4 Title: Unsupervised pre-training of image features on non-curated data Year: (2019)
Ref_id:b5 Title: Unsupervised learning of visual features by contrasting cluster assignments Year: (2020)
Ref_id:b6 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b7 Title: Joint appearance-motion representations for enhanced motion generation in video models Year: (2025)
Ref_id:b8 Title: Diffusion models for multi-modal generative modeling Year: (2024)
Ref_id:b9 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b10 Title: Exploring simple siamese representation learning Year: (2021)
Ref_id:b11 Title: Vision transformers need registers Year: (2023)
Ref_id:b12 Title: Studying image diffusion features for zero-shot video object segmentation Year: (2025)
Ref_id:b13 Title: Diffusion models beat GANs on image synthesis Year: (2021)
Ref_id:b14 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b15 Title: Sigmoid-weighted linear units for neural network function approximation in reinforcement learning Year: (2018)
Ref_id:b16 Title: Diffusion models and representation learning: A survey Year: (2024)
Ref_id:b17 Title: Mdtv2: Masked diffusion transformer is a strong image synthesizer Year: (2023)
Ref_id:b18 Title: Unsupervised representation learning by predicting image rotations Year: (2018)
Ref_id:b19 Title: Online bag-of-visualwords generation for self-supervised learning Year: (2021)
Ref_id:b20 Title: MOCA: Self-supervised representation learning by predicting masked online codebook assignments Year: (2024)
Ref_id:b21 Title: Toward more reliable 3d pose estimation Year: (2023-06)
Ref_id:b22 Title: Bootstrap your own latent-a new approach to selfsupervised learning Year: (2020)
Ref_id:b23 Title: Gem: A generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control Year: (2024)
Ref_id:b24 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b25 Title: Unsupervised semantic correspondence using stable diffusion Year: (2023)
Ref_id:b26 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b27 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b28 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b29 Title: What to hide from your students: Attention-guided masked image modeling Year: (2022)
Ref_id:b30 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b31 Title:  Year: (2025)
Ref_id:b32 Title: Improved precision and recall metric for assessing generative models Year: (2019)
Ref_id:b33 Title: Pretraining image backbones with deep generative models Year: (2023)
Ref_id:b34 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b35 Title: Open-vocabulary object segmentation with diffusion models Year: (2023)
Ref_id:b36 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b37 Title: Beyond generation: Exploring generalization of diffusion models in few-shot segmentation Year: (2023)
Ref_id:b38 Title: Diffusion hyperfeatures: Searching through time and space for semantic correspondence Year: (2023)
Ref_id:b39 Title: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b40 Title: Self-supervised learning of pretext-invariant representations Year: (2020)
Ref_id:b41 Title: 4m: Massively multimodal masked modeling Year: (2023)
Ref_id:b42 Title: Diffusion models beat gans on image classification Year: (2023)
Ref_id:b43 Title: Generating images with sparse representations Year: (2021)
Ref_id:b44 Title: Improved denoising diffusion probabilistic models Year: (2021-07)
Ref_id:b45 Title: Unsupervised learning of visual representations by solving jigsaw puzzles Year: (2016)
Ref_id:b46 Title: DINOv2: Learning robust visual features without supervision Year: (2024)
Ref_id:b47 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b48 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b49 Title: Learning transferable visual models from natural language supervision Year: ()
Ref_id:b50 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b51 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b52 Title: Improved techniques for training gans Year: (2016)
Ref_id:b53 Title: Any-to-any generation via composable diffusion Year: (2023)
Ref_id:b54 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b55 Title: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features Year: (2025)
Ref_id:b56 Title: Plug-and-play diffusion features for text-driven image-to-image translation Year: (2023)
Ref_id:b57 Title: Representation learning with contrastive predictive coding. arXiv e-prints Year: (2018)
Ref_id:b58 Title: A connection between score matching and denoising autoencoders Year: (2011)
Ref_id:b59 Title: Denoising diffusion autoencoders are unified selfsupervised learners Year: (2023)
Ref_id:b60 Title: Simmim: A simple framework for masked image modeling Year: (2022)
Ref_id:b61 Title: Diffusion model as representation learner Year: (2023)
Ref_id:b62 Title: Fasterdit: Towards faster diffusion transformers training without architecture modification Year: (2024)
Ref_id:b63 Title: Language model beats diffusion -tokenizer is key to visual generation Year: (2024)
Ref_id:b64 Title: Representation alignment for generation: Training diffusion transformers is easier than you think Year: (2025)
Ref_id:b65 Title: Sigmoid Loss for Language Image Pre-Training Year: ()
Ref_id:b66 Title: ICCV Year: (2023)
Ref_id:b67 Title: A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence Year: (2023)
Ref_id:b68 Title: World-consistent video diffusion with explicit 3d modeling Year: (2024)
Ref_id:b69 Title: Unleashing text-to-image diffusion models for visual perception Year: (2023)
Ref_id:b70 Title: Fast training of diffusion models with masked transformers Year: (2023)
Ref_id:b71 Title: Image bert pre-training with online tokenizer. International Conference on Learning Representations (ICLR) Year: ()
Ref_id:b72 Title: Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer Year: (2024)
