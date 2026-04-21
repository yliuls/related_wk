Title: Masked Autoencoders Are Effective Tokenizers for Diffusion Models
Abstract: Recent advances in latent diffusion models have demonstrated their effectiveness for highresolution image synthesis. However, the properties of the latent space from tokenizer for better learning and generation of diffusion models remain under-explored. Theoretically and empirically, we find that improved generation quality is closely tied to the latent distributions with better structure, such as the ones with fewer Gaussian Mixture modes and more discriminative features. Motivated by these insights, we propose MAE-Tok, an autoencoder (AE) leveraging mask modeling to learn semantically rich latent space while maintaining reconstruction fidelity. Extensive experiments validate our analysis, demonstrating that the variational form of autoencoders is not necessary, and a discriminative latent space from AE alone enables state-of-the-art performance on ImageNet generation using only 128 tokens. MAETok achieves significant practical improvements, enabling a gFID of 1.69 with 76! faster training and 31! higher inference throughput for 512!512 generation. Our findings show that the structure of the latent space, rather than variational constraints, is crucial for effective diffusion models. Code and trained models are released 1 .

Section: Introduction
Diffusion models (Sohl-Dickstein et al., 2015a;Ho et al., 2020;Rombach et al., 2022a;Peebles & Xie, 2023) have recently emerged as a powerful class of generative models, achieving state-of-the-art (SOTA) performance on various image synthesis tasks (Deng et al., 2009;Ghosh et al., 2024).
Although originally formulated in pixel space (Ho et al., 2020;Dhariwal & Nichol, 2021), subsequent research has shown that operating in a latent space -a compressed representation typically learned by a tokenizer -can substantially improve the efficiency and scalability of diffusion models (Rombach et al., 2022a). By avoiding the high-dimensional pixel domain during iterative diffusion and denoising steps, latent diffusion models dramatically reduce computational overhead and have quickly become the de facto paradigm for high-resolution generation (Esser et al., 2024).
However, a key question remains: What constitutes a "good" latent space for diffusion? Early work primarily employed Variational Autoencoders (VAE) (Kingma, 2013) as tokenizers, which ensure that the learned latent codes follow a relatively smooth distribution (Higgins et al., 2017) via a Kullback-Leibler (KL) constraint. While VAEs can empower strong generative results (Ma et al., 2024;Li et al., 2024b;Deng et al., 2024), they often struggle to achieve high pixel-level fidelity in reconstructions due to the imposed regularization (Tschannen et al., 2025). In contrast, recent explorations with plain Autoencoders (AE) (Hinton & Salakhutdinov, 2006;Vincent et al., 2008) produce higherfidelity reconstructions but may yield latent spaces that are insufficiently organized or too entangled for downstream generative tasks (Chen et al., 2024b). Indeed, more recent studies emphasize that high fidelity to pixels does not necessarily translate into robust or semantically disentangled latent representations (Esser et al., 2021;Yao & Wang, 2025); leveraging latent alignment with pre-trained models can often improve generation performance further (Li et al., 2024c;Chen et al., 2024a;Qu et al., 2024;Zha et al., 2024).
In this work, we attempt to answer this question by investigating the interaction between the latent distribution learned by tokenizers, and the training and sampling behavior of diffusion models operating in that latent space. Specifically, we study AE, VAE and the recently emerging representation aligned VAE (Li et al., 2024c;Chen et al., 2024a;Zha et al., 2024;Yao & Wang, 2025), by fitting a Gaussian mixture model (GMM) into their latent space. Empirically, we show that a latent space with more discriminative features, whose GMM modes are fewer, tends to produce a lower diffusion loss. Theoretically, we prove that a latent distribution with fewer GMM modes indeed leads to a lower loss of diffusion models and thus to better sampling during inference.
Motivated by these insights, we demonstrate that diffusion models trained on AEs with discriminative latent space are enough to achieve SOTA performance. We propose to train AEs as Masked Autoencoders (MAE) (He et al., 2022;Xie et al., 2022;Wei et al., 2022), a self-supervised paradigm that can discover more generalized and discriminative representations by reconstructing proxy features (Zhang et al., 2022). More specifically, we adopt the transformer architecture of tokenizers (Yu et al., 2021;2024c;Li et al., 2024c;Chen et al., 2024a) and randomly mask the image tokens at the encoder, whose features need to be reconstructed at the decoder (Assran et al., 2023). To maintain a pixel decoder with high reconstruction fidelity, we adopt auxiliary shallow decoders that predict the features of unseen tokens from seen ones to learn the representations, along with the pixel decoder which is normally trained as previous tokenizers. The auxiliary shallow decoders introduce trivial computation overhead during training. This design allows us to extend the MAE objective that reconstructs masked image patches, to simultaneously predict multiple targets, such as HOG (Dalal & Triggs, 2005) features (Wei et al., 2022), DINOv2 features (Oquab et al., 2023), CLIP embeddings (Radford et al., 2021;Zhai et al., 2023), and Byte-Pair Encoding (BPE) indices with text (Huang et al., 2024).
Furthermore, we reveal an interesting decoupling effect: the capacity to learn a discriminative and semantically rich latent space at the encoder can be separated from the capacity to achieve high reconstruction fidelity at the decoder. In particular, a higher mask ratio (40-60%) in MAE training often degrades immediate pixel-level quality. However, by freez-ing the AE's encoder, thus preserving its well-organized latent space, and fine-tuning only the decoder, we can recover strong pixel-level reconstruction fidelity without sacrificing the semantic benefits of the learned representations.
Extensive experiments on ImageNet (Deng et al., 2009) demonstrate the effectiveness of MAETok. It addresses the trade-off between reconstruction fidelity and discriminative latent space by training the plain AEs with mask modeling, showing that the structure of latent space is more crucial for diffusion learning, instead of the variational forms of VAEs. MAETok achieves improved reconstruction FID (rFID) and generation FID (gFID) using only 128 tokens for the 256→256 and 512→512 ImageNet benchmarks.
Our contributions can be summarized as follows:
• Theoretical and Empirical Analysis: We establish a connection between latent space structure and diffusion model performance through both empirical and theoretical analysis. We reveal that structured latent spaces with fewer Gaussian Mixture Model modes enable more effective training and generation of diffusion models.
• MAETok: We train plain AEs using mask modeling and show that simple AEs with more discriminative latent space empower faster learning, better generation, and higher throughput of diffusion models, showing that the variational regularization of VAE is not necessary.
• SOTA Generation Performance:
Diffusion models of 675M parameters trained on MAETok with 128 tokens achieve performance comparable to previous best models on 256 ImageNet generation and outperform previous 2B USiT at 512 resolution with 1.69 gFID and 304.2 IS.
this section cite: ['b30', 'b54', 'b13', 'b22', 'b30', 'b19', 'b36', 'b27', 'b49', 'b12', 'b70', 'b28', 'b76', 'b18', 'b81', 'b56', 'b87', 'b87', 'b81', 'b25', 'b80', 'b78', 'b89', 'b82', 'b0', 'b11', 'b78', 'b53', 'b57', 'b88', 'b31', 'b13']

Section: On the Latent Space and Diffusion Models
To study the relationship of latent space for diffusion models, we start with popular tokenizers, including AE (Hinton & Salakhutdinov, 2006), VAE (Kingma, 2013), representation aligned VAE, i.e., VAVAE (Yao & Wang, 2025). We train our own AE and VAE tokenizers under the same training recipe and the same dimension for fair comparison. We train diffusion models on them and establish connections between latent space properties and the quality of the final image generation through empirical and theoretical analysis.
this section cite: ['b28', 'b36', 'b81']

Section: Empirical Analysis.
Inspired by existing theoretical work (Chen et al., 2022;2023;Benton et al., 2024), our investigation of the connection between latent space and generation quality starts with a high-level intuition. With optimal diffusion model parameters, such as sufficient total time steps and adequately small discretization steps, and with assumed similar capacity of tokenizer decoders, the generation quality of diffusion models, i.e., the learned latent distribution, is dominated by the denoising network's training loss (Chen et al., 2022;2023;Benton et al., 2024), while the effectiveness of training diffusion model via DDPM (Ho et al., 2020) heavily depends on the hardness of learning the latent space distribution (Shah et al., 2023;Diakonikolas et al., 2023;Gatmiry et al., 2024). Specially, when the training data distribution is too complex and multi-modal, i.e., not discriminative enough, the denoising network may struggle to capture such entangled global structure of latent space, resulting in a degraded generation quality.
Building upon this intuition, we use the Gaussian Mixture Models (GMM) to evaluate the number of modes in alternative latent space representations, where a higher number of modes indicates a more complex structure. The details of GMM training are included in Appendix B.3. Fig. 2a analyzes the GMM fitting by varying the number of Gaussian components and comparing their negative log-likelihood losses (NLL) across different latent spaces, where a lower NLL indicates better fitting quality. We observe that, to achieve comparable fitting quality, i.e., similar GMM losses, VAVAE requires fewer modes compared to VAE and AE.
Fewer modes are sufficient to adequately represent the latent space distributions of VAVAE compared to those of AE and VAE, highlighting simpler global structures in its latent space. Correspondingly, Fig. 2b reports the training losses of diffusion models with AE, VAE, and VAVAE, which (almost) align with the GMM losses shown in Fig. 2a, where fewer modes correspond to lower diffusion losses and better gFID. This alignment validates our intuition, confirming that latent spaces with fewer modes and thus more separated and discriminative features can reduce the learning difficulty and lead to better generation quality of diffusion models.
Theoretical Analysis. After observing experimental phenomena that align with our high-level intuition, we further present a concise theoretical analysis here to justify the rationale behind it, with more details provided in Appendix A.
Following the empirical analysis setup, we first consider a latent data distribution in d dimensions modeled as a GMM with K equally weighted Gaussians:
p 0 = 1 K K i=1 N (µ → i , I),(1)
Considering the classic diffusion model DDPM (Ho et al., 2020) and following the training objective as Shah et al. (2023), the score matching loss of DDPM at timestep t is
min w E[↑s w (x, t) ↓ ↔ x log p t (x)↑ 2 ],(2)
where s w (x, t) represents the denoising network and ↔ x log p t (x) denotes the oracle score function.
Then, we establish the following theorem to show that more modes typically require larger training sample sizes for diffusion models to achieve comparable generation quality.
Theorem 2.1. (Informal, see Theorem A.7) Let the data distribution be a mixture of K Gaussians as defined in Eq. (1). Then assume the norm of each mode is bounded by some constants, let d be the data dimension, T be the total time steps, and ω be a proper target error parameter. In order to achieve a O(T ω 2 ) error in KL divergence between data distribution and generation distirbution, the DDPM algorithm may require using n ↗ n ↑ number of samples:
n ↑ = ! K 4 d 5 B 6 ε 2 , (3
)
where the upper bound of the mean norm satisfies max i ↑µ i ↑ ↘ B.
Theorem 2.1 combines Theorem 16 from (Shah et al., 2023) and Theorem 2.2 from (Chen et al., 2023), showing that to achieve a comparable generation quality O(T ω 2 ), latent spaces with more modes (K) require a larger training sample size, scaling as O(K 4 ).This theoretically help explain
… ViT Encoder ℇ ViT Decoder 𝒟 𝑚 𝑚 𝑚 … … 𝑒 𝑒 𝑒 𝑒 𝑒 … 𝑒 1 Aux. Decoder 𝒟 𝑎𝑢𝑥 1 𝑒 1 𝑒 1 𝑒 1 𝑒 1 … 𝑒 2 Aux. Decoder 𝒟 𝑎𝑢𝑥 2 𝑒 2 𝑒 2 𝑒 2 𝑒 2 𝑚 𝑒 Autoencoder Auxiliary Decoders Other Targets … Latent Token 𝑧 Latent Repre. ℎ Enc. Mask Token 𝑚 Dec. Image Token 𝑑 HOG Features DINOv2 Features 𝐿 𝑁 Figure 3. Model architecture of MAETok. We adopt the plain 1D autoencoder (AE) as tokenizer, with a vision transformer (ViT) encoder E and decoder D. MAETok is trained using mask modeling at encoder, with a mask ratio of 40-60%, and predict multiple target features, e.g., HOG, DINO-v2, and CLIP features, of masked tokens from the unmasked ones using auxiliary shallow decoders. why, under a finite number of training samples, latent spaces with more modes (e.g., AE and VAE) produce worse generations with higher gFID. We provide additional experimental results in Appendix A, demonstrating that these latent distributions share comparable upper bounds B, thus justifying our focus primarily on the impact of mode number K.
this section cite: ['b8', 'b2', 'b8', 'b2', 'b30', 'b64', 'b15', 'b21', 'b30', 'b64', 'b64', 'b4']

Section: Method
Motivated by our analysis, we show that the variational form of VAEs may not be necessary for diffusion models, and simple AEs are enough to achieve SOTA generation performance with 128 tokens, as long as they have discriminative latent spaces, i.e., with fewer GMM modes. We term our method as MAETok, with more details as follows.
this section cite: []

Section: Architecture
We build MAETok upon the recent 1D tokenizer design with learnable latent tokens (Yu et al., 2024c;Li et al., 2024c;Chen et al., 2024a). Both the encoder E and decoder D adopt the Vision Transformer (ViT) architecture (Dosovitskiy et al., 2021;Yu et al., 2021), but are adapted to handle both image tokens and latent tokens, as shown in Fig. 3.
Encoder. The encoder first divides the input image I ≃ R H↓W ↓3 into N patches according to a predefined patch size P , each mapped to an embedding vector of dimension D, resulting in image tokens x ≃ R N ↓D . In addition, we define a set of L learnable latent tokens z ≃ R L↓D . The encoder transformer takes the concatenation of image patch embeddings and latent tokens [x; z] ≃ R (N +L)↓D as its input, and outputs the latent representations h ≃ R L↓H with a dimension of H from only the latent tokens:
h = E ([x; z]) .(4)
Decoder. To reconstruct the image, we use a set of N learnable image tokens e ≃ R N ↓H . We concatenate these mask tokens with h as the input to the decoder, and takes only the outputs from mask tokens for reconstruction:
x = D([e; h]]).
(5)
We then use a linear layer on top of x ≃ R N ↓D to regress the pixel values and obtain the reconstructed image Î.
Position Encoding. To encode spatial information, we apply 2D Rotary Position Embedding (RoPE) to the image patch tokens x at the encoder and the image tokens e at the decoder. In contrast, the latent tokens z (and their encoded counterparts h) use standard 1D absolute position embeddings, since they do not map to specific spatial locations. This design ensures that patch-based tokens retain the notion of 2D layout, while the learned latent tokens are treated as a set of abstract features within the transformer architecture.
Training objectives. We train MAETok using the standard tokenizer losses as in previous work (Esser et al., 2021):
L = L recon + ϑ 1 L percep + ϑ 2 L adv ,(6)
with L recon , L percep , and L adv denoting as pixel-wise meansquare-error (MSE) loss, perceptual loss (Larsen et al., 2016;Johnson et al., 2016;Dosovitskiy & Brox, 2016;Zhang et al., 2018), and adversarial loss (Goodfellow et al., 2020;Isola et al., 2018), respectively, and ϑ 1 and ϑ 2 being hyperparameters. Note that MAETok is a plain AE architecture, therefore, it does not require any variational loss between the posterior and prior as in VAEs, which simplifies training.
this section cite: ['b17', 'b82', 'b18', 'b39', 'b33', 'b16', 'b90', 'b23', 'b32']

Section: Mask Modeling
Token Masking at Encoder. A key property of MAETok is that we introduce mask modeling during training, following the principles of MAE (He et al., 2022;Xie et al., 2022), to learn a more discriminative latent space in a self-supervised way. Specifically, we randomly select a certain ratio, e.g., 40%-60%, of the image patch tokens according to a binary masking indicator M ≃ R N , and replace them with the learnable mask tokens m ≃ R D before feeding them into the encoder. All the latent tokens are maintained to more heavily aggregate information on the unmasked image tokens and used to reconstruct the masked tokens at the decoder output.
this section cite: ['b25', 'b80']

Section: Auxiliary Shallow Decoders.
In MAE, a shallow decoder (He et al., 2022) or a linear layer (Xie et al., 2022;Wei et al., 2022) is required to predict the target features, e.g., raw pixel values, HOG features, and features from pre-trained models, of the masked image tokens from the remaining ones. However, since our goal is to train MAE as tokenizers, the pixel decoder D needs to be able to reconstruct images in high fidelity. Thus, we keep D as a similar capacity to E, and incorporate auxiliary shallow decoders to predict additional feature targets, which share the same design as the main pixel decoder but with fewer layers. Formally, each auxiliary decoder D j aux takes the latent representations h and concatenate with their own d j as inputs, and output ŷj as the reconstruction of their feature target
y j ≃ R N ↓D j : ŷj = D j aux ([e j ; h]; ϖ),(7)
where D j denotes the dimension of target features. We train these auxiliary decoders along with our AE using additional MSE losses at only the masked tokens according to the masking indicator M , similarly to Xie et al. (2022):
L mask = j M ⇐ ŷj ↓ y j 2 2 .(8)
this section cite: ['b25', 'b80', 'b78', 'b80']

Section: Pixel Decoder Fine-Tuning
While mask modeling encourages the encoder to learn a better latent space, high mask ratios can degrade immediate reconstruction. To address this, after training AEs with mask modeling, we freeze the encoder, thus preserving the latent representations, and fine-tune only the pixel decoder for a small number of additional epochs. This process allows the decoder to adapt more closely to frozen latent codes of clean images, recovering the details lost during masked training. We use the same loss as in Eq. ( 6) for pixel decoder fine-tuning and discard all auxiliary decoders in this stage.
this section cite: []

Section: Experiments
We conduct comprehensive experiments to validate the design choices of MAETok, analyze its latent space, and benchmark the generation performance to show its superiority.
this section cite: []

Section: Experiments Setup
Implementation Details of Tokenizer. We use XQ-GAN codebase (Li et al., 2024d) to train MAETok. We use ViT-Base (Dosovitskiy et al., 2021), initialized from scratch, for both the encoder and the pixel decoder, which in total have 176M parameters. We set L = 128 and H = 32 for latent space. Three MAETok variants are trained on 256→256 ImageNet (Deng et al., 2009), and 512→512 ImageNet, and a subset of 512→512 LAION-COCO (Schuhmann et al., 2022) for 500K iterations, respectively. In the first stage training with mask modeling on ImageNet, we adopt a mask ratio of 40-60% , set by ablation, and 3 auxiliary shallow decoders for multiple targets of HOG (Dalal & Triggs, 2005), DINO-v2-Large (Oquab et al., 2023), and SigCLIP-Large (Zhai et al., 2023) features. We adopt an additional auxiliary decoder for tokenizer trained on LAION-COCO, which predicts the discrete indices of text captions for the image using a BPE tokenizer (Cherti et al., 2023;Huang et al., 2024). Each auxiliary decoder has 3 layers also set by ablation. We set ϑ 1 = 1.0 and ϑ 2 = 0.4. For the pixel decoder fine-tuning, we linearly decrease the mask ratio from 60% to 0% over 50K iterations, with the same training loss. More training details of tokenizers are shown in Appendix B.1.
Implementation Details of Diffusion Models. We use SiT (Li et al., 2024a) and LightningDiT (Yao & Wang, 2025) for diffusion-based image generation tasks after training MAETok. We set the patch size of them to 1 and use a 1D position embedding, and follow their original training setting for other parameters. We use SiT-L of 458M parameters for the analysis and ablation study. For main results, we train SiT-XL of 675M parameters for 4M steps and Light-ningDiT for 400K steps on ImageNet of resolution 256 and 512. More details are provided in Appendix B.2.
Evaluation. For tokenizer evaluation, we report the reconstruction Frechet Inception Distance (rFID) (Heusel et al., 2017), peak-signal-to-noise ratio (PSNR), and structural similarity index measure (SSIM) on ImageNet and MS-COCO (Lin et al., 2014) validation set. For the latent space evaluation of the tokenizer, we conduct linear probing (LP) on the flatten latent representations and report accuracy. To evaluate the performance of generation tasks, we report generation FID (gFID), Inception Score (IS) (Salimans et al., 2016), Precision and Recall (Kynkäänniemi et al., 2019) (in Appendix C.1), with and without classifier-free guidance (CFG) (Ho & Salimans, 2022), using 250 inference steps.
this section cite: ['b17', 'b13', 'b63', 'b11', 'b53', 'b88', 'b9', 'b31', 'b81', 'b26', 'b46', 'b62', 'b37', 'b29']

Section: Design Choices of MAETok
We first present an extensive ablation study to understand how mask modeling and different designs affect the reconstruction of tokenizer and, more importantly, the generation of diffusion models. We start with an AE and add different components to study both rFID of AE and gFID of SiT-L.
this section cite: []

Section: Mask Modeling.
In Table 1a, we compare AE and VAE with mask modeling and also study the proposed fine-tuning of the pixel decoder. For AE, mask modeling significantly improves gFID and slightly deteriorates rFID, which can be recovered through the decoder fine-tuning stage without sacrificing generation performance. In contrast, mask modeling only marginally improves the gFID of VAE, since the imposed KL constraint may hinder latent space learning.
Reconstruction Target. In Table 1b, we study how different reconstruction targets affect latent space learning in mask modeling. We show that using the low-level reconstruction features, such as the raw pixel (with only a pixel decoder) and HOG features, can already learn a better latent space, resulting in a lower gFID. Adopting semantic teachers such as DINO-v2 and CLIP instead can significantly improve gFID.
Combining different reconstruction targets can achieve a balance in reconstruction fidelity and generation quality.
this section cite: []

Section: Mask Ratio.
In  as highlighted in previous works (He et al., 2022;Wei et al., 2022;Xie et al., 2022). A low mask ratio prevents the AE from learning more discriminative latent space. A high mask ratio imposes a trade-off between reconstruction fidelity and the latent space quality, and thus generation performance.
this section cite: ['b25', 'b78', 'b80']

Section: Auxiliary Decoder Depth.
We study the depth of auxiliary decoder in Table 1d with multiple reconstruction targets. We show that a decoder that is too shallow or too deep could hurt both the reconstruction fidelity and generation quality. When the decoder is too shallow, the combined target features may confuse the latent with high-level semantics and low-level details, resulting in a worse reconstruction fidelity. However, a deeper auxiliary decoder may learn a less discriminative latent space of the AE with its strong capacity, and thus also lead to worse generation performance.
We include more ablation study on the number of learnable latent tokens and 2D RoPE in Appendix C.4.
this section cite: []

Section: Latent Space Analysis
We further analyze the relationship between the latent space of the AE variants and the generation performance of SiT-L.
this section cite: []

Section: Latent Space Visualization.
We provide a UMAP visualization (McInnes et al., 2018) in Fig. 4 to intuitively compare the latent space learned by different variants of AE. Notably, both the AE and VAE exhibit more entangled latent embeddings, where samples corresponding to different classes tend to overlap substantially. In contrast, MAETok shows distinctly separated clusters with relatively clear boundaries between classes, suggesting that MAETok learns more discriminative latent representations. In line with our analysis in Section 2 and Fig. 2, a more discrimina- Latent Distribution and Generation Performance. We assess the latent space's quality by studying the relationship between the linear probing (LP) accuracy on the latent space, as a proxy of how well semantic information is preserved in the latent codes, and the gFID for generation performance. In Fig. 5a, we observe tokenizers with more discriminative latent distributions, as indicated by higher LP accuracy, correspondingly achieve lower gFID. This finding suggests that when features are well-clustered in latent space, the generator can more easily learn to generate high-fidelity samples. We further verify this intuition by tracking gFID throughout training, shown in Fig. 5b, where MAETok enables faster convergence, with gFID rapidly decreasing with lower values than the AE or VAE baselines. A high-quality latent distribution is shown to be a crucial factor in both achieving strong final generation metrics and accelerating training.
this section cite: ['b50']

Section: Main Results

this section cite: []

Section: Generation.
We compare SiT-XL and LightningDiT based on variants of MAETok in Tables 2 and 3 for the 256→256 and 512→512 ImageNet benchmarks, respectively, against other SOTA generative models. Notably, the naive SiT-XL trained on MAETok with only 128 tokens and plain AE architecture achieves consistently better gFID and IS without using CFG: it outperforms REPA (Yu et al., 2024d)  Table 4. Comparison of various continuous tokenizers. † indicates the tokenizer is trained on other data than ImageNet. MAETok achieves a better trade-off of compression and reconstruction.
and gFID with CFG to 1.65. These results demonstrate that the structure of the latent space (see Fig. 4), instead of the variational form of tokenizers, is vital for the diffusion model to learn effectively and efficiently. We show a few selected generation samples in Fig. 1, and more uncurated visualizations are included in Appendix C.5.
Reconstruction. MAETok also offers strong reconstruction capabilities on ImageNet and MS-COCO, as shown in Table 4. Compared to previous continuous tokenizers, including SD-VAE (Rombach et al., 2022a), DC-AE (Chen et al., 2024b), VA-VAE (Yao & Wang, 2025), SoftVQ-VAE (Chen et al., 2024a), and TexTok (Zha et al., 2024), MAE-Tok achieves a favorable trade-off between the quality of the reconstruction and the size of the latent space. On 256→256 ImageNet, using 128 tokens, MAETok attains an rFID of 0.48 and SSIM of 0.763, outperforming methods such as SoftVQ in terms of both fidelity and perceptual similarity, while using half of the tokens in TexTok (Zha et al., 2024). On MS-COCO, where the tokenizer is not directly trained, MAETok still delivers robust reconstructions. At resolution of 512, MAETok maintains its advantage by balancing compression ratio and the reconstruction quality.
this section cite: ['b81', 'b87', 'b87']

Section: Discussion

this section cite: []

Section: References
Ref_id:b0 Title: Self-supervised learning from images with a joint-embedding predictive architecture Year: (2023)
Ref_id:b1 Title: All are worth words: A vit backbone for diffusion models Year: (2023)
Ref_id:b2 Title: Nearly d-linear convergence bounds for diffusion models via stochastic localization Year: (2024)
Ref_id:b3 Title: Masked generative image transformer Year: (2022)
Ref_id:b4 Title: Improved analysis of scorebased generative modeling: User-friendly bounds under minimal smoothness assumptions Year: (2023)
Ref_id:b5 Title: Softvqvae: Efficient 1-dimensional continuous tokenizer Year: (2024)
Ref_id:b6 Title: Visual generation without guidance Year: (2025)
Ref_id:b7 Title: Deep compression autoencoder for efficient high-resolution diffusion models Year: (2024)
Ref_id:b8 Title: Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions Year: (2022)
Ref_id:b9 Title: Reproducible scaling laws for contrastive language-image learning Year: (2023)
Ref_id:b10 Title: Manifold-constrained classifier free guidance for diffusion models Year: (2024)
Ref_id:b11 Title: Histograms of oriented gradients for human detection Year: (2005)
Ref_id:b12 Title: Causal diffusion transformers for generative modeling Year: (2024)
Ref_id:b13 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b14 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b15 Title: Sq lower bounds for learning mixtures of separated and bounded covariance gaussians Year: (2023)
Ref_id:b16 Title: Generating images with perceptual similarity metrics based on deep networks Year: (2016)
Ref_id:b17 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b18 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b19 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b20 Title: Mdtv2: Masked diffusion transformer is a strong image synthesizer Year: (2023)
Ref_id:b21 Title: Learning mixtures of gaussians using diffusion models Year: (2024)
Ref_id:b22 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2024)
Ref_id:b23 Title: Generative adversarial networks Year: (2020)
Ref_id:b24 Title: Rethinking the objectives of vector-quantized tokenizers for image synthesis Year: (2023)
Ref_id:b25 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b26 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b27 Title: betavae: Learning basic visual concepts with a constrained variational framework Year: (2017)
Ref_id:b28 Title: Reducing the dimensionality of data with neural networks Year: (2006)
Ref_id:b29 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b30 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b31 Title: Classification done right for vision-language pre-training Year: (2024)
Ref_id:b32 Title: Imageto-image translation with conditional adversarial networks Year: (2018)
Ref_id:b33 Title: Perceptual losses for real-time style transfer and super-resolution Year: (2016)
Ref_id:b34 Title: A style-based generator architecture for generative adversarial networks Year: (2019)
Ref_id:b35 Title: Guiding a diffusion model with a bad version of itself Year: (2024)
Ref_id:b36 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b37 Title: Improved precision and recall metric for assessing generative models Year: (2019)
Ref_id:b38 Title: Applying guidance in a limited interval improves sample and distribution quality in diffusion models Year: (2024)
Ref_id:b39 Title: Autoencoding beyond pixels using a learned similarity metric Year: (2016)
Ref_id:b40 Title: Autoregressive image generation using residual quantization Year: (2022)
Ref_id:b41 Title: Scalable autoregressive image generation with mamba Year: (2024)
Ref_id:b42 Title: Masked generative encoder to unify representation learning and image synthesis Year: (2023)
Ref_id:b43 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b44 Title: Autoregressive image generation with folded tokens Year: (2024)
Ref_id:b45 Title: Xq-gan: An open-source image tokenization framework for autoregressive generation Year: (2024)
Ref_id:b46 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b47 Title: Customize your visual autoregressive recipe with set autoregressive modeling Year: (2024)
Ref_id:b48 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b49 Title: Exploring flow and diffusionbased generative models with scalable interpolant transformers Year: (2024)
Ref_id:b50 Title: Uniform manifold approximation and projection for dimension reduction Year: (2018)
Ref_id:b51 Title: Finite scalar quantization: Vq-vae made simple Year: (2023)
Ref_id:b52 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b53 Title:  Year: (2023)
Ref_id:b54 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b55 Title: Robust latent matters: Boosting image generation with sampling error synthesis Year: (2025)
Ref_id:b56 Title: Unified image tokenizer for multimodal understanding and generation Year: (2024)
Ref_id:b57 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b58 Title: Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems Year: (2019)
Ref_id:b59 Title: Generating diverse high-fidelity images with vq-vae-2 Year: (2019)
Ref_id:b60 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b61 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b62 Title: Improved techniques for training gans Year: (2016)
Ref_id:b63 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b64 Title: Learning mixtures of gaussians using the ddpm objective Year: (2023)
Ref_id:b65 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b66 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b67 Title: Denoising diffusion implicit models Year: (2022)
Ref_id:b68 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b69 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b70 Title: Generative infinite-vocabulary transformers Year: (2025)
Ref_id:b71 Title: Regularizing generative adversarial networks under limited data Year: (2021)
Ref_id:b72 Title: Score-based generative modeling in latent space Year: (2021)
Ref_id:b73 Title: Conditional image generation with pixelcnn decoders. Advances in neural information processing systems Year: (2016)
Ref_id:b74 Title: Neural discrete representation learning Year: (2017)
Ref_id:b75 Title: Attention is all you need Year: (2023)
Ref_id:b76 Title: Extracting and composing robust features with denoising autoencoders Year: (2008)
Ref_id:b77 Title: Embedding-free image generation via bit tokens Year: (2024)
Ref_id:b78 Title: Masked feature prediction for self-supervised visual pre-training Year: (2022)
Ref_id:b79 Title: Vila-u: a unified foundation model integrating visual understanding and generation Year: (2024)
Ref_id:b80 Title: Simmim: A simple framework for masked image modeling Year: (2022)
Ref_id:b81 Title: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models Year: (2025)
Ref_id:b82 Title: Vector-quantized image modeling with improved vqgan Year: (2021)
Ref_id:b83 Title: Language model beats diffusion -tokenizer is key to visual generation Year: (2024)
Ref_id:b84 Title: Randomized autoregressive visual generation Year: (2024)
Ref_id:b85 Title: An image is worth 32 tokens for reconstruction and generation Year: (2024)
Ref_id:b86 Title: Representation alignment for generation: Training diffusion transformers is easier than you think Year: (2024)
Ref_id:b87 Title: Language-guided image tokenization for generation Year: (2024)
Ref_id:b88 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b89 Title: How mask matters: Towards theoretical understandings of masked autoencoders Year: (2022)
Ref_id:b90 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b91 Title: Studying classifier (-free) guidance from a classifier-centric perspective Year: (2025)
Ref_id:b92 Title: Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99% Year: (2024)
