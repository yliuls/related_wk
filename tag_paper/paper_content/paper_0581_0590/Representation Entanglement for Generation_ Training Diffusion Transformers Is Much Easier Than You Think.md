Title: Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think
Abstract: REPA and its variants effectively mitigate training challenges in diffusion models by incorporating external visual representations from pretrained models, through alignment between the noisy hidden projections of denoising networks and foundational clean image representations. We argue that the external alignment, which is absent during the entire denoising inference process, falls short of fully harnessing the potential of discriminative representations. In this work, we propose a straightforward method called Representation Entanglement for Generation (REG), which entangles low-level image latents with a single high-level class token from pretrained foundation models for denoising. REG acquires the capability to produce coherent image-class pairs directly from pure noise, substantially improving both generation quality and training efficiency. This is accomplished with negligible additional inference overhead, requiring only one single additional token for denoising (<0.5% increase in FLOPs and latency). The inference process concurrently reconstructs both image latents and their corresponding global semantics, where the acquired semantic knowledge actively guides and enhances the image generation process.

Section: Introduction
Generative models have undergone significant evolution [3,4,5,6,7], demonstrating remarkable success across diverse applications [8,9,10,11,12,13,14]. Recent progress in high-fidelity image synthesis has been driven by several key innovations: Latent Diffusion Models (LDM) [15] introduced a stable two-phase training framework, while Diffusion Transformers (DiT) [16] enhanced scalability through transformer-based architectures. Building upon these developments, Scalable Interpolant Transformers (SiT) [2] further unified the approach through continuous-time stochastic interpolants for diffusion training. Despite these advances, achieving high-fidelity synthesis remains a substantial resource for model convergence. While recent techniques such as masked training [17,18] and multiscale optimization [19] partially alleviate computational costs and accelerate model convergence. However, fundamental optimization challenges persist when relying solely on architecture changes.
Recent studies demonstrate that enhanced generative models can acquire more discriminative representations, positioning them as capable representation learners [20,1,21]. However, as quantified by CKNNA metrics [22], these features still underperform compared to those from pretrained vision models [23,24,25]. This performance gap has motivated approaches leveraging pretrained visual encoder features to accelerate generative model training convergence. For example, REPA [1] employs implicit feature-space alignment between diffusion models and foundation vision models (see Fig. 2(a)), while REPA-E [21] extends this alignment by enabling end-to-end VAE tuning, and quantitatively demonstrates that enhanced alignment (via increased CKNNA scores directly improves generation fidelity.) However, the external alignment of REPA, which is absent during the entire denoising inference process, falls short of fully harnessing the potential of discriminative information (see Fig. 2(b)). We suggest this structure likely impedes further advancements in discriminative semantic learning and overall generative capability.
To address these limitations, we propose a straightforward method called Representation Entanglement for Generation (REG), an efficient framework that unleashes the potential of discriminative information through explicitly reflows discriminative information into the generation process (see Fig. 2(c)). REG entangles low-level image latents with a single high-level class token from pretrained foundation models during training by applying synchronized noise injection to both of them with spatial concatenation. The denoising inference process concurrently reconstructs both image latents and their corresponding global semantics from random noise initialization, where the acquired semantic knowledge actively guides and enhances the image generation process (see Fig. 2(d)). REG achieves significant improvements in generation quality, training convergence speed, and discriminative semantic learning, all while introducing minimal computational cost through the addition of just one token (less than 0.5% FLOPs and latency in Tab. 4). On class-conditional ImageNet benchmarks at 256×256 resolution (see Fig. 2(e)), SiT-XL/2 + REG achieves 63× and 23× faster training convergence compared to SiT-XL/2 and SiT-XL/2 + REPA, respectively. Notably, SiT-L/2 + REPA trained for 400K iterations surpasses the performance of SiT-XL/2 + REPA trained for 4M iterations (see Tab. 1).
In summary, our specific contributions are as follows:
• We propose REG, an efficient framework that entangles low-level image latents with a single high-level class token from pretrained foundation models for denoising.
• REG significantly enhances generation quality, training convergence speed, and discriminative semantic learning while introducing negligible computational overhead.
• On ImageNet generation benchmarks, REG achieves 63× and 23× faster training convergence than SiT and REPA.
this section cite: ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b1', 'b16', 'b17', 'b18', 'b19', 'b0', 'b20', 'b21', 'b22', 'b23', 'b24', 'b0', 'b20']

Section: Related work
Generative models for image generation. Traditional approaches such as DDPM [5] and DDIM [26] perform iterative noise removal in pixel space, while LDM [15] operates in compressed latent spaces through pretrained autoencoders. Architecturally, early U-Net-based diffusion models [5,27,15] rely on iterative denoising, whereas modern transformer-based frameworks like DiT [16] and SiT [2] leverage self-attention mechanisms for superior spatial pattern modeling. Despite these advances, existing methods typically require extensive training iterations to achieve convergence. Current acceleration techniques often necessitate significant architectural modifications, such as masked training paradigms [17,18] or multi-scale optimization strategies [28,19]. In contrast, we propose REG, which achieves dual improvements in generation quality and training efficiency while introducing minimal inference overhead (requiring just one additional token during denoising). Crucially, REG accomplishes these gains while preserving the original model architecture, demonstrating that superior training dynamics can be achieved without structural compromises.
this section cite: ['b4', 'b25', 'b14', 'b4', 'b26', 'b14', 'b15', 'b1', 'b16', 'b17', 'b27', 'b18']

Section: Generative models as representation learners.
Extensive research has established that intermediate features in diffusion models inherently encode rich semantic representations [1,21], with demonstrated discriminative capabilities across diverse vision tasks including semantic segmentation [29,30,31], depth estimation [32], and controllable image editing [33,34,35]. Recent advancements have further developed knowledge transfer paradigms from diffusion models to efficient networks through techniques like RepFusion's dynamic timestep optimization [36] and DreamTeacher's cross-model feature distillation [37]. Notably, DDAE [20] confirms that improved diffusion models yield higher-quality representations, establishing a direct correlation between generation capability and representation learning performance. Building upon these insights, we propose to systematically integrate discriminative representations into the generative forward process, enabling persistent discriminative guidance throughout denoising inference.
this section cite: ['b0', 'b20', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b19']

Section: Generative models with external representations.
Prior research [38,39,40] has explored augmenting diffusion models through auxiliary components. For example, RCG [41] employs a secondary diffusion model to generate the class token for adaLN-condition [42] in unconditional generation. In contrast, our approach eliminates the need for additional models by leveraging a single class token as part of the input to provide discriminative guidance, simultaneously enhancing both dis-criminative semantic learning and conditional generation performance. Recent advancements have incorporated visual representations from foundation models to accelerate diffusion training. REPA [1] improves semantic representation quality through feature alignment between early diffusion layers and pretrained vision features, while REPA-E [21] extends this framework by enabling end-to-end VAE tuning. However, these methods rely on external alignment mechanisms that do not take the discriminative representations as the input and denoising, which are unable to produce discriminative representations during inference to guide the generation process. Our proposed REG framework structurally integrates spatial visual representations with semantic class embeddings derived from foundation models. This architectural design enables the denoising phase to concurrently refine localized pattern restoration and holistic conceptual representation, thereby establishing context-aware semantic steering that persists throughout the entire generative process.
this section cite: ['b37', 'b38', 'b39', 'b40', 'b41', 'b0', 'b20']

Section: Method
We propose REG, an efficient framework that provides discriminative guidance by entangling image latents with foundation model class token (Fig. 2(c, d)). Section 3.1 covers preliminaries, followed by REG's detailed description in Section 3.2.
this section cite: []

Section: Preliminaries
Our work is based on Scalable Interpolant Transformers (SiT) [2], which provide a unified perspective to understand flow and diffusion models. We first introduce the relevant preliminaries. Flow and diffusion models both leverage stochastic processes to gradually transform Gaussian noise ϵ ∼ N (0, I) into data samples x * . This process can be unified as
x t = α t x * + σ t ϵ,(1)
where α t is a decreasing and σ t an increasing function of time t. Flow-based models typically interpolate between noise and data over a finite interval, while diffusion models define a forward stochastic differential equation (SDE) that converges to a Gaussian distribution as t → ∞.
Sampling from these models can be achieved via either a reverse-time SDE or a probability flow ordinary differential equation (ODE), both of which yield the same marginal distributions for x t . The probability flow ODE is:
ẋt = v(x t , t),(2)
where the velocity field v(x, t) can be formulated by the conditional expectation:
v(x, t) = E[ ẋt | x t = x] = αt E[x * | x t = x] + σt E[ϵ | x t = x].(3)
To synthesize data, we can integrate Eqn. (3) in reverse time, initializing from random Gaussian noise ϵ ∼ N (0, I). This process yields samples from p 0 (x), serving as an approximation to the true data distribution p(x). This velocity can be estimated by a model v θ (x t , t), which is trained to minimize the following loss function:
E x * ,ϵ,t [∥v θ (x t , t) -αt x * -σt ϵ∥ 2 ].(4)
The reverse-time SDE can describe the probability distribution p t (x) of x t at time t, which can be expressed as:
dx t = v(x t , t)dt - 1 2 w t s(x t , t)dt + √ w t dw t ,(5)
with s(x, t) denoting the score that can be computed via the conditional expectation:
s(x t , t) = -σ -1 t E[ϵ | x t = x].(6)
The score can be reformulated in terms of the velocity v(x, t):
s(x, t) = σ -1 t • α t v(x, t) -αt x α t σt -αt σ t .(7)
We can learn the velocity field v(x, t) and use it to compute the score s(x, t) when using an SDE for sampling.
this section cite: ['b1']

Section: Representation entanglement for generation
REG training process. Given the clean input image x * , we obtain image latents z * ∈ R Dz×Cz×Cz via VAE encoder [15] and image feature f * ∈ R N ×D vf from vision foundation encoder (e.g. DI-NOv2 [23]), where C z × C z denotes the latent spatial resolution, and D z is the channel dimension. Besides, N represents the number of visual tokens, and D vf is the embedding dimension of vision foundation encoder. In REPA, the absence of the ability to autonomously generate discriminative representations to guide generation in inference may reduce the leverage of discriminative information effectively. We introduce the class token cls * ∈ R 1×D vf from the vision foundation model to entangle with image latents for providing the discriminative guidance. Here are the specific details:
We inject noise into both the class token and image latents as a paired input for the SiT forward process [2]. Specifically, given two Gaussian noise samples ϵ z ∼ N (0, I) and ϵ cls ∼ N (0, I) with sizes R Dz×Cz×Cz and R 1×D vf respectively, we perform interpolation operations at continuous time t ∈ [0, 1] as follows:
z t = α t z * + σ t ϵ z ; cls t = α t cls * + σ t ϵ cls ,(8)
This defines intermediate states z t (noised latents) and cls t (noised class token) in the forward diffusion process, where α t and σ t control the generation trajectory. Then, we patchify z t into z ′ t ∈ R N ×D ′ z , D ′ z is the embedding dimension. Afterwards, the class token cls t is projected into the same embedding space via a linear layer to obtain cls ′ t ∈ R 1×D ′ z . Finally, we concatenate them to form h t = [cls ′ t ; z ′ t ] ∈ R (N +1)×D ′ z
, which serves as the input to the subsequent SiT blocks. We perform alignment at specific transformer layers n, where n = 4 for SiT-B/2 + REG and n = 8 for all other variants, maintaining consistency with REPA. Specifically, we align the projected hidden state feature h ϕ (H [n]  t ) ∈ R (N +1)×Dvf with the reference representation y * ∈ R (N +1)×D vf which is concatenated by cls * and f * . h [n]   t ∈ R (N +1)×D ′ z denotes output of the n-th SiT block, h ϕ is a trainable MLP projection, and sim(•, •) represents the cosine similarity. The alignment loss is defined as:
L REPA (θ, ϕ) := -E xt,ϵ,t 1 N N n=1 sim(y * , h ϕ (h [n] t )) .(9)
In addition to alignment, the training objective includes velocity prediction for both the noised image latents z t and class token cls t . The prediction loss is formulated as:
L pred = E x * ,ϵ,t ∥v (z t , t) -αt z * -σt ϵ z ∥ 2 + β ∥v (cls t , t) -αt cls * -σt ϵ cls ∥ 2 , (10
)
where v (•, t) is the velocity prediction function, and β > 0 controls the relative weighting between the image latents and class token denoising objectives. The final training loss integrates both prediction and alignment objectives, where λ > 0 governs the relative weight of the alignment loss compared to the denoising loss. Specifically, the total loss L total is formulated as:
L total = L pred + λL REPA .(11)
REG inference process. The framework requires no auxiliary networks to generate the class token. REG jointly reconstructs both image latents and corresponding global semantics from random noise initialization. It acquired semantic knowledge actively guides and enhances the generation quality.
In general, REG demonstrates three key advantages over existing approaches: (1) Improved utilization of discriminative information. REG directly integrates discriminative information as part of the input during training, enabling both autonomous generation and consistent application of semantic guidance during inference. This design aims to address a limitation of REPA, which cannot autonomously generate discriminative representations to guide generation in inference. Because it relies on an external alignment mechanism during training to utilize discriminative features, rather than incorporating them as input and applying the corresponding denoising task.   We adhere strictly to the standard training protocols of SiT [2] and REPA [1]. Experiments are conducted on the ImageNet dataset [51], with all images preprocessed to 256×256 resolution via center cropping and resizing, following the ADM framework [43]. Each image is encoded into a latent representation using the Stable Diffusion VAE [15]. Model architectures B/2, L/2, and XL/2 (with 2 × 2 patch processing) follow the SiT specifications [2]. For comparability, we fix the training batch size to 256 and adopt identical learning rates and Exponential Moving Average (EMA) configurations as REPA [1]. Additional implementation details are provided in the Appendix.
this section cite: ['b14', 'b22', 'b1', 'b1', 'b0', 'b50', 'b42', 'b14', 'b1', 'b0']

Section: Evaluation protocol.
To comprehensively evaluate image generation quality across multiple dimensions, we employ a rigorous set of quantitative metrics including Fréchet Inception Distance (FID) [52] for assessing realism, structural FID (sFID) [53] for evaluating spatial coherence, Inception Score (IS) [54] for measuring class-conditional diversity, precision (Prec.) for quantifying sample fidelity, and recall (Rec.) [55] for evaluating coverage of the target distribution, all computed on a standardized set of 50K generated samples to ensure statistical reliability. We further supplement these assessments with CKNNA [22] for analyzing feature-space characteristics. Sampling follows REPA [1], using the SDE Euler-Maruyama solver with 250 steps. Full evaluation protocol details are provided in the Appendix.  2(e)). At 4M steps, REG achieves a record-low FID of 1.8, demonstrating superior scalability and efficiency across model sizes.
Comparison with SOTA methods. Tab. 2 presents a comprehensive comparison against recent SOTA methods utilizing classifier-free guidance. Our framework achieves competitive performance using the REPA's same guidance interval [56] with significantly reduced training cost. REG matches SiT-XL's quality in just 80 epochs (17× faster than SiT-XL's 1400 epochs) and surpasses REPA's 800-epoch performance at 480 epochs, highlighting its superior training efficiency and convergence properties. Additional experiments in the Appendix include the results of more training steps, validating the REG's robustness, scalability, and cross-task generalization.
this section cite: ['b51', 'b52', 'b53', 'b54', 'b21', 'b0', 'b55']

Section: Computational cost comparison.
We compare the computational efficiency of REG and REPA under the same model scale (SiT-XL/2) in Tab. 4. REG introduces only a marginal increase in parameter count (+0.30%) and FLOPs (+0.38%) relative to REPA, while maintaining nearly identical latency (6.21s vs. 6.18s, +0.49%). Despite the minimal computational overhead, REG yields substantial improvements in generation quality, achieving a 56.46% relative reduction in FID, alongside a 50.19% increase in IS. These results demonstrate that REG simultaneously improves generation quality and computational efficiency, highlighting its effectiveness as a general-purpose enhancement for generative models.
this section cite: []

Section: Ablation Studies
Different discriminative guidance. We systematically investigate the impact of different pretrained vision encoders and their corresponding class tokens as target representations in Tab. 3. Among all configurations, DINOv2-B achieves the best performance with the lowest FID (15.22) and highest IS (94.64). Notably, all evaluated target representations consistently surpass the REPA, providing empirical evidence that class tokens derived from self-supervised models enhance generation fidelity.
Alignment depth. As shown in Tab. 3, we compare the effects of applying the REPA loss at different network depths. Our analysis reveals that applying the loss in earlier layers yields superior results, which is consistent with REPA's findings. Notably, our method demonstrates consistent improvements over REPA across all configurations, achieving FID reductions ranging from 4.19 to 7.16 points. We attribute these gains to the direct insertion of the class token, which provides discrete global guidance to all layers. This enables adaptive integration of discriminative semantics throughout the network, in contrast to REPA's indirect supervision mechanism, where only selected features are aligned with the target representation. As a result, REG allows remaining layers to capture richer high-frequency details than REPA, contributing to the observed improvements.
this section cite: []

Section: Effect of β.
Tab. 3 systematically evaluates the impact of varying the loss weight β, which controls the contribution of the class token alignment loss. Among the tested values, β = 0.03 achieves the best overall performance across all evaluation metrics. Consequently, this value was adopted as the default parameter for all subsequent experiments.  Effectiveness of entanglement alone. Tab. 6 evaluates the impact of incorporating class tokens from various pretrained self-supervised encoders into SiT-B/2 without applying representation alignment. The results demonstrate that class token entanglement alone consistently enhances generation quality, with FID improvements ranging from 0.95 to 6.33 points across all variants. Notably, DINOv2-B delivers optimal performance, achieving a 19.18% FID reduction and 35.86% IS improvement compared to the SiT-B/2 baseline. These findings indicate that the model can effectively leverage high-level semantic guidance from the class token, even in the absence of explicit alignment, highlighting the robustness and general utility of class token-based entanglement for generative modeling.
this section cite: []

Section: Enhancing the discriminative semantic learning of generative models
We systematically measure REG, SiT, and REPA's CKNNA scores across training steps, network layers, and timesteps to assess the discriminative semantics of dense features. For fair comparison, we follow REPA's evaluation protocol: We compute CKNNA scores exclusively between spatially averaged generative model dense features and averaged DINOv2-g dense features, while class token are not involved in calculations. Here are the specific situations:
Training steps analysis. Fig. 3(a) shows the positive correlation between CKNNA and FID scores across training steps at layer 8 (t=0.5). It reveals that both REPA and REG achieve improved semantic alignment (higher CKNNA) with better generation quality (lower FID). Notably, REG consistently outperforms REPA in both metrics throughout training, demonstrating its superior capacity for discriminative semantic learning through discriminative semantics guidance.
Layer-wise progression. At 400K training steps (t=0.5) in Fig. 3(b), both REG and REPA exhibit similar CKNNA patterns: semantic scores gradually increase until reaching the peak at layer n=8 (where alignment loss is computed), then progressively decrease. Crucially, REG achieves consistently higher semantic scores than REPA and SiT across all network layers. This improvement stems from REG's innovation of entangling low-level image latents with high-level class token from pretrained foundation models. Through attention mechanisms, REG effectively propagates these discriminative semantics to guide the model in understanding low-level features in early layers, while later layers subsequently focus on predicting high-frequency details.
Timestep robustness. Evaluation of CKNNA at layer 8 (400K steps) demonstrates REG's consistent superiority across all timesteps in Fig. 3(c). This robustness confirms its stable, high-level semantic guidance capability throughout the entire noise spectrum, enabling reliable discriminative semantic performance regardless of noise intensity during generation.
this section cite: []

Section: Conclusion
This paper presents Representation Entanglement for Generation (REG), a simple and efficient framework that firstly introduces image-class denoising paradigm instead of the current pure image denoising pipeline, which fully unleashes the potential of discriminative gains for generation. REG entangles low-level image latents with a single high-level class token from pretrained foundation models, achieved via synchronized noise injection and spatial concatenation. The denoising process simultaneously reconstructs both image latents and corresponding global semantics, enabling active semantic guidance that enhances generation quality while introducing minimal computational cost through the addition of just one token. Extensive experiments demonstrate REG's superior performance in generation fidelity, accelerating training convergence, and discriminative semantic learning, validating its effectiveness and scalability.
this section cite: []

Section: References
Ref_id:b0 Title: Representation alignment for generation: Training diffusion transformers is easier than you think Year: (2024)
Ref_id:b1 Title: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b2 Title: Vqgan-clip: Open domain image generation and editing with natural language guidance Year: (2022)
Ref_id:b3 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b4 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b5 Title: Improving and generalizing flow-based generative models with minibatch optimal transport Year: (2023)
Ref_id:b6 Title: Zero-shot text-to-image generation Year: (2021)
Ref_id:b7 Title: Token merging for training-free semantic binding in text-to-image synthesis Year: (2024)
Ref_id:b8 Title: Gem: A generalizable ego-vision multimodal world model Year: (2024)
Ref_id:b9 Title: Jtav: Jointly learning social media content representation by fusing textual, acoustic, and visual features Year: (2018)
Ref_id:b10 Title: Codegen: An open large language model for code with multi-turn program synthesis Year: (2022)
Ref_id:b11 Title: Ledit: Your length-extrapolatable diffusion transformer without positional encoding Year: (2025)
Ref_id:b12 Title: Anchor token matching: Implicit structure locking for training-free ar image editing Year: (2025)
Ref_id:b13 Title: From cradle to cane: A two-pass framework for high-fidelity lifespan face aging Year: (2025)
Ref_id:b14 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b15 Title: Scalable diffusion models with transformers Year: (2006)
Ref_id:b16 Title: Masked diffusion transformer is a strong image synthesizer Year: (2023)
Ref_id:b17 Title: Mdtv2: Masked diffusion transformer is a strong image synthesizer Year: (2023)
Ref_id:b18 Title: Alleviating distortion in image generation via multiresolution diffusion models Year: (2024)
Ref_id:b19 Title: Denoising diffusion autoencoders are unified self-supervised learners Year: ()
Ref_id:b20 Title: Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers Year: (2025)
Ref_id:b21 Title: The platonic representation hypothesis Year: (2024)
Ref_id:b22 Title: Dinov2: Learning robust visual features without supervision Year: (2023)
Ref_id:b23 Title: An empirical study of training self-supervised vision transformers Year: (2021)
Ref_id:b24 Title: Learning transferable visual models from natural language supervision Year: (2008)
Ref_id:b25 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b26 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b27 Title: Parameter-inverted image pyramid networks Year: (2024)
Ref_id:b28 Title: Label-efficient semantic segmentation with diffusion models Year: (2021)
Ref_id:b29 Title: Unleashing text-to-image diffusion models for visual perception Year: (2023)
Ref_id:b30 Title: Addp: Learning general representations for image recognition and generation with alternating denoising diffusion process Year: (2023)
Ref_id:b31 Title: Depthfm: Fast monocular depth estimation with flow matching Year: (2024)
Ref_id:b32 Title: Prompt-to-prompt image editing with cross attention control Year: (2022)
Ref_id:b33 Title: Diffusion autoencoders: Toward a meaningful and decodable representation Year: (2022)
Ref_id:b34 Title: Unsupervised representation learning from pre-trained diffusion probabilistic models Year: (2022)
Ref_id:b35 Title: Diffusion model as representation learner Year: (2023)
Ref_id:b36 Title: Dreamteacher: Pretraining image backbones with deep generative models Year: (2023)
Ref_id:b37 Title: Stabilize the latent space for image autoregressive modeling: A unified perspective Year: (2024)
Ref_id:b38 Title: Spae: Semantic pyramid autoencoder for multimodal generation with frozen llms Year: (2023)
Ref_id:b39 Title: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models Year: (2025)
Ref_id:b40 Title: Return of unconditional generation: A selfsupervised representation generation method Year: (2024)
Ref_id:b41 Title: Film: Visual reasoning with a general conditioning layer Year: (2018)
Ref_id:b42 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b43 Title: Understanding diffusion objectives as the elbo with simple data augmentation Year: ()
Ref_id:b44 Title: Simple diffusion: End-to-end diffusion for high resolution images Year: ()
Ref_id:b45 Title: Classifier-free diffusion guidance Year: ()
Ref_id:b46 Title: All are worth words: A vit backbone for diffusion models Year: ()
Ref_id:b47 Title: Diffit: Diffusion vision transformers for image generation Year: ()
Ref_id:b48 Title: Fast training of diffusion models with masked transformers Year: ()
Ref_id:b49 Title: Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer Year: ()
Ref_id:b50 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b51 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b52 Title: Generating images with sparse representations Year: (2021)
Ref_id:b53 Title: Improved techniques for training gans Year: (2016)
Ref_id:b54 Title: Improved precision and recall metric for assessing generative models Year: (2019)
Ref_id:b55 Title: Applying guidance in a limited interval improves sample and distribution quality in diffusion models Year: (2024)
Ref_id:b56 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b57 Title: Decoupled weight decay regularization Year: (2017)
