Title: ★STARFLOW: Scaling Latent Normalizing Flows for High-resolution Image Synthesis
Abstract: Text conditioned high-resolution samples of variable aspect ratios generated from our 3.8B STARFlow model. Resolutions are adjusted for the ease of visualization.

Section: Introduction
Recent years have witnessed remarkable progress in high-resolution text-to-image generative modeling, with state-of-the-art approaches predominantly falling into two distinct categories. On one hand, diffusion models (Ho et al., 2020;Rombach et al., 2022;Peebles & Xie, 2023;Esser et al., 2024) operating in continuous space have set new benchmarks in image quality. However, their reliance on iterative denoising processes renders both training and inference computationally intensive. On the other hand, autoregressive image generation methods (Yu et al., 2022;Sun et al., 2024;Tian et al., 2024)-inspired by the success of large language models (LLMs, Brown et al., 2020;Dubey et al., 2024)-avoid such inefficiencies by modeling images in discrete space via quantization; yet, this quantization can impose stringent limitations and adversely affect fidelity. More recently, a promising trend has emerged to explore hybrid models (Li et al., 2024;Gu et al., 2024b;Fan et al., 2024) that apply autoregressive techniques directly in continuous space. However, the inherently distinct characteristics of these two paradigms introduce additional complexity in effective unification.
In this paper, we turn our eyes on the yet another modeling approach -Normalizing Flows (NFs, Rezende & Mohamed, 2015;Dinh et al., 2016), a family of likelihood based models that have received relatively little attention in the recent wave of Generative AI. We start from inspecting TARFlow (Zhai et al., 2024), a recently proposed model that combines a powerful Transformer architecture with autoregressive flows (AFs, Kingma et al., 2016;Papamakarios et al., 2017). While TARFlow demonstrates promising results on the potential of NFs as a modeling principle, it remains unclear whether it can perform as a scalable method, in comparison to other approaches such as diffusion and discrete autoregressive models. To this end, we propose STARFlow, a family of generative models that shows for the first-time that NF models can successfully generalize to highresolution and large-scale image modeling. We first provide a theoretical insight on why AFs can be capable generative models by showing the universality of multi-block AFs in modeling continuous distributions. On top of this, we propose a novel deep-shallow architecture. We found that the architecture configuration, e.g., the number of flows as well as the depth and width of the Transformer for each flow, plays a pivotal role to the model's performance. While TARFlow (Zhai et al., 2024) proposes to uniformly allocate model depth among all flows, we found that it is beneficial to have a skewed architecture design, where we allocate most of the model parameters to the first AF block (i.e., the one closest to the prior), which is followed by a few shallow but non-negligible blocks. Importantly, our model still yields a stand-alone normalizing-flow framework that supports end-toend maximum-likelihood training in continuous space, thereby sidestepping the quantization limits inherent to discrete models. Rather than operating directly in data space, we instead learn AFs in the latent space of pretrained autoencoders. Crucially, we demonstrate that NFs align naturally with compressed latents-an intuitive yet vital observation-enabling far superior modeling of highresolution inputs, as verified in our experiments, compared with training directly on pixels. Similar to TARFlow, noise injection proves essential: by fine-tuning the decoder, we train the model on noisy latents and at the same time simplify the original sampling pipeline. Moreover, we revisit the classifier-free guidance (CFG) algorithm for AFs from a more principled way and propose a novel guidance algorithm, which substantially improves image quality, especially at high guidance weights in text-to-image generation tasks.
Together, these innovations represent the first demonstration of NF models applied to large-scale, highresolution image generation. Our approach offers a scalable and efficient alternative to conventional diffusion-based and autoregressive approaches, achieving competitive performance on benchmarks for both class-conditioned image and large-scale text-to-image synthesis. Moreover, our framework is highly flexible, and we demonstrate that it easily enables interesting settings such as image inpainting and instruction based image editing by finetuning.
this section cite: ['b28', 'b53', 'b47', 'b16', 'b64', 'b55', 'b58', 'b2', 'b13', 'b39', 'b17', 'b52', 'b12', 'b65', 'b37', 'b46', 'b65']

Section: Preliminaries

this section cite: []

Section: Normalizing Flows
In this paper, we consider Normalizing Flows (NFs, Rezende & Mohamed, 2015;Dinh et al., 2014Dinh et al., , 2016) ) as the class of likelihood method that follows the change of variable formula. Given continuous inputs x ∼ p data , x ∈ R D , a NF learns an invertible transformation f θ : R D → R D (with θ being the parameters) which maps data x into the noise space f θ (x), and can be trained with maximum likelihood estimation (MLE):
max θ E x∼p data log p NF (x; θ) = log p 0 (f θ (x); θ) + log det ∂f θ (x) ∂x ,(1)
where the first term rewards sending data to high-density regions of the prior p 0 , while the Jacobian term penalizes excessive local volume shrinkage, ensuring the transformation remains bijective and does not collapse nearby points onto a lower-dimensional set. One automatically obtains a generative model by inverting f θ , with a sampling procedure z ∼ p 0 (z), x = f -1 θ (z).
this section cite: ['b52', 'b11', 'b12']

Section: Autoregressive Flows and TARFlow
An interesting variant of NFs is autoregressive flows (AFs, Kingma et al., 2016;Papamakarios et al., 2017). In the simplest affine form, an AF constructs z = f θ (x) = {µ θ , σ θ }(x) as a standalone invertible model with the forward (x → z) and sampling (z → x) process:
z d = (x d -µ θ (x <d )) /σ θ (x <d ), x d = µ θ (x <d ) + σ θ (x <d ) • z d , ∀d ∈ [1, D],(2)
where x 0 is a constant <sos>. This can be seen as "next-token prediction" with affine transformation, and training with Eq. ( 1) where the Jacobian term becomes extremely simple as -D d=1 log σ θ (x <d ). The extension to multi-channel inputs x ∈ R D×C (e.g., C = 3 for RGB image) is immediate as channels at each step can be treated as conditionally independent. We omit the channel dim henceforth.
Recently, Zhai et al. (2024) introduced TARFlow, a compelling framework for building performant NFs for image data. Specifically, TARFlows can be viewed as a special form of AFs by pairing causal-Transformer blocks with an extension of classical AF formulation -stacking multiple AF layers whose autoregressive ordering alternates from one layer to the next. To be concrete, with T flows, we have
z = f T θ • f 2 θ • • • • • f 1 θ (x),
where each block f t θ (.) processes the input in its own ordering x π = (x π1 , . . . , x π D ) (a permutation of {x 1 . . . x D }), enabling the stack to capture dependencies in both directions of the data sequence. Training is still performed end-to-end:
max θ E x∼p data log p AF (x; θ) = - 1 2 ∥z∥ 2 2 - T t=1 D d=1 log σ t θ (x t π <d ),(3)
where x t = f t θ (x t-1 ) defines the forward propagation (Eq. ( 2)); we denote the data x = x 0 and the final output z = x T is modeled with standard Gaussian. Additionally, Zhai et al. (2024) also proposed several techniques to improve the modeling capability, including noise augmented training, score-based denoising and incorporating guidance (Ho & Salimans, 2021).
this section cite: ['b37', 'b46', 'b65', 'b65', 'b26']

Section: STARFlow
In this section, we propose Scalable Transformer Autoregressive Flow (STARFlow), a method that pushes the frontier of NF based high-resolution image generation. We first establish-on theoretical grounds-AFs' expressivity as a general modeling method in § 3.1, based on which we propose our core approaches by improving TARFlow in several key aspects: (1) a better architecture configuration ( § 3.2), (2) a working recipe of learning in the latent space ( § 3.3) and (3) a novel guidance algorithm ( § 3.4). An illustration of the learning and inference pipeline is presented in Fig. 4.
this section cite: []

Section: Why TARFlows are Capable Generative Models?
While empirical results confirm that TARFlow is highly competitive (Zhai et al., 2024), we ask-from a modeling perspective-whether they are expressive enough to warrant scaling. Here, we claim: Proposition 1. Stacked autoregressive flows with T ≥ 2 blocks of D autoregressive steps and alternating orderings are highly expressive for modeling continuous densities on R D .
this section cite: ['b65']

Section: Sketch of Proof.
Let's consider T = 2. Without loss of generality, we model f θ = f a θ • f b θ where f a θ and f b θ employ reversed orderings (forward and backward) for data x ∈ R D :
x d = µ b θ (x <d ) + σ b θ (x <d ) • y d , y d = µ a θ (y >d ) + σ a θ (y >d ) • z d , z d ∼ N (0, I), d ∈ [1, D]. (4)
We assume z ∼ N (0, I) under the base distribution. This yields the autoregressive factorization p(x) = D d=1 p(x d | x <d ) as follows:
p(x d | x <d ) = N x d | μθ (x <d , y >d ), σ2 θ (x <d , y >d )I • p(y >d | x <d )dy >d ,(5)
where μθ = µ b θ (x <d ) + µ a θ (y >d )σ b θ (x <d ), σθ = σ a θ (y >d )σ b θ (x <d ) defined in Eq. ( 4). For every d < D, we have y >d ̸ = ∅, so Eq. ( 5) is a latent-variable marginalization of Gaussians (mixturelike) and can represent complex continuous conditionals. For the final coordinate d = D we have y >D = ∅. Eq. ( 5) reduces to a single Gaussian and the expressivity is restricted. This restriction can be lifted by extending additional augmented variables. Moreover, the above derivation only uses the base assumption z d ∼ N (0, 1) in the generative direction. In general, conditioning on observed coordinates induces a non-Gaussian latent distribution (i.e., q θ (z d | x <d ) is not necessarily Gaussian). Consequently, the resulting conditional q θ (x D | x <D ) can be even more complex than a single Gaussian. Additional derivation details appear in the Appendix A. □
The preceding proposition clarifies why we can safely scale-up AFs on large data. Even in the minimal setting T = 2 where full universality is not attained, the resulting limitation is negligible in high-dimensional domains such as natural images. The derivation in § 3.1 motivates a redesign of scalable AF architectures within realistic computational budgets, emphasizing that we need not greatly expand the number of flow blocks-indeed (even T = 2 often suffices). However, the remark leaves unresolved how best to allocate compute across those blocks. We first inspect the proposed architecture configuration in TARFlow, which suggests to allocate equal sized Transformer layers for each flow. Interestingly, in our reproduced TARFlow results, we see that most effective compute (measured through the lens of guidance) concentrates in just the top few AF blocks (see motivating examples Fig. 3). We conjecture that end-to-end training drives the network to exploit layers closest to the noise, a behavior that contrasts that of diffusion models.
this section cite: []

Section: Proposed Architecture
Deep-shallow Architecture Our architecture can be intuitively considered as an extension of standard autoregressive language models (e.g., LLaMA (Dubey et al., 2024)) with a general deepshallow design. At inference time, a deep AF block first autoregressively generates x 1 from noise z, followed by a sequence of shallow AF blocks that iteratively refine it to x N , all while keeping the total number of blocks T small. Given a total depth budget L, we instantiate the model as l(T ): one deep l-layer block and T -1 shallow 2-layer blocks, satisfying L = l + 2(T -1). This asymmetric design turns the deep block into a Gaussian language model, while the shallow stack plays the role of a learned image tokenizer.
Conditional STARFlow This design naturally extends to conditional generation by simply prepending the control signal (e.g., class label, caption) to the input of the flow. Interestingly, our preliminary experiments show that conditioning only the deep block-while leaving the shallow blocks to focus solely on local image refinement-incurs no loss in performance. This not only simplifies the overall architecture, but also enables seamless initialization of the deep block with any pre-trained language model, without major modifications. As a result, our image generator can be directly integrated into any LLM's semantic space, eliminating the need for a separate text encoder.
this section cite: ['b13']

Section: Moving to Latent Space
Analogy to Stable Diffusion (SD, Rombach et al., 2022) w.r.t standard diffusion models, STARFlow directly models the latent space of a pretrained autoencoders x ≈ D( x), x = E(x), enabling highresolution image generation. For instance, when using SD-1.4 autoencoderfoot_0 , one can reduce input shape from 256 × 256 to 32 × 32. As noted by Zhai et al. (2024), injecting a proper amount of Gaussian noise, instead of small dequantization noise Dinh et al. (2016); Ho et al. (2019), is crucial for stable training and high quality sampling. This then makes it necessary to perform an additional score-based denoising step to clean up the noise components in the samples Zhai et al. (2024).
In the context of latent normalizing flows, however, the added noise becomes an integral component of the latent representation. Specifically, we encode each sample as x ∼ q enc = N E(x); σ 2 L I . We perform preliminary search for the noise scale (σ L ) to based on the choice of autoencoders. For example, we set σ L = 0.3 throughout the paper.
Learning Learning in the latent space leaves additional flexibility that the flow model can focus on high-level semantics and leave the low-level local details with the pixel decoder. In this way, AF acts as a learnable prior for the latents. Following VAEs (Kingma & Welling, 2013), we optimize the entire model by maximizing the evidence lower-bound (ELBO) where the entropy term is constant:
max θ,ϕ E x∼qenc( x|x),x∼p data [log p AF ( x; θ) + log p dec (x| x; ϕ) -log q enc ( x|x)] ,(6)
where ϕ are the parameters of decoder p dec which transforms the noisy latents back to the pixel space.
Here, we jointly train the AF prior and pixel decoder, freezing the encoder distribution -as in SD-, which stabilizes training and decouples their optimization. Relaxing the encoder q enc and training with the full ELBO loss including entropy regularization are left for future work.
Pixel Decoder As shown in Eq. ( 6), the prertaiend decoder has to be adapted in order to decode from the noisy latents. Different from Zhai et al. (2024) which relies on gradient-based denoising, modeling in the latent allows a simpler solution by directly fine-tuning the decoder over noisy latents:
min ϕ L (D(E(x + σϵ); ϕ), x) ,(7)
where following Esser et al. (2021), L = L L2 + L LPIPS + βL GAN . We empirically observe consistently better performance than score-based denoising technique proposed in (Zhai et al., 2024), with FID decreasing from 2.96 to 2.40 on ImageNet-256. See Appendix C for more discussions.
this section cite: ['b53', 'b65', 'b12', 'b27', 'b65', 'b36', 'b65', 'b15', 'b65']

Section: Revisiting Classifier-Free Guidance for Autoregressive Flows
Classifier-free guidance (CFG), originally introduced for diffusion models (Ho & Salimans, 2021), has become a cornerstone in modern generative modeling, proving broadly effective across various architectures, including AR models (Yu et al., 2022). At a high level, CFG amplifies the difference between conditional and unconditional predictions, encouraging more mode-seeking behavior.
In the context of AFs, Zhai et al. (2024) made the first attempt to apply CFG by linearly extrapolating the mean and variance at each step (Eq. ( 2)):
μc = µ c + ω(µ c -µ u ) and σc = σ c + ω(σ c -σ u ) 2 ,
where ω > 0 denotes the guidance weight. While effective to some extent, this naïve formulation lacks principled justification, leaving unclear how µ and σ should be jointly modulated under guidance. Furthermore, as shown in Fig. 5, this approach becomes unstable at high guidance weights-precisely the regime required for visually compelling results in text-to-image generation.
We propose to revisit CFG from the perspective of score function, the original intuition of Ho & Salimans (2021). In short, we want to sample from a guided distribution p which score satisfies:
∇ x log pc (x) = ∇ x log p c (x) + ω (∇ x log p c (x) -∇ x log p u (x)) .(8)
It is generally non-trivial to determine pc for every flow block. Fortunately, under the design of our proposed model, guidance is only required in the deep block, which functions as a Gaussian Language Model ( § 3.2). Therefore, Eq. ( 8) can be easily simplified into the following:
Proposition 2. Given p u = N (µ u , σfoot_1 u I), and p c = N (µ c , σ 2 c I), the guided distribution pc is also Gaussian pc = N (μ c , σ2 c I) and satisfies:
μc = µ c + ωs 1 + ω -ωs • (µ c -µ u ), σc = 1 √ 1 + ω -ωs • σ c ,(9)
where s = σ 2 c /σ 2 u and ω > 0.
proof : A detailed derivation is provided in the Appendix A. □ Notably, when σ c = σ u , Eq. ( 9) reduces to the standard CFG used in diffusion models. However, directly applying Eq. ( 9) can lead to severe numerical instability, as the denominator 1 + ω -ωs may approach zero or even become negative. To address this, we propose clipping s via s = CLIP(s, 0, 1), motivated by the intuition that the guided distribution should be more mode-seeking than the original, implying that 1 + ω -ωs ≥ 1 for any ω, therefore s ≤ 1.
this section cite: ['b26', 'b64', 'b65', 'b26']

Section: Applications
STARFlow is a versatile generative model that not only produces diverse, high-quality images under various conditions but also extends naturally to downstream applications. We showcase two examples: image inpainting and editing. Training-Free Inpainting We first map the masked image to the latent space, replacing masked regions with Gaussian noise. Reverse sampling is then performed, restoring unmasked pixels with ground truth. We perform generation iteratively until the final inpainted output.
this section cite: []

Section: Interactive Generation and Editing
We finetune STARFlow on an image editing dataset (Fig. 6b), enabling joint modeling of generation and editing with a single conditional AF model. Its invertibility also allows direct image encoding, making it suitable for interactive use.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Settings
Dataset We experiment with STARFlow on both class-conditioned and text-to-image generation tasks. For the former, we conduct experiments on ImageNet-1K (Deng et al., 2009) including 256 × 256 and 512 × 512 resolutions. For text-to-image, we show two settings: a constrained setting CC12M (Changpinyo et al., 2021), where each image is accompanied by a synthetic caption following (Gu et al., 2024a). We also demonstrated a scaled setting where our models trained an in-house dataset with CC12M, in total ∼ 700M text-image pairs.
Evaluation In line with prior works, we report Fréchet Inception Distance (FID) (Heusel et al., 2017) to quantify the the realism and diversity of generated images. For text-to-image generation, we use MSCOCO 2017 (Lin et al., 2014) validation set to assess the zero-shot capabilities of these models. We also report additional evaluation (e.g., GenEval (Ghosh et al., 2023)) in Appendix C.
this section cite: ['b8', 'b4', 'b25', 'b42', 'b19']

Section: Model and Training Details
We implement all models following the setup of Dubey et al. (2024), using RoPE (Su et al., 2024) for positional encoding. By default, we set the architecture to d(N ) = 18(6) with a model dimension of 2048 (XL) and 24(6) with a dimension of 3096 (XXL) for classconditioned and text-to-image models, respectively ( § 3.2), resulting in 1.4B and 3.8B parameters. Since STARFlow operates in a compressed latent space, we are able to train all models with a patch size of p = 1. For text-to-image models, we use T5-XL (Raffel et al., 2020) as the text encoder. To showcase the generality of our approach, we also train a variant where the deep block is initialized from a pretrained LLM (Gemma2 (Team et al., 2024) in this case), without additional text encoder.
All models are pre-trained at 256 × 256 resolution on 400M images with a global batch size of 512. High-resolution finetuning is done by increasing input length. For text-to-image models, variablelength inputs are supported via mixed-resolution training: images are pre-classified into 9 shape buckets and flattened into sequences for unified processing. See Appendix B for detailed settings.
this section cite: ['b13', 'b54', 'b49', 'b57']

Section: Results
Comparison with Baselines We benchmark our approach on class-conditioned ImageNet-256, comparing against diffusion and autoregressive models across both discrete and continuous domains (Table 1). For fair comparison, we train a TARFlow model Zhai et al. (2024) in pixel space with a similar parameter count and original architecture (8 flows, 8 layers each, width 1280). We also train a variant with our deep-shallow design, identical to STARFlow except for using pixel inputs with linearly scaled patch sizes. Among NF models, the deep-shallow architecture consistently outperforms the standard design, and switching to latent-space inputs yields further gains. Our method achieves competitive results compared to other baselines (Tables 1 and 2). Note the FID on ImageNet 256 × 256 is near saturated to the upper-bound of the finetuned decoder (see additional details in Appendix B). Zero-shot evaluations on COCO (Table 3) show strong performance on Table 1: Class-cond ImageNet 256×256 (FID-50K) Model FID↓ # Param. Diffusion Models ADM (Dhariwal & Nichol, 2021) 10.94 554M CDM (Ho et al., 2022b) 4.88 -LDM (Rombach et al., 2022) 3.60 400M RIN (Jabri et al., 2022) 3.76 410M DiT (Peebles & Xie, 2023) 2.27 675M SiT (Ma et al., 2024) 2.06 675M
Autoreg. (discrete) VQGAN (Esser et al., 2021) 15.78 1.4B RQTran (Lee et al., 2022) 3.80 3.8B LlamaGen-3B (Sun et al., 2024) 2.18 3.1B VAR (Tian et al., 2024) 1.73 2.0B
Autoreg. (continuous) Jetformer (Tschannen et al., 2024b) 6.64 2.75B MAR-AR (Li et al., 2024) 4.69 479M MAR (Li et al., 2024) 1.55 943M DART (Gu et al., 2024b) 3.82 820M GIVT (Tschannen et al., 2024a) 2.59 -Normalizing Flow TARFlow (Zhai et al., 2024) a 5.56 1.3B TARFlow + deep-shallow 4.69 1.4B STARFlow (Ours) 2.40 1.4B a Implemented using their official codebase.
Table 2: Class-cond ImageNet 512×512 (FID-50K) Model FID↓ # Param. ADM-U (Dhariwal & Nichol, 2021) 3.85 731M DiT-XL/2 (Peebles & Xie, 2023) 3.04 674M LEGO (Zheng et al., 2024b) 3.74 681M MaskDiT-G (Zheng et al., 2024a) 2.50 730M EDM2-XXL(Karras et al., 2024) 1.25 1.5B STARFlow (Ours) 3.00 1.4B
Table 3: Zero-shot T2I on COCO (FID-30K) Method FID↓ # Param.
DALL•E (Ramesh et al., 2021) 27.5 12B CogView2 (Ding et al., 2021) 24.0 6B Make-A-Scene (Gafni et al., 2022) 11.8 -DART (Gu et al., 2024b) 11.1 800M DALL•E 2 (Ramesh et al., 2022) 10.4 5.5B GigaGAN (Kang et al., 2023) 9.1 1B Muse (Chang et al., 2023) 7.9 3B Imagen (Ho et al., 2022a) 7.3 3B Parti-20B (Yu et al., 2022) 7.2 20B eDiff-I (Balaji et al., 2022) 7.0 9B STARFlow-CC12M 10.3 3.8B STARFlow-CC12M-Gemma 11.4 2.4B STARFlow-FullData 9.1 3.8B text-conditioned generation, demonstrating that NFs can also serve as a scalable and competitive generative modeling framework.
Qualitative Results Fig. 7 and Appendix Fig. 10 present representative class-and text-conditioned generations, respectively. Our method delivers high-resolution images over a wide range of aspect ratios, with perceptual quality comparable to state-of-the-art diffusion and autoregressive approaches. Fig. 9 also highlights our model's support for image editing. Further qualitative and interactive editing results appear in Appendix G, underscoring the breadth and fidelity of our outputs.
this section cite: ['b65', 'b53', 'b32', 'b47', 'b44', 'b15', 'b38', 'b55', 'b58', 'b39', 'b39', 'b50', 'b10', 'b18', 'b51', 'b33', 'b3', 'b64', 'b0']

Section: Comparison with Diffusion and Autoregressive Models
We further compare STARFlow with diffusion and autoregressive (AR) models to analyze training dynamics. Fig. 8a shows FID trajectories  Figure 9: Example of Image editing using STARFlow. Given an input image and simple description, our model can seamlessly edit the contents based on various instruction using with the learned model prior.
using nearly identical architectures. While the FID gap between STARFlow and the baselines is smaller when computed over 4,096 samples, STARFlow consistently achieves the lowest FID at every training checkpoint when evaluated with 50,000 samples. This suggests that STARFlow produces more diverse outputs, which may not be fully captured with smaller evaluation sets. Comparison of CFG Strategies As shown in the Fig. 8b, the original strategy used in Zhai et al. (2024) exhibits a sharp "dip-and-spike" behavior: it achieves its best FID at similar guidance weight as the newly proposed CFG, but then degrades quickly as you move away from that optimum. Even when using the "annealing trick" (Zhai et al., 2024), performance still suffers dramatically both scales. By contrast, our proposed CFG not only improves on the original's best point-without additional tricks-but-more importantly-maintains nearly the same quality over a much wider range of guidance weights, which gives more flexibility in tuning text-conditioned generation tasks.
this section cite: ['b65', 'b65']

Section: Scalability Analysis
To assess the scalability, we perform a study by varying the depth of the deep block and tracking performance over training. Fig. 8c reports negative log-likelihood (NLL) and Fig. 8d shows FID with 4096 samples across iterations. Both metrics indicate that deeper models converge faster and achieve better final performance, demonstrating the increased capacity.
this section cite: []

Section: Ablation on Model Design
To validate the theoretical insights from Prop. 1, we study how model expressivity varies with the number of layers T in the deep block. Performance drops sharply when T < 2, while models with T ≥ 2 perform similarly-consistent with Prop. 1. We also ablate the number and depth of deep blocks in Figs. 8e and 8f, finding that block depth is more critical than quantity, providing practical guidance for architectural design.
5 Related Work Continuous Normalizing Flows, Flow Matching, and Diffusion Models Normalizing Flows (NFs) can be extended to continuous-time via Continuous Normalizing Flows (CNFs) (Chen et al., 2018), which model transformations as ODEs. This relaxes the need for explicit invertible mappings and simplifies Jacobian computation to a trace (Grathwohl et al., 2018), though it requires noisy stochastic estimators (Hutchinson, 1989). Flow Matching (Lipman et al., 2023), inspired by CNFs, learns sample-wise interpolations between prior and data using vector fields grounded in Tweedie's Lemma (Efron, 2011). While CNFs and NFs optimize exact likelihoods through invertible mappings, Flow Matching aligns more closely with diffusion models, sharing variational training objectives.
Autoregressive Models Discrete autoregressive models, especially large language models (Brown et al., 2020;Dubey et al., 2024;Guo et al., 2025), dominate modern generative AI by scaling next-token prediction. Scaling laws (Kaplan et al., 2020) show predictable gains with more data and parameters. These models now power leading multimodal systems for both understanding and generation (Liang et al., 2024;Sun et al., 2024;Tian et al., 2024;Li et al., 2025).
To overcome information loss from quantization, recent work extends AR modeling to continuous spaces, using mixture-of-Gaussians (Tschannen et al., 2024a,b) or diffusion decoding (Li et al., 2024;Gu et al., 2024b;Fan et al., 2024). Hybrid approaches also emerge, unifying AR and diffusion paradigms (Gu et al., 2024a;Zhou et al., 2024;OpenAI, 2024).
this section cite: ['b6', 'b21', 'b31', 'b43', 'b14', 'b2', 'b13', 'b24', 'b34', 'b41', 'b55', 'b58', 'b40', 'b39', 'b17', 'b68']

Section: Conclusion and Limitation
We have presented STARFlow, the first latent based normalizing flow model that scales to high resolution images and large scale text to image modeling. Our results demonstrate that normalizing flows are scalable generative modeling method, and is capable of achieving comparable results to strong diffusion and autoregressive baselines.
There are also limitations to our work. For example, we have exclusively relied on pretrained autoencoders for simplicity, but it leaves the question of a potential joint latent-NF model design unexplored. Moreover, in this work we have primarily focused on training high-quality models, which comes at the cost of un-optimized inference speed. Additionally, our evaluation has been restricted to class-and text-conditional image generation on standard benchmarks; how well the approach generalizes to other modalities (e.g., video, 3D scenes) or more diverse, real-world data distributions remains to be seen.
this section cite: []

Section: References
Ref_id:b0 Title: Text-to-image diffusion models with an ensemble of expert denoisers Year: (2022)
Ref_id:b1 Title: Improving image generation with better captions Year: (2023)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: Text-to-image generation via masked generative transformers Year: (2023-07)
Ref_id:b4 Title: Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts Year: (2021)
Ref_id:b5 Title: Pixartalpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2023)
Ref_id:b6 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b7 Title: Scaling instruction-finetuned language models Year: (2022)
Ref_id:b8 Title: ImageNet: A Large-scale Hierarchical Image Database Year: (2009)
Ref_id:b9 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b10 Title: Cogview: Mastering text-to-image generation via transformers Year: (2021)
Ref_id:b11 Title: Non-linear independent components estimation Year: (2014)
Ref_id:b12 Title: Density estimation using real nvp Year: (2016)
Ref_id:b13 Title: The llama 3 herd of models Year: (2024)
Ref_id:b14 Title: Tweedie's formula and selection bias Year: (2011)
Ref_id:b15 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b16 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b17 Title: Scaling autoregressive text-to-image generative models with continuous tokens Year: (2024)
Ref_id:b18 Title: Make-a-scene: Scenebased text-to-image generation with human priors Year: (2022)
Ref_id:b19 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2023)
Ref_id:b20 Title:  Year: (2016)
Ref_id:b21 Title: Free-form continuous dynamics for scalable reversible generative models Year: (2018)
Ref_id:b22 Title: Kaleido diffusion: Improving conditional diffusion models with autoregressive latent modeling Year: (2024)
Ref_id:b23 Title: Dart: Denoising autoregressive transformer for scalable text-to-image generation Year: (2024)
Ref_id:b24 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b25 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b26 Title: Classifier-free diffusion guidance Year: (2021)
Ref_id:b27 Title: Flow++: Improving flow-based generative models with variational dequantization and architecture design Year: (2019)
Ref_id:b28 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b29 Title: Imagen video: High definition video generation with diffusion models Year: (2022)
Ref_id:b30 Title: Cascaded diffusion models for high fidelity image generation Year: (2022)
Ref_id:b31 Title: A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines Year: (1989)
Ref_id:b32 Title: Scalable adaptive computation for iterative generation Year: (2022)
Ref_id:b33 Title: Scaling up gans for text-to-image synthesis Year: (2023)
Ref_id:b34 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b35 Title: Guiding a diffusion model with a bad version of itself Year: (2024)
Ref_id:b36 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b37 Title: Improved variational inference with inverse autoregressive flow Year: (2016)
Ref_id:b38 Title: Autoregressive image generation using residual quantization Year: (2022)
Ref_id:b39 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b40 Title: Fractal generative models Year: (2025)
Ref_id:b41 Title: A survey of multimodel large language models Year: (2024)
Ref_id:b42 Title: Common Objects in Context. European Conference on Computer Vision Year: (2014)
Ref_id:b43 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b44 Title: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b45 Title: Gpt-4o system card Year: (2024-04-12)
Ref_id:b46 Title: Masked autoregressive flow for density estimation Year: (2017-12-04)
Ref_id:b47 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b48 Title: Sdxl: improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b49 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b50 Title: Zeroshot text-to-image generation Year: (2021)
Ref_id:b51 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b52 Title: Variational inference with normalizing flows Year: (2015-07)
Ref_id:b53 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b54 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b55 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b56 Title: Chameleon: Mixed-modal early-fusion foundation models Year: (2024)
Ref_id:b57 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b58 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b59 Title: Generative infinite-vocabulary transformers Year: (2024)
Ref_id:b60 Title: An autoregressive generative model of raw images and text Year: (2024)
Ref_id:b61 Title: Attention Is All You Need Year: (2017)
Ref_id:b62 Title: Next-token prediction is all you need Year: (2024)
Ref_id:b63 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2024)
Ref_id:b64 Title: Scaling autoregressive models for content-rich text-to-image generation Year: (2022)
Ref_id:b65 Title: Normalizing flows are capable generative models Year: (2024)
Ref_id:b66 Title: Fast training of diffusion models with masked transformers Year: ()
Ref_id:b67 Title: Learning stackable and skippable LEGO bricks for efficient, reconfigurable, and variable-resolution diffusion modeling Year: ()
Ref_id:b68 Title: Transfusion: Predict the next token and diffuse images with one multi-modal model Year: (2024)
