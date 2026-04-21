Title: Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment
Abstract: Figure 1: Extreme super-resolution of photorealistic images by CoZ with up to 64× magnification (top) and 256× magnification (bottom). Fine details such as textures on a wall, wrinkles on a flag, and leaf veins are clearly seen.

Section: Introduction
The field of generative modeling has witnessed remarkable progress, enabling the synthesis of highly realistic data across various modalities, including images, text, and audio. A key application benefiting from these advancements is single-image super-resolution (SISR), which aims to reconstruct highresolution (HR) details from a low-resolution (LR) input image. Super-resolution is a problem of core interest for effectively bridging the gap between low-cost imaging sensors and high-fidelity visual information; its usages range from enhancing consumer photographs and legacy media to improving critical details in medical imaging, satellite surveillance, and scientific visualization [2,29,31,41,44]. The standard approach to SISR is based on the posterior probability distribution:
p(x H | x L )(1)
where the goal is to sample a plausible HR image x H for a given input LR image x L . However, the mapping from x L to x H is highly complex and fundamentally ill-posed: a single LR image can correspond to a multitude of plausible HR images. This makes directly modeling the distribution extremely challenging for large magnification factors, and early attempts relying on interpolation or regression often produced blurry results [11,14,20,51]. Recent emergence of powerful generative models (e.g., diffusion-based models) has led to significant advancement in this task, providing strong generative priors over natural images that enable the synthesis of realistic textures and details consistent with the low-resolution input.
Specifically, existing methods leveraging such generative priors largely fall into two categories. One line of work frames SR as an inverse problem, utilizing a pre-trained generative model as a prior during inference time to find a realistic HR image consistent with the LR input [6-9, 21, 22].
While such inverse problem-solving methods benefit from being training-free, they typically require lengthy iterative optimization or sampling processes at inference time to enforce data consistency (i.e., ensuring the downsampled HR prediction matches the original LR input), making them computationally expensive. Another line of work aims to incorporate this data consistency directly into the model's training objective, thereby enabling much faster inference [32,43,48,49,55,56]. Modern state-of-the-art models within this category are capable of producing high-quality super-resolved images, even in a single inference step [48,56].
However, these fast, trained super-resolution models suffer from a significant limitation: they are inherently upper-bounded by their training configuration and tend to collapse when presented with inputs requiring magnification beyond what they were trained on [23,25,58]. This failure occurs because the model's internal representations and learned restoration functions are tightly coupled to the specific scale and degradation seen during training [35]. Applying it outside this domain violates its learned assumptions, leading to severe artifacts, blurry outputs, or a complete failure to generate meaningful high-frequency details [12,14,20]. This lack of robustness severely restricts the practical applicability of these otherwise powerful models, demanding new models to be trained when the desired magnification factor exceeds what can be currently provided, which is highly inefficient.
In this work, we therefore propose to solve a fundamental question: How can we effectively utilize super-resolution models to explore much higher resolutions than they were originally trained for? Solving this question is critical in that it addresses the practical need for flexible and arbitrary-scale super-resolution, allowing users to magnify images to desired levels without being constrained by model training specifics. Furthermore, training models for extremely high magnification factors (e.g., 16x, 32x) directly is often computationally prohibitive due to memory and time constraints [46]. Enabling the extension of existing, well-trained models (e.g., 4x SR models) to higher factors offers a significantly more resource-efficient pathway to achieving extreme resolutions.
To address these fundamental challenges, we present Chain-of-Zoom (CoZ), a novel framework for achieving extreme-resolution image generation beyond the training configurations of conventional super-resolution models. Specifically, we introduce intermediate scale-state modeling to bridge the gap between a low-resolution (LR) input and a high-resolution (HR) target image. These intermediate scale-states enable the decomposition of the conditional distribution in Eq. (1) into a series of tractable components, forming the basis of a scale-level autoregressive (AR) framework. Within this framework, models can progressively generate high-quality images at resolutions previously considered unattainable. In particular, building on the scale-level AR-2 model, we further propose a multi-scale-aware prompt extraction technique. This approach leverages Vision-Language Models (VLMs) to extract descriptive text prompts by attending to multiple scale-states throughout the zooming process, enabling semantically aligned and coherent super-resolution. This is from the observation that at extreme resolutions, conditioning provided by the original signal x L becomes insufficient, thus leading to unreasonable hallucinations by the SR model in cases.
Furthermore, to obtain text prompts of even richer detail that aligns with human preference, we fine-tune the prompt-extraction VLM under a novel RLHF pipeline leveraging GRPO [34]. A core part of this pipeline is the utilization of a critic VLM to score the outputs of the prompt extraction VLM, thus guiding it to produce prompts more aligned to human preference. Incorporated into the CoZ framework, our final VLM model successfully guides the super-resolution process towards reasonable high-quality results.
In summary, our contributions are as follows:
• We present Chain-of-Zoom, a scale-level autoregressive framework that decomposes superresolution into a sequence of intermediate scale-states and multi-scale-aware prompts, enabling any existing SR model to reach much higher magnifications without retraining.
• We propose a novel RL pipeline for tuning prompt-extraction VLMs with GRPO. This pipeline incorporates appropriate reward functions and a critic reward model to endue multi-scale aware reasoning capabilities to the prompt-extraction VLM.
this section cite: ['b1', 'b28', 'b30', 'b40', 'b43', 'b10', 'b13', 'b19', 'b50', 'b31', 'b42', 'b47', 'b48', 'b54', 'b55', 'b47', 'b55', 'b22', 'b24', 'b57', 'b34', 'b11', 'b13', 'b19', 'b45', 'b33']

Section: Related Work
Multi-Scale Image Generation and Super-Resolution. Unconditional multi-scale generators synthesize ever-larger images by passing coarse outputs through successive refinement stages. Cascaded Diffusion Models [17] pioneer this coarse-to-fine pipeline, while AnyresGAN [3], Scalespace-GAN [47], Generative Powers of Ten [45], ZoomLDM [53], and Make-a-Cheap-Scaling [16] share weights across latent zoom levels to reach megapixel resolutions. Because they are generation-based, these methods do not enforce consistency with a given low-resolution input. For true SR, PULSE [27] searches a GAN latent space, and Zoomed In, Diffused Out [28] alternates diffusion denoising with explicit up-sampling, but both do not explore extreme resolutions as in this work.
this section cite: ['b16', 'b2', 'b46', 'b44', 'b52', 'b15', 'b26', 'b27']

Section: Autoregressive Factorizations.
Classic autoregressive models such as PixelCNN, PixelRNN [39,40] and VAR [38] predict spatial tokens sequentially within a fixed resolution. Pixel Recursive SR [10] extends this to super-resolution by autoregressing over pixels after each enlargement-effective for small factors but computationally prohibitive at extreme scales. The proposed CoZ instead autoregresses over scale-states: we factorize p(x H | x L ) into a tractable sequence of intermediate zoom distributions, enabling arbitrarily high magnifications without retraining at every factor.
Diffusion-Based Super-Resolution. Diffusion models have become the de-facto approach for high-fidelity SISR. SR3 [32] first denoised noisy HR guesses into realistic outputs with diffusion models. StableSR [43] reuses a diffusion prior for faster convergence, and prompt-aware variants (e.g., SeeSR [49], SUPIR [55]) add textual conditioning to bolster semantic faithfulness. OSEDiff [48] distills the multi-step chain into a one-step denoising. Because of its accuracy and efficiency, we adopt OSEDiff as the backbone SR module in our CoZ demonstrations. However, CoZ is model-agnostic: the same scaling strategy can wrap any existing text-guided diffusion (or non-diffusion) SR network.
this section cite: ['b38', 'b39', 'b37', 'b9', 'b31', 'b42', 'b48', 'b54', 'b47']

Section: RL for Vision-Language Guidance.
Reinforcement learning with human feedback (RLHF) is now widely used to align VLM behaviour with user preference. Early vision-grounded efforts such as LLaVA-RLHF [36] and LLaVACritic [50] employ reward models or critic networks to refine image-conditioned dialogue. Generalized Reward Policy Optimization (GRPO) was introduced by Shao et al. [34] as a policy-space alternative to PPO [33]. GRPO has since been adopted in vision tasks outside SR: Seg-Zero [26] uses GRPO to train VLMs for open-set semantic segmentation, while MetaSpatial [30] applies it to 3-D spatial reasoning in virtual environments. Building on these precedents, we are the first to bring GRPO to prompt-extraction in super-resolution. Our pipeline fine-tunes a prompt-extraction VLM with a composite reward objective unexplored in prior SR work.
3 Chain-of-Zoom
this section cite: ['b35', 'b49', 'b33', 'b32', 'b25', 'b29']

Section: Intermediate Scale-State Modeling
In the CoZ framework, we propose to bridge the gap between a target HR image x H ∈ R dn and an input LR image x L ∈ R d0 by introducing intermediate scale-states x i ∈ R di . Suppose that an image generative process is modeled as a sequence (x 0 , x 1 , ..., x n ) where x 0 := x L , x n := x H , and consecutive states have dimension ratio s (i.e. d i = sd i-1 ) larger than 1. Under the Markov assumption, the joint distribution could be modeled as p(
x 0 , x 1 , ..., x n ) = p(x 0 ) n i=1 p(x i |x i-1
). However, if the model follows a Markov chain structure, relying solely on the transition probability p(x i |x i-1 ) leads to loss of high-frequency details as n increases (see Fig. 3). Inspired by recent work in inverse problems [8,21] that demonstrate the effectiveness of text embeddings in reducing the solution space and improving super-resolution between consecutive scales, we therefore introduce latent variables c i through text embeddings. The text prompt extraction supplements information of the overall zoom process.
Importantly, to reduce hallucinations caused by incorrect text guidance across scale, we find that multi-scale aware text extraction is necessary by feeding x i-1 and the coarser state x i-2 in prompt generation, leading to the conditional probability for the prompt:
p ϕ (c i | x i-1 , x i-2 ).
(2) Therefore, instead of using the Markov assumption, we propose AR-2 modeling of the image generative process with multi-scale-aware prompts as latent variables:
p(x 0 , x 1 , ..., x n ) = p(x 0 , x 1 ) n i=2 p(x i |x i-1 , x i-2 ),(3)
p(x i |x i-1 , x i-2 ) = p(x i |x i-1 , x i-2 , c i )p(c i |x i-1 , x i-2 )dc i .(4)
Then, the joint distribution of the sequence
(x 0 , c 1 , x 1 , ..., c n , x n ) is expressed as follows: Proposition 1.
Given a sequence of scale-states x i that follows a AR-2 structure and latent variables c i that satisfy Eq. ( 2), the joint distribution is expressed as
p(x 0 , c 1 , x 1 , ..., c n , x n ) = p(x 0 , c 1 , x 1 ) n i=2 p(x i |x i-1 , x i-2 , c i )p(c i |x i-1 , x i-2 ).(5)
Now, our objective function is maximizing the likelihood of the entire joint distribution of x i and c i . Taking the logarithm of Eq. ( 5), we get the objective function to be maximized:
L = log p(x 0 , c 1 , x 1 ) Linit + n i=2 log p(x i |x i-1 , x i-2 , c i ) LSR + n i=2 log p(c i |x i-1 , x i-2 ) LVLM (6
)
where L init represents the initial super-resolution step.
this section cite: ['b7', 'b20']

Section: Training Objective
The additive structure of the components in Eq. ( 6) allows for the independent optimization of each term. We achieve this via next x i prediction and next c i prediction, using parameterized models θ and ϕ, respectively.
Next x i prediction. The training objective L SR represents the likelihood of x i given previous scale-states x i-1 , x i-2 and description c i for x i . Under the assumption that the distribution p(
x i |x i-1 , x i-2 , c i ) := N (x i ; f θ (x i-1 , x i-2 , c i ), σ 2 I)
is Gaussian, where the parameterized model f θ predicts the conditional mean of the distribution, the likelihood of x i is equivalent to
log p(x i |x i-1 , x i-2 , c i ) = - 1 2σ 2 ∥x i -f θ (x i-1 , x i-2 , c i )∥ 2 + C(7)
where C = -di 2 log(2πσ 2 ). To reduce the computational complexity of training f θ , our key idea is that its dependency to x i-2 is only through the multi-scale-aware prompt, i.e.
c i = c i (x i-1 , x i-2 ), leading to f θ (x i-1 , x i-2 , c i ) = f θ (x i-1 , c i (x i-1 , x i-2 )).
p ϕ (c i | x i-1 , x i-2 ) = Ti t=1 p ϕ (c i,t | x i-1 , , x i-2 , c i,<t )(8)
where c i,<t = (c i,1 , • • • , c i,t-1 ). Maximizing the log-likelihood log p ϕ (c i | x i-1 , x i-2 ) therefore amounts to minimizing the negative log-likelihood (cross-entropy) for each token: Eq. ( 9) is exactly the standard next-token cross-entropy loss used to pre-train modern VLMs; hence our framework can employ any off-the-shelf VLM whose weights already maximize this objective.
L (i) VLM = -log p ϕ (c i | x i-1 , x i-2 ) = - Ti t=1 log p ϕ (c i,t | x i-1 , x i-2 , c i,<t ).(9)
Inference. Given pre-trained parameterized models θ and ϕ, the sequence (x 0 , c 1 , x 1 , ..., c n , x n ) can be generated recursively. Starting from the low-resolution image x L = x 0 , a description for the next scale,
c 1 ∼ p ϕ (c 1 | x 0 ), is first sampled. Then, the next scale state is generated by sampling x 1 ∼ p θ (x 1 | x 0 , c 1 ). For subsequent steps, the description at scale i is sampled as c i ∼ p ϕ (c i | x i-1 , x i-2 )
, followed by sampling the image at that scale as
x i ∼ p θ (x i | x i-1 , x i-2 , c i ).
This sequential sampling process generates specific, plausible high-resolution outputs x n without needing to model the full marginal distribution p(x 0 , ..., x n ) explicitly. When using SR backbone models that require input and output dimensions to be identical (e.g., Stable-diffusion-based SR models [43,48,49,55]), a fixed-size window is cropped from the HR image and resized to the required dimension. Thus, super-resolution operates in local regions, and achieving outputs of entire images would require multiple runs of CoZ.
this section cite: ['b42', 'b47', 'b48', 'b54']

Section: Training Multi-Scale-Aware Prompt Extraction using RL
At extreme magnification factors, the visual evidence in the input image becomes extremely sparse, causing the SR backbone model to rely more heavily on text prompts. To curb the ensuing drift towards implausible high-frequency hallucinations, we fine-tune the prompt-extraction VLM so that its textual guidance aligns with human aesthetic and semantic preferences. Our fine-tuning pipeline (Fig. 4) adopts Generalized Reward Policy Optimization (GRPO). For each zoom step i, the VLM receives multi-scale image crops (x i-2 , x i-1 ) and produces a candidate prompt c i . The prompt is scored by a set of task-specific reward functions, and the weighted sum R(c i ) drives the GRPO update to align the VLM prompts with human preference. The overall reward R(c i ) is a weighted sum of three components, each targeting a distinct failure mode observed during preliminary experiments:
R(c i ) = w critic R critic + w phrase R phrase + w rep R rep (10
)
Critic Preference Reward (R critic ). A stronger vision-language critic VLM judges the candidate prompt in the context of the input multi-scale image crops and assigns a raw score in [0, 100]. We linearly rescale this score to [0, 1] and treat it as a proxy for human preference, thereby imbuing the prompt-extraction VLM with the critic VLM's higher-level semantic priors.
this section cite: []

Section: Phrase-Exclusion Reward (R phrase ).
Multi-image conditioning occasionally leads the promptextraction VLM to emit viewpoint markers such as "first image" or "second image," which are meaningless to the downstream SR model. We therefore issue a reward of 1 if none of a predefined blacklist of such phrases appear, and 0 otherwise.
this section cite: []

Section: Repetition Penalty (R rep ).
Following Yeo et al. [54], we compute the fraction of repeated n-grams in the prompt and give a negative reward (down to -1) for a higher repetition ratio.
this section cite: ['b53']

Section: Experiments

this section cite: []

Section: Experimental Settings
We adopt the setup of prior work [49,48] and train OSEDiff [48] as the backbone SR model with the LSDIR [24] dataset and 10K images from FFHQ [18]. We use Stable Diffusion 3.0 [13] as the backbone diffusion model and adopt a coarse-to-fine training strategy: first training on random degradation, and then training specifically for 4× magnifications. Text guidance is provided by Degradation-Aware Prompt Extractor (DAPE) [49] as the naive prompt extractor, while Qwen2.5-VL-3B-Instruct [37] is used as the prompt-extraction VLM. RLHF training with GRPO is performed with InternVL2.5-8B [5] as the critic VLM. The same dataset used for training the backbone SR model is also used for GRPO training, and weights are given as: w critic = 1.0, w phrase = 0.5, w rep = 0.5.
Evaluation is performed on the training datasets of DIV2K [1] and DIV8K [15], consisting of 800 images and 1500 images, respectively. Each image is resized and center-cropped to resolution of 512 × 512 to be input to the SR model. For four recursions, the HR image of the previous zoom is center-cropped and resized by a scale of 4 back to the resolution of 512 × 512.
this section cite: ['b48', 'b47', 'b47', 'b23', 'b17', 'b12', 'b48', 'b36', 'b4', 'b0', 'b14']

Section: Comparison Results
We perform comparison across four recursions for various methods. Specifically, we compare between nearest neighbor interpolation, direct magnification via one-step SR, and three versions of the proposed CoZ leveraging different prompts (i.e., Null, DAPE, VLM).
this section cite: []

Section: Qualitative Comparison.
Qualitative results in Fig. 5 show that nearest neighbor interpolation and one-step direct SR fall off at higher scales, while CoZ variants produce images of better quality. Thus, incorporating VLM prompts helps overcome the sparsity of the original input signal. Quantitative Comparison. Quantitative results are given in Tab. 1. Due to the non-availability of ground-truth images for 256× magnifications, we follow [27] and evaluate performance on no-reference perceptual metrics. Specifically, we use the metrics NIQE [57], MUSIQ [19], MANIQApipal [52], CLIPIQA [42] for a thorough evaluation. At low scales (i.e., Scale 4×), difference between methods is minimal, but at high scales (i.e., Scales 64×, 256×) the proposed framework shows consistently better performance. Furthermore, prompts by DAPE show comparable performance at low scales but fall off at higher scales, while VLM-generated prompts exhibit significantly better performance, supporting our claim that prompt-extraction by VLMs make up for the deficient visual conditioning provided by the initial image.
Quantitative Comparison with Baseline Methods. In Tab. 2, we further provide quantitative comparison with baseline methods: arbitrary-scale SR methods (LIIF [4]) and direct super-resolution of diffusion-based SR methods (SeeSR [49], S3Diff [56]). All three baselines show greatly degraded performance at high magnifications. For the case of S3Diff, we additionally show quantitative results for applying CoZ to the pretrained, freely accessible S3Diff, leveraging our multi-scale aware GRPO fine-tuned VLM. All results clearly confirm the significant performance improvement by CoZ. Additional results for performing CoZ with OSEDiff leveraging the Stable Diffusion v2.1 backbone is provided in Appendix E.
this section cite: ['b26', 'b56', 'b18', 'b51', 'b41', 'b3', 'b48', 'b55']

Section: GRPO for VLM
GRPO Training. Reward graphs for training the prompt-extraction VLM with different critic VLMs are shown in Fig. 6 and Fig. 7. When using InternVL2.5-8B [5] as the critic VLM, phrase exclusion reward and repetition penalty converge to 1.00 and 0.00 (respectively) in the early stages of training, while the critic reward increases gradually throughout the training process. Similar trends are observed when using Qwen2.5-VL-7B-Instruct [37] as the critic VLM, proving the robustness of our method. Additional quantitative results for this case is provided in Appendix F.
this section cite: ['b4', 'b36']

Section: Preference Alignment.
Using an off-the-shelf VLM for prompt-extraction can cause unwanted hallucinations to occur in the zoom process. An example case is shown in Fig. 8 (Top), where the off-the-shelf VLM generates improper prompts due to insufficient knowledge of the initial image at high magnifications. By inducing the VLM to generate multi-scale-aware prompts by conditioning on (x i-1 , x i-2 ), we can produce more suitable prompts Fig. 8 (Middle). Finally, using the VLM fine-tuned with GRPO we can produce high-quality samples while reducing unwanted hallucinations as in Fig. 8 (Bottom). We further prove that the VLM after undergoing GRPO training is better aligned with human preference through user study. For this, we follow prior work [27], and perform a MOS (mean-opinion-score) test on various samples. Results and details are included in Appendix D.
this section cite: ['b26']

Section: Conclusion
This paper tackles the long-standing scalability gap in single-image super-resolution: state-of-the-art models excel at their trained scale factors yet fail when asked to enlarge images far beyond that range. Specifically, we introduced Chain-of-Zoom (CoZ), a scale-level autoregressive framework that transforms any existing SR backbone into an extreme-magnification engine by decomposing the LR to HR mapping into a sequence of intermediate scale-states and multi-scale-aware prompts. CoZ is model-agnostic, requires no retraining of the base network, and thus offers a cost-effective path up to extreme resolutions. In particular, to maintain semantic coherence as visual evidence thins out, we leverage a multi-scale-aware prompt extractor driven by a VLM fine-tuned through a GRPObased RLHF pipeline. Overall, CoZ yields sharp, realistic results at extreme scales while keeping inference efficient. By decoupling super-resolution performance from fixed training magnifications and demonstrating the value of aligned textual guidance, our work opens new avenues for resourcefrugal image enhancement and lays a foundation for future exploration of learned zoom policies, domain-specific reward functions, and adaptive backbone selection.
Limitation and Potential Negative Impacts. While CoZ enables extreme super-resolution with high visual fidelity, it requires repeated application for extreme magnification, which may cause error accumulation over iterations. Moreover, high-fidelity generation from low-resolution inputs may raise concern regarding misinformation or unauthorized reconstruction of sensitive visual data.
this section cite: []

Section: References
Ref_id:b0 Title: Ntire 2017 challenge on single image super-resolution: Dataset and study Year: (2017)
Ref_id:b1 Title: Imaging intracellular fluorescent proteins at nanometer resolution. science Year: (2006)
Ref_id:b2 Title: Any-resolution training for high-resolution image synthesis Year: (2022)
Ref_id:b3 Title: Learning continuous image representation with local implicit image function Year: (2021)
Ref_id:b4 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b5 Title: Improving diffusion models for inverse problems using manifold constraints Year: (2022)
Ref_id:b6 Title: Diffusion posterior sampling for general noisy inverse problems Year: (2023)
Ref_id:b7 Title: Prompt-tuning latent diffusion models for inverse problems Year: (2023)
Ref_id:b8 Title: Decomposed diffusion sampler for accelerating large-scale inverse problems Year: (2024)
Ref_id:b9 Title: Pixel recursive super resolution Year: (2017)
Ref_id:b10 Title: Learning a deep convolutional network for image super-resolution Year: (2014)
Ref_id:b11 Title: Image super-resolution using deep convolutional networks Year: (2015)
Ref_id:b12 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b13 Title: Example-based super-resolution Year: (2002)
Ref_id:b14 Title: Div8k: Diverse 8k resolution image dataset Year: (2019)
Ref_id:b15 Title: Make a cheap scaling: A self-cascade diffusion model for higher-resolution adaptation Year: (2024)
Ref_id:b16 Title: Cascaded diffusion models for high fidelity image generation Year: (2022)
Ref_id:b17 Title: A style-based generator architecture for generative adversarial networks Year: (2019)
Ref_id:b18 Title: Musiq: Multi-scale image quality transformer Year: (2021)
Ref_id:b19 Title: Cubic convolution interpolation for digital image processing Year: (2003)
Ref_id:b20 Title: Regularization by texts for latent diffusion inverse solvers Year: (2023)
Ref_id:b21 Title: Flowdps: Flow-driven posterior sampling for inverse problems Year: (2025)
Ref_id:b22 Title: Photo-realistic single image superresolution using a generative adversarial network Year: (2017)
Ref_id:b23 Title: Lsdir: A large scale dataset for image restoration Year: (2023)
Ref_id:b24 Title: Enhanced deep residual networks for single image super-resolution Year: (2017)
Ref_id:b25 Title: Seg-zero: Reasoningchain guided segmentation via cognitive reinforcement Year: (2025)
Ref_id:b26 Title: Pulse: Self-supervised photo upsampling via latent space exploration of generative models Year: (2020)
Ref_id:b27 Title: Zoomed in, diffused out: Towards local degradation-aware multi-diffusion for extreme image super-resolution Year: (2024)
Ref_id:b28 Title: Multi-input cardiac image super-resolution using convolutional neural networks Year: (2016)
Ref_id:b29 Title: Metaspatial: Reinforcing 3d spatial reasoning in vlms for the metaverse Year: (2025)
Ref_id:b30 Title: MR image reconstruction from highly undersampled k-space data by dictionary learning Year: (2010)
Ref_id:b31 Title: Image super-resolution via iterative refinement Year: (2021)
Ref_id:b32 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b33 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b34 Title: zero-shot" super-resolution using deep internal learning Year: (2018)
Ref_id:b35 Title: Aligning large multimodal models with factually augmented rlhf Year: (2023)
Ref_id:b36 Title: Qwen2.5-vl Year: (2025-01)
Ref_id:b37 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b38 Title: Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders Year: (2016)
Ref_id:b39 Title: Pixel recurrent neural networks Year: (2016)
Ref_id:b40 Title: Deep residual learning for single-image super-resolution of multi-spectral satellite imagery Year: (2019)
Ref_id:b41 Title: Exploring clip for assessing the look and feel of images Year: (2023)
Ref_id:b42 Title: Exploiting diffusion prior for real-world image super-resolution Year: (2024)
Ref_id:b43 Title: A comprehensive review on deep learning based remote sensing image super-resolution methods Year: (2022)
Ref_id:b44 Title: Generative powers of ten Year: (2024)
Ref_id:b45 Title: Deep learning for image super-resolution: A survey Year: (2020)
Ref_id:b46 Title: Learning images across scales using adversarial training Year: (2024)
Ref_id:b47 Title: One-step effective diffusion network for real-world image super-resolution Year: (2024)
Ref_id:b48 Title: Seesr: Towards semantics-aware real-world image super-resolution Year: (2024)
Ref_id:b49 Title: Llava-critic: Learning to evaluate multimodal models Year: (2024)
Ref_id:b50 Title: Image super-resolution via sparse representation Year: (2010)
Ref_id:b51 Title: Maniqa: Multi-dimension attention network for no-reference image quality assessment Year: (2022)
Ref_id:b52 Title: Latent diffusion model for multi-scale image generation Year: (2024)
Ref_id:b53 Title: Demystifying long chain-ofthought reasoning in llms Year: (2025)
Ref_id:b54 Title: Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild Year: (2024)
Ref_id:b55 Title: Degradation-guided one-step image super-resolution with diffusion priors Year: (2024)
Ref_id:b56 Title: A feature-enriched completely blind image quality evaluator Year: (2015)
Ref_id:b57 Title: Image super-resolution using very deep residual channel attention networks Year: (2018)
Ref_id:b58 Title: Swift: a scalable lightweight infrastructure for fine-tuning Year: (2025)
