Title: Unlocking Dataset Distillation with Diffusion Models
Abstract: Dataset distillation seeks to condense datasets into smaller but highly representative synthetic samples. While diffusion models now lead all generative benchmarks, current distillation methods avoid them and rely instead on GANs or autoencoders, or, at best, sampling from a fixed diffusion prior. This trend arises because naive backpropagation through the long denoising chain leads to vanishing gradients, which prevents effective synthetic sample optimization. To address this limitation, we introduce Latent Dataset Distillation with Diffusion Models (LD3M), the first method to learn gradient-based distilled latents and class embeddings endto-end through a pre-trained latent diffusion model. A linearly decaying skip connection, injected from the initial noisy state into every reverse step, preserves the gradient signal across dozens of timesteps without requiring diffusion weight fine-tuning. Across multiple ImageNet subsets at 128 × 128 and 256 × 256, LD3M improves downstream accuracy by up to 4.8 percentage points (1 IPC) and 4.2 points (10 IPC) over the prior state-of-the-art. The code for LD3M is provided at https://github.com/Brian-Moser/prune_and_distill.

Section: Introduction
Large-scale datasets fuel the advancements in modern computer vision but demand substantial computational resources and raise scalability concerns [27,2,25,14]. Dataset distillation emerges as a compelling solution, aiming to synthesize a small set of information-rich samples that preserve the essence of the original dataset [35,42,4]. While early methods operated directly in pixel space, a promising recent direction involves leveraging powerful generative models as priors [13,1]. By optimizing compact latent codes instead of raw pixels, these approaches, exemplified by GLaD using StyleGAN-XL [5], can generate higher-resolution synthetic images (128 × 128 and beyond) that generalize better across diverse network architectures.
However, GAN-based priors like in GLaD suffer from complex multi-space latent optimization and require cumbersome inversion processes for initialization [5,43,3,21]. Diffusion models [18,28], having surpassed GANs as the state-of-the-art image generators [9], represent a natural next step. Yet, inherent vanishing gradients severely hinder their direct application to dataset distillation. Optimizing latents through the long denoising chain leads to exponentially decaying gradients, which prevents effective learning of the synthetic data [19,17].
Mathematically, if Z is the initial latent code to be optimized and z 0 the final denoised state after T steps, the gradient ∂L/∂Z is a product of Jacobians:
∂L ∂Z = ∂L ∂z 0 • T t=1 ∂z t-1 ∂z t • ∂z T ∂Z .(1)
If the norm of each Jacobian ∂z t-1 /∂z t is bounded by λ < 1, the product term T t=1 (∂z t-1 /∂z t ) diminishes towards zero as T increases. This gradient decay is empirically observable and significant: Our analysis confirms that in standard diffusion models, gradient norms for Z decrease nearly tenfold as T increases from 10 to 90 (see Table 1).
Table 1: Gradient norms for the initial latent code Z across different maximum diffusion steps T for fixed inputs. The decrease in norm as T increases empirically demonstrates the vanishing gradient problem, which severely hinders the optimization of Z through the standard reverse diffusion process.
T 10 20 30 40 50 60 70 80 90 ∥L/∂Z∥ × 10 4 58. 1 33 .5 19.8 15.4 13.9 12.1 10.4 8.7 6.5This critical bottleneck has forced prior diffusion-based distillation attempts to circumvent end-to-end optimization entirely, resorting instead to sampling or selecting fixed representations from pre-trained models [12,34,16]. These approaches, while computationally faster, forfeit the fine-grained gradientmatching optimization crucial for potential benefits like enhanced privacy of distilled data or their robustness against adversarial attacks [7,37,44,10].
To unlock diffusion models for true dataset distillation, we introduce Latent Dataset Distillation with Diffusion Models (LD3M). Our core contribution is a tailored modification to the reverse diffusion process (Equation 7) that introduces linearly decaying residual connections, specifically designed to enhance gradient flow for optimizing latent representations Z and conditioning codes c in the context of dataset distillation. This mechanism enables, for the first time, effective end-to-end optimization of distilled latent codes Z and class embeddings c through a pre-trained latent diffusion model. LD3M is readily compatible with existing distillation objectives and diffusion model architectures. Experiments across numerous ImageNet subsets demonstrate that LD3M significantly outperforms the state-of-the-art at 128 × 128 and 256 × 256 resolutions, achieving superior cross-architecture generalization, i.e., 4.8 percentage points (1 IPC) and 4.2 points (10 IPC), and faster distillation times.
this section cite: ['b26', 'b1', 'b24', 'b13', 'b34', 'b41', 'b3', 'b12', 'b0', 'b4', 'b4', 'b42', 'b2', 'b20', 'b17', 'b27', 'b8', 'b18', 'b16', 'b11', 'b33', 'b15', 'b6', 'b36', 'b43', 'b9']

Section: Preliminaries
Dataset Distillation: Let T = (X r , Y r ), where X r ∈ R N ×H×W ×C , be a real image classification dataset and N its cardinality. The goal is to compress T into a small synthetic set S = (X s , Y s ), where X s ∈ R M ×H×W ×C , where M is the total number of synthetic samples with M = C • IP C, C the number of classes and IP C the Images Per Class (IPC). We aim to achieve M ≪ N with
S * = arg min S L(S, T )(2)
where L is a distillation loss between the distilled set S and the real dataset T . Common choices for L include matching gradients (Dataset Condensation, DC [42]), feature distributions (Distribution Matching, DM [41]), or model parameter trajectories (Matching Training Trajectories, MTT [4]). We refer to the supplementary material for detailed definitions.
this section cite: ['b41', 'b40', 'b3']

Section: Dataset Distillation with Generative Priors:
To improve the quality, resolution, and generalization of distilled images, recent work incorporates deep generative models as priors [5]. Instead of optimizing pixels directly, they optimize latent codes Z ∈ R M ×h×w×d with h • w • d ≪ H • W • C, fed into a pre-trained generator D. The optimization objective becomes
Z * = arg min Z L(D(Z), T ).(3)
3 Related Work GAN Priors (GLaD): GLaD [5] pioneered distillation with generative priors using a pre-trained StyleGAN-XL [30]. While successful, it inherits GAN-specific drawbacks: (1) Optimizing the Z is noised to initialize the reverse diffusion at z T . A pre-trained LDM denoiser iteratively refines the state (z t → z t-1 ). Key innovation: Residual connections (red arrows) inject z T with linearly decaying weight into each step (Equation 7), enhancing gradient flow. The final latent z 0 is decoded (D) into images S, which are optimized using a standard distillation algorithm.
complex, multi-level latent space (W + ) required for high quality is computationally demanding [43].
(2) Initializing latent codes Z from real images x requires solving a costly GAN inversion problem (min Z L(x, D(Z))) [3,36], hindering standard initialization practices [21].
this section cite: ['b4', 'b29', 'b42', 'b2', 'b35', 'b20']

Section: Prior Diffusion-based Distillation Attempts:
The inherent vanishing gradient problem ( §1) significantly challenges using diffusion models for end-to-end distillation. Faced with this gradient barrier, existing diffusion methods for distillation have adopted non-optimization strategies:
• Sampling/Selection Methods: Minimax Diffusion [16] and D4M [34] use criteria to select or sample representative latents from a diffusion model, avoiding backpropagation to latents entirely. While faster, this bypasses gradient-based optimization that is needed for privacy or robustness, also crucial motivations for dataset distillation [7,37,44,10].
• Autoencoder-Only Methods: Duan et al. [12] leverage the pre-trained autoencoder from LDM but do not utilize the diffusion process itself. They optimize latent codes Z that are directly decoded by D, essentially using only the autoencoder, not its core denoising mechanism. This simplifies optimization but fails to exploit the diffusion prior.
In contrast, LD3M is designed to directly optimize latent codes Z by enabling gradient flow through the reverse process.
Residual Connections for Gradient Flow: Enhancing gradient flow in deep networks is commonly addressed using residual connections, famously introduced in ResNets [17] or LSTMs [19]. Similar ideas have appeared within diffusion models; for instance, SAGE [23] used residuals connecting to the initial noise to enable adversarial latent search. While conceptually related, our contribution differs significantly: we introduce structured, linearly decaying residuals injected at every step from the initial noisy latent z T , explicitly designed for end-to-end optimization of distilled latent codes through the diffusion chain for the unique characteristics of dataset distillation.
this section cite: ['b15', 'b33', 'b6', 'b36', 'b43', 'b9', 'b11', 'b16', 'b18', 'b22']

Section: Decoupled Distillation Methods:
Distinct from methods optimizing synthetic data via generative priors, another line of work decouples distillation from end-to-end training. Methods like SRe2L [40] and others [39,32] leverage statistics (e.g., from BatchNorm) of pre-trained networks to recover informative images, offering scalability benefits but following a fundamentally different optimization strategy than gradient-matching approaches like LD3M.
this section cite: ['b39', 'b38', 'b31']

Section: Latent Dataset Distillation with Diffusion Models (LD3M)
Our approach enables end-to-end optimization of synthetic data directly through a pre-trained LDM.
As shown in Figure 1, LD3M optimizes both initial latent codes Z and conditioning codes c. These are processed by a modified reverse diffusion process to generate expressive latent states z 0 , which are then decoded into images, S = D(z 0 ). This section details the core components: our modified diffusion sampling process designed to boost gradient flow ( §4.1), the efficient initialization strategy ( §4.2), and the gradient checkpointing used for memory efficiency ( §4.3).
this section cite: []

Section: Sampling Process
We leverage a pre-trained LDM [28] without fine-tuning its weights. Standard LDM operation involves a reverse diffusion process p θ that iteratively denoises a state z t , starting from noise z T , conditioned on an embedding c (typically derived from class labels) [38,26]. Each step t predicts a less noisy state z t-1 based on the current state z t and condition c. The standard update rule [18,29] calculates the subsequent state z t-1 using the predicted mean µ θ and variance σ 2 t :
µ θ (c, z t , γ t ) = 1 √ α t z t - 1 -α t √ 1 -γ t f θ (c, z t , γ t )(4)
z t-1 ← µ θ (c, z t , γ t ) + σ 2 t ε t ,(5)
where f θ is the LDM's pre-trained noise prediction network (usually a U-Net), α t and γ t relate to the noise schedule, and ε t ∼ N (0, I) is random noise added at step t.
this section cite: ['b27', 'b37', 'b25', 'b17', 'b28']

Section: Algorithm 1 Latent Dataset Distillation with Diffusion Models (LD3M)
Input: randomly selected collection X s , pre-trained encoder E, pre-trained decoder D, pre-trained denoiser µ θ with frozen parameters θ, noise levels σ t .
Z = E (X s ) z T ∼ q(z T | Z) for t = T, . . . , 1 do ε t ∼ N (0, I) z t-1 ← (1 -t T ) • µ θ (c, z t , γ t ) + t T • z T + σ 2 t ε t end for X syn ← D (z 0 ) Return: X syn
For dataset distillation, our goal is to learn the optimal initial latent representations Z and conditioning codes c that minimize the distillation loss L between the synthetic images and the target dataset:
Z * , c * = arg min Z,c L(D[p θ (z 0 |z T , c)], T ), where z T ∼ q(z T |Z).(6)
Here, p θ (z 0 |z T , c) denotes the final state z 0 resulting from the T -step reverse process starting from z T , which itself is a noised version of the learnable Z obtained via the forward process q. We want to highlight that the sampling of ε introduces stochasticity during inference. We fix, however, for one sampling phase, the residual variable z T to be constantly the same.
To effectively minimize distillation losses like gradient or trajectory matching (Equation 6), which rely on fine-grained alignment between synthetic and real data processing, requires guiding the generative process. While conditioning c provides class guidance, optimizing the initial latent Z offers the necessary degrees of freedom to shape the generated sample D (z 0 ) precisely. Relying solely on fixed Z or only optimizing c proved insufficient empirically, yielding suboptimal results resembling simple LAION-5B [31] data retrieval similar to D4M [34].
Vanishing gradients inherent in backpropagating through the T steps of Equation 5 present the primary obstacle to optimizing Equation 6 [19]. To overcome this, we introduce a simple yet effective modification to the reverse step, injecting a residual connection from the initial state z T :
z t-1 ← (1 -t T ) • µ θ (c, z t , γ t ) + t T • z T Modified Mean +σ 2 t ε t .(7)
Crucially, while this modification deviates from standard sampling aimed at matching the distribution of real images, its purpose here is to enable gradient flow for distillation optimization, a task where downstream utility, not photorealism [42,4], is paramount (see supplementary material for detailed discussion). In other words, our method breaks the sampler's fidelity to the original data distribution in order to better achieve the distillation objective of matching feature distributions.
In conclusion, the modification replaces the standard predicted mean µ θ with a weighted average of µ θ and the initial state z T . The weight of z T decreases linearly from 1 (at t = T ) to nearly 0 (as t → 0). This creates a direct pathway for gradients from the loss L (computed using z 0 ) back to z T , and thus to the learnable Z, bypassing the long chain of Jacobian products that causes gradients to vanish. The enhanced gradient flow to Z can be conceptually represented as:
∂L ∂Z = T t=1 1 - T -1 T • ∂L ∂z t • ∂z t ∂z t-1 • ... • ∂z 0 ∂Z Original (Decaying) Path + t T • ∂L ∂z t-1 ∂z t-1 ∂z T ∂z T ∂Z Enhanced Path via Skip Connection .(8)
A comprehensive description can be found in Algorithm 1.
Notes on Markovian Property: z t-1 depends on z t and z T , but not on any earlier states such as z t+1 , z t+2 , and so on. Therefore, the probability distribution for z t-1 only depends on z t and the fixed initial state z T , which is constant throughout the diffusion process. Thus, we have: p(z t-1 |z t , z t+1 , . . . , z T ) = p(z t-1 |z t , z T ). This confirms that LD3M remains Markovian.
Notes on Generalisability: Our gradient enhancement technique (Equation 7) applies to various diffusion model architectures beyond LDMs, suggesting broader potential for future work. In this study, we focused on LDMs primarily as a proof of concept, leveraging readily available pre-trained models and LDM's foundational role in latent-space diffusion.
this section cite: ['b30', 'b33', 'b18', 'b41', 'b3']

Section: Efficient Latent Code Initialization
Standard practice in dataset distillation initializes synthetic data using real images from the target classes [21]. GAN-based methods like GLaD face a significant challenge here, requiring complex and costly GAN inversion techniques to find latent codes Z that reconstruct target real images x [36].
LD3M benefits immensely from the autoencoder structure inherent in LDMs. We initialize the learnable latent codes Z init simply by encoding a small, randomly selected set of real images X s using the pre-trained image encoder. Similarly, the initial conditioning codes c init are obtained using the pre-trained class embedding network.
this section cite: ['b20', 'b35']

Section: Memory Efficiency via Gradient Checkpointing
Optimizing through the T steps of the reverse diffusion process, even with our modification, can be memory-intensive. Following GLaD [5], we employ gradient checkpointing [6] to manage VRAM usage. The procedure involves:
1. Perform the forward pass S = D(p θ (z 0 |z T , c)) without storing intermediate activations.
2. Calculate the distillation loss L(S, T ) and the gradient with respect to the output, ∂L/∂S. 3. To compute the gradient ∂L/∂Z (and ∂L/∂c), recompute the necessary segments of the forward pass through the diffusion process and decoder D, storing only the activations needed for the immediate backward pass segment.
This avoids storing the full computation graph, which GLaD also exploits for a single generator pass.
this section cite: ['b4', 'b5']

Section: Experiments
We evaluate LD3M against relevant baselines, primarily the state-of-the-art latent generative prior method GLaD [5], following its experimental setup for fair comparison. We conduct extensive experiments on 10 diverse 10-class subsets of ImageNet-1k [8] at 128 × 128 (IPC=1, IPC=10) and 256 × 256 (IPC=1) resolutions, as well as CIFAR-10. Key implementation details are in §5.1; full hyperparameters and setup details are in the supplementary material.
this section cite: ['b4', 'b7']

Section: Setup Details
Datasets & Evaluation: We use ImageNet subsets (ImNet-A to E, ImNette, ImWoof, Birds, Fruits, Cats) from [5,20,4] and CIFAR-10. Following standard protocol, we distill datasets using DC [42], DM [41], or MTT [4] and evaluate by training unseen architectures (AlexNet [22], VGG-11 [33], ResNet-18 [17], ViT [11]) from scratch on the distilled set, reporting mean test accuracy (± std. dev.) over 5 runs.  Table 4: Cross-architecture performance (%) with 10 IPC on ImageNet A-E (128 × 128). LD3M (bold, blue) consistently outperforms Pixel Space and GLaD with, for instance, an improvement of +2.52% and +3.46% with DC and DM, respectively. Distil. Space Alg. All ImNet-A ImNet-B ImNet-C ImNet-D ImNet-E DC 42.3±3.5 52.3±0.7 45.1±8.3 40.1±7.6 36.1±0.4 38.1±0.4 pixel space DM 44.4±0.5 52.6±0.4 50.6±0.5 47.5±0.7 35.4±0.4 36.0±0.5 DC 45.9±1.0 53.1±1.4 50.1±0.6 48.9±1.1 38.9±1.0 38.4±0.7 GLaD DM 45.8±0.6 52.8±1.0 51.3±0.6 49.7±0.4 36.4±0.4 38.6±0.7 DC 47.1±1.2 55.2±1.0 51.8±1.4 49.9±1.3 39.5±1.0 39.0±1.3 LD3M DM 47.3±2.1 57.0±1.3 52.3±1.1 48.2±4.9 39.5±1.5 39.4±1.8
Models: We use ConvNet-5/ConvNet-6 [15] for distillation. As in GLaD [5], we use AlexNet [22], VGG-11 [33], ResNet-18 [17], and a Vision Transformer [11] for evaluating the distilled dataset quality for unseen architectures. For LD3M, we use the public ImageNet pre-trained LDM [28] with its 2× compression autoencoder, without fine-tuning. Default diffusion steps T = 10 (128 2 ) or T = 20 (256 2 ) are used unless specified. For GLaD comparison, the ImageNet pre-trained StyleGAN-XL [30] is used.
this section cite: ['b4', 'b19', 'b3', 'b41', 'b40', 'b3', 'b21', 'b32', 'b16', 'b10', 'b14', 'b4', 'b21', 'b32', 'b16', 'b10', 'b27', 'b29']

Section: Results

this section cite: []

Section: Main Results: LD3M Outperforms State-of-the-Art.
We first establish LD3M's effectiveness against the primary baseline GLaD and other state-of-the-art methods on CIFAR-10 (Table 2). At IPC=1, LD3M matches or exceeds GLaD using DC/DM distillation. The sampling-based D4M [34] delivered ≈10% accuracy for all tested models. At IPC=50, LD3M surpasses reported results from D4M and decoupled SRe2L [40] with MTT (though optimization and IPC differ, see caption). We subsequently focus our main analysis on GLaD [5] as the leading state-of-the-art baseline for latent generative prior gradient-matching distillation.
Moving to the more challenging ImageNet subsets, LD3M consistently outperforms GLaD across resolutions and IPC values. For IPC=1 at 128 × 128 (Table 3), LD3M achieves the best overall ). This advantage persists at IPC=10 (Table 4), where LD3M improves average accuracy over GLaD by +2.5% (DC) and +3.5% (DM), reaching over 47% overall. The performance gains extend to 256 × 256 resolution (Table 5), where LD3M surpasses GLaD by +6% on average, even when comparing differently pre-trained generators (ImageNet, FFHQ, random), highlighting the robustness of leveraging the LDM prior via LD3M.
this section cite: ['b33', 'b39', 'b4']

Section: Ablation Studies: Why LD3M Works.
The diffusion process itself is crucial for LD3M's improved performance over autoencoder-only approaches. Table 6 provides empirical evidence: removing the diffusion steps ("w/o diffusion", akin to Duan et al. [12]) results in performance comparable to GLaD (35.3% avg.), which is significantly lower than full LD3M utilizing the diffusion process (36.5% avg.). This +1.2 percentage points highlight that simply using the LDM's autoencoder is insufficient; successfully optimizing through the reverse process is necessary to unlock the performance benefits observed, validating our core approach over simpler AE-only methods. Naturally, we can expect similar limitations with few-or one-step diffusion models in the context of dataset distillation.
Furthermore, both learnable latents and our proposed gradient enhancement drive LD3M's effectiveness. Table 7 dissects these contributions: optimizing only conditioning c yields poor results (15.8% avg.), but making the latent Z learnable provides a substantial boost (22.3% avg.), confirming the necessity of latent optimization motivated in §4.1. Critically, however, this alone does not surpass GLaD; only by subsequently adding our enhanced gradient flow (Equation 7) does LD3M achieve its SOTA performance (28.1% avg., vs. 26.6% for GLaD). This clearly demonstrates that while learnable latents offer vital degrees of freedom, they cannot be effectively optimized through standard diffusion backpropagation; the enhanced gradient flow enabled by our residual connection (Equation 7) is the key component that makes end-to-end optimization through diffusion truly effective for this task.
Robustness and Practical Considerations. LD3M exhibits robustness in various aspects. For instance, standard initialization using real images encoded via the LDM's encoder significantly benefits DC and DM performance compared to Gaussian noise initialization (Table 8), while being vastly simpler than GLaD's GAN inversion ( §4.2). Visually (Figure 2 and Figure 3), LD3M generates abstract but class-informative images that appear more consistent across different LDM pre-training datasets (ImageNet, FFHQ, Random) compared to GLaD. Our analysis of diffusion steps T (Figure 4) reveals an optimal performance/runtime trade-off around T = 10 -40.  LD3M also offers computational flexibility. Our analysis indicates an optimal balance between performance and distillation time occurs with around T = 10 -40 diffusion steps (Figure 4). Using T = 20, LD3M requires slightly less peak GPU memory on an A100-40GB compared to GLaD (29.4GB vs 31.2GB) and completes the distillation process faster (574 min vs 693 min). This computational efficiency, coupled with the inherent flexibility to adjust the number of diffusion steps (T ) based on available resources, enhances LD3M's practical appeal.
this section cite: ['b11']

Section: Conclusion
We addressed the critical challenge preventing end-to-end dataset distillation through powerful diffusion models: vanishing gradients across the long denoising chain. We introduced LD3M, which unlocks optimization through diffusion priors via a simple yet effective modification: injecting linearly decaying residual connections from the initial noisy state into each reverse step. This novel approach enhances gradient flow sufficiently to effectively learn both latent codes (Z) and conditioning (c) without altering pre-trained model weights. This direct optimization contrasts sharply with previous diffusion distillation methods that circumvented the gradient challenge by relying solely on sampling fixed representations [34,16].
this section cite: ['b33', 'b15']

Section: References
Ref_id:b0 Title: Multidiffusion: Fusing diffusion paths for controlled image generation Year: (2023)
Ref_id:b1 Title: Extreme classification (dagstuhl seminar 18291) Year: (2019)
Ref_id:b2 Title: Neural photo editing with introspective adversarial networks Year: (2017)
Ref_id:b3 Title: Dataset distillation by matching training trajectories Year: (2022)
Ref_id:b4 Title: Generalizing dataset distillation via deep generative prior Year: (2023)
Ref_id:b5 Title: Training deep nets with sublinear memory cost Year: (2016)
Ref_id:b6 Title: Provable and efficient dataset distillation for kernel ridge regression Year: (2024)
Ref_id:b7 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b8 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b9 Title: Privacy for free: How does dataset condensation help privacy? Year: (2022)
Ref_id:b10 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b11 Title: Dataset distillation in latent space Year: (2023)
Ref_id:b12 Title: Spotdiffusion: A fast approach for seamless panorama generation over time Year: (2024)
Ref_id:b13 Title: Predictability and surprise in large generative models Year: (2022)
Ref_id:b14 Title: Dynamic few-shot visual learning without forgetting Year: (2018)
Ref_id:b15 Title: Efficient dataset distillation via minimax diffusion Year: (2023)
Ref_id:b16 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b17 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b18 Title: The vanishing gradient problem during learning recurrent neural nets and problem solutions Year: (1998)
Ref_id:b19 Title: A smaller subset of 10 easily classified classes from imagenet, and a little more french Year: (2019)
Ref_id:b20 Title: Dataset condensation via efficient synthetic-data parameterization Year: (2022)
Ref_id:b21 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b22 Title: Discovering failure modes of text-guided diffusion models via adversarial search Year: (2023)
Ref_id:b23 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b24 Title: Less is more: Proxy datasets in nas approaches Year: (2022)
Ref_id:b25 Title: Diffusion models, image super-resolution and everything: A survey Year: (2024)
Ref_id:b26 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b27 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b28 Title: Image super-resolution via iterative refinement Year: (2022)
Ref_id:b29 Title: Stylegan-xl: Scaling stylegan to large diverse datasets Year: (2022)
Ref_id:b30 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b31 Title: Generalized large-scale data condensation via various backbone and statistical matching Year: (2024)
Ref_id:b32 Title: Very deep convolutional networks for large-scale image recognition Year: (2014)
Ref_id:b33 Title: Dataset distillation via disentangled diffusion model Year: (2024)
Ref_id:b34 Title:  Year: (2018)
Ref_id:b35 Title: Gan inversion: A survey Year: (2022)
Ref_id:b36 Title: Towards adversarially robust dataset distillation by curvature regularization Year: (2025)
Ref_id:b37 Title: Diffusion models: A comprehensive survey of methods and applications Year: (2023)
Ref_id:b38 Title: Dataset distillation in large data era Year: (2023)
Ref_id:b39 Title: Squeeze, recover and relabel: Dataset condensation at imagenet scale from a new perspective Year: (2023)
Ref_id:b40 Title: Dataset condensation with distribution matching Year: (2023)
Ref_id:b41 Title: Dataset condensation with gradient matching Year: (2020)
Ref_id:b42 Title: Hierarchical features matter: A deep exploration of gan priors for improved dataset distillation Year: (2024)
Ref_id:b43 Title: Beard: Benchmarking the adversarial robustness for dataset distillation Year: (2024)
Ref_id:b44 Title: Images distilled by MTT in LD3M for IPC=1 Year: ()
