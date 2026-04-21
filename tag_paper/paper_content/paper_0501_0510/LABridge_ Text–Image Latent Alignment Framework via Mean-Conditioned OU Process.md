Title: LABridge: Text-Image Latent Alignment Framework via Mean-Conditioned OU Process
Abstract: Diffusion models have emerged as state-of-the-art in image synthesis.However, it often suffer from semantic instability and slow iterative denoising. We introduce Latent Alignment Framework (LABridge), a novel Text-Image Latent Alignment Framework via an Ornstein-Uhlenbeck (OU) Process, which explicitly preserves and aligns textual and visual semantics in an aligned latent space. LABridge employs a Text-Image Alignment Encoder (TIAE) to encode text prompts into structured priors that are directly aligned with image latents. Instead of a homogeneous Gaussian, Mean-Conditioned OU process smoothly interpolates between these text-conditioned priors and image latents, improving stability and reducing the number of denoising steps. Extensive experiments on standard text-to-image benchmarks show that LABridge achieves better text-image alignment metric and competitive FID scores compared to leading diffusion baselines. By unifying text and image representations through principled latent alignment, LABridge paves the way for more efficient, semantically consistent, and high-fidelity text to image generation.

Section: Introduction
Diffusion models [Sohl-Dickstein et al., 2015, Song and Ermon, 2019, Ho et al., 2020, Song et al., 2021] represent a significant advancement in generative modeling, demonstrating state-of-the-art performance in diverse tasks such as text generation [Li et al., 2022, Wu et al., 2023], high-fidelity image synthesis [Rombach et al., 2022, Ramesh et al., 2022, Ho et al., 2022, Shao et al., 2023, Lin et al., 2025], image restoration [Blattmann et al., 2023, Brooks et al., 2024], and 3D content creation [Liu et al., 2023, Evans et al., 2024]. These models typically operate by defining a forward diffusion process that gradually adds noise to data, transforming it into a simple prior distribution (often Gaussian), and then learning a reverse process to generate data by iteratively denoising samples drawn from this prior. The mathematical foundation often relies on stochastic differential equations (SDEs) [Song et al., 2021] or discrete-time Markov chains, enabling powerful sampling and manipulation strategies like accelerated generation [Mei et al., 2024], timestep analysis, and model distillation [Shao et al., 2025, Xie et al., 2024, Nguyen et al., 2024, Kang et al., 2024].
this section cite: ['b40', 'b41', 'b15', 'b42', 'b23', 'b36', 'b34', 'b16', 'b38', 'b25', 'b2', 'b3', 'b27', 'b10', 'b42', 'b30', 'b39', 'b31', 'b19']

Section: Existing Challenges
Despite their success, standard diffusion models face limitations, particularly in text-to-image generation, stemming largely from their reliance on a fixed, often unstructured prior:
1. Instability and Ambiguity from Homogeneous Priors: Conventional methods map the entire diverse data distribution q data (x) to a single, fixed prior, typically N (0, I). This forces distinct semantic concepts (e.g., "cat," "dog," "landscape") onto the same simple latent structure. This collapsing of priors can lead to instability, as the reverse process must disentangle these varied semantics from a homogeneous starting point. Furthermore, it can create ambiguity in score estimation (∇ xt log p(x t )), as paths originating from different initial data points might overlap significantly in the latent space near t = T , making the learned score an average that lacks precision for any specific semantic direction. We provide analysis in Appendix E.3 2. Inefficient Sampling: As shown in Fig. 1 (a), the diffusion sampling process follows a curved path. Without strong guidance, especially in early steps, the process can be slow, requiring many iterations (NFE -Number of Function Evaluations) to converge to a high-fidelity image that accurately reflects the conditioning. We provide analysis in Appendix D.3 3. Weak Text-Vision Alignment: While conditioning mechanisms inject textual information, the fundamental diffusion process still operates between the image manifold and Gaussian prior. This indirect connection can limit the precise alignment between the generated image and complex or nuanced text prompts, especially for out-of-distribution concepts (analysis in Appendix D.2).
Our Innovations: LABridge To overcome these challenges, we introduce LABridge, a framework designed to enhance text-vision alignment and accelerate sampling in diffusion models. LABridge leverages two core ideas: a dedicated encoder TIAE to create structured, text-conditioned priors, and an OU diffusion process to connect image latents directly to these priors.
1. Text Encoder for Structured Priors: We employ a encoder to process text prompts (y) and generate corresponding latent representations µ T (y). The latent µ T (y) acts as a structured, textspecific prior mean in the aligned latent space of an image autoencoder. The TIAE ensures that semantically similar texts map to nearby priors, preserving semantic structure. 2. OU Diffusion Process for Alignment and Stability: We adopt OU process explicitly models the stochastic path between the image latent x 0 (obtained from a VAE encoder) and a distribution centered around the text-specific prior µ T (y), i.e., N (µ T (y), σ 2 T I). The OU process, known for its mean-reverting property, naturally pulls the state towards the target mean µ T (y), inherently promoting stability and alignment.
LABridge offers a new perspective: it frames text-to-image generation as learning a stochastic process between the image latent manifold and a manifold of text-conditioned priors. This explicit alignment via the process mechanism enhances semantic consistency and, by providing a better starting point and direction, significantly speeds up the sampling process.
In summary, our contributions are threefold:
• We propose LABridge, a novel framework utilizing an encoder TIAE to generate structured text-conditioned priors µ T (y) and an OU diffusion process to align them with image latents x 0 .
• We demonstrate that the OU process mechanism, combined with structured priors, improves text-vision alignment and inherently accelerates sampling by providing a more certain trajectory, supported by theoretical analysis.
• We validate LABridge experimentally, showing significant improvements in sampling and textimage consistency metrics while maintaining competitive image fidelity compared to baselines.
this section cite: []

Section: Preliminaries
In this section, we introduce the essential background concepts. Additional technical details, and extended definitions are deferred to Appendix A.
this section cite: []

Section: Diffusion Processes and Score Matching
Diffusion Process. Let x ∈ R d follow the data distribution q data (x). A forward diffusion process defines a sequence of latent variables {x t } t∈[0,T ] starting from x 0 ∼ q data (x) and evolving towards a simple prior distribution p T (x T ) = N (0, I) as t goes from 0 to T . This evolution is often described by a stochastic differential equation (SDE) [Song et al., 2021]:
dx t = f (x t , t) dt + g(t) dw t ,(1)
where f (•, t) is the drift function, g(t) is the diffusion coefficient, and w t is a standard Wiener process. Generating new data involves reversing this process. The corresponding reverse-time SDE is given by [Anderson, 1982]:
dx t = f (x t , t) -g(t) 2 ∇ xt log p t (x t ) dt + g(t) d wt ,(2)
where wt is a Wiener process running backward in time, and p t (x t ) is the marginal probability density of x t . The crucial term is the score function ∇ xt log p t (x t ).
Score Matching. In practice, the true score ∇ xt log p t (x t ) is unknown and is approximated by a time-dependent neural network s θ (x t , t), often conditioned on additional information y (like text embeddings), denoted s θ (x t , t, y). This network is trained by minimizing a score matching objective [Hyvärinen andDayan, 2005, Vincent, 2011]. For many diffusion processes (like VP and VE), the conditional score ∇ xt log p t (x t | x 0 ) is tractable. A common training objective is:
L SM (θ) = E t∼U (0,T ),x0∼qdata,ϵ∼N (0,I) λ(t) s θ (α t x 0 + σ t ϵ, t, y) -∇ xt log p t (x t | x 0 ) 2 , (3
) where p t (x t | x 0 ) = N (x t ; α t x 0 , σ 2 t I) defines the transition kernel, and λ(t) is a weighting function. For this Gaussian kernel, ∇ xt log p t (x t | x 0 ) = -(x t -α t x 0 )/σ 2 t = -ϵ/σ t .
This leads to the widely used noise prediction objective:
L denoise (θ) = E t,x0,ϵ λ ′ (t) ϵ θ (α t x 0 + σ t ϵ, t, y) -ϵ 2 ,(4)
where ϵ θ is the network predicting the noise ϵ, related to the score network by s θ (x t , t, y) = -ϵ θ (x t , t, y)/σ t .
this section cite: ['b42', 'b0', 'b18']

Section: Diffusion Bridge Models
While standard diffusion maps data to a fixed prior, a diffusion bridge connects two specified endpoint distributions, p 0 (x 0 ) and p T (x T ), which can both be complex. This is particularly relevant for tasks involving paired data (x 0 , x T ) ∼ q data (x 0 , x T ), such as image translation or, in our case, aligning image latents x 0 with text-derived priors µ T .
this section cite: []

Section: Stochastic Bridges via h-Transform.
Given a forward SDE Eq. ( 1), Doob's h-transform provides a way to condition the process to start at x 0 at t = 0 and end exactly at x T = y at t = T . The resulting bridge SDE is:
dx t = f (x t , t) + g(t) 2 ∇ xt log p T |t (y | x t ) dt + g(t) dw t ,(5)
where
p T |t (x T | x t )
is the transition probability density of the original SDE from time t to T . The term h(x t , t, y, T ) = ∇ xt log p T |t (y | x t ) is the "guidance" term ensuring the endpoint constraint.
For linear SDEs with Gaussian transitions (like VP, VE, OU), this term is often tractable.
Denoising Diffusion Bridge Models [Zhou et al., 2023]. Instead of exact endpoints, we often want to sample from a conditional distribution q(x 0 | x T ) given paired data (x 0 , x T ) ∼ q data . This can be achieved by reversing a bridge process designed such that its marginals q(x 0 , x T ) approximate q data (x 0 , x T ). Zhou et al. [2023] show that the reverse SDE for sampling x t given x T = y is:
dx t = f (x t , t) -g 2 (t) ∇ xt log q t (x t | x T = y) -∇ xt log p T |t (y | x t ) dt + g(t) d wt ,(6)
where ∇ xt log q t (x t | x T = y) is the score of the conditional bridge distribution.
this section cite: ['b49', 'b49']

Section: Mean-Conditioned Ornstein-Uhlenbeck Process
Ornstein-Uhlenbeck (OU) Process. The OU process is a mean-reverting stochastic process often used to model systems returning to equilibrium. Its SDE is:
dx t = θ (µ -x t ) dt + σ dw t ,(7)
where θ > 0 is the rate of mean reversion, µ is the equilibrium mean, and σ is the volatility. The drift term θ(µ -x t ) pulls the state x t towards µ.
this section cite: []

Section: Ornstein-Uhlenbeck Bridge (OUB).
An OU process is a conditioned OU process that starts at x 0 at t = 0 and ends at x T = y at t = T . Its SDE can be derived using Doob's h-transform. For the standard OU process with σ = 1 starting from x 0 at t = 0, the transition density to time T is Gaussian:
p T |0 (x T | x 0 ) = N (x T ; µ + (x 0 -µ)e -θT , σ 2 2θ (1 -e -2θT )I).
The OUB derived from this inherits the mean-reverting property but ensures the endpoint constraint. In our work, we use a specific form of OU process connecting x 0 to a distribution around µ T (y), defined by the transition:
q(x t | x 0 , µ T ) = N x t ; x 0 e -θt + µ T (1 -e -θt ), σ 2 2θ (1 -e -2θt )I .(8)
This corresponds to an OU process SDE dx t = θ(µ T -x t )dt + σdw t . This structure is key to LABridge. The parameter θ controls the strength of reversion towards the text-conditioned prior mean µ T . OUB processes can encompass VP and VE under specific parameter choices.
3 Proposed Method: LABridge
this section cite: []

Section: Motivation: Structured Priors and Directed Diffusion
A primary challenge lies in directly using text as latent representations. Text data is fundamentally different from typical latent variables. It is inherently infinite and unstructured; the space of possible sentences and meanings is vast and does not easily map to predefined, discrete categories or a simple, low-dimensional manifold. Converting raw text directly into a structured latent code suitable for generative models is non-trivial.
Instead, we must first represent text in a suitable format, typically through powerful pretrained embedding models (like CLIP, T5) that capture semantic meaning in high-dimensional vectors (E y ). However, simply using these embeddings as conditioning signals for a standard diffusion process (mapping image I to noise ϵ) still relies on the model implicitly learning the complex relationship between the text embedding space and the image manifold during the denoising process initiated from a generic prior.
This motivates the need to represent text in a continuous latent space that is explicitly aligned with the latent space of images. Rather than mapping images to generic noise, we propose mapping images x 0 to text-conditioned priors µ T (y) that live in the similar latent space. This requires: 1. A mechanism to map text embeddings E y to target latent priors µ T (y). 2. Ensuring these priors µ T (y) are semantically consistent (similar texts map to nearby priors) and aligned with corresponding image latents. 3. A generative process that efficiently connects these aligned endpoint distributions. The corresponding text y is embedded to E y by E Emb and then mapped to a target prior mean µ T (y) by the Text-Vision Alignment Encoder E TE (TIAE). An OU diffusion process learns the stochastic path between x 0 and the distribution N (µ T (y), σ 2 T I). During inference, sampling starts from x T and follows the learned reverse process dynamics towards x 0 , which is then decoded to an image Î by D VAE .
this section cite: []

Section: LABridge Framework Overview
As depicted in Fig. 3, LABridge integrates three key components: 1. Pretrained Image Autoencoder: We utilize a frozen Variational Autoencoder (VAE) with encoder E VAE and decoder D VAE . The encoder maps an input image I to a latent representation x 0 = E VAE (I) in R d , effectively moving the diffusion process to a compressed latent space. The decoder reconstructs the image Î = D VAE (x 0 ). 2. Text-Vision Alignment Encoder (TIAE): This novel component, denoted E TE , is responsible for bridging the semantic gap between text and vision. It takes text embeddings E y (obtained from a frozen pretrained text model E Emb , e.g., CLIP [Radford et al., 2021]) as input and outputs a target prior mean µ T (y) = E TE (E y ) ∈ R d . The TIAE is specifically trained to ensure that µ T (y) is both semantically meaningful (reflecting the content of y) and aligned with the corresponding image latents x 0 in the VAE's latent space. 3. Mean-Conditioned OU Diffusion Process: Instead of a standard diffusion process mapping to
N (0, I), we employ an OU diffusion process [Zhou et al., 2023] to model the stochastic transition between the image latent x 0 and the text-conditioned prior distribution N (µ T (y), σ 2 T I). The process is parameterized by a score network s θ (x t , t, y) or, equivalently, a noise prediction network ϵ θ (x t , t, y). The OU process inherently incorporates mean reversion towards µ T (y), promoting stability and directed sampling.
this section cite: ['b49']

Section: TIAE Architecture and Training Objective
Architecture. To ensure seamless integration and potentially leverage existing performant architectures, we adopt a structure similar to modern diffusion models for the TIAE E TE , specifically using blocks inspired by DiT [Peebles and Xie, 2023] which handle conditional inputs effectively. It takes the text embedding E y as input and outputs the prior mean µ T (y).
Training Objective. The TIAE is trained to produce priors µ T (y) that are: (a) aligned with corresponding image latents x 0 , and (b) preserve the semantic structure of the text embeddings E y . We use a composite loss:
(a) Latent Alignment Loss (L align ): Encourages the TIAE output µ T (y) to be close to the VAE latent x 0 for corresponding image-text pairs (I, y).
L align = E (x0,y)∼qdata ∥µ T (y) -x 0 ∥ 2 2 ,(9)
where x 0 = E VAE (I) and µ T (y) = E TE (E y ). This loss is visualized in Fig. 4.
(b) Semantic Consistency Loss (L sem ): Ensures that the distances between latent priors reflect the distances between text embeddings (using cosine similarity).
L sem = E yi,yj sim cos (µ T (y i ), µ T (y j )) -sim cos (E yi , E yj ) 2 . (10
)
This encourages the structure of the text embedding space to be preserved in the prior space, as illustrated conceptually in Fig. 5. Generated Latent Distribution TIAE Mean Std dev Text Embedding Model pretrained & frozen Text Text Embedding All Latent Distribution Text1 Text2 Text3 Text4 TIAE Text1 Embedding Text Embedding Model pretrained & frozen Text1 Text2 Embedding Text2 Text3 Embedding Text3 Text4 Embedding Text4 ... ... Alignment Embedding Similarity Distribution Divergence (a) Latent Representation (b) Similarity Alignment Figure 5: Illustration of the latent space alignment and distribution analysis for TIAE. Text embeddings (left) are encoded into latent distributions centered at µ T (center). The semantic consistency loss L sem aims to ensure that similar text inputs (e.g., Text1, Text2) produce close latent means/distributions, preserving semantic proximity, as shown by the embedding similarity matrix (right) compared to the latent distribution similarity.
this section cite: ['b32']

Section: TIAE

this section cite: []

Section: Text Embedding Model
(c) Reconstruction Loss (L rec ): To ensure µ T (y) can be decoded into meaningful images, we can add a reconstruction term comparing the decoded prior to the original image.
L rec = E (x0,y)∼qdata ∥D VAE (µ T (y)) -I∥ p p ,(11)
The total TIAE loss is L TIAE = w a L align + w s L sem + w r L rec , with weights w a , w s , w r .
this section cite: []

Section: Integration with OU Diffusion Process
Once the TIAE E TE is trained and frozen, we train the diffusion model component. LABridge employs an OU diffusion process defined by the forward stochastic differential equation (SDE):
dx t = θ(µ T (y) -x t )dt + σdw t , t ∈ [0, T ],(12)
where µ T (y) = E TE (E Emb (y)) is the target mean obtained from the frozen TIAE for a given text prompt y. This SDE defines a process that starts near x 0 at t = 0 and is drawn towards µ T (y) as t → ∞. The transition kernel q(x t |x 0 , y) corresponding to this SDE, assuming x t depends on x 0 and the target mean µ T (y).
The goal is to learn the reverse process to sample x 0 ∼ q(x 0 |y) starting from the prior p(x T |y) = N (x T ; µ T (y), σ 2 T I). The reverse process is governed by the score function ∇ xt log q t (x t |y). We follow the standard practice in diffusion models and train a neural network ϵ θ (x t , t, y) to predict the noise ϵ that generated x t from x 0 and µ T (y) via Eq. 8. The noise prediction objective for the OU process is:
L Bridge = E t∼U (0,T ),(x0,y)∼qdata,ϵ∼N (0,I) w ′ (t) ∥ϵ θ (x t , t, y) -ϵ∥ 2 2 ,(13)
where w ′ (t) is a time-dependent weighting function. The reverse probability flow ODE used for sampling is:
dx t = [θ(µ T (y) -x t ) + σ 2 2σ t ϵ θ (x t , t, y)]dt.(14)
this section cite: []

Section: Training and Inference Algorithms
The overall training procedure for LABridge consists of two sequential stages (details of training procedure are provided in Algo. 1 in Appendix B. ):
• Stage 1: TIAE Training. Train the Text-Vision Alignment Encoder E TE using the composite loss L TIAE on paired image-text data, with the VAE and text embedder frozen.
• Stage 2: OU Process Training. Freeze the trained E TE . Train the noise prediction network ϵ θ for the OU process using the denoising objective L Bridge (Eq. 13).
The overall inference procedure for LABridge mainly based on Eq. 14 (details of inference procedure are provided in Algo. 2 in Appendix B.).
this section cite: []

Section: Theoretical Guarantees
We provide theoretical justification for the advantages of LABridge, highlighting improvements in text-vision alignment, sampling efficiency and stability standard diffusion models using fixed priors. Detailed statements and proofs are provided in Sec. D and E.
Theorem 4.1 (Enhanced Text-Vision Alignment). The TIAE training objective, particularly L align and L sem , explicitly optimizes the prior mean µ T (y) to be (a) close to the corresponding image latent mean E[x 0 |y] and (b) preserve the semantic structure of the text embeddings E y . The OU process formulation reinforces this alignment during diffusion training and generation. (Ref: Thm. D.3, Prop. D.4 in the appendix). Theorem 4.2 (Sampling Acceleration via Informed Initialization and Dynamics). LABridge accelerates sampling due to two factors: (i) Reduced Initial Error: Starting the reverse process from x T ∼ N (µ T (y), σ 2 T I) provides an initial state closer (in expectation) to the target conditional mean E[x 0 |y] compared to starting from N (0, σ 2 T I). (Ref: Thm. D.5 in appendix). (ii) Directed Drift: The OU reverse dynamics (Eq. 14) include an explicit mean reversion term θ(µ T (y) -x t ) which provides additional drift towards the text-aligned prior mean µ T (y), supplementing the learned score/noise term and offering stronger guidance than standard diffusion drifts . (Ref: Thm. D.6 in appendix). Theorem 4.
this section cite: []

Section: (Improved Sampling Stability).
The inherent mean-reverting property of the OU process drift term θ(µ T (y) -x t ) enhances the stability of the reverse sampling process, making it less prone to divergence compared to processes with zero or origin-centric drift, especially when score estimates may be imperfect. (Ref: Thm. D.8 in appendix). Theorem 4.4 (Tighter Evidence Lower Bound). Using the text-conditioned prior p(x T |y) = N (µ T (y), σ 2 T I) results in a smaller expected KL divergence between the forward process endpoint distribution q(x T |x 0 , y) and the prior, compared to using a fixed prior p(x T ) = N (0, σ 2 T I), provided E[x 0 |y] varies significantly with y and µ T (y) approximates it well. This leads to a tighter ELBO. (Ref: Thm. D.9).
this section cite: []

Section: Experiment
We conducted a series of experiments to verify the effectiveness of LABridge under various settings.
Experiment Setup All code was performed on 8 A100 GPUs machine. For the first part, we employ DiT-XL/2 model to learn from scratch on benchmark dataset. We adopt NV-Embed-v2foot_0 as pretrained
Method COCO-10K MJHQ-30K Text-Alignment FID ↓ CLIP ↑ FID ↓ CLIP ↑ GenEval ↑ DPG ↑ Stable Diffusion V1.5 Comparision SD15-Base [Rombach et al., 2022] 15.81±.04 28.03±.08 13.54±.03 28.40±.01 0.48±.04 70.64±.01 SD15 + CF Liu et al. [2024] 14.83±.05 28.52±.07 12.62±.02 28.90±.03 0.54±.04 71.35±.01 SD15 + LABridge (VP, UNet) 13.82±.04 29.01±.06 11.63±.03 29.42±.02 0.57±.04 72.42±.02 Stable Diffusion XL Comparision SDXL-Base [Rombach et al., 2022] 11.68±.04 28.83±.04 10.55±.01 29.63±.01 0.56±.03 75.52±.02 SDXL + CF (Liu et al. [2024]) 12.69±.04 29.33±.03 9.59±.02 30.15±.03 0.62±.03 76.53±.02 SDXL + LABridge(VP, UNet) 12.72±.02 29.82±.01 8.55±.03 30.63±.01 0.65±.03 77.21±.01 Stable Diffusion 3.5 Medium DiT Comparision SD3.5-M [Esser et al., 2024] 9.88±.03 29.91±.04 8.45±.01 30.71±.03 0.62±.03 83.31±.02 SD3.5-M + CF Liu et al. [2024] 8.89±.04 30.43±.02 7.28±.02 31.23±.03 0.65±.03 84.31±.02 SD3.5-M + LABridge(VP, DiT) 7.86±.05 30.92±.01 7.44±.03 31.72±.02 0.67±.03 85.31±.01 Stable Diffusion 3.5 Large DiT Comparision SD3.5-L [Esser et al., 2024] 7.33±.03 30.88±.03 5.84±.02 31.41±.02 0.66±.03 84.52±.02 SD3.5-L + CF Liu et al. [2024] 6.33±.04 31.36±.04 4.84±.03 31.89±.02 0.68±.03 85.52±.02 SD3.5-L + LABridge(VP, DiT) 5.34±.03 31.87±.02 3.82±.02 32.39±.01 0.69±.03 85.28±.01 PixArt DiT Comparision PixArt-α [Chen et al., 2023] 11.24 ±.02 29.52±.03 9.65±.02 30.01±.04 0.49±.03 75.42±.04 PixArt-α + CF Liu et al. [2024] 10.23±.03 30.02±.04 8.62±.03 30.53±.03 0.51±.03 76.65±.04 PixArt-α + LABridge (VP, DiT) 9.23±.02 30.51±.03 7.63±.04 31.01±.02 0.54±.03 77.76±.03
Table 2: Quantitative comparison of state-of-the-art models across various architectures and benchmarks for different metrics.
this section cite: []

Section: S DX L + C r o s s F l o w S DX L + T I A E ( UNe t ) S D3 . 5 -M ( 5 0 NF E S ) S D3 . 5 -M+ C r o s s F l o w S D3 . 5 -M + T I A E ( Di T )
A The superior performance on these benchmarks underscores the strong competitiveness of our method in improving alignment between text and generated images. Moreover, to verify the generalizability of TIAE, we tested TIAE in combination with various other community text-to-image plugins. The results are shown in Fig. 7, demonstrating the robustness.
this section cite: []

Section: Ablation Studies.
We conducted extensive ablation studies to evaluate the effectiveness of the different loss functions used in the LABridge training method, aiming to validate its correctness. The CLIP and FID metrics were tested on COCO-10k and MJHQ-30K, with the results summarized in Tab.
3. These results show that the training process combining all three loss functions consistently delivers the best performance across most backbones and benchmark datasets. In contrast, using a single loss or a combination of two losses yields slightly inferior results. This highlights the effectiveness of our proposed LABridge approach.
this section cite: []

Section: Conclusion
We presented LABridge, a novel text-to-image generation framework designed to overcome semantic instability and slow sampling inherent in diffusion models. By employing a TIAE to create structured, text-conditioned prior aligned with image latents, and utilizing an OU diffusion bridge to connect these representations, LABridge establishes explicit text-vision consistency. LABridge offers a robust pathway towards efficient, high-quality conditional image synthesis.
this section cite: []

Section: References
Ref_id:b0 Title: Reverse-time diffusion equation models Year: (1982)
Ref_id:b1 Title: All are worth words: A vit backbone for diffusion models Year: (2023)
Ref_id:b2 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b3 Title: Video generation models as world simulators Year: (2024)
Ref_id:b4 Title: Coyo-700m: Image-text pair dataset Year: (2022)
Ref_id:b5 Title: Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2023)
Ref_id:b6 Title: Causal diffusion transformers for generative modeling Year: (2024)
Ref_id:b7 Title: ImageNet: A largescale hierarchical image database Year: (2009)
Ref_id:b8 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b9 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b10 Title: Fast timing-conditioned latent audio diffusion Year: (2024)
Ref_id:b11 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2024)
Ref_id:b12 Title: Efficient diffusion training via min-snr weighting strategy Year: (2023)
Ref_id:b13 Title: Clipscore: A referencefree evaluation metric for image captioning Year: (2021)
Ref_id:b14 Title: GANs trained by a two time-scale update rule converge to a local Nash equilibrium Year: (2017)
Ref_id:b15 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b16 Title: Imagen video: High definition video generation with diffusion models Year: (2022)
Ref_id:b17 Title: Equip diffusion models with llm for enhanced semantic alignment Year: (2024)
Ref_id:b18 Title: Estimation of non-normalized statistical models by score matching Year: (2005)
Ref_id:b19 Title: Distilling diffusion models into conditional gans Year: (2024)
Ref_id:b20 Title: Improved precision and recall metric for assessing generative models Year: (2019)
Ref_id:b21 Title: Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation Year: (2024)
Ref_id:b22 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b23 Title: Diffusion-lm improves controllable text generation Year: (2022)
Ref_id:b24 Title: Sdxl-lightning: Progressive adversarial diffusion distillation Year: (2024)
Ref_id:b25 Title: Diffusion adversarial post-training for one-step video generation Year: (2025)
Ref_id:b26 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b27 Title: Audioldm: Text-to-audio generation with latent diffusion models Year: (2023)
Ref_id:b28 Title: Flowing from words to pixels: A framework for cross-modality evolution Year: (2024)
Ref_id:b29 Title: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b30 Title: Codi: Conditional diffusion distillation for higher-fidelity and faster image generation Year: (2024)
Ref_id:b31 Title: Swiftbrush: One-step text-to-image diffusion model with variational score distillation Year: (2024)
Ref_id:b32 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b33 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b34 Title: Hierarchical textconditional image generation with CLIP latents Year: (2022)
Ref_id:b35 Title: Hyper-sd: Trajectory segmented consistency model for efficient image synthesis Year: (2024)
Ref_id:b36 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b37 Title: Improved techniques for training gans Year: (2016)
Ref_id:b38 Title: Building bridge across the time: Disruption and restoration of murals in the wild Year: (2023)
Ref_id:b39 Title: Rayflow: Instanceaware diffusion acceleration via adaptive flow trajectories Year: (2025)
Ref_id:b40 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b41 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b42 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b43 Title: Relay diffusion: Unifying diffusion process across resolutions for image synthesis Year: (2023)
Ref_id:b44 Title: Givt: Generative infinite-vocabulary transformers Year: (2025)
Ref_id:b45 Title: A connection between score matching and denoising autoencoders Year: (2011)
Ref_id:b46 Title: Ar-diffusion: Auto-regressive diffusion model for text generation Year: (2023)
Ref_id:b47 Title: Em distillation for one-step diffusion models Year: (2024)
Ref_id:b48 Title: Fast training of diffusion models with masked transformers Year: (2023)
Ref_id:b49 Title: Denoising diffusion bridge models Year: (2023)
