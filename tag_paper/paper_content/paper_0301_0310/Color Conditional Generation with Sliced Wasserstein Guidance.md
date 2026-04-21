Title: Color Conditional Generation with Sliced Wasserstein Guidance
Abstract: We propose SW-Guidance, a training-free approach for image generation conditioned on the color distribution of a reference image. While it is possible to generate an image with fixed colors by first creating an image from a text prompt and then applying a color style transfer method, this approach often results in semantically meaningless colors in the generated image. Our method solves this problem by modifying the sampling process of a diffusion model to incorporate the differentiable Sliced 1-Wasserstein distance between the color distribution of the generated image and the reference palette. Our method outperforms state-ofthe-art techniques for color-conditional generation in terms of color similarity to the reference, producing images that not only match the reference colors but also maintain semantic coherence with the original text prompt. Our source code is available at https://github.com/alobashev/sw-guidance. Figure 1: Color-conditional generation by Sliced Wasserstein guidance achieves unprecedented match with a reference color palette without transferring other stylistic features.

Section: Introduction
To get a desired picture from text-to-image models we usually need a precise prompt and a bit of luck. However, natural language is not expressive enough to accurately describe colors, and even specific 39th Conference on Neural Information Processing Systems (NeurIPS 2025). terms such as "turquoise blue" yield varying tones. Moreover, prompt length constraints make full palette descriptions impractical. Using reference images for color styles addresses these limitations and establishes the color transfer problem, that is, applying a reference color style to a content image.
Color transfer is closely related to the artistic style transfer. Notably, artistic style is not linked to the depicted objects but is instead shared between patches of an image. This insight was utilized in the seminal work by Gatys et al. [1], where artistic style is defined as the distribution of activations in the VGG-19 network [2]. To match the artistic style between generated and reference images, Gatys et al. minimized the difference between the Gram matrices of activations from internal layers of VGG-19 (which is equivalent to matching the first two moments of the distributions of activations).
Strictly speaking, the style loss by Gatys et al. is not a proper distance in the space of probability distributions, as, for instance, Jensen-Shannon [3], Total variation and various Wasserstein distances [4]. Unfortunately, these metrics are hard to approximate in a differentiable fashion. To address the complications of Wasserstein metrics, a new family of metrics called Sliced Wasserstein (SW) distances was developed in 2012 [5,6]. First, Sliced Wasserstein distances are differentiable. Second, they can be efficiently estimated from samples. Importantly, for bounded distributions, convergence of the Sliced Wasserstein distance implies the convergence of all moments. However, in highdimensional spaces the sliced approach requires a large number of projections to accurately estimate the distance. To generalize the SW distance and enhance its performance in higher dimensions other its variants were proposed [7,8,9,10,11,12,13,14].
Following Gatys et al., various CNN-based color transfer methods were proposed, such as DPST [15], WCT [16], PhotoWCT [17], WCT2 [18], PhotoNAS [19], PhotoWCT2 [20], and DAST [21]. These algorithms can address the problem of color-conditional image generation, transferring reference colors to the image created by a text-to-image model.
Another way to achieve color conditioning is to control the generation process of a diffusion model [22,23,24]. This problem setting is broadly called the stylized image generation. The approaches for stylized generation could be categorized into three groups:
Modification of weights The first group includes additive corrections of a model's weights, which require fine-tuning for every new style of images: Textual Inversion [25], DreamBooth [26], and LoRA [27]. The introduction of ControlNet [28] and T2I-Adapter [29] in 2023 enabled adjustments of weights in a single pass of a hyper-network. ControlNets and adapters are trained on fairly large paired datasets and cover tasks such as pose, depth, and edge conditioning.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28']

Section: Modification of attention
The examples of attention-related algorithms are IP-Adapter [30], StyleAdapter [31], StyleDrop [32], StyleAligned [33], InstantStyle [34,35]. Training-free, they change attention output on each step and are effective for controlling structural and high-level features, such as painting style and composition, but do not target a color distribution separately.
this section cite: ['b29', 'b30', 'b31', 'b32', 'b33', 'b34']

Section: Modification of sampling
The third way to impose a condition is to add a new term to the denoising process. The first work of this kind was classifier guidance [22], which requires a specific classifier trained on noisy data samples 1 . Diffusion Posterior Sampling (DPS) [36] addresses the main weakness of classifier guidance by replacing a noisy classifier with a composition of a predicted noiseless image and a classifier trained on clean data (i.e., any pre-trained one). Universal Diffusion Guidance [37] and FreeDoM [38] generalize the DPS approach by replacing the MSE loss used by DPS with a general distance function. These ideas were further developed in RB-Modulation [39].
In current approaches to stylized image generation style and color conditioning are often entangled, making it challenging to control these aspects independently. Our goal is to propose a way to condition solely on color and independently control the palette and general style of an image.
this section cite: ['b21', 'b35', 'b36', 'b37', 'b38']

Section: Our Contributions
This work makes the following key contributions:
• For the first time, we incorporate the differentiable Sliced Wasserstein distance and its generalizations into the conditioning of a diffusion model
• We achieve state-of-the-art results in a problem of color-specific conditional generation, without transferring unwanted textures or other stylistic features (see Fig. 1).
this section cite: []

Section: Background

this section cite: []

Section: Conditioning Process in Diffusion Models
Diffusion models [40,41] are a class of generative models that learn to iteratively denoise a data distribution. To describe the conditioning process in diffusion models, we use Bayes' rule to express the posterior distribution in terms of the gradient of the log-likelihood and the unconditional score:
∇ xt log p(x t |y) = ∇ xt log p(y|x t ) + ∇ xt log p(x t ),(1)
where y represents the conditioning, and x t is the noisy sample at noise level t.
Diffusion Posterior Sampling (DPS) [36] introduced an approximation for the conditional likelihood based on a predicted noiseless sample,
x0 = E(x 0 |x t ), as p(y|x t ) ≈ p(y|x 0 (x t )).(2)
In the DPS approach, the authors considered the gradient of the log-likelihood as follows:
∇ xt log p(y|x 0 (x t )) = - 1 σ 2 ∇ xt ||y -A(x 0 (x t ))|| 2 , (3
)
where A is an operator, generally non-linear, that extracts the condition y from the predicted noiseless sample x0 , and σ is a positive hyperparameter. For example, A could extract the CLIP [42] embedding from x0 , and y could be a target prompt embedding.
Universal Diffusion Guidance [37] and FreeDoM [38] extend the DPS approximation by proposing a more general distance function D in the space of conditions Y . Specifically, for y ∈ Y , the gradient of the logarithm of the posterior distribution is given by:
∇ xt log p(y|x 0 (x t )) = - 1 σ 2 ∇ xt D (y, A(x 0 (x t ))) .(4)
This formulation is more flexible and lets y and A(x 0 (x t )) to be a more complicated objects than vectors in R d as long as we can define a differentiable distance function between them. In the next section, we will define the Sliced Wasserstein distance as a suitable distance D between two probability measures.
this section cite: ['b39', 'b40', 'b35', 'b41', 'b36', 'b37']

Section: Sliced Wasserstein distance
A classical formulation of color transfer problem is to align two probability distributions in the 3-dimensional RGB space. Specifically, the color distributions of a content image and a reference image can be represented as probability density functions, denoted by π 0 and π 1 respectively. The objective in guided diffusion models is to match the generated sample's probability density π 0 with the reference π 1 .
Wasserstein distances, rooted in optimal transport theory, appear to be natural for this task as they measure the cost of transporting one probability distribution to match another [4]. The Wasserstein distance of order p is
W p (π 0 , π 1 ) = inf π∈Π(π0,π1) X0×X1 ||x -y|| p dπ(x, y) 1/p ,(5)
Calculating W p (π 0 , π 1 ) for many samples can be computationally prohibitive, also a Wasserstein distance is hard to differentiate through, because its value is itself a result of an optimization procedure inf over all transport plans Π(π 0 , π 1 ), i.e. over all joint distributions with marginals π 0 and π 1 .
To alleviate this issue, the Sliced Wasserstein (SW) distance was introduced [5], offering a more computationally tractable alternative by reducing high-dimensional distributions to one-dimensional projections where the Wasserstein distance can be computed more straightforwardly. The Sliced p-Wasserstein distance is defined as [5,6]:
SW p (π 0 , π 1 ) = S d-1 W p p (P θ π 0 , P θ π 1 ) dθ 1/p ,(6)
where S d-1 is the unit sphere in R d with S d-1 dθ = 1, P θ is a linear projection onto a onedimensional subspace defined by θ and W p p is an ordinary p-Wasserstein distance by Eq.5.
this section cite: ['b3', 'b4', 'b4', 'b5']

Section: Method
Below we give a detailed description of SW-Guidance algorithm. We placed some necessary theoretical fact, Proposition 1 and Lemma 2, at the end of this section.
The general scheme of the algorithm is illustrated in Fig. 2. We denote x T , . . . , x 0 as the latent states of our diffusion sampling, where x T is a sample from a normal distribution, x 0 is a noiseless sample and x0 (x t ) is a prediction of x 0 for given x t . D(x 0 ) is a decoded image from the latent space of diffusion to the real image domain. Lastly, Ψ is a feature extractor, which in our case is the color distribution of an image in RGB color space.
this section cite: []

Section: Algorithm 1
Color Conditional Generation with Sliced Wasserstein Guidance 1: Initialize latent vector x T ∼ N (0, I), set learning rate λ lr , y -samples from the reference color distribution 2: for t = T to 1 do 3: u ← 0 ▷ Initialize control vector 4:
for j = 1 to M do 5:
x ′ t ← x t + u 6:
Get prediction of last latent x0 ← DDIM(t, x ′ t )
7:
Get ŷ0 ← VAE(x 0 ) ▷ Decode latent to image 8:
for k = 1 to K do ▷ Sliced Wasserstein
9: Project samples on a random direction θ 10: Update loss L ← L + |cdf ŷ0cdf y | 11: end for 12: Update control vector u ← u -λ lr ∇ u L(u) 13: end for 14: Update latent x * t ← x t + u 15: Get denoised latent x t-1 ← DDIM(t, x * t ) 16: end for
The proposed Algorithm 1 initializes a noise tensor x T sampled from a latent normal distribution. Over T diffusion timesteps, the noise tensor is iteratively refined. Following each denoising step, a predicted result x0 is decoded to obtain an image ŷ0 = D(x 0 ). To modulate guided diffusion within each timestep, we add an auxiliary control tensor u following [39]. That is, u is initialized with zeros and we set x ′ t = x t + u. Then we predict original sample x0 (x ′ t ) = x0 (u) and compute gradient of the Sliced Wasserstein distance (SW) between the color distribution of a reference image π ref and the predicted ŷ0 (u) = D(x 0 (u)) with a respect to u
L(u) = SW 1 (π ŷ0(u) , π ref ),(7)
The control vector u is optimized over M steps to shift the reverse diffusion process toward the reference's color distribution. This optimization accumulates gradients w.r.t u M times and minimizes the loss function L, Eq.7. Let us note that by Lemma 2 the minimization of the loss L will lead to a weak convergence of generated color distribution π ŷ0(u) towards the reference π ref with the convergence of all moments.
The loss function can incorporate first two moments (mean µ and covariance σ) of π ŷ0(u) and π ref
L(ŷ 0 (u)) = (µ ŷ0 -µ ref ) 2 + (σ ŷ0 -σ ref ) 2 + SW(π ŷ0 , π ref ),(8)
The impact of adding the first two moments (see Fig. 5), along with other variants of the SW distance such as Generalized [8], Distributional [9], and Energy-Based [11] SW distances, is studied in the Experiments section.
Let u ⋆ be a shift, obtained after M steps of Eq. 7 optimization. Then we set x ⋆ t = x t + u ⋆ and perform usual DDIM [43] denoising step for x ⋆ t with classifier-free guidance to obtain x t-1 . Full algorithm for a latent diffusion model with classifier-free guidance is listed in the Appendix.
Efficient Computation of Sliced Wasserstein Let F 0 and F 1 be two cumulative distribution functions of 1-dimensional probability distributions π 0 and π 1 . Then the Wasserstein distance of order p between π 0 and π 1 has a form (Rachev and Rüschendorf, 1998, Theorem 3.1.2 [44])
W p (π 0 , π 1 ) = 1 0 F -1 0 (y) -F -1 1 (y) p dy 1 p(9)
Formally, it involves differentiable estimation of inverse cumulative density functions. However, in the case of p = 1 the Proposition 1 allows us to replace the difference of inverse CDFs by absolution difference of CDFs, making it much easier to compute
W 1 (π 0 , π 1 ) = ∞ -∞ |F 0 (x) -F 1 (x)| dx(10)
Moreover, since all color distributions in RGB space have a compact support (unit cube), one can employ guarantees of Lemma 2, which in fact states a convergence of general p-Wasserstein distances given convergence of the 1-Wasserstein distance. These facts justify the selection of 1-Wasserstein instead of general p-Wasserstein.
this section cite: ['b38', 'b7', 'b8', 'b10', 'b42', 'b43']

Section: Differentiable Approximation of CDF
We approximate the cumulative distribution function (CDF) by sorting samples from the distribution. Once the samples {x i } n i=1 are sorted, the CDF can be directly obtained by assigning a rank to each sorted sample. For a given sample x i , its rank (i.e., its position in the sorted array) divided by the total number of samples n provides the CDF value at that point. If {x (i) } represents the sorted samples, the CDF at x (i) is given by:
CDF(x (i) ) = i n(11)
This sorting operation is differentiable, so the CDF is also differentiable. To achieve a good approximation of the true underlying CDF, a large number of samples n is required.
this section cite: []

Section: Theoretical Justification
We need Proposition 1 for efficient sampling, as it allows one to avoid computing the inverse CDF. Proposition 1. Let F and G be two cumulative distribution functions. Then,
1 0 F -1 (t) -G -1 (t) dt = R |F (x) -G(x)| dx,(12)
where F -1 and G -1 are the quantile functions (inverse CDFs) of F and G, respectively.
Lemma 2 provides the theoretical foundation for our optimization procedure for multidimensional Borel probability measures µ n and µ on R d . Lemma 2. Let µ n and µ be Borel probability measures on the unit cube in R d . If the numerical sequence
lim n→∞ SW (µ n , µ) = 0 (13
)
then the sequence µ n converges to µ weakly, and all moments of µ n converge to the moments of µ.
Proofs for Proposition 1 and Lemma 2 are provided in the Appendix. Table 1: Quantitative evaluation, SDXL [46]. We measure palette similarity with 2-Wasserstein distance between the color distribution of the generated image and the reference image. CLIP-IQA and CLIP-T are quality and content scores. The color transfer methods [18,19,20,52,53,54,55,56] are applied to the unconditional SDXL generations. Note, that SW-Guidance has the highest CLIP-T among other stylized generation algorithms [30,34,39]. For visual comparisons, see the Appendix.
2-Wasserstein distance [4] ↓ Algorithm mean ± std of mean SW-Guidance SDXL (ours) 0.0297 ± 0.0005 hm-mkl-hm [52] 0.0543 ± 0.0011 hm [53] 0.0856 ± 0.0016 PhotoWCT2 [20] 0.1028 ± 0.0014 ModFlows [54] 0.1125 ± 0.0016 MKL [55] 0.1191 ± 0.0017 CT [56] 0.1333 ± 0.0018 WCT2 [18] 0.1347 ± 0.0017 PhotoNAS [19] 0.1608 ± 0.0017 InstantStyle SDXL [34] 0.1758 ± 0.0028 IP-Adapter SDXL [30] 0.2193 ± 0.0032 Unconditional SDXL [48] 0.3824 ± 0.0059
RB-Modulation [39] Stable Cascade 0.3795 ± 0.0133 Content scores CLIP-IQA [51] ↑ CLIP-T [42] ↑ 0.285 ± 0.004 0.270 ± 0.002 0.259 ± 0.003 0.277 ± 0.002 0.244 ± 0.003 0.282 ± 0.002 0.225 ± 0.003 0.276 ± 0.002 0.257 ± 0.003 0.282 ± 0.002 0.238 ± 0.003 0.283 ± 0.002 0.230 ± 0.003 0.284 ± 0.002 0.179 ± 0.002 0.288 ± 0.002 0.167 ± 0.002 0.279 ± 0.002 0.332 ± 0.003 0.238 ± 0.002 0.247 ± 0.002 0.214 ± 0.002 0.239 ± 0.003 0.294 ± 0.002 0.323 ± 0.006 0.266 ± 0.003
this section cite: ['b45', 'b17', 'b18', 'b19', 'b49', 'b50', 'b51', 'b52', 'b53', 'b29', 'b33', 'b38', 'b3', 'b49', 'b50', 'b19', 'b51', 'b52', 'b53', 'b17', 'b18', 'b33', 'b29', 'b46']

Section: Experiments
As a successor to Universal Diffusion Guidance [37], the proposed method is not tied to a specific architecture and can be paired with latent or pixel-space diffusion models. For our experiments we have selected Stable Diffusion 1.5 [45] and Stable Diffusion XL [46] (Dreamshaper-8 [47] and RealVisXL-V4 [48]) with the DDIM scheduler [43].
this section cite: ['b36', 'b44', 'b45', 'b46', 'b42']

Section: Test set
The experiments are conducted on images generated from the first 1000 prompts taken from the ContraStyles dataset [49]. Our color references are 1000 photos from Unsplash Lite [50]. We refer to these prompts and photos as the test set. A training set is not needed for our algorithm.
Metrics To measure stylization strength, we calculate the Wasserstein-2 distance between color distributions in RGB space. Two content-related metrics are based on CLIP embeddings [42]. CLIP-IQA [51] is a cosine similarity between a generated image and pre-selected anchor vectors that define "good-looking" pictures. CLIP-T [42] is a cosine similarity between CLIP representations of a text prompt and an image generated from this prompt. In other words, the CLIP-T score indicates whether a modified sampling process still follows the initial text prompt, while CLIP-IQA measures the overall quality of the pictures.
Baselines As discussed earlier, the problem of color-conditional generation can be solved by first creating an image from a text prompt and then performing a color transfer with a specialized color transfer algorithm. Therefore, the largest family of baselines consists of algorithms of this kind applied to the output of SD1.5 and SDXL: Histogram matching (hm) [53], CT [56], MKL [55,57], WCT2 [18], PhotoNAS [19], PhotoWCT2 [20], ModFlows [54]. The baseline "hm-mkl-hm" is a combination of histogram matching and MKL taken from the library [52]. In addition, we take three of the currently available baselines for stylized generation: IP-Adapter [30], InstantStyle [34], and RB-Modulation [39], though stylized generation is not exactly the problem we aim to solve. While recent work [58] also proposes an algorithm for color conditional generation with diffusion models, we exclude direct comparisons due to absence of open-source implementation. The term Unconditional indicates that no post-processing steps or controls were applied. We provide CLIP-IQA and CLIP-T metrics for Unconditional SDXL and Unconditional SD1.5 as a reference.
this section cite: ['b47', 'b41', 'b48', 'b41', 'b50', 'b53', 'b52', 'b54', 'b17', 'b18', 'b19', 'b51', 'b49', 'b29', 'b33', 'b38', 'b55']

Section: Comparison Results
The results in Table 1 prove that SW-Guidance achieves superior performance in color-conditional generation compared to all baseline methods. In particular, SW-Guidance has the minimal Wasserstein distance to the reference palette. At the same time, SW-Guidance has the highest CLIP-T among other algorithms for stylized generation (i.e. IP-Adapter, InstantStyle and RB-Modulation). This indicates the ability of SW-Guidance to follow the prompt without adding irrelevant features from the reference image in contrast to other stylization methods. In terms of overall image quality SW-Guidance holds the second place according to CLIP-IQA. Qualitative comparison with stylized generation is given in Fig. 3, with additional visual examples available in the Appendix. The Appendix also contains SD-1.5 performance scores and examples comparable to those shown in Table 1.   Compatibility with ControlNets SW-Guidance can be combined with other control methods to define the image layout, see Fig. 4 for the canny control and Fig. 6 for the depth map control. Our method supports any picture, representing a palette, as in the second row, Fig. 4.
Note that stylizing algorithms, such as InstantStyle, transfer not only color but also other features (see Fig. 3), making it difficult to control color separately. Fig. 6 shows that for InstantStyle the text prompt guiding the color is ignored because it contradicts the features of the reference image (i.e denim dress). Our method is more flexible and sets a red shade which aligns with the reference palette.
Relying only on text prompts for color control is inconvenient. Moreover, color naming is often connotative, and words like "lavender", "emerald" and "lime" can introduce unintended content details, as shown in Fig. 8. Please refer to the Appendix for more examples.
With all this said, we conclude that the proposed SW-Guidance is superior in color stylization while maintaining both integrity with the textual prompt and the quality of the produced images.
this section cite: []

Section: Ablation study

this section cite: []

Section: Generation time
The generation time dependence on M (inner steps) and K (number of slices) for SD-1.5 is shown in Fig. 7. For our main experiments we set M = 10 and K = 10, which results in 30 seconds for SD-1.5 and around 1 minute for SDXL to generate an image on Nvidia RTX 4090 GPU. This represents an improvement compared to the 2 minutes required by RB-modulation.
this section cite: []

Section: Mean and covariance terms
The impact of adding the first two moments to the SW distance is presented in Fig. 5 and Table 3. The best results are obtained with the loss function by Eq.7 (SW only). Mean and covariance terms (Eq.8, Moments + SW) do not increase color similarity and tend to produce images of worse quality. Moments-only guidance is insufficient.
Dependence on learning rate This experiment can be found in Appendix section.
this section cite: []

Section: Limitations and Discussion
The first important limitation of the proposed guidance is its sensitivity to the information about colors in text prompts, especially when they contradict the selected style reference. A clash between the textual and SW guidance typically results in visual artifacts, so detailed textual palette descriptions should be avoided.
Secondly, combining this method with existing stylizing attention-based approaches is not guaranteed to work, as strong stylizing methods could also lead to a clash of color guidance. Ideally, other conditioning should be disentangled from the color information. This collision effect is a subject for further research. As an example, we provide a joint run of InstantStyle and SW-Guidance (Fig. 9).
The last point we would like to discuss is the current implementation's requirement to differentiate through a U-net. Theoretically, this requirement could be avoided, but like the previous point, it requires additional study.
To sum up, this paper presents SW-Guidance, a novel training-free technique for color-conditional generation that can be applied to a range of denoising diffusion probabilistic models. Our study covers the SD-1.5 and SDXL architectures, and for both implementations, we achieved superior results in color similarity compared to color transfer algorithms and models for stylized generation. Numerically, we show the ability of SW-Guidance to maintain integrity with the textual prompt and preserve the quality of the produced images. Our qualitative examples demonstrate the absence of unwanted textures and irrelevant features from the reference image.   Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The work does not use any LLM for methodology development or any original part.
this section cite: []

Section: A Sliced Wasserstein Distances
Sliced Wasserstein Distance Wasserstein distances appear to be natural for our task of color transfer as they measure the cost of transporting one probability distribution to match another [4].
The Wasserstein distance of order p is
W p (π 0 , π 1 ) = inf π∈Π(π0,π1) X0×X1 ||x -y|| p dπ(x, y) 1/p ,(14)
where Π(π 0 , π 1 ) represents the set of all joint distributions with marginals π 0 and π 1 . However, directly computing W p (π 0 , π 1 ) is computationally expensive and difficult to differentiate through, because its value is itself a result of an optimization procedure inf over all transport plans Π(π 0 , π 1 ).
To overcome this issue, the sliced Wasserstein (SW) distance was introduced [5], offering a more computationally tractable alternative by reducing high-dimensional distributions to one-dimensional projections where the Wasserstein distance can be computed more straightforwardly. The sliced p-Wasserstein distance is defined as [5,6]:
SW p (π 0 , π 1 ) = S d-1 W p p (P θ π 0 , P θ π 1 ) dθ 1/p ,(15)
where S d-1 is the unit sphere in R d with S d-1 dθ = 1, P θ is a linear projection onto a onedimensional subspace defined by θ ( Radon transformation in general) and W p p is an ordinary p-Wasserstein distance by Eq.14.
A known issue with the Sliced Wasserstein (SW) distance arises when sampling parameters θ for projections. As noted in [8], uniformly sampled θ values on the unit sphere S d-1 in high dimensions tend to be nearly orthogonal. This resulting in W 2 (P θ π 0 , P θ π 1 ) ≈ 0 with high probability. Consequently, these projections fail to provide discriminative information about the differences between the distributions π 0 and π 1 .
this section cite: ['b3', 'b4', 'b4', 'b5', 'b7']

Section: Distributional Sliced Wasserstein Distance
The Distributional Sliced Wasserstein (DSW) distance, proposed in [9] generalizes the SW distance by introducing a probability distribution σ(θ) over the slicing directions and defined as:
DSW p (π 0 , π 1 ) = = sup σ S d-1 W p p (P θ π 0 , P θ π 1 ) σ(θ)dθ 1/p ,(16)
where the optimization sup is performed w.r.t probability distributions σ over unit sphere S d-1 , with
S d-1 σ(θ)dθ = 1.
this section cite: ['b8']

Section: Energy-Based Sliced Wasserstein Distance
The Energy-Based Sliced Wasserstein (EBSW) distance, introduced in [11], provides an alternative to the optimization-based approach of DSW by defining a slicing distribution σ π0,π1 (θ; f, p) based on the projected Wasserstein distances:
σ π0,π1 (θ; f, p) ∝ f (W p p (P θ π 0 , P θ π 1 )),(17)
where f is a monotonically increasing energy function (e.g., f (x) = e x ) that emphasizes directions with larger projected Wasserstein distances. Using this slicing distribution, the EBSW distance is defined as:
EBSW p (π 0 , π 1 ; f ) = E θ∼σπ 0 ,π 1 (θ;f,p) W p p (P θ π 0 , P θ π 1 ) 1/p .(18)
To improve computational efficiency, importance sampling is used, with a proposal distribution σ 0 (θ) to sample directions and weight them according to the ratio:
w π0,π1,σ0,f,p (θ) = f (W p p (P θ π 0 , P θ π 1 )) σ 0 (θ) .(19)
Generalized Sliced Wasserstein Distance The Generalized Sliced Wasserstein (GSW) distance [8] replaces the Radon transform with a generalized Radon transform that depends on a defining function g(x, θ). Formally, for a function I, the generalized Radon transform is defined as:
GI(t, θ) = R d I(x)δ(t -g(x, θ)) dx,(20)
where δ is the Dirac delta function. Using the generalized Radon transform, the GSW distance between two distributions π 0 and π 1 is defined as:
GSW p (π 0 , π 1 ) = Ω θ W p p (GI π0 (•, θ), GI π1 (•, θ)) dθ 1/p ,(21)
where Ω θ is a compact set of feasible parameters for the function g(x, θ) (e.g.,
Ω θ = S d-1 for g(x, θ) = ⟨x, θ⟩).
For empirical distributions π 0 and π 1 , represented by samples {x i } N i=1 and {y j } N j=1 , the GSW distance can be approximated as:
GSW p (π 0 , π 1 ) ≈ 1 L L l=1 N n=1 g(x i[n] , θ l ) -g(y j[n] , θ l ) p 1/p ,(22)
where x i[n] and y j[n] denote the sorted indices of {g(x i , θ l )} N i=1 and {g(y j , θ l )} N j=1 , respectively, for each sampled θ l .
this section cite: ['b10', 'b7']

Section: B Theoretical Justification
This section contains proofs of Proposition 1 and Lemma 2 from the main text (here they are numbered as Proposition 4 and Lemma 5). Though the statement of Proposition 4 can be found in the literature, its formal treatment is omitted [4,59]. Here we provide its detailed proof for Borel probability measures on R. It restricts us to non-decreasing, right-continuous cumulative distribution functions F , Fig 10. We need Proposition 4 for efficient sampling, as it allows one to avoid computing the inverse CDF. First we prove Lemmas 1, 2 and 3.
this section cite: ['b3', 'b56']

Section: Lemma 1.
Let F be a cumulative distribution function (CDF) on R, and let F -1 (t) = inf{x ∈ R | F (x) ≥ t} be its quantile function for t ∈ [0, 1]. Then:
{t ∈ [0, 1] | F -1 (t) ≤ a} = {t ∈ [0, 1] | F (a) ≥ t}.(23)
Proof. L.H.S. ⇒ R.H.S.: Suppose t ′ ∈ {t ∈ [0, 1] | F -1 (t) ≤ a}. Then, there exists
x ′ = F -1 (t ′ ) such that x ′ ≤ a.
By the definition of the quantile function
F -1 (t ′ ), x ′ is the infimum of the set {x | t ′ ≤ F (x)}.
Under the assumptions that F is right-continuous, the infimum x ′ belongs to the set, and therefore F (x ′ ) ≥ t ′ . Since F (x) is non-decreasing and a ≥ x ′ , it follows that F (a) ≥ F (x ′ ) ≥ t ′ . Hence, t ′ ∈ {t ∈ [0, 1] | F (a) ≥ t}.
R.H.S. ⇒ L.H.S.: Suppose t ′ ∈ {t ∈ [0, 1] | F (a) ≥ t}, but t ′ / ∈ {t ∈ [0, 1] | F -1 (t) ≤ a}, i.e. t ′ such that t ′ ≤ F (a) and F -1 (t ′ ) > a. However, by the definition of x ′ = F -1 ( t), x ′ is the infimum of the set {x | F (x) ≥ t ′ }. Since a < x ′ , a cannot belong to this set, implying F (a) < t ′ , which contradicts the assumption t ′ ≤ F (a). Thus, there is no t in the R.H.S. that does not also belong to the L.H.S.
From these, we conclude that the two sets are equal:
{t ∈ [0, 1] | F -1 (t) ≤ a} = {t ∈ [0, 1] | F (a) ≥ t}.(24)
Lemma 2. Let F be a cumulative distribution function (CDF) on R. Then the quantile function F -1 (t) = inf{x ∈ R | F (x) ≥ t}, defined for t ∈ [0, 1], is measurable with respect to the Borel sigma algebra.
Proof. To show that F -1 (t) : ([0, 1], B [0,1] ) → (R, B R ) is measurable, we must prove that for any Borel set B ⊂ R, the preimage:
{t ∈ [0, 1] | F -1 (t) ∈ B} ∈ B [0,1] . (25
)
The Borel sigma algebra B R is generated by intervals of the form (-∞, b]. Hence, it suffices to prove that for any b ∈ R, the set {t ∈
[0, 1] | F -1 (t) ∈ (-∞, b]} (26
)
is measurable.
Consider the preimages of (-∞, b]: Lemma 3. Let a and b be two real numbers. Then:
{t ∈ [0, 1] | F -1 (t) ∈ (-∞, b]} = = {t ∈ [0, 1] | F -1 (t) ≤ b} = /by Lemma 1/ = {t ∈ [0, 1] | F (b) ≥ t} =[0, F (b)](27)
|a -b| = R |I a≥u -I b≥u | du, (28
)
where I x≥u is the indicator of the set {x ∈ R|x ≥ u}.
Proof. First, suppose a ≥ b. Consider three cases for u:
Hence, we conclude:
1 0 F -1 (t) -G -1 (t) dt = R |F (x) -G(x)| dx.(38)
Lemma 5 (Lemma 2 in the main text) provides the theoretical foundation for our optimization procedure for multidimensional Borel probability measures µ n and µ on R d . Lemma 5. Let µ n and µ be Borel probability measures on the unit cube [0, 1
] d ⊂ R d . If lim n→∞ SW (µ n , µ) = 0,(39)
then µ n converges weakly to µ, and all moments of µ n converge to the moments of µ.
Proof. Consider the ball B(0, R) of radius R, that contains the unit cube. Then a Borel probability measure on the cube [0, 1] d can be extended to the Borel probability measure on B(0, R) by assigning measure zero to any Borel set outside of the cube.
Now we can use Lemma 5.1.4 from [60], which states that for the 1-Wasserstein distance W 1 there exists a constant C d > 0 such that for all Borel probability measures µ, ν on B(0, R)
0 ≤ W 1 (µ, ν) ≤ C d R d d+1 SW 1 (µ, ν) 1 d+1 . (40
)
Since µ n and µ are supported on the unit cube in R d , we take R = √ d, which is a sufficient radius to bound the unit cube. From the assumption that lim n→∞ SW 1 (µ n , µ) = 0, we have:
lim n→∞ C d R d d+1 SW 1 (µ n , µ) 1 d+1 = 0. (41
)
Using the squeeze Theorem for (40), it follows that:
lim n→∞ W 1 (µ n , µ) = 0.(42)
By Definition 6.8 (iv) and Theorem 6.9 of [4], the convergence W 1 (µ n , µ) → 0 implies that µ n converges weakly to µ. Specifically, for any x 0 ∈ B(0, R) and all continuous functions φ with
|φ| ≤ C (1 + d(x 0 , x)), C ∈ R one has lim n→∞ φ(x) dµ n (x) = φ(x) dµ(x).(43)
For our case d(x 0 , x) ≤ 2R, so φ is bounded, and integration over the B(0, R) could be replaced with integration over the unit cube by a construction of our extension of µ n and µ.
Given a (finite) multi-index ᾱ = (α 1 , α 2 , . . . , α d ), one can define the moment:
m ᾱ = x α1 1 x α2 2 • • • x α d d dµ(x).(44)
Polynomial functions ϕ(x) = x ᾱ are bounded and continuous on the unit cube because x i ≤ 1 for all i ∈ {1, . . . , d}, ensuring all terms x ᾱ ≤ 1. Thus, weak convergence implies that for all multi-indices ᾱ,
lim n→∞ x ᾱ dµ n (x) = x ᾱ dµ(x),(45)
i.e., all moments of µ n converge to the corresponding moments of µ component-wise.
Algorithm 2 Color Conditional Generation with Sliced Wasserstein Guidance for latent text-to-image diffusion Require: DDIM: Diffusion DDIM scheduler s θ : UNet model D: Decoder of the Variational Autoencoder E: Encoder of the Variational Autoencoder τ : Text embeddings for conditioning I ref : Reference image γ: Guidance scale factor M : Number of optimization steps Initialize x t ∼ N (0, I) 1: for t in {0, . . . , T -1} do 2: u ← 0 (tensor with same shape as x t ) 3:
for j in {1, . . . , M } do 4:
x ′ t ← x t + u 5: ϵ ← s θ (x ′ t , t, τ ) 6:
x0 ← DDIM(ϵ, t, x ′ t )
7:
I gen ← D(x 0 ) 8:
Table 4: Text-to-image generation conditioned on a reference color distribution. Quantitative evaluation, SD1.5 [47]. 2-Wasserstein distance between the color distributions measures color similarity, CLIP-IQA and CLIP-T are quality and content scores. All color transfer methods [18,19,20,52,53,54,55,56] are applied to the Unconditional SD1.5 generations.
this section cite: ['b57', 'b3', 'b17', 'b18', 'b19', 'b49', 'b50', 'b51', 'b52', 'b53']

Section: 2-Wasserstein distance [4] ↓
Algorithm mean ± std of mean SW-Guidance SD-1.5 (ours) 0.0328 ± 0.0003 hm-mkl-hm [52] 0.0572 ± 0.0011 hm [53] 0.0896 ± 0.0019 PhotoWCT2 [20] 0.1085 ± 0.0016 ModFlows [54] 0.1182 ± 0.0015 Colorcanny ControlNet SD-1.5 [61] 0.1183 ± 0.0016 MKL [55] 0.1274 ± 0.0018 CT [56] 0.1412 ± 0.0019 WCT2 [18] 0.1425 ± 0.0018 PhotoNAS [19] 0.1724 ± 0.0017 InstantStyle SD-1.5 [34] 0.2802 ± 0.0043 Unconditional SD-1.5 0.4062 ± 0.0063
Content scores CLIP-IQA [51] ↑ CLIP-T [42] ↑ 0.2221 ± 0.0029 0.2624 ± 0.0017 0.2013 ± 0.0030 0.2656 ± 0.0017 0.2054 ± 0.0029 0.2700 ± 0.0016 0.1796 ± 0.0026 0.2621 ± 0.0016 0.2035 ± 0.0030 0.2640 ± 0.0016 0.1953 ± 0.0025 0.2600 ± 0.0018 0.1880 ± 0.0028 0.2700 ± 0.0016 0.1826 ± 0.0027 0.2713 ± 0.0016 0.1819 ± 0.0026 0.2761 ± 0.0016 0.2878 ± 0.0027 0.2590 ± 0.0015 0.1891 ± 0.0020 0.2554 ± 0.0018 0.2010 ± 0.0023 0.2837 ± 0.0016
this section cite: ['b49', 'b50', 'b19', 'b51', 'b58', 'b52', 'b53', 'b17', 'b18', 'b33']

Section: C Additional results

this section cite: []

Section: Dependence on learning rate
The effect of learning rates on the performance of sliced Wassersteinbased guidance is given in Fig. 16. The learning rate has a significant impact on the 2-Wasserstein distance, with an optimal value of 0.04, beyond which the loss plateaus and then increases. In contrast, the CLIP-IQA and CLIP-T metrics exhibit linear relationships with respect to the learning rate, suggesting no minimum or optimal value within the range tested.
Text prompts to control the color Using text prompts for controlling the color has several major issues. The first row of Fig. 15 shows that the red color specified by the prompt is often ignored. The second row of Fig. 15 shows how the same prompt applied to another control image produces completely different color distribution. It also introduces content details due to connotative words like "denim", "warm" and "soft". Removing these words alters the colors, making the prompt design tedious. Please note, that color naming is often connotative, and words like "bloody red" and "lime" will introduce content details.
this section cite: []

Section: Content Diversity Evaluation
To evaluate the content diversity of the generated images, we computed the FID between unconditional SDXL generations and those obtained using various style guidance methods. To mitigate potential effects of color distribution alignment on the FID, all evaluations were conducted after conversion to grayscale histogram normalized images. The results on our generated dataset are summarized in Table 5. The results show that SW-Guidance maintains content diversity comparable to other state-of-the-art stylization methods. While there is a slight FID increase compared to simple moment matching (which provides weaker color control), our method preserves substantially more diversity than stronger stylization techniques such as IP-Adapter and RB Modulation.   4 for the quantitative comparison.   13 0.0500 0.0525 0.0550 0.0575 0.0600 2-Wasserstein 2-Wasserstein Distance 0.220 0.225 0.230 0.235 0.240 CLIP-IQA CLIP-IQA Score 0.02 0.04 0.06 0.08 0.10 0.12 Learning Rate 0.22 0.24 0.26 CLIP-T CLIP-T Score Content (SDXL) MKL PhotoWCT2 ModFlows hm hm-mkl-hm Ours Style  1 in the main text for the quantitative comparison.
11
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The claims made in abstract and introduction are well supported in the main part.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The paper contains the dedicated section, that discusses limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: The paper does include theoretical grounds discussed in section Method. All proofs also presented in the Appendix.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The paper provides explicit explanation on how the results could be reproduced altogether with models architecture and source code https://anonymous.4open.  science/r/sw-guidance-3E7D.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We describe all the data used for evaluating and provide a source code. We plan to publish all the data in the case of acceptance. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.
) necessary to understand the results? Answer: [Yes] Justification: This information can be found in the source code and Experiments section. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: Error assessment is presented in comparison tables. 8. Experiments compute resources Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: This information can be found in Experiment and Supplementary sections. 9. Code of ethics Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer:[Yes] Justification: We got acknowledged with the NeurIPS Code of Ethics and confirm that our research follows its guidelines. 10. Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: The color style guidance generation is domain-specific and primarily intended for artistic and stylistic control in generative models. The method does not introduce new mechanisms for semantic manipulation, identity generation, or content creation that could be directly associated with misinformation or surveillance. As such, we do not anticipate any broader societal impacts, either positive or negative. 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: The paper poses no such risks. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: All datasets included in paper are properly cited and link to them are included. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: We do not introduce any datasets. We use available datasets and credit their sources. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: Crowdsourcing is not used in this study. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The study does not involve human participants or subjects. 16. Declaration of LLM usage
For the case b > a, by a similar argument, integral is not zero only when:
b ≥ u ≥ a |I a≥u -I b≥u | = 1.
and therefore, the integral reduces to
R |I a≥u -I b≥u | du = b a 1 du = b -a.(30)
Thus, in all cases:
|a -b| = R |I a≥u -I b≥u | du.(31)
Proposition 4. Let F and G be cumulative distribution functions (CDFs) on R. Then:
1 0 F -1 (t) -G -1 (t) dt = R |F (x) -G(x)| dx,(32)
where F -1 and G -1 are the quantile functions (generalized inverse CDFs) of F and G, respectively.
Proof. Note, that by Lemma 2 both F -1 and G -1 are measurable and therefore the L.H.S exists. By Lemma 3, its absolute value can be represented as:
1 0 F -1 (t) -G -1 (t) dt = = 1 0 R I F -1 (t)≥u -I G -1 (t)≥u du dt.(33)
Using the property of indicator functions I F -1 (t)≥u = 1 -I F -1 (t)<u , the integral becomes:
1 0 R I F -1 (t)≥u -I G -1 (t)≥u du dt = 1 0 R -I F -1 (t)<u + I G -1 (t)<u du dt = 1 0 R -I F -1 (t)≤u + I G -1 (t)≤u du dt.(34)
where the last equality is correct since function under the Lebesgue integral can be changed on a set of measure zero. Using Lemma 1 we rewrite indicators:
1 0 R -I t≤F (u) + I t≤G(u) du dt (35
)
By Fubini's theorem (justified as the integrand is non-negative and measurable), we can switch the order of integration:
R 1 0 -I t≤F (u) + I t≤G(u) dt du. (36) P gen ← pixels_from_image(I gen ) 9: K ← 10 ▷ Number of slices 10: for k in {1, . . . , K} do 11: R ← rand_rotation_matrix() 12: P R gen ← P T gen R 13: P R ref ← P T ref R 14: for d in {1, . . . , 3} do 15: x rot ← P R gen [:, d] 16: y rot ← P R ref [:, d] 17: cdf x ← get_cdf(x rot ) 18: cdf y ← get_cdf(y rot ) 19: L ← L + mean(|cdf xcdf y |) 20: end for 21: end for 22:
g u ← ∇ u L(u) 23: g u ← gu std(gu) 24: u ← u -λg u 25:
end for 26:
x * t ← x t + u 27: ϵ cond ← s θ (x * t , t, τ ) 28: ϵ uncond ← s θ (x * t , t, ∅) 29: ϵ guided ← ϵ uncond + γ(ϵ cond -ϵ uncond )
30:
x t ← DDIM(ϵ guided , t, x * t ) 31: end for
this section cite: ['b13']

Section: D Experimental Details
The experiments were conducted on images generated by SD-1.5 (Dreamshaper-8) and SDXL (RealVisXL-V4) using the first 1000 ContraStyles prompts [49]. No negative prompts or negative embeddings were used.
We fixed the CFG scale to 5 and the resolution to 768x768 for SDXL. For SD-1.5, the CFG scale was set to 8 and the resolution to 512x512. Both the SDXL and SD pipelines used the DDIM scheduler with 30 inference steps. Images for RB-Modulation were produced by Stable Cascade with a resolution of 1024x1024 and a total of 30 inference steps (20 for stage C and 10 for stage B). Method-specific settings are provided below.
Baselines For InstantStyle, the SDXL and SD-1.5 scales were set to 1.0. For IP-Adapter, the SDXL scale was set to 0.5 because higher scales tended to ignore the text prompt, producing variations of a reference image. The Colorcanny ControlNet for SD-1.5 had a conditioning scale of 1.0. For SW-Guidance, the SD-1.5 learning rate was lr = 0.04. In the SDXL version of SW-Guidance, we did not apply gradient normalization (line 23, Algorithm 2) and set the constant lr = 0.01 • 10 4 = 100.
For evaluation, we used publicly available models and algorithms (i.e., none of them were re-trained or re-implemented). We ran color transfer baselines with the default settings provided by the authors.
We observed that PhotoNAS demonstrated a dependency on the resolution of input images. Specifically, the method was optimized for 512×512 inputs and exhibited noticeable variations in performance, including high-frequency defects when images of different resolutions were used. Therefore, the evaluations for SDXL and DreamShaper were different, as SDXL outputs images in higher resolutions.
this section cite: []

Section: Metrics
The 2-Wasserstein distance was estimated with 3000 randomly sampled points using the "emd" function from the POT library [62]. The CLIP-IQA metric implementation was taken from the 'piq' Python library [63]. The CLIP-T metric was calculated using the model "openai/clip-vit-large-patch14" with an embedding dimension of 768.
Hardware The experiments were conducted on a single workstation equipped with two Nvidia RTX 4090 GPU accelerators and 256 GB of RAM.
this section cite: ['b59', 'b60']

Section: Prompts for illustrations

this section cite: []

Section: References
Ref_id:b0 Title: Image style transfer using convolutional neural networks Year: (2016)
Ref_id:b1 Title: Very deep convolutional networks for large-scale image recognition Year: (2014)
Ref_id:b2 Title: Entropy and distance of random graphs with application to structural pattern recognition Year: (1985)
Ref_id:b3 Title: Optimal transport: old and new Year: (2009)
Ref_id:b4 Title: Wasserstein barycenter and its application to texture mixing Year: (2011-06-02)
Ref_id:b5 Title: Sliced and radon wasserstein barycenters of measures Year: (2015)
Ref_id:b6 Title: Max-sliced wasserstein distance and its use for gans Year: (2019)
Ref_id:b7 Title: Generalized sliced wasserstein distances Year: (2019)
Ref_id:b8 Title: Distributional sliced-wasserstein and applications to generative modeling Year: (2020)
Ref_id:b9 Title: Hierarchical sliced wasserstein distance Year: (2022)
Ref_id:b10 Title: Energy-based sliced wasserstein distance Year: (2024)
Ref_id:b11 Title: Sliced wasserstein with random-path projecting directions Year: (2024)
Ref_id:b12 Title: Stereographic spherical sliced wasserstein distances Year: (2024)
Ref_id:b13 Title: Hierarchical hybrid sliced wasserstein: A scalable metric for heterogeneous joint distributions Year: (2024)
Ref_id:b14 Title: Deep photo style transfer Year: (2017)
Ref_id:b15 Title: Universal style transfer via feature transforms Year: (2017)
Ref_id:b16 Title: A closed-form solution to photorealistic image stylization Year: (2018)
Ref_id:b17 Title: Photorealistic style transfer via wavelet transforms Year: (2019)
Ref_id:b18 Title: Ultrafast photorealistic style transfer via neural architecture search Year: (2020)
Ref_id:b19 Title: Photowct2: Compact autoencoder for photorealistic style transfer resulting from blockwise training and skip connections of high-frequency residuals Year: (2022)
Ref_id:b20 Title: Domain-aware universal style transfer Year: (2021-10)
Ref_id:b21 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b22 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b23 Title: Glide: Towards photorealistic image generation and editing with text-guided diffusion models Year: (2021)
Ref_id:b24 Title: An image is worth one word: Personalizing text-to-image generation using textual inversion Year: (2022)
Ref_id:b25 Title: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation Year: (2023)
Ref_id:b26 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b27 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b28 Title: T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models Year: (2024)
Ref_id:b29 Title: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models Year: (2023)
Ref_id:b30 Title: Styleadapter: A single-pass lora-free model for stylized image generation Year: (2023)
Ref_id:b31 Title: Styledrop: Text-to-image synthesis of any style Year: (2024)
Ref_id:b32 Title: Style aligned image generation via shared attention Year: (2024)
Ref_id:b33 Title: stantstyle: Free lunch towards style-preserving in text-to-image generation Year: (2024)
Ref_id:b34 Title: Instantstyleplus: Style transfer with content-preserving in text-to-image generation Year: (2024)
Ref_id:b35 Title: Diffusion posterior sampling for general noisy inverse problems Year: (2022)
Ref_id:b36 Title: Universal guidance for diffusion models Year: (2023)
Ref_id:b37 Title: Freedom: Trainingfree energy-guided conditional diffusion model Year: (2023)
Ref_id:b38 Title: Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control Year: (2024)
Ref_id:b39 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b40 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b41 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b42 Title: Denoising diffusion implicit models Year: (2021)
Ref_id:b43 Title: Mass Transportation Problems: Volume 1: Theory Year: (2006)
Ref_id:b44 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b45 Title: SDXL: Improving latent diffusion models for high-resolution image synthesis Year: (2024)
Ref_id:b46 Title:  Year: (2024)
Ref_id:b47 Title: Unsplash lite dataset 1.2.2 Year: (2023)
Ref_id:b48 Title: Exploring clip for assessing the look and feel of images Year: (2023)
Ref_id:b49 Title: Plenopticam v1.0: A light-field imaging framework Year: (2021)
Ref_id:b50 Title: Gray-level transformations for interactive image enhancement Year: (1977)
Ref_id:b51 Title: Color style transfer with modulated flows Year: (2024)
Ref_id:b52 Title: The linear monge-kantorovitch linear colour mapping for example-based colour transfer Year: (2007)
Ref_id:b53 Title: Color transfer between images Year: (2001)
Ref_id:b54 Title: Python implementation of colour transfer algorithm based on linear mongekantorovitch solution Year: (2023)
Ref_id:b55 Title: Color alignment in diffusion Year: (2025)
Ref_id:b56 Title: Topics in optimal transportation Year: (2021)
Ref_id:b57 Title: Unidimensional and evolution methods for optimal transportation Year: (2013)
Ref_id:b58 Title: Color-canny controlnet Year: (2023)
Ref_id:b59 Title: Pot: Python optimal transport Year: (2021)
Ref_id:b60 Title: Pytorch image quality: Metrics for image quality assessment Year: (2022)
