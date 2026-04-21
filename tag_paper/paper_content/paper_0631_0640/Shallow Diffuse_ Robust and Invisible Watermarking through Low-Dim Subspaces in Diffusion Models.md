Title: Shallow Diffuse: Robust and Invisible Watermarking through Low-Dim Subspaces in Diffusion Models
Abstract: The widespread use of AI-generated content from diffusion models has raised significant concerns regarding misinformation and copyright infringement. Watermarking is a crucial technique for identifying these AI-generated images and preventing their misuse. In this paper, we introduce Shallow Diffuse, a new watermarking technique that embeds robust and invisible watermarks into diffusion model outputs. Unlike existing approaches that integrate watermarking throughout the entire diffusion sampling process, Shallow Diffuse decouples these steps by leveraging the presence of a low-dimensional subspace in the image generation process. This method ensures that a substantial portion of the watermark lies in the null space of this subspace, effectively separating it from the image generation process. Our theoretical and empirical analyses show that this decoupling strategy greatly enhances the consistency of data generation and the detectability of the watermark. Extensive experiments further validate that Shallow Diffuse outperforms existing watermarking methods in terms of consistency.

Section: Introduction
Diffusion models [1,2] have recently become a new dominant family of generative models, powering various commercial applications such as Stable Diffusion [3,4], DALL-E [5,6], Imagen [7], Stable Audio [8] and Sora [9]. These models have significantly advanced the capabilities of text-to-image, text-to-audio, text-to-video, and multi-modal generative tasks. However, the widespread usage of AI-generated content from commercial diffusion models on the Internet has raised several serious concerns: (a) AI-generated misinformation presents serious risks to societal stability by spreading unauthorized or harmful narratives on a large scale [10][11][12]; (b) the memorization of training data by those models [13][14][15][16][17] challenges the originality of the generated content and raises potential copyright infringement issues; (c) iterative training on AI-generated content, known as model collapse [18][19][20][21][22] can degrade the quality and diversity of outputs over time, resulting in repetitive, biased, or low-quality generations that may reinforce misinformation and distortions in the wild Internet.
To deal with these challenges, watermarking is a crucial technique for identifying AI-generated content and mitigating its misuse. Typically, it can be applied in two main scenarios: (a) the server scenario, where given an initial random seed, the watermark is embedded into the image during the generation process; and (b) the user scenario, where given a generated image, the watermark is injected in a post-processing manner; (as shown in the left two blocks in Figure 2 top). Traditional watermarking methods [23][24][25][26] are mainly designed for the user scenario, embedding detectable watermarks directly into images with minimal modification. However, these methods are susceptible to attacks. For example, the watermarks can become undetectable with simple corruptions such as blurring on watermarked images. More recent methods considered the server scenario [27][28][29][30][31][32], enhancing robustness by integrating watermarking into the sampling process of diffusion models.  [29], RingID [31], and Shallow Diffuse. For each technique, we sampled watermarks using two distinct random seeds and obtained the respective watermarked images. (Bottom) Trade-off between consistency (measured by PSNR, SSIM, LPIPS) and robustness (measured by TPR@1%FPR) for Tree-Ring Watermarks, RingID, and Shallow Diffuse.
For example, recent works [29,31] embed the watermark into the initial random seed in the Fourier domain and then sample an image from the watermarked seed. As illustrated in Figure 1, these methods frequently result in inconsistent watermarked images because they substantially distort the original Gaussian noise distribution. Moreover, since they require access to the initial random seed, it limits their use in the user scenario. To the best of our knowledge, there is no robust and consistent watermarking method suitable for both the server and user scenarios (a more detailed discussion of related works is provided in Appendix B).
To address these limitations, we proposed Shallow Diffuse, a robust and consistent watermarking approach that can be employed for both the server and user scenarios. In contrast to prior works [29,31], which embed watermarks into the initial random seed and tightly couple watermarking with the sampling process, Shallow Diffuse decouples these two steps by exploiting the low-dimensional subspace structure inherent in the generation process of diffusion models [33,34]. The key insight is that, due to the low dimensionality of the subspace, a significant portion of the watermark will lie in its null space, which effectively separates the watermarking from the sampling process (see Figure 2 for an illustration). Our theoretical and empirical analyses demonstrate that this decoupling strategy significantly improves the consistency of the watermark. Moreover, Shallow Diffuse is flexible for both server and user scenarios, with better consistency as well as independence from the initial random seed.
Our contributions. In summary, our proposed Shallow Diffuse offers several key advantages over existing watermarking techniques [23][24][25][26][27][28][29][30][31][32] that we highlight below:
• Flexibility. Watermarking via Shallow Diffuse works seamlessly under both server-side and userside scenarios. In contrast, most of the previous methods only focus on one scenario without an easy extension to the other; see Table 1 and Table 2 for demonstrations.
• Consistency and robustness. By decoupling the watermarking from the sampling process, Shallow Diffuse achieves better consistency and comparable robustness. Extensive experiments (Table 1 and Table 2) support our claims, with extra ablation studies in Figure 5a and Figure 5b.
• Provable guarantees. The consistency and detectability of our approach are theoretically justified. Assuming a proper low-dimensional image data distribution (see Assumption 1), we rigorously establish bounds for consistency (Theorem 1) and detectability (Theorem 2).
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b28', 'b30', 'b28', 'b30', 'b28', 'b30', 'b32', 'b33', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31']

Section: Preliminaries
We start by reviewing the basics of diffusion models [1,2,35], followed by several key empirical properties that will be used in our approach: the low-rankness and local linearity of the diffusion model [33,34].
this section cite: ['b0', 'b1', 'b34', 'b32', 'b33']

Section: Preliminaries on Diffusion Models
Basics of diffusion models. In general, diffusion models consist of two processes:
• The forward diffusion process. The forward process progressively perturbs the original data x 0 to a noisy sample x t for some integer t ∈ [0, T ] with T ∈ Z. As in [1], this can be characterized by a conditional Gaussian distribution p t (x t |x 0 ) = N (x t ; √ α t x 0 , (1-α t )I d ). Particularly, parameters {α t } T t=0 sastify: (i) α 0 = 1, and thus p 0 = p data , and (ii) α T = 0, and thus p T = N (0, I d ).
• The reverse sampling process. To generate a new sample, previous works [1,[35][36][37] have proposed various methods to approximate the reverse process of diffusion models. Typically, these methods involve estimating the noise ϵ t and removing the estimated noise from x t recursively to obtain an estimate of x 0 . Specifically, One sampling step of Denoising Diffusion Implicit Models (DDIM) [36] from x t to x t-1 can be described as:
x t-1 = √ α t-1 x t - √ 1 -α t ϵ θ (x t , t) √ α t :=f θ,t (xt) + 1 -α t-1 ϵ θ (x t , t),(1)
where ϵ θ (x t , t) is parameterized by a neural network and trained to predict the noise ϵ t at time t.
From previous works [38,39], the first term in Equation (1), defined as f θ,t (x t ), is the posterior mean predictor (PMP) that predict the posterior mean E[x 0 |x t ]. DDIM could also be applied to a clean sample x 0 and generate the corresponding noisy x t at time t, named DDIM Inversion.
One sampling step of DDIM inversion is similar to Equation (1), by mapping from x t-1 to x t . For any t 1 and t 2 with t 2 > t 1 , we denote multi-time steps DDIM operator and its inversion as x t1 = DDIM(x t2 , t 1 ) and x t2 = DDIM -Inv(x t1 , t 2 ).
this section cite: ['b0', 'b0', 'b34', 'b35', 'b36', 'b35', 'b37', 'b38']

Section: Text-to-image (T2I) diffusion models & classifier-free guidance (CFG).
The diffusion model can be generalized from unconditional to T2I [3,4], where the latter enables controllable image generation x 0 guided by a text prompt c. In more detail, when training T2I diffusion models, we optimize a conditional denoising function ϵ θ (x t , t, c). For sampling, we employ a technique called classifier-free guidance (CFG) [40], which substitutes the unconditional denoiser ϵ θ (x t , t) in Equation ( 1) with its conditional counterpart εθ (x t , t, c) that can be described as εθ (x t , t, c) = (1 -η)ϵ θ (x t , t, ∅ ∅ ∅) + ηϵ θ (x t , t, c).. Here, ∅ ∅ ∅ denotes the empty prompt, and η > 0 denotes the strength for the classifier-free guidance. For simplification, for any t 1 and t 2 with t 2 > t 1 , we denote multi-time steps CFG operator as x t1 = CFG(x t2 , t 1 , c). DDIM and DDIM inversion could also be generalized to T2I version, denoted by x t1 = DDIM(x t2 , t 1 , c) and x t2 = DDIM -Inv(x t1 , t 2 , c).
this section cite: ['b2', 'b3', 'b39']

Section: Local Linearity and Intrinsic Low-Dimensionality in PMP
In this work, we leverage two key properties of the PMP f θ,t (x t ) introduced in Equation (1) for watermarking diffusion models. Parts of these properties have been previously identified in recent papers [33,41,42], and have been extensively analyzed in [34]. At a given timestep t ∈ [0, T ], consider the first-order Taylor expansion of the PMP f θ,t (x t + λ∆x) at the point x t :
l θ (x t ; λ∆x) := f θ,t (x t ) + λJ θ,t (x t ) • ∆x,(2)
where ∆x ∈ S d-1 is a perturbation direction with unit length, λ ∈ R is the perturbation strength, and J θ,t (x t ) = ∇ xt f θ,t (x t ) denotes the Jacobian of f θ,t (x t ). Within a certain range of noise levels, the learned PMP f θ,t exhibits local linearity, and its Jacobian J θ,t ∈ R d×d is low rank: • Low-rankness of the Jacobian J θ,t (x t ). As shown in Figure 2(a) of [34], the rank ratio for t ∈ [0, T ] consistently displays a U-shaped pattern across various network architectures and datasets: (i) it is close to 1 near either the pure noise t = T or the clean image t = 0, (ii) J θ,t (x t ) is low-rank (i.e., the numerical rank ratio is below 10 -2 ) for all diffusion models within the range t ∈ [0.2T, 0.7T ].
• Local linearity of the PMP f θ,t (x t ). As shown in [34,43], the mapping f θ,t (x t ) exhibits strong linearity across a large portion of the timesteps, i.e., f θ,t (x t + λ∆x) ≈ l θ (x t ; λ∆x), a property that holds consistently true across different architectures trained on different datasets.
this section cite: ['b32', 'b40', 'b41', 'b33', 'b33', 'b33', 'b42']

Section: Watermarking by Shallow-Diffuse
Algorithm 1 Unconditional Shallow Diffuse
1: Inject watermark: 2: Input: original image x0 for the user scenario (initial random seed xT for the server scenario), watermark λ∆x, embedding timestep t * , 3: Output: watermarked image x W 0 , 4: if user scenario then 5: xt * = DDIM -Inv (x0, t * ) 6: else server scenario 7: xt * = DDIM (xT , t * ) 8: end if 9: x W t * ← xt * + λ∆x, x W 0 ← DDIM x W t * , 0 10: ▷ Embed watermark 11: Return: x W 0 12: 13: Detect watermark: 14: Input: Attacked image xW 0 , watermark λ∆x, embedding timestep t * , 15: Output: Distance score η, 16: xW t * ← DDIM -Inv xW 0 , t * 17: η = Detector xW t * , λ∆x 18: Return: η
This section introduces Shallow Diffuse, a training-free watermarking method designed for diffusion models. Building on the benign properties of PMP discussed in Section 2.2, we describe how to inject and detect invisible watermarks in unconditional diffusion models in Section 3.1 and Section 3.2, respectively. Algorithm 1 outlines the overall watermarking method for unconditional diffusion models. In Section 3.3, we generalize our approach to T2I diffusion models as shown in Figure 2.
this section cite: []

Section: Injecting Invisible Watermarks
Consider an unconditional diffusion model ϵ θ (x t , t) as introduced in Section 2.1. Instead of injecting the watermark ∆x in the initial noise, we inject it in a particular timestep t * ∈ [0, T ] with
x W t * = x t * + λ∆x,(3)
where λ ∈ R is the watermarking strength,
x t * = DDIM -Inv (x 0 , t * ) under
the user scenario and x t * = DDIM (x T , t * ) under the server scenario. Based upon Section 2.2, we choose the timestep t * so that the Jacobian of the PMP J θ,t (x t * ) = ∇ xt f θ,t (x t * ) is low-rank. Moreover, based upon the linearity of PMP discussed in Section 2.2, we approximately have f θ,t (x W t * ) = f θ,t (x t * ) + λJ θ,t (x t * )∆x ≈0 ≈ f θ,t (x t * ),
where the watermark ∆x is designed to span the entire space R d uniformly; a more detailed discussion on the pattern design of ∆x is provided in Section 3.2. The key intuition for Equation (4) to hold is that, when r t * = rank(J θ,t (x t * )) is low, a significant proportion of λ∆x lies in the null space of J θ,t (x t * ), so that J θ,t (x t * )∆x ≈ 0.
Therefore, the selection of t * is based on the requirement that f θ,t (x * t ) is locally linear and that the rank of its Jacobian satisfies r * t ≪ d. In practice, we choose t * = 0.3T based on results from the ablation study in Section 5.4. As a result, the injection in Equation (4) preserves better consistency without changing the predicted x 0 . In the meanwhile, it remains highly robust because any attack on x 0 would remain disentangled from the watermark, so that λ∆x remains detectable.
In practice we employ the DDIM method instead of PMP for sampling high-quality images, but the above intuition still carries over to DDIM. From Equation (1), when we inject the watermark ∆x into x * t as given in Equation (3), we know that
x W t * -1 = DDIM(x W t * , t * -1) ≈ √ α t * -1 f θ,t (x t * ) + √ 1 -α t * -1 √ 1 -α t * (x t * + λ∆x - √ α t * f θ,t (x t * )) ,(5)
where the approximation follows from Equation (4). This implies that the watermark λ∆x is embedded into the DDIM sampling process entirely through the second term of Equation ( 5) and it decouples from the first term, which predicts x 0 . Therefore, similar to our analysis for PMP, the first term in Equation ( 5) maintains the consistency of data generation, whereas the difference in the second term, highlighted in blue, serves as a key feature for watermark detection, which we will discuss next. In Section 4, we provide rigorous proofs validating the consistency and detectability of our approach.
this section cite: []

Section: Watermark Design and Detection
Second, building on the watermark injection method described in Section 3.1, we discuss the design of the watermark pattern and the techniques for effective detection.
this section cite: []

Section: Watermark pattern design.
Building on the method proposed by [29], we inject the watermark in the frequency domain to enhance robustness against adversarial attacks. Specifically, we adapt this approach by defining a watermark λ∆x for the input x t * at timestep t * as follows:
λ∆x := DFT -Inv (DFT (x t * ) ⊙ (1 -M ) + W ⊙ M ) -x t * ,(6)
where the Hadamard product ⊙ denotes the element-wise multiplication. Additionally, we have the following for Equation (6):
• Transformation into the frequency domain. Let DFT(•) and DFT -Inv(•) denote the forward and inverse Discrete Fourier Transform (DFT) operators, respectively. As shown in Equation (6), we first apply DFT(•) to transform x t * into the frequency domain, where the watermark is introduced via a mask. Finally, the modified input is transformed back into the pixel domain using DFT -Inv(•).
• The mask and key of watermarks. M is the mask used to apply the watermark in the frequency domain, as shown in the top-left of Figure 3, and W denotes the key of the watermark. Typically, the mask M is circular, with the white area representing 1 and the black area representing 0 in Figure 3. The mask is used to modify specific frequency bands of the image. Specifically, circular mask M has a radius of 8. In the following, we discuss the design of M and W in detail.
In contrast to prior methods [29,31], which design the mask M to modify the low-frequency components of the initial noise input, we construct M to target the high-frequency components of the image. While modifying low-frequency components is effective due to the concentration of image energy in those bands, such approaches often introduce significant visual distortion when watermarks are embedded (see Figure 1 for illustration). In contrast, as shown in Figure 3, our method introduces minimal distortion by operating on the high-frequency components, which correspond to finer details and inherently contain less energy. This effect is further amplified in our case, as we apply the perturbation to x t * , which is closer to the clean image x 0 , rather than to the initial noise used in [29,31]. To isolate the high-frequency components, we apply the DFT without shifting and centering the zero-frequency component, as illustrated in the bottom-left of Figure 3.
In designing the key W , we follow [29]. The key W is composed of multi-rings and each ring has the same value drawn from Gaussian distribution; see the top-right of Figure 3 for an illustration. Further ablation studies on the choice of M , W , and the effects of selecting low-frequency versus high-frequency regions for watermarking can be found in Table 8.
this section cite: ['b28', 'b28', 'b30', 'b28', 'b30', 'b28']

Section: Watermark detection.
During watermark detection, suppose we are given a watermarked image xW 0 with certain corruptions, we apply DDIM Inversion to recover the watermarked image at timestep t * , denoted as xW t * = DDIM -Inv xW 0 , t . To detect the watermark, following [27,29], the Detector(•) in Algorithm 1 computes the following p-value:
η = sum(M ) • ||M ⊙ W -M ⊙ DFT xW t * || 2 F ||M ⊙ DFT xW t * || 2 F ,(7)
where sum(•) is the summation of all elements of the matrix. Ideally, if xW t * is a watermarked image, M ⊙ W = M ⊙ DFT xW t * and η = 0. When xW t * is a non-watermarked image, M ⊙ W ̸ = M ⊙ DFT xW t * and η > 0. By selecting a threshold η 0 , non-watermarked images satisfy η > η 0 , while watermarked images satisfy η < η 0 . The theoretical derivation of the p-value η could be found in [27].
this section cite: ['b26', 'b28', 'b26']

Section: Extension to Text-to-Image (T2I) Diffusion Models
So far, our discussion has focused exclusively on unconditional diffusion models. Next, we show how our approach can be readily extended to T2I diffusion models, which are widely used in practice. Specifically, Figure 2 provides an overview of our method for T2I diffusion models, which can be flexibly applied to both server and user scenarios:
• Watermark injection. Shallow Diffuse embeds watermarks into the noise corrupted image x t * at a specific timestep t * = 0.3T . In the server scenario, given x T ∼ N (0, I d ) and prompt c, we calculate x t * = CFG (x T , t * , c). In the user scenario, given the generated image x 0 , we compute x t * = DDIM -Inv (x 0 , t * , ∅ ∅ ∅), using an empty prompt ∅ ∅ ∅. Next, similar to Section 3.1, we apply DDIM to obtain the watermarked image
x W 0 = DDIM x W t * , 0, ∅ ∅ ∅ .
• Watermark detection. During watermark detection, suppose we are given a watermarked image xW 0 with certain corruptions, we apply the DDIM Inversion to recover the watermarked image at timestep t * , denoted as xW t * = DDIM -Inv xW 0 , t * , ∅ ∅ ∅ . We detect the watermark ∆x in xW t * by calculating η in Equation (7), with detail explained in Section 3.2.
this section cite: ['b6']

Section: Theoretical Justification
In this section, we provide theoretical justifications for the consistency and the detectability of Shallow Diffuse for unconditional diffusion models. We begin by making the following assumptions on the watermark and the diffusion process. Assumption 1. Suppose the following holds for the PMP f θ,t (x t ) introduced in Equation (1):
• Linearity: For any t and ∆x ∈ S d-1 , we always have f θ,t (x t +λ∆x) = f θ,t (x t )+λJ θ,t (x t )∆x.
• L-Lipschitz continuous: we assume that
f θ,t (x) is L-Lipschiz continuous ||J θ,t (x)|| 2 ≤ L, ∀x ∈ R d , t ∈ [0, T ]
It should be noted that these assumptions are mild. The L-Lipschitz continuity is a common assumption for diffusion model analysis [44][45][46][47][48][49]. The approximated linearity have been shown in [34] with the assumption of data distribution to follow a mixture of low-rank Gaussians. For the ease of analysis, we assume exact linearity, but it can be generalized to the approximate linear case with extra perturbation analysis.
Now consider injecting a watermark λ∆x in Equation ( 3), where λ > 0 is a scaling factor and ∆x is a random vector uniformly distributed on the unit hypersphere S d-1 , i.e., ∆x ∼ U(S d-1 ). Then the following hold for f θ,t (x t ).
Theorem 1 (Consistency of the watermarks). Suppose Assumption 1 holds and ∆x ∼ U(S d-1 ).
Define xW 0,t := f θ,t (x t + λ∆x), x0,t := f θ,t (x t ). Then the ℓ 2 -norm distance between xW 0,t and x0,t is bounded by:
|| xW 0,t -x0,t || 2 ≤ λLh(r t ),(8)
with probability at least
1 -r -1 t . Here, h(r t ) = rt d + 18π 3 d-2 log (2r t ).
Theorem 1 guarantees that injecting the watermark λ∆x would only change the estimation by an amount of λLh(r t ) with a constant probability, where h(r t ) only depends on the rank of the Jacobian r t (r t ≪ d) rather than the ambient dimension d. Since r t is small, Equation (8) implies that the change in the prediction would be small. Given the relationship between PMP and DDIM in equation 1, the consistency also applies to practical use. Moreover, in the following, we show that the injected watermark remains detectable based on the second term in Equation ( 5). Theorem 2 (Detectability of the watermark). Suppose Assumption 1 holds and ∆x ∼ U(S d-1 ). With x W t given in Equation (3), define x W t-1 = DDIM x W t , t -1 and xW t = DDIM -Inv x W t-1 , t . The ℓ 2 -norm distance between xW t and x W t can be bounded by:
|| xW t -x W t || 2 ≤ λLh(max{r t-1 , r t })[-g (α t , α t-1 ) + g (α t-1 , α t ) (1 -Lg (α t , α t-1 ))] (9
)
with probability at least
1 -r -1 t -r -1 t-1 . Here, g(x, y) := √ 1-y √ x- √ 1-x √ y √ 1-x , ∀x, y ∈ (0, 1). h(r t ) = rt d + 18π 3 d-2 log (2r t ).
Similarly, the term h(max{r t-1 , r t }) is small because it only depends on the rank of the Jacobian r t or r t-1 (r t-1 , r t ≪ d) rather than the ambient dimension d. Additionally, the term -g (α t , α t-1 ) + g (α t-1 , α t ) (1 -Lg (α t , α t-1 )) is also a small number based on the design of α t for variance preserving (VP) noise scheduler [1]. Together, this implies that the difference between xW t and x W t is small and x W t could be recovered by xW t from one-step DDIM. Therefore, Theorem 2 implies that the injected watermark can be detected with high probability.
this section cite: ['b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b33', 'b7', 'b0']

Section: Experiments
In this section, we present a comprehensive set of experiments to demonstrate the robustness and consistency of Shallow-Diffuse across various datasets. We begin by highlighting its performance in terms of robustness and consistency in both the server scenario (Section 5.1) and the user scenario (Section 5.2). We further explore the trade-off between robustness and consistency in Section 5.3. Lastly, we provide extra multi-key identification experiments in Appendix C.2 and ablation studies on watermark pattern design (Appendix C.3), watermarking embedded channel (Appendix C.4), watermark injecting timestep t (Section 5.4) and inference steps (Appendix C.5).
this section cite: []

Section: Comparison baselines.
For the server scenario, we select the following non-diffusion-based methods: DWtDct [23], DwtDctSvd [23], RivaGAN [50], StegaStamp [51]; and diffusion-based methods: Stable Signature [28], Tree-Ring Watermarks [29], RingID [31], and Gaussian Shading [30]. In the user scenario, we adopt the same baseline methods, except for Stable Signature, as this method are not suitable for this setting.
Evaluation datasets. We use Stable Diffusion 2-1-base [3] as the underlying model for our experiments, applying Shallow diffusion within its latent space. For the server scenario (Section 5.1), all diffusion-based methods are based on the same Stable Diffusion, with the original images x 0 generated from identical initial seeds x T . Non-diffusion methods are applied to these same original images x 0 in a post-watermarking process. A total of 5000 original images are generated for evaluation in this scenario. For the user scenario (Section 5.2), we utilize the MS-COCO [52], and DiffusionDB datasets [53]. The first one is a real-world dataset, while DiffusionDB is a collection of diffusion model-generated images. From each dataset, we select 500 images for evaluation. For the remaining experiments in Section 5.3 and Appendix C, we use the server scenario and sample 100 images for evaluation.
Evaluation metrics. To evaluate image consistency, we use peak signal-to-noise ratio (PSNR) [54], structural similarity index measure (SSIM) [55], and Learned Perceptual Image Patch Similarity (LPIPS) [56], comparing watermarked images to their original counterparts. In the server scenario, we also assess the generation quality of the watermarked images using Contrastive Language-Image Pretraining Score (CLIP-Score) [57] and Fréchet Inception Distance (FID) [58]. To evaluate robustness, we plot the true positive rate (TPR) against the false positive rate (FPR) for the receiver operating characteristic (ROC) curve. We use the area under the curve (AUC) and TPR when FPR = 0.01 (TPR @1% FPR) as robustness metrics.
Attacks. Robustness is comprehensively evaluated both under clean conditions (no attacks) and with 15 types of attacks. Following [59], we categorized them into three groups, including: distortion attack (JPEG compression, Gaussian blurring, Gaussian noise, color jitter, resize and restore, random drop, median blurring), regeneration attack (diffusion purification [60], VAE-based image compression models [61,62], stable diffusion-based image regeneration [63], 2 times and 4 times rinsing regenerations [59]) and adversarial attack (black-box and grey-box averaging attack [64]).
Here, we report only the TPR at 1% FPR for the average robustness across each group and all attacks. Detailed settings and full experiment results of these attacks are provided in Appendix C.1.
this section cite: ['b22', 'b22', 'b49', 'b50', 'b27', 'b28', 'b30', 'b29', 'b2', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b58', 'b63']

Section: Server Scenario Consistency and Robustness
Table 1 compares the performance of Shallow Diffuse with other methods in the server scenario.
For reference, we also apply stable diffusion to generate images from the same random seeds, without adding watermarks (referred to as "SD w/o WM" in Table 1). In terms of generation quality, Shallow Diffuse achieves the best FID and CLIP scores among all diffusion-based methods. It also demonstrates superior generation consistency, achieving the highest PSNR, SSIM, and LPIPS scores. Regarding robustness, Shallow Diffuse performs comparably to Gaussian Shading and RingID, while outperforming the remaining methods. Although Gaussian Shading and RingID show similar levels of generation quality and robustness in the server scenario, their poor consistency makes them less suitable for the user scenario.
this section cite: []

Section: User Scenario Consistency and Robustness
Under the user scenario, Table 2 presents a comparison of Shallow Diffuse against other methods. In terms of consistency, Shallow Diffuse outperforms all other diffusion-based approaches.
To measure the upper bound of diffusion-based methods, we apply stable diffusion with x0 = DDIM(DDIM -Inv(x 0 , t, ∅), 0, ∅) , and measure the data consistency between x0 and x 0 (denoted in SD w/o WM in Table 2). The upper bound is constrained by errors introduced through DDIM inversion, and Shallow Diffuse comes the closest to reaching this limit. For non-diffusion-based methods, which are not affected by DDIM inversion errors, better image consistency is achievable. However, as visualized in Figure 4, Shallow Diffuse also demonstrates strong generation consistency. In terms of robustness, Shallow Diffuse performs comparably to RingID and Gaussian shading, while outperforming all other methods across both datasets. Notably, RingID and Gaussian achieve high robustness at the sacrifice of poor generation consistency (see Table 2 and Figure 4). In contrast, Shallow Diffuse is the only method that balances strong generation consistency with high watermark robustness, making it suitable for both user and server scenarios.
this section cite: []

Section: Trade-off between Consistency and Robustness
Figure 1 bottom illustrates the trade-off between consistency and robustness 2 for Shallow Diffuse and other baselines. As the radius of M increases, the watermark intensity λ also increases, reducing image consistency but improving robustness. By adjusting the radius of M , we plot the trade-off using PSNR, SSIM, and LPIPS against TPR@1%FPR. From Figure 1 bottom, curve of Shallow Diffuse is consistently above the curve of Tree-Ring Watermarks and RingID, demonstrating Shallow Diffuse's better consistency at the same level of robustness.
this section cite: []

Section: Ablation Study over Injecting Timesteps.
Figure 5 shows the relationship between the watermark injection timestep t and both consistency and robustness 3 . Shallow Diffuse achieves optimal consistency at t = 0.2T and optimal robustness (a) Consistency (b) Robustness at t = 0.3T . In practice, we select t = 0.3T . This result aligns with the intuitive idea proposed in Section 3.1 and the theoretical analysis in Section 4: low-dimensionality enhances both data generation consistency and watermark detection robustness. However, according to [34], the optimal timestep r t for minimizing r t satisfies t * ∈ [0.5T, 0.7T ]. We believe the best consistency and robustness are not achieved at t * due to the error introduced by DDIM -Inv. As t increases, this error grows, leading to a decline in both consistency and robustness. Therefore, the best tradeoff is reached at t ∈ [0.2T, 0.3T ], where J θ,t (x t ) remains low-rank but t is still below t * . Another possible explanation is the gap between the image space and latent space in diffusion models. The rank curve in [34] is evaluated for an image-space diffusion model, whereas Shallow Diffuse operates in the latent-space diffusion model (e.g., Stable Diffusion).
this section cite: ['b33', 'b33']

Section: Conclusion
We proposed Shallow Diffuse, a novel and flexible watermarking technique that operates seamlessly in both server-side and user-side scenarios. By decoupling the watermark from the sampling process, Shallow Diffuse achieves enhanced robustness and greater consistency. Our theoretical analysis demonstrates both the consistency and detectability of the watermarks. Extensive experiments further validate the superiority of Shallow Diffuse over existing approaches.
this section cite: []

Section: References
Ref_id:b0 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b1 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b2 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b3 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b4 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b5 Title: Improving image generation with better captions Year: (2023)
Ref_id:b6 Title: Photorealistic text-toimage diffusion models with deep language understanding Year: (2022)
Ref_id:b7 Title: Long-form music generation with latent diffusion Year: (2024)
Ref_id:b8 Title: Video generation models as world simulators Year: (2024)
Ref_id:b9 Title: Defending against neural fake news Year: (2019)
Ref_id:b10 Title: Generative language models and automated influence operations: Emerging threats and potential mitigations Year: (2023)
Ref_id:b11 Title: The malicious use of artificial intelligence: Forecasting, prevention, and mitigation Year: (2018)
Ref_id:b12 Title: On memorization in diffusion models Year: (2023)
Ref_id:b13 Title: Diffusion art or digital forgery? investigating data replication in diffusion models Year: (2023)
Ref_id:b14 Title: Understanding and mitigating copying in diffusion models Year: (2023)
Ref_id:b15 Title: Detecting, explaining, and mitigating memorization in diffusion models Year: (2023)
Ref_id:b16 Title: Wasserstein proximal operators describe score-based generative models and resolve memorization Year: (2024)
Ref_id:b17 Title: Towards theoretical understandings of self-consuming generative models Year: (2024)
Ref_id:b18 Title: Self-consuming generative models go MAD Year: (2024)
Ref_id:b19 Title: A tale of tails: Model collapse as a change of scaling laws Year: (2024)
Ref_id:b20 Title: Ai models collapse when trained on recursively generated data Year: (2024)
Ref_id:b21 Title: Ai models fed ai-generated data quickly spew nonsense Year: (2024)
Ref_id:b22 Title: Digital watermarking and steganography Year: (2007)
Ref_id:b23 Title: Circularly symmetric watermark embedding in 2-d dft domain Year: (2001)
Ref_id:b24 Title: Svd-based digital image watermarking scheme Year: (2005)
Ref_id:b25 Title: An optimized image watermarking method based on hd and svd in dwt domain Year: (2019)
Ref_id:b26 Title: Robust image watermarking using stable diffusion Year: (2024)
Ref_id:b27 Title: The stable signature: Rooting watermarks in latent diffusion models Year: (2023)
Ref_id:b28 Title: Tree-rings watermarks: Invisible fingerprints for diffusion images Year: (2023)
Ref_id:b29 Title: Gaussian shading: Provable performance-lossless image watermarking for diffusion models Year: (2024)
Ref_id:b30 Title: Ringid: Rethinking tree-ring watermarking for enhanced multi-key identification Year: (2024)
Ref_id:b31 Title: ROBIN: Robust and invisible watermarks for diffusion models with adversarial optimization Year: (2024)
Ref_id:b32 Title: Diffusion models learn low-dimensional distributions via subspace clustering Year: (2024)
Ref_id:b33 Title: Exploring low-dimensional subspaces in diffusion models for controllable image editing Year: (2024)
Ref_id:b34 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b35 Title: Denoising diffusion implicit models Year: (2021)
Ref_id:b36 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b37 Title: The emergence of reproducibility and consistency in diffusion models Year: (2024)
Ref_id:b38 Title: Understanding diffusion models: A unified perspective Year: (2022)
Ref_id:b39 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b40 Title: Zero-shot unsupervised and text-based audio editing using DDPM inversion Year: (2024-07)
Ref_id:b41 Title: On the posterior distribution in denoising: Application to uncertainty quantification Year: (2024)
Ref_id:b42 Title: Understanding generalizability of diffusion models requires rethinking the hidden gaussian structure Year: (2024)
Ref_id:b43 Title: Generative modeling with denoising auto-encoders and langevin sampling Year: (2020)
Ref_id:b44 Title: Convergence for score-based generative modeling with polynomial complexity Year: (2022)
Ref_id:b45 Title: Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions Year: (2023)
Ref_id:b46 Title: Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions Year: (2023)
Ref_id:b47 Title: Sample complexity bounds for score-matching: Causal discovery and generative modeling Year: (2023)
Ref_id:b48 Title: Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data Year: (2023)
Ref_id:b49 Title: Robust invisible video watermarking with attention Year: (2019)
Ref_id:b50 Title: Stegastamp: Invisible hyperlinks in physical photographs Year: (2020)
Ref_id:b51 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b52 Title: Large-scale prompt gallery dataset for text-to-image generative models Year: (2022)
Ref_id:b53 Title: Digital image processing Year: (2005)
Ref_id:b54 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b55 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b56 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b57 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b58 Title: Benchmarking the robustness of image watermarks Year: (2024)
Ref_id:b59 Title: Diffusion models for adversarial purification Year: ()
Ref_id:b60 Title: Learned image compression with discretized gaussian mixture likelihoods and attention modules Year: (2020)
Ref_id:b61 Title: Variational image compression with a scale hyperprior Year: (2018)
Ref_id:b62 Title: Generative autoencoders as watermark attackers: Analyses of vulnerabilities and threats Year: (2023)
Ref_id:b63 Title: Can simple averaging defeat modern watermarks? Year: (2024)
Ref_id:b64 Title: Combined dwt-dct digital image watermarking Year: (2007)
Ref_id:b65 Title: Dwt-dct-svd based watermarking Year: (2008)
Ref_id:b66 Title: Redmark: Framework for residual diffusion watermarking based on deep networks Year: (2020)
Ref_id:b67 Title: Convolutional neural network-based digital image watermarking adaptive to the resolution of image and watermark Year: (2020)
Ref_id:b68 Title: Hidden: Hiding data with deep networks Year: (2018)
Ref_id:b69 Title: Deep generative models through the lens of the manifold hypothesis: A survey and new connections Year: (2024)
Ref_id:b70 Title: Diffusion models encode the intrinsic dimension of data manifolds Year: (2024)
Ref_id:b71 Title: A geometric view of data complexity: Efficient local intrinsic dimension estimation with diffusion models Year: (2024)
Ref_id:b72 Title: Entanglement and the foundations of statistical mechanics Year: (2006)
