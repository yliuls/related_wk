Title: Adaptive 3D Reconstruction via Diffusion Priors and Forward Curvature-Matching Likelihood Updates
Abstract: Reconstructing high-quality point clouds from images remains challenging in computer vision. Existing generative models, particularly diffusion models, based approaches that directly learn the posterior may suffer from inflexibility-they require conditioning signals during training, support only a fixed number of input views, and need complete retraining for different measurements. Recent diffusion-based methods have attempted to address this by combining prior models with likelihood updates, but they rely on heuristic fixed step sizes for the likelihood update that lead to slow convergence and suboptimal reconstruction quality. We advance this line of approach by integrating our novel Forward Curvature-Matching (FCM) update method with diffusion sampling. Our method dynamically determines optimal step sizes using only forward automatic differentiation and finite-difference curvature estimates, enabling precise optimization of the likelihood update. This formulation enables high-fidelity reconstruction from both single-view and multi-view inputs, and supports various input modalities through simple operator substitution-all without retraining. Experiments on ShapeNet and CO3D datasets demonstrate that our method achieves superior reconstruction quality at matched or lower NFEs, yielding higher F-score and lower CD and EMD, validating its efficiency and adaptability for practical applications. Code is available at here.

Section: Introduction
Three-dimensional reconstruction has become increasingly important across diverse applications including robotics, autonomous driving, augmented reality, and virtual environments. Among various 3D representations, point clouds serve as a fundamental data structure for representing objects and scenes due to their simplicity and flexibility. However, generating high-quality point clouds that accurately capture intricate details remains challenging, particularly when working with limited input information such as single-view images.
Recent advances in deep generative models, particularly diffusion models, have shown remarkable success in generating high-fidelity images [11,14] and 3D data. Diffusion models use an iterative denoising process to progressively transform random noise into structured outputs, making them effective for capturing complex geometric patterns. In the domain of point cloud generation, researchers have begun exploring diffusion-based approaches with promising results [19,39,38,24,20,21,34].
While diffusion models offer powerful generative capabilities, applying them to 3D reconstruction presents unique challenges due to its nature as an inverse problem. In typical inverse problems (formulated as y = Ax), iterative optimization methods solving least-squares objectives can determine optimal step sizes analytically using gradients that incorporate A ⊤ (the adjoint of A). However, 3D object rendering represents a particularly challenging case where the rendering operator is complex and non-linear, making the computation of the adjoint operation intractable. Since classical step-size   20], BDM [34]), and our approach. Our method achieves higher fidelity reconstructions with better F-scores (0.382) than existing approaches while using fewer function evaluations, particularly excelling at preserving fine structural details.
formulas require an adjoint of the forward operator, the absence of a tractable renderer adjoint complicates step-size selection. This fundamental challenge impacts how researchers approach diffusion-based point cloud generation, especially when incorporating image-based guidance.
Current image-to-point-cloud methods predominantly learn the score of the posterior distribution ∇ log p(X|y) directly, where X represents the point cloud and y represents image measurements. This direct approach incurs significant limitations: it necessitates including images as conditioning signals during training [20], restricts models to a fixed number of input views without specialized encoders [8], and requires computationally expensive retraining whenever measurement types change (e.g., from RGB images to depth maps).
A promising alternative approach [22] decomposes the posterior p(X|y) into a trainable prior p(X) and an updatable likelihood p(y|X), employing Diffusion Posterior Sampling (DPS) [6] with gradient updates via ∇ log p(y|X) for Gaussian splatting-based 3D reconstruction. While this decomposition is conceptually straightforward and modular, current implementations struggle with a critical limitation: determining appropriate step sizes for the likelihood updates. The non-linear nature of 3D rendering prevents analytical step size determination, leading existing methods to rely on heuristic, fixed step sizes [22]. This results in slow convergence, suboptimal reconstruction quality, and often necessitates additional 2D diffusion models for refinement, further complicating the pipeline.
To address these limitations, we present a novel approach that combines a point cloud diffusion model with Forward Curvature-Matching (FCM) optimization. Our approach, illustrated in Fig. 2, computes an adaptive step size using a Barzilai-Borwein rule and refines it with an Armijo backtracking condition, enabling more precise control. Our key insight is that by incorporating FCM's principled, curvature informed step size determination into the diffusion sampling process without any adjoint operations, we can effectively navigate the complex optimization landscape of 3D reconstruction.
Unlike previous DPS-based methods that rely on heuristic step sizes for the likelihood update, our approach employs FCM optimization to dynamically determine optimal step sizes. The key innovation is our reliance solely on the differentiable forward pass for curvature-informed step-size determination, obviating the adjoint. This enhancement enables significantly more efficient and accurate optimization during the diffusion sampling process. The technical contributions of our work include:
• We integrate the FCM method with the reverse process of diffusion models, enabling high-fidelity point cloud reconstruction that accurately matches input images.
• Our gradient-based updates are not constrained by the number of input images, allowing for point cloud reconstruction from either single-view or multi-view images without modifying the base model.
• Our method can be applied to various measurement modalities (such as RGB image to 3D object or depth map to 3D object) by simply substituting the appropriate operator rather than retraining the entire model, significantly enhancing flexibility and efficiency.
We demonstrate the effectiveness of our approach by reconstructing colored point clouds from both synthetic and real-world datasets. Our method achieves more accurate reconstruction with fewer neural function evaluations (NFEs) compared to existing techniques, validating the efficiency of our FCM-based likelihood optimization. We further demonstrate the adaptability of our approach by applying it to both multi-view reconstruction and depth map to point cloud generation without retraining, highlighting its potential for diverse applications. The remainder of this paper is organized as follows: Section 2 reviews related work, Section 3 presents the proposed method, Section 4 details our experimental results, and Section 5 concludes with a discussion of future directions.
2 Related Work 3D Reconstruction from Images. As interest in 3D content creation continues to grow, research on reconstructing 3D shapes from 2D observations has advanced significantly. This challenging task requires inferring complete 3D structures, including both visible and occluded regions, from limited viewpoints. The difficulty is compounded by the scarcity of large-scale 3D datasets.
Various 3D representations have been explored for reconstruction, each with distinct advantages: mesh-based methods [13,32,3] offer compact representation but struggle with topological complexity; voxel-based approaches [15] provide a regular structure but face resolution limitations; point cloud methods [20,34,16] offer flexibility with additional rendering requirements; implicit functions [23,10,5,12] enable high-quality rendering but are computationally intensive; and Gaussian splatting techniques [31,30,22] balance quality and efficiency.
Point cloud generative models have evolved from early GAN-based [1,9,28] and VAE-based [36] approaches to more recent diffusion-based methods. Diffusion models offer several advantages: stable training dynamics, high-quality generation capabilities, flexibility in conditioning, and a strong probabilistic foundation. The seminal work by Luo et al. [19] introduced diffusion models for point cloud generation, with subsequent research extending these methods for various applications [38,17].
Building on these advancements in 3D generative modeling, recent diffusion-based approaches have significantly advanced image-to-point cloud reconstruction. PC 2 [20] performs single-view reconstruction by denoising a point cloud with projection conditioning, which ensures geometric consistency between the reconstruction and input view. However, it directly learns the posterior distribution, requiring images during training and limiting adaptability to varying input conditions. Bayesian Diffusion Models (BDM) [34] offer a complementary perspective by factorizing the 3D reconstruction task into a learned score of the prior ∇ log p(X) trained solely on 3D shapes and a learned score of the posterior ∇ log p(X|y) trained with paired image-shape data. During inference, the prior and posterior models exchange intermediate outputs over multiple denoising steps. While this "fusion-with-diffusion" paradigm is effective, BDM relies on a PC 2 -like trained posterior score function that requires images during training, thus limiting its adaptability to varying input modalities.
this section cite: ['b10', 'b13', 'b18', 'b38', 'b37', 'b23', 'b19', 'b20', 'b33', 'b19', 'b33', 'b19', 'b7', 'b21', 'b5', 'b21', 'b12', 'b31', 'b2', 'b14', 'b19', 'b33', 'b15', 'b22', 'b9', 'b4', 'b11', 'b30', 'b29', 'b21', 'b0', 'b8', 'b27', 'b35', 'b18', 'b37', 'b16', 'b19', 'b33']

Section: Diffusion Posterior Sampling.
Diffusion Posterior Sampling (DPS) [6] proposes a framework for solving inverse problems using diffusion models without retraining for each new measurement type. This approach decomposes the posterior p(X|y) into a pre-trained prior p(X) and an adaptable likelihood term p(y|X). During sampling, the intermediate predictions are adjusted using gradient updates from the likelihood term.
Recent works applying DPS to 3D reconstruction include GSD [22], which uses DPS with Gaussian Splatting for view-guided 3D generation. However, these methods rely on heuristic, manually-tuned step sizes for the likelihood update, which often requires careful calibration for each task and can lead to suboptimal convergence or reconstruction quality.
Sampling Phase FCM Likelihood Update Point Cloud Diffusion Model 𝜖 𝜃 (𝐗 𝑡 ) 𝑿 𝑡 𝐗: Point Cloud 𝐲: Reference Image ℛ(⋅):Rendering Operator FCM ෩ 𝐗 0|𝑡 𝐗 𝑡-1 𝐲 DDIM sampling 𝐗 0|𝑡 ℛ(𝐗) 𝐲 -2 ℒ(𝐗) = 𝛿 = 𝛿 0 ෩ 𝐗 0|𝑡 2 𝒈 𝑘 2 𝒈 𝑘 = ∇ℒ ෩ 𝐗 0|𝑡 𝑘 𝒈′ 𝑘 = ∇ℒ( ෩ 𝐗 0|𝑡 𝑘 -𝛿𝒈 𝑘 ) 𝛼 𝑘 𝑟𝑎𝑤 = 𝒈 𝑘 2 2 𝒈 𝑘 ⊤ 𝒉 𝑘 + 𝜖 ෩ 𝐗 0|𝑡 𝑘+1 = ෩ 𝐗 0|𝑡 𝑘 -𝛼 𝑘 𝒈 𝑘 ෩ 𝐗 0|𝑡 𝑘+1 = ෩ 𝐗 0|𝑡 𝑘 -𝛼𝑘 2 𝒈 𝑘 If, ℒ(෩ 𝐗 0|𝑡 𝑘+1 )> ℒ ෩ 𝐗0|𝑡 -𝜂𝐹𝐶𝑀𝛼𝑘𝒈𝑘 𝒈 𝑘+1 = ∇ℒ( ෩ 𝐗 0|𝑡 𝑘+1 ) 𝑘 ← 𝑘 + 1 𝒉 𝑘 = 𝒈 𝑘 -𝒈′ 𝑘 𝛿 Else,
this section cite: ['b5', 'b21']

Section: Adaptive Step Size Methods in Optimization.
Our FCM approach has roots in several foundational optimization techniques while introducing novel algorithmic elements. In numerical optimization, determining appropriate step sizes is a well-studied challenge with various classical solutions. Quasi-Newton methods [25] approximate the Hessian using rank-one or rank-two updates (e.g., BFGS, L-BFGS [18]), but require matrix storage and operations. Barzilai-Borwein (BB) methods [4] provide scalar approximations to the secant equation using the differences of consecutive iterates and gradients. Line search techniques with Armijo [2] or Wolfe conditions [33] ensure sufficient descent but typically involve multiple function evaluations.
Building on these foundations, FCM introduces several innovations specifically for diffusion-based 3D reconstruction: (1) a scale-adaptive curvature probe (δ k = δ 0 • ∥x k ∥ ∥g k ∥ ) that automatically calibrates to the geometry of point clouds and gradient magnitudes, (2) a forward-difference directional curvature estimate that requires no adjoint operations of the renderer-critical for complex neural renderers where adjoint computation is intractable, (3) a robust BB-inspired step-size computation combined with principled capping that offers theoretical guarantees, and (4) a "once-only" Armijo check.
this section cite: ['b24', 'b17', 'b3', 'b1', 'b32']

Section: Method
Our goal is to perform high-quality, flexible 3D reconstruction by decomposing the posterior distribution p(X | y) into a learned prior p θ (X) and a likelihood update p(y | X) that does not require separate training. We train only the score of the prior ∇ log p θ (X) on unlabeled 3D data. Then, at inference, we incorporate the measurement information (e.g., single-view or multi-view images, depth maps) through an adaptive Forward Curvature-Matching (FCM) update, which approximates ∇ log p(y | X).
Any forward operator R (e.g., a differentiable renderer for images or a map from 3D to depth measurements) can be plugged in to guide the generation of point clouds via the same trained diffusion prior. This design separates the learned model from the measurement modality, eliminating the need for retraining whenever the measurement operator changes. In this section, we detail our method in four parts. First, we describe how we train the diffusion model ∇ log p θ (X). Next, we present our differentiable renderer R for the image-based scenario. We then introduce the FCM-based likelihood update, highlighting why FCM is needed in non-linear settings and how step sizes are optimally determined through a principled approach. Finally, we extend the method to the multi-view setting.
this section cite: []

Section: Diffusion Prior for Point Clouds
We begin by training a diffusion model p θ (X) on a large dataset of colored point clouds. Following the standard DDPM [11] framework, we define a forward diffusion process that corrupts a clean point cloud X 0 into X T with Gaussian noise over T timesteps. The reverse process is modeled by a neural network that estimates the noise at each timestep. Formally, in the forward process:
q(X t | X t-1 ) = N X t ; 1 -β t X t-1 , β t I ,(1)
where β t is a variance schedule. This process can be written in closed form from
X 0 : q(X t | X 0 ) = N X t ; √ ᾱt X 0 , (1 -ᾱt ) I ,(2)
with ᾱt = t s=1 (1 -β s ). The reverse process approximates p θ (X t-1 | X t ) via a learned Gaussian:
p θ (X t-1 | X t ) = N X t-1 ; µ θ (X t , t), Σ θ (X t , t) .(3)
During training, we minimize the simplified loss:
L = E t,X0,ϵ ∥ϵ -ϵ θ (X t , t)∥ 2 ,(4)
where ϵ ∼ N (0, I).
Once trained, we use the DDIM sampler [29] for inference, generating point clouds from noise in fewer steps. Let ϵ (t) θ (X t ) be the noise estimate at step t. Then the DDIM update from X t to X t-1 is:
Xt-1 = √ ᾱt-1 X0|t + 1 -ᾱt-1 -σt(η) 2 ϵ (t) θ (Xt) + σt(η) ϵt,(5)
where
X0|t = X t - √ 1 -ᾱt ϵ (t) θ (X t ) √ ᾱt and σ t (η) = η (1-ᾱt-1) (1-ᾱt)
1 -ᾱt ᾱt-1 , is a variance term controlling the sampling stochasticity. This DDIM sampler, combined with our trained model, provides a 3D prior that can generate plausible point clouds.
this section cite: ['b10', 'b28']

Section: Differentiable Renderer as the Measurement Operator
Our method only requires that R be differentiable, so both R(X) and its gradient ∇ X ∥y -R(X)∥ 2 can be computed. In this section we introduce a forward operator R that projects a point cloud X into 2D measurements.
A point cloud X comprises points {(x i , y i , z i , f i )}, where (x i , y i , z i ) are 3D coordinates and f i includes attributes such as color. Each point is projected onto the 2D image plane using known camera parameters. At each pixel (u, v), R collects the K points with the smallest depth values z i (i.e., the nearest points along the viewing direction) and blends their colors via alpha compositing:
R color (X)[u, v] = K i=1 α i i-1 j=1 (1 -α j ) f i .(6)
Here, the opacity α i is computed from the image space footprint as
α i = 1 - ρ 2 i r 2 , (7
)
where r is the radius of the rasterizer and ρ i is the Euclidean distance between the center of the pixel and the projected position of the point in the image space. The product term i-1 j=1 (1 -α j ) ensures that closer points dominate the final color, while partially occluded points contribute less. Repeating this calculation for each pixel (u, v) yields a 2D image matching the resolution of y.
In addition to color-based rendering, R can produce a depth map by applying inverse-square weighting to each point's distance. At each pixel (u, v), the depth is computed from the same set of K nearest points: Category EMD(×10) CD(×10) F-score PC 2 [20] BDM [34] Ours PC 2 [20] BDM [34] Ours PC 2 [20] BDM [34] Ours airplane 0.587 0.577 0.476 0.399 0.417 0.378 0.498 0.543 0.543 car 0.565 0.723 0.517 0.558 0.664 0.460 0.262 0.289 0.386 chair 0.701 0.643 0.662 0.636 0.613 0.679 0.241 0.271 0.282 table 0.735 0.647 0.691 0.703 0.656 0.727 0.240 0.268 0.319 Average 0.647 0.648 0.587 0.574 0.588 0.561 0.310 0.343 0.382 Table 1: Quantitative evaluation of single-view 3D reconstruction on the ShapeNet dataset. NFEs were matched equally across our method, PC 2 , and reconstruction model of BDM (T = 256). For BDM, additional NFEs were incurred due to the prior model (T = 20).
R depth (X)[u, v] = K i=1 1 di K i=1 1 d 2 i ,(8)
so that points closer to the camera have a larger influence on the final depth. Repeating this process for each pixel yields a 2D depth map matching the resolution of y.
Because we do not learn a dedicated score function for ∇ log p(y | X), different operators R can be swapped in with minimal effort. If y is a single-view image, then R = R color with a single camera. For multi-view input, each view is rendered separately and their pixel or feature errors are averaged, as described in Section 3.4. If y is a depth map, then R = R depth from Eq. ( 8).
this section cite: []

Section: Likelihood Update via Forward Curvature-Matching
In standard diffusion posterior sampling (DPS) [6], one iteratively updates the current sample X t with a term proportional to the gradient ∇ X log p(y | X). However, for complex, non-linear forward operators R, determining an appropriate step size is non-trivial. Previous approaches resort to heuristics [6] or empirically tuned factors [22] to balance the data fidelity term with the learned diffusion prior. While this can be effective, it may hamper convergence speed or degrade reconstruction quality if not carefully tuned.
To address these limitations, we propose Forward Curvature-Matching (FCM), a novel algorithm designed specifically for diffusion-based 3D reconstruction. The development of FCM was guided by key requirements: working without adjoint operations (intractable for neural renderers), maintaining predictable computational cost, and using universal parameters across different reconstruction tasks.
Our approach relies on a key insight: we can estimate curvature information through a scaled directional probe without requiring full Hessian approximations [25]. For the measurement loss L(x) = ∥y -R(x)∥ 2 , given the current estimate x k and gradient g k = ∇L(x k ), we compute: The figure shows renderings from our method and PC 2 , illustrating the higher fidelity and better preservation of details in our reconstructions.
δ k = δ 0 • ∥x k ∥ ∥g k ∥ , x ′ k = x k -δ k • g k ,(9)
g ′ k = ∇L(x ′ k ), h k = g k -g ′ k δ k (10
)
This h k approximates ∇ 2 L(x k ) • g k along the gradient direction. The scale-adaptive probe (δ k ) automatically calibrates to the geometry of the point cloud, a crucial advantage over traditional finite-difference approaches [7].
We then compute a Barzilai-Borwein-inspired [4] step size, modified for robustness:
α raw k = ∥g k ∥ 2 ⟨g k , h k ⟩ + ε , α k = min{α raw k , 1/L} (11
)
where ε = 10 -12 and L is the Lipschitz constant of ∇L. The capping mechanism ensures stability while maintaining theoretical guarantees. Unlike classical line searches that require multiple function evaluations [25], we incorporate a single Armijo check [2]:
if L(x k -α k g k ) > L(x k ) -η FCM • α k • ∥g k ∥ 2
, we halve α k once and accept.
This design yields a fixed computational cost of exactly two backward and three forward passes per step-significantly more efficient than traditional optimization methods like L-BFGS [18] or Wolfe line searches [33] with unpredictable evaluation counts.
this section cite: ['b5', 'b5', 'b21', 'b24', 'b6', 'b3', 'b24', 'b1', 'b17', 'b32']

Section: Theoretical Guarantees
Our approach is built on the following assumptions, which are typically satisfied in the context of 3D reconstruction:
Assumption 3.1 (Smoothness). L is L-smooth: ∥∇L(u) -∇L(v)∥ ≤ L • ∥u -v∥. Assumption 3.2 (Lower bound). L inf := inf x L(x) > -∞.
this section cite: []

Section: Assumption 3.3 (Local convexity).
L is convex on the set of iterates (which is typically small or "benign" in practice).
This approach provides theoretical guarantees on convergence and optimality, as captured in the following theorem: Theorem 3.4 (Guaranteed Loss Decrease). Let c = min{ ηFCM 2L , 1 8L }. Our FCM algorithm ensures:
L(x k+1 ) ≤ L(x k ) -c • ∥∇L(x k )∥ 2(12)
When integrated into the DDIM sampling process, FCM preserves the contraction properties of diffusion models: Ground Truth (Novel View) 5 Views 1 View 3 Views Figure 6: Qualitative results of multi-view reconstruction. The figure displays point cloud reconstructions using varying numbers of input views, demonstrating the enhancement in reconstruction quality as more views are incorporated. Proposition 3.5 (Contraction Preservation). Under Assumptions 3.1-3.3 and α k ≤ 1/L, the combined DDIM+FCM operator remains a contraction in expectation, thus preserving the diffusion contraction property.
Our FCM method uses fixed constant η FCM = 10 -4 for all tasks, this principled approach leads to faster convergence and higher-quality reconstructions compared to methods that rely on heuristic step sizes. Detailed proofs and additional theoretical analysis are provided in the Appendix.
this section cite: []

Section: Multi-View Reconstruction
The same FCM-based likelihood update extends naturally to multi-view reconstruction. Suppose we have N images {y i } N i=1 with known camera parameters. We define
L MV (X) = 1 N N i=1 y i -R i (X) 2 ,(13)
where R i is the differentiable renderer for the i-th viewpoint. The gradient ∇ X L MV (X) can be used in Algorithm 1 (replacing the single-view line ∥y -R(•)∥ with the multi-view average). As the number of views grows, reconstruction quality improves, yet the diffusion prior remains the same, illustrating the modality-agnostic nature of our approach.
By training only the diffusion prior on unlabeled 3D shapes and introducing an FCM-based likelihood update with an arbitrary forward operator R, we achieve a flexible, adaptive 3D reconstruction pipeline. The FCM approach ensures stable and fast convergence even with non-linear rendering operators, outperforming fixed-step DPS approaches.
this section cite: []

Section: Experiments
We evaluate the reconstructed point clouds using three different metrics: Earth Mover's Distance (EMD), L-1 Chamfer Distance (CD), and F-score at a threshold of 0.01. Details of the implementation are provided in the appendix.
this section cite: []

Section: ShapeNet.
In our method, colors are essential during the rendering process. However, sampling colored point clouds from mesh-based objects is a challenging task. To address this, we train our model using the dataset provided by KeypointNet [37]. The color information in the KeypointNet point cloud does not correspond to the actual mesh color in ShapeNet. Instead, the model assigns colors according to object parts.
We perform our evaluation using the categories {airplane, car, chair, table} from the ShapeNet rendered image dataset [35]. Views EMD(×10) CD(×10) F-score 1 0.587 0.561 0.382 3 0.436 0.386 0.512 5 0.425 0.361 0.548 Table 3: Comparison with DPS-based methods. The table presents reconstruction metrics for our method versus DDPM+DPS and DDIM+DPS, demonstrating our approach's superior performance with fewer NFEs.
CO3D. The CO3D dataset is a large-scale collection of real-world multi-view images from common object categories. It provides a colored point cloud obtained using COLMAP from multi-view images, which is then used for model training and evaluation. We perform our evaluation using the categories hydrant and teddybear from the CO3D dataset.
this section cite: ['b36', 'b34']

Section: Quantitative Results
We evaluate the performance of reconstruction in the ShapeNet dataset. In Tab. 1, our method is compared with PC 2 [20] and BDM [34]. In the original paper, BDM is evaluated using 4,096 points, whereas PC 2 is evaluated using 8,192 points. In this work, we adopt the evaluation approach of PC 2 for quantitative experiments. For BDM, we adopted the blending method that achieved the best results in their study and used PC 2 as the reconstruction model. Our method achieves the best results in all metrics. In this experiment, we ensured that the NFEs for all other models were set similarly for a fair comparison. Detailed comparisons with the settings proposed by their studies and quantitative results on CO3D are provided in the appendix.
this section cite: ['b19', 'b33']

Section: Qualitative Results
In Fig. 3 we show the reconstructed point clouds of different models using the ShapeNet dataset. In Fig. 4 we present a comparison of the rendered results of reconstructed colored point clouds using the CO3D dataset. Other models fail to accurately follow the given image in their rendering results for the reference view, instead focusing on generating a plausible object within the learned category. However, our method achieves the highest level of detail for the reference image.
this section cite: []

Section: Adaptivity Analysis
Our method has the advantage of performing various tasks without requiring retraining of the model.
In this section, we demonstrate this capability through multi-view reconstruction and depth map reconstruction. The models used in this section are the same as those used in the previous section for the ShapeNet dataset. Fig. 6 and Tab. 2 illustrate the effectiveness of our method in multi-view reconstruction. As the number of views increases, the generated point cloud becomes more refined, demonstrating the improved quality of reconstruction. Fig. 5 presents the results of applying our method to depth maps rendered using Eq. 8. The results show high fidelity to the reference depth map and the ability to generate natural-looking objects.
this section cite: []

Section: Ablation Study
To show the effectiveness of our method, we compare with other DPS-based methods. Fig. 7 and Tab. 3 compare our method with DPS-based approaches. Fig. 7 presents the plot of the difference in L2 norm between the reference image and the rendered image during the sampling process over timesteps. We observed that both DDPM+DPS and DDIM+DPS methods achieve their best performance at the step size of 0.05. The reason DPS-based methods struggle to follow the reference image is that they update with a fixed step size, leading to suboptimal convergence. It demonstrates that our method converges more optimally compared to other approaches. As shown in Tab. 3, our method achieves the best point cloud reconstruction performance. Since the DDPM sampling process does not approximate X0 , and the iterative FCM updates from noisy X t using measurement y are not ideal, we exclude the DDPM+FCM scheme from our comparison. Qualitative comparisons with DPS-based methods are provided in the appendix.
this section cite: []

Section: Conclusion
In this paper, we proposed the novel point cloud diffusion sampling approach for adaptive 3D reconstruction. Our method reconstructs the colored point cloud by updating it using likelihood ∇ log p(y | X) with given images through FCM during the reverse process of the point cloud diffusion model. In our experiments, we qualitatively demonstrate high-fidelity reconstruction of reference images with color, generating high-quality point cloud structures compared to prior works. Moreover, we quantitatively surpass previous works in point cloud reconstruction performance. Our method is applicable to various tasks, demonstrating its versatility. Additionally, it can be extended to different domains (e.g., Gaussian Splatting, meshes, etc.), highlighting its adaptability. As future work, we are interested in exploring larger datasets across diverse domains.
this section cite: []

Section: References
Ref_id:b0 Title: Learning representations and generative models for 3D point clouds Year: (2017)
Ref_id:b1 Title: Minimization of functions having Lipschitz continuous first partial derivatives Year: (1966)
Ref_id:b2 Title: LIST: Learning implicitly from spatial transformers for single-view 3D reconstruction Year: (2023)
Ref_id:b3 Title: Two-point step size gradient methods Year: (1988)
Ref_id:b4 Title: Single-stage diffusion NeRF: A unified approach to 3D generation and reconstruction Year: (2023)
Ref_id:b5 Title: Diffusion posterior sampling for general noisy inverse problems Year: (2023)
Ref_id:b6 Title: Numerical methods for unconstrained optimization and nonlinear equations Year: (1996)
Ref_id:b7 Title: DiffPoint: Single and multi-view point cloud reconstruction with ViT based diffusion model Year: (2024)
Ref_id:b8 Title: A Papier-Mache approach to learning 3D surface generation Year: (2018)
Ref_id:b9 Title: NerfDiff: Single-image view synthesis with NeRF-guided distillation from 3D-aware diffusion Year: (2023)
Ref_id:b10 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b11 Title: LRM: Large reconstruction model for single image to 3D Year: (2024)
Ref_id:b12 Title: Self-supervised 3D mesh reconstruction from single images Year: (2021)
Ref_id:b13 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b14 Title: Image-to-voxel model translation for 3D scene reconstruction and segmentation Year: (2020)
Ref_id:b15 Title: RGB2Point: 3D point cloud generation from single RGB images Year: (2025)
Ref_id:b16 Title: Enhancing diffusion-based point cloud generation with smoothness constraint Year: (2024)
Ref_id:b17 Title: On the limited memory BFGS method for large scale optimization Year: (1989)
Ref_id:b18 Title: Diffusion probabilistic models for 3D point cloud generation Year: (2021)
Ref_id:b19 Title: Projection-conditioned point cloud diffusion for single-image 3D reconstruction Year: (2023)
Ref_id:b20 Title: DiT-3D: Exploring plain diffusion transformers for 3D shape generation Year: (2023)
Ref_id:b21 Title: GSD: View-guided Gaussian splatting diffusion for 3D reconstruction Year: (2024)
Ref_id:b22 Title: DiffRF: Rendering-guided 3D radiance field diffusion Year: (2023)
Ref_id:b23 Title: Point-E: A system for generating 3D point clouds from complex prompts Year: (2022)
Ref_id:b24 Title: Numerical optimization Year: (2006)
Ref_id:b25 Title: Accelerating 3D deep learning with PyTorch3D Year: (2020)
Ref_id:b26 Title: RepKPU: Point cloud upsampling with kernel point representation and deformation Year: (2024)
Ref_id:b27 Title: Sung Woo Park, and Junseok Kwon. 3D point cloud generative adversarial network based on tree structured graph convolutions Year: (2019)
Ref_id:b28 Title: Denoising diffusion implicit models Year: (2021)
Ref_id:b29 Title: Splatter image: Ultra-fast single-view 3D reconstruction Year: (2024)
Ref_id:b30 Title: LGM: Large multi-view Gaussian model for high-resolution 3D content creation Year: (2024)
Ref_id:b31 Title: Pixel2Mesh: Generating 3D mesh models from single RGB images Year: (2018)
Ref_id:b32 Title: Convergence conditions for ascent methods Year: (1971)
Ref_id:b33 Title: Bayesian diffusion models for 3D shape reconstruction Year: (2024)
Ref_id:b34 Title: DISN: Deep implicit surface network for high-quality single-view 3D reconstruction Year: (2019)
Ref_id:b35 Title: Point-Flow: 3D point cloud generation with continuous normalizing flows Year: (2019)
Ref_id:b36 Title: KeypointNet: A large-scale 3D keypoint dataset aggregated from numerous human annotations Year: (2020)
Ref_id:b37 Title: LION: Latent point diffusion models for 3D shape generation Year: (2022)
Ref_id:b38 Title: 3D shape generation and completion through point-voxel diffusion Year: (2021)
Ref_id:b39 Title: Reference Image Reference View Novel Views Figure 21: Analysis of failure cases in single-view reconstruction on CO3D Year: ()
