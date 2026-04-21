Title: Repurposing Marigold for Zero-Shot Metric Depth Estimation via Defocus Blur Cues
Abstract: Recent monocular metric depth estimation (MMDE) methods have made notable progress towards zero-shot generalization. However, they still exhibit a significant performance drop on out-of-distribution datasets. We address this limitation by injecting defocus blur cues at inference time into Marigold, a pretrained diffusion model for zero-shot, scale-invariant monocular depth estimation (MDE). Our method effectively turns Marigold into a metric depth predictor in a training-free manner. To incorporate defocus cues, we capture two images with a small and a large aperture from the same viewpoint. To recover metric depth, we then optimize the metric depth scaling parameters and the noise latents of Marigold at inference time using gradients from a loss function based on the defocus-blur image formation model. We compare our method against existing state-of-the-art zero-shot MMDE methods on a self-collected real dataset, showing quantitative and qualitative improvements. Our implementation is available at https://github.com/chinmay0301ucsd/DiffusionCam.

Section: Introduction
Estimating metric depth from a single camera viewpoint is a central problem in computer vision with numerous downstream applications, including 3D reconstruction [24], autonomous driving [50], and endoscopy [31]. This task, known as monocular metric depth estimation (MMDE), is fundamentally ill-posed due to inherent depth-scale ambiguity [46]. Multi-view methods [65] avoid this ambiguity but are often expensive and impractical in settings like endoscopy or microscopy. Training data-driven MMDE methods is challenging, as it requires accounting for a diverse set of camera parameters and metric depth scales. As a result, existing MMDE models struggle in zero-shot settings, i.e., they generalize poorly to unseen datasets. Recent advances in zero-shot MMDE [75] have demonstrated improved generalization, but there is still a considerable performance drop on unseen datasets.
In contrast to MMDE, monocular relative depth estimation (MDE) methods recover a relative depth map, factoring out the physical depth scale. This enables using large-scale datasets with diverse depth ranges [41] for training data-driven MDE methods. As a result, MDE methods achieve better zero-shot generalization at significantly lower training cost than MMDE methods, as shown by recent results [70,71,26]. However, despite favorable performance on benchmarks, the absence of metric scale in MDE outputs precludes their applicability in downstream tasks requiring absolute depth.
Existing data-driven MMDE methods commonly suffer from two failure modes: undesirable coupling between image texture and depth predictions (fig. 3), and inaccurate estimation of the scene's physical scale (see fig. 4). The first issue of texture coupling also affects MDE methods (see fig. 3) unless explicitly mitigated through complex training procedures that account for texture variation [22,55].
Current state-of-the-art generative MDE/MMDE methods [47,13] are predominantly diffusion-based. Several previous methods [13,47,32] incorporate diffusion-based depth denoising in their pipelines, achieving highly detailed depth maps -but are not zero-shot. [48] achieves zero-shot MMDE with a diffusion-based approach by incorporating diverse field of view (FOV) augmentations in training, but it is not open source and lags behind transformer-based methods in performance. Marigold [26] is trained by fine-tuning Stable Diffusion-v2 on synthetic depth data. It achieves high-quality zero-shot MDE and supports test time refinement, but is not applicable natively for MMDE. Our approach uses defocus cues to refine the relative depth predictions from Marigold (or similar methods [16]), enabling its application to MMDE. Prior work [60] also proposes a similar strategy for dense MMDE from a sparse metric depth map using Marigold and test time optimization. In contrast, our method does not require a sparse depth map as input, and solely relies on RGB images and a priori known scene bounds. Using defocus blur cues, our method resolves inaccuracies in monocular depth while also estimating the global scaling parameters for metric depth.
this section cite: ['b24', 'b50', 'b31', 'b46', 'b65', 'b75', 'b41', 'b70', 'b71', 'b26', 'b22', 'b55', 'b47', 'b13', 'b13', 'b47', 'b32', 'b48', 'b26', 'b16', 'b60']

Section: Diffusion model priors for inverse problems
We frame MMDE as an inverse problem under the defocus blur image formation model. This framing closely relates to the recent work on solving linear inverse problems using pre-trained diffusion models [12]. [9,11,54,10] incorporate the forward model constraints while sampling pixel-space diffusion models pre-trained on smaller datasets. As a result, these methods require many steps while sampling the diffusion model and have limited generalizability as priors. [53,44] use latent variable diffusion models (LDMs) as priors, resulting in better in-the-wild generalizability. We use Marigold as the LDM, but instead of incorporating the defocus forward model during sampling, we optimize the latent noise vector based on the error between the observed and predicted image using the forward model. Our approach is inspired by recent methods [37,36,77,61,17] which uses noise optimization in conjunction with a differentiable auxiliary guidance loss to improve the sampling quality of the diffusion model based on text input. While these methods use trained models as differentiable proxies for guidance, we use a physics-based imaging forward model. [34] also uses a physics-based forward model with a diffusion prior, but requires re-training the diffusion prior from scratch.
Depth estimation from passive camera physics cues A substantial body of research on MMDE leverages camera physics, including methods like depth-from-defocus (DfD), - [64,20,58,2], phase/aperture masks [28,78,76,3,67], and dual pixel sensors [18,68,1]. Classical approaches produce very coarse depth maps. Most DfD methods need a well-aligned [66] multi-image focal stack [29,57], to achieve good depth quality. Optical mask-based methods pose a harder inverse problem of jointly estimating both AIF and the depth map. We capture only 2 images at the same focus distance and different apertures, requiring no image alignment and AIF estimation. This simplifies depth refinement with minimal added capture time. Previous work has also used variable apertures for classical [15] and learning-based [56] depth estimation methods, but it is not zero-shot. While learning-based approaches [19,8,66,18] help improve the depth map quality for these methods, they don't generalize well to out-of-distribution scenes. Dual pixel-based methods are popular for phone cameras, [18,38], but are tied to the specific camera architecture. While we show results with a standard DSLR sensor, our approach can be adapted to dual-pixel-style camera architectures as well.
this section cite: ['b12', 'b9', 'b11', 'b54', 'b10', 'b53', 'b44', 'b37', 'b36', 'b77', 'b61', 'b17', 'b34', 'b64', 'b20', 'b58', 'b1', 'b28', 'b78', 'b76', 'b3', 'b67', 'b18', 'b68', 'b0', 'b66', 'b29', 'b57', 'b15', 'b56', 'b19', 'b8', 'b66', 'b18', 'b18', 'b38']

Section: Preliminaries
Diffusion-based monocular depth estimation Our approach is built on top of Marigold [26], which is a monocular depth estimator trained by fine-tuning the denoising U-Net of StableDiffusion-v2 (SDv2) [43] on synthetic depth data. Marigold allows sampling the conditional distribution of the monocular depth given an input image, p(d 0 |x). In particular, Marigold attempts to generate clean monocular depth maps, d 0 given a clean input image x by first sampling d T ≈ N (0, I) from an i.i.d Gaussian distribution, and then iteratively denoising it (where each intermediate step is denoted by d t ) according to a fixed noise schedule with parameters α t , σ t . As SDv2 is a latent-diffusion model, the entire generative process happens on encoded latent depth maps z (d) 0 with latent images z (x) as conditioning. Training Marigoldfoot_1 (which we denote as xϕ (•)) involves using the standard denoising loss common in image diffusion works, but on depth maps rather than images, and passes the image input into the model as well through channel-wise concatenation:
E z (d) 0 ,z (x) ∼D,t∼p(t),ϵ∼N (0,I) [w(t)∥z (d) 0 -xϕ (α t z (d) 0 + σ t ϵ, z (x) , t)∥ 2 2 ],(1)
where w(t) is a time-dependent weighting function. At inference time, x is first mapped to a lower dimensional latent vector z (x) = E(x) through VAE encoder E in SDv2. The inference process starts with sampling a noisy latent depth vector z
T ∼ N (0, I), which is iteratively refined by applying the denoiser xϕ (z
(d) t , z (x) , t) to obtain z (d) 0 over T sampling steps. z (d) 0 is then decoded through the SD decoder D to produce the output depth map d = D(z (d) 0 ). Marigold assumes both z (x) , z (d) ∈ R M .
For faster inference, we use the latent consistency model version of Marigold, Marigold-LCM, and obtain z (d) 0 with a single inference step from z (d) T . Monocular depth d can be mapped to metric depth d m by an affine transform; more details in section 4.
this section cite: ['b26', 'b43']

Section: Modeling defocus cues
Under the thin lens camera assumption, defocus blur manifests as a uniform blur kernel, with a depth-dependent diameter. The blur kernel diameter for a point source placed at a distance d away from the camera is determined by the circle of confusion (CoC) [40] equation
c(d) = f 2 N |d -F | d(F -f )s ,(2)
where the focal length f , focus distance F , and F-stop N are camera parameters that are known from the image EXIF data. s denotes the pixel size in physical units (m). The CoC defines a depth-dependent point spread function (PSF) h(i, j | u, v, d) that predicts the response at pixel coordinate (i, j) from a point source at lateral coordinate (u, v) and depth d under defocus blur. Note that (u, v) can represent depth-normalized pixel coordinates under an ideal pinhole projection, allowing us to write d = d m [u, v] (valid for general non-volumetric scenes that assume a single depth value per coordinate). We assume that the PSF is shift-invariant for a given depth d, i.e.
h(i, j | u, v, d) = h(i -u, j -v | 0, 0, d m [u, v]) = h(i -u, j -v | d m [u, v]
), which is simply the on-axis PSF, modeled as a disc (assuming circular aperture) with radius given by CoC equations, translated to be centered at (i, j). To ensure smooth optimization, we include a linear fall-off at the boundary of the discontinuous disc kernel, similar to [62]. The PSF can then be expressed as
h(i, j | u, v, d) = W (i, j | u, v, d) i,j W (i, j | u, v, d) , where(3)
W (i, j | u, v, d) =      1, m ≤ c(d)-1 2 c(d)+1 2 -m, c(d)-1 2 < m ≤ c(d)+1 2 0, c(d)+1 2 > m ,(4)
m = (i -u) 2 + (j -v) 2 .(5)
We assume the PSF to be normalized and explicitly account for exposure and energy balancing during image capture and processing (see section 4). While the above PSF can also be approximated as an isotropic 2D Gaussian [20], we opt for the disc parameterization in eq. ( 5) similar to [62,51], as it better approximates the PSF of the real camera compared to the Gaussian approximation in [20], as shown in fig. 2. We can see from eq. ( 2) that an image captured with a very small aperture (high N ) would have negligible defocus blur due to very small CoC values. Such an image is referred to as the all-in-focus (AIF) image, x. Given the AIF x, the blurred image x b can be approximated (neglecting occlusion) as a spatially varying convolution between h and x,
x b (i, j) = x(u, v) • h(i -u, j -v; d m [u, v])dudv(6)
For simplicity, we denote the above image formation forward model as
x b = g(x, d m , f, F, N ).(7)
The AIF image x, captured at a high F-stop, serves as the blur-free input for both Marigold and the camera blur model eq. ( 7), while the low F-stop image x b provides defocus cues for MMDE.
Inference-time optimization Without loss of generality, any generative model (GAN, Diffusion or flow-based) ϕ : R M → R N can be construed as a mechanism to map a simple probability distribution, such as an i.i.d. Gaussian distribution, ε ∼ N (0, I) ∈ R M to a non-linear N -dimensional manifold, such as images or audio, through a differentiable generative process x = ϕ(ε). Inference time optimization [37] refers to manipulating the generation process by updating the initial noise ε based on gradients from a differentiable loss function L(x) on the generated sample x,
ϵ → ϵ -∇ ϵ L(x).(8)
To further ensure that ϵ still lies close to the Gaussian manifold after the gradient updates, ε can be rescaled to have a L2 norm of √ M as in [45], which is a valid approximation for samples drawn from a high dimensional Gaussian distribution as per the Gaussian annulus theorem [4]. This holds for most generative models, as the initial noise vectors are typically high-dimensional (M > 50). In our case, ε corresponds to the noise latent z (d) T in Marigold, which we optimize using a loss function (eq. ( 11)) governed by the defocus blur forward model eq. ( 7).
this section cite: ['b40', 'b62', 'b20', 'b62', 'b51', 'b20', 'b37', 'b45', 'b4']

Section: Method
We capture two images per scene: x, with F-stop (N aif = 22) and exposure time t aif , serves as the blur-free all-in-focus (AIF) image. A second image, x b , is captured at a lower F-stop (N b = 8) with exposure time t b , thereby providing strong depth-varying defocus cues. The forward model in eq. ( 7) assumes radiometrically linear images (no gamma correction or non-linear processing) and energy constancy between the AIF and blurred images. We use raw images to satisfy these assumptions. Since total captured energy scales with exposure time (t) and aperture area (∝ (f /N ) 2 ) [25], we scale x b by the factor taif tb •
N 2 b N 2 aif
to match the energy in x. Note that we vary the exposure time to ensure well-exposed measurements across F-stop settings, while fixing the camera gain.
We frame metric depth estimation as an inverse problem, with the defocus blur image formation process in eq. ( 6) as the forward model, and Marigold as the monocular depth prior. To obtain scale-invariant monocular depth d ∈ [0, 1], we use Marigold-LCM, which takes in as input the AIF x (encoded to z (x) ), and a learnable depth latent vector z (d) T ∼ N (0, I). A single inference step of Marigold-LCM gives us the denoised depth latent z
(d) 0 = xϕ z (d) T , z (x) , 1 , which is decoded to monocular depth d = D(z (d) 0
). The predicted monocular depth d ∈ [0, 1] is then mapped to metric depth by affine transforming d with a learnable metric scale (α) and offset (β) per scene, d m = α • d + β. To ensure that α, β remain bounded and differentiable, we parameterize them as α = s max • σ(a) and β = s min • σ(b), where σ(•) is the sigmoid function; a, b are unconstrained learnable parameters initialized to 0, and s max and s min are the upper and lower scene depth bounds, respectively, which we assume are known a priori (valid for indoor scenes). To summarize, the metric depth (d m ) can be expressed using the optimizable parameters a, b, z
T as:
d m = s max • σ(a) • D xϕ z (d) T , z (x) , 1 + s min • σ(b) := y a, b, z (d) T . (9
)
The optimized metric depth d m can then be recovered by solving:
d m = arg min d m =y a,b,z (d) T ||x b -g(x, d m , f, F, N )|| 2 2 (10
) subject to z (d) T 2 = √ M ,(11)
where g(•) denotes the defocus blur forward model (eq. ( 7)), x b and x are the captured blurred and AIF images, respectively. By optimizing the learnable parameters a, b, z
T , we incorporate defocus blur cues for both correct metric scale recovery (a, b) and refining the initial depth estimate by Marigold (z (d) T ). See fig. 1 for the overview of our method.
this section cite: ['b25']

Section: Motivating Synthetic Toy Examples
We demonstrate that our approach resolves texture-depth coupling, accurately recovering metric depth in a synthetic scene with a textured plane at constant depth. While such a plane may seem simple, distinguishing it from a flat 2D image/poster or an actual 3D scene is challenging when viewed from a single viewpoint. Data-driven methods are biased towards predicting depths that reflect surface variations even in the absence of true depth changes, i.e., their outputs are strongly texture coupled. However, supplementing the AIF image with a simulated blurred image provides defocus cues that help our method disambiguate a flat poster from a 3D scene, as demonstrated in fig. 3 using toy examples of textured planes with constant depth. While learning-based methods and initial Marigold outputs suffer from texture coupling and scale errors, our method corrects both, producing accurate, constant-depth maps that outperform all baselines.
this section cite: []

Section: Accelerating inference
We use the distilled latent consistency model version of Marigold (Marigold-LCM) to reduce the number of sampling steps significantly. We observe that a single sampling step suffices for our case, which significantly speeds up inference-time optimization [36,14] relative to the normal 20-50 inference steps [37,61] that Marigold [60] uses. We show an ablation study with more sampling steps in supplement.E. We also implement custom CUDA kernels for the Disc-PSF forward model. This provides a 2.5x speed up over the PyTorch implementation provided by [62] while being more memory efficient, allowing our method to scale to higher-resolution images. Using a single sampling step of Marigold-LCM allows us to compute gradients w.r.t z (d) T without gradient checkpointing as previously done in [37].
this section cite: ['b36', 'b14', 'b37', 'b61', 'b60', 'b62', 'b37']

Section: Optimization details
We run the optimization for 200 iterations, which takes roughly 3.5-4 minutes on an NVIDIA A-40 GPU with peak memory usage of 15 GB. We use the Adam optimizer with a learning rate of 1.5 × 10 -3 for z (d) T , 5 × 10 -3 for (a, b), and default values for optimizer parameters. Note that we use the same scene bounds s min = 1.49, s max = 3.5 for all the real scenes in our dataset. These values represent a conservative upper bound on the potential maximum scale and offset in the real dataset. Please see supplement.A for more details.
this section cite: []

Section: Experiments and Results

this section cite: []

Section: Dataset details
Our method requires 2 images captured with different apertures (but same viewpoint) to integrate defocus cues. Standard monocular depth datasets [49,35] typically capture in-focus images at a single aperture per scene, making them unsuitable for evaluating our method. While one could simulate defocused images with eq. ( 7) and RGBD data, this does not capture the model mismatch (occlusion, diffraction) in the physical image formation process. Therefore, to fairly evaluate our method, we construct a hardware capture setup, shown in fig. 2, comprising an Intel RealSense depth camera rigidly mounted to a DSLR camera (Canon EOS 5D Mark II). We use this system to collect a custom real-world dataset of 7 unique scenes, evaluating our method against learning-based MMDE baselines. We choose scenes with diverse subjects and depth profiles, placed within the operating depth range of RealSense (0.3-3.8m) to ensure accurate ground truth depth. Note that the CoC changes negligibly with depth beyond these distances, making defocus cues unreliable. For each scene, we capture images at 6 different apertures, f /4, f /8, f /11, f /13, f /16, and f /22, with the latter serving as the AIF image x. The blurred image, x b , is selected from the lower F-stop images (see fig. 5 for comparison of F-stop setting on depth map quality). As described in section 4, we try to maintain a similar ratio of the exposure time and the camera aperture area for all the measurements taken for a scene. The lens focal length (f ) and F-stop (N ) are provided by the camera EXIF data, and the focus distance F is read manually from the lens's analog focus scalefoot_2 . Please see supplement.B for camera parameters and other dataset details, and supplement.H for more discussion on ground truth depth map quality.
Evaluation Metrics We evaluate the predicted metric depth, dm , from our method against the RealSense ground truth depth d (with overloaded notation), using the metrics in [5,39]. Specifically, we compute absolute relative error i denote the RealSense and predicted depths at pixel i, respectively. We also report point-cloud-based metrics, i.e., Chamfer Distance (CD) and the aggregated F1 score (FA) as defined in [39]. We align the ground truth and predicted depth map, similar to [19], and compute the metrics for pixels with non-zero values across the aligned depth maps. See supplement.C for more details on the depth map alignment/calibration procedure.
(REL) = 1 M M i=1 |di-dm i | di , root mean squared error (RMSE) = 1 M M i=1 |d i -dm i | 2 , average log error (log10) = 1 M M i=1 | log 10 d i -log 10 dm i |,
this section cite: ['b49', 'b35', 'b5', 'b39', 'b39', 'b19']

Section: Quantitative and qualitative results
We evaluate our method against some of the recently popular MMDE methods UniDepth [39], Metric3D [74], and MLPro [7] on our collected dataset (table 2) and the NYUv2 [35] test set. Both Metric3D and UniDepth are provided with the required camera parameters as input. We outperform existing methods qualitatively (fig. 4) and quantitatively (table 2) on all the evaluation metrics, averaged over all 7 scenes in our collected dataset. Please see supplement.D for per-scene quantitative metrics on our dataset and supplement.H for depth error visualizations. On the NYUv2 test set, our method is on par with the MMDE baselines. See supplement.J for more details. For our dataset, the MMDE methods (MLPro, Metric3D) recover sharper details and are on-par with our method on some scenes (see fig. 4), but they lack consistency in their overall performance across all scenes. Our method achieves better consistency across varying scene conditions. Leveraging defocus cues enables our approach to recover the correct depth scale while also resolving relative depth errors in some cases (fig. 4 insets). We also evaluate our method (row 4 in table 2) using a Gaussian PSF [20] in the forward model. While previous work [20] uses the Gaussian PSF for training models for unsupervised depth recovery, we find that the model mismatch between the Gaussian PSF and the PSF of the real camera (fig. 2) leads to severe performance degradation in our case, compared to using the Disc PSF, which matches the real camera PSF better. This highlights the value of a physically consistent forward model, even with strong learned priors.
Comparing with fine-tuning based methods Our method relies on optimizing noise-latents of a relative depth foundation model (RDFM) such as Marigold or Geowizard at test-time. Our approach (inference-time optimization) can be valuable in zero-shot settings, where fine-tuning-based methods would struggle due to the domain gap. We validate this by comparing our method with MMDE models obtained by fine-tuning RDFMs (DepthAnythingv2 [71], Marigold) on Metric depth datasets with a restricted depth range, as required by our method (section 4). We evaluate the following fine-tuning-based baselines on our collected dataset in table 1 -DepthAnythingv2 [71] fine-tuned on the Hypersim [42] dataset (row 1), NYU-v2 train set (row 2), and Marigold Fine-Tuned on NYUv2 (row 3). Our method (row 4, table 1) outperforms the fine-tuning-based baselines. This validates the utility of our inference-time method in zero-shot settings, where fine-tuning-based methods can struggle due to a domain gap. Please see supplement.K for details on fine-tuning these baselines and additional results on the NYUv2 test set. In addition to fine-tuning-based methods, we also outperform [56], which estimates metric depth from multi-aperture inputs via self-supervised learning with a differentiable forward model. See Supplement.L for more details on the comparison with [56].
Method RMSE ↓ REL ↓ log10 ↓ δ1 ↑ δ2 ↑ δ3 ↑ CD ↓ FA ↑ [71] FT
Extending our method beyond Marigold Our method is plug-and-play and, in principle, works with any diffusion-based model like Marigold that exposes a differentiable mapping from its latent space to depth estimates. This enables test-time latent optimization, which allows for incorporating defocus cues without retraining. We demonstrate this by using Geowizard as the diffusion backbone [16] instead of Marigold. GeoWizard is a stable-diffusion-based monocular depth + normal predictor, trained on a more complex data distribution. Our method achieves similar performance with either of the backbones (Rows 5,6 in table 2), but shows minor patch-level artifacts with GeoWizard, likely due to stronger texture-depth coupling. See Supplement.I for visualizations and hyperparameter details. Our method with the Disc PSF outperforms all the MMDE baselines averaged over all scenes in our dataset. The disc PSF, being more consistent with the real camera PSF, outperforms the Gaussian PSF. UniDepth, Metric3D, and our method are provided with camera intrinsics parameters during inference.
Method RMSE ↓ REL ↓ log10 ↓ δ1 ↑ δ2 ↑ δ3 ↑ CD ↓ FA ↑MLPro
this section cite: ['b39', 'b74', 'b7', 'b35', 'b20', 'b20', 'b71', 'b71', 'b42', 'b56', 'b56', 'b16']

Section: Ablation Studies
Only single blurred image as input: In our method, Marigold takes the all-in-focus (AIF) image (F/22) as input to predict relative depth, which is combined with learnable scale-offset parameters to synthesize a blurred image. The synthesized image is compared with a captured blurred image (F/8) that provides metric depth cues [52]. To assess the contribution of the AIF, we remove the AIF and instead provide a single moderately blurred image (F/16) as input to both Marigold and the loss function (eq. ( 11)). This ablation also evaluates whether the diffusion prior alone is strong enough to operate with a single modestly blurred image. Removing the AIF input leads to a large RMSE increase (1.36 vs. 0.346) and visible artifacts (fig. 6), indicating that while the diffusion prior captures coarse depth cues, the AIF input is essential for accurate metric depth estimation.
Sensitivity to α, β initialization: Gradient-based methods are known to be susceptible to local minima for non-convex optimizations. We thus evaluate the sensitivity of our method to the initialization for α, β. We run the optimization in (eq. ( 11)) with a fixed z  For the TOYS scene, we observe that using a single blurred image as input results in severely inaccurate relative depth (middle), with all toys, guitar, and the monitor at similar relative depths. Our proposed method (right) recovers the depth ordering between the objects more accurately.
values of α and β. In fig. 5 (right), we observe that performance (measured by δ 1 ) drops for small initialization values but remains stable across a broad range around α = 0.5, β = 0.5, supporting the robustness of our chosen initialization. We also ablate on sensitivity to initial z
(d) T in supplement.E.
Improvements in relative depth from defocus cues We quantitatively evaluate the effect of defocus cues in our method on improving the relative depth quality. Optimizing only α, β while holding z
T constant leads to a performance drop (table 3). This highlights the role of defocus cues in refining the relative depth initially predicted by Marigold. Please see supplement.F for visualizations.
Method RMSE ↓ REL ↓ log10 ↓ δ1 ↑ δ2 ↑ δ3 ↑ Ours α,
β opt 0.297 0.156 0.069 0.743 0.957 0.99 Ours 0.273 0.125 0.052 0.879 0.975 0.991
Table 3: Relative depth quality Optimizing the noise latent along with the affine parameters (ours) performs better than optimizing only the affine parameters (α, β opt).
this section cite: ['b52']

Section: Different aperture sizes
We analyze how the aperture (F-stop) used for capturing the blurred image affects our performance. To do this, we use a scene from the NYU-v2 [35], an indoor RGBD dataset with high-quality ground truth depth annotations. This synthetic setup allows evaluating large F-stops that cannot be captured with our camera, while isolating aperture size from forward model mismatches in a real setup. Using the ground truth depth and our forward model (eq. ( 7)), we simulate the blurred images x b at varying F-stops (N values) and compute the error metrics between the ground truth and our predicted depth. We observe in fig. 5 that the performance (measured in RMSE) degrades for extreme aperture sizes. This is expected, as extreme blur (high or low) makes the inverse problem ill-posed, and an optimal blur level is key for accurate depth recovery. While N = 13 appears to be optimal in simulation (ignoring model mismatch), it underperforms N = 8 on average for real scenes, likely due to low contrast and insufficient blur cues in some of the scenes (STAIRS, PLANE). Please see supplement.G for results across all apertures captured in the real dataset.
this section cite: ['b35']

Section: Limitations and Future Work
While our method outperforms data-driven MMDE baselines, it remains significantly slower at inference time. Our method is best suited to scenes with a small depth range for which defocus blur offers high depth sensitivity. We observe that if the initial Marigold prediction is severely incorrect in some regions (visualized in supplement.F), the optimization may not always be able to fully correct them (ours for SHOERACK, THORDOG in fig. 4). A possible extension of our method is to jointly estimate the AIF and depth map from a single blurred input, as previously explored in non-zero-shot approaches [19,1]. Our framework can broaden the utility of pre-trained depth priors to scientific applications involving depth-dependent imaging processes such as hyperspectral imaging [27], endoscopy [31], and microscopy [72]. While we avoid discretizing the depth map [23], our forward model loses accuracy at occlusion boundaries. We envision further improvements through better PSF engineering (coded aperture masks) and more accurate forward modeling of defocus blur. Another promising direction is to adapt feed-forward methods such as [71] for handling multi-aperture inputs through test-time adaptation and fine-tuning [73]. Large-scale multi-aperture datasets could make such methods practical for these settings.
this section cite: ['b19', 'b0', 'b27', 'b31', 'b72', 'b23', 'b71', 'b73']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer:. [Yes] Justification: We demonstrate evidence of the claims in the abstract in the section 5.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: In section 6, we discuss the limitations of our method.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [NA]
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We will mention all the hyperparameters in the supplement.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in the supplemental material? Answer: [Yes] Justification: We will release the code after acceptance. But the dataset will be provided as an anonymous link in the supplement.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: We specify implementation details in the paper; additional details will be included in the supplement.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: We perform thorough ablations on different initialization, and characterize the effects of different aperture sizes on our method through a synthetic study. In the supplement, we characterize the effect of the stochasticity of the initial latent noise vector of the diffusion model by repeating an experiment 10 times with different random initializations for the latent noise vector, and report the variance in the observed final metrics.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: We mention the GPU used, memory usage, and optimization time for our algorithm.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We confirm to the ethics guidelines.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: Our work has no particularly negative societal impacts.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: Our work poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer:[Yes] Justification: We have cited the datasets and models we used for comparison. These datasets have been widely used in published papers on this topic.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: We will release the self-collected dataset in the supplementary material with guidelines on how to use it, and all the camera parameters used while capturing it.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: No crowdsourcing involved. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: No human subjects involved Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: This paper is not related to LLM research or applications. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b16']

Section: References
Ref_id:b0 Title: Defocus deblurring using dual-pixel data Year: (2020)
Ref_id:b1 Title: Focal flow: Measuring distance and velocity with defocus and differential motion Year: (2016)
Ref_id:b2 Title: Proceedings, Part III 14 Year: (2016)
Ref_id:b3 Title: Diffusercam: lensless single-exposure 3d imaging Year: (2017)
Ref_id:b4 Title: Central limit theorems for gaussian polytopes Year: (2007)
Ref_id:b5 Title: Zoedepth: Zero-shot transfer by combining relative and metric depth Year: (2023)
Ref_id:b6 Title: Midas v3. 1-a model zoo for robust monocular relative depth estimation Year: (2023)
Ref_id:b7 Title: Depth pro: Sharp monocular metric depth in less than a second Year: (2024)
Ref_id:b8 Title: Deep depth from defocus: how can defocus blur improve 3d estimation using dense neural networks? Year: (2018)
Ref_id:b9 Title: Diffusion posterior sampling for general noisy inverse problems Year: ()
Ref_id:b10 Title: Improving diffusion models for inverse problems using manifold constraints Year: ()
Ref_id:b11 Title: Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction Year: (2022)
Ref_id:b12 Title: A survey on diffusion models for inverse problems Year: (2024)
Ref_id:b13 Title: Diffusiondepth: Diffusion denoising approach for monocular depth estimation Year: (2024)
Ref_id:b14 Title: Enhancing one-step text-to-image models through reward-based noise optimization Year: (2024)
Ref_id:b15 Title: Range estimation by optical differentiation Year: (1998)
Ref_id:b16 Title: Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image Year: (2024)
Ref_id:b17 Title: An image is worth one word: Personalizing text-to-image generation using textual inversion Year: (2022)
Ref_id:b18 Title: Learning single camera depth estimation using dual-pixels Year: (2019)
Ref_id:b19 Title: Passive snapshot coded aperture dual-pixel rgb-d imaging Year: (2024)
Ref_id:b20 Title: Single image depth estimation trained via depth from defocus cues Year: (2019)
Ref_id:b21 Title: Deep depth from focus Year: (2018)
Ref_id:b22 Title: Diffusion-based visual foundation model for high-quality dense prediction Year: (2024)
Ref_id:b23 Title: Depth from defocus with learned optics for imaging and occlusion-aware depth estimation Year: (2021)
Ref_id:b24 Title: A construct-optimize approach to sparse view synthesis without camera pose Year: (2024)
Ref_id:b25 Title: On the relation between time and intensity in photographic exposure Year: (1926)
Ref_id:b26 Title: Repurposing diffusion-based image generators for monocular depth estimation Year: (2024)
Ref_id:b27 Title: Imaging depth variations in hyperspectral imaging: development of a method to detect tumor up to the required tumor-free margin width Year: (2019)
Ref_id:b28 Title: Image and depth from a conventional camera with a coded aperture Year: (2007)
Ref_id:b29 Title: Depth recovery from light field using focal stack symmetry Year: (2015)
Ref_id:b30 Title: Matting and depth recovery of thin structures using a focal stack Year: (2017)
Ref_id:b31 Title: Self-supervised monocular depth estimation for gastrointestinal endoscopy Year: (2023)
Ref_id:b32 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b33 Title: Focus on defocus: bridging the synthetic to real domain gap for depth estimation Year: (2020)
Ref_id:b34 Title: Osmosis: Rgbd diffusion prior for underwater image restoration Year: (2024)
Ref_id:b35 Title: Indoor segmentation and support inference from rgbd images Year: (2012)
Ref_id:b36 Title: DITTO-2: Distilled diffusion inference-time t-optimization for music generation Year: ()
Ref_id:b37 Title: DITTO: Diffusion inference-time t-optimization for music generation Year: ()
Ref_id:b38 Title: Dual pixel exploration: Simultaneous depth estimation and image restoration Year: (2021)
Ref_id:b39 Title: Unidepth: Universal monocular metric depth estimation Year: (2024)
Ref_id:b40 Title: A lens and aperture camera model for synthetic image generation Year: (1981)
Ref_id:b41 Title: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer Year: (2020)
Ref_id:b42 Title: Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding Year: (2021)
Ref_id:b43 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b44 Title: Solving linear inverse problems provably via posterior sampling with latent diffusion models Year: (2024)
Ref_id:b45 Title: Norm-guided latent space exploration for text-to-image generation Year: (2024)
Ref_id:b46 Title: Learning depth from single monocular images Year: (2005)
Ref_id:b47 Title: The surprising effectiveness of diffusion models for optical flow and monocular depth estimation Year: (2023)
Ref_id:b48 Title: Zero-shot metric depth with a field-of-view conditioned diffusion model Year: (2023)
Ref_id:b49 Title: High-accuracy stereo depth maps using structured light Year: (2003)
Ref_id:b50 Title: Mgnet: Monocular geometric scene understanding for autonomous driving Year: (2021)
Ref_id:b51 Title: Dr. bokeh: Differentiable occlusion-aware bokeh rendering Year: (2024)
Ref_id:b52 Title: Fully self-supervised depth estimation from defocus clue Year: (2023)
Ref_id:b53 Title: Solving inverse problems with latent diffusion models via hard data consistency Year: (2023)
Ref_id:b54 Title: Pseudoinverse-guided diffusion models for inverse problems Year: (2023)
Ref_id:b55 Title: Depthmaster: Taming diffusion models for monocular depth estimation Year: (2025)
Ref_id:b56 Title: Aperture supervision for monocular depth estimation Year: (2018)
Ref_id:b57 Title: Accurate depth and normal maps from occlusion-aware focal stack symmetry Year: (2017)
Ref_id:b58 Title: Depth from defocus: A spatial domain approach Year: (1994)
Ref_id:b59 Title: Depth from defocus in the wild Year: (2017)
Ref_id:b60 Title: Marigold-dc: Zero-shot monocular depth completion with guided diffusion Year: (2024)
Ref_id:b61 Title: End-to-end diffusion latent optimization improves classifier guidance Year: (2023)
Ref_id:b62 Title: An implicit neural representation for the image stack: Depth, all in focus, and high dynamic range Year: (2023)
Ref_id:b63 Title: Bridging unsupervised and supervised depth from focus via all-in-focus supervision Year: (2021)
Ref_id:b64 Title: Rational filters for passive depth from defocus Year: (1998)
Ref_id:b65 Title: Foundationstereo: Zero-shot stereo matching Year: (2025)
Ref_id:b66 Title: Learning depth from focus in the wild Year: (2022)
Ref_id:b67 Title: Phasecam3d-learning phase masks for passive single view depth estimation Year: (2019)
Ref_id:b68 Title: Defocus map estimation and deblurring from a single dual-pixel image Year: (2021)
Ref_id:b69 Title: Depth from focusing and defocusing Year: (1993)
Ref_id:b70 Title: Depth anything: Unleashing the power of large-scale unlabeled data Year: (2024)
Ref_id:b71 Title: Depth anything v2 Year: (2024)
Ref_id:b72 Title: Miniscope3d: optimized single-shot miniature 3d fluorescence microscopy Year: (2020)
Ref_id:b73 Title: Rapid network adaptation: Learning to adapt neural networks using test-time feedback Year: (2023)
Ref_id:b74 Title: Metric3d: Towards zero-shot metric 3d prediction from a single image Year: (2023)
Ref_id:b75 Title: Survey on monocular metric depth estimation Year: (2025)
Ref_id:b76 Title: Joint image and depth estimation with mask-based lensless cameras Year: (2020)
Ref_id:b77 Title: TiNO-Edit: Timestep and noise optimization for robust diffusion-based image editing Year: (2023)
Ref_id:b78 Title: Coded aperture pairs for depth from defocus Year: (2009)
