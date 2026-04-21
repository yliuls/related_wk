Title: EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting
Abstract: Figure 1: Free-trajectory 3DGS under high speed. (Top) The overall paradigm. The colored dots in the top row represent the event data (red: positive, blue: negative). We leverage continuous event streams to aid discrete video frames captured along free trajectories in high-speed scenarios, jointly optimizing camera poses and reconstructing the 3DGS. Our method surpasses current state-of-the-art methods in terms of both rendered results (middle) and pose estimation (bottom).

Section: Introduction
In recent years, Neural Radiance Fields (NeRF) [1,2,3] and 3D Gaussian splatting (3DGS) [4,5] have made significant progress in novel view synthesis tasks. Given a set of posed images of the same scene, they optimize an implicit or explicit scene representation using volume rendering. While subsequent methods [2,3,6,7] excel with posed images, reconstructing scenes from videos with free camera trajectories remains challenging despite its applications in VR/AR, video stabilization, and mapping. To tackle this challenging task, several efforts have been made.
Accurate pose estimation is often difficult to obtain in free-trajectory scenarios, which directly impacts the quality of scene reconstruction. One line of work draws inspiration from Simultaneous Localization and Mapping (SLAM). They [8,5,9] follow its optimization paradigm, progressively optimizing camera trajectories and alternating between camera pose and scene refinement. Another line of work [10,9,11,5,12] explores incorporating additional geometric or motion priors such as depth estimation [13,14] or optical flow [15] to establish constraints beyond photometric rendering loss. While these methods can render photo-realistic images in typical free-trajectory scenarios, both their rendering quality and pose estimation accuracy degrade significantly in high-speed scenarios (or equivalently low-frame-rate scenarios) as shown in Fig. 1. Such high-speed scenarios have essential applications such as autonomous driving and First-Person View (FPV) exploration.
The performance degradation of prior methods can be attributed to two primary factors. First, the limited number of camera observations leads to an under-constrained scene reconstruction problem. This can cause the scene representation to converge to a trivial solution [16,17,12], where the model overfits to the training views without capturing the correct underlying geometry structure. Second, the substantial discrepancies between consecutive frames, resulting in diminished overlapping regions, violate the implicit assumption of continuous motion between adjacent frames, which is leveraged by previous methods. Moreover, geometric and motion priors like optical flow and feature matching become unreliable in such scenarios. These significant violations greatly exacerbate the ill-posedness of the joint optimization of scene and camera poses.
Event camera is a bio-inspired image sensor that asynchronously records per-pixel brightness changes, offering advantages such as high temporal resolution, high dynamic range, and no motion blur [18,19,20,21,22,23]. The brightness information recorded in the event stream can effectively complement the missing scene information between consecutive frames. Moreover, the event data naturally encodes the motion information of the scene [24,25,26], containing rich motion cues. These properties make event cameras well-suited for scene reconstruction tasks in high-speed and freetrajectory scenarios. However, seamlessly integrating the aforementioned benefits of event cameras is nontrivial. First, 3DGS renders absolute pixel brightness, which aligns with image data. Event cameras, however, record sparse differential brightness changes. Directly integrating the differential operations into 3DGS may amplify noise and lead to ill-conditioned optimization problems with high sensitivity to parameter initialization and perturbations. Second, event cameras encode motion through continuous spatio-temporal trajectories of events. In contrast, frame-based data inherently discretizes continuous motion, forcing traditional methods to rely on correspondence matching, which fails in high-speed scenarios with large inter-frame displacements. These fundamental challenges require carefully designed method that bridges the gap between the event data and 3DGS optimization.
In this work, we propose Event-Aided Free-Trajectory 3DGS, dubbed EF-3DGS, a framework that integrates event data into the scene optimization process to fully leverage its high temporal resolution property. Our approach comprises three key components: (1) In the Event Generation Model (EGM), we introduce an event-based re-render loss, which extends the 3DGS optimization to the continuous event stream. This allows us to utilize the brightness cues encoded in the event stream between adjacent frames, providing rich supervisory signals to alleviate the insufficient sparse view issues.
(2) In the Linear Event Generation Model (LEGM), regarding the pose estimation challenge, we introduce the CMax [27] framework to exploit the spatio-temporal correlations of events. We obtain the motion field by leveraging the pseudo-depth from 3DGS rendering and the relative camera motion between consecutive frames. We then warp the events triggered by the same edge along the motion trajectories to maximize the sharpness of the image of warped events (IWE), thereby estimating the motion that best matches the current spatio-temporal event patterns. Furthermore, through the LEGM [28,29], we establish a connection between motion and brightness changes. This allows us to constrain the 3DGS in the gradient domain using the IWE. (3) As most event data primarily records scene brightness changes, lacking color information, we introduce photometric bundle adjustment (PBA) and a Fixed-GS strategy to address this. PBA recovers color by optimizing reprojection errors onto RGB frames, while Fixed-GS enables separate optimization of scene structure and color.
Our main contributions are summarized as follows:
• We introduce event cameras into the task of free-trajectory scene reconstruction for the first time. Its advantage of high temporal resolution and low latency showcases the potential of event data for scene reconstruction tasks in challenging scenarios.
• We derive our method from the underlying imaging principles of event cameras and design the corresponding loss functions that mine the motion and brightness information encoded in event data and seamlessly integrate them into the 3DGS optimization.
• Experiments on both public benchmarks and real-world datasets demonstrate that our method significantly outperforms existing state-of-the-art approaches in terms of both rendering quality and trajectory estimation accuracy.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b1', 'b2', 'b5', 'b6', 'b7', 'b4', 'b8', 'b9', 'b8', 'b10', 'b4', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b11', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b0', 'b26', 'b27', 'b28', 'b2']

Section: Related Works

this section cite: []

Section: Joint Pose and Scene Optimization.
The research community has recently focused on developing methods [12,8,30,31,10,32,5] that can be optimized without requiring precomputed camera poses. A line of work has focused on improving the stability of the optimization process. GARF [31] and BARF [32] both find that the high-frequency position encoding is prone to local minima and try to improve it. For example, GARF [31] proposes using Gaussian activation to replace the sinusoidal position encoding. Another line of work has investigated incorporating additional constraints to make the problem more tractable. LocalRF [8] leverages the prior assumption of continuous motion between adjacent frames and progressively adds and optimizes camera poses. More recent approaches [10,8,5] leverage pre-trained networks, i.e., monocular depth estimation and optical flow estimation. Exploiting 3DGS's explicit representation, CF-3DGS [5] directly backprojects Gaussian points using depth maps. While the aforementioned methods have made notable progress, they have yet to fully address the challenges posed by high-speed scenarios or rely on a good pose initialization. Our approach addresses these issues by leveraging motion and brightness cues from event streams.
this section cite: ['b11', 'b7', 'b29', 'b30', 'b9', 'b31', 'b4', 'b30', 'b31', 'b30', 'b7', 'b9', 'b7', 'b4', 'b4']

Section: Event-Based Novel View Synthesis.
Recent works have explored the integration of event cameras [33,34,35] into the NeRF or 3DGS framework. Early approaches, such as E-NeRF [36] and EventNeRF [37], utilize event-based generative models, minimizing the difference between the rendered brightness changes and observed brightness changes. Building upon this, Robust e-NeRF [38] incorporates a more realistic imaging model into the event-based framework, accounting for factors like refractory periods and noise. Beyond event-based NeRF, efforts have also been made to integrate event data into image-based methods. For instance, E2NeRF [39] and EvDeblurNeRF [40] leverage the Event Double Integral (EDI) [41] model to address the deblurring problem, while DE-NeRF [42] and EvDNeRF [43] leverage the high temporal resolution property of event cameras to capture fast-moving elements in dynamic scene. More recently, Event-3DGS [44] and EaDeblur-GS [45] have extended previous approaches to 3D Gaussian Splatting, achieving superior rendering quality and real-time performance. A key distinction of our work is that, unlike the prior methods that rely on accurate precomputed poses, we target free-trajectory scenarios, jointly optimizing for both the camera poses and the scene representation. Furthermore, while previous works have been limited to simulated and simple environments, we evaluate our approach in large-scale outdoor scenarios with complex motions and lighting conditions.
this section cite: ['b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44']

Section: Preliminary
3DGS [4] parametrizes the 3D scene as a set of 3D gaussians {G k } K k=1 that carry the geometric and appearance information. Each 3D Gaussian is characterized by several learnable properties, including its center position µ ∈ R 3 , opacity α ∈ [0, 1], spherical harmonics (SH) features f k ∈ R 3×16 for view-dependent color c ∈ R 3 , rotation matrix R ∈ R 3×3 (stored in quaternion form), scale factor s ∈ R 3 . The shape of each Gaussian is defined by the covariance matrix Σ and the center (mean) point µ, G(x) = exp(-1 2 (x -µ) T Σ -1 (x -µ)). During rendering, a tile-based rasterizer is applied to enable fast sorting and α-blending. The color of each pixel is calculated via blending N ordered overlapping points:
C(r) = N i=1 c i α i i-1 j=1 (1 -α j ),(1)
where c i is calculated from spherical harmonics and view direction, α i is the multiplication of opacity and the transformed 2D Gaussian and r denotes the image pixel. With the forward rendering procedure, we can optimize 3DGS by minimizing a weighted combination loss of L 1 and L D-SSIM between observation and rendered pixels: L color = (1 -λ)L 1 ( Î, I) + λL D-SSIM ( Î, I), where λ is balancing weight which is set to 0.2 following [4]. By integrating depth d i in Equation (1) along the ray, we can also obtain a expected depth value D(r):
D(r) = N i=1 d i α i i-1 j=1 (1 -α j ).(2)
4 Method
The overall framework is shown in Fig. 2. Given a video of a free-trajectory {I i } captured at time {t i } and the event stream ε = {e k }, our goal is to reconstruct the 3DGS of the scene and the corresponding camera trajectory{T i }. Following the analysis-by-synthesis paradigm of 3DGS, we extend this approach by incorporating event camera data through two fundamental imaging principles: the Event Generation Model (EGM) and Linear Event Generation Model (LEGM). To address the absence of color information in events and ensure cross-view consistency, we further introduce photometric bundle adjustment (PBA) and a Fixed-GS training strategy. The byproduct IPWE is utilized to establish additional constraints on 3DGS.
this section cite: ['b3', 'b3']

Section: EGM Driven Optimization
The EGM describes how event cameras asynchronously record pixel-wise brightness changes. When the logarithmic brightness change at a pixel u k = (x k , y k ), exceeds a predefined contrast threshold C, ∆L(u k , t k ) . = L(u k , t k ) -L(u k , t k -δt) = p k C, (3) where L . = log(I) is the logarithm of intensity, p k ∈ {-1, +1} indicates the polarity of brightness changes, and t k is the triggered timestamp.
As shown in Fig. 2 (1.1), to leverage the high temporal resolution of events, we first divide the time interval between two adjacent video frames I i and I i+1 into N smaller subintervals ε i,j = {e k |t i,j ≤ t k ≤ t i,j+1 , ∆t = ti+1-ti N , t i,j = t i + j • ∆t}. This allows us to form accumulated event frames at a higher temporal resolution:
E i,j = e k ∈εi,j p k .(4)
We then reconstruct the latent intensity image I t at any intermediate time t ∈ {t i,j } by integrating the accumulated events with the most recent frame:
I t = I i,j = I i,0 • exp( j-1 n=0 E i,n • C) if j > 0 I i,0 if j = 0 . (5
)
This latent intensity image provides a supervisory signal for our event-based rendering loss:
L EGM = (1 -λ)L 1 ( Ît , I t ) + λL D-SSIM ( Ît , I t ).(6)
By enforcing consistency between rendered and latent intensity images, this loss effectively utilizes the brightness information encoded in event streams between adjacent frames, addressing the challenge of sparse viewpoints in high-speed scenarios.
this section cite: []

Section: Unified CMax and LEGM Optimization
While L EGM leverages the brightness change information recorded by events, it does not explicitly exploit the motion information encoded in the event stream. To address this, we introduce the Contrast Maximization (CMax) [27,46,47,48] framework and the LEGM [28,29,49]. These models complement the previous EGM-driven optimization.
Under constant scene illumination, events are triggered by the motion of scene edges, forming continuous trajectories in (x, y, t) space. As shown in Fig. 2 (1.2), by warping(back-projecting) events along the correct motion trajectories, we can obtain a sharp image of warped events (IWE). Therefore, the sharpness of the IWE can serve as an indication of the accuracy of the estimated motion. This insight motivates us to derive the motion field by leveraging the rendered depth from 3DGS using eq. ( 2) and the relative camera motion between neighboring timestamps. By optimizing the sharpness of the IWE, we can obtain the optimal motion field, which in turn helps to improve the geometric accuracy of the 3DGS and the camera poses.
As shown in Fig. 3, for efficiency, we adopt a piece-wise warping approach instead of warping individual events. Specifically, for current timestamp t ref = t i,j , we warp the event frames from previous r sub-intervals: E i,j-m→j = warp(E i,j-m , F i,j→j-m ),
where m ∈ [0, r], F i,j-m→j is the optical flow derived from the rendered depth D in eq. ( 2) and relative pose T i,j→j-m between two timestamps:
F i,j→j-m = Π(T i,j→j-m Π -1 (x, y, D)) -(x, y),(8)
where T i,j→j-m = T i,j-m T -1 i,j , Π projects a 3D point to image coordinates and Π -1 unprojects a pixel coordinate and depth into a 3D point. Then the image of piece-wise warped events (IPWE) at timestamp t i,j is computed by averaging the warped event frames:
IPWE i,j = 1 r + 1 j m=j-r E i,m→j ≈ 1 C ∆L.(9)
Following the Cmax framework, we maximize the variance of the IPWE, which is equivalent to minimize its opposite:
L cm = -Var(IPWE i,j ).(10)
Furthermore, based on the LEGM [28,29], the brightness change ∆L at pixel u can be approximated by the dot product of the image gradient ∇L and the optical flow u (note that L is the logarithm of an image):
∆L(u) = -∇L • u ≈ L(u) -L(u + u).(11)
It is noteworthy that the IPWE also encodes brightness change information. Combining eq. ( 9) and eq. ( 11), we establish a connection between the IPWE and the brightness changes of the rendered images:
C • IPWE i,j = L(u) -L(u + F i,j→j+1 ). (12
)
Note that to compute F i,j→j+1 , we estimate T i,j+1 by leveraging the assumption of locally linear motion from T i,j-1 and T i,j . Based on this relationship, we formulate an additional gradient-based loss:
L grad = ||C • IPWE i,j -( L(u) -L(u + F i,j→j+1 ))|| 2 , (13
)
where L is the logarithm of synthesised image Ît . Finally, the full LEGM loss is defined as:
L LEGM = λ cm L cm + λ grad L grad ,(14)
where λ cm and λ grad are the balancing weight.
this section cite: ['b26', 'b45', 'b46', 'b47', 'b27', 'b28', 'b48', 'b27', 'b28']

Section: Photometric Bundle Adjustment
The aforementioned event-based constraints, L EGM and L LEGM , leverage the brightness change and motion information encoded in the event data to constrain 3DGS. However, as event cameras only record brightness changes and lack color perception, directly applying them to 3DGS optimization may lead to inconsistent color rendering. To ensure cross-view consistency of the 3DGS rendering, we introduce the Photometric Bundle Adjustment (PBA) term.
Specifically, as shown in Fig. 2 (1.3), for a randomly sampled timestamp t ∈ {t i,j }, we establish the following photometric reprojection error:
L P BA = u∈P Is∈F ||I s (u ′ ) -Î(u)|| 2 ,(15)
where u ′ = Π(T i,j-r→j Π -1 (x, y, D(u))) represent the coordinate on target view projected from the pixel u of source view I s , P denotes the pixel samples of current frame, and F is the candidates of target video frames. We select F to be the nearest previous video frame in consideration of computation costs.
Methods Pose-Free Input 6 FPS 4 FPS 3 FPS 2 FPS 1 FPS PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ F2-NeRF × F 23.55 0.75 0.34 22.97 0.72 0.36 22.25 0.69 0.40 21.64 0.68 0.44 20.63 0.64 0.51 Nope-NeRF ✓ F 13.86 0.51 0.67 13.81 0.51 0.67 13.79 0.51 0.67 13.50 0.51 0.68 13.72 0.51 0.68 LocalRF ✓ F 23.94 0.73 0.36 23.05 0.71 0.39 22.49 0.69 0.40 21.20 0.66 0.44 19.42 0.63 0.48 CF-3DGS ✓ F 26.05 0.78 0.31 25.03 0.77 0.33 23.73 0.74 0.36 22.08 0.68 0.42 20.53 0.65 0.46 EvDeblurNeRF × E+F 22.43 0.71 0.38 21.23 0.69 0.42 20.09 0.65 0.49 17.52 0.62 0.55 15.19 0.55 0.60 ENeRF × E+F 23.62 0.73 0.37 22.84 0.70 0.38 21.85 0.69 0.41 20.52 0.66 0.46 18.09 0.60 0.52 Event-3DGS(E+F) × E+F 26.32 0.78 0.33 25.37 0.76 0.34 24.59 0.75 0.37 23.44 0.72 0.38 22.41 0.69 0.39 EvCF-3DGS ✓ E+F 26.07 0.78 0.32 25.48 0.77 0.33 24.61 0.75 0.36 22.81 0.70 0.38 21.73 0.67 0.43 EF-3DGS(Ours) ✓ E+F 26.66 0.79 0.30 26.01 0.78 0.30 25.38 0.77 0.31 24.43 0.74 0.34 23.96 0.72 0.36
Table 1: Quantitative evaluations on Tanks and Temples dataset. The best results are highlighted in bold. Note that, in the "input" column, "F" denotes traditional frame input, while "E+F" denotes hybrid frame and event input.
By minimizing L P BA across sampled views, we encourage the 3DGS model to produce geometrically and photometrically consistent renderings across events and video frames, thus effectively resolving color inconsistencies inherent in event data.
this section cite: []

Section: Fixed-GS Training Strategy
The L P BA term alone is insufficient to fully mitigate color distortion issues. To further address this challenge, we propose a two-stage Fixed-GS scene optimization strategy that takes advantage of 3DGS's explicit attribute representation. In the first stage, all the parameters are optimizable and the optimization is performed across all timestamps:
G * θ , T * i,j = argmin µ,α,r,s,f,Ti,j L event , t ∈ {t i,j },(16)
where µ, α, r, s, f is the position, opacity, rotation, scale factor and spherical harmonics of the Gaussians, and t is the sampled timestamp during training. This stage results in a scene reconstruction with accurate structure and brightness, albeit with potential color distortions due to the dominant colorless event supervision overwhelming the sparse RGB frame color supervision. The second stage focuses on recovering accurate color information. During this phase, optimization is conducted exclusively on video frames. We optimize only the spherical harmonic coefficients of the Gaussians while keeping other parameters fixed:
G * θ = argmin f L color , t ∈ {t i,0 } (17
)
The ratio between the first and second stages is empirically set to 4:1. This approach allows us to effectively address the color distortion problem while preserving the structural and brightness information obtained from the event data.
this section cite: []

Section: Overall Training Pipeline
Assembling all loss terms, we get the overall loss function:
L event = L EGM + L LEGM + λ P BA L P BA ,(18)
where λ P BA are the weighting factor. Note that since event cameras typically record only the changes in brightness intensity, the L EGM and L LEGM losses are computed in the grayscale domain, whereas the L P BA loss is calculated in RGB color space. We incorporate dynamic scene allocation strategies from LocalRF [8] for handling extended video sequences. Our overall training pipeline builds upon the progressive optimization scheme of CF-3DGS [5] while introducing novel components to integrate event stream data for robust free-trajectory scene reconstruction. Please refer to Section A.3 for the algorithm pipeline and additional implementation details.
this section cite: ['b7', 'b4']

Section: Experiments

this section cite: []

Section: Dataset
Tanks and Temples. We conduct comprehensive experiments on the Tanks and Temples dataset [50]. Similar to LocalRF [8], we adopt 9 scenes, covering large-scale indoor and outdoor scenes. For each scene, we sample a video clip with a 50-second duration, typically featuring free camera trajectories  and covering a considerable distance. Following LocalRF [8], we apply 4× spatial downsampling to the videos. To evaluate the robustness under varying camera speeds, we employ varying temporal downsampling of 6 FPS, 4 FPS, 3 FPS, 2 FPS, and 1 FPS. The reduction in frame rate effectively creates larger inter-frame displacements, simulating high-speed scenarios. To synthesize realistic event data, we first upsample the original videos by [51] and then apply the simulator V2E [52].
this section cite: ['b49', 'b7', 'b7', 'b50', 'b51']

Section: RealEv-DAVIS.
Due to the lack of free-trajectory event camera datasets, we introduce RealEv-DAVIS. Using a DAVIS346 camera that simultaneously captures frames and events at 346×260 resolution, we record 40-second handheld sequences at 25 FPS. We employ COLMAP for groundtruth poses. For SLOW scenarios, we retain every second frame, while for FAST scenarios, we keep only one frame per five frames. Further details are provided in Section A.2.
this section cite: []

Section: Implementation details
We follow the optimization parameters by the configuration outlined in the 3DGS [4]. We optimize the camera poses in the representation of quaternion rotation. The initial learning rate is set to 10 -5 and gradually decays to 10 -6 until convergence. The balancing weight λ cm , λ grad and λ P BA is empirically set to 0.1, 0.2 and 0.5. For the division of events between adjacent frames, we maintain a constant interval of 1 6 s for Tanks and Temples and 1 25 s for RealEv-DAVIS, setting the number of subinterval N accordingly. For example, in Tanks and Temples, N equals 2 for 3FPS and 6 for 1FPS. This ensures adherence to the constant brightness assumption within each sub-interval and provides adequate events for the following CMax warping. The intervals of neighboring warping r in CMax are set to 3. The contrast threshold C is set to 0.25 for Tanks and Temples and 0.21 for RealEv-DAVIS. We provide detailed ablation studies on these hyperparameters and additional implementation details in Section A.4.
this section cite: ['b3']

Section: Experimental Setup
Metrics. We evaluate all the methods from two aspects: novel view synthesis and pose estimation. For the novel view synthesis task, we report the standard metrics PSNR, SSIM [53], and LPIPS [54]. For the pose estimation task, we adopt the Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) metrics [55,56], as delineated in [10]. Since these metrics are inherently influenced by frame rate, we upsample all estimated poses to a consistent temporal resolution before evaluation for fair comparison across different frame rate settings. Methods Input SLOW FAST NVS Pose NVS Pose PSNR↑ SSIM↑ RPE t ↓ RPE r ↓ PSNR↑ SSIM↑ RPE t ↓ RPE r ↓ LocalRF F 20.83 0.6074 3.60 2.07 17.62 0.5192 5.22 2.96 CF-3DGS F 22.68 0.6287 2.49 1.55 17.59 0.5204 3.68 2.17 EvDeblurNeRF E+F 20.61 0.6064 --17.98 0.5269 --Event-3DGS (E+F) E+F 23.43 0.6456 --20.04 0.5515 --EvCF-3DGS E+F 22.89 0.6317 1.78 0.82 19.13 0.5380 2.70 1.28 EF-3DGS(Ours) E+F 23.65 0.6466 1.41 0.69 21.12 0.5620 1.80 0.89
Table 3: Rendering and pose estimation results on RealEv-DAVIS. Complete data and additional metrics are provided in the supplementary material.
Baselines. For a fair comparison, we focus on two categories of methods: (1) For frame-based approaches, we selected methods specifically addressing free-trajectory scenarios, such as LocalRF [8] and F2-NeRF [57]. We also include pose-free methods like Nope-NeRF [10] and CF-3DGS [5]. (2) For event-frame hybrid methods, we consider approaches that fuse events and frames, including ENeRF [36], EvDeblurNeRF [40] and Event-3DGS [44]. Since no existing method integrates events for free-trajectory scenarios, we implement EvCF-3DGS as a competitive baseline that leverages an event-based frame interpolation network (Time Lens [58]) to temporally upsample frames before feeding them into CF-3DGS.
this section cite: ['b52', 'b53', 'b54', 'b55', 'b9', 'b7', 'b56', 'b9', 'b4', 'b35', 'b39', 'b43', 'b57']

Section: Experimental Results
We select every ten frames as a test image for NVS evaluation following LocalRF [8]. Since the camera poses are unknown in our setting, we need to estimate the poses of test views. As in iNeRF [59], we freeze the 3DGS model, initialize the test poses with the poses of the nearest training frames, and optimize the test poses by minimizing the photometric error between rendered images and test views.
this section cite: ['b7', 'b58']

Section: Results on RealEv-DAVIS.
Table 3 validates our approach on the real-world RealEv-DAVIS dataset. EF-3DGS outperforms top-performing methods and handles real-world scenes effectively. In FAST scenarios, our method shows nearly 1dB PSNR improvement over the best baselines. This confirms our advantage in high-speed scenarios where frame-based methods struggle. Fig. 4 and Fig. 5 show our method preserves fine details and maintains accurate trajectories even during rapid motion, addressing key limitations of traditional approaches.
LEGM LLEGM LP BA Fixed GS NVS Pose PSNR↑ SSIM ↑ LPIPS ↓ RPEt ↓ RPEr ↓ ATE ↓ 20.53 0.65 0.46 0.1057 0.9768 0.8972 ✓ 22.16 0.68 0.42 0.0651 0.7529 0.5779 ✓ 21.07 0.67 0.44 0.0830 0.8869 0.7231 ✓ 20.96 0.65 0.46 0.0938 0.9875 0.9112 ✓ ✓ 22.83 0.68 0.40 0.0523 0.6387 0.3981 ✓ ✓ ✓ 23.46 0.70 0.37 0.0523 0.6387 0.3981 ✓ ✓ ✓ 23.09 0.70 0.38 0.0487 0.6259 0.3753 ✓ ✓ ✓ ✓ 23.96 0.72 0.36 0.0487 0.6259 0.3753 Table 4: Effect of each component in EF-3DGS. The best results are highlighted in bold. Results on Tanks and Temples. Tables 1 and 2 demonstrate two key findings: (1) Our event-aided approach achieves up to 3dB higher PSNR and nearly 40% lower trajectory error at 1FPS compared to frame-based methods, indicating the critical value of event data in high-speed scenarios. (2) Our method maintains 1.55dB PSNR advantage over Event-3DGS at 1FPS, confirming that our integration framework effectively exploits the nature of event data beyond merely using it. Fig. 4 shows our method produces sharper edges and finer textures, while Fig. 5 illustrates we achieve more accurate trajectory estimation.
Performance under Varying Camera Speeds As shown in Table 2 and Table 3, while all methods degrade as the frame rate decreases, our approach shows remarkable resilience. The performance gap widens significantly at lower frame rates, with our PSNR advantage over CF-3DGS [5] increasing from 0.61dB at 6FPS to 3.43dB at 1FPS. Notably, our method also consistently outperforms other event-based methods (EvCF-3DGS and Event-3DGS). This confirms not only the value of event data in challenging scenarios but also the superiority of our integration approach.
this section cite: ['b4']

Section: Ablation Studies
Effect of Each Component Table 4 presents a comprehensive ablation study of our key components under the challenging 1FPS setting on Tanks and Temples. L EGM serves as the foundation of our approach, providing substantial improvements in both rendering quality (+1.63dB PSNR) and pose accuracy by enabling rich supervision between discrete frames. Building upon this, L LEGM extracts motion information from events and constrains 3DGS in the gradient domain, significantly improving pose estimation while modestly enhancing rendering quality. L P BA , though designed to address color inconsistency issues, not only improves rendering quality but also enhances pose estimation accuracy by establishing geometric and photometric consistency across views. The Fixed-GS training strategy, while having no impact on pose optimization, significantly improves rendering quality by effectively separating structure and color optimization. We provide more intuitive ablation visualizations in Section A.5.
this section cite: []

Section: Robustness to Pose Disturbance
To validate the robustness of different methods under inaccurate pose initialization, a common challenge in practical scenarios, we introduce varying degrees of perturbations to the initial camera poses estimated by COLMAP. Specifically, following BARF [32], we parametrize the camera poses p with the se(3) Lie algebra. For each scene, we synthetically perturb the camera poses with additive noise δp ∼ N (0, nI), where n is the noise level. Then, each method is initialized with the noised poses, after which the optimization is performed. The results are illustrated in Fig. 6. Notably, Event-3DGS [44], which lacks the capability to optimize camera poses, exhibits a drastic performance degradation as the magnitude of pose disturbances increases. This observation validates the critical importance of joint pose-scene optimization. Furthermore, Our proposed framework demonstrates superior tolerance across all perturbation levels. Even under significant noise, our method experiences substantially less degradation in both rendering quality and trajectory accuracy.
this section cite: ['b31', 'b43']

Section: Conclusions
In this work, we propose Event-Aided Free-Trajectory 3DGS (EF-3DGS), a novel framework that seamlessly integrates event camera data into the task of reconstructing 3DGS from casually captured free-trajectory videos. Our method effectively leverages the high temporal resolution and motion information encoded in event streams to enhance the 3DGS optimization process, leading to improved rendering quality and accurate camera pose estimation. By introducing the Event Generation Model and Linear Event Generation Model, we bridge the gap between differential event data and absolute brightness rendering in 3DGS. The proposed photometric bundle adjustment and Fixed-GS strategy further ensure accurate color recovery and scene structure optimization. Extensive experiments on both public benchmarks and real-world datasets validate the effectiveness of our approach, demonstrating significant improvements over state-of-the-art methods in both rendering quality and trajectory estimation accuracy. Future work would explore self-adaptive parameter adjustment strategies to enhance the method's versatility and ease of use across various reconstruction tasks.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: References
Ref_id:b0 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2020)
Ref_id:b1 Title: Mip-nerf 360: Unbounded anti-aliased neural radiance fields Year: (2022-06)
Ref_id:b2 Title: Instant neural graphics primitives with a multiresolution hash encoding Year: (2022-07)
Ref_id:b3 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023-07)
Ref_id:b4 Title: Colmap-free 3d gaussian splatting Year: (2024-06)
Ref_id:b5 Title: Tensorf: Tensorial radiance fields Year: (2022)
Ref_id:b6 Title: Zip-nerf: Anti-aliased grid-based neural radiance fields Year: (2023-10)
Ref_id:b7 Title: Progressively optimized local radiance fields for robust view synthesis Year: (2023-06)
Ref_id:b8 Title: A construct-optimize approach to sparse view synthesis without camera pose Year: (2024)
Ref_id:b9 Title: Nope-nerf: Optimising neural radiance field with no pose prior Year: (2023-06)
Ref_id:b10 Title: Robust dynamic radiance fields Year: (2023)
Ref_id:b11 Title: Sparf: Neural radiance fields from sparse and noisy poses Year: (2023-06)
Ref_id:b12 Title: Boosting monocular depth estimation models to high-resolution via content-adaptive multi-resolution merging Year: (2021-06)
Ref_id:b13 Title: Vision transformers for dense prediction Year: (2021-10)
Ref_id:b14 Title: Raft: Recurrent all-pairs field transforms for optical flow Year: (2020)
Ref_id:b15 Title: Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs Year: (2022-06)
Ref_id:b16 Title: Sparsenerf: Distilling depth ranking for few-shot novel view synthesis Year: (2023-10)
Ref_id:b17 Title: Event-based vision: A survey Year: (2022)
Ref_id:b18 Title: Event-aided direct sparse odometry Year: (2022-06)
Ref_id:b19 Title: Ultimate slam? combining events, images, and imu for robust visual slam in hdr and high-speed scenarios Year: (2018)
Ref_id:b20 Title: Event-based optical flow via transforming into motion-dependent view Year: (2024)
Ref_id:b21 Title: Event-based asynchronous hdr imaging by temporal incident light modulation Year: (2024)
Ref_id:b22 Title: Event-driven video restoration with spiking-convolutional architecture Year: (2023)
Ref_id:b23 Title: Globally optimal contrast maximisation for event-based motion estimation Year: (2020-06)
Ref_id:b24 Title: Tackling event-based lip-reading by exploring multigrained spatiotemporal clues Year: (2024)
Ref_id:b25 Title: A hybrid neuromorphic object tracking and classification framework for real-time systems Year: (2024)
Ref_id:b26 Title: A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation Year: (2018-06)
Ref_id:b27 Title: Event-based camera pose tracking using a generative event model Year: (2015)
Ref_id:b28 Title: Asynchronous, photometric feature tracking using events and frames Year: (2018-09)
Ref_id:b29 Title: Nerf-: Neural radiance fields without known camera parameters Year: (2022)
Ref_id:b30 Title: Gaussian activated neural radiance fields for high fidelity reconstruction and pose estimation Year: (2022)
Ref_id:b31 Title: Barf: Bundle-adjusting neural radiance fields Year: (2021-10)
Ref_id:b32 Title: Emotive: Event-guided trajectory modeling for 3d motion estimation Year: (2025)
Ref_id:b33 Title: Event-based tracking any point with motionaugmented temporal consistency Year: (2024)
Ref_id:b34 Title: Urnet: uncertainty-aware refinement network for event-based stereo depth estimation Year: (2025)
Ref_id:b35 Title: E-nerf: Neural radiance fields from a moving event camera Year: (2023)
Ref_id:b36 Title: Eventnerf: Neural radiance fields from a single colour event camera Year: (2023-06)
Ref_id:b37 Title: Robust e-nerf: Nerf from sparse & noisy events under non-uniform motion Year: (2023-10)
Ref_id:b38 Title: E2nerf: Event enhanced neural radiance fields from blurry images Year: (2023-10)
Ref_id:b39 Title: Mitigating motion blur in neural radiance fields with events and frames Year: (2024)
Ref_id:b40 Title: Bringing a blurry frame alive at high frame-rate with an event camera Year: (2019-06)
Ref_id:b41 Title: Deformable neural radiance fields using rgb and event cameras Year: (2023-10)
Ref_id:b42 Title: Evdnerf: Reconstructing event data with dynamic neural radiance fields Year: (2024-01)
Ref_id:b43 Title: Event-3dgs: Event-based 3d reconstruction using 3d gaussian splatting Year: (2025)
Ref_id:b44 Title: Eadeblur-gs: Event assisted 3d deblur reconstruction with gaussian splatting Year: (2024)
Ref_id:b45 Title: Globally-optimal contrast maximisation for event cameras Year: (2022)
Ref_id:b46 Title: Event cameras, contrast maximization and reward functions: An analysis Year: (2019-06)
Ref_id:b47 Title: Cmax-slam: Event-based rotational-motion bundle adjustment and slam system using contrast maximization Year: (2024)
Ref_id:b48 Title: Formulating event-based image reconstruction as a linear inverse problem with deep regularization using optical flow Year: (2022)
Ref_id:b49 Title: Tanks and temples: benchmarking large-scale scene reconstruction Year: (2017-07)
Ref_id:b50 Title: Real-time intermediate flow estimation for video frame interpolation Year: (2022)
Ref_id:b51 Title: Video to events: Recycling video datasets for event cameras Year: (2020-06)
Ref_id:b52 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b53 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018-06)
Ref_id:b54 Title: A tutorial on quantitative trajectory evaluation for visual(-inertial) odometry Year: (2018)
Ref_id:b55 Title: A benchmark for the evaluation of rgb-d slam systems Year: (2012)
Ref_id:b56 Title: F2-nerf: Fast neural radiance field training with free camera trajectories Year: (2023-06)
Ref_id:b57 Title: Time lens: Event-based video frame interpolation Year: (2021-06)
Ref_id:b58 Title: Inverting neural radiance fields for pose estimation Year: (2021)
