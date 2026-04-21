Title: RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes
Abstract: Figure 1: (a) Overview of our RGB-only supervised camera parameter optimization. (b) Front view of the 3D Gaussian field reconstructed by our camera estimates at time t. (c) 2D renderings (RGB and depth) at time t with quantitative metrics. Our optimization is not only significantly more efficient and accurate, but also avoids overfitting the reconstruction to specific viewpoints. Record3D is a mobile app that factory-calibrates the intrinsic and uses LiDAR sensors to collect metric depth for camera pose estimates, thus does not have valid runtime.

Section: Introduction
Despite recent progress in visual odometry, efficiently and accurately optimizing camera parameters 1(focal length + rotation&translation) from casually collected RGB dynamic-scene videos remains a big challenge. Although the most predominant COLMAP [32] method 2 is RGB-only supervised, it suffers from its lengthy runtime and requisite of GT motion masks to mask out the outlier moving stuff. In Table 1, most recent approaches [6,45,3,42,59,46,56,44] attempted to improve through being supervised by additional GT priors such as focal length, metric depth, 3D point clouds, camera poses, and motion masks, which are typically unavailable in casually collected videos. We cannot help but ask a natural question: Is it possible to accurately and efficiently estimate camera parameters in dynamic scenes in an RGB-only supervised manner -the most minimal form of supervision?
Existing RGB-only supervised methods [42,45,20,59, 3] make obvious improvements, but they mostly rely on multiple pre-trained dense prediction models [38,13,31] to compensate for the inaccuracies of individual pseudo-supervision sources, resulting in performance degradation if any of them fails. They also cannot adaptively exclude moving outliers without GT motion supervision. Besides, their high computational latency always leads to lengthy runtimes. Further discussion of related work is provided in Section 2.
Table 1: Categorization of supervision of current methods. Ours, casualSAM [58], and Robust-CVD [16] are RGB-only supervised, while our performance is the best as shown in section 4.
this section cite: ['b31', 'b5', 'b44', 'b2', 'b41', 'b45', 'b43', 'b41', 'b44', 'b19', 'b37', 'b12', 'b30', 'b15']

Section: Supervision

this section cite: []

Section: Static Scene Dynamic Scene
GT 3D Point Cloud & Camera Pose Dust3r [46], Fast3r [52], Mast3r [18], Spann3r [40], VGGT [41] Monst3r [56], Cut3r [44], Stereo4D [11], Easi3r [4] GT Focal Length CF-3DGS [6], Nope-NeRF [1], LocalNeRF [22]
this section cite: ['b45', 'b51', 'b17', 'b39', 'b40', 'b43', 'b10', 'b3', 'b5', 'b0', 'b21']

Section: + Metric Depth

this section cite: []

Section: DROID-SLAM [39] + GT Motion Priors
GFlow [45], LEAP-VO [3] GT Motion Priors RoDynRF [20], COL w/ mask [32], ParticleSfM [59]
RGB-Only VGGSfM [42], FlowMap [35], InstantSplat [5], COL w/o mask [32] Robust-CVD [16], casualSAM [58], Ours (ROS-Cam)
Based on these insights, we propose ROS-Cam, an RGB-only supervised, accurate, and efficient camera parameter optimization method, with a brief performance overview in Figure 1. Specifically, to minimize reliance on pre-trained dense prediction models while still establishing robust and maximally sparse hinge-like relations across the video as accurate pseudo-supervision (bottom right corner in Figure 2), we propose the novel patch-wise tracking filters built solely on a pre-trained point tracking (PT) model. This formulation effectively avoids inaccurate tracking trajectories extracted across frames and computational latency induced by the noisy dense prediction as pseudo-supervision.
However, the extracted pseudo-supervision includes a portion of trajectories belonging to moving outliers. To eliminate the influence of such outliers, we introduce a learnable uncertainty associated with each calibration point, where each is a learnable 3D position in the world coordinates, corresponding to one extracted tracking trajectory. We model such uncertainty parameters with the Cauchy distribution, which can deal with heavy tails better than, e.g., the Gaussian distribution, and propose the novel Average Cumulative Projection error and Cauchy loss for the outlier-aware joint optimization of the calibration points, focal length, rotation, translation, and uncertainty parameters. Unlike casualSAM and LEAP-VO, which assign uncertainty parameters to 2D pixels, our approach associates uncertainties with sparse 3D calibration points, resulting in significantly fewer learnable parameters and reduced runtime, as shown in Table 3.
Such joint optimization is prone to getting trapped in local minima. To address this, we analyze the asymptotic behavior of the Softplus function and the analytical minima of the inner convex term in losses to propose a two-stage optimization strategy to accelerate and stabilize the optimization. We evaluate the performance of our method through extensive experiments on 5 popular public datasets -NeRF-DS [50], DAVIS [28], iPhone [7], MPI-Sintel [2], and TUM-dynamics [36], demonstrating our superior performance. Our contributions can be summarized as follows.
• We propose the first RGB-only supervised, accurate, and efficient camera parameter optimization method in dynamic scenes with three key components: (1) patch-wise tracking filters;
(2) outlier-aware joint optimization; and (3) a two-stage optimization strategy.
• We present exhaustive quantitative and qualitative experiments and extensive ablation studies that demonstrate the superior performance of our proposed method and the contribution of each component.
2 Related Works Dynamic Scene Reconstruction/Novel View Synthesis (NVS). Existing methods for reconstructing objects and scenes use a variety of 3D representations, including planar [8,9], mesh [51,55], point cloud [48,57], neural field [23,50,37,29,21], and the recently introduced Gaussian explicit representations [49,14,10,47,53]. NeRF [23] enables high-fidelity NVS. Some methods [25,26,29,50,15,24] also extend NeRF to dynamic scenes, while others [51,55,54,43,37] build on them to extract high-quality meshes. However, NeRF-based methods have the limitation of a long training time. Recently, 3DGS [14] effectively addressed this issue by using 3D Gaussian-based representations and presented Differential-Gaussian-Rasterization in CUDA. 3DGS optimizes 3D Gaussian ellipsoids as dynamic scene representations associated with attributes such as position, orientation, opacity, scale, and color. Several studies [47,53] also have used 3DGS for dynamic scenes, achieving near real-time dynamic scene novel view synthesis. However, both NeRF-based and 3DGS-based methods heavily rely on COL w/ mask to estimate camera parameters.
Camera Parameter Optimization. Many efforts have been made to overcome the shortcomings of COLMAP, particularly for dynamic scenes. But each suffers from some constraints. In Table 1, we present a categorization of supervision of current SOTA methods. Supervised by additional GT focal length, CF-3DGS [6], Nope-NeRF [1], and LocalNeRF [22] leverage a pre-trained monocular depth estimation model [31] to estimate camera poses and the static scene jointly. The most representative SLAM-based method -DROID-SLAM [39], leverages both GT focal length and metric depth as supervision. GFlow and LEAP-VO [45, 3] extend it to dynamic scenes with both GT focal length and motion priors as supervision. Although VGGSfM, FlowMap, InstantSplat, COL w/o mask [42,35,5,32] eliminate the GT focal length requirement by leveraging pre-trained PT models [38,13], they cannot handle the moving objects in dynamic scenes. RoDynRF [20], COL w/ mask [32], and ParticleSfM [59] simply tackle such a problem by incorporating GT motion supervision like GFlow. Recently, DUSt3Rbased methods [46,52,18,40,41] and their dynamic-scene counterparts [56, 44, 11, 4] explored feed-forward camera parameter prediction by training on large-scale static and dynamic scene datasets, respectively, in a fully supervised manner -that is, using GT 3D point clouds and camera poses as supervision, requiring several days' training on high-end GPUs. However, unlike LLMs that benefit from abundant language data, such metric 3D supervision is relatively scarce in the vision area, leading to frequent domain gaps when these models are applied to unseen data. In contrast, Robust-CVD [16], casualSAM [58] and our method conduct camera parameter optimization for dynamic scenes in a more general RGB-only supervised way. However, as shown in Section 4, their performance is significantly worse than ours.
this section cite: ['b44', 'b2', 'b19', 'b31', 'b41', 'b34', 'b4', 'b15', 'b49', 'b27', 'b6', 'b1', 'b35', 'b7', 'b8', 'b50', 'b54', 'b47', 'b22', 'b49', 'b36', 'b28', 'b20', 'b48', 'b13', 'b9', 'b46', 'b52', 'b22', 'b24', 'b25', 'b28', 'b49', 'b14', 'b23', 'b50', 'b54', 'b53', 'b42', 'b36', 'b13', 'b46', 'b52', 'b5', 'b0', 'b21', 'b30', 'b38', 'b41', 'b34', 'b4', 'b31', 'b37', 'b12', 'b19', 'b31', 'b45', 'b51', 'b17', 'b39', 'b40', 'b15']

Section: Method
Under RGB-only supervision, RGB frames F i , i ∈ [0, N -1] (N is frame count) are given. Our proposed patch-wise tracking filters (Section 3.1) extract H robust and maximally sparse hingelike tracking trajectories as pseudo-supervision, where each corresponds to one calibration point P cali h ∈ R 3 , h ∈ [0, H] in the world coordinates. Under such pseudo-supervision and our newly proposed ACP error and Cauchy loss, the calibration points P cali , focal length f ∈ R, quaternion matrix Q ∈ R N ×4 , translation t ∈ R N ×3 , and motion-caused uncertainty parameters Γ ∈ R H >0 are jointly optimized (Section 3.2). Γ is the scale parameter of the Cauchy distribution which is used to model such uncertainty parameters, associated with each calibration point, to reduce the erroneous influence of moving outliers. By analyzing the Softplus limits and convex minima in losses, we propose a simple but effective two-stage optimization strategy (Section 3.3) to enhance the stability and optimization speed.
this section cite: []

Section: Patch-wise Tracking Filters
Built on a pre-trained PT model, we observe that its attention mechanism assigns higher attention weights to pixels with more accurate tracking results which are always texture-rich pixels with large gradient norms. Inspired by it, as shown in Figure 2, we propose the patch-wise texture filter to identify the high-texture patches within F t and the patch-wise gradient filter to select the pixel with the highest gradient norm within each identified patch. While tracking such identified points, the visibility filter keeps removing trajectories that become invisible and the patch-wise distribution filter keeps the one with the largest gradient norm when multiple moving points enter the same patch. As shown in appendix E.2.1 (fig. 10), our method only retains the robust and accurate trajectories as pseudo-supervision. Patch-wise Texture Filter. Highly distinguishable points, that can be tracked reliably, belong to highly nonuniform (textured) neighborhoods. To identify such neighborhoods, our patch-wise texture filter computes a texture map T i ∈ 1 H/w×W/w , giving a measure of texture level for each w × w patch where H and W denote the height & width of F i . We represent the texture level of a patch by
T i [m, n] = 1{Σ i [m, n] > τ var • σ * }(1)
, where Σ i ∈ R H/w×W/w is the intensity variance, σ * = max(Σ i ), τ var is the percentage threshold of minimum variance for the patch to be selected, and m, n ∈ [0, H/w -1], [0, W/w -1]. The texture levels of a patch are represented by 1 for the selected patches and 0 for the others.
Patch-wise Gradient Filter. Within the identified patches, our patch-wise gradient filter computes the intensity gradient norm map G i ∈ R H×W of F i , and selects the point with the largest gradient norm within each patch. This yields the pool of potentially distinguishable points, forming potential trajectories, namely,
P potential m,n = arg max p (Gi[mw : mw + w, nw : nw + w]), p → pixel locations (2
)
Visibility Filter. We find that current PT models [13,12,30] still tend to suffer from reduced tracking accuracy when a point becomes occluded and later reappears, due to the disruption of temporal feature continuity. Thus, if any P in any F i becomes invisible, our visibility filter deletes it by the dot product P • V, V ∈ {0, 1} ∼ P, where V = 0 if a point is invisible.
Patch-wise Distribution Filter. This filter enforces a more even point distribution within each frame, preventing them from clustering into a small region as the viewpoint changes. It also helps reduce susceptibility to loss of resolution which might result in triangulation errors. We keep the highest-gradient tracking point P * in each patch P at m,n of F i , as follows: P * = arg max P Gi[ P ∈ P atm,n], if 1( P ∈ P atm,n) > 1
As shown in Figure 2, locations and indices of P * are stored in P i ∈ R B×2 and I i ∈ R B , i ∈ [0, H -1], acting as pseudo-supervision in the outlier-aware joint optimization. Each iteration starts at F t , t = arg min t (-1 ∈ I t ), and ends until each frame contains exactly B tracked points.
this section cite: ['b12', 'b11', 'b29']

Section: Outlier-aware Joint Optimization
Outlier-aware Joint Optimization Mechanism. Under the obtained pseudo-supervision, P cali , f , Q, t and Γ are jointly optimized. We first project P cali-homo ∈ R H×4 (the homogeneous coordinates of P cali obtained by concatenating 1) onto each frame by
P proj-homo i = P cali-homo [Ii] • Ri ti 0 1 T • K T (4
)
P proj i = P proj
-homo i [:, : 2]/P proj-homo i [:, 3] (5)
, where i ∈ [0, N -1] and P proj ∈ R N ×B×2 . P proj-homo ∈ R N ×B×4 denotes the homogeneous 2D location of the projection P proj . The perspective projection matrix K ∈ R 4×4 is derived from f , and the world-to-camera transformation matrix consists of rotation R i and translation t i . We assume constant f like SOTA [59,58,20]. Notably, we learn the quaternion matrix Q i instead of optimizing the R i and additional constraints. This optimization approach circumvents the difficult-to-enforce orthogonality and ±1 determinant constraints required for rotation matrices during optimization.
Figure 3: Outlier-aware Joint Optimization.
• represents P t and P t ′ on each frame. The static samples p ′cali and p ′′cali can establish concrete triangulation relations with their corresponding P t , P t ′ , and cameras, resulting in lower γ ′ and γ ′′ . In contrast, the dynamic sample p ′′′cali exhibits the opposite behavior. Table 3, and appendix E.1.1 (table 9, table 10, table 11).
However, the extracted pseudo-supervision always contains moving outliers. To mitigate its impact, without any GT motion priors, we identify such outliers by modeling the uncertainty their presence may cause in the observed distributions of the inlier points. We introduce the uncertainty Γ ∈ R H associated with P cali ∈ R H and incorporate the Cauchy distribution
f (x; x 0 , Γ) = 1 πΓ[1+( x-x 0 Γ ) 2 ]
, Γ > 0 to model the uncertainty parameter Γ since this distribution can better handle the heavy tails than, e.g., the Gaussian distribution. As depicted in Figure 3, during optimization, inliers are expected to have low uncertainty, while outliers have high uncertainty. Since the scale parameter Γ in f (x; x 0 , Γ) is required to be strictly positive, we introduce a new parameter Γ raw which we obtain Γ from using the Softplus function Γ = log(1 + e Γ raw ), Γ raw ∈ R H . This effectively ensures Γ ∈ R H >0 is differentiable and has smooth gradients. Losses. To down-weight outliers by learned Γ, we replace the commonly used projection error E proj = ∥P proj -P∥ 2  2 with our proposed Average Cumulative Projection (ACP) error, defined as:
E ACP h∈[0,H-1] = 1 {I=h} • ∥P proj -P∥ 2 2 1 {I=h}(6)
, where E ACP ∈ R H and • denotes the element-wise matrix multiplication. For each P cali h , we accumulate the errors between its corresponding projection and tracking locations across the video, then take the average as E ACP h∈[0,H-1] . Furthermore, we propose the novel Cauchy loss L cauchy in terms of the negative-log-likelihood log (Γ + (x-x0) 2 Γ ) of f (x; x 0 , Γ) where we replace x -x 0 with E ACP as eq. ( 7). Our total loss L total in Equation ( 8) consists of L cauthy and a depth regularization term R depth to encourage positive depth. With the estimated camera parameters, we use 4DGS [47] for scene reconstruction. Reconstruction and loss derivation details are in appendix A and appendix B.
L cauchy = 1 H H h=0 log (Γ + (E ACP ) 2 Γ )(7)
L total = L cauthy + R depth , R depth = 1 N N i=0 -ReLU(P proj-homo i [:, 3])(8)
this section cite: ['b19', 'b46']

Section: Two-stage Optimization Strategy
To avoid convergence to local minima, we propose this strategy based on an analysis of the asymptotic behavior of the Softplus function and the analytical minima of the inner convex term in L cauthy . Stage 1 focuses on rapid convergence, while Stage 2 aims for stable convergence by initializing Γ raw to the ACP error after Stage 1. The effectiveness of it is concretely demonstrated in Table 7.
Stage 1.
In the Softplus function, Γ = log(1 + e Γ raw ) ≈ Γ raw , as Γ raw → +∞. So in Stage 1, we fix Γ raw = 1 and optimize only P cali , f , Q, and t for quick convergence. The loss will converge to a certain value beyond the global minimum, as there is no proper Γ to down-weight outliers. Stage 2. The inner term Φ = x + O x , O > 0 of L cauchy is convex. Assuming a constant O ∈ R + and solving for min x Φ(x), we have x * = √ O. Similarly, in Stage 2, if Γ raw is randomly initialized to values largely different from E ACP stage1 (the ACP error from Stage 1), convergence will be highly unstable. Therefore, we initialize Γ raw = E ACP stage1 , and optimize P cali , f , Q, t, and Γ raw jointly.
this section cite: []

Section: Experiments
To demonstrate the superiority of our method, we show extensive quantitative and qualitative results in this section. For NeRF-DS [50], DAVIS [28], and iPhone [7] datasets without GT camera parameters, we feed the camera parameters from different methods to 4DGS [47], while keeping all other factors the same, and evaluate each NVS performance (PSNR, SSIM, and LPIPS). Regarding the MPI-Sintel [2] and TUM-dynamics [36] datasets with GT camera parameters, we directly evaluate methods by ATE, RPE trans, and RPE rot metrics. In all tables, the best and second-best results are bold and underline. More about datasets, and evaluation metrics are in appendix C and appendix D.
this section cite: ['b49', 'b27', 'b6', 'b46', 'b1', 'b35']

Section: Implementation Details
The optimization is conducted on 1 NVIDIA A100 40GB GPU with Adam [27] optimizer and learning rates l Q = 0.01, l t = 0.01, l f = 1.0, l P cali = 0.01, and l Γ raw = 0.01. We also choose to build our patch-wise tracking filters on CoTracker  Table 2:
NVS Evaluation on NeRF-DS [50] and DAVIS [28]. (PSNR↑/SSIM↑/LPIPS↓) * is supervised by additional GT priors. Ours is the best among these two datasets.
this section cite: ['b26', 'b49', 'b27']

Section: Method

this section cite: []

Section: NeRF-DS DAVIS
RoDynRF[20] * 23.033/0.749/0.385 -COL w/ mask * 32.174/0.923/0.147 -COL w/o mask 29.348/0.875/0.224 9.196/0.236/0.435 casualSAM[58] 21.230/0.686/0.463 19.032/0.486/0.482 Ours 33.552/0.938/0.118 22.292/0.709/0.279
Table 3: Runtime Evaluation on NeRF-DS [50], DAVIS [28], and iPhone [7], covering frame count from 50 to 900. * is supervised by additional GT priors.
Our method is the most efficient.
this section cite: ['b49', 'b27', 'b6']

Section: Method NeRF-DS DAVIS iPhone
RoDynRF [20] * 29.6h 27.4h 28.5h COL w/ mask * 1.5h --COL w/o mask 1.8h 0.51h 9.53h casualSAM [58] 10.5h 0.28h 4.07h Ours 0.83h 0.03h 0.33h
Table 4: Camera Pose Evaluation on TUMdynamics [36]. Other results are from Cut3r [44] and Monst3r [56]. Performance of DROID-SLAM [39] is from casualSAM [58]. Our method achieves the best overall performance among all RGB-only supervised methods, and even better than the ones supervised by additional GT priors.
Supervision Method ATE↓ RPE trans↓ RPE rot↓ GT 3D Point Cloud & Camera Pose Monst3r [56] 0.098 0.019 0.935 Dust3r [46] 0.083 0.017 3.567 Mast3r [18] 0.038 0.012 0.448 Cut3r [44] 0.046 0.015 0.473 GT Focal Length + GT Motion Prior LEAP-VO [3] 0.046 0.027 0.385 GT Focal Length + Metric Depth DROID-SLAM [39] 0.043 --GT Motion Priors ParticleSfM [59] ---RGB-Only Robust-CVD [16] 0.153 0.026 3.528 casualSAM [58] 0.071 0.010 1.712 Ours 0.065 0.010 0.987
this section cite: ['b35', 'b43', 'b38']

Section: Time Efficiency Evaluation
In Table 3, we present the average runtime evaluations. Our average runtime on NeRF-DS, DAVIS, and iPhone is 55%, 11%, and 8% of that of the second-fastest methods, while keeping the best performance as shown in table 2. We attribute it to three main reasons: 3) The two-stage optimization strategy highly accelerates the optimization speed. As seen from Table 7, omitting the two-stage strategy leads to a dramatic performance drop after the same iterations, indicating more iterations, and thus time, are needed to achieve the same performance.  Besides, in Figure 4, we see that our method exhibits a linear growth (at the rate of about 1/800 hours per frame) vs COL w/omask whose runtime growth is roughly exponential. This difference will be increasingly significant as the video length increases, which can also demonstrate the superior time efficiency of our method compared with other RGB-only supervised methods. We exclude the casualSAM here since its runtime is too large to fit here.
this section cite: []

Section: Camera Pose Evaluation
We follow the same evaluation setup of Cut3r [44] and Monst3r [56] on TUM-dynamics [36] and evaluate all videos of the synthetic MPI-Sintel [2] dataset.
Quantitative Evaluation. In Table 5 and Table 4, our method has the best performance among all RGB-only supervised approaches. Our method also achieves comparable or even better results than others that require additional GT priors as supervision. We attribute it to our accurate and robust pseudo-supervision, derived from RGB-only input, enabling effective outlier-aware joint optimization. Besides, our uncertainty modeling and loss design effectively down-weight the impact of moving outliers. Since COL w/ mask and COL w/o mask always fail on MPI-Sintel [2], as observed by us and [20,56], we exclude comparisons with them here. However, RGB-only supervised methods including ours perform not very well in some special cases, which is discussed in the limitations.
this section cite: ['b43', 'b35', 'b1', 'b19']

Section: Qualitative Evaluation.
In Figure 6, we show our estimated camera trajectories alongside the GT on MPI-Sintel [2]. Our estimated camera trajectories can perfectly overlap with the GT, which provides qualitative support to the higher accuracies seen in the quantitative results in Table 5.
this section cite: ['b1']

Section: NVS Evaluation
Since NeRF-DS [50], DAVIS [28], and iPhone [7] datasets do not provide GT camera parameters, we follow [6,20,45,19] by inputting camera estimates of different methods into the same 4D reconstruction pipeline -4DGS [47], and evaluate the NVS performance. Such NVS performance reveals the quality of the camera parameter estimation.
this section cite: ['b49', 'b27', 'b6', 'b5', 'b19', 'b44', 'b18', 'b46']

Section: Quantitative Evaluation.
In Table 2, our method is the best on NeRF-DS [50] (long videos w/ little blur, textureless regions, and specular moving objects) and DAVIS [28] (short videos w/ low parallax and rapid object movement), demonstrating our more accurate camera estimates. We skip COL w/ mask and RoDynRF on DAVIS [28] because they are not RGB-only supervised methods and require supervision beyond RGB frames, and have already underperformed compared to ours on NeRF-DS [50]. Besides, our pseudo-supervision extraction built on the PT models [13,12] performs better on low-parallax videos, which remains challenging for the pre-trained depth model [31].
Regarding the iPhone [7] dataset (videos w/ irregular camera movement and object movement), it provides so-called 'GT' camera parameters obtained by Record3D which is a paid mobile app obtaining camera results by LiDAR sensors. However, we observe that such so-called 'GT' camera parameters are occasionally unreliable. As shown in Table 6 and fig. 7, besides being the best among all RGB-only supervised methods, our method can occasionally beat Record3D.
this section cite: ['b49', 'b27', 'b27', 'b49', 'b12', 'b11', 'b30', 'b6']

Section: Qualitative Evaluation.
We evaluate the quality of the rendered RGB images and depth maps in Figure 7, Figure 8, and Figure 5. Beyond superior RGB renderings, our camera estimates yield the highest-quality depth maps, offering more convincing evidence of accurate scene geometry than RGB renderings. It indicates that our estimated camera parameters enable the model to learn the correct dynamic scene representations rather than overfitting to training views. In the first row of Figure 7, ours performs the best (surpassing even Record3D), especially in rendered depth; whereas   in the second row, our method does not match Record3D, but is still better than other RGB-only supervised works. This is because Record3D is not originally designed for dynamic scenes, so when a scene contains larger irregular movements, its performance will be worse (such observations are also supported by the numerical results in Table 6). In contrast, our method is more robust in various scenarios, consistently maintaining high standards. Ablation Study. In Table 7, the loss of any filter results in less robust relations across video, leading to poor camera estimates and NVS performance. Further, the removal of any of Γ, E ACP , or the two-stage strategy will harm the results due to outliers. This indicates that w/o such a strategy, increasing training iterations is a necessary but not sufficient condition for comparable results. We also overcome the limitations of COLMAP [32] by improving the performance of different scene optimization models [53,47] with our camera estimates. As reported in CoTracker3 [12], CoTracker [13] performs worse than CoTracker3. However, in table 8, the performance of our proposed method is nearly independent of building on the particular PT model. This further sup- ports our claim that our patch-wise tracking filters effectively exact only the accurate trajectories as pseudo-supervision.
this section cite: ['b31', 'b52', 'b46', 'b11', 'b12']

Section: Conclusion and Limitation
We proposed a new RGB-only supervised, accurate, and efficient camera parameter optimization method in casually collected dynamic-scene videos. Our method effectively tackles the challenge of precisely and efficiently estimating per-frame camera parameters under the situation of having no additional GT supervision (e.g. GT motion masks, focal length, 3D point clouds, metric depth, and camera poses) other than RGB videos, which is the most common scenario in real-world and consumer-grade reconstructions. Our method may serve as a step towards high-fidelity dynamic scene reconstruction from casually captured videos.
Although our proposed method is currently the most state-of-the-art RGB-only supervised, accurate, and efficient camera parameter optimization method in dynamic scenes, there are still several limitations. We assume a constant focal length throughout the video. While this assumption is reasonable and currently common to SOTA, the task of accurate and efficient camera parameter optimization for dynamic scene videos with zooming effects under RGB-only supervision remains an open problem. Another common challenge for RGB-only supervised methods, not addressed in this paper, is maintaining robustness in scenes dominated by large moving objects. As shown in fig. 9, the screen space is occupied by the moving human and dragon. It is challenging for our method to establish robust and maximally sparse hinge-like relations as accurate pseudo-supervision because most of the extracted trajectories belong to outliers. CasualSAM [58] struggles due to the rapid changes in depth maps from frame to frame, making 3D space alignment difficult. We plan to maintain consistency in our input setup and address these challenges as part of future research.
[56] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825, 2024.
[57] Qiang Zhang, Seung-Hwan Baek, Szymon Rusinkiewicz, and Felix Heide. Differentiable point-based radiance fields for efficient view synthesis. In SIGGRAPH Asia 2022 Conference Papers, pages 1-12, 2022.
[58] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and William T Freeman. Structure and motion from casual videos. In European Conference on Computer Vision, pages 20-37. Springer, 2022.
[59] Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and Yong-Jin Liu. Particlesfm: Exploiting dense point trajectories for localizing moving cameras in the wild. In European Conference on Computer Vision, pages 523-542. Springer, 2022.
this section cite: []

Section: References
Ref_id:b0 Title: Nope-nerf: Optimising neural radiance field with no pose prior Year: (2023)
Ref_id:b1 Title: A naturalistic open source movie for optical flow evaluation Year: (2012)
Ref_id:b2 Title: Leap-vo: Long-term effective any point tracking for visual odometry Year: (2024)
Ref_id:b3 Title: Easi3r: Estimating disentangled motion from dust3r without training Year: (2025)
Ref_id:b4 Title: Georgios Pavlakos, et al. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds Year: (2024)
Ref_id:b5 Title: Colmap-free 3d gaussian splatting Year: (2024-06)
Ref_id:b6 Title: Monocular dynamic view synthesis: A reality check Year: (2022)
Ref_id:b7 Title: Automatic photo pop-up Year: (2005)
Ref_id:b8 Title: Tour into the picture: using a spidery mesh interface to make animation from a single image Year: (1997)
Ref_id:b9 Title: 2d gaussian splatting for geometrically accurate radiance fields Year: (2024)
Ref_id:b10 Title: Stereo4d: Learning how things move in 3d from internet stereo videos Year: (2024)
Ref_id:b11 Title: Cotracker3: Simpler and better point tracking by pseudo-labelling real videos Year: (2024)
Ref_id:b12 Title: Cotracker: It is better to track together Year: (2023)
Ref_id:b13 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b14 Title: Infonerf: Ray entropy minimization for few-shot neural volume rendering Year: (2022)
Ref_id:b15 Title: Robust consistent video depth estimation Year: (2021)
Ref_id:b16 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b17 Title: Grounding image matching in 3d with mast3r Year: (2024)
Ref_id:b18 Title: Self-calibrating 4d novel view synthesis from monocular videos using gaussian splatting Year: (2024)
Ref_id:b19 Title: Robust dynamic radiance fields Year: (2023)
Ref_id:b20 Title: Nerf in the wild: Neural radiance fields for unconstrained photo collections Year: (2021)
Ref_id:b21 Title: Progressively optimized local radiance fields for robust view synthesis Year: (2023)
Ref_id:b22 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b23 Title: Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs Year: (2022)
Ref_id:b24 Title: Nerfies: Deformable neural radiance fields Year: (2021)
Ref_id:b25 Title: Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields Year: (2021)
Ref_id:b26 Title:  Year: (2017)
Ref_id:b27 Title: The 2017 davis challenge on video object segmentation Year: (2017)
Ref_id:b28 Title: D-nerf: Neural radiance fields for dynamic scenes Year: (2021)
Ref_id:b29 Title: Segment anything meets point tracking Year: (2023)
Ref_id:b30 Title: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer Year: (2020)
Ref_id:b31 Title: Structure-from-motion revisited Year: (2016)
Ref_id:b32 Title: Record3d -point cloud animation and streaming Year: (2019)
Ref_id:b33 Title: Very deep convolutional networks for large-scale image recognition Year: (2014)
Ref_id:b34 Title: Flowmap: High-quality camera poses, intrinsics, and depth via gradient descent Year: (2024)
Ref_id:b35 Title: A benchmark for the evaluation of rgb-d slam systems Year: (2012)
Ref_id:b36 Title: Nerfstudio: A modular framework for neural radiance field development Year: (2023)
Ref_id:b37 Title: Raft: Recurrent all-pairs field transforms for optical flow Year: (2020)
Ref_id:b38 Title: Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras Year: (2021)
Ref_id:b39 Title: 3d reconstruction with spatial memory Year: (2024)
Ref_id:b40 Title: Vggt: Visual geometry grounded transformer Year: (2025)
Ref_id:b41 Title: Vggsfm: Visual geometry grounded deep structure from motion Year: (2024)
Ref_id:b42 Title: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction Year: (2021)
Ref_id:b43 Title: Continuous 3d perception model with persistent state Year: (2025)
Ref_id:b44 Title: Recovering 4d world from monocular video Year: (2024)
Ref_id:b45 Title: Dust3r: Geometric 3d vision made easy Year: (2024)
Ref_id:b46 Title: 4d gaussian splatting for real-time dynamic scene rendering Year: (2024)
Ref_id:b47 Title: Point-nerf: Point-based neural radiance fields Year: (2022)
Ref_id:b48 Title: Gs-slam: Dense visual slam with 3d gaussian splatting Year: (2024)
Ref_id:b49 Title: Nerf-ds: Neural radiance fields for dynamic specular objects Year: (2023)
Ref_id:b50 Title: Building animatable 3d neural models from many casual videos Year: (2022)
Ref_id:b51 Title: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass Year: (2025)
Ref_id:b52 Title: Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction Year: (2024)
Ref_id:b53 Title: Volume rendering of neural implicit surfaces Year: (2021)
Ref_id:b54 Title: Samyak Rawlekar, and Narendra Ahuja. Learning implicit representation for reconstructing articulated objects Year: (2024)
