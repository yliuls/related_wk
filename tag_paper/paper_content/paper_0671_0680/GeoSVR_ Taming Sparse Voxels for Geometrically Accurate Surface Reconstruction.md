Title: GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction
Abstract: Geometric Sparse-Voxel Reconstruction.Our method, abbreviated as GeoSVR, delivers high-quality surface reconstruction for intricate real-world scenes based on explicit sparse voxels. Our superiority is exhibited compared to the state-of-the-art approaches built upon Gaussian Splatting, which encounter rough, inaccurate, or incomplete recovery problems even with help from external estimators, excelling in delicate details capturing with high completeness and top-tier efficiency.

Section: Introduction
Surface reconstruction from multi-view images has been a critical long-term problem in computer vision and graphics. In recent years, with the development of Neural Radiance Fields (NeRF) [43], impressive performances [71,59,38,60] have been shown by combining volume rendering with signed distance functions (SDF) to learn implicit fields from input images, yet are mostly computationally expensive. More recently, with the rise of 3D Gaussian Splatting (3DGS) [32], surface reconstruction with explicit sparse representation is making rapid progress [25,28,78,15,11,62], enabling efficient and high-quality geometry learning for a wider range of scenarios.
While significant advancements have been achieved in these 3DGS-based approaches, the methodological limitations are emerging as a bottleneck. One critical problem lies in the reliance on well-structured point clouds initialization. Typically provided by multi-view geometry (MVG) approaches [51,52], the point clouds inevitably contain inaccurate and uncovered regions due to appearance ambiguities, which further aggravates difficulties for 3DGS to refine these challenging areas accurately in geometry, becoming an inherent flaw. This spatial incompleteness further hinders the full potential of rapidly evolving geometry foundation models [17,69,2] in attempts [15,14,36,64], obstructing their ability to drive a quality revolution in surface reconstruction. Another key issue is the lack of clearly defined edges in the Gaussian primitives, making the geometry ambiguous from both the representation clarity [28,55] and the calculation precision trade-offs [55,48,42].
Exploring another possibility, this paper presents GeoSVR that tames sparse voxels to achieve accurate and delicate surface reconstruction. Unlike previous explicit approaches based on 3DGS, we start with a recently proposed SVRaster [55] that combines sparse voxels with rasterization to efficiently refine the scene via level of details. Initialized with fully covered coarse voxels constantly, the full potential can be maintained to model any part inside the scene with completeness. And with clearly bounded voxels, geometric details can be better identified compared to the Gaussians or smooth neural fields. However, while obtaining distinct characteristics, challenges come correspondingly.
Despite competitive surface quality yielded by vanilla SVRaster, significant geometric distortion persists during the sparse voxels optimization, due to the absence of a strong structure prior like the point clouds used in 3DGS, hindering further surface refinement. To fully exploit the strength of the densely covered representation, we resort to the current rapidly evolving and increasingly well-established monocular depth as the geometry cue to provide dense and easy-to-fetch scene constraints. However, a key problem arises for our highly accuracy-required surface reconstruction task: how to effectively utilize this good but not perfect external constraint, while preserving wellreconstructed geometries from being hurt by errors to avoid quality degradation. To address this, we first adopt the patch-wise depth regularizer [35] to facilitate local geometry learning, and based on which, a Voxel-Uncertainty Depth Constraint is proposed, evaluating the geometric uncertainty of each voxel and adaptively determining the degree of reliance on external cues at pixel level. By modulating internal photometric and external depth supervisions for confident and ambiguous regions, our approach enables effective and robust scene constraints, even for well-reconstructed geometries.
Investigating the voxel-based surface formation, we then focus on geometric accuracy refinement and develop Sparse Voxel Surface Regularization. Since the gradients are shared only with the nearest neighbors, challenges exist in composing these extremely local and tiny sparse voxels to ideal surfaces. First, inspired by previous MVG-regularized approaches [20,16,11,50,12], we try to adopt the widely used explicit multi-view geometry constraint [26] to help build geometrically correct surfaces. Nevertheless, the sparse voxel's extreme locality made this plane-based geometry regularization less effective in enforcing a regional geometry constraint. To enlarge the refinement of per voxel, we conduct an interval sample to randomly drop out a portion of voxels to simplify the learned scene during geometry regularization, thus forcing each tiny voxel to keep a global geometry consistency. Second, from the perspective of voxel's surface representation, we introduce two voxelwise regularizations: A Surface Rectification to restrict the surface formation to be aligned with a unique voxel to reduce depth bias; and according to the metioned voxel uncertainty, a Scaling Penalty to eliminate the participation of the geometrically inaccurate large voxel in the surface formation. With the help of both, sharp and accurate surfaces are facilitated in the reconstruction.
In summary, our main contributions are as follows:
• An exploration GeoSVR to build explicit voxel-based framework for accurate surface reconstruction, taming sparse voxels to enable delicate and complete geometry learning.
• A Voxel-Uncertainty Depth Constraint that maximizes the utilization of external depth cue while avoiding quality degradation by the proposed voxel uncertainty evaluation, enabling effective and robust scene constraints for highly accuracy-required surface reconstruction.
• A Sparse Voxel Surface Regularization for surface geometric accuracy refinement, which enlarges the global geometry consistency constraint for tiny voxels and facilitates reconstructing sharp and accurate surfaces by regularizing the voxel-based surface formation.
• Extensive experiments on DTU, Tanks and Temples, and Mip-NeRF 360 datasets demonstrate that the proposed GeoSVR achieves superior performance compared to the state-ofthe-arts in reconstructing accurate surfaces across diverse challenging scenarios, excelling in detail preservation and high completeness while maintaining computational efficiency.
2 Related Works Differentiable Radiance Fields. In recent years, radiance fields have made significant progress in 3D reconstruction by learning scenes directly with differentiable rendering. Neural Radiance Fields (NeRF) [43] is one of the most important foundations, which uses a large MLP to memorize 3D scene and renders through differentiable volume rendering, yet is weak in efficiency. Later, hybird representations come by proposing neural grids [56,45,27], plane decompositions [9,10,18,8], or sparse voxels [19,39,74] with or without neural networks. However, a weakness is that these methods always assume all the grids are uniform in scale, limiting the quality and scalability. More recently, 3D Gaussian Splatting (3DGS) [32] represents radiance fields by a set of anisotropic 3D Gaussians and renders with differentiable splatting using rasterization, achieving remarkably successful balances between fast and high-quality scene reconstruction [40,76,49]. However, due to the complexity of tremendous intersected Gaussians, a view-inconsistent rendering problem is exhibited, and is hard to fix without large efficiency trade-offs [48,42,44]. Also, the reliance on sparse point clouds brings additional uncertainty. To this end, SVRaster [55] combines efficient rasterization with explicit non-uniform sparse voxels to achieve definite, robust, and high-quality scene representation, with less mandatory dependency on structure prior as well. Nevertheless, its potential for accurate geometry learning has not been fully explored, which is an open yet invaluable problem for 3D reconstruction.
this section cite: ['b42', 'b70', 'b58', 'b37', 'b59', 'b31', 'b24', 'b27', 'b77', 'b14', 'b10', 'b61', 'b50', 'b51', 'b16', 'b68', 'b1', 'b14', 'b13', 'b35', 'b63', 'b27', 'b54', 'b54', 'b47', 'b41', 'b54', 'b34', 'b19', 'b15', 'b10', 'b49', 'b11', 'b25', 'b42', 'b55', 'b44', 'b26', 'b8', 'b9', 'b17', 'b7', 'b18', 'b38', 'b73', 'b31', 'b39', 'b75', 'b48', 'b47', 'b41', 'b43', 'b54']

Section: Surface Reconstruction with Learnable Fields.
Reconstructing surfaces from multi-view images has been a long-standing problem. While multi-view stereo-based methods [26,81,52,70] rely on a modular pipeline with multiple decoupled stages, earlier neural approaches [72,47] are proposed to represent surfaces implicitly with an MLP to learn geometry directly from images. Further advancements such as UNISURF [46], NeuS [59], VolSDF [71] represent implicit surfaces by signed distance functions integrated with differentiable volume rendering and achieve better reconstructed details. Based on these, methods with improvements like geometry regularizations [16,77,20,58] and efficient grid representations [38,65] extended the quality and available scenarios. However, the trade-off between training time and quality for complex scenes is still a serious challenge.
More recently, Gaussian-based surface reconstruction has arisen with 3DGS by offering better explicit geometry with much higher efficiency. SuGaR [25] first focuses on extracting Gaussians as mesh surfaces with alignment regularization. Then, more efforts appear by integrating 3DGS with SDF [75,13,41,66,79,36] or improved representations [28,78,15], significantly progressing the surface quality. Specifically, 2DGS [28] and GSurfel [15] propose squeezing Gaussians as 2D surfels for a better aligned surface, and GOF creates an opacity field to allow dense and detailed mesh extraction. However, due to the loose geometry constraint and photometric ambiguities, challenges in accuracy still exist. For the SDF-integrated approaches, since additional MLP and also grid are usually required, problems of over-smoothness [79] and limitations for large unbounded scenes [41] may occur. To conquer the accurate reconstruction on challenging regions, following previous successes [20,16,12], PGSR inherits the idea of surfels and incorporates the multi-view geometry constraint [26], which is widely used in multi-view stereo, to regularize planar accuracy, but the production leans to be over-smooth, due to the unsharp Gaussians similarly in 2DGS [41,75]. Meanwhile, some works [15,14,36,62] attempt to resort to external geometry cues from geometry foundation models [17,2,69] for regularization. However, the performance is still far from what the cues could fully provide, mainly caused by the methodological bottleneck in the strict demand of high-quality initial points, for which the effect of special densification is also limited [11,14]. Instead, this work explores another sparse voxel representation to escape the strict initialization requirement and pursue a clearer geometry representation, achieving superior accurate, detailed, and complete surface reconstruction. 3 Method
this section cite: ['b25', 'b51', 'b69', 'b71', 'b46', 'b45', 'b58', 'b70', 'b15', 'b76', 'b19', 'b57', 'b37', 'b64', 'b24', 'b74', 'b12', 'b40', 'b65', 'b78', 'b35', 'b27', 'b77', 'b14', 'b27', 'b14', 'b78', 'b40', 'b19', 'b15', 'b11', 'b25', 'b40', 'b74', 'b14', 'b13', 'b35', 'b61', 'b16', 'b1', 'b68', 'b10', 'b13']

Section: Preliminaries: Sparse Voxels Rasterization
Representation. Sparse Voxels Rasterization (SVRaster) [55] represents scene with density field based on sparse voxels, which are organized in an Octree of the size w s ∈ R and center w c ∈ R 3 . Each voxel keeps a set of SH coefficients v sh for voxel color, and densities v geo ∈ [0, +∞] 2×2×2 , separately on the eight voxel corners to model a trilinear inside density field for geometry. A voxel is identified with the index v = {i, j, k} at Octree level l, and its size v s and center v c are given as:
v s = w s × 2 -l , v c = w c -0.5 × w s + v s × v (1
) Rendering. During rendering, SVRaster adopts α-blending similar to NeRF and 3DGS. Inside each voxel, SVRaster evenly samples K points in the ray segment of length ∆t between the ray-voxel intersections, and composes voxel-wise α with trilinear interpolation interp(•) by volume rendering. Then, the α-blending is available to render the pixel-wise color C that corresponds to the ray:
C = N i=1 T i α i c i , T i = i-1 j=1 (1 -α j ) ; α = 1 -exp(- ∆t K K k=1 interp(v geo , q k )), (2
)
where α i and c i are alpha and view-dependent color of the i-th intersected voxel, and q k is the local position of the k-th sample point in the voxel. According to α-blending, we can render the pixel-wise normal N and depth D. The pixel-wise depth D can be given similarly via per-point distance rendering. For voxel's normal, the analytical gradient is calculated at the voxel center q c :
n = normalize (∇ q interp (v geo , q c )) .(3)
Adaptive Octree Control. To adaptively adjust the scene Octree during training, SVRaster prunes the voxels with the least blending weight T α, and accumulates an α-weighted priority based on loss gradients to select the voxels that need to be subdivided to the next level to represent finer details.
Challenges in Surface Reconstruction. Despite strengths in geometric completeness and clarity, challenges exist correspondingly: 1) With little native constraint, the optimization often encounters heavy geometry distortion and blocks further improvement.
2) The impact scope of a single voxel could be quite local, which is unfavorable to accurate surface formation. Exploring tackling these two challenges, we present GeoSVR for high-quality voxel-based surface reconstruction, as in Figure 2 3
this section cite: ['b54']

Section: .2 Voxel Geometric Uncertainty for Scene Constraint
Unlike previous approaches based on SDF [20,77] or 3DGS [28,78,14,11] that benefit from the structure constraints from geometric [1] or sparse points initialization [32], the highly expressive and constant-initialized sparse voxels require an essential scene constraint to effectively ensure the geometry converges to approximately correct surfaces, preparing for a further accuracy refinement.
Problem in Monocular Depth Cue. Inspired by previous works [77,64], we turn attention to the increasingly well-established monocular depth [5,21,67,69], which provides dense, efficient, and full-time available constraints for scene geometry optimization. Moreover, this dense cue natively matches the spatially complete voxels to fulfill its potential for compensating appearance ambiguities.
However, the problem of how to maximally utilize this attractive but not perfectly accurate prior in the highly accuracy-required surface reconstruction remains a long-standing difficulty. Despite considerable relevant studies [77,58,15,14,64,62,36], a solution is still absent to evaluate the learned geometry's confidence to determine external cue reliance, which causes only over-conservative strategies to be available, but could still degrade the quality by the included errors [77,14].
this section cite: ['b19', 'b76', 'b27', 'b77', 'b13', 'b10', 'b0', 'b31', 'b76', 'b63', 'b4', 'b20', 'b66', 'b68', 'b76', 'b57', 'b14', 'b13', 'b63', 'b61', 'b35', 'b76', 'b13']

Section: Voxel Geometric Uncertainty.
In this work, we aim to solve the problem by evaluating geometric uncertainty from the representational capability: 1) In SVRaster, each voxel contains a trilinear density field to represent the geometry of the cube space with length of v s = w s × 2 -l . Then, the accuracy for an under-captured geometry is strictly limited, negatively related to the level l of corresponding voxels. 2) During optimization, SVRaster progressively subdivides voxels at l with largest gradients to the next level l + 1. Consequently, the voxels at lower levels denote either regions with fewer texture constraints or less view coverage, both associated with high uncertainties.
Inspired by these two tight couplings of uncertainty and voxel's level, we abstract a level-aware geometric uncertainty that explicitly correlates with Octree level l to guide identifying scene constraint targets. For a voxel v at level l, its base and geometric uncertainties U base and U geom are given by:
U base (l) = w s β(l + l 0 ) , U geom (v) = U base (l) • (1 -exp (-v geo )) ,(4)
where β is a scaling factor, combined with Octree size w s as global scene scale. l 0 is the starting level. Geometric uncertainty U geom (v) is composed of the level-dependent base uncertainty U base and the voxel density, indicating a voxel at low level with critical geometry leads to higher uncertainty. Derived in the Appendix Section B, we simplify powers and exponents while preserving the same trend to prevent numerical blowup in later applying.
this section cite: []

Section: Voxel-Uncertainty Depth Constraint.
Based on the uncertainty, we next design the constraint to enable effective and reliable monocular depth integration. To effectively apply the monocular depth as supervision, we resort to a patch-wise global-local depth loss [35] for better scale alignment and facilitating the geometry knowledge learning. Then, integrating the geometric uncertainty into the pixel-wise constraint, we first render an Octree level map L for efficient pixel-wise uncertainty calculation, directly gathering the volume density term of Eq. ( 4) with α of Eq. ( 2) via rasterization:
L = N i=1 T i α i l i , α = 1 -exp(- ∆t K K k=1 interp(v geo , q k )).(5)
Next, converting uncertainty to weight, we produce a pixel-wise modulation on depth constraint. To ensure adaptive and robust constraints for various stages and scenarios, we obtain statistics of L to set the hyperparameters in Eq. ( 4). Specifically, for scale-independence, let the scale term of w s /β equal to per-view global level scale w l = max(L) -min(L), and set l 0 = -min(L) to define the coarsest level of the view. Then, derived from U geom , the geometry uncertainty weight W unc follows:
W unc = w l max(1, L -min(L)) , w l = max(L) -min(L)(6)
Finally, given the estimated monocular depth D as the constraint for the rendered depth D, W unc is applied to the patch-wise depth loss L D-patch [35] for per-pixel constraint reweight:
L D-unc (D, D) = W unc • L D-patch (D, D).(7)
As a result, Voxel-Uncertainty Depth Constraint L D-unc pays minimal attention to the voxels with low uncertainty to be confident of the native photometric constraint, while enhancing highly uncertain ones to rely on external cue for solving geometry ambiguities. The effect can be illustrated in Figures 2 and 6, where level map L is shown to obtain a more uniform range of values for better visual effect.
this section cite: ['b34', 'b34']

Section: Sparse Voxel Surface Regularization
Despite the scene constraint exerted, a coarsely correct reconstruction does not exhibit the full potential of sparse voxels. Therefore, we next investigate the capability of sparse voxels for highly accurate surface formation under explicit geometry constraint and finer voxel-level regularizations.
this section cite: []

Section: Geometry Regularization with Voxel Dropout.
Serving as an explicit and strict constraint, homography patch warping has shown great effect in classical MVS [22,53,81] and recent related works [20,16,11,50,12,58], which we also try to apply in our method. Typically, considering a source view and a reference view with image I s and I r , we warp the image point x ′ in the pixel patch P of I s to the image point x in I r of the reference view by the plane-induced homography H [53]:
x = Hx ′ , H = K s (R s R T r + R s (R T s t s -R T r t r )n T n T p )K -1 r , (8
)
where p is the intersected 3D point calculated from depth D, and the normal n is from Eq. 3. K is camera intrinsics, and [R, t] is the extrinsics of each view. Then, an occlusion-aware NCC loss [11] is applied between the warped P and its target in I r . However, we observe that despite improvements brought, this technique does not work as ideally as in previous approaches. Due to the extreme locality of the tiny voxels that connect to only the nearest neighbors by a few corners, the planar constraint becomes less effective, leading to redundant wrong structures being produced.
To solve this problem, our idea is to enlarge the regularization for each voxel by breaking these incorrectly organized geometries, enforcing the tiny voxels to obey a more global geometry consistency instead of only their own tiny scopes. During the process, we conduct an interval sample of the voxels with a random ratio in [γ, 1] while calculating the full-scale depth D and normal N. Therefore, only a subset of voxels is used to represent the scene, while the others are temporally dropped out. Then, the regularization enforces each voxel to respond to the geometry consistency of a larger area, including where the dropped-out voxels belong, for a forced break and correction of the ill geometries. Surface Rectification. Subsequently, we focus on the bias between the trilinear voxel density field and the weight contribution in rendering, which causes misaligned surfaces from rendering and voxel density. As in Figure 3, due to the trilinear local-linked voxel fields, the density increase of one voxel will implicate the neighbors, resulting in decentralized densities that makes the highest rendering weight w biased to the side regions but not the correct highest density position, like Then, for these voxels, we penalize the density at p e but encourage at p o to form a sharp segmentation of the surface and empty spaces, with a penalty term including the voxel's rendering contribution w:
R rec = w • I(v ∈ V s ) • (interp(v geo , p e ) -interp(v geo , p o )), w = T α.(10)
Since then, the surface in rendering can be rectified to be aligned to the density, as in Figure 3 a.2.
Scaling Penalty. Inspired by the voxel geometric uncertainty in Sec. 3.2, we present a simple yet effective regularizer that penalizes the voxels occupying a long sampling distance, which denotes a less accurate geometry modeling. Normalized with globally minimal voxel size min(v s ), it follows:
R sp = w • interp(v geo , q c ) • max(0, log 2 ( ∆t min(v s )
)), where q c = (0.5, 0.5, 0.5).
this section cite: ['b21', 'b52', 'b19', 'b15', 'b10', 'b49', 'b11', 'b57', 'b52', 'b10']

Section: Loss Function
The total objective is composed of the photometric loss L photo from SVRaster, the depth constraint L D-unc from Eq. ( 7), NCC loss for geometry regularization, and the voxel regularizations in Sec. 3.3:
L = L photo + ηL D-unc + τ L NCC + µ 1 R rec + µ 2 R sp .(12)
In this work, we set the weights of η = 0.1, τ = 0.01, µ 1 = 10 -5 , and µ 2 = 10 -6 , respectively.  4 Experiments Implementation Details. Our code is implemented with PyTorch and CUDA kernels, built upon SVRaster [55]. In the experiments, we train each model with 20, 000 iterations, with the learning rates for density and SHs at degree 0 and the others of 0.05, 0.01, and 0.00025 in Adam [33] optimizer. We use DepthAnythingV2 [69] to provide the depth cues. The patch size of 7 × 7 is used for patch warping, and γ in voxel dropout is set to 0.5 and 0.3 for DTU and TnT datasets. The Octree setups keep the same as in [55], and the prune interval is increase to 2, 000 for finer expression. In our method, we use TSDF for mesh extraction. All experiments are conducted on RTX 3090 Ti GPUs.
this section cite: ['b54', 'b32', 'b68', 'b54']

Section: Comparision
Dataset. We use the prevailing DTU, Tanks and Temples (TnT), and Mip-NeRF 360 datasets for evaluation. The scene selections of DTU and TnT are consistent with previous works [71,59,38,28], preprocessed following 2DGS [28] and Neuralangelo [38]. The voxel size of TSDF is set to 0.002 for DTU and is calculated for TnT following PGSR [11]. The images in DTU and TnT are downsampled 2×, and in Mip-NeRF 360 are downsampled 2× or 4× following [32] for indoor and outdoor scenes.
Baselines. We take the state-of-the-art surface reconstruction approaches as baselines, including implicit (e.g., NeuS [59], Neuralangeo [38], Geo-NeuS [20]) and explicit methods (e.g., 2DGS [28], GOF [78], PGSR [11]). Among them, MonoSDF [77], GSurfel [15], VCR-GauS [14], GS2Mesh [62], and MonoGSDF [36] take external geometry cues from pre-trained depth and/or normal models for regularization. Basic representations like 3DGS [32] and SVRaster [55] are also included.
this section cite: ['b70', 'b58', 'b37', 'b27', 'b27', 'b37', 'b10', 'b31', 'b58', 'b37', 'b19', 'b27', 'b77', 'b10', 'b76', 'b14', 'b13', 'b61', 'b35', 'b31', 'b54']

Section: Surface Reconstruction.
To evaluate surface reconstruction performance, we make comparisons on the DTU and TnT datasets with Chamfer distance and F1-score. The results are reported in Table 1 and   On the two datasets, our method also retains fast training comparable to the 3DGS-based methods. In Figure 4 and 5, we visualize the reconstructed meshes from ours and competitive baselines. Our productions obtain both the best accuracy and completeness. And due to the basis of the initial prior-free and densely covered sparse voxels, GeoSVR can handle the reflective regions and areas with insufficient coverage, where the 3DGS-based methods are limited, due to insufficient initialization points. Additionally, GeoSVR performs better than previous geometry cue-reliant methods (e.g., VCR-GauS) that may lead to oversmoothing and underfitting.
Appearance Reconstruction. Achieving accurate geometry reconstruction, our method maintains the capability for high-quality novel view synthesis as well. In Table 3, we compare our methods on the Mip-NeRF 360 dataset with the baselines in the aspect of rendering quality. Our method exhibits competitive performance among the surface reconstruction methods as well as the NVSspecific baselines such as our basis SVRaster. Due to the lack of geometry ground-truth, we do not evaluate the surface reconstruction quality. Qualitative comparisons can be found in the Appendix.
this section cite: []

Section: Ablation Study
In this section, we verify the effect of our designs on the Tanks and Temples [34] dataset and report the mesh reconstruction metrics. The quantitative scores are reported in Table 4. As references, the reproduced SVRaster and PGSR with TSDF are reported in the comparison. Additionally, we summarize the baselines with external cues in Table 5 to exhibit the effect of the methodology itself. Scene Constraint. Scene constraint dominates an essential start for further refinement. In Table 4, we observe that regularizations of sparse depth from SfM points and monocular depth with inverse loss both help less, while the patch-wise depth loss of Table 4 B breaks through to improve the geometry effectively. A step further, even though the reconstruction already achieves a high quality (0.552 in F1), our Voxel-Uncertainty Depth Constraint still remarkably recognizes the uncertain regions and refines the geometry and preserving the well-reconstructed parts, as shown in Figure 6 and Table 4 E.
this section cite: ['b33']

Section: Multi-view Regularization.
Based on the scene constraint, we then analyse the multi-view regularization part. Consistent with the conclusions like in previous works [20,16,11], adding the explicit multi-view geometry objective can hugely improve the geometry accuracy, yet by relieving the local trap of sparse voxels, our Voxel Dropout strategy further improves the multi-view consistency to a higher level and exceeds the patch-warping regularized reference method with monocular depth.
this section cite: ['b19', 'b15', 'b10']

Section: Voxel Regularization.
To further facilitate the surface-depth consistency, we apply the Surface Rectification and Scaling Penalty for the voxels to get finer surfaces. As shown in Table 4 D and Figure 7, the voxel regularization designs benefit the formation of accurate surfaces from the perspective of voxel-based representation, therefore improving the geometry both quantitatively and qualitatively.
this section cite: []

Section: Conclusion
In this work, we have presented GeoSVR, an explicit voxel-based framework that explores and extends the under-investigated potential of sparse voxels to deliver accurate, detailed, and complete surface reconstruction with high efficiency. Our study first analyzes voxel uncertainty in geometry representation to distinguish the confidence of learned geometry, enabling effective and robust scene constraint from external cues. Next, we investigate the problem of voxel-based surface refinement, reconstructing surfaces with superior quality by our solution. In the future, it will be interesting to explore enhancing voxel's globality to conquer challenges like varying lights and textureless regions.
[80] Haoliang Zhao, Huizhou Zhou, Yongjun Zhang, Jie Chen, Yitong Yang, and Yong Zhao. Highfrequency stereo matching network. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1327-1336, 2023.
[81] Enliang Zheng, Enrique Dunn, Vladimir Jojic, and Jan-Michael Frahm. Patchmatch based joint view selection and depthmap estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1510-1517, 2014.
this section cite: []

Section: References
Ref_id:b0 Title: Sal: Sign agnostic learning of shapes from raw data Year: (2020)
Ref_id:b1 Title: Rethinking inductive biases for surface normal estimation Year: (2024)
Ref_id:b2 Title: Mipnerf 360: Unbounded anti-aliased neural radiance fields Year: (2022)
Ref_id:b3 Title: Stereo anywhere: Robust zero-shot deep stereo matching even where either stereo or mono fail Year: (2025)
Ref_id:b4 Title: Midas v3. 1-a model zoo for robust monocular relative depth estimation Year: (2023)
Ref_id:b5 Title: Depth pro: Sharp monocular metric depth in less than a second Year: (2024)
Ref_id:b6 Title: Raysplats: Ray tracing based gaussian splatting Year: (2025)
Ref_id:b7 Title: Hexplane: A fast representation for dynamic scenes Year: (2023)
Ref_id:b8 Title: Efficient geometryaware 3d generative adversarial networks Year: (2022)
Ref_id:b9 Title: Tensorf: Tensorial radiance fields Year: (2022)
Ref_id:b10 Title: Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction Year: (2024)
Ref_id:b11 Title: Recovering fine details for neural implicit surface reconstruction Year: (2023)
Ref_id:b12 Title: Neural implicit surface reconstruction with 3d gaussian splatting guidance Year: (2023)
Ref_id:b13 Title: Vcrgaus: View consistent depth-normal regularizer for gaussian surface reconstruction Year: (2024)
Ref_id:b14 Title: Highquality surface reconstruction using gaussian surfels Year: (2024)
Ref_id:b15 Title: Improving neural implicit surfaces geometry with patch warping Year: (2022)
Ref_id:b16 Title: Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans Year: (2021)
Ref_id:b17 Title: K-planes: Explicit radiance fields in space, time, and appearance Year: (2023)
Ref_id:b18 Title: Plenoxels: Radiance fields without neural networks Year: (2022)
Ref_id:b19 Title: Yew Soon Ong, and Wenbing Tao. Geo-neus: Geometry-consistent neural implicit surfaces learning for multi-view reconstruction Year: (2022)
Ref_id:b20 Title: Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image Year: (2024)
Ref_id:b21 Title: Accurate, dense, and robust multiview stereopsis Year: (2009)
Ref_id:b22 Title: Relightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing Year: (2024)
Ref_id:b23 Title: Irgs: Inter-reflective gaussian splatting with 2d gaussian ray tracing Year: (2024)
Ref_id:b24 Title: Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering Year: (2024)
Ref_id:b25 Title: Multiple view geometry in computer vision Year: (2003)
Ref_id:b26 Title: Trimiprf: Tri-mip representation for efficient anti-aliasing neural radiance fields Year: (2023)
Ref_id:b27 Title: 2d gaussian splatting for geometrically accurate radiance fields Year: (2024)
Ref_id:b28 Title: Transparentgs: Fast inverse rendering of transparent objects with gaussians Year: (2025)
Ref_id:b29 Title: Mvsanywhere: Zero-shot multi-view stereo Year: (2025)
Ref_id:b30 Title: Large scale multi-view stereopsis evaluation Year: (2014)
Ref_id:b31 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b32 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b33 Title: Tanks and temples: Benchmarking large-scale scene reconstruction Year: (2017)
Ref_id:b34 Title: Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization Year: (2024)
Ref_id:b35 Title: Monogsdf: Exploring monocular geometric cues for gaussian splatting-guided implicit surface reconstruction Year: (2024)
Ref_id:b36 Title: Tsgs: Improving gaussian splatting for transparent surface reconstruction via normal and de-lighting priors Year: (2025)
Ref_id:b37 Title: Neuralangelo: High-fidelity neural surface reconstruction Year: (2023)
Ref_id:b38 Title: Neural sparse voxel fields Year: (2020)
Ref_id:b39 Title: Scaffoldgs: Structured 3d gaussians for view-adaptive rendering Year: (2024)
Ref_id:b40 Title: 3dgsr: Implicit surface reconstruction with 3d gaussian splatting Year: (2024)
Ref_id:b41 Title: Ever: Exact volumetric ellipsoid rendering for real-time view synthesis Year: (2024)
Ref_id:b42 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b43 Title: 3d gaussian ray tracing: Fast tracing of particle scenes Year: (2024)
Ref_id:b44 Title: Instant neural graphics primitives with a multiresolution hash encoding Year: (2022)
Ref_id:b45 Title: Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction Year: (2021)
Ref_id:b46 Title: Deepsdf: Learning continuous signed distance functions for shape representation Year: (2019)
Ref_id:b47 Title: Stopthepop: Sorted gaussian splatting for view-consistent real-time rendering Year: (2024)
Ref_id:b48 Title: Octreegs: Towards consistent real-time rendering with lod-structured 3d gaussians Year: (2024)
Ref_id:b49 Title: Improving neural surface reconstruction with feature priors from multi-view images Year: (2024)
Ref_id:b50 Title: Structure-from-motion revisited Year: (2016)
Ref_id:b51 Title: Pixelwise view selection for unstructured multi-view stereo Year: (2016)
Ref_id:b52 Title: Accurate multiple view 3d reconstruction using patch-based stereo for large-scale scenes Year: (2013)
Ref_id:b53 Title: Dropout: a simple way to prevent neural networks from overfitting Year: (2014)
Ref_id:b54 Title: Sparse voxels rasterization: Real-time high-fidelity radiance field rendering Year: (2025)
Ref_id:b55 Title: Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction Year: (2022)
Ref_id:b56 Title: Dn-splatter: Depth and normal priors for gaussian splatting and meshing Year: (2025)
Ref_id:b57 Title: Neuris: Neural reconstruction of indoor scenes using normal priors Year: (2022)
Ref_id:b58 Title: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction Year: (2021)
Ref_id:b59 Title: Neurodin: A two-stage framework for high-fidelity neural surface reconstruction Year: (2024)
Ref_id:b60 Title: Foundationstereo: Zero-shot stereo matching Year: (2025)
Ref_id:b61 Title: Gs2mesh: Surface reconstruction from gaussian splatting via novel stereo views Year: (2024)
Ref_id:b62 Title: 3dgut: Enabling distorted cameras and secondary rays in gaussian splatting Year: (2025)
Ref_id:b63 Title: Surface reconstruction from 3d gaussian splatting via local structural hints Year: (2024)
Ref_id:b64 Title: Voxurf: Voxel-based efficient and accurate neural surface reconstruction Year: (2022)
Ref_id:b65 Title: Gsurf: 3d reconstruction via signed distance fields with direct gaussian supervision Year: (2024)
Ref_id:b66 Title: Depth anything: Unleashing the power of large-scale unlabeled data Year: (2024)
Ref_id:b67 Title: Depth anything: Unleashing the power of large-scale unlabeled data Year: (2024)
Ref_id:b68 Title: Depth anything v2 Year: (2024)
Ref_id:b69 Title: Mvsnet: Depth inference for unstructured multi-view stereo Year: (2018)
Ref_id:b70 Title: Volume rendering of neural implicit surfaces Year: (2021)
Ref_id:b71 Title: Multiview neural surface reconstruction by disentangling geometry and appearance Year: (2020)
Ref_id:b72 Title: Absgs: Recovering fine details in 3d gaussian splatting Year: (2024)
Ref_id:b73 Title: Plenoctrees for real-time rendering of neural radiance fields Year: (2021)
Ref_id:b74 Title: Gsdf: 3dgs meets sdf for improved neural rendering and reconstruction Year: (2024)
Ref_id:b75 Title: Mip-splatting: Alias-free 3d gaussian splatting Year: (2024)
Ref_id:b76 Title: Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction Year: (2022)
Ref_id:b77 Title: Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes Year: (2024)
Ref_id:b78 Title: Neural signed distance function inference through splatting 3d gaussians pulled on zero-level set Year: (2024)
