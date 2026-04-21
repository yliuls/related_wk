Title: TGA: True-to-Geometry Avatar Dynamic Reconstruction
Abstract: Recent advances in 3D Gaussian Splatting (3DGS) have improved the visual fidelity of dynamic avatar reconstruction. However, existing methods often overlook the inherent chromatic similarity of human skin tones, leading to poor capture of intricate facial geometry under subtle appearance changes. This is caused by the affine approximation of Gaussian projection, which fails to be perspective-aware to depth-induced shear effects. To this end, we propose True-to-Geometry Avatar Dynamic Reconstruction (TGA), a perspective-aware 4D Gaussian avatar framework that sensitively captures fine-grained facial variations for accurate 3D geometry reconstruction. Specifically, to enable color-sensitive and geometry-consistent Gaussian representations under dynamic conditions, we introduce the Perspective-Aware Gaussian Transformation that jointly models temporal deformations and spatial projection by integrating Jacobian-guided adaptive deformation into the homogeneous formulation. Furthermore, we develop Incremental BVH Tree Pivoting to enable fast frame-by-frame mesh extraction for 4D Gaussian representations. A dynamic Gaussian Bounding Volume Hierarchy (BVH) tree is used to model the topological relationships among points, where active ones are filtered out by BVH pivoting and subsequently re-triangulated for surface reconstruction. Extensive experiments demonstrate that TGA achieves superior geometric accuracy. Project

Section: Introduction
The demand for personalized, high-fidelity reconstruction of human faces and heads under complex facial movements is fundamental to a wide range of applications, including digital twins, film production, graphics simulation, and entertainment. In particular, acquiring dynamic and geometrically accurate 3D head reconstructions from multi-view recordings is a common requirement for generating digital and virtual replicas of real individuals.
Despite recent advances in high-fidelity 3D Gaussian Splatting (3DGS) [1]-based avatar reconstruction, existing methods [2,3,4] often overlook the inherent chromatic similarity of human skin tones. Specifically, under varying viewpoints and frame transitions, subtle and gradual facial expression changes remain challenging for vanilla 3DGS. The root cause lies in the limited perspective-awareness (we will discuss in Sec. 3.2) of affine approximation used in Gaussian projection, which compromises accurate color blending. As a result, the simplistic projection model struggles to maintain chromatic consistency and further geometric fidelity for dynamic face modeling. In response, we propose True-to-Geometry Avatar Dynamic Reconstruction (TGA), a perspectiveaware 4D Gaussian avatar framework that sensitively captures fine-grained facial appearance variations to enable geometrically accurate 3D mesh reconstruction from 4D Gaussian representations.
We first introduce a Perspective-Aware Gaussian Transformation that jointly models temporal deformations and spatial projection effects, enhancing perspective-awareness to subtle changes in avatar facial appearance. Traditional 3DMM [5,6]-based Gaussian methods [2,3,4,7] warp primitives based on the area of their parent triangle across time steps. Such uniform scaling leads to either extending or shrinking Gaussian coverage, resulting in over-or under-blending of colors in facial regions. To address this, we apply Jacobian-guided deformation to adaptively warp each Gaussian according to the directional variation of its parent triangle, ensuring precise color coverage across dynamic frames. Furthermore, to be perspective-aware to the projection process for better capture of the intricate geometry under subtle changes in avatar skin tone and facial expressions, we adopt a homogeneous formulation in place of the insufficient affine approximation in vanilla 3DGS, enabling perspective-consistent projections. The Jacobian-guided deformation is jointly incorporated into the homogeneous formulation. As a result, color blending becomes more chromaticity-sensitive, reinforcing geometry-color alignment and providing a reliable foundation for downstream mesh extraction.
Based on the obtained Gaussian avatar field, we introduce an opacity field [8] for surface extraction. The geometrically accurate avatar surface is extracted by directly identifying a level set of an opacityguided signed distance field on tetrahedra, which are triangulated from Gaussians and their bounding points. To enable fast dynamic mesh extraction, we design a straightforward yet effective Incremental BVH Tree Pivoting approach to adaptively update the tetrahedral grids. Specifically, we dynamically organize the bounding volume hierarchy (BVH) of Gaussians in a binary tree, which models the topological relationships among primitives. As the BVH tree updates by branch rotation according to Gaussian dynamics, the BVH is pivoted to simulate the topological movements of Gaussians. Active 3D points-referred to as hopping Gaussians-that contribute to facial expression changes, are filtered out by BVH Pivoting. These filtered regions are then incrementally triangulated, thereby accelerating the surface extraction process.
Overall, our contributions are the following:
• We propose a perspective-aware 4D Gaussian avatar framework that captures intricate geometry under subtle facial variations, by integrating Jacobian-guided adaptive deformation with homogeneous projection to enable Perspective-Aware Gaussian Transformation.
• We design an Incremental BVH Tree Pivoting approach, which filters out and re-triangulates hopping Gaussians. This enables fast and adaptive surface extraction by focusing computation on active regions undergoing facial expression changes.
• We empirically demonstrate the advanced performance of the proposed method, demonstrating significant improvements in reconstruction accuracy, dynamic capability, training efficiency, and inference time. 2 Related Work 3D Morphable Face Models. Parametric template models based on PCA have become fundamental in computer graphics and vision for representing human body geometry, including the face [6,9,10] and head [5,11], with extensions to the neck [12] and the full body [13]. To overcome the rigid linearity of PCA, more recent approaches [14,15,16,17,18,19,20,21] replace traditional PCAbasis underlying classical mesh-based 3DMMs. Furthermore, neural-based methods [22,23,24] enhance expression realism with continuous, implicit morphable representations of geometry.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b1', 'b2', 'b3', 'b6', 'b7', 'b5', 'b8', 'b9', 'b4', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23']

Section: Dynamic Avatar Representations.
Avatars have inherent dynamics, especially when performing actions such as smiling or speaking, which are accompanied by significant topological changes. This makes the representation of dynamic scenes more complex and challenging. Neural Radiance Fields (NeRFs) [25,26,27]-based methods [25,28,29,30,31,32,33] can capture temporal changes and model such dynamics but not computationally feasible. 3DGS [1]-based dynamic methods [34,35,36,37,38,39,40] have emerged as a more efficient alternative, but remain insensitive to chromatic variations due to their affine approximation. Building upon the 3DGS paradigm, we enhance it with a Perspective-Aware Gaussian Transformation module for improved dynamic modeling.
Human Head Reconstruction. Previous works [41,42,43,44,45] have explored NeRF-based volume rendering to model avatar heads with detailed appearance. Recent approaches [46,47,48,3,49,40,50] incorporate implicit deformation fields to capture frame-wise Gaussian motion. Meanwhile, another line of work [51,52,2,53,4] explicitly rig Gaussians to 3DMM-based meshes for controllable facial animation. Topo4D [54] further reconstructs dynamic meshes and high-fidelity textures via topology-bound Gaussians, NPGA [55] leverages neural parametric head models [56] for learned forward deformations, ScaffoldAvatar [57] employs patch-based expressions with hierarchical Gaussian splatting for high-fidelity avatars. While most 3DGS-based methods target photorealistic rendering and animation, Topo4D [54], SurFhead [4], and our work focus on geometry-accurate facial mesh reconstruction. Specifically, our method builds upon the explicit 3DMM-3DGS binding and deformation framework for better controllability.
this section cite: ['b24', 'b25', 'b26', 'b24', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b0', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b2', 'b48', 'b39', 'b49', 'b50', 'b51', 'b1', 'b52', 'b3', 'b53', 'b54', 'b55', 'b56', 'b53', 'b3']

Section: Method
With the goal of achieving geometrically accurate 4D avatar reconstruction in our mind, we first focus on capturing subtle variations in facial appearance while ensuring deformation consistency across frames (Sec. 3.2). By leveraging trained Gaussians with high sensitivity to facial features, our approach further enables fast and precise avatar mesh extraction (Sec. 3.3). We start with the overview of GaussianAvatars [2] and also define the main symbols(Sec. 3.1).
this section cite: ['b1']

Section: Preliminary
Representation. GaussianAvatars [2] associates each triangle of the FLAME mesh [5], tracked by VHAP [58], with a 3D Gaussian and moves coherently with its corresponding parent-triangle across time steps. Specifically, each Gaussian is parameterized by a center position µ, a positivedefinite, diagonal matrix scaling matrix S, and a rotation matrix R on the neutral FLAME mesh [5]:
G(x) = e -1 2 (x-µ) ⊤ Σ -1 (x-µ)
where Σ = RS 2 R ⊤ . Besides, the Gaussian primitive has appearance properties, a prior opacity α and color c. To deform the canonical 3D Gaussian to the posed space, the vanilla 3DMM-Gaussians method transforms its position and covariance attributes according to its parent:
R ′ = rR µ ′ = krµ + t S ′ = kS,(1)
where the isotropic scalar k is derived from the triangle's relative extent, t and r are the barycenter and relative rotation matrix of the parent-triangle, respectively.
Rendering. Given the 3D representation θ = {µ, Σ, α, c}, the trainable parameters are optimized through the following differentiable rendering function
C(p) = N n=1 c n α n D(µ, Σ, γ) n-1 m=1 (1 -α m D(µ, Σ, γ)),(2)
where C(p) is the rendering color at pixel p of rendered image C, and D(µ, Σ, γ) is a divergence of view ray γ from θ computed from the projected 2D Gaussians by EWA volume splatting [59].
this section cite: ['b1', 'b4', 'b57', 'b4', 'b58']

Section: Perspective-Aware Gaussian Transformation
Our goal is to capture intricate geometric details under subtle variations in avatar skin tone and facial expression. Previous 3DGS-based methods [2,3,4,7] rely solely on affine approximations when projecting Gaussians onto 2D image planes. However, as illustrated by the blue circle in Fig. 3, this naive projection leads to unreliable alpha blending, resulting in chromatic inconsistencies and spatial distortions. To address this issue, it is essential to revisit the projection mechanism from the perspective of Perspective Awareness.
As illustrated by the affine-approximated projected blue Gaussian in Fig. 3, even though its center is correctly splatted using a perspective projection (as shown in 1), the outermost isocontour remains significantly misaligned. This misalignment arises because the affine projection naively projects the Gaussian orthogonally onto the image plane, neglecting the depth information in the Gaussian covariance. Such a simplification fails to be perspective-aware enough to subtle chromatic variations.
To achieve true perspective awareness, it is imperative to 'look into' the Gaussian itself and fully exploit its depth dimension-only then can we faithfully simulate perspective projection.
Specifically, we first introduce a Jacobian-based deformation mechanism to adaptively warp Gaussians across frames. This deformation is integrated into a homogeneous formulation that preserves depth information, enabling a perspective-aware projection. As a result, the model can sensitively capture intricate geometric details driven by subtle chromatic variations in facial expressions.
this section cite: ['b1', 'b2', 'b3', 'b6']

Section: Jacobian Gradient for Adaptive Deformation.
While the rigging method in Eq. 1 is computationally efficient, it struggles to preserve the chromatic consistency across frames. Specifically, its rigid and linear isotropic scaling uniformly extends or shrinks Gaussians according to their parent-triangle's property (Fig. 3 (a)), leading to under-blending discontinuity across frames. To address this, inspired by [60], we introduce an advanced warping method for Gaussians:
JE = Ê J = ÊE -1 , (3
)
where E is composed of edge direction vectors of the binding triangle, and Ê is deformed by Eq. 1.
Then, Gaussians are warped as follows:
Σ = JRSS ⊤ R ⊤ J ⊤ µ ′ = Jµ + t.(4)
By doing so, Gaussians gain adaptive and anisotropic deformability (Fig. 3 (b)) and maintain chromatic consistency across frames, making them ready to capture subtle variations.
this section cite: ['b59']

Section: Homogeneous Formulation for Gaussian Projection.
To be perspective-aware for Gaussian depth dimension and further capture subtle changes in avatar skin tone and facial expressions, we extend prior methods [61,62] originally designed for 2D Gaussians by introducing a homogeneous formulation that replaces the affine approximation for 3D Gaussians. This accurately models the outermost isocontour under perspective projection and further enables high blending sensitivity to fine-grained facial appearance changes (green circle in Fig. 3.( 2)). The homogeneous transformation for 3D Gaussian to be normalized in a local tangent plane is:
H = s u r u s v r v s w r w µ 0 0 0 1 = RS µ 0 1 . (5
)
To jointly account for non-uniform deformations driven by facial expressions across frames, we further refine the Gaussian's in the local metric through a Jacobian-based transformation. This enables adaptive Gaussian deformation while preserving global perspective accuracy. Specifically, we parameterize a viewing ray γ passing through a pixel at (x, y) as the intersection of two perpendicular planes: the x-plane h x = (-1, 0, 0, x) ⊤ and the y-plane h y = (0, -1, 0, y) ⊤ . The ray is then transformed into the deformed Gaussian's local space by:
h u/v = (MHJ) -1 -⊤ h x/y = (MHJ) ⊤ h x/y ,(6)
where M is the transformation matrix from world to screen space. It should be noticed that as combining transformation MHJ, the viewpoint, 3D anisotropic structure of Gaussian and facial expressions, are encoded into perspective projection. After projection, we normalize the homogeneous coordinates to recover 3D positions of the viewing ray in the deformed tangent plane:
h ′ u = ( h 1 u h 4 u , h 2 u h 4 u , h 3 u h 4 u , 1) h ′ v = ( h 1 v h 4 v , h 2 v h 4 v , h 3 v h 4 v , 1)(7)
where h ′i u denotes for the i-th component. This step performs a perspective-divide by normalizing the homogeneous coordinates (dividing by h 4 ), which reconstructs the true 3D intersection point between the viewing ray and the deformed tangent plane of the Gaussian in perspective space. Unlike the affine approximation that assumes an orthogonal footprint, this perspective-divide "looks into" the Gaussian covariance. Finally, we evaluate the divergence of a camera ray from Gaussian in a straightforward way. Specifically, with
m = (h ′1 u , h ′2 u , h ′3 u ) × (h ′1 v , h ′2 v , h ′3 v ) l = (h ′1 u , h ′2 u , h ′3 u ) -(h ′1 v , h ′2 v , h ′3 v ),(8)
we define the divergence D derived from the perpendicular distance ϕ * between the Gaussian center and its closest point on the viewing ray in local tangent plane, ∥v∥ stand for magnitude of a vector v:
D(µ, Σ, γ) = e -1 2 (ϕ * ) 2 ϕ * = ∥m∥ ∥l∥ . (9
)
Volume rendering. Then, we obtain screen points by H ′ = MHJ, the center o of projected splat and the outermost bounds e is computed as:
o i = ⟨f , H ′ i • H ′ 4 ⟩ e i = o 2 i -⟨f , H ′ i • H ′ i ⟩ f = (1, 1, 1, -1) ⟨(1, 1, 1, -1), (H ′ 4 • H ′ 4 )⟩ (10
)
where H ′ i is the i-th row of H ′ , ⟨x, y⟩ stands for dot product. The volumetric alpha blending is conducted to compute the rendered color in a perspective-aware way:
C(p) = K k=1 c k α k D(µ k , Σ k , γ) k-1 j=1 (1 -α j D(µ j , Σ j , γ))).(11)
Optimization. We introduce a deformed normal regularization to maintain the learned structure of often-occluded regions such as teeth and eyeballs:
L nr = ∥n d w -n w ∥ 2(12)
where n w denotes the Gaussian normal in the w-th frame, computed as the inverse camera ray direction -γ in the deformed tangent plane and then transformed back to world space: n w = Rot w • (-γ), Rot w is the rotation component of matrix MHJ. The deformed normal is n d w = J -⊤ w n 0 , where J w is from Eq. 3 in the w-th frame and n 0 is the canonical normal.
Finally, we optimize our model with the following loss:
L = L c + λ d L d + λ n L n + λ s L scaling + λ p L position + λ nr L nr (13
)
where L c is a combination of photometric loss L rgb and a D-SSIM term following 3DGS [1]. To ensure accurate geometry reconstruction, we incorporate geometric loss terms from [8], including depth-distortion L d and normal consistency L n . Toward a better alignment between Gaussians and parent triangles, we use regularization terms L scaling and L position from [2]. We set the hyperparameters following these works and λ nr as 0.01.
this section cite: ['b60', 'b61', 'b0', 'b7', 'b1']

Section: Incremental BVH Tree Pivoting for Fast Avatar Reconstruction
To enable rapid and geometry-accurate extraction of avatar meshes from the opacity fields introduced in Sec. 3.2, we adopt an incremental triangulation strategy for dynamic points induced by facial expressions, referred to as 'hopping points'. The core of our approach lies in organizing Gaussians into a dynamic BVH tree, which accurately simulates vertex topological movements through Gaussian BVH pivoting, thereby facilitating efficient and adaptive mesh reconstruction.
this section cite: []

Section: Incremental Triangulation
For the initial frame, we employ 3D Delaunay triangulation to generate tetrahedral grids for each Gaussian and its bounding points. We then perform opacity evaluation, defined as the minimum opacity across all visible and relevant training views (depending on different areas in the FLAME model, e.g., the side view for hair) that observe the point. Finally, binary search is conducted over the opacity-SDF field of all tetrahedral grids to locate the zero level set.
For subsequent frames, we employ incremental triangulation guided by hopping points filtering via dynamic Gaussian BVH tree pivoting, as illustrated in Fig. 4, significantly reducing computation time compared to full re-triangulation. Since Gaussian movements fluctuate between frames, applying a fixed threshold to screen Gaussian centers and covariances can result in: 1) incomplete culling, causing a significant degradation in mesh extraction quality; or 2) excessive computation, which reduces the frame rate. Leveraging on our dynamic Gaussian BVH, which accurately simulates the topological movements of vertices, hopping points are swiftly filtered through branch rotation.
this section cite: []

Section: Hopping Gaussians Filtering via Dynamic BVH Pivoting
We begin by constructing a static binary radix BVH tree [63] from a given set of 3D Gaussians, where each leaf node represents the tight bounding box of a Gaussian cluster, and each internal node denotes the bounding box encompassing its two child nodes. To dynamically simulate the topological movements of Gaussians over time, we perform pivoting of the Gaussian BVH tree at each frame.
this section cite: ['b62']

Section: BVH Pivoting.
To better illustrate this topological simulation, we demonstrate our BVH structure shown in Fig. 4, where yellow "a", red "b", blue "c", and green "d" bounding boxes correspond to facial features, alongside its tree representation of the avatar. As facial expressions occur (e.g., eye opening), these bounding volumes are locally refitted according to the movement of enclosed Gaussians. As the extent of box "d" significantly expands to tightly enclose both "a" and "b", the BVH tree achieves branch rotation (swapping between "b" and "c") to reorganize the hierarchy and do topological simulation. It is important to emphasize that rather than balancing the tree, these rotations are employed to simulate Gaussian movement across volumes, minimizing the overall bounding extent cost-sometimes by introducing imbalance into the tree structure. Specifically, we implement these binary-tree rotations inspired by [64].
this section cite: ['b63']

Section: Branch Rotations.
On each frame, the BVH tree is updated by a post-order traversal, see Sec. B.1 for details. For rotations, see part (d) of Fig. 2, which illustrates candidate node swaps considered in pivoting. Lift Rebalancer (red rotation) occurs between a parent node's child and grandchild, adjusting subtree hierarchy levels to optimize local structure. Reorder Rebalancer (blue rotation) works among siblings to bypass local optima and achieve better global optimization. Each rotation operation is assigned a cost to filter hopping Gaussians. The green rotation produces mirrored-equivalent trees from Reorder Rotation and is omitted to avoid redundant computations.
this section cite: []

Section: Handling Hopping Points.
When the rotation cost of the middle level nodes exceeds a threshold, the Gaussians within exhibit strong mobility and are filtered as hopping points, and will be incrementally 3D triangulated.
this section cite: []

Section: Experiments

this section cite: []

Section: Experiments Settings
Implementation Details. To initialize TGA, we adopt VHAP [58] to preprocess the multi-view RGB video dataset for head tracking. We compare our approach against state-of-the-art avatar reconstruction methods via both qualitative and quantitative experiments. Specifically, we evaluate TGA on Chamfer distance, normal error, and recall [65] using the Multiface dataset [66], which provides 3D ground truth. We further evaluate the performance in terms of mesh extraction time, mesh rendering quality, and Gaussian-based novel view synthesis and self-reenactment quality. Thanks to our perspective-correct ray tracing for precise evaluation of Gaussian contributions, TGA converges within 300k iterations. All experiments are performed on NVIDIA RTX 4090 GPUs, using the same hyperparameters as GaussianAvatars [2].
GT Ours NPHM HRN SurFHead Topo4D Input Method L 1 -CD↓ MAE↓ Recall@2.5mm↑ HRN [67] 2.64 22.3 0.698 3DDFA [68] 4.35 22.9 0.649 NHA [42] 6.02 28.9 0.462 NPHM [24] 3.35 20.5 0.764 SF [4] 2.50 24.8 0.751 Topo4D [54] 2.33 19.3 0.772 Ours 2.16 17.7 0.802
Figure 6: Comparison of reconstructed meshes and normals on the Multiface [66]. Although frontalview differences appear minor, 3D error metrics show that TGA remains the closest to the GT.
this section cite: ['b57', 'b64', 'b65', 'b1', 'b67', 'b41', 'b23', 'b3', 'b53', 'b65']

Section: Memory Overheads.
The memory overheads of training and mesh extraction is no more than 24 gigabytes since we load images on-the-fly. When scaling to longer frame sequences and with stable number of Gaussians, the training memory usage remains mostly constant, and the storage for BVH tree pivoting is related to the number of Gaussians since it is conducted frame by frame.
Datasets. We evaluate our method on the NeRSemble [33], Multiface [66] and NHA Dataset [42].
The NeRSemble captures detailed facial dynamics, and the data is calibrated with sub-millimeter accurate camera poses and high-quality foreground segmentation. The Multiface dataset captures subjects covering dense multiview camera captures, rich facial expressions, and ground truth mesh to evaluate the 3D reconstruction efficiency. The NHA real dataset contains sequences that are suitable for the evaluation of full dynamic head approaches. We use it to evaluate the novel-view synthesis and self-reenactment rendering performance of our method.
this section cite: ['b32', 'b65', 'b41']

Section: Baselines
HRN [67] is a hierarchical representation network that achieves detailed face reconstruction. It generates a displacement map from each view and fuse them to obtain the final mesh. 3DDFA-V3 [68] uses geometric guidance for facial part segmentation for face reconstruction. We reconstruct a mesh for each view and then fuse them to obtain a multi-view-consistent final mesh. NHA [42] is learned from a monocular RGB portrait video that features a range of different expressions and views. NPHM [24] generates a signed distance field of a human head given an identity code and an expression code, and can then be translated into a mesh via marching cubes [69]. SurFhead [4] employs 2DGS [70] and GaussianAvatars [2] to reconstruct photorealistic avatars and high-fidelity surface normals and meshes from videos. Topo4D [54] introduces novel texture regularization.
this section cite: ['b66', 'b67', 'b41', 'b23', 'b68', 'b3', 'b69', 'b1', 'b53']

Section: Avatar Reconstruction Results

this section cite: []

Section: Mesh Geometry.
We compare TGA against our baselines on the NeRSemble dataset [33] by reconstructing each avatar from 16-view RGB sequences and present qualitative results in Fig. 5.
Comparative experimental results demonstrate that TGA can faithfully capture facial shape and expression details, greatly aiding avatar identity recognition and accurate emotion interpretation. Although HRN [67] reconstructs detailed wrinkle patterns, it still misses personalized eye and nose features. 3DDFA-V3 [68] exhibits jaw-shape inaccuracies. NHA [42] employs high-capacity neural networks for photorealistic rendering, but its geometry remains underconstrained. NPHM [24] excels at expression representation yet fails to preserve identity. Topo4D [54] and SurFHead [4] achieve strong identity reconstruction; however, Topo4D lacks fine facial detail, and SurFHead is constrained by depth maps rendered from 3DGS.
Additionally, we conduct a qualitative and quantitative evaluation of the reconstructed meshes on the Multiface Dataset [66] in Fig. 6. We focus on three metrics: L 1 -Chamfer distance, normal MAE (Mean Angular Error), and Recall [65] which measures the percentage of ground-truth points within a 2.5 mm threshold of any reconstructed point. TGA consistently produces meshes that closely match the ground truth from both frontal and side viewpoints.
Mesh Rendering. Moreover, we evaluate the mesh-based rendering results on the four subjects (shown in Fig. 5) from the frontal view in Tab. 1. Since TGA currently does not incorporate intrinsic decomposition or reflectance modeling, we employ flat shading based solely on vertex colors for rendering. We additionally evaluate the temporal quality and consistency of our results on temporal using VMAF [71], a metric designed to capture both perceptual quality and temporal coherence. The qualitative mesh rendering results are represented in the appendix C.2.
this section cite: ['b32', 'b66', 'b67', 'b41', 'b23', 'b53', 'b3', 'b65', 'b64', 'b70']

Section: SurFhead

this section cite: []

Section: GHA GT Ours
Figure 7: Self-reenactment on the NHA [42] and NeRSemble [33] dataset.
this section cite: ['b41', 'b32']

Section: Gaussians Rendering.
Although our method primarily targets geometry-accurate 3D reconstruction, TGA also exhibits strong performance in self-reenactment and novel-view synthesis. We present the comparison with SurFhead [4] and GHA [51] in Fig. 7 and Tab. 1, respectively. The self-reenactment results indicate that, by leveraging the Perspective-Aware Gaussian Transformation, the proposed method can achieve more faithful rendering of dynamic regions (eyes and mouth) with fewer iterations than other 3DGS-based avatar methods. Additionally, we present sequential novel view synthesis rendering frames and their corresponding PSNR curves. The Fig. 8 demonstrates the stability and temporal consistency of TGA.
Inference Time. Furthermore, we benchmark mesh-extraction times against baseline methods in Tab. 1. For Gaussian-based methods, meshes are extracted via Truncated Signed Distance Function fusion on 3DGS-rendered depth maps following 2DGS [70]. To ensure fairness, we selected sequences of 100 frames from the Multiface Dataset [66], using 16 key views for tracking and resizing images to the resolution employed by NeRSemble [33]. HRN [67], 3DDFA [68], and NPHM [24] takes approximately 2-5 minutes per frame to inference. Perspective-aware Gaussian Transformation (PGT). We demonstrate the geometric (CD, MAE and Recall for extracted meshes) and chromatic (PSNR, SSIM, LPIPS for Gaussian-based rendering) impact of the perspective-aware Gaussian transformation in Tab. 2 and Fig. 9. First, the homogeneous formulation yields a noticeable improvement in mesh quality. Second, the Jacobian gradient is essential for guiding deformation of canonical Gaussians in dynamic regions, such as the eyes (red box in Fig. 9). Without the transformed normal loss, Jacobian guided deformation fails to propagate effectively into the geometry, as evidenced by the results in Tab. 2.
Ours Full w/o Jacobian Input RGB Ours Full w/o Homogeneous w/o Jacobian w/o Homogeneous 1) frame n 2) frame n+1 a) Fixed Δ = 0.0039 b) Fixed Δ = 0.0034 c) Ours Hopping Figure 10: Effect of hopping Gaussians filtering.
Hopping Points Filtering. Furthermore, we evaluate hopping-point filtering via our BVH pivoting mechanism in Fig. 10, rather than relying on a fixed ∆ threshold on Gaussian centers across frames. When filtering with a fixed ∆ Gaussian center between frames, the blue points illustrate (a) incomplete culling and (b) excess screening. In contrast, our BVH pivoting accurately filters out the hopping Gaussians (c). It should be noted that when the avatar slightly opens the eyes or mouth, the forehead topology remains largely unchanged, obviating the need for re-triangulation in that region. Accelerated Mesh Extraction. We also evaluate mesh extraction speed across different training iterations by reporting the number of mesh vertices at key checkpoints and the time required for mesh extraction per frame. As training progresses, although the scene contains more points, the Gaussian BVH tree has learned a better topology of the avatar face. As shown in Tab. 4, while the total triangulation time (3rd column) increases with O(n log n) complexity, our method effectively reduces the computational burden (4th column).
this section cite: ['b3', 'b50', 'b69', 'b65', 'b32', 'b66', 'b67', 'b23']

Section: Discussion
Limitations. Our method struggles in regions like hair and eyes, where translucency, non-rigid motion, and strong specular reflections violate modeling assumptions, leading to opacity inconsistency and hollow-eye artifacts. It also fails under severe expression deformations, causing mesh tearing or topology distortion, since the BVH-hopping Gaussians assume smooth motion. Performing a fresh triangulation can effectively remove accumulated errors.
this section cite: []

Section: References
Ref_id:b0 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b1 Title: Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians Year: (2024)
Ref_id:b2 Title: Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians Year: (2024)
Ref_id:b3 Title: Surfhead: Affine rig blending for geometrically accurate 2d gaussian surfel head avatars Year: (2024)
Ref_id:b4 Title: Learning a model of facial shape and expression from 4d scans Year: (2017)
Ref_id:b5 Title: A morphable model for the synthesis of 3d faces Year: (2023)
Ref_id:b6 Title: Geometric adjustment of gaussian head avatar Year: (2024)
Ref_id:b7 Title: Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes Year: (2024)
Ref_id:b8 Title: A 3d face model for pose and illumination invariant face recognition Year: (2009)
Ref_id:b9 Title: Large scale 3d morphable models Year: (2018)
Ref_id:b10 Title: High-fidelity 3d digital human head creation from rgb-d selfies Year: (2021)
Ref_id:b11 Title: Hack: Learning a parametric head and neck model for high-fidelity animation Year: (2023)
Ref_id:b12 Title: Expressive body capture: 3d hands, face, and body from a single image Year: (2019)
Ref_id:b13 Title: Generating 3d faces using convolutional mesh autoencoders Year: (2018)
Ref_id:b14 Title: Facescape: a large-scale high quality 3d face dataset and detailed riggable 3d face prediction Year: (2020)
Ref_id:b15 Title: Spiralnet++: A fast and highly efficient mesh convolution operator Year: (2019)
Ref_id:b16 Title: Faceverse: a fine-grained and detail-controllable 3d face morphable model from a hybrid dataset Year: (2022)
Ref_id:b17 Title: Authentic volumetric avatars from a phone scan Year: (2022)
Ref_id:b18 Title: Fitme: Deep photorealistic 3d morphable model avatars Year: (2023)
Ref_id:b19 Title: Relightify: Relightable 3d faces from a single image via diffusion models Year: (2023)
Ref_id:b20 Title: Fitdiff: Robust monocular 3d facial shape and reflectance estimation using diffusion models Year: (2025)
Ref_id:b21 Title: Daniel Cremers, and Christian Theobalt. i3dmm: Deep implicit 3d morphable model of human heads Year: (2021)
Ref_id:b22 Title: Imface: A nonlinear 3d morphable face model with implicit neural representations Year: (2022)
Ref_id:b23 Title: Mononphm: Dynamic head reconstruction from monocular videos Year: (2024)
Ref_id:b24 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b25 Title: Instant neural graphics primitives with a multiresolution hash encoding Year: (2022)
Ref_id:b26 Title: Tensorf: Tensorial radiance fields Year: (2022)
Ref_id:b27 Title: Nerfies: Deformable neural radiance fields Year: (2021)
Ref_id:b28 Title: Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields Year: (2021)
Ref_id:b29 Title: Neural 3d video synthesis from multi-view video Year: (2022)
Ref_id:b30 Title: Humanrf: High-fidelity neural radiance fields for humans in motion Year: (2023)
Ref_id:b31 Title: Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields Year: (2023)
Ref_id:b32 Title: Nersemble: Multi-view radiance field reconstruction of human heads Year: (2023)
Ref_id:b33 Title: 4d gaussian splatting for real-time dynamic scene rendering Year: (2024)
Ref_id:b34 Title: Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction Year: (2024)
Ref_id:b35 Title: Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis Year: (2024)
Ref_id:b36 Title: Spacetime gaussian feature splatting for real-time dynamic view synthesis Year: (2024)
Ref_id:b37 Title: Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle Year: (2024)
Ref_id:b38 Title: From tokens to nodes: Semantic-guided motion control for dynamic 3d gaussian splatting Year: (2025)
Ref_id:b39 Title: Haif-gs: Hierarchical and induced flow-guided gaussian splatting for dynamic scene Year: (2025)
Ref_id:b40 Title: Mixture of volumetric primitives for efficient neural rendering Year: (2021)
Ref_id:b41 Title: Neural head avatars from monocular rgb videos Year: (2022)
Ref_id:b42 Title: Instant volumetric head avatars Year: (2023)
Ref_id:b43 Title: Havatar: Highfidelity head avatar via facial model conditioned neural radiance field Year: (2023)
Ref_id:b44 Title: Voltemorph: Real-time, controllable and generalizable animation of volumetric representations Year: (2024)
Ref_id:b45 Title: A point-based shape model for real-time head avatar animation with 3d gaussian splatting Year: (2024)
Ref_id:b46 Title: Headgas: Real-time animatable head avatars via 3d gaussian splatting Year: (2024)
Ref_id:b47 Title: Gaussianheads: End-to-end learning of drivable gaussian head avatars from coarse-to-fine representations Year: (2024)
Ref_id:b48 Title: Flashavatar: High-fidelity head avatar with efficient gaussian embedding Year: (2024)
Ref_id:b49 Title: Hravatar: High-quality and relightable gaussian head avatar Year: (2025)
Ref_id:b50 Title: Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians Year: ()
Ref_id:b51 Title: Splattingavatar: Realistic real-time human avatars with mesh-embedded gaussian splatting Year: (2024)
Ref_id:b52 Title: 3d gaussian blendshapes for head avatar animation Year: (2024)
Ref_id:b53 Title: Topo4d: Topology-preserving gaussian splatting for high-fidelity 4d head capture Year: (2024)
Ref_id:b54 Title: Npga: Neural parametric gaussian avatars Year: (2024-06)
Ref_id:b55 Title: Learning neural parametric head models Year: ()
Ref_id:b56 Title: Scaffoldavatar: High-fidelity gaussian avatars with patch expressions Year: (2025)
Ref_id:b57 Title: Vhap: Versatile head alignment with adaptive appearance priors Year: (2024-09)
Ref_id:b58 Title: Ewa volume splatting Year: (2001)
Ref_id:b59 Title: Drivable 3d gaussian avatars Year: (2023)
Ref_id:b60 Title: A hardware architecture for surface splatting Year: (2007)
Ref_id:b61 Title: Perspective accurate splatting Year: (2004)
Ref_id:b62 Title: Maximizing parallelism in the construction of bvhs, octrees, and k-d trees Year: (2012)
Ref_id:b63 Title: Fast, effective bvh updates for animated scenes Year: (2012)
Ref_id:b64 Title: What do single-view 3d reconstruction networks learn? Year: (2019)
Ref_id:b65 Title: A dataset for neural face rendering. in arxiv Year: (2022)
Ref_id:b66 Title: A hierarchical representation network for accurate and detailed face reconstruction from in-the-wild images Year: (2023)
Ref_id:b67 Title: 3d face reconstruction with the geometric guidance of facial part segmentation Year: (2024)
Ref_id:b68 Title: Marching cubes: A high resolution 3d surface construction algorithm Year: (1998)
Ref_id:b69 Title: 2d gaussian splatting for geometrically accurate radiance fields Year: (2024)
Ref_id:b70 Title: Toward a practical perceptual video quality metric Year: (2016-06)
Ref_id:b71 Title: 3d gaussian ray tracing: Fast tracing of particle scenes Year: (2024)
Ref_id:b72 Title: 3dgut: Enabling distorted cameras and secondary rays in gaussian splatting Year: ()
Ref_id:b73 Title: Does 3d gaussian splatting need accurate volumetric rendering? Year: (2025)
