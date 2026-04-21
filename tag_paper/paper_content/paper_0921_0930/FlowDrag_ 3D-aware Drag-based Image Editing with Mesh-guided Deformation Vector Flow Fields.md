Title: FlowDrag: 3D-aware Drag-based Image Editing with Mesh-guided Deformation Vector Flow Fields
Abstract: Drag-based editing allows precise object manipulation through point-based control, offering user convenience. However, current methods often suffer from a geometric inconsistency problem by focusing exclusively on matching user-defined points, neglecting the broader geometry and leading to artifacts or unstable edits. We propose FlowDrag, which leverages geometric information for more accurate and coherent transformations. Our approach constructs a 3D mesh from the image, using an energy function to guide mesh deformation based on user-defined drag points. The resulting mesh displacements are projected into 2D and incorporated into a UNet denoising process, enabling precise handle-to-target point alignment while preserving structural integrity. Additionally, existing drag-editing benchmarks provide no ground truth, making it difficult to assess how accurately the edits match the intended transformations. To address this, we present VFD (VidFrameDrag) benchmark dataset, which provides ground-truth frames using consecutive shots in a video dataset. FlowDrag outperforms existing drag-based editing methods on both VFD Bench and DragBench.

Section: Introduction
The advancements in text-to-image (T2I) generation diffusion models (Ramesh et al., 2022;Saharia et al., 2022;Nichol et al., 2021;Li et al., 2023) have significantly enhanced image generation capabilities. Leveraging pretrained T2I diffusion models (Rombach et al., 2022) trained on large scale dataset, the image editing field has also seen substantial progress. Text-based image editing (Hertz et al., 2022; Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). While other methods only optimize around user-specified points, failing to preserve the Statue of Liberty's structure (first row), FlowDrag maintains overall integrity. In the second row, FlowDrag stably rotates the woman's face from her nose without distorting her hat or hand, whereas other methods fail to maintain geometric consistency. Tumanyan et al., 2023;Cao et al., 2023;Koo et al., 2024a), which uses user-provided prompts, has shown promising results but often struggles with precise and fine-grained edits. Even slight variations in text can lead to vastly different results, making it difficult for users to achieve consistent and detailed modifications, thereby highlighting the challenges of achieving precision in text-to-image editing. To address this, DragGAN (Pan et al., 2023) introduced a pointdragging method, enhancing user convenience for more precise editing. However, due to the constraints of GANs, struggled with general image performance. This limitation led to interest in exploring drag-based image editing using diffusion models, which can be categorized into two main approaches: motion-based (Shi et al., 2024;Ling et al., 2023;Liu et al., 2024;Zhang et al., 2024) and gradientguidance-based (Mou et al., 2023;2024) methods. The motion-based approach in drag-based image editing consists of two processes: motion supervision and point tracking.
Motion supervision measures the difference between the handle and target points in the UNet decoder's feature map and applies it to optimize the latent, gradually shifting the handle point toward the target. Subsequently, point tracking updates the handle point's position in the feature map, ensuring alignment with the progressively edited result. In contrast, gradient-based methods derive inspiration from score-based (Song et al., 2020b;Dhariwal & Nichol, 2021) diffusion model, utilizing gradient guidance governed by an energy function to perform the drag-based editing.
However, existing drag editing methods often suffer from a geometric inconsistency problem, leading to artifacts or significant deviations from the original object's structure. These issues become evident in the first row of Fig. 1(b)-(c), where the existing model fails to preserve the statue's integrity, causing the arm and torch to become noticeably altered and leading to visible artifacts. Similarly, in the second row of Fig. 1, we attempt to rotate a woman's face by placing a drag arrow on her nose. While her hat and the hand resting on her face should naturally follow this rotation, existing methods shift the nose but fail to preserve the hat and hand, resulting in unnatural deformation. We identified that these problems primarily stem from current methods' exclusive focus on matching feature correspondence between user-defined handle and target points, neglecting the broader geometric context of the image, as shown in Fig. 2(b). This issue is particularly prominent in edits that require preserving rigid parts of the object. Therefore, we define any transformation that must maintain rigidity (such as rotation, relocation, or pose changes) as a "Rigid Edit." In contrast, transformations that do not require strict rigidity, such as rescaling (which alters proportions), are referred to as "Non-rigid Edits." In this paper, we focus on "Rigid Edits" to ensure structural integrity during drag editing.
To overcome these limitations, we propose FlowDrag, a novel method designed to ensure stable and accurate image edits by preserving geometric information. Our approach proceeds through several stages. First, we construct a 3D mesh from the original image to represent the object's geometry. Second, we employ a progressive SR-ARAP (Levi & Gotsman, 2014) approach to deform the mesh from userdefined handle points to target points, preserving the object's geometric integrity throughout the transformation, as illustrated in Fig. 2(c). Third, after the mesh deformation, we compute the differential coordinates between the original and modified mesh and project them onto a 2D vector field.
Finally, this vector field is integrated into the motion supervision phase of the UNet's denoising process, which enhances both spatial accuracy and edit stability. Unlike existing methods restricted to user-defined drag editing points, Flow-Drag leverages a continuous displacement field derived from mesh deformations, preserving both spatial and geometric coherence for more reliable results.
We also propose a new drag-editing benchmark called VFD-Bench. Existing benchmarks, such as DragBench, do not provide ground-truth edited images for each input, making it difficult to accurately assess editing quality. The Image Fidelity metric (1-LPIPS) measures similarity between the original input and the edited result, often assigning lower scores to successful geometry-preserving edits. For instance,
this section cite: ['b25', 'b28', 'b21', 'b16', 'b27', 'b6', 'b34', 'b1', 'b22', 'b29', 'b17', 'b18', 'b42', 'b19', 'b15']

Section: Related Work

this section cite: []

Section: Text-based Image Editing
Early GAN-based methods (Patashnik et al., 2021;Xia et al., 2021) edited images by inverting them into StyleGAN latent spaces conditioned on textual descriptions. However, these methods had limited flexibility due to inherent trade-offs between generalized editing capability and reconstruction quality. Recently, diffusion-based methods (Kawar et al., 2023;Couairon et al., 2022;Brooks et al., 2023) enabled more precise and diverse edits by directly guiding the diffusion process with text prompts. Prompt-to-Prompt (Hertz et al., 2022) and Plug-and-Play (Tumanyan et al., 2023) further improved editing precision through attention feature injection mechanisms. More recent work also explored adjusting object poses and perspectives (Cao et al., 2023;Koo et al., 2024a;Yoon et al., 2024a). With advances in text-based editing techniques, various studies utilized these methods for tasks such as improving editing efficiency (Koo et al., 2024b;Deutch et al., 2024), human image animation (Yoon et al., 2024c), and virtual try-on (Hong et al., 2025).
However, text-based methods often lack the precision required for fine-grained control, as minor textual variations can result in unintended edits. This limitation has motivated the emergence of drag-based editing methods, which offer more direct and intuitive control.
this section cite: ['b23', 'b37', 'b10', 'b3', 'b0', 'b6', 'b34', 'b1', 'b4', 'b8']

Section: Drag-based Image Editing
Drag-based image editing modifies images using user-drawn drags, offering precise and interactive control for tasks such as rotation, relocation, and rescaling. While early GANbased methods (Pan et al., 2023) suffered from limited edit fidelity, diffusion models significantly improved both editing quality and diversity. Diffusion-based drag editing is divided into two categories: motion-based and gradientguidance-based. Motion-based methods (Shi et al., 2024;Ling et al., 2023;Liu et al., 2024;Zhang et al., 2024) rely on motion supervision and point tracking to iteratively shift handle points toward targets, preserving the original structure. Specifically, DragDiffusion optimizes the DDIM latent at a specific timestep (t=35), while Drag Your Noise targets bottleneck features in the U-Net across all timesteps. Good-Drag further introduces the AlDD framework, alternating drag operations and denoising across multiple timesteps to reduce cumulative changes and enhance fidelity. In contrast, gradient-guidance-based methods (e.g., DragonDiffusion (Mou et al., 2023), DiffEditor (Mou et al., 2024)) use gradient updates driven by an energy function derived from score-based diffusion models (Song et al., 2020b;Dhari-wal & Nichol, 2021), enabling more creative edits but often causing artifacts or reduced fidelity. Although motion-based approaches better preserve visual fidelity, they still struggle with geometric integrity due to limited structural understanding. To address this, we incorporate 3D mesh deformation to add explicit geometric information in 2D.
this section cite: ['b22', 'b29', 'b17', 'b18', 'b42', 'b19', 'b20']

Section: Preliminary

this section cite: []

Section: DDIM Inversion
DDIM (Song et al., 2020a) eliminates the stochastic elements of DDPM (Ho et al., 2020), producing a deterministic, non-Markovian process for precise control over diffusion steps. A U-Net denoiser network, ϵ θ , enables both sampling (Eq. ( 1)) from noise to image and inversion (Eq. ( 2)) from image to noise. Here, α t denotes the noise schedule at step t, and z t is the latent representation at that step.
z t+1 = α t+1 α t z t + 1 -αt+1 αt -1 ϵ θ (z t , t), (1
)
z * t = α t α t-1 z * t-1 + 1-αt αt-1 -1 ϵ θ (z * t-1 , t -1). (2)
In drag editing, DDIM Inversion is applied to obtain the final DDIM latent z t by progressively adding noise from z 0 to z t . As detailed in Section 3.2, this latent is optimized via motion supervision to refine editing results.
this section cite: ['b7']

Section: Diffusion Latent Optimization in Drag Editing
Motion Supervision Motion-based drag editing typically applies a motion supervision loss, L ms , to iteratively shift n handle points {h k i } n i=1 toward target points {t i } n i=1 . Formally, the motion supervision loss is defined as:
L ms z k t = n i=1 q∈P(h k i ,R) F q+δi z k t -sg F q z k t 1 + λ z k t-1 -sg(z 0 t-1 ) ⊙ 1 -M 1 ,(3)
where
δ i = ti-h k i ∥ ti-h k i ∥2 is the normalized direction from h k i to t i , P(h k i , R
) is a patch of radius R around h k i , F denotes the U-Net feature map, k is denoising timestep and sg(•) is the stop-gradient operator. The term λ weights the regularization that keeps {z k t-1 } close to the reference {z 0 t-1 } outside the masked region M . At each iteration, we compute ∂L ms /∂z k t and update z k t via gradient descent:
z k+1 t = z k t -η ∂ L ms z k t ∂ z k t ,(4)
where η is the learning rate. This process gradually moves each handle point closer to its target in the latent space.
this section cite: []

Section: Step 1. 3D Mesh Generation

this section cite: []

Section: 2D Vector Flow Field

this section cite: []

Section: Image-to-3D Diffusion Model

this section cite: []

Section: Depth-based Approach Diffusion-based Approach

this section cite: []

Section: Depth Map Input Image Input Image

this section cite: []

Section: Step 2. Progressive Mesh Deformation and 2D Vector Flow Generation

this section cite: []

Section: Input Image
Movable Vertices Constraint Vertices Handle Point Target Point Handle Vertex 𝒗 𝒉 Target Vertex 𝒗 𝒕 Intermediate Vertex 𝒗 𝒉 𝒌 Edit Mask 𝒗 𝒉 𝒗 𝒕 𝒗 𝒉 𝒌 3D Mapping Progressive SR-ARAP ••• Top-view Final Deformed Mesh Original Mesh Difference 2D Projection 𝒌-th iteration DepthMesh DiffMesh Point Tracking Following motion supervision, point tracking updates the handle points {h k i } by searching for the best matching features within a local patch. Specifically,
h k+1 i = arg min q∈P(h k i ,R2) F q z k+1 t -F h 0 i z t 1 , (5
)
where
P(h k i , R 2 ) is a square patch of radius R 2 around h k i .
This step aligns each handle point with the corresponding features in the updated latent. Alternating motion supervision and point tracking guides handle points progressively closer to their targets.
this section cite: []

Section: Geometric Mesh Deformation
In this section, we introduce widely utilized geometric mesh deformation methods, the As-Rigid-As-Possible (ARAP) (Sorkine & Alexa, 2007) approach and its enhanced variant, Smoothed Rotation As-Rigid-As-Possible (SR-ARAP) (Levi & Gotsman, 2014). Let M = (V, F ) represent the source mesh, where V consists of vertices v i = (x i , y i , z i ) in R 3×V , and F forms faces that are triangles formed by these vertices. The deformed mesh is denoted as M = ( V , F ), where V consists of the deformed vertex positions vi = (x i , ŷi , ẑi ), maintaining the same connectivity.
ARAP. ARAP aims to preserve local rigidity while allowing controlled deformations. Initially, users designate certain vertices as constraints, including vertices explicitly moved to desired positions (e.g., handle points moved to target points in drag editing) and vertices outside the editable region that remain unchanged. The remaining vertices are considered movable. Setting these constraint vertices transforms the original mesh V to the new constraint positions V , from which optimization begins. The ARAP method adjusts movable vertex positions by minimizing an energy function, maintaining local rigidity under fixed constraints. The energy function is defined as:
EARAP(M ) = i∈V j∈N (i) wij si Ri vi -vj -vi -vj 2 ,(6)
where R i and s i are internally optimized rotation matrices and local scale factors for each vertex i (with s i set to 1, preventing scale changes). The terms w ij represent cotangent weights, reflecting the stiffness or rigidity between vertices. The notation N (i) denotes the set of neighbors for vertex i, specifically those vertices directly connected by an edge, and j denotes each neighboring vertex within this set. Here, vi ,v j represent vertex positions optimized during deformation, whereas v i , v j refer to positions from the original undeformed mesh. During the optimization process, these positions vi ,v j are iteratively updated to minimize the energy function, with only movable vertices adjusted, while constraints remain fixed as boundary conditions.
this section cite: ['b32', 'b15']

Section: SR-ARAP.
Smoothed Rotation ARAP extends the ARAP formulation by adding a rotation-consistency term:
ESR-ARAP(M ) = EARAP(M ) + α i∈V j∈N (i) ∥Ri -Rj∥ 2 , (7
)
where R i and R j are the per-vertex rotation matrices, α is a regularization parameter. This extra term penalizes significant rotational discrepancies between adjacent vertices, resulting in smoother deformations. Specifically, a larger rotation difference ∥R i -R j ∥ 2 increases the energy, prompting the optimization process to minimize these differences.
As a result, rotations among adjacent vertices remain as consistent as possible, ensuring smoother and more natural deformations. These formulations facilitate controlled mesh deformation, beneficial for achieving smoother and locally rigid structural modifications.
this section cite: []

Section: Method
FlowDrag addresses the geometric inconsistency problem in drag-based editing by incorporating 3D mesh deformation. First, we generate a 3D mesh from the input image using a depth-based or diffusion-based approach (Fig. 3-Step 1).
Next, we deform the mesh using progressive SR-ARAP and derive a 2D vector flow from the deformed mesh (Fig.
this section cite: []

Section: 3-Step 2).
Finally, we integrate the 2D vector flow into the motion-based drag-edit pipeline to achieve geometry-aware movements, and utilize the 2D projection of the deformed mesh as layout features to preserve structural consistency and ensure accurate shape preservation (Fig. 4-Step 3). The following sections describe each of these steps in detail.
this section cite: []

Section: 3D Mesh Generation: Depth-based and Diffusion-based Approaches
We first generate a 3D mesh from the input image by extracting a depth map and using it as a foundation. For depth extraction, we adopt the Marigold (Ke et al., 2024) model, which reliably provides depth information from a single image within a few seconds. The depth-based mesh construction then proceeds in three stages. First, in the vertex mapping step, each pixel's depth value is converted into a corresponding vertex coordinate in 3D space. Next, the facet formation step connects adjacent vertices whose depth values are sufficiently similar, forming smooth surfaces.
Finally, the artifact reduction step excludes connections between vertices with large depth differences, minimizing discontinuities and effectively separating the foreground object from its background. Algorithm 1 in the appendix summarizes this procedure in detail.
Although this depth-based approach is simple and fast, it cannot account for unseen regions from a single viewpoint. Therefore, we additionally employ image-to-3D diffusion models (Xiang et al., 2024;Zhao et al., 2025), which infer hidden structures to produce more complete meshes. We refer to the mesh generated by our depth-based approach as DepthMesh, and the one produced by the diffusion-based approach as DiffMesh, as illustrated in Fig. 3-Step 1. We employ both meshes in our experiment to explore different levels of geometry detail and completeness.
this section cite: ['b11', 'b38', 'b43']

Section: Progressive Mesh Deformation and 2D Vector Flow Generation
We begin by mapping the user-defined handle point, target point, and mask from the 2D drag-editing setting onto our 3D mesh M = (V, F ). In this mapping, a handle vertex v h ∈ V corresponds to the handle point, and its target vertex v t ∈ V corresponds to the target point. The masked region indicates which vertices are movable, while all other vertices become constraints (see Fig. 3-Step2). Following the ARAP principle, each movable vertex adjusts its position by minimizing the ARAP energy, preserving local rigidity. In contrast, the constrained vertices remain fixed at their designated positions.
Progressive Deformation with SR-ARAP. In ARAPbased methods, the handle vertex v h is typically moved directly to its target position v t , followed by mesh deformation through ARAP energy minimization. However, when the distance between v h and v t is large, directly moving v h can cause abrupt local distortions and unnatural deformation (Sorkine & Alexa, 2007;Chen et al., 2017). Such distortions arise because large vertex displacements significantly stretch local edges, increasing ARAP energy and potentially causing convergence to suboptimal local minima.
To mitigate these issues, we propose a progressive deformation strategy with two components. First, we incrementally move the handle vertex toward its target over K iterations:
v (k+1) h = v (k) h + λ v t -v (k) h , 0 < λ ≤ 1, (8
)
where v
E SR-ARAP+InterStep M (k+1) = E SR-ARAP M (k+1) + β i∈V v(k+1) i - v(k) i 2 , (9
)
where v(k) i and v(k+1) i are the positions of the movable vertices at iterations k and k+1. The parameter β adjusts the strength of this regularization, with higher values promoting smoother transitions and smaller vertex movements, while lower values allow larger displacements.
2D Vector Flow Generation. After K iterations, the original mesh M converges to a final deformed mesh M . We then project both meshes onto the 2D image plane, denoted by π(M ) and π( M ), respectively. The 2D vector flow Φ is defined based per-vertex displacements:
Φ = ∆x i , ∆y i | ∆x i = x ′ i -x i , ∆y i = y ′ i -y i ,(10)
where (x i , y i ) and (x ′ i , y ′ i ) are the 2D coordinates of vertex i in π(M ) and π( M ). Thus, by capturing how each vertex moves from M to M , the vector flow Φ encodes the geometric changes induced by our progressive SR-ARAP approach. In the next section, we show how both Φ and π( M ) integrate into the motion-based drag-edit pipeline, providing geometry-aware guidance for image editing.
this section cite: ['b32', 'b2']

Section: Vector Flow-based Drag Editing
Vector Flow Sampling We focus on selecting an optimal set of vectors from the 2D flow field Φ for motion supervision (Eq.3) and point tracking (Eq. 5). First, we uniformly sample an N × N grid of candidate vectors within the edit mask, taking N positions along each axis. To refine this set, we explore two approaches: (1) Magnitude-based sampling, which sorts all candidates by displacement magnitude
this section cite: []

Section: Input Image

this section cite: []

Section: GT Image Input Image GT Image Input Image GT Image
Figure 5. Example images from VFD-Bench. We constructed this drag-based image editing dataset by selecting closely spaced frames from DAVIS, LOVEU-TGVE, and copyright-free Pexels videos, focusing on noticeable changes in pose or structure.
and keeps only the top few, and (2) Uniform sub-sampling, which retains vectors at regular intervals to ensure broad coverage. Either method yields a final subset Φ, typically containing 5-30 vectors that capture the most significant or representative displacements. We then restrict the summation in Eq.3 to q ∈ Φ, focusing the motion supervision on these carefully selected flow vectors.
this section cite: []

Section: Layout Feature Injection
In addition to leveraging our vector flow for latent optimization, we introduce a guidance branch utilizing the 2D projection of the deformed mesh, π( M ), to provide complementary geometric context. Specifically, as illustrated in Fig. 4, we first invert π( M ) via DDIM Inversion to obtain a latent noise representation. During the denoising process, we inject selected attention features from this representation into the primary drag-edit branch. Previous diffusion studies (Wu et al., 2023b;Yoon et al., 2024b) have shown that earlier timesteps establish broad structural outlines, while later timesteps refine finer details. Following this insight, we inject attention features only from earlier or intermediate timesteps, embedding approximate layout information derived from the deformed mesh. This ensures broader geometric context is transferred without imposing overly specific or potentially conflicting details, as the deformed mesh may not perfectly align with the user's final desired details. Thus, this layout feature injection complements vector-flow-based optimization by explicitly providing structural context, guiding the main edit branch toward a more geometrically consistent edited result.
this section cite: []

Section: VFD-Bench Dataset
Existing drag-based editing benchmarks, such as Drag-Bench, provide input images along with user-defined handle/target points and masks, but do not include ground-truth (GT) edited images. As a result, commonly used metrics like Image Fidelity (IF) and Mean Distance (MD) are computed between the input and the edited output. Specifically, IF (measured as 1-LPIPS) assesses how closely the edited image resembles the original, while MD uses DIFT features to measure how effectively the handle points have moved to their targets. However, in cases where rotation or pose changes are successfully introduced, IF often yields low scores simply because the edited object differs from the original layout, even though the edit is successful.
To address these limitations, we propose VFD-Bench, a new dataset that provides an explicit GT image for each input. As illustrated in Fig. 5, VFD-Bench is constructed by selecting pairs of video frames (from sources such as DAVIS (Pont-Tuset et al., 2017), LOVEU-TGVE (Wu et al., 2023a), TVR (Lei et al., 2020) and copyright-free clips on Pexelsfoot_0 ) where an object undergoes a clear change in pose or structure. In total, we construct 250 input-GT pairs suitable for drag-based editing. For each pair, the handle points and target points are defined based on the differences between the input and ground-truth (GT) images, with 1-5 drags assigned per sample. Further details regarding the dataset annotation process and the number of images per category are provided in Appendix C.
this section cite: ['b24', 'b14']

Section: Evaluation Metrics
Unlike previous benchmarks, VFD-Bench enables direct comparison of edited images to actual GT results. We measure Image Fidelity using both PSNR (at the RGB level) and LPIPS (at the feature level). For assessing how well handle points align with target points, we retain the Mean Distance (MD) metric. Since video frames can contain changing backgrounds unrelated to the target object, we compute all metrics within the user mask to focus on the edited region. This approach allows for a more reliable and detailed evaluation of drag-based editing methods.
this section cite: []

Section: Experiments

this section cite: []

Section: Implementation Details
We validate FlowDrag using the pre-trained Stable Diffusion 1.5 model (Rombach et al., 2022), processing images at a resolution of 512 × 512. For image encoding and decoding, we use VQ-VAE (Razavi et al., 2019). Following prior motion-based approaches (Ling et al., 2023;Zhang et al., 2024;Liu et al., 2024), we fine-tune the input image with LoRA (Hu et al., 2021) (rank = 16) for 200 steps. DDIM Inversion is applied up to step 38 (75% of the total 50 denoising steps), as in (Zhang et al., 2024), with layout feature injection at timestep t ′ = 30. For 3D mesh generation, we primarily utilize DiffMesh. However, when DiffMesh exhibits significant artifacts or deviates substantially from the original image, we employ DepthMesh instead. In Progressive SR-ARAP mesh deformation, we compute w ij in  (Shi et al., 2024) 0.89 33.70 DragNoise (Liu et al., 2024) 0.63 33.41 FreeDrag (Ling et al., 2023) 0.70 35.00 GoodDrag (Zhang et al., 2024) 0.86 22.96 FlowDrag (ours) 0.82 22.88  (Shi et al., 2024) 24.5 0.78 36.52 DragNoise (Liu et al., 2024) 22.3 0.68 36.21 FreeDrag (Ling et al., 2023) 22.0 0.74 38.32 GoodDrag (Zhang et al., 2024) 25.2 0.82 25.65 FlowDrag (Ours) 26.3 0.85 24.51
Eq. 6 using Open3D library, and set α between 0.2 and 0.4.
this section cite: ['b27', 'b26', 'b17', 'b42', 'b18', 'b9', 'b42', 'b29', 'b18', 'b17', 'b42', 'b29', 'b18', 'b17', 'b42']

Section: Additionally, we conduct an ablation study on the Inter-Step
Smoothness term (Section 6.4), exploring β ∈ [0, 1] in Eq. 9.
To integrate vector flow into drag editing, we uniformly sample points from a 20 × 20 grid (N = 20) within the 2D flow field over the edit mask, selecting between 5 and 30 points for optimization. This setup balances computational efficiency with adequate coverage of the flow field.
this section cite: []

Section: Datasets and Evaluation Metrics
Datasets We evaluate FlowDrag on the DragBench (Shi et al., 2024) and our proposed VFD-Bench. DragBench contains 205 images of diverse content, along with 349 pairs of handle and target points. Each image has one or more dragging instructions (i.e., handle-target point pairs) and a mask that specifies the editable region. VFD-Bench, introduced in Section 5, includes 250 images selected from video sources (DAVIS, TGVE, TVR, and Pexels), each paired with a ground-truth (GT) edit.
Metrics We evaluate both fidelity and precision for each edited image. On DragBench, where no ground truth (GT) is provided, we measure Image Fidelity (IF) as 1-LPIPS and compute the Mean Distance (MD) using DIFT (Tang et al., 2023) to track how well handle points move to their targets. In VFD-Bench, which provides GT images, we quantify fidelity with both PSNR (RGB-level) and LPIPS (feature-level) while retaining MD for point alignment, as described in Section 5. Since VFD-Bench frames can involve irrelevant background changes, we compute metrics within the user-defined mask to focus on the edited region.
this section cite: ['b29', 'b33']

Section: GoodDrag FlowDrag (ours) DiffEditor User Edit FreeDrag
Figure 6. Qualitative results with recent drag-based image editing systems. Our FlowDrag produces outputs that maintain geometric consistency more effectively than other methods.
this section cite: []

Section: Experimental Results
Qualitative Comparisons. We compare FlowDrag with current drag-based editing methods in Fig. 6. Although motion-based approaches generally outperform the gradientguided DiffEditor, FreeDrag and GoodDrag often neglect the object's broader spatial structure by relying solely on user drag points. In contrast, FlowDrag leverages a vector flow field from mesh deformation, enabling more cohesive transformations. For example, in the first row of Fig. 6, only one drag point is provided. DiffEditor fails to move the face, FreeDrag relocates the face but leaves the hat unchanged, and GoodDrag removes the brim. However, FlowDrag adjusts both the face and the hat, illustrating how spatially informed vectors yield more natural edits with fewer artifacts. Additional qualitative comparisons on DragBench and VFD-Bench are provided in Appendix F.
Quantitative Results. On DragBench, FlowDrag achieves the best Mean Distance (MD), indicating effective dragging, as shown in Table 1. However, DiffEditor scores highest on the 1-LPIPS fidelity metric because it induces minimal edits. In VFD-Bench, which includes actual ground-truth images from real video frames, FlowDrag outperforms all methods in PSNR, 1-LPIPS, and MD, demonstrating its effectiveness in preserving geometric consistency while delivering accurate edits.  When 10 vectors are sampled, both metrics achieve their highest values, indicating the importance of optimal vector selection.
this section cite: []

Section: User Study.
We conducted a user study on 50 images from DragBench and VFD-Bench, comparing FlowDrag with DiffEditor, FreeDrag, and GoodDrag. We recruited 25 volunteers to rank each method's edited results (4 = best, 1 = worst) based on drag accuracy and image quality. As shown in Fig. 7, FlowDrag consistently received higher scores than the other methods in both aspects.
this section cite: []

Section: Ablation Study
We conducted an ablation study in FlowDrag on VFD-Bench by evaluating the following three aspects: (1) influence of the regularization parameter β in progressive SR-ARAP deformation, (2) effect of selected vector count on performance, (3) magnitude-based vs. uniform sub-sampling in 2D vector flow field sampling.
this section cite: []

Section: Influence of β in Progressive SR-ARAP We investigate the effect of varying the parameter β of the Inter-Step Smoothness term in Eq.9 on local rigidity preservation during mesh deformation.
To quantify this, we introduce a rigidity measure (see Appendix A.1) based on mean edge length ratios (MELR) and the Mean ARAP Error (mARAP Error ) between the original mesh M and its deformed counterpart M . The results for β ∈ {0.2, 0.4, 0.6, 0.8, 1.0} are presented in Table 4. Higher β values penalize large vertex displacements between iterations, resulting in smoother deformations but potentially restricting flexibility. Notably, β = 0.8 yields the best results, achieving maximal mean edge length ratio and mean ARAP error, indicating effective shape preservation with stable and smooth vertex transitions.
Mesh Deformation Results DiffMesh User Edit Editing Constraints Handle Vertex Target Vertex Handle Points Target Points Movable Vertices Constraint Vertices Deformation Vector
this section cite: []

Section: 2D vector flow
Figure 9. Visualization of the mesh-guided editing process using DiffMesh. Our FlowDrag process begins with user-defined edits specifying handle and target points. DiffMesh reconstructs a structured 3D mesh, from which editing constraints (movable and constraint vertices) are defined. Guided by these constraints, the mesh undergoes deformation, yielding clear deformation vectors. These deformation vectors are projected onto a 2D vector flow field, where representative vectors are sampled. These sampled vectors provide accurate geometric guidance, resulting in stable and precise image editing outcomes.
Table 3. Comparison of sampling strategies using MD and 1-LPIPS metrics on VFD-Bench.
this section cite: []

Section: Sampling Strategy MD ↓ 1-LPIPS ↑
Uniform Sub-sampling 24.75 0.83 Magnitude-based Sampling 24.51 0.85
Effect of Selected Vector Count on Performance We analyzed the impact of the number of selected vectors in Φ on the PSNR metric. Fig. 8 shows the results, indicating the best performance at 10 vectors. These results emphasize the importance of selecting an optimal vector count for effective editing performance.
Magnitude-based vs. Uniform Sub-sampling in 2D Vector Flow Field Sampling We compare magnitude-based sampling and uniform sub-sampling using Mean Distance (MD) and Image Fidelity (1-LPIPS). As shown in Table 3, magnitude-based sampling consistently achieves lower MD and higher 1-LPIPS scores, indicating more effective vector selection. This highlights the benefit of prioritizing vectors with larger magnitudes in the flow field.
this section cite: []

Section: More Visualization Results
Detailed visualizations of FlowDrag's mesh-guided editing process are shown in Fig. 9 and Fig. 16. Additional sensitivity analysis on how variations in mesh deformation influence editing outcomes can be found in Appendix D, and further analysis of mesh deformation efficiency is provided in Appendix E.
this section cite: []

Section: Limitation and Future Work
FlowDrag effectively addresses geometric inconsistencies prevalent in current drag-based editing methods, but several limitations remain. First, our method relies on stable mesh deformation, inherently limiting feasible dragging distances. Thus, our method is optimal for moderate dragging operations that preserve structural coherence. Additionally, FlowDrag primarily supports rigid edits, but struggles with significant content creation or removal tasks requiring major structural changes. Lastly, FlowDrag projects 3D mesh deformation onto a 2D plane, inherently losing detailed 3D structural information. This issue is compounded by FlowDrag's reliance on Stable Diffusion, which lacks 3D understanding. While our approach enhances geometric coherence, it cannot fully preserve the original 3D geometry. We believe future work could benefit from exploring inherently 3D-aware or motion-aware diffusion models, such as video diffusion, to better capture object dynamics and 3D structures, enhancing 3D understanding in image editing.
this section cite: []

Section: Conclusion
In this paper, we propose FlowDrag, a framework designed to address geometric inconsistencies in drag-based image editing via 3D mesh deformation. Additionally, we introduce VFD-Bench, a benchmark providing explicit groundtruth edits. Our experiments demonstrate FlowDrag's superior editing quality and geometric coherence on both Drag-Bench and VFD-Bench.
this section cite: []

Section: References
Ref_id:b0 Title: Instructpix2pix: Learning to follow image editing instructions Year: (2023)
Ref_id:b1 Title: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing Year: (2023)
Ref_id:b2 Title: Rigidity controllable as-rigid-as-possible shape deformation Year: (2017)
Ref_id:b3 Title: Diffusion-based semantic image editing with mask guidance Year: (2022)
Ref_id:b4 Title: Turboedit: Text-based image editing using few-step diffusion models Year: (2024)
Ref_id:b5 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b6 Title: Prompt-to-prompt image editing with cross attention control Year: (2022)
Ref_id:b7 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b8 Title: Ita-mdt: Image-timestep-adaptive masked diffusion transformer framework for image-based virtual try-on Year: (2025)
Ref_id:b9 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b10 Title: Imagic: Text-based real image editing with diffusion models Year: (2023)
Ref_id:b11 Title: Repurposing diffusion-based image generators for monocular depth estimation Year: (2024)
Ref_id:b12 Title: Frequency-aware latent refinement for enhanced nonrigid editing Year: (2024)
Ref_id:b13 Title: Wavelet-guided acceleration of text inversion in diffusion-based image editing Year: (2024)
Ref_id:b14 Title: Tvr: A large-scale dataset for video-subtitle moment retrieval Year: (2020)
Ref_id:b15 Title: Smooth rotation enhanced asrigid-as-possible mesh animation Year: (2014)
Ref_id:b16 Title: Stylediffusion: Promptembedding inversion for text-based editing Year: (2023)
Ref_id:b17 Title: Point tracking is not you need for interactive point-based image editing Year: (2023)
Ref_id:b18 Title: Drag your noise: Interactive point-based editing via diffusion semantic propagation Year: (2024)
Ref_id:b19 Title: Dragondiffusion: Enabling drag-style manipulation on diffusion models Year: (2023)
Ref_id:b20 Title: Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing Year: (2024)
Ref_id:b21 Title: Towards photorealistic image generation and editing with text-guided diffusion models Year: (2021)
Ref_id:b22 Title: Drag your gan: Interactive point-based manipulation on the generative image manifold Year: (2023)
Ref_id:b23 Title: Text-driven manipulation of stylegan imagery Year: (2021)
Ref_id:b24 Title: The 2017 davis challenge on video object segmentation Year: (2017)
Ref_id:b25 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b26 Title: Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems Year: (2019)
Ref_id:b27 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b28 Title: Photorealistic text-to-image diffusion models with deep language understanding Year: (2022)
Ref_id:b29 Title: Dragdiffusion: Harnessing diffusion models for interactive point-based image editing Year: (2024)
Ref_id:b30 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b31 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b32 Title: As-rigid-as-possible surface modeling Year: (2007)
Ref_id:b33 Title: Emergent correspondence from image diffusion Year: (2023)
Ref_id:b34 Title: Plugand-play diffusion features for text-driven image-toimage translation Year: (2023)
Ref_id:b35 Title: Cvpr 2023 text guided video editing competition Year: (2023)
Ref_id:b36 Title: Freeinit: Bridging initialization gap in video diffusion models Year: (2023)
Ref_id:b37 Title: Textguided diverse face image generation and manipulation Year: (2021)
Ref_id:b38 Title: Structured 3d latents for scalable and versatile 3d generation Year: (2024)
Ref_id:b39 Title: Dilutional noise initialization for diffusion video editing Year: (2024)
Ref_id:b40 Title: Frequency adapting group for diffusion video editing Year: (2024)
Ref_id:b41 Title: Tpc: Test-time procrustes calibration for diffusion-based human image animation Year: (2024)
Ref_id:b42 Title: Towards good practices for drag editing with diffusion models Year: (2024)
Ref_id:b43 Title: Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation Year: (2025)
