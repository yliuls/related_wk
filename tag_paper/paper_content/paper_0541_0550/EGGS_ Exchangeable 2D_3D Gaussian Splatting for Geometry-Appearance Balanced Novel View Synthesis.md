Title: EGGS: Exchangeable 2D/3D Gaussian Splatting for Geometry-Appearance Balanced Novel View Synthesis
Abstract: Figure 1: Comparison of 3DGS, 2DGS, and our EGGS. While 3DGS achieves high-fidelity appearance, it often produces inaccurate geometry, with imprecise surfaces and blurred edges. 2DGS improves geometric consistency across views but suffers from reduced appearance quality due to over-smoothed surfaces and loss of detail. In contrast, EGGS employs an exchangeable hybrid Gaussian representation that achieves both accurate geometry and high-quality appearance.

Section: Introduction
Novel view synthesis (NVS) is a fundamental task in computer graphics and computer vision, with broad applications in augmented reality (AR), virtual reality (VR), and autonomous driving [1,2,3]. Neural Radiance Fields (NeRF) [4] reconstruct implicit radiance fields via differentiable volume rendering. Despite achieving photorealistic appearance and accurate geometry, NeRF-based methods [5,6,7,8,9,10] typically suffer from long training times and slow rendering speeds. 3D Gaussian Splatting (3DGS) [11] has emerged as an efficient alternative, leveraging anisotropic 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
23.85 22.96 0.168 0.195 1.96 0.8 0.09 0.32 20.21 23.52 0.833 0.802 PSNR LPIPS CD F1 OOD SSIM 3DGS 2DGS CD: Chamfer Distance OOD: View Consistency in Out-of-Distribution Setting
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10']

Section: Method Gaussian Type
Rasterizer Type Exchange Regularization Setting 3DGS SIGGRAPH'23 [11] 3D 3D ✗ -General SuGaR CVPR'24 [14] 3D 3D ✗ Normal General GaussianPro ICML'24 [13] 3D 3D ✗ Normal General 2DGS SIGGRAPH'24 [12] 2D 2D ✗ Depth & Normal General GS Surfels SIGGRAPH'24 [15] 2D 3D ⋆ ✗ Depth & Normal General TextureGS ECCV'24 [16] 2D 2D ✗ Depth & Normal General HybridGS CVPR'25 [17] 3D + 2D* 3D ✗ -Transient HorizonGS CVPR'25 [18] 3D Comparison between EGGS and related works. Prior works either use only single representation or do not explore complementary advantages of 3D and 2D Gaussians. ⋆ Gaussian Surfel [15] directly sets the z-scale of 3D Gaussian to zero and uses the rasterizer from 3DGS. * HybridGS [17] uses image-frame single-view 2D Gaussians [19,20] instead of 2D Gaussians in the 3D space [12].
3D Gaussians for real-time, high-quality rendering. While 3DGS excels in appearance fidelity, its anisotropic nature often leads to multi-view inconsistencies, limiting geometric accuracy [12,13]. As shown in Figure 1, this can lead to inaccurate edges and surfaces.
Following 3DGS, a line of work has focused on improving its geometric accuracy and reconstruction quality through additional regularization and novel representations, as shown in Figure 2 (right). SUGAR [14] and GaussianPro [13] introduce normal-based regularization, such as planar loss, to align Gaussian normals and encourage flatter shapes, thereby improving surface consistency. Gaussian Surfles [15] and GOF [21] incorporate additional geometry-aware constraints to enhance spatial coherence. 2D Gaussian Splatting (2DGS) [12] replaces 3D ellipsoids with 2D surfels, significantly improving multi-view consistency and geometric accuracy, as shown in Figure 2 (left). However, this comes at the cost of degraded appearance quality, as surfel-based representations struggle to preserve high-frequency details. TextureGS [16] attempts to decouple appearance and geometry within the 2DGS framework, but the single representation still limits overall rendering performance. Recently, HybridGS [17] combines 3DGS with image-space 2D Gaussians to address transient objects, but its radiance field remains fully represented by 3D Gaussians. HorizonGS [18], designed for varying-altitude scenes, decodes 2D Gaussians for surface reconstruction and 3D Gaussians for view synthesis separately via an MLP in ScaffoldGS [22]. While effective in their target domains, these methods do not explore a unified hybrid radiance representation. As a result, the complementary strengths of 2DGS and 3DGS in geometry and appearance remain underutilized.
Effectively combining 3D and 2D Gaussians to jointly improve appearance and geometry is non-trivial, as simply mixing the two representations does not necessarily improve reconstruction quality [18].
To start, the geometric accuracy of 2D Gaussians relies on a ray-splat-intersection-based rasterizer designed to enforce multi-view consistency. Using the projection-based 3DGS rasterizer to render 2D Gaussians can lead to suboptimal geometry [15]. Moreover, Gaussian parameters change significantly during training. For instance, 3D Gaussians may flatten to approximate surfaces, while 2D Gaussians may expand volumetrically to capture thin structures or translucent effects. Fixing the Gaussian type throughout optimization can limit the model's expressiveness. Finally, relying solely on photometric loss is insufficient to balance geometry and appearance. Additional regularization is required to guide the optimization of hybrid representations. Most importantly, the regularization strategy should account for the distinct characteristics of 3D and 2D Gaussians.
In response to these challenges, we introduce Exchangeable Gaussian Splatting (EGGS), an adaptive hybrid representation that unifies 2D and 3D Gaussian splatting in a single framework. EGGS provides a practical and efficient solution for high-quality novel view synthesis and 3D reconstruction.
Our main contributions are as follows:
• To preserve the complementary strengths of 3D and 2D Gaussians, we develop Hybrid Gaussian Rasterization, a unified rendering framework that supports both projection-based and ray-splat-intersection-based rasterization. We implement this framework with CUDA for efficient optimization, and ensure compatibility with existing 3DGS and 2DGS pipelines.
• We propose Adaptive Type Exchange, which enables an exchangeable hybrid of 2D and 3D
Gaussians. We use effective rank as an auxiliary criterion to determine whether each Gaussian should dynamically switch its type during training, resulting in a more flexible and contentadaptive representation. 2
• To better balance geometry and appearance, we introduce Frequency-Decoupled Optimization, a regularization strategy in the frequency domain. Using the Discrete Wavelet Transform (DWT), we extract low-frequency components to guide scene geometry and high-frequency components to refine appearance. We supervise 3D and 2D Gaussians asymmetrically to exploit their distinct characteristics, where high-frequency signals guide 3D Gaussians toward detailed appearance, while low-frequency signals supervise 2D Gaussians for geometric consistency.
• We conduct extensive experiments demonstrating that EGGS significantly improves the trade-off between appearance fidelity and geometric accuracy. It outperforms both 3DGS and 2DGS in appearance quality, while achieving geometric accuracy and multi-view consistency comparable to 2DGS. Moreover, EGGS serves as a versatile representation that performs well in challenging scenarios such as few-shot and out-of-distribution view synthesis.
this section cite: ['b10', 'b13', 'b12', 'b11', 'b14', 'b15', 'b16', 'b17', 'b14', 'b16', 'b18', 'b19', 'b11', 'b11', 'b12', 'b13', 'b12', 'b14', 'b20', 'b11', 'b15', 'b16', 'b17', 'b21', 'b17', 'b14']

Section: Related Works
Radiance Fields for Novel View Synthesis. Neural Radiance Fields (NeRF) [4] have emerged as a fundamental approach for novel view synthesis [23], representing scenes as continuous volumetric functions optimized via differentiable rendering. While NeRF achieves high-fidelity reconstruction, it requires dense sampling and significant computational resources. Subsequent works have improved either quality [24,25] or efficiency [5,7,26,6], but the excessive training and rendering time remains a major bottleneck. To address this, recent efforts have explored more efficient alternatives, such as 3D Gaussian Splatting (3DGS) [11], which represents scenes using a set of 3D Gaussians that can be efficiently rasterized and optimized for real-time rendering. To further improve the performance and efficiency of 3DGS, several extensions have been proposed. ScaffoldGS [22] introduces a voxel-based representation where an MLP is used to decode 3D Gaussians within each voxel. 3DGS-MCMC [27] formulates Gaussian densification as a Markov Chain Monte Carlo sampling process, enabling a more efficient and adaptive distribution of Gaussians across the scene.
this section cite: ['b3', 'b22', 'b23', 'b24', 'b4', 'b6', 'b25', 'b5', 'b10', 'b21', 'b26']

Section: Geometry-Appearance-Balanced Gaussian Splatting.
While 3DGS achieves high appearance fidelity and is efficient in both training and rendering, the anisotropic nature of 3D Gaussians often exhibits multi-view inconsistency, resulting in limited geometric accuracy. To address this, several works propose geometry regularization techniques. SUGAR [14] and GaussianPro [13] introduce normal-based regularization to encourage flatter Gaussians that better align with scene surfaces. Gaussian Surfels [15] and GOF [28] further enforce depth accuracy and normal consistency to enhance geometric reconstruction. Instead of relying solely on regularization, 2DGS [12] adopts a 2D surfel representation with a specialized ray-splat-intersection rasterizer, ensuring multi-view consistency and significantly improving geometric accuracy compared to 3DGS. It also incorporates additional depth and normal regularization. However, this comes at the cost of reduced appearance quality, as 2D surfels struggle to preserve high-frequency detail. TextureGS [16] attempts to decouple geometry and appearance modeling within the 2DGS framework, but its appearance fidelity remains limited due to the inherent drawbacks of the 2D representation.
As demonstrated in Figure 2 (left), 3D Gaussians achieve better appearance quality in PSNR, SSIM, and LPIPS. In contrast, 2D Gaussians offer superior view consistency and geometric fidelity, resulting in more robust PSNR under out-of-distribution (OOD) conditions, improved point cloud accuracy in Chamfer Distance (CD), and higher depth accuracy in F1 score. As shown in Figure 2 (right), most existing methods [12,11,14,13,29,30,22,31,32,33,34] rely on a single Gaussian representation to reconstruct radiance fields, which limits their flexibility and adaptability. Although HybridGS [17] incorporates both 3D Gaussians and image-space 2D Gaussians to better handle transient content, its radiance field remains solely represented by 3D Gaussians. A radiance field that jointly leverages both 2D and 3D Gaussians remains largely unexplored. It is still unclear how 2D and 3D Gaussians can be made exchangeable during training and how to fully exploit their complementary strengths in appearance and geometry. We provide a more detailed discussion in Appendix B.
this section cite: ['b13', 'b12', 'b14', 'b27', 'b11', 'b15', 'b11', 'b10', 'b13', 'b12', 'b28', 'b29', 'b21', 'b30', 'b31', 'b32', 'b33', 'b16']

Section: Method
We provide an overview of the EGGS framework in Figure 3. To enable the joint training of 2D and 3D Gaussians within a unified framework, we first introduce Hybrid Gaussian Rasterization in Section 3.1, which supports both ray-splat-intersection-based rendering for 2D Gaussians and projection-based rendering for 3D Gaussians. Next, we present Adaptive Type Exchange in Section 3.2,
Hybrid Rasterization 2D Gaussians 3D Gaussians Frequency-Decoupled Regulation ℒ !"#"$ Hybrid Gaussian Model 𝑥 ! SfM & Initialization Sec 3.1 ℒ #"% ℒ &'(& Adaptive Type Exchange DWT DWT Sec 3.2 Sec 3.3 Projection Input Ray-splat intersection 2D → 3D 3D → 2D
erank=2.3 erank=2.3 erank=1.9 erank=1.9
Figure 3: Overview of the EGGS framework. We initialize 2D and 3D Gaussians from sparse points obtained via structure-from-motion (SfM) [35,36]. Their parameters are then jointly optimized using our CUDA-accelerated differentiable hybrid rasterization. To enhance the flexibility of the hybrid representation, Adaptive Type Exchange is introduced to allow each Gaussian to switch between 2D and 3D types during training. Finally, we apply Discrete Wavelet Transform (DWT) [37] and introduce Frequency-Decoupled Optimization to balance geometric accuracy and appearance fidelity.
which enables dynamic switching between 2D and 3D types during optimization. Finally, to optimize the hybrid model for balanced geometric consistency and appearance fidelity, we propose Frequency-Decoupled Optimization in Section 3.3, a supervision strategy that leverages the distinct frequency characteristics of 2D and 3D Gaussians.
this section cite: ['b34', 'b35', 'b36']

Section: Hybrid Gaussian Rasterization
Differentiable rasterization was introduced in 3DGS to enable gradient-based optimization of Gaussian parameters using a projection-based pipeline for real-time rendering. 2DGS later developed a ray-splat-intersection-based rasterizer tailored to 2D surfel representations, improving multi-view consistency and geometric accuracy. However, the architectural distinction between these two rasterization pipelines makes it non-trivial to render and optimize a hybrid model within a unified framework. While 2D Gaussians can be viewed as degenerate 3D Gaussians with zero scale along the z-axis, directly rendering them with the 3D rasterizer leads to geometric inaccuracies [12]. This is due to the affine projection approximation used in 3DGS, which introduces distortion at all points except the Gaussian center. We further analyze this issue in Section 4.2.
To leverage the complementary strengths of 3D and 2D Gaussians, it is necessary to render them within a unified framework. To this end, we propose Hybrid Gaussian Rasterization, which integrates both projection-based and ray-splat-intersection-based pipelines. In our rasterizer, each Gaussian primitive G is parameterized by a center µ ∈ R 3 , scale s ∈ R 3 , rotation quaternion r ∈ R 4 , opacity α ∈ R, and spherical harmonic (SH) color coefficients f ∈ R 3×(l+1) 2 , where l is the degree of view-dependent color. The view-dependent RGB color c is decoded from f . The Gaussian shape is defined by the covariance matrix Σ = RSS T R T , where R ∈ R 3×3 is the rotation matrix derived from r, and S = diag(s x , s y , s z ) ∈ R 3×3 is the scaling matrix. We augment each Gaussian with a type specifier t ∈ {0, 1} to indicate whether it is a 2D (t = 0) or 3D (t = 1) Gaussian. 2D Gaussians are initialized with s z = 0, while the remaining parameters follow the initialization of 3DGS. As shown in Figure 4, we rasterize Gaussians according to their types, where affine projection is used for 3D Gaussians and ray-splat-intersection is used for 2D Gaussians. The contribution of G 3d i and G 2d i is evaluated by computing the distance from the image-space pixel
x p to G proj i or G 2d
i in the 2D image plane or tangent frame, respectively:
di = (ui(xp) 2 + vi(xp) 2 if ti = 0 (xp -µ ′ 3d,i ) T Σ ′-1 i (xp -µ ′ 3d,i ) otherwise (1
)
where µ ′ 3d,i and Σ i are the projected center and covariance of the 3D Gaussian computed via affine projection, and u i (x p ) and v i (x p ) denote the coordinates of the intersection between the ray through x p and the 2D Gaussian. The distance d i can be computed simultaneously for both 3D and 2D Gaussians. The final contribution of each Gaussian is then computed uniformly as αi = α i e -1 2 di , where α i is the opacity of the i-th Gaussian. With the above formulation, both 3D and 2D Gaussians can be rendered in a single α-blending pass:
C(x p ) = i∈N c i αi i-1 j=1 (1 -αj )(2)
where the final color at pixel x p is computed from color c i and contribution αi of each Gaussian primitive. To support efficient and parallel rendering, we implement our hybrid rasterizer in CUDA. More details on initialization and densification are provided in Appendix A, and those on projectionbased and ray-splat-intersection-based rasterization procedures are deferred to Appendix C.
this section cite: ['b11']

Section: Adaptive Type Exchange
While the type specifier introduced in Section 3.1 enables unified rendering of 2D and 3D Gaussians, each Gaussian primitive is initialized with a fixed type. Such fixed type assignment can limit the expressiveness of the model, as Gaussians may naturally deviate from their initial type during optimization. For example, 3D Gaussians may become increasingly flat to better model surfaces, while 2D Gaussians may take on more volumetric properties to capture semi-transparent regions. To fully exploit the flexibility of the hybrid model, the type of each Gaussian should dynamically adapt to its evolving geometric characteristics. To this end, we propose Adaptive Type Exchange, which allows each Gaussian to switch between 2D and 3D types during training.
The key to Adaptive Type Exchange is detecting discrepancies between a Gaussian's assigned type and its effective geometric dimensionality. Therefore, we introduce the effective rank (erank) [38,39] as an indicator of this dimensionality, allowing the model to determine when type switching is needed during training. Given a Gaussian G with scaling s = (s x , s y , s z ), we define its effective rank as:
erank(G) = exp - 2 i=0 qi ∥q∥1 log qi ∥q∥1 , where q = (s 2 x , s 2 y , s 2 z ).(3)
As illustrated in Figure 5, erank provides a principled signal for deciding when to switch types. A perfectly isotropic 3D Gaussian has erank = 3, while a flattened Gaussian approaches erank = 2.
If a Gaussian primitive G i is assigned as 3D (t i = 1) but its effective rank falls below a threshold θ e , we mark it for conversion to 2D by setting t ′ i = 0. Similarly, we update 2D Gaussians to 3D (t ′ i = 1) when their effective rank exceeds the threshold. Yet, merely flipping the type specifier can lead to unstable parameter transitions, as the s z scale is treated as least significant in 2D Gaussians. To ensure stable conversion, we reparameterize the covariance of 3D Gaussians during switching and adjust gradient flow to s z for 2D Gaussians.
Reparameterization. (3D → 2D) All three scales of a 3D Gaussian are initially optimized, whereas 2D Gaussians ignore the s z scale during ray-splat-intersection-based rasterization. Accordingly, when converting a 3D Gaussian to 2D, only s x and s y are retained and s z is discarded. However, directly discarding s z can lead to instability during training when it is not the least significant scale. To prevent this, we reparameterize the 3D Gaussian before conversion so that s z corresponds to the smallest axis. The key to stable conversion is aligning s z with the least significant scale while preserving covariance Σ = RSS T R T . We first construct the converted scaling matrix S * using a permutation matrix P that moves
erank=2.8 erank=2.8 erank=1.8 erank=1.8 Effective rank threshold 𝜃 ! = 2.05 2D Gaussians 3D Gaussians Reparameterization Scale modulation Unchanged erank=2.3 erank=2.3 2𝐷 → 3𝐷 erank=1.9 erank=1.9 3𝐷 → 2𝐷 the least significant scale to the z-axis:
S * = P SP T (4
)
P is set to P x or P y if s x or s y is the least significant scale, respectively, where:
P x =   0 1 0 0 0 1 1 0 0   , P y =   0 0 1 1 0 0 0 1 0   (5)
Then, to ensure the covariance Σ remain unchanged in Σ * = R * S * S * R * T , we set the rotation of 2D Gaussian as R * = RP T . We note that R * is converted to quaternions during optimization. To ensure a valid conversion, the rotation matrix R * must be orthogonal with a positive determinant. P x and P y are designed to preserve these properties. Additional details are provided in Appendix D. aligning geometric expressiveness with visual transparency. The intuition behind this design is that 2D-to-3D transitions often occur in regions with semi-transparent or volumetric effects that flat primitives cannot represent well. This modulation enables s z to be optimized throughout training, while ensuring its updates remain stable and smoothly conditioned on opacity:
α * i = α i e -λz×s * z (6
)
where s * z denotes the activated z-axis scale, computed via a soft gating function:
s * z = sigmoid sz-θz Tz s z (7
)
The soft scale modulation allows a 2D Gaussian to remain effectively two-dimensional when s z is insignificant, in which case s * z approaches zero. Conversely, as a 2D Gaussian evolves toward a more volumetric form, an increasing s z leads to reduced opacity, effectively enabling the representation of semi-transparent or volumetric effects. Additional details on the effective rank threshold, 3D Gaussian reparameterization and permutation, and 2D Gaussian scale modulation are provided in Appendix D.
this section cite: ['b37', 'b38']

Section: Frequency-Decoupled Optimization
With our hybrid rasterization and adaptive type exchange mechanism, 2D and 3D Gaussians can be jointly optimized within a unified and flexible framework. However, relying solely on photometric loss is insufficient to effectively optimize the hybrid model for balanced geometry and appearance. 2D and 3D Gaussians exhibit distinct characteristics during optimization and specialize in different aspects of the scene. 2D Gaussians are better suited for enforcing geometric consistency, while 3D Gaussians excel at capturing high-frequency appearance details. To fully leverage these complementary strengths, we introduce Frequency-Decoupled Optimization, a supervision strategy that decouples low-and high-frequency components and assigns them asymmetrically to 2D and 3D Gaussians, respectively.
Frequency Decoupling via Discrete Wavelet Transform. As shown in Figure 6, scene information can be effectively separated in the frequency domain. High-frequency components typically correspond to fine details that are refined in later training stages (e.g., 7K iterations), while low-frequency components capture overall scene geometry and are optimized earlier. This frequency-based separation aligns well with the complementary roles of 3D Gaussians in modeling appearance and 2D Gaussians in capturing geometry. Motivated by this, we introduce Frequency-Decoupled Optimization to supervise the hybrid model in the frequency domain. We apply DWT [37] to decompose the ground truth image I into low-and high-frequency components: I l , I h = DWT(I). The same transformation is applied to the rendered image Î to obtain Îl and Îh .
g ! "#$ g ! %!&% g ! '#"#( Conflicting frequency gradients Project Non-conflicting Gradients Normal direction of g ! "#$
Figure 7: Illustration of frequency gradient projection. We project gradients from highfrequency loss onto the normal vector of gradients from low-frequency for 2D Gaussians.
The frequency loss is defined as
L i = ∥ Îi -I i ∥ 2 2
for i ∈ {low, high}. We also include the standard appearance loss used in 3DGS:
L color = (1 -λ)L 1 + λL D-SSIM .
With access to frequency-specific losses, a naïve strategy is to combine all terms and apply them uniformly to all Gaussians:
L = L color + λ low L low + λ high L high .(8)
where L low and L high are applied equally to all 2D and 3D Gaussians. While supervision is decoupled in the frequency domain, this approach overlooks the distinctions of each representation.
this section cite: ['b36']

Section: Asymmetrical Gradient Update with Projected Conflicts.
We denote the gradients to G i from L color , L low and L high as g color i , g low i and g high i , respectively. As illustrated in Figure 7, conflicting gradients can arise when losses from different frequency components are directly applied to update Gaussian parameters (i.e., Eq.( 8)), diminishing the effectiveness of frequency-based regularization. Such conflicts stem from the distinct characteristics of 2D and 3D Gaussians. 2D Gaussians are more effective at capturing overall geometry and ensuring multi-view consistency, where low-frequency signals offer more relevant guidance, while high-frequency gradients may counteract this by encouraging appearance-driven updates. Conversely, 3D Gaussians
this section cite: []

Section: Algorithm 1: Frequency-Decoupled Optimization
Require :Gaussians {Gi} N -1 i=0 , appearance loss L color , frequency loss L low and L high .
g color , g low , g high ← ∇GL color , ∇GL low , ∇GL high ; // Process per Gaussian gradient conflict for i ← 0 to N -1 do // There are conflict Gradients in different frequencies if g low i • g high i < 0 then if ti == 0 then g high i ← g high i -g high •g low ||g low || 2 g low ; (9
) // Type is 2D, project g high i onto normal of g low i else g low i ← g low i -g high •g low ||g high || 2 g high ; (10
) // Type is 3D, project g low i onto normal of g high i ∆Gi = g color i + g low i + g high i return Update ∆G
specialize in modeling fine-scale appearance and benefit more from high-frequency supervision, whereas low-frequency signals contribute less to their performance.
To address this issue, we propose an asymmetrical update strategy that applies frequency supervision based on Gaussian type, as shown in Algorithm 1. For each Gaussian, we check for potential gradient conflicts by computing the inner product between g low i and g high i , where a negative value indicates divergent update directions [40]. When such conflict is detected, we retain the frequency component most relevant to the Gaussian type and project the other. Specifically, for 2D Gaussians, we preserve supervision from lowfrequency and remove the conflicting component of high-frequency by projecting g high i onto the normal vector of g low i , as shown in Eq.( 9). Similarly, for 3D Gaussians, we retain g high i and project g low i as indicated in Eq.( 10). This asymmetrical supervision ensures each Gaussian is updated along its most informative direction while minimizing interference from less relevant frequency signals. More details on DWT and the gradient projection strategy are provided in Appendix E and Appendix F, respectively.
this section cite: ['b39']

Section: Experiments
Datasets and Metrics. We evaluate EGGS on several widely used benchmarks. For appearance evaluation, we use Mip-NeRF360 [25], LLFF [41], Tanks&Temples [42], and DTU [43]. For geometry evaluation, we use DTU, which provides ground-truth point clouds, and Tanks&Temples, which offers ground-truth depth maps. Additional dataset details are provided in Appendix A. Following prior work [25,11,13,12], we report PSNR, SSIM [44], and LPIPS [45] to evaluate the appearance quality of synthesized novel views. For geometry, we follow [12,39] and report Chamfer Distance [46] on DTU to assess reconstruction accuracy.
Baselines. To demonstrate the effectiveness of EGGS, we compare against several singlerepresentation methods that use either 3D or 2D Gaussians. For 3D Gaussian-based methods, we include vanilla 3DGS [11], GaussianPro [13] and GOF [28], which incorporate geometric regularization, and FreGS [47], which introduces frequency-based supervision. For 2D Gaussian-based methods, we consider vanilla 2DGS [12] and TextureGS [16], which improves the appearance fidelity of 2D Gaussians. Additional discussion of related methods is provided in Appendix B.
Implementation. We implement our hybrid rasterizer based on the CUDA rasterization code of 3DGS [11]. We used the Haar filter for the DWT [37,48]. For Frequency-Decoupled Optimization, we set the weight for the frequency components as λ low = 0.2 and λ low = 0.4. For Adaptive Type Switch, we set the erank threshold as 2.05. We offer more details about our training pipeline and parameter setting in Appendix A. All experiments are conducted on an A5000 GPU.
this section cite: ['b24', 'b40', 'b41', 'b42', 'b24', 'b10', 'b12', 'b11', 'b43', 'b44', 'b11', 'b38', 'b45', 'b10', 'b12', 'b27', 'b46', 'b11', 'b15', 'b10', 'b36', 'b47']

Section: 2DGS
Ours GaussianPro  Geometry. We evaluate geometry reconstruction quality on Tanks&Temples and DTU. As shown in Figure 9, both 2DGS and EGGS produce more accurate depth maps than 3DGS, with sharper surfaces and clearer edges. However, 2DGS sacrifices appearance fidelity due to the lack of high-frequency detail. In contrast, EGGS improves geometry over 3DGS while also preserving appearance quality. Table 2 reports Chamfer Distance on the DTU dataset, where EGGS outperforms 3DGS and SUGAR. Note that SUGAR, 2DGS, and GOF prioritize surface reconstruction and mesh extraction, often at the cost of appearance. Although 2DGS is slightly more accurate geometrically, EGGS achieves a better trade-off, offering stronger appearance quality alongside competitive geometry.
this section cite: []

Section: Efficiency.
Table 3 compares the model size and training time of EGGS with 3DGS, 2DGS, and GaussianPro on LLFF and Tanks&Temples. While 2DGS uses the fewest Gaussians, its training time exceeds that of 3DGS. GaussianPro enhances appearance quality over 3DGS but incurs significantly higher training cost. In contrast, EGGS strikes a favorable balance, requiring fewer Gaussians than both 3DGS and GaussianPro, while achieving the shortest training time among all methods.
this section cite: []

Section: RGB Depth Map
Ground Truth 3DGS 2DGS Ours
this section cite: []

Section: Ablation and Generalization Analysis
Ablation Study. We evaluate the effectiveness of each component in EGGS in Table 4. Row i is the vanilla 3DGS baseline. In row ii, we adopt a hybrid 2D/3D representation but rasterize all Gaussians using the 3DGS rasterizer, which leads to performance degradation. Row iii incorporates our hybrid rasterizer, which renders Gaussians according to their type. However, this setting still lacks flexibility and regularization. Row iv incorporates adaptive type exchange to enhance the flexibility. Rows v-vii study frequency-based supervision, which provides only limited gains for non-hybrid 3DGS (row v) but is more effective in the hybrid setting. The full model in row vii achieves the best performance, indicating that decoupled frequencies more effectively exploit the strengths of the exchangeable hybrid representation. We provide more ablation and analysis in Appendix F.
Training View Test View Ground Truth 3DGS 2DGS Ours
this section cite: []

Section: Generalization Analysis.
We evaluate the robustness of EGGS in challenging scenarios, including few-shot and out-of-distribution (OOD) settings. Following prior work, we use LLFF [41] for few-shot evaluation [49,50] and OOD-NVS [51] for OOD evaluation [52]. More details are provided in Appendix A. As shown in Table 11 and Figure 10, EGGS achieves robust performance in both settings, benefiting from its balanced multi-view consistency and appearance fidelity. This indicates that the hybrid representation generalizes better than single-type baselines.
We also emphasize that EGGS serves as a general underlying representation and is compatible with various optimization strategies developed for specialized settings [49,50,53,54,55,51]. We discuss these orthogonal techniques in Appendix B, and provide further remarks on limitations and broader impacts in Appendix H.
this section cite: ['b40', 'b48', 'b49', 'b50', 'b51', 'b48', 'b49', 'b52', 'b53', 'b54', 'b50']

Section: Conclusion
This paper presents EGGS, a hybrid Gaussian Splatting framework that combines the appearance fidelity of 3D Gaussians with the geometric accuracy of 2D Gaussians. The design integrates Hybrid Gaussian Rasterization for unified rendering, Adaptive Type Exchange for flexible representation, and Frequency-Decoupled Optimization to balance geometry and appearance. EGGS outperforms both 2D-and 3D-only baselines across multiple benchmarks. Future work includes extending the hybrid representation to more diverse and challenging scenarios.
Table 6: Details on the datasets used for evaluation of appearance and geometry.
this section cite: []

Section: References
Ref_id:b0 Title: Text-to-3d using gaussian splatting Year: (2024)
Ref_id:b1 Title: Art3d: 3d gaussian splatting for text-guided artistic scenes generation Year: (2024)
Ref_id:b2 Title: Dc-gaussian: Improving 3d gaussian splatting for reflective dash cam videos Year: (2024)
Ref_id:b3 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b4 Title: Tensorf: Tensorial radiance fields Year: (2022)
Ref_id:b5 Title: Block-nerf: Scalable large scene neural view synthesis Year: (2022)
Ref_id:b6 Title: Plenoxels: Radiance fields without neural networks Year: (2022)
Ref_id:b7 Title: Tetra-nerf: Representing neural radiance fields using tetrahedra Year: (2023)
Ref_id:b8 Title: Urban radiance fields Year: (2022)
Ref_id:b9 Title: Nerfstudio: A modular framework for neural radiance field development Year: (2023)
Ref_id:b10 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b11 Title: 2d gaussian splatting for geometrically accurate radiance fields Year: (2024)
Ref_id:b12 Title: Gaussianpro: 3d gaussian splatting with progressive propagation Year: (2024)
Ref_id:b13 Title: Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering Year: (2024)
Ref_id:b14 Title: Highquality surface reconstruction using gaussian surfels Year: (2024)
Ref_id:b15 Title: Texture-gs: Disentangling the geometry and texture for 3d gaussian splatting editing Year: (2024)
Ref_id:b16 Title: Hybridgs: Decoupling transients and statics with 2d and 3d gaussian splatting Year: (2024)
Ref_id:b17 Title: Horizon-gs: Unified 3d gaussian splatting for large-scale aerial-to-ground scenes Year: (2024)
Ref_id:b18 Title: Image-gs: Content-adaptive image representation via 2d gaussians Year: (2024)
Ref_id:b19 Title: Gaussianimage: 1000 fps image representation and compression by 2d gaussian splatting Year: (2024)
Ref_id:b20 Title: Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes Year: (2024)
Ref_id:b21 Title: Scaffoldgs: Structured 3d gaussians for view-adaptive rendering Year: (2024)
Ref_id:b22 Title: Novel view synthesis in tensor space Year: (1997)
Ref_id:b23 Title: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields Year: (2021)
Ref_id:b24 Title: Mipnerf 360: Unbounded anti-aliased neural radiance fields Year: (2022)
Ref_id:b25 Title: Instant neural graphics primitives with a multiresolution hash encoding Year: (2022)
Ref_id:b26 Title: 3d gaussian splatting as markov chain monte carlo Year: (2024)
Ref_id:b27 Title: Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes Year: (2024)
Ref_id:b28 Title: A hierarchical 3d gaussian representation for real-time rendering of very large datasets Year: (2024)
Ref_id:b29 Title: Mip-splatting: Alias-free 3d gaussian splatting Year: (2024)
Ref_id:b30 Title: Adaptive 3d gaussian representation from probabilistic masks Year: (2024)
Ref_id:b31 Title: Supergaussians: Enhancing gaussian splatting using primitives with spatially varying colors Year: (2024)
Ref_id:b32 Title: Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps Year: (2024)
Ref_id:b33 Title: Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces Year: (2024)
Ref_id:b34 Title: Structure-from-motion revisited Year: (2016)
Ref_id:b35 Title: Pixelwise view selection for unstructured multi-view stereo Year: (2016)
Ref_id:b36 Title: Continuous and discrete wavelet transforms Year: (1989)
Ref_id:b37 Title: The effective rank: A measure of effective dimensionality Year: (2007)
Ref_id:b38 Title: Effective rank analysis and regularization for enhanced 3d gaussian splatting Year: (2024)
Ref_id:b39 Title: Gradient surgery for multi-task learning Year: (2020)
Ref_id:b40 Title: Local light field fusion: Practical view synthesis with prescriptive sampling guidelines Year: (2019)
Ref_id:b41 Title: Tanks and temples: Benchmarking large-scale scene reconstruction Year: (2017)
Ref_id:b42 Title: Large scale multi-view stereopsis evaluation Year: (2014)
Ref_id:b43 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b44 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b45 Title: Recovering intrinsic scene characteristics Year: (1978)
Ref_id:b46 Title: Fregs: 3d gaussian splatting with progressive frequency regularization Year: (2024)
Ref_id:b47 Title: Wavelets and filter banks Year: (1996)
Ref_id:b48 Title: Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization Year: (2024)
Ref_id:b49 Title: Fsgs: Real-time few-shot view synthesis using gaussian splatting Year: (2024)
Ref_id:b50 Title: Splatformer: Point transformer for robust 3d gaussian splatting Year: (2025)
Ref_id:b51 Title: Mip-splatting: Alias-free 3d gaussian splatting Year: (2024)
Ref_id:b52 Title: Gaussian splatting with few view matching and multi-stage training Year: (2024)
Ref_id:b53 Title: Excavating multi-view priors for gaussian splatting from sparse input views Year: (2024)
Ref_id:b54 Title: Consolidating gaussian surfel splatting for sparse-view surface reconstruction Year: (2024)
Ref_id:b55 Title: E3d-bench: A benchmark for end-toend 3d geometric foundation models Year: (2025)
Ref_id:b56 Title: Dust3r: Geometric 3d vision made easy Year: (2024)
Ref_id:b57 Title: Grounding image matching in 3d with mast3r Year: (2024)
Ref_id:b58 Title: Vggt: Visual geometry grounded transformer Year: (2025)
