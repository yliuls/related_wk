Title: Gaussian Herding across Pens: An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS
Abstract: 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for radiance field rendering, but it typically requires millions of redundant Gaussian primitives, overwhelming memory and rendering budgets. Existing compaction approaches address this by pruning Gaussians based on heuristic importance scores, without global fidelity guarantee. To bridge this gap, we propose a novel optimal transport perspective that casts 3DGS compaction as global Gaussian mixture reduction. Specifically, we first minimize the composite transport divergence over a KDtree partition to produce a compact geometric representation, and then decouple appearance from geometry by fine-tuning color and opacity attributes with far fewer Gaussian primitives. Experiments on benchmark datasets show that our method (i) yields negligible loss in rendering quality (PSNR, SSIM, LPIPS) compared to vanilla 3DGS with only 10% Gaussians; and (ii) consistently outperforms stateof-the-art 3DGS compaction techniques. Notably, our method is applicable to any stage of vanilla or accelerated 3DGS pipelines, providing an efficient and agnostic pathway to lightweight neural rendering. The code is publicly available at https://github.com/DrunkenPoet/GHAP 3DGS PUP-3DGS GHAP-3DGS (ours)

Section: Introduction
Real-time 3D scene reconstruction and rendering dynamically generates photorealistic 3D representations from sensor data (e.g., multi-view images, LiDAR) with minimal latency, enabling critical applications in augmented/virtual reality (AR/VR), autonomous navigation, and immersive media [1,2]. The current state-of-the-art, 3D Gaussian Splatting (3DGS) [3], iteratively learns 3D anisotropic Gaussian primitives with color and opacity attributes to model these scenes. During rendering, these 3D Gaussians are projected to 2D screens and α-blended to achieve real-time photorealistic synthesis.
However, 3DGS faces significant efficiency challenges: its iterative densification process often produces millions of redundant Gaussians for complex scenes [4,5,6]. This inefficiency leads to high memory/storage costs and increased per-frame rendering time, limiting deployment on resource-constrained platforms like mobile and AR/VR devices [7].
Figure 2: Comparison of heuristic pruning and our method. The original mixture (left) with 10 3 components is reduced to 5% using either pruning (middle) or our method (right). Our method better preserves the overall structure.
A common strategy to improve efficiency is compaction [8,9,10,7]reducing the number of Gaussians while preserving rendering fidelity. Fewer primitives result in reduced storage needs and faster rendering, improving per-frame performance. This approach is viable because 3DGS densification inherently generates redundant primitives [4,5,6]. Existing methods achieve compaction via pruning or random subset selection [11,12,5,13,14,15,6,4,16]. These strategies naïvely discard Gaussian primitives; while simple to implement, they are often ineffective. In particular, they tend to lose critical structural details (as shown in Fig. 1) or distort the underlying geometry (as shown in Fig. 2). Such losses can degrade rendering quality, particularly in regions with fine-scale features or complex material properties. These limitations motivate our key question:
How to design an efficient compaction method that preserves 3D spatial and structural geometry?
To address this, we frame compaction as an optimization problem, where the goal is to approximate the original 3DGS representation with fewer Gaussians. Our solution leverages a statistical perspective, treating the scene's geometric structure as a probabilistic model and employing principled reduction techniques to preserve fidelity.
(i) Geometric compaction via GMR. We first observe that the geometry of a 3DGS representationdefined by the positions, covariances, and opacities of its Gaussian primitivescan be interpreted as a Gaussian mixture model (GMM). Here, the mixture density is a convex combination of individual Gaussian densities, weighted by their opacities. This formulation naturally connects 3DGS compaction to Gaussian Mixture Reduction (GMR), a well-studied problem in statistics where a high-order GMM is approximated by one with fewer components while minimizing a divergence measure.
(ii) Appearance optimization. While geometry is compacted via GMR, the appearance of the scenegoverned by the color attributes of the Gaussiansmust also be preserved. To achieve this, we decouple the optimization of geometry (position, covariance) and appearance (color, opacity). After GMR-based compaction, we fine-tune the color and opacity of the reduced set of Gaussians using the standard 3DGS training pipeline. This two-stage strategy ensures that the compacted model maintains both geometric accuracy and photorealistic rendering quality.
For GMR, we minimize the composite transportation divergence [17], which is rooted in optimal transport theory [18,19] and allows an effective algorithm. We tailor this GMR algorithm (detailed in Section 3.2.1) so that it scales well in scenes like 3DGS with an extensive amount of Gaussians. Crucially, our GMR optimizer does not merely select a subset of existing Gaussians; instead, it creates new primitives that can dynamically adjust their positions and covariances to better approximate the underlying geometry (as shown in Figure 2). Our method is fundamentally algorithm-agnostic: it functions as a plug-and-play module that can enhance both the standard 3DGS pipeline and any of its variants (Section 4), applicable at any stage of training to boost computational efficiency.
To summarize, our contributions are:
• We open a new pathway to view 3DGS representations as a Gaussian mixture and perform compaction from the perspective of Gaussian mixture reduction via optimal transport. This contrasts with prior compaction methods that ignore geometric structure and often produce distortions, whereas our approach preserves geometric fidelity, offering a new and impactful direction for 3DGS.
• We are the first to adapt GMR to 3DGS. We introduce a novel cost function that yields closedform, low-cost updates. We also develop a block-wise GMR algorithm guided by a KD-tree, enabling efficient largescale scene compaction. These strategies are non-trivial and bridges theory with practical scalability.
• Our method is post-hoc and compatible with any existing 3DGS pipeline, making it highly practical and broadly applicable. With minimal overhead, our approach achieves SOTA compaction performance, both in quality and efficiency.
• Empirical results demonstrate that our method preserves rendering quality at 10% retention ratio.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b6', 'b3', 'b4', 'b5', 'b10', 'b11', 'b4', 'b12', 'b13', 'b14', 'b5', 'b3', 'b15', 'b16', 'b17', 'b18']

Section: Related Works
Compaction Techniques in 3DGS. 3DGS compaction seeks to minimize the Gaussian count while preserving image quality. Existing work falls into two operations: densification (where to add) and pruning (what to drop). Most current approaches rely on per-Gaussian heuristic scores.
For densification, Taming3DGS [10] ranks candidate Gaussians by combining gradient, pixel coverage, per-view saliency, and core attributes including opacity, depth, and scale; Color-cued GS [20] considers the view-independent spherical harmonics coefficient gradient to better capture color cues; and GaussianPro [9] guides growth using depth and normal maps. In the pruning phase, LightGaussian [11], Mini-Splatting [12], RadSplat [5], and AtomGS [13] compute an importance score from each Gaussian's accumulated ray contribution, typically a mix of volume, opacity, transmittance, and hit count, and discard the lowest-ranked Gaussians. Gradient-aware variants prune by per-Gaussian gradients (Trimming-the-Fat) [14] or the second-order sensitivity score derived from the Hessian matrix (PUP-3DGS) [15]. The score can also be trained via a learnable mask, as in LP-3DGS [6], Compact3DGS [4], and HAC [16]. Moreover, multi-view consistency criteria discard Gaussians unseen by keyframes [21] or visible only in real but not virtual views [22]. For a comprehensive survey of 3DGS compression and compaction, we refer readers to [8,23].
Despite their success, most existing strategies evaluate each Gaussian independently, leaving open the question of whether the retained set is truly the best global surrogate. Our work addresses this gap from a probabilistic perspective via Gaussian mixture reduction.
this section cite: ['b9', 'b19', 'b8', 'b10', 'b11', 'b4', 'b12', 'b13', 'b14', 'b5', 'b3', 'b15', 'b20', 'b21', 'b7', 'b22']

Section: 3DGS from Probabilistic Distribution Point of View.
Kheradmand et al. [24] formulate 3DGS as a Markov chain Monte Carlo process and use stochastic gradient Langevin dynamics to migrate dropped Gaussians onto retained ones, partially recycling lost information. However, their update remains pairwise and lacks convergence guarantees under a principled divergence. Moreover, this scheme is coupled with the original 3DGS pipeline, limiting its generalizability to other variants.
this section cite: ['b23']

Section: Method: Gaussian Herding across Pens

this section cite: []

Section: Probabilistic Scene Representation
Let φ(x; µ, Σ) = |2πΣ| -foot_0 exp(-(x -µ) Σ -1 (x -µ)) be the PDF of a Gaussian distribution with mean µ and covariance Σ. A Gaussian mixture with n components is a distribution with density:
φ n (x) = n i=1 α i φ(x; µ i , Σ i ),
where α i are the mixture weights satisfying α i > 0 1 . In the context of 3DGS, let x ∈ R 3 be spatial coordinates and f (x) represent the implicit surface function (i.e., geometric shape and opacity that
3DGS training module T steps Appearance feature Geometric feature Appearance opt. T steps Geometric feature Appearance feature Gaussians Blockwise GMR Geometric feature KD-tree partition B A C E F G (0,0,0) A B C D E F G (-3,-1,3) (1,-1.5,-2) (-4,-2,-1) (2,1,-1) (-1,2,1) (3,5,2)
this section cite: []

Section: GHAP

this section cite: []

Section: Gaussians

this section cite: []

Section: Gaussians/block
Gaussians {α j , c j , µ j , Σ j } n j=1 {ᾱ j , cj , μj , Σj } m j=1 Figure 3: An illustration of the proposed GHAP approach. The process begins with full-resolution 3DGS training to obtain initial geometric and appearance features. These Gaussians are then spatially partitioned using a KD-tree and grouped into blocks-analogous to sheep pens. We then perform blockwise Gaussian Mixture Reduction (GMR) to approximate the geometric shape within each block using a much smaller number of Gaussians. This step is analogous to the popular kernel herding method [25]. Finally, a lightweight appearance refinement step further optimizes the appearance feature of the reduced set. This multi-stage pipeline progressively guides the Gaussians in each block-analogous to herding across pens-toward a compact and high-fidelity representation.
excludes color). The training process in 3DGS learns opacity parameter α i , location parameter µ i , and shape parameter Σ i such that
φ n (x) ≈ f (x), ∀x ∈ X ,
where X is the 3D scene volume. Therefore, the geometry of the 3D scene can be effectively represented by a Gaussian mixture. Then, each of these Gaussian primitive is associated with its own color c i . Both the geometry and appearance attributes are important for high quality rendering.
this section cite: ['b24']

Section: Compaction via Optimal Transport
Motivated by the observation that many 3DGS algorithms [4,5,6] produce a significant number of redundant Gaussians during training, we improve rendering efficiency through compactionreducing the number of Gaussian primitives to achieve lower memory usage and faster rendering while preserving visual fidelity. The process consists of two key phases:
1. Geometric Compaction via GMR: Leveraging our probabilistic interpretation, we formulate compaction as Gaussian Mixture Reduction (GMR) [26], approximating the original Gaussian mixture with redundant components by one with fewer components. This yields a compacted geometric representation: φm (x) = m j=1 ᾱj φ(x; μj , Σj ), where m n. This step modifies only the Gaussian positions (μ j ) and covariances ( Σj ), leaving appearance attributes unchanged.
this section cite: ['b3', 'b4', 'b5', 'b25']

Section: Appearance Optimization:
The reduced Gaussians are initialized with appearance attributes (colors, opacities) and fine-tuned for optimal rendering performance. This step optimizes appearance only, maintaining geometric consistency.
Our approach decouples geometry and appearance optimization while using standard 3DGS training to preserve quality. Our training pipeline is visualized in Figure 3 and we describe the details below.
this section cite: []

Section: Geometric Compaction via GMR
T c (φ n , φ m ) = inf    n i=1 m j=1 π ij c(φ(•; µ i , Σ i ), φ(•; µ j , Σ j )) : m j=1 π ij = α i , n i=1 π ij = α j    .
The CTD generalizes optimal transport [18] to mixtures, treating each component as a discrete distribution in the space of Gaussian distributions. The cost function measures the cost of moving one unit of Gaussian from one location to another, and π ij measures the corresponding amount of mass that is being moved. The total cost is proportional to the cost and the mass, and the divergence is the smallest transportation cost to move the original mixture to the target mixture. The reduced mixture becomes {ᾱ j , μj , Σj } = arg min
{α j ,µ j ,Σ j } T c (φ n , φ m ).(1)
With this formulation, the Gaussian mixture after compaction has optimal guarantee. The solution also guides the choice of m to balance compactness and fidelity.
As shown in Zhang et al. [17], (1) can be solved using the effective iterative algorithm in Algorithm 1.
Algorithm 1 GMR via k-means Clustering
1: Initialize {μ (0) j , Σ(0) j } m j=1 2: for t=1,. . . , do 3: Assignment Step: 4: for i = 1 to n do O(nm) 5: Assign component i to cluster C j that minimizes c(φ(•; µ i , Σ i ), φ(•; μ(t-1) j , Σ(t-1) j )) 6: end for 7: Update Step: 8: for j = 1 to m do O(nm) 9: Compute new cluster center: μ(t) j , Σ(t) j = arg min i∈Cj α i c(φ(•, µ i , Σ i ), φ(•, µ, Σ)) 10: end for 11: if no change in assignments then 12: for j = 1 to m do O(n) 13: Compute weight: ᾱj = i∈Cj α i 14: end for 15: break 16:
end if 17: end for The algorithm reduces to a k-means variant in Gaussian space: 1) The assignment step follows the same principle as traditional k-means, but replaces the Lfoot_1 distance between vectors with a cost function c(•, •) between Gaussian distributions. 2) The update step generalizes the cluster center computation: In traditional k-means, centers are updated as arithmetic averages (barycenters w.r.t. L 2 distance) of vectors in each cluster. In this algorithm, centers become barycenters of Gaussians in each cluster, minimized w.r.t. cost function c(•, •). Thus, standard k-means emerges as a special case when using L 2 distance on vectorized Gaussian parameters.
While the standard GMR algorithm provides optimal theoretical guarantees, its direct application to 3DGS compaction proves computationally prohibitive. Although the algorithm must converge in finite steps [17] 2 , the assignment step involves nm evaluations of the cost function per iteration. In typical 3D scenes, the number of Gaussians scales as n = Ω(10 5 )foot_2 , and even after 95% reduction, each iteration would still require at least 10 8 operations and memory storage. For the update step, the computational cost for the cluster center depends on the pre-specified cost function c(•, •). The KL divergence considered in Zhang et al. [17] suffers from (a) significant overhead of computing O(ρs 2 log n) covariance matrices inversions, and (b) numerical instability due to small eigenvalues of covariance matrices. To overcome this challenge, we introduce two key optimizations designed for 3DGS:
• Blockwise GMR via KD-Tree: To improve computational and memory efficiency during training, we partition the scene into spatially blocks and perform GMR within each block. As demonstrated in Remark 1, this blockwise approach yields significant computational savings. While both KDtrees [27] and Octrees [28] are effective for spatial partitioning in 3D space, we employ a KD-tree for two key advantages. First, it produces more balanced partitions across regions. Second, it avoids unnecessary subdivisions in sparse regions that would waste computational resources. Our KD-tree is constructed solely from Gaussian centers {µ i } n i=1 (justified by the observed small eigenvalues of covariance matrices). Each split uses the median coordinate value, creating 2 d blocks at depth d. We set d = log 2 (n/s) to ensure blocks contain at most s Gaussians with s n, then reduce each block to m = ρs components (ρ = retention ratio). Remark 1 (Computational Cost Comparison). Our blockwise approach reduces the per-iteration computational cost from O(ρn 2 ) to O(ρs 2 ) per block. 4 With 2 depth = O(log n) blocks in total, the overall complexity becomes O(ρs 2 log n). For typical values of n = 10 5 , s = 10 3 and ρ = 0.05, this reduces the cost from 10 8 to approximately 10 5 operations-a substantial improvement. The savings become even more pronounced for larger n. Furthermore, the reduction steps can be executed in parallel across blocks, offering additional computational speedup.
• Efficient Cost Function: We introduce a novel cost function that overcomes the limitations of the KL divergence used in [17] by being computationally efficient without sacrificing approximation quality. Our proposed divergence is:
c(φ(•; µ, Σ), φ(•; µ , Σ )) = µ -µ 2 2 + Σ -Σ 2 F ,(2)
which offers three significant advantages: First, it preserves distributional similarity, as Gaussian distributions are uniquely determined by their mean and covariance. Second, the assignment step requires only efficient vector and matrix norm computations. Third, the update step simplifies to calculating weighted averages, thereby avoiding the computationally expensive covariance matrix inversions required by the KL divergence:
μ(t) j = i∈Cj α i µ i i∈Cj α i , Σ(t) j = i∈Cj α i Σ i i∈Cj α i .
this section cite: ['b17', 'b16', 'b16', 'b26', 'b27', 'b16']

Section: Appearance Optimization
Algorithm 2 GHAP: 3DGS Compaction via Blockwise GMR 1: Input:
Trained 3DGS model for T steps to obtain {(α i , µ i , Σ i , c i )} n i=1 , retention ratio ρ 2: Output: Compacted {(ᾱ j , μj , Σj , cj )} m j=1 3: Stage 1: Geometric Compaction 4: 1. Build KD-tree from Gaussians {µ i } n i=1 with depth d = log 2 (n/s) O(nd log n) 5: 2. For each leaf block B k : O(nmT /2 d ) 6:
Run Algorithm 1 to reduce to ρs Gaussians 7: Stage 2: Appearance Optimization 8: 1. Initialize appearance for each φj : 9: cj ← c i * and ᾱj ← α i * where i * = arg min i∈[n] µ i -μj 2 10: 2. Fine-tune {ᾱ j , cj } using standard 3DGS rendering pipeline for T steps Following geometric compaction, we initialize the appearance attributes (opacity and color) of the compacted Gaussian primitives. For each primitive in the reduced mixture, we assign the appearance parameters from its closest counterpart in the original Gaussian mixture. Using these initial values, we then optimize the appearance attributes through backpropagation within the standard 3DGS training pipeline used in the first stage. We optimize the opacity instead of directly using the values from the GMR algorithm because its output weights do not necessarily satisfy the constraint that opacity must be between 0 and 1. Fine-tuning the opacity leads to better visualization performance.
this section cite: []

Section: Training Details with GHAP Algorithm
Integrating these components, we present our complete training pipeline in Algorithm 2, called Gaussian Herding Across Pens (GHAP). The process begins with standard 3DGS optimization for T steps, followed by blockwise GMR. We then freeze the geometric parameters (µ and Σ) while fine-tuning the appearance attributes (opacity α and color c) through an additional T -step 3DGS optimization. This procedure can be applied iteratively throughout training as needed.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets. For a comprehensive evaluation of GHAP algorithm, we use three real-world datasets: Tanks & Temples [29], Mip-NeRF 360 [30], and Deep Blending [31], which cover varying levels of detail, lighting conditions, and scene complexities. For each dataset, we adopt the same scenes as in [8].
• Tanks & Temples: We evaluate two unbounded outdoor scenes, "Truck" and "Train", both featuring centered viewpoints.
• Mip-NeRF 360: We test on a mix of indoor and outdoor scenes, including "Bicycle", "Bonsai", "Counter", "Flowers", "Garden", "Kitchen", "Room", "Stump", and "Treehill", all with centered viewpoints.
• Deep Blending: We include two indoor scenes, "Dr. Johnson" and "Playroom", where the viewpoint is directed outward.
Baselines. To evaluate the effectiveness of our proposed method, we compared it against four strong compaction techniques: LightGaussian [11], PUP-3DGS [15], Trimming the Fat [14], and MesonGS [32], as well as four end-to-end 3DGS variants: Mini-Splatting(-D) [12], AtomGS [13], 3DGS-MCMC [24], and LocoGS [33]. Notably, Mini-Splatting-D and AtomGS were employed as backbone models in our approach, while the other variants were used for direct comparative evaluation against the compaction methods. A consistent evaluation protocol was established to ensure fair and reliable conclusions. For post-training compaction methods applicable to pre-trained models-including GHAP, LightGaussian, PUP-3DGS, Trimming the Fat, and MesonGS-we initialized all from the same backbone model (trained using vanilla 3DGS, Mini-Splatting-D, or AtomGS for 15k iterations) and applied their respective compaction procedures directly, excluding any compressionspecific modules. All models subsequently underwent identical fine-tuning for 15k iterations to achieve the target retention ratio. For the other end-to-end variants (MiniSplatting, 3DGS-MCMC and LocoGS), we executed training for 30k iterations under their default configurations. Detailed experimental steps for each method can be found in the appendix.
Evaluation Metrics. We assess 3DGS compaction using standard metrics for rendering quality. We report: (1) PSNR, measuring pixel-level accuracy; (2) SSIM, evaluating perceptual similarity based on luminance, contrast, and structure; and (3) LPIPS, capturing perceptual distance via a learned model. Higher PSNR/SSIM and lower LPIPS indicate better quality. For each method, we also report the corresponding number of Gaussian primitives.
this section cite: ['b28', 'b29', 'b30', 'b7', 'b10', 'b14', 'b13', 'b31', 'b11', 'b12', 'b23', 'b32']

Section: Quantitative Results
We first show that our approach outperforms other SOTA compaction techniques. Second, we demonstrate the effectiveness of our GHAP compaction method as a plug-in module within 3DGS and its variants. We present the experimental results followed by our key findings.
this section cite: []

Section: Comparison with SOTA.
We compare our method against a comprehensive set of baselines, which can be categorized into two groups:
• End-to-End Compact 3DGS Variants: Mini-Splatting [12], 3DGS-MCMC [24], and LocoGS [33]. Note that for LocoGS, the final number of Gaussians is not a user-controllable parameter.
• Post-Training Compaction Baselines: LightGaussian [11], PUP-3DGS [15], Trimming the Fat [14], and MesonGS [32]. These methods all use a standard 3DGS backbone and apply pruning-based techniques for compaction.
Our approach is evaluated in two configurations: 3DGS+GHAP and MiniSplatting+GHAP. The former uses the same vanilla 3DGS backbone and 10% retention rate as the pruning baselines for a direct comparison of compaction strategies. The latter replaces the built-in pruning step in Mini-Splatting with our GHAP algorithm to demonstrate its effectiveness on a different backbone.
Quantitative results are summarized in Table 1, with methods grouped by their final primitive count. As expected, LocoGS achieves strong performance due to its larger primitive count. Among methods with comparable primitive counts (Group 2), our 3DGS+GHAP achieves superior performance in SSIM and PSNR, with a marginally lower LPIPS score. The advantage of our compaction strategy is further evident when using the Mini-Splatting backbone. Our MiniSplatting+GHAP outperforms other compaction-based approaches while often using fewer primitives. As shown in Fig. 4 (left), this performance lead is consistent across a wide range of retention ratios, not just at ρ = 0.1.
Runtime & Memory Usage Comparison. Crucially, the improved performance of our method does not come at a computational cost. As depicted in Fig. 4 (middle), our method's runtime is faster than all baselines except the exceptionally swift Trimming the Fat. While our method exhibits a slightly higher memory footprint (in Fig. 4 right) during compaction due to pairwise distance computation in each KD-tree block, the difference is not substantial (less than an order of magnitude).
this section cite: ['b11', 'b23', 'b32', 'b10', 'b14', 'b13', 'b31']

Section: As a Plug-In Compaction Method.
Our method can be used as a plug-in compaction method in various 3DGS training algorithms. This demonstrates the broad applicability of our proposed method.
To verify this, we apply our compaction method within various 3DGS pipelines. Specifically, we Retention Ratio (%)
21 22 23 PSNR (dB) 10 20 30 40 50 Retention Ratio (%) 0 5 10 15 20 25 Time (s) 10 20 30 40 50 Retention Ratio (%) 3.0 3.5 4.0 4.5 5.0 Memory (GB) GHAP (ours) LightGaussian PUP-3DGS Trimming the Fat MesonGS consider three representative variants as the backbone: 3DGS [3] , AtomGS [13], and Mini-Splatting-D [12]. Each of them employs a distinct densification strategy, and our method can be directly embedded into the pipeline without extensive engineering effort. For each backbone, we evaluate performance under two retention ratios (10% and 20%). Tab. 2 summarizes our experimental results. The quantitative results in Tab. 2 demonstrate that our method effectively preserves the backbone models' visual quality, even at an extreme retention rate of 10%. Notably, on some scenes (highlighted in bold), the compacted model's performance surpasses that of the uncompacted backbone.
Beyond quantitative metrics, we provide a qualitative analysis by visualizing multiple scenes before and after compaction in Fig. 5. As evidenced in the figure, our method successfully preserves rendering quality across most scenes while using only 10% of the Gaussian primitives. Interestingly, in certain cases, our compaction not only preserves but surpasses the original quality. A representative example is the "Kitchen" scene (Mini-Splatting-D backbone). We conjecture this improvement occurs because Mini-Splatting-D lacks a pruning mechanism, often generating an over saturated Visual results for various scenes under different 3DGS backbones, compacted to 10% of their primitives using our GHAP method. Our approach preserves rendering quality with negligible loss. In some cases (e.g., "Kitchen"), compaction even improves quality by regularizing an over saturated Gaussian distribution.
set of Gaussians that introduce visual artifacts (e.g., the unnatural shadows in the lower-left region).
Our method acts as a global regularizer, mitigating this issue by reducing unnecessary density while improving the overall expressiveness and preserving the underlying 3D structure. Naturally, our approach is inherently limited by the quality of its input. If the original model suffers from significant artifacts due to a lack of primitives in certain regions, our compaction cannot resolve these fundamental issues. This limitation is demonstrated in the "Bicycle" scene, where artifacts present in the Atom-GS backbone persist after compaction.
this section cite: ['b2', 'b12', 'b11']

Section: Ablation Studies
Influence of KD Tree Depth. To validate the effectiveness of the KD-tree partitioning strategy, we provide an ablation study on it. Due to the large scale of the three primary datasets, low KD-tree depths result in an excessive number of points per block, making it infeasible to run the GMR algorithm. Therefore, we conduct ablation experiments on the smaller mic scene from NeRF-Synthetic. Results in Fig. 6 show that increasing KD-tree depth reduces memory usage and runtime, while PSNR first improves and then declines. This indicates that moderately finer partitions allow GMR to compact regions more effectively, whereas overly fine splits may fragment primitives and degrade quality.
Loss Function Design. As shown in Tables 1 and 2, our method exhibits a slight underperformance on the LPIPS metric. To address this, we investigate whether incorporating an LPIPS loss term can yield improvements.
Our baseline loss function follows the vanilla 3DGS formulation, using an L1-to-SSIM ratio of 8:2. We experiment with two new weighting schemes that include LPIPS: (1) L1:SSIM:LPIPS = 8:1:1, and (2) L1:SSIM:LPIPS = 6:2:2. The results of this ablation study on the Tanks&Temples dataset are shown in Fig. 7 (with additional results in the Appendix). Our analysis reveals a trade-off between perceptual and distortion metrics: increasing the weight of the LPIPS loss improves LPIPS scores but leads to a slight deduction in PSNR and SSIM.
this section cite: []

Section: GMR versus Random Subsampling.
We conduct an ablation study on the Tanks&Temples dataset to evaluate the contribution of each stage in our pipeline: geometric compaction and appearance optimization. A random subsampling baseline is included to assess the effectiveness of our design choices. The results, presented in Table 3, compare two compaction schemes followed by the same finetuning procedure. Our findings demonstrate that: first, our compaction procedure is significantly more effective than random subsampling and other pruning baselines (as shown in Tab. 1). Second, the subsequent appearance optimization stage provides substantial quantitative improvements for both compaction approaches. The significant performance gain over the baseline validates the necessity and effectiveness of both stages in our proposed pipeline.
this section cite: []

Section: Conclusion and Discussion
We propose an optimal transport-based Gaussian mixture reduction framework for 3D Gaussian Splatting, achieving compact yet faithful representations. By minimizing composite transport divergence with appearance fine-tuning, our method preserves high visual fidelity while retaining only 10% of Gaussians, outperforming prior compaction techniques. The framework scales efficiently via block-wise KD-tree partitioning and integrates seamlessly with diverse 3DGS pipelines.
Future directions include enhancing robustness across challenging scene types, incorporating perceptual objectives, developing multi-scale and overlap-aware partitioning, adopting auto-tuned schedules, and extending to dynamic 3DGS for real-time temporal rendering.
this section cite: []

Section: References
Ref_id:b0 Title: Weidong Yang, and Ying He. 3D Gaussian splatting as new era: A survey Year: (2024)
Ref_id:b1 Title: A survey on 3D Gaussian splatting Year: (2024)
Ref_id:b2 Title: 3D Gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b3 Title: Compact 3D Gaussian representation for radiance field Year: (2024)
Ref_id:b4 Title: RadSplat: Radiance field-informed Gaussian splatting for robust real-time rendering with 900+ FPS Year: (2024)
Ref_id:b5 Title: LP-3DGS: Learning to prune 3D Gaussian splatting Year: (2024)
Ref_id:b6 Title: Towards consistent real-time rendering with LOD-structured 3D Gaussians Year: (2024)
Ref_id:b7 Title: 3DGS.zip: A survey on 3D Gaussian splatting compression methods Year: (2024)
Ref_id:b8 Title: GaussianPro: 3D Gaussian splatting with progressive propagation Year: (2024)
Ref_id:b9 Title: Taming 3DGS: High-quality radiance fields with limited resources Year: (2024)
Ref_id:b10 Title: Light-Gaussian: Unbounded 3D Gaussian compression with 15x reduction and 200+ FPS Year: (2024)
Ref_id:b11 Title: Mini-splatting: Representing scenes with a constrained number of Gaussians Year: (2024)
Ref_id:b12 Title: AtomGS: Atomizing Gaussian splatting for high-fidelity radiance field Year: (2024)
Ref_id:b13 Title: Trimming the fat: Efficient compression of 3D Gaussian splats through pruning Year: (2024)
Ref_id:b14 Title: PUP 3D-GS: Principled uncertainty pruning for 3D Gaussian splatting Year: (2024)
Ref_id:b15 Title: HAC: Hash-grid assisted context for 3D Gaussian splatting compression Year: (2024)
Ref_id:b16 Title: Gaussian mixture reduction with composite transportation divergence Year: (2023)
Ref_id:b17 Title: Computational optimal transport: With applications to data science Year: (2019)
Ref_id:b18 Title: Optimal Transport: Old and New Year: (2009)
Ref_id:b19 Title: Color-cued efficient densification method for 3D Gaussian splatting Year: (2024)
Ref_id:b20 Title: Gaussian splatting SLAM Year: (2024)
Ref_id:b21 Title: NEDS-SLAM: A neural explicit dense semantic SLAM framework using 3D Gaussian splatting Year: (2024)
Ref_id:b22 Title: 3D Gaussian splatting: Survey, technologies, challenges, and opportunities Year: (2025)
Ref_id:b23 Title: 3D Gaussian splatting as Markov Chain Monte Carlo Year: (2024)
Ref_id:b24 Title: Super-samples from kernel herding Year: (2012)
Ref_id:b25 Title: A look at Gaussian mixture reduction algorithms Year: (2011)
Ref_id:b26 Title: Multidimensional binary search trees used for associative searching Year: (1975)
Ref_id:b27 Title: Geometric modeling using octree encoding. Computer graphics and image processing Year: (1982)
Ref_id:b28 Title: Tanks and temples: Benchmarking large-scale scene reconstruction Year: (2017)
Ref_id:b29 Title: Deep blending for free-viewpoint image-based rendering Year: (2018)
Ref_id:b30 Title: Mipnerf 360: Unbounded anti-aliased neural radiance fields Year: (2022)
Ref_id:b31 Title: Mesongs: Post-training compression of 3d gaussians via efficient attribute transformation Year: (2024)
Ref_id:b32 Title: Locality-aware gaussian compression for fast and high-quality rendering Year: (2025)
Ref_id:b33 Title: Optimal transport for Gaussian mixture models Year: (2018)
Ref_id:b34 Title: A Wasserstein-type distance in the space of Gaussian mixture models Year: (2020)
Ref_id:b35 Title: GauHuman: Articulated Gaussian splatting from monocular human videos Year: (2024)
Ref_id:b36 Title: GPS-Gaussian: Generalizable pixel-wise 3D Gaussian splatting for real-time human novel view synthesis Year: (2024)
Ref_id:b37 Title: DreamGaussian: Generative Gaussian splatting for efficient 3D content creation Year: (2024)
Ref_id:b38 Title: DrivingGaussian: Composite Gaussian splatting for surrounding dynamic autonomous driving scenes Year: (2024)
Ref_id:b39 Title: 4D Gaussian splatting for real-time dynamic scene rendering Year: (2024)
Ref_id:b40 Title: Highquality surface reconstruction using Gaussian surfels Year: (2024)
Ref_id:b41 Title: GaussianEditor: Swift and controllable 3D editing with Gaussian splatting Year: (2024)
