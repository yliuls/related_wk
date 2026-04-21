Title: TRACE: YOUR DIFFUSION MODEL IS SECRETLY AN INSTANCE EDGE DETECTOR
Abstract: High-quality instance and panoptic segmentation has traditionally relied on dense instance-level annotations such as masks, boxes, or points, which are costly, inconsistent, and difficult to scale. Unsupervised and weakly-supervised approaches reduce this burden but remain constrained by semantic backbone constraints and human bias, often producing merged or fragmented outputs. We present TRACE (TRAnsforming diffusion Cues to instance Edges), showing that text-to-image diffusion models secretly function as instance edge annotators. TRACE identifies the Instance Emergence Point (IEP) where object boundaries first appear in self-attention maps, extracts boundaries through Attention Boundary Divergence (ABDiv), and distills them into a lightweight one-step edge decoder. This design removes the need for per-image diffusion inversion, achieving 81× faster inference while producing sharper and more connected boundaries. On the COCO benchmark, TRACE improves unsupervised instance segmentation by +5.1 AP, and in tag-supervised panoptic segmentation it outperforms point-supervised baselines by +1.7 PQ without using any instance-level labels. These results reveal that diffusion models encode hidden instance boundary priors, and that decoding these signals offers a practical and scalable alternative to costly manual annotation.

Section: INTRODUCTION
Panoptic segmentation unifies semantic and instance segmentation and underpins real-world applications, such as autonomous driving (Elharrouss et al., 2021;Zendel et al., 2022). However, achieving reliable instance-level delineation has long relied on dense pixel-wise annotations such as masks, boxes, or points, which are prohibitively expensive, inconsistent across annotators, and fundamentally hard to scale. These limitations motivate the search for annotation-free alternatives that can retain the fine granularity of supervised methods without the cost of labeling.
Recent unsupervised and weakly-supervised approaches (Sick et al., 2025;Li et al., 2024) attempt to bypass dense labeling. Unsupervised instance segmentation (UIS) eliminates explicit annotation by clustering semantic features from pretrained vision transformers (Caron et al., 2021;Oquab et al., 2023), but these models are inherently optimized for semantic similarity across images rather than instance separation within an image. As shown in Fig. 2, existing UIS methods (Wang et al., 2023a;Li & Shin, 2024) often merge adjacent objects of the same class or fragment single instances, and rely on heuristic assumptions such as a predefined number of objects. Parallel advances in weaklysupervised semantic segmentation have demonstrated near-supervised performance, achieving up to 99% fully supervised accuracy on VOC 2012 (Everingham et al., 2010) using only image-level tags (Jo et al., 2023;2024a). However, extending this success to the panoptic setup still requires point or box annotations to disambiguate instances. These additional annotations remain costly and error-prone, particularly when objects overlap or when annotators are inconsistent.
Our key insight is that self-attention maps of text-to-image diffusion models (Podell et al., 2023;Esser et al., 2024) encode instance-aware cues early in denoising. As shown in Fig. 1(a), crossattention does not reliably separate adjacent objects even with an explicit prompt, whereas selfattention at specific timesteps reveals instance-level structure. During the denoising process, the model transitions from noise to instance-level structure and then to semantic content. This raises To answer this, we introduce TRACE (TRAnsforming diffusion Cues to instance Edges), a framework that decodes instance boundaries directly from pretrained text-to-image diffusion models. As illustrated in Fig. 1(b), TRACE first identifies the Instance Emergence Point (IEP) by measuring temporal divergence to select the timestep where the instance structure is most pronounced. It then applies Attention Boundary Divergence (ABDiv) to score criss-cross differences in self-attention and generate initial edge maps. At this stage, pixels within the same object exhibit nearly identical self-attention distributions, whereas pixels across different objects diverge sharply; this divergence peaks on true instance boundaries and provides a direct signal for instance edge extraction. To reduce the computational cost of the per-image forward process at test time, these edges are distilled into a one-step predictor that integrates the diffusion backbone with an edge decoder. The resulting edges are used as boundary priors in downstream segmentation methods (Wang et al., 2023a;Jo et al., 2024a), guiding the propagation to cleanly separate adjacent objects by splitting merged regions along instance boundaries (Fig. 2).
Our key contributions are summarized as follows:
• We observe that self-attention in diffusion models briefly yet reliably reveals instance-level structure during denoising, unlike common vision transformers (see Tab. 5).
• The proposed TRACE unifies two key ideas for annotation-free instance boundary discovery: the Instance Emergence Point and Attention Boundary Divergence.
• TRACE enables annotation-free instance and panoptic segmentation: 1) Improves unsupervised instance segmentation baselines by +4.4 AP with only 6% runtime overhead; 2) With tag supervision, surpasses point-supervised panoptic models, up to +7.1 PQ on VOC 2012; 3) As seeds for SAM, outperforms open-vocabulary detectors by up to +16.5 PQ (stuff).
this section cite: ['b18', 'b103', 'b77', 'b9', 'b65', 'b45', 'b20', 'b33', 'b67', 'b19']

Section: Ground Truth TRACE (Ours) Baseline (MaskCut)
Baseline w/ TRACE
Figure 2: Effect of TRACE. (Left) Our instance edges decoded from diffusion self-attention for reconnection of fragmented masks and separation of adjacent objects, with white dotted circles marking corrected boundaries. (Right) Consistent AP mk gains over baselines (Wang et al., 2023a;Li & Shin, 2024) without instance-level annotations.
this section cite: ['b45']

Section: RELATED WORK
Unsupervised Instance Segmentation. Instance segmentation aims to delineate individual objects and typically requires pixel-level annotations. Early methods (Wang et al., 2022;Ishtiak et al., 2023) learn pseudo masks from external features (Wang et al., 2021) but require training from scratch and show limited accuracy. Recent approaches, such as MaskCut (Wang et al., 2023a), U2Seg (Niu et al., 2024), and UnSAM (Wang et al., 2024b), cluster features from pretrained vision transformers like DINO (Caron et al., 2021). These models strong at semantic grouping across images but are not explicitly designed for separating instances within an image. Therefore, these clustering methods often rely on heuristics such as a maximum number of instances or confidence thresholds and tend to merge adjacent objects of the same category.
To improve instance separation, CutS3D (Sick et al., 2025) and CUPS (Hahn et al., 2025) incorporate monocular depth estimators (Ke et al., 2024;Yang et al., 2024) to split objects at different ranges. However, depth-based approaches struggle when neighboring objects lie at similar depth and they degrade on distant or small objects where estimated depth becomes blurry. By contrast, our diffusion-based strategy (TRACE) extracts instance boundaries from self-attention of pretrained diffusion models (Peebles & Xie, 2023;Bao et al., 2023;Podell et al., 2023;Esser et al., 2024). This boundary-centric cue does not assume the number of instances, is robust to object scale and distance, and refines existing unsupervised pipelines (Wang et al., 2023a;Li & Shin, 2024) without any supervision or retraining, achieving up to 29.1% higher performance on COCO compared to depth-based methods (see Tab. 1).
Weakly-supervised Semantic and Panoptic Segmentation. Panoptic segmentation jointly requires semantic masks for "stuff" regions (e.g., grass) and instance masks for "thing" objects (e.g., person), which makes it one of the most annotation-intensive tasks in segmentation. To reduce this labeling cost, weakly-supervised panoptic segmentation have been explored. Image-level class tags (Shen et al., 2021) alone cannot separate instances. Bounding boxes are ill-suited for panoptic supervision because they provide only coarse rectangles for "thing" objects and cannot define the non-overlapping pixel-wise regions required for "stuff" regions. Consequently, point annotations (Fan et al., 2022;Li et al., 2023b;2024) have become the dominant form of weak supervision. However, points vary across annotators and are often placed near object centers, which produces partial or missed instances and leaves adjacent objects merged (see Fig. 3).
Image & GT
1. Partial Point 2. Missed Point Point Annotations TRACE (Ours) Human Bias in Weak Labels Unbiased Instance Edge Meanwhile, in weakly-supervised semantic segmentation, recent tagsupervised approaches (Jo et al., 2024a;Yang et al., 2025a;b) show that image tags alone can reach about 95% of fully supervised accuracy on the PASCAL VOC 2012 benchmark, indicating that tags are sufficient for semantics but not for instance separation. Therefore, we revisit tag supervision and inject instance structure using diffusion priors: TRACE attaches to a tag-supervised semantic model (Jo et al., 2023;2024a) and converts its pseudo semantic masks into pseudo panoptic masks by supplying instance-aware boundaries from diffusion self-attention. This model-agnostic design uses only image-level tags, cleanly separates adjacent objects, and first surpasses pointsupervised panoptic baselines (Li et al., 2024) on VOC and COCO benchmarks (see Tab. 2).
this section cite: ['b82', 'b31', 'b81', 'b64', 'b9', 'b77', 'b26', 'b37', 'b97', 'b66', 'b5', 'b67', 'b19', 'b45', 'b75', 'b21', 'b28', 'b33']

Section: Diffusion-Driven and Open-Vocabulary Segmentation.
Recent approaches, including DiffCut (Couairon et al., 2024), DiffSeg (Tian et al., 2024), and ConceptAttention (Helbling et al., 2025), repurpose the self-and cross-attention maps of pretrained text-to-image diffusion models (Peebles & Xie, 2023;Bao et al., 2023;Esser et al., 2024;Podell et al., 2023) for semantic segmentation by analyzing attention at a fixed timestep without inversion. In parallel, open-vocabulary segmentation builds on contrastive pretraining (Radford et al., 2021) to map free-form text to visual concepts, enabling text-conditioned masks. Despite progress, such models (Liu et al., 2024b;You et al., 2023;Zhao et al., 2025) typically underperform closed-vocabulary segmentation and struggle to produce reliable seeds under multi-tag inputs because they are hard to obtain from captions with limited tag coverage or in scenes containing multiple nearby objects. Compared to them, TRACE yields higher panoptic quality than open-vocabulary detection when used as TRACE seeds for SAM in Sec. 6.
this section cite: ['b16', 'b80', 'b29', 'b66', 'b5', 'b19', 'b67', 'b68', 'b102', 'b105']

Section: Attention Boundary Divergence

this section cite: []

Section: METHOD
In this section, we outline an overview of TRACE in Fig. 4 for a comprehensive understanding of our framework. Section 3.2 introduces the Instance Emergence Point (IEP), which selects the denoising step where instance structure is most pronounced. Section 3.3 describes Attention Boundary Divergence (ABDiv), which converts criss-cross self-attention differences into a pseudo edge map. Section 3.4 details a one-step distillation that trains an edge decoder with the diffusion backbone to produce connected edges and enable real-time inference. Section 3.5 shows how our edges integrate with downstream segmentation models through Background-Guided Propagation (BGP).
3.1 BACKGROUND Models. Diffusion (Rombach et al., 2022;Podell et al., 2023) and flow matching models (Esser et al., 2024;Lipman et al., 2022) generate images by learning a reverse process from noise to data. Although their objectives differ, both families use similar transformer backbones with self-and cross-attention. We refer to either as a diffusion model since our method is compatible with both.
In typical text-to-image implementations, a VAE encoder E maps an input image I into a latent X 0 = E(I), a denoising network ϵ θ iteratively predicts noise (or velocity) on latents X t conditioned on an optional text embedding c, and a VAE decoder D turns a generated image Î = D( X0 ). TRACE reads only self-attention maps from a diffusion model and does not require text prompts.
this section cite: ['b69', 'b67', 'b19', 'b55']

Section: Self-Attention Collection and Aggregation.
For a latent X t ∈ R HW ×d at step t, we form queries Q t = X t W Q and keys K t = X t W K . The self-attention for block k (averaged over heads, rows sum to 1) is SA k t (X t ) = softmax(Q t K ⊤ t / √ d) ∈ [0, 1] HW ×HW ; we ignore cross-attention and do not provide prompts. To fuse maps from blocks k = 1, . . . , N that may operate at different spatial sizes w k (e.g., multi-scale stages in U-Net; for single-scale DiT-style backbones U is identity), we upsample to w max and average: SA(X t ) = 1 N N k=1 U k→max SA k t (X t ) . We implement this with PyTorch forward hooks H on attention blocks; on each forward of the denoising network ϵ θ with inputs (X 0 , t), the hook collects {SA k t }, performs the aggregation, and returns the map, H(ϵ θ , X 0 , t) = SA(X t ), without altering model outputs.
this section cite: []

Section: IDENTIFYING THE INSTANCE-AWARE DENOISING STEP
We next ask: at which point of the denoising trajectory does self-attention truly become instanceaware? Early in denoising, self-attention maps are almost indistinguishable from noise. As steps proceed, we observe a sharp rise in the Kullback-Leibler (KL) divergence between consecutive maps. This peak coincides with the emergence of clear object boundaries, after which divergence gradually decreases as object shape stabilizes while semantics continue to refine. During inversion, the trajectory unfolds in reverse order: semantic → instance → noise.
Motivated by our observation, we propose the Instance Emergence Point (IEP) as the timestep t ⋆ where this divergence is maximized:
t ⋆ = argmax t∈{τ1,...,τ N } D KL (SA(X tprev ) ∥ SA(X t ))(1)
where τ 0 < • • • < τ N are discrete timesteps (with t = τ k , t prev = τ k-1 ). The self-attention map at this point, SA(X t ⋆ ), is denoted as SA inst and serves as our instance-aware representation. Specifically, KL divergence is a natural choice (Tian et al., 2024) because each row of SA(X t ) is a probability distribution. Unlike mean-squared or absolute differences, KL's log-scale sensitivity amplifies subtle but meaningful variations in high-dimensional self-attention that directly align with boundary emergence (see Tab. 4). In practice, we adopt a fixed inversion stride of 100 steps for efficiency and accuracy. A detailed analysis of step size and the distribution of the optimal timestep t ⋆ across diffusion backbones is provided in Fig. 7. Neighboring pixels within the same instance exhibit similar self-attention maps, whereas those across instance boundaries differ, as shown in Fig. 5. We convert this contrast into edges with Attention Boundary Divergence (ABDiv), a simple non-parametric score that transforms instanceaware self-attention maps into boundary maps without clustering or annotations. We apply ABDiv on the instance-aware map SA inst = SA(X t ⋆ ) identified in Sec. 3.2. For a pixel (i, j), ABDiv aggregates the divergence between opposite 4-neighbors:
this section cite: ['b80']

Section: EXTRACTING INSTANCE EDGES FROM SELF-ATTENTION
ABDiv(SA) i,j := D KL SA i+1,j ∥ SA i-1,j + D KL SA i,j+1 ∥ SA i,j-1 .(2)
By default, ABDiv is computed with a 4-neighborhood using opposite pairs (left/right and top/bottom) as defined in Eq. 2. An 8-neighborhood extension that adds diagonal pairs achieves the same accuracy while increasing computation by approximately 2×, so we adopt the 4-neighborhood in all experiments. Published as a conference paper at ICLR 2026 3.4 ONE-STEP SELF-DISTILLATION WITH EDGE DECODER Starting from the instance-aware self-attention map SA inst = SA(X t ⋆ ) identified in Sec. 3.2, we obtain an initial edge map E by applying ABDiv as defined in Eq. 2. Inspired by pseudo-labeling strategies for weak supervision (Ahn et al., 2019;Jo et al., 2024a), we adopt a reliability-based thresholding to mitigate label noise from ambiguous self-attention signals. Specifically, we define a ternary map using the mean µ and standard deviation σ of ABDiv scores: pixels > µ + σ are edges 1, < µ -σ are interior 0, and the intermediate range is marked as uncertain -1. These uncertain pixels are explicitly excluded from the loss computation, which effectively suppresses false positives while maintaining high recall (see ablation in Appendix E.1 and Tab. 10).
To replace a per-image IEP+ABDiv computation at inference with a single pass, we fine-tune the diffusion backbone using Low-Rank Adaptation (Hu et al., 2022) and jointly train an edge decoder G ϕ . Beyond efficiency, the decoder also learns to complete fragmented edges, following common practice in boundary detection (Xie & Tu, 2015;Xiao et al., 2018;Su et al., 2023). Let I, Î ∈ R H×W ×3 be the original and reconstructed images, E, Ê = G ϕ (H(ϵ θ , I, t = 0)) ∈ [-1, 1] H×W be the pseudo edge map of ABDiv and the edge predicted by G ϕ . Our training objective is L(θ, ϕ) = L rec (θ) + L edge (θ, ϕ) = ∥I -Î∥ 2 + DiceLoss(E, Ê). After training, a single forward pass at t=0 produces a connected and precise edge map and removes the need for IEP and ABDiv during inference, as shown in Fig. 4. Full algorithmic details are provided in Appendix C. We now use our instance edges to regularize and complete segmentation masks. Given segmentation masks, connected component labeling treats the TRACE edges as separators and assigns unique labels to connected regions that are not cut by edges. Inspired by (Ahn et al., 2019), we design the Background-Guided Propagation (BGP), as shown in Fig. 6, propagate each fragmented mask inside its instance boundaries to close gaps and produce smooth regions. We then iteratively merge overlapping masks whose intersection over union exceeds τ BGP = 0.5 until convergence. This produces complete instance masks with our edges. We empirically find that performance remains stable over typical choices of τ BGP on VOC, so we simply use 0.5 in all experiments.
this section cite: ['b1', 'b30', 'b94', 'b91', 'b79', 'b1']

Section: SEMANTIC-TO-INSTANCE MASK REFINEMENT WITH INSTANCE EDGES

this section cite: []

Section: EXPERIMENTS
Implementation Details. For fair comparison, we follow standard protocols (Wang et al., 2023a;Li et al., 2023b) and run all experiments on a single NVIDIA A100 GPU. Stable Diffusion 3.5 Large (SD3.5-L) (Esser et al., 2024), our default backbone, performs best overall among five diffusion backbones evaluated (Tab. 5), with VOC and COCO as the main benchmarks and five additional datasets reported in Appendix E. Training details and evaluation metrics appear in Appendix C.
this section cite: ['b19']

Section: Unsupervised Instance Segmentation.
In Tab. 1, TRACE refines masks produced by existing UIS methods Wang et al. (2023a); Li & Shin (2024); Wang et al. (2024b) and consistently improves performance AP mk with gains ranging from +3.6 to +5.3 points. For clarity, we group results into training-free and fine-tuned methods, where the latter relies on a Mask R- CNN He et al. (2017) trained on pseudo instance masks. In particular, compared to the depth prior (Sick et al., 2025), TRACE attains higher AP mk on COCO (+2.2/+2.1 on 2014/2017), highlighting the advantage of diffusion-driven instance edges. Qualitative results appear in Fig. 17.
Weakly-supervised Panoptic Segmentation. We refine semantic masks from tag-supervised methods (Jo et al., 2023;2024a) with instance-aware edges to form pseudo panoptic masks and then train a standard Mask2Former (Cheng et al., 2022) following the common evaluation protocol. In particular, DHR (Jo et al., 2024a) with TRACE surpasses point-supervised counterparts (Li et al., 2023b;2024) by using only image-level tags (see Tab. 2 and Fig. 3), indicating that our edges provide the instance geometry that tag supervision lacks. Qualitative examples are provided in Fig. 16.  (Li et al., 2021) ResNet-50 M 67.9 66.6 92.9 --43.6 49.3 35.0 80.6 52.6 Mask2Former* (Cheng et al., 2022) ResNet-50 M 73.6 72.6 93.5 90.6 80.5 51.9 57.7 43.0 --PSPS (Fan et al., 2022) ResNet-50 P 49.8 47.8 89.5 --29.3 29.3 29.4 --Panoptic FCN (Li et al., 2021) ResNet-50 P 48.0 46.2 85.2 --31.2 35.7 24.3 --Point2Mask* (Li et al., 2023b) ResNet-50 P 53.8 51.9 90.5 --32.4 32.6 32.2 75.1 41.5 EPLD (Li et al., 2024) ResNet-50 P 56.6 54.9 89.6 --34.2 33.6 35.3 --Point2Mask* (Li et al., 2023b) Swin-L P 61.0 59.4 93.0 --37.0 37.0 36.9 75.8 47.2 EPLD (Li et al., 2024) Swin-L P 68.5 67.3 93.4 --41.0 39.9 42.7 --JTSM (Shen et al., 2021) ResNet-18-WS I 39.0 37.1 77.7 --5.3 8.4 0.7 30.8 7.8 MARS* (Jo et al., 2023) ResNet-50 I 41.4 39.8 85.3 83.0 57.8 11.7 13.3 10.2 58.3 11.8 + TRACE (Ours)
ResNet-50 I 50.4 48.5 88.9 86.6 60.1 29.5 31.1 28.9 62.5 39.3 DHR* (Jo et al., 2024a) ResNet-50 I 45.0 43.3 88.3 83.3 59.8 18.3 17.5 18.1 69.3 14.8 + TRACE (Ours)
ResNet-50 I 56.9 55.2 91.0 88.4 63.4 32.8 32.7 32.9 75.5 42.5 + TRACE (Ours) Swin-L I 69.8 68.4 96.2 94.5 71.2 43.1 42.5 43.5 83.8 55.3 M: Full mask supervision (upper bound), P: One point per instance, I: Image-level tags only (no instance annotations) * Reproduced results using the publicly accessible code. The rest are the values reported in the publication.
this section cite: ['b45', 'b77', 'b33', 'b14', 'b51', 'b14', 'b21', 'b51', 'b75', 'b33']

Section: ABLATION STUDY
Table 3: Effect of key components on COCO 2014 with the UIS baseline (Li & Shin, 2024).
IEP (Sec. 3.2) ABDiv (Sec. 3.3) Distill (Sec. 3.4) AP mk (a) × × × 3.1 (b) × ✓ × 3.2 (c) ✓ ✓ × 4.8 (d) ✓ ✓ ✓ 8.2
Component Ablation. Table 3 shows how each stage steers TRACE from purely semantic cues toward instance delineation. Because the Instance Emergence Point (IEP) marks the timestep where semantic attention first becomes instance-aware, it cannot be evaluated on its own: without the boundary scoring of ABDiv there is no measurable edge signal. Accordingly, case (b) applies Attention Boundary Divergence (ABDiv) at a purely semantic timestep, following prior diffusion approaches (Tian et al., 2024;Couairon et al., 2024), and yields almost no gain, confirming that semantic self-attention alone cannot reveal instance edges. Introducing IEP in case (c) pinpoints the denoising step where diffusion self-attention transitions from semantic grouping to instance structure, enabling ABDiv to capture instance boundaries. Finally, case (d) adds one-step self-distillation to compress these transient cues into a single-pass predictor, eliminating per-image IEP and ABDiv at inference and cutting latency from 3682 ms to just 45 ms per image (about 81× faster) while preserving and even strengthening edge connectivity.
this section cite: ['b45', 'b80', 'b16']

Section: Self-Distillation with Reconstruction.
During one-step self-distillation (Sec. 3.4, we jointly optimize edge prediction and image reconstruction. While the edge loss L edge compels the student to reproduce the teacher's instance boundaries, edges alone can overfit to noisy or incomplete supervision. Adding a reconstruction loss L rec anchors the decoder to global image structure, stabilizing training and suppressing artifacts along low-contrast boundaries. This auxiliary objective yields smoother and more coherent edges and provides measurable gains in both accuracy and perceptual quality (AP mk from 8.9 to 9.4; SSIM from 0.71 to 0.83) without adding inference cost. LLaVA-13B EVA02-E DINOv2-G SD1.5 with IEP Image & GT SDXL with IEP DINOv1 (ViT-B) CLIP (ViT-B) SD3.5-L SD3.5-L with IEP without IEP with IEP Figure 8: Visualization of self-attention maps with PCA.
this section cite: []

Section: Instance Emergence Analysis.
distributions across datasets and classes, as well as the empirical superiority of our KL criterion over alternative metrics (e.g., Entropy and Wasserstein).
this section cite: []

Section: Superiority of Generative Diffusion Priors.
To verify whether instance-aware cues are specific to diffusion models, we evaluate TRACE across 10 different backbones, including 5 diffusion and 5 nondiffusion foundation models (Oquab et al., 2023;Fang et al., 2024;Liu et al., 2024a;Siméoni et al., 2025;Podell et al., 2023) (see Tab. 5). Note that for non-diffusion backbones lacking temporal trajectories, we apply ABDiv (Sec. 3.3) directly to their self-attention maps. Remarkably, even the smallest diffusion model, PixArt-α (0.6B) (Chen et al., 2024), achieves 7.1 AP mk , significantly outperforming the massive 72B-parameter Qwen2.5-VL (Bai et al., 2025) (4.1 AP mk ). This confirms that TRACE leverages the unique generative nature of diffusion models, where instance boundaries emerge during denoising (IEP; Sec. 3.2), rather than typical semantic features found in discriminative or multimodal models. Figure 8 visualizes this distinction: diffusion self-attention tightens along object boundaries at IEP, whereas non-diffusion attention collapses into coarse semantic blobs, failing to separate Baseline (ProMerge) HED PiDiNet Canny TRACE (Ours) Baseline w/ TRACE w/ Canny w/ HED w/ PiDiNet DiffusionEdge w/ DiffusionEdge ✓ Instance-aware Edge Gradient Edge Texture-contrast Edge Texture-contrast Edge Texture-contrast Edge Figure 10: Instance-aware edge comparison with existing edge alternatives.
adjacent instances. Furthermore, within the diffusion family, performance correlates positively with model capacity (SD1.5 → FLUX.1), demonstrating that TRACE effectively scales with stronger generative priors while remaining model-agnostic in its applicability.
this section cite: ['b65', 'b23', 'b78', 'b67', 'b11', 'b4']

Section: Why Annotation-Free Instance Edges?
Recent efforts toward instance and panoptic segmentation often combine open-vocabulary detectors (Liu et al., 2024b;You et al., 2023;Zhao et al., 2025) with SAM (Kirillov et al., 2023), where the detector supplies instance-level boxes and SAM refines them into masks. Despite this progress, such pipelines still depend on box annotations and struggle when scenes involve many adjacent objects or ambiguous text prompts. TRACE provides a different alternative: it extracts instance edges directly from diffusion self-attention, requiring no instance supervision while offering clean separation of objects (Fig. 9). These edges are complementary to SAM, since SAM excels at refining seeds into precise instance masks while TRACE provides those seeds. This combination exceeds all supervised open-vocabulary baselines. In addition, when integrated with tag-supervised semantic models, TRACE supplies the missing instance geometry and converts their outputs into complete panoptic masks, outperforming point-supervised methods on VOC and COCO.
this section cite: ['b102', 'b105', 'b41']

Section: Limitations of Conventional Edge Detectors.
Figure 10 tests whether conventional edges can replace TRACE for instance segmentation by swapping each detector's edge map into our BGP pipeline (Sec. 3.5). AP mk falls from 9.4 (TRACE) to 1.2 with Canny (Canny, 1986), 3.7 with HED (Xie & Tu, 2015), 4.1 with PiDiNet (Su et al., 2023), and 4.3 with DiffusionEdge (Ye et al., 2024). The gap reflects an objective mismatch: these methods are trained to predict RGB intensity-change contours (not instance edges), which makes them sensitive to texture and illumination. Standard edge detection benchmarks (e.g., BSDS (Arbelaez et al., 2010)) prioritize low-level texture or color contrast, which is misaligned with the goal of instance boundary detection. To evaluate instanceaware boundary quality, we construct a new benchmark from COCO 2014 panoptic masks, defining pixels between distinct segments as ground-truth edges (see Fig. 11). To assess quality, we report ODS and OIS metrics (Xie & Tu, 2015;Su et al., 2023) for boundary precision and clDice (Shit et al., 2021) for topological connectivity (see details in Appendix F). Table 6 compares TRACE against representative edge detectors (Canny, 1986;Xie & Tu, 2015;Su et al., 2023;Ye et al., 2024). TRACE achieves an ODS of 0.889, more than doubling the performance of the strongest baseline (DiffusionEdge, 0.428). Conventional methods suffer from high false positives due to their sensitivity to internal textures, resulting in low precision for instance delineation. In contrast, TRACE effectively suppresses non-boundary gradients by leveraging diffusion priors. Furthermore, the superior clDice score (0.826) confirms that TRACE produces topologically connected boundaries, which is a critical property for successfully separating adjacent instances in downstream segmentation tasks.
Limitations. While TRACE demonstrates consistent improvements across 11 real-world benchmarks, including autonomous driving (see results in Tabs. 1, 2, 12, and 13), we identify limitations in specialized domains. First, for tiny instances (≈ 0.01% area) in satellite imagery (Wei et al., 2020;Waqas Zamir et al., 2019), performance degrades due to the spatial compression of the VAE in latent diffusion models. Second, on out-of-distribution medical images (e.g., histopathology) (Kumar et al., 2020;Naylor et al., 2019), the natural-image priors of standard diffusion backbones result in misaligned instance boundaries. We provide detailed quantitative results (Tabs. 14 and 15) and qualitative failure cases (Fig. 15) for these scenarios in Appendix E.4. Computational Overhead. In Tab. 7, TRACE introduces minimal additional cost across different evaluation settings. For training-free unsupervised instance segmentation in Tab. 1(a), TRACE refines each image's masks during inference, which increases latency by only about 2% compared to the ProMerge (Li & Shin, 2024). In contrast, for weakly-supervised panoptic segmentation (Tab. 1(b), Tab. 2), TRACE is used only once during training to refine pseudo instance or panoptic masks before the teacher network (e.g., Mask2Former) is trained, so there is no runtime overhead at inference.
this section cite: ['b8', 'b94', 'b79', 'b100', 'b3', 'b94', 'b79', 'b76', 'b8', 'b94', 'b79', 'b100', 'b88', 'b87', 'b43', 'b61', 'b45']

Section: CONCLUSION
TRACE demonstrates that text-to-image diffusion models naturally encode recoverable instance structure. By locating the Instance Emergence Point, extracting boundaries through self-attention, and compressing them into a fast one-step decoder, TRACE delivers sharp and connected instance edges in real time without any prompts, points, or boxes, or masks. Our extensive evaluation across diverse diffusion architectures confirms that this capability is intrinsic to the generative diffusion prior, consistently yielding superior instance boundary precision and topological connectivity compared to non-diffusion baselines and conventional edge detectors. These edges act as annotation-free instance seeds that boost both interactive systems like SAM and unsupervised/weakly-supervised pipelines, surpassing point-and box-supervised alternatives. Looking forward, the same principle opens opportunities for video panoptic segmentation, medical imaging, and open-vocabulary grouping where text and TRACE can be combined for scalable panoptic perception.
this section cite: []

Section: References
Ref_id:b0 Title: Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation Year: (2018)
Ref_id:b1 Title: Weakly supervised learning of instance segmentation with inter-pixel relations Year: (2019)
Ref_id:b2 Title: Methods of information geometry Year: (2000)
Ref_id:b3 Title: Contour detection and hierarchical image segmentation Year: (2010)
Ref_id:b4 Title: Qwen2.5-vl technical report Year: (2025)
Ref_id:b5 Title: All are worth words: A vit backbone for diffusion models Year: (2023)
Ref_id:b6 Title: Talking to dino: Bridging self-supervised vision backbones with language for open-vocabulary segmentation Year: (2025)
Ref_id:b7 Title: COCO-Stuff: Thing and stuff classes in context Year: (2018)
Ref_id:b8 Title: A computational approach to edge detection Year: (1986)
Ref_id:b9 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b10 Title: Learning to generate text-grounded mask for open-world semantic segmentation from only image-text pairs Year: (2023)
Ref_id:b11 Title: Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2024)
Ref_id:b12 Title: Class re-activation maps for weakly-supervised semantic segmentation Year: (2022)
Ref_id:b13 Title: Per-pixel classification is not all you need for semantic segmentation Year: (2021)
Ref_id:b14 Title: Maskedattention mask transformer for universal image segmentation Year: (2022)
Ref_id:b15 Title: The cityscapes dataset for semantic urban scene understanding Year: (2016)
Ref_id:b16 Title: Diffcut: Catalyzing zero-shot semantic segmentation with diffusion features and recursive normalized cut Year: (2024)
Ref_id:b17 Title: Qa-clims: Question-answer cross language image matching for weakly supervised semantic segmentation Year: (2023)
Ref_id:b18 Title: Panoptic segmentation: A review Year: (2021)
Ref_id:b19 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b20 Title: The pascal visual object classes (VOC) challenge Year: (2010)
Ref_id:b21 Title: Pointly-supervised panoptic segmentation Year: (2022)
Ref_id:b22 Title: Associating inter-image salient instances for weakly supervised semantic segmentation Year: (2018)
Ref_id:b23 Title: Eva-02: A visual representation for neon genesis Year: (2024)
Ref_id:b24 Title: Are we ready for autonomous driving? the kitti vision benchmark suite Year: (2012)
Ref_id:b25 Title: Lvis: A dataset for large vocabulary instance segmentation Year: (2019-06)
Ref_id:b26 Title: Scene-centric unsupervised panoptic segmentation Year: (2025)
Ref_id:b27 Title: Unsupervised semantic segmentation by distilling feature correspondences Year: (2022)
Ref_id:b28 Title: Mask r-cnn Year: (2017)
Ref_id:b29 Title: Conceptattention: Diffusion transformers learn highly interpretable features Year: (2025)
Ref_id:b30 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b31 Title: Exemplar-freesolo: Enhancing unsupervised instance segmentation with exemplars Year: (2023)
Ref_id:b32 Title: Mwsis: Multimodal weakly supervised instance segmentation with 2d box annotations for autonomous driving Year: (2024)
Ref_id:b33 Title: Mars: Model-agnostic biased object removal without additional supervision for weakly-supervised semantic segmentation Year: (2023-10)
Ref_id:b34 Title: DHR: Dual features-driven hierarchical rebalancing in inter-and intra-class regions for weakly-supervised semantic segmentation Year: (2024)
Ref_id:b35 Title: TTD: Text-tag selfdistillation enhancing image-text alignment in clip to alleviate single tag bias Year: (2024)
Ref_id:b36 Title: COIN: Confidence score-guided distillation for annotation-free cell segmentation Year: (2025)
Ref_id:b37 Title: Repurposing diffusion-based image generators for monocular depth estimation Year: (2024)
Ref_id:b38 Title: Beyond semantic to instance segmentation: Weakly-supervised instance segmentation via semantic knowledge transfer and self-refinement Year: (2022)
Ref_id:b39 Title: Semantic-aware superpixel for weakly supervised semantic segmentation Year: (2023)
Ref_id:b40 Title: Panoptic feature pyramid networks Year: (2019)
Ref_id:b41 Title: Segment anything Year: (2023)
Ref_id:b42 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b43 Title: A multi-organ nucleus segmentation challenge Year: (2020)
Ref_id:b44 Title: Weakly supervised semantic segmentation via adversarial learning of classifier and reconstructor Year: (2023)
Ref_id:b45 Title: Promerge: Prompt and merge for unsupervised instance segmentation Year: (2024)
Ref_id:b46 Title: Towards noiseless object contours for weakly supervised semantic segmentation Year: (2022)
Ref_id:b47 Title: Point-supervised panoptic segmentation via estimating pseudo labels from learnable distance Year: (2024)
Ref_id:b48 Title: Weakly-and semi-supervised panoptic segmentation Year: (2018-09)
Ref_id:b49 Title: Semanticaware instance mask generation for box-supervised instance segmentation Year: (2023)
Ref_id:b50 Title: Point2mask: Point-supervised panoptic segmentation via optimal transport Year: (2023)
Ref_id:b51 Title: Fully convolutional networks for panoptic segmentation Year: (2021)
Ref_id:b52 Title: Panoptic segformer: Delving deeper into panoptic segmentation with transformers Year: (2022)
Ref_id:b53 Title: Microsoft COCO: Common objects in context Year: (2014)
Ref_id:b54 Title: Clip is also an efficient segmenter: A text-driven approach for weakly supervised semantic segmentation Year: (2023-06)
Ref_id:b55 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b56 Title:  Year: (2024)
Ref_id:b57 Title: Adaptive early-learning correction for segmentation from noisy annotations Year: (2022)
Ref_id:b58 Title: Grounding dino: Marrying dino with grounded pre-training for open-set object detection Year: (2024)
Ref_id:b59 Title: The role of context for object detection and semantic segmentation in the wild Year: (2014)
Ref_id:b60 Title: Emerdiff: Emerging pixel-level semantic knowledge in diffusion models Year: (2024)
Ref_id:b61 Title: Segmentation of nuclei in histopathology images by deep regression of the distance map Year: (2019)
Ref_id:b62 Title: Dataset diffusion: Diffusion-based synthetic data generation for pixel-level semantic segmentation Year: (2023)
Ref_id:b63 Title: On power chi expansions of f -divergences Year: (2019)
Ref_id:b64 Title: Unsupervised universal image segmentation Year: (2024)
Ref_id:b65 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b66 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b67 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b68 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b69 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b70 Title: Boundary-enhanced co-training for weakly supervised semantic segmentation Year: (2023-06)
Ref_id:b71 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b72 Title: Token contrast for weakly-supervised semantic segmentation Year: (2023)
Ref_id:b73 Title: Self-supervised nuclei segmentation in histopathological images using attention Year: (2020)
Ref_id:b74 Title: Objects365: A large-scale, high-quality dataset for object detection Year: (2019-10)
Ref_id:b75 Title: Toward joint thing-and-stuff mining for weakly supervised panoptic segmentation Year: (2021)
Ref_id:b76 Title: cldice-a novel topology-preserving loss function for tubular structure segmentation Year: (2021)
Ref_id:b77 Title: Cuts3d: Cutting semantics in 3d for 2d unsupervised instance segmentation Year: (2025)
Ref_id:b78 Title:  Year: (2025)
Ref_id:b79 Title: Lightweight pixel difference networks for efficient visual representation learning Year: (2023)
Ref_id:b80 Title: Diffuse attend and segment: Unsupervised zero-shot segmentation using stable diffusion Year: (2024-06)
Ref_id:b81 Title: Dense contrastive learning for self-supervised visual pre-training Year: (2021)
Ref_id:b82 Title: Learning to segment objects without annotations Year: (2022)
Ref_id:b83 Title: Cut and learn for unsupervised object detection and instance segmentation Year: (2023)
Ref_id:b84 Title: Videocutler: Surprisingly simple unsupervised video instance segmentation Year: (2024)
Ref_id:b85 Title: Segment anything without supervision Year: ()
Ref_id:b86 Title: Tokencut: Segmenting objects in images and videos with self-supervised transformer and normalized cut Year: (2023)
Ref_id:b87 Title: isaid: A large-scale dataset for instance segmentation in aerial images Year: (2019)
Ref_id:b88 Title: Hrsid: A highresolution sar images dataset for ship detection and instance segmentation Year: (2020)
Ref_id:b89 Title: Datasetdm: Synthesizing data with perception annotations using diffusion models Year: ()
Ref_id:b90 Title: NeurIPS Year: (2023)
Ref_id:b91 Title: Unified perceptual parsing for scene understanding Year: (2018)
Ref_id:b92 Title: Yew Soon Ong, and Chen Change Loy. Mosaicfusion: Diffusion models as data augmenters for large vocabulary instance segmentation Year: (2025)
Ref_id:b93 Title: C2am: Contrastive learning of class-agnostic activation map for weakly supervised object localization and semantic segmentation Year: (2022)
Ref_id:b94 Title: Holistically-nested edge detection Year: (2015)
Ref_id:b95 Title: Openvocabulary panoptic segmentation with text-to-image diffusion models Year: (2023-06)
Ref_id:b96 Title: Multi-class token transformer for weakly supervised semantic segmentation Year: (2022)
Ref_id:b97 Title: Depth anything: Unleashing the power of large-scale unlabeled data Year: (2024)
Ref_id:b98 Title: Exploring clip's dense knowledge for weakly supervised semantic segmentation Year: (2025)
Ref_id:b99 Title: Ffr: Frequency feature rectification for weakly supervised semantic segmentation Year: (2025)
Ref_id:b100 Title: Diffusionedge: Diffusion probabilistic model for crisp edge detection Year: (2024)
Ref_id:b101 Title: A simple framework for text-supervised semantic segmentation Year: (2023-06)
Ref_id:b102 Title: Ferret: Refer and ground anything anywhere at any granularity Year: (2023)
Ref_id:b103 Title: Unifying panoptic segmentation for autonomous driving Year: (2022)
Ref_id:b104 Title: K-net: Towards unified image segmentation Year: (2021)
Ref_id:b105 Title: Yoloe: Real-time seeing anything Year: (2025-10)
Ref_id:b106 Title: Learning deep features for discriminative localization Year: (2016)
Ref_id:b107 Title: Semantic understanding of scenes through the ade20k dataset Year: (2019)
Ref_id:b108 Title: Extract free dense labels from clip Year: (2022)
Ref_id:b109 Title: Weakly supervised instance segmentation using class peak response Year: (2018)
Ref_id:b110 Title: Weaktr: Exploring plain vision transformer for weakly-supervised semantic segmentation Year: (2023)
