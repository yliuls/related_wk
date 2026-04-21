Title: SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation
Abstract: Few-shot segmentation aims to segment unseen categories from just a handful of annotated examples. This requires mechanisms to identify semantically related objects across images and accurately produce masks. We note that Segment Anything 2 (SAM2), with its prompt-and-propagate mechanism, provides strong segmentation capabilities and a built-in feature matching process. However, we show that its representations are entangled with task-specific cues optimized for object tracking, which impairs its use for tasks requiring higher level semantic understanding. Our key insight is that, despite its class-agnostic pretraining, SAM2 already encodes rich semantic structure in its features. We propose SANSA (Semantically AligNed SegmentAnything 2), a framework that makes this latent structure explicit, and repurposes SAM2 for few-shot segmentation through minimal task-specific modifications. SANSA achieves state-of-the-art on few-shot segmentation benchmarks designed to assess generalization and outperforms generalist methods in the popular in-context setting. Additionally, it supports flexible promptable interaction via points, boxes, or scribbles, and remains significantly faster and more compact than prior approaches. Code at: https://github.com/ClaudiaCuttano/SANSA.

Section: Introduction
Segmenting images is a core problem in Computer Vision, yet achieving high-quality results typically requires extensive human effort to annotate pixel-level masks. Moreover, conventional semantic segmentation methods [13,26,34,16] struggle to generalize to unseen categories. Inspired by the human ability to recognize novel objects from just a few examples, few-shot segmentation (FSS) [38,69,41,59] has emerged as a paradigm that leverages a small set of labeled reference samples to guide the segmentation of target images containing arbitrary, previously unseen classes.
To this end, recent work has turned to visual foundation models (VFMs) [7], which offer rich visual representations and strong generalization capabilities [52,82]. A natural approach [42,83,62] is to decouple the few-shot segmentation task into two stages: feature matching followed by promptable segmentation. This is typically achieved by combining DINOv2 [50], known for its strong semantic correspondence capabilities [86,44,85,63], with Segment Anything [35], which excels at producing high-quality segmentation masks [35,90,87]. While effective, these modular approaches add computational overhead and require prompt engineering to coordinate multiple VFMs [89].
We observe that Segment Anything 2 (SAM2) [54] offers an alternative paradigm. Designed for Video Object Segmentation, it operates as a prompt-and-propagate framework, where an object is specified via its mask and tracked across frames. To achieve this, SAM2 introduces a Memory Attention mechanism to implicitly match features across video frames and propagate masks over time with high spatial precision. While this feature matching is originally intended for object tracking * Equal contribution.
this section cite: ['b12', 'b25', 'b33', 'b15', 'b37', 'b69', 'b40', 'b59', 'b6', 'b52', 'b82', 'b41', 'b83', 'b62', 'b50', 'b86', 'b43', 'b85', 'b63', 'b34', 'b34', 'b90', 'b87', 'b89', 'b54']

Section: 39th Conference on Neural Information Processing Systems (NeurIPS 2025).

this section cite: []

Section: Low semantic shift

this section cite: []

Section: High visual similarity
High semantic shift Low visual similarity Figure 1: We evaluate frozen SAM2 on few-shot segmentation tasks on four datasets with varying degrees of semantic variability. On datasets [10,18] with low semantic shift and high intra-class visual similarity, SAM2 matches or even outperforms state-of-the-art APSeg [28]. However, on more challenging datasets like COCO and LVIS, with high semantic shift (e.g., cruise ship vs. rowboat), its performance drops significantly, compared with GF-SAM [83]. The bottom row illustrates examples of ground-truth masks in both scenarios.
Frozen SAM2 SANSA (ours)
Figure 2: We extract SAM2 features from object instances across diverse images and visualize their distribution using the first three principal components of PCA, mapped to RGB channels. The features appear entangled, with clusters mixing across categories, highlighting the lack of a coherent semantic structure in the original feature space. After adapting the feature space with SANSA, well-defined clusters emerge: semantically similar instances group together, forming coherent structures despite intra-class variation in visual appearance.
based on visual similarity, we note that this architecture inherently unifies two capabilities central to FSS within a single model: dense feature matching and high-quality mask generation. Building on this analogy, we propose repurposing the prompt-and-propagate framework to address the FSS task, by reinterpreting the temporal dimension of videos as a collection of semantically related images. This raises the central question: instead of limiting SAM2 to object tracking, can it generalize beyond visual similarity to perform semantic tracking based on shared concepts across images?
Addressing this inquiry means, in turn, to answer whether SAM2 has implicitly learned semanticaware representations, despite its class agnostic pre-training. To answer this, we conduct a toy experiment in Fig. 1, testing SAM2 performance on FSS datasets with different semantic variability. Interestingly, we observe that in low-semantic-shift scenarios, SAM2 achieves comparable or even higher performance with respect to state-of-the-art methods. However, on challenging datasets with high semantic shift, its performance drops drastically. A straightforward conclusion from these preliminary results would be that SAM2 may not have learned discriminative representations in its class-agnostic pretraining, making it unsuited for tasks requiring semantic alignment.
We challenge this interpretation, based on the observation that SAM2 pretraining, focused on instance matching across frames, shares similarities with self-supervised learning frameworks [77,76,29,5], which are known to elicit semantic understanding by enforcing feature invariance across views [11,4]. Given this analogy, we posit that SAM2 does encode semantic information, which is however entangled with instance-specific features optimized for object tracking, reflecting experimental evidence in Fig. 1. If our intuition is true, it implies that i) this structure could be disentangled through lightweight transformations [1,32,37], such as adapter modules and ii) it should be learnable from a set of base classes and generalize to unseen categories [57,17]. To this end, we i) intentionally use one of the simplest adapters in the literature, namely AdaptFormer [31], and ii) verify the generalization hypothesis through extensive experiments in strict few-shot benchmarks.
Building on this insight, we introduce SANSA (Semantically AligNed SegmentAnything 2), and show how to expose SAM2 latent semantic structure, repurposing its Memory Attention mechanism to shift from visual similarity to semantic similarity. The effectiveness of SANSA is illustrated in Fig. 2, where a 3D PCA, computed on unseen classes, reveals the emergent semantic organization of our features. We complement our method with a novel training objective designed to exploit SAM2 temporal continuity to convert each target image into a pseudo-annotation for subsequent frames.
Our contributions are the following:
• We are the first to investigate the semantic structure within SAM2. We show that such semantics can be disentangled through bottleneck transformations, enabling a unified approach that reinterprets few-shot segmentation as the task of tracking semantic concepts across images; • We validate SANSA through extensive experiments, achieving SOTA performance in strict FSS benchmarks while also outperforming generalist approaches in the 'in-context' scenario. Our experiments reveal that SAM2 encodes coarse-to-fine semantics, from high level concepts (e.g. dog vs. cat, +6.3% on COCO-20 i ) to fine-grained distinctions at category-level (e.g. dalmatian vs. bulldog, +8.3% on LVIS-92 i ) and part-level (e.g. hand vs. arm +4.6% on Pascal-Part); • By supporting prompts like points, boxes, or scribbles, our approach enables a wide range of downstream task, such as data annotation without the need for costly pixel-level masks. Finally, by exploiting the tight integration of memory attention and mask decoding in SAM2, we avoid the need for auxiliary models or complex pipelines, setting a new SOTA with a framework more than 3× faster than competitors, and 4-5× smaller in parameter count.
this section cite: ['b9', 'b17', 'b27', 'b83', 'b77', 'b76', 'b28', 'b4', 'b10', 'b3', 'b0', 'b31', 'b36', 'b57', 'b16', 'b30']

Section: Related works
Few-shot segmentation aims to segment a target image given an annotated reference. Early methods relied on compressed prototype representations [38,69,36,41,22], later replaced by attention-and correlation-based approaches [68,84,47,46,30] to better capture pixel-level relationships. More recently, research focused on leveraging the large-scale pretraining and generalization capabilities of vision foundation models [89,87,42,71,75]. Matcher [42] and GF-SAM [83] utilize a training-free pipeline: DINOv2 extracts correspondences which are used to prompt SAM for segmentation. Similarly, VRP-SAM [62] leverages a frozen SAM with an external encoder for feature matching. Recent works [87,89,44] explore the use of a single VFM [7] to jointly handle semantic understanding and mask prediction. DiffewS [89] exploit the emergent semantic correspondences in StableDiffusion [56] to unify the process, while SegIC [44] combines a frozen DINOv2 with a lightweight segmentation decoder. Yet, recent findings [86,85] suggest that diffusion and DINOv2 features offer complementary but disjoint strengths: the former offers spatial precision but weak semantics, the latter strong semantics but sparse matches. In contrast, we posit that SAM2 unifies both properties: its features possess high spatial granularity and implicitly encode semantic information. We show that this latent semantic structure can be extracted, effectively enabling FSS within a unified architecture.
this section cite: ['b37', 'b69', 'b35', 'b40', 'b21', 'b68', 'b84', 'b47', 'b46', 'b29', 'b89', 'b87', 'b41', 'b71', 'b75', 'b41', 'b83', 'b62', 'b87', 'b89', 'b43', 'b6', 'b89', 'b56', 'b43', 'b86', 'b85']

Section: Semantic correspondences and Foundation Models.
Finding correspondences between images is a longstanding problem in Computer Vision [6,43,33,55,49]. While early deep-learning approaches train dedicated models to establish semantic correspondences [33,55], recent works [86,63] have shown that VFMs enable generalization across tasks [66,2,73,19]. Among them, DINOv2 [50] and StableDiffusion [56] have demonstrated a compelling ability to establish semantic correspondences between images [86,49,85]. Recently, Segment Anything 2 [54] established itself as a foundation model for Video Object Segmentation. We observe that its pretraining, entailing matching object instances across frames under viewpoint changes and motion blur, closely parallels self-supervised learning frameworks [77,76,29] that derive semantic understanding through self-similarity training. However, the extent to which SAM2 embeddings encode (if any) semantic concepts has not been studied yet. Recent applications in specialized domains [91,3] utilize a frozen SAM2 in lowsemantic-shift scenarios (e.g. propagating masks across slices of 3D imagery given a support example). However, frozen SAM2 shows poor performance in standard FSS benchmarks requiring higher level semantic understanding. In this work, we shed light on this matter, providing insights into SAM2 feature structure and showing that semantic content can be disentanged from its embeddings.
this section cite: ['b5', 'b42', 'b32', 'b55', 'b49', 'b32', 'b55', 'b86', 'b63', 'b66', 'b1', 'b73', 'b18', 'b50', 'b56', 'b86', 'b49', 'b85', 'b54', 'b77', 'b76', 'b28', 'b91', 'b2']

Section: Method
We structure our investigation around the following key research questions: (1) Can semantic information be effectively extracted from SAM2 features? (2) Can this extraction occur without impairing the functionality of the SAM2 decoder, thereby maintaining its precise segmentation performance? (3) Finally, if a mapping that enhances semantic structure within SAM2 features can be learned, does this mapping generalize effectively to unseen classes?
Target image Semantic-aware features K reference image(s) with annotation (mask, point, box or scribble) Prompt Encoder Memory Attention Mask Decoder Target image Reference image(s) Memory Bank Memory Encoder store in memory bank AdaptFormer Image Encoder (visualized via 3D PCA) Dense Feature Matching High quality Mask Generation Given k annotated reference images and a target image, we construct a pseudo-video by concatenating them, then leverage SAM2 streaming pipeline to process reference frames together with their annotations sequentially. We restructure SAM2 feature space to make its latent semantic structure explicit, enabling mask propagation based on semantic similarity from reference to target. The emergent semantic structure is visualized by the 3D PCA projection of F.
These questions directly shape the design of our approach. Throughout the rest of the manuscript, we provide empirical evidence supporting affirmative answers to each.
this section cite: []

Section: Task Definition.
We consider the general k-shot segmentation setting, where the model is given
K reference pairs R = (x k r , a k r ) K k=1
, each consisting of an image x k r ∈ R H×W ×3 and its mask annotation a k r ∈ [0, 1] H×W . Given a target image x t ∈ R H×W ×3 , the goal is to predict y t ∈ [0, 1] H×W , the segmentation mask of objects in x t that are semantically aligned with the reference. As shown in Fig. 3, we interpret the references and the target as a sequence of frames in a pseudo-video M:
M = [x k r , a k r ] K k=1 ∪ [x t , ∅], (1
)
where only the k reference frames are annotated.
this section cite: []

Section: From Object Tracking to Semantic Tracking with SAM2
Segment Anything Model 2 extends SAM [35] to Promptable Video Object Segmentation. Like its predecessor, SAM2 comprises three main elements: an Image Encoder, a Prompt Encoder, and a Mask Decoder. Specifically, given an image x k r with features F k r , and a prompt a k r ∈ mask, point, box , the Mask Decoder processes them to produce the segmentation mask ŷk r . The key innovation of SAM2 lies in its extension to video domain: masks can be propagated across new unannotated frames x t without additional prompts, thanks to a memory mechanism. As shown in Fig. 3, we conceptually decompose SAM2 architecture into two functional components:
• Dense Feature Matching: Comprising the Memory Encoder, Memory Bank, and Memory Attention, this module establishes dense correspondences across frames. Concretely, given an object reference mask ŷk r (either predicted or given as prompt), the Memory Encoder constructs its memory representation, by fusing the mask with frame features F k r :
I k r = F k r + conv_down(ŷ k r ). (2
)
This representation is stored in the Memory Bank. Subsequently, the features F t of target unannotated frames undergo a cross-attention (Memory Attention) that aims at establishing dense correspondences between the current frame and the memory representations from previous frames:
F t,match = Attention Q(F t )K([I 0 r , ..., I k r ]) T V ([I 0 r , ..., I k r ]),(3)
where I k r are past representations and Q(•), K(•), V (•) the query, key, and value projections.
• High-quality Mask Generation: For unannotated frames, the Mask Decoder is tasked with refining the coarse features matches F t,match , which encode dense correspondences with prior object representations, to produce the segmentation output ŷt (cf Fig. 3).
We propose to repurpose the memory-based feature matching and mask decoding mechanisms, reinterpreting the temporal dimension of videos as a collection of semantically related images. Thus, rather than tracking a specific object across continuous frames, we aim at tracking its semantic class.
We highlight two advantages of this formulation: first, unlike recent approaches [62,44,89], our model naturally supports variable k-shots without modifications; second, our solution seamlessly supports promptable FSS, where prompts can take the form a k r ∈ mask, point, box, scribble , removing the reliance on pixel-level annotations. Finally, we note that reference frames are encoded in Memory Bank without undergoing Memory Attention (cf Fig. 3). By avoiding cross-referencing, we ensure predictions for the target image to be invariant to the ordering of reference images.
this section cite: ['b34', 'b62', 'b43', 'b89']

Section: SAM2 Feature Adaptation
At the core of the feature matching mechanism lie the learned feature representations, central to the Memory Attention mechanism: the reference object representation I r is constructed from F r (Eq. ( 2)), and matching is performed via cross-attention between target features F t and I r (Eq. ( 3)).
Our goal is repurposing the Memory Attention to shift from instance-level to semantic-level matching. To achieve this, we introduce minimal architectural changes and instead focus on restructuring the feature space itself. Specifically, we seek to induce a semantic organization of the features, enabling dense correspondences to reflect semantic similarity rather than visual-similarity.
We hypothesize that SAM2 features already encode semantic concepts, albeit entangled with signals specific to tracking, such as instance-level details and spatial biases. If such structure exists, then it should be learnable from a set of base classes by training few parameters [1,32,57]. Thus, we opt for simple AdaptFormer [14] blocks, although our analysis is not tied to the adaptation method, as we will show in the Experimental Section. We integrate AdaptFormer blocks within the last two layers of the Image Encoder, as these encode higher-level semantic representations. Given down-and up-projection matrices W down ∈ R d, d, W up ∈ R d,d , an AdaptFormer block operates token-wise:
A(x) = σ(x • W down ) • W up , (4
)
where σ is a ReLu and d < d is the bottleneck dimensionality. The adapted features are summed in a residual fashion in the backbone transformer blocks:
x self = Attention(x),(5)
x ′ = MLP(x self ) + x self + A(x self ),(6)
The backbone weights are kept frozen and we only train projections W down and W up .
this section cite: ['b0', 'b31', 'b57', 'b13']

Section: Training objective
Following standard practice in FSS, we adopt an episodic training paradigm [67,59,69,36]. We have access to a set of training episodes, each one containing annotated instances from a single class. We denote these episodes as {x i , y i }, where y i ∈ [0, 1] H×W is the mask that segments these instances.
Leveraging our model inherent ability to process sequences of variable length, we create challenging training examples by inverting the standard k-shot setup: instead of predicting a single target image from multiple references, the model receives a single labeled reference image and is tasked with propagating the concept to multiple unlabeled target images. Formally, we define the training clip:
M train = [x r , a r ] ∪ [x j t , ∅] J j=1 ,(7)
where {x r , a r } is the annotated reference image and {x j t } J j=1 , are the J unlabeled target images. We feed M train to our model, which propagates the provided concept across target frames to predict the masklet {ŷ j t } J j=1 . Initially, the Memory Bank is populated with the reference representation I r . We propose to condition the prediction for each target frame on the reference as well as on previous target predictions, by encoding in the Memory Bank the predicted representation I j t , computed as in Eq. ( 2). This design transforms each intermediate frame into a pseudo-reference for subsequent frames. This objective discourages overfitting to individual image pairs and encourages robust, semantically grounded correspondences, forcing the model to disentangle semantics from low-level features. We supervise the predicted {ŷ j t } J j=1 with a Binary Cross-Entropy loss and a Dice loss [45].
this section cite: ['b67', 'b59', 'b69', 'b35', 'b45']

Section: Visualizing SANSA Feature Space
In this section we analyze how adaptation reshapes SAM2 feature space. In Fig. 4a, we extract features for SAM2 and SANSA for a set of objects belonging to unseen classes for our trained model, and visualize their first two Principal Components (PCs), labeled by class. Frozen SAM2 features show weak class discriminability, reinforcing the hypothesis that semantic structure is entangled with task-specific features tailored for the original tracking objective, explaining SAM2 poor FSS performance. Consequently, the leading PCs reveal a mix of semantic and non-semantic signals.
In contrast, our SANSA features show clear semantic discriminability, mapping novel classes into well-defined semantic clusters. In Fig. 4b, we visualize SANSA features using PCA, computed jointly across images, where the first three PCs are mapped to RGB channels. The visualization spans three images containing instances from both unseen (e.g., person, chair, ball) and seen (e.g., umbrella) categories. The consistent color mapping across instances of the same category highlights strong semantic grouping, mapping similar concepts coherently in feature space despite visual variability.
Finally, in Fig. 4c, we shift from coarse category level to fine-grained correspondences at the part level. Following [86], we cluster object features with K-Means and match centroids across image pairs via the Hungarian Algorithm, then visualize matched clusters with shared colors to assess semantic consistency. The results show that, despite not being trained with part-level supervision, SANSA features encode part-level understanding (e.g. hand vs arm, handlebar vs wheel), spatial layout (e.g. upper vs lower wheel), and produce representations that align across images.
We provide an in-depth study of semantic representations in Appendix B, using PCA, clustering, and linear probing to analyze how semantics are encoded in SAM2 and made explicit through adaptation.
this section cite: ['b86']

Section: COCO-20 i FSS-1000 1-shot 5-shot 1-shot 5-shot 1-shot 5-shot
Training-free methods SAM 2 [54] -SAM2-L 224 M 16.5 26.3 32.2 44.2 73.0 84.3PerSAM [87] ICLR'24 SAM-H 641 M 11.5 -23.0 -71.2 -Matcher [42] ICLR '24 DINOv2-L+SAM-H 945 M 33.0 40.0 52.7 60.7 87.0 89.6 GF-SAM [83] NeurIPS'24 DINOv2-L+SAM-H 945 M 35.2 44.2 58.7 66.8 88.0 88.9
Strict k-shot segmentation AMFormer [72] NeurIPS'23 ResNet101 49 M --51.0 57.3 --HMNet [75] NeurIPS'24 RN50 39 M --52.1 58.9 --VRP-SAM [62] CVPR'24 RN50 + SAM-H 670 M 28.3 -53.9 -87.7 -SegIC [44] ECCV '24 DINOv2-G 1.2 B 40.5 -53.6 -88.5 -DiffewS [89] NeurIPS'24 StableDiffusion 890 M 33.9 43.7 52.2 60.7 90.2 90.6SANSA NeurIPS'25 SAM2-L 234 M 48.8 53.9 60.2 64.3 91.4 92.1 4 Experiments
Implementation details. We employ SAM2 with Hiera-Large [58] as encoder. AdaptFormer [14] is inserted into the last two blocks, with hidden size set to 0.3× the block channel dimension in the strict few-shot setting and 0.8× in the generalist. SAM2 is frozen, and only the adapters are trained (∼10M params in the strict case and ∼25M in the generalist). We train with AdamW and learning rate 10 -4 for 5 epochs (strict) and 20 (generalist), with k=1 (a single annotated reference) and sequence length J=3. The same model is evaluated on 1-shot and 5-shot. Full details are in Appendix H.
this section cite: ['b54', 'b87', 'b41', 'b83', 'b72', 'b75', 'b62', 'b43', 'b89', 'b58', 'b13']

Section: Datasets. COCO-20 i
[48] is built on MSCOCO [39] and consists of 80 classes split into four folds, each with 20 classes. FSS-1000 [23] contains 1000 classes, with 520 for training, 240 for validation, and 240 for testing. LVIS-92 i [42] is more challenging, selecting 920 classes from LVIS [24], divided in 10 folds. PASCAL-Part [42] includes four superclasses with 56 object parts across 15 classes. PACO-Part is built from PACO [53] and contains 303 classes, split in four folds.
this section cite: ['b38', 'b22', 'b41', 'b23', 'b41', 'b53']

Section: Strict Few-Shot Segmentation Setting
We first evaluate our method in strict FSS setting. Following standard protocols [72,62,61], we train on base classes and evaluate on novel classes with k-shots. These experiment address the question of whether the semantic mapping learned on base classes can transfer meaningfully to novel categories.
Few-shot segmentation. In Tab. 1, we compare SANSA, besides specialist models, such as AM-Former [72] and HMNet [75], with the most relevant and recent approaches based on Foundation Models. These include methods that, like ours, leverage a single Foundation Model, such as SegIC [44] and DiffewS [89], as well as modular two-stage pipelines like VRP-SAM [62]. For completeness, we also report results from generalist training-free methods built on DINOv2 and SAM, including PerSAM [87], Matcher [42], GF-SAM [83], as well as our baseline, i.e. frozen SAM2.
In the one-shot setting, SANSA consistently outperforms all prior methods, demonstrating superior generalization to unseen classes. Specifically, we surpass the best direct competitors SegIC, VRP-SAM and DiffwS by +8.3%, +6.3%, and +1.2% on LVIS-92 i , COCO-20 i , and FSS-1000, respectively. This performance gap remains consistent also against training-free approaches: compared to GF-SAM, SANSA achieves gains of +13.6%, +1.5%, and +3.4%. On the challenging LVIS-92 i , including 920 fine-grained categories (e.g., bulldog, dalmatian), DINO-based methods (SegIC, GF-SAM, Matcher) tend to underperform, possibly due to DINO overly-semantic features [20,86] (e.g., grouping distinct breeds under the concept of "dog"). Here, SANSA achieves a substantial +8.3% gain, suggesting that SAM2 encodes a latent hierarchical semantic structure, capturing both high-level semantic concepts (dominant in COCO-20 i ) and fine-grained distinctions (crucial for LVIS-92 i ). VRP-SAM SANSA(ours) Params 670M 234M point 38.4 53.4 +15.0 scribble 47.3 53.1 +5.8 box 49.7 54.3 +4.6 mask 53.9 60.2 +6.3 scribble box point Regarding the 5-shot setting, we note that SegIC and VRP-SAM do not provide an inference pipeline to extend to k-shot. In contrast, SANSA natively models correspondences across multiple reference frames and it outperforms the best competitor, DiffewS, by +10.2% and +3.6% on LVIS-92 i and COCO-20 i , respectively. Compared to training-free models, SANSA shows improvements of +9.7%, +3.2%, on LVIS-92 i and FSS-1000, while suffering a -2.5% gap on COCO-20 i w.r.t. GF-SAM. We also report results of upgrading SAM-based baselines to SAM2 in Appendix D.
Performance-Efficiency Trade-off. In Fig. 5, we analyze the trade-off between model size, inference speed (img/s), and performance. SANSA achieves state-of-the-art results while being the most lightweight solution. Specifically, it is: i) over 3× faster than the direct competing method (GF-SAM), ii) more compact, introducing only adapter parameters on top of SAM2 (totaling 234M), and iii) substantially more accurate, outperforming GF-SAM by +13.6% mIoU on the challenging LVIS-92 i benchmark. Importantly, SANSA keeps the SAM2 architecture entirely frozen, meaning that by storing only the adapter weights, it retains SAM2 state-of-the-art performance on Video Object Segmentation while also achieving top-tier results on few-shot segmentation within a single model.
A large-scale annotation study quantifying speed/quality trade-offs is presented in Appendix C.
this section cite: ['b72', 'b62', 'b61', 'b72', 'b75', 'b43', 'b89', 'b62', 'b87', 'b41', 'b83', 'b19', 'b86']

Section: One-Shot Promptable Segmentation.
Our framework keeps SAM2 decoder frozen, maintaining its capability to generate masks from any prompt (i.e. points, scribbles, or boxes). As such, during inference, users can segment reference objects by providing a simple point, without requiring costly pixel-level masks. In Fig. 6 we evaluate the performance with different prompts on strict FSS setting on COCO-20 i against VRP-SAM, which also supports such prompts. SANSA shows gains of +15.0%, +4.6%, and +5.8% when using points, scribbles, and boxes as annotations. More importantly, the performance drop from masks to point prompts is substantially smaller for SANSA (-6.8%) compared to VRP-SAM (-15.5%). We attribute this to the tightly coupled design of our architecture, which maximizes feature reuse by jointly leveraging the same representations for prompt encoding, feature matching, and mask decoding. Prompt generation details and qualitative results are in Appendix E.
this section cite: []

Section: Generalist In-Context Setting
Following recent few-shot segmentation works [44,89,42,83], we evaluate SANSA in the in-context segmentation setting, where a single generalist model is tested across multiple benchmarks (cf. Tab. 2).
Given the lack of a standardized training protocol, we explicitly report additional datasets used by each method beyond ADE20K and COCO, which are shared across all approaches. We evaluate three configurations of our approach: i) a minimal setup using only COCO and ADE20K, ii) an extended version incorporating LVIS, following the setup of SegIC [44], and iii) a variant including PACO to mitigate object-level bias and reinforce part segmentation capabilities.
When trained only on COCO and ADE20K coarse categories, SANSA exhibits strong out-of-domain generalization on the LVIS benchmark, outperforming DiffewS and SINE by +4.8% and +5.0%, respectively. Moreover, when LVIS is included in the training set (matching SegIC in-domain setup), SANSA further improves performance and surpasses SegIC by +5.4%. Despite not being trained at the part level, SANSA exhibits strong cross-task generalization capabilities, surpassing generalist baselines by a large margin (+7.5% and +8.4% over DiffewS, the best competitor on Pascal-Part and Paco-Part, respectively), including those trained on part segmentation datasets [71,40]. Our method is only outperformed by training-free models, which, by design, are not biased by object-level training. To address this bias, we follow [71,40] and augment our training set with PACO, leading to +6.7% on Paco-Part (in-domain) and +4.6% on Pascal-Part (out-of-domain) against the strong baseline of GF-SAM. Finally, we highlight that, within the generalist model category, SANSA is the most compact solution, with only 250 M parameters.
this section cite: ['b43', 'b89', 'b41', 'b83', 'b43', 'b71', 'b39', 'b71', 'b39']

Section: Generalization across domains and styles
To explore how SANSA generalizes beyond the scope of standard benchmarks, we follow [62,42] and present qualitative examples drawn from in-the-wild image pairs, shown in Fig. 7. In each case, the model is tasked with few-shot segmentation using a reference image from COCO [39] and a target image collected from the web, offering a complementary view to benchmark evaluations.
this section cite: ['b62', 'b41', 'b38']

Section: Reference image
Target images These examples introduce a different challenge compared to traditional benchmarks, focusing on domain and style shifts, such as when the target image is a cartoon or stylized sketch, or when the visual appearance changes dramatically between the reference and target. These results demonstrate SANSA strong generalization. Importantly, this robustness across domains and styles is inherited from the frozen SAM2. Our method preserves this capability by learning a semantic mapping within the frozen feature space, enabling reliable correspondence even under significant distribution shifts.
this section cite: []

Section: Ablation studies
We investigate three central aspects of our design: adaptation vs. fine-tuning, adapter architecture and capacity, and adapter placement. Results are summarized in Tab. 3.
this section cite: []

Section: Why adaptation instead of fine-tuning?
A central question of our work is how to distill knowledge from a pretrained model (i.e., SAM2) while preserving generalization, a key challenge in FSS. To this end, we evaluate fine-tuning strategies targeting the decoder, QKV projections, backbone, and full model, and compare them with inserting adapters into frozen weights. Results show that adaptation outperforms fine-tuning, indicating that it better preserves SAM2 pretrained priors, shifting the embedding space toward task-relevant semantics without altering the underlying representations.
What makes an adaptation strategy effective? Basic adapters such as LoRA [32], Adapter [31], and AdaptFormer [14] all yield similar gains around ∼27% mIoU over frozen SAM2. The slight gains (∼2%) with Adapter and AdaptFormer w.r.t. LoRA suggest that a simple non-linearity can refine this structure but not fundamentally reshape it. By contrast, increasing adapter capacity, either by enlarging the bottleneck or using more complex designs such as MONA [80], reduces generalization. These results show that effective adaptation thus requires simplicity and constraint: low-capacity, bottlenecked modules best expose SAM2 latent semantics, while excessive capacity tends to overfit.
Where should adapters be placed? Prior works often insert adapters throughout the entire backbone. In our case, we find that adapting only the last two stages is sufficient, as these layers already capture high-level semantic information, which is the focus of our disentanglement objective.
Additional ablations, including backbone scale and training objective, are provided in Appendix D.
this section cite: ['b31', 'b30', 'b13', 'b80']

Section: Conclusions
In this work, we introduced SANSA, which enhances SAM2 to accept and propagate a visual prompt across frames, and showed that its memory attention mechanism can be re-focused towards semantic correspondences by restructuring the feature space. We addressed three fundamental inquiries: (1) we demonstrated that semantic information can be extracted from SAM2 features through lightweight bottleneck transformations, answering our initial research question, (2) we showed that this can be achieved while keeping SAM2 frozen, thereby preserving its segmentation capabilities, and finally (3) we experimentally verified that the learned semantic mapping generalizes robustly to novel classes, with SOTA performance on strict few-shot benchmarks and against generalist models. Beyond technical contributions, SANSA offers practical advantages: it supports diverse prompts for annotation-efficient applications, and offers a 3x speed increase. Finally, our results suggest that foundation models like SAM2 may contain richer task-adaptable knowledge than their original objectives imply, a direction worthy of future exploration.
this section cite: []

Section: References
Ref_id:b0 Title: Intrinsic dimensionality explains the effectiveness of language model fine-tuning Year: (2021)
Ref_id:b1 Title: Foundation models defining a new era in vision: a survey and outlook Year: (2025)
Ref_id:b2 Title: Exploring the Potential of SAM2 for Few-Shot Medical Image Segmentation without Fine-tuning Year: (2024)
Ref_id:b3 Title: Vicreg: Variance-invariance-covariance regularization for selfsupervised learning Year: (2022)
Ref_id:b4 Title: Vicregl: Self-supervised learning of local visual features Year: (2022)
Ref_id:b5 Title: Surf: Speeded up robust features Year: (2006)
Ref_id:b6 Title: On the opportunities and risks of foundation models Year: (2021)
Ref_id:b7 Title: Few-shot segmentation without meta-learning: A good transductive inference is all you need? Year: (2021)
Ref_id:b8 Title: Language models are few-shot learners Year: (2020)
Ref_id:b9 Title: Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration Year: (2013)
Ref_id:b10 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b11 Title: Pixel matching network for cross-domain few-shot segmentation Year: (2024)
Ref_id:b12 Title: Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs Year: (2017)
Ref_id:b13 Title: Adaptformer: Adapting vision transformers for scalable visual recognition Year: (2022)
Ref_id:b14 Title: Detect what you can: Detecting and representing objects using holistic models and body parts Year: (2014)
Ref_id:b15 Title: Masked-attention mask transformer for universal image segmentation Year: (2022)
Ref_id:b16 Title: Meta-adapter: An online few-shot learner for vision-language model Year: (2023)
Ref_id:b17 Title: Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic) Year: (2019)
Ref_id:b18 Title: Infusing wisdom in sam2 for textdriven video segmentation Year: (2025)
Ref_id:b19 Title: Understanding self-supervised features for learning unsupervised instance segmentation Year: (2023)
Ref_id:b20 Title: The pascal visual object classes (voc) challenge Year: (2010)
Ref_id:b21 Title: Self-support few-shot semantic segmentation Year: (2022)
Ref_id:b22 Title: A mutually supervised graph attention network for few-shot segmentation: The perspective of fully utilizing limited samples Year: (2022)
Ref_id:b23 Title: Lvis: A dataset for large vocabulary instance segmentation Year: (2019)
Ref_id:b24 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b25 Title: Mask r-cnn Year: (2017)
Ref_id:b26 Title: On the effectiveness of adapter-based tuning for pretrained language model adaptation Year: (2021)
Ref_id:b27 Title: Apseg: auto-prompt network for cross-domain few-shot semantic segmentation Year: (2024)
Ref_id:b28 Title: Efficient visual pretraining with contrastive detection Year: (2021)
Ref_id:b29 Title: Cost aggregation with 4d convolutional swin transformer for few-shot segmentation Year: (2022)
Ref_id:b30 Title: Parameter-efficient transfer learning for NLP Year: (2019)
Ref_id:b31 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b32 Title: Fcss: Fully convolutional self-similarity for dense semantic correspondence Year: (2017)
Ref_id:b33 Title: Panoptic segmentation. in 2019 ieee Year: (2018)
Ref_id:b34 Title: Segment anything Year: (2023)
Ref_id:b35 Title: Learning what not to segment: A new perspective on few-shot segmentation Year: (2022)
Ref_id:b36 Title: Measuring the intrinsic dimension of objective landscapes Year: (2018)
Ref_id:b37 Title: Adaptive prototype learning and allocation for few-shot segmentation Year: (2021)
Ref_id:b38 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b39 Title: A simple image segmentation framework via in-context examples Year: (2024)
Ref_id:b40 Title: Learning non-target knowledge for few-shot semantic segmentation Year: (2022)
Ref_id:b41 Title: Segment anything with one shot using all-purpose feature matching Year: (2023)
Ref_id:b42 Title: Distinctive image features from scale-invariant keypoints Year: (2004)
Ref_id:b43 Title: Segic: Unleashing the emergent correspondence for in-context segmentation Year: ()
Ref_id:b44 Title:  Year: (2024)
Ref_id:b45 Title: V-net: Fully convolutional neural networks for volumetric medical image segmentation Year: (2016)
Ref_id:b46 Title: Hypercorrelation squeeze for few-shot segmentation Year: (2021-10-03)
Ref_id:b47 Title: Msi: Maximize supportset information for few-shot segmentation Year: (2023)
Ref_id:b48 Title: Feature weighting and boosting for few-shot segmentation Year: (2019)
Ref_id:b49 Title: Neural congealing: Aligning images to a joint semantic atlas Year: (2023)
Ref_id:b50 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b51 Title: A benchmark dataset and evaluation methodology for video object segmentation Year: (2016)
Ref_id:b52 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b53 Title: Paco: Parts and attributes of common objects Year: (2023)
Ref_id:b54 Title: Segment anything in images and videos Year: (1920)
Ref_id:b55 Title: End-to-end weakly-supervised semantic alignment Year: (2018)
Ref_id:b56 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b57 Title: Meta-learning with latent embedding optimization Year: (2018)
Ref_id:b58 Title: Hiera: A hierarchical vision transformer without the bells-and-whistles Year: ()
Ref_id:b59 Title: One-shot learning for semantic segmentation Year: (2017)
Ref_id:b60 Title: Efficient video object segmentation via modulated cross-attention memory Year: (2024)
Ref_id:b61 Title: Dense cross-query-and-support attention weighted mask aggregation for few-shot segmentation Year: (2022)
Ref_id:b62 Title: VRP-SAM: SAM with visual reference prompt Year: (1920)
Ref_id:b63 Title: Emergent correspondence from image diffusion Year: (2023)
Ref_id:b64 Title: Joint recovery of dense correspondence and cosegmentation in two images Year: (2016)
Ref_id:b65 Title: Three things everyone should know about vision transformers Year: (2022)
Ref_id:b66 Title: Splicing vit features for semantic appearance transfer Year: (2022)
Ref_id:b67 Title: Matching networks for one shot learning Year: (2016)
Ref_id:b68 Title: Few-shot semantic segmentation with democratic attention networks Year: (2020)
Ref_id:b69 Title: Panet: Few-shot image semantic segmentation with prototype alignment Year: (2019)
Ref_id:b70 Title: Images speak in images: A generalist painter for in-context visual learning Year: (2023)
Ref_id:b71 Title: Seggpt: Towards segmenting everything in context Year: (2009)
Ref_id:b72 Title: Focus on query: Adversarial mining transformer for few-shot segmentation Year: (2023)
Ref_id:b73 Title: Stronger fewer & superior: Harnessing vision foundation models for domain generalized semantic segmentation Year: (2024)
Ref_id:b74 Title: Eliminating feature ambiguity for few-shot segmentation Year: (2024)
Ref_id:b75 Title: Hybrid mamba for few-shot segmentation Year: (2025)
Ref_id:b76 Title: Instance localization for self-supervised detection pretraining Year: (2021)
Ref_id:b77 Title: Inscon: Instance consistency feature representation via self-supervised learning Year: (2022)
Ref_id:b78 Title: Associating objects with transformers for video object segmentation Year: (2021)
Ref_id:b79 Title: Decoupling features in hierarchical propagation for video object segmentation Year: (2022)
Ref_id:b80 Title: 5%> 100%: Breaking performance shackles of full fine-tuning on visual recognition tasks Year: (2025)
Ref_id:b81 Title: Free-form image inpainting with gated convolution Year: (2019)
Ref_id:b82 Title: On the power of foundation models Year: (2023)
Ref_id:b83 Title: Bridge the Points: Graph-based Few-shot Segment Anything Semantically Year: (2024)
Ref_id:b84 Title: Few-shot segmentation via cycle-consistent transformer Year: (2021)
Ref_id:b85 Title: Telling left from right: Identifying geometry-aware semantic correspondence Year: (2024)
Ref_id:b86 Title: A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence Year: (2023)
Ref_id:b87 Title: Personalize segment anything model with one shot Year: (2023)
Ref_id:b88 Title: Semantic understanding of scenes through the ade20k dataset Year: (2019)
Ref_id:b89 Title: Unleashing the potential of the diffusion model in few-shot semantic segmentation Year: (2009)
Ref_id:b90 Title: Segment everything everywhere all at once. Advances in neural information processing systems Year: (2023)
Ref_id:b91 Title: Rethinking Few-Shot Medical Image Segmentation by SAM2: A Training-Free Framework with Augmentative Prompting and Dynamic Matching Year: (2025)
