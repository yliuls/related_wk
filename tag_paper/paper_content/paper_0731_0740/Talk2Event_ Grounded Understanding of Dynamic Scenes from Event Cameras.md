Title: Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras
Abstract: Grounded scene understanding from event streams. This work presents Talk2Event, a novel task for localizing objects from event cameras using natural language, where each unique object in the scene is defined by four key attributes: ①Appearance, ②Status, ③Relation-to-Viewer, and ④Relation-to-Others. We find that modeling these attributes enables precise, interpretable, and temporally-aware grounding across diverse dynamic environments in the real world.

Section: Introduction
Event cameras [11,23,80] have emerged as a promising alternative to traditional frame-based sensors, offering unique advantages such as microsecond-level latency [107,6], low power consumption [75,79,22], and robustness to motion blur and low-light conditions [77,38,40,20]. These properties make event cameras highly suitable for high-speed and dynamic scenarios, as demonstrated in various perception tasks including detection [30,27,63], segmentation [82,31,32,45], and visual odometry [10,70,35]. However, a key capability remains unexplored in the event domain: visual groundingthe ability to localize objects in the scene based on free-form language descriptions. Visual grounding [93,54] is a cornerstone of multimodal perception, enabling applications such as human-AI interaction, language-guided navigation, and open-vocabulary object localization [91,40]. While extensive efforts have been made in frame-based [99,97,96] and 3D grounding [100,102,13,1,106,51] across images [95], videos [55], and remote sensing data [81,104,47,110], these benchmarks are built upon dense sensors that struggle under motion blur, lighting changes, or fastmoving objects. Despite their advantages, event cameras have not been studied in this context, leaving open questions about how to bridge asynchronous sensing with free-form, natural language.
To fill this gap, we introduce Talk2Event, the first benchmark for language-driven object grounding in event-based perception. The dataset provides 5,567 scenes, 13,458 annotated objects, and 30,690 high-quality referring expressions. To move beyond coarse descriptions, we introduce four grounding attributes -①Appearance, ②Status, ③Relation-to-Viewer, and ④Relation-to-Othersthat explicitly capture spatiotemporal and relational cues critical for grounding in dynamic environments. As shown in Fig. 1, our dataset features multi-caption supervision and fine-grained attribute annotations, setting a new standard for multimodal, temporally-aware event-based grounding.
Complementing the dataset, we propose EventRefer, an attribute-aware grounding framework that models the four grounding attributes via a Mixture of Event-Attribute Experts (MoEE). MoEE dynamically fuses attribute-specific features, allowing the model to adapt to appearance, motion, and relational cues. By treating attributes as co-located pseudo-targets, our design provides dense supervision without increasing decoder complexity. At inference, a lightweight fusion selects the most informative attributes for precise grounding. Supporting event-only, frame-only, and event-frame fusion, EventRefer outperforms strong baselines [37,58], especially in dynamic scenes.
The key contributions of this work can be summarized as follows:
• Talk2Event, the first large-scale event-based visual grounding benchmark, with linguistically rich and attribute-aware annotations spanning 5,567 scenes and 30,690 expressions.
• A multi-attribute annotation protocol that captures appearance, motion, egocentric relations, and inter-object context, enabling interpretable and compositional grounding.
• EventRefer, an attribute-aware grounding framework with a mixture of event-attribute experts, achieving state-of-the-art performance across event-only, frame-only, and fusion settings.
2 Related Work Dynamic Visual Perception. Event cameras have advanced dynamic scene understanding under highspeed or low-light conditions, supported by benchmarks in driving and indoor scenarios [12,3,28,112,73] and synthetic datasets for scalable training [42,25,17]. Recent works address robustness to noise and illumination changes [14,115,89], extending applications to action recognition [74,109,5] and autonomous driving [7,116,19,36,69]. Popular tasks include object detection [26,117,111,48,27,30,65,98], semantic segmentation [2,24,85,84,82,44,39,94,4], optical flow [29,113,114], and SLAM or odometry [76,35]. However, these focus on geometric or low-level semantics, leaving open-vocabulary grounding unexplored. Talk2Event fills this gap as the first benchmark linking event data and natural language for multimodal, temporally grounded understanding.
this section cite: ['b10', 'b22', 'b79', 'b106', 'b5', 'b74', 'b78', 'b21', 'b76', 'b37', 'b39', 'b19', 'b29', 'b26', 'b62', 'b81', 'b30', 'b31', 'b44', 'b9', 'b69', 'b34', 'b92', 'b53', 'b90', 'b39', 'b98', 'b96', 'b95', 'b99', 'b101', 'b12', 'b0', 'b105', 'b50', 'b94', 'b54', 'b80', 'b103', 'b46', 'b109', 'b36', 'b57', 'b11', 'b2', 'b27', 'b111', 'b72', 'b41', 'b24', 'b16', 'b13', 'b114', 'b88', 'b73', 'b108', 'b4', 'b6', 'b115', 'b18', 'b35', 'b68', 'b25', 'b116', 'b110', 'b47', 'b26', 'b29', 'b64', 'b97', 'b1', 'b23', 'b84', 'b83', 'b81', 'b43', 'b38', 'b93', 'b3', 'b28', 'b112', 'b113', 'b75', 'b34']

Section: Visual Grounding.
Object localization from RGB images has been widely studied using regionranking [86,88,97] and transformer-based methods [41,37]. These models typically learn from short phrases on static datasets [78,49,15], with extensions to video grounding [53] and RGB-D scenes [13,1,106,100]. Despite advances, existing datasets rely on dense frames or depth, lacking temporally sparse, high-speed sensing like events. Our work introduces the first benchmark and method for grounding in asynchronous event data, where our proposed EventRefer further models attribute-aware reasoning to bridge motion, spatial, and relational cues in event streams.
Multimodal Dynamic Scene Understanding. Beyond RGB, grounding has been explored in point clouds [102] and remote sensing [81,104,47,110]. 3D methods either rely on proposal-based Table 1: Summary of visual grounding benchmarks. We compare datasets from aspects including:
1 Sensor ( Frame, RGB-D, LiDAR, Event), 2 Type, 3 Statistics (number of scenes, objects, referring expressions, and average length per caption), and supported 4 Attributes for grounding, i.e., ①Appearance (δ a ), ②Status (δ s ), ③Relation-to-Viewer (δ v ), ④Relation-to-Others (δ o ).
Dataset Venue Sensory Scene Statistics Attributes Data Type Scene Obj. Expr.
Len. δ a δ s δ v δ o RefCOCO+ [101] ECCV'16 Static 19,992 49,856 141,564 3.53 ✓ ✗ ✗ ✗ RefCOCOg [101] ECCV'16 Static 26,711 54,822 85,474 8.43 ✓ ✗ ✗ ✓ Nr3D [1] ECCV '20 Static 707 5,878 41,503 -✓ ✗ ✗ ✓ Sr3D [1] ECCV'20 Static 1,273 8,863 83,572 -✓ ✗ ✗ ✓ ScanRefer [13] ECCV '20 Static 800 11,046 51,583 20.3 ✓ ✗ ✗ ✓ Text2Pos [43] CVPR '22 Static -6,800 43,381 -✓ ✗ ✓ ✗ CityRefer [68] NeurIPS '23 Static -5,866 35,196 -✓ ✗ ✗ ✓ Ref-KITTI [90] CVPR'23 Static 6,650 -818 -✓ ✗ ✓ ✗ M3DRefer [105] AAAI'24 Static 2,025 8,228 41,140 53.2 ✓ ✗ ✓ ✗ STRefer [52] ECCV '24 Static 662 3,581 5,458 -✓ ✗ ✗ ✗ LifeRefer [52] ECCV'24 Static 3,172 11,864 25,380 -✓ ✗ ✗ ✗ Talk2Event Ours Dynamic 5,567 13,458 30,690 34.1 ✓ ✓ ✓ ✓ matching [21,103,59,46] or direct regression [64,56,33,60]. Recent works have started to explore vision-language models for event data [50,57], but none address grounding with spatial boxes or multi-attribute reasoning. To push beyond frame-based methods, our work expands the frontier of multimodal grounding by introducing the first large-scale benchmark and method for grounding in event-based dynamic scenes, while connecting to broader efforts in multimodal scene understanding.
this section cite: ['b85', 'b87', 'b96', 'b40', 'b36', 'b77', 'b48', 'b14', 'b52', 'b12', 'b0', 'b105', 'b99', 'b101', 'b80', 'b103', 'b46', 'b109', 'b100', 'b100', 'b0', 'b0', 'b12', 'b42', 'b67', 'b89', 'b104', 'b51', 'b51', 'b20', 'b102', 'b58', 'b45', 'b63', 'b55', 'b32', 'b59', 'b49', 'b56']

Section: Talk2Event: Dataset & Benchmark
In this section, we first introduce the formal task definition of event-based visual grounding and its multimodal grounding objectives (Sec. 3.1), and then present the data curation pipeline of Talk2Event, featuring linguistically rich, attribute-aware annotations built on real-world driving data (Sec. 3.2).
this section cite: []

Section: Task Formulation: Visual Grounding from Event Streams
Problem Definition. We define event-based grounding as the task of localizing an object in dynamic scenes captured by event cameras, based on a free-form language description. Formally, given a voxelized event representation E and a referring expression S = {w 1 , w 2 , . . . , w C } of C tokens, the goal is to predict a bounding box b = (x, y, w, h) that correctly localizes the referred object.
Event cameras produce asynchronous streams of events E = {e k } N k=1 , where each event e k = (x k , y k , t k , p k ) encodes the spatial coordinates, timestamp, and polarity p k ∈ {-1, +1}. Following prior work [30,63], we discretize the stream into a spatiotemporal voxel grid, that is:
E(p, τ, x, y) = e k ∈E δ(p -p k ), δ(x -x k , y -y k ), δ(τ -τ k ),(1)
where τ k = t k -ta t b -ta × T maps each timestamp to one of T temporal bins within the observation window [t a , t b ]. This process produces a dense 4D tensor E ∈ R 2×T ×H×W that preserves the spatiotemporal structure and polarity of the events, making it compatible with modern backbones.
this section cite: ['b29', 'b62']

Section: Benchmark Configuration.
In addition to the event voxel grid E, our benchmark optionally provides synchronized frames F ∈ R 3×H×W captured at timestamp t 0 . This design supports three evaluation configurations: grounding with 1 event data only, 2 frame data only, or a 3 combination of both. This setup allows systematic analysis of individual modalities and their fusion in dynamic scenes.
this section cite: []

Section: Grounding Objectives.
To facilitate fine-grained, interpretable, and compositional grounding, we annotate each referring expression with four attribute categories that capture complementary aspects of the target object and its surrounding context:
• Appearance: Describes static visual properties of the object, such as category, color, size ("large", "small"), and geometric shape. This attribute supports traditional appearance-based localization. • Status: Refers to dynamic behaviors or states, including motion (e.g., "moving", "stopped"), trajectory (e.g., "turning", "approaching"), or action (e.g., "crossing"). This is crucial for grounding objects in dynamic environments captured by high temporal-resolution sensors.
this section cite: []

Section: F(+1)
Displacement Displacement Displacement Timestamp:
• Relation-to-Viewer: Captures egocentric spatial relationships between the object and the observer, such as position (e.g., "on the left", "in front"), distance (e.g., "nearby", "far"), or perspective (e.g., "facing towards", "looking in the same direction"). This supports view-conditioned grounding.
• Relation-to-Others: Models relational context with other objects in the scene, such as spatial arrangements (e.g., "next to a bus", "behind a car") or joint configurations (e.g., "two pedestrians walking together"). This enables context-aware disambiguation in crowded or complex scenes.
We design these four attributes to explicitly expose the diverse spatiotemporal cues that are critical for grounding in event-based streams (along with the optional frames for more appearance and semantic cues). As summarized in Tab. 1, existing grounding benchmarks primarily focus on static scenes and lack such structured attribute-level supervisions, which are crucial for understanding dynamic scenes.
this section cite: []

Section: Dataset Curation
We build Talk2Event on top of DSEC [28], a large-scale dataset featuring time-synchronized events and high-resolution frames captured in diverse urban environments. Our goal is to transform this raw sensory data into a comprehensive event-based visual grounding benchmark with linguistically rich and attribute-aware annotations, as depicted in Fig. 1. Below, we detail our curation pipeline.
this section cite: ['b27']

Section: Context-Aware Referring Expression Generation.
As illustrated in Fig. 2, we leverage temporal context to generate rich and diverse referring expressions. Given two surrounding frames at t 0 -∆t and t 0 + ∆t (∆t = 200 ms), we prompt Qwen2-VL [87] to describe the target object at t 0 . This context exposes object displacement and scene dynamics, encouraging descriptions that capture both appearance and motion, as well as spatial and relational cues. We generate three distinct expressions per object, refined through human validation for correctness and diversity. On average, each object is described by 34.1 words -making Talk2Event one of the most linguistically rich grounding datasets. Attribute-specific word clouds in Fig. 2 further highlight how our prompting covers the four attributes introduced in Sec. 3.1. Due to space limits, detailed elaborations are placed in the Appendix.
this section cite: ['b86']

Section: Attribute Annotation and Verification.
Each expression is further decomposed into four compositional attributes -①Appearance (δ a ), ②Status (δ s ), ③Relation-to-Viewer (δ v ), and ④Relation-to-Others (δ o ) -using a semi-automated pipeline that combines fuzzy matching with language model assistance. Human verification ensures the semantic accuracy of these annotations, providing structured and interpretable supervision for multi-attribute grounding.
this section cite: []

Section: Quality Assurance.
We apply rigorous filtering and validation to ensure data quality: (i) visibility filtering removes small, occluded, or ambiguous objects; (ii) redundancy filtering ensures that the three captions per object are linguistically distinct; (iii) attribute validation checks that each caption  meaningfully references at least one attribute. This pipeline yields 5,567 curated scenes, 13,458 annotated objects, and 30,690 high-quality referring expressions, establishing Talk2Event as a robust resource for studying multimodal, language-driven grounding in dynamic environments.
Due to space limits, additional annotation details and dataset examples are placed in the Appendix.
this section cite: []

Section: EventRefer: Attribute-Aware Grounding Framework
Building upon Talk2Event, we introduce a novel grounding framework that leverages the rich but implicit information in event streams (see Fig. 3). Different from frames, events carry almost no appearance texture; however, their asynchronous nature excels at capturing motion cues and relationships among objects over time. We aim to inject missing semantic cues, preserve the finegrained temporal resolution of events, and explicitly model the contribution of each attribute.
this section cite: []

Section: Positive Word Matching from Attributes
Grounding requires knowing where in an expression the referred object or its attributes are mentioned. Manual token-span labels, as provided for RGB corpora [37,41], are impractical for the four attributes and the expressions in Talk2Event. Instead, we employ a lightweight fuzzy matcher: for each attribute δ i in {δ a , δ s , δ v , δ o }, e.g., a short cue phrase such as "moving left" for status (δ s ), the matcher will locate all occurrences (with synonym handling) in the raw referring expression.
The expression is then tokenized, each matched character span is projected onto token indices to form a binary positive map m i ∈ {0, 1} C , where [m i ] j = 1 if and only if token j lies inside any span for attribute δ i , and C is the encoded token length. Finally, we apply softmax to m i , assigning equal probability to each positive position and 0 elsewhere, which encourages the model to attend to all tokens expressing attribute δ i while remaining robust to paraphrasing.
this section cite: ['b36', 'b40']

Section: MoEE: Mixture of Event-Attribute Experts
Attribute-Aware Masking. The event features are extracted with a recurrent Transformer backbone following RVT [30], while the referring expressions are embedded with RoBERTa [61]. As shown in Fig. 3, we concatenate these embeddings together and feed them into a DETR [9]-style Transformer encoder, yielding hidden states H ∈ R B×Q×C , where B is the batch size, Q is the number of queries, and C is the channel dimension. For each attribute δ i , we construct a binary mask m att i = m i ∨ m 0 , where m 0 represents the union of all tokens that do not belong to any specific attribute, providing general contextual information (referred to as public context). This ensures that attribute-specific reasoning retains surrounding context, improving robustness to incomplete or noisy attribute cues. Applying the mask gives the attribute-specific hidden states H att i = m att i ⊙ H, which spotlight the positions relevant to attribute δ i while retaining neighbouring context. These four parallel features are then passed to the mixture-of-experts fusion module for further information processing.
this section cite: ['b29', 'b60', 'b8']

Section: Mixture-of-Experts Fusion.
Each attribute-aware feature H att i is first refined by a lightweight FFN, producing expert features H exp i . We mean-pool over the query dimension to obtain a compact descriptor hi ∈ R B×C for each expert. The four descriptors are concatenated and passed through a learnable projection W ∈ R C×4 to generate gating logits. Following previous work [108], a small Gaussian perturbation encourages exploration, that is:
λ = softmax(([ h1 ; h2 ; h3 ; h4 ] W) + σϵ), ϵ ∼ N (0, 1),(2)
where σ is a learnable scale. The final fused representation is H fuse = 4 i=1 λ i H exp i . The weights λ i adaptively emphasize whichever attribute cues are most informative for the current sample (e.g., motion cues at night, appearance cues in daylight) and make the model's prediction process more interpretable: large signals of status (δ s ) imply reliance on motion, whereas a high weight of relationto-viewer (δ v ) highlights egocentric relations. The injected noise prevents early collapse to a single expert and empirically improves robustness across lighting and speed variations.
this section cite: ['b107']

Section: Effective Multi-Attribute Fusion for Grounding
During training, every data sample yields one ground-truth box b and four attribute token maps {m i } 4  i=1 . The decoder, however, produces a single token-distribution logit mn ∈ R C for each query n, together with its box bn . To exploit every attribute without inflating the head, we treat the four attributes as co-located pseudo-objects during matching and later fuse their scores, giving dense, consistent supervision and precise grounding.
Training as Multi-Object Grounding. We treat this problem as in a multi-object setting: The target list is duplicated four times (one per attribute), i.e., {b i } 4  i=1 , and the Hungarian matching is applied between queries and the target bounding box. The cost C (n,i) for assigning query n to the target i is:
C (n,i) = β box ∥ bn -b i ∥ 1 + GIoU( bn , b i ) + β attr L attr mn , m i ,(3)
which combines a box regression term (L1 loss plus the Generalized Intersection over Union GIoU) with the attribute alignment loss L attr (cross-entropy on the soft token maps). The weights β box and β attr balance spatial accuracy against textual grounding. Since pseudo-targets share the same box but different token maps, this encourages the decoder to converge to a single spatial prediction with distinct textual alignments. The total loss sums matching costs over all query-target pairs.
Inference. At test time, each query outputs one box and its token logit mn . We build the four attribute maps for the caption as in Sec. 4.1, then score every query between target i by score (n,i) = ⟨softmax( mn ), softmax(m i )⟩, i.e., the dot-product between predictions and the probability distribution of positive tokens. The final prediction is the box with the highest score. This late fusion works because the box geometry is shared across attributes; what changes is the confidence level at which the query's language head fires on the respective token sets, allowing the model to rely on the attribute that carries the clearest signal for the current scene.
This multi-attribute fusion design lets us exploit all four attributes without enlarging the decoder, then merge their evidence at test time into a single, reliable score, improving grounding accuracy and interpretability while keeping the framework compact.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Settings
Baselines & Competitors. We benchmark EventRefer against three groups of methods. 1 Frame-Only: we retrain traditional visual grounding methods [41,37] on Talk2Event and report zero-shot results from the large-scale generalist models [67,58,16]. 2 Event-Only: as no event-based grounding method exists, we adapt leading event perception methods [30,92,72,118,83] by attaching a DETR Transformer and a grounding head. 3 Event-Frame Fusion: we re-implement leading event-frame fusion perception methods [27,8,111,63] under the same DETR Transformer and grounding head. We additionally built a simple fusion baseline "RVT+ResNet+Attention". All baselines receive the full referring expression but are supervised only with class-name positive tokens, following prior practice. This emphasizes the gains of our multi-attribute supervision in our approach.
this section cite: ['b40', 'b36', 'b66', 'b57', 'b15', 'b29', 'b91', 'b71', 'b117', 'b82', 'b26', 'b7', 'b110', 'b62']

Section: Implementation Details.
All models are built in PyTorch [71]. For the event-frame fusion model, the frames are encoded with a ResNet-101 [34] pre-trained on ImageNet [18]; multi-scale features are flattened and concatenated, each token having 256 channels. We train with AdamW [62] using learning rates of 1×10 -6 (frame backbone), 5×10 -6 (textual encoder) and 5×10 -5 (remaining layers). The DETR Transformer weights are initialized from BUTD-DETR [37], and the event backbone weights are from FlexEvent [63]. Due to space limits, see the appendix for more details.
Evaluation Metrics. Following practice, we report Top-1 Acc., i.e., the proportion of samples whose highest-scoring box overlaps the ground truth by at least the chosen IoU threshold. We use a stringent threshold IoU@0.95 to stress precise localization, and complement it with mean IoU over all predictions for a holistic boundary measure. Please refer to the appendix for more details.
this section cite: ['b70', 'b33', 'b17', 'b61', 'b36', 'b62']

Section: Comparative Study
Traditional Visual Grounding. We first compare the traditional frame-based grounding models [41,37] and more recent generalist models, i.e., OWL-ViT [67], GroundingDINO [58], and YOLO-World [16]. As shown in Tab. 2 (top), EventRefer achieves 55.47% mAcc and 85.76% mIoU, outperforming all baselines. Notably, we observe substantial improvements on small or dynamic objects such as pedestrians (+5.0%) and riders (+24.4%), demonstrating the ability to leverage attribute-level reasoning beyond simple appearance matching.
Grounding from Event Streams. In the event-only setting, we compare with state-of-the-art eventbased perception methods [30,92,72,118,83]. As shown in Tab. 2 (middle), despite their strong detection capabilities, these methods are not explicitly designed for language grounding. EventRefer, by contrast, achieves 31.96% mAcc and 76.46% mIoU, outperforming all event-based baselines. Additionally, we find that event-only performance is generally lower than frame-based models, which is expected since event streams lack rich texture and appearance details. However, the ability to capture motion dynamics and temporal changes brings advantages, especially in low-light or high-speed scenarios where frame-based models tend to struggle.
this section cite: ['b40', 'b36', 'b66', 'b57', 'b15', 'b29', 'b91', 'b71', 'b117', 'b82']

Section: Input & Query
Frame (GT Bbox) RVT DAGr EventRefer (Ours)
A gray-colored sedan parked on the left side of the road, in front of the viewer and followed by another gray-colored vehicle.
this section cite: []

Section: Car
A bicycle with a rider is on the right side of the road, ahead of a car to its left and a motorcyclist to its right, moving forward.
this section cite: []

Section: Bicycle
A blond-haired pedestrian in dark clothing is behind a white car and next to a cyclist, moving from left to right in the image.  .60 0.45 0.62 0.55 0.42 0.40 0.42 Car Ped Bus Rider Truck Bike Motor Expert (𝛿 𝐨 ) Expert (𝛿 𝐚 ) Expert (𝛿 𝐬 ) Expert (𝛿 𝐯 ) Event-Only 1.00 0.75 0.50 0.25 0.00 (a) Class-Wise Activations (Event Only) 0.38 0.32 0.55 0.68 0.60 0.72 0.65 Car Ped Bus Rider Truck Bike Motor Event-Frame Fusion 1.00 0.75 0.50 0.25 0.00 Expert (𝛿 𝐨 ) Expert (𝛿 𝐚 ) Expert (𝛿 𝐬 ) Expert (𝛿 𝐯 )
(b) Class-Wise Activations (Event-Frame Fusion)
Figure 5: Class-wise attribute expert activations. We visualize the proportion of each attribute experts in MoEE, under two grounding settings. The top-1 proportion of each class is highlighted.
this section cite: []

Section: Fusion between Event and Frame.
Combining event streams with RGB frames provides complementary benefits, leveraging both high-temporal motion cues and rich appearance information.
As shown in Tab. 2 (bottom), EventRefer surpasses all existing fusion baselines such as DAGr [27] and FlexEvent [63]. Notably, we observe consistent improvements across all object categories, particularly rider (+11.7%), bicycle (+4.8%), and truck (+3.1%), indicating that attribute-aware fusion effectively balances appearance, motion cues, and object relationships for robust grounding.
Qualitative Assessments. Fig. 4 shows qualitative examples comparing our approach with two strong baselines, RVT [30] and DAGr [27], under the event-frame fusion setting. We observe that previous methods often fail to precisely align the bounding box with the described object due to their limited grounding capability. In contrast, EventRefer produces tighter and more semantically accurate predictions, successfully leveraging attribute-aware reasoning to handle complex descriptions. Due to space limits, please refer to our appendix for additional analyses and visualizations.
this section cite: ['b26', 'b62', 'b29', 'b26']

Section: Ablation Study
We conduct detailed ablations to analyze the contribution of each design in our framework, using the event-only setting throughout. The results are from the validation set of our Talk2Event dataset.
this section cite: []

Section: Component Analysis.
We first evaluate the three key components in EventRefer: positive word matching (PWM), multi-attribute fusion (MAF), and the mixture of event experts (MoEE). As shown in Tab. 3, adding PWM alone improves mAcc from 22.07% to 26.38% by linking token-level supervision with attribute spans. MAF alone achieves 27.01% by enabling independent reasoning over different attributes. Combining both pushes performance to 29.66%, confirming their complementarity. Finally, adding MoEE achieves the best 31.96%, demonstrating the benefit of adaptive expert fusion that dynamically weighs attribute importance across varying scenes.
this section cite: []

Section: Effects of Different Attributes.
Next, we investigate the individual impact of each attribute on grounding performance. As shown in Tab. 4, using only appearance (δ a ) achieves a strong baseline of 27.98% mAcc, indicating the importance of visual descriptions such as class shape and object    boundary. Using status (δ s ) yields an improvement of 28.90%, showing that motion cues provide complementary benefits, especially in dynamic scenes. Interestingly, viewer-centric (δ v ) and objectcentric (δ o ) relations alone yield slightly lower performance, but their combined contribution with appearance and status results in the best performance of 31.96%. This highlights the value of modeling all four attributes jointly, as they capture different aspects of the spatiotemporal context.
this section cite: []

Section: Fusion Strategies.
We compare different strategies for fusing the four attribute-aware features.
As shown in Tab. 5, simple additive fusion achieves 28.39% mAcc, while concatenation yields a slightly lower 27.50%. Attention-based fusion improves performance to 29.66%, showing the benefit of learning adaptive weights. However, our proposed MoEE achieves the highest 31.96% mAcc, significantly outperforming all other strategies. This result highlights MoEE's ability to not only fuse attribute features effectively but also to adaptively emphasize the most informative attributes based on scene dynamics, object properties, and modality signals.
this section cite: []

Section: Class-Wise Activations.
We further analyze the average attribute activations for each object class. In the event-only setting (Fig. 5a), small dynamic classes such as Rider and Bike rely more on status cues, while larger static objects like Bus and Truck favor appearance and viewer relations. In the event-frame fusion setting (Fig. 5b), appearance cues become the most dominant across all classes, yet status and relational cues remain important for highly dynamic or interaction-heavy objects like Pedestrian and Rider. These findings demonstrate that EventRefer not only adapts to input modality but also to object category, promoting interpretable and task-specific grounding behavior.
this section cite: []

Section: Activations vs. Event Response Strength.
To understand how event density affects attribute reliance, we first quantify the event response strength by counting the total number of events within a fixed spatial-temporal window. Specifically, we compute the number of activated pixels in the event voxel grid (e.g., 2 × T × H × W ) for each sample. Based on this metric, we group samples into seven levels, from low to high response strength, and visualize the top-1 and top-2 expert activations. In the event-only setting (Fig. 6a), viewer-centric relations (δ v ) and appearance (δ a ) dominate low-response scenes, reflecting reliance on static context when little motion is present. As event density increases, status (δ s ) and relational cues (δ o ) become more important, capturing the dynamics of moving objects and their interactions. In the event-frame fusion setting (Fig. 6b), appearance remains dominant, while status and relational cues gain more influence in highly dynamic scenes. This highlights the adaptive behavior of MoEE in leveraging the most informative attributes based on input conditions.
this section cite: []

Section: Conclusion
We presented Talk2Event, the first large-scale benchmark for language-driven object grounding in dynamic event streams. Built on real-world driving data, we introduce linguistically rich, attribute-aware annotations that capture appearance, motion, and relational context -key factors often overlooked in traditional grounding benchmarks. To tackle this, we proposed EventRefer, an attribute-aware grounding framework that adaptively fuses attribute-specific cues. Extensive experiments demonstrate that our approach outperforms strong baselines. We hope this work will inspire future research at the intersection of event-based perception, visual grounding, and robust multimodal scene understanding.
this section cite: []

Section: References
Ref_id:b0 Title: Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes Year: (2020)
Ref_id:b1 Title: Ev-segnet: Semantic segmentation for event-based cameras Year: (2019)
Ref_id:b2 Title: End-to-end davis driving dataset Year: (2017)
Ref_id:b3 Title: Halsie: Hybrid approach to learning segmentation by simultaneously exploiting image and event modalities Year: (2024)
Ref_id:b4 Title: Pedro: An event-based dataset for person detection in robotics Year: (2023)
Ref_id:b5 Title: A 240× 180 130 db 3 µs latency global shutter spatiotemporal vision sensor Year: (2014)
Ref_id:b6 Title: Muses: The multi-sensor semantic perception dataset for driving under uncertainty Year: (2024)
Ref_id:b7 Title: Embracing events and frames with hierarchical feature refinement network for object detection Year: (2024)
Ref_id:b8 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b9 Title: Low-latency event-based visual odometry Year: (2014)
Ref_id:b10 Title: Recent event camera innovations: A survey Year: (2024)
Ref_id:b11 Title: M3ed: Multi-robot, multi-sensor, multienvironment event dataset Year: (2023)
Ref_id:b12 Title: Scanrefer: 3d object localization in rgb-d scans using natural language Year: (2020)
Ref_id:b13 Title: Ecmd: An event-centric multisensory driving dataset for slam Year: (2024)
Ref_id:b14 Title: Advancing visual grounding with scene knowledge: Benchmark and method Year: (2023)
Ref_id:b15 Title: Yolo-world: Real-time open-vocabulary object detection Year: (2024)
Ref_id:b16 Title: Label-free event-based object recognition via joint learning with image reconstruction from events Year: (2023)
Ref_id:b17 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b18 Title: Evreal: Towards a comprehensive benchmark and analysis suite for event-based video reconstruction Year: (2023)
Ref_id:b19 Title: Hue dataset: High-resolution event and frame sequences for low-light vision Year: (2024)
Ref_id:b20 Title: Free-form description guided 3d visual graph network for object grounding in point cloud Year: (2021)
Ref_id:b21 Title: 10 a 1280× 720 backilluminated stacked temporal contrast event-based vision sensor with 4.86 µm pixels, 1.066 geps readout, programmable event-rate controller and compressive data-formatting pipeline Year: (2020)
Ref_id:b22 Title: Event-based vision: A survey Year: (2022)
Ref_id:b23 Title: Video to events: Recycling video datasets for event cameras Year: (2020)
Ref_id:b24 Title: End-to-end learning of representations for asynchronous event-based data Year: (2019)
Ref_id:b25 Title: Pushing the limits of asynchronous graph-based object detection with event cameras Year: (2022)
Ref_id:b26 Title: Low-latency automotive vision with event cameras Year: (2024)
Ref_id:b27 Title: Dsec: A stereo event camera dataset for driving scenarios Year: (2021)
Ref_id:b28 Title: E-raft: Dense optical flow from event cameras Year: (2021)
Ref_id:b29 Title: Recurrent vision transformers for object detection with event cameras Year: (2023)
Ref_id:b30 Title: Hierarchical neural memory network for low latency event processing Year: (2023)
Ref_id:b31 Title: Evsegsnn: Neuromorphic semantic segmentation for event data Year: (2024)
Ref_id:b32 Title: Transrefer3d: Entity-and-relation aware transformer for fine-grained 3d visual grounding Year: (2021)
Ref_id:b33 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b34 Title: Event-aided direct sparse odometry Year: (2022)
Ref_id:b35 Title: Learning monocular dense depth from events Year: (2020)
Ref_id:b36 Title: Bottom up top down detection transformers for language grounding in images and point clouds Year: (2022)
Ref_id:b37 Title: Towards robust event-based networks for nighttime via unpaired day-to-night event translation Year: (2024)
Ref_id:b38 Title: Hpl-ess: Hybrid pseudo-labeling for unsupervised event-based semantic segmentation Year: (2024)
Ref_id:b39 Title: Efficient learning of event-based dense representation using hierarchical memories with adaptive update Year: (2024)
Ref_id:b40 Title: Mdetr-modulated detection for end-to-end multi-modal understanding Year: (2021)
Ref_id:b41 Title: N-imagenet: Towards robust, fine-grained object recognition with event cameras Year: (2021)
Ref_id:b42 Title: Text2pos: Text-to-point-cloud cross-modal localization Year: (2022)
Ref_id:b43 Title: Openess: Eventbased semantic scene understanding with open vocabularies Year: (2024)
Ref_id:b44 Title: Eventfly: Event camera perception from ground to the sky Year: (2025)
Ref_id:b45 Title: Multi-modal data-efficient 3d scene understanding for autonomous driving Year: (2025)
Ref_id:b46 Title: Geochat: Grounded large vision-language model for remote sensing Year: (2024)
Ref_id:b47 Title: Sodformer: Streaming object detection with transformer using events and frames Year: (2023)
Ref_id:b48 Title: Referring transformer: A one-step approach to multi-task visual grounding Year: (2021)
Ref_id:b49 Title: Eventvl: Understand event streams via multimodal large language model Year: (2025)
Ref_id:b50 Title: Seeground: See and ground for zeroshot open-vocabulary 3d visual grounding Year: (2025)
Ref_id:b51 Title: Wildrefer: 3d object localization in large-scale dynamic scenes with multi-modal visual data and natural language Year: (2024)
Ref_id:b52 Title: Exploring optical-flow-guided motion and detectionbased appearance for temporal sentence grounding Year: (2023)
Ref_id:b53 Title: A survey on text-guided 3d visual grounding: elements, recent advances, and future directions Year: (2024)
Ref_id:b54 Title: Context-aware biaffine localizing network for temporal sentence grounding Year: (2021)
Ref_id:b55 Title: Refer-it-in-rgbd: A bottom-up approach for 3d visual grounding in rgbd images Year: (2021)
Ref_id:b56 Title: Eventgpt: Event stream understanding with multimodal large language models Year: (2024)
Ref_id:b57 Title: Grounding dino: Marrying dino with grounded pre-training for open-set object detection Year: (2024)
Ref_id:b58 Title: Cross-task knowledge transfer for semi-supervised joint 3d grounding and captioning Year: (2024)
Ref_id:b59 Title: Joint top-down and bottom-up frameworks for 3d visual grounding Year: (2025)
Ref_id:b60 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b61 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b62 Title: Flexevent: Towards flexible event-frame object detection at varying operational frequencies Year: (2024)
Ref_id:b63 Title: 3d-sps: Single-stage 3d visual grounding via referred point progressive selection Year: (2022)
Ref_id:b64 Title: Event-based vision meets deep learning on steering prediction for self-driving cars Year: (2018)
Ref_id:b65 Title: Scaling open-vocabulary object detection Year: (2023)
Ref_id:b66 Title: Simple open-vocabulary object detection with vision transformers Year: (2022)
Ref_id:b67 Title: Cityrefer: geography-aware 3d visual grounding dataset on city-scale point cloud data Year: (2023)
Ref_id:b68 Title: Event-intensity stereo: Estimating depth by the best of both worlds Year: (2021)
Ref_id:b69 Title: Continuous-time visualinertial odometry for event cameras Year: (2018)
Ref_id:b70 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b71 Title: Scene adaptive sparse transformer for event-based object detection Year: (2024)
Ref_id:b72 Title: Learning to detect objects with a 1 megapixel event camera Year: (2020)
Ref_id:b73 Title: Sleep activity recognition with event cameras Year: (2024)
Ref_id:b74 Title: A qvga 143 db dynamic range frame-free pwm image sensor with lossless pixel-level video compression and time-domain cds Year: (2010)
Ref_id:b75 Title: Evo: A geometric approach to event-based 6-dof parallel tracking and mapping in real time Year: (2016)
Ref_id:b76 Title: High speed and high dynamic range video with an event camera Year: (2019)
Ref_id:b77 Title: Zero-shot grounding of objects from natural language queries Year: (2019)
Ref_id:b78 Title: Jooyeon Woo, et al. 4.1 a 640× 480 dynamic vision sensor with a 9µm pixel and 300meps address-event representation Year: (2017)
Ref_id:b79 Title: Neuromorphic stereo vision: A survey of bio-inspired sensors and algorithms Year: (2019)
Ref_id:b80 Title: Visual grounding in remote sensing images Year: (2022)
Ref_id:b81 Title: Ess: Learning event-based semantic segmentation from still images Year: (2022)
Ref_id:b82 Title: Evrt-detr: Latent space adaptation of image detectors for event-based vision Year: (2024)
Ref_id:b83 Title: Dual transfer learning for event-based end-task prediction via pluggable event to image translation Year: (2021)
Ref_id:b84 Title: Evdistill: Asynchronous events to end-task learning via bidirectional reconstruction-guided cross-modal knowledge distillation Year: (2021)
Ref_id:b85 Title: Learning two-branch neural networks for imagetext matching tasks Year: (2018)
Ref_id:b86 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b87 Title: Neighbourhood watch: Referring expression comprehension via language-guided graph attention networks Year: (2019)
Ref_id:b88 Title: Event stream-based visual object tracking: A high-resolution benchmark dataset and a novel baseline Year: (2024)
Ref_id:b89 Title: Referring multi-object tracking Year: (2023)
Ref_id:b90 Title: Towards open vocabulary learning: A survey Year: (2024)
Ref_id:b91 Title: Leod: Label-efficient object detection for event cameras Year: (2024)
Ref_id:b92 Title: Towards visual grounding: A survey Year: (2024)
Ref_id:b93 Title: Cross-modal learning for event-based semantic segmentation via attention soft alignment Year: (2024)
Ref_id:b94 Title: Are vlms ready for autonomous driving? an empirical study from the reliability, data, and metric perspectives Year: (2025)
Ref_id:b95 Title: Tubedetr: Spatio-temporal video grounding with transformers Year: (2022)
Ref_id:b96 Title: Cross-modal relationship inference for grounding referring expressions Year: (2019)
Ref_id:b97 Title: Event camera data pre-training Year: (2023)
Ref_id:b98 Title: A fast and accurate one-stage approach to visual grounding Year: (2019)
Ref_id:b99 Title: Sat: 2d semantics assisted training for 3d visual grounding Year: (2021)
Ref_id:b100 Title: Modeling context in referring expressions Year: (2016)
Ref_id:b101 Title: Visual programming for zero-shot open-vocabulary 3d visual grounding Year: (2024)
Ref_id:b102 Title: Instancerefer: Cooperative holistic understanding for visual grounding on point clouds through instance multi-level contextual referring Year: (2021)
Ref_id:b103 Title: Rsvg: Exploring data and models for visual grounding on remote sensing data Year: (2023)
Ref_id:b104 Title: Mono3dvg: 3d visual grounding in monocular images Year: (2024)
Ref_id:b105 Title: 3dvg-transformer: Relation modeling for visual grounding on point clouds Year: (2021)
Ref_id:b106 Title: Deep learning for event-based vision: A comprehensive survey and benchmarks Year: (2023)
Ref_id:b107 Title: Convolution meets lora: Parameter efficient finetuning for segment anything model Year: (2024)
Ref_id:b108 Title: Exact: Language-guided conceptual reasoning and uncertainty estimation for event-based action recognition and more Year: (2024)
Ref_id:b109 Title: A unified large vision-language model. for remote sensing visual grounding Year: (2024)
Ref_id:b110 Title: Rgb-event fusion for moving object detection in autonomous driving Year: (2023)
Ref_id:b111 Title: The multivehicle stereo event camera dataset: An event camera dataset for 3d perception Year: (2018)
Ref_id:b112 Title: Ev-flownet: Self-supervised optical flow estimation for event-based cameras Year: (2018)
Ref_id:b113 Title: Unsupervised event-based learning of optical flow, depth, and egomotion Year: (2019)
Ref_id:b114 Title: Cear: Comprehensive event camera dataset for rapid perception of agile quadruped robots Year: (2024)
Ref_id:b115 Title: Seeing behind dynamic occlusions with event cameras Year: (2023)
Ref_id:b116 Title: From chaos comes order: Ordering event representations for object recognition and detection Year: (2023)
Ref_id:b117 Title: State space models for event cameras Year: (2024)
