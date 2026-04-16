Title: OpenWorldSAM: Extending SAM2 for Universal Image Segmentation with Language Prompts
Abstract: The ability to segment objects based on open-ended language prompts remains a critical challenge, requiring models to ground textual semantics into precise spatial masks while handling diverse and unseen categories. We present OpenWorldSAM, a framework that extends the prompt-driven Segment Anything Model v2 (SAM2) to open-vocabulary scenarios by integrating multi-modal embeddings extracted from a lightweight vision-language model (VLM). Our approach is guided by four key principles: i) Unified prompting: OpenWorldSAM supports a diverse range of prompts, including category-level and sentence-level language descriptions, providing a flexible interface for various segmentation tasks. ii) Efficiency: By freezing the pre-trained components of SAM2 and the VLM, we train only 4.5 million parameters on the COCO-stuff dataset, achieving remarkable resource efficiency. iii) Instance Awareness: We enhance the model's spatial understanding through positional tie-breaker embeddings and cross-attention layers, enabling effective segmentation of multiple instances. iv) Generalization: OpenWorldSAM exhibits strong zero-shot capabilities, generalizing well on unseen categories and an open vocabulary of concepts without additional training. Extensive experiments demonstrate that OpenWorldSAM achieves state-of-the-art performance in open-vocabulary semantic, instance, and panoptic segmentation across multiple benchmarks. Code is available at https://github.com/GinnyXiao/OpenWorldSAM.

Section: Introduction
Image segmentation has long been constrained to closed-vocabulary settings, where models can only recognize objects from a predefined taxonomy [1,2,3,4,5,6,7,8]. However, real-world applications, e.g., Embodied AI [9,10], demand systems that can understand open-ended language descriptions (from single nouns like "pedestrian" to rich referring expressions such as "the man in a red shirt") and segment novel objects unseen during training. This open-vocabulary segmentation problem poses two core challenges: (1) Semantic grounding -mapping free-form text to visual entities, and (2) Instance awareness -distinguishing multiple objects that match the same description.
Detection-centric methods [11,12] relied on two-stage pipelines, first detecting class-agnostic mask proposals then classifying them with vision-language models (VLMs), e.g., CLIP [13] and ALIGN [14]. While effective, such approaches struggle with complex queries and specialize exclusively in semantic segmentation, lacking versatility. Recent generalist models [15,16] explore unified architectures that jointly handle vision and language, allowing a single model to perform detection, segmentation, and grounding tasks. These generalist models demonstrate impressive flexibility, but they typically involve resource-intensive pre-training. The emergence of promptable segmentation models like the Segment Anything Model (SAM) [17,18] offered new possibilities -it introduced
Interactive (Point, Box) Visual Prompts Open-Vocabulary Category-level Prompts Instance Panoptic Semantic Sentence-Level Prompts
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17']

Section: Referring Expression

this section cite: []

Section: OpenWorld SAM SAM2
Figure 1: Overview of the proposed framework. The green region highlights the SAM v2 baseline, supporting visual prompts (e.g., boxes, points) for interactive segmentation. Our OpenWorldSAM extension integrates open-vocabulary language understanding, enabling both category-level segmentation across semantic, instance, panoptic tasks and referring expression segmentation.
a paradigm shift by allowing users to segment arbitrary objects using simple visual prompts (e.g., points, boxes). Trained on an extensive dataset, these models exhibit remarkable generalization and interactive capabilities. However, they inherently lack semantic understanding. Subsequent attempts to combine SAM with large language models (LLMs) [19,20,21] achieved language awareness but at prohibitive computational costs, imposing overwhelming overhead.
We posit that an ideal open-vocabulary segmenter should: (i) Natively support textual prompts without cascaded classification components, (ii) Preserve the knowledge of the vision foundation models like SAM without adding large overhead, and (iii) Segment multiple possible instances that could correspond to a single query. To this end, we propose OpenWorldSAM, an open-vocabulary extension to the SAM v2 (SAM2) architecture that satisfies these requirements. OpenWorldSAM injects language understanding while retaining SAM2's core strengths through a lightweight language adapter (≈4.5M trainable parameters), unifing category-level instance, semantic, and panoptic segmentation, and sentence-level referring expression segmentation (Figure 1).
Specifically, we feed the image and descriptive text input into a frozen multi-modal encoder and obtain fused semantic representations. These serve as prompts to SAM2' mask decoder that produces masks for any described object or region. We introduce a positional tie-breaker mechanism to resolve ambiguities when a text query could apply to multiple regions, allowing the model to perform multi-instance segmentation. Furthermore, our adapter employs a soft prompting technique that uses cross-attention between textual queries and image features, sharpening localization by allowing semantic contexts to focus toward relevant image areas. By combining these design innovations, OpenWorldSAM can accurately identify and segment arbitrary objects described by text, all while using only frozen pre-trained encoders and a tiny trainable adaptation module.
VOC-20 (mIoU) ADE-857 (mIoU) ADE-150 (mIoU) ADE-150 (PQ) SCAN-40 (mIoU) SCAN-20 (PQ) SCAN-20 (mIoU) SUN-37 (mIoU) PC-459 (mIoU) PC-59 (mIoU) In summary, OpenWorldSAM represents a new paradigm of "segment anything in the open world". It inherits SAM's interactiveness while being guided by flexible language prompts. Our contributions include:
1. We introduce OpenWorldSAM, a unified interface that supports various open-vocabulary segmentation tasks. We propose an efficient language adapter with tie-breaker and cross-attention soft prompting, improving multi-object localization. 2. OpenWorldSAM achieves state-of-the-art zeroshot performance across six benchmarks (Figure 2), setting a new standard for open-vocabulary segmentation (e.g., 60.4 mIoU on ADE20K [22]). OpenWorldSAM also acheives strong performance in referring expression segmentation (74.0 cIoU on RefCOCOg [23]) with substantially fewer resources compared to recent models. 3. Our work demonstrates that lightweight architectural interventions can unlock zero-shot segmentation capabilities rivaling specialized models while preserving SAM2's efficiency and interactivity. 2 Related Work
this section cite: ['b18', 'b19', 'b20', 'b21', 'b22']

Section: Open-vocabulary segmentation.
Recent advances in open-vocabulary segmentation have leveraged vision-language models (VLMs) [13,14] to overcome the constraints of traditional closed-set segmentation models. Early approaches like LSeg [24], RegionCLIP [25] and OWL-ViT [26] established a baseline by introducing a contrastive learning framework to align image embeddings with CLIP-based text embeddings for zero-shot detection/segmentation. Subsequent methods [11,27] scaled effectively by using weak supervision of large-scale images with captions (up to millions of regions) or text-only signals, enabling more flexible and broader semantic coverage. Two-stage approaches like MaskCLIP [28] and OVSeg [12] further refined this paradigm by generating mask proposals using MaskFormer [29] followed by CLIP-based classification, notably boosting accuracy through mask-adapted fine-tuning. Another line of works formulated this task as a visual grounding problem and established region-text fusion [30,31,32,33]. More recently, unified architectures such as ODISE [34], X-Decoder [15], SEEM [35], OpenSeeD [36], HIPIE [37], Semantic-SAM [38] and APE [16] have integrated multiple segmentation tasks into a single framework, showing significant progress towards general-purpose models, but they typically required resource intensive pre-training.
Extending SAM for text-prompted segmentation. The Segment Anything Model (SAM) [17,18] achieved a breakthrough in promptable segmentation by training on 1 billion masks, enabling it to generate high-quality masks for visual prompts. A flurry of recent works have explored infusing SAM with semantic or language understanding to move beyond its original prompt types. Grounded-SAM [39] is a pioneering effort that combines an open-vocabulary detector GroundingDINO [31] to generate bounding boxes from a text query, then feeds those boxes as prompts into SAM. Fast-SAM [40] matches CLIP embeddings with regions of interest. LLM-centric works [41,20,19,21] attempt to map large LLMs or VLMs language embeddings into SAM or SAM-like decoder's prompt latent space to enable referring expression segmentation. Among these, LISA [19] pioneered "maskas-text-embedding" approach but was limited to single-object queries. LISA++ [42] introduced instance awareness through additional instruction-tuning data, though it requires LLMs to explicitly enumerate objects-a computationally expensive process. EVF-SAM [43] recently demonstrated a lightweight alternative, integrating SAM with a multi-modal BEiT-3 encoder [44] (0.7B parameters). While achieving state-of-the-art referring segmentation accuracy with minimal parameters, it remains constrained to single-object queries. Inspired by the success of EVF-SAM, we enhance SAM further into the domain of open-vocabulary segmentation, where the goal is to segment and label all objects ("things" and "stuff") in the scene with open-set categories.
this section cite: ['b12', 'b13', 'b23', 'b24', 'b25', 'b10', 'b26', 'b27', 'b11', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b14', 'b34', 'b35', 'b36', 'b37', 'b15', 'b16', 'b17', 'b38', 'b30', 'b39', 'b40', 'b19', 'b18', 'b20', 'b18', 'b41', 'b42', 'b43']

Section: Methodology
Motivation and key challenges. A fundamental limitation of SAM-like architectures is their inability to resolve multi-instance ambiguity from a single prompt. While visual prompts (e.g., points) may occasionally lack granularity specificity-for instance, a click on a backpack could imply segmentation of either the backpack or the entire person (Figure 3a)-they inherently localize to a single spatial region. Language prompts, however, introduce a distinct challenge: a text query like "zebra" may correspond to multiple spatially disjoint objects (Figure 3b), with no prior knowledge of instance counts. Prior attempts to add language capabilities either rely on segmentation-thenclassification pipelines (losing end-to-end training) or require costly region-level text grounding
Hungarian Matching … … Norm 𝑲𝑽 3× … … … … … Self-Attention 𝑸 Cross Attention Norm MLP … … (a) Inputs & Encoder Outputs (b) Model Pipeline (c) Soft Prompting -Zooming in … 𝒕 ! , 𝒕 " , … , 𝒕 # 🔥 𝒖 $%&'(() 𝒖 &*+, … … + + = = … … … … … … … 𝑭 -.×-.
this section cite: []

Section: SAM2 Image Encoder
Input: {𝐼} Output: {𝑭} Input: {𝐼, 𝑇}
this section cite: []

Section: Multi-Modal Encoder
{𝒑 0'1$ } MLP Projector Output: {𝒖} "giraffe"/ "rock" Text Prompts: … … 🔥 𝒒 {!:4} 𝒒' {!:4} SAM2 Prompt Encoder SAM2 Mask Decoder   during pre-training. Our key insight addresses this gap: SAM2's mask decoder can inherently segment multiple instances if equipped with diverse positional guidance, i.e., learned cues that disentangle identical semantic queries into spatially distinct segmentation targets.
Architecture overview. Figure 4 depicts our framework which comprises: (i) a hierarchical SAM2 image encoder that extracts image features, (ii) a multi-modal vision-language encoder that jointly ingests the image and text prompt, (iii) a lightweight MLP projector, (iv) learnable positional tiebreakers for multi-instance queries, (v) a soft prompting Transformer block that aligns text-image features with SAM2's image features, and (vi) the SAM2 mask decoder producing final masks. Only a small language adapter with components (iii-v) is trained; all other backbones remain frozen.
Multi-modal encoder. We leverage BEiT-3 [44] to encode the input description into a semantic embedding. Given an image I and a text prompt T (e.g., a category name or a referring expression), we feed both modalities into BEiT-3's encoder to obtain joint visual-text embeddings. Concretely, tokens of T and patch embeddings of a downsampled I are concatenated and processed by BEiT-3, yielding a set of feature vectors {f [CLS] , f 1 , . . . }. We take the classification token f [CLS] as a compact summary denoted as p lang of the prompt conditioned on the image content.
We adopt BEiT-3 because its early-fusion training on image-text pairs equips it with rich, bidirectional semantics-crucial for reasoning about unseen classes. Compared with CLIP-style contrastive imagetext matching using only the features from the last encoder layers, BEiT-3 exposes finer cross-modal interactions. By embedding the text while it sees the image, the encoder already localizes the concept loosely (e.g., "giraffe" vs. "rock" in Figure 4) before any downstream segmentation, preventing the mask decoder from learning semantics from scratch.
Prompt projection. BEiT-3 emits 1,024-D tokens, whereas SAM's prompt channels are 256-D. A two-layer MLP acts as a projector that (i) preserves the coarse semantics of p lang ∈ R d1024 and (ii) learns to highlight dimensions that are most useful for mask prediction: u = MLP(p lang ) ∈ R d256 .
this section cite: ['b43']

Section: Positional tie-breaker and multi-instance queries generation. The projected visual-text embedding
u captures what to segment but lacks awareness of how many instances exist and where they are. To enable multi-instance segmentation, we propose K learnable positional tie-breaker vectors {t 1 , . . . , t K } ⊂ R d256 that perturb u into K distinct queries:
q i = u + t i , i = 1, . . . , K.(1)
These perturbations serve two purposes: 1) Positional disambiguation: Each t i nudges the query towards different spatial regions (Figure 3b), mimicking how human annotators might click different points to segment each zebra. 2) Instance diversity: The tie-breakers are optimized during training to maximize coverage of distinct instances, preventing query collapse. Conceptually these queries play the role of the "object queries" in DETR [45]. Crucially, they impose segmentation distinction for the same language semantics, making positional tie-breaking a novel and key feature for OpenWorldSAM.
In practice K = 20 covers >99% images in COCO [46]; for larger scenes K can be increased trivially.
this section cite: ['b44', 'b45']

Section: Soft-prompting via cross-attention.
The perturbed queries {q i } interact with SAM2's image features through a 3-layer Transformer [47] in Figure 4c, which alternates self-attention (queries talk to each other, promoting diversity) and cross-attention (queries look at image features). Each language-aware query is refined on-the-fly by cross-attention with the frozen SAM2 features. SAM2's image encoder follows a hierarchical vision Transformer ("Hiera" [48,49]) that outputs three features {F 256×256 , F 128×128 , F 64×64 } with 256 2 , 128 2 , and 64 2 spatial resolutions, respectively. We operate on the level-3 features with 64 2 resolution as they optimally balance precision for retainaing boundary details and computational efficiency (16× cheaper than full-resolution attention). They are also used by SAM2 for mask decoding by default [18]. The soft prompting Transformer computes q ′ i = CrossAttn(q i , F 64×64 ), i = 1, . . . , K, whose key/value inputs are the flattened level-3 features F 64×64 ∈ R 4096×256 . This step grounds the language-aware queries in SAM2's high-resolution visual features, resolving ambiguities (e.g., distinguishing adjacent zebras by stripe patterns).
Mask decoding and class assignment. The refined queries {q ′ i } are input to SAM2's mask decoder alongside level-3 image features. We inject the queries as the prompt tokens in place of, e.g., point or box prompts in the original SAM2's prompt encoder to obtain prompt embeddings. The prompt embeddings are then passed to the mask decoder which outputs K masks and corresponding confidence scores. We assign each mask the original text prompt T as its class label, since the generation is fully conditioned on T and thus inherits the semantic identity.
Training. All heavy visual (Hiera) and vision-language (BEiT-3) encoders are kept frozen to preserve their pre-trained knowledge and avoid costly retraining. Only the MLP projector, tie-breakers, and the soft prompting Transformer are learnable. For each training sample and prompt, we match the K predicted masks to the ground-truth masks of class T via Hungarian matching [45], then apply a focal loss, encouraging precise segmentation of all instances described by the prompt. The tie-breakers t i ∈ R d256 are implemented as learnable parameters randomly initialized from a normal distribution. During training, the Hungarian matching loss naturally encourages each t i to specialize in different spatial regions. Notably, this mechanism requires no explicit supervision about instance counts.
Inference. From the predicted K masks, we derive results for three segmentation tasks: semantic, instance, and panoptic. For semantic segmentation, we merge masks sharing the same class label, weighted by their confidence scores. For instance segmentation, we apply confidence-score filtering to remove masks below a certain threshold, followed by non-maximum suppression (NMS) to eliminate highly overlapping masks and retain distinct object instances. Similarly, for panoptic segmentation, we perform confidence-based filtering and NMS, ensuring each pixel is uniquely assigned to either a "thing" (instance) or "stuff" (semantic) label.
Optionally, we perform a two-stage inference. In this setup, masks obtained from the first inference stage are used as visual prompts fed back into SAM2's mask decoder, which refines mask contours. Qualitatively, two-stage inference improves the precision of mask boundaries for correct predictions (Appendix B). However, quantitative analysis (Table 1) reveals that the second inference stage provides minimal improvements in segmentation metrics, suggesting it mainly enhances visual quality rather than overall accuracy.
this section cite: ['b46', 'b47', 'b48', 'b17', 'b44']

Section: Experiments
Datasets and metrics. We train OpenWorldSAM on the COCO2017-Stuff [46] dataset with panoptic annotations following X-Decoder [15]. The training set contains 104k images. We evaluate the model in a zero-shot setting on eight segmentation tasks across five diverse datasets: ADE20K-150/857 [22], PASCAL VOC-20 [50], PASCAL Context-59/459 [51], ScanNet-20/40 [52], and SUN-RGBD-37 [53]. Evaluation metrics include panoptic quality (PQ), mean average precision (mAP), and mean intersection-over-union (mIoU), corresponding to panoptic, instance, and semantic segmentation tasks, respectively. For referring segmentation, we finetune on RefCOCOg UMD training split. Following prior works, we report the cumulative intersection over the cumulative union (cIoU) metric on the RefCOCOg UMD validation split.
Implementation. We implement our model in PyTorch. We initialize the visual model with the weights of SAM2-Hiera-Large [18] and the VLM encoder with the weights of EVF-SAM BEIT-3-Large [43]. It is trained for 25 epochs on COCO-Stuff using the AdamW optimizer with a learning rate of 1e-4, batch size 8, on a single NVIDIA A100 GPU. Image resolution is set to 1024 for SAM2 and 224 for BEiT-3. Number of postional tie-breaks is set to 20 for COCO dataset. Our implementation details can be found in Appendix A.
this section cite: ['b45', 'b14', 'b21', 'b49', 'b50', 'b51', 'b52', 'b17', 'b42']

Section: Open-Vocabulary Segmentation Evaluation Protocols and Challenges
Ambiguity of open vocabulary evaluation. Most prior open-vocabulary segmentation methods-including X-Decoder [15], OVSeg [12], and MaskCLIP [28]-adopt a Global-Matching protocol: for each predicted mask, a model matches it against the entire dataset vocabulary using precomputed text embeddings and selects the best-aligned class. However, this strategy can be problematic when applied to datasets like ADE20K, which contain hundreds of fine-grained and overlapping labels. As observed in OVSeg [12], this leads to semantically reasonable predictions being marked incorrect under exact label matching: "The ground-truth category is 'building' while our model predicts 'skyscraper'. " This ambiguity stems from the inherent subjectivity of language: synonymous or closely related concepts may be indistinguishable in a visual context, yet only one is accepted by the ground truth. We observe similar issues in our own qualitative analysis. As shown in Figure 5, X-Decoder predictions on ADE20K-857 often produce valid but non-canonical labels (e.g., 'road' instead of 'runway', or 'screen' instead of 'arcade machine'), resulting in unfair penalization.
this section cite: ['b14', 'b11', 'b27', 'b11']

Section: Oracle-Prompts evaluation.
To address this, we introduce an alternative evaluation strategy: Oracle Prompts-during evaluation, we explicitly provide the ground-truth class names as prompts. This mimics the intended use case of prompt-based models like SAM, which are inherently interactive and conditioned on user input. Under this protocol, the model does not have to resolve linguistic ambiguity across the full label space; it segments what the user asks for. We report results under both settings: Table 1 shows baseline performance using the global matching protocol, consistent with prior works. Table 2 revisits X-Decoder under the oracle-prompt protocol for a more equitable comparison to OpenWorldSAM, which by design is evaluated under oracle prompts. We believe this approach provides a more fair assessment of SAM-style models in open-vocabulary segmentation.
this section cite: []

Section: Open-Vocabulary Segmentation Performance Analysis
Zero-shot open-vocabulary transfer. OpenWorldSAM generalizes out-of-the-box to a broad set of segmentation tasks without any weight adaptation. As shown in Table 1, it achieves state-ofthe-art performance across almost all datasets and evaluation metrics. Its performance consistently surpasses strong baselines such as X-Decoder and APE, despite using only 4.5M trainable parameters. On ADE20K-857, OpenWorldSAM achieves 33.1% mIoU, outperforming the previous best (X-Decoder) by +23.9 absolute points (9.2 → 33.1). On PASCAL Context-459, it achieves 47.5% mIoU,  [24] 112 (M) ✔ ✔ ✔ ✘ ✘ --18.0 3.8 * 46.5 7.8 * * -* ZegFormer (B) [55] 60 (M) ✔ ✔ ✔ ✔ ✘ --* 8.1 80.7 * * * * --OpenSeg (B) [11] 86 (M) ✔ ✘ ✔ ✔ ✘ --26.4 8.1 70.2 44.8 11.5 * * -* OVSeg (B) [12] 0.6 (M) ✔ ✔ ✘ ✘ ✘ --29.6 9.0 94.5 55. 7 12.4 * * -* MaskCLIP (L) [28] 428 (M) ✔ ✔ ✘ ✘ ✘ 15.1 6.0 23.7 8.2 * 45.9 10.0 * * * * OpenSeeD (L) [36] 39 (M) ✔ ✘ ✔ ✔ ✔ 19.7 15.0 23.4 * * * * * * * * X-Decoder-Seg + (B) [15] 28 (M) ✔ ✔ ✘ ✘ ✘ 16.9 9.5 23.8 4.6 97.8 64.7 12.1 32.2 35.1 33.8 18.5 X-Decoder (L) [15] 38 (M) ✔ ✔ ✔ ✔ ✘ 21.8 13.1 29.6 9.2 97.7 64.0 16.1 43.0 49.5 39.5 29.7 APE-B (L) [16] 42 (M) ✔ ✔ ✔ ✔ ✔ 26. 4 23.5 29.0 9.2 95.8 58.3 21.0 * * * * ESC-Net [56] 451 (M) ✔ ✔ ✘ ✘ ✘ --41.8 18.1 98.3 65.6 27.0 * * -* OpenWorldSAM 4.5 (M) ✔ ✔ ✘ ✔ ✘ 35.2 16.9 60.4 33.1 98.0 73.7 47.5 67.7 65.0 41.9 55.6 + two-stage inference 4.5 (M) ✔ ✔ ✘ ✔ ✘ 36.3 15.6 58.0 32.6 97.6 72.6 45.8 68.2 64.8 39.9 54.1 Table 2: Oracle-Prompts evaluation of open-vocabulary segmentation models. We report the state-of-the-art (SOTA) model X-Decoder [15]'s performance under both evaluation protocols. Other methods are omitted either because: 1) they are not SOTA, or 2) they do not support oracle-prompts evaluation. Model Evaluation Protocol ADE-150 ADE-857 VOC-20 PC-59 PC-459 SUN-37 SCAN-40 mIoU mIoU mIoU mIoU mIoU mIoU mIoU X-Decoder (L) [15] Global-Matching (default) 29.6 9.2 97.7 64.0 16.1 43.0 29.7 X-Decoder (L) Oracle-Prompts 51.5 29.1 98.1 75.5 42.3 67.1 49.1 OpenWorldSAM Oracle-Prompts (default) 60.4 33.1 98.0 73.7 47.5 67.7 55.6
improving over APE's 21.8% by +25.7 points, and on ScanNet-40, it reaches 55.6% mIoU, a +25.9 point improvement over X-Decoder's 29.7%. On AP score we under-perform APE, which included extra detection datasets, e.g., Objects365 [57], in their training recipe for better localization.
We attribute our strong performance to the model's prompt-conditioned decoding mechanism, which directly leverages language input to guide mask prediction. This is particularly advantageous when the target concept is known at query time. In contrast, global retrieval-based models such as X-Decoder must resolve ambiguity across the entire vocabulary space, which introduces classification error. While one might argue that differing evaluation protocols confound the comparison, it's important to note that both families of models require the same semantic input-the only difference lies in when and how that input is used.
Oracle-Prompts evaluation. As SAM-style models are designed for interactive segmentation, oracle prompts closely reflect practical use cases-such as human-in-the-loop annotation, robotic object search, or dynamic UI feedback. To fairly compare with the state-of-the-art generalist model X-Decoder [15], we also evaluate it under oracle prompts: we restrict its vocabulary to the ground-truth classes for each image. As shown in Table 2, OpenWorldSAM continues to outperform even under these controlled conditions. Notably, on large-vocabulary datasets such as ADE20K-857 and PASCAL Context-459, OpenWorldSAM achieves 33.1% and 47.5% mIoU, surpassing X-Decoder by +4.0 and +5.2 points, respectively. This highlights our model's superior language grounding ability in longtailed, fine-grained category distributions. On smaller datasets like PASCAL Context-59 and PASCAL VOC-20, where most categories overlap with COCO, X-Decoder slightly outperforms our model (75.5% vs. 73.7% mIoU and 98.1% vs. 98.0%), suggesting it benefits more from class memorization in such settings. Moreover, Figure 5 illustrates that global matching often fails despite producing correct masks. Conditioning on oracle prompts significantly reduces this ambiguity, highlighting the robustness of our evaluation protocol and the effectiveness of prompt-based segmentation.
Qualitative Results. Figure 5 presents example outputs of OpenWorldSAM on challenging scenes, with comparisons to X-Decoder under both evaluation protocols. In one example 5(a), an image from ADE20K-857 containing a game room scene is segmented by our model using prompts for  various objects ("ceiling, light, seat, person, arcade machine"). OpenWorldSAM accurately masks each object and stuff region, whereas X-Decoder misclassifies the "arcade machine" due to confusion between similar semantic objects under Global-Matching, and produces fragmented masks for the person and seat under Oracle-Prompts. Similarly in example 5(b), X-Decoder misclassifies the "wall" and proposes object masks for prompts that did not exist in the ground truth (e.g., "window glass") under Global-Matching, and failed to segment "plant" under Oracle-Prompts. This showcases our model's clear understanding of category semantics (thanks to the VLM prompt) combined with precise mask delineation (thanks to SAM2's capability). More qualitative results in Appendix C.
this section cite: ['b23', 'b54', 'b10', 'b11', 'b6', 'b27', 'b35', 'b14', 'b14', 'b15', 'b3', 'b56', 'b14']

Section: Referring Expression Segmentation Performance Analysis
Performance. As shown in Table 3 and Figure 6, OpenWorldSAM achieves strong performance on the RefCOCOg validation set, obtaining a cIoU of 74.0%, significantly outperforming earlier generalist models like SEEM and X-Decoder (≈65%), and competitive with specialized models such as GLaMM (71.2%) and UNINEXT (74.4%). Notably, OpenWorldSAM reaches this accuracy using just BEiT-3 encoder with 673M parameters and an additional 4.5M trainable parameters, substantially fewer than recent large-scale models like LISA, GLaMM, and u-LLaVA, which rely on much larger vision-language foundations (7B+ parameters) and multiple additional datasets. While EVF-SAM achieves higher cIoU (77.0%), OpenWorldSAM inherits SAM's interactive features, offering unique flexibility in interative segmentation tasks. Furthermore, with its tie-breaker and soft-prompting modules, OpenWorldSAM can also perform general segmentation tasks, distinguishing it from higher-scoring yet narrower models.
this section cite: []

Section: Ablation Studies
We systematically validate OpenWorldSAM's design through zero-shot transfer on ADE20K-150/857 benchmark and fintuning on RefCOCOg benchmark.
E1 ✘ ✘ ✘ ✔ 1.2 (M) ✘ ✔ ✘ 0.4 1.0 1.2 0.2 E2 ✔ ✘ ✘ ✔ 1.3 (M) ✔ ✘ ✘ - 9.5 - - E3 ✔ ✘ ✘ ✔ 1.3 (M) ✔ ✔ ✘35
.1 17.1 56.8 32.2 E4 ✔ ✔ ✘ ✔ 674.0 (M) ✔ ✔ ✘ 13.6 3.5 24.4 10.6 E5 ✔ ✘ ✔ ✔ 4.5 (M) ✔ ✔ ✔ 35.2 16.9 60.4 33.1 E6 ✔ ✔ ✔ ✔ 677.2 (M) ✔ ✔ ✔ 15.9 3.8 23.6 11.2
Multi-modal encoder analysis. In Table 4, we compares performances using different VLM encoders and fusion methods (early fusion vs. late fusion). BEiT-3's early cross-modal fusion (joint text-image processing across all layers) outperforms CLIP's late fusion (last-layer concatenation) by +33.9 mIoU, +21.2 PQ, and +13.3 AP on ADE-150, demonstrating that deep semantic integration is critical for aligning language concepts with visual regions, echoing findings by EVF-SAM [43].
Visual Context Matters. Table 4 demonstrates that removing visual inputs to BEiT-3 (text-only) causes catastrophic performance collapse (-34.4 mIoU on ADE-150). This confirms that SAM's segmentation backbone cannot ground textual semantics without explicit visual-textual co-encoding.
this section cite: ['b42']

Section: Optimal Training Strategy.
In Table 5, we varied the trainable modules in OpenWorldSAM (thus varying total new parameters from 1.2M to 770M). We found in E5 that freezing BEiT-3 and training only the language adapter module (tie-breaker + cross-attention, 4.5M parameters) yields optimal performance (60.4 mIoU ADE-150). Notably, comparing E6 vs E5 and E4 vs E3, we found fine-tuning the entire BEiT-3 encoder (673M parameters) significantly degrades accuracy (mIoU drops from 60.4 to 23.6), likely due to underfitting on sparse category label prompts compared to its original web-scale pretraining.
Positional tie-breaker vs. none. Comparing E3 vs E1 in Table 5, positional tie-breaker boosts AP from 1.0% to 17.1%. As shown in Figure 3, without the tie-breaker, the model usually collapses on one instance of the class (especially if the one was particularly salient among others). This confirms the necessity of this component for reliable instance segmentation.
this section cite: []

Section: Cross-Attention layer removal.
As shown in Table 5, E5 vs E3, removing the cross-attention layers expectedly led to inferior performance (-1.5 mIoU on ADE-150 and -0.9 mIoU on ADE-857). This indicates that cross-attention helps align prompts to the intended visual regions.
this section cite: []

Section: Conclusion
OpenWorldSAM bridges the gap between promptable segmentation and open-vocabulary understanding by unifying SAM's segmentation prowess with vision-language models' semantic grounding. This approach generalizes across tasks (semantic/instance/panoptic) and prompts (nouns/sentences), offering practitioners a unified tool for real-world scenarios where novel objects and ambiguous queries are the norm. Three innovations drive this success: (1) Positional tie-breakers enable multi-instance segmentation from single-text queries, resolving a critical limitation of SAM-like architectures.
(
Cross-modal soft prompting dynamically aligns language semantics with SAM's visual space, ensuring precise localization without costly LLMs. (3) Frozen foundation synergy leverages pretrained knowledge from SAM and BEiT-3, proving that dense prediction tasks benefit as much as classification from parameter-efficient adaptation. Beyond technical contributions, OpenWorldSAM advances a paradigm for extending segmentation foundations: instead of training monolithic models, strategic adaptation of frozen components achieves open-world readiness at minimal cost. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.
this section cite: []

Section: A Experimental Settings

this section cite: []

Section: A.1 Pre-training
We implement our model in PyTorch, building on the Detectron2 [68] framework. We initialize the base models with the public weights of SAM2-Hiera-Largefoot_0 and BEIT-3-Largefoot_1 . The model is pre-trained for 25 epochs on the COCO-2017 training split (104K images) [46]. We use the panoptic annotations, which provide pixel-accurate masks and category labels for all 132 thing and stuff classes. Training is conducted on a single NVIDIA A100 (80 GB) GPU with a batch size of 8. Optimization employs AdamW (learning rate 1e-4). A step decay scheduler drops the learning rate by a factor of 0.1 at 89% and 96% of the total iterations. Compared with recent generalist models, our recipe is markedly more data-efficient (see Table 6).
this section cite: ['b67', 'b45']

Section: A.2 Zero-Shot Evaluation
We evaluate semantic, instance, and panoptic segmentation in a zero-shot setting. For instance segmentation and panoptic segmentation, we apply confidence-score filtering to remove masks with scores below 0.7, followed by non-maximum suppression (NMS) with IoU threshold 0.5 to remove duplicate detections and retain distinct object instances. The confidence scores, originally termed "estimated IoU scores" in SAM [17,18], are direct outputs from SAM2's mask decoder. These scores were optimized during SAM2's pre-training to select high-quality (i.e., confident) mask outputs. The open-vocabulary benchmark comprises 5 datasets covering 8 different segmentation tasks; statistics are summarized in Table 7. We show a comprehensive evaluation protocol for openvocabulary segmentation in various vocabulary sizes and image domains.
this section cite: ['b16', 'b17']

Section: A.3 Finetuning
For referring-expression segmentation we fine-tune the pre-trained checkpoint on RefCOCOg UMD training split for 10 epochs. Because images from RefCOCOg were seen during pre-training (with category labels substituted for referring expressions ground truth), we adopt a conservative learning rate of 1e-5. We use a batch size of 8 during training.
this section cite: []

Section: B Qualitative Comparison on Two-Stage-Inference
During inference, we perform an optional two-stage inference. First, the model predicts multi-instance masks. These masks are then fed back as visual prompts, and SAM2's mask decoder is run a second time to refine the contours. Figure 7 illustrates the visual improvement. However, quantitative gains are marginal across segmentation metrics (see Sec.
this section cite: []

Section: C Additional Zero-Shot Qualitative Results
Figure 8 showcases multiple challenging indoor scenes drawn from ADE20K-150/857 [22], and PASCAL Context-459 [51]. In each sub-panel, we compare example outputs of OpenWorldSAM with comparisons to X-Decoder under both global-matching and oracle-prompts evaluation protocols.
Panel (a) (ADE20K-150) top row depicts a cluttered bedroom. OpenWorldSAM cleanly delineates thin structures such as the "closet" edge and the narrow "lamp stem", and assigns a single coherent mask to the "cushion". X-Decoder fragments the closet and mis-classifies the cushion as a generic "pillow" under global matching. Under oracle-prompts, X-decoder fails to predict "cushion". Similarly, the bottom row depicts an airport conveyor belt. X-Decoder mis-classifies the "bulletin board" as the "crt screen", the "box" as the "trade name" under global matching, and still mis-classifies the "box" under oracle-prompts.
Panel (b) (ADE20K-857) top row shows a dining area. Under the global-matching protocol, X-Decoder hallucinates "rug"/"rocking chair" labels and fragments the "sofa bed" pixels. The bottom row shows a cluttered living room where X-Decoder outputs fragmented low-quality masks and false predictions under both evaluation protocols. In comparison, our model preserves category fidelity-introducing no extra labels-and produces noticeably cleaner chair boundaries, illustrating the synergy between BEiT-3 language grounding and SAM2's high-resolution masks.
Panel (c) (PASCAL Context-459) top row shows that X-Decoder fails to predict the "cloth" object. The bottom row is an indoor scene crowded with small objects ("cd", "speaker", "chair"). OpenWorld-SAM retrieves almost every queried category (except for "cd") and suppresses false positives such as "calendar" and "ladder" that appear in X-Decoder's output, demonstrating stronger open-vocabulary grounding and sharper instance separation.
this section cite: ['b21', 'b50']

Section: D Limitations -Outdoor Generalization
Despite strong results on indoor and everyday photographs, OpenWorldSAM under-performs on driving datasets such as Cityscapes [69] and BDD10K [70] (Table 8). Fine-tuning on Cityscapes narrows the gap, yet performance still trails methods explicitly exposed to multi-domain data.
Understanding the source of this shortfall is essential for future extensions.
Observed failure modes. Figure 9 shows high IoU for broad stuff regions (e.g. road, sky), but a sharp drop for small or elongated thing instances. Correspondingly, AP remains low for motorcycle, person, bicycle, etc.  Hypotheses.
(i) Domain shift. COCO images are mostly handheld and indoor, whereas Cityscapes/BDD10K contain forward-looking dash-cam frames with motion blur, glare and night scenes. X-Decoder was co-trained on web-scale image-text pairs that include many outdoor photos, so its visual encoder has wider coverage. Large-scale multi-domain training is known to mitigate domain shift [71].
(ii) Resolution bottleneck. Cityscapes frames are 2048×1024. Rescaling to 1024×1024 (SAM default) reduces poles and traffic lights to nearly one pixel at the feature stride of 16×. X-Decoder keeps an FPN branch at 8×, preserving thin structures.
Take-away. COCO-only pre-training for OpenWorldSAM leaves a blind spot for urban driving imagery-particularly for distant, thin or cluttered objects under challenging lighting. Bridging the gap likely requires (i) explicit exposure to outdoor domains and (ii) higher-resolution feature branches. We leave large-scale outdoor pre-training and depth-aware augmentation for future work.
this section cite: ['b68', 'b69', 'b70']

Section: E Model Structure Details
Table 9 summarizes the architectural differences between OpenWorldSAM and competing models, detailing each method's visual backbone, segmentation head, text encoder, and training-image resolution.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: In our abstract and introduction, we talk about our contribution on proposing a novel framework for open-vocabulary segmentation. We provide extensive experiments on comprehensive datasets to support this claim. We also conduct in-depth ablation studies that verifies the effectiveness of our model design.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss the limitations in Appendix D, which is about the model generalization quality to outdoor scenes and self-driving scenes.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: Our work does not include theoretical assumptions and proofs.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We provide the detailed methodology and experimental setup in Sections 3 and 4. Moreover, we provide all source codes to reproduce the results, including training scripts (detailed configurations included) and evaluation scripts (model checkpoints included). We will open source the code on GitHub after acceptance.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We provide the source codes as supplementary material. We provide instructions that contain the exact command and environment needed to run to reproduce the results. We provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: Justification: Training and test details can be found in Sections 3 and 4, and Appendix A.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: We did not include the error bars as we fix the random seed for every experiment, reducing the impact from data loader and other parameters initialization.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We provide the details of computer resources in Section 4. All experiments can be run on a single A100 GPU. We also provide analysis on trainable parameters in Section 4.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: Our experiments conform to the NeurIPS Code of Ethics.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
this section cite: []

Section: Answer: [NA]
Justification: There is no social impact of this work.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: The paper poses no such risks.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: Our model and its code development are based on baseline works which are credited in the paper. Our datasets are the standard benchmarks that are widely used in academia.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Table 9: Architectural choices for recent open-vocabulary models: visual backbone, base segmentation model, text/multi-modal encoder, and training image size.
this section cite: []

Section: Method Visual Backbone
Base Model Text Encoder Image Size
this section cite: []

Section: Short Long
MSeg (B) [54] HRNet-W48 (65 M) HRNet-Seg -1024 1024 GroupViT (S) [27] ViT-S/16 (22 M) GroupViT Transformer 224 224 LSeg+ (B) [24] CLIP ViT-B/16 (86 M) DenseCLIP CLIP 512 512 ZegFormer (B) [55] CLIP ViT-B/16 (86 M) ZegFormer CLIP 640 640 OpenSeg (B) [11] ResNet-101 (45 M) OpenSeg CLIP/ALIGN 640 640 OVSeg (B) [12] CLIP ViT-B/16 (86 M) MaskFormer CLIP 640 640 MaskCLIP (L) [28] CLIP ViT-L/14 (307 M) MaskCLIP CLIP 1024 1024 X-Decoder [15] DaViT-L (196 M) X-Decoder CLIP 1024 1024 OpenSeeD [36] Swin-L (197 M) MaskDINO UniCL 1024 1024 SEEM [35] DaViT-L (196 M) X-Decoder CLIP 800 1333 APE (B) [16] ViT-L (307 M) DETA CLIP 1024 1024 PolyFormer (L) [59] Swin-L (197 M) PolyFormer BERT 1024 1024 UNINEXT (H) [32] ViT-H (632 M) DINO BERT 320∼800 1333 PixelLM [61] CLIP ViT-L/14 (307 M) PixelLM LLaMA2-13B 448 448 LISA [19] SAM ViT-H (636 M) SAM Vicuna-7B 1024 1024 GLaMM [20] SAM ViT-H (636 M) SAM Vicuna-7B 1024 1024 u-LLaVA [21] SAM ViT-H (636 M) SAM Vicuna-7B 1024 1024 EVF-SAM [43] SAM ViT-H (636 M) SAM BEiT-3 1024 1024 EVF-SAM2 [43] SAM2 Hiera-L (224 M) SAM2 BEiT-3 1024 1024 OpenWorldSAM SAM2 Hiera-L (224 M) SAM2 BEiT-3 1024 1024
this section cite: ['b53', 'b26', 'b23', 'b54', 'b10', 'b11', 'b27', 'b14', 'b35', 'b34', 'b15', 'b58', 'b31', 'b60', 'b18', 'b19', 'b20', 'b42', 'b42']

Section: E.1 Possible Text Encoder Alternatives
We argue that the key ingredients for open-vocabulary segmentation are backbone-agnostic: any strong interactive segmenter can supply high-resolution mask decoding, while any pretrained vision-language encoder can provide semantics. What is missing is a lightweight adaptor that (i) aligns the two embedding spaces, (ii) scales to multiple object instances from a single text query, and (iii) preserves the efficiency that makes interactive segmentation attractive in the first place.
Our OpenWorldSAM is a general plug-in architecture that satisfies these desiderata while keeping all heavy backbones frozen. Although we instantiate the framework with SAM2 and BEiT-3 in this paper, neither component is required by design; alternative interactive decoders or vision-language encoders can be swapped in with only minor re-training of the adapter.
Table 10 surveys representative VLM encoders that could replace BEiT-3 in OpenWorldSAM with ≤5 M adaptor parameters. All rows assume the heavy backbone is frozen; only the 256-D projector and tie-breakers are fine-tuned.
Adaptor fine-tuning recipe (all encoders). Freeze all VLM weights and SAM2 decoder; initialize a d in ×256 MLP projector and K 256-D tie-breaker embeddings (default K = 20, total ≈5M params). For training, one could use unchanged Hungarian matching loss on COCO.
Takeaway. Early-fusion encoders (VLMo, OFA, Florence-2) require zero architectural change beyond projector resizing and are therefore the most promising immediate swaps. Dual-encoders (CLIP family) need a shallow cross-attention adaptor to overcome missing image context. Larger hybrids (BLIP-2, Kosmos-2, PaLI) open research directions (multi-query tie-breakers, OCR) at the cost of real-time guarantees.
this section cite: []

Section: F Additional Ablation Studies
We provide additional ablation studies on the number of tie-breaker tokens and the number of cross-attention layers.
this section cite: []

Section: F.1 Effect of varying tie-breaker tokens
We set the hyperparameter K = 20, meaning for each prompt (e.g., a category name), our model can identify up to 20 distinct objects. For crowded scenes containing more than 20 objects per category, increasing K is straightforward and advisable. In practice, COCO images typically contain a moderate number of distinct categories and instances (the original COCO paper reports "on average, our dataset contains 3.5 categories and 7.7 instances per image." [46]). The chosen value should match or exceed the maximum expected number of objects per category. For reference, DETR [45] used 100 total queries, aligning roughly with the maximum number of objects per image. Our choice (K = 20) results, on average, in approximately 70 queries per image (20 queries × 3.5 categories), providing ample coverage for typical scenes.
Further, [81] observed that increasing queries initially improved Average Precision (AP), but then plateaued or even slightly declined when queries became excessive, indicating redundancy in higher query counts. However, recall does improve with more queries, since more detection slots increase the chance to find each object.
We conducted additional ablation experiments in Table 11 by varying K, pretrained on COCO and evaluated on ADE20K instance segmentation. Observations. (1) Increasing K from 10 to 20 improves recall and AP; beyond 20 gains saturate, mirroring the behavior reported for DETR-style object queries; (2) Average Recall with max 100 detections per image (AR@100) improve when increasing K from 10 → 20 → 30;
(3) K = 20 is optimal for balancing precision and recall in standard datasets.
this section cite: ['b45', 'b44', 'b80']

Section: F.2 Effect of varying number of cross-attention layers
In Table 12, we observe consistently higher accuracy with 3-layer cross-attention across datasets, confirming the importance of multi-layer cross-attention. However, a single-layer variant significantly narrows the gap with fewer parameters (2.4M vs. 4.5M), suggesting a practical compromise between parameter count and accuracy.
this section cite: []

Section: G Inference Speed Analysis
We conducted detailed profiling to quantify the impact of adding the VLM and our adapter modules to SAM. In Table 13 and 14, we present inference timing breakdowns for processing a single 1024 × 1024 image on an NVIDIA A5000 GPU, averaged over five independent runs.
this section cite: []

Section: Summary (single prompt).
SAM modules total time: 373.35 ms (81.0%), Non-SAM modules total time: 87.21 ms (18.9%), Non-SAM overhead: 87.21 ms.
Summary (six prompts). SAM modules total time: 539.72 ms (78.4%), Non-SAM modules total time: 148.65 ms (21.6%), Non-SAM overhead: 148.65 ms.
this section cite: []

Section: Takeaway.
The profiling results show that adding the VLM and adapter modules results in only a moderate increase in inference time (approximately 19-22% overhead). Most computational cost remains within SAM's backbone and mask decoder.
Mask Decoder scaling. sam_mask_decoder cost grows almost linearly with (K × P ).
• Going from 1 → 20 queries (same prompt) adds ∼41 ms.
• Going from 1 prompt → 6 prompts (120 queries) adds a further ∼162 ms.
Note that one text prompt mimics user clicks 20 times on an image. If automatic mask generation is desired without user intervention, SAM's built-in auto-mask generator uses a dense 32 × 32 grid of point prompts, incurring significantly higher costs compared to our text-based prompting approach.
Overall overhead. Relative to one vanilla SAM2 call, our pipeline is approximately 39% slower for a single prompt (332 → 461 ms). However, it becomes approximately 3× more efficient when handling three or more prompts, as the backbone and VLM overhead are amortized. Thus, our enhancements introduce manageable overhead, maintaining practical usability in real-world applications.
this section cite: []

Section: References
Ref_id:b0 Title: Fully convolutional networks for semantic segmentation Year: (2015)
Ref_id:b1 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b2 Title: Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs Year: (2017)
Ref_id:b3 Title: Encoder-decoder with atrous separable convolution for semantic image segmentation Year: (2018)
Ref_id:b4 Title: Mask r-cnn Year: (2017)
Ref_id:b5 Title: Segformer: Simple and efficient design for semantic segmentation with transformers Year: (2021)
Ref_id:b6 Title: Masked-attention mask transformer for universal image segmentation Year: (2022)
Ref_id:b7 Title: Mask dino: Towards a unified transformer-based framework for object detection and segmentation Year: (2023)
Ref_id:b8 Title: Active open-vocabulary recognition: Let intelligent moving mitigate clip limitations Year: (2024)
Ref_id:b9 Title: Findanything: Open-vocabulary and object-centric mapping for robot exploration in any environment Year: (2025)
Ref_id:b10 Title: Scaling open-vocabulary image segmentation with image-level labels Year: (2022)
Ref_id:b11 Title: Open-vocabulary semantic segmentation with mask-adapted clip Year: (2023)
Ref_id:b12 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b13 Title: Scaling up visual and vision-language representation learning with noisy text supervision Year: (2021)
Ref_id:b14 Title: Generalized decoding for pixel, image, and language Year: (2023)
Ref_id:b15 Title: Aligning and prompting everything all at once for universal visual perception Year: (2024)
Ref_id:b16 Title: Segment anything Year: (2023)
Ref_id:b17 Title: Segment anything in images and videos Year: (2024)
Ref_id:b18 Title: Reasoning segmentation via large language model Year: (2024)
Ref_id:b19 Title: Glamm: Pixel grounding large multimodal model Year: (2024)
Ref_id:b20 Title: u-llava: Unifying multi-modal tasks via large language model Year: (2024)
Ref_id:b21 Title: Scene parsing through ade20k dataset Year: (2017)
Ref_id:b22 Title: Modeling context between objects for referring expression understanding Year: (2016)
Ref_id:b23 Title: Languagedriven semantic segmentation Year: (2022)
Ref_id:b24 Title: Regionclip: Region-based language-image pretraining Year: (2022)
Ref_id:b25 Title: Simple open-vocabulary object detection Year: (2022)
Ref_id:b26 Title: Groupvit: Semantic segmentation emerges from text supervision Year: (2022)
Ref_id:b27 Title: Maskclip: Masked self-distillation advances contrastive language-image pretraining Year: (2023)
Ref_id:b28 Title: Per-pixel classification is not all you need for semantic segmentation Year: (2021)
Ref_id:b29 Title: Grounded language-image pre-training Year: (2022)
Ref_id:b30 Title: Grounding dino: Marrying dino with grounded pre-training for open-set object detection Year: (2024)
Ref_id:b31 Title: Universal instance perception as object discovery and retrieval Year: (2023)
Ref_id:b32 Title: Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world detection Year: (2022)
Ref_id:b33 Title: Open-vocabulary panoptic segmentation with text-to-image diffusion models Year: (2023)
Ref_id:b34 Title: Segment everything everywhere all at once. Advances in neural information processing systems Year: (2023)
Ref_id:b35 Title: A simple framework for open-vocabulary segmentation and detection Year: (2023)
Ref_id:b36 Title: Hierarchical open-vocabulary universal image segmentation Year: (2023)
Ref_id:b37 Title: Semantic-sam: Segment and recognize anything at any granularity Year: (2023)
Ref_id:b38 Title: Grounded sam: Assembling open-world models for diverse visual tasks Year: (2024)
Ref_id:b39 Title:  Year: (2023)
Ref_id:b40 Title: Refsam: Efficiently adapting segmenting anything model for referring video object segmentation Year: (2023)
Ref_id:b41 Title: An improved baseline for reasoning segmentation with large language model Year: (2023)
Ref_id:b42 Title: Evf-sam: Early vision-language fusion for text-prompted segment anything model Year: (2024)
Ref_id:b43 Title: Image as a foreign language: Beit pretraining for vision and vision-language tasks Year: (2023)
Ref_id:b44 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b45 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b46 Title: Attention is all you need Year: (2017)
Ref_id:b47 Title: Hiera: A hierarchical vision transformer without the bells-and-whistles Year: (2023)
Ref_id:b48 Title: Window attention is bugged: how not to interpolate position embeddings Year: (2023)
Ref_id:b49 Title: The pascal visual object classes challenge Year: (2011)
Ref_id:b50 Title: The role of context for object detection and semantic segmentation in the wild Year: (2014)
Ref_id:b51 Title: Scannet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017)
Ref_id:b52 Title: Sun rgb-d: A rgb-d scene understanding benchmark suite Year: (2015)
Ref_id:b53 Title: Mseg: A composite dataset for multi-domain semantic segmentation Year: (2020)
Ref_id:b54 Title: Decoupling zero-shot semantic segmentation Year: (2022)
Ref_id:b55 Title: Effective sam combination for open-vocabulary semantic segmentation Year: (2025)
Ref_id:b56 Title: Objects365: A large-scale, high-quality dataset for object detection Year: (2019)
Ref_id:b57 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b58 Title: Polyformer: Referring image segmentation as sequential polygon generation Year: (2023)
Ref_id:b59 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b60 Title: Pixellm: Pixel reasoning with large multimodal model Year: (2024)
Ref_id:b61 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b62 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b63 Title: Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos Year: (2025)
Ref_id:b64 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b65 Title: Infusing wisdom in sam2 for text-driven video segmentation Year: (2025)
Ref_id:b66 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b67 Title:  Year: (2019)
Ref_id:b68 Title: The cityscapes dataset Year: (2015)
Ref_id:b69 Title: A diverse driving video database with scalable annotation tooling Year: (2018)
Ref_id:b70 Title: Multi-domain semantic segmentation with overlapping labels Year: (2022)
Ref_id:b71 Title: Vlmo: Unified vision-language pre-training with mixture-of-modality-experts Year: (2022)
Ref_id:b72 Title: Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework Year: (2022)
Ref_id:b73 Title: Florence-2: Advancing a unified representation for a variety of vision tasks Year: (2024)
Ref_id:b74 Title: Eva-clip: Improved training techniques for clip at scale Year: (2023)
Ref_id:b75 Title: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features Year: (2025)
Ref_id:b76 Title: Coca: Contrastive captioners are image-text foundation models Year: (2022)
Ref_id:b77 Title: A jointly-scaled multilingual language-image model Year: (2022)
Ref_id:b78 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b79 Title: Kosmos-2: Grounding multimodal large language models to the world Year: (2023)
Ref_id:b80 Title: What are expected queries in end-to-end object detection Year: (2022)
