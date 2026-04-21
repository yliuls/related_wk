Title: PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding
Abstract: Real-world objects are composed of distinctive, object-specific parts. Identifying these parts is key to performing fine-grained, compositional reasoning-yet, large multimodal models (LMMs) struggle to perform this seemingly straightforward task. In this work, we introduce PARTONOMY, an LMM benchmark designed for pixel-level part grounding. We construct PARTONOMY from existing part datasets and our own rigorously annotated set of images, encompassing 862 part labels and 534 object labels for evaluation. Unlike existing datasets that simply ask models to identify generic parts, PARTONOMY uses specialized concepts (e.g., agricultural airplane), and challenges models to compare objects' parts, consider part-whole relationships, and justify textual predictions with visual segmentations. Our experiments demonstrate significant limitations in state-of-the-art LMMs (e.g., LISA-13B achieves only 5.9% gIoU), highlighting a critical gap in their part grounding abilities. We note that existing segmentation-enabled LMMs (segmenting LMMs) have two key architectural shortcomings: they use special [SEG]   tokens not seen during pretraining which induce distribution shift, and they discard predicted segmentations instead of using past predictions to guide future ones. To address these deficiencies, we propose PLUM, a novel segmenting LMM that uses span tagging instead of segmentation tokens and that conditions on prior predictions in a feedback loop. We find that pretrained PLUM outperforms existing segmenting LMMs on reasoning segmentation, VQA, and visual hallucination benchmarks. In addition, PLUM finetuned on our proposed Explanatory Part Segmentation task is competitive with segmenting LMMs trained on significantly more segmentation data. Our work opens up new avenues towards enabling fine-grained, grounded visual understanding in LMMs.

Section: Introduction
Real-world objects can be decomposed into distinctive parts. A banana boat (Fig. 2), for instance, consists of a seating tube, a handle, a hull, and an inflation valve. Such parts characterize each concept, differentiating one object from another. The ability to recognize and distinguish between parts is an important element of holistic object understanding, with applications ranging from explainable object recognition [3,16,28,8,58], to part-based novel concept design [10], and robotic manipulation [52,30,14]. Decomposing objects into their key building blocks allows models to reason about objects at a granular level [16,29], allowing for more complex and nuanced interactions.
Unfortunately, Large Multimodal Models (LMMs), the backbones of today's multimodal systems, lack strong part recognition abilities [16,29,33]. While they perform well on visual reasoning [43,13,42] and visual hallucination tasks [23], we find that they are unable to accurately identify * Equal contribution.
this section cite: ['b2', 'b15', 'b27', 'b7', 'b57', 'b9', 'b51', 'b29', 'b13', 'b15', 'b28', 'b15', 'b28', 'b32', 'b42', 'b12', 'b41', 'b22']

Section: 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
What visible parts does this banana boat have in common with a fishing boat?
It has a hull.
this section cite: []

Section: Part Intersection
What is the name of this object?
Okay, what parts make this look like a torpedo? It has a fuel tank, …, and a guidance system.
this section cite: []

Section: Whole-to-Part
What parts does this banana boat have which a fishing boat does not?
It has a handle, inflation valve and a seating tube.
this section cite: []

Section: Part Difference
What visible parts does the agricultural airplane in the image have? "The agricultural airplane in the image has: fixed landing gear, a propulsion component, a spraying rig, and wings."
this section cite: []

Section: Part Identification
What parts does this object have?
It has a spraying rig, … , and fixed landing gear.
Okay, based on the parts, what is this object? This is an agricultural airplane.
this section cite: []

Section: Part-to-Whole

this section cite: []

Section: User PLUM (Ours)

this section cite: []

Section: Partonomy Instance Construction through Question-Answer Mutation
Question + Answer Q: What visible parts does this banana boat in the image have in common with a canoe?
A: It has a hull.
this section cite: []

Section: Mutate
A: It has a hull and a seating tube.
A: It has a paddle.
A: It has an inflation valve and a paddle.
this section cite: []

Section: …
Mutated Answers Mutation Operations + -+ +
Fixed landing gear Cockpit Propulsion Component Spraying Rig Wings Fuel Tank Fins Guidance System Propeller Warhead ... ... This is a torpedo. Given an input image, a segmentation-enabled LMM selects a textual explanation and generates part segmentation masks which serve as textual and visual rationale for its answer choice. Our question-answer mutation framework generates challenging answer choices by predicting part co-occurrence and by selecting parts from confusable objects.
object parts in an image, occasionally regurgitating object parts memorized from text-only pre-training (e.g. "a fish must have a fin"). Worse, LMMs that can generate segmentation masks, segmenting LMMs [19,37,41,49], lack the ability to ground these fine-grained regions despite being trained on part segmentation data. This severely limits LMMs' utility in real-world scenarios that require fine-grained, part-level understanding.
To quantify the part recognition abilities of LMMs, we propose Explanatory Part Segmentation, a task that assesses LMMs' ability to recognize object parts, associate objects with their distinctive parts, and use these grounded parts to predict object labels. We then introduce PARTONOMY, a comprehensive benchmark for the Explanatory Part Segmentation task. We construct PARTONOMY from existing part segmentation datasets [11,7,35] and our manually-annotated evaluation dataset of 1K specialized object-centric images with complex part annotations. This subset, PARTONOMY-Core, contains 862 distinct part labels-more than any existing part datasets (Table 1).
We then note two shortcomings of existing segmenting LMMs' architectures [19,37,41]. First, they always rely on special [SEG] tokens not seen during pretraining, potentially hindering downstream performance by introducing distribution shift. Second, these segmenting LMMs discard their predictions after each output, missing the opportunity to incorporate prior information contained by the masks they predicted during the decoding process. This design is in contrast to modern generative frameworks, which condition future predictions on past ones [48,44]. Based on these observations, we propose PLUM, a Part-Level Understanding LMM. PLUM uses a text span tagging module to avoid special segmentation tokens that induce distribution shift from the pre-trained LLM, and employs a mask feedback mechanism to condition on past predictions (Section 4). Our results show that pretrained PLUM retains its general reasoning abilities far better than other segmenting LMMs, achieving stronger zero-shot segmentation performance and competitive finetuned performance to models trained on significantly more segmentation data.
this section cite: ['b18', 'b36', 'b40', 'b48', 'b10', 'b6', 'b34', 'b18', 'b36', 'b40', 'b47', 'b43']

Section: Related Work
Reasoning in Large Multimodal Models Reasoning capabilities in Large Language Models (LLMs) uncovered by prompting techniques such as Chain-of-Thought (CoT) [51,18] have led to increased interest in their application to LMMs [31,55,12]. Previous work shows that reasoning abilities of LLMs allow them to generate textual rationales given image inputs, allowing them to handle complex visual reasoning tasks such as A-OKVQA [42] and ScienceQA [31]. Nonetheless, LLMs' output space is confined to text, limiting their spatial understanding and often leading to hallucinatory text outputs [26]. While recent efforts on visual compositional reasoning attempt to mitigate the gap between the text and image modalities in LMMs [54,55,32] by using external modules such as object detectors [6] or code interpreters [46], most attempts don't truly reflect the innate visual reasoning capabilities of LMMs. Our proposed model and a recent line of LMMs [19, 37, 41] try to accomplish this by interleaving textual and visual rationale generated through segmentations.
Segmentation-Enabled Large Multimodal Models LMMs such as LISA [19] and GLaMM [37] have demonstrated the ability to generate text and grounded segmentation masks. Despite being trained on part-level segmentation datasets such as PACO [35] and Pascal-Part [9], they struggle to exhibit a part-level understanding of visual concepts. While they demonstrate the ability to understand complex textual instructions [19,37,41,53], current LMMs fail to relate concept-indicative parts to their wholes, as shown in Fig. 2. Frequently, they fail to generate the specialized segmentation token (e.g., [SEG]) added to their vocabulary, leading to no masks being generated for the parts. Even state-of-the-art LMMs that seemingly "reason" struggle to establish attributive relationships between objects and parts, implying that current models and datasets lack the coverage and capacity to handle complex part understanding and grounding tasks. This observation motivates the proposal of our new Explanatory Part Segmentation task, which requires LMMs to segment objects' parts (i.e. producing visual rationale) while generating the corresponding text rationale.
this section cite: ['b50', 'b17', 'b30', 'b54', 'b11', 'b41', 'b30', 'b25', 'b53', 'b54', 'b31', 'b5', 'b45', 'b18', 'b36', 'b34', 'b8', 'b18', 'b36', 'b40', 'b52']

Section: Part Semantic Segmentation
Part segmentation is the task of decomposing objects into their constituent parts through segmentation [7,56,57,11,35]. While this task has been studied in openvocabulary [24,25,59,40] and multiple segmentation tasks [20], no existing work has evaluated LMMs' ability to segment objects' "concept-indicative" parts-those that help define the object category. In fact, most recent efforts on segmentation-enabled LMMs focus on concept labels or referring expressions [37], ignoring part segmentation altogether. Our PARTONOMY dataset, which includes our manually annotated PARTONOMY-Core evaluation set, integrates existing part-level segmentation datasets such as PACO [35] and PartImagenet [11] to further the part and object-level diversity of our benchmark.
this section cite: ['b6', 'b55', 'b56', 'b10', 'b34', 'b23', 'b24', 'b58', 'b39', 'b19', 'b36', 'b34', 'b10']

Section: PARTONOMY: A Dataset for Explanatory Part Segmentation

this section cite: []

Section: Task Overview
We motivate our task definition by characterizing a model with part understanding. First, such a model should be able to identify parts. Given an image of an object, the model can list visible parts and ground them in the image. Second, this model should be able to compare and associate object parts. It recognizes that both dogs and tables have legs, despite their difference in form. On the other hand, it understands that a passenger plane and a biplane both have wings, but that the biplane's double wings are a feature that distinguishes it from other aircraft. Finally, this model should be able to use its part knowledge to predict object labels based on their parts. Identifying a key feature, like a large scope, suggests to the model that a rifle is likely a sniper rather than assault rifle.
To evaluate these elements of part understanding, we define the Explanatory Part Segmentation task. In this task, a model is provided with an image and a question about an object's parts (e.g. Table 1: Comparison between part segmentation datasets. † indicates usage in PARTONOMY. "C" refers to common objects (e.g. chair, airplane), "O" to organisms (e.g. dogs, snake), and "S" to specialized objects (e.g. intersecting lines, highway map, fighter jet). PARTONOMY-Core has over three times as many object labels and four times as many part labels as the widely used PACO dataset, has more part labels than PartImageNet++ (which has twice as many object labels), and contains specialized object parts annotated on object-centric images.
Datasets # Object Labels # Part Labels # Object-Part Labels # Images # Seg. Masks Object-Centric Images Object Domain PASCAL-Part † 20 30 193 10,103 111,960 ✗ C, O PartImageNet † 158 14 14 24,000 112,000 ✓ C, O PartImageNet++ 1000 818 3,308 100K 406.4K ✓ C, O PACO † 75 200 456 84,027 641,000 ✗ C PARTONOMY-Core † 534 862 1,976 1,068 4,968 ✓ C, O, S PARTONOMY 606 975 2,507 74,500 407,101 ✓ C, O, S
"What visible parts does the agricultural plane in the image have?" Fig. 1). The model must then select the best response and generate segmentation masks for the corresponding parts to explain its selection (e.g. "The agricultural plane has wings, a propulsion component, and a spraying rig"). Motivated by our characterization of part understanding, we define three classes of questions (Fig. 1):
Part Identification questions ask the LMM to identify then segment an object's visible parts. These questions test LMMs' ability to recognize and ground parts without hallucination.
Part Comparison questions ask the LMM to identify an object's visible parts and compare or contrast them to the parts of another object. These questions test models' knowledge of objects' common parts. Concretely, let P I and P C be the parts of an object in the image and the parts of a separate comparative concept. We define two subtasks:
• Part Intersection. The model is asked which visible parts the object in the image has in common with a specified query concept, P I ∩ P C , then segments them.
• Part Difference. The model is asked which visible parts the object in the image has which the query concept does not, P I \ P C , then segments them.
Part-Whole Reasoning asks the LMM to identify an object or its parts as a consequence of the other. These questions assess whether the model can apply its part knowledge to identify objects, or use an object to identify its parts. Subtasks include:
• Part-to-Whole: The model is asked to identify and segment an object's visible parts, and based on the predicted parts, determine the object label.
• Whole-to-Part: The model is asked to identify the object in the image, and based on the predicted object, identify and segment its visible parts.
The subtasks assess decomposable object recognition, where an object and its parts each provide evidence for the other's identity.
this section cite: []

Section: Dataset Construction
We introduce the PARTONOMY dataset to facilitate training and evaluation on Explanatory Part Segmentation. PARTONOMY consists of three training and evaluation subsets-PARTONOMY-PACO, PARTONOMY-PartImageNet, and PARTONOMY-PASCAL Part-which are constructed from their respective datasets' part annotations [35,11,7]. We further contribute an evaluation-only subset of 1K images of domain-specific objects, which we term PARTONOMY-Core.
PARTONOMY-Core Construction. To construct the PARTONOMY-Core ontology, we start from broad object categories containing decomposable objects-for example, airplanes, garden tools, weapons, and boats (details on dataset construction in the Appendix). We then manually select objects which provide category coverage and which have readily identifiable parts.
With object classes selected, we use the Bing search API to download a preliminary set of object images, and prompt an LLM (Llama 2-70B [47]) to generate part names for each object which are visible and specific to that object or category. We manually review each object's assigned parts, removing those which are not outwardly visible or are not commonly found on the object. Part annotation proceeds using a combination of CVAT.AI and a mask annotation interface we developed to streamline the annotation process from multiple annotatorsfoot_0 . Parts are further refined and pruned during the annotation process depending on their visibility and frequency.
this section cite: ['b34', 'b10', 'b6', 'b46']

Section: Explanatory Part Segmentation Data Generation Pipeline.
Our Explanatory Part Segmentation data generation pipeline is applicable to any part dataset containing object names, part segmentations, and part names. We start with the ground truth set of object parts for each image and format these in natural language as the parts the model must identify and ground. For Part Comparison questions, we sample a separate object class with parts in common with those in the image, then intersect (for Part Intersection) or subtract (for Part Difference) its parts to form the ground truth set of parts.
After constructing the answer choice with the ground-truth parts, we create incorrect answer choices for each question.
We adopt an answer mutation framework to generate plausible, challenging wrong answers. For a set of ground truth parts, we repeatedly apply mutation operations which add, remove, or replace an existing part. This process keeps wrong answers close to the original to require deep part understanding of the evaluated model. Instead of randomly sampling parts for mutation operations-which could result in unrelated part additions (e.g. "The airplane in the image has wings, a row of windows, and an ice cream cone")-we select those most related to the existing parts and object. We train logistic regressors on part co-occurrence to predict likely parts given the current set of parts, and restrict wrong answer parts to those from the same object category, if available (e.g., wrong parts for an airplane come from other airplanes' parts). Challenging wrong object answers for Part-Whole Reasoning questions are sampled in a similar way, selecting objects with high semantic similarity to the ground truth object as measured by word embeddings (we use Sentence Transformers [39] to measure similarity).
Differences from Existing Datasets. PARTONOMY is the only part segmentation dataset designed for use with VLMs, testing not only part identification but also part reasoning and grounding. The data generation pipeline's extensibility and capacity to generate challenging questions serve as an important asset for future part-based pretraining and evaluation.
PARTONOMY-Core has, to our knowledge, the most part classes of any part segmentation dataset, with four times the number of part labels of the widely used PACO dataset [35] and more part labels than PartImageNet++ [22] which has twice as many object classes. It has object-centric images for consistent evaluation, unlike datasets like PACO, which frequently have partially occluded objects. PARTONOMY-Core is lightweight to evaluate on, with only 1K images, but covers a wide range of concepts with balanced instances (2 images per object), more than any dataset other than PartImageNet++. These qualities, coupled with PARTONOMY-Core's use of technical domains with highly-specific objects (e.g., electric coffee grinder, city map, and combat drone), make it a unique contribution for part segmentation evaluations.
this section cite: ['b38', 'b34', 'b21']

Section: Evaluation

this section cite: []

Section: Explanatory Part Segmentation requires models to choose the correct textual response and segment parts in the image.
Text Evaluation To evaluate textual part predictions, we prefer a multiple choice over a generative setting to avoid ambiguities in phrasing-e.g., where the model identifies a clip but the annotations list the part as a clamp-and incomplete part annotations, e.g., where the model identifies a valid part that isn't annotated. We provide the model with one correct and four incorrect answer choices. Incorrect choices either include non-visible parts or lack visible parts present in the correct response.
We select the predicted answer choice via language modeling probability, as is common in VQA [2,21]. For Part-Whole Reasoning questions, we select an answer choice twice in sequence: once to predict the set of parts, and once to predict the object.
We evaluate answer selection via accuracy (random = 20% for 5 answer choices) for both part prediction (all questions) and object prediction (Part-Whole Reasoning questions). However, some wrong answer choices are better than others. Precision and recall capture the similarity of the predicted parts to the ground truth set, and we adopt these as more fine-grained measurements of part recognition by the LMMs.
Large Language Model (LLM) Q: What parts does this [banana boat] have in common with a [fishing boat] ? Text Embeddings Mask Decoder A: It has an inflation valve, a seating tube, and a handle {ℎ 𝑖 𝐿 } 𝑖 𝑁 Focal-Tversky Loss Hard Easy Hard Bidirectional Self-Attention Block (Span Extractor) Mask Decoder Projector 𝛁 𝜽 𝓛 CLIP Encoder Mask Encoder It … inflation valve seating tube handle … … O O B I O O B I O B I O Feedback Loop PLUM Downsampled Feature Maps Mask Encoder + FiLM Convolution LayerNorm FiLM GeLU Span Embedding (t=0) Mask (t=0) … Convolution LayerNorm FiLM GeLU Mask Encoder + FiLM Convolution LayerNorm FiLM GeLU Span Embedding (t=n) Mask (t=1) Convolution LayerNorm FiLM GeLU … Feedback Loop Patch-wise Attention Pool Q Figure 3: Overview of PLUM. PLUM is not dependent on special tokens (e.g., <SEG>) added during finetuning to generate segmentation masks. PLUM uses a bidirectional span extractor that automatically determines which tokens should be passed to the mask decoder to generate segmentations. A feedback loop based on SAM's mask decoder enables PLUM to condition future segmentations on those past.
this section cite: ['b1', 'b20']

Section: Segmentation Evaluation
We evaluate part segmentations via gIoU (global IoU), which measures the average IoU over part annotations [19,37]. micro-gIoU averages part IoUs over all masks in the dataset, measuring how well the model segments the most common parts. macro-gIoU averages part IoUs for each image, then averages these image IoUs over the entire dataset. This metric is less affected by common parts, measuring how well a model segments parts in general.
this section cite: ['b18', 'b36']

Section: PLUM: Part-Level Understanding LMM

this section cite: []

Section: Shortcomings of LMMs on Part Understanding
We find that existing LMMs are unable to accurately identify parts in an image. Even segmenting LMMs [19,37,41] trained on part segmentation datasets such as PACO [35] and Pascal-Part [7] exhibit poor performance on part-level segmentation (Fig. 4 and Table 2). We identify two key architectural deficiencies of segmenting LMMs: (1) They rely on special tokens for segmentation (e.g. [SEG] or <p></p>). These tokens are not seen during pretraining, so we hypothesize that their addition to the vocabulary and subsequent finetuning perturbs models' original token distributions (Table 5). (2) They discard prior mask predictions when segmenting in sequence, conditioning only on past text during generation. Incorporating prior mask predictions would likely help maintain consistency and better localize future predictions (Fig. 5a).
this section cite: ['b18', 'b36', 'b40', 'b34', 'b6']

Section: Proposed Method
Based on these observations we propose PLUM, a segmenting LMM with part-level understanding. PLUM consists of a vision-language model (initialized from LLaVA [28]) which takes image and text inputs, along with a mask decoder (initialized from SAM's decoder [17]) that generates segmentation masks.
Let h L i ∈ R d be the VLM's last-layer embedding of token i (i = 1, . . . , N ) of the output sequence. We process these embeddings along two complementary pathways: (i) the Span Extractor, a bidirectional self-attention block that tags beginning (B), inside (I), and outside (O) [36] positions of tokens to segment, and (ii) a projection head that maps B/I embeddings into "mask queries," regularized by KL divergence. An overview is given in Fig. 3.
this section cite: ['b27', 'b16', 'b35']

Section: Token-level Query Selection (Span Extractor)
The Span Extractor enables the selection of segmentation-relevant text spans to pass to the mask decoder without the use of a dedicated segmentation token. Given the last-layer token embeddings {h L i } N i=1 , we apply a two-layer token-wise MLP before infusing global context by passing the embeddings through a bidirectional Transformer encoder block. This bidirectional attention is critical for reliable BIO span tagging, as otherwise the LLM's causal masking prevents embeddings from seeing future context. A final projection layer maps these contextualized embeddings to {B, I, O} logits. We train the span extraction module using cross entropy loss L span where B, I tags correspond to part names to segment.
During inference, contiguous B → I chains are greedily merged to form text spans that are projected into segmentation queries. Note, we also enable users of PLUM to override the automatic tags with manual span selection, enabling interactive, interpretable "highlight-to-segment" behavior.
Query Projection with KL Constraint Let S = {(i s , j s )} N+ s=1 be the set of contiguous B → I spans produced by the span extractor, where i s and j s are the start and end token indices of span s, and N + = |S| is the total number of such spans in the sequence. For every span token k ∈ [i s , j s ] we obtain a "mask-query" vector q k = g(h L k ) ∈ R m , with g(•) a learned MLP projection. To keep the span representations close to the pre-trained backbone VLM's manifold, we pool the last-layer embeddings of each span 3 and impose a Gaussian KL penalty against the corresponding frozen teacher embedding t L is:js :
L KL = 1 N+ N+ s=1 h L is:js -t L is :js 2 2 2σ 2
. This term is applied only to B/I spans, preventing their hidden states from drifting away from the original language-representation space and thereby preserving the VLM's textual reasoning ability.
this section cite: []

Section: Mask Feedback Loop
To incorporate previously predicted masks into the mask decoding process, we inject feature-wise linear modulation (FiLM) [34] layers into the SAM decoder's mask encoder (Fig. 3). These layers allow us to encode the mask while conditioning on prior text spans, providing semantics beyond a raw binary mask. We use this modified mask encoder to encode each prior mask into a stack of text-enhanced feature maps. The stack of feature maps is pooled into a single feature map via patch-wise attention pooling (over the stack dimension), with a learned feature map providing an attentional query for each patch. This pooled feature map representing all prior predicted masks is fed into the mask decoder (along with the pooled text embeddings) to generate the next mask.
Segmentation Mask Generation Tagged token embeddings q k are average pooled and passed to the mask decoder, generating a mask Mi . With ground-truth mask M i we adopt the Focal-Tversky loss [1],
L seg = 1 N+ yi̸ =O L FT (M i , Mi ).
Focal-Tversky loss is a generalized version of the DICE loss [45]. This gives the overall objective equal to
L = L LM + λ 1 L span + λ 2 L KL + λ 3 L seg + λ 4 L BCE ,
where L LM is the standard language-generation loss and L BCE is per-pixel binary cross-entropy as adopted from [19]. The BIO head precisely extracts segmentation spans, while the Focal-Tversky loss, biased toward recall (α = 0.7) and precision (β = 0.3), encourages sharper, high-IoU masks at fine-grained image regions. For additional details on the hyperparameter setting, refer to §A.1
this section cite: ['b33', 'b0', 'b44', 'b18']

Section: Experiments
Implementation Details We use a pre-trained LMM, LLaVA-7B, and LLaVA-llama2-13B [28] as backbones for PLUM (Sec. 4). PLUM follows the consecutive two-stage finetuning process: (1) PLUM is first finetuned with a randomly sampled mixture of PACO-LVIS [35], Pascal Parts [7], PartImageNet [11], COCO-Stuff [5], ADE20k [56], the RefCOCO line of datasets [15], a VQA dataset from LLaVA (llava_instruct_150k), and a Reasoning Segmentation [19] dataset. This setting is similar to the previous line of segmentation-enabled LMMs [19,37,41], and we refer to the stage-1 checkpoint of PLUM as the zero-shot (or pretrained) baseline throughout this paper. (2) To further finetune PLUM on our PARTONOMY training dataset, we take PARTONOMY-PACO, -PartImageNet and -PascalParts to construct a training split and a validation split. Note, we do not use the PARTONOMY-Core split as training data and use it only as evaluation data. We refer to the Appendix for additional details on the hyperparameter settings and training details.
Baselines To evaluate PLUM's proposed changes, we use LISA [19], GLaMM [37], and PixelLM [41] as our primary baselines. All of these models use LLaVA as the base LMM and use SAM-style decoders to generate segmentation masks [28,17].
For segmentation, we also evaluate X-Decoder, SEEM, and Grounded SAM 2 as general openvocabulary segmentation models [59,60,40,38]. As they do not understand question-based prompts, we provide them with the ground truth (gt) parts to segment individually. We also include SegLLM, a segmenting LMM based on LLaVAv1.5 and HIPIE [27,50]. HIPIE is a decoder built for multi-scale and part segmentation, and SegLLM is trained to perform multi-round segmentation on parts, allowing it to refer back to previously predicted masks. The similarity of this mechanism to our feedback loop motivates us to include SegLLM as a baseline, despite the imperfect comparison due to its use of a newer LMM and different mask decoder. For text evaluations, we include a random baseline (which randomly selects answers) to situate the models' part precision and recall. GPT-4o 4 , a frontier model, provides an upper bound on performance. GPT-4o has an advantage as it must be provided with all four answer choices at once, allowing it to take advantage of shortcuts not available to the other models (like identifying the base answer from which the wrong answer choices are generated).
this section cite: ['b27', 'b34', 'b6', 'b10', 'b4', 'b55', 'b14', 'b18', 'b18', 'b36', 'b40', 'b18', 'b36', 'b40', 'b27', 'b16', 'b58', 'b59', 'b39', 'b37', 'b26', 'b49']

Section: Explanatory Part Segmentation
Part Identification and Comparison Questions In Table 2, PLUM outperforms LISA and GLaMM on all three part-segmentation question types in the zero-shot setting. We attribute this gain to (i) span-level constraints that keep pre-trained textual semantics intact and (ii) our mask-feedback loop, which refines each mask using its visual history. By contrast, Table 3 shows only marginal gaps in text-only metrics (P, R, Acc.). PARTONOMY's answer choices intentionally contain extensive lexical overlap, demonstrating the language models' difficulties in comparing similar answer choices. Part-Whole Reasoning Questions The Part-Whole Reasoning results of Table 2 show that knowing the object label prior to part segmentation leads to better mask prediction performance-the pretrained models obtain higher Whole2Part than Part2Whole scores. This suggests that part mask prediction benefits from object label conditioning. The advantage obtained by object conditioning evaporates once the models have been trained on sufficient part data, however, as shown by the finetuned models.
Similarly, in Table 3 the models' increase in object accuracy after conditioning on the object's parts, and their increase in part accuracy after conditioning on the object label, underscores the utility of jointly predicting object labels and parts.
this section cite: []

Section: Additional Downstream Tasks and Ablation Study
We further evaluate PLUM on non part-centric downstream tasks to evaluate its generalizability and whether our choice to omit special [SEG] tokens preserves PLUM's pretraining knowledge. For segmentation, we choose the Reasoning Segmentation task [19], which requires the model to reason about the referenced object before segmenting it. To evaluate PLUM's general visual reasoning, we select the Visual Question Answering (VQA) tasks TextVQA [43] and GQA [13], and a visual hallucination task, POPE [23].
Reasoning Segmentation PLUM demonstrates strong generalization to the Reasoning Segmentation task proposed in [19]. As shown in Table 4, PLUM outperforms existing open-vocabulary segmentation models such as X-Decoder and OVSeg, and also generates more accurate segmentations than LISA, the model proposed for the task.
this section cite: ['b18', 'b42', 'b12', 'b22', 'b18']

Section: Distribution Shift Induced by Special Tokens
To investigate whether PLUM's removal of special tokens mitigates distribution shift from LLaVA's pre-trained representations, we compare PLUM's performance on VQA [43,13,23] tasks to those of models which use [SEG] tokens and to that of the base LLaVA model Table 5. To our surprise, the performance of state-of-the-art segmenting LMMs deteriorates significantly on VQA and visual hallucination [23] tasks. Through inspection, we find that LISA tends to only generates irrelevant [SEG] tokens. While PixelLM performs somewhat better, it still suffers due to its use of multiple specialized tokens [41]. In contrast, PLUM outperforms the other segmenting LMMs, even outperforming the LLaVA backbone on 2/3 tasks. This strong performance demonstrates that our paradigm of segmentation through span extraction preserves the visual reasoning capacity of the LMM. (b) Effect of λKL on segmentation (PartIma-geNet) and visual reasoning (TextVQA). Ablating Key PLUM Components Removing the iterative mask-feedback loop lowers micro-gIoU by 9.6% and macro-gIoU by 8%, though the span-based tag extractor alone still beats the LISA-13B baseline by 8.4% macro-gIoU. When both components are active, PLUM tops LISA by 3.5% (micro) and 20% (macro), showing that tag extraction broadens long-tail coverage while feedback refines segmentation accuracy. Effect of KL Divergence Increasing the KL-alignment weight λ KL from 0 to 1.0 steadily trades segmentation for reasoning: PartImageNet micro-gIoU falls by nearly 20%, whereas TextVQA accuracy sees a 75% performance improvement. We set λ KL = 0.1 in this work.
this section cite: ['b42', 'b12', 'b22', 'b22', 'b40']

Section: Limitations and Broader Impacts
Limitations Our work advances fine-grained, part-level understanding but still faces several constraints. Although PARTONOMY-Core includes the largest number of part labels to date, with 862 categories across 534 objects, it omits some rare or domain-specific parts and lacks the object diversity of [11]. Expanding its coverage would further enhance LMM comprehension. While PLUM mitigates major architectural limitations of prior segmenting LMMs via BIO tagging and a FiLM-based feedback loop, it still struggles with small or ambiguous parts and may not scale efficiently to high-resolution imagery.
Broader Impacts Part grounding is crucial for compositional visual reasoning and safety-critical domains such as robotic manipulation, assistive systems, and automated inspection. Such scenarios require accurate and interpretable part understanding, with mistakes in identifying and the grounding of parts leading to catastrophic consequences. By introducing PARTONOMY and PLUM, we aim to foster more reliable and interpretable LMMs. The high computational cost of large segmenting LMMs also underscores the need for more efficient, sustainable architectures. We hope our benchmark and analyses inspire continued research toward robust, efficient, and responsible part-grounding methods.
this section cite: ['b10']

Section: Conclusion
We introduce Explanatory Part Segmentation with the PARTONOMY benchmark to evaluate part-level visual reasoning and segmentation at scale. PARTONOMY spans 606 object labels and 2,507 part labels; its evaluation split, PARTONOMY-Core, alone contributes 534 objects, 862 unique parts, 1,068 images and 4,968 pixel masks-tripling PACO's object diversity and quadrupling its part vocabulary. By analysing current segmentation-enabled LMMs, we pinpoint two systemic flaws-(i) distribution-shift from pre-trained weights and (ii) discarded visual context-and address them with PLUM, a span-tagging, mask-feedback LMM that interleaves textual and visual reasoning without extra tokens. PLUM outperforms fine-tuned LISA-13B on ReasonSeg and adds 31.8% relative performance improvement on TextVQA compared to the LLaVA-13B backbone. Together, PARTONOMY and PLUM lay a quantitative and methodological foundation for future research on fine-grained, compositional, and interpretable multimodal models. Airplane Attack Agriculture Tankers Kitchen Knife Coffee Maker Refrigerator Landing Gear Spraying Rig Drip Tray Water Tank Concept-Part Hierarchical Ontology Question-Answer Mutator Framework Explanatory Segmentation Tasks Difference Q: What parts does this {object} have in common with {ground-truth-label} and NOT with {contrastive label} A: The object has the following parts in common with {ground-truth-label}: {part_1}, ... {part_K} Input Image Segmenting LMM A: The object has the following parts in common with {ground-truth-label}: {part_1}, ... {part_K} {part_1}, ... {part_K} → {part_1}, ... {part_K'} Mutation Operations Correct Answer {part_1}, ... {part_K} {part_1}, ... {part_K} {part_1}, ... {part_K}, {part_K+1} Deletion Insertion Substitution Positive Seg. Inpu Segme (e.g., LIS Bottom Intersection Segmentation Q: What parts does this {ground-truth-label} have in common with {contrastive label} A: It has: {part_1}, ... {part_K} Negative Segmentation Q: What parts does this {ground-truth-label} NOT have in common with {contrastive label} A: It has: {part_1}, ... {part_K} Whole2Part Q1: What is the name of this object? A1: This is a {predicted-truth label} Q2: Okay, then what parts make this look like a {predicted-label}? A2: It has: {part_1}, ... {part_K} Part2Whole Q1: What parts of this object highlight its most distinctive parts? A1: It has: {part_1}, ... {part_K} Q2: Okay, based on the parts you observed/highlighted, what is the name of this object? A2: It's a {predicted-truth label} The total loss is a weighted sum of (i) language modeling cross-entropy, (ii) BIO span classification loss, (iii) Focal-Tverskyfoot_3 and pixel-wise BCE for masks, (iv) KL divergence to a frozen teacher snapshot of LLaMA. Loss weights follow Table 6; random seed is fixed to 42.
For model evaluation, we first divide the performance evaluation to two facets: (i) pixel-level mask prediction evaluation as in Table 2, and (ii) multiple choice answer selection evaluation. Note, for multiple choice answer selection, we take the argmin over the entropy of each answer choice (i.e., the argmax over the sequence-level probability of each answer choice), and greedily select the answer choice with the lowest entropy. For pixel-level mask prediction, we provide the ground-truth answer choice and their part text (or [SEG] for LISA, PixelLM and GLaMM) so that the models can be evaluated solely on their mask prediction performance per part text.
this section cite: []

Section: References
Ref_id:b0 Title: A novel focal tversky loss function with improved attention u-net for lesion segmentation Year: (2019)
Ref_id:b1 Title: Vqa: Visual question answering Year: (2015)
Ref_id:b2 Title: Miracle: An online, explainable multimodal interactive concept learning system Year: (2024)
Ref_id:b3 Title: ACCESS: Advancing Innovation: NSF's Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support Year: (2023-07)
Ref_id:b4 Title: Coco-stuff: Thing and stuff classes in context Year: (2018)
Ref_id:b5 Title: Divide and conquer: Answering questions with object factorization and compositional reasoning Year: (2023-06)
Ref_id:b6 Title: Detect what you can: Detecting and representing objects using holistic models and body parts Year: (2014)
Ref_id:b7 Title: InstructBLIP: Towards general-purpose vision-language models with instruction tuning Year: (2023)
Ref_id:b8 Title: The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results Year: ()
Ref_id:b9 Title: Novel concept design with affordance composition Year: (2025)
Ref_id:b10 Title: Partimagenet: A large, high-quality dataset of parts Year: (2022)
Ref_id:b11 Title: Multi-modal latent space learning for chain-of-thought reasoning in language models Year: (2024)
Ref_id:b12 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b13 Title: Robot manipulation in salient vision through referring image segmentation and geometric constraints Year: (2024)
Ref_id:b14 Title: ReferItGame: Referring to objects in photographs of natural scenes Year: (2014-10)
Ref_id:b15 Title: Finer: Investigating and enhancing fine-grained visual concept recognition in large vision language models Year: (2024-11)
Ref_id:b16 Title: Segment anything Year: (2023)
Ref_id:b17 Title: Large language models are zero-shot reasoners Year: (2022)
Ref_id:b18 Title: Reasoning segmentation via large language model Year: (2024)
Ref_id:b19 Title: Semantic-sam: Segment and recognize anything at any granularity Year: (2023)
Ref_id:b20 Title: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation Year: (2022)
Ref_id:b21 Title: Partimagenet++ dataset: Scaling up part-based models for robust recognition Year: (2024)
Ref_id:b22 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b23 Title: Open-vocabulary semantic segmentation with mask-adapted clip Year: (2023)
Ref_id:b24 Title: Gres: Generalized referring expression segmentation Year: (2023)
Ref_id:b25 Title: Mitigating hallucination in large multi-modal models via robust instruction tuning Year: ()
Ref_id:b26 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b27 Title: Visual instruction tuning Year: (2023)
Ref_id:b28 Title: Democratizing fine-grained visual recognition with large language models Year: (2024)
Ref_id:b29 Title: Composable part-based manipulation Year: (2024)
Ref_id:b30 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b31 Title: Compositional chain-of-thought prompting for large multimodal models Year: (2024)
Ref_id:b32 Title: Synthesize diagnose and optimize: Towards fine-grained vision-language understanding Year: (2024)
Ref_id:b33 Title: Film: Visual reasoning with a general conditioning layer Year: (2018)
Ref_id:b34 Title: Paco: Parts and attributes of common objects Year: (2023)
Ref_id:b35 Title: Text chunking using transformation-based learning Year: (1999)
Ref_id:b36 Title: Glamm: Pixel grounding large multimodal model Year: (2024)
Ref_id:b37 Title: Segment anything in images and videos Year: (2024)
Ref_id:b38 Title: Sentence-bert: Sentence embeddings using siamese bert-networks Year: (2019)
Ref_id:b39 Title: Grounded sam: Assembling open-world models for diverse visual tasks Year: (2024)
Ref_id:b40 Title: Pixellm: Pixel reasoning with large multimodal model Year: (2024)
Ref_id:b41 Title: Aokvqa: A benchmark for visual question answering using world knowledge Year: (2022)
Ref_id:b42 Title: Towards vqa models that can read Year: (2019)
Ref_id:b43 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b44 Title: Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations Year: (2017-09-14)
Ref_id:b45 Title: Vipergpt: Visual inference via python execution for reasoning Year: (2023)
Ref_id:b46 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b47 Title: Attention is all you need Year: (2017)
Ref_id:b48 Title: Llm-seg: Bridging image segmentation and large language model reasoning Year: (2024)
Ref_id:b49 Title: Hierarchical open-vocabulary universal image segmentation Year: (2023)
Ref_id:b50 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b51 Title: Creative robot tool use with large language models Year: (2023)
Ref_id:b52 Title: Osprey: Pixel understanding with visual instruction tuning Year: (2024-06)
Ref_id:b53 Title: A benchmark for compositional visual reasoning Year: (2022)
Ref_id:b54 Title: Multimodal chain-of-thought reasoning in language models Year: (2023)
Ref_id:b55 Title: Scene parsing through ade20k dataset Year: (2017)
Ref_id:b56 Title: Semantic understanding of scenes through the ade20k dataset Year: (2019)
Ref_id:b57 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b58 Title: Generalized decoding for pixel, image, and language Year: (2023)
Ref_id:b59 Title: Segment everything everywhere all at once. Advances in neural information processing systems Year: (2023)
Ref_id:b60 Title: A Part-to-Whole question Year: ()
Ref_id:b61 Title: Figure 7: Examples of the different question types from the Partonomy-Core dataset Year: ()
