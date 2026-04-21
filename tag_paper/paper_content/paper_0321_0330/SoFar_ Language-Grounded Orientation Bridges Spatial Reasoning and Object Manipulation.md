Title: SOFAR: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation
Abstract: Figure 1: We introduce the concept of Semantic Orientation, which refers to natural languagegrounded object orientations, such as the "cutting" direction of a knife or the "handle" direction of a cup. To support this, we construct OrienText300K, a large-scale object-text-orientation pairs dataset.

Section: 
X Y Z Pose Estimation Category / Instance Template Needed Only axis, the relationship with instruction is unclear "Blow Wind" "Top" "Back" "Pick up" "Fan" "Front"
this section cite: []

Section: Semantic Orientation

this section cite: []

Section: Without any template Training on large datasets
Use natural language as inputs directly link to user instructions
this section cite: []

Section: Affordance Map

this section cite: []

Section: Training on limited category datasets

this section cite: []

Section: Only position information Without interact direction

this section cite: []

Section: Introduction
We observe that current VLMs struggle with understanding object orientation, making them insufficient for 6-DoF robot manipulation planning. Consider some everyday scenarios: cutting bread in half with a knife, righting a tilted wine glass, or plugging a cord into a power strip. Previous approaches [10,12,8] primarily focused on understanding "where are the knife and wine glass" while ignoring their orientations-such as the "blade direction" of the knife and the "up direction" of the glass. This oversight makes it challenging to accomplish these 6-DoF manipulation tasks.
More importantly, different orientations of an object hold varying semantic significance. The capability of connecting specific orientations to their semantic meanings is essential for languageguided robot manipulations. For example, inserting a pen into a pen holder requires aligning the pen tip with the direction of the pen holder's opening; righting a wine glass necessitates aligning the glass's top with the z-axis in the world coordinate frame; and plugging into a power strip involves understanding the "insertion" direction, which is perpendicular to the power strip's surface. However, translating a specific language description into a desired orientation is challenging for existing VLMs.
To move forward, we introduce language-grounded orientation that bridges spatial reasoning and object manipulation, characterized by the following:
• From Position Awareness to Orientation Awareness. While prior works [10,12,8] emphasize position relationship, orientation understanding is equally critical for defining the full 6-DoF of object pose or end-effector poses [16,120,124,60]. Orientation awareness involves understanding object orientations and their relationships in the open world, enabling robots to complete tasks requiring precise alignment and rearrangement.
• From Orientation to Semantic Orientation. Traditional orientation, defined relative to a base frame or template model [104,58,120,16], is insufficient for open-world manipulation guided by language instructions [108,49]. We introduce semantic orientation, linking orientational vectors of an object to open-vocabulary prompts (e.g., the "handle" direction of a knife or "plug-in" direction of a USB). This bridges geometric reasoning with functional semantics, enabling robots to interpret task-specific orientation changes.
Achieving such open-world orientation understanding requires rich world knowledge. To this end, we design both the model architecture and the dataset accordingly. We propose PointSO, a generalizable cross-modal 3D Transformer [114,26,89,91] for semantic orientation prediction. To train it at scale, we construct OrienText300K, a large-scale dataset comprising over 350K 3D models with diverse orientation-text pairs. These annotations are from Objaverse [20] and generated automatically by prompting GPT-4o [48] with rich semantic queries covering both intra-object spatial reasoning and inter-object manipulation contexts-eliminating the need for costly robot-collected data.
To enable comprehensive spatial reasoning, we develop SOFAR, an integrated system that combines PointSO with foundation models such as SAM [57]. Given an RGB-D input, SAM segments the scene, and PointSO estimates object orientations to build an orientation-aware 3D scene graph. The graph together with the image is fed into a VLM to generate chain-of-thought [119] spatial reasoning, supporting both positional and orientational planning for downstream robotic manipulation.
In addition, we introduce Open6DOR V2, a large-scale benchmark for 6-DoF object rearrangement in simulation, which supports both open-loop and closed-loop control. Our method significantly outperforms state-of-the-art VLMs and VLA models-even those trained on expensive robot trajectories-across both simulated and real-world tasks. We also introduce 6-DoF SpatialBench, a new spatial visual-question-answering benchmark to rigorously assess orientation-aware reasoning.   In summary, we propose Semantic Orientation as a new representation that bridges spatial reasoning and robotic manipulation, enabling open-vocabulary, template-free orientation understanding for unseen objects. We introduce OrienText300K, a large-scale dataset including 350K diverse objects & orientations and 8M images through careful filtering and annotating. We develop the SOFAR system, which enhances spatial reasoning with 6-DoF scene graph and achieves SOTA performance on Open6DOR, SimplerEnv, and generalizes across embodiments (e.g., grippers, suction cups, dexterous hands) and tasks (e.g., manipulation, navigation, VQA) without any task-specific fine-tuning. Finally, we present two new benchmarks, Open6DOR V2 and 6-DoF SpatialBench, to evaluate 6-DoF rearrangement and spatial reasoning.
2 Semantic Orientation: Connecting Language and Object Orientation
this section cite: ['b9', 'b11', 'b7', 'b9', 'b11', 'b7', 'b15', 'b119', 'b123', 'b59', 'b103', 'b57', 'b119', 'b15', 'b107', 'b48', 'b113', 'b25', 'b88', 'b90', 'b19', 'b47', 'b56', 'b118']

Section: Definition of Semantic Orientation
Traditionally, object orientation is defined within a reference frame using quaternions or Euler angles to describe relative rotations. However, in interactive tasks, orientations often carry semantic meaning. Humans naturally interpret orientation in a semantic, reference-free manner. For example, plugging in a charger involves aligning the metal prongs with the socket's opening direction-a semantically grounded alignment. Motivated by this, we define an object's Semantic Orientation as a unit vector that captures the direction corresponding to a given language description. Formally, for an object X and a description ℓ, the semantic orientation s X ℓ ∈ S(2) is defined as:
s X ℓ = F(X, ℓ).(1)
Here, ℓ is open-vocabulary phrase referring to general directions (e.g., front, top), object parts (e.g., handle, cap), or interactions (e.g., pour out, plug-in). An object X can be associated with multiple semantic orientations by varying the language input, forming a set S X = {s X ℓ1 , s X ℓ2 , . . . , s X ℓn }. These orientations provide a semantic basis for describing and transforming the object's rotation.
this section cite: []

Section: OrienText300K: Orientation-Text Paired Data at Scale
Our goal is to develop an orientation model capable of identifying semantic orientations in open-world settings using large-scale 3D data. To support this, we introduce OrienText300K, a curated dataset of 3D models annotated with diverse language-guided orientation labels. The dataset is constructed from Objaverse [20], which contains approximately 800K Internet-sourced 3D models across a wide range of categories. Since the raw data includes noisy annotations and low-quality samples, we apply a rigorous filtering process. Using Blender, we render over 8M high-quality images under carefully designed lighting conditions to ensure fidelity for training.
this section cite: ['b19']

Section: Data Filtering
To ensure high-quality data for generating semantic orientation annotations, we apply a dedicated filtering strategy that retains only the samples meeting the following six criteria. ❶ Standard orthogonal view only. Samples in random views will be filtered. ❷ Clean objects without the ground for auxiliary visualization. ❸ Reasonable objects that have sufficient spatial reasoning potentials. ❹ High-quality objects. Blurry and wrong samples are filtered. ❺ Distinguishable objects. Abstract and meaningless objects are filtered. ❻ Non-scene objects for object-centric understanding. However, it is non-trivial to conduct filtering on such big data using manual labor. Inspired by recent works showing large VLMs are human-aligned judgers [147,121,85], we employ GPT-4o [48] by prompting requirements above. To be specific, the multi-view images of 3D objects are concatenated together with our designed prompts into GPT-4o, and GPT-4o will decide whether samples should be filtered. The filtered dataset yields 350K+ clean samples, significantly reducing data noise.
this section cite: ['b146', 'b120', 'b84', 'b47']

Section: Data Annotation
As mentioned in the introduction, VLMs struggle to produce accurate object orientation values, which presents a significant challenge for data generation. Fortunately, VLMs are powerful discriminators capable of distinguishing between different views through multimodal understanding. We believe that the initial stage of data cleaning effectively removed a large amount of misaligned data, leaving behind a set of properly aligned instances capable of producing standard orthogonal views. We then leverage GPT-4o to interpret the semantic content across six views and generate semantic-view pairs accordingly. Throughout the annotation process, both human modelers in Objaverse and ChatGPT serve as our annotators, supplying the necessary knowledge to produce both view-aligned data and semantically grounded annotations.
this section cite: []

Section: Quality Validation
To validate annotation quality, we construct a validation set containing 208 samples with manually labeled filtering criteria and semantic orientation labels, respectively. From Fig. 3b, we observe that GPT-4o achieves an average accuracy of 88.3% and 97.1% accuracy on filtering and annotating, respectively. This provides a quality guarantee of our OrienText300K.
this section cite: []

Section: PointSO: A Cross-Modal 3D Transformer for Semantic Orientation Prediction
We introduce PointSO, a plain Transformer-based architecture [114] with cross-modal 3D-language fusion as our orientation model. As illustrated in Fig. 4, PointSO takes the object's 3D point clouds and a language description as inputs, and predicts the corresponding semantic orientation. 3D and Language Embeddings Given an object's point cloud X = {x i ∈ R 3 |i = 1, 2, . . . , N } with N 3D points defined in (x, y, z) Cartesian space, and an arbitrary language description ℓ, we first embed both into discrete token embeddings. For the 3D point clouds, we follow [26,136,89] to first sample N s seed points using farthest point sampling (FPS) and then group inputs with KNN for point feature embedding with a local geometric extraction network such as lightweight PointNet [86,87]. An MLP head is used which maps a special [CLS] token [28] to a predicted direction. As for the language inputs, we adopt CLIP [97] and use the global token as cross-modal fusion inputs.
this section cite: ['b113', 'b25', 'b135', 'b88', 'b85', 'b86', 'b27', 'b96']

Section: Cross-Modal Fusion
We perform cross-modal fusion by injecting global text features into each layer of the 3D Transformer using a simple yet effective strategy: adding the text token to every point token. While other fusion methods such as cross-attention, adapters, or concatenation along spatial or channel dimensions are possible, we empirically find that token-wise addition performs best (see Appendix C.3). This effectiveness may stem from the short language inputs, where summation helps reinforce their influence across layers.
Optimization Let F SO represent the PointSO model parameterized by θ SO (the CLIP is kept frozen and thus its parameters are not included). Given every object point cloud X i ∈ D OrienText300K in the OrienText300K dataset, where each object is labeled with a language set L i = {ℓ i j , j = 1, 2, . . . , Q} and the corresponding ground truth semantic orientation set, S i = {s i j , j = 1, 2, . . . , Q}. The optimization is to minimize the negative cosine similarity L cos (v, k) = 1 -v•k ∥v∥•∥k∥ between predicted and the ground truth semantic orientations:
min θSO Xi∈DOrienText300K ℓ i j ∈Li L cos F SO (X i , ℓ i j ), s i j .(2)
Figure 5: Overview of SOFAR system. Given RGB-D images and language instructions, SOFAR first leverages a VLM to identify relevant object phrases and semantic orientations. Then utilizes foundation models Florence-2 [125], SAM [57], and our PointSO for object segmentation and semantic orientation estimation. This information forms a 6-DoF scene graph, which the VLM uses alongside the RGB image to perform spatial understanding tasks or generate manipulation actions.
this section cite: ['b124', 'b56']

Section: SOFAR: Semantic Orientation Bridges Spatial Reasoning and Object Manipulation
Our proposed PointSO model now paves the way for off-the-shelf object-centric spatial orientation understanding. However, it remains challenging to extend such object-centric spatial understanding for scene-level spatial reasoning both in the digital world (e.g., 6-DoF visual question answering) and in the physical world (e.g., robot manipulations). To bridge this gap, we build an integrated reasoning system where a powerful VLM acts as an agent and reasons about the scene while communicating with off-the-shelf models including PointSO and SAM [57]. Fig. 5 illustrates an overview of our proposed framework, aiming at Semantic Orientation For Autonomous Robots (SOFAR).
this section cite: ['b56']

Section: Scene Graph with 6-DoF Information
To integrate both the positional & orientational interaction relationships of objects, we use a scene graph with 6-DoF information to represent the environment.
this section cite: []

Section: Position & Orientation Information Extraction
Given a language query Q, we first prompt a visionlanguage model F VLM to extract a task-relevant set of object phrases P = {p i | i = 1, 2, . . . , M }. Each phrase p i represents a language description of an object relevant to Q. Using the SAM [57] & Florence-2 [125], we perform language-conditioned segmentation to obtain a corresponding object set X = {X i | i = 1, 2, . . . , M }, where X i is the 3D point cloud of the i-th object. Each object is assigned a unique ID for use in Set-of-Mark (SoM) prompting [129]. We then prompt the VLM to generate a set of task-specific orientation descriptions L i for related objects, and use pretrained PointSO to infer their semantic orientations, resulting in a semantic orientation set S i .
this section cite: ['b56', 'b124', 'b128']

Section: 6-DoF Scene Graph
From the segmented object set X , we construct an 6-DoF scene graph G = (V, E) with M nodes. Each node o i ∈ V encodes the following semantic and spatial attributes: ❶ object phrase p i with a unique instance ID; ❷ 3D position c i = (x, y, z) ∈ R 3 from the object's centroid; ❸ bounding box size b i = (h, w, l) ∈ R 3 ; ❹ semantic orientation set S i along with its corresponding description set L i . Each edge e ij ∈ E represents the relative translation and size ratio between two connected objects o i and o j .
this section cite: []

Section: Spatial-Aware Task Reasoning
We encode the 6-DoF scene graph G into descriptive language and input it to the VLM alongside the RGB image I and query Q. This enriched spatial representation enables the VLM to perform accurate spatial reasoning by leveraging its visual and linguistic understanding.
Chain-of-Thought Spatial Reasoning Most robot manipulation tasks involving rigid objects can be abstracted as applying transformations to adjust their position and orientation. To guide the VLM in generating such transformations from language instructions, we adopt a CoT reasoning process [119] that decomposes the reasoning into three steps: (i) analyzing the scene with the query Q and object 77.1 70.4 33.3 4.2 44.4 3.7 43.3 77.1 63.0 30.6 12.5 50.0 11.1 45.0 81.3 81.5 44.4 20.8 50.0 22.2 53.9 85.4 85.2 52.8 29.2 72.2 33.3 62.2 0 10 20 30 40 50 60 70 80 90 100 Position-Simple Position-Hard Orientation-Simple Orientation-Hard Comprehensive 6-DoF Manipulation Total CoPa ReKep-Auto SoFar-LLaVA SoFar Success Rate (%) 0 5 10 15 20 25 Planning Time Costs (s) 15 tasks, 30 Assets 10 tasks, 24 Assets 12 tasks, 12 Assets 8 tasks, 8 Assets 8 tasks, 44 Assets 7 tasks, 18 Assets 60 tasks, 136 Assets nodes V; (ii) computing the desired position and orientation of the target object; (iii) predicting the target position ci and semantic orientation set Si for each object. Given the initial state c i and S i , the full 6-DoF transformation P i is computed. Specifically, translation is obtained by t i = cic i , and rotation R i is estimated from S i and Si using the Kabsch-Umeyama algorithm [52, 53, 112].
Low-Level Motion Execution Following CoPa [44], we integrate task-specific grasping and motion planning. Object or part segmentation is performed using Florence-2 [125] and SAM [57], followed by grasp candidate generation via GSNet [33]. The optimal grasp is selected by considering both grasp quality and heuristics. Based on instruction, SOFAR predicts the object's translation and rotation, defining the transformation from grasp to placement. We employ OMPL [103] to generate a collision-free trajectory, initializing joint positions at the midpoint to ensure smooth and safe motion. Tasks and Evaluations We construct 60 real-world tasks involving over 100 objects, following the Open6DOR benchmark [25]. The tasks are divided into three tracks-position, orientation, and comprehensive & 6-DoF-each with simple and hard variants. The position track assesses spatial reasoning from basic (e.g., front/back/left/right) to more complex relations (e.g., between/center/custom). The orientation track includes part-level orientation in the simple setting, and fine-grained angle estimation in the hard setting. The comprehensive and 6-DoF tracks evaluate complex instruction understanding and simultaneous control over position and orientation. Each task is repeated three times to ensure statistical robustness. More details and visualizations are available in Appendix D.1.
this section cite: ['b118', 'b43', 'b124', 'b56', 'b32', 'b102', 'b24']

Section: Results
As shown in Fig. 7, SOFAR consistently outperforms baselines across all tracks, especially on orientation and 6-DoF tasks, while maintaining low planning overhead. We also demonstrate SOFAR's embodiment generality with different end-effectors, including dexterous hands and suction cups, as illustrated in Fig. 6. Additional robot setups and generalization results are provided in Appendix A.  Using free-text descriptions to extract semantic orientations from object point clouds is challenging. In Objaverse [20], we manually annotate 128 diverse objects and construct the Ori-enText300K val split to evaluate the directional prediction accuracy of PointSO. We train different model variants on OrienText300K, and the results in Table 2 report performance across different angular thresholds ranging from 45°to 5°. PointSO still has an accuracy rate of 60% even under a 5°threshold.
this section cite: ['b19']

Section: Semantic Orientation Prediction
In the real world, obtaining complete object point clouds is often difficult. To evaluate the robustness of PointSO under such conditions, we introduce three types of input perturbations: random rotations, partial single-sided observations, and Gaussian noise. As reported in Table 3, the accuracy at the 45°threshold reflects the model's resilience to these corruptions.
this section cite: []

Section: 6-DoF Object Rearrangement Evaluation on Open6DOR V2
To evaluate 6-DoF object rearrangement capabilities, we extend the original Open6DOR benchmark [25], which primarily focuses on final pose estimation, into a more comprehensive setting that includes both perception and execution evaluation. We migrate its scenes into a robosuite-based simulation environment [151], following the task interface defined by LIBERO [64], and name this new benchmark Open6DOR V2. Results are reported in Table 1. For perception tasks, we adopt the original Open6DOR [25] evaluation protocol and compare with the same baselines. SOFAR achieves the best performance, demonstrating strong spatial understanding and zero-shot generalization. For execution tasks, we compare against the pretrained Octo [107] and the LIBERO-finetuned Open-VLA [56], all evaluated in the same robosuite environment to minimize domain shift. While both baselines show limited success due to poor generalizability, SOFAR reaches around 40% success rate using a vanilla execution pipeline. We note that certain objects are intrinsically difficult to manipulate, suggesting the need for more robust policies incorporating prehensile grasping and adaptive strategies to improve performance on Open6DOR V2.
this section cite: ['b24', 'b150', 'b63', 'b24', 'b106', 'b55']

Section: Simulation Object Manipulation Evaluation on SIMPLER [62]
We conduct quantitative evaluations of SOFAR's zero-shot execution performance on Google Robot tasks & Widow-X tasks and compare it to baselines including Octo [107], OpenVLA [56] and more concurrent works [61,94]. The robot follows the planned trajectory generated by the planning module, as described in Sec. 3.2, to execute the task. Furthermore, leveraging the error detection and re-planning capabilities of VLMs [48,1], we can make multiple attempts following a single-step execution failure to approximately achieve a closed-loop effect. For fairness, we limit the maximum number of attempts to three. Detailed visualizations and analyses are provided in the Appendix B.5. As shown in Tables 4 and 5, despite the training data for Octo and OpenVLA including Google Robot tasks, SOFAR demonstrates superior zero-shot performance compared to most baselines.
Figure 8: Real-world orientation-aware navigation. We present both the third-person view and the egocentric view, annotating the predicted orientation of the interacted objects.
this section cite: ['b106', 'b55', 'b60', 'b93', 'b47', 'b0']

Section: Orientation-Aware Robotic Navigation
In navigation tasks, reaching an object from its functional side is crucial for subsequent manipulation-for example, approaching a microwave from the front to open its door. To support such scenarios, we extend semantic orientation to the navigation domain. As shown in Fig. 8, a quadruped robot is tasked with reaching both the correct position and the appropriate facing direction. This orientation-aware constraint enhances the navigation process by ensuring precise alignment with the desired orientation, thereby improving task performance in scenarios where directionality is critical.
this section cite: []

Section: Spatial Reasoning Evaluation on 6-DoF SpatialBench
Table 6: Spatial reasoning evaluation on 6-DoF SpatialBench. Depth-Esti: Use depth estimation methods such as Metric3D [135] or Moge [117].
this section cite: ['b134', 'b116']

Section: Method Depth-Esti
Position Orientation Total rel. abs. rel. abs.
this section cite: []

Section: Blind Evaluation with LLMs

this section cite: []

Section: GPT-3.5-Turbo [7]
✗ 24.5 24.9 26.7 27.5 25.7 GPT-4-Turbo [82] ✗ 27.2 27.3 29.2 27.9 27.8
this section cite: ['b81']

Section: General VLMs
LLaVA-1.5 [68] ✗ 30.9 24.5 28.
3 25.8 27.2 GPT-4o-mini [48] ✗ 33.3 26.9 32.5 23.8 31.0 GPT-4o [48] ✗ 49.4 28.4 44.2 25.8 36.2 VLMs with Spatial Awareness SpaceLLaVA [10] ✗ 32.4 30.5 30.9 24.9 28.2 SpaceMantis [10] ✗ 33.6 29.2 27.2 25.0 28.9 SpatialBot [8] ✓ 50.9 21.6 39.6 22.9 32.7 RoboPoint [137] ✗ 43.8 30.8 33.8 25.8 33.5 SOFAR ✓ 59.6 33.8 54.6 31.3 43.9
To assess spatial understanding with full 6-DoF awareness, we introduce 6-DoF SpatialBench, a VQA benchmark designed to evaluate both positional and orientational comprehension. Unlike prior benchmarks [12,8,29,106] that primarily emphasize coarse positional reasoning (e.g., "to the left," "nearest") and often overlook orientation or rely on relative metrics, we provide a more fine-grained evaluation with quantitative annotations. It consists of 223 humanannotated samples, each containing an RGB image and a multiple-choice question with 4 options. The benchmark includes two tracks: position and orientation, covering tasks such as object counting, spatial relations, and objectfacing direction. All questions and ground-truth answers are curated through human annotation. We evaluate SOFAR on 6-DoF Spatial-Bench against several VLMs and comparable methods as baselines, as presented in Table 6. SO-FAR consistently outperforms other methods across both tracks, achieving over 18% improvement.
this section cite: ['b11', 'b7', 'b28', 'b105']

Section: Limitations & Conclusions
One notable limitation for decoupled systems like SOFAR is that the execution may fail due to a sub-module error, as shown in Appendix B.8, i.e., robots may place target objects with an error transformation because of unstable grasping or inaccurate visual perception. For example, the pen will be placed in an unexpected pose due to the rotation during execution. Future works include integrating scalable data and more advanced models and exploring the potential of combining end-to-end and such decoupled methods, and expanding SOFAR to more applications.
We propose semantic orientation, a language-grounded representation that links object orientations with intuitive descriptors (e.g., "plug-in direction"), bridging geometric reasoning and functional semantics. To support this, we construct OrienText300K, a large-scale dataset of 3D models with semantic orientation annotations. Our PointSO model, integrated within the SOFAR system, demonstrates strong performance in both simulated and real-world robotic manipulation tasks. Justification: N/A Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: ()
Ref_id:b1 Title: 3d semantic parsing of large-scale indoor spaces Year: (2016-06-27)
Ref_id:b2 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b3 Title: RT-H: action hierarchies using language Year: (2024)
Ref_id:b4 Title: On the opportunities and risks of foundation models Year: ()
Ref_id:b5 Title: RT-1: robotics transformer for real-world control at scale Year: (2023)
Ref_id:b6 Title: Language models are few-shot learners Year: (2020)
Ref_id:b7 Title: Spatialbot: Precise spatial understanding with vision language models Year: (2024)
Ref_id:b8 Title: GOAT: GO to any thing Year: (1938)
Ref_id:b9 Title: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities Year: (2024)
Ref_id:b10 Title: Koushil Sreenath, Saurabh Gupta, and Xue Bin Peng. Learning smooth humanoid locomotion through lipschitz-constrained policies Year: ()
Ref_id:b11 Title: Spatialrgpt: Grounded spatial reasoning in vision-language models Year: (2009)
Ref_id:b12 Title: Yolo-world: Real-time open-vocabulary object detection Year: (2024)
Ref_id:b13 Title: Expressive whole-body control for humanoid robots Year: (1938)
Ref_id:b14 Title: Open x-embodiment: Robotic learning datasets and RT-X models Year: (2023)
Ref_id:b15 Title: Open-vocabulary object 6d pose estimation Year: (2024)
Ref_id:b16 Title: Scannet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017-07-21)
Ref_id:b17 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning Year: (2023)
Ref_id:b18 Title: Objaverse-xl: A universe of 10m+ 3d objects Year: (2023)
Ref_id:b19 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b20 Title: Voxel R-CNN: towards high performance voxel-based 3d object detection Year: (2021)
Ref_id:b21 Title: BERT: pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b22 Title: PLA: language-driven open-vocabulary 3d scene understanding Year: (2023)
Ref_id:b23 Title: Lowis3d: Languagedriven open-world instance-level 3d scene understanding Year: (2024)
Ref_id:b24 Title: Open6dor: Benchmarking open-instruction 6-dof object rearrangement and A vlm-based approach Year: (2024)
Ref_id:b25 Title: Autoencoders as cross-modal teachers: Can pretrained 2d image transformers help 3d representation learning? Year: (2023)
Ref_id:b26 Title: DreamLLM: Synergistic multimodal comprehension and creation Year: ()
Ref_id:b27 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: ()
Ref_id:b28 Title: Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models Year: (2024)
Ref_id:b29 Title: Flowbot3d: Learning 3d articulation flow to manipulate articulated objects Year: (2022-07-01)
Ref_id:b30 Title: Point transformer Year: (2021)
Ref_id:b31 Title: Point-gcc: Universal self-supervised 3d scene pre-training via geometry-color contrast Year: (2024-10-28)
Ref_id:b32 Title: Graspnet-1billion: A large-scale benchmark for general object grasping Year: (2020)
Ref_id:b33 Title: MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting Year: (2024-07)
Ref_id:b34 Title: Scene-llm: Extending language model for 3d visual understanding and reasoning Year: (2024)
Ref_id:b35 Title: The theory of affordances Year: (1979)
Ref_id:b36 Title: Human hands as probes for interactive object understanding Year: (2022)
Ref_id:b37 Title: MVTN: multi-view transformation network for 3d shape recognition Year: (2021)
Ref_id:b38 Title: Dexvlg: Dexterous vision-language-grasp model at scale Year: (2025)
Ref_id:b39 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b40 Title: Omnih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning Year: (2024)
Ref_id:b41 Title: Learning getting-up policies for real-world humanoid robots Year: ()
Ref_id:b42 Title: 3d-llm: Injecting the 3d world into large language models Year: (2023-12-10)
Ref_id:b43 Title: Copa: General robotic manipulation through spatial constraints of parts with foundation models Year: (2024)
Ref_id:b44 Title: A3VLM: actionable articulation-aware vision language model Year: (2024-11-09)
Ref_id:b45 Title: Voxposer: Composable 3d value maps for robotic manipulation with language models Year: (2023)
Ref_id:b46 Title: Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation Year: (2024)
Ref_id:b47 Title:  Year: ()
Ref_id:b48 Title: Do as I can, not as I say: Grounding language in robotic affordances Year: (2022-12-18)
Ref_id:b49 Title:  Year: ()
Ref_id:b50 Title: Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models Year: ()
Ref_id:b51 Title: A solution for the best rotation to relate two sets of vectors Year: (1976)
Ref_id:b52 Title: A discussion of the solution for the best rotation to relate two sets of vectors Year: (1978)
Ref_id:b53 Title: Dream2real: Zero-shot 3d object rearrangement with vision-language models Year: (2024)
Ref_id:b54 Title:  Year: ()
Ref_id:b55 Title: Openvla: An open-source vision-language-action model Year: ()
Ref_id:b56 Title: Segment anything Year: (2023)
Ref_id:b57 Title: Megapose: 6d pose estimation of novel objects via render & compare Year: (2022-12-18)
Ref_id:b58 Title: BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b59 Title: Manipllm: Embodied multimodal large language model for object-centric robotic manipulation Year: (2024)
Ref_id:b60 Title: Towards generalist robot policies: What matters in building visionlanguage-action models Year: (2024)
Ref_id:b61 Title: Evaluating real-world robot manipulation policies in simulation Year: (2024-11-09)
Ref_id:b62 Title: Code as policies: Language model programs for embodied control Year: (2023-06-02)
Ref_id:b63 Title: LIBERO: benchmarking knowledge transfer for lifelong robot learning Year: (2023)
Ref_id:b64 Title: 3daxiesprompts: Unleashing the 3d spatial task capabilities of GPT-4V Year: (2023)
Ref_id:b65 Title: Instruction-following agents with jointly pre-trained vision-language models Year: ()
Ref_id:b66 Title: Visual instruction tuning Year: (2023)
Ref_id:b67 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b68 Title: Grounding DINO: marrying DINO with grounded pre-training for open-set object detection Year: (2024-10-04)
Ref_id:b69 Title: HOI4D: A 4d egocentric dataset for category-level human-object interaction Year: ()
Ref_id:b70 Title: Interactive humanoid: Online full-body motion reaction synthesis with social affordance canonicalization and forecasting Year: (2023)
Ref_id:b71 Title: Leaf: Learning frames for 4d point cloud sequence understanding Year: (2023)
Ref_id:b72 Title: Crossvideo: Self-supervised cross-modal contrastive learning for point cloud video understanding Year: (2024)
Ref_id:b73 Title: Group-free 3d object detection via transformers Year: (2021)
Ref_id:b74 Title: Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors Year: (1936)
Ref_id:b75 Title: Voxel transformer for 3d object detection Year: (2021)
Ref_id:b76 Title: Robomatrix: A skill-centric hierarchical framework for scalable robot task planning and execution in open-world Year: (2024)
Ref_id:b77 Title: Voxnet: A 3d convolutional neural network for real-time object recognition Year: (2015)
Ref_id:b78 Title: Where2act: From pixels to actions for articulated 3d objects Year: (2021)
Ref_id:b79 Title:  Year: (2021)
Ref_id:b80 Title: Gpt-4v(ision) system card Year: (2023)
Ref_id:b81 Title: GPT-4 technical report Year: (2023)
Ref_id:b82 Title: Masked autoencoders for point cloud self-supervised learning Year: ()
Ref_id:b83 Title: Openscene: 3d scene understanding with open vocabularies Year: ()
Ref_id:b84 Title: Dreambench++: A human-aligned benchmark for personalized image generation Year: (2024)
Ref_id:b85 Title: Pointnet: Deep learning on point sets for 3d classification and segmentation Year: (2017)
Ref_id:b86 Title: Pointnet++: Deep hierarchical feature learning on point sets in a metric space Year: (2017)
Ref_id:b87 Title: Learning to move with affordance maps Year: (2020)
Ref_id:b88 Title: Contrast with reconstruct: Contrastive 3d representation learning guided by generative pretraining Year: (2023)
Ref_id:b89 Title: VPP: efficient conditional 3d generation via voxel-point progressive representation Year: (2023)
Ref_id:b90 Title: Shapellm: Universal 3d object understanding for embodied interaction Year: (2024-10-04)
Ref_id:b91 Title: Gpt4point: A unified framework for point-language understanding and generation Year: (2024)
Ref_id:b92 Title: Pointnext: Revisiting pointnet++ with improved training and scaling strategies Year: ()
Ref_id:b93 Title: Exploring spatial representations for visual-languageaction model Year: (2025)
Ref_id:b94 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b95 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b96 Title: Learning transferable visual models from natural language supervision Year: ()
Ref_id:b97 Title: Chatgpt and open-ai models: A preliminary review Year: (2023)
Ref_id:b98 Title: LEAP hand: Low-cost, efficient, and anthropomorphic hand for robot learning Year: (2023)
Ref_id:b99 Title: Perceiver-actor: A multi-task transformer for robotic manipulation Year: (2022-12-18)
Ref_id:b100 Title: Learning social affordance for human-robot interaction Year: (2016)
Ref_id:b101 Title: Multi-view convolutional neural networks for 3d shape recognition Year: (2015)
Ref_id:b102 Title: The open motion planning library Year: (2012)
Ref_id:b103 Title: Onepose: One-shot object pose estimation without CAD models Year: (2022)
Ref_id:b104 Title: Emu: Generative pretraining in multimodality Year: (2024)
Ref_id:b105 Title: Space3dbench: Spatial 3d question answering benchmark Year: ()
Ref_id:b106 Title: Octo: An open-source generalist robot policy Year: (2024)
Ref_id:b107 Title: Understanding natural language commands for robotic navigation and mobile manipulation Year: (2011)
Ref_id:b108 Title: Robots that use language Year: (2020)
Ref_id:b109 Title: Llama: Open and efficient foundation language models Year: (2023)
Ref_id:b110 Title:  Year: ()
Ref_id:b111 Title: Least-squares estimation of transformation parameters between two point patterns Year: (1991)
Ref_id:b112 Title: Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data Year: (2019-11-02)
Ref_id:b113 Title: Attention is all you need Year: (2017)
Ref_id:b114 Title: Bridgedata V2: A dataset for robot learning at scale Year: (2023-11-09)
Ref_id:b115 Title: Graspness discovery in clutters for fast and accurate grasp detection Year: (2021)
Ref_id:b116 Title: Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision Year: ()
Ref_id:b117 Title: Orient anything: Learning robust object orientation estimation from rendering 3d models Year: ()
Ref_id:b118 Title: Chain of thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b119 Title: Foundationpose: Unified 6d pose estimation and tracking of novel objects Year: (2024)
Ref_id:b120 Title: Gpt-4v(ision) is a human-aligned evaluator for text-to-3d generation Year: ()
Ref_id:b121 Title: 3d shapenets: A deep representation for volumetric shapes Year: (2015)
Ref_id:b122 Title: SAPIEN: A simulated part-based interactive environment Year: (2020)
Ref_id:b123 Title: Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes Year: (2018)
Ref_id:b124 Title: Florence-2: Advancing a unified representation for a variety of vision tasks Year: (2024)
Ref_id:b125 Title: Pointcontrast: Unsupervised pre-training for 3d point cloud understanding Year: (2020)
Ref_id:b126 Title: Pointllm: Empowering large language models to understand point clouds Year: (2023)
Ref_id:b127 Title: Universal manipulation policy network for articulated objects Year: (2022)
Ref_id:b128 Title: Set-of-mark prompting unleashes extraordinary visual grounding in GPT-4V Year: (2023)
Ref_id:b129 Title: Thinking in space: How multimodal large language models see, remember, and recall spaces Year: ()
Ref_id:b130 Title: Learning interactive real-world simulators Year: (2024)
Ref_id:b131 Title: Homerobot: Open-vocabulary mobile manipulation Year: (2023-11-09)
Ref_id:b132 Title: A scalable active framework for region annotation in 3d shape collections Year: (2016)
Ref_id:b133 Title: Syncspeccnn: Synchronized spectral CNN for 3d shape segmentation Year: (2017)
Ref_id:b134 Title: Metric3d: Towards zero-shot metric 3d prediction from A single image Year: (2023)
Ref_id:b135 Title: Point-bert: Pre-training 3d point cloud transformers with masked point modeling Year: (2022)
Ref_id:b136 Title: Robopoint: A vision-language model for spatial affordance prediction for robotics Year: (2024)
Ref_id:b137 Title: Learning to manipulate anywhere: A visual generalizable framework for reinforcement learning Year: (2024-11-09)
Ref_id:b138 Title: Uni-navid: A video-based vision-language-action model for unifying embodied navigation tasks Year: (2024)
Ref_id:b139 Title: CLIP-FO3D: learning free open-world 3d scene representations from 2d dense CLIP Year: (2023)
Ref_id:b140 Title: Llama-adapter: Efficient fine-tuning of large language models with zero-initialized attention Year: (2024)
Ref_id:b141 Title: Positional prompt tuning for efficient 3d representation learning Year: (2024)
Ref_id:b142 Title: Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge Year: (2025)
Ref_id:b143 Title: Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning Year: ()
Ref_id:b144 Title: Learning fine-grained bimanual manipulation with low-cost hardware Year: (2023)
Ref_id:b145 Title: 3d-vla: A 3d vision-language-action generative world model Year: (2024)
Ref_id:b146 Title: Judging llm-asa-judge with mt-bench and chatbot arena Year: (2004)
Ref_id:b147 Title: 3d implicit transporter for temporally consistent keypoint discovery Year: (2023)
Ref_id:b148 Title: Code-as-monitor: Constraint-aware visual programming for reactive and proactive robotic failure detection Year: (2025)
Ref_id:b149 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2024)
Ref_id:b150 Title: robosuite: A modular simulation framework and benchmark for robot learning Year: (2020)
Ref_id:b151 Title: RT-2: vision-language-action models transfer web knowledge to robotic control Year: (2023-11-09)
