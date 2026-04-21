Title: Spatial Understanding from Videos: Structured Prompts Meet Simulation Data
Abstract: Visual-spatial understanding, the ability to infer object relationships and layouts from visual input, is fundamental to downstream tasks such as robotic navigation and embodied interaction. However, existing methods face spatial uncertainty and data scarcity, limiting the 3D spatial reasoning capability of pre-trained visionlanguage models (VLMs). To address these challenges, we present a unified framework for enhancing 3D spatial reasoning in pre-trained VLMs without modifying their architecture. This framework combines SpatialMind, a structured prompting strategy that decomposes complex scenes and questions into interpretable reasoning steps, with ScanForgeQA, a scalable question-answering dataset built from diverse 3D simulation scenes through an automated construction process designed for fine-tuning. Extensive experiments across multiple benchmarks demonstrate the individual and combined effectiveness of our prompting and fine-tuning strategies, and yield insights that may inspire future research on visual-spatial understanding.

Section: Introduction
Visual-spatial understanding, the ability to infer spatial relationships and the layout of objects from visual input, is a core component of human perception [1,2,3]. From a single image, human observers can intuitively estimate distances, relative sizes, and even infer occluded structures. As intelligent systems become increasingly embedded in real-world applications such as autonomous driving [4,5,6], robotic navigation [7,8,9], and augmented reality [10,11,12], it becomes crucial to endow models with similar spatial reasoning capabilities for robust perception and interaction.
Unfortunately, a single image is inherently limited in capturing the complexity of real-world 3D scenes, constraining its utility in practical scenarios [13,14,15]. To address this, point clouds have become a mainstream representation for 3D scene understanding due to their ability to encode rich geometric information [16,17]. Yet, generating high-quality point clouds typically requires expensive sensors and incurs significant computational overhead, limiting scalability and accessibility.
These limitations motivate the pursuit of vision-only solutions that operate on scanning videos or multi-view images of scenes. Such approaches offer a more human-like and scalable pathway to spatial understanding [18]. However, performing 3D spatial reasoning from scanning videos presents two significant challenges: (1) Spatial Uncertainty. In the absence of explicit depth information, models must infer 3D structure from inherently limited 2D observations. This process is further complicated by occlusions, perspective distortions, and texture ambiguities, all of which introduce significant spatial uncertainty. Effectively addressing this challenge demands multi-step logical reasoning across frames to reconstruct coherent spatial layouts. (2) Data Scarcity. Existing datasets for this task are limited in both scale and diversity, restricting the ability of vision-language models (VLMs) to acquire robust spatial knowledge and perceptual capabilities. Moreover, these datasets involve scans of real-world scenes, which leads to poor scalability. This highlights the need for scalable and extensible data sources to support effective spatial reasoning in VLMs.
To address these challenges, we propose a dual approach for enhancing 3D spatial reasoning in pre-trained VLMs, without modifying their underlying architecture. First, we introduce SpatialMind, a structured Chain-of-Thought (CoT) prompting strategy that guides VLMs through step-by-step reasoning over spatial relationships. Second, we present ScanForgeQA, a large-scale synthetic question-answering (QA) dataset constructed from diverse 3D simulation scenes using an automated generation pipeline. Fine-tuning VLMs on this dataset equips them with spatial commonsense knowledge, significantly improving their generalization to unseen spatial layouts. We have validated our approach through extensive experiments across multiple benchmarks. Results demonstrate the individual and combined effectiveness of our prompting and fine-tuning strategies, and yield insights that may inspire future research on visual-spatial understanding.
Our contributions are summarized as follows:
• We introduce SpatialMind, a spatial prompting strategy that decomposes spatial reasoning into structured steps, enabling pre-trained VLMs to perform multi-step inference over spatial relationships from visual input alone.
• We develop a scalable dataset generation pipeline to construct ScanForgeQA, a synthetic spatial question-answering dataset that enables VLMs to acquire spatial commonsense through fine-tuning.
• Experimental results validate the effectiveness and generalizability of both SpatialMind and ScanForgeQA, with their combination achieving further gains and providing valuable insights for future research.
2 Related Work 2D Image Spatial Understanding focuses on modeling spatial relationships among objects within the 2D image. Most existing models are trained on 2D images paired with textual descriptions, which offer limited cues about 3D structure. Consequently, their capacity for spatial reasoning remains constrained. To mitigate this, several approaches, such as SpatialVLM [13], SpatialRGPT [14], and SpatialBot [15], have been proposed. These methods enhance the spatial understanding by fine-tuning models on datasets specifically designed for spatially grounded QA tasks. To enable more comprehensive evaluation, recent studies [19,20,21,22,23] have introduced hierarchical benchmarks that assess models across varying levels of spatial reasoning complexity. Parallel efforts have explored more explicit forms of spatial interaction [24,25]. For example, point-based methods [26,27] interpret spatial instructions by predicting specific target points. Building on this trend, SpatialCoT [28] proposes a two-stage strategy that aligns multimodal inputs with spatial coordinates and incorporates CoT reasoning to better address complex embodied tasks. Despite these advancements, model performance often degrades in complex real-world 3D environments, highlighting the limitations of 2D-based approaches in representing complex 3D scenes.
3D Indoor Spatial Understanding focuses on enabling intelligent agents to identify object positions and infer their spatial relationships within enclosed environments, thereby supporting both object manipulation and interactive scene comprehension. Early 3D models are trained on standard indoor datasets [29,30,31,32,33,34,35] using point clouds to facilitate downstream tasks like 3D object detection and instance segmentation [36,37,38,39] and primarily focus on object-level geometry and appearance features [40,41,42,43]. More recent work extends this focus to complex indoor scenes, emphasizing inter-object spatial relationships and holistic scene-level understanding. To address challenges such as geometric complexity and annotation sparsity, many of these models employ cross-modal strategies that combine point cloud data with auxiliary multi-view 2D images [16,17,44]. Inspired by the way humans perceive spatial layouts through vision alone, emerging research [1,45,18] has begun to explore purely vision-based approaches to 3D spatial understanding. These methods rely solely on visual inputs, such as scanning videos, without requiring explicit 3D priors like point clouds. This line of work offers a more practical and scalable alternative for real-world deployment.
In this context, we further investigate whether purely vision-based inputs can provide a more effective solution for indoor scene understanding.
Step1: Identify objects of interest Step2: Estimate position coordinates for these objects Step3: Calculate the distance between the bed and the monitor …… Step1: Identify objects of interest Step2: Estimate position coordinates for these objects Step3: Calculate the distance between the bed and the monitor ……
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b12', 'b13', 'b14', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b15', 'b16', 'b43', 'b0', 'b44', 'b17']

Section: Scanning Video

this section cite: []

Section: Reference Point

this section cite: []

Section: Local Modeling

this section cite: []

Section: Instance Estimation

this section cite: []

Section: Coordinate Mapping
Select an appropriate object as a global reference point, then redefine the positions of other objects relative… 𝑿𝑿 𝒀𝒀 𝒁𝒁 Cognition Generation Monitor: locate one meter to the left of the reference Bed: locate 3 meters in front of the reference point …… VLM Question 2D Grid Description Router Question Decomposition 2 Step1: Identify objects of interest Step2: Estimate position coordinates for these objects Step3: Calculate the distance between the bed and the monitor ……
this section cite: []

Section: 3D Map

this section cite: []

Section: Scene Decomposition 1
Figure 1: Illustration of our SpatailMind prompting strategy.
this section cite: []

Section: SpatialMind Prompting Strategy
As shown in Figure 1, our SpatialMind prompting strategy consists of two main components: 1) Scene Decomposition, where the 3D scene depicted in the video is transformed into multiple different representations; and 2) Question Decomposition, in which the question is broken down into a sequence of fine-grained reasoning steps. Further details can be found in Appendix D.
this section cite: []

Section: Scene Decomposition
The scene decomposition process includes three sequential steps: local modeling, coordinate mapping, and cognition generation.
this section cite: []

Section: Local Modeling.
The first step processes scanning video frames to extract object instances and their relative spatial configurations within localized coordinate systems. To handle scene complexity and reduce the search space, we leverage GPT-4ofoot_0 to identify all objects mentioned across the questions associated with a given scene, using them as candidate targets. For each frame i, we prompt VLMs to detect a subset of objects {c ij } from the candidate targets and estimate their positions p local ij ∈ R 3 . These positions are defined relative to a randomly selected reference object (i.e., origin) within the same frame, forming a local 3D map:
L i = (c ij , p local ij ) | j = 1, . . . , n i ,(1)
where n i denotes the number of objects in frame i. Because each video frame captures only a limited field of view, the same object may appear across multiple frames from different perspectives. Thus, this step focuses on accurate per-frame object detection and spatial localization, laying the foundation for subsequent alignment in a global coordinate system.
this section cite: []

Section: Coordinate Mapping.
To integrate spatial information across video frames, this step transforms all locally detected object positions into a unified global coordinate system. The global origin is defined by selecting the reference object in the first frame. To estimate motion between frames, we prompt the VLM to infer the relative rotation and translation between adjacent frames. These relative transformations are accumulated sequentially to compute each frame's transformation T i with respect to the global coordinate system:
T i = i k=1 R k,k-1 t k,k-1 0 1 ,(2)
where R k,k-1 and t k,k-1 denote the relative rotation and translation from frame k -1 to frame k, respectively. This accumulated approach provides more stable and accurate alignment than directly estimating each frame's absolute pose. Using these transformations, each object's local coordinates are converted into global coordinates via homogeneous transformation:
p global ij 1 = T i • p local ij 1 ,(3)
where p global ij denotes the global coordinates of the object j in the frame i. This step ensures that all detected objects across frames are positioned consistently within the same 3D space. Since objects may appear in multiple frames under different perspectives, we merge duplicate detections based on spatial proximity and semantic consistency via prompting. The result is a global 3D map of the scene:
G = (c k , p global k ) N k=1 ,(4)
where N is the total number of all object instances in the entire scene. This map serves as a unified spatial abstraction that captures the overall layout from egocentric scanning videos.
this section cite: []

Section: Cognition Generation.
Beyond constructing a 3D map, we explore two additional formats for representing scene structure: a 2D spatial grid and natural language descriptions. We define a regular 2D grid over the global scene, typically aligned with the XY -plane. Each grid cell corresponds to a fixed real-world area (e.g., 1 meter per cell, denoted by cell size s ). Each object c k is mapped to a discrete grid location (i k , j k ):
(i k , j k ) = x k s , y k s ,(5)
where (x k , y k ) are the horizontal components of the object's global position p global k . In parallel, we generate natural language descriptions of object locations relative to a designated reference point. Using prompting, the model produces statements such as {"monitor": "locate 1 meter to the left of the reference point"}. These descriptions serve as a human-interpretable form of spatial cognition, bridging visual perception and symbolic reasoning.
this section cite: []

Section: Question Decomposition
Different types of spatial questions require distinct reasoning strategies [46]. To accommodate this diversity, we first categorize questions into several types (e.g., object size, relative distance, and relative direction). For each category, we design a dedicated reasoning procedure using GPT-4o, followed by human verification to ensure correctness and interpretability. For instance, consider a question from the "relative distance" category: Among the refrigerator, window, and microwave, which object is closest to the door? The reasoning process for this type follows four structured steps: 1) Identify all mentioned objects, 2) Estimate the spatial coordinates of all relevant objects, 3) Compute the pairwise distances between the door and each candidate object, and 4) Select the object with the minimum distance as the answer. During inference, the system correspondingly selects the appropriate reasoning procedure based on the identified question type.
To perform 3D spatial reasoning, we feed the VLMs with the input scanning video, one form of scene representation (e.g., 3D map, 2D grid, or textual position descriptions), the question, and the corresponding step-by-step reasoning plan. To assess which scene representation format is most interpretable for VLMs, we have conducted comparative experiments, as shown in Figure 3.
this section cite: ['b45']

Section: ScanForgeQA Dataset Construction
The construction of the ScanForgeQA dataset involves a three-stage pipeline, illustrated in Figure 2. These stages are: 1) Scene Construction, where single-room 3D environments are created; 2) Scan Creation, in which egocentric videos are simulated by scanning through the constructed scenes; and 3) QA Generation, where textual question-answering pairs are automatically generated based on object annotations and the spatial layout of each scene.
this section cite: []

Section: Scene Construction
To ensure diversity and richness in single-room scene collection, we adopt two parallel strategies: Separation. We modify existing scene datasets to leverage available resources effectively. Specifically, we utilize the 3D-FRONT dataset [47], which contains 6,813 multi-room scenes furnished with diverse 3D objects and annotated with detailed layout semantics and high-quality textures. Since our focus is on single-room environments, we disassemble each multi-room scene into individual rooms. For each scene, we isolate and load one room at a time, along with its corresponding ceiling and walls, and save it as an independent instance. This disassembly process yields 44,427 single-room scenes. We further filter out uncommon room types (e.g., garage, auditorium) and those lacking sufficient object content (e.g., aisle, stairwell). The final dataset consists of 34,116 single-room scenes across six common categories: bedroom, kitchen, bathroom, living room, dining room, and storage room.
this section cite: ['b46']

Section: Synthesis.
To introduce additional diversity and originality, we synthesize novel room layouts using a LLM-guided generation approach. Specifically, we adopt HoloDeck [48], a 3D generation framework that leverages LLMs to parse natural language prompts, retrieve matching assets from large-scale 3D object repositories such as Objaverse [49], and optimize their spatial arrangement to form semantically meaningful scenes. To drive the generation process, we first use GPT-4o to create diverse textual descriptions for various room types. For example, a bedroom may be described as: "A bedroom with a bed, window, armchair, and wardrobe". We define eight room categories, including two additional types-office and store-and generate 20 distinct descriptions for each. These prompts are fed into HoloDeck to produce corresponding room layouts, with human verification to ensure spatial plausibility and realism. This synthesis process yields 160 additional single-room scenes.
this section cite: ['b47', 'b48']

Section: Scan Creation
To simulate egocentric scanning videos from the constructed single-room scenes, we implement a scanning procedure using the Unity engine 3 . Each scene is scanned using two complementary strategies designed to emulate natural human visual exploration: Orbit Scan. We define a circular trajectory centered in the room at a height of approximately 1.5 meters, corresponding to typical adult eye level. The circle's diameter is set to two-thirds of the shorter side of the room. The camera is randomly initialized at a point on this path and moves along the circle either clockwise or counterclockwise. An image is captured every 5 degrees of rotation, resulting in 72 frames per orbit scan. This strategy provides a comprehensive 360-degree panoramic view of the scene.
this section cite: []

Section: Navigation Scan.
To simulate movement through the environment, we label navigable ground regions based on object categories and generate a navigation mesh using the NavMesh Baking API. We randomly select two objects as the navigation start and end points and compute the shortest path between them on the mesh. Among the candidate paths, the two longest are chosen for scanning to achieve a more complete coverage of the scene. For each path, the camera first performs a 360-degree rotation at the starting point, capturing an image every 12 degrees (30 images total). It then traverses the path toward the destination, during which 12 frames are uniformly sampled. Upon arrival, another 360-degree rotation is performed, again capturing 30 images. In total, 72 frames are recorded per path. Due to the limited size of indoor environments, rotational movement yields more visual variation than translation; hence, fewer frames are captured during motion.
this section cite: []

Section: QA Generation
To generate diverse supervised fine-tuning (SFT) data and enhance the 3D spatial reasoning capabilities of existing VLMs, we define three categories of question types: attribute estimation, spatial reasoning, and hypothesis analysis. These categories encompass both quantitative and qualitative dimensions, and cover both open-set and closed-set scenarios. Below, we describe each category in detail, along with the methodology for deriving corresponding ground-truth answers.
Attribute Estimation. This type focuses on static properties of objects and scenes, such as object count ("How many chairs are in the room?"), object size ("What is the length of the longest side of the refrigerator in meters?"), room size ("What is the size of this room in square meters?"), and room type ("Based on the object layout, what is the most likely type of room (e.g., kitchen)?"). Ground-truth answers for these questions are directly derived from 3D scene annotations and object metadata provided in the dataset. Spatial Reasoning. This category targets interobject spatial relationships, requiring models to infer positional and geometric properties such as distance, orientation, and contact. Representative question types include: relative distance ("Which of these objects (refrigerator, couch, ceiling light) is closest to the TV?"), absolute distance ("What is the distance between the couch and the table in meters?"), relative direction ("If I am standing by sofa and facing the table, which side is the trash can on?"), and contact relationship ("Is there a gap between the bed and the headboard?"). For distancerelated questions, we compute Euclidean distances between object centroids in the global 3D coordinate space. For contact relationships, object dimensions are also considered to determine physical adjacency. To resolve relative direction, we define an object's front as the side oriented toward the room center. Angular sectors are divided clockwise into four directional categories: right (45°-135°), back (135°-225°), left (225°-315°), and front (315°-45°). For example, an object located at 80°relative to the reference point is classified as being on the right.
this section cite: []

Section: Hypothesis Analysis.
This category introduces conditional reasoning under hypothetical scenarios, often requiring geometric and commonsense inference. A typical example is operation feasibility ("Considering only object dimensions, is it feasible to place the television on the table?"). Feasibility is determined by comparing object dimensions. For stacking, the movable object's length and width must be smaller than those of the supporting surface. For embedding (e.g., fitting an item into a drawer), the object's height must also fall within the bounds of the specific container's volume.
A comparison with existing 3D QA datasets (e.g., SPARTUN3D [50], MSQA [51], and 3D-LLM [17]) is presented in Table 1. The full ScanForgeQA dataset includes 34,276 single-room scenes, 103K simulated video scans, and 925K question-answering pairs for training. Leveraging synthetic environments allows scalable and controlled data generation across diverse spatial scenarios. Additional implementation details are provided in the Appendix C. 5 Experiments
The experimental settings (including benchmarks, baselines, etc.) and more experimental results can be found in the Appendix A and B.
this section cite: ['b49', 'b50', 'b16']

Section: Performace Comparison
We investigated the following five key questions to assess our approach:
Q1: Which scene representation format is most interpretable by VLMs? Figure 3 presents a performance comparison across different representation formats: no additional spatial context (Base), inclusion of a 3D map (+Map), a 2D grid (+Grid), and object-centric textual descriptions (+Des). Across all models, a consistent trend emerges: the +Des variant outperforms others, followed by 8 16 24 32 Fr a m e 128 256 384 512 R e s o l u t i o n 4 5 6 7 Gain (+Both -Base) 4.5 5.0 5.5 6.0 6.5 +Grid, while +Map yields the least improvement. These results suggest that current VLMs are more adept at interpreting one-dimensional textual descriptions than high-dimensional structured spatial formats. Consequently, we adopted the textual description format in subsequent experiments as the default scene representation.
How do SpatialMind and ScanForgeQA impact VLM performance? As shown in Table 2, we progressively applied the SpatialMind prompting strategy and the ScanForgeQA fine-tuning data across a range of VLMs, varying in architectures, parameter size, and openness (including both openand closed-source models). The results reveal three key findings: 1) Both SpatialMind prompting and ScanForgeQA fine-tuning consistently improve visual-spatial understanding across models. This includes large-scale proprietary models such as Gemini-1.5 Profoot_2 and GPT-4o, demonstrating the effectiveness and generalizability of our approaches. 2) Model size affects the relative benefit of prompting versus fine-tuning. Larger models (e.g., 72B) benefit more from prompting, which enhances their reasoning capabilities, while smaller models (e.g., 7B) show greater improvements through fine-tuning. For instance, Qwen2.5-VL-7B gains 6.1% from fine-tuning, compared to only 2.0% from prompting. 3) Humans and VLMs exhibit complementary strengths. Human participants excel in qualitative tasks (e.g., achieving 100% accuracy on the Appearance Order task) but perform poorly on precise quantitative estimations (e.g., Object Size). In contrast, VLMs show strong quantitative reasoning ability and, in some cases, even surpass human-level performance. This contrast underscores the potential of VLMs to complement human perception in spatial tasks.
Can combining prompting and fine-tuning yield further gains? To assess whether SpatialMind and ScanForgeQA provide complementary benefits, we applied the SpatialMind prompting strategy to models that have already been fine-tuned on the ScanForgeQA dataset. The results, reported in the "+Both" row of Table 2, show consistent performance improvements across all evaluated models. These findings confirm that the two approaches are complementary.
this section cite: []

Section: Does the improvement generalize to other spatial benchmarks?
To assess the generalizability of our framework, we conducted evaluations on multiple benchmarks, including OpenEQA [52], ScanQA [53], and SQA3D [54]. As shown in Table 3, both SpatialMind prompting and ScanForgeQA fine-tuning lead to consistent performance gains across all benchmarks. These results validate the robustness of our approach and confirm its applicability across diverse spatial tasks and datasets.
this section cite: ['b51', 'b52', 'b53']

Section: Does fine-tuning affect performance on other tasks?
To investigate whether enhancing visualspatial capabilities via fine-tuning adversely impacts a model's general performance, we conducted evaluations on MVBench [55] and Video-MME [56], two broad multi-task video benchmarks. As shown in Figure 4, fine-tuning with ScanForgeQA slightly improves performance on MVBench but leads to a marginal drop on Video-MME. This difference likely stems from MVBench containing spatial reasoning tasks, while Video-MME focuses more on event comprehension. To mitigate this trade-off, we further experimented with mixed fine-tuning, combining a small proportion (5% and 10%) of traditional data from ShareGPT4Video [57] with ScanForgeQA. Results show that What will be the first-time appearance order of the following categories in the video: door, towel, refrigerator, microwave? this strategy achieves improved performance, surpassing the original Qwen2.5-VL-7B baseline, suggesting that spatial fine-tuning can be harmonized with broader capabilities through data balancing.
this section cite: ['b54', 'b55', 'b56']

Section: Ablation Study
In this section, we explored the impact of various design choices, including prompting strategies, fine-tuning datasets, frame sampling strategies, and input resolution, on the performance of VLMs.
this section cite: []

Section: On prompting strategy.
To isolate the contributions of each component in the SpatialMind prompting strategy, we evaluated two variants: one containing only the question component (CoT-Question) and another containing only the scene description (CoT-Scene). As shown in Table 4, both variants independently improve spatial reasoning performance, but are less effective than the full combined prompt. Notably, the scene description contributes more significantly to model performance than the reasoning steps, suggesting its central role in facilitating spatial understanding.
this section cite: []

Section: On fine-tuning data.
To investigate the effectiveness of our proposed ScanForgeQA against existing spatial datasets, we fine-tuned Qwen2.5-VL-7B on SQA3D [54] and ScanQA [53]. As shown in Table 4, fine-tuning on either of these datasets results in lower performance compared to ScanForgeQA, and even reduces accuracy on tasks involving precise spatial estimation (e.g., Room Size). This is primarily due to the limited presence of fine-grained spatial estimation samples in the existing datasets. Importantly, both datasets and the VSI-Bench benchmark originate from the same source (i.e., ScanNet [31]), resulting in minimal data discrepancy. This contrast emphasizes the advantage of our simulated data generation pipeline.
On frames and resolution. To evaluate the robustness of our approach, we analyzed performance sensitivity to the number of input frames and image resolution. Figure 5 visualizes the performance of the +Both variant and the baseline Qwen2.5-VL-7B under various configurations. Our method consistently outperforms the baseline across all settings, with performance further improving as the number of frames and resolution increase. This indicates that our approach remains stable and effective under varying visual input conditions.
this section cite: ['b53', 'b52', 'b30']

Section: Qualitative Analysis
In Figure 6, we presented two illustrative examples from VSI-Bench, comparing predictions from the baseline Qwen2.5-VL-7B and our enhanced variant (+Both). In Case (a), Qwen2.5-VL-7B fails to produce the correct directional prediction, likely due to its limited capacity for 3D spatial reasoning. In contrast, our method successfully identifies the correct answer. Case (b) involves a simpler spatial reasoning task, however, Qwen2.5-VL-7B still fails, potentially due to insufficient object localization.
Our enhanced variant, benefiting from both structured prompting and spatially grounded fine-tuning, demonstrates notable improvements in accuracy and reasoning robustness.
this section cite: []

Section: Conclusion
In this work, we present an effective framework for enhancing visual-spatial reasoning in VLMs without modifying their underlying architecture. This makes our approach readily adaptable across models of varying scales and types. By integrating the structured prompting strategy (SpatialMind) with an automatically constructed dataset (ScanForgeQA), we enable VLMs to more effectively interpret and reason about 3D spatial relationships in complex visual scenes. Extensive evaluations across multiple spatial reasoning benchmarks demonstrate that our framework consistently improves accuracy, robustness, and generalization. Furthermore, our analysis reveals that prompting and fine-tuning play complementary roles in advancing visual-spatial understanding.
this section cite: []

Section: References
Ref_id:b0 Title: Thinking in space: How multimodal large language models see, remember, and recall spaces Year: (2024)
Ref_id:b1 Title: Llava-vsd: Large language-and-vision assistant for visual spatial description Year: (2024)
Ref_id:b2 Title: Optimus-2: Multimodal minecraft agent with goal-observation-action conditioned policy Year: (2025)
Ref_id:b3 Title: Drivevlm: The convergence of autonomous driving and large vision-language models Year: ()
Ref_id:b4 Title: Explicit granularity and implicit scale correspondence learning for point-supervised video moment localization Year: (2024)
Ref_id:b5 Title: Attribute-guided collaborative learning for partial person re-identification Year: (2023)
Ref_id:b6 Title: Palm-e: an embodied multimodal language model Year: (2023)
Ref_id:b7 Title: Multifactor adaptive vision selection for egocentric video question answering Year: (2024)
Ref_id:b8 Title: Optimus-1: Hybrid multimodal memory empowered agents excel in long-horizon tasks Year: (2024)
Ref_id:b9 Title: Hourvideo: 1-hour videolanguage understanding Year: (2024)
Ref_id:b10 Title: Exo2ego: Exocentric knowledge guided mllm for egocentric video understanding Year: (2025)
Ref_id:b11 Title: Object-shot enhanced grounding network for egocentric video Year: (2025)
Ref_id:b12 Title: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities Year: (2024)
Ref_id:b13 Title: Spatialrgpt: Grounded spatial reasoning in vision-language models Year: (2025)
Ref_id:b14 Title: Spatialbot: Precise spatial understanding with vision language models Year: (2024)
Ref_id:b15 Title: Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning Year: (2024)
Ref_id:b16 Title: 3d-llm: Injecting the 3d world into large language models Year: (2023)
Ref_id:b17 Title: Gpt4scene: Understand 3d scenes from videos with vision-language models Year: (2025)
Ref_id:b18 Title: Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models Year: (2024)
Ref_id:b19 Title: Sphere: Unveiling spatial blind spots in vision-language models through hierarchical evaluation Year: (2024)
Ref_id:b20 Title: A survey on video temporal grounding with multimodal large language model Year: (2025)
Ref_id:b21 Title: Attentive moment retrieval in videos Year: (2018)
Ref_id:b22 Title: Optimus-3: Towards generalist multimodal minecraft agents with scalable task experts Year: (2025)
Ref_id:b23 Title: Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors Year: ()
Ref_id:b24 Title: Rag-guided large language models for visual spatial description with adaptive hallucination corrector Year: (2024)
Ref_id:b25 Title: Robopoint: A vision-language model for spatial affordance prediction in robotics Year: ()
Ref_id:b26 Title: Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics Year: (2024)
Ref_id:b27 Title: Advancing spatial reasoning through coordinate alignment and chain-of-thought for embodied task planning Year: (2025)
Ref_id:b28 Title: Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data Year: ()
Ref_id:b29 Title: Matterport3d: Learning from rgb-d data in indoor environments Year: (2017)
Ref_id:b30 Title: Scannet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017)
Ref_id:b31 Title: Procthor: Large-scale embodied ai using procedural generation Year: (2022)
Ref_id:b32 Title: Multiscan: Scalable rgbd scanning for 3d environments with articulated objects Year: (2022)
Ref_id:b33 Title: Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai Year: ()
Ref_id:b34 Title: Scannet++: A highfidelity dataset of 3d indoor scenes Year: (2023)
Ref_id:b35 Title: Softgroup for 3d instance segmentation on point clouds Year: (2022)
Ref_id:b36 Title: Point transformer v3: Simpler faster stronger Year: (2024)
Ref_id:b37 Title: Open3dis: Open-vocabulary 3d instance segmentation with 2d mask guidance Year: (2024)
Ref_id:b38 Title: Unscene3d: Unsupervised 3d instance segmentation for indoor scenes Year: (2024)
Ref_id:b39 Title: Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following Year: (2023)
Ref_id:b40 Title: Shapellm: Universal 3d object understanding for embodied interaction Year: (2024)
Ref_id:b41 Title: Lion: Linear group rnn for 3d object detection in point clouds Year: (2024)
Ref_id:b42 Title: Pointllm: Empowering large language models to understand point clouds Year: (2024)
Ref_id:b43 Title: Lexicon3d: Probing visual foundation models for complex 3d scene understanding Year: (2024)
Ref_id:b44 Title: Improved visual-spatial reasoning via r1-zero-like training Year: (2025)
Ref_id:b45 Title: Multimodal dialog system: Relational graph-based context-aware question understanding Year: (2021)
Ref_id:b46 Title: 3d-front: 3d furnished rooms with layouts and semantics Year: (2021)
Ref_id:b47 Title: Language guided generation of 3d embodied ai environments Year: (2024)
Ref_id:b48 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b49 Title: Spartun3d: Situated spatial understanding of 3d world in large language models Year: (2024)
Ref_id:b50 Title: Multi-modal situated reasoning in 3d scenes Year: (2024)
Ref_id:b51 Title: Embodied question answering in the era of foundation models Year: (2024)
Ref_id:b52 Title: Scanqa: 3d question answering for spatial scene understanding Year: (2022)
Ref_id:b53 Title: The Eleventh International Conference on Learning Representations Year: ()
Ref_id:b54 Title: Mvbench: A comprehensive multi-modal video understanding benchmark Year: (2024)
Ref_id:b55 Title: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2024)
Ref_id:b56 Title: Sharegpt4video: Improving video understanding and generation with better captions Year: (2024)
Ref_id:b57 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2024)
Ref_id:b58 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b59 Title: Videollama 3: Frontier multimodal foundation models for image Year: (2025)
Ref_id:b60 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b61 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b62 Title: Gpt-4o system card Year: (2024)
