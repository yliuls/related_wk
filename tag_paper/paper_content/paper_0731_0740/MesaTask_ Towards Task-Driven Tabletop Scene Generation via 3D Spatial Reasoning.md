Title: MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning
Abstract: The ability of robots to interpret human instructions and execute manipulation tasks necessitates the availability of task-relevant tabletop scenes for training. However, traditional methods for creating these scenes rely on time-consuming manual layout design or purely randomized layouts, which are limited in terms of plausibility or alignment with the tasks. In this paper, we formulate a novel task, namely task-oriented tabletop scene generation, which poses significant challenges due to the substantial gap between high-level task instructions and the tabletop scenes. To support research on such a challenging task, we introduce MesaTask-10K, a large-scale dataset comprising approximately 10,700 synthetic tabletop scenes with manually crafted layouts that ensure realistic layouts and intricate inter-object relations. To bridge the gap between tasks and scenes, we propose a Spatial Reasoning Chain that decomposes the generation process into object inference, spatial interrelation reasoning, and scene graph construction for the final 3D layout. We present MesaTask, an LLM-based framework that utilizes this reasoning chain and is further enhanced with DPO algorithms to generate physically plausible tabletop scenes that align well with given task descriptions. Exhaustive experiments demonstrate the superior performance of MesaTask compared to baselines in generating task-conforming tabletop scenes with realistic layouts.

Section: Introduction
A fundamental challenge in robotic manipulation is enabling robots to accurately interpret human instructions and successfully execute complex tasks accordingly. The conventional pipeline for achieving this involves task definition, simulatable tabletop scene construction, and policy training. However, traditional scene construction methods, which rely on manual design or purely randomized layouts, are limited by their labor-intensive nature and the resulting constraints on diversity and plausibility, ultimately hindering the generalization of learned policies. Therefore, automatic taskoriented tabletop scene generation emerges as a promising approach for effectively bridging the gap between task descriptions and scenes. Crucially, tabletop scene generation must satisfy three key requirements: covering task variables, enabling scene interactivity, and ensuring realistic layouts, thereby facilitating the learning of robust policies.
Existing scene generation methods [27,6,11] often start from a single scene image and attempt to recover the corresponding tabletop scene through object retrieval and layout optimization. Unfortunately, their ability to understand under-specified task instructions still requires empirical corroboration. Other approaches [37,35,2] leverage powerful language models (LLMs) to interpret
this section cite: ['b25', 'b5', 'b10', 'b35', 'b33', 'b1']

Section: Mesa Task

this section cite: []

Section: MesaTask-10K
Kitchen Counter Kitchen Counter
this section cite: []

Section: Bathroom Vanity

this section cite: []

Section: Dinning Table
Figure 1: We present MesaTask, a novel LLM-based framework for generating task-oriented 3D tabletop scenes directly from high-level human instructions, featuring realistic layouts, articulated objects, and complex inter-object relations like stacking and containment. To support this task, we introduce a large-scale dataset of tabletop scenes, MesaTask-10K, comprising over 12, 000 3D assets, 11, 708 tabletop scenes with manually crafted layouts covering 6 common indoor table types. task prompts and then synthesize tabletop scenes in a zero-shot manner. Nevertheless, these methods are hindered by inherent limitations, no matter the inevitable occlusion in scene images or the lack of fine-tuning on a scene dataset, which significantly impede the modeling of realistic table layouts and complex inter-object relations, such as stacking and containment, within the scene. As a result, task-oriented tabletop scene generation remains a challenging problem due to the scarcity of datasets and the substantial gap between task instructions and scene layouts.
To tackle these challenges, we collect a first-of-its-kind dataset of synthetic tabletop scenes with manually crafted layouts, dubbed MesaTask-10K. As shown in Figure 1, our dataset comprises approximately 10, 700 diverse tabletop scenes, spanning six common indoor table categories, including office tables, dining tables, kitchen counters, and more. The 3D objects in MesaTask-10K originate from a large asset library containing over 12, 000 rigid and articulated 3D assets, each with detailed semantic information, such as object category, description, and materials, and featuring a comprehensive taxonomy of over 200 object classes on the tables. As claimed in [27], pretrained 2D image generative models better capture scene and object configurations both at the scene level and in fine-grained inter-object relations. Inspired by this, our dataset is built upon diverse tabletop scene images with diversity and realistic layouts, generated by a large text-to-image model [12] pretrained on massive internet data. To obtain a coarse layout from the scene image, we estimate the depth [33] of each image, extract the instance point cloud, and acquire the 3D bounding box of objects. We then leverage the object descriptions labeled by VLM [1] to retrieve suitable 3D assets from the library and construct an initial replica of the tabletop scenes. Subsequently, human annotators meticulously refine these 3D layouts, adjusting the object size and positions as per the image prompt, addressing inaccuracies from occlusion and ensuring complex inter-object relations. Ultimately, all the scenes are put into a physical simulator, IsaacSim [19], to prevent object collisions.
Confronted with the significant gap between tasks and scenes, we propose a novel paradigm referred to as Spatial Reasoning Chain, decomposing task-to-tabletop scene generation into a structured chain of thought (CoT). Given a high-level task description, this chain of thought begins with the inference of requisite objects, accompanied by their semantic attributes and spatial interrelations, based on which a complete scene graph is formed, and finally leads to a concrete 3D layout of objects on the table. To establish trainable spatial reasoning chains with our dataset, we design a set of delicate rules to extract the object attributes and inter-object relations, thus forming a scene graph for each tabletop scene. Subsequently, we leverage a multimodal large language model, taking scene graphs and rendered scene images as input, to generate corresponding task information and detailed spatial reasoning descriptions for training.
Thanks to our structured reasoning chains, it's convenient to empower LLM with 3D spatial reasoning and scene generation capability. In this paper, we propose MesaTask, a novel LLM-based framework for task-oriented tabletop scene generation. Specifically, we initially employ the supervised finetuning (SFT) strategy on our constructed reasoning data to inject the LLM with 3D spatial reasoning capabilities. However, MesaTask occasionally generates unsatisfactory tabletop scenes with minor object collisions and misalignment with the given task. To circumvent this hurdle, we devise paired training data and leverage a conventional RL algorithm, namely Direct Preference Optimization (DPO), to boost our MesaTask model, thereby ensuring that the generated scenes are devoid of object collisions and exhibit improved conformity with the provided task descriptions.
For a more comprehensive performance assessment, we leverage powerful VLMs to evaluate the rendered scene images from multiple perspectives, including task-scene alignment, physical viability, scene layout plausibility, etc. Through extensive experiments, our MesaTask framework is capable of generating physically plausible tabletop scenes with realistic layouts, outperforming baseline methods in terms of FID, VLM-based metrics, and the user study. In particular, our generated tabletop scenes strictly conform to given task instructions and exhibit rich inter-object relations, such as stacking or containing. In summary, our contributions are threefold:
• We pioneer the formulation of Task-to-Scene generation task, which aims to generate physically plausible tabletop scenes directly from high-level task descriptions.
• We introduce MesaTask-10k, a large-scale tabletop scene dataset with human-crafted realistic layouts, characterized by rich inter-object relations and a tremendous amount of synthetic 3D object assets.
• Along with the delicate design of spatial reasoning chains, we propose MesaTask, an LLM-based framework endowed with the capability of 3D spatial reasoning and tabletop scene generation, achieving superior performance across various evaluation criteria.
this section cite: ['b25', 'b31', 'b0', 'b17']

Section: Related Work
Tabletop Scenes Dataset. Recent works have explored various approaches to constructing tabletop scene datasets. LVDiffusor [39] uses large VLMs to generate semantically plausible tabletop scene images, which are constrained in the 2D domain and lack 3D spatial information. StructFormer [17] and StructDiffusion [16] collect 3D object rearrangement data under language guidance using abstract geometric relations, which allows testing structural reasoning but lacks semantic richness and realism for real-world deployment. SetItUp [30] presents manually designed functional scenes reflecting realworld usage like dining or working, but the object sets are fixed and lack diversity. TO-Scene [29] provides a large-scale and richly annotated 3D tabletop dataset built by professional designers. However, its top-down, click-to-place annotation paradigm restricts the inclusion of intricate spatial relationships such as nesting and stacking. Despite these efforts, existing datasets frequently exhibit limitations in terms of data scale, layout, or realism. Accordingly, we introduce a large-scale tabletop scene dataset with diverse real-world 3D objects, realistic layouts, and rich 3D spatial relationships.
this section cite: ['b37', 'b15', 'b14', 'b28', 'b27']

Section: Scene Reconstruction from A Single Image.
It's a long-standing problem to reconstruct 3D scenes from a single image. A line of previous methods [3,40,15,18] attempt to reconstruct the scenes by compressing the input images with an encoder and mapping the image features back to the 3D space via a decoder. Based on advancements in 3D object generation [41,13], MIDI [11] is capable of generating scenes with diverse 3D objects but struggles to generate complex inter-object relationships. Some other methods [38,10,4,14,6,27] typically entail a multi-stage process, comprising object segmentation, occlusion completion, image-to-3D generation, and layout optimization. This protracted workflow inevitably gives rise to error accumulation, particularly in regions with severe occlusions. Moreover, these methods fall short of generating scenes from underspecified task descriptions. In contrast, our LLM-based framework is inherently designed to fit the task-oriented tabletop scene generation.
this section cite: ['b2', 'b38', 'b13', 'b16', 'b39', 'b11', 'b10', 'b36', 'b9', 'b3', 'b12', 'b5', 'b25']

Section: LLM-Based Scene Generation.
Inspired by the prosperity of Large Language Models (LLMs), many researchers have exploited the capabilities of powerful LLMs to perform 3D scene generation. For instance, LayoutGPT [8] explores direct 3D layout generation through in-context learning. Furthermore, some methods [9,2,37] are built upon commercial LLMs and use multi-stage prompting to achieve open-vocabulary and dataset-free generation in a zero-shot manner. However, these methods encounter substantial challenges in modeling complex inter-object relations. LLPlace [35] 3D Asset Library "a white bowl" "a half lemon"
this section cite: ['b7', 'b8', 'b1', 'b35', 'b33']

Section: Reference Image

this section cite: []

Section: Wrong Size Collision

this section cite: []

Section: Redundant Objects

this section cite: []

Section: Point Cloud

this section cite: []

Section: …

this section cite: []

Section: Isometric Placement

this section cite: []

Section: Manual Craft
Physical Simulation
this section cite: []

Section: 2000+ Hours
Object retrieval
1. Infer common object 2. Imagine spatial relation Infer Object List： Lemon Knife Bowl Spoon Cooking pot Cutting board Generate Object relation: Cutting boards, knife, and spoon are on the left. Nearby is a white bowl and a glass jar. In the middle, there's a wooden plate with chopsticks. On the right, there are a white pot, a bowl of yellow lemons.
this section cite: []

Section: Text-to-Image Model

this section cite: []

Section: Generated Images
"Generate some diverse {Kitchen Counter} descriptions.
this section cite: []

Section: Description = "{Infer Object List} , {Object relation} "

this section cite: []

Section: Coarse Tabletop Scene Construction

this section cite: []

Section: Tabletop Scene Image Generation

this section cite: []

Section: Human-Assisted Layout Refinement Dataset Statistics

this section cite: []

Section: Final Scene

this section cite: []

Section: Depth Anything V2

this section cite: []

Section: Grounded SAM

this section cite: []

Section: Vision Language Model
Figure 2: The dataset construction pipeline. First, an LLM is used to generate diverse tabletop scene descriptions, including relevant object lists and spatial relations. Conditioned on the scene description, a text-to-image model synthesizes reference images, from which coarse 3D layouts are built using depth estimation, object detection, and 3D asset retrieval. These layouts are refined through human annotations and physical simulation to ensure spatial plausibility, yielding high-quality 3D layouts.
attempts to fine-tune the LLM via supervised fine-tuning (SFT) on meticulously crafted 3D scene datasets, albeit without a specific focus on tabletop scenes. In contrast, we also leverage an LLMbased framework for tabletop scene generation, but our approach involves training on a large-scale scene dataset with manually crafted layouts, thereby empowering our model with superior capabilities for generating realistic layouts and intricate inter-object relationships.
this section cite: []

Section: MesaTask-10K Dataset
Inspired by ARCHITECT [27], we build the MesaTask-10K dataset upon diverse tabletop scene images generated by a pretrained text-to-image model [12], ensuring realistic scene layouts and complex inter-object relationships.
Tabletop Scene Image Generation. To better facilitate manipulation tasks, we intend to synthesize diverse tabletop scene images of six common indoor table types in our daily life: office table, dining table, kitchen counter, coffee table, bathroom vanity, and dressing table. As illustrated in Figure 2, the pretrained LLM [1] is guided to output an object list on the table and their spatial relations, respectively, which are subsequently combined to form final scene descriptions. Conditioned on these scene descriptions, FLUX [12], a cutting-edge text-to-image model, is utilized to produce a diverse range of reference scene images w.r.t. six distinct table categories.
Coarse Tabletop Scene Construction. To create 3D replicas of scene images, we first collect a high-quality 3D asset library through meticulous asset curation from two datasets, namely Objaverse [7], and PartNet-Mobility [28]. It's noteworthy that our library consists of over 12, 000 rigid and interactive objects along with rich object semantic information, including object category, text descriptions, and materials. As shown in Figure 2, Grounded-SAM [1] is employed to identify all the object instances in the scene image, and a multimodal LLM like GPT-4o is responsible for providing the corresponding semantic information for subsequent 3D object retrieval. During the object retrieval process, we specifically rely on textual descriptions of objects rather than their visual appearance, considering severe occlusions in the tabletop scene images. Furthermore, we utilize Depth Anything v2 [32] to construct the point cloud of tabletop scenes, and leverage instance masks to obtain 3D bounding boxes for each object within the scene, thereby yielding a coarse 3D layout of the tabletop scene.
this section cite: ['b25', 'b0', 'b6', 'b26', 'b0', 'b30']

Section: Human-Assisted Layout Refinement.
Owing to the intricate inter-object relationships and severe occlusions in the reference images, the obtained coarse scene layouts inevitably contain various flaws, including inaccuracies in object scale, redundant object instances, and object collisions or floating, as shown in Figure 2. To the best of our knowledge, these awkward issues can only be effectively addressed with human assistance. Therefore, 20 expertize annotators undertake a manual layout refinement in Blender, wherein they adjust the object size and positions, as well as eliminate redundant instances, following the reference images.
During annotation, annotators are provided with the coarse 3D scene in GLB format, which includes a Unitree H1 robot model with an absolute height of 1.7m to facilitate the construction of metric-scale 3D scenes, along with reference images and all object snapshots from the images. They will adjust each object's relative size and position with reference to the given tabletop scene images, calibrate the overall scene scale using the H1 model, and rotate objects to match their orientations in the images. On average, annotators spend 10 to 20 minutes on each tabletop scene. Subsequently, we put all tabletop scenes into the physical simulator to prevent object collisions, manually exclude unsatisfactory scenes, and finally create our dataset.
this section cite: []

Section: Dataset Statistics.
MesaTask-10K is a large-scale dataset with approximately 10, 700 diverse tabletop scenes spanning six common indoor table categories. Meanwhile, our curated 3D asset library contains a vast collection of over 12, 000 diverse objects, covering a broad spectrum of more than 200 object classes that are typically encountered on tables. In particular, the 100 object categories that occur the most frequently within this library are visually represented in Figure 2. Moreover, there are roughly 15 objects per tabletop scene on average, and the distribution of the object number is also visualized in Figure 2. We believe that our MesaTask-10K dataset possesses substantial potential to drive research advancements in the realm of task-oriented tabletop scene generation.
this section cite: []

Section: Method
In this section, we present our novel LLM-based framework MesaTask for generating realistic 3D tabletop scenes from manipulation task descriptions. The crux of our approach lies in endowing an LLM with the capability of 3D spatial reasoning, enabling it to infer the complex spatial arrangements necessary to fulfill the requirements of a given task. We formalize the problem in Section 4.1 and describe the system outlined in Figure 3. To empower the LLM with 3D spatial reasoning, we propose a novel spatial reasoning chain in Section 4.2 and introduce our model's training via SFT and DPO algorithms in Section 4.3.
this section cite: []

Section: Problem Formulation
Task-oriented tabletop scene generation aims at generating suitable tabletop scenes S from highlevel manipulation task instructions T. Following prior works [34,25,35], a tabletop scene S is a composition of N 3D objects arranged in a specific 3D layout L = {l 1 , l 2 , ..., l N }. The layout of each 3D object l i = [p i , s i , θ i , t i ] is defined by its location p ∈ R 3 , axis-aligned 3D bounding box size s ∈ R 3 , rotation angle around the vertical axis θ ∈ R, and its textual description t detailing the category, shape, and appearance. Given a task instruction T, we will leverage a pretrained LLM model to generate more detailed task information, including the table environment description E, a sequence of decomposed goals G for this given task, and a set of task-relevant objects O. Based on them, the scene generation model M is responsible for generating a corresponding 3D scene layout L:
L = M (E, G, O) , [E, G, O] = LLM(T).(1)
Based on the object descriptions in the layout L, appropriate 3D assets will be retrieved from the 3D asset database to form a complete tabletop scene S. It's noteworthy that the generated tabletop scenes will contain all the 3D objects recommended in O and typically include many other objects to ensure realistic layouts.
this section cite: ['b32', 'b23', 'b33']

Section: Spatial Reasoning Chain
While task instructions are typically conveyed through natural language expressions, 3D scene layouts are inherently represented with structured spatial configurations. Considering the large gap between tasks and tabletop scenes, we propose the spatial reasoning chain to decompose the challenging
Task info: Environment: A tabletop with a desk lamp, two kettle, a tray holding a potted plant and a bowl of fruit, and an empty bowl. Sub-Goal: "Pick all fruit out of the bowl on the tray", "Place all fruit into the empty bowl on the right", "Pick up the tray with plant and empty bowl", "Place the tray near the desk lamp on the left side" Task related objects: ["Fruit", "Bowl", "Tray", "Lamp"] in left of right of above left of left of left of right of DPO Data Construction right of above right of left of Final Result Layout Perturbation Task Object Removal Edge Damage MesaTask-10K User Task instructions: "Organize fruit from bowl into empty bowl and place tray near lamp." Object Completion 2 kettles, 1 desk lamp, 1 tray, 1 bowl with fruit, 1 empty bowl, 1 potted plant 5 fruits Interrelation inference Desk lamp … left side … empty bowl … right side… open space … middle … Scene Graph (tray_0, is at, left center) (fruit_0, in, bowl_1) (tray_0, face to, front_right) … Spatial Reasoning Chain Object retrieval [ {object: tray_0 size: [10.4, 14.6, 3.2] position: [36.8, 13.0, 1.5] rotation: 30.0 description: "wooden tray…"}, Scene Layout Rendered Image Scene Description Scene Graph Task Instruction Task Info Complete Object Lists Complete Interrelations Given a task instruction, we extract detailed task information including environment, sub-goals, and task-relevant objects. A structured spatial reasoning chain performs object list completion, interrelation inference, and scene graph construction, which guides the generation of 3D layouts. Final scenes are obtained via 3D asset retrieval. 2) Reasoning Data Construction (bottom). Based on scene graphs and descriptions of our MesaTask-10K dataset, A multimodal LLM is leveraged to produce task instructions, detailed task information, and complete object lists and interrelations. 3) DPO Data Construction (upper right). To enable DPO training, we generate negative examples by randomly perturbing object positions or relations and removing key objects from normal layouts.
task-to-scene generation problem into a structured chain of thought (CoT), significantly easing the training and inference of LLM-based models like ours.
Task-to-Scene Generation via Spatial Reasoning Chain. The spatial reasoning chain encompasses three pivotal steps, namely object list completion, interrelationship inference, and scene graph construction, which function as an effective bridge between input tasks and desirable 3D layouts. In the object list completion stage, the generation model M is motivated to infer a complete list of 3D objects V given the aforementioned task-relevant objects O. Then, the model M will generate inter-object relations E expressed with text descriptions, conditioned on the given task instructions and typical object co-occurrence patterns. With graph nodes V and graph edges E, the scene graph G(V, E) can be represented as:
[V, E] = M (E, G, O) , L = M G(V, E)(2)
Notably, to better guide the generation of 3D layouts, we further enrich the graph nodes by incorporating objects' coarse positions and orientations expressed in natural language. Specifically, the orientation is discretized into eight categories, namely front, back, left, right, left-front, left-back, right-front, and right-back, with a 45-degree quantization. The coarse positions correspond to a 3 × 3 grid of the table, including center, front, back, left-center, right-center, left-front, right-front, left-back, right-back. Therefore, given a task description, our spatial reasoning chain will prompt the model to sequentially reason about the scene composition, the spatial interrelationship, the scene graph, and ultimately, the 3D scene layout.
this section cite: []

Section: Reasoning Data Construction for Model Training.
To guide the model's reasoning process along our designed spatial reasoning chain, we construct massive reasoning data for training by using our collected tabletop scene dataset, MesaTask-10K. For a certain scene S inside, we first assign coarse positions and orientations for 3D objects on the table following the quantization rules above, infer the inter-object relations based on the 3D layouts, and finally obtain a complete scene graph G S . To compensate for the spatial relations missing in the scene graph, we also utilize a multimodal LLM (MLLM) like GPT-4o [1] to output a detailed scene description D based on a high-quality rendering image I of the tabletop scene and the scene graph. Given the scene graph G S and scene descriptions D, the multimodal LLM is prompted to generate a complete object list V and inter-object relations E, as well as the corresponding task instructions T, in particular including aforementioned detailed task informations z, i.e., [E, G, O] following:
D = MLLM(I, G S ), [T, E, G, O, V, E] = MLLM D, G S .(3)
this section cite: ['b0']

Section: LLM-based Framework for Tabletop Scene Generation
Our proposed MesaTask framework is a novel paradigm for tabletop scene generation, comprising an LLM-based model M for 3D layout generation and a post-processing module responsible for 3D asset retrieval. Given the constructed spatial reasoning data, we perform supervised fine-tuning (SFT) on the MesaTask model, thereby empowering it with the capability to reason about spatial relationships and generate structured 3D layouts from high-level task instructions. Despite the SFT on our high-quality data, the MesaTask model still generates suboptimal 3D layouts, including minor object collisions, unreasonable inter-object relationships misaligned with the task, and the omission of crucial task-relevant objects. Accordingly, we employ the Direct Preference Optimization (DPO) algorithm [21] to tackle such issues.
this section cite: ['b19']

Section: DPO Data Construction and Training.
To facilitate the DPO training, we construct massive training pairs with positive and negative 3D layouts. Here, positive data stands for the high-quality 3D layouts sourced from our dataset, MesaTask-10K, while the negative data is generated by intentionally corrupting the positive layouts in three distinct ways, each corresponding to a specific shortcoming of our MesaTask model after the SFT. For a 3D layout L from our dataset L + , we randomly select a subset of objects and perturb their positions, rotations, and sizes to deliberately create object collisions, resulting in the negative layout L - col reflecting the collisions. Then, some normal inter-object relations are damaged by altering their relation types, leading to a negative sample L - rel contradicting the task instruction. Finally, we manually remove one or more critical objects to create a negative layout L - obj that neglects task-relevant objects. Therefore, we can obtain a negative dataset L -with three distinct layout corruptions. Along with the corresponding task instructions T , we represent the whole paired dataset D for the DPO training with:
D = (L + , L -, T ), L -= {(L - col , L - rel , L - obj )} for L ∈ L (4
)
With the constructed dataset D, we optimize our MesaTask model via the DPO objective following
max π θ E (L + ,L -,T)∈D log σ β log π θ (L + | T) π ref (L + | T) -β log π θ (L -| T) π ref (L -| T) ,(5)
where π θ is the policy of the fine-tuned LLM, σ(•) represents the sigmoid function for preference scoring, and β is a temperature parameter that controls the sharpness of the preference margin between positive and negative layouts. With the DPO training, our MesaTask model seeks to acquire a policy π 0 that favors normal 3D layouts L + , thereby alleviating three limitations observed in the model after the SFT and consequently enhancing the overall quality of generated tabletop scenes.
this section cite: []

Section: Experiment

this section cite: []

Section: Experiment setup
Dataset. We build our training data based on the training split of our MesaTask-10k dataset, which contains 10, 000 tabletop scenes. For each scene, we generate five task instructions following the reasoning data creation process above, resulting in a total of 50, 000 task-scene pairs for the supervised fine-tuning. During the stage of DPO training, we construct the paired dataset using 5, 000 previously unseen scenes, where each normal layout sample corresponds to two disrupted layouts on average, thereby yielding a total of 10, 000 positive-negative layout pairs for the DPO training.
Implementation details. We adopt Qwen3-8b [31] as the base LLM for both supervised fine-tuning (SFT) and direct preference optimization (DPO). We perform full-parameter fine-tuning in both stages. In the SFT stage, the model is trained for one epoch using the learning rate of 1 × 10 -5 .
In the DPO stage, we train for one epoch, with the learning rate of 1 × 10 -6 . All experiments are conducted on a cluster of eight A800 GPUs. Baselines. We evaluate our model against two categories of benchmark methods. The first is closed-source large language models, specifically GPT-4o, where we perform our task in a zero-shot manner. The second category comprises modular scene generation methods, like Holodeck [37] and I-Design [2]. These approaches are originally targeted for indoor scene generation, and are now adapted to fit our task without changing their core frameworks, which are noted as Holodeck-table and I-Design-table here.
this section cite: ['b29', 'b35', 'b1']

Section: Metrics
We first employ Fréchet Inception Distance (FID) to measure the realism of the generated scenes and the success rate to reflect the syntactic correctness of the LLM-generated output format. A 100% success rate indicates that all the model's outputs are interpretable and can be directly parsed for downstream object retrieval, ultimately enabling the construction of tabletop scenes. Moreover, to conduct a more comprehensive evaluation, we propose the GPT-score, a metric designed to assess the multi-dimensional performance of generated scenes, including Consistency with Task (CwT), Object Size Reasonableness (OSR), Placement Plausibility & Intersections (PPI), Layout Coherence & Realism (LCR), and Object Visibility (OV).
this section cite: []

Section: Comparison to baselines
For a fair comparison, each method generates 500 tabletop scenes according to the corresponding task instructions, which will serve as the evaluation corpus for the aforementioned metrics.
Quantative evaluation. As shown in Table 1, our method MesaTask demonstrates superior overall performance across all evaluation protocols. In comparison to multiple baseline methods, MesaTask achieves significantly better performance in terms of FID, thereby indicating its capability to generate more realistic tabletop scenes. With respect to the GPT-based multi-dimensional metrics, MesaTask consistently outperforms alternative baseline methods, with particularly notable advantages in CwT and LCR. This superior performance reflects enhanced task-scene alignment and more plausible tabletop layouts, which can be attributed to the high-quality MesaTask-10K dataset we constructed.
To assess the perceptual quality of these generated scenes, a total of 127 participants are invited to conduct a comprehensive user study from three distinct assessment dimensions. The scores presented in Table 1 further confirm that our method achieves the most favorable outcomes in terms of human preference. More evaluation details are put in the supplementary materials.
this section cite: []

Section: Qualitative results
To further substantiate the superior performance of MesaTask, Figure 4 presents qualitative comparisons between our method and three representative baselines. MesaTask consistently produces more realistic and diverse scenes, with a greater number of objects arranged with semantically meaningful and spatially coherent layouts. Moreover, its outputs align more closely with task instructions, capturing nuanced spatial relations such as stacking, containment, and precise object relocation. In contrast, baseline methods often generate overly simplistic or symmetrical layouts, miss key objects, or struggle to interpret complex spatial commands. These qualitative results highlight MesaTask's impressive ability to model task-driven tabletop scenes with plausible layouts that support real-world robotic manipulation.
this section cite: []

Section: MesaTask GPT-4o I-Design-table Holodeck-table
Environment: A tabletop with various stationary items including markers, pens, books, and pot plants mostly centered and to the back middle.
Task: Group all markers and pens on the right side of the laptop.
this section cite: []

Section: Objects cluster: Marker, Pen, Laptop
Environment: A tabletop with a candle on a tray at center, cosmetic products on the left, and a potted plant with brushes on the right.
Task: Remove the candle from the tray and place it in front of the potted plant.
this section cite: []

Section: Objects cluster: Candle, Tray, Plant
Environment: Tabletop with a large succulent plant at center, smaller plant to left, bowl of fruit in front, and various candles and a pitcher.
Task: Arrange fruits from bowl onto the table near the large plant. Task: Create a neat corner by grouping the lamp, potted plants, and pen holder. SFT DPO Task: Organize fruit from bowl into empty bowl and place tray near lamp. Task: Put the metallic coffee pot to the left side of the plate.
this section cite: []

Section: Ablation study
To better understand the impact of each component in our framework, we conduct a comprehensive ablation study analyzing the contribution of spatial reasoning and preference tuning via Direct Preference Optimization (DPO). Table 1 presents the results of a quantitative ablation study conducted on MesaTask. The removal of either the spatial reasoning module or the DPO training component results in a measurable degradation of MesaTask's overall performance. The qualitative comparisons shown in Figure 5 illustrate the advantages brought by the supplementary DPO training. As observed, DPO effectively alleviates common failure modes exhibited by the SFT-only model, including severe object collisions, the absence of task-relevant objects, and erroneous inter-object relationships. These improvements validate the effectiveness of our adopted DPO training in ensuring coherent table layouts and correct interrelations, leading to realistic tabletop scenes that exhibit greater visual plausibility and enhanced functional fidelity to the given task instructions.
this section cite: []

Section: Generalization capability
To validate the generalization capability of our proposed method, we select tabletop categories not present in MesaTask-10K, including nightstands, TV stands, and side tables from household scenes, as well as cashier counters from shop scenes. For these four tabletop categories, we employ GPT-4o to generate plausible tasks, with 16 tasks assigned to each category. As shown in Table 2, MesaTask exhibits robust generalization capability when tested on the four unseen table categories. The performance of MesaTask across all metrics is comparable to its performance on the test set of six seen categories from MesaTask-10K, as listed in Table 1. Notably, in the case of cashier counters, even though cash registers are not included in MesaTask-10K, our method can accurately generate their descriptions and sizes while placing them correctly. Notably, we can not compute FID since these new scenes are not included in MesaTask-10K. Figure 6 additionally presents several generated tabletop scenes, which belong to these four novel tabletop categories.
this section cite: []

Section: Conclusion
In this paper, we introduce a novel task, namely task-oriented tabletop scene generation, which presents significant challenges owing to the substantial disparity between high-level task instructions and scene layouts. To support this demanding task, we propose a large-scale dataset, MesaTask-10K, consisting of roughly 10, 700 tabletop scenes that span six distinct indoor table categories. Thanks to our proposed spatial reasoning chain, our LLM-based framework MesaTask will sequentially reason about the scene composition, the spatial interrelationship, the scene graph, and ultimately, the 3D scene layout, based on which 3D assets are retrieved to form a complete tabletop scene.
In evaluations, MesaTask demonstrates superior performance over existing baselines in accurately conforming to task instructions and modeling complex inter-object relations. We believe our dataset and framework will inspire a promising research direction and unveil new challenges in this field.
Limitations and future work. MesaTask relies on 3D object retrieval, which naturally limits the object diversity to what's available in our 3D object database. In the future, we will explore integrating 3D object generation methods based on bounding box conditions into our tabletop scene generation pipeline. This should allow us to create various objects and more realistic tabletop scenes.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: I-design: Personalized llm interior designer Year: (2024)
Ref_id:b2 Title: Single-view 3d scene reconstruction with high-fidelity shape and texture Year: (2024)
Ref_id:b3 Title: Comboverse: Compositional 3d assets creation using spatially-aware diffusion guidance Year: (2024)
Ref_id:b4 Title: Manitaskgen: A comprehensive task generator for benchmarking and improving vision-language agents on embodied decision-making Year: (2025)
Ref_id:b5 Title: Automated creation of digital cousins for robust policy learning Year: (2024)
Ref_id:b6 Title: A universe of annotated 3d objects Year: (2022)
Ref_id:b7 Title: Layoutgpt: Compositional visual planning and generation with large language models Year: (2023)
Ref_id:b8 Title: Anyhome: Open-vocabulary generation of structured and textured 3d homes Year: (2024)
Ref_id:b9 Title: Reparo: Compositional 3d assets generation with differentiable 3d layout alignment Year: (2024)
Ref_id:b10 Title: Multi-instance diffusion for single image to 3d scene generation Year: (2024)
Ref_id:b11 Title: Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner Year: (2024)
Ref_id:b12 Title: Scenethesis: A language and vision agentic framework for 3d scene generation Year: (2025)
Ref_id:b13 Title: Towards high-fidelity single-view holistic reconstruction of indoor scenes Year: (2022)
Ref_id:b14 Title: Structdiffusion: Object-centric diffusion for semantic rearrangement of novel objects Year: (2022)
Ref_id:b15 Title: Structformer: Learning spatial structure for language-guided semantic rearrangement of novel objects Year: (2022)
Ref_id:b16 Title: Total3dunderstanding: Joint layout, object pose and mesh reconstruction for indoor scenes from a single image Year: (2020)
Ref_id:b17 Title: Isaac sim 4.0 -robotics simulation and synthetic data generation Year: (2024)
Ref_id:b18 Title: Atiss: Autoregressive transformers for indoor scene synthesis Year: (2021)
Ref_id:b19 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b20 Title: Grounded sam: Assembling open-world models for diverse visual tasks Year: (2024)
Ref_id:b21 Title: Rethinking the inception architecture for computer vision Year: (2016)
Ref_id:b22 Title: Evaluating semantic coherence in text-conditioned 3d indoor scene synthesis Year: (2025)
Ref_id:b23 Title: Diffuscene: Denoising diffusion models for generative indoor scene synthesis Year: (2024)
Ref_id:b24 Title: Drake: Model-based design and verification for robotics Year: (2019)
Ref_id:b25 Title: Architect: Generating vivid and interactive 3d scenes with hierarchical 2d inpainting Year: (2024)
Ref_id:b26 Title: Sapien: A simulated part-based interactive environment Year: (2020)
Ref_id:b27 Title: To-scene: A large-scale dataset for understanding 3d tabletop scenes Year: (2022)
Ref_id:b28 Title: set it up!": Functional object arrangement with compositional generative models Year: (2024)
Ref_id:b29 Title:  Year: (2024)
Ref_id:b30 Title: Depth anything v2 Year: (2024)
Ref_id:b31 Title: Depth anything v2 Year: (2024)
Ref_id:b32 Title: Physcene: Physically interactable 3d scene synthesis for embodied ai Year: (2024)
Ref_id:b33 Title: The 3d indoor scene layout generation and editing via large language model Year: (2024)
Ref_id:b34 Title: Llm-driven indoor scene layout generation via scaled human-aligned data synthesis and multi-stage preference optimization Year: (2025)
Ref_id:b35 Title: Language guided generation of 3d embodied ai environments Year: (2024)
Ref_id:b36 Title: Cast: Component-aligned 3d scene reconstruction from an rgb image Year: (2025)
Ref_id:b37 Title: Lvdiffusor: Distilling functional rearrangement priors from large models into diffusor Year: (2024)
Ref_id:b38 Title: Holistic 3d scene understanding from a single image with implicit representation Year: (2021)
Ref_id:b39 Title: Clay: A controllable large-scale generative model for creating high-quality 3d assets Year: (2024)
Ref_id:b40 Title: Internscenes: A large-scale simulatable indoor scene dataset with realistic layouts Year: (2025)
