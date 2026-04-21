Title: ORIGAMISPACE: Benchmarking Multimodal LLMs in Multi-Step Spatial Reasoning with Mathematical Constraints
Abstract: Spatial reasoning is a key capability in the field of artificial intelligence, especially crucial in areas such as robotics, computer vision, and natural language understanding. However, evaluating the ability of multimodal large language models (MLLMs) in complex spatial reasoning still faces challenges, particularly in scenarios requiring multi-step reasoning and precise mathematical constraints. This paper introduces ORIGAMISPACE, a new dataset and benchmark designed to evaluate the multi-step spatial reasoning ability and the capacity to handle mathematical constraints of MLLMs through origami tasks. The dataset contains 350 data instances, each comprising a strictly formatted crease pattern (CP diagram), the Compiled Flat Pattern, the complete Folding Process, and the final Folded Shape Image. We propose four evaluation tasks: Pattern Prediction, Multi-step Spatial Reasoning, Spatial Relationship Prediction, and End-to-End CP Code Generation. For the CP code generation task, we design an interactive environment and explore the possibility of using reinforcement learning methods to train MLLMs. Through experiments on existing MLLMs, we initially reveal the strengths and weaknesses of these models in handling complex spatial reasoning tasks.

Section: Introduction
Spatial reasoning is a core component of artificial intelligence [1,2], with wide applications in robotics [3], autonomous driving [4], and geographic information systems [5]. Although multimodal large language models (MLLMs) demonstrate outstanding performance in various vision and language tasks [6,7], they face challenges in imagining spatial transformations and grasping spatial relationships in image and text spaces. Evaluating their spatial reasoning ability has become an important task.
Multi-step reasoning and constraints are critical yet underexplored areas in spatial intelligence. Current spatial reasoning benchmarks typically focus on understanding static images or simple scenes [8]. Some studies are dedicated to comparing and reasoning about spatial relationships between image pairs, but lack attention to continuous spatial transformations [9,10]. Some studies propose multi-step spatial reasoning but do not involve interaction with the environment and lack constraints found in real-world tasks [11]. These limitations indicate a current need for a new benchmark to more comprehensively evaluate the capabilities of MLLMs in complex spatial reasoning scenarios.
Origami art offers an ideal platform for evaluating complex spatial reasoning abilities [12]. Origami involves a sequence of ordered folding operations, where each step depends on the result of the previous one, embodying the essence of multi-step reasoning. Furthermore, the origami process is governed by explicit geometric constraints, such as folds must occur along straight lines, and the paper cannot be torn or separated; all origami operations are defined by strict mathematical constraints (Kawasaki's Theorem, Huzita-Hatori axioms, etc.) [13,14]. The transformation from a two-dimensional crease pattern (CP diagram) through multiple folding steps to a three-dimensional folded shape image requires strong spatial imagination and reasoning abilities. To bridge the gap of existing benchmarks, this paper introduces the ORIGAMISPACE dataset and benchmark. This dataset contains 350 meticulously collected origami data instances, including a CP diagram, its corresponding compiled flattened pattern, illustrations of the complete folding process, and the final folded shape. The diversity and complexity of the data cover various origami types. We improve the existing origami compiler, enabling it to output detailed flattened diagrams that include crease locations and stacking relationships, support interactive simulation with MLLMs, and provide more comprehensive error feedback. Based on this dataset, we design four challenging evaluation tasks: pattern prediction, spatial relationship prediction, multi-step spatial reasoning, and end-to-end CP code generation, which comprise 1,500 multiple-choice questions and 120 code generation questions. For the code generation task, we meticulously design a comprehensive evaluation strategy to measure the quality of the generated CP code across multiple dimensions.
The core advantages of ORIGAMISPACE lie in its authenticity (derived from real origami designs), multi-step reasoning characteristics (reflecting the inherent process of origami), and rigorous mathematical constraints (precisely verifiable through origami theorems). We evaluate the performance of various MLLMs on ORIGAMISPACE, and introduce environmental learning and reinforcement learning methods for the code generation task, which opens up new perspectives and effective avenues for assessing and enhancing the spatial reasoning abilities of MLLMs.
The main contributions of this paper include:
• We introduce ORIGAMISPACE, a dataset containing 350 high-quality origami data instances, and optimize the existing origami compiler, enabling it to provide more comprehensive feedback.
• We design four challenging tasks centered around spatial reasoning, including 1,500 multiplechoice questions and 120 CP code generation questions, which is the first benchmark to evaluate the multi-step spatial reasoning ability of MLLMs under mathematical constraints.
• We conduct a comprehensive evaluation of existing MLLMs and develop a complete interactive environment for the end-to-end CP code generation task, and explore environmental learning and reinforcement learning methods through this environment.
Folded Shape Image Different from the strictly compiled flat pattern, the folded shape image provides a direct, intuitive visualization of the final origami shape. It is typically a photograph or 3D rendering.
Folding Process The folding process refers to the multi-step sequence of transforming the original paper into the final shape. This folding process is gathered from various origami tutorials and cannot be represented in a standardized format, existing only as natural images.
We manually check and verify all data to ensure that 1) all CP diagrams can be compiled into the compiled flat pattern and correspond to the folded shape image; 2) the names of all origami data correspond to the folded shape image, with no potential for confusion (such as indistinguishable birds); and 3) all folding processes are feasible. In addition to this part of the data, we also collect 471 groups of data without intermediate folding processes for the subsequent training of the model.
this section cite: ['b8']

Section: Compiler
The current origami compiler computes the final state achievable by a CP diagram under all mathematical constraints, thereby compiling the compiled flat pattern. We have optimized this process: 1) During compilation, we mark each crease, allowing us to locate the position of every crease in the compiled image. 2) We further compute the paper stacking order information, clarifying the top-bottom relationship of different paper regions in the compiled flat pattern. 3) We construct an interface for direct interaction between MLLMs and the compiler, enabling the model to call this system more conveniently to complete origami simulations. 4) We improve the error feedback system of the compiler. Specifically, it returns four types of errors:
this section cite: []

Section: CP Code Syntax Error (CSE)
Validates the existence, format, and validity of inter-references of core data structures in the CP code (such as vertex coordinates vertices_coords, edge-vertex relationships edges_vertices, and face-vertex relationships faces_vertices). It also checks if crease types (e.g., 'B', 'M', 'V', 'F', 'U') are predefined characters, and verifies if Euler's formula for planar graphs is satisfied: V -E + F = 2, where V, E, and F represent the number of vertices, edges, and faces, respectively.
this section cite: []

Section: Geometrically Impossible Fold (GIF)
Refers to cases where the CP code geometrically violates fundamental origami principles, making the fold physically unrealizable. For example, violating local flat-foldability conditions at a vertex (such as Maekawa's theorem |M -V | = 2 or Kawasaki's theorem α i = 2π), or specified crease angle combinations would require the paper to be stretched or torn.
Paper Self-Intersection/Penetration (PSI) Occurs when logically incompatible situations are found while deducing the relative positions and layering order of different paper sections after folding. This may manifest as a cycle in the calculated paper layering relationships (e.g., layer A is above layer B, layer B is above layer C, and layer C is, in turn, above layer A), or in a 2D unfolded representation, different paper regions are assigned to overlapping positions that would cause physical penetration.
this section cite: []

Section: Ambiguous Folding State (AFS)
This error occurs when a given CP code, due to its inherent under-constrained nature (e.g., allowing multiple valid mountain-valley assignments for creases, or lacking critical information such as crease types or angles), can be compliantly folded into multiple different stable geometric structures, or prevents the compiler from uniquely determining the layering order when processing complex overlapping paper regions.
this section cite: []

Section: Dataset Statistics
In ORIGAMISPACE, the distribution of different types of origami is relatively even. To ensure data diversity, we choose origami models covering different levels of complexity and types of folds, such as animals, plants, geometric shapes, etc. The average number of folding steps for origami models is 8.2, but the variation between different models varies greatly, ranging from a minimum of 3 steps to a maximum of 25 steps. Appendix A presents more detailed data analysis, including the themes and names of all origami data and the proportion of different folding steps.
this section cite: []

Section: Task
Based on ORIGAMISPACE, we propose four tasks to evaluate the spatial reasoning capabilities of MLLMs comprehensively.
this section cite: []

Section: Pattern Prediction
This task evaluates the model's ability to understand the folding process from the CP diagram and imagine the final 3D shape. For this task, the input is the CP diagram, and MLLMs are required to predict the resulting folded shape image based on it. To enable better quantitative evaluation, we structure this task as a multiple-choice question. The correct option is the name of the target shape.
For the incorrect options, three origami enthusiasts design three options for each diagram, adhering to criteria that require them to be easily distinguishable from the correct option; not be variations of the same concept (e.g., if the correct option is a cat, incorrect options are not lions, leopards, etc.); and be close to potential folded states based on the CP diagram (e.g., removing a few key creases makes a boat's CP diagram similar to a hat). We create 350 questions for this task. See Appendix B.1 for the specific annotation rules.
this section cite: []

Section: Multi-step Spatial Reasoning
This task evaluates the model's ability to understand the dynamic origami process and the logical relationships between steps. The input for the task is a set of images that collectively show several key steps of a complete origami process. However, the order of these images is randomly shuffled. MLLMs need to infer the correct chronological order in which these steps occur, based on their understanding of the geometric state changes in the images. To better quantify the model's performance, we structure this task as a multiple-choice question. The correct option is the sequence of steps that represents the unique correct folding process (for example, "1-2-3-4"). For the incorrect options, we generate multiple logically incorrect sequences of steps (for example, "1-2-4-3", "4-1-2-3", etc.). These incorrect sequences may contain partially correct local orders but contain errors in the overall flow, in order to test the model's grasp of the complete, coherent process. We design 250 such questions, and the average number of steps per question is 7.5.
this section cite: []

Section: Spatial Relationship Prediction
This task evaluates the model's ability to predict spatial relationships and geometric properties after the folding process is complete. For this task, the input is the CP diagram. The model is required to predict specific spatial relationships between parts of the origami model after it is fully folded. The task comprises three types of multiple-choice questions designed to test this ability: 1) Spatial Pose Localization: Determining the specific 3D position of a point from the original paper in the final model, including its pose within a reference frame (e.g., on a table, facing upwards). 2) Layering Relationship Analysis: Determining the paper stacking order after folding, requiring analysis of covering relationships during the folding process and identifying how many paper layers form a specific region (e.g., the thickest region). 3) Geometric Change Analysis: Predicting how specific geometric features (such as angles, distances, areas, etc.) change from the flat CP diagram to the final folded state. For example, predicting the relative angle or spatial distance between two original line segments after folding. The correct answers for all three question types are obtained using our optimized compiler. Incorrect options are then manually designed. We design 900 multiple-choice questions (300 for each type) for this task. See Appendix B.2 for specific annotation rules.
this section cite: []

Section: End-to-End CP Code Generation
This task requires the MLLM to generate corresponding CP code based on a compiled flat layout and an image of the folded shape. Ideally, this CP code should compile into a folded pattern identical to the target shape. To comprehensively evaluate the quality of the generated results, we have designed a multidimensional evaluation framework.
this section cite: []

Section: Compilation Attempt and Evaluation
The CP code generated by the model will first be attempted to be compiled using our origami compiler (see Section 3.2 for details). If the compilation fails, the model will return one or more error types. If the compilation succeeds, meaning the CP code is syntactically valid, geometrically foldable, and free of self-intersections, and produces a definite folded state, the system will compare the compilation result with the reference result across the following four dimensions:
s v = e -0.5 |Vgen-V ref | min(Vgen,V ref )
), edge connectivity (e.g., similarity of degree distribution, number of connected components), face relationships (e.g., number of faces, distribution of face sizes), and the distribution similarity of crease types ("M", "V", "B", etc.).
2) Geometric Similarity (GS) This dimension focuses on the spatial characteristics of the compiled model. It evaluates point position similarity by calculating the bidirectional Hausdorff distance dH between the normalized 3D point sets of the generated and reference compiled models (score s p = e -k•d H , where k is a sensitivity coefficient, e.g., 5). It assesses angular similarity by comparing the distribution of dihedral angles at the creases, and evaluates size and proportion similarity by comparing the aspect ratios of the overall bounding boxes of the models.
this section cite: []

Section: 3) Constraint Satisfaction (CS)
This dimension evaluates whether the successfully compiled CP code, beyond the basic foldability ensured by the compiler, further adheres to the physical and mathematical constraints of origami. This includes comparing the presence and matching degree of critical constraint types (Taco-Taco, Taco-Tortilla, transitivity constraints) and checking for satisfaction of fundamental theorems of local flat-foldability, such as Maekawa's theorem (the difference between the number of mountain creases M and valley creases V around a vertex is |M -V | = 2) and Kawasaki's theorem (the sum of the angles α i of creases around a vertex is α i = 2π or 0).
this section cite: []

Section: 4) Final Folded State (FFS)
This dimension directly compares the final 3D model shape compiled from the generated CP with the reference compiled 3D model. It primarily evaluates overall shape similarity by calculating the Hausdorff distance of the point sets, and where possible (if the model provides layering information), compares the layering relationships between facets, including paper stacking order information that may be obtained during the compilation process.
Total Score: The final total score S total is a weighted average of the scores s dim from each evaluation dimension: S total = dim w dim • s dim . By default, each of the four dimensions accounts for 25% of the weight (w dim = 0.25), and w dim = 1. This score ranges from 0 to 1 (S total ∈ [0, 1]), reflecting the overall quality of the generated CP code. For more details on the evaluation process, please refer to Appendix D.
this section cite: []

Section: Model
Pattern Prediction Multi-step Spatial Reasoning Spatial Relationship Prediction Spatial Pose Localization Layering Relationship Geometric Change Open-source Models MiniCPM-o 2.6 26.99±0.42 30.11±1.54 28.98±0.88 30.50±1.00 23.75±0.09 llava-1.5-7b 27.23±1.47 29.05±1.90 29.06±2.71 30.94±0.97 25.51±0.57 deepseek-vl2 28.40±0.07 30.01±0.06 26.71±1.40 29.05±0.23 24.30±1.10 NVILA-15B 28.33±1.09 32.51±0.90 30.60±1.22 31.00±1.53 26.48±0.76 VideoLLaMA3-7B 29.01±1.23 30.86±0.14 29.06±0.02 28.74±1.04 27.80±0.35 Qwen2.5-VL-7B 28.40±0.82 31.51±0.30 28.43±0.08 28.05±0.04 28.83±0.72 Qwen2.5-VL-32B 34.15±0.39 36.82±0.48 33.51±0.99 32.59±0.48 30.51±0.15 Qwen2.5-VL-72B 36.29±0.11 39.10±0.88 35.68±1.69 38.04±0.70 31.89±0.85 InternVL2.5-78B 36.76±0.75 38.55±0.08 38.01±0.11 37.66±0.13 32.48±0.48 Close-source Models Claude-3.5-Sonnet 35.89±1.47 45.07±0.64 39.55±0.63 40.19±0.11 39.73±0.10 GPT-4o 42.71±0.66 51.81±0.48 48.24±1.73 50.42±0.59 46.72±0.50 Gemini2.5-Flash 35.01±0.16 48.92±0.13 40.15±0.60 39.91±1.09 40.01±1.63 Gemini2.5-pro 42.68±0.14 53.45±0.74 49.06±0.07 47.68±0.07 47.10±0.82 Human Performance human(common) 51.18 88.52 55.12 50.55 50.15 human(expert) 98.45 100.00 96.44 92.10 85.38 Table 1: Accuracy (%) of various MLLMs on different spatial reasoning tasks. Bold or underlined values indicate best performance across open-source models and all models, respectively.
this section cite: []

Section: Experiments

this section cite: []

Section: Models
We evaluate multiple representative MLLMs. For open-source models, we evaluate MiniCPM-o 2.6 [25],NVILA-15B [26], llava-1.5-7b [27], VideoLLaMA3 [28], Qwen2.5-VL-[7B/32B/72B] [29], deepseek-vl2 [30], InternVL2.5-78B [31]. For proprietary models, we evaluate Claude-3.5-Sonnet [32], gpt-4o [33], Gemini2.5-[flash/pro] [34]. For all these models, we adopt the original model and official instruction formats.
this section cite: ['b29']

Section: Baseline
We recruit two categories of people to complete the first three tasks. The first category consists of five laypersons recruited via a crowdsourcing platform, and the second category comprises three experts with extensive origami experience. Specific details of the human evaluation are provided in Appendix B.3. For the CP code generation task, we adopt the following settings:
In-context learning In this setting, we provide the model with detailed task instructions and a set of CP code examples. The instructions will introduce the meaning represented by each part of the CP code and all the constraints that must be followed. MLLMs need to generate the complete CP code in one go based on these instructions and examples.
this section cite: []

Section: Environmental learning
In this setting, MLLMs no longer attempt to generate the complete CP code in one go, but instead engage in iterative interaction with the compiler. Specifically, the MLLM will first perform planning, then generate CP code. The compiler will return its compilation result, and the model then performs inference based on the returned compilation result, subsequently choosing to add or delete creases, iterating in this manner. We set the upper limit of interaction rounds to 10.
Reinforcement learning Through a constructed compilation environment, we explore a reinforcement learning approach. We utilize the 471 sets of data mentioned in Section 3.1 for training, sampling data in the same process as in environmental learning. The reward mechanism is set as follows: (1) for ease of presentation.
Model
Intermediate reward: After modifying the code, if compilation is successful, a reward is given based on the quality progress of the current partial CP code (S partial -S partial_prev , where S partial is a quickly evaluated partial quality score), plus a small basic compilation success reward. If compilation fails, a fixed negative penalty is given. (2) Step penalty: A small negative reward is received for each action taken to encourage efficiency. (3) Final reward: After the interaction ends, the result of the evaluation function defined in Section 4.4 serves as the main reward. We adopt TRICO [35] for training on qwen2.5-vl-32B, which is a PPO-based [36], more efficient MLLMs multi-turn reinforcement learning algorithm. Specific training settings and parameters can be found in Appendix E.
this section cite: []

Section: Main Results
Tasks 1 to 3 primarily focus on spatial analysis and prediction. The results shown in Table 1 are the average of three runs for different MLLMs, from which we observe that: 1) For MLLMs, ORIGAMISPACE is a challenging task; the performance of poor-performing models is close to random guessing (25%), and even for the best-performing models, there is a significant gap compared to human performance, especially in multi-step spatial reasoning. 2) Despite the different task types, the relative performance ranking of various models largely remains consistent, with Gemini 2.5-pro and GPT-4o demonstrating the best spatial reasoning ability. 3) Human experts perform well on all tasks, demonstrating the task's upper bound. 4) MLLMs perform worst on the Spatial Relationship Prediction task, especially the sub-tasks involving Geometric Change, indicating significant difficulty for models in understanding fine-grained, internal spatial structures.
Table 2 presents the results of different methods and models on Task 4. We observe the following: 1) Impact of learning settings: The results clearly indicate the significant impact of learning settings on performance. In-context learning shows relatively limited performance. Environmental learning brings significant performance improvements, demonstrating that through iterative interaction with the compiler, planning, and trial-and-error based on feedback, models can overcome the limitations of one-shot generation. Reinforcement learning shows potential, as the trained Qwen2.5-VL-32B surpassed the performance of a 72B model. 2) There are significant performance differences among different models, with top-tier closed-source models exhibiting the best spatial reasoning capabilities.
Figure 3: The impact of interaction rounds on the compilation pass rate and total score of different models.
this section cite: []

Section: Impact of Mathematical Constraints
Mathematical constraints present a primary challenge in generating valid CP codes for the ORIGAMIS-PACE task. Table 2 indicates that failing to satisfy constraints is the main bottleneck for compilation failures; even when provided with detailed instructions, models struggle to strictly adhere to these complex rules, leading to persistently high compilation failure rates. Interactive processes with the environment enhance models' ability to follow constraints, demonstrating that models can learn and internalize rules from feedback. Compared to environmental learning, reinforcement learning also shows improvement in constraint satisfaction, proving the effectiveness of specific reward mechanisms. However, even with interactive learning, precisely satisfying all mathematical constraints remains a significant challenge for top-tier models (such as GPT-4o and Gemini 2.5-pro, whose constraint satisfaction score is only 56.99% under environmental learning settings). This reveals MLLMs' deficiencies in deep multi-step geometric and layering reasoning and highlights the value of the fine-grained feedback and constraint satisfaction evaluation introduced in this study.
this section cite: []

Section: Impact of Interaction Rounds in Environmental Learning
Figure 3 illustrates the impact of interaction rounds on model performance across different dimensions under the environmental learning setting. We observe that as the number of interaction rounds increases, model performance improves in various aspects, particularly the compilation pass rate. However, performance tends to saturate after 8-10 rounds, indicating that interaction primarily helps overcome initial learning obstacles but struggles to break through the model's inherent bottlenecks. Weaker models, limited by their understanding capabilities, reach their upper limit in fewer rounds. The reinforcement learning-trained Qwen2.5-VL-32B also follows a similar trend, but due to policy optimization, it may reach its performance ceiling in fewer rounds.
this section cite: []

Section: Conclusion
In this paper, we introduce ORIGAMISPACE, a novel benchmark specifically designed to address the underexplored areas of multi-step spatial reasoning and constraint adherence in Multimodal Large Language Models (MLLMs). Leveraging the inherent complexities of origami, ORIGAMISPACE provides 350 meticulously curated data instances and an enhanced compilation program to facilitate in-depth evaluation. The benchmark features four challenging tasks, including pattern prediction, spatial relationship prediction, multi-step spatial reasoning, and end-to-end code generation, making it the first to assess MLLMs' multi-step spatial reasoning under rigorous mathematical constraints. Our comprehensive evaluation of existing MLLMs and exploration of reinforcement learning methods for code generation highlight the utility of ORIGAMISPACE in not only assessing current capabilities but also in paving new ways to enhance the spatial intelligence of MLLMs.
this section cite: []

Section: References
Ref_id:b0 Title: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities Year: (2024)
Ref_id:b1 Title: Mind the Gap: Benchmarking Spatial Reasoning in Vision-Language Models Year: (2025)
Ref_id:b2 Title: RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics Year: (2024)
Ref_id:b3 Title: Lidar-llm: Exploring the potential of large language models for 3d lidar understanding Year: ()
Ref_id:b4 Title: Developing Spatial Thinking through the Earthcomm Learning Model: Exploring the Role of Earth Science in the Community Year: (2024)
Ref_id:b5 Title: Mm-llms: Recent advances in multimodal large language models Year: (2024)
Ref_id:b6 Title: The revolution of multimodal large language models: a survey Year: (2024)
Ref_id:b7 Title: Benchmark evaluations, applications, and challenges of large vision language models: A survey Year: (2025)
Ref_id:b8 Title: CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning Year: (2016)
Ref_id:b9 Title: Super-CLEVR: A Virtual Benchmark to Diagnose Domain Robustness in Visual Reasoning Year: (2023)
Ref_id:b10 Title: LEGO-Puzzles: How Good Are MLLMs at Multi-Step Spatial Reasoning? Year: (2025)
Ref_id:b11 Title: Origami engineering Year: (2024)
Ref_id:b12 Title: The Kawasaki identity and the fluctuation theorem Year: (2004)
Ref_id:b13 Title: Origami axioms and circle extension Year: (2011)
Ref_id:b14 Title: Vlue: A multi-task benchmark for evaluating vision-language models Year: (2022)
Ref_id:b15 Title: Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations Year: (2016)
Ref_id:b16 Title: NLVR2 Visual Bias Analysis Year: (2019)
Ref_id:b17 Title: StepGame: A New Benchmark for Robust Multi-Hop Spatial Reasoning in Texts Year: (2022)
Ref_id:b18 Title: Recent results in computational origami Year: (2002)
Ref_id:b19 Title: A computational algorithm for origami design Year: (1996)
Ref_id:b20 Title: Self-foldability of rigid origami Year: (2017)
Ref_id:b21 Title: Architected origami materials: how folding creates sophisticated mechanical properties Year: (2019)
Ref_id:b22 Title: Using origami design principles to fold reprogrammable mechanical metamaterials Year: (2014)
Ref_id:b23 Title: Generalization of rigid-foldable quadrilateral-mesh origami Year: (2009)
Ref_id:b24 Title: MiniCPM-V: A GPT-4V Level MLLM on Your Phone Year: (2024)
Ref_id:b25 Title: NVILA: Efficient Frontier Visual Language Models Year: (2024)
Ref_id:b26 Title: LLaVA-OneVision: Easy Visual Task Transfer Year: (2024)
Ref_id:b27 Title: VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs Year: (2024)
Ref_id:b28 Title:  Year: (2025)
Ref_id:b29 Title: DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding Year: (2024)
Ref_id:b30 Title: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling Year: (2025)
Ref_id:b31 Title:  Year: (2024)
Ref_id:b32 Title: Introducing GPT Year: (2024)
Ref_id:b33 Title: Gemini 1.5 Pro and Flash: A step forward in multimodal understanding Year: (2024)
Ref_id:b34 Title: VAGEN: Training VLM Agents with Multi-Turn Reinforcement Learning Year: (2025)
Ref_id:b35 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b36 Title: Square base fold -Geometry -4 Year: ()
Ref_id:b37 Title:  Year: ()
Ref_id:b38 Title: Letter I -Alphabet -3 Year: ()
Ref_id:b39 Title: Minus Sign -Symbols -3 Year: ()
Ref_id:b40 Title:  Year: ()
Ref_id:b41 Title: Letter L -Alphabet -4 Year: ()
Ref_id:b42 Title: Cross Mark/X -Symbols -4 Year: ()
Ref_id:b43 Title:  Year: ()
Ref_id:b44 Title: Diamond shape -Geometry -5 Year: ()
Ref_id:b45 Title: Water Drop -Nature -5 Year: ()
Ref_id:b46 Title: Trapezoid -Geometry -5 Year: ()
Ref_id:b47 Title: Lucky Star strip prep -Decorations -5 16. Comma symbol -Symbols -5 Year: ()
Ref_id:b48 Title: Single French Fry -Food -5 18. Letter C -Alphabet -5 Year: ()
Ref_id:b49 Title: Fish Fin -Animals -5 20. Check Mark -Symbols -5 Year: ()
Ref_id:b50 Title: Small Flag -Decorations -6 24. Simple Leaf -Plants -6 Year: ()
Ref_id:b51 Title: Arrow -Symbols -6 Year: ()
Ref_id:b52 Title: Band-aid -Items -6 Year: ()
Ref_id:b53 Title: Screw -Tools -6 28. Letter F -Alphabet -6 Year: ()
Ref_id:b54 Title: Number 2 -Numbers -6 30 Year: ()
Ref_id:b55 Title: Bread Slice -Food -6 32. Plate -Items -6 Year: ()
Ref_id:b56 Title: Simple Cloud -Nature -6 34 Year: ()
Ref_id:b57 Title: Ice Lolly/Popsicle Stick -Food -6 36 Year: ()
Ref_id:b58 Title: Pointed Bookmark -Items -7 38 Year: ()
Ref_id:b59 Title:  Year: ()
Ref_id:b60 Title: Simple Pen/Pencil outline -Items -7 44 Year: ()
Ref_id:b61 Title:  Year: ()
Ref_id:b62 Title: Letter Z -Alphabet -7 48. Simple Fish -Animals -7 Year: ()
Ref_id:b63 Title: Ice Cream Cone base -Food -7 54. Pointy Hat -Clothing -7 Year: ()
Ref_id:b64 Title: Crescent Moon -Nature -7 56 Year: ()
Ref_id:b65 Title: Simple Ghost -Decorations -7 58 Year: ()
Ref_id:b66 Title: Cheese Slice -Food -7 60 Year: ()
Ref_id:b67 Title: Simple Cup -Items -8 62 Year: ()
Ref_id:b68 Title: Simple Pig Face -Animals -8 66 Year: ()
Ref_id:b69 Title: Croissant shape -very simple -Food -8 78. Egg shape -flat -Food -8 Year: ()
Ref_id:b70 Title: Lollipop -circle on stick -Food -8 82. Simple Hammer outline -Tools -8 Year: ()
Ref_id:b71 Title: Simple Saw outline -Tools -8 84 Year: ()
Ref_id:b72 Title: Simple Bow -Decorations -8 86 Year: ()
Ref_id:b73 Title: Simple Book -Items -9 88. Snail Shell -Animals -9 Year: ()
Ref_id:b74 Title: Simple Snake -Animals -9 90. Tulip Head -Plants -9 Year: ()
Ref_id:b75 Title: Simple Shield -Toys -9 Year: ()
Ref_id:b76 Title: Bird Silhouette -very simple -Animals -9 Year: ()
Ref_id:b77 Title: Square Coaster -Items -9 94. Flat Christmas Tree -Plants -9 Year: ()
Ref_id:b78 Title: Fishbone -Animals -9 Year: ()
Ref_id:b79 Title: Lemon slice -Food -9 Year: ()
Ref_id:b80 Title: Donut -flat with hole -Food -9 Year: ()
Ref_id:b81 Title: Pretzel shape -very simple -Food -9 Year: ()
Ref_id:b82 Title: Fried Egg -flat -Food -9 102. Hot Dog in bun -flat -Food -9 Year: ()
Ref_id:b83 Title: Sushi Roll -simple cylinder end -Food -9 104. Tea Bag with string -Food -9 Year: ()
Ref_id:b84 Title: Simple Vase outline -Items -9 106. Simple Wrench outline -Tools -9 Year: ()
Ref_id:b85 Title: Simple Axe outline -Tools -9 108. Dinosaur Egg -Animals -9 Year: ()
Ref_id:b86 Title: Bow Tie -Clothing -9 110. Candy Cane -Food -9 Year: ()
Ref_id:b87 Title: Letter J -Alphabet -9 112. Stop Sign -octagon shape -Symbols -9 Year: ()
Ref_id:b88 Title: Medium Difficulty Origami Models Year: ()
Ref_id:b89 Title:  Year: ()
Ref_id:b90 Title: Sun with rays Year: ()
Ref_id:b91 Title: House with roof Year: ()
Ref_id:b92 Title:  Year: ()
Ref_id:b93 Title: Shuriken -single piece Year: ()
Ref_id:b94 Title: Watermelon Slice -Food -10 18. Banana -Food -10 Year: ()
Ref_id:b95 Title:  Year: ()
Ref_id:b96 Title:  Year: ()
Ref_id:b97 Title: Ring with simple gem Year: ()
Ref_id:b98 Title: Letter B -Alphabet -10 Year: ()
Ref_id:b99 Title: Modular Box Corner Unit -simple -Modular -10 Year: ()
Ref_id:b100 Title: Thermometer -simple -Items -10 Year: ()
Ref_id:b101 Title: Letter H -Alphabet -10 30 Year: ()
Ref_id:b102 Title: Caterpillar -simple segments Year: ()
Ref_id:b103 Title: Letter K Year: ()
Ref_id:b104 Title: Simple Sofa -front view -Furniture -10 Year: ()
Ref_id:b105 Title: Pouch -simple -Items -11 Year: ()
Ref_id:b106 Title: Seagull -simple flying -Animals -11 Year: ()
Ref_id:b107 Title: Slipper -flat -Clothing -11 44. Mobile Phone -flat -Items -11 Year: ()
Ref_id:b108 Title: Simple Bed -top view -Furniture -11 50 Year: ()
Ref_id:b109 Title: Crane -traditional -Animals -12 Year: ()
Ref_id:b110 Title: Lantern -simple flat -Items -12 Year: ()
Ref_id:b111 Title: Wallet/Coin Purse -simple -Items -12 Year: ()
Ref_id:b112 Title: Samurai Helmet/Kabuto -Traditional -12 Year: ()
Ref_id:b113 Title: Cupcake paper -Food -12 Year: ()
Ref_id:b114 Title: Gingerbread Man -flat -Food -12 Year: ()
Ref_id:b115 Title: Geometric Pattern tile -Geometry -12 Year: ()
Ref_id:b116 Title: Cake Slice -flat -Food -12 72. Fish Bowl -flat simple -Items -12 Year: ()
Ref_id:b117 Title: Letter G -Alphabet -12 Year: ()
Ref_id:b118 Title: Pirate Hat -simple flat -Clothing -12 Year: ()
Ref_id:b119 Title: Letter M -Alphabet -12 Year: ()
Ref_id:b120 Title: Simple Street Lamp post -Items -12 Year: ()
Ref_id:b121 Title: Jumping Frog base -Animals -13 78 Year: ()
Ref_id:b122 Title: Ice Cream with scoop -Food -13 Year: ()
Ref_id:b123 Title: Tent -A-frame -Items -13 Year: ()
Ref_id:b124 Title: Chef Hat -simple flat -Clothing -13 Year: ()
Ref_id:b125 Title:  Year: ()
Ref_id:b126 Title: Chicken -simple -Animals -14 Year: ()
Ref_id:b127 Title: Grapes -simple bunch -Food -14 96. Tulip with stem -Plants -14 Year: ()
Ref_id:b128 Title: Crown -fuller -Clothing -14 Year: ()
Ref_id:b129 Title: Gift Box -flat with bow -Decorations -14 Year: ()
Ref_id:b130 Title: Baseball Cap -flat -Clothing -14 100 Year: ()
Ref_id:b131 Title: Finger Puppet Bear -Toys -14 102 Year: ()
Ref_id:b132 Title: Hamburger -simple layers -Food -14 Year: ()
Ref_id:b133 Title: Dog House -simple front -Items -14 Year: ()
Ref_id:b134 Title: Mailbox -simple -Items -14 106 Year: ()
Ref_id:b135 Title: Snake -Coiled Snake -Animals -15 110. Lion Face -Animals -15 Year: ()
Ref_id:b136 Title: Tiger Face -Animals -15 112 Year: ()
Ref_id:b137 Title: Diamond shape -faceted look -Decorations -15 120. Modular Star -3 simple points -Decorations -15 Year: ()
Ref_id:b138 Title: Halloween Bat -hanging -Decorations -15 Year: ()
Ref_id:b139 Title: Pen Holder -very simple cylinder -Items -15 Year: ()
Ref_id:b140 Title: Ladies Hat -wide brim simple -Clothing -15 Year: ()
Ref_id:b141 Title: Lock -simple -Items -16 130 Year: ()
Ref_id:b142 Title: Teapot -simple flat -Items -16 Year: ()
Ref_id:b143 Title: Compass Rose -4 points -Symbols -16 Year: ()
Ref_id:b144 Title: Guitar -simple flat -Musical Instruments -17 Year: ()
Ref_id:b145 Title: Snowflake -simple 6-point -Decorations -17 Year: ()
Ref_id:b146 Title:  Year: ()
Ref_id:b147 Title: D Star -simple module -Decorations -18 Year: ()
Ref_id:b148 Title: Tissue Box Cover -simple sleeve -Items -18 Year: ()
Ref_id:b149 Title: Difficult Origami Models Year: ()
Ref_id:b150 Title: Elephant -standing -Animals -20 Year: ()
Ref_id:b151 Title: L-shape Year: ()
Ref_id:b152 Title: Torch with flame -Items -20 Year: ()
Ref_id:b153 Title: Lidded Box -separate lid & base Year: ()
Ref_id:b154 Title: Spaceship -simple rocket style -Vehicles -22 Year: ()
Ref_id:b155 Title: Pyramid -more detailed base -Architecture -22 Year: ()
Ref_id:b156 Title: Sailboat -more detailed -Vehicles -23 Year: ()
Ref_id:b157 Title: Hourglass shape -Items -23 Year: ()
Ref_id:b158 Title: Dog Toy -squeaky bone shape -Toys -23 Year: ()
Ref_id:b159 Title: Floor Lamp -Furniture -23 Year: ()
Ref_id:b160 Title: Hot Air Balloon -simple 3D -Vehicles -24 Year: ()
Ref_id:b161 Title: Camera -simple 3D body -Items -24 24 Year: ()
Ref_id:b162 Title: Bridge -simple arch -Architecture -24 Year: ()
Ref_id:b163 Title: Vase -with some shaping -Items -24 Year: ()
Ref_id:b164 Title: Shoji Screen -simple panel -Traditional -24 Year: ()
Ref_id:b165 Title: Horse -standing -Animals -25 Year: ()
Ref_id:b166 Title: Bee -detailed wings -Animals -25 32. Owl -with features -Animals -25 Year: ()
Ref_id:b167 Title: Treasure Chest -simple -Items -25 Year: ()
Ref_id:b168 Title: Church -simple front -Architecture -25 Year: ()
Ref_id:b169 Title: Microphone with stand base -Items -25 Year: ()
Ref_id:b170 Title: Rook chess piece shape -Toys -25 38 Year: ()
Ref_id:b171 Title: Photo Frame -standing type -Items -25 Year: ()
Ref_id:b172 Title: Sofa -more detailed -Furniture -25 Year: ()
Ref_id:b173 Title: Panda -sitting -Animals -26 42. Kangaroo with joey pouch outline Year: ()
Ref_id:b174 Title: Eiffel Tower -simplified flat -Landmarks -26 Year: ()
Ref_id:b175 Title: Tent -more complex dome like -Items -26 Year: ()
Ref_id:b176 Title: Turtle -with shell detail -Animals -27 Year: ()
Ref_id:b177 Title: Eagle -spread wings -Animals -27 Year: ()
Ref_id:b178 Title: Windmill building with vanes -Architecture -27 Year: ()
Ref_id:b179 Title: Violin -simplified profile -Musical Instruments -27 Year: ()
Ref_id:b180 Title: Backpack -with straps -Items -27 Year: ()
Ref_id:b181 Title: Dragonfly -more detailed -Animals -27 Year: ()
Ref_id:b182 Title: Lion -standing -Animals -28 Year: ()
Ref_id:b183 Title: Deer/Stag -Animals -28 Year: ()
Ref_id:b184 Title: Crocodile/Alligator -simple form -Animals -28 Year: ()
Ref_id:b185 Title: Pentagonal Box -simple -Items -28 Year: ()
Ref_id:b186 Title: Train Engine -simple profile -Vehicles -28 Year: ()
Ref_id:b187 Title: Castle -simple front -Architecture -28 Year: ()
Ref_id:b188 Title: Old Telephone -receiver and body -Items -28 Year: ()
Ref_id:b189 Title: Saxophone -simplified profile -Musical Instruments -28 Year: ()
Ref_id:b190 Title: Accordion -simplified -Musical Instruments -28 Year: ()
Ref_id:b191 Title: Butterfly -more detailed -Animals -28 Year: ()
Ref_id:b192 Title: Christmas Wreath -simple modular -Decorations -28 Year: ()
Ref_id:b193 Title: Unicorn -simple standing -Animals -28 Year: ()
Ref_id:b194 Title: Laptop -open -Items -28 Year: ()
Ref_id:b195 Title: Peacock -simplified tail -Animals -29 Year: ()
Ref_id:b196 Title: Fire Truck -basic shape -Vehicles -29 Year: ()
Ref_id:b197 Title: Police Car -basic shape -Vehicles -29 Year: ()
Ref_id:b198 Title: Ambulance -basic shape -Vehicles -29 Year: ()
Ref_id:b199 Title: Grand Piano -simplified top view -Musical Instruments -29 Year: ()
Ref_id:b200 Title: Ice Cream Truck -simple profile -Vehicles -29 Year: ()
Ref_id:b201 Title: Octopus -with 8 simple tentacles -Animals -30 Year: ()
Ref_id:b202 Title: Scorpion -Animals -30 Year: ()
Ref_id:b203 Title: Dinosaur T-Rex -simple standing -Animals -30 Year: ()
Ref_id:b204 Title: Hexagonal Box -simple -Items -30 Year: ()
Ref_id:b205 Title: Bicycle -very simplified profile -Vehicles -30 Year: ()
Ref_id:b206 Title: Motorcycle -very simplified profile -Vehicles -30 Year: ()
Ref_id:b207 Title: Pirate Ship -simplified -Vehicles -30 Year: ()
Ref_id:b208 Title: Double Decker Bus -simple profile -Vehicles -30 Year: ()
Ref_id:b209 Title: Reindeer -simple standing -Animals -30 Year: ()
