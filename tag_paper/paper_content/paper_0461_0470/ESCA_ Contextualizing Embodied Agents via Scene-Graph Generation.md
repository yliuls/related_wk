Title: ESCA: Contextualizing Embodied Agents via Scene-Graph Generation
Abstract: Multi-modal large language models (MLLMs) are making rapid progress toward general-purpose embodied agents. However, existing MLLMs do not reliably capture fine-grained links between low-level visual features and high-level textual semantics, leading to weak grounding and inaccurate perception. To overcome this challenge, we propose ESCA, a framework that contextualizes embodied agents by grounding their perception in spatial-temporal scene graphs. At its core is SGClip, a novel, open-domain, promptable foundation model for generating scene graphs that is based on CLIP. SGClip is trained on 87K+ open-domain videos using a neurosymbolic pipeline that aligns automatically generated captions with scene graphs produced by the model itself, eliminating the need for human-labeled annotations. We demonstrate that SGClip excels in both prompt-based inference and task-specific fine-tuning, achieving state-of-the-art results on scene graph generation and action localization benchmarks. ESCA with SGClip improves perception for embodied agents based on both open-source and commercial MLLMs, achieving state of-the-art performance across two embodied environments. Notably, ESCA significantly reduces agent perception errors and enables open-source models to surpass proprietary baselines. We release the source code for SGCLIP model training at https://github.com/video-fm/LASER and for the embodied agent at https://github.com/video-fm/ESCA.

Section: Introduction
Recent advances in large-scale pretraining have enabled foundation models to assist with a wide range of tasks, from language and vision understanding [2,15,52,83] to mathematical problem solving [96,18] and code generation [21,61,55]. However, it remains an open challenge to realize embodied agents that are capable of doing household chores, training alongside humans in physical activities, or providing care for the aging [39,13]. A critical first step toward this goal is equipping agents with fine-grained perception to ground abstract goals in physical interactions.
Embodied Agent Environment I want to use a machine to prepare toasted bread for breakfast, plea… Instruction Feedback … ✔ Move succeeded > no collision detected Move right by 0.25 units. Action ESCA (ours) Task Planning Visual Description Language Plan Executable Plan target: toaster; objects: bread(326, 155), cabine… 1. Move forward by 0.25… move_right(0.25); … below on a b o v e be hi nd counter bread cabi window Scene Graph SGClip EB-Habitat EB-Navigat Concept Extraction 🔥 InternVL entities:
[toast, bread] attributes:
[yellow] relations: [on]
this section cite: ['b1', 'b14', 'b51', 'b82', 'b95', 'b17', 'b20', 'b60', 'b54', 'b38', 'b12']

Section: Entity Identification
Reflect & Reason the toaster is in front of…
this section cite: []

Section: Transfer Protocol Transfer Protocol

this section cite: []

Section: EB-Alfred

this section cite: []

Section: EB-Manipul
Figure 1: An overview of embodied agent pipeline augmented with ESCA. In each cycle, the agent takes in an instruction and the environmental feedback and outputs a concrete executable action, through a sequence of perception, reasoning, and planning. The action is then executed and the environment will provide the next state. Notably, ESCA contextualizes the task planner with grounded visual features represented as a scene graph.
Despite progress, multi-modal large language models (MLLMs) still struggle to build spatially and temporally grounded world models. Their inability to reliably ground visual features with spatialtemporal relations creates a disconnect between conceptual semantics and pixel-level observations [89,60]. This lack of structured, fine-grained scene understanding severely limits their effectiveness in embodied environments. In fact, our empirical analysis shows that up to 69% of agent failures stem from perception errors, highlighting the need for frameworks that can bridge this gap.
To address this challenge, we propose integrating structured scene graphs into the perception, reasoning, and planning pipelines of MLLM-based embodied agents. While prior work has explored enhancing MLLMs with external visual grounding modules, these approaches typically rely on open-domain object detection models such as Grounding DINO [56] and YOLO [27]. However, these models are primarily designed for object identification and often overlook semantic attributes, inter-object relationships, and temporal consistency.
In this work, we introduce ESCA (Embodied and Scene-Graph Contextualized Agent), a framework designed to contextualize MLLMs through open-domain scene graph generation (Figure 1). Much like the bioluminescent lure of a deep-sea anglerfish, which illuminates its surroundings to reveal otherwise hidden prey, ESCA provides structured visual grounding that helps MLLMs make sense of complex and ambiguous sensory environments. A key feature of ESCA is selective grounding: rather than injecting full scene graphs, which may degrade performance, the MLLM first identifies the subset of objects, attributes, and relations most pertinent to the instruction, then determines the essential entities for task completion. This mechanism is supported by our transfer protocol, which performs probabilistic reasoning over object names, attributes, and spatial relations to construct prompts enriched with the most relevant scene elements. At its core is SGClip, a CLIP-based model that captures semantic visual features, including entity classes, physical attributes, actions, interactions, and inter-object relations.
Through experiments on four challenging embodied environments, we demonstrate that ESCA consistently improves the performance of all evaluated MLLMs, including both open-source and proprietary models. By providing structured and grounded scene graphs, ESCA significantly reduces perception errors, laying the foundation for more reliable reasoning and planning. Beyond its integration with MLLM-based agents, we show that SGClip, when evaluated independently, exhibits strong zero-shot generalization, is promptable for task-specific scene understanding, and remains fine-tunable for downstream tasks such as action recognition.
In summary, our contributions are as follows: (1) we present ESCA, a general framework for contextualizing MLLM-based embodied agents through selective scene graph generation; (2) we introduce the transfer protocol for enriching prompts with probabilistically inferred scene-specific information for diverse embodied benchmarks; (3) we introduce SGClip, a generalizable and finegrained scene graph generation model, along with ESCA-Video-87K, an MLLM annotated dataset;
Concept Extraction Prompt SGClip b e l o w on a b o v e counter bread cabi 🔥 InternVL
this section cite: ['b88', 'b59', 'b55', 'b26']

Section: MLLM

this section cite: []

Section: Concept Extraction Prompt
Obj. Ident. STSG Pred.
this section cite: []

Section: Concept Extraction Prompt

this section cite: []

Section: 🔥 InternVL

this section cite: []

Section: MLLM

this section cite: []

Section: Scene Graph Augmented

this section cite: []

Section: Summarization Prompt
You are a robot operating in a home. You are given the environment for performing the task: "I want to use machine…" Please extract the most visually identifiable features and ignore the more subtle ones. Please deduce the object names with a synonym. For example, substitute "freshly baked baguette" to "loaves"… You are supposed to output in JSON.
You need to describe current visual state, summarize interactions, environment feedback, and the scene, and reason why the last action or plan failed… The current scene objects are:
this section cite: []

Section: Visual Description
Candidate toaster loc: -2 frames before: none -prev frame: [0, 75, …] -curr frame: [320, 32…] Description: I am gettin closer to the toaster but I need to also pick… Figure 2: A detailed illustration of the visual description module, which involves concept extraction, object identification, scene graph prediction, and visual summarization. We also illustrate sample MLLM prompts used in a kitchen environment for the concept extraction and summarization steps.
(4) we conduct extensive evaluations demonstrating the effectiveness and versatility of ESCA and SGClip across both embodied agent and scene understanding tasks.
this section cite: []

Section: ESCA: A Framework for Embodied Agents

this section cite: []

Section: Background
Embodied Environments. Recent research on embodied agents has been accelerated by the availability of simulated environments such as VisualAgentBench [57] and EmbodiedBench [91], which provide rich, multimodal task suites covering navigation, manipulation, and interaction across diverse scenarios. These benchmarks pose challenges that require agents to operate in both a) low-level action spaces such as continuous control signals over robot joints, and b) high-level action spaces such as programmatic instructions and skills that abstract over low-level actions.
These tasks are typically modeled as Partially Observable Markov Decision Processes (POMDPs), represented as 7-tuple (S, A, Ω, T , O, L ins , R), where S is the unobservable state space, A is the task-specific action space, Ω is the visual perception space where I t ∈ Ω is an image frame at time t, T : S × A → S is the state transition dynamics, O : S → Ω relates the underlying state to the observations, L ins is the natural language instruction for the agent, and R : S → {0, 1} is the reward function indicating whether the task has been completed.
this section cite: ['b56', 'b90']

Section: Embodied Agents and MLLM-Based Embodied Agents.
A typical embodied agent interacts with the environment by maintaining a history of observations and actions: h t = (I 0 , a 0 , I 1 , . . . , a t-1 , I t ), where a t is the action taken by the agent at time t. The agent selects the next action by conditioning on the instruction L and history h t via a policy π(a t | L ins , h t ). An MLLM-based embodied agent realizes such a policy by leveraging a multi-modal model that processes both: a) imagery data I t , and b) textual data, including the instruction L ins and textual representations of actions a ∈ A.
Established by [91], recent MLLM-based agent architectures often decompose the policy evaluation process into structured stages inspired by cognitive reasoning workflows (Figure 1): 1) Visual Description: extracting and summarizing visual inputs; 2) Reflection: integrating observations with historical context to build situational awareness; 3) Reasoning: inferring task-relevant insights based on the combined context; 4) Language Plan: generating high-level plans or action sequences in natural language; 5) Executable Plan: translating plans into concrete actions executable by the agent.
this section cite: ['b90']

Section: Challenges of MLLM-Based Embodied Agents.
Despite recent progress, MLLM-based embodied agents remain fragile in complex environments due to compounded errors across perception, reasoning, and planning [39,13]. Perception errors include hallucinated objects, misrecognized entities or actions, and incorrect spatial relationships. Reasoning errors arise when agents fail to correctly infer spatial relations or recognize task termination states. These issues propagate into planning, where agents may skip critical steps or generate invalid plans due to inaccurate state estimation.
this section cite: ['b38', 'b12']

Section: The ESCA Framework
At the core of ESCA is a simple yet effective idea: contextualizing MLLM-based agents with grounded, structured scene-graph information to improve visual description. Specifically, given a language instruction L ins and interaction history h t , the agent generates a multi-modal message sequence of images and text. While existing approaches rely on end-to-end MLLMs to implicitly perform this step, ESCA decomposes the process into four modular stages below (Figure 2).
this section cite: []

Section: Selective Concept Extraction.
This module extracts concepts with MLLM guided by carefully designed prompts based on the instruction L ins and the history h t . Rather than producing only free-form natural descriptions, ESCA requires the MLLM to explicitly extract structured concepts that are most relevant to the query, which can be classified as follows. a) entity classes such as (car, knife, person, cup); b) attributes including physical properties (red, small, broken) and semantic states (close-by, far, moving, sitting); c) relations covering spatial relations (behind, above) and interactions (cutting, cooking).
Concept extraction is guided by two signals: 1) the instruction L ins , which highlights target entities, attributes, or relations the agent should focus on, and 2) the visual feedback I t , which provides spatial context such as environmental structures (e.g., barriers, pathways) and dynamic interactions (e.g., objects being manipulated). Overall, this MLLM-generated structured output provides a targeted concepts set c = {c 1 , c 2 , . . . } for subsequent object identification and scene graph generation.
this section cite: []

Section: Object Identification.
Given the extracted concepts and the agent's visual feedback, the second module grounds these concepts to specific image segments σ = {σ 1 , σ 2 , . . . } each represented by a bit-mask. This step isolates visual elements in the frame that correspond to classified entities or attributes, enabling downstream semantic inference over localized visual units rather than raw pixels.
The object identification module is implemented using a multi-stage pipeline. First, Grounding DINO (GD) [56], a vision-language detection model, takes the extracted concepts and the visual input to predict bounding boxes for the mentioned entities. These bounding boxes are further refined into precise segmentation masks using SAM2 [67], a segmentation model that produces pixel-accurate regions. Leveraging the GD + SAM2 pipeline improves computational efficiency and offers better generalization to both attribute and relational concepts, as further detailed in the Appendix.
this section cite: ['b55', 'b66']

Section: Scene Graph Prediction.
We then construct a scene graph (SG) [38,87,34] by grounding the extracted concepts into structured visual elements. Specifically, the SG contains two types of probabilistic facts: 1) unary facts of the form p :: c i (σ j ), each describing an entity σ j with their classes, attributes, or states represented as c i ; 2) relational facts of the form p :: c i (σ j , σ k ), each representing that a relation or an interaction c i between a pair of grounded entities σ j and σ k . Notably, each predicted fact is associated with a confidence score denoted as p. This probabilistic formulation allows the SG to capture uncertainty and distributions over possible scene interpretations.
Implementation-wise, we build SGClip, a CLIP-based model [65] fine-tuned for open-domain scene graph prediction. The design of SGClip is guided by three aforementioned key desiderata. First, it supports open-domain concept coverage, enabling it to generalize beyond fixed taxonomies. Second, it is adaptable to extracting diverse types of information, including entity classes, attributes, and inter-object relations. Third, it produces probabilistic predictions, allowing the model to capture uncertainties. We provide a more detailed discussion of SGClip and its training in Section 3.
this section cite: ['b37', 'b86', 'b33', 'b64']

Section: Visual Summarization and Validation.
This module distills these multi-modal signals into a list of messages to contextualize the agent's reasoner and planner. Concretely, the summarizer takes a prompt and is responsible for transforming the structured scene graph into natural descriptions, while also validating the consistency between the visual feedback and the underlying structured scene graph. While the summary is customizable via our transfer protocol, the set of messages include 1) the current view (and potentially historic ones) augmented with visualized bounding boxes served as markers, 2) image segments corresponding to key entities, 3) the textual description of scene graph and segments, and 4) an analysis of history actions and how they caused the current scene.
this section cite: []

Section: Transfer Protocol
To enable ESCA to generalize across different downstream tasks, we define a general transfer protocol based on the customization of two prompt templates, positioned at the entry and exit points of the entire visual description module. The goal of this unified transfer protocol is to maximize adaptability of ESCA across tasks with diverse planning strategies, action spaces, and reasoning requirements, while maintaining a consistent interface.
this section cite: []

Section: Concept Extraction Prompt

this section cite: []

Section: Concept Extraction Prompt
🔥 InternVL MLLM Scene Graph Augmented Summarization Prompt SGClip SGClip SGClip SGClip SGClip SGClip bread cabinet toaster ... softmax … brown big brown big not brown not big softmax softmax … … above cutting behind <norel> softmax softmax softmax … (a) Entity classes Concept Extraction Prompt SGClip b e l o w on a b o v e counter bread cabi 🔥 InternVL I want to use a machin… Instruction Visual Feedback … Visual Description Candidate toaster loc: -2 frames before: none -prev frame: [0, 75, …] -curr frame: [320, 32…] Description: I am gettin closer to the toaster… MLLM Concept Extraction Prompt Obj. Ident. STSG Pred.
this section cite: []

Section: Concept Extraction Prompt
🔥 InternVL MLLM Scene Graph Augmented Summarization Prompt SGClip SGClip brown big brown big not brown not big softmax softmax … … (b) Attributes Concept Extraction Prompt SGClip b e l o w on a b o v e counter bread cabi 🔥 InternVL I want to use a machin… Instruction Visual Feedback … MLLM Concept Extraction Prompt Obj. Ident. STSG Pred. Specifically, the first prompt, the Concept Extraction Prompt, specifies the required JSON output format and indicates, or enumerates if possible, task-specific concepts that the agent should focus on.
this section cite: []

Section: Concept Extraction Prompt
The second prompt, the Visual Summarization Prompt, guides the model to produce a contextualized summary that integrates both grounded image segments and task-specific textual elements, such as target objects, desired states, and environmental constraints. Together, these prompts provide a principled way to adapt ESCA to diverse embodied AI tasks without retraining the core system. We present a case study of the transfer protocol in Figure 2 and provide further details in Section 4.
this section cite: []

Section: SGClip Model
To enable the generation of spatial-temporal scene graphs in an open-domain setting, especially the embodied environments, we develop SGClip, a CLIP-based foundation model [65] for structured scene understanding. SGClip is designed to operate in an open-domain fashion, recognizing a wide and extensible set of concepts. Secondly, it must adapt to different types of concepts, while be capable of generalizing to unseen visual and textual domains. Finally, it must produce probabilistic predictions to capture uncertainty.
To meet these goals, SGClip builds on CLIP's vision-language architecture, which naturally supports joint reasoning over images and textual phrases. However, deploying CLIP directly is insufficient, as it lacks specialization for structured scene graph prediction. To bridge this gap, we fine-tune SGClip to balance adaptability and generalizability. In this section, we describe 1) how to handle different types of concepts through inference time adaptation (Section 3.1), 2) how to overcome the lack of data by collecting a model-driven self-supervision dataset (Section 3.2), and 3) how to learn without relying on human annotations via a self-supervised neurosymbolic learning pipeline (Section 3.3).
this section cite: ['b64']

Section: Model Architecture and Inference Time Adaptation
At its core, SGClip (Scene Graph CLIP) is a single CLIP-based model designed to score concept relevance within an image. Formally, it operates as SGClip(σ, c) ∈ R |c| , where σ is the input image and c is the a set of candidate concepts, producing a logit score for each concept that reflects the model's confidence in its presence. SGClip supports three distinct inference modes to handle different types of concepts: entity classes cclass , attributes cattr , and binary relations crela . Illustrated in Figure 3, each mode requires a specialized formulation of the input concepts and scoring process, effectively allowing SGClip to operate flexibly across these concept types:
Entity classes. In this setting, SGClip is used to identify the most likely entity class presented in an image segment. Let cclass denote the list of candidate entity classes. Since an entity is typically assumed to belong to a single class, we apply softmax normalization over the logit scores produced by SGClip for these candidates: softmax(SGClip(σ, cclass )).
Attributes. To estimate the likelihood that a specific segment σ possesses a particular attribute c, we construct a binary contrast between the attribute and its negation by evaluating softmax(SGClip(σ, {c, ¬c})), where ¬c denotes the negated textual phrase (e.g., "not red" for the attribute "red"). The first element of the resulting probability distribution corresponds to the model's estimated likelihood. To improve computational efficiency, we perform batched evaluation by merging all attribute-contradiction pairs into a single concept set c * attr = cattr ∪ {¬c | c ∈ cattr }.
this section cite: []

Section: Textual Input
Visual Input
this section cite: []

Section: SGClip

this section cite: []

Section: LLaVA-Video-178K
The ego car is driving forward, but stopped before an intersection since the traffic light was red; there is a bus passing during the red light.
this section cite: []

Section: Synthetic Captions
∃e,i,l,b. ego(e) ⋀ intersect(i) ⋀ bus(b) ⋀ light(l) ⋀ (far(i,e) ⋀ ¬on(b, i)) ⋀ (red(l) ⋀ close(i,e) ⋀ on(b, i)) Prog. Specifications Object Traces entities: [bus, car, …]; attr: [red]; state: [forwa… Extracted Concepts SGClip LASER Scene Graph Generation Spatial-Temporal Alignment Check w/ Semantic & Contrastive Loss Weaksupervision Label Semantic Parsing Concept Extraction Object Tracking & Segmentation Weaklysupervised Learning Video Captioning … ♢ ESCA-Video-87K
this section cite: []

Section: Binary relations.
For binary relation prediction, the goal is to determine whether a relation c holds between two segments σ i and σ j . To do this, we first compute a bounding region σ * ij that tightly encloses both segments. Within this region, we apply distinct color tinting to σ i and σ j to indicate their directional roles (subject and object). To provide additional relational context, especially for interactions like "cutting," we augment the relation phrase by including the predicted entity classes of the subject and object, generating "(robot, cutting, cabbage)". Relation predictions are thus conditioned on the classes of both the subject and the object. Specifically, for each segment σ i , we compute its most likely class ν i by selecting the top prediction from the entity class:
ν i = c class u , where u = argmax u∈1...|cclass| SGClip(σ i , cclass ) u .
We then form the augmented relation phrase as (ν i , c, ν j ). Similar to attribute prediction, we contrast the candidate relation with a special token <norel> denoting "no relation," and compute softmax(SGClip(σ * ij , {(ν i , c, ν j ), <norel>})), to obtain the probability of whether the relation c holds between object i and j. In practice, this process is batched over all segment pairs and relation concepts to maximize efficiency: c * rela = {(ν i , c, ν j ) | i, j ∈ 1 . . . |σ|, c ∈ crela } ∪ {<norel>}.
this section cite: []

Section: ESCA-Video-87K Dataset
We adopt the neurosymbolic weak-supervision pipeline introduced in LASER [34], which enables learning fine-grained STSGs from weak supervision signals derived from spatial-temporal programmatic specifications, eliminating the need for costly manual annotations. While the details of this learning pipeline are provided in Section 3.3, we begin by introducing the ESCA-Video-87K dataset, the dataset we curate and use to train SGClip.
The ESCA-Video-87K dataset is constructed from the publicly available LLaVA-Video-178K dataset [97], and consists of 87K short video clips, each paired with natural language captions generated by GPT-4 [35]. As illustrated in Figure 4, these captions are first processed to extract relevant concepts which are then fed into GD [56] and SAM2 [67] to obtain object traces, which are sequences of object segmentations that evolve across multiple video frames. In addition, these concepts are also used to assist in generating spatial-temporal programmatic specifications, expressed in a linear temporal logic-based language. To construct these specifications, we develop a semantic parsing pipeline, again leveraging GPT-4, which converts high-level captions into structured temporal statements. These specifications formally describe how the semantics of object traces evolve, capturing temporal relations using operators such as "until", "finally", or "always".
In summary, each data point in ESCA-Video-87K is represented as a 5-tuple ( Ī, L cap , Σ, c, ϕ), where Ī = {I 1 , I 2 , . . . } is the video, L cap is the associated natural language caption, Σ = {σ 1 , σ2 , . . . } is the set of object traces, c = {c 1 , c 2 , . . . } is the set of extracted concepts, and ϕ is the spatialtemporal programmatic specification. This rich, multi-level annotation enables training models like SGClip without requiring manual scene graph labeling. We defer additional details about the dataset construction process and data statistics to the Appendix.
this section cite: ['b33', 'b96', 'b34', 'b55', 'b66']

Section: Neurosymbolic Learning Pipeline
Given the ESCA-Video-87K, our goal is to fine-tune the SGClip model using the provided object traces, concepts, and spatial-temporal programmatic specifications. This is achieved by aligning the scene graphs generated by SGClip with the expected specifications [34], where the degree of alignment serves as the learning signal (Figure 4). To perform this alignment in a differentiable manner, we leverage the Scallop programming language [50], enabling symbolic alignment checks to be integrated into end-to-end gradient-based learning.
Specifically, the alignment loss computation mirrors the inference-time adaptation procedure described in Section 3.1, where different types of concepts are processed differently but unified under the same model. The pipeline is further enhanced with semantic losses, derived from evaluating common-sense and temporal constraint satisfaction, as well as a contrastive loss that encourages the model to distinguish between matched and unmatched scene graph-specification pairs. Additional details of the training process are provided in the Appendix.
this section cite: ['b33', 'b49']

Section: Embodied Environments and Transfer Protocol Setup
We evaluate our approach on EmbodiedBench [45], a benchmark suite designed to assess MLLMbased embodied agents. We focus on two environments: EB-Navigation and EB-Manipulation, each of which requires different levels of perception, reasoning, and control, and may benefit from a contextualized visual description. To adapt ESCA to these tasks, we apply our transfer protocol by designing two specialized prompts for each environment. Full prompt templates are provided in the Appendix; here, we summarize the core challenges and how ESCA addresses them.
EB-Navigation is built on AI2-THOR [42] and focuses on visual navigation tasks where the agent must locate target objects based on language instructions, such as "navigate to the laptop." The agent relies solely on egocentric visual input and textual feedback, navigating through a space using eight low-level movement and rotation actions. Existing models often fail by generating correct high-level plans but producing incorrect low-level actions due to poor spatial grounding from an egocentric perspective. ESCA addresses this by generating accurate scene graphs that capture spatial relations and object positions. With the scene graphs, we design prompts that incorporate numerical bounding box data and temporal movement cues, helping the agent to localize targets more precisely.
EB-Manipulation extends VLMBench [100] for evaluating low-level robotic manipulation. The agent controls a 7-DoF robotic arm using discretized action spaces, with additional signals such as YOLO bounding boxes [27] and global 3D object pose estimates to assist manipulation. Existing models struggle to ground object concepts into actionable spatial representations, leading to perception failures that disrupt downstream planning and control. ESCA improves this by grounding target features into precise visual segments, enabling prompts that describe object attributes, semantic relations, and 3D spatial coordinates, giving the agent more reliable geometric context.
EB-Habitat builds upon Language Rearrangement task [78], simulated via Habitat 2.0 [77], and primarily evaluates high-level task decomposition and planning capabilities. The action space is limited to atomic high-level actions, such as navigate/pick/place/open/close, from which the agent is instructed to complete tasks such as "Find a toy airplane and move it to the right counter". Common failure cases of existing models include invalid actions arising from failure to identify and remember object displacements in the scene. Our model improves this by managing scene graphs of the current and desired state of the target object, which improves awareness of task progression and limits the number of focus objects.
EB-Alfred is based on the ALFRED dataset [72] and AI2-THOR [42]. It evaluates agents on highlevel household tasks involving eight skill types like "pick up" or "turn off." The agent receives egocentric observations and textual feedback on action validity, performing actions on objects. While agents receive egocentric observations and action feedback, existing models tend to repeat the same mistakes because they fail to reflect on how past actions influence the current state. ESCA addresses With ESCA and SGClip, all models consistently outperform the baselines.
this by generating scene graphs that describe both the current and target states symbolically, enabling prompts that support causal reasoning. This allows the agent to recognize how previous actions led to the current situation and to deduce the necessary state changes to achieve the task goal.
this section cite: ['b44', 'b41', 'b99', 'b26', 'b77', 'b76', 'b71', 'b41']

Section: Empirical Evaluation
Our experiments are designed to address two key research questions: (1) How effectively does ESCA, together with SGClip, improve embodied agent performance through structured scene graph generation? and (2) How generalizable and adaptable is SGClip when evaluated independently on open-domain, zero-shot, and downstream transfer tasks? We now detail our experimental setup and present empirical results addressing both questions.
this section cite: []

Section: Experimental Setup.
We evaluate ESCA in EB-Navigation, EB-Manipulation, EB-Habitat, and EB-Alfred, four environments that demand fine-grained perception to support both low-level and high-level control. To assess the general applicability of ESCA, we integrate it with four diverse MLLMs: InternVL-2.5-38B-MPO [14] , Qwen2.5-VL-72B-Ins [3], Gemini-2.0-flash [62], and GPT-4o [35]. For each MLLM experiment, we use the model for both the concept extraction and visual summarization steps (Figure 2). To further benchmark ESCA's impact, we compare to performance of MLLMs augmented with existing visual grounding modules, including Grounding DINO [56] and Ultralytics-YOLO11 [27].
For evaluating SGClip independently, we consider out-of-domain scene graph benchmarks, including OpenPVSG [90], Action Genome [36], and VidVRD [70], comparing SGClip against strong baselines such as CLIP [65], InternVL-6B [15], BIKE [85], and Text4Vis [84]. To assess SGClip's downstream adaptability beyond structured scene graph prediction, we further test the fine-tunability on the ActivityNet action recognition dataset by applying a transfer protocol.
this section cite: ['b13', 'b2', 'b61', 'b34', 'b55', 'b26', 'b89', 'b35', 'b69', 'b64', 'b14', 'b84', 'b83']

Section: Decisive Progression

this section cite: []

Section: Reasoning & Plan
The kettle is located in the center of the image, slightly towards the back, on the countertop near the stove. My plan is to move rightward to bypass the desk.
this section cite: []

Section: Reasoning & Plan
The image shows a kitchen with a refrigerator on the… The kettle is not visible in the current view. I have rotated to the right once, but the kettle is still not visible.
this section cite: []

Section: 🔥 InternVL

this section cite: []

Section: Exploration
Behavioral Degeneration Ha WR SR RE IA Co Perc. (30%) Reas. (26%) Plan. (44%) InternVL + ESCA Ha WR SR RE IA Co Perc. (69%) Reas. (11%) Plan. (20%) InternVL Figure 7: Error decomposition 1 of InternVL with or without ESCA, manually inspected on 60 EB-Navigation tasks.
ESCA for Embodied Agents. As shown in Figure 5, ESCA-augmented MLLMs consistently outperform their non-contextualized baselines across both EB-Navigation and EB-Manipulation. Remarkably, on EB-Navigation, even the open-source InternVL-2.5, when augmented with ESCA, surpasses the base performance of the proprietary GPT-4o model. While integrating Grounding DINO or YOLO improves baseline models, ESCA provides additional, substantial gains. For example, Gemini-2.0 with ESCA achieves over 10% improvement, while GPT-4o, already boosted by YOLO, still benefits from an additional 6% performance gain on EB-Manipulation.
Through qualitative analysis of end-to-end agent behaviors, we observe that ESCA consistently improves the agent's perceptual grounding, leading to more effective task execution. As illustrated in Figure 6, an agent powered by InternVL with ESCA successfully identifies the kettle early in the episode and navigates directly toward it. In contrast, the base InternVL model fails to recognize the target and ultimately collapses onto the wall. This observation is further supported by the error decomposition analysis shown in Figure 7, where we find that ESCA reduces the overall perception error rate from 69% to 30%. We provide additional detailed experimental results in the Appendix.
this section cite: []

Section: Generalizability and Adaptability of SGClip.
Evaluating SGClip's zero-shot generalization, Figure 9 shows that SGClip trained on ESCA-Video-87K consistently outperforms CLIP on OpenPVSG, Action Genome, and VidVRD, demonstrating strong out-of-domain robustness. Further, SGClip shows strong adaptability, achieving notable improvements when fine-tuned on VidVRD (details provided in the Appendix). Beyond scene graph tasks, Figure 10 highlights SGClip's downstream transferability to action recognition on ActivityNet. Fine-tuned with only 1% of the training data, SGClip outperforms state-of-the-art zero-shot video recognition baselines. With 5% of the data (approximately 800 videos), SGClip achieves 92.10% accuracy, approaching the performance of InternVideo2-6B with end-to-end finetuning on the ActivityNet dataset.
6 Related Works Scene Graph in planning. Scene graphs [38,87,31,33] are symbolic representations that encode the semantic structure of an image or video by identifying objects as nodes and their relationships as edges [58,43]. They play a central role in a variety of vision-related tasks, including visual question answering [44,28,64], image captioning [92,102], and image generation [26,46]. More recently, scene graphs have been increasingly adopted in the domain of robotic planning, for enhanced robustness [30,81], or verifiable planning [37,66,16]. To enable seamless integration with multimodal large language model (MLLM) agents, ESCA constructs scene graphs from 2D image inputs and dynamically updates them through embodied interaction with the environment.
this section cite: ['b37', 'b86', 'b30', 'b32', 'b57', 'b42', 'b43', 'b27', 'b63', 'b91', 'b101', 'b25', 'b45', 'b29', 'b80', 'b36', 'b65', 'b15']

Section: Embodied Agents.
Embodied agents [22,99] are autonomous systems that perceive, reason, and interact within physical or simulated environments. To evaluate their capabilities, varies benchmarks [45,86] have been proposed, ranging from vision-language navigation [68,4,17] and object manipulation [19,53] to interactive instruction following and long-horizon planning [72,73,20]. 0% 1% 5% 70 80 90 100 BIKE Text4vis InternVL-6B 76.34 80.1 92.1 % Data used for fine-tuning Accuracy (%) SGClip CLIP Figure 10: Down-stream finetunability on action recognition, evaluated on ActivityNet dataset. We also illustrate zero-shot baselines (BIKE and Text4vis) as well as a fully-supervised baseline (InternVL-6B).
Recent advances in large language models (LLMs) [8,35,79,80] and multimodal large language models (MLLMs) [2,15,52,83] are driving progress toward general-purpose embodied agents [1,5,9,32,94,59]. Furthermore, recent MLLMs have been trained end-to-end to directly generate low-level numerical control commands [10,7]. ESCA introduces a general framework for augmenting vision-driven, MLLM-based agents with structured scene graph information.
this section cite: ['b21', 'b98', 'b44', 'b85', 'b67', 'b3', 'b16', 'b18', 'b52', 'b71', 'b72', 'b19', 'b7', 'b34', 'b78', 'b79', 'b1', 'b14', 'b51', 'b82', 'b0', 'b4', 'b8', 'b31', 'b93', 'b58', 'b9', 'b6']

Section: Neurosymbolic Methods with LLM and MLLM.
A growing trend for enhancing the reasoning capabilities and robustness of large language models (LLMs) is to incorporate structured representations and leverage symbolic algorithms to reason over them [40,23,93,48,47,51,49,76,6]. These efforts span diverse domains, including code generation [21,61,55], mathematical problem solving [96,18], and verifiable planning [75,54,12,95]. Recent work has extended this paradigm to the low level control domain, training MLLMs with structured scene graphs in an end-to-end manner [7,71,63,101]. ESCA follows this neurosymbolic direction by introducing SGClip, which is trained in MLLM-augmented self-supervised manner enabled by neuro-symbolic methodology, using structured scene graphs as an intermediate representation to guide learning.
this section cite: ['b39', 'b22', 'b92', 'b47', 'b46', 'b50', 'b48', 'b75', 'b5', 'b20', 'b60', 'b54', 'b95', 'b17', 'b74', 'b53', 'b11', 'b94', 'b6', 'b70', 'b62', 'b100']

Section: Conclusion and Limitations
We introduced ESCA, a framework for contextualizing embodied agents through scene graph generation, powered by SGClip, a promptable, open-domain scene graph model. Through a general transfer protocol, ESCA adapts to diverse tasks and consistently improves agent performance across multiple environments and MLLMs. Beyond embodied tasks, SGClip demonstrates strong generalization and adaptability on open-domain scene graph and action recognition benchmarks.
Limitations. Despite its strong performance, our framework has several limitations. First, the use of large language models for high-level planning introduces latency, making it unsuitable for real-time low-level control. Second, the system relies on 2D visual inputs and lacks support for 3D representations like point clouds, limiting depth-aware reasoning and spatial precision. Finally, while ESCA leverages MLLMs to generate coherent plans, it lacks formal mechanisms for verifying intermediate and final states during execution.
8 Checklist
this section cite: []

Section: References
Ref_id:b0 Title: Akash Srivastava, and Pulkit Agrawal. Compositional foundation models for hierarchical planning Year: (2023)
Ref_id:b1 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b2 Title:  Year: (2025)
Ref_id:b3 Title: Objectnav revisited: On evaluation of embodied agents navigating to objects Year: (2020)
Ref_id:b4 Title: Debidatta Dwibedi, and Dorsa Sadigh. Rt-h: Action hierarchies using language Year: (2024)
Ref_id:b5 Title: Lobster: A gpu-accelerated framework for neurosymbolic programming Year: (2025)
Ref_id:b6 Title: A vision-language-action flow model for general robot control Year: (2024)
Ref_id:b7 Title: On the opportunities and risks of foundation models Year: (2021)
Ref_id:b8 Title: Rt-2: Vision-language-action models transfer web knowledge to robotic control Year: (2023)
Ref_id:b9 Title: Rt-1: Robotics transformer for realworld control at scale Year: (2022)
Ref_id:b10 Title: Activitynet: A large-scale video benchmark for human activity understanding Year: (2015)
Ref_id:b11 Title: A framework for neurosymbolic robot action planning using large language models Year: (2024)
Ref_id:b12 Title: Can we rely on llm agents to draft long-horizon plans? Year: (2024)
Ref_id:b13 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2025)
Ref_id:b14 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b15 Title: Optimal scene graph planning with large language model guidance Year: (2024)
Ref_id:b16 Title: Robothor: An open simulation-toreal embodied ai platform Year: (2020)
Ref_id:b17 Title: Metacognitive capabilities of llms: An exploration in mathematical problem solving Year: (2024)
Ref_id:b18 Title: Manipulathor: A framework for visual object manipulation Year: (2021)
Ref_id:b19 Title: Building open-ended embodied agents with internet-scale knowledge Year: (2022)
Ref_id:b20 Title: The robots are coming: Exploring the implications of openai codex on introductory programming Year: (2022)
Ref_id:b21 Title: Autonomous agents as embodied ai Year: (1997)
Ref_id:b22 Title: Pal: Program-aided language models Year: (2023)
Ref_id:b23 Title: The" something something" video database for learning and evaluating visual common sense Year: (2017)
Ref_id:b24 Title: Ego4d: Around the world in 3,000 hours of egocentric video Year: (2022)
Ref_id:b25 Title: Learning canonical representations for scene graph to image generation Year: (2020)
Ref_id:b26 Title: Yolov8 to yolo11: A comprehensive architecture in-depth comparative review Year: (2025)
Ref_id:b27 Title: Scene graph reasoning for visual question answering Year: (2020)
Ref_id:b28 Title: spacy: Industrial-strength natural language processing in python Year: (2020)
Ref_id:b29 Title: What's left? concept grounding with logicenhanced foundation models Year: (2023)
Ref_id:b30 Title: Language model as planner and formalizer under constraints Year: (2025)
Ref_id:b31 Title: On the limit of language models as planning formalizers Year: (2025)
Ref_id:b32 Title: Scallop: From probabilistic deductive databases to scalable differentiable reasoning Year: (2021)
Ref_id:b33 Title: LASER: A neuro-symbolic framework for learning spatio-temporal scene graphs with weak supervision Year: (2025)
Ref_id:b34 Title: Gpt-4o system card Year: (2024)
Ref_id:b35 Title: Action genome: Actions as compositions of spatio-temporal scene graphs Year: (2020)
Ref_id:b36 Title: Sequential manipulation planning on scene graph Year: (2022)
Ref_id:b37 Title: Image retrieval using scene graphs Year: (2015)
Ref_id:b38 Title: Llms can't plan, but can help planning in llm-modulo frameworks Year: (2024)
Ref_id:b39 Title: Beliefbank: Adding memory to a pre-trained language model for a systematic notion of belief Year: (2021)
Ref_id:b40 Title: The kinetics human action video dataset Year: (2017)
Ref_id:b41 Title: AI2-THOR: an interactive 3d environment for visual AI Year: (2017)
Ref_id:b42 Title: Visual genome: Connecting language and vision using crowdsourced dense image annotations Year: (2017)
Ref_id:b43 Title: Visual question answering over scene graph Year: (2019)
Ref_id:b44 Title: Embodied agent interface: Benchmarking llms for embodied decision making Year: (2024)
Ref_id:b45 Title: Pastegan: A semi-parametric method to generate image from scene graph Year: (2019)
Ref_id:b46 Title: Iris: Llm-assisted static analysis for detecting security vulnerabilities Year: (2025)
Ref_id:b47 Title: Relational programming with foundational models Year: (2024)
Ref_id:b48 Title: Relational programming with foundational models Year: (2024-03)
Ref_id:b49 Title: Scallop: A language for neurosymbolic programming Year: (2023)
Ref_id:b50 Title: Scallop: A language for neurosymbolic programming Year: (2023-06)
Ref_id:b51 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b52 Title: Softgym: Benchmarking deep reinforcement learning for deformable object manipulation Year: (2021)
Ref_id:b53 Title: Llm+p: Empowering large language models with optimal planning proficiency Year: (2023)
Ref_id:b54 Title: Exploring and evaluating hallucinations in llm-powered code generation Year: (2024)
Ref_id:b55 Title: Grounding dino: Marrying dino with grounded pre-training for open-set object detection Year: (2024)
Ref_id:b56 Title:  Year: (2024)
Ref_id:b57 Title: Visual relationship detection with language priors Year: (2016)
Ref_id:b58 Title: Eureka: Human-level reward design via coding large language models Year: (2024)
Ref_id:b59 Title: Visual agentic ai for spatial reasoning with a dynamic api Year: (2025)
Ref_id:b60 Title: Linc: A neurosymbolic approach for logical reasoning by combining language models with first-order logic provers Year: (2023)
Ref_id:b61 Title: Introducing gemini 2.0: Our new ai model for the agentic era Year: (2024-12)
Ref_id:b62 Title: Task-oriented hierarchical object decomposition for visuomotor control Year: (2024)
Ref_id:b63 Title: Scene graph refinement network for visual question answering Year: (2022)
Ref_id:b64 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b65 Title: Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning Year: (2023)
Ref_id:b66 Title:  Year: (2024)
Ref_id:b67 Title: Habitat: A platform for embodied ai research Year: (2019)
Ref_id:b68 Title: Annotating objects and relations in user-generated videos Year: (2019)
Ref_id:b69 Title: Video visual relation detection Year: (2017)
Ref_id:b70 Title: Composing pre-trained objectcentric representations for robotics from" what" and" where" foundation models Year: (2024)
Ref_id:b71 Title: Alfred: A benchmark for interpreting grounded instructions for everyday tasks Year: (2020)
Ref_id:b72 Title: Alfworld: Aligning text and embodied environments for interactive learning Year: (2020)
Ref_id:b73 Title: Hollywood in homes: Crowdsourcing data collection for activity understanding Year: (2016)
Ref_id:b74 Title: Pddl planning with pretrained large language models Year: (2022)
Ref_id:b75 Title: Mayur Naik, and Eric Wong. Data-efficient learning with neural programs Year: (2024)
Ref_id:b76 Title: Habitat 2.0: Training home assistants to rearrange their habitat Year: (2021)
Ref_id:b77 Title: Large language models as generalizable policies for embodied tasks Year: (2024)
Ref_id:b78 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b79 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b80 Title: Programmatically grounded, compositionally generalizable robotic manipulation Year: (2023)
Ref_id:b81 Title: Internvid: A large-scale video-text dataset for multimodal understanding and generation Year: (2023)
Ref_id:b82 Title: Visual chatgpt: Talking, drawing and editing with visual foundation models Year: (2023)
Ref_id:b83 Title: Revisiting classifier: Transferring vision-language models for video recognition Year: (2023)
Ref_id:b84 Title: Bidirectional cross-modal knowledge exploration for video recognition with pre-trained vision-language models Year: (2023)
Ref_id:b85 Title: Gibson env: Real-world perception for embodied agents Year: (2018)
Ref_id:b86 Title: Scene graph generation by iterative message passing Year: (2017)
Ref_id:b87 Title: Advancing high-resolution video-language representation with large-scale video transcriptions Year: (2022)
Ref_id:b88 Title: Thinking in space: How multimodal large language models see, remember, and recall spaces Year: (2024)
Ref_id:b89 Title: Panoptic video scene graph generation Year: (2023)
Ref_id:b90 Title: Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents Year: (2025)
Ref_id:b91 Title: Auto-encoding scene graphs for image captioning Year: (2019)
Ref_id:b92 Title: Improved logical reasoning of language models via differentiable symbolic programming Year: (2023)
Ref_id:b93 Title: Building cooperative embodied agents modularly with large language models Year: (2023)
Ref_id:b94 Title: Pddlego: Iterative planning in textual environments Year: (2024)
Ref_id:b95 Title: Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? Year: (2024)
Ref_id:b96 Title: Video instruction tuning with synthetic data Year: (2024)
Ref_id:b97 Title: Llava-video: Video instruction tuning with synthetic data Year: (2025)
Ref_id:b98 Title: See and think: Embodied agent in virtual environment Year: (2024)
Ref_id:b99 Title: Vlmbench: A compositional benchmark for vision-and-language manipulation Year: (2022)
Ref_id:b100 Title: Neurostrata: Harnessing neurosymbolic paradigms for improved design, testability, and verifiability of autonomous cps Year: (2025)
Ref_id:b101 Title: Comprehensive image captioning via scene graph decomposition Year: (2020)
Ref_id:b102 Title: Procnets: Learning to segment procedures in untrimmed and unconstrained videos Year: (2017)
Ref_id:b103 Title: Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment Year: (2024)
