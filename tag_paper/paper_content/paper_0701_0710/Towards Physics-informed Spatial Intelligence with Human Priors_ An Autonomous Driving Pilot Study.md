Title: Towards Physics-informed Spatial Intelligence with Human Priors: An Autonomous Driving Pilot Study
Abstract: How to integrate and verify spatial intelligence in foundation models remains an open challenge. Current practice often proxies Visual-Spatial Intelligence (VSI) with purely textual prompts and VQA-style scoring, which obscures geometry, invites linguistic shortcuts, and weakens attribution to genuinely spatial skills. We introduce Spatial Intelligence Grid (SIG): a structured, grid-based schema that explicitly encodes object layouts, inter-object relations, and physically grounded priors. As a complementary channel to text, SIG provides a faithful, compositional representation of scene structure for foundation-model reasoning. Building on SIG, we derive SIG-informed evaluation metrics that quantify a model's intrinsic VSI, which separates spatial capability from language priors. In few-shot in-context learning with state-of-the-art multimodal LLMs (e.g. GPT-and Gemini-family models), SIG yields consistently larger, more stable, and more comprehensive gains across all VSI metrics compared to VQA-only representations, indicating its promise as a data-labeling and training schema for learning VSI. We also release SIGBench, a benchmark of 1.4K driving frames annotated with ground-truth SIG labels and human gaze traces, supporting both grid-based machine VSI tasks and attention-driven, human-like VSI tasks in autonomous-driving scenarios.

Section: Introduction
Currently, Visual Question Answering (VQA) is recognized as a natural test of visual-spatial intelligence (VSI), as it requires interpreting images, understanding object relationships, and inferring context to produce correct answers using text [1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17][18][19]. In computer vision, most researchers agree that VSI involves accurately perceiving, manipulating, and reasoning about visual and spatial information, such as size, location, and correlations given an input image, using textual description [20]. In a stateof-the-art study, three VQA questions appear on the first page that illustrate their VSI understanding: "What is the distance between the keyboard and the TV? How many cabinets are in the room? And how height the stool is ?" [13] And correct answers indicate effective VSI learning by the model.
However, are VQA tasks ideal for evaluating spatial intelligence? For human beings, it is defined by Dr. Gardner and Dr. Lohman as "the ability to generate, retain, retrieve, and transform wellstructured visual images" [21]. In computer vision, compared to other well-explored vision problems, VSI's core challenge lies in extracting 3D geometric insights from 2D images or videos and then presenting them textually [13,22]. VQA inherently interweaves linguistic proficiency with spatial reasoning. However, even without visual input, humans can form a mental map of their surroundings by relying on auditory cues, like the way sound reverberates in a space [23]. Some researchers argue that VQA's reliance on textual replies may not fully capture the underlying spatial intelligence [24][25][26][27].
They advocate integrating additional physical priors or alternative more spatial-like representations to reflect spatial reasoning, like how human beings think.
Figure 1: Examples of Human VSI in Painting. Abraham Bosse, a French artist and theorist illustrates a systematic, grid-based method for achieving visual-spatial correlations in rendering three-dimensional space on a two-dimensional canvas (left) [28], and he employed this grid-based visual scheme in portrait painting (middle) [29,30]. Procedures of drawing a cast with graphical priors from scratch (right). [31] Historical art practices offer an inspiring perspective: a grid-canvas is very helpful for humans to perceive the spatial priors. During the Renaissance, artists like Albrecht Dürer, Leonardo da Vinci and Vincent van Gogh famously used grid-based "drawing machines," as often depicted in historical artworks [32][33][34][35][36]. By imposing visual observations onto structured grids (see Fig. 1), artists could systematically decompose a 3D scene into spatial regions, then record or interpret its geometry and graphical correlations. As illustrated in Fig. 1 middle, portrait painters often proceeded from these organized graphical representations to finely detailed textual descriptions, mirroring how human VSI naturally moves from a structured visual framework to richer linguistic content. Consequently, once trained to perceive spatial relationships accurately for painters, the specific subject matter becomes less significant: any subject, regardless of its nature, is ultimately perceived as a unique combination of graphs (nodes, shapes, edges, and color notes) on these grids. Collectively, these visual graphs define the appearance of objects such as a cast, a flower, or a human head [37].
Can graphical priors on a grid-based canvas be used as machine representations of VSI? Like a painter's trained eye, machines with grid or graph-based abstractions gain a structured view that captures spatial relationships and hierarchies. This simplifies scene decomposition and supports richer interpretation and language grounding. From our perspective, an ideal VSI representation is a combination of a structured conceptual model of a scene that encodes geometric and topological relationships, typically through grid-or graph-formatting priors. In this framework, visual elements (e.g. shapes, edges, etc.) are mapped onto discrete spatial partitions and connected to capture both local and global information. By systematically organizing visual data into these relational structures, such an ideal VSI representation supports robust reasoning, efficient spatial manipulation, and a natural pathway from pure visual analysis to higher-level semantic or linguistic descriptions.
Building upon these insights, we propose a novel VSI representation format, called spatial intelligence grid (SIG), and then conduct foundational testing on existing Multimodal Large Language Models (MLLMs) in one of the most critical VSI application tasks: autonomous driving (AD) [38][39][40][41][42]. AD demands real-time perception, precise modeling of multiple dynamic objects, and a high level of situational awareness in diverse environments. The ability to efficiently attend to elements locations, correlations and movements in the driving scene, such as other vehicles, pedestrians and road signs is essential for safe and effective operation. Given the complexity, flexibility, high stakes, and the potential impacts, AD serves as an ideal stress test for evaluating and refining SIG-based VSI frameworks. To facilitate this, we introduce a novel benchmark, SIGBench, and systematically investigate three fundamental tasks: 1) SIG-based VSI Evaluation of MLLMs and Metrics: Evaluate whether existing MLLMs can answer spatial queries based on SIG representations, and propose novel evaluation metrics for SIG based on graph similarity theory; 2) SIG-Empowered In-Context Learning (ICL) for VQA: Investigate how SIG-informed VSI features enhance spatial intelligence in VQA tasks through few-shot ICL, and 3) Human v.s. Machine VSI Attention based on SIG: Examine how human VSI attention differs from machine-based approaches within SIG, and explore methods to achieve more human-like SIG in order to improve human-machine interactions. In this research, our contribution can be summarized as follows:
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b12', 'b20', 'b12', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41']

Section: Textualization Abstraction

this section cite: []

Section: Human-like Spatial Intelligence Grid

this section cite: []

Section: Physical Constraints

this section cite: []

Section: Spatial Intelligence Grid

this section cite: []

Section: Vehicle Traffic Sign Traffic Light

this section cite: []

Section: Visual Grounding

this section cite: []

Section: Ego Vehicle

this section cite: []

Section: Cognitive Priors

this section cite: []

Section: Human-like Attention Map

this section cite: []

Section: Human Driver

this section cite: []

Section: Gaze Attention

this section cite: []

Section: Attended Region Non-salient Region

this section cite: []

Section: What represents Visual-Spatial Intelligence the best?

this section cite: []

Section: Reasoning Adaptation
Human-like Spatial Relation Graph
this section cite: []

Section: Directional Relation:
The self is at the back of the black truck 1.
The black truck 1 is at the back of the light 1.
The light 1 is at the front right of the sign 1.
The sign 1 is at the front of the self. The self is at the back of the light 1.
The sign 1 is at the front left of the black truck 1.
this section cite: []

Section: Human-like Spatial Relation Paragraph
Proximal Relation:
The self is at a distance from the black truck 1.
The black truck 1 is at a distance from the light 1.
The light 1 is close to the sign 1.
The sign 1 is far from the self. The self is far away from the light 1.
The sign 1 is at a distance from the black truck 1.
this section cite: []

Section: Proximal Relation:
The ego vehicle is at a distance from the black truck 1.
The black truck 1 is at a distance from the traffic light. The traffic light is close to the traffic sign. The traffic sign is far from the ego vehicle. The ego vehicle is far away from the traffic light. The traffic sign is at a distance from the black truck 1. 2 Related Work General VSI in MLLMs. MLLMs have demonstrated promising performance across a variety of visual tasks such as object detection, segmentation, and image captioning by virtue of their unified vision-language representations and strong reasoning capabilities [43][44][45][46][47][48][49]. Building on these strengths, recent work has extended MLLMs to tackle visual-spatial reasoning queries such as which object is at the left-most position in this image, through architectural adaptations and fine-tuning strategies designed to emphasize spatial relations [19,[50][51][52][53][54][55][56]. To provide accurate and in-depth evaluation of a model's general VSI, several comprehensive visual-task benchmarks incorporate dedicated visual-spatial reasoning sections [1][2][3][4][5][6][7][8], while others focus exclusively on assessing visualspatial reasoning ability over images [9][10][11][12][16][17][18] or videos [13][14][15] in VQA-style. Although these benchmarks yield clear right-wrong scores, they often conflate failures of visual grounding (e.g. object detection errors) with genuine visual-spatial reasoning mistakes, particularly in scenes containing many similar entities (e.g. multiple vehicles with different brands, colors and shapes).
this section cite: ['b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b18', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b15', 'b16', 'b17', 'b12', 'b13', 'b14']

Section: VSI with Scenario-based Physical Constraints.
In real-world applications such as robotics [62][63][64], AR/VR [65][66][67] and AD [68][69][70], the implementation of VSI must respect domain-specific physical, environmental and regulatory constraints. For AD in particular, models must not only interpret scene geometry but also adhere to traffic rules and kinematic feasibility across perception [71][72][73][74][75][76], planning [68,[77][78][79][80], and control modules [81][82][83][84][85]. A suite of VQA-style benchmarks has emerged to evaluate visual-spatial reasoning under these constraints [58-61, 86, 87], leveraging large-scale open-source AD datasets [88][89][90][91][92], shown in Tab. 1. However, by framing questions solely as textimage queries, these benchmarks may overlook the rich spatial structure exposed by bird's-eye-view (BEV) representation that are widely used in AD perception and planning pipelines. Human-Like VSI. Human scene understanding is guided by both overt gaze patterns (where we look) and covert cognitive maps (how we represent spatial relations). Gaze or saliency prediction models, which estimate pixel-level attention maps, have provided deep insights into visual prioritization in generic images [93][94][95][96][97] and driving scenarios [98][99][100][101]. Complementing this, the notion of a cognitive map captures the internal spatial layout that humans use to reason about object relations [102] and has recently been integrated into MLLM frameworks for video understanding [13]. In our approach, we leverage an human-like SIG to emulate both gaze-driven saliency and structured spatial representations, thereby enabling a human-like evaluation of visual-spatial reasoning, particularly within complex AD environments.
this section cite: ['b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b75', 'b67', 'b76', 'b77', 'b78', 'b79', 'b80', 'b81', 'b82', 'b83', 'b84', 'b87', 'b88', 'b89', 'b90', 'b91', 'b92', 'b93', 'b94', 'b95', 'b96', 'b97', 'b98', 'b99', 'b100', 'b101', 'b12']

Section: Methods

this section cite: []

Section: Grid-based Visual-Spatial Intelligence
We introduce SIG, a grid-based representation format for VSI, as illustrating in Fig. 2. In AD scenarios, roadways will be a physical constraints and the primary entities we focus on are the ego-vehicle and other traffic nodes, scuh as vehicles, traffic signs, signal lights and traffic lanes.
Based on SIG, we can extract a directed spatial relation graph (SRG) that describes the spatial relation (direction+distance in grid) of each object and spatial relation paragraph (SRP) that describes spatial relation of each object within a text manner. To quantitatively assess a model's VSI, we propose three novel evaluation metrics: multi-level spatial matching (MLSM), spatial relation graph similarity (SRGS) and semantic relational distance (SRD). MLSM compares object positions directly within the SIG representation, capturing absolute localization accuracy. SRGS measures both node-wise and edge-wise correspondence between predicted and ground-truth (GT) SRG, emphasizing relation classification and structure. SRD computes a semantic relational distance between predicted and GT prepositions in SRP, evaluating the fidelity of both directional and proximal relations. To isolate core spatial reasoning, our evaluation focuses exclusively on ego-vehicle (self), other vehicles, traffic lights, and traffic signs, omitting traffic lanes from the quantitative metrics because many scenes have unreliable lane cues such as faded or blurred markings and no markings at intersection or rural roads (we provide evaluation metrics including traffic lanes for an additional subset with clear traffic lanes). Before evaluating predicted SIG using GT SIG, we will firstly align the position of self in predicted SIG to which in GT SIG. Next, we apply the same offset for all other objects in predicted SIG.
this section cite: []

Section: Multi-level Spatial Matching.
To evaluate SIG-based spatial representations independently of visual-grounding noise, we introduce a multi-threshold graph matching protocol inspired by tracking metrics such as MOTA [103] and HOTA [104]. MLSM first performs bipartite matching between predicted and GT objects (vehicles, traffic signs, traffic lights) using a cost function c v for vehicles and c sl for traffic signs and traffic lights, which can be calculated by
c v = d * ω c * ω o * ω t , c sl = d * ω o (1
) 𝛂 Level 1 Level 2 Level 3 TP1 FP1 FN1 TP2 FP2 FN2 TP3 FP3 FN3 0 0 1 1 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 1 2 1 0 0 1 0 0 0 1 1 3 1 0 0 1 0 0 0 1 1 Multi-
Level Spatial Matching Example: evaluating matched truck in GT-Pred SIGs def level_1(gt, pred): # requires same type (e.g., bus) return type(gt) == type(pred) def level_2(gt, pred): # requires same type and order return level_1(gt, pred) and order(gt) == order(pred) def level_3(gt, pred): # requires same type, order, color return level_2(gt, pred) and color(gt) == color(pred) Multi-Level Requirements Pseudocode Multi-Level Spatial Matching Pseudocode pairs = bipartite_match() # gt-pred pairs for alpha in [1, ..., n]: # distance threshold for gt, pred in pairs: if distance(gt, pred) <= alpha: # subscripts indicate matching levels if level_1(gt, pred): TP_1 += 1 if level_2(gt, pred): TP_2 += 1 if level_3(gt, pred): TP_3 += 1 else: FP_1 += 1, FP_2 += 1, FP_3 += 1 FN_1 += 1, FN_2 += 1, FN_3 += 1 GT : black truck 1 Pred : blue truck 1 color type order Type and order matched, satisfying level 1 and 2 Spatial distance within threshold 𝛼, level 1 and 2 are satisfied, TP count as one. Spatial distance within threshold 𝛼, But level 3 is not satisfied, TP count as zero. Spatial Relation Graph Similarity Example: evaluating GT and Pred SRGs GT-Pred edge pairs differ, edge substitution cost applied GT-Pred node pairs differ, node substitution cost applied GT-Pred node pairs fully aligned, no cost applied GT node missing in Pred, node insertion cost applied GT edge missing in Pred, edge insertion cost applied Spatial Relation Graph Similarity Pseudocode node_pairs = bipartite_match() # gt-pred node pairs edge_pairs = edge_pairs(node_pairs) # gt-pred edge pairs # add node sub. cost for unequal gt-pred nodes for node_gt, node_pred in node_pairs: if not node_gt == node_pred: cost += c_node_sub # add edge sub. cost for unequal gt-pred edges for edge_gt, edge_pred in edge_pairs: if not edge_gt == edge_pred: cost += c_edge_sub # add insertion cost for unmatched gt nodes for node_gt not in node_pairs[gt]: cost += c_node_ins cost += c_edge_ins * N_gt # number of gt nodes # add deletion cost for unmatched pred nodes for node_pred not in node_pairs[pred]: cost += c_node_del cost += c_edge_del * N_pred # number of pred nodes self GT SRG (boxed area from GT SIG) front right (1.7) back left (1.7) sign 1 black truck 1 light 1 Pred SRG (boxed area from Pred SIG) front right (2.2) back left (2.2) sign 1 self blue truck 1 light 1 black truck 1 light 1 sign 1 GT Pred Distance black truck 1 blue truck 1 2 sign 1 missing +∞ light 1 light 1 10 Object Bipartite Matching GT SIG black truck 1 light 1 sign 1 Pred SIG blue truck 1 light 1 bipartite matched MLLM where ω c , ω o , ω t means the weight for color, order and type matching, respectively. Here d is the euclidean distance between objects' position on SIG, and each weight ω c , ω o , ω t equals one when the attribute is unmatched and drops below one when it matches, thus granting larger spatial tolerances for objects that are matched in type, order, and/or color.
After matching, true positives (TP) are object pairs whose grid position lie within a distance threshold α and object attributes are aligned. False negatives (FN) are GT objects with no predicted match; false positives (FP) are predicted objects with no GT match. We evaluate three hierarchical matching levels for vehicles: (1) same type, (2) same type + order, (3) same type + order + color, and one level for signs and lights (same order), respectively. Over a set of n thresholds α ∈ [1, . . . , n], we compute precision P α , recall R α , F1-score F1 α and association accuracy AssA α and normalize them to get overall P, R, F1 and AssA, as showing in Fig. 3. Detailed calculation are shown in Appendix A.1.
Let n denote the total number of objects (n is used similarly in the following complexity analysis). For MLSM, constructing the cost matrix for vehicles, traffic signs, and traffic lights has complexity O(n 2 ), bipartite matching requires O(n 3 ) [105], and multi-threshold matching takes O(n). Thus, the overall complexity of MLSM is
T MLSM (n) = O(n 2 ) + O(n 3 ) + O(n) ⇒ O(n 3 ).
Spatial Relation Graph Similarity. Different from MLSM, SRGS evaluates the correspondence of individual relations (edges) between a predicted SRG and its GT counterpart. We quantify this through the computation of the graph edit distance (GED) [106] which measures the number of operations needed to edit the predicted SRG to GT. Let's denote directed SRG as
G = (V, E). V = {v i } n i=1
denotes the set of all n nodes, where v i represents an instance (e.g. ego-vehicle, other vehicles, traffic signs, traffic lights). E ⊆ V × V denotes set of edges, where directed edge e ij = (v i , v j ) ∈ E encodes spatial relation (direction + distance in SIG) from v i to v j .
We further calculate the SRGS from two perspectives: node edit distance and edge edit distance. The computation is based on bipartite matching between nodes in the GT and predicted SRG. Let M be the set of all matched pairs (v i , vi ′ ), where v i ∈ V is a GT node and vi ′ ∈ V is its corresponding predicted node, as determined by the bipartite matching algorithm. The node edit distance considers three types of costs: 1) Substitution cost δ sub (v i , vi ′ ): the cost of modifying a predicted node vi ′ to match a GT node v i with different position or attributes; 2) Deletion cost δ del (v i ′ ): the cost of removing an unmatched predicted node vi ′ ; 3) Insertion cost δ ins (v j ): the cost of adding an unmatched GT node v j . The total node edit distance between the GT graph G (with node set V) and the predicted graph Ĝ (with node set V) can then be denoted as
D N (G, Ĝ) = (vi,v i ′ )∈M δ sub (v i , vi ′ ) + vi∈V δ del (v i ) + vj ∈ V δ ins (v j )(2)
Detailed computations of each cost function are provided in Appendix A.2. Similarly, we define the edge edit distance D E (G, Ĝ) with the edge substitution cost δ E sub (e i , êi ′ ), edge deletion cost δ E del (ê i ′ ), and edge insertion cost δ E ins (e j ). Combining D N and D E , we can calculate weighted total graph edit distance D total and weighted graph similarity score S by
D total = γ D N (G, Ĝ) + β D E (G, Ĝ), S = max 0, 1 - D total D max ∈ [0, 1] (3
)
where γ, β are weights and D max denotes worst-case distance (all nodes and edges unmatched).
For SRGS, constructing the fully connected graph requires O(n) for all nodes and O(n 2 ) for all edges. Bipartite matching again requires O(n 3 ), and GED, involving node-to-node and edge-toedge comparisons, takes O(n) + O(n 2 ). Therefore, the overall complexity of SRGS is T SRGS = 2(O(n) + O(n 2 )) + O(n 3 ) ⇒ O(n 3 ).
this section cite: ['b102', 'b103', 'b104', 'b105']

Section: 5
Figure 4: Directional Relation Circle. The semantic relational distance between any two prepositions is the smallest step count around the circle (e.g. between "at the back left of" and "at the front of" is 3 instead of 5).
this section cite: []

Section: Semantic Relational Distance.
To quantify the fidelity of directional and proximal preposition predictions in SRP, we assign each preposition a position on a discrete scale and measure the minimal cyclic or linear separation from GT. Directional relations are arranged cyclically, illustrated as Fig. 4. Proximal relations are ordered linearly from closest to furthest: adjacent to, close to, at a distance from, far from, far away from. Here the semantic relational distance between "adjacent to" and "far from" is 3. We compute MAE and MSE based on semantic relational distance and accuracy. Detailed equations are shown in Appendix A.3. For SRD, we compute pointwise distances. With 8 directional and 5 proximal relations, the time complexity is T SRD = O(8n) + O(5n) ⇒ O(n).
this section cite: []

Section: SIG-based In-context Learning for VSI
While existing methods focus on improving MLLM's VSI based on depth, detection bounding box (bbox) and segmentation mask and convert them into textual representation such as QA for learning [50][51][52], we investigate SIG's potential as direct representation for improving VSI using ICL. Let's denote a dataset D = {(x i , y i )} N i=1 containing N image-SIG pairs. The ICL process for outputting answer y q of query x q can be formulated as
y q = F M (x q ; P)(4)
where F M is the MLLM and example prompt P = concat(x 1 , y 1 , . . . , x k , y k ). In our settings, we randomly select k image-SIG pairs where x i contains image with annotated bboxes of vehicles and a task description prompt, and y i contains the GT SIG (content of a JSON file, e.g. vehicles:{black truck 1:[5,3]}; traffic_signs:{sign 1:[3,5]}...) and SRP derived from GT SIG. Our main insight is to let the MLLM learn the corresponding spatial relation between object position in image and SIG and we demonstrate SIG's strong generalization ability with experiments in Sec. 4.3.
this section cite: ['b49', 'b50', 'b51']

Section: Human-Like VSI with Grid
In AD scenario, human decision-making is guided by selective attention: drivers focus on a subset of objects rather than all of them in the scene. We incorporate this human-like bias by integrating gaze or saliency predictions [93,98] into SIG. Let p i = (u i , v i , 1) ⊤ denote an image pixel and w i = (X i , Y i , 1) ⊤ denote corresponding grid cell on SIG. We can calculate the the homographic matrix H = [h ij ] ∈ R 3×3 by solving the equations of X i and Y i as (a) Integration (b)
X i = h 11 u i + h 12 v i + h 13 h 31 u i + h 32 v i + h 33 , Y i = h 21 u i + h 22 v i + h 23 h 31 u i + h 32 v i + h 33 (5
)
this section cite: ['b92', 'b97']

Section: Spatial Intelligence Grid Creation
Given the driving scene image with bounding boxes on certain vehicles, please identify vehicles, traffic signs, traffic lights, and traffic lanes by estimating their center positions on a 10×10 BEV grid, and output as a JSON dictionary… ```json\n{ "vehicles":{"black car 1": [5,5]}, "traffic_signs":{"sign 1": [6,8], "sign 2": [8,8]} "traffic_lights": {"light 1": [3,8]}, "traffic_lanes": {" GT Human-like SIG using singular value decomposition (SVD). We then project and normalize the raw attention map A Image with image size into the SIG attention map A SIG using
A SIG (i, j) = A Image ⌊x ij ⌋, ⌊y ij ⌋ -min u,v A Image (u, v) max u,v A Image (u, v) -min u,v A Image (u, v) , s.t.[x ij , y ij , 1] ⊤ = H -1 [i, j, 1] ⊤ (6)
Thus, we can combine A SIG and SIG to create SIG with attention weight (a.k.a. human-like SIG).
this section cite: ['b5', 'b7']

Section: Human-Like Spatial Relation Graph Similarity.
To weight graph edit distance by human-like perceptual importance, we scale each node-edit cost by its corresponding attention weight from A SIG . Edge-edit costs are similarly weighted by the average attention weight of the two incident nodes. This yields an attention-aware SRGS that penalizes errors on highly attended objects more severely.
this section cite: []

Section: Human-Like Semantic Relational Distance.
To apply human gaze attention as a weight for the calculation of human-like SRD, we multiply the SRD between predicted and GT prepositions by the mean attention of the two referenced objects. For instance, if the GT relation is black car 1 is {at the back left of} white car 2, the predicted relation is {at the front of}, and the attention weights for black car 1 and white car 2 are 3 and 4. The semantic directional distance in this example is 3 (shown in Fig. 4) and the human-like SRD for this example can be computed as 3 × 3+4 2 = 10.5.
this section cite: []

Section: Experiment

this section cite: []

Section: SIG Benchmark
Benchmark Overview. We introduce SIGBench, a benchmark for quantifying both grid-based and human-like VSI in MLLMs within AD scenario. SIGBench comprises 1,423 frames, each annotated with (i) SIG and human-like SIG, (ii) SRP and human-like SRP and (iii) gaze attention map in image size. The annotation pipeline is shown in Fig. 5 (a). Let f denotes the camera focal length. We create attention map by firstly estimating a circular attention radius using
r = min(r w , r h ) s.t. r w = f • I w s w • tan fov w 2 , r h = f • I h s h • tan fov h 2(7)
where I w , I h are image width and height in pixels, s w , s h are the corresponding sensor dimensions in millimeters, and fov w and fov h are the human horizontal and vertical field of view. Then we accumulate attention map from 6 consecutive frames to create GT attention map. For annotation of SIG, we annotate vehicles inside the bounding box using "color+type+order" label (e.g. black car 1), where order runs leftmost to rightmost in the image. We also annotate traffic signs/lights that are clearly visible using "type+order" (e.g. sign 1, light 1). Detailed annotation pipeline and data distribution of SIGBench and SIGBench-tiny regarding to number of objects in each sample can be found in Appendix B. In general, SIGBench contains two main task clusters: grid-based VSI tasks: spatial intelligence grid creation (SIGC) and spatial relation paragraph filling (SRPF) and human-like VSI tasks: human-like SIGC and SRPF, and gaze prediction.
Table 3: Quantitative comparison of 3-shot ICL for general VSI tasks on SIGBench-tiny. Z-S means zero-shot, ICL-MC meaning ICL using multiple-choice VQA and ICL-SIG meaning ICL using SIG. light red indicates the results that is worse than zero-shot after applying ICL on GPT-4o and Gemini-2.5-Pro. Models Type MLSM SRGS SRD (Directional) SRD (Proximal) MC (Acc) P↑ R↑ F1↑ AssA↑ S↑ WS↑ MAE↓ MSE↓ Acc↑ MAE↓ MSE↓ Acc↑ Dir.↑ Prox.↑ GPT-4o Z-S 0.522 0.432 0.462 0.316 0.327 0.321 1.792 4.827 0.186 0.858 1.324 0.346 0.056 0.292 ICL-MC 0.545 0.431 0.468 0.320 0.324 0.323 1.600 4.106 0.218 0.920 1.621 0.365 0.144 0.337 ICL-SIG 0.592 0.438 0.479 0.328 0.337 0.357 1.593 4.094 0.220 0.775 1.271 0.436 0.172 0.309 Gemini-2.5-Pro Z-S 0.464 0.570 0.496 0.345 0.224 0.210 1.151 2.426 0.295 0.721 1.100 0.439 0.247 0.413 ICL-MC 0.477 0.617 0.524 0.366 0.185 0.187 1.174 2.667 0.325 0.845 1.387 0.384 0.172 0.348 ICL-SIG 0.556 0.608 0.565 0.406 0.305 0.307 1.126 2.396 0.316 0.578 0.729 0.493 0.305 0.447 Human-Like VSI with Grid Tasks. For gaze prediction task, it evaluates an MLLM's ability to predict the human gaze attention map for frame i based on attention map from frames i -5 to i -1 as human gaze always follow a spatial-temporal format. We want to measure how well the model anticipates where a driver would pay attention to given attention maps from previous frames. For human-like SIGC and SRPF tasks, they are similar as grid-based VSI SIGC and SRPF, but incorporate human gaze attention into the evaluation. Each object is assigned an attention weight from SIG attention map A i SIG for frame i, reflecting its relative importance to a human observer. These attention-weighted tasks are used to reveal whether an MLLM can prioritize spatial relations between objects according to their importance to current scene in a human-like manner.
this section cite: []

Section: Model Selection
Models. We evaluate several top-tier MLLMs on SIGBench mainly from five modal families: 1) open-source models such as InternVL [49] and Qwen-VL [44]; 2) Proprietary models including OpenAI GPT [43], Google Gemini [45] and Anthropic Claude [48]. The specific models in each model family with their complete names we used during experiment are listed in detail in Tab. 2.
Evaluation Metrics. For gaze prediction tasks, we follow the widely used metrics in gaze/saliency prediction task such as Person's Correlation Coefficient (PCC), KL-Divergence and Information Gain (IG) [93,98]. For grid-based SIGC and SRPF, we use MLSM, SRGS and SRD mentioned in Sec. 3.1. For human-like SIGC and SRPF, we use human-like SRGS and SRD mentioned in Sec. 3.3.
this section cite: ['b48', 'b43', 'b42', 'b44', 'b47', 'b92', 'b97']

Section: Results and Analysis for Grid-based VSI on SIGBench
Zero-shot Inference on SIGBench, showing in Tab. 2. The performance of several MLLMs in grid-based SIGC and SRPF using zero-shot inference are shown in Tab. 2. In general, Gemini-2.5-Pro achieves the best performance in both MLSM and SRD (Directional), illustrating its strong ability in spatial understanding of object position and direction. GPT-4o achieves the best in SRGS and the second best in MLSM and SRD, revealing its strong capability in figuring spatial-relation between objects. Besides, Claude-3.7-Sonnet demonstrate strong capability in understanding proximal distance between objects in text and second best in SRGS. By analyzing failure cases, we observe that small or peripheral objects are often missed or mislocalized and substantial overlap (high IoU) between objects exacerbates this by causing identity conflation and incorrect grid-cell placement.
this section cite: []

Section: SIG-based ICL with Random Sample Selection, showing in Tab. 3.
To evaluate SIG's advantage over conventional VQA-style representation for VSI, we conduct ICL on GPT-4o and Gemini-2.5-Pro, whose performance are outstanding among all models in Tab. 2. We randomly select 90 samples from SIGBench as SIGBench-tiny and generate 4-8 multiple-choice questions targeting directional and proximal relations, mimicking existing VQA benchmarks for VSI for each image. In addition, we randomly selected 3 images (outside SIGBench-tiny) and annotated full VQA pairs covering every object in their ground-truth SIGs for training. we conduct a 3-shot ICL using SIG (ICL-SIG) and VQA (ICL-MC) annotations as the input representation, respectively and evaluate on SIGBench-tiny using our proposed SIGC and SRPF metrics alongside the accuracy of annotated VQA tasks (MC).
The key findings are: 1) the model's VSI consistently improves with ICL-SIG. Across both models, ICL-SIG improves nearly every metric relative to zero-shot baselines; 2) Even using randomly sampled images as context examples without sophisticated sampling strategy, ICL-SIG generally improves VSI over ICL-MC, especially for MLSM, SRGS and SRD (Proximal); 3) Compared to ICL-SIG, the improvements brought by ICL-MC is unstable, which result in lower performance than zero-shot in metrics such as SRGS-S in both models. To demonstrate the potential variability of SIGbased ICL with difference choice of data samples, we conduct further ablation studies in Appendix B.4. These findings highlights SIG's superior fidelity in encoding VSI compared to traditional VQA and its potential as a new representation schema for improving VSI in MLLMs.
this section cite: []

Section: Empirical Runtime Analysis of Evaluation Metrics
Runtime efficiency is crucial for real-time applications like AD. We report the empirical per-frame runtimes of our proposed evaluation metrics on SIGBench under varying object numbers, as summarized in Tab. 4.
this section cite: []

Section: Cross-Domain Generalizability.
Although our initial motivation arises from AD, SIG as a data representation is fundamentally domain-agnostic and can be applied wherever a fixed ontology of object types exists. For demonstration, we construct two proof-of-concept benchmarks based on subsets from MS COCO [107] and ARKitScenes [108], which we denote as SIG-COCO and SIG-ARKitScenes, respectively. We conduct both zero-shot inference and ICL experiments with GPT-4o on these benchmarks, with results reported in Tab. 5. ICL-SIG consistently outperforms zero-shot inference across both benchmarks, suggesting SIG generalizes effectively beyond AD domain.
this section cite: ['b106', 'b107']

Section: Results and Analysis for Human-Like VSI with Grid on SIGBench
The results for gaze prediction, human-like SIGC and SRPF are shown in Tab. 6. Surprisingly, InternVL2.5-26B and Qwen-VL-2.5-7B achieves the best performance in gaze prediction task. After looking at the their output, we found that different from other models that apply operations such as gaussian blur, edge detection after taking the average of five previous attention map, these two models only take the average (details in Appendix C.3). Model rankings under human-SRGS and human-SRD closely mirror the grid-based VSI results (Tab. 2), indicating that incorporating attention weights doesn't alter relative performance between MLLMs. This shows that current MLLMs still struggle to prioritize scene objects with human-like selectivity in AD settings.
this section cite: []

Section: Conclusion and Discussion
We propose a novel representation for visual-spatial intelligence (VSI) called spatial intelligence grid (SIG) and introduce a suite of graph-based evaluation metrics that leverage its structured topology to enable more precise and general VSI assessment. Through experiments on different MLLMs, we demonstrate that SIG-based few-shot in-context learning consistently delivers larger, more stable and more comprehensive VSI enhancement than using traditional VQA-style prompts solely, underscoring SIG's superior capacity to encode complex spatial relations. Based on SIG, we create SIGBench, a benchmark with image-SIG/human gaze attention pairs, which is designed to evaluate both grid-based machine VSI and human-like attention-driven spatial reasoning under our proposed metrics. Taken together, these contributions offer a principled data schema and a practical yardstick for VSI. Despite these advances, our study has two limitations remaining for future works: (i) SIGBench currently targets single-frame settings and therefore does not assess tracking or dynamic object-object relations that require temporal context; and (ii) while SIG proves effective for in-context learning, SIG-based fine-tuning and reinforcement learning with human feedback remain unexplored.
this section cite: []

Section: References
Ref_id:b0 Title: Seed-bench: Benchmarking multimodal large language models Year: (2024-06)
Ref_id:b1 Title: Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension Year: (2024)
Ref_id:b2 Title: Mmbench: Is your multi-modal model an all-around player? Year: (2024)
Ref_id:b3 Title: Seed-bench-2: Benchmarking multimodal large language models Year: (2023)
Ref_id:b4 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2024)
Ref_id:b5 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b6 Title: Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models Year: (2025)
Ref_id:b7 Title: Eureka: Evaluating and understanding large foundation models Year: (2024)
Ref_id:b8 Title: Visual spatial reasoning Year: (2023)
Ref_id:b9 Title: Mind the gap: Benchmarking spatial reasoning in vision-language models Year: (2025)
Ref_id:b10 Title: Is a picture worth a thousand words? delving into spatial reasoning for vision language models Year: (2024)
Ref_id:b11 Title: Seeing from another perspective: Evaluating multi-view understanding in mllms Year: (2025)
Ref_id:b12 Title: Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces Year: (2024)
Ref_id:b13 Title: Urbanvideo-bench: Benchmarking vision-language models on embodied intelligence with video data in urban spaces Year: (2025)
Ref_id:b14 Title: Sti-bench: Are mllms ready for precise spatial-temporal world understanding? Year: (2025)
Ref_id:b15 Title: Does spatial cognition emerge in frontier models Year: (2025)
Ref_id:b16 Title: Unfolding spatial cognition: Evaluating multimodal models on visual simulations Year: (2025)
Ref_id:b17 Title: Mmsi-bench: A benchmark for multi-image spatial intelligence Year: (2025)
Ref_id:b18 Title: Multispatialmllm: Multi-frame spatial understanding with multi-modal large language models Year: (2025)
Ref_id:b19 Title: Learning spatial relationships in computer vision Year: (1996)
Ref_id:b20 Title: Human abilities: their nature and measurement Year: (2013)
Ref_id:b21 Title: Geoai: spatially explicit artificial intelligence techniques for geographic knowledge discovery and beyond Year: (2020)
Ref_id:b22 Title: Frames of Mind: The Theory of Multiple Intelligences Year: (2011)
Ref_id:b23 Title: Natural language understanding and inference with mllm in visual question answering: A survey Year: (2025)
Ref_id:b24 Title: Visual question answering: A survey of methods, datasets, evaluation, and challenges Year: (2025)
Ref_id:b25 Title: Learning physical graph representations from visual scenes Year: (2020)
Ref_id:b26 Title: Action genome: Actions as compositions of spatiotemporal scene graphs Year: (2020)
Ref_id:b27 Title: Algemeene manier van de hr. Desargues, tot de practijck der perspectiven, gelijck tot die der meet-kunde, met de kleyne voet-maat, mitsgaders der plaatsen, en proportien van de stercke en flaauwe rakingen, of kleuren. by Dancker Danckertsz.# woonende in de Kalver-straat Year: ()
Ref_id:b28 Title: Screen wise, screen play: Jacques de lajoue and the ruses of rococo Year: (2013)
Ref_id:b29 Title: Alberti's veil Year: ()
Ref_id:b30 Title: Drawing Made Easy Year: (1919)
Ref_id:b31 Title: The life and art of Albrecht Dürer Year: (2023)
Ref_id:b32 Title: The" Invention" of Dürer as a Renaissance Artist Year: (2010)
Ref_id:b33 Title:  Year: (1926)
Ref_id:b34 Title: Art in History, 600 BC-2000 AD: Ideas in Profile Year: (2015)
Ref_id:b35 Title: A new interpretation of the grid system reform in the late period Year: (2021)
Ref_id:b36 Title: The art of romare bearden a resource for teachers Year: (2003)
Ref_id:b37 Title: Motion tracks: A unified representation for human-robot transfer in few-shot imitation learning Year: (2025)
Ref_id:b38 Title: An online trajectory guidance framework via imitation learning and interactive feedback in robot-assisted surgery Year: (2025)
Ref_id:b39 Title: A survey of imitation learning: Algorithms, recent developments, and challenges Year: (2024)
Ref_id:b40 Title: Prompting multi-modal tokens to enhance end-to-end autonomous driving imitation learning with llms Year: (2024)
Ref_id:b41 Title: Hierarchical generative adversarial imitation learning with mid-level input generation for autonomous driving on urban environments Year: (2024)
Ref_id:b42 Title: Gpt-4o system card Year: (2024)
Ref_id:b43 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b44 Title: Gemini: a family of highly capable multimodal models Year: (2023)
Ref_id:b45 Title: The llama 3 herd of models Year: (2024)
Ref_id:b46 Title: Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding Year: (2024)
Ref_id:b47 Title: The claude 3 model family: Opus, sonnet, haiku Year: (2024)
Ref_id:b48 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b49 Title: Spatialbot: Precise spatial understanding with vision language models Year: (2024)
Ref_id:b50 Title: Spatialvlm: Endowing visionlanguage models with spatial reasoning capabilities Year: (2024-06)
Ref_id:b51 Title: Spatialrgpt: Grounded spatial reasoning in vision-language models Year: (2024)
Ref_id:b52 Title: Topv-nav: Unlocking the top-view spatial reasoning potential of mllm for zero-shot object navigation Year: (2025)
Ref_id:b53 Title: Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models Year: (2024-11)
Ref_id:b54 Title: Sparkle: Mastering basic spatial capabilities in vision language models elicits generalization to spatial reasoning Year: (2025)
Ref_id:b55 Title: From text to space: Mapping abstract spatial models in llms during a grid-world navigation task Year: (2025)
Ref_id:b56 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b57 Title: Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario Year: (2023)
Ref_id:b58 Title: Nuscenes-mqa: Integrated evaluation of captions and qa for autonomous driving datasets using markup annotations Year: ()
Ref_id:b59 Title: Nuplanqa: A large-scale dataset and benchmark for multi-view driving scene understanding in multi-modal large language models Year: (2025)
Ref_id:b60 Title: Nuscenes-spatialqa: A spatial understanding and reasoning benchmark for vision-language models in autonomous driving Year: (2025)
Ref_id:b61 Title: Rt-1: Robotics transformer for real-world control at scale Year: (2023)
Ref_id:b62 Title: π0: A vision-language-action flow model for general robot control Year: (2024)
Ref_id:b63 Title: Rt-2: Vision-language-action models transfer web knowledge to robotic control Year: (2023)
Ref_id:b64 Title: Hourvideo: 1-hour video-language understanding Year: (2024)
Ref_id:b65 Title: Ego4d: Around the world in 3,000 hours of egocentric video Year: (2022)
Ref_id:b66 Title: Egoschema: A diagnostic benchmark for very long-form video language understanding Year: (2023)
Ref_id:b67 Title: Drivevlm: The convergence of autonomous driving and large vision-language models Year: ()
Ref_id:b68 Title: Drivegpt4: Interpretable end-to-end autonomous driving via large language model Year: (2023)
Ref_id:b69 Title: Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving Year: ()
Ref_id:b70 Title: Language-guided 3d object detection in point cloud for autonomous driving Year: (2023)
Ref_id:b71 Title: Language prompt for autonomous driving Year: (2023)
Ref_id:b72 Title: Referring multi-object tracking Year: (2023)
Ref_id:b73 Title: Openscene: 3d scene understanding with open vocabularies Year: (2023)
Ref_id:b74 Title: Clip2scene: Towards label-efficient 3d scene understanding by clip Year: (2023)
Ref_id:b75 Title: Unsupervised 3d perception with 2d vision-language distillation for autonomous driving Year: (2023)
Ref_id:b76 Title: Talk to the vehicle: Language conditioned autonomous navigation of self driving cars Year: (2019)
Ref_id:b77 Title: Ground then navigate: Language-guided navigation in dynamic scenes Year: (2023)
Ref_id:b78 Title: Alt-pilot: Autonomous navigation with language augmented topometric maps Year: (2023)
Ref_id:b79 Title: Can you text what is happening? integrating pre-trained language encoders into trajectory prediction models for autonomous driving Year: (2023)
Ref_id:b80 Title: Languagempc: Large language models as decision makers for autonomous driving Year: (2023)
Ref_id:b81 Title: Drive as you speak: Enabling human-like interaction with large language models in autonomous vehicles Year: (2023)
Ref_id:b82 Title: Bevgpt: Generative pretrained large model for autonomous driving prediction, decision-making, and planning Year: ()
Ref_id:b83 Title: Driving with llms: Fusing object-level vector modality for explainable autonomous driving Year: (2023)
Ref_id:b84 Title: Dilu: A knowledgedriven approach to autonomous driving with large language models Year: (2023)
Ref_id:b85 Title: Reason2drive: Towards interpretable and chain-based reasoning for autonomous driving Year: ()
Ref_id:b86 Title: Drivelm: Driving with graph visual question answering Year: (2023)
Ref_id:b87 Title: nuscenes: A multimodal dataset for autonomous driving Year: (2020)
Ref_id:b88 Title: Scalability in perception for autonomous driving: Waymo open dataset Year: (2020)
Ref_id:b89 Title: One million scenes for autonomous driving: Once dataset Year: (2021)
Ref_id:b90 Title: CARLA: An open urban driving simulator Year: (2017-11-15)
Ref_id:b91 Title: Nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles Year: (2022)
Ref_id:b92 Title: What do different evaluation metrics tell us about saliency models? Year: (2016)
Ref_id:b93 Title: A benchmark of computational models of saliency to predict human fixations Year: (2012)
Ref_id:b94 Title: Quantitative analysis of human-model agreement in visual saliency modeling: A comparative study Year: (2013)
Ref_id:b95 Title: Cat2000: A large scale fixation dataset for boosting saliency research Year: (2015)
Ref_id:b96 Title: Appearance-based gaze estimation with deep learning: A review and benchmark Year: (2024)
Ref_id:b97 Title: Predicting the driver's focus of attention: the dr (eye) ve project Year: (2018)
Ref_id:b98 Title: Dada: Driver attention prediction in driving accident scenarios Year: (2022)
Ref_id:b99 Title: Modeling drivers' situational awareness from eye gaze for driving assistance Year: (2025-11)
Ref_id:b100 Title: Predicting driver attention in critical situations Year: (2019)
Ref_id:b101 Title: The cognitive map in humans: spatial navigation and beyond Year: (2017)
Ref_id:b102 Title: Evaluating multiple object tracking performance: the clear mot metrics Year: (2008-01)
Ref_id:b103 Title: Hota: A higher order metric for evaluating multi-object tracking Year: (2021-02)
Ref_id:b104 Title: Introduction to algorithms Year: (2009)
Ref_id:b105 Title: A survey of graph edit distance Year: (2010-02)
Ref_id:b106 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b107 Title: Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data Year: (2021)
Ref_id:b108 Title: Grounded sam: Assembling open-world models for diverse visual tasks Year: (2024)
