Title: Disentangled Concepts Speak Louder Than Words: Explainable Video Action Recognition
Abstract: Effective explanations of video action recognition models should disentangle how movements unfold over time from the surrounding spatial context. However, existing methods-based on saliency-produce entangled explanations, making it unclear whether predictions rely on motion or spatial context. Language-based approaches offer structure but often fail to explain motions due to their tacit nature-intuitively understood but difficult to verbalize. To address these challenges, we propose Disentangled Action aNd Context concept-based Explainable (DANCE) video action recognition, a framework that predicts actions through disentangled concept types: motion dynamics, objects, and scenes. We define motion dynamics concepts as human pose sequences. We employ a large language model to automatically extract object and scene concepts. Built on an ante-hoc concept bottleneck design, DANCE enforces prediction through these concepts. Experiments on four datasets-KTH, Penn Action, HAA500, and UCF-101-demonstrate that DANCE significantly improves explanation clarity with competitive performance. We validate the superior interpretability of DANCE through a user study. Experimental results also show that DANCE is beneficial for model debugging, editing, and failure analysis. Our project page is available at https://jong980812.github.io/DANCE/

Section: Introduction
Recent advances in video action recognition [56,11,28,13,32,53,30,2] have led to impressive performance across diverse benchmarks. To deploy such high-performing video models in real-world applications, explaining their predictions becomes essential for ensuring trust, transparency, and accountability [10,20]. Despite this need, the decision-making processes of video action recognition models remain largely opaque, and systematic approaches to explanation are still underexplored. From a cognitive science perspective, humans interpret complex information more effectively when it is presented in a structured format-i.e., broken down into meaningful and separable components [36,34,35,51]. Interpretability further improves when each component is expressed in a clear and unambiguous manner [10,55,33]. Notably, humans perceive actions by separately analyzing two distinct factors: (i) how movements evolve over time (temporal dynamics) and (ii) what physical context surrounds those movements, such as objects and scenes (spatial context) [17,16,27]. Therefore, to align with human reasoning process, a video explainable AI (video XAI) should explain a model's prediction in a structured way-explicitly disentangling and attributing its decisions to temporal dynamics and spatial context. Meanwhile, existing approaches in video XAI largely follow two strategies: extending image-based feature attribution methods [15,23,42,14,46] across the time axis [50,18,31] or clustering spatiotemporal tubelets to discover high-level concepts [26,21,43]. However, these methods do not
this section cite: ['b55', 'b10', 'b27', 'b12', 'b31', 'b52', 'b29', 'b1', 'b9', 'b19', 'b35', 'b33', 'b34', 'b50', 'b9', 'b54', 'b32', 'b16', 'b15', 'b26', 'b14', 'b22', 'b41', 'b13', 'b45', 'b49', 'b17', 'b30', 'b25', 'b20', 'b42']

Section: 𝑡

this section cite: []

Section: Motion dynamics concept

this section cite: []

Section: Baseball bat
Little league field
this section cite: []

Section: Object concept Scene concept
VTCD [26] 𝑡 … 𝑡 "Jersey and field make sense. But motion? That's tacit."
"Baseball jersey" "Baseball field"
"The person rotates their upper body slightly while standing, then shifts their weight to the left leg in preparation for the swing. They follow through by lifting the bat upward in a curved motion, using their arms and upper body to generate force."
Field? Helmet? Bat?
Motion?
"Aha! Now I get it!"
Figure 1: Disentangled concepts speak louder than words. Spatio-temporal attribution methods provide unstructured explanations that are often ambiguous to human users. Given a Baseball Swing video, (a) visual explanations from 3D-saliency [50] and VTCD [26] fail to clarify whether the prediction is driven by motion (e.g., torso twist), objects (e.g., jersey/helmet), or scene context (e.g., baseball field). (b) Language-based approaches offer more structure but remain ambiguous for motion, as it is tacit knowledge-intuitively understood but hard to verbalize. Verbal descriptions of motion often lack clarity and are difficult to interpret. (c) In contrast, DANCE disentangles motion and context to provide structured explanations. Pose sequences capture motion dynamics in an intuitive, appearance-invariant form, while we clearly convey object and scene concepts via text.
disentangle temporal dynamics from spatial context in their explanations. Instead, they highlight localized regions of input videos in an unstructured and entangled manner-making it difficult to attribute predictions to specific types of evidence. For example, in Figure 1 (a), for the action Baseball Swing, it remains unclear whether the model's prediction is based on the twisting motion of the torso, the appearance of the player (e.g., the jersey), or the scene (e.g., baseball field).
A potential direction for structured explanations is to use language-based approaches [38,60,39,37,1]. Language-based approaches could provide structured and human-readable text descriptions, as illustrated in Figure 1 (b). While these methods can effectively capture spatial context or high-level semantics, they often struggle to express motion dynamics clearly. This challenge arises because motion dynamics often fall under the category of tacit knowledge-knowledge that is intuitively understood and applied, but difficult to verbalize or explain explicitly [41]. As shown in Figure 1 (b), verbally describing the swinging motion of a torso is nontrivial, and even when verbalized, such descriptions tend to be overly verbose and cognitively difficult for users to interpret. To address the challenges of structured and motion-aware explanation, we propose Disentangled Action aNd Context concept-based Explainable (DANCE) video action recognition framework. As illustrated in Figure 1 (c), DANCE provides explanations based on three disentangled concept types: (i) motion dynamics, (ii) object, and (iii) scene. To capture fine-grained temporal patterns, we define motion dynamics concepts as human pose sequences. These pose sequences offer an appearanceagnostic representation of motion, enabling users to intuitively understand how an action unfolds over time without being distracted by irrelevant visual factors such as clothing or background. In parallel, we define object and scene concepts as action-related elements extracted using a large language model, allowing a model to incorporate spatial context concepts without manual annotation. To ensure inherent explainability, DANCE adopts an ante-hoc design based on the concept bottleneck framework [25]. We insert a concept layer between the backbone and the final classifier, enforcing the model to first predict concept activations before predicting the final action label. The concept layer comprises nodes for motion dynamics, object, and scene concepts. This disentangled design ensures that action predictions are explicitly grounded in both dynamic (pose sequence-based) and static (object and scene) concept types. As a result, explanations produced by DANCE are not only faithful to the model's reasoning but also well-aligned with human cognitive mechanisms.
To validate the effectiveness of DANCE, we conduct experiments on four video action recognition datasets: KTH [45], Penn Action [61], HAA500 [9], and UCF-101 [49]. Results show that DANCE significantly enhances explanation clarity by disentangling motion dynamics and spatial context, while maintaining competitive recognition performance against a model without interpretability.
A user study further demonstrates that explanations generated by DANCE are more faithful and interpretable compared to those of prior approaches. Extensive qualitative comparisons also highlight the superior structure and transparency of DANCE's explanations. Finally, we showcase the practical utility of DANCE across several downstream tasks-including model debugging, editing, and failure case analysis -underscoring its broader potential for understanding and improving video recognition models.
We summarize our major contributions as follows:
• We propose DANCE, a novel video XAI framework that provides structured and motionaware explanations by disentangling motion dynamics and spatial context concepts.
• We introduce a label-free pipeline that automatically discovers motion dynamics concepts via clustering of human pose sequences and spatial context concepts through LLM querying.
The proposed pipeline allows DANCE to capture fine-grained motion patterns, objects, and scenes without manual annotations.
• We conduct comprehensive evaluations across four datasets, assessing both explainability and performance through a user study, qualitative comparisons, and ablation studies. We further demonstrate the practical utility of DANCE in model debugging and editing.
this section cite: ['b25', 'b49', 'b25', 'b37', 'b59', 'b38', 'b36', 'b0', 'b40', 'b24', 'b44', 'b60', 'b8', 'b48']

Section: Related Work
Video action recognition. The video action recognition task is classifying human actions from a temporally trimmed input video. Early approaches, such as two-stream CNNs [48], 3D CNNs [54,12,13], and temporal shift modules [32], jointly encode spatial and temporal features to capture motion and context. More recently, transformer-based architectures [3,40,11,53,28,2] have achieved substantial performance gains, largely due to large-scale pretraining. Despite these advances, the decision-making processes of most video action recognition models remain opaque, as predictions are made through complex, non-interpretable feature interactions. In this work, we propose DANCE that predicts actions based on disentangled, human-interpretable concepts, thus making the model's reasoning process more transparent and well-aligned with human cognition.
this section cite: ['b47', 'b53', 'b11', 'b12', 'b31', 'b2', 'b39', 'b10', 'b52', 'b27', 'b1']

Section: Explainable video action recognition.
Explaining the decision-making process of video action recognition models remains a relatively under-explored area. We can categorize existing methods into post-hoc explanation approaches-such as feature attribution [50,31,18]-and concept discovery methods [21,26,43]. These approaches typically use attribution or optimization techniques to identify input regions (e.g., pixels or spatio-temporal tubelets) that contribute most to the model's prediction. As a result, the explanations they produce are often unstructured and entangled, making it difficult to attribute predictions to distinct types of reasoning, such as motion dynamics versus spatial context. Moreover, optimization-based methods [21,26,43] require additional computational cost, as they need an optimization phase every time for generating an explanation. In contrast, we propose an ante-hoc framework that explicitly disentangles temporal dynamics and spatial context concepts to produce structured, human-aligned explanations. Because DANCE is inherently explainable by design, it can generate interpretable explanations in a single forward pass, without a post-hoc optimization. Disentangled/Decomposed explanations. Recently, there have been efforts to provide explanations disentangled into interpretable components in the image domain [62,24,4,8,25,38,60,39,47]. Concept bottleneck models (CBMs) insert a concept layer between the backbone and the final classifier, forcing predictions to be made explicitly through disentangled human-interpretable concepts [25,38,60,39,47]. Other approaches disentangle what concepts influence the prediction and where the concepts occurs [52,1,4]. A separate line of work addresses the limitations of attributionbased explanations-entangled attribution maps-by disentangling intermediate-layer representations into concept subspaces [8] or selecting a compact set of informative attribution regions [5]. While we also aim to provide disentangled explanations, our work differs in that we tackle the under-explored problem of explainable video action recognition. To the best of our knowledge, DANCE is a pioneering work in video XAI by explicitly disentangling motion dynamics and spatial context concepts, enabling structured, interpretable, and human-aligned explanations for video model decisions.
this section cite: ['b49', 'b30', 'b17', 'b20', 'b25', 'b42', 'b20', 'b25', 'b42', 'b61', 'b23', 'b3', 'b7', 'b24', 'b37', 'b59', 'b38', 'b46', 'b24', 'b37', 'b59', 'b38', 'b46', 'b51', 'b0', 'b3', 'b7', 'b4']

Section: DANCE
We introduce DANCE, an explainable video action recognition framework that produces structured and motion-aware explanations for its predictions. As illustrated in Figure 2, DANCE explains each prediction using three types of disentangled concepts: (i) motion dynamics, (ii) objects, and (iii) scenes. To capture fine-grained temporal patterns, we define motion dynamics concepts as representative human pose sequences extracted from training videos. These pose sequences offer appearance-agnostic representations of temporal motion, allowing users to clearly understand how an action unfolds over time-without being distracted by visual factors such as clothing, objects, or Input video 𝐕 Video features 𝐱 ∈ ℝ ! … … 0.8 0.7 0.5 Concept activations 𝐳 ∈ ℝ " … … … × 0.4 × 0 .5 … … 0.9 Figure 2: Overview of DANCE.
Given an input video, DANCE first extracts video features using a pretrained video backbone encoder. Then, three disentangled concept layers project the video features onto their own concept space-motion dynamics, object, and scene-producing disentangled activations. The interpretable classification layer linearly combines these concept activations to predict the action class. By explicitly disentangling concept types, DANCE provides structured explanations that better align with how humans perceive actions by separating motion dynamics from the spatial context.
background. In parallel, we use a large language model (LLM) to extract spatial context concepts, identifying relevant objects and scenes associated with each action.
To ensure a transparent prediction, we adopt an ante-hoc design based on the concept bottleneck framework [25,38]. As shown in Figure 2, we insert a concept layer between the backbone and the final classifier. Given a video, DANCE first predicts the activation of disentangled concepts, then uses these activations to produce the final action prediction. Through our disentangled concept bottleneck design, we ensure that an explanation from DANCE is structured and motion-aware. The remainder of this section is organized as follows. We introduce the concept bottleneck architecture in Section 3.1, detail our concept discovery process in Section 3.2, present training procedures in Section 3.3.
this section cite: ['b24', 'b37']

Section: Preliminary: Concept Bottleneck Model
Let us denote a training dataset as D = (V i , c i , y i ) N i=1 , where V i is the i-th input video, c i ∈ {0, 1} M is a binary vector indicating the presence of M concepts, and y i ∈ {0, 1} K is a one-hot vector indicating the ground-truth action label among K classes. Given a training sample (V i , c i , y i ), we first extract a D-dimensional video-level feature vector x i = f (V i ) ∈ R D using a pre-trained video backbone encoder f (•). We then project x i into M concept activations using a linear concept layer g(•; W C ) parameterized by weights W C ∈ R M×D : z i = g(x i ; W C ) ∈ R M . Then, a linear classifier h(•; W A ) with a softmax activation predicts an action label based on the concept activations: ŷi = h(z i ; W A ) ∈ R K . Unlike prior works [25,38], DANCE explicitly disentangles three types of concepts-motion dynamics, objects, and scenes-to provide structured and more intuitive explanations. To achieve this, we partition the parameters of the concept layer W C into three disjoint parameters: W C = [W m C ; W o C ; W s C ], where W m C ∈ R M m ×D , W o C ∈ R M o ×D , and W s C ∈ R M s ×D correspond to the parameters for motion dynamics, object and scene concepts, respectively. Here, M m , M o , and M s denote the number of motion dynamics, object and scene concepts, respectively. We represent motion dynamics concepts using 2D human pose sequences, which explicitly capture how the human body moves over time in an appearance-agnostic manner. For both object and scene concepts, we use intuitive text descriptions, e.g., baseball bat, tennis court, that reflect the spatial context associated with each action.
this section cite: ['b24', 'b37']

Section: Concept Discovery
For each concept type, we first discover a representative set of concepts using only the training videos from the target dataset. We then automatically annotate each video with the presence or absence of these concepts, without requiring any human supervision.
this section cite: []

Section: Motion Dynamics Concept
As shown in Figure 3 (a), we extract 2D pose sequences from all training videos and apply clustering to discover representative motion dynamics concepts. This allows us to build a compact and interpretable vocabulary of movement patterns. Key clip selection. In video sequences, not all clips are equally informative [22,7,6]; only a few temporally localized segments-such as wind-up, stride, or release in a baseball pitch-contain Input video 𝐕# Ex) "For the action <class_name>, list the most important physical objects that commonly appear when this action occurs. A baseball A baseball bat A barbell . . An exercise mat A tennis net A tennis Racket 𝑡 (c) Object concept discovery 𝐏 ! " 𝐏 ! # 𝐀 !," 𝐸𝑞(2) 𝐂 ! % Aggregating … … 1 … … 1 𝐀 !,# Video embedding 𝑬!(𝐎) 𝐸𝑞(3) 𝑡 Example of 𝐕 ! Pose estimator 𝑡 Pose Sequences Extraction Clustering Discovered motion dynamics concepts 𝐶𝑜𝑛𝑐𝑒𝑝𝑡 𝑘: One arm downswing 𝐶𝑜𝑛𝑐𝑒𝑝𝑡 1: Raise two hand 𝑡 … … Aggregating all pose sequences 𝐸"(𝑽𝒊) Prompt Object concept set 𝐎 … … 0.6 0.7 0.1 … 𝐂 / ! % 𝑀 % (d) Object concept pseudo labeling (b) Motion dynamics concept labeling Concept label Concept label Similarity vector (a) Motion dynamics concept discovery Text Encoder GPT-4o[19] S key clips with length L centered at keyframes identified by a keyframe detection algorithm. We then apply a 2D pose estimator to obtain human pose sequences from these key clips. By clustering all pose sequences across the training set, we cluster them to define each cluster as a motion dynamics concept. (b) For each video, we derive binary motion dynamics concept labels by aggregating the cluster assignment tensor across its key clips. (c) To discover object concepts, we query GPT-4o [19] with prompts containing action class names, yielding a set of object concepts for the dataset. (d) Given a video and the object concept set, we compute concept pseudo labels using a vision-language dual encoder. Specifically, we obtain a concept pseudo label vector by multiplying the object concept embedding matrix with the video embedding vector. We can obtain scene concept labels analogously.
distinctive motion cues critical for action recognition. To focus on such informative segments, we extract pose sequences from key clips only. We first detect keyframes by running an off-the-shelf method * using pixel value differences. For each selected keyframe, we extract a short video clip V s i of fixed length L centered at that frame, where s ∈ {1, • • • , S} is the key clip index. This targeted sampling strategy allows us to concentrate the concept discovery process on primitive, discriminative motion patterns that are frequently shared across different instances of the same action class-and in some cases, across classes. Please refer to the supplementary materials. Pose sequence extraction. For each key clip V s i , we apply a 2D pose estimation model [59] to every frame to obtain a pose sequence P s i ∈ R L×J×2 , where J is the number of joints. To ensure high-quality motion dynamics representations, we filter out pose sequences with low average joint confidence or large discontinuities in joint coordinates between consecutive frames. For further implementation details, please refer to the supplementary materials. Concept discovery. To discover motion dynamics concepts, we first aggregate all pose sequences from the training videos into a unified set: P = N i=1 S s=1 P s i , where P s i denotes the pose sequence from the s-th key clip of the i-th video. To group similar motion patterns, we apply a clustering algorithm, e.g., FINCH [44], to the aggregated set P, as illustrated in Figure 3 (c). We flatten each pose sequence into a feature vector before clustering. We define each resulting cluster as a distinct motion dynamics concept and assign it a unique concept index k ∈ {1, . . . , M m }, where M m is the total number of motion dynamics concepts. Based on the clustering results, we construct a binary cluster assignment tensor A = [a i,s,k ] ∈ {0, 1} N×S×M m , where each element a i,s,k is defined as:
a i,s,k =
1, if P s i belongs to cluster k, 0, otherwise.
(
Concept labeling. For each training video V i , we assign motion dynamics concept labels by checking whether any of its pose sequences P i belong to a given cluster k:
c m i,k = I( S ∑ s=1 a i,s,k ),(2)
* https://github.com/joelibaceta/video-keyframe-detector where I(•) is the indicator function. As a result, we obtain a binary motion dynamics concept label vector c m i ∈ {0, 1} M m indicating which motion dynamics concepts are present in video V i . Note that this labeling process is entirely unsupervised, requiring no manual annotations.
this section cite: ['b21', 'b6', 'b5', 'b18', 'b58', 'b43']

Section: Object and Scene Concept
Concept Discovery. To discover intuitive and clear concepts in an unsupervised manner, we leverage a large language model, as illustrated in Figure 3 (d). For each action class, we query GPT-4o [19] with two prompts: i) "For the <action class>, list the most important physical objects that commonly appear when this action occurs." and ii) "List the most common places or background scenes where <action class> typically occurs. Do not include objects or equipment". These prompts yield a diverse and semantically meaningful set of candidate object and scene concepts associated with each action. To improve concept quality and reduce redundancy, we follow prior work [38] by applying post-processing filters-removing overly long phrases, near-duplicates, and concepts that are overly similar to the action class name. For more implementation details and examples, please refer to the supplementary materials. Concept Pseudo Labeling. To avoid manual concept annotation, we employ a vision-language dual encoder [57] to generate concept pseudo labels for each training video V i . Let E V (•) and E T (•) denote the video and text encoders of the dual encoder, respectively. We first obtain a video embedding vector E V (V i ) ∈ R D , where D is the shared embedding dimension. We then encode the object concept set O obtained from GPT-4o using the text encoder to obtain an object concept embedding matrix E T (O) ∈ R M o ×D , where M o denotes the number of object concepts. Given object concept embeddings E T (O) and the video embedding E V (V i ), we compute the object pseudo concept label vector co i as:
co i = E T (O)E V (V i ) ∈ [0, 1] M o .(3)
Note that co i is a soft label. We can obtain scene concept labels cs i ∈ [0, 1] M s analogously using the scene concept embedding matrix E T (S) ∈ R M s ×D and the video embedding E V (V i ). For more details, please refer to the supplementary materials.
this section cite: ['b18', 'b37', 'b56']

Section: Training
We freeze the pretrained video backbone encoder f (•) and train the concept layer g(•; W C ) and the final classification layer h(•; W A ) in two separate stages, using concept labels derived in Section 3.2. For brevity, we omit the batch dimension and the sample index i in the following descriptions. Motion dynamics concept layer. Since the motion dynamics label c m i ∈ {0, 1} M m , derived from (2), is a multi-label binary vector rather than a one-hot vector, we train the motion dynamics concept parameters using the binary cross-entropy loss. Given the motion dynamics concept activations z m = g(x; W m C ), we apply the sigmoid activation σ (•) to each element and define the loss as:
L m = - 1 M m M m ∑ k=1 [c m k log(σ k (z m )) + (1 -c m k ) log(1 -σ k (z m ))].(4)
Here, σ k (z m ) denotes the k-th element of σ (z m ).
this section cite: []

Section: Object and scene concept layer.
To train the object concept layer, we follow prior work [38] and apply the cosine cubed loss between the pseudo label vector co (derived from (3)) and the object concept activations z o = g(x; W o C ). The cosine cubed loss emphasizes directional alignment between the predicted and target concept representations while being invariant to scale. For additional details, we refer readers to the methodology described in [38]. We train the scene concept layer in an analogous manner using the corresponding pseudo labels and activations.
L o = - z o3 • co 3 z o3 2 • co 3 2 . (5
)
Interpretable classifier. We freeze the learned concept layers and train a final linear classifier that predicts actions solely based on the concept activations. By enforcing predictions to be made through disentangled and interpretable concepts, we promote transparent and structured explanations. Let z = [z m ; z o ; z s ] denote the concatenated motion dynamics, object, and scene concept activations. We train the classifier using the standard cross-entropy loss between the ground-truth action label y and
Pred : Bench Press Input video 𝑡 1.3 1.1 0.7 Concept contribution Concepts Bench 𝑡 𝑡 𝑡 𝑡 GradCAM[46]
Saliency Tubes [50] (a) DANCE explanation (b) Saliency-based explanation  𝑡 𝑡 𝑡 Object : Volleyball Poles Basketball Shoot Volleyball Set 𝑡 𝑡 𝑡 𝑡 𝑡 𝑡 Tennis serve Place: Tennis academy Baseball Pitch Object: A pitcher's mound (a) Basketball Shoot vs Volleyball Set HAA100 dataset Penn Action dataset (b) Baseball Pitch vs Tennis Serve Ours is much better Ours is slightly better About the same Ours is slightly worse Ours is much worse 5-Strong Agree 4 -Agree 3-Neutral 2-Disagree 1-Strong Disagree (b) Interpretability of the motion dynamics concept (a) Interpretability of DANCE Percentage of user responses Ours Language-based Expert Knowledge Avg Score : 2.3 Avg Score : 3.4 Avg score : 4.3 47.3% 21.0% 0% Expert Knowledge 0.0 0.25 0.5 0.75 1.0 Which explains better? GPT VTCD Ours Ours Ours Figure 6: Interpretability of DANCE. (a)
We present user study results from pairwise comparisons evaluating the interpretability of DANCE against three video XAI baselines: (i) a concept bottleneck model using spatio-temporal concepts generated by GPT-4o [19], (ii) VTCD [26], a spatio-temporal saliency-based explanation method, and (iii) a concept bottleneck model using spatio-temporal concepts from UCF-101 attributes [49]. (b) We report user study results comparing the interpretability of three concept types: (i) language-based concepts generated by GPT-4o, (ii) expert-defined concepts based on UCF-101 attributes [49], and (iii) our proposed motion dynamics concepts.
the action prediction ŷ = h(z; W A ), with a regularization term to enhance interpretability [58,38]:
L cls = - 1 K K ∑ k=1 y k log( ŷk ) + λ [(1 -α) 1 2 ∥W A ∥ F + α ∥W A ∥ 1,1 ],(6)
where ŷk denotes the k-th element of ŷ, ∥•∥ F represents the Frobenius norm, ∥•∥ 1,1 is the element-wise ℓ 1 norm, and λ and α are balancing hyperparameters.
this section cite: ['b37', 'b37', 'b49', 'b18', 'b25', 'b48', 'b48', 'b57', 'b37']

Section: Experimental Results
In this section, carefully design and conduct rigorous experiments to answer the following research questions: (1) Does DANCE generate explanations that are easy for humans to interpret in the context of action prediction? (Section 4.1) (2) Can DANCE detect changes in the temporal domain, such as reversed input sequences? (Section 4.1) (3) What is the performance trade-off, if any, when interpretability is introduced into a previously non-interpretable model? (Section 4.2) (4) Can DANCE be effectively used for model debugging and editing? (Section 4.3) For more details on the dataset and implementation, please refer to the supplementary materials.
this section cite: []

Section: Analysis
In this section, we examine whether DANCE produces explanations that are easily interpretable for humans in the context of action prediction. To this end, we first define concept contribution to quantify the importance of each concept neuron, and then provide sample-level and model-level explanations, followed by the results of a user study evaluation. Additional visualizations and further details for this section are provided in the supplementary materials.
this section cite: []

Section: Concept contribution.
We define the concept contribution as the product of a concept activation and the concept weight associated with the predicted class and we use this term consistently throughout the paper. To visualize a motion dynamics concept, we select the pose sequence closest to the cluster medoid and use it as the representative example.  𝑡 𝑡 1.3 0.9 0.6 Concept contribution Concepts Input video 𝑡 GT : Gym Plank Pred : Push up 𝑡 Figure 8: Failure case analysis using DANCE. With intuitive explanations, DANCE can help failure case analysis.
Sample-level explanation. In Figure 4, we visualize the top-3 contributing concepts along with the input video. In Figure 4 (a), DANCE leverages the motion concepts "lowering" and "lifting", along with the object concept "Bench", which together support the correct prediction. In contrast, as shown in Figure 4 (b), saliency-based methods [46,50] produce spatio-temporally entangled explanations, making it unclear whether the model bases its decision on the object itself, i.e., "barbell" or its motion, i.e., up-down movement when predicting Bench Press.
Model-level explanation. In Figure 5, we visualize concept-to-class weights for pairs of similar action classes using Sankey diagrams. The thickness of each edge represents the relative contribution weight. DANCE provides the model's decision basis, helping users understand how it discriminates between similar actions. For example, in Figure 5 (a), Basketball Shoot and Volleyball Set share common motion concepts (shown in light khaki), while DANCE distinguishes between them using subtle motion differences (shown in pink) and the object concept, "Volleyball Poles."
this section cite: ['b45', 'b49']

Section: Interpretability of DANCE.
To evaluate the effectiveness of DANCE in explaining model predictions, we conduct a user study comparing it against three baseline explainable video action recognition methods: (i) a CBM using entangled spatio-temporal concepts generated by GPT-4o [19], (ii) VTCD [26], a spatio-temporal saliency-based method, and (iii) a CBM using spatio-temporal concepts from UCF-101 attributes [49]. In each question, we ask each participant a pairwise comparison question, by showing a video along with explanations from DANCE and one of the baselines: "Which explanation helps you better understand why the model predicted the action?" We collect responses on a five-point Likert scale, ranging from "DANCE is much better" to "the other method is much better." As shown in Figure 6 (a), more than 70% of the responses fall into "ours is much better" or "ours is slightly better" categories across all three pairwise comparisons. These results showcase that users perceive DANCE as more intuitive and trustworthy than (i) language-based explanations, (ii) unstructured saliency-based attributions, and (iii) expert-defined concepts. For more details, please refer to the supplementary materials.
this section cite: ['b18', 'b25', 'b48']

Section: Interpretability of the proposed motion dynamics concept.
We conduct a user study evaluating the interpretability of three CBMs using the following temporal concepts: (i) language-based concepts generated by GPT-4o [19], (ii) expert-defined concepts based on UCF-101 attributes [49], and (iii) our proposed motion dynamics concepts. For each baseline, we randomly select one concept and visualize the top six video clips whose top-1 concept activation corresponds to the selected concept.
We then presented the concept alongside the activated video clips, and asked participants the following question: "How well does the given concept match the actions or motions shown in the video?" We collect the response on a five-point Likert scale, where higher scores indicate stronger perceived alignment and interpretability. As shown in Figure 6 (b), our motion dynamics concept achieves the highest average score of 4.3, with 89.7% of participants rating it 4 or 5. In contrast, language-based and expert-defined concepts receive lower average scores of 2.3 and 3.4, respectively. These results indicate that the proposed motion dynamics concept is significantly more intuitive and aligned with human perception of motion, providing more interpretable explanations. For more details, please refer to the supplementary materials.
this section cite: ['b18', 'b48']

Section: Sanity check.
Here, we check the sanity of DANCE from the perspectives of model behavior and explanation quality. In Figure 7, we compare the predictions and top-2 contributing concepts of both methods for (i) the original video and (ii) the same video played backward. DANCE correctly predicts Bowing FullBody for the original video leveraging the visualized motion dynamics concepts. For the backward video, DANCE predicts Burpee by leveraging "standing up" like motions as visualized, Table 1: Video action recognition performance. We report the Top-1 accuracy (%) of the baselines with and without interpretability as well as DANCE, all using the same backbone encoder [53].
Method KTH [45] Penn Action [61] HAA-100 [9] UCF-101 [49] Baseline w/o interpretability 89.7 97.8 73.5 88.4 CBM [25] w/ UCF-101 attributes ---86.8 LF-CBM [38] w/ entangled language concepts 87.4 96.3 66.5 85.5 LF-CBM [38] w/ disentangled language concepts 89.9 97.7 65.3 83.7 DANCE 91.1 98.1 70.7 87.5 Intervention 𝑡 2.2 1.6 0.7 Concept contribution Deactivating! 𝑡 Original Pred : Table Tennis Shot Table Tennis Club Input Video 0.0 Concepts GT : Cricket Shot 𝑡 New Pred : Cricket Shot
this section cite: ['b52', 'b44', 'b60', 'b8', 'b48', 'b24', 'b37', 'b37']

Section: → →
Figure 9: Sample-level intervention. Deactivating the irrelevant concept leads to fixing the misprediction to a correct predcition.
Golf Swing 𝑡 Cross-domain video
this section cite: []

Section: Domain shift example
In-domain video
this section cite: []

Section: Performance Evaluation Interpretability and performance do not always trade-off.
We compare DANCE with a baseline model without interpretability, as shown in Table 1. DANCE achieves slightly higher accuracy on KTH and Penn Action, while showing a modest drop of 2.8 points on HAA-100 and 0.9 points on UCF-101. These results indicate that DANCE can deliver intuitive and structured explanations without substantially compromising classification performance-and in some cases, even improving it. Clearer concepts improves performance. We further investigate whether the clarity of concept representations affects model performance. To this end, we compare DANCE with the following baselines: (i) concept bottleneck model with UCF-101 attributes [49], (ii) a label-free concept bottleneck model [38] using spatio-temporally entangled concepts generated by GPT-4o, and (iii) a variant of the label-free concept bottleneck model that uses the same object and scene concepts as DANCE, but uses GPT-4o-generated temporal concepts. Across all datasets, DANCE consistently outperforms these baselines, demonstrating that employing clearer and disentangled concept representations-particularly for motion dynamics-can lead to improved action recognition performance.
this section cite: ['b48', 'b37']

Section: Model Editing
Here, we demonstrate the utility of DANCE in debugging itself by analyzing sample-level concept contributions in misclassifications and inspecting class-level weights. We provide additional results of model editing in the supplementary materials.
this section cite: []

Section: Sample-level intervention.
In Figure 9, we illustrate how a user can intervene the model by removing a specific concept in the case of misclassification. For example, in Figure 9 (a), the model initially predicts
Table Tennis Shot, largely influenced by the scene-level concept "Table tennis club." When this concept is deactivated, the model leverages motion dynamics concepts and correctly classifies the input as Cricket Shot. The results demonstrates that DANCE supports fine-grained, transparent control over predictions, allowing users to actively adjust model behavior.
this section cite: []

Section: Cross domain class-level intervention.
In Figure 10, we demonstrate class-level intervention using DANCE. This experiment evaluates whether such adjustments can resolve performance drop caused by severe distribution shift. We evaluate on UCF-101-SCUBA [29], a variant of UCF-101 [49] where test video backgrounds are altered to induce a domain shift. As shown in Figure 10, for the Volleyball Spiking class, the model activates a relevant motion dynamics concept but ignores it due to a zero weight. By assigning a weight of 1.0 to this concept, we correct 98 misclassifications with only one additional error, significantly improving overall accuracy by 2.5 points (84.0% → 86.5%). Further adjusting weights for the Golf Swing and Tennis Swing classes result in a 4.3 point accuracy improvement (77.7% → 82.0%). These findings highlight DANCE's ability to support post hoc model debugging and performance recovery under severe domain shifts, without retraining.
this section cite: ['b28', 'b48']

Section: Conclusions
In this paper, we propose DANCE to address the challenge of explaining video action recognition models in a structured and motion-aware manner. DANCE grounds its predictions in three humaninterpretable concept types-motion dynamics, objects, and scenes-enabling cognitively aligned and transparent explanations. This design facilitates intuitive understanding of model behavior by explicitly separating temporal and spatial reasoning. Through extensive experiments and practical use cases, we show that DANCE delivers clearer and more faithful explanations while maintaining competitive recognition performance. Moreover, DANCE supports effective model editing-even under severe domain shifts-without requiring retraining. We believe our work offers valuable insights to the XAI and video understanding communities and will help inspire future research.
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The main claims in the abstract and introduction accurately summarize the contributions and scope of the paper. The claims are supported by experimental results, as detailed in section 4.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The limitations of our work are thoroughly discussed in the dedicated "Limitations" section of supplementary materials.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: The paper does not include any theoretical results.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The paper provides all necessary details to reproduce the main experimental results, including detailed descriptions of the dataset, model architecture, and evaluation metrics. These are thoroughly documented in the supplementary material.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We provide code in the supplementary material.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes]
Justification: The paper specifies all the necessary training and test details, including data splits, hyperparameters, and the type of optimizer used. These details are presented in the supplementary materials.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [No] Justification: While the paper provides some information about the experimental results, it lacks detailed reporting of error bars or other statistical significance measures. This may affect the interpretation of the results. Future revisions will aim to include these details.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper does not involve the release of data or models with high risk for misuse, hence this question is not applicable.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We use publicly available datasets (UCF-101 [49], Penn Action [61], KTH [45], HAA-500 [9]) under their academic licenses. We also use the pre-trained VideoMAE [53] encoder (Apache 2.0 license) and the ViTPose (Apache 2.0 license) pose estimator [59]. For object/scene concept generation, we query GPT-4o [19], which is used in accordance with OpenAI's usage policy. All sources are cited, and licenses are respected.
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: We introduce a new implementation of the DANCE framework, including code for concept discovery, training, and visualization. If released, the code will include documentation, usage instructions, and evaluation scripts. We also curate a subset of the HAA-500 dataset (HAA-100) to facilitate interpretable evaluation. The curation process and selection criteria are clearly described in the supplementary material.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [Yes] Justification: The paper includes a user study with human subjects to evaluate the interpretability of concept-based explanations. Key methodological details are provided in the main paper. The supplementary material includes full task instructions, interface screenshots, and compensation information.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [Yes] Justification: The study involving human subjects was reviewed by an Institutional Review Board (IRB) and received an exemption determination based on minimal risk and anonymized data collection. No personally identifiable information was collected, and all participants were informed and gave consent prior to participation.
this section cite: ['b48', 'b60', 'b44', 'b8', 'b52', 'b58', 'b18']

Section: 
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We report compute resource details-including GPU type, memory, and per-experiment training time-in the supplementary materials. This includes estimates of training time for each dataset, hardware specifications, ensuring reproducibility. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research conducted in this paper fully conforms with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: We include a discussion of potential broader impacts in the supplementary materials. Specifically, we highlight the positive societal impact of improving the interpretability of video action recognition models by disentangling motion dynamics from spatial context-enhancing transparency and trust in applications such as healthcare, education, and safety-critical systems. We also acknowledge potential risks, such as misuse in surveillance scenarios, and emphasize the importance of responsible deployment and ongoing monitoring.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [Yes] Justification: We employed a large language model (LLM), specifically GPT-4o, to generate candidate object and scene concepts that form part of our concept vocabulary. These LLM-generated concepts serve as human-interpretable descriptors used in our explanation framework. The LLM was not involved in model training or prediction but contributed to the construction of interpretable spatial concepts, which are integral to our method. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/  LLM) for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Www: a unified framework for explaining what where and why of neural networks by interpretation of neuron concepts Year: (2024)
Ref_id:b1 Title: Learning disentangled video representations of action and scene for holistic video understanding Year: (2024)
Ref_id:b2 Title: Is space-time attention all you need for video understanding Year: (2021)
Ref_id:b3 Title: This looks like that: deep learning for interpretable image recognition Year: (2019)
Ref_id:b4 Title: Less is more: Fewer interpretable region via submodular subset selection Year: (2024)
Ref_id:b5 Title: Video action recognition with attentive semantic units Year: (2023)
Ref_id:b6 Title: Shuffle and attend: Video domain adaptation Year: (2020)
Ref_id:b7 Title: Disentangled explanations of neural network predictions by finding relevant subspaces Year: (2024)
Ref_id:b8 Title: Humancentric atomic action dataset with curated videos Year: (2021)
Ref_id:b9 Title: Towards a rigorous science of interpretable machine learning Year: (2017)
Ref_id:b10 Title: Multiscale vision transformers Year: ()
Ref_id:b11 Title: X3d: Expanding architectures for efficient video recognition Year: (2020)
Ref_id:b12 Title: Slowfast networks for video recognition Year: (2019)
Ref_id:b13 Title: Understanding deep networks via extremal perturbations and smooth masks Year: (2019)
Ref_id:b14 Title: Towards automatic concept-based explanations Year: (2019)
Ref_id:b15 Title: Separate visual pathways for perception and action Year: (1992)
Ref_id:b16 Title: Brain areas involved in perception of biological motion Year: (2000)
Ref_id:b17 Title: Swag-v: explanations for video using superpixels weighted by average gradients Year: (2022)
Ref_id:b18 Title: Gpt-4o system card Year: (2024)
Ref_id:b19 Title: Formalizing trust in artificial intelligence: Prerequisites, causes and goals of human trust in ai Year: (2021)
Ref_id:b20 Title: Spatial-temporal concept based explanation of 3d convnets Year: (2023)
Ref_id:b21 Title: Adascan: Adaptive scan pooling in deep convolutional neural networks for human action recognition in videos Year: (2017)
Ref_id:b22 Title: Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav) Year: (2018)
Ref_id:b23 Title: Improving explainability of disentangled representations using multipath-attribution mappings Year: (2022)
Ref_id:b24 Title: Concept bottleneck models Year: (2009)
Ref_id:b25 Title: Understanding video transformers via universal concept discovery Year: (2008)
Ref_id:b26 Title: A new neural framework for visuospatial processing Year: (2011)
Ref_id:b27 Title: Cast: Cross-attention in space and time for video action recognition Year: ()
Ref_id:b28 Title: Mitigating and evaluating static bias of action representations in the background and the foreground Year: (2023)
Ref_id:b29 Title: Videomamba: State space model for efficient video understanding Year: (2024)
Ref_id:b30 Title: Towards visually explaining video understanding networks with perturbation Year: ()
Ref_id:b31 Title: TSM: Temporal Shift Module for Efficient Video Understanding Year: (2019)
Ref_id:b32 Title: The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery Year: (2018)
Ref_id:b33 Title: Evidence-based principles for how to design effective instructional videos Year: (2021)
Ref_id:b34 Title: Nine ways to reduce cognitive load in multimedia learning Year: (2003)
Ref_id:b35 Title: Principles for managing essential processing in multimedia learning: Segmenting, pretraining, and modality principles. The Cambridge handbook of multimedia learning Year: (2005)
Ref_id:b36 Title: Clip-dissect: Automatic description of neuron representations in deep vision networks Year: (2023)
Ref_id:b37 Title: Label-free concept bottleneck models Year: (2009)
Ref_id:b38 Title: Coarse-to-fine concept bottleneck models Year: (2024)
Ref_id:b39 Title: Keeping your eye on the ball: Trajectory attention in video transformers Year: (2021)
Ref_id:b40 Title: Implicit learning and tacit knowledge Year: (1989)
Ref_id:b41 Title: Why should i trust you? explaining the predictions of any classifier Year: (2016)
Ref_id:b42 Title: Exploring explainability in video action recognition Year: ()
Ref_id:b43 Title: Efficient parameter-free clustering using first neighbor relations Year: (2019)
Ref_id:b44 Title: Recognizing human actions: a local svm approach Year: (2004)
Ref_id:b45 Title: Grad-cam: Visual explanations from deep networks via gradient-based localization Year: (2008)
Ref_id:b46 Title: Incremental residual concept bottleneck models Year: (2024)
Ref_id:b47 Title: Two-stream convolutional networks for action recognition in videos Year: (2014)
Ref_id:b48 Title: A dataset of 101 human action classes from videos in the wild Year: (2009)
Ref_id:b49 Title: Saliency tubes: Visual explanations for spatio-temporal convolutions Year: (2008)
Ref_id:b50 Title: Cognitive load during problem solving: Effects on learning Year: (1988)
Ref_id:b51 Title: Craft: Concept recursive activation factorization for explainability Year: (2023)
Ref_id:b52 Title: VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training Year: (2009)
Ref_id:b53 Title: Learning spatiotemporal features with 3d convolutional networks Year: (2015)
Ref_id:b54 Title: Animation: can it facilitate? Year: (2002)
Ref_id:b55 Title: Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning Year: (2023)
Ref_id:b56 Title: Internvid: A large-scale video-text dataset for multimodal understanding and generation Year: (2023)
Ref_id:b57 Title: Leveraging sparse linear layers for debuggable deep networks Year: (2021)
Ref_id:b58 Title: Vitpose: Simple vision transformer baselines for human pose estimation Year: (2022)
Ref_id:b59 Title: Post-hoc concept bottleneck models Year: (2023)
Ref_id:b60 Title: From actemes to action: A stronglysupervised representation for detailed action understanding Year: (2013)
Ref_id:b61 Title: Learning disentangled semantic spaces of explanations via invertible neural networks Year: (2023)
