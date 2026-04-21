Title: Interactive Cross-modal Learning for Text-3D Scene Retrieval
Abstract: Text-3D Scene Retrieval (T3SR) aims to retrieve relevant scenes using linguistic queries. Although traditional T3SR methods have made significant progress in capturing fine-grained associations, they implicitly assume that query descriptions are information-complete. In practical deployments, however, limited by the capabilities of users and models, it is difficult or even impossible to directly obtain a perfect textual query suiting the entire scene and model, thereby leading to performance degradation. To address this issue, we propose a novel Interactive Text-3D Scene Retrieval Method (IDeal), which promotes the enhancement of the alignment between texts and 3D scenes through continuous interaction. To achieve this, we present an Interactive Retrieval Refinement framework (IRR), which employs a questioner to pose contextually relevant questions to an answerer in successive rounds that either promote detailed probing or encourage exploratory divergence within scenes. Upon the iterative responses received from the answerer, IRR adopts a retriever to perform both feature-level and semantic-level information fusion, facilitating scene-level interaction and understanding for more precise re-rankings. To bridge the domain gap between queries and interactive texts, we propose an Interaction Adaptation Tuning strategy (IAT). IAT mitigates the discriminability and diversity risks among augmented text features that approximate the interaction text domain, achieving contrastive domain adaptation for our retriever. Extensive experimental results on three datasets demonstrate the superiority of IDeal. Code is available at https://github.com/Yangl1nFeng/IDeal.

Section: Introduction
Recent years have witnessed natural language interfaces to embodied intelligence systems [1,2,3,4,5] become increasingly prevalent in our daily lives. This opens up further opportunities for natural language-based interaction with intelligent agents, such as a user verbally instructing agents to perform tasks in a specific scene. Before executing any task, an agent must first retrieve the scene relevant to the user's intent. This requirement has spurred recent works on Text-3D Scene Retrieval (T3SR) [6,7], which enables the retrieval of 3D point-cloud scenes using linguistic queries. Such a language-scene alignment capability lays a critical foundation for enabling agents to generalize across scenes and environments.
Although existing dedicated methods [6] achieve promising performance for T3SR by facilitating fine-grained query text and scene understanding, such success often relies on the assumption that A white mattress is on a brown bed, with a dumbbell placed on it.
What object is on the opposite side of the bed?
Provide more details about the bed.
Uninformative query. Informative enough. Retriever Scene0073 Bed Chair Dumbbell … Never seen mattress, dumbbell. Too long sentence.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b5']

Section: Round i+2
The black chair without arms opposite the bed.
A white mattress is on a brown bed, with a dumbbell placed on it.
FIND bed in my room.
this section cite: []

Section: Interaction text
What color is bed?
Is it white? What is the shape?
this section cite: []

Section: …
Bed Retriever Questioner Round 0 A brown bed.
A brown square bed. A brown square bed.
A brown square bed.
A square bed.
Figure 1: Overview of interactive text-3D scene retrieval. The above gray part illustrates the two specific challenges, while the below white part shows the illustrations of our interactive framework.
the queries provided are information-complete. However, such an assumption is often violated in real-world scenarios due to the inherent limitations of text inputs and models, such as incomplete one-shot descriptions of user intent [8], ambiguous descriptions [6], domain shifts [9], and limited generalization of the models. As a result, the performance and robustness of the models remain persistently constrained, and relying solely on limited internal knowledge is insufficient to overcome this inherent bottleneck.
To break through the bottleneck, recent studies [8,10,11] have explored integrating external knowledge from Large Language Models (LLMs) to enhance their understanding and alignment abilities. However, such approaches typically require prohibitively expensive fine-tuning or retraining of offline models [10,12]. Some other attempts [8,13,14] have proposed interactive cross-modal retrieval frameworks that incorporate LLMs and Vision-Language Models (VLMs) to facilitate more fine-grained understanding and alignment, thereby iteratively evolving the retrieval performance. Although these methods have demonstrated remarkable effectiveness in image-text matching [8] and video-text retrieval [15,16], they still face two tricky challenges in the T3SR setting, as shown in Figure 1. Firstly, since the scale and complexity of 3D scenes, these methods lack holistic perspectives beyond a localized focus during interaction, e.g., the LLMs tend to focus on the salient objects in the scene and ignore the fine-grained details at a scene-level perspective, limiting the depth and breadth of LLM interaction, as demonstrated in Table 1. Secondly, existing retrieval models exhibit limited generalization ability to biased text domains, limiting their effectiveness in handling realistic interaction texts that exhibit domain gaps.
To address the aforementioned challenges, this paper proposes a novel Interactive Text-3D Scene Retrieval method (IDeal) to conduct continuous interaction between the T3SR models and external users (e.g., LLMs), achieving the active alignment between text queries and 3D scenes, as depicted in Figure 1. Our IDeal consists of two components: an Interactive Retrieval Refinement framework (IRR) and an Interaction Adaptation Tuning strategy (IAT), as illustrated in Figure 2. More specifically, IRR coordinates three specialized agents (i.e., questioner, answerer, and retriever) to perform multiround interaction. First, the questioner adaptively determines whether to continue probing object details or to pursue divergence by exploring the broader scene, based on the assessment of the current round's description. Based on this, it continuously formulates context-relevant questions to the answerer. After receiving responses, the retriever iteratively integrates information at both the feature and semantic levels, facilitating comprehensive scene-level understanding for progressively precise re-rankings. To mitigate the domain shift between training queries and interactive texts, IAT proposes adapting the retriever toward the interaction text domain. Specifically, IAT leverages LLMs to generate more realistic augmented texts that closely resemble the interaction text domain. Subsequently, IAT robustly mitigates the discriminability and diversity theoretical risks in the features of the augmented texts for domain gap bridging, thereby ensuring an unbiased understanding of the interaction texts by the retriever. The contributions of this paper are as follows:
• We propose a novel Interactive Text-3D Scene Retrieval Method (IDeal), which actively enhances alignment between text queries and 3D scenes through ongoing interactions.
• An Interactive Retrieval Refinement framework (IRR) is presented to enable a deep interaction for comprehensive scene exploration, leading to progressively improved retrieval.
• An Interaction Adaptation Tuning strategy (IAT) is proposed, which facilitates the transfer of the retriever to the interaction text domain, promoting improved interaction.
• We conduct extensive comparison experiments on text-3D scene datasets. Our IDeal remarkably outperforms the existing methods, demonstrating its superiority.
this section cite: ['b7', 'b5', 'b8', 'b7', 'b9', 'b10', 'b9', 'b11', 'b7', 'b12', 'b13', 'b7', 'b14', 'b15']

Section: Related Work
Cross-Modal Retrieval. Cross-Modal Retrieval (CMR) [17,18,19,20,21,22] aims to match corresponding results across modalities for a given query, bridging the gap caused by modal heterogeneity. In recent years, CMR has garnered significant attention in fields such as Image-Text Retrieval [14,23,24,25], Video-Text Retrieval [26,27], 2D-3D Retrieval [28,29], Pointcloud-Text Matching [6]. The primary challenge of CMR lies in effectively aligning multimodal data. To address this issue, most existing works could be broadly categorized into two groups: 1) Coarse-grained retrieval [30,31,32] directly maps multimodal data into a shared space, aiming for a more straightforward and computationally efficient alignment. 2) Fine-grained retrieval [33,34] seeks to establish local associations between fine-grained features across modalities (e.g., regions in images, words in texts). These local associations are then progressively integrated to form precise cross-modal correspondences. This paper focuses on a more challenging CMR task, i.e., Text-3D Scene Retrieval, involving obscure spatial cues and sophisticated 3D scenes (including issues such as viewpoints and occlusions [35,36]). Although prior work [6] performs well with comprehensive descriptions, it struggles with online and blurry queries. To this end, we propose an Interactive T3SR solution that iteratively incorporates online feedback to achieve more precise and practical scene retrieval.
Interactive Learning. Unlike traditional learning paradigms [6,37], interactive learning emphasizes the continuous improvement of a model's behavior through ongoing interactions with the environment or users. Specifically, several pioneering works [38,39] leverage simple forms of user feedback (e.g., preferred sample selection and relevance scoring) to iteratively achieve improved training quality or better satisfy user-specific requirements during testing. With the development of Large Language Models (LLMs), other studies [8,40,41] have begun exploring question-answering interactions through free-form text dialogue, closely replicating natural human communication. For example, several methods leverage iterative interactions to continuously refine the retrieval query for better reranking. Recently, more advanced methods such as PlugIR [13], MERLIN [16], ICL [42], and LLaVA-ReID [43] have integrated LLMs for context-aware question generation, mining more visual details. However, these methods cannot be effectively generalized to T3SR due to the differences in tasks and data domains shown in Figure 1. In this paper, we develop an interactive framework tailored for T3SR to help the offline models adapt to complex scene perception.
this section cite: ['b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b13', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b5', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b5', 'b5', 'b36', 'b37', 'b38', 'b7', 'b39', 'b40', 'b12', 'b15', 'b41', 'b42']

Section: Method

this section cite: []

Section: Problem Formulation
Given a text query set T = {t i } nt i=1 and a 3D scene gallery C = {c j } nc j=1 , where t i and c j represent i-th text and j-th scene, n t = |T | and n c = |C| means the sample number, and | • | denotes the volume of set. The purpose of T3SR is to use the text query to match the ideal 3D scenes from the gallery, where there exists correspondence y i,j ∈ {0, 1}, indicating whether the points are matched (i.e., y ij = 1), or unmatched (i.e., y ij = 0). Existing methods [6,37] typically train an offline model to achieve encoding of multimodal data, followed by meticulous single-turn retrieval. They assume that user-provided text queries are information complete, overlooking the practical fact that queries are often partial, ambiguous, or even exhibit domain shift.
To address these issues, an interactive Text-3D scene retrieval method, i.e., IDeal, is proposed to bridge the interaction between the retrieval models and external agents, progressively overcoming the aforementioned query limitations and achieving improved alignment between texts and 3D scenes. More specifically, IDeal asks question q l i (l ∈ {1, • • • , r}) about i-th sample based on the users' previous response a l-1 i (a 0 i = t i ), where r is the upper limit of rounds. Subsequently, the external agents recall details and answer a l i of the target 3D scene c j (y ij = 1), forming a dialogue context
D i = {a 0 i , (q 1 i , a 1 i ), • • • (q r i , a r i )} composed of question-answer pairs (q • i , a • i ).
Both the j-round response text and 3D scenes are projected into a shared feature space by a trained retrieval model, which can be: u j i = f r (a j i ; θ) and v i = f r (c i ; θ), where θ denotes the learnable parameters.
this section cite: ['b5', 'b36']

Section: Init (a0):
There is a white bed in my room.
this section cite: []

Section: 3D Scene Set
… 3D Point-cloud Encoder Text Encoder Objects: bed, chair, cabinet curtain, bin, dumbbells… Scene: The bed is placed along the its right in the corner, creatupper-left wall, with a white cabinet. A black chair sits facing away from the bed … ar-1: There is a bed here and a messy cabinet next to it. Router qk: What color are the bed and cabinet? Is there anything on the bed? q: Are there other objects in the room that are not mentioned? Retriever Shared space ak-1: There is a bed here, and a messy cabinet next to it. Questioner Answerer (LLM / User) Memory: the bed is placed along the wall, with a white cabinet to its right in the corner. A pair of dumbbells were lying on floor nearby … Probe for details Diverge to explore Init a1 … Initial Query and r-Round Answers Matched Scene Semantic-level Fusion Feature-level Fusion Combined Project ǁ 𝜀 (a) Feature-level Fusion ii. Scene Reconstruction i. Object Extraction (b) Semantic-level Fusion LLM Matching fused text / target scene features + + Central aggregation Average fusion + Uninformative / informative text features Weighted linear fusion + + + Away from the current describing area: dresser, chair … ar: The black chair without arms opposite the white bed. a0, a1 -ar a2 ar ar qr Boundary Set Central set Hypersphere
this section cite: []

Section: Interactive Retrieval Refinement Framework
In this section, we introduce the Interactive Retrieval Refinement framework (IRR), which coordinates three agents (i.e., questioner, answerer, and retriever) to enable iterative interaction. In the following sections, we will elaborate on them by introducing their interaction process.
this section cite: []

Section: Adaptive Questioning
To achieve a deep interaction for T3SR, we present an adaptive questioner, which enables pertinent questioning to the answerer for both detailed and comprehensive exploration of complex 3D scenes. Firstly, a question router is employed, which determines the focus of questioning based on the feature distribution in the shared space. More specifically, the router assesses whether the previous-round description is informative by computing a Cross-modal Affinity Entropy (E), formulated as:
E(u r-1 i ) = - j∈N k (u r-1 i ) p(u r-1 i , v j ) log p(u r-1 i , v j ),(1)
where
N k (u r-1 i ) means the k-nearest-neighbor index of u r-1 i
, and affinity probability p(u r-1 i , v j ) is formulated as:
p(u r-1 i , v j ) = exp(S(u r-1 i , v j )/τ ) l∈N k (u r-1 i ) exp(S(u r-1 i , v l )/τ ) ,(2)
where S represents the computation of similarity between features, and τ is a temperature parameter.
However, the distribution of scene features extracted by the trained retrieval models is fixed and inherently non-uniform, with regions of over-density and under-density introducing a structural density bias [44]. To mitigate the bias, we introduce a Density Compensated Factor for each scene feature, which is formulated as:
ρ(v i ) = 1 (1/k) j∈N k (v j ) D(vi,vj )+ϵ
, where D represents the distance calculation and ϵ is a minimal constant for numerical stability. Based on this, we try to approximately correct the original similarity score S(u r-1 i , v j ) by incorporating the Density Compensated Factor ρ(v i ), which could be written as: S(u
(r-1) i , v j ) = S(u (r-1) i , v j )/ ρ(v j ).
Subsequently, this corrected similarity is brought back into Equations ( 1) and (2) to obtain a Density Compensated Affinity Entropy, denoted as Ẽ. This process approximates a Bayesian Correction leveraging prior density estimationfoot_0 , mitigating the impact of the inherent bias in scene feature distribution.
Leveraging the fairer metric Ẽ, our questioner categorize descriptions with Ẽ > β as uninformative, prompting an LLM to generate questions for detail probe (i.e., Q 1 ) for attribute and spatial relationship detail refinements within the described area. Conversely, when Ẽ ≤ β, the descriptions are considered informative, triggering the adopting of questions for diverge exploration (i.e., Q 2 ) that inquire about object arrangements not previously discussed in the dialogue.
this section cite: ['b43']

Section: Iterative Retrieval
After completing the questioning, an LLM is employed to simulate the external user acting as an answerer to answer the questions, following existing interactive approaches [8,13]. It receives multi-round questions and provides responses based on its memory. In this paper, we adopt text modality to simulate the memory, which better approximates how humans recall information in mind.
Upon receiving the response descriptions from the answerer, we construct a retriever that can be seamlessly integrated with existing cross-modal models [37,45], enabling iterative scene retrieval. It can obtain the scene retrieval predictions for any given text within the shared feature space. Specifically, given a text feature u i , the prediction can be formulated as:
p(u i ) = [p(u i , v 1 ), p(u i , v 2 ), . . . , p(u i , v nc )] ⊤ ,(3)
where
p(u i , v j ) = exp(S(u i , v j ))/ nc l=1 exp(S(u i , v l ))
denotes the probability that the i-th text retrieves the j-th scene. Accordingly, our retriever first utilizes the initial query to compute an initial retrieval prediction p1 (u i ). Subsequently, the feature-level and semantic-level fusion of interactive responses is conducted to achieve more precise scene retrieval.
For the fusion of interactive response feature, on one hand, considering responses to Q 1 are refinements of the previous-round descriptions, we apply a weighted linear fusion to incorporate supplementary information. This strategy enables the preservation of core semantic cues from previous rounds while emphasizing newly introduced details:
u j i = αu j i + (1 -α)u j-1 i
, where q j i ∈ Q 1 , α is a trade-off weight. On the other hand, benefiting from Q 2 , the other response features and the aforementioned fused features capture variations across different regions of the scene. To fuse them into a comprehensive feature, inspired by [46], we model their distribution around the target scene by encapsulating them within a minimum enclosing hypersphere:
(o * i , R * i ) = arg min oi,Ri R i : u j i o ⊤ i ≤ R i , ∀j ,(4)
where o * i and R * i are the center and radius of the hypersphere, respectively. Based on this, features near the hypersphere boundary are grouped into a boundary set U 1  i , while the remainder constitute the central set U 2 i . To balance fusion robustness and feature discrimination, potentially noisy boundary features in U 1  i are aggregated at the hypersphere center and averaged with cleaner features in U 2 i to yield the final fused response feature: ūi = 1  2 o * i + 1
|U 2 i | u j i ∈U 2 i u j i .
This fused feature is then input into Equation (3) to obtain an interactive feature prediction p2 ( ūi ).
However, the aforementioned feature-level fusion can not fully capture the holistic semantics of the responses. To address this limitation, we leverage an LLM to reconstruct the 3D scene from all responses in textual space. More specifically, inspired by Chain-of-Thought (CoT) [47], we decompose this process into object extraction and scene reconstruction for a more stable and comprehensive scene summary. The LLM first identifies the scene objects across multi-round responses and then summarizes an object-centric scene reconstruction text. Finally, the texts are encoded into feature s i , from which a interactive semantic prediction p3 (s i ) is computed using Equation (3).
Finally, the initial and interactive predictions are combined through weighted fusion to obtain the final scene retrieval prediction as follows:
pc
(u i ) = λ 1 p1 (u i ) + λ 2 p2 ( ūi ) + λ 3 p3 (s i ),(5)
where pc (u i ) is the final retrieval prediction, λ 1 , λ 2 , and λ 3 are trade-off parameters. Benefiting from the adaptive questioning and the comprehensive retrieval information fusion, our IDeal can alleviate the limitations of initial queries through progressive interaction.
this section cite: ['b7', 'b12', 'b36', 'b44', 'b45', 'b46', 'b2']

Section: Interaction Adaptation Tuning
Although IRR can exploit interactions to promote retrieval quality, the limited text domain of the retriever remains a bottleneck that restricts further performance improvements. To overcome this limitation, we propose an Interaction Adaptation Tuning strategy (IAT), which enhances texts to approximate the domain of interaction texts. We begin by integrating information and descriptions of the same scenes from the training data to construct simulated memory for text augmentation. Following IRR, we first provide a training-databased answerer (i.e., an LLM) with the constructed memory and initial queries. We then simulate the IRR interaction process by iteratively posing a fixed number of Q 1 and Q 2 questions. The response descriptions yield augmented texts that closely approximate the interaction scenario.
After obtaining the enriched augmented texts, inspired by the contrastive domain adaptation paradigm [48,49], we try to minimize the theoretical risk R(θ) associated with our retriever among the augmented text features, as formulated below. It facilitates model adaptation of the retriever to the augmented text domain without requiring access to its implementation details.
R(θ) = R dis (θ) + R div (θ) = E Ũ -E Ũ + S( ũ+ i , ũi ) + E Ũ -S( ũ- i , ũi ) ,(6)
where the two components R dis (θ) and R div (θ) respectively reflect discriminability and diversity risks, E denotes expectation, E Ũ is taken with respect to the distribution for the target domain features (i.e., augmented text features Ũ = { ũi } nt i=1 ), and Ũ+ and Ũis the distribution for the corresponding positive features ũ+ i and negative features ũi , respectively. On the one hand, minimizing the discriminability risk R dis (θ) requires encouraging the augmented text features to align closely with those belonging to the same scenes. However, the scale and complexity of scenes often lead to substantial variability even among features corresponding to the same scenes. This introduces significant uncertainty in the selection of positive samples, complicating the risk optimization process. 3 To handle this, we adopt the corresponding 3D scene features as substitutes for the text features to construct positive pairs. This is based on the assumption that the scene features encoded by the well-trained model are more stably located near the center of the corresponding description distribution. Finally, we mitigate the aforementioned discriminability risk R dis (θ) by minimizing a negative log-based proxy loss term L dis , which could be written as follows:
L dis = - b i=1 nc j=1 y ij log S( ũi , v j ),(7)
where b is the size of the mini-batch. On the other hand, motivated by [50], we attempt to approximate the minimization of the divergence risk R div (θ) by minimizing its upper bound, i.e.,
sup (R div (θ)) ∼ E ũ-∼ Ũ -S( ũ, ũ-) ; V ũ-∼ Ũ -S( ũ, ũ-) ,(8)
where sup(•) means the upper bound and V is the variance. We can obviously see that the divergence risk is affected by the mean and variance of the selected negative samples. Yet existing methods [48] usually treat others within the same mini-batch as negative samples for contrastive learning, thereby minimizing the expectation term. Due to the stochasticity of mini-batch sampling, similar samples may be mistakenly chosen as negatives, which increases the variance of negative samples, enlarging the upper bound of the divergence risk.
To tackle it, we propose a weighted complementary contrastive loss as a surrogate objective to achieve divergence risk optimization more robustly, which can be formulated as:
L div = b i=1 b j̸ =i exp (-max (0, S( ũi , ũj ) -γ)) Weighting term log (1 -S( ũi , ũj )) Complementary contrastive term , (9
)
where γ is a threshold, above which samples are assigned lower weights. Minimizing the complementary contrastive term optimizes the expectation over negative pairs, while the weighting component can mitigate the impact of high-variance false-negative samples.
Finally, we combine both terms to obtain our loss for domain adaptation tuning, as follows:
L = λL dis + (1 -λ)L div , (10
)
where λ is a hyperparameter to control the contribution of each component. Minimizing this proxy loss facilitates the reduction of domain adaptation risk, thereby enabling the retriever to better adapt to the domain of interaction text.
Table 1: Performance comparison on ScanRefer, Nr3D, and Sr3D in terms of R@1, R@5, R@10, and their sum (Rsum). † denotes the use of coarse-grained descriptions as memory.
Methods ScanRefer Nr3D Sr3D R@1 R@5 R@10 Rsum R@1 R@5 R@10 Rsum R@1 R@5 R@10 Rsum w/ coarse-grained descriptions: VSE∞ (CVPR'21) 9.7 33.1 50.2 93.0 5.8 21.5 32.5 59.8 5.5 18.9 27.4 51.8 CHAN (CVPR'23) 9.4 32.
3 52.1 93.8 7.5 15.0 32.1 54.6 5.4 21.3 34.8 61.5 HREM (CVPR'23) 10.2 34.0 51.4 95.6 7.3 18.1 31.9 57.3 5.4 21.9 33.6 60.9 CRCL (NeurIPS'23) 10.3 32.4 49.8 92.5 8.1 22.5 33.2 63.8 4.9 19.5 31.9 56.3 RoMa (TMM'25) 11.4 34.8 54.4 100.6 6.5 24.8 37.6 68.9 7.8 27.3 39.3 74.4 IDeal 16.0 42.7 59.8 118.5 11.7 34.8 50.4 96.9 10.3 30.2 48.5 89.0 w/ fine-grained descriptions: ChatIR † (NeurIPS'23) 21.8 55.4 73.1 150.3 15.6 40.3 58.1 114.0 12.1 35.9 50.3 98.3 Rewrite † (ICMR'24) 17.4 47.0 63.7 128.1 17.1 31.4 45.8 94.3 12.4 28.1 40.4 80.9 MERLIN † (EMNLP'24) 31.1 68.8 83.8 183.7 21.8 55.0 71.8 148.6 14.2 42.0 60.3 116.5 BASELINE: IR † 29.9 68.0 83.3 181.2 18.0 51.6 60.2 129.8 14.6 39.2 55.3 109.1 BASELINE SUM † 34.4 69.5 85.1 189.0 22.5 55.2 67.5 145.2 16.4 41.5 62.1 120.0 IDeal † 37.8 71.8 86.4 196.0 26.4 62.7 78.7 167.8 20.2 43.1 63.1 126.4 4 Experiments 4.1 Experimental Setting Datasets, baselines, and evaluation metrics: We adopt the ScanNet 3D scene set along with several description sets (i.e., ScanRefer [51], Nr3D [52], Sr3D [52], and SceneDepict-3D2T [6]) to conduct experiments, where ScanRefer, Nr3D, and Sr3D are employed as query sets, and SceneDepict-3D2T is employed to simulate fine-grained memory. To verify the superiority of our IDeal, we introduce eleven comparative baseline methods: five conventional offline cross-modal matching methods (i.e., VSE∞ [45], CHAN [53], HREM [54], CRCL [37], and RoMa [6]), three interactive cross-modal retrieval methods (i.e., ChatIR [8], Rewrite [41], and MERLIN [16]), and two additional strong interactive baselines (IR and SUM). More specifically, the Iterative Reranking (IR) involves multiround interaction, where the results are iteratively re-ranked based on the response of each round. The Summary reranking (SUM) also involves interaction, but ultimately aggregates all answers into a comprehensive description for matching.
Table 2: Performance comparison on ScanRefer and Nr3D in terms of R@1, R@5, R@10, and their sum. +IDeal indicates plugging the model into our IDeal. † denotes the use of fine-grained descriptions as memory.
Methods ScanRefer Nr3D R@1 R@5 R@10 Rsum R@1 R@5 R@10 Rsum VSE∞ 9.7 33.1 50.2 93.0 5.8 21.5 32.5 59.8 +IDeal 13.3 38.9 57.6 109.8 8.7 27.5 42.1 78.3 VSE∞ † 14.9 42.3 61.5 118.7 16.4 47.5 55.2 119.1 +IDeal † 35.8 70.6 85.0 191.4 21.2 52.1 68.4 141.7 CRCL 10.3 32.4 49.8 92.5 8.1 22.5 33.2 63.8 +IDeal 13.4 35.5 56.1 105.0 7.4 25.4 38.3 71.1 CRCL † 17.5 45.1 58.3 120.9 13.4 44.5 51.5 109.4 +IDeal † 31.7 66.9 83.5 182.1 15.8 50.4 64.4 130.6 RoMa 9.7 33.1 50.2 93.0 8.3 27.9 37.2 73.4 +IDeal 16.0 42.7 59.8 118.5 11.7 34.8 50.4 96.9 RoMa † 16.7 44.8 61.6 123.1 17.4 48.5 57.5 123.4 +IDeal † 37.8 71.8 86.4 196.0 25.4 60.7 75.7 161.8
In addition, we follow [55,56] to report R@1, R@5, R@10, and their summation (Rsum) as the evaluation metrics. Due to the space limitation, more details of datasets, prompts, and additional experiments are provided in the Supplemental Material.
Implementation details: All methods are implemented in PyTorch and carried out on GeForce RTX 3090 GPUs. We adhere to the experimental settings of [6] for all method implementations. We adopt widely-used DGCNN [57] and BERT [58] to obtain fine-grained features for 3D point clouds and texts, respectively. To implement interaction, we explore two approaches to constructing memory: 1) Coarse-grained description: We leverage an LLM to generate rich expansions of queries, serving as memory without introducing any additional information leakage. 2) Fine-grained description: In line with existing interactive methods [43,13], we simulate the user's memory in real-world scenarios using fine-grained scene descriptions, albeit with access to partial additional information.
In our experiments, we utilize Qwen-7B-Instruct [59] as our investigated LLM for the interaction experiments.
this section cite: ['b47', 'b48', 'b49', 'b47', 'b50', 'b51', 'b51', 'b5', 'b44', 'b52', 'b53', 'b36', 'b5', 'b7', 'b40', 'b15', 'b54', 'b55', 'b5', 'b56', 'b57', 'b42', 'b12', 'b58']

Section: Comparison on Text-3D Scene Retrieval
Table 1 presents a comparison between our IDeal and conventional and interactive cross-modal matching methods under two memory settings. Table 2 further demonstrates the performance gains brought by integrating our interactive framework into conventional single-round methods. These results could yield the following observations: 1) Even without additional fine-grained information, our IDeal achieves competitive performance, highlighting its ability to uncover complementary information implicitly embedded in the queries, gradually alleviating inherent query limitations. 2) Compared to existing interactive methods, our IDeal also achieves superior performance with access to fine-grained descriptions. This demonstrates that our interactive questioning and retrieval strategies enable an ongoing understanding of user requirements and support a comprehensive interpretation of complex scenes. 3) Our IDeal can be seamlessly integrated into conventional cross-modal retrieval methods and yields substantial performance gains under both memory settings. This suggests that, beyond equipping offline models with interactive capabilities, IDeal empowers them to more effectively comprehend complex descriptions through interaction. In this section, we conduct an ablation study to evaluate the contribution of each proposed component to our IDeal. Specifically, we first ablate the router in the questioner, restricting it to ask either Q 1 or Q 2 continuously. In addition, we remove each of the three retrieval prediction strategies, and we examine removing CoT prompting in the reconstruction in the proposed retriever. Finally, we investigate the effect of not using the IAT strategy for domain adaptation and sequentially ablate its two loss terms. The results in Table 3 lead to the following observation: 1) Removing or replacing any component from IDeal results in performance degradation, highlighting the contribution of each component. Specifically, the adaptive questions generated by our questioner facilitate a meticulous and comprehensive understanding of scenes. The various feature aggregation strategies in the retriever contribute to precise scene matching. 2) Removing or substituting IAT components consistently leads to performance degradation, underscoring the necessity of text domain alignment and adaptation risk minimization in our IAT.
this section cite: []

Section: Ablation Study

this section cite: []

Section: Parameter Analysis
To evaluate the sensitivity of our IDeal to different hyperparameter settings, we plot the retrieval performance versus different values on ScanRefer, as shown in Figure 3. The experimental results lead to the following observation: 1) For our retriever, tuning greater weights to interactive and reconstruction predictions helps achieve a well-balanced trade-off that fully leverages the interactive responses. Additionally, a higher feature fusion weight (e.g., α = 0.75) represents a emphasis on the integration of discriminative features from the refined descriptions, leading to more effective interaction feature fusion. 2) For our proposed questioner, using reasonable and moderate settings of k and β (e.g., k = 20, β = 2.0) enables accurate identification of informative descriptions, thereby supporting reasonable decisions on question types in the next round.
3) During domain adaptation tuning, a relatively wide range of λ and γ values in IAT (i.e., λ ∈ [0.2, 0.5] and γ ∈ [0.1, 0.8]) ensures effective contrastive adaptation and mitigates the impact of false negatives. Case 3 Initial Query Ground Truth Ours Merlin Round 1 Answer Round 2 Answer Round 5 Answer Round 1 Answer Round 2 Answer Round 4 Answer there is a large painting on the wall . there is a chair at the end of the table close to the painting. we are talking about the other chair that is close to the trash can to the right side of the painting . it is brown and wooden . [Rank 4] this is a description of the scene where a brown wooden chair is near the gray trash can to the right of the painting. it is also mentioned that there is a gray trash can to the right of the table. there is a brown chair at the end of the table as well. additionally, there are several matching chairs with brown leather surrounding the table… … there is a rectangular refrigerator, Next to fridge, there is a black square TV… … the brown wooden chair is near the trash can to the right of the painting, with some empty space between them [Rank 2] [Rank 1] [Rank 1] The large painting on the wall appears to be representational rather than abstract, as it depicts a realistic scene. Besides the table and chairs already described... The color scheme of the room seems to consist mainly of neutral tones such as browns, grays, and blacks. [Rank 5] The second chair … painting… first chair. Table… round trashcan… Round 5 Answer The round trashcan… table. the second chair … The round trashcan. table… the surrounding of second chair … [Rank 4] [Rank 4] [Rank 4] Case 1 Case 2 Initial Query Round 1 Answer Initial Query Round 2 Answer this is a computer monitor with two others to the left and right. it is surrounded by additional monitors in front and diagonally. there are screens placed side by side providing ample workspace. there seem to be more screens angled forward to offer better visibility … this is a computer monitor with two others to the left and right. there appears to be other monitors also in front of it and diagonally. [Rank 16] [Rank 4] this is a long brown/plain table. it is situated near a single chair which is very close to the wall. the single chair is placed right by the wall. there is no indication of any windows or doors being adjacent to the table or chair. the lengthy beige unadorned surface (table) stands alongside the solitary stool. a long brown table. it is located near a single chair which is very close to the wall.
this section cite: []

Section: Visualization Analysis
To provide a comprehensive analysis of IDeal, we conduct a series of visualization experiments. Specifically, we first present the changes in retrieval performance of IDeal across multiple rounds of interaction, as shown in Figure 4, to analyze the incremental gains brought by each interaction. In addition, we visualize several representative cases, as illustrated in Figure 5. The observations can be drawn from the results: 1) Interaction consistently improves performance over the first several rounds (five rounds with fine-grained and three rounds with coarse-grained descriptions). Although redundant interactions may inevitably cause performance to saturate or even slightly degrade due to LLM hallucinations or excessively long texts, these results indicate that a reasonable number of interaction steps can effectively enhance the query and improve retrieval performance. 2) Under the setting with coarse-grained memory texts, IDeal can infer and decompose object attributes and relationships within the queries through interaction, leading to improved retrieval performance. Moreover, with the integration of fine-grained memory, IDeal leverages targeted and switchable questioning to elicit informative responses, continually improving retrieval precision. In contrast, MERLIN [16] frequently generates redundant descriptions confined to local details of scenes.
this section cite: ['b15']

Section: Conclusion
In this paper, we propose a novel Interactive Text-3D Scene Retrieval Method, namely IDeal, to address the Text-3D Scene Retrieval (T3SR). Our IDeal integrates two components: the Interactive Retrieval Refinement Framework (IRR) and the Interaction Adaptation Tuning strategy (IAT). Specifically, IRR continuously conducts adaptive questioning and comprehensive response fusion, enabling holistic exploration of 3D scenes for more precise retrieval. IAT performs contrastive domain adaptation for the retriever toward realistic texts, overcoming the performance bottleneck during interaction. Extensive experiments demonstrate the superiority of our IDeal in T3SR task.
this section cite: []

Section: Limitations and Potential Impact Statement:
Although our work has taken the initial step forward in interactive T3SR, there are some limitations and potential impacts that should be acknowledged. First, the performance of the methods is relatively low. Second, we employ LLMs, and more stable and unbiased LLMs and interaction approaches merit further exploration in the future.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction clearly state the claims made.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: The limitations are discussed in the Conclusion Section.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: The Analysis and proof are provided in the supplementary material.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The implementation and training details are clearly described for reproduction in our main paper and supplementary material.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code will be released publicly after in-peer review.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: The experiment settings are clearly presented in the paper and supplementary material.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: Error bars are not reported because it would be too computationally expensive for experiments involving LLMs.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: The compute resources are reported in the experiment settings.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: The research conducted in the paper conform with the NeurIPS Code of Ethics in every respect.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: The broader impacts are discussed with the limitations.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: All datasets and models used in this paper are publicly available.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: Proper citations are provided throughout the document and the licenses will be included with the code when it is released.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: The document will accompany the code upon its release.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This work does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [Yes] Justification: We have fully disclosed the details of the use of the adopted LLMs in our supplementary material. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: A 3d vision-language-action generative world model Year: (2024)
Ref_id:b1 Title: 3d-vista: Pre-trained transformer for 3d vision and text alignment Year: (2023)
Ref_id:b2 Title: Sceneverse: Scaling 3d vision-language learning for grounded scene understanding Year: (2024)
Ref_id:b3 Title: 360loc: A dataset and benchmark for omnidirectional visual localization with cross-device queries Year: (2024)
Ref_id:b4 Title: An embodied generalist agent in 3d world Year: (2023)
Ref_id:b5 Title: Pointcloud-text matching: Benchmark dataset and baseline Year: (2025)
Ref_id:b6 Title: where am i?" scene retrieval with language Year: (2024)
Ref_id:b7 Title: Chatting makes perfect: Chatbased image retrieval Year: (2023)
Ref_id:b8 Title: Test-time adaptation for cross-modal retrieval with query shift Year: (2024)
Ref_id:b9 Title: Image clustering with external guidance Year: (2024)
Ref_id:b10 Title: Simple baselines for interactive video retrieval with questions and answers Year: (2023)
Ref_id:b11 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b12 Title: Interactive text-toimage retrieval with large language models: A plug-and-play approach Year: (2024)
Ref_id:b13 Title: Learning to rematch mismatched pairs for robust cross-modal retrieval Year: (2024)
Ref_id:b14 Title: Learning to retrieve videos by asking questions Year: (2022)
Ref_id:b15 Title: Merlin: Multimodal embedding refinement via llm-based iterative navigation for text-video retrieval-rerank pipeline Year: (2024)
Ref_id:b16 Title: C3cmr: Cross-modality cross-instance contrastive learning for cross-media retrieval Year: (2022)
Ref_id:b17 Title: Deep cross-modal hashing with ranking learning for noisy labels Year: (2024)
Ref_id:b18 Title: Ugncl: Uncertainty-guided noisy correspondence learning for efficient cross-modal matching Year: (2024)
Ref_id:b19 Title: Dica: Disambiguated contrastive alignment for cross-modal retrieval with partial labels Year: (2025)
Ref_id:b20 Title: She: Streaming-media hashing retrieval Year: ()
Ref_id:b21 Title: Dual selfpaced cross-modal hashing Year: (2024)
Ref_id:b22 Title: Composing object relations and attributes for image-text matching Year: (2024)
Ref_id:b23 Title: Deep evidential learning with noisy correspondence for cross-modal retrieval Year: (2022)
Ref_id:b24 Title: Robust self-paced hashing for cross-modal retrieval with noisy labels Year: (2025)
Ref_id:b25 Title: Text-adaptive multiple visual prototype matching for video-text retrieval Year: (2022)
Ref_id:b26 Title: Xiuqiang He, and Yueting Zhuang. Hierarchical cross-modal graph consistency learning for video-text retrieval Year: (2021)
Ref_id:b27 Title: Rono: robust discriminative learning with noisy labels for 2d-3d cross-modal retrieval Year: (2023)
Ref_id:b28 Title: Robust unsupervised multimodal learning with noisy pseudo labels Year: (2024)
Ref_id:b29 Title: Esa: External space attention aggregation for image-text retrieval Year: (2023)
Ref_id:b30 Title: Multi-view visual semantic embedding Year: (2022)
Ref_id:b31 Title: Hierarchical consensus hashing for cross-modal retrieval Year: (2023)
Ref_id:b32 Title: Similarity reasoning and filtration for image-text matching Year: (2021)
Ref_id:b33 Title: Improving cross-modal retrieval with set of diverse embeddings Year: (2023)
Ref_id:b34 Title: Multi-view transformer for 3d visual grounding Year: (2022)
Ref_id:b35 Title: Gs-cpr: Efficient camera pose refinement via 3d gaussian splatting Year: (2024)
Ref_id:b36 Title: Crossmodal active complementary learning with self-refining correspondence Year: (2024)
Ref_id:b37 Title: Revisiting relevance feedback for clip-based interactive image retrieval Year: (2024)
Ref_id:b38 Title: Interactive deep clustering via value mining Year: (2025)
Ref_id:b39 Title: Large language models are zero-shot rankers for recommender systems Year: (2024)
Ref_id:b40 Title: Enhancing interactive image retrieval with query rewriting using large language models and vision language models Year: (2024)
Ref_id:b41 Title: Human-centered interactive learning via mllms for text-to-image person re-identification Year: (2025)
Ref_id:b42 Title: Llava-reid: Selective multi-image questioner for interactive person re-identification Year: (2025)
Ref_id:b43 Title: Density biased sampling: An improved method for data mining and clustering Year: (2000)
Ref_id:b44 Title: Learning the best pooling strategy for visual semantic embedding Year: (2021)
Ref_id:b45 Title: Support vector data description Year: (2004)
Ref_id:b46 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b47 Title: Attracting and dispersing: A simple approach for source-free domain adaptation Year: (2022)
Ref_id:b48 Title: Model adaptation: Historical contrastive learning for unsupervised domain adaptation without source data Year: (2021)
Ref_id:b49 Title: Revisiting source-free domain adaptation: a new perspective via uncertainty control Year: ()
Ref_id:b50 Title: Scanrefer: 3d object localization in rgb-d scans using natural language Year: (2020)
Ref_id:b51 Title: Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes Year: (2020)
Ref_id:b52 Title: Fine-grained image-text matching by crossmodal hard aligning network Year: (2023)
Ref_id:b53 Title: Learning semantic relationship among instances for image-text matching Year: (2023-06)
Ref_id:b54 Title: Stacked cross attention for image-text matching Year: (2018)
Ref_id:b55 Title: Learning with noisy correspondence for cross-modal matching Year: (2021)
Ref_id:b56 Title: Dynamic graph cnn for learning on point clouds Year: (2019)
Ref_id:b57 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b58 Title: Qwen technical report Year: (2023)
