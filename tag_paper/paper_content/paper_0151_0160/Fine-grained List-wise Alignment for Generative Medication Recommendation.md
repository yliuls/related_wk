Title: Fine-grained List-wise Alignment for Generative Medication Recommendation
Abstract: Accurate and safe medication recommendations are critical for effective clinical decision-making, especially in multimorbidity cases. However, existing systems rely on point-wise prediction paradigms that overlook synergistic drug effects and potential adverse drug-drug interactions (DDIs). We propose FLAME, a finegrained list-wise alignment framework for large language models (LLMs), enabling drug-by-drug generation of drug lists. FLAME formulates recommendation as a sequential decision process, where each step adds or removes a single drug. To provide fine-grained learning signals, we devise step-wise Group Relative Policy Optimization (GRPO) with potential-based reward shaping, which explicitly models DDIs and optimizes the contribution of each drug to the overall prescription. Furthermore, FLAME enhances patient modeling by integrating structured clinical knowledge and collaborative information into the representation space of LLMs. Experiments on benchmark datasets demonstrate that FLAME achieves state-ofthe-art performance, delivering superior accuracy, controllable safety-accuracy trade-offs, and strong generalization across diverse clinical scenarios. Our code is available at https://github.com/cxfann/Flame.

Section: Introduction
Accurate and safe medication recommendation is essential for clinical decision-making, especially in complex cases involving multimorbidity [1,2]. In these scenarios, clinicians must consider not only the therapeutic effects of individual drugs, but also their potential interactions and cumulative safety risks. Recent advances in AI have led to the development of automated medication recommendation systems [3,4,5], yet their effectiveness remains limited in practice.
Traditional approaches typically adopt a point-wise prediction paradigm, where each drug is evaluated independently based on structured patient data such as diagnoses, procedures, and drug codes [6,7]. While longitudinal models [8] attempt to capture patient history, they are still constrained by their reliance on discrete labels and lack the capacity to represent unstructured clinical context. More recently, large language models (LLMs) have emerged as a promising solution, due to their strong language understanding and ability to incorporate free-text information such as clinical notes [9,10]. However, most LLM-based methods still follow the point-wise formulation, constructing the final drug set by aggregating drugs with high predicted scores. This formulation introduces a fundamental limitation: it overlooks the synergistic effects and safety constraints that exist between drugs. In practice, effective prescriptions must balance therapeutic efficacy with the risk of adverse drug-drug interactions (DDIs) [11]. Point-wise models, by design,
-0.38 +0.40 +1.35 -1.37 � 1 � 2 � 3 � 1 � 4 � 2 � 3 � 4 � 5 � 1 � 3 � 6 � 1 � 2 � 3 � 1 � 4 � 2 � 3 � 4 � 5 � 1 � 3 � 6 +0.7 +0.7 -0.3 +0.7 -0.3 +0.8 +0.2 +0.2 +0.2 +0.8 -0.8 +0.7 � 1 � 2 � 3 � 1 � 4 � 2 � 3 � 4 � 5 � 1 � 3 � 6 -0.38 +0.40 +1.35 -1.37 Outcome-based advantages Evaluate each step Step-wise advantages Lis t-wis e ad van tag es (a) Same advantage for all tokens? (b) Step-wise signals, finer advantages！ cannot account for this interplay. To address this, we argue that medication recommendation should be viewed as a list-wise decision-making problem, where the goal is to generate a coherent set of drugs that jointly optimize accuracy and safety.
Existing list-wise methods in NLP often rely on reinforcement learning (RL) techniques to align model outputs with desired properties. One promising approach is Group Relative Policy Optimization (GRPO) [12], which updates model preferences based on relative advantages among a group of candidate outputs. Yet standard GRPO operates at the sequence level (Fig. 1 (a))-assigning a single scalar reward to the entire output-making it difficult to assign credit to individual actions.
To this end, we propose FLAME (Fine-grained List-wise Alignment for generative Medication rEcommendation), an LLM-based framework that formulates drug generation as a drug-by-drug decision process. At its core is a novel extension of GRPO-step-wise GRPO-which models the generation as a sequence of state transitions. Each step adds or removes a drug, and reward shaping is applied via a potential function to provide token-level feedback (Fig. 1 (b)) throughout the generation process. This structure enables FLAME to learn nuanced and controllable decision policies.
To support comprehensive patient modeling, FLAME incorporates multi-source medical knowledge into the LLM via hybrid representations. In addition to the LLM's natural language inputs, we inject structured clinical features (e.g., codes, embeddings from prior models) into the token space, enabling the model to capture both textual context and collaborative signals. We build on Llama3.1-Aloe-Beta-8B [13], a domain-specific LLM with enriched medical knowledge.
We evaluate FLAME on benchmark datasets including MIMIC-III [14], MIMIC-IV [15], and eICU [16]. Experimental results show that FLAME achieves state-of-the-art accuracy while maintaining controllable safety trade-offs. Moreover, FLAME exhibits strong generalization across time and institutions, validating its adaptability to real-world clinical scenarios.
Our contributions are summarized as follows:
• We propose FLAME, an LLM-based list-wise medication recommendation framework that generates prescriptions drug-by-drug, explicitly modeling drug interactions and integrating multi-source medical knowledge via hybrid representations.
• We introduce step-wise GRPO, which models medication recommendation as a sequential state transition process and leverages potential-based reward shaping to provide fine-grained feedback for each drug-level decision, enabling controllable and clinically safer recommendations.
• Extensive experiments on benchmark datasets demonstrate that FLAME achieves state-of-the-art performance in accuracy, safety-accuracy trade-offs, and cross-dataset generalization, validating its effectiveness in diverse clinical settings.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15']

Section: Related Work
In this section, we provide a brief review of medication recommender systems and LLM fine-tuning methods, highlighting their respective strengths and limitations.
this section cite: []

Section: Medication Recommendation

this section cite: []

Section: Existing medication recommendation methods can be broadly categorized into instance-based and longitudinal approaches, based on how patient information is modeled.
Instance-based methods, such as LEAP [6], treat the medication recommendation task as a multiinstance multi-label learning problem. These approaches typically rely on structured features extracted from a single patient visit. In contrast, longitudinal methods incorporate temporal information from patients' hospitalization histories to model long-term disease progression and treatment trajectories. For example, GameNet [7] utilizes hospitalization records and a graph augmented memory module to model DDIs. SafeDrug [3] enhances this by using dual molecular graph encoders to capture drug structural information. COGNet [4] introduces a copy-or-predict mechanism for medication recommendation from historical data, while MoleRec [8] focuses on the relationship between health status and molecular substructures. RAREMed [5] uses a pretrain-finetune framework for structured representation extraction, and NLA-MMR [17] applies a multimodal alignment framework to jointly learn from patient and drug views.
Recently, the development of LLMs has brought new possibilities for medication recommendation. Instead of relying solely on structured codes, these approaches leverage natural language to represent patient conditions more comprehensively. For example, LAMO [10] integrates structured diagnoses and procedures with unstructured textual descriptions of patients, enabling LLMs to model clinical context in a semantically rich manner and generate more personalized drug suggestions.
Many existing methods, including recent LLM-based approaches, still follow the point-wise prediction paradigm [18,19,20], assigning independent scores or binary labels to individual drugs while ignoring inter-drug dependencies. In contrast, we formulate medication recommendation as a listwise decision process and introduce step-wise GRPO for fine-grained reward modeling, integrating medical knowledge, collaborative signals, and LLM-based semantic understanding.
this section cite: ['b5', 'b6', 'b2', 'b3', 'b7', 'b4', 'b16', 'b9', 'b17', 'b18', 'b19']

Section: LLM Fine-Tuning Approaches
Fine-tuning LLMs for complex decision tasks has evolved from supervised fine-tuning (SFT) to preference-based reinforcement learning approaches such as reinforcement learning-based finetuning (RLHF). While RLHF methods like Proximal Policy Optimization (PPO) [21] offer online adaptability, they incur high overhead due to their multi-component design. Recent advances such as Direct Preference Optimization (DPO) [22] reduce this complexity by leveraging static preference data, but struggle with dynamic optimization.
GRPO [12] further improves training efficiency and policy stability by introducing group-wise relative preference comparisons, achieving strong performance in structured reasoning tasks. However, GRPO remains outcome-based-assigning rewards only to complete outputs-thus failing to capture steplevel quality variations within structured decision processes.
Several recent efforts have sought to improve GRPO. DAPO [23] introduces token-level policygradient losses to mitigate length bias, ensuring each token contributes equally; however, it still adopts a list-wise advantage formulation, lacking localized reward assignment. StepGRPO [24] decomposes responses into sequential actions and supplements outcome-based evaluation with steplevel accuracy and validity rewards. Yet, it continues to aggregate advantages at the list level and often depends on ground-truth process data. While these works enhance signal fidelity and stability, they do not resolve the fine-grained reward allocation problem we address.
On the contrary, step-wise GRPO is a fine-grained alignment method that decomposes generation into decision steps and applies a potential-based reward mechanism to guide each sub-decision. This enables LLMs to better learn the internal logic of complex tasks such as medication recommendation.
this section cite: ['b20', 'b21', 'b11', 'b22', 'b23']

Section: Preliminary
This section formulates the medication recommendation problem through three components: electronic health record, DDI graph, and the optimization goal. We then introduce Group Relative Policy Optimization, a reinforcement learning framework that models outcome-based advantages.
this section cite: []

Section: Medication Recommendation
Electronic Health Records (EHRs). Each patient's EHR [25] contains both structured and unstructured clinical information and is represented as a sequence of multivariate visits. For a patient j with V historical visits, the EHR is denoted as
X (j) V = [x (j) 1 , x (j) 2 , . . . , x (j) V ]. Each visit x (j) v ∈ X (j) V
consists of the following components:
x (j) v = [(f (j) v , n (j) v ), M(j)
v ]. where f (j) v ∈ {0, 1} |F | is a multi-hot vector encoding structured features from the set F (e.g., demographics, diagnoses, and procedures), and n (j) v is the corresponding unstructured clinical note in natural language. The set M (j) v denotes the medications prescribed during the v-th visit. For simplicity, we omit the patient index j when it is clear from context.
this section cite: ['b24']

Section: Drug-Drug Interaction (DDI) Graph.
The DDI graph [11] models harmful pairwise interactions between medications and is represented as a symmetric binary adjacency matrix D ∈ {0, 1} |M|×|M| , where D ij = 1 indicates a known adverse interaction between medications m i and m j .
Task Definition. Given a patient's historical visit records X V -1 , as well as the structured fields and unstructured note of the current visit (f V , n V ), the objective is to generate an output o V , representing the recommended medication set M V , such that: (1) it closely approximates the ground-truth prescription M GT ; and (2) it minimizes the risk of harmful drug-drug interactions as defined by the DDI graph D.
this section cite: ['b10']

Section: Group Relative Policy Optimization (GRPO)
GRPO [12] is a reinforcement learning algorithm that improves policies via group-wise reward normalization. Given a prompt q ∼ P (Q) from the task distribution, GRPO samples a group of G candidate outputs {o i } G i=1 from the current policy π θ old . Each output is assigned a scalar reward, and the updated policy π θ is optimized to favor relatively better responses within the group.
The training objective (omitting clipping for brevity) is:
J GRP O (θ) = E q,{oi}   1 G G i=1 1 |o i | |oi| t=1 π θ (o i,t |o i,<t , q) π θ old (o i,t |o i,<t , q) Âi,t -βD KL [π θ ∥π ref ]   , (1
)
where β controls the KL divergence regularization toward a reference policy π ref .
The token-level advantage Âi,t is uniformly derived from the normalized reward of the full response:
Âi,t = ri = r i -mean(r) std(r) ,(2)
where r i is the scalar reward for the i-th output, r = [r 1 , . . . , r G ] is the reward vector for the group.
Although GRPO effectively promotes group-wise preference learning, its outcome-level reward is applied uniformly across all tokens, ignoring variation in token-level quality. This coarse credit assignment hinders fine-grained learning and limits optimization efficiency.
this section cite: ['b11']

Section: Method: FLAME
We first introduce step-wise GRPO, the core of FLAME's fine-grained alignment. We then describe the optimization process, followed by the two-stage recommendation framework. Finally, we present the multi-source knowledge fusion strategy.
this section cite: []

Section: Step-wise GRPO
Output Segmentation. To enable fine-grained credit assignment, we decompose the LLM output into multiple decision steps [26], each corresponding to a semantically coherent token span (e.g., a single medication). Given a generated sequence o i = [o 1 i , o 2 i , . . . , o
o i = Ni n=1 o (n) i , where o (n) i = [o bn i , . . . , o en i ].(3)
t Step(t) 0 1 2 3 4 5 6 7 8   1 1 1 2 2 2 2 33 … 𝒎 𝟏 𝒎 𝟐 𝒎 𝟑 𝒎 𝟏 𝒎 𝟐 𝒎 𝟑 … 𝒎 𝟏 𝒎 𝟐 𝒎 𝟑 ∆ ∆ ∆ መ 𝐴 𝑖,0 መ 𝐴 𝑖,1 መ 𝐴 𝑖,2 መ 𝐴 𝑖,3 መ 𝐴 𝑖,4 መ 𝐴 𝑖,5 መ 𝐴 𝑖,6 መ 𝐴 𝑖,7 መ 𝐴 𝑖,8 Each o (n) i denotes the token span of the n-th step. The process is illustrated in (Fig. 2). This segmentation allows us to reinterpret generation as a Markov Decision Process, where each step forms a state-action pair. This framing enables localized reward signals that better align step-wise actions with the global output quality.
Potential-based Reward Shaping. We extend the standard GRPO advantage (Eq. 2) with a shaping term that captures quality changes between consecutive steps:
Âi,t = ri + λ • F (step(t), step(t) -1)(4)
Here, step(t) maps token t to its step index, and F (•, •) measures step-wise potential difference. λ is a weighting coefficient. To preserve policy invariance [27], we define F as the difference of a real-valued potential function φ:
F (n, n -1) = γ • φ(n) -φ(n -1)(5)
with γ = 1, assuming equal importance across steps.
this section cite: ['b25', 'b26']

Section: Final Advantage.
The final advantage used for policy optimization becomes:
Âi,t = ri + λ (φ(step(t)) -φ(step(t) -1))(6)
This formulation delivers dense training signals aligned with intermediate decision quality [28], encouraging token-level improvements toward globally effective prescriptions.
this section cite: ['b27']

Section: Fine-grained Alignment for Medication Recommendation
We apply step-wise GRPO to our list-wise medication recommendation task by treating each decision step as a span representing a single medication. We define:
• State s n : the patient profile and current medication set M n .
• Action a n : adding or removing a medication m n .
The state transition follows:
M n+1 = M n ∪ {m n }, if adding, M n \ {m n }, if removing. (7
)
To evaluate the quality of intermediate states, we define a step-level potential function φ(step(t)) incorporating: (1) Correctness: Jaccard similarity to the ground-truth set M GT , (2) Adherence: constraint violation with respect to the candidate set M C (all unique medications in the processed training data), and (3) Safety: DDI-based risk from the interaction graph D. The potential function is given by:
φ(step(t)) = Jaccard(M n , M GT ) -α • DDI(M n , D) -β • RefusalRate(M n , M C ) (8
)
where: This potential function φ(step(t)) allows computing local advantage via step-wise differences (Eq. 6), assigning credit based on incremental improvement in the medication set M n . The overall list-level reward is defined as the net potential change from the initial to the final step: R i = φ(N i ) -φ(0) (9) Theorem 4.1 (Optimal Policy Equivalence under Reward Reshaping).
Jaccard(M n , M GT ) = |M n ∩ M GT | |M n ∪ M GT | , RefusalRate(M n , M C ) = |{m ∈ M n | m / ∈ M C }| |M n | . DDI(M n , D) = |{(m i , m j ) | m i , m j ∈ M n , i < j, D[m i , m j ] = 1}|
Let the token-level reward for each token t in the generated output o i be defined under the following two schemes:
(1) Outcome-based reward: r i,t = R i , if t is the terminal token, 0, otherwise
(2
) Step-wise shaped reward: r ′ i,t = φ(step(t)) -φ(step(t) -1)(11)
Then, both reward formulations yield the same optimal policy.
this section cite: []

Section: The proof is provided in Appendix A.
Using the potential-shaped advantage Âi,t in Eq. ( 6), we optimize the policy via step-wise GRPO, while retaining the same objective form as standard GRPO (Eq. ( 1)).
this section cite: []

Section: Two-stage Recommendation Framework
We adopt a cascaded framework to generate an accurate and safe medication set M v for a patient's v-th visit. The process consists of two stages: drug-level filtering and list-level refinement. The former is implemented via a binary classification model, and the latter through a policy fine-tuned with step-wise GRPO, as illustrated in Fig. 3.
this section cite: []

Section: Drug-level Classifier (π cls ).
We implement π cls as an LLM-based binary classifier, obtained by SFT the Llama3.1-Aloe-Beta-8B [13] base model, that evaluates the relevance of each candidate drug m ∈ M c given the patient input x. The model returns a Yes/No decision, producing a personalized subset:
M p = {m ∈ M c | π cls (x, m) = Yes} .(12)
This step establishes personalized relevance by filtering drugs based on individual compatibility prior to joint reasoning.
this section cite: ['b12']

Section: List-wise Policy (π list ).
To support global optimization, we introduce a list-wise policy π list that performs instruction-conditioned edits over M p . Given patient input x, a medication list M, and an instruction (Add Drug or Remove Drug), the model predicts a modification set:
∆M = π list (x, M, Instruction). (13) We initialize π list with π cls , leveraging its drug-patient matching ability, and further adapt it to instruction-driven editing via supervised fine-tuning and step-wise GRPO (see Section 4.1).
Inference. At inference, we first apply π cls to obtain M p using Eq. ( 12), then perform list-level edits:
∆M add = π list (x, M p , Add Drug), ∆M remove = π list (x, M p , Remove Drug).(14)
The final recommendation is obtained by applying the edits:
M v = (M p ∪ ∆M add ) \ ∆M remove .(15)
This two-stage procedure combines individualized assessment with global reasoning for more controllable and context-aware recommendations. Implementation details, including prompt templates and training setup, are provided in Appendix C.
this section cite: []

Section: Multi-source Knowledge Fusion
Traditional methods often encode structured inputs (e.g., diagnoses, procedures) into embeddings and integrate external knowledge via architectural modules such as graph encoders [3,7]. However, LLMs operate primarily on unstructured text, making the incorporation of structured signals non-trivial.
We propose a fusion mechanism that injects collaborative signals from structured sources into LLMs [29,30], as illustrated in Fig. 4. For each structured entity, we construct two complementary representations: a textual embedding e text ∈ R d from natural language descriptions, and a collaborative embedding e collab ∈ R d ′ from a domain-specific encoder. The collaborative embedding is projected into the LLM space via a learnable linear map P : R d ′ → R d , yielding:
e fused = Concat(e text , P(e collab ))(16)
this section cite: ['b2', 'b6', 'b28', 'b29']

Section: Experiments
We begin by detailing the experimental setup. We then evaluate FLAME against baselines across three dimensions: accuracy, safety-accuracy trade-offs, and cross-dataset generalization. Finally, an ablation study assesses the contribution of each component in our framework.
this section cite: []

Section: Experimental Settings
Datasets. We use real-world EHR datasets: MIMIC-III [14] for training and evaluation, and MIMIC-IV [15] and eICU [16] for generalization testing. DDI relations are obtained from TWOSIDES [11]. Following prior works, MIMIC-III is split into training/validation/test sets (4:1:1).
this section cite: ['b13', 'b14', 'b15', 'b10']

Section: Implementation.
All methods are implemented in PyTorch and trained on NVIDIA A100 GPUs. Details on preprocessing, hyperparameters, and prompts are in Appendices B and C.
Baselines. We compare FLAME with: (1) Traditional methods: LEAP [6], GAMENet [7], Safe-Drug [3], COGNet [4], MICRON [31], MoleRec [8], NLA-MMR [17], RAREMed [5]; (2) LLMbased method: LAMO [10].
this section cite: ['b5', 'b6', 'b2', 'b3', 'b30', 'b7', 'b16', 'b4', 'b9']

Section: Metrics.
Correctness is evaluated by Jaccard similarity and F1 score; safety by DDI rate. Definitions and calculation details are in Appendix C.
this section cite: []

Section: Overall Performance Comparison
We evaluate FLAME from three perspectives: correctness, safety controllability, and generalizability. EHR datasets contain inherent DDI risks (e.g., 13.69% in MIMIC-III), meaning data fitting does not guarantee safe prescriptions. Minimizing DDIs often compromises accuracy, making the correctness-safety trade-off a key metric for practical utility.
Unlike prior works reporting a single performance point, we explicitly decouple correctness and safety in evaluation, enabling a clearer assessment of a model's reasoning ability and controllability under varying safety constraints. To assess generalization, we further test on MIMIC-IV and eICU, examining performance under distribution shifts beyond the training domain.
this section cite: []

Section: Correctness Comparison.

this section cite: []

Section: Generalization Comparison.
We assess generalization through temporal and external validations. For temporal validation, models trained on MIMIC-III are evaluated on MIMIC-IV, which covers newer time periods and ICD-10 codes. Fig. 5(b) shows that LLM-based models (FLAME, LAMO) degrade less over time compared to structured-data baselines, highlighting the robustness of language-based representations to coding shifts. FLAME further surpasses LAMO due to its integration of multi-source knowledge and step-wise reasoning.
For external validation, we evaluate on eICU, a multi-center dataset with distributional and coding differences. As shown in Fig. 5(c), LLM-based methods again outperform structured-input baselines. FLAME achieves superior out-of-distribution performance, demonstrating the advantage of combining list-wise decision modeling with collaborative knowledge fusion.
0.00 0.05 0.10 0.15 DDI 0.30 0.35 0.40 0.45 0.50 Jaccard D 0,0,&,,, FLAME RareMed MoleRec 2008-2010 2011-2013 2014-2016 2017-2019 MIMIC-IV Time Period 0.0 0.1 0.2 0.3 0.4 Jaccard E 0,0,&,9 FLAME LAMO RareMed MoleRec FL A M E LA M O R a re M e d M o le R e c 0.00 0.05 0.10 0.15 0.20 0.25 Jaccard F H,&8
this section cite: []

Section: Ablation Study
We conduct ablations to assess the impact of each component in FLAME, including the list-wise decision model, step-wise GRPO, and multi-source knowledge fusion.
this section cite: []

Section: List-wise Decision Model.
We first examine the roles of π cls and π list . Table 2 (a)-(b) shows that removing π cls significantly harms performance, highlighting the importance of individualized filtering. Excluding π list also degrades results, indicating that list-wise modeling is essential for capturing decision structures beyond point-wise analysis.
We further evaluate the two-stage training of π list . Skipping SFT limits task-form alignment, while omitting step-wise GRPO weakens preference learning (Table 2 (c)-(d)). These results confirm the necessity of both SFT and step-wise preference alignment.
this section cite: []

Section: Step-wise GRPO.
Replacing step-wise GRPO with standard GRPO (Table 2 (e)) results in clear performance drops, demonstrating that outcome-only rewards are insufficient. Fig. 6 shows that step-wise GRPO provides denser feedback, accelerating learning and improving stability.
this section cite: []

Section: Multi-source Knowledge Fusion.
Integrating structured codes and unstructured clinical notes significantly boosts performance; removing either degrades results (Table 2
this section cite: []

Section: (f)-(g)).
To analyze embedding strategies, we replace collaborative embeddings with random vectors ẽcollab rand . While random embeddings offer slight gains over text-only inputs, they lack meaningful domain knowledge, resulting in a notable gap from FLAME (Table 2 (h)-(j)).
this section cite: []

Section: LLM Backbone.
Replacing the Llama3.1-Aloe-Beta-8B backbone with general large language models (LLaMA2 and LLaMA3) leads to noticeable performance drops (Table 2 (k)-(l)). This highlights the critical role of domain-specific medical knowledge embedded in Llama3.1-Aloe-Beta-8B for accurate medication recommendation.
this section cite: []

Section: In-Depth Analysis
To further understand the empirical behaviors of FLAME, we conduct in-depth analyses on the necessity of the list-wise refinement policy π list , and the effectiveness of step-wise GRPO. These analyses complement the quantitative results and provide qualitative evidence for our design choices.
this section cite: []

Section: Role of the List-wise Policy π list .
In Table 2(b), removing π list results in a relatively small numerical drop (Jaccard 0.4836 → 0.4785), which may appear marginal at first glance. However, this is expected, since π list acts as a lightweight refinement module rather than a primary predictor. On the MIMIC-III test set, π cls outputs lists of average length 20.83, while π list performs only ∼2.03 add/remove edits per patient on average. These small-scale edits are crucial: they primarily resolve subtle drug-drug dependencies and redundancies that point-wise classifiers cannot capture.
We identify two major benefits of π list . (1) Safety-accuracy controllability. π cls is optimized to mimic ground truth prescriptions, producing DDI rates (0.1336) close to the real data distribution (0.1369). While effective for data fitting, this limits its ability to adapt to stricter safety constraints required in real-world clinical practice. In contrast, π list incorporates the safety penalty α (Eq. 8), enabling explicit and controllable adjustment of safety levels. (2) Modeling drug dependencies. Since π cls predicts each drug independently, it cannot capture co-prescription patterns or redundancies. For instance, Piperacillin (ID 130) is a broad-spectrum β-lactam antibiotic overlapping in indication with Cefepime (ID 92), Meropenem (ID 45), and Ampicillin (ID 113). In such cases, π cls may recommend redundant combinations or omit suitable alternatives. By performing list-level reasoning, π list corrects these issues effectively: across the test set, it edited 28 Piperacillin-containing lists with an 89% correctness rate, demonstrating its capacity for meaningful refinement beyond point-wise learning.
Table 3: Examples of π list refinements. IDs 45, 92, 113, and 130 denote antibiotics with overlapping effects. Correct, over-predicted, and missing drugs are colored green, red and blue, respectively. Drugs in parentheses indicate those absent from M p but present in the ground truth list.
Case ID π cls Output M p π list Refinement 266 30, 76, 113, 137, 6, 74, 130, (14, 104, 114), ... add: 14, 114; remove: 130 887 2, 71, 99, 8, 22, 130, (34, 48, 57, 80, 92, 126), ... add: 34, 48, 57, 92; remove: 130
As shown in Table 3, π list rectifies redundant and missing predictions by exploiting list-wise signals.
In Case 266, π cls simultaneously predicts IDs 113 and 130, which share overlapping effects, while omitting several relevant items. π list removes the redundant 130 and adds the missing entries, thereby enhancing the precision of the generated medication list. In Case 887, the point-wise model again over-predicts 130 and misses 92, two drugs with overlapping indications, while π list corrects both errors through list-level reasoning. These examples demonstrate that even lightweight list-wise edits can yield clinically coherent refinements by leveraging inter-drug dependencies absent in point-wise prediction, confirming the essential role of π list beyond its modest aggregate numerical gain.
this section cite: []

Section: Effectiveness of Step-wise GRPO.
Step-wise GRPO is the optimization backbone for π list . While the absolute improvement over standard GRPO in Table 2(e) is modest, the relative effect is substantial: list-wise refinement's advantage over point-wise classification increases from 0.0028 (standard GRPO) to 0.0051 (step-wise GRPO), an 82.1% relative gain. Additionally, Fig. 6 shows faster reward convergence and more stable training dynamics under step-wise GRPO, owing to its denser intermediate feedback rather than reliance on terminal rewards-an important trait for long edit sequences such as medication list refinement.
this section cite: []

Section: Conclusion
We presented FLAME, an LLM-based list-wise medication recommendation framework that formulates prescription generation as a sequential drug-by-drug decision process. By introducing step-wise GRPO with potential-based reward shaping, FLAME enables fine-grained credit assignment, effectively capturing drug interactions and enforcing safety constraints such as DDIs. Through hybrid representations that fuse structured clinical data, collaborative signals, and unstructured textual information, FLAME enhances the LLM's capacity for accurate and clinically grounded recommendations.
Extensive experiments demonstrate FLAME's superior accuracy, controllable safety-accuracy tradeoffs, and strong generalization across temporal and institutional shifts. These results highlight the potential of fine-grained list-wise alignment for building reliable, adaptable clinical decision support systems. In future work, we plan to further improve FLAME's generalization in diverse and evolving healthcare environments, advancing towards scalable and equitable AI-driven solutions for real-world clinical practice. In addition, we will explore the integration of physician decisions with FLAME to build a human-in-the-loop medication recommendation system.
this section cite: []

Section: References
Ref_id:b0 Title: Deep learning for medication recommendation: a systematic survey Year: (2023)
Ref_id:b1 Title: Debiased, longitudinal and coordinated drug recommendation through multi-visit clinic records Year: (2022)
Ref_id:b2 Title: Safedrug: Dual molecular graph encoders for recommending effective and safe drug combinations Year: (2021)
Ref_id:b3 Title: Conditional generation net for medication recommendation Year: (2022)
Ref_id:b4 Title: Leave no patient behind: Enhancing medication recommendation for rare disease patients Year: (2024)
Ref_id:b5 Title: Leap: learning to prescribe effective and safe treatment combinations for multimorbidity Year: (2017)
Ref_id:b6 Title: Gamenet: Graph augmented memory networks for recommending medication combination Year: (2019)
Ref_id:b7 Title: Molerec: Combinatorial drug recommendation with substructure-aware molecular representation learning Year: (2023)
Ref_id:b8 Title: Large language models encode clinical knowledge Year: (2023)
Ref_id:b9 Title: Addressing overprescribing challenges: Fine-tuning large language models for medication recommendation tasks Year: (2025)
Ref_id:b10 Title: Data-driven prediction of drug effects and interactions Year: (2012)
Ref_id:b11 Title: Deepseekmath: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b12 Title: Aloe: A family of fine-tuned open healthcare llms Year: (2024)
Ref_id:b13 Title: Mimic-iii, a freely accessible critical care database Year: (2016)
Ref_id:b14 Title: Mimic-iv, a freely accessible electronic health record dataset Year: (2023)
Ref_id:b15 Title: The eicu collaborative research database, a freely available multi-center database for critical care research Year: (2018)
Ref_id:b16 Title: Natural language-assisted multi-modal medication recommendation Year: (2024)
Ref_id:b17 Title: Sprec: Self-play to debias llm-based recommendation Year: (2025)
Ref_id:b18 Title: Cirs: Bursting filter bubbles by counterfactual interactive recommender system Year: (2023)
Ref_id:b19 Title: How do recommendation models amplify popularity bias? an analysis from the spectral perspective Year: (2025)
Ref_id:b20 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b21 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b22 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b23 Title: R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization Year: (2025)
Ref_id:b24 Title: Electronic health records to facilitate clinical research Year: (2017)
Ref_id:b25 Title: Process-supervised llm recommenders via flow-guided tuning Year: (2025)
Ref_id:b26 Title: Policy invariance under reward transformations: Theory and application to reward shaping Year: (1999)
Ref_id:b27 Title: From r to q * : Your language model is secretly a q-function Year: (2024)
Ref_id:b28 Title: Collm: Integrating collaborative embeddings into large language models for recommendation Year: (2025)
Ref_id:b29 Title: Llara: Large languagerecommendation assistant Year: (2024)
Ref_id:b30 Title: Change matters: Medication change prediction with recurrent residual networks Year: (2021)
Ref_id:b31 Title: Mole-bert: Rethinking pre-training graph neural networks for molecules Year: (2023)
