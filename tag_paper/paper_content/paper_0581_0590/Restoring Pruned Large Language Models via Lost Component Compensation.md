Title: Restoring Pruned Large Language Models via Lost Component Compensation
Abstract: Pruning is a widely used technique to reduce the size and inference cost of large language models (LLMs), but it often causes performance degradation. To mitigate this, existing restoration methods typically employ parameter-efficient fine-tuning (PEFT), such as LoRA, to recover the pruned model's performance. However, most PEFT methods are designed for dense models and overlook the distinct properties of pruned models, often resulting in suboptimal recovery. In this work, we propose a targeted restoration strategy for pruned models that restores performance while preserving their low cost and high efficiency. We observe that pruning-induced information loss is reflected in attention activations, and selectively reintroducing components of this information can significantly recover model performance. Based on this insight, we introduce RestoreLCC (Restoring Pruned LLMs via Lost Component Compensation), a plug-and-play method that contrastively probes critical attention heads via activation editing, extracts lost components from activation differences, and finally injects them back into the corresponding pruned heads for compensation and recovery. RestoreLCC is compatible with structured, semi-structured, and unstructured pruning schemes. Extensive experiments demonstrate that RestoreLCC consistently outperforms state-of-the-art baselines in both general and task-specific performance recovery, without compromising the sparsity or inference efficiency of pruned models 2 .

Section: Introduction
Large language models (LLMs) have achieved remarkable success in various natural language processing (NLP) tasks like commonsense reasoning, math problem solving, and text completion [1,2,3]. However, their large parameter counts demand substantial computational resources for deployment and inference. To democratize LLMs, pruning has emerged as a key technique to reduce model size and accelerate inference [4]. Pruning typically involves two steps: weight pruning and performance restoration. Weight pruning methods can be roughly categorized into structured pruning, such as LLM-Pruner [5] and SlimGPT [6], semi-structured pruning, and unstructured pruning, such as SparseGPT [7] and Wanda [4]. These methods estimate the importance of parameters and zero out unimportant ones to reduce model size. Performance restoration is a critical step in mitigating the performance degradation caused by weight pruning, aiming to recover model capability by adjusting weights through language modeling or instruction tuning datasets [8,9,10].
As full fine-tuning (FT) still requires substantial computational resources, parameter-efficient finetuning (PEFT) has become the mainstream approach for restoring pruned models. Recent pruning methods, including LLM-Pruner, SparseGPT, Wanda, and SlimGPT, all adopt LoRA [11] to recover performance. Although seemingly straightforward, applying existing PEFT methods, such as LoRAbased approaches [11,12,13] and representation engineering [14,15,16], to pruned LLMs raises concerns.
These PEFT methods are originally designed for dense models, where they adapt LLMs to downstream tasks by training a small subset of parameters. When applied to pruned models, they often overlook pruning-specific characteristics, such as the need to account for lost information, leading to inefficient parameter search and suboptimal restoration. As shown in Figure 1, attention heads recovered using LoRA achieve only limited improvement in logit predictions, resulting in suboptimal performance on the final task accuracy. In contrast, directly reintroducing the pruned components into the pruned attention heads (LCC) significantly improves both the outputs of the attention heads and the overall model accuracy. Built on this insight, we propose RestoreLCC (Restoring Pruned LLMs via Lost Component Compensation), which explicitly reconstructs critical information lost during pruning to bridge the performance gap between pruned and dense models.
Specifically, our study in § 3 reveals that important information removed by pruning can be captured in attention head activations, and reintroducing components of this lost information can substantially restore the performance of the pruned model. However, this insight also presents challenges, such as determining which attention heads to select and how to estimate component vectors that encode critical information for compensation. To overcome these challenges, RestoreLCC incorporates two main mechanisms: (1) contrastive probing, a general approach that leverages activation editing to construct contrastive sample pairs and probes a subset of attention heads critical for performance recovery; and (2) lost component compensation (LCC), which retrieves pruning-induced lost information from these key heads. This information is decomposed into components, represented as vectors that capture the lost information directions. We optimize their magnitudes to enable targeted restoration along these directions, aggregate them into a single informative component, and finally inject it into the pruned model to recover performance. Unlike existing PEFT methods that restore pruned LLMs in an unguided manner, RestoreLCC explicitly compensates for key components lost during pruning, offering a targeted and effective restoration strategy.
We empirically evaluate RestoreLCC against other performance restoration methods across all three types of pruned models (e.g., structured, semi-structured, and unstructured), on both general recovery and task-specific settings across a wide range of LLMs of different sizes. Our results show that RestoreLCC significantly improves pruned model performance under general recovery settings while maintaining similar inference speed and sparsity ratios, outperforming existing PEFT methods. Furthermore, under task-specific recovery settings, RestoreLCC successfully recovers task-specific information and enables higher pruning ratios, where other PEFT methods often fail. Contributions. Our main contributions are: (1) We observe that pruning-induced information loss is reflected in attention activations, and that selectively restoring key components can significantly recover model performance ( § 3); (2) We propose RestoreLCC, a method that learns the magnitudes of important component directions lost during pruning and reintroduces them to restore pruned models effectively ( § 4); (3) Extensive experiments across various LLMs and pruning schemes demonstrate that RestoreLCC consistently outperforms existing restoration baselines ( § 5).
2 Related work LLM Pruning. Pruning reduces model size and speeds up inference by removing less important weights. Unstructured pruning removes individual weights irrespective of position, as in SparseGPT [7], which uses approximate Hessian-based reconstruction, and Wanda [4], which ranks weights by magnitude and activation norms. DSOT [17] and ALPS [18] further refine sparsity via optimization. Semi-structured pruning enforces patterns like N:M sparsity [19], and many unstructured methods adapt to this format [4,7,18]. Structured pruning removes entire blocks (e.g., rows or columns) to enhance hardware efficiency. Recent methods include LLM-Pruner [5], Compresso [20], LoRAPrune [21], and SlimGPT [6].
Performance Restoration and PEFT. Pruning methods such as LLM-Pruner [5], LoRAPrune [21], SparseGPT [7], Wanda [4], and SlimGPT [6] consistently benefit from a performance restoration step, typically involving fine-tuning on language modeling (e.g., C4 [8], WikiText [9]) or instructiontuning datasets (e.g., Alpaca [10]). While full fine-tuning (FT) is effective, it remains computationally expensive, often requiring days on large GPU clusters. Parameter-efficient fine-tuning (PEFT) offers a cheaper alternative by updating a small subset of parameters. PEFT techniques include adapterbased [22,23,24], prompt-based [25,26], LoRA-based (e.g., LoRA [11], AFLoRA [27], VERA [12], DoRA [13]), and representation-engineering methods (e.g., RED [14], ReFT [15], LoFit [16]). Among them, LoRA is the most widely adopted in modern pruning pipelines. In addition to PEFT methods, FLAP [28] proposes a bias compensation strategy to reduce the pruning loss. EoRA [29] provides a fine-tuning-free approach to recover pruned models by searching low-rank spaces in a task-specific eigenspace to minimize compression loss.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b3', 'b7', 'b8', 'b9', 'b10', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b0', 'b1', 'b6', 'b3', 'b16', 'b17', 'b18', 'b3', 'b6', 'b17', 'b4', 'b19', 'b20', 'b5', 'b4', 'b20', 'b6', 'b3', 'b5', 'b7', 'b8', 'b9', 'b21', 'b22', 'b23', 'b24', 'b25', 'b10', 'b26', 'b11', 'b12', 'b13', 'b14', 'b15', 'b27', 'b28']

Section: Insight: injecting pruned components back effectively restores models
Preliminaries. We begin by clarifying several key terms and the scope of this study. Current LLM pruning methods primarily target the weight matrices of both attention and feed-forward (FFN) modules. For example, Wanda removes 50% of the weights in each matrix within both the attention and FFN modules (setting them to zero), resulting in an overall sparsity of 50%. Accordingly, in our work, all weight matrices within the LLM are pruned, and we focus on restoring the pruned model by compensating through attention heads. An activation refers to the output of a specific module within the Transformer architecture. A pruned activation is the output produced by a pruned module. Although its dimensionality remains identical to that of the original activation, it typically contains less information because the underlying weight matrices have been pruned. Importantly, the activations themselves are not pruned-only the associated weight matrices are. Sparsity denotes the proportion of weight parameters that have been pruned.
Pruning inherently leads to information loss, which becomes more pronounced at higher pruning ratios. In this section, we demonstrate that the pruned activations contain critical information, including discriminative components essential for downstream NLP tasks. By reintroducing the components, the performance of the pruned model can be substantially restored.
Recent studies have shown that different attention heads specialize in distinct functions when performing NLP tasks [30,31,32]. Motivated by this, we investigate how pruning-induced loss of head activations affects the model's retained functional capacity and performance. Let MultiHead denote the multi-head attention output in a Transformer block with H attention heads. The output can be expressed as:
MultiHead x l = concat z (l,0) , . . . , z (l,H-1) W O ,(1)
where x l is the input to the l-th layer, z (l,h) is the output of the h-th head, and W O is a shared output projection matrix. Denote by z
(l,h) d and z (l,h) p the activations of the h-th head in the dense and pruned models, respectively, for the same input. The lost activation can be computed as Eq. 2. Our objective is to analyze whether δz (l,h) carries critical information that could aid in model recovery. δz (l,h)
d-p = z (l,h) d -z (l,h) p .(2)
Given N samples, we denote the activation loss matrix for all samples as ∆Z (l,h) = [δz (l,i) 0,d-p ; . . . ; δz (l,h) N -1,d-p ] ∈ R N ×d h , where d h is the dimensionality of the head activation. To better characterize the structure of the lost information, we apply singular value decomposition (SVD) to ∆Z (l,h) , as shown in Eq. 3, which decomposes the activation loss matrix into a set of orthogonal latent components. This formulation enables us to identify the dominant directions of lost activation information and quantify their impact on each sample.
∆Z (l,h) = U (l,h) Σ l,h) V (l,h) ⊤ = d h i=1 σ i u (l,h) i v (l,h) i ≈ K i=1 σ i u (l,h) i v (l,h) i .(3)
For the output z (l,h) p of a pruned attention head, the lost principal components can be approximated by
c (l,h) = K i=1 α (l,h) i v (l,h) i ,(4)
where c (l,h) ∈ R d h , and α
(l,h) i denotes the average of σ i u (l,h) i
, representing the mean projection coefficients across samples. Finally, the activation output of the pruned attention head z (l,h) p can be compensated and recovered by injecting the estimated lost components c (l,h) back:
z (l,h) c = z (l,h) p + c (l,h) ,(5)
where we omit the sample index i in z (l,h) i,c
and z (l,h) i,p for simplicity. To evaluate the predictive behavior of an attention head's activation, we follow the theories of LogitLens [33] and prior work on interpreting LLMs in embedding space [34]. Specifically, we project the activation into the embedding space and compute its prediction over the vocabulary space:
p (l,h) zc = LM_Head ϕ z (l,h) c W O,h ,(6)
where LM_Head denotes the LLM's prediction head, ϕ(•) is the layer normalization function, W O,h is the output projection matrix in W O corresponding to the h-th head, p (l,h) zc ∈ R |V| , V represents the model's vocabulary, and |V| is the vocabulary size. Following IOI [35], we compute the logit difference, defined as the logit assigned to the correct token minus the logit assigned to the wrong token. This difference directly reflects the model's confidence and faithfulness in predicting the correct token over an incorrect alternative.
this section cite: ['b29', 'b30', 'b31', 'b32', 'b33', 'b34']

Section: Empirical Study.
To evaluate whether reintroducing lost components can aid in performance recovery, we conduct an empirical study using BoolQ [36], a widely used commonsense reasoning dataset. We adopt LLaMA-7B [2] as the dense backbone model and apply Wanda [4], a state-of-theart pruning method, to prune the model to 50% sparsity. We randomly sample 1,000 examples from BoolQ, where both the dense and pruned models are tasked with answering "yes" or "no" questions. We extract the attention head activations from the pruned model and reconstruct the compensated activations using the main components, as described in Eqs. 2-5. To quantify model confidence, we compute the logit difference λ using:
λ = p (l,
h) zc [yes] -p (l,h) zc [no] or p (l,h) zc [no] -p (l,h) zc [yes], We also report final accuracy, i.e., the predictions from the last layer, to assess the effect of component compensation on the model's output. 5 10 15 20 25 30 Layer Index 0.8 1.0 1.2 1.4 1.6 1.8 Logit Gain l30.a13 l12.a16 l12.a19 l20.a17 l26.a16 l19.a6 l16.a25 l5.a20 l18.a14 l11.a18 l10.a0 l14.a31 l6.a6 l27.a14 l8.a10 l10.a2 l10.a4 l13.a18 l12.a24 l13.a19
Attention Head Index l 3 0 . a 1 3 l 1 9 . a 6 l 2 6 . a 1 6 l 7 . a 2 6 l 2 0 . a 1 3 0.0 0.5 1.0 1.5 2.0 2.5 Logit Gain k=27 k=13 k=9 k=2 k=12 component 1 component K Figure 3: Logit gain of attention heads recovered with different components.
Finding 1. Reintroducing lost components to selected pruned attention heads can significantly restore model performance. As shown in Figure 1, we use the top-10 (K = 10 in Eq. 4) lost components to reconstruct the pruned activations. The left figure demonstrates that this compensation substantially recovers the pruned activations. Specifically, the logit difference is restored to a level comparable to that of the original dense model. Furthermore, compensating these attention heads also leads to a notable improvement in the model's final output accuracy.
this section cite: ['b35', 'b1', 'b3']

Section: Finding 2. Different attention heads vary in importance and exhibit distinct recovery behaviors.
We manually select and examine the outputs of several attention heads, along with their restoration using the top-10 lost components. To quantify the direct effect of compensation, we calculate the logit gains as δλ = λ recovered -λ pruned , which directly captures the improvement brought by reintroducing the lost components. Figure 2 shows the recovered logit gains for these heads. The results reveal that attention heads respond differently to the compensation process, indicating that not all can be effectively restored using the lost principal components.
Finding 3. Discriminative information may reside in minor components rather than in the principal ones. Figure 3 illustrates the recovery performance using either the top principal component or a selected minor component. Notably, since the coefficients of minor components are extremely small, we scale them by a factor of 1000 for visualization and evaluation. Surprisingly, incorporating certain minor components can lead to substantially better restoration performance compared to using the leading principal component.
this section cite: []

Section: Extension to FFN.
Each Transformer layer comprises a multi-head attention (MHA) module and a feed-forward network (FFN) module. In the above analysis, we focus on compensating for the MHA rather than the FFN. To further justify this design choice, that is, using MHA instead of FFN, we provide additional theoretical and empirical analyses in Appendix A.
To summarize, the above analysis highlights not only the strong potential of leveraging lost components for performance restoration, but also several key challenges: (1) How to select pruned attention heads that are important for recovery? (2) How to determine the positions and coefficients of key components? and (3) How to apply the findings to more general tasks beyond BoolQ?
this section cite: []

Section: RestoreLCC: restoring pruned LLMs via lost component compensation
To address these challenges and enable universal performance restoration for pruned LLMs, we propose RestoreLCC, shown as Figure 4, which consists of two key mechanisms: (1) contrastive probing, a general method that uses activation editing to create contrastive sample pairs and identify critical attention heads; and (2) lost component compensation (LCC), which optimizes the magnitudes of directional components for the lost information. The optimized components are then injected back into the model to restore performance.
this section cite: []

Section: Contrastive probing
Contrastive Sample Construction. We construct contrastive samples for general NLP tasks to support activation editing and probing to localize important heads. Given a dataset, either task-
Lost Component Compensation Dense Model Pruned Model ... ... H heads L layers 𝒎𝒎 + 𝒎𝒎 - ... ... ... ... Contrastive Probing ... ... ∆𝐙𝐙 = U 𝜮𝜮 V 𝒗𝒗 𝟏𝟏 𝒗𝒗 𝟐𝟐 𝒗𝒗 𝟑𝟑 𝒗𝒗 𝟒𝟒 ... 𝜷𝜷 𝟏𝟏 𝒗𝒗 𝟏𝟏 𝜷𝜷 𝟐𝟐 𝒗𝒗 𝟐𝟐 𝜷𝜷 𝟑𝟑 𝒗𝒗 𝟑𝟑 𝜷𝜷 𝟒𝟒 𝒗𝒗 𝟒𝟒 Train magnitudes Dense Head Sparse Head Restoration for these directions 𝒄𝒄 learned = � 𝜷𝜷 𝒊𝒊 𝒗𝒗 𝒊𝒊 + 𝒃𝒃 Important Head Identification specific (e.g., BoolQ) or general (e.g., the Alpaca dataset), every sample is organized as (q, r + ), where q is the question and r + is the correct (positive) response. All responses are collected into a set [r + 0 , . . . , r + N -1 ]. We use a sentence encoder, such as MiniLM-L6 [37], to encode all responses. For each sample, we select the most similar response (excluding the correct one) based on cosine similarity as the negative response r -. This process converts each sample into a contrastive tuple (q, r + , r -). Note that this method is universal and can be applied to any dataset. We provide examples of constructed samples from the BoolQ and Alpaca datasets in Appendix B.
this section cite: ['b36']

Section: Activation Editing.
Recent studies suggest that high-dimensional activations in LLMs are approximately orthogonal with high probability [38]. An activation can be guided toward a desired generation space by adding a steering vector or a task-specific function vector [39,32]. For pruned LLMs, we assume the lost principal components can be reintroduced to recover the activation and restore the correct response. Formally: recovered question activation ≈ pruned question activation + lost important component. Based on this, the recovered activation for a question is defined as
z q c = z q p + c q , (7
)
where we omit the layer and head indices (l, h) for simplicity. Here, q indicates the activation is taken from the last token of the question, and the other symbols follow the notation in Eq. 5.
Attention Head Probing. Ideally, the recovered activation z q c should contain sufficient information to generate the correct response. It should be consistent with the activation of the complete and correct sample, denoted as z q+r + d , which is obtained from the last token of the full sequence (including both the question q and the positive response r + ) in the original dense model. Additionally, it should be contrastive with respect to the negative sequence activation z q+rd . However, some attention heads may be inactive, and certain components may contribute little to performance restoration. In such cases, the recovered activation z q c may not adequately distinguish between z q+r + d and z q+rd .
In this way, identifying important attention heads can be formulated as a natural language inference (NLI) task. If an attention head is important and the corresponding component is useful, then the recovered activation z q c should entail the correct sequence activation z
q+r + d and contradict the negative sequence activation z q+rd . Specifically, activation pairs are constructed as m + = [z q c , z q+r + d ] with label 1 (entailment) and m -= [z q c , z q+rd
] with label 0 (contradiction). A probing classifier, composed of a linear layer followed by a sigmoid activation, is trained to assess the discriminative power of each recovered head activation. The importance of attention heads is then ranked based on the accuracy of their corresponding probing classifiers.
this section cite: ['b37', 'b38', 'b31']

Section: Lost component compensation
We now have a list of important attention heads along with their corresponding lost components v i based on Eq. 3. The goal is to use these components to approximate the missing information and recover the performance of pruned LLMs.
this section cite: []

Section: Lost Component Estimation.
As observed in § 3, the usefulness of a component does not necessarily correlate with its coefficient. Minor components may also carry critical information and enable more effective recovery of pruned LLMs. To address this, we consider all possible components. Since the vectors v i are orthogonal and unit-normalized, we treat each v i as a potential direction of lost information. We keep these directions fixed and learn a scalar magnitude for each, representing its importance. Formally, for an attention head, we model its lost component containing important information as:
c learned = d h i=1 β i v i + b.(8)
Here, v i is obtained from Eq. 3 and remains fixed during training, while β i is a trainable scalar indicating the importance of each direction. The term b ∈ R d h is a trainable bias vector for the attention head, acting as a hedging vector to provide flexibility in cases where important information lies outside the span of the predefined directions.
Component Compensation. For each pruned attention head, its output activation is recovered as zp = z p + c learned . It is worth noting that the final learned component c learned is a constant bias vector, capturing important information that was lost for all samples as a result of pruning. The recovered activation zp is then passed to the subsequent computations, such as those described in Eq. 1, allowing the model to better approximate the behavior of the original dense LLM.
this section cite: []

Section: Extension of RestoreLCC to FFNs.
It is worth noting that RestoreLCC can be seamlessly extended to FFN modules. Our objective is to restore pruned models in which all weight matrices-both in the attention and FFN modules-have been pruned. We provide further theoretical and empirical analyses in Appendix A to justify why we apply RestoreLCC to attention heads instead of FFNs.
this section cite: []

Section: Overhead analysis of sparsity and inference speed
LLM pruning aims to produce a sparse model, while performance restoration seeks to close the performance gap between the pruned and dense models. It is important to ensure that performance restoration improves the pruned model's accuracy without compromising its sparsity or inference efficiency.
this section cite: []

Section: Sparsity Analysis.
Each compensated attention head introduces a learned vector c learned ∈ R d h . In the worst case, where all heads in a layer are compensated, the total number of additional parameters introduced across the projection matrices (q_proj, k_proj, v_proj, o_proj) within the multi-head attention mechanism is:
Parameter Overhead = 2d l 4d 2 l = 1 2d l ,
where d l is the hidden size of the layer and d h = d l /H. Since d l is typically larger than 1000, the increase in parameters is less than 0.05%, which has minimal impact on sparsity.
Inference Speed. To maintain inference efficiency, the learned vector c learned is absorbed as the constant bias vector in the multi-head attention block. This modification introduces almost no additional computation during inference and preserves the speed of the pruned model. Therefore, RestoreLCC effectively preserves the pruned model's sparsity and inference speed while recovering its performance. Additional empirical evidence is provided in Appendix D.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental settings
Metrics. To evaluate the restoration effectiveness of RestoreLCC, we experiment with representative pruned LLMs from diverse pruning strategies. Specifically, we use Wanda [4] for unstructured pruning, SparserGPT [7] for semi-structured pruning, and SlimGPT [6] for structured pruning. The pruning ratio is set to 50% for unstructured and semi-structured methods, and 20% for structured pruning, with C4 [8] as the calibration dataset. Following these works, we assess perplexity (PPL) of language modeling on the held-out WikiText [9] and accuracy of several commonsense reasoning benchmarks, including BoolQ [36], HellaSwag [40], WinoGrande [41], ARC-easy [42], ARCchallenge [42], RTE [43], and OpenBookQA [44], all evaluated using the lm-eval-harness framework [45]. We consider two restoration settings:
• General Recovery. It is widely adopted in previous work. We follow SlimGPT and use the Alpaca instruction dataset [10] for tuning. Evaluations are performed in a zero-shot setting across language modeling and commonsense reasoning tasks.
• Task-Specific Recovery. While prior work focuses on general recovery, we highlight the importance of task-specific restoration, as pruned LLMs should also perform effectively when deployed in specialized domains. To evaluate this, we increase the sparsity ratio and restore pruned models using 100 3 training examples (e.g. 100 BoolQ samples) from the target task (e.g., BoolQ).
this section cite: ['b3', 'b6', 'b5', 'b7', 'b8', 'b35', 'b39', 'b40', 'b41', 'b41', 'b42', 'b43', 'b44', 'b9']

Section: Models and Implementations.
We conduct our main experiments using LLaMA-7B/13B models [2]. To evaluate the universality and scalability of RestoreLCC, we provide additional results on LLMs of varying sizes and families, including LLaMA-30B, LLaMA-2-7B/13B, LLaMA-3-8B [2], Vicuna-7b-v1.5 [46], Tulu-2-7B [47], Qwen-3-8B/14B [48], and DeepSeek-R1-Qwen3-8B [49] in Appendix K. For RestoreLCC, the number of components K is set to 1 to identify important attention heads, and we select 10%-25% of them for recovery. Additional implementation details are also provided in Appendix K.
Baselines. To ensure a comprehensive evaluation, we compare RestoreLCC with comprehensive baselines methods as follows:
• LoRA [11], which fine-tunes low-rank adapters to update the parameters of pruned LLMs. It is widely used for performance restoration in recent SOTA pruning methods [4,5,6,7].
• DORA [13], which enhances LoRA by decomposing low-rank adaptation.
• FLAP [28], which recovers the pruned model by bias compensation.
• EoRA [29], which searches low-rank spaces in the eigenspace to minimize compression loss.
• LoFiT [16], which is a SOTA representation engineering method that intervenes in attention activations to adapt LLMs to downstream tasks.
this section cite: ['b1', 'b1', 'b45', 'b46', 'b47', 'b48', 'b10', 'b3', 'b4', 'b5', 'b6', 'b12', 'b27', 'b28', 'b15']

Section: Performance of general recovery
Table 1 reports results across three pruning regimes on LLaMA-7B. Under unstructured pruning at 50% sparsity, RestoreLCC achieves 58.83% mean accuracy, outperforming the best baseline, LoFiT (56.82%), by +2.01%, and reduces PPL to 6.93. In the semi-structured pruning setting, RestoreLCC improves mean accuracy to 55.00%, yielding a +2.65% gain over the best baseline DoRA (52.35%), while also lowering PPL from 9.16 to 8.99. Under structured pruning at 20% sparsity, RestoreLCC reaches 59.76% accuracy, improving over DoRA (58.51%) by +1.25%. Across all settings, RestoreLCC consistently outperforms prior recovery methods in both accuracy and perplexity, recovering performance close to the dense model (59.99%). Table 1: Performance of general recovery on zero-shot language modeling (PPL) and commonsense reasoning tasks (accuracy) using LLaMA-7B. Best scores are bolded. Method PPL ↓ BoolQ ↑ RTE↑ HellaSwag ↑ WinoGrande ↑ ARC-e ↑ ARC-c ↑ OBQA ↑ Mean ↑ Dense Model 5.68 75.02 66.79 56.95 69.93 75.29 41.72 34.20 59.99 Unstructured Pruning at 50% Sparsity Wanda (Base Model for Recovery) 7.26 71.07 54.87 51.86 65.98 69.19 37.03 28.60 54.09 LoRA 7.09 72.05 58.84 52.93 66.85 71.68 39.16 32.40 56.27 DoRA 7.11 71.83 62.45 54.09 65.19 71.13 39.76 31.60 56.58 EoRA 7.14 74.16 60.29 51.27 68.27 70.96 37.63 28.60 55.88 LoFiT 7.35 72.29 64.26 54.79 65.19 69.99 38.40 32.80 56.82 RestoreLCC (Ours) 6.93 72.84 69.68 56.34 65.98 71.80 40.96 34.20 58.83 (+2.01) Semi-Structured Pruning (N:M=2:4) at 50% Sparsity SparseGPT ((Base Model for Recovery) 11.04 69.45 54.51 43.12 60.93 60.90 30.20 23.80 48.99 LoRA 9.32 70.89 58.12 48.81 63.77 64.60 31.06 24.40 51.66 DoRA 9.16 71.41 59.21 49.16 62.04 65.99 33.45 25.20 52.35 FLAP 10.57 68.81 54.15 44.48 64.40 63.97 30.03 24.00 49.98 EoRA 9.87 71.68 59.93 44.65 64.33 63.72 30.38 23.80 51.21 LoFiT 10.02 70.24 58.48 49.41 63.06 64.06 32.51 27.00 52.11 RestoreLCC (Ours) 8.99 73.61 63.90 51.71 65.11 68.14 33.53 29.00 55.00 (+2.65) Structured Pruning at 20% Sparsity SlimGPT (Base Model for Recovery) 7.46 75.99 62.09 53.73 67.72 72.14 39.33 31.80 57.54 LoRA 7.66 76.48 65.70 55.25 66.77 72.01 40.36 32.60 58.45 DoRA 7.54 76.79 64.62 55.40 66.77 72.35 40.61 33.00 58.51 LoFiT 7.86 75.90 63.54 56.26 67.80 71.34 40.53 33.60 58.42 RestoreLCC (Ours) 7.53 76.48 68.59 57.05 69.46 72.01 40.53 34.20 59.76 (+1.25)  Table 2 reports the recovery performance of LLaMA-13B; detailed results are provided in Table 17. Under unstructured pruning, RestoreLCC achieves 62.46% mean accuracy with a PPL of 6.08, corresponding to a +1.04% gain over the best baseline. In the semi-structured setting, RestoreLCC outperforms LoFiT by +1.73%. For structured pruning, RestoreLCC reaches 63.41% accuracy, surpassing DoRA by +1.45%. These results verify that RestoreLCC generalizes well to larger models. and consistently provides superior recovery across all pruning types.
this section cite: []

Section: Task-specific recovery
Table 3 evaluates task-specific recovery on LLaMA-7B under three sparsity settings. For unstructured pruning at 60% sparsity, RestoreLCC achieves a mean accuracy of 61.77%, improving over LoFiT (58.21%) by +3.56%. In the semi-structured pruning, RestoreLCC achieves 63.74% mean accuracy, outperforming LoFiT by +3.06%. For structured pruning at 40% sparsity, RestoreLCC again yields the best mean accuracy (59.55%), outperforming LoRA by +3.4%. In addition, we increase the pruning ratio by a challenging 10-20%. RestoreLCC successfully recovers performance, demonstrating its effectiveness in enabling higher sparsity ratios for LLM pruning.
this section cite: []

Section: Ablation study

this section cite: []

Section: Effects of Contrastive Probing on Identifying Important Attention Heads.
We apply contrastive probing to identify attention heads critical for performance recovery. Table 4 compares performance using heads selected by contrastive probing (RestoreLCC) versus randomly selected heads (w/o probing). The 1.26% degradation in the latter case confirms that contrastive probing effectively identifies heads essential for recovery. We also conduct experiments with two alternative head-selection strategies: (1) MSE-selected heads: selecting heads with the smallest MSE between the outputs of the dense and pruned models. (2) KL-selected heads: selecting heads with the smallest KL divergence. It can be observed that our probing-based selection consistently identifies important heads and is more effective than other metric-based approaches.
this section cite: []

Section: Effects of
d h i=1 β i v i .
We estimate the lost information from pruning using d h i=1 β i v i , which captures both direction and magnitude. To evaluate its impact, we remove this term and tune c learned using only the bias vector in Eq. 8. This results in a 1.70% performance drop, underscoring the importance of recovering both directional and magnitude information in pruned models.
Effects of Bias Vector. We introduce a bias vector in Eq. 8 to serve as a hedging term, allowing flexibility when important information lies outside the span of predefined directions. Table 4 reports the results of removing the bias vector (w/o b), showing a 0.57% performance drop, which confirms its role in optimizing the final learned component.
this section cite: []

Section: Interpreting the learned component
l30a13 '_no' , 'no' , ' _yes' , 'No' , 'yes' l29a10 '_yes' , '_young' _Young' , 'no' , 'yes' l26a16 'yes' , '_yes' , ' _off ' , 'no' , 'YES' Heads Top-5 Decoded Tokens
this section cite: []

Section: Figure 5: Visualization of learned components for different attention heads.
To examine the information encoded in the learned component c learned and its role in performance recovery, we illustrate task-specific recovery on BoolQ (binary yes/no answering). Following § 3, we use LogitLens to project c learned into the embedding space and show the top-5 decoded tokens in Figure 5. The results suggest that c learned captures task-relevant signals, such as indicating "yes" or "no" answers for BoolQ examples.
this section cite: []

Section: Further analysis
We provide additional analysis to verify the universality and efficiency of RestoreLCC as follows:
• Hyperparameter Sensitivity (Appendix C): We evaluate RestoreLCC under varying numbers of attention heads and components in Eq
. 3. Results demonstrate its stability. • Overhead and Efficiency Analysis ( § 4.3 and Appendix D): We compare the trainable parameters, inference speed and overhead sensitivity of RestoreLCC with other baselines. The results verify that RestoreLCC restores pruned LLMs without compromising sparsity or inference efficiency. • Parameter Visualization (Appendix E): We visualize the trained directions, magnitudes, and biases, offering deeper insight into the internal mechanisms of RestoreLCC. • Comparison with Full-Parameter Tuning (Appendix F): We present experimental results that compare RestoreLCC with full-parameter tuning (FT). • Cross-Task Portability and Generalization of Probing (Appendix G): We discuss the cross-task portability and generalization of the contrastive probing module in Appendix G. • Efficiency at Scale (Appendix H): We demonstrate the efficiency of RestoreLCC on a larger LLM (LLaMA-70B) and evaluate its latency. • Compatibility with Quantized Models. (Appendix H): We conducted experiments with 4-bit quantization on the pruned model and verify RestoreLCC's compatibility with heavily quantized models. • Effect of Probing Samples. (Appendix J): We study the effect of the number of probing samples on RestoreLCC. • Evaluation on More LLMs (Appendix K): Experiments on LLMs including LLaMA-30B, LLaMA-2-7B/13B, LLaMA-3-8B [2], Vicuna-7b-v1.5 [46], Tulu-2-7B [47], Qwen-3-8B/14B [48], and DeepSeek-R1-Qwen3-8B [49] further validate the universality and scalability of Re-storeLCC.
this section cite: []

Section: Conclusion
In this work, we propose RestoreLCC, a targeted strategy for restoring the performance of pruned LLMs without compromising their sparsity or inference speed. RestoreLCC integrates two key mechanisms: (1) contrastive probing, which leverages activation editing to probe critical attention heads, and (2) lost component compensation (LCC), which estimates and restores the lost directional information in pruned heads. Extensive experiments across diverse pruning settings and LLMs demonstrate the effectiveness of RestoreLCC in recovering model performance.
Limitation. We assign learnable magnitudes to all components to compensate for pruned attention heads, allowing less important ones to be down-weighted. However, this still may cause overfitting of unimportant components. In future work, we plan to pre-select relevant components before learning their magnitudes, reducing both overfitting and the number of trainable parameters. In addition, Moore and Chaudhuri [50] utilize activation noise to probe network structure and identify redundant neurons. They also discuss a novel way of leveraging activation noise for neuron identification. We believe activation noise could similarly be used to enhance contrastive probing, and we plan to explore this direction in future work.
this section cite: ['b49']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b2 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b3 Title: A simple and effective pruning approach for large language models Year: (2023)
Ref_id:b4 Title: Llm-pruner: On the structural pruning of large language models Year: (2023)
Ref_id:b5 Title: Slimgpt: Layer-wise structured pruning for large language models Year: (2024)
Ref_id:b6 Title: Massive language models can be accurately pruned in one-shot Year: (2023-07)
Ref_id:b7 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b8 Title: Pointer sentinel mixture models Year: (2017)
Ref_id:b9 Title: Stanford alpaca: An instruction-following llama model Year: (2023)
Ref_id:b10 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b11 Title: Vector-based random matrix adaptation Year: (2024)
Ref_id:b12 Title: weight-decomposed low-rank adaptation Year: (2024)
Ref_id:b13 Title: Advancing parameter efficiency in finetuning via representation editing Year: (2024-08)
Ref_id:b14 Title: Reft: Representation finetuning for language models Year: (2024)
Ref_id:b15 Title: Lofit: Localized fine-tuning on llm representations Year: (2024)
Ref_id:b16 Title: Dynamic sparse no training: Training-free fine-tuning for sparse llms Year: (2024)
Ref_id:b17 Title: Alps: Improved optimization for highly sparse one-shot pruning for large language models Year: (2024)
Ref_id:b18 Title: Accelerating sparse deep neural networks Year: (2021)
Ref_id:b19 Title: Compresso: Structured pruning with collaborative prompting learns compact large language models Year: (2023)
Ref_id:b20 Title: LoRAPrune: Structured pruning meets low-rank parameter-efficient fine-tuning Year: (2024-08)
Ref_id:b21 Title: Parameter-efficient transfer learning for NLP Year: (2019-06)
Ref_id:b22 Title: Parameter-efficient multi-task fine-tuning for transformers via shared hypernetworks Year: (2021-08)
Ref_id:b23 Title: Compacter: efficient low-rank hypercomplex adapter layers Year: (2021)
Ref_id:b24 Title: The power of scale for parameter-efficient prompt tuning Year: (2021-11)
Ref_id:b25 Title: Residual prompt tuning: improving prompt tuning with residual reparameterization Year: (2023-07)
Ref_id:b26 Title: AFLoRA: Adaptive freezing of low rank adaptation in parameter efficient fine-tuning of large models Year: (2024-08)
Ref_id:b27 Title: Fluctuation-based adaptive structured pruning for large language models Year: (2024)
Ref_id:b28 Title: Eora: Trainingfree compensation for compressed llm with eigenspace low-rank approximation Year: (2024)
Ref_id:b29 Title: Interpreting and improving large language models in arithmetic calculation Year: (2024-07)
Ref_id:b30 Title: Model tells you what to discard: Adaptive kv cache compression for llms Year: (2024)
Ref_id:b31 Title: Dressing up llm: Efficient stylized question-answering via style subspace editing Year: (2025)
Ref_id:b32 Title: Interpreting gpt: the logit lens Year: (2020)
Ref_id:b33 Title: Analyzing transformers in embedding space Year: (2023-07)
Ref_id:b34 Title: Interpretability in the wild: a circuit for indirect object identification in gpt-2 small Year: (2023)
Ref_id:b35 Title: BoolQ: Exploring the surprising difficulty of natural yes/no questions Year: (2019-06)
Ref_id:b36 Title: Deep selfattention distillation for task-agnostic compression of pre-trained transformers Year: (2020)
Ref_id:b37 Title: Overparameterized random feature regression with nearly orthogonal data Year: (2023-04)
Ref_id:b38 Title: Function vectors in large language models Year: (2024)
Ref_id:b39 Title: HellaSwag: Can a machine really finish your sentence Year: (2019-07)
Ref_id:b40 Title: Winogrande: an adversarial winograd schema challenge at scale Year: (2021-08)
Ref_id:b41 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b42 Title: Glue: A multi-task benchmark and analysis platform for natural language understanding Year: (2018)
Ref_id:b43 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018-11)
Ref_id:b44 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b45 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b46 Title: Camels in a changing climate: Enhancing lm adaptation with tulu 2 Year: (2023)
Ref_id:b47 Title: Qwen3 technical report Year: (2025)
Ref_id:b48 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b49 Title: Using noise to probe recurrent neural network structure and prune synapses Year: (2020)
Ref_id:b50 Title: Interpreting and improving large language models in arithmetic calculation Year: (2024)
Ref_id:b51 Title: Locating and editing factual associations in gpt Year: (2022)
Ref_id:b52 Title: Transformer feed-forward layers are key-value memories Year: (2021-11)
