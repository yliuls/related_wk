Title: Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
Abstract: Gating mechanisms have been widely utilized, from early models like LSTMs [1] and Highway Networks [2] to recent state space models [3], linear attention [4], and also softmax attention [5,6]. Yet, existing literature rarely examines the specific effects of gating. In this work, we conduct comprehensive experiments to systematically investigate gating-augmented softmax attention variants. Specifically, we perform a comprehensive comparison over 30 variants of 15B Mixture-of-Experts (MoE) models and 1.7B dense models trained on a 3.5 trillion token dataset. Our central finding is that a simple modification-applying an head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA)-consistently improves performance. This modification also enhances training stability, tolerates larger learning rates, and improves scaling properties. By comparing various gating positions and computational variants, we attribute this effectiveness to two key factors: (1) introducing non-linearity upon the low-rank mapping in the softmax attention, and (2) applying query-dependent sparse gating scores to modulate the SDPA output. Notably, we find this sparse gating mechanism mitigates 'massive activation' [7], 'attention sink' [8], and enhances long-context extrapolation performance, and we also release related codes and models to facilitate future research. Furthermore, the most effective SDPA output gating is used in the Qwen3-Next models.

Section: Introduction
Gating mechanism is well-established in neural networks. Early architectures, such as LSTMs [1], Highway Networks [2] and GRUs [9], pioneer the use of gating to control information flow across time steps or layers and improve gradient propagation. This principle persists in modern architectures. Recent sequence modeling works, including state-space models [3,10] and attention mechanisms [11,12,13,4,14,15,16,17,18,5,6] commonly apply gating, often to modulate the outputs of tokenmixer components. Despite its widespread adoption and empirical success, most recent works do not look into the gating mechanisms like the gating scores and their effect on the model's hidden states.
Insufficient understanding hinders assessing gating's true contribution, especially when confounded with other architectural factors. For instance, while Switch Heads [19,20] introduces a sigmoid gating to select top-K attention head experts, our experiments reveal an interesting finding (Appendix A.1): substantial performance gains persist even when reduced to a single expert, where the gate simply modulates the value output. This strongly suggests the gating itself provides significant intrinsic value, separate from the routing mechanism. Similarly, in Native Sparse Attention (NSA) [21], while overall performance improvements are demonstrated, they do not disentangle the contributions of its gating mechanism from the effects of the sparse attention design itself. These considerations underscore the need to rigorously disentangle the effects of gating from other architectural components. Performance comparison (Test PPL and MMLU) of 15B MoE models with gating applied at various positions. Gating after SDPA (G1) yields the best overall results. Gating after the Value layer (G2) also demonstrates notable improvements, particularly in PPL. Right: Training loss comparison (smoothed, 0.9 coeff.) over 3T tokens between baseline and SDPA-gated 1.7B dense models under identical hyperparameters. Gating results in lower final loss and substantially enhanced training stability, mitigating loss spikes. This stability allows for potentially higher learning rates and facilitates better scaling.
In this work, we investigate gating mechanisms in the standard softmax attention [22] (Sec.2.2). Specifically, we introduce gating at distinct positions (Fig. 1): after the query (G 4 ), key (G 3 ), and value projections (G 2 ); following the Scaled Dot Product Attention (SDPA) outputs (G 1 ); and after the final dense output layer (G 5 ). Our exploration covers gating variants including elementwise and headwise, head-specific and head-shared, as well as additive and multiplicative forms. We find that: (i) applying SDPA output head-specific gating (G 1 ) yields the most significant performance improvements (e.g., up to 0.2 PPL reduction and 2 points on MMLU); (ii) the SDPA output gating also improves training stability, nearly eliminating loss spikes, enabling larger learning rates and enhancing model scalability.
We identify two factors contributing to the efficacy of gating: (i) Non-Linearity. The two consecutive linear layers -the value (W v ) and dense (W O ) projections -can be rewritten into one low-rank linear projection. Therefore, introducing non-linearity through gating at positions G 1 or G 2 can increase the expressiveness of this low-rank linear transformation (Sec. 4.1). (ii) Sparsity. Although non-linear gating variants consistently enhance performance, we observe that their gains vary. Our analysis further reveals that the pronounced sparsity of the gating scores is another crucial factor, introducing input-dependent sparsity to SDPA outputs (Sec. 4.2). Sparse alse gating eliminates the massive activation [7] and attention sink [8]: the initial tokens have large activation values in the corresponding hidden states (Tab. 4) and disproportionately dominate attention scores (Fig. 2, Sec. 4.3). Previous work [8,7,23] explains attention sinks as an accumulation of redundant attention due to non-negative softmax normalization. Empirically, we verify that when query-dependent sparse gating is applied at the SDPA output, both our dense and MoE models (trained on 3.5T tokens) exhibit no attention sink. Furthermore, these models demonstrate superior performance in length generalization, achieving a gain of over 10 points on RULER [24](Sec.4.4).
this section cite: ['b0', 'b1', 'b8', 'b2', 'b9', 'b10', 'b11', 'b12', 'b3', 'b13', 'b14', 'b15', 'b16', 'b17', 'b4', 'b5', 'b18', 'b19', 'b20', 'b21', 'b6', 'b7', 'b7', 'b6', 'b22', 'b23']

Section: Practical Recommendation.
For best results, apply elementwise SDPA gating G 1 (i.e., gating after the attention-weighted value projection) and train with a moderately increased learning rate.
this section cite: []

Section: Gated-Attention Layer

this section cite: []

Section: Preliminary: Multi-Head Softmax Attention
Given an input X ∈ R n×dmodel , where n is the sequence length and d model is the model dimension, the computation of transformer's attention layer [22] could be divided into four stages.
this section cite: ['b21']

Section: QKV Linear Projections:
The input X is linearly transformed into queries Q, keys K, and values V using learned weight matrices W Q , W K , W V ∈ R dmodel×d k and Q, K, V ∈ R n×d k :
Q = XW Q , K = XW K , V = XW V .(1)
Scaled Product Dot-Product Attention (SDPA): computes attention scores between queries and keys, followed by a softmax normalization. The output is a weighted sum of the values:
Attention(Q, K, V ) = softmax QK T √ d k V,(2)
where QK T √ d k ∈ R n×n represents the scaled dot-product similarity matrix, and softmax(•) ensures the attention weights are no-negative and sum to 1 across each row.
this section cite: []

Section: Multi-Head Concatenation:
In multi-head attention, the above process is repeated for h heads, with each head having its projection matrices
W i q , W i k , W i v .
All heads' outputs are concatenated:
MultiHead(Q, K, V ) = Concat(head 1 , . . . , head h ),(3)
where
head i = Attention(QW i Q , KW i K , V W i V ). Final Output Layer: The concatenated SDPA output is passed to the output layer W o ∈ R hd k ×dmodel : O = MultiHead(Q, K, V )W o .(4)
this section cite: []

Section: Augmenting Attention Layer with Gating Mechanisms
The gating mechanism is formalized as:
Y ′ = g(Y, X, W θ , σ) = Y ⊙ σ(XW θ ),(5)
where Y is the input to be modulated, X is another input used to compute the gating scores 1 , W θ refers to the learnable parameters of gate, σ is an activation function (e.g., sigmoid), and Y ′ is the gated output. The gating score, σ(XW θ ), effectively acts as a dynamic filter, controlling the information flow from Y by selectively preserving or erasing its features.
This work comprehensively investigates variants of gating mechanisms within the attention layer.
Our exploration focuses on five key aspects: (5) Activation Function. We mainly consider two common activation functions: SiLU [25] and sigmoid. We only use SiLU for additive gating due to its unbounded output range, and sigmoid only gives scores in [0, 1]. Additionally, to further dissect the mechanisms underlying gating's effectiveness, we also consider Identity Mapping or RMSNorm [26] (detailed in Sec 4.1). Unless otherwise specified, we employ head-specific, multiplicative gating utilizing the sigmoid activation function (σ(x) = 1 1+e -x ).
3 Experiments
this section cite: ['b24', 'b25']

Section: Experimental Setups

this section cite: []

Section: Model Architecture and Training Settings
We conduct experiments on both MoE models (15B total parameters with 2.54B activated, 15A2B) and dense models (1.7B total parameters). The 15A2B MoE models utilize 128 total experts with top-8 softmax gating, fine-grained experts [27], global-batch LBL [28], and z-loss [29]. We adopt group query attention (GQA) [30] for the attention part. More detailed model architecture configurations are discussed in Appendix A.2. We train the models on subsets of a 4T high-quality tokens, encompassing multilingual, math, and general knowledge content. A sequence length of 4096 is used. More detailed configurations, such as learning rate and batch size (bsz), will be introduced in each part. Other hyperparameters follow the default values of the AdamW optimizer. Since the parameters and flops introduced by the gating are relatively small, the wall-time latency introduced by gating is less than 2%. Evaluation We test the few-shots results on popular benchmarks, including, Hellaswag [31] for English, MMLU [32] for general knowledge, GSM8k [33] for math reasoning, HumanEval [34] for coding, C-eval [35] and CMMLU [36] for Chinese proficiency. We also test the perplexity (PPL) on diverse held-out test sets, including domains like English, Chinese, Code, Math, Law and Literature.
this section cite: ['b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35']

Section: Main Results

this section cite: []

Section: Gated Attention for MoE models
We first compare different gatings on the training-efficient MoE-15A2B models. All models use a scheduler that warms up to a maximum LR of 2e-3 in 1k steps and decays using cosine to 3e-5. We use a global bsz of 1024, comprising 100k optimization steps. The results are summarized in Tab. 1. To provide a fair comparison, we supplement the vanilla MoE baseline (row 1) with parameter expansion methods, including increasing the number of key-value heads (row 2), increasing the number of query heads (row 3), and increasing both the total and activated number of experts (row 4). These methods introduce a comparable or greater number of parameters than the gating mechanisms. Overall, adding gating at the value layer (G 2 ) and SDPA output (G 1 ) reduces PPL by more than 0.2, outperforming various parameter-expanding baselines. However, gating at G 1 achieves better PPL and benchmark results. As long as different heads receive distinct gating scores, the granularity of gating and the choice of activation function have relatively minor impacts. We will further analyze the reasons behind these observations in Analysis (Sec 4.2).
this section cite: []

Section: Gated Attention for Dense Models.
We also conduct experiments on dense models following [37] to validate SDPA output sigmoid gating. When using gating, we reduce the width of FFN to maintain the parameter size. Most experiments use optimized hyperparameters for the baseline. For instance, for the 1.7B model trained on 400B tokens, we use a maximum LR of 4e-3 and a bsz of 1024. For training on 3.5T tokens, we increase the maximum LR to 4.5e-3 and the bsz to 2048. Prior work has established that while increased network depth, large learning rates, and large batch sizes can significantly improve model performance [38,39,40] and distributed training efficiency, they often introduce training instabilities [39,41,42]. We observe that applying gating largely reduces the loss spikes [43,42] during training (Fig. 2 right), suggesting a promising role for gating in enhancing training stability. Therefore, we introduce another setting characterized by an increased number of layers, a higher maximum learning rate, and a larger batch size to further probe gating's stabilizing effects.
Tab. 2 6,12). While adding sandwich norm [44] restores convergence, the improvement is negligible. In contrast, increasing the maximum LR in models with gating results in a noticeable improvement. In summary, we identify SDPA element-wise gating as the most effective method to augment the attention mechanism. Incorporating the SDPA output gate enables stable training under larger learning rates and batch sizes-regimes where the baseline often becomes unstable. This suggests that the optimal hyperparameter configuration shifts when using gating. In practice, one effective way to leverage the gate is to start from the baseline's optimal batch size and moderately increase the learning rate. Further jointly tuning batch size and learning rate may yield additional gains.
this section cite: ['b36', 'b37', 'b38', 'b39', 'b38', 'b40', 'b41', 'b42', 'b41', 'b5', 'b11', 'b43']

Section: Analysis: Non-Linearity, Sparsity, and Attention-Sink-Free
In this section, we conduct a series of experiments to explore why such a simple gating mechanism can yield significant improvements in performance and training stability. Here are the takeaways according to our analysis: (1) Gatings enhancing non-linearity consistently lead to performance gains (Sec 4.1); (2) The most effective SDPA elementwise gate introduces strong input-dependent sparsity (Sec 4.2), which then helps to eliminate the 'massive activation' and 'attention sink' phenomenon. Inspired by prior works that utilize group norm for the SDPA output [14,45], with the same setting in Sec. 3.2.1, we apply RMSNorm [26] independently to the output of each attention head before concatenation. As shown in Tab. 3 row 5, applying RMSNorm, which introduces almost no additional parameters, also leads to a significant reduction in PPL.
this section cite: ['b13', 'b44', 'b25']

Section: Non-linearity Improves the Expressiveness of Low-Rank Mapping in Attention
In multi-head attention, the output of the i-th token, corresponding to the k-th head, can be expressed:
o k i = ( i j=0 S k ij • X j W k V )W k O = i j=0 S k ij • X j (W k V W k O ),(6)
where W k O is the parameters of the output layer W O corresponding to the k-th headfoot_1 . Here, S k ij is the attention score of the i-th token attending to the j-th token in the k-th head, X j is the input to the attention for token j, and X j W k V is the value output of token j in the k-th head. From Equ. 6, we can merge W k V W k O into one low-rank linear mapping applied over all X j as d k < d model . With GQA, W V is shared among heads within the same group, further diminishing the expressiveness.
Given that adding non-linearity between two linear mappings can improve their expensiveness [46], we have two modifications to mitigate the low-rank problem:
o k i = i j=0 S k ij • Non-Linearity-Map(X j W k V ) W k O ,(7)
o k i = Non-Linearity-Map i j=0 S k ij • X j W k V W k O .(8)
Notably, adding gating at the G 2 (Tab. 3 row 3) position corresponds to the first modification (Equ. 7), while adding gating (row 4) or group normalization (row 5) at the G 1 position corresponds to the second (Equ. 8). This also explains why adding gating or normalization at the G 5 position after W O has no effect (Tab. 1 row 9) -it does not address the lack of non-linearity between W V and W O .
For additive gating at G 1 , the output of gating passes through SiLU (Tab. 3 row 4), also introducing some non-linearity, which explains the observed performance gains, albeit smaller than those achieved by multiplicative gating. Based on these insights, we conduct two additional experiments: (i) Adding SiLU only at the G 1 position without introducing additional parameters (Tab. 3 row 6). Notice this simple modification also leads to a modest reduction in PPL, but most benchmark scores remain unchanged. (ii) Removing SiLU from additive gating, such that the output of X j after gating is directly added at the G 1 position (Tab. 3 row 7). This further diminishes the gains of addictive gating.
In summary, the enhanced performance associated with effective gating variants is likely attributable to the introduction of non-linearity between W V and W O . Although applying gating at positions G 1 and G 2 can can both introduce this non-linearity, these applications yield differing performance gains. This observed difference motivates us to further analyze the impacts of gating at these two positions.
this section cite: ['b45']

Section: Gating Introduces Input-Dependent Sparsity
We analyze the gating scores (Tab. 1, 'Gate Score' column) of models with gating applied at the value (G 2 ) and SDPA output (G 1 ) positions, evaluated on the test language modeling data. The mean gating scores for all layers are presented in Table 4, with the score distributions visualized in Fig. 3 (layer-wise scores in Appendix A.3). Key observations include:
(i) Effective Gating Scores are Sparse. SDPA output gatings (Elementwise/headwise) exhibit the lowest mean gating scores. Furthermore, the SDPA output gating score distribution shows a high concentration near 0, indicating substantial sparsity, consistent with its superior performance.
(ii) Head-Specific Sparsity Matters. Enforcing shared gating scores across attention heads increases the overall gating scores and diminishes performance gains. Observations (i) and (ii) underscore the importance of head-specific gating, aligning with previous research demonstrating that individual attention heads capture distinct aspects of the input [47,48,49,50].
(iii) Query-Dependency Matters. The scores for value gating (G 2 ) are higher than those for SDPA output gating (G 1 ), and the performance is inferior. This suggests that gating score sparsity is more effective when query-dependent rather than determined by the key and value. Specifically, SDPA output gating scores are derived from the hidden states corresponding to the current query (e.g. the Non-Linearity-Map in Eq 8 depends on X i ), whereas value gating scores are derived from hidden states associated with past keys and values (e.g. the Non-Linearity-Map in Eq 7 depends on each X j ). This implies that gating score sparsity may filter out irrelevant contextual information for the query.
To further validate the importance of query-dependency, we introduce input-independent gating by zero-initializing learnable parameters (q × d k ), applying a sigmoid function, and multiplying it with the SDPA output. As shown in row (6), input-independent gating improves upon the baseline, likely due to the introduction of non-linearity. Moreover, the high gating scores reinforce that effective sparsity should be input-dependent.
Table 4: Performance of different gating methods with varying activation functions and average gate scores.
'Act-Func' refers to the activation function used for computing the gating scores, while 'M-Act' denotes the rounded maximum activation values of the hidden states output by each layer of the model. Additionally, 'F-Attn' represents the attention score of the first token, with higher values indicating more pronounced 'attention sink'.
this section cite: ['b46', 'b47', 'b48', 'b49']

Section: Method Act-Func Gate Score M-Act F-Attn PPL Hellaswag MMLU GSM8k
Baseline --1053 0.467 6.026 73.07 58.79 52.92 (2) SDPA Elementwise Gate Sigmoid 0.116 94 0.048 5.761 74.64 60.82 55.27 (3) SDPA Headwise Gate Sigmoid 0.172 98 0.073 5.792 74.50 60.05 54.44 (4) SDPA Elementwise Head-shared Gate Sigmoid 0.271 286 0.301 5.801 74.34 60.06 53.15 (5) v Elementwise Gate Sigmoid 0.221 125 0.297 5.820 74.38 59.17 51.33 (6) SDPA Input Independent Gate Sigmoid 0.335 471 0.364 5.917 73.64 59.02 52.40 (7) SDPA Elementwise Gate NS-sigmoid 0.653 892 0.451 5.900 74.05 60.05 52.75 0.0 0.2 0.4 0.6 0.8 1.0 Gating Score 0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 Normalized Density SDPA_Elementwise_Gate Scores Distribution Mean = 0.116 Distribution 0.0 0.2 0.4 0.6 0.8 1.0 Gating Score 0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 Normalized Density v_Elementwise_Gate Scores Distribution Mean = 0.221 Distribution 0.0 0.2 0.4 0.6 0.8 1.0 Gating Score 0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 Normalized Density SDPA_Elementwise_HS_Gate Scores Distribution Mean = 0.271 Distribution
which constrains the gating scores between [0.5, 1.0]. This ensures introducing non-linearity while removing gating score sparsity. As shown in Tab. 4 row (7), the gains of NS-sigmoid gating are inferior to those of SDPA output sigmoid gating. In Appendix A.3, we provide a more detailed discussion on how sparse gating scores affect the sparsity (the proportion of values below the threshold) in SDPA hidden states. We will discuss the impact of different sparsity levels on model behavior, including reducing the 'attention sink', in the next section.
this section cite: []

Section: SDPA Output Gating Reduces Massive Activation and Attention-Sink
Based on the observation that gating introduces sparsity to the SDPA output in an input-dependent manner, we hypothesized that this mechanism can filter out context irrelevant to the current query token, thereby mitigating the attention sink [8,7]. Correspondingly, early work [51] finds that in explicit top-k sparse attention-where only the most relevant tokens are attended to-the attention sink phenomenon does not occur. To verify this, we analyze the distribution of attention scores (averaged over all heads) and the proportion of attention scores allocated to the first token (Fig. 2, Tab. 4, 'F-Attn' column). Massive activations [7] (large values in hidden states) are belied to lead to the attention sink to their corresponding tokens. Inspired by this, we also compute the mean of the maximum hidden state activations across layers, as shown in the 'M-Act' column of Tab. 4. More detailed layer-wise results are provided in the Appendix A.4.
We can observe: (i) Head-wise and element-wise query-dependent sigmoid gating at the SDPA output (G 1 ) largely reduces the attention score allocated to the first token and decreases massive activations.
(ii) Enforcing shared gating scores across heads or applying gating only after the value projection (G 2 ) decreases massive activations, but does not reduce attention scores to the first token. This reinforces the importance of head-specific gating and suggests that massive activations are not a necessary condition for attention sinks. (iii) Reducing the input-dependence of gating (row 6) or using NS-sigmoid to reduce sparsity (row 7) intensifies both massive activations and attention sink.
Collectively, these observations indicate that input-dependent, head-specific gating of the SDPA output introduces significant sparsity, thereby mitigating the attention sink. Furthermore, sparsity in the SDPA outputs reduces massive activations within the model, with increased sparsity leading to smaller activations. This may explain the improved training stability with gating: by reducing massive activations, the model is less susceptible to numerical errors during BF16 training [52]. We also observe that massive activations originate primarily from early layers (e.g., layer 5), where the FFN outputs large values, consistent with [53]. Once added to the residual stream, these activations are propagated through subsequent layers via the pre-norm mechanism. This aligns with the effectiveness of sandwich normalization [44] in enhancing training stability (Table 2, row 7): applying LayerNorm to the FFN output prevents these large activations from entering the residual stream. Based on the attentionsink-free pattern, we evaluate the SDPA gating's effect in the longcontext setting. Specifically, we extend the context length for the models trained on 3.5T tokens. We increase the RoPE [54] base from 10k to 1M and continue training on data with a sequence length of 32k for an additional 80B tokens. This gives us models with a context length of 32k. Subsequently, we use YaRN [55] to extend the context length to 128k. We evaluate models on the RULER benchmark [24] and summarize results in Tab. 5. We observe the following: (i) Under the 32k setting, models with gating slightly outperform the baseline. This suggests that within the training length, the attention sink phenomenon may not hurt the model's long-context performance. (ii) When the context length is extended to 128k using YaRN, both the baseline and gated models experience a decline within the original 32k range. This observation is consistent with previous works on extending context length by modifying RoPE [56,55,57]. Even though the decline is less pronounced for models with gating. (iii) At context lengths of 64k and 128k, the gated attention models outperform the baseline signifantly. From these observations, we hypothesize that adding gating helps the model adapt to the context-length extension. A possible explanation is that baseline models rely on attention sinks to adjust the distribution of attention scores. [57] derives the effects of changing the RoPE based on the attention and hidden state distributions. When techniques like YaRN are applied to modify the RoPE base, the attention sink pattern may struggle to adapt in a training-free manner, leading to a noticeable drop in performance. In contrast, models with gating primarily rely on input-dependent gating scores to control information flow, making them more robust to such changes.
this section cite: ['b7', 'b6', 'b50', 'b6', 'b51', 'b52', 'b43', 'b53', 'b54', 'b23', 'b55', 'b54', 'b56', 'b56']

Section: SDPA Output Gating Facilitates Context Length Extension

this section cite: []

Section: Related Works

this section cite: []

Section: Gating in Neural Networks
Gating mechanisms have been widely adopted in neural networks. Early works such as LSTMs [1] and GRUs [9] introduce gates to regulate information flow across time steps, addressing gradient vanishing/exploding issues by selectively retaining or discarding information. Highway Networks [2] extend this concept to feedforward networks, enabling the successful training of very deep architectures. SwiGLU [25] introduce gating mechanisms into transformer FFN layers, enhancing their expressive power and becoming a standard component in many open-source LLMs [58,37].
Several works on state-space models [3,10,59] and Linear Attention, such as FLASH [4], RetNet [14], Lightning Attention [17,60,61], and Gated Delta Networks [18], also incorporate gating modules to controlinformation of token-mixer modules. AlphaFold2 [5] and Forgetting Transformer [6] introduce gating mechanisms to the output of softmax attention. GaAN [11] uses gating to control each attention head's importance for learning on graphs. Some works [62,13,63,15,16,64,65,66] also apply operations similar to gating to augment softmax attention. Attention on Attention (AoA) [12] also modulates the attention output with a sigmoid gating, depending on the query. Although these works demonstrate the effectiveness of gating, a comprehensive understanding of its precise mechanisms and the reasons behind its effectiveness still needs exploration. This could contribute to a broader appreciation of gating's importance beyond RNNs and facilitate designs that better leverage gating's unique advantages. For example, while Switch Heads [20,19], NSA [21], and MoSA [67] employ sigmoid-based gating [68] for selection, further investigation into isolating gating's specific contribution could offer valuable insights. Comparisons with baselines incorporating similar gating mechanisms in standard transformers could offer a more refined perspective on the effectiveness of their proposed selection mechanisms. The work most closely related to ours is Quantizable Transformers [69], which also finds that applying gating in softmax attention alleviates extreme attention concentration and outliers in hidden states in encoder models like BERT and ViT. While this work primarily leverages gating to eliminate outliers for model quantization, we provide a detailed analysis of various gating variants, uncovering their benefits through enhanced non-linearity and sparsity, as well as improved training stability. Building on these insights, we scale up gated attention models, demonstrating gating's broad applicability and impact.
this section cite: ['b0', 'b8', 'b1', 'b24', 'b57', 'b36', 'b2', 'b9', 'b58', 'b3', 'b13', 'b16', 'b59', 'b60', 'b17', 'b4', 'b5', 'b10', 'b61', 'b12', 'b62', 'b14', 'b15', 'b63', 'b64', 'b65', 'b11', 'b19', 'b18', 'b20', 'b66', 'b67', 'b68']

Section: Attention Sink
StreamingLLM [8] formally identifies 'attention sink', in which specific tokens receive large attention scores. Similarly, in ViT, some redundant tokens act as 'registers' to store attention scores [70]. Later, Massive Activation [7] shows that excessive attention scores are assigned to tokens associated with massive activation values. However, our work reveals that value output (G 2 ) gating eliminates massive activations, yet attention sinks persist, indicating that massive activations are necessary for attention sinks. Similarly, attention sinks are characterized as non-informative 'key biases' that store redundant attention scores, arguing that softmax's inherent normalization dependency drives this behavior [23]. Experimental attempts to modify softmax attention, such as replacing softmax with unnormalized sigmoid attention [71,23], using explicit top-k sparse attention [51], adding softmax attention gate or clip [69], calibrating attention scores [72], and modifying softmax computation [73] and denominator [74], show promise in mitigating attention sinks. Another stream of works try to move the sink tokens from input tokens to manually added components, like 'registers' [70], 'meta tokens' [75] and learnable 'sink' [76]. Our work shows that sparse gating after SDPA eliminates attention sinks in both dense (1B-parameter) and MoE (15B-parameter) models, even when trained on 3.5T tokens. Furthermore, we uncover the potential of eliminating attention sinks to benefit context-length extension.
this section cite: ['b7', 'b69', 'b6', 'b22', 'b70', 'b22', 'b50', 'b68', 'b71', 'b72', 'b73', 'b69', 'b74', 'b75']

Section: Conclusion and Limitations
This work systematically investigates gating mechanisms in softmax-attention, revealing their significant impact on performance, training stability, and attention dynamics. This simple mechanism enhances non-linearity, introduces input-dependent sparsity, and eliminates 'attention sink'. Additionally, gating facilitates context length extension, allowing models to generalize effectively to longer sequences without retraining. We will release the 'attention-sink-free' models, providing a foundation for future research into attention mechanisms.
The broader implications of non-linearity on the dynamics of attention and the overall training process remain under-explored. We don't provide a theoretical explanation for how attention sinks influence the model's ability to generalize to longer sequences.
this section cite: []

Section: References
Ref_id:b0 Title: Long short-term memory Year: (1997)
Ref_id:b1 Title:  Year: (2015)
Ref_id:b2 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b3 Title: Transformer quality in linear time Year: (2022-07-23)
Ref_id:b4 Title: Highly accurate protein structure prediction with alphafold Year: (2021)
Ref_id:b5 Title: Forgetting transformer: Softmax attention with a forget gate Year: (2025)
Ref_id:b6 Title: Massive activations in large language models Year: (2024)
Ref_id:b7 Title: Efficient streaming language models with attention sinks Year: (2023)
Ref_id:b8 Title: Gate-variants of gated recurrent unit (gru) neural networks Year: (2017)
Ref_id:b9 Title: Transformers are ssms: Generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b10 Title: Gaan: Gated attention networks for learning on large and spatiotemporal graphs Year: (2018)
Ref_id:b11 Title: Attention on attention for image captioning Year: (2019)
Ref_id:b12 Title: Not all attention is needed: Gated attention network for sequence data Year: (2020)
Ref_id:b13 Title: Retentive network: A successor to transformer for large language models Year: (2023)
Ref_id:b14 Title: Improving transformers with dynamically composable multi-head attention Year: (2024)
Ref_id:b15 Title: Selective attention: Enhancing transformer through principled context control. Advances in Neural Information Processing Systems Year: (2024)
Ref_id:b16 Title: Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models Year: (2024)
Ref_id:b17 Title: Gated delta networks: Improving mamba2 with delta rule Year: (2024)
Ref_id:b18 Title: Moeut: Mixture-of-experts universal transformers Year: (2024)
Ref_id:b19 Title: Switchhead: Accelerating transformers with mixture-of-experts attention Year: (2024)
Ref_id:b20 Title: Native sparse attention: Hardware-aligned and natively trainable sparse attention Year: (2025)
Ref_id:b21 Title: Attention is all you need Year: (2017)
Ref_id:b22 Title: When attention sink emerges in language models: An empirical view Year: (2024)
Ref_id:b23 Title: Ruler: What's the real context size of your long-context language models Year: (2024)
Ref_id:b24 Title: Glu variants improve transformer Year: (2020)
Ref_id:b25 Title: Root mean square layer normalization Year: (2019)
Ref_id:b26 Title: Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models Year: (2024)
Ref_id:b27 Title: Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models Year: (2025)
Ref_id:b28 Title: St-moe: Designing stable and transferable sparse expert models Year: (2022)
Ref_id:b29 Title: Gqa: Training generalized multi-query transformer models from multi-head checkpoints Year: (2023)
Ref_id:b30 Title: Hellaswag: Can a machine really finish your sentence Year: (2019)
Ref_id:b31 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b32 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b33 Title:  Year: (2021)
Ref_id:b34 Title: C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models Year: (2024)
Ref_id:b35 Title: Cmmlu: Measuring massive multitask language understanding in chinese Year: (2023)
Ref_id:b36 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b37 Title: An empirical model of large-batch training Year: (2018)
Ref_id:b38 Title: Deepnet: Scaling transformers to 1,000 layers Year: (2022)
Ref_id:b39 Title: Aditya Vardhan Varre, and Nicolas Flammarion. Why do we need weight decay in modern deep learning? Year: (2024)
Ref_id:b40 Title: Glm-130b: An open bilingual pre-trained model Year: (2022)
Ref_id:b41 Title: Spike no more: Stabilizing the pre-training of large language models Year: (2023)
Ref_id:b42 Title: Palm: Scaling language modeling with pathways Year: (2023)
Ref_id:b43 Title: Cogview: Mastering text-to-image generation via transformers Year: (2021)
Ref_id:b44 Title:  Year: (2024)
Ref_id:b45 Title: On the number of linear regions of deep neural networks Year: (2014)
Ref_id:b46 Title: Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned Year: (2019)
Ref_id:b47 Title: Spatten: Efficient sparse attention architecture with cascade token and head pruning Year: (2021)
Ref_id:b48 Title: -context learning and induction heads Year: (2022)
Ref_id:b49 Title: Smarttrim: Adaptive tokens and attention pruning for efficient vision-language models Year: (2023)
Ref_id:b50 Title: Explicit sparse transformer: Concentrated attention through explicit selection Year: (2019)
Ref_id:b51 Title: Numerical error analysis of large language models Year: (2025)
Ref_id:b52 Title: Interpreting the repeated token phenomenon in large language models Year: (2025)
Ref_id:b53 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b54 Title: Yarn: Efficient context window extension of large language models Year: (2023)
Ref_id:b55 Title: Extending context window of large language models via positional interpolation Year: (2023)
Ref_id:b56 Title: Longred: Mitigating short-text degradation of long-context large language models via restoration distillation Year: (2025)
Ref_id:b57 Title: The llama 3 herd of models Year: (2024)
Ref_id:b58 Title: Stuffed mamba: Oversized states lead to the inability to forget Year: (2024)
Ref_id:b59 Title: Various lengths, constant speed: Efficient language modeling with lightning attention Year: (2024)
Ref_id:b60 Title: Minimax-01: Scaling foundation models with lightning attention Year: (2025)
Ref_id:b61 Title:  Year: (2020)
Ref_id:b62 Title: Multi-step deductive reasoning over natural language: An empirical study on out-of-distribution generalisation Year: (2022)
Ref_id:b63 Title: Hard-attention gates with gradient routing for endoscopic image computing Year: (2024)
Ref_id:b64 Title:  Year: (2025)
Ref_id:b65 Title: Star with bilinear mapping Year: (2025-06)
Ref_id:b66 Title: Mixture of sparse attention: Contentbased learnable sparse attention via expert-choice routing Year: (2025)
Ref_id:b67 Title: Approximating two-layer feedforward networks for efficient transformers Year: (2023)
Ref_id:b68 Title: Quantizable transformers: Removing outliers by helping attention heads do nothing Year: (2023)
Ref_id:b69 Title: Vision transformers need registers Year: (2024)
Ref_id:b70 Title: Theory, analysis, and best practices for sigmoid self-attention Year: (2024)
Ref_id:b71 Title: Unveiling and harnessing hidden attention sinks: Enhancing large language models without training through attention calibration Year: (2024)
Ref_id:b72 Title: Softpick: No attention sink, no massive activations with rectified softmax Year: (2025)
Ref_id:b73 Title: Attention is off by one Year: (2023)
Ref_id:b74 Title: A hybrid-head architecture for small language models Year: (2024)
Ref_id:b75 Title: OpenAI. gpt-oss-120b & gpt-oss-20b model card Year: (2025)
