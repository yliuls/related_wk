Title: UMoE: Unifying Attention and FFN with Shared Experts
Abstract: Sparse Mixture of Experts (MoE) architectures have emerged as a promising approach for scaling Transformer models. While initial works primarily incorporated MoE into feed-forward network (FFN) layers, recent studies have explored extending the MoE paradigm to attention layers to enhance model performance. However, existing attention-based MoE layers require specialized implementations and demonstrate suboptimal performance compared to their FFN-based counterparts. In this paper, we aim to unify MoE designs in attention and FFN layers by introducing a novel reformulation of the attention mechanism, that reveals an underlying FFN-like structure within attention modules. Our proposed architecture, UMoE, achieves superior performance through attention-based MoE layers while enabling efficient parameter sharing between FFN and attention components.

Section: Introduction
Scaling plays a crucial role in advancing the capabilities of large language models [1,2,3]. However, this scaling advantage comes with substantial computational costs, making continued scaling increasingly impractical. Sparse Mixture-of-Experts (MoE) architectures have emerged as a promising solution by selectively activating only a subset of model parameters-termed experts-for each input [4,5,6,7]. This approach effectively decouples model size from computational cost, enabling efficient scaling with minimal overhead.
Recent work has demonstrated the effectiveness of MoE in Transformer architectures [8,9,10,5,11], particularly when applied to feed-forward neural network (FFN) layers. Building on this success, several studies have explored extending MoE to attention layers [12,13,14], indicating potential for performance gains through attention scaling. Despite the potential, we find that existing MoE attention layers demonstrate suboptimal performance compared to FFN-MoE approaches, when provided with similar computational and parametric budgets. This performance gap challenges the practical utility of attention-MoE architectures, as parameters allocated to scaling attention layers might be more effectively utilized for scaling FFNs instead.
We identify two distinctions between attention-MoE and FFN-MoE implementations that likely account for the observed performance differential: (1) the different expert design between attention and FFN layers, and (2) attention-MoE's necessity to compromise the expressiveness of vanilla attention mechanisms to accommodate sparse computation [12]. Motivated by these observations, we investigate a compelling question: can we reformulate attention to reveal an underlying structure compatible with the same expert design as FFN layers, without compromising the expressive power  of the attention mechanism? This is a challenging question due to the inherent complexity of attention mechanisms, including multiple projections and softmax calculations, which fundamentally differ from the straightforward two-matrix multiplication pattern of FFNs.
To bridge this structural gap, we reformulate the attention mechanism to reveal its underlying FFN-like structure. Our reformulation decomposes attention into two sequential operations: token mixing and token-wise expert processing. The token-wise expert processing, consisting of two consecutive matrix multiplications, can be implemented as an FFN with a small intermediate size. This implementation naturally aligns with recent advances in fine-grained FFN expert design [6,15,3], enabling unified expert architectures and parameter sharing across both attention and FFN layers.
Based on this insight, we introduce UMoE, a unified MoE architecture that abstracts Transformer layers into three fundamental components: experts, token mixing operations, and routers, as shown in Fig. 1. The experts, implemented as standard two-layer FFNs, serve as the primary components for token processing and knowledge storage. The token mixing operations facilitate contextual information exchange through weighted summation of tokens. Routers are employed to dynamically dispatch tokens to the most relevant experts to enable sparse computation. In UMoE, the distinction between the FFN and attention layers lies solely in the expert inputs: FFN layers process tokens independently, while attention layers process tokens simultaneously through weighted summation. This unified design not only simplifies the architecture but also enables parameter-efficient scaling through expert sharing between attention and FFN components.
To evaluate the effectiveness of UMoE, we conduct extensive experiments across various model sizes and tasks, including pre-training and zero-shot evaluations. With the reformulated attention mechanism, the attention-based MoE layers of UMoE match or exceed the performance of previous FFN-based MoE layers. Moreover, by sharing parameters across attention and FFN modules, UMoE achieves superior performance in fully MoE architectures while maintaining the same parameter count. We also present a detailed routing analysis of UMoE, revealing expert specialization patterns across modules, with higher-ranked experts demonstrating interpretable attention patterns. Our code is available at https://github.com/ysngki/UMoE.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b4', 'b10', 'b11', 'b12', 'b13', 'b0', 'b11', 'b5', 'b14', 'b2']

Section: Related Work
Sparse Mixture-of-Experts (MoE). Sparse Mixture-of-Experts (MoE) models have gained increasing attention for their ability to scale model capacity while maintaining computational effi-ciency [4,9,10,5,11]. The core component of these models is the sparsely activated MoE sub-layer, which selectively activates different parameter subsets for different inputs. In recent Transformerbased implementations, MoE architectures primarily replace feed-forward network (FFN) layers with MoE sub-layers. Each MoE layer consists of a collection of experts, denoted as {E i } N i=1 , where each expert E i is implemented as an FFN. Tokens are routed to a subset of experts through a routing mechanism, with the top-k router [4] being the most prevalent approach. Despite advances in routing mechanisms [16,17,18,19,15], the top-k router remains widely adopted due to its simplicity and robust performance [9]. For a given token x ∈ R d , where d is the hidden dimension, and a trainable weight matrix W r ∈ R N ×d , the top-k router computes the probability distribution over experts as:
p = softmax(W r x).(1)
The set of top-k experts T is then selected based on p, where |T | = k. Each expert processes the token independently and the final output of the MoE layer is computed as the weighted combination of these k experts' outputs:
y = i∈T p i E i (x),(2)
where each expert is implemented as an FFN with two matrices and a non-linear activation function.
this section cite: ['b3', 'b8', 'b9', 'b4', 'b10', 'b3', 'b15', 'b16', 'b17', 'b18', 'b14', 'b8']

Section: MoE for Attention.
Several recent approaches have explored extending the MoE paradigm to attention layers in Transformers [12,13], with a primary focus on expert design. Because attention layers lack the consecutive matrix multiplication pattern found in FFNs, these approaches necessitate expert designs that differ from FFN-MoE models. The Mixture-of-Attention (MoA) [12] propose to conceptualize individual attention heads as experts, scaling attention layers by increasing the number of attention heads. However, introducing sparsity into attention layers presents a significant challenge: query vectors computed by a specific expert (or head) require corresponding key and value vectors from the same expert, necessitating identical expert activation across all tokens. To address this constraint, MoA implements distinct query and output projections per head while maintaining shared key and value projections across attention heads.
SwitchHead [13] presents an alternative approach to implementing the MoE paradigm in attention layers. Rather than treating entire attention heads as experts, SwitchHead designates individual projection matrices within heads as experts. A straightforward implementation maintains four separate MoE sub-layers per head for query, key, value, and output projections. While scaling all projections yields performance improvements, empirical results show that value and output projections benefit most significantly from scaling.
In contrast to these approaches, UMoE unifies attention-MoE and FFN-MoE through a novel reformulation of the multi-head attention mechanism, enabling the shared expert design and parameters across both attention and FFN layers.
Other Related Work. Several studies have explored connections between MoE and attention from different perspectives. MoH [20] proposes using MoE for pruning attention heads in LLMs by continuing pre-training with a routing function. During inference, certain output projections (W o ), viewed as experts, are selectively skipped based on routing decisions. Taking a different approach, MH-MoE [21] incorporates concepts from multi-head attention to enhance FFN-based MoE models. Instead of routing original input tokens to experts, MH-MoE decomposes each token into multiple low-dimensional sub-tokens, which are then processed in parallel by diverse sets of experts.
this section cite: ['b11', 'b12', 'b11', 'b12', 'b19', 'b20']

Section: Method
The attention mechanism is the core of Transformers [22], processing token hidden states to capture contextual relationships. However, its structure differs from FFN layers, which complicates the unification of MoE designs across both modules. In this section, we present two alternative formulations of attention, pre-mixing and post-mixing, that reveal an inherent FFN-like structure within attention layers. Based on these formulations, we introduce a novel MoE architecture, UMoE.
def UMoELayer (x , X ) : # x : [1 , d ] , X : [n , d ] # ## Attention MoE indices , probs = TopKRouter ( x ) # Assign token x to Experts residual_x = x . copy () K = X @ W_k q_shared = x @ W_q for i , p in zip ( indices , probs ) : q = q_shared + x @ W_a [ i ] @ W_b [ i ] # K and V ( the hidden states X ) are shared across experts . y = Attention ( Q =q , K =K , V = X ) residual_x += p * Experts [ i ]( y ) x = residual_x # ## FFN MoE indices , probs = TopKRouter ( x ) # Assign token x to Experts residual_x = x . copy () for i , p in zip ( indices , probs ) : residual_x += p * Experts [ i ]( x ) return residual_x
Figure 3: Implementation details of a UMoE layer. The input consists of a sequence X containing n token hidden states and x representing the final hidden state. For simplicity, this implementation focuses on computing the output for the last token.
this section cite: ['b21']

Section: Formulations of Attention
Preliminaries. Consider a sequence of token hidden states X ∈ R n×d , where n is the sequence length and d is the hidden dimension. In multi-head attention, each token attends to all other tokens in the sequence through query, key, and value projections. For a single token x (e.g., the last token in the sequence for simplicity), its attention output is computed as:
q = xW q , K = XW k , V = XW v ,(3)
a = softmax qK ⊤ √ d k , o = aV,(4)
where W q , W k ∈ R d×d k and W v ∈ R d×dv are learnable matrices, respectively, and a ∈ R n is the attention weight. To enhance representation capacity, this process is repeated h times in parallel, and the outputs are combined:
y = [o 1 ; o 2 ; • • • ; o h ]W o ,(5)
where W o ∈ R hdv×d projects the concatenated outputs back to the original dimension d.
Pre-Mixing Formulation. While multi-head attention is typically expressed using concatenation, it can be equivalently expressed as a sum of per-head outputs, which helps reveal its connection to FFN layers. By decomposing W o into small matrices W i o ∈ R dv×d along the feature dimension, we can express the output as:
y = h i=1 o i W i o = h i=1 (a i XW i v )W i o (6
) = h i=1 (a i X)(W i v W i o ).(7)
This reformulation provides two distinct interpretations, as shown in Fig. 2: • Eq. 6:
The conventional view where value vectors are first aggregated then projected back into the hidden space with an output projection.
• Eq. 7: A new interpretation where token hidden states are first aggregated into contextualized representations, i.e., weighted averages of all tokens, before being processed by the value (W i v ) and output (W i o ) projections. We term this formulation as pre-mixing attention. While both interpretations yield same outputs, the pre-mixing formulation enables the grouping of W o and W v . This grouping reveals that pre-mixing attention exhibits a two-layer structure analogous to FFN modules, which can be implemented as a linear FFN with no activation function.
Post-Mixing Formulation. Alternatively, we can rearrange the computation as:
y = h i=1 a i (XW i v W i o ).(8)
In this formulation, token hidden states are transformed by two successive projections independently for each token, before being aggregated using the attention weights.
this section cite: []

Section: UMoE
By grouping W v and W o , both pre-mixing and post-mixing attention can be naturally interpreted as a MoE architecture, aligning with established FFN-MoE practices. Using pre-mixing attention as an example, let the expert E(x) := xW v W o . The multi-head attention can then be reformulated as:
y = h i=1 E i (a i X).(9)
By increasing the number of experts and introducing a routing mechanism, such as a top-k router, we derive a MoE architecture, denoted as UMoE-Att. The output of a UMoE-Att layer is:
y = i∈T p i E i (a i X),
where T is the set of activated experts.
Referring to Eq. 2, we observe that the primary distinction between FFN-MoE layers and UMoE-Att layers lies in their expert inputs: FFN experts operate on individual token hidden states x, while attention experts process weighted combinations of all token hidden states. This reveals a relationship: FFN-MoE layers can be interpreted as a specialized case of pre-mixing attention layers where the attention matrix is constrained to an identity matrix, limiting each token to self-attention only.
this section cite: []

Section: Fully MoE Architecture.
Both the experts in UMoE-Att and the FFN layers of Transformer consist of two consecutive matrices. While attention layer experts utilize a relatively small intermediate size (d v ), FFN layers typically employ larger dimensions. Recent advances in FFN-MoE models suggest the efficacy of using FFN layers with reduced intermediate sizes as experts [6,15,3]. This insight enables the direct adoption of experts in attention layers for FFN layers, resulting in a fully MoE architecture, denoted as UMoE. Fig. 1 illustrates the architecture of a UMoE layer., where the MoE paradigm is applied to both FFN and attention layers using a shared expert set. Notably, to facilitate parameter sharing, experts are implemented as two-layer FFNs with an intermediate size of d v and incorporate a non-linear activation function between matrix multiplications.
Pre-mixing Implementation. The token mixing operation in pre-mixing attention is a weighted summation over token hidden states, which can be implemented as vanilla attention, accepting Q, K, V matrices as input and producing an output matrix. Each token generates distinct query vectors for different experts, while values (hidden states) and their associated keys are shared across experts.
To generate expert-dependent queries for input tokens, each expert requires an additional query projection matrix, leading to a notable increase in parameters. To mitigate the parameter count disparity with existing MoE models, where experts typically comprise two matrices, we employ low-rank matrices [23] for query projection within UMoE experts. For a given token x, the query for expert i is computed as:
q i = xW q + xW i a W i b ,(11)
where the first term is shared across all experts, while the second term is expert-specific with unique parameters, W i a ∈ R d×r and W i b ∈ R r×d k , for each expert. Fig. 3 presents the pseudo-code of a UMoE layer.
Putting It All Together. As illustrated in Fig. 1, UMoE integrates three key components: (1) experts implemented as fine-grained FFNs with dual low-rank query projection matrices, (2) pre-mixing attention mechanism utilizing shared keys and values across experts, and (3) the top-k router for expert selection. It is noteworthy that while MoA [12] also shares keys and values across experts, the values of MoA are the results after applying a value linear transformation to the input token hidden states. In contrast, the values of UMoE directly refer to the input token hidden states.
this section cite: ['b5', 'b14', 'b2', 'b22', 'b11']

Section: Discussion

this section cite: []

Section: References
Ref_id:b0 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b1 Title: Scaling laws for sparsely-connected foundation models Year: (2024)
Ref_id:b2 Title: Scaling laws for fine-grained mixture of experts Year: (2024-07)
Ref_id:b3 Title: Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer Year: (2017)
Ref_id:b4 Title: Switch transformers: scaling to trillion parameter models with simple and efficient sparsity Year: (2022-01)
Ref_id:b5 Title: Towards ultimate expert specialization in mixture-of-experts language models Year: (2024-08)
Ref_id:b6 Title: Openmoe: an early effort on open mixture-of-experts language models Year: (2024)
Ref_id:b7 Title:  Year: (2024)
Ref_id:b8 Title: Olmoe: Open mixture-of-experts language models Year: (2024)
Ref_id:b9 Title:  Year: (2024)
Ref_id:b10 Title: St-moe: Designing stable and transferable sparse expert models Year: (2022)
Ref_id:b11 Title: Mixture of attention heads: Selecting attention heads per token Year: (2022-12-07)
Ref_id:b12 Title: Switchhead: Accelerating transformers with mixture-of-experts attention Year: (2024)
Ref_id:b13 Title: Llama-moe v2: Exploring sparsity of llama from perspective of mixture-of-experts with post-training Year: (2024)
Ref_id:b14 Title: XMoE: Sparse models with fine-grained and adaptive expert selection Year: (2024)
Ref_id:b15 Title: Hash layers for large sparse models Year: (2021)
Ref_id:b16 Title: BASE layers: Simplifying training of large, sparse models Year: (2021-07-24)
Ref_id:b17 Title: Mixture-of-experts with expert choice routing Year: (2022)
Ref_id:b18 Title: Harder task needs more experts: Dynamic routing in MoE models Year: (2024-08)
Ref_id:b19 Title: Moh: Multi-head attention as mixture-ofhead attention Year: ()
Ref_id:b20 Title: Multi-head mixture-of-experts Year: (2024)
Ref_id:b21 Title: Attention is all you need Year: (2017)
Ref_id:b22 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b23 Title: Parallelizing linear transformers with the delta rule over sequence length Year: (2024)
Ref_id:b24 Title: Mamba: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b25 Title: Transformer feed-forward layers are key-value memories Year: (2021)
Ref_id:b26 Title: Towards a unified view of sparse feed-forward network in pretraining large language model Year: (2023-12)
Ref_id:b27 Title: Neuron-level knowledge attribution in large language models Year: ()
Ref_id:b28 Title: The fineweb datasets: Decanting the web for the finest text data at scale. CoRR, abs/2406.17557, 2024 Year: ()
Ref_id:b29 Title: Pointer sentinel mixture models Year: (2017-04-24)
Ref_id:b30 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b31 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b32 Title: Deepspeed-moe: Advancing mixture-ofexperts inference and training to power next-generation AI scale Year: (2022-07-23)
Ref_id:b33 Title: Roformer: Enhanced transformer with rotary position embedding Year: ()
Ref_id:b34 Title: Universal transformers Year: (2019)
Ref_id:b35 Title: Moeut: Mixture-of-experts universal transformers Year: ()
