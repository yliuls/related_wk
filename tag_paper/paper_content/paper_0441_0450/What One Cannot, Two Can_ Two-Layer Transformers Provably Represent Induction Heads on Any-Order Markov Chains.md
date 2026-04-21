Title: What One Cannot, Two Can: Two-Layer Transformers Provably Represent Induction Heads on Any-Order Markov Chains
Abstract: In-context learning (ICL) is a hallmark capability of transformers, through which trained models learn to adapt to new tasks by leveraging information from the input context. Prior work has shown that ICL emerges in transformers due to the presence of special circuits called induction heads. Given the equivalence between induction heads and conditional k-grams, a recent line of work modeling sequential inputs as Markov processes has revealed the fundamental impact of model depth on its ICL capabilities: while a two-layer transformer can efficiently represent a conditional 1-gram model, its single-layer counterpart cannot solve the task unless it is exponentially large. However, for higher order Markov sources, the best known constructions require at least three layers (each with a single attention head)leaving open the question: can a two-layer single-head transformer represent any k th -order Markov process? In this paper, we precisely address this and theoretically show that a two-layer transformer with one head per layer can indeed represent any conditional k-gram. Thus, our result provides the tightest known characterization of the interplay between transformer depth and Markov order for ICL. Building on this, we further analyze the learning dynamics of our two-layer construction, focusing on a simplified variant for first-order Markov chains, illustrating how effective in-context representations emerge during training. Together, these results deepen our current understanding of transformer-based ICL and illustrate how even shallow architectures can surprisingly exhibit strong ICL capabilities on structured sequence modeling tasks. Code is available at the link.

Section: Introduction
"A complex system that works is invariably found to have evolved from a simple system that worked."
-John Gall, Systemantics (1975) Transformers, powered by the attention mechanism, have emerged as the dominant architecture in machine learning, achieving state-of-the-art performance across a wide range of domains, including ˚Corresponding author. : Equal contribution / equal mentorship. Authors listed alphabetically; either may be considered last author.
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
natural language processing [6], computer vision [32], and complex reasoning tasks [23,36]. A key factor underpinning this success is their ability to efficiently model sequences and perform in-context learning (ICL)-adapting to unseen tasks during inference by leveraging the relevant input context [22]. It is well known that ICL emerges in transformers due to the presence of special circuits called induction heads [22,31].
Intuitively, these circuits enable transformers to implement a "copy-and-match" mechanism by copying earlier tokens from the input and matching them with the desired context for next-token prediction. For example, a first-order induction head mimics the functionality r. . . , A, B, . . . , As Ñ B. Capitalizing on the connection between induction heads and conditional k-gram models, a recent active line of work has leveraged k th -order Markov chains as a simple yet powerful framework to analyze how transformers learn induction heads, with the framework referred to as Markov-ICL [10,20,26,7,5]. In particular, using Markovian input sequences, the goal is to understand how transformers learn to represent conditional k-grams-equivalent to k th -order induction heads (Def. 2).
Building upon the Markov-ICL setting, Bietti et al. [5], Edelman et al. [10], and Nichani et al. [20] demonstrate that a conditional 1-gram model can be efficiently represented by a two-layer (i.e., two attention layers), single-head transformer. In contrast, Sanford et al. [29] show that a one-layer, single-head counterpart cannot solve this task unless its hidden dimension is exponentially larger than that of the two-layer model. On the other hand, for higher-order processes, Edelman et al. [10] and Nichani et al. [20] argue that low-depth transformers need the number of heads scaling linearly in k, in order to learn k th -order Markov processes. However, Rajaraman et al. [26] establish a surprising result that a three-layer, single-head transformer can represent the conditional k-gram model for any k ě 1, and thereby learn k th -order Markov models in-context.
Together, these findings illustrate that while a 1-layer transformer cannot efficiently represent an induction head, a 3-layer model with 1 head per layer suffices for all Markov orders k ě 1. However, these results leave open a natural and important question:
Can a two-layer, single-head transformer learn k th -order Markov processes in-context?
In this paper, we approach this question from two perspectives-representational power and learning dynamics. In particular, we affirmatively answer this on the representation front and make partial progress on the learning dynamics through the following key contributions:
this section cite: ['b5', 'b31', 'b22', 'b35', 'b21', 'b21', 'b30', 'b9', 'b19', 'b25', 'b6', 'b4', 'b4', 'b9', 'b19', 'b28', 'b9', 'b19', 'b25']

Section: Main Contributions.
1. Improving upon the best known three-layer construction, we prove that a two-layer single-head transformer is sufficient to represent the conditional k-gram model, and thus can learn k th -order Markov chains in-context (Sec. 4). To the best of our knowledge, this is the tightest characterization of transformer depth and Markov order for Markov-ICL.
2. In the course of establishing this result, we uncover an interesting tradeoff between depth and width: the existing three-layer single-head construction (depth is three, width is one) is equivalent to a two-layer architecture with two heads in the first layer and one in the second (depth is two, width is two). (Sec. 3) 3. For first-order Markov chains, we prove that gradient descent on our simplified two-layer transformer learns an induction head, and thereby the in-context conditional empirical distribution. (Sec. 5)
Comparison with other Markov-ICL works vis-a-vis transformer architecture. In this paper, as we focus on higher-order processes, we consider the general transformer architecture (Subsec. 2.2) closer to real-world models with relative positional encodings [30], softmax attention [32], MLPs, and layer normalization [2]. For first-order processes, past literature has considered simplified variants: Bietti et al. [5] use frozen and fixed random embeddings with no MLPs in the first layer, Edelman et al. [10] consider attention-only models with no MLPs and embeddings, and Nichani et al. [20] study a variant called disentangled transformer, splitting the residual and attention streams, with the hidden dimension scaling with depth. While our gradient descent result for first-order processes (Sec. 5) is similar to that of Nichani et al. [20], the architecture is fundamentally different. In Tab. 1, we compare the parameter counts of our construction against alternative constructions. Our results show that, as both the order and sequence length increase, our architecture remains the most compact in terms of parameter efficiency.
Previous works Markov order Architecture # Parameters Bietti et al. [5] 1 2-layer, 1-head 9d 2 `dS `T d Edelman et al. [10] 1 2-layer, 1-head 21S 2 `3T S Nichani et al. [20] k 2-layer, k-head pk `pk `1q 2 qpS `T q 2 `T pS `T q Rajaraman et al. [26] k 3-layer, 1-head 15p6S `3q 2 `p6S `3qp3T `2S `10q Ours k 2-layer, 1-head 9p6S `3q 2 `p6S `3qp2T `2S `9q
Table 1: Comparison with prior works on transformers with Markov-ICL. Here k ě 1 is the Markov order, T the sequence length, and S the state-space (vocabulary) size. Note that Bietti et al. [5] do not explicitly state what d is and assume it to be large enough. To the best of our knowledge, our work obtains the tightest known characterization of transformer depth and Markov order for higher-order processes. The bit precision for our representation results is the same as that of Rajaraman et al. [26]: it suffices to have ΩplogpT q `kq bits per parameter, with Op1{T q additive approximation error. A detailed breakdown of the parameter count computations can be found in App. I. Additional experiments analyzing the effect of varying bit precisions are presented in Subsec. H.4.
Notation. Scalars are denoted by lowercase italic letters (e.g., x, y), vectors by lowercase bold letters (e.g., x, y), and matrices by uppercase bold letters (e.g., A, B). The matrices 0 mˆn and 1 mˆn represent the all-zeros and all-ones matrices in R mˆn , respectively. For norms and inner products we use standard notation, i.e., norms are written as || ¨||, with ||x|| 2 denoting the Euclidean norm. The inner product in Euclidean space between two vectors x and y is denoted by xx, yy.
The indicator function is denoted either by Ip¨q or 1 p¨q , depending on the context. The vocabulary S " t1, 2, . . . , Su of cardinality |S| " S denotes the finite state-space of the Markov Chains studied in this paper. A sequence x 0:T is defined as x 0:T :" x 0 , x 1 , . . . , x T , where T P N. For any x P S, its one-hot embedding is denoted by e S x P R S . Op¨q denotes the big O notation.
this section cite: ['b29', 'b31', 'b1', 'b4', 'b9', 'b19', 'b19', 'b4', 'b25']

Section: Related Work
In recent years, transformers have been widely studied from both theoretical and mechanistic perspectives [35,14,19]. Foundational results [37,25,33] have established their universality and Turing-completeness, with further work examining their ability to model formal languages [4,16] and implement algorithmic behaviors such as induction heads [22]. Concurrently, recent studies have explored how transformers perform ICL, including their learning dynamics and phase transitions [31,34,3,8,27,3,1,11,15]. On this front, our work is most closely related to recent works studying Markov-ICL, where the sequential inputs are modeled as Markov processes. In Bietti et al. [5] and Edelman et al. [10], the authors discover a stage-wise learning procedure for first-order sources, and show that a 2-layer, 1-head transformer can represent a first-order induction head. Makkuva et al. [17,18] study the optimization landscape and learning dynamics of a 1-layer, 1-head model for a first-order global Markov chain, unveiling the significance of weight-tying and Markov switching probabilities. Nichani et al. [20] studies the learning dynamics of 2-layer, 1-head disentangled transformer, illustrating it learns to implement a first-order induction head. For higher-order sources, Nichani et al. [20] constructs a 2-layer, k-head architecture, whereas Rajaraman et al. [26] shows that a 3-layer, 1-head model suffices to represent a conditional k-gram model. In contrast, we show that a 2-layer, 1-head transformer suffices to represent arbitrary-order Markov models in-context.
this section cite: ['b34', 'b13', 'b18', 'b36', 'b24', 'b32', 'b3', 'b15', 'b21', 'b30', 'b33', 'b2', 'b7', 'b26', 'b2', 'b0', 'b10', 'b14', 'b4', 'b9', 'b16', 'b17', 'b19', 'b19', 'b25']

Section: Background
In this section, we provide the requisite preliminaries on Markov processes, the conditional k-gram model, and the Transformer.
this section cite: []

Section: Markov Processes and the Conditional k-gram Model
A D B A Q D A D C ? A D … Match last 𝑘 symbols Uniform Distribution Input:
(a) The conditional k-gram model (Def. 2). It first (1) identifies the positions in the sequence where the preceding k tokens match the current context (blue), and (2) returns the empirical distribution over the symbols (green) that follow these matched positions.
… 0 1 2 3 4 5 6 7 8 T-1 … A D B A Q D A D C D … Input: Positions Attention:
A D T (b) k th -order induction head (Def. 4) The attention pattern concentrates on positions in the sequence where the preceding k tokens exactly match the final k tokens. These are the positions the model attends to most strongly, as they reflect the same context as the current prediction target.
A k th -order Markov process is a time-homogeneous stochastic process over a finite state space, where the probability of the next symbol depends only on the most recent k symbols in the sequence [21]. More formally, it is defined as follows:
Definition 1. (k th -order Markov Chains) A stochastic process x 0:T :" x 0 , x 1 , ¨¨¨, x T over a finite state space S is said to be a k th -order Markov Chain with transition kernel πp¨q if, for all k ď n ď T , the following condition holds: πpx n`1 " s | x n , x n´1 , . . . , x 0 q " πpx n`1 " s | x n , x n´1 , . . . , x n´k`1 q, @s P S.
We now define the empirical equivalent of the kernel π, the conditional k-gram model:
Definition 2. (Conditional k-gram model) Given a sequence x 0:T :" x 0 , x 1 , ¨¨¨, x T in S T `1, the conditional k-gram model p π k ps | x T , x T ´1, . . . , x 0 q computes the empirical probability of observing a symbol s P S matching the last k tokens, i.e. p π k ps | x 0 , x 1 , ¨¨¨, x T q :" T ř i"k Ipx i " s, x i´1 " x T , x i´2 " x T ´1, . . . , x i´k " x T ´k`1 q T ř i"k Ipx i´1 " x T , x i´2 " x T ´1, . . . , x i´k " x T ´k`1 q assuming the denominator is non-zero. Thus conditional k-gram can be interpreted as a simple count-based estimator of the transition kernel π, using the in-context count estimate for the predictive probability distribution.
this section cite: ['b20']

Section: Data generation (Random Markov sequences).
To empirically generate the input data, i.e. random Markov sequences of order k, we first sample each row of the transition matrix π independently from a Dirichlet prior with parameter vector 1, which is equivalent to an uniform distribution on the S-dimensional probability simplex. The initial k tokens of each sequence are sampled uniformly from S k , and the remaining tokens are generated according to the sampled kernel π. To sample an input batch, we first sample multiple independent transition kernels, generate sequences from each, and then aggregate these sequences to form a batch for training or evaluation.
this section cite: []

Section: Transformer Model
While there exist several attention-based transformer architectures in literature, we adopt the formulation described in Shaw et al. [30] and Dai et al. [9], which employs relative positional encodings. Each layer in our transformer comprises softmax-based self-attention, optionally followed by a multilayer perceptron (MLP). Following the construction outlined in Rajaraman et al. [26], we define a L-layer transformer with multi-headed attention as follows:
Definition 3. (Multi-head Attention Transformer) Require: Input x 0:T , number of layers L, number of attention heads per layer H ℓ Ensure: Output distribution P θ p¨| x 0:T q x p1q n Ð Embpx n q for n " 0, 1, . . . , T for ℓ " 1 to L do for n " 0 to T do for h " 1 to H ℓ do r x pℓ,head:hq n :"
ř n i"0 att pℓ,head:hq n,i ¨´W pℓ,head:hq V x pℓq i `ppℓ,head:h,V q n´i ēnd for r x pℓ`1q n :" x pℓq n `řH ℓ h"1 r x pℓ,head:hq n Ź Residual connection x pℓ`1q n :" MLPpx pℓq n , r x pℓ`1q n
q Ź MLP with layer normalization and skip connections end for end for
logits T :" W o x pL`1q T `bo P θ p¨| x 0:T q :" f plogits T q,
Where the attention weights are defined as:
att pℓ,head:hq n,i :" softmax i ´AW pℓ,head:hq K x pℓq i `ppℓ,head:h,Kq n´i , W pℓ,head:hq Q x pℓq i E¯.
Here ℓ denotes the layer index and h the index of the attention head within a layer. The matrices W pℓ,head:hq Q , W pℓ,head:hq K , W pℓ,head:hq V P R dˆd are the query, key, and value projection matrices, respectively The vectors p pℓ,head:h,Kq n´i and p pℓ,head:h,V q n´i P R d represent the relative positional encodings that modulate attention based on token distance. Finally, the output of the transformer is projected using W o x pL`1q T `bo P R S , mapping the representation from the embedding dimension d to the output vocabulary size S. Although a softmax is typically used for output normalization, in our setup we define f p¨q " ReLUp¨q. Note that the attention mechanism described is causal self-attention.
this section cite: ['b29', 'b8', 'b25']

Section: Warm Up: Construction with Two Heads in Layer One, One in Layer Two
As a warm up, in this section, we prove our first main result that we can represent k th -order Markov processes with a 2-layer, 2-head transformer. Towards the same, we recall that the k th -order in-context counting estimator, i.e. conditional k-gram model, defined in Subsec. 2.1 is closely related to the classical Laplacian smoothing, which in turn is the optimal Bayes estimator for the next-token predictive distribution [12,28]. In view of this fact, in order to estimate the optimal conditional distribution of each token in-context, it is natural to ask: how does a transformer represent this conditional k-gram model? To this end, Rajaraman et al. [26] introduces the notion of a k-th order induction head. This mechanism enables the attention layer of a transformer to assign at each time step the highest attention weight to past tokens whose length-k context matches that of the current token. For the sake of completeness, we recall the formal definition.
Definition 4. (k th -order induction head) An attention layer with a single head is said to implement a k th order induction head if, for any input sequence px 0 , x 1 , . . . , x T q P S T `1, and for any fixed index i ď T , the attention score att T,i at position i is maximized if and only if its preceding k tokens exactly match the final k tokens of the sequence. That is, att T,i is maximized if and only if x i´j " x T ´j`1 for all i P t1, 2, . . . , ku.
The salience of this higher-order induction head is immediately reflected by the fact that such a mechanism is indeed what is precisely needed to implement the conditional k-gram estimator from Def. 2. Capitalizing on this, Rajaraman et al. [26] proves the best known result for higher-order Markov processes, through a 3-layer, 1-head transformer architecture, which we recall below to motivate our main result: Theorem 1. (Theorem 4 in Rajaraman et al. [26]): The conditional k-gram model can be represented using a transformer with three layers, each containing a single attention head. The layers are separated by MLP blocks and include relative positional encodings as well as layer normalization. The embedding dimension scales as OpSq.
Proof sketch. We provide a brief outline of the proof of Thm. 1. For each position n in the sequence s 0:T , the first attention layer effectively learns a hard attention map concentrated on the indices tn, n ´1, . . . , n ´k `1u. Specifically, within a subspace of its embedding space, this layer computes u n "
´řk´1 i"0 3 i ¨eS xn´i
this section cite: ['b11', 'b27', 'b25', 'b25', 'b25']

Section: ¯{ ´řk´1
i"0 3 i ¯, with attention weights attn n,i " 3 i { ´řk´1 j"0 3 j for i P tn, n ´1, . . . , n ´k `1u, and zero otherwise. Intuitively, this u n allows to capture the past-k context starting at position n. Similarly, the second attention layer focuses on the positions tn ´1, n ´2, . . . , n ´ku, and computes v n "
´řk i"1 3 i ¨eS xn´i
this section cite: []

Section: ¯{ ´řk
i"1 3 i ¯, capturing the context shifted by one position. After these two layers, the embedding at position n contains both v n {}v n } 2 and the normalized vector u n {}u n } 2 , assuming appropriate choices of embedding dimensions, query/key/value matrices, positional encodings, and MLP layers with layer normalization. In particular, the architecture can be configured to compute the ℓ 2 -norm via layer norm, as detailed in Subsec. A.1. The third attention layer then functions as a k-th order induction head: at position n, the attention score is made proportional to the cosine similarity xu T , v n y{ p}u T } 2 }v n } 2 q. This quantity equals 1 when u T " v n , and is strictly less than 1 otherwise. As the softmax temperature tends to infinity, or equivalently, as the attention logits are scaled by a constant tending to infinity, the model approaches the behavior of a conditional k-gram estimator. Hence, the third layer effectively implements a k-th order induction head.
We are now ready to prove our first main result. Specifically, we show that the aforementioned construction with 3 layers and a single head can be adapted to a two-layer counterpart with two heads in the first layer and one head in the second.
this section cite: []

Section: Theorem 2.
(Two heads in the first layer, one in the second): The conditional k-gram model can be represented using a transformer with two layers, the first containing two attention heads and the second containing a single attention head. The layers are separated by MLP blocks and include relative positional encodings as well as layer normalization. The embedding dimension is 6S `3.
this section cite: []

Section: Remark 1.
(Depth-width tradeoff) Thm. 2 thus reveals an interesting tradeoff between depth (number of attention layers) and width (maximum number of attention heads per layer). While our construction here has both depth and width two, its counterpart in Thm. 1 instead has depth three and width one. Thus our new architecture trades off depth with an additional attention head, thereby increasing the width.
Proof sketch. We provide a brief sketch of the proof for Thm. 2, using the same notation as that of the Thm. 1, and focusing on the key differences between the two architectures.
Layer One. For each position n in the sequence s 0:T , the first attention head in the first layer learns a focused attention pattern over the positions tn, n´1, . . . , n´k`1u. In a subspace of the embedding space, this head computes the representation u n "
´řk´1 i"0 3 i ¨eS xn´i
this section cite: []

Section: ¯{ ´řk´1
i"0 3 i ¯. On the other hand, the second attention head in this layer focuses on the preceding context, attending to positions tn ´1, n ´2, . . . , n ´ku, and computes the representation v n "
´řk i"1 3 i ¨eS xn´i ¯{ ´řk i"1 3 i ¯.
After this layer, the embedding at position n contains both the vectors u n {}u n } 2 and v n {}v n } 2 , achieved by choosing an appropriate choice of embedding dimensions, query, key, and value matrices, positional encodings, and MLPs with layer normalization.
this section cite: []

Section: Layer Two.
The second attention layer now plays the role of an induction head, analogous to the third layer in Theorem 1. At position n, this head computes an attention score that is proportional to the cosine similarity xu T , v n y{}u T } 2 }v n } 2 . This score attains the maximum value of one when u T " v n , and strictly less otherwise. As the temperature parameter in the softmax becomes large, or equivalently as the inner product is multiplied by a large constant, the attention mechanism approaches the behavior of a conditional k-gram estimator. Therefore, this two-layer transformer, with two heads in the first layer and a single head in the second, is sufficient to represent the conditional k-gram model. We refer to App. A for the full proof.
this section cite: []

Section: Main Result: Two-Layer Single-Head Construction

this section cite: []

Section: Learnt Attention Map

this section cite: []

Section: Layer -1

this section cite: []

Section: Learnt Attention Map Optimal Attention Map

this section cite: []

Section: Layer -2
Figure 2: Attention maps learnt by a two-layer, single-head transformer trained on sequences generated by random Markov chains of order 3 (Subsec. 2.1). (i) in the first layer, the attention map shows a clear pattern: attention weights increase monotonically along the first three lower-diagonals and drop to zero beyond that. This suggests that the relative positional bias is maximized the diagonal with an offset of ´k " ´3, i.e., the third diagonal below the main diagonal, which is consistent with the construction in Sec. 4, (ii) in the second layer, the attention map closely resembles the ideal attention pattern required to approximate the conditional k-gram estimator. We note that all experiments were conducted using standard initialization schemes. Additional experiments with different orders of markov chains and experimental details are provided in App. H. In Subsec. H.2, we also experimentally demonstrate that single-layer transformers fail to solve the induction head task with the same order of parameters. Finally, in Subsec. H.3, we test the robustness of the two-layer, single-head model to noise in the input sequences.
While Thm. 2 above demonstrates that the conditional k-gram can be represented by a two-layer transformer, it however relies on a two-head construction, thus still leaving open our motivating question: can a 2-layer, 1-head transformer represent the conditional k-gram? In this section, we precisely address this gap and realize a 2-layer, 1-head model to represent k th -order Markov processes for any k ě 1. The key idea behind our approach is to leverage the MLP-and specifically, the non-linearities such as ReLU and LayerNorm-to isolate symbols at particular positions that are useful for estimating induction heads. This contrasts with prior constructions, which primarily focused on the attention mechanism while underutilizing the role of the MLP and its non-linear components. As a result, our construction highlights how these non-linear MLP elements can play a critical role in enabling in-context learning, demonstrating their significance. We now present our main result.
Theorem 3. (Two-layer, single-head construction): The conditional k-gram model can be represented using a transformer with two layers, both containing a single attention head. The layers are separated by MLP blocks and include relative positional encodings as well as layer normalization. The embedding dimension is 6S `3.
Remark 2. (Parameter count and bit precision) We note that the above construction utilizes 9p6S 3q 2 `p6S `3qp2T `2S `9q parameters. A slightly more efficient construction in terms of parameter count with 8p6S `3q 2 `p6S `3qp2T `2S `9q parameters can be found in App. B. For both these results, it suffices to have ΩplogpT q `kq bits per parameter, with Op1{T q additive approximation error, where T is the sequence length.
Proof Sketch. We now provide an outline of the proof of Thm. 3, using notation consistent with the constructions in Thm. 1 and Thm. 2.
Layer One. For each position n in the sequence s 0:T , the attention head in the first layer attends to the preceding context, focusing on positions tn´1, n´2, . . . , n´ku, and computes the representation v n "
´řk i"1 3 i ¨eS xn´i
this section cite: []

Section: ¯{ ´řk
i"1 3 i ¯, with attention weights attn n,i " 3 i { ´řk j"1 3 j ¯for i P tn ´1, . . . , n ´ku, and zero otherwise. This matches the optimal attention pattern for Layer-1 illustrated in Fig. 2.
this section cite: []

Section: MLP.
In contrast to previous constructions, the MLP situated between the two attention layers plays a central role in our proof and is structured to reconstruct the complementary representation u n " ´řk´1
i"0 3 i ¨eS xn´i ¯{ ´řk´1 i"0 3 i ¯, which corresponds to a hard attention pattern over the positions tn, n ´1, . . . , n ´k `1u. Recall that this u n was constructed in Thm. 2 thanks to a second attention head, absent in our architecture.
Our main idea to tackle this issue is to isolate the one-hot embedding e S
x n´k from v n computed in the first layer. This is achieved by embedding the entire one-hot vocabulary within the weight matrix of the first MLP layer. When combined with a suitable bias and followed by a ReLU activation and layer normalization, this configuration filters out all but the component aligned with x n´k , thereby producing e S
x n´k . At this stage, the embedding also retains the representation v n , computed by the first attention layer. Subsequent layers, along with appropriately configured skip connections, implement a linear transformation that combines e S
x n´k , v n , and the token embedding e S xn , the latter of which is introduced via a skip connection. Since v n and e S
x n´k occupy orthogonal subspaces of the embedding, and all required components are explicitly available, this construction yields the intermediate representation u n " ´eS xn {
ř k´1 i"0 3 i ¯`´3 e S x n´k v n ¯{ ´řk´1 i"0 3 i ¯´´3 k ¨eS x n´k ¯{ ´řk´1 i"0 3 i ¯,
which simplifies exactly to the desired form u n "
´řk´1 i"0 3 i ¨eS xn´i ¯{ ´řk´1 i"0 3 i ¯.
With appropriately chosen weights, layer normalization, and skip connections, the resulting embedding at position n encodes both u n {}u n } 2 and v n {}v n } 2 .
Layer Two. As in the second layer of Thm. 2, the attention mechanism in the second layer is configured to act as a k th -order induction head by computing an attention score at position n that is proportional to the cosine similarity pxu T , v n y{}u T } 2 }v n } 2 q. This score reaches its maximum value of one when u T " v n , and is strictly less than one otherwise. As the softmax temperature tends to infinity, or equivalently as the attention logits are scaled by a constant that tends to infinity, the resulting distribution converges to that of a conditional k-gram estimator. App. B details the complete proof.
Importance of non-linearities. This construction demonstrates that non-linear components such as ReLU activations and layer normalization are not merely auxiliary, but play a critical role in enabling the transformer architecture to express higher-order inductive structures that are essential for in-context learning.
Note. Due to the problem's non-convexity, multiple constructions may enable two-layer, single-head transformers to implement a k th -order induction head; our approach offers one such construction motivated by two key insights. First, empirical observation: for datasets generated from k th -order Markov chains, we observe that in a trained transformer model, the first-layer attention weights increase monotonically along the first k lower diagonals before dropping sharply to near-zero (see Fig. 2 and Subsec. H.1). Second, prior theoretical construction: as demonstrated in Rajaraman et al. [26], the attention pattern can be designed to be proportional to a dyadic sum representation. Taken together, they offer both empirical and theoretical grounding for our construction of k th -order induction heads.
this section cite: ['b25']

Section: Gradient Descent Analysis
Thm. 3 establishes a representation result showing that a two-layer transformer with single attention heads can implement the conditional k-gram model. However, this does not guarantee that current optimization algorithms such as gradient descent (or its stochastic variants) can recover such a solution in practice. To this end, in this section we focus on first-order Markov chains and show that gradient descent on a reduced variant of our two-layer transformer indeed learns an induction head, and thereby the in-context conditional 1-gram. On this note, we would like to emphasize that first-order Markov analysis a critical first-step and a building block for challenging higher-order analysis [20].
Building towards our result, we quickly recall a few recent works [10,20] that also analyze the gradient dynamics on first-order Markov data with simplified transformer architectures. Specifically, Edelman et al. [10] study an attention-only linear transformer without MLPs and softmax operations, while Nichani et al. [20] analyze a disentangled transformer, with residual and attention streams split. On the other hand, our transformer architecture in Def. 3 consists of relative positional encodings, softmax attention, MLP with ReLU activations, and layer normalization. While non-linearities such as ReLU and layer normalization play a crucial role in our representation result, as illustrated in Sec. 4, at the same time, they make the gradient analysis challenging and intractable even for a simple first-order Markov setup.
To tackle this, we study a reduced model wherein we only treat the salient components of our two-layer architecture in Thm. 3, such as positional encodings and attention layer, as parameters, and treating the rest as approximately optimal. This enables for a a faithful reproduction of the ICL phenomenon exhibited by the parent model, whilst being tractable. This is akin to the reparameterization strategies of Makkuva et al. [18] and Nichani et al. [20] for gradient-flow/gradient-descent analysis. Alternatively, this can also be viewed as a form of good parameter initialization, similar to that of [10,20]. We now start with our technical assumptions on the data distribution and the transformer architecture. We use notation from Subsec. 2.2, dropping the head-identifying superscript.
this section cite: ['b19', 'b9', 'b19', 'b9', 'b19', 'b17', 'b19', 'b9', 'b19']

Section: Assumptions.
(i) Data distribution assumptions: Following [20], we assume the prior over transition kernels enforces non-degeneracy, positive transitions, and constant mean (see App. C).
(ii) The positional embeddings used for the keys in the first attention layer are scalar-valued, i.e. p p1,Kq n 9 p n , where p n is a scalar.
(iii) W p1q Q,K,V and p p1,V q are chosen so that the attention weight for position n ´i is given by att p1q n,n´i " exppp i q{ ´ři j"0 exppp j q ¯P t0, 1u, and the corresponding value output is v n " ř n i"0 att p1q n,n´i ¨eS xn´i . (iv) We assume the MLP is optimal-i.e., it outputs the correct u n even from suboptimal v n ; this holds for first-order Markov chains via skip connections.
(v) W p2q Q,K are chosen such that the attention in this layer is defined as att p2q T,i " exp pa 2 ¨xv i , u T yq { ´řT j"0 exppa 2 ¨xv j , u n yq ¯, where a 2 P R is the attention scalar.
(vi) W p2q V is chosen so that the final logit logit T "
ř T i"0 att p2q n,i e xi P R S ,
this section cite: ['b19']

Section: Loss function.
We minimize the cross-entropy loss for next-token prediction [20]:
Lpθq " ´Eπ"Pπ, x 0:T "π « ÿ sPS π px T `1 " s | x 0:T q log plogit T psq `εq ff ,(1)
where ε ą 0 is a small additive constant, P π p¨q denotes the prior over Markov transition kernels, and logit T psq is the logit corresponding to the symbol s.
Training algorithm. We denote the set of trainable parameters as θ " pp, a 2 q, where the positional scalars p " rp 0 , p 1 , . . . , p T s T and a 2 is the attention scalar. Similar to [20], we adopt a two-stage training procedure. (i) In the first stage, we optimize the positional scalars p using gradient descent for T 1 steps with learning rate η 1 , (ii) in the second stage, we freeze p and train only a 2 for an additional T 2 steps using a separate learning rate η 2 . A key distinction in our setup is how we treat layer normalization during training. While in the first stage we omit all layer norms in the MLP, during the second step we reintroduce them. This stage-wise enables a tractable theoretical analysis. We refer to App. C for further details. We now present our main result: Theorem 4. (Convergence of the training Algorithm): Suppose that Assumptions (i)-(vi) hold and that the Markov sequence length T satisfies T `1 ě polypγ ´1, Sq for some constant γ ą 0. Then there exist ε ą 0, learning rates η 1 , η 2 , and step counts T 1 , T 2 , such that the output of the above two-stage training algorithm, θ " ptp i u T i"0 , â2 q, satisfies Lp θq ´L˚À logpT `1q pT `1q cγ , for a constant c independent of pγ, Sq, and the optimal loss L ˚:"
´E " p1{Sq ř s,s 1 πps 1 | sq log πps 1 | sq ı .
Thus Thm. 4 illustrates that the transformer model trained via the two-stage algorithm achieves near-optimal loss asymptotically. Furthermore, we can also show that θ approximates the conditional 1-gram model, i.e. first-order induction head, with vanishing error as sequence length grows (see App. C). For completeness, perturbative analysis of the architecture consisting of two attention heads in the first layer and one in the second (see Sec. 3) for any-order Markov chains can be found in App. D.
this section cite: ['b19', 'b19']

Section: Conclusion
In this paper, improving over prior three-layer constructions, we show that a two-layer, single-head transformer can represent any k-th order Markov process. Thus our result gives the tightest known depth characterization for in-context learning of conditional k-grams. Additionally, we provide a gradient descent analysis of how such representations emerge during training on first-order Markov data. While these results deepen our theoretical understanding of ICL and show that compact transformer architectures can be both expressive and learnable, several open questions remain. In particular, fully characterizing the learning dynamics for higher-order processes is an important avenue of future research. Further, we view our contributions as a first step toward understanding the fundamental limits of ICL in compact transformer architectures, offering not only a tractable setting for theoretical analysis, but also a potential pathway toward more efficient model designs.
this section cite: []

Section: References
Ref_id:b0 Title: What learning algorithm is in-context learning? Year: (2023)
Ref_id:b1 Title: Layer normalization Year: (2016)
Ref_id:b2 Title: Transformers as statisticians: Provable in-context learning with in-context algorithm selection Year: (2023)
Ref_id:b3 Title: On the Ability and Limitations of Transformers to Recognize Formal Languages Year: (2020-11)
Ref_id:b4 Title: Birth of a transformer: A memory viewpoint Year: (2023)
Ref_id:b5 Title: Language models are few-shot learners Year: (2020)
Ref_id:b6 Title: Unveiling induction heads: Provable training dynamics and feature learning in transformers Year: (2024)
Ref_id:b7 Title: In-context learning with transformers: Softmax attention adapts to function lipschitzness Year: (2024)
Ref_id:b8 Title: Transformer-xl: Attentive language models beyond a fixed-length context Year: (2019)
Ref_id:b9 Title: The evolution of statistical induction heads: In-context learning markov chains Year: (2024)
Ref_id:b10 Title: The developmental landscape of in-context learning Year: (2024)
Ref_id:b11 Title: Essai philosophique sur les probabilités Year: (1814)
Ref_id:b12 Title: Markov chains and mixing times Year: (2017)
Ref_id:b13 Title: How do transformers learn topic structure: Towards a mechanistic understanding Year: (2023)
Ref_id:b14 Title: Dual operating modes of in-context learning Year: (2024)
Ref_id:b15 Title: Transformers learn shortcuts to automata Year: (2023)
Ref_id:b16 Title: Attention with markov: A framework for principled analysis of transformers via markov chains Year: (2024)
Ref_id:b17 Title: Local to global: Learning dynamics and effect of initialization for transformers Year: (2024)
Ref_id:b18 Title: Understanding transformers via n-gram statistics Year: (2024)
Ref_id:b19 Title: How transformers learn causal structure with gradient descent Year: (2024)
Ref_id:b20 Title: Markov chains. Number 2 Year: (1998)
Ref_id:b21 Title: -context learning and induction heads Year: (2022)
Ref_id:b22 Title: OpenAI. Gpt-4 technical report Year: (2023)
Ref_id:b23 Title: GPT-2 modular codebase implementation Year: (2025-01)
Ref_id:b24 Title: Attention is Turing-complete Year: (2021)
Ref_id:b25 Title: Transformers on markov data: Constant depth suffices Year: (2024)
Ref_id:b26 Title: Towards understanding how transformers learn in-context through a representation learning lens Year: (2024)
Ref_id:b27 Title: Universal coding, information, prediction, and estimation Year: (1984)
Ref_id:b28 Title: One-layer transformers fail to solve the induction heads task Year: (2024)
Ref_id:b29 Title: Self-attention with relative position representations Year: (2018)
Ref_id:b30 Title: What needs to go right for an induction head? a mechanistic study of in-context learning circuits and their formation Year: (2024)
Ref_id:b31 Title: Attention is all you need Year: (2017)
Ref_id:b32 Title: Statistically meaningful approximation: a case study on approximating Turing machines with transformers Year: (2022)
Ref_id:b33 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b34 Title: Thinking like transformers Year: (2021)
Ref_id:b35 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b36 Title: Are transformers universal approximators of sequence-to-sequence functions Year: (2020)
