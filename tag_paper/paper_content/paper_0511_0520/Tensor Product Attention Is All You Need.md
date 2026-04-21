Title: Tensor Product Attention Is All You Need
Abstract: Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor Product Attention (TPA), a novel attention mechanism that uses tensor decompositions to represent queries, keys, and values compactly, substantially shrinking the KV cache size at inference time. By factorizing these representations into contextual low-rank components and seamlessly integrating with RoPE and any possible position encoding mechanisms, TPA achieves improved model quality alongside memory efficiency. Based on TPA, we introduce the Tensor ProducT ATTenTion Transformer (T6), a new model architecture for sequence modeling. Through extensive empirical evaluation on language modeling tasks, we demonstrate that T6 surpasses or matches the performance of standard Transformer baselines, including Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Multi-Head Latent Attention (MLA) across various metrics, including perplexity and a range of established evaluation benchmarks. Notably, TPA's memory efficiency and computational efficiency at the decoding stage enable processing longer sequences under fixed resource constraints, addressing a critical scalability challenge in modern language models. Project

Section: Introduction
Large language models (LLMs) have revolutionized natural language processing, demonstrating exceptional performance across tasks [5,12,58,6]. As these models evolve, their ability to process longer contexts becomes increasingly important for sophisticated applications such as document analysis, complex reasoning, and code completion. However, managing longer sequences during inference poses significant computational and memory challenges, particularly due to the storage of key-value (KV) caches [70,34]. Because memory consumption grows linearly with sequence length, the maximum context window is limited by practical hardware constraints. A variety of solutions have been explored to address this memory bottleneck. Some approaches compress or selectively prune cached states through sparse attention patterns [10] or token eviction strategies [70,62,42], though such methods risk discarding tokens that may later prove important. Other work proposes off-chip storage of key-value states [17], at the expense of increased I/O latency. Attention variants like Multi-Query Attention (MQA) [46] and Grouped-Query Attention (GQA) [2] reduce per-token cache requirements by sharing keys and values across heads, but often compromise flexibility or require significant architectural modifications. Meanwhile, low-rank weight factorization methods such as LoRA [20] effectively reduce fine-tuning memory, yet do not address the KV cache overhead that dominates inference at runtime. The recently introduced Multi-Head Latent Attention Figure 1: Tensor Product Attention (TPA) within the Tensor ProducT ATTenTion Transformer (T6). In each TPA layer, the input hidden state x t is processed by linear layers to produce latent factor matrices for query (e.g., A Q (x t ), B Q (x t )), key (e.g., A K (x t ), B K (x t )), and value (e.g., A V (x t ), B V (x t )). Rotary Position Embedding (RoPE) is applied to the B Q (x t ) and B K (x t ) factors. The query, key, and value tensors for each attention head are then formed by the tensor product of these factor matrices (e.g., Q t = 1 R Q A Q (x t ) ⊤ B Q (x t )). Finally, the TPA output is computed using scaled dot-product attention, followed by a linear projection of the concatenated results from all heads.
(MLA) in Deepseek-V2 [32] caches compressed key-value representations but encounters difficulties with efficient Rotary Position Embedding (RoPE) [52] integration, necessitating additional positionencoded parameters per head. To overcome the limitations of existing approaches, we introduce Tensor Product Attention (TPA), illustrated in Figure 1. TPA is a novel attention mechanism that employs tensor factorizations for queries (Q), keys (K), and values (V). By dynamically factorizing activations rather than static weights (as in LoRA), TPA constructs low-rank, contextual representations. This approach substantially reduces KV cache memory usage while offering improved representational capacity. In practice, TPA can decrease memory overhead by an order of magnitude compared to standard Multi-Head Attention (MHA), alongside achieving lower pretraining validation loss (perplexity) and better downstream performance. A key advantage of TPA is its native compatibility with rotary positional embeddings (RoPE) [52] and any possible position encodings, enabling a straightforward drop-in replacement for multi-head attention (MHA) layers in modern LLM architectures such as LLaMA [58], Qwen [3], and Gemma [56]. Our main contributions are summarized as follows:
1. We propose Tensor Product Attention (TPA), a mechanism that factorizes Q, K, and V activations using contextual tensor decompositions. This achieves a substantial reduction in inferencetime KV cache size relative to standard attention mechanisms [60], MHA, MQA, GQA, and MLA, while also improving performance. In addition, we analyze existing attention mechanisms and reveal that MHA, MQA, and GQA can be expressed as non-contextual variants of TPA. 2. We introduce the Tensor ProducT ATTenTion Transformer (T6), a new TPA-based model architecture for sequence modeling. In language modeling experiments, T6 consistently improves or matches validation perplexity and downstream evaluation performance, all while maintaining a reduced KV cache size. 3. We demonstrate that TPA integrates seamlessly with RoPE [52] and any possible position encodings as well as output gate and KV shifting, facilitating its easy adoption in popular foundation model architectures like LLaMA, Gemma, and Qwen.
4. We develop FlashTPA Decoding, an efficient autoregressive inference algorithm for TPA. Our empirical results show that FlashTPA Decoding can be faster than optimized MHA, MQA, GQA, and MLA decoding methods, particularly for long sequences.
this section cite: ['b4', 'b11', 'b57', 'b5', 'b69', 'b33', 'b9', 'b69', 'b61', 'b41', 'b16', 'b45', 'b1', 'b19', 'b31', 'b51', 'b51', 'b57', 'b2', 'b55', 'b59', 'b51']

Section: Background
In this section, we briefly review Scaled Dot-Product Attention, Multi-Head Attention [60], and introduce key notations. Other attention mechanisms like Multi-Query Attention (MQA) [46], Grouped Query Attention (GQA) [2], Multi-head Latent Attention (MLA) [32,33], and Rotary Position Embedding (RoPE) [52] are further discussed in the Appendix F. Notations. We use bold uppercase letters (e.g., X, Q) for matrices, bold lowercase (e.g., a, b) for vectors, and italic uppercase (e.g., W Q i ) for learnable parameter matrices. We denote by [n] the set {1, . . . , n} for some positive integer n. We use ⊤ to denote the transpose of a vector or a matrix. Let d model be the embedding dimension, h the number of attention heads, d h the dimension per head, x t ∈ R dmodel the input for the t-th token at a given attention layer, X ∈ R T ×dmodel denotes the input embeddings for T tokens, and Q, K, V ∈ R T ×h×d h denote the queries, keys, and values of h heads for T tokens. With a little abuse of notation, Q i , K i , V i ∈ R T ×d h denote the i-th head of queries, keys, and values, and Q t , K t , V t ∈ R h×d h denote the heads of the query, key, and value for t-th token. Throughout the paper, W Q , W K , W V denote projection matrices for queries, keys, and values, respectively. In multi-head attention, each head is associated with its own set of W Q i , W K i , W V i , and each has dimension W Q i , W K i , W V i ∈ R dmodel×d h . 5 Similarly, we have an output projection matrix W O ∈ R (h•d h )×dmodel . We define the tensor product of two vectors as follows: for vectors a ∈ R m , b ∈ R n , the tensor product of a and b is: a ⊗ b = C ∈ R m×n , with C ij = a i b j , where a i is the i-th element of a, b j is the j-th element of b, and C ij is the (i, j)-th entry of C. The vectorization of a matrix C ∈ R m×n , denoted vec(C) ∈ R mn , stacks the columns of C into a single column vector. For example, if C = [c 1 , c 2 , . . . , c n ] where c j are columns, then vec(C) = [c ⊤ 1 , c ⊤ 2 , . . . , c ⊤ n ] ⊤ .
this section cite: ['b59', 'b45', 'b1', 'b31', 'b32', 'b51']

Section: Scaled Dot-Product Attention
Scaled dot-product attention [60] determines how to focus on different parts of an input sequence by comparing queries (Q) and keys (K). It produces a weighted combination of the values (V). Formally, the attention output is:
Attention(Q, K, V) = Softmax QK ⊤ √ d h V,
where Q ∈ R n×d h , K ∈ R n×d h , and V ∈ R n×dv for n tokens. The softmax is applied row-wise over the n keys for each query.
this section cite: ['b59']

Section: Multi-Head Attention (MHA)
Multi-Head Attention (MHA) [60] extends scaled dot-product attention by dividing the model's internal representation into several heads. Each head learns different projections for queries, keys, and values, allowing the model to attend to different types of information from different representational subspaces. For each token embedding x t ∈ R dmodel , MHA computes each head i as follows:
Q t,i = (W Q i ) ⊤ x t ∈ R d h , K t,i = (W K i ) ⊤ x t ∈ R d h , V t,i = (W V i ) ⊤ x t ∈ R d h , head i = Attention Q i , K i , V i ,
where W Q i , W K i , W V i ∈ R dmodel×d h are learnable projection matrices for the i-th head, and Q i , K i , V i ∈ R T ×d h are the query, key, and value matrices for the i-th head over T tokens. After computing each head's attention output, the results are concatenated and mapped back to the model's original dimension via another learnable linear projection matrix W O ∈ R hd h ×dmodel : MHA(X) = Concat head 1 , . . . , head h W O . MHA enables the model to capture a rich set of dependencies by allowing each head to focus on different aspects of the input sequence. We also discuss how MHA, MQA, and GQA relate to TPA in the Section 4.
this section cite: ['b59']

Section: Tensor Product Attention
In this section, we provide a detailed description of our proposed Tensor Product Attention (TPA), which enables contextual low-rank factorization for queries, keys, and values. First, we explain how TPA factorizes these components, specifying tensor shapes. Next, we describe TPA's integration into the multi-head attention framework and its benefits for reducing KV cache memory consumption during inference. Finally, we demonstrate RoPE's seamless integration with TPA, including a pre-rotated variant for efficiency.
this section cite: []

Section: Tensor Factorization of Queries, Keys, and Values
Let d attn := h d h denote the total attention projection dimension. Typically one sets d attn = d model , but this is not required: when d attn ̸ = d model , the projection matrices W Q , W K , W V map from R dmodel into R dattn and W O maps R dattn back to R dmodel . Standard attention projects the entire sequence into three tensors, Q, K, V ∈ R T ×h×d h , where Q t , K t , V t ∈ R h×d h denote the slices for the t-th token.
this section cite: []

Section: Contextual Factorization.
Instead of forming each head's query, key, or value via a single linear map, TPA factorizes each Q t , K t , V t into a sum of (contextual) tensor products whose ranks are R Q , R K , and R V , respectively, and may differ. Specifically, for each token t, with a small abuse of notation, we define:
Q t = 1 R Q R Q r=1 a Q r (x t ) ⊗ b Q r (x t ), K t = 1 R K R K r=1 a K r (x t ) ⊗ b K r (x t ), V t = 1 R V R V r=1 a V r (x t ) ⊗ b V r (x t ),(3.1)
where
a Q r (x t ), a K r (x t ), a V r (x t ) ∈ R h ,b Q r (x t ), b K r (x t ), b V r (x t ) ∈ R d h . Hence, for queries, each tensor product a Q r (x t ) ⊗ b Q r (x t ) : R h × R d h → R h×d h
contributes to the query slice Q t ∈ R h×d h . Analogous definitions apply to the key slice K t and value slice V t .
this section cite: []

Section: Latent Factor Maps.
Each factor in the tensor product depends on the token's hidden state x t . For example, for queries, we can write:
a Q r (x t ) = W a Q r x t ∈ R h , b Q r (x t ) = W b Q r x t ∈ R d h , where W a Q r ∈ R h×dmodel and W b Q r ∈ R d h ×dmodel
are learnable weight matrices. Similar linear maps produce the factors for keys and values. One often merges the rank index into a single output dimension. For instance, for queries:
a Q (x t ) = W a Q x t ∈ R R Q •h , b Q (x t ) = W b Q x t ∈ R R Q •d h , which are then reshaped into A Q (x t ) ∈ R R Q ×h and B Q (x t ) ∈ R R Q ×d h (where each row of A Q (x t )
corresponds to an a Q r (x t ) ⊤ and each row of B Q (x t ) to a b Q r (x t ) ⊤ ). The query tensor for token t can then be expressed as:
Q t = 1 R Q A Q (x t ) ⊤ B Q (x t ) ∈ R h×d h .
this section cite: []

Section: This operation is equivalent to Q
t = 1 R Q R Q r=1 a Q r (x t )(b Q r (x t )) ⊤ , where a Q r is the r-th column of A Q (x t ) ⊤ and (b Q r ) ⊤ is the r-th row of B Q (x t )
. Repeating for all tokens reconstitutes Q ∈ R T ×h×d h . Similar procedures are applied to obtain K and V with ranks R K and R V , respectively. Scaled Dot-Product Attention. Once Q, K, V are factorized, multi-head attention proceeds as in standard Transformers. For each head i ∈ {1, . . . , h}:
head i = Softmax 1 √ d h Q i (K i ) ⊤ V i ,(3.2)
where Q i , K i , V i ∈ R T ×d h are the slices along the head dimension. Concatenating these h heads along the last dimension yields an R T ×(h•d h ) tensor, which is projected back to R T ×dmodel by an output weight matrix
W O ∈ R (h•d h )×dmodel : TPA(Q, K, V) = Concat head 1 , . . . , head h W O . (3.3)
Parameter Initialization. We use Xavier initialization [15] for the factor weight matrices; details are in the Appendix G.
this section cite: ['b14']

Section: RoPE Compatibility and Acceleration
In a typical workflow of adding RoPE to standard multi-head attention, one first computes Q t , K s ∈ R h×d h of the t-th token and s-th token and then applies:
Q t → Q t = RoPE t (Q t ), K s → K s = RoPE s (K s ).
(3.4)
this section cite: []

Section: Direct Integration.
A useful optimization is to integrate RoPE directly into the TPA factorization. For example, one can pre-rotate the token-dimension factors:
B K (x t ) := RoPE t B K (x t ) = B K (x t )T t ,
(3.5) yielding a pre-rotated key representation:
K t = 1 R K R K r=1 a K r (x t ) ⊗ RoPE t b K r (x t ) = 1 R K A K (x t ) ⊤ B K (x t ).
Here, RoPE t is applied to each row of B K (x t ) (i.e., to each b K r (x t ) vector). Thus, each cached key factor corresponds to a RoPE-rotated key slice. This removes the need to rotate cached keys at decoding time; the current-step query (which is not cached) can still be rotated on the fly at negligible cost. Depending on hardware and performance requirements, different RoPE integration strategies can be adopted for training and inference. Theorem 3.1 (RoPE's Compatibility with TPA). Let Q t be factorized by TPA as
Q t = 1 R Q A Q (x t ) ⊤ B Q (x t ) ∈ R h×d h ,
where A Q (x t ) ∈ R R Q ×h and B Q (x t ) ∈ R R Q ×d h . Then we have:
RoPE t (Q t ) = Q t T t = 1 R Q A Q (x t ) ⊤ B Q (x t ),(3.6
) where B Q (x t ) := B Q (x t )T t = RoPE t B Q (x t ) (RoPE applied row-wise to B Q (x t )). Furthermore, let Q t = RoPE t (Q t ) = Q t T t and K s = RoPE s (K s ) = K s T s be the RoPE-transformed query/key slices. Then RoPE's standard relative-position identity is preserved:
Q t K ⊤ s = Q t T t-s K ⊤ s , equivalently RoPE t-s (Q t ) K ⊤ s = Q t K ⊤
s , where T t-s := T t T ⊤ s . In particular, for any head i (the i-th row), if q t,i , k s,i ∈ R 1×d h and q t,i = q t,i T t , k s,i = k s,i T s , then q t,i k ⊤ s,i = q t,i T t-s k ⊤ s,i . Theorem 3.1 indicates that TPA does not break RoPE's relative translational property. We prove it in the Appendix D.1.
this section cite: []

Section: KV Caching and Memory Reduction
In autoregressive decoding, standard attention caches K t , V t ∈ R h×d h for each past token t. This accumulates to R T ×h×d h for keys and R T ×h×d h for values, i.e., 2 T h d h total. TPA Factorized KV Caching. Instead of storing the full K t and V t , TPA stores only their factor components. Specifically, for each past token t, we cache:
A K (x t ), B K (x t ) and A V (x t ), B V (x t ), where A K (x t ) ∈ R R K ×h , B K (x t ) ∈ R R K ×d h (pre-rotated), A V (x t ) ∈ R R V ×h , B V (x t ) ∈ R R V ×d h . Hence, the memory cost per token is R K (h + d h ) for K + R V (h + d h ) for V = ( R K + R V ) h + d h . Compared to the standard caching cost of 2 h d h , the ratio is (R K +R V ) (h+d h ) 2 h d h .
For large h and d h (typically d h = 64 or 128), setting R K , R V ≪ h (e.g., rank 1 or 2) often yields substantial reduction of KV cache size. Table 1 provides a comparative overview of different attention mechanisms, including TPA and its variants, focusing on KV cache size per token and the number of parameters in an attention layer.
MHA 2hdh 4dmodel h dh h h MQA 2dh 2dmodel dh (h + 1) h 1 GQA 2Gdh 2dmodel dh (h + G) h G MLA dc + d R h d ′ c (dmodel + hdh + hd R h ) +dc(dmodel + 2hdh) +dmodel(hdh + d R h ) h h TPA (RK + RV )(h + dh) dmodel(RQ + RK + RV )(h + dh) + dmodel hdh h h TPA (KVonly) (RK + RV )(h + dh) dmodel(RK + RV )(h + dh) + 2dmodel hdh h h TPA (Non-contextual A) (RK + RV )dh (RQ + RK + RV )(dmodeldh + h) + dmodel hdh h h TPA (Non-contextual B) (RK + RV )h (RQ + RK + RV )(dmodelh + dh) + dmodel hdh h h
4 Expressing MHA, MQA, GQA as Non-contextual TPA
We demonstrate that standard Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA) can be expressed as special, non-contextual variants of Tensor Product Attention (TPA). This is achieved by imposing specific constraints on the TPA factors, particularly by making the head-dimension factors (a) independent of the input token (x t ).
this section cite: []

Section: MHA as Non-contextual TPA
Standard Multi-Head Attention (MHA) can be precisely formulated as a TPA where the rank is equal to the number of heads (R Q = R K = R V = h), and the head-dimension factors are fixed, non-contextual basis vectors. To recover MHA, we set the rank R Q = h and define the factors for each head i ∈ [h] as follows:
• Contextual token factor: This is the standard linear projection for the i-th head's query:
b Q i (x t ) = (W Q i ) ⊤ x t ∈ R d h • Non-contextual head factor:
This factor is a scaled standard basis vector, independent of x t :
a Q i = h • e i ∈ R
h where e i is the i-th standard basis vector (a vector of zeros with a one at the i-th position).
Substituting these into the TPA equation, the 1/R Q = 1/h scaling factor cancels with the scaling of the a Q i factor:
Q t = 1 h h i=1 (h • e i ) ⊗ (W Q i ) ⊤ x t = h i=1 e i ⊗ (W Q i ) ⊤ x t
The resulting tensor product, e i ⊗ b Q i (x t ), produces an h × d h matrix where only the i-th row is non-zero and contains the vector (b Q i (x t )) ⊤ . Summing these matrices for i = 1, . . . , h assembles the complete query tensor Q t , where the i-th row is precisely the query vector for the i-th head in standard MHA. An analogous construction applies to the key (K t ) and value (V t ) tensors. Thus, MHA is equivalent to a non-contextual TPA where the head-dimension factors are fixed and orthogonal, effectively assigning a dedicated rank component to each attention head.
this section cite: []

Section: MQA and GQA as Non-contextual TPA
Similarly, Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) can be seen as non-contextual TPAs where the key and value tensors are formed with a rank lower than the number of heads.
• MQA as Rank-1 TPA (for K and V). In MQA, all h query heads share a single key and value.
This corresponds to a TPA with ranks R K = 1 and R V = 1. The key tensor K t is formed using a single, non-contextual head-dimension factor a K = 1 h (a vector of all ones) and a single contextual token-dimension factor b K (x t ) = (W K ) ⊤ x t :
K t = 1 1 1 h ⊗ b K (x t )
This creates an h × d h matrix where every row is the same shared key vector (b K (x t )) ⊤ . The same logic applies to the value tensor V t . The queries remain full-rank (R Q = h) as in MHA.
• GQA as Rank-G TPA (for K and V). GQA is an intermediate approach where h heads are divided into G groups, with heads in the same group sharing a key and value. This is equivalent to a TPA with ranks R K = G and R V = G. The key tensor is formed by summing G components:
K t = 1 G G j=1 a K j ⊗ b K j (x t )
Here, b K j (x t ) is the shared key vector for group j. The non-contextual factor a K j is a scaled mask vector, defined as a K j = G • mask j , where the mask j vector has ones for heads belonging to group j and zeros elsewhere. This scaling cancels the 1/G pre-factor:
K t = 1 G G j=1 (G • mask j ) ⊗ b K j (x t ) = G j=1 mask j ⊗ b K j (x t )
For example, with h = 8 heads and G = 2 groups (2 KV heads), the factor for the first group of 4 heads would be a K 1 = 2 • [1, 1, 1, 1, 0, 0, 0, 0] ⊤ . This construction correctly assembles the final key tensor by broadcasting each group's shared key to its designated heads without any unintended extra scaling. This perspective highlights that MHA, MQA, and GQA are specific instances of a more general TPA framework, where expressiveness and parameter sharing are controlled by the rank and the nature (contextual vs. non-contextual) of the tensor factors.
this section cite: []

Section: Model Architectures
We propose a new architecture called Tensor ProducT ATTenTion Transformer (T6), which uses our Tensor Product Attention (TPA) in place of standard MHA (multi-head attention) or GQA (grouped-query attention). Building upon the query, key, and value tensors Q, K, V ∈ R T ×h×d h defined in Section 3.1, T6 utilizes the overall architecture of LLaMA [58] while changing the selfattention block to our TPA-based version. The feed-forward network (FFN) adopts a SwiGLU layer, as in [47,58].
this section cite: ['b57', 'b46', 'b57']

Section: Rotary Positional Embedding (RoPE).
As discussed in Section 3.2, RoPE [52] is applied to the Q and K. Within TPA, we pre-rotate the factor b Q t (x t ) and b K s (x s ) directly, so that each K s is already rotated prior to caching, see Equation (3.5) and Theorem 3.1. SwiGLU Feed-Forward Network. Following [47,58], our T6 uses a SwiGLU-based Feed-Forward Network (FFN): FFN(x) = σ(x W 1 ) ⊙ (x W 2 ) W 3 , where σ is the SiLU (a.k.a., swish) nonlinearity, ⊙ is element-wise product, and W 1 , W 2 , W 3 are learnable parameters. Note that other activation functions can also be used. Overall T6 Block Structure. Putting everything together, one T6 block consists of:
x ← x + TPA RMSNorm(x) , x ← x + SwiGLU-FFN RMSNorm(x) .
We place norm layers (e.g., RMSNorm) before each sub-layer. Stacking L such blocks yields a T6 model architecture with L layers.
this section cite: ['b51', 'b46', 'b57']

Section: FlashTPA Decoding Algorithm
For efficient autoregressive inference with Tensor Product Attention (TPA), we introduce FlashTPA Decoding. This algorithm is optimized for generating one token at a time by leveraging the factorized representation of queries, keys, and values. The core idea, illustrated in Figure 2, is to perform attention computations using a sequence of Einstein summations ("einsum") that operate directly on these factorized components. This avoids materializing the full query, key, and value tensors, which is particularly beneficial as the Key-Value (KV) cache grows with sequence length. The detailed definitions of the input factorized components and the step-by-step pseudo-code for FlashTPA Decoding are provided in Algorithm 2. An optimized Triton kernel implementation is outlined in Algorithm 3 (see Appendix B.1). or ⊙ denote Einstein summation contractions or element-wise products respectively, and the green rounded rectangle is the softmax operation. Shapes are shown for a single query (N = 1) interacting with M cached items in the common rank-1 setting R K = R V = 1. We use a head-first layout (H, M ) for logits and attention weights; the cached head factors a K cache and a V cache are shown transposed relative to their natural token-major layout for readability. H is the number of heads, R Q is the query rank, and D, E are respective feature dimensions for the B Q /b K cache and b V cache factors. Scaling factors are omitted for visual clarity. This sequence of factorized operations allows FlashTPA Decoding to compute the attention output efficiently. Consequently, TPA is not only memory-efficient due to its smaller KV cache footprint but can also be computationally efficient during inference. The experimental results for FlashTPA decoding time are presented in Section 6.2.
BQ (RQ, D) b K cache (M, D) D S (1) (M, RQ) AQ (H, RQ) RQ S (2) (M, H) a K cache (M, H) ⊙ L (H, M ) Softmax α (H, M ) a V cache (H, M ) ⊙ O (A) (H, M ) b V cache (M, E) M O (H, E)
this section cite: []

Section: Experiments

this section cite: []

Section: Language Modeling Tasks
All experiments reported in this paper are implemented based on the nanoGPT codebase [24], and we pretrain our models using the FineWeb-Edu 100B dataset [37]. The dataset contains 100 billion tokens for training and 0.1 billion tokens for validation. We compare T6 against the baseline Llama architecture [58] with SwiGLU activation [47] and RoPE embeddings [52], as well as Llama variants that replace Multi-Head Attention (MHA; [60]) with Multi-Query Attention (MQA; [46]), Grouped Query Attention (GQA; [2]), or Multi-head Latent Attention (MLA; [32]). In our experiments, the number of heads h is adjusted for each attention mechanism to ensure that all attention mechanisms have the same number of parameters as the standard Multi-Head Attention (MHA), which has 4d 2 model parameters per attention layer. We train models at four scales: small (124M parameters), medium (353M), large (773M), and XL (1.5B). We pretrain all models for 50B tokens (roughly half an epoch over FineWeb-Edu-100B). Details on architecture hyperparameters and training hardware are shown in Appendix H.1.
Training & Validation Curves. Figure 4 compares validation loss curves for the medium (353M), large (773M), and XL (1.5B) models on FineWeb-Edu-100B. Training loss curves are provided in Appendix Figure 3. Overall, TPA (red curves) and its simpler variant TPA-KVonly (pink curves) (see Appendix G) converge as fast as or faster than the baselines (MHA, MQA, GQA, MLA) while also achieving visibly lower final validation losses. For instance, in Figure 4(b), TPA and TPA-KVonly remain below the MHA baseline in terms of validation loss at nearly all training stages. Meanwhile, Multi-Head Latent Attention (MLA) [32] (blue curves) generally trains more slowly and yields higher validation losses. Validation Perplexity. Figure 9 (in the Appendix) shows the validation perplexities of the mediumand large-scale models. Mirroring the loss curves, TPA and TPA-KVonly steadily outperform MHA, MQA, GQA, and MLA over the course of training. By the end of pretraining (around 49B tokens), TPA-based approaches achieve the lowest perplexities in most configurations. Downstream Evaluation. We evaluate zero-shot and two-shot performance on standard benchmarks, including ARC [63], BoolQ [13], HellaSwag [64], OBQA [39], PIQA [4], WinoGrande [43], and MMLU [18], using the lm-evaluation-harness codebase [14]. For ARC-E, ARC-C, Hel-laSwag, OBQA, PIQA, and SciQ, we report accuracy norm; for other tasks, we report standard accuracy. Due to the page limitation, we only display the zero-shot evaluation results of medium and large models here in Tables 2 and 3. Zero-shot evaluation of small and XL models are displayed in Tables 11 and 12 in the appendix. Moreover, we also present 2-shot evaluation results in Tables 13, 14, 15 and 16 in the appendix. For the medium-size (353M) models (Table 2 for 0-shot and Table 14 in appendix for 2-shot), TPA generally ties or outperforms all competing methods, achieving, for example, an average of 51.41% in zero-shot mode versus MHA's 50.11%, MQA's 50.44%, and MLA's 50.13%. When given two-shot prompts, TPA again leads with 53.12% average accuracy. A similar trend appears for the large-size (773M) models (Table 3), where TPA-KVonly attains the highest average (53.52% zero-shot). For the XL size models (1.5B) (Table 12 in the appendix), TPA-KV only achieves the highest average (55.03% zero-shot). Our experiments confirm that TPA consistently matches or exceeds the performance of established attention mechanisms (MHA, MQA, GQA, MLA) across medium and large model scales. (a) Medium models (353M) (b) Large models (773M) (c) XL models (1.5B) Figure 4: The validation loss of medium-size (353M), large-size (773M) as well as XL-size (1.5B) models, with different attention mechanisms on the FineWeb-Edu 100B dataset.
Table 2: The evaluation results of medium models with different attention mechanisms pre-trained using FineWeb-Edu 100B dataset (0-shot with lm-evaluation-harness). The best scores in each column are bolded. Abbreviations: HellaSw. = HellaSwag, W.G. = WinoGrande. Method ARC-E ARC-C BoolQ HellaSw. OBQA PIQA W.G. MMLU SciQ Avg. MHA 59.51 29.52 59.60 45.68 34.20 68.82 53.43 23.33 76.90 50.11 MQA 57.62 31.91 59.45 45.69 35.40 69.31 53.51 26.47 74.60 50.44 GQA 58.67 31.48 58.29 45.45 35.20 68.50 54.46 24.58 76.50 50.35 MLA 56.65 29.52 57.83 46.05 34.60 69.42 52.80 24.62 79.70 50.13 TPA-KVonly 58.01 30.12 58.01 45.95 35.60 69.10 53.12 25.39 75.10 50.04 TPA 58.38 31.57 59.39 46.83 37.00 70.02 54.06 25.52 79.90 51.41
this section cite: ['b23', 'b36', 'b57', 'b46', 'b51', 'b59', 'b45', 'b1', 'b31', 'b31', 'b62', 'b12', 'b63', 'b38', 'b3', 'b42', 'b17', 'b13']

Section: Experimental Results on FlashTPA Decoding
This section presents an evaluation of FlashTPA's decoding time in comparison to several other optimized attention mechanisms. We benchmark FlashTPA against FlashMHA [45], FlashGQA, FlashMQA, and FlashMLA [23]. It is important to note that our current FlashTPA implementation utilizes Triton [57]. While the compared methods are typically available as highly optimized CUDA kernels, these experiments provide initial insights into FlashTPA's potential. Development of a CUDAbased FlashTPA kernel is ongoing and is expected to yield further performance improvements.  The evaluations were performed with batch sizes selected from {1, 2, 4, 8, 16}, model embedding dimensions (d model ) chosen from {1024, 2048, 3072}, and sequence lengths ranging from 2 12 (4,096) to 2 19 (524,288). For all experiments, the dimension per head (d h ) was fixed at 64. The ranks for TPA's factorized components (R Q , R K , R V ) were set to (16, 1, 1), and for GQA configurations, the number of key-value head groups was 4. The decoding time per token, measured as log 2 (time) in seconds, is plotted against log 2 (sequence length). Lower values on the y-axis indicate faster decoding times. Results are presented in Figure 5 for an embedding dimension of 2048 (corresponding to 32 attention heads). Additional results for embedding dimensions of 1024 (16 heads, Figure 8) and 3072 (48 heads, Figure 7) are provided in Appendix B.
this section cite: ['b44', 'b22', 'b56']

Section: Conclusion
We introduced Tensor Product Attention (TPA), which factorizes query, key, and value matrices into rank-R tensor products dependent on the token's hidden state. Storing only the factorized key/value components during autoregressive decoding substantially decreases the KV memory size with improved performance compared with MHA, MQA, GQA, and MLA. The approach is fully compatible with RoPE (and can store pre-rotated keys). Variants of TPA include factorizing only the key/value or sharing basis vectors across tokens. Overall, TPA offers a powerful mechanism for compressing KV storage while improving the model performance, thereby enabling longer sequence contexts under constrained memory.
A Toward Faster Computation Without Materializing Q, K and V Our objective in this section is to compute attention without explicitly forming Q, K, V, by contracting their factorized representations in a cache-and throughput-friendly order. Recall from Equation (3.1) that each per-token slice Q t , K t , V t ∈ R h×d h is a sum of rank-1 outer products. Unless otherwise stated we use the per-factor normalizations s
Q =1/R Q , s K =1/R K , s V =1/R V .
We make the batch/time/head/rank/value dimensions explicit and introduce the shorthands D := d h and E := d v (typically E=D):
A Q ∈ R B×Tq×R Q ×H , B Q ∈ R B×Tq×R Q ×D , A K ∈ R B×T k ×R K ×H , B K ∈ R B×T k ×R K ×D , A V ∈ R B×T k ×R V ×H , B V ∈ R B×T k ×R V ×E .
Indices b, q, k, h, r, s, u, d, e denote batch, query position, key position, head, query-rank, key-rank, value-rank, feature (D), and value feature (E). We write T := T q = T k for full-sequence attention; in decoding, T q = 1 and we denote the cache length by M = T k .
Convention. For a single token, the main text defines A * (x t ) ∈ R R * ×H and B * (x t ) ∈ R R * ×D , with
Q t = 1 R Q A Q (x t ) ⊤ B Q (x t ).
Accordingly, throughout this appendix we index A Q as A Q [b, q, r, h] (rank-major). Some implementations may store A * transposed as (H × R * ) for memory layout, this is equivalent, since all uses contract over the rank index.
High-level idea. We first compute head-shared feature-space dot products between B Q and B K , then mix them with head-specific A Q , A K to obtain logits, apply the masked softmax, and finally aggregate values via A V , B V . This ordering avoids materializing any T q × h × D queries/keys/values.
this section cite: []

Section: References
Ref_id:b0 Title: Keyformer: Kv cache reduction through key tokens selection for efficient generative inference Year: (2024)
Ref_id:b1 Title: GQA: training generalized multi-query transformer models from multi-head checkpoints Year: (2023)
Ref_id:b2 Title: Qwen technical report Year: (2023)
Ref_id:b3 Title: PIQA: reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b4 Title: Language models are few-shot learners Year: (2020)
Ref_id:b5 Title: Sparks of artificial general intelligence: Early experiments with gpt-4 Year: (2023)
Ref_id:b6 Title: Olora: Orthonormal low-rank adaptation of large language models Year: (2024)
Ref_id:b7 Title: Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling Year: (2024)
Ref_id:b8 Title: Longlora: Efficient fine-tuning of long-context large language models Year: (2024)
Ref_id:b9 Title: Generating long sequences with sparse transformers Year: (2019)
Ref_id:b10 Title: Rethinking attention with performers Year: (2021)
Ref_id:b11 Title: Palm: Scaling language modeling with pathways Year: (2023)
Ref_id:b12 Title: Boolq: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b13 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b14 Title: Understanding the difficulty of training deep feedforward neural networks Year: (2010)
Ref_id:b15 Title: Hyperattention: Long-context attention in near-linear time Year: (2024)
Ref_id:b16 Title: Fastdecode: High-throughput gpu-efficient llm serving using heterogeneous pipelines Year: (2024)
Ref_id:b17 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b18 Title: Towards 10 million context length llm inference with kv cache quantization Year: (2024)
Ref_id:b19 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b20 Title: Multi-matrix factorization attention Year: (2024)
Ref_id:b21 Title: High-rank updating for parameterefficient fine-tuning Year: (2024)
Ref_id:b22 Title: Flashmla: Efficient mla decoding kernels Year: (2025)
Ref_id:b23 Title:  Year: (2022)
Ref_id:b24 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b25 Title: {InfiniGen}: Efficient generative inference of large language models with dynamic {KV} cache management Year: (2024)
Ref_id:b26 Title: A tighter complexity analysis of sparsegpt Year: (2024)
Ref_id:b27 Title: Relora: Highrank training through low-rank updates Year: (2023)
Ref_id:b28 Title: Inflora: Interference-free low-rank adaptation for continual learning Year: (2024)
Ref_id:b29 Title: Conv-basis: A new paradigm for efficient attention inference and gradient computation in transformers Year: (2024)
Ref_id:b30 Title: Beyond linear approximations: A novel pruning approach for attention matrix Year: (2024)
Ref_id:b31 Title: Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model Year: (2024)
Ref_id:b32 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b33 Title: KIVI: A tuning-free asymmetric 2bit quantization for KV cache Year: (2024)
Ref_id:b34 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b35 Title: Stochastic gradient descent with warm restarts Year: (2016)
Ref_id:b36 Title:  Year: (2024)
Ref_id:b37 Title: A kernel-based view of language model fine-tuning Year: (2023)
Ref_id:b38 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b39 Title: fairseq: A fast, extensible toolkit for sequence modeling Year: (2019)
Ref_id:b40 Title: Analyzing and reducing catastrophic forgetting in parameter efficient tuning Year: (2024)
Ref_id:b41 Title: Sparq attention: Bandwidth-efficient LLM inference Year: (2024)
Ref_id:b42 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2020)
Ref_id:b43 Title: Linear transformers are secretly fast weight programmers Year: (2021)
Ref_id:b44 Title: Flashattention-3: Fast and accurate attention with asynchrony and low-precision Year: (2024)
Ref_id:b45 Title: Fast transformer decoding: One write-head is all you need Year: (2019)
Ref_id:b46 Title: Glu variants improve transformer Year: (2020)
Ref_id:b47 Title: Loldu: Low-rank adaptation via lower-diag-upper decomposition for parameter-efficient fine-tuning Year: (2024)
Ref_id:b48 Title: The trade-off between universality and label efficiency of representations from contrastive learning Year: (2023)
Ref_id:b49 Title: Loki: Low-rank keys for efficient sparse attention Year: (2024)
Ref_id:b50 Title: The extreme pull between cache and effect: From MHA, MQA, GQA to MLA Year: (2024-05)
Ref_id:b51 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b52 Title: Kv cache in shadows for high-throughput long-context llm inference Year: (2024)
Ref_id:b53 Title: Retentive network: A successor to transformer for large language models Year: (2023)
Ref_id:b54 Title: QUEST: query-aware sparsity for efficient long-context LLM inference Year: (2024)
Ref_id:b55 Title: Open models based on gemini research and technology Year: (2024)
Ref_id:b56 Title: Triton: An intermediate language and compiler for tiled neural network computations Year: (2019)
Ref_id:b57 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b58 Title: Transformer dissection: An unified understanding for transformer's attention via the lens of kernel Year: (2019)
Ref_id:b59 Title: Attention is all you need Year: (2017)
Ref_id:b60 Title: Smoothquant: Accurate and efficient post-training quantization for large language models Year: (2023)
Ref_id:b61 Title: Efficient streaming language models with attention sinks Year: (2024)
Ref_id:b62 Title: Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering Year: (2019)
Ref_id:b63 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019)
Ref_id:b64 Title: The expressive power of low-rank adaptation Year: (2024)
Ref_id:b65 Title: Sinklora: Enhanced efficiency and chat capabilities for long-context large language models Year: (2024)
Ref_id:b66 Title: The hedgehog & the porcupine: Expressive linear attentions with softmax mimicry Year: (2024)
Ref_id:b67 Title: Adaptive budget allocation for parameter-efficient fine-tuning Year: (2023)
Ref_id:b68 Title: Trained transformers learn linear models in-context Year: (2023)
Ref_id:b69 Title: H2o: Heavy-hitter oracle for efficient generative inference of large language models Year: (2023)
Ref_id:b70 Title: Continual forgetting for pre-trained vision models Year: (2024)
