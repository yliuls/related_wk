Title: Two Heads Are Better than One: Simulating Large Transformers with Small Ones
Abstract: The quadratic complexity of self-attention prevents transformers from scaling effectively to long input sequences. On the other hand, modern GPUs and other specialized hardware accelerators are well-optimized for processing small input sequences in transformers during both training and inference. A natural question arises: can we take advantage of the efficiency of small transformers to deal with long input sequences? In this paper, we show that transformers with long input sequences (large transformers) can be efficiently simulated by transformers that can only take short input sequences (small transformers). Specifically, we prove that any transformer with input length N can be efficiently simulated by only O((N/M ) 2 ) transformers with input length M ≪ N , and that this cannot be improved in the worst case. However, we then prove that in various natural scenarios including average-case inputs, sliding window masking and attention sinks, the optimal number O(N/M ) of small transformers suffice.

Section: Introduction
The transformer architecture [VSP + 17] has revolutionized modern machine learning, natural language processing and computer vision. It achieves state-of-the-art performance on various tasks such as language reasoning [Dee21,Ope20], image recognition [KDW + 21, CMS + 20] and many others. At the core of the transformer architecture is the attention mechanism, which captures correlations between all pairs of tokens. However, this is also a major bottleneck for transformers, as the quadratic complexity (in both time and memory) of the attention mechanism prohibits effective scaling of transformers as the sequence grows in length. Moreover, it has been theoretically proved that the quadratic complexity cannot be avoided (under popular complexity-theoretic assumptions) [AS24]. To address this fundamental issue, there has been a fruitful literature on the design of "subquadratic alternatives" to transformers, where researchers come up with mechanisms that replace the attention mechanism and take subquadratic time (usually close to linear time) [KKL20, CLD + 21, DFE + 22, KMZ24, BPC20, GD23]. However, they usually have worse performance than standard transformers, especially on downstream tasks and translation [VPSP23,JBKM24,AY25].
In the meantime, modern GPUs are increasingly optimized for handling short-to-moderate transformer contexts [WXQ + 21, DFE + 22]. Some companies are even producing specialized hardware for efficient transformer inference that have superior performance on inputs of length between 128 to 2048 [Kim24,Etc24]. This approach motivates the following questions:
Can we use small transformers to perform tasks more efficiently than large transformers? Are (multiple) small transformers inherently capable of dealing with long contexts?
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
In this paper, we give positive answers to these questions through the lens of representational strength, which studies whether one can select parameters for transformers so that they can perform certain tasks of interest. The representational strength of transformers has been studied broadly in recent years [BHBK24, LAG + 23, SHT24, MS23], and it is believed that it is one of the core reasons why transformers outperform previous architectures such as RNN and LSTM [WDL25, AET + 24, SHT23].
Our problem can be stated as follows. Suppose that we have all the parameters of a large transformer T with input length N , as well as an input X that we would like to evaluate T on. However, to evaluate T (X), we are not allowed to perform any particularly complicated computations; we are restricted to simple operations, and to making use of a small transformer O (as an oracle) that can only take input sequences of length M ≪ N . We can input into O any sequence and parameters that we can easily compute, and obtain its output. Our goal is to minimize the number of calls to O that we need to obtain T (X) for arbitrary input X of length N .
Our main results show that roughly O((N/M ) 2 ) oracle calls suffice, which we show is optimal. In addition, our algorithm requires minimal processing outside of the oracle calls, and it has properties needed for efficient training and inference, including that the gradients of its parameters are easily computed, and that its oracle calls can be computed in parallel in only O(L) rounds of adaptivity, where L is the number of layers in the large transformer T .
In addition, we show that in many scenarios arising in practice, such as when certain masking schemes are used, or when the data is not "worst-case" and satisfies some boundedness guarantees, the information-theoretically optimal O(N/M ) oracle calls suffice.
Our results provide a new way to deal with long input sequences for transformers, as we prove that any computation performed by large transformers can be decomposed into computations that only use smaller transformers. If the oracles are implemented using a quadratic number of floating-point operations, then our algorithm also still requires a quadratic amount of floating-point operations.
However, if modern GPUs enable faster transformer inference with respect to the "wall-clock" time when the input sequence is short-to-moderate, then our algorithms allow faster wall-clock time inference. For example, if the oracle can compute the output using O(M ) wall-clock time compared to the standard O(M 2 ) time, then the total wall-clock running time of our algorithms will be O(N 2 /M ).
Our approach is fundamentally different from designing "subquadratic alternatives" to transformers [KKL20, CLD + 21, BPC20, KMZ24, GD23]. In particular, our algorithm preserves the representational strength of transformers (or even improves it), whereas it has been shown that all the subquadratic alternatives to transformers will lose representational strength as they cannot capture all the pairwise relationship even approximately [AY25].
Now we define our model of computation and state our main contributions in more detail.
this section cite: ['b8', 'b27', 'b1', 'b33', 'b17', 'b2', 'b19', 'b10', 'b2']

Section: Computational Model for Simulating Large Transformers
We now describe our model of computation in more detail. We are careful to allow only very simple operations beyond oracle calls, to ensure that the vast majority of computation can be performed by efficient hardware for evaluating small transformers, and that the number of oracle calls accurately measures the complexity of the problem.
We are given a large transformer T with input length N , L layers, H attention heads in each layer, and embedding dimension d (all the parameters, including the query, key, value matrices in each of its attention head and multilayer perceptron functions). Throughout this paper, we assume that L, H ≪ N, d = O(log N ), Ω(log N ) ≤ M < o(N ), and one can typically imagine M ≈ √ N . Our goal is to design an algorithm that (approximately) output T (X) ∈ R N ×d for arbitrary input X (length at most N ).
We have a limited set of operations we can perform as part of the algorithm. We criticaly have access to a small transformer (oracle) O that can take as input a sequence of length at most M ≪ N , as well as the parameters for a transformer which has L ′ layers and H ′ attention heads in each layer, and outputs the transformer evaluated on that sequence. Our algorithm is allowed to:
1. Feed the oracle O with input sequences and parameters which are currently in memory to obtain its output; 2. Processing: Edit existing vectors or matrices in memory by padding at most O(d 2 ) fixed numbers (constants independent of the input) to them, or arranging (concatenating) matrices in memory.
We also assume that all the numbers in input matrices, parameters, and algorithms have O(log N )-bit representations. We say that such an algorithm simulates T if it always outputs a Y ∈ R N ×d such that ∥Y [i, :] -T (X)[i, :]∥ 2 ≤ ε∥T (X)[i, :]∥ 2 for all i ∈ [N ], and for very small error ε = Θ( 1 2 N ). We want to design algorithms that simulate T with as fewer oracle calls as possible. (Such an ε is essentially unavoidable in limited precision architectures, but we will see it will be very helpful in some algorithms below. We also emphasize that our main result, Theorem 1.1, is an exact computation in the unlimited precision scenario with ε = 0.)
Notice that any such algorithm can be viewed as a composition of oracles and the padding function. Since we only allow for very simple processing, it is straightforward to compute the gradients of the padding functions, so training the large model could be done via computing the gradients of the small transformer oracles.
this section cite: []

Section: Main Results
Quadratic small transformers are sufficient and necessary for worst-case inputs. As summarized below, our main result shows that any computation performed on a large transformer can be decomposed into multiple instances of computation performed on smaller transformers with the same computational complexity or floating-point operations. Since the oracle can only tell us the final output instead of intermediate embeddings, it might be somewhat surprising that we are able to utilize all the layers in small transformers. Theorem 1.1 (Theorem 3.4, Theorem 3.5). For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d, there exists an algorithm that simulates T with O(( N M ) 2 • HL H ′ L ′ ) calls to a transformer oracle with L ′ layers, H ′ attention heads in each layer, input length M , embedding dimension O( dH ′ L ′ H ). The result still holds when we add causal masking to both large and small transformers.
Notice that these simulations are tight, and roughly (N/M ) 2 oracle calls are necessary in the worstcase due to computational complexity constraints. To see this, note that when L = H = L ′ = H ′ = 1, a straightforward algorithm can compute the responses of T oracle calls in time only Õ(T M 2 ). Thus, since it is known that even approximation of a large attention requires time Ω(N 2-o(1) ) (under standard complexity-theoretic assumptions) [AS24], we must have T ≥ ((N/M ) 2 ) 1-o(1) .
One might be concerned with the fact that O((N/M ) 2 ) small transformers have many more parameters than one large transformer, since each transformer has Θ(d 2 ) parameters, independent of the sequence length. However, this is not a problem because in our construction, we reuse the parameters such that the total number of parameters does not depend on N . In fact, all the query, key and value matrices that we feed into the oracle share most entries with the query, key, value matrices in the large transformer that are given. We ultimately only have a small, constant factor blowup on the number of parameters.
Linear small transformers are weaker but sufficient with average-case inputs. Even though we cannot use O(N/M ) oracle calls to simulate a large transformer in the worst case, we show that it is possible when we have reasonable additional assumptions on the queries, keys and values. Theorem 1.2 (Informal version of Theorem 4.1). Let T be a transformer with L layers, H attention heads in each layer, input length N and embedding dimension d. Suppose that the queries, keys and values in the attention heads are all somewhat bounded in how much they may differ from each other. Then, there exists an algorithm using O( N M • HL H ′ L ′ ) oracle calls to simulate T .
Models such as Hierarchical Transformers [PZV + 19, LL19, CDF + 22] split the input sequence into chunks of size M and send each chunk into a transformer before aggregating the outputs. We give the first provable guarantees for this approach, showing that O(N/M ) small transformers have approximately equal expressivity as a large transformer when the input data satisfies our assumptions.
This provides a possible explanation of the success of Hierarchical Transformers and relevant ideas from an expressivity viewpoint.
On the other hand, we supplement Theorem 1.2 with its converse, which shows that a linear number of small transformers are at most as expressive as one large transformer for worst-case inputs (we only prove the statement for single-head transformers to illustrate the message). As a result, when the inputs follow assumptions in Theorem 1.2, a linear number of small transformers are equivalent to a larger one in expressive power.
this section cite: ['b1']

Section: Theorem 1.3 (Theorem 4.2).
Given N/M instances of single layer, single head transformers with input length M and embedding dimension d, there exists an algorithm that simulates them with one call of a single layer, single head transformer with input length O(N ) and embedding dimension O(d), along with O(N/M ) many matrix multiplications of size M × d × d.
We briefly comment on the small matrix multiplications in Theorem 1.3. Note that they could be computed in the straightforward way in nearly linear time O(M d 2 ) (since d = O(log N )) and thus do not substantially contribute to the total running time. This implies, in particular, that they could not simulate the transformer oracles on their own, and are only "assisting" the large transformer oracle. Their presence seems unavoidable because of the Θ(N/M ) weight matrices of the oracles which must be simulated by a single large transformer, which only has a constant number of weight matrices. Moreover, we emphasize that our other constructions are even simpler, and do not need such small matrix computations outside of the oracle calls.
Efficient simulation of transformers with sliding window and StreamingLLMs. Sliding window and StreamingLLM [XTC + 24] are popular ways to make transformer inference more memory efficient. Both sliding window and StreamingLLM are based on the observation that certain attention scores are often higher than others. Sliding window is based on the intrinsic structure of languages, where each token is typically more correlated to the previous few tokens. Therefore, for each query we only take into account the contributions of the keys that are positionally close to it. The StreamingLLM framework is motivated by the observation that autoregressive LLMs have a surprisingly large amount of attention score concentrated to the initial tokens, and thus each query only takes into account keys that are positionally close to it, as well as the first few (usually 3 ∼ 5) keys, which are called "attention sinks".
We show that in both cases we can use a linear number of small transformer oracle calls to simulate them, even in the worst case. As summarized below, our result indicates that oracles can capture efficient attention based on sliding windows and attention sinks efficiently. Theorem 1.4 (Theorem 5.1). For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d, constant-size sliding window, there exists an algorithm that simulates T with O( N M • HL H ′ L ′ ) calls to a transformer oracle with L ′ layers, H ′ attention heads in each layer, input length M and embedding dimension O( dH ′ L ′ H ) with causal masking. This result still holds if we have constant-size attention sinks.
this section cite: []

Section: Related Work
Representational strength and limitations of transformers. The representational strength of transformers has been intensively studied in recent years from a variety of perspectives. To list a few, [MSS22, MS23, SMW + 24] study the class of problems that transformers can solve from a circuit complexity viewpoint; [BAG20, LAG + 23, Hah20] aim to understand whether transformers can recognize formal languages; [SHT23] focus on reasoning tasks and show that transformers are inherently capable of solving sparse averaging; [SHT24] gives important connections between transformers and the massively parallel computation model; [HSK + 25, HWL + 24] uses computational complexity to characterize the computational limits of diffusion transformers and low-rank adaptation for transformers; [LCW23] studies attention's capability of approximating sparse matrices; [YBR + 20, YCB + 20] shows that transformers and many subquadratic variants are universal approximators for sequence-to-sequence functions;
Fast attention mechanisms. There has been a fruitful literature of dealing with long input sequence by designing "subquadratic alternatives" to transformers, which are variants on the attention mechanism which can be performed in subquadratic time. For example, researchers have studied various sparse attention mechanisms that only consider the query-key pairs that have high correlation, including Reformer [KKL20], Longformer [BPC20], and Hyperattention [HJK + 24]. Additionally, there has been work on kernel/low-rank attention that approximates attention mechanism using kernels such as Performer [CLD + 21] and Polysketchformer [KMZ24], and there has been a growing interest in state space models such as Mamba [GD23]. See [TDBM22] for a comprehensive survey on efficient attention mechanisms. However, [AY25] proves that none of these subquadratic models can capture all pairwise correlations even approximately as the sequence length grows.
this section cite: ['b29', 'b30', 'b20', 'b5', 'b21', 'b11', 'b32', 'b2']

Section: Preliminaries

this section cite: []

Section: Transformers
We first define the standard attention mechanism in Transformer. Definition 2.1 (Attention Mechanism). Given input X ∈ R N ×d , query, key, value matrices
W Q , W K , W V ∈ R d×m , the attention mechanism computes Attn(X) = softmax((XW Q )(XW K ) ⊤ )(XW V ) ∈ R N ×m .
Here we say N is the context length, and d is the embedding dimension. We will also call each attention mechanism an attention head in the transformer architecture. For notational convenience, we let {q 1 , . . . , q N } ∈ R m , {k 1 , . . . , k N } ∈ R m , {v 1 , . . . , v N } ∈ R m be the rows of XW Q , XW K , XW V respectively. We will call them the queries, keys and values. As a result, the attention mechanism is computing
1 N j=1 exp(⟨q i , k j ⟩) N j=1 exp(⟨q i , k j ⟩) • v j
for each query q i .
Another important component in transformers is the multilayer perceptron (MLP). An MLP is a feed-forward, fully-connected neural network consisting of one or more hidden layers using ReLU activation. The universal approximation theorem states that any continuous function with a finite support can be approximated by a neural network with one hidden layer [HSW89]. In light of this, in many relevant works [SHT23,SHT24], MLPs are modeled as arbitrary functions on compact domains.
In this paper, our goal is to use small transformers to simulate large transformers, and we would like to ensure that the MLPs in the small transformers are as simple as possible. We will therefore assume that MLPs in small transformers compute functions ϕ : R d → R O(d) such that:
1. They are at least as strong as the MLPs in the large transformers, i.e. they can do whatever computation that MLPs in the large transformers can do, and 2. They can do basic arithmetic operations on the input vector x ∈ R d or pad fixed numbers to it (both are simple continuous functions) as long as they take O(d 2 ) time.
When a MLP ϕ is applied on a matrix, it will be applied row-wise to output another matrix. In other words, it is applied on each token given a sequence of tokens.
An attention layer f with H attention heads consists of H attention mechanisms with query embedding W Q h , W K h , W V h ∈ R m×m for the h-th attention such that m = d H . The input X is partitioned into H matrices X[:, D 1 ], . . . , X[:, D H ] ∈ R N ×m column-wise, where D i = { (i-1)d H + 1, . . . , id H } for all i, such that the h-th attention head computes softmax((X[:,
D h ]W Q h )(X[:, D h ]W V h ) ⊤ )(X[:, D h ]W V h ) ∈ R N ×m .
All attention outputs are concatenated column-wise and fed through a layer MLP ψ such that the output of attention layer f is
f (X) := ψ softmax((X[:, D h ]W Q h )(X[:, D h ]W V h ) ⊤ )(X[:, D h ]W V h ) H h=1 ∈ R N ×d .
Definition 2.2 (Transformer). A transformer T with L layers and H attention heads in each layer consists of an input MLP ϕ : R d ′ → R d applied token-wise on the input X ∈ R N ×d ′ , L attention layers f 1 , . . . , f L : R N ×d → R N ×d which contain L layer MLPs ψ 1 , . . . , ψ L applied token-wise at the end of each attention layer. For each 2 ≤ ℓ ≤ L,
X (1) = ψ 1 (f 1 (ϕ(X))), X (ℓ) = ψ ℓ (f ℓ (X (ℓ-1) )).
Finally, the transformer T outputs T (X) = X (L) .
We will simplify the notion of positional encoding into input MLP ϕ and assume that the input MLP has positional information of the tokens. In other words, if x i = X[i, :] is the i-th input token, then ϕ(x i ) is also a function of i.
Transformer is powerful as computational model, and we refer the readers to Appendix A for more operations that transformers can do that will be useful in our proofs.
this section cite: ['b15', 'b29', 'b30']

Section: Transformer Oracle.
A transformer oracle O is a small transformer that can only take inputs of length at most M (its embedding dimension, number of layers, number of heads in each layer, causal masking etc will be specified in result statements). In this paper we are mostly concerned with simulating transformers with large input length N using transformer oracles with input length M such that M ≪ N (recall that we assume Ω(log N ) ≤ M < o(N )).
this section cite: []

Section: Causal masking, sliding window and StreamingLLM
We first define the most commonly used causal masking in attention heads.
this section cite: []

Section: Definition 2.3 (Causal Masking Attention).
Given input X ∈ R N ×d , query, key, value matrices W Q , W K , W V ∈ R d×m , the attention mechanism with causal masking computes
Attn(X) = softmax mask((XW Q )(XW K ) ⊤ ) (XW V ) ∈ R N ×m ,
where the mask function sets all upper triangular entries (not including diagonal entries) to -∞.
Another commonly used masking scheme for efficient transformer inference/training is sliding window, where we only keep the keys whose indices are close to the query index.
this section cite: []

Section: Definition 2.4 (Sliding Window Attention).
Given input X ∈ R N ×d , query, key, value matrices W Q , W K , W V ∈ R d×m and window size r ≥ 1, the attention mechanism with sliding window of size r computes
Attn(X) = softmax window((XW Q )(XW K ) ⊤ ) (XW V ) ∈ R N ×m ,
where the window function sets all entries in {(i, j) : j > i or j ≤ i -r} to -∞.
In other words, for each query q i we only look at k j such that i -r + 1 ≤ j ≤ i.
Finally, StreamingLLM [XTC + 24] is a framework designed for efficient training with a finite length window. Upon having a fixed-size sliding window, each query also attends to the first s keys (called "attention sinks"), where s is usually a small positive constant (around 3 ∼ 5).
this section cite: []

Section: Definition 2.5 (Attention Sink).
Given input X ∈ R N ×d , query, key, value matrices W Q , W K , W V ∈ R d×m and window size r ≥ 1, sink size s ≥ 1, the attention mechanism with attention sink computes
Attn(X) = softmax sink((XW Q )(XW K ) ⊤ ) (XW V ) ∈ R N ×m ,
where the sink function sets all entries in {(i, j) : j > i or s < j ≤ i -r} to -∞.
Transformers with causal masking attention, sliding window, and StreamingLLMs are defined exactly the same as transformers except that we replace standard attention mechanisms by attention with causal masking, attention with sliding window and attention with sinks.
this section cite: []

Section: Notation
Throughout the paper, we denote X ∈ R N ×d as the input to the large transformer, where N is the input length and d is the embedding dimension. For a N × d matrix X, we use X[i, :] to denote its i-th row, X[:, j] to denote its j-th column, and X[i, j] to denote its (i, j)-th entry. Given sets S ⊆ [N ], D ⊆ [d], we use X[S, :] to denote the submatrix consisting of the rows in S, X[:, D] to denote the submatrix consisting of the columns in D, and X[S, D] to denote the submatrix consisting of the entries in S × D of X.
We use W Q , W K , W V to denote the query, key and value matrices for attention heads, and we use q i , k i , v i to denote the i-th row of XW Q , XW K , XW V respectively. We also let S t = {(t -1)M + 1, tM } for all 1 ≤ t ≤ N/M .
We use 1 a×b to denote the a × b matrix whose entries are all 1, and 0 a×b to denote the a × b matrix whose entries are all 0.
We now turn to proving our results. We give proof sketches and main ideas here; full proofs are deferred to the appendix.
this section cite: []

Section: Quadratic calls suffice for simulation
In this section, we prove that O(( N M ) 2 • HL H ′ L ′ ) small transformers with L ′ layers and H ′ attention heads in each layer suffice to simulate a large transformer with L layers and H attention heads in each layer (Theorem 3.4). Our proof roadmap is illustrated in Figure 1, where the arrows A → B indicate that A can be simulated by B. Complete proofs of all the statements can be found in Appendix B. We first show that this is the case when they both only have a single attention head, i.e H ′ = H = L = L ′ = 1. Lemma 3.1. For any single layer, single head transformer T with input length N , embedding dimension d, there exists an algorithm that simulates T with O( N 2 M 2 ) calls to a transformer oracle with input length M and embedding dimension O(d).
Proof Sketch. Our high-level idea is to partition the N × N attention matrix of T into N 2 M 2 blocks of size M × M , and then use a constant number of oracles to separately handle each block. In particular, each block corresponds to two sub-intervals of length M out of the input sequence of length N (one interval for the rows, or queries, and one interval for the columns, or keys), so we can aim to have an oracle for sequence length O(M ) compute the contribution of each block. To be more precise, as in Definition 2.1 above, let
{q 1 , . . . , q N } ∈ R m , {k 1 , . . . , k N } ∈ R m , {v 1 , . . . , v N } ∈ R m
be the rows of XW Q , XW K , XW V respectively. Define a i,j = exp(⟨q i , k j ⟩), and b i,j = exp(⟨q i , k j ⟩) • v j .
The goal of the attention mechanism is to compute, for all i,
1 N j=1 exp(⟨q i , k j ⟩) • N j=1 exp(⟨q i , k j ⟩) • v j = N j=1 b i,j N j=1 a i,j = N/M t=1 j∈St b i,j N/M t=1 j∈St a i,j .
The main technical difficulty is that one oracle call is not able to give us information on the sum of a i,j (or b i,j ) over all j ∈ [N ]. However, we show that one oracle call allows us to compute j∈St a i,j , where S t = {(t -1)M + 1, . . . , tM } such that N/M oracle calls suffice to give us N j=1 a i,j . We do this by adding in one synthetic token to the sequence so that its contribution is fixed, i.e. its inner product with all the keys will be the same and known. In addition, we assign its corresponding value token to be 0 and other value tokens to be 1 such that the output does not contain the synthetic token's value, while the normalizing term still counts its contribution. As a result, the output of the oracle will give us j∈St a i,j j∈St a i,j + a , where a is the attention value (that we set and thus know in advance) for the normalizing term. This information allows us to compute j∈St a i,j as we can solve a linear equation using MLP. Secondly, we will directly feed the oracle with X[S t , :],
W Q , W K , W V to obtain j∈St b i,j j∈St a i,j
, and since we already know j∈St a i,j , we can compute j∈St b i,j , which will furthermore give us N j=1 b i,j by summing them up.
We now move on to generalizing Lemma 3.1 to general H, L, H ′ , L ′ , i.e., when the transformer T and the oracles can have multiple heads and layers. A first attempt to do this might use different layers of O to simulate different layers of T , but this appears difficult to implement, since Lemma 3.1 requires some processing between layers of attentions that is not available to us when the different layers are connected only through MLPs within an oracle. Indeed, since the output of each attention head needs processing before it can be used in another attention head, it is unclear how to take advantage of more than one layer of each oracle in this way. We instead take a different approach: we use all the attention heads in all the layers of the oracle completely independently from each other to simultaneously simulate Θ(H ′ L ′ ) different attention heads, and we use these all together to simulate one layer at a time of T . First, it is not hard to generalize Lemma 3.1 to general H, L (but still H ′ = L ′ = 1) by separately simulating each attention head in T regardless of which layer it is in: Corollary 3.2. For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d, there exists an algorithm that simulates T with O(( N M ) 2 • HL) calls to a single head, single layer transformer oracle with input length M and embedding dimension O( d H ).
Next we show that a transformer with L ′ layers, H ′ attention heads in each layer, input length M , and embedding dimension O( dH ′ L ′ H ) can be used to simulate H ′ L ′ instances of single head, single layer transformers with input length M and embedding dimension d H . In other words, we are able to independently use each attention head in the transformer, regardless of which of the L ′ layers it appears in: Lemma 3.3. One transformer with L ′ layers, H ′ attention heads in each layer, input length M and embedding dimension O( dH ′ L ′ H ) can be used to simulate H ′ L ′ independent instances of single layer, single head transformers with input length M and embedding dimension d H .
Proof Sketch. Consider first when L ′ = 1. A transformer with H ′ attention heads naturally partitions (by definition) the embedding dimension O( dH ′ H ) into H ′ parts of size O( d H ), and each head separately computes an attention mechanism on one of those parts. The result follows almost directly, with some care to details about MLPs and aggregation.
More care is needed when L ′ > 1. We partition the Θ( dH ′ L ′ H ) coordinates of the embedding dimension into L ′ parts of size Θ( dH ′ H ) each, and the key idea is that each layer will operate on one of those parts while leaving the rest unchanged. Indeed, weights for the query and keys can be selected so that only the relevant part of the coordinates will impact the attention matrices at each layer, then weights for the values can be selected so that the other parts are passed through the layer without being changed.
Finally, we combine Lemma 3.1, Corollary 3.2 and Lemma 3.3 to obtain our main result. Theorem 3.4. For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d, there exists an algorithm that simulates T with O(( N M ) 2 • HL H ′ L ′ ) calls to a transformer oracle with L ′ layers, H ′ attention heads in each layer, input length M , embedding dimension O( dH ′ L ′ H ). We additionally show that these results still hold when both the large and small transformers have causal masking. The proofs are similar to the proof of Theorem 3.4, and are deferred to Appendix B. Theorem 3.5. For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d and causal masking, there exists an algorithm that simulates T with O(( N M ) 2 • HL H ′ L ′ ) calls to a transformer oracle with L ′ layers, H ′ attention heads in each layer, input length M , embedding dimension O( dH ′ L ′ H ) and causal masking.
4 Efficient simulation with average-case input assumptions 4.1 Linear calls suffice for average-case inputs
In this section we prove that if the queries, keys and values in the attention heads are somewhat bounded in how much they may differ from each other, then O( N M ) small transformers suffice to approximate a large transformer. We provide a proof sketch below, and the complete proof can be found in Appendix C.1. Theorem 4.1. Let T be a transformer with L layers, H attention heads in each layer, input length N and embedding dimension d. Suppose there exist absolute constants C, D > 0 such that 1 C ≤ a i,j ≤ C, and DN • max
j ∥b i,j ∥ 2 ≤ N j=1 b i,j2
where a i,j = exp(⟨q i , k j ⟩), b i,j = exp(⟨q i , k j ⟩) • v j for any query q i , k j , v j in any attention head. There exists an algorithm using O( N M • HL H ′ L ′ ) oracle calls to a small transformer with L ′ layers, H ′ attention heads in each layer, embedding dimension O( dH ′ L ′ H ) to obtain an (1 + ε) approximation of T with probability at least 0.9 for any fixed constant ε > 0.
Proof Sketch. The high-level idea is to partition the queries into N/M parts of size M each, and then permute the keys (but not the queries) using a random permutation τ . We then aim to approximate the desired quantities N j=1 exp(⟨q i , k j ⟩) and N j=1 exp(⟨q i , k j ⟩) • v j using rescalings of j∈St exp(⟨q i , k τ (j) ⟩) and j∈St exp(⟨q i , k τ (j) ⟩) • v τ (j) (where S t is the part of size M that contains query q i ). This can be seen as estimating the desired sums by sampling only M of the N summands at random. At the same time, by blocking the queries and keys like this, the samples can be computed using oracle calls similar to Lemma 3.1 above.
this section cite: []

Section: Linear small transformers are weaker than large transformers
We also show that N/M small transformers can be simulated by a large transformer along with an oracle for performing very small matrix multiplications (M × d with d × d). We only prove the statement for single head transformers to illustrate the need for linear amount of oracle calls.
Theorem 4.2. Given N/M instances of single layer, single head transformers with input length M and embedding dimension d, there exists an algorithm that simulates them with one call of a single layer, single head transformer with input length O(N ) and embedding dimension O(d), along with O(N/M ) many matrix multiplications of size M × d × d.
The key idea behind Theorem 4.2 is to concatenate the tokens from all N/M input sequences into a single long sequence of length N , but then slightly increase the embedding dimension in a way which makes tokens from different short sequences highly uncorrelated with each other. Thus, the large attention will not give much weight to pairs of tokens from different short sequences. The complete proof is delayed to Appendix C.2.
this section cite: []

Section: Simulation of transformers with sliding window and StreamingLLMs
In our last section, we show that small transformers work well when we add sliding window masking to attention heads, and when in the StreamingLLM framework. Our constructions are similar to above, but with additional techniques to take advantage of the masking structures, and can be found in the Appendix D.
When we consider sliding window attention, we only need to deal with keys that are close to each query. It is intuitive to see that in such scenario, the attention scores of consecutive M -r queries can be covered with a M × M block matrix, which can be computed using our oracle. Transformers with attention sink is similar to transformers with sliding window, except that we have an extra "sink window" in the beginning for all the queries. These sinks windows can be computed using O( N M ) oracle calls. Theorem 5.1. For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d, constant-size sliding window, there exists an algorithm that simulates T with O( N M • HL H ′ L ′ ) calls to a transformer oracle with L ′ layers, H ′ attention heads in each layer, input length M and embedding dimension O( dH ′ L ′ H ) with causal masking. This result still holds if we have constant-size attention sink.
this section cite: []

Section: Acknowledgments
We thank Daniel Hsu, Jingwen Liu and Vatsal Sharan for useful discussions.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Abstract highlights this paper's main results and motivations.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: Even though the motivation comes from practice, this paper is purely theoretical. It mentions that the goal is to provide a theoretical framework for possible future empirical followups.
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
Answer: [Yes]
Justification: All the model assumptions are clearly stated in introduction section.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [NA] Justification: This is a purely theoretical paper with no experiment.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: This is a purely theoretical paper with no experiment. Guidelines:
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
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: This is a purely theoretical paper with no experiment. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: This is a purely theoretical paper with no experiment. Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [NA] Justification: This is a purely theoretical paper with no experiment. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: This paper does not have harmful content and only has theoretical analysis and proofs.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: The authors are unaware of any possible societal impacts.
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
Answer: [NA]
Justification: This is a purely theoretical paper with no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: This paper mentions related work to the best knowledge of the authors.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: This is a purely theoretical paper with no new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: This is a purely theoretical paper with no use of human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This is a purely theoretical paper with no crowdsourcing or human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method of this paper does not use LLMs. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: A Transformers as Basic Functions
We show that a single layer, single head transformer (attention mechanism with two MLPs) can act as a few different basic functions on the input matrix X that we will need later.
First we show that we can turn any matrix X into a matrix that is consisted of a block matrix of ones and zeros everywhere else.
Lemma A.1. There exists a fixed matrix U ∈ R (d+1)×d and input MLP ϕ : R d → R d+1 such that for any input X ∈ R N ×d ,
ϕ(X)U = 1 a×b 0 0 0 for any a ≤ N, b ≤ d. Proof. Let x i = X[i, :] for all i ∈ [N ]. Define ϕ(x i ) = (x i , 1) if i ≤ a and ϕ(x i ) = (x i , 0) if i ≥ a + 1. Let U = 0 d×b 0 d×(d-b) 1 1×b 0 d×(d-b)
It is straightforward to check that ϕ(X)U is the matrix desired.
Using this, we can construct a transformer that computes the sum of all input tokens.
Lemma A.2. There exists a single layer, single head transformer with embedding dimension d such that, on input matrix X ∈ R N ×d , it computes
N i=1 X[i, :] ∈ R 1×d . Proof. By Lemma A.1, there exists W Q , W K such that (XW Q )[i, :] = (XW K )[i, :] = 1 1×d and k i = N • X[i, :] for all 1 ≤ i ≤ N .
As a result, the output for any token will exactly be
N i=1 X[i, :].
A single layer, single head transformer also allows us to construct a look-up table such that each token can find information from other tokens. Lemma A.3 (Lemma D.1 of [SHT24]). Given input matrix X ∈ R N ×d , an indexing function τ : R d × [N ] → [N ] and f : R m → R m , there exists a single layer, single head transformer with embedding dimension d such that the i-th output is ρ(X[τ (X[i, :], i), :]).
this section cite: ['b30']

Section: B Missing Proofs in Section 3
Lemma B.1 (Lemma 3.1). For any single layer, single head transformer T with input length N , embedding dimension d, there exists an algorithm that simulates T with O( N 2 M 2 ) calls to a transformer oracle with input length M and embedding dimension O(d).
Proof. Let T be any single layer, single head transformer with query, key, value matrices W Q , W K , W V ∈ R d×d with arbitrary input X ∈ R N ×d . Let B be an upper bound of the absolute value of all entries in X, W Q , W K , W V . Without loss of generality we can assume the first MLP ϕ is the identity function because otherwise we will compose it with the input MLP in the oracles. In addition, we can also assume that the layer MLP is the identity function, and this is because if not, we can first set the layer MLP in our oracle to be the identity function to compute the output of the large transformer before the layer MLP, and then use O( N M ) oracles as layer MLP to compute the final output.
Define (as usual) Q = XW Q , K = XW K , V = XW V and q i = Q[i, :], k i = K[i, :], v i = V [i, :] for all 1 ≤ i ≤ N , and let S t = {(t -1)M, . . . , tM } for all 1 ≤ t ≤ N M . Our goal is to simulate softmax(q i K ⊤ )V = N j=1 exp(⟨q i , k j ⟩) • v j N j=1 exp(⟨q i , k j ⟩) =: N j=1 b i,j N j=1 a i,j = N/M t=1 B i,t N/M t=1 A i,t(1)
for all 1 ≤ i ≤ N , where we define
a i,j := exp(⟨q i , k j ⟩), b i,j := exp(⟨q i , k j ⟩) • v j .
and A i,t := j∈St a i,j and B i,t = j∈St b i,j . Our algorithm can be summarized as the following two steps:
this section cite: []

Section: (Step 1) We calculate A i,t
for all 1 ≤ i ≤ N, 1 ≤ t ≤ N/M using ( N M ) 2 oracle calls.
this section cite: []

Section: (Step 2) For each t, we exactly compute
Bi,t Ai,t for all i ∈ [N ] using ( N M ) 2 oracle calls. Since we already know A i,t for all i, t, we can now compute
N/M t=1 B i,t N/M t=1 A i,t = N/M t=1 Bi,t Ai,t • A i,t N/M t=1 A i,t
for all i. Notice that this can be done either trivially or with O(( N M ) 2 ) oracle calls with Lemma A.2 and the fact that we allow MLPs to do division.
Step 1: Calculating A i,t for all i, t. We first consider the case when i ∈ S t . For any 1 ≤ t ≤ N M , define
W Q ′ = W Q 0 0 1 , W K ′ = W K 0 0 1 ∈ R (d+1)×(d+1)
and MLP ϕ such that
Q ′ := ϕ(X[S t , :]) • W Q ′ = X[S t , :] 1 M ×1 0 1×d 1 • W Q 0 d×1 0 1×d 1 =     q (t-1)M +1 1 . . . . . . q tM 1 0 1     , K ′ = ϕ(X[S t , :]) • W K ′ = X[S t , :] 1 M ×1 0 1×d 1 • W K 0 d×1 0 1×d 1 =     k (t-1)M +1 1 . . . . . . k tM 1 0 1     , V ′ = 1 M ×d 0 M ×1 0 1×d 0 . (We can construct W V ′
and add an extra dimension using MLP such that we obtain V ′ using Lemma A.1, but we omit it for simplicity. In particular, we will only need to use the first column of V ′ , so for what follows below, V ′ could be (1, . . . , 1, 0) ⊤ ∈ R (M +1)×1 . We keep our notation consistent and let it have d + 1 columns.) Therefore, we have
Q ′ (K ′ ) ⊤ =     ⟨q (t-1)M +1 , k (t-1)M +1 ⟩ + 1 . . . ⟨q (t-1)M +1 , k tM ⟩ + 1 1 . . . . . . . . . . . . ⟨q tM , k (t-1)M +1 ⟩ + 1 . . . ⟨q tM , k tM ⟩ + 1 1 1 . . . 1 1     .
Finally, we can calculate that the entries of softmax(Q ′ (K ′ ) ⊤ )V ′ are given by j∈St exp(⟨q i , k j ⟩ + 1)
j∈St exp(⟨q i , k j ⟩ + 1) + exp(1) = j∈St a i,j j∈St a i,j + exp(0) = A i,t A i,t + exp(0) for all i ∈ S t . Therefore, we can use the MLP layer to calculate A i,t as:
A i,t = exp(0) • Ai,t Ai,t+exp(0) 1 - Ai,t Ai,t+exp(0)
. Now we compute A i,t when i ∈ S t ′ for some t ′ ̸ = t. The high-level idea is similar, but now we feed the oracle with [X[S t , :], X[S t ′ , :]] ∈ R M ×2d and we let the MLP ϕ be such that
ϕ([X[S t , :], X[S t ′ , :]]) = X[S t , :] X[S t ′ , :] 1 0 1×d 0 1×d 1 ∈ R (M +1)×(2d+1)
and we furthermore define
W Q ′′ =   0 d×d 0 d×1 W Q 0 d×1 0 1×d 1 1×d   , W K ′′ =   W K 0 d×1 0 d×d 0 d×1 0 1×d 1 1×d   ∈ R (2d+1)×(d+1)
such that d+1) .
Q ′′ := ϕ([X[S t , :], X[S t ′ , :]]) • W Q ′′ =     q (t ′ -1)M +1 1 . . . . . . q t ′ M 1 0 1×d 1     ∈ R (M +1)×(d+1) K ′′ := ϕ([X[S t , :], X[S t ′ , :]]) • W K ′′ =     k (t-1)M +1 1 . . . . . . k tM 1 0 1×d 1     ∈ R (M +1)×(d+1) V ′′ := 1 M ×d 0 M ×1 0 1×d 0 ∈ R (M +1)×(
Therefore, we have
Q ′′ (K ′′ ) ⊤ =     ⟨q (t ′ -1)M +1 , k (t-1)M +1 ⟩ + 1 . . . ⟨q (t ′ -1)M +1 , k tM ⟩ + 1 1 . . . . . . . . . . . . ⟨q t ′ M , k (t-1)M +1 ⟩ + 1 . . . ⟨q t ′ M , k tM ⟩ + 1 1 1 . . . 1 1     .
We can now compute A i,t for all i ∈ S t ′ for the exact same reason as above. Step 1 requires ( N M ) 2 oracle calls.
this section cite: []

Section: Step 2: Computing
Bi,t Ai,t for all i, t. If i ∈ S t , we can compute Bi,t Ai,t using one oracle call simply by feeding the oracle with X[S t , :], W Q , W K , W V . It remains to compute Bi,t Ai,t for all i ∈ S t ′ where t ′ ̸ = t. We feed the oracle with [X[S t , :], X[S t ′ , :]] and let
W Q ′′′ = 0 d×d W Q , W K ′′′ = W K 0 d×d , W V ′′′ = W V 0 d×d ∈ R 2d×d
such that
Q ′′′ := [X[S t , :], X[S t ′ , :]] • W Q ′′′ =    q (t ′ -1)M +1 . . . q t ′ M    , K ′′′ := [X[S t , :], X[S t ′ , :]] • W K ′′′ =    k (t-1)M +1 . . . k tM    , V ′′′ := [X[S t , :], X[S t ′ , :]] • W V ′′′ =    v (t-1)M +1 . . . v tM    .
A simple calculation shows that softmax(Q ′′′ (K ′′′ ) ⊤ )V ′′′ gives us Bi,t Ai,t for i ∈ S t ′ . In total we need N M oracle calls in step 2.
this section cite: []

Section: Corollary B.2 (Corollary 3.2).
For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d, there exists an algorithm that simulates T with O(( N M ) 2 • HL) calls to a single head, single layer transformer oracle with input length M and embedding dimension O( d H ).
Proof. We simply compute T layer by layer and head by head. For each layer, we compute the output of each attention head, which requires O(( N M ) 2 • H) oracle calls using Lemma B.1, concatenate and repeat for each layer.
this section cite: []

Section: Lemma B.3 (Lemma 3.3).
One transformer with L ′ layers, H ′ attention heads in each layer, input length M and embedding dimension O( dH ′ L ′ H ) can be used to simulate H ′ L ′ independent instances of single layer, single head transformers with input length M and embedding dimension d H .
Proof. Given H ′ L ′ independent instances, we label the instances by (h, ℓ) ∈ H ′ ×L ′ and concatenate them together by columns to X ∈ R M × dH ′ L ′ H such that
X = [X 1,1 , X 1,2 , . . . , X 1,L ′ , X 2,1 , . . . , X H ′ ,L ′ ],
where X h,ℓ is the input of the (h, ℓ)-th instance for h ∈ [H ′ ], ℓ ∈ [L ′ ]. As a result, in the first layer,
[X h,1 , . . . , X h,L ′ ] ∈ R M × dL ′ H is sent to attention head h. Let W Q , W K , W V ∈ R d H × d
H denote the query, key and value matrices in the (h, 1)-th instance. We construct
W Q ′ , W K ′ , W V ′ as W Q ′ =      W Q 0 d H × d H . . . 0 d H × d H      , W K ′ =      W K 0 d H × d H . . . 0 d H × d H      , W V ′ =      W V 0 d H × d H . . . 0 d H × d H      ∈ R (dL ′ /H)×d/H
and layer MLP the same as the layer MLP in instance (h, 1) such that attention head h in layer 1 exactly computes the (h, 1)-th instance. Also observe that we are not performing any computation regarding (h, ℓ)-th instance for any ℓ ̸ = 1. The same argument holds for all h, and therefore layer one of the large transformer computes (h, 1)-th instance for all 1 ≤ h ≤ H ′ . Since the outputs of all H ′ attention heads are stored at predefined locations after each layer, we can repeat this process for L ′ times to compute all instances.
Theorem B.4 (Theorem 3.4). For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d, there exists an algorithm that simulates T with O(( N M ) 2 • HL H ′ L ′ ) calls to a transformer oracle with L ′ layers, H ′ attention heads in each layer, input length M , embedding dimension O( dH ′ L ′ H ).
Proof. This follows from Corollary B.2 and Lemma B.3.
this section cite: []

Section: Theorem B.5 (Theorem 3.5).
For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d and causal masking, there exists an algorithm that simulates T with O(( N M ) 2 • HL H ′ L ′ ) calls to a transformer oracle with L ′ layers, H ′ attention heads in each layer, input length M , embedding dimension O( dH ′ L ′ H ) and causal masking.
The high-level idea is identical to Theorem B.4. We first show that a single layer, single head transformer with input length M and causal masking can compute i j=1 exp(⟨q i , k j ⟩) for all 1 ≤ i ≤ M given input and all parameters. Claim B.6. Given any X ∈ R M ×d , W Q , W K , W V ∈ R d×d , one calls to a single layer, single head transformer oracle O with input length M , embedding dimension O(d) and causal masking suffices to compute i j=1 exp(⟨q i , k j ⟩) for all 1 ≤ i ≤ M . Proof. We define
W Q ′ = W Q 0 1×d 0 d×1 1 , W K ′ = W K 0 1×d 0 d×1 1 ∈ R (d+1)×(d+1)
such that
Q ′ := ϕ(X) • W Q ′ = 0 1×d 1 X 1 M ×1 • W Q 0 1×d 0 d×1 1 =     0 1 q 1 1 . . . . . . q M 1     ∈ R (M +1)×(d+1) K ′ = ϕ(X) • W K ′ = 0 1×d 1 X 1 M ×1 • W K 0 1×d 0 d×1 1 =     0 1 k 1 1 . . . . . . k M 1     ∈ R (M +1)×(d+1) V ′ = 0 0 1×d 0 M ×1 1 M ×d ∈ R (M +1)×(d+1) .
As a result, we have
Q ′ (K ′ ) ⊤ =     1 1 . . . 1 1 ⟨q 1 , k 1 ⟩ . . . ⟨q 1 , k M ⟩ . . . . . . . . . . . . 1 ⟨q M , k 1 ⟩ . . . ⟨q M , k M ⟩     .
Finally, we can calculate that the (i, j)-th entry (2 ≤ i ≤ M + 1) of the oracle output, softmax(mask(Q
′ (K ′ ) ⊤ ))V ′ , is i-1 j=1 exp(⟨q i-1 , k j ⟩ + 1) i-1 j=1 exp(⟨q i-1 , k j ⟩ + 1) + exp(1) = i-1 j=1 a i-1,j i j=1 a i-1,j + exp(0) .
Finally, we can define the MLP such that it outputs
exp(0) • i-1 j=1 ai-1,j i j=1 ai-1,j +exp(0) 1 - i-1 j=1 ai-1,j i j=1 ai-1,j +exp(0) = i-1 j=1 a i-1,j
for all 2 ≤ i ≤ M + 1.
Proof of Theorem B.5. The proof is similar to the proof of Theorem B.4. First notice that Lemma B.3 still holds if we add causal masking to the transformers because the proof is not affected by causal masking. In addition, the analog of Corollary B.2 will hold even if we add causal masking to the transformers if we can show that Lemma B.1 holds under causal masking since the proof is the same. Therefore, it suffices to prove Lemma B.1 under causal masking.
Let T be any single layer, single head transformer with query, key, value matrices W Q , W K , W V ∈ R d×d and causal masking, with arbitrary input X ∈ R N ×d . For the same reason as Lemma B.1, we assume without loss of generality that both the input MLP and layer MLP are identity functions, and we use the same notation as in Lemma B.1. Our goal is to approximate
i j=1 exp(⟨q i , k j ⟩) • v j i j=1 exp(⟨q i , k j ⟩) =: i j=1 b i,j i j=1 a i,j .
for all 1 ≤ i ≤ N . Notice that we already include causal masking in this expression by only summing over j ≤ i.
Define A (1) i,t := j∈St∩[i] a i,j , A (2) i,t := j∈St-[i] a i,j ⇒ A i,t = A (1) i,t + A (2) i,t and B (1) i,t := j∈St∩[i] b i,j , B (1) i,t + B
(2) i,t .
For each i ∈ S t , we want to compute
t-1 t ′ =1 B i,t ′ + B (1) i,t t-1 t ′ =1 A i,t ′ + A (1) i,t = t-1 t ′ =1 ( B (1) i,t ′ A (1) i,t ′ • A (1) i,t ′ + B (2) i,t ′ A (2) i,t ′ • A (2) i,t ′ ) + B (1) i,t A (1) i,t • A (1) i,t t-1 t ′ =1 (A (1) i,t + A (2) i,t ) + A (1) i,t . Just like Lemma B.1, we will compute A (1) i,t ′ , A (2) i,t ′ , B (1) i,t ′ A (1) i,t ′ , B (2) i,t ′ A (2) i,t ′ with our oracle independently for all 1 ≤ t ′ ≤ t -1 (each with one oracle call). In addition, A (1) i,t can be computed with a single oracle call using Claim B.6, and B (1) i,t A (1) i,t can be computed using a single oracle call by simply feeding the oracle with X[S t , :], W Q , W K , W V . To compute A (1) i,t ′ and B (1) i,t ′ A (1) i,t ′ with one oracle each for all i, t ′ , we can use the exact same construction in step 1 in the proof of Lemma B.1 (this works because our oracle also has causal masking). To compute A (2) i,t ′ and B (2) i,t ′ A (2) i,t ′
, we first define ϕ such that
ϕ(X[S t , :], X[S t ′ , :]) =       x tM 0 1×d x tM -1
x t ′ M . . . . . .
x (t-1)M +1 x (t ′ -1)M +2 0 1×d x (t ′ -1)M +1       .
Notice that this is valid because ϕ has information on the position of all the tokens. Furthermore, Lemma A.3 allows us to use our oracle to perform this computation as well. We also let
W Q ′ = W Q 0 d×d , W K ′ = 0 d×d W K such that Q ′ := ϕ(X[S t , :], X[S t ′ , :]) • W Q ′ =       q tM q tM -1 . . . q (t-1)M +1 0 1×d       , K ′ := ϕ(X[S t , :], X[S t ′ , :]) • W K ′ =       0 1×d k t ′ M k t ′ M -1 . . . k (t ′ -1)M +1       , V ′ := 0 1×d 0 1 M ×d 0 M ×1 . Now notice that Q ′ (K ′ ) ⊤ =       0 ⟨q tM , k t ′ M ⟩ • • • ⟨q tM , k (t ′ -1)M +1 ⟩ 0 ⟨q tM -1 , k t ′ M ⟩ • • • ⟨q tM -1 , k (t ′ -1)M +1 ⟩ . . . . . . . . . . . . 0 ⟨q (t-1)M +1 , k t ′ M ⟩ • • • ⟨q (t-1)M +1 , k (t ′ -1)M +1 ⟩ 0 0 • • • 0       Therefore, the (tM -i + 1)-th row of softmax(mask(Q ′ (K ′ ) ⊤ ))V ′ , which corresponds to q i , is computing t ′ M j=i+1 exp(⟨q i , k j ⟩) t ′ M j=i+1 exp(⟨q i , k j ⟩) + exp(0)
, which allows us to compute
t ′ M j=i+1 exp(⟨q i , k j ⟩) = A(2)
i,t ′ exactly. Furthermore,
this section cite: []

Section: B
(2) i,t ′ A
(2) i,t ′ can be computed by letting
W Q ′′ = W Q 0 d×d , W K ′′ = 0 d×d W K , W V ′′ = 0 d×d W V such that Q ′′ := ϕ(X[S t , :], X[S t ′ , :]) • W Q ′′ =     q tM q tM -1 . . . q (t-1)M +1     , K ′′ := ϕ(X[S t , :], X[S t ′ , :]) • W K ′′ =     k t ′ M k t ′ M -1 . . . k (t ′ -1)M +1     , V ′′ :=     v t ′ M v t ′ M -1 . . . v (t ′ -1)M +1     .
In total we need O(( N M ) 2 ) oracle calls, and the proof is complete.
this section cite: []

Section: C Missing Proofs in Section 4
C.1 Missing Proofs in Section 4.1 Theorem C.1 (Theorem 4.1). Let T be a transformer with L layers, H attention heads in each layer, input length N and embedding dimension d. Suppose there exist absolute constants C, D > 0 such that 1 C ≤ a i,j ≤ C, and DN • max
j ∥b i,j ∥ 2 ≤ N j=1 b i,j2
where a i,j = exp(⟨q i , k j ⟩), b i,j = exp(⟨q i , k j ⟩) • v j for any query q i , k j , v j in any attention head. There exists an algorithm using O( N M • HL H ′ L ′ ) oracle calls to a small transformer with L ′ layers, H ′ attention heads in each layer, embedding dimension O( dH ′ L ′ H ) to obtain an (1 + ε) approximation of T with probability at least 0.9 for any fixed constant ε > 0.
Proof. By Corollary B.2 and Lemma B.3, it suffices to prove the Theorem with H = L = 1 because when we generalize, all the attention head computation will be in parallel with each other.
Let T be any attention head with query, key, value matrices W Q , W K , W V ∈ R d×m with arbitrary input X ∈ R N ×d . Without loss of generality we can assume the first MLP ϕ is the identity function because otherwise we will compose our oracle MLP with ϕ. In addition, we can also assume that the layer MLP is the identity function, and this is because if not, we can still use identity functions in our oracle layer MLPs to compute the output of the large transformer before the layer MLP, and then use O( N M ) oracles as MLP to compute the final output.
Define Q = XW Q , K = XW K , V = XW V such that q i = Q[i, :], k i = K[i, :], v i = V [i, :] for all 1 ≤ i ≤ N , and let S t = {(t -1)M, . . . , tM } for all 1 ≤ t ≤ N M . Our goal is to approximate softmax(q i K ⊤ )V = N j=1 exp(⟨q i , k j ⟩) • v j N j=1 exp(⟨q i , k j ⟩) = N j=1 b i,j N j=1 a i,j(2)
for all 1 ≤ i ≤ N . The algorithm will be divided into two parts like before.
this section cite: []

Section: Step 1: Approximating
N j=1 a i,j for all 1 ≤ i ≤ N . Let i ∈ S t for some 1 ≤ t ≤ N M . We pick a random permutation τ of [N ], and by Lemma A.3 we can use O(N/M ) oracle calls 1 (one for each X[S t , :]) to map x i to [x i , x τ (i) ] for all 1 ≤ i ≤ N . Now we use the first MLP ϕ in the oracles to map (X[S t , :], X[τ (S t ), :]) to ϕ(X[S t , :], X[τ (S t ), :]) = X[S t , :] X[τ (S t ), :] 1 0 1×d 0 1×d 1 and we furthermore define
W Q ′ =   W Q 0 d×1 0 d×d 0 d×1 0 1×d 1   , W K ′ =   0 d×d 0 d×1 W K 0 d×1 0 1×d 1  
in the oracle such that
Q ′ := ϕ(X[S t , :], X[τ (S t ), :])W Q ′ =     q (t-1)M +1 1 . . . . . . q tM 1 0 1     ∈ R (M +1)×(d+1) , K ′ := ϕ(X[S t , :], X[τ (S t ), :])W K ′ =     k τ ((t-1)M +1) 1 . . . . . . k τ (tM ) 1 0 1     ∈ R (M +1)×(d+1) , V ′ := 1 M ×d 0 M ×1 1 d×1 1 .
Now we can compute j∈St a i,τ (j) for all i using N M oracle calls (the remaining proof is exactly the same as the proof of Lemma 3.1). Our estimator of N j=1 a i,j will be N M j∈St a i,τ (j) .
Our estimator is unbiased because
E j∈St a i,τ (j) = M N • N j=1 a i,j .
We can use Hoeffding's Inequality to show that
Pr j∈St a i,τ (j) • N M - N j=1 a i,j ≥ ε 4 N j=1 a i,j = Pr j∈St a i,τ (j) - M N N j=1 a i,j ≥ εM 4N N j=1 a i,j ≤ 2 exp - 2ε 2 M 2 16N 2 • ( N j=1 a i,j ) 2 M (C -1 C ) 2 ≤ 2 exp - 2ε 2 M (N/C) 2 16N 2 C 2 = 2 exp - 2ε 2 M 16C 4 , which is at most 1 20N if M ≥ 8C 4 log(40N ) ε 2
, which is true by our assumption on M . A union bound over all i ∈ [N ] allows us to show that we get a (1 + ε/4) approximation of N j=1 a i,j for all i with probability at least 0.95.
this section cite: []

Section: Step 2: Approximating
N j=1 bi,j N j=1 ai,j for all 1 ≤ i ≤ N . Let i ∈ S t for some 1 ≤ t ≤ N M . We pick a random permutation τ of [N ], and define ϕ([X[S t , :], X[τ (S t ), :]]) = [X[S t , :], X[τ (S t ), :]] and furthermore
W Q ′′ = W Q 0 d×d , W K ′′ = 0 d×d W K , W V ′′ = 0 d×d W V .
The output softmax(Q ′′ (K ′′ ) ⊤ )V ′′ gives us exactly j∈St b i,τ (j) j∈Sj a i,τ (j) , which will be our estimator. First observe that
E j∈St b i,τ (j) = M N • N j=1 b i,j .
We can again use Hoeffding's Inquality to show that
Pr N M j∈St b i,τ (j) - N j=1 b i,j 2 ≥ ε 4 N j=1 b i,j 2 ≤ 2 • exp - ε 2 M • ∥ N j=1 b i,j ∥ 2 2 128N 2 (max j ∥b i,j ∥ 2 ) 2 ≤ 2 • exp - ε 2 M D 2 128 which is at most 1 20N when M ≥ 128(log d+log 40+log N ) ε 2 D 2
. We have shown that
N (1 + ε/4)M ≤ N j=1 a i,j j∈St a i,τ (j) ≤ N (1 -ε/4)M
, and now we prove
N j=1 b i,j -j∈St b i,τ (j) • N j=1 a i,j j∈St a i,τ (j) 2 ≤ ε • N j=1 b i,j 2 , which concludes the proof. Indeed, we first decompose N j=1 b i,j -j∈St b i,τ (j) • N j=1 a i,j j∈St a i,τ (j) 2 = N j=1 b
i,j - j∈St b i,τ (j) • N M + (1 -δ) j∈St b i,τ (j) • N M 2 ≤ N j=1 b i,j - j∈St b i,τ (j) • N M 2 + (1 -δ) j∈St b i,τ (j) • N M 2 ,(3)
where
1 1 + ε/4 ≤ δ := M • N j=1 a i,j N • j∈St a i,τ (j) ≤ 1 1 -ε/4
.
this section cite: []

Section: Now we already know
N j=1 b i,j - j∈St b i,τ (j) • N M 2 ≤ ε 4 • N j=1 b i,j 2 ,
and therefore Equation 3 can be further bounded by
ε 4 • N j=1 b i,j 2 + (1 -δ)(1 + ε/4) N j=1 b i,j 2 ≤ ε 4 • N j=1 b i,j 2 + ε 4 + ε (1 + ε/4) N j=1 b i,j 2 ≤ ε • N j=1 b i,j 2 ,
which concludes the proof.
this section cite: []

Section: C.2 Missing Proofs in Section 4.2
Theorem C.2 (Theorem 4.2). Given N M instances of single layer, single head transformers with input length M and embedding dimension d, there exists an algorithm that simulates them with one call of a single layer, single head transformer with input length O(N ) and embedding dimension O(d), along with O( N M ) many matrix multiplications of size M × d × d.
Proof. We assume that the input/layer MLPs in all the instances are identity functions for the same reason as in Lemma B.1. Let X i ∈ R M ×d be the input, W Q i , W K i , W V i ∈ R d×d be the query, key and value matrices for each instance 1 ≤ i ≤ N M . Our goal is to calculate softmax((
X i W Q i )(X i W K i ) ⊤ )(X i W V i ) for all 1 ≤ i ≤ N M . We first compute X i W Q i , X i W K i , X i W V i for all 1 ≤ i ≤ N M .
For convenience we will denote
q i,j = (X i W Q i )[j, :], k i,j = (X i W K i )[j, :], v i,j = (X i W V i )[j, :] such that our goal is to compute softmax(q i,j (X i W K i ) ⊤ )(X i W V i ) for all 1 ≤ i ≤ N M , 1 ≤ j ≤ M . For convenience we let ∥X i ∥ ∞,2 • ∥W Q i ∥ 2 , ∥X i ∥ ∞,2 • ∥W K i ∥ 2 , ∥X i ∥ ∞,2 • ∥W V i ∥ 2 ≤ C
≤ poly(N ), for some C (because we assume that all entries have O(log N ) bit representation, and therefore each parameter is at most poly(N )). As a result, -C 2 ≤ ⟨q i,j , k i ′ ,j ′ ⟩ ≤ C 2 , ∥v i,j ∥ 2 ≤ C for any 1 ≤ i, i ′ ≤ N/M, 1 ≤ j, j ′ ≤ M . Let B ∈ R to be determined. Define u 1 , . . . , u N M ∈ {0, -B} r such that r r/2 ≥ N/M (we only need r ≤ O(log(N/M )) by Stirling approximation) and all u i have exactly r/2 zeros (if there are more vectors than needed, simply make sure they are distinct), and let v i = B • (1, . . . , 1) + u i for all 1 ≤ i ≤ N M such that ⟨u i , v i ⟩ = 0 for all i. For any i ̸ = j, notice that there must exist an index at which u i is -B and v j is B. This is because the set of nonzeros in v i is the compliment of the set of nonzeros in u i , and the former set must be different from the set of nonzeros in v j . Now append u i to q i,j to get q ′ i,j and v i to k i,j to get k ′ i,j for all 1 ≤ j ≤ M . Set d ′ = d + r. For each pair of q ′ i,j
and k ′ i ′ ,j ′ :
• If i = i ′ , i.e. q i and k i ′ are in the same small transformer instance, then
⟨q ′ i,j , k ′ i ′ ,j ′ ⟩ = ⟨q i,j , k i ′ ,j ′ ⟩ + ⟨u i , v i ′ ⟩ = ⟨q i,j , k i ′ ,j ′ ⟩
• If i ̸ = i ′ , i.e. q i and k i ′ are in different small transformer instances, then
⟨q ′ i,j , k ′ i ′ ,j ′ ⟩ = ⟨q i,j , k i ′ ,j ′ ⟩ + ⟨u i , v i ′ ⟩ ≤ ⟨q i,j , k i ′ ,j ′ ⟩ -B 2 .
Now we let Q ∈ R N ×d ′ be the matrix of q ′ i,j and K ∈ R N ×d ′ be the matrix of k ′ i,j and V = X i W V i . We set X = [Q, K, V ] ∈ R N ×(3d ′ ) (note that we can always pad zeros to V to make dimensions match),
W Q = I d ′ ×d ′ 0 d ′ ×d ′ 0 d ′ ×d ′ , W K = 0 d ′ ×d ′ I d ′ ×d ′ 0 d ′ ×d ′ , W V = 0 d ′ ×d ′ 0 d ′ ×d ′ I d ′ ×d ′ such that [Q, K, V ]W Q = Q, [Q, K, V ]W K = K, [Q, K, V ]W V = V.
this section cite: []

Section: 
Furthermore, for each query (q i,j , u i ), its inner product between all keys in the same small transformer will be preserved, while its inner product between all keys in a different small transformer will be at most C 2 -B 2 . Therefore, for each query q i,j , we can calculate the error as: M j ′ =1 exp(⟨q i,j , k i,j ′ ⟩) M ℓ=1 exp(⟨q i,j , k i,ℓ ⟩)
• v i,j ′ - N/M i ′ =1 M j ′ =1 exp(⟨q ′ i,j , k ′ i ′ ,j ′ ⟩) N/M ℓ=1 M ℓ ′ =1 exp(⟨q ′ i,j , k ′ ℓ,ℓ ′ ⟩) v i ′ ,j ′ 2 ≤ M j ′ =1 exp(⟨q i,j , k i,j ′ ⟩) M ℓ=1 exp(⟨q i,j , k i,ℓ ⟩) • v i,j ′ - M j ′ =1 exp(⟨q ′ i,j , k ′ i,j ′ ⟩) N/M ℓ=1 M ℓ ′ =1 exp(⟨q ′ i,j , k ′ ℓ,ℓ ′ ⟩) • v i,j ′ 2 + N C exp(C 2 -B 2 ) M ℓ=1 exp(⟨q i,j , k i,ℓ ⟩) ≤ M j ′ =1 exp(⟨q i,j , k i,j ′ ⟩) M ℓ=1 exp(⟨q i,j , k i,ℓ ⟩) • v i,j ′ - exp(⟨q i,j , k i,j ′ ⟩) N/M ℓ=1 M ℓ ′ =1 exp(⟨q ′ i,j , k ′ ℓ,ℓ ′ ⟩) • v i,j ′ 2 + N C exp(C 2 -B 2 ) M exp(-C 2 ) . Now notice that N/M ℓ=1 M ℓ ′ =1 exp(⟨q ′ i,j , k ′ ℓ,ℓ ′ ⟩) = M ℓ=1 exp(⟨q i,j , k i,ℓ ⟩) + N/M ℓ=1,ℓ̸ =i M ℓ ′ =1 exp(⟨q ′ i,j , k ′ ℓ,ℓ ′ ⟩)
and that
M exp(-C 2 ) ≤ M ℓ=1 exp(⟨q i,j , k i,ℓ ⟩) ≤ M exp(C 2 ) N exp(-C 2 -B 2 ) ≤ N/M ℓ=1,ℓ̸ =i M ℓ ′ =1 exp(⟨q ′ i,j , k ′ ℓ,ℓ ′ ⟩) ≤ N exp(C 2 -B 2 ).
As a result,
1 M ℓ=1 exp(⟨q i,j , k i,ℓ ⟩) - 1 N/M ℓ=1 M ℓ ′ =1 exp(⟨q ′ i,j , k ′ ℓ,ℓ ′ ⟩) ≤ N exp(C 2 -B 2 ) M 2 exp(-C 2 ) = N exp(2C 2 -B 2 ) M 2 .
Therefore, we can further upper bound the error by
M • exp(C 2 ) • C • N exp(2C 2 -B 2 ) M 2 + N C exp(C 2 -B 2 ) M exp(-C 2 ) = N C exp(2C 2 -B 2 )(1 + exp(C 2 )) M .
Therefore, we can set B ≤ poly(N ) to be sufficiently large such that the error is O( 1 2 N ).
this section cite: []

Section: D Missing Proofs in Section 5
Theorem D.1 (Theorem 5.1). For any transformer T with L layers, H attention heads in each layer, input length N , embedding dimension d, constant-size sliding window, there exists an algorithm that simulates T with O( N M • HL H ′ L ′ ) calls to a transformer oracle with L ′ layers, H ′ attention heads in each layer, input length M and embedding dimension O( dH ′ L ′ H ) with causal masking. This result still holds if we have constant-size attention sink.
Proof. The proof goes in the same way as Theorem B.5, where we assume without loss of generality that H = L = H ′ = L ′ = 1. We start with the sliding window scenario where the size of sliding window is r. Notice that for each query q i (i ≥ r + 1) we want to compute
i j=i-r+1 exp(⟨q i , k j ⟩) = i j=1 exp(⟨q i , k j ⟩) - i-r j=1 exp(⟨q i , k j ⟩), and i j=i-r+1 exp(⟨q i , k j ⟩) • v j = i j=1 exp(⟨q i , k j ⟩) • v j - i-r j=1 exp(⟨q i , k j ⟩) • v j
given window size r. In addition, we can compute r j=1 exp(⟨q i , k j ⟩) and r j=1 exp(⟨q i , k j ⟩) • v j for all i ≤ r with 2 oracle calls, as shown in the proof of Theorem B.5. We will partition {r + 1, . . . , N } into chunks S ′ 1 = {r + 1, . . . , M }, S ′ 2 = {M + 1, . . . , 2M -r}, . . . of size M -r and approximate the terms above for all i ∈ S ′ t in each chunk using constant many oracle calls, which suffices since r is a constant. Without loss of generality we only prove our claim for S ′ 1 as the proof can be easily generalized to the remaining S ′ t . Indeed, observe that i j=1 exp(⟨q i , k j ⟩) and
i-r j=1 exp(⟨q i , k j ⟩)
for all i ∈ S ′ 1 can both be computed exactly with one oracle call using the proof of Claim B.6. Furthermore, one oracle call suffices to compute either
i j=1 exp(⟨q i , k j ⟩) • v j i j=1 exp(⟨q i , k j ⟩) or i-r j=1 exp(⟨q i , k j ⟩) • v j i-r j=1 exp(⟨q i , k j ⟩)
for all i ∈ S ′ 1 because we can feed the oracle with X[S 1 , :] and use W Q to project X[S 1 , :] to {q r+1 , . . . , q M } and use W K to project X[S 1 , :] to {k 1 , . . . , k M -r } (similar to proof of Lemma B.1).
Finally, when there are attention sinks, we simply need to further calculate s j=1 exp(⟨q i , k j ⟩) and
s j=1 exp(⟨q i , k j ⟩) • v j . Observe that s j=1 exp(⟨q i , k j ⟩) = i j=1 exp(⟨q i , k j ⟩) - i j=s+1 exp(⟨q i , k j ⟩), s j=1 exp(⟨q i , k j ⟩) • v j = i j=1 exp(⟨q i , k j ⟩) • v j - i j=s+1 exp(⟨q i , k j ⟩) • v j .
Each of the four terms can be computed with the exact same technique as in sliding window with 1 oracle call.
this section cite: []

Section: References
Ref_id:b0 Title: Zoology: Measuring and improving recall in efficient language models Year: (2024)
Ref_id:b1 Title: Fast attention requires bounded entries Year: (2024)
Ref_id:b2 Title: Fundamental limitations on subquadratic alternatives to transformers Year: (2025)
Ref_id:b3 Title: On the Ability and Limitations of Transformers to Recognize Formal Languages Year: (2020-11)
Ref_id:b4 Title: Separations in the representational capabilities of transformers and recurrent architectures Year: (2024)
Ref_id:b5 Title: The long-document transformer Year: (2020)
Ref_id:b6 Title: An exploration of hierarchical attention transformers for efficient long document classification Year: (2022)
Ref_id:b7 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b8 Title: Scaling language models: Methods, analysis & insights from training gopher Year: (2021)
Ref_id:b9 Title: Flashattention: fast and memory-efficient exact attention with io-awareness Year: (2022)
Ref_id:b10 Title: Etched is Making the Biggest Bet in AI Year: (2024-06)
Ref_id:b11 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b12 Title: Theoretical limitations of self-attention in neural sequence models Year: (2020)
Ref_id:b13 Title: Hyperattention: Long-context attention in near-linear time Year: (2024)
Ref_id:b14 Title: Computational limits of low-rank adaptation (loRA) fine-tuning for transformer models Year: (2025)
Ref_id:b15 Title: Multilayer feedforward networks are universal approximators Year: (1989)
Ref_id:b16 Title: On statistical rates and provably efficient criteria of latent diffusion transformers (dits) Year: (2024)
Ref_id:b17 Title: Repeat after me: transformers are better than state space models at copying Year: (2024)
Ref_id:b18 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b19 Title: RNGD preview: The world's most efficient AI chip for LLM inference Year: (2024-10-18)
Ref_id:b20 Title: Reformer: The efficient transformer Year: (2020)
Ref_id:b21 Title: Polysketchformer: fast transformers via sketching polynomial kernels Year: (2024)
Ref_id:b22 Title: Transformers learn shortcuts to automata Year: (2023)
Ref_id:b23 Title: On the expressive flexibility of self-attention matrices Year: (2023-06)
Ref_id:b24 Title: Hierarchical transformers for multi-document summarization Year: (2019-07)
Ref_id:b25 Title: The parallelism tradeoff: Limitations of logprecision transformers Year: (2023)
Ref_id:b26 Title: Saturated transformers are constant-depth threshold circuits Year: (2022)
Ref_id:b27 Title: Language models are few-shot learners Year: (2020)
Ref_id:b28 Title: Jesús Villalba, Yishay Carmiel, and Najim Dehak. Hierarchical transformers for long document classification Year: ()
Ref_id:b29 Title: Representational strengths and limitations of transformers Year: (2023)
Ref_id:b30 Title: Transformers, parallel computation, and logarithmic depth Year: (2024)
Ref_id:b31 Title: What formal languages can transformers express? a survey Year: (2024)
Ref_id:b32 Title: Efficient transformers: A survey Year: (2022-12)
Ref_id:b33 Title: State spaces aren't enough: Machine translation needs attention Year: (2023)
Ref_id:b34 Title: Attention is all you need Year: (2017)
Ref_id:b35 Title: RNNs are not transformers (yet): The key bottleneck on in-context retrieval Year: (2025)
Ref_id:b36 Title: Lightseq2: Accelerated training for transformer-based models on gpus Year: (2021)
Ref_id:b37 Title: Efficient streaming language models with attention sinks Year: (2024)
Ref_id:b38 Title: Are transformers universal approximators of sequence-to-sequence functions Year: (2020)
Ref_id:b39 Title: O(n) connections are expressive enough: Universal approximability of sparse transformers Year: (2020)
