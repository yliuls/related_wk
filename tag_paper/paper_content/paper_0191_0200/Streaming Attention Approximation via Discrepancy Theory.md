Title: Streaming Attention Approximation via Discrepancy Theory
Abstract: Large language models (LLMs) have achieved impressive success, but their high memory requirements present challenges for long-context token generation. In this paper we study the streaming complexity of attention approximation, a key computational primitive underlying token generation. Our main contribution is BalanceKV, a streaming algorithm for ϵ-approximating attention computations based on geometric process for selecting a balanced collection of Key and Value tokens as per Banaszczyk's vector balancing theory. We complement our algorithm with space lower bounds for streaming attention computation. Besides strong theoretical guarantees, BalanceKV exhibits empirically validated performance improvements over existing methods, both for attention approximation and end-to-end performance on various long context benchmarks.

Section: Introduction
Transformer-based models are the foundation of ongoing artificial intelligence revolution. Their applications span a wide range of domains, from leading-edge language models (LLM) [1,65] to text-to-image [58,66,69], text-to-video synthesis [70], coding assistance [68] and even in multimodal domains across text, audio, image, and video [53]. At the core of these models is the Transformer architecture, powered by the self-attention mechanism [73], which enables effective capture of pairwise correlations across tokens in an input sequence. As these models scale in size and context length [41], they face significant computational challenges, particularly in terms of memory usage. Efficiency and accuracy are essential to unlock the full potential of LLMs in generating long sequences.
Space bottlenecks in transformer models. Most large language models, along with multimodal and video models, adopt an autoregressive, decoder-only architecture. This architecture generates tokens sequentially, applying attention dynamically to each newly generated token. To avoid redundant attention score computations during the generation phase, these models explicitly store the key and value embeddings of previously generated tokens in a cache in each attention layer. Thus, a major challenge is the fact that the memory complexity of storing previously generated key value embeddings scales with both the model size (i.e., the number of layers and attention heads) and, critically, the context size. Additionally, each model session typically requires its own dedicated cache for storing key value embeddings, further exacerbating memory usage. This growing demand has become a significant bottleneck, affecting both memory consumption and computational speed, particularly for models handling long context lengths.
this section cite: ['b0', 'b63', 'b56', 'b64', 'b66', 'b67', 'b51', 'b70', 'b40']

Section: Streaming attention computation.
The main reason for the need of storing the past key and value embeddings is for the attention computation happening inside each self attention layer during token generation after processing a context -to generate the next token, each self attention layer computes the attention between the query embedding of the current token and the key and value embeddings of all the tokens that were previously generated or part of the context. In this paper we study the streaming attention approximation problem -the problem of approximately computing attention using a small amount of space, i.e. without storing all previously seen key and value embeddings. Our main contribution is BALANCEKV, a novel provably correct algorithm for streaming attention approximation based on discrepancy theory. The core of our approach is a vector balancing algorithm from discrepancy theory that exploits the geometry of key and value tokens to deduce a small subset of them that well approximates the operations happening inside a self-attention layer. We complement our algorithm with a lower bound on the streaming complexity of approximating attention.
An algorithm for streaming attention approximation can directly be used for compressing the key value cache which stores the past key value embeddings in each layer in an LLM, thus improving the efficiency of LLM token generation. We empirically evaluate BALANCEKV both on the problem of approximating attention and on end-to-end generation tasks, showing performance gains.
this section cite: []

Section: Related Work
For discrepancy theory, Banaszczyk's seminal works [6,7] establishing theoretical guarantees for vector set discrepancy have sparked research in the vector balancing problem [17]. This led to algorithmic developments in both offline [8] and online [9,3,43] settings. The vector balancing problem has particular relevance to streaming and sublinear algorithms, as minimizing a dataset's discrepancy yields small subsets that effectively preserve the original dataset's properties. Recently [55,15] extend these discrepancy theory ideas for kernel density estimation using sublinear memory.
A simple yet effective approach is quantizing previously generated key value embeddings with fewer bits [80,78,26,40,49,35,84,81]. Another line of work focuses on token-level pruning, where redundant or less important tokens get evicted from the set of all previously generated key value embeddings [10,85,48,76,83,46]. Many of the works in this line have used accumulated attention scores to select important previously generated tokens [85,46,76]. Recent works extend those methods to an adaptive way of budget allocation across layer [14] and head [30].
this section cite: ['b5', 'b6', 'b16', 'b7', 'b8', 'b2', 'b42', 'b53', 'b14', 'b77', 'b75', 'b25', 'b39', 'b48', 'b34', 'b81', 'b78', 'b9', 'b82', 'b47', 'b73', 'b80', 'b45', 'b82', 'b45', 'b73', 'b13', 'b29']

Section: Overview of Our Contributions
In this work we take the token subset selection approach to reduce the memory complexity of LLM token generation: store and maintain only a subset of previously generated key and value embeddings corresponding to a few "important" tokens in the sequence. Of course, the central question is how to define "importance" of tokens. Our approach here is to apply discrepancy theory, which, at a high level, considers a token important if it is crucial to preserving the projection of the total collection of tokens onto some direction in the token space. This leads to the idea of selecting a subset of tokens that is "balanced" simultaneously in every direction. Inspired by the recent breakthrough result of [3] on online discrepancy minimization, we design a method for balancing key-value pairs online using small space, namely our BALANCEKV algorithm. Interestingly, this algorithm is online, i.e. the importance of a token is determined only by preceding tokens -in sharp contrast with state of the art heuristics for token selection such as PyramidKV [14] and SnapKV [46], whose performance, as we show, our algorithm matches or improves upon. Our contributions are:
1. In Section 3 we propose BALANCEKV, an algorithm for recursively compressing the set of previously generated tokens using a geometric correlated sampling process based on discrepancy theory. We show that BALANCEKV gives provable guarantees for streaming attention approximation under the bounded ℓ 2 norm assumption (Theorem 3.1). Using tools from communication complexity, we also show a lower bound on the memory complexity of any algorithm for streaming attention approximation in Section 3. Section 2 contains the formal problem formulation of streaming attention approximation, its applicability to key value cache compression, as well as a technical overview of the main results and techniques of Section 3. 2. In Section 4 we empirically evaluate our algorithm in various settings. In Section 4.1 we show our approach leads to a lower relative error for single layer attention approximation for open-source LLMs including Llama-3.1-8B-Instruct [27] and Ministral-8B-Instruct-2410 [52] as compared to uniformly sampling keys and values in the cache. Section 4.1 we also perform ablation studies to show how various parameters in our algorithm affect the relative error for single layer attention approximation. In Sections 4.2 and 4.3 we perform end to end experiments on various benchmarks such as LongBench [5] using models of various sizes such as Llama-3.1-8B-Instruct,Qwen-2.5-14B-Instruct and Qwen-2.5-32B-Instruct [77,71], and Needle in a Haystack [39]. We show that our provable method for attention approximation when applied to key value cache compression performs better compared to previous existing token subset selection heuristics on end to end tasks. Finally in Section 4.4 we present system efficiency metrics regarding our implementation.
this section cite: ['b2', 'b13', 'b45', 'b26', 'b4', 'b74', 'b68', 'b38']

Section: Technical Overview
In this section, we first set up the formal problem formulation that we tackle, followed by an overview of our techniques and our main results.
this section cite: []

Section: Streaming Attention Approximation: Formulation and Motivation
Autoregressive Transformers generate tokens one by one and each depends on the previously generated tokens. When Transformers process a sequence of tokens, the attention mechanism operates by computing three types of embeddings for each token at every layer: query, key and value. The query and key capture how different tokens interact, while the value is the actual content to be aggregated. Such interactions are quantified by so-called attention scores, obtained by applying the softmax to the inner product between the query of a given token and the keys of all others. These scores determine how much each previous token's value contributes to the final output. Once the keys and values are computed for a given token, they do not need to be recomputed when generating subsequent tokens.
Formally, suppose that we have a stream of query, key and value embeddings (q 1 , k 1 , v 1 ), . . . , (q n , k n , v n ), that is the j-th token is represented as a triplet of (q j , k j , v j ) where q j , k j , v j ∈ R d for all j ∈ [n]. Let K j , V j ∈ R j×d be matrices defined by stacking those keys and values in their respective rows. To compute the following at every step j to generate j + 1 token, is called the streaming attention problem:
Attn(q j , K j , V j ) := softmax K j • q j √ d T • V j .(1)
Keeping all of the key-value pairs in the cache is prohibitively expensive, especially for long sequences. Instead, we opt for approximate computation by sampling a few key-value pairs. Specifically, our goal is to construct an algorithm that at every time step j computes an estimator z j for Attn(q j , K j , V j ) in sublinear in n time and memory. In particular for given precision ε > 0, z j should satisfy the following error constraint:
∥z j -Attn(q j , K j , V j )∥ 2 ≤ ε softmax K j • q j √ d 2 ∥V j ∥ F .(2)
A sublinear in n time and memory algorithm to compute z j will require knowledge of significantly less key-value pairs than K j , V j , thus reducing the size of the key value cache needed to store them. This motivates the study of streaming attention approximation, as an algorithm for this can directly be used for key value cache compression during LLM token generation. In the next section we discuss how we will construct such an estimator z j at a high level.
this section cite: []

Section: SOFTMAXBALANCE: Attention Approximation via Discrepancy Theory
We now start with presenting the main ideas of our approach. By the definition of softmax, Equation (1) can be written as
Attn(q j , K j , V j ) = 1 Z j exp K j • q j √ d T • V j ,
where for a matrix A we write exp(A) to denote entry-wise exponential function to A and Z j := i∈[j] exp(⟨k i , q j ⟩/ √ d). Our approach to approximate Attn(q j , K j , V j ) consists of two subroutines which approximate:
1. Softmax normalization Z j = i∈[j] exp(⟨k i , q j ⟩/ √ d), 2. Matrix-vector product between V j and exp(K j • q j / √ d).
To understand our main idea, suppose we are at the end of the stream (i.e., j = n) and we store all key-value pairs (k 1 , v 1 ), . . . , (k n , v n ). Then for an arbitrary query q n we aim to approximate the matrix-vector product exp(
K n • q n / √ d) T • V n = i∈[n] exp(⟨k i , q n ⟩/ √ d)v i
by choosing a subset of the rows of K n and V n of size at most n/2 which corresponds to a compression rate of 0.5. Suppose we can design an algorithm which splits the set C of all keys and values into two groups C ′ and C\C ′ so that the matrix-vector product function for any query vector q n is roughly equal over C ′ and C\C ′ that is informally,
{k,v}∈C ′ exp ⟨k, q n ⟩ √ d v ≈ {k,v}∈C\C ′ exp ⟨k, q n ⟩ √ d v.
Then, we are able to approximate the matrix-vector product function with either one of the sums above since informally:
{k,v}∈C exp ⟨k, q n ⟩ √ d v ≈ 2 {k,v}∈C ′ exp ⟨k, q n ⟩ √ d v.
Therefore, it would suffice to keep the smaller subset of C ′ and C\C ′ as the desired subset of key value embeddings and discard the rest. If we wanted to compress the key value cache to a smaller size by a factor 2 T for some T , we would recursively compress the selected subset using the same procedure T -1 more times.
A similar goal is captured by the vector balancing problem studied extensively in discrepancy theory; given a set of vectors C = {k 1 , . . . , k n } ⊂ R d with ∥k j ∥ 2 ≤ 1 for all j, partition them into two groups C ′ , C \ C ′ such that for any q ∈ R d it holds k∈C ′ ⟨k, q⟩ ≈ k∈C\C ′ ⟨k, q⟩ with high probability. The Self-Balancing Walk algorithm [3] is a breakthrough result for the above vector balancing problem. However we need to develop an algorithm for the vector balancing problem with respect to function exp(⟨k, •⟩/ √ d)v instead of the inner product function ⟨k, •⟩.
Our first contribution is to develop an algorithm for our task, building upon the result from the self-balancing walk [3], which essentially randomly partitions the set of keys and values C into C ′ and C \ C ′ such that the following holds with high probability under the assumptions that the norms of the query and key embeddings are bounded,
{k,v}∈C ′ exp ⟨k, q n ⟩ √ d v - {k,v} / ∈C ′ exp ⟨k, q n ⟩ √ d v 2 ≤ O (log(nd)) • max j∈[n] ∥v i ∥ 2 .
We refer to this algorithm as SOFTMAXBALANCE, its formal guarantee is presented in Theorem 3.3 and its pseudocode is presented in Algorithm 2. Theorem 3.3 shows that SOFTMAXBAL-ANCE succeeds to divide C into subsets C ′ and C\C ′ which are balanced with respect to function exp(⟨k, •⟩/ √ d)v up to an error which only has logarithmic dependence on the size of C. In addition, SOFTMAXBALANCE can accept as input value vectors of arbitrary dimension s. Therefore, if instead of the value vectors v 1 , . . . , v n ∈ R d we input the set of scalars v 1 = • • • = v n = 1, we will get an algorithm for the vector balancing problem with respect to function exp(⟨k, •⟩/ √ d). This implies that we can use SOFTMAXBALANCE to compress the key value cache to even approximate the softmax normalization i∈[n] exp(⟨k i , q n ⟩/ √ d). We now discuss how to use SOFTMAXBALANCE for streaming attention approximation, i.e. to use it to compute an estimator z j satisfying Equation (2).
this section cite: ['b2', 'b2']

Section: BALANCEKV: Implementing SOFTMAXBALANCE in Streaming
For a sequence of n tokens and a given memory budget of t ≪ n, we aim to design a procedure which applies SOFTMAXBALANCE to select from n key-value embeddings a set of at most t in the streaming setting and can compute an estimator z j satisfying Equation (2) for all steps j in the stream. In the streaming setting one needs to consider the following aspects. As described in the previous section, one iteration of SOFTMAXBALANCE only allows one to select a n/2 sized subset of n key-value embeddings, which is higher than the desired budget of t embeddings. This can be easily mitigated by recursively applying SOFTMAXBALANCE 2 log(n/t) times, each time halving the set of key-value embeddings. However, this cannot be implemented in the streaming as we have a limited memory budget of t which prohibits us from storing all key-value embeddings during recursion.
To deal with this, we use the classical merge and reduce technique used in the design of streaming algorithms [13,51,33]. MERGEANDREDUCE algorithm is a recursive binary tree-based approach that allows one to implement SOFTMAXBALANCE recursively in a streaming setting with the total memory not exceeding O(dt), where O(•) supresses polynomial in log n factors, under the assumption that the norms of queries and keys are bounded. The guarantees of MERGEANDREDUCE are presented in Theorem 3.4, its pseudocode in Algorithm 4 and a visual representation in Figure 2. If the norms of all value embeddings in the stream are the same up to constant factors, that is for all i, j ∈ [n] 0.5 ≤ ∥v i ∥ 2 /∥v j ∥ 2 ≤ 2, then the outputs of MERGEANDREDUCE can be used to construct an estimator z j satisfying our attention approximation guarantee of equation Equation (2) with precision ε for t = O( √ d/ε). However, the value embeddings may have very different norms.
Our main algorithm BALANCEKV (pseudocode in Algorithm 1) deals with this issue by grouping the key-value embeddings in the stream according to the norms of the value embeddings, running a separate instance of MERGEANDREDUCE on each group, and combining the outputs of each instance of MERGEANDREDUCE. BALANCEKV constructs a final estimator z j satisfying Equation ( 2) with precision ε only using O(d √ d/ε) memory and O(d 2 /ε 2 ) runtime per every step j of the stream, assuming the norms of query and key embeddings are bounded. Existing methods [83] subsample keys and values independently in the cache, and thus have a 1/ε 2 dependence on ε in total memory. The guarantees of BALANCEKV are presented in Theorem 3.1.
Finally using the lower bound on the communication complexity of INDEX, we show a lower bound on the memory complexity of any algorithm for streaming attention approximation in Theorem 3.2.
this section cite: ['b12', 'b50', 'b32', 'b1', 'b80']

Section: Main Theoretical Results
Our main algorithm for streaming attention approximation is BALANCEKV. It takes in as input a stream of n tokens (q 1 , k 1 , v 1 ), (q 2 , k 2 , v 2 ), . . . , (q n , k n , v n ) and at every step of the stream outputs an estimate z j to Attn(q j , K j , V j ) (see Equation (1) for the definition of Attn(.)) satisfying Equation (2) with precision ε. Assuming that the ℓ 2 norms of q j , k j are at most r for all j, BALANCEKV uses total space O(d
√ de 2r 2 / √ d • 1/ε) and uses O(d 2 e 4r 2 / √ d • 1/ε 2
) runtime at each step j of the stream to output z j . Our main theorem is as follows.
Theorem 3.1. For any r, ε > 0, any positive integers n, d, any set of tokens (q 1 , k 1 , v 1 ), (q 2 , k 2 , v 2 ), . . . , (q n , k n , v n ) where q j , k j , v j ∈ R d satisfy ∥q j ∥ 2 , ∥k j ∥ 2 ≤ r for all j, consider an invocation of BALANCEKV with batch size t = O √ de 2r 2 / √ d /ε and compression rate 2 -T with T = log(n/t).
Then BALANCEKV outputs a vector z j satisfying Equation A pseudocode of BALANCEKV is described in Algorithm 1. At its core BALANCEKV relies on our main discrepancy based algorithm, namely SOFTMAXBALANCE-see Section 3.1 for details on SOFTMAXBALANCE. BALANCEKV uses the output of SOFTMAXBALANCE to compute estimates of the numerator and denominator of Attn(q j , K j , V j ) and returns the desired attention approximation z j for each streamed index j. There are two subtleties, however. First, it is important to bucket tokens in the stream according to the norm of the value vectors -see lines 5 and 6. Second, a direct application of SOFTMAXBALANCE would require too much memory space. To ensure small space usage, we apply a classical streaming technique, namely the MERGEANDREDUCE algorithm on top of SOFTMAXBALANCE to reduce the space consumption. The space reduction achieved by MERGEANDREDUCE is by running a logarithmic number of copies of SOFTMAXBALANCE in a tree-like fashion. More details are introduced in Section 3.2.
Algorithm 1 BALANCEKV((q j , k j , v j ) n j=1 , r, t, T, ε)
1: input: stream of n tokens (q j , k j , v j ), diameter r, batch size t, compression rate 2 -T , precision parameter ε.
2: // Bucket the stream and maintain log(n) instances of MERGEANDREDUCE, MR-NUMERATORi, for each bucket to approximate the numerator of Attn(qj, Kj, Vj); and one instance, MR-DENOMINATOR, to approximate its denominator.
3: v max ← 0 4: repeat 5:
Find an index i such that 2 i ≥ ∥v j ∥ 2 ≥ 2 i-1 6:
Send (k j , v j ) as input to MR-NUMERATOR i // Bucket the stream by ∥v∥2 7:
v max ← max {∥v j ∥ 2 , v max } 8: Erase all MR-NUMERATOR i with 2 i ≤ ε 2n e -r 2 √ d v max
// Erase small norm buckets 9:
C 0 i , . . . , C T i ← the output of MR-NUMERATOR i 10: V l ← ∪ i C l i for l = 0, . . . , T // Combine the outputs of MR-Numeratori 11:
Send (k j , 1) as input to MR-DENOMINATOR 12:
K 0 , . . . K T ← MR-DENOMINATOR 13: output: z j = T l=0 2 l {k,v}∈V l exp ⟨k,q j ⟩ √ d v T l=0 2 l {k,v}∈K l exp ⟨k,q j ⟩ √ d 14:
j ← j + 1 15: until token stream ends To summarize, BALANCEKV groups tokens in the stream according to the norms of the corresponding value embeddings, runs a separate instance of MERGEANDREDUCE on each group, and combines the outputs of each instance to construct the final estimate for Attn(q j , K j , V j ) at each step j ∈ [n]. Next we present SOFTMAXBALANCE and MERGEANDREDUCE. The full proof of Theorem 3.1 is given in appendix Section A.1. Finally we state the theorem which provides a lower bound on the memory complexity of any algorithm for streaming attention approximation below, its full proof is provided in appendix Section C. Theorem 3.2. Suppose that r 2 ≤ d.
Any streaming algorithm which on input ({k 1 , v 1 }, . . . , {k n , v n }, q), ∥q∥ 2 , ∥k i ∥ 2 ≤ r, outputs z q satisfying Equation (2) with probability 0.999 has space complexity Ω min{ 1 ε 2 , d exp(2r 2 / √ d)} .
this section cite: ['b0']

Section: SOFTMAXBALANCE
We now present our main discrepancy based compression algorithm, SOFTMAXBALANCE. Given a sequence of key and value embeddings C = {(k 1 , v 1 ), . . . (k n , v n )} (with key and value embeddings having possibly different dimensions), the goal of SOFTMAXBALANCE is to produce a partition of C into subsets C ′ , C \ C ′ such that for any query q ∈ R d we have that
(k,v)∈C ′ exp(⟨k, q⟩/ √ d)v ≈ (k,v)∈C\C ′ exp(⟨k, q⟩/ √ d)v
with high probability. Without loss of generality assume that |C ′ | ≤ |C|/2, we can then output 2 (k,v)∈C ′ exp(⟨k, q⟩/ √ d)v as an approximation to (k,v)∈C exp(⟨k, q⟩/ √ d)v, thus achieving a factor 2 compression. Its description is presented in Algorithm 2 below. We note that while SOFTMAXBALANCE takes as input a sequence of key and value embeddings, it can nevertheless be used to compute the softmax normalization: we simply run it on the keys, with the corresponding value vector one-dimensional and all equal to 1 -see line 11 in BALANCEKV, where SOFTMAXBALANCE is called within the corresponding invocation of MERGEANDREDUCE with value vectors as 1s. It's guarantees are as follows.
Theorem 3.3. Given sets K = {k 1 , . . . , k n } ⊂ R d , V = {v 1 , . . . , v n } ⊂ R s , and failure probability δ > 0, define C to be the dataset of pairs C = {(k 1 , v 1 ), . . . , (k n , v n )}. There exists a randomized algorithm, SOFTMAXBALANCE, which outputs a subset C ′ ⊂ C, |C ′ | ≤ |C|/2, such that, for any Algorithm 2 SOFTMAXBALANCE((k j , v j ) j , r key , r value , δ) 1: input: stream of ≤ n key-value embeddings (k j , v j ), radii r key , r value : max j ∥k j ∥ 2 ≤ r key , max j ∥v j ∥ 2 ≤ r value , probability of failure δ. 2: R ← exp(r 2 key /2 √ d) • r value 3: c ← 30 log(n/δ) 4: Initialize zero vector η ← {0} 5: for j from 1 and until the end of the stream do 6:
y ← exp(⟨k i , k j ⟩/ √ d)⟨v i , v j ⟩ i∈[j]7:
if y T η > c • R 2 then FAIL 8:
p j ← 1 2 -y T η 2c•R 2 9:
η j ← +1 with probability p j -1 o.w.
10:
Add a new zero coordinate η j+1 ← 0 11: end for 12: if |{(k i , v i ) :
η i = 1}| ≤ |{(k i , v i ) : η i = -1}| then 13: output: {(k i , v i ) : η i = 1} 14: else 15: output: {(k i , v i ) : η i = -1} 16: end if vector q ∈ R d , with probability at least 1 -δ, {k,v}∈C ′ exp ⟨k, q⟩ √ d v - {k,v} / ∈C ′ exp ⟨k, q⟩ √ d v 2 ≤ O √ s • log(ns/δ) • exp ∥q∥ 2 2 2 √ d • exp max j∈[n] ∥k j ∥ 2 2 2 √ d • max j∈[n] ∥v j ∥ 2 .
The runtime of SOFTMAXBALANCE is O((d + s)n 2 ) and memory is O((d + s)n).
The proof of the above theorem uses the breakthrough result of [3] for the vector balancing problem, one of the main problems in discrepancy theory. Given a set of vectors k 1 , . . . , k n the result of [3] produces a subset C of these vectors of at most half the size such that for any vector q we have that k∈C ⟨k, q⟩ ≈ k∈[n]\C ⟨k, q⟩ with high probability. Our main contribution is an algorithm for the vector balancing problem with respect to the function exp(⟨k, •⟩/ √ d)v as compared to ⟨k, •⟩ in the case of [3]. We defer the proof of Theorem 3.3 to Appendix A.2.
this section cite: ['b2', 'b2', 'b2']

Section: MERGEANDREDUCE
As briefly mentioned above in Section 3, MERGEANDREDUCE is a streaming version of SOFT-MAXBALANCE. The idea is to partition the stream of tokens into batches of size t, apply SOFT-MAXBALANCE to the batches to reduce the size of each batch by a constant factor, and then repeat recursively -see Fig. 2 in the appendix.
If we set batch size t to be about 1/ε (see Theorem 3.4 below for the more precise setting), we obtain a streaming algorithm that approximates j i=1 exp(⟨k i , q j ⟩/ √ d)v i at any point j in the stream using total space O(d
√ de 2r 2 / √ d /ε) and runtime O(d 2 e 4r 2 / √ d /ε 2 ) per step,
where r is an upper bound on the norms of key and query embeddings.
As before, an important aspect is that MERGEANDREDUCE can handle value embeddings of dimension not necessarily equal to that of key and query embeddings. Thus, when run on scalars v i = 1 for all i, it can also be used to approximate softmax normalization at any point j in the stream. This is the main subroutine used in BALANCEKV to approximate Attn(q j , K j , V j ). Its pseudocode description is presented in Appendix A.
3.1, and its proof is in Appendix A.3.2. The formal guarantees are 0.1 0.2 0.3 0.4 0.5 Compression Rate 0.1 0.2 0.3 0.4 0.5 Relative Error Relative Error vs Compression Rate for Llama Layer 1 -BalanceKV Layer 1 -Uniform Layer 2 -BalanceKV Layer 2 -Uniform Layer 5 -BalanceKV Layer 5 -Uniform 0.1 0.2 0.3 0.4 0.5 Compression Rate 0.1 0.2 0.3 0.4 0.5 0.6 Relative Error Relative Error vs Compression Rate for Mistral Layer 1 -BalanceKV Layer 1 -Uniform Layer 2 -BalanceKV Layer 2 -Uniform Layer 5 -BalanceKV Layer 5 -Uniform Theorem 3.4. For any r, ε > 0, any set of tokens (q
1 , k 1 , v 1 ), . . . , (q n , k n , v n ) where q j , k j ∈ R d satisfy ∥q j ∥ 2 , ∥k j ∥ 2 ≤ r, v j ∈ R s for s ≤ d suppose, batch size t = O( √ se 2r 2 / √ d /ε)
and compression rate 2 -T with T = log(n/t).
Then MERGEANDREDUCE on input parameters t, r, d, s, ε, outputs at every step j of the stream subsets of key-value embedding pairs C 0 , . . . , C T ⊂ C := {(k
1 , v 1 ), . . . , (k n , v n )} such that, z j := T i=0 2 i {k,v}∈C i exp ⟨k,qj ⟩ √ d v, satisfies with probability at least 1 -1/poly(n), j i=1 exp ⟨k i , q j ⟩ √ d v i -z j 2 ≤ εj • e -r 2 / √ d • max i∈[n] ∥v i ∥ 2 .
Total memory of the algorithm is O(d
√ se 2r 2 / √ d /ε), its j-th iteration runtime is O(dse 4r 2 / √ d /ε 2 ).
this section cite: []

Section: Experiments
In this section we now present our experimental results. The full details of all sections as well as the experimental setup and implementation can be found in Appendix B.
this section cite: []

Section: Ablation Studies on Single Layer Attention Approximation
We evaluate the effectiveness of BALANCEKV for approximating attention in individual layers of Llama-3.1-8B-Instruct [27] and Ministral-8B-Instruct-2410 [52] on the TriviaQA dataset from LongBench [5]. Specifically, we examine layers 1, 2, and 5 and compare against independent uniform sampling key and value embeddings.
Due to space limitations, we provide the full experimental details in Appendix B.1. For each layer, we approximate attention for recent tokens using a compressed cache that retains a fixed number of initial and recent embeddings, alongside intermediate ones selected via BALANCEKV, and measure its relative error against exact attention. We vary the compression rate 2 -T ∈ {1/2, 1/4, 1/8, 1/16}.
As shown in Fig. 1, BALANCEKV consistently yields lower relative approximation error than approximating attention by uniform sampling past key value pairs across all settings, empirically validating its advantage as predicted by Theorem 3.1.
For a fixed dataset and layer, we also analyzed how the performance and runtime of BALANCEKV depend on the batch size and compression rate. More precisely, we repeat the single-layer attention approximation experiment TriviaQA and layers 1 and 15 of Llama-3.1-8B-Instruct, for batch size ∈ [64, 128, 256] and compression rate 2 -T ∈ [1/2, 1/4, 1/8]. The results are presented in Figure 3.
As this experiment suggests, the quality of attention approximation increases as the size of the block doubles, while the runtime becomes slower as also proven theoretically.
Method qasper multi hotpotqa 2wiki gov multinews trec triviaqa samsum p.count p.ret lcc repo-p average Qwen2.5-32B-Instruct Exact (Baseline) 44. 56 50.65 69.14 60.39 21.3  19.52  75.33 81.14  43.17  22.0 99.67 50.9 35.22 51.77  StreamingLLM 20.12 34.35 51.84 48.23 1909 17.10 61.00 51.14 28.52 23.33 41.33 39.19 26.54 35.52 PyramidKV 34.47 46.33 67.78 55.92 15.16 15.39 69.33 63.24 40.36 22.67 99.33 48.32 34.35 47.13 SnapKV 36.21 46.78 66.64 57.02 16.35 16.07 70.33 77.53 41.08 22.0 99.33 49.04 35.62 48.77 Uniform 39.28 43.82 64.82 57.84 23.10 19.50 73.00 81.63 39.70 22.00 92.00 44.97 32.19 48.76 BALANCEKV 40.14 43.17 64.46 58.06 22.26 20.32 73.00 80.68 41.07 22.33 92.0 44.95 32.43 48.84 Qwen2.5-14B-Instruct Exact (Baseline) 43.39 52.63 64.06 53.71 28.07 22.4 74.67 88.75 44.81 22.33 99.0 63.69 46.3 54.14 StreamingLLM 20.73 32.62 49.93 42.39 21.63 18.69 59.67 74.96 29.57 11.67 63.0 46.16 32.25 38.71 PyramidKV 31.76 46.6 62.83 50.0 19.08 17.68 65.0 85.52 42.61 22.0 99.33 60.62 44.4 49.8 SnapKV 32.95 47.53 61.96 50.34 20.29 18.28 60.67 88.75 42.97 22.67 99.33 61.32 45.84 50.22 Uniform 37.07 41.69 61.11 49.96 29.18 22.67 71.67 87.89 40.34 22.0 84.33 58.0 42.57 49.88 BALANCEKV 37.02 41.96 61.74 50.9 29.26 22.64 71.67 88.1 41.14 23.67 87.67 58.64 43.63 50.62 Llama-3.1-8B-Instruct Exact (Baseline) 42.87 48.54 52.05 38.6 31.31 22.07 71.67 91.85 42.36 20.37 98.13 49.62 42.73 50.17  StreamingLLM 20.65 30.71 39.14 32.43 23.10  18.70  58.00 83.87  28.85 20.36 97.26 33.69 30.46 39
this section cite: ['b26', 'b4']

Section: End-to-end Evalution on LongBench
Next we evaluate BALANCEKV on LongBench dataset [5], which tests long-context understanding across tasks like QA, summarization, few-shot learning, synthetic reasoning, and code completion. Specifically, we test a version of uniform length distribution (LongBench-E). During inference, we compress the key-value cache in the prefill stage using a uniform compression rate of (approximately) 0.25 across all methods, while retaining all streamed embeddings during the decoding phase. We compare against StreamingLLM [76], SnapKV [46], PyramidKV [14], and uniform sampling (see Section 4.1), using their implementations from MInference [37]. The evaluation follows the Long-Bench protocol, using three pre-trained models at different scales including Llama-3.1-8B-Instruct, Qwen-2.5-14B-Instruct and Qwen-2.5-32B-Instruct. The results are reported in Table 1.
Notably, BALANCEKV consistently achieves the best overall performance among compression methods and across all models, demonstrating its effectiveness in preserving model quality for cache compression. The full experimental setup details can be found in Appendix B.2.
this section cite: ['b4', 'b73', 'b45', 'b13', 'b36']

Section: Needle-In-A-Haystack Benchmark
We evaluate BALANCEKV on the "Needle-In-A-Haystack" benchmark [39], comparing it against SnapKV, PyramidKV, StreamingLLM, and uniform sampling using Llama-3.1-8B-Instruct. The test challenges the model to retrieve a specific sentence (the "needle") embedded at an arbitrary position within a long context (the "haystack"). Following the setup in [29], we hide the needle at varying depths, from 0% to 100% of the total context length, across documents ranging from approximately 4K to 100K tokens. As in the previous experiments, all methods are evaluated under a fixed compression ratio of approximately 0.25.
To further enhance performance, we introduce an augmented version of BALANCEKV that deterministically preserves a small set of tokens whose key embeddings are strongly anti-correlated with the rest. The standard BALANCEKV procedure is then applied to the remaining tokens within each layer. As a result, BALANCEKV achieves an average accuracy of 0.99, outperforming SnapKV (0.83), PyramidKV (0.90), StreamingLLM (0.31), and uniform sampling (0.90). Detailed heatmaps of performance across different context lengths and needle depths are in Figure 4 in Appendix B.3.
this section cite: ['b38', 'b28']

Section: System Efficiency Metrics
We measure wall-clock times for both the prefill stage (including cache compression) and the decoding stage using a random input of length 16,384 tokens, followed by the generation of 1,024 tokens.
Results are averaged over 10 independent runs, with the minimum runtime reported to enhance robustness. The full results are provided in Table 2 in Appendix B.4.
this section cite: []

Section: All compression methods incur some prefill overhead compared to the uncompressed baseline (Exact).
While StreamingLLM achieves the fastest decoding speed, it suffers from significantly lower accuracy (see Table 1). Among the remaining methods, BALANCEKV achieves the lowest prefill latency and consistently delivers the best trade-off between efficiency and accuracy. This demonstrates that our discrepancy-based approach not only scales well in theory but also brings practical gains in end-to-end system performance, making it a compelling choice for real-world deployment scenarios.
this section cite: []

Section: Additional Experiments
We additionally conduct the following experiments and provide their results in Appendix B.5 due to the space limitation.
1. Our main theorem (Theorem 3.4) relies on an upper bound of ℓ 2 norms of both query and key vectors. To validate this, we investigate the ℓ 2 norms of queries, keys, and values (QKV) on the TriviaQA dataset from LongBench [5] using Llama-3.1-8B-Instruct. Specifically, we analyze prompts in TriviaQA and compute the average ℓ 2 norms of all QKV vectors across all layers and attention heads during the prefill stage. The key findings are that all QKV norms consistently concentrate around some constants (15 for query, 15 for key, and 3 for value) with small confidence intervals (CI). Importantly, the norms remain stable across a wide range of sequence lengths, suggesting that these norms do not grow with input sequence length.
2. We perform the evaluation of BALANCEKV and the uniform sampling when applied to the InternVL2.5-8B multimodal LLM for compression rates 1/4 and 1/16, for evaluation on the MS COCO image captioning dataset. The experiment was run on a NVIDIA A100 GPU with 80 GB VRAM.
3. We repeat the experiment in Section 4.2 in the extremely low compression rate regime on some of the datasets from LongBench [5]. More specifically, we compress the key-value cache in the prefill stage of inference using a uniform compression rate of (approximately) 0.8, 0.9, and 0.95 with uniform sampling as well as BALANCEKV, while retaining all streamed embeddings during the decoding phase. BALANCEKV demonstrates improved performance over uniform sampling across each of the compression rates and datasets.
4. We augment Section 4.2 by adding comparison to ClusterGen [83] using Llama-3.1-8B-Instruct. The results are reported in Table 6.
this section cite: ['b4', 'b80']

Section: Conclusion
We propose BALANCEKV, a token pruning method grounded in discrepancy theory. BALANCEKV enables approximate attention computation, which we both establish theoretically and validate empirically. To demonstrate the effectiveness of BALANCEKV as a KV cache compression algorithm, we conduct end-to-end experiments on a range of popular benchmarks and models of varying sizes. Finally, our work introduces a theoretical problem of optimal streaming attention space complexity.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Gqa: Training generalized multi-query transformer models from multi-head checkpoints Year: (2023)
Ref_id:b2 Title: Discrepancy minimization via a selfbalancing walk Year: ()
Ref_id:b3 Title: Quarot: Outlier-free 4-bit inference in rotated llms Year: (2024)
Ref_id:b4 Title: Longbench: A bilingual, multitask benchmark for long context understanding Year: (2023)
Ref_id:b5 Title: Balancing vectors and gaussian measures of n-dimensional convex bodies Year: (1998)
Ref_id:b6 Title: On series of signed vectors and their rearrangements Year: (2012)
Ref_id:b7 Title: Constructive algorithms for discrepancy minimization. 51th Annual IEEE Symposium on Foundations of Computer Science (FOCS '2010) Year: (2010)
Ref_id:b8 Title: Online vector balancing and geometric discrepancy Year: (2019)
Ref_id:b9 Title: The long-document transformer Year: (2020)
Ref_id:b10 Title: Weighted minwise hashing beats linear sketching for inner product estimation Year: (2023)
Ref_id:b11 Title: Concentration inequalities Year: (2003)
Ref_id:b12 Title: Adversarial robustness of streaming algorithms through importance sampling Year: (2021)
Ref_id:b13 Title: Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling Year: (2024)
Ref_id:b14 Title: A quasi-monte carlo data structure for smooth kernel evaluations Year: (2024)
Ref_id:b15 Title: Similarity estimation techniques from rounding algorithms Year: (2002)
Ref_id:b16 Title: Balancing vectors in any norm Year: (2018)
Ref_id:b17 Title: Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models Year: (2024)
Ref_id:b18 Title: Sampling methods for inner product sketching Year: (2024)
Ref_id:b19 Title: Sampling methods for inner product sketching Year: (2024)
Ref_id:b20 Title: Flashattention-2: Faster attention with better parallelism and work partitioning Year: (2023)
Ref_id:b21 Title: An elementary proof of a theorem of johnson and lindenstrauss Year: (2003)
Ref_id:b22 Title: 8-bit matrix multiplication for transformers at scale Year: (2022)
Ref_id:b23 Title: Qlora: Efficient finetuning of quantized llms Year: (2024)
Ref_id:b24 Title: Spqr: A sparse-quantized representation for near-lossless llm weight compression Year: (2023)
Ref_id:b25 Title: Qaq: Quality adaptive quantization for llm kv cache Year: (2024)
Ref_id:b26 Title: The llama 3 herd of models Year: (2024)
Ref_id:b27 Title: Gptq: Accurate post-training quantization for generative pre-trained transformers Year: (2022)
Ref_id:b28 Title: Data engineering for scaling language models to 128k context Year: (2024)
Ref_id:b29 Title: Not all heads matter: A head-level kv cache compression method with integrated retrieval and reasoning Year: (2024)
Ref_id:b30 Title: Rabitq: Quantizing high-dimensional vectors with a theoretical error bound for approximate nearest neighbor search Year: (2024)
Ref_id:b31 Title: A framework for few-shot language model evaluation Year: (2023)
Ref_id:b32 Title: Frequent directions: Simple and deterministic matrix sketching Year: (2016)
Ref_id:b33 Title: Hyperattention: Long-context attention in near-linear time Year: (2023)
Ref_id:b34 Title: Towards 10 million context length llm inference with kv cache quantization Year: (2024)
Ref_id:b35 Title: Super-bit locality-sensitive hashing Year: (2012)
Ref_id:b36 Title: Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention Year: (2024)
Ref_id:b37 Title: Extensions of lipschitz maps into banach spaces Year: (1986)
Ref_id:b38 Title: Needle in a haystack-pressure testing llms. Github Repository Year: (2023)
Ref_id:b39 Title: Gear: An efficient kv cache compression recipefor near-lossless generative inference of llm Year: (2024)
Ref_id:b40 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b41 Title: Extreme kv cache compression via sparse coding over universal dictionaries Year: (2024)
Ref_id:b42 Title: Optimal online discrepancy minimization Year: (2023)
Ref_id:b43 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b44 Title: How long can open-source llms truly promise on context length? Year: (2023)
Ref_id:b45 Title: Llm knows what you are looking for before generation Year: (2024)
Ref_id:b46 Title: Activation-aware weight quantization for llm compression and acceleration Year: (2023)
Ref_id:b47 Title: Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time Year: (2024)
Ref_id:b48 Title: A tuning-free asymmetric 2bit quantization for kv cache Year: (2024)
Ref_id:b49 Title: Binary iterative hard thresholding converges with optimal number of measurements for 1-bit compressed sensing Year: (2024-10)
Ref_id:b50 Title: Finding repeated elements Year: (1982)
Ref_id:b51 Title: Introducing gpt-4o Year: (2024)
Ref_id:b52 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b53 Title: Near-optimal coresets for kernel density estimates Year: (2020)
Ref_id:b54 Title: High-dimensional estimation with geometric constraints. Information and Inference: A Year: (2017)
Ref_id:b55 Title: Efficiently scaling transformer inference Year: (2023)
Ref_id:b56 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b57 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b58 Title: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation Year: (2023)
Ref_id:b59 Title: Flashattention-3: Fast and accurate attention with asynchrony and low-precision Year: (2024)
Ref_id:b60 Title: Fast transformer decoding: One write-head is all you need Year: (2019)
Ref_id:b61 Title: Flexgen: High-throughput generative inference of large language models with a single gpu Year: (2023)
Ref_id:b62 Title: Kv cache in shadows for high-throughput long-context llm inference Year: (2024)
Ref_id:b63 Title:  Year: (2024)
Ref_id:b64 Title: Adobe firefly Year: (2023)
Ref_id:b65 Title:  Year: (2024)
Ref_id:b66 Title:  Year: (2022)
Ref_id:b67 Title: Sora: Creating video from text Year: (2024)
Ref_id:b68 Title: Qwen2.5: A party of foundation models Year: (2024-09)
Ref_id:b69 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b70 Title:  Year: (2017)
Ref_id:b71 Title: Huggingface's transformers: State-of-the-art natural language processing Year: (2019)
Ref_id:b72 Title: Cs 15-859: Algorithms for big data -lecture 11 Year: (2020)
Ref_id:b73 Title: Efficient streaming language models with attention sinks Year: (2023)
Ref_id:b74 Title: Qwen2 technical report Year: (2024)
Ref_id:b75 Title: No token left behind: Reliable kv cache compression via importance-aware mixed precision quantization Year: (2024)
Ref_id:b76 Title: Orthogonal random features. Advances in neural information processing systems Year: (2016)
Ref_id:b77 Title: Quantizing weight and key/value cache for large language models gains more Year: (2024)
Ref_id:b78 Title: Qjl: 1-bit quantized jl transform for kv cache quantization with zero overhead Year: (2024)
Ref_id:b79 Title: KDEformer: Accelerating transformers via kernel density estimation Year: (2023-07)
Ref_id:b80 Title: Token generation in sublinear time and memory Year: (2024)
Ref_id:b81 Title: Kv cache is 1 bit per channel: Efficient large language model inference with coupled quantization Year: (2024)
Ref_id:b82 Title: H2o: Heavy-hitter oracle for efficient generative inference of large language models Year: (2024)
