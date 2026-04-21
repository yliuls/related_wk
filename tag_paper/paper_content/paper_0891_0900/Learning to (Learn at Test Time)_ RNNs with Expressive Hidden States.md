Title: Learning to (Learn at Test Time): RNNs with Expressive Hidden States
Abstract: Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their hidden states. We present a practical framework for instantiating sequence modeling layers with linear complexity and expressive hidden states. The key idea is to make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. Since the hidden state is updated by training even on test sequences, our layers are called Test-Time Training (TTT) layers. We consider two instantiations: TTT-Linear and TTT-MLP, whose hidden state is a linear model and a two-layer MLP respectively. We evaluate our instantiations at the scale of 125M to 1.3B parameters, comparing with a strong Transformer and Mamba, a modern RNN. Similar to Transformer, TTT-Linear and TTT-MLP can keep reducing perplexity by conditioning on more tokens, while Mamba cannot after 16k context. TTT-MLP still faces challenges in memory I/O, but shows larger potential in long context, pointing to a promising direction for future research.

Section: Introduction
This version of the paper has been abridged to fit the page limit of ICML camera ready. Please read our arXiv version instead: https://arxiv.org/abs/2407.04620.
In 2020, the OpenAI scaling law paper (Kaplan et. al (Kaplan et al., 2020)) showed that LSTMs (a type of RNN) could not scale similarly to Transformers or effectively use long context. Now, with modern RNNs and best practices, Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
we re-evaluate these findings in Figure 1.
On the left, we observe that Mamba (Gu & Dao, 2023) -one of the most popular RNNs today -scales similarly to a strong Transformer, showing great progress since the LSTMs in 2020. However, on the right, we observe the same issue with Mamba as Kaplan et al. did with LSTMs. Tokens later in a sequence should be easier to predict on average, since they condition on more information. This is indeed the case for Transformer, whose average perplexity at each token index decreases throughout its 32k context. In contrast, the same metric plateaus for Mamba after 16k. This result represents an awkward reality for existing RNNs. On one hand, the main advantage of RNNs (vs. Transformers) is their linear (vs. quadratic) complexity. This asymptotic advantage is only realized in practice for long context, which according to Figure 8 is after 8k. On the other hand, once context is long enough, existing RNNs such as Mamba struggle to actually take advantage of the extra information being conditioned on.
The difficulty with long context is inherent to the very nature of RNN layers: Unlike self-attention, RNN layers have to compress context into a hidden state of fixed size. As a compression heuristic, the update rule needs to discover the underlying structures and relationships among thousands or potentially millions of tokens. This need is inherently challenging. In this paper, we begin with the observation that self-supervised learning can compress a massive training set into the weights of a model such as an LLM, which often exhibits deep understanding about the semantic connections among its training data -exactly what we need from a compression heuristic.
TTT layers. Motivated by this observation, we make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. Since the hidden state is updated by training even on test sequences, these RNN layers are called Test-Time Training (TTT) layers. We introduce two simple instantiations: TTT-Linear and TTT-MLP, where the hidden state is a linear model and a twolayer MLP, respectively. TTT layers can be integrated into any network architecture and optimized end-to-end, similar to RNNs layers and self-attention.  (right). Evaluations follow Kaplan et al. (Kaplan et al., 2020). Left: Scaling trends on the Pile with 8k context, zoomed in between 350M and 1.3B parameters. Right: Similar to Transformer, TTT-Linear and TTT-MLP can keep reducing perplexity by conditioning on more tokens, while Mamba cannot after 16k context. All methods have matched training FLOPs as Mamba 1.4B.
Wall-clock time. We apply two techniques to make TTT layers more efficient on modern GPUs and TPUs. First, similar to the standard practice of taking gradient steps on mini-batches of sequences during regular training for better parallelism, we use mini-batches of tokens during TTT. Second, we develop a dual form for operations inside each TTT mini-batch. The dual form is equivalent in output to the naive implementation, but trains more than 5× faster on our TPUs.
this section cite: ['b26', 'b20', 'b26']

Section: Contributions and limitations.
The idea of using linear models as hidden states has already been well studied in DeltaNet (Schlag et al., 2021;Yang et al., 2024). Since our first version was released, RNN layers with matrix (linear) hidden states have also been further advanced in Mamba 2 (Dao & Gu, 2024) and Gated DeltaNet (Yang et al., 2023). Compared to this line of work, our contribution is a practical framework that can instantiate arbitrary neural networks as hidden states. However, such instantiations can still require substantial wall-clock time, even after applying our improvements in efficiency. It remains to be seen whether our framework can produce instantiations that either overcome this limitation or offer benefits outweighing it.
this section cite: ['b38', 'b53', 'b13', 'b52']

Section: Method
All sequence modeling layers can be viewed from the perspective of storing historic context into a hidden state, as shown in Figure 2. For example, RNN layers -such as LSTM (Hochreiter & Schmidhuber, 1997) and Mamba (Gu & Dao, 2023) layers -compress context into a state of fixed size across time. This compression has two consequences. On one hand, mapping an input token x t to output token z t is efficient, because both the update rule and output rule take constant time per token. On the other hand, the performance of RNN layers in long context is limited by the expressive power of its hidden state s t .
Self-attention can also be viewed from the perspective above, except that its hidden state, commonly known as the Key-Value (KV) cache, is a list that grows linearly with t. Its update rule simply appends the current KV tuple to this list, and the output rule scans over all tuples up to t to form the attention matrix. The hidden state explicitly stores all historic context without compression, making self-attention more expressive than RNN layers for long context. However, scanning this linearly growing hidden state also takes linearly growing time per token.
To remain both efficient and expressive in long context, we need a better compression heuristic. Specifically, we need to compress thousands or potentially millions of tokens into a hidden state that can effectively capture their underlying structures and relationships.
this section cite: ['b24', 'b20']

Section: TTT as updating a hidden state
The process of parametric learning can be viewed as compressing a massive training set into the weights of a model. Specifically, we know that models trained with self-supervision can capture the underlying structures and relationships behind their training data (Le, 2013) -exactly what we need from a compression heuristic.
LLMs themselves are great examples. Trained with the self-supervised task of next-token prediction, their weights can be viewed as a compressed form of storage for existing knowledge on the internet. By querying LLMs, we can extract knowledge from their weights. More importantly, LLMs often exhibit a deep understanding of the semantic connections among existing knowledge to express new pieces of reasoning (Achiam et al., 2023).
Our key idea is to use self-supervised learning to compress the historic context x 1 , .
. . , x t into a hidden state s t , by making the context an unlabeled dataset and the state a model. Concretely, the hidden state s t is now equivalent to update ... output Hidden state Input tokens Output tokens Output rule Update rule Initial state Update rule Output rule Cost Naive RNN s 0 = vector() W t , the weights of a model f , which can be a linear model, a small neural network, or anything else. The output rule is simply: z t = f (x t ; W t ). Intuitively, the output token is just the prediction on x t , made by f with the updated weights W t . The update rule is a step of gradient descent on some self-supervised loss ℓ:
s t = σ (θ ss s t-1 + θ sx x t ) z t = θ zs s t + θ zx x t O(1) Self-attention s 0 = list() s t = s t-1 .append(k t , v t ) z t = V t softmax K T t q t O(t) Naive TTT W 0 = f.params() W t = W t-1 -η∇ℓ(W t-1 ; x t ) z t = f (x t ; W t ) O(1)
W t = W t-1 -η ∇ℓ(W t-1 ; x t ),(1)
with learning rate η. 1 From the compression point of view, every heuristic needs to decide which input to remember or forget. Our W remembers inputs that produce large gradients -intuitively, inputs that make W learn a lot.
One choice of ℓ is reconstructing x t itself. To make the learning problem nontrivial, we first process x t into a corrupted input xt (details in Subsection 2.3), then optimize:
ℓ(W ; x t ) = ∥f (x t ; W ) -x t ∥ 2 .(2)
Similar to denoising autoencoders (Vincent et al., 2008), f needs to discover the correlations between dimensions of x t in order to reconstruct it from partial information xt . We discuss more sophisticated formulations of the selfsupervised task in Subsection 2.3.
As with other RNN layers and self-attention, our algorithm that maps an input sequence x 1 , . . . , x T to output sequence z 1 , . . . , z T can be programmed into the forward pass of a sequence modeling layer, using the hidden state, update rule, and output rule above. Even at test time, our new layer still trains a different sequence of weights W 1 , . . . , W T for every input sequence. Therefore, we call it the Test-Time Training (TTT) layer.
1 For now, consider W0 = 0. We will discuss more sophisticated techniques for initializing W in Subsection 2.7.
this section cite: ['b31', 'b0', 'b48']

Section: Training a network with TTT layers
The forward pass of a TTT layer also has a corresponding backward pass. Our forward pass only consists of standard differentiable operators except the gradient operator ∇. However, ∇ just maps one function to another, in this case ℓ to ∇ℓ, and ∇ℓ is also composed of differentiable operators. Conceptually, calling backward on ∇ℓ means taking gradients of gradients -a well explored technique in meta-learning (Maclaurin et al., 2015).
TTT layers have the same interface as RNN layers and selfattention, therefore can be replaced in any larger network architecture, which usually contains many of these sequence modeling layers. Training a network with TTT layers also works the same way as training any other language model, such as a Transformer. The same data, recipe, and objective such as next-token prediction can be used to optimize parameters of the rest of the network.
We refer to training the larger network as the outer loop, and training W within each TTT layer as the inner loop. An important difference between the two nested learning problems is that the inner-loop gradient ∇ℓ is taken w.r.t. W , the parameters of f , while the outer-loop gradient is taken w.r.t the parameters of the rest of the network, which we will denote by θ rest . Throughout this paper, outer-loop parameters are always denoted by θ with various subscripts.
this section cite: ['b33']

Section: Learning a self-supervised task for TTT
Arguably the most important part of TTT is the selfsupervised task, because it determines the kind of features that W will learn from the test sequence. So how should we design this task? The final goal of TTT is for z t = f (x t ; W t ) to perform well on language modeling. Instead of handcrafting a self-supervised task from human priors, we take a more end-to-end approach -directly optimizing the selfsupervised task for the final goal of next-token prediction.
Concretely, we learn the self-supervised task as part of the outer loop. Starting from the naive reconstruction task in Equation 2, we add some outer-loop parameters to make this task learnable. In Subsection 2.1, we did not specify the corruption that produces xt from x t . One design is to make it a low-rank projection xt = θ K x t , where θ K is a learnable matrix. 2 Following the terminology of multi-view reconstruction, θ K x t is called a training view.
Moreover, perhaps not all the information in x t is worth remembering, so the reconstruction label can be another low-rank projection θ V x t instead of x t . Here θ V x t is called the label view, where θ V is also learnable. In summary, our new self-supervised loss is:
ℓ(W ; x t ) = f (θ K x t ; W ) -θ V x t 2 . (3
)
Since both W and various θs appear together in Equation 3, we emphasize again their difference in nature. In the inner loop, only W is optimized, therefore written as an argument of ℓ; the θs are "hyper-parameters" of this loss function. In the outer loop, θ K , θ V , θ Q are optimized alongside θ rest , and W is merely a hidden state, not a parameter.
Lastly, the training view θ K x t has fewer dimensions than x t , so we can no longer use the output rule in Subsection 2.1. The simplest solution is to create a test view θ Q x t , and change our output rule to:
z t = f (θ Q x t ; W t ) .(4)
This solution has an additional benefit. The training and label views specify the information in x t that is compressed into W t and propagated forward through time. The test view specifies potentially different information that is mapped to the current output token z t and propagated forward through network layers, therefore adds more flexibility to the selfsupervised task.
this section cite: []

Section: Parallelization with mini-batch TTT
The naive TTT layer developed so far is already efficient in the number of floating point operations (FLOPs). However, its update rule W t = W t-1 -η∇l(W t-1 ; x t ) cannot be parallelized, because W t depends on W t-1 in two places: before the minus sign and inside ∇l. Since ∇l contains the bulk of the computation, we focus on making this second part parallel.
We approach this systems challenge through concepts in the TTT framework. There are many variants of gradient descent (GD). Its general update rule can be expressed as: 1 2 4 8 16 32 64 128 256 512 1024 2048 TTT mini-batch size b (log scale) 11 11.2 11.4 11.6 Perplexity (log scale) 1 2 4 8 16 32 64 128 256 512 1024 2048 TTT mini-batch size b (log scale) 0 100 200 Time (ms) Ws at end of mini-batch Total for Ws and z1, , zT Figure 4. Ablations on TTT mini-batch size b, where b = 1 is online GD and b = T is batch GD. We choose b = 16 for all experiments in this paper. Left: Smaller b improves perplexity since more GD steps are taken. The perplexity of 11.09 at b = 16 corresponds to the final result of TTT-Linear in Figure 6. Right: Forward time in dual form, with context length T = 2048. Total time (orange) can be decomposed into time for computing the W s at the end of every mini-batch (blue) and time for z1, . . . , zT .
W t = W t-1 -η G t = W 0 -η t s=1 G s ,(5)
where G t is the descent direction. Note that once we have calculated G t for t = 1, . . . , T , we can then obtain all the W t s through a cumsum by the second half of Equation 5. Our naive update rule, known as online gradient descent,
uses G t = ∇l(W t-1 ; x t ).
To parallelize G t for t = 1, . . . , T , we can take all of them w.r.t. W 0 . This variant with G t = ∇ℓ (W 0 ; x t ) is known as batch gradient descent, since t s=1 ∇ℓ (W 0 ; x s ) is the same as the gradient w.r.t. W 0 over x 1 , . . . , x t as a batch. However, in batch GD, W t is effectively only one gradient step away from W 0 , in contrast to online GD, where W t is t steps away from W 0 . Therefore, batch GD has a smaller effective search space, which ends up hurting performance for language modeling.
Our proposed solution -mini-batch gradient descent -is shown in Figure 3. Denote the TTT batch size by b. We use G t = ∇ℓ (W t ′ ; x t ), where t ′ = tmod(t, b) is the last timestep of the previous mini-batch (or 0 for the first mini-batch), so we can parallelize b gradient computations at a time. Empirically, b controls a trade-off between speed and quality, as shown in Figure 4. We chose b = 16 for all experiments in this paper.
this section cite: []

Section: Dual form
The parallelization introduced above is necessary but not sufficient for efficiency in wall-clock time. Modern accelerators specialize in matrix-matrix multiplications, known as matmuls. For example, the NVIDIA A100 GPU contains highly optimized units called TensorCores that can only perform a single operation -multiplying two matrices each of size 16 × 16. Without enough of these matmuls, the TensorCores are idle, and most of the potential for the A100 is unrealized.
Unfortunately, the TTT layer developed so far even with mini-batch still has very few matmuls. Consider the simplest case of ℓ, where θ K = θ V = θ Q = I, for only the first TTT mini-batch of size b. In addition, consider f as a linear model. Copying Equation 2, our loss at time t is: ℓ (W 0 ; x t ) = ∥W 0 x t -x t ∥ 2 . As discussed in Subsection 2.4, we can parallelize the computation of:
G t = 2(W 0 x t -x t )x T
t , for t = 1, . . . , b. However, we cannot compute all b of the G t s through a single matmul. Instead, we need b outer products to compute them one by one. To make matters worse, for each x t ∈ R d , G t is d × d, which incurs much heavier memory footprint and I/O cost than x t for large d.
To solve these two problems, we make a simple observation: We do not actually need to materialize G 1 , . . . , G b as long as we can compute W b at the end of the minibatch, and the output tokens z 1 , . . . , z b (see Figure 3). Now we demonstrate these computations with the simplified TTT-Linear case above. Denote X = [x 1 , . . . , x b ], then:
W b = W 0 -2η(W 0 X -X)X T .
So W b can be conveniently computed with a matmul. To compute Z = [z 1 , . . . , z b ], we know that:
z t = f (x t ; W t ) = W 0 x t -2η t s=1 (W 0 x s -x s )x T s x t . (6) Denote δ t = t s=1 (W 0 x s -x s )
x T s x t and the matrix ∆ = [δ 1 , . . . , δ b ]. We can derive that:
∆ = (W 0 X -X) mask X T X ,(7)
where mask is the upper triangular mask with zeros (similar to the attention mask, but with zeros instead of infinities), and the term W 0 X -X can be reused from the computation of W b . Now ∆ is also conveniently computed with matmuls. Plugging ∆ back into Equation 6, we obtain Z = W 0 X -2η∆.
We call this procedure the dual form, in contrast to the primal form before this subsection, where the Gs and W s are explicitly materialized. As discussed, the two forms are equivalent in output. The terminology of primal and dual follows prior work that has explored similar mathematical formulations outside of TTT (Irie et al., 2022;Bishop & Nasrabadi, 2006;Rosenblatt, 1958). In Appendix A, we show that the dual form still works when f is a neural network with nonlinear layers.
Time complexity of the primal form within a TTT minibatch is O(b × d 2 ). Time complexity of the dual form is O(b × d 2 ) for computing W b alone, then an additional O(b 2 ×d) for computing z 1 , . . . , z b . Compared to the primal, the dual form sacrifices theoretical complexity for hardware utilization. In practice, d is typically a few hundred and b is chosen to be only 16. As a consequence, wall-clock time for computing z 1 , . . . , z b is relatively small, as observed in the right panel of Figure 4. In our JAX implementation, training with the dual form is more than 5× faster than with primal.
this section cite: ['b25', 'b5', 'b36']

Section: Theoretical equivalences
In Subsection 2.1, we mentioned that f can be a linear model or a neural network. In Subsection 2.4, we also discussed three variants of the update rule: online GD, batch GD, and mini-batch GD. Each of these 2 × 3 combinations induces a different instantiation of the TTT layer. We now show that among these induced instantiations, the TTT layer with a linear model and batch GD is equivalent to linear attention (Katharopoulos et al., 2020).
Theorem 1. Consider the TTT layer with f (x) = W x as the inner-loop model, batch gradient descent with η = 1/2 as the update rule, and W 0 = 0. Then, given the same input sequence x 1 , . . . , x T , the output rule defined in Equation 4produces the same output sequence z 1 , . . . , z T as linear attention.
Proof. By definition of ℓ in Equation 3,
∇ℓ (W 0 ; x t ) = -2(θ V x t )(θ K x t ) T . By definition of batch GD: W t = t s=1 (θ V x s )(θ K x s ) T .
Plugging W t into the output rule in Equation 4, we obtain the output token:
z t = f (θ Q x t ; W t ) = t s=1 (θ V x s )(θ K x s ) T (θ Q x t )
, which is the definition of linear attention.
In Table 1, we first empirically verify the equivalence above with an improved implementation of linear attention. Then, to illustrate the contribution of each of our components (including some that will be introduced in the next subsection), we add them row by row to the TTT layer that is equivalent to linear attention, and ultimately obtain our proposed instantiation called TTT-Linear. The change from batch GD to mini-batch GD contributes the most improvement by a large margin.
While the space of models × optimizers is already large, machine learning is much richer than optimizing the parameters W t of a model f . There are also nonparametric learners, such as nearest neighbors, support vector machines (SVMs), and kernel ridge regression. By definition, nonparametric learners do not have parameters W t , and instead directly uses training data x 1 , . . . , x t . Hence we use the notation f (x; x 1 , . . . , x t ). We now show that for a particular nonparametric learner, the induced TTT layer is equivalent to self-attention.
Theorem 2. Consider the TTT layer with the Nadaraya- Watson estimator (Bierens, 1988;Cai, 2001), defined as:
f (x; x 1 , . . . , x t ) = 1 t s=1 κ(x, x s ) t s=1 κ(x, x s ) y s , (8
)
where y s = θ V x s , and
κ (x, x ′ ; θ K , θ Q ) ∝ e (θ K x) T θ Q x ′ (9
)
is a kernel with bandwidth hyper-parameters θ K and θ Q .
Then given the same input sequence x 1 , . . . , x T , the output rule defined in Equation 4produces the same output sequence z 1 , . . . , z T as self-attention.
Proof. Plugging y s and κ above into Equation 8gives us the definition of self-attention.
Appendix B contains a detailed explanation of the Nadaraya-Watson estimator and kernel κ above. In contrast to Theorem 1, Theorem 2 does not produce a different implementation from attention.
this section cite: ['b27', 'b4', 'b9']

Section: Implementation details
Instantiations of f . We propose two variants of TTT layers -TTT-Linear and TTT-MLP, differing only in their instantiations of f . For TTT-Linear, f lin (x) = W x, where W is square. For TTT-MLP, f MLP has two layers similar to the MLPs in Transformers. Specifically, the hidden dimension is 4× the input dimension, followed by a GELU activation (Hendrycks & Gimpel, 2016). For better stability during TTT, f always contains a Layer Normalization (LN) and residual connection. That is, f (x) = x + LN(f res (x)), where f res can be f lin or f MLP .
Learnable W 0 . The TTT initialization W 0 is shared between all sequences, even though subsequent weights W 1 , . . . , W T are different for each input sequence. Instead of setting W 0 = 0, we can learn it as part of the outer loop. Since outer-loop parameters are always denoted by θs instead of W s, we assign an alias θ init = W 0 . In practice, θ init Configuration Ppl. Diff.
Linear attention (Katharopoulos et al., 2020)  adds a negligible amount of parameters comparing to the reconstruction views θ K , θ Q , θ V , because both its input and output are low dimensional. Empirically, we observe that learning W 0 significantly improves training stability.
Learnable η. The learning rate is usually the most important hyper-parameter for gradient descent, so we experiment with learning the inner-loop learning rate η in Equation 5as part of the outer loop. We make η a function of the input token (therefore different across time) for additional flexibility. Concretely, we design η(x) = η base σ(θ lr • x), where the learnable vector θ lr is an outer-loop parameter, σ is the sigmoid function, and the scalar η base is the base learning rate, set to 1 for TTT-Linear and 0.1 for TTT-MLP. Alternatively, η(x) can also be interpreted as a gate for ∇ℓ.
Backbone architecture. The cleanest way to integrate any RNN layer into a larger architecture would be to directly replace self-attention in a Transformer, known in this context as a backbone. However, existing RNNs such as Mamba (Gu & Dao, 2023) and Griffin (De et al., 2024) all use a different backbone from Transformers. Most notably, their backbone contains temporal convolutions before the RNN layers, which might help collect local information across time. After experimenting with the Mamba backbone, we find that it also improves perplexity for TTT layers, so we incorporate it into our proposed method. See Figure 9 (in Appendix) for details.
this section cite: ['b22', 'b27', 'b20', 'b14']

Section: Experiments
We evaluate TTT-Linear and TTT-MLP by comparing with two baselines -Transformer and Mamba, a modern RNN.
Our main codebase is based on EasyLM (Geng, 2023), an open-source project for training and serving LLMs in JAX. Datasets. Following the Mamba paper (Gu & Dao, 2023), we perform standard experiments with 2k and 8k context lengths on the Pile (Gao et al., 2020), a popular dataset of documents for training open-source LLMs (Black et al., 2022). However, the Pile contains few sequences of length greater than 8k (de Vries, 2023). To evaluate capabilities in long context, we also experiment with context lengths ranging from 1k to 32k in 2× increments, on a subset of the Pile called Books3, which has been widely used to train LLMs in long context (Liu et al., 2024).
Backbone architecture. As discussed in Subsection 2.7, Transformer and Mamba use different backbones, and TTT-Linear and TTT-MLP always use the Mamba backbone unless noted otherwise. As an ablation study, Figure 6 and Figure 7 contain TTT layers within the Transformer backbone. When a figure contains both the Transformer backbone and Mamba backbone, we denote them by (T) and (M), respectively.
Protocols. To ensure fairness to our baselines, we strictly follow the evaluation protocols in the Mamba paper when possible. For each evaluation setting (e.g., dataset, context length, and method), we experiment with four model sizes: 125M, 350M, 760M, and 1.3B parameters. For Mamba, the corresponding sizes are 130M, 370M, 790M, and 1.4B, as Mamba does not follow the Transformer configurations. All models are trained with the Chinchilla recipe described in the Mamba paper and reproduced in our Appendix C.
this section cite: ['b19', 'b20', 'b18', 'b6', 'b32']

Section: Short context: the Pile
From Figure 6, we make a few observations:
• At 2k context, TTT-Linear (M), Mamba, and Transformer have comparable performance, as the lines mostly overlap. TTT-MLP (M) performs slightly worse under large FLOP budgets. Even though TTT-MLP has better perplexity than TTT-Linear at every model size, the extra cost in FLOPs offsets the advantage.
• At 8k context, both TTT-Linear (M) and TTT-MLP (M) perform significantly better than Mamba, in contrast to the observation at 2k. Even TTT-MLP (T) with the Trans- former backbone performs slightly better than Mamba around 1.3B. A robust phenomenon we observe throughout this paper is that as context length grows longer, the advantage of TTT layers over Mamba widens.
• At 8k context, Transformer still has good (if not the best) perplexity at every model size, but its line is not competitive because of the cost in FLOPs.
this section cite: []

Section: Effect of backbone.
Switching the TTT layers from Mamba backbone into Transformer backbone has two effects. First, TTT layers with Mamba backbone perform better in our evaluations so far. Second, with Mamba backbone, TTT-MLP at best is only comparable to TTT-Linear; but with Transformer backbone, TTT-MLP is clearly better. We hypothesize that the temporal convolutions in the Mamba backbone help more when the sequence modeling layer has a less expressive hidden state. The linear model is less expressive than the MLP, therefore benefits more from the convolutions. We will revisit this hypothesis in the next subsection.
this section cite: []

Section: Long context: Books
To evaluate capabilities in long context, we experiment with context lengths ranging from 1k to 32k in 2× increments, using a popular subset of the Pile called Books3. The training recipe here is the same as that for Pile. From the subset of results in Figure 7, we make a few observations:
• At 2k context on Books, all the observations from Pile 2k still hold, except that Mamba now performs slightly better than TTT-Linear (whereas their lines roughly overlapped for Pile 2k).
• At 32k context, both TTT-Linear (M) and TTT-MLP (M) perform better than Mamba, similar to the observation from Pile 8k. Even TTT-MLP (T) with the Transformer backbone performs slightly better than Mamba at 32k context.
• TTT-MLP (T) is only slightly worse than TTT-MLP (M) at 1.3B scale. As discussed, it is hard to derive an empirical scaling law due to the lack of a clean linear fit. However, the strong trend for TTT-MLP (T) suggests that the Transformer backbone might be more suitable for larger models and longer context beyond our evaluations.
We only ablate the backbones for 2k and 32k due to the cost of training LLMs. For future work, we believe that given TTT layers with even more expressive hidden states, the Mamba backbone with convolutions will be unnecessary.
Transformer finetuning. While we have been training Transformers from scratch following the Mamba paper, in practice this approach is rarely used for long context. The standard practice is to train a Transformer in short context, then finetune in long context. To reflect this practice, we add another baseline, TF finetune, for context lengths 4k and above. This baseline starts from the model trained (according to the Chinchilla recipe) on Books 2k, then uses 20% more tokens to finetune at the designated context length, following the Llama Long paper (Xiong et al., 2023). See details of the TF finetune recipe in Appendix C.
Experiments in Figure 1 (right). Compared to TTT-Linear, TTT-MLP with matched FLOPs performs worse at short context but better at long context. This observation matches our expectation that the MLP as hidden state is more expressive than the linear model: The larger capacity of a more expressive hidden state is well-utilized in long context (therefore an advantage), but redundant in short context (therefore a disadvantage in our setting with matched FLOPs). The Transformer in this figure is TF finetune, which is the stronger baseline in 32k context. Details of the experiments in Figure 1 are included in Appendix C. Our complete results for context lengths 1k, 2k, 4k, 8k, 16k, 32k, including TF finetune, are in Figure 11 (in Appendix).
this section cite: ['b51']

Section: Wall-clock time
LLM training and inference can be decomposed into forward, backward, and generate. Prompt processing during inference, also known as prefill, is the same operation as forward during training, except that the intermediate activations do not need to be stored for backward. Since both forward (during training and inference) and backward can be parallelized, we use the dual form. Generating new tokens, also known as decode, is inherently sequential, so we use the primal form.
Due to resource constraints, our experiments are written in JAX and run on TPUs. On a v5e-256 TPU pod, the Transformer baseline takes 0.30s per iteration of training at context 2k, while TTT-Linear takes 0.27s per iteration, already 10% faster without any systems optimization. However, Mamba (implemented in PyTorch, Triton, and CUDA) can only run on GPUs, so for fair comparison, we also rewrite our method into GPU kernels. We only write inference kernels for this work because the training kernel would require substantial effort and cannot be used on our TPUs.
Figure 8 shows the latency of our inference kernel for forward (prefill) and generate (decode). All models are 1.3B (1.4B for Mamba). As expected, time per token grows linearly for Transformer as the context length increases, but stays roughly constant for the other methods. Note that our Transformer baseline is significantly faster that in the Mamba paper, because we use vLLM (Kwon et al., 2023), a state-of-the-art serving system, instead of the HuggingFace Transformer (Wolf et al., 2019).
this section cite: ['b29', 'b50']

Section: Related Work

this section cite: []

Section: Learning at Test Time
The idea of learning at test time has a long history in machine learning. One of the earliest versions of this idea is called local learning (Bottou and Vapnik (Bottou & Vapnik, 1992)): For each test input, train on its neighbors before making a prediction. This procedure has been effectively applied to models ranging from SVMs (Zhang et al., 2006) to modern LLMs (Hardt & Sun, 2023). Next, we discuss two relevant lines of work in detail: test-time training and fast weights.
this section cite: ['b7', 'b55', 'b21']

Section: TEST-TIME TRAINING
The core idea of Test-Time Training (TTT) is that each test instance defines its own learning problem, where this test instance alone is the target of generalization (Sun et al., 2020). Concretely, for each test instance x, the conventional practice is to predict f (x), using a predictor f that is optimized for all training instances on average. TTT first formulates a learning problem defined by x, then trains a model f x on x (often with f as initialization), and predicts f x (x).
Since the test instance comes without its label, the learning problem can only be formulated with a self-supervised task. Prior work has shown that TTT with reconstruction significantly improves performance especially on outliers (Gandelsman et al., 2022). Improvements become even more pronounced when testing on video frames that arrive in a stream and TTT is autoregressive (Wang et al., 2023), as f t is trained on past frames x 1 , . . . , x t . The autoregressive connection makes (Wang et al., 2023) most relevant to our paper. Conceptually, the biggest difference between our pa-per and prior work is that our reconstruction task is learned in an outer loop, instead of handcrafted with human priors.
this section cite: ['b44', 'b17', 'b49', 'b49']

Section: FAST WEIGHTS
The general idea of fast weights is to update the parameters of a "fast" model on only the most relevant data, as opposed to the conventional practice of updating a "slow" model on all data (Tieleman & Hinton, 2009). This idea has existed since the 1980s (Hinton & Plaut, 1987). The most relevant data can be the test instance itself, therefore TTT can be viewed as a special case of fast weights. Compared to fast weights, TTT embraces the idea of formulating an explicit learning problem, where the test instance is the target of generalization. Our update rule is also an explicit step of optimization.
The idea of fast weight programmers (FWPs) is to update the fast weights with a "slow" model (Schmidhuber, 1992). As a modern example for language modeling, Clark et al. (Clark et al., 2022) give a Transformer a final layer of fast weights, whose initialization is trained as slow weights. Our innerloop weights W can be viewed as "fast" and outer-loop weights θ as "slow". Therefore, networks containing TTT layers can be viewed as a special case of FWPs (Kirsch & Schmidhuber, 2021), similar to how TTT can be viewed as a special case of fast weights.
Modern RNN layers such as linear attention (Katharopoulos et al., 2020;Schlag et al., 2020) and DeltaNet (Schlag et al., 2021;Yang et al., 2024) are inspired by the idea of FWPs. Given their relevance to our work, we discuss these modern RNN layers in detail in the next subsection.
this section cite: ['b46', 'b23', 'b40', 'b12', 'b28', 'b27', 'b37', 'b38', 'b53']

Section: Modern RNN layers
Our baseline, Mamba (Gu & Dao, 2023), is only one of the many recent RNN layers that inherit the linear (matrix) hidden states of linear attention (Katharopoulos et al., 2020;Schlag et al., 2020). Some more recent examples are RWKV (Peng et al., 2024), xLSTM (Beck et al., 2024), and Gated Linear Attention (GLA) (Yang et al., 2023). The most relevant work is DeltaNet (Schlag et al., 2021), which is equivalent to TTT-Linear with inner-loop mini-batch size 1, without the Layer Norm and residual connection. (Yang et al., 2024) further improve the performance of DeltaNet and enable parallelized updates across tokens (in our terms, across inner loop mini-batches). Since our first version was released, RNN layers with matrix (linear) hidden states have also been further advanced in Mamba 2 (Dao & Gu, 2024) and Gated DeltaNet (Yang et al., 2023).
Compared to this line of work, our contribution is a practical framework that can instantiate arbitrary neural networks as hidden states. However, such instantiations can still require substantial wall-clock time, even after applying our improvements in efficiency. For example, TTT-MLP is effective in terms of FLOPs, as shown in Figure 1. But the additional complexity of the MLP structure increases wall-clock time much more relative to FLOPs, as shown in Figure 8. It remains to be seen whether our framework can produce instantiations that either overcome this limitation or offer benefits outweighing it.
this section cite: ['b20', 'b27', 'b37', 'b35', 'b2', 'b52', 'b38', 'b53', 'b13', 'b52']

Section: Learning to Learn
For decades, researchers have been arguing that learning to learn, also known as meta-learning or bi-level optimization, should be a critical component of intelligence (Schmidhuber, 1987;Bengio et al., 1990;Thrun & Pratt, 1998;Lake et al., 2017). In prior work such as (Andrychowicz et al., 2016), (Finn et al., 2017) and (Metz et al., 2018), the inner loop learns from an entire dataset at a time instead of a sequence, so the outer loop needs a collection of datasets or tasks. In short, the outer loop is "one level above" regular training. Since it is hard to collect millions of datasets, this outer loop is hard to scale.
In contrast, for TTT, each sequence itself is a dataset and defines its own generalization problem. The inner loop is "one level below" regular training, so our outer loop is only another solution to the canonical problem of supervised learning, instead of a new problem setting like generalization across datasets.
this section cite: ['b39', 'b3', 'b45', 'b30', 'b1', 'b16', 'b34']

Section: Future work
The search space for effective instantiations inside this framework is huge, and our paper has only taken a baby step. Fortunately, if our perspective holds, then heuristics from regular training can transfer to test-time training, and search can be efficient. Next we outline some especially promising directions for future work:
• Systems optimization. Our systems optimization in Subsection 3.3 has been preliminary at best, and there are many ways to improve it. In addition, pipeline parallelism through time might allow us to process long sequences of millions of tokens on multiple devices together.
• Longer context and larger models. Constrained by our academic resources, we have not trained with millions or billions in context length, which would also require larger models according to Figure 12. The advantage of TTT layers should become more pronounced in longer context.
• More ambitious instantiations of f . When context length becomes longer, f would also need to be larger. For video tasks and embodied agents, whose context length can easily scale up to millions or billions, f could be a convolutional neural network.
this section cite: []

Section: Impact statement
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.
As discussed in Subsection 2.5, the goal of the dual form is to compute XK+1 and W 1 b , . . . , W K b with only matmuls and light-weight operations such as sums, σ, and σ ′ . To achieve this goal, we avoid explicitly computing the intermediate variables: G k t and W k t for t = 1, . . . , b. The dual form first computes ∇ XK+1 l = XK+1 -Y , then takes a standard backward pass.
For k = K, . . . , 1:
• ∇ Z k l = σ ′ k Z k ⊙ ∇ Xk+1 l • ∇ Xk l = W k 0 T ∇ Z k l • ∇ W k 0 l = ∇ Z k l Xk T
where σ ′ is applied element-wise, and ⊙ is element-wise multiplication.
Now we can already compute
W k b = W k 0 -∇ W k 0 l.
To compute the output tokens, we do another forward pass. For k = 1, . . . , K:
• Zk = W k 0 Xk -∇ Z k l • mask Xk T Xk • Xk+1 = σ Zk
By the end of the forward pass, we have computed XK+1 .
While this forward pass is non-standard, it only contains matmuls, sums, σ, and mask, therefore is efficient like the standard forward pass.
this section cite: []

Section: A.4. Derivation
To derive the dual form, we show that:
Zk = W k 0 Xk -∇ Z k l • mask Xk T Xk
is the same as what would be computed in the primal form. Specifically, we show that each column zk t of Zk in the second forward pass of the dual equals to W k t xk t in the forward pass of the primal. We invoke a simple fact.
= [v 1 , . . . , v b ]. 3 Define vt = t s=1 a T s q t v s , and V = [v 1 , . . . , vb ], then V = V • mask(A T Q). Now plug A = Xk , Q = Xk , V = ∇ Z k l, and V = W k
Xk -Zk into the fact above, we have shown the desired equality.
Note that the σ k and σ ′ k used above can be extended to arbitrary functions that are not necessarily element-wise operations, including normalization layers. This extension can be achieved through, for example, vjp (vector-Jacobian product) in standard libraries for automatic differentiation such as JAX and PyTorch. However, the dual form cannot accelerate operations inside σ or its vjp. for our Transformer baseline in long context. Starting at context length 4k, we try θ = 500, 000 following the Llama Long paper (Xiong et al., 2023), and use the better perplexity for Transformer (both pretrain and finetune).
Transformer finetuning. Finetuning starts a new cosine schedule with the same optimization hyper-parameter as training from scratch, except the peak learning rate. We try three peak learning rates for finetuning: 1e-5, 1e-4, and 1e-3, and select for the best perplexity. We observe that 1e-4 works the best for the 125M models, while 1e-5 works the best for 350M and larger. This observation is reasonable considering that the end learning rate for the Chinchilla recipe is 1e-5.
Learning rate for TTT. As mentioned in Subsection 2.7, the inner-loop base learning rate η base is set to 1 for TTT-Linear and 0.1 for TTT-MLP. Our heuristic for setting η base is similar to how people set the outer-loop learning rate for regular training: We tried η base ∈ {0.01, 0.1, 1, 10} and used the largest value that does not cause instabilities. For TTT-MLP, we use linear warmup for η base over 10% of the training steps, similar to regular training. The number of training steps in the inner loop is T /b (assume divisible). For TTT-Linear, we tried linear warmup in the inner loop but did not observe a difference.
this section cite: ['b51']

Section: Experiments in Figure 1 (right).
To ensure fairness to Mamba, all methods in these experiments have matched training FLOPs and are trained with the same recipe (last row of Table 2) as Mamba 1.4B. For TTT-Linear and TTT-MLP, matched training FLOPs also imply matched inference FLOPs. Transformer (TF finetune) has 2.8× the inference FLOPs, giving it an advantage as our baseline. To match training FLOPs with Mamba, Transformer has 19 blocks instead of 24. For TTT-Linear and TTT-MLP, their training FLOPs are already close to those of Mamba, so we only need to change the hidden dimension of the MLP blocks from 5504 to 5808 for TTT-Linear and 5248 for TTT-MLP.
Gradient checkpointing through time. By default, libraries such as JAX and PyTorch save the intermediate activations during a forward pass so they can be reused during the backward pass. However, for a TTT layer with W as hidden state, this default saves W 1 , . . . , W T , which uses too much memory. With TTT mini-batch and the dual form, we still need to save (assume divisible) κ = T /b W s at the end of the mini-batches. A standard technique to save memory in this scenario is gradient checkpointing (Chen et al., 2016), which is usually applied through layers, but we apply it through time.
10 19 10 20 FLOPs (log scale) 10 1 11 12 13 14 15 16 17 18 Perplexity (log scale) 1k TF pretrain Mamba TTT-Linear TTT-MLP 10 19 10 20 FLOPs (log scale) 10 1 9 11 12 13 14 15 16 17 Perplexity (log scale) 2k TF pretrain Mamba TTT-Linear TTT-MLP 10 19 10 20 FLOPs (log scale) 10 1 9 11 12 13 14 15 16 17 Perplexity (log scale) 4k TF finetune TF pretrain Mamba TTT-Linear TTT-MLP 10 19 10 20 FLOPs (log scale) 10 1 12 14 16 Perplexity (log scale) 8k TF finetune TF pretrain Mamba TTT-Linear TTT-MLP 10 18 10 19 10 20 FLOPs (log scale) 10 1 12 14 16 18 Perplexity (log scale) 16k TF finetune TF pretrain Mamba TTT-Linear TTT-MLP 10 18 10 19 10 20 FLOPs (log scale) 10 1 12 14 16 18 Perplexity (log scale) 32k TF finetune TF pretrain Mamba TTT-Linear TTT-MLP Figure 11. Complete results on Books, presented by context lengths. Figure 7 in Subsection 3.2 presents the subset of results for context lengths 2k and 32k. 2k 4k 8k 16k 32k context length (log scale) 16.0 16.5 17.0 17.5 18.0 18.5 perplexity (log scale) 125M TF finetune TF pretrain Mamba TTT-Linear TTT-MLP 2k 4k 8k 16k 32k context length (log scale) 11.4 11.6 11.8 12.0 12.2 12.4 12.6 12.8 perplexity (log scale) 350M TF finetune TF pretrain Mamba TTT-Linear TTT-MLP 2k 4k 8k 16k 32k context length (log scale) 10 1 9.4 9.6 9.8 10.2 10.4 perplexity (log scale) 760M TF finetune TF pretrain Mamba TTT-Linear TTT-MLP 2k 4k 8k 16k 32k context length (log scale) 8.4 8.6 8.8 9.0 9.2 9.4 perplexity (log scale) 1.3B TF finetune TF pretrain Mamba TTT-Linear TTT-MLP Figure 12. An alternative view of our complete results on Books, presented by model sizes, with context length as the x-axis. For all methods trained from scratch, perplexity becomes worse once the context length becomes too large. This trend is not observed with TF finetune, except for one case at the 125M scale. The best context length increases for larger models (trained from scratch).
this section cite: ['b10']

Section: References
Ref_id:b0 Title: Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Learning to learn by gradient descent by gradient descent Year: (2016)
Ref_id:b2 Title: Extended long short-term memory Year: (2024)
Ref_id:b3 Title: Learning a synaptic learning rule Year: (1990)
Ref_id:b4 Title: The nadaraya-watson kernel regression function estimator Year: (1988)
Ref_id:b5 Title: Pattern recognition and machine learning Year: (2006)
Ref_id:b6 Title: Gpt-neox-20b: An open-source autoregressive language model Year: (2022)
Ref_id:b7 Title: Local learning algorithms Year: (1992)
Ref_id:b8 Title: Variable kernel estimates of multivariate densities Year: (1977)
Ref_id:b9 Title: Weighted nadaraya-watson regression estimation Year: (2001)
Ref_id:b10 Title: Training deep nets with sublinear memory cost Year: (2016)
Ref_id:b11 Title: A tutorial on kernel density estimation and recent advances Year: (2017)
Ref_id:b12 Title: Meta-learning fast weight language models Year: (2022)
Ref_id:b13 Title: Transformers are ssms: Generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b14 Title: Mixing gated linear recurrences with local attention for efficient language models Year: (2024)
Ref_id:b15 Title: In the long (context) run Year: (2023)
Ref_id:b16 Title: Model-agnostic metalearning for fast adaptation of deep networks Year: (2017)
Ref_id:b17 Title: Testtime training with masked autoencoders Year: (2022)
Ref_id:b18 Title: The pile: An 800gb dataset of diverse text for language modeling Year: (2020)
Ref_id:b19 Title: A Simple And Scalable Training Framework for Large Language Models Year: (2023-03)
Ref_id:b20 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b21 Title: Test-time training on nearest neighbors for large language models Year: (2023)
Ref_id:b22 Title: Gaussian error linear units (gelus) Year: (2016)
Ref_id:b23 Title: Using fast weights to deblur old memories Year: (1987)
Ref_id:b24 Title: Long short-term memory Year: (1997)
Ref_id:b25 Title: The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlights of attention Year: (2022)
Ref_id:b26 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b27 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b28 Title: Meta learning backpropagation and improving it Year: (2021)
Ref_id:b29 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b30 Title: Building machines that learn and think like people Year: (2017)
Ref_id:b31 Title: Building high-level features using large scale unsupervised learning Year: (2013)
Ref_id:b32 Title: World model on million-length video and language with blockwise ringattention Year: (2024)
Ref_id:b33 Title: Gradientbased hyperparameter optimization through reversible learning Year: (2015)
Ref_id:b34 Title: Meta-learning update rules for unsupervised representation learning Year: (2018)
Ref_id:b35 Title: Eagle and finch: Rwkv with matrixvalued states and dynamic recurrence Year: (2024)
Ref_id:b36 Title: The perceptron: a probabilistic model for information storage and organization in the brain Year: (1958)
Ref_id:b37 Title: Learning associative inference using fast weight memory Year: (2020)
Ref_id:b38 Title: Linear transformers are secretly fast weight programmers Year: (2021)
Ref_id:b39 Title: Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta Year: (1987)
Ref_id:b40 Title: Learning to control fast-weight memories: An alternative to dynamic recurrent networks Year: (1992)
Ref_id:b41 Title: Glu variants improve transformer Year: (2020)
Ref_id:b42 Title: Normformer: Improved transformer pretraining with extra normalization Year: (2021)
Ref_id:b43 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2023)
Ref_id:b44 Title: Test-time training with self-supervision for generalization under distribution shifts Year: (2020)
Ref_id:b45 Title: Learning to learn: Introduction and overview Year: (1998)
Ref_id:b46 Title: Using fast weights to improve persistent contrastive divergence Year: (2009)
Ref_id:b47 Title:  Year: (2023)
Ref_id:b48 Title: Extracting and composing robust features with denoising autoencoders Year: (2008)
Ref_id:b49 Title: Test-time training on video streams Year: (2023)
Ref_id:b50 Title: Huggingface's transformers: State-of-the-art natural language processing Year: (2019)
Ref_id:b51 Title:  Year: (2023)
Ref_id:b52 Title: Gated linear attention transformers with hardware-efficient training Year: (2023)
Ref_id:b53 Title: Parallelizing linear transformers with the delta rule over sequence length Year: (2024)
Ref_id:b54 Title: Root mean square layer normalization Year: (2019)
Ref_id:b55 Title: Discriminative nearest neighbor classification for visual category recognition Year: (2006)
