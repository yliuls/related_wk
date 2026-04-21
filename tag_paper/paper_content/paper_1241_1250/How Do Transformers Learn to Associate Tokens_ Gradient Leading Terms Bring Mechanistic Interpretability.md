Title: HOW DO TRANSFORMERS LEARN TO ASSOCIATE TOKENS: GRADIENT LEADING TERMS BRING MECHANISTIC INTERPRETABILITY
Abstract: Semantic associations such as the link between "bird" and "flew" are foundational for language modeling as they enable models to go beyond memorization and instead generalize and generate coherent text. Understanding how these associations are learned and represented in language models is essential for connecting deep learning with linguistic theory and developing a mechanistic foundation for large language models. In this work, we analyze how these associations emerge from natural language data in attention-based language models through the lens of training dynamics. By leveraging a leading-term approximation of the gradients, we develop closed-form expressions for the weights at early stages of training that explain how semantic associations first take shape. Through our analysis, we reveal that each set of weights of the transformer has closed-form expressions as simple compositions of three basis functions-bigram, token-interchangeability, and context mappings-reflecting the statistics of the text corpus and uncovering how each component of the transformer captures semantic associations based on these compositions. Experiments on real-world LLMs demonstrate that our theoretical weight characterizations closely match the learned weights, and qualitative analyses further show how our theorem shines light on interpreting the learned associations in transformers.

Section: INTRODUCTION
Large language models (LLMs) based on self-attention have shown strong capabilities in capturing both factual knowledge and qualitative aspects of the human world (Grattafiori et al., 2024;Yang et al., 2025;Team et al., 2024;Achiam et al., 2023). This progress has sparked growing interest in understanding why these models work so well and, in particular, what kinds of internal structures emerge during training (Engels et al., 2024;Li et al., 2023a;Meng et al., 2022;Cunningham et al., 2023). Among these structures, semantic associations are especially foundational to language modeling (Harris, 1954;Firth, 1957;Miller & Charles, 1991), as they enable models to connect words and concepts in ways that support generalization and coherent text generation. While recent studies have identified specific mechanisms such as induction heads (Olsson et al., 2022), linear semantic relations (Nanda et al., 2023), and topic clustering (Li et al., 2023b), we still lack a principled account of how semantic associations arise during the training of attention-based transformers.
By semantic associations, we mean the statistical and functional relationships between tokens that encode meaning-for example, the link between "bird" and "flew", the interchangeability of "car" and "truck" in adjectival contexts, or the coupling of "country" and "capital". These associations have long been recognized in linguistics under the lens of distributional semantics (Harris, 1954). In modern transformers, such associations are not explicitly programmed but instead emerge through gradient-based optimization over large corpora. Understanding how these structures crystallize during training is therefore essential not only for connecting deep learning with linguistic theory but also for developing a mechanistic foundation of representation learning in large language models.
In this work, we develop a theory for the emergence of semantic associations in attention-based language models trained on natural language data, through the lens of training dynamics. A formal Figure 1: To understand the emergence of associative features, we analyze the training dynamics of Transformers by focusing on the gradient leading terms for weights, which allows us to identify interpretable basis functions that characterize each weight by their compositions. Empirical validation confirms that our weight characterizations match the actual ones learned in practical transformers.
analysis of training dynamics is attractive as it allows us to rigorously discuss how modern language models learn features and capabilities. Unfortunately, the training dynamics of transformers are highly complex, which has led prior work to adopt unrealistic assumptions that diverge from practice: (1) synthetic structured language (Li et al., 2023b;Yang et al., 2024), (2) simplified model architectures without, e.g., positional encoding or residual connections (Tian et al., 2023;Huang et al., 2025), and (3) non-standard training, such as sequential component-wise training or partially frozen weights (Bietti et al., 2023;Li et al., 2023b). While these prior works provide valuable theoretical insights, their departures from realistic conditions raise concerns about the generalizability of their insights to LLMs used in practice. In contrast, we ground our study in a more realistic setting by focusing on naturalistic text distributions and attention-based transformers with positional encodings, optimized with a standard training procedure (Brown et al., 2020). This is essential to minimize the gap between our theory and practical use.
Our key technical innovation is to analyze training dynamics of the transformer at an early stage, through the leading term of an expansion of the gradients for each set of weights. In particular, transformers are known to acquire many core behaviors early in training-including semantic relationsand persist through convergence (Olsson et al., 2022;Elhage et al., 2021;Nanda et al., 2023). This makes the early phase not only empirically important but also analytically tractable. During this stage, gradient updates admit a closed-form approximation: the leading term dominates parameter updates before higher-order corrections accumulate. Leveraging this, we show that the learned weight matrices (including the output matrix, value matrices, query-key matrices) can be expressed as simple compositions of three basis functions: a bigram mapping, which captures next token dependencies; an interchangeability mapping, which reflects functional similarity across tokens (e.g., synonyms or shared grammatical roles); and a context mapping, which encodes longer-range prefix-suffix co-occurrence.
Through experiments on a natural language dataset, we verify that the learned weights in an attention-based transformer model closely match our theoretical closed-form expressions, and further demonstrate that this holds even beyond the early stage. We also show rich qualitative examples of how each weight component of the transformer captures the actual word-wise semantic associations characterized by our theorem. Furthermore, we verify that our theoretically characterized features are correlated with the behavior of real-world language model. Figure 1 depicts an overview of our analysis, and we summarize our contributions as follows:
1. We present the first explicit characterization of weights in attention-based transformers trained on real-world text corpora under the next-token prediction loss; 2. We interpret the features learned in weights as compositions of bi-gram, interchangeability, and context mappings, and then show how these basis functions capture semantic association across words;
3. We finally validate our theoretical interpretation on both self-attention models and practical LLM, demonstrating the generality and relevance of our theorems.
this section cite: ['b17', 'b44', 'b37', 'b0', 'b13', 'b29', 'b7', 'b18', 'b14', 'b30', 'b34', 'b32', 'b27', 'b18', 'b27', 'b45', 'b38', 'b20', 'b4', 'b27', 'b5', 'b34', 'b12', 'b32']

Section: RELATED WORKS
Understanding emergence of features in Transformers. Many works have considered the training dynamics of transformers under controlled settings to interpret their feature learning (Tian et al., 2023;Bietti et al., 2023;Nichani et al., 2024;Kim & Suzuki, 2024). A line of them investigates how low-level associative features, such as bigram structure (Bietti et al., 2023), cyclic structure (Huang et al., 2025), and co-occurrence (Tian et al., 2023;Yang et al., 2024), are learned from data. There are also multiple works that analyze how high-level capabilities, such as chain-of-thought (Kim & Suzuki, 2025), topic clustering (Li et al., 2023b;Jiang et al., 2024), reasoning or memorization (Yao et al., 2025), and in-context learning capability (Nichani et al., 2024;Bietti et al., 2023;Wang et al., 2024a;Kim & Suzuki, 2024;Edelman et al., 2024), are obtained during training. Although insightful, they often assume structured or abstract language data (Li et al., 2023b;Nichani et al., 2024;Yang et al., 2024), unrealistic model architecture (Tian et al., 2023;Cui et al., 2024;Troiani et al., 2025), and adjusted training strategies far from practice (Bietti et al., 2023;Kim & Suzuki, 2024;Huang et al., 2025), which depart from reality. In contrast, our theoretical analysis is grounded in natural language data, realistic architecture, and a standard training strategy. As a result, our theory substantially reduces the gap between formal analysis and practical use, which is further corroborated by our empirical validations.
Understanding feature learning beyond Transformers. Recent work has also explored how models learn data-dependent features through dynamics for non-transformer models as well (Dandi et al., 2023;Ba et al., 2022;Mousavi-Hosseini et al., 2023). However, this line of work similarly considers abstractions of language, such as Gaussian data (Ba et al., 2022), single or multi-index models (Damian et al., 2024;Dandi et al., 2023), or spiked models (Wang et al., 2024b;Mousavi-Hosseini et al., 2023), and considers measures of data complexity with Hermite expansions (Bietti et al., 2022;Damian et al., 2024;Lee et al., 2024). On the contrary, we adopt a realistic theoretical setup to analyze features in transformers, which remains the dominant architecture in practice.
this section cite: ['b38', 'b4', 'b33', 'b22', 'b4', 'b20', 'b38', 'b45', 'b23', 'b27', 'b21', 'b46', 'b33', 'b4', 'b22', 'b10', 'b27', 'b33', 'b45', 'b38', 'b6', 'b39', 'b4', 'b22', 'b20', 'b9', 'b1', 'b31', 'b1', 'b33', 'b9', 'b31', 'b3', 'b33', 'b25']

Section: PRELIMINARY
3.1 PROBLEM STATEMENT Semantic associations are foundational for language models: they enable models to go beyond memorizing sequences and instead generalize across contexts (Hinton, 1984), infer latent structure (Wu et al., 2018), and generate coherent text. Despite their importance, the mechanisms by which transformers acquire these associations during training remain poorly understood. Towards a mechanistic and theory-grounded interpretation of LLMs in a more realistic setup, we pose the question:
How do semantic associations emerge during the training of attention-based language models on natural language data?
It is worth noting that we focus here on general natural language data, rather than synthetically structured or abstractive language, which has been considered in previous works (Yang et al., 2024;Nichani et al., 2024;Huang et al., 2025). This is essential to minimize the gap between our theory and practical use, since real-world text is highly diverse and is not restricted to a specific structure.
In addition, prior studies (Olsson et al., 2022;Elhage et al., 2021;Nanda et al., 2023) have shown that critical semantic and reasoning abilities, such as induction heads and linear semantic relations, can already emerge in the early stage and be preserved through convergence. This makes the early stage of training a natural and necessary focus for theoretical analysis, which we now develop.
this section cite: ['b19', 'b43', 'b45', 'b33', 'b20', 'b34', 'b12', 'b32']

Section: MODEL ARCHITECTURE
Prior works have analyzed the training dynamics of attention-based models under simplifying assumptions, such as restricting attention to low rank (Cui et al., 2024), removing causal masking (Tian et al., 2023;Yang et al., 2024), without positional encodings (Bietti et al., 2023) or residual streams (Huang et al., 2025). In line with Nichani et al. (2024), we study an attention-based architecture that retains these components: positional encodings, causal masking, and residual streams. To further align with practice, we employ a relative positional encoding scheme, as in T5 (Raffel et al., 2020), rather than augmenting embeddings with absolute position vectors. We begin by introducing the necessary notation before formally defining the transformer computation.
Let V = {e 1 , ..., e j , ..., e |V| } denote the set of vocabulary. For an input sequence of length T , we represent the input as a matrix X ∈ R T ×|V| , where each row of X is the one-hot encoding of the t-th token in the sequence. In an L-layer transformer, the parameters associated with self-attention are given by {W (l) , P (l) , V (l) } L l=1 together with W O , where W (l) ∈ R |V|×|V| is the key-query matrix of layer l, V (l) ∈ R |V|×|V| is the value matrix, P (l) ∈ R T ×T is the learned relative positional encoding, and W O ∈ R |V|×|V| is the output matrix. The model with input X is defined as follows.
Definition 3.1 (Attention-Based Transformer). Given an input matrix X ∈ R T ×|V| , the L-layer attention-based transformer with parameters Θ = {W (l) ,
P (l) , V (l) } L l=1 ∪ {W O } is defined as F Θ (X) = h (L) W O ,(1)
where h L is defined by the recurrence relation, i.e.,
h (l) = h (l-1) + S(Mask(h (l-1) W (l) h (l-1)⊤ + DM(P (l) )))h (l-1) V (l) and h (0) = X, (2
)
where S(•) represents the softmax function, DM(v) maps the ith element of v to the (-i+1)th subdiagonal, and Mask(•) denotes the operator of attention mask. This architecture is in line with Nichani et al. (2024), and recent work shows that self-attention-only models can match the performance of architectures with MLP layers (Wang et al., 2025).
this section cite: ['b6', 'b38', 'b45', 'b4', 'b20', 'b33', 'b36', 'b33', 'b41']

Section: TRAINING SETUP
Learning objective. To align with standard language modeling practice and ensure comparability with prior works (Huang et al., 2025;Nichani et al., 2024), we adopt the standard cross-entropy objective: given N input matrices X 1 , ..., X N with sequence length T and corresponding output matrices Y 1 , ..., Y N , where Y i ∈ R T ×|V| , the objective function is defined as
L(Θ) = -1 N T N i=1 T t=1 log S(F θ (X i ) [t] )Y [t]⊤ i ,(3)
where M [t] denotes the t-th row of a matrix M and Y
[t]
i corresponds to the one-hot embedding for the t + 1-th token of the sequence corresponding to X i .
this section cite: ['b20', 'b33']

Section: Gradient descent.
We analyze the evolution of the parameters under full-batch gradient descent with a constant learning rate η. Under gradient descent, the parameters are updated as follows:
Θ(t) = Θ(t -1) -η∇ Θ L(Θ).(4)
Due to the nonlinear complexities of the gradient, deriving an exact form for even one of the weight matrices after t steps is challenging. We address this challenge by considering a leading-order approximation technique, allowing for a closed-form expression of the gradients and weights while yielding a tight approximation of the full gradient.
this section cite: []

Section: THEORETICAL ANALYSIS
In Section 4.1, we provide theorems demonstrating that the weights of attention-based transformers remain close to their gradient leading terms for O(1/η) steps under both zero and Gaussian initializations. Then, Section 4.2 uncovers how three basis functions, which are crucial to express token associations and language structure, are encapsulated in those gradient leading terms, and how these three functions are compounded to shape the desiderata of the transformers' weight matrices.
this section cite: []

Section: MAIN THEOREMS
Under the setup described in Sec. 3, we obtain the following results for attention-based transformers.
Figure 2: Illustration of theoretical results. We characterize weight matrices of the attention-only transformer as compositions of three basis functions: bigram mapping, interchangeability mapping, and context mappings. We illustrate how these mappings are composed across weight matrices to learn semantic associations between a given query token and its surrounding text.
Theorem 4.1. (Informal) Given an attention-based transformer (Def. 3.1) under sufficiently small Gaussian initialization, with L ≤ √ T /4, after s gradient descent steps with learning rate η ≥ 1 T , if s ≤ η -1 min( 5 8 √ T , 1 12L ), then for all layers l = 1, . . . , L,
W O -sη B F ≤ 3s 2 η 2 ,(5)
V (l) - s 2 η 2 Φ⊤ B⊤ F ≤ 12s 3 η 3 ,(6)
W (l) -3 s 4 + 2 s 3 η 4 Q F ≤ 13s 5 η 5 T,(7)
P (l) -3 s 4 + 2 s 3 η 4 ∆ F ≤ 13s 5 η 5 T,(8)
where ∥ • ∥ F is the Frobenius norm, B corresponds to a bigram statistic, Φ corresponds to a context co-occurrence statistic, Q corresponds to a token-to-token correlation based on a composition of B and Φ, and ∆ corresponds to a relative position correlation based on the same feature as Q.
The above Theorem shows that any finite-depth L-layer attention-based transformer (Def. 3.1) has the same characterization for its weights uniformly across all layers under a zero-initialization (Theorem D.9) and a small Gaussian initialization (Theorem 4.1), suggesting that all layers of the model capture common associative features from natural language as a starting point before evolving differently as training progresses (Figure 6). As seen in Figure 2, compositions of these features form the leading terms of the output matrix ( B), value matrix ( Φ B⊤ ), and query-key matrix ( Q). We walk through these matrices in Section 4.2.1 and how they form the weights of the model in Section 4.2.2. The formal theorem and proofs are in Appendix D.
this section cite: []

Section: INTERPRETATION OF THEOREMS
In the previous section, we showed that the model parameters can be approximated by key corpus statistics B, Φ, Q and ∆. Now, we discuss the definitions of these statistics by first introducing three basis functions and explaining how their composition characterizes the model's behavior.
this section cite: []

Section: THREE BASIS FUNCTIONS SHAPING ASSOCIATIVE FEATURES
(1) Bigram mapping B. The (i, j)-th element in Bij corresponds to a correlation between token e i and token e j based on how likely e i is to be directly followed by e j as a bigram. More precisely,
Bij = P t (e i )P t (e j |e i ) -P t (e i )/|V|,(9)
where P t (e i ) is the relative frequency of e i over all tokens in the dataset X 1 , ..., X N and P t (e j |e i ) is the relative frequency of e j given that the previous token was e i . The product between P t (e i ) and P t (e j |e i ) forms an estimate of the likelihood of e i followed by e j appearing as a bigram and the second term -P t (e i )/|V| simply acts as a centering term such that each row sums to 0.
(2) Interchangeability mapping ΣB. We study ΣB = B⊤ B, the correlation matrix of B, which captures correlations between pairs of tokens based on a frequency-weighted similarity of their previous-token distributions. From Eq. ( 9), the (i, j)-th element of ΣB can be represented as
P t (e i )P t (e j ) Frequency weighting |V| k=1 P t (e ← k |e i )P t (e ← k |e j ) Previous token similarity .
In essence, Eq. ( 10) shows that ΣB captures a symmetric relationship between tokens based on how similar of a function or role they play across different contexts. Specifically, in Eq. ( 10), we can see that the corresponding row, which acts as a feature for token e i captures its associations with interchangeable tokens captured by the previous token similarity factor and frequent tokens captured by the frequency weights. Similarities in previous token distributions are an indicator of functional similarities or interchangeability, as this captures structural patterns such as nouns being preceded by articles or adjectives and objects being preceded by common descriptors. This interchangeability map, ΣB, acts a building block of characterizations for the weights W (l) and (l) as illustrated in Figure 2. We depict a simple example of a word-wise correlation captured by ΣB in Figure 1.
P
(3) Context mapping Φ. The (i, j)-th element of Φ corresponds to a correlation between token e i and e j based on how likely e j is to appear as a prefix of e i . This can be written as
1 T T k=1 1 k k m=1 P t (the k + 1 -th token is e i , the m -th token is e j ) -µ j ,(11)
the pond contains fish 7 24 4 Figure 3: An example of Φ with arrows pointing to prefix tokens for "fish" with context summary scores on edges. Larger values indicate the token appears more frequently in the context of "fish".
where µ j centers the columns of Φ to be 0. Considering each row as an embedding for a token e i , which represents an average of the tokens that appear in its context, i.e., smoothed context.
More precisely, the strength of the association from token e i to e j is determined by the average probability that e j appears in the context of e i over possible positions of e i and e j . This matrix can be interpreted as assigning a representation to a token based on a summary of the possible contexts that token e i appears in. This allows for learning associations between words that capture richer semantic relationships than bigram features. For example, we could expect to see correlations between animal and habitat, country and capital, or emotions and facial expressions (See Figure 3). This context mapping Φ is a core building block of the gradients for the query-key attention W (l)  and value V (l) matrices as shown in Figure 2.
this section cite: []

Section: COMPOSITION OF BASIS FUNCTIONS FOR SEMANTIC ASSOCIATION
We now show how these three basis functions, bigram mapping B, interchangeability mapping ΣB, and context mapping Φ, are compounded to characterize four classes of weight matrices of the transformer.
(1) Output matrix W O . As shown in Eq. ( 5), B is the leading term of W O , and thus the mapping from embedding vectors to output predictions can be understood by examining the matrix product e i B for a token embedding e i . The j-th element of the resulting output vector is Bij , and each Bij includes a factor of P t (e i ). This implies that tokens are scored according to how frequently they occur in the average next-token distribution of e i , and explain how models at early stages effectively learn bigram-like patterns.
(2) Value matrix V (l) . The leading term of the value matrix V (l) can be expressed as Φ⊤ B⊤ as noted in Eq. ( 6), which acts as a composition of a context summary and bigram mapping. Because Φ⊤ captures longer-term dependencies and B⊤ captures only bigram statistics, the resulting embedding from V (1) still endows the original token representations with semantic properties similar to those of Φ⊤ as seen in Figure 2.
(3) Attention matrix W (l) . Theorem 4.1 characterizes the attention weight (a shared query-key matrix) as Q, which is constructed as a composition of ΣB, Φ, the input matrix X i , the output matrix Y i , etc. We note that this compound feature captures a token-to-token correlation determined by how predictive one token is of the other's next-token distribution based on the context and interchangeability mappings. We walk through an overview of the construction of Q in three steps (See Appendix A for details).
1. Input-output matching scoring in context. As a preliminary step, we first define a composed feature ΣB Φ by multiplying the interchangeability mapping ΣB with the transpose of the context mapping Φ. This composition utilizes local interchangeability to map a token to a class of similar tokens and utilizes the context mapping to capture longer-range semantic correlations shared by the set of similar tokens. Using this feature, for each sample, we assign scores between each input and output token.
2. Masking and centering. The auto-regressive constraint is enforced by masking future tokens, keeping only scores from input tokens that precede the output token. Then, the resulting scores for each output token are centered and normalized based on its position.
this section cite: []

Section: Next-to-query shift and averaging.
The scores between each input and output token are then shifted so that the same score is assigned instead to be between the input token and the token directly preceding the output token. Then, the scores are averaged across all samples.
(4) Positional encoding P (l) . The closed-form characterization ∆ of the positional encoding P (l)  follows a very similar composition to Q, with the main difference being that the correlations are mapped to positional differences rather than to the vocabulary-space differences (See Lemma D.1).
this section cite: []

Section: HOW THE WEIGHTS COOPERATE
To illustrate how the weights work together and provide further context on the role of each of the weights as functions, we consider the leading-term computation of a single-layer attention-based model. Dropping constant factors to focus on the interactions between features, the leading terms of the entire model computation can be written as
S Mask X QX ⊤ + DM(∆) X Φ⊤ B⊤ + X B.(12)
We can further decompose this into XW O and the computation from the self-attention block is:
S Mask X QX ⊤ + DM(∆) X Φ⊤ ΣB.(13)
Q and ∆ capture correlations between two tokens or two positions based on how predictive the first token/position is of the next-token distribution of the second token/position according to ( Φ⊤ ΣB) ⊤ . Notice that the attended tokens are mapped to the output space by Φ⊤ ΣB, the same feature that determines the correlations for attention. As a result, the self-attention block effectively attends to tokens that, under the value and output matrix projection, lead to better next-token prediction. Thus, we find that while the residual stream XW O provides an average prediction of the next token, Q enables the model to refine this prediction by selectively focusing on tokens most indicative of the next-token given its current parameters, those capturing corpus association statistics.
Implication. By considering an end-to-end analysis of the model under simultaneous training of layers and by decomposing the weights, we obtain a clear interpretation of how different components collaborate to form semantic representations and can rigorously contextualize the function of each component in the full computation of attention-based transformers. While these features only yield small changes in the actual text output, they provide important insight into how the model's behavior develops during training. For example, if early training already associates fish with pond (as in Figure 3), we expect such relationships to be a useful anchor for later training, allowing the model to complete more complex sentences, e.g., "A pond in the garden was filled with colorful fish that sparkled in the sunlight", coherently with learned semantic associations.
this section cite: []

Section: 3-LAYER ATTENTION-BASED TRANSFORMER
We begin with an experimental setting designed to closely mirror our theory, enabling direct verification of results and analysis of the semantic relationships embedded in the learned weights. For clearer interpretability, we use the TinyStories dataset (Eldan & Li, 2023), truncated to the 3,000 most frequently occurring words, which also defines the model's vocabulary. A 3-layer self-attention model defined in Definition 3.1 is then trained with sequence length T = 200.
this section cite: ['b11']

Section: Verification of theory.
To verify Theorem 4.1, we measure the cosine similarity between the learned weights and their corresponding leading terms at checkpoints over the first 100 epochs of SGD using a batch size of 2048 for computational tractability with a learning rate of 0.005. We also consider the cosine similarity between the learned weights and their leading terms when using a larger learning rate of 0.05 to understand how features evolve at later stages with respect to the leading term gradients. We provide results for both settings in Table 1 and Figure 4. The results show that the learned weights maintain strong agreement with the theoretical predictions: even after 30 epochs, all weights achieve a cosine similarity of at least 0.9. Moreover, all parameter matrices have a cosine similarity above 0.7, even after 100 epochs, where the loss had dropped from 8.00 to 5.35. These findings suggest that the features predicted by the theorem not only characterize the model dynamics during the early stage, but also remain informative well beyond it. We provide results for a BPE tokenization and for a causal analysis
in Appendix B, and we elaborate experimental details for the TinyStories experiments in Appendix C. the park little bird ball dog big tree man box red ball car dress balloon truck blocks apples shirt hat to the play go be help see her make do (a) Examples for B they she they he it one lily timmy tom her happy happy sad excited scared proud angry nice curious surprised wanted saw had wanted asked went loved ran looked took (b) Examples for ΣB fish fish big small pond lake water catch sea boat flower beautiful yellow butterfly hose garden bloom pretty daisy field birds bird tree up park nest flowers tweety sky flew (c) Examples for Φ Semantic structure. To validate our interpretation of associative features, we collect for each token the top 30 most correlated tokens under each of the basis functions: the bigram mapping ( B), interchangeability mapping (ΣB), and context matrix ( Φ), constructed from the TinyStories corpus. We provide examples of tokens where the expected semantic relationships can be observed in Figure 5. Under B, we see that the word "red" is correlated with common objects such as "truck"
that would be described by the word "red". Under Φ, we can see that the word "fish" is correlated with common settings where fish would appear, such as "pond" or "lake".
this section cite: []

Section: TRANSFORMERS IN PRACTICE
Setup. To evaluate how well our theoretical results extend to practical LLMs, we analyze token relationships learned from OpenWebText (Gokaslan et al., 2019), a real-world large-scale dataset with text from millions of webpages, in Pythia-1.4B (Biderman et al., 2023) and compare them with our theoretical predictions, examining how these relationships evolve across layers on datasets and models reflecting real-world complexities. We choose the Pythia model family, as they are opensourced and uniquely provide access to intermediate checkpoints, enabling fine-grained analysis of training dynamics and interpretability (Marks et al., 2024;Gallego-Feliciano et al., 2025). Unlike our theoretical setting, Pythia includes additional components such as MLP and multi-head attention, making it impossible to directly read off average token correlations from the weights. In order to interpret the layer-wise representations in terms of token-token correlations, we perform the analysis through the following steps:
1. We pass in each token e i as the input to the transformer.
2. For each token and from each layer l, we collect the following embeddings: the input to layer l h i,l,pre , the output of the l-th layer h i,l,post , and the output of the l-th layer without the MLP component h i,l,attn .foot_0 3. The embeddings h i,l,pre form the rows of E l,pre ∈ R |V|×d which represents a mapping from the input embeddings of layer l to tokens. Similarly, the embeddings h i,l,post and h i,l,attn form the rows of E l,post ∈ R |V|×d and E l,attn ∈ R |V|×d respectively.
this section cite: ['b16', 'b2', 'b28', 'b15']

Section: Attention correlations.
To analyze the correlations captured by the attention weights at each layer, we compute the product of the key and query mappings for each head and average these products, which we will call A l,emb ∈ R d×d . We then multiply the mapping E l,pre on both sides of A l,emb to convert the average attention mapping into a token-basis attention weight matrix A l,tok . Finally, we consider token correlations captured by A l,tok by using its covariance matrix, which we compare with the covariance matrix of Q, the leading-order attention mapping term from our theorem.
this section cite: []

Section: Embedding correlations.
To analyze the correlations captured by the value mapping and the MLP, we consider the token-token correlations captured by the output of each layer. Utilizing the covariance matrix of E l,post allows for direct comparison with the covariance matrix of the leading value matrix term Φ⊤ B⊤ , since the matrices themselves have different dimensions. Furthermore, this enables us to control for shifts in the embedding space.
Comparison methodology. We compute the leading term matrices using 100K samples from OpenWebText. To control for differences in model architecture, we normalize each row of the leading term weights to have unit norm. Then, we compute cosine similarities between the corresponding covariance matrices across layers and across checkpoints. We perform the same analysis on the FineWeb
(Penedo et al., 2024) dataset and provide results in Appendix B. More details on the experimental setup are in Appendix C. 1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Layer Number Attention Mapping Cosine Similarity 1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Layer Number No MLP Embedding Cosine Similarity 1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Layer Number Embedding Mapping Cosine Similarity 0.0 0.2 0.4 0.6 0.8 1.0 Cosine Similarity 0.0 0.2 0.4 0.6 0.8 1.0 Cosine Similarity 0.0 0.2 0.4 0.6 0.8 1.0 Cosine Similarity Cosine Similarity Across Checkpoints Results. We provide a visualization of results in Figure 6, where we can see that, at the early stage of training, there is very strong agreement between the Pythia embeddings and our leading-term features. We can see that for the embedding mapping, the token representations strongly match our theoretical analysis across all layers, and similarly for the attention weights, excluding only the first layer. We can see that as the model continues training, the weights gradually drift from fixed associative features to represent richer knowledge beyond association, starting with the earlier layers. However, it still maintains these features to a large extent for relatively longer steps. This suggests that our analysis on attention-based models generalizes with the addition of multi-head attention or MLP and acts as a starting point for a finer-grained analysis of full training dynamics.
this section cite: []

Section: MLP ablation.
We perform an ablation at each layer by performing the embedding correlation analysis using E l,attn , which is based on only the output of the attention block and excludes the MLP component. The results for this analysis can be seen in the middle plot of Figure 6. We can see that the correlations captured by embeddings with and without the MLP are similar except at the first layer. This suggests that at the first layer, the MLP maps tokens to embeddings with structures similar to that of the leading-order value matrix term and maintains a similar structure at later layers. Based on these initial results, one possible hypothesis is that the MLP at early stages functions similarly to the leading-term value mapping.
this section cite: []

Section: Individual attention heads.
In order to capture a fine-grained understanding of the attention block, we perform the analysis on attention correlations using individual attention heads. We perform this analysis at an early (Layer 2), middle (Layer 13), and late layer (Layer 24) to also understand how heads may evolve differently at different stages of the model. In Figure 7, we find that different layers evolve differently with respect to the gradient leading-term for attention mappings.
The earlier layers learn the leading-term features at a slower rate, as seen by the high similarity (red) appearing at later steps, especially for layer 2. We can also see that layer 13 exhibits faster specialization of attention heads than the other layers, as seen by the high variance in each column at later steps for layer 13. This provides insight into the rate of specialization of attention heads and suggests that intermediate layers are where specialization initially occurs.
1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 Head Index Layer 2 0 1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 Head Index Layer 13 1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 Head Index Layer 24 0.2 0.3 0.4 0.5 0.6 0.7 0.8 Cosine Similarity Per-Head Attention Cosine Similarity Across Training Steps
this section cite: []

Section: ETHICS STATEMENT
We provide a novel theorem that characterizes the roles of weights in the transformer model, which is a de facto standard building block of modern LLMs. We try to uphold high standards of scientific excellence by making minimal assumptions for theoretical analysis while providing practical implications on mechanistic interpretability. The new insights on emerging features we presented contribute to a better understanding and diagnosis of the representation learning of transformers, which makes a big step towards transparent and reliable AI. As our study considers a setup of training from scratch on public datasets, there is no direct privacy issue and harm. The authors also acknowledge and respect the ethics of confidentiality and fairness, and confirmed that there are no identified violations of them.
this section cite: []

Section: REPRODUCIBILITY STATEMENT
All the theoretical analyses in this work are accompanied by full proofs with detailed step-by-step explanations (in the Appendix) for verification, reproduction, and reuse. We elaborate on the details of our setup in the main body of the paper for all empirical validations, and the code is available here. In addition, our choice of models pursues maximum reproducibility and accessibility, given its simple and fully open-sourced configurations.
this section cite: []

Section: A DETAILED DESCRIPTION ON WEIGHT CHARACTERIZATION
The token-to-token correlation captured by Q is determined by how strongly correlated one token is with the other's next-token distribution. These correlations are captured by Q i where each element Q i jk of Q i measures for X i , the correlation between the token at position j and the token at position k + 1. This correlation between the token at position j and position k + 1 gets mapped back to a correlation between the tokens at positions j and k through X ⊤ i Q i X i . let Q i given in Eq. ( 16) be the per-example correlation matrix computed from input-output token pairs in ith input,
Q = 1 N T N i=1 X ⊤ i Q i X i .(14)
We walk through an overview of the construction of Q i in four steps and provide the detailed computation in Appendix A.
Feature composition. As a preliminary step, we first define a composed feature ΣB Φ by multiplying the interchangeability mapping ΣB with the context mapping Φ. Each entry corresponds to the average product of path weights from token e i to e j with one step on ΣB and one step on Φ. This composition utilizes local interchangeability to map a token to its more general functional class and utilizes the context-summary to capture longer-range semantic correlations shared by tokens in the functional class. We will refer to the resulting feature as the composed feature for simplicity in the remaining steps.
Scoring input-output pairs. For each input X i and its corresponding output Y i , we utilize the composed feature, ΣB Φ, to compute correlation scores between input and output tokens as seen in Figure 2.
(Y i -U O )ΣB ΦX ⊤ i ,(15)
where U O is a baseline matrix with all elements set to 1/|V|. This assigns a correlation score to each input-output token pair according the composed feature.
this section cite: []

Section: Masking and centering.
The auto-regressive constraint is enforced by masking future tokens, keeping only scores from input tokens that precede the output token. The resulting scores for each output token are centered and normalized based on its position. matrix is then centered so that the scores for each output token sum to zero, yielding the per-example matrix Q i .
Q i = ein tjk, tk→tj J i , (Y i -U O )ΣB ΦX ⊤ i ,(16)
where J i is the masking operator and ein denotes an Einstein summation.
Next to Query Mapping. Lastly, the scores between each input and output token are then mapped to be the correlation between the input token and the token preceding the output token. In this way, the model learns to attend to the input token when it expects the next token to be the output token.
Aggregation across the dataset. Finally, we map per-example correlations back to the vocabulary space and average over all N inputs and T tokens per input:
Q = 1 N T N i=1 X ⊤ i Q i X i .(17)
In this way, each token is associated with the average correlations to other tokens across the dataset.
this section cite: []

Section: B ADDITIONAL EXPERIMENTS
BPE tokenization. We train a 3-layer attention-based model on TinyStories as in Section 5.1 using a BPE tokenization with vocabulary size of 10,000. We train the model for 10 epochs with a learning rate of 0.005 and measure the cosine similarity between the theoretical and actual weights. We report the minimum over the 10 epochs in
Table 2. Weights Min. Cosine Attention 0.999914 Value 0.998800 Output 0.997891 Table 2: Minimum cosine similarities between theoretical and actually learned weights across all epochs. Results from a 3-layer attention-based model trained on TinyStories and with a BPE tokenization.
Causal intervention. We aim to understand how the model output changes when removing the leading terms from each of the weights. We perform this analysis on the 3-layer attention-based transformers trained on TinyStories with a learning rate of 0.05. Unlike most causal intervention settings, the features considered have a general function rather than a specific function applicable to a narrower setting, and therefore, we expect removing the leading terms to result in performance degradation across the dataset. As a result, we choose to focus on the extent to which the output distribution changes when the leading term component is removed for each weight matrix. For each weight matrix, we remove the projection of the weight matrix onto its corresponding leading term. After removing this projection, we compute the loss of the resulting model on the dataset. We provide the results of this intervention in Table 3. We can see that the output layer has the largest effect on the loss, while the attention weights have the least. This behavior is predicted by the theory as the output layer has the largest order update, while the attention weights have the smallest order updates.
Weights Loss Original 5.349 Attention Layer 0 5.350 Attention Layer 1 5.352 Attention Layer 2 5.361
Value Layer 0 6.192 Value Layer 1 6.526 Value Layer 2 6.520 Output 8.287
Table 3: Loss of the attention-based model on TinyStories after the leading term component from each weight matrix is removed. The first row corresponds to the original model.
Validation on additional dataset.
We perform the analysis in Section 5.2 on the token-token correlations captured by embeddings in Pythia-1.4B except instead of using OpenWebText, we use FineWeb (Penedo et al., 2024). We provide the results in Figure 8 where we see very similar results as with OpenWebText.
1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Layer Number Attention Mapping Cosine Similarity 1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Layer Number No MLP Embedding Cosine Similarity 1 2 4 8 1 6 3 2 6 4 1 2 8 2 5 6 5 1 2 Training Step 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 Layer Number Embedding Mapping Cosine Similarity 0.0 0.2 0.4 0.6 0.8 1.0 Cosine Similarity 0.0 0.2 0.4 0.6 0.8 1.0 Cosine Similarity 0.0 0.2 0.4 0.6 0.8 1.0 Cosine Similarity Cosine Similarity Across Checkpoints (FineWeb)
this section cite: ['b35']

Section: C EXPERIMENTAL DETAILS
TinyStories Experiments We collect the vocabulary from TinyStories treating each word, punctuation mark, or number as a token and use the 3000 most common tokens. We then filter out samples that include tokens outside of the set of 3000. For training, we use 65536 of the filtered samples with sequence length at least 201 and truncate all sequences to 201 tokens for training and computing theoretical leading terms. For the BPE tokenization, we tokenize the dataset using a vocabulary size of 10,000, and for training, we use samples with sequence length at least 201 and truncate all sequences to 201 tokens for training and computing theoretical leading terms. We compute the theoretical matrices using the first batch.
Pythia Experiments We use the first 100k samples of OpenWebText/FineWeb with length at least 512 characters to perform the analysis.
We utilize 4 A100 GPUs with 80GB of memory. These experiments can be performed with less compute by reducing batch size or sequence length.
this section cite: []

Section: D PROOFS
∥•∥ will be the operator norm unless denoted otherwise.
this section cite: []

Section: D.1 PROOF OF 1-LAYER THEOREM
Lemma D.1 (General Gradient Form). Under the setting described, we have that
∂L ∂W O = -1 N T N i=1 h (1)⊤ i R i (18
) ∂L ∂V (1) = -1 N T N i=1 X i A (1)⊤ i R ⊤ i W ⊤ O (19
) ∂L ∂W (1) = -1 N T N i=1 X ⊤ i ein tjk,tk→tj (J i , (R i W ⊤ O V (1)⊤ X ⊤ i ))X i (20
) ∂L ∂P (1) = -1 N T ein tjk,jk→t D, N i=1 ein tjk,tk→tj (J i , (R i W ⊤ O V (1)⊤ X ⊤ i ))(21)
where A
i = S(Mask(X i W (1) X ⊤ i + P (1) )), R i = Y i -S(F θ (X i )), J i ∈ R T ×T ×T with J i,t = Diag(A (1)[t] i ) -A (1)[t]⊤ i A (1)[t] i(1)
being the Jacobian of the softmax function for the tth token in the sequence, D ∈ R T ×T ×T with D t being a matrix with ones along the (-t + 1)th sub-diagonal and zeros elsewhere, and ein is used to denote an Einstein summation.
Proof. We start by considering the derivative of the loss with respect to F θ (X i ) [t] which is
Y [t] i -S(F θ (X i ) [t] )(22)
and derivative of F θ (X i ) [t] with respect to W O is h
(1)[t] i
. Then it follows that
∂L ∂W O = -1 N T N i=1 h (1)⊤ i R i(23)
Now, we consider the gradient with respect to V (1) using the chain rule which gives
∂L ∂V (1) = -1 N T N i=1 X i A (1)⊤ i R ⊤ i W ⊤ O (24
)
Published as a conference paper at ICLR 2026 Now, we consider the gradient with respect to A
i as an intermediate step towards the gradient with respect to W (1) , P (1) . Using the chain rule as before, we have 1) , we have that the derivative of A
∂L ∂A (1) i = -1 N T N i=1 R i W ⊤ O V (1)⊤ X ⊤ i (25
) Letting B (1) i = X i W (1) X ⊤ i + P (
1)[t] i with respect to B (1)[t] i is J i,t = Diag(A (1)[t] i ) -A (1)[t]⊤ i A (1)[t] i
Then, in order to get the gradient of the loss with respect to B
(1) i , we need to consider the contribution from each t resulting in the Einstein summation
∂L ∂B (1) i = -1 N T N i=1 ein tjk,tk→tj (J i , R i W ⊤ O V (1)⊤ X ⊤ i )(27)
From the chain rule, we can derive the gradient with respect to both W (1) and P (1) .
∂L ∂W (1) = -1 N T N i=1 X ⊤ i ein tjk,tk→tj (J i , R i W ⊤ O V (1)⊤ X ⊤ i )X i (28
) ∂L ∂P (1) = -1 N T ein tjk,jk→t D, N i=1 ein tjk,tk→tj (J i , R i W ⊤ O V (1)⊤ X ⊤ i )(29)
where D t has ones along the (-t + 1)th sub-diagonal and zeros elsewhere. This completes the proof.
Lemma D.2 (Softmax Jacobian Norm). The norm of the Jacobian of the softmax function applied to a vector with i elements unmasked is at most 1/ √ i.
Proof. If v is the masked vector before softmax is applied, then the Jacobian as S(v) j (1 -S(v) j ) as the jth element on the diagonal and -S(v) j S(v) k for the element in the jth row and kth column with j ̸ = k. The norm is maximized when the output is the uniform distribution and is less than 1/ √ i.
this section cite: []

Section: Lemma D.3 (First Gradient Step).
Under the setting described, after one gradient step, we have that
W O = η( B)(30)
W (1) , V (1) , P (1) = 0 (31
)
where B a |V | × |V | matrix where the jth row is the average next-token distribution of the jth token in the vocabulary weighted by the relative frequency of token j across the dataset and centered to have the row sum be 0.
Proof. From Lemma 1.1, as the parameters are initially zero, we can see that W (1) , V (1) , P (1) all have gradients of zero and therefore remain as 0. For W O , as the value matrix is initially zero, h
(1) i = X i and as W O is zero, the output distribution for every token is the uniform distribution. Let U O ∈ R T ×|V | represent the resulting output with each element 1/|V |. Then, we have that
∂L ∂W O = -1 N T N i=1 X ⊤ i (Y i -U O )(32)
We consider the sum of each of the terms X ⊤ i Y i and X ⊤ i U O . First, we consider the sum of
X ⊤ i Y i . The jth row of X ⊤ i Y i is a |V |-dimensional
vector with each element being the number of times the corresponding token appears after each occurrence of the jth token in the vocabulary in X i . Then, summing over all i and dividing by N T results in each row mapping to the average next-token distribution weighted by the frequency of the token corresponding to the row. We can write this as
B = 1 N T N i=1 X ⊤ i Y i =     α 1 P 1 α 2 P 2 . . . α |V | P |V |    (33)
where α j is the relative frequency of the jth token in the dataset and P j is the average next-token distribution for token j. For the sum U ⊤ O X i divided by N T , we simply get that every row is α j times the uniform distribution over the vocabulary, and we will denote this matrix by U . Then, we have that
∂L ∂W O = -(B -U )(34)
and therefore after the first step,
W O = η(B -U )(35)
Then, as B = B -U , this completes the proof.
this section cite: []

Section: Lemma D.4 (Second Gradient Step).
Under the setting described, after two gradient steps, we have that
W O -2η B F ≤ η 2 |V | (36
) V (1) -η 2 Φ⊤ B⊤ F ≤ 2η 3 |V | (37
)
W (1) , P (1) = 0 (38
)
where B is as defined in the previous lemma and Φ is given by
Φjk = P(e k → e j ) -µ Φ,k(39)
where P(e k → e j ) corresponds to the empirical probability that e j is the current token and e k is in its prefix and µ Φ,k is the value that sets each column sum to 0.
Proof. First, as V (1) remains at zero after the first step, we have that the gradients for W (1) , P (1)  are zero and therefore, they remain at zero after the second step. We now consider the forward pass after the first gradient step. As the value matrix remains as zero, we have that
F θ (X i ) = ηX i B(40)
Then, by the Softmax Jacobian lemma, we have that
∥S(F θ (X i )) -U O ∥ F ≤ η |V | X i B F ≤ η √ T |V | (41
)
Then, we have that
∥R i -(Y i -U O )∥ F ≤ η √ T |V | (42
)
and by Lemma 1.1 and that ∥X i ∥ ≤ √ T , we have
∂L ∂W O + B F ≤ 1 N T N i=1 ηT |V | = η |V | (43
)
Then, it follows that after the second gradient step,
W O -2η B F ≤ η 2 |V | (44
)
Now, we consider the gradient with respect to V (1) . By Lemma 1.1, we have that
∂L ∂V (1) = -1 N T N i=1 ηX ⊤ i A (1)⊤ i (Y i -S(F θ (X i ))) B⊤ (45
)
and since W (1) , P (1) = 0, A
(1) i = A 0 where the tth row of A 0 has the first t elements equal to 1/t and the rest equal to 0. Then, by equation 41, we have that
ηX ⊤ i A 0 (Y i -S(F θ (X i ))) B⊤ -ηX ⊤ i A ⊤ 0 (Y i -U O ) B⊤ F ≤ η 2 T |V | ∥A 0 ∥ (46
)
Then, using the discrete Hardy's inequality with p = 2, we have that ∥A 0 ∥ ≤ 2 and
∂L ∂V (1) - 1 N T N i=1 ηX ⊤ i A ⊤ 0 (Y i -U O ) B⊤ F ≤ 2η 2 |V | (47
)
Now, we will analyze
1 N T N i=1 ηX ⊤ i A ⊤ 0 (Y i -U O ) B⊤ (48
)
Since η B is independent of i, we can move it outside the sum and we can analyze
1 N T N i=1 X ⊤ i A ⊤ 0 (Y i -U O )(49)
We start by consider the form of X ⊤ i A ⊤ 0 . Since the tth row of A 0 has 1/t as the first t elements and zeros for all other elements, we have that the jth element of the tth column of
X ⊤ i A ⊤ 0 is γ i (e j , t) t (50
)
where e j represents the jth token in the vocabulary and γ i (e j , t) is the number of occurrences of e j in the first t tokens of
X i . Letting Φ ′ = 1 N T N i=1 X ⊤ i A ⊤ 0 Y i , we have that Φ ′ jk = 1 N T N i=1 T t=1 1(X [t+1] i = e k ) γ i (e j , t) t (51
)
Swapping the order of the sums, we have
Φ ′ jk = 1 N T T t=1 1 t N i=1 1(X [t+1] i = e k )γ i (e j , t)(52)
Then, as γ i (e j , t) = t m=1 1(X [m]   i = e j ), we have that
Φ ′ jk = 1 T T t=1 1 t t m=1 1 N N i=1 1(X [t+1] i = e k , X [m] i = e j )(53)
Then, as 1 N N i=1 corresponds to an average over the dataset, the average over N corresponds to the empirical probability of having a sequence with the t + 1th token equal to e k and the mth token equal to e j , which we will denote as P(x t+1 = e k , x m = e j ). Then, this gives
Φ ′ jk = 1 T T t=1 1 t t m=1 P(x t+1 = e k , x m = e j )(54)
Then, we have an average over m ∈ [t] which results in the average probability that the t + 1th token is e k and e j is in the first t tokens. We will denote this as P(e j ∈ x 1:t , x t+1 = e k ). This gives
Φ ′ jk = 1 T T t=1 P(e j ∈ x 1:t , x t+1 = e k ) (55
)
This probability of e j being in the prefix of x t+1 = e k is averaged over the different positions of e k to get an average probability that e j is in the prefix given that e k is the current token, which we will denote as P(e j → e k ) and
Φ ′ jk = P(e j → e k )(56)
Now, we consider
U P = 1 N T N i=1 X ⊤ i A ⊤ 0 U O .
Then, we have that
U P jk = 1 N T N i=1 T t=1 γ i (e j , t) |V |t (57
)
Rearranging the sum and decomposing γ i , we get
U P jk = 1 T T t=1 1 t|V | t m=1 P(x m = e j ) (58
)
Then, if we consider the average over positions m and t, we get the average probability that e j is in the first t tokens over all t multiplied by 1/|V |. We can notice that the sum of the jth row of U P and Φ ′ are the same. Then, setting
Φ′ = Φ ′ -U P (59
) we have ∂L ∂V (1) + η Φ′ B⊤ F ≤ 2η 2 |V | (60
)
Then, it follows that after two gradient steps,
V (1) -η 2 Φ′ B⊤ F ≤ 2η 3 |V |(61)
Defining Φ = Φ′⊤ , we have
V (1) -η 2 Φ⊤ B⊤ F ≤ 2η 3 |V | (62
)
This completes the proof.
this section cite: []

Section: Lemma D.5 (Third Gradient Step).
Under the setting described, after three gradient steps with η, we have that
W O -3η B F ≤ 3η 2 (63) V (1) -3η 2 Φ⊤ B⊤ F ≤ 2η 3(64)
W (1) -2η 4 Q F ≤ 2η 5 T (65
)
P (1) -2η 4 ∆ F ≤ 2η 5 T (66
)
Proof. First, we start with bounding the norm of W O , V (1) , A 0 . We have that
∥W O ∥ F ≤ 2η + η 2 |V | (67) V (1) ≤ 2η 2 + 2η 3 |V | (68
)
∥A 0 ∥ ≤ 2 (69
)
Now, we consider the deviation of the output from the uniform distribution. We start by bounding the norm of
X i + A 0 X i V (1) X i + A 0 X i V (1) ≤ 1 + 5η 2 √ T (70
)
Then, we can upper bound the norm of F θ (X i )
∥F θ (X i )∥ F ≤ 5η 2 1 + 5η 2 √ T ≤ 4η √ T (71
)
and then using the Softmax Jacobian lemma, we have that
∥S(F θ (X i )) -U O ∥ F ≤ 4η √ T |V | (72
)
Then, it follows that
∂L ∂W O + B F ≤ 1 N T N i=1 8ηT |V | + 5η 2 T = 2η(73)
and
W O -3η B F ≤ η 2 |V | + 2η 2 ≤ 3η 2(74)
Now, we consider the gradient with respect to V (1) . Since
∥Y i -S(F θ (X i ))∥ ≤ √ 2T , we have that ∂L ∂V (1) + 2η Φ⊤ B⊤ F ≤ 2 4η |V | 5η 2 + η 2 |V | ≤ 22η 2 |V | ≤ η 2(75)
Then, we have that after the third step,
V (1) -3η 2 Φ⊤ B⊤ F ≤ 2η 3(76)
Now, we consider the gradient with respect to W (1) , P (1) which according to Lemma 1.1 are
∂L ∂W (1) = -1 N T N i=1 X ⊤ i ein tjk,tk→tj (J i , R i W ⊤ O V (1)⊤ X ⊤ i )X i (77
) ∂L ∂P (1) = -1 N T ein tjk,jk→t D, N i=1 ein tjk,tk→tj (J i , R i W ⊤ O V (1)⊤ X ⊤ i ) (78
) We start by analyzing R i W ⊤ O V (1)⊤ X ⊤ i . First, We will use that ∥Y i -U O ∥ ≤ √ Tand
Φ ≤ 1 T max i ∥X i ∥ ∥A 0 ∥ ∥Y i -U O ∥ ≤ 2 T T = 2(79)
We know by the previous lemma and the bound on the deviation of the output that
R i W ⊤ O V (1)⊤ X ⊤ i -2η 3 (Y i -U O ) B⊤ B ΦX ⊤ i F ≤ 25η 4 T |V + 5T η 4 2 |V | + 4η 4 T |V | ≤ 3η 4 T 2 (80
)
We start by considering the structure of
Y i B⊤ B ΦX ⊤ i . We first consider the simpler multiplication of e j B⊤ B Φe ⊤ k . First, we define Σ B = B⊤ B which has Σ Bmn = |V | l=1 α 2 l
Plm Pln where α l is the relative frequency of token l and Pl is the average next-token distribution of token e l centered at 0. This corresponds to a similarity measure of the previous tokens of e m and e n with common tokens more heavily weighted. Then, we have that
e j Σ B Φe ⊤ k = |V | m=1 Σ Bjm P(e m → e k ) (81
)
where P(e m → e k ) is the probability that e m is in the prefix of e k centered at 0. We can then interpret the each element (Σ B Φ) jk as a measure of assocation between token j and k based on a two-step chain of (interchangeability mapping, suffix token mapping). Essentially, how often does token e k succeed token e j and similar tokens. We will let Ḡ = Σ B Φ. Now, we can consider 2η 3 Y i ḠX ⊤ i . This results in a T × T matrix where the jk-th element is P(X
[k] i → 2 X [j+1] i
) and we will denote this as g ijk . Then, 2η 3 (Y i -U O ) ḠX ⊤ i will have elements centered to have row sums of 0 and we will let the centered elements be ḡijk . Then, we consider the einsum of J and 2η 3 (Y i -U O ) ḠX ⊤ i . The tth column of the resulting matrix will the be product of J t and the tth column of 2η 3 (Y i -U O ) ḠX ⊤ i . This results in the tth row having the form
2η 3                   ḡi1t t . . . ḡitt t 0 . . . 0          -          µg,it t . . . µg,it t 0 . . . 0                   ⊤ (82
)
The tth row is 2η 3 ḡijt for 1 ≤ j ≤ t weighted by 1/t and centered to have a row sum of 0 and we will refer to the centered and weighted elements 2η 3 q ijt and the resulting matrix
Q i . Letting Q = 1 N T N i=1 X ⊤ i Q i X i , we have that ∂L ∂W (1) + 2η 3 Q F ≤ 2η 4 T(83)
where we have used that the squared Frobenius norm of the einsum is the sum of the norms of each column and that ∥J t ∥ = 1/t. Then, it follows that
W (1) -2η 4 Q F ≤ 2η 5 T(84)
Now, we consider the gradient with respect to each element of
P (1) ∂L ∂P (1) m + 2η 3 N T N i=1 Tr(D -m Q i ) F ≤ 2η 4 √ T(85)
Since the trace is a linear function, we let ∆ m = Tr(D -m 1 N T N i=1 Q i ) and let ∆ be the vector consisting of ∆ m , and we have
∂L ∂P (1) + 2η 3 ∆ F ≤ 2η 4 T (86
)
and it follows that
P (1) -2η 4 ∆ F ≤ 2η 5 T(87)
This completes the proof.
this section cite: []

Section: Theorem D.6 (Early Stage Features).
Under the setting described, for s ≤ η -1 3 8T 3/8 , for T ≥ 3, |V | ≥ 500, we have that after s gradient descent steps with learning rate η,
W O -sη B F ≤ 3s 2 η 2(88)
V (1) - s 2 η 2 Φ⊤ B⊤ F ≤ 4s 3 η 3(89)
W (1) -3 s 4 + 2 s 3 η 4 Q F ≤ 6s 5 η 5 T (90
)
P (1) -3 s 4 + 2 s 3 η 4 ∆ F ≤ 6s 5 η 5 T(91)
Proof. We will prove the result by induction. The previous lemmas form the base case. The first phase will be bounding the deviation of the output from the uniform distribution after s gradient steps. To start, we bound the norm of A i , the resulting attention mapping for X i . We know from Lemma D.2, that for each row of the attention mapping
∥(A i -A 0 )[t, :]∥ ≤ 1 √ t (X i W (1) X ⊤ i [t, : t] + P (1) [: t](92)
Using the fact that there are t elements in
(X i W (1) X ⊤ i + P (1) )[t, : t] with magnitude at most max km |W (1) km | + max m |P (1)
m | and applying the inductive hypothesis for P (1) and W (1) , we have that
∥(A i -A 0 )[t, :]∥ ≤ 3 s 4 + 2 s 3 η 4 (max km | Qkm | + max m |∆ m |) + 12s 5 η 5 T √ t (93
) Since the maximum magnitude of an element of Σ B is |V | k=1 α 2 k ≤ 1
and each column of Φ⊤ has elements with magnitudes that sum to at most 1, we know that each element of Ḡ has magnitude at most 1. Then, it follows that
max km | Qi,km | ≤ 1 and therefore max km | Qkm |, max m |∆ m | ≤ 1.
Then, we have
∥(A i -A 0 )[t, :]∥ ≤ 6 s 4 + 4 s 3 η 4 + 12s 5 η 5 T √ t (94
)
Then, summing the upper bounds on the squared norms of each row, and using that r q=1 1/q ≤ 1 + log r we have
∥A i -A 0 ∥ F ≤ 6 s 4 + 4 s 3 η 4 √ T + 12s 5 η 5 T 1 + log T(95)
Then, we have that
A (1) i ≤ 2 + 6 s 4 + 4 s 3 η 4 √ T + 12s 5 η 5 T 1 + log T(96)
Then, upper bounding 6 s 4 + 4 s 3 by 2s 4 , we have
A (1) i ≤ 2 + 2s 4 η 4 √ T + 12s 5 η 5 T 1 + log T(97)
Now, using that sη ≤ 3 8T 3/8 , we have that
2s 4 η 4 √ T + 12s 5 η 5 T 1 + log T ≤ 2sη(98)
and
A (1) i ≤ 2 + 2sη(99)
Now, we bound the norm of V (1) which by the inductive hypothesis we have is at most
s 2 η 2 2 √ 2 + 4s 3 η 3(100)
which is at most
s 2 η 2 √ 2 + 4s 3 η 3(101)
and since sη
≤ 1 3 , V (1) F ≤ 4s 2 η 2(102)
Then, we have that
X i + A (1
)
i X i V (1) F ≤ √ T 1 + 16s 2 η 2(103)
Since sη ≤ min 3 8T 3/8 and by the inductive hypothesis we have that
∥F θ (X i )∥ F ≤ 2 √ T sη + 3s 2 η 2 (104) Since sη ≤ 3 8T 3/8 , we have ∥F θ (X i )∥ F ≤ 4sη √ T(105)
Then, by Lemma D.2, we have
∥S(F θ (X i )) -U O ∥ F ≤ 4sη T |V |(106)
Now, we will utilize the bound on the deviation of the output from the uniform distribution as well to perform the inductive step for W O . We have based on the bound that after the (s + 1)th step
∂L ∂W O + B F ≤ 8sη |V | + 12s 2 η 2 ≤ 6sη(107)
Then, after (s + 1) steps, we have that
W O -(s + 1)η B F ≤ 3η 2 s 2 + 6sη 2 ≤ 3η 2 (s + 1) 2(108)
Now, we perform the inductive step for V (1) using the bound on the deviation of the attention pattern and on the output deviation. We have that after the (s + 1)th step
∂L ∂V (1) + sη Φ⊤ B⊤ F ≤ ∥R i -(Y i -U O )∥ F A (1) i ∥X i ∥ ∥W O ∥ + ∥(Y i -U O )∥ A (1) i -A 0 F ∥X i ∥ ∥W O ∥ + ∥(Y i -U O )∥ ∥A 0 ∥ ∥X i ∥ W O -B F (109)
Applying upper bounds and using that |V | ≥ 500, we have
∂L ∂V (1) + sη Φ⊤ B⊤ F ≤ 1 T 24sη |V | T ηs + 4T s 2 η 2 + 6T s 2 η 2 ≤ 12s 2 η 2(110)
Then, after (s + 1) steps, we have that
V (1) - s + 1 2 η 2 Φ⊤ B⊤ F ≤ 4s 3 η 3 + 12s 2 η 3 ≤ 4(s + 1) 3 η 3(111)
Now, we perform the inductive step for W (1) utilizing the earlier bounds on the output and attention pattern deviations. We start by bounding the deviation between s 3 -s 2 2 η 3 Σ B Φ and W ⊤ O V (1)⊤ . By the inductive hypothesis and 2 ≤ |V |, we have that
W ⊤ O V (1)⊤ - s 3 -s 2 2 η 3 Σ B Φ F ≤ 8s 4 η 4 + 3 √ 2s 4 η 4 ≤ 13s 4 η 4 (112
)
Then, for each R i W ⊤ O V (1)⊤ X ⊤ i since |V | ≥ 500, we have that R i W ⊤ O V (1)⊤ X ⊤ i - s 3 -s 2 2 η 3 (Y i -U O )Σ B ΦX ⊤ i F ≤ 20s 4 η 4 T |V | + 13s 4 η 4 T ≤ 14s 4 η 4 T(113)
Now, in order to consider the deviation of the einsum of J and
s 3 -s 2 2 η 3 (Y i -U O )Σ B Φ⊤ X ⊤ i
, we need to first bound the deviation of the Jacobian of the current attention pattern from J. We do so by considering the deviation for each J t . As proven earlier, we have that
∥(A i -A 0 )[t, :]∥ ≤ 6 s 4 + 4 s 3 η 4 + 12s 5 η 5 T √ t (114
)
Then, we have that for the current Jacobian for the sample X i corresponding to the tth row which will call J t,i
J t,i -J t = Diag(A i [t, :] -A 0 [t, :]) -A 0 [t, :](A i [t, :] -A 0 [t, :]) ⊤ -(A i [t, :] -A 0 [t, :])A 0 [t, :] ⊤ -(A i [t, :] -A 0 [t, :])(A i [t, :] -A 0 [t, :]) ⊤ (115
)
and it follows then that for t ≥ 2
∥J t,i -J t ∥ 2 ≤ ∥A i [t, :] -A 0 [t, :]∥ ∞ + 2 √ t ∥A i [t, :] -A 0 [t, :]∥ 2 + ∥A i [t, :] -A 0 [t, :]∥ 2 2(116)
Then, as
∥A i [t, :] -A 0 [t, :]∥ ∞ ≤ ∥A i [t, :] -A 0 [t, :]∥ 2(117)
Published as a conference paper at ICLR 2026 we have that ∥J t,i -J t ∥ 2 ≤ 6 s 4 + 4 s 3 η 4 + 12s 5 η 5 T √ t 1 + 2 √ t + 6 s 4 + 4 s 3 η 4 + 12s 5 η 5 T √ t (118)
Since J 1,i is always all zeros, we can ignore this term and for t ≥ 2, we have that as sη ≤ 3 8T 3/8 , ∥J t,i -J t ∥ 2 ≤ 5s 2 η 2 (119)
Then, we have that
s 3 -s 2 2 η 3 Q i -ein tjk,tk→tj (J i , R i W ⊤ O V (1)⊤ X ⊤ i ) F ≤ ∥J i -J∥ 2 s 3 -s 2 2 η 3 (Y i -U O )Σ B ΦX ⊤ i F + ∥J i ∥ 2 R i W ⊤ O V (1)⊤ X ⊤ i - s 3 -s 2 2 η 3 (Y i -U O )Σ B ΦX ⊤ i F (120
)
Then, as ∥J t ∥ 2 = 1 t , ∥J i ∥ 2 ≤ 3 2 + 5s 2 η 2 √ T ≤ 2, and sη ≤ 3 8T 3/8 , we have that
s 3 -s 2 2 η 3 Q i -ein tjk,tk→tj (J i , R i W ⊤ O V (1)⊤ X ⊤ i ) F ≤ 5 √ 2s 5 η 5 T +28s 4 η 4 T ≤ 30s 4 η 4 T (121
)
Then, we have that ∂L ∂W (1) +
s 3 -s 2 2 η 3 Q F ≤ 30s 4 η 4 T (122
)
Then, we have that after (s + 1) steps,
W (1) -3 s + 1 4 + 2 s + 1 3 η 4 Q F ≤ 6s 5 η 5 T + 30s 4 η 5 T ≤ 6(s + 1) 5 η 5 T (123) Finally, as we have the bound on the deviation from Q i , we have that for P (1) , ∂L ∂P (1) + s 3 -s 2 2 η 3 ∆ F ≤ 30s 4 η 4 T (124) and that after (s + 1) steps, P (1) -3 s + 1 4 + 2 s + 1 3 η 4 ∆ F ≤ 6s 5 η 5 T + 30s 4 η 5 T ≤ 6(s + 1) 5 η 5 T (125) D.2 PROOF OF MULTI-LAYER THEOREM Lemma D.7 (General Gradient Form). Under the setting described, defining
S (l) i = ein tjk, tk→tj J (l) i , G (l) i V (l)⊤ h (l-1)⊤ i , (126
) G (l-1) i = G (l) i + A (l)⊤ i G (l) i V (l)⊤ + S (l) i h (l-1) i W (l)⊤ + S (l)⊤ i h (l-1) i W (l) , (127
) with G (L) i = R i W ⊤ O (128) we have that ∂L ∂W O = -1 N T N i=1 h (L)⊤ i R i , (129
) ∂L ∂V (l) = -1 N T N i=1 h (l-1)⊤ i A (l)⊤ i G (l) i , (130
) ∂L ∂W (l) = -1 N T N i=1 h (l-1)⊤ i S (l) i h (l-1) i , (131
) ∂L ∂P (l) = -1 N T ein tjk, jk→t D, N i=1 S (l) i ,(132)
where
A (l) i = S(Mask(h (l-1) i W (l) h (l-1)⊤ i + P (l) )), R i = Y i -S(F θ (X i )), J (l) i ∈ R T ×T ×T with J (l) i,t = Diag(A (l)[t] i ) -A (l)[t]⊤ i A (l)[t] i
being the Jacobian of the softmax function at the lth attention layer for the tth token in the sequence, D ∈ R T ×T ×T with D t being a matrix with ones along the -tth sub-diagonal and zeros elsewhere, and ein denotes an Einstein summation.
Proof. We begin by considering the derivative of the loss with respect to F θ (X i ) [t] , which is
∂L ∂F θ (X i ) [t] = Y [t] i -S(F θ (X i ) [t] ) = -R [t] i (133) Since ∂F θ (X i ) [t] ∂W O = h (L)[t] i (134
) it follows that ∂L ∂W O = -1 N T N i=1 h (L)⊤ i R i (135
)
We now consider the gradient through each attention layer in terms of the current and previous layer embeddings h l) , where
(l-1) i , h (l) i . Let h = h (l-1) i , A = A (l) i , G = G (l) i = ∂L/∂U (l) i , V = V (
U (l) i = (Ah)V . We have ∂L ∂V (l) = (Ah) ⊤ G = h ⊤ A ⊤ G (136
)
Summing over i and normalizing yields
∂L ∂V (l) = -1 N T N i=1 h (l-1)⊤ i A (l)⊤ i G (l) i(137)
The gradient for
M = Ah is δ M = GV (l)⊤ , δ A = δ M h ⊤ , δ(1)
h = A ⊤ δ M .
Next, through the row-wise softmax, each row Jacobian is
J (l) i,t = Diag(A (l)[t] i ) -A (l)[t] i A (l)[t]⊤ i (138)
Stacking these gives a tensor J (l) i . Applying it row-wise to δ A gives
S (l) i = ein tjk, tk→tj J (l) i , G (l) i V (l)⊤ h (l-1)⊤ i (139) Finally, back-propagating through Ã = hW (l) h ⊤ + P (l) gives ∂L ∂W (l) = h ⊤ S (l) i h, (140
) ∂L ∂P (l) = ein tjk, jk→t (D, S (l) i )(141)
Summing over i and normalizing,
∂L ∂W (l) = -1 N T N i=1 h (l-1)⊤ i S (l) i h (l-1) i (142
) ∂L ∂P (l) = -1 N T ein tjk, jk→t D, N i=1 S (l) i(143)
Collecting all contributions to h (l-1) i
gives
G (l-1) i = G (l) i + A (l)⊤ i G (l) i V (l)⊤ + S (l) i h (l-1) i W (l)⊤ + S (l)⊤ i h (l-1) i W (l)(144)
Since at the last layer,
G (L) i = ∂L ∂h (L) i = ∂L ∂Z i W ⊤ O = R i W ⊤ O (145
)
we can inductively apply the recurrence and collecting the per-layer parameter derivatives gives the desired expressions for ∂L/∂W O , ∂L/∂V (l) , ∂L/∂W (l) , and ∂L/∂P (l) .
this section cite: []

Section: Lemma D.8 (First Step, Multi-Layer Zero-Initialization).
Under the setting described, after one gradient step, we have that
W O = η( B)(146)
W (l) , V (l) , P (l) = 0 (147
)
for 1 ≤ l ≤ L where B is a |V | × |V | matrix where the jth row is the average next-token distribution of the jth token in the vocabulary weighted by the relative frequency of token j across the dataset and centered to have the row sum be 0.
Proof. By Lemma D.7,
∂L ∂W O = -1 N T N i=1 h (L)⊤ i R i = -1 N T N i=1 X ⊤ i (Y i -U O ) = -(B -U ) ≡ -B (148
)
where B and U are defined the same as in the one-layer case. A single gradient step gives
W O = -η ∂L ∂W O = η B (149
)
At initialization W O = 0, so by Lemma D.7 the upstream gradient from layer L is
G (L) i = R i W ⊤ O = 0 (150
)
Using the recurrence (Lemma D.7),
G (l-1) i = G (l) i + A (l)⊤ i G (l) i V (l)⊤ + S (l) i h (l-1) i W (l)⊤ + S (l)⊤ i h (l-1) i W (l) (151) Since G (L) i = 0 and W (l) , V (l) = 0 for all l, we inductively get G(l)
i = 0 for every l. Now, the layerwise gradients are (Lemma D.7)
∂L ∂V (l) = -1 N T N i=1 h (l-1)⊤ i A (l)⊤ i G (l) i = 0 (152
)
and with
S (l) i = ein tjk, tk→tj J (l) i , G (l) i V (l)⊤ h (l-1)⊤ i we also have S (l) i = 0 (because G (l) i = 0 or V (l,0) = 0), hence ∂L ∂W (l) = -1 N T N i=1 h (l-1)⊤ i S (l) i h (l-1) i = 0 (153
) ∂L ∂P (l) = -1 N T ein tjk,jk→t D, N i=1 S (l) i = 0 (154
) Therefore a single gradient step leaves W (l) , V (l) , P (l) = 0 for 1 ≤ l ≤ L).
Theorem D.9 (Early Stage Features, Multi-Layer). Fix a depth L ≤ √ T 4 and assume zero initialization for all parameters. Under the setting described, for s ≤ η -1 min 1 12L , 5 8 √ T
with T ≥ 60 and |V | ≥ 500, after s gradient descent steps with learning rate η we have, uniformly for every layer 1 ≤ l ≤ L, W O -sη B F ≤ 3s 2 η 2 (155)
V (l) - s 2 η 2 Φ⊤ B⊤ F ≤ 12s 3 η 3 (156
)
W (l) -3 s 4 + 2 s 3 η 4 Q F ≤ 13s 5 η 5 T (157
)
P (l) -3 s 4 + 2 s 3 η 4 ∆ F ≤ 13s 5 η 5 T (158
)
where B, Φ, Q, and ∆ are as in the one-layer analysis (row-centered bigram matrix, centered prefix-statistics operator, and the third-step structures, respectively).
Proof. We prove the bounds simultaneously for all layers with induction.
By the previous lemma, with zero initialization and one step,
W O = η B, W (l) = 0, V (l) = 0, P (l) = 0 for 1 ≤ l ≤ L).
This gives the base case. Now we prove the inductive step. Assume the four bounds hold after s steps, with (s + 1)η ≤ min 1 12L , 5 8 √ T . We derive bounds on the deviations in the attention patterns, activations, and outputs in the forward pass after s steps. Now, we start with a bound on the deviations in the activations from X i at each row. Since, each row of A (l) i sums to 1, we have that h (l) i [t, :] -X i [t, :] ≤ h (l-1) i [t, :] -X i [t, :] + h (l-1) i [t, :] V (l) (159) and h (l) i [t, :] ≤ (1 + V (l) ) h (l-1) i [t, :] (160) Then, by the inductive hypothesis and sη ≤ 1 12L , we have that V (l) F ≤ 3 2 s 2 η 2 . Using this and that h (0) i = X i which has unit norm, we have that across all layers and rows h (l) i [t, :] ≤ 1 + 3 2 s 2 η 2 L (161) and as sη ≤ 1 12L and as (1 + c/L) L ≤ 1 + 2c for c ≤ 1, we have that h (l) i [t, :] ≤ 1 + sη 4 (162) Using this and again that h (0) i = X i , we have that for all rows and layers, h (l) i [t, :] -X i [t, :] ≤ L 1 + sη 4 3 2 s 2 η 2 ≤ 2s 2 η 2 L ≤ sη 6 (163) again using that sη ≤ 1 12L . Let A 0 be the uniform causal attention with the t-th row having the first t elements equal to 1/t and the remaining elements being 0. For each row, of A (l) i , we have that A (l) i = S(Mask(h (l-1) i [t, :]W (l) h (l-1)⊤ i + DM(P (l) )[t, :])) (164) Decomposing h (l-1) i [t, :] as X i [t, :] + (h (l-1) i [t, :] -X i [t, :]), we get by the inductive hypothesis and that max km | Qkm |, max m |∆ m | ≤ 1 as shown in the one-layer case that MASK(h (l-1) i [t, :]W (l) h (l-1)⊤ i + DM(P (l) )[t, :]) ≤ 6 s 4 + 4 s 3 η 4 √ t + (C W + C P )s 5 η 5 T + 2 h (l-1) i [t, :] -X i [t, :] 6 s 4 + 4 s 3 η 4 √ T + h (l-1) i [t, :] -X i [t, :] 2 6 s 4 + 4 s 3 η 4 √ T (165) By our earlier bounds, we have then MASK(h (l-1) i [t, :]W (l) h (l-1)⊤ i + DM(P (l) )[t, :]) ≤ s 4 η 4 √ t + 26s 5 η 5 T + sη 3 s 4 η 4 √ T + s 2 η 2 36 s 4 η 4 √ T (166) which we can upper bound by MASK(h (l-1) i [t, :]W (l) h (l-1)⊤ i + DM(P (l) )[t, :]) ≤ s 4 η 4 √ t + 21 2 s 3 η 3
Then, by Lemma D.2, we have
(A (l) i -A 0 )[t, :] ≤ s 4 η 4 + 21s 3 η 3 2 √ t ≤ 11s 3 η 3 ≤ 7s 2 η 2 √ T ≤ sη √ T(168)
Then, we also have that A
(l) i -A 0 F ≤ 7s 2 η 2 ≤ sη(169)
From the deviation bounds on the activations and the inductive control of W O ,
∥F θ (X i )∥ F = ∥h (L) i W O ∥ F ≤ ∥h (L) i ∥ F ∥W O ∥ ≤ (1 + sη 4 ) √ T (sη + 3s 2 η 2 ) ≤ 2sη √ T (170) Applying Lemma D.2 gives ∥S(F θ (X i )) -U O ∥ F ≤ 2sη T |V | .(171)
Then as in the one-layer case but using equation 171 and accounting for deviations in the hidden state from
X i , ∂L ∂W O + B F ≤ 4sη |V | + 4sη ≤ 5sη(172)
Then, after s + 1 steps, we have
W O -(s + 1)η B F ≤ 3s 2 η 2 + 5sη 2 ≤ 3(s + 1) 2 η 2 (173) From Lemma D.7, ∂L ∂V (l) = - 1 N T i h (l-1)⊤ i A (l)⊤ i R i W ⊤ O (174
)
Considering the deviation from each of the terms, we have
∂L ∂V (1) + sη Φ⊤ B⊤ ≤ 36s 2 η 2(175)
Then, we have that after s + 1 steps,
V (l) - s + 1 2 η 2 Φ⊤ B⊤ F ≤ 12s 3 η 3 + 36s 2 η 3 ≤ 12(s + 1) 3 η 3
Published as a conference paper at ICLR 2026 From Lemma D.7, ∂L ∂W
(l) = - 1 N T i h (l-1)⊤ i S (l) i h (l-1) i (176
) ∂L ∂P (l) = - 1 N T ein tjk, jk→t D, i S (l) i(177)
with S (l)
i = ein J (l) i , G (l) i V (l)⊤ h (l-1)⊤ i
. As in the one-layer bound, we can use the bound on the attention pattern to control J
(l) i . We have that for t ≥ 2 ∥J t,i -J t ∥ 2 ≤ 1 + 2 √ t ∥A i [t, :] -A 0 [t, :]∥ 2 + ∥A i [t, :] -A 0 [t, :]∥ 2 2 (178) Then, as we have that ∥A i [t, :] -A 0 [t, :]∥ 2 ≤ sη √ T (179) it follows that ∥J t,i -J t ∥ 2 ≤ 10sη √ T (180) Since J 1,i is always all zeros, we can ignore this term and for t ≥ 2, we have that as sη ≤ 5 8 √ T , ∥J t,i -J t ∥ 2 ≤ 10sη √ T (181) Now, we bound the deviation of G
(l) i from η(Y i -U O ) B⊤ . Starting from layer L, we have G (L) i -sη(Y i -U O ) B⊤ F ≤ 2sη T |V | (2sη) + √ T (3s 2 η 2 ) ≤ 4s 2 η 2 √ T (182
)
We will let the bound on the deviation at layer l be D G,l . Now, we consider the bound for each layer l,
G (l-1) i -sη(Y i -U O ) B⊤ F ≤ D G,l + 4s 2 η 2 G (l) i + 2 S (l) i (2 √ T )(2s 4 η 4 T )(183)
Since we also need the norm of S (l)
i to iterate through layers, we bound the norm of S
(l) i , S (l) i F ≤ J (l) i G (l) i V (l) h (l-1) i F ≤ 5 2 G (l) i ( 3 2 s 2 η 2 )(2 √ T ) ≤ 8s 2 η 2 G (l) i √ T ≤ 5sη G (l) i(184)
Using this upper bound back in the recurrence for D G,l , we have
G (l-1) i -sη(Y i -U O ) B⊤ F ≤ D G,l + 4s 2 η 2 G (l) i + 40s 5 η 5 T 3/2 G (l) i(185)
Using sη ≤ 5 8 √
T , we have
G (l-1) i -sη(Y i -U O ) B⊤ F ≤ D G,l + 14s 2 η 2 G (l) i(186)
We can now write a recurrence for G
(l) i as G (l) i ≤ sη (Y i -U O ) B⊤ + D G,l , we have G (l-1) i ≤ (1 + 14s 2 η 2 )( sη(Y i -U O ) B⊤ + D G,l ) ≤ (1 + 14s 2 η 2 )(sη √ 2T + D G,l ) (187
) Utilizing this with the recurrence for D G,l , we can then write a recurrence only in terms of D G,l and find that for all l, D G,l ≤ 12s 2 η 2 √ T as L ≤ √ T 4 and sη ≤ min 1 12L , 5 8 √ T . Then, we also have that for all l, G l i ≤ sη √ T + 12s 2 η 2 √ T ≤ 2sη √ T . Then, we have that as ∥J t ∥ 2 = 1 t , ∥J i ∥ 2 ≤ 3 2 + 10s 2 η 2 ≤ 2, and sη ≤ min 1 12L , 5 8 √ T , s 3 -s 2 2 η 3 Q i -S (l) i F ≤ 64s 4 η 4 T (188) This produces ∂L ∂W (l) + s 3 -s 2 2 η 3 Q F ≤ 64s 4 η 4 T (189) and similarly, ∂L ∂P (l) + s 3 -s 2 2 η 3 ∆ F ≤ 64s 4 η 4 T (190) and hence after (s + 1) steps W (l) -3 s + 1 4 + 2 s + 1 3 η 4 Q F ≤ 13(s + 1) 5 η 5 T (191) P (l) -3 s + 1 4 + 2 s + 1 3 η 4 ∆ F ≤ 13(s + 1) 5 η 5 T (192) Lemma D.10 (Gaussian Initialization Operator Norm). Under the setting described and with all parameters initialized from N (0, v 2 |V | 2+2ξ ) for ξ ≥ 0 and T ≤ |V |, we have that with probability at least 1 -(3L + 1) exp -|V | 1+2ξ 4
, for all 1 ≤ l ≤ L, 193) Proof. We start with W O . Using a concentration bound on Gaussian random matrices, we have that 194) Then, setting t = |V | 1/2+ξ , we have that
∥W O ∥ , V (l) , W (l) , P (l) ≤ 3v |V | 1/2(
P v|V | 1+ξ W O ≥ 2 |V | + t ≤ e -t 2 /2(
P ∥W O ∥ ≥ 3v |V | 1/2 ≤ exp - |V | 1+2ξ 2(195)
Then, with probability at least
1 -exp -|V | 1+2ξ 2 , ∥W O ∥ ≤ 3v |V | 1/2(196)
We can apply the same argument for each of V (l) , W (l) to derive the same bound. Since P (l) is smaller than the other matrices and has the same initialization, we can also apply the same bound. Applying a union bound on the probability of failures for each of the weights, we have that with probability at least 1
-(3L + 1) exp -|V | 1+2ξ 2 , all of W O , V (l) , W (l) , P (l) have operator norm at most 3v |V | 1/2 .
Lemma D.11 (Gaussian Initialization Frobenius Norm). Under the setting described and with all parameters initialized from N (0, v 2 |V | 2+2ξ ) for ξ ≥ 0 and T ≤ |V |, we have that with probability at least 1 -(3L + 1) exp -|V | 2+2ξ 4 , for all 1 ≤ l ≤ L,
∥W O ∥ F , V (l) F , W (l) F , P (l) F ≤ 2v(197)
Proof. We start with W O . Using Lemma 1 from Laurent & Massart (2000), we have that
P ∥W O ∥ 2 F ≥ v 2 |V | 2+2ξ (|V | 2 + 2|V | √ t + 2t) ≤ e -t(198)
Then, setting t = |V | 2+2ξ 4
, we have that
P ∥W O ∥ 2 F ≥ 3v 2 ≤ exp - |V | 2+2ξ 4(199)
Then, with probability at least
1 -exp -|V | 2+2ξ 4 , ∥W O ∥ F ≤ 2v(200)
We can apply the same argument for each of V (l) , W (l) to derive the same bound. For P (l) , we have
P P (1) 2 F ≥ v 2 |V | 2+2ξ (T + 2 √ T t + 2t) ≤ e -t(201)
Then, setting t = |V | 2+2ξ 4 and using that T ≤ |V |, we have that
P P (1) 2 F ≥ 3v 2 ≤ exp - |V | 2+2ξ 4(202)
Then, with probability at least
1 -exp -|V | 2+2ξ 4 , P (1) F ≤ 2v(203)
Applying a union bound on the probability of failures for each of the weights, we have that with probability at least 1 -(3L + 1) exp -|V | 2+2ξ 4 , all of W O , V (l) , W (l) , P (l) have Frobenius norm at most 2v.
Theorem D.12 (Gaussian Initialization (Multi-Layer)). Assume the setting of D.9 with depth L ≤ √ T 4 , all parameters initialized i.i.d. from N 0, v 2 |V | 2+2ξ with v ≤ η 2 T 2 , T ≤ |V |, and learning rate η ≥ T -1 . Then, with probability at least 1 -(3L + 1) exp -|V | 1+2ξ 2 + exp -|V | 2+2ξ 4 for s ≤ η -1 min 1 12L , 5 8 √ T
with T ≥ 60 and |V | ≥ 500, after s gradient descent steps with learning rate η we have, uniformly for every layer 1 ≤ l ≤ L,
W O -sη B F ≤ 3s 2 η 2(204)
V (l) - s 2 η 2 Φ⊤ B⊤ F ≤ 12s 3 η 3(205)
W (l) -3 s 4 + 2 s 3 η 4 Q F ≤ 13s 5 η 5 T (206
)
P (l) -3 s 4 + 2 s 3 η 4 ∆ F ≤ 13s 5 η 5 T (207
)
where B, Φ, Q, and ∆ are as in the one-layer analysis (row-centered bigram matrix, centered prefix-statistics operator, and the third-step structures, respectively).
Proof. We start by noting that as long the proof holds when v = η 2 T 2 and we show that the first gradient step satisfies the inductive hypothesis used in Theorem D.9, then the proof will be complete. We will prove that the first gradient step satisfies the inductive hypothesis with v = η 2 T 2 .
Published as a conference paper at ICLR 2026
We will condition on the event that the results of Lemmas D.10 and D.11 holds. Then, our results will hold with probability at least
1 -(3L + 1) exp - |V | 1+2ξ 2 + exp - |V | 2+2ξ4
and we have that at initialization for all 1 ≤ l ≤ L
∥W O ∥ , V (l) , W (l) , P (l) ≤ 3η 2 T 2 |V | 1/2(208)
and
∥W O ∥ F , V (l) F , W (l) F , P (l) F ≤ 2η 2 T 2(209)
Now, we start with a bound on the deviations in the activations from X i at each row. Since, each row of A (l) i sums to 1, we have that h (l) i [t, :] -X i [t, :] ≤ h (l-1) i [t, :] -X i [t, :] + h (l-1) i [t, :] V (l) (210) and h (l) i [t, :] ≤ (1 + V (l) ) h (l-1) i [t, :] (211) Using that V (l) ≤ 3η 2 T 2 |V | 1/2 and that h (0) i = X i which has unit norm, we have that across all layers and rows h (l) i [t, :] ≤ 1 + 3η 2 T 2 |V | 1/2 L (212) and as η ≤ 1 12L and as (1 + c/L) L ≤ 1 + 2c for c ≤ 1, we have that h (l) i [t, :] ≤ 1 + η 7/2 2 (213) Using this and again that h (0) i = X i , we have that for all rows and layers, h
i [t, :] -X i [t, :] ≤ L 1 + η 7/2 2 3η 2 T 2 |V | 1/2 ≤ η 7/2 3(l)
again using that sη ≤ 1 12L . Let A 0 be the uniform causal attention with the t-th row having the first t elements equal to 1/t and the remaining elements being 0. For each row, of A (l) i , we have that
A (l) i = S(Mask(h (l-1) i [t, :]W (l) h (l-1)⊤ i + DM(P (l) )[t, :]))(215)
By our earlier bounds, we have then MASK(h
(l-1) i [t, :]W (l) h (l-1)⊤ i + DM(P (l) )[t, :]) ≤ 1 + η 7/2 2 2 3η 2 T 2 |V | 1/2 √ T + 3η 2 T 2 |V | 1/2 ≤ 6η 7/2 √ T (216
)
where we have use 1 T ≤ η and η ≤ 1 12L . Then, by Lemma D.2, we have
(A (l) i -A 0 )[t, :] ≤ 6η 7/2 √ T t(217)
Then, we also have that
A (l) i -A 0 F ≤ 6η 7/2 √ T 1 + log T ≤ 6η 7/2(218)
From the deviation bounds on the activations and the initial bound on W O ,
∥F θ (X i )∥ F = ∥h (L) i W O ∥ F ≤ ∥h (L) i ∥ F ∥W O ∥ ≤ (1 + η 7/2 2 ) √ T 3η 2 T 2 |V | 1/2 ≤ 4η 4(219)
Applying Lemma D.2 gives
∥S(F θ (X i )) -U O ∥ F ≤ 4η 4 |V |(220)
Then following the argument in the zero-initialization case,
∂L ∂W O + B F ≤ 4η 4 |V | + η 5/2 √ T ≤ 2η 3(221)
Then, after the first step,
W O -η B F ≤ 3η 2 T 2 |V | 1/2 + 2η 4 ≤ 3η 4 ≤ 3η 2 (222) From Lemma D.7, ∂L ∂V (l) = - 1 N T i h (l-1)⊤ i A (l)⊤ i R i W ⊤ O(223)
Considering the deviation from each of the terms, we have
∂L ∂V (1) F ≤ 15η 2 T 2 |V | 1/2 ≤ 15η 9/2(224)
Then, we have that after the first step
V (l) F ≤ 2η 2 T 2 + 15η 11/2 ≤ 3η 4 ≤ 12η 3 From Lemma D.7, ∂L ∂W (l) = - 1 N T i h (l-1)⊤ i S (l) i h (l-1) i (225
) ∂L ∂P (l) = - 1 N T ein tjk,jk→t D, i S (l) i(226) with S (l)
i = ein J (l) i , G(l)
i V (l)⊤ h (l-1)⊤ i
. As in the zero-initialization case, we can use the bound on the attention pattern to control J (l) i . We have that for t ≥ 2
∥J t,i -J t ∥ 2 ≤ 1 + 2 √ t ∥A i [t, :] -A 0 [t, :]∥ 2 + ∥A i [t, :] -A 0 [t, :]∥ 2 2(227)
Then, as we have that
∥A i [t, :] -A 0 [t, :]∥ 2 ≤ 6η 7/2 √ T (228
)
it follows that
∥J t,i -J t ∥ 2 ≤ 15η 7/2 √ T (229
)
Since J 1,i is always all zeros, we can ignore this term and for t ≥ 2, we have that,
∥J i -J∥ 2 ≤ 15η 7/2(230)
Now, we bound the norm of G
i . Starting from layer L, we have
G (L) i F ≤ √ 2T 3η 2 T 2 |V | 1/2 ≤ 5η 4(231)
Published as a conference paper at ICLR 2026
We will let the bound on the deviation at layer l be D G,l . Now, we consider the bound for each layer l,
G (l-1) i F ≤ D G,l + 5 2 D G,l 3η 2 T 2 |V | 1/2 + 2 S (l) i √ 2T 3η 2 T 2 |V | 1/2 ≤ (1 + 8η 9/2 )D G,l + 9η 4 S (l) i
(232) Since we also need the norm of S (l) i to iterate through layers, we bound the norm of
S (l) i , S (l) i F ≤ J (l) i G (l) i V (l) h (l-1) i F ≤ 5 2 D G,l 3η 2 T 2 |V | 1/2 √ 2T ≤ 8η 4 D G,l(233)
Using this upper bound back in the recurrence for D G,l , we have
G (l-1) i F ≤ (1 + 8η 9/2 + 72η 8 )D G,l(234)
Then for all l, D G,l ≤ 6η 4 as L ≤
√ T 4 and η ≤ min 1 12L , 5 8 √ T . Then, we also have that for all l, G l i ≤ 6η 4 . Then, we have that as ∥J t ∥ 2 = 1 t , ∥J i ∥ 2 ≤ 3 2 + 15η 7/2 ≤ 2, and sη ≤ min 1 12L , 5 8 √ T , S (l) i F ≤ 48η 8 (235) This produces ∂L ∂W (l) F ≤ 48η 8 (236) and similarly, ∂L ∂P (l) F ≤ 48η 8 (237) and hence after the first step W (l) F ≤ 48η 9 + 3η 2 T 2 ≤ 4η 5 T P (l) F ≤ 48η 9 + 3η 2 T 2 ≤ 4η 5 T
this section cite: ['b24']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Highdimensional asymptotics of feature learning: How one gradient step improves the representation Year: (2022)
Ref_id:b2 Title: Pythia: A suite for analyzing large language models across training and scaling Year: (2023)
Ref_id:b3 Title: Learning single-index models with shallow neural networks Year: (2022)
Ref_id:b4 Title: Birth of a transformer: A memory viewpoint Year: (2023)
Ref_id:b5 Title: Language models are few-shot learners Year: (2020)
Ref_id:b6 Title: A phase transition between positional and semantic learning in a solvable model of dot-product attention Year: (2024)
Ref_id:b7 Title: Sparse autoencoders find highly interpretable features in language models Year: (2023)
Ref_id:b8 Title: The computational complexity of learning gaussian single-index models Year: ()
Ref_id:b9 Title: How two-layer neural networks learn, one (giant) step at a time Year: (2023)
Ref_id:b10 Title: The evolution of statistical induction heads: In-context learning markov chains Year: (2024)
Ref_id:b11 Title: Tinystories: How small can language models be and still speak coherent english Year: (2023)
Ref_id:b12 Title: A mathematical framework for transformer circuits Year: (2021)
Ref_id:b13 Title: Not all language model features are one-dimensionally linear Year: (2024)
Ref_id:b14 Title: Papers in Linguistics 1934-1951 Year: (1957)
Ref_id:b15 Title: Hidden dynamics of massive activations in transformer training Year: (2025)
Ref_id:b16 Title:  Year: (2019)
Ref_id:b17 Title: The llama 3 herd of models Year: (2024)
Ref_id:b18 Title:  Year: (1954)
Ref_id:b19 Title:  Year: (1984)
Ref_id:b20 Title: Non-asymptotic convergence of training transformers for next-token prediction Year: (2025)
Ref_id:b21 Title: On the origins of linear representations in large language models Year: (2024)
Ref_id:b22 Title: Transformers learn nonlinear features in context: nonconvex mean-field dynamics on the attention landscape Year: (2024)
Ref_id:b23 Title: Transformers provably solve parity efficiently with chain of thought Year: (2025)
Ref_id:b24 Title: Adaptive estimation of a quadratic functional by model selection Year: (2000)
Ref_id:b25 Title: Neural network learns low-dimensional polynomials with sgd near the information-theoretic limit Year: (2024)
Ref_id:b26 Title: Inference-time intervention: Eliciting truthful answers from a language model Year: (2023)
Ref_id:b27 Title: How do transformers learn topic structure: Towards a mechanistic understanding Year: (2023-07)
Ref_id:b28 Title: Sparse feature circuits: Discovering and editing interpretable causal graphs in language models Year: (2024)
Ref_id:b29 Title: Mass-editing memory in a transformer Year: (2022)
Ref_id:b30 Title: Contextual correlates of semantic similarity Year: (1991)
Ref_id:b31 Title: Gradient-based feature learning under structured data Year: (2023)
Ref_id:b32 Title: Progress measures for grokking via mechanistic interpretability Year: (2023)
Ref_id:b33 Title: How transformers learn causal structure with gradient descent Year: (2024)
Ref_id:b34 Title: -context learning and induction heads Year: (2022)
Ref_id:b35 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: (2024)
Ref_id:b36 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b37 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b38 Title: Scan and snap: Understanding training dynamics and token composition in 1-layer transformer Year: (2023)
Ref_id:b39 Title: Fundamental limits of learning in sequence multi-index models and deep attention networks Year: (2025)
Ref_id:b40 Title: How transformers implement induction heads: Approximation and optimization analysis Year: (2024)
Ref_id:b41 Title: Attention-only transformers via unrolled subspace denoising Year: (2025)
Ref_id:b42 Title: Nonlinear spiked covariance matrices and signal propagation in deep neural networks Year: (2024)
Ref_id:b43 Title: Learning semantic structure-preserved embeddings for cross-modal retrieval Year: (2018)
Ref_id:b44 Title: Qwen3 technical report Year: (2025)
Ref_id:b45 Title: Training dynamics of transformers to recognize word co-occurrence via gradient flow analysis Year: (2024)
Ref_id:b46 Title: An analysis for reasoning bias of language models with small initialization Year: (2025)
