Title: Memory Mosaics at scale
Abstract: Memory Mosaics [Zhang et al., 2025], networks of associative memories, have demonstrated appealing compositional and in-context learning capabilities on medium-scale networks (GPT-2 scale) and synthetic small datasets. This work shows that these favorable properties remain when we scale memory mosaics to large language model sizes (llama-8B scale) and real-world datasets. To this end, we scale memory mosaics to 10B size, we train them on one trillion tokens, we introduce a couple architectural modifications ("memory mosaics v2"), we assess their capabilities across three evaluation dimensions: training-knowledge storage, new-knowledge storage, and in-context learning. Throughout the evaluation, memory mosaics v2 match transformers on the learning of training knowledge (first dimension) and significantly outperforms transformers on carrying out new tasks at inference time (second and third dimensions). These improvements cannot be easily replicated by simply increasing the training data for transformers. A memory mosaics v2 trained on one trillion tokens still perform better on these tasks than a transformer trained on eight trillion tokens. Memory Mosaics v2Transformers show an interesting induction head mechanism [Olsson et al., 2022], that is, predict b after sequence [. . . , a, b, . . . , a]. This mechanism contributes to their in-context learning ability. According to Bietti et al. [2023], position encoding and asymmetric query-key extractors (e.g. 3 both keys and values shall be vectors in R d . 4 This Gaussian kernel smoothing not only converges to the true conditional expectation E(K|V ) when n → ∞ and β = √ n [Nadaraya, 1964, Watson, 1964], but also makes it easy to compute gradients. 5 For the easy of reading, we reuse the "attention score" notion toin associative memories.

Section: Introduction
In Machine Learning, compositional capabilities and in-context/out-of-distribution learning capabilities have been continuously pursued but remain challenging. Early attempts to achieve these goals include pursuing disentanglement via various statistical "independence" [Comon, 1994, Roth et al., 2022], pursuing out-of-distribution/learning from the perspective of optimization on multiple environments [Finn et al., 2017, Arjovsky et al., 2019, Bengio et al., 2019]. In contrast, transformerbased models demonstrate certain compositional capabilities and early in-context learning abilities. However, we still lack a clear understanding of how current transformers achieve these capabilities, and why earlier models were unable to. Memory mosaics [Zhang et al., 2025], networks of simple key-value associative memories (without position encoding), offer a comparatively transparent way to understand how composition or disentanglement occur. Trained and evaluated on medium-scale networks and synthetic datasets, memory mosaics reveal promising superior in-context learning abilities. Therefore, we ask "To peruse a strong and general new task learning capability, how can we scale memory mosaics to large networks and real-world datasets?"
The contribution of this work: 1) We successfully scale up memory mosaics to llama-8B scale, using one trillion real-world training tokens. The resulting network is named as Memory Mosaics v2.foot_0 foot_1 Compared to memory mosaics, memory mosaics v2 made three architectural modifications, including an adaptive bandwidth of associative memory, a gated time-variant key feature extractor, and a 3-level memory design. 2) We propose three evaluation dimensions to comprehensively assess model ability (from i.i.d. to o.o.d. scenarios). 3) Our memory mosaics v2 demonstrate superior new-task learning capabilities (with fewer examples and less priori knowledge from human designers). This paper is organized as follows. Section 2 introduces background knowledge on associative memories. Section 3 presents the architecture of memory mosaics v2. Then section 4 describes the training process, section 5 evaluates memory mosaics v2 and transformers across three dimensionstraining (persistent) knowledge storage, new knowledge storage, and in-context learning. Section 6 discusses the failure of replicating memory mosaics v2 by simply increasing the training data (×8 more data) for transformers. Section 7 studies the advantage of memory mosaics v2 in fine-tuning. Finally, section 8 provides discussion and further directions.
this section cite: ['b10', 'b27', 'b12', 'b0', 'b4', 'b35']

Section: Background on Associative Memory
General speaking, memory mosaics architecture [Zhang et al., 2025] replaces attention blocks in transformers [Vaswani et al., 2017] with associative memories. This section provides the background on associative memories, highlights the connection and differences between "associative memory" in memory mosaics and "attention" in transformers.
Associative Memory Associative memories have a long history in both psychology and computer science, referring to relationships between unrelated items. In this work, we follow the definition from Zhang et al. [2025], according to which an associative memory is a device that can store key-value pairs {(k 1 , v 1 ) . . . (k n , v n )} and retrieve values given a corresponding key:
3 k → f (k; {(k 1 , v 1 ) . . . (k n , v n )})(1)
The key-value pairs are stored in a set, and thus can be assumed to be permutation invariant. This exchangeability property suggests that we can view an associative memory as a device that estimates a conditional probability distribution P (V |K) on the basis of the sample (k 1 , v 1 ) . . . (k n , v n ) of keyvalue pairs. The retrieval function is then a conditional expectation over this estimated distribution:
f k; {(k 1 , v 1 ) . . . (k n , v n )} = E(V | K = k) .(2)
This conditional expectation can be estimated by kernel regression, e.g. Gaussian kernel regression: 4
f k; {(k 1 , v 1 ) . . . (k n , v n )} = n i=1 e -β∥k-ki∥ 2 n i=1 e -β∥k-ki∥ 2 v i ,(3)
where β controls the bandwidth of Guassian kernel.
this section cite: ['b35', 'b31', 'b35']

Section: Connection between associative memory and attention
Associative memory in Equation 3 is closely connected to attention [Bahdanau et al., 2015] when all key vectors k i share the same squared norm. That is, expression (3) becomes: 5
f k; {(k 1 , v 1 ) . . . (k n , v n )} = n i=1 e β k ⊤ ki n j=1 e β k ⊤ kj v i .(4)
Moreover, the size of associative memory (i.e., the number of key-value examples) is analogous to the sequence length in attention.
this section cite: ['b1']

Section: Differences between associative memory and attention
The associative memory viewpoint is conceptually simple and transparent. This simplicity contributes several key differences in associative memory (compared with attention), including: 1) L 2 normalized key vectors with an explicit bandwidth parameter β, 2) a symmetric kernel with the same formula for keys as queries, and 3) the absence of explicit position encoding. These differences further contribute to the superior compositional capabilities and in-context learning capabilities in memory mosaics.
q T = W q x T , k T = W k x T ) are essential for transformers to achieve this induction head mechanism with at least two layers of attention. Inspired by studies of induction head mechanism, Memory Mosaics [Zhang et al., 2025] construct associative memories using keys to represent the recent past and values to represent the near future (Figure 2 left): k T = φ θ (x T , x T -1 , . . . ), v T = ψ θ (x T +1 , x T , . . . )
This simple designer allows memory mosaics to get ride of explicit position encoding, use the same key as query, perform induction head with only one layer. The resulting memory mosaics also reveal appealing in-context learning capabilities on small synthetic datasets.
Based on memory mosaics, this section introduces Memory Mosaics v2, aiming at a stronger and more general in-context learning ability on broader real-world tasks (without loss of performance on other common benchmarks). Compared to memory mosaics, memory mosaics v2 incorporates three architecture modifications, including an adaptive bandwidth in associative memory, a gated time-variant key feature extractor, and a 3-level memory design.
this section cite: ['b35']

Section: Adaptive bandwidth in Gaussian kernel smoothing
Memory mosaics use one fixed bandwidth parameter β for different sizes n of associative memory (Equation 1). It is well known that bandwidth controls the bias-variance trade-off [Hastie et al., 2009] of kernel regression (memory-based) methods. That is, for a given distribution, the optimal bandwidth depends on the number of examples (key-value pairs in associative memory). Inspired by the asymptotic Mean Integrated Squared Error kernel bandwidth estimation approach where [García-Portugués, 2024], memory mosaics v2 schedule β in Equation 4 as:
1/ √ β ∝ n -1/(p+4)
β = β 1 n α + β 0 ,(6)
where β 0 ≥ 0, β 1 > 0, 1 > α > 0 are learnable parameters (Check Appendix Table 6 for reparameterization and initialization details). I.e., the more key-value pairs (examples), the smaller bandwidth 1/ √ β.
this section cite: ['b17', 'b13']

Section: Gated time-variant key feature extractor
Memory mosaics employs a simple time-invariant leaky averaging to extract key features:
k T = Norm kT with kT = kT + λ kT -1 kT = W φ x T (7
)
The averaging weights in Equation 7 are fixed and independent of the semantic input x. As a result, semantically similar cases, such as "tom-and-jerry" and "tom---and---jerry", may receive different key features. Inspired by recurrent-style networks [Peng et al., 2023, Gu and Dao, 2023, Beck et al., 2025], memory mosaics v2 utilize the following gated time-variant key feature extractor:foot_3
k T = Norm kT with kT = g T kT + λ T kT -1 kT = W φ x T g t = e Wgx T ∈ R , λ T = e -|W λ x T | ∈ R ,(8)
where W φ , W g , W λ are learnable parameters, the averaging weights λ T ∈ R and the exponential gate g T ∈ R semantically depend on input x T . See Appendix Figure 9 for graphical illustrations.
For key feature extractor, memory mosaics v2 reuses the same convolutional key extractor as in memory mosaics:
v T = α ψ Norm vT with vT = γ ṽT + (1 -γ) ṽT +1 ṽT = W ψ x T ,(9)
where γ, α ψ ∈ R and W ψ are learnable parameters.
this section cite: ['b25', 'b15', 'b2']

Section: 3-level memory
Transformer architecture [Vaswani et al., 2017] consists of attention blocks and feedforward neural network blocks. The former handles local contextual information from an input sequence, while the latter stores global persistent information shared by different training sequences. Memory mosaics [Zhang et al., 2025] simplify the attention and the feedforward network in transformer as contextual associative memory and persistent memory, respectively. This simplification reduces the dependence between the "attention score" and the token position, as shown in Figure 1. Compared with transform- ers (Figure 1 left), the attention scores in memory mosaics (Figure 1 right) exhibit a structured pattern.
That is, attention scores on near-tokens (positions) heavily depend on positions, while attention scores on far-tokens are almost invariant to token positions. Inspired by this experimental discovery, memory mosaics v2 replace each contextual associative memory in memory mosaics with two associative memories, short-term memory and long-term memory, using distinct parameters (as in Figure 2).
this section cite: ['b31', 'b35']

Section: Short-term memory
The short-term memory at position t only stores key-value pairs of neartokens, ranging from t -h + 1 to t -1, implementing Eq. ( 4) as:
f k; {(k t-h+1 , v t-h+1 ) . . . (k t-1 , v t-1 )} = t-1 i=t-h+1 e β k ⊤ ki t-1 j=t-h+1 e β k ⊤ kj v i .(10)
Long-term memory In contrast, the long-term memory skips near tokens and only stores key-value pairs before position t -m, implementing Eq. ( 4) as f (k; {(k 1 , v 1 ) . . . (k t-m , v t-m )}). 7 By setting m < h, memory mosaics v2 create an overlap between long-term and short-term memory, resulting in a soft boundary between these two memories. Eventually, the outputs of many long-term memories and short-term memories are concatenated together, following by a linear projection W o .
Persistent Memory Memory mosaics v2 implements persistent memory using dense two-layer neural networks with SwiGLU activation [Shazeer, 2020] due to computational efficiency concerns.foot_6
Associative memory
this section cite: ['b28']

Section: Feature Extractor
Output time series tracks ( ) without looking forward into ( ) but using memory matches.
this section cite: []

Section: Associative Memory retrieves past key-value pairs by current key
Value time series Represents the near future of the input time series.
this section cite: []

Section: Key time series
Represents the recent past of the input time series.
Input time series normalize long-term memories Persistent memories Decoding layer Nb repeated blocks normalize short-term memories t-m 1 t t-h 1 t t-1 long-term memories short-term memories Attention Mask ignore near key-value pairs focus on near key-value pairs Implementations store persistent knowledges of training data Persistent memories • associative memories, or • dense neural networks, or • mix-of-expert sparse networks
this section cite: []

Section: Training
We train two memory mosaics v2 of difference sizes (small/large). Memory mosaics v2 small (llama-1.5B scale) contains 24 layers, 2048 hidden dimensions, and 16 heads, trained on 200 billion tokens of a diverse datamix. Memory mosaics v2 large (llama-8B scale) increases the number of layers to 32, hidden dimensions to 4096, and the number of heads to 32, trained on 1 trillion tokens of the same datamix. Both models are trained on 4,096 context length, followed by a fine-tuning process on 32,768 context length. Other training details are provided in Appendix C.
this section cite: []

Section: Stochastic long-term memory size
During training, memory mosaics v2 samples the long-term memory delay step m from [64,256], sets the short-term memory window size h = 256. At inference, m is set to 64. This stochastic long-term memory training setup encourages the allocation of positioninvariant signals to long-term memory and position-dependent signals to short-term memory (as shown in Figure 1). The experimental results in Appendix G Table 12 show that this training setup enhances context-length extrapolation ability by more than 15%. Baseline We train two baseline transformers (small/large) with the same configurations as their memory mosaics v2 counterparts. Unless otherwise specified, in this work, transformer models use llama architecture [Grattafiori et al., 2024] with multi-head attention.
this section cite: ['b14']

Section: Three evaluation dimensions
The evaluation design provides a means to assess the specific properties of a system. Memory mosaics v2 aims at the ability to learn new tasks with fewer examples and less task-specific priori knowledge [Zhang, 2025]. Thus, to fully assess this capability, this section adopts three evaluation dimensions.
• Persistent-knowledge storage and retrieval, the ability of persistent-memory to store and retrieve knowledge of training dataset. This capability prepares knowledge that could be reused in other tasks during inference. We use common language benchmarks to access this aspect.
• New-knowledge storage and retrieval, the ability to store and retrieve new information of test dataset. It is a prerequisite for "learning" new tasks via memory-based methods. We employ "multi-unrelated-documents storing and question-answering" tasks to evaluate this aspect.
• In-context Learning, directly evaluates the ability to learn new tasks with fewer examples and less task-specific priori knowledge. We use multiclass classification to assess this aspect.
this section cite: ['b34']

Section: Persistent-knowledge storage and retrieval
Table 1 evaluates both memory mosaics v2 and baseline transformers on 19 commonly used language benchmarks, showing that they perform closely on these benchmarks.This is expected since both models share the same persistent memory architecture. How do we know whether these benchmarks access persistent-knowledge ability rather than newknowledge ability? To answer this question, we re-evaluate these benchmarks on memory mosaics v2 but with long-term memory being removed after training. The underlying reason is that if a task solely relies on the information stored in persistent memory and retrieved by short-term memory, removing long-term memory should not significantly affect performance.
Table 2 shows that removing long-term memory after training does not degrade the performance of 13 common benchmarks. This suggests that these 13 tasks are almost exclusively based on information stored in persistent memory and retrieved by short-term memory. In contrast, Appendix Table 9 indicates that the other 6 benchmarks perform poorly when long-term memory is removed.
Based on these findings, we use the 13 tasks to evaluate persistent knowledge storage and retrieval capability. The results (Table 1) show that memory mosaics v2 and transformers perform similarly in this evaluation dimension, suggesting that both models are capable of effectively storing and retrieving persistent knowledge. Computation and # parameters concerns Table 2 summarizes the size of parameters and computation required for transformers and memory mosaics v2. Interestingly, removing long-term memory from memory mosaics v2 after training achieves a comparable transformer performance on the 13 persistent-knowledge benchmarks, while using fewer parameters and computations.
this section cite: []

Section: New-knowledge storage and retrieval
The new-knowledge storage and retrieval ability is a prerequisite for learning new tasks via memorybased methods (e.g., Gaussian kernel regression), because the data of new tasks must be adequately "stored" before learning (Note that memory-based methods are lazy methods). To illustrate this point, consider a poor goldfish with 7-second memory -how can it possibly learn a 90-minute movie? Similarly, a model with limited new-knowledge storage ability will struggle to learn information that exceeds its storage (memory) capacity.
Task description To assess this ability, we employ two "multi-unrelated-documents questionanswering" tasks from the RULER benchmark [Hsieh et al., 2024]. These tasks involve multiple concatenated realistic articles followed by a question related to one of these articles, requiring the model to find the correct answer based on the correct article. 9 A prompt example is:
Answer the question based on the given documents. The following are given documents. Document 1: [...] Document2: [...] [...] Document 20: [...] Question: What religion were the Normans? Answer:
These tasks are notably more challenging than typical 'needle-in-a-haystack' benchmarks [Kamradt, 2023], owing to their high information entropy. The typical 'needle-in-a-haystack' task is too easy, resulting in many models achieving near-perfect performance. See Table 13 in Appendix for details.
this section cite: ['b18', 'b19']

Section: Main results
Table 3 compares memory mosaics v2 and transformers, pretrained on a 4k context length, on these question-answer tasks. Memory mosaics v2 outperforms transformers on 4k tasklength by 1.4%∼5.6%. Similarly, Table 4 presents the same comparison, but with both models fine-tuned at a 32k context length. As task lengths increase to 32k, the "multi-unrelated-documents question-answering" tasks become more challenging. At this increased difficulty level, memory mosaics v2 significantly outperforms transformers by 12.3% to 14.8%. The failures of many potential baselines Many memory compression algorithms, such as RNNs, xLSTM [Beck et al., 2025], rwkv [Peng et al., 2023], and state-space models [Gu and Dao, 2023], fail on this task by construction because they cannot store all articles before reading the question. Similarly, local-window memory approaches, such as Alibi position encoding Press et al. [2021] and sliding-window attention Beltagy et al. [2020], also struggle for the same reason. 10 This incompetent of memory compression algorithms has also been experimentally demonstrated by Hsieh et al. [2024] and Li et al. [2024]. Also, see Appendix G for these experimental evidences.
Extrapolating context length (without fine-tuning) Context length extrapolation (without finetuning) not only is computationally appealing, but also reveals the model's consistency in handling context. Unfortunately, transformers (with ROPE position encoding) struggle to extrapolate context length, as shown in Table 3. 11 In contrast, memory mosaics v2, trained on 4k context length, not only outperform transformers on 4k length, but also perform well after extrapolating context length ×4 ∼ ×8 times without any fine-tuning or adaptation.
this section cite: ['b2', 'b25', 'b15', 'b26', 'b3', 'b18', 'b20']

Section: In-context learning
Having demonstrated the new-knowledge storage and retrieval ability of memory mosaics v2, this section takes a step further to evaluate its capacity to learn new tasks or distributions at inference time. This ability is also commonly referred to as in-context learning.
Tasks description To assess the in-context learning ability, we employ classic multiclass classification problems, 12 adopted from Li et al. [2024]. The classification tasks include:
• Banking77 [Casanueva et al., 2020] is a banking-intent classification task with 77 target categories.
Each example has an average length of 24 tokens.
• Tacred [Zhang et al., 2017] is a relation classification task of two objects in a sentence, extracted from newswire or webtext, with 41 target categories. Each example has an average length of 77 tokens.
• Goemotion [Demszky et al., 2020] is an emotion classification task of Reddit comments with 28 target categories. Each example has an average length of 26 tokens.
To solely evaluate the ability to learn new tasks (reduce the influence of training knowledge), we create an anonymous version with anonymous target labels (e.g. "class 1", "class 2") for each classification task. The original classification setup with semantic labels (e.g. "happy", "angry") is referred to as semantic version.
In this section, we adopt a few-shot learning setup where each "shot" consists of one (x, y) example from each possible target label category. By collecting multiple shots, we create an n-shot classification task. To encode these (x, y) examples for memory mosaics v2 and transformers, we serialize the (x, y) pairs into a sequence followed by a test query x test . 13 A prompt example is: Main Results Figure 3 compares the performance of memory mosaics v2 and transformers in three classification tasks with semantic target labels. The horizontal axis represents the number of shots, while the vertical axis represents the classification accuracy on x test . We can observe two phenomena: 1) memory mosaics v2 consistently improve classification performance as it sees more demonstration shots (blue curves). In contrast, transformers struggle to maintain their performance and exhibit counterintuitively degraded performance as more demonstrations are provided (red curves). 2) Memory mosaics v2 significantly outperform transformers by more than 10%. Appendix H provides machine is intelligent, it often proves that human designers are intelligent. Please recall that a child does not prepare all questions before going to school. 11 The comparison ignores many memory compression and local window approaches [Press et al., 2021, Beltagy et al., 2020], because they fail on this evaluation by construction. 12 We choose classic classification problems over other fancy benchmarks for two reasons. Firstly, the mechanisms underlying classification are well-studied, allowing us to confidently attribute good or poor performance to the system's properties. Secondly, classification tasks can be designed to be arbitrarily different from the training set by changing the classification boundary, making it easier to measure the ability to learn new distributions. In contrast, as of this writing, many fancy benchmarks may not offer the same level of control and fine-grained analysis. 13 Transformers are known to be sensitive to the prompt strategies [Gupta et al., 2024, Mirzadeh et al., 2024], such as the delimiter before x and y, shuffling/not-shuffling the (x, y) examples within each shot. To reduce the influence of prompt strategies, we evaluate each classification task with different delimiters ("[space]" and "\n"), shuffled/non-shuffled (x, y) examples. Then choose the best prompt strategy for each n-shot classification task. Check appendix I for prompt examples.
a similar comparison on a smaller model size (∼1.5B), with an even larger margin. Appendix E further summarizes the comparison under matched model size or computation (FLOPs). Figure 4 presents a similar comparison as Figure 3, but on anonymous target labels. Again, memory mosaics v2 significantly outperforms transformers on all classification tasks.
2 4 6 8 10 12 14 16 number of shots 0.725 0.750 0.775 0.800 0.825 0.850 0.875 0.900 Accuary banking77 with semantic label 2 3 4 5 6 7 8 9 10 number of shots 0.50 0.52 0.54 0.56 0.58 0.60 0.62 Accuary 10.2% tacred with semantic label 2 4 6 8 10 12 14 16 number of shots 0.26 0.27 0.28 0.29 0.30 0.31 Accuary goemotion with semantic label Transformer large Memory Mosaics v2 large Figure 3: Semantic label in-context learning comparison between memory mosaics v2 and transformer. Memory mosaics v2 significantly outperform transformers on in-context learning with a large margin (more than 10%). Meanwhile, memory mosaics v2 benefits from more demonstration shots (x-axis), unlike transformers. 2 4 6 8 10 12 14 16 number of shots 0.60 0.65 0.70 0.75 0.80 0.85 0.90 Accuary banking77 with anonymous label 2 3 4 5 6 7 8 9 10 number of shots 0.15 0.20 0.25 0.30 0.35 0.40 0.45 Accuary 15.5% tacred with anonymous label 2 4 6 8 10 12 14 16 number of shots 0.06 0.07 0.08 0.09 0.10 0.11 Accuary goemotion with anonymous label Transformer large Memory Mosaics v2 large Figure 4: Anonymous label in-context learning comparison between memory mosaics v2 and transformers. Memory mosaics v2 significantly outperform transformers on all classification tasks. In summary, the experiments demonstrate that memory mosaics v2 not only outperform transformer by a significant margin (more than 10%) on in-context learning, but also consistently improve performance as more demonstrations are provided. These results highlight the superior in-context learning ability of Memory Mosaics v2. 2 4 6 8 10 number of shots 0.25 0.30 0.35 0.40 0.45 Accuary tacred with semantic label 2 4 6 8 10 number of shots 0.05 0.10 0.15 0.20 0.25 0.30 Accuary tacred with anonymous label Transformer small Transformer small + long-short term attention Memory Mosaics v2 small Augment transformer with long-short term attention Memory mosaics (v2) contains several unique components that are not applicable to transformers, such as the symmetric key and query, and the adaptive bandwidth. One seemingly applicable component for transformers is the separation of long-term and short-term memories introduced in Section 3.3. However, Figure 5 shows that augmenting a transformer with long-short-term attention does not help it overcome the limitations of in-context learning. These phenomena imply that memory mosaics (v2) is not simply a transformer variation but represents a different architecture.
Computation and Parameter Concerns On the last two evaluation dimensions (new knowledge storage and retrieval, and in-context learning), memory mosaics v2 outperform transformers by more than 10% with slightly more parameters. This 10% advantage holds even when comparing under the same number of parameters or the same computational budget. See Appendix Figure 12 for details.
this section cite: ['b20', 'b7', 'b36', 'b11', 'b26', 'b3', 'b16', 'b22']

Section: Risk-return trade-off of frontier-model-sized memory mosaics v2
Having demonstrated the superior new tasks learning ability of memory mosaics v2 up to 9.9 billion parameters and 1 trillion training tokens, this section analyzes the "risk-return trade-off" to further scale memory mosaics v2 to the size of the frontier model, unveiling potential benefits and challenges.
Two Approaches To train a large frontier foundational model, one can either:
1) take a low-risk-low-return approach by investing more resources (GPUs and data) and reusing old recipes (e.g. architecture), or 2) take a middle-risk-high-return approach by trying new smart techniques.
Taking the first approach, one can take advantage of existing software, hardware, experiences, and datasets to quickly "reproduce" a huge foundational model. However, this approach is unlikely to result in a model that stands out from others, as it is based on shared recipes. In contrast, taking the latter approach may require optimizing software and hardware, adapting techniques, a sharp sense of research direction, and possessing a keen sense of research direction along with strong problem solving abilities. 14 Despite the high requirements for personnel, this approach holds the potential for tremendous breakthroughs. Ultimately, the decision between these two approaches depends on the available resources and personnel. To aid in this decision-making process, this section provides a simple and brutal comparison:
How much more data does the transformer recipe approach need to match the performance of memory mosaics v2?
this section cite: []

Section: Comparison of two approaches
To answer this question, we compare the new tasks learning abilityfoot_10 of memory mosaics v2 and transformers trained on various amounts of data. Specifically, multiple transformer models are trained on 200B, 1T, and 8T training tokens, while a memory mosaics v2 is trained on 1T training tokens. New-knowledge storage and retrieval Table 5 shows the comparison on the new-knowledge storage and retrieval ability. Training on the same number of tokens (1T), transformers lag behind memory mosaics v2 by 12.3% (41.1% vs 53.4%). ×8 times more training tokens (8T) improves the performance of transformers. However, the resulting transformer (trained on 8T tokens) still lags behind memory mosaics v2 (trained on 1T tokens) by 6.5% (46.9% vs 53.4%). Although further increasing training data may improve the performance of transformers in this evaluation dimension, it comes at the cost of significantly larger training cost (time and resource). Moreover, a serious problem occurs: we are running out of data!   In-context learning Figures 6 and 7 show the comparison on in-context learning ability. For semantic label tasks (Figure 6), ×8 times more training data helps transformers (8T data) match the performance of memory mosaics v2 (1T data). However, for the more challenging anonymous label tasks, more training data cannot help transformers. Contour-intuitively, transformers trained on more training data (8T) exhibit a degraded performance on anonymous label tasks (Figure 7). In summary, ×8 more training data helps transformers in certain new task learning benchmarks. However, the resulting transformers (8T data) still lag behind memory mosaics v2 trained on 1T data. More importantly, in anonymous label tasks that heavily rely on the new task learning ability, more training data cannot help transformers. These experiments answer the initial question: "How much data does the transformer recipe approach need to match the performance of memory mosaics v2?".
7 Fine-tuning speed: who can fine-tune with one minibatch?
Despite the strong in-context learning capability of memory mosaics v2 shown in Section 5.3, it may still be attractive to fine-tune a model for a specific domain in order to either reduce inference costs or improve in-domain performance. It is generally expected that such models can be efficiently fine-tuned for a new domain using a comparatively small number of examples. Figure 8 compares the fine-tuning speed (in terms of data size) of memory mosaics v2 and transformers. Both models, pre-trained on 4k context windows, were fine-tuned to 32k context length using the recipe described in Section 4 and evaluated on the same RULER tasks (32k task-length) described in Section 5.2. Surprisingly, a single fine-tuning mini-batch (one optimization step) on memory mosaics v2 yields a 22% accuracy improvement. Two fine-tuning mini-batches on memory mosaics v2 are sufficient to reach the optimal performance. In contrast, a transformer fine-tuned with 800 mini-batches still lags behind memory mosaics v2 fine-tuned with a single mini-batch.
this section cite: []

Section: Discussion and future direction
This work scales memory mosaics (named memory mosaics v2) to llama-8B scale, demonstrating superior performance on new task learning, outperforming transformers by more than 10%. The three evaluation dimensions introduced in this work provide a transparent and controlled assessment of model capabilities, particularly focusing on the new task learning. The risk-return trade-off analysis reveals the weakness of the mainstream "more data more computation" belief, highlighting research opportunities on other smart techniques. One future direction is to reduce the computational cost for very long context lengths using fuzzy hashing Breitinger et al. [2014], Chen et al. [2024] and hierarchical memory Yuan et al. [2025], Lu et al. [2025] approaches.
this section cite: ['b6', 'b9', 'b33', 'b21']

Section: Memory Mosaics at scale Supplementary Material
A Gated time-variant key feature extractor & convolutional value extractor
this section cite: []

Section: B Training data sequence length distributions
Figure 10 shows the distributions of the length of the training data sequence truncated to 4096 or 32,768 max length.
this section cite: []

Section: C Training details
hyperparameters For all Memory Mosaics v2 and baseline Transformer models 16 , we use a consistent set of hyperparameters. That is, a batch size of 1024, a sequence length of 4096, an adamw optimizer with β 1 = 0.9 and β 2 = 0.95 accompanied by a L 2 weight decay of 0.1 and a gradient norm clip of 1, a learning rate warm-up of 2000 iterations followed by a cosine learning rate scheduler that reduces the learning rate by a factor of 100 at the end. The initial learning rates (after warm-up) are set to 3e-4 for "small" models and 1e-3 for "large" models.
We also employ document-wise attention mask, where the attention scores are only computed within each sequence (document) in the training data, to reduce computation cost. Two special tokens, "<|begin_of_text|>" and "<|end_of_text|>" are appended at the begining and ending of a sequence, respectively.
During training, memory mosaics v2 samples the long-term memory delay step m from [64, 256], sets the short-term memory window size h = 256. At inference, m is set to 64, as illustrated in Figure 11.
Randomly Overlapped long-term & short-term memory Short-term memory Attention mask from t-1 to t-h. h is fixed (to 256) throughout training and inference Long-term memory attention mask from t-m to 1 m is uniformly sampled [64, 256] throughout training, fixed to 64 at inference t-1 t-h t-m 1 2 2 1 It is worth noting that these hyperparameters were originally searched and optimized for the baseline transformer models. We transfer these hyperparameters to memory mosaics v2 without further hyperparameter searching. Thus, it is possible that this hyperparameter setup is suboptimal for memory mosaics v2.
this section cite: []

Section: Parameter Initialization and reparameterization
Table 6 summarizes the parameter initialization methods and reparameterization tricks. W 1 , W 2 , W 3 refer to the parameters in persistent memory that are implemented as two-layer dense neural networks, W 2 SiLU (W 1 (x)) * W 3(x) . SiLU (x) = x • sigmoid(x) is an activation function. d ∈ {2048, 4096} indicates the hidden dimension of Memory Mosaics v2 small and large. d ′ ∈ {6144, 14336} indicates the hidden dimension of the two-layer neural networks in persistent memory. l indicates the depth of the Mosaics blocks, starting from 0.
β 0 adaptive bandwidth β 0 = e min(θ,10) θ = 1.5 β 1 adaptive bandwidth β 1 = e min(θ,10) θ = 1.5 α adaptive bandwidth α = min(|θ|, 1) θ = 1/3 α ψ feature extractor α ψ = e min(|θ|,15) θ = 0 γ feature extractor - U (0, 1) W ψ , W φ , W g , W λ , W o long-short memory - min max(N (0, σ), -3σ), 3σ , σ = 1 √ 2d(l+1) W 1 , W 3 persistent memory - min max(N (0, σ), -3σ), 3σ , σ = 1 √ 2d(l+1) W 2 persistent memory - min max(N (0, σ), -3σ), 3σ , σ = 1 √ 2d ′ (l+1) W e , W c embedding & classifier - min max(N (0, σ), -3σ), 3σ , σ = 1 √ 2d
this section cite: []

Section: D Failures of memory compression baselines
Many memory compression algorithms, such as RNNs, xLSTM [Beck et al., 2025], rwkv [Peng et al., 2023], and state-space models [Gu and Dao, 2023], fail on new-task storage and retrieval and in-context learning evaluation dimensions by construction. The reason is that these memory compression algorithms lack the ability to store large amounts of information before getting a command on how to process the information. One might argue to play around this shortage by reading the "command" before storing the large amounts of information. However, this process involves task-specific priori knowledge from human designers. In the end, instead of proving the machine is intelligent, it often proves that human designers are intelligent. Please recall that a child does not prepare all questions before going to school.
This incompetent of memory compression algorithms has been experimentally demonstrated by Hsieh et al. [2024] and Li et al. [2024] on both RULER benchmarks and in-context learning tasks.
Table 7 compares memory compression methods (rwkv-v5-7b and mamba-2.8b-slimpj) and noncompression method (llama2-7b) on RULER long-context tasks. It is clear that memory compression methods perform poorly as the required context length (i.e., required information storage space) increases. Similarly, Table 8 compares memory compression methods (rwkv-5-world 7b and Mamba-2.8B) and non-compression method (qwen-1.5-7b-base and mistral-7b-v0.2-base) on in-context learning tasks (Tacred few-shot classification [Zhang et al., 2017]). In this challenging in-context scenario, memory compression methods just don't work at all. Please note that this section shouldn't be used to criticize or hinder the study of memory compression methods. Memory compression methods have their advantages. In persistent-knowledge storage and retrieval evaluation dimension, they performs very well. For model efficiency, memory compression methods reveal a charming computation complexity. The goal of this section is to explain why this paper doesn't choose memory compression methods as baselines.  [Zhang et al., 2017]). Memory compression methods fail on all cases. Numbers are copied from Li et al. [2024] Table 4. model 1-shot 2-shots 3-shots 4-shots 5-shots qwen-1.5-7b-base 7b 38.7 47.3 45.2 43.6 40.6 mistral-7b-v0.2-base 53.3 53.1 51.6 48.0 42.3 rwkv-5-world 7b 2.3 2.6 1.0 0 1.2 Mamba-2.8B 0 0 0 0 0
this section cite: ['b2', 'b25', 'b15', 'b18', 'b20', 'b36', 'b36', 'b20']

Section: E Model efficiency comparison
As we emphasized in main text, model efficiency (e.g. model service, throughput, VRAM, etc.) is not the goal of this work. Many engineering works can be performed to adapt memory mosaics v2 to a custom use case or hardware. To aid in these potential adaptations, Figure 12 provides a model efficiency comparison in both computation (FLOPs) and model size (number of parameters) viewpoints. The results show that memory mosaics v2 outperforms transformer by more than 10% under either the same FLOPs or the parameters budgets.
this section cite: []

Section: F Additional results on persistent-knowledge storage and retrieval
Table 9 shows six language benchmarks in which removing long-term memory from memory mosaics v2 after training degrades its performance.
this section cite: []

Section: G Additional results on new-knowledge storage and retrieval
Table 10 shows that removing long-term memory from memory mosaics v2 after training degrades the performance on the RULER question-answer tasks by 20%∼30%. This indicates that the ruler question-answer tasks rely on long-term memory to perform well. Table 11 compares memory mosaics v2 large and other public base models on RULER question-answer tasks. Memory mosaics v2 large outperforms these models across all task lengths. Table 12 illustrates the effect of the stochastic long-term memory size training setup introduced in Section C. This stochastic long-term memory size setup is used to encourage the allocation of position-invariant signals and position-dependent signals to long-term and short-term memories. Table 13 compares memory mosaics v2 and transformers on a typical 'needle-in-a-haystack' task from RULER [Hsieh et al., 2024]. The typical 'needle-in-a-haystack' is too easy such that many models can achieve a near-perfect performance.  transformer small 32k 99.4 99.0 98.2 97.8 memory mosaics v2 small 32k 100.0 100.0 100.0 100.0 transformer large 32k 100.0 100.0 100.0 99.6 memory mosaics v2 large 32k 100.0 100.0 100.0 100.0
this section cite: ['b18']

Section: H Additional results for in-context learning
Figure 13 and 14 shows the in-context learning comparison between memory mosaics v2 small and transformer small (llama-1.5B scale).
2 4 6 8 10 12 14 16 number of shots 0.60 0.65 0.70 0.75 0.80 0.85 Accuary banking77 with semantic label 2 3 4 5 6 7 8 9 10 number of shots 0.25 0.30 0.35 0.40 0.45 Accuary 16.7% tacred with semantic label 2 4 6 8 10 12 14 16 number of shots 0.12 0.14 0.16 0.18 0.20 0.22 0.24 Accuary goemotion with semantic label Transformer small Memory Mosaics v2 small Figure 13: Semantic label in-context learning comparison between memory mosaics v2 and transformer. Memory mosaics v2 significantly outperform transformers on in-context learning with a large margin (more than 10%). Meanwhile, memory mosaics v2 benefits from more demonstration shots (x-axis), unlike transformers. 2 4 6 8 10 12 14 16 number of shots 0.3 0.4 0.5 0.6 0.7 0.8 Accuary banking77 with anonymous label 2 3 4 5 6 7 8 9 10 number of shots 0.10 0.15 0.20 0.25 0.30 Accuary 12.7% tacred with anonymous label 2 4 6 8 10 12 14 16 number of shots 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 Accuary goemotion with anonymous label Transformer small Memory Mosaics v2 small Figure 14: Anonymous label in-context learning comparison between memory mosaics v2 and transformers. Memory mosaics v2 significantly outperform transformers on all classification tasks. I Prompt examples of multiclass classification tasks I.1 Banking77 classification with semantic labels We sweep the delimiter from "[return]" and "[space]", leads to the following two prompts: "Given a customer service query, please predict the intent of the query. The predict answer must come from the demonstration examples with the exact format. The examples are as follows: service query: I am still waiting on my card? intent category: city_arrival service query: My card has been found. Is there any way for me to put it back into the app? intent category: city_linking ... service query: Can I get a card even if I live outside the UK? intent category: " "Given a customer service query, please predict the intent of the query. The predict answer must come from the demonstration examples with the exact format. The examples are as follows: service query: I am still waiting on my card? intent category: city_arrival service query: My card has been found. Is there any way for me to put it back into the app? intent category: city_linking ... service query: Can I get a card even if I live outside the UK? intent category:"
For each prompt with either "[return]" or "[space]" delimiter, we also try to shuffle the demonstration example (i.e., service query: [...], intent category:[...]) orders within each one shot. This shuffling process provides another two more prompts.
this section cite: []

Section: I.2 Banking77 classification with anonymous labels
Anonymous tasks use the same set of prompts except that anonymous tasks replace semantic labels (e.g. city_arrival, city_linking) with anonymous labels (e.g. class_00, class_01).
this section cite: []

Section: I.3 Goemotion classification with semantic labels
We sweep the delimiter from "[return]" and "[space]", leads to the following two prompts:
"Given a comment, please predict the emotion category of this comment. The predict answer must come from the demonstration examples with the exact format. The examples are as follows: comment: Her upper lip always looks terrible -such an easy fix, can u believe she is so vain and never bothers to wax emotion category: embarrassment comment: No problem. I'm happy to know it's not what you meant. emotion category: joy ... comment: These refs have it out for the colts. I didn't realize we traded our MVP 11 to KC either. emotion category: " "Given a comment, please predict the emotion category of this comment. The predict answer must come from the demonstration examples with the exact format. The examples are as follows: comment: Her upper lip always looks terrible -such an easy fix, can u believe she is so vain and never bothers to wax emotion category: embarrassment comment: No problem. I'm happy to know it's not what you meant. emotion category: joy ... comment: These refs have it out for the colts. I didn't realize we traded our MVP 11 to KC either. emotion category:"
For each prompt with either "[return]" or "[space]" delimiter, we also try to shuffle the demonstration example orders within each one shot. This shuffling process provides another two more prompts.
this section cite: []

Section: I.4 Goemotion classification with anonymous labels
Anonymous tasks use the same set of prompts except that anonymous tasks replace semantic labels with anonymous labels (e.g. class_00, class_01).
this section cite: []

Section: I.5 Tacred classification with semantic labels
We sweep the delimiter from "[return]" and "[space]", leads to the following two prompts:
"Given a sentence and a pair of subject and object entities within the sentence, please predict the relation between the given entities. The examples are as follows:
sentence: But US and Indian experts say it has hesitated to take action against Lashkar-e-Taiba, which means "The Army of the Pure, "believing that the Islamic militants
could prove useful in pressuring its historic rival India. the relation between Lashkar-e-Taiba and Army of the Pure is: org:alternate_names sentence: The offer from ITW, the Glenview, Ill, diversified manufacturer of engineered products, represents a premium of 85 percent to the Manitowoc bid. the relation between ITW and Glenview is: org:city_of_headquarters ... sentence: The statement from North Korea, carried by the country's official Korean Central News Agency, did not mention Kim by name, but South Korean Unification Ministry spokesman Kim Ho-nyeon said the North's state media has before used such wording to refer to him. the relation between Korean Central News Agency and North Korea is: " "Given a sentence and a pair of subject and object entities within the sentence, please predict the relation between the given entities. The examples are as follows: sentence: But US and Indian experts say it has hesitated to take action against Lashkar-e-Taiba, which means "The Army of the Pure, "believing that the Islamic militants could prove useful in pressuring its historic rival India. the relation between Lashkar-e-Taiba and Army of the Pure is: org:alternate_names sentence: The offer from ITW, the Glenview, Ill, diversified manufacturer of engineered products, represents a premium of 85 percent to the Manitowoc bid. the relation between ITW and Glenview is: org:city_of_headquarters ... sentence: The statement from North Korea, carried by the country's official Korean Central News Agency, did not mention Kim by name, but South Korean Unification Ministry spokesman Kim Ho-nyeon said the North's state media has before used such wording to refer to him. the relation between Korean Central News Agency and North Korea is:"
For each prompt with either "[return]" or "[space]" delimiter, we also try to shuffle the demonstration example orders within each one shot. This shuffling process provides another two more prompts.
this section cite: []

Section: I.6 Tacred classification with anonymous labels
Anonymous tasks use the same set of prompts except that anonymous tasks replace semantic labels with anonymous labels (e.g. class_00, class_01).
this section cite: []

Section: J Computation resources
All experiments are conducted on H100 GPUs with 80GB VRAM.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: The main claims made in the abstract and introduction do accurately reflect the contribution and scope made by the paper.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: Check Section "Computation and # parameters concerns" for example.
"Limitations" section does discuss the limitations of the work performed by the authors, including scope of the claims made, strong assumptions, and how robust the results are to violations of the assumptions.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: We do not include theorems.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: Check experimental setups Section 4, Section 5, and Appendix C. Guidelines: The paper clearly explains the training setup for experiments run in the paper. These explanations are sufficient to reproduce the experiments. The code will be released upon acceptance of the paper.
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/pu  blic/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: Check Experiment Section4 and Appendix C. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: The in-context learning result figures show the performance of different evaluation configurations together with a fitting curve. The many accuracy results of different evaluation configurations provides a means to estimate variance. Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: 
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: Check Appendix J.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: The research conducted in the paper does confirm, in every respsect, with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes]
Justification: The paper discusses the potential impacts of such a works and the resulting effects introduced by these considerations.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: not applicable.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: The creators and original owners of assets used in the paper properly credited and the license and terms of use explicitly mentioned.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable
). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing or research with human subjects Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing or research with human subjects Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2019)
Ref_id:b1 Title: Neural machine translation by jointly learning to align and translate Year: (2015)
Ref_id:b2 Title: Extended long short-term memory Year: (2025)
Ref_id:b3 Title: The long-document transformer Year: (2020)
Ref_id:b4 Title: A meta-transfer objective for learning to disentangle causal mechanisms Year: (2019)
Ref_id:b5 Title: Birth of a transformer: A memory viewpoint Year: (2023)
Ref_id:b6 Title: Approximate matching: Definition and terminology. techreport nist special publication 800-168. national institute of standards and technology Year: (2014)
Ref_id:b7 Title: Efficient intent detection with dual sentence encoders Year: (2020)
Ref_id:b8 Title: Transformer flops Year: (2023)
Ref_id:b9 Title: Lsh sampling for efficient llm generation Year: (2024)
Ref_id:b10 Title: Independent component analysis, a new concept Year: (1994)
Ref_id:b11 Title: Goemotions: A dataset of fine-grained emotions Year: (2020)
Ref_id:b12 Title: Model-agnostic meta-learning for fast adaptation of deep networks Year: (2017)
Ref_id:b13 Title: Notes for Nonparametric Statistics Year: (2024)
Ref_id:b14 Title: The llama 3 herd of models Year: (2024)
Ref_id:b15 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b16 Title: Changing Answer Order Can Decrease MMLU Accuracy Year: (2024-06)
Ref_id:b17 Title: The elements of statistical learning Year: (2009)
Ref_id:b18 Title: Ruler: What's the real context size of your long-context language models Year: (2024)
Ref_id:b19 Title: Needle in a haystack -pressure testing llms Year: (2023)
Ref_id:b20 Title: Long-context llms struggle with long in-context learning Year: (2024)
Ref_id:b21 Title: Moba: Mixture of block attention for long-context llms Year: (2025)
Ref_id:b22 Title: Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models Year: (2024-10)
Ref_id:b23 Title: On estimating regression. Theory of Probability and Its Applications Year: (1964)
Ref_id:b24 Title: -context learning and induction heads Year: (2022)
Ref_id:b25 Title: Reinventing rnns for the transformer era Year: (2023)
Ref_id:b26 Title: Train short, test long: Attention with linear biases enables input length extrapolation Year: (2021)
Ref_id:b27 Title: Disentanglement of correlated factors via hausdorff factorized support Year: (2022)
Ref_id:b28 Title: GLU variants improve transformer Year: (2020)
Ref_id:b29 Title: Augmenting self-attention with persistent memory Year: (2019)
Ref_id:b30 Title: Principles of risk minimization for learning theory Year: (1991)
Ref_id:b31 Title: Attention is all you need Year: (2017)
Ref_id:b32 Title: Smooth regression analysis Year: (1964)
Ref_id:b33 Title: Native sparse attention: Hardware-aligned and natively trainable sparse attention Year: (2025)
Ref_id:b34 Title: Ai for the open-world: the learning principles Year: (2025)
Ref_id:b35 Title: Memory mosaics Year: (2025)
Ref_id:b36 Title: Positionaware attention and supervised data improve slot filling Year: (2017)
