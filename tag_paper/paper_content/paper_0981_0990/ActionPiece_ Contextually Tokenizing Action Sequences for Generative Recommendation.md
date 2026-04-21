Title: ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation
Abstract: Generative recommendation (GR) is an emerging paradigm where user actions are tokenized into discrete token patterns and autoregressively generated as predictions. However, existing GR models tokenize each action independently, assigning the same fixed tokens to identical actions across all sequences without considering contextual relationships. This lack of contextawareness can lead to suboptimal performance, as the same action may hold different meanings depending on its surrounding context. To address this issue, we propose ActionPiece to explicitly incorporate context when tokenizing action sequences. In ActionPiece, each action is represented as a set of item features. Given the action sequence corpora, we construct the vocabulary by merging feature patterns as new tokens, based on their co-occurrence frequency both within individual sets and across adjacent sets. Considering the unordered nature of feature sets, we further introduce set permutation regularization, which produces multiple segmentations of action sequences with the same semantics. Our code is available at: https://github.com/  google-deepmind/action_piece.

Section: Introduction
Generative recommendation (GR) (Tay et al., 2022;Rajput et al., 2023;Zheng et al., 2024;Zhai et al., 2024) is an emerging paradigm for the sequential recommendation task (Hidasi et al., 2016;Kang & McAuley, 2018). By tokenizing the user actions (typically represented by the
Token Sequence 1 Segment 2 14844 21063 284 76 679 20155 1100 7995 1734 6784 Token Sequence 2 747 284 941 76 362 276 21063 8316 19895 24364 6784 1734 Each action is represented as an unordered feature set. This figure presents two possible tokenized sequences, where features are grouped into different segments. The same action can be tokenized into different tokens depending on the surrounding context. A detailed case study can be found in Section 4.5.
interacted items) into discrete tokens, GR models learn to autoregressively generate tokens, which are then parsed into recommended items. These tokens share a compact vocabulary that does not scale with the item pool size, improving model scalability, memory efficiency, and recommendation performance. The input action sequence is vital in understanding user intentions (Hidasi et al., 2016;Li et al., 2017;Kang & McAuley, 2018), which organizes a user's historical interactions in chronological order. The same action (e.g., purchasing the same item) may have different meanings in different action sequences. Evidence of taking a certain action can be found in the context, such as whether other items in the sequence share the same brand, color tone, or price range (Zhang et al., 2019;Zhou et al., 2020;Hou et al., 2022;2023;Yuan et al., 2023).
Despite the importance of contextual relations among actions, existing methods tokenize each action independently of its context (summarized in Table 1). The typical pipeline for tokenizing action sequences involves two steps: (1) Tokenizing each action/item individually into a pattern of tokens; (2) Replacing each action in the input sequence with its corresponding token pattern. In this way, the tokens do not explicitly contain the context. Instead, they solely rely on the autoregressive model's parameters being well-trained to generalize effectively in understanding the context, which challenges the capabilities of GR models. As a compari-son, tokenization in language modeling also originates from context-independent methods, such as word-level tokenization (Sutskever et al., 2014;Bahdanau et al., 2015). A decade of progress has led to most tokenization methods for modern large language models (LLMs) (OpenAI, 2022;Anil et al., 2023;Touvron et al., 2023;Zhao et al., 2023) adopting context-aware approaches, including BPE (Sennrich et al., 2016) and Unigram tokenization (Kudo, 2018), which tokenize the same word roots along with their adjacent context into different tokens.
In this work, we aim to make the first step towards contextaware tokenization for modeling action sequences. In analogy to how characters or bytes serve as the basic units in language modeling, we consider the associated features of an item as initial tokens. The idea is to iteratively find the most commonly co-occurring pairs of tokens among the training action sequences, then merge them into new tokens to represent segments of context. However, it's non-trivial to achieve this. Unlike text, where characters naturally form a sequence, the features associated with an action form an unordered set (Zhang et al., 2019;Zhou et al., 2020). Thus, the proposed tokenization algorithm should be applied on sequences of token sets. We need to carefully consider which pairs of tokens should be counted, whether within a single set or between two adjacent sets, and how much weight should be given to these different types of relationships.
To this end, we propose ActionPiece, which enables the same actions to be tokenized into different tokens based on their surrounding context. (1) Vocabulary construction begins by initializing the vocabulary to include every unique feature as initial tokens. The vocabulary is then constructed by iteratively learning merge rules. Each merge rule specifies that a pair of tokens can be merged into a new token. In each iteration, we enumerate the training corpus to count the co-occurrence of existing tokens. Considering the structural differences between token pairs, e.g., whether they occur within a single set or between two adjacent sets, we assign different weights to different pairs. (2) Segmentation refers to dividing raw features in action sequences into groups that can be replaced by tokens from the vocabulary. To fully exploit the unordered nature of the feature set of each action, we introduce set permutation regularization. By randomly permuting the features within each set, we can produce multiple token sequences of a single action sequence that preserve the same semantics. These variations act as natural augmentations for training data and enable inherent ensembling during model inference.
this section cite: ['b51', 'b41', 'b72', 'b64', 'b12', 'b22', 'b12', 'b27', 'b22', 'b66', 'b74', 'b14', 'b62', 'b49', 'b1', 'b0', 'b52', 'b71', 'b43', 'b24', 'b66', 'b74']

Section: Related Work
Generative recommendation. Conventional sequential recommendation models often relies on large embedding tables to store representations for all items, leading to signif-Table 1. Comparison of different action tokenization methods for generative recommendation. "Contextual" denotes whether the same actions can be tokenized into different tokens based on the surrounding context. "Unordered" denotes whether the item features or semantic IDs are used in an order-agnostic manner.
this section cite: []

Section: Action Tokenization Example Contextual Unordered
Product Quantization VQ-Rec (Hou et al., 2023) Hierarchical Clustering P5-CID (Hua et al., 2023) Residual Quantization TIGER (Rajput et al., 2023) Text Tokenization LMIndexer (Jin et al., 2024) Raw Features HSTU (Zhai et al., 2024) SentencePiece SPM-SID (Singh et al., 2024) ActionPiece Ours icant engineering and optimization challenges (Hidasi et al., 2016;Li et al., 2017;Kang & McAuley, 2018). Generative recommendation (Rajput et al., 2023;Zheng et al., 2024;Zhai et al., 2024;Deldjoo et al., 2024;Hou et al., 2025) addresses these issues by tokenizing each item as tokens from a shared vocabulary. By autoregressively generating the next tokens as recommendations, this generative paradigm offers benefits such as memory efficiency (Rajput et al., 2023;Yang et al., 2024;Ding et al., 2024), scalability (Zhai et al., 2024;Liu et al., 2024b), and easier alignment with LLMs (Zheng et al., 2024;Jin et al., 2024;Tan et al., 2024;Li et al., 2025). Existing research has developed different action tokenization techniques, such as hierarchical clustering (Hua et al., 2023;Si et al., 2024), quantization (Rajput et al., 2023;Wang et al., 2024a;Zhu et al., 2024a), or jointly training with recommendation models (Liu et al., 2025).
Other works incorporate additional modalities like collaborative filtering (Petrov & Macdonald, 2023;Wang et al., 2024c;b;Liu et al., 2024b;a) and natural language (Zheng et al., 2024;Jin et al., 2024;Hou et al., 2024b;Zhang et al., 2025a). However, current methods tokenize each action independently, ignoring the surrounding context. In this work, we propose the first context-aware action tokenization method, where the same actions are tokenized differently in different action sequences.
Tokenization for language modeling. Tokenization is the process of transforming raw text into discrete token sequences (Kudo & Richardson, 2018). Early word-level methods are context-independent and struggle to tokenize out-of-vocabulary words (Sutskever et al., 2014;Bahdanau et al., 2015). Consequently, subword-level tokenization has gradually become the more mainstream choice. The vocabularies of these subword-level tokenizers are constructed iteratively, either bottom-up (starting with a small vocabulary and merging commonly occurring token pairs as new tokens) (Wu et al., 2016;Sennrich et al., 2016), or top-down (starting with a large vocabulary and pruning tokens to minimize likelihood decrease) (Kudo, 2018;Yehezkel & Pinter, 2023). Once the vocabulary is built, the text can be seg-
this section cite: ['b15', 'b19', 'b41', 'b20', 'b64', 'b47', 'b12', 'b27', 'b22', 'b41', 'b72', 'b64', 'b6', 'b18', 'b41', 'b60', 'b7', 'b64', 'b72', 'b20', 'b50', 'b26', 'b19', 'b46', 'b41', 'b29', 'b38', 'b72', 'b20', 'b25', 'b49', 'b1', 'b59', 'b43', 'b24', 'b61']

Section: Algorithm 1 ActionPiece Vocabulary Construction
input Sequence corpus S ′ , initial tokens V0, target size Q output Merge rules R, constructed vocabulary V 1: Initialize vocabulary V ← V0 # Each initial token corresponds to one unique item feature 2: R ← ∅ 3: while |V| < Q do 4: # Count: accumulate weighted token co-occurrences 5: count(•, •) ← Count(S ′ , V) # Algorithm 2 6: # Update: merge a frequent token pair into a new token 7: Select (cu, cv) ← arg max (c i ,c j ) count(ci, cj) 8: S ′ ← Update(S ′ , {(cu, cv) → cnew}) # Algorithm 3 9: R ← R ∪ {(cu, cv) → cnew} # New merge rule 10:
V ← V ∪ {cnew} # Add new token to the vocabulary 11: end while return R, V mented either using the same method employed during vocabulary construction or based on additional objectives (He et al., 2020;Provilkov et al., 2020;Hofmann et al., 2022;Schmidt et al., 2024). As an analogy, existing action tokenizers are context-independent and function like "word-level" language tokenizers. In this work, we take the first step toward context-aware subaction-level action tokenizer.
this section cite: ['b11', 'b39', 'b13', 'b42']

Section: Method
In this section, we present ActionPiece, a context-aware method for tokenizing action sequences for generative recommendation. First, we formulate the task in Section 3.1. Then, we introduce the proposed tokenizer, covering vocabulary construction and segmentation, in Section 3.2. Finally, we describe the model training and inference process using ActionPiece-tokenized sequences in Section 3.3.
this section cite: []

Section: Problem Formulation
Given a user's historical actions S = {i 1 , i 2 , . . . , i t }, organized sequentially by their timestamps, the task is to predict the next item i t+1 the user will interact with.
Action as an unordered feature set. In the development of modern recommender systems, each item i j is usually associated with a set of features A j (Zhang et al., 2019;Zhou et al., 2020;Cheng et al., 2016). Assuming there are m features per item, the k-th feature of item i j is denoted as f j,k ∈ F k , where F k is the collection of all possible choices for the k-th feature. Compared to representing actions using ordered semantic IDs (e.g., those produced by RQ-VAE (Rajput et al., 2023;Singh et al., 2024)), the unordered set setting offers two key advantages: (1) It does not require a specific order among features, which aligns better with how items or actions are represented in most recommender systems; (2) It enables the inclusion of more general discrete and numeric features, such as category, brand, and price (Pazzani & Billsus, 2007;Juan et al., 2016).
Action sequence as a sequence of sets. Representing each item as an unordered set, the input action sequence can be written as S ′ = {A 1 , A 2 , . . . , A t }, which is a chronologically ordered sequence of sets. There is no order within each set, but there are orders between the features from different sets. The tokenizer design should account for the ordered and unordered relationships among features.
this section cite: ['b66', 'b74', 'b4', 'b41', 'b47', 'b37', 'b21']

Section: Generative recommendation task.
In this work, we aim to design a tokenizer that maps an input action sequence S ′ to a token sequence C = {c 1 , c 2 , . . . , c l }, where l denotes the number of tokens in the sequence. Note that l is typically greater than the number of actions t. Next, we train a GR model to autoregressively generate tokens {c l+1 , . . . , c q }, which can be parsed as next-item predictions ît+1 .
this section cite: []

Section: Contextual Action Sequence Tokenizer
The proposed tokenizer is designed to transform action sequences (represented as sequences of feature sets) into token sequences. In the ActionPiece-tokenized sequences, each token corresponds to a set containing varying numbers of features. For example, a token can represent: (1) a subset of features from one item; (2) a single feature; (3) all features of one item; or (4) features from multiple items. We also label these four types of tokens in Figure 1. Below, we first describe how to construct the ActionPiece tokenizer's vocabulary given a corpus of action sequences (Section 3.2.1). Then, we introduce how to segment action sequences into a new sequence of sets, where each set corresponds to a token from the constructed vocabulary (Section 3.2.2).
this section cite: []

Section: VOCABULARY CONSTRUCTION ON ACTION SEQUENCE CORPUS
Given a corpus of action sequences S ′ , the goal of vocabulary construction is to create a vocabulary V of Q tokens. Each token represents a combination of features that frequently occur in the corpus. Similar to BPE (Sennrich et al., 2016), we construct the vocabulary using a bottom-up approach. The process starts with an initial vocabulary of tokens V 0 . The construction proceeds iteratively, adding one new token to the vocabulary at each iteration until the predefined target size is reached. Each iteration consists of two consecutive steps: count, where the most frequently occurring token pair is identified, and update, where the corpus is modified by merging the selected pair into a new token.
An algorithmic workflow is illustrated in Algorithm 1.
Vocabulary initialization. In BPE, each token represents a sequence of bytes. Thus, the most fundamental unitsthe initial tokens-are single bytes, which form the initial vocabulary of BPE. Similarly, each token in ActionPiece represents a set of features. Therefore, we initialize Action-Piece with a vocabulary in which each token represents a set containing one unique item feature. Formally, we denote the initial vocabulary as V 0 = {c = {f }|f ∈ F 1 ∪ . . . ∪ F m }.
After initializing the vocabulary, each action sequence (of feature sets) can be represented as a sequence of token sets.
Count: context-aware token co-occurrence counting. In each iteration of vocabulary construction, the first step is to count the co-occurrence of token pairs in the corpus. These pairs capture important feature combinations, which are encoded by creating new tokens. There are two types of token co-occurrence within a sequence of sets: (1) two tokens exist within the same set, or (2) two tokens exist in adjacent sets in the sequence. Notably, the second type allows ActionPiece to explicitly include context information.
Weighted co-occurrence counting. In one-dimensional token sequences (e.g., text), all token pairs are typically treated equally. However, in sequences of token sets, token pairs vary based on their types and the sizes of their respective sets. To account for these differences, we propose assigning different weights to token pairs. To determine the weight for each token pair, we relate sequences of token sets to token sequences by randomly permuting the tokens within each set and flattening them into a single token sequence. Let P (c, c ′ ) represent the expected probability that tokens c and c ′ are adjacent in the flattened sequence. For two tokens from the same set, we have:
P (c 1 , c 2 ) = P (c 2 , c 1 ) = |A i | -1 |Ai| 2 = 2 |A i | , c 1 , c 2 ∈ A i ,(1)
and for two tokens from adjacent sets, we have:
P (c 1 , c 3 ) = 1 |A i | × |A i+1 | , c 1 ∈ A i , c 3 ∈ A i+1 .
(2) By considering the probabilities of all adjacent token pairs in the flattened sequence as 1, the weights for token pairs in the original sequence of token sets correspond to the probabilities given in Equations ( 1) and ( 2). An illustration is shown in Figure 2.
Accumulating co-occurrence weights. The weights described above are calculated based solely on the cooccurrence type and the set size. They do not take into account the specific tokens being analyzed. Tokens c i and c j might appear in the same set in one sequence but in two adjacent sets in another sequence. By iterating through the corpus, we sum up the weights for each token pair whenever they appear together multiple times.
Update: corpus updating with action-intermediate nodes. The next step in each iteration is to merge the token pair with the highest accumulated co-occurrence weight. Since token merging may change the set size, we use a double-ended linked list (Zouhar et al., 2023) to maintain each action sequence, where each node represents a set of tokens. Merging tokens within the same set is straightforward, i.e., replacing the two tokens with a new one. However, merging tokens from two adjacent sets is more complex, e.g., determining which set should include the new token.
Intermediate Node. We introduce the concept of "intermediate node" to handle tokens that combine features from multiple sets. Initially, all nodes in the maintained linked lists contain features specific to their corresponding actions. These nodes are referred to as "action nodes." (1) When tokens from two adjacent action nodes are being merged, we insert a new intermediate node between the two action nodes. The new token is stored in the intermediate node, and the merged tokens are removed from their respective action nodes;
(2) When merging tokens from an action node and an intermediate node, the new token replaces the original token in the intermediate node. The reason is that this new token also combines features from multiple actions. After the merge, the token from the action node is removed.
Following the above update rules ensures that there is at most one intermediate node between any two action nodes, and each intermediate node contains no more than one token. When calculating co-occurrence weights involving an intermediate node, it can simply be treated as a set of size 1.
Efficient implementation. Naively counting and updating the corpus requires a total time complexity of O(QN Lm 2 ), where Q is the target vocabulary size, N is the number of action sequences in the training corpus, and L is the average length of these sequences. However, it is unnecessary to count co-occurrences from scratch in each iteration. This is
Merge tokens in action & intermediate nodes Merge tokens in one action node Merge tokens in two adjacent action nodes Action Node Intermediate Node New Token Original Token Insert a new intermediate node New token in the intermediate node Linked List New token in the original action node because only a small portion of the maintained linked lists is modified compared to the previous iteration.
this section cite: ['b43', 'b77']

Section: Data structures.
To address this, we propose creating inverted indices to map token pairs to all the linked lists that contain them. A global heap is maintained to return the token pair with the highest accumulated co-occurrence. The key challenge lies in updating these data structures. We carefully compute the changes in accumulated co-occurrences and update the inverted indices. For the heap, we employ a lazy-update strategy. We insert the latest weights with a tag. When fetching a value from the heap, we check the tag to verify if the value is up-to-date. If it is not, we discard the value and fetch the next one.
Time complexity. Let H = O(N Lm) represent the maximal heap size. Using the proposed algorithm, we successfully reduce the original time complexity to O(log Q log H • N Lm 2 ), achieving efficient vocabulary construction. In practice, the later iterations take significantly less time than the initial ones. This is expected and because tokens with higher accumulated co-occurrence weights typically appear frequently in the early stages. However, the overall construction time benefits from the reduced amortized complexity. Further details about the vocabulary construction algorithm are provided in Appendix C.
this section cite: []

Section: SEGMENTATION BY SET PERMUTATION REGULARIZATION
Segmentation is to convert original action sequences into a sequence of feature sets. Each set in the segmented sequence corresponds to a token in the vocabulary.
Naive segmentation. One segmentation strategy in Action-Piece involves applying the same technique used to construct the vocabulary. Specifically, this technique iteratively identifies token pairs with high priorities (represented by the IDs of tokens, where tokens added earlier may have higher priority). However, we observed that this strategy can lead to a bias, where only a subset of tokens in the vocabulary is frequently used (as shown empirically in Section 4.4.2).
this section cite: []

Section: Set permutation regularization (SPR).
To address this issue and account for the unordered nature of sets, we propose set permutation regularization, which generates multiple segmentations for each action sequence. The key idea is to avoid enumerating all possible pairs between tokens in a set or adjacent sets. Instead, we generate a random permutation of each set and treat it as a one-dimensional sequence. By concatenating all the permutations, we create a long token sequence. This sequence can then be segmented using traditional BPE segmentation methods (Sennrich et al., 2016). In this approach, different permutations can produce distinct segmented token sequences with the same semantics. These sequences serve as natural augmentations for model training (Section 3.3.1) and enable inherent ensembling during model inference (Section 3.3.2).
this section cite: ['b43']

Section: Generative Recommendation Models

this section cite: []

Section: TRAINING ON AUGMENTED TOKEN SEQUENCES
For an action sequence and its ground-truth next action in the training corpus, we tokenize them into token sequences C in and C out , respectively. Taking C in as input, we train a Transformer encoder-decoder module (Raffel et al., 2020) to autoregressively generate C out (e.g., next-token prediction objective (Rajput et al., 2023)). During training, we tokenize the action sequence using the set permutation regularization described in Section 3.2.2 in each epoch. This approach naturally augments the training sequences, which empirically improves model performance, as shown in Section 4.3.
this section cite: ['b40', 'b41']

Section: INFERENCE-TIME ENSEMBLING
During model inference, we tokenize each action sequence q times using set permutation regularization. By passing these q tokenized sequences through the model, we obtain q output ranking lists (e.g., using beam search for inference when TIGER (Rajput et al., 2023) is the GR backbone). We then combine these ranking lists by averaging the scores of each predicted item. This approach applies data-level ensembling, which has been shown to enhance recommendation performance, as discussed in Section 4.4.3.
this section cite: ['b41']

Section: Discussion
Orders in action sequences. In recommender systems, user actions are typically represented as sets of features, such as the title and price of the associated item. These features typically have no inherent order within a single action. However, in sequential recommendation tasks, a user's historical actions are usually ordered by timestamp to capture temporal behavioral dynamics. Building on this, we model action sequences as sequences of feature sets: while the features within each action remain unordered, the temporal ordering of actions is preserved. This evolving composition of features over time captures meaningful sequential patterns.
ActionPiece vs. BPE. While ActionPiece follows a similar algorithmic framework as BPE, the key distinction lies in the data formats they are designed to model. BPE operates on one-dimensional byte sequences, whereas ActionPiece is tailored for tokenizing sequences of feature sets. Modeling each action as an unordered set aligns better with the inherent structure of action sequences. For clarity, we summarize the key differences in Table 2.
Efficiency impact of SPR. Despite introducing set permutation regularization, the training efficiency remains comparable to existing methods such as TIGER. Feature permutation is performed on the CPU and runs asynchronously alongside TPU/GPU-based model updates, resulting in no noticeable degradation in training speed. At inference time, SPR introduces additional FLOPs due to the ensemble of augmented test cases. However, the overall latency remains comparable to baseline methods, as the augmented versions can be processed in parallel across multiple computing devices (e.g., TPUs or GPUs). This parallelism offsets the added computation, enabling our method to maintain efficient inference despite the use of inference-time ensembling. "Sports and Outdoors" (Sports), "Beauty" (Beauty), and "CDs and Vinyl" (CDs). Each user's historical reviews are considered "actions" and are sorted chronologically as action sequences, with earlier reviews appearing first. To evaluate the models, we adopt the widely used leave-lastout protocol (Kang & McAuley, 2018;Zhao et al., 2022;Rajput et al., 2023), where the last item and second-to-last item in each action sequence are used for testing and validation, respectively. The statistics of the processed datasets are shown in Table 3. More details about the datasets can be found in Appendix F.
this section cite: ['b22', 'b70', 'b41']

Section: References
Ref_id:b0 Title: A family of highly capable multimodal models Year: (2023)
Ref_id:b1 Title: Neural machine translation by jointly learning to align and translate Year: (2015)
Ref_id:b2 Title: TALLRec: An effective and efficient tuning framework to align large language model with recommendation Year: (2023)
Ref_id:b3 Title: CAN: feature co-action network for click-through rate prediction Year: (2022)
Ref_id:b4 Title: Wide & deep learning for recommender systems Year: (2016)
Ref_id:b5 Title: Getting the most out of your tokenizer for pre-training and domain adaptation Year: (2024)
Ref_id:b6 Title: A review of modern recommender systems using generative models (gen-recsys) Year: (2024)
Ref_id:b7 Title: Inductive generative recommendation via retrieval-based speculation Year: (2024)
Ref_id:b8 Title: The faiss library Year: (2024)
Ref_id:b9 Title: Optimized product quantization for approximate nearest neighbor search Year: (2013)
Ref_id:b10 Title: Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5) Year: (2022)
Ref_id:b11 Title: Dynamic programming encoding for subword segmentation in neural machine translation Year: (2020)
Ref_id:b12 Title: Session-based recommendations with recurrent neural networks Year: (2016)
Ref_id:b13 Title: An embarrassingly simple method to mitigate undesirable properties of pretrained language model tokenizers Year: (2022)
Ref_id:b14 Title: Towards universal sequence representation learning for recommender systems Year: (2022)
Ref_id:b15 Title: Learning vector-quantized item representation for transferable sequential recommenders Year: (2023)
Ref_id:b16 Title: Bridging language and items for retrieval and recommendation Year: (2024)
Ref_id:b17 Title: Large language models are zero-shot rankers for recommender systems Year: ()
Ref_id:b18 Title: Generative recommendation models: Progress and directions Year: (2025)
Ref_id:b19 Title: How to index item ids for recommendation foundation models Year: (2023)
Ref_id:b20 Title: Language models as semantic indexers Year: (2024)
Ref_id:b21 Title: Field-aware factorization machines for ctr prediction Year: (2016)
Ref_id:b22 Title: Self-attentive sequential recommendation Year: (2018)
Ref_id:b23 Title: Large language models meet collaborative filtering: An efficient all-round llm-based recommender system Year: (2024)
Ref_id:b24 Title: Subword regularization: Improving neural network translation models with multiple subword candidates Year: (2018)
Ref_id:b25 Title: SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing Year: (2018)
Ref_id:b26 Title: Semantic convergence: Harmonizing recommender systems via two-stage alignment and behavioral semantic tokenization Year: (2025)
Ref_id:b27 Title: Neural attentive session-based recommendation Year: (2017)
Ref_id:b28 Title: LLaRA: Large language-recommendation assistant Year: (2024)
Ref_id:b29 Title: End-to-end learnable item tokenization for generative recommendation Year: (2025)
Ref_id:b30 Title: Multimodal generative recommendation with transformer model Year: (2024)
Ref_id:b31 Title: Multi-behavior generative recommendation Year: (2024)
Ref_id:b32 Title: Image-based recommendations on styles and substitutes Year: (2015)
Ref_id:b33 Title: Introducing Meta Llama 3: The most capable openly available LLM to date Year: (2024)
Ref_id:b34 Title: Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models Year: (2022)
Ref_id:b35 Title:  Year: (2022)
Ref_id:b36 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b37 Title: Content-based recommendation systems Year: (2007)
Ref_id:b38 Title: Generative sequential recommendation with gptrec Year: (2023)
Ref_id:b39 Title: Bpe-dropout: Simple and effective subword regularization Year: (2020)
Ref_id:b40 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b41 Title: Recommender systems with generative retrieval Year: (2023)
Ref_id:b42 Title: Tokenization is more than compression Year: (2024)
Ref_id:b43 Title: Neural machine translation of rare words with subword units Year: (2016)
Ref_id:b44 Title: Deep crossing: Web-scale modeling without manually crafted combinatorial features Year: (2016)
Ref_id:b45 Title: Language representations can be what recommenders need: Findings and potentials Year: (2025)
Ref_id:b46 Title: Generative retrieval with semantic tree-structured item identifiers via contrastive learning Year: (2024)
Ref_id:b47 Title: Better generalization with semantic ids: A case study in ranking for recommendations Year: (2024)
Ref_id:b48 Title: Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer Year: (2019)
Ref_id:b49 Title: Sequence to sequence learning with neural networks Year: (2014)
Ref_id:b50 Title: Idgenrec: Llm-recsys alignment with textual id learning Year: (2024)
Ref_id:b51 Title: Transformer memory as a differentiable search index Year: (2022)
Ref_id:b52 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b53 Title: Deep & cross network for ad click predictions Year: (2017)
Ref_id:b54 Title: Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems Year: (2021)
Ref_id:b55 Title: Learnable tokenizer for llm-based generative recommendation Year: (2024)
Ref_id:b56 Title: Content-based collaborative generation for recommender systems Year: (2024)
Ref_id:b57 Title: Eager: Twostream generative recommender with behavior-semantic collaboration Year: (2024)
Ref_id:b58 Title: Transformers: State-of-the-art natural language processing Year: (2020)
Ref_id:b59 Title: Google's neural machine translation system: Bridging the gap between human and machine translation Year: (2016)
Ref_id:b60 Title: Unifying generative and dense retrieval for sequential recommendation Year: (2024)
Ref_id:b61 Title: Incorporating context into subword vocabularies Year: (2023)
Ref_id:b62 Title: Where to go next for recommender systems? id-vs. modality-based recommender models revisited Year: (2023)
Ref_id:b63 Title: Soundstream: An end-to-end neural audio codec Year: (2021)
Ref_id:b64 Title: Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations Year: (2024)
Ref_id:b65 Title: Recommendation as instruction following: A large language model empowered recommendation approach Year: (2025)
Ref_id:b66 Title: Feature-level deeper self-attention network for sequential recommendation Year: (2019)
Ref_id:b67 Title: Towards scalable semantic representation for recommendation Year: (2024)
Ref_id:b68 Title: CoLLM: Integrating collaborative embeddings into large language models for recommendation Year: (2025)
Ref_id:b69 Title: Towards a unified, comprehensive and efficient framework for recommendation algorithms Year: (2021)
Ref_id:b70 Title: A revisiting study of appropriate offline evaluation for top-n recommendation algorithms Year: (2022)
Ref_id:b71 Title: A survey of large language models Year: (2023)
Ref_id:b72 Title: Adapting large language models by integrating collaborative semantics for recommendation Year: (2024)
Ref_id:b73 Title: Enhancing graph contrastive learning with reliable and informative augmentation for recommendation Year: (2025)
Ref_id:b74 Title: S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization Year: (2020)
Ref_id:b75 Title: Cost: Contrastive quantization based semantic tokenization for generative recommendation Year: ()
Ref_id:b76 Title: Collaborative large language model for recommender systems Year: (2024)
Ref_id:b77 Title: A formal perspective on byte-pair encoding Year: (2023)
