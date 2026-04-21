Title: Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger
Abstract: Recent advancements in Large Vision Language Models (LVLMs) have significantly improved performance in Visual Question Answering (VQA) tasks through multimodal Retrieval-Augmented Generation (RAG). However, existing methods still face challenges, such as the scarcity of knowledge containing reasoning examples and erratic responses from retrieved knowledge. To address these issues, in this study, we propose a multimodal RAG framework, termed RCTS, which enhances LVLMs by constructing a Reasoning Context-enriched knowledge base and a Tree Search re-ranking method. Specifically, we introduce a self-consistent evaluation mechanism to enrich the knowledge base with intrinsic reasoning patterns. We further propose a Monte Carlo Tree Search with Heuristic Rewards (MCTS-HR) to prioritize the most relevant examples. This ensures that LVLMs can leverage high-quality contextual reasoning for better and more consistent responses. Extensive experiments demonstrate that our framework achieves state-of-the-art performance across multiple VQA datasets, significantly outperforming both In-Context Learning (ICL) and Vanilla-RAG methods. It highlights the effectiveness of our knowledge base and reranking method in improving LVLMs. Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models StrongerSimilarity Search Top-N Contexts Embeddings MCTS Re-ranking Knowledge Base Q: Which rhetorical appeal is primarily used in this ad? A: logos (reason) C: **Identify the type of appeal:** The ad claims that the Vilaplus vacuum picks up more dirt than User QuestionIn this food web, which organism contains matter that eventually moves to the bat star? Below is a food web from an ocean ecosystem in Monterey Bay.

Section: 
1. Introduction  Recently, large vision language models (LVLMs) (Achiam et al., 2023;Bai et al., 2023;Chen et al., 2024) exhibit remarkable efficacy across diverse visual question answering (VQA) tasks, being capable of processing multiple images concurrently and, furthermore demonstrating the ability for in-context learning (ICL) (Alayrac et al., 2022;Wang et al., 2024b). These capabilities facilitate the application of multimodal retrieval-augmented generation (RAG) (Gao et al., 2023), a training-free approach that involves augmenting the input prompt by retrieving relevant multimodal corpus from an external knowledge base through semantic similarity calculation. This approach demonstrates that introducing external knowledge effectively reduces the probability that LVLMs generate incorrect content.
Existing LVLMs (Achiam et al., 2023;Bai et al., 2023;Chen et al., 2024) are prone to hallucination issues (Huang et al., 2023), which manifest in two primary forms: generating factual inconsistent with real-world facts (e.g., misstating historical events or political news), and producing erratic responses misaligned with user instructions or questions (e.g., failing to answer user queries). To mitigate factual inconsistencies, existing multimodal RAG methods (Chen et al., 2022;Caffagni et al., 2024;Yan & Xie, 2024) leverage external knowledge (e.g., Wikipedia or Web Search) to transform LVLMs' responses from unknown (lacking factual ground-ing) to known (factually supported). However, addressing instruction misalignment poses a distinct challenge. An intuitive approach is to enhance user prompts by prepending few-shot example pairs through in-context learning (Alayrac et al., 2022). While effective, manual curation of such examples limits scalability.
For issue instruction misalignment, a compelling question arises as to whether multimodal RAG can be integrated into in-context learning, transitioning LVLMs' responses from merely known to better understood (know-how reasoning) by prepending it with retrieved examples. Specifically, it can achieve more reliable responses by retrieving and reasoning over similar examples through in-context learning. However, several possible challenges hinder the practical application of multimodal RAG for addressing this question: i) The retrieved sample question-answer pairs are formatted in a rigid, formulaic manner (e.g., 'The answer is A' for multiple-choice questions), which limits the LVLMs to capture underlying logical patterns. This inspires us to build a more comprehensive knowledge base with reasoning contexts. ii) Retrieved examples may not consistently result in positive outcomes, due to the inherent limitations of incontext learning and the diversity of users' queries, which deserves more discussion. Hence, the focus here centers on instruction misalignment with two main aspects: Firstly, the construction of the knowledge base with reasoning contexts to optimally enhance generation and facilitate in-context learning. Secondly, the strategic re-ranking of retrieved examples to prioritize more suitable samples, thereby promoting efficient and accurate response generation.
In this study, we propose a multimodal RAG framework with Reasoning Context and Tree Search, named RCTS, aiming at constructing a comprehensive knowledge base with reasoning contexts and optimizing the order of contextual examples to improve the question answering performance of LVLMs. For our knowledge base component, we introduce an automated reasoning contexts generation method for question-answer pairs, which helps LVLMs acquire intrinsic reasoning patterns. For the proposed multimodal RAG framework, our method begins with hybrid retrieval for an initial sampling from the knowledge base. Subsequently, we employ a re-ranking mechanism to organize the retrieved samples, enhancing the efficacy of incontext learning. The re-ranked Top-K samples with the generated reasoning contexts are then concatenated with the user's question to facilitate optimal answer generation by LVLMs. Regarding the re-ranking process, we propose a tree search approach with heuristic rewards to re-order the retrieved samples. This ensures the identification and prioritization of the most beneficial contextual examples for the final generation phase, thereby enhancing the overall answer quality. Besides, the reasoning contexts we generated before also allows for a quantitative assessment of the potential benefits offered by the retrieved samples, reinforcing the efficacy of our tree search method.
To validate the effectiveness of our proposed method, we conduct extensive experiments across multiple reasoning VQA datasets, including ScienceQA (Lu et al., 2022), MMMU (Yue et al., 2024), and MathV (Wang et al., 2024a).
Our method also excels in non-reasoning VQA datasets such as VizWiz (Gurari et al., 2018) and VSR-MC (Liu et al., 2023). As depicted in Fig. 1, across various sizes and types of LVLMs, our proposed approach significantly outperforms the zero-shot baseline. Besides, compared to the strategy of randomly selecting examples as context, i.e., ICL, our method yields an average of 3% improvement, demonstrating that our framework elevates LVLMs from mere known to better understood. Additionally, compared to Vanilla-RAG, our method surpasses performance by more than 3% on all models (4.2% on Qwen2-VL (7B), 3.9% on InternVL-2 (8B)), indicating that the knowledge base with reasoning contexts and the tree search with answer heuristic rewards effectively re-rank examples that enhance answer accuracy. Qualitative analysis further corroborates the efficacy of our method.
Our contributions are summarized as follows:
• We introduce a multimodal RAG framework, termed RCTS, to enhance LVLMs by constructing a comprehensive knowledge base with reasoning contexts and re-ranking for highly relevant contexts.
• We develop an automatically constructed reasoning context mechanism grounded in VQA pairs to construct the knowledge base with reasoning contexts, and further propose a tree search strategy with answer heuristic rewards for re-ranking retrieved samples.
• Experiments show that our method achieves significant performance improvements on multiple VQA datasets, demonstrating the effectiveness of the reasoning context and the proposed re-ranking mechanism.
this section cite: ['b1', 'b3', 'b7', 'b2', 'b1', 'b3', 'b7', 'b6', 'b5', 'b16', 'b2', 'b18', 'b22']

Section: Related Work
Large Visual Language Models. Large Visual Language Models (LVLMs) have emerged as a significant research focus, leveraging the capabilities of powerful Large Language Models (LLMs) (Touvron et al., 2023;Jiang et al., 2023;Yang et al., 2024;Abdin et al., 2024) to tackle vision-language tasks. These versatile LVLMs demonstrate exceptional performance, particularly in visual questionanswering (VQA) tasks (Team et al., 2023;Achiam et al., 2023;Bai et al., 2023;Liu et al., 2024), pointing toward a promising avenue for achieving artificial general intelligence. Nevertheless, these models face challenges with knowledge-based VQA due to issues such as hallucina-
#3 #1 … … … … … … … MLP C C Vision Encoder Text Encoder Vision Encoder Text Encoder UQ …
Figure 2. Overview of the proposed framework. RCTS adopts a novel multimodal retrieval-augmented generation framework specifically for visual question answering tasks. Aiming at enhancing the capabilities of the large vision-language models, our method consists of three components. (1) We construct a knowledge base with reasoning contexts by a self-consistent evaluation mechanism. (2) To support the multimodal knowledge base, we employ a hybrid embedding strategy for relevant samples retrieval. (3) Given the uncertainty of the retrieved samples, we propose an improved Monte Carlo Tree Search algorithm with heuristic rewards for sample re-ranking.
tions-where responses are generated from nonexistent content-and inherent biases (Li et al., 2023). Additionally, the lack of efficient knowledge retrieval mechanisms impedes their ability to integrate external knowledge bases for reasoning (Caffagni et al., 2024). In this study, we investigate strategies for constructing comprehensive external knowledge bases to augment the capabilities of LVLMs.
Multimodal In-context Learning. Multimodal in-context learning exemplifies a paradigm in which model weights remain unchanged, and improves output quality by adjusting the model's input (Dong et al., 2022;Alayrac et al., 2022;Han et al., 2023). A typical in-context learning prompt comprises two elements: demonstrations and new queries. Demonstrations involve multiple VQA pairs, each comprising a complete question accompanied by visual information and its corresponding answer. In contrast, new queries consist of questions posed to the model. Leveraging the emergent capabilities of LVLMs, these models can reference demonstrations to some extent to address new questions (Zhao et al., 2023;Zhang et al., 2024b). With the benefit of not requiring fine-tuning model parameters, incontext learning has emerged as a favored paradigm for applying LVLMs. In this study, we construct the reasoning context as an integral part of the context based on VQA pairs, to enrich the reasoning knowledge of the context.
this section cite: ['b11', 'b17', 'b0', 'b10', 'b1', 'b3', 'b5', 'b2', 'b22', 'b22']

Section: Multimodal Retrieval-augmented Generation.
While RAG is well-established in LLMs, its application within LVLMs remains relatively underexplored. Systems such as KAT (Gui et al., 2021), REVIVE (Lin et al., 2022), and RE-VEAL (Hu et al., 2023) show promise in addressing queries involving common-sense reasoning, yet they struggle with more complex, knowledge-intensive tasks like Encyclopedic VQA (E-VQA) (Mensink et al., 2023) and Infoseek (Chen et al., 2023). These limitations are largely due to their constrained ability to fetch and integrate precise information from expansive encyclopedic knowledge bases. RATP (Pouplin et al., 2024) leverages MCTS and RAG to enhance the self-reflection and self-critique capabilities across numerous private healthcare documents. EchoSight (Yan & Xie, 2024) attempts to address these challenges through a two-stage process, combining visual-only retrieval and multimodal reranking, thereby enhancing the alignment between retrieved textual knowledge and visual content. However, this method risks losing the association and intrinsic knowledge of visual text due to the conversion of visual information into text. In contrast, our approach considers multimodal information in both the retrieval and reranking stages, thereby preserving the integrity of the knowledge base information more effectively.
this section cite: ['b22', 'b7', 'b16']

Section: Methodology
Humans always learn by examples. This cognitive process can be conceptualized as exploring isomorphic structures across diverse examples, thereby improving the extraction of heuristic insights (Van Gog & Rummel, 2010). Drawing inspiration from this cognitive paradigm, we hypothesize that LVLMs can similarly benefit from contextually relevant examples for in-context learning.
this section cite: ['b12']

Section: Problem Statement
Existing multimodal retrieval-augmented generation (RAG) techniques (Yan & Xie, 2024;Li et al., 2024) primarily address open-domain questions that LVLMs fail to answer without an external knowledge base. In contrast, we focus on scenarios where user queries fall within the scope of LVLMs' capabilities, albeit with potential inaccuracies. As shown in Fig. 2, LVLMs can take advantage of relevant examples retrieved from the knowledge base to obtain more precise and reliable responses.
this section cite: ['b16']

Section: Predicted Reasoning Context :
To determine the correct answer, we need to analyze the image and its historical context. 1. **Image Analysis**: The image depicts the Great Wall of China, a monumental structure that stretches across mountains and valleys. It is known for its historical significance in defending against invasions. 2. **Historical Context**: The Great Wall ...
this section cite: []

Section: …
The answer is D.
The answer is B.
this section cite: []

Section: B.
D.
The answer is D.  Knowledge Base. We define the knowledge base consisting of M visual question-answer pairs, denoted as D KB = {x i } M i=1 . Each x i encompasses an image I i , a question Q i , its corresponding reference answer A i , and an associated reasoning context C i (See Sec. 3.2). Formally, this can be expressed as x i := (I i , Q i , A i , C i ).
this section cite: []

Section: Goal.
Our framework leverages the user's query (I u , Q u ) to retrieve K pertinent question-answer pairs X ret = (x 1 , x 2 , ..., x K ) from the existing knowledge base D KB . Subsequently, the framework generates predicted answer ỹ using large vision-language model G:
ỹ ∼ G ([I u ; Q u ; X ret ]) , X ret ⊆ D KB .
(
)1
The goal is to develop a multimodal RAG framework that effectively integrates retrieved information with in-context learning to make the predicted answer ỹ align closely with the ground-truth response. It is worth noting that our framework is training-free and can be adaptively extended to multiple domains by simply expanding the knowledge base.
this section cite: []

Section: Reasoning Context with Self-Consistent Evaluation
Existing knowledge bases usually include visual questionanswer pairs without detailed reasoning procedures, which makes it difficult to provide valuable context for responses even if relevant examples are retrieved. To alleviate this issue, drawing from Auto-CoT (Zhang et al., 2022), we propose a method capable of automatically generating reasoning contexts for visual question-answer pairs to enhance contextual information during generation. We leverage a self-consistency mechanism of LVLMs to generate candidate reasoning contexts and utilize mutual answer prediction for reasoning context verification.
Specifically, as in Fig. i=1 . These predicted answers are evaluated with the ground truth answer A kb to obtain a set of prediction scores {Score i } Nc i=1 . Finally, the candidate reasoning context with the highest score is selected as the associated reasoning context.
this section cite: ['b21']

Section: Knowledge Retrieval with Hybrid Embeddings
As shown in Fig. 2, considering that both the knowledge base and user queries contain multimodal information, we employ hybrid-modal retrieval approaches rather than relying solely on a single modality. Following (Lin et al., 2023;2024b), given user's query consisting of an image I u and a question Q u , we first use a text encoder F L and an image encoder F I with linear function to obtain their embeddings with the same dimension d. The formulation is as follows:
E Tu = F L (Q u ) ∈ R l Tu ×d ; E Iu = F I (I u ) ∈ R l Iu ×d ,(2)
where l Tu and l Iu denote the total number of tokens of question Q u and image I u , respectively.
To enable hybrid-modal retrieval, all token-level embeddings are concatenated for retrieval, i.e.,
E u = [E Tu , E Iu ] ∈ R (l Tu +l Iu )×d .
Similarly, to maintain consistency with user queries, we utilize the same text questions and images, excluding answers from the knowledge base for the retrieval process. The knowledge base hybrid embeddings are defined as:
E KB = {E i } M i=1 = {[E Ti , E Ii ]} M i=1 .(3)
Self-Consistency Reward Finally, we compute the relevance score r between user queries embeddings E u and each knowledge base embeddings E KBi as follows:
r(E u , E i ) = lu j=1 li max k=1 E uj E i ⊤ k ,(4)
where l u = l Tu + l Iu and l i represent the number of tokens in hybrid embeddings, respectively. Therefore, the
final relevance scores r(E u , E KB ) = {[r(E u , E i )]} M i=1 .
The Top-N pertinent question-answer pairs X ret-N = (x 1 , x 2 , ..., x N ) are chosen by relevance scores r.
this section cite: ['b3']

Section: Re-ranking by Tree Search with Heuristic Rewards
This stage aims to re-rank the retrieved samples for selecting the most pertinent samples as the context prompt, facilitating efficient and accurate answer generation. Specifically, we adopt the Monte Carlo Tree Search (MCTS) methodology (Browne et al., 2012), a technique primarily employed in designing game-playing bots, to enhance the sample selection and re-ranking processes. Since MCTS can effectively balance exploring diverse samples and exploiting highquality ones through simulated trajectories, solving combinatorial optimization in context selection, we formulate the task as a sequential decision-making problem and propose a Monte Carlo Tree Search with Heuristic Reward (MCTS-HR) strategy, as shown in Fig. 4. A detailed workflow of our proposed MCTS-HR is provided in Appendix A.
Formally, we initialize a root node with a zero-shot response derived from the user's query. Then, existing nodes are ranked and selected for expansion using a greedy sampling strategy based on visit times N (a) and node values Q(a). During node expansion, action is sampled from an action space constructed from the retrieval samples X ret . When the maximum depth is reached, the algorithm performs a simulation by concatenating actions and the user query to form a K-shot prompt, generating a response for evaluation. This response is then assessed with a reward function R, and the reward value Q is backpropagated to update the tree's value information. Following the standard MCTS procedure, the upper confidence bound for trees (UCT) values of all nodes are then updated to guide further exploration. The algorithm iterates through these stages, re-ranking retrieved samples and refining responses until a termination condition, such as a maximum number of rollouts (referring to the number of simulations) or an early stopping strategy, is met. Below, we introduce the key elements of our algorithm.
this section cite: ['b4']

Section: Actions Construction and Selection.
Unlike most MCTSbased methods (Zhang et al., 2024a;Qi et al., 2024) in LLMs that rely on human-defined prompts as actions to construct the tree, our approach employs question-answer pairs retrieved from the external knowledge base as candidate actions A. Formally, we employ hybrid embeddings to retrieve the N most relevant question-answer pairs, where N ≫ K, and constitute the full action space:
A = {[x 1 , s 1 ], [x 2 , s 2 ], ..., [x N , s N ]},(5)
where x i = (I i , Q i , A i , C i ), s i denotes the normalized similarity score between the retrieved pair x i and user's query.
During the node expansion stage, let C ⊂ A be the set of actions that have already been selected (i.e., actions in the parent node). The remaining valid actions available for sampling are A valid = A \ C. Then, MCTS takes an action a i ∼ P (a i ) from the action space A valid using similaritybased probability distribution:
P (a i ) = s i j,aj ∈A valid s j ,(6)
where the selected action a i serves as the re-ranked example x i , and this process continues iteratively until it reaches its maximum depth K, thereby completing a branch of the MCTS. Finally, the sequence of K actions extracted from the current branch is concatenated with the user's query to form a K-shot prompt, thereby completing a branch simulation and obtaining the response for this branch.
this section cite: ['b8']

Section: Self-Consistency and Mutual Heuristic Rewards.
Another critical component of MCTS is the reward function R, which evaluates the value of each action and directs the tree expansion. Unlike the traditional MCTS-based LLM methods (Qi et al., 2024;Zhang et al., 2024a), which directly uses a language model as reward function R to score the node response, we propose a self-consistency heuristic reward strategy to get the self-reward value Q S alongside a mutual heuristic reward strategy to get the mutual-reward value Q M based on the in-context consistency.
For self-consistency heuristic reward strategy, assuming that the predicted K-shot response ỹi at branch i is denoted as ỹi = ( Ãi , Ci ). The user questions (I u , Q u ) and the prediction Ci are concatenated to generate multiple answers
{A (n) i } Ns n=1 . In theory, these answers A (n) i
should be consistent with the originally predicted answers Ãi . According to the above heuristic rules, the self-reward value Q S,i can be expressed as follows:
Q S,i = 1 N s Ns n=1 R Ã′(n) i , Ãi ,(7)
where For mutual heuristic reward strategy, we posit that if the answer to a question is correct, it will positively contribute to other questions, and vice versa. Therefore, we greedily pick N m samples {(I n , Q n )} Nm n=1 from the actions space A to serve as subsequent mutual heuristic samples. For branch i, we utilize the user's question and the predicted response ỹi as contextual prompts, with the selected N m samples' questions as the reference question and its corresponding answer as the ground truth answer A gt(n) i
A (n) i ∼ G [I u ; Q u ; Ci ],
. The predicted answer Ã(n) i for the reference question should be consistent with the ground truth answer. Thus, the mutual-reward value Q M,i can be represented as:
Q M,i = 1 N m Nm n=1 R Ã(n) i , A gt(n) i ,(8)
where
Ã(n) i ∼ G ([I u ; Q u ; ỹi ; I n ; Q n ]).
And the final reward value Q i for each branch i consists of self-reward value Q S,i and mutual-reward value Q M,i with a weighted summation as: where α is a weighting parameter that controls the importance of the self-reward and mutual-reward values. More details in Section 4.4.
Q i = α • Q S,i + (1 -α) • Q M,i ,(9)
Reward Backpropagation. After obtaining the reward value Q, we then propagate this reward value to its parent and ancestor nodes. Formally, if the reward value of any element in the child node set Children(p) changes, the reward value of the parent node Q(p) is updated to:
Q ′ (p) = 1 2 Q(p) • N (p) + Q(c) N (p) + 1 + max i∈Children(p) Q(i) ,(10)
where N (p) denotes visit times of the parent node p. Q(c) represents the reward value of the changed child node c. max i∈Children(p) Q(i) represents the highest quality value among all child nodes of parent node p.
This formula takes into account not only the reliability of the answers of all child nodes in the parent node p, but also the reward value of the answer of the most outstanding child.
this section cite: ['b8']

Section: Experiments

this section cite: []

Section: Datasets
In our experimental benchmark, we carry out comprehensive experiments with three common reasoning VQA datasets in extensive domains, including ScienceQA (Lu et al., 2022), MMMU (Yue et al., 2024) and MathV (Wang et al., 2024a). Additionally, we compare methods on simpler, nonreasoning VQA datasets using VizWiz (Gurari et al., 2018) and VSR-MC (Liu et al., 2023). Following the original splits of these VQA datasets, we construct the knowledge base with the training set and build the evaluation set with the testing set, respectively. Tab. 1 presents the size statistics of the knowledge base and the evaluation set. Please refer to Appendix B for details and examples of the datasets.
this section cite: ['b18', 'b22']

Section: Implementation Details
The proposed framework is applicable to mainstream LVLMs, thus we evaluate our method on various LVLMs across different scales and types, such as  Qwen2-VL (2B/7B) (Wang et al., 2024c), and InternVL-2 (8B) (Chen et al., 2024). Both models support multi-image input, enabling prompt concatenation with multi-image context. For efficiency, LVLMs with over 7B parameters are implemented in 4-bit quantization by AWQ (Lin et al., 2024a) on a single 4090 24GB GPU. Besides, we utilize the frozen BERT-base model and the ViT-L followed by a 2-layer MLP both adapted from PreFLMR (Lin et al., 2024b) as our text and vision encoders, respectively. For the setting of multiple rounds of LVLMs generation, we set N c = N p = 10, N s = N m = 5. For the setting of our MCTS-HR, we adopt the same number of few-shot samples with K = 3, i.e., a maximum tree depth of 3. The number of initial retrieval examples is set to N = 20 as the action space of MCTS-HR.
The maximum width of the tree is set to 3 for more action exploration. We set the default rollouts with P = 10, and the reward weight with default α = 0.2.
Section 4.4 details ScienceQA MMMU MathV 20 40 60 80 Datasets Accuracy (%) (a) Reward influence on different datasets. Self-Reward Mutual-Reward Hybrid-Reward 4 6 10 18 34 20 40 60 80 89.7 90.2 91.4 91.9 92.2 51.3 52.0 53.3 53.3 54.0 24.0 25.7 29.0 28.6 29.2 Number of rollouts Accuracy (%) (b) Rollout influence on different datasets ScienceQAtest MMMU-Dev MathVtestmini more discussion about these parameters.
this section cite: ['b7']

Section: Main Results
Tab. 2 demonstrates the comparison results with representative methods using various LVLMs on reasoning VQA datasets, including ScienceQA, MMMU, and MathV. As in Tab. 2, Vanilla-RAG (top retrieval) (Lin et al., 2024b) has achieved a performance improvement compared to both Zero-Shot and In-Context Learning (ICL) (Han et al., 2023) with random retrieval examples on most datasets, suggesting that semantic-aware example selection is crucial for LVLMs' reasoning. In particular, our proposed RCTS demonstrates substantial gains across all benchmarks. Notably, for Qwen2-VL (2B), RCTS achieves 78.99% on ScienceQA, surpassing both Zero-Shot by +11.81% and Vanilla-RAG by +7.05%. The improvements are even more pronounced in the mathematical reasoning dataset, RCTS elevates Qwen2-VL (7B) from 24.67% (Vanilla-RAG) to 28.95%, establishing new state-of-the-art results. Additionally, we evaluate non-reasoning VQA datasets with VizWiz and VSR-MC. Given that responses in these datasets User Question In a rectangle JKLM the angle bisector in J intersects the diagonal KM in N. The distance of N to LM is 1 and the distance of N to KL is 8. How long is LM? Choices: (A) 8 + 2 2. (B) 11 -2 (C) 10 (D) 8 + 3 2
Predicted Answer:
The answer is B.
BECAUSE: Let's denote the length of LM as x. Since the angle bisector in J intersects the diagonal KM in N, we have two similar triangles, JNL and KMN. By the property of ...
this section cite: []

Section: Predicted Answer:
The answer is A.
BECAUSE: Let's analyze the given information and use geometric principles to solve the problem. 1. **Identify the given information:** -Rectangle 𝐽𝐾𝐿𝑀 ... Question: The quadrilat eral 𝐴𝐵𝐶𝐷 with side length 4cm has the same area as triangl e 𝐸𝐶𝐷. What is the perpendicular distance from point 𝐸 to t he li ne 𝑔? Choices: (A) 8 cm (B) 4 + 2 3 cm (C) 12 cm (D) 10 × 2 cm (E) It depends on the position of 𝐸.
Answer: The answer is C.
this section cite: []

Section: #1
Question:
One of the two si des of a rectangl e has length 6 cm. In the rectangle circl es are drawn next to each other i n such a way that their centres form an equilateral triangle. W hat is the shortest dist ance between t he two grey circl es (in cm)? Choices: (A) 1 (B) 2 (C) 2 3 -2 (D) 𝜋/2 (E) 2
Answer: The answer is C.
this section cite: []

Section: #2 Question:
Two v ertices of a square lie on a semi-circle as shown, while the other two li e on i ts di ameter. The radius of the circle is 1 cm. How bi g is the area of the square?
Choices: (A) 0.8 𝑐𝑚 2 (B) Τ 𝜋 4 𝑐𝑚 2 (C) 1 𝑐𝑚 2 (D) Τ 4 3 𝑐𝑚 2 (E) 2 3
Answer: The answer is A.
#3 Question:
Two ci rcles have their centres on the same diagonal of a square. They touch each other and t he sides of t he square as shown. The square has side length 1 cm. What is the sum of the radii of the circl es in centimetres?
Choices:
(A)
1 2 (B) 1 2 (C) 2 -1 (D) 2 -2
Answer: The answer is D.
this section cite: []

Section: #5
Question:
One of the two si des of a rectangl e has length 6 cm. In the rectangle circl es are drawn next to each other i n such a way that their centres form an equilateral triangle. W hat is the shortest dist ance between t he two grey circl es (in cm)? Choices:
(A) 1 (B) 2 (C) 2 3 -2 (D) 𝜋/2 (E) 2
Answer: The answer is C.  typically consist of a single word or a brief sentence, we only introduce the knowledge base without reasoning context. As presented in Tab. 3, our approach demonstrates consistent effectiveness with +1.61% and +3.05% enhancements on VizWiz and VSR-MC respectively compared to Vanilla-RAG, confirming its versatility and robustness.
this section cite: []

Section: Ablation Study Key Components.
To validate the effectiveness of key components in our RCTS, we separately eliminate the reasoning context and MCTS-HR evaluating on various VQA datasets. As shown in Tab. 4, using MCTS-HR or Reasoning Context alone always has a positive effect, such as MCTS on ScienceQA (+2.24%) and Reasoning Context on MathV (+2.3%) in Qwen2-VL (7B). The same model applies to the following. Our full method with both MCTS and reasoning context achieves better performance across all datasets, suggesting that the designed two mechanisms complement and enhance each other. Besides, the wide variety of questions covered by the MMMU results in limited performance improvement (+3.33%), attributable to an insufficient number of analogous samples within the knowledge base.  hybrid-reward on three datasets. Obviously, using hybridrewards performs best on all three datasets, validating our design intent. Besides, as shown in Tab. 5, we perform a sensitivity analysis on the importance weight α of hybrid rewards and the default value for α was set to 0.2.
this section cite: []

Section: Rewards in MCTS.

this section cite: []

Section: Different Rollouts.
The number of rollouts is an important factor in RCTS performance. Fig. 5 (b) shows the performance on the three datasets with different rollouts. It can be seen that as the number of rollouts increases, the performance on the three datasets shows a consistent trend. We finally set the rollouts with P = 10 to balance the computational overhead and performance.
this section cite: []

Section: Discussion
Reliability of Reasoning Context. Tab. 6 demonstrates the reliability and accuracy of the reasoning context generated by our self-consistency evaluation strategy. Specifically, we evaluate the accuracy of the ground-truth answer with the predicted answer, which is generated by splicing the question and the reasoning context into a prompt that yields the corresponding answer. As illustrated in Tab. 6, the generated reasoning context provides precise and comprehensive responses for simpler datasets like ScienceQA. For more complex questions, our strategy still yields a substantial proportion of correct reasoning context. These results underscore the effectiveness of our proposed method.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-3 technical report: A highly capable language model locally on your phone Year: (2024)
Ref_id:b1 Title: Gpt-4 technical report Year: (2023)
Ref_id:b2 Title: Flamingo: a visual language model for fewshot learning Year: (2022)
Ref_id:b3 Title: Qwen-vl: A frontier large visionlanguage model with versatile abilities Year: (2023)
Ref_id:b4 Title: A survey of monte carlo tree search methods Year: (2012)
Ref_id:b5 Title: Wiki-llava: Hierarchical retrieval-augmented generation for multimodal llms Year: (2024)
Ref_id:b6 Title: Multimodal retrieval-augmented generator for open question answering over images and text Year: (2022)
Ref_id:b7 Title: Retrieval augmented thought process for private data handling in healthcare Year: (2024)
Ref_id:b8 Title: Mutual reasoning makes smaller llms stronger problemsolvers Year: (2024)
Ref_id:b9 Title: Mastering the game of go with deep neural networks and tree search Year: (2016)
Ref_id:b10 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b11 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b12 Title: Example-based learning: Integrating cognitive and social-cognitive research perspectives Year: (2010)
Ref_id:b13 Title: Measuring multimodal mathematical reasoning with mathvision dataset Year: (2024)
Ref_id:b14 Title: Learning to retrieve incontext examples for large language models Year: (2024-03)
Ref_id:b15 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b16 Title: Advancing visuallanguage models with wiki knowledge Year: (2024)
Ref_id:b17 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b18 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b19 Title: Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b Year: (2024)
Ref_id:b20 Title: On the out-of-distribution generalization of multimodal large language models Year: (2024)
Ref_id:b21 Title: Automatic chain of thought prompting in large language models Year: (2022)
Ref_id:b22 Title: Empowering vision-language model with multi-modal in-context learning Year: (2023)
