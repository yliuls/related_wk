Title: Reasoning Planning for Language Models
Abstract: Selecting an appropriate reasoning method for a given query remains a key challenge in language model generation. Existing approaches typically generate multiple candidate responses and use an aggregation strategy to select the output answer, often assuming that more candidate answers yield higher accuracy. We revisit this assumption through a rigorous theoretical analysis, deriving accuracy bounds for standard aggregation methods under fixed generation distributions and candidate sizes. Building on these insights, we introduce EPIC, an Ensemble PlannIng with Contrastive learning framework to learn a shared representation space that captures both model reasoning abilities and query-method compatibility. EPIC incorporates our probability bounds as a regularizer in a utility-driven optimization that balances accuracy and computational cost. Experiments on diverse mathematical reasoning tasks show that EPIC consistently selects optimal reasoning methods, improving accuracy while reducing computational overhead. Our code can be found at https://github.com/nguyenngocbaocmt02/EPIC.

Section: Introduction
Large Language Models (LLMs) have demonstrated remarkable abilities to understand and reason in human natural language. These advancements have transformed applications, including travel planning [Xie et al., 2024a], AI teaching platforms [Jin et al., 2024], and human population simulations [Park et al., 2023, Bui et al., 2025]. However, even with a pre-trained LLM, the computational expense of serving LLM-powered systems remains a significant bottleneck due to the massive scale of the models [Lin et al., 2024], the quadratic complexity of the attention mechanism [Dao et al., 2022], and the token-by-token nature of auto-regressive generation [Zhang et al., 2025]. This high computational cost significantly hinders the broader application of LLMs in practical scenarios, particularly in resource-constrained environments such as edge devices, real-time applications, and small-scale businesses. This challenge becomes even more pronounced in tasks that require advanced reasoning, such as automated theorem proving [Wu et al., 2022], mathematical problem solving [Trinh et al., 2024], code generation [Jiang et al., 2024, Li et al., 2025a], or heuristic discovery [Romera-Paredes et al., 2024]. LLMs often fail to produce accurate responses in these scenarios in a single pass. Instead, they rely on iterative generation strategies combined with aggregation or search techniques, such as best-of-N sampling [Stiennon et al., 2020] or Monte Carlo Tree Search [Xie et al., 2024b], to refine and select the most appropriate response. Throughout this paper, we refer to these iterative strategies as Reasoning Methods.
A key limitation of current approaches lies in their static application of reasoning methods, where the same technique is applied uniformly across all user queries. However, not all reasoning methods are equally effective or efficient for every query. This observation leads to our central research question: Could we select the most suitable reasoning method for a given user query to balance the trade-off between accuracy and efficiency before generating the answer?
As a starting point, we consider a universe of methods, denoted by M. Each technique in M is formally characterized by a tuple (LM, ReStrat, Agg, Config, N ), where LM denotes the base language model, ReStrat denotes the reasoning strategy (e.g. Monte Carlo Tree Search, Beam Search, Best-of-N), Config is a collection of relevant configuration parameters (e.g., temperature of a sampling-based decoding strategy), Agg denotes an aggregation technique (e.g., majority voting or score-based voting), and N is the number of candidate answers for aggregation. Importantly, this formulation is broad enough to subsume a wide variety of test-time compute methods [Snell et al., 2025], ranging from simple prompting techniques [Wei et al., 2022, Yao et al., 2023, Brown et al., 2020] and standard decoding strategies [Xie et al., 2024b, Wang et al., 2022] to more specialized intervention-based approaches [Li et al., 2023, Nguyen et al., 2025a,b]. However, in this work, we focus on a representative subset of methods rather than exhaustively covering the entire space.
Contributions. We introduce EPIC, an Ensemble PlannIng with Contrastive learning framework that recommends matching an input question and an appropriate reasoning method in the universe of methods M. EPIC learns jointly the embedding of each reasoning method and a neural mapping from the input question to the embedding space. Two main components guide the learning process:
• a contrastive loss, which pulls the question embedding towards the reasoning method with the highest utility for that question. The utility value is composed of a weighted combination of the accuracy and the inference cost, measured by the number of tokens generated. The user controls the accuracy-cost trade-off through a scalar parameter, balancing the preferences across different conflicting deployment criteria.
• a regularizer term, which exploits the commonality among methods that share four components (LM, ReStrat, Config, Agg), but differ only by the number of candidate answers N . This regularizer aims to improve the sample efficiency of the training procedure by grounding these methods relatively on the scale of N .
At inference time, EPIC maps the test-time input question to the embedding space and selects the reasoning method with the highest similarity (or scores) for answer generation. Extensive experiments on the MATH dataset demonstrate EPIC's advantage: compared to individual reasoning models in the universe of methods, EPIC can reduce the number of tokens (or cost) by 75% while maintaining the same level of accuracy.
Our paper unfolds as follows: Section 2 discusses related work on LLM reasoning. Section 3 studies the probabilistic bounds of common aggregation methods. Section 4 delineates our EPIC framework for matching reasoning methods with input questions, and Section 5 presents the extensive numerical results of the mathematical reasoning task.
this section cite: ['b12', 'b26', 'b3', 'b19', 'b6', 'b36', 'b30', 'b11', 'b27', 'b29', 'b28', 'b35', 'b41', 'b2', 'b15']

Section: Related Work
We review advances in LLM reasoning algorithms and inference-time scaling, highlighting their emerging impact on output quality and computational efficiency.
this section cite: []

Section: Reasoning algorithms and inference-time scaling.
A naive reasoning process may not generate the correct solutions for complex reasoning tasks. To identify and choose the correct solution within the distribution, Self-Consistency (SC) samples multiple outputs from the LLM and selects the final response by majority voting [Wang et al., 2022]. Another similar approach is best-of-N sampling, which uses a reward model or function to choose the answer with the highest reward [Stiennon et al., 2020]. Both methods enhance the quality of the output, but increase the computational cost by a factor of sampling times. To explore potential reasoning paths, tree-search-based methods are proposed, such as Tree-of-Thought [Yao et al., 2024], Monte Carlo Tree Search (MCTS) [Wan et al., 2024, Zhang et al., 2024, Guan et al., 2025], Forest-of-Thought [Bi et al., 2025]. Damani et al. [2025] indicates that searching over a tree structure is more effective in discovering a correct solution than simply sampling responses in parallel for more complex tasks.
Despite applying different reasoning methods to problems with various levels of complexity, inferencetime scaling on these methods also significantly improves the output quality. Beeching et al. [2024] demonstrates that the accuracy on the MATH-500 benchmark improves as the amount of test-time computation (number of generations per problem) increases for algorithms such as best-of-N, beam search, and diverse verifier tree search (DVTS). Guan et al. [2025] conduct extensive MCTS rollouts and achieve an average accuracy of 53.3% on 15 questions of AIME24 benchmark.
Cost-effective reasoning. Although inference-time scaling significantly enhances LLM's reasoning capabilities, this approach incurs substantial computational overhead and often leads to inefficient use of computational resources. Recent work finds that performance gains from various inference-time scaling strategies exhibit significant variability across different levels of prompt difficulty [Snell et al., 2025]. Drawing from this evidence, they effectively allocate inference-time compute according to question difficulty, with four times less computation than the best-of-N baseline. However, the method incurs considerable computational costs to assess question difficulty. Damani et al. [2025] train lightweight probes built upon LLM's hidden representations to quickly predict if allocating more computation to a question will improve the response quality. To efficiently scale best-of-N sampling, Manvi et al. [2024] introduces a highly cost-effective self-evaluation paradigm that does not rely on an external reward model, incurring costs only from generating a single token.
Whereas most studies focus on effectively and efficiently scaling a particular reasoning algorithm, we focus on pairing suitable reasoning methods with various questions, considering both accuracy and cost. We conduct our study based on OpenR [Wang et al., 2024a], an open-source framework for LLM reasoning that integrates multiple strategies, including greedy decoding, best-of-N, beam search, and MCTS.
this section cite: ['b34', 'b29', 'b42', 'b31', 'b7', 'b1', 'b5', 'b0', 'b7', 'b28', 'b5', 'b22']

Section: Probabilistic Analysis of Aggregation Accuracy
We observe that many methods in the universe M could share common features: they could use the same base language model, reasoning strategy, configuration parameters, and aggregation methods, and they could differ by only the amount of test-time compute, or how many samples N they need to generate before aggregation. To exploit this information, we first need to understand how different sample sizes N affect the quality of the output. We analyze the probabilistic performance of an aggregation method for a specific question q as the number of samples N varies. All probability quantities in this section are conditioned on q, but this condition is omitted to avoid clutter. Let Ỹ be a random variable representing the final answer extracted from a sampled solution to a question q generated by a model. Importantly, Ỹ refers specifically to the final answer, not the reasoning process or steps leading to it. In practice, the model is trained to enclose Ỹ in a LaTeX box to make extraction easier. We suppose that the support set of Ỹ is finite: Y = {y 1 , y 2 , . . . , y K }. The stochastic generation process of a model specified by the tuple (LM, ReStrat, Config) produces a probability distribution over Y:
Pr( Ỹ = y k ) = p k , where K k=1 p k = 1 and p k ≥ 0 ∀k.
Without any loss of generality, we denote y 1 ∈ Y as the only correct answer to the question q. After sampling N independent samples following the above distribution, an aggregation method Agg is applied to obtain the output answer. We focus on characterizing the probability that the output answer is y 1 , which means that the output answer is a correct solution to question q.
this section cite: []

Section: Majority Voting
Given N samples generated by the language model, majority voting counts the frequency of each unique answer among N candidate answers. Then it outputs the answer with the highest count as the output answer. We refer to this aggregation method as Majority Vote. We have the following result. Theorem 3.1 (Majority voting). If p 1 > p k for all k = 2, . . . , K, then
Pr(Majority Vote picks y 1 ) ≥ 1 - K k=2 e -N ( √ p1- √ p k ) 2 . (1a
)
If p 1 < p k for some k = 2, . . . , K, then
Pr(Majority Vote picks y 1 ) ≤ e -N ( √ p k - √ p1) 2 . (1b
)
Note that the bound (1a) approaches 1 as N → ∞, which implies perfect accuracy. In contrast, the bound (1b) approaches 0 as N → ∞, implying a complete failure.
this section cite: []

Section: Aggregation using Summation of Scores
Given N samples generated by a model, we first pass them through a reward model to obtain a reward score for each sample. Two popular types of reward models are Outcome Reward Models (ORM) [Cobbe et al., 2021, Yu et al., 2023], which provide a single scalar reward for each complete solution trajectory, and Process Reward Models (PRM) [Luo et al., 2024, Lightman et al., 2023a], which provide step-by-step feedback and aggregate it, typically by summing or taking the minimum, to obtain a final score for the sample. While PRM is used throughout this work, our method is flexible and can be applied with any reward models.
For each unique answer, we sum the PRM scores of all samples that generate that answer. The final outcome is selected as the answer with the highest total (summed) reward score across all samples [Wang et al., 2024a, Li et al., 2022]. We suppose that PRM returns a score for answer y k following a Gaussian distribution N (µ k , σ 2 k ) for all k. We call this aggregation method PRM Vote. We have the following result. Theorem 3.2 (Voting with score sum). If p 1 µ 1 > p k µ k for all k = 2, . . . , K, then
Pr(PRM Vote picks y 1 ) ≥ 1- K k=2 inf t k >0 exp N p 1 e -t k µ1+ 1 2 t 2 k σ 2 1 -1 + N p k e t k µ k + 1 2 t 2 k σ 2 k -1 . (2a) If p 1 µ 1 < p k µ k for some k = 2, . . . , K, then Pr(PRM Vote picks y 1 ) ≤ inf t>0 exp N p k e -tµ k + 1 2 t 2 σ 2 k -1 + N p 1 e tµ1+ 1 2 t 2 σ 2 1 -1 . (2b)
All infimum problems in (2) are convex optimization problems. While no analytical expression for the optimal value t is available, we could tractably find t k for each term using Newton's method. Moreover, we could observe a similar conclusion as N tends to infinity: the bound (2a) approaches 1 while the bound (2b) approaches 0.
this section cite: ['b4', 'b41', 'b21']

Section: Aggregation using Maximum of Scores
This aggregation method follows the same setup as in Section 3.2: given N samples, we use the PRM to assign a reward score to each sample. For each unique answer, we take the maximum PRM score among all samples that produce that answer. The final prediction is the answer with the highest such maximum. We suppose that PRM returns a score for answer y k following a Gaussian distribution N (µ k , σ 2 k ) for all k. We call this aggregation method PRM Max. We have the following result. Theorem 3.3 (Voting with score maximum). Let
Φ k (t) := Φ t -µ k σ k , k = 1, . . . , K,
where Φ is the cumulative distribution function of the standard normal distribution.
If σ 1 > σ k for all k = 2, . . . , K, then
Pr(PRM Max picks y 1 ) ≥ 1 - K k=2 inf t∈R (1 -p 1 [1 -Φ 1 (t)]) N + 1 -(1 -p k [1 -Φ k (t)]) N . (3a) If σ k > σ 1 for some k = 2, . . . , K, then Pr(PRM Max picks y 1 ) ≤ inf t∈R (1 -p k [1 -Φ k (t)]) N + 1 -(1 -p 1 [1 -Φ 1 (t)]) N .(3b)
All infimum problems in (3) are one-dimensional and can be efficiently solved using standard numerical methods. Moreover, we observe similar asymptotic behavior as N increases: if σ 1 > σ k for all k = 2, . . . , K, the bound in (3a) approaches 1 as N → ∞, while if σ k > σ 1 for some k, the bound in (3b) approaches 0 as N → ∞.
this section cite: []

Section: Ensemble Planning with Contrastive Learning
Given a universe of methods M = {1, . . . , M } consisting of M reasoning methods in total, EPIC aims to create an ensemble model on M that assigns to any input question x from the test environment an appropriate method i ∈ M that could deliver a desirable accuracy-cost trade-off. We first discuss our modeling of the accuracy-cost trade-off in Section 4.1, then we describe the training phase and inference phase in Sections 4.2 and 4.3. We conclude this section by discussing our design choices.
this section cite: []

Section: Accuracy-Cost and Utility
We possess a training dataset of n question-answer pairs denoted as D = {x j , y j } n j=1 , where x j is a question statement, and y j is the corresponding true answer. In the training phase, we deploy a reasoning model Φ i , i ∈ M, to each question x j . The generated solution is Φ i (x j ). We record whether Φ i (x j ) is accurate by comparing it to the ground-truth answer y j , and obtain the accuracy signal
a i,j = Accuracy(Φ i (x j ), y j ) ∈ [0, 1].(4)
If Φ i is a deterministic method, then (4) is a simple binary indicator Accuracy(Φ i (x j ), y j ) = 1(Φ i (x j ) = y j ). When Φ i is a stochastic method, then we average the accuracy over five seed numbers to get a percentage accuracy. The value a i,j indicates whether method i succeeds in answering question j. Moreover, we also record how many tokens the method i ∈ M costs to generate the answer. This token count is denoted by ci,j > 0. Because a i,j ∈ [0, 1], we normalize the token count by passing ci,j through a non-decreasing function ϕ : R + → R + , then dividing by the maximum transformed cost to ensure that the cost c i,j ∈ [0, 1] has the same scale with a i,j :
c i,j = ϕ(c i,j ) max i ′ ∈M (ϕ(c i ′ ,j ))
.
To balance cost and success rate, we establish the utility function that is the convex combination of accuracy a i,j and normalized cost c i,j as u(a i,j , c i,j ) = λa i,j + (1 -λ)(1 -c i,j ),
where λ ∈ [0, 1] is a trade-off parameter, and the utility admits a value between 0 and 1. If λ = 0, then u(a i,j , c i,j ) = 1 -c i,j , which implies that the utility depends only on the generation cost. In this way, we tend to favor the cheapest reasoning method, regardless of how effective it is at generating accurate answers. On the other end of the spectrum, when λ = 1, u(a i,j , c i,j ) = a i,j , implying that the utility is entirely derived from the accuracy. In this way, we tend to favor the most powerful reasoning method, regardless of its cost. To simplify the notation, we omit the parameter λ, and use the shorthand u i,j = u(a i,j , c i,j ).
The product of the data preparation process is a processed dataset {x j , (u i,j ) i∈M } n j=1 containing the training question and the corresponding utility of each reasoning method for that question. This dataset will be used in the subsequent contrastive learning process.
this section cite: []

Section: Contrastive Representation Learning with Probability Regularization
We now describe the core component of our framework that matches the input question with the appropriate reasoning method. We represent each question x j in the training dataset by its features f j ∈ R D . A lightweight neural network g θ : R D → R d maps each question feature vector f j to produce a dense embedding g θ (f j ) in a d-dimensional vector space. EPIC aims for an information compression with d ≪ D. Moreover, each reasoning method i ∈ M is assigned a trainable embedding vector v i ∈ R d , which is the same dimension as the question embeddings. EPIC uses a simple multi-layer perceptron for θ.
We now train the question embedding network parameter θ and the method embedding vectors v i jointly. One component in the training loss is the popular contrastive loss function, InfoNCE loss [Oord et al., 2018]. We identify a positive method for each question x j , denoted as m + (x j ). Given the utility values defined in (5), we can identify the method with the highest utility for question x j : m + (x j ) = arg max i∈M u(a i,j , c i,j ), Evaluate Four distinct circles are drawn in a plane. What is the maximum number of points where at least two of the circles intersect?
Reasoning method 1 ...
this section cite: ['b25']

Section: Reasoning method 2
Reasoning method
this section cite: []

Section: Reasoning Method Embeddings
Find the constant term in the expansion of
this section cite: []

Section: Loss Loss Loss
Figure 1: Our method employs the regularized representation learning loss (8) to learn both the reasoning method representation vectors, denoted as v 1 , . . . , v M , and the question embedding network parameters θ. During inference, we route suitable math questions to the appropriate reasoning method by computing the similarity between the input questions and the learned method representations.
Color codes on problem difficulty levels are provided for illustration purposes only.
This leads to the contrastive loss component:
ℓ contrastive (θ, v 1 , . . . , v M ) = 1 n n j=1 -log exp(s(g θ (f j ), v m+(xj ) ) i∈M exp(s(g θ (f j ), v i )) .(6)
Above, s is a similarity score function that measures the similarity of a question embedding g θ (f j ) with the method embedding v i . Standard choices for s are the dot product similarity measure or the negative 2-norm. The contrastive component (6) aims to pull g θ (f j ) close to the positive method v m+(xj ) , and push g θ (f j ) far away from the negative methods i ̸ = m + (x j ). The loss in (6) is the categorical cross-entropy loss of classifying the positive method, with the fraction inside the logarithm being the model's prediction.
The second component of the loss function is a regularization term: Two methods that share the same tuple (base model, reasoning strategy, aggregation technique and configuration) but differ only by the compute budget N should conform to a relative performance metric because they both inherit the same stochastic generator. We postulate the following regularization term:
ℓ reg (θ, v 1 , . . . , v M ) = 1 n n j=1 (i,i ′ )∈M (i,i ′ ) differ only by N s(g θ (f j ), v i ) s(g θ (f j ), v i ′ ) - target j i target j i ′ 2 .(7)
This regularizer promotes the fraction of the similarities to be close to the fraction of the target quantities. Ideally, we should use target i j = Pr(method i picks the correct answer for question x j ), which is the intrinsic characteristic of the stochastic generator. However, this probability value is not readily available, therefore we leverage the bounds in Section 3 as target values, and empirically compute these target values as follows: For each question j and core configuration (generation method, temperature, aggregation method, etc.), we generate 80 solutions (5 independent runs of N = 16 with different seed numbers) to obtain a set of distinct solutions y 1 , . . . , y K . We then estimate the parameters pk , μk , σk from these 80 solutions. We can then empirically identify whether the lower or upper bound of the probability is active and assign the target value as either the lower or upper bound with the corresponding size N . The bound provided in Theorem 3.3 is valid when N is large enough. For smaller N , we use the empirical accuracy as an alternative.
Combining two loss terms (6) and ( 7), we obtain the training problem
min θ min v1,...,v M ∈R d ℓ contrastive (θ, v 1 , . . . , v M ) + τ ℓ reg (θ, v 1 , . . . , v M ),(8)
where τ > 0 is a hyperparameter aiming to promote the sample efficiency of the training procedure.
this section cite: []

Section: Inference Time Matching
At inference time, we pass any new question x new , or equivalently its feature vector f new , through the trained network g θ to obtain the question embedding g θ (f new ). We then find the top-1 reasoning method by m ⋆ = arg max i∈M s(g θ (f new ), v i ), that maximizes the similarity score between the question and the trained representation vector v 1 , . . . , v M of the reasoning models. We then deploy method m ⋆ to answer this question.
this section cite: []

Section: Discussions
We now discuss the necessity and importance of the design choices of our EPIC method.
Discussion 1 (Questions' feature vector). There are multiple ways to obtain the feature vector f j for each question x j . For example, we can take f j as the activation of the last token of x j extracted from one of the layers (potentially the last layer) of the language model. This approach does not incur any additional memory requirement because we do not need to load any auxiliary models onto the device. However, the activation dimension of the language models is usually high: for example, in Qwen2.5-Math-7B-Instruct [Yang et al., 2024a], D = 3584. This high dimensionality could prohibit efficient training of the representation parameters θ. Alternatively, we can use a lightweight model to map x j to f j . This could incur additional memory overhead but generate a lower D as input to the network g θ . In the experiment, we will use a lightweight sentence embedding all-MiniLM-L6-v2foot_0 that has only 22.6 million parameters and incurs only 80MB of additional VRAM. The corresponding feature dimension is D = 384.
Discussion 2 (Importance of embedding network g θ ). Given the question features f 1 , . . . , f n ∈ R D , one could simplify the representation learning problem (8) by optimizing the method embedding vectors v 1 , . . . , v M directly on the space of R D . This is equivalent to setting d = D, and letting g θ collapse into an identity mapping. However, the proximity between two question features f j and f j ′ does not convey enough information about the similarity regarding hardness, resource utilization, and suitability with methods. Moreover, learning the method embedding v j on R D is more difficult than on the smaller dimension space R d . Hence, learning in R D is inefficient. This observation necessitates the use of a lightweight question map g θ .
Discussion 3 (Adaptive method insertion). Given a universe of models M, problem (8) optimizes one vector v i for a reasoning model i ∈ M. Alternatively, we could use another network h ϑ that could take a (text) description of a reasoning method and output the respective embedding vector in the representation space R d . Having the second network h ϑ could unlock several new capabilities: (i) for a new reasoning method that is not in M, we could quickly obtain its embedding and predict its performance on the questions, (ii) we could inverse engineer to design a better reasoning method. Unfortunately, training h ϑ requires a meaningful textual description of the reasoning methods. This is currently outside the scope of this paper, and we leave it for subsequent work.
this section cite: []

Section: Numerical Experiments
In this section, we present numerical experiments showcasing the performance of EPIC on the math answering task. Experiments for the code generation task are relegated to Appendix D.4.
Dataset. We use the MATH dataset [Hendrycks et al., 2021] as a training set, utilizing its training split of 7,500 math problems with solutions, as defined in Hendrycks et al. [2021]. For the code generation experiment, we use the LiveCodeBench dataset [Jain et al., 2025]. More details are in Appendix D.4. For evaluation, we test on the MATH500 test split, which contains 500 samples, as defined in Lightman et al. [2023b]. We also use the test set of the GSM8K [Cobbe et al., 2021] dataset to evaluate the transferability of the method embedding vectors learned in Section 4.2.
Base models. We employ Qwen2.5-Math-7B-Instructfoot_1 as our generation model, and math-shepherdmistral-7b-prm [Wang et al., 2024b] foot_2 as our reward model in the PRM framework. These models are fixed throughout our main experiments. For transferability experiments, we augment our universe of Our ensemble planner performances with varying λ ∈ {0, 0.25, 0.5, 0.75, 1} are highlighted in red, and individual reasoning models in M are plotted in blue. The boundary of the ensemble planners covers the individual models in the universe M. The Upper Bound (UB) under M is the proportion of questions that at least one method in M could successfully solve.
methods to include Qwen2.5-Math-1.5B. For code generation experiments, we use Qwen2.5-Coder-3B-Instruct and Qwen2.5-Coder-7B-Instruct.
Performance metrics. We use the accuracy to measure the quality of generation and average token counts to evaluate the efficiency of each method. For accuracy, we use the automatic gradingfoot_3 provided by previous work Lightman et al. [2023b] to evaluate the accuracy of a generated solution in (4). To measure average token counts, we set the hyperparameter 'max new token' to 2048 for all methods and compute the average number of tokens generated.
Universe of methods M. We generate M consisting of 81 distinct methods, spanning a variety of reasoning strategies, aggregation techniques, and parameter configurations. A complete description is provided in Appendix B.
this section cite: ['b8', 'b8', 'b10', 'b4']

Section: Dataset generation for contrastive learning.
For a deterministic method i in M, we run inference on each question x j once and record the accuracy and number of generated tokens. For the sampling method, we run the inference on each question five times to obtain a mean estimate of a i,j and c i,j .
Baselines. We compare EPIC against three categories of baselines: (i) individual reasoning methods from the universe M, (ii) strong large-model references including DeepSeek-V3 and OpenAI-o1mini, and (iii) alternative reasoning selection methods-RA, Offline Ada-BoK [Damani et al., 2025], DRA-λ, and CL-λ. All ensemble baselines and EPIC are trained and evaluated on the same M for fair comparison. Further details are provided in Appendix C.
this section cite: ['b5']

Section: Reproducibility.
All experiments are conducted on a single machine with 8× NVIDIA RTX A5000 GPU and Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.
this section cite: []

Section: Comparison between EPIC and Baselines
In the first experiment, we benchmark how EPIC, an ensemble model, outperforms individual reasoning models in the universe M. Table 1 presents the test performance comparison between our ensemble planner and individual reasoning methods, computed based on average accuracy and the number of generated tokens. We apply the regularization parameter τ = 10 -3 based on its better numerical results than other values shown in Appendix D. Our method with λ = 0.25 achieves an accuracy of 86.4%, matching the best-of-16 approach while using significantly fewer tokens: EPIC generates 1859.2 tokens while best-of-16 generates 10036.2 tokens. On a relative scale, this is a 5x reduction in the token counts at the same accuracy level. Compared to beam search, our approach at λ = 0.25 achieves better accuracy with a 29.5% reduction in token usage. With λ = 1.00, our method achieves the highest accuracy (89.4%) at a significantly lower token count (6,921.7).
To better visualize the performance of EPIC, we plot in Figure 2 a scatter plot locating the accuracycost trade-off of EPIC instances and representative reasoning models from M. Our EPIC instances (red) all lie on the frontier, thereby boosting the performance of the inference phase.
Since EPIC is an ensemble method constructed from the universe of reasoning methods M, the performance of EPIC is constrained by the capacity of the universe M itself. We could compute the best possible accuracy of the whole universe M on the test set: M could solve a question if there is at least one method from M that could generate a correct answer. Computing this value yields an upper bound of approximately 91.2%. Figure 2 highlights that our method provides a flexible trade-off between efficiency and accuracy, approaching the upper bound (dashed vertical line) while maintaining computational efficiency.
this section cite: []

Section: Transferability
We now examine the transferability of EPIC across both model scales and datasets. Table 2 summarizes results for two complementary settings: (a) transferring from the MATH to the GSM8K dataset, and (b) applying EPIC in a cost-aware multi-model environment with Qwen2.5-Math-1.5B and Qwen2.5-Math-7B.
this section cite: []

Section: Evaluating EPIC with Cost-Aware Multi-Model Reasoning
EPIC remains effective even when reasoning methods use heterogeneous base models. Previously, our universe M contained 81 methods built solely on Qwen2.5-Math-7B-Instruct. We now augment this space with an additional 81 methods using Qwen2.5-Math-1.5B-Instruct. Because larger models are computationally more expensive, we approximate the cost of each method as the product of its parameter count (in billions) and the number of generated tokens. This proxy aligns with real-world inference costs; alternatively, FLOPs or API pricing could be used.
Table 2b shows that EPIC adapts effectively across cost regimes. At λ = 0.25, EPIC achieves an accuracy of 86.2, matching the 7B-Best-of-4 method, while reducing the cost by over 50% (8047.8 vs. 17495.8). At λ = 1.0, EPIC attains the highest overall accuracy (89.0), outperforming 7B-Bestof-16 while maintaining roughly half the computational cost (35705.4 vs. 70253.4). These results demonstrate EPIC's capacity to balance accuracy and cost by dynamically leveraging reasoning methods from different model sizes.
this section cite: []

Section: Transfer to Another In-Domain Dataset
To further assess generalization, we evaluate EPIC trained on MATH and test it on GSM8K [Cobbe et al., 2021], another widely used arithmetic reasoning benchmark. As shown in Table 2a, GSM8K is a simpler dataset; hence, absolute gains are smaller. Nonetheless, EPIC achieves the best accuracy (95.0%) while requiring fewer tokens than high-cost baselines such as Best-of-8 or Best-of-16. This indicates that EPIC's learned representations transfer across related reasoning distributions and continue to yield efficient inference-time behavior.
this section cite: ['b4']

Section: Additional Experiments
We perform ablation studies to better understand the impact of the representation dimension, the regularization parameter τ , and the tradeoff parameter λ, the utility function, the transferability, and the generalization on the code generation task. In the experiment of representation dimension, we fix λ = 0.5 and vary dimension d ∈ {16, 32, 64, 128}. Our results show a general trend: as d increases, accuracy improves, while average token count stabilizes or slightly decreases. This result empirically confirms the expectation that increasing the embedding dimension could boost the performance of our method. In the experiment on the impact of λ, we observe a clear cost-accuracy trade-off as we fix d = 64 and varies λ ∈ {0.00, 0.25, 0.50, 0.75, 1.00}. In the ablation study of the utility function, we switch to an alternative functional form, as shown in equation ( 5), and observe a decrease in performance, indicating that our design choice is superior. Due to space constraints, further experimental details are provided in Appendix D.
this section cite: []

Section: Conclusion
We introduced EPIC, the Ensemble PlannIng with Contrastive learning framework, a contrastive learning framework that plans optimal reasoning strategies for language models by matching questions to suitable methods. Our analysis established new accuracy bounds for common aggregation techniques, which directly inform a regularization term to guide more sample-efficient learning. Experiments on mathematical reasoning benchmarks demonstrate that EPIC leverages these theoretical insights to achieve strong improvements in both accuracy and inference cost, showcasing the value of principled modeling for reasoning method selection.
Fei Yu, Anningzhe Gao, and Benyou Wang. Ovm, outcome-supervised value models for planning in mathematical reasoning. arXiv preprint arXiv:2311.09724, 2023. Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-MCTS * : LLM self-training via process reward guided tree search. In Advances in Neural Information Processing Systems, volume 37, pages 64735-64772, 2024. Xiang Zhang, Tianze Ling, Zhi Jin, Sheng Xu, Zhiqiang Gao, Boyan Sun, Zijie Qiu, Jiaqi Wei, Nanqing Dong, Guangshuai Wang, et al. π-PrimeNovo: an accurate and efficient non-autoregressive deep learning model for de novo peptide sequencing. Nature Communications, 16(1):267, 2025.
this section cite: []

Section: References
Ref_id:b0 Title: Scaling test-time compute with open models Year: (2024)
Ref_id:b1 Title: Forest-of-thought: Scaling test-time compute for enhancing LLM reasoning Year: (2025)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: Mixture-of-personas language models for population simulation Year: (2025-07)
Ref_id:b4 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b5 Title: Learning how hard to think: Input-adaptive allocation of LM computation Year: (2025)
Ref_id:b6 Title: Flashattention: Fast and memoryefficient exact attention with IO-awareness Year: (2022)
Ref_id:b7 Title: rStar-Math: Small LLMs can master math reasoning with self-evolved deep thinking Year: (2025)
Ref_id:b8 Title: Measuring mathematical problem solving with the MATH dataset Year: (2021)
Ref_id:b9 Title: Openai o1 system card Year: (2024)
Ref_id:b10 Title: Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code Year: (2025)
Ref_id:b11 Title: A survey on large language models for code generation Year: (2024)
Ref_id:b12 Title: Teach ai how to code: Using large language models as teachable agents for programming education Year: (2024)
Ref_id:b13 Title: S*: Test time scaling for code generation Year: (2025)
Ref_id:b14 Title: S*: Test time scaling for code generation Year: (2025)
Ref_id:b15 Title: Inference-time intervention: Eliciting truthful answers from a language model Year: (2023)
Ref_id:b16 Title: Making large language models better reasoners with step-aware verifier Year: (2022)
Ref_id:b17 Title: Let's verify step by step Year: (2023)
Ref_id:b18 Title: Let's verify step by step Year: (2023)
Ref_id:b19 Title: AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration Year: (2024)
Ref_id:b20 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b21 Title: Improve mathematical reasoning in language models by automated process supervision Year: (2024)
Ref_id:b22 Title: Adaptive inference-time compute: LLMs can predict if they can do better, even mid-generation Year: (2024)
Ref_id:b23 Title: Task-driven layerwise additive activation intervention Year: (2025-04)
Ref_id:b24 Title: Structured pruning for diverse best-of-n reasoning optimization Year: (2025-07)
Ref_id:b25 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b26 Title: Generative agents: Interactive simulacra of human behavior Year: (2023)
Ref_id:b27 Title: Mathematical discoveries from program search with large language models Year: (2024)
Ref_id:b28 Title: Scaling llm test-time compute optimally can be more effective than scaling parameters for reasoning Year: (2025)
Ref_id:b29 Title: Learning to summarize with human feedback Year: (2020)
Ref_id:b30 Title: Solving olympiad geometry without human demonstrations Year: (2024)
Ref_id:b31 Title: Alphazero-like tree-search can guide large language model decoding and training Year: (2024)
Ref_id:b32 Title: OpenR: An open source framework for advanced reasoning with large language models Year: (2024)
Ref_id:b33 Title: Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations Year: (2024)
Ref_id:b34 Title: Self-consistency improves chain of thought reasoning in language models Year: (2022)
Ref_id:b35 Title: Chain-of-thought prompting elicits reasoning in Large Language Models Year: (2022)
Ref_id:b36 Title: Autoformalization with Large Language Models Year: (2022)
Ref_id:b37 Title: TravelPlanner: A benchmark for real-world planning with language agents Year: (2024)
Ref_id:b38 Title: Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning Year: (2024)
Ref_id:b39 Title: Qwen2.5 technical report Year: (2024)
Ref_id:b40 Title: Math technical report: Toward mathematical expert model via self-improvement Year: (2024)
Ref_id:b41 Title: Tree of thoughts: Deliberate problem solving with Large Language Models Year: (2023)
Ref_id:b42 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2024)
