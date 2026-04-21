Title: INTRINSIC ENTROPY OF CONTEXT LENGTH SCALING IN LLMS
Abstract: Long Context Language Models have drawn great attention in the past few years. There has been work discussing the impact of long context on Language Model performance: some find that long irrelevant context could harm performance, while some experimentally summarize loss reduction by relevant long context as Scaling Laws. This calls for a more thorough understanding of how long context impacts Language Modeling. In this work, we (1) propose to use 'Intrinsic Entropy' for explaining the impact of context length on language modeling; and(2) conduct experiments on natural language and synthetic data, validating our proposed theoretical assumptions and deductions. Our theoretical framework can provide practical insights such as establishing that training dataset size dictates an optimal context length and bounds context length scaling for certain cases. We hope our work may inspire new long context Language Models, as well as future work studying the physics of Language Models. 1 † CPHOS is an academic non-profit organization. 1 Code for experiments is available at: https://github.com/JingzheShi/  NLPCtlScalingAndBounds.2 We discuss more about previous work in Appendix J.

Section: INTRODUCTION
As language-model capacity has rapidly increased and long context has become crucial for tasks such as reasoning and retrieval, recent work has focused on extending context length. A variety of methods have been proposed to support long-context language models (Su et al., 2023;Katharopoulos et al., 2020;Gu & Dao, 2024;Peng et al., 2023;Sun et al., 2024). At the same time, prior work reports mixed outcomes: some studies show that long irrelevant context worsens LM performance (Xu et al., 2024;Levy et al., 2024), some summarize gains from relevant long context as scaling laws (Xiong et al., 2024), and work in other domains such as time series shows that even relevant long context can hurt performance (Shi et al., 2024). These observations call for a more thorough understanding of how context length affects language-model performance.
Previous theories explain scaling laws with respect to dataset and model size (Bahri et al., 2024;Sharma & Kaplan, 2020;Chen et al., 2025). However, most of them do not study how context length impacts scaling laws for language modeling, thus they cannot contribute directly to the problem. 2   In this work, we analyze the impact of context length by decomposing the total loss into 2 components. As shown in Figure 1, the components are: the Bayes Risk, representing the loss of an optimal language model given certain context length, and the Approximation Loss, the loss caused by the gap between the optimal language model and the trained model. The total loss is the sum of these 2 components, the balance between which would possibly lead to seemingly counter-intuitive behaviors like longer context worsens the performance. Such analysis works for analyzing impact of context length on multiple scenarios, such as for pretraining targeting Cross-Entropy loss, or when evaluating down-stream metrics, as shown in Figure 1. Building on this decomposition, we introduce Intrinsic Entropy, which measures how much information is available to an LLM at a given context length for a dataset. Starting from simple assumptions about Intrinsic Space and information entropy, we derive the relationship among Cross Entropy Loss, Intrinsic Entropy, and Context Length. We then validate these assumptions and deductions on both natural language and synthetic data. Our main contributions are:
• We analyze context-length effects through the trade-off between Bayes Risk and Approximation Loss, showing that longer context does not always improve performance.
• We introduce Intrinsic Entropy and use it to explain language modeling behavior across different context lengths.foot_0 • We validate the theoretical assumptions and deductions with experiments on both real language data and synthetic data.
We hope our work may inspire future work when it comes to explaining context impact and/or training new long context Language Models.
this section cite: ['b30', 'b16', 'b10', 'b23', 'b31', 'b35', 'b19', 'b34', 'b29', 'b3', 'b27', 'b6']

Section: ASSUMPTIONS, DEDUCTIONS AND OBSERVATIONS FOR LANGUAGE MODELING
2.1 PRELIMINARIES 2.1.1 PRELIMINARY: LOSS DECOMPOSITION It is common in ML studies to decompose the loss into Bayes Risk (the minimum loss possible, achieved by the theoretically optimal Bayesian Model), and Approximation Loss (the loss measuring the ability of a trained model actually to approximate the Bayesian Model). Specifically for Cross-Entropy loss H, we have (please refer to Appendix B.1 for formal definitions and derivation details):
H(P, Q l ) =R Bayes + L Approx =H(P, P l ) + D KL (P l , Q l )(1)
Where P = p(x 0 |x -∞:0 ) is the distribution of Natural Language (or our experimented dataset), P l = p(x 0 |x -l:0 ) is the Bayesian Model for context length l and Q l = q(x 0 |x -l:0 ) is the learned Language Model of context length l. R Bayes = H(P, P l ) is the Bayes Risk of optimal model (the assumed 'limit' when we have infinite data points and model parameters) and L Approx = D KL (P l , Q l ) is the Approximation Loss, which can be affected by dataset size D, etc. The Bayes Risk is model or data agnostic, only related to natural language itself and is limited only by visible context length.
this section cite: []

Section: PRELIMINARY: INTRINSIC SPACE
In previous work (Bahri et al., 2024;Cheng et al., 2023), as a common practice, the 'Data Manifold' is often defined as the middle feature representation of well-trained neural networks, and assumptions are made on this kind of mid-representation, with experiments to validate these assumptions.
(Intrinsic Space is defined as the space where the Data Manifold lies.) We follow such practice in main paper for clarity.
Meanwhile, the Data Manifold can be more formally defined by a mapping from input data to some Intrinsic Space which satisfies a certain set of properties, and mid-representation of well-trained neural networks are assumed to have such properties, which can be experimentally validated. This is an equivalent yet more formal perspective. In Appendix D, we formally define the Intrinsic Space and derive related results in our work with such perspective for completeness.
this section cite: ['b3', 'b7']

Section: PRELIMINARY: OUTLINES
In Section 2.2 we propose the definition of Intrinsic Entropy, and discuss how to bridge Bayes Risk with it, thus explaining how context length impacts Bayes Risk.
Approximation Loss, or how well the trained model learns Bayesian Model, is related to Intrinsic Dimension in previous work of Scaling Laws (Sharma & Kaplan, 2022;Shi et al., 2024). In Section 2.3 we discuss more about how the context length impacts Approximation Loss from this perspective.
We further derive that the balance between Bayes Risk and Approximation Loss would lead to an optimal context length which increases with the size of the training dataset. Our theoretical deduction and experiments on language are presented in Section 3.
this section cite: ['b28', 'b29']

Section: BAYES RISK WITH CONTEXT LENGTH: AN INTRINSIC ENTROPY PERSPECTIVE
In this section we discuss to bridge context length and Bayes Risk with the concept of Intrinsic Entropy.
this section cite: []

Section: BAYES RISK AND ENTROPY IN INTRINSIC SPACE: DERIVED FROM FIRST PRINCIPLES
'Information Entropy' is defined as the amount of information carried in the Intrinsic Space. Here are detailed assumptionsfoot_1 as definitions:foot_2
• Assumption 1. Information Entropy of Intrinsic Space for Bayes Model lim l→∞ S(P l ) = S(P ∞ ) is finite, which is the Information Entropy of next token prediction of language itself.
• Assumption 2. ∀l 1 , l 2 such that l 1 < l 2 , S(P l1 ) < S(P l2 ). This is because a longer context contains more information.
• Assumption 3. Linear Entropy Relationship: The Information Entropy w.r.
t. Next Token Prediction, defined as S ntp (P l ) = H(P 0 )-H(P l ), is linear with the Entropy in the Intrinsic Space of the Bayes Model, i.e., S ntp (P l ) = k * S(P l ) + b, and 0 < k < 1. A formal definition can be found in Appendix D. S ntp is smaller than S since the Intrinsic Space contains important information on previous tokens that are important for the prediction of future tokens, while S ntp is related only to the next token. The hidden state in RNNs contain more information than only the next token to predict. For example, consider a character-level RNN that predicts the sentence '1 + 2 equal ', the next character to predict is 's', but the hidden state should contain information about answer '3' for the latter tokens. With these assumptions, we can derive that the Bayes Risk is linear with respect to the Intrinsic Entropy: R Bayes = H(P, P l ) = -k * S(P l ) + Const (2) This linear relationship is observed in experiments for LMs in Section 2.2.2, and for synthetic data in Section 4.3. Note that by Assumptions 1 and 2 we derive:foot_3 ∂R Bayes ∂l < 0, and lim l→∞ ∂R Bayes ∂l = 0.
this section cite: []

Section: BAYES RISK AND INTRINSIC ENTROPY: EXPERIMENT MEASUREMENT
We use well-trained Large Language Models to conduct experiments for approximating the Bayes Risk H(P l ) on certain text corpora. We find that:
H(P, P l ) ≈ C 0 + C/l γ(3)
approximates the experimented behavior well on both OpenWebText and other text corpora (please refer to Appendix H and Appendix E for detailed figures and results on different datasets). 2), we measure the Intrinsic Entropy in the hidden representation space of well-trained Language Models. For a given context length, we gather the hidden state of the final layer for the last token across multiple (≥ 10000) samples, and use Gaussian Kernel Density Estimation (Gaussian-KDE) to estimate the Information Entropy of the distribution in this space. We also provide an alternative eigenvalue-based estimation method in Appendix H.
this section cite: []

Section: Experimentally measure Intrinsic Entropy using Gaussian-KDE To validate the linear relationship between Bayes Risk and Intrinsic Entropy (Equation
We conduct experiments on multiple Language Models: Llama-3.1-8B, Qwen3-8B-Base, and RecurrentGemma-9B, all evaluated on a subset of the OpenWebText dataset. As shown in Figure 2, the linear relationship between Cross Entropy loss and Gaussian-KDE measured Intrinsic Entropy holds across different model architectures, validating our theoretical assumptions:
R Bayes ≈ -k * S(P l ) + Const,
which aligns well with Equation 2, thus validating our entropy-based deduction. Note that for RecurrentGemma-9B, several outlier points at very low context lengths exhibit significantly higher CE loss than other models, indicating that RecurrentGemma-9B is not a good approximation of the Bayes Model at those context lengths; excluding these outliers, the linear relationship still holds.
this section cite: []

Section: APPROXIMATION LOSS WITH CONTEXT LENGTH: AN INTRINSIC DIMENSION PERSPECTIVE
Approximation Loss in the Training Scenario Previous work summarizes scaling laws (Kaplan et al., 2020;Hoffmann et al., 2022) as L Approx (D) = C 0 + A/D α for dataset size D. This has been explained from an intrinsic-space perspective in work such as (Bahri et al., 2024;Sharma & Kaplan, 2022), where α ≈ c/dim and dim is the manifold dimension of the data/model under a uniform-distribution assumption in intrinsic space. In Appendix D.2, we derive this rigorously from weaker assumptions (Theorem 1, Theorem 2). As assumed in Section 2.2.1, the Intrinsic Dimension should increase with l. Combined with previous results on α = c/dim(l), we have,
L Approx = C 0 + A(l)/D α(l) , ∂α ∂l < 0.(4)
This shows that longer context makes it harder for the model to approximate the Bayes model.
this section cite: ['b14', 'b12', 'b3', 'b28']

Section: Approximation Loss in the Inference Scenario
The analysis above considers the training scenario, where both training data size D and context length l jointly determine approximation loss. We now consider a second case: a pre-trained model with fixed parameters evaluated on downstream tasks with varying visible context length l vis at inference time.
In this scenario, model parameters are fixed, and approximation loss L Approx (l vis ) depends on how well the fixed model approximates the Bayes model P lvis for each visible context length. As l vis increases, P lvis lies in a higher-dimensional intrinsic space, making it harder for a fixed-capacity model to approximate. Hence, ∂L Approx /∂l vis > 0: approximation loss increases with visible context length for a fixed model.
Moreover, for a harder downstream task (e.g., one that requires information spread across a wider context range), the Bayes model is more complex and thus harder to approximate, leading to a larger approximation loss overall. This implies that, for a fixed model, harder tasks have larger approximation loss at any given l vis .
this section cite: []

Section: DEDUCTION: OPTIMAL CONTEXT LENGTH
In this section, we present deductions from the theory in Section 2. In both training and inference scenarios, Bayes Risk decreases with context length l (Section 2.2), while Approximation Loss increases with l (Section 2.3). The balance between these two opposing trends leads to an optimal context length.
In general, let l denote context length (either training context length or visible context length at inference), let θ t denote task-specific parameters that affect Bayes Risk (e.g., task difficulty γ in Position-Weighted Ruler-QA1, or the distribution of relevant information across context), and let θ m denote model/data-specific parameters that affect Approximation Loss (e.g., training dataset size D, or model capacity). The total loss can be written as:
Loss(l, θ t , θ m ) = R Bayes (l, θ t ) + L Approx (l, θ m ),(5)
where ∂R Bayes /∂l < 0 with lim l→∞ ∂R Bayes /∂l = 0 (Section 2.2), and ∂L Approx /∂l > 0 (Section 2.3). Since R Bayes is a decreasing convex function of l and L Approx is increasing in l, the derivative of total loss, ∂ l Loss = ∂ l R Bayes + ∂ l L Approx , transitions from negative (Bayes  (2) the critic point corresponds to a smaller optimal context length for tasks with larger γ (i.e. tasks requiring less long context abilities). Right: Intrinsic Entropy measured on samples truncated to certain context lengths. The Intrinsic Entropy shows increment of intrinsic information when increasing context length, and resembles acc-ctl curves for larger γ.
Risk dominates) to positive (Approximation Loss dominates), yielding an optimal context length l * where ∂ l Loss = 0. Note that R Bayes depends on θ t : in inference, different tasks (e.g., different γ in Position-Weighted Ruler-QA1) have different distributions of relevant information across context, leading to different rates at which R Bayes decreases with l.
Moreover, when θ m varies such that L Approx decreases (e.g., more training data in the training scenario, or a stronger model in the inference scenario), the point where
∂ l L Approx balances |∂ l R Bayes |
shifts to a larger l, so the optimal context length l * increases. Conversely, when θ t varies such that R Bayes decreases faster with l (e.g., tasks where relevant information is distributed across a wider context range, corresponding to smaller γ), Bayes Risk keeps decreasing at larger l, also leading to a larger l * .
this section cite: []

Section: EXPERIMENTAL MEASUREMENT OF OPTIMAL CONTEXT LENGTH FOR TRAINING
We conduct experiments on a subset of OpenWebText with a sufficiently long context length. We use nanogpt (Karpathy, 2022) and train a model with GPT-2 (Radford et al., 2019) architecture (GPT-2-124M, 12-head transformers, 768-dim feature vector, with half the transformer layers (12 → 6) to reduce GPU memory for long contexts). We train GPT-2 on different context lengths with different amounts of training data (200M, 250M, 300M, 350M, 500M, 750M tokens), until the validation loss increases.
We show our results in Figure 1 and Figure 15. As shown both theoretically and experimentally, there does exist an optimal context length, beyond which even relevant long context would increase validation loss of pretraining Language Models. Such optimal context length would increase with training dataset size. We also provide similar experiments to prove an optimal context length exists on a synthetic dataset, as shown in Appendix F. More details for our experiment settings are presented in Appendix I.
this section cite: ['b15', 'b24']

Section: EXPERIMENTAL MEASUREMENT OF OPTIMAL CONTEXT LENGTH ON DOWNSTREAM TASKS
The analysis above focuses on training. We also study context-length effects on downstream tasks, where a trained model is evaluated with varying visible context length. We observe that optimal context length also exists for downstream tasks, and that the optimum increases with task contextlength requirements. As shown in Figure 8, on the RULER benchmark, most Qwen3 series models show an optimal context length for qa 1, fwe, and cwe subtasks.
To study the impact of task properties on this 'optimal context length' phenomenon, we propose a Position-Weighted Ruler-QA1 benchmark: instead of a uniform query distribution, query probability depends on the distance of the golden paragraph to the end of input: P (x) ∝ (1 -x/L) γ . Different γ values correspond to tasks focused on different context ranges. As shown in Figure 3 and Figure 4, an optimal context length exists for each γ, and a smaller γ (i.e., a task requiring more long-context ability) typically leads to a larger optimal context length.
This result can be interpreted through Bayes Risk and Approximation Loss decomposition. Intuitively, some tasks require larger context lengths to solve (i.e., the Bayes Risk for that metric decreases more slowly with context length than for metrics such as next-token Cross Entropy), so they benefit from more context. However, because model performance eventually degrades at long context (i.e., Approximation Loss still increases with context length), the balance of these two terms still produces an optimal context length. More details and additional results on models and RULER subtasks are provided in Appendix A.
this section cite: []

Section: PROOF OF CONCEPT WITH SYNTHETIC DATA

this section cite: []

Section: LIST OF POINTS TO PROVE
In this section, we conduct experiments on a synthetic dataset, explaining the Bayes Risk and related theories we proposed in Section 2.2. With this synthetic dataset, we would like to prove the following,
• Point 1. Cross Entropy Loss is approximately linear with Intrinsic Entropy (Assumption 3 in Section 2.2.1). Shown in Section 4.3.
• Point 2. By measuring Entropy in Intrinsic Space of well-trained models, one could obtain a valid measurement that is linear with Cross Entropy Loss (Section 2.2.2). Shown in Section 4.4.
this section cite: []

Section: CONSTRUCTION OF SYNTHETIC DATA: THE 'POSITION WEIGHTED MULTITASK SPARSE PARITY' DATASET
In previous work, a common practice is to mask the leftmost tokens and leave l tokens before the token-to-predict visible to Language Models, as shown in Figure 5. Although this may not show the impact of important tokens to final answer perplexity (e.g., it fails to show the importance of the second key info in Figure 5), this method aligns well with our setting of increasing context length.
Although the next token to predict might depend on several pieces of key information, we see from Figure 5 that the first key token would raise model perplexity.
Inspired by this concept in Figure 5 and the 'multitask sparse parity dataset previously studied in (Michaud et al., 2024;Barak et al., 2022), we propose the 'position-weighted multitask sparse parity dataset. In detail, each input consists of L 'context bits, each bit lies in {0, 1}. Each subtask takes xor on two certain bits in the context bits, and the answer to some sample is the answer of the only activated subtask, as shown in Figure 5. We use 60 context bits and 200 tasks. From 11th to Although seeing both pieces of information are necessary to answer the question, perplexity rises dramatically only when the first piece of information is masked. Right: An example of our synthetic data. Each sub-task corresponds to 2 context bits of fixed position. At each time, exactly one sub-task is activate, and the ground truth output is calculated by taking XOR over the 2 context bits of the activate task. As shown in the example, the answer for Subtask 1,2,3 is 0 ⊕ 0 = 0, 0 ⊕ 1 = 1 and 1 ⊕ 1 = 0 respectively, but since the thrid bit is 1 for control bits, only Subtask 3 is activated and the final answer is 0. However, for a model of context length 7, it cannot see the 9th bit required by subtask 3, making it unable to predict the answer correctly.
the 60th bit, each bit corresponds to the max bit of two tasks: #T ask| max(bit1,bit2)=i = 2, ∀i ∈ {11, 12, . . . , 60}.
We assign different frequencies to different tasks, approximating the real-world situation where tasks requiring nearer bits are more often. In all, Bayes Risk, or the minimum Cross Entropy Loss, is:
R Bayes (ctl) =M inCELoss(ctl) =( task s.t. max(bit1,bit2)>ctl f req(task) log 2)/ task f req(task) ≈A + B/(ctl + C) α
More details are shown in Table 1.
this section cite: ['b22', 'b4']

Section: TRANSFORMER-BASED SYNTHETIC MODEL WITH ENTROPY MEASUREMENTS
We use a 3-layer causal Transformer, with embedding dimension 208 and FFN dimension 832, RoPE embedding with base frequency 4000; input sequence length is always 60+1, with 60 context tokens (either 0, 1 or ?) and 1 task tokens (chosen from task tokens of vocab size 200).
We use 100 tasks and 60 task bits. From 11th to the 60th bit, each bit corresponds to the max bit of two tasks: that is, #T ask| max(bit1,bit2)=i = 2, ∀i ∈ {11, 12, . . . , 60}.
During training, 50% of the samples are unmasked, while for the other 50% samples, we mask the last X task bits to be 0.5, where X is a random int from 60 -10 to 60 -60. This ensures our model to be able to handle mask bits, and also ensures it can learn uncommon tasks (relying on context bits that are at the end of the context bits) well. We train the model on large enough dataset so that it approximates the Bayes Model well (please refer to Table 1 in Appendix for more details).
After the model has been trained, we measure its eigen values, as shown in Figure  the Intrinsic Space (right figure), where the case N = 200 (all eigen values) are also shown in the middle figure.
This validates Point 1: Cross Entropy Loss is approximately linear with Intrinsic Entropy as measured by the sum of log eigenvalues.
this section cite: []

Section: ENTROPY IN INTRINSIC SPACE: SYNTHETIC DATASET VALIDATION
Figure 7 shows the measured results of Intrinsic Entropy on the synthetic dataset, which follows a linear relationship with the Cross Entropy Calculated (Theo CE) and Cross Entropy loss measured (Exp CE).
This provides evidence for Point 2 in Section 4.1: we can measure entropy in the intrinsic space using eigenvalue-based methods or density-based methods, and both show linear relationships with Cross Entropy Loss, validating our entropy-based theoretical framework.
this section cite: []

Section: CONCLUSION AND DISCUSSIONS

this section cite: []

Section: References
Ref_id:b0 Title: Llama-3.1-8B Qwen-3-8B-Base RecurrentGemma Year: ()
Ref_id:b1 Title: Intrinsic dimensionality explains the effectiveness of language model fine-tuning Year: (2020)
Ref_id:b2 Title: Intrinsic dimensionality explains the effectiveness of language model fine-tuning Year: (2021-08)
Ref_id:b3 Title: Explaining neural scaling laws Year: (2024-06)
Ref_id:b4 Title: Hidden progress in deep learning: SGD learns parities near the computational limit Year: (2022)
Ref_id:b5 Title: The role of deductive and inductive reasoning in large language models Year: (2025)
Ref_id:b6 Title: L$ˆ2$m: Mutual information scaling law for long-context language modeling Year: (2025)
Ref_id:b7 Title: Bridging information-theoretic and geometric compression in language models Year: (2023-12)
Ref_id:b8 Title:  Year: (2024)
Ref_id:b9 Title:  Year: (2024)
Ref_id:b10 Title: Mamba: Linear-time sequence modeling with selective state spaces Year: (2024)
Ref_id:b11 Title: Understanding scaling laws with statistical and approximation theory for transformer neural networks on intrinsically low-dimensional data Year: (2024)
Ref_id:b12 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b13 Title: RULER: What's the real context size of your long-context language models? Year: (2024)
Ref_id:b14 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b15 Title:  Year: (2022)
Ref_id:b16 Title: Nikolaos Pappas, and Franc ¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b17 Title: Attention consistency for llms explanation Year: (2025)
Ref_id:b18 Title: Statistical Physics Year: (1980)
Ref_id:b19 Title: Same task, more tokens: the impact of input length on the reasoning performance of large language models Year: (2024)
Ref_id:b20 Title: Human motion instruction tuning Year: (2025)
Ref_id:b21 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b22 Title: The quantization model of neural scaling Year: (2024)
Ref_id:b23 Title: Reinventing rnns for the transformer era Year: (2023)
Ref_id:b24 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b25 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2023)
Ref_id:b26 Title: Squad: 100,000+ questions for machine comprehension of text Year: (2016)
Ref_id:b27 Title: A neural scaling law from the dimension of the data manifold Year: (2020)
Ref_id:b28 Title: Scaling laws from the data manifold dimension Year: (2022)
Ref_id:b29 Title: Scaling law for time series forecasting Year: (2024)
Ref_id:b30 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2023)
Ref_id:b31 Title: Learning to (learn at test time): Rnns with expressive hidden states Year: (2024)
Ref_id:b32 Title: Razorattention: Efficient kv cache compression through retrieval heads Year: (2024)
Ref_id:b33 Title: Scaling laws with vocabulary: Larger models deserve larger vocabularies Year: (2024)
Ref_id:b34 Title: Effective long-context scaling of foundation models Year: (2024-06)
Ref_id:b35 Title: Retrieval meets long context large language models Year: (2024)
Ref_id:b36 Title:  Year: (2025)
