Title: AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders
Abstract: Speculative Decoding (SD) accelerates large language model inference by employing a small draft model to generate predictions, which are then verified by a larger target model. The effectiveness of SD hinges on the alignment between these models, which is typically enhanced by Knowledge Distillation (KD). However, conventional KD methods aim to minimize the KL divergence between the draft and target models across all tokens, a goal that is misaligned with the true objective of SD, which is to maximize token acceptance rate. Therefore, draft models often struggle to fully assimilate the target model's knowledge due to capacity constraints, leading to suboptimal performance. To address this challenge, we propose AdaSPEC, a novel method that incorporates selective token filtering into the KD process. AdaSPEC utilizes a reference model to identify and filter out difficult-to-fit tokens, enabling the distillation of a draft model that better aligns with the target model on simpler tokens. This approach improves the overall token acceptance rate without compromising generation quality. We evaluate AdaSPEC across diverse tasks, including arithmetic reasoning, instruction-following, coding, and summarization, using model configurations of 31M/1.4B and 350M/2.7B parameters. Our results demonstrate that AdaSPEC consistently outperforms the state-of-the-art DistillSpec method, achieving higher acceptance rates across all tasks (up to 15%).

Section: Introduction
Large language models (LLMs) have revolutionized natural language processing, achieving impressive performance across a wide range of tasks. Models like GPT-4 [25] and Llama 3 [11] demonstrate state-of-the-art results in various natural language understanding and generation tasks [2,27], including highly complex tasks such as summarization [23] and mathematical reasoning [9,13]. However, as these models grow in size and complexity, their inference becomes increasingly computationally intensive, leading to practical challenges in deployment, including slow generation speeds and significant output latency.
To address these shortcomings, current approaches primarily focus on achieving a trade-off between efficiency and performance through two main strategies. The first involves compressing the model scale to enhance the capability of smaller models, often using techniques like Knowledge Distillation (KD) [14]. The second approach employs methods such as quantization to enable faster computation. However, these strategies inevitably lead to a sacrifice of performance, either due to a loss of representational capacity during compression or reduced accuracy resulting from optimization for speed. As a result, there is a growing need for methods that can maintain the high performance of LLMs while significantly improving their inference efficiency.
Recently, Speculative Decoding (SD) [16,7] has emerged as a promising paradigm for accelerating LLM inference without sacrificing performance. Unlike model compression or quantization, which modify the model architecture or parameters, SD accelerates generation by restructuring the decoding process itself. Specifically, it introduces a lightweight draft model that speculatively generates multiple candidate tokens, which are then verified by the larger target model. This paradigm preserves the target model's predictive quality while substantially reducing the number of expensive forward passes, offering a new efficiency-performance trade-off.
The core of SD lies in the design of the draft model. This model is typically much smaller than the target model, even ranging from one-tenth to one-hundredth of the size, enabling faster token generation while maintaining a certain level of capability. Consequently, the actual inference speed-up achieved by SD relies on the draft model closely aligning its predictions with the target model's output distribution. Typically, this alignment is achieved by pre-training and fine-tuning both models on the same datasets, yielding a pair of homogeneous models from the same family, sharing the same architecture but differing in size. However, training two models on the same datasets does not necessarily produce optimal alignment, especially given the significant scale disparity between the draft and target models. This difference in scale makes the draft model prone to prediction errors. To address this challenge, state-of-the-art methods employ KD techniques to refine the draft model, rather than relying solely on direct fine-tuning [32]. However, optimizing fidelity metric (e.g., forward KL divergence) does not necessarily lead to a high acceptance rate. Worse still, it may waste the draft model's limited capacity on tokens that are inherently hard to learn and unlikely to be accepted anyway. Additionally, these methods may encounter issues such as the loss failing to converge. Given these challenges, there is a critical need for SD-specific training regimes that effectively balance model capacity constraints with prediction accuracy requirements.
Fortunately, we observe substantial variation in the difficulty of learning individual tokens during KD, which has critical implications for transferring knowledge from the teacher (target) model to the student (draft) model. Instead of mimicking the full output distribution of the target model, the draft model only needs to produce correct predictions on the subset of tokens that is easy enough to propose. During the process of distillation, we identify a subset of "hard" tokens that pose particular challenges for the student model to learn and to predict accurately, regardless of training efforts. Conversely, other tokens are relatively easy to assimilate. We argue that uniformly emphasizing the loss on both "easy" and "hard" tokens may be counterproductive. Attempting to reduce the loss on difficult tokens often comes at the expense of increasing the loss on easy tokens, resulting in suboptimal learning across both categories. To address this issue, we propose a novel approach: deliberately excluding "hard" tokens from the training process. By focusing the loss function exclusively on "easy" tokens, we can more effectively utilize the limited capacity of the student model, thus achieving better alignment with the teacher model on these tokens. This strategic exclusion of hard-to-learn tokens allows the student model to concentrate its resources on mastering the more accessible aspect of the teacher's knowledge, potentially leading to improved overall performance in SD tasks. Our approach thus maximizes the alignment between the draft and target models within the constraints of the draft model's capacity.
In this study, we propose AdaSPEC, a novel Knowledge Distillation method designed to bridge the capacity gap between the draft and target models in SD. AdaSPEC operates in two phases:
1: Reference Model Distillation and Token Filtering: A reference model, initialized as a copy of the draft model, is distilled using the target model as its teacher. For simplicity, we assume that the target model has been well fine-tuned to downstream tasks of our interests. Here the reference model serves a crucial role as a token filter. It identifies "hard" tokens-those that are difficult for smaller models to predict accurately-by comparing the perplexity differences between the reference and draft models on the training data.
2: Selective Draft Model Distillation: Finally, the draft model undergoes distillation using a filtered dataset. The reference model removes the previously identified "hard" tokens, allowing the draft model to focus its limited capacity on learning to predict the remaining, more manageable tokens accurately.
We conduct extensive experiments on a wide range of models and downstream tasks, where we benchmark AdaSPEC against DistillSpec and find that AdaSPEC sucessfully pushes the limit of SD-across all tasks and model setups, AdaSPEC consistently achieves higher acceptance rates (up to 15%; see Table 1).
this section cite: ['b24', 'b10', 'b1', 'b26', 'b22', 'b8', 'b12', 'b13', 'b15', 'b6', 'b31']

Section: Preliminaries
In this section, we provide a formal overview of the foundational concepts. We begin with the mathematical framework of SD, followed by a description of various evaluation metrics. Finally, we explore language model families and their significance in enabling techniques like SD to bridge performance gaps between models of different sizes.
this section cite: []

Section: Speculative Decoding.
Speculative Decoding [7,16,29,20,32,6,17,18,30,6,21,26,32,20] is originally proposed to accelerate LLM inference by employing a compact draft model to predict potential output sequences in advance and then verified by a larger target model. The typical framework of SD is formulated as follows. Let M p and M q denote the large target model and the compact draft model, respectively. SD leverages the draft model to autoregressively generate γ tokens
z ≜ {z i } γ i=1 ∼ q θ (• | x) based on the input x = [x 1 , x 2 , . . . , x t ]
, which includes the prompt and previously generated tokens. The target model then verifies these proposed tokens by evaluating their probabilities {p(z i | x, z <i )} γ i=1 in parallel. Both models generate probability distributions p(z i+1 | x, z <i ) and q(z i+1 | x, z <i ) for each token i = 1, . . . , γ in a single forward pass. Using a greedy decoding strategy, only the tokens with the highest probabilities are selected for generation or verification. The sampling functions are:
S p (z <i ) = arg max zi+1 p(z i+1 | x, z <i ),(1)
S q (z <i ) = arg max zi+1 q(z i+1 | x, z <i ),(2)
for each i = 1, . . . , γ. The complete sampling and verification process is detailed in Appendix A.1
Acceptance Rate. The acceptance rate, α, measures the accuracy of the draft model M q compared to the target model M p . It is calculated as:
α = accept accept + reject .(3)
Here, accept and reject are the count of tokens accepted and rejected by M p , respectively. A higher α indicates greater alignment between M p and M q , facilitating faster inference in practical scenarios.
this section cite: ['b6', 'b15', 'b28', 'b19', 'b31', 'b5', 'b16', 'b17', 'b29', 'b5', 'b20', 'b25', 'b31', 'b19']

Section: Block Efficiency and Wall-time Improvement.
Block efficiency [7,16], τ , quantifies the average number of tokens generated per iteration. It is defined as the expected number of accepted tokens per block, with a maximum value of γ + 1 for a block size of γ. The block efficiency can also be expressed in terms of the acceptance rate α [16]:
τ (x) = 1 -α γ+1 1 -α .(4)
This metric evaluates how effectively M q approximates M p . The speed-up factor for the total wall-time is given by:
Speed-up = τ (x) γc + 1 , (5
)
where c is the cost coefficient, representing the ratio of the time taken by a single execution of M q to that of M p .
Language Model Families. Modern language models are often developed as part of a family of models that share the same core architecture but differ in scale, typically measured by the number of parameters or the size of the training dataset. These families, such as Llama 3 [11], BERT [10], and Pythia [5], are designed to enable researchers and practitioners to balance computational efficiency and performance based on specific use cases. Within a family, smaller models are generally used for tasks requiring faster inference or lower computational cost, while larger models are leveraged for tasks demanding higher accuracy and richer representations. This structural consistency within a family allows for techniques like Knowledge Distillation [14] and Speculative Decoding [7,16] to transfer knowledge or align predictions effectively between models of varying sizes.
this section cite: ['b6', 'b15', 'b15', 'b10', 'b9', 'b4', 'b13', 'b6', 'b15']

Section: Method
We introduce AdaSPEC, an adaptive distillation framework for SD that enhances the alignment between a target model and a smaller draft model through selective Knowledge Distillation. Given a target model M p fine-tuned for a specific downstream task, AdaSPEC consists of two key steps: (1) constructing a reference model M ref and (2) selectively distilling knowledge from M p and M ref to the draft model M q .
this section cite: []

Section: Step 1: Constructing the Reference Model.
The reference model M ref is constructed by distilling M p on a downstream task dataset D using the DistillSpec framework [32]. The objective is to minimize the forward KL divergence between the target model and the reference model:
L KD = E x∼D,y∼P (y|x) [K (P (y|x)||R(y|x))] ,(6)
where x represents the input prefix, y denotes the generated context, K denotes the forward KL divergence, P (y|x) represents the probability distribution of the target model, and R(y|x) corresponds to the probability distribution of the reference model.
this section cite: ['b31']

Section: Step 2: Selective Knowledge Distillation for the Draft Model.
To identify learnable tokens for the draft model M q , we compute token-wise losses based on the predicted distributions of M ref and M q . Specifically, for each token w, the token-wise KL divergence losses are computed as:
L ref (w) = K (P (w | context)||R(w | context)) ,(7)
L draft (w) = K (P (w | context)||Q(w | context)) ,(8)
where Q(w | context) is the probability predicted for token w by the draft model, given the context.
Next, we calculate the difference in token-wise losses:
∆ L (w) = L draft (w) -L ref (w).(9)
Tokens with higher ∆ L (w) represent a larger performance gap between M q and M ref relative to M p , suggesting that these tokens are not yet well aligned but are highly learnable for the draft model. Accordingly, we select the subset of tokens with larger ∆ L (w) values, as they are most promising for improving the alignment between the draft and target models. Specifically, we denote
S = { w | ∆ L (w) is among the top k×100% of all tokens }, k ∈ [0, 1].
Therefore, the overall loss for training the draft model M q is:
L distill = 1 k • |y| |y| i=1 I [y i ∈ S] • L draft (y i ), (10
)
where I[•] is the indicator function that equals 1 if the condition inside the brackets is satisfied, and 0 otherwise. It ensures that only the selected learnable tokens contribute to the loss calculation. The whole filtering process is shown in figure 1.
this section cite: []

Section: Experiments
We evaluate AdaSPEC through comprehensive experiments across diverse domains and conduct detailed ablation studies to analyze its impact on the acceptance rate α.
this section cite: []

Section: Experimental Setup
Our experimental framework employs GPT-like decoder-only Transformer models in two distinct configurations, designed to evaluate performance across different parameter scales while maintaining tokenizer consistency for SD:
• Small-to-Large Model Configuration: A draft model Pythia-31M paired with target model Pythia-1.4B [5]. These models share architecture and tokenizer, providing an ideal test case for same-family knowledge transfer.
• Medium-to-Large Model Configuration: A draft model CodeGen-350M paired with target model Phi-2 [24,1]. While from different families, these models use an aligned tokenizer to ensure token-level consistency, allowing us to evaluate cross-family KD.
We test these two configurations on a diverse set of five tasks, each representative of a specific domain to provide a robust evaluation framework for AdaSPEC: GSM8K [9] (A benchmark for multi-step arithmetic reasoning), Alpaca [27] (A comprehensive instruction following dataset), MBPP [3] (A Python programming challenge set for code generation), CNN/Daily Mail [22] (A long-form summarization task), and XSUM [23] (An extreme summarization challenge).
this section cite: ['b4', 'b23', 'b0', 'b8', 'b26', 'b2', 'b21', 'b22']

Section: Reference Model Training.
To ensure a consistent starting point and fair comparison, both the draft model and the reference model are initialized from the same pre-trained model. For each task, we first fine-tune the target model on the task-specific dataset to establish a strong baseline. The reference model is then trained using the method from DistillSpec [32].
this section cite: ['b31']

Section: Baseline Setup
We compare AdaSPEC against DistillSpec [32], the current state-of-the-art method for SD. Although AdaSPEC builds upon DistillSpec's training framework for its reference model, it introduces novel token selection mechanism. To evaluate its effectiveness, we evaluate both methods under two settings: a resource-efficient scenario with fixed training duration and a scenario optimized for maximum performance:
• 3-Epoch Setting: Both reference and draft models are trained for exactly 3 epochs, a standard practice in LLM fine-tuning that balances task-specific performance with general capability retention [4]. This controlled training duration effectively prevents overfitting while ensuring adequate task adaptation. This setting evaluates model effectiveness under typical resource constraints and provides insights into rapid adaptation scenarios.
• Optimal-Epoch Setting: Models are trained for a variable number of epochs, treated as a tunable hyperparameter, to maximize task-specific performance. While this approach may lead to overfitting to the specific task at the expense of performance on other tasks, it allows us to thoroughly evaluate the upper bound of performance. The optimal number of epochs is determined empirically. Specifically, for GSM8K, the number of target epochs is chosen according to validation accuracy, while for the rest of the experiments it is chosen according to validation perplexity. Afterwards, we distill the reference model and pick the one with highest α on validation set. Eventually, this model serves as reference to train our draft model. For robustness, we only select the optimal epoch from 1, 3, 6, 10, 15, 20 and 30 (for XSUM and CNN/Daily Mail we select from 1, 3, 6, 10 for training efficiency). This configuration enables evaluation of both methods under less constrained scenarios, where achieving optimal task performance takes precedence over maintaining general capabilities.
While Zhou et al. [32] employs a more extensive training schedule in their DistillSpec experiments, our study adopts a more resource-efficient approach due to computational constraints. In the Optimal-Epoch Setting, we limit training to a maximum of 30 epochs, striking a balance between performance optimization and computational feasibility. Complete hyperparameter configurations and training specifications for both DistillSpec and AdaSPEC are detailed in Appendix A.2.
this section cite: ['b31', 'b3', 'b31']

Section: Main Results
We summarize the main experimental results in Table 1. Acceptance Rate Analysis. We evaluate performance using the acceptance rate α, defined as the proportion of draft-model-generated tokens validated by the target model. As shown in Table 1, AdaSPEC consistently achieves higher acceptance rates than DistillSpec across all tasks and model configurations, demonstrating superior draft-target model alignment.
this section cite: []

Section: Analysis
To provide detailed insights into AdaSPEC's effectiveness, we conduct in-depth analyses on two representative configurations:
• Pythia-31M/1.4B on GSM8K (3-Epoch): This configuration examines performance on arithmetic reasoning under constrained training conditions, representing scenarios with limited computational resources and the need for generalization. Since reasoning is typically considered as an additional capability beyond general language modeling, this setup ensures that the model retains its core abilities while effectively handling arithmetic tasks.
• Pythia-31M/1.4B on CNN/Daily Mail (Optimal-Epoch): This setup investigates extractive summarization with extended training, demonstrating the model's ability to optimize for task-specific objectives. In real-world applications, models are sometimes specifically deployed for summarizing long-form contents such as news reports, emails, or web pages, requiring dedicated fine-tuning. Thus, the Optimal-Epoch setting is chosen to maximize the model's summarization capabilities.
this section cite: []

Section: Task-Level Acceptance Rate Distribution.
We first analyze the distribution of acceptance rates across tasks for both methods. As illustrated in Figure 2, AdaSPEC demonstrates consistently superior performance compared to DistillSpec. The acceptance rate histograms for both tasks exhibit a significant rightward shift under AdaSPEC, indicating more frequent successful draft predictions. This systematic improvement in acceptance rate suggests that AdaSPEC's selective distillation approach effectively enhances draft-target model alignment across diverse task contexts.
this section cite: []

Section: Logit Margin Distributions Across Tokens.
Next, we analyze the distribution of top-2 logit margins across tokens for both methods. The logit margin, defined as the difference between the logits of the top-1 and top-2 predicted tokens, serves as a measure of prediction confidence. A positive margin indicates a correct draft model prediction, while a negative margin signifies an incorrect prediction that would be rejected in SD.
As shown in Figure 2, AdaSPEC demonstrates superior logit margin distributions compared to DistillSpec across both GSM8K and CNN/Daily Mail datasets. AdaSPEC exhibits:
• Higher frequency and magnitude of positive margins, indicating more frequent and confident correct predictions.
• Lower frequency and magnitude of negative margins, suggesting less frequent and less severe prediction errors.
These patterns demonstrate that AdaSPEC achieves better draft-target model alignment through its selective distillation approach, enabling more effective knowledge transfer from the target model to the draft model.
this section cite: []

Section: KL-Divergence Distribution Across Tokens.
We further analyze the Kullback-Leibler (KL) divergence between draft and target models' token prediction distributions on both GSM8K and CNN/Daily Mail datasets. As illustrated in Figure 2, AdaSPEC exhibits consistently lower KL divergence values compared to DistillSpec across both tasks, demonstrated by significant leftward shifts in the distributions. This systematic reduction in KL divergence across different tasks and tokens indicates that AdaSPEC's selective distillation approach achieves tighter alignment between draft and target model predictions, corroborating our previous findings on acceptance rates and logit margins.
this section cite: []

Section: Case Studies.
We conduct detailed case studies on GSM8K and CNN/Daily Mail datasets. A consistent pattern emerges: AdaSPEC's prediction errors form nearly a subset of DistillSpec's errors, as illustrated in Figure 3. This pattern demonstrates the general effectiveness of AdaSPEC's targeted training approach in improving alignment and reducing inference discrepancies.
GSM8K, with its natural division between mathematical and non-mathematical tokens, offers particularly insightful analysis. During training, AdaSPEC predominantly selects mathematics-related tokens for focused learning (see Appendix A.3). During inference, this selective approach translates into significantly improved prediction accuracy for mathematical tokens compared to DistillSpec, as shown in Figure 3. These results demonstrate AdaSPEC's ability to identify and prioritize task-critical tokens during training, leading to more precise draft-target model alignment.
this section cite: []

Section: Ablation Study
To systematically evaluate the effectiveness of different components in AdaSPEC, we conduct comprehensive ablation studies across the following four key dimensions. All experiments are conducted on GSM8K and MBPP with Pythia 1.4B (target) and Pythia 31M parameters (draft). All models are trained for 3 epochs.
this section cite: []

Section: Token Selection Mechanism.
To evaluate our token selection strategy, we compare models trained on the top 40% of tokens (selected based on KL-divergence margin) against those trained on the bottom 40%. As shown in Table 2, models trained on the top 40% tokens consistently outperform those trained on the bottom 40%, with the latter performing even worse than the reference model.
The improvement is particularly pronounced on the MBPP dataset, where token selection yields up to a 6% performance gain. These results demonstrate that AdaSPEC effectively enhances model alignment by focusing on more learnable tokens during Knowledge Distillation.
this section cite: []

Section: Training Method.
To demonstrate that AdaSPEC's benefits extend beyond Knowledge Distillation, we replace the distillation process for both reference and draft models with direct fine-tuning. Table 3   and (2) fine-tuned draft models achieve up to 4% improvement over their reference counterparts, indicating that our token selection mechanism's benefits generalize beyond distillation to broader training scenarios.
this section cite: []

Section: Distillation Method.
We expand AdaSPEC to more distillation approaches: Reverse KL (RKL) and Total Variation Distance (TVD) [28]. With k = 0.4 for all methods, we observe that token selection significantly improves the acceptance rate by 6% on MBPP when using forward KL. However, when using RKL and TVD, the acceptance rate performance degrades. This is primarily attributed to the inherent limitations of RKL and TVD as distillation objectives, which struggle to effectively align the draft and target models in the context of SD. It is worth noting that DistillSpec [32] uses TVD as the distillation function with a batch size of 32 and a training step of 300,000. This prolonged training process not only requires substantial computational resources but also results in the problem of overfitting. Considering these factors, we ultimately select forward KL divergence as our distillation objective. Token Selection Ratio. To investigate the impact of token selection ratio, we vary k and compare the final acceptance rate of the draft model. Results in Fig 4 show that typically, lower k values result in better final acceptance rate. To strike a balance between training efficiency and performance, we finally choose k = 0.4 in most cases.
this section cite: ['b27', 'b31']

Section: Additional Experimental Results
Wall Clock Speed-up. To investigate AdaSPEC's potential to accelerate end to end decoding in a real world setting, we use frontier inference engine vLLM [15] on one single A100 GPU and report speed-up in Table 5. Results show that an expected 10∼20% speed-up could be easily achieved compared with DistillSpec, demonstrating the effectiveness of our approach. Table 6: Vicuna-7B-v1.3 [8, 31] with 3-Epoch finetuning following original EAGLE recipe. Here, training accuracy refers to firstgenerated-token accuracy in the training set. Eagle Eagle + AdaSPEC Training Accuracy ↑ 75.3% 76.3% Speed (s/sentence) ↓ 8.85 8.06 (-8.9%) Speed (tokens/s) ↑ 63.48 68.21 (+7.45%) Table 7: Acceptance rate of larger model configuration with 3-Epoch GSM8K. GSM8K DistillSpec 84.43% AdaSPEC 86.21%
Results on Larger Models. We conduct an additional GSM8K evaluation using a combination of the Qwen2.5-0.5B and Qwen2.5-32B models. When trained with 3 epochs, AdaSPEC reaches an acceptance rate of 86.21% while DistillSpec achieves 84.43%, as shown in Table 7. This shows that our approach can easily scale up to larger models.
this section cite: ['b14']

Section: Dicussion
Size Gap Between Target and Draft Models. In traditional SD settings, the size gap between the draft model and the target model is often within 10x. In this work, we demonstrate that AdaSPEC effectively bridges the performance gap, even when the size difference is substantial -up to 64 times in our experiments. By leveraging selective token filtering and Knowledge Distillation, AdaSPEC can enhance token acceptance rates and help maintain generation quality, providing more opportunities to use significantly smaller draft models.
this section cite: []

Section: Model Size Gap and Performance Gains.
From Table 1, we observe that the performance gain of AdaSPEC over DistillSpec becomes more pronounced as the size gap between the reference and target models increases (e.g., from CodeGen-350M → Phi-2 to Pythia-31M → 1.4B). This trend is consistent across both 3-Epoch and Optimal-Epoch settings. The result aligns well with our motivation: when the capacity discrepancy between models widens, direct Knowledge Distillation tends to suffer from representation mismatch, making it harder for the smaller model to absorb all teacher signals uniformly. AdaSPEC's adaptive mechanism mitigates this issue by selectively aligning easier tokens first, effectively narrowing the transfer gap. Consequently, the larger the size difference, the greater the relative improvement AdaSPEC achieves.
Connection with Lin et al. [19]. A similar token selection method is proposed in Lin et al. [19], which focuses on identifying and prioritizing harder-to-learn tokens (opposite to the motivation of AdaSPEC) during pre-training. Different from their design, our approach focuses on addressing the limited capacity of the draft model in SD. Specifically, we focus on identifying and filtering out challenging tokens, allowing the draft model to concentrate on learning easier-to-predict tokens. Our selective distillation process ensures that the draft model aligns more effectively with the target model on tokens that are more tractable, given its constrained capacity. By doing so, we maximize the draft model's limited resources while maintaining high-quality predictions in SD tasks. Thus, the essential difference lies in the distinct objectives of pre-training and Speculative Decoding.
this section cite: ['b18', 'b18']

Section: Limitations.
As a preliminary study on selective training for SD, we limit our study on simple loss-related token filter. In future work, one can design more adaptive filtering strategies as well as integrate AdaSPEC with tree-based or multi-step verification frameworks to further improve both speed and quality of LLM inference.
this section cite: []

Section: Conclusion
We present AdaSPEC, a novel approach for training more efficient draft models for SD. AdaSPEC introduces selective token filtering based on reference model perplexity gaps, enabling draft models to focus limited capacity on tokens where alignment with the target model is most achievable. Experiments show it outperforms baselines in arithmetic reasoning, instruction following, code generation, and summarization with higher acceptance rates.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-2: The surprising power of small language models Year: ()
Ref_id:b1 Title: Raft: A real-world few-shot text classification benchmark Year: (2022)
Ref_id:b2 Title: Program synthesis with large language models Year: (2021)
Ref_id:b3 Title: Deepseek llm: Scaling open-source language models with longtermism Year: (2024)
Ref_id:b4 Title: Pythia: A suite for analyzing large language models across training and scaling Year: (2023)
Ref_id:b5 Title: Simple llm inference acceleration framework with multiple decoding heads Year: (2024)
Ref_id:b6 Title: Accelerating large language model decoding with speculative sampling Year: (2023)
Ref_id:b7 Title: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023-03)
Ref_id:b8 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b9 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b10 Title: The llama 3 herd of models Year: (2024)
Ref_id:b11 Title: Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour Year: (2018)
Ref_id:b12 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b13 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b14 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b15 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b16 Title: Speculative sampling requires rethinking feature uncertainty Year: (2024)
Ref_id:b17 Title: Eagle-2: Faster inference of language models with dynamic draft trees Year: (2024)
Ref_id:b18 Title: Rho-1: Not all tokens are what you need Year: (2024)
Ref_id:b19 Title: Online speculative decoding Year: (2023)
Ref_id:b20 Title: Accelerating large language model serving with tree-based speculative inference and verification Year: (2024)
Ref_id:b21 Title: Abstractive text summarization using sequence-to-sequence rnns and beyond Year: (2016)
Ref_id:b22 Title: Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization Year: (2018)
Ref_id:b23 Title: Codegen: An open large language model for code with multi-turn program synthesis Year: (2022)
Ref_id:b24 Title: OpenAI. Gpt-4 technical report Year: (2024)
Ref_id:b25 Title: Specexec: Massively parallel speculative decoding for interactive llm inference on consumer devices Year: (2024)
Ref_id:b26 Title: Stanford alpaca: An instruction-following llama model Year: (2023)
Ref_id:b27 Title: f-divergence minimization for sequence-level knowledge distillation Year: (2023)
Ref_id:b28 Title: Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation Year: (2023)
Ref_id:b29 Title: Draft& verify: Lossless large language model acceleration via self-speculative decoding Year: (2024-08)
Ref_id:b30 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b31 Title: Distillspec: Improving speculative decoding via knowledge distillation Year: (2023)
