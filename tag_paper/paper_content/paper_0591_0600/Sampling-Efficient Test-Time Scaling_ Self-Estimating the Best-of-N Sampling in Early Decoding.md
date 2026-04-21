Title: Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding
Abstract: Test-time scaling enhances large language model performance by allocating additional compute resources during inference. Best-of-N (BoN) sampling serves as a common sampling-based scaling technique, broadening the search space in parallel to find better solutions from the model distribution. However, its cost-performance trade-off is still underexplored. Two main challenges limit the efficiency of BoN sampling: (1) Generating N full samples consumes substantial GPU memory, reducing inference capacity under limited resources. (2) Reward models add extra memory and latency overhead, and training strong reward models introduces potential training data costs. Although some studies have explored efficiency improvements, none have addressed both challenges at once. To address this gap, we propose Self-Truncation Best-of-N (ST-BoN), a decoding method that avoids fully generating all N samples and eliminates the need for reward models. It leverages early sampling consistency in the model's internal states to identify the most promising path and truncate suboptimal ones. In terms of cost, ST-BoN reduces dynamic GPU memory usage by over 80% and inference latency by 50%. In terms of cost-performance trade-off, ST-BoN achieves the same performance as Full-BoN while saving computational cost by 70%-80%, and under the same cost, it can improve accuracy by 3-4 points.

Section: 
[START] We know that […] 5 $. Therefore , we have that $. 6.
[…] 21 }\ cdot
[START] We have that […] 6 ^ 2 \ cdot 10 ^ 2 = […] 2 \ cdot 5 ^ 2 […] =\ boxed { 225 }. [EOS] Sampling 1 Sampling 2 Sampling 3 Sampling 4 Sampling 5 Earliest Estimation Time: all pairwise inconsistent Self-Estimating the (most likely) Best Sampling Index (Length 𝑐) Large Language Model [START] We have that […] 6 ^ 2 = 2 ^ 2 \ […] \ begin { [START] We have that […] 15 = 3 \ cdot 5 ,\ quad […] cdot 3 ^ [START] We can write […] Therefore , we can write $$ 6 ^{ […] 2 ^{ 21 Buffer Window (Length 𝜏) 2 3 2 3 4 … 2 2 2 Sampling 2
this section cite: []

Section: Self-Truncating

this section cite: []

Section: Final Answer
(Length 𝑇 -𝑐 -𝜏) 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
this section cite: []

Section: Introduction
Large Language Models (LLMs) possess strong generative and reasoning capabilities after extensive training on large-scale corpora [7,1,10], and the excellent solution to a problem is typically embedded within the model distribution [26,9,48,53,30]. However, autoregressive decoding focuses on locally optimal solutions, overlooking more promising thinking paths. Hence, the test-time scaling technique [40] is proposed to help LLMs expand their thinking space by adding compute during inference.
Best-of-N (BoN) [42,28,40] serves as a widely used sampling-based scaling paradigm. By sampling N responses from the LLM and selecting the best one with a re-ranking strategy at inference time, BoN fully leverages the potential ability within the model distribution. The core of BoN sampling lies in scoring and re-ranking multiple candidates. A well-known method is self-consistency [47], which selects the most frequent answer as the final solution. Furthermore, with a trained reward model (RM) [45,46,35], BoN sampling can handle richer tasks by scoring candidates and selecting the one with the highest score, aiming to increase test-time computational costs regardless of expense to enhance performance [40].
Beyond performance, the issue of cost-performance trade-offs in BoN sampling has been less explored. Under existing paradigms, BoN sampling faces two key challenges that hinder efficient deployment: (i) Full Generation Overhead: Traditional BoN requires fully generating all N samples, referred to as Full-BoN in this paper. Although GPU parallelization can partially mitigate time latency, additional memory overhead is unavoidable, especially for complex reasoning with long generated sequences.
(ii) Limitations of Reward Models: RMs are effective, but occupy additional memory and time costs.
Meanwhile, training a strong RM is costly due to the scarcity of high-quality feedback data, and their domain-specific nature restricts generalizability across tasks (e.g., from math to open-ended QA).
Some efforts have been made to improve the efficiency of BoN sampling by using RMs to score and discard sampling prefixes [41], or to branch after pruning [25,36], thereby reducing some memory overhead of full generation. However, RM inference still incurs non-negligible latency, and sharing GPU memory with generators further restricts caching capacity. Moreover, the inherent limitations of RMs make these methods less suitable for objective tasks like reasoning (see Appendix B). Therefore, no existing method overcomes both challenges at once to achieve deeper efficiency optimizations.
To fill this gap, we introduce a novel decoding technique called Self-Truncation Best-of-N (ST-BoN). Inspired by the unsupervised advantage of consistency strategies [47], we first theoretically demonstrate the feasibility of foreshadowing the final consistency using sampling consistency in early decoding. Then, we design an internal consistency measure that uses latent space information [49] to identify and preserve the most promising sample among the N samples in early decoding. This framework allows us to effectively self-estimate and truncate suboptimal samplings in early decoding without relying on reward model intervention, thereby saving memory and speeding up generation.
The detailed pipeline of ST-BoN is illustrated in Figure 1, it makes the following contributionsfoot_0 :
• Cost: Compared to Full-BoN, ST-BoN truncates suboptimal samples in early decoding, thereby freeing up memory and reducing the dynamic GPU memory load by over 80%. This early truncation also accelerates later generations, reducing inference latency by up to 50%. Notably, these efficiency gains are achieved without accounting for the overhead of extra reward models.
• Cost-Performance Trade-offs: We conduct extensive experiments to demonstrate that ST-BoN strikes a strong balance between cost and performance. When reaching the performance of Full-BoN at the certain N values, ST-BoN can save computational cost by 70% to 80%. Also, when consuming similar computational costs, ST-BoN improves accuracy by 3 to 4 points. In addition, ST-BoN is applicable across a wide range of domains. These results show that ST-BoN can provide a flexible solution to balance cost and performance when faced with limited resources.
this section cite: ['b6', 'b0', 'b9', 'b25', 'b8', 'b47', 'b52', 'b29', 'b39', 'b41', 'b27', 'b39', 'b46', 'b44', 'b45', 'b34', 'b39', 'b40', 'b24', 'b35', 'b46', 'b48']

Section: Preliminary
Autoregressive Decoding in Language Models. Let p θ represent a language model. For a given prompt and question, p θ will generate a token sequence Y 1:T = y 1 y 2 ...y T autoregressively, where T is the generation length and y T = [EOS]. Each token y t is sampled as follows:
y t ∼ p θ (•|prompt, question, y ≺t ) ∈ R |V| ,(1)
where V is the vocabulary. In greedy decoding, the model selects the token with the highest probability of p θ to generate y t . In contrast, sampling decoding typically uses multinomial sampling, such as topk [13] or top-p [20], to let models sample y t from p θ after truncation to generate diverse sequences.
Best-of-N Sampling. As the name suggests, Best-of-N sampling involves sampling N sequences to find the best one. Let Y 1 , Y 2 , . . . , Y N be N independent sampling sequences, then each sample Y i is assigned a scalar score S(Y i ), and the best sample index is identified as arg max i {S(Y i )} N i=1 . The simplest scoring approach is to use an externally trained reward model. If avoiding the intervention of reward models, the consistency strategy is the most common unsupervised alternative, introduced by self-consistency [47], which found a positive correlation between response consistency and accuracy; that is, when multiple reasoning paths lead to the same answer, confidence in its correctness increases. In self-consistency, the exact answer A i is extracted from Y i , and the consistency score S(Y i ) is computed based on the frequency with which A i co-occurs with other answers:
S(Y i ) = N j=1,j̸ =i I(A i = A j ) N -1 ,(2)
where I(•) is the indicator function. More generally, "consistency" can be any continuous measure. For example, [21] suggested using semantic similarity to assess sample consistency in open-ended tasks, calculated through the distance between sentence embeddings.
this section cite: ['b12', 'b19', 'b46', 'b20']

Section: ST-BoN: Self-Truncation Best-of-N Decoding

this section cite: []

Section: Theoretical Support: Early Consistency Foreshadows Final Consistency
We aim to overcome the efficiency bottlenecks of both full generation and reward models. Without a reward model, consistency strategies offer a promising unsupervised approach to select optimal samples. However, existing consistency methods require full generation. Naturally, we hope that early consistency can foreshadow the final consistency: a sample closer to others in early decoding is more likely to reach the correct answer when the decoding ends. If this holds, such consistency propagation can be utilized to estimate the most promising sampling in early decoding.
Intuitively, considering the cumulative effect of autoregressive generation, sampling sequences that already exhibit large differences in early decoding is less likely to converge closely by the end. Theoretically, this can be modeled as probabilistic monotonicity: as early consistency increases, the probability lower bound for final consistency surpassing a given constant also increases.
Consistency can be mathematically converted into a distance measure, where higher consistency implies a smaller distance. Let D(•, •) : Y × Y → R + be an arbitrary metric function defined over a metric space, used to quantify the distance between two partial sequences Y i 1:t and Y j 1:t at time t. We demonstrate that early consistency can probabilistically enhance the likelihood of final consistency. Specifically, let Y 1 denote the primary sampling. At an early decoding time t (t < T ), if the distance between the partial sequence Y 1 1:t and the other N -1 partial sequences {Y i 1:t } 2≤i≤N becomes smaller under a given metric D, the lower bound of the probability that the distance between the final full sequence Y 1  1:T and the other N -1 full sequences {Y i 1:T } 2≤i≤N is less than a certain value will increase, meaning it is more likely to meet a specific consistency requirement.
Theorem 1. Let d i t = D(Y 1 1:t , Y i 1:t ) denote the distance between sampling sequence Y 1 1:t and the i-th sampling sequence Y i 1:t at decoding time t. For a given constant ϵ, there exist a constant Γ such that: Pr [S T ≤ ϵ | S t ] ≥ 1 -Γ T -t ϵ S t , where 1 ≤ t < T , S T = N i=2 d i T and S t = N i=2 d i t . Proof Sketch. We assume (i) local Lipschitz continuity, i.e., for any two prefixes, the next-token distributions differ in TV distance by at most L times their sequence distance, and (ii) bounded increments, so appending a single token can increase the distance by at most M . Under maximal coupling, two continuations diverge at step t + 1 with probability at most Ld i t . Thus the expected distance evolves as E[d i t+1 | d i t ] ≤ d i t + M (Ld i t ) = (1 + LM ) d i t . Summing over i yields E[S t+1 ] ≤ ΓS t with Γ = 1 + LM . Iterating this recursion shows that deviations grow at most exponentially: E[S T ] ≤ Γ T -t S t . Finally, applying Markov's inequality converts the expectation bound into a high-probability guarantee: Pr[S T ≥ ϵ | S t ] ≤ Γ T -t S t /ϵ, implying the claim in Theorem 1.
The detailed proof is shown in Appendix A. As S t increases, the probability lower bound of S T ≤ ϵ also increases. This confirms the potential for early consistency that foreshadows the final consistency.
this section cite: []

Section: Method: Early Internal Consistency Estimates the Most Promising Sampling
We have proven the high-probability foreshadow from early consistency to final consistency, providing theoretical feasibility for utilizing sampling consistency in early decoding to estimate the most promising path. Upon this, we are ready to present the main ST-BoN algorithm, which involves defining an effective measure of sampling consistency by designing a suitable metric function D.
When Does Self-Estimation Start? Before defining the measure, we first identify the earliest self-estimation time t for early consistency. We aim to achieve ultimate efficiency improvement, so the earliest time can occur when all samplings become pairwise inconsistent. Let this special time be c, it satisfies the following condition:
i,j,i̸ =j I Y i 1:c = Y j 1:c = 0 & ∀t < c, i,j,i̸ =j I Y i 1:t = Y j 1:t > 0.(3)
In practical inference, we enable GPU parallelization and perform pairwise sequence-equality checks at each time. If any sample reaches [EOS] during this process, it is terminated immediately, and the current time step is recorded as c. In practice, this situation is very rare, as the condition in Eq.3 is typically satisfied early on, as shown in Figure 3.
this section cite: []

Section: Main Algorithm: How is Self-Estimation Done?
Now we start to define an effective measure of sampling consistency. At such an early decoding stage, semantic information is limited, and textual differences between samplings are minimal, making it hard to distinguish them solely based on output text. Therefore, we consider utilizing the more informative hidden states of LLMs to represent early sampling sequences through the model's internal information.
Chain-of-Embedding (CoE) offers a promising idea, which is the representation technique in the latent space [49] that links all hidden states from input to output, capturing the latent thinking path from reading to writing. It extracts hidden states across layers to form a progressive chain. Consider a model with L layers and an output length T , then the hidden state at layer l (0 ≤ l ≤ L) and position t (1 ≤ t ≤ T ) is represented as z l t . The sentence embedding for layer l is calculated as h T l = 1 T T t=1 z l t [38,50]. Thus, CoE is defined as a progressive chain of sentence embeddings:
H 1:T := h T 0 → h T 1 → • • • → h T L .
The CoE feature F(H) is quantified as:
F(H 1:T ) = 1 L • L-1 l=0 M (h l , h l+1 ) M (h 0 , h L ) - A(h l , h l+1 ) A(h 0 , h L ) ,(4)
where
M (h i , h j ) = ||h i -h j || 2 and A(h i , h j ) = arccos h ⊤ i • h j / (||h i || 2 • ||h j || 2 )
. A larger F(H 1:T ) indicates a greater curvature in the latent thinking path when generating Y 1:T .
Using the CoE measure, we define the distance D(Y i 1:c , Y j 1:c ) between two partial sequences Y i 1:c and Y j
1:c at time c as the squared difference of their CoE features. The consistency score S(Y i 1:c ) is then calculated as the average distance between the Y i 1:c and the other N -1 samples. Specifically:
D(Y i 1:c , Y j 1:c ) = F(H i 1:c ) -F(H j 1:c ) 2 , S(Y i 1:c ) = 1 N -1 N j=1,j̸ =i D(Y i 1:c , Y j 1:c ).(5)
A smaller S(Y i 1:c ) indicates higher consistency between the i-th sample and the other samples. After computing scores for all N samples, we rerank them and choose the sample with the lowest score: i c = arg min i {S(Y i 1:c )} N i=1 , making the i c -th sample the optimal estimate at time c.
How Long Does Self-Estimation Last? Performing self-estimation at a single moment can introduce randomness, since pairwise sequence differences only begin to emerge at time c, and these differences may not be substantial. To migrate this, we define a Buffer Window: the LLM continues to generate for an additional τ steps beyond time c, where τ is the window length. We set τ ∝ c, i.e., τ = mc, where m is a proportional constant. The intuition is that a smaller c indicates an early divergence between samplings, which reflects greater randomness, so a smaller window may suffice to capture later differences. In contrast, a larger c suggests lower randomness, which requires a larger window to capture sufficient later differences. At each time c ′ within the buffer window, we obtain the optimal estimation i c ′ = arg min i {s(Y i 1:c ′ )} N i=1 using Eq.5. This produces τ + 1 optimal estimation {i c ′ } c+τ c ′ =c , and the final optimal estimation i final is determined by selecting the most frequent one:
Count(i) = c+τ t=c I(i t = i) (1 ≤ i ≤ N ), i final = arg max i {Count(i)} N i=1 .(6)
this section cite: ['b48', 'b37', 'b49']

Section: Cost Analysis: Towards Efficient Best-of-N Sampling
Compared to Full-BoN, ST-BoN reduces significant computational cost during inference, which is reflected in two aspects: Space and Time. The space is related to GPU memory overhead, and the time is reflected in inference latency. We will analyze the two parts between ST-BoN and Full-BoN. During model inference, GPU memory overhead is mainly impacted by the model weights and the KV cache. While model weight loading is necessary, the KV cache is the main memory bottleneck. Autoregressive parallel sampling can quickly cause Out-of-Memory (OOM) issues due to KV cache accumulation, with OOM events tied to peak memory overhead. Therefore, we assess ST-BoN's memory optimization by comparing its reduction in peak memory overhead to that of Full-BoN.
Assuming that N samples are generated in parallel on the GPU (i.e., the batch size is N ), the KV cache usage is linearly related to the sampling size N and the generation length T . In the dataset D, for inputs X ∼ D and outputs Y ∼ p θ (•|X), we let P T be the length distribution of full generation, and P c be the distribution of the earliest estimation time c. When excluding basic model weight memory: (i) For Full-BoN without reward models, peak memory occurs at time T with batch size N . Note that if a reward model is added, the basic memory overhead of Full-BoN will increase, further reducing the available inference space. (ii) For ST-BoN, the peak memory can occur at time c with batch size N , or at time T with batch size 1. We set m = 1 in our main experiments, so that τ = c, and the memory reduction rate R D on dataset D can be approximated as:
R D = 1 - max {N • (E c∼Pc [c] + τ ), E T ∼P T [T ]} N • E T ∼P T [T ] = 1 -max 2 • E c∼Pc [c] E T ∼P T [T ] , 1 N .(7
) MATH TheoremQA MMLU 0 100 200 300 400 500 600 Token Length 22.63 29.97 27.01 536.07 378.02 332.25 c Pc[c] T PT[T] 0 10 20 30 40 Sampling Size N 0.65 0.70 0.75 0.80 0.85 0.90 0.95 Token Reduction Rate R MATH TheoremQA MMLU Figure 3: (Left) The mathematical expectations E c∼Pc [c] and E T ∼P T [T ] of the earliest estimation time c and the full generation length T across different datasets; (Right) The computed memory reduction rate R D under different sampling sizes N .
We conduct statistical experiments on the distributions P c and P T using the Llama3-8B-Instruct model across three datasets: MATH [18], TheoremQA [8], and MMLU [17]. Figure 2 visualizes the respective distributions, showing a clear shift to the left of P c compared to P T . This indicates that the earliest estimation time c appears much earlier than the completion of sampling. Furthermore, we calculate E c∼Pc [c] and E T ∼P T [T ], set various sampling sizes N , and use Eq.7 to find R D . Figure 3 illustrates that R D easily surpasses 80% for N ≥ 5. According to Eq.7, the upper limit of R D is 1 -2E c∼Pc [c]/E T ∼P T [T ], and increases with the sampling size N until it reaches this limit. In addition, significant variations in E T ∼P T [T ] compared to minimal variations in E c∼Pc [c] make domain optimization primarily dependent on E T ∼P T [T ].
For example, the MATH dataset has a longer full generation, leading to a higher optimization upper limit than other domains. In summary, R D increases with the sampling size N and the complexity of the task.
this section cite: ['b17', 'b7', 'b16']

Section: Inference Latency: Wall-Clock Time
Inference latency is another critical aspect that affects computational cost. In Full-BoN, while N -sampling can run in parallel on the GPU, it still introduces more delay compared to single-sample inference. Our ST-BoN method leverages early self-truncation, enabling LLMs to resume singlesample generation after time c + τ . However, sequence equality checks and self-estimation within the buffer window require serial vector processing, potentially adding latency. To address this, we will evaluate inference latency using wall-clock time as the measure. We perform statistical experiments using the Llama3-8B-Instruct model on the MATH and TheoremQA datasets. For Full-BoN, we assess scenarios with and without a reward model, highlighting the extra time needed to load and run the reward model. Figure 4 presents the results, demonstrating that ST-BoN significantly reduces inference latency compared to Full-BoN. Overall, wall-clock time latency drops by nearly 50%, with the reduction ratio growing as N increases. This efficiency stems from the self-truncating operation, which compensates for delays from self-estimation. Additionally, incorporating a reward model into Full-BoN further increases inference latency.
this section cite: []

Section: Experiments

this section cite: []

Section: Setup
Datasets. We select four datasets for objective tasks: MATH [18], TheoremQA [8], GPQA [37], and MMLU [17]. They span a range of domains, including mathematics, theorem application, science reasoning, and general knowledge, and present a significant difficulty level. We also select two datasets for subjective tasks: CNNDM [34] and AlpacaFarm [11]. The former is the summarization task about the open-ended generation, and the latter is the instruction-following task about the preference alignment. We adopt the revised CNNDM [51] for its higher-quality gold summaries.
this section cite: ['b17', 'b7', 'b36', 'b16', 'b33', 'b10', 'b50']

Section: Models.
We mainly adopt 7B+ parameter models with the Zero-Shot-CoT generation paradigm [52,23], including Qwen2.5-7B-Instruct [55], Llama3-8B-Instruct [10], and Mistral-7B-Instruct-v0.3 [22]. We also test the Qwen2.5-72B-Instruct [55] to validate generalization across model scales.
Baselines. We compare our ST-BoN decoding with two Full-BoN decoding paradigms: Full-BoN without a reward model (Full-BoN w/o RM) and Full-BoN with a reward model (Full-BoN w/ RM).
In objective tasks: Full-BoN w/o RM uses the classic self-consistency through majority voting [47], and Full-BoN w/ RM employs the process reward model (PRM) Skywork-o1-Open-PRM-7Bfoot_1 foot_2 . In subjective tasks: Full-BoN w/o RM applies the modified self-consistency through semantic similarity [21], and Full-BoN w/ RM uses the ArmoRM-Llama-3-8B [45] for preference rewards.
Implementation. We use the sampling strategy combining top-k [13], top-p [20], and temperature T [19], with k = 20, p = 0.95, and T = 0.7. The buffer window length τ is set to c in our main experiments. Detailed hyperparameter analysis is provided in Section 6.2 and 6.3. All baselines are implemented using the HuggingFace Transformers   Evaluation. We mainly evaluate the balance between computational cost and performance. The computational cost lies in two dimensions in Section 4: memory and time. We use the cost of greedy decoding as the baseline, and the cost of each paradigm is calculated as follows:
• Memory Cost M cost : Let M bm , M peak , and M rm denote the memory usage of the base LLM weights, the peak memory overhead during inference, and the reward model weights, respectively, for each paradigm. Let M greedy peak denote the peak memory during greedy decoding:
M cost := (M bm + M peak ) / (M bm + M greedy peak ), ST-BoN & Full-BoN w/o RM, (M bm + M peak + M rm ) / (M bm + M greedy peak ), Full-BoN w/ RM.(8)
• Time Cost T cost : T cost is denoted as the ratio of the wall-clock time of each paradigm to that of greedy decoding. For Full-BoN w/ RM, we consider that both the generator and the verifier are persistently stored in GPU memory, and thus ignore the loading time of the RM.
The overall computational cost A cost is given by:
A cost := M cost • T cost .(9)
Regarding performance, Accuracy is used as a metric in reasoning scenarios. To reduce random error, we follow [16] to report the average results over four evaluation runs. In open-ended scenarios, Rouge-L [29] and Human Scoring are applied for the summarization, and Win Rate (WR) [59] judged by GPT-4o-Turbo [1] is used for the instruction-following. Metric details are shown in Appendix C. MATH MMLU 0.0 0.2 0.4 0.6 0.8 1.0 Consistency 0.56 0.63 0.35 0.43 0.17 0.25 0.22 0.24 1 Random Estimation w/ CoE; w/ Buffer Window (Standard ST-BoN) w/ CoE; w/o Buffer Window w/ Semantic; w/ Buffer Window w/ String; w/ Buffer Window MATH MMLU 0.0 0.2 0.4 0.6 0.8 1.0 0.82 0.88 0.72 0.56 0.45 0.52 0.36 0.33 MATH MMLU 0.0 0.2 0.4 0.6 0.8 1.0 0.92 0.95 0.76 0.74 0.78 0.82 0.65 0.63 MATH MMLU 0.0 0.2 0.4 0.6 0.8 1.0 1.0 1.0 0.98 0.95 0.91 0.93 0.86 0.83
Figure 6: The early-final consistencies between early self-estimation and final correctness in different domains and sub-datasets with the Qwen2.5-7B-Instruct model. D i represents the subset containing all cases from dataset D that produces i correct answers out of N samplings.
foreshadows final consistency -is reasonable and holds in practice. In addition, Full-BoN w/ RM outperforms the others in three datasets more related to reasoning (except MMLU), which aligns with the test-time scaling goal of "increasing inference cost to improve performance" [40]. On the other hand, under the same computational cost, ST-BoN can scale to a much larger N than Full-BoN. For example, when Full-BoN w/o RM uses N = 10 and Full-BoN w/ RM uses N = 3, ST-BoN can reach nearly N = 80. This gives it access to much more samples, leading to a 3-4 point higher accuracy across all datasets. These results show that under the same computational cost, ST-BoN can achieve better performance than Full-BoN.
Finally, we also observe the domain generalization of ST-BoN. Full-BoN w/o RM performs noticeably worse on MMLU, a knowledge dataset, compared to the other three reasoning datasets. In contrast, ST-BoN maintains its cost-performance trade-off advantage. This suggests that Full-BoN w/o RM relies on a strong and general RM to be effective, while ST-BoN avoids this potential cost.
By not requiring domain-specific priors, ST-BoN remains effective across diverse domains.
this section cite: ['b51', 'b22', 'b54', 'b9', 'b21', 'b54', 'b46', 'b20', 'b44', 'b12', 'b19', 'b18', 'b15', 'b28', 'b58', 'b0', 'b39']

Section: Results II: Subjective Tasks
To evaluate the domain generalization of ST-BoN, we further test it on subjective tasks. The results for Qwen2.5-7B-Instruct are shown in Figure 5b, with results of other models shown in Appendix D.1. First, Full-BoN w/o RM performs poorly, indicating that semantic consistency cannot identify optimal samples. Second, Full-BoN w/ RM underperforms on summarization tasks due to a domain mismatch: the reward model is typically trained on open-domain QA preferences, which rarely cover summarization, leading to OOD issues. Finally, on instruction-following tasks, although Full-BoN w/ RM outperforms ST-BoN at the same sampling size N , ST-BoN can reach its performance by increasing N , while reducing computational cost by approximately 50%. This highlights ST-BoN's cost-performance trade-off on subjective tasks and further demonstrates its domain robustness. 6 In-depth Analysis 6.1 Component Ablation: Why ST-BoN Works?
Our method is motivated by the idea that early consistency may foreshadow final consistency, then we use CoE to self-estimate early consistency, and propose a buffer window to mitigate noise. In this part, we will ablate the two components: CoE measure and buffer window, to evaluate their necessity.
Baseline. The standard ST-BoN involves "w/ CoE & w/ Buffer Window". First, we remove the buffer window, allowing the LLM to perform only one self-estimation at the earliest estimation time c. Next, to further assess the necessity of using hidden states of LLMs (i.e., CoE) for early consistency computation, we compare two unsupervised output measure: semantic and string. Specifically, in Eq.5, we replace D(Y i 1:c , Y j 1:c ) with ||[h i 1:c ] L -[h j 1:c ] L || 2 2 and 1/Rouge-L(Y i 1:c , Y j 1:c ), respectively. Evaluation. We assess the Early-Final Consistency between early self-estimation and final correctness. Specifically, assume each case in dataset D is sampled N times. The dataset is divided 1.4x 1.8x 2.2x 2.6x Computational Cost (The Factor Relative to Greedy) 76 77 78 79 Accuracy p=0.95, T=0.7 k = 25 k = 20 k = 15 k = 10 1.4x 1.8x 2.2x 2.6x 76 77 78 79 k=20, T=0.7 p = 0.99 p = 0.95 p = 0.90 p = 0.85 1.4x 1.8x 2.2x 2.6x 76 77 78 79 k=20, p=0.95 T = 1.2 T = 1.0 T = 0.7 T = 0.5 i=0 , where D i contains cases that yield i correct answers out of N samples. For each D i , we compute the proportion of cases where the LLM's self-estimated best sampling produces a correct final answer, representing early-final consistency. Naturally, as i decreases, the estimation becomes more challenging, and the random consistency expectation is i/N .
Results. We evaluate on the MATH and MMLU datasets due to their diverse domains and difficulty levels. With N = 5, we exclude subsets D 0 and D 5 , as they represent cases that are entirely correct or incorrect. Figure 6 presents the results using the Qwen2.5-7B-Instruct model (Results for other models are in Appendix D.3), we find that: (i) "w/ buffer window" consistently outperforms that "w/o buffer window", demonstrating its effectiveness. (ii) For all subsets, "w/ CoE" significantly outperforms random estimation, with standard ST-BoN achieving excellent consistency. In contrast, "w/ semantic" and "w/ string" hover around the random baseline, highlighting the critical role of CoE in early consistency measure. Thus, each module in the current ST-BoN plays an essential role.
In fact, we have also discovered that using CoE consistency under full generation, yields better results than using self-consistency with majority voting. This provides further evidence that CoE's effectiveness in measuring early consistency is not random. The results are shown in Appendix D.4.
this section cite: ['b5']

Section: Window Length τ Ablation
In ST-BoN, the buffer window length τ is the only hyperparameter. We study its impact by varying τ in different datasets with Qwen2.5-7B-Instruct, with a fixed sampling size N = 10.
Figure 7 shows that when τ < c, the performance gain relative to cost increases faster. However, beyond c, this gain slows significantly, making c the optimal choice for balancing performance and cost in our experiments. Furthermore, if ignoring costs, performance continues to show fluctuating improvements as τ increases, suggesting the potential of task-specific τ tuning. For example, on the MATH dataset, performance improves more steadily with larger τ values, probably due to its longer outputs, where early estimates are less informative, and need a larger buffer window to capture more information. Adaptive τ based on task complexity can be a promising direction for future work.
this section cite: []

Section: Sampling Strategy Robustness
We also evaluate the robustness of ST-BoN across different sampling strategies by varying top-k, top-p, and temperature T settings, and analyze their impact on performance and computational cost.
Figure 8 shows the results on the MMLU dataset using the Qwen2.5-7B-Instruct model. We observe that variations in k and p have a minimal impact on performance and cost, while changes in T affect both more significantly. Unlike k and p, which only shift the truncation point of the probability distribution without affecting token relativity, T alters token probabilities, making the distribution more uniform and increasing sampling randomness as T increases. This results in an earlier occurrence of the earliest estimation time c, thereby reducing computational cost. However, an earlier estimation time c might capture less effective information, potentially as a side effect of improved efficiency. We recommend using a moderate T value, such as 0.7, to strike a balance between performance and cost.
this section cite: []

Section: Case Study
We also conduct case studies for further comparison. We find that ST-BoN can better address ambiguous scenarios, such as when sampling answers are all inconsistent or majority voting encounters high randomness, compensating for these limitations. Details are available in Appendix E.
this section cite: []

Section: Related Work and Comparisons

this section cite: []

Section: Test-Time Scaling
BoN aims to achieve test-time scaling [40]. The idea is to increase the compute during inference while setting cost aside, to push the performance ceiling. Existing approaches fall into two broad lines [32]: Sequential scaling lengthens a single reasoning path during generation and seeks deeper deliberation, reflection, or insight [16]. Parallel scaling, which captures the spirit of BoN, increases the number of reasoning paths generated at once and seeks greater diversity in reasoning [47,6]. Other methods, such as ToT [57], GoT [5], and Monte Carlo Tree Search [44], share the same objective but expand from a reasoning-structure perspective, taking different research tracks than BoN.
this section cite: ['b39', 'b31', 'b15', 'b46', 'b5', 'b56', 'b4', 'b43']

Section: Best-of-N Sampling
BoN sampling was originally proposed for inference-time preference alignment [33], using a reward model to select the best sample and further fine-tune the base model [42]. Its effectiveness has been theoretically supported by several studies [12,31,4,56]. BoN has also seen broad adoption in reasoning tasks [40], notably in the self-consistency paradigm [47], which uses majority voting to select answers unsupervisedly. More recently, process reward models [28,46] have enabled direct scoring within BoN for reasoning. However, reward models raise computational costs and are vulnerable to overoptimization, and rely heavily on their model capabilities [14].
this section cite: ['b32', 'b41', 'b11', 'b30', 'b3', 'b55', 'b39', 'b46', 'b27', 'b45', 'b13']

Section: Efficient Best-of-N
The core optimization idea of ST-BoN can be summarized as "depth pruning", which eliminates some suboptimal samples in early decoding to reduce memory usage. SBoN [41] also considers removing local segments; CARDS [25] and TreeBoN [36] continue to branch after discarding them. However, these methods still rely on a reward model for scoring, so the additional inference latency and memory usage remain high. More importantly, they have been validated only under the preference alignment, but in reasoning domains, reward models may be difficult to score randomly segmented process fragments. Appendix B offers a detailed comparison, highlighting the greater efficiency of ST-BoN and its broader applicability in different domains.
Another type of optimization idea can be summarized as "breadth pruning", aiming to adaptively reduce the actual sampling number under a fixed budget N . For example, ASC [2] stops sampling early based on answer frequency, and ESC [27] dynamically adjusts the required sampling sizes from an entropy perspective. Although these methods lower token output and reduce memory usage, they rely on serial sampling with adaptive stopping, which compromises the parallelism benefits of BoN and limits efficiency on modern GPUs. As a result, they are suboptimal in terms of inference latency.
Certainly, another research line studies distilling the BoN policy during training to replace multiple samplings with a single one, reducing inference-time latency. Methods like BoNBoN [15], BOND [39], and vBoN [3] focus on this line, shifting the computational cost from the inference phase to the training phase. Overall, this line explores more about the potential of learning the BoN distribution at training time, making orthogonal contributions compared to direct inference-time optimization.
this section cite: ['b40', 'b24', 'b35', 'b1', 'b26', 'b14', 'b38', 'b2']

Section: Conclusion
We propose ST-BoN decoding, which enables LLMs to self-estimate the most promising sampling without fully generating N samples or using reward models. ST-BoN significantly reduces GPU memory overhead and inference latency while demonstrating better cost-performance trade-offs than Full-BoN in both objective and subjective tasks from reasoning to preference alignment.
this section cite: []

Section: Limitations
• (1) Model Transparency: Since accessing the hidden states of the LLM is required, ST-BoN does not apply to closed-source models such as OpenAI's GPT-4 series [1]. However, with the rapid development of open-source LLMs [16], we believe that research on white-box methods is crucial, as it can provide stronger interpretability and help us better explore the internal mechanisms of LLMs.
• (2) Length Adaptivity: As mentioned in Section 6.2, there is still room for improvement in ST-BoN's ability to adaptively adjust the window length τ based on task complexity. Longer responses may require longer windows to capture more information. To better balance the cost, we can observe the average generation length of the task during greedy decoding and adjust the window size accordingly. This ensures that performance is maximized within an acceptable cost range.
this section cite: ['b0', 'b15']

Section: Societal Impact
• (1) Trustworthiness: ST-BoN relies on internal consistency signals to guide decoding, moving the selection process into the model's latent space. Prior work [47] has shown a strong correlation between consistency and correctness. Building on this, our theoretical analysis (Section 3.1) and empirical findings (Section 6.1) further support the reliability of latent-space decision-making. Together, these two parts provide a solid foundation for the model's trustworthiness under our method.
• (2) Safety: ST-BoN introduces no additional safety risks. The sampling follows the model's native distribution without adversarial perturbation, and the consistency-based selection mechanism operates without external reward models, thus avoiding interference from out-ofdistribution signals.
• (3) Biases: ST-BoN inherits the base model's biases, as it does not perform explicit debiasing.
However, by leveraging multiple parallel samplings, it can identify and truncate transient biases that arise sporadically during generation. These biased trajectories may be excluded as outliers through the consistency-based strategy, thereby potentially mitigating their impact.
this section cite: ['b46']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Let's sample step by step: Adaptiveconsistency for efficient reasoning and coding with llms Year: (2023)
Ref_id:b2 Title:  Year: (2024)
Ref_id:b3 Title: Chirag Nagpal, and Ananda Theertha Suresh. Theoretical guarantees on the best-of-n alignment policy Year: (2024)
Ref_id:b4 Title: Graph of thoughts: Solving elaborate problems with large language models Year: (2024)
Ref_id:b5 Title: Large language monkeys: Scaling inference compute with repeated sampling Year: (2024)
Ref_id:b6 Title: Language models are few-shot learners Year: (2020)
Ref_id:b7 Title: Theoremqa: A theorem-driven question answering dataset Year: (2023)
Ref_id:b8 Title: Do not think that much for 2+ 3=? on the overthinking of o1-like llms Year: (2024)
Ref_id:b9 Title: The llama 3 herd of models Year: (2024)
Ref_id:b10 Title: Alpacafarm: A simulation framework for methods that learn from human feedback Year: (2024)
Ref_id:b11 Title: Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking Year: (2023)
Ref_id:b12 Title: Hierarchical neural story generation Year: (2018)
Ref_id:b13 Title: Scaling laws for reward model overoptimization Year: (2023)
Ref_id:b14 Title: Bonbon alignment for large language models and the sweetness of best-of-n sampling Year: (2024)
Ref_id:b15 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b16 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b17 Title: Measuring mathematical problem solving with the math dataset Year: ()
Ref_id:b18 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b19 Title: The curious case of neural text degeneration Year: (2019)
Ref_id:b20 Title: Lightweight reranking for language model generations Year: (2024)
Ref_id:b21 Title: Mistral 7b Year: (2023)
Ref_id:b22 Title: Large language models are zero-shot reasoners Year: (2022)
Ref_id:b23 Title: Neural text summarization: A critical evaluation Year: (2019)
Ref_id:b24 Title: Cascade reward sampling for efficient decoding-time alignment Year: (2024)
Ref_id:b25 Title: Common 7b language models already possess strong math capabilities Year: (2024)
Ref_id:b26 Title: Escape sky-high cost: Early-stopping self-consistency for multi-step reasoning Year: (2024)
Ref_id:b27 Title: Ilya Sutskever, and Karl Cobbe. Let's verify step by step Year: (2023)
Ref_id:b28 Title: Rouge: A package for automatic evaluation of summaries Year: (2004)
Ref_id:b29 Title: icl: Demonstration-retrieved in-context learning Year: (2024)
Ref_id:b30 Title: Controlled decoding from language models Year: (2023)
Ref_id:b31 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b32 Title: Browser-assisted question-answering with human feedback Year: (2021)
Ref_id:b33 Title: Abstractive text summarization using sequence-to-sequence rnns and beyond Year: (2016)
Ref_id:b34 Title: Skywork o1 Team. Skywork-o1 open series Year: (2024-11)
Ref_id:b35 Title: Treebon: Enhancing inference-time alignment with speculative tree-search and best-of-n sampling Year: (2024)
Ref_id:b36 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2023)
Ref_id:b37 Title: Out-of-distribution detection and selective generation for conditional language models Year: (2022)
Ref_id:b38 Title: Aligning llms with best-of-n distillation Year: (2024)
Ref_id:b39 Title: Scaling llm test-time compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b40 Title: Fast best-of-n decoding via speculative rejection Year: (2024)
Ref_id:b41 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b42 Title: Attention is all you need Year: (2017)
Ref_id:b43 Title: Alphazero-like tree-search can guide large language model decoding and training Year: (2024)
Ref_id:b44 Title: Interpretable preferences via multi-objective reward modeling and mixture-of-experts Year: (2024)
Ref_id:b45 Title: Math-shepherd: Verify and reinforce llms step-by-step without human annotations Year: (2024)
Ref_id:b46 Title: Self-consistency improves chain of thought reasoning in language models Year: (2022)
Ref_id:b47 Title: Evaluating mathematical reasoning in multilingual contexts Year: (2025)
Ref_id:b48 Title: Latent space chain-ofembedding enables output-free llm self-evaluation Year: (2024)
Ref_id:b49 Title: Embedding trajectory for out-of-distribution detection in mathematical reasoning Year: (2024)
Ref_id:b50 Title: Element-aware summarization with large language models: Expert-aligned evaluation and chain-of-thought method Year: (2023)
Ref_id:b51 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b52 Title: Few-shot named entity recognition with joint token and sentence awareness Year: (2023)
Ref_id:b53 Title: An implementation of generative prm Year: (2024)
Ref_id:b54 Title: Qwen2.5 technical report Year: (2024)
Ref_id:b55 Title: Asymptotics of language model alignment Year: (2024)
Ref_id:b56 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2024)
Ref_id:b57 Title: The lessons of developing process reward models in mathematical reasoning Year: (2025)
Ref_id:b58 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
