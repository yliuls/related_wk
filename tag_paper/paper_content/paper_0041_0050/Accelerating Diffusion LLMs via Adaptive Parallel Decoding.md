Title: Accelerating Diffusion LLMs via Adaptive Parallel Decoding
Abstract: The generation speed of current LLMs is bottlenecked by autoregressive decoding, where tokens are predicted sequentially one by one. Alternatively, diffusion large language models (dLLMs) theoretically allow for parallel token generation, but in practice struggle to achieve the speed of autoregressive models without significantly sacrificing quality. We therefore introduce adaptive parallel decoding (APD), a novel method that dynamically adjusts the number of tokens sampled in parallel. We achieve this by defining a multiplicative mixture between the dLLM marginal probabilities and the joint probability of sequences under a small auxiliary autoregressive model. This inverts the standard setup of speculative decoding, where the goal is to sample from a large autoregressive verifier by drafting from a smaller model. We further optimize APD by enabling KV caching and limiting the size of the masked input. Altogether, our method puts forward three tunable parameters to flexibly tradeoff throughput and quality. We show that APD provides markedly higher throughput with minimal quality degradations on downstream benchmarks.

Section: Introduction
Large language models (LLMs) have remarkable text generation capabilities and have attracted evergrowing interest and widespread adoption. However, a significant impediment to their deployment lies in the speed of text generation [43]. The dominant paradigm, autoregressive models [32], generates tokens one by one in a sequential manner. While this approach has yielded state-of-the-art results in terms of quality, the inherent sequentiality of generation limits throughput and hinders real-time applications, especially as models continue to scale in size [19]. In addition, the recent phenomenon of test-time scaling [35] and reasoning models [13] suggest that generation speed will be an important bottleneck for increasing LLM capabilities. This challenge has spurred research into alternative approaches beyond autoregressive sequential sampling.
One promising alternative to purely sequential generation is offered by diffusion large language models (dLLMs) [42]. These models, inspired by successes in image generation [36], theoretically permit the parallel generation of multiple tokens simultaneously, offering a path towards significantly faster inference. In this work, we challenge the assumption that dLLMs, exemplified by the opensource models Dream [41] and Llada [29], can be practically used for parallel generation without additional modifications. We find that the quality of these models is best achieved by generating tokens one at a time (one timestep per token), and attempts to exploit the parallelizability of diffusion 39th Conference on Neural Information Processing Systems (NeurIPS 2025).
Prompt Autoregressive Decoding Adaptive Parallel Decoding Latency: 7.08 seconds Throughput: 37 tokens per second Latency: 2.75 seconds Throughput: 59 tokens per second models suffer a reduction in quality. Thus, current state-of-the-art dLLMs, namely Dream and Llada as currently conceived, fail to match the speed and quality of autoregressive LLMs.
This gap between theoretical and practical performance creates an opportunity for novel decoding mechanisms that can effectively harness the parallel generation capabilities of dLLMs while maintaining high fidelity to the target text distribution. The core challenge is that when sampling multiple tokens in parallel, one only has access to the marginal distribution of each token, which ignores inter-token dependencies [23]. A decoding algorithm that increases parallelism when sampling from dLLM and maintains generation quality must consider the joint distribution that captures these dependencies.
To address this challenge, we introduce Adaptive Parallel Decoding (APD), a novel decoding algorithm designed to dynamically modulate the number of tokens sampled in parallel during the generation process with dLLMs. In APD, we first fix the generation order of the dLLM to be left to right. This unexpected change makes the dLLM autoregressive, but we empirically show that it maintains generation quality and in some cases can improve it. By using the dLLM autoregressively, we can then also use a smaller autoregressive model to determine which subset of tokens sampled in parallel adequately captures joint dependence. Our criterion depends on a multiplicative mixture between the diffusion model and autoregressive model. Note that while speculative decoding uses a large model to verify samples from a small draft model, our task is to use a smaller verification model to check the quality of a larger model. This dynamic leads to a fundamentally new problem statement.
Our contributions extend beyond the conceptual framework of APD. We further optimize the decoding process by incorporating practical improvements to diffusion model sampling such as Key-Value (KV) caching, a technique traditionally associated with autoregressive models [31]. We also gain significant speed by limiting the size of the masked input to the model. These simple changes substantially increase the computational efficiency of diffusion model inference.
Collectively, we introduces three distinct tunable parameters. These parameters provide practitioners with the flexibility to navigate the inherent trade-off between generation throughput and output quality, allowing for tailored configurations that can meet diverse application-specific requirements. This paper will demonstrate that Adaptive Parallel Decoding offers a significant step forward in accelerating dLLM inference (and LLM inference in general). We present empirical evidence showing that APD achieves substantially higher throughput compared to existing LLM decoding strategies, all while incurring only minimal degradations in quality across a range of downstream benchmark tasks.
The subsequent sections will discuss the technical details of APD, present a comprehensive set of experiments validating our claims, and discuss the broader implications of our findings for the future of efficient LLM generation.
this section cite: ['b42', 'b31', 'b18', 'b34', 'b12', 'b41', 'b35', 'b40', 'b28', 'b22', 'b30']

Section: Background
In the following sections, we will define notation to be used throughout the paper. We consider a data point as a sequences of n tokens x = (x 1 , ..., x n ). For sets of indices Q, O ⊆ {1, ..., n} for which Q ∩ O = ∅, a masked language model p D ( • | • ; θ) computes the marginal probabilities of tokens with query indices Q conditioned on tokens with observed indices O.
p D (x Q | x O ; θ) = i∈Q p θ (x i | x O )(1)
where p θ is a learned conditional distribution parameterized by θ. In this work, a diffusion language model is a masked language model, and these terms can be used interchangeably.
this section cite: []

Section: Discrete Diffusion Language Models
Discrete diffusion models [26] have recently emerged as a promising alternative to traditional autoregressive approaches for language generation, offering benefits such as non-autoregressive generation capabilities and inherent iterative refinement [39]. Discrete diffusion models are masked language models [34] trained to reverse a data corruption process q that stochastically converts a clean sequence of tokens x 0 to a noisy x t , gradually converting clean tokens to [MASK] over time t
q t|0 (x t i | x 0 i ) =    t, if x t i = [MASK] 1 -t, if x t i = x 0 i 0 otherwise q t|0 (x t | x 0 ) = i q t|0 (x t i | x 0 i )(2)
Given this noise process, dLLMs are trained to maximize a lower bound on the log-likelihood computed using conditional distributions of clean data [30].
log p θ (x 0 ) ≥ E t∼U (0,1),x t ∼q(x t |x 0 ) 1 t log p D (x 1(x t =[MASK]) | x 1(x t ̸ =[MASK]) ; θ)(3)
While dLLMs are trained in theory to randomly unmask tokens at inference time, in practice decoding heuristics are used to determine the order of tokens to unmask [20]. For example, to achieve the best results, Dream unmasks tokens according to lowest entropy as assessed by the diffusion model at each timestep. Llada performs the best when unmasking according to the highest probability, which is referred to as "confidence" based decoding [29].
In light of these inference time modifications, diffusion language models are able to compete with autoregressive model in terms of generation quality. However, in realistic scenarios, the quality of a generation must be weighed against the speed. When we consider both of these factors, current open-source dLLMs are far from competitive with their open-source autoregressive counterparts.
In Table 1, we find that dLLMs can perhaps achieve competitive quality, but their throughput is a fraction of Qwen2.5 7B [40]. Another observation is that in order to achieve competitive performance on GSM8K [6], dLLMs must generate 256 tokens in 256 timesteps, which is actually sequential. We see that decreasing the number of timesteps can increase throughput at the expense of quality. Nonetheless, the throughput will not approach autoregressive speed without drastic reduction in quality. Interestingly, we find the simplest decoding order, just left to right, yields good results in most scenarios, and in the case of GSM8K the best results.
this section cite: ['b25', 'b38', 'b33', 'b29', 'b19', 'b28', 'b39', 'b5']

Section: From Sequential to Parallel Sampling
In this work, we utilize diffusion models with a particular noise schedule: denoising left to right. This is equivalent to sampling autoregressively and can be used to compute an exact autoregressive likelihood.
p AR (x; θ) = n i=1 p D (x i | x <i ; θ)(4)
To further parallelize this process, one can decode multiple tokens at a time semi-autoregressively [38] from groups of size k. Decoding in this manner samples from the following distribution
p SAR (x; θ, k) = ⌊(n-1)/k⌋+1 i=1 p D (x (ik-k+1:ik) | x <(ik-k+1) ; θ)(5)
where it follows that p AR (x; θ) = p SAR (x; θ, 1) for k = 1. A diffusion model can parallelize the term inside the product by sampling from each marginal distribution independently. Although this will lead to faster generations, the intra-group independence assumption is poor and leads to lower quality generations. We observe this empirically in Figure 2: as we increase the number of tokens sampled at a time, throughput increases, but downstream accuracy decreases. Thus, there exists a tradeoff between parallelization and speed with sample quality. However, instead of fixing k, our core insight is that the number of tokens to generate in parallel can be chosen adaptively. In this work, we argue that doing so significantly reduces the sharp tradeoff between speed and quality.
this section cite: ['b37']

Section: Method

this section cite: []

Section: Problem Statement
Our goal is to sample from groups G = {(s 1 , e 1 ), (s 2 , e 2 ), ..., (s l , e l )} where (s i , e i ) is a tuple that denotes start and end indices inclusive and e i + 1 = s i+1 for all i. Note that Equation 5 is a special
In choosing G, we must balance the following two goals -(1) Speed: minimize |G| and (2) Quality: minimize the distance between p APD and p AR .
Achieving only one out of two goals is trivial. For maximum speed, we can sample from all the marginals of the diffusion model in one shot such that G = {(1, n)} and |G| = 1. This would result in a significant drop in quality. In fact, the drop in quality can be quantified by mutual information
I pAR (x; θ) = KL(p AR (x 1:n ; θ) || p D (x 1:n ; θ))(7)
Alternatively, we can set G = {(1, 1)...(n, n)}, in which case |G| = n and p APD = p AR . This is very slow because it maximizes the number of sequential iterations. For architectural reasons to be discussed in Section 3.3, it is also very slow to sample autoregressively from a diffusion model because it cannot perform KV caching. This architectural difference distinguishes our problem setting from speculative decoding [22], because computing an autoregressive likelihood in a diffusion model is slow and sequential, whereas in an autoregressive model it is fast and parallelizable. While speculative decoding does not map onto our problem, we shall argue that adaptive parallel decoding achieves both goals together.
this section cite: ['b21']

Section: Adaptive Parallel Decoding
In adaptive parallel decoding, we assume to have access to a small autoregressive model pAR that can compute the likelihood of sequences in parallel. While p D only computes marginal probabilities over tokens, pAR computes a joint probability, allowing it to model dependencies between tokens.
For brevity, we shall omit model parameters θ.
To select G, we shall focus our analysis on a subproblem. Given samples xt , ..., xn ∼ p D (• | x <t ), we must select k such that xt:t+k are close to p AR . We do not have access to p AR because evaluating its likelihood for each token will give us the worst-case speed. Instead, we have access to a small, inaccurate model pAR that can quickly evaluate likelihood in parallel. To approximate p AR , we shall define a target distribution which we use to determine how many tokens k are accepted. This target distribution p T should satisfy the following desiderata:
(1) if p D (X = c) = 1 and pAR (X = c) < 1, then p T (X = c) = 1 (2) if p D (X = c) < 1 and pAR (X = c) = 1, then p T (X = c) = 1 Property (1) follows from the fact that p AR (x 1:k ) ≥ k i=1 p D (x i ) -k + 1 by Bonferroni's inequality [10], which means if p D (x i ) = 1 for all i, then p AR (x 1:k ) = 1. Put verbally, if the marginals given by the diffusion model are 1, its joint distribution or p AR will also have probability 1. Property (2) is difficult to justify formally because we do not have a relationship between p AR and pAR . Assuming pAR reasonably approximates the joint distribution, property (2) is a desirable heuristic because a token with joint probability 1 should be accepted.
A multiplicative mixture of distributions, also known as a product of experts [16], fulfills the two requirements above. We define the multiplicative mixture of p D and pAR as follows
p T (x) = 1 Z p D (x) R pAR (x) 1-R (8
)
where Z is the normalizing constant. Our mixture is defined in terms of a hyperparameter R ∈ [0, 1].
When it is high, it gives the diffusion model more weight. We may now give an accept criteria based on target distribution p T . We adopt an accept criteria similar to existing parallel sampling methods based on universal coupling [1]. Formally, a universal coupler is a function g which for a distribution p and source of randomness r satisfies
P r∼U [0,1] (g(p, r) = x) = p(x)(9)
For categorical distributions, the Gumbel-Softmax Trick [18] is a universal coupler [1]. An important property of a universal coupler is that for two distributions p, p ′ , their samples from a universal x ← concat(x, xt:t+k-1 )
▷ Append accepted tokens 14:
t ← t + k 15: end while 16: return x coupler with a shared source of randomness are likely to be the same given that the distributions p and p ′ are similar [21].
P r (g(p, r) ̸ = g(p ′ , r)) ≤ 2 TV(p, p ′ ) (10
)
where TV is total variation distance. Thus, in our algorithm we sample from the diffusion model using the Gumbel-Softmax trick as a universal coupler g and source of randomness r: xt , ..., xn ∼ g(p D , r).
We then sample from our target: ŷt , ..., ŷn ∼ g(p T , r), and we accept all tokens that are the same between ŷi and xi until the first disagreement. Because the goal is to sample from p AR , we always accept the first proposed token xt . The full algorithm is given in Algorithm 1. While the same algorithm is valid even when using different Gumbel randomness to sample from the target and proposal distributions, intuitively using the same randomness will maximize the number of accepted tokens.
Algorithm 1 provides a sampling procedure which takes as input a diffusion model p D , a small autoregressive model pAR , and tunable parameter R ∈ [0, 1]. When R = 1, the target distribution is p D and the algorithm will accept every token from the diffusion model in one shot. When R = 0, the algorithm does not "trust" the diffusion model and instead only accepts tokens that pAR accepts.
Our sampling algorithm is agnostic to the implementation of p D and pAR , but in practice architectural optimizations are used for additional speed. For example, when computing the joint logits of the autoregressive model, we can use KV caching [31] to avoid redundant computation on t previous tokens. The only requirement for p D is that it can draw samples from the marginal distributions of the suffix in parallel. However, if we consider that the implementation of p D is a Transformer-based masked language model, further speedups can be achieved.
this section cite: ['b9', 'b15', 'b0', 'b17', 'b0', 'b20', 'b30']

Section: Recompute KV Window
In our above method, we center sampling parallel as the primary way to improve decoding speed. However, overall we aim to maximize the throughput of diffusion models by all means necessary, so we also must consider architectural factors. We enable KV caching for tokens outside a sliding window of size W . For a detailed explanation of KV caching, see Appendix B. Although KV caching with a diffusion model trained with bidirectional masking can induce arbitrary out-of-distribution behavior, empirically we observe very little performance degradation (Appendix B). The intuitive reason is that tokens sufficiently far from the rightmost token will have small attention weight, so inaccuracies in their KV will not dramatically change the overall attention computation.
this section cite: []

Section: Maximum Masked Lookahead
Because we are sampling from a diffusion model left to right autoregressively, we can exploit the fact that the input will contain a large block of contiguous [MASK] tokens as the suffix. The simple fix is in fewer parallel tokens per iteration but maintain high quality. In particular, we achieve a far better tradeoff compared to naively parallelizing semi-autoregressively. Notably, generating over 5 tokens per iteration on average is possible with APD while maintaining ∼80% accuracy on GSM8K. At the expense of some quality, even over 100 tokens per second is possible.
to set a maximum length M for this suffix. Because the attention computation has O(n 2 ) complexity, decreasing the size of the input can lead to significant speed improvements. Unfortunately, the size of the masked lookahead M can change the output distribution by changing the probability of the end of sentence token [EOS]. Empirically, the maximum masked lookahead M can be tuned to gain significant speedup with minimal quality loss.
this section cite: []

Section: Experiments
In our method, we define three tunable parameters, which we will briefly summarize:
1. Multiplicative Mixture Weight R: Higher results in higher throughput, lower quality 2. Recompute KV Window W : Lower results in higher throughput, lower quality 3. Maximum Masked Lookahead M : Lower results in higher throughput, lower quality Our experimental goal is to empirically analyze the tradeoff in speed and generation quality as these parameters vary.
this section cite: []

Section: Implementation
In our experiments, we use Dream 7B Instruct [41] as the diffusion model p D and Qwen2.5 0.5B [40] as the approximate autoregressive model pAR . Both models have demonstrated impressive capabilities on math, science, and reasoning benchmarks. Conveniently, Dream 7B is a diffusion model that has been distilled from Qwen2.5 7B, so it is more likely to have distributional overlap with Qwen2.5 0.5B. These models satisfy another prerequisite; they share the same tokenizer. Although our method is theoretically applicable to Llada 7B, which exhibits strong left to right generation capabilities, Llada 7B is trained with a non-standard tokenizer that no existing autoregressive model utilizes.
this section cite: ['b40', 'b39']

Section: Experimental Configuration
For the following experiments, we load the models in BF16 precision and run them on single NVIDIA 24GB A5000 GPU connected to a Colfax CX41060s-EK9 4U Rackmount Server with AMD EPYC (Genoa) 9124 processors. We operate using the LM Evaluation Harness [11] standard implementation of benchmarks with a few modifications and evaluate on GSM8K [6], GPQA [33], and MATH [15], and HumanEval [4]. See Appendix C for precise details.
this section cite: ['b10', 'b5', 'b32', 'b14', 'b3']

Section: Tradeoffs
For analyzing tradeoffs, we vary each parameter in isolation to measure the impact of each on performance. In later experiments, we plot configurations of the parameters jointly. Each plot shows the Grade School Math 8K (GSM8K) [6] accuracy with 500 samples, and the throughput measured by number of tokens generated divided by generation time. We measure the standard error of throughput and accuracy with respect to the variation in samples, i.e. different math problem.
Figure 3 shows that as we vary the multiplicative mixture weight R, we achieve a range of speed, quality outcomes. We see for small R, we accept fewer tokens per iteration but maintain high quality. Remarkably, it is possible to generate over 5 tokens an iteration on average and achieve close to the same accuracy as generating 1 token per iteration. As we increase R, the accuracy drops but not drastically. Constrast this with Figure 2, where increasing the number of tokens sampled in parallel leads to a precipitous drop in generation quality. With our method, by dynamically choosing when to sample from the diffusion model in parallel, we are able to achieve much higher token acceptance rates without losing generation quality. In Table 2, we track statistics of APD and find that the parallel acceptance rate is high.
We also analyze the impact of changing the recompute KV window W for left to right autoregressive generation from the diffusion model, sampling with with p AR (the same as K = 1). In Figure 6 (Appendix B), we observe the same relationship between accuracy and throughput, but the tradeoff is not as strong. As we decrease the window W , we can achieve a nontrivial speedup at almost no expense to quality.
Finally, we examine the speed-quality throughput exhibited by altering the maximum masked lookahead M . We again observe that there is no free lunch that can increase speed with zero degradations in quality. In this case, we find that decreasing M increases throughput but can significantly alter quality by virtue of the fact that the generation length decreases. In general, works have shown a relationship between increased "thinking time" (the number of generated tokens) and reasoning strength [27]. Thus, changing the output distribution of p D in this way can damage generation quality, especially for complex reasoning tasks.
All together, the parameters in question each exhibit a unique tradeoff between speed and quality. We believe giving the user or practitioner more flexibility to balance these factors at inference time is an undoubted strength of our approach.
this section cite: ['b5', 'b26']

Section: Pareto Frontier
The Pareto frontier characterizes the set of optimal configurations when faced with the inherent trade-off between two or more conflicting objectives, which in the case of LLM inference is speed and quality. A specific LLM configuration is considered Pareto optimal [25] if it is impossible to improve its performance in one objective without simultaneously incurring a detrimental effect on  another objective, such as increased latency. The Pareto frontier is, therefore, the collection of all such non-dominated configurations, representing the attainable boundary of performance.
In Figure 5, we show this frontier over several tasks and model configurations. Dream 7B (K = 1) is a naive baseline that decodes one token at a time, and it therefore occupies low throughput and high density regions. According to our previous assumptions, it will also upper bound the quality of Dream with other decoding parameters. We observe that Dream 7B with our configuration of parameters given by ADP, achieves much greater speed with minimal performance degradation from the base K = 1 performance. We also find that Dream with ADP is faster than the autoregressive Qwen 7B and even Qwen 0.5B. Note that if Qwen 7B used speculative decoding with a Qwen 0.5B draft model, it will still never exceed the throughput of ADP. Thus, Dream with ADP is Pareto-optimal, because no model can dominate it in speed and quality. Similar to throughput, ADP is also much faster in terms of latency. Diffusion models benefit from the fact that they generate fewer tokens to stay within a fixed context window, while autoregressive models can generate much longer reasoning traces that may ramble and not improve quality.
this section cite: ['b24']

Section: Qualitative Examples
Though we evaluate on standard benchmarks, APD can also quickly generate for more open-ended tasks. In Appendix D, we show examples with average number of parallelized tokens on a dataset of persuasive writing prompts [8].
this section cite: ['b7']

Section: Related Work
Several works propose architectural modifications to LLMs to enable multi-token prediction: Mask-Predict [12], Medusa [3], and DynaMo [37]. While these works are promising, our focus is multitoken prediction in dLLMs. Other recent works seek to improve dLLM speed. For example, block diffusion [2] enables KV caching, but requires training, unlike our method. Discrete copula diffusion [23] is an inference time method that aims to reduces the number of denoising steps required, but does not offer a tunable speed, quality tradeoff. More generally, algorithmic approaches to speeding up LLM inference include cascades [28], lookahead decoding [9], and speculative decoding [28], which are methods that are generally only applicable in autoregressive models. Building on speculative decoding, works have used different architectures as the draft model including any-order autoregressive models [14] and dLLMs [5]. We emphasize that using a large dLLM to draft in parallel presents fundamentally distinct challenges that are not solved with speculative decoding. Finally, we highlight that quantization is another promising and orthogonal approach for offering a strong speed quality tradeoff in LLMs [24,7].
this section cite: ['b11', 'b2', 'b36', 'b1', 'b22', 'b27', 'b8', 'b27', 'b13', 'b4', 'b23', 'b6']

Section: Conclusion
In this work, we pursue the significant challenge of increasing inference speed in large language models. We introduced Adaptive Parallel Decoding (APD), a novel algorithm that enables substantially faster sampling from dLLMs. APD uniquely restructures the dLLM into a left-to-right autoregressive process and leverages a smaller autoregressive model to assess the quality of parallel-generated token candidates. Complemented by optimizations such as KV caching and limited masked inputs, APD offers tunable parameters that allow a flexible trade-off between generation speed and output quality. This research offers a significant advancement in making dLLMs a more viable and efficient alternative for fast text generation.
this section cite: []

Section: References
Ref_id:b0 Title: Parallel sampling via counting Year: (2024)
Ref_id:b1 Title: Block diffusion: Interpolating between autoregressive and diffusion language models Year: (2025)
Ref_id:b2 Title: Simple llm inference acceleration framework with multiple decoding heads Year: (2024)
Ref_id:b3 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b4 Title: Speculative diffusion decoding: Accelerating language generation through diffusion Year: (2024)
Ref_id:b5 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b6 Title: The case for 4-bit precision: k-bit inference scaling laws Year: (2023)
Ref_id:b7 Title: Measuring the persuasiveness of language models Year: (2024)
Ref_id:b8 Title: Break the sequential dependency of llm inference using lookahead decoding Year: (2024)
Ref_id:b9 Title: Janos Galambos. Bonferroni inequalities. The Annals of Probability Year: (1977)
Ref_id:b10 Title: The language model evaluation harness Year: ()
Ref_id:b11 Title: Mask-predict: Parallel decoding of conditional masked language models Year: (2019)
Ref_id:b12 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b13 Title: Reviving any-subset autoregressive models with principled parallel sampling and speculative decoding Year: (2025)
Ref_id:b14 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b15 Title: Products of experts Year: (1999)
Ref_id:b16 Title: Enabling autoregressive models to fill in masked tokens Year: (2025)
Ref_id:b17 Title: Categorical reparameterization with gumbel-softmax Year: (2016)
Ref_id:b18 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b19 Title: Train for the worst, plan for the best: Understanding token ordering in masked diffusions Year: (2025)
Ref_id:b20 Title: Approximation algorithms for classification problems with pairwise relationships: Metric labeling and markov random fields Year: (2002)
Ref_id:b21 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b22 Title:  Year: (2024)
Ref_id:b23 Title: Scaling laws in extremely low-bit llm quantization Year: (2025)
Ref_id:b24 Title: Visualizing the pareto frontier Year: (2008)
Ref_id:b25 Title: Discrete diffusion language modeling by estimating the ratios of the data distribution Year: (2023)
Ref_id:b26 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b27 Title: Faster cascades via speculative decoding Year: (2024)
Ref_id:b28 Title: Large language diffusion models Year: (2025)
Ref_id:b29 Title: Your absorbing discrete diffusion secretly models the conditional distributions of clean data Year: (2024)
Ref_id:b30 Title: Efficiently scaling transformer inference Year: (2022)
Ref_id:b31 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b32 Title: Gpqa: A graduate-level google-proof qa benchmark Year: (2023)
Ref_id:b33 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b34 Title: Scaling llm test-time compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b35 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b36 Title: Dy-naMo: Accelerating language model inference with dynamic multi-token sampling Year: (2024-06)
Ref_id:b37 Title: Semi-autoregressive neural machine translation Year: (2018)
Ref_id:b38 Title: Remasking discrete diffusion models with inference-time scaling Year: (2025)
Ref_id:b39 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b40 Title:  Year: (2025)
Ref_id:b41 Title: Scaling reasoning in diffusion large language models via reinforcement learning Year: (2025)
Ref_id:b42 Title: A survey on efficient inference for large language models Year: (2024)
