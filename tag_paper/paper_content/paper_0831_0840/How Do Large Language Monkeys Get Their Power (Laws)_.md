Title: How Do Large Language Monkeys Get Their Power (Laws)?
Abstract: Recent research across mathematical problem solving, proof assistant programming and multimodal jailbreaking documents a striking finding: when (multimodal) language model tackle a suite of tasks with multiple attempts per task -succeeding if any attempt is correct -then the negative log of the average success rate scales a power law in the number of attempts. In this work, we identify an apparent puzzle: a simple mathematical calculation predicts that on each problem, the failure rate should fall exponentially with the number of attempts. We confirm this prediction empirically, raising a question: from where does aggregate polynomial scaling emerge? We then answer this question by demonstrating per-problem exponential scaling can be made consistent with aggregate polynomial scaling if the distribution of singleattempt success probabilities is heavy tailed such that a small fraction of tasks with extremely low success probabilities collectively warp the aggregate success trend into a power law -even as each problem scales exponentially on its own. We further demonstrate that this distributional perspective explains previously observed deviations from power law scaling, and provides a simple method for forecasting the power law exponent with an order of magnitude lower relative error, or equivalently, ∼2 -4 orders of magnitude less inference compute. Overall, our work contributes to a better understanding of how neural language model performance improves with scaling inference compute and the development of scaling-predictable evaluations of (multimodal) language models.

Section: Introduction
Scaling behaviors of large neural language models have surprised and fascinated engineers, scientists and society alike (Hestness et al., 2017;Kaplan et al., 2020;Brown et al., 2020a;Hoffmann et al., 2022;Ganguli et al., 2022;Sorscher et al., 2022;Wei et al., 2022b;Schaeffer et al., 2023;OpenAI et al., 2024), shaping engineering, economic and governmental interests in frontier AI systems (Bommasani et al., 2021;Eloundou et al., 2023;Anderljung et al., 2023;Wang et al., 2023;Reuel et al., 2024;Besiroglu et al., 2024a;Maslej et al., 2024). For a more thorough exposition of relevant literature, please see Related Work (Section 6).
One direction of renewed interest is inference-time compute scaling, whereby compute is controllably increased at inference to improve the performance of a model, e.g., Pachocki et al. (2024). In this direction, recent research discovered that language model success rates scale predictably with the number of independent attempts made at accomplishing a task. Specifically, in a paper titled, "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling, " Brown et al. (2024) studied how language model performance changes at mathematical problem solving and coding problems when k independent attempts are sampled per problem. Performance on the i-th problem was measured using the expected (over attempts) success rate (Kulal et al., 2019;Chen et al., 2021), defined as:
pass i @k def = E k Attempts I[Any attempt on i-th problem succeeds] .
(1) Using the unbiased and numerically stable estimator of Chen et al. (2021) (for details, see Appendix B), Brown et al. (2024) found that the negative log averaged-over-Pproblems success rate falls as a power law with the number of independent attempts per problem k:
log 1 P P i=1
pass i @k ≈ ak -b ,(2)
for model-specific and benchmark-specific constants a, b > 0 (Fig. 1 Top). Soon after, on a separate topic of jailbreaking multimodal language models via text, image and audio  2024) found the negative log average pass ratelog(pass D @k) at solving mathematical problems scales polynomially (i.e., as a power law) with the number of independent attempts per problem k. Bottom: Hughes et al. (2024) similarly found the negative log average attack success ratelog(ASR D @k) when jailbreaking multimodal language models scales polynomially with the number of jailbreak attempts per prompt. Should such power law scaling be expected? From where do large language monkeys obtain their power (laws)? attacks, independent work by Hughes et al. (2024) studied jailbreaking success rates when k independent attempts are made per harmful prompt. Performance was measured using Attack Success Rate (ASR) at k:
ASR i @k def = E k Attempts I[Any attack on i-th prompt succeeds] .
(3)
This "Best-of-N Jailbreaking" attack similarly discovered that the negative log averaged-over-P -prompts attack success rate fell as a power law with the number of jailbreak attempts per prompt k:
log 1 P P i=1
ASR i @k ≈ ak -b ,(4)
for model-specific and modality-specific constants a, b > 0 (Fig. 1 Bottom). For the specific coefficients from both papers, see Appendix. C. As a minor matter of terminology, both papers frame their results in terms of "coverage" -the fraction of problems that can be solved after k attempts per problem -but as Brown et al. (2024) pointed out, coverage is equivalent to the average success rate (Appendix D); we prefer this latter framing as it avoids the binary implication that each problem either is or is not solved after k attempts.
this section cite: ['b35', 'b44', 'b37', 'b29', 'b83', 'b76', 'b15', 'b26', 'b4', 'b68', 'b53', 'b20', 'b47', 'b20', 'b39', 'b39', 'b20']

Section: Should Power Law Scaling Be Expected?
Should we expect large language monkeys to have such power (laws)? That is, should the negative log of the average success rate scale polynomially with the number of independent attempts k? As we now explain mathematically and demonstrate empirically, such polynomial scaling with k is perhaps surprising because, for any single problem, the negative log success rate at k should fall exponentially with k; the intuition is that pass i @k is 1 unless all attempts fail, and since attempts are independent, the probability that all fail is exponentially unlikely with the number of attempts.
Mathematically, on any given attempt, the model has probability pass i @1 of solving the i-th problem. Recalling that pass i @k is defined as 1 if any of the k attempts succeed, 0 otherwise, by linearity of expectation and by independence of the k attempts, we can rewrite pass i @k as:
pass i @k = E k Attempts 1 -I[All k Attempts Fail] (5
) = 1 - k j=1 E 1 Attempt I[j-th Attempt Fails] . (6
)
The probability that the j-th attempt fails is one minus the probability that the j-th attempt succeeds. Since each attempt is i.i.d. with success probability pass i @1, we find
pass i @k = 1 -(1 -pass i @1) k .(7)
For large k, (1pass i @1) k will be small. Recalling that the Taylor Series expansion of log(1 + x) for small x is
∞ i=1 (-1) i-1 x i /i ≈ x, we have: -log(pass i @k) = -log 1 -(1 -pass@1) k (8) ≈ (1 -pass i @1) k .(9)
Thus, for any single problem, we should expect the negative log expected (over attempts) success rate to fall exponentially with k, not polynomially with k.
To confirm this claim, we plotted the scaling of model performance on each problem -measured either by log(pass i @k) or bylog(ASR i @k) -against the number of independent attempts k. We specifically used Brown et al. (2024)'s data of the Pythia language model family (Biderman et al., 2023) solving 128 mathematical problems from MATH Hendrycks et al. (2021) as well as Hughes et al. (2024)'s data from jailbreaking frontier AI systems -Claude, GPT4 (OpenAI et al., 2024), Gemini (Team et al., 2024a;b) and Llama 3 8B Instruction Tuned (IT) (Grattafiori et al., 2024) -on 159 prompts from HarmBench (Mazeika et al., 2024). For each individual mathematical problem and jailbreaking prompt, we found the negative log expected (over attempts) success rates fall exponentially with k as expected (Fig. 3), including on Llama 3 8B IT which does not exhibit an aggregate power law (Fig. 1).
10 0 10 2 10 4 Num. Attempts per Problem k 10 -1 10 0 10 1 -log(pass D @k) Average Power Law Scaling 10 0 10 2 10 4 Num. Attempts per Problem k 10 -1 10 0 10 1 -log(pass i @k) Per-Problem Exponential Scaling 10 -5 10 -3 10 -1 pass i @1 10 -1 10 0 10 1 p D (pass i @1) Pass@1 Distribution Over Problems = + -log(pass D @k) ∝ k -b p D (pass i @1) ∝ (pass i @1) b-1
this section cite: ['b54']

Section: Distribution of Per-Problem Single-Attempt Success Rates Creates Power Law Scaling
How does polynomial scaling of the negative log average success rate emerge from exponential scaling of the negative log per-problem success rate? The answer to this question must lie in the distribution D over benchmark problems of single attempt (i.e., k = 1) success rates because this distribution's density p D (pass i @1) links the per-problem scaling behavior to the aggregate scaling behavior via the definition of the aggregate success rate pass D @k:
pass D @k def = E passi@1∼D pass i @k(pass i @1) = 1 - 1 0 (1 -pass i @1) k p D (pass i @1) d pass i @1 .(10)
Based on a known result that power laws can originate from an appropriately weighted sum of exponential functions (Appendix E.1), we begin by considering simple distributions for the single-attempt success probabilities and asking which yield power law scaling betweenlog(pass D @k)
and k, as well as what properties of the distributions set the scaling exponent. In Appendices E.3-E.8, we derive that several simple distributions yield power law scaling with different exponents whereas others do not:
log pass Uniform(0, β≤1) @k ∝ k -1 .
log pass Beta(α,β) @k ∝ k -α .
log pass Kumaraswamy(α, β) @k ∝ k -α .
log pass ContinuousBernoulli(λ<1/2) @k ∝ k -1 .
log pass Reciprocal(0<α<β<1) @k ∝ (1α) k k .
To test this understanding, we examined whether the data of Brown et al. (2024) and Hughes et al. (2024) had per-problem single-attempt success rate distributions that matched one of these simple distributions (Fig. 4). We found that the distributions could indeed be well fit by a 3-parameter Kumaraswamy(α, β, a = 0, c) distribution with scale parameter c (Fig. 4, black dashed lines); we found the scale parameter was critical to obtain good fits because the standard 2-parameter Kumaraswamy distribution is supported on (0, 1) whereas most single-attempt success distributions have a smaller maximum such as 0.01 or 0.1.
More generally, what are the distributional properties that create such power law scaling and that set the specific power law exponent? As we now show, the negative log average success rate will exhibit power law scaling in k with exponent b if and only if the distribution over problems of single-attempt success probabilities itself behaves like a power law near 0 with exponent b -1:
Theorem 3.1 (Sufficiency of Power-Law Left Tail in Dis-10 -2 10 -1 10 0 10 1 -log(pass i @k) Model = Pythia 70M Model = Pythia 160M Model = Pythia 410M 10 0 10 1 10 2 10 3 10 4 Num. Attempts per Problem k Model = Pythia 1B 10 0 10 1 10 2 10 3 10 4 Num. Attempts per Problem k 10 -2 10 -1 10 0 10 1 -log(pass i @k) Model = Pythia 2.8B 10 0 10 1 10 2 10 3 10 4 Num. Attempts per Problem k Model = Pythia 6.9B 10 0 10 1 10 2 10 3 10 4 Num. Attempts per Problem k Model = Pythia 12B Large Language Monkeys Model Pythia 70M Pythia 160M Pythia 410M Pythia 1B Pythia 2.8B Pythia 6.9B Pythia 12B 10 -2 10 -1 10 0 10 1 -log(ASR i @k) Model = Claude 3.5 Sonnet Model = Claude 3.5 Opus Model = Gemini 1.5 Flash 10 0 10 1 10 2 10 3 10 4 Num. Attempts per Prompt k Model = Gemini 1.5 Pro 10 0 10 1 10 2 10 3 10 4 Num. Attempts per Prompt k 10 -2 10 -1 10 0 10 1 -log(ASR i @k) Model = GPT4o Mini 10 0 10 1 10 2 10 3 10 4 Num. Attempts per Prompt k Model = GPT4o 10 0 10 1 10 2 10 3 10 4 Num. Attempts per Prompt k Model = Llama 3 8B IT Best-of-N Jailbreaking Model Claude 3.5 Sonnet Claude 3.5 Opus Gemini 1.5 Flash Gemini 1.5 Pro GPT4o Mini GPT4o Llama 3 8B IT
Figure 3: Per-problem performance scales exponentially with the number of attempts per problem k. Top: Pythia language models on 128 problems from MATH, with performance on the i-th problem measured aslog(pass i @k). Bottom: Frontier AI models on jailbreaking prompts from HarmBench, with performance on the i-th problem measured as log(ASR i @k). In both settings, on each problem, the negative log per-problem success rate falls exponentially with the number of independent attempts k. However, the negative log average success rate falls as a power law with k (black). tribution of Single-Attempt Success Rates). Let D be a probability distribution on [0, 1] with PDF p D (pass i @1). Suppose there exist constants b > 0, C > 0, θ > 0 and δ > 0 such that, for all 0 < pass i @1 < δ, we have
p D (pass i @1) = C•(pass i @1) b-1 + O (pass i @1) b-1+θ .
Then, for large k,
-log pass D @k ∼ C Γ(b) k -b .
Theorem 3.2 (Necessity of Power-Law Left Tail in Distribution of Single-Attempt Success Rates). Let D be a distribution over pass i @1 ∈ [0, 1] with PDF p D (pass i @1). Suppose there exist constants b > 0 and A > 0 such that for large k,
-log pass D @k ∼ A k -b .
Then, under mild regularity assumptions, the probability density must satisfy
p D (pass i @1) ∼ A Γ(b) (pass i @1) b-1 as pass i @1 → 0 + .
In Fig. 2, we illustrate this connection schematically. For proofs, see Appendices E.8 and E.9. These results clarify that wheneverlog(pass D @k) exhibits power-law decay in k with exponent b, the distribution over problems of single-attempt success rates must have "polynomial weight" near pass i @1 = 0, i.e. p D (p) = Θ(p b-1 ).
To offer intuition, we know that each problem is being solved by the model (or equivalently, each prompt is jailbreaking the model) exponentially quickly. If one looks across all problems in the benchmark, some have pass i @1 so small that they remain unsolved for many, many attempts. Whether these "tiny-pass i @1" problems still matter at large k depends on how many such problems there are. Polyno-mial density near 0 "piles up" enough hard problems in just the right way such that even though each of those problems is being solved exponentially quickly, the aggregate success rate over problems decreases at only a power-law rate in k. A more succinct mathematical summary is that, for a compound binomial distribution, the lower tail probability controls the upper tail of the marginal survivor function. 4. Lack of Distributional Structure Explains Deviations from Power Law Scaling Notably, previous papers observed that not every model exhibits power law scaling in every setting. To highlight one, Hughes et al. (2024) observed that when jailbreaking Meta's Llama 3 8B Instruction Tuned (IT) model (Grattafiori et al., 2024), thelog(ASR D @k) fell faster than any power law (Fig. 1), i.e., the ASR D @k rose much more quickly than the other frontier AI systems. Based on our mathematical insights and the empirical per-problem single-attempt attack success rates (Fig. 4), we can understand why: Llama 3 8B IT could be successfully jailbroken on every prompt within the permitted sampling budget and thus had no heavy left tail necessary to create the aggregate power law scaling.
this section cite: ['b20', 'b39']

Section: A New Distributional Estimator for Predicting Power Law Scaling
A natural consequence of this connection between the scaling oflog(pass D @k) and the left tail of the distribution p D (pass i @1) is that the distribution of single-attempt success rates can be used to predict whether power-law scaling will appear and if so, what the intercept and exponent of the power law will be. To do this, one can fit the distribution pD (pass i @1) and then simulate how pass D @k will scale with k (Fig. 5) using the relationship:
pass D @k def = 1 - 1 0 (1 -pass i @1) k pD (pass i @1) d pass i @1 .(11)
To empirically test this claim, we compared the standard least squares regression estimator (in log-log space) (Hoffmann et al., 2022; Caballero et al., 2022; Besiroglu et al., 2024b) against a distributional estimator. To motivate our distributional estimator, we first need explain a key obstacle and how the distributional estimator overcomes it. The obstacle is that there are problems or prompts whose single-attempt success probabilities pass i @1 lie between (0, 1/Number of Samples) such that, due to finite sampling, we lack the resolution to measure. While we do not know the true single-attempt success probability for the problems that lie in this interval, we do know how many problems fall into this left tail bucket, and we can fit a distribution's parameters such that the distribution's probability mass in the interval (0, 1/Number of Samples) matches the empirical fraction of problems in this tail bucket. Thus, our distributional estimator works by first selecting a distribution (e.g., a scaled 3-parameter Beta distribution), discretizing the distribution 0.0 0.2 0.4 0.6 Power Law Exponent (Kumaraswamy-Binomial) 0.0 0.1 0.2 0.3 0.4 0.5 0.6 Power Law Exponent (Least Squares) Large Language Monkeys Model Pythia 70M Pythia 160M Pythia 410M Pythia 1B Pythia 2.8B Pythia 6.9B Pythia 12B Benchmark MATH 0.0 0.2 0.4 0.6 Power Law Exponent (Kumaraswamy-Binomial) 0.0 0.1 0.2 0.3 0.4 0.5 0.6 Power Law Exponent (Least Squares) Best-of-N Jailbreaking Model Claude 3.5 Sonnet Claude 3.5 Opus Gemini 1.5 Flash Gemini 1.5 Pro GPT4o Mini GPT4o Llama 3 8B IT Modality Text (2) the distributional estimator of pass i @1 assuming a scaled Kumaraswamy-Binomial distribution. Using all available data to fit both estimators, we find agreement between the least-squares estimate (ordinate) and the distribution-derived estimate (abscissa) for both Pythia models on MATH (left) and for frontier AI systems on HarmBench (right). For an explanation of why the two estimators match more closely for Large Language Monkeys than for Best-of-N Jailbreaking, see Appendix A. according to the sampling resolution 1/Number of Samples and performing maximum likelihood estimation under the discretized distribution's probability mass function.
We tested this distributional estimator in two different ways. First, focusing on Large Language Monkeys, we used all available real data from all problems and all samples per problem to compare the standard least squares regression estimator against the distributional estimator. We found close agreement between the two estimators (Fig. 6), giving us a sense that the two estimators yield reasonably consistent estimates under large sampling budgets.
Second, the distributional estimator also comes with another benefit: it directly provides an estimate of the power law's exponent b in a k -b . Estimating the power law's exponent is especially valuable because the exponent dictates how success rates are improving with increasing inference compute. To test how the distributional estimator and least squares estimator compare at recovering the true asymptotic power law exponent, we generated synthetic data so that we would have ground-truth knowledge of the true power law exponent, then backtested how the two scaling estimators compare at recovering the true exponent (Alabdulmohsin et al., 2022a;Owen, 2024) by subsampling data with fewer problems and fewer samples per problem. We found that the distributional estimator obtains significantly better sample efficiency, with approximately an order of magnitude lower relative error def = | b -b|/b compared with the least squares estimator (Fig. 7), or equivalently, ∼2 -4 orders of magnitude less inference-compute. The distributional estimator performs well even under distributional mismatch.
this section cite: []

Section: Related Work
Research into scaling laws of deep neural networks has a rich history spanning theoretical foundations, empirical validations, and diverse applications. The earliest investigations discovered power law scaling in simple machine learning settings (Barkai et al., 1993;Mhaskar, 1996;Pinkus, 1999). However, the modern era of scaling laws began with breakthrough studies in neural language models (Hestness et al., 2017;Kaplan et al., 2020;Brown et al., 2020b), catalyzing extensive research across multiple directions. The theoretical understanding of scaling laws has advanced significantly (Spigler et al., 2020;Bousquet et al., 2020;Hutter, 2021;Sharma & Kaplan, 2022;Maloney et al., 2022;Roberts et al., 2022;Bahri et al., 2024;Michaud et al., 2024;Paquette et al., 2024;Atanasov et al., 2024;Bordelon et al., 2024a;b;Lin et al., 2024;Brill, 2024), complemented by comprehensive empirical studies (Rosenfeld et al., 2020;Henighan et al., 2020;Gordon et al., 2021;Tay et al., 2021;Ghorbani et al., 2021;Tay et al., 2022b;Zhai et al., 2022;Alabdulmohsin et al., 2022b;Dehghani et al., 2023;Bachmann et al., 2023). In the context of language models, researchers have explored scaling behaviors in various aspects: context length (Xiong et al., 2023), in-context learning (Chan et al., 2022;Agarwal et al., 2024;Arora et al., 2024), vocabulary size (Tao et al., 2024), and jailbreaking attempts (Anil et al., 2024;Hughes et al., 2024). Studies have also investigated scaling dynamics in fine-tuning (Kalajdzievski, 2024;Zhang et al., 2024), transfer learning (Hernandez et al., 2021), and the impact of repeated data (Hernandez et al., 2022;Muennighoff et al., 2023). Architectural considerations have been extensively studied, including network design (Tay et al., 2022a;Clark et al., 2022), nested models (Kudugunta et al., 2023), pruning strategies (Rosenfeld et al., 2021), and precision requirements (Dettmers & Zettlemoyer, 2023;Kumar et al., 2024;Sun et al., 2025). Research has also addressed multimodal extensions (Aghajanyan et al., 2023;Cherti et al., 2023) and inference optimization (Sardana et al., 2023;Brown et al., 2024;Snell et al., 2024a;Wu et al., 2024;Chen et al., 2024). The field has expanded to encompass diverse domains including reinforcement learning (both single-agent (Jones, 2021;Hilton et al., 2023;Neumann & Gros, 2024) and multi-agent (Neumann & Gros, 2022)), graph networks (Liu et al., 2024), diffusion models (Mei et al., 2024;Liang et al., 2024), and associative memory models (Romani et al., 2013;Cabannes et al., 2024;Schaeffer et al., 2024c). Recent work has explored emerging phenomena such as inverse scaling (McKenzie et al., 2024), unique functional forms (Caballero et al., 2022), scaling patterns across model families (Ruan et al., 2024;Polo et al., 2024), and downstream capabilities (Srivastava et al., 2023;Wei et al., 2022a;Hu et al., 2024;Schaeffer et al., 2024b;Snell et al., 2024b;Wu & Lo, 2024). Researchers have also investigated critical challenges including data contamination (Schaeffer, 2023;Jiang et al., 2024;Dominguez-Olmedo et al., 2024), model-data feedback loops (Dohmatob et al., 2024;Gerstgrasser et al., 2024;Kazdan et al., 2024), and overtraining effects (Gao et al., 2023;Gadre et al., 2024). Additional contributions include studies in sparse autoencoders (Gao et al., 2024), biologically-plausible backpropagation (Filipovich et al., 2022), and self-supervised learning for vision (Schaeffer et al., 2024a). Recent efforts have also focused on reconciling apparent contradictions in scaling behaviors (Besiroglu et al., 2024b;Porian et al., 2024).
this section cite: ['b10', 'b57', 'b65', 'b35', 'b44', 'b84', 'b18', 'b40', 'b80', 'b52', 'b69', 'b9', 'b58', 'b64', 'b7', 'b50', 'b19', 'b71', 'b32', 'b88', 'b97', 'b91', 'b8', 'b96', 'b0', 'b6', 'b87', 'b5', 'b39', 'b43', 'b98', 'b33', 'b34', 'b60', 'b37', 'b46', 'b72', 'b22', 'b48', 'b86', 'b1', 'b74', 'b20', 'b42', 'b36', 'b62', 'b61', 'b51', 'b56', 'b49', 'b70', 'b55', 'b73', 'b66', 'b85', 'b38', 'b94', 'b75', 'b41', 'b24', 'b23', 'b45', 'b45', 'b91', 'b28', 'b30', 'b27', 'b67']

Section: Discussion and Future Directions
This work advances our mathematical understanding of how and why language model performance improves with additional inference compute through repeat sampling. By establishing rigorous theoretical foundations for these empirically-observed power laws, our work provides practitioners with principled ways to understand and predict model performance when scaling inference compute. The distributional perspective we develop explains previously puzzling deviations from power law scaling and enables more efficient estimation of scaling parameters.
Two related questions are why such distributional structure exists in the single-attempt success rates and whether one should expect such structure to appear in future benchmarks. We conjecture there are at least two reasons: (1) benchmark design, in that benchmarks are intentionally crafted that problems have a spread of difficulty without being too easy or too hard, and (2) selection bias, in that more interesting patterns such as power law scaling are more likely to garner more interest from the research community.
Despite focusing on scaling inference compute, our paper contributes a new hypothesis for an open question in scaling pretraining compute: why are neural scaling laws power laws? Just as the scaling behavior oflog(pass D @k) only becomes clear for large k, so too might the scaling behavior of pretraining cross entropy with pretraining compute C. Specifically, suppose the pretraining cross entropy L as a function of pretraining compute C is a sum of many functions which decay at different rates:
L(C) = ω 1 C α + A C α + o 1 C α ,
where α is the smallest (positive) polynomial exponent and ω(1/C α ) represents functions that decay more slowly than any polynomial. Initially, for small C, the dominant term may be unclear, but as pretraining compute is scaled up across 8 -10 orders of magnitude, the leading order term dominates and an approximate power law emerges:
L(C) ≈ const + A C α + 0 as C → ∞.
Thus, a power law relationship may only be reasonable for sufficiently large pretraining compute C, which in turn may require excluding the lowest pretraining compute models in order to obtain good predictions, justifying a widespread empirical practice (Kaplan et al., 2020). We designate possible functions hiding in ω(1/C α ) and o(1/C α ) as the dark matter of neural scaling laws.
this section cite: ['b44']

Section: Impact Statement
Our findings have important practical implications for the deployment of large language models, as they can help organizations more accurately forecast compute requirements and make informed trade-offs between model size, inference costs, and performance targets. The mathematical framework we develop could also generalize beyond language models to other domains where similar scaling phenomena emerge. While our work is primarily theoretical, we acknowledge that advances in language model capabilities can have broad societal impacts. We hope that better understanding these fundamental scaling behaviors will help the research community develop more efficient and reliable AI systems.
this section cite: []

Section: C. Fitting Power Laws to Large Language Monkeys and Best-of-N Jailbreaking
We fit power laws to a subset of data from Large Language Monkeys (Brown et al., 2024) and from Best-of-N Jailbreaking (Hughes et al., 2024), specifically Pythia language models (Biderman et al., 2023) on the MATH benchmark (Hendrycks et al., 2021) and frontier AI models -Claude, GPT4 (OpenAI et al., 2024), Gemini (Team et al., 2024a;b) and Llama 3 (Grattafiori et al., 2024) -on the HarmBench jailbreaking benchmark (Mazeika et al., 2024). We show the functional forms and the fit parameters in Table 1 and Table 2 respectively. To fit the parameters, for Large Language Monkeys, we simply minimized the squared error between the actual and predictedlog(pass D @k), and for Best-of-N Jailbreaking, we similarly minimized the squared error between the actual and predictedlog(ASR D @k)).
Note: Llama 3 8B IT does not exhibit power law scaling under Best-of-N Jailbreaking (shown in Fig. 1, bottom). Model Benchmark a b Pythia 70M MATH 8.026 0.194 Pythia 160M MATH 6.591 0.280 Pythia 410M MATH 5.524 0.286 Pythia 1B MATH 5.452 0.315 Pythia 2.8B MATH 4.104 0.336 Pythia 6.9B MATH 4.255 0.348 Pythia 12B MATH 4.113 0.370 Table 1: Large Language Monkeys (Brown et al., 2024) fitted power law parameters on 128 mathematical problems from MATH (Hendrycks et al., 2021). Functional Form:log(pass D @k) = a k -b .
this section cite: ['b20', 'b39', 'b13', 'b54', 'b20']

Section: Model Modality a b
Claude 3.
5 Opus Text 2.630 0.448 Claude 3.5 Sonnet Text 3.436 0.312 GPT4o Text 3.639 0.395 GPT4o Mini Text 3.637 0.492 Gemini 1.5 Flash Text 6.158 0.303 Gemini 1.5 Pro Text 6.296 0.256 Llama 3 8B IT Text --Table 2: Best-of-N Jailbreaking (Hughes et al., 2024) fitted power law parameters on text jailbreak prompts from Harm-Bench (Mazeika et al., 2024). Functional Form:log(ASR D @k) = a k -b . Note: Llama 3 8B Instruction Tuned (IT) does not exhibit power law scaling.
If p ∼ Uniform(α, β), the expectation of (1p) k is:
E (1 -p) k = 1 β -α β α (1 -p) k dp.
Evaluating the integral gives:
E (1 -p) k = (1 -α) k+1 -(1 -β) k+1 (β -α) • (k + 1) .
Thus, the aggregate success rate becomes:
pass Uniform(α,β) @k = 1 - (1 -α) k+1 -(1 -β) k+1 (β -α) • (k + 1) .
Case A: α > 0 If α > 0, then both (1α) and (1β) are strictly less than 1. As k → ∞, (1α) k+1 and (1β) k+1 decay exponentially. Hence:
E (1 -p) k ∼ (1 -α) k+1 (β -α) • (k + 1)
, and pass Uniform(α,β) @k approaches 1 exponentially fast:
pass Uniform(α,β) @k ∼ 1 - (1 -α) k+1 (β -α) • (k + 1)
.
Thus, the negative log of the aggregate success rate decays exponentially: k) .
-log pass Uniform(α,β) @k ∼ e -Ω(
Case B: α = 0 When α = 0, the uniform distribution is over [0, β]. In this case:
E (1 -p) k = 1 β • 1 -(1 -β) k+1 k + 1 .
For large k, (1β) k+1 becomes exponentially small, and:
E (1 -p) k ∼ 1 β • 1 k + 1 .
The aggregate success rate is then:
pass Uniform(0,β) @k ∼ 1 - 1 β • k .
The negative log exhibits power-law scaling:
-log pass Uniform(0,β) @k ∼ 1 β • 1 k . Special Case: Uniform(0, 1) If β = 1, the distribution is uniform on [0, 1]. In this case: E (1 -p) k = 1 k + 1
, and the success rate becomes:
pass Uniform(0,1) @k = 1 - 1 k + 1
.
For large k:
log pass Uniform(0,1) @k ∼ 1 k .
How Do Large Language Monkeys Get Their Power (Laws)?
Step 1: Split the integral into two parts. Fix a constant c > 0. Write
I k = c/k 0 [• • • ] dx + 1 c/k [• • • ] dx def = I k,left + I k,right ,
where [• • • ] indicates the same integrand. In the region x ∈ [c/k, 1], we have (1x) k ≤ e -k x ≤ e -c . Hence I k,right = O e -c . Since c can be made arbitrarily large, I k,right becomes negligible compared to any polynomial in 1/k.
Step 2: Approximate the integrand in the small-x region. On [0, c/k], we use the approximation log(1
-x) = -x + O(x 2 ). Thus (1 -x) k = exp k log(1 -x) = exp -k x + O(k x 2 ) . Since x ≤ c/k implies k x 2 ≤ c 2 /k = O(1/k),
and exp(ϵ) = 1 + O(ϵ), we get
(1 -x) k = exp(-k x) exp(O(1/k)) = exp(-k x) 1 + O 1 k . Furthermore, since (1 -y) m = 1 -my + O(y 2 ), for small x (1 -x α ) β-1 = 1 -(β -1)x α + O(x 2α ) = 1 + O x α .
In the region x ≤ c/k, that error is O k -α . Hence, within the small-x region, the integrand
(1 -x) k α β x α-1 1 -x α β-1
can be approximated by
α β x α-1 e -k x + O k -α x α-1 e -k x .
Thus
I k,left = c/k 0 α β x α-1 e -k x dx + O k -α c/k 0 x α-1 e -k x dx + O e -c . Step 3: Substitution u def = k x.
To handle c/k 0
x α-1 e -k x dx, we substitute u = k x. Then x = u/k, dx = du/k, and the upper limit x = c/k becomes u = c. Hence,
c/k 0 x α-1 e -k x dx = c 0 u k α-1 e -u du k = k -α c 0 u α-1 e -u du.
As c → ∞, c 0 u α-1 e -u du → Γ(α), and for finite c the remainder is O e -c . Therefore,
1 0 x α-1 e -k x dx = k -α Γ(α) + O k -α e -c ,
and absorbing the constant c into big-O notation gives
1 0 x α-1 e -k x dx = k -α Γ(α) + O k -α-ϵ for some ϵ > 0.
Multiplying by the factor α β, we deduce that
I k = α β Γ(α) k -α + O k -α-ϵ .
Step 4: Final conclusion for the success rate. Recall pass Kumaraswamy(α,β) @k = 1 -I k . Hence pass Kumaraswamy(α,β) @k = 1α β Γ(α) k -α + O k -α-ϵ .
Since this tends to 1, its negative log is governed by the magnitude of α β Γ(α) k -α . Using the expansionlog(1y) = y + O(y 2 ) as y → 0, we get
-log pass Kumaraswamy(α,β) @k = α β Γ(α) k -α + o k -α .
That is precisely polynomial (power-law) decay in the negative log success rate with exponent α.
E.6. Continuous Bernoulli Distribution:
pass i @1 ∼ ContinousBernoulli(λ)
Next, suppose the model's pass i @1 probabilities follow a Continuous Bernoulli distribution. The probability density function of this distribution over the support x ∈ [0, 1] is:
f (x; λ) def = C(λ)λ x (1 -λ) 1-x (35) C(λ) def = 2 if λ = 1/2 2 tanh -1 (1-2λ) 1-2λ otherwise . (36
)
The density can equivalently be rewritten in a more convenient form for our purposes:
f (x; λ) = C(λ)λ x (1 -λ)(1 -λ) -x = C(λ)(1 -λ) λ 1 -λ x (37)
Because the individual success probability is low in our data, we shall consider the small λ < 1/2 regime. We follow the same approach as with the Kumaraswamy distribution.
Step 1: Write the aggregate pass rate. The aggregate pass rate is defined as: pass ContinuousBernoulli(λ) @k = 1 -I k , where
I k def = 1 0 (1 -p) k f (p; λ) dp.
Substituting the density f (p; λ), we get:
I k = 1 0 (1 -p) k C(λ) λ p (1 -λ) 1-p dp.
Step 2: Simplify using an exponential form. Using the exponential rewriting:
λ p (1 -λ) 1-p = (1 -λ) exp p log λ 1-λ
, the integral becomes:
I k = C(λ) (1 -λ) 1 0 (1 -p) k exp p log λ 1-λ dp.
Step 3: Dominance of the small-p region. For large k, (1p) k decays exponentially unless p is close to 0. Thus, the main contribution to the integral arises from the region p ∈ [0, c/k], where c > 0 is a constant. Decompose the integral:
I k = c/k 0 [• • • ] dp + 1 c/k [• • • ] dp def = I k,left + I k,right .
In the region p ∈ [c/k, 1], we have (1p) k ≤ e -kp ≤ e -c , making I k,right = O(e -c ), which is negligible compared to 1/k. Thus, we focus on I k,left :
I k,left = C(λ) (1 -λ) c/k 0 (1 -p) k exp p log λ 1-λ dp.
Step 4: Approximate the integrand. For p ∈ [0, c/k], use the same approximations from the Kumaraswamy derivation:
(1p) k = e -kp 1 + O(p) , exp p log λ 1-λ = 1 + O(p).
Thus, the integrand becomes:
(1p) k exp p log λ 1-λ = e -kp 1 + O(p) .
this section cite: ['b39', 'b54']

Section: Step 5: Change of variables.
Let u def = kp, so p = u/k and dp = du/k. The integral becomes:
I k,left = C(λ) (1 -λ) c 0 e -u 1 + O(u/k) du k .
Split the integral:
I k,left = C(λ) (1 -λ) k c 0 e -u du + O 1 k 2 .
As c → ∞, c 0 e -u du → 1. Thus:
I k,left = C(λ) (1 -λ) k + O 1 k 2 .
Since I k,right = O(e -c ) is negligible, we have:
I k = C(λ) (1 -λ) k + O 1 k 2 .
Step 7: Final conclusion for the success rate. Recall:
pass ContinuousBernoulli(λ) @k = 1 -I k .
For large k, this implies:
pass ContinuousBernoulli(λ) @k = 1 - C(λ) (1 -λ) k + O 1 k 2 .
Using the expansionlog(1y) = y + O(y 2 ) for small y, we find:
-log pass ContinuousBernoulli(λ) @k = C(λ) (1 -λ)k -1 + o(k -1 ).
That is precisely polynomial (power-law) decay in the negative log success rate with exponent -1.
As a side comment, recall that tanh -1 (x) = 1 2 log 1+x 1-x , the normalizing constant C(λ) can be rewritten as:
C(λ) = 2 1 -2λ 1 2 log 1 + (1 -2λ) 1 -(1 -2λ) = 1 1 -2λ log 1 -λ λ .(38)
Thus, for small λ, note that C(λ) ≈ log(1/λ) =log(λ). For k ≪log(λ), the 1/k formula is valid. However, near k ≈log(λ), the leading termlog(λ)/k becomes of order 1, and for k ≫log(λ), the success rate is now very close to 1. Consequently, we see that if λ is very small, there is a soft cutoff scale around k ≈log(λ).
E.7. Any Continuous Distribution with p(pass i @1) = c > 0 Suppose that the distribution over pass i @1 is continuous and has constant non-zero density near 0:
f (0) = c > 0(39)
Because the density is continuous at 0 with f (0) = c > 0, there exist some δ > 0 such that:
f (p) = c + O(p) for all p ∈ [0, δ].(40)
Because the small pass i @1 region dominates for large k, a similar argument to the Kumaraswamy argument and Continuous Bernoulli argument yields power law scaling with respect to k with exponent -1:
-log pass D @k = c k -1 + o(k -1 ).(41)
This result is consistent with the Continuous Bernoulli, where c is given by f ContinuousBernoulli(λ) (0; λ) = C(λ)(1λ) for λ < 1/2. This result reveals that the Continous Bernoulli is just one instance of a larger family: any continuous distribution with non-zero constant density at pass i @1 = 0 will exhibit power law scaling with exponent -1.
E.8. Reciprocal Distribution: pass i @1 ∼ Reciprocal(a, b)
Next, suppose the model's pass i @1 ∼ Reciprocal(a, b) distribution with 0 < a < b < 1. The probability density function of this distribution over the support x ∈ [a, b] is:
f (x; a, b) = 1 (log(b) -log(a)) x(42)
As with the other distributions, the aggregate success rate after k attempts is:
pass Reciprocal(a,b) @k = E pass i @k = 1 -I k , where I k def = b x=a (1 -x) k 1 (log b -log a) x dx.
We aim to show that I k is on the order of (1-a) k k .
The main contribution to the integral arises from the vicinity of x = a, because (1x) k decays rapidly as x grows away from a.
this section cite: []

Section: Step 1: Change of variable. Define y
def = x -a, so the domain x ∈ [a, b] becomes y ∈ [0, b -a]. Then (1 -x) k = (1 -a) -y k ,and
I k = 1 log(b/a) b-a y=0 (1 -a) -y k 1 a + y dy.
Step 2: Expansion near y = 0. For small y, write (1a)y = (1a) 1 -y 1-a ; hence log (1a)y = log(1a) + log 1 -y 1-a .
Using log(1z) = -z + O(z 2 ) for small z, we get
log (1 -a) -y = log(1 -a) - y 1 -a + O y 2 (1-a) 2 , so (1 -a -y) k = exp k log(1 -a) -k y 1-a + O k y 2 (1-a) 2 .
In particular, for y up to c/k, the term k y 2 = O(1) remains bounded, so
(1 -a -y) k = (1 -a) k exp -k y 1-a 1 + O 1 k .
Step 3: The integral is dominated by y ∈ [0, O( 1 k )]. For large k, exp -k y 1-a decays quickly once y exceeds a multiple of 1-a k . Consequently, the integral from y = c 0 /k to ba is exponentially small in k. On [0, c 0 /k], we also have (a + y) -1 = 1 a + O 1 k . Thus
I k = 1 log(b/a) c0/k y=0 (1 -a -y) k 1
a + y dy + (exponentially small tail).
this section cite: []

Section: Substitute our approximation from Step 2 into the integrand:
(
1 -a -y) k 1 a + y = (1 -a) k exp -k y 1-a 1 a + O 1 k .
Step 4: Change variable u = k y 1-a . Then y = (1-a) u k and dy = 1-a k du. The upper limit y = c 0 /k corresponds to
u = c 0 1-a 1 , so c0/k y=0 exp -k y 1-a dy = c0 (1-a) u=0 e -u 1 -a k du.
Letting c 0 → ∞ only contributes an e -c0 (1-a) factor to the tail, which vanishes. Hence
∞ y=0 exp -k y 1-a dy = 1 -a k ∞ u=0 e -u du = 1 -a k .
Putting all factors together,
I k = 1 log(b/a) (1 -a) k 1 a + O 1 k 1 -a k + (exponentially small in k).
Thus in big-Theta form,
I k = Θ (1-a) k k .
Conclusion. Since pass Reciprocal(a,b) @k = 1 -I k , we get
pass Reciprocal(a,b) @k = 1 -Θ (1-a) k k .
Moreover, usinglog(1y) = y + O(y 2 ) for small y, it follows that log pass Reciprocal(a,b) @k = Θ (1-a) k k .
Hence the negative log aggregate success rate converges to 1 exponentially fast in k, which is not a power law in k.
this section cite: []

Section: Sufficient Condition for Power-Law Scaling in Negative Log of Aggregate Success
Theorem E.1. Let D be a probability distribution on [0, 1] with PDF f (p). Suppose there exist constants b > 0, C > 0, θ > 0 and δ > 0 such that, for all 0 < p < δ, we have f (p) = C p b-1 + O p b-1+θ .
Then, for large k,
1 -pass D @k = C Γ(b) k -b + O k -b-min( 1,θ) , which implies -log pass D @k = C Γ(b) k -b + o k -b .
Equivalently, including the leading constant),
-log pass D @k ∼ C Γ(b) k -b .
this section cite: []

Section: Proof. Step 1. Decompose the key integral.
Define
I k def = 1 -pass D @k = 1 0 (1 -p) k f (p) dp.
For a positive constant c > 0, split I k :
I k = c/k 0 (1 -p) k f (p) dp + 1 c/k (1 -p) k f (p) dp def = I k,left + I k,right .
Right Tail Bound (I k,right ). For p ≥ c/k, observe (1p) k ≤ e -k p ≤ e -c . Hence
I k,right = 1 c/k (1 -p) k f (p) dp ≤ e -c 1 0 f (p) dp = e -c .
Since c can be made arbitrarily large, e -c can be driven below any power of 1/k. Thus I k,right = o k -α for any α > 0.
We may therefore focus on
I k,left = c/k 0 (1 -p) k f (p) dp,
knowing that I k,right is negligible in polynomial-type estimates.
Step 2. Use the assumed behavior of f (p) near p = 0. By hypothesis, for p up to some δ > 0, f (p) = C p b-1 + O p b-1+θ .
Choose c/k < δ, so p ≤ c/k < δ for p in the left integral. Then
I k,left = c/k 0 (1 -p) k C p b-1 + O p b-1+θ dp.
Split it into main term and error term:
I k,left = C c/k 0 (1 -p) k p b-1 dp + c/k 0 (1 -p) k O p b-1+θ dp.
Denote these T main and T err , respectively.
this section cite: []

Section: Step 3. Approximate (1p) k by e -kp and control the error.
For p in [0, c/k], expand log(1p) = -p + O(p 2 ). Thus (1p) k = exp k log(1p) = e -k p exp O(k p 2 ) = e -k p 1 + O(k p 2 ) .
this section cite: []

Section: Step 5. Error term T err .
Recall
T err = c/k 0 (1 -p) k O p b-1+θ dp.
Exactly the same substitution (1p) k = e -kp + O(k p 2 e -k p ) plus u = k p shows
T err = O c/k 0 p b-1+θ e -k p dp + O c/k 0 k p b+1+θ e -k p dp .
When substituting u = k p, the exponent on p increases by +1 each time if we multiply by k, so each term is of order k -b-θ or smaller. Concretely,
c/k 0 p b-1+θ e -k p dp = k -b-θ c 0 u b-1+θ e -u du = O k -b-θ ,
and similarly for the second term, which is even smaller. Hence
T err = O k -b-θ .
Step 6. Putting it all together. Summarize: 1,θ) .
I k,left = T main + T err = C Γ(b) k -b + O k -b-1 + O k -b-θ . Thus I k,left = C Γ(b) k -b + O k -b-min(
Recalling the tail piece I k,right = e -c = o k -α for any α, we obtain
I k = I k,left + I k,right = C Γ(b) k -b + O k -b-min(1,θ) . Hence 1 -pass D @k = I k ∼ C Γ(b) k -b .
Final negative-log argument. Since
pass D @k = 1 -I k = 1 -C Γ(b) k -b + O k -b-min(1,θ) ,
for large k it is very close to 1. Then
-log pass D @k = -log 1 -C Γ(b) k -b + • • • .
Using the expansionlog(1x) = x + O(x 2 ) as x → 0, and here x = C Γ(b) k -b , we get
-log pass D @k = C Γ(b) k -b + o k -b .
In the "∼" notation including the leading coefficient:
-log pass D @k ∼ C Γ(b) k -b .
This completes the proof.
this section cite: []

Section: 
understanding across millions of tokens of context, 2024b. URL https://arxiv.org/abs/2403.05530.
this section cite: []

Section: 
, Prasad, K., Upasani, K., Plawiak, K., Li, K., Heafield, K., Stone, K., El-Arini, K., Iyer, K., Malik, K., Chiu, K., Bhalla, K., Lakhotia, K., Rantala-Yeary, LPapakipos, Z., Singh, A., Srivastava, A., Jain, A., Kelsey,  A., Shajnfeld, A., Gangidi, A., Victoria, A., Goldstand,  A., Menon, A., Sharma, A., Boesenberg, A., Baevski, A.,  Feinstein, A., Kallet, A., Sangani, A., Teo, A., Yunus, A.,  Lupu, A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poulton, A., Ryan, A., Ramchandani, A., Dong, A., Franco,  A., Goyal, A., Saraf, A., Chowdhury, A., Gabriel, A.,  Bharambe, A., Eisenman, A., Yazdan, A., James, B.Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid,  A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A.,  Garriga-Alonso, A., Kluska, A., Lewkowycz, A., Agarwal, A., Power, A., Ray, A., Warstadt, A., Kocurek, A. W.,  Safaya, A., Tazarv, A., Xiang, A., Parrish, A., Nie, A.,  Hussain, A., Askell, A., Dsouza, A., Slone, A., Rahane,  A., Iyer, A. S., Andreassen, A., Madotto, A., Santilli, A.,  Stuhlmüller, A., Dai, A., La, A., Lampinen, A., Zou, A.,  Jiang, A., Chen, A., Vuong, A., Gupta, A., Gottardi, A.,  Norelli, A., Venkatesh, A., Gholamidavoodi, A., Tabassum, A., Menezes, A., Kirubarajan, A., Mullokandov, A.,  Sabharwal, A., Herrick, A., Efrat, A., Erdem, A., Karakaş,  A
this section cite: []

Section: A. Clarification of How Large Language Monkeys and Best-of-N Jailbreaking Sampled Data
In this manuscript, we used the phrasing of "independent attempts," which is not fully correct. In this appendix section, we clarify why we chose this terminology, what likely impacts we believe this inaccuracy may have had on our results, and how to correct the paper accordingly.
Large Language Monkeys (Brown et al., 2024) indeed drew 10, 000 independent attempts per problem, but Best-of-N Jailbreaking (Hughes et al., 2024) sampled data slightly different: for each problem, jailbreaking attempts were drawn until either a successful jailbreak was obtained or until a maximum limit of 10, 000 attempts was hit. Samples were also drawn in minibatches of size 60, making the (in)dependence of samples a bit tricky.
We omitted this nuance because it offers a second-order correction to our paper's main story while offering little additional insight. Neither of our theorems and none of our main text figures change. We suspect that this slightly different sampling procedure explains why, in Fig. 6, the estimated power law exponents between the least squares power law estimator and the distributional power law estimator deviate more significantly from identity for Best-of-N Jailbreaking than for Large Language Monkeys. A natural way to correct for this is to use a beta-negative binomial distribution rather than a beta-binomial distribution, with an additional correction for the maximum number of attempts. For more information, please see Appendix H.
this section cite: ['b20', 'b39']

Section: B. Estimating Success Rates Using Chen et al. (2021)'s Estimator
In this manuscript, we defined pass i @k and ASR i @k as:
pass i @k def = E k Attempts I[
At least 1 attempt by the model solves the i-th problem] ASR i @k def = E k Attempts I[At least 1 attempt jailbreaks the model on the i-th prompt]
Throughout this manuscript, to estimate pass i @k and ASR@k, we used the unbiased and lower variance estimator introduced by Chen et al. (2021): for the i-th problem, we sampled n ≫ k attempts per problem, counted the number of successful attempts c, and then swept k to compute an estimate of pass i @k for different k values:
pass i @k = 1 - n-c k n k (12
)
Two comments: Firstly, n as used here has no relationship with the number of problems in the benchmark (Sec. 1), and secondly, our notation differs slightly from that of Chen et al. ( 2021), but the ideas are consistent. A numerically stable Python implementation of the estimator is provided in Fig. 8:
def estimate_success_rate_at_k_per_problem(n: int, c: int, k: int) -> float: """ :param n: number of total attempts on this problem. :param c: number of correct attempts on this problem. :param k: k in pass_i@$k$.
""" if n -c < k: return 1.0 return 1.0 -np.prod(1.0 -k / np.arange(n -c + 1, n + 1)) To reiterate a point made by Chen et al. (2021), estimating pass i @k as 1 -(1pass i @1) k is biased (Fig. 9). D. Mathematical Equivalence Between Coverage and Average Success Rate Brown et al. (2024) and Hughes et al. (2024) phrase their research in terms of "coverage", defined as the fraction of problems that can be solved or the fraction of prompts that can jailbreak a model, but as Brown et al. (2024) comment and we here derive, the coverage is mathematically equivalent to the average pass i @k (equivalently, ASR@k. due to two simple probabilistic primitives: (1) linearity of expectation, (2) the expectation of an indictor random variable of some event is the probability of said event and (3) the definition of pass i @k:
10 2 10 4 n 10 -2 10 -1 pass@k True pass i @1: 0.001 10 2 10 4 n 10 -1 10 0 pass@k True pass i @1: 0.01 10 2 10 4 n 10 0 3 × 10 -1 4 × 10 -1 6 × 10 -1 pass@k True pass i @1: 0.1 k 3.0 10.0 32.0 100.0 316.0 1000.0 Estimator 1 -(1 -p) k 1 -n-c k / n k
E Prompts Attempts Coverage def = E Problems Attempts Fraction of Problems Solved After k Attempts = E Problems E Attempts|Problem I Problem Solved After k Attempts = E Problems pass problem @k = pass D @k
In our work, we prefer phrasing along the lines of "success rate" over "coverage" because success rate avoids coverage's binary implication that each problem/prompt is either "solved" or "not solved".
this section cite: ['b20', 'b39', 'b20']

Section: E. Aggregate Power Laws from a Probability Distribution over Exponential Functions E.1. Preliminaries: Power Laws from Weighted Exponential Functions
A known result is that power laws can emerge from appropriately weighted sums of exponential functions, e.g., (Bochud & Challet, 2006;Elkies, 2016;Bousquet et al., 2020). For a concrete example with a short proof:
x -r = 1 Γ(r) ∞ 0 p r-1 e -px dp,(13)
where Γ(r)
def = ∞ 0 s r-1 e -s ds is the Gamma function. The proof is via u-substitution u def = p x: 1 Γ(r) ∞ 0 p r-1 e -px dp = 1 Γ(r) ∞ 0 (u/x) r-1 e -u du x (14
) = 1 Γ(r) x -r ∞ 0 u r-1 e -u du (15
) = 1 Γ(r) x -r Γ(r) (16
) = x -r(17)
In our particular context, we are interested in the scaling with k of the expected success rate over problems sampled from the benchmark's data distribution:
pass D @k def = E passi@1∼D pass i @k(18)
distribution (over problems in a benchmark) of pass i @k scores that yields power law scaling with respect to the number of attempts k:
-log 1 n n i=1 pass i @k ≈ ak -b .(19)
for constants a, b > 0.
this section cite: ['b14', 'b25', 'b18']

Section: E.2. Delta Distribution: pass
i @1 ∼ δ(p), p ∈ (0, 1)
To start with a negative result, we will show that not all distributions of the per-problem success probabilities pass i @1 yield aggregate power law scaling. Suppose that the model's pass i @1 probabilities across the benchmarks' problems are all exactly p ∈ (0, 1). For brevity, let p i def = pass i @1. Then the aggregate success rate is:
E pi∼δ(p) [pass i @k] = 1 -E pi [(1 -p i ) k ](20)
= 1 0 δ(p) (1 -p i ) k dp i (21) = (1 -p) k .(22)
Recalling that the expansion of log(•) for small x islog(1x) = x + O(x 2 ), in our case, we obtain:
-log 1 -E pi∼δ(p) [pass@k] = (1 -p) k + O((1 -p) 2k ) = (1 -p) k + o((1 -p) k ).(23)
Thus, in the large k regime, we find the negative log aggregate success rate exhibits exponential scaling with k as we intuitively expect.
this section cite: []

Section: E.3. Uniform Distribution: pass
i @1 ∼ Uniform(α, β)
Suppose pass i @1 probabilities follow a uniform distribution Uniform(α, β) where 0 ≤ α < β ≤ 1. The aggregate success rate after k attempts is defined as:
pass Uniform(α,β) @k def = 1 -E (1 -p) k . E.4. 2-Parameter Beta Distribution: pass i @1 ∼ Beta(α, β)
Suppose that the model's pass i @1 probabilities across the benchmark problems follow a Beta distribution:
pass i @1 ∼ Beta(α, β)
The probability density function of this distribution over the support x ∈ (0, 1) is:
f (x; α, β) def = 1 B(α, β) x α-1 (1 -x) β-1 ,(24)
where α > 0, β > 0 and B(•, •) is the Beta function. For brevity, let p i def = pass i @1. Under our assumed Beta distribution:
pass Beta(α,β) @k def = 1 -E pi∼Beta(α,β) [(1 -p i ) k ] (25
) = 1 - 1 0 p α-1 i (1 -p i ) β-1 B(α, β) (1 -p i ) k dp i (26) = 1 - Γ(α + β) Γ(α)Γ(β) Γ(α)Γ(β + k) Γ(α + β + k)(27)
where Γ(•) is again the Gamma function. The Γ(α) terms cancel, and a standard asymptotic result of the gamma function for large k tells us that:
Γ(β + k) Γ(α + β + k) ∼ k -α ,(28)
and thus:
Γ(α + β) Γ(β) Γ(β + k) Γ(α + β + k) ∼ Γ(α + β) Γ(β) k -α .(29)
Recalling again that the expansion of log(•) for small x islog(1x) = x + O(x 2 ), in our case, we obtain:
-log pass D @k = Γ(α + β) Γ(β) k -α + O(k -2α ) = Γ(α + β) Γ(β) k -α + o(k -α ).(30)
From this final result, we see that under a Beta distribution and in the large k regime, the negative log aggregate success rate exhibits polynomial (power-law) scaling with k for exponent α E.5. Kumaraswamy Distribution: pass i @1 ∼ Kumaraswamy(α, β)
Next, suppose the model's pass i @1 probabilities follow a Kumaraswamy distribution. The probability density function of this distribution over the support x ∈ (0, 1) is:
f (x; α, β) def = α β x α-1 (1 -x α ) β-1(31)
Again for brevity, let p i def = pass i @1. Under our assumed Kumaraswamy distribution:
pass Kumaraswamy(α,β) @k def = 1 -E pi∼Kumaraswamy(α,β) [(1 -p i ) k ] (32
) = 1 - 1 0 (1 -p) k • α β p α-1 (1 -p α ) β-1 dp. (33
)
Define the integral
I k def = E (1 -p) k = 1 0 (1 -x) k α β x α-1 1 -x α β-1 dx.(34)
We aim to analyze I k for large k. Notice that (1x) k is exponentially small in k unless x is very close to 0. Thus, intuitively, most of the contribution to I k arises from x ∈ [0, O(1/k)].
Since p ≤ c/k, we get k p 2 ≤ c 2 /k, which is bounded for large k. Consequently, (1p) k = e -k p + O k p 2 e -k p .
We will use this in both T main and T err .
Step 4. Main term T main .
T main = C c/k 0 (1 -p) k p b-1 dp. Substituting (1 -p) k = e -k p + O k p 2 e -k p , T main = C c/k 0 e -k p p b-1 dp + C c/k 0 O k p b+1 e -k p dp.
Call these two integrals T 1 and T 2 .
T 1 term.
T 1 = C c/k 0 p b-1 e -k p dp.
Make the substitution u def = k p. Then p = u/k, dp = du/k, and p b-1 = k -b+1 u b-1 . The upper limit p = c/k becomes u = c. Thus
T 1 = C c 0 u k b-1 e -u du k = C k -b c 0 u b-1 e -u du. As c → ∞, c 0 u b-1 e -u du → Γ(b). So T 1 = C k -b Γ(b) -R c , where |R c | = O e -c .
By choosing c large after k → ∞, we conclude
T 1 = C Γ(b) k -b + o k -b .
T 2 term.
T 2 = C c/k 0 O k p b+1 e -k p dp.
Inside the integral, k p b+1 e -k p is the main factor. Substituting u def = k p again,
p b+1 = u k b+1 = k -b-1 u b+1 . Hence T 2 = C O(1) c/k 0 k p b+1 e -k p dp = O(k) c/k 0 p b+1 e -k p dp.
Substitute u = k p and dp = du/k. Then
T 2 = O(k) c 0 u k b+1 e -u du k = O(k) k -b-2 c 0 u b+1 e -u du = O k -b-1 .
Thus T 2 is of strictly smaller order than k -b .
Combine T 1 and T 2 :
T main = C Γ(b) k -b + O k -b-1 .
this section cite: []

Section: E.9. Necessary Condition for Power Law Scaling from Distribution over pass i @1
Theorem E.2. Let D be a probability distribution over [0, 1] with a PDF f (p) satisfying the following regularity near p = 0:
• No point mass at p = 0. So 1 0 f (p) dp = 1, and f is a genuine PDF on (0, 1].
• Continuity and nonnegative behavior near p = 0. There exist δ > 0 such that f is continuous on [0, δ] and has no pathological oscillations or singularities that violate integrability.
Define the aggregate success rate at k attempts:
pass D @k def = 1 0 1 -(1 -p) k f (p) dp
and relatedly
I k def = 1 0 (1 -p) k f (p) dp = 1 -pass D @k .
Assume that there exist constants A > 0 and b > 0 such that for large k:
-log pass D @k ∼ A k -b Then I k = A k -b + o k -b ,
and under the mild regularity assumptions above,
f (p) ∼ A Γ(b) p b-1 as p → 0 + . Proof. Step 1. Relating I k to -log(pass D @k).
By definition,
pass D @k = 1 -I k , I k = 1 0 (1 -p) k f (p) dp. Since -log pass D @k ∼ A k -b , we have, for large k, pass D @k = exp -A k -b (1 + o(1)) .
When x is small, exp(-x) = 1x + O(x 2 ). Thus
I k = 1 -pass D @k = A k -b + o k -b . So I k ∼ A k -b .
Step 2. Restricting to a small interval near p = 0.
Since (1p) k decays exponentially once p is on the order of 1/k or larger, we split: (a) Ratio to e -kp . For p ∈ 0, c k , define the ratio
I k def = 1 0 (1 -p) k f (p) dp = c/k 0 (1 -p) k f (p) dp + 1 c/k (1 -p) k f (p) dp
R k (p) def = (1 -p) k e -k p .
We will show that R k (p) stays close to 1 uniformly in p ∈ [0, c/k] for large k. Indeed,
(1 -p) k = exp k log(1 -p) , log(1 -p) = -p - p 2 2 - p 3 3 -. . . . Hence log(1 -p) + p = - p 2 2 - p 3 3 -. . . = O p 2 as p → 0.
Multiplying by k, we get k log(1p) + p = O k p 2 .
Since 0 ≤ p ≤ c k implies k p 2 ≤ c 2 k , which → 0 as k → ∞, it follows that k log(1 -p) = -k p + O 1 k .
Exponentiating:
(1 -p) k = e -k p exp O 1 k = e -k p 1 + O 1 k . Thus R k (p) = (1 -p) k e -k p = 1 + O 1 k , with the O( 1 k ) bound uniform for all p ∈ [0, c/k]. In other words, there is some constant M > 0 (independent of k) such that R k (p) -1 ≤ M k for all p ∈ 0, c k . (b) Integral expression using R k (p). Hence on [0, c/k], (1 -p) k f (p) = e -k p R k (p) f (p).
Thus
I k,left = c/k 0 e -k p f (p) R k (p) dp. Define ∆ k (p) def = R k (p) -1, which satisfies |∆ k (p)| ≤ M/k. Then I k,left = c/k 0 e -k p f (p) dp + c/k 0 e -k p f (p) ∆ k (p) dp.(43)
Step 4. Substitution u = k p and deriving f (p) ∼ p b-1 .
(a) The leading part. Focus on the first term of equation 43:
c/k 0 e -k p f (p) dp. Substitute u def = k p, so p = u k and dp = 1 k du. The upper limit p = c k becomes u = c. Thus c/k 0 e -k p f (p) dp = c 0 e -u f u k du k . Hence c/k 0 e -k p f (p) dp = 1 k c 0 e -u f u k du. (b) The error part. The second term in equation 43 has ∆ k (p) = R k (p) -1 satisfying |∆ k (p)| ≤ M k . So c/k 0 e -k p f (p) ∆ k (p) dp ≤ M k c/k 0 e -k p f (p) dp.
But the integral
c/k 0 e -k p f (p)
dp is precisely the leading part we just considered. Thus the error is bounded by M k times a term that will turn out to be Θ(k -b ). Hence the error is subleading if b < 1 is not the case-but even then, we can keep track of it systematically.
Overall, combining both terms, we get
I k,left = 1 k c 0 e -u f u k du + O 1 k • (leading integral) .(44)
(c) Matching Θ(k -b ). Since I k = I k,left + I k,right with I k,right negligible, we have
I k = 1 k c 0 e -u f u k du + (small corrections). But by hypothesis, I k ∼ α k -b . Thus k • I k = c 0 e -u f u k du + (smaller terms) ∼ α k 1-b .(45)
Hence the expression
c 0 e -u f u k du must be Θ k 1-b for large k. Since u k is small for 0 ≤ u ≤ c, we are effectively sampling f near 0. For the integral to produce k 1-b , we deduce f u k = Θ u k b-1 , i.e. f must behave like p b-1 near p = 0. Rewriting the constant in front, one obtains f u k = u k b-1 some positive constant .
(We then identify that constant with α Γ(b) by matching the integral precisely, just as in the prior argument.)
Step 5. Conclusion. We have thus shown that over p ∈ [0, c/k], one has
(1 -p) k = e -k p 1 + O( 1 k ) ,
and upon integrating, the required k -b form for I k forces
f (p) = A Γ(b) p b-1 + o p b-1 , as p → 0 + .
this section cite: []

Section: This completes the necessity proof.
Remark (Mild Regularity). If f had bizarre oscillations or nonintegrable singularities near 0, the integral 1 0 (1-p) k f (p) dp might not produce a clean k -b . Typically, we impose monotonicity or at least continuity near p = 0, no atom at p = 0, and f (0) = 0 if b > 1 or f (0) > 0 if b = 1, etc. These assumptions exclude pathological behaviors and guarantee that the local shape of f (p) drives a clean power law.
this section cite: []

Section: F. Maximum Likelihood Estimation of Scaled Beta-Binomial Distribution
To model the distribution of pass i @1, we can perform maximum likelihood estimation on a scaled three-parameter Beta-Binomial distribution, which we chose because each attempt on the i-th problem is an i.i.d. Bernoulli random variable with success probability pass i @1, and we introduced a scale parameter because the largest pass i @1 values were typically 1-2 orders of magnitude less than 1.0 (the maximum of the unscaled beta distribution's support).
In greater detail, as background, the 4-parameter Beta distribution has PDF p Y (y; α, β, a, c)
def = (y -a) α-1 (c -y) β-1 (c -a) α+β-1 B(α, β) ,(46)
where B(•, •) is the Beta function. If the minimum a is fixed at 0 and the maximum c is constrained to a < c < 1, then the scaled three parameter Beta distribution simplifies to:
f P (p; α, β, a = 0, c) = p α-1 (c -p) β-1 c α+β-1 B(α, β) . (47
)
We want the PMF of a three-parameter Beta-Binomial distribution based on this scaled Beta distribution. For n samples and x successes, the PMF is:
P (X = x; α, β, c, n) def = c 0 n x p x (1 -p) n-x f P (p; α, β, a = 0, c) dp (48
) = n x 1 c α+β-1 B(α, β) c 0 p x+α-1 (1 -p) n-x (c -p) β-1 dp.(49)
Using a change of variable p def = c z, the PMF can be rewritten as
P (X = x; α, β, c, n) = n x c x B(α, β) 1 0 z x+α-1 (1 -z) β-1 (1 -cz) n-x dz (50
) = n x c x B x + α, β B(α, β) 2 F 1 -(n -x), x + α; x + α + β; c ,(51)
where 2 F 1 (•, •; •; •) is the (Gauss) hypergeometric function.
this section cite: []

Section: G. Maximum Likelihood Estimation of Scaled Kumaraswamy-Binomial Distribution
To model the distribution of pass i @1, we can perform maximum likelihood estimation on a scaled three-parameter Kumaraswamy-Binomial distribution, which we chose because each attempt on the i-th problem is an i.i.d. Kumaraswamy random variable with success probability pass i @1, and we introduced a scale parameter because the largest pass i @1 values were typically 1-2 orders of magnitude less than 1.0 (the maximum of the unscaled beta distribution's support).
In greater detail, the scaled three parameter Kumaraswamy distribution simplifies to:
f P (p; α, β, a = 0, c) = αβ c α p α-1 (1 -(p/c) α ) β-1 ,(52)
over the support (0, c). The rescaled Kumaraswamy-Binomial distribution then has PMF:
P (X = x; α, β, c, n) = n x α β c α c 0 p x+α-1 (1 -p) n-x 1 -p c α β-1 dp.(53)
One can perform a change of variable p def = cz, but simplifying yields sums of hypergeometric functions that add little conceptual clarity and so we resort to numerical integration using Python's mpmath library (mpmath development team, 2023).
this section cite: []

Section: H. Maximum Likelihood Estimation of Scaled Beta-Negative Binomial Distribution
To model the distribution of pass i @1, we can perform maximum likelihood estimation on a scaled three-parameter Beta-Negative Binomial distribution. Recall that the scaled three parameter Beta distribution is:
f P (p; α, β, a = 0, c) = p α-1 (c -p) β-1 c α+β-1 B(α, β) .(54)
We want the PMF of a three-parameter Beta-Negative Binomial distribution based on this scaled Beta distribution. For r desired successes, the PMF that we first draw x failures is:
P (X = x; α, β, c, r) = c 0 x + r -1 x p r (1 -p) x NegBin(r,p) p α-1 (c -p) β-1 c α+β-1 B(α, β) scaled Beta PDF dp (55
) = x + r -1 x 1 c α+β-1 B(α, β) c 0 p r+α-1 1 -p x c -p β-1 dp.(56)
Next, substitute p = c z =⇒ dp = c dz which rescales the domain [0, c] to [0, 1]. Under this change:
p r+α-1 = (c z) r+α-1 = c r+α-1 z r+α-1 , (c -p) β-1 = c -c z β-1 = c(1 -z) β-1 = c β-1 (1 -z) β-1 , (1 -p) x = 1 -c z x .
Putting these into the integrand:
p r+α-1 1 -p x c -p β-1 dp = c r+α-1 z r+α-1 1 -cz x c β-1 (1 -z) β-1 c dz .
Factor out the constants in c:
= c r+α-1 c β-1 c z r+α-1 (1 -cz) x (1 -z) β-1 dz. Since c r+α-1 • c β-1 • c = c r+α+β-1 , we get p r+α-1 (1 -p) x (c -p) β-1 dp = c r+α+β-1 z r+α-1 (1 -z) β-1 (1 -cz) x dz.
Plugging back into P (X = x; α, β, c, r) and simplifying:
P (X = x; α, β, c, r) = x + r -1 x c r B(α, β) 1 0 z r+α-1 (1 -z) β-1 1 -c z x dz.(57)
We can re-express this using the (Gauss) hypergeometric function 2 F 1 (•, •; •; •):
P (X = x; α, β, c, r) = x + r -1 x c r B r + α, β B α, β 2 F 1 -x, r + α; r + α + β; c .(58)
this section cite: []

Section: References
Ref_id:b0 Title: Many-shot in-context learning Year: (2024)
Ref_id:b1 Title: Scaling laws for generative mixed-modal language models Year: (2023)
Ref_id:b2 Title: Revisiting neural scaling laws in language and vision Year: (2022)
Ref_id:b3 Title: Revisiting neural scaling laws in language and vision Year: (2022)
Ref_id:b4 Title: Frontier ai regulation: Managing emerging risks to public safety Year: (2023)
Ref_id:b5 Title: Many-shot jailbreaking Year: (2024)
Ref_id:b6 Title: Bayesian scaling laws for in-context learning Year: (2024)
Ref_id:b7 Title: Scaling and renormalization in high-dimensional regression Year: (2024)
Ref_id:b8 Title: Scaling mlps: A tale of inductive bias Year: (2023)
Ref_id:b9 Title: Explaining neural scaling laws Year: (2024)
Ref_id:b10 Title: Scaling laws in learning of classification tasks Year: (1993)
Ref_id:b11 Title: Economic impacts of ai-augmented r&d Year: (2024)
Ref_id:b12 Title: Chinchilla scaling: A replication attempt Year: (2024)
Ref_id:b13 Title: Pythia: A suite for analyzing large language models across training and scaling Year: (2023)
Ref_id:b14 Title: Optimal approximations of power-laws with exponentials Year: (2006)
Ref_id:b15 Title: On the opportunities and risks of foundation models Year: (2021)
Ref_id:b16 Title: A dynamical model of neural scaling laws Year: (2024)
Ref_id:b17 Title: How feature learning can improve neural scaling laws Year: (2024)
Ref_id:b18 Title: A theory of universal learning Year: (2020)
Ref_id:b19 Title: Neural scaling laws rooted in the data distribution Year: (2024)
Ref_id:b20 Title: Large language monkeys: Scaling inference compute with repeated sampling Year: (2024)
Ref_id:b21 Title: Language models are few-shot learners Year: (2020)
Ref_id:b22 Title: The case for 4-bit precision: k-bit inference scaling laws Year: (2023)
Ref_id:b23 Title: A tale of tails: Model collapse as a change of scaling laws Year: (2024)
Ref_id:b24 Title: Training on the test task confounds evaluation and emergence Year: (2024)
Ref_id:b25 Title: Is there a way to express an power law decay as a series of exponentials? Year: (2016)
Ref_id:b26 Title: Gpts are gpts: An early look at the labor market impact potential of large language models Year: (2023)
Ref_id:b27 Title: Scaling laws beyond backpropagation Year: (2022)
Ref_id:b28 Title: Language models scale reliably with over-training and on downstream tasks Year: (2024)
Ref_id:b29 Title: Predictability and surprise in large generative models Year: (2022)
Ref_id:b30 Title:  Year: (2024)
Ref_id:b31 Title: Measuring mathematical problem solving with the math dataset Year: ()
Ref_id:b32 Title: Scaling laws for autoregressive generative modeling Year: (2020)
Ref_id:b33 Title: Scaling laws for transfer Year: (2021)
Ref_id:b34 Title: Scaling laws and interpretability of learning from repeated data Year: (2022)
Ref_id:b35 Title: Deep learning scaling is predictable, empirically Year: (2017)
Ref_id:b36 Title: Scaling laws for single-agent reinforcement learning Year: (2023)
Ref_id:b37 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b38 Title: Predicting emergent abilities with infinite resolution evaluation Year: (2024)
Ref_id:b39 Title: Best-of-n jailbreaking Year: (2024)
Ref_id:b40 Title: Learning curve theory Year: (2021)
Ref_id:b41 Title: Investigating data contamination for pre-training language models Year: (2024)
Ref_id:b42 Title: Scaling scaling laws with board games Year: (2021)
Ref_id:b43 Title: Scaling laws for forgetting when finetuning large language models Year: (2024)
Ref_id:b44 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b45 Title: Collapse or thrive? perils and promises of synthetic data in a selfgenerating world Year: (2024)
Ref_id:b46 Title: Nested transformer for elastic inference Year: (2023)
Ref_id:b47 Title: Search-based pseudocode to code Year: (2019)
Ref_id:b48 Title: Scaling laws for precision Year: (2024)
Ref_id:b49 Title: Scaling laws for diffusion transformers Year: (2024)
Ref_id:b50 Title: Scaling laws in linear regression: Compute, parameters, and data Year: (2024)
Ref_id:b51 Title: Towards neural scaling laws on graphs Year: (2024)
Ref_id:b52 Title: A solvable model of neural scaling laws Year: (2022)
Ref_id:b53 Title: Artificial intelligence index report 2024 Year: (2024)
Ref_id:b54 Title: A standardized evaluation framework for automated red teaming and robust refusal Year: (2024)
Ref_id:b55 Title: Inverse scaling: When bigger isn't better Year: (2024)
Ref_id:b56 Title: Bigger is not always better: Scaling properties of latent diffusion models Year: (2024)
Ref_id:b57 Title: Neural networks for optimal approximation of smooth and analytic functions Year: (1996)
Ref_id:b58 Title: The quantization model of neural scaling Year: (2024)
Ref_id:b59 Title: mpmath: a Python library for arbitrary-precision floating-point arithmetic Year: ()
Ref_id:b60 Title: Scaling data-constrained language models Year: (2023)
Ref_id:b61 Title: Scaling laws for a multiagent reinforcement learning model Year: (2022)
Ref_id:b62 Title: Alphazero neural scaling and zipf's law: a tale of board games and power laws Year: (2024)
Ref_id:b63 Title: Learning to reason with LLMs Year: ()
Ref_id:b64 Title: + 3 phases of compute-optimal neural scaling laws Year: (2024)
Ref_id:b65 Title: Approximation theory of the mlp model in neural networks Year: (1999)
Ref_id:b66 Title: Sloth: scaling laws for llm skills to predict multi-benchmark performance across families Year: (2024)
Ref_id:b67 Title: Resolving discrepancies in compute-optimal scaling of language models Year: (2024)
Ref_id:b68 Title: Open problems in technical ai governance Year: (2024)
Ref_id:b69 Title: The principles of deep learning theory Year: (2022)
Ref_id:b70 Title: Scaling laws of associative memory retrieval Year: (2013)
Ref_id:b71 Title: A constructive prediction of the generalization error across scales Year: (2020)
Ref_id:b72 Title: On the predictability of pruning across scales Year: (2021-07)
Ref_id:b73 Title: Observational scaling laws and the predictability of language model performance Year: (2024)
Ref_id:b74 Title: Beyond chinchilla-optimal: Accounting for inference in language model scaling laws Year: (2023)
Ref_id:b75 Title: Pretraining on the test set is all you need Year: (2023)
Ref_id:b76 Title: Are emergent abilities of large language models a mirage? Year: (2023)
Ref_id:b77 Title: Towards an improved understanding and utilization of maximum manifold capacity representations Year: (2024)
Ref_id:b78 Title: Why has predicting downstream capabilities of frontier ai models with scale remained elusive? Year: (2024)
Ref_id:b79 Title: Bridging associative memory and probabilistic modeling Year: (2024)
Ref_id:b80 Title: Scaling laws from the data manifold dimension Year: (2022)
Ref_id:b81 Title: Scaling llm testtime compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b82 Title: Predicting emergent capabilities by finetuning Year: (2024)
Ref_id:b83 Title: Beyond neural scaling laws: beating power law scaling via data pruning Year: (2022)
Ref_id:b84 Title: Asymptotic learning curves of kernel methods: empirical data versus teacher-student paradigm Year: (2020)
Ref_id:b85 Title:  Year: (2023)
Ref_id:b86 Title: Scaling laws for floating point quantization training Year: (2025)
Ref_id:b87 Title: Scaling laws with vocabulary: Larger models deserve larger vocabularies Year: (2024)
Ref_id:b88 Title: Scale efficiently: Insights from pretraining and fine-tuning transformers Year: (2021)
Ref_id:b89 Title: Scaling laws vs model architectures: How does inductive bias influence scaling Year: (2022)
Ref_id:b90 Title: Transcending scaling laws with 0.1 URL Year: ()
Ref_id:b91 Title: Scientific discovery in the age of artificial intelligence Year: (2023)
Ref_id:b92 Title: Emergent abilities of large language models, 2022a Year: ()
Ref_id:b93 Title: Emergent abilities of large language models Year: (2022)
Ref_id:b94 Title: U-shaped and inverted-u scaling behind emergent abilities of large language models Year: (2024)
Ref_id:b95 Title: Inference scaling laws: An empirical analysis of computeoptimal inference for problem-solving with language models Year: (2024)
Ref_id:b96 Title: Effective long-context scaling of foundation models Year: (2023)
Ref_id:b97 Title: Scaling vision transformers Year: (2022)
Ref_id:b98 Title: When scaling meets llm finetuning: The effect of data, model and finetuning method Year: (2024)
