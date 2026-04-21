Title: OVERCOMING JOINT INTRACTABILITY WITH LOSSLESS HIERARCHICAL SPECULATIVE DECODING
Abstract: Verification is a key bottleneck in improving inference speed while maintaining distribution fidelity in Speculative Decoding. Recent work has shown that sequence-level verification leads to a higher number of accepted tokens compared to token-wise verification. However, existing solutions often rely on surrogate approximations or are constrained by partial information, struggling with joint intractability. In this work, we propose Hierarchical Speculative Decoding (HSD), a provably lossless verification method that significantly boosts the expected number of accepted tokens and overcomes joint intractability by balancing excess and deficient probability mass across accessible branches. Our extensive large-scale experiments demonstrate that HSD yields consistent improvements in acceptance rates across diverse model families and benchmarks. Moreover, its strong explainability and generality make it readily integrable into a wide range of speculative decoding frameworks. Notably, integrating HSD into EAGLE-3 yields over a 12% performance gain, establishing state-of-the-art decoding efficiency without compromising distribution fidelity. Code is available at https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding.

Section: INTRODUCTION
Inference speed has become paramount for Large Language Models (LLMs) (Achiam et al., 2023;Touvron et al., 2023;Bai et al., 2023), which generate text auto-regressively. Recent advances in test-time scaling (OpenAI, 2024;Guo et al., 2025;Yu et al., 2025;Peng et al., 2025) have further underscored its importance. While techniques like pruning (Frankle and Carbin, 2018;Sun et al., 2023a) and quantization (Shen et al., 2020;Xiao et al., 2023) improve efficiency but sacrifice performance, Speculative Decoding (Leviathan et al., 2023) achieves speedups while preserving the target model's distribution, making it a particularly appealing alternative. It adopts a smaller model to make proposals and a larger model to select from them with a grounded verification strategy. Most approaches prioritize the drafting phase, but further gains face diminishing returns. Driven by the verification bottleneck, recent methods (Cai et al., 2024;Zhou et al., 2024;Narasimhan et al., 2024) trade off fidelity for speed, relying on task-specific tuning; their performance typically remains constrained to carefully curated scenarios.
Recent work (Sun et al., 2024;Qin et al., 2025) shows that jointly verifying draft tokens can improve the expected number of accepted tokens, but faces joint intractability: simply applying the resampling strategy used in tokenwise verification (Leviathan et al., 2023) would require full joint probabilities over all possible decoding paths to correctly recover the target distribution, which is computationally infeasible. To address this, (Qin et al., 2025) employs a lossy fixed acceptance threshold, while (Sun et al., 2024) proposes Blockwise Verification, which provably recovers the target distribution. However, Blockwise Verification still falls short of the ideal case, and both its underlying mechanism and compatibility with other methods remain unclear.
In this work, we propose Hierarchical Speculative Decoding (HSD), a provably lossless verification method built upon a novel hierarchical branch resampling strategy. In speculative decoding, resam- pling recovers portions of the target distribution that exceed the draft probability. As illustrated in Figure 1, HSD organizes multiple resampling distributions hierarchically across successive levels, with each distribution recovering only the partial target within its branch and resampling occurring immediately after the last accepted token. This design ensures the full target distribution is recovered in expectation while maximizing the expected number of accepted tokens, pushing the limits of lossless verification and enabling more efficient decoding. Notably, Blockwise verification focuses on independent verification with unclear potential for integration, while our method is designed to easily combine with other approaches, such as the widely adopted multi-draft setups.
In summary, our contributions are as follows:
• We introduce Hierarchical Speculative Decoding (HSD), a lossless and explainable verification method that integrates seamlessly with existing speculative decoding frameworks while remaining largely orthogonal to them.
• HSD delivers a practical advance in inference scaling, achieving an average 6.7% improvement in decoding speed across diverse benchmarks and model sizes while preserving distributional fidelity, with efficiency gains of up to 12.3% on individual datasets.
• HSD further improves decoding speed across multi-draft settings. Notably, integrating HSD into EAGLE-3 yields over 12% performance gain, establishing new state-of-the-art decoding efficiency without compromising distribution fidelity.
this section cite: ['b0', 'b26', 'b1', 'b18', 'b9', 'b31', 'b19', 'b7', 'b22', 'b28', 'b13', 'b2', 'b33', 'b17', 'b25', 'b20', 'b13', 'b20', 'b25']

Section: RELATED WORK
Follow-up research on speculative decoding (Leviathan et al., 2023) can be organized into two main phases: the drafting phase and the verification phase.
this section cite: ['b13']

Section: Drafting Phase.
Drafting methods can be grouped into three categories: (1) Single-draft. Early SD methods (Leviathan et al., 2023) inspired PaSS (Monea et al., 2023) and Draft&Verify (Zhang et al., 2024), improving efficiency via multi-token generation or selective layer skipping. GLIDE (Du et al., 2024) (shared KV-cache) offers further speedups but requires task-specific tuning.
(2) Retrieval-based. LLM-A (Yang et al., 2023) and ReST (He et al., 2023) generate drafts from reference texts, potentially reducing latency, but face database limitations, distribution gaps, and reliance on greedy decoding.
(3) Multi-draft. Tree-attention frameworks-SpecInfer (Miao et al., 2024), Medusa (Cai et al., 2024), and Eagle (Li et al., 2024;Fan et al., 2026)-expand many branches, quickly exhausting memory. Medusa and Eagle also predict drafts from the target model's hidden features rather than a separate draft model, further boosting speed but requires task-specific tuning.
Verification Phase. Verification methods trade fidelity for speed. Lossless approaches (Sun et al., 2023b;Yang et al., 2024;Hu et al., 2025) guarantee exact recovery but are costly. Block Verification (Sun et al., 2024) partially alleviates this bottleneck but offers limited improvement and low interpretability and integrity. Lossy methods-including BiLD (Kim et al., 2023), MTAD (Qin et al., 2025), DistillSpec (Zhou et al., 2024), Medusa-2 (Cai et al., 2024), SpecCascade (Narasimhan et al., 2024) and CoS (Fu et al., 2025) increase speed but compromise distribution fidelity and require task-specific tuning. In addition, Medusa and EAGLE always accept the first draft token to improve throughput, trading off exact recovery of the target distribution.
this section cite: ['b13', 'b16', 'b32', 'b5', 'b29', 'b10', 'b15', 'b2', 'b14', 'b6', 'b30', 'b11', 'b25', 'b12', 'b20', 'b33', 'b2', 'b17', 'b8']

Section: REVISITING TOKENWISE SPECULATIVE DECODING
In tokenwise speculative sampling (Leviathan et al., 2023), each token x t is drafted from q(x t ) and verified against p(x t ). It is accepted with probability h(x t ) = min{1, p(x t )/q(x t )}, or rejected and replaced from P res (x t ). Thus the probability that x t is finally produced ("yielded") is: P (x t yielded) = P (x t drafted and accepted) + P (x t drafted and rejected, x t resampled). (1)
Accept term. If x t is proposed by q and accepted,
P (x t drafted and accepted) = q(x t ) h(x t ) = q(x t ) min{1, p(x t )/q(x t )}.(2)
Resampling term. When a draft xt is rejected, the verifier resamples from
P res (x t ) = p(x t ) -min{p(x t ), q(x t )} xt∈V p(x t ) -min{p(x t ), q(x t )} .
The total probability of rejection is xt∈V q(x t )(1 -h(x t )), giving P (x t drafted and rejected,
x t resampled) = xt∈V q(x t )(1 -h(x t )) P res (x t ).(3)
Final distribution. The sum xt∈V q(x t ) (1-h(x t )) corresponds to the total excess mass assigned by the draft distribution to tokens where it allocates more probability than the target, while the denominator of P res (x t ) measures the total deficient mass, i.e., the probability assigned by the target to tokens where it allocates more than the draft. For tokenwise distributions these match (D LK (q, p) = D LK (p, q)), so they cancel, yielding
P (x t is yielded) = q(x t )h(x t ) + D LK (q, p) p(x t ) -q(x t )h(x t ) D LK (p, q) = p(x t ).
this section cite: ['b13']

Section: THEORETICAL FOUNDATIONS OF HIERARCHICAL SPECULATIVE DECODING
For any lossless speculative decoding, the probability of generating an output decomposes into two parts: (1) the probability a draft is accepted, becoming the final output, and (2) the probability a draft is rejected, triggering a corrective resampling step. In token-wise speculative decoding, resampling is straightforward because each token's probability is directly accessible. In contrast, full joint probabilities over sequences are intractable for auto-regressive models. Hierarchical Speculative Decoding (HSD) overcomes this via hierarchical branch resampling, where multiple resampling distributions at different levels recover partial target distributions, which together statistically recover the full distribution. This section formalizes the theoretical foundations.
this section cite: []

Section: RECOVERY OF PARTIAL DISTRIBUTIONS
To guide recovery within accessible subsets, we extend the divergence from Leviathan et al. (2023) to partial distributions. Let ω be a token or sequence, Ω the full sample space, and p(•), q(•) the target and draft distributions. For Ω ′ ⊆ Ω, define the generalized divergence: Definition 1. Generalized Divergence. Given two distributions p and q over a sample space Ω, and a subset Ω ′ ⊆ Ω, the generalized divergence over Ω ′ is defined as:
D Ω ′ (p, q) = ω∈Ω ′ max{p(ω) -q(ω), 0}.(4)
The generalized divergence D Ω ′ (p, q) measures the total deficient mass, i.e., how much probability mass is missing in the draft q relative to the target p within the subset Ω ′ . The reverse divergence D Ω ′ (q, p) measures the corresponding excess mass. In the whole space Ω, this is symmetric (see Lemma 1 in Section A.1 ) and reduces to the divergence from Leviathan et al. (2023) (see Lemma 2 in Section A.4), which underpins standard token-wise speculative decoding.
Next, we formalize the condition under which the partial target distribution is fully recoverable:
Theorem 1. Partial Distribution Recovery. A target distribution over Ω ′ ⊆ Ω can be fully recovered via resampling iff D Ω ′ (p, q) ≤ D Ω ′ (q, p). (See proof in Section A.2.)
Intuitively, this ensures the "trigger mass" in the draft is sufficient to compensate for the deficit in the target distribution. Over the full space Ω, symmetry guarantees full recoverability.
this section cite: ['b13', 'b13']

Section: RESAMPLING WITHIN THE ACCESSIBLE BRANCH
With these definitions, we analyze resampling within accessible branches along a draft sequence.
Although computing full joint probabilities is intractable, the probabilities of all next tokens over the vocabulary V are accessible given any prefix X 1:t-1 . We define a branch as:
Branch(X 1:t-1 ) = {X 1:t = (X 1:t-1 , xt ) | xt ∈ V}.(5)
Branch divergence will guide redistribution of excess probability mass to correct local deficits.
Since only joint probabilities p(X 1:t ) within a given branch Branch(X 1:t-1 ) are available, we introduce branch divergence to quantify local deficits in the draft:
Definition 2. Branch Divergence D Branch (p, q | X 1:t-1 ) = X1:t∈Branch(X1:t-1) max{p (X 1:t ) -q (X 1:t ) , 0}(6)
Branch divergence captures how much probability mass is missing locally. Unlike total divergence, it is inherently asymmetric, motivating the definition of branch asymmetry:
Definition 3. Asymmetry of Branch Divergence ∆ Branch (X 1:t-1 ) = D Branch (p, q | X 1:t-1 ) -D Branch (q, p | X 1:t-1 )(7)
Asymmetry essentially reflects the probabilistic imbalance within the current branch. Here, ∆ Branch > 0 indicates a deficit that cannot be corrected within the branch alone, while ∆ Branch < 0 represents excess mass available to support other branches. It can be computed as follows:
Theorem 2. Quantifying Asymmetry of Branch Divergence (see proof in Section A.3): ∆ Branch (X 1:t-1 ) = p (X 1:t-1 ) -q (X 1:t-1 ) ,(8)
From Theorem 1 and Theorem 2, we conclude that resampling can fully recover the target distribution over a branch whenever the draft has enough probability mass to cover the deficit:
Corollary 3. The target distribution over the Branch(X 1:t-1 ) can be recovered via resampling, under the following condition: p(X 1:t-1 ) ≤ q(X 1:t-1 ) or, equivalently, r(X 1:t-1 ) ≤ 1
where r (X 1:t-1 ) = p(X1:t-1) q(X1:t-1) denotes the probability ratio.
For drafts of length γ, the full target distribution cannot be recovered by applying verification solely within the accessible Branch(X 1:γ-1 ). However, we observe that the unused probability mass in certain branches can be leveraged to compensate for the unrecoverable mass in other branches, from a statistical perspective. This motivates the hierarchical branch resampling approach discussed next.
this section cite: []

Section: RESAMPLING IN A HIERARCHY OF ACCESSIBLE BRANCHES
Accessible branch divergences naturally form a hierarchical structure that enables systematic redistribution of excess probability mass. Specifically:
Theorem 4.
this section cite: []

Section: Hierarchy of Branch Divergence
The total positive asymmetry of branch divergence across child branches is equal to the parent branch divergence, and vice versa. Specifically:
∆Branch(X1:t-2,xt-1)>0
∆ Branch (X 1:t-2 , xt-1 ) = D Branch (p, q | X 1:t-2 ), and vice versa, (10)
where X 1:t-2 , xt-1 ranges over all possible Branches with the shared prefix X 1:t-2 , and X 1:t-2 is the accessible branch along the draft sequence. (See Section A.5 for the proof.)
This result guarantees that excess mass from overrepresented branches can be aggregated to offset deficits in underrepresented branches. Thus, hierarchical branch resampling guarantees exact recovery of the target distribution, even when individual branches cannot. This provides a rigorous theoretical foundation for deriving Hierarchical Speculative Decoding.
Algorithm 1 Naive HSD
Require: Target probabilities: {p(•), ..., p(•|X1:γ)} Require: Draft probabilities: {q(•), ..., q(•|X1:γ-1)} Require: Draft tokens X1:γ = {x1, ..., xγ} 1: Initialize τ = 0 2: for t in γ : 1 do 3: Sample ηt ∼ U (0, 1) 4: if ht ≥ ηt then 5: Set τ = t #accept X1:t 6: break 7: else 8: Set τ = t -1 #reject xt 9: continue #step back 10: end if 11: end for 12: if τ = γ then 13: Sample token from p(•|X1:γ) #bonus token 14: else 15: for t in τ : γ -1 do 16: Sample token from Pres(• | X1:t) #resample 17: end for 18: end if Ensure: [X1:τ , xτ+1, . . . , xγ] Algorithm 2 HSD Require: Target probabilities: {p(•), ..., p(•|X1:γ)} Require: Draft probabilities: {q(•), ..., q(•|X1:γ-1)} Require: Draft tokens X1:γ = {x1, ..., xγ} 1: Initialize τ = 0 2: for t in γ : 1 do 3: Sample ηt ∼ U (0, 1) 4: if ht ≥ ηt then 5: Set τ = t #accept X1:t 6: break 7: else 8: Set τ = t -1 #reject xt 9: continue #step back 10: end if 11: end for 12: if τ = γ then 13: Sample token from p(•|X1:γ) #bonus token 14: else 15: Sample token from P * res (• | X1:τ ) #resample 16: end if Ensure: [X1:τ , token]
5 HIERARCHICAL SPECULATIVE DECODING Guided by the theoretical foundations, we first develop a naive algorithm (see 5.1) that exactly recovers the target distribution. The procedure evaluates a candidate sequence X 1:γ and scans backward to identify the longest accepted prefix X 1:τ , then recursively resamples positions τ + 1 through γ using the corresponding distributions from the resampling hierarchy. This naive approach, however, still requires γ -τ + 1 additional calls to the target model, since the resampled branches are inaccessible. To remove this overhead, we introduce Capped Branch Resampling, yielding our final Hierarchical Speculative Decoding (HSD). HSD recovers the target distribution with just one resampling step within the accessible branches. Concretely, after the resampling step at line 15 in Algorithm 2, HSD only needs to sample from the target distribution to continue generation until γ, which can be replaced by another speculative decoding step, eliminating additional target calls.
this section cite: []

Section: NAIVE HIERACHICIAL SPECULATIVE DECODING
Specifically, the acceptance probability is computed according to the following formula:
Acceptance Probability h γ = min{r(X 1:γ ), 1}, and when t < γ:
h t = D Branch (p, q | X 1:t ) max{D Branch (p, q | X 1:t ), D Branch (q, p | X 1:t )} ,(11)
Branch Resampling Probability (line 17 in Algorithm 1):
P res (x t | X 1:t-1 ) = max {p (X 1:t ) -q (X 1:t ) , 0} D Branch (p, q | X 1:t-1 )(12)
Branch Divergence D Branch (p, q | X 1:t-1 ) is defined in Definition 2. By construction, the Branch Resampling Probability is defined within the accessible Branch(X 1:t-1 ), i.e., P res (X 1:t | Branch(X 1:t-1 )), which reduces to the token-level form P res (x t | X 1:t-1 ).
The probability of the Target Model generating a sequence X 1:γ can be decomposed into two disjoint events: (i) full acceptance of the draft, or (ii) at least one rejection followed by resampling:
P X 1:γ is yielded = P X 1:γ is sampled as draft, X 1:γ is accepted + X1:γ ̸ =X1:γ P ( X1:γ sampled and rejected, X 1:γ resampled). (13
)
Accept term: probability for the case when X 1:γ is sampled as draft and then directly accepted.
P X 1:γ is sampled as draft, X 1:γ is accepted = q(X 1:γ ) sample probability min{r(X 1:γ ), 1} accept probability at γ(14)
If r(X 1:γ ) ≤ 1, this equals to the target probability p(X 1:γ ). Otherwise, it is equal to q(X 1:γ ), and the residual probability p(X 1:γ ) -q(X 1:γ ) is compensated via resampling.
Resampling term (partially resampled): This term accounts for all cases where X 1:γ is obtained by resampling. Note that the accepted prefix must exactly match the corresponding subsequence of X 1:γ for this contribution to apply. Therefore, we can further decompose it by summing over all possible positions τ + 1 of the first rejected token, with τ being the length of the longest accepted prefix: γ τ=0 Xτ+1:γ P ( X1:γ sampled and rejected,
X 1:γ resampled) = γ τ =0 Xτ+1:γ q(X 1:τ Xτ+1:γ ) • γ t=τ +1 (1 -h t ) • h τ X 1:τ • γ t=τ +1 P res (x t )(15)
Explanation of terms:
1. Sampling: q(X 1:τ Xτ+1:γ ) is the probability of generating the initial draft sequence. 2. Backward Scan: γ t=τ +1 (1-h t ) corresponds to scanning backward from the end, rejecting tokens until the first accepted prefix is found. 3. Acceptance: h τ is the probability of accepting the longest prefix X 1:τ . 4. Resampling: γ t=τ +1 P res (x t ) resamples the remaining positions to recover exactly the target probability. This decomposition defines the procedure underlying Algorithm 1 and provides the basis for its provable losslessness. The complete proof is given in Section B.2, together with an illustrative example Section B.1 showing how naive HSD recovers the target distribution.
this section cite: []

Section: HIERARCHICAL SPECULATIVE DECODING WITH CAPPED BRANCH RESAMPLING
To introduce the capped branch sampling, we first define the Maximum Prefix Ratio Index. Definition 4. Maximum Prefix Ratio Index For candidate tokens X 1:t , the Maximum Prefix Ratio Index m(X 1:t ) is the position in the prefix X 1:t-1 where the joint probability ratio r(X
1:i ) is maximized; if no prefix exceeds 1, we set m(X 1:t ) = 0: m(X 1:t ) = arg max 1≤i<t r(X 1:i ) or 0 if max 1≤i<t r(X 1:i ) ≤ 1.
Based on the Maximum Prefix Ratio Index, we define the Capped Prefix Ratio r * as follows:
Definition 5. Capped Prefix Ratio r * (X 1:t ) = min{r(X 1:m(X1:t) ), 1}r(X m(X1:t)+1:t ).
(16) By Definition 5, we have r(X 1:m(X1:t) ) > 1, and according to Equation ( 16), this implies the identity r * (X 1:t ) = r X m(X1:t)+1:t .
Then we define the Capped Branch Divergence:
Definition 6. Capped Branch Divergence
D * Branch (p, q | X 1:t-1 ) = X1:t∈Branch(X1:t-1); r * (X1:t)>1 (r * (X 1:t ) -1) q (X 1:t ) (17
)
D * Branch (q, p | X 1:t-1 ) = X1:t∈Branch(X1:t-1); r * (X1:t)≤1 (1 -r * (X 1:t )) q (X 1:t )(18)
Finally, the acceptance probability is computed according to the following formula:
Acceptance Probability h γ = min{r * (X 1:γ ), 1}, and when t < γ:
h t = D * Branch (p, q | X 1:t ) D * Branch (q, p | X 1:t ) ,(19)
Capped Branch Resampling Probability (line 15 in Algorithm 2):
P * res (x t | X 1:t-1 ) = max {q (X 1:t ) (r * (X 1:t ) -1) , 0} D * Branch (p, q | X 1:t-1 )(20)
We refer to the above strategy as Capped Branch Resampling. It plays a central role in enabling efficient resampling within the hierarchical branch resampling framework. The resampling distribution in Equation ( 20) enables recovery of the full target distribution with only a single resampling step for branches with negative asymmetry. The remaining positions can then be directly sampled from the target model, aligning with the start of the next speculative decoding step and thus incurring no extra computational cost.
We briefly clarify the core mechanism by which capping preserves the target joint distribution. From Definition 5 and Definition 6, it follows that D * Branch (p, q | X 1:t ) = X1:t∈Branch(X1:t-1) max{q X 1:m(X1:t) p X m(X1:t)+1:t -q (X 1:t ) , 0}. Through the acceptance probability and resampling probability at position t, we essentially guarantee that the probability of obtaining X 1:t is equal to q X 1:m(X1:t) p X m(X1:t)+1:t , partially recovering the probability of the fragment X m(X1:t)+1:t . And the deficient probability mass p(X 1:m(X1:t) ) -q(X 1:m(X1:t) ) is statistically recovered from the resampling distributions in higher hierarchies, which corresponds to the fragments X1:m(X1:t) of other trajectories. An illustrative example in Section C.1 demonstrates how the algorithm recovers loss over the entire path, with a further explanation of the capped ratio provided in Section C.3.
this section cite: []

Section: COMPUTATIONAL EFFICIENCY
The verification stage in HSD adds negligible overhead compared to the savings from reduced target model forward passes. Thanks to parallelized computations across both the vocabulary and draft positions, HSD is nearly as efficient as tokenwise verification. Runtime measurements (Appendix H) show that verification accounts for less than 1% of total decoding time, with the majority still spent on draft and target forward passes. These results demonstrate that HSD is not only theoretically lossless but also practically efficient, as further confirmed by our experiments in Section 6.
this section cite: []

Section: GSM8K Example
Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?
To determine Eliza's earnings for the week, we need to calculate both her regular pay and her overtime pay.
1. ** Calculate Regular Pay: ** -Eliza's regular rate is $10 per hour. -She worked 40 hours at her regular rate. -Regular pay: 40 hours × $10/hour = $400. 2. ** Calculate Overtime Pay: ** -Eliza worked a total of 45 hours, so
this section cite: []

Section: ILLUSTRATIVE EXAMPLE
We use a GSM8K question as a running example to demonstrate HSD (see Figure 2). This example emphasizes the hierarchical acceptance mechanism and the capping behavior that are key to HSD.
Next-token probabilities. Under the given prefix, the large (target) and small (draft) models produce the next-token probabilities: The corresponding draft tokens are: {she, work, ed, 45, -, 40, =, 5, hours, of}.
Joint probabilities and ratios. We compute the joint probabilities along the draft: {p(X1:t)} γ t=1 = {0.7156, 0.7156, 0.7156, 0.7156, 0, 0, 0, 0, 0, 0}, {q(X1:t)} γ t=1 = {0.8771, 0.6929, 0.4513, 0.1170, 0.0792, 0.0118, 0.0068, 0.0068, 0.0031, 0.0011}, {r(X1:t)} γ t=1 = {0.8159, 1.0327, 1.5855, 6.1171, 0, 0, 0, 0, 0, 0}.
These ratios exhibit early growth above 1 (at t = 2, 3, 4) and collapse to 0 once the target probability vanishes (from t ≥ 5).
Maximum prefix indices and capped ratios. Following Definition 4, the maximum prefix indices and capped ratios are {m(X1:t)} γ t=1 = {0, 0, 2, 3, 4, 4, 4, 4, 4, 4} and {r * (X1:t)} γ t=1 = {0.8159, 1, 1, 1, 0, 0, 0, 0, 0, 0} .
Capped branch divergences and acceptance. On the full vocabulary branch Branch(X 1:t-1 ), we evaluate the capped branch divergences: The hierarchical acceptance (Eq. 19) then yields {ht} γ t=1 = {0.1231, 1, 1, 1, 0, 0, 0, 0, 0, 0} . Acceptance saturates at t = 2, 3, 4, implying the first four tokens are validated, i.e., n match = 4.
Comparison to tokenwise verification. For a tokenwise baseline that validates strictly left-to-right, the per-position magnitudes are {h tokenwise t } γ t=1 = {0.8159, 1, 1, 1, 0, 0, 0, 1, 0, 1} . Since the baseline commits at the first position, an initial h 1 = 0.8159 may trigger rejection and discard the entire draft block.
this section cite: []

Section: EXPERIMENTS
In this section, we empirically demonstrate the superiority of HSD with comparison on various benchmarks and configurations, comprehensive ablation studies, and in-depth analysis of results.
this section cite: []

Section: EXPERIMENT SETTING
Experiments Setup. Experiments are conducted with the widely adopted GPTQ-quantized 8-bit instruction-tuned Qwen2.5 series (Bai et al., 2023). By default, we employ the 0.5B as the draft model and 72B as the target models, with a temperature of 1. We leverage GSM8K (Cobbe et al., 2021) for mathematical problem-solving, HumanEval (Chen et al., 2021) for code generation, and CNN/DailyMail (See et al., 2017) for text summarization. All experiments were conducted on a single NVIDIA H20 GPU with 96 GB of memory, unless otherwise specified.
Baselines and Metrics. We compare two lossless verification methods-Token-wise and Blockwise-using two metrics: Block Efficiency (tokens/step) and Decoding Speed (tokens/second). Block Efficiency measures the average tokens generated per serial call to the target model, reflecting intrinsic efficiency independent of hardware. Decoding Speed indicates tokens produced per second for practical reference. Additional details and extended evaluations are in Section E.
this section cite: ['b1', 'b4', 'b3', 'b21']

Section: EXPERIMENT RESULTS

this section cite: []

Section: Main results.
Table 1 summarizes the performance of HSD across datasets and model scales using the Qwen2.5 suite (0.5B as draft,14B, 32B, and 72B as targets). Overall, HSD consistently improves both Block Efficiency (BE) and Decoding Speed (DS) relative to Tokenwise and Blockwise verification. For GSM8K, the gains are stable across scales, with BE improvements of 5.2%-5.4% at 14B/32B and 3.3% at 72B, accompanied by DS increases of up to 10.7%. On HumanEval, the effect is more pronounced: BE rises by 9.5% and 12.3% at 14B and 32B, while DS improves by 9.3% and 11.4%; even at 72B, HSD maintains positive margins (3.3% BE, 4.5% DS). For CNN/DailyMail, the improvements are moderate but consistent, with BE gains of 4.2%-8.4% and DS gains of 3.4%-7.2%. On average, HSD provides consistent advantages over Tokenwise and Blockwise verification, with improvements of approximately 6.2% in BE and 6.7% in DS.
this section cite: []

Section: Multi-draft.
To demonstrate the compatibility of HSD, we compare it with token-wise verificaiton in a multi-draft setting. For simplicity-and without loss of generality-we adopt Recursive Reject Sampling (RRS) with replacement (Yang et al., 2024) as the baseline for its scalability and independence from complex tree attention mechanisms. Notably, since it is not straightforward to extend blockwise verification to the multi-draft setup, we omit it from our comparison. We evaluated multi-draft generation with 11 candidate drafts in Table 2, and HSD yields an average 5.9% improvement in Block Efficiency and 4.7% improvement in Decoding Speed over token-wise decoding.   Ablation on Temperature. We conduct a systematic evaluation of sampling temperature's effect on decoding efficiency, with t ∈ {0.6, 0.8, 1.0} (Table 3(a)). HSD consistently outperforms other approaches across all temperature settings, demonstrating its robustness to temperature variations.
Ablation on Draft Length. We evaluate draft lengths γ ∈ {5, 10, 15} tokens, where HSD consistently outperforms baselines with increasing efficiency gains (Table 3( b)). At γ = 15, HSD achieves peak performance with 7.88 tokens/step in block efficiency and 52.95 steps/second in decoding speed, representing improvements of 3.58% and 3.88% over Tokenwise, respectively. The consistent performance advantage across all draft lengths demonstrates HSD's robust scalability.
Extended Results. We conducted additional experiments using Llama-3.1-70B-Instruct and Llama-3.1-8B-Instruct pair (non-quantized version), with model weights distributed on 8 H20 GPUs. The results are shown in Table 4a. Moreover, we integrated HSD into the SOTA EAGLE-3-LLaMa3.1-Instruct-8B (γ = 7) by replacing its tokenwise verifier in Table 4b. Following EAGLE-3, we accept at least the first draft token for a fair comparison. Note that EAGLE-3 utilizes top-K sampling for drafting, making all draft probabilities equal to 1. In this case, any verification method theoretically degenerates into the same behavior and the observed gain in block efficiency of HSD is likely influenced by sampling stochasticity and floating-point precision. However, the observed significant practical speedup in decoding speed is expected, since our implementation (see Section F) avoids the explicit loops in EAGLE's implementation of tokenwise verification.
From Lemma 1, we know that D Ω (p, q) = D Ω (q, p), so we can write:
D Ω (p, q) = D Ω (p, q) + D Ω (q, p) 2 = 1 2 x∈Ω max{p(x) -q(x), 0} + x∈Ω max{q(x) -p(x), 0} (A.11)
Therefore, D Ω (p, q) = D LK (p, q), completing the proof.
this section cite: ['b30']

Section: A.5 HIERARCHY OF DIVERGENCE
Proof. Proof of Theorem 4.
From Theorem 2, we recall that:
∆ Branch (X 1:t-2 , xt-1 ) = p(X 1:t-2 , xt-1 ) -q(X 1:t-2 , xt-1 ). (A.12)
Therefore, summing over the cases where this difference is positive gives:
∆Branch(X1:t-2,xt-1)>0 ∆ Branch (X 1:t-2 , xt-1 ) = xt-1∈V max {p(X 1:t-2 , xt-1 ) -q(X 1:t-2 , xt-1 ), 0} . (A.13) By Definition 2, this is precisely the branch divergence one level higher D Branch (p, q | X 1:t-2 ), thus completing the proof. B LOSSLESS OF NAIVE HIERARCHICAL SPECULATIVE DECODING B.1 ILLUSTRATIVE EXAMPLE For example, consider the case where r(X 1:γ ) > 1, r(X 1:γ-1 ) > 1, and r(X 1:γ-2 ) ≤ 1.
The accept term is simply equal to q(X 1:γ ), so we only need to check whether the resampling term equals p(X 1:γ ) -q(X 1:γ ). According to Equation ( 12), we know P res (x γ-2 | X 1:γ-1 ) = 0. Consequently, contributions from positions earlier than γ -1 in the sum above vanish, which implies that the resampling term for X 1:γ arises solely from resampling at positions γ and γ -1 as follows:
xγ P sample X 1:γ-1 xγ , reject xγ , accept X 1:γ-1 , resample x γ + Xγ-1:γ P sample X 1:γ-2 xγ-1:γ , reject Xγ-1:γ , accept X 1:γ-2 , resample X γ-1:γ = xγ q(X 1:γ-1 xγ ) draft probability • (1 -h γ ) reject backwards at τ + 1 = γ • h γ accept X1:γ-1 • P res (x t ) resample at τ + 1 = γ + xγ-1 xγ q(X 1:γ-2 xγ-1 xγ ) draft probability • (1 -h γ )(1 -h γ-1 ) reject backwards at τ +1 = γ -1 • h γ-2 accept X1:γ-1 •P res (x γ-1 )P res (x γ ) resample at τ + 1 = γ (A.14)
From Definition 2 that the excess probability mass that triggers resampling
D Branch (q, p | X 1:γ-1 ) = xγ q(X 1:γ-1 xγ )(1 -h γ ).
Then we have:
=D Branch (q, p|X 1:γ-1 ) • 1 • p(X 1:γ ) -q(X 1:γ ) D Branch (p, q|X 1:γ-1 ) + xγ-1 D Branch (q, p|X 1:γ-2 xγ-1 )(1 - D Branch (p, q|X 1:γ-2 xγ-1 ) D Branch (q, p|X 1:γ-2 xγ-1 ) )P res (x γ-1 )P res (x γ ) (A.15)
From Definition 3 and Theorem 4, we know that xγ-1 D Branch (q, p|X 1:γ-2 xγ-1 ) -D Branch (p, q|X 1:γ-2 xγ-1 ) = D Branch (q, p|X 1:γ-2 ). Then we have:
= D Branch (q, p|X 1:γ-1 ) D Branch (p, q|X 1:γ-1 ) • (p(X 1:γ ) -q(X 1:γ ))+ D Branch (q, p|X 1:γ-2 ) • p(X 1:γ-1 ) -q(X 1:γ-1 ) D Branch (p, q|X 1:γ-2 ) • p(X 1:γ ) -q(X 1:γ ) D Branch (p, q|X 1:γ-1 ) (A.16)
As definition 4, define the capped ratio at the end of the draft as
r * (X 1:γ ) := min{r(X 1:m ), 1} r(X m+1:γ | X 1:m ) = r(X m+1:γ | X 1:m ) ≤ 1,
and the accept term
A γ := q(X 1:γ ) r * (X 1:γ ).
We will also use three resample contributions: T γ (at level γ), T m (at level m), and T n (at level n).
two-peak example: n < m < γ From definition 4, we have r(X 1:n ) > 1, then r(X 1:m ) > r(X 1:n ), and no larger value occurs in (m, γ). This forces r(X n+1:m | X 1:n ) > 1; otherwise m could not be a new maximum.
Step 1: accept + top-level resample Since r * (X 1:γ ) = r(X m+1:γ | X 1:m ) ≤ 1, A γ = q(X 1:γ ) r(X m+1:γ | X 1:m ) = q(X 1:m ) p(X m+1:γ | X 1:m ), T γ = 0, so H 1 := A γ + T γ = q(X 1:m ) p(X m+1:γ | X 1:m ). Intuition. The suffix X m+1:γ is now under p; the prefix X 1:m is still under q. Step 2: add the m-term Let R n→m := r(X n+1:m | X 1:n ) > 1. The resample at level m contributes T m := q(X 1:m ) (R n→m -1) p(X m+1:γ | X 1:m ),hence
H 2 := H 1 + T m = R n→m q(X 1:m ) p(X m+1:γ | X 1:m ) = q(X 1:n ) p(X n+1:γ | X 1:n ).
Intuition. The block X n+1:m is converted to p; only X 1:n remains under q.
Step 3: add the n-term If r(X 1:n ) > 1, T n := q(X 1:n ) (r(X 1:n ) -1) p(X n+1:γ | X 1:n ), H 3 := H 2 + T n = p(X 1:γ ). If instead r(X 1:n ) ≤ 1, then T n = 0 and H 2 = p(X 1:γ ) already.
Intuition. Each nonzero term "tops up" the exact deficit of q on its block until the whole path is under p. Thus
A γ + T γ + T m + T n = p(X 1:γ )
in this two-peak case, exhibiting the (lossless) invariance of the total probability under the HSD accept-resample rule.
this section cite: []

Section: C.2 GENERAL PROOF
Definition 7 (Sequence of Unique Capping Indices). For a given maximum sequence length γ, the sequence of maximum prefix ratio indices (m(1), m(2), . . . , m(γ)) is generated according to Definition 4. Let U be the set of unique values in the sequence of capping indices:
U = {m(t) | 1 < t ≤ γ} (A.47)
The Sequence of Unique Capping Indices, denoted by M * , is the ordered sequence of the elements in U:
M * = (m * 1 , . . . , m * L ) (A.48)
where m * 1 < . . . < m * L and L is the total number of unique capping points.
With these definitions, we can now establish the key properties of the prefix-capped joint ratio: Lemma 5 (Property of r * (X 1:i ) between neighboring unique capping indices). Let m * l and m * l+1 be two consecutive unique capping indices, and suppose
m * l < i < m * l+1 . (A.49)
For every such i, we have r * (X
1:i ) ≤ 1. Since r * (X 1:γ ) = min{r(X 1:m * L ), 1}r(X m * L +1:γ ) and r(X 1:m * L ) > 1, we have r * (X 1:γ ) = r(X m * L +1:γ ). Case 1: If r(X m * L +1:γ ) ≤ 1, then: min 1, r * (X 1:γ ) q(X 1:γ ) + max 0, r * (X 1:γ ) -1 q(X 1:γ-1 ) p(x γ | X 1:γ-1 ) = r(X m * L +1:γ ) q(X 1:γ ) + 0 = r(X m * L +1:γ ) q(X 1:γ ) = q(X 1:m * L ) p(X m * L +1:γ | X 1:m * L ) = F L (A.59) Case 2: If r(X m *
L +1:γ ) > 1, then there would be another unique capping index beyond m * L , contradicting the definition of m * L as the last unique capping index. Therefore, we must have r(X m * L +1:γ ) ≤ 1, and thus:
min 1, r * (X 1:γ ) q(X 1:γ ) + max 0, r * (X 1:γ ) -1 q(X 1:γ-1 ) p(x γ | X 1:γ-1 ) = F L (A.60)
Therefore, we have:
P X 1:γ is generated = F L + L l=1 (F l-1 -F l ) = F L + (F 0 -F 1 ) + (F 1 -F 2 ) + • • • + (F L-1 -F L ) = F L + F 0 -F L = F 0 (A.61)
Now we evaluate F 0 . From Definition 10, we have:
F 0 = q(X 1:m * 0 ) p(X m * 0 +1:γ | X 1:m * 0 ) (A.62)
By our convention, m * 0 = 0, so:
F 0 = q(X 1:0 ) p(X 1:γ | X 1:0 ) = 1 • p(X 1:γ ) = p(X 1:γ ) (A.63) Therefore: P X 1:γ is generated = p(X 1:γ ) (A.64)
This completes the proof of lossless recovery.
this section cite: []

Section: C.3 A EXTENDED EXPLAINATION OF CAPPED RATIO
Let r(x 1 ), r(x 2 | x 1 ), . . . , r(x t | X 1:t-1 ) ∈ R >0 be a sequence of ratios.
Define the cumulative product up to index t as:
r(X 1:t ) = t i=1 r(x i | X 1:i-1 ), (A.65)
where X 1:0 is equal to the prefix.
Let j * be the last index (up to k) such that:
j * = max j ≤ k r(x j | X 1:j-1 ) > 1 and j i=1 r(x i | X 1:i-1 ) > 1 (A.66)
Then the capped cumulative product Rk is given by:
r * (X 1:t ) =   j * i=1 r(x i | X 1:i-1 )   •   k i=j * +1 r(x i | X 1:i-1 )   (A.67)
This ensures that the cumulative product is capped at the last index j * such that the individual ratio r(x j * |X 1:j * -1 ) > 1 and the cumulative product up to that point also exceeds 1.
When γ is 3, lets show simplest example to show the recovery of target probability.
P (X 1:3 is accepted ) = q(X 1:3 ) (A.68) P (X 1:3 is resampled) = γ=3 i=0 P (x γ , x γ-1 , . . . , x γ-i are resampled | X γ-i+1 ) = D * Branch (q, p | X 1:3 ) • max((r(x 3 ) -1)q(X 1:3 ), 0) D * Branch (q, p | X 1:3 ) + D * Branch (q, p | X 1:2 ) • max((r(x 2 ) -1)q(X 1:2 ), 0) D * Branch (q, p | X 1:2 ) • p(x 3 |X 1:2 ) + D * Branch (q, p | x 1 ) • max((r(x 1 ) -1)q(x 1 ), 0) D * Branch (q, p | x 1 ) • p(x 3 |X 1:2 )p(x 2 |x 1 ) (A.69)
Let's take γ = 3 as an example, only if r(X 1:3 ) > 1, the resampled portion of probability mass is needed. Suppose r(X 1:2 ) > 1 with r(x 1 ) > 1 and r(x 2 ) < 1:
= p(x 3 |X 1:2 )p(x 2 |x 1 )q(x 1 ) -q(X 1:3 ) + 0 + p(x 1 )p(x 2 |x 1 )p(x 3 |X 1:2 ) -q(x 1 )p(x 2 |x 1 )p(x 3 |X 1:2 ) = p(X 1:3 ) -q(X 1:3 ) (A.70) D EXPECTED NUMBER OF ACCEPTED TOKENS
We conduct efficiency analysis based on the expected acceptance length E[τ ]. For a given draft length γ, the expected number of accepted tokens for the tokenwise speculative decoding Leviathan et al. (2023), blockwise verification Sun et al. (2024), and our HSD are as follows: Lemma 9. Expected Number of Accepted Tokens (See Section D.1 for proof.)
E[τ ]token = γ i=1 i k=1 h token k , E[τ ]block = γ i=1 1 - γ k=i 1 -h block k , E[τ ]branch = γ i=1 1 - γ k=i (1 -h k ) (A.71)
We establish Theorem 7, which guarantees that HSD is more efficient than other lossless methods:
Theorem 7. HSD and Blockwise Achieves Better Expected Number of Accepted Tokens E[τ ] branch ≥ E[τ ] block ≥ E[τ ] token (A.72) where equality holds in both inequalities if and only if γ = 1. (See Section D.2 for proof.) We reveal that limitations on acceptance probability in each method directly cause the gap from the ideal case w.r.t. expected accepted tokens. Let r(x t ) = p(xt) q(xt) . The acceptance probability of the entire draft h γ is ideally min { γ t=1 r(x t ), 1}.
In contrast, tokenwise acceptance is h token = Let τ ∈ {0, 1, . . . , γ} denote the number of accepted tokens in a decoding attempt. Since τ is a non-negative, integer-valued random variable, the tail-sum identity applies with lattice spacing a = 1.
so the claim holds.
Inductive step. Assume (A.85) holds for some t -1 ≥ 0. Using the recurrence, p t = min 1, p t-1 r t .
(A.88) By the induction hypothesis,
p t-1 = min 0≤s≤t-1 t-1 i=s+1 r i . (A.89) Substituting, p t = min 1, min 0≤s≤t-1 t-1 i=s+1 r i r t . (A.90)
Multiplying every candidate product in the inner minimum by r t and then taking the outer minimum yields exactly all suffix products t i=s+1 r i (A.91) for s = 0, . . . , t -1, together with the empty product 1 for s = t. Hence (A.85) holds for t, completing the induction. And obviously, p t < r(X start:t ), where start ∈ (1, t -1)
h block t = xt+1 (p t r(x t+1 |X 1:t ) -1) + q(x t+1 | X 1:t ) xt+1 (p t r(x t+1 |X 1:t ) -1) + q(x t+1 | X 1:t ) + 1 -p t = xt+1 (min r(x t+1 ),r(X t:t+1 ),r(X t-1:t+1 ), . . . ,r(X 1:t+1 ) -1) + q(x t+1 | X 1:t ) xt+1 (min r(x t+1 ), r(X t:t+1 ), r(X t-1:t+1 ),. . .,r(X 1:t+1 ) -1) + q(x t+1 | X 1:t )+1-p t (A.92) Since min r(x t+1 ), r(X t:t+1 ), r(X t-1:t+1 ), . . . , r(X 1:t+1 ) ≤ r(X m(X1:t+1)+1:t+1 ), h block t ≤ xt+1 (r(X m(X1:t+1)+1:t+1 ) -1) + q(x t+1 | X 1:t ) xt+1 (r(X m(X1:t+1)+1:t+1 ) -1) + q(x t+1 | X 1:t ) + 1 -p t ≤ xt+1 (r(X m(X1:t+1)+1:t+1 ) -1) + xt+1 (r(X m(X1:t+1)+1:t+1 ) -1) + + 1 -p t (A.93) From equation A.83: xt+1 (1 -r(X m(X1:t+1)+1:t+1 )) = q(X m(t+1)+1:t ) -p(X m(t+1)+1:t ) = (1 -r(X m(t+1)+1:t ))q(X m(t+1)+1:t ) ≤ (1 -p t )q(X m(t+1)+1:t ) ≤ (1 -p t ) (A.94) Since h branch t = xt+1 [r(X m(X1:t+1)+1:t+1 ) -1] + xt+1 [r(X m(X1:t+1)+1:t+1 ) -1] + + xt+1 (1 -r(X m(X1:t+1)+1:t+1 )) ≥ xt+1 [r(X m(X1:t+1)+1:t+1 ) -1] + xt+1 [r(X m(X1:t+1)+1:t+1 ) -1] + + 1 -p t ≥ h block t (A.95)
Table A.1: Comparison of different algorithm performance on GSM8K with Qwen-2.5. We list the average and standard deviation across 5 runs with different seeds. Method Tokenwise Blockwise Ours Block Efficiency 6.40±0.10 6.51±0.09 6.64±0.04 Decoding Speed 31.52±0.06 31.70±0.05 32.61±0.02 Table A.2: Comparison of task performance across model sizes and methods. Metric Method 72B 32B 14B GSM8K (Accuracy) Tokenwise 0.8213 0.8213 0.8327 HSD 0.8517 0.8479 0.8327 Table A.3: Ablation on capping mechanism. Dataset Method ACC BE DS GSM8K HSD 84.40±1.75% 6.76±0.05 33.63±0.53 HumanEval HSD 80.61±0.69% 5.60±0.06 29.15±0.43 GSM8K HSD + Capping 84.96±0.93% 6.63±0.06 32.73±0.55 HumanEval HSD + Capping 82.47±1.15% 5.45±0.08 27.54±0.48 E EXTENDED EXPERIMENTS Result Robustness To prove the robustness of our experiments and guarantee fair comparison, we conduct additional experiments with different methods as shown in Table A.1. We observe that our method demonstrates stable performance and exceeds both tokenwise and blockwise methods on average.
this section cite: ['b13', 'b25']

Section: Verification of Task Performance
We compare our method with the token-wise approach on GSM8K. As shown in Table A.2, our method achieves equivalent (or better) accuracy among different model sizes, demonstrating the preserved distributional fidelity.
this section cite: []

Section: Capped Prefix Ratio Ablation Study
We conducted ablations on the role of the capped prefix ratio in Algorithm 2 (HSD), which is essential to preserve distributional fidelity, as shown in Table A.3. Removing capping (i.e., directly using the uncapped ratio to compute divergences) yields a slight increase in efficiency, but at the cost of varying degrees of performance degradation (which may or may not be obvious depending on the task).
this section cite: []

Section: F PYTHON IMPLEMENTATION
We provide the Python implementation of our Hierarchical Speculative Decoding (HSD) algorithm in Listing 2, which builds upon the token-wise speculative decoding approach from Hugging Face Wolf et al. (2020) Transformers v4.46.3, shown in Listing 1 for comparison. Following Hugging Face, our implementation eliminates the use of an explicit for-loop by leveraging an equivalent masking mechanism: we perform parallel sampling across all positions to determine whether to accept or reject subsequences of varying lengths, and then select the longest accepted prefix as the final output.
Published as a conference paper at ICLR 2026 Listing 2 Hierarchical Speculative Decoding HSD.py (cont.) 1 if is_done_candidate[:] and n_matches == candidate_length: 2 # Output length is assumed to be 'n_matches + 1'. Since we won't generate another ↩→ token with the target model 3 # due to acceptance on EOS we fix 'n_matches' 4 n_matches -= 1 5 # valid_tokens = new_candidate_input_ids[:, : n_matches + 1] 6 valid_tokens = candidate_input_ids[:, -candidate_length:] 7 8 else: 9 # Next token selection: if there is a rejection, adjust the distribution from the main ↩→ model before sampling. 10 gamma = candidate_length 11 p_n_plus_1 = p[:, candidate_length, :] 12 if n_matches < gamma: 13 p_prime = p_primes[:, n_matches] 14 p_prime = p_prime/p_prime.sum(-1, keepdim=True) 15 else: 16 p_prime = p_n_plus_1 17 18 # The selected tokens include the matches (if any) plus the next sampled tokens 19 # because if n_matches=0, we add one resampled token for sure, if n_matches=10, we add ↩→ one more for sure 20 # as well, because the previous if checked not stop and n_matches-candidate_length ↩→ will be 0 causing problem 21 if n_matches > 0 and n_matches<candidate_length: 22 valid_tokens = candidate_input_ids[:, -candidate_length:n_matches-candidate_length ↩→ ] 23 if not stop(candidate_input_ids[:, :n_matches-candidate_length], scores=None): 24 t = torch.multinomial(p_prime, num_samples=1) 25 valid_tokens = torch.cat( 26 (valid_tokens, t), dim=-1) 27 else: 28 n_matches = n_matches-1 29 else: 30 t = torch.multinomial(p_prime, num_samples=1) 31 if n_matches==0: 32 valid_tokens = t 33 else: 34 valid_tokens = candidate_input_ids[:, -candidate_length:] 35 valid_tokens = torch.cat( 36 (valid_tokens, t), dim=-1) 37 38 return valid_tokens, n_matches
as target and draft models on a single H200 GPU. As shown in Table A.4, the verification stage consistently accounts for less than 1% of the total decoding time. At the same time, the vast majority of runtime is spent in the draft and target forward passes. Here, the draft forward pass accounts for about 24% of the runtime, and the target forward pass accounts for about 72% in both blockwise and HSD. The verification stage of HSD is about 20% faster than that of blockwise.
Table A.4: Runtime breakdown of Blockwise and HSD. Component Blockwise Mean (ms/token) Blockwise % HSD Mean (ms/token) HSD % Total 34.168 100.00% 33.788 100.00% Prefill 0.913 2.67% 0.915 2.71% Draft Forward 8.210 24.03% 7.865 23.28% Target Forward 24.690 72.26% 24.695 73.09% KV Cache Input 0.007 0.02% 0.005 0.01% KV Cache Output 0.070 0.20% 0.067 0.20% Logits Processing 0.093 0.27% 0.103 0.31% Verification 0.160 0.47% 0.127 0.37% Other 0.025 0.08% 0.012 0.03%
this section cite: []

Section: G INTEGRATION WITH RECURSIVE REJECT SAMPLING IN THE MULTI-DRAFT SETUP
We demonstrate in Algorithm 3 that our HSD algorithm is compatible with existing lossless multidraft verification methods, exemplified by Recursive Reject Sampling (RRS) with replacement Yang et al. (2024). Notably, independently sampled parallel draft sequences do not guarantee the existence of an additional draft sequence that shares the accepted subsequence as its prefix.
this section cite: ['b30']

Section: 
Thus, D Ω (p, q) = D Ω (q, p), completing the proof.
this section cite: []

Section: A.2 PARTIAL DISTRIBUTION RECOVERY
Proof of Theorem 1. Let P (w is yielded) denote the total probability of producing w ∈ Ω ′ . By construction, this can be decomposed as P (w is yielded) = P (w is drafted & accepted) + P (w is drafted & rejected, w is resampled), (A.3) where acceptance occurs with probability h(w) = min{p(w)/q(w), 1}, and resampling follows the distribution P res (• | Ω ′ ) with total trigger mass D Ω ′ (q, p). Here, the total trigger mass represents the sum of probabilities of all draft outcomes in Ω ′ that are rejected. Hence, P (w is yielded) = h(w) q(w) + D Ω ′ (q, p) P res (w | Ω ′ ). (A.4) Noting that h(w) q(w) = min{p(w), q(w)}, we have P (w is yielded) = min{p(w), q(w)} + D Ω ′ (q, p) P res (w | Ω ′ ). (A.5) To match the target distribution exactly (P (w is yielded) = p(w)), we require P res (w | Ω ′ ) = p(w) -min{p(w), q(w)} D Ω ′ (q, p) = max{p(w) -q(w), 0} D Ω ′ (q, p) . (A.6) Summing over all w ∈ Ω ′ gives
w∈Ω ′ P res (w | Ω ′ ) = D Ω ′ (p, q) D Ω ′ (q, p) . (A.7)
For P res (• | Ω ′ ) to be a valid probability distribution, this sum must not exceed 1. Therefore, the necessary and sufficient condition is D Ω ′ (p, q) ≤ D Ω ′ (q, p), (A.8) which completes the proof.
this section cite: []

Section: A.3 QUANTIFICATION ANALYSIS OF ASYMMETRY
Proof. From Definition 3 and Definition 2, we obtain:
∆ Branch (X 1:t-1 ) = X1:t∈Branch(X1:t-1) max {p (X 1:t ) -q (X 1:t ) , 0} -X1:t∈Branch(X1:t-1) max {q (X 1:t ) -p (X 1:t ) , 0} = X1:t∈Branch(X1:t-1) p (X 1:t ) -X1:t∈Branch(X1:t-1) q (X 1:t ) = xt∈V p (X 1:t-1 ) p (x t | X 1:t-1 ) -xt∈V q (X 1:t-1 ) q (x t | X 1:t-1 ) = p (X 1:t-1 ) -q (X 1:t-1 ) (since xt∈V p(x t | X 1:t-1 ) = 1) (A.9)
this section cite: []

Section: A.4 RELATION TO THE DIVERGENCE IN LEVIATHAN ET AL. (2023)
Lemma 2. The total divergence is equivalent to the divergence defined in Leviathan et al. (2023) for token distributions over the full sample space.
Proof. Following Leviathan et al. (2023), let x denote a token, and omit conditions in the token probabilities for simplicity. From Definition 3.2 in Leviathan et al. (2023), we have:
D LK (p, q) = x∈Ω p(x) -q(x) 2 = 1 2 x∈Ω max{p(x) -q(x), 0} + x∈Ω max{q(x) -p(x), 0} (A.10)
Published as a conference paper at ICLR 2026
We know from Definition 3 and Theorem 2 that p(X 1:γ ) -q(X 1:γ ) = D Branch (p, q|X 1:γ-1 ) -D Branch q, p|X 1:γ-1 . Then we have:
= D Branch (q, p|X 1:γ-1 ) D Branch (p, q|X 1:γ-1 ) • (p(X 1:γ ) -q(X 1:γ ))+ D Branch (p, q | X 1:γ-1 ) -D Branch (q, p | X 1:γ-1 ) D Branch (p, q | X 1:γ-1 ) • (p(X 1:γ ) -q(X 1:γ )) = p(X 1:γ ) -q(X 1:γ ) (A.17) xγ P X 1:γ-1 xγ is sampled, xγ is rejected, X 1:γ-1 is accepted, x γ is resampled + (A.18) X ′ γ-1:γ P X 1:γ-2 Xγ-1:γ is sampled, Xγ-1:γ is rejected, X 1:γ-2 is accepted, X γ-1:γ is resampled (A.19) = xγ q(X 1:γ-1 xγ ) draft probability • (1 -h γ ) reject backwards at τ + 1 = γ • h γ accept X1:γ-1 • P res (x t ) resample at τ + 1 = γ + (A.20) xγ-1 xγ q(X 1:γ-2 xγ-1 xγ ) draft probability • (1 -h γ )(1 -h γ-1 ) reject backwards at τ + 1 = γ -1 • h γ-2 accept X1:γ-1 • P res (x γ-1 )P res (x γ ) resample at τ + 1 = γ (A.21)
From Definition 2 that the excess probability mass that triggers resampling
D Branch (q, p | X 1:γ-1 ) = xγ q(X 1:γ-1 xγ )(1 -h γ ). Then we have = D Branch (q, p|X 1:γ-1 ) • 1 • p(X 1:γ ) -q(X 1:γ ) D Branch (p, q|X 1:γ-1 ) + (A.22) xγ-1 D Branch (q, p|X 1:γ-2 xγ-1 )(1 - D Branch (p, q|X 1:γ-2 xγ-1 ) D Branch (q, p|X 1:γ-2 xγ-1 ) )P res (x γ-1 )P res (x γ ) (A.23)
From Definition 3 and Theorem 4, we know that xγ-1 D Branch (q, p|X 1:γ-2 xγ-1 ) -D Branch (p, q|X 1:γ-2 xγ-1 ) = D Branch (q, p|X 1:γ-2 ). Then we have
= D Branch (q, p|X 1:γ-1 ) D Branch (p, q|X 1:γ-1 ) • (p(X 1:γ ) -q(X 1:γ ))+ (A.24) D Branch (q, p|X 1:γ-2 ) • p(X 1:γ-1 ) -q(X 1:γ-1 ) D Branch (p, q|X 1:γ-2 ) • p(X 1:γ ) -q(X 1:γ ) D Branch (p, q|X 1:γ-1 ) (A.25)
We know from Definition 3 and Theorem 2 that p(X 1:γ ) -q(X 1:γ = D Branch (p, q|X 1:γ-1 ) -D Branch q, p|X 1:γ-1 . Then we have
= D Branch (q, p|X 1:γ-1 ) D Branch (p, q|X 1:γ-1 ) • (p(X 1:γ ) -q(X 1:γ ))+ (A.26) D Branch (p, q | X 1:γ-1 ) -D Branch (q, p | X 1:γ-1 ) D Branch (p, q | X 1:γ-1 ) • (p(X 1:γ ) -q(X 1:γ )) (A.27) = p(X 1:γ ) -q(X 1:γ ) (A.28) B.2 GENERAL PROOF
Lemma 3 (Rejection-Resampling Sum Reduction (Tokenwise)). Let 0 < m < γ be such that the acceptance ratios satisfy: r(x γ ) > 1, r(x γ-1 ) > 1, . . . , r(x γ-m+1 ) > 1, r(x γ-m ) ≤ 1.
(A.29) Then, the total probability of obtaining the output via resampling over the last m positions is:
m-1 i=0 P (x γ-i is rejected) i j=0 P (X 1:γ-j is resampled) = p(X 1:γ ) -q(X 1:γ ).
(A.30)
Proof. We begin by defining auxiliary quantities to simplify the notation. For i = 0, 1, . . . , m, let
∆ + i := D Branch (q, p | X 1:γ-i ), ∆ - i := D Branch (p, q | X 1:γ-i ), (A.31)
where ∆ - i quantifies the probability mass to be corrected due to overestimation by q, and ∆ + i represents the mass available to be allocated from alternate paths.
Define also the recursive product term:
P i := i j=0 ∆ + j -∆ - j ∆ + j+1 , for 0 ≤ i ≤ m -1. (A.32)
Using these, the rejection-resample contribution becomes:
m-1 i=1 P (x γ-i is rejected) i j=0 P (X 1:γ-j is resampled) = m-1 i=1 ∆ - i P i + (∆ + m-1 -∆ - m-1 )P m-1 . (A.33)
Now observe the recurrence:
∆ + k+1 P k+1 = (∆ + k -∆ - k )P k , (A.34) which implies: (∆ + k -∆ - k )P k = ∆ + k+1 P k+1 . (A.35)
We apply this recurrence in reverse to simplify equation (1) by telescoping the sum:
m-1 i=1 ∆ - i P i + (∆ + m-1 -∆ - m-1 )P m-1 = m-2 i=1 ∆ - i P i + ∆ + m-1 P m-1 = m-3 i=1 ∆ - i P i + ∆ + m-2 P m-2 . . . = ∆ + 1 P 1 = ∆ + 0 -∆ - 0 = p(X 1:γ ) -q(X 1:γ ), (A.36)
where the final equality follows from the definition:
∆ + 0 -∆ - 0 = D Branch (q, p | X 1:γ ) -D Branch (p, q | X 1:γ ) = p(X 1:γ ) -q(X 1:γ ). (A.37)
This completes the proof.
this section cite: ['b13', 'b13', 'b13']

Section: Lemma 4 (No Resampling of Earlier Prefixes (Tokenwise)).
Let X 1:γ = [x 1 , x 2 , . . . , x γ ] be a token block, and suppose that for some index m, the acceptance ratios satisfy:
r(x γ ) > 1, r(x γ-1 ) > 1, . . . , r(x γ-m+1 ) > 1, r(x γ-m ) ≤ 1. (A.38)
Then for all t ≤ γ -m, the resampling probability satisfies:
P (X 1:t is resampled) = 0. (A.39)
Proof. We use the resampling probability formula:
P res (X 1:t ) = max {p(X 1:t ) -q(X 1:t ), 0} max {D Branch (p, q | X 1:t ), D Branch (q, p | X 1:t )} . (A.40)
At position t = γ -m, we are given that the acceptance probability
r(x γ-m ) = min 1, p(X 1:γ-m ) q(X 1:γ-m ) ≤ 1, (A.41) implying p(X 1:γ-m ) < q(X 1:γ-m ). Therefore, p(X 1:γ-m ) -q(X 1:γ-m ) ≤ 0, (A.42)
and hence:
P res (X 1:γ-m ) = 0. (A.43)
This completes the proof.
Theorem 5 (Lossless).
P (yield X 1:γ ) = p(X 1:γ ). (A.44)
Proof. The total probability is the sum of the acceptance and resampling paths. We analyze two cases based on the relative probabilities.
Case 1: p(X 1:γ ) < q(X 1:γ ) In this case, the acceptance probability for the draft is p(X1:γ ) q(X1:γ ) . The probability of generating X 1:γ via resampling is 0, as there is no probability deficit to recover.
P (yield X 1:γ ) = P (X 1:γ is accepted) + P (X 1:γ is resampled) = q(X 1:γ ) • p(X 1:γ ) q(X 1:γ ) + 0 = p(X 1:γ ). (A.45)
Case 2: p(X 1:γ ) ≥ q(X 1:γ ) Here, the acceptance probability for the draft is 1. The resampling path must compensate for the probability deficit. Per lemma 3 and lemma 4, the total probability of all relevant resampling paths is exactly p(X 1:γ ) -q(X 1:γ ).
P (yield X 1:γ ) = P (X 1:γ is accepted) + P (X 1:γ is resampled) = q(X 1:γ ) • 1 + p(X 1:γ ) -q(X 1:γ ) = p(X 1:γ ). (A.46)
These two cases cover all probability events. In both cases, the total probability correctly recovers p(X 1:γ ), proving the method is lossless.
this section cite: []

Section: C LOSSLESS OF HIERARCHICAL SPECULATIVE DECODING C.1 ILLUSTRATIVE EXAMPLE
Let p(•) be the target and q(•) the draft. For a prefix
X 1:t , r(X 1:t ) := p(X 1:t ) q(X 1:t ) , r(X a+1:b | X 1:a ) := p(X a+1:b | X 1:a ) q(X a+1:b | X 1:a ) , so r(X 1:b ) = r(X 1:a ) r(X a+1:b | X 1:a ).
Let m be the last (largest) index < γ at which the running maximum of r(X 1:t ) is attained and exceeds 1; let n < m be the previous such index (two-peak case).
Lemma 6 (Property of r * (X 1:m * l ) at unique capping indices). Let m * l-1 and m * l be two consecutive unique capping indices, we have
r * (X 1:m * l ) = r(X m * l-1 +1:m * l ) > 1 .
We now define the acceptance and resampling probability masses:
Definition 8 (Accepted Probability Mass). The probability mass for accepting the full sequence X 1:γ is:
P (X 1:γ is accepted) = min(1, r * (X 1:γ )) q(X 1:γ ), (A.50)
Definition 9 (Resampling Probability Mass). Let X 1:γ be a full sequence of length γ, and let M * = (m * 1 , m * 2 , . . . , m * L ) be its Sequence of Unique Capping Indices. The total probability mass under the draft q and target p of generating this sequence can be decomposed as:
Total Generation Probability P X 1:γ is generated = P X 1:γ is accepted + P X 1:γ is resampled = min 1, r * (X 1:γ ) q(X 1:γ ) + L l=1 max 0, r(X m * l-1 +1:m * l ) -1 q(X 1:m * l ) p(X m * l +1:γ | X 1:m * l ) + max 0, r * (X 1:γ ) -1 q(X 1:γ-1 ) p(x γ | X 1:γ-1 ) (A.51)
We now establish the key lemma that characterizes the resampling probability mass:
Lemma 7 (Hierarchical Resampling Probability Mass). The total generation probability can be decomposed into acceptance and resampling masses as stated in Definition 9. Only unique capping indices contribute to resampling mass, and the explicit form for the resampling mass at each unique capping index is:
P X 1:m * l is resampled p X m * l +1:γ | X 1:m * l = max 0, r(X m * l-1 +1:m * l ) -1 q(X 1:m * l ) p(X m * l +1:γ | X 1:m * l ) (A.52)
To prove the lossless property, we introduce the segmented probability function:
Definition 10 (Segmented Probability Function). For each l ∈ {1, . . . , L}, we define the segmented probability function F l as:
F l = q X 1:m * l p X m * l +1:γ | X 1:m * l = m * l i=1 q(x i | X 1:i-1 ) γ i=m * l +1 p(x i | X 1:i-1 ) , (A.53)
This function represents a hybrid probability measure that uses the draft distribution q up to position m * l and the target distribution p for the remaining positions, where X 1:0 is equal to the prefix.
We establish the telescoping property of resampling mass:
Lemma 8 (Telescoping of Resampling Mass). For each l ∈ {1, . . . , L}, the mass of the resampling at the unique capping index m * l can be expressed as:
P X 1:m * l is resampled = F l-1 -F l . (A.54)
Proof. we need to show that the resampling mass at the unique capping index m(l) equals F l-1 -F l .
1. EXPRESS F l-1 IN TERMS OF F l . We have
P X 1:m * l is resampled = r(X m * l-1 +1:m * l ) -1 q(X 1:m * l ) p(X m * l +1:γ | X 1:m * l ) First note q X 1:m * l+1 = q X 1:m * l q X m * l +1:m * l+1 | X 1:m * l , and p X m * l +1:m * l+1 | X 1:m * l = r X m * l +1:m * l+1 q X m * l +1:m * l+1 | X 1:m * l . Hence F l-1 = q X 1:m * l p X m * l +1:γ | X 1:m * l = q X 1:m * l p X m * l +1:m * l+1 | X 1:m * l p X m * l+1 +1:γ | X 1:m * l+1 = q X 1:m * l r(X m * l +1:m * l+1 ) q(X m * l +1:m * l+1 | X 1:m * l ) p X m * l+1 +1:γ | X 1:m * l+1 = r X m * l +1:m * l+1 q(X 1:m * l ) q(X m * l +1:m * l+1 | X 1:m * l ) p X m * l+1 +1:γ | X 1:m * l+1 = r X m * l +1:m * l+1 q X 1:m * l+1 p X m * l+1 +1:γ | X 1:m * l+1 = r X m * l +1:m * l+1 F l .
this section cite: []

Section: COMPUTE THE DIFFERENCE F
l-1 -F l . F l-1 -F l = r(X m * l +1:m * l+1 ) F l -F l = r(X m * l +1:m * l+1 ) -1 F l = r(X m * l +1:m * l+1 ) -1 q X 1:m * l+1 p X m * l+1 +1:γ | X 1:m * l+1 = r(X m * l-1 +1:m * l ) -1 q(X 1:m * l ) p(X m * l +1:γ | X 1:m * l ).
This completes the proof that the resampling mass at segment l equals F l-1 -F l .
Theorem 6 (Lossless Recovery). Under the prefix-adaptive speculative decoding scheme, the total probability of generating any sequence X 1:γ equals the target distribution probability:
P X 1:γ is generated = p(X 1:γ ). (A.55)
Proof. From Lemma 7, we have the total generation probability decomposition:
P X 1:γ is generated = P X 1:γ is accepted + P X 1:γ is resampled + P x γ is resampled = min 1, r * (X 1:γ ) q(X 1:γ ) + L l=1 max 0, r(X m * l-1 +1:m * l ) -1 q(X 1:m * l ) p(X m * l +1:γ | X 1:m * l ) + max 0, r * (X 1:γ ) -1 q(X 1:γ-1 ) p(x γ | X 1:γ-1 ) (A.56)
From Lemma 8, we know that for each l ∈ {1, . . . , L}:
F l-1 -F l = r(X m * l-1 +1:m * l ) -1 q(X 1:m * l ) p(X m * l +1:γ | X 1:m * l ) (A.57)
Therefore, we can rewrite the generation probability as:
P X 1:γ is generated = min 1, r * (X 1:γ ) q(X 1:γ ) + L l=1 (F l-1 -F l ) + max 0, r * (X 1:γ ) -1 q(X 1:γ-1 ) p(x γ | X 1:γ-1 ) (A.58)
Published as a conference paper at ICLR 2026 Lemma 10 (Tail Expectation). Let X be a non-negative random variable with values in {na : n = 0, 1, 2, . . . } for some a > 0. Then:
E[X] = a ∞ k=1
Pr(X ≥ k).
(A.73)
Proof. Start with the right-hand side:
a ∞ k=1 Pr(X ≥ ka) = a ∞ k=1 ℓ≥k Pr(X = ℓa) = a ∞ ℓ=1 Pr(X = ℓa) ℓ k=1 1 = ∞ ℓ=1 ℓa • Pr(X = ℓa) = E[X].(
A.74) D.1 EXPECTED TOKEN LENGTH DERIVATION TOKEN WISE SPECULATIVE DECODING Referring to Block-wise Verification Sun et al. (2024), the authors prove that it achieves a longer expected token length than the token-wise verification Leviathan et al. (2023) (see Appendix B.2 in Sun et al. (2024)). HIERARCHICAL SPECULATIVE DECODING Let η 1 , . . . , η γ ∼ U(0, 1) be the random draws used in verification. The accepted length is defined as: τ := max {i ≤ γ : η i ≤ h i } , (A.75) where h i is the acceptance probability at step i. By the tail-sum identity:
E[τ ] = γ i=1
Pr(τ ≥ i).
(A.76)
If we define the event S i := {η i ≤ h i }, and assume independence of the draws, then:
Pr(τ ≥ i) = 1 - γ k=i (1 -h k ). (A.77)
Substituting into Equation (A.76), we obtain:
E[τ ] = γ i=1 1 - γ k=i (1 -h k ) . (A.78) BLOCKWISE VERIFICATION
In Algorithm 2 (blockwise decoding), the decoding continues even if some η i > h block i ; the resampling happens only at the end. Therefore, the token count τ still satisfies the same form.
Let h block i be the acceptance probability at step i computed via blockwise rules, and define events:
S i := {η i ≤ h block i }, so Pr(S i ) = 1 -h block i .
(A.79)
We then have:
Pr(τ ≥ i) = 1 - γ k=i (1 -h block k ), (A.80)
and hence the expected number of accepted tokens under blockwise decoding is:
E[τ ] block = γ i=1 1 - γ k=i (1 -h block k ) (A.81) D.2 TOKEN LENGTH COMPARISON
We re-express the acceptance probability to compare token length between block-wise speculative decoding and our method ( Equation ( 19)). This yields a more precise comparison via the directional divergence expressions Equation ( 17) and Equation ( 18).
this section cite: []

Section: Capped Branch Divergence Difference
The difference of capped branch divergence is calculated as:
D * Branch (p, q | X 1:t ) -D * Branch (q, p | X 1:t ) = xt+1 (r * (X 1:t+1 ) -1) q(X 1:t ) = xt+1 min{r(X 0:m(t+1) ), 1}r(X m(t+1)+1:t+1 ) -1 q(X 0:m(t+1) )q(X m(t+1)+1:t+1 ) = xt+1 r(X m(X1:t+1)+1:t+1 ) -1 q(X 1:t+1 ) (A.82)
Branch Acceptance Probability Combine equations (A.82), the acceptance ratio of hierarchical speculative decoding is:
h branch t = D * Branch (p, q | X 1:t ) D * Branch (q, p | X 1:t ) = D * Branch (p, q | X 1:t ) D * Branch (p, q | X 1:t ) + (1 -r(X m(X1:t+1)+1:t+1 ))q(X 1:t+1 ) = [r(X m(X1:t+1)+1:t+1 ) -1] + [r(X m(X1:t+1)+1:t+1 ) -1] + + (1 -r(X m(X1:t+1)+1:t+1 )) (A.83)
where [a] + is equal to max{a, 0} Blockwise Acceptance Ratio Algorithm 2 (blockwise decoding), blockwise keeps an internal clamp p t = min{p t-1 r(x t |X 1:t-1 ), 1}, which could be simplified based on Suffix-minimum characterization of p t Lemma 11 (Suffix-minimum characterization of p t ). Let {r i } ∞ i=1 ⊆ [0, ∞) and define the sequence {p t } t≥0 recursively by
p 0 = 1, p t = min p t-1 r t , 1 , t ≥ 1. (A.84)
Then for every t ≥ 0
p t = min 0≤s≤t t i=s+1 r i ,(
with the empty product for s = t equal to 1).
(A.85)
Equivalently, p t = min 1, r t , r t-1 r t , . . . , r 1 r 2 • • • r t . (A.86)
Proof. We prove (A.85) by induction on t.
Base case (t = 0). For t = 0 the right-hand side becomes min 0≤s≤0 (empty product) = 1 = p 0 , (A.87)
Listing 1 Tokenwise Speculative Decoding (SD) SD.py 1 import torch 2 3 def SD(candidate_input_ids, candidate_logits, new_logits): 4 """ 5 Args: 6 candidate_input_ids (Tensor): Token IDs from the draft model. Shape: [batch_size, ↩→ seq_len] 7 candidate_logits (Tensor): Logits from the draft model. Shape: [batch_size, seq_len, ↩→ vocab_size] 8 new_logits (Tensor): Logits from the target model. Shape: [batch_size, seq_len, ↩→ vocab_size] 9 Returns: 10 n_matches (int): Number of accepted tokens from the draft model. 11 valid_tokens (Tensor): Accepted token prefix with one new token sampled. Shape: [ ↩→ batch_size, n_matches+1] 12 """ 13 14 # Convert logits to probabilities 15 q = candidate_logits.softmax(dim=-1) 16 p = new_logits.softmax(dim=-1) 17 18 candidate_length = candidate_logits.shape[1] 19 new_candidate_input_ids = candidate_input_ids[:, -candidate_length:] 20 21 # Extract token-wise probabilities for the candidate tokens 22 q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1) 23 p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1) 24 25 probability_ratio = p_i / q_i 26 is_accepted = torch.rand_like(probability_ratio) <= probability_ratio 27 28 # assuming batch size = 1 29 n_matches = ((∼is_accepted).cumsum(dim=-1) < 1).sum() # this is 'n' in algorithm 1 30 31 # Next token selection: if there is a rejection, adopt the resampling distribution. 32 if n_matches < candidate_length: 33 p_n_plus_1 = p[:, n_matches, :] 34 q_n_plus_1 = q[:, n_matches, :] 35 p_prime = torch.clamp((p_n_plus_1 -q_n_plus_1), min=0) 36 p_prime.div_(p_prime.sum()) 37 else: 38 p_prime = p[:, n_matches, :] 39 40 # Ensure we don't generate beyond max_len or an EOS token. 41 if is_done_candidate[0] and n_matches == candidate_length: 42 43 # Output length is assumed to be 'n_matches + 1'. Since we won't generate another ↩→ token with the target model 44 # due to acceptance on EOS we fix 'n_matches' 45 n_matches -= 1 46 valid_tokens = candidate_input_ids[:, -candidate_length:] 47 48 else: 49 # Next token selection: if there is a rejection, adjust the distribution from the main ↩→ model before sampling. 50 # The selected tokens include the matches (if any) plus the next sampled tokens 51 if n_matches > 0: 52 if n_matches < candidate_length: 53 valid_tokens = candidate_input_ids[:, -candidate_length:n_matches -↩→ candidate_length] 54 if not stop(valid_tokens, scores=None): 55 t = torch.multinomial(p_prime, num_samples=1) 56 valid_tokens = torch.cat( 57 (valid_tokens, t), dim=-1) 58 else: 59 n_matches = n_matches-1 60 else: 61 valid_tokens = candidate_input_ids[:, -candidate_length:] 62 if not stop(valid_tokens, scores=None): 63 t = torch.multinomial(p_prime, num_samples=1) 64 valid_tokens = torch.cat( 65 (valid_tokens, t), dim=-1) 66 else: 67 n_matches = n_matches -1 68 else: 69 t = torch.multinomial(p_prime, num_samples=1) 70 valid_tokens = t 71 72 return valid_tokens, n_matches Listing 2 Hierarchical Speculative Decoding (HSD) HSD.py 1 import torch 2 3 def HSD(candidate_input_ids, candidate_logits, new_logits): 4 """ 5 Args: 6 candidate_input_ids (Tensor): Token IDs from the draft model. Shape: [batch_size, ↩→ seq_len] 7 candidate_logits (Tensor): Logits from the draft model. Shape: [batch_size, seq_len, ↩→ vocab_size] 8 new_logits (Tensor): Logits from the target model. Shape: [batch_size, seq_len, ↩→ vocab_size] 9 Returns: 10 n_matches (int): Number of accepted tokens from the draft model. 11 valid_tokens (Tensor): Accepted token prefix with one new token sampled. Shape: [ ↩→ batch_size, n_matches+1] 12 """ 13 14 # Convert logits to probabilities 15 q = candidate_logits.softmax(dim=-1) 16 p = new_logits.softmax(dim=-1) 17 candidate_length = candidate_logits.shape[1] 18 new_candidate_input_ids = candidate_input_ids[:, -candidate_length:] 19 20 # Extract token-wise probabilities for the candidate tokens 21 q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1) 22 p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1) 23 24 # Compute cumulative joint probabilities for draft and target model 25 q_prev = torch.roll(q_i, shifts=1, dims=1) 26 q_prev[:, 0] = 1.0 27 q_cumprod = torch.exp(torch.log(q_prev).cumsum(dim=1)).unsqueeze(-1) 28 q_next = q_cumprod * q[:, :candidate_length] 29 p_prev = torch.roll(p_i, shifts=1, dims=1) 30 p_prev[:, 0] = 1.0 31 p_cumprod = torch.exp(torch.log(p_prev).cumsum(dim=1)).unsqueeze(-1) 32 33 # Constrain p_cumprod with q_cumprod for computing the capped resampling distribution 34 ratio = p_cumprod / q_cumprod 35 previous_max = 1 36 new_p_previous = torch.ones_like(p_cumprod).to(p_cumprod.device) 37 for k in range(candidate_length): 38 if ratio[:, k] > previous_max: 39 previous_max = ratio[:, k] 40 new_p_previous[:, k] = p_cumprod[:, k] / previous_max 41 p_next = new_p_previous * p[:, :candidate_length] 42 43 # Construct resampling distribution p' 44 diffs = p_next -q_next 45 p_plus = torch.clamp(diffs, min=0.0) 46 p_minus = torch.clamp(-diffs, min=0.0) 47 p_primes = p_plus / torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, ↩→ keepdim=True)) 48 49 # Step-back probability: reject prefix with 1 -mass of p' 50 step_back_probs = 1 -p_primes.sum(dim=-1) 51 step_back = torch.rand_like(step_back_probs) < step_back_probs 52 53 # Find first position to stop (from the end) 54 if step_back.all(): 55 stop_positions = 0 56 else: 57 stop_positions = candidate_length -n_matches -1 -torch.flip(∼step_back, [-1]).max ↩→ (-1, keepdim=True)[1] 58 59 # Mask to decide which tokens are accepted 60 select = torch.zeros_like(step_back).to(step_back.device) 61 62 # apply cumprod on the ratio instead of the raw probabilities to avoid underflow 63 probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1) 64 is_accepted = torch.rand_like(probability_ratio) <= probability_ratio 65 66 # only decide to accept or not at the last position based on the joint probability ratio 67 # assign 0 to all positions when the full draft is rejected, otherwise assign 1 to the ↩→ rest of the positions 68 select[torch.arange(p_primes.shape[0]), stop_positions] = ∼is_accepted[:, -1:] 69 is_accepted = 1 -torch.cumsum(select, dim=-1) 70 71 #### assume batch_size=1 for the current implementation 72 n_matches = is_accepted.sum().item() Algorithm 3 Hierarchical Speculative Sampling with Recursive Rejection Sampling Require: Draft tokens: X k 1:t = {x k 1 , ..., x k γ } K k=1 ; Target probabilities for all draft tokens: {p(•), ..., p(•|X k 1:γ )} K k=1 ; Draft probabilities for all draft tokens: {q(•), ..., q(
•|X k 1:γ )} K k=1 ; 1: Initialize τ = 0; 2: Initialize {x 1 i } γ 1 ; 3: for k in 1 : K do 4: if X 1:τ = X k 1:τ then 5: for j in τ + 1 : γ do 6: Set x j = x k j #select draft X k τ +1:γ for verification 7:
end for
8: 9: for t in γ : τ + 1 do 10: Compute acceptance probability h t from Equation (19) based on the corresponding probabilities for the draft tokens: {x τ +1 , ..., x γ } 11: Sample η t ∼ U (0, 1) 12: if h t ≥ η t then 13: Set τ = t 14: break 15: else 16: Set τ = t -1 17: continue 18: end if 19: end for 20: else 21: continue #skip draft X k 1:γ due to prefix mismatch 22: end if 23: 24: if τ = γ then 25: Sample token from p(•|X 1:γ ) #accept the entire selected draft and sample a bonus token 26: break 27: else 28:
Compute P * res (• | X 1:τ ); 29: Set p(•|X 1:τ ) = P * res (• | X 1:τ ); #set P * res (• | X 1:τ ) as new target distribution Set r(•|X 1:τ ) = P * res (•|X1:τ ) q(x|X1:τ ) #set r(• | X 1:τ ) as new probability ratio 30:
end if
31: 32: end for Sample token from P * res (• | X 1:τ ) Ensure: [X 1:τ , token] Published as a conference paper at ICLR 2026 H COMPUTATION EFFICIENCY
We begin by noting that the computational cost of the verification stage in HSD is effectively as efficient as that of tokenwise verification in practice.
While there are minor differences in the computational cost of verification-whether using any of the three verification methods-these differences are insignificant in practice compared to the reduction in target model forward passes. Indeed, block efficiency (or equivalently, the acceptance rate) remains the most meaningful metric for evaluating performance. A detailed complexity analysis is provided in the revised version below: Both HSD (Eqs. 17-20 in our paper) and blockwise verification (Eqs. 4-5 in Sun et al. (2024)) require summing over the vocabulary to compute the acceptance probability at each position. Therefore, HSD introduces no theoretical overhead compared to blockwise verification.
Moreover, our implementation is more efficient than that of blockwise verification (Appendix A in Sun et al. (2024) ). By leveraging an equivalent masking mechanism, we eliminate the for-loop and compute probabilities for all positions in parallel. This makes HSD nearly as efficient as tokenwise verification. While tokenwise verification only sums over the vocabulary at a rejected position, our HSD computation is fully parallelized via tensor operations across both the vocabulary and the draft length γ, which is typically much smaller than the vocabulary size V.
Importantly, the computational cost of verification is negligible relative to the reduction in target model forward passes, which is the main bottleneck in verification. Below, we compare the verification cost with the forward-pass reduction for a batch size of 1.
Since the vocabulary size V is much larger than the draft length γ, the main cost of HSD arises from computing branch divergences, which requires only 4γV FLOPs: Using a context length L past = 1024, the per-token FLOPs for Qwen2.5 models are:
1. r * (X 1:t ) -1 → γV FLOPs 2. Selecting (r * (X 1:t ) -1 > 0) via the max operator → γV
• Qwen2.5-0.5B: 0.374 GFLOPs • Qwen2.5-72B: 62.915 GFLOPs As shown in Table 1 in our paper, all methods achieve block efficiency larger than 2. Consequently, the cost of HSD verification is negligible relative to the reduction in target model forward passes.
To directly quantify the overhead of the verification stage, we evaluate verification cost on 100 GSM8K problems using GPTQ-quantized 8-bit Qwen2.5-72B-Instruct and Qwen2.5-0.5B-Instruct
this section cite: ['b25', 'b25']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Qwen technical report Year: (2023)
Ref_id:b2 Title: Simple llm inference acceleration framework with multiple decoding heads Year: (2024)
Ref_id:b3 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b4 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b5 Title: Glide with a cape: a low-hassle method to accelerate speculative decoding Year: (2024)
Ref_id:b6 Title: Flatter tokens are more valuable for speculative draft model training Year: (2026)
Ref_id:b7 Title: The lottery ticket hypothesis: Finding sparse, trainable neural networks Year: (2018)
Ref_id:b8 Title: Fast large language model collaborative decoding via speculation Year: (2025)
Ref_id:b9 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b10 Title: Rest: Retrieval-based speculative decoding Year: (2023)
Ref_id:b11 Title: Towards optimal multi-draft speculative decoding Year: (2025)
Ref_id:b12 Title: Speculative decoding with big little decoder Year: (2023)
Ref_id:b13 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b14 Title: Eagle: speculative sampling requires rethinking feature uncertainty Year: (2024)
Ref_id:b15 Title: Accelerating large language model serving with tree-based speculative inference and verification Year: (2024)
Ref_id:b16 Title: Pass: Parallel speculative sampling Year: (2023)
Ref_id:b17 Title: Faster cascades via speculative decoding Year: (2024)
Ref_id:b18 Title: Openai o1 system card Year: (2024)
Ref_id:b19 Title: Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl Year: (2025)
Ref_id:b20 Title: Optimized multitoken joint decoding with auxiliary model for llm inference Year: (2025)
Ref_id:b21 Title: Get to the point: Summarization with pointer-generator networks Year: (2017-07)
Ref_id:b22 Title: Q-bert: Hessian based ultra low precision quantization of bert Year: (2020)
Ref_id:b23 Title: A simple and effective pruning approach for large language models Year: (2023)
Ref_id:b24 Title: Spectr: Fast speculative decoding via optimal transport Year: (2023)
Ref_id:b25 Title: Block verification accelerates speculative decoding Year: (2024)
Ref_id:b26 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b27 Title: Transformers: State-of-the-art natural language processing Year: (2020-10)
Ref_id:b28 Title: Smoothquant: Accurate and efficient post-training quantization for large language models Year: (2023)
Ref_id:b29 Title: Inference with reference: Lossless acceleration of large language models Year: (2023)
Ref_id:b30 Title: Multi-candidate speculative decoding Year: (2024)
Ref_id:b31 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b32 Title: Draft& verify: Lossless large language model acceleration via self-speculative decoding Year: (2024)
Ref_id:b33 Title: Distillspec: Improving speculative decoding via knowledge distillation Year: (2024)
