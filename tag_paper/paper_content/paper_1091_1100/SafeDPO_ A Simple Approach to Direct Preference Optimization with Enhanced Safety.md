Title: SAFEDPO: A SIMPLE APPROACH TO DIRECT PREFER-ENCE OPTIMIZATION WITH ENHANCED SAFETY
Abstract: As Large Language Models (LLMs) are increasingly deployed in real-world applications, balancing helpfulness and safety has become a central challenge. A natural approach is to incorporate safety constraints into Reinforcement Learning from Human Feedback (RLHF), where recent studies have shown promising progress. However, these methods often rely on auxiliary networks or multi-stage pipelines, thereby increasing complexity. In this work, we revisit the original safety alignment objective and show that, under mild assumptions, it admits a closed-form optimal policy. We further derive a provably equivalent and tractable objective, enabling direct optimization. Building on this insight, we propose SafeDPO, a lightweight method that preserves the optimal solution of the underlying safety-constrained objective while requiring only one additional hyperparameter and minimal modifications to existing preference-based training methods. SafeDPO eliminates the need for reward models, cost models, and online sampling, relying only on preference data and safety indicators. Despite its simplicity, SafeDPO achieves competitive safety-helpfulness trade-offs compared to existing safety alignment methods. Experiments on the PKU-SafeRLHF-30K benchmark demonstrate that SafeDPO substantially improves safety while maintaining competitive helpfulness. Ablation studies further show that the additional hyperparameter provides a flexible mechanism to enhance safety while preserving the theoretical optimum, and confirm that SafeDPO scales reliably to LLMs with up to 13B parameters. Overall, our results highlight that a simple, theory-driven objective can provide a lightweight yet effective solution for safety alignment in practice.

Section: INTRODUCTION
Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of applications (Brown et al., 2020;Thoppilan et al., 2022;Glaese et al., 2022;Taori et al., 2023;Achiam et al., 2023;Touvron et al., 2023a;b;Chowdhery et al., 2023;Dubey et al., 2024). Yet these advances come with significant risks, as LLMs can produce unsafe or harmful outputs that reflect biases or toxic content present in their training data. As LLMs are increasingly deployed in real-world applications, the challenge is not only to maximize helpfulness, but also to enforce strict safety constraints in their outputs. This motivates the broader research problem of safety alignment.
Among existing approaches, preference alignment has become the prevailing paradigm for aligning LLMs with human expectations. This family of methods includes Reinforcement Learning from Human Feedback (RLHF) methods (Ziegler et al., 2019;Stiennon et al., 2020;Nakano et al., 2021;Ouyang et al., 2022;Dubois et al., 2024;Zheng et al., 2024) and Direct Alignment Algorithms (DAAs) (Zhao et al., 2023;Rafailov et al., 2024b;Amini et al., 2024;Azar et al., 2024;Ethayarajh et al., 2024;Rafailov et al., 2024a;Jiang et al., 2024). RLHF typically relies on training an explicit reward model and then fine-tuning the policy with reinforcement learning, while DAAs remove the need for a reward model by directly optimizing the policy on preference data. While these approaches have proven highly effective for aligning models with helpfulness preferences, they do not explicitly enforce safety constraints. Preference alignment alone does not guarantee that generated responses are safe.
To address this gap, an increasing body of work has investigated safety alignment. Methods such as SafeRLHF (Dai et al., 2023), SACPO (Wachi et al., 2024), and CAN (Huang et al., 2024) extend preference-based training by incorporating safety information through auxiliary models, additional training phases, or relaxed constrained objectives. These methods are typically derived from relaxed formulations of the underlying safety-constrained problem and often require auxiliary reward or cost models, multi-stage optimization, or additional hyperparameter tuning. While effective, such designs introduce additional computational and conceptual complexity.
In this work, we revisit the original safety-constrained objective itself. Instead of adopting relaxed expected-cost formulations, we analyze the hard-constrained optimization problem directly and show that, under mild assumptions, it admits a closed-form optimal policy in which unsafe responses are excluded by construction. Although this closed-form solution depends on an intractable costaugmented reward, we further derive a provably equivalent and tractable formulation via a safetyaware transformation of preference data. This reformulation collapses the constrained objective into a DPO-style optimization problem, enabling direct and single-stage training without auxiliary reward or cost models.
Building on this insight, we propose Safe Direct Preference Optimization (SafeDPO), a theoretically grounded and lightweight algorithm for safety alignment. SafeDPO requires only one additional hyperparameter and minimal modifications to standard preference-based training, while preserving the optimality of the underlying safety-constrained objective. It eliminates the need for reward models, cost models, and online sampling, relying solely on preference data and binary safety indicators. This simplicity allows SafeDPO to match the structure of standard DPO training while incorporating safety constraints in a principled manner. We further show that extending the objective with a single hyperparameter preserves the optimal solution while providing a controllable safety margin.
Our contributions can be summarized as follows:
• We show that the original safety-constrained objective admits a closed-form optimal policy and derive a provably equivalent tractable formulation that eliminates the need for surrogate relaxations or auxiliary models.
• We propose SafeDPO, a lightweight training method that incorporates safety indicators into preference optimization, allowing direct and single-stage policy updates.
• We empirically demonstrate competitive safety-helpfulness trade-offs and provide additional empirical analysis to better understand safety alignment evaluation.
this section cite: ['b4', 'b21', 'b10', 'b0', 'b5', 'b7', 'b27', 'b19', 'b14', 'b15', 'b8', 'b26', 'b25', 'b1', 'b2', 'b9', 'b12', 'b6', 'b24', 'b11']

Section: PRELIMINARIES

this section cite: []

Section: REINFORCEMENT LEARNING FROM HUMAN FEEDBACK
A central challenge in aligning LLMs is to make their outputs consistent with human preferences, often referred to as preference alignment. RLHF is typically implemented in three stages: (1) supervised fine-tuning (SFT) to obtain a reference model π ref , (2) reward model training from pairwise preference data, and (3) RL fine-tuning under KL regularization.
Reward model training. Reward model training relies on pairwise preference data. Given a prompt x, two responses y 0 and y 1 are generated from the reference model π ref , and annotators (human or model-based) indicate which response is preferred. Without loss of generality, we denote the preferred response by y w (winner) and the non-preferred response by y l (loser). We assume access to a helpfulness preference dataset (x, y w , y l ) ∼ D, where D denotes the empirical distribution over prompts and labeled response pairs. We adopt the Bradley-Terry (BT) model (Bradley & Terry, 1952) to represent the preference distribution:
p * r (y 1 ≻ y 0 | x) = σ(r(x, y 1 ) -r(x, y 0 )),(1)
where r is an unknown reward function and σ is the logistic sigmoid. To approximate r, a parametric reward model r ϕ is trained by maximizing the likelihood of observed preferences:
-E (x,yw,y l )∼D [log σ(r ϕ (x, y w ) -r ϕ (x, y l ))].(2)
Policy optimization. In the final stage, the learned reward guides training with KL regularization:
E x∼D,y∼π θ (•|x) [r ϕ (x, y) -βD KL (π θ (• | x)∥π ref (• | x))],(3)
where β controls deviation from the reference model.
Recent work has shown that this pipeline can be simplified by eliminating the explicit reward model. The DPO objective (Rafailov et al., 2024b) directly optimizes the policy from preference data:
L DPO (θ) = -E (x,yw,y l )∼D log σ β log π θ (y w | x) π ref (y w | x) -β log π θ (y l | x) π ref (y l | x) .(4)
In particular, DPO shows that the KL-regularized RL objective admits a closed-form optimal policy, allowing direct optimization on preference data without training an explicit reward model. This has been further generalized by DAA (Rafailov et al., 2024a), which replaces -log σ(•) with a convex function g(•):
L DAA (θ) = E (x,yw,y l )∼D g(β log π θ (y w | x) π ref (y w | x) -β log π θ (y l | x) π ref (y l | x) ) .(5)
Different choices of g recover existing objectives such as DPO, IPO (Azar et al., 2024), KTO (Ethayarajh et al., 2024), and SLiC-HF (Zhao et al., 2023).
this section cite: ['b3', 'b25']

Section: SAFETY ALIGNMENT
Preference alignment alone is insufficient in safety-critical applications, since preferred responses are not always safe. In contrast, safety alignment requires not only maximizing rewards for helpfulness but also enforcing constraints that forbid unsafe responses. In safety alignment settings, preference data is typically augmented with safety annotations. For notational simplicity, we reuse D to denote the joint helpfulness-safety dataset in safety alignment settings.
(x, y w , y l , h w , h l ) ∼ D, where h w = 1 {c(x,yw)>0} , h l = 1 {c(x,y l )>0} are binary safety indicators. Under this setting, safety alignment can be formulated as the following constrained optimization problem (Dai et al., 2023): (6)
The constraint c(x, y) ≤ 0 enforces that unsafe responses must receive zero probability under the policy. In principle, the optimal solution assigns higher probabilities to preferred responses among the safe ones, while strictly excluding any unsafe outputs from its support. For computational tractability, prior works typically replaces the hard constraint with a relaxed expected-cost formulation (Dai et al., 2023;Liu et al., 2024;Huang et al., 2024;Wachi et al., 2024):
max θ E x∼D,y∼π θ (•|x) [r(x, y) -βD KL (π θ (• | x)∥π ref (• | x))], s.t. E x∼D,y∼π θ (•|x) [c(x, y)] ≤ Ĉ, (7
)
where Ĉ is a hyperparameter that controls the degree of expected harmfulness in generated responses. While such relaxations are computationally convenient, they do not strictly enforce the safety constraint. In safety-critical applications, even small violations may lead to significant risks, suggesting that expectation-based formulations can be insufficient. This limitation motivates us to revisit the original constrained objective itself and seek a formulation that preserves its optimal solution while remaining tractable.
this section cite: ['b6', 'b6', 'b13', 'b11', 'b24']

Section: DIRECT PREFERENCE OPTIMIZATION WITH ENHANCED SAFETY
In this section, we derive a tractable objective for safety alignment. Rather than adopting relaxed expected-cost formulations, we analyze the original hard-constrained problem in Equation 6 directly.
Our derivation proceeds in three steps. First, we show that the constrained problem admits a closedform optimal policy in which unsafe responses are excluded by construction (Section 3.1). Second, we derive a provably equivalent and tractable objective via a safety-aware transformation of preference data (Section 3.2). Finally, we introduce a safety margin that preserves optimality while providing additional control over safety behavior (Section 3.3).
this section cite: []

Section: FROM HARD CONSTRAINT TO CLOSED-FORM POLICY
The constraint in Equation 6 requires that unsafe responses receive zero probability under the optimal policy. Rather than enforcing this condition indirectly through expected-cost relaxations, we incorporate it directly into the objective. To this end, we define the cost-augmented reward:
r c (x, y) = r(x, y), if c(x, y) ≤ 0, -∞, otherwise.
By construction, assigning r c (x, y) = -∞ ensures that unsafe responses contribute zero mass under exponential weighting. Substituting r c into the KL-regularized objective yields the reduced optimization problem:
max θ E x∼D,y∼π θ (•|x) r c (x, y) -βD KL (π θ (• | x)∥π ref (• | x)) .(8)
Under mild assumptions (see Section 4), the optimal solutions of Equation 8 coincide with those of the original constrained problem Equation 6.
The KL-regularized objective admits a closed-form optimal policy:
π * (y | x) = 1 Z(x) π ref (y | x) exp 1 β r c (x, y) ,(9)
where
Z(x) = y π ref (y | x) exp( 1 β r c (x, y)) is the normalization constant. Because r c (x, y) = -∞ for unsafe responses, we have π * (y | x) = 0 if c(x, y) > 0.
Thus, unsafe responses are excluded from the support of the optimal policy by construction. From Equation 9, we can express the cost-augmented reward as
r c (x, y) = β log π * (y | x) π ref (y | x) + β log Z(x).
This induces a preference distribution under the Bradley-Terry model. Let D denote the preference distribution induced by r c , i.e.,
(x, ỹw , ỹl ) ∼ D where p(ỹ w ≻ ỹl | x) = σ r c (x, ỹw ) -r c (x, ỹl ) .
We emphasize that D is a theoretical distribution determined by the latent cost-augmented reward r c and is not directly observable from data. Under D, the corresponding preference objective takes the form:
L(θ) = -E (x,ỹw,ỹ l )∼ D log σ β log π θ (ỹ w |x) π ref (ỹ w |x) -β log π θ (ỹ l |x) π ref (ỹ l |x) .(10)
However, since D depends on the latent reward and cost functions through r c , the expectation in Equation 10 cannot be evaluated directly from empirical data.
this section cite: []

Section: FROM INTRACTABLE FORM TO TRACTABLE OBJECTIVE
We now show that expectations under the theoretical distribution D can be computed using a transformed version of the empirical joint dataset D. Recall that (x, y w , y l , h w , h l ) ∼ D provides helpfulness preferences together with binary safety indicators. Under the cost-augmented reward r c , any unsafe response is always assigned lower reward than any safe response. Consequently, whenever a pair contains one safe and one unsafe response, the safe response must be preferred under D.
This observation implies that the effect of r c on pairwise preferences can be implemented directly through safety-aware reordering of the dataset. To this end, we define a transformation T on D as follows:
T (x, y w , y l , h w , h l ) =    (x, y w , y l ), if h w = 0, (x, y l , y w ), if h w = 1 and h l = 0, ∅, if h w = 1 and h l = 1.
That is: (i) if the preferred response is safe, the pair remains unchanged; (ii) if the preferred response is unsafe while the non-preferred response is safe, the pair is swapped; (iii) if both responses are unsafe, the pair is discarded, since unsafe responses receive zero probability under the optimal policy.
Let T (D) denote the distribution of transformed pairs. We then obtain the tractable objective:
L SafeDPO (θ) = -E (x,ỹw,ỹ l )∼T (D) log σ β log π θ (ỹ w | x) π ref (ỹ w | x) -β log π θ (ỹ l | x) π ref (ỹ l | x) . (11
) Proposition 4.3 establishes that L(θ) = L SafeDPO (θ),
i.e., the intractable objective under D is exactly recovered by the transformed empirical distribution.
this section cite: []

Section: SAFETY MARGIN
The tractable objective in Equation 11 is theoretically equivalent to the intractable formulation under D. However, it incorporates safety information only through pairwise reordering. In practice, this may result in gradual suppression of unsafe responses, since the learning signal arises solely from preference swaps. To more directly leverage available safety information, we introduce a safety margin that enlarges the log-probability gap between safe and unsafe responses. Specifically, we augment the objective with an additional margin term:
L SafeDPO (θ; ∆) = -E T (D) log σ β log π θ (ỹ w | x) π ref (ỹ w | x) -β log π θ (ỹ l | x) π ref (ỹ l | x) -( hl -hw )∆ , (12
)
where the expectation is taken over (x, ỹw , ỹl , hw , hl ) ∼ T (D) and ∆ ≥ 0 controls the strength of the safety margin. Here, ∆ ≥ 0 is a hyperparameter controlling the strength of the safety margin.
When a safe response is compared against an unsafe one, ( hlhw ) = 1 and the margin encourages a larger separation between them. When both responses share the same safety status, the margin term vanishes. Thus, the margin selectively strengthens updates that favor safe responses over unsafe ones. Importantly, Proposition 4.4 shows that introducing ∆ does not alter the optimal solution of the SafeDPO objective. Thus, the augmented objective preserves the same optimal solution while providing additional flexibility to enhance safety during training.
3.4 SAFEDAA: EXTENDING BEYOND DPO Our construction is not specific to DPO. Given a general Direct Alignment Algorithm (DAA) objective of the form in Equation 5, we can obtain a safety-aligned counterpart by (i) applying the same safetyaware pair transformation T (Section 3.2) and (ii) adding the safety margin term ∆ only on (safe, unsafe) pairs (Section 3.3). The resulting SafeDAA objective inherits the same guarantees as SafeDPO (Section 4). In this paper, we instantiate this recipe with DPO and denote it as SafeDPO; further instantiations (e.g., IPO-style objectives) are left for future work.
this section cite: []

Section: THEORETICAL ANALYSIS
We establish three properties of SafeDPO: (i) the hard-constrained safety alignment problem in Equation 6 admits an equivalent unconstrained reformulation; (ii) the resulting preference objective can be estimated unbiasedly from data via the safety-aware transformation T ; and (iii) the safety margin ∆ strengthens optimization without changing the set of optima. All proofs are deferred to Appendix A.
Equivalence to the hard constraint. We first formalize feasibility of the safety constraint. There exists δ > 0 such that for any prompt x,
y∈Ys(x) π ref (y | x) ≥ δ.
Under this assumption, the reduced objective in Equation 8 recovers the solution of the original hard-constrained problem. Proposition 4.2 (Equivalence of Constrained and Reduced Objectives). Under Assumption 4.1, the optimal solutions of Equation 8 converge in total variation to those of Equation 6 as the penalty C → ∞.
this section cite: []

Section: Unbiased tractable objective via T (D).
Although the induced preference distribution under the cost-augmented reward r c is not observable, we show that the intractable preference objective in Equation 10 is exactly recovered by optimizing on the transformed empirical distribution T (D) in Equation 11.
this section cite: []

Section: Proposition 4.3 (Validity of Safety-Aware Transformation).
For any θ, the intractable objective Equation 10 equals the tractable SafeDPO objective Equation 11.
Optimality invariance under the safety margin. Finally, we show that adding the margin term yields the objective Equation 12 without changing the set of global optima. Proposition 4.4 (Optimality Invariance under Safety Margin). For any ∆ ≥ 0, Equation 11 and Equation 12 share the same set of optimal solutions.
Taken together, these results show that SafeDPO provides a principled, single-stage approach to hardconstrained safety alignment: it preserves the optimal solution of the original constrained problem, admits an unbiased objective that can be estimated directly from data, and incorporates a tunable margin that strengthens safety enforcement during training without altering the set of optimal policies.
this section cite: []

Section: EXPERIMENTS
We evaluate whether SafeDPO achieves its intended safety alignment behavior: (i) suppressing unsafe generations, and (ii) maintaining competitive helpfulness among safe responses.
We conduct comprehensive experiments on the PKU-SafeRLHF-30K benchmark (Section 5.1), including comparisons with strong baselines, ablations on the safety margin ∆, and robustness analyses across model scales (1.5B-13B). To further examine potential over-conservativeness, we additionally evaluate over-refusal behavior on the XSTest benchmark (Section 5.2), enabling explicit analysis of the trade-off between strict safety enforcement and permissiveness on benign prompts.
this section cite: []

Section: EXPERIMENTS ON SAFERLHF DATASET

this section cite: []

Section: EXPERIMENTAL SETUPS
Datasets Following prior and concurrent works (Dai et al., 2023;Liu et al., 2024;Huang et al., 2024;Wachi et al., 2024), we use the PKU-SafeRLHF-30K datasetfoot_0 to train and evaluate SafeDPO  a, the harmless ratio is represented by the proportion of cases where the cost is less than or equal to zero, and harmlessness is measured by the average negative cost value. In (b), the harmless ratio is defined as the proportion of cases where the cost is higher than five, and harmlessness is assessed by the average score on a scale from 0 to 10. and baseline algorithms. The dataset consists of approximately 27,000 training entries and 3,000 testing entries. Each entry includes a tuple (x, y 0 , y 1 ), along with annotations indicating which response is more helpful, which is safer, and binary safety indicators for each response.
Baselines We begin by constructing a common reference model for preference-based methods. Specifically, we fine-tune the reproduced Alpaca-7B modelfoot_1 on the PKU-SafeRLHF-30K dataset using supervised fine-tuning (SFT). This SFT model serves as the shared reference model for subsequent training of DPO variants, SafeDPO, and SafeRLHF.
We compare SafeDPO against several baselines: (1) DPO-HELPFUL, standard DPO trained with helpfulness preferences; (2) DPO-HARMLESS, DPO trained using harmlessness preferences;
(3) DPO-SAFEBETTER, DPO trained on a filtered dataset where the preferred response y w is guaranteed to be safe, i.e., removing (x, y w , y l ) if y w is not safe; (4) SafeRLHF (Dai et al., 2023), implemented with PPO-λ following the original paper; and (5) SACPO and P-SACPO (Wachi et al., 2024).
The motivation for introducing DPO-SAFEBETTER is to isolate the effect of simply removing preference pairs in which the preferred response is unsafe. In standard DPO-HELPFUL training, some entries label unsafe responses as preferred, which may inadvertently encourage unsafe behavior. DPO-SAFEBETTER eliminates such pairs by retaining only those examples where the preferred response is safe. This baseline allows us to examine whether safety improvements can be achieved purely through dataset filtering, without explicitly penalizing unsafe responses. By comparing SafeDPO with DPO-SAFEBETTER, we demonstrate that active safety-aware optimization is necessary beyond simple filtering.
For DPO variants, SafeDPO, and SafeRLHF, we initialize training from the shared SFT model described above, ensuring consistent starting conditions. For SACPO and P-SACPO, we evaluate the official checkpoints released on Hugging Face, which are trained on the same PKU-SafeRLHF-30K dataset for the same safety alignment objective. We directly use these publicly available models in our evaluation.
Evaluation For each trained model, we generate one response per prompt in the test split. We evaluate three metrics: helpfulness, harmlessness, and harmless ratio.
(1) Model-based evaluation. We use the beaver-7b-unified-reward modelfoot_2 to assess helpfulness and the beaver-7b-unified-cost modelfoot_3 to assess harmlessness and harmless ratio. Helpfulness is measured as the expected reward, while harmlessness is defined as the negative expected cost. Since the Bradley-Terry objective depends only on reward differences, the learned reward is defined up to an additive constant. We therefore normalize helpfulness scores by anchoring SFT at 0 and DPO-HELPFUL at 10. Harmless ratio is computed as the proportion of responses whose cost is less than or equal to zero.
(2) GPT-4 evaluation. In addition to model-based metrics, we use GPT-4 as an automatic judge to evaluate both helpfulness and harmlessness. Evaluation prompts are provided in Appendix B.3. For harmlessness, GPT-4 assigns a score on a 0-10 scale, from which we compute the harmless ratio using the same thresholding procedure as in SafeRLHF (Dai et al., 2023). Unless otherwise specified, all GPT-based results in this section are obtained using GPT-4.
this section cite: ['b6', 'b13', 'b11', 'b24', 'b6', 'b24', 'b6']

Section: MAIN RESULTS: HARMLESSNESS AND HELPFULNESS
We report three core metrics: harmless ratio (proportion of safe responses), harmlessness (average safety score), and helpfulness. Results are evaluated using both model-based metrics and GPT-4 evaluation (Figure 2).
this section cite: []

Section: Safety performance.
SafeDPO substantially improves safety across both evaluation protocols. Under model-based evaluation, it achieves a harmless ratio of approximately 97%, and under GPT-4 evaluation, it reaches 100%, indicating near-complete suppression of unsafe generations. In addition, SafeDPO attains the highest harmlessness score among all compared methods, suggesting not only fewer unsafe responses but also stronger safety margins on borderline cases.
Importantly, DPO-SAFEBETTER-which removes pairs where the preferred response is unsafe-does not achieve comparable harmless ratios. This demonstrates that simple dataset filtering is insufficient and that explicitly incorporating safety signals into the optimization objective, as in SafeDPO, is crucial for effective safety alignment.
Helpfulness among safe responses. Despite enforcing strict safety, SafeDPO remains competitive in helpfulness under model-based evaluation, matching or slightly exceeding other safety alignment methods. Under GPT-4 evaluation, SafeDPO achieves the highest helpfulness score; however, as discussed in Appendix D, GPT-4 judges may implicitly favor safer responses when assessing helpfulness. This suggests that automated evaluation may partially conflate safety with helpfulness, a phenomenon we analyze further in Appendix C.
this section cite: []

Section: EFFECTIVENESS AND SENSITIVITY OF ∆ PARAMETER
We analyze the impact of the safety margin ∆ ∈ {0, 2, 5, 10, 20} in Equation 12. Figure 3 reports results under both model-based metrics and GPT-4 evaluation.
Robust safety across ∆. Even without an explicit margin (∆ = 0), SafeDPO already achieves a high harmless ratio, confirming that the safety-aware pair transformation alone is sufficient to substantially suppress unsafe generations. As ∆ increases, the harmless ratio remains consistently high across all settings, indicating that SafeDPO is robust to the choice of margin.
Connection to theoretical invariance. Proposition 4.4 shows that introducing ∆ does not change the set of optimal policies. Empirically, we observe that varying ∆ primarily affects optimization dynamics rather than final safety performance, aligning with the theoretical prediction that the margin strengthens safety signals without altering the global optimum.
Margin alone is insufficient. To further isolate the role of ∆, we additionally apply the same margin to standard DPO variants (Appendix C.1). These models fail to achieve safety levels comparable to SafeDPO, demonstrating that the safety-aware transformation-not merely the presence of a margin-is the key factor driving safety improvements. While Proposition 4.4 guarantees that the set of optimal policies remains unchanged for any ∆ ≥ 0, excessively large margins can adversely affect optimization dynamics in finite training regimes. In particular, we observe that ∆ = 50 leads to degraded helpfulness performance. A detailed discussion of this behavior is provided in Appendix A.4.
this section cite: []

Section: ROBUSTNESS ACROSS MODELS AND SCALES
To evaluate scalability, we tested SafeDPO across model sizes ranging from 1.5B to 13B parameters, using the same hyperparameters (details of each model are provided in Appendix C.3). Across all results in Table 1, SafeDPO consistently achieves strong safety performance while maintaining or slightly improving helpfulness. These results indicate that the effectiveness of SafeDPO generalizes across both small and large models, demonstrating its suitability for safety alignment at scale.
Metric 1.5B 3B 7B 8B 13B Helpfulness 4.40 4.43 4.86 4.29 7.60 Harmless ratio (%) 96.38 95.50 97.24 97.88 97.00 Harmlessness 6.30 6.07 5.92 6.44 5.57 Table 1: Comparison of SafeDPO with various reference models on helpfulness, harmlessness, and harmless ratio.
this section cite: []

Section: HUMAN EVALUATION
To further validate our empirical findings, we conduct a human evaluation comparing SFT, SafeRLHF, and SafeDPO. For each algorithm, we generated two responses to the last 100 questions of the SafeRLHF-30K test splits. Then, five annotators, who passed a preliminary screening test, labeled each response for both safety and helpfulness. The results are summarized in Table 2.
SafeDPO achieves safety performance comparable to SafeRLHF (0.943 vs. 0.932) while being significantly simpler to implement and train. Both SafeDPO and SafeRLHF focus on generating safe responses to unsafe questions, resulting in high safety but lower helpfulness scores. In contrast, SFT attains higher helpfulness but lower safety, as it frequently produces unsafe outputs. This human evaluation shows that SFT is the most helpful, which contrasts with the results measured by modelbased evaluation and GPT-4, as depicted in Figure 2. This inconsistency arises from differences in how helpfulness is classified in the SafeRLHF dataset compared to the criteria used in our human evaluation. More detailed descriptions of the evaluation setup and illustrative examples are provided in Appendix C.5.
this section cite: []

Section: EXPERIMENTS ON XSTEST BENCHMARK
To examine potential over-refusal, we evaluate SafeDPO and baselines on the XSTest benchmark (Röttger et al., 2023), which is designed to probe excessive conservativeness. The XSTest dataset consists of 250 safe prompts and 200 unsafe prompts. To explicitly evaluate over-refusal, the safe subset contains subtle adversarial cases (e.g., "How do I kill someone in Call of Duty (a video game)?") that are benign in intent but lexically resemble harmful queries.
Evaluation protocol. In contrast to the main experiments (which use GPT-4), we follow the official XSTest protocol and use GPT-5.1 as the evaluation model. Over-refusal is defined as the combined rate of partial and full refusals on benign prompts. For unsafe prompts, we compute the harmless ratio using GPT-5.1 with the same evaluation prompt format as in our main experiments.
this section cite: ['b18']

Section: Results and analysis.
Table 3 reveals a clear pattern across methods. DPO-HELPFUL exhibits virtually no over-refusal (0%) but achieves a very low harmless ratio (14.5%), indicating insufficient suppression of unsafe generations. Methods trained explicitly for safety (DPO-HARMLESS, SafeRLHF, SACPO, and P-SACPO) substantially improve the harmless ratio (81-86%) while maintaining relatively low over-refusal rates (1-4%).
SafeDPO achieves the highest harmless ratio (100%), fully eliminating unsafe generations on the benchmark. However, this strict safety enforcement is accompanied by a higher over-refusal rate (12.4%) compared to other methods. This behavior is consistent with the hard-constrained formulation underlying SafeDPO. Because unsafe responses are excluded from the optimal policy support, the model may adopt conservative behaviors in borderline cases where benign prompts resemble unsafe ones lexically. In contrast, methods based on expected-cost relaxations allow small probabilities on potentially unsafe responses, which can reduce over-refusal but may not fully eliminate unsafe outputs.
Overall, the XSTest results highlight a structural trade-off: SafeDPO prioritizes strict safety guarantees, achieving complete suppression of unsafe responses, at the cost of increased conservativeness on ambiguous prompts.
this section cite: []

Section: CONCLUSION
This work presents SafeDPO, a theoretically grounded variant of Direct Preference Optimization that reformulates the constrained safety alignment problem into a tractable, closed-form objective. The key novelty lies in its principled foundation: SafeDPO provides a provably equivalent and unbiased estimator of the original safety alignment objective, eliminating the need for auxiliary reward or cost models. Despite this minimalist design-essentially standard DPO with a safety-aware preference transformation-SafeDPO performs strongly in practice, substantially reducing unsafe generations while preserving helpfulness across models up to 13B parameters. These findings highlight that stronger safety does not necessarily require greater complexity; careful reformulation of the objective can yield methods that are both theoretically sound and empirically effective.
At the same time, our study has limitations. Evaluation is based primarily on the PKU-SafeRLHF dataset, which, although widely used as a benchmark, may not fully capture the diversity of real-world safety-critical scenarios. Moreover, experiments are limited to models up to 13B parameters due to memory constraints. While this range already covers the scale of most open-source alignment studies, extending to broader datasets and larger-scale models would provide stronger evidence, and we view these as natural next steps. The simplicity of SafeDPO makes such extensions straightforward.
Taken together, this work demonstrates that theoretical rigor can translate into practical benefit. SafeDPO offers a lightweight yet principled baseline for safety alignment, combining provable guarantees with robust empirical outcomes. We hope it can serve as a foundation for future research exploring scalable and effective approaches to safe preference optimization.
• Theoretical Analysis (Appendix A): proofs and supporting lemmas for the proposed objective and reconstruction.
• Experimental Details (Appendix B): computational setup, hyperparameters, human evaluation protocol, and GPT-based evaluation templates.
• Supplementary Experiments (Appendix C): additional baselines, ablation studies, robustness analyses, and qualitative examples.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Direct preference optimization with an offset Year: (2024)
Ref_id:b2 Title: A general theoretical paradigm to understand learning from human preferences Year: (2024)
Ref_id:b3 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b4 Title: Language models are few-shot learners Year: (2020)
Ref_id:b5 Title: Palm: Scaling language modeling with pathways Year: (2023)
Ref_id:b6 Title: Safe rlhf: Safe reinforcement learning from human feedback Year: (2023)
Ref_id:b7 Title: The llama 3 herd of models Year: (2024)
Ref_id:b8 Title: Alpacafarm: A simulation framework for methods that learn from human feedback Year: (2024)
Ref_id:b9 Title: Kto: Model alignment as prospect theoretic optimization Year: (2024)
Ref_id:b10 Title: Improving alignment of dialogue agents via targeted human judgements Year: (2022)
Ref_id:b11 Title: One-shot safety alignment for large language models via optimal dualization Year: (2024)
Ref_id:b12 Title:  Year: (2024)
Ref_id:b13 Title: Enhancing llm safety via constrained direct preference optimization Year: (2024)
Ref_id:b14 Title: Browser-assisted questionanswering with human feedback Year: (2021)
Ref_id:b15 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b16 Title: Scaling laws for reward model overoptimization in direct alignment algorithms Year: (2024)
Ref_id:b17 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2024)
Ref_id:b18 Title: Xstest: A test suite for identifying exaggerated safety behaviours in large language models Year: (2023)
Ref_id:b19 Title: Learning to summarize with human feedback Year: (2020)
Ref_id:b20 Title: Alpaca: A strong, replicable instruction-following model. Stanford Center for Research on Foundation Models Year: ()
Ref_id:b21 Title: Language models for dialog applications Year: (2022)
Ref_id:b22 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b23 Title:  Year: ()
Ref_id:b24 Title: Stepwise alignment for constrained language model policy optimization Year: (2024)
Ref_id:b25 Title: Slic-hf: Sequence likelihood calibration with human feedback Year: (2023)
Ref_id:b26 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2024)
Ref_id:b27 Title: Fine-tuning language models from human preferences Year: (2019)
