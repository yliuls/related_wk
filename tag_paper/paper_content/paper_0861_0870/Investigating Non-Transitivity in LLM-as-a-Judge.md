Title: Investigating Non-Transitivity in LLM-as-a-Judge
Abstract: Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% → 96.4% and 82.1% → 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (SWIM) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency. Investigating Non-Transitivity in LLM-as-a-Judge Judge Evaluation A B C Construct preference matrices for each instruction. Conduct pairwise comparisons through a round robin tournament. Compute Elo scores with the Bradley-Terry model.

Section: Introduction
The growing adoption of large language models (LLMs) as generalist systems for complex, open-ended tasks (Ope-nAI et al., 2023;Meta AI, 2024b) presents a critical challenge: the lack of a universally accepted gold-standard evaluation. In many cases, multiple valid responses exist for a given task, complicating the establishment of effective benchmarks. Consequently, a new paradigm for evaluating open-ended tasks focuses on quantifying the alignment of LLMs with human preferences (Ouyang et al., 2022) -an 1 AI Centre, UCL 2 UK AI Security Institute. Correspondence to: Yi Xu <y.xu.23@ucl.ac.uk>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). aspect existing automatic metrics cannot adequately assess. However, human evaluation is costly and lacks scalability (Karpinska et al., 2021). As a result, LLM-based evaluators are now widely used to automate the process, with pairwise comparisons proving particularly effective in aligning with human ratings (Liusie et al., 2024;Liu et al., 2024;Chiang et al., 2023;Li et al., 2023;Lin et al., 2025;Zheng et al., 2023;Samvelyan et al., 2024;Khan et al., 2024).
The typical pipeline for LLM-based automatic evaluation frameworks is using pairwise comparisons between a target model and a fixed baseline model, where an oracle model serves as the judge. By calculating the relative win rate against the baseline model, such comparisons enable ranking target models. However, it is unclear whether using a fixed baseline provides consistent results. If the judge exhibits non-transitive preferences, such as favoring A over B, B over C, but C over A, the resulting rankings can become sensitive to the choice of the baseline model (Figure 2). In this work, we investigate the existence and impact of non-transitivity within AlpacaEval (Li et al., 2023), which has been largely overlooked in previous work. AlpacaEval is a popular pairwise comparison framework that employs GPT-4-Turbo as the fixed baseline model. We introduce Soft Non-Transitivity Deviation (SNTD) as a metric to measure the degree of soft non-transitivity in the judge's continuous preferences and find that LLMs exhibit both hard and soft non-transitive preferences. Additionally, previous studies have demonstrated that LLMs often exhibit various biases (Gallegos et al., 2024) such as position bias (Zheng et al., 2023;Wang et al., 2024;Zhou et al., 2024b), which can lead to spurious correlations in the judge's preferences. We show that the occurrence of non-transitivity is jointly influenced by position bias and the judge model's inherent non-transitive reasoning abilities.
To address the above, we propose the use of round-robin tournaments in the pairwise comparison setting, overcoming the need for a fixed baseline model. We subsequently apply the Bradley-Terry model (Bradley & Terry, 1952) to score models based on tournament outcomes, yielding a more consistent ranking compared to baseline-fixed ranking. To address the computational cost in the round-robin tournament, we propose Swiss-Wise Iterative Matchmaking (SWIM) tournaments to improve efficiency while preserving the robustness of model comparisons.
Our contributions are as follows: 1) We show that LLMs exhibit non-transitive preferences when performing pairwise comparisons. Additionally, we observe that the aggregation of instruction-level non-transitive relationships culminates in model-level non-transitivity (Figure 1). We demonstrate that such non-transitivity makes the ranking highly sensitive to the choice of the baseline model. Changing the baseline model makes the rank order inconsistent and unstable, highlighting the importance of proposing new ranking methods. 2) We find that while position bias significantly contributes to non-transitivity, it is not the sole cause. Our experiments confirm that position switching outperforms random assignment in mitigating position bias for stronger judges when using continuous values for judge's preferences, with reductions ranging from 17% to 44%.
3) We demonstrate that applying round-robin tournaments combined with the Bradley-Terry model reduces the impact of non-transitivity, resulting in more robust rankings. This method also aligns better with human evaluations of model rankings in Chatbot Arena. Finally, we introduce SWIM, an efficient method for adding models with nearly identical performance compared to naive round-robin tournaments.
this section cite: ['b29', 'b18', 'b24', 'b23', 'b8', 'b22', 'b40', 'b34', 'b19', 'b15', 'b40', 'b36', 'b6']

Section: Related Work
LLM-as-a-Judge. The LLM-as-a-Judge (Zheng et al., 2023) evaluation method leverages frontier models to rank responses to open-ended queries without explicit groundtruths. A common approach involves using a fixed baseline model for pairwise comparisons to assess the performance of the target model, as seen in frameworks such as VicunaEval (Chiang et al., 2023), AlpacaEval (Li et al., 2023), and Arena-Hard (Li et al., 2024). The target models are then ranked on the basis of their win rates against the baseline.
However, an implicit assumption in these frameworks is that transitivity holds in preference judgments, which has not been empirically verified. Transitivity requires that if an LLM judge prefers model m A over m B and m B over m C , it must consequently prefer m A over m C . Violations of transitivity can result in unstable rankings that undermine the evaluation framework's reliability (Figure 2). To address this gap, we examine the robustness of current LLM ranking methodologies by extending the AlpacaEval framework to investigate the existence of non-transitivity, aiming to establish a more rigorous foundation for the LLM evaluation system.
Non-
Transitivity in Zero-sum Games. Prior work has explored non-transitivity in two-player zero-sum games within multi-agent reinforcement learning. Balduzzi et al. (2019) characterize agent interactions through convex polytopes, using their dimensionality to decompose transitive and cyclic components. Czarnecki et al. (2020) demonstrate that realworld strategy spaces exhibit a spinning top distribution, where non-transitivity peaks at middling performance levels but diminishes at either lower or higher levels. Given the presence of non-transitivity, evaluating a strategy based on its performance against a single opponent does not reliably reflect its true capability. Therefore, previous achievements in complex games such as StarCraft (Vinyals et al., 2019) and Dota 2 (OpenAI et al., 2019) employ population-based self-play training and evaluate agents through tournamentstyle competitions against diverse opponents. Mirroring the population-based evaluation paradigm that succeeded in non-transitive games, we adopt tournament-based comparisons in LLM-as-a-Judge frameworks to mitigate ranking instability induced by non-transitivity.
this section cite: ['b40', 'b8', 'b20']

Section: Methods

this section cite: []

Section: Measuring Non-Transitivity in Pairwise Comparisons
We employ an LLM, denoted as m J , to conduct pairwise comparisons between models m A and m B . The objective is to determine which of the two outputs, o
A or o
this section cite: []

Section: Soft Transitivity Deviation.
To address this limitation, we propose Soft Non-Transitivity Deviation (SNTD) to measure the degree of non-transitivity for a single instruction with a triplet of models, defined as:
SNTD(m A , m B , m C | I i ) = 1 3 × E JSD ϕ(o (i) A , o (i) B | m J , I i )∥ φ(o (i) A , o (i) B | m J , I i ) + JSD ϕ(o (i) B , o (i) C | m J , I i )∥ φ(o (i) B , o (i) C | m J , I i ) + JSD ϕ(o (i) A , o (i) C | m J , I i )∥ φ(o (i) A , o (i) C | m J , I i ) ,(3)
where the Jensen-Shannon divergence (JSD) quantifies the discrepancy between observed win rates ϕ and estimated win rates φ under transitivity assumptions, as defined below.
this section cite: []

Section: Estimated Win Rate.
We denote the latent quality of the outputs from models m A , m B , and m C on instruction
I i as γ (i) A , γ (i) B
, and γ (i) C , respectively. Given empirical observations ϕ, Bradley-Terry model estimate the quality gap as:
s (i) AB = γ (i) A -γ (i) B = ln ϕ(o (i) A , o (i) B | m J , I i ) 1 -ϕ(o (i) A , o (i) B | m J , I i ) . (4)
Based on that, we can estimate the expected win rate φ under transitivity between any two models from a triplet (m A , m B , m C ) by utilizing the observed win rates between the other two pairs as (See Appendix B.4 for the derivation):
φ(o (i) A , o (i) B | m J , I i ) = 1 1 + e -(s (i) AC -s (i) BC )
.
(5)
this section cite: []

Section: Measuring Model Performance
In this section, we define metrics to quantify and rank model performance given a model pool M, instruction dataset I, and judge m J .
Win Rate Against Baseline. Through currying the judge function with a fixed baseline model m base , we define the win rate against the baseline model as a rating function:
R base (•) = 1 |I| Ii∈I E ϕ(•, o (i) m base | m J , I i ) . (6)
Bradley-Terry Coefficients. Given a series of pairwise comparisons, we employ the Bradley-Terry (BT) model to convert comparison outcomes into coefficients β i ∈ R that quantify the strength of model m i . The optimal BT coefficients β are estimated by maximizing the likelihood:
β = arg max β i j̸ =i W i,j • ln 1 1 + e (βj -βi) ,(7)
where W i,j represents the number of times model i wins against model j. Rather than using discrete labels {0, 1} to count victories, we utilize the judge's preferences as soft labels, defining W i,j = I k ∈I J(m i ≻ m j | I k ), which yields more accurate estimations (See Appendix D).
this section cite: []

Section: Elo Rating.
To establish a standardized measure of model performance, we convert Bradley-Terry coefficients to Elo ratings (Elo, 1966) by setting ξ i = 400 log 10 β i . Under this system, the probability of model m i winning against model m j is expressed as:
P (m i ≻ m j ) = 1 1 + 10 (ξj -ξi)/400 .
(8)
this section cite: ['b14']

Section: Tournament-Based Ranking
We formalize the LLM-as-a-Judge evaluation as a multiplayer game framework, where evaluated language models act as players. Each player's strategy space is defined by its response generation approach under given instructions.
When the judge exhibits non-transitive evaluation behavior, model assessment through fixed-opponent comparisons cannot provide reliable rankings, leading us to characterize this evaluation framework as a non-transitive game.
Round-Robin Tournament. Tournament-based competition with diverse opponents has been established as an effective approach for performance evaluation in non-transitive games (OpenAI et al., 2019;Vinyals et al., 2019), as it enables robust assessment of relative capabilities while mitigating the impact of non-transitivity. Based on this insight, we propose a round-robin tournament structure where each model engages in pairwise evaluation against every other model in the pool, with evaluations conducted by judge m J over instruction set I. This method enables comprehensive model evaluation through comparisons against a diverse population of models rather than relying on a fixed perspective for assessment. We subsequently apply the Bradley-Terry model to comparison outcomes to assign scores, which are then converted into Elo scores for the final ranking.
Swiss-Wise Iterative Matchmaking Tournament. While round-robin evaluation yields reliable rankings, it presents significant computational challenges at scale. Incorporating a new model into a leaderboard of size M necessitates M model-level comparisons compared to a single comparison in baseline-fixed frameworks. To address this computational bottleneck, we propose the Swiss-Wise Iterative Matchmaking (SWIM) tournament (Algorithm 1), drawing inspiration from binary search and Swiss-system tournaments. Our approach dynamically adjusts matchmaking based on Bradley-Terry coefficients, focusing comparisons near model capability boundaries in a logarithmic manner, thereby reducing the number of comparisons to ⌈log 2 (M )⌉.
this section cite: ['b27']

Section: Evaluation Setup
Datasets. We use the AlpacaEval dataset (Li et al., 2023), which includes a wide variety of instruction types, such as information search tasks and coding problems.
Participating models. We evaluate 20 models that appear on both the AlpacaEval and Chatbot Arenafoot_0 leaderboards (see Appendix A.1 for details).
Scenarios. We denote significant performance advantages with ≫ and marginal advantages with ≈. For each scenario, we select representative model triplets based on the win rates of participating models from the AlpacaEval leaderboard (see Appendix A.2 for details).
Judge models. For consistency with AlpacaEval, we maintain the judge configuration and prompt templates. We examine non-transitivity in judgments using two models: GPT-4-Turbofoot_1 and GPT-3.5-Turbo (OpenAI et al., 2023), both with the temperature set to 0. The detailed prompt is provided in Appendix G.1.
Position Switching. LLMs are known to exhibit biases and inconsistencies based on the order of outputs presented in the prompt (Zheng et al., 2023;Pezeshkpour & Hruschka, 2024;Raina et al., 2024). To mitigate this bias, we employ
this section cite: ['b40', 'b30', 'b32']

Section: Non-Transitive Judge Preferences
In this section, we investigate the judge's non-transitive behaviors and analyze their underlying mechanisms.
this section cite: []

Section: Increased Non-Transitivity with Similar Model
As shown in Table 1, non-transitivity emerges across all four scenarios when GPT-4-Turbo serves as the judge. Both PNT and SNTD generally increase as the performance gap between model pairs (m A , m B ) or (m B , m C ) narrows. Notably, while scenarios LL and ML have identical PNT scores, scenario ML exhibits a higher SNTD value, indicating more non-transitivity. This discrepancy highlights the limitation of the PNT-it fails to capture the continuous nature of judge preferences in assessing non-transitivity. Notably, we observe similar trends across other judges and datasets, confirming the generality of the finding (See Appendix B.2).
Weaker Judge is More Non-Transitive. Replicating our evaluation with GPT-3.5-Turbo as the judge reveals an intriguing pattern (Table 1): both PNT and SNTD values are consistently higher than those observed with GPT-4-Turbo and remain relatively stable across all scenarios, suggesting a persistent and substantial level of non-transitivity.
Previous studies have demonstrated that GPT-4-Turbo pos-sesses stronger reasoning capabilities and exhibits significantly less bias compared to GPT-3.5-Turbo (Zheng et al., 2023). We hypothesize that the strong non-transitivity observed with GPT-3.5-Turbo stems from its inability to distinguish the quality differences among outputs, as it is generally considered to have weaker instruction-following abilities than most participating models (Chiang et al., 2024;Lin et al., 2025;Li et al., 2023;White et al., 2025). This inability leads to preferences driven by bias predominantly, which is empirically validated in Section 4.3.
this section cite: ['b40', 'b9', 'b22', 'b38']

Section: Aggregate Non-Transitivity
We
use J(m A ≻ m B ) = 1 |I| Ii∈I J(m A ≻ m B | I i )
to denote the averaged pairwise preference, representing the model-level win rate between m A and m B . We subsequently perform pairwise comparisons across all models and present the win rate matrix in Figure 1 with GPT-4-Turbo as the judge to assess whether instance-level non-transitivity extends to the model-level.
Hard Non-Transitivity at Model Level is Mild. Surprisingly, we detect no instances of hard non-transitivity (e.g., m a ≻ m b , m b ≻ m c , and m a ≺ m c ) at the model level, which we partially attribute to the effectiveness of calibration and randomness mitigation techniques. When implementing a more aggressive approach-where positions are randomly assigned for each evaluation, reducing the process to a single call-we observe occurrences of hard nontransitivity (see Appendix C.2). Nevertheless, model-level non-transitive cases remain notably rare. We hypothesize that this scarcity stems primarily from the low proportion of non-transitive evaluations when using GPT-4-Turbo as the judge. Given the sparsity of non-transitive comparisons,
Investigating Non-Transitivity in LLM-as-a-Judge MA ≫ MB MB ≫ MC MA ≫ MC 0.0 0.2 0.4 0.6 0.8 1.0 Proportion .66 .69 .81 .34 .31 .19 .21 .17 .27 .79 .83 .73 Lead & Lead (LL) MA ≫ MB MB ≈ MC MA ≫ MC .66 .59 .72 .34 .41 .28 .21 .23 .27 .79 .77 .73 Lead & Margin (LM) MA ≈ MB MB ≫ MC MA ≫ MC .61 .69 .75 .39 .31 .25 .22 .17 .27 .78 .83 .73 Margin & Lead (ML) MA ≈ MB MB ≈ MC MA ≈ MC .59 .62 .59 .41 .38 .41 .23 .27 .17 .77 .73 .83 Margin & Margin (MM) Consistent (GPT-4) Ambiguous (GPT-4) Consistent (GPT-3.5) Ambiguous (GPT-3.5)
Figure 3. Larger performance gaps lead to more consistent preferences. We quantify the proportion of consistent preferences of GPT-4-Turbo and GPT-3.5-Turbo across four scenarios differentiated by relative model performance, where ≫ denotes substantial performance advantages and ≈ indicates marginal differences.
-40 -20 0 20 40 Win Rate Gap between MA and MB -40 -20 0 20 40 Win Rate Gap between MB and MC -40 -20 0 20 40 Win Rate Gap between MA and MB 0 10 20 30 40 50 0.0 0.1 0.2 0.3 0.4 0.5 0.6 Number of Non-Transitivity Normalized Degree of Non-Transitivity their aggregated effect is likely overwhelmed by the predominance of transitive evaluations, thus preventing the emergence of observable non-transitivity at the model level.
Despite this, notable instances of soft non-transitivity remain evident, leading to inconsistent ranking as shown by an example in Figure 1. Specifically, while GPT-4-Turbo achieves a win rate of 0.50 against GPT-4o, and GPT-4o wins against Claude-3-Opus with a rate of 0.68, transitivity would predict a win rate of 0.68 for GPT-4-Turbo against Claude-3-Opus. However, the observed rate of 0.72 reveals a subtle violation of transitivity at the model level.
Limitations of the Baseline-Fixed Framework. We further quantify the sensitivity of baseline-fixed frameworks. For each participating model m, we apply the rating function R m (•) to generate rankings, resulting 20 distinct ranking lists. We find that only 20% of models maintain consistent rank positions across all rankings. Moreover, when comparing any pair of ranking lists, only 61% of models preserve their rank positions on average. These findings demonstrate that rankings are highly sensitive to the choice of baseline, indicating that baseline-fixed frameworks produce inconsistent and unreliable model evaluations.
Influence of Model Performance Difference. We further investigate the relationship between non-transitivity and the performance gap among model triplets within all participating models. For each triplet, we define the x-axis as the win rate difference between models m A and m B from the AlpacaEval leaderboard and the y-axis as the difference between m B and m C . The computed PNT and SNTD values, visualized in Figure 4, demonstrate that non-transitivity intensifies as the win rate differences between both model pairs decrease. Both metrics peak near the origin, indicating that non-transitivity is most pronounced when comparing models of similar capabilities (See Appendix B.5 for implementation details).
this section cite: []

Section: Non-Transitivity is Jointly Influenced by Position Bias and Judge's Inherent Reasoning Abilities
Position Bias in Judge Preferences. During the evaluation, we observe that both judges exhibit position bias. Specifically, when evaluating two models on a given instruction, we define a preference as consistent if the judge's preference maintains its relationship to 0.5 (either consistently above or below) with position switching. We report the proportion of consistent preferences in each scenario, using GPT-4-Turbo and GPT-3.5-Turbo as judges (Figure 3).
In all scenarios except MM, both judges show the highest preference consistency when comparing m A and m C , attributable to the substantial performance gap. A potential explanation is that AlpacaEval may have limited discriminative ability when evaluating models with similar capabilities, meaning the presumed performance gap does not hold. Moreover, GPT-3.5-Turbo shows a markedly lower preference consistency than GPT-4-Turbo, indicating that its evaluations are primarily driven by position bias rather than comparing output qualities.
this section cite: []

Section: Factors of Non-Transitivity.
We further categorize instructions into two groups: ambiguous and consistent. An instruction is considered consistent only when the preferences
LL LM ML MM 0.0 0.2 0.4 0.6 0.8 1.0 Proportion 352 289 280 215 1 1 421 468 493 521 32 47 32 67 GPT-4 LL LM ML MM 352 289 280 215 1 1 409 451 468 507 44 64 57 81 GPT-4 (Random Choice) LL LM ML MM 28 34 34 34 605 590 587 603 172 181 184 168 GPT-3.5 LL LM ML MM 28 34 34 34 611 614 614 614 166 157 157 158 GPT-3.5 (Random Choice) Transitive (Consistent) Non-transitive (Consistent) Transitive (Ambiguous) Non-transitive (Ambiguous)
Figure 5. Proportion of (non-)transitive instructions across all scenarios, as evaluated by GPT-4-Turbo and GPT-3.5-Turbo. When evaluating model triplets with GPT-3.5-Turbo as judge, over 96% of instructions exhibit position bias effects. In contrast, GPT-4-Turbo demonstrates substantially higher evaluation consistency. Our analysis reveals that position switching provides more effective bias mitigation than random assignment for less position-biased judges.
between (m A , m B ), (m B , m C ), and (m A , m C ) are all consistent, implying that all comparisons are not influenced by position bias. Otherwise, the instruction is categorized as ambiguous, as at least one of the comparisons is affected by position bias. We report the proportion of non-transitive cases in Figure 5. We find that ambiguous instruction exhibits significantly higher non-transitivity rates compared to consistent instructions, suggesting position bias is indeed a contributing factor. Furthermore, when using GPT-3.5-Turbo as the judge, the proportion of ambiguous instructions exceeds 96%, validating that it exhibits a much stronger position bias than GPT-4-Turbo.
Interestingly, we find non-transitivity still occurs within consistent instructions, with GPT-4-Turbo serving as the judge, indicating that position bias is not the sole cause of non-transitivity. Therefore, we argue that non-transitivity arises from two primary factors. The first is the inherent reasoning capability of the model, which is non-transitive due to the judge's latent comparison criteria. When the quality of the outputs is similar, the judge may display preferences akin to a rock-paper-scissors dynamic. The second factor is the position bias, which affects the judge's preferences. These two factors interact and compound the occurrence of non-transitivity.
this section cite: []

Section: Stronger Position Bias Increases
E[ϕ(o (i) A , o (i) B | m J , I i )] -E[ϕ(o (i) B , o (i) A | m J , I i )]
. Using GPT-4-Turbo as the judge, we evaluate all triplet permutations and partition PD values into bins. As shown in Figure 6-Left, the proportion of non-transitive cases increases with PD, demonstrating a strong positive correlation.
Usefulness of Position Switching. Instead of using position switching, we repeat the experiment by randomly assigning the positions of the outputs in the prompt (Figure 5). Since all preferences in the consistent instruction are consistent, the proportion of non-transitive cases remains unchanged. However, for ambiguous instructions, we observe divergent effects: GPT-4-Turbo exhibits a significant increase in nontransitivity, while GPT-3.5 shows a slight decrease.
The distributions of judge preference (see Appendix B.6) show distinct evaluation patterns between judges. When mitigating GPT-3.5-Turbo's position bias through position switching, the model tends to generate more uncertain outcomes (averaged preference ≈ 0.5). In contrast, GPT-4-Turbo exhibits different characteristics: while position switching occasionally introduces uncertainty, its debiased preferences generally maintain clear output distinctions. This finding suggests that position switching can reduce non-transitivity for stronger judges that are less affected by position bias, with reductions ranging from 17% to 44%. However, for weaker judges that are more susceptible to position bias, it may have the opposite effect.
Prompting Strategies to Mitigate Non-transitivity. We explore various prompting strategies to address non-transitivity in model judgments. Our analysis focuses on Scenario MM, where the capabilities of the compared models are closely matched, making it easier to observe both non-transitive behaviors and the effects of different prompts. Our findings show that providing judges with a structured evaluation checklist (Cook et al., 2024) would marginally reduce nontransitive cases. Interestingly, while incorporating Chain-of-Thought reasoning (Wei et al., 2022) helps mitigate position bias, it also leads to a higher incidence of non-transitive preferences. Moreover, allowing the judge to declare ties not only increases position bias but also further amplifies non-transitivity. See Appendix C.3 for detailed results.
this section cite: ['b10', 'b37']

Section: Results of Tournament-Based Ranking
We conduct a round-robin tournament to obtain pairwise comparisons and apply the Bradley-Terry model to compute ratings, which are then converted to Elo scores. The resulting Elo scores and rankings for all 20 evaluated models are presented in Table 9 in the Appendix.
To assess the effectiveness of our framework, we consider the human preference ranking from the Chatbot Arena as the reference. We compute the Spearman and Kendall correlations between our round-robin-based ranking and the Chatbot Arena. We also compare these correlations with those between the AlpacaEval and the Chatbot Arena. As shown in Table 2, our method achieves higher correlations, with a 4% increase in Spearman correlation and a 5.2% increase in Kendall correlation.
Length-Controlled Winrate. To mitigate verbosity bias and ensure a fair comparison, we adopt the generalized linear model with the same weights as Length-Controlled AlpacaEval (Dubois et al., 2024) to derive length-controlled preferences. Using these preferences, we compute the length-controlled Bradley-Terry coefficients, which are then converted to length-controlled Elo scores. Table 2 shows that our length-controlled round-robin ranking further improves correlations, with a 1.4% increase in Spearman corre-lation and a 4.2% increase in Kendall correlation compared to length-controlled AlpacaEval.
Performance of SWIM. We demonstrate that both roundrobin-based ranking and SWIM-based ranking outperform AlpacaEval, as shown in Figure 6-Right. We do not compare performance under length control, as the generalized linear model is an empirical approach that may be less interpretable, potentially affecting fairness.
this section cite: ['b13']

Section: Limitations and Future Work
Our study has several limitations. While AlpacaEval provides diverse instructions, it may not fully capture real-world open-ended tasks, necessitating validation of our method across broader domains. Additionally, extending our findings to judge models beyond GPT-4-Turbo and GPT-3.5-Turbo is an important direction for future work. Furthermore, while our benchmark relies on human rankings from Chatbot Arena, inherent human biases (Chen et al., 2024) may introduce non-transitivity in human preferences, fundamentally limiting the achievable alignment between automated and human evaluations.
Secondly, our focus on pairwise comparisons leaves open questions about non-transitivity in pointwise evaluations. While pointwise methods inherently avoid position bias caused by output ordering, converting these scores to pairwise comparison (A > B if score(A) > score(B)) may introduce new forms of non-transitivity, depending on the granularity and consistency of rating criteria. Future work should investigate whether such conversions preserve transitivity and identify conditions that modulate cyclic preferences.
Finally, our analysis relies on the Bradley-Terry model, which assumes transitive model-level preferences by assigning each model a global scalar score. While we do observe instance-level non-transitivity in our pairwise comparisons, these cases are relatively rare, and hard non-transitivity in the aggregated model-level preferences is mild. Therefore, we find the Bradley-Terry model sufficient for our ranking purposes. Nevertheless, we acknowledge that this implementation may not fully capture the nuanced capabilities of models. We leave this as a direction for future work, focusing on more expressive alternatives that parameterize model capabilities in a multi-dimensional space (Duan et al., 2017), which remains a promising and under-explored approach for improving the robustness of LLM-as-a-judge evaluations.
this section cite: ['b7', 'b12']

Section: Conclusion
In this paper, we comprehensively study the impact of nontransitivity in the current LLM-based framework with pairwise settings, filling a gap in this area of research. Our findings show that non-transitivity can be observed at the instruc-tion level during judgment and is related to the reasoning capability of the judge. The aggregation of instruction-level non-transitivity further leads to model-level non-transitivity, revealing the limitations of the baseline-fixed framework, as the rankings in this setting depend on the choice of the baseline model. Our analysis also demonstrates that position bias is a key factor in non-transitivity, with systematic position switching proving more effective than random assignment in reducing non-transitivity for stronger judges.
To address the above, we propose a baseline-free framework utilizing round-robin tournaments with Bradley-Terry model, which captures non-transitivity patterns and demonstrates better alignment with human. Recognizing the computational constraints of round-robin tournaments, which require O(nm 2 ) instruction-level comparisons for ranking m models across n instructions, we propose SWIM tournaments. This approach achieves O(nm log m) complexity through dynamic matching, substantially reducing computational cost while maintaining nearly identical performance. The code and data are available at https:  //github.com/yix8/llm-nontransitivity.
A. LLM Details.
In this section, we provide detailed information about all models participating in the ranking evaluation for our experiments.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2024)
Ref_id:b1 Title:  Year: (2024)
Ref_id:b2 Title: When benchmarks are targets: Revealing the sensitivity of large language model leaderboards Year: (2024-08)
Ref_id:b3 Title: Model card and evaluations for claude models Year: (2023)
Ref_id:b4 Title: The claude 3 model family: Opus, sonnet, haiku Year: (2024)
Ref_id:b5 Title: Open-ended learning in symmetric zero-sum games Year: (2019-06)
Ref_id:b6 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b7 Title: Humans or LLMs as the judge? a study on judgement bias Year: (2024-11)
Ref_id:b8 Title: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023-03)
Ref_id:b9 Title: Chatbot arena: An open platform for evaluating LLMs by human preference Year: (2024-07)
Ref_id:b10 Title: Ticking all the boxes: Generated checklists improve llm evaluation and generation Year: (2024)
Ref_id:b11 Title: Real world games look like spinning tops Year: (2020)
Ref_id:b12 Title: A generalized model for multidimensional intransitivity Year: (2017-05-23)
Ref_id:b13 Title: Length-controlled alpacaeval: A simple debiasing of automatic evaluators Year: (2024)
Ref_id:b14 Title: The USCF Rating System: Its Development, Theory, and Applications Year: (1966)
Ref_id:b15 Title: Bias and fairness in large language models: A survey Year: (2024-09)
Ref_id:b16 Title: A family of highly capable multimodal models Year: (2023)
Ref_id:b17 Title: Mistral 7b Year: (2023)
Ref_id:b18 Title: The perils of using Mechanical Turk to evaluate open-ended text generation Year: (2021-11)
Ref_id:b19 Title: Debating with more persuasive LLMs leads to more truthful answers Year: (2024-07)
Ref_id:b20 Title: From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline Year: (2024)
Ref_id:b21 Title: Alpacaeval: An automatic evaluator of instruction-following models Year: ()
Ref_id:b22 Title: Benchmarking LLMs with challenging tasks from real users in the wild Year: (2025)
Ref_id:b23 Title: Aligning with human judgement: The role of pairwise preference in large language model evaluators Year: (2024)
Ref_id:b24 Title: LLM comparative assessment: Zero-shot NLG evaluation through pairwise comparisons using large language models Year: (2024)
Ref_id:b25 Title: Introducing meta llama 3: The most capable openly available llm to date Year: (2024)
Ref_id:b26 Title: Introducing llama 3.1: Our most capable models to date Year: (2024)
Ref_id:b27 Title:  Year: (2019)
Ref_id:b28 Title:  Year: (2023)
Ref_id:b29 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b30 Title: Large language models sensitivity to the order of options in multiple-choice questions Year: (2024-06)
Ref_id:b31 Title:  Year: (2024-05-01)
Ref_id:b32 Title: Is LLM-as-a-judge robust? investigating universal adversarial attacks on zero-shot LLM assessment Year: (2024-11)
Ref_id:b33 Title: Verbosity bias in preference labeling by large language models Year: (2023)
Ref_id:b34 Title: Rainbow teaming: Open-ended generation of diverse adversarial prompts Year: (2024)
Ref_id:b35 Title: Grandmaster level in starcraft ii using multi-agent reinforcement learning Year: ()
Ref_id:b36 Title: Large language models are not fair evaluators Year: (2024-08)
Ref_id:b37 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b38 Title: Livebench: A challenging, contamination-limited LLM benchmark Year: (2025)
Ref_id:b39 Title: Empowering large pre-trained language models to follow complex instructions Year: (2024)
Ref_id:b40 Title: Judging LLM-as-a-judge with MT-bench and chatbot arena Year: (2023)
Ref_id:b41 Title: Fairer preferences elicit improved humanaligned large language model judgments Year: (2024-11)
Ref_id:b42 Title: Batch calibration: Rethinking calibration for in-context learning and prompt engineering Year: (2024)
Ref_id:b43 Title: Starling-7b: Improving helpfulness and harmlessness with RLAIF Year: (2024)
