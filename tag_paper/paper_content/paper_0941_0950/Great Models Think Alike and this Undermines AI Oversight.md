Title: Great Models Think Alike and this Undermines AI Oversight
Abstract: As Language Model (LM) capabilities advance, evaluating and supervising them at scale is getting harder for humans. There is hope that other language models can automate both these tasks, which we refer to as "AI Oversight". We study how model similarity affects both aspects of AI oversight by proposing Chance Adjusted Probabilistic Agreement (CAPA): a metric for LM similarity based on overlap in model mistakes. Using CAPA, we first show that LLM-as-a-judge scores favor models similar to the judge, generalizing recent self-preference results. Then, we study training on LM annotations, and find complementary knowledge between the weak supervisor and strong student model plays a crucial role in gains from "weak-to-strong generalization". As model capabilities increase, it becomes harder to find their mistakes, and we might defer more to AI oversight. However, we observe a concerning trend -model mistakes are becoming more similar with increasing capabilities, pointing to risks from correlated failures. Our work underscores the importance of reporting and correcting for model similarity, especially in the emerging paradigm of AI oversight.

Section: Introduction
Machine Learning model capabilities have improved immensely over the last few years. Scaling up the amount of data used for training has played a crucial role in these improvements (Kaplan et al., 2020). Initially, most of the gains in Language Model (LM) capabilities came from scaling pretraining data (Llama Team, 2024a). Recently, there Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
Figure 1. Our Main Contributions. We develop a novel probabilistic metric for model similarity, CAPA (κp), which adjusts for chance agreement due to accuracy. Using this, we find (1) LLM-asa-judge scores are biased towards more similar models controlling for the model's capability (2) Gain from training strong models on annotations of weak supervisors (weak-to-strong generalization) is higher when the two models are more different, (3) Concerningly, model errors are getting more correlated as capabilities increase.
is increasing interest in post-training, either with human preferences (Ouyang et al., 2022), or task-specific expert annotations (Lightman et al., 2024). Collecting human preferences or annotations is slow and expensive. Therefore, with increasing model capabilities, an attractive alternative is to use LMs to annotate training data (Gilardi et al., 2023) and score model outputs (Zheng et al., 2023), to boost both training (Stiennon et al., 2020) and evaluation (Li et al., 2024b). In this paper, we refer to both these techniques together as AI oversightfoot_0 .
Can we rely on AI oversight going forward? This remains a topic of much debate. In this work, we study oversight from the perspective of model similarity. When assessing or teaching humans, it is well recognized that individuals have different strengths and weaknesses. Similarly, two models with 50% accuracy may misclassify completely different samples and thus be highly dissimilar (having differ-ent 'strengths'). To measure model similarity, we build on error consistency (Geirhos et al., 2020), which measures overlap in the samples where two models err beyond what is expected by chance due to the two models' accuracies. In Section 2, we extend the error consistency metric in two crucial ways -1) by counting differences in predictions rather than correctness for each sample, and 2) incorporating output probabilities. In this way, our novel similarity metric, Chance Adjusted Probabilistic Alignment (CAPA), allows us to quantify functional similarity between models. We use this to analyze both evaluation and training using AI oversight as depicted in Figure 1:
1. LLM-as-a-Judge. Prior work has shown that LM judges are biased towards their own generations (Liu et al., 2024;Panickssery et al., 2024). It might seem possible to avoid this concern by simply using a different model as the judge. However, just like human evaluators prefer candidates with similar traits (Bagues & Perez-Villadoniga, 2012), could LM judges also exhibit this affinity bias? In Section 3, we study this using CAPA, finding LM judges indeed assign higher scores to models that are more similar to themselves.
this section cite: ['b33', 'b60', 'b48', 'b29', 'b26', 'b49', 'b61', 'b1']

Section: Training LMs on annotations of other LMs.
Next, we study the effect of similarity on inter-LM training setups, where one model annotates data used to train another model. We hypothesize that performance gained through such training leverages complementary knowledge among models, and is thus inversely proportional to CAPA. We investigate this hypothesis in Section 4, following the weakto-strong generalization setup (Burns et al., 2024), where a strong (larger) student model is shown to outperform the weaker (smaller) supervisor whose annotations it is finetuned on. Indeed, we find performance gains are higher when the weak supervisor and the strong student model are more different. Moreover, our findings indicate a higher performance ceiling for weak-to-strong generalization than previously estimated, if the weak supervisor's complementary knowledge is leveraged effectively.
3. With increasing LM capability errors are becoming more correlated. AI oversight is gaining popularity as capabilities increase. The above results show the benefits of diverse models for AI oversight -less similarity between models reduces bias in LLM-as-a-judge, and also leads to greater gains when training on LM annotations. Unfortunately, in Section 5 we find that as popular frontier LMs become more capable, their mistakes become more similar as captured by CAPA. This trend indicates a risk of common blind-spots and failure modes when using AI oversight, which is concerning for safety (Kenton et al., 2024).
Overall, our work proposes a novel probabilistic metric for model similarity, and demonstrates the risks of correlated mistakes in the emerging paradigm of AI oversight. We hope the community shifts towards releasing sample-wise model predictions alongside benchmark scores (Burnell et al., 2023;Ghosh et al., 2024), as they enable richer analysis like measuring similarity.
this section cite: ['b9', 'b35', 'b8', 'b29']

Section: Methodology: Measuring LM Similarity
We begin by describing how we quantify model similarity.
this section cite: []

Section: Background
Functional similarity: Prior work on model similarity has focused on two classes of similarity measures: representational and functional similarity (see Klabunde et al. (2025) for a recent survey). Representation similarity metrics focus on the weights and activations of the networks (Kornblith et al., 2019), comparing how features are represented internally. In contrast, functional similarity metrics focus on the input-output behavior of the model. Functional similarity metrics are more broadly applicable than representation metrics as (1) they allow comparisons across model families and architectures and (2) are applicable for models behind an API (where weights are not released). Functional similarity metrics are more interpretable because they operate on data samples rather than noisy, complex internal representations (Golechha & Dao, 2024). Despite large architectural differences across models and model families, their outputs can still be fairly similar. Moreover, Geirhos et al. (2020) argue models with similar internal mechanisms make similar mistakes, and thus mistakes can proxy whether models use similar internal mechanisms. Therefore, in the present work we focus on functional similarity metrics.
Error Consistency: A popular similarity metric designed in the context of comparing mistakes of image-classifiers to humans is error consistency (Geirhos et al., 2020). It quantifies the overlap on samples where two models make mistakes while normalizing for chance overlap due to accuracy. First, they define c obs as the "observed error overlap" i.e., the fraction of samples on which both models are correct or both models are wrong. This itself is used a metric in recent work on LM similarity, Dutta et al. (2024). However, as Geirhos et al. (2020) point out, c obs has a crucial shortcoming: two independent models with high accuracy will have a higher c obs by chance than two models with low accuracy ( 1). An independent model here is one that is correct on a uniform random subset (size corresponding to accuracy) of samples, and wrong on the others. For instance, two independent models with 90% accuracy will agree on at least 81% of the samples by chance, whereas for two models with 50% accuracy, the lower-bound on chance agreement drops to 25%. Consequently, to account for this, Geirhos et al. (2020) calculate the "expected error overlap" (c exp ) as c exp = acc 1 • acc 2 +(1 -acc 1 )(1 -acc 2 ) where acc i is the accuracy of model i. Similar to Cohen's κ (Cohen, 1960), error consistency (Eq. 1) is then defined as the fraction of ex-Table 1. Comparison of Functional Model Similarity Metrics. Only our metric, CAPA, satisfies all three desiderata: 1 Adjusts for accuracy -The metric should not inflate scores for high accuracy model pairs due to lesser scope to disagree. 2 Distinguishes different mistakes -The metric should consider different wrong predictions as a disagreement. 3 Incorporates probabilities -The metric should use the probability distribution over predictions provided by the models.
this section cite: ['b37', 'b42', 'b30', 'b26', 'b26', 'b21', 'b26', 'b26', 'b16']

Section: Metric
Adjusts for Distinguishes Incorporates Accuracy different mistakes Probabilities %Flips = 1 -c obs (Dutta et al., 2024) ✗ ✗ ✗ Cohen's κ, Scott's π, Fleiss κ ✗ ✓ ✗ %Agreement (Zheng et al., 2023) ✗ ✓ ✗ Error Consistency (Geirhos et al., 2020) ✓ ✗ ✗ Pearson / Matthew's Correlation of Errors ✓ ✗ ✗ Divergence metrics like KL, JSD ✗ ✓ ✓ CAPA (Ours) ✓ ✓ ✓ cess agreement observed (c obs -c exp ) from what is possible beyond chance (1 -c exp ):
k = c obs -c exp 1 -c exp ,(1)
this section cite: ['b21', 'b26']

Section: Our Contribution
We identify two key limitations of error consistency (k):
Does not distinguish differing mistakes ( 2 ): If two models make wrong but different predictions, error consistency still counts that as an agreement. For example, two models that are always wrong, even in different ways, have perfect error consistency (k = 1). It thus overestimates similarity.
Does not capture probability information ( 3 ): For comparison to humans, error consistency assumes a single top prediction, whereas models inherently output a probability distribution. Ignoring these probabilities can lead to incorrect conclusions about model similarity. Consider two models whose outputs are [0.49, 0.51] and [0.51, 0.49]. Despite their small differences, binary labels would classify them as making entirely different predictions (0 and 1). Conversely, models with predictions [0.99, 0.01] and [0.51, 0.49] may share the same binary output (0 and 0) but differ significantly in confidence distribution.
Novel Metric. We redefine c obs and c exp to address the above limitations. For clarity we adjust the notation of our agreement metrics to c p obs and c p exp . To compute c p obs we directly use the model output probabilities (Eq.2), thus accounting for disagreement on incorrect options and better capturing model similarity. This approach lets us calculate c p obs without sample-wise ground-truth annotations. For c p exp , we take into account that the model output predictions can span over multiple options rather than only looking at sample-wise accuracy.
Definition. We define κ p in the context of Multiple Choice Questions (MCQs), which is the format of many popular benchmarks for LMs. We provide a detailed derivation in Appendix A.1, with extensions to classification and exact match evaluations in Appendix A.3.
Observed Agreement (c p obs ): It represents the probability of agreement if the model's predictions were sampled based on the observed likelihoods assigned over options. Formally,
c p obs = 1 |D| x∈D oi∈O(x) p 1 (o i ) • p 2 (o i ),(2)
where p 1 (o i ) and p 2 (o i ) are the output probabilities for model 1 and 2, respectively, on a data sample x for option o i . O(x) are the possible options: O(x) = [o i , . . . , o n ], and |D| is the total number of data points.
Chance Agreement (c p exp ): To account for higher accuracies inflating c p obs , we normalize by the agreement expected from two independent models. First, we define p j as the average probability model j assigns to the correct option across all samples. For a perfectly calibrated model p j approaches accuracy, thus aligning with the motivations in error consistency. Then, we define independent models as assigning p j probability to the correct option, and uniformly distributing the remaining 1 -p j probability over the incorrect options. The latter is necessary, as there is no coherent concept of "classes" for MCQ data, i.e. the options can be permuted. This prevents us from computing class marginals for the remaining options, such as in inter-annotator agreement metrics like Cohen's Kappa, Scott's Pi (Scott, 1955), Fleiss' Kappa (Fleiss et al., 1981). Formally,  In this simulation, for each model we select an independent random subset of samples as correct, with the first having a fixed 90% accuracy, while for the second accuracy is varied from 50% to 90%. CAPA correctly reports 0 similarity when models have uncorrelated errors.
c p exp = p 1 • p 2 chance agreement on correct option + (1 -p 1 ) • (1 -p 2 ) • 1 |D| x∈D 1 |O(x)| -1 chance agreement on incorrect option
Finally, the equation for CAPA is:
κ p = c p obs -c p exp 1 -c p exp (4
)
Interpretation. We prove κ p is bounded between -1 and 1 in Appendix A.6. A value of 0 means the models have the same agreement as independent models given their accuracies. A negative value means the models disagree, and a positive value indicates they agree beyond independent models with their accuracy. As κ p increases, it means models make more similar mistakes, their errors become more correlated, and they are functionally less different. We use these interpretations interchangably.
Alternatives and Justification. We summarize comparisons to existing functional similarity measures based on key desiderata ( 1 -3 ) in Table 1. To demonstrate the empirical effects, we perform a simulation, where we take two binary classification models, that are correct on an independent random subset of samples (more details in Appendix A.4). Since these models are independent, any agreement is by chance, and their predictions are entirely uncorrelated. In such a case, we want a metric that reports 0 similarity. In Figure 2 we report metric behavior when one model has a fixed 90% accuracy, while varying the accuracy of the other model from 50% to 90%. Our proposed metric, CAPA, indeed reports 0 similarity, indicating that the models are independent and their predictions uncorrelated. This is in contrast to popular metrics like Cohen's κ (Cohen, 1960), used to measure agreement between two annotators, Jensen Shannon (JS) Distance, used to measure a normalized divergence between two probability distributions, and %Agreement, which is the number of same predictions made by the two models. All three alternative metrics do not adjust for chance agreement due to higher accuracy, and thus lead to higher similarity scores being reported as the second model's accuracy increases, even though the two models continue to make independent and uncorrelated predictions. In Appendix A.1 we prove that CAPA is a generalization of error consistency and reduces to it for binary classification tasks if we discretize predictions to 1 on the correct option and 0 on others. As such, in the simulation presented here, error consistency would be quite similar to CAPA, as both metrics adjust for accuracy. In Appendix A.7 we show that CAPA demonstrates the expected trend most clearly also in settings when models have correlated predictions (Appendix Figure 89). Lastly, CAPA can be extended beyond pairwise comparisons to multiple models (Appendix A.2), and for completeness, we present probabilistic extensions for Cohen's κ, Scott's π, Fleiss' κ F in Appendix A.5.
this section cite: ['b23', 'b16']

Section: Affinity Bias in LLM-as-a-Judge
Evaluating free-response model generations automatically at scale is tricky (Biderman et al., 2024). This has led to popular leaderboards like Arena-hard-auto (Li et al., 2024b), AlpacaEval 2.0 (Dubois et al., 2024), and Aidan-Bench (McLaughlin et al., 2024) adopting LLM-as-a-Judge for scoring. Recent work cautions that language models are biased towards their own outputs (Liu et al., 2024;Panickssery et al., 2024), and these leaderboards assume that excluding the judge model from the rankings circumvents this problem. However, human interviewers have been shown to also be biased towards candidates with similar knowledge and traits, a phenomenon called affinity bias (Bagues & Perez-Villadoniga, 2012). We study whether LMs also exhibit a bias toward similar models. This would indicate that it is not sufficient to just use a held-out LM as the judge; one must account for the confounder of similarity.
this section cite: ['b4', 'b18', 'b49', 'b61', 'b1']

Section: Experimental Setup
To study whether LM judges prefer more similar models, we evaluate a large set of judges and models on MMLU-Pro (Wang et al., 2024), a benchmark for hard problem solving questions across 14 domains. We filter 8,707 questions that can also be answered in a free-text response style, without options, following Myrzakhan et al. (2024). Each question is posed to every evaluated model as an MCQ and as an open-style question. The per-sample results of the former are used to compute the similarities of judge-model pairs, whereas responses to the latter are given to an LLMas-a-judge. The judge has to make a binary decision on whether a free-text response is correct or wrong. This is done based on its own internal knowledge without access to a ground-truth solution. We call the average of binary judgments across the benchmark the model's Judgment Score for a given judge. Using a parallel MCQ evaluation with ground-truth answers allows us to compare the judgment scores with verifiable accuracy measurements (details and comparisons with alternatives are in Appendix B.3), consistent with prior scalable oversight research (Bowman et al., 2022). We compute pairwise similarity with CAPA, κ p , across 9 judge and 39 model pairs. For a complete overview of models investigated, question filtering, inference setup, and prompts used for judges, see Appendix B.
this section cite: ['b57', 'b7']

Section: Results & Discussion
Q1: Do LM Judges favor more similar models? As a motivating example, Qwen2.5-72B-Instruct as a judge scores Qwen2.5-7B-Instruct as being 71% correct, while the more capable (41% vs 51% MCQ accuracy) Llama-3.1-70B-Instruct is deemed less accurate at 67%. In Figure 3 we show that model favoritism extends beyond selfor familypreference to all models that are functionally similar. We find a significant (p < 0.01) positive correlation (average Pearson r=0.84) between LLM-as-ajudge scores and model similarity (κ p ) for all judges. To ensure this trend is not only an artifact caused by aggregating across all MMLU-Pro categories, per-category results for the same experiment are shown in the Appendix B.5. These exhibit the same affinity bias observed here for all different topics.
Q2: Is this merely due to better accuracy? Note that while κ p adjusts for inflation in agreement of highly accurate models, we do expect models with lower accuracy to be less similar with highly capable judge models, and thus have lower κ p . To control for the accuracy of the evaluated model we perform multiple regression and partial correlation analysis (Table 2). The multiple regression analysis shows that both, accuracy and similarity, have a significant positive effect on the judge score. The coefficient of accuracy increases for more capable judge models, consistent with prior work showing improved alignment with human judges (Thakur et al., 2024). We find that especially for small models (<32B) the effect of similarity is greater. The partial correlation results control for accuracy and confirm that there is still a significant effect of similarity on judgment scores even for the best judge models. Altogether, the statistical analysis confirms that judgment scores are confounded by affinity bias. For extended results also including model size as a confounder please see Appendix B.1.
Q3: Does this transfer to LLM-as-a-judge for binary preference in free-form generation? In most cases, LLMas-a-judge is not used to check the technical correctness of responses, but rather to match human preference by selecting the better one from a pair of responses. To extend our findings to the additional domain of chat responses, we provide an additional experiment on the AlpacaEval benchmark in Appendix B.4 (Dubois et al., 2023). We show that the Elo obtained by using the same judges as before to obtain binary preferences still strongly correlates with model similarity on MMLU-Pro. This indicates the existence of an affinity bias in settings other than QA.
this section cite: ['b17']

Section: Learning from Complementary Knowledge of LM Annotators
We now study the role of similarity in AI supervising training. This can allow scaling up training data by reducing reliance on costly and time-intensive expert human inputs.
There is hope that models can learn from other models to improve further even on tasks where they surpass human capabilities (Hughes et al., 2024). Where could this improvement come from? We hypothesize that the existence of complementary knowledge or complementary capabilities between two LMs can be one mechanism for driving improvements from LM annotations, if effectively leveraged. This complement can exhibit itself in the form of differing predictions on training data, and can thus be quantified using functional similarity between the supervisor and student model. Lower κ p is indicative of more complementary knowledge, and as an implication of our hypothesis, should inversely correlate with the performance gain of a student model when trained on the supervisor's annotations. et al. (2024) study training a larger student model on annotations from a small task-finetuned "expert" teacher. They find the student can outperform the supervisor, a phenomenon they call "weak to strong generalization". We study this setup as it can seem counter-intuitive when viewed from the lens of accuracies. How can training a 60% ac-
this section cite: []

Section: Experimental Setup

this section cite: []

Section: Burns
G-2-2b Q-2.5-1.5b L-3.2-3b P-2-2.7b Both Complementary G -2 -9 b Q -2 .5 -7 b L -3 .1 -8 b G-2-2b Q-2.5-1.5b L-3.2-3b P-2-2.7b Elicitable G -2 -9 b Q -2 .5 -7 b L -3 .1 -8 b Neither 0.0 0.2 0.4 0.6 0.8 1.0 Weak-to-Strong Trained Model Accuracy Correct Wrong Wrong Correct Strong Elicited Model Prediction Weak Supervisor Prediction curacy student on a 70% accuracy task-finetuned teacher lead to 75% accuracy? We adopt a lens of complementary knowledge to understand weak-to-strong generalization.
Following Burns et al. (2024), we define "weak" and "strong" based on model size. We study 4 weak models in the 1 -3B parameter range, and 3 strong models in the 7 -9B parameter range, for a total of 12 model pairs, and 15 of the binary classification NLP tasks studied in Burns et al. (2024), specified in Table 9. The smaller models do have lower accuracy (before finetuning) than the bigger ones across tasks, as shown in Appendix Figure 17, justifying calling them "weak" and "strong". We measure similarity between the weak supervisor and base student model on the validation set. We then perform weak-to-strong training on the student model, using the confidence-weighted finetuning objective proposed in Burns et al. (2024). We investigate if similarity is an apriori predictor of performance gained on the test set. The full setup is consistent with the openweight model reproduction by (Scherlis et al., 2024), and is described in Appendix C.1.
this section cite: ['b9', 'b9', 'b9']

Section: Results & Discussion
Q1: Does Complementary Knowledge Influence Performance Gain? Figure 4 shows that for all model combinations, similarity between the weak supervisor and initial strong student inversely correlates with the improvement obtained from weak-to-strong training (r = -0.85). Even Table 3. Accuracy gains possible from weak-to-strong training. We average accuracies across 15 datasets and 12 model pairs (180 training runs) and report gaps to the student model's initial accuracy. Complementary knowledge transfer can enable higher gains than the previously considered ceiling estimate from elicitation.
this section cite: []

Section: Model Accuracy Gap
Initial Strong Student 75.1% Weak Supervisor
this section cite: []

Section: +4.1 Weak to Strong Trained Student

this section cite: []

Section: +7.4

this section cite: []

Section: Ceiling Estimate
Ground-truth Elicitation (previous) +11.2 Elicitation ∪ Complementary (ours)
+14.1 after using partial correlation analysis to control for the accuracy gap between the weak supervisor and strong student, similarity is inversely correlated with weak-to-strong gain (r = -0.35, p < 0.01). Thus, tasks where the supervisor and student make less correlated errors tend to yield greater improvements. This contributes towards understanding why gains from weak to strong training vary across tasks, an open question posed by Burns et al. (2024). The observation is not specific to our similarity metric, Appendix Figure 16 shows the same trend holds with alternate metrics, albeit with lesser variance explained.
this section cite: ['b9']

Section: Q2. Does Complementary Knowledge Add Beyond Elicitation?
The original explanation for performance gains from weak-to-strong generalization is that the weak supervisor "elicits" the latent knowledge in the superior representations of the stronger student (Burns et al., 2024). To investigate whether complementary knowledge adds to this explanation or is subsumed within it, we first obtain the strong model with "upper-bound" elicitation, by finetuning it on ground-truth annotations. We refer to this as the strong elicited model. We can then separate the test data into four parts based on whether the strong elicited and weak supervisor model were correct or wrong, measuring average accuracy of the weak-to-strong model on each part to disentangle gains from different factors. The experiment setup is discussed further in Appendix C.2.
Figure 5 reports aggregate values across 15 tasks for 12 model pairs. Accuracy on the bottom-left quadrant (avg. 71.9%) can only be due to successful elicitation, as here the weak supervisor was wrong. Accuracy on the top-right quadrant (avg. 42.2%) can only be due to complementary knowledge transfer as here the upper-bound elicitation model was wrong. This confirms that elicitation plays an important role in weak-to-strong generalization, with complementary knowledge transfer from the weak supervisor also contributing to significant gains. Q3. Where can weak-to-strong training improve? The strong elicited model is considered to represent upper-bound performance. However, recent work has shown that the performance of the strong elicited model can be surpassed (Shi et al., 2025), indicating it is not really an "upper-bound". Indeed, the strong elicited model's performance does not account for potential gains from leveraging the complementary knowledge of the weak supervisor. We compute a new ceiling, taking the union of correct predictions between the weak supervisor and strong elicited model, which is significantly higher as shown in Table 3. Interestingly, on the training set, the weak-to-strong trained model shows similar accuracy on the top-left and bottom-right quadrants as shown in Figure 15. Yet, when generalizing to unseen samples, it falls back more often to its initial priors. We hope this analysis guides future work on improving weak-to-strong training methodology, by highlighting leveraging complementary knowledge as a concrete avenue for improvement.
this section cite: ['b9']

Section: Models are making more similar mistakes as capabilities increase
The previous two sections highlighted two major advantages of having access to more diverse LMs: a) it leads to less biased judges, b) it can drive more performance gains from training on LM annotations. This points to the importance of diversity, or lower model similarity, for AI oversight. As AI oversight becomes increasingly relevant with advancing capabilities, we now study similarity trends in existing LMs across different levels of capability. It has been shown model representations across modalities are converging with increasing capabilities (Huh et al., 2024). Does this also lead to more similar mistakes?
this section cite: []

Section: Experimental Setup
We collect sample-wise evaluation files for 130 official models from the OpenLLM Leaderboard 2 released by Hugging-Face, listed in Appendix D.5. We use MMLU-Pro (Wang et al., 2024) and Big Bench Hard (BBH) (Suzgun et al., 2023) as they measure a broad range of capabilities using MCQ, and frontier models have reasonable accuracies while not saturating these datasets. We first bucket these models into five performance percentile ranges. Then, for each model, we compute its mean similarity (κ p ) with models in the same bucket from different developers, to prevent confounding from distillation or continual training. More setup details are provided in Appendix D.1. In Appendix D.3 we also report pairwise results, and using the extension of κ p for sets of M > 2 models.
this section cite: []

Section: Results & Discussion
Q1. Are model errors becoming more correlated with improving capabilities? Figure 6 shows a strong positive correlation between model capabilities and κ p , which measures similarity beyond chance agreement due to accuracy. In Appendix D.4 we find this also holds across individual categories in both datasets, not just in aggregate.
this section cite: []

Section: Potential Implications.
If this trend continues, it could mean greater affinity bias when using LM judges, and lower potential for gains from inter-LM training in the context of our earlier results. It could undermine benefits from using LM juries by compromising independence and amplifying collective biases. Most concerningly, our results indicate that as model blind-spots get harder to detect, making us defer more to AI oversight, models also make more similar mistakes, posing safety risks from correlated failures.
Q2. Why are model errors becoming more correlated? This is an interesting research direction in itself. We perform a preliminary analysis in Appendix D.2, summarizing key conclusions here. First, we observe only a slight increase in similarity for harder questions in our datasets, indicating difficulty is not a significant confounder for this trend. We find this trend is stronger in instruction-tuned models, and using alternative architectures like Mamba (Gu & Dao, 2024) may not be enough to increase diversity.
this section cite: ['b31']

Section: Related Work
There is increasing interest in finding differences between models for applications like visual tools for comparative analytics (Strobelt et al., 2021;Kahng et al., 2024), efficient human evaluation (Boubdir et al., 2023), comparing learning algorithms (Shah et al., 2023), identifying side-effects of API updates (Eyuboglu et al., 2024) or quantization (Dutta et al., 2024). Prior work has also looked at qualitatively describing differences between data distributions (Zhong et al., 2022;2023;Dunlap et al., 2024;2025). Our work proposes metrics to quantify LM differences (or similarity). Huh et al. ( 2024) used representation similarity metrics (Kornblith et al., 2019;Bansal et al., 2021) to show convergence in visual representations and their alignment with language representations. In contrast, we show model mistakes are becoming more correlated as capabilities improve, using sample level evaluations (Burnell et al., 2023) such as those available on OpenLLMLeaderboard (Myrzakhan et al., 2024) and HELM (Bommasani et al., 2023). Geirhos et al. (2020) proposed measuring "error consistency" between image classifiers and humans, with Geirhos et al. (2021) showing an early trend of data-rich models making more similar mistakes to humans. We enrich this metric, distinguishing between different mistakes and incorporating probabilistic information. Bommasani et al. (2023) used a similar metric to demonstrate homogenization in outcomes when using LMs for decision making. Our work shows that such risks from algorithmic monoculture (Kleinberg & Raghavan, 2021) are increasingly relevant as LM capabilities improve.
Our results on AI judges fall in a broader line highlighting their pitfalls (Zheng et al., 2025). These include biases such as favoring verbose texts or options at certain positions (Koo et al., 2024;Ye et al., 2025). Interestingly, these biases are also sometimes found in human annotators (Chen et al., 2024). In fact, there is rich literature documenting biases in human judgements of other humans. One such bias is affinity bias, where recruiters prefer candidates with similar knowledge and skills as them (Bagues & Perez-Villadoniga, 2012). We show LM judges also systematically favor other models that make similar mistakes, generalizing previous results that showed LMs favor their own outputs (Liu et al., 2024;Panickssery et al., 2024). Overall, we believe AI evaluators should be accompanied with formal checks like consistency (Fluri et al., 2024).
A second aspect of AI oversight is using another model's supervision to train better models. This is similar to training on text generated by an LM (Chiang et al., 2023) with ongoing debates about its benefits (Kazdan et al., 2025), and an emerging paradigm of exploiting a gap in difficulty between solution generation and evaluation (Song et al., 2025). In this paper, we study the more established setup of training LMs on LM annotations, where Burns et al. (2024) demonstrated the phenomenon of weak to strong generalization, and it has been leveraged for other applications like image classification (Guo et al., 2024) and aligning models (Zhu et al., 2025). Prior work has attempted to understand weak to strong generalization, notably using "misfit error" (Charikar et al., 2024), which shows that the student's disagreement with the weak supervisor after weak to strong training correlates with its accuracy gap from the weak supervisor. Instead, we show similarity between the weak supervisor and strong student can apriori predict gains from weak-to-strong training. The benefit of model diversity has previously been discussed in related settings like knowledge distillation for image classifiers (Roth et al., 2024) and training chess models that outperform the humans they are trained on (Zhang et al., 2024).
this section cite: ['b32', 'b6', 'b22', 'b21', 'b19', 'b10', 'b42', 'b3', 'b8', 'b57', 'b5', 'b26', 'b27', 'b5', 'b38', 'b41', 'b12', 'b1', 'b49', 'b61', 'b24', 'b13', 'b34', 'b9', 'b65', 'b11']

Section: Conclusion, Limitations, Future Work
Our paper shows the importance of measuring functional similarity for language models. We derive a novel, probabilistic metric for model similarity, CAPA (κ p ). We then use it to study the implications of similarity for AI oversight -showing affinity bias in AI judges, and the role of complementary knowledge when training on LM annotations, such as in weak-to-strong generalization. AI oversight will become more relevant as capabilities improve, so our finding that increasing capabilities could lead to more correlated errors is particularly concerning. Thus, we believe measuring and accounting for model similarity is going to be increasingly important. We now list some limitations of our work and avenues for future work, that can help develop a better understanding of model similarity and its implications.
this section cite: []

Section: References
Ref_id:b0 Title: Smollm2 -with great data, comes great performance Year: (2024)
Ref_id:b1 Title: Do recruiters prefer applicants with similar skills? evidence from a randomized natural experiment Year: (2012)
Ref_id:b2 Title: Towards evaluations-based safety cases for ai scheming Year: (2024)
Ref_id:b3 Title: Revisiting model stitching to compare neural representations Year: (2021)
Ref_id:b4 Title:  Year: (2024)
Ref_id:b5 Title: Holistic evaluation of language models Year: (2023)
Ref_id:b6 Title: Which prompts make the difference? data prioritization for efficient human llm evaluation Year: (2023)
Ref_id:b7 Title:  Year: (2022)
Ref_id:b8 Title: Rethink reporting of evaluation results in ai Year: (2023)
Ref_id:b9 Title: Weak-to-strong generalization: Eliciting strong capabilities with weak supervision Year: (2024)
Ref_id:b10 Title: A portfolio approach to research funding Year: (2025)
Ref_id:b11 Title: Quantifying the gain in weak-to-strong generalization Year: (2024)
Ref_id:b12 Title: Humans or LLMs as the judge? a study on judgement bias Year: (2024)
Ref_id:b13 Title: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023-03)
Ref_id:b14 Title: The matthews correlation coefficient (mcc) is more informative than cohen's kappa and brier score in binary classification assessment Year: (2021)
Ref_id:b15 Title: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b16 Title: A coefficient of agreement for nominal scales Year: (1960)
Ref_id:b17 Title: Alpacafarm: A simulation framework for methods that learn from human feedback Year: (2023)
Ref_id:b18 Title: Length-controlled alpacaeval: A simple debiasing of automatic evaluators Year: (2024)
Ref_id:b19 Title: Describing differences in image sets with natural language Year: (2024)
Ref_id:b20 Title: Discover and quantify qualitative differences in large language models Year: (2025)
Ref_id:b21 Title: Accuracy is not all you need Year: (2024)
Ref_id:b22 Title: Model changelists: Characterizing updates to ml models Year: (2024)
Ref_id:b23 Title: The measurement of interrater agreement Year: (1981)
Ref_id:b24 Title: Evaluating superhuman models with consistency checks Year: (2024)
Ref_id:b25 Title: A framework for few-shot language model evaluation Year: (2023)
Ref_id:b26 Title: Beyond accuracy: quantifying trial-by-trial behaviour of cnns and humans by measuring error consistency Year: (2020)
Ref_id:b27 Title: Partial success in closing the gap between human and machine vision Year: (2021)
Ref_id:b28 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b29 Title: Onebench to test them all: Sample-level benchmarking over open-ended capabilities Year: (2023)
Ref_id:b30 Title: Challenges in mechanistically interpreting model representations Year: (2024)
Ref_id:b31 Title: Linear-time sequence modeling with selective state spaces Year: (2024)
Ref_id:b32 Title: Llm comparator: Visual analytics for side-byside evaluation of large language models Year: (2024)
Ref_id:b33 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b34 Title: Collapse or thrive? perils and promises of synthetic data in a selfgenerating world Year: (2025)
Ref_id:b35 Title: On scalable oversight with weak llms judging strong llms Year: (2024)
Ref_id:b36 Title: Looking beyond the surface: A challenge set for reading comprehension over multiple sentences Year: (2018)
Ref_id:b37 Title: Similarity of neural network models: A survey of functional and representational measures Year: (2025)
Ref_id:b38 Title: Algorithmic monoculture and social welfare Year: (2021)
Ref_id:b39 Title: To ship or not to ship: An extensive evaluation of automatic metrics for machine translation Year: (2021)
Ref_id:b40 Title: Composable interventions for language models Year: (2025)
Ref_id:b41 Title: Benchmarking cognitive biases in large language models as evaluators Year: (2024)
Ref_id:b42 Title: Similarity of neural network representations revisited Year: (2019)
Ref_id:b43 Title: Reliability in content analysis: Some common misconceptions and recommendations Year: (2004)
Ref_id:b44 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b45 Title: The WMDP benchmark: Measuring and reducing malicious use with unlearning Year: (2024)
Ref_id:b46 Title: From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline Year: (2024)
Ref_id:b47 Title: Textbooks are all you need ii: phi-1.5 technical report Year: (2023)
Ref_id:b48 Title: Let's verify step by step Year: (2024)
Ref_id:b49 Title: LLMs as narcissistic evaluators: When ego inflates evaluation scores Year: (2024)
Ref_id:b50 Title: The llama 3 herd of models Year: (2024)
Ref_id:b51 Title: Llama Team. Llama 3.2 model card Year: (2024)
Ref_id:b52 Title: Multi-agent actor-critic for mixed cooperative-competitive environments Year: (2017)
Ref_id:b53 Title: An adversarial perspective on machine unlearning for AI safety Year: (2025)
Ref_id:b54 Title: Aidanbench: Stress-testing language model creativity on open-ended questions Year: (2024)
Ref_id:b55 Title: Phi-4 technical report Year: (2024)
Ref_id:b56 Title: Ministral 8b instruct model card Year: (2024)
Ref_id:b57 Title: Open-llmleaderboard: From multi-choice to open-style questions for llms evaluation, benchmark, and arena Year: (2024)
Ref_id:b58 Title: Adversarial NLI: A new benchmark for natural language understanding Year: (2020)
Ref_id:b59 Title:  Year: (2024)
Ref_id:b60 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b61 Title: LLM evaluators recognize and favor their own generations Year: (2024)
Ref_id:b62 Title: Bleu: a method for automatic evaluation of machine translation Year: (2002)
Ref_id:b63 Title: Goal driven discovery of distributional differences via language descriptions Year: (2023)
Ref_id:b64 Title: going on a vacation" takes longer than "going for a walk": A study of temporal commonsense understanding Year: (2019)
Ref_id:b65 Title: Weakto-strong preference optimization: Stealing reward from weak aligned model Year: (2025)
Ref_id:b66 Title: Extended Multiple Regression Analysis with Model Size as a Year: ()
Ref_id:b67 Title:  Year: ()
Ref_id:b68 Title:  Year: ()
Ref_id:b69 Title:  Year: ()
Ref_id:b70 Title:  Year: ()
Ref_id:b71 Title: Judge Score Validity Against Reference Year: ()
Ref_id:b72 Title:  Year: ()
Ref_id:b73 Title:  Year: ()
Ref_id:b74 Title:  Year: ()
Ref_id:b75 Title:  Year: ()
Ref_id:b76 Title:  Year: ()
Ref_id:b77 Title:  Year: ()
Ref_id:b78 Title: Experimental Setup to Perform Free-Form Inference on Filtered MMLU Year: ()
Ref_id:b79 Title: Experimental Setup for LLM-as-a-Judge on Filtered MMLU Year: ()
Ref_id:b80 Title:  Year: ()
Ref_id:b81 Title:  Year: ()
Ref_id:b82 Title:  Year: ()
Ref_id:b83 Title:  Year: ()
Ref_id:b84 Title:  Year: ()
Ref_id:b85 Title:  Year: ()
Ref_id:b86 Title: Table 7. LMs used as LLM-as-a-Judge Judge Model Name google/gemma-2-9b-it Year: (2024)
Ref_id:b87 Title: 2025) meta-llama/Meta-Llama-3.1-8B-Instruct (Llama Team, 2024a) meta-llama/Meta-Llama-3.1-70B-Instruct (Llama Team, 2024a) meta-llama/Llama-3.3-70B-Instruct (Llama Team Year: (2024)
Ref_id:b88 Title: LMs Evaluated on the Filtered MMLU-Pro Benchmark Model Name Base Models Instruction-tuned Models Year: ()
Ref_id:b89 Title:  Year: (2024)
Ref_id:b90 Title: -8B-Instruct meta-llama/Meta-Llama-3.1-70B meta-llama/Meta-Llama-3.1-70B-Instruct meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.3-70B-Instruct Phi-4 Family Year: ()
Ref_id:b91 Title: Qwen2.5 Family Year: (2025)
Ref_id:b92 Title: B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2 Year: ()
