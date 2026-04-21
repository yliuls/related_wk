Title: Adaptive Prediction-Powered AutoEval with Reliability and Efficiency Guarantees
Abstract: Selecting artificial intelligence (AI) models, such as large language models (LLMs), from multiple candidates requires accurate performance estimation. This is ideally achieved through empirical evaluations involving abundant real-world data. However, such evaluations are costly and impractical at scale. To address this challenge, autoevaluation methods leverage synthetic data produced by automated evaluators, such as LLMs-as-judges, reducing variance but potentially introducing bias. Recent approaches have employed semi-supervised prediction-powered inference (PPI) to correct for the bias of autoevaluators. However, the use of autoevaluators may lead in practice to a degradation in sample efficiency compared to conventional methods using only real-world data. In this paper, we propose R-AutoEval+, a novel framework that provides finite-sample reliability guarantees on the model evaluation, while also ensuring an enhanced (or at least no worse) sample efficiency compared to conventional methods. The key innovation of R-AutoEval+ is an adaptive construction of the model evaluation variable, which dynamically tunes its reliance on synthetic data, reverting to conventional methods when the autoevaluator is insufficiently accurate. Experiments on the use of LLMs-as-judges for the optimization of quantization settings for the weights of an LLM, for prompt design in LLMs, and for test-time reasoning budget allocation in LLMs confirm the reliability and efficiency of R-AutoEval+.

Section: Introduction 1.Context and Motivation
Selecting an artificial intelligence (AI) model among multiple candidates necessitates accurately estimating each model's performance. Typically, performance assessment involves actively employing each model to gather relevant empirical evidence or data. To mitigate the substantial cost and practical burden of real-world testing, autoevaluation leverages automated tools to evaluate model performance without direct human intervention [35,31,12,8,52,5].
Standard evaluation based on human judgment -referred to as Eval -and autoevaluation -AutoEval -each present distinct advantages and drawbacks. Eval provides unbiased estimates of a model's performance but requires costly annotation. In contrast, AutoEval is cheaper, as it can rely on abundant synthetic data, but it may introduce estimation bias [35,12]. Consequently, neither approach, in isolation, ensures reliable model evaluation, potentially leading to erroneous evaluation outcomes.
Reliability in evaluation methods can be established via confidence intervals that accurately capture the true expected performance at a specified coverage level, or through testing strategies that detect whether target performance levels are met at specified false detection probabilities. For Eval, both Figure 1: How to select the lightest quantized Llama-3.1-8B-Instruct model [24] (in the MX quantization format [41]) that guarantees up to 10% performance drop as compared to the unquantized version (BF16) (for the TriviaQA task [28])? (left) Ground-truth risk R for different MX quantization settings, requiring massive human-labeled data. (right) Performance drop and corresponding model size for the models chosen via Eval, AutoEval [35,31,52], R-Eval [47], R-AutoEval [20], and the proposed R-AutoEval+. We adopt Llama-3.3-70B-Instruct [24] BF16/MX6/MX4 as the autoevaluators, and set target risk in (1) to α = 0.1 and target reliability in (2) to 1 -δ = 0.9. Maximum values are reported within the 1.5 interquartile range (IQR) range [34] across 500 independent experiments (see Sec. 4 for details).
approaches can be implemented using standard statistical methods [26,33,47], which we refer to as R-Eval.
this section cite: ['b34', 'b30', 'b11', 'b7', 'b51', 'b4', 'b34', 'b11', 'b23', 'b40', 'b27', 'b34', 'b30', 'b51', 'b46', 'b19', 'b23', 'b33', 'b25', 'b32', 'b46']

Section: Achieving reliability in
AutoEval is more challenging. Recent works, including [10,22,21,42], have successfully applied a semi-supervised inference framework known as prediction-powered inference (PPI) [3,4] to correct for the inherent bias of AutoEval by leveraging a small amount of human-labeled, real-world data. These methodologies, referred to here as R-AutoEval, either guarantee reliability only asymptotically -as the sizes of both synthetic and real-world data sets grow indefinitely [10,22,21] -or lack explicit sample efficiency guarantees -i.e., they do not provably yield narrower confidence intervals or higher test powers compared to R-Eval [20].
In this paper, we introduce R-AutoEval+, a novel autoevaluation framework that provides finitesample (non-asymptotic) reliability guarantees while also ensuring improved (or at least no worse) sample efficiency compared to R-AutoEval. The primary innovation of R-AutoEval+ is its sequential construction of the model evaluation variable, enabling it to adaptively adjust its reliance on synthetic data based on evolving assessments of the autoevaluator's quality. R-AutoEval+ seamlessly reverts to R-Eval when synthetic data are deemed to be of insufficient quality, while otherwise employing a weighted variant of R-AutoEval to enhance efficiency.
At a technical level, R-AutoEval+ leverages two primary methodologies. The first is the gametheoretic testing-by-betting approach [43,38] to mean estimation introduced by [47]. The second is PPI++ [4], an enhanced variant of PPI that incorporates a regularization coefficient to control the estimator's dependence on autoevaluator-generated data [4].
this section cite: ['b9', 'b21', 'b20', 'b41', 'b2', 'b3', 'b9', 'b21', 'b20', 'b19', 'b42', 'b37', 'b46', 'b3', 'b3']

Section: Overview of the Main Results
To clarify the main concepts, consider the problem of selecting a large language model (LLM) from multiple candidate models characterized by different sizes. These models are derived from a single base LLM through quantization with varying average bitwidths using the MX format [41]. As illustrated in left panel of Fig. 1, our goal is to identify models that maintain performance comparable to the full-precision baseline, tolerating at most a predefined performance degradation threshold α (e.g., α = 10% in the figure).
Formally, consider a bounded loss function ℓ(X, Y ) ∈ [0, 1] that measures the performance of a given candidate model. In the context of Fig. 1, this loss quantifies the performance gap between the candidate quantized LLM and its full-precision counterpart. Our primary objective is to verify whether or not the expected loss, or risk, defined as R = E[ℓ(X, Y )], exceeds a target level α, i.e., whether the following risk-controlling condition can be satisfied R ≤ α.
(1) The challenge is that evaluating the risk R necessitates access to a large number of human-generated, i.e., real-world, responses Y corresponding to queries X.
this section cite: ['b40']

Section: Eval and AutoEval
Given n pairs of real-world data
D n = {(X i , Y i )} n i=1 with (X i , Y i ) i.i.d.
∼ P XY = P X P Y |X for i = 1, ..., n, the standard Eval approach estimates the risk R for any given candidate model via the empirical average
Rn = (1/n) n i=1 ℓ(X i , Y i ).
In contrast, AutoEval does not assume access to real data D n , relying instead the availability of a pre-trained autoevaluator f : X → Y, where X and Y denote the respective domains of input and output X, Y . In the example of Fig. 1, the autoevaluator is a larger LLM, serving as an LLM judge [52]. Specifically, given an unlabeled data
D unl N = { Xi } N i=1 with Xi i.i.d. ∼ P X , AutoEval estimates the risk as Rf N = (1/N ) N i=1 ℓ( Xi , f ( Xi ))
, thus using the outputs from the LLM judge as labels [35,31,52].
Lacking formal uncertainty quantification, the risk-controlling condition (1) cannot be guaranteed via the risk estimates Rn and Rf N provided by Eval and AutoEval. That is, even if the evaluation outcome for a candidate model satisfies the inequality Rn ≤ α, or Rf N ≤ α, its actual performance may still fail to meet the condition (1). In the example shown in Fig. 1, LLMs selected via Eval and AutoEval exhibit performance degradations that exceed the target threshold of α = 10%. Furthermore, for AutoEval, the performance degradation becomes larger as the quality of the LLM judge deteriorates.
this section cite: ['b51', 'b34', 'b30', 'b51']

Section: R-Eval and R-AutoEval
Reliable Eval (R-Eval) [47] and Reliable AutoEval (R-AutoEval) [20] endow Eval and AutoEval, respectively, with reliability guarantees by formulating model evaluation as a binary hypothesis test problem. Accordingly, these evaluation protocols test the null hypothesis H 0 : R > α that the model's risk exceeds the target α against the alternative hypothesis H 1 : R ≤ α that the risk-controlling condition (1) is satisfied.
Define as T n ∈ {0, 1} the test output, with T n = 1 indicating the decision for the alternative hypothesis H 1 : R ≤ α. Then, an evaluation procedure is said to be reliable at level 1 -δ if the probability of incorrectly concluding that a candidate model satisfies the requirement (1) does not exceed the level δ ∈ (0, 1), i.e.,
Pr T n = 1|R > α ≤ δ.(2)
While Eval constructs the test decision based only on real data D n , R-AutoEval incorporates also synthetic data by leveraging PPI [3] as we detail in Sec. 2.
As illustrated in Fig. 1, selecting the smallest LLM among the candidates deemed risk-controlling by R-Eval and R-AutoEval, i.e., the models with respective testing results being T n = 1, indeed results in a performance drop that remains within the tolerated risk level of α = 10%. However, as also shown in Fig. 1, R-AutoEval may select a model with a larger size than R-Eval, highlighting the potential inefficiency due to the use of an LLM judge.
this section cite: ['b46', 'b19', 'b2']

Section: R-AutoEval+
This paper proposes R-AutoEval+, a novel reliable autoevaluation method that adaptively tunes its reliance on synthetic data based on an evolving reliability assessments of the autoevaluator. This approach balances synthetic and real data, reverting to conventional methods when the accuracy of the autoevaluator is insufficient, while maintaining rigorous statistical guarantees.
R-AutoEval+ is not only reliable in the sense of satisfying the condition (2), but it also provides a guaranteed improvement (possibly not strict) in terms of sample efficiency. To formalize this property, define the sample complexity of an evaluation method producing test variable T n as the smallest average size of the real-world data set D n necessary to conclude that a model satisfies the requirement (1) under the reliability constraint (2), i.e., [48] n min (δ) = E[min{n :
T n = 1}|R ≤ α].(3)
An evaluation scheme with smaller sample complexity n min (δ) can generally identify more efficient candidate models with the same amount of real data. The main result is informally outlined as follows.
Theorem 1 (Informal). Under mild regularity assumptions, for sufficiently low tolerated unreliability level δ, R-AutoEval+ is provably more sample efficient than both R-Eval [47] and R-AutoEval [20], i.e.,
n R-AutoEval+ min (δ) ≤ min n R-Eval min (δ), n R-AutoEval min (δ) .(4)
Furthermore, this inequality is strict when the autoevaluator is sufficiently accurate.
this section cite: ['b47', 'b46', 'b19']

Section: Preliminaries: R-Eval and R-AutoEval
In this section, we first review the testing-by-betting framework [43,37,47,48,38], and then review R-Eval [47] and R-AutoEval [20]. Further connections to the state-of-the-art can be found in Appendix A.
this section cite: ['b42', 'b36', 'b46', 'b47', 'b37', 'b46', 'b19']

Section: Testing-By-Betting
Consider the problem of testing the null hypothesis H 0 : R > α against the alternative hypothesis H 1 : R ≤ α based on the observations of n i.i.d. bounded random variables {q i } n i=1 with q i ∈ [m, M ] providing unbiased estimates of the risk, i.e., E[q i ] = R. An e-value E n is a nonnegative statistic of the observations {q i } n i=1 whose expectation under the null hypothesis does not exceed 1, i.e., E[E n |R > α] ≤ 1 [45]. An e-value E n can be interpreted as providing evidence in favor of the alternative hypothesis H 1 , i.e., of the risk-controlling condition (1) [43]. With an e-value E n , the test
T n = 1(E n ≥ 1/δ)(5)
meets the reliability condition (2) due to Markov's inequality for a fixed n.
The testing-by-betting approach constructs an e-value E n sequentially by processing the observations q i one by one over index i = 1, ..., n. Specifically, an e-value can be obtained via the product of the contributions of each observation q i as [47,6]
E n = n i=1 1 -λ i (q i -α) ,(6)
where E 0 = 1 and λ i ∈ [0, 1/(M -α)) is an arbitrary function of the past observations {q j } i-1 j=1 . To verify that the quantify (6) is an e-value, one can use the independence of the observations {q i } n i=1 and the unbiasedness assumption
E[q i ] = R as E[E n |R > α] = n i=1 (1 -λ i (E[q i |R > α] -α)) ≤ 1.
The e-value has an interpretation in terms of a sequential betting game, which supports the design of the sequence of bets λ i using online convex optimization [47]. In this game, at each round i, based on the observations {q j } i-1 j=1 , a gambler bets an amount of her wealth measured by variable λ i on the outcome q i ≤ α that the next observation q i does not exceed the target α. Accordingly, the e-value E n in (6) represents the wealth accumulated after n rounds of betting with initial wealth E 0 = 1 [47].
A more general form of e-value has been also proposed that allows the gambler to have S ≥ 1 different betting strategies {λ s,i } S s=1 , with each strategy λ s,i being responsible for a fraction w s,i of the wealth. To elaborate, fix a probability vector w i = [w 1,i , ..., w S,i ], which, like the betting strategies {λ s,i } S s=1 , may depend on the past observations {q j } i-1 j=1 . The resulting e-value is defined as the convex combination [47, B.8]
E n = n i=1 S s=1 w s,i 1 -λ s,i (q i -α) .(7)
this section cite: ['b44', 'b42', 'b46', 'b5', 'b46', 'b46']

Section: R-Eval and R-AutoEval
Both R-Eval and R-AutoEval follow the testing-by-betting approach presented in the previous subsection. Specifically, R-Eval computes the e-value E R-Eval n in (6) using directly the observations i that incorporate both real data and synthetic data. In particular, each effective observation
q i = ℓ i of the losses ℓ i = ℓ(X i , Y i ) ∈ [m = 0, M = 1], obtaining the decision T R-Eval n in (5).
this section cite: []

Section: R-AutoEval
q i = ℓ f i uses the real data point (X i , Y i ) ∈ D n together with r = ⌊N/n⌋ autoevaluated samples {( Xi , f ( Xi ))} r•i r•(i-1)+1
with Xi ∈ DN . Note that this divides the set of unlabeled samples DN equally across the effective observations. Specifically, R-AutoEval uses effective observations obtained via PPI [3] by correcting the bias of the empirical autoevaluated risk using the real data sample as
ℓ f i = 1 r r•i i ′ =r•(i-1)+1 ℓ( Xi ′ , f ( Xi ′ )) autoevaluator data + ℓ(X i , Y i ) -ℓ(X i , f (X i )) bias correction .(8)
One can readily check that this is an unbiased estimate of the risk, i.e., E[ℓ f i ] = R, with support interval ℓ f i ∈ [m = -1, M = 2]. However, owing to the possible lower quality of the autoevaluator's labels f ( Xi ), using the effective observations (8) may result in a larger sample complexity compared to R-Eval (see Fig. 1).
this section cite: ['b2']

Section: R-AutoEval+: AutoEval with Reliability and Efficiency Guarantees
In this section, we present the proposed autoevaluation method, R-AutoEval+. R-AutoEval+ aims at balancing the importance assigned to synthetic and real-world data in the effective observations (8). Through the proposed mechanism, R-AutoEval+ reverts to using the effective observations q i = ℓ i of R-Eval when the autoevaluator is inaccurate, while leveraging the effective observations of R-AutoEval, q i = ℓ f i , when the autoevaluator is very accurate. The approach is based on an adaptive construction of the effective observations used in the e-value statistic (7), dynamically tuning the reliance on synthetic data based on evolving reliability assessments of the autoevaluator.
The key idea is to weight the contribution of synthetic data in the e-value with a factor ρ ∈ [0, 1], so that setting ρ = 0 recovers AutoEval, while setting ρ = 1 yields R-AutoEval. In order to automatically identify the best value of the factor ρ, R-AutoEval+ processes the real-world data samples sequentially across index i = 1, ..., n, tracking the performance of S possible candidate values {ρ s } S s=1 , with
0 = ρ 1 < • • • < ρ S = 1.(9)
Specifically, R-AutoEval+ maintains adaptive weights {w s,i } S s=1 across index i = 1, ..., n, with weight w s,i associated to candidate value ρ s . The resulting effective observations are used in (7).
After providing a more detailed description of R-AutoEval+, this section demonstrates that R-AutoEval+ can provably enhance the sample efficiency of both R-Eval and R-AutoEval.
this section cite: ['b7', 'b6', 'b6']

Section: Adaptive Effective Observations and E-Values
At each round i, R-AutoEval+ computes S effective observations {ℓ f s,i } S s=1 , with each effective observation ℓ f s,i weighting the contribution of synthetic data via the factor ρ s . In particular, generalizing (8) via PPI++ [4], each s-th effective observation is given by
ℓ f s,i = ρ s r r•i i ′ =r•(i-1)+1 ℓ( Xi ′ , f ( Xi ′ )) autoevaluator data + ℓ(X i , Y i ) -ρ s • ℓ(X i , f (X i )) bias correction ,(10)
where the factor ρ s multiplies the contributions of the autoevaluator. The observation ( 10
E n = n i=1 S s=1 w s,i 1 -λ s,i (ℓ f s,i -α) . (11
)
Note that this is indeed a valid e-value, i.e., E[E n |R > α] ≤ 1. The autoevaluator reports the correct loss with probability γ = 0.99 (top), γ = 0.9 (middle), and γ = 0.7 (bottom). R-AutoEval+ assigns larger weights to synthetic data, i.e., to larger values of ρ s , when the autoevaluator is of higher quality.
In the e-value (11), the weights w s,i associated with each factor ρ s are updated over index i = 1, ..., n depending on the evidence accumulated up to round i -1. To do this, define as
E s,i = i j=1 1 -λ s,j (ℓ f s,j -α) (12
)
the e-value (6) computed using only the effective observations ℓ f s,i up to round i. Intuitively, a larger value of the quantity E s,i indicates that the factor ρ s yields a large evidence in favor of the alternative (risk-controlling) hypothesis (1).
Following this logic, the weight w s,i is updated as
w s,i = w s,0 • E s,i-1 S s ′ =1 w s ′ ,0 • E s ′ ,i-1 , (13
)
for all s = 1, ..., S, with initial strictly positive weights {w s,0 } S s=1 , where The update (13) aims at identifying the most informative effective observations ℓ f s,i for s = 1, ..., S by sequentially processing the available data. The following simple example demonstrates its operation. Example 1. Consider a setting where the autoevaluator yields the same output as the human judge with probability γ. Specifically, assume a binary loss ℓ(X, Y ) ∈ {0, 1} with mean equal to the risk R = 0.1, and on autoevaluated loss ℓ(X, f (X)) given by ℓ(
X, f (X)) = ℓ(X, Y ) • (1 -ϵ) + (1 - ℓ(X, Y ))
• ϵ, where ϵ ∈ {0, 1} has mean 1 -γ. Fix the target risk α = 0.12 and the synthetic-to-real data ratio r = 10.
Fig. 2 provides an heatmap of the evolution of the weights {w s,i } S s=1 over index i for S = 100 uniformly spaced candidate factors in the range [0, 1]. We set initial weights w s,0 = 1/S for s = 1, ..., S. The top figure corresponds to γ = 0.99, the middle figure to γ = 0.9 and the bottom figure to γ = 0.7, so that the quality of the autoevaluator decreases in going from the top panel to the bottom panel. The figure shows that, as the autoevaluator becomes less reliable, the update (13) correctly decreases the reliance of the e-value (11) on the effective observations that leverage synthetic data. Specifically, it is seen from the figure that the weights {w s,i } S s=1 tend to concentrate around the values 0.9, 0.5, and 0 for γ = 0.99, 0.9, and 0.7, respectively.
The overall procedure of R-AutoEval+ is summarized in Algorithm 1 (Appendix B).
this section cite: ['b3']

Section: Sample Efficiency Guarantees
In this section, we analyze the sample efficiency of R-AutoEval+. We start by reviewing existing results on R-Eval and R-AutoEval, and then we introduce a formal version of Theorem 1 (see Sec. 1) on the sample efficiency of R-AutoEval+.
this section cite: []

Section: Sample complexity of R-Eval and R-AutoEval
Reference [48] showed that the sample complexity of the e-value (6) can be analyzed by considering its maximum expected logarithmic per-round increment under the alternative hypothesis, i.e.,
g ⋆ = E[log(1 -λ ⋆ (q i -α))|R ≤ α],(14)
where λ ⋆ is the optimal constant betting variable
λ ⋆ = arg max λ∈[0,1/(M -α)) E[log(1 -λ(q i - α))|R ≤ α]. Specifically, defining E n (λ ⋆ ) = n i=1 (1 -λ ⋆ (q i -α))
for the e-value (6) with constant λ i = λ ⋆ for all rounds i = 1, ..., n, we have the following result.
this section cite: ['b47']

Section: Theorem 2 (Sample complexity of testing-by-betting (6) [48, Theorem 3.3]).
Assume that: (A1) the betting strategy λ i admits sublinear regret with respect to the optimal constant λ ⋆ , i.e., log E n (λ ⋆ )log E n = o(n) (see Appendix C, for an example of such betting strategy), and (A2) the instantaneous logarithmic increment of the e-value E n (λ ⋆ ) has finite σ-th central moment for some σ > 2, i.e.,
E[| log(1 -λ ⋆ • (q i -α)) -g ⋆ | σ |R ≤ α] < ∞.
Then, the sample complexity (3) obtained by the test variable T n in (5) with the e-value E n in (6) admits the limit
lim δ→0 + n min (δ) log(1/δ) = 1 g ⋆ .(15)
To interpret this result, consider the second-order Taylor approximation log(1 -y) ≃ -y -y 2 /2, which yields (see also [48, Sec. 4.2])
1 g ⋆ ≃ 2 1 + Var(q i |R ≤ α) (α -R) 2 (16
)
According to this approximation, the sample complexity grows with the variance Var(
q i |R ≤ α) = E[|q i -R| 2 |R ≤ α]
of the observation under the alternative hypothesis.
The result (15) can be readily applied to obtain the sample complexities of R-Eval and R-AutoEval by setting q i = ℓ i and q i = ℓ f i , respectively, in the e-value (6). In the presence of a low-quality autoevaluator, it is known that PPI can increase the variance of the risk estimates [3,4,22,10]. Therefore, based on the approximation (16), R-AutoEval may have a larger sample complexity than R-Eval when the autoevaluation is not sufficiently accurate.
this section cite: ['b2', 'b3', 'b21', 'b9']

Section: Sample complexity of R-AutoEval+ R-AutoEval+ optimizes not only the S betting strategies {λ s,i } S
s=1 , but also the weights {w s,i } S s=1 associated to the factors {ρ s } S s=1 determining the reliance of the effective observations (10) on the autoevaluated data. Since the weight update (13) has the form of exponential weights [23], the following property follows from existing results on online convex optimization [18, Lemma 1]. Lemma 1 (Sublinear regret for the weights (13)). For any set of betting strategies {λ s,i } S s=1 and for any positive initial weights {w s,0 } S s=1 , the weight update strategy (13) satisfies
max s=1,...,S n i=1 log 1 -λ s,i (ℓ f s,i -α) log Es,n - n i=1 log S s=1 w s,i 1 -λ s,i (ℓ f s,i -α) log En ≤ max s=1,...,S log 1 w s,0 .(17)
Intuitively, this result implies that the weight update (13) can identify the best factor ρ s in the set {ρ s } S s=1 , determining an optimal level of reliance on autoevaluated data. In fact, the first term in (17) represents the maximum, i.e., most informative, e-value among all the e-values {E s,n } S s=1 corresponding to the S candidate factors {ρ s } S s=1 . This result, in turn, suggests that R-AutoEval+ may be able to outperform both R-Eval and R-AutoEval, reducing to the former when autoevaluated data is of poor quality and to the latter when the autoevaluated data is sufficiently accurate. The following result formalizes this intuition.
this section cite: ['b9', 'b22']

Section: Theorem 3 (Sample complexity of R-AutoEval+).
For every s = 1, ..., S, suppose that the betting strategy λ s,i satisfies assumptions (A1) and (A2) in Theorem 2 with E s,n in lieu of E n and q i = ℓ f s,i . Then, the sample complexity of R-AutoEval+ satisfies the following limit
lim δ→0 + n R-AutoEval+ min (δ) log(1/δ) ≤ min s=1,...,S 1 g s,⋆ . (18
)
Proof. See Appendix F.2.
The limit (18) implies the main result (4) in Theorem 1. In fact, by (15), the ratio 1/g s,⋆ in (18) corresponds to the scaling of the sample complexity of R-Eval and R-AutoEval by setting s = 1 and s = S, respectively. The next example shows that the inequality in (4) can be strict. Example 2. Consider again the example setting in Example 1. The sample complexity n min (δ) is plotted in the top part of Fig. 3 as a function of log(1/δ) for α = 0.12, S = 10, r = 10, and for (a) γ = 0.99, (b) γ = 0.9, and (c) γ = 0.7. The results are averaged over 100 independent experiments and we use the universal portfolio betting strategy (see Appendix C) [15, 16, 48]. The figure confirms the linear trend of the sample complexity with respect to the term log(1/δ). Furthermore, it shows that the inequality (4) can indeed be strict, with R-AutoEval+ outperforming both R-Eval and R-AutoEval. The bottom part of the figure plots the maximum expected logarithmic increment of the e-value, g s,⋆ , in (18) as a function of the factor ρ s for (a) γ = 0.99, (b) γ = 0.9, (c) γ = 0.7. As the autoevaluator becomes less (more) reliable, i.e., as γ decreases (increases), the maximum value of the expected increment g s,⋆ is obtained for values of ρ s closer to zero (one), making R-AutoEval+ behave as R-Eval (R-AutoEval). More generally, using the approximation (16), one can conclude that the sample complexity of R-AutoEval+ is strictly smaller than for R-Eval and R-AutoEval as long as the variance of the effective observation ℓ f s,i in (10) for some index s different from s = 1 and s = S is strictly smaller than for the effective observations ℓ i and ℓ f i in (8). As shown in [4, Example 6.1], this condition is satisfied when the autoevaluator is sufficiently accurate.
this section cite: []

Section: Experimental Results 1
For experimental validation, we consider three model selection applications: 1) selecting the lightest quantized LLM with guaranteed performance drop as compared to the baseline model on the TriviaQA data set [28] (see Fig. 1); 2) selecting the shortest prompt template for an LLM with guaranteed accuracy on the Instruct-Induction task [27]; and 3) test-time reasoning budget allocation with guaranteed performance enhancement on the GSM8K data set [13]. We set S = 10 with ρ s being uniformly spaced in the range [0, 1] and choose initial weights as w s,0 = 1/S. We refer to Appendix E for results with different choices of such hyperparameters. All the results in this section are reported after averaging over 100 independent experiments, and 2 H100 GPUs are used for LLM executions. the MX format [40]. The candidate formats have different configurations specified by parameters [k 1 , k 2 , d 1 , d 2 , M] ∈ N 5 where k 1 , k 2 are the first, second block granularity levels, d 1 , d 2 are the first, second scale bit-width levels, and M is the mantissa bit-width [40,Table II]. We set k 1 ∈ {16, 64}, d 1 = 8, d 2 = 1, M ∈ {3, 4, ..., 10} with k 1 /k 2 = {2, 4, 8}. Model selection is carried out based on fixed sequence testing (FST) [7], visiting the candidates in order of decreasing average bitwidth. FST guarantees the family-wise error rate (FWER) with error probability no larger than δ.
For the autoevaluator, we adopt a larger Llama-3.1-70B-Instruct [24], whose quality is controlled by adjusting the weight precision of the autoevaluator from full-precision BF16 to MX6/MX4, with average bitwidth decreasing from 16 to 6/4 [41]. We set δ = 0.1, α = 0.1, n = 150, and r = 5 for Fig. 1 while vary n from 100 to 300 with r = 3 for Fig. 4. Fig. 1 reports maximum values within the 1.5 interquartile range (IQR) range [34] across 500 independent experiments (the corresponding full box plot can be found in Appendix E). 2) Selecting the shortest prompt template: For this second task, the candidate set consists of 25 prompt templates designed to enhance the zero-shot performance of Llama-3.1-8B-Instruct using the larger Llama-3.1-70B-Instruct [24] via the forward mode generation of automatic prompt engineering (APE) [53]. Model selection is carried out via the Bonferroni correction [9], thus applying the test (5) with δ/25 in lieu of δ.
For the autoevaluator, we adopt in-context learning [11] on the Llama-3.1-70B-Instruct with prompt examples randomly chosen from the same held-out data set used for APE. The accuracy of the autoevaluator is controlled by varying the number of prompt examples from 1 to 7. We set δ = 0.1, n = 200, r = 9, with α chosen as the minimum value in the set {0.05, 0.1, ..., 0.95} for which R-AutoEval [20] finds at least one reliable prompt template with the strongest autoevaluator. We select the longest prompt template if the model selection algorithm does not select any template.  3) Test-time reasoning budget allocation: For the last task, the candidate set consists of computation budgets for the reasoning mode of the Qwen3-1.7B [51] base model, varying between 128 and 1280 tokens. Model selection is carried out via FST, visiting the candidates in order of decreasing reasoning budget. For the autoevaluator, we adopt different kinds of pre-trained LLMs, ranging from large-scale models such as GPT-4.1 [1] to light-weight models such as BitNet b1.58 [32]. We set δ = 0.1, n = 1000, r = 4, and α = 0.03 (i.e., reasoning should improve accuracy by at least 3%).
Table 1 confirms again the efficiency gain of R-AutoEval+ as compared to R-Eval and R-AutoEval, saving up to 127 tokens over R-Eval and up to 66 tokens over R-AutoEval on average. Choosing the autoevaluator from the same family of the model is seen to substantially reduce the gain of autoeval-based approaches. For instance, Llama-3.2-3B-Instruct autoevaluator achieves much lower accuracy than Qwen3-32B autoevaluator (66% vs. 82%) but it significantly helps reducing the number of reasoning tokens: R-AutoEval+ saves 90 tokens over R-Eval when using Llama-3.2-3B-Instruct autoevaluator, while it saves 42 tokens over R-Eval when using Qwen3-32B autoevaluator; R-AutoEval saves 83 tokens over R-Eval when using Llama-3.2-3B-Instruct autoevaluator, while it requires 24 tokens more than R-Eval when using Qwen3-32B autoevaluator. Such behavior can be understood as a consequence of positive feedback of LLM judges within the same family [52], also known as preference leakage [30], which makes the bias correction (8) more challenging.
We refer to Appendix E for further details and additional experiments.
this section cite: ['b27', 'b26', 'b12', 'b39', 'b39', 'b6', 'b23', 'b40', 'b33', 'b23', 'b52', 'b8', 'b10', 'b19', 'b50', 'b0', 'b31', 'b51', 'b29']

Section: Conclusion and Further Discussions
This work introduced R-AutoEval+, a novel autoevaluation method that can provably enhance the efficiency of the state-of-the-art evaluation method while maintaining strict finite-sample reliability guarantees. The theoretical properties of R-AutoEval+ were confirmed by experimental results on LLM quantization and LLM prompting with LLM judges as autoevaluators.
Some limitations of this work are as follows: (i) R-AutoEval+ requires access to real-world unlabeled data; (ii) the discrete set of candidate factors determining reliance on synthetic data are fixed a priori; and lastly, (iii) the sample efficiency guarantee in Theorem 3 only holds for sufficiently high target reliability levels 1 -δ. Addressing these limitations may leverage the tools in [3,16,36], and we leave these directions to future work.
Another interesting direction for future research includes combining the benefits of R-AutoEval+ in adaptively weighting synthetic data with the complementary advantages of methods that actively select real data [49,46,54,17,50].
this section cite: ['b2', 'b15', 'b35', 'b48', 'b45', 'b53', 'b16', 'b49']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Learn then test: Calibrating predictive algorithms to achieve risk control Year: (2021)
Ref_id:b2 Title: Prediction-powered inference Year: (2023)
Ref_id:b3 Title: Ppi++: Efficient predictionpowered inference Year: (2023)
Ref_id:b4 Title: Reference-guided verdict: Llms-as-judges in automatic evaluation of free-form text Year: (2024)
Ref_id:b5 Title: Distribution-free, risk-controlling prediction sets Year: (2021)
Ref_id:b6 Title: Multiple testing in clinical trials Year: (1991)
Ref_id:b7 Title: Attributed question answering: Evaluation and modeling for attributed large language models Year: (2022)
Ref_id:b8 Title: Teoria statistica delle classi e calcolo delle probabilita Year: (1936)
Ref_id:b9 Title: Autoeval done right: Using synthetic data for model evaluation Year: (2024)
Ref_id:b10 Title: Language models are few-shot learners Year: (2020)
Ref_id:b11 Title: The price of debiasing automatic metrics in natural language evaluation Year: (2018)
Ref_id:b12 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b13 Title: Active learning with statistical models Year: (1996)
Ref_id:b14 Title: Universal portfolios Year: (1991)
Ref_id:b15 Title: Universal portfolios with side information Year: (2002)
Ref_id:b16 Title:  Year: (2025)
Ref_id:b17 Title: Follow the leader if you can, hedge if you must Year: (2014)
Ref_id:b18 Title: The llama 3 herd of models. arXiv e-prints Year: (2024)
Ref_id:b19 Title: Semi-supervised risk control via prediction-powered inference Year: (2024)
Ref_id:b20 Title: Auto-evaluation with few labels through post-hoc regression Year: (2024)
Ref_id:b21 Title: Stratified prediction-powered inference for hybrid language model evaluation Year: (2024)
Ref_id:b22 Title: A decision-theoretic generalization of on-line learning and an application to boosting Year: (1997)
Ref_id:b23 Title: The llama 3 herd of models Year: (2024)
Ref_id:b24 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b25 Title: Probability inequalities for sums of bounded random variables. The collected works of Wassily Hoeffding Year: (1994)
Ref_id:b26 Title: Instruction induction: From few examples to natural language task descriptions Year: (2022)
Ref_id:b27 Title: Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension Year: (2017)
Ref_id:b28 Title: Efficient algorithms for universal portfolios Year: (2002-11)
Ref_id:b29 Title: Preference leakage: A contamination problem in llm-as-a-judge Year: (2025)
Ref_id:b30 Title: How not to evaluate your dialogue system: An empirical study of unsupervised evaluation metrics for dialogue response generation Year: (2016)
Ref_id:b31 Title: Bitnet b1. 58 2b4t technical report Year: (2025)
Ref_id:b32 Title: Empirical bernstein bounds and sample variance penalization Year: (2009)
Ref_id:b33 Title: Variations of box plots Year: (1978)
Ref_id:b34 Title: Amanda Cercas Curry, and Verena Rieser. Why we need new evaluation metrics for nlg Year: (2017)
Ref_id:b35 Title: Tight concentrations and confidence sequences from the regret of universal portfolio Year: (2023)
Ref_id:b36 Title: Hypothesis testing with e-values Year: (2024)
Ref_id:b37 Title: Game-theoretic statistics and safe anytime-valid inference Year: (2023)
Ref_id:b38 Title: Coqa: A conversational question answering challenge Year: (2019)
Ref_id:b39 Title: With shared microexponents, a little shifting goes a long way Year: (2023)
Ref_id:b40 Title: Microscaling data formats for deep learning Year: (2023)
Ref_id:b41 Title: Hyperband-based bayesian optimization for black-box prompt selection Year: (2024)
Ref_id:b42 Title: Testing by betting: A strategy for statistical and scientific communication Year: (2021)
Ref_id:b43 Title: Etude critique de la notion de collectif Year: (1939)
Ref_id:b44 Title: E-values: Calibration, combination and applications Year: (2021)
Ref_id:b45 Title: Cer-eval: Certifiable and cost-efficient evaluation framework for llms Year: (2025)
Ref_id:b46 Title: Estimating means of bounded random variables by betting Year: (2024)
Ref_id:b47 Title: Universal log-optimality for general classes of e-processes and sequential hypothesis tests Year: (2025)
Ref_id:b48 Title: Active, anytime-valid risk controlling prediction sets Year: (2024)
Ref_id:b49 Title: Active multiple testing with proxy p-values and e-values Year: (2025)
Ref_id:b50 Title: Qwen3 technical report Year: (2025)
Ref_id:b51 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b52 Title: Large language models are human-level prompt engineers Year: (2022)
Ref_id:b53 Title: Active statistical inference Year: (2024)
