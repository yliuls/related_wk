Title: Conformal Prediction as Bayesian Quadrature
Abstract: As machine learning-based prediction systems are increasingly used in high-stakes situations, it is important to understand how such predictive models will perform upon deployment. Distributionfree uncertainty quantification techniques such as conformal prediction provide guarantees about the loss black-box models will incur even when the details of the models are hidden. However, such methods are based on frequentist probability, which unduly limits their applicability. We revisit the central aspects of conformal prediction from a Bayesian perspective and thereby illuminate the shortcomings of frequentist guarantees. We propose a practical alternative based on Bayesian quadrature that provides interpretable guarantees and offers a richer representation of the likely range of losses to be observed at test time.

Section: Introduction
Machine learning systems based on deep learning are increasingly used in high-stakes settings, such as medical diagnosis or financial applications. These settings impose unique constraints on the performance of these systems: we want them to produce good outcomes in the aggregate, but also do so fairly and with a guarantee of a low probability of harm. However, predictive models based on deep learning can be difficult to interpret, and commercial models increasingly tend to offer little information about the techniques used in training. This creates a new challenge: How can we flexibly and reliably quantify the suitability of a model for deployment without making too many assumptions about how the model was trained or in which settings it will be used?
Recent research on quantifying uncertainty has employed methods based on conformal prediction (Vovk et al., 2005), which aim to provide guarantees for model performance in a distribution-free way. However, these techniques are based on ideas from frequentist statistics, making it difficult to incorporate prior knowledge that might be available about specific models. For example, in a particular setting we might have access to some information about the distribution of the data that is likely to be encountered, and can construct tighter guarantees on the performance of models by making use of this information. Moreover, they focus on controlling the expected loss averaged over many unobserved datasets rather than focusing on the actual set of observations.
In this paper, we show how methods for guaranteeing model performance can be understood and extended by viewing them from a Bayesian perspective. We develop a framework in which we explicitly model uncertainty in the quantile values associated with particular observations, providing a nonparametric tool for characterizing possible distributions where the model might be deployed that is appropriately constrained by observed data. This framework allows us to draw upon methods from the fields of statistical prediction analysis (Aitchison & Dunsmore, 1975) and probabilistic numerics (Cockayne et al., 2019;Hennig et al., 2022) to develop guarantees that are interpretable and make adaptive use of available information.
We show that two popular uncertainty quantification methods, split conformal prediction (Vovk et al., 2005;Papadopoulos et al., 2002) and conformal risk control (Angelopoulos et al., 2024), can both be recovered as special cases of our framework. Our approach gives a more complete characterization of the performance of these approaches, as we are able to determine the full distribution of possible outcomes rather than a single point estimate. Since our approach is grounded in Bayesian probability, we can easily incorporate knowledge relevant to evaluating the performance of these models when it is present, such as monotonicity or distributional assumptions, while defaulting to existing methods when absent. Our results show that Bayesian probability, while it is often discarded due to the apparent need to specify prior distributions, is actually well-suited for distribution-free uncertainty quantification.
this section cite: ['b32', 'b0', 'b5', 'b10', 'b32', 'b21', 'b2']

Section: Background
Conformal prediction methods apply a wrapper on top of black-box predictive models to be able to subject them to statistical analysis. In order to generate meaningful predictions about future performance, it is assumed that we have access to a small calibration dataset that is representative of the deployment conditions. Performance on this dataset then provides the foundation for generating predictions about future performance. We begin by reviewing existing current distribution-free uncertainty quantification techniques and Bayesian quadrature methods.
this section cite: []

Section: Distribution-free Uncertainty Quantification Techniques
Uncertainty quantification techniques provide guarantees on the future performance of a black-box predictive model mapping inputs X to outputs Y based on a calibration set consisting of X 1 , . . . , X n and Y 1 , . . . , Y n . Different approaches do so in different ways. For more information on these techniques, refer to Shafer & Vovk (2008) or Angelopoulos & Bates (2023).
this section cite: ['b24', 'b1']

Section: Split Conformal Prediction
The goal of Split Conformal Prediction (Vovk et al., 2005;Papadopoulos et al., 2002) is to generate a prediction set or interval that contains the ground-truth output with high probability. This is often expressed in terms of the coverage level 1 -α. It relies on a score function s(x, y) which measures the disagreement between a predictor's output and the ground truth.
The conformal guarantee is
Pr (Y n+1 / ∈ C(X n+1 )) ≤ α,(1)
where
C(X n+1 ) = {y : s(X n+1 , y) ≤ q}(2)
and q is the
⌈(n+1)(1-α)⌉ n quantile of s 1 = s(X 1 , Y 1 ), . . . , s n = s(X n , Y n ).
Here, C(X n+1 ) is a prediction set or interval which aims to include the ground-truth output.
this section cite: ['b32', 'b21']

Section: Conformal Risk Control
In Conformal Risk Control (Angelopoulos et al., 2024), the goal is to generalize conformal prediction to more general loss functions that are monotonic functions of a single parameter λ. Conformal Risk Control (CRC) proceeds by viewing the coverage guarantee (1) as the expected value of a 0-1 loss. It is assumed that the maximum possible value of the loss is B and that the problem is "achievable" by design in that there exists some setting λ max that satisfies the conformal guarantee. Additionally, each loss function L i (λ) is assumed to be a monotonic non-increasing function of λ. The guarantee offered by Conformal Risk Control is of the form
E ℓ(C λ(X n+1 ), Y n+1 ) ≤ α,(3)
where
λ = inf λ : n n + 1 Rn (λ) + B n + 1 ≤ α(4)
and Rn (λ) = 1 n n i=1 L i (λ) is the empirical risk.
this section cite: ['b2']

Section: Bayesian Quadrature
Bayesian quadrature (Diaconis, 1988;O'Hagan, 1991) is a general technique for evaluating integrals that allows for uncertainty in the integrand. It estimates the value of an integral b a f (x) dx by the following four steps: (1) place a prior p(f ) on functions, (2) evaluate f at x 1 , x 2 , . . . , x n , (3) compute a posterior given the observed values of f by Bayes' rule, and ( 4
) estimate b a f (x) dx. Suppose that f (x i ) = y i for i = 1, 2, . . . , n. The posterior over f is p(f | x 1:n , y 1:n ) ∝ p(f ) n i=1 δ(y i -f (x i )),(5)
where δ(•) is the Dirac delta function. The posterior mean then provides an estimate for the integral:
b a f (x) dx ≈ b a f n (x) dx, where (6)
f n (t) = E(f (t) | x 1:n , y 1:n ).(7)
It has been demonstrated that many classical quadrature procedures such as the trapezoid rule can be recovered by placing a Gaussian process prior on functions (Karvonen & Särkkä, 2017).
this section cite: ['b6', 'b20', 'b15']

Section: Summary and Prospectus
Bayesian quadrature provides an illustration of how a primarily numerical method can be connected to Bayesian inference, and in doing so potentially admit additional information about the underlying function that can be incorporated via a prior distribution. In next section, we will see how a similar approach can be applied to conformal prediction, identifying a Bayesian framework that reproduces existing distribution-free uncertainty quantification techniques. The challenge in doing so is that we want guarantees of the style obtained from Bayesian models, but we want to make the approach as general as possible in its assumptions about the underlying distribution. We solve this problem via an approach inspired by probabilistic numerics to construct a nonparametric characterization of the underlying distribution based on the calibration set.
this section cite: []

Section: Decision-theoretic Formulation
In this section we show how split conformal prediction and conformal risk control can be formulated as instances of a general decision problem.
Let z = (z 1 , . . . , z n ) be a set of calibration data where each observation z i = (x i , y i ) consists of an input and a ground truth label. Let θ denote the true state of nature that defines a shared density f (z i | θ) for the data. 1 A new test point z new is assumed to have the same distribution. Let λ be a control parameter (e.g. threshold) that must be chosen based on the calibration data. We assume the presence of a loss function L(θ, λ) which quantifies the loss incurred by selecting λ when the true state of nature is θ.
The decision-theoretic goal is to choose a decision rule λ(z) that controls the risk:
R(θ, λ) = L(θ, λ(z))f (z | θ) dz.(8)
It is often desirable to choose λ so that it is robust to any possible state of nature θ. The maximum risk is defined as
R(λ) = sup θ R(θ, λ).(9)
In distribution-free uncertainty quantification applications, it is often trivial to achieve arbitrarily low risk (for example by forming prediction sets covering the entire output space). We thus want to find decision rules whose risk is upper bounded by a constant α:
R(λ) ≤ α,(10)
and use another criterion (such as expected prediction set size) to select among these. We call a rule that satisfies (10) an α-acceptable decision rule.
this section cite: []

Section: Recovering Split Conformal Prediction
We now show how split conformal prediction is a special case of this decision-theoretic problem. Let L scp (θ, λ) be the miscoverage loss:
L scp (θ, λ) = Pr{s(z new ) > λ} (11) = 1 -Pr{s(z new ) ≤ λ} = 1 -1{s(z new ) ≤ λ}f (z new | θ) dz new ,
where s is an arbitrary nonconformity function.
Proposition 3.1. Define s i ≜ s(z i ) for i = 1, . . . , n and let s (1) ≤ s (2) ≤ . . . ≤ s (n) be the corresponding order 1 In the interest of notational convenience, we assume densities and integrals over zi but these may be replaced by probability mass functions and summations as appropriate.
statistics. Let λ scp be the following decision rule:
λ scp = s (⌈(n+1)(1-α)⌉) , if ⌈(n + 1)(1 -α)⌉ ≤ n ∞, otherwise.(12)
Then λ scp is an α-acceptable decision rule for the miscoverage loss L scp defined in (11).
Proof. Proofs for all theoretical results may be found in Appendix B.
Therefore the prediction set can be constructed as in (2):
C scp (x new ) = {y ∈ Y : s(x new , y) ≤ λ scp },(13)
and by Proposition 3.1, C scp satisfies the conformal guarantee from (1).
this section cite: []

Section: Recovering Conformal Risk Control
Conformal risk control generalizes split conformal prediction by considering losses that are monotonic non-increasing functions of a single parameter λ.
L crc (θ, λ) = ℓ(z new , λ)f (z new | θ) dz new ,(14)
where ℓ(z new , λ) is an individual loss function that is monotonically non-increasing in λ.
Proposition 3.2. Let λ crc be the following decision rule:
λ crc = inf λ : 1 n + 1 n i=1 ℓ(z i , λ) + B ≤ α . (15)
Then λ crc is an α-acceptable decision rule for L crc defined in (14). Note in particular that when ℓ(z, λ) can be expressed in the form ℓ(C λ (x n+1 ), y n+1 ), this recovers the conformal risk control guarantee from (3).
this section cite: []

Section: Our Approach
We introduce our approach by reinterpreting split conformal prediction and conformal risk control as special cases of a more general Bayesian procedure. In order to do so, we borrow ideas from both Bayesian quadrature (Diaconis, 1988;O'Hagan, 1991) and distribution-free tolerance regions (Guttman, 1970). Bayesian quadrature (Section 2.2) solves a numerical integration problem by placing a prior on functions and using Bayesian inference to compute a distribution over the value of the integral. Distributionfree tolerance regions provide a distribution over quantile spacings that holds regardless of the original underlying distribution. Putting these ideas together allows us to extend conformal prediction by producing bounds on expected loss tailored to the actual losses observed in the calibration set.
The remainder of this section is structured as follows. In Section 4.1, we discuss the relationship between risk control and Bayes risk. In Section 4.2, we describe a general approach for using Bayesian quadrature to bound the posterior risk. In Section 4.3, we make the quadrature "distributionfree" by removing the dependence on a prior over functions. In Section 4.4 we handle uncertainty in the evaluation locations of the function by applying results that characterize the spacing between consecutive quantiles. In Section 4.5, we show how to use these results to produce an upper bound on the expected loss. Finally, in Section 4.6, we show how previous conformal prediction techniques can be viewed as a special case of our procedure that only considers the expectation of the posterior loss.
this section cite: ['b6', 'b20', 'b9']

Section: Bayes Risk
The risk R(θ, λ) measures the expected loss for one who already knows the true state of nature θ but not the particular data observed. However, in practical applications the situation is reversed: we do know the observed data but there is uncertainty about the state of nature. Therefore, we want a decision rule that protects against high loss for a range of possible θ. This idea is expressed as the integrated risk:
r(π, λ) = R(θ, λ)π(θ) dθ,(16)
where the prior π(θ) ≥ 0 measures the relative importance of the different possible states of nature. It is well-known that the minimizer of the integrated risk is the so-called Bayes decision rule:
λ π ≜ argmin λ r(λ | z),(17)
where r(λ
| z) is the posterior risk r(λ | z) = E(L λ | z) = L(θ, λ(z))π(θ | z) dθ, (18
) and π(θ | z) ∝ π(θ)f (z | θ).
Interestingly, the worst-case integrated risk of a decision rule is identical to its maximum risk ( 9)
r(λ) ≜ sup π r(π, λ) = sup θ R(θ, λ) = R(λ).(19)
We can therefore focus on bounding the worst-case integrated risk r(λ), since this will also bound the maximum risk R(λ).
this section cite: []

Section: Reformulation as Bayesian Quadrature
We now turn our attention to finding λ minimizing the posterior risk (18). Consider risks that can be expressed as the expectation over individual losses:
L(θ, λ) = ℓ(z new , λ)f (z new | θ) dz new .(20)
It is well-known that the expectation of a random variable is equal to the definite integral of its quantile function over its domain (Shorack, 2000, p. 116). Consider the distribution function of individual losses induced by λ for a particular value of θ:
F (ℓ) ≜ Pr{ℓ(z new , λ) ≤ ℓ | θ} (21
)
The corresponding quantile function is:
K(t) ≡ F -1 (t) = inf{ℓ : F (ℓ) ≥ t},(22)
and the expected loss given K is simply 1 0 K(t) dt. Instead of performing posterior inference over θ, we propose to take an approach inspired by Bayesian quadrature that places a corresponding prior over K. Figure 1 shows a schematic overview of Bayesian quadrature in this setting and how our proposed approach differs. The posterior risk given the observed individual losses ℓ i ≜ ℓ(z i , λ) for i = 1, . . . , n becomes:
E(L | ℓ 1:n ) = J[K]p(K | ℓ 1:n ) dK,(23)
where J[K] ≜ 1 0 K(t) dt and we have suppressed the dependence on λ for notational convenience. The posterior over quantile functions can be expressed as:
p(K | ℓ 1:n ) = p(K | t 1:n , ℓ 1:n )p(t 1:n | ℓ 1:n ) dt 1:n (24) p(K | t 1:n , ℓ 1:n ) ∝ π(K) n i=1 δ(ℓ i -K(t i )).(25)
This resembles the Bayesian quadrature problem from Section 2.2, except the evaluation sites t 1 , . . . , t n are unknown. Fortunately, the distribution of t 1 , . . . , t n is independent of the true distribution of the losses, as we shall now show.
this section cite: []

Section: Elimination of the Prior Distribution
In order to address the dependence of the posterior risk on the prior π(K), we derive an upper bound on the posterior expected loss. The bound takes the form of a weighted sum of the observed losses, where the weights are determined by the spacing between consecutive quantiles. Theorem 4.1. Let t (0) = 0, t (n+1) = 1, and ℓ (n+1) = B.
Then
sup π E(L | t 1:n , ℓ 1:n ) ≤ n+1 i=1 u i ℓ (i) ,(26)
where u i = t (i) -t (i-1) . The posterior is formed via Bayes' rule after observing a set of loss values and quantile levels. However, in practice quantile levels are not directly observed. Middle: Our approach combines properties of quantile spacings with a right rectangular integration rule to construct an upper bound on the posterior distribution of the expected loss. Randomly sampled spacings and corresponding quantile functions are shown in blue along with a 95% credible interval for each quantile level in black. Right: The posterior distribution for a random variable L + that upper bounds the expected loss is constructed by integrating over the unknown quantile levels.
Theorem 4.1 is based on the definite integral of the "worstcase" quantile function that is consistent with the observations. This strategy eliminates the need to specify a prior or evaluate an integral over functions K. We now turn our attention to handling the uncertainty over the quantiles t 1:n .
this section cite: []

Section: Random Quantile Spacings
We now appeal to a result about distribution-free tolerance regions that characterizes the distribution of spacings between consecutive ordered quantiles. Knowledge of this distribution will allow us to handle the input noise in the quadrature problem.
Lemma 4.2 (Distribution of Quantile Spacings (Aitchison & Dunsmore, 1975, p. 140)). Suppose that ℓ 1 , . . . , ℓ n are drawn i.i.d. with continuous 2 distribution function F . Let t i = F (ℓ i ) and u i = t (i) -t (i-1) , where by convention t (0) = 0 and t (n+1) = 1. Then (u 1 , u 2 , . . . , u n+1 ) ∼ = Dir(1, . . . , 1).
We are now ready to present our algorithm for bounding the expected loss E(L | ℓ 1:n ).
this section cite: []

Section: Bound on Maximum Posterior Risk
Putting together Lemma 4.2 and Theorem 4.1 allows us to bound the maximum posterior risk.
Theorem 4.3. Define ℓ (i) to be the order statistics of ℓ 1 , . . . , ℓ n for i = 1, . . . , n and ℓ (n+1) ≜ B. Let L + be 2 The correspondence to a Dirichlet distribution holds exactly for continuous distributions. Weighted sums of Dirichlet random variates stochastically dominate weighted sums of discrete quantile spacings, and thus due to space constraints we only consider continuous distributions here.
the random variable defined as follows:
U 1 , . . . , U n+1 ∼ Dir(1, . . . , 1), L + = n+1 i=1 U i ℓ (i) . (27)
Then for any b ∈ (-∞, B],
inf π Pr(L ≤ b | ℓ 1:n ) ≥ Pr(L + ≤ b).(28)
Theorem 4.3 states that L + stochastically dominates the posterior riskfoot_2 , which allows us to directly form upper confidence bounds as follows.
Corollary 4.4. For any desired confidence level β ∈ (0, 1), define
b * β = inf b {b : Pr(L + ≤ b | ℓ 1:n ) ≥ β}.(29)
Then inf π Pr(L ≤ b | ℓ 1:n ) ≥ β for any b ≥ b * β .
The critical value b * β can be calculated by applying techniques for bounding linear combinations of Dirichlet random variables (Ng et al., 2011, p. 63). Alternatively, straightforward Monte Carlo simulation of L + is often sufficient, and is the approach we take in our experiments. An illustration is shown in Figure 2.
this section cite: []

Section: Recovering Conformal Methods
This perspective puts the previous distribution-free uncertainty techniques in a new light. Taking the expected value This can underestimate the true expected loss (shown here: estimated expected loss 0.45 vs. true expected loss 0.50). Right: Our approach makes use of the fact that the quantile spacings are drawn from a Dirichlet distribution. By considering the full distribution over quantiles, we gain a more complete view of the expected loss. Shown here is one sample drawn from this distribution, which estimates the expected loss as 0.58.
of L + , we find
E(L + ) = n+1 i=1 E(U i )ℓ (i) = 1 n + 1 n i=1 ℓ i + B . (30)
The Conformal Risk Control decision rule (15) then is simply the infimum over λ for which E(L + ) ≤ α.
For, split conformal prediction, the individual loss is defined as ℓ i = 1 -1{s i ≤ λ}. Therefore, suppose that λ = s (k) .
The expected value of L + then becomes:
E(L + ) = 1 n + 1 n + 1 - n i=1 1{s i ≤ s (k) } (31) = 1 - k n + 1(32)
Therefore, E(L + ) ≤ α is satisfied whenever k ≥ (n + 1)(1 -α), and in particular by k * = ⌈(n + 1)(1 -α)⌉. This recovers (12) when ⌈(n + 1)(1 -α)⌉ ≤ n.
Putting these results together, we have recovered standard conformal prediction techniques but have the additional flexibility of considering the distribution of L + rather than the expected value alone. Our experiments explore the value of this approach.
this section cite: []

Section: Experiments
The primary goal of our experiments is to demonstrate the utility of producing a posterior distribution over the expected loss. We conduct experiments on both synthetic data and calibration data collected from MS-COCO (Lin et al., 2014).
For each data setting, we randomly generate M = 10,000 data splits. Each method is used to select λ with the goal of controlling the risk such that R(θ, λ) ≤ α for unknown θ. We compare algorithms on the basis of both the relative frequency of incurring risk greater than α and the prediction set size of the chosen λ. The ideal algorithm would select λ such that the relative frequency of exceeding the target risk is at most a target failure rate of 1 -β = 0.05 while minimizing prediction set size.
As demonstrated in Section 4.6, our method recovers conformal risk control by taking the expected value of L + . Therefore, in order to demonstrate the effect of targeting a conditional guarantee (as opposed to a marginal one as in conformal risk control), we use our Bayesian quadraturebased method to compute the decision rule based on the one-sided highest posterior density (HPD) interval:
λ β hpd ≜ inf λ {λ : Pr(L + ≤ α | ℓ 1:n ) ≥ β},(33)
by finding the corresponding critical values b * β according to ( 29) via Monte Carlo simulation of Dirichlet random variates with 1000 samples. We include Risk-controlling Prediction Sets (RCPS) (Bates et al., 2021) with Hoeffding upper confidence bound as an additional baseline. Code for our experiments is publicly available on Github.foot_3
this section cite: ['b18']

Section: Synthetic Binomial Data
We first sample directly from a known loss distribution so that we can directly compute the frequency of excessively large risk. Here the loss distribution is chosen to be a scaled binomial distribution, normalized to have a maximum loss of B = 1 and probability of failure set to 1 -λ. This was
ℓ(z i , λ) = 1 K K k=1 1{V ik > λ},(34)
where V ik ∼ Uniform(0, 1) for i = 1, . . . , n and k = 1, . . . , K. This loss is therefore monotonically nonincreasing in λ and achieves zero loss at λ max = 1. We set n = 10, K = 4, and α = 0.4.
Since the expectation of the loss (34) is 1 -λ, any trial for which λ < 0.6 constitutes a risk exceeding the α threshold.
The relative frequency of trials exceeding this risk threshold are tabulated in Table 5.1. A histogram of the chosen λ for each of the methods across all 10,000 trials is shown in Figure 3. For conformal risk control, the mean risk across all trials was 0.3363 ± 0.0007 and for our approach λ 0.95 hpd the mean risk was 0.1758 ± 0.0006. In order to visualize the distribution of L + , we plot a histogram of L + according to ( 27) estimated with 100,000 Dirichlet samples for three settings of λ ∈ {0.7, 0.8, 0.9}. The results are shown in Figure 4.
this section cite: []

Section: Synthetic Heteroskedastic Data
In this experiment we also use 10,000 random trials. We use n = 200 calibration samples each. To achieve heteroskedasticity, we let X ∼ U [0, 4] and Y | X ∼ N (0, X 2 ). The prediction intervals are then formed as [-λ, λ] where λ is selected by each method. The loss is the miscoverage loss and the target loss is set to α = 0.1 (i.e. 90% coverage). The maximum allowable risk failure rate is set to 5% (i.e. β = 0.95). The results are show in Table 5.2.
this section cite: []

Section: False Negative Rate on MS-COCO
We also compare methods on controlling the false negative rate of multilabel classification on the MS-COCO dataset (Lin et al., 2014). The experimental setup mirrors that used by Angelopoulos & Bates (2023, Section 5.1). Each random split contains 1000 calibration examples and 3952 test examples. The results of this experiment are summarized in Table 5.3.
this section cite: ['b18']

Section: Discussion
Our results in Table 5.1 demonstrate that even though the Conformal Risk Control marginal guarantee holds, a significant number of individual trials (21.20%) may incur risk exceeding the target threshold. In contrast, by using the more conservative HPD criterion, very few of the trials (0.03%) exceeded the target risk. In Table 5.2, both RCPS and our method achieve failure rate below the target of 5% but our method achieves significantly smaller prediction intervals.
These results point to the qualitative difference in a marginal guarantee, which averages over many possible yet unobserved data sets vs. a conditional guarantee which focuses on knowledge about the state of nature conditioned on the calibration data actually observed. Previous work on conditional guarantees (Barber et al., 2021;Gibbs et al., 2024) has focused on input-conditional guarantees, where the guarantee is conditioned on for all in the input domain. Guarantees of this nature have been shown to be generally impossible without stronger distribution assumptions. Our guarantees are perhaps better characterized by the term "data-conditional guarantee", where we condition on the set of observed loss values. Our experiments demonstrate the practical benefits of this by achieving decisions that produce smaller prediction sets and intervals while not violating the constraint on maximum allowable failure rate. Our guarantees, in contrast, do not rely on strong distribution assumptions that would be necessary to produce an input-conditional guarantee.
The results are again confirmed in Table 5.3 on MS-COCO, which show that the marginal guarantees of Conformal Risk Control lead to an even greater percentage of trials exceeding the risk threshold. On the other hand, RCPS is able to control the risk but this comes at the cost of larger prediction sets. Our approach successfully balances these two concerns, producing prediction intervals that are shorter than baselines while not exceeding the maximum acceptable failure rate. It is also clear that the distribution of the expected loss upper bound L + in Figure 4 provides a more complete view of the range of possible losses and its dependence on λ, a perspective that is not offered by previous methods.
Our goal in this work is to show that the Bayesian viewpoint unlocks a richer interpretation compared to previous works, which focus on marginal guarantees that as we have shown in the paper correspond to the posterior mean. In order to draw an explicit correspondence between our work and previous approaches, the dependence on the prior was removed in Section 4.3. The intuition is that that any rational decision maker operating according to the rules of probability, regardless of prior (sufficiently expressive), would agree with the upper-bounding distribution of we derive. Naturally, commitment to a specific choice of prior would   lead to tighter distributions over the posterior risk, and in future work we seek to bridge these fields even further by exploring specific choices of priors over quantile functions.
The limitations of our method lie primarily in the two main assumptions it makes. First, it assumes that the data at deployment time are independent and identically distributed to the calibration data. Second, it assumes an upper bound B on the losses. If either of these assumptions do not hold, then the guarantees produced by our method are no longer valid. Additionally, the bounds produced by our method are conservative in the sense that they hold for any choice of prior for the loss distribution (provided that the prior is consistent with the calibration data). Therefore, if the two aforementioned assumptions do hold, the actual loss values may be significantly less than indicated by our method.
Overall, our approach demonstrates how conformal prediction techniques can be recovered and extended using Bayesian probability, all without having to specify a prior distribution. This Bayesian formulation is highly flexible due to its nonparametric nature, yet is amenable to incorporating specific information about the distribution of losses likely to be encountered. In practical applications, maximizing the risk with respect to all possible priors may be too conservative, and thus future work may explore the effect of specific priors on the risk estimate.
this section cite: ['b3', 'b8']

Section: References
Ref_id:b0 Title: Statistical Prediction Analysis Year: (1975)
Ref_id:b1 Title: Conformal prediction: A gentle introduction Year: (2023)
Ref_id:b2 Title: Conformal risk control Year: (2024)
Ref_id:b3 Title: The limits of distribution-free conditional predictive inference. Information and Inference: A Year: (2021-06)
Ref_id:b4 Title: Distribution-free, risk-controlling prediction sets Year: ()
Ref_id:b5 Title: Bayesian probabilistic numerical methods Year: (2019)
Ref_id:b6 Title: Bayesian numerical analysis Year: (1988)
Ref_id:b7 Title: Quantile learnthen-test: Quantile-based risk control for hyperparameter optimization Year: (2024)
Ref_id:b8 Title: Conformal Prediction With Conditional Guarantees Year: (2024-09)
Ref_id:b9 Title: Statistical Tolerance Regions: Classical and Bayesian. Griffin's Statistical Monographs and Courses Year: (1970)
Ref_id:b10 Title: Probabilistic Numerics: Computation as Machine Learning Year: (2022)
Ref_id:b11 Title: Posterior distribution of percentiles: Bayes' theorem for sampling from a population Year: (1968-06)
Ref_id:b12 Title: de Finetti's theorem, induction, and A n or Bayesian nonparametric predictive inference (with discussion) Year: (1988)
Ref_id:b13 Title: Fast predictive uncertainty for classification with Bayesian deep networks Year: (2022)
Ref_id:b14 Title: Bayes-optimal prediction with frequentist coverage control Year: (2023-05)
Ref_id:b15 Title: Classical quadrature rules via Gaussian processes Year: (2017)
Ref_id:b16 Title: A First Course in the Calculus of Variations Year: (2014)
Ref_id:b17 Title: Distribution-free predictive inference for regression Year: (2018-07)
Ref_id:b18 Title: Common objects in context Year: (2014)
Ref_id:b19 Title:  Year: (2011)
Ref_id:b20 Title: Bayes-Hermite quadrature Year: (1991-11)
Ref_id:b21 Title: Inductive confidence machines for regression Year: (2002)
Ref_id:b22 Title: Calcul des Probabilités. Georges Carré Year: (1896)
Ref_id:b23 Title: Conformal validity guarantees exist for any data distribution (and how to find them) Year: (2024)
Ref_id:b24 Title: A tutorial on conformal prediction Year: (2008)
Ref_id:b25 Title:  Year: (2003)
Ref_id:b26 Title: Probability for Statisticians Year: (2000)
Ref_id:b27 Title: Empirical Processes with Applications to Statistics Year: (2009)
Ref_id:b28 Title: Quantile risk control: A flexible framework for bounding the probability of high-loss predictions Year: (2023)
Ref_id:b29 Title: Bayesian Optimization with Conformal Prediction Sets Year: (2023-04)
Ref_id:b30 Title: Nonparametric estimation II. Statistically equivalent blocks and tolerance regions-the continuous case Year: (1947)
Ref_id:b31 Title: Nonparametric estimation, III. Statistically equivalent blocks and multivariate tolerance regions-the discontinuous case Year: (1948)
Ref_id:b32 Title: Algorithmic Learning in a Random World Year: (2005)
Ref_id:b33 Title: Nonparametric predictive distributions based on conformal prediction Year: (2017-05)
Ref_id:b34 Title: Determination of sample sizes for setting tolerance limits Year: (1941)
