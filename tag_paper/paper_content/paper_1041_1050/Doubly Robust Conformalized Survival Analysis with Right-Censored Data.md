Title: Doubly Robust Conformalized Survival Analysis with Right-Censored Data
Abstract: We present a conformal inference method for constructing lower prediction bounds for survival times from right-censored data, extending recent approaches designed for more restrictive type-I censoring scenarios. The proposed method imputes unobserved censoring times using a machine learning model, and then analyzes the imputed data using a survival model calibrated via weighted conformal inference. This approach is theoretically supported by an asymptotic double robustness property. Empirical studies on simulated and real data demonstrate that our method leads to relatively informative predictive inferences and is especially robust in challenging settings where the survival model may be inaccurate.

Section: Introduction

this section cite: []

Section: Background and Motivation
Survival analysis focuses on time-to-event data, with applications in many fields including clinical trials, engineering, and marketing. For example, in a clinical trial, researchers may aim to predict how long a cancer patient is likely to survive based on individual characteristics and treatments received. Two central goals are modeling the probability that an event will not occur before a given time and predicting the actual event time. These tasks are complicated by the fact that the data are censored-the exact event time may be unknown due to study limitations or participant withdrawal.
While traditional methods, such as the Kaplan-Meier estimator and parametric models like the Cox proportional hazards model (Cox, 1972), are valued for their interpretability, they struggle in high-dimensional settings or when their assumptions are violated. As a result, machine learning (ML) approaches are gaining popularity (Ishwaran et al., 2008;Katzman et al., 2018), despite the difficulty of obtaining uncertainty estimates and statistical guarantees.
A promising approach to providing rigorous statistical inferences for ML models in survival analysis was recently introduced by Candès et al. (2023) and refined by Gui et al. (2024). Their conformal inference (Vovk et al., 2005;Lei & Wasserman, 2014) framework can use any survival model to compute a lower prediction bound (LPB) for an individual's survival time, supported by rigorous statistical guarantees.
LPBs indicate the time beyond which a patient is expected to survive with at least (1 -α) probability, for some fixed level α ∈ (0, 1). They can be useful in many applications, including for priorizing treatments under resource constraints. When the data are limited or the model overfits, LPBs tend to be conservatively low, reflecting greater uncertainty. By quantifying uncertainty on an individual basis, LPBs are potentially able to distinguish between patients with confidently high survival expectations and those with greater uncertainty. This can lead to actionable insights for high-stakes applications, mitigating the risks associated with over-reliance on potentially inaccurate ML predictions.
this section cite: ['b7', 'b13', 'b15', 'b4', 'b11', 'b25', 'b16']

Section: Main Challenges and Contributions
A limitation of the methods proposed by Candès et al. (2023) and Gui et al. (2024) is their focus on type-I censoring, a scenario not representative of many practical cases. In type-I censoring, all censoring times need to be observed, including for individuals who experienced an event. Observations are represented as ( T , C), where T = min(T, C), with T as the event time and C as the censoring time. For example, in a clinical trial with a fixed end date, C is the time from enrollment to the trial's end, and patients are either censored (T > C) or experience the event (T < C).
These methods, however, do not extend to situations where the censoring times are unobserved for individuals who experience the event-a more common practical scenario known as right-censoring. Under right-censoring, we only observe ( T , I[T < C]), where I[T < C] indicates whether the event occurred before censoring. If T < C, the censoring time C is unknown. For example, in a survival study, T is the time until death, and C is the time until withdrawal or the study's end. If a patient dies (T < C), we observe T = T but not C. Conversely, for censored individuals (T > C), we observe T = C but not T . This incomplete information complicates conformal inference, requiring a novel approach.
We address this challenge by introducing a method for constructing informative LPBs from right-censored data, applicable to any survival model. This extends the approaches of Candès et al. (2023) and Gui et al. (2024) using a twostep process. First, we fit a censoring model to estimate the conditional probability of censoring and use it to impute unobserved censoring times, transforming the right-censored dataset into a semi-synthetic type-I censored dataset. Second, we fit a survival model to estimate survival probabilities and construct LPBs using the imputed dataset.
This method works well in practice and is doubly robust (Bang & Robins, 2005) in theory, ensuring asymptotically valid LPBs if either the censoring model or the survival model is consistently estimated, even if the other is not.
this section cite: ['b4', 'b11', 'b4', 'b11', 'b0']

Section: Related Work
This is not first work extending conformal inference to rightcensored data. Qi et al. (2024) tackled this challenge by using the Kaplan-Meier estimate to impute the latent survival times, operating under the assumption that-despite its lack of covariate adjustment-it can approximate the conditional survival function reasonably well; see Appendix A1.3. However, this assumption is not always easy to justify. Further, unlike ours, their approach is not doubly robust as it relies entirely on having a good approximation of the conditional survival distribution-the very quantity we are trying to infer. As we shall see, our method performs similarly to that of Qi et al. (2024) in easier settings where the survival model already yields approximately valid LPBs, and is able to offer more robust coverage in harder scenarios.
This work contributes to a growing literature on conformal inference beyond exchangeability (Barber et al., 2023), particularly for handling incomplete data. Other works focused on unobserved counterfactuals (Lei & Candès, 2021), missing covariates (Zaffran et al., 2023), weak supervision (Cauchois et al., 2024), and label noise (Feldman et al., 2023;Sesia et al., 2024;Clarkson et al., 2024).
Borrowing from Candès et al. (2023) and Gui et al. (2024), we use weighted conformal inference techniques (Tibshirani et al., 2019) to address the challenge that, under rightcensoring, the missing data may not be missing at random.
this section cite: ['b19', 'b19', 'b1', 'b17', 'b26', 'b5', 'b10', 'b21', 'b6', 'b4', 'b11', 'b23']

Section: Methods

this section cite: []

Section: Problem Setup and Assumptions
We consider a sample of n individuals, indexed by [n] := {1, . . . , n}, drawn i.i.d. from some unknown population.
For each i ∈ [n], let X i ∈ X ⊆ R p represent a vector of p features, T i > 0 the survival time, and C i > 0 the censoring time. Define the event indicator E i = I(T i < C i ) ∈ {0, 1}. The observed time for each individual is Ti = min(T i , C i ). Candès et al. (2023) and Gui et al. (2024) study a type-I censoring scenario, where the available data are D t1-c := {(X i , Ti , C i )} n i=1 , which includes both C i and Ti for all n individuals, along with their features X i . In that setting, they construct an LPB for the survival time T n+1 of a new random individual, with features X n+1 , from the same population. Their methods are reviewed in Appendix A1.
In the right-censoring scenario considered in this paper, the data are D r-c := {(X i , Ti , E i )} n i=1 . These data are generally less informative than those found in a type-I censoring setting, because they only include the true censoring times for individuals who did not experience an event. This makes existing methods not directly applicable.
Given a right-censored data set D r-c , our goal is to predict the survival time T n+1 of a new test individual with covariates X n+1 , sampled from the same distribution as the previous data points. Ideally, we would like to construct an LPB for T n+1 , denoted by L(X n+1 ; D r-c ), that provides (1 -α) marginal coverage at the desired level α ∈ (0, 1):
P T n+1 ≥ L(X n+1 ; D r-c ) ≥ 1 -α.(1)
However, since exact finite-sample guarantees for survival analysis are generally unachievable without strong assumptions, we instead aim to construct LPBs that are approximately valid in finite samples. These LPBs are designed to satisfy a relaxed, asymptotic version of (1), which can be understood as a form of double robustness: they are asymptotically valid as long as at least one of two key population quantities is estimated consistently, even if the other is not.
We will formally define this double robustness property in Section 3, but for now, we focus on presenting our method, starting with its main underlying assumption.
Assumption 2.1. The observed data set D r-c := {(X i , Ti , E i )} n i=1 is generated by applying right-censoring to a latent data set {(X i , T i , C i )} n+1  i=1 , consisting of covariates, survival times, and censoring times for n + 1 individuals, sampled i.i.d. from some joint distribution P X,T,C = P X • P T |X P C|X , where T ⊥ ⊥ C | X. The features and true survival time of the test data point, {X n+1 , T n+1 }, are sampled independently from P X • P T |X .
This setup can be summarized as follows:
(X i , T i , C i ) i.i.d. ∼ P X • P T |X • P C|X , ∀i ∈ [n], E i = I(T i < C i ), Ti = min(T i , C i ), (X n+1 , T n+1 ) ind. ∼ P X • P T |X .(2)
Above, P X , P T |X , and P C|X are arbitrary and unknown.
The assumption T ⊥ ⊥ C | X, known as "conditional independent censoring", states that T and C are independent given X. This standard assumption in survival analysis is also made by Candès et al. (2023) and Gui et al. (2024).
In addition to the calibration dataset D r-c , our method assumes access to an independent training dataset D r-c train , consisting of analogous observations (X, T , E) from a separate group of individuals. These individuals are typically expected to come from the same population, though this is not strictly required for the theoretical results in this paper.
Following a split-conformal inference approach, we use D r-c train to train censoring and survival models, and D r-c to transform their outputs into survival LPBs.
this section cite: ['b4', 'b11', 'b4', 'b11']

Section: DR-COSARC

this section cite: []

Section: METHOD OVERVIEW
We name our method DR-COSARC, which stands for Doubly Robust COnformalized Survival Analysis under Right Censoring. This method uses two ML models: a survival model Msurv approximating P T |X and a censoring model
Mcens approximating P C|X . As detailed below, Msurv guides the construction of a candidate LPB for T n+1 as a function of X n+1 . This candidate LPB can be adjusted either lower or higher via a scalar tuning parameter, calibrated using D r-c to (approximately) achieve the desired coverage level 1 -α. The censoring model Mcens plays two key roles. First, it is used to impute the latent censoring times, simulating a type-I censoring scenario. Second, it helps account for covariate shift in the calibration data, similar to Candès et al. (2023) and Gui et al. (2024).
this section cite: ['b4', 'b11']

Section: TRAINING SURVIVAL AND CENSORING MODELS
The models Msurv and Mcens can be trained using any survival analysis technique, like the standard Cox proportional hazards model, or more sophisticated ML approaches, including random survival forests (Ishwaran et al., 2008).
Both models can be trained on the same right-censored training set D r-c train . For Msurv , the event indicator is defined as usual, with a value of 1 indicating that T < C. For
Mcens , the same techniques can be applied after flipping the event indicator: a value of 1 now indicates that the event did not occur before the censoring time (C < T ).
this section cite: ['b13']

Section: IMPUTING THE MISSING CENSORING TIMES
To simplify the notation, and without much loss of generality, we assume the conditional distribution of C given X = x has a continuous density, denoted by f C|X (c | x), with respect to the Lebesgue measure, for any x ∈ X . Then, we let F
this section cite: []

Section: For example, if Mcens is a Cox proportional hazards model, it estimates the hazard function h
C (c | x) = fC|X (x | x)/[1 -FC|X (c | x)]
using a simple formula, from which fC|X can be derived. Alternatively, if Mcens is a random survival forest, standard implementations provide a nonparametric estimate of the conditional survival function 1 -FC|X (c | x). This estimate can be smoothly interpolated and differentiated to obtain fC|X . For further details on computing fC|X using standard survival analysis models, see the software repository accompanying this paper.
Next, leveraging our estimate fC|X of f C|X , we will transform the right-censored dataset D r-c into a synthetic dataset Dt1-c , designed to mimic the (unobserved) dataset D t1-c that would have been collected under a type-I censoring scenario.
For any i ∈ [n], consider a right-censored random sample (X i , Ti , E i ) from ( 2). Since C i is latent, we replace it with a synthetic "imputed" time, Ĉi , computed as follows. If E i = 0, we know that C i < T i . In this case, the true value of C i is observed and equal to Ti , so we can directly set Ĉi = Ti . Otherwise, if E i = 1, we know that C i > T i = Ti , but the true value of C i remains unknown. Fortunately, however, we have two key pieces of information that allow us to obtain a sensible "guess" Ĉi of C i : the property T ⊥ ⊥ C | X in Assumption 2.1 and the estimate fC|X of f C|X .
Concretely, if E i = 1, we sample Ĉi from the distribution of C | X = X i , Ti , C > Ti , independent of everything else. Thanks to the assumption that T ⊥ ⊥ C | X, the probability density of Ĉi , as a function of the dummy variable c ∈ R, can be written as:
Ĉi ∼ fC|X (c | X i ) Â(X i , Ti ) I[c > Ti ],(3)
where
Â(X i , Ti ) = ∞ Ti fC|X (c | X i )dc.(4)
This procedure, outlined in Algorithm 1, provably leads to a synthetic sample (X i , Ti , Ĉi ) that shares the same distribution as the ideal sample (X i , Ti , C i ) that would be obtained from (2) under type-I censoring, provided that Assumption 2.1 holds and fC|X is equal to f C|X . Proposition 2.2. Under Assumption 2.1, let P X, T ,C denote the distribution of (X i , Ti , C i ), for any i ∈ Implementing Algorithm 1 requires computing the normalization constant in (3), defined by the one-dimensional integral in (4). This can be quickly evaluated either analytically or numerically, depending on the censoring model Mcens . Then, sampling Ĉi from (3) can be performed numerically using inverse transform sampling, which is also computationally fast. Further implementation details are provided in the software repository accompanying this paper.
After imputing ( Ĉ1 , . . . , Ĉn ) using Algorithm 1, either the method of Candès et al. (2023) or Gui et al. (2024) can be applied by substituting the imputed values Ĉi for the unobserved C i . The specific steps for each approach are detailed in Sections 2.2.4 and 2.2.5, respectively.
We emphasize that, in practice, Algorithm 1 must be applied using an estimate fC|X of f C|X . Nonetheless, as long as fC|X is reasonably accurate, our two-step method is anticipated to yield approximately valid inferences. This parallels the expected behavior of the approaches proposed by Candès et al. (2023) and Gui et al. (2024), which achieve approximately valid survival LPBs using conformal weights derived from an estimated censoring model. We will formalize this intuition later in Section 3 by establishing double robustness results for our method, which are qualitatively analogous to those derived by Candès et al. (2023) and Gui et al. (2024) under the simpler type-I censoring scenario.
this section cite: ['b4', 'b11', 'b4', 'b11', 'b4', 'b11']

Section: DR-COSARC WITH FIXED CUTOFFS
We now describe how to implement our method by integrating Algorithm 1 with the approach of Candès et al. (2023), originally designed for data with type-I censoring, which we apply with C i replaced by Ĉi for all i ∈ [n].
The approach of Candès et al. (2023), outlined by Algorithm A1 in Appendix A1.1, entails three main steps. First, the focus is shifted from constructing an LPB for T n+1 to constructing an LPB for (T n+1 ∧ c 0 ) := min{T n+1 , c 0 }, where c 0 > 0 is a pre-defined cutoff constant. Second, D t1-c is filtered to include only samples where C i ≥ c 0 . The filtered dataset is denoted by I cal = {i ∈ [n] : C i ≥ c 0 }. Third, a standard conformal prediction method for noncensored data is applied to calibrate an LPB for (T n+1 ∧ c 0 ).
In the third step above, the output LPB is obtained by calibrating a candidate bound denoted as fa (X n+1 ; Msurv ), which depends on the survival model Msurv as well as on a tunable parameter a. For example, leveraging conformalized quantile regression (CQR) (Romano et al., 2019), one can use fa (x; Msurv ) = qα (x; Msurv )-a, for a ∈ R, where qα (x; Msurv ) is an estimated α-quantile of the conditional distribution of T | X, given by the model Msurv .
The main challenge is that the data (X i , Ti ∧ c 0 ) for i ∈ I cal are not exchangeable with the test point (X n+1 , T n+1 ∧ c 0 ) due to the condition C > c 0 , which shifts their distribution. However, Candès et al. (2023) noted that this difference is a covariate shift, enabling the use of weighted conformal inference (Tibshirani et al., 2019). This adjusts for the distribution shift by re-weighting the samples based on an estimate ĉ(x) of the conditional censoring probabilities c(x) := P [C > c 0 | X = x], obtained from Mcens .
At first sight, it would seem that the method of Gui et al. (2024) can be directly applied to the imputed dataset D imputed output by Algorithm 1. Achieving double robustness, however, requires an additional step. Let L′ (X n+1 ) denote the survival LPB computed by applying the method of Gui et al. (2024) to the imputed dataset D imputed . Instead of directly outputting L′ (X n+1 ), our method takes the minimum of L′ (X n+1 ) and an uncalibrated estimate qα (X n+1 ) of the α-quantile of T | X = X n+1 , provided by the survival model Msurv . While this adjustment is important in theory to ensure the double robustness of our method, it often has a small impact in practice, as we will show empirically, because it is often true that L′ (X n+1 ) ≤ qα (X n+1 ).
Algorithm 2 summarizes the main ideas of this implementation of our method. See Appendix A2.1 and Algorithm A4 therein for further implementation details.
this section cite: ['b4', 'b4', 'b20', 'b4', 'b23', 'b11', 'b11']

Section: DR-COSARC WITH ADAPTIVE CUTOFFS
A limitation of the method of Candès et al. (2023), inherited by Algorithm 2, is its sensitivity to the choice of c 0 . If c 0 is too small or too large, the resulting LPBs tend to be too low to be informative, with the optimal choice often depending on the data in a complex manner. While Candès et al. (2023) provide a heuristic for tuning c 0 , it is not always guaranteed that a suitable value of c 0 even exists, as noted by Gui et al. (2024) and confirmed by our numerical experiments.
To address this, Gui et al. (2024) extended the approach of Candès et al. (2023) by allowing c 0 to vary across individuals based on their features X. Their approach can also use quantile regression (Romano et al., 2019) to compute can-Algorithm 2 DR-COSARC with Fixed Cutoffs input Pre-trained censoring model Mcens , pre-trained survival model Msurv , right-censored data D r-c = {(X i , Ti , E i )} n i=1 , significance level α ∈ (0, 1), test covariates X n+1 , fixed threshold c 0 > 0. 1: Impute ( Ĉ1 , . . . , Ĉn ) using Mcens (Algorithm 1). 2: Assemble D imputed := {(X i , Ti , Ĉi )} n i=1 . 3: Compute an estimate ĉ(x) of c(x), using Mcens . 4: Apply Algorithm A1 (Appendix A1.1) using D imputed , obtaining a preliminary LPB L′ (X n+1 ). 5: Using Msurv , compute an estimate qα (x) of the αquantile of the distribution of
T | X = x. 6: Compute L(X n+1 ) = min{ L′ (X n+1 ), qα (X n+1 )}. output A 1 -α survival LPB L(X n+1 ).
didate LPBs in the form fa (x; Msurv ) = qα (x; Msurv ) -a, for a ∈ R, where qα (x; Msurv ) represents an estimate of the true conditional α-quantile of the distribution of T | X, although this is not the only choice. Their method, described in Algorithm A2 in Appendix A1.2, often leads to much more informative LPBs.
Integrating Algorithm 1 with the approach of Gui et al. (2024) leads to the implementation of our method described by Algorithm 3, which often produces more informative LPBs compared to Algorithm 2. See Appendix A2.2 and Algorithm A5 therein for further implementation details.
Algorithm 3 DR-COSARC with Adaptive Cutoffs input Pre-trained censoring model Mcens , pre-trained survival model Msurv , right-censored data D r-c = {(X i , Ti , E i )} n i=1 , significance level α ∈ (0, 1), test covariates X n+1 . 1: Impute ( Ĉ1 , . . . , Ĉn ) using Algorithm 1. 2: Assemble D imputed := {(X i , Ti , Ĉi )} n i=1 . 3: Apply Algorithm A2 (Appendix A1.2) using D imputed , obtaining a preliminary LPB L′ (X n+1 ). 4: Using Msurv , compute an estimate qα (x) of the αquantile of the distribution of
T | X = x. 5: Compute L(X n+1 ) = min{ L′ (X n+1 ), qα (X n+1 )}. output A 1 -α survival LPB L(X n+1 ).
this section cite: ['b4', 'b4', 'b11', 'b11', 'b4', 'b20', 'b11']

Section: Double Robustness
Although it operates on right-censored data, DR-COSARC is asymptotically doubly robust as the training and calibration samples grow, akin to the methods of Candès et al. (2023) and Gui et al. (2024) under type-I censoring.
While we focus on the asymptotic regime here, Appendix A5 also provides finite-sample coverage bounds for both the fixed-and adaptive-cutoff implementations of our method. Those bounds are valuable in theory to prove the asymptotic double robustness, but they are too loose to be practically useful on their own. Nonetheless, despite the difficulty of finite-sample analyses for this problem, DR-COSARC performs quite well empirically, especially in its adaptive-cutoff implementation, as shown in Section 4.
this section cite: ['b4', 'b11']

Section: Double Robustness with Fixed Cutoffs
We begin by studying Algorithm 2. For concreteness, we focus on the implementation detailed by Algorithm A4, which uses quantile regression (Romano et al., 2019) to compute candidate LPBs in the form fa (x; Msurv ) = qα (x; Msurv )a, for a ∈ R, where qα (x; Msurv ) represents an estimate of q α (x), the true conditional α-quantile of the distribution of T | X, provided by the pre-trained survival model Msurv .
Recall that D train denotes the independent training dataset, with size N = |D train |, used to train Msurv and Mcens . To establish double robustness, we assume that at least one of these models consistently estimates the relevant quantities. Assumption 3.1. The following two limits hold:
lim N →∞ E 1 ĉ(X) - 1 c(X) = 0, lim N →∞ n→∞ n • E ∞ T f C|X (c | X) A(X, T ) - fC|X (c | X) Â(X, T ) dc = 0.
Assumption 3.1 requires that, with increasing training data, the censoring model Mcens consistently estimates the two closely related relevant quantities: the conditional censoring probabilities c(x) used in the approach of Candès et al. (2023), and the conditional censoring density f C|X (c | x) used in Algorithm 1. Importantly, fC|X (c | x) must converge to f C|X (c | x) at a rate faster than the growth of the calibration sample size n, suggesting that a larger portion of the available data should be allocated for training. Assumption 3.2. The following two conditions hold:
(i) There exists a constant b > 0 such that, for any ϵ > 0,
P [T ≥ q α (x) + ϵ | X = x] ≥ 1-α-bϵ almost surely with respect to P X . (ii) lim N →∞ E [|q α (X) -q α (X)|] = 0.
Assumption 3.2 requires that the survival model Msurv consistently estimates q α (x), the conditional α-quantile of T | X, while also assuming that the true distribution of T | X satisfies a relatively mild smoothness condition.
Theorem 3.3. Under Assumption 2.1, if either Assumption 3.1 or Assumption 3.2 holds, the survival LPB L(X n+1 ) produced by Algorithm 2, implemented as detailed Appendix A2.1, has asymptotically valid marginal coverage: lim N →∞,n→∞
P T n+1 ≥ L(X n+1 ) ≥ 1 -α.
Further, under Assumption 3.2, it also has approximate conditional coverage, in the sense that, for any ϵ > 0, lim
N →∞ P P T n+1 ≥ L(X n+1 ) | X n+1 > 1 -α -ϵ = 1.
this section cite: ['b20']

Section: Double Robustness with Adaptive Cutoffs
We study Algorithm 3, focusing for concreteness on the specific implementation detailed in Appendix A2.2. Now, Assumption 3.1 is replaced by Assumption 3.4, which is similar. Its first limit says that the function ĉa should be an accurate estimate (up to a scaling constant) of c a , since in that case (c
a (x)/ĉ a (x))/E [c a (X)/ĉ a (X)] ≈ 1 for all x.
Assumption 3.4. The following two limits hold:
lim N →∞ sup a∈[0,1] E c a (X n+1 )/ĉ a (X n+1 ) E [c a (X)/ĉ a (X)] -1 = 0, lim N →∞ n→∞ n • E ∞ T f C|X (c | X) A(X, T ) - fC|X (c | X) Â(X, T ) dc = 0.
Further, some mild technical conditions are needed.
Assumption 3.5. The function fa (x) used to compute candidate bounds by the approach of Gui et al. (2024) in Algorithm 3 (see Algorithm A5 for details), is continuous in a for P X -almost all x. Further, for any a, there exists a constant γa > 0 such that 1/ĉ a (x) ≤ γa for P X -almost all x.
Then, we can prove Algorithm 3 is also doubly robust.
Theorem 3.6. Under Assumptions 2.1 and 3.5, if either Assumption 3.4 or Assumption 3.2 holds, the LPB L(X n+1 ) produced by Algorithm 3, implemented as detailed in Appendix A2.2, has asymptotically valid marginal coverage:
lim N →∞,n→∞ P T n+1 ≥ L(X n+1 ) ≥ 1 -α.
Further, under Assumption 3.2, it also has approximate conditional coverage, in the sense that, for any ϵ > 0, lim
N →∞ P P T n+1 ≥ L(X n+1 ) | X n+1 > 1 -α -ϵ = 1.
Next, we will verify that the empirical behavior of our method mirrors this appealing theoretical property.
this section cite: ['b11']

Section: Numerical Experiments

this section cite: []

Section: Setup
Synthetic data. We consider three data-generating distributions, summarized in Table A1 (Appendix A3.1), which span a range of interesting settings, partly inspired by related previous works. In each setting, p = 100 covariates X = (X 1 , . . . , X p ) are generated independently, while T and C are sampled independently conditional on X, from a log-
normal distribution-log T | X ∼ N (µ(X), σ(X))-or an exponential distribution-C | X ∼ Exp(λ(X)).
The three settings are ordered by decreasing difficulty. The first two simulate challenging scenarios where accurate survival modeling is difficult, emphasizing the importance of conformal inference. In contrast, the third setting facilitates easier survival model fitting, where raw LPBs from the model already provide approximately valid coverage.
Design and Performance Metrics. We generate independent training, calibration, and test datasets, each with 1000 samples. Right-censoring is simulated by replacing the true T and C with T = min(T, C) and E = I(T ≤ C). The censored data are used to fit survival and censoring models, as specified below. Using these models and the calibration data, we compute 90% survival LPBs for the test set. Performance is evaluated by the average proportion of test points where the true survival time exceeds the LPB (targeting 90%) and the average LPB value, with larger values indicating more informative LPBs. To standardize comparisons across distributions, all LPBs are normalized by dividing by the average oracle lower bound in each setting. All experiments are repeated 100 times, and results are averaged.
Models. We consider four model families for Mcens and Msurv , ensuring consistent comparisons across different calibration methods. The models are as follows: (1) grf, a generalized random forest (R package grf); (2) survreg, an accelerated failure time model (AFT) with a lognormal distribution (R package survival); (3) rf, a generalized random forest (R package randomForestSRC); (4) cox, the Cox proportional hazards model (R package survival).
this section cite: []

Section: Calibration Methods.
We compare six methods. Oracle is an idealized "method" that knows P T |X and directly returns the lower 10% quantile, without using the data. Uncalibrated outputs the raw 90% LPB produced by Msurv without any calibration. Naive CQR applies CQR using Tn+1 = T n+1 ∧ C n+1 as the target of inference instead of T n+1 , typically leading to very small LPBs (Candès et al., 2023). KM Decensoring refers to the method of Qi et al. (2024), reviewed in Appendix A1.3. DR-COSARC (fixed) and DR-COSARC (adaptive) are our methods, implemented as detailed in Appendix A2.1 and A2.2, respectively.
To simplify comparisons, all calibration methods are applied using the same survival model. While the Uncalibrated approach could, in principle, benefit from a survival model trained on a larger dataset, doing so would complicate comparisons and does not affect our main conclusions. As shown in Appendix A3, increasing the training sample size does not resolve the reliability issues of the Uncalibrated method in the challenging settings where this approach fails.
this section cite: ['b4', 'b19']

Section: Leveraging Prior Knowledge on P C|X .
To examine the effect of incorporating prior knowledge about the censoring distribution, we fit Mcens using only the first p 1 ≤ p covariates, assuming C is independent of (X p1+1 , . . . , X p ) given (X 1 , . . . , X p1 ). In the data-generating distributions used in these experiments (Table A1), C | X is independent of X 11 , . . . , X 100 in all settings. Therefore, for 10 ≤ p 1 ≤ p = 100, this prior knowledge helps improve the censoring model by excluding irrelevant predictors and mitigating overfitting. We start with p 1 = 10 and later evaluate the impact of larger p 1 , representing weaker prior knowledge. The case p 1 = p corresponds to no prior knowledge, where all covariates are used to fit the censoring model.
this section cite: []

Section: Results
Figure 1 compares the performance of the six methods in settings 1-3, based on the grf models.
In the first setting, the Uncalibrated method leads to undercoverage. KM Decensoring provides no improvement in this case, as the Kaplan-Meier survival curve it uses to impute T | T > C fails to reasonably approximate the true distribution of T | T > C, X. In contrast, DR-COSARC achieves coverage close to the desired 90% level. However, its coverage is still slightly below the target, and the average value of its lower bounds is noticeably lower than that of the oracle, reflecting the high intrinsic difficulty of this setting.
In the second setting, the Uncalibrated method continues to be invalid, as does KM Decensoring. However, DR-COSARC performs well, achieving 90% coverage and providing relatively high (more informative) LPBs, approaching the oracle's performance. This success is due to its ability to model the censoring distribution accurately.
In the third setting, all methods except Naive CQR perform similarly. In this simpler scenario, Msurv is highly accurate, making conformal calibration less necessary.
this section cite: []

Section: Impact of the Censoring Model Quality
Figure 2 presents results from experiments similar to those in Figure 1, using only a subset of the available 1000 training samples to fit the censoring model. The aim is to examine how the quality of the censoring model affects the performance of our method, specifically in the challenging setting 1. The results indicate that when the censoring model is trained using fewer samples-leading to lower-quality imputation-our method fails to provide valid coverage, performing comparably to the approach of Qi et al. (2024). However, as the number of training samples increases and the quality of the censoring model improves, the coverage of our method approaches the desired 90% level, consistent with its double robustness property. When the censoring model is trained with all 1000 available samples, the coverage reaches the target level, and the experimental setup aligns with that of Figure 1. Performance is measured by empirical coverage, aiming for 90% nominal coverage (dashed red line), and the average value of the lower bound (higher is better, provided the coverage is valid). The number of training samples available to fit the survival and censoring models is 1000. The three settings correspond to situations in which fitting an accurate survival model is increasingly easy, with rigorous conformal calibration being most essential in setting 1 and less crucial in setting 3. The results demonstrate the double robustness property of our method, which requires only one of the survival or censoring models to be accurate in order to achieve valid coverage. Other details are as in Figure 1.
this section cite: ['b19']

Section: Additional Experiments

this section cite: []

Section: Application to Real Data
We apply our method to seven publicly available datasets: VALCT, PBC, GBSG, METABRIC, COLON, HEART, and RETINOPATHY. These datasets cover a range of study designs and sizes; Table A3 in Appendix A4 provides details on the number of observations, covariates, and data sources.
We apply standard preprocessing to each dataset to handle outliers, missing values, and ensure compatibility with all learning algorithms. Zero survival times are replaced with half the smallest non-zero time in the dataset, missing values are imputed using the median for numeric variables and the mode for categorical variables, and rare factor levels are merged into an "other" category or removed for binary factors. Features with high pairwise correlations are iteratively filtered, and linearly redundant variables are removed. See Appendix A4 for additional details about preprocessing.
We compare our method against the same three benchmark approaches considered in Section 4: Uncalibrated, Naive CQR, and KM Decensoring. Because the ground truth data distribution is unknown for these data, the Oracle method cannot be included. Additionally, as the experiments in Section 4 demonstrate that the adaptive-cutoff implementation of our method consistently outperforms the fixed-cutoff implementation, we focus solely on the adaptive version here, referring to it simply as DR-COSARC for clarity.
All methods use the same four types of model as in Section 4 to estimate the survival distribution (grf, survreg, rf, and cox), with the censoring distribution always estimated using grf. The datasets are split into 60% for training, 20% for calibration, and 20% for testing, and each experiment is repeated 100 times using independent random splits.
We evaluate the performance of the survival LPBs produced by each method on the test set in terms of estimated average coverage (targeting the nominal 1 -α level) and average LPB value (higher is better). Since the test data are censored, the true survival times for censored individuals are unobserved, making exact coverage evaluation infeasible. Following the approach of Gui et al. (2024), we estimate empirical lower and upper bounds for the average coverage:
βlow = P[ L(X n+1 ) ≤ Tn+1 ] and βupp = 1 -P[ L(X n+1 ) > Tn+1 , T n+1 ≤ C n+1 ]. These bounds satisfy βlow ≤ P[ L(X n+1 ) ≤ T n+1 ] ≤ βupp .
To simplify comparisons, we also report a point estimate of the coverage, defined as the midpoint βmid = ( βlow + βupp )/2. Figure 3 summarizes the distribution of average estimated coverage across the seven datasets and four models, at level α = 0.1, with Tables A4-A7 in Appendix A4 reporting the detailed results obtained in each setting. Figure A18 in Appendix A4 summarizes similar results obtained using different values of α. Overall, the results indicate that Uncalibrated and KM Decensoring tend to achieve slightly lowerthan-expected coverage, while Naive CQR is overly conservative. In contrast, DR-COSARC consistently achieves average coverage closer to the desired level.
These findings align with the results on synthetic data presented in Section 4. The relatively modest undercoverage observed with Uncalibrated and KM Decensoring in Figure 3 reflects the comparatively simpler nature of these datasets, where the survival model seems reasonably well calibrated even without conformal inference, in contrast to the more challenging synthetic scenarios discussed earlier. Of course, in practice one would typically not know whether the fitted survival model is sufficiently accurate, and our conformal inference method is precisely designed to offer an additional layer of protection in those cases.
this section cite: ['b11']

Section: Discussion
This paper introduces a novel conformal inference method for constructing lower prediction bounds (LPBs) for survival times from right-censored data, extending recent methods designed for type-I censoring. The proposed approach is asymptotically doubly robust in theory and demonstrates strong empirical performance, producing LPBs that are both informative and robust compared to alternative methods.
Our numerical experiments revealed two key insights. First, the adaptive implementation of our method, inspired by Gui et al. (2024), significantly outperforms the fixed-cutoff version (Candès et al., 2023), and we recommend its use in practice. Second, real data experiments showed that uncalibrated survival models often produce reasonably well-calibrated raw LPBs, though they may fail in more complex scenarios.
Our method performs relatively well in these challenging cases, where conformal inference is most critical.
A limitation of our method is its focus on lower prediction bounds, similar to Candès et al. (2023) and Gui et al. (2024). However, Holmes & Marandon (2024) very recently proposed a method for constructing also corresponding upper bounds, suggesting opportunities for combining these approaches. Another promising direction for future work is to extend our method to handle possible data errors, such as inaccuracies in observed times or mislabeled events, building on ideas from Sesia et al. (2024).
Finally, future work could explore strategies to reduce the algorithmic variability of our method, which arises from both random data splitting and stochastic imputation of latent censoring times. Potential directions include the use of e-values (Vovk & Wang, 2021;Bashari et al., 2023) or adopting a full-conformal approach (Vovk et al., 2005).
this section cite: ['b11', 'b4', 'b4', 'b11', 'b21', 'b24', 'b2', 'b25']

Section: References
Ref_id:b0 Title: Doubly robust estimation in missing data and causal inference models Year: (2005)
Ref_id:b1 Title: Conformal prediction beyond exchangeability Year: (2023)
Ref_id:b2 Title: Derandomized novelty detection with FDR control via conformal e-values Year: (2023)
Ref_id:b3 Title: The 5-year prognosis for vision in diabetes Year: (1980)
Ref_id:b4 Title: Conformalized survival analysis Year: (2023)
Ref_id:b5 Title: Predictive inference with weak supervision Year: (2024)
Ref_id:b6 Title: Split conformal prediction under data contamination Year: (2024)
Ref_id:b7 Title: Regression models and life-tables Year: (1972)
Ref_id:b8 Title: Covariance analysis of heart transplant survival data Year: (1977)
Ref_id:b9 Title: The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups Year: (2012)
Ref_id:b10 Title: Conformal prediction is robust to dispersive label noise Year: (2023)
Ref_id:b11 Title: Conformalized survival analysis with adaptive cut-offs Year: (2024)
Ref_id:b12 Title: Two-sided conformalized survival analysis Year: (2024)
Ref_id:b13 Title: Random survival forests Year: (2008)
Ref_id:b14 Title: The statistical analysis of failure time data Year: (2002)
Ref_id:b15 Title: Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network Year: (2018)
Ref_id:b16 Title: Distribution-free prediction bands for non-parametric regression Year: (2014)
Ref_id:b17 Title: Conformal inference of counterfactuals and individual treatment effects Year: (2021)
Ref_id:b18 Title: Levamisole and fluorouracil for adjuvant therapy of resected colon carcinoma Year: (1990)
Ref_id:b19 Title: Conformalized survival distributions: A generic post-process to increase calibration Year: (2024)
Ref_id:b20 Title: Conformalized quantile regression. Advances in neural information processing systems Year: (2019)
Ref_id:b21 Title: Adaptive conformal classification with noisy labels Year: (2024)
Ref_id:b22 Title: The Cox model Year: (2000)
Ref_id:b23 Title: Conformal prediction under covariate shift Year: (2019)
Ref_id:b24 Title: E-values: Calibration, combination and applications Year: (2021)
Ref_id:b25 Title: Algorithmic learning in a random world Year: (2005)
Ref_id:b26 Title: Conformal prediction with missing values Year: (2023-07)
