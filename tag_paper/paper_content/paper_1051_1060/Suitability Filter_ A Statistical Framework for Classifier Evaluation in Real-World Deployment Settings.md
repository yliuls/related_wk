Title: Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Deployment Settings
Abstract: Deploying machine learning models in safetycritical domains poses a key challenge: ensuring reliable model performance on downstream user data without access to ground truth labels for direct validation. We propose the suitability filter, a novel framework designed to detect performance deterioration by utilizing suitability signals-model output features that are sensitive to covariate shifts and indicative of potential prediction errors. The suitability filter evaluates whether classifier accuracy on unlabeled user data shows significant degradation compared to the accuracy measured on the labeled test dataset. Specifically, it ensures that this degradation does not exceed a pre-specified margin, which represents the maximum acceptable drop in accuracy. To achieve reliable performance evaluation, we aggregate suitability signals for both test and user data and compare these empirical distributions using statistical hypothesis testing, thus providing insights into decision uncertainty. Our modular method adapts to various models and domains. Empirical evaluations across different classification tasks demonstrate that the suitability filter reliably detects performance deviations due to covariate shift. This enables proactive mitigation of potential failures in high-stakes applications.

Section: Introduction
Machine learning (ML) models often operate in dynamic, uncertain environments. After a model is tested on a holdout set, a satisfactory evaluation result typically leads to production deployment. However, if test and deployment covariate 0.0 0.2 0.4 0.6 0.8 1.0 Per-Sample Prediction Correctness Probability 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 Estimated Probability Density m Suitability Filter Test Data UNSUITABLE User Data SUITABLE User Data SUITABLE SUITABLE within margin UNSUITABLE
Figure 1. A model M is suitable for use on Du if its accuracy does not fall below the accuracy on Dtest by more than a predefined margin m. The suitability filter calculates per-sample prediction correctness probabilities for both test and user datasets and compares the two distributions through statistical non-inferiority testing. The dashed vertical lines represent the mean values of the distributions corresponding to the estimated accuracies.
distributions differ, performance can drop and cause harm. For example, credit risk models trained on limited historical data may fail in new contexts, disproportionately harming underserved communities through unfair denials or higher interest rates (Kozodoi et al., 2022). Ideally, deployed predictions could be compared directly to ground truth for realtime performance monitoring. However, ground truth may be unavailable (e.g., limited expert labeling (Culverhouse et al., 2003)), unobservable (e.g., counterfactual outcomes in healthcare (Tal, 2023)), or only available much later (e.g., recidivism in law enforcement (Travaini et al., 2022)), thereby causing significant monitoring challenges in deployment.
In this work, we tackle the challenge of determining whether classification accuracy on unlabeled user data degrades significantly compared to a labeled holdout dataset-an issue not directly addressed by existing methods. Our approach combines insights from distribution shift detection, unsupervised accuracy estimation, selective prediction, and dataset inference into a novel performance deterioration detector.
Central to our solution is the suitability filter, an auxil-iary function f s : X → {SUITABLE, INCONCLUSIVE}. Given an unlabeled user dataset D u ∼ D target sampled from the target deployment domain, and a labeled test dataset D test ∼ D source sampled from the original training domain, the filter assesses whether classifier accuracy on D u falls below that on D test by more than a predefined margin m. Our work proposes both (i) a framework for the suitability filter; as well as (ii) a well-performing default instantiation of the filter using domain-agnostic suitability signals that are broadly applicable across classifiers, independent of the model architecture or the training algorithm.
To arrive at its decision, the suitability filter relies on suitability signals (model output features such as maximum logit/softmax or predictive entropy). These signals are sensitive to covariate shifts and can indicate potential prediction errors. In particular, we design a per-sample prediction correctness estimator leveraging these suitability signals. This allows us to assess consistency in the model's predictive behavior on both D u and D test by aggregating sample-level suitability signals. As a result, we are able to detect subtle shifts indicative of changes in model performance. As illustrated in Figure 1, we then compare the means of these distributions (i.e., the estimated accuracies) to arrive at a suitability decision. Our decisions rely on statistical testing to assess whether the estimated difference in means is significant, thus offering a measure of predictive uncertainty.
To ensure the reliability of suitability decisions, we study the statistical guarantees for the suitability filter. Specifically, we identify the theoretical conditions that ensure a bounded false positive rate for end-to-end suitability decisions. We also consider the practical scenarios where such condition may not hold and provide a relaxation of this theoretical condition. This allows model providers to ensure reliability of suitability decisions in spite of theoretical limitations.
Building on these theoretical insights, we empirically show that the filter consistently detects performance deviations arising from various covariate shifts, including temporal, geographical, and subpopulation shifts. Specifically, we assess the effectiveness of our approach using real-world datasets from the WILDS benchmark (Koh et al., 2021). These include FMoW-WILDS for land use classification (Christie et al., 2018), CivilComments-WILDS for text toxicity classification (Borkan et al., 2019), and RxRx1-WILDS for genetic perturbation classification (Taylor et al., 2019). Furthermore, we explore how accuracy differences between user and test datasets impact the filter's sensitivity, analyze calibration techniques to control false positives, and conduct ablations on suitability signals, sample sizes, margins, significance levels, and classifier options.
In summary, our key contributions are the following:
1. We introduce suitability filters as a principled way of de-tecting model performance deterioration during deployment. Our filters detect covariate shift via an unlabeled representative dataset provided by the model user.
2. We propose a statistical testing framework to build suitability filters that aggregate various signals and output a suitability decision. Leveraging formal hypothesis testing, our approach enables control of the false positive rate via a user-interpretable significance level.
3. We theoretically analyze the end-to-end false positive rates of our suitability filters and provide sufficient conditions for bounded false positive rates. We then consider a practical relaxation of this condition, and suggest an adjustment to the prediction margin that maintains our end-to-end bounded false error guarantees.
4. We demonstrate the practical applicability of suitability filters across 29k experiments on realistic data shift scenarios from the WILDS benchmark. On FMoW-WILDS, for example, we are able to detect performance deterioration of more than 3% with 100% accuracy as can be seen in Figure 4.
this section cite: ['b45', 'b11', 'b74', 'b76', 'b44', 'b9', 'b4', 'b75']

Section: Related Work
Our work builds on insights from distribution shifts, accuracy estimation, selective prediction, and dataset inference.
this section cite: []

Section: Distribution Shift Detection.
Distribution shift detection methods aim to identify changes between training and deployment data distributions (Quiñonero-Candela et al., 2022), generally requiring access to ground truth labels. Early research emphasizes detecting shifts in highdimensional data using approaches such as statistical testing on model confidence distributions (Rabanser et al., 2019) or leveraging model ensembles (Ovadia et al., 2019;Arpit et al., 2022). Recent efforts increasingly prioritize interpreting shifts (Kulinski & Inouye, 2023;Koh et al., 2021;Gulrajani & Lopez-Paz, 2021) and mitigating their impact on model performance (Cha et al., 2022;Zhou et al., 2021;Wiles et al., 2021;Zhou et al., 2022;Wang et al., 2022). Some works argue that while small shifts are unavoidable, the focus should be on harmful shifts that lead to significant performance degradation (Podkopaev & Ramdas, 2021;Ginsberg et al., 2022). These approaches aim to detect covariate shifts and subsequently assess their impact on performance. To do so, they rely on ground truth labels or model ensembles to evaluate harmfulness. This assumption makes these techniques unsuitable for our setting where we aim to detect performance degradation without label access.
this section cite: ['b66', 'b67', 'b58', 'b0', 'b46', 'b44', 'b31', 'b5', 'b87', 'b84', 'b88', 'b80', 'b65', 'b27']

Section: Unsupervised Accuracy Estimation.
Unsupervised accuracy estimation, also known as AutoEval (Automatic Model Evaluation (Deng & Zheng, 2021)), aims to estimate a model's classification accuracy (a continuous metric) on unseen data without relying on ground truth labels. Early approaches in this field primarily centered on model confidence, calculated as the maximum value of the softmax output applied to the classifier's logits, and related metrics which we demonstrate to be valuable suitability signals (Hendrycks & Gimpel, 2016;Garg et al., 2022;Kivimäki et al., 2024;Bialek et al., 2024;Guillory et al., 2021;Lu et al., 2023;Wang et al., 2023;Hendrycks & Dietterich, 2018;Deng et al., 2023). Our work differs from these approaches in three key ways: we focus on reliably detecting performance deterioration (a binary decision) in relation to a labeled test dataset using statistical testing.
Selective Classification. Selective classification techniques aim to detect and reject inputs a model would likely misclassify, while maintaining high coverage and accepting as many samples as possible (Chow, 1957;El-Yaniv et al., 2010). In contrast to selective classification, we do not reject or accept individual input data samples. Instead, we leverage sample-level signals and aggregate them to provide a statistically grounded suitability decision for the entire dataset. Initial selective classification methods for neural networks base the rejection mechanism on the model prediction confidence (Hendrycks & Gimpel, 2016;Geifman & El-Yaniv, 2017), a signal that we also leverage in our work.
this section cite: ['b36', 'b23', 'b43', 'b3', 'b30', 'b52', 'b81', 'b35', 'b15', 'b8', 'b17', 'b36', 'b24']

Section: Dataset Inference.
Our approach is inspired by dataset inference (Maini et al., 2021), a technique used to determine whether a model was trained on a particular dataset.
Similarly to dataset inference, we compare suitability distributions between two different data samples through statistical hypothesis testing. However, in contrast to dataset inference, we focus only on evaluation and aim to detect possible performance deterioration, essentially reversing the null and alternative hypotheses. Moreover, dataset inference relies on representative data from both sample domainsthe original source and the deployed target domain-to train a confidence regressor. Instead, we assume that label access is only available in data sampled from the source domain.
this section cite: ['b55']

Section: Problem Formulation
Our suitability filter framework distinguishes between the model provider, who trains and tests the classifier on the source distribution, and the model user, who applies the model to (possibly distributionally shifted) target data.
this section cite: []

Section: Model Provider.
Let Y = {1, . . . , k} denote the label space, representing the set of all possible output labels for a classification problem with k classes. We define our predictor as a model M : X → Y mapping inputs from a covariate space X to classification decisions. A model provider trains such a model M on labeled data sampled from a source distribution D source over domain X . Specifically, the provider usually partitions the data into two disjoint subsets: a training dataset D train ∼ D source , which is used to optimize the parameters of M and a test dataset D test ∼ D source , which is reserved to evaluate the performance of M on unseen data.
To ensure an unbiased evaluation of model performance, these two datasets are disjoint, i.e., D train ∩ D test = ∅.
Model User. A model user interested in deploying model M on their data provides an unlabeled, representative data sample D u ∼ D target . In most scenarios of practical interest, D target differs from D source , i.e., D target ̸ = D source . Note that if D u were to be drawn from the same distribution as D test , the model's performance on both datasets would be identical in expectation, eliminating the need for the suitability filter.
The user might be a third party looking to use the model, or the model provider and user could be the same party.
this section cite: []

Section: Suitability Filter.
The suitability filter assesses whether the performance of classifier M on unlabeled user data D u degrades relative to the known performance on the labeled test dataset D test . In our work, we focus on model accuracy as the performance metric. We define suitability as follows:
Definition 3.1 (Suitability). Given a classifier M : X → Y, a test data sample D test ∼ D source , a user data sample D u ∼ D target , and a performance deviation margin m ∈ R, we define M as suitable for use on D u if and only if the estimated accuracy of M on D u deviates at most by m from the accuracy on D test . Formally:
1 |D u | x∈Du I{M (x) = O(x)} ≥ 1 |D test | (x,y)∈Dtest I{M (x) = y} -m.
(1) Here, I{•} is the indicator function and O(x) represents an oracle that provides the true label y for any input x (the ground truth label is unavailable for samples x ∈ D u ).
Definition 3.2 (Suitability Filter). Given a model M , a test dataset D test , a user dataset D u , a performance metric g and a performance margin m as in Definition 3.1, we define a suitability filter to be a function f s : X → {SUITABLE, INCONCLUSIVE} that outputs SUITABLE if and only if M is suitable for use on D u according to Definition 3.1 with high probability and INCONCLUSIVE otherwise.
this section cite: []

Section: Method
The suitability filter is introduced as a statistical hypothesis test designed to assess if the performance of a model on user data D u deviates from its performance on a test dataset D test by more than a specified margin m. By aggregating
D u ∼ D target D test ∼ D source D sf ∼ D source D train ∼ D source Classifier M : X → Y Classifier M : X → Y Prediction correctness probability estimator C : R s → [0, 1] Prediction correctness probability estimator C : R s → [0, 1] Classifier trained on D train : M : X → Y Signal s 2 ∈ R Signal s 1 ∈ R . . . Signal s s ∈ R Logit w ⊤ s(x; M ) + b 0.5 1 p c σ(•)
Per-sample prediction correctness probability estimator trained on D sf : C : R s → [0, 1]. Aggregated into p c over all data points.
p c Density p c Density Statistical Hypothesis Test m p c
Density
H 0 : µ target < µ source -m INCONCLUSIVE if p > α SUITABLE if p ≤ α Classifier M is suitable for deployment on D u .
Acc(M, D u ) ≥ Acc(M, D test ) -m holds with high probability.
w 1 w 2 w N Figure 2
. Schematic overview of the suitability filter. The suitability filter assesses whether model performance on a user sample Du deviates from its performance on the test dataset Dtest. This is achieved by combining different suitability signals {s1, . . . , ss} to estimate per-sample prediction correctness and comparing the distribution of these estimates between the two datasets using a statistical test.
a diverse set of suitability signals predictive of classifier correctness, the test compares predicted accuracy between D u and D test using a non-inferiority test to ensure the mean performance difference does not exceed the performance margin m (Wellek, 2002;Walker & Nowacki, 2011). We present an schematic overview of our approach in Figure 2.
this section cite: ['b83', 'b79']

Section: Suitability Signals
The first step in constructing the suitability filter is to select a set of signals {s 1 , . . . , s S } that are predictive of per-sample classifier prediction correctness. These signals are inherently dependent on the model
M and capture information about its predictions and confidence levels. As discussed in Section 2, a variety of signals have been proposed in the literature on unsupervised accuracy estimation, selective classification, and uncertainty quantification. Such signals include but are not limited to the maximum logit/softmax scores, the energy of the logits, or the predictive entropy. The exact signals used in this work have been selected to ensure the broad applicability of the suitability filter across diverse settings as outlined in more detail in our experiments (Section 5) and in Appendix A.4.2 and A.4.3. We note that any signal that can be computed for an individual sample and is predictive of prediction correctness can be incorporated into our framework, allowing for flexible extension based on the specific task, dataset, or model M .
this section cite: []

Section: Per-Sample Prediction Correctness Estimator
To learn a per-sample prediction correctness estimator, we require the model provider to have a separate, labeled holdout dataset D sf ∼ D source . While ultimately the goal is to assess performance on the unlabeled D u ∼ D target provided by the user, the hold-out dataset D sf serves as a proxy to train the parameters of the prediction correctness estimator. This dataset is essential because it enables the suitability filter to learn the relationship between different signals and classifier prediction correctness. D sf has to be separate from both D train and D test to avoid overfitting to these samples.
For each sample x ∈ D sf , the selected signals {s 1 , . . . , s S }, which are functions of both the sample and the model M , are evaluated, normalized, and aggregated into a single feature vector s(x; M ) = [s 1 (x; M ), s 2 (x; M ), . . . , s S (x; M )] ∈ R S . The suitability filter framework leverages this feature vector s(x; M ) to predict whether the model M correctly classifies the input x. This is achieved by training a prediction correctness classifier C : R S → {0, 1} that estimates the per-sample probability of prediction correctness p c (x) on the hold-out dataset D sf . In particular, we want to minimize the binary cross-entropy loss between the true correctness label c = I{M (x) = y} and p c (x) for each (x, y) ∈ D sf . We instantiate C as a logistic regressorfoot_0 which models the prediction correctness probability p c (x) = σ(w ⊤ s(x; M ) + b), where σ(z) = 1 1+e -z is the sigmoid function.
We can then leverage C to estimate per-sample prediction correctness for user data samples x ∈ D u (since calculating p c (x) does not require ground truth label access) as well as the test data D test . Next, we discuss steps to verify and ensure that C generalizes effectively to D u ∼ D target .
Calibration. To ensure that the mean estimated probability of prediction correctness directly reflects accuracy, p c (x) must be well-calibrated for both samples from D source and D target . However, absent specific assumptions about the differences between D source and D target , achieving this desired calibration is impossible in practice (David et al., 2010).
One reasonable assumption under which such calibration issues can be mitigated is if the potential target distributions consist of subpopulations of the source distribution. In credit scoring, for instance, the target distribution may include subpopulations S, such as minority groups or individuals with limited credit histories, who are underrepresented in the training data. In such scenarios, multicalibration techniques can ensure that C provides accurate predictions for every subpopulation S ∈ C, thereby improving reliability across all possible D target (Hébert-Johnson et al., 2018). Here, C denotes a collection of computationally identifiable subsets of the support of D source and i ∼ S is a sample drawn from D source conditioned on membership in S.
When no assumptions about D source and D target can be made, achieving reliable calibration is challenging. Calibrating C on D sf (e.g., using Platt's method (Platt et al., 1999) or temperature scaling (Guo et al., 2017)) ensures that the classification correctness estimator C provides reliable estimates of the probability that model M correctly classifies samples in D sf ∼ D source . While, in theory, this calibration extends to D test ∼ D source , we generally cannot assume calibration on D u ∼ D target . Our approach to addressing this issue combines ongoing quality assurance checks with appropriate margin adjustments and will be discussed in more detail in Section 4.4.
this section cite: ['b12', 'b34', 'b63', 'b32']

Section: Non-Inferiority Testing
Non-inferiority testing is a statistical method used to assess whether the performance of a new treatment, model, or method is not significantly worse than a reference or control by more than a pre-specified margin m (Wellek, 2002;Walker & Nowacki, 2011). Unlike other statistical tests, which typically test for a difference between distributions, this test aims to confirm that the new method is not inferior by more than a margin of m. Consequently, the null hypothesis is that the method is inferior, in contrast to the usual null hypothesis of no difference.
this section cite: ['b83', 'b79']

Section: Correctness Distributions.
If the per-sample prediction correctness estimator C is well-calibrated, the mean of the estimated prediction correctness probabilities across a dataset approximates the accuracy of the model M on that dataset. Formally, let p c [D test ] and p c [D u ] denote the vectors of estimated prediction correctness probabilities for the test dataset D test and the user dataset D u , respectively:
p c [D test ] := p c (x 1 ), . . . , p c (x |Dtest| ) ∈ [0, 1] |Dtest| (2) p c [D u ] := p c (x 1 ), . . . , p c (x |Du| ) ∈ [0, 1] |Du|(3)
Here, p c (x i ) represents the estimated probability of prediction correctness for each sample x i .
this section cite: []

Section: Hypothesis Setup.
We define the true means of the estimated prediction correctness probabilities for data drawn from the source and target distributions as follows:
µ source := E x∼Dsource [p c (x)] (4) µ target := E x∼Dtarget [p c (x)](5)
The primary goal of the non-inferiority test is to compare the true mean predicted correctness between the two distributions and determine whether µ target is not lower than µ source by more than a pre-specified margin m. This is formally expressed as the following hypothesis testing setup:
H 0 : µ target < µ source -m (6) H 1 : µ target ≥ µ source -m(7)
The null hypothesis H 0 posits that the estimated performance on the user dataset is worse than on the test dataset by more than the margin m. The alternative hypothesis H 1 asserts that the estimated performance on the user dataset is either better than, equivalent to or not worse than that on the test dataset within the allowed margin m. We conduct the statistical non-inferiority test using a one-sided Welch's t-test (see Appendix A.1.1).
this section cite: []

Section: Suitability Decision
Finally, the decision on the suitability of the model for the user dataset is based on the outcome of this non-inferiority test. If the test indicates non-inferiority, we conclude that the model's performance on D u is acceptable and we output SUITABLE. If the test fails to reject the null hypothesis, the model is either unsuitable for the user dataset or the number of samples provided was insufficient to determine suitability and hence we return INCONCLUSIVE. To ensure the reliability of these suitability decisions, we next discuss statistical guarantees and the conditions under which they hold for the end-to-end suitability decision.
this section cite: []

Section: Statistical Guarantees.
To account for miscalibration errors, we define δ-calibration as follows:
Definition 4.1 (δ-Calibration). Let p c (x) denote the estimated probability of prediction correctness for a sample x with predicted label M (x) and true label y. Assuming that p c (x) has a well-defined probability density function f c (ν)
over [0, 1], we say p c (x) is δ-calibrated if P [M (x) = y | p c (x) = ν] = ν + ϵ(ν),(8)
∀ν ∈ [0, 1] with calibration error
1 0 ϵ(ν)f c (ν) dν = δ for 0 ≤ |δ| ≪ 1.
Under the assumption of testing two independent and normally distributed samples, the non-inferiority test ensures a controlled false positive rate (FPR), bounding the probability of incorrectly concluding non-inferiority. Theorem 4.2 (Non-Inferiority Test Guarantee). Let µ source and µ target represent the true mean prediction correctness for the source and target distributions, respectively. Assuming that these samples are independent and normally distributed, a non-inferiority test based on Welch's t-test at significance level α guarantees that the probability of rejecting the null hypothesis H 0 : µ target < µ source -m (i.e., concluding µ target ≥ µ source -m) when H 0 is true is controlled at α:
P(Reject H 0 | H 0 is true) ≤ α, (9
)
where m is the non-inferiority margin (Lehmann et al., 1986;Wellek, 2002).
The following results extend this guarantee to the end-toend suitability filter under δ-calibration for the correctness estimator C with respect to both D source and D target . All expectations and probabilities are over samples (x, y) ∼ X × Y unless specified otherwise. Lemma 4.3 (Expectation of Correctness). Under δcalibration as defined in Definition 4.1, the deviation between the true probability of prediction correctness and the estimation by classifer C is given by:
E[p c (x)] -P[M (x) = y] = δ.(10)
Proof in Appendix A.1.2.
We use Lemma 4.3 to derive the end-to-end guarantee for the suitability filter. Corollary 4.4 (Bounded False Positive Rate for Suitability Filter under δ-Calibration). Given a prediction correctness estimator C that is δ-calibrated on both the source and target distributions with δ source and δ target , respectively, let us define m ′ := m + δ source -δ target and conduct a noninferiority test with H 0 : µ target < µ source -m ′ . The probability of incorrectly rejecting H 0 (i.e., returning SUITABLE) when the model accuracy on D target is lower than on D source by more than a margin m is upper bounded by the significance level α. Proof in Appendix A.1.3.
The following remark details the limits of these guarantees. Remark 4.5 (Impossibility of Bounded False Positive Rate without δ-Calibration). If the calibration deviations δ source and δ target are not provided or are not much smaller than 1, it is not possible to choose m ′ according to Corollary 4.4. Hence, without δ-calibration, no guarantees on the false positive rate of the suitability filter can be provided.
this section cite: ['b48', 'b83']

Section: Practical Considerations.
Under perfect calibration, the calibration errors vanish, i.e., δ source = δ target = 0, eliminating the need for any margin correction. However, achieving Margin adjustment m ′ = m + ∆test -∆u. perfect calibration in practice is rare. In most real-world deployments, accurately determining the calibration errors δ source and especially δ target can be difficult. Consequently, adjusting the margin as proposed in Corollary 4.4 may be challenging. To address this, we draw inspiration from best practices in quality assurance and propose that the model owner periodically collects a small labeled dataset, Du , from a potential user of the system. Given access to the test dataset D test , the model owner can compute both the estimated accuracies (µ u , µ test ) as approximated by C, as well as ground truth accuracies (Acc u , Acc test ). This enables an empirical evaluation of accuracy estimation errors, ∆ u and ∆ test , which correspond to δ target and δ source , respectively:
∆ = 1 N N i=1 p c (x i ) -I{M (x i ) = y i }(11)
Following the margin adjustment in Corollary 4.4, the updated margin is:
m ′ = m + ∆ test -∆ u .(12)
The intuition behind this adjustment is that the decisions output by the suitability filter reflect the expected ground truth suitability decisions even in the presence of prediction errors as can also be seen in Figure 3. Regular recalibration and careful margin tuning ensure that C continues to provide reliable estimates, even in the presence of distribution shifts or evolving deployment conditions.
this section cite: []

Section: Experimental Evaluation
To evaluate the performance of our proposed suitability filter, we conduct a series of experiments with different datasets,
Table 1. Evaluating detection performance of the proposed suitability filter on FMoW-WILDS, RxRx1-WILDS and
CivilComments-WILDS for m = 0 with both ID and OOD user data. We report the area under the curve for ROC and PR (capturing the tradeoffs at various significance thresholds), as well as accuracy and the true false positive rate at α = 0.05. We also report 95% confidence intervals based on 3 models M trained on the same Dtrain with different random seeds.
this section cite: []

Section: DATASET ACC FPR ROC PR
FMOW-WILDS ID 81.8 ± 3.1% 0.027 ± 0.033 0.969 ± 0.023 0.967 ± 0.029 FMOW-WILDS OOD 91.9 ± 2.5% 0.018 ± 0.017 0.965 ± 0.016 0.891 ± 0.035 RXRX1-WILDS ID 100.0 ± 0.0% 0.000 ± 0.000 1.000 ± 0.000 1.000 ± 0.000 RXRX1-WILDS OOD 97.5 ± 7.2% 0.031 ± 0.088 0.997 ± 0.006 0.989 ± 0.024
CI V I LCO M M E N T S-WILDS ID
93.3 ± 5.3% 0.002 ± 0.007 0.997 ± 0.008 0.971 ± 0.067 model architectures, and naturally occurring distribution shift types from the WILDS benchmark (Koh et al., 2021).
this section cite: ['b44']

Section: General Evaluation Setup
We evaluate the suitability filter on FMoW-WILDS (Christie et al., 2018), CivilComments-WILDS (Borkan et al., 2019) and RxRx1-WILDS (Taylor et al., 2019). For each dataset, we follow the recommended training paradigm to train a model M using empirical risk minimization and the pre-defined D train ∼ D source . We then further split the provided in-distribution (ID) and out-of-distribution
(OOD) validation and test splits into folds as detailed in Appendix A.2.2 (16 ID and 30 OOD folds for FMoW-WILDS, 4 ID and 8 OOD folds for RxRx1-WILDS, and 16 ID folds for CivilComments-WILDS). We conduct two types of experiments: first, each ID fold is used as the user dataset (D u ), and the remaining ID data is split into 15 subsets, used as D test and D sf . This yields 16×15×14 experiments for FMoW-WILDS, 4×15×14 for RxRx1-WILDS, and 16×15×14 for CivilComments-WILDS. Second, each OOD fold is used as D u , and the ID data is split into 15 subsets, used for D test and D sf . This yields 30×15×14 experiments for FMoW-WILDS and 8×15×14 for RxRx1-WILDS.
We define the binary suitability ground truth as Acc(M, D u ) ≥ Acc(M, D test ) -m. While statistical guarantees are discussed under margin adjustments in Section 4.4, achieving the necessary calibration error estimates in practice is challenging. In particular, obtaining a reliable approximation for δ target requires access to a small labeled user dataset Du , which may not always be available. Moreover, even if Du is collected, its representativeness of the true deployment distribution D target is uncertain, introducing potential biases in the accuracy estimation error ∆ u .
To account for this in our experiments, we set m ′ = m for the non-inferiority test, effectively using the predefined margin without additional corrections. We discuss this in more detail in Appendix A.4.1. We evaluate suitability decisions by computing the ROC AUC across significance levels, capturing the trade-off between true and false posi-tives. Additionally, we report PR AUC, accuracy, and false positive rate at the common α = 0.05 threshold.
this section cite: ['b9', 'b4', 'b75']

Section: Suitability Signals
We use the following suitability signals in our instantiation of the suitability filter (more details in Appendix A.2.1):
conf max: Maximum confidence from softmax. margin loss: Difference in cross-entropy loss between the predicted class and next best class. energy: Logits energy, computed as the negative log-sum-exponential, measuring model certainty.
this section cite: []

Section: Results
As our work introduces a novel problem setting with no existing baselines for direct comparison, the primary objective of the following is to provide an intuition for the conditions under which our approach works effectively, its limitations, and the factors influencing its performance.
Table 1 summarizes the performance of the proposed suitability filter across three benchmark datasets from the WILDS collection: FMoW-WILDS, RxRx1-WILDS, and -7 % t o -6 % -6 % t o -5 % -5 % t o -4 % -4 % t o -3 % -3 % t o -2 % -2 % t o -1 % -1 % t o 0 % 0 % t o 1 % 1 % t o 2 % 2 % t o 3 % 3 % t o 4 % 4 % t o 5 % 5 % t o 6 % 6 % t o 7 % Acc(M, Du) Acc(M, Dtest)
this section cite: []

Section: CivilComments-WILDS.
Although ID and OOD results cannot be directly compared due to the differing numbers of ground truth positives and negatives, interesting trends still emerge. On FMoW-WILDS for example, we observe higher accuracy and a lower FPR at the 5% significance level for OOD user data, while ROC AUC and PR AUC are higher for ID user data. This discrepancy may stem from class imbalance: across OOD experiments, we have nearly three times as many true negatives as true positives, making it easier to achieve high accuracy despite it generally being harder to maintain discriminative performance in an OOD setting. The latter is also confirmed for RxRx1-WILDS, where we see decreased performance on OOD user data compared to ID user data. Another noteworthy observation is the high overall performance on RxRx1-WILDS. The reason for this is that we observe large differences in model performance on RxRx1-WILDS depending on the fold considered, as can be seen in Table 3 ( Appendix). This variation helps the suitability filter detect performance deterioration more easily, as larger performance differences enhance its ability to identify changes. This sensitivity of suitability decisions to differences in accuracy between the user and test datasets is also illustrated in Figure 4 on FMoW-WILDS for m = 0. The ideal relationship would be a step function, where SUITABLE decisions occur only when user dataset accuracy exceeds test accuracy. However, achieving this requires a perfect estimate of accuracy on D u , which is impossible without ground truth labels. In practice, we observe that the slope of the suitability decision curve is flatter than the ideal step function. There are a few erroneous SUITABLE decisions when the accuracy difference is below 0%, indicating occasional false positives. However, for differences < -3% (indicating a performance deterioration of at least 3%, this is the case for 8.4k experiments out of nearly 29k in total), our proposed suitability filter achieves 100% accuracy. Additionally, some false neg-atives are observed in the range [0%, 3%], reflecting scenarios where the empirical evidence provided by D u and D test is insufficient to reject the inferiority null hypothesis at the chosen significance level α = 0.05. However, for accuracy difference buckets exceeding 3%, the percentage of SUIT-ABLE decisions consistently exceeds 80% and increases to 100% above 6% of accuracy difference, demonstrating the robustness of the approach in scenarios with sufficiently large accuracy differences. Additional experiments, results, and interpretations can be found in Appendix A.4.
this section cite: []

Section: Discussion
Conclusion. We introduce the suitability filter, a novel framework for evaluating whether model performance on unlabeled downstream data in real-world deployment settings deteriorates compared to its performance on test data. We present an instantiation for classification accuracy that leverages statistical hypothesis testing. We provide theoretical guarantees on the false positive rate of suitability decisions and propose a margin adjustment strategy to account for calibration errors. Through extensive experiments on real-world datasets from the WILDS benchmark, we demonstrate the effectiveness of suitability filters across diverse covariate shifts. Our findings highlight the potential of suitability filters as a practical tool for model monitoring, enabling more reliable and interpretable deployment decisions in dynamic environments. Suitability filters provide an effective way to expose model capabilities and limitations and thus enable auditable service level agreements (SLAs).
this section cite: []

Section: References
Ref_id:b0 Title: Ensemble of averages: Improving model selection and boosting performance in domain generalization Year: (2022)
Ref_id:b1 Title: Agreement-on-the-line: Predicting the performance of neural networks under distribution shift Year: (2022)
Ref_id:b2 Title: Controlling the false discovery rate: a practical and powerful approach to multiple testing Year: (1995)
Ref_id:b3 Title: Estimating model performance under covariate shift without labels Year: (2024)
Ref_id:b4 Title: Nuanced metrics for measuring unintended bias with real data for text classification Year: (2019)
Ref_id:b5 Title: Domain generalization by mutual-information regularization with pre-trained models Year: (2022)
Ref_id:b6 Title: Detecting errors and estimating accuracy on unlabeled data with self-training ensembles Year: (2021)
Ref_id:b7 Title: Mandoline: Model evaluation under distribution shift Year: (2021)
Ref_id:b8 Title: An optimum character recognition system using decision functions Year: (1957)
Ref_id:b9 Title: Functional map of the world Year: (2018)
Ref_id:b10 Title: Estimating generalization under distribution shifts via domain-invariant representations Year: (2020)
Ref_id:b11 Title: Do experts make mistakes? a comparison of human and machine indentification of dinoflagellates Year: (2003)
Ref_id:b12 Title: Impossibility theorems for domain adaptation Year: (2010)
Ref_id:b13 Title: Are labels always necessary for classifier accuracy evaluation? Year: (2021)
Ref_id:b14 Title: What does rotation prediction tell us about classifier accuracy under varying testing environments Year: (2021)
Ref_id:b15 Title: Confidence and dispersity speak: Characterizing prediction matrix for unsupervised accuracy estimation Year: (2023)
Ref_id:b16 Title: Unsupervised supervised learning i: Estimating classification and regression errors without labels Year: (2010)
Ref_id:b17 Title: On the foundations of noise-free selective classification Year: (2010)
Ref_id:b18 Title: To annotate or not? predicting performance drop under domain shift Year: (2019)
Ref_id:b19 Title: Reverse testing: an efficient framework to select amongst classifiers under sample selection bias Year: (2006)
Ref_id:b20 Title: Selective prediction-set models with coverage rate guarantees Year: (2023)
Ref_id:b21 Title: Dropout as a bayesian approximation: Representing model uncertainty in deep learning Year: (2016)
Ref_id:b22 Title: Selective classification via one-sided prediction Year: (2021)
Ref_id:b23 Title: Leveraging unlabeled data to predict out-of-distribution performance Year: (2022)
Ref_id:b24 Title: Selective classification for deep neural networks Year: (2017)
Ref_id:b25 Title: Selectivenet: A deep neural network with an integrated reject option Year: (2019)
Ref_id:b26 Title: Bias-reduced uncertainty estimation for deep neural classifiers Year: (2019)
Ref_id:b27 Title: A learning based hypothesis test for harmful covariate shift Year: (2022)
Ref_id:b28 Title: A learning based hypothesis test for harmful covariate shift Year: (2023)
Ref_id:b29 Title: Instance segmentation model evaluation and rapid deployment for autonomous driving using domain differences Year: (2023)
Ref_id:b30 Title: Predicting with confidence on unseen distributions Year: (2021)
Ref_id:b31 Title: In search of lost domain generalization Year: (2021)
Ref_id:b32 Title: On calibration of modern neural networks Year: (2017)
Ref_id:b33 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b34 Title: Multicalibration: Calibration for the (computationallyidentifiable) masses Year: (2018)
Ref_id:b35 Title: Benchmarking neural network robustness to common corruptions and perturbations Year: (2018)
Ref_id:b36 Title: A baseline for detecting misclassified and out-of-distribution examples in neural networks Year: (2016)
Ref_id:b37 Title: Aries: Efficient testing of deep neural networks via labeling-free accuracy estimation Year: (2023)
Ref_id:b38 Title: Densely connected convolutional networks Year: (2017)
Ref_id:b39 Title: Self-adaptive training: beyond empirical risk minimization Year: (2020)
Ref_id:b40 Title: Estimating the accuracies of multiple classifiers without labeled data Year: (2015)
Ref_id:b41 Title: Assessing generalization of sgd via disagreement Year: (2021)
Ref_id:b42 Title: A method for stochastic optimization Year: (2014)
Ref_id:b43 Title: Confidence-based estimators for predictive performance in model monitoring Year: (2024)
Ref_id:b44 Title: Wilds: A benchmark of in-thewild distribution shifts Year: (2021)
Ref_id:b45 Title: Fairness in credit scoring: Assessment, implementation and profit implications Year: (2022)
Ref_id:b46 Title: Towards explaining distribution shifts Year: (2023)
Ref_id:b47 Title: Simple and scalable predictive uncertainty estimation using deep ensembles Year: (2017)
Ref_id:b48 Title: Testing statistical hypotheses Year: (1986)
Ref_id:b49 Title: Learning a data-driven policy network for pretraining automated feature engineering Year: (2023)
Ref_id:b50 Title: Deep gamblers: Learning to abstain with portfolio theory Year: (2019)
Ref_id:b51 Title: Fixing weight decay regularization in adam Year: (2017)
Ref_id:b52 Title: Predicting out-of-distribution error with confidence optimal transport Year: (2023)
Ref_id:b53 Title: Co-validation: Using model disagreement on unlabeled data to validate classification algorithms Year: (2004)
Ref_id:b54 Title: Performance prediction under dataset shift Year: (2022)
Ref_id:b55 Title: Dataset inference: Ownership resolution in machine learning Year: (2021)
Ref_id:b56 Title: K-means clustering based feature consistency alignment for label-free model evaluation Year: (2023)
Ref_id:b57 Title: A multiple testing procedure for clinical trials Year: (1979)
Ref_id:b58 Title: Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift Year: (2019)
Ref_id:b59 Title: Contrastive automated model evaluation Year: (2023)
Ref_id:b60 Title: Energy-based automated model evaluation Year: (2024)
Ref_id:b61 Title: Estimating accuracy from unlabeled data: A probabilistic logic approach Year: (2017)
Ref_id:b62 Title: Estimating accuracy from unlabeled data: A bayesian approach Year: (2016)
Ref_id:b63 Title: Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods Year: (1999)
Ref_id:b64 Title: Clinical trials: a practical approach Year: (2013)
Ref_id:b65 Title: Tracking the risk of a deployed model and detecting harmful distribution shifts Year: (2021)
Ref_id:b66 Title: Dataset shift in machine learning Year: (2022)
Ref_id:b67 Title: Failing loudly: An empirical study of methods for detecting dataset shift Year: (2019)
Ref_id:b68 Title: Selective classification via neural network training dynamics Year: (2022)
Ref_id:b69 Title: Learning to validate the predictions of black box machine learning models on unseen data Year: (2019)
Ref_id:b70 Title: Imagenet large scale visual recognition challenge Year: (2015)
Ref_id:b71 Title: Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter Year: (2019)
Ref_id:b72 Title: Learning to validate the predictions of black box classifiers on unseen data Year: (2020)
Ref_id:b73 Title: Label-free model evaluation with semi-structured dataset representations Year: (2021)
Ref_id:b74 Title: Target specification bias, counterfactual prediction, and algorithmic fairness in healthcare Year: (2023)
Ref_id:b75 Title: Rxrx1: An image set for cellular morphological variation across many experimental batches Year: (2019)
Ref_id:b76 Title: Machine learning and criminal justice: A systematic review of advanced methodology for recidivism risk prediction Year: (2022)
Ref_id:b77 Title: A bag-ofprototypes representation for dataset-level applications Year: (2023)
Ref_id:b78 Title: Predicting neural network accuracy from weights Year: (2020)
Ref_id:b79 Title: Understanding equivalence and noninferiority testing Year: (2011)
Ref_id:b80 Title: Generalizing to unseen domains: A survey on domain generalization Year: (2022)
Ref_id:b81 Title: Toward auto-evaluation with confidence-based category relation-aware regression Year: (2023)
Ref_id:b82 Title: The generalization of 'student's'problem when several different population varlances are involved Year: (1947)
Ref_id:b83 Title: Testing statistical hypotheses of equivalence Year: (2002)
Ref_id:b84 Title: A fine-grained analysis on distribution shift Year: (2021)
Ref_id:b85 Title: On the importance of feature separability in predicting out-ofdistribution error Year: (2024)
Ref_id:b86 Title: Predicting out-of-distribution error with the projection norm Year: (2022)
Ref_id:b87 Title: Domain generalization with mixstyle Year: (2021)
Ref_id:b88 Title: Domain generalization: A survey Year: (2022)
