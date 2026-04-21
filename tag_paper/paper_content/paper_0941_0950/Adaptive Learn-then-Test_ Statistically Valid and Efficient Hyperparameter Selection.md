Title: Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection
Abstract: We introduce adaptive learn-then-test (aLTT), an efficient hyperparameter selection procedure that provides finite-sample statistical guarantees on the population risk of AI models. Unlike the existing learn-then-test (LTT) technique, which relies on conventional p-value-based multiple hypothesis testing (MHT), aLTT implements sequential data-dependent MHT with early termination by leveraging e-processes. As a result, aLTT can reduce the number of testing rounds, making it particularly well-suited for scenarios in which testing is costly or presents safety risks. Apart from maintaining statistical validity, in applications such as online policy selection for offline reinforcement learning and prompt engineering, aLTT is shown to achieve the same performance as LTT while requiring only a fraction of the testing rounds.

Section: Introduction

this section cite: []

Section: Context and Motivation
The safe and reliable deployment of AI applications, or apps for short, hinges on the possibility of certifying their performance (Seshia et al., 2022;Tegmark & Omohundro, 2023). Depending on the problem, this may require controlling the missed detection probability in medical imaging (Lu et al., 2022;Mehrtash et al., 2020), ensuring safety measures for control policies (Lindemann et al., 2023;Zecchin et al., 2024), or verifying the correctness of the answers given by a large language model (LLM) (Quach et al., 2023).
In practice, before deployment, AI apps can be often calibrated by selecting hyperparameters based on data set aside for this purpose or based on rounds of real-world testing.
In the current scaling-centric era of large, Internet-scale, data sets (Xiao, 2024), hyperparameter optimization is typically modeled as a bandit problem in which the goal is the minimization of the training loss over a set of discrete configurations. The configurations may include prompt designs for the fine-tuning of large language models (LLMs) (Zhou et al., 2023;Quach et al., 2023;Schneider et al., 2024), architectural choices related to model scale (Nasir et al., 2024) or to the timing of early decisions (Schuster et al., 2022), or policy settings for offline reinforcement learning (Paine et al., 2020;Fujimoto & Gu, 2021).
The learn-then-test (LTT) method introduced by (Angelopoulos et al., 2021) has recently emerged as a formal framework to address hyperparameter selection over a discrete space. LTT treats the calibration task of hyperparameter selection as a multiple hypothesis testing (MHT) problem. Accordingly, it associates each hyperparameter in a candidate set to the null hypothesis that the hyperparameter does not meet a reliability requirement on the population risk. Hypotheses are tested using p-values, and the probability of mistakenly detecting a hyperparameter as reliable is guaranteed via family-wise error rate (FWER)-controlling statistical procedures (Keselman & Rogan, 1977). This way, LTT ensures finite-sample, high-probability guarantees on the population risk of the selected hyperparameters.
To prevent p-hacking (Head et al., 2015), LTT's guarantees apply only to non-adaptive MHT procedures. However, related work on hyperparameter optimization has shown the significant benefits that can be accrued by adaptive exploration strategies that test hyperparameters sequentially in a data-driven manner (Swersky et al., 2014;Rakotoarison et al., 2024). This work aims at improving the data efficiency of LTT by leveraging recent advances in sequential MHT based on e-processes (Vovk & Wang, 2021;Waudby-Smith & Ramdas, 2024;Xu et al., 2021).
this section cite: ['b44', 'b54', 'b28', 'b31', 'b26', 'b64', 'b36', 'b62', 'b66', 'b36', 'b42', 'b32', 'b43', 'b34', 'b15', 'b0', 'b21', 'b19', 'b53', 'b37', 'b58', 'b61', 'b63']

Section: Related Work
Finite-sample statistical guarantees on inferential outputs can be obtained via conformal prediction methods (Shafer & Vovk, 2008;Angelopoulos et al., 2023) and, more generally, via conformal risk control and risk-controlling set-valued predictions (Angelopoulos et al., 2022;Bates et al., 2021). Figure 1. An example application of aLTT to reliable prompt optimization (Zhou et al., 2023;Quach et al., 2023;Schneider et al., 2024). A set Λ of candidate prompts for a movie recommender is generated using an LLM and/or prior experience. Prompts serve as an example of a discrete set of hyperparameters to be optimized using aLTT. The goal is to select a subset Λrel ⊆ Λ of prompts that guarantee a sufficiently high recommendation accuracy. To this end, aLTT applies a sequence of data-dependent testing rounds with adaptive termination. Specifically, at each testing round t, aLTT estimates the performance of a subset of hyperparameters I t ⊆ Λ through held-out data or real-world testing. The subset to be tested is selected based on prior testing outcomes, and the process stops as soon as a sufficiently large reliable subset Λrel is identified. An additional post-calibration selection step can be applied to choose a single hyperparameter λ from the selected subset Λrel based on users preferences.
These methods calibrate classification or regression models by setting a threshold hyperparameter based on held-out data to control the size of a prediction set. Calibrated predictors can be leveraged in predict-then-optimize control tasks to offer reliability guarantees (Vovk & Bendtsen, 2018;Lindemann et al., 2023;Zecchin et al., 2024).
Beyond prediction sets, the problem of calibrating an AI app via the selection of hyperparameters from a candidate pool has been addressed through the LTT framework (Angelopoulos et al., 2021). For example, as illustrated in Figure 1, for the problem of prompt engineering (Zhou et al., 2023), the initial set of candidate hyperparameters encompasses instruction prompt templates generated by an LLM and/or via prior experience. Leveraging p-value-based MHT via FWER-controlling procedures, LTT selects a subset of candidates that come with high-probability population risk guarantees.
FWER guarantees are often too conservative, potentially resulting in empty calibration sets. The false discovery rate (FDR) is an alternative and less strict criterion that is often preferred for MHT in fields such as genetics (van den Oord & Sullivan, 2003), neuroimaging (Genovese et al., 2002), online advertising (Berman & Van den Bulte, 2022), and finance (Harvey & Liu, 2020).
E-values have gained popularity in MHT due to their advantages over p-values (Vovk & Wang, 2021;Shafer & Vovk, 2019;Ramdas et al., 2020). Similarly to p-values, e-values measure the statistical plausibility of the null hypothesis. Specifically, an e-value can be thought of as a special type of p-value that has additional robustness properties. The key property of interest in this paper is that, unlike p-values, e-values can be readily combined to obtain e-processes, making it possible to devise sequential testing strategies with anytime safety (Wang & Ramdas, 2022;Ramdas et al., 2023). E-processes have been applied to problems such as sequential change detection (Shin et al., 2022), multiple bandit testing (Xu et al., 2021), two-sample testing (Shekhar & Ramdas, 2023), and mean estimation of bounded random variables (Waudby-Smith & Ramdas, 2024).
Hyperparameter optimization is a vast field focused on the optimization of the hyperparameters of training algorithms, such as learning rate, weight decay, and dropout rate (Swersky et al., 2014;Pedregosa, 2016;Maclaurin et al., 2015;Lindauer et al., 2022). Hyperparameter optimization typically operates in continuous domains, and it can serve as a preliminary step for the identification of candidate hyperparameters. While hyperparameter optimization does not provide statistical guarantees on the population risk, the goal of hyperparameter selection methods such as LTT is to formally test a subset of candidate hyperparameters for statistical validity.
this section cite: ['b46', 'b2', 'b1', 'b4', 'b66', 'b36', 'b42', 'b57', 'b26', 'b64', 'b0', 'b66', 'b56', 'b17', 'b8', 'b18', 'b58', 'b47', 'b39', 'b60', 'b40', 'b49', 'b63', 'b48', 'b61', 'b53', 'b35', 'b29', 'b25']

Section: Main Contributions
The main contributions of this paper are as follows.
• We introduce adaptive LTT (aLTT), a data-efficient hyperparameter selection method that provides finite-sample guarantees on the population risk of AI apps. The main technical underpinning of aLTT is e-process-based MHT, which supports statistical validity while enabling data-dependent sequential testing (Xu et al., 2021). Unlike LTT, as illustrated in Figure 1, aLTT adaptively tests subsets of hyperparameters that are chosen based on the evidence accumulated in the previous rounds, allowing also for the early termination of the calibration process. aLTT guarantees rigorous control over FWER and FDR, while significantly reducing the number of testing rounds.
• We study two practical scenarios requiring hyperparameter selection, namely online policy selection for offline reinforcement learning (Fujimoto & Gu, 2021) and automated prompt engineering (Zhou et al., 2023). In both cases, aLTT is shown to deliver reliable and effective hyperparameters using only a small fraction of the testing rounds required by LTT.
this section cite: ['b63', 'b15', 'b66']

Section: Problem Definition

this section cite: []

Section: Setting
Let M λ be an AI app whose operation is determined by a vector of hyperparameters λ. The performance of a hyperparameter vector λ when tested at input data Z is measured by a risk function R(λ, Z) ∈ [0, 1]. Accordingly, the population risk with respect to an unknown data distribution P Z is defined as
R(λ) = E P Z [R(λ, Z)].
(1)
As illustrated in Figure 1, for a given discrete subset Λ = {λ 1 , . . . , λ N } of hyperparameters and a user-specified reliability level α ∈ [0, 1], we aim at determining the subset of hyperparameters in set Λ that conforms with the required reliability level α, i.e.,
Λ rel = {λ ∈ Λ : R(λ) ≤ α}.(2)
The complementary set, comprising unreliable hyperparameters, is accordingly defined as
Λ unrel = Λ \ Λ rel = {λ ∈ Λ : R(λ) > α}.(3)
Since identifying the entire set Λ rel in (2) is impossible owing to the lack of knowledge about the data distribution P Z , the goal is producing a subset of hyperparameters Λrel ⊆ Λ that contains as many reliable hyperparameters from subset Λ rel as possible while controlling the number of unreliable hyperparameters from subset Λ unrel mistakenly included in subset Λrel .
this section cite: []

Section: Performance Criteria
Definition 2.1 ((α, δ)-FWER-controlling set). For a given reliability level α ∈ [0, 1] and an error level δ ∈ [0, 1], a hyperparameter subset Λrel ⊆ Λ is (α, δ)-FWER-controlling set if it satisfies the requirement
FWER( Λrel ) := Pr |Λ unrel ∩ Λrel | ≥ 1 ≤ δ. (4)
where the probability is evaluated with respect to the distribution of the subset Λrel .
The FWER guarantee (4) imposes that the probability that the calibration set Λrel contains an unreliable hyperparameter is bounded by δ.
Definition 2.2 ((α, δ)-FDR-controlling set). For a given reliability level α ∈ [0, 1] and an error level δ ∈ [0, 1], a hyperparameter subset Λrel ⊆ Λ is (α, δ)-FDR-controlling set if it satisfies the inequality
FDR( Λrel ) := E |Λ unrel ∩ Λrel | | Λrel | | Λrel | ≥ 1 ≤ δ, (5
)
with the average evaluated with respect to the distribution of the subset Λrel .
Accordingly, a testing procedure that outputs (α, δ)-FDRcontrolling sets guarantees that the expected fraction of unreliable hyperparameters in the predicted set Λrel is bounded by δ. Thus, ensuring the (α, δ)-FWER condition automatically also guarantees the (α, δ)-FDR requirement.
Since any FWER or FDR level can be trivially satisfied by a procedure that returns the empty set Λrel = ∅, it is important to gauge the informativeness of the testing procedure via the true positive rate (TPR), which corresponds to the expected fraction of reliable models in the predicted set Λrel , i.e.,
TPR( Λrel ) = E |Λ rel ∩ Λrel | |Λ rel | .(6)
this section cite: []

Section: Sequential and Adaptive Hyperparameter Selection
To produce the estimated subset of reliable hyperparameters, Λrel , we adopt a general sequential testing procedure that, at each round t ≥ 1, operates as follows.
1 Hyperparameter subset selection: A subset of hyperparameters I t ⊆ Λ is selected for testing.
2 Testing: Empirical risk estimates R t = {R(λ i , Z t i )} λi∈I t are obtained, one for each candidate hyperparameter λ i in the selected subset I t , using held-out data or real-world testing. The random variable Z t i ∼ P Z describes the data used to test hyperparameter λ i at round t. The random variables {Z t i } i∈I t can be arbitrarily dependent, and thus one may reuse the same data to test all hyperparameters λ ∈ I t .
3 Evidence update: Evidence accumulated up to time t, including both the observed risks and the subset of queried models, is updated as
D t = D t-1 ∪ {(I t , R t )}.
The testing procedure outlined above is fully specified by the tuple Π = ({Q t } t≥1 , A, T ), encompassing a family of acquisition policies {Q t } t≥1 , a decision rule A, and a calibration horizon T , which are defined as follows.
• Acquisition policy: At each round t, the acquisition policy Q t determines the hyperparameters I t to be tested at step 1 . If the policy Q t uses the evidence D t-1 to select the hyperparameter set I t , it is said to be adaptive; otherwise, it is non-adaptive. Both adaptive and non-adaptive acquisition policies can incorporate prior knowledge, which we denote as D 0 .
• Decision rule: The decision rule A uses the evidence D T available at the end of the last calibration round T to produce the estimated set Λrel of reliable hyperparameters.
• Number of calibration rounds: The number of calibration rounds T , is said to be adaptive, if the stopping condition T = t is determined by the evidence D t-1 . Otherwise, when it is predetermined based solely on prior knowledge D 0 , the calibration horizon T is said to be non-adaptive.
this section cite: []

Section: (Non-Adaptive) Learn-then-Test
In this section, we review LTT, a non-adaptive hyperparameter selection procedure devised to meet the (α, δ)-FWER guarantee (Angelopoulos et al., 2021). LTT associates to each hyperparameter λ i ∈ Λ the null hypothesis
H i : R(λ i ) > α(7)
that the population risk R(λ i ) in (1) violates the target reliability level α. For each null hypothesis H i a p-value P i is a non-negative random variable that satisfies the inequality
Pr[P i ≤ x|H i ] ≤ x(8)
for every x ∈ [0, 1]. By the definition (8) a p-value P i provides evidence for the validity of hypothesis H i . This is in the sense that a small value of P i is unlikely to occur if H i is true.
Given a pre-specified calibration horizon T and a nonadaptive acquisition policy {Q t } t≥1 , LTT queries at each round t ≥ 1 the subset of hyperparameters Q t (D 0 ) = I t , obtaining the corresponding risk estimates R t = {R(λ i , Z t i )} λi∈I t . LTT uses the accumulated evidence at the end of the testing process, D T , to compute a valid p-value for the null hypothesis (7) using, for instance, the Hoeffding-Bentkus concentration inequality introduced in (Bates et al., 2021). Based on the collection of p-values P = {P i } N i=1 , LTT selects a subset of hyperparameters ΛLTT using a FWERcontrolling algorithm A FWER (P). A variant of LTT that is FDR-controlling can be readily obtained by using an FDRcontrolling selection rule A FDR (P). Examples of FWER and FDR controlling procedures are provided in the Supplementary Material.
this section cite: ['b0', 'b4']

Section: Adaptive Learn-Then-Test
In this section, we introduce adaptive LTT (aLTT), a hyperparameter selection scheme that supports adaptive acquisition policies and an adaptive number of calibration rounds. The algorithmic description of aLTT is given in Algorithm 1.
this section cite: []

Section: Hypothesis Testing via E-Processes
The proposed aLTT scheme applies MHT based on e-values and e-processes (Shafer, 2021;Ramdas et al., 2023). For each null hypothesis H i in (7), an e-value E i is a nonnegative random variable with an expectation no larger than 1 when H i is true, i.e.,
E[E i |H i ] ≤ 1.(9)
By Markov's inequality, an e-value E i can be turned into a p-value P i as P i = 1/E i , since the inequality (8) is satisfied as
Pr 1 E i ≤ x H i ≤ E[E i |H i ]x ≤ x, ∀x ∈ [0, 1]. (10
)
For each null hypothesis H i , given an observation Z and a fixed µ ∈ (0, 1/(1α)), a valid e-value is given by (Waudby-Smith & Ramdas, 2024)
E i = (1 + µ(α -R(λ i , Z))).(11)
The e-value (11) has the interpretation of wealth growth in a betting setting. Accordingly, one can think of parameter µ in (11) as the amount of the current wealth that the gambler bets on the hypothesis H i being false, i.e., on the validity of the assumption R(λ i ) ≤ α that the hyperparameter λ i is reliable. In fact, if µ > 0, when R(λ i , Z) ≤ α, the gambler's wealth in (11) increases; while, when R(λ i , Z) > α, the quantity (9) the gambler's wealth (11) decreases.
An e-process for hypothesis H i is a sequence of random variables {E t i } t≥1 such that, for any stopping time T , which may depend on all previously collected evidence, the random variable E T i is a valid e-value. Using the e-value (11), considering the general iterative testing framework in Section 2.3, an e-process for the null hypothesis H i in (7) can be obtained as the product
E t i = τ ≤t:λi∈I τ (1 + µ τ i (α -R(λ i , Z τ i ))),(12)
where the betting strategy µ t i ∈ (0, 1/(1α)) can be optimized as a function of the past risk estimates {R(λ i , Z τ i )} τ <t and E 0 i = 1. Based on the discussion above, the e-process (12) represents the wealth accumulated up to time t by a gambler making sequential bets {µ τ i } τ <t (Shafer & Vovk, 2019;Waudby-Smith & Ramdas, 2024). As such, the gambler's wealth up to time t, E t i , provides evidence against the null hypothesis that the hyperparameter λ i is unreliable.
In a similar way, an anytime-valid p-value for hypothesis H i is a sequence of random variables {P t i } t≥1 such that, for any stopping time T , the random variable P T i is a valid p-value. Given an e-process {E t i } t≥1 for hypothesis H i , the sequence
P t i = 1 max τ ≤t E τ i (13)
is an anytime-valid p-value for the hypothesis H i (Ramdas et al., 2023).
Algorithm 1 Adaptive Learn-Then-Test (aLTT) Require: Candidate hyperparameters Λ, prior knowledge D 0 , reliability level α, error tolerance level δ, acquisition policy {Q t } t≥1 , FWER/FDR-controlling selection rule A FWER (•)/A FDR (•), betting strategy {µ t i } t≥1,i=1,...,N , maximum number of iterations t max and minimal hyperparameter set cardinality d Ensure: (α, δ)-FWER/FDR-controlling hyperparameter set ΛaLTT,T t ← 1 while t ≤ t max ∧ | ΛaLTT,t | ≤ d do Select hyperparameters I t = Q t (E t-1 ) and receive risk estimates R t = {R(λ i , Z t i )} λi∈I t Update evidence D t = D t-1 ∪ {(I t , R t )} and e-processes E t as in (14) if FWER-control then Compute p-values P t as in (13) ΛaLTT,t ← A FWER (P t ) else if FDR-control then ΛaLTT,t ← A FDR (E t ) end if t ← t + 1 end while Return: ΛaLTT,T 4.2. Adaptive Acquisition Policy aLTT applies an adaptive acquisition policy {Q t } t≥1 and an adaptive calibration horizon T . Specifically, at each calibration round t ≥ 1, aLTT's acquisition policy Q t uses the e-processes E t-1
= {E t-1 i } N
i=1 in (12) to choose which subset of hyperparameters, I t , to test next. Examples of acquisition functions I t = Q t (E t-1 ) will be provided in the next section.
For the selected hyperparameters in set I t , aLTT obtains the risk estimates R t = {R(λ i , Z t i )} λi∈I t and updates the associated e-processes using the recursive formula (12), i.e.,
E t i = (1 + µ t i (α -R(λ i , Z t i )))E t-1 i , if λ i ∈ I t E t-1 i , otherwise.(14)
With this information, a prediction set ΛaLTT,t is evaluated by employing either an FWER-controlling method A FWER (P t ) based on the p-values P t = {P t i } N i=1 in (13); or an FDR-controlling procedure A FDR (E t ), such as the e-Benjamini-Hochberg (eBH) method, reviewed in the Supplementary Material (Wang & Ramdas, 2022), using directly the e-values (14).
aLTT terminates the calibration procedure whenever there are at least d hyperparameters in set ΛaLTT,t , i.e., | ΛaLTT,t | ≥ d, or a maximum number of iterations t max have been reached. This allows aLTT to stop the data acquisition phase early when a sufficiently large number of reliable hyperparameters have been identified. .
this section cite: ['b45', 'b40', 'b47', 'b61', 'b40', 'b60']

Section: Hyperparameter Subset Selection
At the end of the calibration process, aLTT uses the current e-processes E T to generate the final prediction set ΛaLTT,T . By the anytime validity properties explained in Section 4.1, if an FWER-controlling method A FWER (P T ) is used, the resulting set ΛaLTT,T is (α, δ)-FWER-controlling; while if an FDR-controlling method is used, the resulting set ΛaLTT,T = A FDR (E T ) is (α, δ)-FDR-controlling.
Theorem 4.1. Given a hyperparameter set Λ, a reliability level α ∈ [0, 1], and an error level δ ∈ [0, 1], aLTT with an FWER or FDR-controlling selection rule returns a final prediction set ΛaLTT,T that satisfies (α, δ)-FWER control or (α, δ)-FDR control, respectively.
this section cite: []

Section: Applications

this section cite: []

Section: Online Policy Selection for Offline Reinforcement Learning
Offline reinforcement learning enables the training of control policies based on a fixed data set collected by using a possibly unknown behavior policy, without any online interaction with the environment (Levine et al., 2020). However, the estimate of the performance of the trained policies obtained from offline data can differ substantially from the actual performance in the real world. This makes it practically essential to validate the policies' performance via online interaction with the environment (Paine et al., 2020;Liu et al., 2023).
To reduce the cost and potential harm of online validation procedures, the number of online interactions of the pretrained candidate policies with the real world must be kept to a minimum (Garcıa & Fernández, 2015). To this end, in this subsection, we investigate the potential benefits of the proposed aLTT framework as a means to select a subset of candidate policies that enjoy performance guarantees with respect to the real-world environment.
this section cite: ['b23', 'b34', 'b27', 'b16']

Section: PROBLEM DEFINITION
We assume a standard Markov decision process (MDP) E = {S, A, P s ′ |s,a , P r|s,a } specified by a state space S; an action space A; a transition kernel P s ′ |s,a , defining the conditional distribution of the next state s ′ ∈ S given the current state s ∈ S and action a ∈ A; and a conditional reward distribution P r|s,a given state s ∈ S and action a ∈ A. We assume that the reward r is bounded and normalized in the [0, 1] interval.
We are given a set of pre-trained control policies Π = {π 1 , . . . , π N }, mapping an observed state s ∈ S to the random action a ∼ π i (s) ∈ A. Each policy π i is identified by a hyperparameter λ i . The goal is to select a subset Λrel ⊂ Λ of hyperparameters that yield reliable policies by using a limited amount of interactions with the real world.
Reliability is measured via the cumulative reward obtained by a policy λ on the MDP E, which is defined as
R(λ, Z) = 1 K K k=1 r k , (15
)
where K is the length of the episode, and the per-episode random variable Z encompasses the initial state s 1 ∼ P s 1 along with the sequence a 1 , r 1 , s 2 , a 2 , r 2 , . . . , s K , a K , r K with actions a k ∼ π i (s k ), rewards r k ∼ P r k |s k ,a k and MDP transitions s k+1 ∼ P s k+1 |s k ,a k . For a user-specified reliability level α ∈ (0, 1), the subset of reliable policies Λ rel ⊆ Λ includes all policies in Λ with average cumulative reward larger than α, Λ rel = {λ ∈ Λ : R(λ) = E P Z [R(λ, Z)] > α}, while the complementary set Λ unrel = Λ \ Λ rel includes the policies that do not satisfy the given reliability requirement.
Control policies are tested sequentially by following the procedure described in Section 2.3, such that at each calibration round t a policy λ t i ∈ Λ is tested using online interactions with the MDP E to obtain the episodic reward value R t i = R(λ t i , Z t i ) in ( 15). For an error threshold δ ∈ [0, 1], the goal of reliable online policy selection is to return a prediction set Λrel ⊆ Λ that is either (α, δ)-FDR controlling or (α, δ)-FWER controlling with a TPR that is as large as possible.
this section cite: []

Section: RESULTS
In our experiments, we consider the Half Cheetah control problem from the OpenAI Gym MuJoCo tasks (Todorov et al., 2012) and use control policies obtained via the offline reinforcement learning algorithm TD3+BC (Fujimoto & Gu, 2021). The TD3+BC algorithm leverages an offline data set D to optimize policies by maximizing the standard deterministic policy gradient objective (Silver et al., 2014), combined with a behavioral cloning regularization term, whose strength is controlled by a hyperparameter λ in [Eq. 5] (Silver et al., 2014). We produce N = 20 different control policies by setting the hyperparameter λ in the TD3+BC training objective on an evenly spaced grid in the interval [0.25, 5]. Unless stated otherwise, we consider a target reliability α = 0.57 and a target FDR requirement δ = 0.1.
We evaluate aLTT with an ϵ-greedy acquisition policy Q t that, at every calibration round t, with probability 1ϵ, selects the hyperparameter λ t i not included in ΛaLTT,t that is associated with the largest e-process value; otherwise, it picks uniformly at random a hyperparameter not in ΛaLTT,t . For reference, we also consider aLTT with a non-adaptive acquisition policy that, at each round t, picks uniformly at random the hyperparameter to be tested regardless of the prediction outcome ΛaLTT,t and the e-process values. As a benchmark, we implement LTT with a random uniform acquisition policy and p-values obtained from the e-processes as in (13). Recall that LTT produces a decision at the end of the calibration process, here at round T = 5000.
Finally, the value of the parameter µ t i in aLTT is set by following the approximate growth rate adaptive to the particular alternative (aGRAPA) betting strategy in (Waudby-Smith & Ramdas, 2024) with other adaptive and nonadaptive betting strategies evaluated in the Supplementary Material.
In Figure 2, we compare the TPR of LTT and aLTT as a function of the calibration round t. We target FWER control on the left and FDR control on the right. By construction, LTT returns uninformative hyperparameter sets up until the termination of the testing procedure. The performance of LTT is the same as aLTT with a non-adaptive acquisition policy, i.e. with ϵ = 1, at t = T . aLTT with an ϵ-greedy acquisition function can benefit from the accumulated evidence to adaptively determine the models to test next. As ϵ decreases, and thus the acquisition policy becomes increasingly driven by evidence, the TPR increases from 0.32 to 0.85 in the case of FWER control and from 0.4 to 0.85 in the case of FDR control. Finally, we note that the TPR of the schemes under FDR control is larger than that obtained under FWER control, reflecting the stricter reliability requirement of FWER control.
In Figure 3, we report the FWER and FDR of aLTT with ϵ = 0.25 as a function of the tolerated error level δ. As the error level δ increases, the empirical FWER and FDR increase accordingly, remaining below the maximum target level δ. However, since FWER control is a stricter requirement, the aLTT prediction set obtained under FWER control delivers lower FDR and FWER levels as compared to aLTT with FDR control.
this section cite: ['b55', 'b15', 'b51', 'b51', 'b61']

Section: Reliable Automated Prompt Engineering
Prompt engineering focuses on designing and refining input instructions for LLMs (Reynolds & McDonell, 2021;Shin et al., 2020). Recent studies have demonstrated the effectiveness of methods that automate this search using LLMs as prompt generators (Zhou et al., 2023;Zhang et al., 2023). However, while LLMs are capable of producing high-quality instruction templates, at a level comparable to human annotators, supervision and testing remain essential to filter out poorly performing prompts.
In this section, we propose applying aLTT to automated prompt engineering (Zhou et al., 2023) to generate instructions with statistical performance guarantees. As illustrated in Figure 1, given a set Λ of candidate instructions generated by an LLM, we use aLTT to sequentially test the instruc- True positive rate of LTT and aLTT with ϵ-greedy acquisition policy for ϵ ∈ {0.25, 0.5, 0.75, 0.95} and non-adaptive acquisition. On the left panel, the prediction sets satisfy FWER control, while on the right panel they meet FDR requirements. In both cases, the tolerance level is δ = 0.1. Results are averaged over the tasks in (Honovich et al., 2022) that yield non-empty reliable prompts set.
tions and identify a sufficiently large number of prompts that meet user-defined reliability requirements.
this section cite: ['b41', 'b50', 'b66', 'b65', 'b66', 'b20']

Section: PROBLEM DEFINITION
Consider a target LLM f (•) that, when fed the concatenation [λ, X] of an instruction λ and data X, produces an output text f ([λ, X]). For example, in the task of movie recommendation illustrated in Figure 1, each prompt λ corresponds to an instruction to generate movie titles similar to those provided in the input list X; while the output f ([λ, X]) represents a list of recommended movies. Following (Zhou et al., 2023;Zhang et al., 2023), the candidate set Λ of instructions, corresponding to the hyperparameters to be tested, is generated using a separate LLM. In particular, we use the Llama3 8B Instruct LLM (Dubey et al., 2024) as the target model f (•), while the initial instruction set Λ is generated through the Llama3.3 70B Instruct model (meta, 2025) using the forward generation mode detailed in (Zhou et al., 2023).
For each instruction λ, the ability of the prompted model f ([λ, X]) to generate high-quality outputs for a test datum
Z = (X, Y ) is measured by a task-dependent loss function R(λ, Z) = ℓ(f ([λ, X]), Y ) ∈ [0, 1].
Focusing on tasks from the instruction induction data set (Honovich et al., 2022), we specifically adopt the 0-1 loss tailored to the given task (Zhou et al., 2023). The goal is to determine a subset Λ rel ⊆ Λ that complies with the condition (2) with a target execution error α = 0.2, and FWER and FDR requirements given by the tolerance parameter δ = 0.1.
this section cite: ['b66', 'b65', 'b12', 'b66', 'b20', 'b66']

Section: RESULTS
For the outlined prompt engineering problem, we evaluate aLTT using an ϵ-greedy acquisition strategy with ϵ ∈ {0.25, 0.5, 0.75, 0.95}, as well as a non-adaptive acquisition strategy whereby the instruction to be tested is selected randomly and independently from prior testing rounds. All schemes employ an online Newton step betting strategy (Waudby-Smith & Ramdas, 2024) for the e-value (12).
In Figure 4, we report the TPR of LTT and aLTT as a function of the number of rounds t. Results are averaged over tasks that yield a final non-empty set Λrel . Plots for individual tasks are included in Appendix B.3. Under both FDR and FWER control, aLTT reduces the number of testing rounds, or equivalently the number of LLM executions, required to achieve a given TPR. In particular, LTT produces a non-empty set only at the end of the testing process, attaining a final TPR less than half of the TPR achieved by aLTT with ϵ-greedy acquisition strategy and ϵ = 0.25. Moreover, aLTT with ϵ = 0.25 identifies 50% of the reliable instructions within the first 1000 testing rounds, while the non-adaptive testing procedure identifies less than 10% of the reliable instructions with the same number of rounds.
The improved efficiency of aLTT translates into superior performance in downstream tasks. To illustrate this point, we follow the described hyperparameter selection process with a post-selection phase that identifies a single hyperparameter λ from the estimated set Λrel of reliable instructions (see Figure 1). Specifically, we adopt the shortest instruction λ ∈ Λrel . This post-selection criterion is motivated by the computational benefits of processing shorter prompts.
In Figure 5, we report the length of the selected instruction λ as a function of the target execution accuracy 1α. Re-sults are averaged over tasks with a final non-empty set Λrel , while plots for individual tasks are included in Appendix B.3. Stricter accuracy requirements are seen to reduce the number of discovered reliable instructions, leading to an increase in the minimal instruction length within Λrel . However, across all reliability levels α, aLTT consistently delivers the shortest instructions, outperforming alternative schemes. This advantage is to be attributed to the data-adaptive acquisition strategy implemented by aLTT, which enables the discovery of a larger number of reliable instructions.
this section cite: ['b61']

Section: Conclusion
We introduced aLTT, a novel framework for hyperparameter selection that implements data-dependent sequential testing via early termination. Unlike the existing LTT, which builds on p-value multiple hypothesis testing (MHT), aLTT is based on sequential MHT via e-processes (Xu et al., 2021). In practical scenarios, including the problem of prompt engineering, this results in more efficient and flexible calibration procedures that maintain statistical validity and the same discovery power as LTT by using only a fraction of testing rounds.
Potential extensions of the aLTT framework include the study of scenarios characterized by distribution shift, data reuse (Wang et al., 2025) and simulation-aided calibration. . Size of the prediction set returned by LTT and aLTT with a top-K ϵ-greedy acquisition policy for varying reliability levels α. As α increases, the reliability requirement becomes stricter, requiring a larger expected reward R(λ) = EP Z [R(λ, Z)] > α. For all values of α, aLTT is able to discover a larger number of reliable policies.
this section cite: ['b63', 'b59']

Section: References
Ref_id:b0 Title: Learn then test: Calibrating predictive algorithms to achieve risk control Year: (2021)
Ref_id:b1 Title: Conformal risk control Year: (2022)
Ref_id:b2 Title: Conformal prediction: A gentle introduction Year: (2023)
Ref_id:b3 Title: Proportionate progress: A notion of fairness in resource allocation Year: (1993)
Ref_id:b4 Title: Distribution-free, risk-controlling prediction sets Year: (2021)
Ref_id:b5 Title: Multiple testing in clinical trials Year: (1991)
Ref_id:b6 Title: Controlling the false discovery rate: a practical and powerful approach to multiple testing Year: (1995)
Ref_id:b7 Title: The control of the false discovery rate in multiple testing under dependency Year: (2001)
Ref_id:b8 Title: False discovery in A/B testing Year: (2022)
Ref_id:b9 Title: Teoria statistica delle classi e calcolo delle probabilita Year: (1936)
Ref_id:b10 Title: Distributional reinforcement learning with quantile regression Year: (2018)
Ref_id:b11 Title: Radio access scheduling using CMA-ES for optimized QoS in wireless networks Year: (2020)
Ref_id:b12 Title: The llama 3 herd of models Year: (2024)
Ref_id:b13 Title: Wireless edge computing with latency and reliability guarantees. Proceedings of the IEEE Year: (2019)
Ref_id:b14 Title: Quantile learnthen-test: Quantile-based risk control for hyperparameter optimization Year: (2016)
Ref_id:b15 Title: A minimalist approach to offline reinforcement learning Year: (2021)
Ref_id:b16 Title: A comprehensive survey on safe reinforcement learning Year: (2015)
Ref_id:b17 Title: Thresholding of statistical maps in functional neuroimaging using the false discovery rate Year: (2002)
Ref_id:b18 Title: False (and missed) discoveries in financial economics Year: (2020)
Ref_id:b19 Title: The extent and consequences of p-hacking in science Year: (2015)
Ref_id:b20 Title: Instruction induction: From few examples to natural language task descriptions Year: (2022)
Ref_id:b21 Title: The Tukey multiple comparison test: 1953-1976 Year: (1977)
Ref_id:b22 Title: Energy delay product. Energy-Efficient High Performance Computing: Measurement and Tuning Year: (2013)
Ref_id:b23 Title: Offline reinforcement learning: Tutorial, review, and perspectives on open problems Year: (2020)
Ref_id:b24 Title: Maximizing energy efficiency in wireless networks with a minimum average throughput requirement Year: (2012)
Ref_id:b25 Title: SMAC3: A versatile Bayesian optimization package for hyperparameter optimization Year: (2022)
Ref_id:b26 Title: Safe planning in dynamic environments using conformal prediction Year: (2023)
Ref_id:b27 Title: When is offline policy selection sample efficient for reinforcement learning Year: (2023)
Ref_id:b28 Title: Fair conformal predictors for applications in medical imaging Year: (2022)
Ref_id:b29 Title: Gradientbased hyperparameter optimization through reversible learning Year: (2015)
Ref_id:b30 Title: Energy efficiency tradeoff mechanism towards wireless green communication: A survey Year: (2015)
Ref_id:b31 Title: Confidence calibration and predictive uncertainty estimation for deep medical image segmentation Year: (2020)
Ref_id:b32 Title: Llmatic: neural architecture search via large language models and quality diversity optimization Year: (2024)
Ref_id:b33 Title: Wireless suite Year: (2020)
Ref_id:b34 Title: Hyperparameter selection for offline reinforcement learning Year: (2020)
Ref_id:b35 Title: Hyperparameter optimization with approximate gradient Year: (2016)
Ref_id:b36 Title: Conformal language modeling Year: (2023)
Ref_id:b37 Title: -context freeze-thaw Bayesian optimization for hyperparameter optimization Year: (2024)
Ref_id:b38 Title: Hypothesis testing with e-values Year: (2024)
Ref_id:b39 Title: Admissible anytime-valid sequential inference must rely on nonnegative martingales Year: (2020)
Ref_id:b40 Title: Gametheoretic statistics and safe anytime-valid inference Year: (2023)
Ref_id:b41 Title: Prompt programming for large language models: Beyond the few-shot paradigm Year: (2021)
Ref_id:b42 Title: Hyperband-based bayesian optimization for black-box prompt selection Year: (2024)
Ref_id:b43 Title: Confident adaptive language modeling Year: (2022)
Ref_id:b44 Title: Toward verified artificial intelligence Year: (2022)
Ref_id:b45 Title: Testing by betting: A strategy for statistical and scientific communication Year: (2021)
Ref_id:b46 Title: A tutorial on conformal prediction Year: (2008)
Ref_id:b47 Title: Game-theoretic foundations for probability and finance Year: (2019)
Ref_id:b48 Title: Nonparametric two-sample testing by betting Year: (2023)
Ref_id:b49 Title: E-detectors: a nonparametric framework for sequential change detection Year: (2022)
Ref_id:b50 Title: Eliciting knowledge from language models with automatically generated prompts Year: (2020)
Ref_id:b51 Title: Deterministic policy gradient algorithms Year: (2014)
Ref_id:b52 Title: Fundamentals of resource allocation in wireless networks: theory and algorithms Year: (2009)
Ref_id:b53 Title: Freeze-thaw Bayesian optimization Year: (2014)
Ref_id:b54 Title: Provably safe systems: the only path to controllable AGI Year: (2023)
Ref_id:b55 Title: Mujoco: A physics engine for model-based control Year: (2012)
Ref_id:b56 Title: False discoveries and models for gene discovery Year: (2003)
Ref_id:b57 Title: Conformal predictive decision making Year: (2018)
Ref_id:b58 Title: E-values: Calibration, combination and applications Year: (2021)
Ref_id:b59 Title: Anytimevalid fdr control with the stopped e-bh procedure Year: (2025)
Ref_id:b60 Title: False discovery rate control with e-values Year: (2022)
Ref_id:b61 Title: Estimating means of bounded random variables by betting Year: (2024)
Ref_id:b62 Title: Rethinking conventional wisdom in machine learning: From generalization to scaling Year: (2024)
Ref_id:b63 Title: A unified framework for bandit multiple testing Year: (2021)
Ref_id:b64 Title: Forking uncertainties: Reliable prediction and model predictive control with sequence models via conformal risk control Year: (2024)
Ref_id:b65 Title: Automatic chain of thought prompting in large language models Year: (2023)
Ref_id:b66 Title: Large language models are human-level prompt engineers Year: (2023)
