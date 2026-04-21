Title: Tackling Biased Evaluators in Dueling Bandits
Abstract: In dueling bandits, an agent explores and exploits choices (i.e., arms) by learning from their stochastic feedback in the form of relative preferences. Prior related studies focused on unbiased feedback. In practice, however, the feedback provided by evaluators can be biased. For example, human users are likely to provide biased evaluation towards large language models due to their heterogeneous background. In this work, we aim to minimize the regret in dueling bandits considering evaluators' biased feedback. We begin with a benchmark case where evaluators' bias information is known. Solving the known-bias case is nontrivial, because the bias cannot be easily decoupled from the feedback. We overcome this challenge and propose an unbiased arm performance estimator and a bias-sensitive dueling bandits algorithm. We manage to analyze the regret, dealing with the complex form of the estimator, and show that the feedback either matching or opposing the ground-truth reduces the regret. Then, we study the case where evaluators' bias information is unknown. The associated estimator can hardly be solved in closed-form due to the non-convexity of the estimator solving problem. We address this challenge and propose an extended bias-sensitive algorithm by incorporating block coordinate descent. This algorithm is proven to achieve the same order of regret (as in the known bias case) with a bounded error. Experiments show that when compared with baselines, our algorithms reduces the regret by up to 86.9%.

Section: Introduction 1.Motivation and Background
Multi-armed bandit (MAB) [1] is a widely used approach for online learning. It explores and exploits a given set of choices (i.e., arms) to minimize a long-term regret. In standard MAB, the reward of the selected arm is commonly represented by a real number, e.g., if pulling an arm of a slot machine returns 5 dollars, then the reward can be represented by 5. As a result, the exploration and exploitation decisions can be made based on these real-valued reward feedback. However, in many practical systems, the real-valued reward feedback is unavailable. For example, consider a company that aims at providing its users with high-quality user experience for question answering tasks by selecting from various large language models (LLMs), e.g., GPT-4 [2], where these LLMs can be thought of arms. Unlike prediction and classification, the output of an LLM is usually paragraphs that are intrinsically subjective. Their ground-truth quality is hard to measure or may not even exist. This makes it difficult to use a real-valued reward to represent the quality of and select an LLM [3,4].
To address the unavailability of real-valued reward feedback, existing studies (e.g., [5][6][7][8][9][10][11][12][13][14]) evaluated arms based on qualitative comparison between a pair of arms, which are referred to as dueling bandits. In these approaches, an agent selects two arms in each round for comparison. The agent then observes the qualitative comparison result between the two arms, based on which the agent makes exploration and exploitation decisions. Interested readers can refer to [15] for a comprehensive survey.
Although these studies (e.g., [9,15]) addressed the lack of real-valued reward feedback, they did not consider an important scenario where the feedback is provided by biased evaluators. For example, [16] suggested that the LLM selection of a company (which serves as an agent) should be based on its users' feedback. However, the users (who serve as evaluators) are humans. Their feedback may be biased due to various factors, e.g., users' expertise or demographic background. Biased feedback can significantly degrade the performance of conventional dueling bandits approaches and increase the long-term regret. We empirically show that the presence of biased evaluators increases the regret of baselines by an average of 8.44 folds (see Appendix K.8).
Some recent studies (e.g., [17,18]) considered biased feedback in conventional MAB settings. However, those approaches are not applicable in dueling bandits due to a lack of real-valued rewards. Other studies (e.g., [19][20][21]) considered pairwise assessment with bias in mobile crowdsourcing, while their goal is to find the best choice or the ranking of choices without considering the long-term exploration-exploitation tradeoff. Thus, their algorithms and analytical frameworks are not applicable to dueling bandits. While adversarial dueling bandits (e.g., [22]) emphasized time-varying winning probability matrices of arms, they do not consider evaluator-specific biased feedback.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b7', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20']

Section: Solution and Approach
In this work, we take into account the feedback provided by biased evaluators and propose biassensitive upper confidence bound (UCB) algorithms with performance guarantee. Our proposed approach for addressing biased evaluators can be readily extended to other dueling bandits algorithms (e.g., relative confidence [10], relative UCB [11], double Thompson sampling [12]) and improve their performance (see Section 5). Specifically, we aim to answer the following questions: Q1 How can we design an unbiased estimator for arm performance and a low-regret dueling bandits algorithm in the presence of evaluators' bias? Q2 What is the performance guarantee of our algorithm? Answering Q1 is challenging. (i) The bias of evaluators is usually unknown a priori. Thus, the algorithm design requires a joint estimation of the evaluators' bias and the winning probability of arms, which makes the corresponding estimator solving problem non-convex. (ii) Even when the bias of evaluators is known, such a design is non-trivial. An intuitive solution is to directly decouple the bias from the observed feedback by equation transformation. However, this is proven to induce an unbounded regret. We overcome the challenge by transforming the estimator design problem into a convex optimization problem and theoretically derive an unbiased estimator and its confidence radius.
Answering Q2 is non-trivial, as the determined estimator and confidence radius from Q1 involve evaluators' heterogeneous bias levels, which makes the regret analysis for conventional dueling bandits inapplicable. We overcome this challenge by applying equation transformation and introducing auxiliary inequalities to support the regret analysis and derive the regret of our proposed algorithms.
Our main contributions are listed as follows:
• To the best of our knowledge, this is the first attempt that considers biased evaluators in dueling bandits. Our approach is applicable to general arm performance models with deterministic winning probability and general bias models that model feedback with conditional probability. Meanwhile, it can be incorporated into existing dueling bandits approaches to reduce their regrets under the presence of evaluators' bias.
• We begin with the case where each evaluators' bias level is known. We overcome challenge Q1-(ii) and propose a bias-sensitive UCB algorithm. To address Q2, we theoretically derive the long-term regret. Analytical results show that our proposed algorithm achieves a sublinear regret, which is of the same order to those in conventional UCB algorithms of dueling bandits.
• We further study the case where each evaluators' bias level is unknown. We overcome challenge Q1-(i) by decoupling evaluators' bias from arm performance estimation when initializing estimators and incorporating block coordinate descent (BCD) [23]. We propose an extended bias-sensitive UCB algorithm, and prove that this extended algorithm achieves the same order of regret as in the known bias case with a bounded error.
• Experiments show that when compared with five baselines, our algorithms reduces the regret by up to 86.9%. The reduction is more significant when the bias levels among evaluators are more heterogeneous. Meanwhile, our estimator can be incorporated into baselines and reduces their regrets by up to 75.9%.
this section cite: ['b8', 'b9', 'b10', 'b21']

Section: System Setup
We consider an agent and a set of M evaluators M = {1, 2, ..., M } whose feedback can be biased. There are a total of K arms, denoted by set K = {1, 2, ..., K}. In each time slot t ∈ T = {1, 2, ..., T }, an arbitrary evaluator arrives. The agent selects two arms for the evaluator. We consider a setting where the evaluator evaluates the selected arms and, at the same time, provides pairwise comparison feedback for the arms. Consider LLM evaluation as an example. A company (agent) selects two LLMs (arms) to serve its users (evaluators). The users observe the inference output of the LLMs and provide pairwise comparison feedback for the two LLMs. The goal of the agent is to minimize the long-term regret of the selected arms (roughly speaking, maximize the chance that the best arm is selected) based on the evaluators' feedback. 2Arm Model: We consider a stochastic setting where an arm outperforms another arm with certain probability [15]. This probability is associated with the ground-truth performance of arms and cannot be observed directly. Let o i ≻ o j denote an observation that arm i ∈ K outperforms arm j ∈ K, and let Pr(o i ≻ o j ) denote the probability that arm i outperforms j. For ease of presentation, we denote
p ij ≜ Pr(o i ≻ o j ).(1)
We assume Pr(o i ≻ o j ) + Pr(o i ≺ o j ) = 1, and do not consider the case where comparing o i and o j leads to tie. As suggested by [24], ties can be handled by giving "half a point" to both arms, reducing the problem to a tie-free case. Note that probability model in (1) generalizes various models as special cases, e.g., Bradley-Terry (BT) model [21] and Logistic model [25].
In dueling bandits, a Condorcet winner (i.e., an arm i with p ij > 1/2 for all j ∈ K \ {i}) may not exist [15]. As in many related works (e.g., [13,14]), we define the best arm using Borda score:
θ i ≜ 1 K-1 j∈K\{i} p ij .(2)
Intuitively, a larger θ i implies a higher probability that arm i beats other arms on average. This metric is suitable. For example, in LLM evaluation, a higher winning probability implies a higher chance that users are satisfied with the inference results of the LLM. We consider Borda winner [13, 14]:foot_1 Definition 1 (Borda Winner). The best arm i * is the arm with the highest Borda score, i.e., i * = arg max i∈K θ i .
this section cite: ['b13', 'b22', 'b19', 'b23', 'b13', 'b11', 'b12']

Section: Evaluator Bias Model:
We use o i ≻ m o j to denote the case where evaluator m ∈ M provides a feedback claiming that arm i outperforms arm j. Note that o i ≻ m o j and o i ≻ o j may not match due to the bias of evaluator m. There are various types of evaluators' bias. In this work, we follow mobile crowdsourcing studies (e.g., [21]) and introduce a coefficient η m to characterize the probability that evaluator m reveals a feedback that matches the ground-truth comparison result:
η m ≜ Pr(o i ≻ m o j | o i ≻ o j ).(3)
That is, given the fact that o i ≻ o j , evaluator m with bias η m claims o i ≻ m o j with probability
η m . Note that Pr(o i ≻ m o j | o i ≻ o j ) + Pr(o i ≺ m o j | o i ≻ o j ) = 1.
Similarly, we exclude the case where the evaluator reports no difference between arms. If this case happens, the evaluator can randomize among the arms with equal probability and provides feedback. The bias model in (3) can characterize various types of bias, such as ambiguity in perception and comparison [26] and diverse roles of the evaluators [21]. Consider bias resulting from diverse roles as an example. If η m = 1, then evaluator m is a perfect evaluator. If η m = 0.5, then evaluator m is a spammer who provides random feedback. If η m = 0, then evaluator m is an attacker which aims to worsen the choice of the agent and always provides opposite feedback.
Based on (3), the probability that evaluator m claims arm i outperforms arm j is given by
p m ij ≜ Pr(o i ≻ m o j ) = η m p ij + (1 -η m )p ji .(4)
Arm Selection and Regret: In time slot t ∈ T , an evaluator arrives, and let m t ∈ M denote this evaluator. The agent selects two arms x 1 (t) ∈ K and x 2 (t) ∈ K for the evaluator using a dueling bandits algorithm (to be proposed in Sections 3 and 4). Let x(t) ≜ {x 1 (t), x 2 (t)}. Note that x 1 (t) ̸ = x 2 (t) must hold before algorithm convergence; otherwise, no comparison between arms is performed and hence there is no exploration in time t. After evaluator m t evaluates both chosen models x 1 (t) and x 2 (t), it sends a binary feedback to the agent, i.e., either
o x1(t) ≻ mt o x2(t) or o x2(t) ≻ mt o x1(t)
. The binary feedback is commonly considered in dueling bandits [15] and is suitable for the scenario that lacks real-valued reward feedback from evaluators. Recall that in the LLM example, it is easy for users to judge which output from the two LLMs is better, while it is difficult for them to give real-valued score for the outputs of LLMs.
In this work, we focus on both average regret and weak regret, which are commonly considered regrets in dueling bandits [1]. The average regret RegA(x(t)) [10,12] and weak regret RegW(x(t))
[27] are defined as the average and maximum Borda score among the two selected arms, respectively:
RegA(x(t)) = θ i * -(θ x1(t) + θ x2(t) )/2,(5)
RegW(x(t)) = θ i * -max{θ x1(t) , θ x2(t) }.(6)
For example, average regret refers to the case where a user retrieves information from the inference outputs of both LLMs. Weak regret refers to the case where a user is satisfied as long as one of the LLMs provides satisfactory output. Since all of our algorithms and theoretical results apply to both average regret and weak regret, we use Reg(x(t)) to denote them.
The goal is to minimize the long-term round-average regret:
min [x(t)] T t=1 1 T T t=1 E[Reg(x(t))].(7)
We solve problem (7) for both known and unknown bias cases in Sections 3 and 4, respectively.
this section cite: ['b19', 'b24', 'b19', 'b13', 'b0', 'b8', 'b10']

Section: Known Bias Case
In this section, we start with the benchmark case where the evaluators' bias η m is known. In practical systems, the bias could be obtained by running pre-evaluation tests, e.g., in the LLM example, the company may estimate user bias through offering queries whose ground-truth answers are known. We consider the setting where the set of available bias is finite. That is, η m ∈ B ≜ {η A 1 , η A 2 , ..., η A B } for all m ∈ M, where B = |B| and the superscript A is short for "available". As long as the number of evaluators is finite, this assumption on finite bias set holds.
We build our algorithm based on UCB. Despite this, our ideas for addressing biased evaluators can be incorporated into various baselines to reduce their regret (see Appendix K.1). Note that even for the known bias case, designing the algorithm is challenging. This is because when estimating the pairwise winning probability of arms, the bias cannot be easily decoupled from the observed feedback provided by evaluators. Meanwhile, the complex form of the winning probability estimator makes deriving the associated confidence radius and analyzing round-average regret further challenging.
this section cite: []

Section: Bias-Sensitive UCB Algorithm
We first present the unbiased estimation of pairwise winning probability of arms and confidence radius calculation respectively. Then, we show the algorithm details.
this section cite: []

Section: 1) Unbiased Arm Performance Estimation:
We aim to design an unbiased estimator of winning probability matrix p ≜ (p ij , i, j ∈ K), 4 which will be incorporated into our bias-sensitive UCB algorithm. Note that it is possible to obtain an unbiased estimator by transforming the problem into conventional dueling bandits via decoupling the bias in (4). However, such an estimator is sensitive to the feedback of spammers, which can lead to an infinite round-average regret (see Appendix A). To deal with this challenge, we first transform the estimator design problem into an optimization problem. Then, we solve the problem to obtain the estimator.
Let N b ij (t) denote the number of feedback claiming o i ≻ m o j , and its evaluator has a bias
η m = η A b . Let B ij (t) ⊆ B denote the set of bias index b such that N b ij (t) + N b ji (t) > 0.
Designing an estimator p(t) = (p ij (t), i, j ∈ K) is equivalent to finding the optimal estimator p(t) that minimizes the difference between the estimated value of p ij using the estimator and the approximate value
pb ij (t) ≜ N b ij (t)/(N b ij (t) + N b ji (t)).
That is, p(t) minimizes the following problem:
min p i,j∈K,b∈Bij (t) (η A b p ij + (1 -η A b )p ji -pb ij (t)) 2 .(8)
Problem ( 8) contains K 2 × M terms, each corresponding to exactly one decision variable p ij . Thus, problem (8) can be equivalently transformed to a set of sub-problems of pij (t):
pij (t) = arg min pij ∥w ij p ij + c ij ∥ 2 ,(9)
where
w ij ≜ (2η A b -1, b ∈ B ij (t)), c ij ≜ (1 -η A b -pb ij (t), b ∈ B ij (t)).
Based on Karush-Kuhn-Tucker (KKT) conditions, the optimal solution to problem (9) satisfies (w ⊤ ij w ij )p ij (t) = -w ⊤ ij c ij . This results in the following unbiased estimator, with proof in Appendix B. Lemma 1 (Arm Performance Estimator). After time slot t, the pairwise winning probability p ij in (1) is estimated by
pij (t) = b∈Bij (t) (2η A b -1) pb ij (t) -(1 -η A b ) b∈Bij (t) (2η A b -1) 2 . (10
)
This estimator is unbiased, i.e., E[p ij (t)] = p ij . Based on (10), if an evaluator tends to be a spammer (i.e., η m is closer to 0.5), a lower weight is assigned to the evaluator's feedback.
2) Confidence Radius Calculation: We now derive the confidence radius of the estimator in Lemma 1. This analysis is more challenging than that in conventional dueling bandits, because the estimator is in the form of a weighted sum of the feedback statistics of evaluators considering their bias. The involved sum, weighting, and shift operations require additional mathematical transformation to solve the confidence radius based on Hoeffding inequality. The proof is given in Appendix C. Definition 2 (Confidence Radius). We define the confidence radius as
Pr(|p ij (t) -p ij | ≤ r ij (t)) ≥ 1 -2/t 2α . That is, r ij (t) is a one-dimensional bound such that |p ij (t) -p ij | ≤ r ij (t)
occurs with a probability no smaller than 1 -2/t 2α , where parameter α > 0 controls the required probability. Proposition 1 (Confidence Radius). The confidence radius r ij (t) in Definition 2 is determined by
r ij (t) = b∈Bij (t) |2η A b -1| α log(t) (N b ij (t)+N b ji (t)) b∈Bij (t) (2η A b -1) 2 , (11
)
where log(t) is of natural base.
this section cite: ['b8']

Section: 3) Algorithm Details:
We now present the bias-sensitive UCB algorithm. The pseudocode is provided in Algorithm 1 of Appendix D. The algorithm iterates for T rounds or until convergence. At the beginning of each time slot t, the agent updates pij (t-1) using (10) and r ij (t-1) using ( 11). Then, it computes the upper confidence bound estimation of probability p ij :
UCB ij (t) = [p ij (t -1) + r ij (t -1)] -,(12)
where [•] -≜ min{•, UCB}. With this operator [•] -, the agent tends to randomly explore if all arms are under-explored. We set UCB = 1 in the experiments [13]. In (12), if pij (t -1) is larger than 1/2, then arm i is likely to outperform arm j based on the historical observation, indicating a higher reward through exploiting model i. If r ij (t -1) is larger, then the uncertainty regarding arms i and j is higher, indicating stronger need to compare arms i and j in the following time slot.
After that, the agents computes the UCB estimation of Borda score:
UCB i (t) = 1 K-1 j∈K\{i} UCB ij (t).(13)
Finally, the agent selects the two arms x 1 (t) and x 2 (t) with the maximum values of UCB i (t):
max x(t) UCB x1(t) (t) + UCB x2(t) (t).(14)
Different from some existing works (e.g., [11]) in dueling bandits that choose the best arm (e.g., with the highest UCB) and its "strongest competitor", our algorithm chooses the best and second best arms (e.g., with the highest and second highest UCB values) for analytical simplicity. In Appendix E, we empirically show that replacing the second arm with the "strongest competitor" may degrade the performance, especially when the number of arms is large or when a Condorcet winner does not exist.
this section cite: ['b11', 'b10', 'b9']

Section: Regret Analysis
We now bound the round-average regret of the proposed algorithm. The proof is given in Appendix F. The proof path follows [13], while it is more difficult due to the complex form of the estimator and confidence radius. Note that we essentially derive the bound for average regret. This bound is also applicable to weak regret by relaxing it to average regret in the proof (see Appendix F).
Theorem 1 (Regret of Bias-Sensitive UCB Algorithm). The bias-sensitive UCB algorithm with T rounds has a round-average regret of
1 T T t=1 Reg(x(t)) ≤ UCB (K(K -1) + 2H) T + 2UCB α log(T ) Γ H + B 2 log(BT 2α ) T + 2BK K -1 • 1 √ T . (15
)
where H = ∞ t=K(K-1)/2+1 t -2α , and
Γ ≜ b∈Bij (t) (2η A b -1) 2 /|B ij (t)|.
According to Theorem 1, we can determine the order of the round-average regret and its sublinearity.
this section cite: ['b11']

Section: Corollary 1 (Sublinear Regret). The round-average regret of Algorithm 1 is sublinear with an order of O( B log(T )/T /Γ).
This sublinearity result is consistent with and generalizes those existing works on dueling bandits without considering evaluators' bias (e.g., [13]). Importantly, Γ reflects the average deviation of the evaluators from spammers. When Γ is larger (i.e., evaluators tend to reveal feedback either matching or opposing the ground-truth), the round-average regret is smaller.
this section cite: ['b11']

Section: Unknown Bias Case
We now solve the case where evaluators' bias is unknown to the agent, and the bias of any evaluator η m belongs to an infinite set [0, 1]. Our approach can be extended to the scenario with finite set of bias by projecting the continuous estimated bias to discrete space. Since the set of evaluators is finite, their bias comprises a finite set B ≜ {η 1 , η 2 , ..., η M }. Let N m ij (t) denote the number of feedback sent by evaluator m and claiming o i ≻ m o j . Let M ij (t) ⊆ M, which can be interpreted as the set of evaluators m such that N m ij (t) + N m ji (t) > 0 for each pair of arms i and j. Let J m (t) denote the set of (i, j) pairs such that N m ij (t) + N m ji (t) > 0 for each m ∈ M. Designing the extended bias-sensitive algorithm is highly non-trivial. This is because the estimation of the arm performance and evaluation bias is highly coupled. In the following, we first present the estimators for arm performance and evaluators' bias. Then, we propose the extended bias-sensitive algorithm that overcomes the aforementioned challenges. Finally, we analyze its regret.
this section cite: []

Section: Arm Performance and Bias Estimation
Let pij (t) and ηm (t) denote the estimation of p ij and η m given our estimator, respectively. After time slot t, the pairwise winning probability p ij in (1) is estimated using the same estimator as in Lemma 1 while replacing the ground-truth η m with the estimated ηm (t), i.e.,
pij (t) = m∈Mij (t) pm ij (t)-(1-ηm (t)) (2η m (t)-1) m∈Mij (t) (2η m (t) -1) 2 . (16
)
Based on a similar idea as estimating the arm performance in Section 3.1, we formulate the problem for estimating the bias of evaluator m ∈ M:
ηm (t) = arg min η 1 2 ∥U m η + b m ∥ 2 + γ 2 ∥η -ηm ∥ 2 . (17
)
In the first term, U m = (2p ij (t) -1, i, j ∈ J m (t)), and b
m = (1 -pij (t) -pm ij (t), i, j ∈ J m (t)), where pm ij (t) ≜ N m ij (t)/(N m ij (t) + N m ji (t)).
It aims to find the best η that minimizes the estimation error of bias given the recent pij (t), similar as that in (9). The second term is introduced for the algorithm to be proposed. Its goal is to restrict the gap between the previous estimation ηm and the new estimation, where γ balances the two terms. Solving (17) via the KKT conditions yields the estimator. Lemma 2 (Bias Estimator). After time slot t, the bias of evaluator η m in (3) is estimated by
ηm (t) = i,j∈Jm(t) (2p ij (t) -1)(p m ij (t) + pij (t) -1) + γ ηm i,j∈Jm(t) (2p ij (t) -1) 2 + γ .(18)
Estimators ( 16) and ( 18) form a system of equations, and solving them jointly yields pij (t) and ηm (t). However, pij (t) and ηm (t) are highly coupled, i.e., the performance estimates depend on the bias estimates and vice versa, and the joint estimation problem is non-convex. Although it is possible to let pij (t) and ηm (t) update iteratively using ( 16) and ( 18), parameter pij (t) usually converges to local optimal solution pij (t) = 0.5 due to the non-convexity. To address this, we decouple the evaluation bias from the arm performance estimation when initializing the estimation in each time slot and propose a BCD-based algorithm [23].
this section cite: ['b7', 'b21']

Section: Extended Bias-Sensitive UCB Algorithm
We present the extended bias-sensitive UCB algorithm. Its pseudocode is given in Algorithm 2 of Appendix G. At the beginning of time slot t, estimators pij (t -1) and ηm (t -1) are computed.
Specifically, estimator pij (t -1) is first set to pm ij (t -1) ≜ N m ij (t -1)/(N m ij (t -1) + N m ji (t -1
)), i.e., the estimation of arm performance ignoring the evaluators' bias. This process decouples the impact of evaluators' bias estimation and that of inaccurate performance and bias estimation in the past time slots. Based on this pij (t -1), estimator ηm (t -1) is computed using (18). Then, according to BCD [23], pij (t -1) and ηm (t -1) are updated in sequence twice. We empirically show in Appendix K.7 that performing such updates twice leads to the best performance. 5 Either increasing or decreasing the rounds of updates leads to regret increase.
After that, the agent estimates the confidence radius rij (t -1) with the estimated bias ηm (t -1):
rij (t -1) = m∈Mij (t) |2η m (t -1) -1| α log(t-1) (N m ij (t-1)+N m ji (t-1)) m∈Mij (t-1) (2η m (t -1) -1) 2 . (19
)
Note that this is not the actual confidence radius for the estimators and thus leads to additional regret in decision making (see Section 4.3). Finally, pij (t -1) and rij (t -1) are substituted into (13) to compute UCB i (t), and the arms that optimize problem (14) are selected.
this section cite: ['b16', 'b21']

Section: Regret Analysis
We first quantify the actual confidence radius under estimators in ( 16) and ( 18), with which we are able to bound the regret of Algorithm 2. The proof is given in Appendix H.
this section cite: []

Section: Lemma 3 (Confidence Radius).
Given the estimators in (16) and (18), the confidence radius is
r • ij (t) = m∈Mij (t) ϕ m ij (t)/(ϵ η m (t)) 2 -φm ij (t) m∈Mij (t) (2η m (t) -1) 2 + m∈Mij (t) |2ηm(t)-1| |ϵ η m (t)| α log(t) N m ij (t)+N m ji (t) m∈Mij (t) (2η m (t) -1) 2 , (20
)
where
ϕ m ij (t) ≜ (p m ij (t) -(1 -η m ))/(2η m -1), φm ij (t) ≜ (p m ij (t) -(1 -ηm (t)))/(2η m (t) -1), ϵ η m (t) ≜ (2η m -1)/(2η m (t)-1).
Quantifying the regret using the difference between rij (t) and r • ij (t) is challenging, because the mapping from the confidence radius to the exact probability an estimation falls within the radius can hardly be solved, due to the complex form of estimators. Thus, we define a parameter ξ(t). Definition 3 (Parameter ξ(t)). For each time slot t, let ξ(t) denote the minimum non-negative value such that ξ(t)
≥ 1 2 HF P |p ij (t) -p ij | ≤ r • ij (t) -P (|p ij (t) -p ij | ≤ rij (t)) , where HF(•) is the tight lower bound of P |p ij (t) -p ij | ≤ r • ij (t) .
As will be seen in Theorem 2, a lower ξ(t) leads to a lower regret. There are various cases that ensure ξ(t) = 0. Although it is hard to derive all the cases due to the complex form of rij (t) and r • ij (t), we list two examples: (i) rij (t) = r • ij (t); (ii) η m (t) = 1 and ηm (t) ∈ (0.5, 1] for all m ∈ M. Then, the round-average regret can be determined, with the proof given in Appendix I. Theorem 2 (Regret of Extended Bias-Sensitive Algorithm). Under Definition 3, the extended biassensitive UCB algorithm based on estimators in (16) and (18) has a round-average regret of
1 T T t=1 Reg(x(t)) ≤ UCB K(K -1) + 2 H + T t=K(K-1)/2+1 ξ(t) T + 2UCB α log(T ) Γ H + B 2 log(BT 2α ) T + 2BK K -1 • 1 √ T . (21
)
When compared with Theorem 1 for known bias case, the round-average regret under unknown bias case has the same order but incorporates an additional bounded error related to ξ(t). If ξ(t) is monotonically decreasing and converges to zero as t → ∞, then this bounded error approaches zero. However, due to the non-convexity of the joint estimation problem, proving this convergence is an open problem under BCD. In Appendix J, we empirically show that this bounded error is small and can approach zero.
this section cite: []

Section: Experiments
We consider that the user bias follows a Beta distribution Beta(α B , β B ) [21]. We use the BT model to model the winning probability of arms [21], i.e., p ij = e si /(e si + e sj ), where s i is a coefficient associated with arm i following Gaussian distribution N (µ, σ 2 ) and a Condorcet winner typically exists under this model. Unless otherwise specified, we set η m ∼ Beta(α B = 2, β B = 1) and s i ∼ N (µ = 0, σ 2 = 2). Through empirical tests, we set
α = α 0 ( m∈Mij (t) (2η m - 1) 2 ) 2 /( m∈Mij (t) |2η m -1|) 2
, where α 0 = 0.51 [11] and η m can be the recent estimated value for unknown bias case. The term α relies on the recent estimation of η m and helps to mitigate the over-exploration due to the presence of evaluators' bias. We set coefficient c = 50. Our code is built based on open source code [28] for dueling bandits. Experiments are conducted on a compute platform with an AMD Ryzen 7 7800X3D (8-core) processor and 64 GB of RAM (4800 MHz). We run each experiment for 100 times and show the average results in this section. The results with standard error can be found in Appendix K.
We compare our algorithms with five baselines in dueling bandits: Relative Confidence (denoted by "RC") [10], Relative UCB (denoted by "RUCB") [11], a Bayesian method Double Thompson Cumulative Average Regret (↓) Cumulative Weak Regret (↓) Arm Heter. σ 2 Bias Concentr. α B Arm Heter. σ 2 Bias Concentr. α B 1.0 2.0 4.0 1.0 2.0 3.0 1.0 2.0 4.0 1.0 2.0 3.0 RC 1374 1338 967 2845 1338 687 596 525 502 1847 525 278 RUCB 1906 2134 1154 2832 2134 1185 1018 1144 719 1829 1144 506 DT 1396 1425 942 2621 1425 640 445 492 375 1428 492 191 MBTW 1220 1509 726 1769 1509 1448 175 162 140 569 162 92 UCB 1283 1426 732 2581 1426 706 553 548 336 1583 548 153 RC-B(*) 1378 1611 1050 2207 1611 803 649 869 727 1119 869 502 RUCB-B(*) 993 1120 709 1191 1120 1055 422 480 370 604 480 446 DT-B(*) 430 411 344 631 411 280 198 210 168 436 210 110 BS-UN(*) 690 689 387 825 689 637 194 161 94 340 161 92 BS-K(*) 654 713 407 554 713 624 116 90 79 60 90 82 Table 1: Performance under diverse arm heterogeneity (denoted by "heter.") and bias concentration (denoted by "concentr.") with 10 arms and 10 evaluators. Our methods are marked with "(*)". The best, second, and third best results are marked in bold, underline, and dashed underline, respectively.
(denoted by "DT") [12], Modified Beat The Winner (denoted by "MBTW") [27], UCB (which follows [13] but omits the cost constraint). Meanwhile, we incorporate our bias-sensitive estimation in Algorithm 2 and obtain bias-sensitive versions of RC, RUCB, and DT (see Appendix K.1). They are denoted by "[Baseline Name]-B". Our Algorithms 1 and 2 are denoted by BS-K and BS-UN for known and unknown bias cases, respectively. We use "(*)" to mark our methods (including the bias-sensitive versions of baselines and our proposed BS-K and BS-UN). Experiments are conducted under unknown bias case, expect for those of BS-K. We show the cumulative regret T t=1 Reg(x(t)) because (i) the values of round-average regret are very small, and (ii) cumulative regret can infer marginal regret in figures. In Table 1, the round-average regret can be obtained by dividing the cumulative regret by T = 10000.
Algorithm Comparison: Tables 1 and 2 show the cumulative regret after 10000 rounds. The algorithm convergence and standard error are shown in Appendix K.2. The numerical bias estimation error can be found in Appendix K.3. We have the following observations. (i) Our proposed BS-UN and BS-K algorithms achieve superior performance under both average and weak regrets, ranked top three among all algorithms for most cases. When compared with RC, RUCB, DT, MBTW, and UCB, the average regret reduction of BS-UN can be up to 71.0%, 70.9%, 68.5%, 56.0%, and 68.0%, respectively; the weak regret reduction of BS-UN can be up to 81.6%, 86.9%, 76.2%, 40.2%, 78.5%, respectively. (ii) The bias-sensitive versions of baselines usually achieve lower regret than their original versions, showing the effectiveness of our estimators. For RUCB and DT, their average regret reduction can be up to 58.0% and 75.9%, respectively; their weak regret reduction can be up to 67.0% and 69.5%, respectively.
Impact of α B : Our BS-K and BS-UN algorithms are more beneficial when bias concentration α B is lower. Specifically, a smaller α B implies a higher degree of evaluators' bias. In Table 1, when α B reduces from 3.0 to 2.0, the average and weak regrets of baselines increase by up to 1.23 times and 2.58 times, respectively. However, the average and weak regret increasing are 0.08 and 0.75 for BS-UN and 0.14 and 0.09 for BS-K, respectively.
Impact of σ 2 : A larger σ 2 implies a higher degree of arm heterogeneity. Since the evaluators' bias is not accounted by the baselines, a larger heterogeneity makes it easier to identify the best arm and hence a lower regret. As the evaluators' bias is accounted by our methods, a moderate heterogeneity can be sufficient for identifying the best arm and reducing the regrets.
Impact of Evaluators and Arms: From Table 2, (i) our methods are not sensitive to the number of evaluators. As the number of evaluators increases from 5 to 20, the average and weak regrets of our BS-UN increase by -0.10 and -0.35 times, respectively; those of our BS-K increase by 0.08 and 0.32 times, respectively. (ii) The increasing in the number of arms increases the regrets of our methods. This is acceptable, because in the LLM evaluation example, the number of users (i.e., evaluators) is always large, while the number of LLMs (i.e., arms) is usually small, e.g., around 10. We further evaluate large-scale settings with 100 evaluators and 100 arms in Appendix K.4.
this section cite: ['b19', 'b19', 'b9', 'b8', 'b9', 'b10', 'b11']

Section: Conclusion and Limitations
This work presents the first study on addressing evaluators' bias in dueling bandits. We overcome the challenge of non-convexity and bias heterogeneity and propose bias-sensitive algorithms with regret bounds. When compared with baselines, our algorithms reduces the regret by up to 86.9%, especially when the evaluators' bias levels are more heterogeneous. Meanwhile, our proposed estimator can be incorporated into baselines and achieve a regret reduction of up to 75.9%.
The main limitations of this work contain four parts. First, the bias of each evaluator is modeled to be deterministic and unchanged across time. To extend the model to stochastic and diverse bias, we may learn from adversarial dueling bandits and extend the techniques from addressing time-varying winning probability to time-varying bias. Second, the regret bound for BS-UN contains a term ξ(t), which was not derived in closed form. It is important to derive the specific expression of it to reveal further insights, overcoming the difficulty in analyzing the performance of a BCD algorithm for non-convex problem. Third, the recent bias modeling is only evaluator-dependent. It is interesting to consider arm-dependent bias, characterizing evaluators' distinctive bias toward arms. Fourth, this work is motivated by human bias in feedback. It would be beneficial to construct real-world experiments with humans for algorithm evaluation.
[27] Erol A. Peköz, Sheldon M. Ross, and Zhengyu Zhang. "DUELING BANDIT PROBLEMS". In: Probability in the Engineering and Informational Sciences 36 (2020), pp. 264-275. URL: https://api.semanticscholar.org/CorpusID:229507898.
this section cite: []

Section: References
Ref_id:b0 Title: Introduction to Multi-Armed Bandits Year: (2019-11)
Ref_id:b1 Title: State of What Art? A Call for Multi-Prompt LLM Evaluation Year: (2024)
Ref_id:b2 Title: Efficient LLM Comparative Assessment: A Product of Experts Framework for Pairwise Comparisons Year: (2024-11)
Ref_id:b3 Title: Beat the mean bandit Year: (2011-07)
Ref_id:b4 Title: Generic exploration and K-armed voting bandits Year: (2013-06)
Ref_id:b5 Title: Regret Lower Bound and Optimal Algorithm in Dueling Bandit Problem Year: (2015-07)
Ref_id:b6 Title: Choice bandits Year: (2020-12)
Ref_id:b7 Title: Neural Constrained Combinatorial Bandits Year: (2023-05)
Ref_id:b8 Title: Relative confidence sampling for efficient on-line ranker evaluation Year: (2014-02)
Ref_id:b9 Title: Relative Upper Confidence Bound for the K-Armed Dueling Bandit Problem Year: (2014-06)
Ref_id:b10 Title: Double Thompson Sampling for Dueling Bandits Year: (2016-12)
Ref_id:b11 Title: Green Dueling Bandits Year: (2023-05)
Ref_id:b12 Title: Think Before You Duel: Understanding Complexities of Preference Learning under Constrained Resources Year: (2024-05)
Ref_id:b13 Title: Preference-based online learning with dueling bandits: A survey Year: (2021-01)
Ref_id:b14 Title: LLMOps: Deployment and Learning in Production Year: (2025-01)
Ref_id:b15 Title: Reward-Biased Maximum Likelihood Estimation for Linear Stochastic Bandits Year: (2020-10)
Ref_id:b16 Title: Exploration through reward biasing: reward-biased maximum likelihood estimation for stochastic multi-armed bandits Year: (2020-07)
Ref_id:b17 Title: Aggregation of pairwise comparisons with reduction of biases Year: (2019-06)
Ref_id:b18 Title: Bias-aware ranking from pairwise comparisons Year: (2024-05)
Ref_id:b19 Title: Pairwise ranking aggregation in a crowdsourced setting Year: (2013-02)
Ref_id:b20 Title: Adversarial Dueling Bandits Year: (2021-07)
Ref_id:b21 Title: Convergence of a Block Coordinate Descent Method for Nondifferentiable Minimization Year: (2021-06)
Ref_id:b22 Title: PAC Rank Elicitation through Adaptive Sampling of Stochastic Pairwise Preferences Year: (2014-07)
Ref_id:b23 Title: The regression analysis of binary sequences Year: (1958)
Ref_id:b24 Title: Crowdsourcing Subjective Annotations Using Pairwise Comparisons Reduces Bias and Error Compared to the Majority-vote Method Year: (2023-10)
Ref_id:b25 Title:  Year: (1906)
Ref_id:b26 Title:  Year: ()
Ref_id:b27 Title:  Year: ()
Ref_id:b28 Title:  Year: ()
Ref_id:b29 Title:  Year: ()
Ref_id:b30 Title:  Year: ()
