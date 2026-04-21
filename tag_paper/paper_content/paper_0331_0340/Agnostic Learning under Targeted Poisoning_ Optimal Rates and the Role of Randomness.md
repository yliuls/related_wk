Title: Agnostic Learning under Targeted Poisoning: Optimal Rates and the Role of Randomness
Abstract: We study the problem of learning in the presence of an adversary that can corrupt an η fraction of the training examples with the goal of causing failure on a specific test point. In the realizable setting, prior work established that the optimal error under such instance-targeted poisoning attacks scales as Θ(dη), where d is the VC dimension of the hypothesis class [Hanneke, Karbasi, Mahmoody, Mehalel, and  Moran (NeurIPS 2022)]. In this work, we resolve the corresponding question in the agnostic setting. We show that the optimal excess error is Θ( √ dη), answering one of the main open problems left by Hanneke et al. To achieve this rate, it is necessary to use randomized learners: Hanneke et al. showed that deterministic learners can be forced to suffer error close to 1 even under small amounts of poisoning. Perhaps surprisingly, our upper bound remains valid even when the learner's random bits are fully visible to the adversary. In the other direction, our lower bound is stronger than standard PAC-style bounds: instead of tailoring a hard distribution separately for each sample size, we exhibit a single fixed distribution under which the adversary can enforce an excess error of Ω( √ dη) infinitely often.

Section: Introduction
Imagine a social network like Facebook or X (formerly Twitter), where recommendation algorithms curate the content shown to users. A business or political entity might attempt to manipulate the system by injecting carefully crafted fake interactions -likes, shares, and comments -into the training data, aiming to subtly boost the visibility of its content for specific target users. The goal is not to disrupt overall system performance, but to force an error on particular instances of interest.
Such scenarios fall outside the scope of classical models of learning, such as PAC learning, which assume that the training data and the test example are independent. When an adversary can modify the training data based on knowledge of the test instance, this independence assumption breaks down. This motivates the study of targeted data poisoning attacks, where the adversary corrupts part of the training set in order to cause failure on a specific test point.
this section cite: []

Section: Main Results
In this subsection, we present the learning model and our main results. The presentation here assumes familiarity with standard concepts from classical learning theory; a more self-contained exposition will be provided in Section 4.
We focus on the instance-targeted poisoning model (see, e.g., Barreno, Nelson, Sears, Joseph, and Tygar [2006]), parameterized by a poisoning budget η ∈ (0, 1) and a sample size n > 0. The interaction proceeds as follows:
Learning under Instance-Targeted Poisoning 1. The adversary selects a distribution D over labeled examples (x, y) ∈ X × {0, 1}.
this section cite: ['b3']

Section: A training sample S
of n examples and a target example (x, y) are drawn independently from D. 3. The adversary observes both S and (x, y), and modifies up to an η-fraction of the examples in S to produce a poisoned sample S ′ . 4. The learner receives S ′ and outputs a hypothesis A(S ′ ) : X → {0, 1}. 5. The learner succeeds if A(S ′ )(x) = y and fails otherwise.
We study this model with respect to a fixed hypothesis class H ⊆ {0, 1} X . As in classical learning theory, we distinguish between the realizable and agnostic settings:
• In the realizable case, there exists a hypothesis h ∈ H with zero loss on D.
• In the agnostic case, no assumption is made on D, and the learner aims to compete with the minimal achievable loss over H.
Previous work by Gao, Karbasi, and Mahmoody [2021a] showed that if the poisoning budget η vanishes with the sample size (η = o(1) as n → ∞), then PAC learnability of H implies learnability under instance-targeted poisoning, both in the realizable and agnostic settings. Their algorithms, however, relies crucially on η being negligible. Subsequently, Hanneke, Karbasi, Mahmoody, Mehalel, and Moran [2022] studied the problem for general values of η, and provided tight bounds on the achievable error in the realizable setting as a function of η and the VC dimension VC(H). They showed that an error rate of O(η • VC(H)) is achievable, and that this rate is optimal up to constant factors. Remarkably, this optimal rate can already be attained by deterministic learners.
In the agnostic setting, however, Hanneke et al. [2022] revealed a strikingly different phenomenon: deterministic learning under instance-targeted poisoning is impossible. Specifically, they showed that for any deterministic learner, an adversary can drive the learner's error close to 1, even when η is small (e.g., η = O(1/ √ n)). This naturally raises the following question:
Main Question Is agnostic learning under instance-targeted poisoning possible using randomized learners?
What is the optimal achievable excess error?
We show that randomness indeed circumvents the impossibility result established by Hanneke et al. [2022] for deterministic learners. Furthermore, we obtain nearly tight bounds on the optimal achievable excess error, matching up to logarithmic factors.
this section cite: ['b20', 'b20', 'b20']

Section: Theorem 1 (Main Result).
Let H be a concept class with VC dimension d, and let η ∈ (0, 1) be the poisoning budget. Then, the following hold:
• (Upper Bound) There exists a randomized learning rule such that for any distribution and against any adversary with poisoning budget η, and sample size of at least n ≥ 1 η , the excess error is at most Õ √ dη . 1
• (Lower Bound) For every learning rule, and n > 0, there exist a distribution D and an adversary with poisoning budget η such that the excess error is at least Ω min √ dη, 1 .
In particular, for every sufficiently large sample size n, the optimal achievable excess error under instance-targeted poisoning is Θ min{ dη, 1} .
1 Note that if n < 1 η , then the adversary's budget is too small to corrupt any examples. In this case, the problem reduces to standard PAC learning, for which the optimal error rate is Θ( d/n).
The upper bound in Theorem 1 and its quantitative version, Theorem 4,are achieved by a proper randomized learner, which selects its output hypothesis with probability proportional to the exponential of (minus) its empirical loss -in the spirit of multiplicative weights algorithms and the exponential mechanism from differential privacy. The connection between data poisoning and differential privacy [Dwork, McSherry, Nissim, and Smith, 2006] is intuitive: differential privacy requires that an algorithm's output be stable under adversarial changes to a single training example. This guarantee extends-with some quantitative degradation-to group privacy, which ensures stability under changes to multiple examples. This notion is closely related to the goal in our setting, where the adversary may corrupt an η-fraction of the training data.
The lower bound is established by first analyzing a simpler setting, the poisoned coin guessing problem, and then using a kind of direct-sum argument to extend the bound to arbitrary VC classes. Further details and intuition behind these constructions are provided in the technical overview below.
this section cite: ['b14']

Section: Private vs. Public Randomness
Following Gao et al. [2021a] and Hanneke et al. [2022], we distinguish between two natural types of adversaries, according to their access to the learner's randomness: private-randomness adversaries and public-randomness adversaries. In the private randomness setting, the adversary does not observe the learner's internal random bits when constructing its poisoning attack. In the public randomness setting, by contrast, the adversary has full knowledge of the learner's random seed before designing the poisoned dataset.
At first glance, one might expect public randomness to reduce to the deterministic case: indeed, once the random seed is fixed, the learner becomes a deterministic function of its (poisoned) input, seemingly vulnerable to the same attacks as any deterministic algorithm. Perhaps surprisingly, this intuition proves false: although the adversary sees the learner's random seed when crafting the poisoned dataset, the underlying distribution over examples is fixed in advance and independently of this randomness. This asymmetry allows, with appropriate design, the construction of learners whose guarantees under public randomness match those achievable under private randomness:
Theorem 2 (Private vs. Public Randomness 2 ). For every learning rule A priv , there exists a learning rule A pub such that for any distribution D, sample size n, and poisoning budget η ∈ (0, 1), max adversary sees internal randomness ExcessLoss(A pub , D, n, η) ≤ max adversary does not see internal randomness
ExcessLoss(A priv , D, n, η). Thus, the theorem implies that whatever guarantees can be achieved against adversaries that do not observe the learner's internal randomness can also be achieved against adversaries who do. Specifically, given any learning rule A priv that achieves a certain excess error against weak adversaries (private randomness), we can efficiently construct a learning rule A pub that achieves the same guarantees against strong adversaries (public randomness). In particular, applying the theorem to an optimal learner A ⋆ priv for the private randomness model yields a learner A ⋆ pub that matches its performance even when the adversary is fully aware of the learner's random bits.
The proof of Theorem 2 relies on carefully coupling the predictions of A priv across all possible input samples. Specifically, we construct A pub to satisfy a monotonicity property: whenever A priv is more likely to make an error on a test point x given training sample S 1 than given training sample S 2 , the learner A pub will err on x for S 1 whenever it errs on x for S 2 . This ensures that adversaries gain no additional advantage from observing the learner's internal randomness.
this section cite: ['b20']

Section: Poisoned Learning Curves
The lower bound in Theorem 1 holds in the standard distribution-free PAC setting: for every learner and every sample size n, there exists a distribution tailored to the sample size n and to the learner that forces a large excess error. This is typical for PAC-style lower bounds, where the hard distribution is allowed to depend on the training set size.
However, in many practical learning scenarios, we think of the underlying distribution as fixed and study the learner's behavior as the sample size n increases -the so-called learning curve. This naturally raises the question: is it possible that for any fixed distribution, as n grows, the excess error under instance-targeted poisoning eventually falls below √ dη? In other words, can better asymptotic behavior be achieved in the universal learning setting [Bousquet, Hanneke, Moran, van Handel, and Yehudayoff, 2021]? The following result shows that the answer is unfortunately negative.
Theorem 3 (Poisoned Learning Curves ). Let H be a concept class with VC dimension d, and let η ∈ (0, 1) be the poisoning budget. Then, for every learning rule A, there exists a distribution D and an adversary that forces an excess error of at least Ω min √ dη, 1 for infinitely many sample sizes n.
Theorem 3 follows by exploiting the proof of Theorem 1. In that proof, we construct a finite set of distributions with the property that, for every learner and every sample size n, there exists a distribution in the set witnessing the lower bound of Theorem 1 at sample size n. Since the set is finite, by a simple pigeonhole principle argument, one of these distributions must witness the lower bound for infinitely many values of n, thus establishing Theorem 3.
this section cite: ['b6']

Section: Organization
The remainder of the manuscript is organized as follows. In Section 2, we provide a technical overview of our approach, focusing on the key ideas behind the proofs. Section 3 discusses related work and places our contributions in context. In Section 4, we formalize the learning model and introduce the adversarial losses studied in this work. In Section 5 we formally state and prove our main results.
this section cite: []

Section: Technical Overview
In this section, we provide a high-level overview of the proof of our main result, Theorem 1. Theorems 2 and 3 are not included in this overview, as their proofs are shorter and structurally simpler than that of the main result; they are deferred to the supplementary material.
this section cite: []

Section: Lower Bound
A natural starting point for understanding the lower bound is the coin problem, introduced by Hanneke  et al. [2022]. 4 In this setting, the learner is shown a training set consisting of n coin tosses, and must predict the outcome of a particular test toss. The adversary is allowed to observe both the training set and the target toss, and can corrupt a small fraction of the training data before the learner sees it. The learner's goal is to guess the target toss correctly despite this targeted poisoning. Formally, the coin problem is equivalent to learning a hypothesis class H = h 0 , h 1 over a single point X = {x}, where h i (x) = i.
this section cite: []

Section: The Poisoned Coin Problem
1. The adversary selects a bias parameter p ∈ [0, 1].
this section cite: []

Section: A training sample S of n examples is drawn i.i.d. from a Bernoulli distribution of bias p.
A target label y is drawn independently from the same distribution.
3. After observing both S and y, the adversary modifies at most an η-fraction of S to obtain a poisoned sample S ′ . 4. The learner receives S ′ and outputs a prediction A(S ′ ) ∈ {0, 1}. 5. The learner succeeds if A(S ′ ) = y, and fails otherwise.
In the absence of poisoning, the optimal strategy for the learner is simple: if p > 1 2 , the learner should always predict 1, resulting in an error rate of 1 -p; if p < 1 2 , the learner should predict 0, with an error rate of p. Thus, while the learner does not observe the bias p directly, it can estimate it from the sample. In the absence of poisoning, this allows the learner to achieve an error arbitrarily close to min(p, 1 -p) as the sample size increases.
When poisoning is allowed, the learner instead observes a corrupted sample S ′ , and we measure its performance relative to the clean optimum. We define the excess error as
excess = P(A(S ′ ) ̸ = y) -min(p, 1 -p).
Our goal is to show that against any learner, there exists an adversary that forces an excess error of Ω( √ η). Toward this end, we consider the function F (p) = E[A(S)], where S is a clean (unpoisoned) training sample drawn from a Bernoulli distribution with bias p. That is, F (p) represents the expected prediction of the learner when trained on clean samples with bias p. It is then convenient to consider the oblivious setting, which can be summarized as follows
The Poisoned Coin Problem with Oblivious Adversary
1. The adversary selects a bias parameter p ∈ [0, 1]. 2. A target label y is drawn from a Bernoulli distribution of bias p. 3. After observing y, the adversary modifies p into p ′ which is close d(p, p ′ ) ≤ η. 4. The learner receives a training sample S ′ of n i.i.d. examples drawn from a Bernoulli distribution of bias p ′ , and outputs a prediction A(S ′ ) ∈ {0, 1}. 5. The learner succeeds if A(S ′ ) = y, and fails otherwise.
One can change a sample S drawn from a distribution with bias p to a sample S ′ which is indistinguishable from one drawn from a distribution with bias p ′ , Thus it is suffice to find lower bounds for the oblivious setup. The advantage of this model is that the excess error can be defined in terms of the function F , hence the lower bound can be derived purely by analyzing properties of such functions.
Fix any learner A. We can assume that its excess error in the absence of poisoning is at most √ η, since otherwise the lower bound is immediate. In particular, this implies that when p = 1 2 + √ η, the learner's expected prediction F (p) must be close to 1, and when p = 1 2 -√ η, F (p) must be close to 0. Thus, as p varies from 1 2 -√ η to 1 2 + √ η, the function F (p) must change by a constant amount (independent of η and n).
By an averaging argument, this implies that over an interval of length O(η), the function F (p) must change by at least Ω( √ η) on average. In particular, there exists a critical point
p ⋆ ∈ [ 1 2 - √ η, 1 2 + √ η] such that |F (p ⋆ -η) -F (p ⋆ + η)| = Ω( √ η).
This critical bias p ⋆ will correspond to a hard instance for the learner, enabling the adversary to enforce the desired lower bound on the excess error.
The preceding argument establishes the desired lower bound for classes of VC dimension 1. Recall that a class H has VC dimension 1 if and only if there exists a point x that is shattered by H-meaning that for every label y ∈ {0, 1}, there exists a hypothesis h ∈ H such that h(x) = y. Thus, learning H on distributions supported on x, reduces to distinguishing between two competing hypotheses on a single point, corresponding exactly to the coin problem described above.
The case of VC dimension d corresponds naturally to a generalization that we call the d-coin problem.
In the d-coin problem, there are d distinct coins x 1 , . . . , x d , each associated with its own unknown bias p i ∈ [0, 1]. The data generation process proceeds as follows:
The d-Coin Problem
Consider the following sampling process:
• The adversary selects a bias vector p 1 , . . . , p d ∈ [0, 1] d .
• A coin x = x i is selected uniformly at random from {x 1 , . . . , x d }.
• The label y is drawn according to a Bernoulli distribution with bias p i (that is, y = 1 with probability p i ).
The rest is as before. The training set of n i.i.d. examples (x i , y i ) is generated according to the sampling process, independently of the target example (x, y). After observing both the training set and the target example, the adversary may modify up to an η-fraction of the training examples to produce a poisoned sample. The learner receives the poisoned sample and attempts to predict the label y of the target coin x. The learner succeeds if its prediction matches y, and fails otherwise.
We further note that, similalrly to how it was in the one coin case, the lower bound is proven using d-coin problem with oblivious advrsary, defined by analogy.
This d-coin problem mirrors the setting of learning a class H with VC dimension d. Indeed, if {x 1 , . . . , x d } is a set shattered by H, then the label of each point x i can behave independently of the others. Assigning biases p i to the labels of each x i corresponds to constructing a distribution over labeled examples consistent with H. Thus, the problem of predicting the label of a randomly chosen target point (under potential data poisoning) mirrors the challenge faced by the learner in the d-coin setting.
The Naive Extension to d Coins. At first glance, it may seem that the lower bound for a single coin should extend directly to the d-coin problem. Since each coin x i has its own independent bias p i , one might expect that the learner's prediction for x i depends only on the outcomes of examples corresponding to x i in the training sample, and not on examples involving other coins.
Under this assumption, the adversary's strategy would be simple: conditioned on the target coin being x i , the adversary would focus its poisoning efforts solely on the training examples involving x i . Since the meta-distribution is uniform over the d coins, the expected number of appearances of x i in the training sample is about n/d. The adversary's global budget allows corrupting an η-fraction of the n examples, and thus roughly a ηd-fraction of the x i examples. Thus, for the target coin, the setting effectively reduces to the single-coin case with sample size about n/d and poisoning budget about ηd. Applying the single-coin lower bound then suggests that the excess error should be at least Ω √ dη , matching the desired bound.
Unfortunately, the above reasoning overlooks an important subtlety. In the d-coin problem, the adversary must commit to the biases p 1 , . . . , p d before the training sample and target example are drawn. Because of this, the learner's prediction for a target coin x i could, in principle, depend on the outcomes of other coins x j (j ̸ = i), whose biases are correlated with p i through the adversary's global choice of parameters.
In particular, if the biases across different coins are not carefully chosen, the learner may be able to infer information about the bias of x i by examining patterns across the entire sample, not just the examples involving x i . This possibility of information leakage -where the behavior of non-target coins reveals something about the target coin -breaks the reduction to the single-coin case. Thus, a more careful argument is needed to establish the desired lower bound.
The Fix: Randomizing the Biases. To overcome this difficulty, we modify the adversary's strategy: rather than fixing the biases p 1 , . . . , p d deterministically, the adversary draws each bias independently at random from a carefully designed distribution over [0,1]. This randomization breaks potential correlations between different coins, ensuring that the behavior of non-target coins carries no useful information about the target coin.
Constructing such a hard distribution requires strengthening the lower bound for the single-coin case (d = 1): instead of selecting a hard bias p tailored to a specific learner, we design a distribution over biases that is universally hard -meaning that for any learner, the expected excess error (over the choice of the bias) remains Ω( √ η). Sampling the biases p 1 , . . . , p d independently from this hard distribution ensures that, for any fixed learner, the expected excess error remains large, and prevents information leakage between coins. With this setup, the adversary effectively reduces the d-coin problem back to d independent copies of the single-coin case, yielding the desired Ω( √ dη) lower bound.
this section cite: []

Section: Upper Bound
Coin Problem. To build intuition for the upper bound, we begin with the coin problem. One natural way to exploit randomness in the learner's strategy is via sub-sampling. Specifically, the learner can randomly select a small sub-sample of size k ∼ 1 η from the training set and predict the label of the test coin by majority vote over this sub-sample.
This simple strategy has two key advantages. First, by standard concentration bounds, a small sub-sample already suffices to estimate the bias of the coin up to a small additive error. Second, and crucially, by anti-concentration, the adversary cannot easily flip the prediction: changing the majority outcome typically requires modifying roughly √ k entries. As long as the poisoned fraction η is small, the adversary is unlikely to control enough points in the sub-sample to alter the majority. In total, this approach yields an excess error of O( √ η) in the coin problem.
this section cite: []

Section: Finite Classes.
A natural next step is to generalize this idea to arbitrary finite hypothesis classes.
One might try the same strategy: draw a small random sub-sample of size k and train an optimal PAC learner on it. This technique was used by Gao et al. [2021a], who showed it suffices for learning under instance-targeted poisoning when the poisoned fraction η vanishes with n. However, this method fails to provide our desired bounds when η is not negligible.
What fails here is robustness. In the coin problem, majority vote has a useful anti-concentration property: to flip the output, the adversary must corrupt roughly √ k points. But for general hypothesis classes, it is unclear whether any natural learning rule exhibits similar resilience to perturbations. In particular, standard PAC learners might change their output significantly in response to a few targeted changes, especially when trained on a small sub-sample. This instability limits the effectiveness of naive sub-sampling in the general case.
this section cite: []

Section: Sampling via Loss Exponentiation. To move beyond naive sub-sampling, it is helpful to ask: what probability distribution over hypotheses does the sub-sampling strategy induce?
In the coin problem, majority voting over a random sub-sample can be interpreted as assigning higher selection probability to the constant label h ≡ 0 or h ≡ 1) that achieves lower empirical loss. More quantitatively, one can show that the induced selection probability is roughly proportional to the exponential of the negative squared loss on the sub-sample: P(h) ∝ exp -λ • LS (h) 2 , for some λ > 0. This reflects the anti-concentration property of the majority vote: flipping the output requires altering many points in the sub-sample (although we note that due to the abovementioned problems, we end up not using the anti-concentration inequalitis per se).
This motivates the use of learners that explicitly reweight hypotheses according to exponentiated losses. In our final algorithm for finite classes, we simplify the squared loss to standard loss and sample from the exponential of its negative: P(h) ∝ exp -λ • LS (h) . This distribution, known from the exponential mechanism in differential privacy and multiplicative weights in online learning, preserves both stability and performance. Small perturbations to the dataset (such as an η-fraction of poisonings) have limited impact on the output, while hypotheses with lower empirical error remain more likely to be selected. Working out the details of this approach yields an excess error bound of O( √ η log m) for finite hypothesis classes of size m.
From Finite to VC Classes. All that remains is to extend the result from finite classes to hypothesis classes of bounded VC dimension. In the case of a finite class H of size m, the exponential sampling strategy achieves excess error Õ( √ η log m). Our goal is to replace the dependence on log m with VC(H), the VC dimension of the class.
A direct application of the multiplicative weights sampling method to an infinite class fails to yield tight bounds, as the performance of the learner would then scale with the (possibly infinite) size of the class, rather than with its VC dimension.
To overcome this, we use a classical reduction based on ε-covers: we construct a finite cover H ′ ⊆ H such that the minimal loss over H ′ approximates that over H up to an additive ε. This allows us to reduce to the finite case without significantly increasing the error.
A standard uniform convergence argument shows that an ε-cover can be constructed from a random sample of size roughly VC(H)/ε 2 . Unfortunately, using such a sample leads to a suboptimal overall bound on the excess error. Instead, we apply a more refined analysis based on the VC dimension of the symmetric difference class h△h ′ : h, h ′ ∈ H, which shows that it suffices to construct the ε-cover from a much smaller subsample of size Õ(VC(H)/ε). This enables us to keep the size of the subsample below the poisoning threshold with high probability, ensuring that the cover is not corrupted.
A similar approach -involving subsampling and refined covering arguments -has been used in the context of differential privacy and algorithmic stability, for example in Bassily et al. [2019], Dagan and Feldman [2020].
Putting everything together, our final algorithm proceeds as follows:
1. Given a sample S, draw a random sub-sample T of size Õ( VC(H)/η).
2. Use T to construct a finite ε-cover H T ⊆ H, where ε = VC(H) • η.
this section cite: ['b4', 'b11']

Section: Run exponential sampling over H T : select h ∈ H T with probability proportional to exp(-λ • LS (h)).
This completes the proof sketch of the upper bound.
this section cite: []

Section: Related Work
Learning under poisoning attacks has been studied in several settings. Earlier research [Valiant, 1985, Kearns and Li, 1993, Sloan, 1995, Bshouty, Eiron, and Kushilevitz, 2002] focused on nontargeted poisoning, where the adversary does not know the test point. Computational aspects of efficient learning under poisoning have been studied under various distributional and algorithmic assumptions [Kalai, Klivans, Mansour, and Servedio, 2008, Klivans, Long, and Servedio, 2009, Awasthi, Balcan, and Long, 2014, Diakonikolas, Kane, and Stewart, 2018]. In particular, Awasthi, Balcan, and Long [2014] achieved nearly optimal learning guarantees (up to constant factors) for polynomial-time algorithms learning homogeneous linear separators under distributional assumptions in the malicious noise model. These results were later extended to the nasty noise model by Diakonikolas, Kane, and Stewart [2018], along with techniques that also apply to other geometric concept classes. In the unsupervised setting, the computational challenges of learning under poisoning attacks have been investigated by Diakonikolas, Kamath, Kane, Li, Moitra, and Stewart [2016], Lai, Rao, and Vempala [2016].
In contrast, our work focuses on instance-targeted poisoning, and investigates the fundamental tradeoffs in error and sample complexity, independent of computational constraints. A related but more demanding task-certifying the correctness of individual predictions under instance-targeted poisoning-was studied by Balcan, Blum, Hanneke, and Sharma [2022]. The instance-targeted model we study was formalized in Gao, Karbasi, and Mahmoody [2021b], who showed that when the poisoning budget η vanishes with the sample size, PAC learnability is preserved. Hanneke, Karbasi, Mahmoody, Mehalel, and Moran [2022] extended this line of work to general (non-vanishing) poisoning rates, and characterized the optimal error in the realizable setting. In particular, they showed that deterministic learners suffice in the realizable case, but fail in the agnostic case, where they can suffer near-maximal error even under minimal poisoning. They left open the question of whether randomized learners could succeed in the agnostic case. Our work resolves this open problem, establishing that randomized learners can indeed achieve meaningful guarantees in the agnostic setting, and characterizing the optimal excess error as Θ( √ dη), where d is the VC dimension.
Other types of targeted attacks have also been studied. In model-targeted attacks, the adversary aims to force the learner to mimic a particular model [Farhadkhani, Guerraoui, Hoang, andVillemaud, 2022, Suya, Mahloujifar, Suri, Evans, andTian, 2021]. Label-targeted attacks aim to flip the learner's prediction on a specific test example [Chakraborty, Alam, Dey, Chattopadhyay, and Mukhopadhyay, 2018]. Jagielski, Severi, Pousette Harger, and Oprea [2021] introduced subpopulation poisoning, where the adversary knows that the test point comes from a specific subset of the population.
More broadly, recent works have explored robustness to adversarial test-time manipulation. Balcan, Hanneke, Pukdee, and Sharma [2023] study reliable prediction under adversarial test-time attacks and distribution shifts. While their work focuses on modifying the test point rather than poisoning the training set, it shares our motivation of provable robustness in challenging environments. Likewise, Goel, Hanneke, Moran, and Shetty [2023] analyze a sequential learning model with clean-label adversaries, allowing abstention on uncertain inputs.
Empirical and algorithmic defenses against instance-targeted and clean-label poisoning have been widely studied. Rosenfeld, Winston, Ravikumar, and Kolter [2020] show that randomized smoothing [Cohen, Rosenfeld, and Kolter, 2019] can mitigate label-flipping attacks, and can extend to replacing attacks such as ours. Follow-up work has explored deterministic defenses [Levine and Feizi, 2020], as well as randomized sub-sampling and bagging [Chen, Li, Wu, Sheng, and Li, 2020, Weber, Xu, Karlas, Zhang, and Li, 2020, Jia, Cao, and Gong, 2020].
Other theoretical works have studied error amplification under targeted poisoning [Mahloujifar andMahmoody, 2017, Etesami, Mahloujifar, andMahmoody, 2020], often with a focus on specific test examples. The empirical study of poisoning attacks in Shafahi, Huang, Najibi, Suciu, Studer, Dumitras, and Goldstein [2018] further illustrates the practical relevance of instance-targeted threats.
this section cite: ['b24', 'b32', 'b7', 'b23', 'b25', 'b0', 'b12', 'b0', 'b12', 'b13', 'b26', 'b1', 'b20', 'b16', 'b19', 'b29', 'b10', 'b27', 'b15']

Section: Preliminaries
As it is usual in learning theory, we consider a concept class H over domain X with a label space Y = {0, 1}; that is, H is a set of functions (also called concepts) from X to Y. We define the set of labeled examples as Z = X × Y and a sample of size n as a sequence S ∈ Z n ; the space of samples is Z ⋆ = ∞ n=1 Z n . A learning rule is a map A : Z ⋆ → Y X that assigns to each sample S a function A(S), called a hypothesis. We also often consider randomized learners that output a distribution over hypotheses.
The sample loss of a hypothesis h : X → Y on a sample S ∈ Z n is
L S (h) := 1 n n i=1 |h(x i ) -y i |.
Similarly, we define the population loss of h with respect to a distribution D over Z as
L D (h) = P (x,y)∼D [h(x) ̸ = y] = E (x,y)∼D |h(x) -y|.
The expected loss of a learner A with a sample size n with respect to D is
L D (A, n) = E S∼D n L D A(S) = E S∼D n (x,y)∼D |A(S)(x) -y|.
The above formula is also applicable to randomized learners, in which case the expectation is also over the internal randomness of A.
Define the normalized Hamming distance between two samples S, S ′ ∈ Z n by
d H (S, S ′ ) = 1 n |{i ∈ [n] : S i ̸ = S ′ i }| .
For any sample S ∈ Z n and η ∈ (0, 1), define the η-ball centered at S by B η (S) = {S ′ ∈ Z n :
d H (S, S ′ ) ≤ η}.
Definition 1 (η-adversarial loss). Let η ∈ (0, 1) be the adversary's budget, let A be a (possibly randomized) learning rule, and let D be a distribution over examples. The η-adversarial loss of A with sample size n with respect to D is defined as
L D,η (A, n) = E S∼D n (x,y)∼D sup S ′ ∈Bη(S) E r |A r (S ′ )(x) -y| ,
where r is the internal randomness of A, and Er can be omitted in case the learner A is deterministic.
Note that in this definition the data is corrupted before the randomness of A, which corresponds to private randomness. Alternatively, we may consider the public randomness model in which we define
L pub D,η (A, n) = E S∼D n (x,y)∼D E r sup S ′ ∈Bη(S) |A r (S ′ )(x) -y| .
Finally, in the spirit of agnostic learning and the poisoned coin problem, the effectiveness of the learner in the presence of poisoning is measured with respect to excess error: Definition 2 (Excess error). For a concept class H, adversary budget η ∈ (0, 1), learning rule A, sample size n, and a distribution over examples D, we define the excess error of A with sample size n against an adversary with budget η on D as
excess H,D,η (A, n) = L D,η (A, n) -inf h∈H L D (h).
The definition of excess is similarly adapted to the case of public randomness, and, as usual, we drop some of the parameters H, D, η, A, or n in the notation for loss/excess if they are clear from the context.
Definition 3 (VC dimension). We say that a concept class H shatters a set X ⊂ X if for any function f : X → {0, 1} there exists h ∈ H such that f (x) = h(x) for all x ∈ X. The VC dimension of H, denoted VC(H), is the largest d for which there exist a set X ⊂ X of size d that is shattered by H. If sets of arbitrary size can be shattered we define VC(H) = ∞. Lemma 4 (Sauer-Shelah lemma -see Lemma 6.10 in Shalev-Shwartz and Ben-David [2014]). Let H be a concept class of VC dimension d = VC(H), and let X ⊂ X be a finite set of unlabeled examples of size n = |X|. Define an equivalence relation on
H by h ∼ X h ′ if h(x) = h ′ (x) for all x ∈ X,
and let H X be an arbitrary set of representatives with respect to this relation. Then we have
|H X | ≤ d i=0 n d . In particular, if n > d + 1 we have |H X | ≤ ( e • n d ) d .
Fnu Suya, Saeed Mahloujifar, Anshuman Suri, David Evans, and Yuan Tian. Model-targeted poisoning attacks with provable convergence. In International Conference on Machine Learning, pages 10000-10010. PMLR, 2021.
Leslie G Valiant. Learning disjunction of conjunctions. In IJCAI, pages 560-566, 1985.
Maurice Weber, Xiaojun Xu, Bojan Karlas, Ce Zhang, and Bo Li. Rab: Provable robustness against backdoor attacks. arXiv preprint arXiv:2003.08904, 2020.
NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: In the abstract we briefly introduce the main question, and in the paper we formally prove it.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: ['b31']

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [NA] Justification: The paper deals with a theoretical filed, in which the definitions are well established and the questions a well defined. It states those problems and prove them formally.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: Everything is formally defined and proved in the paper.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [NA] Justification: The paper does not include experiments Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper does not include experiments requiring code Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [NA]
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
this section cite: []

Section: Answer: [Yes]
Justification: There are no human subjects nor are there any data-related concerns. And the paper dose not seem to have any potential harmful consequences Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: While the paper deal with foundational research, the practical motivation is well stated and backed by examples. being a theoretical paper lesson its potential impact, yet the positive advantage it can yield are clear from the motivating examples.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: The paper poses no such risks since it contains no models or private data.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [NA] Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets ( if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: Proofs

this section cite: []

Section: Proof of Theorem 1
We prove Theorem 1 in a form of Theorem 4 below, which gives quantitative version of the bounds announced in it.
this section cite: []

Section: Theorem 4 (Main result -quantitative version).
Let H be a concept class with VC dimension d ≥ 1 and suppose η < 1 4d . Then there exits a randomized learner A such that for any distribution D on X × Y and all n ≥ 1 η we have:
excess H,D,η (A, n) ≤ 36 ηd log e ηd .
This bound is also tight up to log factors; that is, for every randomized learner A, η < 1 d , and n ≥ 6 η log 64 √ dη , there exists a distribution D such that
excess H,D,η (A, n) ≥ √ dη 36 .
There will be no separate proof of Theorem 4. Instead, the entire Section 5.1 is treated as such proof. In particular, the upper bound on the excess is established in Lemma 9, and the lower bound in Lemma 14. We also remark that Theorem 1 is a direct consequence of Theorem 4. Indeed the bounds in Theorem 4 are quantitative versions of the ones in Theorem 1 so we just need to justifies the assumptions of Theorem 4.
For the upper bound the assumption η < 1 4d is safe since otherwise a bound of Õ( √ dη) = O(1) is trivial, while the assumption η < 1 n was already assumed in Theorem 1, where we remarked that without it the adversary is unable to poison even a single point.
For the lower bound, we may assume η < 1 d , since otherwise √ dη = Ω(1) and the result is clear. The assumption n ≥ 6 η log 64 √ dη is valid since without it, standard PAC learning arguments shows that the desired lower bounds can be forced even without poisoning. Indeed, by 28.2.22 in Shalev-Shwartz and Ben-David [2014], an excess loss of Ω( d n ), can always be forced, even without poisoning. In the case of n ≤ 6 η log 64 √ dη , this becomes the desired Ω( √ dη).
this section cite: ['b31']

Section: Upper bound
We address loss of learners with poisoning via prediction stable learners: Definition 5 (Prediction stability). The η-prediction stability of a random learner A with sample size n with respect to a distribution D is
λ n (A|D, η) = E S∼D n (x,y)∼D sup S ′ ∈Bη(S) P r A r (S)(x) ̸ = A r (S ′ )(x) = E S∼D n (x,y)∼D sup S ′ ∈Bη(S) E r |A r (S ′ )(x) -A r (S)(x)| .
Note that, trivially,
max λ n (A|D, η), L D (A, n) ≤ L D,η (A, n) ≤ λ n (A|D, η) + L D (A, n),
where the second inequality can be trivially rewritten as
excess H,D,η (A, n) ≤ λ n (A|D, η) + excess H,D (A, n),(1)
where
excess H,D (A, n) = L D (A H , n) -inf h∈H L D (h).
So minimizing the loss of a learner with poisoning is equivalent to finding an efficient learner that is prediction stable.
Proposition 6. Let H ⊆ Y X be a finite concept class of size |H| = m and let η > 0 be the poisoning budget. Then there exists a randomized learner A H such that for every distribution D over Z and every n > 0, it holds
λ n (A H |D, η) ≤ 4 η log m, excess H,D (A H , n) ≤ η log m + 4 log 2m n .
Proof. We define the learner A H in two stages. First, we define the learner A, that samples according to the exponential mechanism with multiplicative weights. That is, it outputs each h ∈ H with probability h) and t = log m η . We then prove that A satisfies the bound on the loss from the statement of the proposition, that is, that
P A(S) = h = e -tL S (h) W , where W = W (S) = h∈H e -tL S (
L D (A, n) ≤ inf h∈H L D (h) + 5 √ η log 2m.
As our second stage, we define another learner B, which will be our target learner, as follows: B samples r ∼ U(0, 1). Then, for a sample S and
x ∈ X , B returns 1 if r ≤ Pr [A r (S)(x) = 1]
, and 0 otherwise. It is easy to see that B is a different learner than A; in particular, B can output hypotheses outside of the class H. However, by construction, for all S and x, it holds
P r [B r (S)(x) = 1] = P r [A r (S)(x) = 1] .
In particular, this easily implies that L D (B, n) = L D (A, n), and so B also satisfies the required bound on the loss. Notice, however, that, unlike with A, for B there is an explicit dependence of the output on the internal randomness r. Thus, the outputs of B on any samples S and S ′ are naturally coupled. We will then utilize this coupling to prove the required bound on the prediction stability of B.
Let us now start with the proof of the bound on L D (A, r). The proof proceeds by several claims.
Claim 1. For any sample S, it holds:
E r L S A r (S) ≤ inf h∈H L S (h) + log m t . Indeed, exp t • E r L S (A r (S)) ≤ [by Jensen's] ≤ E r exp t • L S (A r (S)) = 1 W h e tL S (h) e -tL S (h) = m W ≤ W = h∈H e -tL S (h) ≥ sup h e -tL S (h) ≤ m • sup h∈H e -tL S (h) -1 ≤ m exp t inf h∈H L S (h) .
Taking the log of both sides we get
E r L S A r (S) ≤ inf h∈H L S (h) + log m t , as needed. Claim 2. E S∼D n sup h∈H |L D (h) -L S (h)| ≤ 2 log 2m n .
Indeed, for any h ∈ H and ε > 0 by Hoeffding's inequality we have
P S∼D n |L S (h) -L D (h)| > ε ≤ 2e -ε 2 n .
And, by union bound, this gives
P S∼D n sup h∈H |L S (h) -L D (h)| > ε ≤ 2me -ε 2 n .
Let us define ε 0 = log 2m n , the value for which 2me -ε 2 0 n = 1, and compute
E S∼D n sup h∈H |L D (h) -L S (h)| = 1 0 P S∼D n sup h∈H |L S (h) -L D (h)| > t dt ≤ ε0 0 1 + 1 ε0 2me -t 2 n dt ≤ x := √ n • t ≤ ε 0 + 2m √ n ∞ ε0 √ n e -x 2 dx ≤ ε 0 √ n = log 2m ≥ 1, so we estimate e -x 2 ≤ xe -x 2 ≤ ε 0 + 2m √ n ∞ ε0 √ n xe -x 2 dx = ε 0 + 2m √ n • - 1 2 e -x 2 ∞ ε0 √ n ≤ ε 0 + 2m √ n • exp(-ε 2 0 n) 2 = log 2m n + m exp(-log 2m) √ n ≤ 2 log 2m n .
This finishes the proof of Claim 2.
Putting Claims 1 and 2 together and recalling that t = log m η , we get the required bound on
L D (A, n): L D (A, n) = E S∼D n L D A(S) ≤ E S∼D n L S A H (S) + 2 log 2m n ≤ inf h∈H L S (h) + log m t + 2 log 2m n ≤ inf h∈H L D (h) + η log m + 4 log 2m n .
Let us now prove the prediction stability bound for B.
Recall that B returns 1 if r ≤ Pr [A r (S)(x) = 1]
, and 0 otherwise, for r ∼ U(0, 1). Again, we start with an intermediate claim, which, somewhat unexpectedly, is still be about A, not B.
Claim 3. Let S, S ′ ∈ Z n be two samples with d H (S, S ′ ) ≤ η. Then, for any h ∈ H, we have
e -2tη • P r [A r (S ′ ) = h] ≤ P r [A r (S) = h] ≤ e 2tη • P r [A r (S ′ ) = h]. Indeed, by the definition of d H , L S (h) ≤ L S ′ (h) + η, so e -tη • e -tL S ′ (h) ≤ e -tL S (h) ≤ e tη • e -tL S ′ (h) ,
where the second inequality is by symmetry. So,
W = h∈H e -tL S (h) ≤ e tη • h∈H e -tL S ′ (h) = e tη • W ′ ,
and so
P r [A r (S ′ ) = h] = e -tL S ′ (h) W ′ ≤ e 2tη • e -tL S (h) W = e 2tη • P r [A r (S ′ ) = h].
And the second inequality is by symmetry. This finishes the proof of Claim 3. Now, let S and S ′ be η-close samples and let x ∈ X . Then
P r B r (S)(x) ̸ = B r (S ′ )(x) = P r B r (S)(x) = 1 and B r (S ′ )(x) = 0 + P r B r (S)(x) = 0 and B r (S ′ )(x) = 1 = P r A r (S)(x) = 1 -P r A r (S ′ )(x) = 1 = h : h(x)=1 P r A r (S)(x) = h - h : h(x)=1 P r A r (S ′ )(x) = h ≤ h P r A r (S)(x) = h -P r A r (S ′ )(x) = h ≤ [By Claim 3] ≤ h max P r A r (S)(x) = h , P r A r (S ′ )(x) = h • (1 -e -2tη ) ≤ 2(1 -e -2tη ) ≤ 4tη = 4 η log m,
where in the last inequality we used the estimate 1 -e -x ≤ x for x ∈ (0, 1). The last bound clearly extends to λ n (B|D, η) ≤ 4 √ η log m, as needed. Because, as argued in the beginning, B satisfies the same guarantee on L D , it thus satisfies both bounds in the statement of the proposition, finishing the proof.
Definition 7 (ε-cover). Let H be a concept class over domain X , let ε > 0 and D be a distribution over X . Then H ⊆ H is called an ε-cover of H with respect to D if for any h ∈ H there is h
′ ∈ H such that Px∼D h(x) ̸ = h ′ (x) ≤ ε. Let E D (H) be the minimal ε > 0 for which H is an ε-cover of H, that is E D (H) = sup h∈H inf h ′ ∈H P x∼D h(x) ̸ = h ′ (x) .
We also use the above definition with distributions D over examples X × Y, rather than just over X . This is done in a natural way, by letting
E D (H) = E D X (H), where D X is the marginal of D. For a set of unlabeled examples X ⊆ X recall we defined an equivalence relation on H by h ∼ X h ′ if h(x) = h ′ (x)
for all x ∈ X, and denoted H X an arbitrary set of representatives with respect to this relation. Extend this definition to labeled samples S = (x 1 , y 1 ), (x 2 , y 2 ), . . . , (x n , y n ) by letting
H S = H X for X = {x 1 , . . . , x n }.
The following lemma is based on Lemma 3.3 in Bassily et al. [2019]. Lemma 8 (Covers for VC classes). For any concept class H with VC dimension d, for any distribution D and any n ≥ d, we have:
E S∼D n E(H S ) ≤ 13d n log 2en d .
Proof. The proof relies on Lemma 3.3 in Bassily et al. [2019], specifically, we use claim (1) in this result which gives
P S∼D n [E(H S ) > ε] ≤ 2( 2en d ) 2d exp( -εn 4 ).
Hence for any ε > 0 we have
E S∼D n [E(H S )] ≤ ε + 2( 2en d ) 2d exp( -εn 4 ).
Taking ε = 12d n log 2en d we get
E S∼D n [E(H S )] ≤ ε + 2( 2en d ) 2d exp(-3d log 2en d ) = 12d n log 2en d + 2( 2en d ) -d ≤ 13d n log 2en d .
Where in the last inequality we used n ≥ d to deduce 2( 2en d ) -d ≤ 12d n log 2en d .
For any sample S = (x 1 , y 1 ), (x 2 , y 2 ), . . . (x n , y n ) and an index set J ⊆ {1, 2, . . . n}, define the subsample S J = (x j1 , y j1 ), (x j2 , y j2 ), . . . (x j k , y j k ) , where J = {j i } k i=1 are the elements of J ordered as j 1 < j 2 ≤< j k . Lemma 9 (Upper bound of Theorem 4). Let H be a concept class with VC dimension d ≥ 1 and suppose η < 1 4d . Then there exits a randomized learner A : Z ⋆ → [0, 1] X such that for any distribution D on X × Y and all n ≥ 1/η we have:
λ n (A|D, η) ≤ 4 ηd log e ηd , excess H,D (A, n) ≤ 32 ηd log e ηd ,and
excess H,D,η (A, n) ≤ 36 ηd log e ηd .
Proof. Let η ∈ (0, 1) and let H be a class of VC dimension d. The learner A is defined as follows: For a distribution D over Z, n ≥ 2, and an n-sample S ∼ D n , let S be a concatenation of samples S 1 and S 2 of sizes n 1 = ⌊n/2⌋ and n 2 = ⌈n/2⌉ respectively, and let k = ⌊ d/4η⌋; note that by an easy computation, the condition n ≥ 1/η implies k ≤ n 1 , n 2 , while the condition η ≤ 1 4d implies k ≥ 1. A samples a k-set J ⊆ [n 1 ], uniformly at random. Then on the subsample T = S 1,J it runs the learner A H T from Proposition 6 using the concept class H T ⊆ H and the subsample S 2 . That is, A(S) = A H T (S 1 ) (S 2 ).
It is easy to see that, by construction, T ∼ D k . Hence, by Proposition 6, we can estimate the (clean) loss of A as
L D (A, n) = E T ∼D k L D (A H T , n 2 ) ≤ E T ∼D k inf h∈H T L D (h) + η log |H T | + 4 log 2|H T | n 2 ≤ E T ∼D k inf h∈H T L D (h) + 9 η log 2|H T | .
Note that the last bound follows from the assumption n 2 = ⌈n/2⌉ > 1 2η . We now separately bound the two terms in the above. The first term is bounded using Lemma 8:
E T ∼D k inf h∈H T L D (h) ≤ [by the definition of ε-cover] ≤ inf h∈H L D (h) + E T ∼D k E D (H T ) ≤ inf h∈H L D (h) + 13d k log 2ek d ≤ inf h∈H L D (h) + 26 ηd log e √ ηd ≤ inf h∈H L D (h) + 13 ηd log e 2 ηd .
For the second term, we apply the Sauer-Shelah lemma 4 to conclude that |H T | ≤ ( e•k d ) d . Note that the above bounds requires k > d + 1, which is enforced by the condition η < 1 4d . Then
E T ∼D k 5 η log 2|H T | ≤ 5 ηd log 2ek d ≤ 5 ηd log e √ ηd ≤ 3 ηd log e 2 dη .
Finally, combining these two estimates, we get:
L D (A, n) ≤ inf h∈H L D (h)13 ηd log e 2 ηd + 3 ηd log e 2 dη ≤ inf h∈H L D (h) + 32 ηd log e ηd , yielding excess H,D (A, n) ≤ 32 ηd log e ηd .
as needed.
Now let us show the stability bound. Let S, S ′ ∈ Z n be samples such that S ′ ∈ B η (S), let x ∈ X and, as above, let J ⊆ [n 1 ], be a k-set, for k = ⌊ d/4η⌋, chosen uniformly at random. Let T = T (J) = S 1,J and T ′ = T ′ (J) = S ′ 1,J . Recall that the rendomness of A is composed of picking J and the internal randomness r of A H T . We have
P J,r A(S)(x) ̸ = A(S ′ )(x) = P J,r A H T (S 2 )(x) ̸ = A H T ′ (S ′ 2 )(x) ≤ P J T (J) ̸ = T ′ (J) + P J T (J) = T ′ (J) • P r A H T (S 2 )(x) ̸ = A H T ′ (S ′ 2 )(x) T = T ′ ≤ . . .
Let us estimate the first probability as follows: Let I = {i ∈ [n 1 ] : S 1,i ̸ = S ′ 1,i }. As S ′ ∈ B η (S), this implies |I| ≤ η • n, and so
P J T (J) ̸ = T ′ (J) = P J S 1,J ̸ = S ′ 1,J = P J |J ∩ I| ≥ 1 ≤ E J |J ∩ I| ≤ kη. Thus . . . ≤ kη + P J T (J) = T ′ (J) • P r A H T (S 2 )(x) ̸ = A H T ′ (S ′ 2 )(x) T = T ′ = kη + P J T (J) = T ′ (J) • P r A H T (S 2 )(x) ̸ = A H T (S ′ 2 )(x) T = T ′ ≤ kη + P J,r A H T (S 2 )(x) ̸ = A H T (S ′ 2 )(x) .
Notice that the only difference between the first and the second expression is that A H T ′ becomes A H T . Now, taking expectation over S ∼ D n and (x, y) ∼ D in the above, we get
λ n (A|D, η) ≤ kη + E S∼D n (x,y)∼D sup S ′ ∈Bη(S) P J,r A H T (S 2 )(x) ̸ = A H T (S ′ 2 )(x) ≤ kη + E S∼D n J (x,y)∼D sup S ′ ∈Bη(S) P r A H T (S 2 )(x) ̸ = A H T (S ′ 2 )(x) ≤ kη + E T ∼D k E S2∼D n 2 (x,y)∼D sup S ′ 2 ∈B2η(S2) P r A H T (S 2 )(x) ̸ = A H T (S ′ 2 )(x) ≤ kη + E T ∼D k λ n2 (A H T |D, 2η) ≤ [by Proposition 6] ≤ kη + E T ∼D k 4 η log |H T | ≤ by |H T | ≤ e • k d d ≤ kη + 4 ηd log ek d ≤ √ ηd 2 + 4 ηd log e √ 4ηd ≤ √ ηd 2 + 2 ηd log e 2 4ηd ≤ 4 ηd log e ηd .
Finally, we estimate excess H,D,η (A, n) by ( 1)
excess H,D,η (A, n) ≤ λ n (A|D, η) + excess H,D (A, n)
≤ 4 ηd log e ηd + 32 ηd log e ηd ≤ 36 ηd log e ηd .
this section cite: ['b4', 'b4']

Section: Lower bound
When it comes to lower bounds on the excess error rate, it is convenient to consider oblivious adversary, which only changes the distribution, not the sample itself, and can be considered a relaxation of the regular sample-changing adversary. This setup can be informally summarized as follows:
1. The adversary selects a distribution D.
this section cite: []

Section: 2.
A labeled example z = (x, y) is drawn from D and is shown to the adversary.
3. The adversary corrupts the distribution D by changing it into D ′ which is η-close to D.
4. The learner draws an n-sample S from the corrupted distribution D ′ and uses it to predict the label y.
The formal definition follows Blanc and Valiant [2024]. Recall that for distributions µ 1 over X 1 and µ 2 over X 2 , a coupling of µ 1 and µ 2 is a distribution µ 12 over X 1 × X 2 , whose marginals on X 1 and X 2 coincide with µ 1 and µ 2 respectively. For distributions D 1 and D 2 over Z, we define the distance betwen them as
d(D 1 , D 2 ) = inf D12 P z1,z2∼D12 [z 1 ̸ = z 2 ],
where the infimum is over all couplings D 12 of D 1 and D 2 . We note that thus defined, d(D 1 , D 2 ) coincides with the total variation distance between D 1 and D 2 , that is:
d(D 1 , D 2 ) = sup E |D 1 (E) -D 2 (E)|,
where the supremum is over all measurable events, see Definition 9 in Blanc and Valiant [2024].
Finally, for a distribution D we define B η (D) to be the set of all distributions D ′ such that d(D, D ′ ) ≤ η.
Definition 10 (η-oblivious adversarial loss). Let η ∈ (0, 1) be the adversary's budget, let A be a (possibly randomized) learning rule, and let D be a distribution over examples. The η-oblivious adversarial loss of A with sample size n and with respect to D is defined by
L obl D,η (A, n) = E (x,y)∼D sup D ′ ∈Bη(D) E S ′ ∼D ′ E r |A r (S ′ )(x) -y| .
The following proposition substantiates the claim that the oblivious adversary can be considered a relaxation of the regular one. Proposition 11. For any n > 0, η ∈ (0, 1), ε > 0, learner A, and distribution D be we have
L D,2η (A, n) + e -nη 3 ≥ L obl D,η (A, n).
Proof. Recall that we want to upper-bound
L obl D,η (A, n) = E (x,y)∼D sup D ′ ∈Bη(D) E S ′ ∼D ′ E r |A r (S ′ )(x) -y| with L D,η (A, n) = E (x,y)∼D E S∼D sup S ′ ∈Bη(S) E r |A r (S ′ )(x) -y| .
We will now bound the part after Ex,y of the first expression in terms of the similar part of the second. For brevity, we denote Er |A r (S ′ )(x) -y| by L(S ′ ), keeping in mind that L also depends on x, y, and A.
sup D ′ ∈Bη(D) E S ′ ∼D ′ L(S ′ ) = sup D ′ P S ′ ∼D ′ S∼D [S ′ ∈ B 2η (S)] • E S ′ ∼D ′ ,S∼D S ′ ∈B2η(S) L(S ′ ) + P S ′ ∼D ′ S∼D [S ′ / ∈ B 2η (S)] • E S ′ ∼D ′ ,S∼D S ′ / ∈B2η(S) L(S ′ ) ≤ . . .
Here we assume that D and D ′ are coupled with a coupling witnessing d(D, D ′ ) ≤ η. Now, in the expression [1] + [2] in brackets, let us estimate [1] and [2] separately, starting from [2]:
[2] = P S ′ ∼D ′ S∼D [S ′ / ∈ B 2η (S)] • E S ′ ∼D ′ ,S∼D S ′ / ∈B2η(S) L(S ′ ) ≤ P S ′ ∼D ′ S∼D [S ′ / ∈ B 2η (S)] ≤ P [Bin (n, η) ≥ 2ηn))] ≤ e -nη 3 .
Here in the second inequality we use the fact that D and D ′ are coupled in such a way that Pz∼D,z ′ ∼D ′ [z ̸ = z ′ ] ≤ η, and hence the event S ′ / ∈ B 2η (S) is equivalent to making at least 2ηn mistakes in n trials, where the probability of an individual mistake is at most η. The final bound is by multiplicative Chernoff bound inequality. Now, for [1]:
[1] = P S ′ ∼D ′ S∼D [S ′ ∈ B 2η (S)] • E S ′ ∼D ′ ,S∼D S ′ ∈B2η(S) L(S ′ ) ≤ P S ′ ∼D ′ S∼D [S ′ ∈ B 2η (S)] • E S ′ ∼D ′ ,S∼D S ′ ∈B2η(S) sup S ′′ ∈B2η(S) L(S ′′ ) ≤ E S ′ ∼D ′ ,S∼D sup S ′′ ∈B2η(S) L(S ′′ ) = E S∼D sup S ′ ∈B2η(S) L(S ′ ).
Note that here we fold back the expectation with conditionals, in the way opposite to how we did it in the beginning, but after changing the function under the expectation. Notably, the estimates for both [1] and [2] no longer depend on D ′ , so
. . . ≤ E S∼D sup S ′ ∈B2η(S) L(S ′ ) + e -nη 3
, yielding the desired bound.
Throughout the section, we are going to assume the following setup. Our label space will be {±1} instead of the usual {0, 1}. Let H ⊆ {±1} X be a concept class of VC dimension d, and let X = {x i } d i=1 be a fixed d-set shattered by X . We will only consider distributions that have uniform marginals over {x i } d i=1 , that is, distributions D such that for all i ≤ 1 ≤ d we have
P (x,y)∼D [x = x i ] = 1 d .
For u ∈ I d , let us define a distribution D u as
P (x,y)∼Du (x, y) = (x i , y) = 1 d (1/2 + yu i ) = 1 d ( 1 2 + u i ) y = 1, 1 d ( 1 2 -u i ) y = -1.
Note that the function u → D u gives a one to one correspondence between distributions with uniform marginals and [-1 2 , 1 2 ] d , which we are going to denote by
I d = [-1 2 , 1 2 ] d . Moreover, if we define the metric on I d as l 1 -norm rescaled by 1/d, that is, d(u, u ′ ) = 1 d ∥u -u ′ ∥ 1 = 1 d d i=1 |u i -u ′ i |, then d(u, u ′ ) = d(D u , D ′ u ). For a fixed n and a learner A : Z ⋆ → [0, 1] X , let us define a function F = F (A, n) : I d → I d as F i (u) = 1 2 E S∼D n A(S)(x i ).
Now, for an arbitrary function F : I d → I d , let us define the η-oblivious adversarial loss of F and η-excess of F with respect to u ∈ I d as
L obl u,η (F ) = 1 d i=1,...,d y∈{±1} sup u ′ ∈Bη(u) (1/2 + yu i ) 1/2 -yF i (u ′ ) ,and
excess obl u,η (F ) = L obl u,η (F ) - 1 d i=1,...,d min(1/2 -u i , 1/2 + u i ).
In particular, for F = F (A, n), we have:
L obl Du,η (A, n) = E (x,y)∼Du sup u ′ ∈Bη(u) E S ′ ∼D u ′ E r |A r (S ′ )(x) -y| 2 = 1 d i=1,...,d y∈{±1} (1/2 + yu i ) • sup u ′ ∈Bη(u) E S ′ ∼D u ′ E r |A r (S ′ )(x) -y| 2 = 1 d i=1,...,d y∈{±1} sup u ′ ∈Bη(u) (1/2 + yu i ) • 1/2 -y • 1/2 • E S ′ ∼D u ′ E r A r (S ′ )(x i ) = 1 d i=1,...,d y∈{±1} sup u ′ ∈Bη(u) (1/2 + yu i ) 1/2 -yF i (u ′ ) = L obl u,η (F (A, n)),
and, similarly,
excess obl Du,η (A, n) = excess obl u,η (F (A, n)).
(2) Note that the 1/2 factor in the first equality for the loss comes from changing the label space from {0, 1} to {±1}.
A poisoning scheme is a collection ξ = {ξ i,y } i∈[d],y∈{±1} of functions from I d to I d . The scheme ξ is said to have a poisoning budget η if d(ξ i,y (u), u) = 1/d • ∥ξ i,y (u) -u∥ 1 ≤ η for all i ∈ [d], y ∈ {±1}, and u ∈ I d . Poisoning schemes are used to explicate the choice of u ′ ∈ B η (u), and so we define L obl u,ξ (F ) and excess obl u,ξ (F ) by tweaking, in an obvious way, the respective definitions of L obl u,η (F ) and excess obl u,η (F ). Trivially, for all F : I d → I d and u ∈ I d , we have
excess obl u,η (F ) = sup ξ excess obl u,ξ (F ),
where the infimum is over all poisoning schemes with budget η. We thus want to prove that for every such F there is u ∈ I d and a poisoning scheme ξ with budget η such that excess obl u,ξ (F ) = Ω( dη). Lemma 12 (One-dimensional lower bound). For any η ∈ (0, 1), there exist a (one dimensional) poisoning scheme ξ = (ξ -1 , ξ 1 ) with poisoning budget η and a distribution U over I such that for any F : I → I we have
E u∼U excess obl u,ξ (F ) ≥ √ η 16 .
Proof. We are going to prove that such ξ and U exist for η ≤ 1/16, and that they guarantee E η = Eu∼U excess obl u,ξ (F ) ≥ √ η 4 . Note that in this case, for η ≥ 1/16 (but still η ≤ 1), we can thus enforce an excess of E 1/16 = 1/16/4 = 1/16, yielding the statement of the lemma for all η ∈ (0, 1).
So let η ≤ 1/16 and let m be a natural number such that √ η/2 ≤ (2m + 1)η ≤ √ η ≤ 1/2; it is easy to see that the condition on η guarantees that such m exists. Let U = {2iη : -m ≤ i ≤ m}.
Define the poisoning scheme (ξ -1 , ξ 1 ) as ξ -1 (2iη) = (2i + 1)η, ξ 1 (2iη) = (2i -1)η, for i = -m, . . . , m and ξ 0 (x) = ξ 1 (x) = x for x ∈ I, x ̸ = 2iη. Clearly, the poisoning budget of ξ = (ξ 0 , ξ 1 ) is η. Let us now estimate the excess error for this poisoning scheme first at points 2iη, for i = -m, . . . , m, and then, in a different way, at points -(2m + 1)η and (2m + 1)η. So, for 0 ≤ i ≤ m, and u = 2iη:
excess obl u,ξ (F ) = y∈{±1} (1/2 + yu) 1/2 -yF (ξ y (u)) -(1/2 -u) = (1/2 + u) 1/2 -F ξ 1 (u) + (1/2 -u) 1/2 + F ξ -1 (u) -(1/2 -u) = F (u + η) -F (u -η) 2 + u 1 -F (u -η) -F (u + η) .
By a similar computation, for -m ≤ i ≤ 0 and u = 2iη:
excess obl u,ξ (F ) = F (u + η) -F (u -η) 2 -u 1 + F (u -η) + F (u + η) .
In either case, for u = 2iη and i = -m, . . . , m, we have:
excess obl u,ξ (F ) ≥ F (u + η) -F (u -η) 2 .(3)
At the same time, for u = (2m + 1)η, ξ -1 (u) = ξ 1 (u) = u, and so
excess obl u,ξ (F ) = y∈{±1} (1/2 + yu) 1/2 -yF (u) -(1/2 -u) = (1/2 + u) 1/2 -F (u) + (1/2 -u) 1/2 + F (u) -(1/2 -u) = 1/2 + F (u) = 1 -1/2 -F (u) = (1/2 + u) 1/2 -F (u) -(1/2 -u) 1/2 -F (u) = u 1 -2F (u) .
Similarly, for u = -(2m + 1)η,
excess obl u,ξ (F ) = -u 1 + 2F (u) .
All in all, for u = (2m + 1)η, we get
excess obl u,ξ (F ) = |u| 1 -2sign(u)F (u) .(4)
Let us now define the tistribution U as follows: let U 1 be a uniform distribution over U = {2iη : -m ≤ i ≤ m}, let U 2 be a uniform distribution over {±(2m + 1)η}, and let U be U 1 or U 2 with probability 1/2 each. Trivially,
E u∼U excess obl u,ξ (F ) = 1 2 E u∼U1 excess obl u,ξ (F ) + 1 2 E u∼U2 excess obl u,ξ (F ).
Let us estimate the two terms in the above using (3) and (4). Below, we denote (2m + 1)η by t:
E u∼U1 excess obl u,ξ (F ) ≥ [by (3)] ≥ E u∼U F (u + η) -F (u -η) 2 = 1 2m + 1 m k=-m F (2kη + η) -F (2kη -η) 2 = F (t) -F (-t) 4m + 2 = η F (t) -F (-t) 2t
.
and
E u∼U2 excess obl u,ξ (F ) ≥ [by (4)] ≥ 1 2 t(1 -2F (t)) + 1 2 t(1 + 2F (-t)) = t -t(F (t) -F (-t)).
Combining, we get
E u∼U excess obl u,ξ (F ) = 1 2 t -t F (t) -F (-t) + η F (t) -F (-t) 2t = t 2 + F (t) -F (-t) η -2t 2 4t ≥ t 2 - 2t 2 -η 4t ≥ √ η ≥ t ≥ √ η 2 ≥ t 2 - η 4t ≥ η 4t ≥ √ η 4 .
Lemma 13 (Lower bound for F : I d → I d ). Let η be a poisoning budget with η < 1/d. There exists a distribution U over I and a poisoning scheme ξ with budget η such that for any function F : I d → I d we have
E u∼U d excess obl u,ξ (F ) ≥ √ dη 16 .
Proof. By Lemma 12, there exists a distribution U over I and a poisoning scheme ξ = ( ξ0 , ξ1 ) with poisoning budget dη such that for any function F : I → I we have
E u∼U excess obl u, ξ (F ) ≥ √ dη 16 .
Note that here we need dη < 1, which is enforced by our condition on η. Let us define a d-dimensional poisoning strategy ξ = {(ξ i,0 , ξ i,1 )} d i=1 as
ξ i,y (u) = u 1 , . . . u i-1 , ξy (u i ), u i+1 , . . . , u d .
Note that since the budget of ξ is dη, the poisoning budget of ξ is η. Now, for a given G : I d → I d , let us define F = F (G) : I → I as
F (t) = E u∼U d 1 d d i=1 G i (u 1 , . . . , u i-1 , t, u i+1 , . . . , u d ) = 1 d d i=1 F i (t),
where
F i (t) = E u∼U d G i (u 1 , . . . , u i-1 , t, u i+1 , . . . , u d ).
In words, F (t) is an expected value of G i (. . . , t, . . . ), where i is one of the d dimensions picked at random, t is plugged into the i'th coordinate, and the rest of coordinates a picked at random according to U. Moreover, by design, the expected excess error of F with ξ is the same as of G with ξ, but let us nevertheless explicate it. In what follows, let min ui := min(1/2 -u i , 1/2 + u i ).
E u∼U d excess obl u,ξ (G) = E u∼U d 1 d i=1,...,d   y∈{±1} (1/2 + yu i ) 1/2 -yG i (ξ i,y (u)) -min ui   = drag E uj ∼U
, for j ̸ = i, inside, all the way down to
G i = 1 d i=1,...,d E ui∼U    y∈{±1} (1/2 + yu i )   1/2 -y • E uj ∼U d-1 j∈[d]-i G i (ξ i,y (u))    -min ui   
= rename u i to t and rewrite using E
u∼U d G i (ξ i,y (u)) = F i ( ξy (u)) = 1 d i=1,...,d E t∼U   y∈{±1} (1/2 + yt) 1/2 -y • F i ( ξy (t)) -min t  
= now drag the average over i all the way down to F i , and use
F (t) = 1 d i F i (t) = E t∼U   y∈{±1} (1/2 + yt) 1/2 -y • F ( ξy (t)) -min t   = E t∼U excess obl t, ξ (F ).
In particular, this implies
E u∼U d excess obl u,ξ (G) ≥ √ dη16
, as needed. Lemma 14 (Lower bound of Theorem 4). For any concept class H of VC dimension d ≥ 1, any randomized learner A : (X × Y) ⋆ → [0, 1] X , poisoning budget η < 1/d and sample size n ≥ 6 η log 64 √ dη , there exists a distribution D such that excess H,D,η (A, n) ≥ inf h∈H L Du (h) + √ dη 16 √ 2 -e -ηn 6
.
In particular, for n > 6 η log 64 √ dη we have
excess H,D,η (A, n) ≥ inf h∈H L Du (h) + √ dη 36 .
Proof. Recall that Lemma 13, applied to F = F (A, n) : I d → I d , for an arbitrary sample size n, implies that there is a distribution D = D u , realizable by H and corresponding to u ∈ I d , and a poisoning scheme ξ with budget η 2 such that
excess obl D,η/2 (A, n) = excess obl u,η/2 (F (A, n)) ≥ excess obl u,ξ (F (A, n)) ≥ √ dη 16 √ 2 .
Here, the first equality is by (2), the second inequality is by the definition of poisoning scheme with budget, and the last one is by Lemma 13, where we change a probabilistic statement into an existential one. Note that here we use the condition η < 1 d for compliance with the lemma. Finally, exchanging the oblivious poisoning with a regular one by Proposition 11, we get:
excess D,η (A, n) ≥ excess obl D,η/2 (A, n) -e -nη/2 3 ≥ √ dη 16 √ 2 -e -nη 6
.
Which gives the first assertion of the Lemma. For the second a simple computation shows that for n ≥ 6 η log 64
√ dη we have √ dη 16 √ 2 -e -ηn 6 ≥ √ dη 16 √ 2 -exp(log -64 √ dη ) = dη( 1 16 √ 2 -1 64 ) ≥ √ dη 36 .
this section cite: ['b5', 'b5']

Section: Proofs of Theorems 2 and 3
Theorem 2. For every learning rule A priv , there exists a learning rule A pub such that for any distribution D, sample size n, and poisoning budget η ∈ (0, 1),
L pub D,η (A pub , n) ≤ L D,η (A priv , n).
Moreover, A pub can be constructed efficiently given black-box access to A priv .
Note that originally Theorem 2 was stated in terms of excess error; its equivalence with the present statement in terms of loss is straightforward.
Proof. In the proof, without losing generality, we identify the probability space of A priv and A pub with [0, 1]. Let us define a function A : Z ⋆ → [0, 1] X , induced by A priv , as
A(S)(x) = E r A priv,r (x).
And let A pub,r (S)(x) = 1 if r ≤ A(S) and 0 otherwise. Let η ∈ (0, 1), and let D be some distribution. For each x ∈ X let us define functions ξ x,0 , ξ x,1 : Z ⋆ → Z ⋆ by ξ x,0 (S) = arg max
S ′ ∈Bη(S) A(S ′ )(x), ξ x,1 (S) = arg min S ′ ∈Bη(S) A(S ′ )(x).
Note that, trivially, for y ∈ {0, 1}, ξ x,y (S) ∈ B η (S) and ξ x,y (S) = arg max
S ′ ∈Bη(S) E r |A priv,r (S ′ )(x) -y|.
We claim that A pub,r ξ x,y (S)(x) = y implies A r (S ′ )(x) = y for all S ′ ∈ B η (S). Indeed
A pub,r ξ x,1 (S)(x) = 1 =⇒ [∀S ′ ∈ B η (S) , r ≤ A(S ′ )] =⇒ [∀S ′ ∈ B η (S) , A pub,r (S)(x) = 1], A pub,r ξ x,0 (S)(x) = 0 =⇒ [∀S ′ ∈ B η (S) , r > A(S ′ )] =⇒ [∀S ′ ∈ B η (S) , A pub,r (S)(x) = 0].
Hence we have
L D,η (A priv , n) = E S∼D n (x,y)∼D sup S ′ ∈Bη(S) E r |A priv,r (S ′ )(x) -y| = E S∼D n (x,y)∼D E r |A priv,r ξ x,y (S) (x) -y| = E S∼D n (x,y)∼D |A ξ x,y (S) (x) -y| = E S∼D n (x,y)∼D E r sup S ′ ∈Bη(S) |A pub,r (S ′ )(x) -y| = L pub D,η (A pub , n).
Theorem 3 (Poisoned Learning Curves). Let H be a concept class with VC dimension d, and let η ∈ (0, 1) be the poisoning budget. Then, for every learning rule A, there exists a distribution D and an adversary that forces an excess error of at least Ω min √ dη, 1 for infinitely many sample sizes n.
Proof. The proof is by careful examination of the proof of Theorem 4, more precisely, of Lemmas 13 and 14. For the rest of the proof assume that η < 1 d , otherwise the result is obvious. First, we note that in Lemma 13, the parameter u ∈ I d of the target distribution D u , demonstrating the high excess of F n = F (A, n), is picked from a distribution U on I d , which has a finite support and whose construction is independent on n. Thus, by pigeonhole principle, there is u ∈ I d such that for D = D u and for infinitely many n
excess obl D,ν (A, n) = excess obl u,ν (F (A, n)) ≥ √ dη 16 .
Which by Proposition 11 implies
excess D,η (A, n) ≥ excess obl D,η/2 (A, n) -e -nη/2 3 ≥ √ dη 16 √ 2 -e -nη 6 .
And since
√ dη 16 -e -nη 6 ≤ √ dη
36 for only finitely many n's, we deduce that for infinitely many n
excess D,η (A, n) ≥ √ dη 36 .
this section cite: []

Section: References
Ref_id:b0 Title: The power of localization for efficiently learning linear separators with noise Year: (2014)
Ref_id:b1 Title: Robustly-reliable learners under poisoning attacks Year: (2022-07-05)
Ref_id:b2 Title: Reliable learning in challenging environments Year: (2023)
Ref_id:b3 Title: Can machine learning be secure? Year: (2006)
Ref_id:b4 Title: Limits of private learning with access to public data Year: (2019-12-08)
Ref_id:b5 Title: Adaptive and oblivious statistical adversaries are equivalent Year: (2024)
Ref_id:b6 Title: A theory of universal learning Year: (2021)
Ref_id:b7 Title: Pac learning with nasty noise Year: (2002)
Ref_id:b8 Title: Anupam Chattopadhyay, and Debdeep Mukhopadhyay. Adversarial attacks and defences: A survey Year: (2018)
Ref_id:b9 Title: A framework of randomized selection based certified defenses against data poisoning attacks Year: (2020)
Ref_id:b10 Title: Certified adversarial robustness via randomized smoothing Year: (2019)
Ref_id:b11 Title: Pac learning with stable and private predictions Year: (2020-07)
Ref_id:b12 Title: Learning geometric concepts with nasty noise Year: (2018)
Ref_id:b13 Title: Robust estimators in high dimensions without the computational intractability Year: (2016)
Ref_id:b14 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b15 Title: Computational concentration of measure: Optimal bounds, reductions, and more Year: (2020)
Ref_id:b16 Title: An equivalence between data poisoning and byzantine gradient attacks Year: (2022)
Ref_id:b17 Title: Learning and certification under instance-targeted poisoning Year: (2021)
Ref_id:b18 Title: Learning and certification under instance-targeted poisoning Year: (2021)
Ref_id:b19 Title: Adversarial resilience in sequential prediction via abstention Year: (1967)
Ref_id:b20 Title: On optimal learning under targeted data poisoning Year: (2022)
Ref_id:b21 Title: Niklas Pousette Harger, and Alina Oprea. Subpopulation data poisoning attacks Year: (2021)
Ref_id:b22 Title: Intrinsic certified robustness of bagging against data poisoning attacks Year: (2020)
Ref_id:b23 Title: Agnostically learning halfspaces Year: (2008)
Ref_id:b24 Title: Learning in the presence of malicious errors Year: (1993)
Ref_id:b25 Title: Learning halfspaces with malicious noise Year: (2009)
Ref_id:b26 Title: Agnostic estimation of mean and covariance Year: (2016)
Ref_id:b27 Title: Deep partition aggregation: Provable defenses against general poisoning attacks Year: (2020)
Ref_id:b28 Title: Blockwise p-tampering attacks on cryptographic primitives, extractors, and learners Year: (2017)
Ref_id:b29 Title: Certified robustness to labelflipping attacks via randomized smoothing Year: (2020)
Ref_id:b30 Title: Poison frogs! targeted clean-label poisoning attacks on neural networks Year: (2018)
Ref_id:b31 Title: Understanding machine learning: From theory to algorithms Year: (2014)
Ref_id:b32 Title: Four Types of Noise in Data for PAC Learning Year: (1995)
