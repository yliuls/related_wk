Title: Which Algorithms Have Tight Generalization Bounds?
Abstract: We study which machine learning algorithms have tight generalization bounds with respect to a given collection of population distributions. Our results build on and extend the recent work of Gastpar et al. (2024). First, we present conditions that preclude the existence of tight generalization bounds. Specifically, we show that algorithms that have certain inductive biases that cause them to be unstable do not admit tight generalization bounds. Next, we show that algorithms that are sufficiently loss-stable do have tight generalization bounds. We conclude with a simple characterization that relates the existence of tight generalization bounds to the conditional variance of the algorithm's loss.

Section: Introduction
Generalization bounds are at the heart of learning theory, and they play a central role in attempts to mathematically explain the behavior of contemporary supervised machine learning systems. A generalization bound is an upper bound of the form
L D (A(S)) ≤ b,(1)
where A(S) is the hypothesis output by learning algorithm A when executed with training set S, and L D (•) represents the loss with respect to the population distribution D. The term b is typically an expression of the form b = L S (A(S)) + c(S, A(S), H),
where L S (•) is the empirical loss, H is a hypothesis class, and c(S, A(S), H) is a 'complexity' term, such as the VC dimension or a spectral norm, etc.
We say that a generalization bound is valid if for every population distribution D, Eq. ( 1) holds with high probability; we say that a valid bound is uniformly tight (Definition 2.4) if for every population distribution, with high probability the difference between the two sides of Eq. ( 1) is small.
Bounding the loss using a generalization bound is quite different from using a validation set. Technically, a generalization bound does not use additional samples beyond the training set S. And while a validation set provides a single post-hoc measurement of the population loss after training is complete, a good generalization bound can provide insight into why a learning algorithm performs well, and can offer guidance for model selection and the development of new learning algorithms. For a generalization bound to be useful in this way, it is important that the bound be tight, so that it can distinguish cases with small population loss from cases with larger loss.
Unfortunately, experimental works have shown that many of the generalization bounds of the form of Eq. ( 2) that have been proposed in the literature are vacuousfoot_0 when applied to contemporary learning algorithms such as deep neural networks (Jiang et al., 2020;Dziugaite et al., 2020;Viallard et al., 2024, Section 4.4). Gastpar, Nachum, Shafer, and Weinberger (2024) offered a partial theoretical explanation for this empirical finding. They considered generalization bound as in Eq. ( 2), namely, bounds that depend only on the training set, the selected hypothesis, and the hypothesis class. They proved that any such bound cannot be uniformly tight in a certain regime (that is typical in practice) where the number of samples is insufficient for uniform convergence. 2Therefore, they recommended focusing on generalization bounds involving expressions of the form c(S, A(S), H, A, D), i.e., bounds that depend on, or are tailored for, a specific learning algorithm A and a specific collection D of population distributions.
This raises the following natural question about generalization bounds that are tailored for a specific pair (A, D):
Question 1. For which pairs of algorithms and distribution collections do there exist tight generalization bounds?
A variant of this question was addressed in Theorems 3, 4 and 5 of Gastpar et al. (2024), but the general case remains open. In this paper we continue investigating this question, and present conditions that are necessary, sufficient, or necessary and sufficient for the existence of tight generalization bounds for a given learning algorithm and distribution collection.
this section cite: ['b23', 'b10', 'b14', 'b14']

Section: Setting
Following Gastpar et al. (2024), we study the existence of tight generalization bounds using a notion of estimability.
Definition 1.1 (Estimability). Let X and Y be sets, let m ∈ N, let A : (X × Y) m → Y X be a learning rule, and let D ⊆ ∆(X × Y) be a collection of distributions. An estimator is a function E : (X × Y) m → R.
Let ε, δ ∈ [0, 1]. We say that A is uniformly estimable (or worst-case estimable) with respect to distributions D with precision ε and confidence δ using m samples if there exists an estimator E such that ∀D ∈ D :
P S∼D m E (S) -L D (A(S)) ≤ ε ≥ 1 -δ.
We say that A is estimable on average with respect to distributions D with precision ε and confidence δ using m samples if there exists an estimator E such that
P D∼U(D),S∼D m E (S) -L D (A(S)) ≤ ε ≥ 1 -δ.
(More briefly, we say that (A, D) is (ε, δ, m)-uniformly estimable, or (ε, δ, m)-estimable on average.)
The connection between estimability and tight generalization bounds is as follows.
Proposition 1.2. Using the notation of Definition 1.1, if (A, D) is (ε, δ, m)-estimable on average, then there exists a generalization bound b(S) (that may depend on A and D) that is 2ε-tight on average, namely
P D∼U(D),S∼D m [b(S) -2ε ≤ L D (A(S)) ≤ b(S)] ≥ 1 -δ. (3
)
Indeed, the generalization bound is simply b(S) = E (S)+ε, where E is the estimator witnessing the estimability of (A, D).
In the other direction, if (A, D) is not (ε, δ, m)-estimable on average, then there exists no generalization bound that satisfies Eq. (3), and in particular no generalization bound can be uniformly tight with precision 2ε (as in Definition 2.4).
Using these definitions, the Question 1 can be rephrased, as follows:
Question 2. Which general and useful conditions are necessary, sufficient, or necessary and sufficient for a tuple (A, D) to be (ε, δ, m)-uniformly estimable, or (ε, δ, m)-estimable on average?
We are specifically interested in addressing Question 2 in settings where the number of samples is not sufficient to guarantee learning in general (in the sense of the VC theorem and uniform convergence for example), because most contemporary machine learning algorithms (such as deep neural networks) are used in such settings.
this section cite: ['b14']

Section: A Simple But Crucial Distinction
To understand our work, it is necessary to keep in mind the following simple distinction:
Learnability and estimability are not the same.
Learning means that an algorithm achieves low population loss; estimability means that an algorithm has a tight generalization bound.
Some algorithms do not learn well, but they are very estimable (e.g., constant algorithms, as in Example 1.5 below). And if the number of samples is too small to allow learning, some interesting algorithms might nonetheless perform well on a certain subset of distributions and also be very estimable.
In the other direction, some algorithms learn well in practice (e.g., achieve low population loss on a distribution of interest), but are nonetheless not estimable for a larger collection D. 3 And if the number of samples is sufficient for learning perfectly, there might nonetheless be some algorithms that are not perfectly estimable (Example 1.4).
Lastly, in some cases there is a learnability-estimability trade-off (see Example 1.8). And while empirical risk minimization is in some sense the optimal learning algorithm, the empirical loss can be far from the best estimator (see Example 1.6).
this section cite: []

Section: Our Results
We investigate which algorithms and collections of distributions are estimable. As we show in Example 1.8 below, estimability is a delicate phenomenon. In particular, changing the sample size by just a small constant number can in some cases drastically change the set of (ε, δ) estimability parameters that are achievable. This means that identifying a simple and tight characterization that precisely determines the number of samples necessary and sufficient for estimability can be a difficult undertaking.
In this paper, we present conditions that preclude estimability, conditions that guarantee estimability, and a condition that is both necessary and sufficient for estimability.
Our first result is a condition that precludes estimability for algorithms that have an inductive bias towards certain subsets of VC classes, showing a connection between estimability and a central notion from traditional learning theory.
Theorem (Informal version of Theorem 3.1). Let H ⊆ {±1} X be a hypothesis class with VC dimension d large enough, and let m ≤ √ d/10. Then there exists a subset F ⊆ H and corresponding realizable distributions D such that any learning rule that has an inductive bias towards F is not (1/4 -o(1), 1/6, m)-estimable on average over D.
Note that the theorem precludes estimability on average, and so in particular it precludes worst-case estimability. The proof of Theorem 3.1 uses the Johnson-Lindenstrauss lemma (Theorem K.1), the probabilistic method, and a technical lemma (Lemma H.1) concerning the estimability of nearly-orthogonal functions.
To the best of our knowledge, this paper is the first to provide a rigorous and general mathematical formulation showing that any finite VC class admits inestimable algorithms. This is somewhat surprising because it means, for instance, that for any neural network architecture, there are some training algorithms for which one will not be able to derive tight generalization bounds (even if the distribution is realizable!). We believe this is a meaningful contribution.
Our next inestimability result is as follows.
Theorem (Informal version of Theorem 3.2). Let H ⊆ {±1} X be a collection of roughly 2 m nearly-orthogonal functions and corresponding realizable distributions D. Then any learning rule that has an inductive bias towards H is not (1/4 -o(1), ∼ 1/6, m)-estimable on average over D.
Theorem 3.2 is partially stronger than Theorem 3.1 in the sense that it shows inestimability for every algorithm that has an inductive bias towards a class of nearly-orthogonal functions, whereas Theorem 3.1 only shows the existence of a subclass with this property. 4 On the other hand, Theorem 3.1 is stronger than Theorem 3.2 in the sense that if Theorem 3.2 is applied to show inestimability for subclasses of a VC class, then it yields inestimability only for m ≤ O 3 √ d , whereas Theorem 3.1 obtains inestimability for all m ≤ O √ d . 5
To show Theorem 3.2, we prove a concentration inequality using the duality of linear programs (Lemma I.1), and then invoke the technical lemma (Lemma H.1).
Remark 1.3. Theorems 3.1 and 3.2 are stated for the case of binary labels, but they immediately imply inestimability also for regression and multi-class classification.
One way to interpret Theorems 3.1 and 3.2 is to consider a scenario where one derives a new generalization bound for a given algorithm, without making explicit distributional assumptions (as is the case for many published generalization bounds), and having a sample size within the regime of our theorems. Such bounds are generally formulated as high probability upper bounds on the population loss. Note that the lack of distributional assumptions means that the bound has to hold (be a valid upper bound) for all distributions, including the families of distributions that appear in our theorems.
But this means, in the light of our theorems, that the considered bound is necessarily very weak for many distributions unless one satisfies at least one of the following items:
1. Exclude in advance all families of distributions with nearly-orthogonal labeling functions, and use this fact in the derivation of the generalization bound.
2. Mathematically show that the algorithm is not biased towards any set of nearlyorthogonal functions. 6   The intuition behind Theorem 3.2 is that having an inductive bias towards a collection H of nearly-orthogonal functions makes the algorithm very unstable -small changes in the training set will cause the algorithm to shift between hypotheses in H, which are all 4 Additionally the quantity hidden by the o(1) notation is smaller in Theorem 3.2 by a quadratic factor (order 1/m vs. 1/ √ m). 5 The limitation m ≤ O 3 √ d when using Theorem 3.2 follows from the tightness of the Johnson-Lindenstrauss (JL) lemma. By the JL lemma, taking a collection F of 2 m orthogonal functions on a high dimensional domain, we can project F using a random projection and obtain a collection F ′ of 2 m functions that are ε-orthogonal defined on a domain of dimension log (2 m ) /ε 2 . In particular, let H be a class with VC dimension d. We want to project F onto an H-shattered set of size d with ε = Θ(1/m). This yields d = m/(Θ(1/m)) 2 = Θ m 3 . The tightness of JL implies that this construction cannot be improved.
6 It is known that there exist at least some neural network architectures which, when trained with SGD, are capable of learning orthogonal functions (such as parities). See Theorem 1 in Abbe and Sandon (2020).
very different from one another. This motivates our next result, which shows that stable algorithms are estimable, as follows.
this section cite: ['b0']

Section: Theorem (Informal version of Theorem 4.3).
Let A be an algorithm that is sufficiently stable with respect to a collection of distributions D (in a sense of loss stability or hypothesis stability similar to Rogers andWagner, 1978, or Kearns andRon, 1999). Then (A, D) is estimable.
Seeing as there are many definitions of stability in the literature, Theorem 4.3 makes a nontrivial conceptual contribution by identifying the "correct" notion of stability for understanding estimability. Other notions of stability, such as leave-one-out stability (Bousquet & Elisseeff, 2002), do not capture estimability as well, as we discuss in Section 4.
An additional motivation for Theorem 4.3 is the intuition that contemporary machine learning algorithms (like deep neural networks trained with gradient descent) might indeed be sufficiently stable. If so, Theorem 4.3 would apply, meaning that it is possible to obtain tight generalization bounds for deep neural networks based on the stability property. To substantiate this intuition, we conduct simple preliminary experiments to estimate the the stability of neural networks in practice. Our empirical findings, presented in Section L, suggest that neural networks are indeed quite stable.
Finally, in Section 5, we present a necessary and sufficient condition for estimability based on the conditional variance of the algorithm's loss. This characterization is formalized in terms of ℓ 2 estimability, which is asymptotically equivalent to average case estimability via Markov's inequality.
Proposition (Proposition 5.2). A is (ε, m)-estimable in ℓ 2 with respect to D if and only if
E[var(L D (A(S)) | S)] ≤ ε.
this section cite: ['b32', 'b25', 'b7']

Section: Examples
We present a few simple examples to showcase the richness of the estimability setting. In this section ε, δ ∈ (0, 1), X is a set, m ∈ N is a sample size, A : (X × {±1}) m → {±1} X is a learning rule, S = ((x 1 , y 1 ), . . . , (x m , y m )) is a training set, and L denotes the 0-1 loss.
Example 1.4 (Perfect learnability does not imply perfect estimability). Let X = [0, 1], let D = ∆(X × {1}) be the set of all distributions of labeled examples (x, y) where x ∈ X and y = 1. The collection D is perfectly learnable, that is, there exists a learning algorithm that always achieves 0 population loss (namely, the learning algorithm that always outputs the constant function h(x) = 1).
Nonetheless, not every learning algorithm is worst-case estimable with respect to D. Indeed, consider the algorithm A that on input S outputs the hypothesis h(x) = -1 x ∈ {x 1 , . . . , x m } +1 otherwise.
For any distribution D ∈ D, L D (A(S)) = D X ({x 1 , . . . , x m }), where D X is the marginal of D on X . Hence, estimating the loss of A is equivalent to a task of support size estimation, which is difficult. Concretely, for any finite set T ⊆ X , let D T = U(T × {1}). Let E be any estimator, and consider an experiment where with probability 1/2, we sample a set T ⊆ X of size |T | = m 2 chosen uniformly at random, and set D = D T , and with probability 1/2 we set D = D U := U(X × {1}). Consider the probability
p = P S∼D m E (S) -L D (A(S)) ≥ 1 2m .
Let E be the event where |{x 1 , . . . , x m }| = m. In the case where D = D T , Claim K.2 implies that P[E] ≥ 1/e. And in the case where D = D U , P[E] = 1. Hence, in both cases, with probability at least 1/e, the estimator receives a sample of m distinct points chosen independently and uniformly from X , and it cannot distinguish between these two cases. However, L D U (A(S)) = 0, whereas L D T (A(S)) = m m 2 = 1 m when E occurs. This implies that p ≥ 1/2e, and so (A, D) is not ( 1 2m , δ, m)-uniformly estimable for any δ < 1/2e. □ Some algorithms are very estimable but are not good learning algorithms, as in the following three examples. Example 1.5 (Constant algorithms are estimable). Let m ≥ log(1/δ)/ε 2 . Let h 0 : X → {±1} be a function, and let A be the constant learning algorithm such that A(S) = h 0 for all S. Then by Hoeffding's inequality, A is (ε, δ, m)-uniformly estimable with respect to the set of all distributions D = ∆(X × {±1}), with estimator E (S) = L S (h 0 ). □ For some algorithms, the empirical loss is not a good estimator, yet the algorithm is still estimable.
Example 1.6 (Memorization). Let Ω log(1/δ)/ε 2 ≤ m ≤ O(ε|X |), and consider the algorithm A that on input S, outputs the hypothesis
h(x) = y {y} = {y i : i ∈ [m] ∧ x i = x} -1 otherwise. For each function f : X → {±1}, let D f = U({(x, f (x)) : x ∈ X }), and let D = {D f : f ∈ {±1} X }. Note that A always has 0 empirical loss, which can be far from the population loss. Nonetheless, (A, D) is (ε, δ)-uniformly estimable, using E (S) = |{i ∈ [m] : y i = 1}| /m. □ Example 1.7 (Most learning rules are estimable). Let d = |X | < ∞, let F = {±1} X , and for each f ∈ F, let D f = U({(x, f (x)) : x ∈ X }). Let A be the set of all mappings (X × {±1}) m → {±1} X
, and consider a mapping A chosen uniformly from the set A. So for every fixed f ∈ F and every fixed sample S of size m consistent with f , A(S) is a function that was chosen uniformly from F . By Hoeffding's inequality,
∀f ∈ F ∀S ∈ supp(D f ) : P A∼U(A) L D f (A(S)) - 1 2 ≥ ε ≤ 2e -2dε 2 .
In particular,
P A∼U(A),f ∼U(F ),S∼(D f ) m L D f (A(S)) - 1 2 ≥ ε ≤ 2e -2dε 2 .
Hence, by Markov's inequality, 99% of learning rules A ∈ A satisfy that (A, {D f } f ∈F ) is (ε, 200e -2dε 2 , m)-estimable on average. □ In both cases, the algorithms are estimable because their loss is guaranteed to be high, namely, the algorithms are poor learners.
Finally, ERM algorithms for learning parity functions are a particularly instructive case. They demonstrate two important phenomena: (1) Estimability can be a very delicate matter, in the sense that changing the sample size by a small additive constant can make all the difference (e.g., any ERM for parities is very estimable with m = d + 10 samples, but not very estimable with m = d); (2) when the sample size is not sufficient for learning all the distributions in the collection D, there can be a trade-off between learning performance and estimability. Algorithms with no inductive bias will perform equally poorly for all distributions, and this makes them estimable. In contrast, algorithms that have an inductive bias towards a subset D ′ ⊆ D can perform well on D ′ , and this can make them less estimable.
Example 1.8 (Parity functions). Let d ∈ N be large enough, X = (F 2 ) d , and let
H = {f w : w ∈ X } ⊆ (F 2 ) X be the class of parity functions such that f w (x) = i∈[d] w i • x i . Let D = {D f } f ∈H with D f = U({(x, f (x)) : x ∈ X }). For a learning rule A and sample size m, let p(m) = P D∼U(D) S∼(D) m [L D (A(S)) = 0].
For sample size m ≥ d + 10, any ERM algorithm for H satisfiesfoot_3 p(m) ≥ 0.999, meaning it learns D well, and hence is (0, 10 -3 , d + 10)-estimable on average.
Similarly, for smaller sample sizes, any ERM for H satisfies p(d) ≥ 0.61, and p(d -1) ≥ 0.38. However, ERM algorithms differ in their degree of estimability for smaller sample sizes. Concretely, there exist ERM algorithms such that for any 6 ≤ m ≤ d there exists a collection D m for which the algorithm is not (0.25, 0.32, m)-estimable on average. In contrast, for the same hard collections D m , ERM algorithms without an inductive bias perform poorly on all distributions for small m, so they are significantly more estimable. □
this section cite: []

Section: Related Works
The works of Nagarajan and Kolter (2019, Theorem 3.1) and Bartlett and Long (2021, Theorem 1) also study cases where generalization bounds fall short of estimating the performance of learning algorithms (while Negrea et al., 2020 provide a response to these claims). They preclude tight algorithm-dependent generalization bounds only for uniform convergence and linear classifiers. Their theorems consider specific distributions (Gaussian in Nagarajan and Kolter, 2019, a different distribution per sample in Bartlett and Long, 2021) and specific types of SGD. In contrast, our work uses the same marginal distribution across all sample sizes, and applies to many algorithms and distributions.
We now mention a few of the algorithm-dependent generalization bounds in the literature. Zhang, Teng, and Zhang (2023) study convex optimization, so their results apply only to a single neuron. While providing matching lower and upper bounds, these bounds match only asymptotically when the sample size n is very large, far from the overparameterized regime relevant for neural networks. Nikolakakis, Haddadpour, Karbasi, and Kalogerias (2023) proposes generalization bounds for algorithms satisfying a certain symmetry property (e.g., full-batch gradient descent) when using smooth losses. These bounds are algorithm-dependent but distribution-free, making no distributional assumptions.
There are a number of information-theoretic generalization bounds that are both algorithm and distribution-dependent, such as Theorem 1 of Xu and Raginsky, 2017. However, such bounds are sometimes difficult to approximate numerically in a tight manner. These bounds are part of the PAC-Bayes framework. 8 Unfortunately, when these PAC-Bayes or informationtheoretic bounds can be approximated in a tight manner, 9 they do not reveal what properties of the (distribution, algorithm) pair allowed for such success in learning and estimation. The works of Haghifam, Moran, Roy, and Dziugiate (2022b) and Rammal, Achille, Golatkar, Diggavi, and Soatto (2022) use the notion of leave-one-out conditional mutual information to derive generalization bounds, which provide another characterization of VC classes and yield non-vacuous generalization bounds for neural networks. Bartlett et al. (2020) establish that benign overfitting in linear regression with Gaussian covariates requires strong control over the data covariance; a line of work initiated by Rubinfeld and Vasilyan (2023) studies learner-tester pairs, where the learner is guaranteed to succeed whenever the distribution is accepted by the tester. These works, like ours, shine light from various angles on the crucial role of distributional assumptions in achieving generalization guarantees.
For a detailed comparison to Gastpar et al. (2024), and further related works, see Appendix A.
this section cite: ['b3', 'b4', 'b33', 'b14']

Section: Preliminaries
All the proofs for theorems appearing in the next section appear in the appendix.
Definition 2.1. For m ∈ N and sets X and Y, a learning rule is a function A : (X × Y) m → Y X . We will also consider learning rules with variable-size input, i.e., A :
(X × Y) * → Y X .
In this paper we informally use the terms 'learning algorithm' and 'learning rule' interchangeably. Both words refer to a function, ignoring considerations of computability. All learning algorithms in this paper are deterministic. 10
Notation 2.2. For a set Ω, we write ∆(Ω) to denote the collection of all probability measures over a measurable space (Ω, F), where F is some fixed σ-algebra that is implicitly understood. We write U(Ω) to denote the uniform distribution over Ω. 10 See Section B for a discussion on how our results can be extended to randomized algorithms.
Definition 2.3. Let m ∈ N, let X , Y be sets, let h : X → Y, let S = ((x 1 , y 1 ), . . . , (x m , y m )) ∈ (X × Y) m ,
this section cite: []

Section: with respect to S is L
S (h) = 1 m i∈[m] 1(h(x i ) ̸ = y i ). The population loss of h with respect to D is L D (h) = P (x,y)∼D [h(x) ̸ = y].
Definition 2.4 (Uniformly tight generalization bound for an algorithm). Let m ∈ N, ε, δ ∈ [0, 1], let X and Y be sets, let m ∈ N, let A : (X × Y) m → Y X be a learning rule, and let b : (X × Y) m → [0, 1] be a generalization bound (that may depend on A). We say that b is uniformly tight for A with precision ε and confidence δ if for any distribution D ∈ ∆(X × Y),
P S∼D m [b(S) -ε ≤ L D (A(S)) ≤ b(S)] ≥ 1 -δ.
Notation 2.5. Let X be a set, let F ⊆ {±1} X be a hypothesis class, and let S ∈ (X × {±1})
* . We denote F S = {f ∈ F : L S (f ) = 0}.
The following definition captures the notion of a learning rule having an inductive bias towards a particular set of hypotheses.
Definition 2.6. Let m ∈ N, let X be a set, and let F ⊆ {±1} X be a hypothesis class. We say that a learning rule A :
(X × {±1}) m → {±1} X is F-interpolating if A(S) ∈ F S for every sample S ∈ (X × {±1}) m such that F S ̸ = ∅.
Remark 2.7. The property of F-interpolation is similar to the more common property of proper empirical risk minimization (proper ERM) for F. However, F-interpolation is a slightly weaker requirement. Specifically, if S is not F-realizable (i.e., F S = ∅), then an F-interpolating learning rule may output any function in {±1} X , whereas a proper learning rule for F must always output a function from F.
Definition 2.8. Let ε ≥ 0, let X be a set, and let F ⊆ {±1} X be a hypothesis class. We say that F is ε-orthogonal with respect to X , denoted F ∈ ⊥ ε,X , if every distinct f, g ∈ F satisfy
E x∼U(X ) [f (x)g(x)] ≤ ε.
For simplicity, we write F ∈ ⊥ ε when X is understood from context.
Proposition 2.9. Let ε > 0 and let F ⊆ {±1} X be ε-orthogonal. Then for any distinct f, g ∈ F,
1 2 - ε 2 ≤ P x∼U(X ) [f (x) = g(x)] ≤ 1 2 + ε 2 . Proof. P x∼U(X ) [f (x) = g(x)] = E x∼U(X ) [1(f (x) = g(x))] = E x∼U(X ) 1 + f (x)g(x) 2 = 1 2 + 1 2 • E x∼U(X ) [f (x)g(x)].
this section cite: []

Section: Conditions that Preclude Estimability
We present two conditions that preclude estimability.
this section cite: []

Section: Inestimability for VC Classes
Theorem 3.1. There exists d 0 > 0 as follows. For any integer d ≥ d 0 , let X be a set, let H ⊆ {±1} X such that VC(H) = d, and let m ∈ N such that m ≤ √ d/10. Then there exists a subset F ⊆ H and a collection D ⊆ ∆(X × {±1}) of F-realizable distributions such that for any F-interpolating learning rule A and for any estimator E : (X × {±1}) m → [0, 1] that may depend on D and A,
P D∼U(D) S∼D m E (S) -L D (A(S)) ≥ 1 4 - 1 2d 1/4 ≥ 1 6 . (4
)
We note that some of the constants appearing in the theorem were chosen for simplicity, and may be slightly improved.
this section cite: []

Section: Inestimability for Nearly-Orthogonal Functions
Theorem 3.2. Let m ∈ N, let X be a set, and let A : (X × {±1}) m → {±1} X be a learning rule. Assume that A is F-interpolating for a set F ⊆ {±1} X ′ where X ′ ⊆ X , 100m 2 ≤ |X ′ | < ∞, F ∈ ⊥ 1/1000m,X ′ and |F| = 2 m + 1. Then there exists a collection of F-realizable distributions D ⊆ ∆(X ′ × {±1}) such that for any estimator function E : (X × {±1}) m → [0, 1] that may depend on D and A,
P D∼U(D) S∼D m E (S) -L D (A(S)) ≥ 1 4 - 1 4000m ≥ 0.16.
We note that here too, the constants appearing in the theorem were chosen for simplicity, and might be improved.
this section cite: []

Section: Sufficient Conditions for Estimability
In Examples 1.5 and 1.6 we saw that the constant algorithm and the memorization algorithm are very estimable. These algorithms are also very stable. Indeed, they always output the same (or essentially the same) hypothesis. 11 In the other direction, Theorem 3.2 shows that certain algorithms that are very unstable, are not estimable. This suggests that stability might play an important role in determining the estimability of an algorithm.
One notion of algorithmic stability that is common in the literature is leave-one-out stability (Bousquet & Elisseeff, 2002). However, it is easy to see that the memorization algorithm, which is estimable and is (intuitively) very stable, does not satisfy their definition of stability. Therefore, we use the following alternative definitions of algorithmic stability, which are similar to Rogers and Wagner (1978) and Kearns and Ron (1999).
Definition 4.1. Let m, k ∈ N, k < m, α, β ∈ [0, 1]. Let X be a set, let A : (X × {±1}) * → {±1} X be a learning rule, and let D ⊆ ∆(X × {±1}). We say that A is (α, β, m, k)-hypothesis stable with respect to D if ∀D ∈ D :
P S1∼D m-k S2∼D k [dist D X (A(S 1 ), A(S 1 • S 2 )) ≤ α] ≥ 1 -β,
where D X is the marginal of D on X , dist P (f, g) = P x∼P [f (x) ̸ = g(x)], and • denotes concatenation.
Definition 4.2. In the notation of Definition 4.1, we say that A is (α, β, m, k)-loss stable with respect to D if ∀D ∈ D :
P S1∼D m-k ,S2∼D k L D (A(S 1 )) -L D (A(S 1 • S 2 )) ≤ α ≥ 1-β.
Theorem 4.3. Let k ∈ N and α 0 , β 0 ∈ (0, 1) such that k ≥ Ω log(1/β 0 )/α 2 0 . Let A be a learning rule that is (α 1 , β 1 , m, k)-hypothesis stable or loss stable with respect to D (as in Definitions 4.1 and 4.2). Then (A, D) is (ε = α 0 + α 1 , δ = β 0 + β 1 , m)-uniformly estimable.
Hence, stability is a sufficient condition for estimability. We remark that it is not a necessary condition. For instance, a learning rule selected at random as in Example 1.7 most likely is estimable (because it has high loss for any distribution), but not hypothesis stable (since for each possible input sample, it outputs a different hypothesis that was chosen at random). To see that loss stability is also not necessary for estimability, fix a degenerate distribution D such that D((x * , 1)) = 1 for some x * , and consider an algorithm A that for samples of size m outputs the constant hypothesis h 1 (x) = 1, and for samples of size m -k outputs the constant hypothesis h 0 (x) = 0. A is perfectly estimable with respect to {D}, but it is not loss stable.
One might object that Theorem 4.3 is of limited utility, because it is hard to check whether a given algorithm is hypothesis stable or loss stable. Our response to this criticism is that in practice, it is quite easy to check whether an algorithm is loss (or hypothesis) stable with respect to a particular population distribution -and indeed we do so in our experiments (see Section L).
The process for estimating loss stability is simple: take a set S of m i.i.d. labeled samples from the population distribution. Randomly choose a subset S ′ of size m -k. Execute the learning algorithm twice, once with training set S to produce a hypothesis h, and another time with training set S ′ to produce a hypothesis h ′ . Use an additional validation set to estimate the difference in population loss between h and h ′ . Repeating this process a number of times and taking an average gives a good estimate of the (m, k)-loss stability. A similar process can be used to estimate hypothesis stability. Simply measure the disagreement between h and h ′ on the validation set (note that in this case, the validation set can be unlabeled, which is an advantage when labeling data is expensive).
this section cite: ['b7', 'b32', 'b25']

Section: A Simple Characterization
The following definition is a variant of Definition 1.1. Such a variant allows us to have a simple characterization of estimability in Proposition 5.2. Namely, to understand whether an algorithm is estimable with respect to a set of distributions, one can examine the quantity
E D∼U(D),S∼D m [var(L D (A(S)) | S)].
Definition 5.1. Let D be a set of distributions and let A be a learning algorithm. We say that A is (ε, m)-estimable in ℓ 2 with respect to D, if there exists an estimator E such that
E D∼U(D),S∼D m (E (S) -L D (A(S))) 2 ≤ ε
We remark that for bounded loss functions, one can move between Definition 5.1 and Definition 1.1 using Markov's inequality. Furthermore, although the characterization in the following theorem is simple, it might provide a technical condition that will be useful for future work.
Proposition 5.2. A is (ε, m)-estimable in ℓ 2 with respect to D if and only if
v := E D∼U(D),S∼D m [var(L D (A(S)) | S)] ≤ ε.
Conceptually, one can think of the quantity v in the case of discrete random variables as follows. A distribution D is samples uniformly from D, and then a sample S is taken i.i.d. from D. This defines a joint distribution (D, S). For every fixed sample s,
v s := var(L D (A(S)) | S = s) = E L -E[L|S = s] 2 | S = s , where L = L D (A(S)). Finally, v is the average v = s P[S = s] • v s .
The same idea applies to general random variables with the appropriate modifications.
The proof of Proposition 5.2 appears in Appendix C.
this section cite: []

Section: Directions for Future Work
Some interesting questions for future work include:
1. Theorems 3.1 and 3.2 imply limitations for estimability when the sample is of size m ≤ O √ d or m ≤ O 3 √ d respectively, where d is the VC dimension of the class of labeling functions (see Footnote 5). Is it possible to obtain similar inestimability results also for larger m? 2. Are there simple conditions that are necessary and sufficient for an algorithm to have tight generalization bounds for all distributions (as in Example 1.5)?
3. Are there simple conditions that are necessary and sufficient for a pair (A, D) of an algorithm and a distribution family to have tight generalization bounds?
symposium on theory of computing, STOC 2023STOC , orlando, fl, usa, june 20-23, 2023STOC (pp. 1643STOC -1656)). ACM. https://doi.org/10.1145/3564246.3585117 Viallard, P., Emonet, R., Habrard, A., Morvant, E., & Zantedeschi, V. (2024). Leveraging pac-bayes theory and gibbs distributions for generalization bounds with complexity measures. In S. Dasgupta, S. Mandt, & Y. Li (Eds.), International conference on artificial intelligence and statistics, 2-4 may 2024, palau de congressos, valencia, spain (pp. 3007-3015, Vol. 238). PMLR. https : / / proceedings . mlr . press / v238 /  viallard24a.html Wang, Z., & Mao, Y. (2023). Tighter information-theoretic generalization bounds from supersamples. arXiv preprint arXiv:2302.02432. Xu, A., & Raginsky, M. (2017). Information-theoretic analysis of generalization capability of learning algorithms. Advances in Neural Information Processing Systems, 30. Zhang, P., Teng, J., & Zhang, J. (2023). Lower generalization bounds for gd and sgd in smooth stochastic convex optimization. arXiv preprint arXiv:2303.10758.
this section cite: []

Section: Technical Appendices and Supplementary Material A Further Discussion of Related Works

this section cite: []

Section: A.1 Comparison to Gastpar et al. (2024)
The estimability setting studied in our paper was introduced by Gastpar, Nachum, Shafer, and Weinberger (2024). In Theorem 3 of their paper, they show a limitation on estimability (a learnability-estimability trade-off) for algorithm-dependent bounds that is fairly abstract and involves a total variation condition that might be hard to check in many cases. In contrast, Theorems 3.1 and 3.2 involve very concrete combinatorial and geometric conditions (VC dimension, orthogonal functions). Theorems 4 and 5 in their paper are more concrete, but they hold only for exactly orthogonal functions with strict algebraic structure (parity functions). In contrast, our Theorem 3.2 applies generally to any nearly-orthogonal function class (including classes that are exactly-orthogonal as a special case).
Unlike Gastpar et al. (2024), our work also presents positive results (Theorem 4.3 and Proposition 5.2), showing cases where generalization bounds for specific algorithms can be tight (even if, e.g., uniform convergence does not hold). The conceptual connections between estimability, stability and conditional variance appearing in those results was not present in Gastpar et al. (2024).
Finally, our techniques also differ from those of Gastpar et al. (2024). We use the Johnson-Lindenstrauss lemma, our technical lemma (Lemma H.1), and the duality of linear programming -expanding the arsenal of tools readily available for the study of estimability.
In summary, our work builds upon the foundation laid by Gastpar et al. (2024), but we make several important contributions that go beyond their results.
this section cite: ['b14', 'b14', 'b14', 'b14', 'b14']

Section: A.2 Stability
In Definitions 4.1 and 4.2, we formalize simple stability conditions that guarantee the existence of tight generalization bounds, as we show in Theorem 4.3. There are many definitions of stability in the literature, and it is important to appreciate that Theorem 4.3 makes a nontrivial conceptual contribution by identifying the "correct" notion of stability for understanding estimability.
Definitions 4.1 and 4.2 are similar to the definition of hypothesis stability and loss stability in Kearns and Ron (1999), Elisseeff, Evgeniou, and Pontil (2005), and Rogers and Wagner (1978). Lei, Jin, and Ying (2022) use another similar definition for stability and utilize it to derive generalization bounds for GD and SGD.
In contrast, our definitions of stability are also reminiscent of the replace-one stability in Bousquet and Elisseeff (2002), but as we explain in Section 4, our definitions overcome an important limitation present in their definition. In particular, the memorization algorithm (Example 1.6), which is very estimable, is not stable according to the definition of stability of Bousquet and Elisseeff (2002), but it is stable according to our definitions.
this section cite: ['b25', 'b12', 'b32', 'b26', 'b7', 'b7']

Section: A.3 Neural Tangent Kernel and Mean-Field Theory
There are many works that study generalization using the neural tangent kernel (NTK) or mean-field theory (MFT) approach. 12 To the best of our knowledge, these works do not provide general necessary or sufficient conditions for generalization bounds to be tight, which is the focus of our work. Additionally, they study generalization bounds for fairly specific families of algorithms such as gradient descent (or idealized versions thereof), while our work applies to a broader and more general class of algorithms.foot_6
this section cite: []

Section: B On Extending Our Results to Randomized Algorithms
For simplicity, in this paper we focus on deterministic learning rules. However, we recognize that the topic of randomized learning algorithms is very important, seeing as most algorithms used in practice today are randomized.
The estimability framework explored in this paper can be extended to handle randomized algorithms as well, and in fact the original work of Gastpar et al. (2024) already contains some initial treatment of randomized algorithms.
We expect that the results presented in this paper can be extended to randomized algorithms, and that the essence of the results remains mostly unchanged.
The first step in such an extension would be to clearly define what estimability means for randomized learning algorithms. A definition that one might initially consider is one where the estimator knows the randomness used by the algorithm, and must output a number that is with high probability close to the true population loss of the randomized algorithm. This definition is not very interesting, because a setting in which the estimator knows the randomness used by the randomized algorithm is equivalent to the setting of a deterministic algorithm, which is already covered by the results in this paper. Nonetheless, it is good to keep this definition in mind, because it means that our results for deterministic algorithms already apply as-is to randomized algorithms (like SGD) once the randomly chosen seed is fixed, which might be a simple and satisfactory approach for many purposes (SGD with a fixed random seed typically performs as well for most purposes as SGD with a fresh randomly-chosen seed).
Perhaps the more "correct" and interesting definition of estimability for randomized learning algorithms is one where the estimator knows the training set, but does not know the randomness used by the learning algorithm, and it is required to output a number that is close with high probability to the expected population loss of the randomized algorithm when executed with this training set (where the expectation is over the randomness of the algorithm). In this setting, we believe the essence of our results carries through, with an important conceptual difference: using randomness, one can always engineer a learning algorithm that is estimable, essentially by adding noise to the output of the algorithm. As the noise in the algorithm's output increases, the expected 0-1 loss of the algorithm becomes closer to 1 /2, and so the algorithm becomes estimable with a trivial estimator that simply always outputs the number 1 /2. (With intermediate amounts of noise, a number between 0 and 1 /2 will be optimal).
Consequently, for randomized algorithms, our lower bounds in Theorems 3.1 and 3.2 can no longer be stated as absolute limitations on estimability. Rather there is now a trade-off between the performance of the algorithm and its estimability. As one adds more noise, the algorithm becomes more estimable, but its performance degrades. Thus, the corresponding theorems for randomized algorithm would state that no algorithm can simultaneously make good predictions for some large set of labeling functions and also be estimable.
On the other hand, the upper bound in Theorem 4.3 that states that stable algorithms are estimable remains basically unchanged for randomized algorithms.
To summarize, under a suitable definition of estimability for randomized algorithms, we expect that our results would not change much, though the statement (and proof) of the lower bounds would be somewhat more complex. We leave this work to future research.
this section cite: ['b14']

Section: C Proof of Proposition 5.2
Proof of Proposition 5.2. The result that the minimum mean-square error (MMSE) estimator corresponds to the conditional expectation is a well-established theorem in probability theory (see, for instance, Section 7.9 in Grimmett and Stirzaker (2020)). For the sake of completeness, we present a proof of this result.
We will use the following simple claim.
Claim C.1. Let c 1 , ..., c k , p 1 , ..., p k ∈ R such that k i=1 p i = 1, then argmin x∈R k i=1 p i • (x -c i ) 2 = k i=1 p i • c i .
The claim follows by taking the derivative of k i=1 p i • (x -c i ) 2 with respect to x which yields the equation:
k i=1 2p i (x -c i ) = 0 that implies x = k i=1 p i c i since k i=1 p i = 1.
The following shows that the estimator E * (S) := E [L D (A(S)) | S] is optimal and the inequality follows from Claim C.1. Let E be any estimator for A.
E D∼U(D),S∼D m (E (S) -L D (A(S))) 2 = S P(S) D∈D P(D|S) (L D (A(S)) -E (S)) 2 = E D∈D P(D|S) (L D (A(S)) -E (S)) 2 ≥ E   D∈D P(D|S) L D (A(S)) - D∈D [P(D|S)L D (A(S))] 2   = E D∈D P(D|S) (L D (A(S)) -E * (S)) 2 = E D∼U(D),S∼D m (E * (S) -L D (A(S))) 2 .
This means that A is square loss (ε, m)-estimable with respect to D if and only if E * can achieve ε accuracy. It achieves such accuracy if and only if E [var(L D (A(S)) | S)] ≤ ε. This follows by the following equalities that complete the proof.
E [var(L D (A(S)) | S)] = E E (L D (A(S)) -E [L D (A(S))|S]) 2 | S = E D∈D P(D|S) (L D (A(S)) -E [L D (A(S)) | S]) 2 = E   D∈D P(D|S) L D (A(S)) - D∈D [P(D|S)L D (A(S))] 2   = E D∈D P(D|S) (L D (A(S)) -E * (S)) 2 = E D∼U(D),S∼D m (E * (S) -L D (A(S))) 2
this section cite: ['b15']

Section: D Details for Example 1.8
For sample size m ≥ d + 10, any ERM algorithm for H satisfies p(m) ≥ 0.999, meaning it learns D well, and hence is (0, 10 -3 , d + 10)-estimable on average. This holds because for an ERM to output the ground truth, it is clearly sufficient that only a single sample-consistent function exists in the concept class (the ground truth). Similarly, in the event that there are t > 1 sample-consistent functions, the success probability is given by 1/t due to the uniform prior over ground truth distributions. Parity functions are fully characterized by their coefficient vector w = [w 1 , . . . , w d ]. Since the labels y are a bilinear function in the inputs x and coefficients w, one can obtain w from m ≥ d linearly independent samples x i by solving the linear system of equation y = Xw with design matrix X ∈ {0, 1} m×d . More generally, X having rank d -k is equivalent to the event of having t = 2 k sample-consistent functions (coefficient vectors) since every additional linearly independent row rules out half of all parity functions. Now assume X consists of all i.i.d. Ber(½) entries and y contains the labels of all samples. The probability of zero population loss can now be obtained from the law of total probability with the probabilities of rank deficiency computed according to Corollary 2.2 in Blake and Studholme (2006).
Similar calculations show that for smaller sample sizes, any ERM for H satisfies p(d) ≥ 0.61, and p(d -1) ≥ 0.38. An application of Theorem 5 in Gastpar et al. (2024) shows that there exist ERM algorithms such that for any 6 ≤ m ≤ d there exists a collection D m for which the algorithm is not (0.25, 0.32, m)-estimable on average. These algorithms have an inductive bias towards a subset F ⊆ H, such that they perform well for distributions labeled by a function from F , and perform poorly for target functions from the complement of F.
this section cite: ['b6', 'b14']

Section: E Proof of Theorem 3.1
Recall the definition of nearly-orthogonal functions (Definition 2.8). The proof of Theorem 3.1 uses a corollary of the Johnson-Lindenstrauss lemma (Theorem K.1), which states that random vectors in a high dimensional space are nearly orthogonal, as follows.
14
Claim E.1. Let ε ∈ (0, 1/2), and let d, n ∈ N such that n ≤ exp dε 2 /54 .
Let U = U {±1} [d] be the uniform distribution over functions [d] → {±1}, and consider a random sequence F of functions F 1 , . . . , F n sampled independently from U. Then
P F ∼U n F ∈ ⊥ ε,[d] ≥ 0.99. Proof of Claim E.1. If n = 1
there is nothing to prove, so we assume n ≥ 2. Let R ∼ U {±1} d×n be a d × n matrix with entries in {±1} chosen independently and uniformly at random. In particular, for each i ∈ [n], the i-th column of R is a vector of d numbers in {±1} chosen independently and uniformly at random. Hence, using e 1 , . . . , e n to denote the standard basis of R n , we identify the vector Re i , which is the i-th column of R, with the random function
F i : [d] → {±1}. Recall that for vectors u, v ∈ R d , ∥u -v∥ 2 2 = ⟨u -v, u -v⟩ = ∥u∥ 2 2 -2 ⟨u, v⟩ + ∥v∥ 2 2 , so ⟨u, v⟩ = ∥u∥ 2 2 + ∥v∥ 2 2 -∥u -v∥ 2 2 2 . (5
)
Invoking Theorem K.1 with s = n, β = 7, V = {e 1 , . . . , e n } ⊆ R n , and d, n, ε as in the claim statement implies that
P R∼U({±1} d×n )   ∀i, j ∈ [n], i ̸ = j : (1 -ε) • 2 ≤ 1 √ d Re i -1 √ d Re j 2 2 ≤ (1 + ε) • 2   ≥ 1 - 1 n β .(6)
Hence, with probability at least 1 -1/n β ≥ 1 -1/2 7 ≥ 0.99 over the choice of F , every distinct i, j ∈ [n] satisfy
E x∼U([d]) [F i (x)F j (x)] = 1 d x∈[d] F i (x)F j (x) = 1 d ⟨Re i , Re j ⟩ (Identifying F i with Re i ) = ∥Re i ∥ 2 2 + ∥Re j ∥ 2 2 -∥Re i -Re i ∥ 2 2 2d (By Eq. (5)) = 1 - 1 2 1 √ d Re i - 1 √ d Re i 2 2
≤ ε, (By Eq. ( 6)) as desired.
14 It is also possible to prove a similar claim by directly using concentration of measure (e.g., Hoeffding's inequality), without using the Johnson-Lindenstrauss lemma.
Proof of Theorem 3.1. Fix an H-shattered set X d ⊆ X with cardinality |X d | = d, and for each f :
X d → {±1} let D f = U({(x, f (x)) : x ∈ X d }).
Note that the distributions D f are H-realizable. We will show that there exists a collection D = {D f : f ∈ F} that satisfies Eq. ( 4), where F ⊆ {±1} X d is a set of k = 2 m + 1 functions.
Consider the following experiment:
1. Sample a sequence of functions G = (G 1 , . . . , G k ) independently and uniformly at random from {±1} X d .
this section cite: []

Section: Sample a function F uniformly from G.
3. Sample a sequence of points X = (X 1 , . . . , X m ) independently and uniformly at random from
X d . (X is sampled independently of (G, F ).) 4. For each i ∈ [m], let Y i = F (X i ), let Y = (Y 1 , . . . , Y m ), and let S = (X 1 , Y 1 ), . . . , (X m , Y m ) .
Let P be the joint distribution of (G, F, X, Y, S). Consider the following events:
• E 1 = {G ∈ ⊥ ε,X d } for ε = 2/d 1/4 . By Claim E.1 and the choice of k, P(E 1 ) ≥ 0.99 for d large enough. 15 • E 2 = |{X 1 , . . . , X m }| = m . By Claim K.
2 and the choice of m, P(E 2 ) ≥ 0.99.
•
E 3 = |G S | = 2 . P(E 3 | E 2 ) ≥ 1/e. To see this, note that each function G i ∈ G \ {F } is chosen independently of F . Hence, the probability that a function G i agrees with F on the m distinct samples in X (i.e., the probability that G i (X j ) = F (X j ) for all j ∈ [m], given E 2 ) is p = 2 -m . The functions in G are chosen independently, so the number T of functions in G \ {F } that agree with F on m distinct samples has a binomial distribution T ∼ Bin(k -1, p). So P[T = 1] = (k -1) • p • (1 -p) k-2 = (1 -p) k-2 ≥ e -p 1-p k-2 (∀p < 1 : 1 -p ≥ e -p/(1-p) ) = 1/e. Let E = E 1 ∩ E 3 .
Combining the above bounds yields
P(E) = P(E 1 ∩ E 3 ) ≥ P(E 3 ) -P E C 1 ≥ P(E 3 | E 2 ) • P(E 2 ) -P E C 1 ≥ 0.99 • 1/e -0.01 > 1/3.
By an averaging argument, this implies that there exists F ⊆ {±1}
X d such that F ∈ ⊥ ε,X d for ε = 2/d 1/4 and P(|G S | = 2 | G = F ) ≥ 1/3.(7)
Fix this F, and let A be an F-interpolating learning rule. From the technical lemma (Lemma H.1), there exists a collection of F-realizable distributions D ⊆ ∆(X d × {±1}) such that for any estimator E : (X × {±1}) m → [0, 1] that may depend on D and A,
P D∼U(D) S∼D m E (S) -L D (A(S)) ≥ 1 4 - ε 4 ≥ 1 2 • P D∼U(D) S∼D m [|F S | = 2] ≥ 1 2 • 1 3 = 1 6 , (By Eq. (7))
as desired.
this section cite: []

Section: F Proof of Theorem 3.2
Proof of Theorem 3.2. We take D = {D f : f ∈ F} where
D f = U({(x, f (x)) : x ∈ X }). Fix a function f * ∈ F, let S ∼ (D f * ) m
, and consider the random variable Z = |F S |. We bound the expectation and variance of Z, and then show a lower bound on the probability that Z ∈ {2, 3}.
Let S = (X 1 , Y 1 ), . . . , (X m , Y m ) and X = {X 1 , . . . , X m }, and let E denote the event in which |X| = m (i.e., S is collision-free). For each f ∈ F, let Z f = 1(∀i ∈ [m] : f (X i ) = Y i ), so that Z = f ∈F Z f . E S∼(D f * ) m [Z | E] = E   f ∈F Z f E   = 1 + f ∈F f ̸ =f * P ∀i ∈ [m] : f (X i ) = Y i E (Z F = 1) ≤ 1 + 2 m • 1 2 + 1 2 • 1 1000m m (By Proposition 2.9) ≤ 1 + e 1/1000 < 2.002. (8) E S∼(D f * ) m [Z | E] ≥ 1 + 2 m • 1 2 - 1 2 • 1 1000m m (By Proposition 2.9) ≥ 1 + e -1/500 . (1 -x ≥ e -x/(1-x) ) (9
) E S∼(D f * ) m Z 2 | E = E     f ∈F Z f     g∈F Z g   E   = E         1 + f ∈F f ̸ =f * Z f         1 + g∈F g̸ =f * Z g     E     (Z f * = 1) = E     1 + 2 f ∈F f ̸ =f * Z f + f ∈F f ̸ =f * g∈F g̸ =f * Z f Z g E     = E     1 + 3 f ∈F f ̸ =f * Z f + f,g∈F \{f * } f ̸ =g Z f Z g E     = 1 + 3 (E[Z | E] -1) + f,g∈F \{f * } f ̸ =g E Z f Z g E . (10
) f,g∈F \{f * } f ̸ =g E Z f Z g E = f,g∈F \{f * } f ̸ =g P ∀i ∈ [m] : f (X i ) = g(X i ) = f * (X i ) E ≤ 2 2m • 1 4 + 3 4 • 1 1000m m (By Claim J.1) = 1 + 3 1000m m ≤ e 3/1000 . (11
) Combining Eqs. (8) to (11) yields Var[Z | E] = E Z 2 | E -(E[Z | E]) 2 ≤ 1 + 3e 1/1000 + e 3/1000 -1 + e -1/500 2 < 1.02. By Lemma I.1, P[Z ∈ {2, 3} | E] ≥ 1 -Var[Z | E] 2 ≥ 0.49. Claim K.2 and |X ′ | ≥ 100m 2 imply that P[E] ≥ 0.99. Hence, P[Z ∈ {2, 3}] ≥ P[E] • P[Z ∈ {2, 3} | E] ≥ 0.99 • 0.49 ≥ 0.48. (12)
Finally, invoking our technical lemma (Lemma H.1) yields
P F ∼U(F ) S∼(D F ) m E (S) -L D F (A(S)) ≥ 1 4 - 1 4000m ≥ P[Z ∈ {2, 3}] 3 ≥ 0.16, as desired.
this section cite: []

Section: G Proof of Theorem 4.3
Proof. If (A, D) is (α, β, m, k)-hypothesis stable, then in particular (A, D) is also (α, β, m, k)loss stable. Hence, it suffices to prove the claim for the case of loss stability. We construct a uniform estimator E as follows. Given a sample S ∈ Z m for Z = (X × {±1}), let S 1 • S 2 = S be the partition of S such that S 1 ∈ Z m-k and S 2 ∈ Z k . Take E (S) = L S2 (A(S 1 )).
By the triangle inequality,
|E (S) -L D (A(S))| ≤ |E (S) -L D (A(S 1 ))| + |L D (A(S 1 )) -L D (A(S))| , so P S∼D m [|E (S) -L D (A(S))| > ε] ≤ P |L S2 (A(S 1 )) -L D (A(S 1 ))| > α 0 ∨ |L D (A(S 1 )) -L D (A(S))| > α 1 ≤ P[|L S2 (A(S 1 )) -L D (A(S 1 ))| > α 0 ] + P[|L D (A(S 1 )) -L D (A(S))| > α 1 ] ≤ β 0 + β 1 = δ,
where the final step follows from Hoeffding's inequality, the choice of k, and the stability of A.
this section cite: []

Section: H Technical Lemma for Inestimability
Lemma H.1. Let m ∈ N, let ε > 0, let X be a finite set, let F ⊆ {±1} X such that F ∈ ⊥ ε,X , and let A : (X × {±1}) m → {±1} X be an F-interpolating learning rule. For each f ∈ F let D f = U({(x, f (x)) : x ∈ X }), and for each k ∈ N let
p k = P F ∼U(F ) S∼(D F ) m [|F S | = k].
Then for any estimator E : (X × {±1}) m → [0, 1] that may depend on A,
P F ∼U(F ) S∼(D F ) m E (S) -L D F (A(S)) ≥ 1 4 - ε 4 ≥ k∈{2,...,|F |} p k k .
this section cite: []

Section: I Concentration Bound via Linear Programming
Lemma I.1. Let n ∈ N, v max ∈ R. Let Z be a random variable taking values in [n] such that µ = E[Z] ∈ [2, √ 2 + 1] and Var[Z] ≤ v max . Then P[Z ∈ {2, 3}] ≥ 1 -v max /2.
We prove this concentration of measure bound using the duality of linear programs (see Section 7.4.1 in Boyd and Vandenberghe, 2014 for an exposition of this approach).
Proof. Let Z ′ = Z -µ. Z ′ is a random variable with E[Z ′ ] = 0 and Var[Z ′ ] = Var [Z]. Furthermore, P[Z ∈ {2, 3}] = P[Z ′ ∈ {2 -µ, 3 -µ}]. We show a lower bound on P[Z ′ ∈ {2 -µ, 3 -µ}] across all distribution of Z ′ with the above moment constraints.
Indeed, let X be a random variable taking values in {1-µ, 2-µ, . . . , n-µ} with E[X] = 0 and Var[X] ≤ v max such that P[X ∈ {2 -µ, 3 -µ}] is minimal. In particular, the distribution of X is a solution to the following minimization problem. min
D X P[X ∈ {2 -µ, 3 -µ}] s.t. E[X] = 0 Var[X] ≤ v max
The minimization problem can be formulated as a linear program with variables p
k = P[X = k -µ] for each k ∈ [n]. min D X p 2 + p 3 s.t. k∈[n] p k ≥ 1 k∈[n] -p k ≥ -1 k∈[n] p k • (k -µ) ≥ 0 k∈[n] p k • (µ -k) ≥ 0 k∈[n] -p k • (k -µ) 2 ≥ -v max ∀k ∈ [n] : p k ≥ 0.
This linear program can be represented as
min (0, 1, 1, 0, . . . , 0) • p s.t.      1 1 . . . 1 -1 -1 . . . -1 1 -µ 2 -µ . . . n -µ µ -1 µ -2 . . . µ -n -(1 -µ) 2 -(2 -µ) 2 . . . -(n -µ) 2         p 1 . . . p n    ≥      1 -1 0 0 -v max      p ≥ 0.
Recall the symmetric duality
min c T x max b T y s.t. ↭ s.t. Ax ≥ b A T y ≤ c x ≥ 0 y ≥ 0. Hence, the dual linear program is max (1, -1, 0, 0, -v max ) • y s.t.       1 -1 1 -µ µ -1 -(1 -µ) 2 1 -1 2 -µ µ -2 -(2 -µ) 2 1 -1 3 -µ µ -3 -(3 -µ) 2 . . . 1 -1 n -µ µ -n -(n -µ) 2          y 1 . . . y 5    ≤         0 1 1 0 . . . 0         y ≥ 0.
A direct calculation shows that the vector
y * = 1, 0, α, 0, 1 2 , α = 1 µ -1 - µ -1 2
is a feasible solution for the dual program for any µ ∈ [2, √ 2 + 1]. The value of the dual program at y * is u = 1 -v max /2. The weak duality theorem for linear programs implies that u is a lower bound on the value of the primal problem. Hence, min P[X ∈ {2 -µ, 3 -µ}] ≥ u. This implies that P[Z ∈ {2, 3}] ≥ u, as desired.
this section cite: ['b8']

Section: J Agreement Between Nearly-Orthogonal Functions
Claim J.1. Let ε > 0, let X be a set, and let f, g, h : X → {±1} such that {f, g, h} ∈ ⊥ ε,X . Then P x∼U(X ) [f (x) = g(x) = h(x)] ≤ 1 4 + 3ε 4 . Proof. Denote a = P x∼U(X ) [f (x) = g(x) = h(x)] b = P x∼U(X ) [f (x) ̸ = g(x) = h(x)] c = P x∼U(X ) [f (x) = g(x) ̸ = h(x)] d = P x∼U(X ) [f (x) ̸ = g(x) ̸ = h(x)]
From {f, g, h} ∈ ⊥ ε,X and Proposition 2.9,
a + b = P x∼U(X ) [g(x) = h(x)] ≤ 1 2 + ε 2 a + c = P x∼U(X ) [f (x) = g(x)] ≤ 1 2 + ε 2 a + d = P x∼U(X ) [f (x) = h(x)] ≤ 1 2 + ε 2 .
Adding these inequalities yields
3a + b + c + d ≤ 3 2 + 3ε 2 .
From the identity a
+ b + c + d = 1, 2a ≤ 1 2 + 3ε 2 ,
so a ≤ 1 4 + 3ε 4 , as desired.
this section cite: []

Section: K Miscellaneous Lemmas
The following result from Achlioptas (2003) is a variant of a lemma of Johnson and Lindenstrauss (1984).
Theorem K.1 (Johnson-Lindenstrauss). Let n, s ∈ N, let ε, β > 0, and let V ⊆ R s be a set with cardinality |V | = n. Let d ∈ N such that
d ≥ 4 + 2β ε 2 /2 -ε 3 /3 ln(n).
Let R be a d × s random matrix such that each entry is chosen independently and uniformly at random from {±1}. Let f R : R s → R d be given by
f R (v) = (1/ √ d) • Rv. Then P R∼U({±1} d×s ) ∀u, v ∈ V : (1 -ε)∥u -v∥ 2 2 ≤ ∥f R (u) -f R (v)∥ 2 2 ≤ (1 + ε)∥u -v∥ 2 2 ≥ 1- 1 n β .
Claim K.2 (Converse to Birthday Paradox). Let d, m ∈ N, and let β ∈ (0, 1). If
m ≤ min d ln 1 β , d 2 then P X∼(U([d])) m [|X| = m] ≥ β.
Proof. We use the inequality 1 -x ≥ e -x/(1-x) , which holds for x < 1.
P X∼(U([d])) m [|X| = m] = 1 • 1 - 1 d • 1 - 2 d • • • 1 - m -1 d ≥ m-1 k=0 exp - k d -k = exp - m-1 k=0 k d -k ( * ) ≥ exp - 2 d m-1 k=0 k ≥ exp - m 2 d ,
where ( * ) follows from m ≤ d/2. Solving expm 2 d ≥ β yields the desired bound.
Theorem K.3 (Hoeffding, 1963). Let a, b, µ ∈ R and m ∈ N. Let Z 1 , . . . , Z m be a sequence of i.i.d. real-valued random variables and let Z = 1 m m i=1 Z i . Assume that E[Z] = µ, and for every i ∈ [m], P[a ≤ Z i ≤ b] = 1. Then, for any ε > 0,
P[|Z -µ| > ε] ≤ 2 exp -2mε 2 (b -a) 2 .
this section cite: ['b1', 'b24', 'b20']

Section: L Experiments

this section cite: []

Section: L.1 Motivation and Setup
Here, we examine if there are practical algorithms that admit loss stability or even hypothesis stability with substantial numerical values. To this end, we conduct experiments over a simple neural network architecture across four datasets: MNIST, FashionMNIST, CIFAR10, and CIFAR10 with random labels (figures 1-4, respectively). Throughout all experiments, we employ one-hidden-layer perceptrons with 512 hidden neurons. We train the models using stochastic gradient descent (SGD) with a momentum factor of 0.9 and a batch size of 1000, optimizing the cross-entropy loss. For every data set, we train the models across learning rates 0.1, 0.035,foot_8 and 0.01. We average all the curves over 10 random seeds (tied for the pairs of networks) and plot the standard deviation for all the curves.
The training procedure is as follows: we train two models in tandem, starting from the same random initialization. The first model is provided with the full training set, whereas the second model has k = 100 data points removed from its training set. These points are drawn uniformly at random before the beginning of the training, and fixed thereafter. After each epoch, we evaluate the training accuracy, test accuracy and hypothesis stability, i.e., the agreement between the two models (which we calculate across the test set).
We set our main focus on the agreement of the models since the most amenable way to show loss stability might be by way of proving hypothesis stability. The latter can perhaps be mathematically proven in the case of neural networks by analyzing the stability of the training dynamics under two slightly different training sets.
this section cite: []

Section: L.2 Results
Across all experiments, the training and test accuracy of the model pairs are essentially identical throughout the training process. This suggests that at least simple models are loss stable across vision tasks. In order to reduce visual clutter, we hence only plot training and test accuracy of the first model (which has access to the full training set), respectively.
We observe higher agreement for simpler data sets and smaller learning rates. For example, the learning rate has a considerable effect on agreement for CIFAR10 (≈ 0.65 for learning rate 0.1 vs ≈ 0.8 for learning rate 0.01).
The key takeaway from Figures 1 through 4is that the agreement is consistently higher than the test accuracy. This relationship ensures that when applying the estimation procedure outlined in Theorem 4.3, we can avoid vacuous predictions of perfect accuracy. In the scenarios presented, the estimated accuracy will always be bounded away from 1, as it can be expressed as test error + (1 -agreement). For instance, with a learning rate of 0.01, the maximum estimated accuracies are: 98% for MNIST (compared to 97.5% test accuracy), 90% for FashionMNIST (87% test accuracy), 72% for CIFAR10 (52% test accuracy), and 65% for CIFAR10 with random labels (10% test accuracy). These results illustrate a strong correlation between stability estimation and data complexity.
We repeat the same experiments, modifying the width of the hidden layer to investigate its impact on stability. The results, summarized in Table 1, reveal a strong positive correlation between network width and stability. This effect is particularly pronounced for more complex tasks, such as CIFAR10 and CIFAR10 with random labels. For instance, in the CIFAR10 random labels setting with a learning rate of 0.01, increasing the width from 256 to 1024 neurons improves agreement from 32% to 50%, highlighting the stabilizing effect of greater network width. • While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: All the experimental details appear in the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
this section cite: []

Section: Answer: [Yes]
Justification: There are error bars in all our figures.
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
Answer: [No] .
Justification: Very simple experiments executed on a laptop, no special resources needed.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: This is a theory paper so there are no ethical concerns for potential harmful consequences.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] .
Justification: Purely theoretical work.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
this section cite: []

Section: 
Proof. Consider the following experiment:
1. Sample a sequence of points X = (X 1 , . . . , X m ) independently and uniformly at random from X .
2. Sample a function F uniformly from F , independently of X.
3. For each i ∈ [m], let Y i = F (X i ), let Y = (Y 1 , . . . , Y m ), and let S = (X 1 , Y 1 ), . . . , (X m , Y m ) .
Let P be the joint distribution of (X, F, Y, S). Fix k ∈ {2, . . . , |F |}, and let s = (x 1 , y 1 ), . . . , (x m , y m ) ∈ (X × {±1}) m with x = (x 1 , . . . , x m ) and y = (y 1 , . . . , y m ) such that |F s | = k. Denote F s = {f 1 , . . . , f k }.
Then for any i, j ∈ Because A is F-interpolating, A(s) ∈ F s . Without loss of generality, denote A(s) = f 1 . From F ∈ ⊥ ε,X and Proposition 2.9, L D f i (f j ) ≥ 1 2 -ε 2 := 2α for all i, j ∈ (16) Hence, for any η ∈ R,
P |L D F (A(S)) -η| ≥ α S = s ≥ 1 k . (17
)
We conclude that for any estimator E : (X × {±1}) m → R,
P(|L D F (A(S)) -E (S)| ≥ α) ≥ k∈{2,...,|F |} P |L D F (A(S)) -E (S)| ≥ α |F S | = k = k∈{2,...,|F |} s: |Fs|=k P |L D F (A(S)) -E (S)| ≥ α S = s • P(S = s) ≥ k∈{2,...,|F |} s: |Fs|=k inf η∈R P |L D F (A(S)) -η| ≥ α S = s • P(S = s) ≥ k∈{2,...,|F |} s: |Fs|=k 1 k • P(S = s) (By Eq. (17)) = k∈{2,...,|F |} 1 k • P(|F S | = k)
as desired.
MNIST FMNIST CIFAR10 CIFAR10 -RAND #N lr Agree #N lr Agree #N lr Agree #N lr Agree 256 0.1 99% 256 0.1 92% 256 0.1 62% 256 0.1 21% 256 0.01 99.5% 256 0.01 97% 256 0.01 71% 256 0.01 32% 512 0.1 99% 512 0.1 94% 512 0.1 67% 512 0.1 30% 512 0.01 99.5% 512 0.01 97% 512 0.01 80% 512 0.01 41% 1024 0.1 99% 1024 0.1 95% 1024 0.1 76% 1024 0.1 39% 1024 0.01 99.5% 1024 0.01 98% 1024 0.01 85% 1024 0.01 50% Table 1: Agreement percentages across datasets with varying number of neurons in the hidden layer (#N) and learning rates (lr). The setup is the same as in L.1 except for the number of training epochs, which is {50, 150, 150, 300} for {MNIST, FMNIST, CIFAR10, CIFAR10 random}, respectively. In scenarios where agreement has not yet reached saturation, agreement is positively correlated with the width of the network.
this section cite: []

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: This is a theory paper and we present all our results informally for better readability before the formal results in later sections.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [NA] Justification: no limitations.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally).
The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: All proofs are in the appendix Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: We run simple experiments and we explicitly mention all hyperparameters to reproduce our results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Justification: the experiments are elementary; nevertheless, if reviewers wish to see the code, we can provide it.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/  guides/CodeSubmissionPolicy) for more details.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] .
Justification: Purely theoretical work.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA] .
Justification: No external resources used.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] .
Justification: No new assets introduced.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] .
Justification: This is a theory paper with no crowdsourcing.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA] Justification: This is a theory paper with no crowdsourcing.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.
Answer: [NA] Justification: No use of LLMs as mentioned above.
Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: On the universality of deep learning Year: (2020)
Ref_id:b1 Title: Database-friendly random projections: Johnson-Lindenstrauss with binary coins Year: (2003)
Ref_id:b2 Title: Mean-field analysis of generalization errors Year: (2023)
Ref_id:b3 Title: Failures of model-dependent generalization bounds for least-norm interpolation Year: (2021)
Ref_id:b4 Title: Benign overfitting in linear regression Year: (2020)
Ref_id:b5 Title: Learners that use little information Year: (2018)
Ref_id:b6 Title: Properties of random matrices and applications. Unpublished report Year: (2006)
Ref_id:b7 Title: Stability and generalization Year: (2002)
Ref_id:b8 Title: Convex optimization Year: (2014)
Ref_id:b9 Title: A generalized neural tangent kernel analysis for two-layer neural networks Year: (2020-12-06)
Ref_id:b10 Title: In search of robust measures of generalization Year: (2020-12-06)
Ref_id:b11 Title: On the role of data in pac-bayes bounds. International Conference on Artificial Intelligence and Statistics Year: (2021)
Ref_id:b12 Title: Stability of randomized learning algorithms Year: (2005)
Ref_id:b13 Title: Generalization error bounds via Rényi-, f-divergences and maximal leakage Year: (2021)
Ref_id:b14 Title: Fantastic generalization measures are nowhere to be found Year: (2024)
Ref_id:b15 Title: Probability and random processes Year: (2020)
Ref_id:b16 Title: Understanding generalization via leave-one-out conditional mutual information Year: (2022)
Ref_id:b17 Title: Understanding generalization via leave-one-out conditional mutual information Year: (2022)
Ref_id:b18 Title: Information-theoretic generalization bounds for black-box learning algorithms Year: (2021)
Ref_id:b19 Title: A new family of generalization bounds using samplewise evaluated cmi Year: (2022)
Ref_id:b20 Title: Probability inequalities for sums of bounded random variables Year: (1963)
Ref_id:b21 Title: Strengthened information-theoretic bounds on the generalization error Year: (2019)
Ref_id:b22 Title: Generalization error bounds for noisy, iterative algorithms via maximal leakage Year: (2023)
Ref_id:b23 Title: Fantastic generalization measures and where to find them Year: (2020-04-26)
Ref_id:b24 Title: Extensions of Lipschitz mappings into a Hilbert space Year: (1984)
Ref_id:b25 Title: Algorithmic stability and sanity-check bounds for leaveone-out cross-validation Year: (1999)
Ref_id:b26 Title: Stability and generalization analysis of gradient methods for shallow neural networks Year: (2022)
Ref_id:b27 Title: In defense of uniform convergence: Generalization via derandomization with an application to interpolating predictors Year: (2019)
Ref_id:b28 Title: Beyond lipschitz: Sharp generalization and excess risk bounds for full-batch GD Year: (2023)
Ref_id:b29 Title: Two-layer neural network on infinite dimensional data: Global optimization guarantee in the mean-field regime Year: (2022-11-28)
Ref_id:b30 Title: Particle dual averaging: Optimization of mean field neural network with global convergence rate analysis Year: (2021-12-06)
Ref_id:b31 Title: On leave-one-out conditional mutual information for generalization Year: (2022)
Ref_id:b32 Title: A finite sample distribution-free performance bound for local discrimination rules Year: (1978)
Ref_id:b33 Title: Testing distributional assumptions of learning algorithms Year: (2023)
