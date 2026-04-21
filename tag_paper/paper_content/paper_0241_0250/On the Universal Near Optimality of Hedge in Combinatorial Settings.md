Title: On the Universal Near Optimality of Hedge in Combinatorial Settings
Abstract: In this paper, we study the classical HEDGE algorithm in combinatorial settings. In each round, the learner selects a vector x t from a set X ⊆ {0, 1} d , observes a full loss vector y t ∈ R d , and incurs a loss ⟨x t , y t ⟩ ∈ [-1, 1]. This setting captures several important problems, including extensive-form games, resource allocation, m-sets, online multitask learning, and shortest-path problems on directed acyclic graphs (DAGs). It is well known that HEDGE achieves a regret of O T log |X | after T rounds of interaction. In this paper, we ask whether HEDGE is optimal across all combinatorial settings. To that end, we show that for any X ⊆ {0, 1} d , HEDGE is near-optimal-specifically, up to a √ log d factor-by establishing a lower bound of Ω T log(|X |)/ log d that holds for any algorithm. We then identify a natural class of combinatorial sets-namely, m-sets with log d ≤ m ≤ √ d-for which this lower bound is tight, and for which HEDGE is provably suboptimal by a factor of exactly √ log d. At the same time, we show that HEDGE is optimal for online multitask learning, a generalization of the classical K-experts problem. Finally, we leverage the near-optimality of HEDGE to establish the existence of a near-optimal regularizer for online shortest-path problems in DAGs-a setting that subsumes a broad range of combinatorial domains. Specifically, we show that the classical Online Mirror Descent (OMD) algorithm, when instantiated with the dilated entropy regularizer, is iterate-equivalent to HEDGE, and therefore inherits its near-optimal regret guarantees for DAGs.

Section: Introduction
Prediction with expert advice is a central problem in online learning [31,13,11,14,2]. In this problem, a learner selects a probability distribution over a set of experts {1, 2, . . . , K} in each round. After making the choice, the learner observes the losses of all experts, which may be assigned adversarially within [-1, 1]. The goal is to minimize the cumulative regret, defined as the difference between the learner's expected total loss and the loss of the best expert in hindsight after T rounds. A simple and widely used algorithm for this setting is HEDGE, introduced by Freund and Schapire [24], which guarantees a regret bound of O( √ T log K).
Since its introduction, the HEDGE algorithm has been extended to a variety of settings, including adversarial bandits [3], continuous action spaces [30], stochastic regimes [34], discounted losses [23], and adaptive learning rates [18]. An important special case is combinatorial settings, where the learner selects a vector x t from a combinatorial set X ⊆ {0, 1} d in each round, observes a loss vector y t , and incurs a loss of ⟨x t , y t ⟩ ∈ [-1, 1]. The objective remains to minimize regret against the best fixed vector x * ∈ X in hindsight.
The combinatorial setting captures a wide variety of problems, including extensive-form games, resource allocation games (e.g., Colonel Blotto problems), online multitask learning problem, {0, 1} d hypercube, perfect matchings, spanning trees, cut sets, m-sets, and online shortest paths in directed acyclic graphs (DAGs). These problems have wide-ranging applications. For instance, extensive-form games provide a foundational framework for modeling sequential games with imperfect information and have been used to build human-level and even superhuman-level AI agents for real-world games [33,[7][8][9]. Online shortest path problems in DAGs arise naturally in applications like network routing [4,17]. Resource allocation games have been widely studied in the context of military strategy, political campaigns, sports, and advertising [6,1].
Given their broad relevance, these combinatorial problems have been extensively studied through the lens of online learning [44,28,29,12,16,36,45]. In the full-information setting, where the learner observes the entire loss vector, an important class of extensive-form games admits optimal algorithms that are efficiently implementable, with HEDGE shown to satisfy both properties in this context [5,19]. Another example is the work of Takimoto and Warmuth [44], which provided an efficient implementation of a variant of HEDGE for online shortest path problems in DAGs. In the bandit setting, refined variants of HEDGE achieve minimax-optimal regret [12,10], though a recent work shows they can still be significantly suboptimal for certain combinatorial families [32].
In this paper, we return to the full-information setting and revisit a natural approach for addressing combinatorial problems: treating each element x ∈ X as an expert and directly applying the HEDGE algorithm. This yields a regret bound of O T log |X | . While this naive approach is often computationally intractable due to the size of X , Farina et al. [20] (building on prior ideas by Takimoto and Warmuth [44]) recently showed that HEDGE and its optimistic variants can be implemented efficiently in a variety of important combinatorial settings that admit an efficient kernel. Given HEDGE's fundamental advantages-both its simplicity and its broad applicability-we are motivated to ask the following natural question:
Does HEDGE achieve optimal regret guarantees for every combinatorial set X ⊆ {0, 1} d ?
this section cite: ['b30', 'b12', 'b10', 'b13', 'b1', 'b23', 'b2', 'b29', 'b33', 'b22', 'b17', 'b32', 'b6', 'b7', 'b8', 'b3', 'b16', 'b5', 'b0', 'b43', 'b27', 'b28', 'b11', 'b15', 'b35', 'b44', 'b4', 'b18', 'b43', 'b11', 'b9', 'b31', 'b19', 'b43']

Section: Our Contributions
In this paper, we address the above question by establishing the following results:
O p t i m a l R e g r e t HEDGE Upper Bound (Folklore) General Lower Bound (Theorem 3.2) m-sets (Section 4) Multitask Learning (Section 5) Online Shortest Path (Section 6) Theorem 4.4 Theorem 4.6 Suboptimality of HEDGE Theorems 5.1, 6.1 HEDGE can be optimal T • log |X | log d T log |X |
this section cite: []

Section: Optimal regret range
Figure 1: An overview of our results. The x-axis indexes different combinatorial decision sets X ⊆ {0, 1} d , and the y-axis shows the optimal regret over T rounds. We show that HEDGE is near-optimal for all X , up to a √ log d factor. For m-sets, HEDGE is provably suboptimal by a factor of √ log d, whereas in structured settings such as online multitask learning and important families of DAGs, it is in fact optimal.
• Universal Near-Optimality: In Theorem 3.2, we show that HEDGE is universally near optimal for combinatorial games by proving that the regret lower bound for any algorithm on any combinatorial set X ⊆ {0, 1} d is Ω max{ T log |X |/ log d, √ T } .
• Suboptimality in Specific Cases: In Theorems 4.4 and 4.6, we show that the lower bound from Theorem 3.2 is tight for a natural class of combinatorial sets X , and that HEDGE is provably suboptimal for these sets. Specifically, for m-sets with log d ≤ m ≤ √ d, we prove that HEDGE necessarily incurs a regret of Ω T log |X | , while we design an Online Mirror Descent (OMD) algorithm that achieves an optimal regret of O T log |X |/ log d .
• Optimality in Specific Cases: In Theorem 5.1, we show that HEDGE is optimal for a specific class of combinatorial sets X . In particular, for online multitask learning-which generalizes the classical K-experts problem-we prove that any algorithm must incur a regret of Ω T log |X | .
Beyond these results, we further investigate the optimality of HEDGE and its connection to regularization-based algorithms in the structured setting of directed acyclic graphs (DAGs), which capture a broad class of combinatorial problems-including extensive-form games, resource allocation game, m-sets, multitask learning, and the {0, 1} d hypercube:
• In Theorem 6.1, we show that HEDGE is minimax optimal when X corresponds to the set of paths from source to sink in a DAG.
• In Section 6.1, we show that OMD with the dilated entropy regularizer achieves a minimaxoptimal regret of O T log |X | for DAGs.
• In Theorem 6.5, we show that OMD with the dilated entropy regularizer is iterate-equivalent to HEDGE on DAGs-thereby inheriting the regret guarantees of HEDGE and demonstrating that HEDGE can be implemented efficiently on DAGs via OMD.
this section cite: []

Section: Related Works
Prediction with Expert Advice. One of the earliest works on online prediction was by Littlestone and Warmuth [31], who introduced the Weighted Majority algorithm. Cesa-Bianchi et al. [13] extended this line of work by studying the setting where experts' predictions lie in the interval [0, 1], while the outcomes are binary. Subsequently, Freund and Schapire [24] addressed the more general setting where both predictions and outcomes lie in [0, 1]. They proposed the HEDGE algorithm and established a regret bound of O( √ T log K), where T is the time horizon and K is the number of experts.
This foundational result gave rise to several important subsequent works. Erven et al. [18] introduced AdaHedge, a variant of HEDGE with adaptive learning rates that achieves a regret of roughly L * T log K, where L * T is the cumulative loss of the best expert. Krichene et al. [30] studied a continuous version of HEDGE for online optimization over compact convex sets S ⊂ R d . In the stochastic setting, Mourtada and Gaïffas [34] analyzed HEDGE with decreasing learning rates and obtained a regret bound of O(log N/∆), where ∆ denotes the sub-optimality gap. For the bandit feedback setting, Auer et al. [3] developed EXP3, a bandit-feedback variant of HEDGE that achieves a regret of O( √ KT ).
this section cite: ['b30', 'b12', 'b23', 'b17', 'b29', 'b33', 'b2']

Section: Combinatorial Settings.
Online learning in combinatorial games has recently received considerable attention. Farina et al. [20] showed that HEDGE and its optimistic variants [15,37] can be implemented efficiently when the combinatorial game admits an efficient kernel. Examples of such problems include extensive-form games, resource allocation games, m-sets, and, more generally, online shortest path problems in directed acyclic graphs (DAGs). Hoda et al. [27] introduced the dilated entropy regularizer for extensive-form games and analyzed its properties (cf. also Farina et al. [21]). Building on this, Bai et al. [5] demonstrated that Online Mirror Descent (OMD) with the a specific variant of the dilated entropy regularizer is iterate-equivalent to HEDGE in extensive-form games. Fan et al. [19] subsequently provided the first OMD-based regret analysis for this regularizer, matching the known bounds for HEDGE. Their analysis introduced a new norm, called the treeplex norm, to facilitate the regret bounds.
In the bandit feedback setting, Cesa-Bianchi and Lugosi [12] analyzed online learning over specific combinatorial sets X ⊆ {0, 1} d , and proposed a variant of HEDGE called COMBAND, achieving an expected regret bound of O dT log |X | for those sets. Bubeck et al. [10] subsequently extended this result to any combinatorial set X ⊆ {0, 1} d , using a variant of HEDGE known as EXP2 with John's exploration. Recently, Maiti et al. [32] showed that this bound is sub-optimal by a factor of d 1/4 for a specific family of directed acyclic graphs, thereby demonstrating that these variants of HEDGE can, in fact, be substantially sub-optimal.
While the above prior works-as well as our own-focus on loss vectors y t such that ⟨x, y t ⟩ ∈ [-1, 1] for all x ∈ X , other works [44,29,36] consider coordinate-wise bounded losses, where each y t [i] ∈ [0, 1] for all i ∈ [d]. Under coordinate-wise bounded losses, Takimoto and Warmuth [44] were the first to implement a variant of HEDGE efficiently on DAGs by leveraging the additivity of losses across the edges of a path. Koolen et al. [29] subsequently analyzed the COMPONENT HEDGE algorithm over various combinatorial sets, including m-sets and DAGs. Rahmanian and Warmuth [36] further extended COMPONENT HEDGE to the k-multipaths problem.
Rademacher Complexity. Orabona and Pál [35] provided a non-asymptotic lower bound of Ω( √ T log N ) for the experts problem by analyzing the supremum of a sum of Rademacher random variables. A series of works [39,38,22] extended this analysis to more general online learning problems-including combinatorial settings-by characterizing regret in terms of sequential Rademacher complexity. Srebro et al. [42] showed that there always exists an instance of Follow-the-Regularized-Leader (FTRL) that is nearly optimal for online linear optimization. This was recently strengthened by Gatmiry et al. [25], who showed that an optimal FTRL instance always exists for online linear optimization. However, the construction of an optimal regularizer for FTRL may incur significant computational overhead.
this section cite: ['b19', 'b14', 'b36', 'b26', 'b20', 'b4', 'b18', 'b11', 'b9', 'b31', 'b43', 'b28', 'b35', 'b43', 'b28', 'b35', 'b34', 'b38', 'b37', 'b21', 'b41', 'b24']

Section: Preliminaries
In this paper, we study the repeated online decision-making problem in a combinatorial setting. Denote by d the dimension of the problem. The agent is given a set of discrete actions X ⊆ {0, 1} d . At each round t = 1, 2, . . . , the agent selects an action x t from the decision set X . The environment simultaneously chooses a loss vector y t ∈ Y, potentially adversarially based on the interaction history F t := {(x τ , y τ )} t-1 τ =1 . The agent then incurs a loss of ⟨x t , y t ⟩ and observes the loss vector y t . The goal of the agent is to minimize the total loss over T rounds, or equivalently, to minimize the cumulative regret:
Regret(T ) := T t=1 ⟨x t , y t ⟩ -min x * ∈X T t=1 ⟨x * , y t ⟩.
We focus on the combinatorial game setting, in which the loss vector is restricted so that the loss incurred at each round is bounded within [-1, 1]. In this case, the loss vector set Y is the polar set of the convex hull co(X ). We note that this assumption covers various settings in the literature, including extensive-form games [20,5]. Formally, we make the following assumption: Assumption 2.1. The loss vector set is defined as Y := y ∈ R d : max x∈X |⟨x, y⟩| ≤ 1 .
The Hedge Algorithm A classical approach for solving this problem is the HEDGE algorithm [24], also known as Multiplicative Weight Updates (MWU). In this algorithm, the agent chooses actions in a randomized manner based on the past performance of the actions. Specifically, let η > 0 be the learning rate, the probability of choosing an action x in round t is proportional to
P t (x) ∝ w t (x) := exp -η t-1 τ =1 ⟨x, y τ ⟩ , ∀x ∈ X .
A classical result shows under learning rate η := log |X |/T , this algorithm has a regret upper bound of O T log |X | in the combinatorial game setting (i.e., under Assumption 2.1). Furthermore, the algorithm requires O(d|X |) time per round to compute the updates. This may be exponential in d when only a succinct representation of the decision set X is provided. It is known that when the kernel of the decision set X can be computed efficiently, it is possible to simulate the MWU algorithm in polynomial time using the KERNELIZED MWU algorithm [20].
this section cite: ['b19', 'b4', 'b23', 'b19']

Section: Proximal Methods
We recall the foundational concepts and notations commonly used in proximal optimization methods. Let X denote the decision set. The proximal methods is built upon on some regularizer φ : co(X ) → R, which is required to be µ-strongly convex with respect to a chosen norm ∥ • ∥ over co(X ). Such a function naturally gives rise to a generalized measure of divergence known as the Bregman divergence, defined for any vectors x ′ , x ∈ X as:
D φ (x ′ ∥ x) := φ(x ′ ) -φ(x) -⟨∇φ(x), x ′ -x⟩.
Among proximal methods, a general approach to solving the online decision problem is the Online Mirror Descent (OMD) algorithm [15]. Let η > 0 be the learning rate, and let φ : co(X ) → R be a strongly convex regularizer. The algorithm maintains a policy in each round t. In the first round, the policy is set so the regularizer is unique minimizer:
x 1 ← argmin x∈co(X ) φ(x).
For each round t = 2, 3, . . . , the agent takes proximal step:
x t ← argmin x∈co(X ) η ⟨y t-1 , x⟩ + D φ (x ∥ x t-1 )
Then, the agent draws and plays an action x t ∈ X by matching its expectation to the proposed policy, i.e., E t [x t ] = x t . It is known that this algorithm achieves the following regret upper bound: Theorem 2.2 (Regret Bound for OMD, [37,43]).
Let ∥ • ∥ and ∥ • ∥ * be a pair of primal-dual norm defined on R d . Let φ be a DGF that is µ-strongly convex on ∥ • ∥. Denote y t as the reward gradient received in episode t. The cumulative regret of running OMD with DGF φ and learning rate η > 0 can be upper bounded by Regret(T ) := T t=1 ⟨ x t , y t ⟩ -min x * ∈X T t=1 ⟨x * , y t ⟩ ≤ 1 η D φ (x * ∥ x 1 ) + η 2µ T t=1 ∥y t ∥ 2 * .
this section cite: ['b14', 'b36', 'b42']

Section: General Notation
We use lowercase boldface letters, such as x, to denote vectors. The notation |x| denotes the element-wise absolute value. For an index set C, let x[C] ∈ R C denote the subvector of x restricted to entries indexed by C, and let |C| denote the cardinality of the set. We write k := {1, 2, . . . , k} and use ∅ to denote the empty set. The probability simplex over a finite set C is denoted by P(C). We use log x to denote the logarithm of x in base 2 and ln x to denote the natural logarithm of x. For non-negative sequences {a n } and {b n }, we write a n ≤ O(b n ) or equivalently b n ≥ Ω(a n ) to indicate the existence of a global constant C > 0 such that a n ≤ Cb n for all n > 0.
Similarly, we write a n = Θ(b n ) to indicate the existence of global constants C 1 , C 2 > 0 such that C 1 b n ≤ a n ≤ C 2 b n for all n > 0. Lastly, we denote by co(X ) the convex hull of a set X .
this section cite: []

Section: Universal Near Optimality of Hedge
In this section, we show that for any given combinatorial decision set X ⊆ {0, 1} d , the HEDGE algorithm achieves a near optimal regret bound in the combinatorial game setting. We begin by stating the following classical result. Lemma 3.1 (Sauer-Shelah Lemma [40,41]). Let C be a family of sets whose union has n elements. A set S is said to be shattered by C if every subset of S can be obtained as the intersection S ∩ C for some set C ∈ C. C shatters a set of size k, if the number of sets in the family satisfies
|C| > k-1 i=0 n i .
By the Sauer-Shelah Lemma, there exists an index set I ⊆ d of size Ω(log |X |/ log d) such that the restriction of X to the coordinates in I equals the full hypercube {0, 1} I . Consequently, any hard instance with loss vectors supported only on coordinates in I is at least as hard as the corresponding instance of the combinatorial game over the hypercube X ′ := {0, 1} I . As any algorithm suffers a regret lower bound of Ω T |I| in the combinatorial game over the hypercube. This yields a regret lower bound of Ω T log |X |/ log d .
We formally state our main result in the following theorem. We defer the proof to Appendix A.
Theorem 3.2. Let X ⊆ {0, 1} d be a decision set and let Y be the corresponding loss vector set that satisfies Assumption 2.1. For any T ≥ 1 and any Algorithm ALG, there exists a sequence of loss vectors y 1 , y 2 , . . . , y T ∈ Y such that the algorithm incurs an expected regret of at least E[Regret(T )] = E T t=1 ⟨x t , y t ⟩ -min x * ∈X T t=1 ⟨x * , y t ⟩ ≥ Ω max T • log |X | log d , √ T .
The expectation is taken over any potential randomness of the algorithm.
this section cite: ['b39', 'b40']

Section: The Sub-Optimality of Hedge on m-Sets
In this section, we consider a specific decision set X -namely, the family of m-sets-to illustrate the suboptimality of the HEDGE algorithm. For an integer m ∈ d/2 , the m-sets problem corresponds to the decision set X := x ∈ {0, 1} d : d i=1 x[i] = m . We show that OMD, with a suitable regularizer, matches the regret lower bound from Theorem 3.2 when log d ≤ m ≤ d/2, whereas HEDGE suffers a regret lower bound of Ω( T log |X |), establishing a suboptimality gap of √ log d.
this section cite: []

Section: The Regret Upper Bound of m-Sets
We begin by presenting our OMD algorithm for m-sets, for any m ∈ d/2 . Previous work [42] shows that there always exists a regularizer that enables the OMD algorithm to achieve a near-optimal regret bound. However, the regularizer is not constructive, and the corresponding regret bound is implicit. Instead, we need to construct a regularizer suitable for the decision set so that the regret bound in Theorem
2.2 is minimized. Specifically, we analyze the OMD algorithm with the following regularizer: φ(x) := d i=1 x[i] 2 + 1 m x[i] ln x[i] . (4.1) According to Theorem 2.2, it suffices to pick a pair of primal-dual norm ∥ • ∥ and ∥ • ∥ * and analyze the strong convexity of φ and also the vector norm ∥y t ∥ * . We define a pair of dual-primal norms: ∥z∥ * := max x∈X |⟨x, z⟩|, ∥z∥ := max ∥y∥ * ≤1
⟨y, z⟩.
First, we state the following proposition that establishes an upper bound on the primal norm ∥z∥.
The proof is done by direct calculation, and we defer the details to Appendix B.
Proposition 4.1. For any vector z ∈ R d , the primal norm ∥ • ∥ is upper bounded by the ℓ 1 and ℓ ∞ norms together, namely,
∥z∥ ≤ 3∥z∥ ∞ + 1 m ∥z∥ 1 .
By applying the above proposition, we establish the strong convexity of the regularizer φ with respect to the primal norm. We defer the proof to Appendix B.
Lemma 4.2. The function φ is 1/9-strongly convex with respect to the primal norm ∥ • ∥.
The following lemma bounds the Bregman divergence under the regularizer φ.
Lemma 4.3. We have D φ (x * ∥ x 1 ) ≤ m + ln(d/m), for any vector x * ∈ X .
We defer the proof to Appendix B. Using the above results, we are able to establish the regret upper bound for running OMD with the regularizer defined in (4.1).
Theorem 4.4. Let 1 ≤ m ≤ d/2. With the choice η := 2(m + ln(d/m))/(9T ), the expected regret of running OMD with the regularizer in (4.1) over m-sets is upper bounded by E[Regret(T )] ≤ O T m + T log(d/m) . Proof. According to the definition of ∥ • ∥ * , we have that ∥y t ∥ * ≤ 1. Combining this with Lemma 4.2 and Lemma 4.3, and applying Theorem 2.2 together with E[x t ] = x t , we conclude that E[Regret(T )] ≤ 1 η D φ (x * ∥ x 1 ) + η 2µ T t=1 ∥y t ∥ 2 * ≤ 1 η m + ln(d/m) + 9η 2 • T ≤ 18T m + 18T ln(d/m),
where the last inequality is given by the choice η.
We show that this regret upper bound is in fact optimal by establishing a matching lower bound. Specifically, we construct our lower bound using the hard instance from Theorem 3.2 along with the hard instance for the K-experts problem (see Lemma F.4). Formally, we show the following:
Theorem 4.5. Consider integers m, d, T such that 1 ≤ m ≤ d/2 ≤ exp(T /3)/2. For the m-sets problem and any Algorithm ALG, there exists a sequence of loss vectors y 1 , y 2 , . . . , y T ∈ Y such that the algorithm incurs a expected regret of at least
E[Regret(T )] = E T t=1 ⟨x t , y t ⟩ -min x * ∈X T t=1 ⟨x * , y t ⟩ ≥ Ω T m + T log(d/m) .
We defer the proof to Appendix B.
this section cite: ['b41']

Section: The Regret Lower Bound of Hedge
We introduce the following theorem, showing that the HEDGE algorithm is strictly sub-optimal on m-sets, when log d ≤ m ≤ √ d. The full proof is deferred to Appendix B.
Theorem 4.6. Consider integers m, d, T such that 1 ≤ m ≤ d/2 ≤ exp(T /3)/2 and m log(d/m) ≤ T . For any η > 0, there exists a there exists a sequence of loss vectors y 1 , y 2 , . . . , y T ∈ Y over m-sets such that the HEDGE algorithm with learning rate η incurs a expected regret of at least E[Regret(T )] = E T t=1 ⟨x t , y t ⟩ -min x * ∈X T t=1 ⟨x * , y t ⟩ ≥ Ω T m log(d/m) .
Proof sketch. We divide the proof into two cases based on how η compares to the base learning rate
η 0 := T -1 m ln(d/m) = Θ T -1 log |X | .
When the learning rate is small, i.e., η ≤ η 0 , we construct a hard instance by assigning the same fixed loss vector y t across all rounds, where y t [i] := 1[i ∈ m ]/m for all i ∈ d . In this case, we can show that HEDGE with a small learning rate incurs a constant regret for any round t ≤ t 0 := Ω T m log(d/m) . Thus, we establish a regret lower bound of Ω T m log(d/m) .
When the learning rate is large, i.e., η > η 0 , we construct a hard instance by setting y t [i] = 0 for all coordinates i ≥ 2 across all rounds. For the first coordinate, the loss y t [1] is assigned in two phases, based on the threshold t 0 := ln(d/m)/η (assuming t 0 ∈ N for simplicity). In Phase 1 (rounds t ≤ t 0 ), we set y t [1] = -1. In Phase 2 (rounds t > t 0 ), the value of y t [1] alternates: it is 1 if t -t 0 is odd, and -1 if t -t 0 is even.
In this case, we can show that HEDGE with a large learning rate incurs a regret of Ω min{η, 1} for every two rounds after t > t 0 . Thus, we establish a regret lower bound via
E[Regret(T )] ≥ (T -t 0 ) • Ω min{η, 1} ≥ Ω T m log(d/m) .
In general, we conclude that HEDGE has a regret lower bound of Ω T m log(d/m) on the m-sets for any learning rate η > 0.
this section cite: ['b0', 'b0']

Section: The Optimality of Hedge for Online Multitask Learning
In the previous section, we showed that the HEDGE algorithm can be strictly suboptimal for certain combinatorial decision sets, such as m-sets. This naturally leads to the question: are there combinatorial settings where HEDGE remains optimal? In this section, we answer this in the affirmative by analyzing the Online Multitask Learning problem-a setting that generalizes the classical K-experts problem and has been studied in the bandit learning literature as the Multi-Task Bandit problem. In this problem, the learner is presented with m ≥ 1 separate expert problems, where the i-th problem involves d i ≥ 2 experts. In each round, the learner selects one expert from each problem and incurs a loss that is the sum of the losses associated with the chosen experts. The goal is to minimize regret with respect to the best expert in each problem in hindsight.
We parameterize the online multitask learning problem as follows: Let d 1:i := i j=1 d j be the total number of experts in the first i expert problems, with d 1:0 := 0. The decision set X is of dimension d = d 1:m given by X = x ∈ {0, 1} d :
d1:i j=d1:i-1+1 x[j] = 1, ∀i ∈ m .
Recall that the adversary is restricted to choose y t such that ⟨x, y t ⟩ ∈ [-1, 1] for all x ∈ X in each round t following from Assumption 2.1. In this case, HEDGE has a regret upper bound of O T log |X | . In the following theorem we show that HEDGE is optimal for the online multitask Learning problem. We construct a hard instance by using the hard instance for the i-th expert problem over
this section cite: []

Section: Minimax Optimal Regularizers for Directed Acyclic Graphs
We consider the online shortest path problem in the Directed Acyclic Graphs (DAGs). Let G = (V, E) be a DAG with the source vertex s ∈ V and the sink t ∈ V . We assume that every vertex v ∈ V is reachable from s and can reach t. Denote by X ⊆ {0, 1} E the set of all s-t paths of the graph G, indexed by the edges in E. Each vertex x ∈ X encodes a s-t path in the graph, where x[e] = 1 indicates that e ∈ E appears in the path. The convex hull of X forms the flow polytope:
co(X ) = x ∈ [0, 1] E : e∈δ + (s) x[e] = e∈δ -(t)
x[e] = 1 and
e∈δ -(v) x[e] = e∈δ + (v)
x[e], ∀v ∈ V , where δ -(v) = {(u, v) ∈ E} and δ + (v) = {(v, w) ∈ E} denotes the set of incoming edges and outgoing edges, respectively. We note that in this case, the loss vector y ∈ Y ⊆ R E is an assignment of the weights such that any s-t path has a weight between -1 and 1.
In the following theorem, we show that HEDGE is minimax optimal for DAGs. The proof involves a careful construction of a DAG, parameterized by upper bounds on the number of edges and paths. The full proof is deferred to Appendix D. Theorem 6.1. For any integers d, N, T such that 16 ≤ 2d ≤ N ≤ 2 d and 3 log N ≤ T , and for any algorithm ALG, there exists a DAG G with at most d edges and at most N paths from source s to sink t, and a corresponding sequence of loss vectors y 1 , y 2 , . . . , y T ∈ Y such that the algorithm incurs a regret lower bound of
E[Regret(T )] = E T t=1 ⟨x t , y t ⟩ -min x * ∈X T t=1
⟨x * , y t ⟩ ≥ Ω T log N .
this section cite: []

Section: OMD with Dilated Entropy Regularizer
While our main focus has been on the HEDGE algorithm, it is natural to consider its close counterpart, Online Mirror Descent (OMD). With an appropriate distance-generating function, OMD is known to be iterate-equivalent to HEDGE in extensive-form games [5,19]. Since DAGs can model such games [32], and HEDGE is minimax-optimal on DAGs, this motivates a closer examination of OMD on DAGs. In this section, we analyze OMD with the dilated entropy regularizer on DAGs and show that it also achieves minimax-optimal regret. Formally, the dilated entropy ψ : co(X ) → R is defined by
ψ(x) := e∈E x[e] ln x[e] - v∈V x[v] ln x[v] = v∈V \{t}:x[v]>0 e∈δ + (v) x[e] ln x[e] x[v] ,
where, by standard convention, 0 ln 0 := 0. Here, x[v] := e∈δ + (v) x[e] for all v ̸ = t, and x[t] := 1.
We note that the regularizer on the policy x ∈ co(X ) is closely related to the Shannon entropy over the chosen action x ∈ X . In fact, consider the following procedure for sampling an action x ∼ D( x) ∈ P(X ): we start from the active vertex u ← s. At each step, we first set x[u] = 1, then randomly pick an edge e = (u, v) ∈ δ + (u) with probability x[e]/ x[u], set x[e] = 1, and move to u ← v. Following this Markovian sampling procedure, one can see that the drawn vector x is consistent with x, i.e., E[x] = x. Furthermore, we have the following: Lemma 6.2. For any x ∈ co(X ), we have
ψ( x) = -H(x) := E x∼D( x) [ln P(x)],
where H(•) denotes the Shannon entropy of the random variable.
The proof of the above lemma is deferred to Appendix D. We will now show that running OMD with the dilated regularizer enjoys a regret upper bound of O T log |X | , based on the OMD regret bound in Theorem 2.2. From Lemma 6.2, we have that ψ is equivalent to the negative entropy of distribution over X . This indicates D ψ (x * ∥ x 1 ) ≤ ln |X |. It remains to pick the norm functions and show the strong convexity. Consider a pair of primal dual norms
∥z∥ * := max x∈X |⟨x, z⟩|, ∥z∥ := max ∥y∥ * ≤1
⟨y, z⟩.
We note that since co(X ) is the flow polytope, its dual, {y : ∥y∥ * ≤ 1}, is closely related to the set of all cuts of the graph. The next lemma shows ψ is strongly convex over primal norm ∥ • ∥. We defer the proof to Appendix D. Lemma 6.3. The function ψ is 1/10-strongly convex with respect to the primal norm ∥•∥ in span(X ).
From the standard OMD analysis, running OMD under the dilated entropy ψ achieves a regret upper bound of O T log |X | . Hence, OMD with dilated entropy is minimax optimal for DAGs.
this section cite: ['b4', 'b18', 'b31']

Section: Equivalence of Dilated Entropy and HEDGE
As shown earlier, both OMD with the dilated entropy regularizer and HEDGE over the set of paths in a DAG achieve minimax optimal regret. This naturally raises a fundamental question: are these two approaches equivalent? In this section, we answer this question in the affirmative.
Let G = (V, E) be a directed acyclic graph (DAG) with a designated source vertex s and sink vertex t, and let X denote the set of all paths from s to t. If G contains only a single source-to-sink path, the equivalence is immediate. Therefore, we focus on the case where G admits multiple such paths.
We begin by state the following lemma, the proof of which is deferred to Appendix D. Lemma 6.4. The dilated entropy ψ is differentiable and strictly convex on the relative interior C := relint(co(X )). Moreover,
lim n→∞ ∥∇ x ψ(x n )∥ 2 = ∞ if {x n } n is sequence of points in C
approaching the boundary of C.
Now, we present the following theorem, which demonstrate that OMD is, in fact, iterate-equivalent to HEDGE. The proof proceeds by formulating the KKT conditions and applying Lemma 6.4 to establish the equivalence. The proof is deferred to Appendix D. Theorem 6.5. OMD with dilated entropy is iterate-equivalent to HEDGE over the set of paths X .
this section cite: []

Section: Conclusion and Future works
We investigated the optimality of the classical HEDGE algorithm in combinatorial online learning settings. While HEDGE achieves a regret of O T log |X | , we established that this rate is nearly optimal-up to a √ log d factor-for any set X ⊆ {0, 1} d . We further identified a class of m-sets for which HEDGE is provably suboptimal and showed that it remains optimal for the multitask learning problem. Finally, we demonstrated that Online Mirror Descent with the dilated entropy regularizer is iterate-equivalent to HEDGE on DAGs, providing a computationally efficient regularization framework for a broad family of combinatorial domains.
Our work opens up several interesting directions for future research. One natural question is whether there exists a family of efficiently constructible regularizers that are near-optimal for the combinatorial sets. We conjecture that negative entropy in a suitably lifted space may serve as such a regularizer. In support of this, we refer the reader to Appendix E, where we show that the conjecture holds for DAGs. Another compelling direction is to explore whether there exist near-optimal variants of the Followthe-Perturbed-Leader algorithm for the combinatorial sets. Since perturbations are often considered more implementation-friendly, exploring near-optimal variants of the Follow-the-Perturbed-Leader algorithm could yield both theoretical and practical advances. Finally, we ask whether there are variants of HEDGE that achieve near-optimal regret for arbitrary finite subsets of R d .
this section cite: []

Section: References
Ref_id:b0 Title: From duels to battlefields: Computing equilibria of blotto and other games Year: (2019)
Ref_id:b1 Title: The multiplicative weights update method: a meta-algorithm and applications Year: (2012)
Ref_id:b2 Title: The nonstochastic multiarmed bandit problem Year: (2002)
Ref_id:b3 Title: Adaptive routing with end-to-end feedback: Distributed learning and geometric approaches Year: (2004)
Ref_id:b4 Title: Efficient phi-regret minimization in extensive-form games via online mirror descent Year: (2022)
Ref_id:b5 Title: Fast and simple solutions of blotto games Year: (2023)
Ref_id:b6 Title: Heads-up limit hold'em poker is solved Year: (2015)
Ref_id:b7 Title: Superhuman ai for heads-up no-limit poker: Libratus beats top professionals Year: (2018)
Ref_id:b8 Title: Superhuman ai for multiplayer poker Year: (2019)
Ref_id:b9 Title: Towards minimax policies for online linear optimization with bandit feedback Year: (2012)
Ref_id:b10 Title: Prediction, learning, and games Year: (2006)
Ref_id:b11 Title:  Year: (2012)
Ref_id:b12 Title: How to use expert advice Year: (1997)
Ref_id:b13 Title: Improved second-order bounds for prediction with expert advice Year: (2007)
Ref_id:b14 Title: Online optimization with gradual variations Year: (2012)
Ref_id:b15 Title: Following the perturbed leader for online structured learning Year: (2015)
Ref_id:b16 Title: Tight bounds for bandit combinatorial optimization Year: (2017)
Ref_id:b17 Title: Adaptive hedge Year: (2011)
Ref_id:b18 Title: On the optimality of dilated entropy and lower bounds for online learning in extensive-form games Year: ()
Ref_id:b19 Title: Kernelized multiplicative weights for 0/1-polyhedral games: Bridging the gap between learning in extensive-form and normal-form games Year: (2022)
Ref_id:b20 Title: Better regularization for sequential decision spaces: Fast convergence rates for nash, correlated, and team equilibria Year: (2025)
Ref_id:b21 Title: Adaptive online learning Year: (2015)
Ref_id:b22 Title: A new hedging algorithm and its application to inferring latent random variables Year: (2008)
Ref_id:b23 Title: A decision-theoretic generalization of on-line learning and an application to boosting Year: (1997)
Ref_id:b24 Title: Computing optimal regularizers for online linear optimization Year: (2024)
Ref_id:b25 Title: The best constants in the khintchine inequality Year: (1981)
Ref_id:b26 Title: Smoothing techniques for computing nash equilibria of sequential games Year: (2010)
Ref_id:b27 Title: Efficient algorithms for online decision problems Year: (2005)
Ref_id:b28 Title: Hedging structured concepts Year: (2010)
Ref_id:b29 Title: The hedge algorithm on a continuum Year: (2015)
Ref_id:b30 Title: The weighted majority algorithm. Information and computation Year: (1994)
Ref_id:b31 Title: Efficient near-optimal algorithm for online shortest paths in directed acyclic graphs with bandit feedback against adaptive adversaries Year: (2025)
Ref_id:b32 Title: Deepstack: Expert-level artificial intelligence in heads-up no-limit poker Year: (2017)
Ref_id:b33 Title: On the optimality of the hedge algorithm in the stochastic regime Year: (2019)
Ref_id:b34 Title: Optimal non-asymptotic lower bound on the minimax regret of learning with expert advice Year: (2015)
Ref_id:b35 Title: Online dynamic programming Year: (2017)
Ref_id:b36 Title: Online learning with predictable sequences Year: (2013)
Ref_id:b37 Title: Relax and localize: From value to algorithms Year: (2012)
Ref_id:b38 Title: Online learning via sequential complexities Year: (2015)
Ref_id:b39 Title: On the density of families of sets Year: (1972)
Ref_id:b40 Title: A combinatorial problem; stability and order for models and theories in infinitary languages Year: (1972)
Ref_id:b41 Title: On the universality of online mirror descent Year: (2011)
Ref_id:b42 Title: Fast convergence of regularized learning in games Year: (2015)
Ref_id:b43 Title: Path kernels and multiplicative updates Year: (2003-10)
Ref_id:b44 Title: Combinatorial bandits for sequential learning in colonel blotto games Year: (2019)
