Title: The Complexity of Symmetric Equilibria in Min-Max Optimization and Team Zero-Sum Games *
Abstract: We consider the problem of computing stationary points in min-max optimization, with a focus on the special case of Nash equilibria in (two-)team zero-sum games. We first show that computing ϵ-Nash equilibria in 3-player adversarial team games-wherein a team of 2 players competes against a single adversaryis CLS-complete, resolving the complexity of Nash equilibria in such settings. Our proof proceeds by reducing from symmetric ϵ-Nash equilibria in symmetric, identical-payoff, two-player games, by suitably leveraging the adversarial player so as to enforce symmetry-without disturbing the structure of the game. In particular, the class of instances we construct comprises solely polymatrix games, thereby also settling a question left open by Hollender, Maystre, and Nagarajan (2024). Moreover, we establish that computing symmetric (first-order) equilibria in symmetric min-max optimization is PPAD-complete, even for quadratic functions. Building on this reduction, we show that computing symmetric ϵ-Nash equilibria in symmetric, 6-player (3 vs. 3) team zero-sum games is also PPAD-complete, even for ϵ = poly(1/n). As a corollary, this precludes the existence of symmetric dynamics-which includes many of the algorithms considered in the literatureconverging to stationary points. Finally, we prove that computing a non-symmetric poly(1/n)-equilibrium in symmetric min-max optimization is FNP-hard.

Section: Introduction
We consider computing local equilibria in constrained min-max optimization problems of the form
min x∈X max y∈Y f (x, y),(1)
where X ⊆ R dx and Y ⊆ R dy are convex and compact constraint sets, and f : X ×Y → R is a smooth objective function. Tracing all the way back to Von Neumann's celebrated minimax theorem [von Neumann, 1928] and the inception of game theory, such problems are attracting renewed interest in recent years propelled by a variety of modern machine learning applications, such as generative modeling [Goodfellow et al., 2014], reinforcement learning [Daskalakis et al., 2020, Bai and Jin, 2020, Wei et al., 2021], and adversarial robustness [Madry et al., 2018, Cohen et al., 2019, Bai et al., 2021, Carlini et al., 2019]. Another prominent class of problems encompassed by (1) concerns computing Nash equilibria in (two-)team zero-sum games [Zhang et al., 2023, 2021, Basilico et al., 2017, von Stengel and Koller, 1997, Carminati et al., 2023, Orzech and Rinard, 2023, Farina et al., 2018, Zhang and An, 2020, Celli and Gatti, 2018, Schulman and Vazirani, 2017], which is a primary focus of this paper.
Perhaps the most natural solution concept-guaranteed to always exist-pertaining to ( 1), when f is nonconvex-nonconcave, is a pair of strategies (x * , y * ) such that both players (approximately) satisfy the associated first-order optimality conditions [Tsaknakis and Hong, 2021, Jordan et al., 2023, Ostrovskii et al., 2021, Nouiehed et al., 2019], as formalized in the definition below.
Definition 1.1. A point (x * , y * ) ∈ X × Y is an ϵ-first-order Nash equilibrium of (1) if ⟨x -x * , ∇ x f (x * , y * )⟩ ≥ -ϵ and ⟨y -y * , ∇ y f (x * , y * )⟩ ≤ ϵ ∀(x, y) ∈ X × Y.
Definition 1.1 can be equivalently recast as a variational inequality (VI) problem: if z := (x, y) and F : z → F (z) := (∇ x f (x, y), -∇ y f (x, y)), we are searching for a point z * ∈ Z := X × Y such that ⟨z -z * , F (z * )⟩ ≥ -2ϵ for all z ∈ Z. Yet another equivalent definition is instead based on approximate fixed points of gradient descent/ascent (GDA); namely, Definition 1.1 amounts to bounding the gradient mappings
∥x * -Π X (x * -∇ x f (x * , y * ))∥ ≤ ϵ ′ , ∥y * -Π Y (y * +∇ y f (x * , y * ))∥ ≤ ϵ ′ (Fixed points of GDA)
for some approximation parameter ϵ ′ > 0 that is (polynomially) dependent on ϵ > 0, where ∥ • ∥ is the (Euclidean) ℓ 2 norm and Π(•) is the projection operator. Other definitions that differentiate between the order of play between players-based on the notion of a Stackelberg equilibrium-have also been considered in the literature [Jin et al., 2020].
The complexity of min-max optimization is well-understood in certain special cases, such as when f is convex-concave (e.g., Korpelevich [1976], Mertikopoulos et al. [2019], Cai et al. [2022], Choudhury et al. [2023], Gorbunov et al. [2022], and references therein), or more broadly, nonconvexconcave [Lin et al., 2020, Xu et al., 2023, Luo et al., 2020]. However, the complexity of general min-max optimization problems, when the objective function f is nonconvex-nonconcave, has remained wide open despite intense efforts in recent years. Daskalakis et al. [2021] made progress by establishing certain hardness results targeting the more challenging setting in which there is a joint (that is, coupled) set of constraints. In fact, it turns out that their lower bounds apply even for linear-nonconcave objective functions (cf. Bernasconi et al. [2024]), showing that their hardness result is driven by the presence of joint constraints-indeed, under uncoupled constraints, many efficient algorithms attaining Definition 1.1 (for linear-nonconcave problems) have been documented in the literature. In the context of min-max optimization, the most well-studied setting posits that players have independent constraints; this is the primary focus of our paper.
this section cite: ['b62', 'b33', 'b22', 'b3', 'b44', 'b17', 'b11', 'b66', 'b54', 'b5', 'b63', 'b53', 'b26', 'b57', 'b37', 'b54', 'b52', 'b36', 'b41', 'b48', 'b42', 'b66', 'b43', 'b6']

Section: Our results
We establish new complexity lower bounds in min-max optimization for computing equilibria in the sense of Definition 1.1; our main results are gathered in Table 1.
Table 1: The main results of this paper. NE stands for Nash equilibrium and FONE for first-order Nash equilibrium (Definition 1.1). We also abbreviate symmetric to "sym." (second column). Adversarial team games We first examine an important special case of (1): adversarial team games [von Stengel and Koller, 1997]. Here, a team of n players with identical interests is competing against a single adversarial player. (In such settings, Definition 1.1 captures precisely the Nash equilibria of the game.) The computational complexity of this problem was placed by Anagnostides et al. [2023] in the complexity class CLS-which stands for continuous local search [Daskalakis and Papadimitriou, 2011]. Further, by virtue of a result of Babichenko and Rubinstein [2021], computing Nash equilibria in adversarial team games when n ≫ 1 is CLS-complete. In the context of this prior work, an important question left open by Anagnostides et al. [2023] concerns the case where n is a small constant, a regime not captured by the hardness result of Babichenko and Rubinstein [2021] pertaining to identical-interest games-in such games, one can simply identify the strategy leading to the highest payoff, which is tractable when n is small.
We show that even when n = 2, computing an ϵ-Nash equilibrium in adversarial team games is CLS-complete. (The case where n = 1 amounts to two-player zero-sum games, known to be in P.) Theorem 1.2. Computing an ϵ-Nash equilibrium in 3-player (that is, 2 vs. 1) adversarial team games is CLS-complete.
Coupled with earlier results, Theorem 1.2 completely characterizes the complexity landscape for computing Nash equilibria in adversarial team games.
Our proof is based on a recent hardness result of Ghosh and Hollender [2024] (cf. Tewolde et al. [2025]), who proved that computing a symmetric ϵ-Nash equilibrium in a symmetric two-player game with identical payoffs is CLS-complete. The key idea in our reduction is that one can leverage the adversarial player so as to enforce symmetry between the team players, without affecting the equilibria of the original game; the basic gadget underpinning this reduction is analyzed in Section 3.1.
Incidentally, our CLS-hardness reduction hinges on a polymatrix adversarial team game, thereby addressing another open question left recently by Hollender et al. [2025].
Theorem 1.3. Theorem 1.2 holds even when one restricts to polymatrix, 3-player adversarial team games.
We complement the above hardness result by further characterizing the complexity of deciding whether an adversarial team game admits a unique (approximate) Nash equilibrium (Theorem 3.5).
this section cite: ['b63', 'b20']

Section: Symmetric min-max optimization
As we have seen, symmetry plays a key role in the proof of Theorem 1.2, but that result places no restrictions on whether the equilibrium is symmetric or not-this is indeed the crux of the argument. The next problem we consider concerns computing symmetric equilibria in symmetric min-max optimization problems, in the following natural sense.
Definition 1.4 (Symmetric min-max optimization). A function f : X × Y → R is called antisym- metric if X = Y and f (x, y) = -f (y, x) ∀(x, y) ∈ X × Y. Furthermore, a point (x, y) ∈ X × Y is called symmetric if x = y.
The associated min-max optimization problem is called symmetric if the underlying function f is antisymmetric.foot_0 Symmetric zero-sum games are ubiquitous in the literature and in practical applications alike. Many popular recreational games used for AI benchmarking, such as poker and battleship, are symmetric when roles are assigned at random; the symmetry assumption is particularly natural as it ensures that no player has an a priori advantage before the game begins.
The study of symmetric equilibria has a long history in the development of game theory, propelled by Nash's pathbreaking PhD thesis [Nash, 1950] (cf. Gale et al. [1951]), and has remained a popular research topic ever since [Tewolde et al., 2025, Emmons et al., 2022, Garg et al., 2018, Ghosh and Hollender, 2024, Mehta, 2014]. Classic examples in game theory, including rock-paper-scissors and matching pennies, are also symmetric; these games were already discussed in the original work of von Neumann [1928].
It is not hard to see that symmetric min-max optimization problems, in the sense of Definition 1.4, always admit symmetric first-order Nash equilibria. What is more, we show that computing such a symmetric equilibrium is in the complexity class PPAD [Papadimitriou, 1994]; this is based on an argument of Etessami and Yannakakis [2010], and complements Daskalakis et al. [2021], who proved that the problem of computing approximate fixed points of gradient descent/ascent-which they refer to as GDAFIXEDPOINT-lies in PPAD. In a celebrated series of work, it was shown that PPAD captures the complexity of computing Nash equilibria in finite games [Daskalakis et al., 2009, Chen et al., 2009]. In this context, we establish that PPAD also characterizes the complexity of computing symmetric first-order Nash equilibria in symmetric min-max optimization problems: Theorem 1.5. Computing a symmetric 1 /n c -approximate first-order Nash equilibrium in symmetric n-dimensional min-max optimization is PPAD-complete for any constant c > 0.
Barring major complexity breakthroughs, Theorem 1.5 precludes the existence of algorithms with complexity polynomial in the dimension and 1/ϵ, where ϵ > 0 measures the precision (per Definition 1.1), under the symmetry constraint of Definition 1.4. This stands in contrast to (nonconvex) minimization problems, wherein gradient descent converges to stationary points at a rate of poly(1/ϵ); even in the regime where ϵ = 1/ exp(n), computing a stationary point of a smooth function is in CLS [Daskalakis and Papadimitriou, 2011], which is a subclass of PPAD [Fearnley et al., 2023]. In fact, our reduction also rules out the existence of polynomial-time algorithms even when ϵ = Θ(1) under some well-believed complexity assumptions (Corollary 4.3).
The proof of Theorem 1.5 is elementary, and is based on the PPAD-hardness of computing symmetric Nash equilibria in symmetric two-player games. Importantly, our reduction gives an immediate, and significantly simpler, proof (Theorem 4.4) of the PPAD-hardness result of Daskalakis et al. [2021], while being applicable even with respect to quadratic and anti-symmetric functions defined on a product of simplexes.
Independent and concurrent work Bernasconi et al. [2024] also considerably simplified the proof of Daskalakis et al. [2021]. Our hardness result hinges on the intermediate problem of finding a Nash equilibrium in symmetric two-player games [Chen et al., 2009], whereas Bernasconi et al. [2024] showed their hardness result via the problem of finding a Nash equilibrium in (multi-player) polymatrix two-action games. The main qualitative difference between the two is that ours applies to simplex domains while the result of Bernasconi et al. [2024] to box domains. The basic idea of both reductions then is that one can enforce the symmetry constraint x ≈ y via coupled constraints.
As a byproduct of Theorem 1.5 and the result of Bernasconi et al. [2024], it follows that any symmetric dynamics-whereby both players follow the same online algorithm, as formalized in Definition 4.5-cannot converge to a first-order Nash equilibrium in polynomial time, subject to PPAD ̸ = P (Theorem 4.6). This already captures many natural dynamics for which prior papers in the literature (e.g., Kalogiannis et al. [2023b]) have painstakingly shown lack of convergence; Theorem 4.6 provides a complexity-theoretic justification for such prior results, while precluding a much broader family of algorithms.
The complexity of non-symmetric equilibria Remaining on symmetric min-max optimization, one natural question arising from Theorem 1.5 concerns the complexity of non-symmetric equilibriadefined as having distance at least δ > 0. Unlike their symmetric counterparts, non-symmetric first-order Nash equilibria are not guaranteed to exist. In fact, we establish the following result. Theorem 1.6. For a symmetric min-max optimization problem, constants c 1 , c 2 > 0, and ϵ = n -c1 , it is NP-hard to distinguish between the following two cases under the promise that one of them holds:
• any ϵ-first-order Nash equilibrium (x * , y * ) satisfies ∥x * -y * ∥ ≤ n -c2 , and • there is an ϵ-first-order Nash equilibrium (x * , y * ) such that ∥x * -y * ∥ ≥ Ω(1).
The main technical piece is Theorem 4.7, which concerns symmetric, identical-interest, two-player games. It significantly refines the hardness result of McLennan and Tourky [2010] by accounting even for poly(1/n)-Nash equilibria.
Team zero-sum games Finally, building on the reduction of Theorem 1.5 coupled with the gadget behind Theorem 1.2, we establish similar complexity results for team zero-sum games, which generalize adversarial team games by allowing the presence of multiple adversaries. In particular, a symmetric two-team zero-sum game and a symmetric equilibrium thereof are in accordance with Definition 1.4-no symmetry constraints are imposed within the same team, but only across teams. We obtain a result significantly refining Theorem 1.5. Theorem 1.7. Computing a symmetric 1 /n c -Nash equilibrium in symmetric, 6-player (3 vs. 3) team zero-sum polymatrix games is PPAD-complete for some constant c > 0.
Unlike our reduction in Theorem 1.5 that comprises quadratic terms, the crux in team zero-sum games is that one needs to employ solely multilinear terms. The basic idea is to again use the gadget underpinning Theorem 1.2, which enforces symmetry without affecting the equilibria of the game, thereby (approximately) reproducing the objective function that establishes Theorem 1.5.
It is interesting to note that the class of polymatrix games we construct to prove Theorem 1.7 belongs to a certain family introduced by Cai and Daskalakis [2011]: one can partition the players into 2 groups so that any pairwise interaction between players of the same group is a coordination game, whereas any pairwise interaction across groups is a zero-sum game. Cai and Daskalakis [2011] showed that computing a Nash equilibrium is PPAD-hard in the more general case where there are 3 groups of players. While the complexity of that problem under 2 groups remains wide open, Theorem 1.7 shows PPAD-hardness for computing symmetric Nash equilibria in such games.
Taken together, our results bring us closer to characterizing the complexity of computing equilibria in min-max optimization.
this section cite: ['b49', 'b28', 'b29', 'b46', 'b55', 'b25', 'b21', 'b15', 'b20', 'b27', 'b6', 'b15', 'b6', 'b6', 'b6', 'b45', 'b9', 'b9']

Section: Further related work
Adversarial team games have been the subject of much research tracing back to the influential work of von Stengel and Koller [1997], who introduced the concept of a team maxmin equilibrium (TME); a TME can be viewed as the best Nash equilibrium for the team. Notwithstanding its intrinsic appeal, it turns out that computing a TME is FNP-hard [Borgs et al., 2010]. Indeed, unlike two-player zero-sum games, team zero-sum games generally exhibit a duality gap-characterized in the work of Schulman and Vazirani [2017].
This realization has shifted the focus of contemporary research to exploring more permissive solution concepts. One popular such relaxation is TMECor, which enables team players to ex ante correlate their strategies [Zhang et al., 2023, 2021, Basilico et al., 2017, Carminati et al., 2022, Farina et al., 2018, Zhang and An, 2020, Celli and Gatti, 2018]. Yet, in the context of extensive-form games, computing a TMECor remains intractable; Zhang et al. [2023] provided an exact characterization of its complexity. Team zero-sum games can be thought of as two-player zero-sum games but with imperfect recall, and many natural problems immediately become hard without perfect recall (e.g., Tewolde et al. [2023]). Parameterized algorithms have been developed for computing a TMECor based on some natural measure of shared information [Zhang et al., 2023, Carminati et al., 2022]. Beyond adversarial team games, Carminati et al. [2023] recently explored hidden-role games, wherein there is uncertainty regarding which players belong in the same team, a feature that often manifests itself in popular recreational games-and used certain cryptographic primitives to solve them.
In contrast, this paper focuses on the usual Nash equilibrium concept, being thereby orthogonal to the above line of work. One drawback of Nash equilibria in adversarial team games is that the (worst-case) value of the team can be significantly lower compared to TME [Basilico et al., 2017]. On the other hand, Anagnostides et al. [2023] showed that ϵ-Nash equilibria in adversarial team games admit an FPTAS, which stands in stark contrast to TME, and indeed, Nash equilibria in general games [Daskalakis et al., 2009, Chen et al., 2009]. This was further strengthened by Kalogiannis et al. [2023aKalogiannis et al. [ , 2024] ] for computing ϵ-Nash equilibria in adversarial team Markov games-the natural generalization to Markov (aka. stochastic) games. Related to Definition 1.1 is the natural notion of a local min-max equilibrium [Daskalakis andPanageas, 2018, Daskalakis et al., 2021]. It is easy to see that any local min-max equilibrium-with respect to a sufficiently large neighborhood of (x * , y * )-must satisfy Definition 1.1 [Daskalakis et al., 2021]. Unlike first-order Nash equilibria, local min-max equilibria are not guaranteed to exist.
Finally, Mehta et al. [2015] showed that in two-player symmetric games, deciding whether a nonsymmetric Nash equilibrium exists is NP-hard, which directly relates to our Theorem 1.6.
this section cite: ['b63', 'b8', 'b57', 'b66', 'b54', 'b5', 'b26', 'b66', 'b66', 'b5', 'b21', 'b15', 'b47']

Section: Preliminaries
Notation We use boldface lowercase letters, such as x, y, z, to represent vectors, and boldface capital letters, such as A, C, for matrices. We denote by x i the ith coordinate of a vector x ∈ R n . We use the shorthand notation [n] := {1, 2, . . . , n}. ∆ n := {x ∈ R n ≥0 : n i=1 x i = 1} is the probability simplex on R n . For i ∈ [n], e i ∈ ∆ n is the ith unit vector. ⟨•, •⟩ denotes the inner product. For a vector x ∈ R n , ∥x∥ 2 = ⟨x, x⟩ is its Euclidean norm. For m ≤ n, x [1•••m] ∈ R m is the vector containing the first m coordinates of x. We sometimes use the O(•), Θ(•), Ω(•) notation to suppress absolute constants. A continuously differentiable function f is L-smooth if its gradient is L-Lipschitz continuous with respect to ∥ • ∥ 2 ; that is, ∥∇f (x) -∇f (x ′ )∥ 2 ≤ L∥x -x ′ ∥ 2 for all x, x ′ .
this section cite: []

Section: Two-player games
In a two-player game, represented in normal-form game, each player has a finite set, let [n], of actions. Under a pair of actions (i, j) ∈ [n] × [n], the utility of the row player is given by R i,j , where R ∈ Q n×n is the payoff matrix of the row player. Further, we let C ∈ Q n×n be the payoff matrix of the column player. Players are allowed to randomize by selecting mixed strategies-points in ∆ n . Under a pair of mixed strategies (x, y) ∈ ∆ n × ∆ n , the expected utility of the players is given by ⟨x, Ry⟩ and ⟨x, Cy⟩, respectively. The canonical solution concept in such games is the Nash equilibrium [Nash, 1951], which is recalled below. Definition 2.1. A pair of strategies (x * , y * ) is an ϵ-Nash equilibrium of (R, C) if ⟨x * , Ry * ⟩ ≥ ⟨x, Ry * ⟩ -ϵ and ⟨x * ,
Cy * ⟩ ≥ ⟨x * , Cy⟩ -ϵ ∀(x, y) ∈ ∆ n × ∆ n .
this section cite: ['b50']

Section: Symmetric two-player games
One of our reductions is based on symmetric two-player games, meaning that R = C ⊤ . A basic fact is that any symmetric game admits a symmetric Nash equilibrium (x * , x * ). Further, computing a Nash equilibrium in a general game can be reduced to computing a symmetric Nash equilibrium in a symmetric game [Nisan et al., 2007, Theorem 2.4]. In conjunction with the hardness result of Chen et al. [2009], we state the following consequence. Theorem 2.2 (Chen et al., 2009). Computing a symmetric 1 /n c -Nash equilibrium in a symmetric two-player game is PPAD-hard for any constant c > 0.
Team zero-sum games A (two-)team zero-sum game is a multi-player game-represented in normal form for the purposes of this paper-in which the players' utilities have a certain structure; namely, we can partition the players into two (disjoint) subsets, such that each player within the same team shares the same utility, whereas players in different teams have opposite utilities-under any possible combination of actions. An adversarial team game is a specific type of team zero-sum game wherein one team consists of a single player. As in Definition 2.1 for two-player games, an ϵ-Nash equilibrium is a tuple of strategies such that no unilateral deviation yields more than an ϵ additive improvement in the utility of the deviator.
this section cite: ['b15', 'b15']

Section: Complexity of adversarial team games
We begin by examining equilibrium computation in adversarial team games.
this section cite: []

Section: CLS-completeness for 3-player games
Computing ϵ-Nash equilibria in adversarial team games was placed in CLS by Anagnostides et al. [2023], but whether CLS tightly characterizes the complexity of that problem remained open-that was only known when the number of players is large, so that the hardness result of Babichenko and Rubinstein [2021] can kick in. Our reduction here answers this question in the affirmative.
We rely on a recent hardness result of Ghosh and Hollender [2024] concerning symmetric, two-player games with identical payoffs. We summarize their main result below. Theorem 3.1 (Ghosh and Hollender, 2024). Computing an ϵ-Nash equilibrium in a symmetric, identical-payoffs, two-player game is CLS-complete.
Now, let A ∈ Q n×n be the common payoff matrix of a two-player game, which satisfies A = A ⊤ so that the game is symmetric. Without loss of generality, we will assume that A i,j ≤ -1 for all i, j ∈ [n]. We denote by A min and A max the minimum and maximum entry of A, respectively (which satisfy A max , A min ≤ -1). The basic idea of our proof is to suitably use the adversarial player so as to force the other two players to play roughly the same strategy (Lemma 3.2), while (approximately) maintaining the structure of the game (Lemma 3.3). The formal proofs are in Section A.1.
this section cite: []

Section: Definition of the adversarial team game
Based on A, we construct a 3-player adversarial team game as follows. The utility function of the adversary reads
u(x, y, z) := ⟨x, Ay⟩ + |A min | ϵ n i=1 (z i (x i -y i ) + z n+i (y i -x i )) + z 2n+1 |A min |.(2)
The adversary selects a strategy z ∈ ∆ 2n+1 , while the team players, who endeavor to minimize (2), select strategies x ∈ ∆ n and y ∈ ∆ n , respectively. (While the range of the utilities in (2) grows with 1 /ϵ, normalizing to [-1, 1] maintains all of the consequences by suitably adjusting the approximation.)
The first important lemma establishes that, in equilibrium, x ≈ y. The basic argument proceeds as follows. By construction of (2), the adversary would be able to secure a large payoff whenever there is a coordinate i ∈ [n] such that |x i -y i | ≫ 0-by virtue of the second term in (2). But that cannot happen in equilibrium, for Player x (or symmetrically Player y) can simply neutralize that term in the adversary's utility by playing x = y.
Lemma 3.2 (Equilibrium forces symmetry). Consider an ϵ 2 -Nash equilibrium (x * , y * , z * ) of the adversarial team game (2) with ϵ 2 ≤ 1 /2. Then, ∥x * -y * ∥ ∞ ≤ 2ϵ.
Having established that x ≈ y, the next step is to make sure that the adversarial player does not distort the original game by much. In particular, we need to make sure that the effect of the second term in (2) is negligible. We do so by showing that z 2n+1 ≈ 1 (Lemma 3.3).
The argument here is more subtle; roughly speaking, it goes as follows. Suppose that
z i ≫ 0 or z n+i ≫ 0 for some i ∈ [n].
Since Player z is approximately best responding, it would then follow that
|y * i -x * i | ≫ 0-otherwise Player z would prefer to switch to action 2n+1. But, if |y * i -x * i | ≫ 0, Player x could
profitably deviate by reallocating probability mass by either removing from or adding to i (depending on whether
y * i -x * i > 0), which leads to a contradiction. Lemma 3.3 (Most probability mass in a 2n+1 ). Given any ϵ 2 -Nash equilibrium (x * , y * , z * ) of the adversarial team game (2) with ϵ ≤ 1 /10, z j ≤ 9ϵ for all j ∈ [2n]. In particular, z 2n+1 ≥ 1 -18nϵ.
By combining Lemmas 3.2 and 3.3, we can complete the reduction from symmetric two-player games with common payoffs to 3-player adversarial team games, as stated below.
Theorem 3.4. Given any ϵ 2 -Nash equilibrium (x * , y * , z * ) in the adversarial team game (2), with
ϵ ≤ 1 /10, (y * , y * ) is a symmetric (21n + 1)|A min |ϵ-Nash equilibrium of the symmetric, two-player game (A, A) (that is, A = A ⊤ ).
this section cite: []

Section: The complexity of determining uniqueness
Another natural question concerns the complexity of determining whether an adversarial team game admits a unique Nash equilibrium. Our next theorem establishes NP-hardness for a version of that problem that accounts for approximate Nash equilibria.
Theorem 3.5. For polymatrix, 3-player adversarial team games, constants c 1 , c 2 > 0, and ϵ = n -c1 , it is NP-hard to distinguish between the following two cases under the promise that one of them holds:
• any two ϵ-Nash equilibria have ℓ 1 -distance at most n -c2 , and • there are two ϵ-Nash equilibria that have ℓ 1 -distance Ω(1).
We will discuss more about the proof of this theorem later in Section 4.2 when we examine the complexity of computing non-symmetric equilibria in symmetric min-max optimization problems. It is also interesting to point out that an adversatial team game can have a unique Nash equilibrium supported on irrational numbers, as we show in Section A.2.
this section cite: []

Section: Complexity of equilibria in symmetric min-max optimization
This section characterizes the complexity of computing symmetric first-order Nash equilibria (Definition 1.1) in symmetric min-max optimization problems in the sense of Definition 1.4; namely, when f (x, y) = -f (y, x) for all (x, y) ∈ X × Y and X = Y.
this section cite: []

Section: Problem definitions and hardness results for symmetric equilibria
Given a continuously differentiable function f : D → R, we set F GDA : D → D to be F GDA (x, y) := D [x -∇ x f (x, y), y + ∇ y f (x, y)] for (x, y) ∈ D, the norm of which measures the fixed-point gap and corresponds to the update rule of GDA with stepsize equal to one; we recall that Player x is the minimizer, while Player y is the maximizer. The domain D is a compact subset of R d for some d ∈ N. Moreover, the projection operator is applied jointly on D. 3 When D can be expressed as a Cartesian product X × Y, the domain set is called uncoupled (and the projection can be done independently), otherwise it is called coupled (or joint).
We begin by introducing the problem of computing fixed points of gradient descent/ascent (GDA) for domains expressed as the Cartesian product of polytopes, modifying the computational problem GDAFIXEDPOINT introduced by Daskalakis et al. [2021].
this section cite: []

Section: GDAFIXEDPOINT Problem.
INPUT:
• Precision parameter ϵ > 0 and smoothness parameter L, • Polynomial-time Turing machine C f evaluating a L-smooth function f : X ×Y → R and its gradient ∇f : X × Y → R d , where
X = {x : A x x ≤ b x } and Y = {y : A y y ≤ b y } are nonempty, bounded polytopes described by input matrices A x ∈ R mx×dx , A y ∈ R my×dy and vectors b x ∈ R mx , b y ∈ R my , with d := d x + d y . OUTPUT: A point (x * , y * ) ∈ X × Y such that ∥(x * , y * ) -F GDA (x * , y * )∥ 2 ≤ ϵ.
Based on GDAFIXEDPOINT, we introduce the problem SYMGDAFIXEDPOINT, which captures the problem of computing symmetric (approximate) fixed points of GDA for symmetric min-max optimization problems. We define our computational problems as promise problems.
this section cite: []

Section: SYMGDAFIXEDPOINT Problem.
INPUT:
• Precision parameter ϵ > 0 and smoothness parameter L,
• Polynomial-time Turing machine C f evaluating a L-smooth, antisymmetric function f : X × X → R and its gradient ∇f : X × X → R 2d , where X = {x : Ax ≤ b} is a nonempty, bounded polytope described by an input matrix A ∈ R m×d and vector b ∈ R m .
OUTPUT: A point (x * , x * ) ∈ X × X such that ∥(x * , x * ) -F GDA (x * , x * )∥ 2 ≤ ϵ.
We start by showing that SYMGDAFIXEDPOINT also lies in PPAD; the fact that GDAFIXEDPOINT is in PPAD-even under coupled domains-was shown to be the case by Daskalakis et al. [2021].
The detailed proof is included in the appendix. Lemma 4.1. SYMGDAFIXEDPOINT is a total search problem and lies in PPAD.
Having established that SYMGDAFIXEDPOINT belongs in PPAD, we now state the first main hardness result of this section. Theorem 4.2 (Complexity for symmetric equilibrium). SYMGDAFIXEDPOINT is PPAD-complete, even for quadratic functions.
The basic idea of the proof is to consider the objective
f (x, y) := 1 2 ⟨y, Ay⟩ - 1 2 ⟨x, Ax⟩ + ⟨y, Cx⟩,(3)
where A is symmetric and C is skew-symmetric. Theorem 4.2 then follows from some elementary calculations, as we show in Section A.3.
For symmetric first-order Nash equilibria, our argument establishes PPAD-hardness for any ϵ ≤ 1 /n c , where c > 0 (as claimed in Theorem 1.5). Moreover, leveraging the hardness result of Rubinstein [2016], we can also immediately obtain constant inapproximability under the so-called exponentialtime hypothesis (ETH) for PPAD-which postulates than any algorithm for solving ENDOFALINE, the prototypical PPAD-complete problem, requires 2 Ω(n) time. Corollary 4.3. Computing an Θ(1)-approximate first-order Nash equilibrium in symmetric ndimensional min-max optimization requires n Ω(log n) time, assuming ETH for PPAD.
The argument of Theorem 4.2 can be slightly modified to imply the main result of Daskalakis et al.
[2021]-with simplex instead of box constraints-as stated below. Theorem 4.4 (PPAD-hardness for coupled domains). The problem GDAFIXEDPOINT is PPAD-hard when the domain is a joint polytope, even for quadratic functions.
The main idea is to add coupled constraints in order to force symmetry: -δ ≤ x i -y i ≤ δ for all i ∈ [n], where, if ϵ is the approximation accuracy, δ is of order Θ ϵ 1/4 . Compared to the equilibrium studied in Daskalakis et al. [2021], the symmetric equilibrium considered in our work is stronger in that it accounts for all deviations, not merely ones on the coupled feasibility set. We present the proof of Theorem 4.4 in Section A.3.
Hardness results for symmetric dynamics Another interesting consequence of Theorem 4.2 is that it precludes convergence under a broad class of algorithms in general min-max optimization.
Definition 4.5 (Symmetric learning algorithms for min-max). Let T ∈ N. A deterministic, polynomial-time learning algorithm A proceeds as follows for any time t ∈ [T ]. It outputs a strategy as a function of the history H (t) it has observed so far (where H (1) := ∅ ), and then receives as feedback g (t) . It then updates H (t+1) := (H (t) , g (t) ). A symmetric learning algorithm in min-max optimization consists of Player x employing algorithm A with history H
x := (∇ x f (x (t) , y (t) )) T t=1 , and Player y employing the same algorithm with history H (t)
y := (-∇ y f (x (t) , y (t) )) T
t=1 . Note that a consequence of the above definition is that both players initialize from the same strategy. Many natural and well-studied algorithms in min-max optimization adhere to Definition 4.5. Besides the obvious example of gradient descent/ascent, we mention extragradient descent(/ascent), optimistic gradient descent(/ascent), and optimistic multiplicative weights-all assumed to be executed simultaneously. A simple non-example is alternating gradient descent(/ascent) [Wibisono et al., 2022, Bailey et al., 2020], wherein players do not update their strategies simultaneously. Theorem 4.6. No symmetric learning algorithm (per Definition 4.5) can converge to ϵ-first-order Nash equilibria in min-max optimization in polynomial time when ϵ = 1 /n c , unless PPAD = P. This is a consequence of our argument in Theorem 4.2: under Definition 4.5 and the min-max optimization problem (3), it follows inductively that x (t) = y (t) and
H (t) x = H (t)
y for all t ∈ [T ]. But computing a symmetric first-order Nash equilibrium is PPAD-hard when ϵ = 1 /n c (Theorem 4.2).
Assuming that P ̸ = PPAD, Theorem 4.6, and in particular its instantiation in team zero-sum games (Theorem 1.7), significantly generalizes some impossibility results shown by Kalogiannis et al. [2023b] concerning certain algorithms, such as optimistic gradient descent(/ascent)-our hardness result goes much further, precluding any algorithm subject to Definition 4.5, albeit being conditional.
this section cite: ['b56', 'b4']

Section: The complexity of non-symmetric fixed points
An immediate question raised by Theorem 4.2 concerns the computational complexity of finding non-symmetric fixed points of GDA for symmetric min-max optimization problems. Since totality is not guaranteed, unlike SYMGDAFIXEDPOINT, we cannot hope to prove membership in PPAD.
In fact, we show that finding a non-symmetric fixed point of GDA is FNP-hard. To do so, we first define formally the computational problem of interest. NONSYMGDAFIXEDPOINT Problem. INPUT:
• Parameters ϵ, δ > 0 and Lipschitz constant L and • Polynomial-time Turing machine C f evaluating a L-smooth antisymmetric function f : X × X → R and its gradient ∇f : X × X → R 2d , where X = {x : Ax ≤ b} is a nonempty, bounded polytope described by a matrix A ∈ R m×d and vector b ∈ R m .
OUTPUT: A point (x * , y * ) ∈ X × X such that ∥x * -y * ∥ 2 ≥ δ and ∥(x * , y * ) -F GDA (x * , y * )∥ 2 ≤ ϵ if it exists, otherwise return NO.
We establish that NONSYMGDAFIXEDPOINT is FNP-hard. Our reduction builds on the hardness result of McLennan and Tourky [2010]-in turn based on earlier work by Gilboa and Zemel [1989], Conitzer and Sandholm [2008]-which we significantly refine in order to account for poly(1/n)-Nash equilibria. Our result, which forms the basis for Theorem 1.6 and Theorem 3.5, is summarized below. Theorem 4.7. For symmetric, identical-interest, two-player games, constants c 1 , c 2 > 0, and ϵ = n -c1 , it is NP-hard to distinguish between the following two cases under the promise that one of them holds:
• any two symmetric ϵ-Nash equilibria have ℓ 1 -distance at most n -c2 , and • there are two symmetric ϵ-Nash equilibria that have ℓ 1 -distance Ω(1).
The proof of Theorem 1.6 now follows by considering the antisymmetric function f (x, y) := y ⊤ Byx ⊤ Bx for a suitable matrix B (defined per the hard instance from Theorem 4.7 based on k-clique). FNP-hardness follows similarly by considering a search version of maximum clique.
Finally, the proof of Theorem 3.5 that was claimed earlier follows immediately by combining Theorem 4.7 with the reduction of Section 3.1, and in particular, Lemmas 3.2 and 3.3.
this section cite: ['b45', 'b32', 'b18']

Section: Team zero-sum games
Our previous hardness result concerning symmetric min-max optimization problems does not have any immediate implications for (normal-form) team zero-sum games since the class of hard instances we constructed earlier contains a quadratic term. Our next result provides such a hardness result by combining the basic gadget we introduced in Section 3.1 in the context of adversarial team games; the basic pieces of the argument are similar to the ones we described in Section 3.1, and so the proof is deferred to Section A.5. Our goal is to prove the following. Theorem 1.7. Computing a symmetric 1 /n c -Nash equilibrium in symmetric, 6-player (3 vs. 3) team zero-sum polymatrix games is PPAD-complete for some constant c > 0.
Let us describe the class of 3 vs. 3 team zero-sum games upon which our hardness result is based on. Based on (2), we define the auxiliary function
δ : ∆ n × ∆ n × ∆ 2n+1 ∋ (x, y, z) → |A min | ϵ n i=1 (z i (x i -y i ) + z n+i (y i -x i )) + |A min |z 2n+1 .
In what follows, the 3 players of the one team will be identified with (x, y, z), while the 3 players of the other team with ( x, ŷ, ẑ). We define the utility of the latter team to be u(x, y, z, x, ŷ, ẑ) = ⟨x, Ay⟩ -⟨ x, A ŷ⟩ + ⟨x, C x⟩ + δ(x, y, ẑ) -δ( x, ŷ, z),
where A is symmetric and C is skew-symmetric. The rest of the argument follows Section 3.1.
this section cite: []

Section: Conclusion and open problems
We have provided a number of new complexity results concerning min-max optimization in general, and team zero-sum games in particular (see Table 1). There are many interesting avenues for future research. The complexity of computing first-order Nash equilibria (equivalently, the GDAFIXEDPOINT problem) remains wide open, but our hardness results suggest a possible approach: as we have seen, in symmetric min-max optimization, computing either symmetric or non-symmetric equilibria is intractable, so it would be enough if one could establish this using the same underlying function-that is, somehow combine our two reductions into one. It would also be interesting to see whether our hardness results can be extended to more structured min-max optimization problems, such as adversarial training and GANs.
this section cite: []

Section: References
Ref_id:b0 Title: Algorithms and complexity for computing Nash equilibria in adversarial team games Year: ()
Ref_id:b1 Title: Settling the complexity of Nash equilibrium in congestion games Year: ()
Ref_id:b2 Title: Recent advances in adversarial training for adversarial robustness Year: ()
Ref_id:b3 Title: Provable self-play algorithms for competitive reinforcement learning Year: (2020)
Ref_id:b4 Title: Finite regret and cycles with fixed step-size via alternating gradient descent-ascent Year: (2020)
Ref_id:b5 Title: Team-maxmin equilibrium: Efficiency bounds and algorithms Year: (2017)
Ref_id:b6 Title: On the role of constraints in the complexity of min-max optimization Year: (2024)
Ref_id:b7 Title: On the computational complexity of decision problems about multi-player Nash equilibria Year: (2019)
Ref_id:b8 Title: The myth of the folk theorem Year: (2010)
Ref_id:b9 Title: On minmax theorems for multiplayer games Year: (2011)
Ref_id:b10 Title: Finite-time last-iterate convergence for learning in multi-player games Year: ()
Ref_id:b11 Title: On evaluating adversarial robustness Year: (2019)
Ref_id:b12 Title: A marriage between adversarial team games and 2-player games: Enabling abstractions, no-regret learning, and subgame solving Year: ()
Ref_id:b13 Title: Hiddenrole games: Equilibrium concepts and computation Year: ()
Ref_id:b14 Title: Computational results for extensive-form adversarial team games Year: (2018)
Ref_id:b15 Title: Settling the complexity of computing two-player Nash equilibria Year: (2009)
Ref_id:b16 Title: Single-call stochastic extragradient methods for structured non-monotone variational inequalities: Improved analysis under weaker conditions Year: ()
Ref_id:b17 Title: Certified adversarial robustness via randomized smoothing Year: (2019)
Ref_id:b18 Title: New complexity results about Nash equilibria Year: (2008)
Ref_id:b19 Title: The limit points of (optimistic) gradient descent in min-max optimization Year: (2018)
Ref_id:b20 Title: Continuous local search Year: (2011)
Ref_id:b21 Title: The complexity of computing a Nash equilibrium Year: (2009)
Ref_id:b22 Title: Independent policy gradient methods for competitive reinforcement learning Year: (2020)
Ref_id:b23 Title: The complexity of constrained min-max optimization Year: ()
Ref_id:b24 Title: For learning in symmetric teams, local optima are global Nash equilibria Year: ()
Ref_id:b25 Title: On the complexity of Nash equilibria and other fixed points Year: (2010)
Ref_id:b26 Title: Ex ante coordination and collusion in zero-sum multi-player extensive-form games Year: (2018)
Ref_id:b27 Title: The complexity of gradient descent: CLS = PPAD ∩ PLS Year: (2023)
Ref_id:b28 Title: On Symmetric Games Year: (1951)
Ref_id:b29 Title: ∃r-completeness for decision versions of multi-player (symmetric) Nash equilibria Year: (2018)
Ref_id:b30 Title: Accelerated gradient methods for nonconvex nonlinear and stochastic programming Year: (2016)
Ref_id:b31 Title: The complexity of symmetric bimatrix games with common payoffs Year: ()
Ref_id:b32 Title: Nash and correlated equilibria: Some complexity considerations Year: (1989)
Ref_id:b33 Title: Generative adversarial nets Year: (2014)
Ref_id:b34 Title: Last-iterate convergence of optimistic gradient method for monotone variational inequalities Year: ()
Ref_id:b35 Title: The complexity of two-team polymatrix games with independent adversaries Year: ()
Ref_id:b36 Title: What is local optimality in nonconvex-nonconcave minimax optimization Year: (2020)
Ref_id:b37 Title: First-order algorithms for nonlinear generalized Nash equilibrium problems Year: (2023)
Ref_id:b38 Title: Emmanouil-Vasileios Vlatakis-Gkaragkounis, Vaggos Chatziafratis, and Stelios Andrew Stavroulakis. Efficiently computing Nash equilibria in adversarial team markov games Year: ()
Ref_id:b39 Title: Towards convergence to Nash equilibria in two-team zero-sum games Year: (2023)
Ref_id:b40 Title: Learning equilibria in adversarial team markov games: A nonconvex-hidden-concave min-max optimization problem Year: ()
Ref_id:b41 Title: The extragradient method for finding saddle points and other problems Year: (1976)
Ref_id:b42 Title: On gradient descent ascent for nonconvex-concave minimax problems Year: (2020)
Ref_id:b43 Title: Stochastic recursive gradient descent ascent for stochastic nonconvex-strongly-concave minimax problems Year: (2020)
Ref_id:b44 Title: Towards deep learning models resistant to adversarial attacks Year: (2018)
Ref_id:b45 Title: Simple complexity from imitation games Year: (2010)
Ref_id:b46 Title: Constant rank bimatrix games are ppad-hard Year: (2014)
Ref_id:b47 Title: Settling some open problems on 2-player symmetric Nash equilibria Year: (2015)
Ref_id:b48 Title: Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile Year: (2019)
Ref_id:b49 Title: Non-cooperative games Year: (1950)
Ref_id:b50 Title: Non-cooperative games Year: (1951)
Ref_id:b51 Title: Algorithmic Game Theory Year: (2007)
Ref_id:b52 Title: Solving a class of non-convex min-max games using iterative first order methods Year: (2019)
Ref_id:b53 Title: Correlated vs. uncorrelated randomness in adversarial congestion team games Year: (2023)
Ref_id:b54 Title: Efficient search of first-order nash equilibria in nonconvex-concave smooth min-max problems Year: (2021)
Ref_id:b55 Title: On the complexity of the parity argument and other inefficient proofs of existence Year: (1994)
Ref_id:b56 Title: Settling the complexity of computing approximate two-player Nash equilibria Year: (2016)
Ref_id:b57 Title: The duality gap for two-team zero-sum games Year: (2017)
Ref_id:b58 Title: Some properties of the Nash equilibrium in 2 × 2 zero-sum games Year: (2022)
Ref_id:b59 Title: The computational complexity of single-player imperfect-recall games Year: ()
Ref_id:b60 Title: Computing game symmetries and equilibria that respect them Year: ()
Ref_id:b61 Title: Finding first-order Nash equilibria of zero-sum games with the regularized nikaido-isoda function Year: ()
Ref_id:b62 Title:  Year: (1928)
Ref_id:b63 Title: Team-maxmin equilibria Year: (1997)
Ref_id:b64 Title: Last-iterate convergence of decentralized optimistic gradient descent/ascent in infinite-horizon competitive markov games Year: ()
Ref_id:b65 Title: Alternating mirror descent for constrained min-max games Year: ()
Ref_id:b66 Title: A unified single-loop alternating gradient projection algorithm for nonconvex-concave and convex-nonconcave minimax problems Year: (2023)
Ref_id:b67 Title: Team belief DAG: generalizing the sequence form to team games for fast computation of correlated team max-min equilibria via regret minimization Year: ()
Ref_id:b68 Title: Converging to team-maxmin equilibria in zero-sum multiplayer games Year: (2020)
Ref_id:b69 Title: Computing ex ante coordinated team-maxmin equilibria in zero-sum multiplayer extensive-form games Year: ()
