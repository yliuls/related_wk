Title: Quantum speedup of non-linear Monte Carlo problems
Abstract: The mean of a random variable can be understood as a linear functional on the space of probability distributions. Quantum computing is known to provide a quadratic speedup over classical Monte Carlo methods for mean estimation. In this paper, we investigate whether a similar quadratic speedup is achievable for estimating non-linear functionals of probability distributions. We propose a quantum-insidequantum algorithm that achieves this speedup for the broad class of nonlinear estimation problems known as nested expectations. Our algorithm improves upon the direct application of the quantum-accelerated multilevel Monte Carlo algorithm introduced by An et al. (2021). The existing lower bound indicates that our algorithm is optimal up to polylogarithmic factors. A key innovation of our approach is a new sequence of multilevel Monte Carlo approximations specifically designed for quantum computing, which is central to the algorithm's improved performance.

Section: Introduction
From classic problems like Buffon's needle (Ramaley, 1969) to modern Bayesian computations (Martin et al., 2023), Monte Carlo methods have proven to be powerful tools for estimating the expected value of a given function. Specifically, the classical Monte Carlo method involves estimating E P [f (X)] = f (x) dP(x) by sampling, where the samples are drawn from P. Assuming f has finite variance, the number of independent and identically distributed (i.i.d.) samples required to produce an estimate of E P [f (X)] with an additive error of ϵ and a given degree (say 95%) of confidence is O(1/ϵ 2 )foot_0 . In the quantum computing setting, using a Grover-type algorithm, it is known (Heinrich, 2002;Montanaro, 2015;Hamoudi and Magniez, 2019;Hamoudi, 2021;Kothari and O'Donnell, 2023) that a quantum-accelerated version of the Monte Carlo technique achieves a quadratic speedup, resulting in a cost of O(1/ϵ) for the same task. For broader links between quantum algorithms and classical stochastic simulation, see Blanchet et al. (2025).
A typical expectation E P [f (X)] functions as a linear functional on the space of distributions, mapping one distribution P to its corresponding expected value. However, many important quantities cannot be written as such linear functionals. For instance, the standard deviation maps a distribution P to sd(P) := E P [X 2 ] -(E P [X]) 2 . This mapping is nonlinear because the expectations are subsequently transformed by quadratic and square-root operations. This work addresses settings where the target quantity is a non-linear functional of P. Our goal is to estimate the nested expectation introduced in the next section. Our main contribution is a quantum-inside-quantum algorithm that achieves a near-optimal cost of O(1/ϵ).
this section cite: ['b31', 'b27', 'b18', 'b28', 'b16', 'b15', 'b25', 'b2']

Section: Nested expectation
We now establish the notation that will be used throughout the rest of the paper. Let P denote a probability distribution on X × Y. Define a function ϕ : X × Y → R and another function g : X × R → R. Let (X, Y ) ∈ X × Y be a pair of random variables with joint distribution P. Our paper addresses the estimation of the nested expectation, which takes the form:
E X g(X, E Y |X [ϕ(X, Y )]) .
(1) Equation ( 1) defines a non-linear function of P, with the non-linearity arising from the non-linear nature of g. If we set γ(x) := E Y |X=x [ϕ(x, Y )] and λ(x) := g(x, γ(x)), the expression (1) can be written more simply as E[λ(X)]. However, it is more challenging than the standard mean estimation problem as λ(x) further depends on a conditional expectation γ(x) that needs to be estimated. Consequently, a standard Monte Carlo method-which depends on computing λ exactly-is not directly applicable in this context.
The phrase "nested expectation", referring to an expectation taken inside another expectation, was formally introduced and defined by Rainforth et al. (2018). It also represents the simplest nontrivial case of "repeated nested expectation" (Zhou et al., 2023;Syed and Wang, 2023;Haji-Ali and Spence, 2023). Some concrete applications are as follows:
• In Bayesian experiment design (Lindley, 1956;Hironaka and Goda, 2023), the expected information loss of the random variable
Y is E X [log(E Y [Pr[X | Y ]])] -E Y [E X|Y [log(Pr[X | Y ])]].
Here, a nested expectation appears in the first term. In typical scenarios, the conditional probability Pr[X | Y ] has a closed-form expression, making it easy to evaluate, unlike Pr[X].
• Given f 1 , . . . , f d which can be understood as treatments, the expected value of partial perfect information (EVPPI) (Giles and Goda, 2019), is
E X [max k E Y |X [f k (X, Y )]] -max k E[f k (X, Y )],
which captures the benefit of knowing X. Here, a nested expectation appears also in the first term.
• In financial engineering, financial derivatives are typically evaluated using expectations and, therefore, Monte Carlo methods are often the method of choice in practice. One of the most popular derivatives is the so-called call option, whose value (in simplified form) can be evaluated as
E Y |X [max(Y -k, 0) | X].
Here, k is the strike price, Y is the price of the underlying asset at a future date, and X represents the available information on the underlying. For instance, the value of a Call on a Call (CoC) option (a call in which the underlying is also a call option with the same strike price) is
E X [max(E Y |X [max(Y -k, 0) | X] -k, 0)].
• The objective function in conditional stochastic optimization (CSO) (Hu et al., 2020;He and Kasiviswanathan, 2024) is formulated as a nested expectation, i.e.,
min x F (x) := min x E ξ [f (E η|ξ [g η (x, ξ)])].
Numerous methods are available for estimating nested expectations. The most natural way is by nesting Monte Carlo estimators. This method works by first sampling i.i.d. X 1 , X 2 , . . . , X m . For each X i , one further samples
Y i,1 , Y i,2 , . . . , Y i,n i.i.d. from Pr[Y | X i ]. Then γ(X i ) can be estimated by γ(X i ) := n j=1 ϕ(X i , Y i,j
)/n, and the final estimator is of the form
m i=1 g(X i , γ(X i )) m = m i=1 g(X i , n j=1 ϕ(X i , Y i,j )/n) m .(2)
Under different smoothness assumptions on g, this nested estimator costs nm = O(1/ϵ 3 ) to O(1/ϵ 4 ) samples to achieve a mean square error (MSE) of up to ϵ 2 (Hong and Juneja, 2009;Rainforth et al., 2018). The Multilevel Monte Carlo (MLMC) technique further improves efficiency to O(1/ϵ 2 ) or O((1/ϵ 2 ) log(1/ϵ) 2 ), as outlined in Section 9.1 of Giles (2015) and Blanchet and Glynn (2015).
If users have access to a quantum computer, An et al. (2021) proposed a QA-MLMC algorithm that improves upon the classical MLMC algorithm. The improvement ranges from quadratic to sub-quadratic-or can even be nonexistent-depending on the parameters of the MLMC framework, detailed in Sections 2.1 and 2.2.
Under the standard Lipschitz assumptions, directly applying the QA-MLMC of An et al. (2021) to our non-linear nested expectation problem incurs a cost of O(1/ϵ 2 ). This represents no improvement over the classical cost! A technical analysis explaining the loss of the quadratic speed-up is provided in Section 3.1.
this section cite: ['b30', 'b35', 'b33', 'b14', 'b26', 'b19', 'b11', 'b23', 'b17', 'b20', 'b30', 'b10', 'b3', 'b0', 'b0']

Section: Our contribution
Our main algorithmic contribution of this paper is a quantum-inside-quantum MLMC algorithm to estimate nested expectations. Under standard technical assumptions, the algorithm achieves a cost of O(1/ϵ) to produce an estimator with ϵ-accuracy, which is proven to be optimal among all quantum algorithms up to logarithmic factors. As such, our algorithm provides a quadratic speedup compared to the classical MLMC algorithm, making it more efficient than the direct adaptation of the quantum algorithm proposed in An et al. (2021). The comparison between our algorithm and existing methods is summarized in Table 1.
Table 1: Cost of estimating (1) with ϵ-accuracy using different methods.
Method Cost Nested estimator (Eq. (2)) Rainforth et al. (2018) O(ϵ -4 ) Classical MLMC (Thm 2.1) Giles (2015) O(ϵ -2 ) QA-MLMC (Thm 2.2) An et al. (2021) O(ϵ -2 ) Q-NESTEXPECT (Ours) O(ϵ -1 )
We provide a brief explanation of the basis of our improvement and the term "quantum-insidequantum", with more comprehensive explanations to follow in later sections. For a given target quantity and an error budget ϵ, every classical MLMC algorithm consists of two components: (1) decomposing the estimation problem into the task of separately estimating the expectations of different parts; and (2) distributing the total error budget ϵ among these parts to minimize the overall computational cost, followed by using (classical) Monte Carlo to estimate each expectation. The QA-MLMC algorithm (An et al., 2021) replaces step (2) with quantum-accelerated Monte Carlo methods. However, we observed that there are multiple ways to perform the decomposition in step (1). Notably, the most natural decomposition for classical MLMC is not optimal for quantum algorithms in our nested expectation problem. To address this, we developed a new decomposition strategy based on a different sequence of quantum-accelerated Monte Carlo subroutines, ultimately achieving the desired quadratic speedup. Thus, the "outside quantum" refers to the quantum-accelerated Monte Carlo algorithm in Step (2) , while the "inside quantum" denotes the quantum subroutines used in Step (1).
Besides addressing the nested expectation problem, we show that quantum computing introduces new flexibility to MLMC. By redesigning the MLMC subroutines to leverage quantum algorithms, additional gains can be achieved. As MLMC is widely applied in computational finance, uncertainty quantification, and engineering simulations, we believe that our strategy can also improve algorithmic efficiency in many of these applications.
Finally, we stress that our quantum-accelerated Monte Carlo method uses quantum-computing algorithms to speed up the estimation of classical stochastic problems; it should not be conflated with Quantum Monte Carlo (QMC) methods, which are a separate class of classical algorithms developed for simulating quantum many-body systems (see, for example, Foulkes et al. (2001)).
The remainder of the paper is organized as follows. Section 2 reviews the fundamentals of MLMC and QA-MLMC. Section 3 outlines the limitations of existing methods, motivates our approach, and presents the new algorithm together with a proof sketch. Section 4 shows how our method accelerates several practical problems that involve nested expectations. Section 5 summarizes the work, notes its limitations, and suggests directions for future research. Detailed proofs are provided in the Appendix.
2 Multilevel Monte Carlo 2.1 Classical MLMC MLMC methods are designed to estimate specific statistics of a target distribution, such as the expectation of a function of a Stochastic Differential Equation (SDE) (Giles, 2008), a function of an expectation (Blanchet and Glynn, 2015), or solutions to stochastic optimization problems, quantiles, and steady-state expectations (Blanchet et al., 2019). Users typically have access to a sequence of approximations, with each successive level offering reduced variance but increased computational cost. The following theorem provides theoretical guarantees for classical MLMC. Theorem 2.1 (Theorem 1 in Giles (2015)). Let µ denote a quantity that we want to estimate. Suppose for each integer l ≥ 0, we have an algorithm A l that outputs a random variable ∆ l with variance V l and computational cost C l . Define s l := l k=0 E[∆ k ] and assume for some (α, β, γ) with α ≥ 1 2 max{γ, β} that the following holds:
(i) |s l -µ| ≤ O(2 -αl ); (ii) V l ≤ O(2 -βl ); (iii) C l ≤ O(2 γl ).
Then for any fixed ϵ < 1/e, one can choose positive integers L, N 0 , . . . , N L depending on (α, β, γ), and construct the estimator µ := L l=0
1 N l N l i=1 ∆ (i) l satisfying E[( µ -µ) 2 ] < ϵ 2 with cost: L l=0 C l • N l =    O(ϵ -2 ), when β > γ O(ϵ -2 (log ϵ) 2 ), when β = γ O(ϵ -2-(γ-β)/α ), when β < γ, here ∆ (1) l , ∆ (2) l , . . . are i.i.d. copies of ∆ l for each l.
The MLMC estimator described in Theorem 2.1 can be rewritten as µ = L l=0 δ l , with each δ l :=
N l i=1 ∆ (i) l /N l acting as a Monte Carlo estimator for E[∆ l ].
The design and analysis of the sequence A l are central to apply Theorem 2.1 in every MLMC application. In the nested expectation problem, we follow Section 9.1 of Giles (2015) and provide a possible MLMC solution here. We define
A 0 as: 1) simulate X; 2) simulate Y given X; 3) output g(X, ϕ(X, Y )). When l ≥ 1, A l is defined in Algorithm 1. Algorithm 1 Classical MLMC for nested expectation: A l (l ≥ 1) 1: Generate X. 2: Generate Y 1 , . . . , Y 2 l conditional on X. 3: Set S o = i odd ϕ(X, Y i ), S e = i even ϕ(X, Y i ). 4: Output: g(X, 2 -l (S e + S o )) -g(X,2 -(l-1) Se)+g(X,2 -(l-1) So) 2 .
Under Assumptions 1-4 given in Section 3, one can check that the sequence of algorithms A l satisfies α = 0.5, β = γ = 1 in Theorem 2.1, with a proof given in Appendix B. Therefore Theorem 2.1 applies and one can estimate the nested expectation with cost O(ϵ -2 (log ϵ) 2 ).
this section cite: ['b0', 'b0', 'b8', 'b9', 'b3', 'b4', 'b10', 'b10']

Section: Quantum-accelerated MLMC in An et al. (2021)
In An et al. (2021), the authors propose a quantum-accelerated MLMC (QA-MLMC) algorithm that improves upon the classical MLMC from Theorem 2.1 in certain parameter regimes. The key insight is as follows: recall that δ l :=
N l i=1 ∆ (i) l /N l is a classical Monte Carlo estimator for E[∆ l ]
, meaning it estimates E[∆ l ] through i.i.d. sampling and then takes the average. This process can be accelerated by applying quantum-accelerated Monte Carlo (QA-MC) techniques, such as those discussed in Montanaro (2015). By replacing the classical Monte Carlo estimator δ l with its quantum counterpart, the following result is shown: Theorem 2.2 (Theorem 2 in An et al. (2021)). With the same assumptions as in Theorem 2.1, there is a quantum algorithm that estimates µ up to additive error ϵ with probability at least 1 -δ, and with cost O(ϵ -1 log(δ -1 )), when β ≥ 2γ O(ϵ -1-(γ-0.5β)/α log(δ -1 )), when β < 2γ.
Comparing QA-MLMC (Theorem 2.2) with classical MLMC (Theorem 2.1) shows only a quadratic speed-up-up to polylogarithmic factors-when β ≥ 2γ. In a typical scenario where β = γ = 1 and α ≥ 0.5, classical MLMC costs O(ϵ -2 ), whereas QA-MLMC costs O(ϵ -1-0.5/α ). In particular, QA-MLMC has cost O(ϵ -2 ) for the nested expectation problem, using the sequence of algorithms A l defined in Algorithm 1 (where α = 0.5). The loss of any quantum speedup prompts consideration of whether the algorithm can be improved. In general, the answer is no, as we show in Appendix C that the QA-MLMC algorithm of An et al. (2021) obeys the following lower bound. Proposition 2.3 (Lower bound for the general QA-MLMC). For any (α, β, γ), there is no quantum algorithm that can solve the problem stated in Theorem 2.1 for all sequences of algorithms {A l } l≥0 using fewer operations than Ω(ϵ -1 ) (when β ≥ 2γ) or Ω(ϵ -1-(γ-0.5β)/α ) (when β ≤ 2γ).
The rest of the paper is dedicated to showing that the nested expectation problem does not suffer from this lower bound, by presenting a faster quantum algorithm with cost O(ϵ -1 ), but tailored to this problem.
this section cite: ['b0', 'b28', 'b0', 'b0']

Section: Main Algorithm
The nested expectation E X [g(X, E Y |X [ϕ(X, Y )])] depends on g and ϕ. We define: Definition 3.1. A function g : X × R → R is uniformly K-Lipschitz in the second component if there exists a positive real number K such that for all x ∈ X and y 1 , y 2 ∈ R,
|g(x, y 1 ) -g(x, y 2 )| ≤ K|y 1 -y 2 |.
We pose five assumptions:
1. The function g is uniformly K-Lipschitz in its second component. 2. There is a number V known to the users satisfying E Y |x [ϕ(x, Y ) 2 ] ≤ V for every x ∈ X . 3. There is a number S known to the users satisfying Var X [g(X, E Y |X [ϕ(X, Y )])] ≤ S.
4. We assume that users can query both ϕ and g, and that each query incurs a unit cost. 5. We have access to the following two randomized classical algorithms Gen X and Gen Y , and every call of either one incurs a unit cost: We briefly review the five assumptions. The first assumption is critical for achieving the O(1/ϵ) upper bound in Theorem 3.2. This assumption holds in many practical problems of interest, particularly for functions like g(x, y) := max{x, y} or x + y. The second and third assumptions are technical conditions to ensure the variance and conditional second moment are well-behaved. For example, the second assumption is satisfied when (X, Y ) follows a regression model Y = f (X) + Noise where noise has zero mean and variance no more than V , and ϕ(x, y) is taken as y -f (x). The fourth assumption is commonplace and is consistently invoked-either explicitly or implicitly-in related works (Giles and Goda, 2019;Rainforth et al., 2018;Syed and Wang, 2023).
The final assumption requires that users have complete access to the sampling procedure itself-for instance, a Python script or pseudocode that produces the distribution-rather than relying on a black-box or API that only delivers samples. This assumption is common in the simulation literature, and holds in all applications we are aware of. Under Assumption 5, if we are given a randomized algorithm that generates a random variable X, we can convert it into a reversible algorithm (with constant factor overhead for the time complexity) and then implement it as a quantum circuit U : |0⟩ → x Pr[X = x]|x⟩|garbage x ⟩ using classical Toffoli and NOT gates (Bennett, 1973).
Here, |garbage x ⟩ denotes a work tape that stores the intermediate states of the sampling procedure, ensuring that the process remains reversible.
We show: Theorem 3.2. With assumptions 1-5 stated above, for any δ ∈ (0, 0.5), we can design an algorithm Q-NESTEXPECT with cost O(K
√ SV log(1/δ)/ϵ) that estimates E[g(X, E[ϕ(X, Y )])
] with an absolute error of no more than ϵ and a success probability of 1 -δ.
We highlight that the standard error metrics differ slightly between classical and quantum algorithms. Classical algorithms are often evaluated based on their expected error or mean-squared error, whereas quantum algorithms typically ensure that the error is small with high probability. The former metric is slightly stronger than the latter; however, under mild additional conditions, the two error measures become equivalent. Appendix A of An et al. (2021) has a detailed comparison between the two types of errors. Our error metric aligns with nearly all quantum algorithms, such as Brassard et al. (2002); Montanaro (2015); Kothari and O'Donnell (2023); Sidford and Zhang (2023).
this section cite: ['b11', 'b30', 'b33', 'b1', 'b0', 'b5', 'b28', 'b25', 'b32']

Section: Roadmap Why the quadratic speedup is lost?
A simple yet important observation is that quantum-accelerated Monte Carlo methods typically offer a quadratic improvement in sample complexity compared to classical Monte Carlo methods. However, this improvement does not automatically translate to a quadratic reduction in computational cost. This distinction arises because the computational cost depends not only on the sample complexity but also on the cost of executing the underlying algorithm.
Suppose the goal is to estimate the expectation of a randomized algorithm A, and the cost of a single execution of A is C(A). The computational cost of classical Monte Carlo requires O(σ 2 /ϵ 2 ) samples, where σ 2 represents the variance of A, and ϵ denotes the desired error tolerance. Consequently, the total computational cost becomes O(C(A)σ 2 /ϵ 2 ). In contrast, the QA-MC algorithm described in Kothari and O'Donnell (2023) (Theorem 3.5) reduces the sample requirement to O(σ/ϵ). This results in a computational cost of O(C(A)σ/ϵ). This implies that QA-MC achieves a nearly quadratic speedup in computational cost only if the cost of running the algorithm A is nearly constant, i.e., C(A) = O(1), or can itself be reduced quadratically by other methods.
The above quadratic reduction is possible when considering the standard nested Monte Carlo estimator for Equation (1), however this estimator is not competitive with the MLMC framework. It operates by estimating the inner expectation E Y |X=x [ϕ(X, Y )] with error O(ϵ/K) (which is sufficient by the Lipschitz property), nested withing an estimator of the outer expectation with error O(ϵ). The product of the sample complexity of the outer estimator A and the cost C(A) of the inner estimator results in an overall complexity of O(K 2 SV /ϵ 4 ) classically, or O(K √ SV /ϵ 2 ) quantumly. This is no better than the classical MLMC estimator with respect to ϵ. Now, consider the MLMC framework. Instead of using a single randomized algorithm A, it uses a sequence of algorithms A l corresponding to levels l = 0, 1, . . . , L. The computational cost C(A l ) generally grows exponentially with l. For example, C(A l ) ∼ 2 l in the MLMC solution for nested expectation problems (Algorithm 1). The highest level L is chosen as Θ(log(1/ϵ)), therefore the cost at the highest level is C(A L ) = O(poly(1/ϵ)).
The main idea from the QA-MLMC (An et al., 2021) is to replace each classical Monte Carlo estimator for E[∆ l ] with QA-MC. While this substitution works in principle, there is a critical issue as the level l approaches the maximum level L: the computational cost C l can grow significantly, sometimes as high as O(poly(1/ϵ)). This growth undermines the quadratic speedup achieved by QA-MLMC because the total cost is no longer dominated by O(1)-cost subroutines. Instead, the increasing C l makes the computational cost scale poorly with the desired error tolerance ϵ, leading to a loss of efficiency. Although the algorithm in An et al. (2021) leverages quantum-accelerated Monte Carlo to estimate E[∆ l ], the achieved speedup is less than quadratic because C(A l ) scales as O(poly(1/ϵ)) instead of O(1) at the highest levels. This growth in cost undermines the expected quadratic speedup.
this section cite: ['b0', 'b0']

Section: How to recover the quadratic speedup?
We can break down Theorem 2.1 to capture its core insights, then leverage them to refine our quantum algorithm. Theorem 2.1 provides two main ideas:
1. The target quantity is written as a limit of quantities with vanishing bias, and then reexpressed as a telescoping sum, µ = lim l→∞ s l = ∞ l=0 (s l+1 -s l ) + s 0 . The MLMC method estimates each level individually, optimizing resource allocation. 2. Estimator Variance Control: At each level l, an estimator ∆ l is designed with expectation E[∆ l ] = s l+1 -s l , and bias and variance diminishing with l, while cost increases with l.
In the classical solution (Algorithm 1), the sequence s l is
s l,classical := E g X, 2 -l 2 l i=1 ϕ(X, Y i ) .
By the law of large numbers, it is clear that s l,classical converges to the target. Algorithm 1 employs standard Monte Carlo to estimate s l,classical -s l-1,classical .
A key observation is that we can design a new sequence of quantum subroutines A l,quantum . They have a computational cost similar to that of the classical A l (in the sense of the γ parameter from Theorem 2.1), while achieving improved statistical properties (parameters α and β in Theorem 2.1).
To achieve this goal, we first define a new sequence s l,quantum that also converges to the target value. Its definition is based on using quantum-accelerated Monte Carlo to estimate the conditional expectation, with progressively higher precision as the sequence advances. Based on this new sequence, our A l,quantum can be naturally defined to estimate s l,quantum -s l-1,quantum . Finally, we can use existing quantum-accelerated Monte Carlo algorithms again to estimate E[∆ l,quantum ] at each level l. Together, these components achieve a quadratic speedup over MLMC, making this approach provably more efficient than a direct application of QA-MLMC.
A further distinction from An et al. (2021) is the choice of the quantum subroutine: whereas An et al. (2021) builds on the QA-MC algorithm in Montanaro (2015), we instead employ the newer QA-MC algorithm of Kothari and O'Donnell (2023). The latter demands less a priori information from the user (compared to Montanaro (2015)) and removes the extra polylogarithmic factors in the error bound ( compared to Hamoudi (2021)).
this section cite: ['b0', 'b0', 'b28', 'b25', 'b28']

Section: Proof agenda, and the median trick
The majority of our work is to show the following:
Proposition 3.3. Under the same assumption as Theorem 2.1, we can design an algorithm Q-NESTEXPECT-0.8 (Algorithm 4) with cost O(K
√ SV /ϵ) that estimates E[g(X, E[ϕ(X, Y )])
] with an absolute error of no more than ϵ and a success probability of 0.8. Proposition 3.3 specifies a fixed success probability of 0.8, whereas Theorem 3.2 allows an arbitrary value 1 -δ. Nevertheless, once we apply the "median trick" detailed below, we can tolerate an arbitrarily small failure probability. The proof of Lemma 3.4 is in Appendix A.4.
this section cite: []

Section: Lemma 3.4 (Median trick).
Given a set of random independent random variables X 1 , . . . , X n and an unknown deterministic quantity z, suppose that each X i satisfies Pr[X i ∈ (z -δ, z + δ)] ≥ 0.8, then Pr[median(X 1 , . . . , X n ) ∈ (z -δ, z + δ)] ≥ 1 -exp(-0.18n).
By invoking Lemma 3.4, we can implement the Q-NESTEXPECT algorithm of Theorem 3.2 by: Run Q-NESTEXPECT-0.8 (Algorithm 4) independently ⌈log(1/δ)/0.18⌉ times, and return the median of the resulting outputs.
this section cite: []

Section: Proving Theorem 3.2 based on Proposition 3.3 is given in Appendix A.4.

this section cite: []

Section: Our algorithm
To introduce our algorithm, we first define in Algorithm 2 a sequence of auxiliary algorithms B l .
Algorithm 2 B l (x) when l ≥ 0 1: Input: x from X.
2: Apply quantum-accelerated Monte Carlo Theorem 3.5 on Gen Y (x) and ϕ to estimate
E Y |X=x [ϕ(X, Y )]
with accuracy 2 -(l+1) /K and success probability at least 1 -2 -(2l+1) (4K 2 V ) -1 . 3: Clip the estimate into the region -√ V , √ V and denote the result by ϕ l (x).
4: Output: g(x, ϕ l (x)).
Algorithm B l with input x estimates g(x, E Y |X=x [ϕ(X, Y )]). Raising the value of l will improve the accuracy but also elevate the computational cost. The typical way we obtain input x for Algorithm B l is by generating it according to X using Gen X . We refer to this combined process as B l (X). The sequence {B l (X)} l≥0 produces random variables whose expectations progressively approximate the target E[g(X, E[ϕ(X, Y )])]. The properties of B l are studied in Lemma 3.6. Now we define A l based on B l . When l = 0, we set A 0 := B 0 (X). When l ≥ 1, we define A l in Algorithm 3.
Algorithm 3 A l when l ≥ 1 1: Sample x from X using Gen X . 2: Apply B l (x) to obtain g(x, ϕ l (x)). 3: Apply B l-1 (x) to obtain g(x, ϕ l-1 (x)). 4: Output: ∆ l := g(x, ϕ l (x)) -g(x, ϕ l-1 (x)).
Algorithm A l is a "coupled" difference of B l and B l-1 . It executes B l and B l-1 using the same input x. Compared to independently executing B l and B l-1 , this coupled implementation maintains the same expectation but ensures that the variance of ∆ l decreases to zero as l increases, which is beneficial for our objective. The properties of A l , and of its output ∆ l , are studied in Lemma 3.7.
this section cite: []

Section: Now we are ready to describe our solution to Proposition 3.3 in Algorithm 4.
Algorithm 4 Q-NESTEXPECT-0.8 1: Input: Accuracy level ϵ. Proof Sketch: To demonstrate that Algorithm 4 satisfies the guarantees stated in Proposition 3.3, we first prove a few lemmas studying the properties of B l and A l . The crucial property we will use from Kothari and O'Donnell (2023) is as follows: Theorem 3.5 (Theorem 1.1 in Kothari and O'Donnell (2023)). Given the access to an algorithm which outputs a random variable with mean µ and variance σ, there is a quantum algorithm that makes O((σ/ϵ) log(1/δ)) calls of the above algorithm, and outputs an estimate µ of µ. The estimate is ϵ-close to µ, that is | µ -µ| ≤ ϵ, with probability at least 1 -δ.
Now we study the computational and statistical properties of B l . The formal statements of the next two lemmas can be found in Appendices A.1 and A.2. Lemma 3.6 (Informal). For every l ≥ 0, with B l defined in Algorithm 2, we have:
1. Cost: For all x, the cost of implementing B l (x) is at most O(2 l K √ V )
2. Mean-squared error and Bias: For all x, the mean-squared error and the bias of the output of B l (x) with respect to g(x, E Y |X=x [ϕ(X, Y )]) are at most, respectively, 2 -2l and 2 -l .
this section cite: ['b25', 'b25']

Section: Variance:
The variance of the output of B l (X) is at most 2 + 2S.
Similarly, we study the properties of A l .
Lemma 3.7 (Informal). For any l ≥ 0, with A l defined in Algorithm 3, we have:
1. Cost: The cost of implementing A l is at most O(2 l K √ V ).
2. Bias: The bias of Optimality of our algorithm: Our algorithm achieves the optimal dependence on ϵ, up to polylogarithmic factors. In particular, when g(x, z) := z (which is 1-Lipschitz), the nested expectation in Equation ( 1) simplifies to E[ϕ]. Thus our problem includes standard Monte Carlo as a special case. Moreover, from known lower bounds on the quantum complexity of mean approximation (Nayak and Wu, 1999;Hamoudi, 2021), estimating the mean with error ϵ requires at least Ω(1/ϵ) operations. Our algorithm matches this dependence while extending to a broader class of problems.
this section cite: ['b29', 'b15']

Section: Applications
We now revisit the examples from Section 1.1 and analyze the cost of applying our algorithm.
Bayesian experiment design: Recall our target is
E X [log(E Y [Pr[X | Y ]])] -E Y [E X|Y [log(Pr[X | Y ])]].
The first term matches Equation ( 1 EVPPI: In the EVPPI example, our target quantity is
E X max k∈{1,...,d} E Y |X [f k (X, Y )] -max k∈{1,...,d} E[f k (X, Y )].
This problem requires a little extra care because the inner function is vector-valued in this case: ϕ(x, y) = (f 1 (x, y), . . . , f d (x, y)) and g(x, z) = max(z 1 , . . . , z d ). Ignoring the dependence on the dimension d, and adapting our estimator to the multidimensional case, one can estimate the first term at a cost of O(V /ϵ) assuming, for instance, that the family of functions f k is uniformly bounded V .
this section cite: []

Section: CoC option pricing:
We have g(x, z) = max{z -k, 0} with ϕ = g. Then, Assumption 1 holds with K = 1. Assuming that both X, Y are bounded between [-a, b] where a, b ≥ 0, and the strike price is k > 0, then Assumptions 2, 3 hold with S = V = max{b -k, a + k} 2 . Theorem 3.2 implies that Q-NESTEXPECT estimates the CoC option price ϵ error with cost O(max{b -k, a + k} 2 log(1/δ)/ϵ).
this section cite: []

Section: Conditional stochastic optimization:
Recall that the goal is to find an x that minimizes,
F (x) := E ξ [f (E η|ξ [g η (x, ξ)])].
Assuming the decision set is finite, X = {x 1 , . . . , x N }, and that each nested expectation F (x i ) satisfies our Assumptions 1-5, then Q-NESTEXPECT returns an additive-ϵ estimate of F at cost O(K √ SV /ϵ). This provides a function-value oracle for F . Using quantum minimum finding (Dürr and Høyer, 1996), the cost to output an x with F (x) ≤ min i F (
x i ) + ε is O(K √ SV N /ϵ).
this section cite: ['b7']

Section: Limitations and Future directions
First, although our algorithm achieves a near-optimal cost for estimating Equation (1) among quantum algorithms, our results are derived under fault-tolerance assumptions and thus remain difficult to realize on current NISQ hardware (Huang et al., 2024). Second, the original QA-MLMC algorithm applies to a wider range of scenarios than estimating the nested expectation. Thus, our contribution should be seen as an improvement of QA-MLMC within a specific yet critical set of problems.
Nevertheless, our work highlights how quantum computing can give MLMC much more freedom to invent novel approximation sequences. In favorable scenarios, like ours, a carefully crafted sequence can lead to a significantly improved (and potentially optimal) algorithm. This raises an intriguing question: which other MLMC applications could achieve a similar quantum quadratic speedup?
We expect that our technique can be extended to closely related problems. For instance, with additional assumptions, classical MLMC also has O(1/ϵ 2 ) cost when g(x, z) = g(z) = 1 z≥0 is a Heaviside function (Giles and Haji-Ali, 2019;Gordy and Juneja, 2010). This quantity plays a key role in computing Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR), which are widely used in finance. Similarly, MLMC has been shown to be effective in conditional stochastic optimization, bilevel optimization problems (Hu et al., 2020(Hu et al., , 2021(Hu et al., , 2024)), which is an extension of our setting with additional optimization steps. We expect our approach can be adapted to address these problems.
Our primary focus is the dependence on ϵ, which is also central to both the classical (Giles, 2015) and quantum MLMC (An et al., 2021) literature. However, our problem also depends on other parameters, such as the Lipschitz constant and the variance. It can readily be extended to vector-valued random variables by using a quantum multivariate estimator (Cornelissen et al., 2022), thereby incurring a dependence on the dimension as well. An interesting open question is whether our current algorithm is optimal with respect to these parameters, or whether improvements are possible.
this section cite: ['b24', 'b12', 'b13', 'b23', 'b21', 'b22', 'b10', 'b0', 'b6']

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction accurately state the main contributions of the paper, including both the algorithmic and theoretical contributions and its scope.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We have discussed the limitations in the conclusion and the introduction.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We state the results in the main text, and provide the proofs in the appendix.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [NA] . Justification: It does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [NA] .
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] .
Justification: The research complies fully with the NeurIPS Code of Ethics. It does not involve sensitive data, human subjects, or potentially harmful applications.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] .
Justification: This paper presents work whose goal is to advance the field of Quantum Computing, and Monte Carlo methods. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper does not introduce or release any new models or datasets that could pose a risk of misuse.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA] .
Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core methods and contributions of the paper do not involve the use of LLM in any important, original, or non-standard way. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Proofs for the Main Algorithm
A.1 Analysis of the B l algorithms Lemma (Formal statement of Lemma 3.6). The algorithm B l defined in Algorithm 2 for all l ≥ 0 has the following properties:
1. (Cost) For any input x, the cost of implementing B l (x) is O(2 l K √ V ).
this section cite: []

Section: (MSE)
For any input x, the mean-squared error (MSE) of the output g(x, ϕ l (
x)) of B l (x) with respect to g(x, E Y |X=x [ϕ(X, Y )]) is at most, E g(x, ϕ l (x)) -g(x, E Y |X=x [ϕ(X, Y )]) 2 ≤ 2 -2l .
this section cite: []

Section: (Bias)
For any input x, the bias of the output g(
x, ϕ l (x)) of B l (x) with respect to g(x, E Y |X=x [ϕ(X, Y )]) is at most, E[g(x, ϕ l (x))] -g(x, E Y |X=x [ϕ(X, Y )]) ≤ 2 -l .
this section cite: []

Section: (Variance)
The variance of the output of B l when the input is sampled according to X is at most
Var X [g(X, ϕ l (X))] ≤ 2 + 2S.
Proof of Lemma 3.6. We prove the properties separately.
Cost of B l . Implementing Step 2-3 of Algorithm 2 incurs a computational cost of O(2 l K √ V ).
To see this, note that it would be sufficient to take O(2 l K √ V ) samples to have success probability 0.8 using Kothari and O'Donnell (2023). When we increase the probability of success to 1 -2 -(2l+1) (4K 2 V ) -1 , we need to multiply this with a factor of O(log(2 2l+1 4K 2 V )).
Thus the total cost is O(2 l K √ V ). MSE: Let ϕ l (x) be the output of Step 2 in B l (x), i.e., the output of the quantum-accelerated Monte Carlo algorithm in Kothari and O'Donnell (2023) before clipping. We claim
ϕ l (x) -E Y |X=x [ϕ(X, Y )] ≥ ϕ l (x) -E Y |X=x [ϕ(X, Y )] ,
i.e., clipping will never increase the error. To see this, notice that
E Y |X=x [ϕ(X, Y )] ≤ √ V because E[Z] 2 ≤ E[Z 2 ]
for any random variable Z. Therefore, we know as a-priori that the target quantity of step 2 is between -V and V . Suppose
ϕ l (x) is between [-V, V ], then ϕ l (x) = ϕ l (x), i.e., no clipping happens. Suppose ϕ l (x) > V , then ϕ l (x) -E Y |X=x [ϕ(X, Y )] ≥ √ V -E Y |X=x [ϕ(X, Y )] = ϕ l (x) -E Y |X=x [ϕ(X, Y )] .
Similar argument holds when ϕ l (x) < -V . Thus, the squared error either decreases or remains unchanged after clipping, establishing the claim.
Since ϕ l (x) has accuracy 2 -(l+1) /K with probability at least 1-2 -(2l+1) (4K 2 V ) -1 , we know ϕ l (x) has the same property. Moreover, the absolute error between ϕ l (x) and E Y |X=x [ϕ(X, Y )] can be no more than 2 √ V due to clipping. Thus, the expected squared error between ϕ l (x) and
E Y |X=x [ϕ(X, Y )] is upper bounded by E ϕ l (x) -E Y |X=x [ϕ(X, Y )] 2 ≤ K -2 2 -2(l+1) × 1 + 4V × 2 -(2l+1) (4K 2 V ) -1 ≤ K -2 2 -2l .
For each fixed x, the absolute error between g(x, ϕ l (x)) and g(
x, E Y |x [ϕ(x, Y )]) is not greater than K| ϕ l (x) -E Y |X=x [ϕ(X, Y )]
| due to the Lipschitzness of g. Therefore, we have the following estimate on the mean-squared error of B l with input x:
E g(x, ϕ l (x)) -g(x, E Y |X=x [ϕ(X, Y )]) 2 ≤ 2 -2l .(3)
Bias: The bias of the output of algorithm B l on input x, with respect to g(
x, E Y |X=x [ϕ(X, Y )]), is E[g(x, ϕ l (x))] -g(x, E Y |X=x [ϕ(X, Y )]) ≤ E g(x, ϕ l (x)) -g(x, E Y |X=x [ϕ(X, Y )]) ≤ E g(x, ϕ l (x)) -g(x, E Y |X=x [ϕ(X, Y )]) 2 ≤ 2 -l .
Here the first two inequalities follows from
|E[X] -a| ≤ E[|X -a|] ≤ E[|X -a| 2 ], and the last inequality follows from Equation (3). Variance: To simplify the notation, define
G(x) as g(x, E Y |X=x [ϕ(X, Y )]). Then Assumption 3 reads Var X [G(X)] ≤ S.
We bound the variance of the output of B l (X) as:
Var X [g(X, ϕ l (X))] ≤ E X [(g(X, ϕ l (X)) -E X [G(X)]) 2 ] ≤ 2E X [(g(X, ϕ l (X)) -G(X)) 2 ] + 2E X [(G(X) -E X [G(X)]) 2 ] ≤ 2E x∼X E[(g(x, ϕ l (x)) -G(x)) 2 ] + 2Var X [(G(X)]
≤ 2 • 2 -2l + 2S ≤ 2 + 2S, where the first inequality uses the fact that Var[X] = min a E[(X -a) 2 ] and the last inequality uses Equation (3).
this section cite: ['b25']

Section: A.2 Analysis of the A l algorithms
Lemma (Formal statement of Lemma 3.7). The algorithm A l defined in Algorithm 3 for all l ≥ 0 has the following properties:
1. (Cost) The cost of implementing A l is O(2 l K √ V ).
this section cite: []

Section: (Bias)
The bias of the sum
l k=0 ∆ k of outputs up to level l is at most 2 -l with respect to E X [g(X, E Y |X [ϕ(X, Y )])], i.e.: l k=0 E[∆ k ] -E X [g(X, E Y |X [ϕ(X, Y )])] ≤ 2 -l .
this section cite: []

Section: (Variance)
The variance of the output ∆ l of A l is at most 2 + 2S when l = 0 and 10 × 2 -2l when l ≥ 1.
Proof of Lemma 3.7. Cost of A l . Clearly implementing A l once costs around twice of the cost of B l . Therefore the claims follows from Lemma 3.6.
Bias of l k=0 ∆ k . Since the expectation of ∆ k when k ≥ 1 is the difference in expectations between the outputs of B k (X) and B k-1 (X), we know the expectation of l k=0 ∆ k is a telescoping sum that is equal to the expectation of the output of the last algorithm B l (X). Therefore, the bias is
l k=0 E[∆ k ] -E[g(X, E[ϕ(X, Y )])] = E X [g(X, ϕ l (X)) -g(X, E Y |X [ϕ(X, Y )])] ≤ E x∼X E[g(x, ϕ l (x))] -g(x, E Y |X=x [ϕ(X, Y )]) ≤ 2 -l
where the last inequality follows from the bias established in Lemma 3.6.
Variance of ∆ l . The case l = 0 follows from Lemma 3.6 and the definition A 0 = B 0 (X).
When l ≥ 1, we can bound the second moment as follows:
E[∆ 2 l ] = E x∼X E g(x, ϕ l (x)) -g(x, ϕ l-1 (x)) 2 ≤ 2E x∼X E g(x, ϕ l (x)) -g(x, E Y |X=x [ϕ(X, Y )]) 2 + 2E x∼X E g(x, ϕ l-1 (x)) -g(x, E Y |X=x [ϕ(X, Y )]) 2 ≤ 2(2 -2l + 2 -2l+2 ) = 10 × 2 -2l
where the last inequality follows from the mean-squared error established in Lemma 3.6. Since the second moment is always no less than the variance, our claim on Var[∆ l ] is proven.
this section cite: []

Section: A.3 Analysis of the Q-NESTEXPECT-0.8 algorithm
Proof of Proposition 3.3. We claim Q-NESTEXPECT-0.8 (Algorithm 4) outputs an estimator that is ϵ-close to E[g(X, E[ϕ(X, Y )])] with probability at least 0.8.
For each l, our failure probability is no larger than 0.1 l+1 . Therefore the probability of at least one failure in the for-loop of Algorithm 4 is at most 0.1 + 0.1 2 + ... ≤ 0.2 by union bound. Thus there is a probability at least 1 -(0.1 + 0.1 2 + ...) ≥ 0.8 such that δ l is ϵ/(2L + 2)-close to E[∆ l ] for every l. Under this (good) event, our final estimator has error
L l=0 δ l - L l=0 E[∆ l ] ≤ L l=0 δ l -E[∆ l ] ≤ (L + 1) ϵ 2L + 2 = ϵ 2 .
At the same time, Lemma 3.7 shows
L l=0 E[∆ l ] is 2 -L -close to E[g(X, E[ϕ(X, Y )])]. Using L = ⌈log 2 (2/ϵ)⌉, we conclude our estimator has error L l=0 δ l -E[g(X, E[ϕ(X, Y )])] ≤ L l=0 δ l - L l=0 E[∆ l ] + L l=0 E[∆ l ] -E[g(X, E[ϕ(X, Y )])] ≤ ϵ,
with probability at least 0.8. The cost of our algorithm is the summation of the costs of L + 1 different quantum-accelerated Monte Carlo algorithms. For l = 0, since A 0 = B 0 (X), the variance of ∆ 0 is at most 2 + 2S by Lemma 3.6. Thus, Theorem 3.5 shows that the cost of estimating E[∆ 0 ] with error ϵ/(2L + 2) is O( √ SL/ϵ) times the cost O(K √ V ) of implementing B 0 (X), which gives O(LK √ SV /ϵ). For l ≥ 1, the variance of ∆ l is at most 10 × 2 -2l by Lemma 3.7. Hence, the cost of estimating
E[∆ l ] is O(2 -l L/ϵ) times the cost O(2 l K √ V ) of implementing A l , which gives O(LK √ V /ϵ).
Summing up all the costs together, and using that L = O(log(1/ϵ)), yields a total complexity of
O(LK √ SV /ϵ + L 2 K √ V /ϵ) = O(K √ SV /ϵ).
this section cite: []

Section: A.4 Median trick and main theorem
Proof of Lemma 3.4 (Median trick). Let
Y i = 1, if X i / ∈ (z -δ, z + δ),0, otherwise,
i = 1, . . . , n, and set S n = n i=1 Y i . Because each X i lies in the interval (z -δ, z + δ) with probability at least 0.8, we have Pr[Y i = 1] ≤ 0.2 and hence E[S n ] = n i=1 E[Y i ] ≤ 0.2n.
The sample median falls outside the interval (z -δ, z + δ) if and only if more than half of the observations do so, i.e., if S n > 1 2 n. Consequently,
Pr[median(X 1 , . . . , X n ) / ∈ (z -δ, z + δ)] = Pr S n > 1 2 n ≤ Pr[S n -E[S n ] ≥ 0.3n].
Because the Y i are independent Bernoulli variables, Hoeffding's inequality gives
Pr[S n -E[S n ] ≥ t] ≤ exp(-2t 2 /n), t > 0.
Taking t = 0.3n yields
Pr[S n > 1 2 n] ≤ exp(-2(0.3n) 2 /n) = exp(-0.18n).
Therefore,
Pr[median(X 1 , . . . , X n ) ∈ (z -δ, z + δ)] = 1 -Pr[median(X 1 , . . . , X n ) / ∈ (z -δ, z + δ)] ≥ 1 -exp(-0.18n), as claimed.
The proof of Theorem 3.2 follows immediately by applying the median trick to the algorithm given in Proposition 3.3.
Proof of Theorem 3.2. The Q-NESTEXPECT algorithm was defined as, Run Q-NESTEXPECT-0.8 (Algorithm 4) independently ⌈log(1/δ)/0.18⌉ times, and return the median of the resulting outputs.
Since the output of Q-NESTEXPECT-0.8 is ϵ-close to Equation (1) with probability at least 0.8 (Proposition 3.3), the median trick (Lemma 3.4) implies that the output of Q-NESTEXPECT is ϵ-close to Equation (1) with probability at least 1 -exp(-0.18⌈log(1/δ)/0.18⌉) ≥ 1 -δ. The cost of Q-NESTEXPECT is also ⌈log(1/δ)/0.18⌉ times the cost of Q-NESTEXPECT-0.8, and is thus O(K √ SV log(1/δ)/ϵ), as claimed.
this section cite: []

Section: B Classical MLMC Parameters
Under Assumptions 1-4 in Section 3, we verify that the classical MLMC sequence defined in Algorithm 1 corresponds to α = 0.5, β = γ = 1 in the MLMC framework (Theorem 2.1) when K, V = O(1) are constant numbers. Firstly, the computational cost of A l is clearly O(2 l ), thus γ = 1. To analyze the variance, it is useful to observe
g X, 1 2 l (S e + S o ) - g(X, 1 2 l-1 S e ) + g(X, 1 2 l-1 S o ) 2 ≤ K 2 l-1 |S e -S o |
because of the Lipschitz continuity. Let Z i = ϕ(X, Y 2i-1 ) -ϕ(X, Y 2i ), it is then clear that the Z i 's have zero mean and are conditionally independent given X. Further,
Var[Z i | X] = Var[ϕ(X, Y 2i-1 ) | X] + Var[ϕ(X, Y 2i ) | X] = 2Var Y [ϕ(X, Y ) | X] ≤ 2V. Therefore E[|S e -S o | 2 ] = Var[S e -S o ] = Var   2 l-1 i=1 Z i   ≤ 2 l V.
this section cite: []

Section: 
Answer: [NA] . Justification: The paper does not include experiments requiring code. Guidelines:
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
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] . Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] . Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets ( if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage
Thus E g X, 1 2 l (S e + S o ) - g X, 1 2 l-1 S e + g X, 1 2 l-1 S o 2 2 ≤ K 2 2 2l-2 E |S e -S o | 2 ≤ K 2 2 2l-1 2 l V = 2K 2 V 2 l = O(2 -l ).
Therefore β = 1, as claimed.
Finally we study the bias, which is
b := l k=0 E[∆ k ] -E[g(X, E[ϕ(X, Y )])] .
where
∆ 0 = g(X, ϕ(X, Y )) and ∆ k = g(X, 2 -k (S e + S o )) -g(X,2 -(k-1) Se)+g(X,2 -(k-1) So) 2 when k ≥ 1. We obtain a telescoping sum simplifying to l k=0 E[∆ k ] = E X g X, 1 2 l (S e + S o ) = E X,Y1,...,Y 2 l g X, 2 l k=1 ϕ(X, Y k ) 2 l .
Therefore, the bias is at most
b ≤ E X,Y1,...,Y 2 l g X, 2 l k=1 ϕ(X, Y k ) 2 l -g(X, E Y |X [ϕ(X, Y )])
(by the triangle inequality)
≤ KE X,Y1,...,Y 2 l 2 l k=1 ϕ(X, Y k ) 2 l -E Y |X [ϕ(X, Y )] (since g is K-Lipschitz) ≤ KE X   E Y1,...,Y 2 l |X 2 l k=1 ϕ(X, Y k ) 2 l -E Y |X [ϕ(X, Y )] 2   (by the Cauchy-Schwarz inequality) = K 2 l/2 E X Var Y |X [ϕ(X, Y )] ≤ K √ V 2 l/2
(by Assumption 2 in Section 3)
Thus, we have α = 0.5, as claimed.
this section cite: []

Section: C Lower Bound for the QA-MLMC Algorithm of An et al. (2021)
We demonstrate Proposition 2.3, namely that the QA-MLMC algorithm of An et al. (2021) is optimal (up to polylogarithmic factors) for solving the problem stated in Theorem 2.1 when the sequence {A l } l≥0 is arbitrary. We consider two parameter regimes for (α, β, γ) separately.
Regime β ≤ 2γ. In that regime, the complexity of the quantum MLMC algorithm of An et al. (2021) is O(ϵ -1-(γ-0.5β)/α ). We observe that the most expensive part of the computation in this algorithm occurs at the last level L = O(log(1/ϵ)/α), where the algorithm uses O(ϵ -1+0.5β/α ) samples, each with cost C L = O(ϵ -γ/α ). We take inspiration from this behavior to design a hard instance for the lower bound. The lower bound proceeds by a reduction to the following query problem:
Definition C.1 (COUNT t,n • PARITY m ). Fix three integers t, n, m with t ≤ n. Define two sets of Boolean matrices M 0 , M 1 ⊂ {0, 1} n×m such that:
• M ∈ M 0 iff each row of M has parity 0 (i.e., j M i,j = 0 mod 2 for all i ∈ [n]),
• M ∈ M 1 iff exactly t rows of M have parity 1.
In the COUNT t,n •PARITY m problem, one is given access to a quantum oracle |i, j⟩ → (-1) Mi,j |i, j⟩ for an unknown M ∈ M 0 ∪ M 1 , and the goal is to decide to which set M belongs to.
By composition properties of the quantum adversary method, the quantum query complexity of COUNT t,n • PARITY m is the product of the complexity of the counting problem COUNT t,n (which is Ω( n/t)) and of the parity problem PARITY m (which is Ω(m)). We refer the reader to (van Apeldoorn, 2021, Lemma 11) for a more detailed version of this argument. Proposition C.2. Any quantum algorithm that solves the COUNT t,n • PARITY m problem with success probability at least 2/3 must make Ω(m n/t) queries to M .
We now explain how to construct a hard instance for the MLMC problem out of an input to the COUNT t,n • PARITY m problem. Fix (α, β, γ) such that α ≥ 1 2 max{γ, β} and β ≤ 2γ. Choose t, n, m such that m = C L = ϵ -γ/α and t/n = ϵ 2 /V L = ϵ 2-β/α . Given M ∈ M 0 ∪ M 1 , define the random variable P M ∈ {0, ϵ β/α-1 } obtained by sampling i ∈ [n] uniformly at random and setting
P M = 0, if PARITY(M i ) = 0, ϵ β/α-1 , if PARITY(M i ) = 1.
We consider the problem of estimating the expectation µ = E[P M ] with error ϵ/2. Observe that µ = 0 if M ∈ M 0 and µ = t/nϵ β/α-1 = ϵ if M ∈ M 1 . Hence, estimating µ ± ϵ/2 allows to solve the COUNT t,n • PARITY m problem on M . By Proposition C.2, this cannot be done faster than with Ω(m n/t) = Ω(ϵ -1-(γ-0.5β)/α ) quantum queries.
It remains to construct a sequence of random variables ∆ l with parameters (α, β, γ) that can approximate P M . All random variables are chosen to be zero, except ∆ L which is chosen to be P M (where
L = log(1/ϵ)/α). The cost to compute ∆ L (naively) is C L = m = ϵ -γ/α = 2 γL . The variance is Var[∆ L ] = Var[P M ] ≤ E[P 2 M ] ≤ ϵ β/α = 2 -βL .
The cost and variance of ∆ l are zero when l ̸ = L. Finally, for any l we have,
l k=0 E[∆ k ] -µ = 0, if l ≥ L, µ ≤ ϵ ≤ 2 -αl , if l < L.
Hence, we have constructed a sequence of algorithms {A l } l≥0 satisfying the conditions listed in Theorem 2.1, but for which no quantum algorithm can solve the estimation problem using fewer than Ω(ϵ -1-(γ-0.5β)/α ) quantum queries.
Regime β ≥ 2γ. In that regime, the complexity of the quantum MLMC algorithm of An et al. (2021) is O(ϵ -1 ). The most expensive part of the computation occurs at the first level l = 0, where the algorithm uses O(ϵ -1 ) samples, each with cost C L = O(1). This is trivial to adapt into a lower bound (in fact, it applies to the regime β ≤ 2γ as well). We know that for standard Monte Carlo, there exists a quantum query problem (Nayak and Wu, 1999;Hamoudi, 2021) for which generating a certain random variable X has cost O(1) and variance Var[X] = O(1), and the quantum complexity of estimating E[X] ± ϵ is Ω(1/ϵ). This can be cast as an MLMC problem by considering only the first level: ∆ 0 = X. We have l k=0 E[∆ k ] -E[X] = 0 for all l, and C l = V l = 0 for all l > 0. Since this sequence applies to any values of (α, β, γ), we have a generic lower bound of Ω(1/ϵ).
this section cite: ['b0', 'b0', 'b0', 'b29', 'b15']

Section: References
Ref_id:b0 Title: Quantum-accelerated multilevel Monte Carlo methods for stochastic differential equations in mathematical finance Year: (2021)
Ref_id:b1 Title: Logical reversibility of computation Year: (1973)
Ref_id:b2 Title: Connecting quantum computing with classical stochastic simulation Year: (2025)
Ref_id:b3 Title: Unbiased Monte Carlo for optimization and functions of expectations via multi-level randomization Year: (2015)
Ref_id:b4 Title: Unbiased multilevel Monte Carlo: Stochastic optimization, steady-state simulation, quantiles, and other applications Year: (2019)
Ref_id:b5 Title: Quantum amplitude amplification and estimation Year: (2002)
Ref_id:b6 Title: Near-optimal quantum algorithms for multivariate mean estimation Year: (2022)
Ref_id:b7 Title: A quantum algorithm for finding the minimum Year: (1996)
Ref_id:b8 Title: Quantum Monte Carlo simulations of solids Year: (2001)
Ref_id:b9 Title: Multilevel Monte Carlo path simulation Year: (2008)
Ref_id:b10 Title: Multilevel Monte Carlo methods Year: (2015)
Ref_id:b11 Title: Decision-making under uncertainty: using mlmc for efficient estimation of evppi Year: (2019)
Ref_id:b12 Title: Multilevel nested simulation for efficient risk estimation Year: (2019)
Ref_id:b13 Title: Nested simulation in portfolio risk measurement Year: (2010)
Ref_id:b14 Title: Nested multilevel Monte Carlo with biased and antithetic sampling Year: (2023)
Ref_id:b15 Title: Quantum sub-gaussian mean estimator Year: (2021)
Ref_id:b16 Title: Quantum Chebyshev's inequality and applications Year: (2019)
Ref_id:b17 Title: Debiasing conditional stochastic optimization Year: (2024)
Ref_id:b18 Title: Quantum summation with an application to integration Year: (2002)
Ref_id:b19 Title: An efficient estimation of nested expectations without conditional sampling Year: (2023)
Ref_id:b20 Title: Estimating the mean of a non-linear function of conditional expectation Year: (2009)
Ref_id:b21 Title: On the bias-variance-cost tradeoff of stochastic optimization Year: (2021)
Ref_id:b22 Title: Contextual stochastic bilevel optimization Year: (2024)
Ref_id:b23 Title: Biased stochastic first-order methods for conditional stochastic optimization and applications in meta learning Year: (2020)
Ref_id:b24 Title: Evaluating a quantum-classical quantum Monte Carlo algorithm with matchgate shadows Year: (2024)
Ref_id:b25 Title: Mean estimation when you have the source code; or, quantum Monte Carlo methods Year: (2023)
Ref_id:b26 Title: On a measure of the information provided by an experiment Year: (1956)
Ref_id:b27 Title: Computing Bayes: From then 'til now Year: (2023)
Ref_id:b28 Title: Quantum speedup of Monte Carlo methods Year: (2015)
Ref_id:b29 Title: The quantum query complexity of approximating the median and related statistics Year: (1999)
Ref_id:b30 Title: On nesting Monte Carlo estimators Year: (2018)
Ref_id:b31 Title: Buffon's noodle problem Year: (1969)
Ref_id:b32 Title: Quantum speedups for stochastic optimization Year: (2023)
Ref_id:b33 Title: Optimal randomized multilevel Monte Carlo for repeatedly nested expectations Year: (2023)
Ref_id:b34 Title: Quantum probability oracles & multidimensional amplitude estimation Year: (2021)
Ref_id:b35 Title: Unbiased optimal stopping via the MUSE Year: (2023)
