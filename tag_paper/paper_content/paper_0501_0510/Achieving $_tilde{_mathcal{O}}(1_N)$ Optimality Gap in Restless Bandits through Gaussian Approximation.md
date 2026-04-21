Title: Achieving Õ(1/N ) Optimality Gap in Restless Bandits through Gaussian Approximation
Abstract: We study the finite-horizon Restless Multi-Armed Bandit (RMAB) problem with N homogeneous arms. Prior work has shown that when an RMAB satisfies a non-degeneracy condition, Linear-Programming-based (LP-based) policies derived from the fluid approximation, which captures the mean dynamics of the system, achieve an exponentially small optimality gap. However, it is common for RMABs to be degenerate, in which case LP-based policies can result in a Θ(1/ √ N ) 1 optimality gap per arm. In this paper, we propose a novel Stochastic-Programmingbased (SP-based) policy that, under a uniqueness assumption, achieves an Õ(1/N ) optimality gap for degenerate RMABs. Our approach is based on the construction of a Gaussian stochastic system that captures not only the mean but also the variance of the RMAB dynamics, resulting in a more accurate approximation than the fluid approximation. We then solve a stochastic program for this system to obtain our policy. This is the first result to establish an Õ(1/N ) optimality gap for degenerate RMABs.

Section: Introduction
The Restless Multi-Armed Bandit (RMAB) problem is an important framework in sequential decisionmaking, where a decision maker selects a subset of tasks (arms) to work on (pull) at each time step to maximize cumulative rewards, under known model parameters [29]. Unlike the classical (restful) bandit [11], in the restless variant, the state of each arm evolves stochastically regardless of whether it is pulled. RMABs have been widely applied in domains such as machine maintenance [9,12], healthcare resource allocation [22,23], and target tracking [19,21], to name a few, where optimal decision-making under uncertainty is critical. A general RMAB is PSPACE-hard [25], and finding optimal policies is computationally challenging, especially as the number of arms grows. Recently, there have also been efforts to use deep learning and reinforcement learning to learn heuristic policies for RMABs, such as [1,18,24,30,31].
In this paper, we focus on the finite-horizon version of the RMAB problem with N homogeneous arms and horizon H, where each arm follows the same (time-dependent, known) state transition and reward function. While computing the exact optimal policy is impractical, the homogeneity of the model allows for the design of computationally efficient policies. One such class of policies is based on fluid approximation, which transforms the original N -armed RMAB problem into a Linear Program (LP), and an LP-based policy can be efficiently computed based on the solution to the LP. Optimality gap. In [15], an LP-based index policy was proposed and achieves an o(1) optimality gap. This gap was later improved to O(log N/ √ N ) in [34], and subsequently to O(1/ √ N ) in [7]. In the numerical experiments of [7], it was observed that while this O(1/ √ N ) gap appears tight for certain problems, in others the gap converges to zero more rapidly. This empirical observation was theoretically confirmed in [36], where it was shown that under a non-degenerate condition (formally defined in Definition 3.1), the gap is at a smaller order of O(1/N ).
Non-degenerate condition. This non-degenerate condition has since become a key assumption in subsequent works. [10] shows that, under this condition, the optimality gap becomes exponentially small when the rounding error induced by scaling the fluid approximation by N is eliminated. Further generalizations of the non-degenerate condition have been made in [8], extending it to multi-action and multi-constraint RMABs (also known as weakly coupled Markov decision processes). In [35], the non-degenerate condition was further extended to settings with heterogeneous arms.
Prevalence of degenerant RMABs. Despite the central role played by the non-degenerate condition, many RMAB problems are degenerate. Notable examples, which originate from real-world applications, have been discussed in [7,8,36]. Moreover, a numerical study presented in Appendix B.1 revealed that a significant proportion (about 50% for some cases) of randomly generated RMABs are degenerate, and almost all of them satisfy the Uniqueness Assumption 4.1. This highlights the practical importance of addressing degenerate RMABs.
However, to the best of our knowledge, in all previous works, when an RMAB is degenerate, the best known optimality gap is O(1/ √ N ). It is widely believed that this upper bound is also order-wise tight under LP-based policies, making it Θ(1/ √ N ). We will formally prove this result in Theorem 4.2 by using an example.
These results led us to ask the following central question of this paper: Does there exist a computationally efficient algorithm for degenerate RMABs with an optimality gap order-wise smaller than O(1/ √ N )? Contributions. This paper answers the question affirmatively and includes the following results:
• We construct a Gaussian stochastic system (see (14)) that more accurately captures the behavior of the N -system than the fluid system. Unlike the fluid system, which serves as a first-order approximation to the stochastic N -system, the Gaussian stochastic system incorporates both the mean and variance of the N -system.
• We demonstrate that the SP-based policy obtained from the Gaussian stochastic system (see Algorithm ( 1 • We further compliment our main result by presenting a degenerate example in Section 3, demonstrating in Theorem 4.2 that not only the LP-based policy has Θ(1/ √ N ) optimality gap, but also the LP upper bound has Θ(1/ √ N ) gap from the optimal value. This indicates that the LP upper bound, which is a widely used baseline, itself is not tight for degenerate RMABs.
As an illustration, consider a degenerate RMAB example with 2 states and a horizon of 2 steps (the details can be found in Section 3.3). Given the small problem size, the optimal policy can be computed exactly through brute-force methods. As shown in Figure 1, the performance of the policy obtained by solving the Gaussian stochastic system is already very close to the true optimal value. Considering N ×optimality gap (i.e. the total optimality gap of the N arms with respect to the optimal policy), it remains bounded under the SP-based policy as N increases, which confirms the Õ(1/N ) optimality gap; while the total optimality gap for the LP-based policy appears to grow unbounded.
this section cite: ['b28', 'b10', 'b8', 'b11', 'b21', 'b22', 'b18', 'b20', 'b24', 'b0', 'b17', 'b23', 'b29', 'b30', 'b14', 'b33', 'b6', 'b6', 'b35', 'b9', 'b7', 'b34', 'b6', 'b7', 'b35', 'b13']

Section: Related work.
A comparison of state-of-the-art results in finite-horizon RMABs with this work is provided in Table 1. Gaussian approximations, rooted in the central limit theorem, are widely used to approximate stochastic processes. In heavy-traffic queueing analysis, this yields the diffusion approximation, where the centered, scaled limit is a diffusion process; see early work in [4,5,16,17]. This literature typically studies fixed policies and establishes convergence (or rates) to the diffusion limit. In contrast, we aim to design near-optimal policies for discrete-time RMABs via a second-order Gaussian approximation.
Closer to our setting is approximate diffusion control (e.g., [2,13,14]), which replaces the original problem with a diffusion control problem solved through the Hamilton-Jacobi-Bellman (HJB) equation. Recent work [6] shows HJBs arise from second-order value-function approximations of general MDPs. However, solving HJBs is notoriously difficult, especially when diffusion coefficients are control dependent, yielding highly nonlinear PDEs. This actually motivates our alternative second-order approach, detailed in Section 3.
this section cite: ['b3', 'b4', 'b15', 'b16', 'b1', 'b12', 'b13', 'b5']

Section: Notational convention.
Vectors are row vectors by default. Throughout the paper, we consider three systems: the N -armed RMAB system, the fluid system, and the Gaussian stochastic system. As a general convention, variables and functions with " ¯" are for the fluid system, while those with " ˜" are for the Gaussian stochastic system. The terms "action" and "control" are used interchangeably in the context of decision-making.
2 Problem formulation RMAB model. We consider the H-horizon Restless Multi-Armed Bandit (RMAB) problem with N homogeneous arms. Each arm is modeled as a Markov Decision Process (MDP) with state space S := {1, 2, . . . , S} and action space A := {0, 1}. At each time step h (1 ≤ h ≤ H), the decision maker decides which arms to take action 1, also referred to as the pulling action, subject to the budget constraint that exactly αN arms should be pulled. Here 0 < α < 1 and we assume αN is an integer. After the actions are applied, the N arms evolve independently. Specifically, the state transitions from s ∈ S N to s ′ ∈ S N with probability
P h (s ′ | s, a) = N n=1 P h (s ′ n | s n , a n ),
where (s n , a n ) denotes the state-action pair for the n-th arm, and P h (• | •, a) is the transition kernel under action a. For convenience, we refer to this RMAB system with N arms as the N -system.
At each time step h, the decision maker collects an additive reward N n=1 r h (s n , a n ), where r h (s, a) denotes the reward for the state-action pair (s, a) at time h and is assumed to be nonnegative without loss of generality. The objective is to find a policy π N , which maps the bandit state vector s h = (s 1,h , s 2,h , . . . , s N,h ) to an action vector a h = (a 1,h , a 2,h , . . . , a N,h ) for each h, that maximizes the total expected reward per arm over the horizon:
V N opt := max π N H h=1 1 N N n=1 E π N [r h (s n,h , a n,h )](1)
s.t. N n=1 a n,h = αN, 1 ≤ h ≤ H.(2)
To reduce the computational complexity, we leverage the fact that the N arms are homogeneous and aggregate the states of individual arms. This gives us a simplified representation of the bandit state, which facilitates subsequent analysis and policy design. Specifically, we represent the bandit state at time h as a vector X h = (X h (s)) s∈S ∈ ∆ S N , where each X h (s) denotes the fraction of arms in state s ∈ S. Here, ∆ S N is a discrete subset of ∆ S , the probability simplex over S, such that each x ∈ ∆ S N satisfies that N x has all integer entries. Similarly, we represent the action at time h as a vector Y h = (Y h (s, a)) s∈S,a∈A ∈ ∆ 2S
N , where each Y h (s, a) denotes the fraction of arms in state s taking action a. We treat Y h as a row vector of length 2S. We also write Y h (•, 0) = (Y h (s, 0)) s∈S and Y h (•, 1) = (Y h (s, 0)) s∈S , which are both row vectors of length S.
Under the new state and action representation, given an action Y h = y h , the state evolves as Under the new state and action representation, a policy π N maps the state X h to an action vector Y h for each h. We also rewrite the reward function as a vector r h = (r h (s, a)) s∈S,a∈A ∈ R 2S . Suppose the initial state of the N -system is X 1 = x ini ∈ ∆ S N . Then the objective of the RMAB problem can be reformulated as:
X h+1 = 1 N s,a U (s,a) h (y h ),(3)
V N opt (x ini , 1) = max π N H h=1 E π N r h Y ⊤ h (4
) s.t. s Y h (s, 1) = α, 1 ≤ h ≤ H,(5)
Y h (•, 0) + Y h (•, 1) = X h , Y h ∈ ∆ 2S N , 1 ≤ h ≤ H.(6)
Here (5) corresponds to the budget constraint, and (6) ensures that Y h is a valid action for state X h .
this section cite: []

Section: Fluid approximation.
One classical approach to address the complexity of RMAB is to consider a fluid approximation as in [7,10,15,34,36], where the stochastic state transition is replaced by its expected value, leading to a deterministic transition at each step:
x h+1 = s,a y h (s, a)P h (• | s, a).(7)
This fluid system smooths out the stochastic fluctuations in the N -system and leads to the following Linear Program (LP), which we refer to as the fluid LP:
V LP (x ini , 1) := max (x h ,y h ) h∈{1,2,...,H} H h=1 r h y ⊤ h (8
) s.t. s y h (s, 1) = α, 1 ≤ h ≤ H,(9)
y h (•, 0) + y h (•, 1) = x h (•), y h ≥ 0, 1 ≤ h ≤ H,(10)
x 1 = x ini , x h+1 (•) = s,a y h (s, a)P h (• | s, a) , 1 ≤ h ≤ H -1.(11)
Let x * = (x * h ) h∈{1,2,...,H} and y * = (y * h ) h∈{1,2,...,H} be an optimal solution to this fluid LP. Note that this LP can be efficiently solved. Furthermore, it has been shown that V N opt (x ini , 1) ≤ V LP (x ini , 1); see, for instance, [36, Lemma 1] or [32,Lemma 3]. That is, the optimal value of the LP provides an upper bound on the optimal value of the RMAB problem. Based on the solution to this LP, LP-based policies can be obtained but have Θ(1/ √ N ) optimality gap per arm in degenerate RMABs, as we pointed out in the Introduction.
this section cite: ['b6', 'b9', 'b14', 'b33', 'b35', 'b31']

Section: Gaussian approximation and SP-based policy

this section cite: []

Section: Gaussian stochastic system
The performance of LP-based policies is inherently limited by how accurately the fluid system approximates the original N -system. To design policies that outperform the LP-based policies, we construct a Gaussian stochastic system that better approximates the original N -system by capturing not only the mean but also the variance of the system. This Gaussian stochastic system is centered around y * , an optimal solution to the fluid LP. We then search for an optimal policy for the Gaussian stochastic system in a neighborhood of y * , which adjusts y * to account for the stochasticity. This policy is then applied to the N -system after properly handling the integer effect.
Specifically, we construct a Gaussian stochastic system with state space ∆ S and action space ∆ 2S . The system has the same initial state x ini as the N -system. Under action vector y h , the state of the Gaussian stochastic system at the next time step, Xh+1 , is a random vector of the following form:
Xh+1 = Proj ∆ S s,a y h (s, a)P h (• | s, a) + Z h / √ N ,(12)
where Z h is a Gaussian random vector with distribution N (0, Γ h (y * h )), and Z h 's are independent across steps. The covariance matrix Γ h (y * h ) is a constant matrix independent of N , with its explicit expression given in Appendix A. The projection, Proj ∆ S (•), is onto the simplex ∆ S under the ℓ 2 -distance. This projection will not be used with a high probability. Indeed, it can be shown that s,a y h (s,
a)P h (• | s, a) + Z h / √
N ∈ ∆ S occurs with probability 1 -O(1/N log N ), see Lemma C.8 of [33] for a proof.
We now explain how the Gaussian stochastic system captures the mean and variance of the N -system. Suppose the action at time h is y h . Ignoring the projection, we have Xh+1 = s,a y h (s,
a)P h (• | s, a) + Z h / √ N . The term s,a y h (s, a)P h (• | s, a)
is the expectation of the state X h+1 in the N -system, which is the same as the state x h+1 in the fluid system under action y h . The additional term, Z h / √ N , is random, and by construction, its covariance matrix matches that of X h+1 in the N -system if y h = y * h . Therefore, this term captures the variance of the N -system when y h is close to the optimal solution y * h of the fluid LP. We remark that a key innovation of our approach is to replace the otherwise (state,action)-dependent diffusion terms with (state,action)-independent ones. We construct the Gaussian system so that the covariance of the noise is fixed-chosen from the fluid-optimal solution y * h -rather than depending on y h . This enables the use of stochastic-programming techniques that require action-independent randomness, such as the EDDP algorithm [20, Algorithm 3] we employ in Section 5. Conceptually, this is a further "simplification" of the classical diffusion approximation: although the covariance is fixed at y * h , the resulting approximation retains the same order of error for our problem (a point established in Theorem 4.1) while remaining computationally tractable. This simplification is justified by scale separation: in the N -system, stochastic fluctuations are O(1/ √ N ) relative to the deterministic drift (see the 1/ √ N factor multiplying Z h in ( 12)). Hence the covariance induced by the fluid-optimal action serves as a robust surrogate for the action-dependent covariance without changing the asymptotic accuracy, a property that typically does not hold in traditional diffusion control but is available here due to the central limit theorem scaling.
this section cite: ['b32']

Section: Stochastic-Programming-based (SP-based) policy
After constructing the Gaussian stochastic system, we search for an optimal policy in the Gaussian stochastic system. However, we restrict the search to a neighborhood of y * , since the Gaussian stochastic system closely approximates the N -system when the state and action of the N -system stay close to y * . In particular, given a fixed parameter δ N := 2 log N/ √ N = Θ(1/ √ N ), we define the following policy class:
Π δ N (y * ) := π : ∀1 ≤ h ≤ H, ∥π(x h , h) -y * h ∥ ∞ ≤ κz h δ N , if ∥x h -x * h ∥ ∞ ≤ z h δ N ; π = π pre for a predefined policy π pre , if ∥x h -x * h ∥ ∞ > z h δ N .(13)
Algorithm 1 Stochastic-Programming-based (SP-based) policy 1: Input: An optimal solution y * to LP (8); constants z h , δ N and κ; a predefined policy π pre 2: if y * is non-degenerate then 3:
Use an LP-based policy 4: Break 5: end if 6: Solve the Gaussian stochastic program (14) to obtain an optimal policy πN, * ∈ Π δ N (y * ) 7: for h = 1 to H do 8:
x h ← state of the N -system at time h 9:
if ∥x h -x * h ∥ ∞ ≤ z h δ N then 10: Y h ← round(π N, * (x h , h)) 11:
else 12:
Y h ← round(π pre (x h , h)) 13:
end if 14:
Apply action Y h 15: end for
Here κ is a positive constant with value κ := max {2 + 6S, 3 + 2r max HS/σ}, where r max := max s,a,h r h (s, a); σ is a constant depending on the fluid LP; z h for 1 ≤ h ≤ H is a recursive sequence. The explicit expressions for these constants are collected in Appendix A of [33]. We note that all of them are independent of N , and can be computed or estimated solely from the fluid LP (8).
We now explain the reasoning behind the definition of the policy class Π δ N (y * ). We restrict corrections to a O(1/ √ N ) neighborhood of the fluid-optimal state-action. When the LP has a unique solution, an optimal N -system policy lies within this neighborhood (see Lemma C.1 of [33] for a proof); hence our restriction retains optimal policies while ensuring that the secondorder approximation incurs only O(1/N ) error (see Lemma C.4 of [33] for a proof). Enlarging the neighborhood inflates the approximation error, whereas shrinking it risks excluding the true optimum-so O(1/ √ N ) is the "right" scale for our method. Moreover, since the smallest gap of interest is O(1/N ) and the process leaves this neighborhood with probability O(1/N log N ) (Lemma C.8 of [33]), the contribution of out-of-neighborhood behavior is negligible. Consequently, we do not distinguish between policies optimized only locally and those also optimized outside the neighborhood; whenever the state exits, we simply follow a predefined policy as in (13).
We then formulate the following Gaussian stochastic program (SP):
max π∈Π δ N (y * ) H h=1 E r h Ỹ⊤ h (14
) s.t. s Ỹh (s, 1) = α, 1 ≤ h ≤ H,(15)
Ỹh (•, 0) + Ỹh (•, 1) = Xh , Ỹh ≥ 0, 1 ≤ h ≤ H,(16)
Xh+1 = Proj ∆ S s,a Ỹh (s, a)P h (• | s, a) + Z h / √ N , 1 ≤ h ≤ H -1.(17)
We now present our SP-based policy, formally described in Algorithm 1. The core idea of this policy is to solve the Gaussian stochastic program and obtain an optimal policy πN, * ∈ Π δ N (y * ). This policy is then applied to the N -system through a simple rounding procedure. It guarantees that the action generated by the algorithm pulls an integer number of agents. Since when computing the first-and second-order approximations, we extend the state and action spaces from their original discrete sets (with granularity 1/N ) to continuous-valued probability simplices. The action computed in this continuous domain must subsequently be converted back to the discrete domain, ensuring that all values multiplied by N are integers and thus applicable to the N -agent problem. This rounding procedure is explicitly detailed in Appendix A, in which we showed alongside that the rounding error is of order O(1/N ).
We remark that we only use the policy πN, * when the RMAB is degenerate. When the RMAB is non-degenerate, our policy defaults to a LP-based policy, which prior work (see Introduction) has shown to achieve an exponentially small optimality gap (in terms of N ) relative to V LP . The definition of non-degeneracy is given below, and it is easy for an algorithm to check whether an RMAB is non-degenerate or not.
Definition 3.1 (Non-degeneracy [8,10,36]). An RMAB is non-degenerate if, its corresponding fluid LP (8) admits an optimal solution y * , such that for each h with 1 ≤ h ≤ H, there exists at least one state s ∈ S such that y * h (s, 0) > 0 and y * h (s, 1) > 0.
this section cite: ['b13', 'b32', 'b7', 'b32', 'b32', 'b32', 'b12', 'b7', 'b9', 'b35']

Section: Illustration of the SP-based policy on a degenerate example
We illustrate the SP-based policy via a two-state RMAB with horizon H = 2 and pulling budget α = 0.5. The rewards are given by r N -system problem. Note that at time h = 2, only r 2 (1, 1) = 1, so the optimal action at h = 2 is to pull as many arms in state 1 as possible, i.e., Y 2 (1, 1) = min{0.5, X 2 (1)}. Therefore, we only need to decide the optimal action at h = 1. We further notice that since X 1 (1) = X 1 (2) = α = 0.5, the entries of Y 1 are all determined by Y 1 (1, 1) as follows
Y 1 (1, 0) = Y 1 (2, 1) = 0.5 -Y 1 (1, 1), Y 1 (2, 0) = Y 1 (1, 1).
Therefore, we only need to optimize Y 1 (1, 1). The N -system optimal policy is given by the solution of the following problem max 0≤Y1(1,1)≤0.5
Y 1 (1, 1) + E [min {0.5, X 2 (1)}] .(18)
Fluid LP and its optimal solution. In the fluid system, we replace X 2 (1) with its mean x 2 (1) =
which exchanges the expectation and the min operator in the N -system problem (18). The optimal solution is y * 1 (1, 1) = 0.2609. One can verify that this problem is degenerate (see Definition 3.1).
this section cite: ['b17']

Section: SP-based policy.
Given the fluid solution y * above, the corresponding Gaussian stochastic program is max 0≤ Ỹ1(1,1)≤0.5
Ỹ1 (1, 1) + E min 0.5, 0.8 -1.15 × Ỹ1 (1, 1) + Z 1 √ N ,(20)
where Z 1 d ∼ N (0, 0.1624). Since we are searching for an optimal solution around y * 1 , let Ỹ1 (1, 1) = y * 1 (1, 1) + c √ N . Then the stochastic program can be written as
y * 1 (1, 1) + 0.5 + 1 √ N max c (c + E [min {0, Z 1 -1.15c}]) ,
which is equivalent to the following problem
max c c + E [min {0, Z 1 -1.15c}] .
There exists an explicit and unique solution to the problem above, and the solution can be numerically computed and is c * d = 0.3940 (for more details please refer to Appendix D of [33]). Therefore, the SP-based policy is given by round( Ỹ * 1 (1, 1)) = round(0.2609 + 0.3940/ √ N ). This SP-based policy outperforms LP-based policies, as illustrated in Figure 1 in the Introduction.
this section cite: ['b32']

Section: Some insights.
We compare the fluid approximation in (19) with the Gaussian approximation in (20) when they both take an action y 1 (1, 1) = Ỹ1 (1, 1) = y * 1 (1, 1) + c/ √ N for a positive constant c. In the fluid system, the reward for h = 2 is min 0.5, 0.8 -1.15 × y 1 (1, 1) , which is capped at 0.5. One can verify that this deviates from the value E [min {0.5, X 2 (1)}] in the N -system by Θ(1/ √ N ), caused by the exchange of the expectation and the min operator. In contrast, in the Gaussian stochastic system, E min 0.5, 0.8 -1.15 × Ỹ1 (1, 1) + Z 1 √ N = 0.5 + E min 0, Z 1 -1.15c √ N , which can be verified to be Õ(1/N ) away from the value E [min {0.5, X 2 (1)}] in the N -system and thus giving a better approximation. This better approximation allows us to find a better policy near y * 1 (1, 1). The inaccuracy of the fluid approximation is in fact a fundamental reason that LP-based policies have a Θ(1/ √ N ) optimality gap for some degenerate RMABs, whereas the correction in our SP-based policy reduces the Θ(1/ √ N ) inaccuracy to Õ(1/N ).
this section cite: ['b18']

Section: Main theoretical results
In this section, we present our main theoretical results. We will frequently use the following quantities. Let V N π (x h , h) and Q N π (x h , y h , h) denote the value function and Q-function of policy π evaluated in the N -system; and Ṽ N π (x h , h) and QN π (x h , y h , h) denote the value function and Q-function of a policy π evaluated in the Gaussian stochastic system. Further, we consider the following optimal policies within the policy class Π δ N (y * ): πN, * ∈ arg max
π∈Π δ N (y * ) Ṽ N π (x h , h),
which is referred to as the locally-SP-optimal policy. We sometimes say that we apply the locally-SP-optimal policy πN, * to the N -system and denote its value function as V N πN, * (x h , h), with the understanding that we apply πN, * with the rounding procedure, detailed in Appendix A. Note that once we obtain an optimal solution to the LP, verifying uniqueness is straightforward [3]. Theorem 4.1 (Global optimality). Consider an RMAB that satisfies the Uniqueness Assumption 4.1. Then the locally-SP-optimal policy, πN, * , when applied to the N -system (with rounding), achieves an optimality gap of Õ(1/N ); i.e.,
this section cite: ['b2']

Section: Global optimality
V N opt (x ini , 1) -V N πN, * (x ini , 1) = Õ(1/N ),
where V N opt is the optimal value function, and V N πN, * is the value function of πN, * , both in the N -system.
A detailed proof of Theorem 4.1 is presented in Appendix C of [33]. Below, we provide an outline and highlight some novel and technically interesting components of the proof. It consists of proving the following statements, as illustrated in Figure 2.
• Prove that V N opt = V N π N, * (Lemma C.1 of [33]), where π N, * is the locally optimal policy within the policy class Π δ N (y * ). This is proved by showing that a (globally) optimal policy for the N -system belongs to the policy class Π δ N (y * ) under Assumption 4.1.
• Prove that |V N π N, * (x, h) -Ṽ N πN, * (x, h)| = Õ(1/N ) (Lemma C.2 of [33]) and | Ṽ N πN, * (x, h) -V N πN, * (x, h)| = Õ(1/N ) (Lemma C.3 of [33]. These two lemmas are enabled by the fact that we restrict the policies π N, * and πN, * to Π δ N (y * ), i.e., a Θ(1/ √ N )-neighborhood of y * , which translates into a Wasserstein distance of Õ(1/N ) between the respective next-state distributions in the N -system and in the Gaussian stochastic system.
We next highlight two interesting components in the proofs.
• Characterization of optimal policies in the N -system. Lemma C.1 of [33] establishes that under Assumption 4.1, there exists an optimal policy of the N -system whose actions are close to the optimal fluid solution y * . This result is noteworthy because optimal policies of the N -system are not well-understood in the literature. Prior work often circumvents this by only studying the LP upper bound V LP , which can be loose as shown in Theorem 4.2.
• Approximate Lipschitz continuity in the Gaussian stochastic system. A key step in proving Lemmas C.2 and C.3 of [33] is to establish an approximate local Lipschitz property of the value function Ṽ N πN, * in the Gaussian stochastic system, where restricting the policy to πN, * introduces technical challenges. We overcome these challenges through the careful construction of an action mapping.
this section cite: ['b32', 'b32', 'b32', 'b32', 'b32', 'b32']

Section: The Θ(1/ √ N ) optimality gap of LP-based policies
To complement Theorem 4.1, we next present a result showing that the Θ(1/ √ N ) optimality gap is fundamental to a large class of LP-based policies. Specifically, consider the following policy class, which includes a large class of LP-based policies such as those in [7,10,36]:
Π fluid (y * ) := π : ∥π(x h , h) -y * h ∥ ∞ ≤ κ∥x h -x * h ∥ ∞ , ∀1 ≤ h ≤ H ,(21)
where (x * , y * ) is an optimal solution of the fluid LP in (8). It has been shown in [10,Theorem 1] that any policy in Π fluid (y * ) has an O(1/ √ N ) optimality gap. We now show that there exist RMAB instances where this optimality gap order is tight. The result is established based on the example given in Section 3.3, and the detailed proof is presented in Appendix D of [33].
this section cite: ['b6', 'b9', 'b35', 'b7', 'b9', 'b32']

Section: Theorem 4.2 (Fluid gap).
There exist RMAB instances for which all LP-based policies in Π fluid (y * ) have an Θ(1/ √ N ) optimality gap; i.e., V N opt (x ini , 1) -V N fluid (x ini , 1) = Θ(1/ √ N ), where V N opt is the optimal value function, and V N fluid is the value function of the optimal policy within the policy class Π fluid (y * ). Moreover, there is an Θ(1/ √ N ) gap between the optimal value of the N -system and the optimal value of the fluid LP in (8
); i.e., V LP (x ini , 1) -V N opt (x ini , 1) = Θ(1/ √ N ).
We remark that although this theorem is proved via a specific example, we believe the result to hold more broadly for many degenerate RMABs. The Θ(1/ √ N ) optimality gap of LP-based policies stems from the Θ(1/ √ N ) approximation error in the fluid approximation, which arises when exchanging the expectation and the minimum operator, as shown in the example in Section 3.3. This phenomenon is common in degenerate RMABs.
this section cite: []

Section: Performance improvement
Under the Uniqueness Assumption 4.1, we have shown that our SP-based policy achieves an Õ(1/N ) optimality gap. When this assumption does not hold, the same optimality gap may not apply. However, Theorem 4.3 below shows that the SP-based policy can still yield improvement over LP-based policies.
Let y * be any optimal solution to the fluid LP in (8), and let πN, * be the corresponding SP-based policy. Recall that Ṽ N πN, * (x ini , 1) and QN πN, * (x ini , y 1 , 1) denote the value function and the Q-function of the policy πN, * in the Gaussian stochastic system, where in the Q-function action y 1 is applied at time 1. Theorem 4.3 states that if the action given by the SP-based policy πN, * at time 1 is "strictly" better than the optimal LP-based action y * 1 in the Gaussian stochastic system, then the SP-based policy improves over any LP-based policy in Π fluid (y * ) by Ω(1/ √ N ) in the N -system.
this section cite: []

Section: Theorem 4.3 (Performance improvement).
If there exists a positive constant ϵ independent of N such that Ṽ N πN, * (x ini , 1) -QN πN, * (x ini , y * 1 , 1) ≥ ϵ/ √ N , then for any π ∈ Π fluid (y * ), we have
V N πN, * (x ini , 1) -V N π (x ini , 1) = Ω(1/ √ N ).
Note that a key strength of this result is that we only need to evaluate the first step action y * 1 in the Gaussian stochastic system, and the result then holds for all policies in Π fluid (y * ). We refer to Appendix E of [33] for a detailed proof of Theorem 4.3. We evaluated the performance of our SP-based policy (Algorithm 1) on a machine maintenance problem [9,12], an RMAB formulation motivated by real-world trade-offs in preventive maintenance and resource allocation. The resulting SP is solved using the EDDP algorithm [20,Algorithm 3]. The details of the experiments are presented in Appendix B, along with an empirical study of the computational complexity of EDDP for solving the SPs arising from RMABs across varying problem sizes.
this section cite: ['b32', 'b8', 'b11', 'b19']

Section: Numerical experiments
We performed two sets of experiments where the problem instances are degenerate: one set where the fluid LP solution is unique (Figure 3a and 3c) and one set where it is not unique (Figure 3b and 3d). For both sets, computing the optimal policies is intractable. In each set of experiments, we evaluated the performance of our SP-based policy and the LP-based policy on a sequence of problems with increasing numbers of machines N , and compared the total reward difference. Figure 3 demonstrates that the improvement of our SP-based policy over the LP-based policy grows with N in both settings.
this section cite: []

Section: Conclusion
In this paper, we proposed an SP-based policy for finite-horizon RMABs, leveraging a carefully constructed Gaussian approximation. Motivated by degenerate examples where fluid approximation alone fails to break the Θ(1/ √ N ) gap, we showed that our policy achieves an Õ(1/N ) optimality gap under the uniqueness assumption, and can achieve Ω( √ N ) improvement over a large class of LP-based policies.
this section cite: []

Section: References
Ref_id:b0 Title: Lagrangian index policy for restless bandits with average reward Year: (2024)
Ref_id:b1 Title: Some solvable stochastic control problems Year: (1980)
Ref_id:b2 Title: Introduction to linear optimization Year: (1997)
Ref_id:b3 Title: Some limit theorems in the theory of mass service, I. Theory of Probability and its Applications Year: (1964)
Ref_id:b4 Title: Some limit theorems in the theory of mass service Year: (1965)
Ref_id:b5 Title: On the Taylor expansion of value functions Year: (2020)
Ref_id:b6 Title: Index policies and performance bounds for dynamic selection problems Year: (2020)
Ref_id:b7 Title: Fluid policies, reoptimization, and performance guarantees in dynamic resource allocation Year: (2023)
Ref_id:b8 Title: A restless bandit approach for capacitated condition based maintenance scheduling Year: (2024)
Ref_id:b9 Title: Linear program-based policies for restless bandits: Necessary and sufficient conditions for (exponentially fast) asymptotic optimality Year: (2023)
Ref_id:b10 Title: Bandit processes and dynamic allocation indices Year: (1979)
Ref_id:b11 Title: Index policies for the maintenance of a collection of machines by a set of repairmen Year: (2005)
Ref_id:b12 Title: Brownian Motion and Stochastic Flow Systems Year: (1985)
Ref_id:b13 Title: Scheduling networks of queues: Heavy traffic analysis of a simple open network Year: (1989)
Ref_id:b14 Title: An asymptotically optimal index policy for finite-horizon restless bandits Year: (2017)
Ref_id:b15 Title: Multiple channel queues in heavy traffic. I Year: (1970)
Ref_id:b16 Title: Multiple channel queues in heavy traffic. II: Sequences, networks, and batches Year: (1970)
Ref_id:b17 Title: Restless and uncertain: Robust policies for restless bandits via deep multi-agent reinforcement learning Year: (2022)
Ref_id:b18 Title: Optimal target tracking with restless bandits Year: (2006)
Ref_id:b19 Title: Complexity of stochastic dual dynamic programming Year: (2022)
Ref_id:b20 Title: Multi-agent task assignment in the bandit framework Year: (2006)
Ref_id:b21 Title: Collapsing bandits and their application to public health intervention Year: (2020)
Ref_id:b22 Title: Risk-aware interventions in public health: Planning with restless multi-armed bandits Year: (2021)
Ref_id:b23 Title: NeurWIN: Neural Whittle index network for restless bandits via deep RL Year: (2021)
Ref_id:b24 Title: The complexity of optimal queuing network control Year: (1999)
Ref_id:b25 Title: Multi-stage stochastic optimization applied to energy planning Year: (1991)
Ref_id:b26 Title: Analysis of stochastic dual dynamic programming method Year: (2011)
Ref_id:b27 Title: Lectures on Stochastic Programming: Modeling and Theory Year: (2021)
Ref_id:b28 Title: Restless bandits: activity allocation in a changing world Year: (1988)
Ref_id:b29 Title: Finite-time analysis of Whittle index based Q-learning for restless multi-armed bandits with neural network function approximation Year: (2023)
Ref_id:b30 Title: Reinforcement learning augmented asymptotically optimal index policy for finite-horizon restless bandits Year: (2022)
Ref_id:b31 Title: An optimal-control approach to infinite-horizon restless bandits: Achieving asymptotic optimality with minimal assumptions Year: (2024)
Ref_id:b32 Title: Achieving O(1/N ) optimality gap in restless bandits through Gaussian approximation Year: (2025)
Ref_id:b33 Title: An asymptotically optimal heuristic for general non-stationary finite-horizon restless multi-armed multi-action bandits Year: (2017)
Ref_id:b34 Title: Leveraging nondegeneracy in dynamic resource allocation. Available at SSRN Year: (2024)
Ref_id:b35 Title: Restless bandits with many arms: Beating the central limit theorem Year: (2021)
