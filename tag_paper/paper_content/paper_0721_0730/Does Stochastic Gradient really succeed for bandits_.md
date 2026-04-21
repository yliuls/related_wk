Title: Does Stochastic Gradient really succeed for Bandits?
Abstract: Recent works of Mei et al. [1,2] have deepened the theoretical understanding of the Stochastic Gradient Bandit (SGB) policy, showing that using a constant learning rate guarantees asymptotic convergence to the optimal policy, and that sufficiently small learning rates can yield logarithmic regret. However, whether logarithmic regret holds beyond small learning rates remains unclear. In this work, we take a step towards characterizing the regret regimes of SGB as a function of its learning rate. For two-armed bandits, we identify a sharp threshold, scaling with the suboptimality gap ∆, below which SGB achieves logarithmic regret on all instances, and above which it can incur polynomial regret on some instances. This result highlights the necessity of knowing (or estimating) ∆ to ensure logarithmic regret with a constant learning rate. For general K-armed bandits, we further show the learning rate must additionally scale inversely with K to avoid polynomial regret. We introduce novel techniques to derive regret upper bounds for SGB, laying the groundwork for future advances in the theory of gradient-based bandit algorithms.

Section: Introduction
A K-armed bandit is a sequential decision-making problem where, at each time t ∈ [T ], a learner chooses an action A t ∈ [K] and receives a reward r t drawn independently at random from the fixed distribution ν At . The objective is to select a policy π, which at each step t maps past observations H t-1 = (A s , r s ) s∈[t-1] to a sampling probability p π k,t = P π (A t = k | H t-1 ) for each arm k ∈ [K], in order to maximize the expected cumulative reward E π [ T t=1 r t ] when actions are chosen according to π. This is equivalent to minimizing the regret, defined by
R π T :=E π T t=1 K k=1 ∆ k 1(A t = k) , with ∀k, ∆ k = max j∈[K] µ j -µ k and µ k = E r∼ν k [r] , (1
)
or equivalently R π T = E π T t=1 K k=1 p π k,t ∆ k ≤ E π T t=1 (1 -p π 1,t ) • max k∈[K] ∆ k ,(2)
where for convenience we assume that arm 1 is the unique optimal arm, i.e., µ 1 > max j∈{2,...,K} µ j .
We further define the minimum gap ∆ := min k:∆ k >0 ∆ k . Finally, we denote by F the family of reward distributions supported on [-1, 1], and assume that ν k ∈ F for all k ∈ A ≤ cB (resp. ≥) and c is independent of T but can be problem-dependent. To hide poly-logarithmic factors in T , we respectively use O and Ω, for example A ≥ T / log(T ) =⇒ A = Ω(T ). Finally, we use A ≍ B when both A ≲ B and B ≲ A hold.
this section cite: []

Section: Motivation and related work
Stochastic gradient ascent methods have been extensively studied in the context of bandits and reinforcement learning, tracing back to foundational works such as Robbins and Monro's stochastic approximation [3], the REINFORCE algorithm [4], and subsequent developments in policy gradient (PG) methods [5]. Despite the empirical success of these algorithms in modern deep reinforcement learning [6][7][8][9][10], the theoretical understanding of their convergence and regret guarantees remains limited. This motivates a closer examination of their foundational components in simpler settings.
We study one such component, the Stochastic Gradient Bandit algorithm (SGB) [11,Chapter 2.8], a softmax policy gradient method for Multi-Armed Bandits (MAB).
A central challenge in analyzing SGB stems from the weakness of its underlying optimization structure, making the analysis difficult even in the case of PG with access to exact gradient information [12].
The optimization challenge arises from the softmax parameterization satisfying only a non-uniform version of the Polyak-Łojasiewicz (PL) inequality [13,Lemma 3]. Consequently, if the sampling probability of the best arm becomes too small, the gradient signal vanishes and SGB may require an exponentially long time to recover, even with access to exact gradients [14,15]. Nevertheless, PG with softmax parameterization and exact gradients has been shown to converge to a globally optimal policy asymptotically [16,Theorem 5.1], and with a O(1/T ) rate [13], although the rate of convergence depends on suitable initialization and problem dependent constants [14,15]. Convergence guarantees have also been obtained with regularization [12] or by modifying the softmax function [17].
The analysis of SGB with stochastic gradients becomes even more convoluted. Contrary to typical policies like UCB [18] or Thompson Sampling [19], the decisions of SGB depend intricately on the order in which all past rewards were collected, and thus cannot be analyzed through simple summary statistics; see Appendix A.3 for a detailed comparison with standard bandit policies. Despite these difficulties, a number of recent works are able to show global convergence of variants of SGB (and its generalization for Markov Decision Process) [20][21][22][23][24][25][26][27]. However, these result require decaying learning rate or regularization resulting in at best O(1/ √ T ) convergence rate [20][21][22][23]. Mei et al. [28] showed that natural policy gradient with oracle baselines achieves a O(1/T ) convergence rate. A closely related algorithm to SGB is SAMBA [29], a policy that achieves logarithmic regret for Bernoulli rewards by performing stochastic gradient ascent directly on the sampling probabilities without relying on the softmax transformation; for a detailed comparison to SGB see Appendix B.2.
Recently, Mei et al. [1] established a regret upper bound of O(log(T )) for SGB with a small constant learning rate η satisfying η ≲ ∆ 2 /K 3/2 where K is the number of arms. In a follow-up work, Mei et al. [2] further proved that SGB asymptotically converges to a globally optimal policy for any constant learning rate. While asymptotic convergence is a desirable property, it does not guarantee favorable regret guarantees, as we discuss in Appendix A.4. This motivates further investigation of the regret properties of SGB without the small rate constraint.
The goal of our work is to characterize the strengths and limitations of SGB, as a principled yet simple and scalable learning rule; which is representative of algorithms that have shown empirical success in more complex, large-scale settings. Importantly, our aim is not to promote SGB over state-of-the-art bandit algorithms such as KL-UCB [30] or Thompson Sampling [31,32]. To achieve our goal, in this work, we depart from optimization-based analyses of SGB and develop a regret-based analysis. Our analysis exploits a novel decomposition into a term that remains logarithmic for all learning rates, and a second term capturing the probability of failure-that is, the chance that the optimal arm fails to stand out-which crucially depends on η. This enables us to determine regimes where SGB succeeds.
this section cite: ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b12', 'b13', 'b14', 'b11', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b19', 'b20', 'b21', 'b22', 'b27', 'b28', 'b0', 'b1', 'b29', 'b30', 'b31']

Section: The Stochastic Gradient Bandit policy (SGB)
Following Sutton and Barto [11, Section 2.8], we define SGB as a randomized policy, with sampling probabilities at time t ≥ 1 given by the softmax transform of some parameters (θ k,t ) k∈[K] ∈ R K , ∀t ∈ [T ], ∀k ∈ [K] : p SGB k,t ∝ exp (θ k,t ) .
In the following, we drop the superscript for simplicity of notation. Given a fixed learning rate η, the parameter of each arm k ∈ [K] is initialized at θ k,1 = 0, and is updated as follows, ∀t ≥ 1 : θ k,t+1 = θ k,t +η•δθ k,t , with δθ k,t := (1-p k,t )•r t •1(A t = k)-p k,t •r t •1(A t ̸ = k) . (4)
For completeness, we provide the pseudo-code of SGB in Algorithm 1 in Appendix A.1. In the remainder of the paper, we use the shorthand notation E t [•] for the conditional expectation E[•|H t-1 ].
From Equation (4), we derive the following first fundamental property of SGB, ∀k ∈ [K], t ≥ 1 : E t [δθ k,t ] = p k,t (1 -p k,t )µ k -p k,t j̸ =k p j,t µ j = p k,t • (E t [∆ At ] -∆ k ) . (5) This equation allows to verify (see Appendix A.1) that SGB indeed performs a stochastic gradient ascent in order to maximize the value of the policy at time t, V SGB t := K k=1 p k,t •µ k , since (δ k,t ) k∈[K] is an unbiased estimate of ∇ θ V SGB t . It also shows that a parameter θ k,t is expected to increase only if the gap of arm k is smaller than E t [∆ At ] = K j=1 p j,t ∆ j , the instantaneous regret of SGB. From (5), we further obtain that if arm 1 is the only optimal arm, the parameter of the best arm satisfies
E[θ 1,T +1 ] = η • E T t=1 δθ 1,t = η • E T t=1 p 1,t E t [∆ At ] ≥ η • E T t=1 p 1,t (1 -p 1,t )∆ (6
)
which is an equality if all sub-optimal arms have the same gap ∆. This allows us to express an upper bound on the regret as a function of the parameter θ 1,T +1 . Defining ∆ max = max k∈[K] ∆ k , and using that (1 -p 1,t ) = p 1,t (1 -p 1,t ) + (1 -p 1,t ) 2 , by plugging (6) into (2) we obtain that
R T ≤ ∆ max • E T t=1 (1 -p 1,t ) ≤ ∆ max η∆ • E [θ 1,T +1 ] Post-convergence term + ∆ max • E T t=1 (1 -p 1,t ) 2 Failure regret .(7)
This regret decomposition, though obtained via a straightforward reformulation, appears to be novel and, to the best of our knowledge, has not been exploited in prior analyses of SGB. In fact, Equation (7) provides a powerful lens through which we develop all the upper bounds in this work, as it splits the regret into two interpretable components. In Section 3, we prove that the post-convergence term is bounded by O(log T ) for any learning rate. Therefore, whether SGB achieves logarithmic regret or not depends entirely on the scaling of the failure regret. We justify this terminology in Appendix F.1, where we discuss an alternative formulation of Eq. ( 7) establishing the relation between failure regret and the sum of probabilities that p 1,t stays bounded away from 1. The empirical study in Section 4 further supports this decomposition: most trajectories of SGB fall into one of two distinct regimes-successful runs, with convergence to the optimal policy at rate O(t -1 ), and failed runs, where the optimal arm remains persistently under-sampled up to horizon T .
In addition to the above regret bound, we present in Appendix A.2 some other elementary properties of SGB that can be directly derived from its update rule (4), and will be useful in our analyses.
this section cite: ['b4', 'b6']

Section: Outline and contributions
We introduced in Section 1.2 an original decomposition of the regret of SGB into a post-convergence term and a failure regret. In Section 2, we build on Eq. ( 7) to provide a tight regret analysis for two-armed bandits when η ≲ ∆ (Thm. 1). We then prove in Thm. 2 that the regret of SGB can be polynomial on some instances if η ≳ ∆. Hence, we characterize the regret of SGB in two-armed bandits, except for a critical regime η ≈ ∆ that we discuss in Section 2.2. In particular, knowing a lower bound on the gap ∆ is necessary and sufficient to tune η to guarantee logarithmic regret. We summarize our results for the two-armed bandit problem in Table 1 below, denoting by F 2 ∆ the class of two-armed bandits with distributions supported on [-1, 1] and gap larger than ∆ ∈ (0, 1).
In Section 3, we consider a general number of arms K. Using lower bound arguments, we first prove a second necessary condition: logarithmic regret can be guaranteed only if η decreases linearly in K (Thm. 3). By combining this result with Thm. 2 we conjecture that the critical tuning of η might be proportional to ∆/K. For the regret analysis, we establish in Lemma 2 that the post-convergence term is logarithmic for any constant learning rate. Then, in Lemma 3, we derive an intermediate
this section cite: []

Section: Logarithmic Critical Polynomial
Table 1: Regret guarantees for SGB as a function of η for two-armed bandits upper bound on the failure regret. The upper bound contains additional terms compared to the case K = 2, causing significant technical challenges that we detail, but leave their dedicated analysis for future work. Nonetheless, as a promising first complete result for K-armed problems, we prove in Theorem 4 that the regret of SGB is logarithmic if ηe 2η ≤ 2∆ K+2 , when all arms have identical gap ∆. Finally, in Section 4 we present some synthetic experiments that illustrate our theoretical findings.
this section cite: []

Section: Characterizing the regret regimes for two-armed bandits
In this section we propose a tight characterization of the regret regimes of SGB when K = 2. We exhibit a separation between logarithmic and polynomial regret close to the gap ∆.
this section cite: []

Section: Regret analysis for η ≤ ∆ -O(∆ 2 )
We directly detail the main result of this section and its complete proof. Theorem 1 (Regret upper bound for K = 2). For a two-armed bandit instance ν ∈ F 2 with a gap ∆ ∈ (0, 1), the regret of SGB tuned with a learning rate η satisfying ηC η < ∆ is upper bounded by R SGB T ≤ log(1 + 4η∆T ) 2η + ∆ 2η(∆ -ηC η )
, with C η := 2 +∞ n=0 (2η) n (n + 2)! ≤ e 2η .
Proof. Since θ 2,t = -θ 1,t for all steps t, by (9), we drop the subscript 1 to simplify the notation.
Starting from the regret decomposition of Equation (7), we readily obtain that for any instance ν,
R SGB T ≤ E [θ T +1 ] η + ∆ • E T t=1(
1 -p t ) 2 . We start by upper bounding E[θ T +1 ]. Jensen inequality provides that E[θ T +1 ] = 1 2 • E log e 2θ T +1 ≤ 1 2 • log E e 2θ T +1 . We then use the update rule of (θ t ) t≥1 to upper bound E[e 2θt ]. From the Taylor expansion of the exponential, for any constant q ∈ R and random variable r supported on [-1, 1], it holds that E[e qr ] = 1 + qE[r]
+ +∞ n=2 q n n! E[r n ] ≤ 1 + qE[r] + q 2 • +∞ n=2 q n-2 n! ≤ 1 + qE[r] + q 2 2 • C |q| 2 . (8
)
Using the notation a t = 2η(1 -p t ) and b t = 2ηp t , for any t ≥ 1 Eq. ( 4) and (8) give that
E t [e 2θt+1 ] = e 2θt • E t [e 2ηδθt ] = e 2θt • p t E t [e atrt |A t = 1] + (1 -p t )E[e -btrt |A t = 2] ≤ e 2θt • 1 + p t • a t µ 1 + afoot_2 t • 0.5 • C at/2 + (1 -p t ) • -b t µ 2 + b 2 t • 0.5 • C bt/2 ≤ e 2θt • 1 + 2p t (1 -p t ) • η∆ + η 2 C η , since a t ∨ b t ≤ 2η . Then, we use the relation e 2θt (1 -p t ) = p t (Eq. (11) in Appendix A) to obtain that ∀t ≥ 1, E t [e 2θt+1 ] ≤ e 2θt + p 2 t • 2 η∆ + η 2 C η =⇒ E e 2θ T +1 ≤ 1 + 2 η∆ + η 2 C η • T, which gives the logarithmic term in the theorem. It remains to upper bound E[ T t=1 (1 -p t ) 2
]. For t ≥ 1, we define x t := 1-pt  pt = e -2θt , and obtain with the same arguments as above that E t [x t+1 ] ≤ x t • 1 -2p t (1 -p t )η∆ + 2p t (1 -p t )η 2 C η = x t -2η(1 -p t ) 2 • (∆ -ηC η ) . By taking expectation on both sides and summing over time steps, we thus obtain that
E[x T +1 -x 1 ] ≤ -2η • (∆ -ηC η ) • E T t=1
(1 -p t ) 2 , from which the result follows by using that E[x 1 -x T +1 ] ≤ x 1 = 1 and that ∆ -ηC η > 0.
this section cite: ['b6', 'b7']

Section: Tightness of the bound
In addition to being strikingly simple, the proof of Thm. 1 is also tight under the assumption that rewards are bounded in [-1, 1]. First, for ν 1 = Rad(∆) and ν 2 = Rad(0), and ηC η = (1-ε)∆ for some ε > 0, the logarithmic (post-convergence) term matches the asymptotic lower bound up to a factor (1-ε) -1 (1+O(∆)) (see D.1 for details). It thus cannot be much improved if ε, ∆ are small. For the constant term, the only possible improvement would be to use higher-order moments for a tighter approximation in Eq. (8). However, for Rademacher distributions we could at most replace C η by 1, so the gain would be minor for small η since we already have that C η ≤ e 2η .
this section cite: ['b7']

Section: Comparison with existing results
Theorem 1 is a fully explicit regret upper bound, which surprisingly does not rely on any techniques from standard analyses of gradient ascent policies. For two arms, we obtain guarantees for a much broader range of learning rates 2 than Mei et al. [1], and even obtain near-optimal logarithmic scaling of the regret if η happens to be close to ∆. We then compare SGB with SAMBA, another policy performing (non-parametric) gradient ascent. For two arms, SAMBA running with parameter α < ∆ achieves a regret bound of α -1 log(T ) [29], which is close to our result. We elaborate on the comparison between the two policies in Appendix B.2. While their analysis extends to K > 2, it is more involved than the proof of Thm. 1, is restricted to Bernoulli rewards, and involves non-explicit and potentially large constants. Finally, we highlight that the logarithmic component of Thm. 1 is valid for all constant η. Hence, for SGB restricting η is required only to guarantee that the failure regret converges, which is a novel insight compared to [29].
Alternative moment-based condition A closer examination of Eq. ( 8) reveals that the analysis of Thm. 1 can be extended or sharpened under general moment assumptions on the reward distribution. For example, if one assumes that sup m≥2 E[r m t ] ≤ s 2 for some s > 0, then the result continues to hold by scaling the term ηC η by s 2 when bounding the failure regret, so ηC η < ∆ s 2 is sufficient for logarithmic regret. Hence, SGB can be fine-tuned under higher-moment conditions, akin to how Bernstein-type inequalities [33, Chapter 2.8] improve confidence intervals in UCB-style algorithms. Such assumptions are often realistic in applications like marketing or online advertising, where reward distributions are typically bounded and the average outcomes (e.g. click rates) are small [34,35]. In Appendix F.5 we also discuss the case of unbounded rewards, e.g. sub-Gaussian. Notably, for SGB these assumptions merely expand the admissible range of learning rates. By contrast, standard approaches would require substantial structural changes, such as tighter confidence intervals for UCB or modified priors/posteriors for TS, to exploit higher-order moment information.
Knowledge of ∆ While assuming that a lower bound on ∆ is known remains a strong requirement, we conclude this section by showing that, even without this knowledge, a learning rate that depends only on the time horizon T leads to meaningful regret guarantees in the two-armed bandit setting.
Corollary 1 (of Theorem 1). If ∆ is unknown, one can set η = log(T ) T , which yields ∀ν ∈ F 2 , R T ≲ log(T ) η • 1(2ηC η ≤ ∆) + ∆T • 1(∆ < 2ηC η ) ≲ T log(T ),
by using the upper bound from the theorem if η is small enough, and R T ≤ ∆T otherwise.
In Proposition 4 (App. F.3) we further prove that a time-varying learning rate η t = log(e ∨ t)/t also yields R T = O( √ T ). While Mei et al. [1,Section 3] hint that such a rate may be an appropriate tuning for SGB, it appears that comparable results to ours have only been been obtained with regularization added to the policy update, as in [20,21]. On the other hand, the regret upper bound of the gap-free variant of SAMBA [29, Thm.2] still involves large problem-dependent constants, while the constants in Corollary 1 are absolute. Hence, the time-dependent tuning of η that we propose stands out as a practical choice when no lower bound on ∆ is available (see Appendix G.1 for experiments).
For the two-armed case, we believe that these theoretical results offer a comprehensive guideline on how to choose η when using SGB in practice: if the learner has access to a lower bound on ∆ < ∆ such that the gap-dependent tuning η ≈ ∆ guarantees that log(1+4η∆T ) 2η ≤ T log(T ) for horizon T , then the gap-dependent tuning should yield better performance, since we believe that the bound of the theorem is tight. Otherwise, the horizon-dependent tuning that we propose can be safely used.
this section cite: ['b0', 'b28', 'b28', 'b33', 'b34', 'b0', 'b19', 'b20']

Section: Necessary condition on fixed η for logarithmic regret
In this section, we prove that the knowledge of the minimum gap ∆ is necessary to tune the fixed learning rate of SGB in order to obtain logarithmic regret. The following Thm. 2 shows that if η is larger than a fixed constant that depends on the gap ∆ and the number of arms K, there exists a K-armed bandit problem with gap ∆ for which SGB has polynomial regret. To define it, for µ ∈ [-1, 1] we denote resp. by Rad(µ) and δ µ the Rademacher and Dirac distributions of mean µ.
this section cite: []

Section: Theorem 2 (Polynomial regret).
Fix ∆ ∈ (0, 1), and consider the instance ν = (ν k ) k∈[K] with ν 1 = Rad(∆) and
ν 2 = • • • = ν K = δ 0 . If the learning rate of SGB satisfies η > λ ∆ := K-1 K log 1 + 2∆ 1-∆ ,
then its regret on the instance ν is lower bounded as follows,
∀ ε > 0, R T = Ω T 1-(1+ε)λ∆/η ,
where Ω hides polylogarithmic terms in T and constants depending on η and ∆.
Proof sketch. Let us assume K = 2 in the following, and introduce the main steps of the proof of the theorem. Details and supporting results can be found in Appendix C, where we directly consider a general number of arms. Since only arm 1 yields non-zero rewards, we directly have
∀t ≥ 1, θ 1,t+1 = θ 1,t + η(1 -p 1,t ) • r t • 1(A t = 1) ,
so we denote by p n+1 the sampling probability of arm 1 after its n first pulls. The proof consists of identifying a scenario with linear regret, and lower bounding its probability as a function of η.
Step 1: Let S be the event that p n+1 ≤ 1 2T , for some fixed value of n. Then, under S the probability that arm 1 is never pulled again after its n first selections is larger than 1/2, in which case the regret is larger than ∆(T -n). It thus holds that R T ≥ 0.5
• ∆ • P(S) • (T -n) + .
Step 2: We lower bound P(S) for a well-chosen value of n = n 0 + n 1 , for two integers n 0 and n 1 . Consider the following scenario: in a preliminary phase, arm 1 collects only -1 rewards from its first n 0 pulls, and in a next phase it collects n 1 rewards with an empirical mean satisfying µ n1 ≈ -∆, while the number of +1s received is never more than the number of -1s throughout this phase. First, we derive in Lemma 6 a lower bound on the probability of this scenario. Then, we prove (Lemma 7) that setting n 1 = O(log(T )) and n 0 = O(log(n 1 )) guarantees that this scenario is included in the event S. From a high-level perspective, the preliminary phase only serves to make p n0+1 small enough to reduce the impact of the ordering of the rewards in the next phase, and the scaling of the lower bound (ignoring log terms) comes from the choice of n 1 .
Lastly, we emphasize that the result should hold with random sub-optimal arms (further assuming that they are "lucky enough" throughout the trajectory), but constant rewards simplify the proof.
this section cite: []

Section: Critical regime
We remark that Theorems 1 and 2 do not provide explicit results about the regret of SGB when η ≈ ∆, that we thus call the critical regime. However, by comparing both theorems and their proofs we conjecture that, on some difficult instances, SGB satisfies E (1 -p t ) 2 ≍ t -∆ η .
Indeed, after summation this scaling makes the failure regret admit a Ω(T 1-∆ η ) lower bound when η > ∆ (see Thm. 2). If there is a smooth interpolation between the logarithmic and polynomial regime, then there might exists a small range of learning rates η ≈ ∆ for which the failure regret could be logarithmic on difficult instances. This is supported by experiments from Section 4 and Appendix G: we observe a relatively smooth evolution of performance metrics (regret, percentage of failed runs) with the learning rate. In the experiments, the critical learning rates, such as η = ∆ when K = 2, provide good empirical performance.
this section cite: []

Section: Results for K > 2 arms
In this section, we explore the theoretical guarantees of SGB for K > 2. We start by proving that the learning rate must decrease with the number of arms in order to guarantee logarithmic regret. Then, we provide preliminary results for upper bounding the regret of SGB by generalizing the proof of Theorem 1. We also provide some intuition about why this case is significantly harder than K = 2, and what we believe to be the right critical scaling for η.
this section cite: []

Section: Necessary scaling of η in K for logarithmic regret
In this part, we exhibit a necessary condition on the learning rate for logarithmic regret, depending on the number of arms K. While we proved Theorem 2 by directly analyzing a difficult instance, on the contrary, the proof technique of Theorem 3 exploits the efficiency of SGB on an easy problem instance, from which we deduce the result using lower bound arguments. We start by presenting the regret upper bound leading to this conclusion. Lemma 1 (Regret upper bound on an easy instance). Let ν ∈ F K be a MAB defied by ν 1 = δ 0 and ν 2 = • • • = ν K = δ -∆ , for some ∆ > 0. Then, for any learning rate η, SGB satisfies
∀ ε ∈ (0, 1), R SGB T ≤ 1 + log (1 + (K -1)T η∆) (1 -ε)η + K 2 ε • ∆ + 1 η log K ε .
Proof sketch. We present the full proof in Appendix D.2. We first show that, since rewards are deterministic, arm 1 remains the mode of the sampling distribution for all steps t, and p k,t /p 1,t is non-increasing for any sub-optimal arm k. Hence, we establish that p 1,t ≥ 1 -ε happens in finite expected time for any threshold ε ∈ (0, 1), which gives the O(ε -1 ) term of the bound. We then prove that, from that stage, 1 -p 1,t decreases exponentially fast with the total number of sub-optimal plays. This careful decomposition allows us to obtain a logarithmic term that does not involve a multiplicative factor of K -1. In Appendix D.2 and G.3 we further discuss how this result might be tightened by a factor K-1 K , which we prove formally for K = 2 (Lemma 14).
Implication on the consistency of SGB Lemma 1 suggests that, for a fixed horizon, the regret of SGB can be arbitrarily small for large η in some easy bandit instances. As a consequence of the well-known asymptotic regret lower bound [36,37], this must come at a cost on other instances. We dedicate Appendix D.1 for a thorough presentation of the lower bound we use in the proof (restated in Theorem. 6), and other technical results needed to derive the following theorem. Theorem 3 (Polynomial regret for η ≳ K -1 ). Let ∆ ∈ (0, 1) and α ∈ (0, 1). Consider the class F K ∆ of K-armed bandit instances with minimum gap at least ∆. If the learning rate satisfies
η > 1 ∆(1 -α) log 1 + 2∆ 1-∆ • 1 K -1 ,
then there exists an instance ν ∈ F K ∆ such that the regret of SGB satisfies R T = Ω(T α ).
Proof. Setting ε -1 = log(3 + T ), Lemma 1 yields R T = η -1 • log(T ) + O( log(T )) for the instance ν 1 = {δ 0 , δ -∆ , . . . , δ -∆ } ∈ F K ∆ , for any η > 0. In contrast, Theorem 6 (adaptation of the Lai & Robbins lower bound) and Lemma 11 (specialization to Dirac distributions) imply that any policy with regret R T = O(T α ) on all instances in
F K ∆ should satisfy lim inf T →∞ R T log(T ) > (K-1)(1-α)∆ log(T ) log(1+ 2∆ 1-∆ )
on this deterministic instance ν 1 . Combining these two results proves the theorem.
this section cite: ['b35', 'b36']

Section: Discussion
The fact that the learning rate has to be inversely proportional to the number of arms might be surprising at first sight. An intuitive explanation is that the update rule of θ 1,t is agnostic to the number of arms: for any step t ≥ 1, the distribution of δθ 1,t conditioned on H t-1 is the same if arm 1 faces a single arm (K = 2) or a mixture of K -1 identical arms. Hence, in order to reduce the speed of convergence of p 1,t , and thus guarantee sufficient exploration of the K -1 alternatives, it is necessary to make the learning rate decrease with K. In contrast, standard bandit policies typically have a separate exploration mechanism for each arm, determined by their respective observations.
Conjectures Combining Thm. 2 and 3, we establish that logarithmic regret is only achievable for η ≲ ∆ ∧ K -1 . This suggests that the critical threshold separating logarithmic from polynomial regret may depend on the ratio ∆/K. We explore this idea further in Appendix D.3, where we present and motivate two conjectures. Conjecture 1, inspired by Theorem 3, posits that regret cannot be too small on an "easy" instance. Conjecture 2 considers a construction with one slightly sub-optimal arm and many very sub-optimal arms: ν 1 = Rad(∆), ν 2 = δ 0 , and
ν 3 = • • • = ν K = δ -1 .
Both suggest that the critical rate, above which SGB may suffer polynomial regret, is near 2∆ K for K-armed bandits.
this section cite: []

Section: References
Ref_id:b0 Title: Alekh Agarwal, Csaba Szepesvári, and Dale Schuurmans. Stochastic gradient succeeds for bandits Year: (2023-07)
Ref_id:b1 Title: Small steps no more: Global convergence of stochastic gradient bandits for arbitrary learning rates Year: (2024)
Ref_id:b2 Title: A stochastic approximation method Year: (1951)
Ref_id:b3 Title: Simple statistical gradient-following algorithms for connectionist reinforcement learning Year: (1992)
Ref_id:b4 Title: Policy gradient methods for reinforcement learning with function approximation Year: (2000)
Ref_id:b5 Title: Deterministic policy gradient algorithms Year: (2014)
Ref_id:b6 Title: Continuous control with deep reinforcement learning Year: (2016-05-02)
Ref_id:b7 Title: Trust region policy optimization Year: (2015)
Ref_id:b8 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b9 Title: Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor Year: (2018)
Ref_id:b10 Title: Reinforcement Learning: An Introduction Year: (2018)
Ref_id:b11 Title: Understanding the impact of entropy on policy optimization Year: (2019-06)
Ref_id:b12 Title: On the global convergence rates of softmax policy gradient methods Year: (2020-07-18)
Ref_id:b13 Title: Escaping the gravitational pull of softmax Year: (2020-12-06)
Ref_id:b14 Title: Softmax policy gradient methods can take exponential time to converge Year: ()
Ref_id:b15 Title: On the theory of policy gradient methods: Optimality, approximation, and distribution shift Year: ()
Ref_id:b16 Title: An alternate policy gradient estimator for softmax policies Year: (2022-03-30)
Ref_id:b17 Title: Finite-time analysis of the multiarmed bandit problem Year: (2002)
Ref_id:b18 Title: Analysis of Thompson sampling for the multi-armed bandit problem Year: (2012)
Ref_id:b19 Title: Global convergence of policy gradient methods to (almost) locally optimal policies Year: (2020)
Ref_id:b20 Title: Beyond exact gradients: Convergence of stochastic soft-max policy gradient methods with entropy regularization Year: (2021)
Ref_id:b21 Title: On the convergence and sample efficiency of variance-reduced policy gradient method Year: (2021)
Ref_id:b22 Title: Towards principled, practical policy gradient for bandits and tabular mdps Year: (2024)
Ref_id:b23 Title: A general sample complexity analysis of vanilla policy gradient Year: (2022-03-30)
Ref_id:b24 Title: Almost sure convergence rates of stochastic gradient methods under gradient domination Year: (2024)
Ref_id:b25 Title: Beyond stationarity: Convergence analysis of stochastic softmax policy gradient methods Year: (2024)
Ref_id:b26 Title: Structure matters: Dynamic policy gradient Year: (2024)
Ref_id:b27 Title: Csaba Szepesvári, and Dale Schuurmans. The role of baselines in policy gradient optimization Year: (2022)
Ref_id:b28 Title: Regret analysis of a markov policy gradient algorithm for multiarm bandits Year: (2023-08)
Ref_id:b29 Title: Kullback-Leibler upper confidence bounds for optimal sequential allocation Year: (2013)
Ref_id:b30 Title: On the likelihood that one unknown probability exceeds another in view of the evidence of two samples Year: (1933)
Ref_id:b31 Title: Thompson sampling for one-dimensional exponential family bandits Year: (2013)
Ref_id:b32 Title: Concentration inequalities : a non asymptotic theory of independence Year: (2013)
Ref_id:b33 Title: Click-through rate prediction in online advertising: A literature review Year: (2022)
Ref_id:b34 Title: Best arm identification in rare events Year: (2023-08-04)
Ref_id:b35 Title: Asymptotically efficient adaptive allocation rules Year: (1985)
Ref_id:b36 Title: Optimal adaptive policies for sequential allocation problems Year: (1996)
Ref_id:b37 Title: Exploration-exploitation tradeoff using variance estimates in multi-armed bandits Year: (2009)
Ref_id:b38 Title: Deviations of stochastic bandit regret Year: (2011)
Ref_id:b39 Title: The fragility of optimized bandit algorithms Year: (2024)
Ref_id:b40 Title: Stochastic multi-armed bandits: Optimal trade-off among optimality, consistency, and tail risk Year: (2023)
Ref_id:b41 Title: Bandit algorithms Year: (2020)
Ref_id:b42 Title: Introduction to multi-armed bandits Year: (2019)
Ref_id:b43 Title: Sample mean based index policies with o (log n) regret for the multi-armed bandit problem Year: (1995)
Ref_id:b44 Title: An empirical evaluation of thompson sampling Year: (2011)
Ref_id:b45 Title: Thompson sampling: An asymptotically optimal finite-time analysis Year: (2012)
Ref_id:b46 Title: Optimality of thompson sampling for gaussian bandits depends on priors Year: (2014-04-22)
Ref_id:b47 Title: Bandit algorithms based on Thompson sampling for bounded reward distributions Year: (2020)
Ref_id:b48 Title: An asymptotically optimal policy for finite support models in the multiarmed bandit problem Year: (2011)
Ref_id:b49 Title: Non-asymptotic analysis of a new bandit algorithm for semi-bounded rewards Year: (2015)
Ref_id:b50 Title: Differentiable meta-learning of bandit policies Year: (2020)
Ref_id:b51 Title: Maillard sampling: Boltzmann exploration done optimally Year: (2022)
Ref_id:b52 Title: Kullback-leibler maillard sampling for multi-armed bandits with bounded rewards Year: (2023)
Ref_id:b53 Title: Meta-learning bandit policies by gradient ascent Year: (2021)
Ref_id:b54 Title: A general recipe for the analysis of randomized multi-armed bandit algorithms Year: (2023)
Ref_id:b55 Title: Perturbedhistory exploration in stochastic multi-armed bandits Year: (2019)
Ref_id:b56 Title: Tor Lattimore, and Mohammad Ghavamzadeh. Garbage in, reward out: Bootstrapping exploration in multi-armed bandits Year: (2019)
Ref_id:b57 Title: Sub-sampling for multi-armed bandits Year: (2014)
Ref_id:b58 Title: The multi-armed bandit problem: An efficient nonparametric solution Year: (2020)
Ref_id:b59 Title: Sub-sampling for efficient non-parametric bandit exploration Year: (2020)
Ref_id:b60 Title: On Limited-Memory Subsampling Strategies for Bandits Year: (2021-07)
Ref_id:b61 Title: From optimality to robustness: Adaptive re-sampling strategies in stochastic bandits Year: (2021)
Ref_id:b62 Title: From dirichlet to rubin: Optimistic exploration in RL without bonuses Year: (2022-07)
Ref_id:b63 Title: Solution d'un probleme Year: (1887)
Ref_id:b64 Title: Lost (and found) in translation: André's actual method and its application to the generalized ballot problem Year: (2008)
Ref_id:b65 Title: A remark on stirling's formula. The American mathematical monthly Year: (1955)
Ref_id:b66 Title: Analysis of bayesian and frequentist strategies for sequential resource allocation. (Analyse de stratégies bayésiennes et fréquentistes pour l'allocation séquentielle de ressources) Year: (2014)
Ref_id:b67 Title: Bounded regret in stochastic multiarmed bandits Year: (2013)
Ref_id:b68 Title: Rémy Degenne, and Odalric-Ambrym Maillard. Fast asymptotically optimal algorithms for non-parametric stochastic bandits Year: (2023)
