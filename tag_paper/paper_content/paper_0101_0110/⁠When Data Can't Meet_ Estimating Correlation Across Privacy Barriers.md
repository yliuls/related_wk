Title: When Data Can't Meet: Estimating Correlation Across Privacy Barriers
Abstract: We consider the problem of estimating the correlation of two random variables X and Y , where the pairs (X, Y ) are not observed together, but are instead separated co-ordinate-wise at two servers: server 1 contains all the X observations, and server 2 contains the corresponding Y observations. In this vertically distributed setting, we assume that each server has its own privacy constraints, owing to which they can only share suitably privatized statistics of their own component observations. We consider differing privacy budgets (ε 1 , δ 1 ) and (ε 2 , δ 2 ) for the two servers and determine the minimax optimal rates for correlation estimation allowing for both noninteractive and interactive mechanisms. We also provide correlation estimators that achieve these rates and further develop inference procedures, namely, confidence intervals, for the estimated correlations. Our results are characterized by an interesting rate in terms of the sample size n, ε 1 , ε 2 , which is strictly slower than the usual central privacy estimation rates. More interestingly, we find that the interactive mechanism is always better than its non-interactive counterpart whenever the two privacy budgets are different. Results from extensive numerical experiments support our theoretical findings. nε 2 1 ε 2 2 , which is again strictly worse than the rates we find under the server level DP constraints.

Section: Introduction
Federated learning is a popular and extensively studied framework in modern machine learning. In traditional federated learning, due to privacy concerns, the servers are not allowed to pool raw data, but are restricted to sharing only sufficiently privatized statistics derived from the local observations. This method is particularly beneficial when training on sensitive data, such as healthcare or finance. The federated scenario is very systematically studied when the separation occurs horizontally, i.e. observations of the same set of features are binned separately into different servers. See, for example, Kairouz et al. [2021], Li et al. [2020a,b], Zhang et al. [2021] and the references therein.
To encourage collaboration on proprietary data across different organizations, however, it is often more reasonable to assume that the federation occurs "vertically", or across features. For example, in healthcare data, a hospital and a pharmaceutical company might have different pieces of information on the same patient: the hospital does not share private clinical information such as patient demographics or test results with the company, which instead has its own private information on the same patient's response to certain drugs. This new framework called vertical federated learning has recently seen studied in Chen et al. [2020], Liu et al. [2024], Wu et al. [2020], Wei et al. [2022], Yang et al. [2019], but a theoretical understanding of estimation and inference has largely been missing. This motivates the current work. We study the correlation of bivariate data from n pairs of samples (X i , Y i ) which are not observed together, but are instead separated into two servers as {X i : 1 ≤ i ≤ n} and {Y i : 1 ≤ i ≤ n}.
To distinguish our results from the influence of estimating the marginal distributions of X and Y , we assume that E(X) = E(Y ) = 0 and Var(X) = Var(Y ) = 1, and (X, Y ) are sub-Gaussian. That is, we assume that our data are pre-normalized to have mean zero and variance one. We revisit the question of normalization in the supplementary material and show both theoretically and in numerical experiments that the rate of correlation estimation is not influenced by this step. In this situation, we consider estimating ρ = E(XY ) from the statistics shared by the two servers: viz., Server 1 releases T 1 (X 1 , . . . , X n ), and Server 2 releases T 2 (Y 1 , . . . , Y n ). To protect user privacy, we impose the differential privacy framework (see, e.g., Abowd et al. [2020], Bassily et al. [2014], Dwork [2006], Karwa and Vadhan [2017]) on T 1 and T 2 ; both of which must satisfy (ε 1 , δ 1 ) and (ε 2 , δ 2 )-DP constraints. For ease of reference, we will somewhat loosely denote the above by a server-level (ε 1 , ε 2 , δ 1 , δ 2 )-DP constraint and introduce specific definitions later. Such distributed privacy requirements are frequently used in federated learning. See, e.g., Auddy et al. [2024], Cai et al. [2024a,b,c], Shen et al. [2022], Wei et al. [2020Wei et al. [ , 2021] ] and references therein.
this section cite: ['b13', 'b26', 'b9', 'b17', 'b24', 'b23', 'b25', 'b0', 'b3', 'b10', 'b14', 'b2', 'b18', 'b21', 'b22']

Section: Main results
The key finding in this work is that the complexity of the correlation estimation in the above setup fundamentally depends on whether or not the statistics T 1 and T 2 are allowed to depend on one another. We now present our main results. Throughout this paper, we assume ε 1 , ε 2 ≤ C for a constant C > 0.
this section cite: []

Section: Non-interactive protocol
In our first set of results, we consider estimating ρ in the non-interactive (NI) framework of stricter privacy requirements, where T 1 and T 2 are constructed independently, i.e., without any interaction or information about one another. In this case, the differential privacy requirements on T 1 and T 2 are as follows. With X = (X 1 , . . . , X n ), Y = (Y 1 , . . . , Y n ), and similarly X ′ , Y ′ (with one data point replaced):
P(T 1 (X) ∈ A|X) ≤ exp(ε 1 )P(T 1 (X ′ ) ∈ A|X ′ ) + δ 1 P(T 2 (Y) ∈ A|Y) ≤ exp(ε 2 )P(T 2 (Y ′ ) ∈ A|Y ′ ) + δ 2 .
Let NI(ε 1 , ε 2 , δ 1 , δ 2 ) to be the class of all correlation estimators constructed using T 1 (X) and T 2 (Y) satisfying the above privacy requirement. The following theorem states the minimax rate for estimating ρ in this scenario. Theorem 1.1. The minimax rate for estimating correlation ρ via a non-interactive procedure satisfying server level (ε 1 , ε 2 , δ 1 , δ 2 )-DP constraints is given by inf ρ∈ NI(ε1,ε2,δ1,δ2)
sup ρ∈[-1,1] E ( ρ -ρ) 2 ≍ L n 1 nε 2 1 + 1 nε 2 2
for a factor L n of order at most O(log(n)), whenever δ 1 , δ 2 = o(n -1 ).
Note that the rate does not depend on δ's. This implies that our rate matching correlation estimator achieves (ε 1 , ε 2 , 0, 0)-DP, and is still rate optimal (up to logarithmic terms) even within NI(ε 1 , ε 2 , δ 1 , δ 2 ), the class of all non-interactive estimators satisfying (ε 1 , δ 1 ) and (ε 2 , δ 2 ) DP constraints for δ 1 , δ 2 are small positive numbers. The rate optimal estimator in this case is given by the correlation of privatized batch means from both servers.
It is useful to compare the above rate with the ones existing in the literature. Firstly, when (X, Y ) are jointly observed, and we impose (ε, δ)-central DP constraints on (X i , Y i ), the optimal correlation estimation rate is given by 1 n 2 ε 2 . See, e.g., Biswas et al. [2020], Cai et al. [2021]. As expected, when ε 1 = ε 2 = ε, this is better than the rate we observe in the current feature separated case, thus highlighting the cost of vertical federation. A second comparison can be made with component-wise local privacy rates, studied in Amorino and Gloter [2023]. The authors there show that in the vertically separated scenario, if we impose (ε 1 , 0) and (ε 2 , 0) local DP constraints, the minimax estimation rate for correlation is given by 1
this section cite: ['b4', 'b5', 'b1']

Section: Interactive protocol
We next move on to a larger class of estimators in the interactive (INT) framework, where we still require server level privacy, but one of the servers is allowed access to the privatized statistic output from the other. In other words, we allow the functions T 1 and T 2 to have one way interaction with each other. This requires making exactly one out of two possible choices. The first possibility is that when constructing T 2 , Server 2 has access to T 1 (X), in addition to its own data Y. The second possibility arises by analogously interchanging the roles of servers 1 and 2. To fix ideas, if we are in the first case, i.e server 2 gets to observe the transcript T 1 , before computing T 2 , the privacy requirements become:
P(T 1 (X) ∈ A|X) ≤ exp(ε 1 )P(T 1 (X ′ ) ∈ A|X ′ ) + δ 1 P(T 2 (Y, T 1 (X)) ∈ A|X, Y) ≤ exp(ε 2 )P(T 2 (Y ′ , T 1 (X)) ∈ A|X, Y ′ ) + δ 2 .
Replacing X with Y and the index 1 with 2 allows one to write the analogous privacy constraint in the second case where Server 1 has access to T 2 (Y). Let INT(ε 1 , ε 2 , δ 1 , δ 2 ) to be the class of all correlation estimators constructed using T 1 (X) and T 2 (Y, T 1 (X)) satisfying the above interactive privacy requirement. The following theorem states the minimax rate for estimating ρ in this scenario. Theorem 1.2. The minimax rate for estimating correlation ρ via a non-interactive procedure satisfying server level (ε 1 , ε 2 , δ 1 , δ 2 )-DP constraints is given by
inf ρ∈ INT(ε1,ε2,δ1,δ2) sup ρ∈[-1,1] E ( ρ -ρ) 2 ≍ L n 1 n(ε 1 ∨ ε 2 ) 2 + 1 n 2 ε 2 1 ε 2 2 .
for a factor L n of order at most O(log(n)), whenever δ 1 , δ 2 = o(n -1 ).
Note that unlike NI, in the INT rate, the dominating term depends on ε 1 ∨ ε 2 , i.e. the less stringent privacy requirement. The stronger privacy requirement i.e., ε 1 ∧ ε 2 appears in the second term, but its effect is mitigated by the better sample size factor n -2 . This leads to INT being a strictly better estimator than NI whenever ε 1 ̸ = ε 2 . An interesting special case is when X are public, meaning ε 1 is a constant, in which case we find (ε 2 , δ 2 )-central DP rates for correlation estimation.
The rate optimal estimator in the interactive case is borne out of a natural idea: the server with a less stringent privacy budget should share their statistics with the other server. That is, if ε 1 > ε 2 , we should allow T 2 to depend on T 1 (X) and Y. The situation is reversed if ε 2 > ε 1 .
In addition to point estimates ρ, we also derive asymptotically valid confidence intervals in both the non-interactive (NI) and interactive (INT) scenarios. That is, we find ( ρ
(NI) L,n , ρ (NI) U,n ) and ( ρ (INT) L,n , ρ (INT) U,n ) such that for fixed α ∈ (0, 1) P ρ (k) L,n ≤ ρ ≤ ρ (k) U,n → 1 -α as n → ∞, for k ∈ {NI, INT}.
We show that our estimation methods are minimax optimal by proving corresponding lower bounds, which to the best of our knowledge, has not been established previously under central differential privacy in a vertically distributed setting. While we follow the classical Le Cam framework, our main technical contribution is a direct control of KL divergence via Fisher information curvature bounds, yielding sharp lower bounds under both noninteractive and one-way interactive protocols. These bounds match our upper bounds up to constants in the Gaussian case and up to logarithmic factors in the sub-Gaussian case. Prior works, such as Hadar et al. [2019], bound KL via mutual information in communication constraint settings; we take a more direct route tailored to central DP. Unlike local DP lower bounds in Amorino and Gloter [2023], our approach handles the more delicate structure of central privacy with vertical data splitting.
The rest of the paper is organized as follows. In Sections 2 and 3 respectively, we describe non-interactive and interactive correlation estimators for bivariate Gaussian and bivariate sub-Gaussian distributions. Section 4 provides minimax lower bounds showing that our estimation procedures are nearly optimal in all cases. Finally, Section 5 shows numerical experiments to corroborate our theoretical results. All proofs are in the supplementary material.
this section cite: ['b12', 'b1']

Section: Non-interactive estimation methods
We first demonstrate an estimation procedure in the non-interactive paradigm. Here Server 1 and Server 2 construct and share T 1 (X) and T 2 (X) without knowledge of one another. As mentioned in the introduction T 1 (X) must satisfy (ε 1 , δ 1 )-DP and T 2 (X) must satisfy (ε 2 , δ 2 )-DP constraints. Our estimator is based on sharing privatized batch means. Choosing m ≥ 1 we separate the n observations in each server into batches of size m as follows: B j = {m(j -1) + 1, . . . , mj} for j = 1, . . . , k where k = ⌊ n m ⌋.
(1)
this section cite: []

Section: Non-interactive correlation estimation for Gaussian distribution
In this subsection, we assume that (X, Y ) ∼ N (0, Σ(ρ)) with (Σ(ρ)) 11 = (Σ(ρ)) 22 = 1 and (Σ(ρ)) 12 = ρ, the bivariate Gaussian distribution with
E(X) = E(Y ) = 0, Var(X) = Var(Y ) = 1 and correlation E(XY ) = ρ.
Our estimation procedure for ρ is through the product of sample means across multiple batches. In order to bound the sensitivity directly, i.e., without clipping, we will use the signs of X i and Y i in place of (X i , Y i ) themselves, to compute our correlation estimator.
X(j) = 1 m i∈Bj sign(X i ), and Ȳ (j) = 1 m i∈Bj sign(Y i )(2)
where B j are as defined in (1) for j = 1, . . . , k. To ensure (ε 1 , 0)-DP and (ε 2 , 0)-DP constraints each server adds Laplace noise to each batch mean and outputs the vectors T 1 (X), T 2 (Y) ∈ R m with elements:
(T 1 (X)) j = √ m( X(j) + Z (j) 1 ) and (T 2 (Y)) j = √ m( Ȳ (j) + Z (j) 2 ) for 1 ≤ j ≤ k,
where
Z (j) l indep ∼ Laplace 0, 2 mε l
for l = 1, 2. We can then compute
η XY = 1 k k j=1 (T 1 (X)) j (T 2 (Y)) j .(3)
Since (X, Y ) are bivariate Gaussians, the covariance above satisfies
E[ η XY ] = 2P(XY > 0) -1 = 1 - 2 arccos(ρ) π , (4
)
which leads to the method-of-moments based private correlation estimator:
ρ (G) NI := cos π 2 (1 -η (P ) XY ) = sin π η (P ) XY 2 .
We would like to emphasize that (4) is precisely where we use the assumption of Gaussianity on (X, Y ). Since the bivariate distribution is completely known once ρ is specified, we can explicitly write P(XY > 0) as a function of ρ, which in turn enables our sign-based estimation procedure. While this can be extended to other bivariate families which are specified by a single correlation parameter ρ, we do not discuss these details for brevity.
To create confidence intervals for ρ, let us define S 2 η to be the sample variance of {(T 1 (X)) j (T 2 (Y)) j : 1 ≤ j ≤ k}. Then we can define the confidence interval:
CI (G) NI (α) := ρ (G) NI - πSη 1-( ρ (G) NI ) 2 2 √ k z 1-α/2 , ρ (G) NI + πSη 1-( ρ (G) NI ) 2 2 √ k z 1-α/2(5)
where z 1-α/2 is the (1 -α/2)-th quantile of the standard Normal distribution.
this section cite: []

Section: Non-interactive correlation estimation for sub-Gaussian distributions
In general, we would deal with non-Gaussian data, and thus the sign-based procedure of the previous section would not be exact anymore. We will use a clipping based estimator for this case. For clipping parameters λ 1 , λ 2 > 0 to be chosen later we replace (2) by
X(j) = 1 m i∈Bj sign(X i )(|X i | ∧ λ 1 ) and Ȳ (j) = 1 m i∈Bj sign(Y i )(|Y i | ∧ λ 2 ) (6
)
where B j are as defined in (1) for j = 1, . . . , k. As before, each server adds Laplace noise to each batch mean and shares:
(T 1 (X)) j = √ m( X(j) + Z (j) 1 ) and (T 2 (Y)) j = √ m( Ȳ (j) + Z (j) 2 ) for 1 ≤ j ≤ k, where Z (j) l indep ∼ Laplace 0, 2λ l mε l
for l = 1, 2. Then we will estimate ρ by the quantity:
ρ (SG) NI = 1 k k j=1 (T 1 (X)) j (T 2 (Y)) j .(7)
Once again defining S 2 ρ to be the sample variance of {(T 1 (X)) j (T 2 (Y)) j : 1 ≤ j ≤ k}, we have the confidence interval:
CI (SG) NI (α) := ρ (SG) NI - S ρ √ k z 1-α/2 , ρ (SG) NI + S ρ √ k z 1-α/2(8)
where z 1-α/2 is the (1 -α/2)-th quantile of the standard Normal distribution. The following theorem states the results for correlation estimator under non-interactive protocol.
Theorem 2.1. The following results hold on the estimation error of ρ using a noninteractive componentwise privacy constrained estimator.
1. When (X, Y ) ∼ N (0, Σ(ρ)) with (Σ(ρ)) 11 = (Σ(ρ)) 22 = 1 and (Σ(ρ)) 12 = ρ, the estimator ρ (G) NI described in Section 2.1 satisfies ρ (G) NI ∈ NI(ε 1 , ε 2 , δ 1 , δ 2 ) and E( ρ (G) NI -ρ) 2 ≲ 1 n 1 ε 2 1 + 1 ε 2 2 if m = 8 ε 1 ε 2 ∨ 1. 2. When (X, Y ) have mean zero, variance one, X is η 1 -sub-Gaussian, Y is η 2 -sub- Gaussian, and E[XY ] = ρ, the estimator ρ (SG) NI described in Section 2.2 satisfies ρ (SG) NI ∈ NI(ε 1 , ε 2 , δ 1 , δ 2 ) and E( ρ (SG) NI -ρ) 2 ≲ log(n) n 1 ε 2 1 + 1 ε 2 2 if m = λ 1 λ 2 ε 1 ε 2 ∨ 1, λ 1 = 2η 1 log(n), and λ 2 = 2η 2 log(n).
3. For any fixed α ∈ (0, 1), the confidence intervals defined in (5) and (8) satisfy P(ρ ∈ CI (k) NI (α)) → 1 -α as n → ∞, for k ∈ {G, SG}.
this section cite: []

Section: Interactive estimation methods
We now show that the rates in the previous section can be improved if we allow a one-step interactive scheme between the two servers. To fix ideas, suppose ε 1 > ε 2 , i.e., the privacy requirement in the first server are less stringent than that in the second one. We will then share the private transcripts involving X to the second server containing the Y observations. This leads to an estimation error rate that improves over the non-interactive protocol.
this section cite: []

Section: Interactive correlation estimation for Gaussian distribution
In this case, our interactive estimator based on signs of (X, Y ) is as follows. Server 1 first communicates to Server 2 the privatized sign vector T 1 (X) with elements:
(T 1 (X)) i = exp(ε 1 ) + 1 (exp(ε 1 ) -1) (2S i -1) sign(X i ) for i = 1, . . . , n
where
S i iid ∼ Bernoulli exp(ε1) exp(ε1)+1
are independent sign flips introduced by Server 1 to protect the privacy of X i . Given T 1 (X) the second server first computes the covariance
η XY,int = 1 n n i=1 (T 1 (X)) i sign(Y i )
and then outputs the privatized version T 2 (Y, T 1 (X)) := η XY,int + Z where Z ∼ Laplace 0, 2(exp(ε 1 ) + 1) n(exp(ε 1 ) -1)ε 2 . (
As before we then have the private correlation estimator
ρ (G) INT = sin π η (P) XY,int 2 .
Similar to the non-interactive case, defining σ 2 η := 1 -exp(ε1
)-1 exp(ε1)+1 2 ( η (P )
XY,int ) 2 allows the confidence interval given by the following.
1. If c * = lim n→∞ 2 √ nσηε2 is finite, then the CI is ρ (G) INT ∓ π ση 1-( ρ (G) INT ) 2 2 √ n exp(ε1)+1 exp(ε1)-1 F -1 * (1 -α/2) (10
)
where for any x ∈ R we define F * (x) := P(Z XY + c * Z Lap ≤ x) for c * = 2/( √ n σ η ε 2 ) and Z Lap ∼ Laplace(0, 1).
2. If 1 √ nε2 diverges as n → ∞, then the CI is ρ (G) INT ± π 1-( ρ (G) INT ) 2 nε2 exp(ε1)+1 exp(ε1)-1 log(α) .
(11)
this section cite: []

Section: Interactive correlation estimation for sub-Gaussian distributions
Following previous sections, Server 1 will send to Server 2 the vector of privatized clipped observations T 1 (X) ∈ R n with elements (T 1 (X)
) i = [X i ] λ1 + Z 1i for a clipping parameter λ 1 > 0 and Z 1i iid ∼ Laplace (2λ 1 /ε 1 ) for i = 1, . . . , n. Then Server 2 can construct ρ (SG) INT = 1 n n i=1 [(T 1 (X)) i Y i ] λ2 + Z 2 .
In the above [x] t := sign(x)(|x| ∧ t) for any x ∈ R and t > 0. Here Z 2 ∼ Laplace (2λ 2 /nε 2 ) is Laplace noise added to ensure DP requirements. In addition to ρ (SG)
INT , Server 2 also outputs a privatized sample variance S 2 ρ of [(T 1 (X)) i Y i ] λ2 for i = 1, . . . , n. Then we have the confidence interval constructed as follows:
1. If c * = lim n→∞ 2λ2 √ nσρε2 is finite, then the CI is ρ (SG) INT - S ρ √ n F -1 * (1 -α/2), ρ (SG) int + S ρ √ n F -1 * (1 -α/2) (12
)
where for any x ∈ R we define F * (x) := P(Z XY + c * Z Lap ≤ x) for c * = 2λ 2 /( √ nS ρ ε 2 ), and Z Lap ∼ Laplace(0, 1).
2. If λ2 √ nε2 diverges as n → ∞, then the CI is
ρ (SG) INT + λ 2 nε 2 log(α), ρ (SG) int - λ 2 nε 2 log(α) . (13
)
The following theorem states the results for correlation estimator under the interactive protocol.
Theorem 3.1. The following results hold on the estimation error of ρ using the above privacy constrained interactive estimator.
1. When (X, Y ) ∼ N (0, Σ(ρ)) with (Σ(ρ)) 11 = (Σ(ρ)) 22 = 1 and (Σ(ρ)) 12 = ρ, the estimator ρ (G) INT described in Section 3.1 satisfies ρ (G) INT ∈ INT(ε 1 , ε 2 , δ 1 , δ 2 ) and E( ρ (G) INT -ρ) 2 ≲ 1 n(ε 1 ∨ ε 2 ) 2 + 1 n 2 ε 2 1 ε 2 2 . 2. When (X, Y ) have mean zero, variance one, X is η 1 -sub-Gaussian, Y is η 2 -sub- Gaussian, and E[XY ] = ρ, the estimator ρ (SG) INT described in Section 3.2 satisfies ρ (SG) INT ∈ INT(ε 1 , ε 2 , δ 1 , δ 2 ) and E( ρ (SG) INT -ρ) 2 ≲ 1 n(ε 1 ∨ ε 2 ) 2 + 1 n 2 ε 2 1 ε 2 2 if λ 1 = 2η 1 log(n) and λ 2 = 4(η 2 ∨ 1)(log(n)) 2 /(ε 1 ∧ 1).
3. For any fixed α ∈ (0, 1), under their respective assumptions, the confidence intervals defined in (10), ( 11), (12), and (13) satisfy P(ρ ∈ CI
INT (α)) → 1 -α as n → ∞, for k ∈ {G, SG}.
this section cite: []

Section: Minimax lower bounds
In this section, we show that the private correlation estimators derived in the previous section are in fact minimax optimal in many cases. Our proof strategy is based on bounding Fisher information of the private transcripts and then using Van Trees inequality. We recall some standard results from parameter estimation theory in the next subsection.
this section cite: []

Section: Fisher information and Van Trees inequality
Let θ be a real-valued parameter taking an unknown value in some interval [a, b]. We observe some random variable (or vector) X with distribution P (x|θ) parameterized by θ.
Assume that P (•|θ) is absolutely continuous with respect to a reference measure µ, for each θ ∈ [a, b], and dP (•|θ)  dµ (x) is differentiable with respect to θ ∈ (a, b) for µ-almost all x. Then the Fisher information of θ w.r.t. X, denoted as I F (X; θ), is defined as
I F (X; θ) ≜ ∂ ∂θ ln dP (•|θ) dµ (x) 2 dP (x|θ). (14
)
The following inequality is well-known. See for example Gill and Levit [1995].
this section cite: ['b11']

Section: Lemma 1 (Van Trees inequality).
Let θ be a real parameter with prior density ζ supported on [a, b] ⊂ R, and let
X ∼ P (• | θ) with conditional density p(x | θ) = dP (•|θ) dµ (x).
this section cite: []

Section: Under some regularity conditions we have that for every estimator
θ = θ(X) with E[( θ -θ) 2 ] < ∞ under the joint law of (X, θ) satisfies E ( θ -θ) 2 ≥ 1 E θ [I F (X; θ)] + I F (ζ) , E θ [I F (X; θ)] = b a I F (X; θ) ζ(θ) dθ, (15
)
where I F (ζ) := b a ζ ′ (θ) 2 ζ(θ)
dθ is the prior Fisher information.
The "regularity conditions" in Lemma 1 are to ensure that one can apply the dominated convergence theorem to exchange certain integrals and differentiations in the calculus. See for example Vaart [1998]. Additionally, assume that
I F (X; θ + ϵ) = I F (X; θ)(1 + η(ϵ)) (16
)
where η(ϵ) < C η for all |ϵ| < c 0 for some numerical constants c 0 < 1 and C η > 0.
this section cite: []

Section: Non interactive
For the non-interactive protocols the servers output transcripts T 1 and T 2 which are (ε 1 , δ 1 ) and (ε 2 , δ 2 )-DP respectively. The transcripts are based on;y on the data from their own servers. An estimator ρ is then calculated after combining T 1 and T 2 .
Our lower bound is shown by the difficulty of correlation estimation when ρ = 0. Let us denote the transcripts by T ≡ (T 1 , T 2 ). As a first step, the next lemma shows that I F (T ; 0) is smaller than a quantity involving the sample size n and the privacy parameters
ε 1 , ε 2 . Lemma 2. Assume that for k = 1, 2 , δ k log(1/δ k ) = O(ε 2 k ).
Let us denote the Fisher information for the transcripts T by I F (T ; ρ). We have that
I F (T ; 0) ≤ 8 π (nε 2 1 ∧ nε 2 2 ).
The local regularity assumption in (16) at ρ = 0 ensures that up to a constant factor, the bound from the above lemma carries over to I F (T ; ρ) for |ρ| ≤ c 0 . For a suitable choice of prior density ζ this in turn implies an upper bound on E 0 [I F (T ; 0)] and allows us to complete the proof by using Van-Trees inequality (Lemma 1). We then have the following lower bound on the minimax risk for estimating ρ in the non-interactive setting. Theorem 4.1. Assume that δ k = o(n -1-ω ) for k = 1, 2 and n(ε 2 1 ∧ ε 2 2 ) → ∞. Then for non interactive protocols the minimax rate is lower bounded by
inf ρ∈NI(ε1,ε2,δ1,δ2) sup ρ∈[-1,1] | ρ -ρ| 2 ≳ 1 n + 1 nε 2 1 + 1 nε 2 2 .
Remark 4.1. The assumption n(ε 2 1 ∧ ε 2 2 ) → ∞ assumes that the minimax rate is going to zero ensuring consistent estimation of ρ in the first place.
this section cite: []

Section: Interactive
We next allow one way interaction among the servers where either of the server can share its transcripts with the other server. Let us denote the set of protocols which allow allow interaction from server 1 to 2 as Π 1→2 , i.e server 2 gets to observe the transcript T 1 , before computing T 2 . We first show the following upper bound on
I F (Π 1→2 ; 0). Lemma 3. Assume that δ 1 log(1/δ 1 ) = o(ε 2 1 ) and δ 2 log(1/δ 2 ) 2 = o(nε 2 1 ε 2 2 ).
Let us denote the Fisher information for the transcripts Π 1→2 by I F (Π 1→2 ; ρ). We have that
I F (Π 1→2 ; 0) ≤ nε 2 1 ∧ n 2 ε 2 1 ε 2 2 .
If we denote the protocol which allow interaction from server 2 to 1 we can show that
I F (Π 2→1 ; 0) ≤ nε 2 2 ∧ n 2 ε 2 1 ε 2 2 . Since we allow for either of the protocols Π ≡ (Π 1→2 , Π 2→1 ) we have that I F (Π; 0) ≤ I F (Π 1→2 ; 0) ∨ I F (Π 2→1 ; 0) ≤ (nε 2 1 ∧ n 2 ε 2 1 ε 2 2 ) ∨ (nε 2 2 ∧ n 2 ε 2 1 ε 2 2 ).
(17) Similar to the non-interactive case we can then use the local regularity assumption in ( 16) and a suitable prior density ζ with Van Trees inequality, leading to the following lower bound on the minimax risk in the interactive setting. Theorem 4.2. Assume that for
k = 1, 2 δ k = o(n -1-ω ) for ω > 0, n(ε 2 1 ∨ ε 2 2 ) → ∞ and n 2 ε 2 1 ε 2 2 → ∞.
Then for interactive protocols the minimax rate is lower bounded by inf ρ∈INT(ε1,ε2,δ1,δ2)
sup ρ∈[-1,1] | ρ -ρ| 2 ≳ 1 n + 1 n(ε 2 1 ∨ ε 2 2 ) + 1 n 2 ε 2 1 ε 2 2 . Remark 4.2.
The assumption n(ε 2 1 ∨ ε 2 2 ) → ∞ and n 2 ε 2 1 ε 2 2 → ∞ assumes that the minimax rate is going to zero, ensuring consistent estimation of ρ in the first place.
this section cite: []

Section: Numerical experiments
We evaluate our non-interactive sign-batch (NI) and interactive sign-flip (INT) estimators across different parameter settings. All our codes can be found at https://  github.com/abhinavc3/distributed-correlation.
this section cite: []

Section: Simulation experiments
In our experiments we write non-normalized to mean that the mean and variances of the marginal distributions are known, and normalized to mean that they are unknown and estimated. We use two generative models.
• Gaussian: (X, Y ) ∼ N (µ, 2Σ(ρ)) with µ = (0.5, 0.5) ⊤ , and Σ(ρ) given by Var(X) = Var(Y ) = 1 and Corr(X, Y ) = ρ. We run each estimator with and without the private normalization step.
• Bounded-factor (sub-Gaussian):
X = U + E 1 , Y = U + E 2 with U ∼ Unif - √ 3 ρ, √ 3 ρ and E i ∼ Unif -3(1 -ρ), 3(1 -ρ)
, so each marginal is centred, variance-one, and bounded hence sub-Gaussian.
For every design point we record mean-squared error (MSE), average confidence-interval (CI) length, empirical coverage (1 -α = 0.95) and the mean CI offset band
E[CI L -ρ] → E[CI U -ρ]
where CI L and CI U are the upper and lower confidence bars. In practice it is sufficient to use the confidence intervals from ( 10) and ( 12) since ( 11) and ( 13) are respectively the limiting versions of the above two.
Parameter Grid. We vary our parameters as below, with 250 replications for each cell:
• Sample size: n ∈ {1000, 1500, 2500, 4000, 6000, 9000}.
• Correlation: ρ ∈ {0, 0.15, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9}.
• Privacy budget: (ε 1 , ε 2 ) ∈ {(0.5, 0.5), (1, 1), (1.5, 0.5)}. Figure 1 compares the mean CI offset bands for n = 1500 and the budget (ε 1 , ε 2 ) = (1.5, 0.5).
With and without normalization the ribbons coincide, indicating that private normalization is cost-free. Figure 2 shows CI width and coverage versus n at ρ = 0.5; both variants adhere to the nominal 95% band. Figure 3 confirms that INT is uniformly more efficient than NI, while normalization leaves MSE unchanged (largest relative difference < 2%).  -0.5 0.0 0.5 0.5 1.0 1.5 2.0 2.5 ε corr mean(CI) for ρ Non-interactive -0.5 0.0 0.5 0.5 1.0 1.5 2.0 2.5 ε corr mean(CI) for ρ Interactive Figure 4: Mean confidence interval bands for non-interactive (left) and interactive (right) methods for estimating the correlation between age and BMI in the Health and Retirement Study (HRS) data. The black dotted line indicates the non-private estimator.
For the sake of brevity we only show the MSE plots (Figure 3 right). The CI bands, coverage and width plots are deferred to the supplementary material.
this section cite: []

Section: Real data experiments
We illustrate our methods using data from the Health and Retirement Study (HRS), a longitudinal survey of older adults in the United States. We focus on two variables-age and body mass index (BMI)-from Wave 2 (year 1993-94) corresponding to around 20k individuals. In this demographic, age and BMI are known to exhibit a mild negative correlation.
We consider a distributed scenario in which the two variables reside on separate servers, and the goal is to estimate their Pearson correlation coefficient ρ. Each server first applies a Central differentially private (CDP) normalization so that the privatized features have approximately zero mean and unit variance. Specifically, we allocate ε = 0.1 for each of the mean and standard deviation estimates. The clipping bounds are chosen based on domain knowledge-[45, 90] for age and [15,35] for BMI-demonstrating a setting where the privacy mechanism leverages prior information rather than data-dependent thresholds.
After normalization, we apply both the non-interactive (NI) and interactive (INT) protocols to obtain private confidence intervals for the estimated correlation ρ. We compare these to the non-private benchmark while varying the privacy budget ε corr , keeping it equal across the two servers. Results are given in Figure 4. As ε corr increases, the private intervals contract and concentrate around the non-private ρ. Moreover, for a fixed ε corr , the INT intervals are consistently shorter than their NI counterparts. Notably, at ε corr = 1, the interactive CI excludes zero while the non-interactive CI includes it-illustrating that privacy noise can increase uncertainty and, in some cases, prevent rejection of the null hypothesis ρ = 0.
this section cite: []

Section: Discussion
Across both distributions and all privacy budgets explored, INT consistently outperforms NI, while the required private normalization step incurs no measurable loss in bias, MSE or interval width. These findings support the theoretical claim that normalization's privacy cost is dominated by the subsequent correlation release.
We discuss two important directions of future work. First, allowing multiple features per server-rather than a single feature-introduces new challenges, particularly in handling inter-feature correlations and maintaining privacy in higher dimensions. Second, extending our methods to heavy-tailed distributions would broaden applicability, as such data often arise in practice and require more robust estimation techniques.
this section cite: []

Section: References
Ref_id:b0 Title: The modernization of statistical disclosure limitation at the us census bureau Year: (2020)
Ref_id:b1 Title: Minimax rate for multivariate data under componentwise local differential privacy constraints Year: (2023)
Ref_id:b2 Title: Minimax and adaptive transfer learning for nonparametric classification under distributed differential privacy constraints Year: (2024)
Ref_id:b3 Title: Private empirical risk minimization: Efficient algorithms and tight error bounds Year: (2014)
Ref_id:b4 Title: Coinpress: Practical private mean and covariance estimation Year: (2020)
Ref_id:b5 Title: The cost of privacy: Optimal rates of convergence for parameter estimation with differential privacy Year: (2021)
Ref_id:b6 Title: Federated nonparametric hypothesis testing with differential privacy constraints: Optimal rates and adaptive tests Year: (2024)
Ref_id:b7 Title: Optimal federated learning for nonparametric regression with heterogeneous distributed differential privacy constraints Year: (2024)
Ref_id:b8 Title: Optimal federated learning for functional mean estimation under heterogeneous privacy constraints Year: (2024)
Ref_id:b9 Title: Vafl: a method of vertical asynchronous federated learning Year: (2020)
Ref_id:b10 Title: Differential privacy Year: (2006)
Ref_id:b11 Title: Applications of the van trees inequality: a bayesian cramér-rao bound Year: (1995)
Ref_id:b12 Title: Communication complexity of estimating correlations Year: (2019)
Ref_id:b13 Title: Rachel Cummings, et al. Advances and open problems in federated learning Year: (2021)
Ref_id:b14 Title: Finite sample differentially private confidence intervals Year: (2017)
Ref_id:b15 Title: A review of applications in federated learning Year: (2020)
Ref_id:b16 Title: Federated learning: Challenges, methods, and future directions Year: (2020)
Ref_id:b17 Title: Vertical federated learning: Concepts, advances, and challenges Year: (2024)
Ref_id:b18 Title: From distributed machine learning to federated learning: In the view of data privacy and security Year: (2022)
Ref_id:b19 Title: Introduction to nonparametric estimation. Springer series in statistics Year: (2009)
Ref_id:b20 Title: Asymptotic statistics. Cambridge series in statistical and probabilistic mathematics Year: (1998)
Ref_id:b21 Title: Federated learning with differential privacy: Algorithms and performance analysis Year: (2020)
Ref_id:b22 Title: Userlevel privacy-preserving federated learning: Analysis and performance optimization Year: (2021)
Ref_id:b23 Title: Guihai Chen, and Thilina Ranbaduge. Vertical federated learning: Challenges, methodologies and experiments Year: (2022)
Ref_id:b24 Title: Privacy preserving vertical federated learning for tree-based models Year: (2020)
Ref_id:b25 Title: Parallel distributed logistic regression for vertical federated learning without third-party coordinator Year: (2019)
Ref_id:b26 Title:  Year: (2021)
