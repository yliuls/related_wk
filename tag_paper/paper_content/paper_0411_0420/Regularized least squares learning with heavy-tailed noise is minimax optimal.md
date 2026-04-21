Title: Regularized least squares learning with heavy-tailed noise is minimax optimal
Abstract: This paper examines the performance of ridge regression in reproducing kernel Hilbert spaces in the presence of noise that exhibits a finite number of higher moments. We establish excess risk bounds consisting of subgaussian and polynomial terms based on the well known integral operator framework. The dominant subgaussian component allows to achieve convergence rates that have previously only been derived under subexponential noise-a prevalent assumption in related work from the last two decades. These rates are optimal under standard eigenvalue decay conditions, demonstrating the asymptotic robustness of regularized least squares against heavy-tailed noise. Our derivations are based on a Fuk-Nagaev inequality for Hilbert-space valued random variables.

Section: Introduction
Given two random variables X and Y , we seek to empirically minimize the expected squared error R(f ) := E (Yf (X)) 2   over functions f in a reproducing kernel Hilbert space H consisting of functions from a topological space X to R. We consider the standard model
Y = f ⋆ (X) + ε
with the regression function f ⋆ : X → R and noise variable ε satisfying E[ε|X] = 0. Given n independent sample pairs (X i , Y i ) drawn from the joint distribution of X and Y , we investigate the classical ridge regression estimate
f α := arg min f ∈H 1 n n i=1 (Y i -f (X i )) 2 + α∥f ∥ 2 H (1
)
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
with regularization parameter α > 0. We adopt the well-known perspective going back to the pathbreaking work [1][2][3][4], which characterizes f α as the solution of a linear inverse problem in H obtained by performing Tikhonov regularization [5] on a stochastic discretization of the integral operator induced by the kernel of H and the marginal distribution of X. Since its inception, this approach has been refined and generalized in a multitude of ways, including more general learning settings and alternative algorithms and applications. We refer the reader to [6][7][8][9][10][11][12][13][14][15][16][17][18][19][20][21][22][23] and the references therein for an overview. A common theme in the above line of work is the derivation of confidence bounds of the excess risk
R( f α ) -R(f ⋆ ) = E[( f α (X) -f ⋆ (X)) 2 ]
i.e., with high probability over the draw of the sample pairs under appropriate regularity assumptions about the regression function f ⋆ and distributional assumptions about ε.
Heavy-tailed noise. In this work, we assume that the real-valued random variable ε has only a finite number of higher conditional absolute moments, i.e., there exists some q ∈ N, q ≥ 3 such that
E[|ε| q |X] < Q < ∞ almost surely.
(
This setting covers noise associated with distributions without a moment generating function-for example the t-distribution, Fréchet distribution, Pareto distribution and Burr distribution (correspondingly centered). In such a setting, the family of Fuk-Nagaev inequalities [24,25] provides sharp nontrivial tail bounds beyond Markov's inequality for sums of heavy-tailed real random variables. These results show that the tail is dominated by a subgaussian term [26] in a small deviation regime (reflecting the central limit theorem) and a polynomial term in a large deviation regime. In order to apply this fact to the integral operator approach, we modify a vector-valued version of the Fuk-Nagaev inequality going back to [27] for random variables taking values in Hilbert spaces. In a practical context, heavy-tailed noise satisfying the moment condition (2) plays a role in fields such as finance, insurance, communication networks and atmospherical sciences [28,29].
Prior work: Bernstein condition. In the aforementioned context of spectral regularization algorithms in kernel learning, existing work generally assumes that ε is subexponential. 1 In particular, the so-called Bernstein condition 2 requires the existence of constants σ, Q > 0 almost surely satisfying
E[|ε| q |X] ≤ 1 2 q!σ 2 Q q-2 almost surely (3
)
for all q ≥ 2. This condition allows to apply a Hilbert space Bernstein inequality [31] to the wellknown integral operator framework in order to obtain convergence results. We refer the reader to [3,[6][7][8][9][10][11]13] for a selection of results in this setting. To our knowledge, all results obtaining optimal rates in this setting rely on the Bernstein tail bound. The importance of the Bernstein inequality in the context of this work is emphasized by the effective dimension [3,32], which measures the capacity of the hypothesis space H relative to the choice of the regularization parameter α and the marginal distribution of X in terms of the eigenvalues of the integral operator. When used as a variance proxy in the Bernstein inequality, the effective dimension is the central tool that allows to derive minimax optimal rates under assumptions about the eigenvalue decay, as first shown by [3] and subsequently refined in the aforementioned work. Due to this elegant connection between eigenvalue decay and concentration, the integral operator formalism has been predominantly focused around the assumption (3) over the last two decades. Similar approaches based on the Bernstein inequality with suitable variance proxies are commonly applied across a variety of estimation techniques in order to obtain fast rates [e.g. 33,34].
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b30', 'b2', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b12', 'b2', 'b31', 'b2', 'b32', 'b33']

Section: Overview of contributions.
In this work, we show that the rates derived under the Bernstein condition (3) in the mentioned literature can equivalently be obtained with the significantly less restrictive higher moment assumption (2) with the same regularization parameter schedules. We consider both the capacity independent setting (i.e., without assumptions about the eigenvalue decay of the integral operator [4,6]) and the more common capacity dependent setting involving the effective dimension [e.g. 3,11,35]. Even though the capacity dependent results are sharper, we dedicate a separate discussion to the capacity independent setting, as it allows a less technical presentation and a simplified and insightful asymptotic dicussion. In both settings, we base our analysis on a Hilbert space version of the Fuk-Nagaev inequality, providing excess risk bounds that exhibit both subgaussian and polynomial tail components. The dominant subgaussian term allows to asymptotically recover the familiar bounds known from the subexponential noise scenario. In the capacity dependent setting, we use the effective dimension not only as variance proxy, but also as a proxy for the higher moments occurring in the Fuk-Nagaev inequality-the resulting bound is sharp enough so that standard assumptions about the eigenvalue decay lead to known optimal convergence rates. This technique directly generalizes the aforementioned approach based on the Bernstein inequality.
Practical implications, future work and limitations. The square loss is often not used in practice when one expects heavy-tailed noise, as it is sensitive to outliers when used without regularization [36]. However, when used with regularization in the presence of noise of the form (2), we show that it exhibits a certain degree of robustness. In particular, (i) it asymptotically achieves the optimal rates with high probability known from the light-tailed setting with the same regularization schedule, (ii) in a large sample setting, the confidence behavior of the excess risk is essentially subexponential, (iii) in a small sample setting, the confidence behavior is polynomial and stronger regularization is required due to the impact of the heavy tails.
We focus on the original well-specified kernel ridge regression setting as investigated by [3] in order to simplify the presentation and highlight the key arguments. However, we expect our approach to transfer to other settings, for example involving more general source conditions [6,37], general spectral filter methods [9,20], misspecified models [11,38], the kernel conditional mean embedding with unbounded kernels on the target space [17], high-and infinite-dimensional output spaces [15,16] and many other settings allowing for the application of the integral operator formalism. We believe that kernel regression with unbounded kernels admitting a finite higher moment can be analyzed with a similar technical approach as the one presented here. Let us mention some limitations of the present work. While our results show a certain degree of robustness of regularized least squares against heavy-tailed noise for q ≥ 3, q ∈ N, we expect our results to directly transfer to all real q > 2, as versions of the real-valued Fuk-Nagaev bound cover this case [39,40]. Currently, this restriction of our results exclusively depends on the validity of the Fuk-Nagaev bound in Hilbert spaces for these q as an artefact of the proof technique by [27], which we modify. Furthermore, for q < 2, it is clear that the square loss is not necessarily well-defined and a different loss such as the Cauchy loss should be used [41]. Our results demonstrate that, when operating in a high confidence setting, heavy-tailed noise may require a significantly higher level of regularization than light-tailed noise, making empirical regularization parameter choice rules as a function of the confidence level very important. Extending the analysis of classical parameter choice rules for deterministic inverse problems [42] to the stochastic setting based on heavy-tailed noise may therefore be an interesting future direction.
Other related work. We are not aware of any specific analysis of kernel ridge regression and the integral operator formalism in the heavy-tailed scenario given by (2) in the literature. However, there exists a wide variety of related results for regression with heavy-tailed noise and robust estimationwe put our results in the context of the most important related work. Optimal rates for (unpenalized) least squares regression over nonparametric hypothesis spaces can generally only be derived in the empirical process context when q in condition (2) is large enough with respect to suitable metric entropy requirements, see [43][44][45][46]. In comparison, our setting allows to recover optimal rates for the reproducing kernel Hilbert space scenario with a regularization schedule which is independent of q. For linear models over finite basis functions, [47] derives exponential concentration of the ridge estimator under finite variance on the noise for fixed α. We also highlight the field of robust estimation techniques outside of the standard least squares context, see e.g. [36,[47][48][49][50][51] and the references therein. While the analysis of robust finite-dimensional linear regression under heavytailed noise requires discussions of the distribution of the covariates and their covariance matrix (often under variance-kurtosis equivalence [52][53][54]), we impose the typical assumption that the kernel is bounded, leading to subgaussian concentration of the embedded covariates in the potentially infinitedimensional feature space. Recently, [41] derived nearly optimal rates for kernel ridge regression with Cauchy loss under (2) with q > 0 depending on the Hölder continuity parameter of the target function using more classical arguments. Finally, from a more technical perspective, the approach by [55] shares similarities with the methods applied in our paper: The authors use a real-valued Fuk-Nagaev inequality to bound the stopping time complexity of stochastic gradient descent for ordinary least squares regression. However, [55] provides results only for the finite-dimensional setting based on a martingale decomposition by assuming a lower bound on the minimal eigenvalue of the covariance matrix. In contrast, our work targets the infinite-dimensional setting based on inverse problem theory without such a lower bound and explicitly proves minimax optimality.
Structure of this paper. We introduce our notation and basic preliminaries in Section 2. In Section 3, we provide the excess risk bound for the capacity independent setting and derive corresponding rates. Section 4 contains an excess risk bound based on the effective dimension and recovers rates which are known to be minimax optimal also in the subexponential noise setting. Finally, in Section 5, we briefly discuss the Fuk-Nagaev inequality used to derive our results. Appendix A contains a numerical experiment which confirms the behavior of the excess risk described by our theoretical results. We report all proofs for the results in the main text in Appendix B and provide additional technical results as individual appendices.
this section cite: ['b3', 'b5', 'b2', 'b10', 'b34', 'b35', 'b2', 'b5', 'b36', 'b8', 'b19', 'b10', 'b37', 'b16', 'b14', 'b15', 'b38', 'b39', 'b26', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b35', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b40', 'b54', 'b54']

Section: Preliminaries
We assume that the reader is familiar with the analysis of linear Hilbert space operators [56,57] and the basic theory of reproducing kernel Hilbert spaces [34,58]. Let π denote the marginal distribution of X and L 2 (π) denote the space of real-valued Lebesgue square integrable functions with respect to π. We write L(H 1 , H 2 ) for the space of bounded linear operators between Hilbert spaces H 1 and H 2 with operator norm ∥•∥ and abbreviate L(H 1 ) = L(H 1 , H 1 ). We additionally consider the space of Hilbert-Schmidt operators S 2 (H 1 , H 2 ) ⊂ L(H 1 , H 2 ) and the space of trace class operators S 1 (H 1 , H 2 ) ⊂ S 2 (H 1 , H 2 ) with norms ∥•∥ S1(H1,H2) and ∥•∥ S2(H1,H2) and the trace tr(•). The adjoint of A ∈ L(H 1 , H 2 ) is written as A * ∈ L(H 2 , H 1 ).
this section cite: ['b55', 'b56', 'b33', 'b57']

Section: Reproducing kernel Hilbert space
We consider the reproducing kernel Hilbert space (RKHS) H consisting of functions from X to R induced by the symmetric positive semidefinite kernel k : X × X → R. with canonical feature map ϕ(x) : X → H, x → k(•, x), i.e. we have the reproducing property f (x) = ⟨f, ϕ(x)⟩ H for all x ∈ X and f ∈ H. Assumption 2.1 (Domain and kernel). We impose the following standard assumptions throughout this paper in order to avoid issues related to measurability and integrability [34, Section 4.3]:
(i) X is a second-countable locally compact Hausdorff space, H is separable (this is satisfied if k is continuous, given that X is separable), (ii) k(•, X) is almost surely measurable in its first argument, (iii) k(X, X) ≤ κ 2 almost surely for some finite constant κ.
All assumptions above hold for commonly used continuous radial kernels on R d such as the Gaussian kernel and the Matèrn kernel. However, the boundedness assumption is violated for polynomial kernels unless X is bounded almost surely.
Integral and covariance operators. Under Assumption 2.1, we may consider the typical linear operators associated with π and k. We consider the embedding operator
I π : H ↩→ L 2 (π), f → [f ] π
identifying f ∈ H with its equivalence class [f ] π ∈ L 2 (π). The adjoint I * π : L 2 (π) → H is given by
I * π f = X ϕ(x)f (x) dπ(x) ∈ H, f ∈ L 2 (π).
We obtain the self-adjoint integral operator T π := I π I * π : L 2 (π) → L 2 (π) induced by k and π as
T π f = X k(•, x)f (x) dπ(x), f ∈ L 2 (π).
The self-adjoint kernel covariance operator C π := I * π I π : H → H is given by
C π = X ϕ(x) ⊗ ϕ(x) dπ(x).
Assumption 2.1(iii) ensures that we have ∥I π ∥ = ∥I * π ∥ ≤ κ as well as ∥T π ∥ = ∥C π ∥ ≤ κ 2 . Moreover, the operators I π and I * π are Hilbert-Schmidt and both T π and C π are therefore self-adjoint positive semidefinite and trace class. By the polar decomposition of I π and I * π , there exist partial isometries U : H → L 2 (π) and Ũ : L 2 (π) → H such that
I π = U (I * π I π ) 1/2 = U C 1/2 π and I * π = Ũ (I π I * π ) 1/2 = Ũ T 1/2 π .(4)
In particular, we have ∥[f ] π ∥ L 2 (π) = ∥I π f ∥ L 2 (π) = ∥C 1/2 f ∥ H for all f ∈ H. We will also frequently use the fact that Assumption 2.1(iii) implies sup x∈X |f (x)| ≤ κ∥f ∥ H .
this section cite: []

Section: Ridge regression
We introduce the standard integral operator formalism for ridge regression in RKHSs, see e.g. [3]. We consider the standard L 2 (P)-orthogonal decomposition of Y with respect to the closed subspace L 2 (P, σ(X)) ⊂ L 2 (P) of σ(X)-measurable functions given by
Y = f ⋆ (X) + ε(5)
with the regression function f ⋆ (X) = E[Y |X] ∈ L 2 (P) and noise variable ε ∈ L 2 (P) satisfying E[ε|X] = 0. Based on the representation (5), we have R(f ) = ∥ff ⋆ ∥ 2 L 2 (π) + ∥ε∥ 2 L 2 (P) for all f ∈ L 2 (π) and hence the excess risk satisfies R(f
) -R(f ⋆ ) = ∥f -f ⋆ ∥ 2 L 2 (π) .
Regularized population solution. We define the regularized population solution
f α : = arg min f ∈H E[(Y -f (X)) 2 ] + α∥f ∥ 2 H = arg min f ∈H ∥I π f -f ⋆ ∥ 2 L 2 (π) + α∥f ∥ 2 H ,
with α > 0, which is alternatively expressed as
f α = (C π + α Id H ) -1 I * π f ⋆ ∈ H(6)
with the identity operator Id H on H.
this section cite: ['b2']

Section: Regularized empirical solution.
We consider sample pairs (X 1 , Y 1 ), . . . , (X n , Y n ) ∼ L(X, Y ) independently obtained from the joint distribution of X and Y . We define the empirical versions of the operators above in terms of
I * π f := 1 n n i=1 ϕ(X i )f (X i ) ∈ H, f ∈ L 2 (π)
as well as
C π := 1 n n i=1 ϕ(X i ) ⊗ ϕ(X i ).
The empirical solution of the learning problem in H with regularization parameter α > 0 is given by the empirical analogue of (6), which we obtain in terms of
f α = ( C π + α Id H ) -1 Υ ∈ H,(7)
where the empirical right hand side Υ ∈ H of the inverse problem is given by
Υ := 1 n n i=1 ϕ(X i )Y i = I * π f ⋆ + 1 n n i=1 ϕ(X i )ε i .(8)
Here, we use the orthogonal decomposition Y i = f ⋆ (X i ) + ε i in the second equivalence. We note that Υ directly serves as an empirically evaluable unbiased estimate of I * π f ⋆ , as I * π f ⋆ itself cannot be empirically evaluated because f ⋆ is unknown. As usual, we interpret the above objects as random variables depending on the product measure P ⊗n through their definition based on the observation pairs (X i , Y i ). In practice, the empirical solution f α can be evaluated in terms of the classical representer theorem, see e.g. [1, Proposition 8].
this section cite: []

Section: Distributional assumptions
We list the assumptions we impose upon the distributions of Y , X and ε. Assumption 2.2 (Moment condition). We consider the model given by (5) and assume that we have almost surely
E[ε|X] = 0, E ε 2 |X < σ 2 and E[|ε| q |X] < Q,(MOM)
for some constants σ 2 > 0, Q > 0 and q ∈ N, q ≥ 3.
We now introduce a classical smoothness assumption in terms of a Hölder source condition [59]. Assumption 2.3 (Source condition). We define the set Ω(ν, R) := {T ν π f | ∥f ∥ L 2 (π) ≤ R} ⊂ L 2 (π) and assume f ⋆ ∈ Ω(ν, R) (SRC) for some smoothness parameter ν ≥ 1/2 and R > 0.
We give the definition of the source set Ω(ν, R) with respect to L 2 (π) and not with respect to H, which is also commonly found in the literature. Furthermore, the source condition is sometimes described in terms of so-called interpolation spaces or Hilbert scales. Our definition can equivalently be expressed in terms of these concepts by appropriately reparametrizing ν, see e.g. [6,11,59] for more details. Remark 2.4 (Well-specified case). In this work, we explicitly consider the condition ν ≥ 1/2, which implies the well-specified setting in which we have Ω(ν, R) ⊂ I π (H). We note the case 0 ≤ ν < 1/2 covers the misspecified setting, in which Ω(ν, R) is allowed to contain elements from L 2 (π) \ I π (H). We expect our approach to transfer to the misspecified setting by combining it with recent technical arguments from the literature which are outside the scope of this work [11,15,16,38,60].
this section cite: ['b4', 'b58', 'b5', 'b10', 'b58', 'b10', 'b14', 'b15', 'b37', 'b59']

Section: Capacity-free excess risk bound
We now provide an excess risk bound and corresponding rates for kernel ridge regression in the heavy-tailed noise setting without additional assumptions about the eigenvalue decay of T π . We present this setting separately from the capacity-based results in the next section, as it allows for a clearer comparison with bounds based on subexponential noise and a simplified asymptotic discussion.
Proposition 3.1 (Main excess risk bound). Let (MOM) and (SRC) be satisfied. For all δ ∈ (0, 1) and n ∈ N such that
C κ log(6/δ) ≤ α √ n, C κ := 2(1 + √ κ) • max{1, κ 2 },(9)
we have
∥I π f α -f ⋆ ∥ L 2 (π) ≤ Rα min{ν,1} + C ⋄ √ α log(6/δ) n + α 2 min{ν,1} log(6/δ) n + η(δ, n) ,
with confidence 1δ, with η(δ, n) := max Q δn q-1
1/q , σ log(6c 1 /δ) n ,
where 0 < C ⋄ is given in (29) and c 1 ≥ 1 is the constant from Proposition 5.1 depending only on q.
Just as in the light-tailed setting, Proposition 3.1 shows that the optimal excess risk is achieved by balancing the contributions of the approximation error (e.g. the model bias) and sample error (e.g. the model variance) by choosing a suitable regularization parameter α depending on n and δ. The term Rα min{ν,1} quantifies the approximation error based on the smoothness of f ⋆ and exhibits the typical saturation effect of ridge regression: the fact that the convergence speed cannot be improved beyond a smoothness level ν = 1 [e.g. 61,62]. The key difference to known results for subexponential noise in this setting [4,6] is the Fuk-Nagaev term η(δ, n) appearing in the sample error, which introduces an additional polynomial dependence on δ and n. We now investigate the consequences of this term.
this section cite: ['b60', 'b61', 'b3', 'b5']

Section: Confidence regimes.
We split the confidence scale into two disjoint intervals depending on whether the subgaussian component or the polynomial component dominates in the term η(δ, n). For n ∈ N, q ≥ 3 we define
D 1 (n, q) := δ ∈ (0, 1) : η(δ, n) = σ log(6c 1 /δ) n = δ ∈ (0, 1) : n ≥ Q 2 σ 2q 1 q-2 δ -2 q-2 • log(6c 1 /δ) -q q-2 , (10
)
D 2 (n, q) := (0, 1) \ D 1 (n, q) .(11)
In what follows, we will refer to D 1 (n, q) as the subgaussian confidence regime and D 2 (n, q) as the polynomial confidence regime. The effective sample size n 0 ensuring subgaussian behavior of η(n, δ) for all n ≥ n 0 is hence We illustrate the behavior of n 0 depending on 1δ for different choices of q in Figure 1 (we note that the involved constant c 1 stems from the Fuk-Nagaev inequality given in Proposition 5.1 and generally depends on q). We choose c 1 = 1 for simplicity to provide a basic intuition-note that c 1 only affects n 0 logarithmically. We refer the reader to [40] for a detailed discussion based on the bound for real-valued random variables.
n 0 := Q 2 σ 2q 1 q-2 δ -2 q-2 • log(6c 1 /δ) -q q-2 . (12
)
Subgaussian confidence regime and convergence rates. We now give an excess risk bound which is similar to the setting with bounded or subexponential noise: it exhibits a logarithmic dependence of the confidence parameter δ and a dependence of the sample size up to n -1/3 depending on the level of smoothness given by ν [4,6]. In the asymptotic large sample context, this bound is dominant and allows us to recover convergence rates.
this section cite: ['b39', 'b3', 'b5']

Section: Corollary 3.2 (Subgaussian confidence regime).
Let (MOM) and (SRC) be satisfied. Then there exist constants c1 , c2 > 0 such that with the regularization schedule
α 1 (n, δ) := c2 log(6c 1 /δ) n 1 2 min{ν,1}+1
we have
∥I π f α1(n,δ) -f ⋆ ∥ L 2 (π) ≤ c1 R log(6c 1 /δ) n min{ν,1} 2 min{ν,1}+1 ,(13)
with confidence 1δ for all δ ∈ D 1 (n, q) and n ∈ N such that α 1 (n, δ) ≤ κ 2 .
The constants c1 and c2 in Corollary 3.2 only depend on R, ν, κ, σ, c 1 , c 2 and can be made explicit, but we omit their closed form here for the sake of a more accessible presentation. We refer the reader to the proof for more details.
this section cite: []

Section: Remark 3.3 (Convergence rates).
We directly obtain convergence rates from the above consideration.
For fixed confidence parameter δ ∈ (0, 1), we see that for all n ≥ n 0 , where n 0 is the effective sample size given in (12), we have δ ∈ D 1 (n, q). Furthermore, there exists some ñ0 ∈ N such that α 1 (n, δ) ≤ κ 2 for all n ≥ ñ0 . Combining these two insights, from Corollary 3.2, we obtain
∥I π f α1(n,δ) -f ⋆ ∥ L 2 (π) ≤ c1 R log(6c 1 /δ) n min{ν,1} 2 min{ν,1}+1
with confidence 1δ for all n ≥ max{n 0 , ñ0 }. We explicitly note that the convergence rates as well as the regularization schedule α 1 (n, δ) match exactly the known results for the capacity-independent setting that have been derived under the assumption of bounded or subexponential noise [4,6].
Polynomial confidence regime. By definition (10), the polynomial confidence regime δ ∈ D 2 (n, q) is relevant in the nonasymptotic investigation whenever n < n 0 . For completeness, we address this setting in Appendix C and show that Proposition 3.1 can yield simplified risk bounds with suitable regularization schedules for α based on δ and n. Depending on δ, these bounds may require a stronger regularization α 2 (n, δ) than the subgaussian confidence setting. In fact, the resulting bound exhibits a polynomial worst-case dependence on δ, which is compensated by a better dependence on the sample size before transitioning to the behavior from the subgaussian confidence regime given by Corollary 3.2. This behavior can be observed in practice, which we confirm in a basic numerical experiment provided in Appendix A.
this section cite: ['b3', 'b5']

Section: Capacity dependent bound and optimal rates
We now improve the results from the previous section and give an excess risk bound that involves the effective dimension, which has been established as a central tool to quantify the algorithm-dependent capacity of the hypothesis space H relative to the distribution π and regularization parameter α in order to derive risk bounds in regularized kernel-based learning under the assumptions of an eigenvalue decay of C π [3,9,10,32]. Definition 4.1 (Effective dimension). For α > 0, we define N (α) := tr C π (C π + α Id H ) -1 < ∞.
We now assume the standard polynomial eigenvalue decay of C π [3,9,11]. Assumption 4.2 (Eigenvalue decay). We assume that the nonincreasingly ordered sequence of nonzero eigenvalues (µ i ) i≥1 of T π satisfies the decay
µ i ≤ Di -1/p , i ∈ N (EVD)
for a constant D > 0 and some p ∈ (0, 1).
this section cite: ['b2', 'b8', 'b9', 'b31', 'b2', 'b8', 'b10']

Section: Under the additional assumption (EVD), we can sharpen Proposition 3.1.
Proposition 4.3 (Capacity-dependent excess risk bound). Let (MOM), (SRC) and (EVD) be satisfied. Suppose that δ ∈ (0, 1) and
log(2/δ) 2κ 2 nα + 2 Dκ √ nα (1+p)/2 ≤ 1.(14)
Then there exists a constant c > 0 not depending on δ and n such that with confidence 1δ, we have
∥I π f α -f ⋆ ∥ L 2 (π) ≤ c α min{ν,1} + log(8/δ) √ αn + N (α) log(8/δ) n + N (α) • η(δ, n, α) ,
where we set η(δ, n, α) := max 1 δn q-1
1/q • 1 αN (α) q-2 2q , log(8c 1 /δ) n .
The constant c is made explicit in the proof. The key idea builds upon the original work of [3]. In particular, we incorporate the effective dimension into the Fuk-Nagaev inequality as a proxy for the q-th absolute moment appearing in the term η(δ, n, α), thereby generalizing the idea to use the effective dimension as a variance proxy in the Bernstein inequality. Corollary 4.4 (Convergence rates). Let (MOM), (SRC) and (EVD) be satisfied. Then for every δ ∈ (0, 1), there exists some n 0 ∈ N such that with the regularization schedule α(n, δ) := log(8c 1 /δ) n 1 2 min{ν,1}+p , we have
∥I π f α(n,δ) -f ⋆ ∥ L 2 (π) ≤ c log(8c 1 /δ) n min{ν,1} 2 min{ν,1}+p(15)
with confidence 1δ for all n ≥ n 0 with a constant c > 0 independent of n and δ. Remark 4.5 (Optimality of rates). The rates provided by Corollary 4.4 match the rates from the literature derived for well-specified kernel ridge regression under the assumption of subexponential noise [3,35]. In particular, these rates are known to be minimax optimal over the class of distributions satisfying (SRC), (EVD) and the Bernstein condition (3). Corollary 4.4 now proves that one can significantly relax the assumption of subexponential noise (3), as rate optimality is already achieved under the condition (MOM). Furthermore, the regularization schedule α(n, δ) is the same as in the light-tailed setting-in particular, it does not depend on q.
this section cite: ['b2', 'b2', 'b34']

Section: Fuk-Nagaev inequality in Hilbert spaces
We discuss the central ingredient for the derivation of the previous results in more detail for convenience. We refer the reader to [24,25] for the original work in the setting of real-valued random variables and [40] for a discussion of the involved constants. We present a sharpened version of a result due to [27, Theorem 3.5.1], which is formulated more generally in normed spaces, but exhibits an excess term that can be removed in the Hilbert space case. We provide the proof in Appendix D. We also note that the proof of the result in [27] is incomplete due to an inconsistent exponential moment bound. We address this issue by deriving an alternative bound. Proposition 5.1 (Fuk-Nagaev inequality; Hilbert space version). Let ξ, ξ 1 , . . . ξ n be independent and identically distributed random variables taking values in a separable Hilbert space X such that
E[ξ] = 0, E ∥ξ∥ 2
X < σ 2 and E[∥ξ∥ q X ] < Q, for some constants σ 2 > 0, Q > 0 and q ∈ N, q ≥ 3. Write S n := i=1 ξ i . Then there exist two constants c 1 > 0 and c 2 > 0 depending only on q such that for every t > 0, we have
P 1 n S n X > t ≤ c 1 Q t q n q-1 + exp -c 2 t 2 n σ 2 . (16
)
Remark 5.2. For simplicity, we may assume that 1 ≤ c 1 when we apply Proposition 5.1.
Confidence regimes. Directly rearranging ( 16) from a tail bound to a confidence interval bound requires to solve a transcendental equation which does not admit a simple closed form solution. However, we can still derive an upper bound on the confidence intervals that reflects the superposition of polynomial and sub-gaussian tail in (16). By introducing
δ := 2 max c 1 Q t q n q-1 , c 1 exp -c 2 t 2 n σ 2 ,(17)
we have P[n -1 ∥S n ∥ X ≤ t] ≥ 1δ by (16). Rearranging (17), we have
t ≥ 2c 1 Q δn q-1 1/q and t ≥ σ log(2c 1 /δ) c 2 n ,(18)
immediately leading to the following confidence bound.
this section cite: ['b23', 'b24', 'b39', 'b26', 'b15', 'b15']

Section: Corollary 5.3 (Confidence bound).
Under the assumptions of Proposition 5.1, for all δ ∈ (0, 1), we have
1 n S n X ≤ max    2c 1 Q δn q-1 1/q , σ log(2c 1 /δ) c 2 n    (19
)
with probability at least 1δ.
For every fixed δ and n → ∞, this shows the typical subgaussian behavior and a convergence rate of 1 n S n of the order n -1/2 with high probability. Interpreting the right hand side as a function of δ for a fixed sample size n however, the above bound characterizes a confidence regime change at δ(n), which we define as the solution to the equation
2c 1 Q δ(n) n q-1 1/q = σ log(2c 1 / δ(n)) c 2 n .(20)
In fact, in the polynomial confidence regime δ < δ(n), the dependence of the upper bound given in Corollary 5.3 on δ is clearly worse than in the subgaussian regime δ ≥ δ(n) which is characterized by a logarithmic dependence on δ. In contrast, the polynomial confidence regime allows for a better sample dependence of n -(q-1)/q .
this section cite: []

Section: Sharpness of the tail bound.
Both the subgaussian term and the polynomial term in the right hand side of the bound given by Proposition 5.1 can generally not be improved without additional assumptions. We repeat a similar argument as the one given in [63, Proposition 9], which is given in the context of linear processes. Let ξ, ξ 1 , . . . , ξ n be independent real-valued random variables drawn from a centered t-distribution with q degrees of freedom, i.e. E[ξ q-c ] < ∞ for all 0 < c ≤ q and let σ 2 := E[ξ 2 ] = q/(q -2). Then [25, Theorem 1.9] shows that we have
P[S n /n > t] = P[S n /σ > nt/σ] ∼ 1 -ϕ(n 1/2 t/σ) + n(1 -F σ -1 ξ (nt/σ)) as n → ∞
for nt/σ ≥ n 1/2 , where Φ is the standard normal cumulative distribution function and F σ -1 ξ is the cumulative distribution function of σ -1 ξ. We now note that we have the basic property F σ -1 ξ (nt/σ) = F ξ (nt) and we can show that the distribution of ξ satisfies 1 -F (nt) ∼ C q /(nt) q as nt → ∞, where C q is a constant depending exclusively on q. In total, we obtain
P[S n /n > t] ∼ 1 -Φ(n 1/2 t/σ) + C q t q n q-1 as n → ∞
for nt/σ ≥ n 1/2 , showing that Proposition 5.1 is asymptotically optimal.
this section cite: []

Section: References
Ref_id:b0 Title: On the mathematical foundations of learning Year: (2002)
Ref_id:b1 Title: Learning from examples as an inverse problem Year: (2005)
Ref_id:b2 Title: Optimal rates for the regularized least-squares algorithm Year: (2007)
Ref_id:b3 Title: Learning theory estimates via integral operators and their approximations Year: (2007)
Ref_id:b4 Title: Solutions of Ill Posed Problems Year: (1977)
Ref_id:b5 Title: On regularization algorithms in learning theory Year: (2007)
Ref_id:b6 Title: On Early Stopping in Gradient Descent Learning Year: (2007)
Ref_id:b7 Title: Kernel ridge vs. principal component regression: Minimax bounds and the qualification of regularization operators Year: (2017)
Ref_id:b8 Title: Optimal rates for regularization of statistical inverse learning problems Year: (2018)
Ref_id:b9 Title: Optimal rates for spectral algorithms with least-squares regression over Hilbert spaces Year: (2020)
Ref_id:b10 Title: Sobolev norm learning rates for regularized least-squares algorithms Year: (2020)
Ref_id:b11 Title: Parallelizing spectrally regularized kernel algorithms Year: (2018)
Ref_id:b12 Title: Learning theory for distribution regression Year: (2016)
Ref_id:b13 Title: Generalization properties of learning with random features Year: (2017)
Ref_id:b14 Title: Optimal rates for vector-valued spectral regularization learning algorithms Year: (2024)
Ref_id:b15 Title: Towards optimal Sobolev norm rates for the vector-valued regularized least-squares algorithm Year: (2024)
Ref_id:b16 Title: Optimal rates for regularized conditional mean embedding learning Year: (2022)
Ref_id:b17 Title: Boosted kernel ridge regression: Optimal learning rates and early stopping Year: (2019)
Ref_id:b18 Title: Distributed kernel-based gradient descent algorithms Year: (2018)
Ref_id:b19 Title: Spectral algorithms for supervised learning Year: (2008)
Ref_id:b20 Title: Learning linear operators: Infinitedimensional regression as a well-behaved non-compact inverse problem Year: (2022)
Ref_id:b21 Title: Kernel instrumental variable regression Year: (2019)
Ref_id:b22 Title: Nonparametric instrumental regression via kernel methods is minimax optimal Year: (2024)
Ref_id:b23 Title: Some probabilistic inequalities for martingales Year: (1973)
Ref_id:b24 Title: Large Deviations of Sums of Independent Random Variables Year: (1979)
Ref_id:b25 Title: High-Dimensional Probability: An Introduction with Applications in Data Science Year: (2018)
Ref_id:b26 Title: Sums and Gaussian vectors Year: (1995)
Ref_id:b27 Title: The fundamentals of heavy tails. Properties, emergence, and estimation Year: (2022)
Ref_id:b28 Title: Modelling extremal events for insurance and finance Year: (1997)
Ref_id:b29 Title: Concentration inequalities. A nonasymptotic theory of independence Year: (2016)
Ref_id:b30 Title: Remarks on inequalities for large deviation probabilities Year: (1986)
Ref_id:b31 Title: Learning bounds for kernel regression using effective data dimensionality Year: (2005)
Ref_id:b32 Title: A Distribution-Free Theory of Nonparametric Regression Year: (2002)
Ref_id:b33 Title: Support Vector Machines Year: (2008)
Ref_id:b34 Title: Optimal rates for regularized least squares regression Year: (2009)
Ref_id:b35 Title: Robust statistics Year: (1981)
Ref_id:b36 Title: Optimal rates for the regularized learning algorithms under general source condition Year: (2017)
Ref_id:b37 Title: On the optimality of misspecified kernel ridge regression Year: (2023)
Ref_id:b38 Title: Probability inequalities for sums of independent random variables Year: (1971)
Ref_id:b39 Title: About the constants in the Fuk-Nagaev inequalities Year: (2017)
Ref_id:b40 Title: On the Robustness of Kernel Ridge Regression Using the Cauchy Loss Function Year: (2025)
Ref_id:b41 Title: Regularization of Inverse Problems Year: (1996)
Ref_id:b42 Title: Convergence rates of least squares regression estimators with heavy-tailed errors Year: (2019)
Ref_id:b43 Title: On least squares estimation under heteroscedastic and heavy-tailed errors Year: (2022)
Ref_id:b44 Title: Upper bounds on product and multiplier empirical processes Year: (2016)
Ref_id:b45 Title: Learning Bounded Subsets of L p Year: (2021)
Ref_id:b46 Title: Robust linear least squares regression Year: (2011)
Ref_id:b47 Title: Loss minimization and parameter estimation with heavy tails Year: (2016)
Ref_id:b48 Title: Empirical risk minimization for heavytailed losses Year: (2015)
Ref_id:b49 Title: Mean estimation and regression under heavy-tailed distributions: A survey Year: (2019)
Ref_id:b50 Title: Challenging the empirical mean and empirical variance: A deviation study Year: (2012)
Ref_id:b51 Title: Sub-Gaussian estimators of the mean of a random vector Year: (2019)
Ref_id:b52 Title: Robust covariance estimation under L 4 -L 2 norm equivalence Year: (2020)
Ref_id:b53 Title: Distribution-free robust linear regression Year: (2021)
Ref_id:b54 Title: Beyond sub-gaussian noises: Sharp concentration analysis for stochastic gradient descent Year: (2022)
Ref_id:b55 Title: Linear Operators in Hilbert Spaces Year: (1980)
Ref_id:b56 Title: Methods of Mathematical Physics I: Functional Analysis Year: (1980)
Ref_id:b57 Title: Reproducing Kernel Hilbert Spaces in Probability and Statistics Year: (2004)
Ref_id:b58 Title: Regularization of Inverse Problems Year: (1996)
Ref_id:b59 Title: On the optimality of misspecified spectral algorithms Year: (2024)
Ref_id:b60 Title: On converse and saturation results for Tikhonov regularization of linear ill-posed problems Year: (1997)
Ref_id:b61 Title: On the saturation effect of kernel ridge regression Year: (2023)
Ref_id:b62 Title: Beyond sub-Gaussian noises: Sharp concentration analysis for stochastic gradient descent Year: (2022)
Ref_id:b63 Title: Discretization error analysis for Tikhonov regularization Year: (2006)
Ref_id:b64 Title: Beating SGD saturation with tail-averaging and minibatching Year: (2019)
Ref_id:b65 Title: Optimum bounds for the distributions of martingales in Banach spaces Year: (1994)
Ref_id:b66 Title: On the Statistical Approximation of Conditional Expectation Operators Year: (2022)
Ref_id:b67 Title:  Year: (2025-09)
Ref_id:b68 Title: Learning theory of distributed spectral algorithms Year: (2017)
Ref_id:b69 Title: Norm inequalities equivalent to Heinz-Löwner theorem Year: (1989)
