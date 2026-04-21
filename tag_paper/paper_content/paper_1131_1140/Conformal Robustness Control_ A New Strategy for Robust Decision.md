Title: CONFORMAL ROBUSTNESS CONTROL: A NEW STRAT-EGY FOR ROBUST DECISION
Abstract: Robust decision-making is crucial in numerous risk-sensitive applications where outcomes are uncertain and the cost of failure is high. Conditional Robust Optimization (CRO) offers a framework for such tasks by constructing prediction sets for the outcome that satisfy predefined coverage requirements and then making decisions based on these sets. Many existing approaches leverage conformal prediction to build prediction sets with guaranteed coverage for CRO. However, since coverage is a sufficient but not necessary condition for robustness, enforcing such constraints often leads to overly conservative decisions. To overcome this limitation, we propose a novel framework named Conformal Robustness Control (CRC), that directly optimizes the prediction set construction under explicit robustness constraints, thereby enabling more efficient decisions without compromising robustness. We develop efficient algorithms to solve the CRC optimization problem, and also provide theoretical guarantees on both robustness and optimality. Empirical results show that CRC consistently yields more effective decisions than existing baselines while still meeting the target robustness level.

Section: INTRODUCTION
In many real-world applications, it is crucial for decision-makers to account for operational risks to avoid irreversible consequences. For example, portfolio management (Markowitz, 1952) aims to maximize returns while navigating the trade-off with risk tolerance. Similar risk-sensitive decisionmaking challenges are also evident in fields such as medical diagnosis (Kiyani et al., 2025) and transportation planning (Patel et al., 2024).
Consider a scenario where we observe an input X, but the corresponding outcome Y is unknown. The decision-maker needs to choose a decision z(X) based on the input X such that the incurred decision loss ϕ(Y, z(X)) does not exceed a certain risk certificate r(X) with high probability. Formally, the (1 -α)-level robustness requirement is given by
P{ϕ(Y, z(X)) ≤ r(X)} ≥ 1 -α.(1)
At the same time, the decision-maker seeks to minimize r(X) to improve efficiency and reduce potential worst-case losses.
Over the years, Conditional Robust Optimization (CRO), introduced by Chenreddy et al. (2022), has become a widely adopted and effective framework for robust decision-making. As an extension of classical robust optimization (Ben-Tal et al., 2009), CRO incorporates covariate information to enhance decision quality, enabling more precise and context-aware responses in complex tasks. In the CRO framework, decisions are derived from a minmax optimization problem using a prediction set U(X), formulated as z U (X) := arg min z∈Z max y∈U (X) ϕ(y, z). By designing U(X) with a regular structure, such as a box or an ellipse, the resulting minmax problem remains convex and can be solved efficiently in polynomial time. The corresponding risk certificate value is defined is below the risk certificate r(X). The prediction set in CRO achieves exact 90% coverage, with r(X) = 1.93. In contrast, CRC meets the 90% robustness requirement, yielding a more efficient decision with r(X) = 1.25.
such as Shang et al. (2017); Bertsimas et al. (2018); Hong et al. (2021), have proposed data-driven prediction sets. With the growing size of data, Chenreddy et al. (2022) explored how covariate information could be leveraged to develop more effective prediction sets, leading to the introduction of the Conformal Robust Optimization (CRO) framework. Subsequent works by Johnstone & Cox (2021); Patel et al. (2024); Sun et al. (2023) incorporated conformal prediction methods to construct prediction sets that satisfy coverage conditions, thereby providing finite-sample robustness guarantees for CRO. Kiyani et al. (2025) derived the explicit form of the optimal prediction set that has the minimum risk certificate under the coverage constraint. However, the construction relies on minimizing the VaR function, which often also leads to intractable formulations if the decision space is continuous (Uryasev & Rockafellar, 2001). In addition, Wang et al. (2023) also considered optimizing the prediction sets in a robust optimization problem, but relaxing the robustness constraint through the conditional Value at Risk transformation (Rockafellar & Uryasev, 2002). Compared to existing work, we impose the exact robustness constraint rather than a coverage constraint on the prediction set, thereby enhancing the generation of more effective decisions.
Conformal prediction is a widely used method for uncertainty quantification, notable for its modelagnostic and distribution-free properties (Vovk et al., 2005;Lei et al., 2018;Angelopoulos et al., 2024a). In predictive inference tasks, the efficiency measure of conformal prediction sets is the size or volume. Recent research has increasingly focused on improving the efficiency of these prediction sets. Several studies, such as Sadinle et al. (2019), Bai et al. (2022), Stutz et al. (2022), and Kiyani et al. (2024b) have formulated constrained optimization problems that minimize the size of prediction sets subject to coverage constraints. In addition, Yang & Kuchibhotla (2025) introduced a sample-splitting approach to select models yielding the smallest prediction sets, followed by constructing split conformal prediction sets (Vovk et al., 2005;Papadopoulos et al., 2002). Differently, Liang et al. (2024) proposed a method that avoids sample splitting while maintaining finite-sample coverage during model selection. In terms of decision efficiency, since the performance of decisions varies significantly with different conformal prediction sets, Chenreddy & Delage (2024) and Yeh et al. (2024) proposed end-to-end learning methods that train the conformal prediction sets by directly minimizing downstream expected decision risk. Moreover, Bao et al. (2025) developed new frameworks for prediction set selection in the CRO problem, which could keep finite-sample robustness control while avoiding sample splitting.
this section cite: ['b27', 'b23', 'b30', 'b12', 'b7', 'b33', 'b8', 'b17', 'b12', 'b19', 'b30', 'b35', 'b23', 'b36', 'b39', 'b31', 'b38', 'b25', 'b32', 'b4', 'b34', 'b41', 'b38', 'b29', 'b26', 'b42', 'b5']

Section: PREDICTION SET OPTIMIZATION WITH ROBUSTNESS CONTROL
3.1 PROBLEM SETUP Let X be the covariate space, and Y be the label space. The primary goal of robust decision is to find a decision policy z(•) : X → Z and a risk certificate function r(•) : X → R that minimizes E[r(X)] subject to the robustness constraint in (1). It is consistent with the Risk Averse Decision Policy Optimization (RA-DPO) problem defined by Kiyani et al. (2025):
min z(•),r(•) E[r(X)] s.t. P{ϕ (Y, z(X)) ≤ r(X)} ≥ 1 -α.(3)
However, directly optimizing over arbitrary forms of z(•) and r(•) is generally difficult. The CRO framework provides a flexible alternative by introducing a prediction set U(•) that maps each covariate x ∈ X to a subset of the label space Y, which relates to both the decision and the associated risk certificate. Specifically, for x ∈ X , z U (x) := arg min z∈Z max y∈U (x) ϕ(y, z), r U (x) = max y∈U (x) ϕ(y, z U (x)).
To identify the optimal decision policy and risk certificate, it is natural to minimize the expected risk certificate under the robustness constraint:
min U (•):X →2 Y E [r U (X)] s.t. P {ϕ (Y, z U (X)) ≤ r U (X)} ≥ 1 -α.(4)
The next theorem shows the equivalence between RA-DPO in (3) and the problem (4), which also means that optimizing the prediction sets will not result in a suboptimal risk certificate. Theorem 3.1. Let z RA-DPO (•), r RA-DPO (•) be the optimal solution of RA-DPO in (3), and let U * be the optimal solution of (4). It holds that E[r RA-DPO (X)] = E[r U * (X)], which means problems (3) and (4) are equivalent in minimizing the expected risk certificate while maintaining robustness control.
We defer the proof of Theorem 3.1 to Appendix B.2. The prior work of Kiyani et al. (2025) also derived a formulation equivalent to RA-DPO in Eq. ( 3), termed Risk Averse Conformal Prediction Optimization (RA-CPO). This formulation optimizes the expected risk certificate over all possible prediction sets subject to a coverage constraint, that is, replacing the constraint in (4) with (2). To construct the "optimal" prediction set that solves RA-CPO, Kiyani et al. (2025) proposed a method based on the minimizer and minimum value of a contextual VaR problem:
z * (x) = arg min z∈Z VaR 1-α (ϕ(Y, z)|X = x) and r * (x) = min z∈Z VaR 1-α (ϕ(Y, z)|X = x).
Here, VaR(•|X = x) denotes the conditional 1 -α population quantile given the covariate X = x.
In the case of classification with a finite decision space, Kiyani et al. (2024a) approximated the optimal prediction set by first estimating the conditional distribution of Y | X, then the associated VaR problem can be solved by traversal. However, this approach does not extend to continuous decision spaces Z, where the VaR problem generally becomes intractable (Uryasev & Rockafellar, 2001).
To address this limit, we consider solving problem (4) over the parametrized prediction set U θ (•), where θ ∈ Θ refers to the model parameters. In regression settings with X = R p , Y = R q , two commonly used types of prediction sets are box and ellipsoidal sets (Johansson et al., 2017;Sun et al., 2023). Their parametrized forms are can be defined as follows.
• Box prediction set. A box-shaped prediction set is constructed by componentwise lower and upper bounds for the response vector. Let h lo θ (•) : R p → R q and h hi θ (•) : R p → R q be models with parameters θ ∈ Θ, then
U θ (x) = y ∈ R q : h lo θ (x) ≤ y ≤ h hi θ (x) .
• Ellipsoidal prediction set. Unlike box sets, ellipsoidal prediction sets account for correlations among components of the response vector. Let µ θ (•) : R p → R q and Σ θ (•) : R p → R q×q denote the mean and covariance model with parameters θ ∈ Θ, then
U θ (x) = y ∈ R q : (y -µ θ (x)) ⊤ Σ -1 θ (x) (y -µ θ (x)) ≤ 1 .
In Appendix E.6, we also provide the example of a parametrized polyhedral set based on the definition in Bärmann et al. (2016).
For a parametrized prediction set U θ (•), we denote the corresponding decision policy and risk certificate functions as z θ (•) ≡ z U θ (•) and r θ (•) ≡ r U θ (•) for short. We then consider the parameterized version of problem (4):
min θ∈Θ E[r θ (X)] s.t. P {ϕ(Y, z θ (X)) ≤ r θ (X)} ≥ 1 -α.(5)
Even though the parametrized optimization can also be applied to the RA-CPO framework, we show that our proposed problem (5) yields a lower risk certificate in Appendix B.3, which further confirms the benefit of robustness constraint over the coverage constraint. In the following subsections, we investigate the optimization problem (5) based on the collected labeled data and provide theoretical results for the robustness and optimality guarantees. Moreover, we also developed a differential algorithm to solve the optimization problem for the continuous decision space.
this section cite: ['b23', 'b23', 'b23', 'b36', 'b18', 'b35', 'b6']

Section: EMPIRICAL OPTIMIZATION WITH CONFORMAL ROBUSTNESS CONTROL
Suppose we have collected an i.i.d. labeled dataset D n = {(X i , Y i )} n i=1 drawn from some distribution P . We first optimize the prediction set by addressing an empirical version of the problem (5), and then apply this prediction set for decision-making. By approximating both the objective and the constraint in (5) via sample averaging, we obtain the following empirical counterpart:
θ = arg min θ∈Θ 1 n n i=1 r θ (X i ) s.t. 1 n n i=1 1 {ϕ (Y i , z θ (X i )) ≤ r θ (X i )} ≥ 1 -α.(6)
To distinguish from coverage control methods and to emphasize the explicit robustness constraint, we refer to this procedure as Conformal Robustness Control (CRC).
A natural approach to solving problem (6) is to consider its dual formulation. Define the Lagrangian function as L(λ; θ) := f (θ) + λg(θ), where
λ ≥ 0 is the Lagrange multiplier, f (θ) = 1 n n i=1 r θ (X i ) and g(θ) = 1 -α -1 n n i=1 1 {ϕ (Y i , z θ (X i )) ≤ r θ (X i )}. The function f (θ)
is typically differentiable if the CRO problem for U θ (x) can be reformulated into a convex programming. In such cases, its gradient can be computed using existing implicit differential tools, see Amos & Kolter (2017) and Agrawal et al. (2019). In contrast, the term g(θ) is non-smooth due to the indicator. To enable gradient-based optimization, we approximate the indicator with a smooth surrogate 1{a ≤ b}
= 1 2 (1 + erf( b-a √2σ
)), where erf(x) = 2 √ π
x 0 e -t 2 dt is the Gaussian error function and σ > 0 controls the smoothness. Replacing the indicator in g(θ) with this surrogate yields a smoothed constraint function g(θ). Similar smoothing techniques have been employed in the optimization problem of conformal prediction (Bai et al., 2022;Kiyani et al., 2024b;Wu et al., 2025). The resulting smoothed dual problem is given by: min θ∈Θ max λ≥0 L(λ; θ), where L(λ; θ) = f (θ) + λg(θ). This smooth approximation enables numerical solution via an alternating gradient descent algorithm. We refer to Davis et al. (2020) and Bolte et al. (2021) for the convergence analysis of similar optimization problems. Implementation details are summarized in Algorithm 1.
Algorithm 1 Prediction Set Optimization with CRC
1: Input: Loss function ϕ, robustness level 1 -α, labeled dataset D n = {(X i , Y i )} n i=1 , parametrized set U θ (•) with θ ∈ Θ, smooth surrogate function 1, learning rate η > 0. 2: Initialize θ ← θ 0 and λ = 0. 3: Compute r θ (X i ) and z θ (X i ) for i ∈ [n]
this section cite: ['b0', 'b4', 'b40', 'b13', 'b9']

Section: THEORETICAL RESULTS
This section presents the theoretical guarantees for the solution to problem (6). The analysis for the smoothed variant (Algorithm 1), being conceptually analogous, are deferred to Appendix D.2. We equip the parameter space Θ with the supremum norm and state the underlying assumptions. Condition 3.1. Loss function ϕ is L ϕ -Lipschitz in decision z for any y ∈ Y. For any x ∈ X , the decision z θ (x) is L z -Lipschitz in θ. The risk certificate r θ (x) is L r -Lipschitz in θ ∈ Θ, and uniformly bounded by a positive constant B r > 0 for any x ∈ X and θ ∈ Θ.
These regularity conditions are mild and typically satisfied in practice. For example, in portfolio optimization with the loss function ϕ(y, z) = -y ⊤ z, the Lipschitz condition holds if Y is bounded. For decision function z θ and risk certificate function r θ , if the CRO problem for prediction set U θ can be transformed into a smooth convex optimization problem, then the Lipschitz property can be derived from the KKT conditions and the implicit function theorem, see Bolte et al. (2021) and Amos & Kolter (2017). The next assumption introduces a mild distributional assumption.
Condition 3.2. Let V θ (X, Y ) = ϕ (Y, z θ (X)) -r θ (X) for data (X, Y ) ∼ P . Suppose that for all θ ∈ Θ the density of V θ (X, Y ) is uniformly bounded by a constant ρ 0 > 0.
The bounded density condition is often needed for concentration guarantees in the conformal prediction literature (Kiyani et al., 2024b;Jung et al., 2023;Lei & Wasserman, 2014). Definition 3.1 (Covering number). Let Θ be a parameter space with the supremum norm ∥ • ∥ ∞ . Given any ϵ > 0, the subset Θ ϵ ⊆ Θ is called an ϵ-covering of Θ if for every θ ∈ Θ, there exists some θ ϵ ∈ Θ ϵ such that ∥θ -θ ϵ ∥ ∞ < ϵ. The covering number N (Θ, ∥ • ∥ ∞ , ϵ) is the smallest cardinality of any ϵ-covering of Θ.
Covering numbers quantify the complexity of a function class and are a fundamental tool in statistical learning theory and convergence analysis (Van Der Vaart & Wellner, 1996). The next two theorems provide a non-asymptotic characterization of the robustness and expected risk certificate value for CRC. Theorem 3.2 (Robustness gap). Let θ be the solution to optimization problem (6). Under Conditions 3.1 and 3.2, for any independent data (X, Y ) ∼ P , conditioning on the labeled data D n , the following inequality holds: with probability at least 1 -n -1 ,
P ϕ Y, z θ (X) ≤ r θ (X) | D n ≥1 -α -∆ n ,
where the robustness gap
∆ n = 5 log(2N (Θ,∥•∥∞,n -1 ))+log n 2n + 4(L ϕ Lz+Lr)ρ0 n .
Theorem 3.3 (Risk certificate optimality). Let θ * ∆n denote the optimal solution of problem (5) at the robustness level 1 -α + ∆ n . Under the same conditions as Theorem 3.2, conditioning on D n , with probability at least
1 -2n -1 , E r θ (X) -r θ * ∆n (X) | D n ≤4B r log(2N (Θ, ∥ • ∥ ∞ , n -1 )) + log n 2n + 4L r n .
For a finite-dimensional parameter space Θ of dimension d, the covering number scales approximately as N (Θ, ∥ • ∥ ∞ , n -1 ) ≍ n d , so both the robustness gap and the expected risk certificate converge to zero at rate O( d log n/n). In Appendix D, we provide more comprehensive theoretical results, such as in the setting where the function class has a finite VC dimension. Remark 3.2. It is worth noting that θ * ∆n denotes the optimal model under a slightly relaxed robustness level 1 -α + ∆ n , rather than the exact level 1 -α. This relaxation is introduced to ensure that θ * ∆n is feasible to the problem (6) with high probability, thereby guaranteeing that the empirical risk certificate of θ is less than that of θ * ∆n with high probability. Finally, leveraging relevant theories of empirical process, we can establish the bounds in Theorems 3.2 and 3.3. Let θ * be the solution to the problem (4). If additional assumptions are imposed regarding the ∥θ * ∆n -θ * ∥ ∞ , such a relaxation may no longer be needed.
this section cite: ['b9', 'b20', 'b24', 'b37']

Section: TEST-TIME DECISION WITH FINITE-SAMPLE ROBUSTNESS CONTROL
In this section, we turn to the practical task of making decisions at a test point X n+1 with unknown label Y n+1 . A straightforward approach is to output the decision z U θ (X n+1 ), where θ is solution to the problem (6). Theorem 3.2 shows that the robustness of the decision z U θ (X n+1 ) converges to the target level asymptotically. To achieve finite-sample robustness control for the decision of the specific test point, we further calibrate the prediction set obtained from Algorithm 1 using both the test data X n+1 and the labeled data {(X i , Y i )} n i=1 . Specifically, we split the labeled dataset D n into a training set
D train = {(X i , Y i )} n0
i=1 and a calibra-
tion set D cal = {(X i , Y i )} n
i=n0+1 , where n 0 < n. We first obtain the optimized prediction set U θ0 (•) using only D train in Algorithm 1. Next, we apply full conformal prediction (Vovk et al., 2005;Lei et al., 2018) to calibrate the prediction set U θ0 (•) based on D cal and X n+1 .
Calibrating the entire parameters θ is computationally expensive and often unnecessary. Instead, we can adjust the prediction set U θ0 (•) by tuning a single radius parameter t ∈ R + , which controls the size of the set and provides an efficient way of model calibration. Following the framework of nested prediction set in Gupta et al. (2022), we call the family {U θ,t (x)} t∈R + nested sets if t 1 ≤ t 2 implies that U θ,t1 (x) ⊆ U θ,t2 (x) for any x ∈ X . For the two examples of prediction sets in Section 3.1, the nested versions are given as follows.
• Nested parametrized box set:
U θ,t (x) = y ∈ R q : h lo θ (x) -t ≤ y ≤ h hi θ (x) + t ;
• Nested parametrized ellipsoidal set:
U θ,t (x) = {y ∈ R q : (y -µ θ (x)) ⊤ Σ -1 θ (x) (y -µ θ (x)) ≤ t}.
Let y ∈ Y be a hypothesized value for the test label Y n+1 , and denote the augmented calibration set as {(X i , Y y i )} n+1 i=n0+1 , where
Y y i = Y i for n 0 + 1 ≤ i ≤ n and Y y n+1 = y.
Given the prediction set U θ0,t (•), the hypothesized radius threshold is computed by
ty = min t ∈ R + : 1 n -n 0 + 1 n+1 i=n0+1 1 ϕ Y y i , z θ0,t (X i ) ≤ r θ0,t (X i ) ≥ 1 -α , (7)
where z θ,t (x) := arg min z∈Z max c∈U θ,t (x) ϕ(c, z) and r θ,t (x) := max c∈U θ,t (x) ϕ(c, z θ,t (x)). Then the calibrated prediction set is given by
U Cal (X n+1 ) = y ∈ Y : ϕ y, z θ0, ty (X n+1 ) ≤ r θ0, ty (X n+1 ) .
Finally, the decision for test point is made by z U Cal (X n+1 ). We name the procedure above as Calibrated CRC (Cal-CRC), and summarize it in Algorithm 2.
Algorithm 2 Cal-CRC 1: Input: Same as Algorithm 1, size of training set n 0 , and test point X n+1 . 2: Sample splitting:
D train = {(X i , Y i )} n0 i=1 and D cal = {(X i , Y i )} n i=n0+1 . 3: Training: Obtain the prediction set U θ0 (•) by running Algorithm 1 on D train . 4: Calibration: U Cal (X n+1 ) ← ∅. 5: for y ∈ Y do 6: Define {(X i , Y y i )} n+1 i=n0+1
, where
Y y i = Y i for n 0 + 1 ≤ i ≤ n and Y y n+1 = y. 7:
Calculate the hypothesized threshold ty via (7).
8:
if ϕ y, z θ0, ty (X n+1 ) ≤ r θ0, ty (X n+1 ) then 9: U Cal (X n+1 ) ← U Cal (X n+1 ) ∪ {y}.
10:
end if 11: end for 12: Make the decision:
z U Cal (X n+1 ) ← arg min z∈Z max y∈U Cal (Xn+1) ϕ(y, z). 13: Output: the decision z U Cal (X n+1 ). Theorem 4.1. If the labeled data {(X i , Y i )} n
i=1 and test data (X n+1 , Y n+1 ) are i.i.d., then we have the finite-sample robustness guarantee
P {ϕ (Y n+1 , z U Cal (X n+1 )) ≤ r U Cal (X n+1 )} ≥ 1 -α.
The finite-sample robustness relies solely on the exchangeability of data, which is identical to that in classical conformal prediction theory (Lei et al., 2018). For implementation, note that the calibrated prediction set U Cal (X n+1 ) is obtained by traversing all possible values of y ∈ Y. In practice, we can apply the discretization technique (Chen et al., 2018) to avoid exhaustive search. The complete implementation is provided in the Appendix B.4. The decision optimality of z U Cal is analyzed in the Appendix B.5, and corresponding simulation results will be provided in Section E.3.
this section cite: ['b38', 'b25', 'b16', 'b25', 'b10']

Section: EXPERIMENTS
In this section, we compare our proposed CRC with two baseline methods for robust decisionmaking: (i) CRO with conformal prediction sets (Sun et al., 2023); (ii) End-to-end (E2E) method (Chenreddy & Delage, 2024;Yeh et al., 2024) to minimize the expected risk certificate. For clarity, we refer to the application of CRC to ellipsoidal prediction sets as CRC-E, and to box prediction sets as CRC-B. The same naming convention is applied to the CRO and E2E methods for consistency. The implementation details of each baseline method are given in Appendix E.2.
We utilize the following metrics to evaluate the performance of three methods. In this simulation, we define the loss function as ϕ(y, z) = -y ⊤ z, where Y = R 2 and Z = {z ∈ [0, 1] 2 : ∥z∥ 1 = 1}. The labeled data and test data are generated by:
Y 1 = -5X 1 -2X 2 2 -e 1 , Y 2 = -3X 2 1 -X 2 -e 2 ,
where
Y = (Y 1 , Y 2 ), X = (X 1 , X 2 ),
and e = (e 1 , e 2 ). The covariate X ∼ N ((1, 1) ⊤ , 2.25 • I 2 ), where I 2 is a 2-dimensional identity matrix. The noise e ∼ N (0, I 2 ) is independent of X. We only consider ellipsoidal prediction sets since the oracle prediction set of Y | X is ellipsoidal under the normal noise setting. Further experimental details will be presented in Appendix E.1. All methods are evaluated over 100 trials, and the average results are reported. 0.05 0.1 0.15 0.2 0.25 Nominal level 7 8 9 10 11 12 13 Loss Risk Certificate 0.05 0.1 0.15 0.2 0.25 Nominal level 7.00 7.05 7.10 7.15 7.20 7.25 7.30 7.35 Loss Decision Loss 0.05 0.1 0.15 0.2 0.25 Nominal level 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 Rate Robustness 0.05 0.1 0.15 0.2 0.25 Nominal level 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Rate Coverage CRC-E E2E-E CRO-E CRC-E E2E-E CRO-E Figure 2: The results of risk certificate, decision loss, robustness, and coverage on synthetic data when varying nominal level α with identical sample size n = 1500. The horizontal gray dashed lines refer to robustness levels. The prediction sets are ellipsoids.
Results. We evaluate the decision performance of CRC and the baseline methods by varying the nominal level α. As shown in Figure 2, CRC consistently outperforms the baselines in both risk certificate and decision loss. In addition, CRC also maintains the robustness level to the nominal target, while the baseline methods tend to be more conservative. With respect to coverage, CRC attains a much lower coverage rate than the robustness level, which verifies the motivation of our method. Additionally, the results for varying sample sizes are shown in Figure 3. CRC-E continues to show strong performance across all metrics, demonstrating its stable advantage. In Figure 6 of Appendix E.3, we present the density plots for risk certificate and decision loss when α = 0.15. The overall density of CRC is shifted towards the lower loss region, further validating its superiority. Since the RAC method proposed Kiyani et al. (2025) is applicable to discrete decision space in classification problem, we conduct the simulation on RAC method by discretizing the label space Y and decision space Z. The results are provided in Appendix E.4. In addition, we also report the simulation results under the polyhedral prediction set in Appendix E.6.
this section cite: ['b35', 'b11', 'b42', 'b23']

Section: US STOCK PROBLEM
We conduct an additional experiment on the portfolio optimization problem using a real-world dataset, following the experimental design outlined in Chenreddy et al. (2022). The dataset comprises historical US stock market data from January 1, 2012, to December 31, 2020, covering 64 stocks across eight different sectors. Daily percentage gains or losses are computed from the adjusted closing prices of consecutive trading days and used as labels. To enhance the input information for the model, we also incorporate the trading volume of individual stocks and several market benchmark indices as covariates. To evaluate the robustness of the methodology, we randomly select 15 stocks from the pool of 64 as the investable asset set in each experiment and repeat the process multiple times to mitigate the influence of random chance. We define the loss function as ϕ(y, z) = -y ⊤ z, where Y = R q and Z = {z ∈ [0, 1] q : ∥z∥ 1 = 1, z ≥ 0}. Results. As shown in Table 1, CRC outperforms the baseline methods in both risk certification and decision loss. In terms of robustness, CRC maintains a level close to the target 1 -α, demonstrating strong stability and adaptability. In contrast, E2E and CRO frequently exceed the nominal robustness target, which indicates the adoption of overly conservative strategies that lead to higher losses and risks. Overall, CRC achieves a superior balance between risk control and decision performance.
this section cite: ['b12']

Section: BATTERY STORAGE PROBLEM
In this subsection, we consider a battery storage control problem based on the frameworks of Donti et al. (2017) and Yeh et al. (2024). Given hourly electricity price forecasts y ∈ R T and contextual covariates over a T -hour horizon, the controller determines the charging power z in ∈ R T , discharging power z out ∈ R T , and the resulting state of charge z state ∈ R T , subject to the constraints for t = 1, . . . , T :
z state 0 = B 2 , z state t = z state t-1 -z out t + γz in t , 0 ≤ z in t ≤ c in , 0 ≤ z out t ≤ c out , 0 ≤ z state t ≤ B.
Here B denotes battery capacity, γ denotes charging efficiency, and c in , c out denotes per-hour power limits. The objective balances three key factors: (1) Profit from arbitrage, which involves buying and selling energy based on the prices y ∈ R T ; (2) Flexibility, which is encouraged by maintaining the battery's state of charge close to half of its total capacity; (3) Battery health, which is preserved by penalizing large charging and discharging magnitudes. The resulting loss function is:
ϕ(y, z) = T t=1 y t z in t -z out t + β z state -B 2 1 2 2 + ε ∥z in ∥ 2 2 + ∥z out ∥ 2 2 ,
Following Donti et al. (2017) and Yeh et al. 2024, we also set T = 24 hours, B = 1, γ = 0.9, c in = 0.5, c out = 0.2, β = 0.1, and ε = 0.05.
Results. Figure 4 presents a comparative analysis of the CRC method against the baselines using ellipsoidal prediction sets. For clearer visualization, negative indicator values are mapped onto the positive half-axis via a sigmoid transformation. The results demonstrate CRC's consistent superiority over both E2E and CRO across all three key metrics. As the nominal level increases, CRC effectively mitigates risk in all measures, sustaining the lowest risk and loss values at higher levels.
In addition, CRC maintains robustness values close to the nominal target, highlighting its stability and adaptability. In contrast, E2E and CRO consistently exceed the robustness target, leading to decisions characterized by excessive conservatism. The results of the box prediction set is presented in Figure 5.
0.05 0.1 0.15 0.2 0.25 Nominal level 0.0 0.2 0.4 0.6 0.8 1.0 Sigmoid(Loss) Risk Certificate 0.05 0.1 0.15 0.2 0.25 Nominal level 0.0 0.1 0.2 0.3 0.4 Sigmoid(Loss) Decision Loss 0.05 0.1 0.15 0.2 0.25 Nominal level 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 Rate Robustness CRC-E E2E-E CRO-E Figure 4: The results of risk certificate, decision loss, and robustness when varying nominal level α on battery storage problem. The prediction sets are ellipsoids. 0.05 0.1 0.15 0.2 0.25 Nominal level 0.15 0.20 0.25 0.30 0.35 0.40 0.45 Sigmoid(Loss) Risk Certificate 0.05 0.1 0.15 0.2 0.25 Nominal level 0.12 0.14 0.16 0.18 0.20 0.22 0.24 Sigmoid(Loss) Decision Loss 0.05 0.1 0.15 0.2 0.25 Nominal level 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 Rate Robustness CRC-B E2E-B CRO-B A USAGE STATEMENT OF LARGE LANGUAGE MODEL We used a large language model solely for improving the fluency and readability of the manuscript. The model was not involved in research ideation, experimental design, data analysis, or result interpretation. All scientific contributions and substantive content were solely produced by the authors. B MORE DISCUSSION ON RELATED WORK B.1 RELATIONSHIP BETWEEN THE CRC PROBLEM AND THE VAR PROBLEM
The following proposition illustrates the relationship between the VaR problem and the CRC problem (4).
Proposition B.1. Let z Q (X) = arg min z∈Z VaR 1-α (ϕ(Y, z) | X)
be the unique minimizer of VaR problem. There exists a prediction set U Q such that z U Q = z Q . Moreover, the robustness constraint is satisfied:
P [ϕ (Y, z U Q (X)) ≤ r U Q (X)] ≥ 1 -α.
Furthermore, there exist cases where z Q = z U * , with U * being the solution to optimization problem (4), and cases where z Q ̸ = z U * .
The above conclusion indicates that, at least in some cases, decision z U * and decision z Q are consistent. When decision z U * and decision z Q are inconsistent, the expected risk certificate generated by decision z U * will also be lower than that of z Q , indicating that z U * still holds practical significance.
Proof of Proposition B.1. To find a prediction set U Q such that z Q equals z U Q , it is sufficient to define prediction set U Q in the following form:
U Q (x) = y ∈ Y : ϕ y, z Q (x) ≤ VaR 1-α ϕ Y, z Q (X) | X = x , ∀x ∈ X .
Next, we will proceed with the verification. Since the coverage constraint is a sufficient condition for the robustness constraint, we have
P {ϕ (Y, z U Q (X)) ≤ r U Q (X)} ≥ P Y ∈ U Q (X) = P ϕ Y, z Q (X) ≤ VaR 1-α ϕ Y, z Q (X) | X ≥ 1 -α.
Thus, we verify that robustness holds. Secondly, based on the definition of quantiles, we have P{Y ∈ U Q (x)} ≥ 1 -α, ∀x ∈ X . Therefore, we can control the upper bound of
VaR 1-α (ϕ (Y, z U Q (X)) | X = x)
in the following way:
VaR 1-α (ϕ (Y, z U Q (X)) | X = x) ≤ max y∈U Q (x) ϕ (y, z U Q (x)) ≤ max y∈U Q (x) ϕ y, z Q (x) ≤ VaR 1-α ϕ Y, z Q (X) | X = x .
That is, z U Q is also the optimal solution in the sense of minimizing the 1 -α quantile. Therefore, z U Q equals z Q if z Q is the unique optimal solution.
For the scenario where decision z Q is equal to decision z U * , consider the following example. Let X = {0, 1} and Z = {0, 1}. Suppose that the density of ϕ(Y, z) given X = x is as follows.
For x ∈ {0, 1}, z = 0, the density is
f (ϕ) = 1 25 [31{0 ≤ ϕ < 7.5} + 1{7.5 ≤ ϕ < 10}] .
For x ∈ {0, 1}, z = 1, the density is
f (ϕ) = 1 25 [I{0 ≤ ϕ < 2.5} + 3I{2.5 ≤ ϕ < 10}] .
It can be verified that when α = 0.1, the optimal solutions of VaR and problem (4) are the same:
z U * = z Q = 0, if x is 0 0, if x is 1 .
For the scenario where decision z Q is not equal to decision z U * , consider the following example. Let X = {0, 1} and Z = {0, 1}. Suppose that the density of ϕ(Y, z) given X = x is as follows.
For x ∈ {0, 1}, z = 0, the density is
f (ϕ) = 1 15 [1{0 ≤ ϕ < 1} + 2 2 k=1 1{4k -3 ≤ ϕ < 4k -1} + 2 k=1 1{4k -1 ≤ ϕ < 4k + 1} + 21{9 ≤ ϕ < 10}].
For x ∈ {0, 1}, z = 1, the density is
f (ϕ) = 1 15 [2I{0 ≤ ϕ < 1} + 2 k=1 I{4k -3 ≤ ϕ < 4k -1} + 2 2 k=1 I{4k -1 ≤ ϕ < 4k + 1} + I{9 ≤ ϕ < 10}].
It can be verified that when α = 0.4 -4ϵ/30 (ϵ is sufficiently small), we have
z Q = 0, if x is 0 0, if x is 1 .
On the contrary, the solution of the problem (4) is different
z U * = 0, if x is 0 1, if x is 1 , U * = {y ∈ Y : ϕ(y, 0) ≤ 6.5 + 2ϵ}, if x is 0 {y ∈ Y : ϕ(y, 1) ≤ 5}, if x is 1 .
Note that in the example above, X and Z are discrete spaces. We can naturally extend them to the continuous spaces [0, 1] while keeping the conclusions unchanged. The specific details are omitted here.
this section cite: ['b14', 'b42', 'b14', 'b42']

Section: B.2 RELATIONSHIP BETWEEN CRC AND RA-DPO, RA-CPO IN KIYANI ET AL. (2025)
For classification problems, Kiyani et al. (2025) proposed the following RA-DPO framework for the optimal decision:
min z(•),r(•) E[r(X)] s.t. P{ϕ (Y, z(X)) ≤ r(X)} ≥ 1 -α.(8)
This optimization problem can be viewed as a marginal version of the VaR problem. In addition, Kiyani et al. (2025) also defined an optimal decision framework based on prediction sets, called RA-CPO, as follows:
min U ( ):X →2 Y E[r U (X)] s.t. P{Y ∈ U(X)} ≥ 1 -α. (9
)
The difference between RA-CPO and CRC lies in the fact that the former employs a coverage constraint rather than a robustness constraint. We can leverage the idea from Theorem 3.2 in Kiyani et al. (2025) to prove the equivalence between the CRC problem and the RA-DPO, RA-CPO problem.
Proof of Theorem 3.1. Since RA-DPO and RA-CPO have been proved to be equivalent in Kiyani et al. (2025), it suffices to establish the equivalence between RA-DPO and CRC.
Let (z RA-DPO (x), r RA-DPO (x)) be an optimal solution to RA-DPO. Define the uncertainty set
U * (x) = y : ϕ y, z RA-DPO (x) ≤ r RA-DPO (x) .
Then,
P {ϕ (Y, z U * (X)) ≤ r U * (X)} ≥ P{Y ∈ U * (X)} ≥ P ϕ Y, z RA-DPO (X) ≤ r RA-DPO (X) ≥ 1 -α.
Thus, U * satisfies the constraint of CRC. Moreover, by definition,
r U * (x) = arg min z∈Z max y∈U * (x) ϕ(y, z) ≤ max y∈U * (x) ϕ y, z RA-DPO (x) ≤ r RA-DPO (x).
Hence,
E[r U * (X)] ≤ E[r RA-DPO (X)]
. This shows that any optimal solution of RA-DPO induces a feasible solution to CRC with a risk certificate at least as good. Conversely, let U * be an optimal solution to CRC. Define
z RA-DPO (x) = z U * (x), r RA-DPO (x) = r U * (x).
This pair is feasible for RA-DPO and satisfies E[r RA-DPO (X)] = E[r U * (X)]. Therefore, RA-DPO and CRC are equivalent, and the theorem follows.
this section cite: ['b23', 'b23', 'b23', 'b23']

Section: B.3 SUPERIORITY OF PARAMETRIZED CRC OVER PARAMETRIZED RA-CPO
The parametric formulation of RA-CPO in ( 9) is given by:
min θ∈Θ E [r θ (X)] s.t. P {Y ∈ U θ (X)} ≥ 1 -α.
The difference between parametrized RA-CPO and parametrized CRC (5) lies in the fact that the former employs a coverage constraint rather than a robustness constraint. The relationship between the two frameworks is formalized in the following proposition. Proposition B.2. For any parameterized prediction set U θ (•), it holds that
E[r θ CRC (X)] ≤ E[r θ RA-CPO (X)],
where θ CRC and θ RA-CPO denote the theoretical optimal solutions of the parametrized CRC and parametrized RA-CPO problems, respectively. Moreover, there exist cases in which the inequality is strict.
In fact, if no constraints are imposed on the prediction set, then as proven in Section B.2, the RA-CPO and CRC frameworks are equivalent. However, in regression settings, prediction sets are generally required to satisfy certain structural properties-such as convexity and boundedness-in addition to being parameterized to render the problem tractable. As a consequence, once the prediction set is parameterized, the solution derived from the CRC problem typically outperforms that obtained via RA-CPO.
Proof. We first show that E[r θ CRC (X)] ≤ E[r θ RA-CPO (X)].
Let U θ RA-CPO be the optimal solution to the RA-CPO problem. Since it also satisfies the constraints of the CRC problem, the inequality follows directly from the definition of the CRC problem.
We now proceed to construct a case where the inequality is strict. Consider a parameterized prediction set of the form:
U θ (x) = {y ∈ R q : (y -µ(x)) ⊤ Σ -1 (x)(y -µ(x)) ≤ θ}, θ ∈ R + , where Y | X ∼ N (µ(X), Σ(X)).
Let the loss function be ϕ(y, z) = -y ⊤ z. Then the coverage probability is given by:
P{Y ∈ U θ (X)} = P{χ 2 q ≤ θ}, where χ 2 q denotes a chi-squared random variable with q degrees of freedom. To analyze the robustness constraint, we derive the dual of the inner maximization in the CRO problem:
max y∈U θ (X) -y ⊤ z θ (X) = √ θ∥Σ 1/2 (X)z θ (X)∥ 2 -µ(X) ⊤ z θ (X).
By the definition of the robustness level, we have:
P ϕ(Y, z θ (X)) ≤ max y∈U θ (X) ϕ(y, z θ (X)) = P -Y ⊤ z θ (X) ≤ √ θ∥Σ 1/2 (X)z θ (X)∥ 2 -µ(X) ⊤ z θ (X) = P -z θ (X) ⊤ (Y -µ(X)) ≤ √ θ∥Σ 1/2 (X)z θ (X)∥ 2 = P -z θ (X) ⊤ (Y -µ(X)) ∥Σ 1/2 (X)z θ (X)∥ 2 ≤ √ θ = P{N (0, 1) ≤ √ θ}.
Therefore, when q ≥ 1, we obtain:
θ CRC = Φ 2 1-α < χ 2 q,1-α = θ RA-CPO ,
where χ 2 q,1-α and Φ 1-α denote the (1 -α)-quantiles of the χ 2 q and N (0, 1) distributions, respectively. In this case, it follows that: t ← 0.
E[r θ CRC (X)] < E[r θ RA-CPO (X)],
7:
s ← 1 n-n0+1 n+1 i=n0+1 1 ϕ Y y i , z θ0,t (X i ) ≤ r θ0,t (X i ) . 8: while s < 1 -α do 9: t ← t + τ 0 . 10: s ← 1 n-n0+1 n+1 i=n0+1 1 ϕ Y y i , z θ0,t (X i ) ≤ r θ0,t (X i ) . 11: end while 12: ty ← t. 13: if ϕ y, z θ0, ty (X n+1 ) ≤ r θ0, ty (X n+1 ) then 14: U Cal (X n+1 ) ← U Cal (X n+1 ) ∪ {y}. 15: end if 16: end for 17: Anti-discretization: U Cal (X n+1 ) ← A -1 ( U Cal (X n+1 )). 18: Output: U Cal (X n+1 ).
this section cite: []

Section: B.5 OPTIMALITY ANALYSIS OF CAL-CRC
Under certain conditions, the discrepancy between r U Cal (X n+1 ) and r U θ0 (X n+1 ) is expected to be negligible. For instance, if ty = 0 for any y ∈ Y, then by the definition of U Cal , we have
U θ0 (X n+1 ) ⊂ U Cal (X n+1 ). Consequently, max y∈U Cal (Xn+1) ϕ(y, z U Cal (X n+1 )) (a) ≥ max y∈U θ0 (Xn+1) ϕ(y, z U Cal (X n+1 )) (b) ≥ max y∈U θ0 (Xn+1) ϕ(y, z U θ0 (X n+1 )),
where (a) follows from the inclusion relationship between the two prediction sets, (b) holds due to the optimality of z U θ0 (X n+1 ) over U θ0 (X n+1 ). On the other hand, from a different perspective,
max y∈U Cal (Xn+1) ϕ(y, z U Cal (X n+1 )) (c) ≤ max y∈U Cal (Xn+1) ϕ(y, z U θ0 (X n+1 )) (d) ≤ max y∈U θ0 (Xn+1) ϕ(y, z U θ0 (X n+1 )),
where (c) is due to the optimality of z U Cal (X n+1 ) over U Cal , and (d) follows from the definition of the U Cal . Combining these results yields r U θ0 (X n+1 ) = r U Cal (X n+1 ). We now consider a more general setting. First, we state the generalized conditions, and then present the corresponding theoretical results. Condition B.1. Assume that for all y ∈ Y, we have | ty | ≤ t 0 , where t 0 is a positive constant.
Condition B.2. Loss function ϕ is L ϕ -Lipschitz in decision z for any y ∈ Y. The decision z θ0,t (X n+1 ) is L z -Lipschitz in t ≤ t 0 . The risk certificate r θ0,t (X n+1 ) is L r -Lipschitz in t ≤ t 0 .
Theorem B.1. Suppose that θ0 is obtained by running Algorithm 1 on the training dataset D train . Under conditions B.1 and B.2 in the calibration process, the following result holds:
r U Cal (X n+1 ) ≤ r U θ0 (X n+1 ) + t 0 (L ϕ L z + L r ).
Proof. For any y ∈ U Cal (X n+1 ), we have
ϕ(y, z θ0 (X n+1 )) ≤ ϕ(y, z θ0, ty (X n+1 )) + t 0 L ϕ L z ≤ r θ0, ty (X n+1 ) + t 0 L ϕ L z ≤ r U θ0 (X n+1 ) + t 0 (L ϕ L z + L r ).
Therefore,
max y∈U Cal (Xn+1) ϕ(y, z Cal (X n+1 )) ≤ max y∈U Cal (Xn+1) ϕ(y, z θ0 (X n+1 )) ≤ r U θ0 (X n+1 ) + t 0 (L ϕ L z + L r ).
Note that r θ0,t (x) is monotonically increasing in t for any x ∈ X . Hence, if the initial model U θ0 already approximately satisfies the 1 -α robustness requirement, the calibrated threshold ty will generally remain small for all y ∈ Y. As a result, U Cal can maintain risk certificates and decision losses comparable to those of the initial model U θ0 . Conversely, if the initial model's robustness is significantly below 1 -α, then although U Cal still guarantee 1 -α robustness, it may produce relatively conservative results.
this section cite: []

Section: C PROOF OF MAIN RESULTS IN SECTION 3.3 C.1 PROOF OF THEOREM 3.2
By leveraging the finite covering property of the function class and large-sample probability inequalities, we aim to prove that the empirical estimates converge uniformly to their expected values, thereby establishing the conclusion of the theorem. First, given an ϵ 2n -covering Θ ϵ2n with smallest cardinality of the function class, and applying the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality (Massart, 1990), we have
P sup t∈R,θ0∈Θϵ 2n 1 n n i=1 1 {V θ0 (X i , Y i ) ≤ t} -P {V θ0 (X, Y ) ≤ t} ≥ ϵ 1n ≤ 2N (Θ, ∥•∥ ∞ , ϵ 2n )e -2nϵ 2 1n ,
where ϵ 1n is the tolerance error, whose specific value will depend on the covering number N (Θ, ∥ • ∥ ∞ , ϵ 2n ) and will be specified later. According to the definition of ϵ 2n -covering, for any given θ ∈ Θ, there exists θ 0 ∈ Θ ϵ2n such that ∥θ -θ 0 ∥ ≤ ϵ 2n . Therefore, the upper bound on the deviation between the empirical estimate and the expected value can be derived as follows:
1 n n i=1 1{V θ (X i , Y i ) ≤ 0} -P {V θ (X, Y ) ≤ 0} ≤ 1 n n i=1 1{V θ (X i , Y i ) ≤ 0} - 1 n n i=1 1{V θ0 (X i , Y i ) ≤ 0}(10)
+ 1 n n i=1 1{V θ0 (X i , Y i ) ≤ 0} -P {V θ0 (X, Y ) ≤ 0} + |P {V θ0 (X, Y ) ≤ 0} -P {V θ (X, Y ) ≤ 0}| .
(11) Leveraging the Lipschitz condition, ( 10) and ( 11) can be bounded as follows:
1 n n i=1 1{V θ (X i , Y i ) ≤ 0} - 1 n n i=1 1{V θ0 (X i , Y i ) ≤ 0} ≤ 1 n n i=1 1 {V θ0 (X i , Y i ) ≤ (L ϕ L z + L r )∥θ -θ 0 ∥} - 1 n n i=1 1 {V θ0 (X i , Y i ) ≤ 0} + 1 n n i=1 1 {V θ0 (X i , Y i ) ≤ -(L ϕ L z + L r )∥θ -θ 0 ∥} - 1 n n i=1 1 {V θ0 (X i , Y i ) ≤ 0} ≤ 4 sup t∈R,θ0∈Θϵ 2n 1 n n i=1 1{V θ0 (X i , Y i ) ≤ t} -P {V θ0 (X, Y ) ≤ t} + sup θ0∈Θϵ 2n P {-(L ϕ L z + L r )ϵ 2n ≤ V θ0 (X, Y ) ≤ (L ϕ L z + L r )ϵ 2n } ,and
|P {V θ0 (X, Y ) ≤ 0} -P {V θ (X, Y ) ≤ 0}| ≤ |P{V θ0 (X, Y ) ≤ 0} -P {V θ0 (X, Y ) ≤ (L ϕ L z + L r )∥θ 0 -θ∥}| + |P{V θ0 (X, Y ) ≤ 0} -P {V θ0 (X, Y ) ≤ -(L ϕ L z + L r )∥θ 0 -θ∥}| ≤ sup θ0∈Θϵ 2n P {-(L ϕ L z + L r )ϵ 2n ≤ V θ0 (X, Y ) ≤ (L ϕ L z + L r )ϵ 2n } .
Finally, we consolidate the above results and obtain
1 n n i=1 1{V θ (X i , Y i ) ≤ 0} -P {V θ (X, Y ) ≤ 0} ≤ 5 sup t∈R,θ0∈Θϵ 2n 1 n n i=1 1{V θ0 (X i , Y i ) ≤ t} -P{V θ0 (X, Y ) ≤ t} + 2 sup θ0∈Θϵ 2n P {-(L ϕ L z + L r )ϵ 2n ≤ V θ0 (X, Y ) ≤ (L ϕ L z + L r )ϵ 2n } . Let ϵ 1n = log(2N (Θ,∥•∥∞,ϵ2n))+log(1/δ) 2n . We have sup θ∈Θ 1 n n i=1 1{V θ (X i , Y i ) ≤ 0} -P {V θ (X, Y ) ≤ 0} ≤5 log (2N (Θ, ∥ • ∥ ∞ , ϵ 2n )) + log(1/δ) 2n + 4(L ϕ L z + L r )ρ 0 ϵ 2n ,(12)
with probability at least 1 -δ. Furthermore, we have, with probability at least 1 -δ,
P ϕ Y, z θ (X) ≤ r θ (X) | D n ≥ 1 -α -5 log (2N (Θ, ∥ • ∥ ∞ , ϵ 2n )) + log(1/δ) 2n -4(L ϕ L z + L r )ρ 0 ϵ 2n ,
where θ is the solution to the CRC problem on dataset D n .
this section cite: []

Section: C.2 PROOF OF THEOREM 3.3
Let θ * be the convenient notation of θ * ∆n . The following formula gives the risk difference between the estimated model θ and the optimal model θ * :
E r θ (X) | D n -E [r θ * (X)] ≤ E[r θ (X) | D n ] - 1 n n i=1 r θ (X i ) (13
)
+ 1 n n i=1 r θ (X i ) - 1 n n i=1 r θ * (X i ) (14
)
+ 1 n n i=1 r θ * (X i ) -E[r θ * (X)] (15
)
For formulas ( 13) and ( 15), we adopt a proof strategy similar to that of Theorem 3.2 to demonstrate that the empirical estimates converge uniformly to their expected value. Given θ ∈ Θ, let θ 0 ∈ Θ ϵ2n be the approximation of θ in ϵ 2n -covering Θ ϵ2n . We have
1 n n i=1 r θ (X i ) -E[r θ (X)] ≤ 1 n n i=1 r θ (X i ) - 1 n n i=1 r θ0 (X i ) + 1 n n i=1 r θ0 (X i ) -E[r θ0 (X)] + |E[r θ0 (X)] -E[r θ (X)]| ≤ sup θ0∈Θϵ 2n 1 n n i=1 r θ0 (X i ) -E[r θ0 (X)] + 2L r ϵ 2n ,
where the last term is derived by applying the Lipschitz condition. According to Hoeffding's inequality, we have:
sup θ0∈Θϵ 2n 1 n n i=1 r θ0 (X i ) -E[r θ0 (X)] ≤ 2B r log(2N (Θ, ∥ • ∥ ∞ , ϵ 2n ) + log(1/δ) 2n ,
with probability at least 1 -δ. Furthermore, we can derive upper bounds for formulas (13) and ( 15). For formula (14), we assume that event (12) in the proof of Theorem 3.2 holds. At this point, since θ is the solution to the finite-sample CRC problem, we deduce the following result:
1 n n i=1 r θ (X i ) - 1 n n i=1 r θ * (X i ) ≤ 0.
Integrating the above conclusions, we can obtain the following result:
E[r θ (X) | D n ] -E[r θ * (X)] ≤ 4B r log(2N (Θ, ∥ • ∥ ∞ , ϵ 2n ) + log(1/δ) 2n + 4L r ϵ 2n .
holds with probability at least 1 -2δ.
this section cite: []

Section: C.3 PROOF OF THEOREM 4.1
Suppose that the calibration set is
D cal = {(X i , Y i )} n i=n0+1 , the test data is (X n+1 , Y n+1
) and a model θ0 has been trained from the training set
D train = {(X i , Y i )} n0
i=1 . First, we demonstrate that the prediction set U Cal (•) achieves 1 -α coverage. Note that
P{Y n+1 ∈ U Cal (X n+1 )} = P ϕ Y n+1 , z θ0, tY n+1 (X n+1 ) ≤ r θ0, tY n+1 (X n+1 ) . Let W = {(X n0+1 , Y n0+1 ), ..., (X n+1 , Y n+1
)} be an unordered set. Note that tYn+1 is measurable with respect to statistic W . We will complete the proof by leveraging the symmetry of the data.
P ϕ Y n+1 , z θ0, tY n+1 (X n+1 ) ≤ r θ0, tY n+1 (X n+1 ) = E E 1 ϕ Y n+1 , z θ0, tY n+1 (X n+1 ) ≤ r θ0, tY n+1 (X n+1 ) | W = E 1 n -n 0 + 1 n+1 i=n0+1 1 ϕ Y i , z θ0, tY n+1 (X i ) ≤ r θ0, tY n+1 (X i ) ≥ 1 -α.
The first equality stems from the law of total expectation. The second equality arises from the symmetry of the data, a technique frequently employed in proofs within conformal prediction methods (Vovk et al., 2005;Liang et al., 2024). The final inequality is derived from the definition of threshold tYn+1 , as referenced in Algorithm 2. Finally, since the coverage constraint is a sufficient condition for the robustness constraint, we can obtain the robustness guarantee, i.e.,
P {ϕ (Y n+1 , z U Cal (X n+1 )) ≤ r U Cal (X n+1 )} ≥ P{Y n+1 ∈ U Cal (X n+1 )} ≥ 1 -α.
this section cite: ['b38', 'b26']

Section: D ADDITIONAL THEORETICAL RESULTS

this section cite: []

Section: D.1 THEORETICAL RESULTS FOR VC/RADEMACHER CLASS
In this section, we present theoretical results on robustness and optimality when the function class has a finite VC dimension. Additionally, we discuss a decision-making method based on partitioning the covariate domain (Chenreddy et al., 2022). Under this approach, the corresponding function class possesses a finite VC dimension, thereby exhibiting relevant convergence properties.
Let VC(C) := VC({(x, y) → 1 {ϕ(y, z θ (x)) ≤ r θ (x)} : θ ∈ Θ}) denote the VC dimension of the robustness-induced classifier class. Theorem D.1 (VC class robustness). Suppose VC(C) = H < ∞. Then there exists an absolute constant C > 0 such that, with probability at least 1 -δ,
P ϕ Y, z θ (X) ≤ r θ (X) | D n ≥ 1 -α -C H n - log(2/δ) 2n .
Proof. By McDiarmid's Inequality, with probability at least 1 -δ,
sup θ∈Θ 1 n n i=1 1{ϕ (Y i , z θ (X i )) ≤ r θ (X i )} -P {ϕ (Y, z θ (X)) ≤ r θ (X)} ≤ E sup θ∈Θ 1 n n i=1 1{ϕ (Y i , z θ (X i )) ≤ r θ (X i )} -P {ϕ (Y, z θ (X)) ≤ r θ (X)} (16
) + log(2/δ) 2n .
The expectation in (16) can be bounded using the standard VC-class Rademacher bounds (Vershynin, 2018, Theorem 8.3.23): there exists a constant C such that
E sup θ∈Θ 1 n n i=1 1{ϕ (Y i , z θ (X i )) ≤ r θ (X i )} -P {ϕ (Y, z θ (X)) ≤ r θ (X)} ≤ C H n .
Combining these results yields that, with probability at least 1 -δ:
P ϕ Y, z θ (X) ≤ r θ (X) | D n ≥ 1 -α -C H n - log(2/δ) 2n .
Theorem D.2 (Rademacher risk). Assume additionally that |r θ (x)| ≤ M for all θ ∈ Θ and x ∈ X .
Let θ * ∆n be the optimal solution of problem (5) at robustness level 1 -α + ∆ n where
∆ n = C H n + log(2/δ) 2n
. Then, with probability at least 1 -2δ,
E r θ (X) -r θ * (X) | D n ≤ 4R n ({r θ (•) : θ ∈ Θ}) + 2M log(4/δ) 2n ,
where R n ({r θ (•) : θ ∈ Θ}) denotes the Rademacher complexity for function class {r θ (•) : θ ∈ Θ}.
Proof. Let θ * = θ * ∆n . We bound the risk difference between the estimated model θ and the optimal model θ * as follows:
E r θ (X) | D n -E [r θ * (X)] ≤ E[r θ (X) | D n ] - 1 n n i=1 r θ (X i ) + 1 n n i=1 r θ (X i ) - 1 n n i=1 r θ * (X i ) + 1 n n i=1 r θ * (X i ) -E[r θ * (X)] ≤ 2 sup θ∈Θ 1 n n i=1 r θ (X i ) -E [r θ (X)](17)
+ 1 n n i=1 r θ (X i ) - 1 n n i=1 r θ * (X i ).(18)
For the term (17), by McDiarmid's inequality, with probability 1 -δ:
sup θ∈Θ 1 n n i=1 r θ (X i ) -E [r θ (X)] ≤ E sup θ∈Θ 1 n n i=1 r θ (X i ) -E [r θ (X)] + 2M log(2/δ) 2n .(19)
The expectation in ( 19) is bounded via Rademacher complexity:
E sup θ∈Θ 1 n n i=1 r θ (X i ) -E [r θ (X)] ≤ 2E sup θ∈Θ 1 n n i=1 ϵ i r θ (X i ) = 2R n ({r θ (•) : Θ}).
For the term (18), whenever
1 n n i=1 1{ϕ (Y i , z θ * (X i )) ≤ r θ * (X i )} -P {ϕ (Y, z θ * (X)) ≤ r θ * (X)} ≤ C H n + log(2/δ) 2n
the definition of problem (6) implies
1 n n i=1 r θ (X i ) - 1 n n i=1 r θ * (X i ) ≤ 0.
By Theorem D.1, this event holds with probability at least 1 -δ. A union bound gives that, with probability at least 1 -2δ: In Chenreddy et al. (2022), the authors introduce a decision-making framework that leverages data-driven learning of underlying structures to categorize individuals into K classes based on their covariates. For each class, a prediction set is constructed, which in turn induces specific decisions and risk certificates. Denote the trained classifier by A : X → [K] and the model parameters by θ. The decisions and risk certificates take the following forms:
E r θ (X) | D n -E [r θ * (X)] ≤ 4R n ({r θ (•) : θ ∈ Θ}) + 2M log(2/δ) 2n . Remark D.1.
z θ (x) = K k=1 z k θ 1{A(x) = k} r θ (x) = K k=1 r k θ 1{A(x) = k}, where z k ∈ Z, r k ∈ R for each k ∈ [K]. Considering a portfolio optimization problem with loss function ϕ(y, z) = -y ⊤ z and Y = R q , the set {(x, y) → 1 {ϕ(y, z θ (x)) ≤ r θ (x)} : θ ∈ Θ}
becomes a subset of the following family:
(x, y) → 1 K k=1 a ⊤ k y -r k 1 {A(x) = k} ≤ 0 : a k ∈ R q , r k ∈ R for all k ∈ [K] .
This family corresponds to a finite-dimensional linear space of functions and therefore has VC dimension at most (q + 1)K. Applying Theorem D.1, we obtain the following convergence guarantee: with probability at least 1 -δ,
P ϕ Y, z θ (X) ≤ r θ (X) | D n ≥ 1 -α -C (q + 1)K n - log(2/δ) 2n ,
Similarly, the function class {r θ (x) : θ ∈ Θ} is uniformly bounded and has VC dimension at most
K + 1. Hence, its Rademacher complexity satisfies R n ({r θ : θ ∈ Θ}) ≤ C ′ K+1 n
for some constant C ′ . This leads to the following bound on the excess risk: with probability at least
1 -2δ, E r θ (X) -r θ * (X) | D n ≤ 4C ′ K + 1 n + 2M log(2/δ) 2n .
It is worth noting that, under the finite VC dimension condition, the resulting convergence rate achieves the order O( 1/n).
this section cite: ['b12', 'b12']

Section: D.2 THEORETICAL RESULTS UNDER SMOOTH CONSTRAINT
In this section, we we analyze the theoretical properties of the optimal solution to the following smoothed optimization problem:
θ = arg min θ∈Θ 1 n n i=1 r θ (X i ) s.t. 1 n n i=1 1 {ϕ (Y i , z θ (X i )) ≤ r θ (X i )} ≥ 1 -α.(20)
Here,
1{a ≤ b} = 1 2 (1 + erf( b-a √ 2σ )), where erf(x) = 2 √ π
x 0 e -t 2 dt is the Gaussian error function and σ controls the smoothness of the surrogate. This formulation provides a smoothed approximation of (6) and serves as the direct optimization target in Algorithm 1. The next two theorems provide a non-asymptotic guarantees of the robustness and expected risk certificate value of the resulting solution.
Theorem D.3 (Robustness). Let Θ ϵ denote an ϵ-covering of the Θ with coverage number N (Θ, ∥ • ∥ ∞ , ϵ), and let θ be the solution of problem (20). Under Conditions 3.1-3.2, for any independent data (X, Y ) ∼ P and conditioning on the labeled data D n , we have
P ϕ Y, z θ (X) ≤ r θ (X) | D n ≥ 1 -α - log(2N (Θ, ∥ • ∥ ∞ , ϵ) + log(1/δ) 2n - 2(L z L ϕ + L r )ϵ √ 2πσ - π 2 σρ 0 ,
with probability at least 1 -δ.
Proof. For any θ ∈ Θ, let θ 0 ∈ Θ ϵ such that ∥θ -θ 0 ∥ < ϵ. We decompose the deviation as follows:
1 n n i=1 1 {ϕ (Y i , z θ (X i )) ≤ r θ (X i )} -E 1 {ϕ (Y, z θ (X)) ≤ r θ (X)} ≤ 1 n n i=1 1 {ϕ (Y i , z θ (X i )) ≤ r θ (X i )} - 1 n n i=1 1 {ϕ (Y i , z θ0 (X i )) ≤ r θ0 (X i )} (21) + 1 n n i=1 1 {ϕ (Y i , z θ0 (X i )) ≤ r θ0 (X i )} -E 1 {ϕ (Y, z θ0 (X)) ≤ r θ0 (X)} (22
)
+ E 1 {ϕ (Y, z θ0 (X)) ≤ r θ0 (X)} -E 1 {ϕ (Y, z θ (X)) ≤ r θ (X)} (23
)
We can apply the Lipschitz condition to bound term (21):
1 n n i=1 1 {ϕ (Y i , z θ (X i )) ≤ r θ (X i )} - 1 n n i=1 1 {ϕ (Y i , z θ0 (X i )) ≤ r θ0 (X i )} ≤ 1 n n i=1 1 {ϕ (Y i , z θ (X i )) ≤ r θ (X i )} -1 {ϕ (Y i , z θ0 (X i )) ≤ r θ0 (X i )} (a) ≤ 1 nσ √ 2π n i=1 |ϕ (Y i , z θ (X i )) -ϕ (Y i , z θ0 (X i ))| + |r θ (X i ) -r θ0 (X i )| (b) ≤ (L z L ϕ + L r )ϵ √ 2πσ ,
where (a) is due to the fact that function 1{•, •} is 1 σ √ 2π -Lipschitz continuous with respect to its both components, and (b) is derived from condition 3.1. For the term (23), we can apply the same method to derive its upper bound:
E 1 {ϕ (Y, z θ0 (X)) ≤ r θ0 (X)} -E 1 {ϕ (Y, z θ (X)) ≤ r θ (X)} ≤ E 1 {ϕ (Y, z θ0 (X)) ≤ r θ0 (X)} -1 {ϕ (Y, z θ (X)) ≤ r θ (X)} ≤ 1 √ 2πσ E [|ϕ (Y, z θ (X)) -ϕ (Y, z θ0 (X))| + |r θ (X) -r θ0 (X)|] ≤ (L z L ϕ + L r )ϵ √ 2πσ .
For term ( 22), by Hoeffding's inequality and a union bound over θ 0 ∈ Θ ϵ , with probability at least 1 -δ,
sup θ0∈Θϵ 1 n n i=1 1 {ϕ (Y i , z θ0 (X i )) ≤ r θ0 (X i )} -E 1 {ϕ (Y, z θ0 (X)) ≤ r θ0 (X)} ≤ log(2N (Θ, ∥ • ∥ ∞ , ϵ)) + log(1/δ) 2n .
Combining these bounds yields:
E 1 ϕ Y, z θ (X) ≤ r θ (X) | D n ≥ 1 -α - log(2N (Θ, ∥ • ∥ ∞ , ϵ)) + log(1/δ) 2n - 2(L z L ϕ + L r )ϵ √ 2πσ
with probability at least 1 -δ. Finally, we quantify the discrepancy between the robustness
E [1{ϕ (Y, z θ (X)) ≤ r θ (X)}] and its smoothed version E 1{ϕ (Y, z θ (X)) ≤ r θ (X)} . Let f θ (•)
denote the density of V θ (X, Y ). Then:
E 1{ϕ(Y, z θ (X)) ≤ r θ (X)} -1{ϕ(Y, z θ (X)) ≤ r θ (X)} = 0 -∞ 1 - 1 2 1 + erf( -t √ 2σ ) f θ (t)dt + +∞ 0 1 2 1 + erf( -t √ 2σ ) f θ (t)dt (a) ≤ ρ 0 2 0 -∞ 1 -erf( -t √ 2σ )dt + ρ 0 2 +∞ 0 1 + erf( -t √ 2σ )dt (b) ≤ π 2 σρ 0 ,
where (a) follows from the bounded density assumption 3.2, and (b) is derived via standard Gaussian integral identities. Incorporating this bound into the previous result, we conclude that with probability at least 1 -δ,
P ϕ Y, z θ (X) ≤ r θ (X) | D n ≥ 1 -α - log(2N (Θ, ∥ • ∥ ∞ , ϵ)) + log(1/δ) 2n - 2(L z L ϕ + L r )ϵ √ 2πσ - π 2 σρ 0 .
The key difference from the non-smoothed case is the presence of the term π 2 σρ 0 , which quantifies the bias introduced by the smoothing. Below, we directly present the relevant optimality theorem, as its proof and conclusions are almost identical to the non-smoothed case. Theorem D.4 (Optimality). Let θ * ∆n be the optimal solution of problem (5) at the robustness level
1 -α + ∆ n where ∆ n = log(2N (Θ,∥•∥∞,ϵ)+log(1/δ) 2n + 2(LzL ϕ +Lr)ϵ √ 2πσ + π 2 σρ 0 . Under the same conditions of Theorem 3.3, conditioning on D n , we have E r θ (X) -r θ * ∆n (X) | D n ≤4B r log(2N (Θ, ∥ • ∥ ∞ , ϵ)) + log(1/δ) 2n + 4L r n ,
with probability at least 1 -2δ.
this section cite: []

Section: E EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

this section cite: []

Section: E.1 EXPERIMENTAL DETAILS OF CRC
To accelerate the alternating optimization of CRC and promote stable convergence, we partition the labeled data into two mutually exclusive parts: the first part is used for pretraining CRC, with the resulting parameters serving as initialization for subsequent alternating optimization; the second part is exclusively dedicated to the alternating optimization phase.
The baseline method employs the same partitioning strategy: the first portion trains the prediction model, while the second portion is used for calibration or solving downstream optimization tasks.
To ensure comparability, all methods uniformly employ the same scoring function and optimization objective in experiments.
Pre-training Pre-training of the CRC can be approached in two ways depending on the shape of the prediction set: For ellipsoidal prediction sets, the neural network outputs the parameters of a multivariate Gaussian, namely the mean vector μ(•) and the covariance matrix Σ(•). We parameterize Σ(•) via a Cholesky factorization, Σ(•) = L(•)L(•) ⊤ , where L(•) is lower triangular.
To guarantee positive definiteness, we add a small diagonal jitter to the predicted covariance, i.e., Σ ′ = Σ+εI which raises the eigenvalue floor and ensures numerical stability of the Cholesky factorization. Additionally, our training objective is to maximize the Gaussian log-likelihood, equivalently to minimize the negative log-likelihood:
L θ = 1 (2π
) d 2 |Σ| 1 2 exp -1 2 (y -µ) ⊤ Σ -1 (y -µ) .
For box prediction sets, we use quantile regression to directly estimate quantiles. Concretely, we train a neural network f θ (x) to output the α-level quantile for input x. The 1 -α confidence interval is constructed as f α/2 θ (x), f 1-α/2 θ (x) . Benefiting from quantile regression, our training objective is to minimize pinball loss. Given a quantile level α ∈ (0, 1) and prediction ŷ = f θ (x), the loss for target y is
L α (y, ŷ) = α (y -ŷ), if y > ŷ, (1 -α) (ŷ -y), if y ≤ ŷ.
Optimization For CRC optimization, we use the cvxpylayers (Agrawal et al., 2019) Python package to implement the implicit function differentiation. The optimization is performed using the Adam optimizer, and we select the optimal combination of learning rates (1e-2, 1e-3, 1e-4) and L2 weight decay values (0, 1e-2, 1e-3) to minimize the optimization loss. Moreover, to mitigate overfitting and ineffective training, 20% of the data used for optimization is held out as a validation set. Early stopping is triggered when the loss on the validation set fails to decrease for 10 consecutive iterations or when the predefined maximum number of iterations is reached.
Smoothing parameters sensitivity For CRC method, we approximate the indicator with a smooth surrogate 1{a ≤ b}
= 1 2 (1 + erf( b-a √2σ
)). We compared the sensitivity of different smoothing parameters σ on CRC. The experimental results are summarized in Table 2. Lagrange multiplier update schedule sensitivity For dual variable λ, we investigated the results of CRC on Lagrange multiplier update schedule sensitivity which refers to the number of model parameter optimization steps performed before each update of λ. The experimental results will be shown in Table 3.
this section cite: ['b0']

Section: CRO
The CRO method is our implementation of the Predict-then-Calibrate framework proposed by Sun et al. (2023). Specifically, we first train a predictive model to parameterize the uncertainty set (e.g., outputting the mean and covariance of ellipsoidal prediction sets). Subsequently, we construct a prediction set on the calibration set that satisfies the target coverage requirement. Finally, the prediction set is directly embedded into a downstream robust optimization problem to solve for decisions and minimize task loss. Thus, this method reduces task loss while enhancing solution stability, all while ensuring coverage.
this section cite: ['b35']

Section: E2E E2E is an end-to-end robust optimization method proposed by Chenreddy & Delage (2024)
and Yeh et al. (2024). Unlike CRO, E2E aims to bridge uncertainty calibration with downstream task objectives by minimizing target loss through global optimization. Specifically, E2E first trains a prediction model capable of outputting parameters of uncertainty sets. It then computes non-conformity scores on the calibration set, determines the threshold q that satisfies the nominal coverage 1-α, and constructs the uncertainty set accordingly. Finally, under this uncertainty set, the robust optimization problem is solved to obtain the current task loss. The gradients of the task loss with respect to model parameters are backpropagated through the differentiable optimization layer to the prediction model, enabling collaborative updates of model parameters and task objectives. Consequently, the model achieves better alignment with real-world decisions while ensuring coverage and reducing task loss. For a fair comparison, we set the loss function in E2E method as the expected risk certificate.
this section cite: ['b42']

Section: E.3 DENSITY PLOT OF SIMULATION IN SECTION 5.1
The density plots of the risk certificate of three methods are given in Figure 6. Compared with other baseline methods, CRC has achieved the best performance.
this section cite: []

Section: E.4 COMPARISON RESULTS OF RAC AND CRC
Based on the RA-CPO/RA-DPO framework, Kiyani et al. (2025) proposed the Risk-Averse Calibration (RAC) method to solve decision-making problems in classification settings. However, this method strictly relies on the finiteness of the label space and the decision space, i.e., |Y| < ∞, |Z| < ∞. Consequently, the RAC method is more suitable for classification problems and is not applicable to regression problems since constructing the prediction set in Kiyani et al. (2025) requires solving the Value-at-Risk optimization problem, which is generally not tractable when the space is continuous. In contrast, our method is grounded in the CRO framework and derives final decisions by directly optimizing over the space of prediction sets, thereby maintaining applicability to continuous decision spaces.
To evaluate the performance of the RAC and CRC methods in regression tasks, we have to make certain adjustments to the RAC method. Specifically, a simple regression problem can be converted into a classification problem via discretization-that is, by partitioning the response and decision space into discrete bins, thus allowing RAC to be applied. However, it is important to note that in general regression settings involving high-dimensional response (such as the 15-dimensional U.S. stock problem in Section 5.2 ), discretization often leads to substantial computational overhead and considerable information loss, making the application of RAC infeasible. To ensure the validity of the RAC method, we consider the following simple regression problem with loss function ϕ(y, z) = -y ⊤ z and decision space Z = {z ∈ [0, 1] 2 : ∥z∥ 1 = 1}. The data is generated by
Y 1 = -1.33 • ϵ 1 Y 2 = -1 + 0.5 • ϵ 2
where Y = (Y 1 , Y 2 ) ∈ R 2 , and ϵ = (ϵ 1 , ϵ 2 ) ∈ R 2 . The covariate X ∼ N (0, I 2 ) is the spurious feature and noise ϵ 1 , ϵ 2 are independent standard Gaussian random variables. When implementing the RAC method, we need to discretize both the decision space and the label space as follows.
• For the decision space Z, we divide the first dimension z 1 ∈ [0, 1] into J equally-spaced points {z 1,1 , . . . , z 1,J }. Due to the constraint z 1 + z 2 = 1, the dicision space is discretized into the finite set Z dis = {(z 1,1 , 1 -z 1,1 ), . . . , (z 1,J , 1 -z 1,J )}.
• For the label space Y, we first restrict each dimension of Y to the interval between its 1% and 99% quantiles. This creates a bounded two-dimensional box, which benefits the RAC method by ensuring a bounded loss. This box is then divided uniformly into L × L regions, and the discretized label space Y dis is composed of the top-right endpoints of these regions.
The experimental results are reported in Table 4.
Table 4: The simulation results of CRC and RAC at the nominal level α = 0.1, where the sample size is n = 2000. For the abbreviation RAC(J, L), the numbers J, L refer to the discretization refinement of decision space and label space, respectively.
this section cite: ['b23', 'b23']

Section: Method Risk Certificate Decision Loss Robustness (%) Coverage (%)
CRC-E 1.384 ± 0.049 0.541 ± 0.065 90.0 ± 0.8 36.1 ± 1.7 RAC(6, 2)
1.730 ± 0.039 0.803 ± 0.021 89.0 ± 1.0 89.8 ± 1.1 RAC(6, 4)
1.687 ± 0.032 0.612 ± 0.072 90.7 ± 0.9 89.9 ± 1.0 RAC(6, 8)
1.592 ± 0.038 0.725 ± 0.042 91.6 ± 0.9 89.8 ± 1.0 RAC(11, 2) 1.732 ± 0.038 0.803 ± 0.021 89.1 ± 1.0 89.8 ± 1.0 RAC(11, 4) 1.684 ± 0.033 0.576 ± 0.066 91.0 ± 0.9 89.9 ± 0.9 RAC(11, 8) 1.583 ± 0.036 0.697 ± 0.042 91.6 ± 0.9 89.9 ± 1.0
this section cite: []

Section: E.5 SIMULATION RESULTS ON CAL-CRC
The experiment results of Cal-CRC under ellipsoid prediction set are shown in Figure 7, where the simulation setting is the same as that in Appendix E.4.
0.05 0.1 0.15 0.2 0.25 Nominal level 0.3 0.6 0.9 1.2 1.5 1.8 2.1 Loss Risk Certificate 0.05 0.1 0.15 0.2 0.25 Nominal level 0.00 0.15 0.30 0.45 0.60 0.75 0.90 Loss Decision Loss 0.05 0.1 0.15 0.2 0.25 Nominal level 0.5 0.6 0.7 0.8 0.9 1.0 Rate Robustness 0.05 0.1 0.15 0.2 0.25 Nominal level 0.0 0.2 0.4 0.6 0.8 1.0 Rate Coverage CRC-E Cal-CRC-E E2E-E CRO-E Figure 7: The results of risk certificate, decision loss, robustness, and coverage on synthetic data when varying nominal level α with identical sample size n = 2000. The horizontal gray dashed lines refer to robustness levels. The prediction sets are ellipsoids. E.6 SIMULATION RESULTS ON POLYHEDRAL PREDICTION SET In this section, we adopt the methodology from Bärmann et al. (2016) to construct a parametric formulation for polyhedral prediction sets and integrate it into our proposed CRC framework. Simulation experiments demonstrate that under polyhedral prediction sets, our method still exhibits better performance compared to baseline approaches.
Following Bärmann et al. (2016), the derivation of a parametric form for polyhedral prediction sets is inspired by the parametric representation of ellipsoidal prediction sets. Let B q = {y ∈ R q : ∥y∥ 2 ≤ 1} denote the unit sphere in R q , and let µ θ (•) : R p → R q and Σ θ (•) : R p → R q×q represent the parameterized mean and covariance functions with parameters θ, respectively. The parametric ellipsoidal prediction set can be equivalently defined as:
U E θ (x) = y ∈ R q : Σ -1/2 θ (x) (y -µ θ (x)) ∈ B q .
Published as a conference paper at ICLR 2026 Now, let B q be a polyhedral outer ϵ-approximation of B q , defined by B q = {y : Ky ≤ k},
where K ∈ R m×q , k ∈ R m are a fixed matrix and vector, respectively, and m denotes the number of polyhedral facets. The corresponding parametric polyhedral prediction set is then given by:
U P θ (x) = y ∈ R q : Σ -1/2 θ (x) (y -µ(x)) ∈ B q = y ∈ R q : KΣ -1/2 θ (x)y ≤ k + KΣ -1/2 θ (x)µ θ (x) .(25)
The construction of B q depends on the dimension q and the approximation tolerance ϵ. For instance, when q = 2 and ϵ = 0.01, an m = 23-facet polyhedron ensures that the approximation error remains below ϵ. In this case, the components in (24) are specified as:
k = 1 23 , K =     a 1 a 2 . . .
a 23     where a i = cos 2πi 23 , sin 2πi 23 for i = 1, ..., 23.
The polyhedral prediction set U P θ can be directly incorporated into the CRC framework. For example, in a portfolio optimization problem with loss function ϕ(y, z) = -y ⊤ z and decision space Z = {z : z ∈ [0, 1] q : ∥z∥ 1 = 1, z ≥ 0}. We can establish that both the decision z θ (x) and the risk certificate r θ (x) are differentiable with respect to θ. This enables the search of optimal prediction sets via gradient-based optimization. Furthermore, the theoretical conditions outlined in Section 3.3 continue to hold, ensuring the validity of the corresponding theorems in this extended setting.
In this simulation, we compared the performance of CRC with other methods. The experimental setup remains consistent with that described in Appendix E.4. The experimental results are summarized in Table 5. In this section, we conducted ablation experiments on CRC to compare the performance of CRC and calibrated method. For CRC, we used the parameters of the pre-trained model as the initial values for iteration. For Cal-CRC, we calibrated the model after CRC optimization. For Cal method, we calibrated the pre-trained model. The experimental setup is the same as Section 5.1 and the experimental results are summarized in Table 6.
this section cite: ['b6']

Section: References
Ref_id:b0 Title: Differentiable convex optimization layers Year: (2019)
Ref_id:b1 Title: Optnet: Differentiable optimization as a layer in neural networks Year: (2017)
Ref_id:b2 Title: Theoretical foundations of conformal prediction Year: (2024)
Ref_id:b3 Title: Conformal risk control Year: (2024)
Ref_id:b4 Title: Efficient and differentiable conformal prediction with general function classes Year: (2022)
Ref_id:b5 Title: Optimal model selection for conformalized robust optimization Year: (2025)
Ref_id:b6 Title: Polyhedral approximation of ellipsoidal uncertainty sets via extended formulations: a computational case study Year: (2016)
Ref_id:b7 Title: Robust Optimization Year: (2009)
Ref_id:b8 Title: Data-driven robust optimization Year: (2018)
Ref_id:b9 Title: Nonsmooth implicit differentiation for machine-learning and optimization Year: (2021)
Ref_id:b10 Title: Discretized conformal prediction for efficient distribution-free inference Year: (2018)
Ref_id:b11 Title: End-to-end conditional robust optimization Year: (2024-07)
Ref_id:b12 Title: Data-driven conditional robust optimization Year: (2022)
Ref_id:b13 Title: Stochastic subgradient method converges on tame functions Year: (2020)
Ref_id:b14 Title: Task-based end-to-end model learning in stochastic optimization Year: (2017)
Ref_id:b15 Title: Worst-case value-at-risk and robust portfolio optimization: A conic programming approach Year: (2003)
Ref_id:b16 Title: Nested conformal prediction and quantile out-of-bag ensemble methods Year: (2022)
Ref_id:b17 Title: Learning-based robust optimization: Procedures and statistical guarantees Year: (2021)
Ref_id:b18 Title: Model-agnostic nonconformity functions for conformal classification Year: (2017)
Ref_id:b19 Title: Conformal uncertainty sets for robust optimization Year: (2021)
Ref_id:b20 Title: Batch multivalid conformal prediction Year: (2023)
Ref_id:b21 Title: Conformal prediction with learned features Year: (2024)
Ref_id:b22 Title: Length optimization in conformal prediction Year: (2024)
Ref_id:b23 Title: Decision theoretic foundations for conformal prediction: Optimal uncertainty quantification for risk-averse agents Year: (2025)
Ref_id:b24 Title: Distribution-free prediction bands for non-parametric regression Year: (2014)
Ref_id:b25 Title: Distributionfree predictive inference for regression Year: (2018)
Ref_id:b26 Title: Conformal prediction after efficiencyoriented model selection Year: (2024)
Ref_id:b27 Title: Pascal Massart. The tight constant in the dvoretzky-kiefer-wolfowitz inequality. The Annals of Probability Year: (1952)
Ref_id:b28 Title: Incorporating asymmetric distributional information in robust value-at-risk optimization Year: (2008)
Ref_id:b29 Title: Inductive confidence machines for regression Year: (2002)
Ref_id:b30 Title: Conformal contextual robust optimization Year: (2024)
Ref_id:b31 Title: Conditional value-at-risk for general loss distributions Year: (2002)
Ref_id:b32 Title: Least ambiguous set-valued classifiers with bounded error levels Year: (2019)
Ref_id:b33 Title: Data-driven robust optimization based on kernel learning Year: (2017)
Ref_id:b34 Title: Learning optimal conformal classifiers Year: (2022)
Ref_id:b35 Title: Predict-then-calibrate: a new perspective of robust contextual LP Year: (2023)
Ref_id:b36 Title: Conditional value-at-risk: optimization approach Year: (2001)
Ref_id:b37 Title: Weak convergence Year: (1996)
Ref_id:b38 Title: Algorithmic learning in a random world Year: (2005)
Ref_id:b39 Title: Learning decision-focused uncertainty sets in robust optimization Year: (2023)
Ref_id:b40 Title: Error-quantified conformal inference for time series Year: (2025)
Ref_id:b41 Title: Selection and aggregation of conformal prediction sets Year: (2025)
Ref_id:b42 Title: End-to-end conformal calibration for optimization under uncertainty Year: (2024)
