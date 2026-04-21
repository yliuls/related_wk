Title: Probabilistic Factorial Experimental Design for Combinatorial Interventions
Abstract: A combinatorial intervention, consisting of multiple treatments applied to a single unit with potentially interactive effects, has substantial applications in fields such as biomedicine, engineering, and beyond. Given p possible treatments, conducting all possible 2 p combinatorial interventions can be laborious and quickly becomes infeasible as p increases. Here we introduce the probabilistic factorial experimental design, formalized from how scientists perform lab experiments. In this framework, the experimenter selects a dosage for each possible treatment and applies it to a group of units. Each unit independently receives a random combination of treatments, sampled from a product Bernoulli distribution determined by the dosages. Additionally, the experimenter can carry out such experiments over multiple rounds, adapting the design in an active manner. We address the optimal experimental design problem within an intervention model that imposes bounded-degree interactions between treatments. In the passive setting, we provide a closed-form solution for the near-optimal design. Our results prove that a dosage of 1 /2 for each treatment is optimal up to a factor of 1 + O( ln(n) /n) for estimating any k-way interaction model, regardless of k, and imply that O kp 3k ln(p) observations are required to accurately estimate this model. For the multiround setting, we provide a near-optimal acquisition function that can be numerically optimized. We also explore several extensions of the design problem and finally validate our findings through simulations.

Section: Introduction
In many domains, it is often of interest to consider the simultaneous application of multiple treatments/actions. For example, in cell biology, perturbing several genes is often necessary to induce a transition in cell state (Takahashi & Yamanaka, 2006). While a single treatment is constrained to a limited range of possible effects, a combinatorial intervention -comprising multiple treatments applied to the same unit -can result in a much wider array of outcomes. Much of this potential stems from the interactive effects between treatments, rather than merely the additive contributions of each. For example, perturbing paralogs (a pair of genes) can have a surprisingly larger effect than the sum of perturbing each gene individually, as one gene may compensate for the other, and only perturbing both simultaneously will effectively disrupt the pathway (Koonin, 2005). However, these interactions make the study of combinatorial interventions considerably more challenging than understanding single interventions alone, as each set of treatments may exhibit distinct interactions.
From the design perspective, the problem of testing combinatorial interventions to analyze the combined effects of various treatments is known as a factorial design (Fisher et al., 1966). Given p possible treatments with large p, it is often infeasible to conduct all possible 2 p combinatorial interventions, which corresponds to a full factorial design. To address the scalability challenge, fractional factorial designs are introduced, which test only a subset of possible combinations. However, selecting this subset is difficult when prior knowledge is limited. Choosing a suboptimal subset may lead to a biased understanding of the experimental landscape. In addition, performing a large and specified subset of combinatorial interventions can be laborious and impractical, as each combination must be precisely assembled. In the perturbation example, this involves synthesizing a unique guide sequence for each combination that targets the specific genes involved (Rood et al., 2024). However, when the combination size is large, this becomes infeasible as a longer guide sequence may lack sufficient penetrance to effectively enter the targeted cells. To tackle these issues, we here formalize and study a scalable and unbiased approach to design factorial experiments.
Inspired by how scientists perform library designs in the lab (Yao et al., 2024), we introduce probabilistic factorial experimental design. In this framework, the experimenter selects a dosage for each possible treatment and applies it to a group of units. Each unit independently receives a random combination of treatments, sampled from a product Bernoulli distribution determined by the specified dosages. In the perturbation example mentioned above, this setup formalizes a high-multiplicity of infection (MOI) perturbation experiment (Yao et al., 2024), where multiple perturbations are applied at various MOI to a plate of cells, and each cell receives a combination of perturbations randomly. The introduction of a probabilistic design via dosages allow us to interpolate between an unbiased but expensive full factorial design and a relatively scalable but restricted fractional factorial design. By adjusting the dosages, we can effectively scale up a full factorial design by controlling the proportion of units receiving each combination in a realistic manner. Crucially, this approach remains unbiased as it does not require restricting the experiment to a predetermined subset of treatments. The question is then how to optimally design the dosages, e.g., in order to efficiently learn the interactions.
Contributions. Our contributions are summarized below.
• We propose and introduce the probabilistic factorial design, motivated by library design experiments in the lab (section 3.1). This setup assumes that treatments are randomly assigned to a group of units according to a prescribed dosage vector. It provides a scalable and flexible approach to implement factorial experiments, which we show to encapsulate both full and fractional factorial designs as special cases.
• Within this framework, we address the problem of optimal experimental design, which involves optimizing the dosage vectors based on a given objective. Our main focus is on learning the underlying combinatorial intervention model using a Boolean function representation assuming bounded-order interactions.
-In the passive setting (section 4.2), we prove that assigning a dosage of 1 /2 to each treatment is near-optimal for estimating any k-way interaction model, leading to a sample complexity of O(kp 3k ln(p)). -In the active setting (section 4.3), we introduce an acquisition function that can be numerically optimized and demonstrate that it is also nearoptimal in theory.
• We explore several extensions to the design problem, including constraints on limited supply, heteroskedastic multi-round noise, and emulation of a target combinatorial distribution (section 5). Finally, we validate our theoretical findings through simulated experiments (section 6).
this section cite: ['b25', 'b16', 'b11', 'b23', 'b28', 'b28']

Section: Related Works
Factorial design. Factorial experimental design has been extensively studied for its efficacy in evaluating multiple treatments simultaneously. Classical methods include full and fractional factorial designs (Fisher et al., 1966), and have been applied to various applications in biology, agriculture, and others (c.f., (Hanrahan & Lu, 2006)). Full factorial design considers all possible treatment combinations, where each treatment may have multiple levels (Deming & Morgan, 1993;Lundstedt et al., 1998;Dean & Voss, 1999). These experiments are sometimes conducted in multiple blocks, where each block is expected to have a controlled condition of external factors and contains one replicate of either all or partial combinations. When the number of total treatments increases, conducting such experiments quickly becomes infeasible. In these cases, fractional factorial design are preferred where a subset of carefully selected treatment combinations are tested (Gunst & Mason, 2009). A 2 -m fractional design is one where 2 p-m samples are used, each with a different combination (Box et al., 1978). These combinations are carefully selected to minimize aliasing. Aliasing occurs when, for the combinations selected, the interactions are linearly dependent (Gunst & Mason, 2009;Mukerjee & Wu, 2007). In a full factorial design, there is linear independence, so there is no confounding when the model is fit. In a fractional design, some aliasing will always occur in a full-degree model; however, methods proposed in the literature select combinations such that the aliasing of important effects (i.e. degree-1 terms) does not occur (Gunst & Mason, 2009). With little prior knowledge, it is common to assume that low-order effects are more important than higher-order interactions and select designs to focus on loworder effects (Cheng, 2016). With a low-degree assumption, aliasing can be avoided entirely. Fractional designs can be classified by their resolution (denoted by R), which determines which interactions can be potentially confounded. For example, a Resolution V fractional design eliminates any confounding between lower than degree-3 interactions, appropriate for degree-2 functions (Montgomery, 2017). Of particular interest in literature are minimum aberration designs, which minimize the number of degree-l terms aliased with degree-R -l terms (Fries & Hunter, 1980;Cheng, 2016). However, scalability to high-dimensional problems remains a challenge, and efficient sampling methods such as Bayesian optimization are proposed (Mitchell et al., 1995;Kerr, 2001;Chang & Cheng, 2018).
The probabilistic setting proposed in this paper serves as a flexible realization of a factorial design that automatically generates a design resembling either a full factorial or a fractional factorial design, depending on the selected dosages. We formally discuss this in section 3.1.
this section cite: ['b11', 'b14', 'b8', 'b18', 'b7', 'b13', 'b3', 'b13', 'b21', 'b13', 'b5', 'b20', 'b12', 'b5', 'b19', 'b15', 'b4']

Section: Learning combinatorial interventions. Modeling combinatorial interventions is crucial for understanding their interactions and designing experiments.
There are multiple ways to model such interventions, often by imposing structures that relate different combinations. For example, the Bliss independence (Bliss, 1939) and Loewe additivity (Loewe, 1926) models are commonly used to describe additive systems where no interactions between treatments are assumed.
An alternative approach is to use a structural causal model (SCM) and the principal of independent causal mechanisms (Eberhardt, 2007;Eberhardt & Scheines, 2007). In particular, this assumes that (1) each single-variable intervention alters the dependency of that variable on its parent variables according to the SCM, and (2) a combinatorial intervention modifies each involved variable according to its respective single-variable intervention and then combines these changes in a factorized joint distribution. Within this framework, various types of interventions, including do-, hard-, and soft-interventions, can be defined (e.g., (Correa & Bareinboim, 2020;Zhang et al., 2023)). However, similar to the Bliss independence and Loewe additivity models, SCM-based approaches cannot capture interactions between treatments.
To model such interactions, one can use a generalized surface model, which can be instantiated via polynomial functions (Lee, 2010) or Gaussian processes (Shapovalova et al., 2022). Alternatively, Boolean functions provide another modeling framework (Agarwal et al., 2023a), where theoretical tools such as the Fourier transform can be leveraged (O'Donnell, 2008). Agarwal et al. has employed this approach, where sparsity and rank constraints are used to enforce structural assumptions on combinatorial interactions. In this paper, we also utilize Boolean functions, where we demonstrate their close relationship with generalized surface models. We show that interactions can be read-off from Fourier coefficients, allowing us to formalize assumptions about the degree of interactions.
this section cite: ['b2', 'b18', 'b9', 'b10', 'b6', 'b29', 'b17', 'b24', 'b22']

Section: Setup and Model
In this section, we propose and define the setup of probabilistic factorial experimental design. We then introduce the outcome model we use to model combinatorial interventions and discuss its applicability to model interactions.
this section cite: []

Section: Probabilistic Factorial Design Setup
Consider p possible treatments with 2 p total combinatorial interventions. In a probabilistic factorial experimental design, the experimenter chooses a vector of dosages, denoted by d = (d 1 , . . . , d p ) ∈ [0, 1] p , and applies the treatments at this level to n homogenous units. For sim-plicity, we consider no interference between units, where each unit independently receives a combinatorial intervention at random. Denote the intervention associated with unit m ∈ [n] = {1, . . . , n} by x m ∈ {-1, 1} p , where x m,i = 1 if and only if it receives a combinatorial intervention that contains treatment i. Here x m is randomly sampled according to a product Bernoulli distribution according to d, where
x m,i = 1 with probability d i , -1 with probability 1 -d i .
(
The experimenter can carry out such experiments for T times, with different dosages d 1 , • • • , d T , potentially in a sequential and adaptive manner. In combinatorial perturbation example in section 1, the dosage vector formalizes the multiplicity of infection of each considered perturbation.
Note that this setup reduces to traditional two-level factorial design (Fisher et al., 1966) by choosing d ∈ {0, 1} p . In particular, for any combinatorial intervention consisting of treatments in S ⊆ [p],
setting d i = 1 if i ∈ S or else d i = 0
gives rise to all units receiving S. Allowing for continuous d ∈ [0, 1] p generalizes this setup by enabling the allocation of units to different combinatorial interventions in a realistic and effective manner controlled by d.
this section cite: ['b11']

Section: Outcome Models for Combinatorial Interventions
Under this setup, we are interested in estimating the average treatment effect of combinatorial interventions. For unit m, we observe its treatment assignment x m and outcome y m ∈ R. We adopt the outcome model proposed by Agarwal et al. (2023b), where y m corresponds to a noisy observation of a real-valued Boolean function f : {-1, 1} p → R, i.e.,
y m = f (x m ) + ϵ m .
Here we assume ϵ m is independent among different units and is normally distributed with mean zero and variance σ 2 . This model choice has the flexibility of allowing for interactions between arbitrary sets of treatments, as we illustrate below.
The class of real-valued Boolean functions admits a representation via the Fourier basis
{ϕ S (x) = i∈S x i | S ⊆ [p]} by f (x) = S⊆[p] β S ϕ S (x).
Here β S = 1 2 p y∈{-1,1} p f (y)ϕ S (y) (see Appendix A for details). The Fourier coefficients are interpretable in the sense that the polynomial instantiation of the generalized response surface model (Lee, 2010) can be expressed in this form, where all the k-way interactions are captured by
{β S | S ⊆ [p], |S| ≤ k}.
In particular, the generalized response surface model can be written as follows.
Polynomial Instantiation. To capture the nonlinear interactions between treatments, we can model the outcome of combinatorial intervention x via
f (x) = p i=1 α i 1 xi=1 + p i,j=1 α ij 1 xi=xj =1 + . . . ,
where α S represents the contribution in the final outcome by the interaction among treatments in S. This model can be represented via the Fourier representation (see Appendix A for details), where
β S = S⊆T α T 2 |T | .
(
In a bounded-order interaction model, α S = 0 for large |S|. In particular, if α S = 0 for |S| > k, then β S = 0 for |S| > k according to Eq. ( 2). This motivates us to make the following assumptions on the Fourier coefficients.
Assumption 3.1 (Bounded-order interactions). The outcome model exhibits bounded-order interactions, i.e., there exists k = o(p) such that
β S = 0 if |S| > k.
We also assume that β is bounded in L 2 norm.
Assumption 3.2. (Boundedness of β) There exists a constant B such that ∥β∥ 2 ≤ B.
this section cite: ['b17']

Section: Optimal Experimental Design
In this section, we focus on optimal experimental design for learning the outcome model f . We consider extensions of these results in section 5. For the objective of learning f , we provide near-optimal design strategies for the choice of dosages d in both passive and adaptive scenarios. We start by introducing the estimators of f . All formal proofs in this section are deferred to Appendix B.
this section cite: []

Section: Estimators
Estimating the Fourier coefficients β accurately in turn gives an accurate estimate of f , as ∥ f (x) -f (x)∥ 2 ≤ ∥ β -β∥ 2 , where f (x) = S⊆[p] βS ϕ S (x). Therefore, it suffices to focus on β.
Denote the collected dataset as D = {(x m , y m ) | m ∈ [n]}.
Let the design matrix be X ∈ R n×K with K = k i=0 p k .
The columns of X corresponds all possible combinations (including size ≤ 1) with interactions, i.e., S ⊆ [p] with |S| ≤ k. The m-th row of X corresponds to the Fourier characteristics of the observed combination x m with
X m,S = ϕ S (x m ) = i∈S x m,i .
Given that X is randomly drawn according to the dosages d and its columns are correlated, it is possible that it is ill-conditioned for a standard linear regression estimator. Therefore, in order to control the estimation error, we use a truncated ordinary least squares (OLS) to estimate β:
β = (X ⊤ X ) -1 X ⊤ Y if K i=1 λ i (X ⊤ X ) -1 ≤ B 2 σ 2 , 0 otherwise.
Here λ denotes the eigenvalues of X ⊤ X and Y is the vector by stacking y m with m ∈ [n]. Note that this results in a null estimator when the eigenvalues are small. We use this to demonstrate the key ideas of our analysis in a simpler form In practice, when X is ill-conditioned, alternative estimators such as ridge regression can be used, where similar theoretical results can be derived (see Appendix B for details). The truncated OLS estimator satisfies the following property, which we utilize in our analysis.
Lemma 4.1. Given a fixed design matrix X , the truncated OLS estimator satisfies
min{ K i=1 σ 2 λ i (X ⊤ X ) ,∥β∥ 2 2 } ≤ E Y ∥ β -β∥ 2 2 ≤ min{ K i=1 σ 2 λ i (X ⊤ X ) , B 2 }.
(3)
this section cite: []

Section: Passive Setting
In this scenario, the experimenter decides the choice of the dosages d in a prospective fashion without considering any data collected in the past. This is in contrast to an active design, where collected data are utilized to decide the current design. Note that the first round of any active setting reduces to the passive scenario, as there is no collected data.
Suppose we have a budget of n units. To select d such that we can obtain the most accurate estimate of β after observing these units, it is natural to optimize the following objective:
E D ∥ β -β∥ 2 2 . (4
)
We show that this objective has a closed-form near-optimal solution of d = ( 1 /2, • • • , 1 /2), regardless of the order of the interactions.
Theorem 4.2. For the truncated OLS estimator,
d = ( 1 /2, • • • , 1 /2) is optimal up to a factor of 1 + O ln(n) n
with respect to Eq. ( 4). In addition, the minimizer of Eq. ( 4) lies in an l ∞ -norm ball centered on the half dosage with radius O ln(n) n .
Note that with the half dosage, the probability of observing any particular combinatorial intervention S ⊆ [p] is 2 -p . Therefore in the passive setting, it is always optimal to evenly administer every treatment so that the observed combinatorial interventions follow a uniform distribution.
Proof sketch. In Lemma 4.1, we show how to bound the expectation of the error ∥ β -β∥ 2 2 with respect to randomness in the outcome Y . To obtain the optimal dosage for Eq. ( 4), we need to additionally take expectation with respect to the randomness of X , which is where the dosages enter as the combinatorial interventions x are sampled from Eq. ( 1).
Note that E[X ⊤ X ] = n • Σ(d) ∈ R K×K , where
Σ(d) S,S ′ = i∈S∆S ′ (2d i -1),(5)
for any S,
S ′ ⊆ [p] such that |S|, |S ′ | ≤ k. 1
Intuition of the optimality of half dosages. For the standard OLS estimator, the expected squared error is
σ 2 K i=1 1 λ i (X ⊤ X )
.
If we directly swap X ⊤ X with its expected value, then we need to minimize
K i=1 1 λ i (Σ(d))
.
Note that tr(Σ(d)) = K for all d. Therefore, by the Cauchy-Schwarz inequality,
K i=1 λ i (Σ(d)) -1 is minimized if and only if λ i (Σ(d)) = 1 for all i ∈ [K], which is satisfied when Σ(d) = I K and d = ( 1 /2, . . . , 1 /2).
To formally show that the expected error is optimized with half dosages, we can use a concentration result for X ⊤ X which can be obtained using an ϵ-net argument (Vershynin, 2018) and Hoeffding's inequality. However, the eigenvalues of X ⊤ X enters the error computation through the denominators, which makes the computation difficult. In particular, E(
K i=1 λ i (X ⊤ X ) -1
) cannot be bounded due to exploding terms when λ min (X ⊤ X ) approaches zero. We resolve this difficulty by utilizing the bounds in Lemma 4.1. For d = (foot_0 /2, . . . , 1 /2), we use the upper bound to show that
E D ∥ β -β∥ 2 2 ≤ Kσ 2 n(1 -δ) + B 2 exp K ln 9 - nδ 2 8K 2
for any 0 < δ < 1. For d such that max i |2d i -1| > 0, we use the lower bound to show that,
E D ∥ β -β∥ 2 2 ≥ 1 -2 exp K ln 9 - δ 2 8K 2 • min{ σ 2 n(1 -max i |2d i -1| + δ) + σ 2 (K -1) n(1 + δ) , ∥β∥ 2 2 }, for any δ > 0. By choosing δ = ( 2 ln n /n) 1 /2 , we obtain that d = ( 1 /2, . . . , 1 /2) is optimal up to a factor of 1 + O( ln(n) n ).
As a corollary of the proof for Theorem 4.2, we can show the error of estimating β decays with a rate of n -1 .
Corollary 4.3. With d = ( 1 /2, . . . , 1 /2), there is
E D ∥ β -β∥ 2 2 ≤ 2Kσ 2 + 1 n for n > n 0 , where n 0 = O K 3 ln K .
Therefore in order to estimate a k-way interaction model correctly, O(K 3 ln(K)) = O kp 3k ln(p) samples suffice.
this section cite: ['b26']

Section: Active Setting
In this setting, the experimenter decides the choice of the dosages d sequentially in multiple rounds, where the observations from previous rounds can be used to inform the choice of dosage. Note that as discussed in Section 4.2, the first round of the active setting degenerates to the passive setting, where the optimal choice is d = ( 1 /2, . . . , 1 /2).
Consider round T > 1. Denote D t as the collected data and let X t be the design matrix obtained by D t at round t ≤ T . The goal is to minimize the following objective
E D T ∥ β -β∥ 2 2 | D 1 ∪ . . . D T .(6)
In this scenario, we can not obtain a closed form solution as the optimal choice of d depends on pre-collected D 1 ∪ . . . D T -1 , which can be arbitrary. However, we show it is possible to derive a near-optimal objective that can be easily computed and numerically optimized.
Theorem 4.4. The following choice of dosage:
d T = argmin d∈[0,1] p K i=1 1 λ i Σ(d) + 1 n T -1 t=1 X ⊤ t X t(7)
Algorithm 1 Active probabilistic factorial experimental design.
1: Initialize X ⊤ X = 0 M ×M . 2: for t = 1 to T do 3: if t = 1 then 4: set d = ( 1 /2, . . . , 1 /2); 5: else 6:
set d t = argmin d∈[0,1] p K i=1 1 λi(Σ(d)+ 1 n t-1 i=1 X ⊤ i Xi) . 7: end if 8:
Gather n observations according to Eq. ( 1) and form design matrix X t .
9: Update X ⊤ X ← X ⊤ X + 1 n X ⊤ t X t 10: end for 11: Return estimated β using all observations.
is optimal up to a factor of 1 + O ln(n) n with respect to Eq. ( 6).
In practice, we solve for d T by numerically optimizing the objective in Eq. ( 7) using the SLSQP solver in Scipy (Virtanen et al., 2020). The number of iterations for the optimizer to converge is roughly O(p 3 ), and the complexity of each iteration is O(nK 2 + K 3 ) (where the first term comes from the matrix multiplication of X T X and the second term comes from computing the eigenvalues of Σ(d)). Recall the definition of K to be the number of interactions under consideration, i.e. K = k i=0 p i = O(p k ) for small k. Therefore, the overall complexity is O(np 3k+3 + p 6k+3 ) for small k. In practice, we may recommend using a proxy, which only involves the inverse of the minimum eigenvalue:
d T = argmin d∈[0,1] p 1 λmin(Σ(d)+ 1 n T-1 t=1 X ⊤ t Xt)
. We found that numerically optimizing this was significantly faster and that the solver was consistently accurate. While the complexity computed above should be the same for this approach, in practice it takes many less iterations to converge. We summarize the procedure for the active setting in Algorithm 1.
this section cite: ['b27']

Section: Extensions
In this section, we consider several extensions and discuss how the design policy changes in different scenarios.
this section cite: []

Section: Limited Supply Constraint
Here, we consider the case where we have additional constraint on the possible dosages d:
p i=1 d i ≤ L, for some 0 < L < p 2 .(8)
We assume L < p 2 , as otherwise d = ( 1 /2, . . . , 1 /2) is feasible and therefore optimal. This case is inspired by a setting where we have supply constraints on treatments, or where we do not want to assign a unit too many treatments at once. Note that the constraint implies that the expected number of treatments assigned to a unit is at most L.
In the passive setting, we derive a closed-form near-optimal dosage for the pure-additive model, i.e. k = 1 in Assumption 3.1. This result requires understanding of the spectrum of Σ(d). In the no-interaction case, we are able to derive the characteristic polynomial for Σ(d), which becomes difficult when k > 1. However, empirical results show that the result, which we now state, to hold for k > 1 as well (see section 6).
Theorem 5.1. For the additive model with k = 1, among the dosages that satisfy the constraint in Eq. ( 8), the uniform dosage d with d i = L p for all i ∈ [p] is optimal up to a factor of 1 + O ln(n) n with respect to Eq. ( 9).
For non-additive models and the active setting, we note that Theorem 4.4, where the feasible region of d is modified according to Eq. ( 8), to still hold. Therefore, although no closed-form solution can be derived, we can still obtain a near-optimal solution via numerical optimization.
this section cite: []

Section: Heteroskedastic Multi-round Case
Our results can easily extend to the scenario where the noise in the outcomes varies by round. This case might be relevant when different rounds of experiments have systematic batch effects, e.g., if they are collected within different labs.
Assume that in round t, the variance of the observed outcome noise is σ 2 t . Note that in this setting, d = ( 1 /2, . . . , 1 /2) is still near-optimal for the first round. However, the optimal choice of dosage at round T becomes
d T = argmin d∈[0,1] p K i=1 1 λ i 1 σ 2 T Σ(d) + 1 n T -1 t=1 1 σ 2 i X ⊤ t X t
where we now scale the observations at round t by 1 σt and use the truncated OLS estimator on this modified dataset (Eq. ( 6)), following a weighted least squares approach.
this section cite: []

Section: Limited Intervention Cardinality
Consider the scenario where the set of possible treatments that can be applied has limited cardinality:
∥d∥ 0 ≤ L, for some 0 < L < p.
Suppose that d i ̸ = 0 for i ∈ D, where the cardinality |D| is bounded by L. Then it holds that X :,S = (-1) |S\D| X :,S∩D . Therefore the design matrix can be written as
X = X D Γ D
where X D denotes the submatrix of X corresponding to columns X :,S with S ⊆ D and Γ D consists of one-hot vectors as columns. In this case, we may estimate β only up to Γ D β, e.g., using the following truncated OLS estimator
(X ⊤ D X D ) -1 X ⊤ D Y if K i=1 λ i (X ⊤ D X D ) -1 ≤ B 2 σ 2 , 0 otherwise.
Note that this has a form similar to β, where using similar arguments as in Section 4, we can show that d i = 1 /2 for i ∈ D is near optimal. Thus, in the passive setting, the nearoptimal strategy becomes selecting a subset of treatments D with |D| ≤ L and setting d i = 1 /2 for i ∈ D and d i = 0 for i / ∈ D. As the estimator for Γ D β directly estimates entries β S of β with S ⊆ D, one can select D based on prior preference of which coefficients of β are of interest.
this section cite: []

Section: Emulating a Target Combinatorial Distribution
We consider a different problem that explores the possibility of emulating a target distribution of combinatorial interventions with one round of probabilistic factorial design.
Formally, let q be an arbitrary distribution over all possible combinatorial interventions, we are interested in approximating q with choices of d. Denote p d as the distribution over combinatorial interventions induced by dosage d. We use KL divergence D(q || p d ) to measure the approximation error. To optimize over d, note that p d is a product distribution and we have
D(q || p d ) = H(q) - p i=1 q i log(d i ) -(1 -q i ) log(1 -d i ),
where q i = xi=1 q(x i ) is the marginal distribution of receiving treatment i under the target distribution, and H(•) denotes the entropy. Minimizing this equality quickly obtains d i = q i , which indicates choosing d based on the marginals of the target distribution. The minimal approximation error is then H(q) -H(q 1 ⊗ . . . q p ), which means we can emulate a target distribution well if it is closed to a product distribution.
this section cite: []

Section: Experiments
We conduct experiments to validate our theoretical results, as well as show a comparison to fractional factorial design, using simulated data. We generate the outcome model f by sampling the Fourier coefficients from the uniform distribution, i.e., β ∼ U(-1, 1) K . We noise the outcomes with standard Gaussian noise. In each of the following simulations, we keep β constant through all iterations of each run. Further details and the code repository can be found in Appendix D.
this section cite: []

Section: Comparison to Fractional Factorial Design
Here we compare the half dosage versus a partial factorial design in the passive setting. We generate a degree-1 Boolean function with p = 8. We use a 2 8-2 Resolution V design with 64 samples for each approach.
The fractional design returns a mean squared error of 0.14 ± 0.062, where the half dosage gives 0.16 ± 0.078 (averaged over 300 trials and with ±1 std). With fewer samples, the careful selection of combinations will make a difference, so the fractional design can outperform the half dosage. But in many cases, especially in biological applications, careful selection of combinations is not possible which is why the much more flexible dosage design is preferable, as it enables the administration of an exponential number of combinations by choosing a linear number of dosages.
However, in the active setting, the optimal dosage can outperform a fractional design. This is discussed further in Section 6.3.
this section cite: []

Section: Passive Setting
In Theorem 4.2, we show that d = ( 1 /2, . . . , 1 /2) is optimal up to a factor of 1 + O( ln(n) n ). Empirically, our validations build on the comparison of estimation error between half dosages and randomly sampled dosage vectors. We consider two different ways to generate dosages in this comparison, as described below.
Simulation 1. Here, we investigate the approximation of β achieved by different dosages d based on their l ∞ -distances from the 1 2 := ( 1 /2, . . . , 1 /2), i.e., d -1 2 ∞ . We consider distances ranging from 0 to .4, where we sample 100 different dosage vectors at each distance. For each dosage, we generate 20 sets of observations and regress on each.  We show these results for three different sets of p, k, and n in Figure 6.2. These values are chosen such that the ratio K/n is kept approximately constant under different number of total treatments, following Corollary 4.3: p = 10, k = 2, n = 200; p = 20, k = 2, n = 1000; and p = 30, k = 2, n = 1000.
Simulation 2. Here, we only consider dosages where each treatment is administered at the same dosage, which we refer to as a uniform dosage. We consider dosage values ranging from .4 to .6, and generate 500 different observation sets for each dosage. We show the approximation error of β against the dosage value in Figure 2 for the three different sets of p, k, and n used in simulation 1.
this section cite: []

Section: Results.
In simulation 1, we see that the approximation error is generally increasing in d -1 2 ∞ . Even with relatively small n (on the scale of O(K), rather than poly(K) in Corollary 4.3), we see that the half dosage seems to be optimal. In simulation 2, we again see that the half dosage exhibits optimality, with U -shaped curves dipping at .5.
this section cite: []

Section: Active Setting
Here, we carry out 10 sequential experimental rounds. We compare our proposed choice of dosage in Theorem 4.4, which we refer to as optimal, to two baselines. The first baseline, referred to as random, randomly chooses a dosage from U(0, 1) p at each round. The second baseline, referred to as half, chooses the dosage of 1 2 at each round. On synthetic data, we find that optimal and half perform similarly when n is relatively large, while clearly outperforming random (Figure 3). In settings where n is relatively small, optimal outperforms half (Figure 4). We also add a partial design baseline, referred to as partial (a Resolution V 2 5-1 design), in the small p setting. In earlier rounds, we see optimal performs the best, and partial catches up after sufficiently many rounds.  Results. We see that random performs consistently worse that optimal and half. For high n (compared to K), the difference between optimal and half is marginal (as seen in Figure 3). However, when n is small, there is a noticeable gap between optimal and half. In the case where there are not many samples (compared to features) per round, we find that the optimal acquisition strategy more clearly outperforms the half strategy. This is because when we have a smaller number of samples, we will need to"correct" as the distribution of combinations will be more lopsided and further away from the uniform distribution. Similarly, this is why optimal can outperform partial in a multiple-round setting, though it may be subpar in a single round. Therefore, in scenarios where each round has few samples, we think it is worth computing the optimal acquisition dosage. When we have a large n relative to p, the half strategy and optimal strategy perform very similarly.
this section cite: []

Section: Extensions
In Theorem 5.1, we proved that the uniform dosage of L p is optimal in the constrained case for the simple additive models. Empirically, we see that this holds for interactive models as well, both in simulations and in numerically optimizing Eq. (4). For example, for the pairwise interaction case, Figure 5 shows the approximation error versus the deviation from the suspected optimal dosage. We see that with L = 2, n = 1000, and varying p = 8, 9, 10, the approximation error increases as we deviate from L p .
this section cite: []

Section: Misspecified model
In the case where we do not know the true degree of the highest-order interaction, our model may be misspecified case. While our theoretical results do not support this case, we conduct experiments that show that the half dosage still appears to be optimal in a single-round setting. Here, we use a Boolean function of full degree (with p = 5), and vary k between 2 and 4. So while the true function features interaction terms of all degrees, our assumption is that only terms of interaction up to k appear in f . We fit the model under these assumed values of k, and observe that a half dosage appears to still lead to the lowest estimation errors in Figure 6.
this section cite: []

Section: Discussion
In this work, we propose and study probabilistic factorial design, a scalable and flexible approach to implementing factorial experiments, which generalizes both full and fractional factorial designs. Within this framework, we tackle the optimal design problem, focusing on learning combinatorial intervention models using Boolean function representations with bounded-degree interactions. We establish theoretical guarantees and near-optimal desgin strategies in both passive and active learning settings. In the passive setting, we prove that a uniform dosage of 1 /2 per treatment is nearoptimal for estimating any k-way interaction model. In the active setting, we propose a numerically optimizable acquisition function and demonstrate its theoretical near-optimality. Additionally, we extend our approach to account for practical constraints, including limited supply, heteroskedastic multi-round noise, and emulating target combinatorial distributions. Finally, these theoretical results are validated through simulated experiments.
Limitations and Future Work. This work has several limitations and assumptions that may be interesting to address in future work. First, we assume a product infection mechanism in the probabilistic design. However, this assumption may not hold in certain scenarios, such as when interference or censoring effects are present. For example, in cell biology, experiments conducted on tissue samples may exhibit spatial interactions among neighboring cells. Additionally, certain treatment combinations may induce cell death, leading to a lack of observable units for those combinations. Second, our combinatorial intervention model could be extended to incorporate unit-specific covariates. The current model assumes that outcomes are determined solely by the received treatment, which suffices for homogenous units and average effects. However, incorporating covariate-based models would enable finer-grained personalized treatment-outcome predictions. Third, while we explore several extensions to the design problem, further investigations into alternative constraints, such as sparse interventions, and alternative objectives, such as optimizing specific outcome variables, could be valuable directions for future work.
this section cite: []

Section: References
Ref_id:b0 Title: Synthetic combinations: A causal inference framework for combinatorial interventions Year: (2023)
Ref_id:b1 Title: Synthetic combinations: A causal inference framework for combinatorial interventions Year: (2023)
Ref_id:b2 Title: The toxicity of poisons applied jointly 1 Year: (1939)
Ref_id:b3 Title: Statistics for Experimenters Year: (1978)
Ref_id:b4 Title: A bayesian approach to the selection of two-level multi-stratum factorial designs Year: (2018)
Ref_id:b5 Title: Theory of factorial design Year: (2016)
Ref_id:b6 Title: A calculus for stochastic interventions: Causal effect identification and surrogate experiments Year: (2020)
Ref_id:b7 Title: Design and analysis of experiments Year: (1999)
Ref_id:b8 Title: Experimental design: a chemometric approach Year: (1993)
Ref_id:b9 Title: Causation and intervention Year: (2007)
Ref_id:b10 Title: Interventions and causal inference Year: (2007)
Ref_id:b11 Title:  Year: (1966)
Ref_id:b12 Title: Minimum aberration 2 k-p designs Year: (1980)
Ref_id:b13 Title: Fractional factorial design Year: (2009)
Ref_id:b14 Title: Application of factorial and response surface methodology in modern experimental design and optimization Year: (2006)
Ref_id:b15 Title: Bayesian optimal fractional factorials Year: (2001)
Ref_id:b16 Title: Orthologs, paralogs, and evolutionary genomics Year: (2005)
Ref_id:b17 Title: Drug interaction: focusing on response surface models Year: (2010)
Ref_id:b18 Title: Experimental design and optimization. Chemometrics and intelligent laboratory systems Year: (1926)
Ref_id:b19 Title: Two-level fractional factorials and bayesian prediction Year: (1995)
Ref_id:b20 Title: Design and Analysis of Experiments Year: (2017)
Ref_id:b21 Title: A Modern Theory of Factorial Design Year: (2007)
Ref_id:b22 Title: Some topics in analysis of boolean functions Year: (2008)
Ref_id:b23 Title: Toward a foundation model of causal cell and tissue biology with a perturbation cell and tissue atlas Year: (2024)
Ref_id:b24 Title: Nonparametric synergy modeling of chemical compounds with gaussian processes Year: (2022)
Ref_id:b25 Title: Induction of pluripotent stem cells from mouse embryonic and adult fibroblast cultures by defined factors Year: (2006)
Ref_id:b26 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b27 Title: Scipy 1.0: fundamental algorithms for scientific computing in python Year: (2020)
Ref_id:b28 Title: Scalable genetic screening for regulatory circuits using compressed perturb-seq Year: (2024)
Ref_id:b29 Title: Active learning for optimal intervention design in causal models Year: (2023)
Ref_id:b30 Title: Experiment Details Code can be found at the linked repository. Below we give a few additional details of our experiments. Hardware and libraries. Experiments were run on a device with a 16 core Intel Core Ultra 7 165H processor with 32 GB RAM, and an NVIDIA RTX 4000 Mobile Ada Generation 12 GB GPU. The code is implemented in Python, utilizing the cupy and numba libraries, among others. The active design optimization was done using scipy SLSQP solver Year: ()
