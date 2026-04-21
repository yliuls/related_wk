Title: CausalPFN: Amortized Causal Effect Estimation via In-Context Learning
Abstract: Causal effect estimation from observational data is fundamental across various applications. However, selecting an appropriate estimator from dozens of specialized methods demands substantial manual effort and domain expertise. We present CausalPFN, a single transformer that amortizes this workflow: trained once on a large library of simulated data-generating processes that satisfy ignorability, it infers causal effects for new observational datasets out of the box. CausalPFN combines ideas from Bayesian causal inference with the large-scale training protocol of prior-fitted networks (PFNs), learning to map raw observations directly to causal effects without any task-specific adjustment. Our approach achieves superior average performance on heterogeneous and average treatment effect estimation benchmarks (IHDP, Lalonde, ACIC). Moreover, it shows competitive performance for real-world policy making on uplift modeling tasks. CausalPFN provides calibrated uncertainty estimates to support reliable decision-making based on Bayesian principles. This ready-to-use model requires no further training or tuning and takes a step toward automated causal inference (https://github.com/vdblm/CausalPFN).

Section: 
Causal inference-estimating the effects of interventions from data-is fundamental across numerous domains, including public policy, economics, and healthcare [75,5,47]. The central challenge lies in estimating causal quantities from observational data: records collected without explicit interventions, where confounding factors can obscure true causal effects. Various causal identification settings have emerged to address this challenge [45,6,10,71]. Perhaps the most common one is to assume no unobserved confounding (ignorability or backdoor) [98,87].
Even within the conceptually straightforward ignorability framework, researchers have developed dozens of specialized causal estimators over the past four decades. Prominent examples include Meta-Learners [57], doubly robust methods [30,52], double machine learning (DML) [16,29], and neural network approaches [104,106,[19][20][21][22], among others [97,56,70,65]. This large number of estimators creates practical challenges as domain expertise is required to select, tune, or design the most appropriate estimator for each application [107,103,27,2,81,73].
this section cite: ['b74', 'b4', 'b46', 'b44', 'b5', 'b9', 'b70', 'b97', 'b86', 'b56', 'b29', 'b51', 'b15', 'b28', 'b103', 'b105', 'b18', 'b19', 'b20', 'b21', 'b96', 'b55', 'b69', 'b64', 'b106', 'b102', 'b26', 'b1', 'b80', 'b72']

Section: * Equal Contribution
The Bayesian paradigm offers an elegant framework to address these challenges [99,46,47,40,9]; rather than manually designing or selecting the best estimator, one can: (1) parameterize an appropriate prior distribution over plausible underlying causal mechanisms, i.e., the data-generating processes (DGPs), (2) define the causal estimand as a functional of the DGP parameters, (3) compute a posterior distribution over DGPs conditioned on observed data, and (4) derive the posterior predictive distribution (PPD) of the causal estimand. However, the practical adoption of Bayesian methods remains limited. Computing posterior distributions typically requires expensive sampling methods [84,40], which often leads researchers to make specific assumptions about the DGPs or priors that are not necessarily reflective of the complexity of the downstream tasks [36,62].
Meanwhile, an emerging area in deep in-context learning suggests using large models that can approximate PPDs by taking the entire list of observations as context and amortize the expensive process of posterior inference [32,31,54]. A successful example is the prior-fitted network (PFN) [80] that achieved remarkable performance in tabular prediction tasks [42,69,37,43,118,76,66]. PFNs employ transformer architectures trained on large-scale simulated DGPs, representing a rich prior, to perform posterior predictive inference via in-context learning; given a dataset of input-output examples as context, they can predict outputs for new inputs. PFNs shift the computational burden from inference time to (pre-)training, producing a single set of model parameters that can make fast and accurate predictions on unseen datasets. However, they are only designed for regression and classification, not for causal inference.
We propose to bridge the large-scale training of amortized models with Bayesian causal inference and introduce CausalPFN, a transformer model for causal effect estimation via in-context learning. Our framework leverages a general-purpose prior, based on the ignorability assumption, to generate a vast collection of simulated DGPs. By training on these diverse DGPs, our method learns to infer the causal estimands directly from observational data. While our approach requires an expensive pre-training phase, once complete it is ready for inference on new datasets with no further training, fine-tuning, or hyperparameter optimization. Hence, CausalPFN is easy-to-use, efficient for inference, and shows remarkably strong performance as an estimator. Figure 1 illustrates the relative performance and efficiency of our method compared to standard baselines. For inference on an unseen dataset, CausalPFN requires only forward passes, whereas baseline methods have additional costs including hyperparameter tuning or cross-validation. We therefore report the computational time for all of these stages for the baselines to reflect the total costs of predicting on a new dataset.
We show CausalPFN's workflow compared to traditional causal inference in Figure 2. Our key contributions are: (i) To our knowledge, for the first time, we demonstrate that a single transformerbased model trained on a diverse library of simulated DGPs can match or surpass specialized estimators across multiple datasets without task-specific tuning. Specifically, CausalPFN achieves the best average rank on CATE across IHDP, ACIC, and Lalonde, and competitive ATE performance, without task-specific tuning. (ii) We highlight CausalPFN's competitive out-of-the-box performance for real-world policy making on various uplift modeling tasks. (iii) We theoretically characterize the assumptions under which CausalPFN's estimates are asymptotically consistent. (iv) We develop a principled uncertainty quantification framework for CausalPFN to produce finite-sample calibrated credible intervals for the estimates. (v) Finally, we release our model's weights with a user-friendly API, streamlining the adoption of CausalPFN as a capable estimator. CausalPFN is fast, ready-to-use, and does not require any further training or hyperparameter tuning.
this section cite: ['b98', 'b45', 'b46', 'b39', 'b8', 'b83', 'b39', 'b35', 'b61', 'b31', 'b30', 'b53', 'b79', 'b41', 'b68', 'b36', 'b42', 'b117', 'b75', 'b65']

Section: Background
Causal Effect Estimation. We adopt the potential-outcomes framework for causal inference [100]. Let T ∈ T denote the treatment from a finite treatment set T , and X ∈ X the observed covariates. For every t ∈ T , Y t ∈ R is the potential outcome under treatment t, while the observed (factual) outcome is Y := Y T . We call the joint distribution P (X, T, {Y t } t∈T , Y ) the data-generating process (DGP), and denote by P obs the marginal distribution over observed triples (X, T, Y ). Given samples from P obs , a central goal is to recover the conditional expected potential outcomes (CEPOs):
µ t (x) := E[Y t | X = x], ∀t ∈ T , x ∈ X .(1)
For binary treatments, two common estimands, average treatment effect (ATE), and conditional average treatment effect (CATE) follow directly from the CEPOs. We refer to CEPOs, CATE, and ATE collectively as causal effects.
ATE : λ := E[Y 1 -Y 0 ] = E[µ 1 (X) -µ 0 (X)],(2)
CATE : τ (x) := E[Y 1 -Y 0 | X = x] = µ 1 (x) -µ 0 (x).(3)
Estimating causal effects from observational data is impossible without further assumptions: different DGPs can induce the same P obs but have different causal effects [87,39,47]. We thus define: Definition 1 (CEPO-Identifiability). For each t ∈ T , CEPO-identifiability holds when µ t can be written as a functional of the observational distribution P obs .
Throughout, we assume strong ignorability, a standard sufficient assumption that makes CEPOs identifiable. Strong ignorability posits that, conditional on observed covariates, treatment assignment has positive probability for all t ∈ T and is independent of all potential outcomes [98,97,89]: Assumption 1 (Strong Ignorability). (i) Y t ⊥ ⊥ T | X for all t ∈ T (Unconfoundedness), and (ii) P (T = t | X) > 0 a.e. for all t ∈ T (Positivity).
this section cite: ['b99', 'b86', 'b38', 'b46', 'b97', 'b96', 'b88']

Section: Bayesian Causal Inference.
A Bayesian formulation of causal inference considers an explicit likelihood model for the DGP [99,84,62]. Let ψ be the parameter that indexes the DGPs
P ψ (X, T, {Y t } t∈T , Y ). A prior π(ψ) encodes domain knowledge on parameters ψ. Given i.i.d. observations D obs = (x (n) , t (n) , y (n) ) N n=1
coming from the observational distribution P ψ obs , Bayes' rule yields the posterior π(ψ | D obs ). For any functional g(ψ
)-for example g(ψ) = E ψ [Y 1 -Y 0 ] for ATE-the posterior predictive distribution (PPD) π g (• | D obs ) := B → I (g(ψ) ∈ B) π(ψ | D obs ) dψ , B ∈ B,(4)
is induced by the posterior distribution π(ψ | D obs ) (B denotes the Borel σ-algebra over R). Point estimates (posterior means) and credible intervals therefore arise automatically from these induced posteriors. Because the posterior is rarely available in closed form, one resorts to approximate inference such as Markov-chain Monte-Carlo (MCMC) [40] or variational inference [68,48]. Such techniques have been applied with flexible priors including nonparametric BART models [40,36], Dirichlet processes [64] and Gaussian processes [3]. In summary, the Bayesian paradigm offers a unified framework for inference on causal estimands and provides automatic uncertainty quantification.
this section cite: ['b98', 'b83', 'b61', 'b39', 'b67', 'b47', 'b39', 'b35', 'b63', 'b2']

Section: Amortizing Posterior Predictive Inference with Prior-Fitted Networks.
Running a new posterior inference for every dataset is computationally demanding, especially with high-dimensional covariates [36,62]. Recent work shows that in-context transformers can amortize Bayesian prediction: instead of sampling from the posterior at test time, a single network is trained to map a context set directly to the PPD [31,32,54,80,37]. PFNs instantiate this idea for supervised learning [42].
Consider a supervised dataset D SL = {(x (n) , y (n) )} N n=1
and a prior π SL on parameters ϕ indexing P ϕ (X, Y ). The Bayesian approach to predict the output for a new input x is to use the PPD
PPD Y X = x, D SL := P ϕ (Y | X = x)π SL ϕ D SL dϕ.(5)
Rather than approximating the posterior distribution π SL ϕ D SL with MCMC or variational inference [49,4,82], PFNs directly parameterize the PPD using a single transformer model q θ Y X, D SL by minimizing the data-prior loss
ℓ θ := E ϕ∼π SL , D SL ∪{X,Y }∼P ϕ -log q θ Y X, D SL .(6)
Crucially, training requires only prior samples (ϕ, D SL ); no posterior sampling is needed. With a suitably rich prior, a single PFN can be applied off-the-shelf to diverse predictive problems [69,43].
this section cite: ['b35', 'b61', 'b30', 'b31', 'b53', 'b79', 'b36', 'b41', 'b48', 'b3', 'b81', 'b68', 'b42']

Section: The Mathematical Framework of CausalPFN
Our primary estimands of interest are the CEPOs from (1). As shown in (2) and (3), CEPOs directly enable estimation of both ATE and CATEs. Therefore, we focus on developing an estimator that can accurately infer these quantities from the observational data. Specifically, we follow the Bayesian paradigm for causal inference, as introduced in Section 2, and parameterize CEPOs as µ t (x ; ψ). Given a suitably rich prior distribution π over the DGPs, which we will explicitly design in Section 4, we define our target as the posterior predictive distribution of CEPOs: Definition 2 (CEPO-PPD). For each t ∈ T and covariate vector x, the CEPO-PPD is
π µt (• | x, D obs ) := B → I(µ t (x ; ψ) ∈ B) π(ψ | D obs ) dψ , B ∈ B.(7)
Consistent Estimation of CEPOs. The CEPO-PPD captures the epistemic uncertainty about the CEPO encoded in the posterior. A concentrated distribution π µt indicates that the observations D obs are informative and enough samples are available to accurately pin down the true CEPO, whereas a high-variance distribution implies that the data is insufficient for estimation. With that in mind, we now study under which conditions increasing the size of the observations D obs allows us to accurately recover the true CEPO from the CEPO-PPD. This is given through the following informal result (re-stated and proven formally in Appendix B) which provides necessary and sufficient conditions on the prior π under which the CEPO-PPDs enable consistent estimation of the CEPOs:
Proposition 1 (Informal). Under mild regularity assumptions, for almost all ψ ⋆ ∼ π and any set of i.i.d. samples D obs ∼ P ψ ⋆ obs , we have that as |D obs | → ∞, E µ∼π µ t (•|x,D obs ) [µ] a.s.
-→ µ t (x ; ψ ⋆ ), ∀t ∈ T , and almost all x ∈ X ,
if and only if the prior π is CEPO-identifiable, that is for almost all ψ ∼ π, the CEPOs µ t (• ; ψ) only depend on the observational distribution P ψ obs (Definition 1).
this section cite: []

Section: (Proof sketch)
We group all DGPs ψ that share the same observational distribution P ψ obs into an equivalence class and induce a prior obtained from π on the resulting quotient space. By Doob's theorem [26]-a classical result from Bayesian consistency theory-the posterior on this new prior almost surely concentrates on the true equivalence class once asymptotically many observations are given. Consequently, for any functional of the observations that is constant within each equivalence class, its posterior predictive converges almost surely to its true value. Importantly, the causal functional of interest, µ t , can be written as a functional of the observations if and only if the corresponding DGP has identifiable CEPOs. Thus, identifiability is both necessary and sufficient to ensure that µ t is constant throughout the equivalence class, and for the consistency result to hold.
(Remark 1) While the algorithms in our paper use strong ignorability, Proposition 1 itself is an entirely general result and can be extended to DGPs that are not necessarily ignorable, but whose CEPOs satisfy identifiability in Definition 1. Importantly for our practical setting, when the prior π enforces strong ignorability, Proposition 1 suggests that the CEPO-PPDs consistently recover the true CEPO.
(Remark 2) Proposition 1 highlights two key design principles for the prior π: (i) π must rule out non-identifiable cases, and, once identifiability is secured, (ii) broadening π increases the chance that a particular ψ ⋆ lies within its support, thus enabling consistent recovery of the true CEPO for that ψ ⋆ .
this section cite: ['b25']

Section: Learning the CEPO-PPD.
Having shown that CEPO-PPDs are useful for estimating the true CEPOs, we now describe how to learn them. Inspired by PFNs, we train a single transformer q θ to approximate the full predictive distribution π µt . To fit this model, we introduce the following loss: Definition 3 (Causal Data-Prior Loss). For any t ∈ T , we define the causal data-prior loss as
L t (θ) := E ψ∼π, D obs ∪{x} ∼ P ψ obs [-log q θ (µ t (x ; ψ) | x, t, D obs )].(9)
In Appendix C, we show that minimizing L t (θ) also minimizes the KL-divergence between the true CEPO-PPD and q θ , leading to q θ (• | x, t, D obs ) ≈ π µt (• | x, D obs ) for all t ∈ T . This entire training
… Prior π P ψi (X, T, {Y t } t∈T , Y ) Sample Query (x, t) Simulate CEPO: µ t (x; ψ i ) log q θ (• | x, t, D obs ) … Implicit Posterior Transformer P ψi P ψ1 P ψ2 𝐗 𝐓 𝐘 Sampled DGP T X Y Simulate D obs process shifts the computational burden from inference to pre-training: rather than evaluating the posterior π(ψ | D obs ) at test time, the model learns to map observational data directly to the corresponding predictive distribution. When the model is well-fitted, the prior satisfies the assumptions of Proposition 1, and D obs is sufficiently large, the predicted q θ accurately pins down the true CEPO.
Figure 3 visually illustrates optimizing the causal data-prior loss using stochastic gradient descent: at each iteration, we sample a DGP ψ i ∼ π, generate an observational dataset D obs from this DGP, and select a query point (x, t). We compute (simulate) the ground-truth CEPO µ t (x ; ψ i ) and feed both the observational data and query to the model. The model outputs a CEPO-PPD, and we update θ using gradient descent to increase the probability assigned to the true CEPO value. Through training, θ minimizes the data-prior loss and implicitly learns to perform posterior predictive inference, and estimate the predictive distribution π µt , without ever explicitly computing the posterior.
this section cite: []

Section: Point & Distributional Estimation of Causal Effects.
Given observational data D obs from an underlying ψ ⋆ , a natural point estimate for CEPOs is the mean of the predicted CEPO-PPD,
E µ∼q θ (•|x,t,D obs ) [µ] ≈ µ t (x ; ψ ⋆
). These CEPO estimates can also form point estimates for CATEs using (3), and for ATEs using (2) by empirical averaging across units in D obs .  While Section 3 presents the framework in general form (arbitrary finite T and identifiability), for implementation we focus on binary treatments T = {0, 1} under strong ignorability. These assumptions reflect the most common settings encountered by practitioners and serve as a natural starting point. Extending the implementation and algorithms to more general settings is left for future work.
this section cite: []

Section: A Scalable Prior.
Here, we focus on designing an appropriate prior π over DGPs that satisfies the theoretical requirements established in Proposition 1. This prior must balance two factors: First, it should contain a rich set of DGPs with sufficient coverage to approximate real-world scenarios-similar to the priors used in successful tabular predictive models like TabPFN [42,43], TabDPT [69], and TabICL [92]. Second, and uniquely for causal inference, all DGPs in our prior must satisfy strong ignorability which directly implies identifiability of the prior. Moreover, the generated DGPs must allow us to access the ground-truth CEPOs, as required by the causal data-prior loss in Definition 3 for training.
To address these requirements, we develop a procedure that can transform any base table from standard tabular priors into a valid causal dataset, illustrated by Figure 4: (i) retrieve a base table with N rows from either a large library of tabular datafoot_0 or synthesize it (details in Appendix D.1);
(ii) randomly select columns with a varying number of covariates as X; (iii) pick two other columns, relabel them as µ 0 (X), µ 1 (X); (iv) optionally add zero-mean noise to µ 0 (X) and µ 1 (X) to obtain Y 0 and Y 1 , or simply set Y 0 = µ 0 (X) and Y 1 = µ 1 (X); these four steps simulate samples from the joint distribution (X, Y 0 , Y 1 ); (v) generate a random function f , leveraging similar synthetic functions as in Hollmann et al. [42] to map covariates to their treatment logits; (vi) sample binary treatments T ∼ Bernoulli(Sigmoid(f (X))); (vii) finally, form the observed outcomes Y := Y T .
The procedure above "simulates" a collection
{t (n) , x (n) , µ (n) 0 , µ (n) 1 , y (n) } N
n=1 from an underlying DGP that can be used to sample the observational data and obtain CEPOs necessary for training (recall Figure 3). This approach guarantees strong ignorability by design: since treatment T is determined solely from X, it is conditionally independent from the potential outcomes Y 0 , Y 1 . Additionally, by applying the sigmoid function, we ensure 0 < P (T = 1 | X) < 1, satisfying positivity. While this procedure primarily targets binary treatments, it can naturally extend to finite discrete treatments.
For the diversity aspect of π, we rely on the empirical success of existing tabular foundation models and the deliberate design in our generation process. Sampling covariates directly from a mix of real and synthetic tables yields data that is more likely to reflect the scenarios the model will face at inference. We assume no distributional assumptions on covariates and potential outcomes. Appendix D.1 details additional mechanisms for controlling treatment effect heterogeneity and positivity in our synthetic DGPs, as well as the detailed configurations of the prior-generation process.
this section cite: ['b41', 'b42', 'b68', 'b91', 'b41']

Section: Model Architecture & Parallel Training.
We model q θ using a PFN-style transformer encoder that receives a sequence of row tokens as context (i.e., D obs ), where each token embeds a triplet (t (n) , x (n) , y (n) ). At every iteration, we embed B Q batched query tokens (t, x). We then apply 20 layers of self-attention and MLP layers, followed by a final projection layer to get q θ (• | x, t, D obs ) for all the (t, x) pairs in the batched query. The transformer uses the asymmetric masking used in PFNs: both context and query tokens attend only to the context tokens, ensuring that the predicted CEPO-PPDs are mutually independent.
To model each CEPO-PPD, we approximate it with a quantized histogram. We discretize the outcome axis into L = 1024 bins and let the network project the query tokens into L logits. We then apply SoftMax to turn the logits into a quantized distribution q θ (• | x, t, D obs )[ℓ], ∀ℓ ∈ [L]. At each round of gradient update, we place a Gaussian with a small σ at the true CEPO µ t (x) and integrate it over bins to obtain Gaussian quantized probabilities N (µ t (x), σ 2 )[ℓ] and minimize the histogram loss:
HL[µ t (x) ∥ q θ ] = - L ℓ=1 N (µ t (x), σ 2 )[ℓ] • log q θ [ℓ]. (10
)
This loss is an approximation to the causal data-prior loss in (9); it coincides in the limit σ → 0 and L → ∞. The histogram loss formulation affords a tractable proxy for the continuous CEPO-PPD. A more detailed overview of the architecture and procedures for point and interval estimates is illustrated in Figure 5; further details (e.g., parameter counts, compute, inference-time techniques, number of prior datasets, scalability, and speed) are available in Appendices D.2, D.3, and D.4.
this section cite: []

Section: Experiments
Baseline Causal Effect Estimators. We compare to a broad suite of baselines. This includes double machine learning (DML) [16,7,29], doubly robust learner (DR-Learner) [57,52], as well as the T-, S-, X-, and domain adaptation learner (DA-Learner), all part of the EconML package [11]. Moreover, we include deep-learning-based methods such as TarNet [104], DragonNet [106], and RA-Net [20], implemented via the CATENets library [19]. Finally, we compare to inverse propensity weighting (IPW) [97], Bayesian regression trees (BART) [40,15], and generalized random forests (GRF) [7].
All the baselines, except for IPW, provide both CATE and ATE estimates.
Importantly, we tune most of the baselines with cross-validation via grid search. The set of hyperparameter, along with the results with default hyperparameters are all detailed in Appendix D.5.
this section cite: ['b15', 'b6', 'b28', 'b56', 'b51', 'b10', 'b103', 'b105', 'b19', 'b18', 'b96', 'b39', 'b14', 'b6']

Section: Benchmarks with Ground-Truth Effects.
A handful of benchmarks provide ground-truth causal effects, allowing us to directly measure estimation errors. Given a dataset of N units with covariates
(n) , τ (x (n) ))} N
n=1 , and a ground-truth ATE λ, we evaluate models using the relative ATE error and the precision in estimation of heterogeneous effects (PEHE) [40]:
RelativeError( λ) = | λ -λ| |λ| , PEHE(τ ) = 1 N N n=1 τ (x (n) ) -τ (x (n) ) 2 .(11)
Here, τ and λ denote the estimated CATE and ATE, respectively. Table 1 compares CausalPFN to all baselines on four standard set of datasets: 100 realizations of IHDP [94,40], 10 realizations of ACIC 2016 [27], and the Lalonde CPS and Lalonde PSID cohorts [58] with their causal effects provided by RealCause (each with 100 realizations) [81]. Our model demonstrates superior performance on both CATE and ATE tasks, remaining within the top models across most benchmarks. To assess the overall performance of each method for CATE estimation, we calculate the average rank of each method across all 310 realizations based on PEHE. For ATEs, we calculate the average rank of each method based on relative errors. CausalPFN outperforms all baselines in terms of average CATE rank, while being competitive for average ATE rank. Notably, our model is trained entirely on simulated data and never sees the evaluation data during pre-training. While some baseline estimators in Table 1 perform well on specific datasets, they underperform on others. In contrast, the consistent performance of CausalPFN suggests that amortized approaches can potentially eliminate the manual burden of task-specific estimator design.
this section cite: ['b39', 'b93', 'b39', 'b26', 'b57', 'b80']

Section: Policy Evaluation on Marketing Randomized Trials.
Ground-truth CATEs are only available for synthetic or semi-synthetic datasets. However, if a randomized controlled trial (RCT) is available, we 0.00 0.25 0.50 0.75 1.00
Treated Fraction (q)
0.0 0.5 1.0 Qini Curve (Q(q)) Hill (1) Hill (2) CausalPFN X Learner Random Figure 6: Hill (1) & Hill (2) Qini curves.
Table 2: Normalized Qini scores (↑ better). All datasets use 50k stratified subsamples, except Hill (1) and Hill (2) , which use the full 64k rows. Columns are normalized to 1.0 for the best model.
this section cite: []

Section: Method
Hill (1) Hill
(2) Criteo X5 Lenta Mega Avg. CausalPFN 0.992 0.968 0.859 0.922 1.000 0.970 0.952 X Learner 0.975 0.980 1.000 0.937 0.771 1.000 0.944 S Learner 1.000 1.000 0.881 1.000 0.651 0.941 0.912 DA Learner 0.985 0.964 0.626 0.929 0.781 0.998 0.881 T Learner 0.991 0.972 0.701 0.964 0.644 0.986 0.876
can still evaluate the quality of a CATE estimator by assessing the performance of policies derived from it. A common tool for evaluating such policies is the Qini curve [93], which plots the cumulative treatment effect when units are ranked in descending order of their predicted CATE.
Formally, let (y (n) , t (n) ) N n=1 denote outcomes and binary treatments from an RCT, and let τ n be the corresponding CATE estimates, ordered so that
τ 1 ≥ • • • ≥ τ N . Define λ(q) := ⌊qN ⌋ n=1 t (n) y (n) r(q) -(1-t (n) )y (n) 1-r(q) , Q(q) := q • λ(q)/λ(1), 0 ≤ q ≤ 1,(12)
where r(q
) = 1 ⌊qN ⌋ ⌊qN ⌋ n t (n)
is the empirical treatment rate for the first q-quantile of units. Because the data comes from an RCT, λ(q) unbiasedly estimates the ATE for the top q-quantile of units ranked by predicted CATEs. Plotting Q(q) against the treated fraction q yields the (normalized) Qini curve, and the area under this curve is called the Qini score. A random ranking produces a baseline curve as a straight line from (0, 0) to (1, 1). The higher the Qini curve lies above this line, the better the model prioritizes high-impact units with larger CATE values, leading to greater lift and policy benefit.
We benchmark CausalPFN on five large marketing RCTs from the scikit-uplift library [74]. The first dataset, Hillstrom [41], includes 64,000 customers randomly assigned to one of three treatments: no e-mail, an e-mail advertising men's merchandise, or an e-mail advertising women's merchandise. The outcome is whether a website visit occurred within two weeks (binary). We consider two causal tasks: Hill (1) -Men's-merchandise e-mail (treatment) vs. no e-mail (control), and Hill (2) -Women's-merchandise e-mail vs. no e-mail. We estimate CATEs using CausalPFN (five-fold honest splitting) and X Learner. Figure 6 shows Qini curves where CausalPFN closely matches X Learner across the targeting range. Notably, Hill (2) shows much greater gains, suggesting focusing on women's-merchandise ad campaigns, compared to men's, can drive more gains in the number of website visits. We also evaluate CausalPFN on four larger campaigns-Lenta, Retail Hero (X5), Megafon (Mega), and Criteo [61, 95, 78, 122]-each with ∼10 6 rows. For tractability, we compute Qini scores on stratified 50k subsamples; Table 2 shows CausalPFN achieves the best mean performance. However, when we run it on full tables (see Table 7 of Appendix D.6), we observe a drop in performance, which aligns with known context-length limitations of PFN-style transformers on large tables [109]. Still, the strong subsample results highlight the potential of scaling CausalPFN to longer contexts, which remains an important future direction.
this section cite: ['b92', 'b73', 'b40', 'b108']

Section: Uncertainty & Calibration.
Recall from Section 3 that for each unit covariate x, CausalPFN can produce both point estimates and credible intervals for the CATE and CEPOs. We do so by drawing 10,000 samples from the quantized distributions q θ (• | x, t, D obs ) and construct credible intervals at any desired significance level α. Here, we evaluate these intervals, focusing on the model's calibration. We also assess a key assumption from Proposition 1-whether the inference-time DGP ψ ⋆ lies within the support of the prior π, and how the model behaves when this assumption is violated.
We define families of synthetic DGPs to simulate both in-distribution and out-of-distribution (OOD) scenarios. Each DGP samples covariates x from a uniform distribution, defines a treatment logit function f and CEPO functions µ t for t ∈ {0, 1}, assigns treatment via T ∼ Bernoulli(Sigmoid(f (x))), and generates potential outcomes as y t = µ t (x) + ϵ t , where ϵ t is drawn from a standard Uniform, Gaussian, or Laplace. We consider two DGP families; Sinusoidal, where f and µ t are functions with sinusoidal components, and Polynomial, where the functions f and µ t are polynomials of varying degree (see Appendix D.7 for detailed configurations). CausalPFN is trained either on the same family it is tested on, or on a different one (OOD). For a unit with covariates x and significance level α, we say the true CATE is covered if τ (x) lies within the predicted 100(1 -α)% interval obtained using samples from q θ . Plotting Bayesian coverage against nominal levels of α yields the CATE calibration curve. As shown in Figure 7 (left), CausalPFN is reliably calibrated under in-distribution settings but becomes severely overconfident when evaluated on OOD DGPs (ψ ⋆ ̸ ∼ π). This aligns with prior observations that neural models often exhibit pathological overconfidence under distribution shift [35,86].
To correct this, we apply a temperature parameter θ T to the SoftMax that outputs the quantized CEPO-PPD from the logits of the model. We aim to tune θ T to minimize the calibration error. However, direct CATE calibration is impossible because τ (x) is never observed at test-time. Instead, we introduce the regression calibration based on observational data: an observed triple (t, x, y) is covered by the predicted credible interval when y lies inside the model's predicted interval for the CEPO-PPD µ t (x ; ψ ⋆ ). With that in mind, we let cov µ (α) and cov τ (α) denote the Bayesian coverage at level α for the regression and CATE calibration curves, respectively, and define
ICE µ := 1 0 ( cov µ (α) -α) dα,and
ICE τ := 1 0 ( cov τ (α) -α) dα,(13)
as the integrated coverage error (ICE) for regression and CATE (negative values = overconfidence).
Note that we do not expect cov µ to be calibrated: regression intervals combine epistemic uncertainty of the CEPO with the irreducible (aleatoric) noise in Y , so ICE µ is biased. Still, it holds a useful signal. Across all model-DGP pairs in Figure 7 (middle), we consistently observe ICE µ ≤ ICE τ : the regression curve sits at or below the CATE curve. While ICE τ is inaccessible without having the true CATE, ICE µ is computable from observational data. Consequently, temperature-scaling the logits to lift cov µ to the diagonal also calibrates the CATE intervals or makes them conservative. We thus tune θ T by grid search to drive ICE µ to zero using a 5-fold calibration on the observational data. The calibrated curves in Figure 7 (right) confirm that, after temperature scaling, CausalPFN's overconfidence on the OOD test-sets disappears. Additional synthetic train-/test-DGP pairs and real-world data experiments appear in Appendix D.7.
Comparison to TabPFN. We also compare against the latest version of TabPFN [43], plugging its regression output as a proxy for CEPO. As Table 3 shows, TabPFN is surprisingly competitive without any causal tuning, yet CausalPFN outperforms it on every benchmark except ACIC 2016. To isolate the benefit of training on a causal prior, compared to the predictive non-identifiable prior in TabPFN, we fine-tune it on our prior for 48 hours on an H100 GPU. This causal fine-tuning boosts the performance and confirms the added value of identifiable priors for causal effect estimation.
this section cite: ['b34', 'b85', 'b42']

Section: Related Work
Single-Dataset Estimators. Common methods for causal effect estimation are trained and applied on a single dataset. Representative examples include the X-, S-, DR-, and RA-Learners, as well as IPW and DML [11]. Alongside these approaches, several neural variants such as TARNet [104], DragonNet [106], CEVAE [68], and NCMs [114,115] have been proposed; however, all of them still require per-dataset training and do not amortize across various datasets. Amortized Causal Inference. Amortized methods train a single network that maps observational data to causal quantities across multiple DGPs. Existing approaches fall into two groups: (i) methods that first recover a causal graph from observational data and then compute interventions on that graph [102,72], following ideas from causal discovery [88,121,53,67,51,50]; and (ii) methods that infer causal effects end-to-end [83,120,14]. Amortization has also been explored in decision-making, where the aim is to learn policies that generalize across environments or tasks [60,59]. While closely related, none of these methods provides a ready-to-use estimator that consistently surpasses specialized single-dataset estimators on standard benchmarks. In contrast, our method is trained once and produces causal effects without any access to or adaptation on the test-time DGPs. Through large-scale training, CausalPFN delivers out-of-the-box performance that exceeds specialized singledataset estimators. Recently, concurrent work by Robertson et al. [96] also applies PFNs to causal effect estimation but lacks a procedure to guarantee the identifiability of the prior data; additionally, we observe relatively poor empirical performance compared to CausalPFN. For further discussion and comparison with this method, refer to Appendix E.
this section cite: ['b10', 'b103', 'b105', 'b67', 'b113', 'b114', 'b101', 'b71', 'b87', 'b120', 'b52', 'b66', 'b50', 'b49', 'b82', 'b119', 'b13', 'b59', 'b58', 'b95']

Section: Scaling In-Context Transformers.
In-context learning with transformers has shown impressive results across a range of domains [13,116,18,25,110]. Although the underlying mechanisms responsible for this success remain an active area of research [1,23,85,111,63,117,112,8,90], increasing model size and training data have consistently and undoubtedly led to stronger performance. This success has recently extended to tabular prediction with models such as TabPFN [42,43], TabDPT [69], and TabICL [92], which are trained on broad prior distributions and perform well on real-world data without fine-tuning. CausalPFN complements these works, demonstrating that-with sufficient scale and training-in-context learning can also be effectively adapted to causal inference.
this section cite: ['b12', 'b115', 'b17', 'b24', 'b109', 'b0', 'b22', 'b84', 'b110', 'b62', 'b116', 'b111', 'b7', 'b89', 'b41', 'b42', 'b68', 'b91']

Section: Conclusions, Limitations, and Future Work
In this paper, we introduced a practical paradigm for amortized causal effect estimation that combines Bayesian causal inference with large-scale tabular training. Despite learning solely from simulated data, CausalPFN matches, and often outperforms, specialized causal estimators across diverse realworld domains. Through amortization, we significantly reduce the burden of estimator selection at inference time, and to foster adoption, we have open-sourced the code and presets.
That said, several important limitations remain: (i) Our approach fundamentally assumes strong ignorability, which is an untestable assumption in practice. Without this condition, CausalPFN has no guarantees of validity. Domain expertise still remains essential to determine whether this method is appropriate or whether alternative approaches should be used. (ii) Our theoretical guarantees rely on idealistic assumptions: a well-specified prior and asymptotically large datasets. We lack finite-sample theory characterizing the estimator's behavior in practical settings. Investigating robustness to prior misspecification and developing finite-sample guarantees remain open problems. Recent work on theory of valid adjustment sets [17] may offer promising directions for addressing these challenges.
(iii) Performance degradation is evident on the largest marketing tables (Table 7), reflective of the known size-scalability trade-off inherent to PFN-style models [43]. (iv) While CausalPFN already supports multi-arm discrete treatments with a finite set T , we have only implemented it for the binary T . Additionally, extending to the continuous treatment setting where T is not finite remains fully unexplored. (v) Finally, our entire implementation relies on the strong ignorability or backdoor assumption. Extending our framework to richer domain-informed priors like instrumental variables can broaden the framework's reach, although designing scalable priors for such cases is non-trivial.
this section cite: ['b16', 'b42']

Section: References
Ref_id:b0 Title: What learning algorithm is in-context learning? investigations with linear models Year: (2023)
Ref_id:b1 Title: Validating causal inference models via influence functions Year: (2019)
Ref_id:b2 Title: Bayesian inference of individualized treatment effects using multi-task gaussian processes Year: (2017)
Ref_id:b3 Title: An introduction to MCMC for machine learning Year: (2003)
Ref_id:b4 Title: Mastering 'Metrics: The path from cause to effect Year: (2014)
Ref_id:b5 Title: Identification of causal effects using instrumental variables Year: (1996)
Ref_id:b6 Title: Generalized random forests Year: (2019)
Ref_id:b7 Title: Transformers as statisticians: Provable in-context learning with in-context algorithm selection Year: (2023)
Ref_id:b8 Title: Sequential decision making with expert demonstrations under unobserved heterogeneity Year: (2024)
Ref_id:b9 Title: Bounds on treatment effects from studies with imperfect compliance Year: (1997)
Ref_id:b10 Title: EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation Year: (2019)
Ref_id:b11 Title: OpenML benchmarking suites Year: (2021)
Ref_id:b12 Title: Language models are few-shot learners Year: (2020)
Ref_id:b13 Title: Black box causal inference: Effect estimation via meta prediction Year: (2025)
Ref_id:b14 Title: Data distributional properties drive emergent in-context learning in transformers Year: (2022)
Ref_id:b15 Title: Double/debiased machine learning for treatment and structural parameters Year: (2018)
Ref_id:b16 Title: Probably approximately correct high-dimensional causal effect estimation given a valid adjustment set Year: (2025)
Ref_id:b17 Title: Meta-in-context learning in large language models Year: (2023)
Ref_id:b18 Title: CATENets: Sklearn-style Implementations of Neural Network-based Conditional Average Treatment Effect (CATE) Estimators Year: (2021)
Ref_id:b19 Title: On inductive biases for heterogeneous treatment effect estimation Year: (2021)
Ref_id:b20 Title: Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms Year: (2021)
Ref_id:b21 Title: Really Doing Great at Estimating CATE? A Critical Look at ML Benchmarking Practices in Treatment Effect Estimation Year: (2021)
Ref_id:b22 Title: Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers Year: ()
Ref_id:b23 Title: The road less scheduled Year: (2024)
Ref_id:b24 Title: A survey on in-context learning Year: (2024)
Ref_id:b25 Title: Application of the theory of martingales Year: (1949)
Ref_id:b26 Title: Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition Year: (2019)
Ref_id:b27 Title: OpenML-CTR23 -A curated tabular regression benchmarking suite Year: ()
Ref_id:b28 Title: Orthogonal statistical learning Year: (2023)
Ref_id:b29 Title: Doubly robust estimation of causal effects Year: (2011)
Ref_id:b30 Title: Conditional neural processes Year: (2018)
Ref_id:b31 Title:  Year: (2018)
Ref_id:b32 Title: AMLB: An AutoML benchmark Year: (2024)
Ref_id:b33 Title: Why do tree-based models still outperform deep learning on typical tabular data? Year: (2022)
Ref_id:b34 Title: On calibration of modern neural networks Year: (2017)
Ref_id:b35 Title: Bayesian regression tree models for causal inference: Regularization, confounding, and heterogeneous effects (with discussion) Year: (2020)
Ref_id:b36 Title: Drift-resilient TabPFN: In-context learning temporal distribution shifts on tabular data Year: (2024)
Ref_id:b37 Title: Querykey normalization for transformers Year: (2020-11)
Ref_id:b38 Title: Causal Inference: What If Year: (2023)
Ref_id:b39 Title: Bayesian nonparametric modeling for causal inference Year: (2011)
Ref_id:b40 Title: Minethatdata e-mail analytics and data mining challenge dataset Year: (2008)
Ref_id:b41 Title: TabPFN: A transformer that solves small tabular classification problems in a second Year: (2023)
Ref_id:b42 Title: Accurate predictions on small data with a tabular foundation model Year: (2025)
Ref_id:b43 Title: Improving regression performance with distributional losses Year: (2018)
Ref_id:b44 Title: Identification and estimation of local average treatment effects Year: (1994)
Ref_id:b45 Title: Bayesian inference for causal effects in randomized experiments with noncompliance Year: (1997)
Ref_id:b46 Title:  Year: (2015)
Ref_id:b47 Title: Identifying causal-effect inference failure with uncertainty-aware models Year: (2020)
Ref_id:b48 Title: An introduction to variational methods for graphical models Year: (1999)
Ref_id:b49 Title: Order-based structure learning with normalizing flows Year: (2023)
Ref_id:b50 Title: Learning to induce causal structure Year: (2023)
Ref_id:b51 Title: Towards optimal doubly robust estimation of heterogeneous causal effects Year: (2023)
Ref_id:b52 Title: Causal autoregressive flows Year: (2021)
Ref_id:b53 Title: Attentive neural processes Year: (2019)
Ref_id:b54 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b55 Title: Heterogeneous treatment effect with trained kernels of the nadaraya-watson regression Year: (2023)
Ref_id:b56 Title: Metalearners for estimating heterogeneous treatment effects using machine learning Year: (2019)
Ref_id:b57 Title: Evaluating the econometric evaluations of training programs with experimental data Year: (1986)
Ref_id:b58 Title: Personalized adaptation via in-context preference learning Year: (2024)
Ref_id:b59 Title: Supervised pretraining can learn in-context reinforcement learning Year: (2023)
Ref_id:b60 Title: Lenta uplift dataset Year: (2020)
Ref_id:b61 Title: Bayesian causal inference: a critical review Year: (2023)
Ref_id:b62 Title: Transformers as algorithms: Generalization and stability in in-context learning Year: (2023)
Ref_id:b63 Title: The how and why of bayesian nonparametric causal inference Year: (2023)
Ref_id:b64 Title: DAG-aware transformer for causal effect estimation Year: (2024)
Ref_id:b65 Title: TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems Year: (2025)
Ref_id:b66 Title: Amortized inference for causal structure learning Year: (2022)
Ref_id:b67 Title: Causal effect inference with deep latent-variable models Year: (2017)
Ref_id:b68 Title: TabDPT: Scaling Tabular Foundation Models on Real Data Year: (2025)
Ref_id:b69 Title: DiffPO: A causal diffusion model for learning distributions of potential outcomes Year: (2024)
Ref_id:b70 Title: Mediation analysis Year: (2007)
Ref_id:b71 Title: Zero-shot learning of causal models Year: (2024)
Ref_id:b72 Title: Empirical analysis of model selection for heterogeneous causal effect estimation Year: (2024)
Ref_id:b73 Title: User guide for uplift modeling and casual inference Year: (2020)
Ref_id:b74 Title: Identification problems in the social sciences Year: (1993)
Ref_id:b75 Title: What exactly has TabPFN learned to do? Year: (2024)
Ref_id:b76 Title: When do neural nets outperform boosted trees on tabular data? Year: (2023)
Ref_id:b77 Title: Megafon uplift dataset Year: (2020)
Ref_id:b78 Title: A detailed treatment of Doob's theorem Year: (2018)
Ref_id:b79 Title: Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers Can Do Bayesian Inference Year: (2022)
Ref_id:b80 Title: RealCause: Realistic causal inference benchmarking Year: (2020)
Ref_id:b81 Title: Bayesian learning for neural networks Year: (2012)
Ref_id:b82 Title: Zero-shot causal learning Year: (2023)
Ref_id:b83 Title: A practical introduction to bayesian estimation of causal effects: Parametric and nonparametric approaches Year: (2021)
Ref_id:b84 Title:  Year: (2022)
Ref_id:b85 Title: Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift Year: (2019)
Ref_id:b86 Title: Causality Year: (2009)
Ref_id:b87 Title: Causal discovery with continuous additive noise models Year: (2014)
Ref_id:b88 Title: Elements of causal inference: Foundations and learning algorithms Year: (2017)
Ref_id:b89 Title: Meta-statistical learning: Supervised learning of statistical inference Year: (2025)
Ref_id:b90 Title: A note on standard Borel and related spaces Year: (2009)
Ref_id:b91 Title: TabICL: A Tabular Foundation Model for In-Context Learning on Large Data Year: (2025)
Ref_id:b92 Title: Using control groups to target on predicted lift: Building and assessing uplift models Year: (2007)
Ref_id:b93 Title: Infant health and development program for low birth weight, premature infants: Program elements, family participation, and child intelligence Year: (1992)
Ref_id:b94 Title: Retail hero (x5) uplift dataset Year: (2020)
Ref_id:b95 Title: Do-pfn: In-context learning for causal effect estimation Year: (2025)
Ref_id:b96 Title: The central role of the propensity score in observational studies for causal effects Year: (1983)
Ref_id:b97 Title: Estimating causal effects of treatments in randomized and nonrandomized studies Year: (1974)
Ref_id:b98 Title: Bayesian inference for causal effects: The role of randomization Year: (1978)
Ref_id:b99 Title: Causal inference using potential outcomes: Design, modeling, decisions Year: (2005)
Ref_id:b100 Title: Contextual multi-armed bandits for causal marketing Year: (2018)
Ref_id:b101 Title: A fixed-point approach for causal generative modeling Year: (2024)
Ref_id:b102 Title: A comparison of methods for model selection when estimating individual treatment effects Year: (2018)
Ref_id:b103 Title: Estimating individual treatment effect: generalization bounds and algorithms Year: (2017)
Ref_id:b104 Title: GLU variants improve transformer Year: (2020)
Ref_id:b105 Title: Adapting neural networks for the estimation of treatment effects Year: (2019)
Ref_id:b106 Title: Benchmarking framework for performance-evaluation of causal inference analysis Year: (2018)
Ref_id:b107 Title: A course on Borel sets Year: (1998)
Ref_id:b108 Title: Retrieval & fine-tuning for in-context tabular models Year: (2024)
Ref_id:b109 Title: Effortless, simulationefficient bayesian inference using tabular foundation models Year: (2025)
Ref_id:b110 Title: Transformers learn in-context by gradient descent Year: (2023)
Ref_id:b111 Title: Max Vladymyrov, Razvan Pascanu, and João Sacramento. Uncovering mesa-optimization algorithms in transformers Year: (2023)
Ref_id:b112 Title: FLAML: A Fast and Lightweight AutoML Library Year: (2021)
Ref_id:b113 Title: The causal-neural connection: Expressiveness, learnability, and inference Year: (2021)
Ref_id:b114 Title: Neural causal models for counterfactual identification and estimation Year: (2023)
Ref_id:b115 Title: An Explanation of In-context Learning as Implicit Bayesian Inference Year: (2022)
Ref_id:b116 Title: Pretraining data mixtures enable narrow model selection capabilities in transformer models Year: (2023)
Ref_id:b117 Title: A closer look at TabPFN v2: Strength, limitation, and extension Year: (2025)
Ref_id:b118 Title: Real-time bidding for online advertising: measurement and analysis Year: (2013)
Ref_id:b119 Title: Towards causal foundation model: on duality between causal inference and attention Year: (2023)
Ref_id:b120 Title: DAGs with NO TEARS: Continuous Optimization for Structure Learning Year: (2018)
Ref_id:b121 Title: A large scale benchmark for uplift modeling Year: (2018)
