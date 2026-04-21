Title: Federated Generalised Variational Inference: A Robust Probabilistic Federated Learning Framework
Abstract: We introduce FEDGVI, a probabilistic Federated Learning (FL) framework that is robust to both prior and likelihood misspecification. FEDGVI addresses limitations in both frequentist and Bayesian FL by providing unbiased predictions under model misspecification, with calibrated uncertainty quantification. Our approach generalises previous FL approaches, specifically Partitioned Variational Inference (Ashman et al., 2022), by allowing robust and conjugate updates, decreasing computational complexity at the clients. We offer theoretical analysis in terms of fixed-point convergence, optimality of the cavity distribution, and provable robustness to likelihood misspecification. Further, we empirically demonstrate the effectiveness of FEDGVI in terms of improved robustness and predictive performance on multiple synthetic and real world classification data sets.

Section: Introduction
Federated learning (FL) is a framework for the collaborative training of a global model by a collection of clients, without requiring proprietary data to be shared with a central server or other participating clients (McMahan et al., 2017). This decentralised approach allows FL to be used on applications with strict data privacy constraints, such as in finance or healthcare (Kairouz et al., 2021). However, due to the sensitive nature and complexity of these domains, both privacy and robustness to model misspecification are paramount.
The frequentist formulation of FL aims to minimise a global loss function by aggregating local gradients from clients. Early works include Federated Averaging (FEDAVG, McMahan et al., 2017) which iterates between training clients lo-cally and averaging updates on the server. This has sparked a large body of research on issues such as communication efficiency, data privacy, and data heterogeneity across clients (Hamer et al., 2020;Malinovsky et al., 2020;Reddi et al., 2021;Chen et al., 2022;Tenison et al., 2023;Tziotis et al., 2023;Li et al., 2024;Demidovich et al., 2025). There has been some work addressing robustness to adversarial clients (Allouah et al., 2024;Bao et al., 2024) and data and system heterogeneity (Chen et al., 2022;Zhao et al., 2023;Heikkilä et al., 2023). However, these only provide point estimates, and do not allow principled uncertainty quantification, as required in many FL applications (Jonker et al., 2024).
In contrast, Bayesian FL approaches aim to update beliefs of a global model with data partitioned across clients. This largely builds on distributed inference methods such as the Bayesian Committee Machine (Tresp, 2000), parallel MCMC (Ahn et al., 2014;Mesquita et al., 2020), or Divide&Conquer SMC (Chan et al., 2023). Expectation Propagation (Minka, 2001;Vehtari et al., 2020) is naturally applicable to the distributed setting where local sites are iteratively refined. This requires computing the cavity distribution that removes local sites from the current approximation. Partitioned Variational Inference (PVI, Bui et al., 2018;Ashman et al., 2022) takes this idea and proposes a distributed variational inference algorithm, which has been extended through MCMC (Guo et al., 2023) and Stochastic Gradient Langevin Dynamics (SGLD) (Mekkaoui et al., 2021). Whilst these approaches quantify uncertainty, they are susceptible to model misspecification which can lead to inaccurate, overconfident predictions (Bernardo & Smith, 2000;Bissiri et al., 2016;Knoblauch et al., 2022).
Current approaches to FL are inherently non-robust to model misspecification which leads to compromised performance and uncalibrated uncertainty quantification. We address these challenges by departing from the traditional Bayesian paradigm and propose a distributed Generalised Variational Inference framework that allows us to deal with model misspecification. In summary, our contributions are:
• We introduce Federated Generalised Variational Inference (FEDGVI), a family of robust probabilistic algorithms for federated learning.
• We prove that FEDGVI is robust to likelihood misspecification (Theorem 4.12).
• We demonstrate that FEDGVI generalises standard approaches such as PVI and FEDAVG (Remarks 4.1 and 4.2) and theoretically justify the use of the cavity distribution (Theorem 4.9).
• We prove that, under suitable conditions, FEDGVI converges to Generalised Bayesian posteriors (Lemma 4.6 and Proposition 4.10) that are computationally tractable.
• We evaluate FEDGVI on a range of synthetic and realworld datasets, across multiple models, demonstrating improved robustness and predictive performance.
In Section 2 we define model misspecification and recall methods that mitigate it in the non-distributed setting. Section 3 introduces our framework, which builds on these concepts and extends them to the federated setting. We analyse the theoretical properties of FEDGVI in Section 4, including provable robustness. Finally, Section 5 studies the empirical performance and gains of FEDGVI with multiple models and real world datasets such as Bayesian Neural Networks on MNIST and FASHIONMNIST. 1
1.1. Related Work Robust Frequentist Federated Learning In the frequentist setting, building on the seminal paper of McMahan et al. (2017), many approaches have aimed at mitigating challenges in FL, such as robustness to adversarial servers through secure aggregation (Chen et al., 2022), to stragglers (Tziotis et al., 2023), heterogenous data in out-ofdistribution generalisation (Tenison et al., 2023), heterogeneous and asynchronous clients (Fraboni et al., 2023), or finding weaknesses in communications (Zhu et al., 2019;Zhao et al., 2023). More recently, work on robust server aggregations achieves robustness against Byzantine clients that aim to deteriorate model performance (Allouah et al., 2024;Bao et al., 2024). However these do not allow principled uncertainty quantification.
Federated Bayesian Inference Federated and distributed Bayesian methods aim to approximate the posterior as if it had been computed with the data of all clients available at a central server. Early work on distributed Bayesian inference includes Bayesian opinion pools (Genest, 1984;Carvalho et al., 2023), and the Bayesian Committee machine (Tresp, 2000), which aim to find a consensus among a collection of Bayesian beliefs. Works that aim to operationalise this in the distributed setting, where data is split IID across clients, 1 Code to reproduce experiments can be found at https://   github.com/Terje-M/FedGVI.
include Expectation Propagation (Minka, 2001;Opper & Winther, 2005;Hasenclever et al., 2017;Vehtari et al., 2020), and consensus based Monte Carlo (Scott et al., 2016). In the Federated setting this assumption is often violated, as data is not split homogeneously and IID across participating devices. From this perspective, most approaches to Bayesian FL can be categorised into finding an approximate posterior through variational inference (Corinzia et al., 2021;Ashman et al., 2022;Kassab & Simeone, 2022;Heikkilä et al., 2023;Hassan et al., 2024;Vedadi et al., 2024;Swaroop et al., 2025), Markov Chain Monte Carlo (Al-Shedivat et al., 2021;Mekkaoui et al., 2021;Kotelevskii et al., 2022;Guo et al., 2023;Hasan et al., 2024), Gaussian Processes (Achituve et al., 2021), or directly learning a Bayesian neural network (Yurochkin et al., 2019;Zhang et al., 2022). Personalised or hierarchical Bayesian FL (Kotelevskii et al., 2022;Zhang et al., 2022;Kim & Hospedales, 2023;Hassan et al., 2023;2024;Vedadi et al., 2024) allows for additional expressibility of client posteriors, especially under heterogeneity. However, none of these are inherently robust to contamination and model misspecification.
Robust Bayesian Inference Although the existing Bayesian FL methods address some of the challenges of federated learning, such as communication constraints and data heterogeneity, they still aim to approximate the Bayesian posterior, which in itself is a flawed objective under model misspecification (Walker, 2013;Berk, 1966;Bernardo & Smith, 2000). In the global, non-federated case, several methods have been proposed to combat misspecification in the Bayesian setting (Grünwald, 2012), with the most promising direction being Generalised Bayesian Inference (Hooker & Vidyashankar, 2014;Bissiri et al., 2016;Ghosh & Basu, 2016a;Jewson et al., 2018;Miller, 2021;Alquier, 2021;Knoblauch et al., 2022;Matsubara et al., 2022). In this work we capitalise on this front and bring robustness to model misspecification in the federated setting.
this section cite: ['b58', 'b44', 'b32', 'b56', 'b68', 'b20', 'b71', 'b73', 'b55', 'b23', 'b4', 'b11', 'b20', 'b83', 'b43', 'b72', 'b1', 'b60', 'b19', 'b62', 'b75', 'b17', 'b10', 'b31', 'b59', 'b14', 'b15', 'b51', 'b58', 'b20', 'b73', 'b71', 'b25', 'b83', 'b4', 'b11', 'b26', 'b18', 'b72', 'b62', 'b65', 'b34', 'b75', 'b69', 'b22', 'b10', 'b46', 'b36', 'b74', 'b70', 'b2', 'b59', 'b52', 'b31', 'b33', 'b0', 'b79', 'b81', 'b52', 'b81', 'b48', 'b35', 'b74', 'b76', 'b13', 'b14', 'b30', 'b38', 'b15', 'b42', 'b61', 'b5', 'b51']

Section: Preliminaries

this section cite: []

Section: Notation and Model Misspecification
Let (Ω, F, P 0 ) be a probability space where P 0 is the data generating process, generating the observable random variables X 1 , ..., X n ≡ X n 1 taking values in the measurable space (Ξ, X ). Further, let Y n 1 be observable random variables depending on X n 1 respectively, taking values in (Υ, Y). Denote their realisations {X i = x i , Y i = y i } n i=1 , which are assumed to be partitioned across M clients {x m , y m } M m=1 each of size n m . Consider hypothesis measures P θ where θ takes values in (Θ, T ), a measurable space, admitting densities p θ . We study elements of P(Θ), the set of all probability measures on (Θ, T ), starting with prior Π and updated to Q, dominated by some common measure µ, and Algorithm 1 FEDGVI SERVER 1: Input: π(θ), Q, D s 2: Define: ℓ
m (θ) = 0, ℓ
s (θ) = 0, q
s (θ) = π(θ) 3: for t = 1, ..., T do 4: for m = 1, ..., M in parallel do
q B (θ) = π(θ) M m=1 p θ (y m ; x m ) /Z(1)
where Z = Θ M m=1 p θ (y m ; x m ) Π(dθ) is the marginal likelihood. Since we do not suppose that the prior Π, nor the likelihood P θ are well specified, i.e. P 0 / ∈ P(Θ), we are in the M-open setting (Bernardo & Smith, 2000), the model misspecified, and the Bayesian posterior inappropriate.
this section cite: ['b14']

Section: Model Misspecification
There are several different ways we can think about model misspecification under the M-open assumption.
Prior Misspecification The traditional Bayesian paradigm assumes that the prior encodes the best available judgement about θ, which beyond simple settings, is never realised (Berger, 1985;Knoblauch et al., 2018). Such misspecification is common; e.g. it is standard to use zero-mean Gaussian distributions on the weights of Bayesian Neural networks. This can have dire effects, for instance Diaconis & Freedman (1986) demonstrate that multimodal priors in a location model can cause the posterior to not accumulate around P 0 , even when the DGP is well specified, i.e. when P 0 ∈ P(Θ).
this section cite: ['b12', 'b50', 'b24']

Section: Likelihood Misspecification
One such example is where the hypothesis of interest is contaminated , and an ε fraction of the data (input and/or output variables) has some unknown data source. Formalising this we follow the definition of Huber (1964):
Definition 2.1 (Huber contamination). Given an ε ∈ (0, 1 2 ) and the uncontaminated distribution P θ of inliers and some contaminating distribution G of outliers, then P 0 is said to be an ε-corrupted version of P θ ; P 0 := (1 -ε)P θ + εG.
this section cite: ['b39']

Section: Robust Bayesian Methods

this section cite: []

Section: Generalised Bayesian Inference (GBI)
Instead of linking the parameter and data through likelihoods, Bissiri et al. 2016) and Miller (2021) formalised a coherent Bayesian framework using loss functions leading to Gibbs posteriors (Alquier et al., 2016). This was further utilised to deal with likelihood misspecification through robust losses, e.g Knoblauch et al. (2018). Let L : Θ × Ξ × Υ → R be such a loss, then the GBI posterior is given by:
Algorithm 2 FEDGVI CLIENT 1: Input: q (t-1) s (θ), Q, {x m , y m }, L m , ℓ (t-1) m (θ), D 2: Optimise q \m (θ) according to Equation (3) 3: Optimise q (t) m (θ) according to Equation (4) 4: Set ∆ (t) m (θ) according to Equation (5) 5: Set ℓ (t) m (θ) ← ℓ (t-1) m (θ) + ∆ (t) m (θ) 6: return: Communicate ∆ (t) m (θ) to SERVER(
q GBI (θ) = π(θ) exp -β M m=1 L(y m ; θ, x m ) /Z (2)
with Z = Θ exp{-β M m=1 L(y m ; θ, x m )}Π(dθ). Here, β ∈ R >0 is a learning rate parameter that determines how much weight we place on the observed data, similar to power posteriors in VI (Grünwald, 2012;Kallioinen et al., 2024). This recovers q B (θ) when the loss is the negative log-likelihood and β = 1.
this section cite: ['b61', 'b6', 'b50', 'b30', 'b45']

Section: Generalised Variational Inference (GVI)
In Knoblauch et al. (2022) GBI is generalised within a variational framework that explicitly accounts for prior and likelihood misspecification. Let D : P(Θ)×P(Θ) → R + be a divergence then the GVI posteriors are defined as:
q GVI (θ) = arg min q∈Q E q(θ) L(y M 1 ; θ, x M 1 ) + D(q : π)
where Q ⊂ P(Θ), making inference tractable. This allows for targeting a larger subspace of posteriors, and through different divergences the effect of the prior can be controlled.
this section cite: ['b51']

Section: Federated Generalised Variational Inference

this section cite: []

Section: Methodology
In this section, we present the proposed federated learning framework, named FEDGVI, that explicitly addresses likelihood and prior misspecification. We aim to learn a robust approximate posterior q s (θ) using partitioned observations across M clients. FEDGVI iterates consist of two steps: a) sending of the current approximate posterior to each client, which is updated through a robust variational objective, and b) aggregating the updates on the server, resulting in a robust approximate posterior; summarised in Algorithms 1 and 2.
Initialisation We set the initial server posterior as the prior, q
s (θ) = π(θ), and the local and server loss ap-proximations to be zero, ℓ
m (θ) = 0 and ℓ
s (θ) = 0 respectively; m denotes a specific client and s the server.
Until Convergence For t = 1, 2, ..., T , we synchronously compute updates locally at each client, and accumulate these at the server to form the new global posterior q (t) s (θ).
this section cite: []

Section: Client
The client receives the current approximate posterior from the server. This will be used as the prior from which a client can compute an updated posterior using their local data. First, however the information of the client's data must be removed by computing the cavity distribution. The cavity distribution acts as the local prior incorporating all previous information from all other clients and is given by:
q \m (θ) ∝ q (t-1) s (θ) exp{-ℓ (t-1) m (θ)} (3
)
The client then computes a robust local approximate posterior with it's local data set {x m , y m } and it's loss function
L (t)
m (•), which is regularised by the divergence, D, and cavity distribution
q (t) m (θ) = arg min q∈Q E q(θ) L (t) m (y m ; θ, x m ) + D(q : q \m ).
(4) This GVI style objective allows the client to be robust to both likelihood misspecification as well as prior misspecification arising due to the cavity. To update the global posterior at the server, the client computes the negative log ratio of the local and global posteriors. In line with existing Bayesian FL (Ashman et al., 2022;Guo et al., 2023), we use a damping parameter τ m ∈ (0, 1], which is analogous to a learning rate as in frequentist FL, to compute the update:
∆ (t) m (θ) = -τ m log q (t) m (θ) q (t-1) s (θ)(5)
The client stores ℓ
(t) m (θ) := ℓ (t-1) m (θ) + ∆ (t) m (θ) and communicates ∆ (t) m (θ) to the server.
Server The loss at the server is updated based on the received client updates,
ℓ (t) s (θ) = ℓ (t-1) s (θ) + M m=1 ∆ (t) m (θ)(6)
By only incorporating clients' updates that have changed we can trivially allow for batched and asynchronous scheduling of clients. The updated loss is then used to compute the new server posterior though a GVI optimisation procedure:
q (t) s (θ) = arg min q∈Q E q(θ) ℓ (t) s (θ) + D s (q : π) (7)
This posterior and loss are passed back to the clients for further refinement at the next iteration until convergence.
3.1.1. HYPERPARAMETERS Ashman et al. (2022) set the damping parameter to τ ∝ 1 M throughout their experiments. This turns out, see Proposition 4.3, to be a reasonable choice when τ = 1 M in combination with D s = D KL since this causes the posterior at the server to be a logarithmic opinion pool induced by an externally Bayesian pooling operator (Genest et al., 1986), ensuring stable convergence. Other hyperparameters arising from the choice of losses and divergences are dependent on the expected amount of model misspecification.
this section cite: ['b10', 'b31', 'b10', 'b27']

Section: Robustness to Likelihood Misspecification
Within our framework we are free to choose the client side losses. We consider the Density-Power divergence based loss (Ghosh & Basu, 2016b), often referred to as βdivergence loss L β , the γ-divergence based losses (Hung et al., 2018), L γ , as well as a score matching loss, L SM , based on the Hyvärinen divergence (Hyvärinen, 2005;Altamirano et al., 2023). In the classification setting, we consider the generalised cross-entropy loss
L (δ) GCE (y i ; θ, x i ) = (1 -p θ (y = y i ; x i ) δ ) δ(8)
for some δ ∈ (0, 1] (Zhang & Sabuncu, 2018). These losses are robust to misspecification because they have a finite supremum (see Definition 4.11). It is important to highlight that GVI and FEDGVI may underperform when using robust losses in the case of correct likelihood specification; see Knoblauch et al. (2022). We can use a Sequential Monte Carlo sampler to estimate the β or γ hyperparameters in L β and L γ (Yonekura & Sugasawa, 2023) or use cross validation to select optimal parameters (Altamirano et al., 2024).
this section cite: ['b40', 'b41', 'b7', 'b82', 'b51', 'b78', 'b8']

Section: Robustness to Prior Misspecification
We mainly consider the weighted Kullback-Leiber divergence, 1 w D KL , (Kullback & Leibler, 1951
) 1 w D KL (q : π) := 1 w E q(θ) log q(θ) π(θ) ,
and the Alpha-Rényi divergence, D
AR ,
D (α) AR (q : π) := 1 α(α -1) log E π (θ) q(θ) π(θ) α .
As examined in Knoblauch et al. (2022), D
AR allows for different prior regularisation depending on how much we trust the prior by placing different weights on it. In future work it would be simple to explore other divergences such as the f -divergences, D f , (Amari, 2016;Alquier, 2021). Similarly to the losses, we can perform cross validation to select the α parameter, however as demonstrated in the ablation study (Figure 6) FedGVI performs favourably under a range of α (and δ) values.
this section cite: ['b53', 'b51', 'b9', 'b5']

Section: Theoretical Results
We now present a theoretical analysis of FEDGVI. We begin by examining the relationship of FEDGVI with other FL algorithms while recovering some of them as special cases, we study the damping parameter, and examine the convergence behaviour of FEDGVI. Then, we turn our attention on robustness to likelihood misspecification, where we first study FEDGVI as distributed GBI, from which we derive a theorem on the necessity of the cavity distribution. Finally, we derive a result for computationally tractable and conjugate FEDGVI, enabling us to present the main theorem on bias-robustness of FEDGVI.
Since it is an open problem where global GVI posteriors converge to under arbitrary divergences, we often have to restrict ourselves to consider the server divergence to be the Kullback-Leibler divergence. This ensures that the posterior at the server will have the structure of a GBI posterior,
q (T ) s (θ) ∝ exp - M m=1 ℓ (T ) m (θ) π(θ)
where we incorporate prior robustness and tractability through the approximate losses.
this section cite: []

Section: Recovering Existing Methods as a Special Case
By choosing specific divergences, loss functions, and variational families, we can recover existing methods as special cases of our framework, which we summarise in Figure 1: Remark 4.2. When D = D s = 0, and Q = {δ θ (θ) : θ ∈ Θ}, with δ θ being the Dirac-delta measure at some element θ, we recover FEDAVG of McMahan et al. (2017).
FEDGVI L, D, Q, M, D s VI -log p θ , D KL , Q, M = 1, D s = D KL PVI -log p θ , D KL , Q, M, D s = D KL ERM L, D = 0, {δ θ }, M = 1, D s = 0 FEDAVG L, D = 0, {δ θ }, M, D s = 0
this section cite: ['b58']

Section: Damping as a Bayesian Logarithmic Opinion Pool
Choosing the damping parameter to be τ = 1/M results in a logarithmic opinion pool. In fact choosing damping parameters such that all of them sum to unity also forms a valid logarithmic opinion pool (Genest et al., 1986).
Proposition 4.3. Assume D s = D KL , and that m τ m = 1 where τ m ≥ 0 ∀m, then the posterior at the server is an externally Bayesian logarithmic opinion pool of the form
q (t) s (θ) = M m=1 q (t) m (θ) τm Θ M m=1 q (t) m (θ) τm dθ , θ -a.e.
See Appendix B.2 for the proof. This results provides a theoretical justification on the previously heuristic use of the damping parameter (as used in PVI, Ashman et al., 2022). Specifically it ensures that this selection of τ leads to a valid distribution and results in more stable convergence.
this section cite: ['b27', 'b10']

Section: Fixed Points of FEDGVI
In this section we study the properties of FEDGVI posteriors when these converge to some fixed point. Specifically, we generalise the fixed point result of PVI (Ashman et al., 2022, Property 2.3) to arbitrary losses.
Proposition 4.4.
Let D s = D KL , D = 1 w D KL , w > 0, and Q ⊂ P(Θ), then if q * s (θ) = π(θ) exp{-ℓ * s (θ)}/Z q * such that ∀m ∈ [M ], ∆ * m (θ) = 0, then q * s(
θ) is a local minimiser of the following GVI objective:
E q(θ) M m=1 L m (y m ; θ, x m ) + 1 w D KL (q : π) (9
)
Remark 4.5. If the loss in Equation ( 9) is convex, then a fixed point of FEDGVI is a global minimum of GVI.
This illustrates that if FEDGVI converges, then the posterior is a (local) minimiser of the GVI objective. We refer to such distributions as fixed points. This recovers Kassab & Simeone (2022, Theorem 1) (which deals with the restricted case of Q = P(Θ)) with a novel proof; see Appendix B.3.
this section cite: ['b10']

Section: Generalised Bayesian Inference
As a consequence of Proposition 4.4 and Remark 4.1, FEDGVI will recover the GBI posterior when Q = P(Θ).
Lemma 4.6. Assuming Q = P(Θ), D = 1 β D KL with β > 0, D s = D KL , and τ = 1, then FEDGVI will recover the GBI posterior after the first iteration.
q (1) s (θ) = q GBI (θ|{x m , y m } M m=1 ) = exp{-β M m=1 L(y m ; θ, x m )}π(θ)/Z
This posterior is invariant under subsequent iterations of FEDGVI, having reached a fixed point.
Moreover, for a damping rate τ = 1/M , the posterior at the server converges pointwise a.e. in Θ to the GBI posterior,
q (T ) s (θ) T →∞ -→ q GBI (θ|{x m , y m } M m=1 ), θ -a.e.
This result, proven in Appendix B.4, is the first step towards likelihood robustness. If we were able to find the GBI posterior efficiently with some robust loss, then the posterior would be robust and computable. Here however, the loss may not vary over different iterations of FEDGVI as in Equation ( 4) and the normaliser may be intractable.
this section cite: []

Section: The Cavity Distribution is Necessary
By further investigating the relationship of FEDGVI with the GBI posterior, we can extend Lemma 4.6 and derive a Theorem under which we are required to use the cavity distribution to regularise the client update. This is in contrast to both PVI, where it's use is heuristically justified, and to other Bayesian FL approaches where the previous posterior is used instead. For this we recall two natural assumptions that any such distribution must satisfy in a federated setting.
Assumption 4.7. No client can have access to the data set of another client.
Assumption 4.8. Each client generates their update equivalently to other clients.
These assumptions combined with Lemma 4.6 lead us to the necessity of the cavity distribution.
Theorem 4.9. Let the assumptions be as in Lemma 4.6 with τ = 1, and assume that the Assumptions 4.7 and 4.8 are satisfied, then (1.) holds if and only if (2.) holds.
1. FEDGVI recovers the generalised Bayesian posterior q GBI (θ) which is invariant under further FEDGVI updates.
this section cite: []

Section: The cavity regularises the client optimisation problem.
This provides a principled justification for the use of the cavity distribution, as defined in Equation (3), in FEDGVI. We provide the proof in Appendix B.5.
this section cite: []

Section: Conjugate Client Updates
Before we present our main result on provable robustness to likelihood misspecification, we first show that we can find a GBI posterior under specific losses in a computationally tractable manner. Assuming that the data generating process has some exponential family distribution, where y ∼ p θ (y),
p θ (y) = exp{η(θ) ⊤ ϕ(y) -A(η(θ)) + h(y)},
such that this is differentiable in y, by using the weighted score matching loss of Altamirano et al. (2023), L w SM , then client updates, using the weighted KL divergence locally, are available in closed form. If we further assume that our model is Gaussian, or has the form of a squared exponential, and that the natural parameters of the DGP are η(θ) = θ, then the client approximation will have a conjugate form.
Proposition 4.10. Assume that the hypothesis p θ (y) has differentiable, exponential family distribution with η(θ) = θ, L (t) m = L w t m SM , and D = 1 β D KL , and the variational family Q is the multivariate Gaussians, then the local posteriors at the clients are conjugate Gaussians. Moreover, Equation (7) will have closed form if D s has closed form between Gaussian distributions.
See Appendix B.6 for the proof. The loss may now depend on the client and iteration t. Most exponential family distributions satisfy the conditions of the proposition, and there are several divergences that allow closed form expressions between Gaussians, such as the Alpha-Rényi , or the α, β, γ-divergences of Cichocki & Amari (2010). Further, this enables the use of intractable likelihood models.
this section cite: ['b7', 'b21']

Section: Provable Robustness to Outliers
For a robust loss function at the clients, and using the weighted KL divergence at the clients and the KL divergence at the server, guarantees that after T iterations, the posterior computed at the server will also be robust to outliers. This means we can achieve robustness at the server by leveraging the robust losses that were derived for GVI. In this, we mean robustness as defined by Ghosh & Basu (2016a) and further developed in Matsubara et al. (2022). We define the empirical DGP of a client as P nm := 1 nm nm i=1 δ xi , and of the entire data set as P n := 1 n M m=1 n m P nm . When this is contaminated by some ε fraction of data centred at some adversarially chosen data point z ∈ Ξ, the misspecified DGP is defined as P n,ε,z := (1 -ε)P n + εδ z .
Definition 4.11. We say that a loss L (t) m (θ; P nm,ε,z ), w.r.t. some prior distribution π(θ), is robust to outliers, if the following hold:
1. sup z∈Ξ d dε L (t) m (θ; P nm,ε,z ) ε=0 ≤ γ (t) (m) (θ), 2. sup θ∈Θ π(θ)γ (t) (m) (θ) < ∞, and 3. Θ π(θ)γ (t) (m) (θ)µ(dθ) < ∞
These conditions ensure that the influence of arbitrary contamination on the local posterior is not arbitrarily bad. In particular the auxiliary function γ (t) m ensures that the influence of an adversarial data point z on the posterior over infinitesimal contaminations, d dϵ q (t) m (θ; P nm,ϵ,z )| ϵ=0 , are finite over all θ and z. Condition 2 ensures the loss increases slowly enough for the local posterior to concentrate around the data, and condition 3 ensures the resulting posterior will be normalisable.
Theorem 4.12. Let D s = D KL , D = 1 w D KL , Q = P(Θ), further assume that the prior is upper bounded and the loss is lower bounded, then if ∀t ∈ [T ] and ∀m ∈ [M ] L (t) m (θ; P nm,ε,z ) is robust, then the posterior generated by FEDGVI will be robust to outliers.
The proof is in Appendix B.7. This result together with Proposition 4.10 is significant as we have robustness under intractable optimisation, and we can choose a provably robust, conjugate loss to generate robust FEDGVI posteriors, which are then computationally efficient to compute.
this section cite: []

Section: Experiments
We evaluate FEDGVI against several other methods, specifically PVI (Ashman et al., 2022), FEDAVG (McMahan et al., 2017), the nonparametric DSVGD (Kassab & Simeone, 2022), the distributed MCMC based DSGLD (Ahn et al., 2014), federated MCMC based FEDPA (Al-Shedivat et al., 2021), and the one shot BCM based approach β-PREDBAYES (Hasan et al., 2024). We provide further details about experiments in Appendix D. We first examine the effect of misspecified likelihoods through the well known clutter problem (Minka, 2001). We generate 100 observations from a Gaussian location model that is contaminated through Definition 2.1 with ε = 0.25 Gaussian noise. The aim is to infer the location parameter θ of the uncontaminated data. We compare FEDGVI with both L β and L SM vs PVI with and without misspecification. We also provide the corresponding MLE results. See Figure 2. Under misspecification both the MLE and PVI fail to recover the true θ, whereas FEDGVI can easily handle different levels of contamination. To demonstrate robustness to likelihood misspecification as in Theorem 4.12, we consider the influence of a single outlier at one of seven clients on the server posterior. Figure 3 demonstrates that the negative log likelihood is not robust in the federated setting, whereas different robust divergence based losses allow only limited influence of outliers on the posterior. We plot this as the divergence between the posterior, had we observed the outlier value at the true mean, against the posteriors that have the outlier be farther from the true mean, using the Fisher-Rao distance (Nielsen, 2023). We next consider a 2D logistic regression example where we generate 100 linearly separable samples from a Gaussian mixture distribution. We inject outliers generated by a third Gaussian distribution and assign them to one of the classes so that the data is no longer linearly separable. We compare FEDGVI with L (0.7) β and D
this section cite: ['b10', 'b46', 'b1', 'b2', 'b33', 'b62']

Section: 1D Clutter Problem

this section cite: []

Section: Influence Function

this section cite: []

Section: 2D Misspecified Logistic Regression
(1.5)
AR against PVI, both with 5 clients. Again, the target is given by PVI only trained on the uncontaminated data. As expected PVI is severely impacted by outliers, whereas FEDGVI is robust to them and closely recovers the target posterior. In this experiment we follow the experimental setup of Kassab & Simeone (2022) and average accuracy over 10 random 80/20 train-test splits, where the training data is split homogeneously across 2 clients. We do not add any label contamination. The results are plotted in Figure 5. The non-robust methods all eventually achieve similar accuracy, however FEDGVI is able to outperform all competing methods, which we argue is due to FEDGVI putting less weight on data points that are less likely to belong to the class.
this section cite: ['b46']

Section: Bayesian Neural Networks on MNIST and FASHIONMNIST
Table 1: Classification accuracy (highest in bold) on uncontaminated test data after training on 10% contaminated MNIST data. We report the best performance across all server iterations. MODEL ACCURACY + STD. 10 CLIENTS 3 CLIENTS FEDAVG 96.64± 0.07 96.34 ± 0.20 FEDPA 94.25± 0.39 95.31± 0.35 β-PREDBAYES 94.90± 0.08 96.73± 0.08 PVI 95.56± 0.18 96.68± 0.07 FEDGVI DAR 96.36± 0.09 97.13 ± 0.13 FEDGVI LGCE 97.06± 0.03 98.04 ± 0.07 FEDGVI DAR+LGCE 97.50± 0.07 98.13± 0.08 VI (1 CLIENT) (96.96± 0.17) GVI (1 CLIENT) (98.13± 0.07) We create label contamination by adding noise to the train-δ = 0 .0 δ = 0 .2 δ = 0 .4 δ = 0 .6 δ = 0 .8 δ = 1 .0 α=0.0 α=0.5 α=1.0 α=1.5 α=2.5 α=5.0 6.16 3.14 1.92 2.26 2.55 2.83 3.68 3.09 2.03 2.01 2.24 2.48 3.46 3.02 2.08 1.95 2.08 2.28 3.32 3.93 2.33 1.96 2.04 2.12 3.16 5.21 2.81 1.83 1.91 2.13 2.63 7.04 4.19 2.05 1.89 1.95 AR . We plot the maximum results achieved as percentage errors on uncontaminated test data after training 5 clients on 10% contaminated data.
Table 2: Classification accuracy (highest in bold) on uncontaminated test data after training on different amounts of contaminated FASHIONMNIST data. For FEDGVI we have fixed α = 2.5 for the α-Rényi divergence. Each Method has data split homogeneously across 3 Clients. We report the best performance during all server iterations. MODEL CONTAMINATION 0% 10% 20% 40% FEDAVG 85.7±0.5 79.0±1.9 71.2±1.5 49.0±6.5 FEDPA 88.1±0.3 87.4±0.2 86.5±0.2 85.4±0.5 β-PREDBAYES 87.6±0.1 87.2±0.1 86.8±0.1 85.8±0.1 PVI 86.2±0.2 85.1±0.1 84.4±0.1 82.8±0.1 FEDGVI δ = 0.0 87.1±0.1 86.2±0.2 85.6±0.1 83.8±0.1 FEDGVI δ = 0.4 88.7±0.2 88.6±0.1 87.0±0.4 78.1±0.4 FEDGVI δ = 0.5 89.0±0.2 88.6±0.2 88.4±0.2 85.1±0.7 FEDGVI δ = 0.8 88.6±0.0 88.4±0.1 88.0±0.0 87.2±0.1 FEDGVI δ = 1.0 88.1±0.1 87.8±0.1 87.5±0.2 86.0±0.3
ing set while leaving the test set unchanged and evaluate performance in this. For MNIST, we add 10% of class dependent label noise, see Figure 7 and Table 1. We further carry out an ablation study on the hyperparameter selection in FEDGVI with the Alpha-Rényi divergence and the generalised cross entropy loss, see Figure 6. This demonstrates that FEDGVI performs well under a variety of different loss and divergence parameters. Note that α = 1 recovers the KL divergence, α = 0 the reverse KL divergence, i.e.
D (0) AR (q : π) = D RKL (q : π) = D KL (π : q)
, and that δ = 0 recovers the negative log-likelihood.
For FASHIONMNIST, in Table 2, we vary the amount of random label contamination, showcasing performance drops under different amounts of misspecification. We use an MLP, for FEDGVI and PVI with 1 hidden layer of 200
0 5 10 15 20 25 Server Iterations t 1 3 10 30 100 % Error (10 Clients) FedGVI D (2.5) AR , L (0.8) GCE FedGVI D (2.5) AR , L N LL FedGVI KL, L (0.8) GCE PVI KL, L N LL GVI D (2.5) AR , L (0.8) GCE VI KL, L N LL 0 5 10 15 20 25 Server Iterations t 1 0.1 NLL (10 Clients) FedGVI D (2.5) AR , L (0.8) GCE FedGVI D (2.5) AR , L N LL FedGVI KL, L (0.8) GCE PVI KL, L N LL GVI D (2.5) AR , L (0.8) GCE VI KL, L N LL 0 5 10 15 20 25 Server Iterations t 1 3 10 % Error (3 Clients) FedGVI D
(2.5) neurons; for FEDAVG, FEDPA, and β-PREDBAYES, two hidden layers with 100 neurons in each. Data is distributed homogeneously across clients, using 5 different, randomly chosen seeds. We demonstrate that under model misspecification, FEDGVI significantly outperforms competing FL methods. Furthermore, FEDGVI incurs no additional computational complexity when compared to PVI. This is due to the KL and Alpha-Rényi divergences having closed form solutions between Multivariate Gaussians with complexity of O(1) in each other, and as we require O(1) additional, constant operations to get the GCE from the NLL.
AR , L (0.8) GCE PVI KL, L N LL GVI D (2.5) AR , L (0.8) GCE VI KL, L N LL 0 5 10 15 20 25 Server Iterations t 1 0.1 NLL (3 Clients) FedGVI D (2.5) AR , L (0.8) GCE PVI KL, L N LL GVI D (2.5) AR , L (0.8) GCE VI KL, L N LL
We provide further experiments in Appendix D on the runtime of FEDGVI against PVI, learning rate selection, stability of posteriors under small perturbations in the robust loss parameters, and showing that using a single hidden layer NN for the competing methods would either negatively, or not significantly, affect their performance.
this section cite: []

Section: Conclusions and Future Work
We have introduced FEDGVI, a novel probabilistic approach to federated learning that is provably robust to model misspecification, and allows for faster, conjugate client updates. The theoretical analysis of FEDGVI demonstrates it's appealing properties; we easily recover existing methods as restricted cases, and characterise the convergence behaviour at fixed points of FEDGVI as solving a global GVI optimisation problem, extending existing theory. Our result on provable robustness to outliers through FEDGVI allows for closed form, conjugate posteriors that are computationally efficient, and robust to model misspecification. In deriving this, we have also shown that the cavity distribution is necessary as predictions would otherwise be overly confident and biased. The robustness of FEDGVI was further demonstrated empirically on multiple synthetic and real-world data sets, showing outperformance of existing FL methods across model architectures and misspecification levels.
An interesting future direction is to extend FedGVI within personalised FL settings (Kotelevskii et al., 2022) and hierarchical Bayesian FL through latent variables (Kim & Hospedales, 2023) as well as through the use of a structured posterior approximation (Hassan et al., 2024), in order to incorporate client level variations. Incorporating the hierarchical model structures and additional inductive biases from such settings, while maintaining conjugacy and favourable computational complexity, remain as open challenges. In future work, we further aim to address the robust Bayesian nonparametric setting of FL through FEDGVI, as well as investigate other types of robustness, including to adversarial and Byzantine attacks, by for instance using a robust aggregator in Equation ( 6), and addressing the open problem of provable robustness to prior misspecification in GVI.
this section cite: ['b52', 'b48', 'b36']

Section: References
Ref_id:b0 Title: Personalized federated learning with Gaussian processes Year: (2021)
Ref_id:b1 Title: Distributed stochastic gradient MCMC Year: (2014)
Ref_id:b2 Title: Federated learning via posterior averaging: A new perspective and practical algorithms Year: (2021)
Ref_id:b3 Title: A general class of coefficients of divergence of one distribution from another Year: (1966)
Ref_id:b4 Title: Byzantine-robust federated learning: Impact of client subsampling and local updates Year: (2024-07)
Ref_id:b5 Title: Non-exponentially weighted aggregation: Regret bounds for unbounded loss functions Year: (2021-07)
Ref_id:b6 Title: On the properties of variational approximations of gibbs posteriors Year: (2016)
Ref_id:b7 Title: Robust and scalable Bayesian online changepoint detection Year: (2023-07)
Ref_id:b8 Title: Robust and conjugate Gaussian process regression Year: (2024-07)
Ref_id:b9 Title: Information Geometry and Its Applications Year: (2016)
Ref_id:b10 Title: Partitioned variational inference: A framework for probabilistic federated learning Year: (2022)
Ref_id:b11 Title: BOBA: Byzantine-robust federated learning with label skewness Year: (2024-05)
Ref_id:b12 Title: Statistical Decision Theory and Bayesian Analysis Year: (1985)
Ref_id:b13 Title: Limiting behavior of posterior distributions when the model is incorrect Year: (1966)
Ref_id:b14 Title: Series in Probability and Statistics Year: (2000)
Ref_id:b15 Title: A general framework for updating belief distributions Year: (2016)
Ref_id:b16 Title: Variational inference: A review for statisticians Year: (2017)
Ref_id:b17 Title: Partitioned variational inference: A unified framework encompassing federated and continual learning Year: (2018)
Ref_id:b18 Title: Bayesian inference for the weights in logarithmic pooling Year: (2023)
Ref_id:b19 Title: Divide-and-conquer fusion Year: (2023)
Ref_id:b20 Title: The fundamental price of secure aggregation in differentially private federated learning Year: (2022-07)
Ref_id:b21 Title: Families of alpha-beta-and gamma-divergences: Flexible and robust measures of similarities Year: (2010)
Ref_id:b22 Title: Variational federated multi-task learning Year: (2021)
Ref_id:b23 Title: Methods with local steps and random reshuffling for generally smooth non-convex federated optimization Year: (2025)
Ref_id:b24 Title: On the consistency of Bayes estimates Year: (1986)
Ref_id:b25 Title: A general theory for federated optimization with asynchronous and heterogeneous clients updates Year: (2023)
Ref_id:b26 Title: A characterization theorem for externally Bayesian groups Year: (1984)
Ref_id:b27 Title: Characterization of externally Bayesian pooling operators Year: (1986)
Ref_id:b28 Title: Robust Bayes estimation using the density power divergence Year: (2016)
Ref_id:b29 Title: Robust estimation in generalized linear models: the density power divergence approach Year: (2016)
Ref_id:b30 Title: The safe Bayesian Year: (2012)
Ref_id:b31 Title: Federated learning as variational inference: A scalable expectation propagation approach Year: (2023)
Ref_id:b32 Title: FedBoost: A communication-efficient algorithm for federated learning Year: (2020)
Ref_id:b33 Title: Calibrated one round federated learning with Bayesian inference in the predictive space Year: (2024)
Ref_id:b34 Title: Distributed Bayesian learning with stochastic natural gradient expectation propagation and the posterior server Year: (2017)
Ref_id:b35 Title: Federated variational inference methods for structured latent variable models Year: (2023)
Ref_id:b36 Title: Scalable vertical federated learning via data augmentation and amortized inference Year: (2024)
Ref_id:b37 Title: Differentially private partitioned variational inference Year: ()
Ref_id:b38 Title: Bayesian model robustness via disparities Year: (2014)
Ref_id:b39 Title: Robust estimation of a location parameter Year: (1964)
Ref_id:b40 Title: Robust mislabel logistic regression without modeling mislabel probabilities Year: (2018)
Ref_id:b41 Title: Estimation of non-normalized statistical models by score matching Year: (2005)
Ref_id:b42 Title: Principles of Bayesian inference using general divergence criteria Year: (2018)
Ref_id:b43 Title: Bayesian federated inference for estimating statistical models based on non-shared multicenter data sets Year: (2024)
Ref_id:b44 Title:  Year: (2021)
Ref_id:b45 Title: Detecting and diagnosing prior and likelihood sensitivity with power-scaling Year: (2024)
Ref_id:b46 Title: Federated generalized Bayesian learning via distributed Stein variational gradient descent Year: (2022)
Ref_id:b47 Title: On the approximation accuracy of Gaussian variational inference Year: (2024)
Ref_id:b48 Title: Hierarchical Bayesian federated learning Year: (2023)
Ref_id:b49 Title: A method for stochastic optimization Year: (2015)
Ref_id:b50 Title: Doubly robust Bayesian inference for non-stationary streaming data with β-divergences Year: (2018)
Ref_id:b51 Title: An optimization-centric view on Bayes' rule: Reviewing and generalizing variational inference Year: (2022)
Ref_id:b52 Title: FedPop: A Bayesian approach for personalised federated learning Year: (2022)
Ref_id:b53 Title: On Information and Sufficiency Year: (1951)
Ref_id:b54 Title: Gradientbased learning applied to document recognition. Proceedings of the IEEE Year: (1998)
Ref_id:b55 Title: The power of extrapolation in federated learning Year: (2024)
Ref_id:b56 Title: From local SGD to local fixed-point methods for federated learning Year: (2020)
Ref_id:b57 Title: Robust generalised Bayesian inference for intractable likelihoods Year: ()
Ref_id:b58 Title: Communication-efficient learning of deep networks from decentralized data Year: (2017)
Ref_id:b59 Title: Federated stochastic gradient Langevin dynamics Year: (2021)
Ref_id:b60 Title: Embarrassingly parallel MCMC using deep invertible transformations Year: (2020)
Ref_id:b61 Title: Asymptotic normality, concentration, and coverage of generalized posteriors Year: (2021)
Ref_id:b62 Title: Expectation propagation for approximate Bayesian inference Year: (2001)
Ref_id:b63 Title: An elementary introduction to information geometry Year: (2020)
Ref_id:b64 Title: A simple approximation method for the fisher-rao distance between multivariate normal distributions Year: ()
Ref_id:b65 Title: Expectation consistent approximate inference Year: (2005)
Ref_id:b66 Title: Statistical inference based on divergence measures Year: (2006)
Ref_id:b67 Title: Kullback-leibler approximation for probability measures on infinite dimensional spaces Year: (2015)
Ref_id:b68 Title: Adaptive federated optimization Year: (2021)
Ref_id:b69 Title: Bayes and big data: The consensus monte carlo algorithm Year: (2016)
Ref_id:b70 Title: Connecting federated ADMM to Bayes Year: (2025)
Ref_id:b71 Title: Gradient masked averaging for federated learning Year: (2023)
Ref_id:b72 Title: A Bayesian committee machine Year: (2000)
Ref_id:b73 Title: Straggler-resilient personalized federated learning Year: (2023)
Ref_id:b74 Title: Federated variational inference: Towards improved personalization and generalization Year: (2024)
Ref_id:b75 Title: Expectation propagation as a way of life: A framework for Bayesian inference on partitioned data Year: (2020)
Ref_id:b76 Title: Bayesian inference with misspecified models Year: (2013)
Ref_id:b77 Title: Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms Year: (2017)
Ref_id:b78 Title: Adaptation of the tuning parameter in general Bayesian inference with robust divergence Year: (2023)
Ref_id:b79 Title: Bayesian nonparametric federated learning of neural networks Year: (2019-06)
Ref_id:b80 Title: Optimal information processing and Bayes's theorem Year: (1988)
Ref_id:b81 Title: Personalized federated learning via variational Bayesian inference Year: (2022-07)
Ref_id:b82 Title: Generalized cross entropy loss for training deep neural networks with noisy labels Year: (2018)
Ref_id:b83 Title: Deep leakage from model in federated learning Year: (2023)
