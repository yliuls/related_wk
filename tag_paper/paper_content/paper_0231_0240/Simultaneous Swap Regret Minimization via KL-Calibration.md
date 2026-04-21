Title: Simultaneous Swap Regret Minimization via KL-Calibration
Abstract: Calibration is a fundamental concept that aims at ensuring the reliability of probabilistic predictions by aligning them with real-world outcomes. There is a surge of studies on new calibration measures that are easier to optimize compared to the classical ℓ 1 -Calibration while still having strong implications for downstream applications. One such recent example is the work by Fishelson et al. (2025) who show that it is possible to achieve Õ(T 1/3 ) pseudo ℓ 2 -Calibration error via minimizing pseudo swap regret of the squared loss, which in fact implies the same bound for all bounded proper losses with a smooth univariate form. In this work, we significantly generalize their result in the following ways: (a) in addition to smooth univariate forms, our algorithm also simultaneously achieves Õ(T 1/3 ) swap regret for any proper loss with a twice continuously differentiable univariate form (such as Tsallis entropy); (b) our bounds hold not only for pseudo swap regret that measures losses using the forecaster's distributions on predictions, but also hold for the actual swap regret that measures losses using the forecaster's actual realized predictions. We achieve so by introducing a new stronger notion of calibration called (pseudo) KL-Calibration, which we show is equivalent to the (pseudo) swap regret with respect to log loss. We prove that there exists an algorithm that achieves Õ(T 1/3 ) KL-Calibration error and provide an explicit algorithm that achieves Õ(T 1/3 ) pseudo KL-Calibration error. Moreover, we show that the same algorithm achieves O(T 1/3 (log T ) -1 3 log(T /δ)) swap regret with probability at least 1 -δ for any proper loss with a smooth univariate form, which implies Õ(T 1/3 ) ℓ 2 -Calibration error. A technical contribution of our work is a new randomized rounding procedure and a non-uniform discretization scheme to minimize the swap regret for log loss.

Section: Introduction
We consider online calibration -a problem of making sequential probabilistic predictions over binary outcomes. Formally, at each time t = 1, . . . , T , a forecaster randomly predicts p t ∈ [0, 1] while simultaneously the adversary chooses y t ∈ {0, 1}, and subsequently the forecaster observes the true label y t . Letting n p denote the number of rounds the forecaster predicts p t = p, the forecaster's predictions are perfectly calibrated if for all p ∈ [0, 1], the empirical distribution of the label conditioned on the forecast being p, i.e., the quantity ρ p := t:pt=p y t /n p , matches p. The ℓ q -Calibration error (q ≥ 1) is then defined as Cal q := p∈[0,1] T t=1 I[p t = p] (p -ρ p )
q .
A related concept used in Fishelson et al. (2025) that we call pseudo calibration error measures the error using the forecaster's conditional distribution P t ∈ ∆ [0,1] at time t, instead of the actual prediction p t . More specifically, the pseudo ℓ q -Calibration error is defined as PCal q := T t=1 E p∼Pt [(p -ρp ) q ], More specifically, given a loss function ℓ : [0, 1] × {0, 1} → R, the swap regret of the forecaster is defined as SReg ℓ := sup σ:[0,1]→[0,1] SReg ℓ σ , where SReg ℓ σ := T t=1 ℓ(p t , y t )-ℓ(σ(p t ), y t ) measures the difference between the forecaster's total loss and the loss of a strategy that always swaps the forecaster's prediction via a swap function σ. Similarly, pseudo swap regret (Fishelson et al., 2025; referred in their work as full swap regret) is defined using the conditional distribution of predictions P t instead of p t itself: PSReg ℓ := sup σ:[0,1]→[0,1] PSReg ℓ σ , where PSReg ℓ σ := T t=1 E p∼Pt [ℓ(p, y t )ℓ(σ(p), y t )]. Fishelson et al. (2025) show that it is possible to achieve PSReg ℓ = Õ(T 1 3 ) when ℓ is the squared loss, which, as we will show, further implies that the same bound holds for any bounded proper loss ℓ with a smooth univariate form (refer to Section 2 for concrete definitions of proper losses and their univariate form).
In this work, we significantly generalize their results by not only recovering their results for pseudo swap regret, but also proving the same Õ(T 1 3 ) bound for new losses such as log loss and those induced by the Tsallis entropy. Moreover, we prove the same bound (either in expectation or with high probability) for the actual swap regret, which was missing in Fishelson et al. (2025). To achieve these goals, we introduce a natural notion of (pseudo) KL-Calibration, where the penalty incurred by the forecaster's prediction p deviating from the empirical distribution of y (conditioned on the forecast being p) is measured in terms of the KL-divergence. Specifically, the KL-Calibration and the pseudo KL-Calibration incurred by the forecaster are respectively defined as KLCal := p∈[0,1] T t=1 I[p t = p]KL(ρ p , p), PKLCal := T t=1 E p∼Pt [KL(ρ p , p)],
where KL(q, p) = q log q p + (1 -q) log 1-q 1-p is the KL-divergence for two Bernoulli distributions with mean q and p respectively. It follows from Pinsker's inequality that KL(ρ p , p) ≥ (ρ p -p) 2 , therefore, KLCal ≥ Cal 2 and PKLCal ≥ PCal 2 , making (pseudo) KL-Calibration a stronger measure for studying upper bounds than (pseudo) ℓ 2 -Calibration.
Contributions and Technical Overview Let L denote the class of bounded (in [-1, 1]) proper losses. Our concrete contributions are as follows.
• In Section 3, we start by discussing the implications of (pseudo) KL-Calibration towards minimizing (pseudo) swap regret. In particular, in subsection 3.1, we show for each ℓ ∈ L 2 , where L 2 is the class of bounded proper losses whose univariate form ℓ(p) := E y∼p [ℓ(p, y)] is twice continuously differentiable in (0, 1), we have SReg ℓ = O(KLCal), PSReg ℓ = O(PKLCal). In subsection 3.2, we show that for each ℓ ∈ L G , where L G is the class of bounded proper losses with a G-smooth univariate form, (pseudo) KL-Calibration implies that SReg
ℓ ≤ G • Cal 2 ≤ G • KLCal, PSReg ℓ ≤ G • PCal 2 ≤ G • PKLCal.
This gives us strong incentives to study PKLCal and KLCal.
• In Section 4, we prove that there exists an algorithm that achieves E[KLCal] = O(T 1 3 (log T )
3 ). To achieve so, we first realize that (pseudo) KL-Calibration is equivalent to the (pseudo) swap regret of the log loss ℓ(p, y) = -y log p -(1 -y) log(1 -p), i.e., KLCal = SReg ℓ , PKLCal = PSReg ℓ . Subsequently, we propose a non-constructive proof for minimizing SReg ℓ ; our proof is based on swapping the forecaster and the adversary via von-Neumann's minimax theorem. Two particularly technical aspects of our proof are the usage of a non uniform discretization, which is contrary to all previous works, and the use of Freedman's inequality for martingale difference sequences.
We remark that our non-constructive proof is motivated from Hu and Wu (2024), who provide a similar proof to show the existence of an algorithm that simultaneously achieves O( √ T log T ) swap regret for any bounded proper loss. However, compared to Hu and Wu (2024), we use a non uniform discretization, which requires a more involved analysis. * Moreover, due to the desired O(T 1 3 ) nature of our final bounds, we cannot merely use Azuma-Hoeffding that guarantees O( √ T ) concentration. The aforementioned reasons combined make our analysis considerably non-trivial and different than Hu and Wu (2024).
Combined with the implications of Section 3, we show the existence of an algorithm that simultaneously achieves the following bounds on E[SReg ℓ ]: (a) O(T 1 3 (log T )  5   6 ) for each ℓ ∈ L\{L 2 ∪ L G }. Notably, our result is better than Luo et al. (2024) who studied the weaker notion of external regret, defined as REG ℓ := sup p∈[0,1] T t=1 ℓ(p t , y t ) -ℓ(p, y t ), and showed that the Follow-the-Leader (FTL) algorithm achieves REG ℓ = O(log T ) for each ℓ ∈ L 2 ∪ L G , however incurs REG ℓ = Ω(T ) for a specific ℓ ∈ L\{L 2 ∪ L G }.
• In Section 5, we propose an explicit algorithm that achieves PKLCal = O(T 1 3 (log T ) 2 3 ). Similar to Fishelson et al. (2025), we utilize the Blum-Mansour reduction for minimizing PSReg ℓ for the log loss. However, our key novelty lies in the usage of a non uniform discretization and a new randomizing rounding procedure (Algorithm 4) for the log loss. Since the log loss is not Lipschitz, we show that the common rounding schemes studied in the literature fail to work for our considered discretization. A natural implication of our result is that, since PSReg ℓ ≤ G • PKLCal for any ℓ ∈ L G , we recover the result of Fishelson et al. (2025). However, since PSReg ℓ = O(PKLCal) for any ℓ ∈ L 2 , we are able to deal with new losses, and even the log loss which is unbounded.
• Finally, in Appendix E, we show that if we only consider the class of bounded proper losses with a smooth univariate form, our algorithm guarantees
Cal 2 = O T 1/3 (log T ) -1 3 log(T /δ) , Msr L G = O G • T 1/3 (log T ) -1 3 log(T /δ)
with probability at least 1 -δ, where Msr L G = sup ℓ∈L G SReg ℓ . This marks the first appearance of a sub-√ T high probability bound for classical ℓ 2 -Calibration via an efficient algorithm.
Related Work Calibration can also be viewed from the lens of simultaneous regret minimization (Kleinberg et al., 2023;Hu and Wu, 2024;Luo et al., 2024). It is known from Kleinberg et al. (2023) that ℓ 1 -Calibrated forecasts can simultaneously lead to sublinear swap regret for all ℓ ∈ L, where recall that L is the class of bounded (in [-1, 1]) proper losses. However, as shown by Qiao and Valiant (2021); Dagan et al. (2024), for any forecasting algorithm there exists an adversary that ensures that Cal 1 = Ω(T 0.54389 ), thereby sidestepping the goal of achieving the favorable √ T style regret guarantee. Despite the limitations of calibration, Hu and Wu (2024) proposed an explicit algorithm that achieves E[sup ℓ∈L SReg ℓ ] = O( √ T log T ). Compared to (Hu and Wu, 2024), we show that a single algorithm in fact achieves Õ(T 1 3 ) swap regret for important subclasses of L and even the log loss, while simultaneously achieving Õ(T 2 3 ) swap regret for any arbitrary ℓ ∈ L. Notably, the result of Hu and Wu (2024) does not apply to the log loss since it does not belong to L. With an appropriate post-processing of the predictions, a stronger analogue of simultaneous swap regret minimization has also been studied in the contextual setting (Garg et al., 2024; referred to as swap omniprediction), where the forecaster competes with functions from a hypothesis class F. Notably, in swap omniprediction, both the loss function and the competing hypothesis are parameterized by the predictions themselves. For this, Garg et al. (2024) showed that it is impossible to achieve O( √ T ) swap omniprediction error for the class of convex and Lipschitz loss functions, even in the simplest setting where F contains the constant 0, 1 functions. Additional related work is deferred to Appendix A.
this section cite: ['b5', 'b5', 'b5', 'b5', 'b11', 'b11', 'b11', 'b14', 'b5', 'b5', 'b12', 'b11', 'b14', 'b12', 'b16', 'b4', 'b11', 'b11', 'b11', 'b8', 'b8']

Section: Preliminaries and Background
Notation For a m ∈ N, [m] denotes the index set {1, . . . , m}. We reserve bold lower-case alphabets for vectors and bold upper-case alphabets for matrices. The notation I[•] refers to the indicator function, which evaluates to 1 if the condition is true, and 0 otherwise. We use e i to represent the i-th standard basis vector (dimension inferred from context), which is 1 at the i-th coordinate and 0 everywhere else. For any k ∈ N, we use ∆ k to represent the (k -1)-dimensional simplex. Moreover, we use ∆ [0,1] to represent the set of all probability distributions over [0, 1]. We use P t , E t to represent the conditional probability, expectation respectively, where the conditioning is over the randomness till time t -1 (inclusive). We use KL(p, q), TV(p, q), χ 2 (p, q) to represent the KL divergence, total variation distance, chi-squared distance between two Bernoulli distributions with means p, q. For a set I, its complement is Ī = Ω\I, where the sample set Ω shall be clear from the context. A twice differentiable function f :
D → R is α-smooth over D ⊂ R if f ′′ (x) ≤ α for all x ∈ D. A function f : W → R is α-exp-concave over a convex set W if the function exp(-αf (w))
is concave over W. We use the notation Õ(•) to hide lower order logarithmic terms.
Proper Losses A loss ℓ : [0, 1] × {0, 1} → R is called proper if E y∼p [ℓ(p, y)] ≤ E y∼p [ℓ(p ′ , y)]
for all p, p ′ ∈ [0, 1]. Intuitively, a proper loss incentivizes the forecaster to report the true distribution of the label. Throughout the paper, we shall be primarily concerned about the family L (or a subset) of bounded proper losses, i.e., L := {ℓ s.t. ℓ is proper and ℓ(p, y) ∈ [-1, 1] for all p ∈ [0, 1], y ∈ {0, 1}}, even though our results hold for (and in fact achieved via) the unbounded log loss. For a proper loss ℓ, the univariate form of ℓ is defined as ℓ(p) := E y∼p [ℓ(p, y)]. It turns out that a the univariate form of a proper loss is concave. Moreover, one can construct a proper loss using a concave univariate form based on the following characterization lemma. Lemma 1 (Theorem 2 in Gneiting and Raftery (2007)). A loss ℓ : [0, 1] × {0, 1} → R is proper if and only if there exists a concave function f such that ℓ(p, y) = f (p) + ⟨g p , y -p⟩ for all p ∈ [0, 1], y ∈ {0, 1}, where g p denotes a subgradient of f at p. Also, f is the univariate form of ℓ.
Examples of proper losses include squared loss ℓ(p, y) = (p -y) 2 , log loss ℓ(p, y) = y log 1 p + (1y) log 1 1-p , spherical loss ℓ(p, y) = -py+(1-p
)(1-y) √ p 2 +(1-p) 2 , etc. Bregman Divergence For a convex function ϕ, let BREG ϕ (x, y) = ϕ(x) -ϕ(y) -⟨∂ϕ(y), x -y⟩ denote the Bregman divergence associated with ϕ. The following lemma is important to our results. Lemma 2 (Lemma 3.8 in Hu and Wu (2024)). Let u : [0, 1] → [-1, 1] be a twice differentiable concave function. Then, we have BREG -u (p, p) = p p |u ′′ (µ)| • (p -µ)dµ.
Problem Setting As mentioned in Section 1, we consider calibration, where the interaction between the forecaster and the adversary is according to the following protocol: at each time t = 1, . . . , T , (a) the forecaster randomly predicts p t ∈ [0, 1] and simultaneously the adversary chooses y t ∈ {0, 1}; (b) the forecaster observes y t . Throughout the paper, we shall consider algorithms that make predictions p t that fall in a finite discretization Z ⊂ [0, 1]. According to ( 1 T t=1 I[pt=p] , ρp = T t=1 ytPt(p) T t=1
Pt(p) . * For simplicity, we assume that the adversary is oblivious, that is it selects y 1 , . . . , y T at time t = 0 with complete knowledge of the forecaster's algorithm. * Our goal is to minimize the (pseudo) KL-Calibration error, which as we show in Section 3, has powerful implications.
As mentioned, the swap regret of the forecaster with respect to a loss function ℓ against a swap function σ : [0, 1] → [0, 1] is SReg ℓ σ = T t=1 ℓ(p t , y t ) -ℓ(σ(p t ), y t ). Swap regret is then defined as SReg ℓ = sup σ:[0,1]→[0,1] SReg ℓ σ . Similarly, the pseudo swap regret is PSReg ℓ = sup σ:[0,1]→[0,1] PSReg ℓ σ , where PSReg ℓ σ = p∈Z T t=1 P t (p)(ℓ(p, y t ) -ℓ(σ(p), y t )). We further define maximum (pseudo) swap regret with respect to the class of bounded proper losses L as * For convenience, we set 0 0 = 0. This is because if np = 0, the forecast pt = p was never made and thus does not contribute to the calibration error.
* However, our results generalize directly to an adaptive adversary who decides yt based on p1, . . . , pt-1.
Msr L := sup ℓ∈L SReg ℓ , PMsr L := sup ℓ∈L PSReg ℓ . For a subset of losses L ′ ⊆ L, we define Msr L ′ and PMsr L ′ similarly, with the supremum over ℓ ∈ L ′ . The usage of ℓ for a bounded proper loss, or the log loss (which does not belong to L) shall be clear from the context.
this section cite: ['b9']

Section: Implications of (Pseudo) KL-Calibration
In this section, we discuss the implications of (pseudo) KL-Calibration towards minimizing the (pseudo) swap regret. In particular, we shall show that (pseudo) KL-Calibration upper bounds the following: (a) (P)SReg ℓ for all ℓ ∈ L 2 (subsection 3.1); (b) (P)Msr L G (subsection 3.2). This gives a strong incentive to study (pseudo) KL-Calibration.
The following proposition, which relates (pseudo) swap regret with Bregman Divergence is central to all subsequent results developed in this work. Proposition 1. For any proper loss ℓ and a swap function σ : [0, 1] → [0, 1], let BREG -ℓ be the Bregman divergence associated with the negative univariate form -ℓ. We have
SReg ℓ σ = p∈Z T t=1 I[p t = p] (BREG -ℓ (ρ p , p) -BREG -ℓ (ρ p , σ(p))) , PSReg ℓ σ = p∈Z T t=1 P t (p) (BREG -ℓ (ρ p , p) -BREG -ℓ (ρ p , σ(p))) ,
where
ρ p = T t=1 I[pt=p]yt T t=1 I[pt=p] , ρp = T t=1 Pt(p)yt T t=1 Pt(p) . Furthermore, SReg ℓ = p∈Z T t=1 I[p t = p]BREG -ℓ (ρ p , p), PSReg ℓ = p∈Z T t=1 P t (p)BREG -ℓ (ρ p , p).
The proof of Proposition 1, deferred to Appendix B, follows by an application of Lemma 1 and is similar to Hu and Wu (2024). Two particularly interesting applications of Proposition 1 are:
• For the squared loss ℓ(p, y) = (p-y) 2 , the univariate form is ℓ(p) = p-p 2 , and
BREG -ℓ (ρ p , p) = (ρ p -p) 2 . Therefore, SReg ℓ = Cal 2 , PSReg ℓ = PCal 2 .
• For the log loss ℓ(p, y) = y log 1 p + (1 -y) log 1 1-p , the univariate form is ℓ(p) = E y∼p [ℓ(p, y)] = -p log p -(1 -p) log(1 -p). Moreover, as can be verified by direct computation, the associated Bregman divergence BREG -ℓ (p, p) is exactly equal to KL(p, p). Therefore, we have SReg ℓ = KLCal, PSReg ℓ = PKLCal. This equivalence between (pseudo) KL-Calibration and (pseudo) swap regret of the log loss shall be our starting tool towards the developments in Sections 4, 5, where we bound KLCal, PKLCal respectively.
Note that since PSReg ℓ ≤ E[SReg ℓ ] trivially holds by definition, PCal 2 and PKLCal are indeed weaker notions compared to Cal 2 and KLCal respectively.
this section cite: ['b11']

Section: (Pseudo) KL-Calibration implies (pseudo) swap regret for all ℓ ∈ L 2
In this subsection, we show that SReg ℓ = O(KLCal), PSReg ℓ = O(PKLCal) for each ℓ ∈ L 2 , where L 2 := {ℓ ∈ L s.t. the univariate form ℓ(p) is twice continuously differentiable in (0, 1)}.
Note that according to Lemma 1, for all ℓ ∈ L, the univariate form must be concave, Lipschitz, and bounded, for the induced loss ℓ(p, y) to be proper and bounded. In addition to these implicit constraints, we require the condition that the second derivative ℓ ′′ (p) is continuous in (0, 1). We state several examples of losses that belong to L 2 . First, the squared loss clearly belongs to L 2 , since its univariate form is ℓ(p) = p -p 2 . Second, consider a generalization of the squared loss via Tsallis entropy, which corresponds to a loss with the univariate form ℓ(p) = -c • p α , where we choose α > 1 and the proportionality constant c > 0 is to ensure that the induced loss ℓ(p, y) is in [-1, 1] (refer Lemma 1). We have, ℓ(p, y) = c(α -1)p α -αcp α-1 y, which is in L 2 . Third, the spherical loss has the univariate form ℓ(p) = -p 2 + (1 -p) 2 and is also contained in L 2 .
The following lemma, derived by Luo et al. (2024), provides a growth rate on the second derivative of any ℓ ∈ L 2 and is a key ingredient for our proof of the desired implication. Lemma 3 (Lemma 2 in Luo et al. (2024)). For a function f that is concave, Lipschitz, and bounded over [0, 1] and twice continuously differentiable over (0, 1), there exists a constant c > 0 such that |f ′′ (p)| ≤ c • max 1 p , 1 1-p for all p ∈ (0, 1).
Using this to bound |u ′′ (p)| in the statement of Lemma 2, we immediately obtain the following proposition whose proof can be found in Appendix B. Proposition 2. Let ℓ ∈ L 2 . Then, we have BREG -ℓ (p, p) = O (KL(p, p)) and thus
SReg ℓ = O(KLCal), PSReg ℓ = O(PKLCal).
Note the constant c ℓ hidden in the O(.) notation in the result above is exactly the constant guaranteed by Lemma 3, which is finite. However, this is not sufficient to conclude that sup ℓ∈L2 c ℓ < ∞ (since L 2 is infinite), therefore, we do not necessarily guarantee that (P)Msr L2 (defined as sup ℓ∈L2 (P)SReg ℓ ) is O((P)KLCal). We remark that this is only a minor technical issue (that has also implicitly appeared in the prior work of Luo et al. (2024)), and our result in Proposition 2 implies that (pseudo) KL-Calibration simultaneously bounds (pseudo) swap regret for all ℓ ∈ L 2 . This in itself is quite meaningful and perfectly aligns with the goal in downstream decision making -to guarantee diminishing (swap) regret for all loss functions simultaneously. Henceforth, all subsequent results related to (pseudo) swap regret for L 2 are stated similarly. We also remark that Proposition 2 holds more generally for any subclass of proper losses where each loss satisfies the growth rate in Lemma 3. To keep the exposition simple, we only state our results for L 2 .
this section cite: ['b14', 'b14', 'b14']

Section: (Pseudo) KL-Calibration implies (pseudo) maximum swap regret against L
G We now consider another class L G , containing proper losses whose univariate form is G-smooth, i.e., L G := {ℓ ∈ L s.t. |ℓ ′′ (p)| ≤ G for all p ∈ [0, 1]}.
Losses that belong to L G include squared loss, spherical loss, Tsallis entropy for α ≥ 2, etc. Notably, the latter does not lie in L G for α ∈ (1, 2). Using Lemma 2 again, along with the fact PCal 2 ≤ PKLCal, Cal 2 ≤ KLCal due to Pinsker's inequality, we immediately obtain the following. Proposition 3. Let ℓ ∈ L G . Then, we have BREG -ℓ (p, p) ≤ G(p -p) 2 , and thus
Msr L G ≤ G • Cal 2 ≤ G • KLCal, PMsr L G ≤ G • PCal 2 ≤ G • PKLCal.
The proof of Proposition 3 is deferred to Appendix B. As already mentioned, Fishelson et al. (2025) proposed an algorithm that achieves PCal 2 = Õ(T 1 3 ), which implies that the same algorithm in fact ensures
PMsr L G = Õ(G • T 1 3
). However, the implications of KLCal, PKLCal allow us get simultaneous guarantees for a broader subclass of proper losses, particularly, L 2 ∪ L G .
this section cite: ['b5']

Section: Achieving KL-Calibration
In this section, we prove that there exists an algorithm that achieves E[SReg ℓ ] = O(T 1 3 (log T ) 5 3 ) for ℓ being the log loss, therefore the same algorithm achieves E[KLCal] = O(T 1 3 (log T ) 5 3 ). Our proof is non-constructive, since it is based on swapping the adversary and the algorithm via the minimax theorem (Theorem 3 in Appendix C), and deriving a forecasting algorithm in the dual game.
Theorem 1. There exists an algorithm that achieves E[SReg ℓ ] = O(T 1 3 (log T ) 5 3 ) for the log loss, where the expectation is taken over the internal randomness of the algorithm.
The proof of Theorem 1 is quite technical and is deferred to Appendix C. We discuss the key novelty of our proof here. Two particularly technical aspects of our proof are the usage of a non uniform discretization, which is contrary to all previous works, and the use of Freedman's inequality for martingale difference sequences (Lemma 8). In particular, we employ the following discretization scheme:
Z = {z 1 , . . . , z K-1 } ⊂ [0, 1], where z i = sin 2 πi
2K and K ∈ N is a constant to be specified later. For convinience, we set z 0 = 0, z K = 1, however, z 0 , z K are not included in the discretization. For our analysis, we require a discretization scheme that satisfies the following constraints: (a
) z i -z i-1 = O( 1 K ) for all i ∈ [K]; (b) max 2 (zi-zi-1,zi+1-zi) zi(1-zi) = O 1 K 2 for all i ∈ [K -1]; (c) K-1 i=1 1 zi(1-zi) = O(K 2 ); and (d) K-1 i=1 1 √ zi(1-zi) = Õ(K). The uniform discretization Z = { 1 K , . . . , K-1 K } satisfies (a)
, (c), (d) above, however, doesn't satisfy (b). As we show in Lemma 6 (Appendix C), our considered non uniform discretization achieves all these required bounds by having a finer granularity close to the boundary of [0, 1], thereby making it suitable for our purpose. The following steps provide a brief sketch of our proof, which is proved for an adaptive adversary and therefore also holds for the weaker oblivious adversary.
Step I We only consider discretized forecasters that make predictions that lie inside Z. Since the strategy space of such forecasters is finite, and that of the adversary is trivially finite, the minimax theorem (Theorem 3) applies, and we can swap the adversary and the algorithm, thereby resulting in the dual game. In this dual game, at every time t, the adversary first reveals the conditional distribution of y t , based on which the forecaster predicts p t . We consider a forecaster F which at time t does the following: (a) it computes pt = E t [y t ]; (b) predicts p t = argmin z∈Z |p t -z|. For such a forecaster, we obtain a high probability bound on SReg ℓ , and subsequently bound E[SReg ℓ ].
Step II Applying Lemma 8, we show that for each i (with
n i = n zi ) T t=1 I[p t = z i ](p t -y t ) ≤ 2 log 2 δ • max n i z i (1 -z i ) + π 2K , log 2 δ
with probability at least 1 -δKT . Using this, we bound |z i -ρ i |, where ρ i is a shorthand for ρ zi .
Notably, the bound above dictates separate consideration of i ∈ I and i ∈ Ī (depending on which term realizes the maximum), where
I := i ∈ [K -1]; n i < log 2 δ zi(1-zi)+ π 2K .
Step III Next, we write SReg ℓ as the sum of two terms SReg ℓ = Term I + Term II, where Term I = i∈I n i KL(ρ i , z i ), Term II = i∈ Ī n i KL(ρ i , z i ), and bound Term I, II individually. Since KL(ρ
i , z i ) ≤ χ 2 (ρ i , z i ) = (ρi-zi) 2 zi(1-zi)
, we utilize the bound on |ρ i -z i | obtained in the previous step and show that Term II = O T K 2 + K log 1 δ . Importantly, the use of Freedman's inequality provides a variance term that mitigates the potentially small denominator of
(ρi-zi) 2 zi(1-zi) . Similarly, we show that Term I = O T K 2 + K(log K) 3 2 log 1 δ . Combining, we ob- tain SReg ℓ = O T K 2 + K(log K) 3 2 log 1 δ with probability at least 1 -δKT . Subsequently, we bound E[SReg ℓ ] by setting δ = 1/T, K = T 1 3 /(log T ) 5 6 .
Equipped with Theorem 1, we prove the following stronger corollary (proof deferred to Appendix C). Corollary 1. There exists an algorithm that achieves the following bounds simultaneously:
E [KLCal] = O(T 1 3 (log T ) 5 3 ), E [Msr L G ] = O(G • T 1 3 (log T ) 5 3 ), E [Msr L2 ] = O(T 1 3 (log T ) 5 3 ), E Msr L\{L G ∪L2} = O(T 2 3 (log T ) 5 6 ),
where the expectation is taken over the internal randomness of the algorithm.
this section cite: []

Section: Achieving Pseudo KL-Calibration
In this section, we propose an explicit algorithm that achieves PSReg ℓ = O(T 1 3 (log T ) 2 3 ) for the log loss, therefore the same algorithm achieves PKLCal = O(T 1 3 (log T ) 2 3 ). Our algorithm is based on the well-known Blum-Mansour (BM) reduction (Blum and Mansour, 2007) and extends the idea from Fishelson et al. (2025). First, we employ a similar but slightly different non uniform discretization scheme that adds two extra end points z 0 and z K to the one used in the previous section (for technical reasons):
Z = {z 0 , z 1 , . . . , z K-1 , z K }, where z 0 = sin 2 π 4K , z i = sin 2 πi 2K for i ∈ [K -1], z K = cos 2 π 4K ,
and K ∈ N is a constant to be specified later. The same scheme was used before by Rooij and Erven (2009); Kotłowski et al. (2016) for different problems. Since the conditional distribution P t has support over Z, taking supremum over all swap functions σ : Z → Z in Proposition 1, we obtain
sup σ:Z→Z PSReg ℓ σ = PSReg ℓ - p∈Z T t=1 P t (p) inf σ:Z→Z BREG -ℓ (ρ p , σ(p)) ≥ PSReg ℓ - (2 - √ 2)π 2 T K 2 ,
where the inequality follows by choosing σ(p) = argmin z∈Z BREG -ℓ (ρ p , z). For this choice of σ, from Kotłowski et al. (2016, page 13), we have BREG -ℓ (ρ p , σ(p)) ≤ 2 -√ 2 π 2 K 2 . Therefore,
PSReg ℓ ≤ sup σ:Z→Z PSReg ℓ σ + 2 - √ 2 π 2 T K 2 ,(2)
and it suffices to bound sup σ:Z→Z PSReg ℓ σ , which we do via the BM reduction. Towards this end, we first recall the BM reduction. The reduction maintains K + 1 external regret algorithms A 0 , . . . , A K . At each time t, let q t,i ∈ ∆ K+1 represent the probability distribution over Z output by A i . Let Q t = [q t,0 , . . . , q t,K ] be the matrix obtained by stacking the vectors q t,0 , . . . , q t,K as columns. We compute the stationary distribution of Q t , i.e., a distribution p t ∈ ∆ K+1 over Z that satisfies Q t p t = p t . With p t being our final distribution of predictions (that is, P t (z i ) = p t,i ), we draw a prediction from it and observe y t . After that, we feed the scaled loss function p t,i ℓ(., y t ) to A i . Let lt,i = p t,i ℓ t ∈ R K+1 be a scaled loss vector, where ℓ t (j) = ℓ(z j , y t ). It then follows from Blum and Mansour (2007, Theorem 5) that sup σ:Z→Z PSReg ℓ σ ≤ K i=0 REG i , where REG i := sup j∈[K+1] T t=1 q t,i -e j , lt,i , i.e., the pseudo swap regret is bounded by the sum of the external regrets of the K + 1 algorithms. We summarize the discussion so far in Algorithm 1. Algorithm 1 BM for log loss Initialize: A i for i ∈ {0, . . . , K} and set q 1 = 1 K+1 , . . . , 1 K+1 ; 1: for t = 1, . . . , T 2: Set Q t = [q t,0 , . . . , q t,K ]; 3: Compute the stationary distribution of Q t , i.e., p t ∈ ∆ K+1 that satisfies Q t p t = p t ; 4: Output conditional distribution P t , where P t (z i ) = p t (i) and observe y t ; 5:
for i = 0, . . . , K 6:
Feed the scaled loss function f t,i (w) = p t,i ℓ(w, y t ) to A i (Algorithm 2) and obtain q t+1,i ; It remains to derive the i-th external regret algorithm A i that minimizes REG i . Note that A i is required to predict a distribution q t,i over Z and is subsequently fed a scaled loss function p t,i ℓ(., y t ) at each time t. We propose to employ the Exponentially Weighted Online Optimization (EWOO) algorithm along with a novel randomized rounding scheme for A i (Algorithm 2).
EWOO was studied by Hazan et al. (2007) for minimizing the regret sup w∈W T t=1 f t (w t ) -f t (w), when W is a convex set, and the loss functions f t 's are exp-concave. Since the log loss is 1-expconcave in p over [0, 1] ((Cesa-Bianchi and Lugosi, 2006, page 46), EWOO i (an instance of EWOO for A i ) with functions {f t,i } T t=1 defined as f t,i (w) = p t,i ℓ(w, y t ) for all w ∈ W, where W = [0, 1] is a natural choice.
Next, we derive a bound on the regret of EWOO i . Towards this end, we realize that the scaled log loss f t,i (w) = p t,i ℓ(w, y t ) is 1-exp-concave since exp(-f t,i (w)) = w ytpt,i (1 -w) (1-yt)pt,i is concave when p t,i ∈ [0, 1]. Appealing to (Hazan et al., 2007, Theorem 7), we then obtain the following: Lemma 4. The regret of Algorithm 3 satisfies sup w∈W T t=1 f t,i (w t,i ) -f t,i (w) ≤ log(T + 1).
Note that at each time t, EWOO i outputs w t,i ∈ [0, 1], however, A i is required to predict a distribution q t,i ∈ ∆ K+1 over Z. Thus, we need to perform a rounding operation that projects the output w t,i of EWOO i to a distribution over Z. In Remark 1 in Appendix D, we show that the following two known rounding schemes: (a) rounding w t,i to the nearest z ∈ Z and setting q t,i as the corresponding one-hot vector; (b) the rounding procedure proposed by Fishelson et al. (2025), cannot be applied to our setting since they incur a Ω(1) change in the expected loss ⟨q t,i , ℓ t ⟩ -ℓ(w t,i , y t ), which is not sufficient to achieve the desired regret guarantee. To mitigate the shortcomings of these rounding procedures, we propose a different randomized rounding scheme for the log loss (Algorithm 4) that achieves a Ofoot_0 K 2 change in the expected loss, as per Lemma 5. Lemma 5. Let p ∈ [0, 1] and p -, p + ∈ Z be neighbouring points in Z such that p -≤ p < p + . Let q be the random variable that takes value p -with probability ∝ p + -p p + (1-p + ) and p + with probability ∝ p-p - p -(1-p -) . Then, for all y ∈ {0, 1}, we have E[ℓ(q, y)] -ℓ(p, y) = O 1 K 2 .
The high-level idea of the proof is as follows: since the log loss is convex in p (for any y ∈ {0, 1}), we have ℓ(q, y) -ℓ(p, y)
≤ ℓ ′ (q, y) • (q -p) = (q-y)(q-p) q(1-q) , which is p q -1 if y = 1, and 1-p 1-q -1 if y = 0. By direct computation of E 1 q and E 1 1-q , we show that E p q -1 = E 1-p 1-q -1 ≤ (p + -p -) 2 • max
Combining everything, we derive the regret guarantee REG i of A i (Algorithm 2). It follows from Lemma 5 that at any time t, the distribution q t,i obtained by rounding the prediction w t,i of EWOO i as per Algorithm 4 satisfies ⟨q t,i , ℓ t ⟩ = ℓ(w t,i , y t ) + O( 1 K 2 ). Multiplying with p t,i and summing over all t, we obtain
T t=1 q t,i -e j , lt,i = T t=1 p t,i ℓ(w t,i , y t ) -T t=1 p t,i ℓ(z j , y t ) + O T t=1 p t,i Kfoot_1 , ≤ sup w∈W T t=1 f t,i (w t,i ) -f t,i (w) + O T t=1 p t,i K 2 = O log T + T t=1 p t,i K 2 , where the last equality follows from Lemma 4. Therefore, the regret REG i of A i satisfies REG i = O log T + 1 K 2 T t=1 p t,i . Summing over all i, we obtain sup σ:Z→Z
PSReg ℓ σ ≤ K i=0 REG i = O K log T + 1 K 2 K i=0 T t=1 p t,i = O K log T + T K 2 . Finally, it follows from (2) that PSReg ℓ = O K log T + T K 2 = O T 1 3 (log T ) 2 3 on choosing K = (T / log T ) 1 3
. Therefore, we have the main result of this section.
Theorem 2. Choosing K = (T / log T )
1 3 , Algorithm 1 achieves PKLCal = O T 1 3 (log T ) 2 3 .
Note that Algorithm 1 requires knowledge of the horizon T to choose the discretization parameter K. However, since (pseudo) KL-Calibration is equivalent to (pseudo) swap regret of the log loss, we can use the doubling trick to avoid the requirement of knowing the time horizon. The analysis of doubling trick for swap regret is exactly identical to that for external regret and is deferred to Algorithm 4 Randomized rounding for log loss (RROUND log ) Input: p ∈ [0, 1], Output: Probability distribution q ∈ ∆ K+1 ; Scheme: Let i ∈ {0, . . . , K -1} be such that p ∈ [z i , z i+1 ). Output q ∈ ∆ K+1 , where
q i = 1 D • z i+1 -p z i+1 (1 -z i+1 ) , q i+1 = 1 D • p -z i z i (1 -z i )
, and q j = 0, ∀j / ∈ {i, i + 1}
with D = p-zi zi(1-zi) + zi+1-p zi+1(1-zi+1) being the normalizing constant.
Cesa- Bianchi and Lugosi (2006). Moreover, as we show in Appendix D, the overall computation cost of Algorithm 1 over T rounds is Õ(T 5foot_3 + T • ST), where ST is the time required to compute the stationary distribution of Q t , which can be obtained efficiently by the method of power iteration; therefore, Algorithm 1 is efficient. In a similar spirit as Corollary 1, we can show Algorithm 1 achieves the following regret bounds simultaneously. The proof is in Appendix D and for most part follows similar to Corollary 1, except that we prove and utilize the bounds (a) PCal 1 ≤ √ T • PCal 2 ; (b) for any ℓ ∈ L, PSReg ℓ ≤ 4PCal 1 . Corollary 2. Algorithm 1 achieves the following bounds simultaneously:
PKLCal = O(T 1 3 (log T ) 2 3 ), PMsr L G = O(G • T 1 3 (log T ) 2 3 ), PMsr L2 = O(T 1 3 (log T ) 2 3 ), PMsr L\{L G ∪L2} = O(T 2 3 (log T ) 1 3 ).
We remark that while we do not have a concrete algorithm for KLCal, in Appendix E, we show that if we only consider L G , then our Algorithm 1 or the algorithm of Fishelson et al. (2025)
already achieves a O(G • T 1 3 (log T ) -1 3 log T δ ) high probability bound for Msr L G .
6 Conclusion and Future Directions
In this paper, we introduced a new stronger notion of calibration called (pseudo) KL-Calibration which not only allows us to recover results for classical (pseudo) ℓ 2 -Calibration, but also obtain simultaneous (pseudo) swap regret guarantees for several important subclasses of proper losses. We also derived the first high probability and in-expectation bounds for Cal 2 . Several interesting questions remain, including (1) obtaining an explicit high probability swap regret guarantee for the log loss, similar to Section E; (2) improving the T Contents 1 Introduction 2 Preliminaries and Background 3 Implications of (Pseudo) KL-Calibration 3.1 (Pseudo) KL-Calibration implies (pseudo) swap regret for all ℓ ∈ L 2 . . . . . . . . 3.2 (Pseudo) KL-Calibration implies (pseudo) maximum swap regret against L G . . . 4 Achieving KL-Calibration 5 Achieving Pseudo KL-Calibration 6 Conclusion and Future Directions A Additional Related Work B Deferred proofs in Section 3 B.1 Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.2 Proof of Proposition 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.3 Proof of Proposition 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C Deferred proofs in Section 4 C.1 Proof of Theorem 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.2 Proof of Corollary 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . D Deferred proofs and discussion in Section 5 D.1 Computational cost of Algorithm 1 . . . . . . . . . . . . . . . . . . . . . . . . . . D.2 Expected loss of common rounding schemes . . . . . . . . . . . . . . . . . . . . . D.3 Proof of Lemma 5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . D.4 Proof of Corollary 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . E High probability bound for maximum swap regret against L G
this section cite: ['b2', 'b5', 'b18', 'b13', 'b10', 'b10', 'b5', 'b3', 'b5']

Section: A Additional Related Work
Simultaneous regret minimization Kleinberg et al. (2023) proposed U-Calibration, where the goal is to simultaneously minimize the external regret REG ℓ for all ℓ ∈ L and provided an algorithm that achieves U-Calibration error UCal := sup ℓ∈L REG ℓ = O( √ T ). In the multiclass setting with K classes, Luo et al. (2024) proved that the minimax error is Θ( √ KT ). With an appropriate postprocessing of the predictions, the concept of U-Calibration has also been extended to the contextual setting (referred to as online omniprediction (Garg et al., 2024)). Very recently, Okoroafor et al. (2025) have shown that it is possible to achieve Õ( √ T ) omniprediction error for a family of bounded variation loss functions against a hypothesis class F with bounded complexity, thereby surpassing the limitations of swap omniprediction.
Weaker notions of calibration Understanding the limitations of online calibration, i.e., Cal 1 = O( √ T ) is impossible, has led to a recent line of work aimed at studying weaker notions of calibration which are still meaningful for downstream loss minimization tasks, e.g., continuous calibration (Foster and Hart, 2021), U-Calibration (Kleinberg et al., 2023), distance to calibration (Qiao and Zheng, 2024;Arunachaleswaran et al., 2025). Particularly, the last two works considered the problem of minimizing the distance to calibration (CalDist 1 ), defined as the ℓ 1 distance between the forecaster's vector of predictions and that of the nearest perfectly calibrated predictor, and proposed a non-constructive, constructive proof respectively that there exists an algorithm that achieves
CalDist 1 = O( √ T ). Since CalDist 1 ≤ Cal 1 ≤ √ T • Cal 2 , our Algorithm 1 in fact ensures that CalDist 1 = O(T 2 3 (log T ) -1 6 log(T /δ))
with probability at least 1 -δ, while simultaneously minimizing swap regret for several subclasses of L.
this section cite: ['b12', 'b14', 'b8', 'b15', 'b6', 'b12', 'b17', 'b0']

Section: B Deferred proofs in Section 3 B.1 Proof of Proposition 1
Proof. For simplicity, we only prove the result for PSReg ℓ σ since the result for SReg ℓ σ follows by simply replacing P t (p) with I[p t = p]. We have the following chain of equalities:
PSReg ℓ σ = p∈Z T t=1 P t (p)(ℓ(p, y t ) -ℓ(σ(p), y t )) = p∈Z T t=1 P t (p) (ℓ(p) + ⟨∂ℓ(p), y t -p⟩ -ℓ(σ(p)) -⟨∂ℓ(σ(p)), y t -σ(p)⟩) = p∈Z T t=1 P t (p) (ℓ(p) + ⟨∂ℓ(p), ρp -p⟩ -ℓ(σ(p)) -⟨∂ℓ(σ(p)), ρp -σ(p)⟩) = p∈Z T t=1 P t (p) (BREG -ℓ (ρ p , p) -BREG -ℓ (ρ p , σ(p))) ,
where the second equality follows from Lemma 1, while the final equality follows by adding and subtracting ℓ(ρ p ). Taking supremum over σ : [0, 1] → [0, 1], we obtain
sup σ:[0,1]→[0,1] PSReg ℓ σ = p∈Z T t=1 P t (p) BREG -ℓ (ρ p , p) - inf σ:[0,1]→[0,1] BREG -ℓ (ρ p , σ(p)) .
Next, we realize that BREG ϕ (x, y) ≥ 0 since ϕ is convex, and the choice of σ(p) = ρp leads to BREG -ℓ (ρ p , σ(p)) = 0. Therefore,
PSReg ℓ = p∈Z T t=1 P t (p) BREG -ℓ (ρ p , p)
which completes the proof.
this section cite: []

Section: B.2 Proof of Proposition 2
Proof. For simplicity, we only consider the case when p ≤ p, since the other case follows exactly similarly. Applying the result of Lemma 2, we obtain
BREG -ℓ (p, p) = p p |ℓ ′′ (µ)| (p -µ)dµ ≤ c • p p 1 µ + 1 1 -µ • (p -µ)dµ,
where the inequality follows from Lemma 3. By direct computation, the integral above evaluates to
p • p p dµ µ + (1 -p) • p p dµ µ -1 = p • log p p + (1 -p) • log 1 - p 1 -p = KL(p, p).
Therefore, we have BREG -ℓ (p, p) ≤ c • KL(p, p), which completes the proof of the first part of the Proposition. The second part follows by combining the result of Proposition 1 with the result obtained above, and taking a supremum over ℓ ∈ L 2 . This completes the proof.
Fix n ∈ [T ], µ ∈ [0, 1], δ ∈ [0, 1]. Applying Lemma 8, we obtain that the following inequality holds with probability at least 1 -δ:
n j=1 X j,i ≤ µV i (n) + 1 µ log 2 δ , where V i (n) = min(n,ni) j=1 ptj (1 -ptj ). To uniformly bound V i (n) in terms of n, we consider the 2 cases n ≤ n i and n > n i . When n ≤ n i , V i (n) can be bounded in terms of z i as follows V i (n) = nz i (1 -z i ) + n j=1 ptj (1 -ptj ) -z i (1 -z i ) = nz i (1 -z i ) + n j=1 (p tj -z i ) • (1 -ptj -z i ) ≤ nz i (1 -z i ) + n j=1 ptj -z i ≤ n z i (1 -z i ) + π 2K ,
where the last inequality follows from Lemma 6. When n > n i , we note that
V i (n) = V i (n i ) ≤ n z i (1 -z i ) + π 2K , since n > n i . Therefore, with probability at least 1 -δ, we have n j=1 X j,i ≤ µn z i (1 -z i ) + π 2K + 1 µ log 2 δ .
Minimizing the bound above with respect to µ ∈ [0, 1], we obtain
n j=1 X j,i ≤ 2 n z i (1 -z i ) + π 2K log 2 δ if n ≥ log 2 δ zi(1-zi)+ π 2K , n z i (1 -z i ) + π 2K + log 2 δ otherwise.
Note that when n <
log 2 δ zi(1-zi)+ π 2K , we can simply bound n z i (1 -z i ) + π 2K + log 2 δ < 2 log 2 δ .
The bounds obtained for both cases can be combined into the following single bound:
n j=1 X j,i ≤ 2 log 2 δ • max n z i (1 -z i ) + π 2K , log 2 δ ,
which holds with probability at least 1 -δ. Taking a union bound, we obtain that
n j=1 X j,i ≤ 2 log 2 δ • max n z i (1 -z i ) + π 2K , log 2 δ holds simultaneously for all i ∈ [K -1], n ∈ [T ] with probability at least 1 -(K -1)T δ ≥ 1 -KT δ. In particular, setting n = n i , we obtain that ni j=1 X j,i ≤ 2 log 2 δ • max n i z i (1 -z i ) + π 2K , log 2 δ (4
)
holds for all i ∈ [K -1] with probability at least 1 -KδT . Equipped with this bound, in the following steps we obtain a high probabilty bound on SReg ℓ (F, A). This shall be used to bound E[SReg ℓ (F, A)] eventually.
We begin by bounding the quantity |z i -ρ i |, which shall be used to obtain the high probability bound on SReg ℓ (F, A). We proceed as
|z i -ρ i | = 1 n i T t=1 I[p t = z i ](z i -y t ) ≤ 1 n i T t=1 I[p t = z i ](z i -pt ) + T t=1 I[p t = z i ](p t -y t ) ≤ max(d i , d i+1 ) + 1 n i ni j=1 X j,i ,
where for each i ∈ [K], we define d i := z i -z i-1 . The first inequality above follows from the Triangle inequality; the second inequality is because,
if p t = z i , we must have pt ∈ [z 0 , z1+z2 2 ] if i = 1, pt ∈ zi-1+zi 2 , zi+zi+1 2 if 2 ≤ i ≤ K -2, and pt ∈ z K-2 +z K-1 2 , 1 if i = K -1, therefore, |p t -p t | ≤ max(d i , d i+1 ). For each i ∈ [K -1], let t i := log 2 δ zi(1-zi)+ π 2K . Next, we write SReg ℓ (F, A) as SReg ℓ (F, A) = i∈I n i KL(ρ i , z i ) Term I + i∈ Ī n i KL(ρ i , z i ) Term II
, where I := {i ∈ [K -1]; n i < t i }, and bound Term I, II individually. We begin by bounding Term II in the following manner:
Term II ≤ i∈ Ī n i χ 2 (ρ i , z i ) = i∈ Ī n i (ρ i -z i ) 2 z i + (ρ i -z i ) 2 1 -z i = i∈ Ī n i (ρ i -z i ) 2 z i (1 -z i ) ≤ i∈ Ī 2n i z i (1 -z i )   (max(di, d i+1 )) 2 +   1 n i ni j=1 X j,i   2    ≤ i∈ Ī 2n i • (max(d i , d i+1 )) 2 z i (1 -z i ) + 8 log 2 δ •   i∈ Ī π 2K • 1 z i (1 -z i ) + 1   = O T K 2 + O K log 1 δ ,
where the first inequality follows since KL(ρ i , z i ) ≤ χ 2 (ρ i , z i ); the second inequality follows from the bound on |z i -ρ i | established above, and since (a + b) 2 ≤ 2a 2 + 2b 2 ; the third inequality follows from (4); the final equality follows from Lemma 6, particularly, we use the bounds
(max(di,di+1)) 2 zi(1-zi) = O( 1 K 2 ) and K-1 i=1 1 zi(1-zi) = O(K 2 ).
To bound Term I, we first note from the proof of Proposition 2 that
n i KL(ρ i , z i ) = sup σ:[0,1]→[0,1] T t=1 I[p t = z i ](ℓ(p t , y t ) -ℓ(σ(p t ), y t )) ≤ n i log 1 sin 2 π 2K ,
where the last inequality is because for the rounds where p t = z i , we have
ℓ(p t , y t ) ≤ max log 1 z i , log 1 1 -z i ≤ max log 1 sin 2 π 2K , log 1 1 -cos 2 π 2K = O(log K). (5) Similarly, 1 0 wµ t,i (w)dw = 1 0 w γ+1 (1 -w) δ dw = B(γ + 2, δ + 1) = Γ(γ + 2)Γ(δ + 1) Γ(γ + δ + 3) .
Taking ratio of the two integrals above and using the identity Γ(z + 1) = zΓ(z), which holds for all z with z > 0, we obtain
1 0 wµ t,i (w)dw 1 0 µ t,i (w)dw = Γ(γ + 2) Γ(γ + 1) • Γ(γ + δ + 2) Γ(γ + δ + 3) = γ + 1 γ + δ + 2 = 1 + δ + 1 γ + 1 -1 .
Clearly, at each time t, both γ and δ can be computed in O(1) time using the previously memorized values corresponding to time t -1. Therefore, INT = O(1). Since K = Θ(T 1 3 ), the overall computation cost over T rounds is Õ(T 5 3 + T • ST).
this section cite: []

Section: D.2 Expected loss of common rounding schemes
We recall the discussion in Section 5: at each time t, EWOO i outputs w t,i ∈ [0, 1], however, A i is required to predict a distribution q t,i ∈ ∆ K+1 over Z. Thus, we need to perform a rounding operation that projects the output w t,i of EWOO i to a distribution over Z. In the remark below, we show that the following two known rounding schemes: (a) rounding w t,i to the nearest z ∈ Z and setting q t,i as the corresponding one-hot vector; (b) the rounding procedure proposed by Fishelson et al. (2025), cannot be applied to our setting since they incur a Ω(1) change in the expected loss ⟨q t,i , ℓ t ⟩ -ℓ(w t,i , y t ), which is not sufficient to achieve the desired regret guarantee.
Remark 1. Let y t = 1 and w t,i = z0+z1
2 . The rounding procedure in (a) above ensures that q t,i = e 0 with probability one. Therefore, ⟨q t,i , ℓ t ⟩ -ℓ(w t,i , y
t ) = ℓ(z 0 , 1) -ℓ z0+z1 2 , 1 = log z0+z1 2z0 . Observe that z1 z0 = sin 2 π 2K sin 2 π 4K = 4 cos 2 π 4K = 2 + 2 cos π 2K .
Therefore, ⟨q t,i , ℓ t ⟩ -ℓ(w t,i , y t ) = log 3 2 + cos π 2K = Ω(1). For the chosen example, the rounding procedure in (b) sets q t,i (0) = q t,i (1) = 1 2 . Thus, ⟨q t,i , ℓ t ⟩ -ℓ(w t,i , y t ) = ℓ(z0,1)+ℓ
(z1,1) 2 -ℓ z0+z1 2 , 1 = log z0+z1 2 √ z0z1 = log 1+4 cos 2 π 4K 4 cos π 4K
= Ω(1).
this section cite: ['b5']

Section: D.3 Proof of Lemma 5
Proof. Since the log loss ℓ(p, y) is convex in p (for any y ∈ {0, 1}), we have
ℓ(q, y) -ℓ(p, y) ≤ ℓ ′ (q, y) • (q -p) = (q -y)(q -p) q(1 -q) = p q -1 if y = 1, 1-p 1-q -1 if y = 0.(8)
Let y = 1. Taking expectation on both sides of (8), we obtain E[ℓ(q, y)] -ℓ(p, y) = E p q -1. To simplify the expressions involved in the computation of E 1 q , we define the normalizing factor D := p + -p p + (1-p + ) + p-p - p -(1-p -) . By direct computation, we have
E 1 q = 1 D p + -p p -p + (1 -p + ) + p -p - p -p + (1 -p -) = 1 D • (p + -p -)(1 -p) p -p + (1 -p -)(1 -p + )
.
Similarly, by direct computation, we obtain
D = p + -p p + (1 -p + ) + p -p - p -(1 -p -) = (p + -p -) (p + p -p + -p(p -+ p + )) p -p + (1 -p -)(1 -p + ) .
Therefore,
E p q -1 = p(1 -p) p + p -p + -p(p -+ p + ) -1 = (p + -p)(p -p -) p + p -p + -p(p -+ p + ) ≤ (p + -p -) 2 p + p -p + -p(p -+ p + )
.
Next, we let y = 0. Taking expectation on both sides of (8), we obtain E[ℓ(q, y)] -ℓ(p, y) = E 1-p 1-q -1, thus, we require to bound E 1 1-q . Direct computation yields
E 1 1 -q = 1 D p + -p p + (1 -p -)(1 -p + ) + p -p - p -(1 -p -)(1 -p + ) = 1 D • p(p + -p -) p -p + (1 -p -)(1 -p + )
.
Substituting the expression for D obtained above, we obtain
E 1 -p 1 -q -1 = p(1 -p) p + p -p + -p(p -+ p + ) -1 = (p + -p)(p -p -) p + p -p + -p(p -+ p + ) ≤ (p + -p -) 2 p + p -p + -p(p -+ p + ) . Let f (p) = p + p -p + -p(p -+ p + ). Since f (p) is linear in p, for any p ∈ [p -, p + ), we have min(f (p -), f (p + )) ≤ f (p) ≤ max(f (p -), f (p + )). Since f (p -) = p -(1 -p -), f (p + ) = p + (1 - p + ), we obtain min p -(1 -p -), p + (1 -p + ) ≤ p + p -p + -p(p -+ p + ) ≤ max p -(1 -p -), p + (1 -p + )
for all p ∈ [p -, p + ). Therefore,
E q [ℓ(q, y)] -ℓ(p, y) ≤ (p + -p -) 2 • max 1 p -(1 -p -) , 1 p + (1 -p + ) = O 1 K 2 ,
where the last equality follows from Lemma 7. This completes the proof.
Lemma 7. Fix a k ∈ N. Let {z i } K i=0 be a sequence where z 0 = sin 2 π 4K , z i = sin 2 πi 2K for i ∈ [K -1], and z K = cos 2 π 4K . For each i = 1, . . . , K, define d i := z i -z i-1 . Then, the following holds true for all i ∈ [K]: (a)
d 2 i zi(1-zi) = O 1
K 2 , and (b)
d 2 i zi-1(1-zi-1) = O 1 K 2 .
Proof. It follows from Lemma 6 that (a), (b) hold for all 2 ≤ i ≤ K -1. For i = 1, since d 1 ≤ z 1 = sin 2 π 2K , we have
d 2 1 z 1 (1 -z 1 ) ≤ sin 4 π 2K sin 2 π 2K cos 2 π 2K = tan 2 π 2K , which is O( 1 K 2 ) for a large K. Similarly, for i = K, d i = cos 2 π 4K -cos 2 π 2K = sin 2 π 2K -sin 2 π 4K ≤ sin 2 π 2K . Therefore, d 2 K z K (1 -z K ) ≤ sin 4 π 2K sin 2 π 4K cos 2 π 4K = 4 sin 2 π 2K ≤ π 2 K 2 ,
where the equality follows from the identity sin 2θ = 2 sin θ cos θ. This completes the proof for (a). For (b), when i = 1, we have
d 2 1 z 0 (1 -z 0 ) ≤ sin 4 π 2K sin 2 π 4K cos 2 π 4K = 4 sin 2 π 2K ≤ π 2 K 2 .
Similarly, when i = K, we have
d 2 K z K-1 (1 -z K-1 ) ≤ sin 4 π 2K sin 2 π 2K cos 2 π 2K = tan 2 π 2K , which is O( 1 K 2 )
for a large K. This completes the proof.
this section cite: []

Section: D.4 Proof of Corollary 2
Proof. Since KLCal ≥ PCal 2 , Algorithm 1 ensures that PCal 2 = O(T 1 3 (log T ) 2 3 ). Next, we show that the PCal 1 satisfies (a) PCal 1 ≤ √ T • PCal 2 ; (b) for any proper loss ℓ, we have PSReg ℓ ≤ 4PCal 1 . The proof is exactly similar to the corresponding variants of (a), (b) above for Cal as shown by Kleinberg et al. (2023). For (a), applying the Cauchy-Schwartz inequality, we obtain
p∈Z t=1 P t (p) |p -ρp | ≤   p∈Z T t=1 P t (p)   1 2   p∈Z T t=1 P t (p)(p -ρp ) 2   1 2 = T • PCal 2 .
Towards showing (b), we first rewrite PSReg ℓ = p∈Z T t=1 P t (p)BREG -ℓ (ρ p , p), which holds for any proper loss ℓ as per Proposition 2. Next, we observe that
BREG -ℓ (ρ p , p) = ℓ(p) -ℓ(ρ p ) + ∂ℓ(p)(ρ p -p) ≤ ∂ℓ(ρ p )(p -ρp ) + ∂ℓ(p)(ρ p -p) ≤ 4 |p -ρp | ,
where the first inequality follows since ℓ(p) is concave
; the second inequality follows by noting that ℓ(p, 1) -ℓ(p, 0) = ∂ℓ(p) as per Lemma 1, and since ℓ(p, y) ∈ [-1, 1], we have |∂ℓ(p)| ≤ 2 for all p ∈ [0, 1]. Substituting the bound on BREG -ℓ (ρ p , p) obtained above into PSReg ℓ , we obtain PSReg ℓ ≤ 4PCal 1 as desired. Since Algorithm 1 ensures PCal 1 = O(T 1 3 (log T ) 1 3 ), we obtain PSReg ℓ = O(T 1 3 (log T ) 1 3 ). Combining the above results with Propositions 2, 3 finishes the proof. E High probability bound for maximum swap regret against L G While we do not have a concrete algorithm for KLCal, in this section, we show that if we only consider L G , then our Algorithm 1 or the algorithm of Fishelson et al. (2025) already achieves a O(G • T 1 3 (log T ) -1 3 log T δ ) high probability bound for Msr L G . To obtain so, we first prove a generic high probability bound that relates Cal 2 with PCal 2 . Subsequently, we instantiate our bound with an explicit algorithm for minimizing PCal 2 and use the result of Proposition 3. Our high probability bound in Theorem 4 is independent of the choice of the discretization Z. Theorem 4. For any algorithm A Cal , with probability at least 1 -δ over the randomness in A Cal 's predictions p 1 , . . . , p T , we have Cal 2 ≤ 6PCal 2 + 96 |Z| log 4|Z| δ . Our proof of Theorem 4 crucially relies on the following version of Freedman's inequality from Beygelzimer et al. (2011). Refer therein for a proof. Lemma 8. Let X 1 , . . . , X n be a martingale difference sequence adapted to the filtration F 1 ⊆ • • • ⊆ F n , where |X i | ≤ B for all i = 1, . . . , n, and B is a fixed constant. Define V := n i=1 E[X 2 i |F i-1 ]. Then, for any fixed µ ∈ 0, 1 B , δ ∈ [0, 1], with probability at least 1 -δ, we have n i=1 X i ≤ µV + log 2 δ µ . Proof of Theorem 4. Before discussing the proof, we introduce some notation. Let Z be enumerated as Z = {z 0 , . . . , z K }, where K = |Z| -1. Observe that at time t, A Cal can be equivalently described by the following procedure: (a) it samples i t from the set {0, . . . , K} with P t (i t = i) = P t (z i ), which we write as P t,i for convenience; (b) forecasts p t = z it . Clearly, I[p t = z i ] = I[i t = i]. For simplicity, we denote ρ zi = ρ i and ρzi = ρi . Under this notation, ρ i , ρi can be expressed as ρ i = T t=1 y t I[i t = i] T t=1 I[i t = i] , ρi = T t=1 y t P t,i T t=1 P t,i . We begin by bounding |ρ i -ρi | using Lemma 8. Fix a i ∈ {0, . . . , K} and define the martingale difference sequences X t := y t (P t,i -I[i t = i]) and Y t := P t,i -I[i t = i]. Observe that |X t | ≤ 1, |Y t | ≤ 1 for all t. Fix a µ i ∈ [0, 1]. Applying Lemma 8 to the sequences X, Y and taking a union bound (over X, Y ), we obtain that with probability at least 1 -δ, T t=1 y t (P t,i -I[i t
= i]) ≤ µ i V X + log 4 δ µ i , T t=1 P t,i -I[i t = i] ≤ µ i V Y + log 4 δ µ i ,(9)
where V X , V Y are given by V X = T t=1 E X 2 t |F t-1 = T t=1 y t • P t,i (1 -P t,i ) ≤ T t=1 P t,i , and V Y = T t=1 E Y 2 t |F t-1 = T t=1 P t,i (1 -P t,i ) ≤ T t=1 P t,i . The upper tail ρ i -ρi can then be bounded in the following manner: ρ i -ρi = T t=1 y t I[i t = i] T t=1 I[i t = i] -T t=1 y t P t,i T t=1 P t,i ≤ T t=1 y t I[i t = i] T t=1 I[i t = i] + µ i T t=1 P t,i + log 4 δ µi -T t=1 y t I[i t = i] T t=1 P t,i = T t=1 y t I[i t = i] T t=1 I[i t = i] T t=1 P t,i • T t=1 P t,i -I[i t = i] + µ i T t=1 P t,i + log 4 δ µi T t=1 P t,i ≤ T t=1 y t I[i t = i] T t=1 I[i t = i] T t=1 P t,i • µ i T t=1 P t,i + log 4 δ µ i + µ i T t=1 P t,i + log 4 δ µi T t=1 P t,i ≤ 2µ i + 2 log 4 δ µ i T t=1 P t,i , where the first and second inequalities follow from (9), while the last inequality follows by bounding y t I[i t = i] ≤ I[i t = i]. The lower tail can be bounded in an exact same manner as ρi -ρ i = T t=1 y t P t,i T t=1 P t,i -T t=1 y t I[i t = i] T t=1 I[i t = i] ≤ T t=1 y t I[i t = i] + µ i T t=1 P t,i + log 4 δ µi T t=1 P t,i -T t=1 y t I[i t = i] T t=1 I[i t = i] = T t=1 y t I[i t = i] T t=1 P t,i T t=1 I[i t = i] • T t=1 I[i t = i] -P t,i + µ i T t=1 P t,i + log 4 δ µi T t=1 P t,i ≤ T t=1 y t I[i t = i] T t=1 I[i t = i] T t=1 P t,i • µ i T t=1 P t,i + log 4 δ µ i + µ i T t=1 P t,i + log 4 δ µi T t=1 P t,i ≤ 2µ i + 2 log 4 δ µ i T t=1 P t,i . Combining both the bounds, we have shown that for a fixed µ i ∈ [0, 1], |ρ i -ρi | ≤ 2µ i + 2 log 4 δ µi T t=1 Pt,i holds with probability at least 1 -δ. Taking a union bound over all i, with probability 1 -δ, we have (simultaneously for all i) T t=1 y t (P t,i -I[i t = i]) ≤ µ i T t=1 P t,i + log 4(K+1) δ µ i , T t=1 P t,i -I[i t = i] ≤ µ i T t=1 P t,i + log 4(K+1) δ µ i ,
|ρ i -ρi | ≤ 2µ i + 2 log 4(K+1) δ µ i T t=1 P t,i .(10)
Consider the function g(µ) := µ + a µ , where a ≥ 0 is a fixed constant. Clearly, min µ∈[0,1] g(µ) = 2 √ a when a ≤ 1, and 1 + a otherwise. Minimizing the bound in (11) with respect to µ i , we obtain
|ρ i -ρi | ≤ 4 log 4(K+1)
δ T t=1 P t,i , when log 4(K + 1) δ ≤ T t=1 P t,i . However, when log 4(K+1) δ > T t=1 P t,i , we obtain that |ρ i -ρi | ≤ 2 + 2 log 4(K+1) δ T t=1 Pt,i . In particular, when T t=1 P t,i is tiny, which is possible if A Cal does not allocate enough probability mass to the index i, the bound obtained is large making it much worse than the trivial bound |ρ i -ρi | ≤ 1 which follows since ρ i , ρi ∈ [0, 1] by definition. Based on this reasoning, we define the set I := i ∈ {0, . . . , K} s.t. log 4(K + 1) δ ≤ T t=1 P t,i ,
and bound (ρ i -ρi ) 2 as (ρ i -ρi ) 2 ≤ 16 log 4(K+1) δ T t=1 Pt,i if i ∈ I, 1 otherwise. (13) Similarly, T t=1 P t,i -I[i t = i] can be bounded by substituting the optimal µ i obtained above in (10); we obtain T t=1 P t,i -I[i t = i] ≤ 2 log 4(K+1) δ T t=1 P t,i if i ∈ I, T t=1 P t,i + log 4(K+1) δ otherwise.
Equipped with (13), ( 14), we proceed to bound Cal 2 in the following manner:
Cal 2 = K i=0 T t=1 I[i t = i] (z i -ρ i ) 2 ≤ 2 K i=0 T t=1 I[i t = i] (z i -ρi ) 2 + (ρ i -ρi ) 2 ,
where the inequality is because (a + b) 2 ≤ 2a 2 + 2b 2 for all a, b ∈ R. To further bound the term above, we split the summation into two terms T 1 , T 2 defined as
T 1 := i∈I T t=1 I[i t = i] (z i -ρi ) 2 + (ρ i -ρi ) 2 , T 2 = i∈ Ī T t=1 I[i t = i] (z i -ρi ) 2 + (ρ i -ρi ) 2 , and bound T 1 and T 2 individually. We bound T 1 as T 1 ≤ i∈I   T t=1 P t,i + 2 log 4(K + 1) δ T τ =1 P τ,i   (z i -ρi ) 2 + 16 log 4(K+1) δ T τ =1 P τ,i = i∈I T t=1 P t,i (z i -ρi ) 2 + 16 log 4(K + 1) δ |I| + 2 i∈I log 4(K + 1) δ T τ =1 P τ,i (z i -ρi ) 2 + 16 log 4(K+1) δ T τ =1 P τ,i ≤ i∈I T t=1 P t,i (z i -ρi ) 2 + 16 log 4(K + 1) δ |I| + 2 i∈I T τ =1 P τ,i (z i -ρi ) 2 + 16 log 4(K+1) δ T τ =1 P τ,i = 3 i∈I T t=1 P t,i (z i -ρi ) 2 + 48 log 4(K + 1) δ |I| ,
where the first inequality follows by substituting the bounds from ( 13), ( 14), while the final inequality follows since by the definition of I in (12), we have log 4(
K+1) δ T τ =1 P τ,i ≤ T τ =1 P τ,i . Next, we bound T 2 as T 2 ≤ i∈ Ī 2 T t=1 P t,i + log 4(K + 1) δ (z i -ρi ) 2 + 1 ≤ 2 i∈ Ī T t=1 P t,i (z i -ρi ) 2 + 2 i∈ Ī T t=1 P t,i + 2 log 4(K + 1) δ Ī ≤ 2 i∈ Ī T t=1 P t,i (z i -ρi ) 2 + 4 log 4(K + 1) δ Ī , where the first inequality follows by substituting the bounds from (13), (14); the second inequality follows by bounding (z i -ρi ) 2 ≤ 1; the final inequality follows from the definition of I (12). Collecting the bounds on T 1 and T 2 , we obtain T 1 + T 2 ≤ 3 K i=0 T t=1 P t,i (z i -ρi ) 2 + 48 log 4(K + 1) δ |I| + 4 log 4(K + 1) δ Ī ≤ 3PCal 2 + 48(K + 1) log 4(K + 1) δ , where the last inequality follows from the definition of PCal 2 and since |I| + Ī = K + 1. Since Cal 2 ≤ 2(T 1 + T 2 ), we have shown that Cal 2 ≤ 6PCal 2 + 96(K + 1) log 4(K + 1) δ (15)
with probability at least 1 -δ. This completes the proof.
Instantiating A Cal in Theorem 4, we obtain the following corollary.
Corollary 3. On choosing K = (T / log T ) 1 3 , Algorithm 1 ensures that with probability at least 1 -δ over its internal randomness
Cal 2 = O T 1 3 (log T ) -1 3 log T δ , Msr L G = O G • T 1 3 (log T ) -1 3 log T δ .
Furthermore, E[Cal 2 ] = O(T
1 3 (log T ) 2 3 ), E[Msr L G ] = O(G • T 1 3 (log T )2
3 ). 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The paper is a theory work and the core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b12']

Section: B.3 Proof of Proposition 3
Proof. For simplicity, we only consider the case when p ≤ p, since the other case follows exactly similarly. Applying the result of Lemma 2, we obtain
BREG -ℓ (p, p) ≤ G p p (p -µ)dµ = G p(p -p) - p2 -p 2 2 = G 2 (p -p) 2 .
The case when p ≤ p follows similarly. Applying the result of Proposition 1, taking a supremum over ℓ ∈ L G , and bounding Cal 2 , PCal 2 in terms of KLCal, PKLCal completes the proof.
this section cite: []

Section: C Deferred proofs in Section 4
C.1 Proof of Theorem 1
Theorem 3 (Von-Neumann's Minimax Theorem). Let M ∈ R r×c for r, c ∈ N. Then,
min p∈∆r max q∈∆c p ⊺ M q = max q∈∆c min p∈∆r p ⊺ M q.
Proof of Theorem 1. We prove a stronger statement that the result holds against any adaptive adversary. In the forecasting setup, let H t-1 = {p 1 , . . . , p t-1 } ∪ {y 1 , . . . , y t-1 } denote the history till time t (exclusive). With complete knowledge about the forecaster's algorithm, an adaptive adversary chooses y t depending on H t-1 . As mentioned in Section 4, we shall consider forecasters that make predictions which belong to the discretization
Z = {z 1 , . . . , z K-1 }, where z i = sin 2 πi 2K,
and K ∈ N is a constant to be specified later. For convinience, we set z 0 = 0, z K = 1, however, z 0 , z K are not included in the discretization. In Lemma 6, we prove some important facts regarding Z which shall be useful for the subsequent analysis. For a deterministic forecaster, p t is obtained via a mapping F t-1 : H t-1 → Z. Similarly, for a deterministic adversary, y t is obtained via a mapping A t-1 : H t-1 → {0, 1}. Therefore, a deterministic forecaster can be represented by the sequence of mappings F = (F 1 , . . . , F T ), and a deterministic adversary can be represented by the sequence A = (A 1 , . . . , A T ). Given F, A, we let SReg ℓ (F, A) denote the swap regret achieved by executing F, A.
Let {F }, {A} be all possible enumerations of F, A respectively, and ∆({F }), ∆({A}) denote the set of all distributions over {F }, {A}. Then, F ∈ ∆({F }), A ∈ ∆({A}) are distributions over {F }, {A} and represent a randomized forecaster, adversary respectively. Note that |{F }| , |{A}| < ∞, since the domain and range of each map F t , A t is finite. Therefore, by Theorem 3, we have
min F∈∆({F }) max A∈∆({A}) E F ∼F,A∼A [SReg ℓ (F, A)] = max A∈∆({A}) min F∈∆({F }) E F ∼F,A∼A [SReg ℓ (F, A)].(3)
For a v ∈ R, to upper bound the quantity on the right hand side of (3) by v, it is sufficient to prove that for any randomized adversary there exists a forecaster F that guarantees that E[SReg ℓ (F, A)] ≤ v. Moreover, swapping the adversary and forecaster allows the forecaster to witness the distribution of y t before deciding p t . Towards this end, we consider a forecaster F which at time t does the following: (a
) it computes pt = E t [y t ]; (b) predicts p t = argmin z∈Z |p t -z|.
For each i ∈ {1, . . . , K -1} and n ∈ [T ], let n i (n) := n t=1 I[p t = z i ]. For convinience, we refer to n i (T ) as n i . Fix a i ∈ [K -1], and define the sequence X 1,i , . . . , X T,i as follows:
X j,i := 0 if j > n i , y tj -ptj if j ≤ n i .
Here t j denotes the j-th time instant when the prediction made is p t = z i . Observe that the sequence X 1,i , . . . , X T,i is a martingale difference sequence with |X j,i | ≤ 1 for all j ∈ [T ]. In the subsequent steps we obtain a high probability bound on prefix sums of this sequence.
Moreover, repeating the exact same steps done to bound Term II above, we can also bound n i KL(ρ i , z i ) as
n i KL(ρ i , z i ) ≤ 2n i z i (1 -z i )   (max(di, d i+1 )) 2 +   1 n i ni j=1 X j,i   2    = O n i K 2 + 8 log 2 δ 2 • 1 n i z i (1 -z i ) = O n i K 2 + log 1 δ 2 • 1 n i z i (1 -z i ) ,
where the first equality follows from Lemma 6 and (4). Taking minimum of the two bounds obtained above, we obtain
n i KL(ρ i , z i ) = O min n i log K, n i K 2 + log 1 δ 2 • 1 n i z i (1 -z i ) = O n i K 2 + min n i log K, log 1 δ 2 • 1 n i z i (1 -z i ) = O n i K 2 + log K log 1 δ • 1 z i (1 -z i ) ,
where the final inequality follows since for a fixed a > 0, min(x, a x ) ≤ √ a holds for all x ∈ R. Summing over i ∈ I, we obtain the following bound on Term I:
Term I = O 1 K 2 i∈I n i + log K log 1 δ • i∈I 1 z i (1 -z i ) = O T K 2 + K(log K) 3 2 log 1 δ ,
where the last equality follows from Lemma 6, particularly,
K-1 i=1 1 √ zi(1-zi) = O(K log K).
Summarizing, we have shown that
Term I = O T K 2 + K(log K) 3 2 log 1 δ , Term II = O T K 2 + K log 1 δ hold simultaneously with probability at least 1 -KT δ. Therefore, SReg ℓ (F, A) = O T K 2 + K(log K) 3 2 log 1 δ(6)
with probability at least 1 -KT δ. To bound E[SReg ℓ (F, A)], we let E be the event in (6). Therefore,
E[SReg ℓ (F, A)] = E[SReg ℓ (F, A)|E] • P(E) + E[SReg ℓ (F, A)| Ē] • P( Ē) = O T K 2 + K(log K) 3 2 log 1 δ + (K log K)T 2 δ = O T K 2 + K(log K) 3 2 log T + K log K = O(T 1 3 (log T )5
3 ), where the second equality follows by using the high probability bound on SReg ℓ (F, A) obtained in (6), and bounding E[SReg ℓ (F, A)| Ē] = O(T log K), which follows from (5); the third equality follows by choosing δ = 1 T 2 ; the final equality follows by choosing K = T 1 3 (log T ) 5 6
. This completes the proof.
Lemma 6. Fix a k ∈ N. Let {z i } K i=0 be a sequence where z 0 = 1, z i = sin 2 πi 2K for i = 1, . . . , K -1, and z K = 1. For each i = 1, . . . , K, define d i := z i -z i-1 . Then, the following holds: (a
) d i ≤ π 2K for all i ∈ [K]; (b) max 2 (di,di+1) zi(1-zi) = O 1 K 2 ; (c) K-1 i=1 1 zi(1-zi) = O(K 2 ); and (d) K-1 i=1 1 √ zi(1-zi) = O(K log K).
Proof. By direct computation, we have
z i -z i-1 = sin 2 πi 2K -sin 2 π(i -1) 2K = cos π(i-1) K -cos πi K 2 = sin π 2K sin π K i - 1 2 ,(7)
where the second equality follows from the identity sin 2 θ = 1-cos 2θ 2 , while the last equality follows from the identity cos α -cos β = 2 sin α+β 2 sin β-α 2 . Since sin θ ≤ θ for all θ ∈ R, and bounding sin θ ≤ 1, we obtain z i -z i-1 ≤ π 2K , which completes the proof for the first part of the lemma. For the second part, we note that
max 2 (d i , d i+1 ) z i (1 -z i ) = max 2 (d i , d i+1 ) sin 2 πi 2K cos 2 πi 2K = 4 • max 2 (d i , d i+1 ) sin 2 πi K ,
where the second equality follows from the identity sin 2θ = 2 sin θ cos θ. It follows from (7) that
max(d i , d i+1 ) = sin π 2K • max sin π K i - 1 2 , sin π K i + 1 2 .
For simplicity, we assume that K is odd, although a similar treatment can be done for even K. Let
1 ≤ i ≤ K-1 2 . Then, max(d i , d i+1 ) = sin π 2K sin π K i + 1 2 . Observe that sin π K i + 1 2 sin πi K = sin πi K cos π 2K + cos πi K sin π 2K sin πi K = cos π 2K + cot πi K sin π 2K ≤ 1 + sin π 2K sin π K ,
where the first equality follows from the identity sin(α + β) = sin α cos β + cos α sin β, while the inequality follows by noting that
cot πi K ≤ cot π K for all 1 ≤ i ≤ K-1 2 . Finally, since sin π 2K sin π K = 1 2 cos π 2K = O(1), we obtain max 2 (di,di+1) zi(1-zi) = O(sin 2 π 2K ) = O( 1 K 2 ).
Next, we consider the case when
K+1 2 ≤ i ≤ K -1. Then, max(d i , d i+1 ) = sin π 2K sin π K i -1 2 .
Repeating a similar analysis as before, we obtain
sin π K i -1 2 sin πi K = sin πi K cos π 2K -cos πi K sin π 2K sin πi K = cos π 2K -cot πi K sin π 2K ≤ 1 + sin π 2K sin π K , which is O(1) as claimed earlier. Therefore, max 2 (di,di+1) zi(1-zi) = O( 1 K 2 ).
Combining both the cases completes the proof of (b) above. For (c), similar to (b), we assume for simplicity that K is odd. Then,
K-1 i=1 1 z i (1 -z i ) = 4 K-1 i=1 1 sin 2 πi K = 8 K-1 2 i=1 1 sin 2 πi K ,
and the summation
K-1 2 i=1 1 sin 2 πi K can be bounded in the following manner: K-1 2 i=1 1 sin 2 πi K ≤ 1 sin 2 π K + K-1 2 1 1 sin 2 πν K dν ≤ 1 sin 2 π K + K 2 1 1 sin 2 πν K dν = 1 sin 2 π K + K π π 2 π K 1 sin 2 ν dν = 1 sin 2 π K + K π cot π K = O(K 2 ).
This completes the proof for (c). Repeating the exact same steps as (c) proves (d). We include the full proof for completeness. Observe that
K-1 i=1 1 z i (1 -z i ) = 2 K-1 i=1 1 sin πi K = 4 K-1 2 i=1 1 sin πi K ,
and the summation
K-1 2 i=1 1 sin πi K can be bounded in the following manner: K-1 2 i=1 1 sin πi K ≤ 1 sin π K + K-1 2 1 1 sin πν K dν ≤ 1 sin π K + K 2 1 1 sin πν K dν = 1 sin π K + K π π 2 π K 1 sin ν dν.
The integral above evaluates to log csc π K + cot π K . Therefore, we have that
K-1 i=1 1 z i (1 -z i ) ≤ 4 csc π K + K π log csc π K + cot π K = O(K log K).
This completes the proof.
this section cite: []

Section: C.2 Proof of Corollary 1
Proof. Let A be the algorithm guaranteed by Theorem 1. By Pinsker's inequality, we get that
A guarantees E[Cal 2 ] = O(T 1 3 (log T ) 5 3 ). Moreover, since Cal 1 ≤ √ T • Cal 2 (Kleinberg et al., 2023, Lemma 13), by Jensen's inequality we have E[Cal 1 ] ≤ T • E[Cal 2 ] = O(T 1 3 (log T ) 5 6
). Next, (Kleinberg et al., 2023, Theorem 12) states that for any proper loss ℓ, we have
SReg ℓ ≤ 4Cal 1 . Therefore, E[SReg ℓ ] ≤ 4E[Cal 1 ] = O(T 2 3 (log T ) 5 6
). Combining this with the result of Proposition 2, 3 completes the proof.
this section cite: []

Section: D Deferred proofs and discussion in Section 5

this section cite: []

Section: D.1 Computational cost of Algorithm 1
The cost of Algorithm 1 at every time step is at most O K 2 + INT + ST , where ST is the time required to compute the stationary distribution of Q t and INT denotes the computation required for evaluating the integral 1 0 wµt,i(w)dw 1 0 µt,i(w)dw in line 3 of Algorithm 3; the O(K 2 ) cost is incurred in forming the matrix Q t , and all other operations in Algorithm 1 can be carried out in time that is no worse than O(K 2 ). For ST, the stationary distribution of Q t can be computed by the method of power iteration; notably, each iteration shall incur cost O(nnz(Q t )), where nnz(Q t ) represents the number of non-zero entries in Q t . Since each column of Q t has at most two non-zero entries (Algorithm 4 randomizes over two adjacent points in the discretization), nnz(Q t ) = Θ(K). For INT, the integral is over [0, 1] and has a closed-form expression in terms of the gamma function Γ(z) :
= 1 0 exp(-t)t z-1 dt as derived below. Recall that f τ,i (w) = p τ,i ℓ(w, y τ ) = -p τ,i (y τ log w + (1 -y τ ) log(1 -w)) = log w -yτ pτ,i (1 -w) -pτ,i(1-yτ ) . Therefore, µ t,i (w) = exp - t-1 τ =1 f τ,i (w) = w t-1 τ =1 yτ pτ,i (1 -w) t-1 τ =1 pτ,i(1-yτ ) . For conve- nience, let γ := t-1 τ =1 y τ p τ,i , δ := t-1 τ =1 p τ,i (1 -y τ ). Then, 1 0 µ t,i (w)dw = 1 0 w γ (1 -w) δ dw = B(γ + 1, δ + 1), where B(z 1 , z 2 ) denotes the beta function, defined as B(z 1 , z 2 ) := 1 0 t z1-1 (1 - t) z2-1 dt. Since B(z 1 , z 2 ) = Γ(z1)Γ(z2)
Γ(z1+z2) for all z 1 , z 2 with z 1 , z 2 > 0, we have 1 0 µ t,i (w)dw = Γ(γ + 1)Γ(δ + 1) Γ(γ + δ + 2) .
Proof. Since Algorithm 1 ensures that PCal 2 = O T K 2 + K log T (refer Section 5), we obtain
Cal 2 = O T K 2 + K log T + K log K δ with probability at least 1 -δ, which is O T 1 3 (log T ) 1 3 log T δ on substituting K. The high probability bound on Msr L G follows since Msr L G ≤ G • Cal 2 . To bound E [Cal 2 ],
we let E denote the event that Cal 2 ≤ ∆, where ∆ := 6PCal 2 + 96(K + 1) log 4(K+1) δ . We then have,
E[Cal 2 ] = E[Cal 2 |E] • P(E) + E[Cal 2 | Ē] • P( Ē) = O T K 2 + K log T + K log K δ + δ • T which is O(T 1 3 (log T ) 2 3
) on substituting δ = 1 T and K. Note that the second equality above follows since E[Cal 2 |E] ≤ ∆ and P(E) ≤ 1, Cal 2 ≤ T and P( Ē) < δ. Finally, bounding
Msr L G ≤ G • Cal 2 finishes the proof.
Instantiating A Cal with the algorithm of Fishelson et al. (2025), we also obtain the exact same guarantee as Corollary 3. Compared to Algorithm 1, the algorithm of Fishelson et al. (2025) is more efficient since it uses scaled online gradient descent for the i-th external regret algorithm, which is more efficient than EWOO i . On the contrary, it does not posses the generality of Algorithm 1 towards minimizing SReg ℓ for all ℓ ∈ L 2 simultaneously.
this section cite: ['b5', 'b5']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: All relevant details related to claims made in the abstract and introduction are either provided in the main body or in the appendix.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Refer to Section 6. Guidelines:
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Assumptions are written in the main body and proofs are deferred to the appendix.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [NA] Justification: The paper is a theory work and does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper is a theory work and does not include experiments requiring code.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: The paper is a theory work and does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: The paper is a theory work and does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [NA]
Justification: The paper is a theory work and does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: The research abides in every respect with the Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA]
Justification: The paper is a theory work and there is no immediate societal impact of the work performed.
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
Justification: The paper is a theory work and poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA]
Justification: The paper is a theory work and does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper is a theory work and does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable
). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper is a theory work and does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The pape is a theory work and does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: References
Ref_id:b0 Title: An elementary predictor obtaining distance to calibration Year: (2025)
Ref_id:b1 Title: Contextual bandit algorithms with supervised learning guarantees Year: (2011)
Ref_id:b2 Title: From external to internal regret Year: (2007)
Ref_id:b3 Title: Prediction, learning, and games Year: (2006)
Ref_id:b4 Title: Improved bounds for calibration via stronger sign preservation games Year: (2024)
Ref_id:b5 Title: Full swap regret and discretized calibration Year: (2025)
Ref_id:b6 Title: Forecast hedging and calibration Year: (2021)
Ref_id:b7 Title: calibeating": Beating forecasters at their own game Year: (2023)
Ref_id:b8 Title: Oracle efficient online multicalibration and omniprediction Year: (2024)
Ref_id:b9 Title: Strictly proper scoring rules, prediction, and estimation Year: (2007)
Ref_id:b10 Title: Logarithmic regret algorithms for online convex optimization Year: (2007)
Ref_id:b11 Title: Predict to Minimize Swap Regret for All Payoff-Bounded Tasks Year: (2024)
Ref_id:b12 Title: U-calibration: Forecasting for an unknown agent Year: (2023)
Ref_id:b13 Title: Online isotonic regression Year: (2016)
Ref_id:b14 Title: Optimal multiclass u-calibration error and beyond Year: (2024)
Ref_id:b15 Title: Near-optimal algorithms for omniprediction Year: (2025)
Ref_id:b16 Title: Stronger calibration lower bounds via sidestepping Year: (2021)
Ref_id:b17 Title: On the distance from calibration in sequential prediction Year: (2024)
Ref_id:b18 Title: Learning the switching rate by discretising bernoulli sources online Year: (2009)
