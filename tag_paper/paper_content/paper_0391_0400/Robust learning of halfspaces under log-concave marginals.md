Title: Robust learning of halfspaces under log-concave marginals
Abstract: We say that a classifier is adversarially robust to perturbations of norm r if, with high probability over a point x drawn from the input distribution, there is no point within distance ≤ r from x that is classified differently. The boundary volume is the probability that a point falls within distance r of a point with a different label. This work studies the task of computationally efficient learning of hypotheses with small boundary volume, where the input is distributed as a subgaussian isotropic log-concave distribution over R d . Linear threshold functions are adversarially robust; they have boundary volume proportional to r. Such concept classes are efficiently learnable by polynomial regression, which produces a polynomial threshold function (PTF), but PTFs in general may have boundary volume Ω(1), even for r ≪ 1. We give an algorithm that agnostically learns linear threshold functions and returns a classifier with boundary volume O(r+ε) at radius of perturbation r. The time and sample complexity of d Õ(1/ε 2 ) matches the complexity of polynomial regression. Our algorithm augments the classic approach of polynomial regression with three additional steps: a) performing the ℓ 1 -error regression under noise sensitivity constraints, b) a structured partitioning and rounding step that returns a Boolean classifier with error opt + O(ε) and noise sensitivity O(r + ε) simultaneously, and c) a local corrector that "smooths" a function with low noise sensitivity into a function that is adversarially robust.

Section: Introduction
A predictor is robust to adversarial examples if for most possible inputs, a small perturbation will not cause the input to be misclassified. We define the boundary volume as the probability, over the input distribution, that a point is close to the boundary: In this work, we discuss computationally efficient learning of classifiers with optimal classification error and boundary volume, thus minimizing the robust risk.
Boundary D,
There is a large body of research on adversarial robustness in machine learning, the focus of which is to assess the robustness of classifiers commonly deployed in practice -see [BCM + 13, GSS14, SZS + 14], as well as [KM18] for an overview of the topic. It is well known that deep networks trained on classic benchmark data sets, such as ImageNet, can be "tricked" into misclassifying a test input by making a perturbation so small that it cannot be detected by humans, and that training robust models is a challenge in practice.
In this work we consider linear threshold functions (halfspaces), one of the most basic and wellknown concept classes. As a robustness benchmark, we consider the robustness of a proper learning algorithm -one that outputs a hypothesis that is a halfspace. For a halfspace, when the input is distributed according the d-dimensional standard Gaussian -or more generally, a subgaussian isotropic log-concave distribution -the probability that an input falls within Euclidean distance r of the classification boundary is O(r); thus a 1 -O(r) fraction of inputs are robust to adversarial perturbations of norm r. A proper agnostic learner would output a hypothesis with the following guarantees:
• Agnostic approximation: Pr (x,y)∼D [y ̸ = h(x)] ≤ opt + ε, where opt is the classification error of the best halfspace, and
• Adversarial robustness: Boundary D,r (h) ≤ O(r).
However, the state of the art for agnostically learning halfspaces is an improper algorithm -an algorithm that does not output a halfspace, but instead outputs a polynomial threshold function (PTF). Polynomial regression with randomized rounding learns halfspaces over N (0, I d ) with time and sample complexity d Õ(1/εfoot_0 ) and satisfies the accuracy guarantee [DGJ + 09, KKMS08]. In fact, the efficiency of polynomial regression is due to the inherent robustness of halfspaces -their small surface area! However, there exist degree-1/ε 2 PTFs with boundary volume Ω(1) even at a small radius of perturbation, 2 and this algorithm makes no guarantee that its output is robust.
No proper agnostic learning algorithm with 2 o(d) running time for general subgaussian isotropic log-concave distributions is known, and for the Gaussian distribution the state of the art for proper learning is d O(1/ε 4 ) [DKK + 21]. The complexity of proper learning in these settings is still an open question.We circumvent the possible difficulty of proper learning by giving an improper algorithm that still satisfies both the accuracy and robustness properties. Our algorithm has time and sample complexity d Õ(1/ε 2 ) , matching the run-time of the best (improper and non-robust) agnostic learning algorithm [DGJ + 09]. Up to polylogarithmic factors in the exponent, this also matches the statistical query lower bound for agnostically learning halfspaces [DKPZ21].
this section cite: ['b27', 'b16']

Section: Main result
Our main result is an improper agnostic learner (Algorithm 1) for halfspaces that outputs an adversarially robust hypothesis. Theorem 1.1 (Adversarially robust learning of halfspaces, informal version of Theorem 3.1). Let D be a distribution over R d × {-1, 1} such that the R d -marginal is subgaussian, isotropic, and log-concave. There is an algorithm that takes a robustness radius r, an accuracy parameter ε, and a sample of size d Õ(1/ε 2 ) from D. With high probability, it outputs a hypothesis h : R d → {-1, 1} with the following guarantees:
• Agnostic approximation: Pr (x,y)∼D [h(x) ̸ = y] ≤ opt + O(ε), where opt is the classification error of the best halfspace.
• Adversarial robustness: Boundary D,r (h) ≤ O(r).
this section cite: []

Section: Technical overview and intermediate results
Our algorithm and its analysis have three main components, which may be of independent interest. Two of the components solve a relaxation of the robust learning problem, and the third transforms the almost-robust hypothesis into a robust one. The relaxed learner produces a hypothesis with small noise sensitivity (a notion of boundary volume with random perturbations instead of adversarial perturbations) and small isolation probability (the probability that the local noise sensitivity around a point is very high). To formally define these two quantities, we need the following notion:
Definition (Local noise sensitivity). We define the local noise sensitivity ϕ f,η (x) at noise scale η as ϕ f,η (x) := E z∼N (0,I d )
[ 1 2 |f (x) -f (x + ηz)|].
We remark that when f is Boolean-valued, this is equivalently ϕ f,η (x) := Pr z∼N (0,I d )
[f (x) ̸ = f (x + ηz)].
We now define the noise sensitivity and the isolation probability:
Definition (Distributional noise sensitivity). The noise sensitivity of a function f at noise scale η on distribution D is defined as
NS D,η (f ) = E x∼D [ϕ f,η (x)].(1)
We remark that when D = N (0, I d ), our definition of distributional noise sensitivity differs from the standard definition of Gaussian noise sensitivity.
this section cite: []

Section: Definition (Isolation probability).
We call a point isolated if its local noise sensitivity is over a threshold. The isolation probability of a function f relative to a threshold t is
iso D,η (f, t) := Pr x∼D [ϕ f,η (x) > t].
Intuitively, a function can have low distributional noise sensitivity and still not be adversarially robust, if the classification boundary lies near many isolated points scattered throughout the domain. For this reason, after learning a hypothesis with low noise sensitivity, we apply the local correction algorithm which we describe below, which eliminates regions of high local noise sensitivity (isolated points).
this section cite: []

Section: Local correction of adversarial robustness (Algorithm 3)
The part of our algorithm that transforms an almost-robust hypothesis into a robust one is a local corrector for adversarial robustness. Local correctors are part of the family of local computation algorithms, or LCAs [RTVX11,ARVX12]: fast randomized algorithms that compute parts of a large object without constructing the object in its entirety. Local correctors, also known as local property reconstructors [ACCL08], are LCAs that evaluate queries to a nearby object that satisfies a desired property; in our case that object is an adversarially robust hypothesis. Our algorithm uses local correction in the fashion of [LRV22]: to "append" a local corrector with a fixed random seed to a non-robust hypothesis in order to globally correct the hypothesis.
We analyze a very simple LCA that makes queries to a function and outputs queries to a nearby function with reduced boundary volume. This algorithm estimates the local noise sensitivity at x, and changes the label of x if the local noise sensitivity is too high. This is, in essence, the smoothing procedure discussed in [LAG + 19, LCWC, CRK19]. We give guarantees on the boundary volume and the error introduced by the smoothing procedure in terms of the relaxed robustness properties of the input function (the noise sensitivity and isolation probability).
Lemma 1.2 (Local corrector for adversarial robustness, informal version of Theorem C.1). Let ε, α, β, r ∈ (0, 1). Let f : R d → {-1, 1} be a degree-k polynomial threshold function with noise sensitivity ≤ α and isolation probability ≤ β. There is an efficient randomized algorithm ROBUSTNESSLCA that makes black-box queries to f and answers black-box queries to a function g : R d → {-1, 1} that satisfies the following with high probability:
• Adversarial robustness: Boundary D,r ≤ O(α + ε).
• Small distance: Pr x∼D [g(x) ̸ = f (x)] ≤ O(β + ε).
We use this algorithm in a "deterministic" fashion, where all calls to the local corrector share the same random seed, which is good with high probability. When the random seed is good, the guarantees hold for all x ∈ R d . 3 This allows us to append the local corrector and its fixed seed to some degree-k PTF, creating a deterministic robust hypothesis that can be evaluated in d O(k) time.
From the guarantees of the local corrector and the desired guarantees of the overall algorithm, we determine that we must feed the local corrector a PTF of error opt + O(ε), noise sensitivity O(r), and isolation probability O(ε).
this section cite: ['b32', 'b4', 'b1', 'b29']

Section: Polynomial regression under noise sensitivity constraints (Algorithm 2)
Our first step in finding such a PTF is to learn a polynomial with low error, noise sensitivity, and isolation probability in the ℓ 1 -distance regime.
this section cite: []

Section: Theorem 1.3 (Learning a polynomial, informal version of Theorem 3.3).
There is an algorithm with time and sample complexity d Õ(1/ε 2 ) that takes samples from the distribution D and, with high probability, returns a degree-Õ(1/ε 2 ) polynomial p with the following properties:
• Accuracy: err(p) ≤ opt + O(ε),
• Low noise sensitivity: NS(p) ≤ O(r + ε), and
• Low isolation probability: iso(p) ≤ O(ε).
We achieve this with convex programming. The ℓ 1 noise sensitivity is a convex constraint, and we use a convex upper bound on the ℓ 1 analogue of the isolation probability. We minimize error over the set of degree-Õ(1/ε 2 ) polynomials under these constraints. To show that a feasible polynomial of error opt + O(ε) exists, we show that the halfspace-approximating polynomial given in [DGJ + 09] satisfies the constraints under any subgaussian isotropic log-concave distribution.
this section cite: []

Section: Randomized partitioning and rounding (Algorithm 5, Algorithm 4)
With p in hand, we must then round p to some polynomial threshold function sign(p -t) that satisfies the error, noise sensitivity, and isolation probability constraints. Simply rounding at a uniformly random threshold t ∼ [-1, 1], as in [KKMS08], does result in error opt + O(ε), noise sensitivity O(r), and isolation probability O(ε) in expectation, but doesn't guarantee that the conditions ever hold simultaneously for the same t. Thus, running the local corrector on just one rounded function would not guarantee a hypothesis with the right properties. In this step, we find a "deterministic mixture" of rounded functions that simultaneously satisfies the three conditions: a partition of the domain into parts, where each part is rounded at a different threshold. The corrector is run on each part separately.
A first attempt at finding such a partition would be to allow it to have O(1/ε 2 ) parts. Observe that the error, noise sensitivity, and isolation probability concentrate with deviation ε when averaging over O(1/ε 2 ) random thresholds. Thus, an equal-weighted mixture of O(1/ε 2 ) independent random roundings would satisfy all of the guarantees simultaneously with high probability. But suppose we partitioned the domain into 1/ε 2 sets of equal mass -we can't guarantee robustness for any point near the boundaries of these sets, and the total volume of these boundaries scales inversely with ε, which is undesirable. To minimize the increase in boundary volume due to partitioning, it is necessary to find a partition of constant size.
We show that there exists a mixture of four rounded functions satisfying all the constraints simultaneously by applying Carathéodory's theorem ( [Car07]), and we find the mixture by linear programming.
To understand how to turn the mixture into a partition, consider the example of handling just the error when y is a deterministic function of x. The error of the mixture is i∈[4]
w i • err(sign(p -t i )) ≤ opt + O(ε),
where w i is the mixing weight of the i th PTF. For each i in the mixture, there is a set of volume err(sign(p -t i )) consisting of the points misclassified by sign(p -t i ). We want to partition the domain such that a w i -fraction of this set falls into the i th part, so that
i∈[4]
1[x is in the i th part and y ̸ = sign(p(x) -t i )] = i∈[4]
w i • err(sign(p -t i )).
We cite a theorem of [DHV06], which says, essentially, that projecting on a random unit vector causes the set to be very close to normally-distributed with high probability. Thus, our partition is according to the inner product with a random unit vector u, and the parts are intervals J 1 , . . . , J 4 of Gaussian mass w 1 , . . . , w 4 respectively. With high probability, the sets of misclassified points and the sets for which robustness is guaranteed by the corrector are all partitioned with the correct weights. Theorem 1.4 (Randomized partitioning, informal version of Theorem 3.4). Let p be a polynomial satisfying the guarantees of Theorem 1.3. There is an algorithm that outputs a unit vector u, rounding thresholds t 1 , . . . , t 4 , and a partition of the real line into intervals J 1 , . . . J 4 such that the hypothesis
h(x) = 4 i=1 ROBUSTNESSLCA(sign(p(x) -t i )) • 1[⟨x, u⟩ ∈ J i ]
has the following properties with high probability:
• Accuracy: err(h) ≤ opt + O(ε),
• Robustness:
Boundary D,r (h) ≤ O(r + ε).
The time complexity is d Õ(1/ε 2 ) .
this section cite: ['b26', 'b7', 'b12']

Section: Verifiable robustness
The work of [GKVZ22] discusses the planting of "backdoors" in learned classifiers -structured violations of the robustness condition -and the impossibility of efficiently distinguishing a robust model from a backdoored one in the most general case. The takeaway is that when training is performed by an untrusted service, it is not generally possible to verify that a hypothesis is robust, even if a description of the hypothesis is available. However, our learning algorithm produces a hypothesis with a specific structure that can be checked, and the reduced expressivity of this structure allows robustness to be verifiable. Under complexity assumptions, the robustness of our algorithm's output can be verified in the following way: Corollary 1.5 (Deterministic verifiability, informal version of Corollary D.1). If P = BPP, then there is a learning algorithm B that, given access to labeled samples from a subgaussian isotropic log-concave distribution, runs in time d Õ(1/ε 2 ) and produces a hypothesis h with the following guaranteesfoot_2 :
• Agnostic approximation: h satisfies Pr (x,y)∼D [h(x) ̸ = y] ≤ opt + O(ε), where opt is the misclassification error of the best halfspace.
• Verifiable robustness: There is an efficient deterministic verifier that takes as input a hypothesis g and a point x ∈ R d , and always rejects if
∃z : ∥z∥ 2 ≤ r and g(x) ̸ = g(x + z).
If g is the output of B, then the verifier accepts with probability at least 1 -O(r + ε) over x drawn from D.
As a result, training can be done entirely by an untrusted service who claims to be using our learning algorithm, but may not be. The verifier checks whether the hypothesis matches the format our learning algorithm produces, rejects if it doesn't match, and performs a robustness test that is sound if it does match. The user must trust that the error of the hypothesis really is opt + O(ε), but does not have to place any trust in the service to guarantee robustness -there is a deterministic algorithm that the user can perform to verify that most points are not near the boundary. Furthermore, in a setting where the user performs some of the construction of the hypothesis (but need not access any training data), verifiable robustness can be made unconditional, but the soundness of the verifier can fail with small probability over the randomness of the hypothesis construction. See Appendix D for further discussion.
1.4 Related work.
Adversarially robust learning. Several recent papers study adversarially robust learning of various concept classes in the distribution-free PAC setting. These papers also generalize the set of adversarial perturbations beyond the ℓ 2 ball. Some consider ℓ p perturbations and some consider fully arbitrary sets.
The works of [CBM18,MHS19] show that there is essentially no statistical separation between robust learning and standard learning; thus any gap in the hardness of these tasks must be computational. The works of [BLPR19, DNV19, ADV19] exhibit concept classes and perturbation sets such that, under standard hardness assumptions, robust learning is computationally harder than standard learning. The work of [SHS + 18] exhibits some concept classes and perturbation sets such that a high robust risk is inevitable.
The work of [MGDS20] gives an algorithm for robustly learning halfspaces in the realizable setting where the data is assumed to be labeled by a halfspace with random classification noise. The works of [DKM20,DKM19] give algorithms for robust, proper, agnostic learning of halfspaces; however, the data distributions are assumed to be supported on the unit ball. When scaled up to match the parameters of our setting -distributions concentrated on a radius-√ d ball -the running times of these algorithms have an exponential dependence on
√ d.
The work of [ADV19] also studies robust learning in the distribution-free setting. They give an algorithm for learning halfspaces and degree-2 PTFs with robustness under ℓ ∞ -bounded perturbations.
In the distribution-specific setting, [GKKW21] gives an algorithm for learning monotone conjunctions with robustness to perturbations of O(log n) Hamming distance, under log-Lipschitz distributions on the Hamming cube. As mentioned earlier, the proper agnostic learner of [DKK + 21] is a robust learner for halfspaces under the Gaussian distribution.
Other work on learning halfspaces. See Section 1 for comparison of our work with [DGJ + 09, KKMS08]. Under general log-concave distributions, [ABL17] gives an algorithm for semiagnostic proper learning of origin-centered halfspaces under log-concave distributions, i.e. the algorithm gives a halfspace with error O(opt) + ε, where opt is the error of best halfspace. In contrast to this, our work focuses on obtaining the optimal opt + ε error. [Dan15] gives a poly-time method of achieving error (1 + α)opt + ε under for any constant α, but via an improper learning algorithm that uses polynomial regressionfoot_3 . The work of [DKK + 21], in addition to the d Õ(1/ε 4 ) -time algorithm for achieving error opt + ε for halfspaces under the Gaussian distribution (discussed in Section 1), gives a poly-time proper learning algorithm for origin-centered halfspaces that achieves error (1 + α)opt + ε (similar to [Dan15]). All algorithms in [DKK + 21] are highly specific to Gaussian distributions and do not extend to, for example, to the uniform distribution over [-1, 1] d and other sub-Gaussian log-concave distributions.
this section cite: ['b23', 'b8', 'b31', 'b30', 'b15', 'b14', 'b2', 'b22', 'b0', 'b10', 'b10']

Section: Preliminaries
When dealing with a distribution D over R d × {±1}, we will sometimes overload the notation to use the symbol D to refer to the R d -marginal of D. Analogously, when dealing with a collection S of pairs {(
x i , y i )} in R d × {±1}
we will sometimes write Pr x∼S [• • • ] to refer to Pr (x,y)∼S [• • • ] when the values of y are not referenced. 2.1 Perturbation and robustness models Definition 2.1 (r-boundary volume). The boundary volume of a function f at radius r on distribution D is defined as Boundary D,r (f ) = Pr x∼D [∃z : ∥z∥ 2 ≤ r and f (x) ̸ = f (x + z)]. Definition 2.2 (Local noise sensitivity). We define the local noise sensitivity ϕ f,η (x) at noise scale η as ϕ f,η (x) := E z∼N (0,I d )
[ 1 2 |f (x) -f (x + ηz)|].
We remark that when f is Boolean-valued, this is equivalently
ϕ f,η (x) := Pr z∼N (0,I d ) [f (x) ̸ = f (x + ηz)].
Definition 2.3 (Distributional noise sensitivity). The noise sensitivity of a function f at noise scale η on distribution D is defined as
NS D,η (f ) = E x∼D [ϕ f,η (x)].
Definition 2.4 (Isolation probability). We call a point isolated if its local noise sensitivity is over a threshold. The isolation probability of a function f relative to a threshold t is
iso D,η (f, t) := Pr x∼D [ϕ f,η (x) > t].
We also use a convex relaxation of the isolation indicator and define ψ to be its expectation:
ψ D,η (f ) := 10 E x∼D [1[ϕ f,η (x) > 0.6] • (ϕ f,η (x) -0.6)].
Note that ψ(f ) is an upper bound on iso(f, 0.7).
this section cite: []

Section: Noise sensitivity approximators
Below, we introduce the notion of a local noise sensitivity approximator. Many of the algorithms we introduce will use such an approximator as a black box they can query (see Algorithm 1, Algorithm 5, Algorithm 4 and Algorithm 3). We first introduce a local noise sensitivity approximator and some related notions, and present a randomized method for efficiently instantiating it. 6Definition 2.5 (Local noise sensitivity approximator for PTFs). For a degree parameter k, an algorithm φ is an ε-accurate noise sensitivity approximator if whenever it is given (i) a degree-k polynomial threshold function f , (ii) a noise rate η ∈ (0, 1), and (iii) a point x ∈ R d , it outputs a value φf,η (x) such that φf,η (x) -ϕ f,η (x) ≤ ε.
Definition 2.6. Let φ be as in the definition above. Then we define the empirical noise sensitivity and isolation probabilities as
NS D,η(
f ) := E x∼D φf,η (x) and iso D,η (f, t) := E x∼D 1[ φf,η (x) > t]. We also define ψD,η (f ) = 10 E x∼D [1[ φf,T,η (x) > 0.6] • (ϕ f,η (x) -0.6)]. 2.2 Distances and errors Definition 2.7 (Error and optimal error). We will denote the error of a hypothesis on a distribution err D (h) := E (x,y)∼D 1 2 |h(x) -y| and remark that when h is Boolean-valued this is equivalently err D (h) := Pr (x,y)∼D [h(x) ̸ = y]. Relative to a sample set we use err S (h) = 1 |S| (x,y)∈S 1 2 |h(x) -y|.
We will often denote by opt the optimal error of a function f with respect to a concept class F:
opt := inf g∈F err D (g).
In this work, F is always the class of linear threshold functions over R d .
this section cite: []

Section: Miscellaneous
We will use the notation of the form a = b ± c as a shorthand for |a -b| ≤ c. We will also drop subscripts, particularly on ϕ and φ, when they are clear from context. Throughout this work, wherever there is noise, the noise scale η is always 10r, where r is the desired robustness radius. We use the notation t ∼ [-1, 1] to denote that t is drawn uniformly from [-1, 1].
this section cite: []

Section: Results and pseudocode of our main algorithm
Our main result is the following. It states the correctness and running time of the main algorithm, ROBUSTLEARN.
Theorem 3.1 (Correctness and complexity of Algorithm 1). Let ε ≥ d -1/7 , let r, δ ∈ (0, 1), and let C be a sufficiently large absolute constant. Furthermore, let φ be an ε-accurate local noise sensitivity approximator for degree-k PTFs over R d , where
k := C log 2 (1/ε) ε 2
, and let the running time and query complexity of φ be τ .
Let F be the class of linear threshold functions over R d and D be a distribution over R d × {-1, 1} whose R d -marginal is isotropic, subgaussian and log-concave. The algorithm ROBUSTLEARN, given i.i.d. sample access to the distribution D, with probability at least 1 -O(δ) returns a hypothesis h : R d → {±1} for which the following hold:
err D (h) ≤ opt + O(ε) Boundary D,r (h) ≤ O(r + ε).
this section cite: []

Section: The running time and sample complexity are
poly d log 2 (1/ε)/ε 2 • log(1/δ) • τ .
Corollary 3.2. By instantiating φ with the algorithm of Appendix A.1, which evaluates queries in poly d log 2 (1/ε)/ε 2 • log(1/δ) time, ROBUSTLEARN runs in total time
poly d log 2 (1/ε)/ε 2 • log(1/δ) .
Algorithm 1 ROBUSTLEARN(D, ε, δ, r):
1: Input: sample access to distribution D over R d × {±1}, error bound ε, 2: confidence bound δ, robustness radius r 3: Uses: local noise sensitivity approximator φ.
(See Definition 2.5) 4: Output: hypothesis h : R d → {-1, 1} 5: p := LEARNREALVALUED(D, ε, r).
(See Algorithm 2) 6: return COMPUTECLASSIFIER(D, p, r, ε, δ).
(See Algorithm 5)
The algorithm has two "phases": LEARNREALVALUED, which solves a convex relaxation of the learning task, and COMPUTECLASSIFIER, which rounds the real-valued hypothesis to a Boolean one with small boundary volume. We present and analyze these algorithms in Appendix B.3 and Appendix C.3 respectively.
Theorem 3.3 (Correctness of Algorithm 2, LEARNREALVALUED). Let r ∈ (0, 1), and let D be a distribution over R d × {±1} whose R d -marginal is a subgaussian isotropic log-concave distribution over R d . Let F be the class of halfspaces on R d . Then, for some sufficiently large absolute constant C, the algorithm LearnRealValued, given ε, δ and sample access to D, runs in time poly d O(log 2 (1/ε)/ε 2 ) and will with probability at least 1 -O(δ) return a degree-O(log 2 (1/ε)/ε 2 ) polynomial p for which the following hold:
E t∼[-1,1] E (x,y)∼D ϕ sign(p-t),10r (x) ≤ 100r + O(ε),(2)
Pr (x,y)∼D E t∼[-1,1] [ϕ sign(p(x)-t),10r (x)] ≥ 0.7 ≤ O(ε),(3)
E t∼[-1,1] Pr (x,y)∼D [sign(p(x) -t) ̸ = y] ≤ opt + O(ε).(4)
Theorem 3.4 (Correctness of Algorithm 5, COMPUTECLASSIFIER). Let ε ≥ d -1/7 and δ, r > 0.
Let φ be an ε-accurate local noise sensitivity approximator for degree-k PTFs over R d . Let D be a distribution over R d × {-1, 1} such that the R d -marginal is isotropic and log-concave. Let p : R d → R be a polynomial of degree at most k satisfying
E t∼[-1,1] E (x,y)∼D [ϕ sign(p-t),10r (x)] ≤ 100r + O(ε),(5)
Pr (x,y)∼D E t∼[-1,1] [ϕ sign(p(x)-t),10r (x)] ≥ 0.7 ≤ O(ε).(6)
Then the algorithm COMPUTECLASSIFIER(D, p, r, ε, δ), given sample access to D and query access to φ, outputs a hypothesis h : R d → {-1, 1} such that the following properties hold with probability at least 1 -O(δ):
err D (h) ≤ E t∼[-1,1]
[err D (sign(p -t))] + O(ε)
Boundary D,r (h) ≤ O(r + ε).
The running time and number of queries to φ are poly(
d k • 1/ε • log(1/δ)).
this section cite: []

Section: Proof of Theorem 3.1.
In this section we prove the main theorem assuming Theorem 3.4 and Theorem 3.3.
Proof of Theorem 3.1. By Theorem 3.3, with probability 1 -O(δ), the output p of LEARNREAL-VALUED is a degree-k polynomial satisfying
E t∼[-1,1] E (x,y)∼D ϕ sign(p-t),10r (x) ≤ 100r + O(ε),(7)
Pr (x,y)∼D E t∼[-1,1] [ϕ sign(p(x)-t),10r (x)] ≥ 0.7 ≤ O(ε),(8)
E t∼[-1,1] Pr (x,y)∼D [sign(p(x) -t) ̸ = y] ≤ opt + O(ε).(9)
This polynomial p is the input to COMPUTECLASSIFIER. We see that the premises of Theorem 3.4 are satisfied by the guarantees of Equation (7) and Equation (8). Taking the conclusions of Theorem 3.4 and combining it with Equation (9) we see that the hypothesis h we return satisfies
err D (h) ≤ E t∼[-1,1] [err D (sign(p -t))] + O(ε) ≤ opt + O(ε),
Boundary D,r (h) ≤ O(r + ε),
which concludes the proof of correctness.
We now analyze the sample complexity and running time. By Theorem 3.3, LEARNREALVALUED takes d O(log 2 (1/ε)/ε 2 ) poly(log(1/δ)) time and samples. By Theorem 3.4, since the degree of p is O(log 2 (1/ε)/ε 2 ) and ε ≥ d -1/7 , COMPUTECLASSIFIER takes d O(log 2 (1/ε)/ε 2 ) poly(log(1/δ)) time and queries to φ. Evaluations of φ take time τ , so the total running time is
poly(d log 2 (1/ε)/ε 2 ) • log(1/δ) • τ ),
and all components succeed with probability ≥ 1 -O(δ).
this section cite: []

Section: A Further preliminaries

this section cite: []

Section: A.1 Randomized implementation of φ
Let T be a sufficiently large set of i.i.d. samples from N (0, I d ). Below, we show that the choice of φ as the following empirical estimate will satisfy Definition 2.5 with high probability over the choice of
T : φf,T,η (x) = 1 |T | z∈T |f (x) -f (x + ηz)| 2 .
this section cite: []

Section: Fact A.1 ([Ant95]).
The VC dimension of the class of polynomial threshold functions of degree k is Θ( d ≤k ) = O(d k ). Fact A.2 (Concentration of expectations from VC dimension). Let C be a concept class and D be a distribution over R d . For some sufficiently large absolute constant C, the following is true. With probability at least 1-δ over a sample S of i.i.d. samples from D, with |S| ≥ C •V C(C) log(1/δ)/ε 2 , the following holds:
sup f ∈C E x∼D [f (x)] - 1 |T | x∈T f (x) ≤ ε.
We will call a sample set "representative" of distribution D for a concept class if it satisfies the above success condition. Claim A.3 (Representative samples provide accurate noise sensitivity estimates). Let C be the class of degree-k PTFs. Let T be a set of points in R d that is representative for N (0, I d ) for C (in the sense of Fact A.2). Then with probability ≥ 1 -δ over T , the following holds for all f ∈ C, x ∈ R d , and η ∈ (0, 1): φf,T,η (x) -ϕ f,η (x) ≤ ε.
Proof. Observe that for any fixed x, the function g(z) = -f (x) • f (x + ηz) is a member of C. Thus by Fact A.2, we have with probability 1 -δ that
E z∼T [g(z)] - E z∼N (0,I d ) [g(z)] ≤ ε.
The claim follows from observing that ϕ f,r (x) = 1 2 (E[g(z)] + 1), and thus
φf,T,η (x) -ϕ f,η (x) = 1 2 E z∼T [g(z)] - E z∼N (0,I d ) [g(z)] ≤ ε.
this section cite: []

Section: A.2 Distances and errors
The following facts about the error and local noise sensitivity after rounding a real-valued function to Boolean follow directly from linearity of expectation.
Fact A.4 (Randomized rounding). Let f and g be real-valued functions. Let the rounded function f t be defined as f t (x) = sign(f (x) -t),
and likewise for g t . Then,
E t∼[-1,1] Pr (x,y)∼D [f t (x) ̸ = y] ≤ E (x,y)∼D |f (x) -y| 2 . E t∼[-1,1] Pr (x,y)∼D [f t (x) ̸ = g t (x)] ≤ E (x,y)∼D |f (x) -g(x)| 2 . Similarly, 7 E t∼[-1,1] [ϕ ft,η (x)] ≤ ϕ f,η (x) and E t∼[-1,1]
[ φft,T,η (x)] ≤ φf,T,η (x)
for any x, η and T .
Finally, the following is a standard fact, see for example [DMR18].
Fact A.5 (TV distance between one-dimensional Gaussians). For all µ 1 , µ 2 in R and σ 1 , σ 2 in R >0 we have
d TV (N (µ 1 , σ 2 1 ), N (µ 2 , σ 2 2 )) ≤ 3|σ 2 1 -σ 2 2 | 2σ 2 1 + |µ 1 -µ 2 | 2σ 1 7 See Appendix A.1 for the definition of φ.
this section cite: ['b18']

Section: A.3 Miscellaneous
We say that f is a randomized Boolean function over R d if f maps every point in R d to a random variable f (x) over {0, 1}. Sometimes we will write f (x) = Ber(p), to denote that f (x) = 1 with probability p and 0 with probability 1 -p. In Appendix E, miscellaneous claims about uniform convergence are shown, which are used in the analysis of Algorithm 2.
this section cite: []

Section: B Pseudocode and analysis of LearnRealValued
In this section we present and analyze our algorithm for learning a polynomial with small noise sensitivity and isolation probability.
Algorithm 2 LEARNREALVALUED(D, ε, r):
1: Input: Sample access to distribution D, error bound ε, robustness radius r 2: Output:
hypothesis h : {-1, 1} d → R 3: Set k := C log 2 (1/ε) ε 2 . 4: for i in {1, • • • , log(1/δ)} do 5: S i := {Cd Ck • log(1/δ)/ε 2 i.i.d. draws from D}.
6:
T i := {Cd Ck • log(1/δ)/ε 2 i.i.d. draws from N (0, I d )} 7:
Let p i be the output of the following convex program: • Domain: polynomials over R d of degree k.
• Minimize: (x,y)∈S |p(x) -y| • Constraints:
-Define φ(x) p,T,10r := 1
|T | z∈T |f (x)-f (x+ηz)| 2 -Define ψp,T,10r := 10 • 1[ φ(x) p,T,10r > 0.6] • ( φ(x) p,T,10r -0.6) -Constraint 1: 1 |S| x∈S φp,T,10r (x) ≤ 100r + ε -Constraint 2: 1 |S| x∈S ψp,T,10r (x) ≤ ε 8: i * := argmin i E (x,y)∼Si |pi(x)-y| 2 9: p := p i * . 10: return p B.1 Facts about log-concave distributions. Definition B.1. A distribution D over R d is is called C-sub-Gaussian, if for any unit vector u: Pr x∼D [|u • x| > t] ≤ exp(-Ct 2 ) for any t ≥ 0. We say D is sub-Gaussian, if D is C-sub-Gaussian for some absolute constant C.
See e.g. [SW14] for the following fact: Fact B.1. If D 1 and D 2 are log-concave distributions over R d , then the convolution of D 1 and D 2 is also a log-concave distribution over R d .
Likewise, the following fact follows directly from the definition of a sub-Gaussian distribution: Fact B.2. If D 1 and D 2 are sub-Gaussian distributions over R d , then the convolution of D 1 and D 2 is also a sub-Gaussian distribution over R d .
The following fact can be found in [SW14]: Fact B.3. If a distribution D over R d is isotropic log-concave, then the projection of D on any linear subspace in R d is likewise log-concave.
The following two facts can be found in [GKK23], [ABL17] and the references therein. Fact B.4. If a distribution D over R d is isotropic log-concave, then for some absolute constant C 3 , for any unit vector u and every interval J ⊂ R, we have Pr x∼D [⟨u, x⟩ ∈ J] ≤ |J|. Remark B.5. It follows from a direct change-of-basis argument that Fact B.4 still holds if the premise that the distribution D is isotropic 8 is replaced by a slightly more general premise that D mean zero and satisfies:
I d×d ⪯ E x∼D [xx T ] ⪯ 200I d×d
For the following fact, see e.g. [DKS18] and the references therein: Fact B.6. If a distribution D over R d is log-concave, then for some absolute constant C 4 , if p is a degree-k polynomial for which E x∼D [(p(x)) 2 ] ≤ 1, then for any B > e k it is the case that
Pr x∼D [(p(x)) 2 > B] ≤ exp(-C 4 B 1/k )
One can use the fact above to conclude the following: Observation B.7. If a distribution D over R d is log-concave, then for some absolute constant C 5 , for every degree-k polynomial p it is the case that
E x∼D [|p(x)|] ≥ E x∼D [(p(x)) 2 ] (C 5 k) C5k
Proof. Without loss of generality, we can assume that E x∼D [(p(x)) 2 ] = 1. For any B ≥ 0, we have
E x∼D [|p(x)|] ≥ E x∼D [|p(x)| 2 1 |p(x)|≤B ] B = 1 -E x∼D [|p(x)| 2 1 |p(x)|>B ] B (10
)
Using Fact B.6, we have
E x∼D [|p(x)| 3 ] ≤ e O(k) + +∞ β=0 Pr x∼D [|p(x)| 3 > β]d β ≤ e O(k) + +∞ β ′ =0 (β ′ ) 3k/2-1 exp(-C 4 β ′ )d β ′ ≤ (O(k)) O(k) ,
which allows us to conclude that
E x∼D [|p(x)| 2 1 |p(x)|>B ] ≤ E x∼D [|p(x)| 3 ] B = (O(k)) O(k) B .
Taking B to equal (C ′ k) C ′ k for a sufficiently large absolute constant C ′ , and substituting the bound above into Equation (10) finishes the proof.
this section cite: ['b34', 'b34', 'b21', 'b0', 'b17']

Section: B.2 Approximating halfspaces with polynomials
Claim B.8. Let h be a linear threshold function over R d . Then, for every absolute constant C, there exists a polynomial p of degree O(1/ε 2 log 2 (1/ε)), such that for any C-sub-Gaussian log-concave
distribution D over R d satisfying I d×d ⪯ E x∼D [xx T ] ⪯ 200I d×d we have E x∼D [|h(x) -p(x)|] ≤ O(ε) Remark B.9.
Recall that a distribution D is log-concave if the logarithm of the probability density function of D is concave. Note that the constants hidden by the O(•) notation in Claim B.8 are allowed to depend on the sub-Gaussianity constant C.
To prove the claim above, we will need the following proposition from [DGJ + 09]: Fact B.10 ([DGJ + 09]). There exist absolute constants C 4 and C 5 , such that for every ε ∈ (0, 0.1), there exists a univariate polynomial P satisfying the following. Denoting a :=
ε 2 C4 log(1/ε) and K := C 5 log 1 ε a = O log 2 1/ε ε 2
, it is the case that • The degree of P is at most K.
• P (t) ∈ [sign(t), sign(t) + ε] for all t in [-1/2, -2a] ∪ [0, 1/2]. 8 Recall that the condition that D is isotropic means that D has zero mean and covariance I d×d .
• P (t) ∈ [-1, 1 + ε] for t in [-2a, 0]
• |P (t)| ≤ 2 • (4t) K for all |t| ≥ 1/2.
The proof below follows closely the line of reasoning in [DGJ + 09] and is included for completeness.
Proof of Claim B.8. For a unit vector u h(x) = sign(u • x -τ ) (assuming τ = 0 for now) If |τ | > log(1/ε) the statement follows immediately by taking P to be the constant polynomial that equals to sign(τ ) everywhere in R d . The error between h and P will be upper-bounded by O(ε) via Definition B.1 and Remark B.5.
Otherwise, we have |τ | ≤ log(1/ε). Let P be the polynomial in Fact B.10 and take w :=
C 6 log 2 1/ε ε 2 1/2
for a sufficiently large absolute constant C 6 . Denoting by D projected the projection of distribution D on the unit vector u, we can bound the error as follows using Fact B.10:
E x∼D [|sign(u • x -τ ) -P ((u • x -τ )/w)|] = E z∼Dprojected [|sign(z -τ ) -P ((z -τ )/w)|] ≤ ε+2 Pr z∼Dprojected [-2a ≤ z/w-τ ≤ 0]+E z∼Dprojected 2 • 4 • z -τ w K + 1 1 |(z-τ )/w|≥1/2 (11
)
The second term above is bounded by O(aw) due to Fact B.4 and Remark B.5. We also note that since |τ | ≤ log(1/ε) and w := C 6
log 2 1/ε ε 2 1/2
, for sufficiently large absolute constant C 6 we gave |τ |/w ≤ 0.1. Therefore, whenever (z -τ )/w| ≥ 1/2 we also have z ≥ 0.4w. This allows us to upper bound
E x∼D [|sign(u • x -τ ) -P ((u • x -τ )/w)|] ≤ ε + O(aw) + E z∼Dprojected 2 • 4 • z w + 0.1 K + 1 1 |z|≥0.4w ≤ ε + O(aw) + E z∼Dprojected 3 • 5 • z w K 1 |z|≥0.4w (12
)
Decomposing the interval [0.4, ∞] into a union ∞ i=0 [0.4 • 2 i , 0.4 • 2 i+1 ] and using Definition B.1 and Remark B.5 we can now upper-bound the last term above as follows:
E z∼Dprojected 3 • 5 • z w K 1 |z|≥0.4w ≤ 2 ∞ i=0 3 • (4 • 2 i ) K Pr z∼Dprojected |z| ≥ 2 i-1 • 0.4w ≤ 6 ∞ i=0 (4 • 2 i ) K exp(-C(2 i-3 w) 2 ) (13
)
Substituting K = O log 2 1/ε ε 2 and w = C 6 log 2 1/ε ε 2 1/2
, we can bound each term in the sum above
(4 • 2 i ) K exp(-C(2 i-3 w) 2 ) ≤ exp (i + 2) • O log 2 1/ε ε 2 -C • C 6 2 i-3 log 2 1/ε ε 2
For a sufficiently large absolute constant C 6 the above is at most ε/2 i+1 , which substituted into Equation (13) gives us
E z∼Dprojected 3 • 5 • z w K 1 |z|≥0.4w ≤ ε.
Finally, substituting the inequality above into Equation (12), we get
E x∼D [|sign(u • x) -P (u • x/w)|] = O(ε),
finishing the proof.
this section cite: []

Section: B.3 Analyzing LearnRealValued.
Theorem 3.3 (Correctness of Algorithm 2, LEARNREALVALUED). Let r ∈ (0, 1), and let D be a distribution over R d × {±1} whose R d -marginal is a subgaussian isotropic log-concave distribution over R d . Let F be the class of halfspaces on R d . Then, for some sufficiently large absolute constant C, the algorithm LearnRealValued, given ε, δ and sample access to D, runs in time poly d O(log 2 (1/ε)/ε 2 ) and will with probability at least 1 -O(δ) return a degree-O(log 2 (1/ε)/ε 2 ) polynomial p for which the following hold:
E t∼[-1,1] E (x,y)∼D ϕ sign(p-t),10r (x) ≤ 100r + O(ε),(2)
Pr (x,y)∼D E t∼[-1,1] [ϕ sign(p(x)-t),10r (x)] ≥ 0.7 ≤ O(ε),(3)
E t∼[-1,1] Pr (x,y)∼D [sign(p(x) -t) ̸ = y] ≤ opt + O(ε).(4)
We first argue that the run-time is indeed d O(log 2 (1ε)/ε 2 . The algorithm operates with polynomials of degree k = O(log 2 (1ε)/ε 2 over R d , which are described via d O (k) coefficients. It remains to show that the constraints for the optimization task are indeed convex in the coefficients in the polynomial p.
Recall that the constraints are:
• Constraint 1: 1 |S| x∈S φp,T,10r (x) ≤ 100r + ε • Constraint 2: 1 |S| x∈S ψp,T,10r (x) ≤ ε
To see that these constraints are indeed convex, first note that, by definition of φp,T,10r (x) and the triangle inequality, we have for any pair of polynomials p 1 , p 2 , x in R d and α ∈ [0, 1] φ(αp1+(1-α)p2),T,10r (x) ≤ α φp1,T,10r (x) + (1 -α) φp2,T,10r (x),
which implies that Constraint 1 is convex. Likewise, reclalling the definition of ψ ψp,T,10r := 10 • 1[ φ(x) p,T,10r > 0.6] • ( φ(x) p,T,10r -0.6), and combining it with Equation ( 14) we see that
ψαp1+(1-α)p2,T,10r (x) ≤ α ψp1,T,10r (x)(1 -α) ψp2,T,10r (x).
We now proceed to the proof of correctness, the following claim will be key to our argument, which will be proven in the next subsection (Appendix B.4): Claim B.11. For any specific iteration i of the algorithm, the polynomial p i will satisfy the following:
E (x,y)∼Si φpi,Ti,10r (x) ≤ 100r + O(ε),(15)
E (x,y)∼Si ψpi,Ti,10r (x) ≤ O(ε),(16)
and the following holds with probability at least 0.9:
E (x,y)∼Si |p i (x) -y| 2 ≤ min f ∈F Pr (x,y)∼D [f (x) ̸ = y] + O(ε),(17)
We now prove Theorem 3.3 assuming Claim B.11. Proof of Theorem 3.3. First we note that for sufficiently large value of the absolute constant C, with probability at least 1 -O(δ) for all i in {1, • • • , log 1/δ}, the set S i will satisfy Claim E.2, Claim E.3 and Fact E.5, and the set T i will satisfy Claim E.2. Furthermore, since the main iteration is repeated O(log 1/δ) times, the we see that with probability 1 -O(δ) the polynomial p = p i * will satisfy Equations 15, 16 and 17. Denoting S = S i * and T = T i * , we have E t∼[-1,1] E (x,y)∼D ϕ sign(p-t),10r (x) Since S satisfies Claim E.2 ≤ E t∼[-1,1] E (x,y)∼S ϕ sign(p-t),10r (x) + O(ε) ≤ ≤ E t∼[-1,1] E (x,y)∼S φsign(p-t),T,10r (x) + O(ε) Since T satisfies Claim A.3, and WLOG ε < 0.02. ≤ E (x,y)∼S φp,T,10r (x) + O(ε) By Fact A.4 ≤ 100r + O(ε) By Equation (15) , which gives us Equation (2). We now derive Equation (3) as follows: Pr (x,y)∼D E t∈[-1,1] [ϕ sign(p(x)-t),10r ] ≥ 0.7 Since S satisfies Claim E.3 ≤ Pr (x,y)∼S [E t∈[-1,1] [ϕ sign(p-t),10r (x)] ≥ 0.67] + O(ε) ≤ ≤ Pr (x,y)∼S [E t∼[-1,1] [ φsign(p-t),T,10r (x)] ≥ 0.65] + O(ε) Since T satisfies Claim A.3, and WLOG ε < 0.02. ≤ O(1) E (x,y)∼S ψp,T,10r (x) + O(ε). By Fact A.4 and definition of ψ (Section 2.1) ≤ O(ε) By Equation (16) Finally, we derive Equation (4) as follows: E t∼[-1,1] [err D [sign(p -t)]] Since S satisfies Fact E.5 ≤ E t∼[-1,1] [ err S [sign(p -t)]] + O(ε) ≤ ≤ E (x,y)∼S |p(x) -y| 2 + O(ε) By Fact A.4 ≤ opt + O(ε) By Equation (17) and defn of opt . B.4 Proving Claim B.11 Finally, we finish this section by proving Claim B.11. Inspecting the algorithm LearnRealValued, we see that the linear program is always feasible for any input dataset S i , because the all-zeros polynomial will satisfy the constraints in the linear program (i.e. Equations 15 and 16). It remains to show that with probability at least 0.9 there exists a polynomial p i satisfying all three of Equations 15, 16 and 17. Suppose f * is the optimal halfspace for D, i.e. Pr (x,y)∼D [f * (x) ̸ = y] = min f ∈F Pr (x,y)∼D [f (x) ̸ = y] .
Let D marginal denote the R d -marginal of D, and let D ′ marginal denote distribution of x + 10rz, where x is sampled from D marginal and z is sampled from N (0, I d ). Since D marginal is assumed to be sub-Gaussian log-concave, Fact B.1 and Fact B.2 tell us that D ′ marginal is also sub-Gaussian log-concave. Furthermore, since D marginal is isotropic (i.e. has mean zero and identity convariance) and r is in (0, 1) we see that
I d ⪯ E x∼D ′ marginal [xx T ] ⪯ 101I d .
Therefore, Claim B.8 tells us that for a sufficiently large absolute constant C ′ there is a polynomial of p * of degree C ′ log 2 (1/ε)
ε 2 for which E x∼Dmarginal [|f * (x) -p * (x)|] ≤ O(ε) (19) E x∼D ′ marginal [|f * (x) -p * (x)|] ≤ O(ε) (20
)
Recall that for each element (x, y) in S i , the variable x is distributed according to D marginal . Similarly, for each (x, y) in S i and z in T i the sum x + 10rz is distributed D ′ marginal . This allows us to use inequalities above to conclude that with probability at least 0.99 we have
E (x,y)∼Si [|p * (x) -f * (x)|] ≤ O(ε) (21) E (x,y)∼Si,z∼Ti [|p * (x + 10rz) -f * (x + 10rz)|] ≤ O(ε)(22)
In the remainder of this section we show that with probability at least 0.9 over the choice of S i and T i the choice p i = p * will indeed satisfy Equations 15, 16 and 17.
this section cite: []

Section: B.4.1 Bounding E x∼D marginal [ϕ f * ,10r (x)] and E x∼D marginal [ψ f * (x)]
Recall that f * (x) = sign(u • x -τ ) for some unit vector u and real τ . Here, we show that f * satisfies the following two inequalities
E x∼Dmarginal [ϕ f * ,10r (x)] ≤ 60r,(23)
E x∼Dmarginal [ψ f * (x)] = 0. (24) Indeed, fix a specific value of x. Recall that ϕ f * ,10r (x) is the probability that sign(u • x -τ ) ̸ = sign(u•(x+10rz)-τ ) where z is sampled from N (0, I d ). With probability at least 0.5 over the choice of Z we have sign(u • z) = sign(u • x -τ ) which leands to sign(u • x -τ ) ̸ = sign(u • (x + 10rz) -τ ). Thus, ϕ f * ,10r (x) ≤ 0.5 and substituting this into the definition of ψ we see that ψ(x) = 0, which proves Equation (24).
Now consider points x in R d such that |u • x -τ | is in the interval [10r • (i -1), 10r • i]
Pr z∼N (0,I d ) [sign(u•x-τ ) ̸ = sign(u•(x+10rz)-τ )] ≤ max 1, 10r 10r(i -1) 2 = max 1, 1 i -1 2
Combining the two observations above, we see that We observe that
E x∼Dmarginal [ϕ f * ,10r (x)] ≤ ∞ i=0 20r max 1, 1 i -1 2 ≤ 60r,
E (x,y)∼Si [ φp * ,Ti (x)] = E (x,y)∼Si z∼Ti p * (x) -p * (x + 10rz) 2 ≤ E (x,y)∼Si z∼Ti f * (x) -f * (x + 10rz) 2 +E (x,y)∼Si p * (x) -f * (x) 2 +E (x,y)∼Si z∼Ti p * (x + 10rz) -f * (x + 10rz) 2 ≤ E (x,y)∼Si [ φf * ,Ti,10r (x)]+E (x,y)∼Si |p * (x) -f * (x)| + E (x,y)∼Si,z∼Ti |p * (x + 10rz) -f * (x + 10rz)| =O(
ε) by Equations 21 and 22 .
Analogously, by inspecting the definition of ψ, we see that for any x we have
ψp * ,Ti (x) -ψf * ,Ti (x) ≤ |f * (x) -p * (x)| + O(1) • E z∼Ti [|f * (x + 10z) -p * (x + 10z)|]]
Averaging over x in S i , we have
E (x,y)∼Si [ ψp * ,Ti (x)] ≤ E (x,y)∼Si [ φf * ,Ti,10r (x)]+O(1)• E (x,y)∼Si |p * (x) -f * (x)| + E (x,y)∼Si,z∼Ti |p * (x + 10rz) -f * (x + 10rz)|
Recall that f * is a {±1}-valued function, which implies that φf * ,Ti,10r (x + 10rz) is in [0, 1] for all x and z. This allows us to use the Chebyshev's inequality to conclude that:
E Ti φf * ,Ti,10r (x) -ϕ f * ,10r (x) ≤ O1
|T i | .
Inspecting the definition of ψ, we see that for any x it is the case that
ψf * ,Ti (x) -ψ f * (x) ≤ O(1) • φf * ,Ti,10r (x) -ϕ f * ,10r (x),
which implies that
E Ti ψf * ,Ti (x) -ψ f * (x) ≤ O1
|T i | Averaging the inequalities above over x in S i , we have:
E Ti E (x,y)∼Si φf * ,Ti,10r (x) -ϕ f * ,10r (x) ≤ O1
|T i | E Ti E (x,y)∼Si ψf * ,Ti (x) -ψ f * (x) ≤ O1
|T i | Thus, with probability at least 0.99 over the choice of T i , we have
E (x,y)∼Si φf * ,Ti,10r (x) -E (x,y)∼Si [ϕ f * ,10r (x)] ≤ E (x,y)∼Si φf * ,Ti,10r (x) -ϕ f * ,10r (x) ≤ O1
|T i | (27) E (x,y)∼Si ψf * ,Ti (x) -E (x,y)∼Si [ψ f * (x)] ≤ E (x,y)∼Si ψf * ,Ti (x) -ψ f * (x) ≤ O 1 |T i |(28)
Again, recalling that f * is a {±1}-valued function, we see that ϕ f * ,10r (x) is in [0, 1] for all x and ψ f * (x) is likewise bounded by O(1) in absolute value. This allows us to use the Chebyshev's inequality to conclude that with probability at least 0.99 over the choice of S i we have:
E (x,y)∼Si [ϕ f * ,10r (x)] -E x∼Dmarginal [ϕ f * ,10r (x)] ≤ O 1 |S i | (29
)
E (x,y)∼Si [ψ f * (x)] -E x∼Dmarginal [ψ f * (x)] ≤ O 1 |S i | (30
) Overall, combining Equation (23), Equation (29), Equation (27) and Equation (25) we see that with probability at least 0.97 over the choice of S i and T i it is the case that E (x,y)∼Si [ φp * ,10r (x)] ≤ 60r + O(ε). Similarly, combining Equation (24), Equation (30), Equation (28) and Equation (26) we see that with probability at least 0.97 over the choice of S i and T i it is the case that E (x,y)∼Si [ ψp * ,Ti (x)] ≤ O(ε). B.4.3 p * has a small empirical error. Finally, we argue that with probability 0.99 we have E (x,y)∼Si |p * (x)-y| 2 ≤ E (x,y)∼D |f * (x)-y| 2 + O(ε). Indeed, Hoeffding's inequality implies that with probability at least 0.995 we have E (x,y)∼Si |f * (x) -y| 2 ≤ E (x,y)∼D |f * (x) -y| 2 + ε, whereas Markov's inequality tells us that with probability 0.995 we have
E (x,y)∼Si |f * (x) -p * (x)| 2 ≤ O(1) • E (x,y)∼D |f * (x) -p * (x)| 2 = O(ε).
Overall, we see that with probability at least 0.99 it holds that
E (x,y)∼Si |p * (x) -y| 2 ≤ E (x,y)∼Si |f * (x) -y| 2 +E (x,y)∼Si |p * (x) -f * (x)| 2 ≤ E (x,y)∼D |f * (x) -y| 2 +O(ε)
C Pseudocode and analysis of ComputeClassifier C.1 Local correction of adversarial robustness Theorem C.1 (Correctness and complexity of Algorithm 3). Let φ be an ε-accurate local noise sensitivity approximator for degree-k PTFs over R d , and let τ be the run-time and query complexity of φ. Then there exists an algorithm ROBUSTNESSLCA(x, g, r) that takes x in R d , query access to a function g : R d → {-1, 1}, perturbation size parameter r in (0, 1). The algorithm makes O(τ ) queries to g, runs in time O(τ + d), and satisfies the specifications below.
Let D be a distribution over R d , and let g be a degree-k PTF. Let the function h : R d → {-1, 1} be defined as h(x) := ROBUSTNESSLCA(x, g, r). Then the following properties hold:
a) Pr x∼D [g(x) ̸ = h(x)] ≤ iso D,10r (g, 0.8). b) Pr x∼D [ϕ g,10r (x) ≤ 0.1] ≥ 1 -O NS D,10r (g) . c) For every x in R d , if φg,10r (x) ≤ 0.1, then for every x ′ with ∥x ′ -x∥ ≤ r we have h(x ′ ) = h(x) .
Algorithm 3 ROBUSTNESSLCA(x, g, r):
1: Input: point x ∈ R d , black-box representation of a function g : R d → {-1, 1}, 2:
robustness radius r 3: Uses: local noise sensitivity approximator φ.
(See Definition 2.5) 4: Output: b ∈ {-1, 1} 5: if φg,10r (x) > 0.8 then 6:
return -g(x) 7: else 8: return g(x)
Proof. Inspecting the algorithm ROBUSTNESSLCA, we see that condition (a) follows directly, as only isolated points have their labels changed. Meanwhile, condition (b) follows directly from Markov's inequality and the definition of NS (Section 2.1.1).
To show condition (c), we first recall that by the assumption that φ is an ε-accurate approximator, for every degree-k polynomial threshold function g and every x in R d we have
φg,10r (x) -ϕ g,10r (x) ≤ ε ≤ 0.01,(31)
where for the last we assumed without loss of generality that ε < 0.01. This implies that whenever φg,10r (x) ≤ 0.1 we have ϕ g,10r (x) ≤ 0.11. Suppose x ′ has distance at most r from x. Since ϕ g,r (x) is defined as ϕ g,10r (x) := Pr
z∼N (0,I d ) [g(x) ̸ = g(x + 10rz)],
we see that
Pr z∼N (0,I d ) [g(x) ̸ = g(x ′ + 10rz)] ≤ Pr z∼N (0,I d ) [g(x) ̸ = g(x + 10rz)] + d T V (N (x, 100r 2 I d ), N ((x ′ ), 100r 2 I d )) ≤ Pr z∼N (0,I d ) [g(x) ̸ = g(x + 10rz)] + ∥x -x ′ ∥ 20r ,
where the last inequality is implied by Fact A.5. Overall, since we know that ϕ g,10r (x) ≤ 0.11 and ∥x -x ′ ∥ ≤ r, we have Pr
z∼N (0,I d ) [g(x) ̸ = g(x ′ + 10rz)] ≤ 0.16.
Now, we consider the following two cases:
• Suppose g(x ′ ) = g(x): then the above implies that ϕ g,10r (x ′ ) ≤ 0.16, which combined with Equation (31) implies that φg,10r (x ′ ) ≤ 0.17. Therefore, by inspecting the algorithm ROBUSTNESSLCA we see that h(x ′ ) = g(x ′ ) = g(x) = h(x).
• On the other hand, suppose g(x ′ ) = -g(x). Then, we have ϕ g,10r (x ′ ) ≥ 1 -0.16 = 0.84, which combined with Equation (31) implies that φg,10r (x ′ ) ≥ 0.83. Therefore, by inspecting the algorithm ROBUSTNESSLCA we see that h(
x ′ ) = -g(x ′ ) = g(x) = h(x).
In either case, we have h(x ′ ) = h(x).
this section cite: []

Section: C.2 Finding good rounding thresholds
Theorem C.2 (Correctness of Algorithm 4). Let ε ∈ (0, 1) and D be a distribution over R d × {±1}.
Let φ be an ε-accurate local noise sensitivity approximator for degree-k PTFs over R d . Let p : R d → R be a polynomial of degree ≤ k satisfying the following:
E t∼[-1,1] E (x,y)∼D [ϕ sign(p-t),10r (x)] ≤ 100r + O(ε),(32)
Pr (x,y)∼D E t∼[-1,1] [ϕ sign(p(x)-t),10r (x)] ≥ 0.7 ≤ O(ε).(33)
The algorithm COMPUTEROUNDINGTHRESHOLDS outputs real numbers t 1 , t 2 , t 3 , t 4 and w 1 , w 2 , w 3 , w 4 ∈ [0, 1] such that i w i = 1. Let the hypotheses g 1 , g 2 , g 3 , g 4 be defined as
g i (x) = sign(p(x) -t i ).
Then the following properties hold with probability at least 1 -O(δ):
• i w i • err D (g i ) ≤ E t∼[-1,1] [err D (sign(p -t))] + O(ε) • i w i • NS D,10r (g i ) ≤ 200r + O(ε).
•
i w i • iso D,10r (g i , 0.8) ≤ O(ε).
We argue that with high probability over the choice of the O(1/ε 2 ) random rounding thresholds, the equal-weighted mixture of the rounded functions has each of the desired properties.
Claim C.3 (Properties of the equal mixture of all rounding thresholds). Let ε ∈ (0, 1). Let p : R d → R be a polynomial of degree ≤ k satisfying the following:
• E t∼[-1,1] E (x,y)∼D [ φsign(p-t),10r (x)] ≤ α • Pr (x,y)∼D E t∈[-1,1] [ φsign(p(x)-t),10r (x)] ≥ 0.75 ≤ O(ε).
Let Θ be a set of 100/ε 2 real numbers drawn uniformly and independently from [-1, 1].
Then the following hold with probability ≥ 0.99: (a) E t∈Θ [err D (sign(p -t))] ≤ E t∼[-1,1] [err D (sign(p -t))] + O(ε) (b) E t∈Θ [E x∼D [ φsign(p-t),10r (x)]] ≤ 2α + O(ε) (c) E t∈Θ [Pr x∼D [ φsign(p-t),10r (x) > 0.8]] ≤ O(ε).
Proof of Claim C.3. We will show that each of the conditions holds with high probability, then union bound.
Algorithm 4 COMPUTEROUNDINGTHRESHOLDS(D, p, r, ε, δ): 1: Input: sample access to D, robustness radius r, 2:
error bound ε, confidence bound δ 3: Uses: local noise sensitivity approximator φ.
(See Definition 2.5) 4: Output: thresholds ⃗ t = t 1 , t 2 , t 3 , t 4 ∈ [-1, 1] and weights ⃗ w = w 1 , w 2 , w 3 , w 4 ∈ [0, 1] 5: Initialize Q := ∅. 6: M ← 100/ε 3 • log 2 (1/δ) i.i.d. samples (x, y) from D 7: for i ∈ [log(1/δ)] do 8:
for j ∈ [100/ε 2 ] do 9: Draw rounding threshold t u.a.r from [-1, 1] and define rounded function p t (x) := sign(p(x) -t).
10: Let err t := err M (p t ).
11:
Let NS t := 1 | M | • x∈M φpt,10r (x).
12:
Let iso t := 1 |M | • x∈M 1[ φpt,10r (x) ≥ 0.8].
13:
Let q t := (err t , NS t , iso t ). for tuple q 1 , q 2 , q 3 , q 4 ∈ Q 4 do 17:
Solve the linear program with variables w 1 , w 2 , w 3 , w 4 defined by the following constraints:
•
4 i=1 w i = 1 • 4 i=1 w i q i,1 ≤ opt + Cε
C is a sufficiently large absolute constant.
•
4 i=1 w i q i,2 ≤ 200r + Cε • 4 i=1 w i q i,3 ≤ Cε 18:
If a solution is found, return ⃗ t = t 1 , t 2 , t 3 , t 4 and ⃗ w = w 1 , w 2 , w 3 , w 4 . 19: return ⊥ (a) Let X = t∈Θ err D (sign(p -t)). We have
E Θ [X] = |Θ| • E t∼[-1,1] [err D (sign(p -t))]
and we apply Hoeffding's inequality:
Pr E t∼Θ [err D (sign(p -t))] > E t∼[-1,1] [err D (sign(p -t))] + ε = Pr [(X -E[X])/|Θ| > ε] ≤ exp(-2ε 2 • 100/ε 2 ) ≤ 0.001.(
b) We have by assumption that E t∼[-1,1] E x∼D [ φsign(p-t) (x)] ≤ α. Let X = t∈Θ E x∼D [ φsign(p-t),10r (x)], which has expectation ≤ |Θ| • α. Then by Hoeffding's inequality, we have Pr[X/|Θ| > α + ε] ≤ Pr[(X -E[X])/|Θ| > ε] ≤ exp(-2ε 2 • 100/ε 2 ) ≤ 0.001. (c) We have by assumption that Pr x∼D [E t∼[-1,1] [ φsign(p-t)] (x) ≥ 0.75] ≤ O(ε). We want to bound E t∼Θ Pr x∼D [ φsign(p-t) (x) > 0.75]. We can upper bound this by O(ε) + E t∼Θ Pr x∼D [ φsign(p-t) (x) -E t ′ ∼[-1,1] [ φsign(p-t) (x)] > 0.05] = O(ε) + Pr x∼D Pr t∼Θ [ φsign(p-t) (x) -E t ′ ∼[-1,1] [ φsign(p-t ′ ) (x)] > 0.05]. Let X = t∈Θ φsign(p-t) (x); by Hoeffding's inequality we have Pr Θ E t∼Θ [ φsign(p-t) (x)] > E t ′ ∼[-1,1] [ φsign(p-t ′ ) (x)] + 0.05 = Pr[(X -E[X]/|Θ|] > 0.05 ≤ exp(-2 • 0.025 • (100/ε 2 )) ≤ ε/1000. Thus, combining all three probabilities, we have: Pr x∼D Pr Θ Pr t∼Θ [ φsign(p-t),10r (x) -E t ′ ∼[-1,1] [ φsign(p-t ′ ),10r (x)] > 0.05] ≤ ε/1000 Then we apply a Markov bound: Pr Θ Pr x∼D,t∼Θ [ φsign(p-t),10r (x) -E t ′ ∼[-1,1] [ φsign(p-t ′ ),10r (x)] > 0.05] > ε ≤ 0.001
Union bounding over all the conditions, we have that all three conditions hold simultaneously with probability at least 0.99 over the choice of Θ.
Now we will show the existence of a mixture using four rounding thresholds from Θ, and show that COMPUTEROUNDINGTHRESHOLDS finds it. We will make use of the following well-known theorem of Carathéodory. Theorem C.4 (Carathéodory's theorem [Car07]). Let P ⊂ R k be a set of points and Conv(P ) be its convex hull. For any p ∈ Conv(P ), there exists a set S ⊆ P of k + 1 points such that p can be written as a convex combination of points in S.
Proof of Theorem C.2. We first note that since φ is an ε-accurate local noise sensitivity approximator for degree-k PTFs, Equations 32 and 33 respectively imply thatfoot_5
E t∼[-1,1] E (x,y)∼D [ φsign(p-t),10r (x)] ≤ 100r + O(ε), (34
)
Pr x∼D t∼[-1,1] [ φsign(p-t),10r (x) > 0.75] ≤ O(ε). (35
)
Thus, the assumptions of Claim C.3 hold with α = 100r. For a threshold t, let the three-dimensional point q t := (err t , NS t , iso t ) be defined as in COMPUTEROUNDINGTHRESHOLDS. First we argue that each of these estimates is O(ε)-accurate. Each estimate is the expectation of a random variable bounded in [0, 1], so by Hoeffding's inequality we have
Pr[|err t -err D (sign(p -t))| ≥ ε] ≤ exp(-2ε 2 • |M |),
and likewise for each NS t and iso t . We set M large enough that this probability is ≤ δ/2 • exp(-100/ε), giving us more than enough room to union bound over the log(1/δ) • 300/ε 2 estimates. Likewise, by the Hoeffding bound, we have with probability 1 -δ that
opt -E t∼[-1,1] [err D (sign(p -t))] ≤ ε.
Thus with probability at least 1 -δ, all estimates are ε-accurate. Then, by Claim C.3, we have that with large constant probability over the randomness of the set Θ, there exists a convex combination q ⋆ of the points {q t : t ∈ Θ} that satisfies the linear constraints
• q ⋆ [1] ≤ E t∼[-1,1] [err D (sign(p -t))] + O(ε) ≤ opt + Cε • q ⋆ [2] ≤ 200r + Cε • q ⋆ [3] ≤ Cε,
for a sufficiently large constant C. That linear combination is the equal-weighted mixture E t∈Θ [q t ]. By Theorem C.4, since the q t 's are points in R 3 , it is possible to write q ⋆ as a convex combination of four members of {q t : t ∈ Θ}. Thus, for some 4-tuple in Q 4 , the linear program is feasible. COMPUTEROUNDINGTHRESHOLDS searches every possible tuple, guaranteeing that a solution is returned.
By repeating log(1/δ) times with independent draws of Θ, the large constant success probability is boosted to 1 -δ.
this section cite: ['b7']

Section: C.3 Randomized set partitioning
Algorithm 5 COMPUTECLASSIFIER(D, p, r, ε, δ): 1: Input: sample access to D, polynomial p, 2:
robustness radius r, error bound ε, confidence parameter δ 3: Uses: local noise sensitivity approximator φ.
(See Definition 2.5) 4: Output: Classifier h : R d → {±1}. 5: (t 1 , t 2 , t 3 , t 4 , w 1 , w 2 , w 3 , w 4 ) ← COMPUTEROUNDINGTHRESHOLDS(D, p, r, ε, δ).
(See Algorithm 4). 6: S test ← C log 2 (1/δ)/ε 2 samples from D (for sufficiently large constant C) 7: for i ∈ [log(1/δ)] do 8: M ← 10d 3 i.i.d. samples (x, y) from D 9:
For i ∈ {1, ..., 4}, define the following Boolean functions: let
h i (x) = ROBUSTNESSLCA(x, sign(p -t i ), r) (See Algorithm 3) RobustIndicator i (x) := 1[ φsign(p-ti),10r (x) ≤ 0.1] 10: For each i ∈ {1, ..., 4}, let μi ← E (x,y)∈M 1[y ̸ = h i (x)] μ′ i ← E (x,y)∈M RobustIndicator i (x).
11:
Obtain an orthonormal collection of vectors {u 1 ,
• • • , u d/2 } orthogonal to span(µ 1 , • • • , µ 4 , µ ′ 1 , • • • , µ ′ 4 ).
12:
Generate a uniformly random unit vector u ⋆ in span(u 1 , • • • , u d/2 ).
this section cite: []

Section: 13:
Compute a partition of the real line into intervals J 1 , . . . J 4 of Gaussian mass w 1 . . . , w 4 respectively.
this section cite: []

Section: 14:
Check if the following hold for each i ∈ [4]:
E (x,y)∼Stest [1[y ̸ = h i (x)] ∧ ⟨x, u ⋆ ⟩ ∈ J i ] = E (x,y)∼Stest [1[y ̸ = h i (x)] • w i ± C ′ ε (for sufficiently large constant C ′ ). E (x,y)∼Stest [RobustIndicator i (x) ∧ ⟨x, u ⋆ ⟩ ∈ J i ] = E x∼Stest [RobustIndicator i (x)] • w i ± C ′ ε) 15: If so, return the function h defined h(x) = 4 i=1 h i (x) • 1[⟨u ⋆ , x⟩ ∈ J i ] 16: return ⊥
In this section, we will prove the correctness of COMPUTECLASSIFIER. As explained in the previous section, the output of COMPUTERANDOMTHRESHOLDS is a list of four weights and four thresholds for rounding. The goal is for a small collection of sets -the sets of points misclassified by each threshold and the sets of points for which ROBUSTNESSLCA guarantees robustness -to each be partitioned such that w i of their mass falls into the i th part for each i.
The partition is a set of intervals along a unit vector chosen uniformly from the subspace orthogonal to the estimated mean vectors of the sets we want to partition. The intervals are chosen to have mass w 1 , . . . , w 4 under the one-dimensional standard Gaussian. We will argue that with high probability over the choice of the random unit vector, all of our sets will be approximately normally distributed in their projections on this vector, and thus the intervals will contain the right proportion of their mass.
The key fact underlying this argument comes from the following theorem of [DHV06]: Theorem C.5 (Concentration of random projections: special case of Theorem 11 of [DHV06]). Let D be a distribution over R d with mean zero and covariance bounded by λ max in operator norm. Assume also that D is supported only on points x satisfying ∥x∥ 2 ≥ ν √ d. For a unit vector v, let D v denote the distribution of ⟨x, v⟩, x ∼ D, and let D ∥•∥ denote the distribution of ∥x∥ 2 / √ d, x ∼ D. With probability at least
1 - λ max ln(1/ε) ε 3 ν 2 • exp -Ω ε 4 ν 2 d λ max ln(1/ε)
over a uniformly random unit vector v, the following holds for all intervals J:
Pr z∼Dv [z ∈ J] - Pr σ∼D ∥•∥ z∼N (0,σ 2 ) [z ∈ J] ≤ ε.
We apply this theorem to the distributions induced by a set we aim to partition. This is our underlying distribution D conditioned on a (potentially randomized) indicator, such as the indicator of a point being misclassified by a hypothesis; we will represent that indicator as r(x). After projecting on the subspace orthogonal to our mean estimates, these distributions will still not have mean exactly zero, so we incorporate an error term into our analysis to account for that. The resulting claim is the following: Claim C.6 (Masses of intervals under random projections). Let r(x) be a randomized Boolean function over R d , P be a d/2-dimensional linear subspace in R d , and J ⊆ R be an interval. Let P be a d/2 × d matrix whose rows are an orthonormal basis for P . Assume the following hold for some β 1 , β 2 ∈ (0, 1) and β 3 ∈ ((d/2) -1/2 , 1):
• Bounded mean: ∥P E x∼D [r(x)x]∥ 2 ≤ β 1 .
• Thin-shell support: Every x for which r(x) can be nonzero satisfies
∥Px∥ 2 √ d/2 = 1 ± β 2 .
• Large support: E x∼D [r(x)] ≥ β 3 .
For unit vector v, let D r v denote the distribution of ⟨x, v⟩, x ∼ D conditioned on r(x) = 1. With probability at least
1 - (1 + β 1 /β 3 ) ln(1/ε) ε 3 β 3 (1 -2β 2 -2β 1 ) • exp -Ω ε 4 β 3 (1 -2β 2 -2β 1 )d ln(1/ε)(1 + β 1 /β 3 )
over v drawn uniformly from unit vectors in P , we have
Pr z∼D r v [z ∈ J] -Pr z∼N (0,1) [z ∈ J] ≤ O(β 1 /β 3 + β 2 + ε). Proof. Let µ := E x∼D,r [x | r(x) = 1].
We will define D r µ,P to be the distribution of P(x -µ) and D ∥•∥ to be the distribution of ∥P(x -µ)∥ 2 / d/2, where x is drawn from D conditioned on r(x) = 1.
We aim to apply Theorem C.5 to the zero-mean distribution D r µ,P in the d/2 dimensional space P , and find the appropriate bounds for ν and λ max .
• Bounding ν: By the thin-shell assumption, every x for which r(x) can be nonzero satisfies ∥Px∥ 2 ≥ d/2(1 -β 2 ). Thus for any such x, we have ∥P(x -µ)∥ 2 ≥ d/2(1 -β 2 ) -∥Pµ∥ 2 by triangle inequality. Now we just need to bound ∥Pµ∥ 2 :
∥Pµ∥ 2 = P E x,r [x | r(x) = 1] 2 = P x∈R n x • D(x) Pr r [r(x) = 1] Pr x,r [r(x) = 1] dx 2 = P E x,r [r(x)x] E x,r [r(x)] 2 = ∥P E x,r [r(x)x]∥ 2 E x,r [r(x)] ≤ β 1 β 3
by the bounded-mean and large support assumptions. This gives us ν ≥ 1 -
β 2 -β1 β3 √ d/2
, and thus
ν 2 ≥ 1 -2β 2 -2 β1 β3 √ d/2 ≥ 1 -2β 2 -2β 1 , by the assumption that β 3 ≥ 1/ d/2.
• Bounding λ max : We want a spectral bound on E x∼D r µ,P [xx T ]. We have:
E x∼D r µ,P [xx T ] = E x∼D,r [r(x)(P(x -µ))(P(x -µ)) T ] E x,r [r(x)] ≤ 1 β 3 P E x∼D xx T + µµ T -xµ T -µx T P T
Since D has mean zero, P and E[xx T ] has operator norm 1, and Pµµ T P T has operator norm ≤ ∥Pµ∥ 2 , we have
λ max = E x∼D r µ,P [xx T ] op ≤ 1 + ∥Pµ∥ 2 β 3 ≤ 1 + β 1 /β 3 β 3 .
Substituting these parameters into Theorem C.5 gives us the claimed failure probability bound. The condition that holds with high probability is that for all intervals J, we have
Pr z∼(D r µ,P )v [z ∈ J] - Pr σ∼D ∥•∥ z∼N (0,σ 2 ) [z ∈ J] ≤ ε.
By the thin-shell assumption, the bound on ∥Pµ∥ 2 , and the triangle inequality,
D ∥•∥ is supported on [1 -β 1 /β 3 -β 2 , 1 + β 1 /β 3 + β 2 ].
Then by the TV distance bound for Gaussians, we have for any σ in the support of D ∥•∥ :
d T V (N (0, 1), N (0, σ 2 )) ≤ 3|1 -σ 2 | 2 ≤ O(β 1 /β 3 + β 2 ), so we have Pr z∼(D r µ,P )v [z ∈ J] -Pr z∼N (0,1) [z ∈ J] ≤ O(ε + β 1 /β 3 + β 2 ).
We now substitute the definition of (D r µ,P ) v and use the fact that v is in P .
Pr z∼(D r µ,P )v [z ∈ J] = Pr x∼D r [⟨P(x -µ), v⟩ ∈ J] = Pr x∼D r [⟨x, v⟩ + ⟨µ, v⟩ ∈ J] = Pr y∼D r v [y + ⟨µ, v⟩ ∈ J].
this section cite: ['b12', 'b12']

Section: So we have
Pr z∼D r v [z + ⟨µ, v⟩ ∈ J] -Pr z∼N (0,1) [z ∈ J] ≤ O(ε + β 1 /β 3 + β 2 ).
By applying the uniform convergence property of Theorem C.5 for all intervals to shift J, we have
Pr z∼D r v [z ∈ J] -Pr z∼N (0,1) [z -⟨µ, v⟩ ∈ J] ≤ O(ε + β 1 /β 3 + β 2 ).
By anticoncentration of the Gaussian distribution and the fact that |⟨µ, v⟩| ≤ β 1 /β 3 , we have
Pr z∼N (0,1) [z -⟨µ, v⟩ ∈ J] = Pr z∼N (0,1) [z ∈ J] ± 2β 1 /β 3 .
Combining everything, we have
Pr z∼D r v [z ∈ J] -Pr z∼N (0,1) [z ∈ J] ≤ O(ε + β 1 /β 3 + β 2 )
with probability at least
1 - (1 + β 1 /β 3 ) ln(1/ε) ε 3 β 3 (1 -2β 2 -2β 1 ) • exp -Ω ε 4 β 3 (1 -2β 2 -2β 1 )d ln(1/ε)(1 + β 1 /β 3 )
as desired.
We now argue that for r indicating a set of large Gaussian volume and P being orthogonal to a good approximation to the mean of r, the assumptions of Claim C.6 are satisfied with small values of β.
The following facts are relevant to this proof. Fact C.7 (Thin-shell concentration of log-concave variables [GM11]). Let X be an isotropic random vector with log-concave density in R d . Then there are universal constants c, C > 0 such that for all
t ≥ 0, Pr ∥X∥ 2 - √ d ≥ t ≤ C exp(-cd 1/2 min((t/ √ d) 3 , t/ √ d)).
Corollary C.8. Let X be an isotropic random vector with log-concave density in R d . Then there is a universal constant C such that for any ε > 0,
Pr ∥X∥ 2 - √ d ≥ C(d log(1/ε)) 1/3 ≤ ε.
Fact C.9 (Thin-shell concentration of Gaussian variables (standard fact)). Let X be a standard Gaussian in R d . Then Pr ∥X∥ 2 -√ d ≥ t ≤ 2 exp(-t/4).
Claim C.10 (Log-concave distributions satisfy assumptions of Claim C.6). Let D be an isotropic log-concave distribution over R d and K be a sufficiently large constant depending on those given by Fact C.7. Let r be a randomized Boolean function such that E x∼D [r(x)] ≥ β, and let μ(r) be such that for all i ∈ [d],
μ(r) i -E x∼D [r(x)x] i ≤ 1/d.
Let P be a d/2-dimensional linear subspace orthogonal to μ(r), and P be a d/2 × d matrix whose rows are an orthonormal basis for P . Let r trunc be the truncation of r defined as
r trunc (x) := r(x) ∧ 1 ∥Px∥ 2 d/2 = 1 ± 10Kd 1/3 ln d d/2 .
Then the following conditions hold:
• Bounded mean: ∥P E x∼D [r trunc (x)x]∥ 2 ≤ O( 1/d).
• Thin-shell support: Every x for which r trunc (x) can be nonzero satisfies
∥Px∥ 2 √ d/2 = 1 ± 10Kd 1/3 ln d √ d/2 .
• Large support:
E x∼D [r trunc (x)] ≥ β -O(1/d). Furthermore, Pr x∼D [r(x) ̸ = r trunc (x)] ≤ O(1/d).
Proof. First we will prove the final inequality,
Pr x∼D [r(x) ̸ = r trunc (x)] ≤ O(1/d).
Since the projection of D by P is a log-concave distribution over R d/2 (Fact B.3), this inequality follows from from Corollary C.8 applied to P(D) with ε = 1/d. We now prove the other conditions.
• Bounded mean: Let
µ(r trunc ) := E x∼D [r trunc (x)x] and µ(r) := E x∼D [r(x)x].
First we claim that for every i,
|µ(r trunc ) i -µ(r) i | ≤ 1/d. We have µ(r) i = µ(r trunc ) i + E[r(x)x i • 1[r(x) ̸ = r trunc (x)]] ≤ µ(r trunc ) i + E[∥x∥ • 1[r(x) ̸ = r trunc (x)] ≤ µ(r trunc ) i + ∞ t=10Kd 1/3 ln d t • C exp(-ct 3 /d)dt ≤ µ(r trunc ) i + C ∞ t=10Kd 1/3 ln d t 3 • exp(-ct 3 /d)dt ≤ µ(r trunc ) i + C ∞ u=(10K ln d) 3 d u • exp(-cu/d)du ≤ µ(r trunc ) i + exp(-(10K ln d) 3 • c) • ((10K ln d) 3 • c + 1) • (d/c) 2 ≤ µ(r trunc ) i + 1/d. (by setting K sufficiently large) A symmetric argument lower bounds µ(r) i by µ(r trunc ) i -1/d. Now we bound ∥Pµ(r trunc )∥ 2 . We have ∥Pµ(r trunc )∥ 2 ≤ sup u∈P,∥u∥ 2 =1 ⟨u, µ(r trunc )⟩. Consider a unit vector u ∈ P . We have ⟨u, µ(r trunc )⟩ = ⟨u, µ(r trunc ) -µ(r)⟩ + ⟨u, µ(r) -μ(r)⟩ + ⟨u, μ(r)⟩ ≤ i∈[d] u i d + i∈[d] u i d + 0 ≤ 2/ √ d.
• Thin-shell support: This follows immediately from the definition of r trunc .
• Large support: This follows immediately from the assumption that E x∼N (0,I d ) [r(x)] ≥ β and the fact that
Pr x∼N (0,I d ) [r(x) ̸ = r trunc (x)] ≤ O(1/d).
Claim C.11 (Accuracy of the mean estimates). Let S be a set of 10d 3 points drawn from a distribution D with covariance I d . Let F be a collection of eight (possibly randomized) Boolean functions over R d . Then with probability at least 1 -O(d -3 ), the following holds for all f ∈ F and i ∈ [d]:
1
T x∈T x i f (x) -E x∼D x i f (x) ≤ 1/d.
Proof. By the fact that D has covariance I d , we have Var(x i ) = 1 and Var( 1
|T | x i ) = 1/|T |, so by Chebyshev's inequality Pr T E x∼D [x i f (x)] - 1 |T | x∈T x i f (x) > 10d 2 |T | ≤ 0.01d -4 .
Set |T | := 10d 3 ; then the deviation becomes 10d 2 |T | = 1 d . Union bounding over F and i ∈ [d], we have with probability at least 1 -O(d -3 ), all f and all i satisfy
1 T x∈T x i f (x) -E x∼D x i f (x) ≤ 1/d.
Claim C.12 (Accurate partitioning of (randomized) sets). Let D be a distribution over R d × {-1, 1} such that the R d -marginal is isotropic log-concave. Let ε > d -1/7 . Let F be the set of eight (possibly randomized) Boolean functions over R d given by RobustIndicator i and ErrorIndicator i (x) := 1[y ̸ = h i (x)] for i ∈ [4]. Let w 1 , . . . , w 4 be the weights returned by COMPUTEROUNDINGTHRESH-OLDS and J 1 , . . . , J 4 be the intervals generated by COMPUTECLASSIFIER. Then with probability at least 1 -O(δ), the following holds for all f ∈ F and all i ∈ [4]:
Pr x∼D [⟨x, u⟩ ∈ J i ∧ f (x) = 1] = Pr x∼D [f (x) = 1] • w i ± O(ε).
Proof. First we will claim that each iteration succeeds with probability 1 -1/ poly(n). Consider only the functions f ∈ F such that Pr[f (x) = 1] ≥ ε; for the others, the statement holds trivially. COMPUTECLASSIFIER estimates the means µ i , µ ′ i ; i ∈ [4] of the functions from its sample of size 10d 3 , then sets P to be a linear subspace orthogonal to all of these. By Claim C.11 we have that with probability 1 -O(d -3 ) the mean-accuracy assumption of Claim C.10 is satisfied. Thus, by Claim C.10, the assumptions of Claim C.6 are satisfied with the following parameters:
• β 1 = O( 1/d) • β 2 = O(d -1/6 ln d) • β 3 = ε -O(1/d).
COMPUTECLASSIFIER then generates a uniform random unit vector u ∈ P and four intervals J 1 . . . J 4 of Gaussian volume w 1 , . . . , w 4 . By Claim C.6, with high probability over u, the following holds for each J i :
Pr x∼D|f (x)=1 [⟨x, u⟩ ∈ J i ] = w i ± O(β 1 /β 3 + β 2 + ε) = w i ± O(ε);
taking the conjunction with the event that f (x) = 1 gives us the desired
Pr x∼D [⟨x, u⟩ ∈ J i ∧ f (x) = 1] = Pr x∼D [f (x) = 1] • (w i ± O(ε)) ≤ Pr x∼D [f (x) = 1] • w i ± O(ε).
The failure probability is
(1 + β 1 /β 3 ) ln(1/ε) ε 3 β 3 (1 -2β 2 -2β 1 ) • exp -Ω ε 4 β 3 (1 -2β 2 -2β 1 )d ln(1/ε)(1 + β 1 /β 3 ) (1 + o(1)) ln(1/ε) ε 4 (1 -o(1)) • exp -Ω ε 5 (1 -o(1))d ln(1/ε)(1 + o(1)) ln(1/ε) ε 4 • exp -Ω ε 5 d ln(1/ε) 1/ε 5 • exp -Ω(ε 6 d)
By our assumption that ε > Ω(d -1/7 ), this term is dominated by exp(-Ω(d 1/7 )). The total failure probability is dominated by the O(d -3 ) probability of failure for the mean estimation. Thus, we have that with probability ≥ 1 -1/ poly(d), no bad tail events occur, and the guarantee holds for all f ∈ F and i ∈ [4].
The following boosting argument is relevant only when δ is smaller than this ≈ d -3 failure probability. In each iteration we test the accuracy of the partition by comparing Pr[⟨x, u⟩ ∈ J i ∧ f (x) = 1] to Pr[f (x) = 1] • w i for each function f . It suffices for these estimates to be accurate up to ε additive error. By a Chernoff bound, for a test set of size Ω(log(1/δ)/ε 2 ), all estimates are ε-accurate with probability at least 1 -δ 2 , so that after union bounding over all iterations the estimates are still accurate with probability at least 1 -δ. Since d ≥ 2, the probability that a good partition is not found in any of the log(1/δ) independent attempts is at most 2 -log(1/δ) = δ. Thus, the total failure probability is at most 2δ.
this section cite: ['b24']

Section: C.4 Correctness and complexity of Algorithm 5.
Now we will finish the analysis of COMPUTECLASSIFIER, restated here for convenience.
Theorem 3.4 (Correctness of Algorithm 5, COMPUTECLASSIFIER). Let ε ≥ d -1/7 and δ, r > 0. Let φ be an ε-accurate local noise sensitivity approximator for degree-k PTFs over R d . Let D be a distribution over R d × {-1, 1} such that the R d -marginal is isotropic and log-concave. Let p : R d → R be a polynomial of degree at most k satisfying
E t∼[-1,1] E (x,y)∼D [ϕ sign(p-t),10r (x)] ≤ 100r + O(ε),(5)
Pr (x,y)∼D E t∼[-1,1]
[ϕ sign(p(x)-t),10r (x)] ≥ 0.7 ≤ O(ε).
Then the algorithm COMPUTECLASSIFIER(D, p, r, ε, δ), given sample access to D and query access to φ, outputs a hypothesis h : R d → {-1, 1} such that the following properties hold with probability at least 1 -O(δ):
err D (h) ≤ E t∼[-1,1]
[err D (sign(p -t))] + O(ε)
Boundary D,r (h) ≤ O(r + ε).
The running time and number of queries to φ are poly(
d k • 1/ε • log(1/δ)).
Proof of Theorem 3.4. By Theorem C.1 and the triangle inequality, we have that all i ∈ [4] satisfy (i) Pr (x,y)∼D [h i (x) ̸ = y] ≤ err D (sign(p -t i )) + iso D,10r (sign(p -t i ), 0.8)
(ii) E x∼D [RobustIndicator i (x)] ≥ 1 -O( NS D,10r (sign(p -t i ))
(iii) For every x such that RobustIndicator i (x) = 1 and x ′ : ∥x -x ′ ∥ ≤ r, we have h i (x ′ ) = h i (x).
First we will handle the error condition. By Theorem C.2, we have that with probability 1 -O(δ), i∈[4]
w i • (err D (sign(p -t i ) + iso D,10r (sign(p -t i ), 0.8)) ≤ E t∼[-1,1]
[err D (sign(p -t))] + O(ε).
Thus, we have i∈[4]
w i • Pr (x,y)∼D [h i (x) ̸ = y] ≤ E t∼[-1,1]
[err D (sign(p -t))] + O(ε).
By Claim C.12, we then have with probability ≥ 1 -O(δ),
i∈[4] Pr (x,y)∼D [h i (x) ̸ = y ∧ ⟨x, u ⋆ ⟩ ∈ J i ] ≤ E t∼[-1,1]
[err D (sign(p -t))] + O(ε).
Therefore, applying the definition of h, we have
err D (h) ≤ E t∼[-1,1] [err D (sign(p -t))] + O(ε).
Now we analyze the robustness condition. By Theorem C.2, we have that with probability ≥ 1-O(δ),
i∈[4] w i • NS D,10r (sign(p -t i )) ≤ O(r + ε). By a Markov bound and the definitions of RobustIndicator and NS, we have the following for each i: E[RobustIndicator i (x)] = Pr[ φsign(p-ti) (x) ≤ 0.1] = 1 -Pr[ φsign(p-ti) (x) > 0.9] = 1 -Pr φ(x) > 0.9/ NS(sign(p -t i )) • E[ φ(x)] ≥ 1 -10 9 NS(sign(p -t i ). Thus we have i∈[4]
w i • E D [RobustIndicator i (x)] ≥ i∈[4] w i • (1 -10 9 E D [ φsign(p-ti),10r (x)]) ≥ 1 - i∈[4] w i • 10 9 E D [ φsign(p-ti),10r (x)] ≥ 1 -O(r + ε).
By Claim C.12 we then have with probability ≥ 1 -δ,
i∈[4] E x∼D [RobustIndicator i (x) • 1[⟨x, u ⋆ ⟩ ∈ J i ]] ≥ 1 -O(r + ε).
We now claim that for each J i , all x with RobustIndicator i (x) = 1 such that ⟨x, u ⋆ ⟩ is at least r away from the interval boundary satisfy the adversarial robustness condition
∀x ′ : ∥x -x ′ ∥ 2 ≤ r and h(x) = h(x ′ ).
Since x is r away from the interval boundary, all x ′ such that ∥x -x ′ ∥ 2 ≤ r satisfy ⟨x ′ , u ⋆ ⟩ ∈ J i , so they are also labeled by h i . By Theorem C.1, all x with RobustIndicator i (x) = 1 satisfy the robustness condition in h i , so, since they are also r away from the boundary, they satisfy the robustness condition in h. Thus we have
Boundary D,r (h) ≥ i∈[4] E x∼D [RobustIndicator i (x) • 1[⟨x, u ⋆ ⟩ ∈ J i ]] -Pr x∼D [⟨x, u ⋆ ⟩ within r of an interval boundary] ≥ 1 -O(r + ε) -O(r),
where the boundary probability is bounded by O(r) due to Fact B.4. Thus when all subroutines are successful, the guarantees of the theorem hold. The total success probability is ≥ 1 -O(δ) after union bounding the failure probabilities of all subroutines. This concludes the proof of correctness.
We now analyze the running time and query complexity to φ. COMPUTEROUNDINGTHRESHOLDS is called once. For log(1/δ) iterations, it estimates err t , NS t , iso t for each of 100/ε 2 rounding thresholds t; this step makes O(log(1/δ)/ε 2 ) queries to φ and takes poly(d k • 1/ε • log(1/δ)) additional time, due to the evaluations of the degree-k polynomial p. Then it solves a linear program of constant size for (100/ε 2 ) 4 iterations. The total running time and query complexity of COMPUTEROUND-INGTHRESHOLDS are dominated by the first term, which is poly(
d k • 1/ε • log(1/δ)).
The rest of COMPUTECLASSIFIER repeats log(1/δ) times and does the following each repetition: a) evaluate 1[h i (x) ̸ = y] and RobustIndicator i (x) for each i ∈ [4] and (x, y) in M , b) obtain a basis orthogonal to the estimated mean vectors, a random unit vector in this space, and a partition of the real line into intervals c) check accuracy of the partition with respect to S test .
Each evaluation of 1[h i (x) ̸ = y] makes one query to φ and takes d O(k) time, as it simply calls ROBUSTNESSLCA, which makes one evaluation of a degree-k PTF and one call to φ. Each evaluation of RobustIndicator i (x) makes one query to φ and takes O(1) additional time. Thus items (a) and (c) take d O(k) time and queries. For item (b), obtaining the basis takes poly(d) time by Gaussian elimination, and approximating the interval boundaries can be done in poly(1/ε) time by using a numerical ε-approximation to the error functionfoot_6
erf(t) = 1 √ π t 0 e -t 2 /2 .
Overall, the total time and query complexity of COMPUTECLASSIFIER is poly(d k • 1/ε • log(1/δ)), as desired.
this section cite: []

Section: D Verifiable robustness
In this section we prove that under complexity assumptions, the robustness guarantee of our learning algorithm can be made efficiently verifiable as discussed in Section 1.3. The verifier certifies that a point satisfies the robustness condition. We formally state this result: Corollary D.1 (Deterministic verifiability). If P = BPP, then there is a learning algorithm B that, given access to labeled samples from a subgaussian isotropic log-concave distribution, runs in time d Õ(1/ε 2 ) • log(1/δ) and produces a hypothesis h with the following guarantees:
• Agnostic approximation: With probability at least 1 -O(δ), Pr (x,y)∼D [h(x) ̸ = y] ≤ opt + O(ε), where opt is the misclassification error of the best halfspace.
• Verifiable robustness: There is a verifier that runs in time d Õ(1/ε 2 ) that takes as input a circuit g and a point x ∈ R d that always rejects if ∃z : ∥z∥ 2 ≤ r and g(x) ̸ = g(x + z).
If g = h, then with probability at least 1 -O(δ) over the randomness of B, the verifier accepts with probability at least 1 -O(r + ε) over x ∼ D.
We use the following fact:
Fact D.2 (Derandomized estimation of φ). If P = BPP, then there exists a deterministic algorithm running in time d O(k) • poly(1/ε) that takes as input a degree-k PTF f , a radius r, and an input x ∈ R d and outputs an estimate φf,10r (x) such that φf,10r (x) = ϕ f,10r (x) ± ε.
this section cite: []

Section: Proof.
Observe that there is a randomized algorithm running in d O(k) /ε 2 time that takes as input a threshold t and decides if ϕ f (x) ≥ t, succeeding with probability ≥ 2/3 whenever ϕ f (x) > t + ε or ϕ f (x) < t -ε. This is the algorithm that samples O(1/ε 2 ) points z from N (0, I d ) and evaluates f (x + 10rz) on each of them to estimate ϕ; its analysis is a Chernoff bound. If P = BPP, then there exists a deterministic algorithm for this decision problem also running in time d O(k) •poly(1/ε). Since ϕ f (x) ∈ [0, 1], we can binary search with log(1/ε) iterations to find a t such that ϕ f (x) ∈ t ± ε.
The learning algorithm is simply ROBUSTLEARN, augmented to provide some extra information, with the estimates of φ provided by the deterministic estimator. The verifier checks that the hypothesis matches a "template;" this will prove that the unknown circuit is in fact of the form that ROBUSTLEARN is supposed to return, and for which our correctness analysis holds. Since for any hypothesis matching the template, any point x for which ϕ(x) ≤ 0.1 satisfies the robustness condition, the verifier will then deterministically estimate ϕ(x).
We define the template below:
Lemma D.3 (Hypothesis template). Fix a set B of basis functions for the set of degree-k polymomials over R d . There is an algorithm COMPILE that takes as input a |B|-length vector ⃗ v of real-valued coefficients, real numbers t 1 , . . . , t 4 , a unit vector u ∈ R d , and real numbers c 1 < c 2 < c 3 . For each i ∈ [4], let the PTF h i be defined
h i (x) = sign b∈B v b • b(x) -t i .
Let the intervals J 1 , . . . , J 4 be the partition of the real line induced by c 1 , c 2 , c 3 . The algorithm outputs a circuit computing the hypothesis
h(x) = i∈4 ROBUSTNESSLCA(x, h i , r) • 1[⟨x, u⟩ ∈ J i ].
It also outputs each of the PTFs h 1 , . . . , h 4 . The running time is d O(k) .
Proof. Observe that there is a deterministic algorithm running in d O(k) time that takes as input all of the given parameters and the input x, and evaluates h(x), using the deterministic implementation of φ in ROBUSTNESSLCA. Therefore there is a circuit of size d O(k) with the same behavior. The algorithm COMPILE takes this circuit and hardcodes all the input parameters except for x, then outputs the resulting circuit. This takes time d O(k) and the output is a circuit that takes x as input and evaluates h(x). By the same argument, in d O(k) time COMPILE can also output the PTFs h 1 , . . . , h 4 .
We include pseudocode of the verifier below. For brevity we will refer to the package of data taken as input by COMPILE as data.
Algorithm 6 VERIFY(g, x, r, ε, data):
1: Input: circuit g, point x ∈ R n , robustness radius r, error tolerance ε, hypothesis parameters data 2: h, h 1 , h 2 , h 3 , h 4 ← COMPILE(data) 3: if g ̸ = h then return reject. 4: Let u be the unit vector in data and J 1 , . . . , J 4 be the partition induced by the interval boundaries c 1 , c 2 , c 3 in data. 5: Let i be the interval such that ⟨x, u⟩ ∈ J i . 6: if φhi,10r (x) > 0.1 -ε then return reject. 7: if |⟨x, u⟩ -c j | ≤ r for any of the interval boundaries c 1 , c 2 , c 3 in data then return reject. 8: return accept.
Proof of Corollary D.1. The learning algorithm B is ROBUSTLEARN, but with the following modification: it records its parameters in a data package, outputs COMPILE(data) as its final hypothesis, and outputs data as well. By the guarantees of ROBUSTLEARN, with probability at least 1 -O(δ), h satisfies the agnostic approximation guarantee. When VERIFY is called on the output h of B, since it is the output of COMPILE, it always passes the first check. By inspecting the proof of Theorem C.1, we see that in fact robustness holds for all x such that φhi (x) ≤ 0.1 -ε and x is at least r distance from each of the interval boundaries. By Fact D.2, we have | φhi (x) -ϕ hi (x)| ≤ ε, thus the verifier rejects all points such that ϕ hi (x) > 0.1. Since it also rejects if x is within r of a boundary, all accepted points satisfy the robustness condition. Furthermore, by inspecting the proofs of Theorem 3.4 and Theorem C.1, we see that whenever ROBUSTLEARN succeeds (probability 1 -O(δ)), we have that at least 1 -O(r + ε) fraction of points x ∼ D satisfy φhi (x) ≤ 0.1 -ε and are at least r away from any boundary, and are thus verifiably robust. Thus, the verifiable robustness guarantee holds.
The running time of COMPILE for PTFs of degree Õ(1/ε 2 ) is d Õ(1/ε 2 ) (Lemma D.3). The running time of VERIFY is this plus the running time of the deterministic estimator for φ, which is also d Õ(1/ε 2 ) (Fact D.2). Thus the total running time is d Õ(1/ε 2 ) . Remark D.4. We note that without the assumption that P = BPP, there is a randomized analogue of Corollary D.1 where the user compiles the final hypothesis, rather than verifying the one provided by ROBUSTLEARN. Rather than outputting a circuit h, ROBUSTLEARN can just output data, and the user can COMPILE it with a randomized implementation of ROBUSTNESSLCA in time d Õ(1/ε 2 ) • log(1/δ), in which case the soundness of the verifier holds with probability 1 -δ for the compiled hypothesis.
this section cite: []

Section: E Uniform convergence claims
Finally, we will need to following observation about the uniform convergence fo empirical approximations of local noise sensitivity ϕ. First, we need the following fact: Fact E.1 ([VW97], also see lecture notes [Duc23]). A function class C of VC dimension ∆ and every distribution D 0 , there is an ε-cover H of C of size at most β := (O(1)/ε) O(∆) . I.e. H is a discrete subset of C of size β and for every f in C we have h in H for which Pr x∼D0 [f (x) ̸ = h(x)] ≤ ε, Claim E.2. Let C be the class of degree-k PTFs over R d , let D be a probability distribution over R d and let η ∈ [0, 1] be fixed. Then, for some constant C, if S is a collection of (d k /ε) C log 1/δ i.i.d. examples from D, then with probability at least 1 -O(δ),
max f ∈C E x∼S [ϕ f,η (x)] -E x∼D [ϕ f,η (x)] ≤ O(ε)(36)
Proof. We now use Fact E.1. For us, C is the class of degree-k PTFs, and we have ∆ = d O(k) .
Taking the distribution D 0 to be an equal-weight mixture of (i) the distribution D in the premise of this claim and (ii) the convolution of D with the normal distribution N (0, ηI d ) we see that for every f in C there is h in H for which 11Pr
x∼D [f (x) ̸ = h(x)] ≤ 2ε,(37)
Pr x∼D z∼N (0,I d ) [f (x + ηz) ̸ = h(x + ηz)] ≤ 2ε,(38)
which via the definition of ϕ and the triangle inequality implies that
|E x∼D [ϕ f,η (x)] -E x∼D [ϕ h,η (x)]| ≤ E x∼D ϕ f,η (x) -ϕ h,η (x) ≤ 2 E x∼D [|f (x) -h(x)|] + E x∼D z∼N (0,I d ) [|f (x + ηz) -h(x + ηz)|] ≤ O(ε). (39
)
By the standard Hoeffding bound, since ϕ and ξ are always in [0, 1], with probability 1 -δ/2 for every h in H we have
E x∼S [ϕ h,η (x)] -E x∼D [ϕ h,η (x)] ≤ 1 |S| O(log(|H|/δ)) ≤ ε,(40)
where the last step for both inequalities follows by substituting |H|, |S|, ∆ and taking C to be a sufficiently large absolute constant. We also know that with for a sufficiently large absolute constant C, with probability 1 -δ the set S satisfies Fact E.6, which means that: sup f1,f2∈C
|Pr x∼S [f 1 (x) ̸ = f 2 (x)] -Pr x∼D [f 1 (x) ̸ = f 2 (x)]| ≤ O(ε).(41)
From the above, we can also conclude that with probability 1 -δ over S for every pair f 1 , f 2 in C we have the following:
Pr x∼S z∼N (0,I d ) [f 1 (x + ηz) ̸ = f 2 (x + ηz)] -Pr x∼D z∼N (0,I d ) [f 1 (x + ηz) ̸ = f 2 (x + ηz)] ≤ E z∼N (0,I d ) Pr x∼S [f 1 (x + ηz) ̸ = f 2 (x + ηz)] -Pr x∼D [f 1 (x + ηz) ̸ = f 2 (x + ηz)]
≤O(ε) by defining f3(x) := f1(x + ηz), f4(x) := f2(x + ηz), observing f3 and f4 are also degree-k PTFs and using Equation ( 41)
≤ O(ε)(42)
Using the definition of ϕ, the triangle inequality, and Equations 41 and 42 we also see that
|E x∼S [ϕ h,η (x)] -E x∼S [ϕ f,η (x)]| ≤ 2 Pr x∼S [h(x) ̸ = f (x)] + Pr x∼S z∼N (0,I d ) [h(x + ηz) ̸ = f (x + ηz)] ≤ 2 Pr x∼D [h(x) ̸ = f (x)] + Pr x∼D z∼N (0,I d ) [h(x + ηz) ̸ = f (x + ηz)] ≤ O(ε), (43
)
where in the last step we substituted Equation (37) and Equation (38). Finally, we put all the inequalities together. By combining the triangle inequality with Equations 39, 40 and 43 we derive Equation (36).
Claim E.3. Let C be the class of degree-k PTFs over R d , let D be a probability distribution over R d and let η ∈ [0, 1] be fixed. Then, for some constant C, if S is a collection of (d k /ε)
C log 1/δ i.i.d. examples from D, then with probability at least 1 -O(δ), for every polynomial p it holds that Pr x∼D E t∈[-1,1] [ϕ sign(p(x)-t),η (x)] ≥ 0.7 ≤ Pr x∼D [E t∈[-1,1] [ϕ sign(p-t),η (x)] ≥ 0.67] + O(ε) (44) Proof. Let C clipped be the class of degree-k clipped polynomials, i.e. functions p clipped : R d → [-1, 1] of the form
p clipped (x) =    1 if p(x) ≥ 1 -1 if p(x) ≤ -1 p(x) otherwise
where p is a degree-k polynomial. We first argue the following Observation E.4. For any distribution D 0 over R d , there is a subset H of C clipped of size (O(1)/ε) O(d k /ε) , i.e. such that for every p clipped in C clipped we have
min h∈H [E x∼D0 [ p clipped (x) -h(x) ]] ≤ O(ε).(45)
Proof. We form H by considering an ε 2 -net H ε 2 PTF for degree-2k PTFs, and taking H to consist of functions of the form h(x) = τ ∈{-1,-1+ε,-1+2ε,...,+1}
τ • f τ (x),
where each f τ (x) is some function in f τ (x). By Fact E.1, we see that H ε 2 PTF has size (O(1)/ε) O(d k ) , and therefore (by a counting argument) the size of H defined above is indeed (O(1)/ε) O(d k /ε) . We can write for every x and p clipped in C clipped the following inequality:
τ ∈{-1,-1+ε,-1+2ε,...,+1} τ • 1[p clipped (x) ∈ (τ, τ + ε]] ≤ p clipped (x) ≤ τ ∈{-1,-1+ε,-1+2ε,...,+1} (τ + ε) • 1[p clipped (x) ∈ (τ, τ + ε]], (46
)
we observe that each indicator 1[p clipped (x) ∈ (τ, τ + ε]] equals to a degree-2k polynomial threshold function, and therefore there is some f τ in H ε 2 PTF for which
E x∼D0 1[p clipped (x) ∈ (τ, τ + ε]] -f τ ≤ O(ε 2 ),
which substituted into Equation (46) allows us to conclude that
E x∼D0 p clipped (x) - τ ∈{-1,-1+ε,-1+2ε,...,+1} τ • f τ (x) ≤ O(ε),
which concludes the proof that for p clipped in C clipped we have h : R d in H for which Equation (45) holds.
We finish the proof of the observation, by resolving one last issue: as defined the set H is not a subset of C clipped . However, if we consider the set H ′ consisting of functions f of the form
H ′ = {f =
argmin g∈C clipped [E x∼D0 [|g(x) -h(x)|]] : h ∈ H, } then we see by the triangle inequality that H ′ still satisfies Equation (45), has a size of at most |H| and H ′ is a subset of C clipped We continue the proof of Claim E.3 We can wite: E t∈[-1,1] [ϕ sign(p(x)-t),η (x)] = Pr t∈[-1,1],z∼N (0,I d ) [|sign(p(x) -t) ̸ = sign(p(x + ηz) -t)|] = E z∼N (0,I d ) p clipped (x) -p clipped (x + ηz) 2 := ϕ p clipped ,η (x) (47)
We will overload the notation for ϕ and define ϕ p clipped ,η (x) to be the expression above. We further define the following auxiliary quantity:
ξ p clipped ,η (x) =    0 if ϕ p clipped ,η (x) ≤ 0.67 1
if ϕ p clipped ,η (x) ≥ 0.7 100 3 (ϕ p clipped ,η (x) -0.67) if ϕ f,η (x) ∈ (0.67, 0.7) By construction, the function satisfies the following properties for every x,
1[ϕ p clipped ,η (x) ≥ 0.67] ≤ ξ p clipped ,η (x) ≤ 1[ϕ p clipped ,η (x) ≥ 0.7](48)
ξ p clipped 1 ,η (x) -ξ p clipped 2 ,η (x) ≤ O(1) • ϕ p clipped 1 ,η (x) -ϕ p clipped 2 ,η (x)(49)
Taking the distribution D 0 to be an equal-weight mixture of (i) the distribution D in the premise of this claim and (ii) the convolution of D with the normal distribution N (0, ηI d ) we see that for every p clipped in C clipped there is h in H for which 12Pr x∼D [ p clipped (x) -h(x) ] ≤ 2ε,
Pr x∼D z∼N (0,I d ) [ p clipped (x + ηz) -h(x + ηz) ] ≤ 2ε,(50)
which via the definition of ϕ for clipped polynomials and the triangle inequality implies that
E x∼D [ϕ p clipped ,η (x)] -E x∼D [ϕ h,η (x)] ≤ E x∼D ϕ p clipped ,η (x) -ϕ h,η (x) ≤ 2 E x∼D [|p clipped (x) -h(x)|] + E x∼D z∼N (0,I d ) [|p clipped (x + ηz) -h(x + ηz)|] ≤ O(ε). (52
)
Together with Equation (49), this implies that
E x∼D [ξ p clipped ,η (x)] -E x∼D [ξ h,η (x)] ≤ O(1) • E x∼D ϕ p clipped ,η (x) -ϕ h,η (x) ≤ O(ε). (53
)
By the standard Hoeffding bound, since ϕ and ξ are always in [0, 1], with probability 1 -δ/2 for every h in H we have
E x∼S [ϕ h,η (x)] -E x∼D [ϕ h,η (x)] ≤ 1 |S| O(log(|H|/δ)) ≤ ε,(54)
E x∼S [ξ h,η (x)] -E x∼D [ξ h,η (x)] ≤ 1 |S| O(log(|H|/δ)) ≤ ε,(55)
where the last step for both inequalities follows by substituting |H|, |S|, ∆ and taking C to be a sufficiently large absolute constant.
this section cite: ['b20']

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Our work focuses on robustly learning linear classifiers under isotropic subgaussian log-concave marginals, and our introduction and abstract reflect this. Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Our algorithm only works under a strong distributional assumption, and it produces a hypothesis that is not a linear classifier. These limitations are thoroughly discussed in the introduction.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
this section cite: []

Section: 
We observe that Fact E.6 and Equation (46) together imply that 13 : sup p clipped 1 ,p clipped 2 ∈C clipped E x∼S [ p clipped 1 (x) -p clipped 2 (x) ] -E x∼D [ p clipped 1 (x) -p clipped 2 (x) ] ≤ O(ε). (56) From the above, we can also conclude that with probability 1 -δ over S for every pair p clipped 1 , p clipped 2 in C clipped we have the following: E x∼S z∼N (0,I d ) [ p clipped 1 (x + ηz) -p clipped 1 (x + ηz) ]-E x∼D z∼N (0,I d ) [ p clipped 1 (x + ηz) -p clipped 2 (x + ηz) ] ≤ E z∼N (0,I d ) E x∼S [ p clipped 1 (x + ηz) -p clipped 2 (x + ηz) ] -E x∼D [ p clipped 1 (x + ηz) -p clipped 2 (x + ηz) ] ≤O(ε) by defining p clipped 3 (x) := p clipped 1 (x + ηz), p clipped 4 (x) := p clipped 2 (x + ηz), and using Equation (56) ≤ O(ε) (57)
We now come back to the setting of Equation (50). p clipped is a function in C clipped and h in H satisfies Equation (50). Using the definition of ϕ, the triangle inequality, and Equations 56 and 57 we also see that
E x∼S [ϕ h,η (x)] -E x∼S [ϕ p clipped ,η (x)] ≤ 2 Pr x∼S [ h(x) -p clipped (x) ] + Pr x∼S z∼N (0,I d ) [ h(x + ηz) -p clipped (x + ηz) ] ≤ 2 Pr x∼D [h(x) ̸ = f (x)] + Pr x∼D z∼N (0,I d ) [h(x + ηz) ̸ = f (x + ηz)] ≤ O(ε), (58)
where in the last step we substituted Equation (50) and Equation (51). By Equation 49, we see that
|E x∼S [ξ h,η (x)] -E x∼S [ξ f,η (x)]| ≤ O(1) • |E x∼S [ϕ h,η (x)] -E x∼S [ϕ f,η (x)]| ≤ O(ε), (59)
Analogously, by combining the triangle inequality with Equations 53, 55 and 59 we derive the following:
max f ∈C E x∼S [ξ f,η (x)] -E x∼D [ξ f,η (x)] ≤ O(ε),(60)
which together with Equation (48) and Equation (47) implies Equation (44).
The following claims follow from VC theory (see e.g. [VDVW09] and the references therein): Fact E.5 (Generalization bound from VC dimension). Let C be a concept class and D be a distribution over R d . For some sufficiently large absolute constant C, the following is true. With probability at least 1 -δ over a sample S of i.i.d. samples from D, with |S| ≥ C • V C(C) log(1/δ)/ε 2 the following holds:
sup f ∈C |err D (f ) -err S (f )| ≤ O(ε).
Fact E.6. Let C be a concept class with VC dimension ∆ and D be a distribution over R d . For a sufficiently large absolute constant C, let S be a collection of C∆ ε 2 log(1/δ) i.i.d. examples from D. Then, the following holds with probability at least 1 -δ sup f1,f2∈C
|Pr x∼S [f 1 (x) ̸ = f 2 (x)] -Pr x∼D [f 1 (x) ̸ = f 2 (x)]| ≤ O(ε).(61)
Answer: [Yes] Justification: Our theorems contain the full sets of assumptions, and proofs are given in the appendix.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: ['b36']

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper does not include experiments requiring code.
Guidelines:
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
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [NA]
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: 9.

this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: Our work fully complies with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
this section cite: []

Section: Answer: [NA]
Justification: There is no societal impact, as the work is purely theoretical.
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
Answer: [NA] Justification: The paper poses no such risks, as the work is purely theoretical.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA] Justification: the paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: the paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: the paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: the paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: LLMs were used only for minor suggestion regarding LaTeX. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: The power of localization for efficiently learning linear separators with noise Year: (2017)
Ref_id:b1 Title: Property-preserving data reconstruction Year: (2008)
Ref_id:b2 Title: On robustness to adversarial examples and polynomial optimization Year: (2019-12-08)
Ref_id:b3 Title: Classification by polynomial surfaces Year: (1995-07)
Ref_id:b4 Title: Space-efficient local computation algorithms Year: (2012)
Ref_id:b5 Title: Evasion attacks against machine learning at test time Year: (2013)
Ref_id:b6 Title: Adversarial examples from computational constraints Year: (2019)
Ref_id:b7 Title: Über den variabilitätsbereich der koeffizienten von potenzreihen, die gegebene werte nicht annehmen. (mit 2 figuren im text) Year: (1907)
Ref_id:b8 Title: Pac-learning in the presence of adversaries Year: (2018)
Ref_id:b9 Title: Certified adversarial robustness via randomized smoothing Year: (2019)
Ref_id:b10 Title: A ptas for agnostically learning halfspaces Year: (2015)
Ref_id:b11 Title: Ilias Diakonikolas, Parikshit Gopalan, Ragesh Jaiswal, Rocco Servedio, and Emanuele Viola. Bounded Independence Fools Halfspaces Year: (2009-02)
Ref_id:b12 Title: A concentration theorem for projections Year: (2006)
Ref_id:b13 Title: Agnostic proper learning of halfspaces under gaussian marginals Year: (2021-08)
Ref_id:b14 Title: Nearly tight bounds for robust proper learning of halfspaces with a margin Year: (2019-12-08)
Ref_id:b15 Title: The complexity of adversarially robust proper learning of halfspaces with agnostic noise Year: (2020)
Ref_id:b16 Title: The optimality of polynomial regression for agnostic learning under gaussian marginals in the sq model Year: (2021)
Ref_id:b17 Title: Learning geometric concepts with nasty noise Year: (2018)
Ref_id:b18 Title: The total variation distance between high-dimensional gaussians with the same mean Year: (2018)
Ref_id:b19 Title: Computational limitations in robust classification and win-win results Year: (2019)
Ref_id:b20 Title: VC-dimension, covering, and packing Year: (2023)
Ref_id:b21 Title: A moment-matching approach to testable learning and a new characterization of rademacher complexity Year: (2023)
Ref_id:b22 Title: On the hardness of robust classification Year: (2021)
Ref_id:b23 Title: Planting undetectable backdoors in machine learning models Year: (2022)
Ref_id:b24 Title: Interpolating thin-shell and sharp large-deviation estimates for isotropic log-concave measures Year: (2011)
Ref_id:b25 Title: Explaining and harnessing adversarial examples Year: (2014)
Ref_id:b26 Title: Agnostically learning halfspaces Year: (2008)
Ref_id:b27 Title: Certified robustness to adversarial examples with differential privacy Year: (2018)
Ref_id:b28 Title: Second-order adversarial attack and certifiable robustness Year: ()
Ref_id:b29 Title: Properly learning monotone functions via local correction Year: (2022)
Ref_id:b30 Title: Efficiently learning adversarially robust halfspaces with noise Year: (2020)
Ref_id:b31 Title: Vc classes are adversarially robustly learnable, but only improperly Year: (2019)
Ref_id:b32 Title: Fast local computation algorithms Year: (2011)
Ref_id:b33 Title: Are adversarial examples inevitable? arXiv preprint Year: (2018)
Ref_id:b34 Title: Log-concavity and strong log-concavity: a review Year: (2014)
Ref_id:b35 Title: Intriguing properties of neural networks Year: (2014)
Ref_id:b36 Title:  Year: (2009)
Ref_id:b37 Title: Weak convergence and empirical processes with applications to statistics Year: (1997)
