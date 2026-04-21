Title: Online Strategic Classification with Noise and Partial Feedback
Abstract: In this paper, we study an online strategic classification problem, where a principal aims to learn an accurate binary linear classifier from interactions with sequentially arriving agents. For each agent, the principal announces a classifier. The agent can strategically exercise costly manipulations on his features to be classified as the favorable positive class. The principal is unaware of the true featurelabel relationship, but observes all reported features and only labels of positively classified agents. We assume that the true feature-label relationship is given by a halfspace model subject to arbitrary feature-dependent but bounded noise (i.e., Massart noise). This problem faces the combined challenges of agents' strategic feature manipulations, partial feedback observations, and label noise. We tackle these challenges by a novel learning algorithm. We show that the proposed algorithm yields classifiers that converge to the clairvoyant optimal classifier and attains a regret rate of O( √ T ) up to poly-logarithmic and constant factors over T cycles.

Section: Introduction
Strategic classification studies the problem of learning robust classifiers in presence of self-interested strategic agents. When subjugated to decision-making aided by classification algorithms, agents may strategically modify their observable features to game the classification algorithms into making decisions that best serve the agents' goals. For example, a bank may use classification to determine whether loan applicants are qualified to grant approvals. The applicants prefer positive classification and loan approvals, so they have the incentive to modify their profiles (e.g., credit score), potentially at certain costs, without actually improving their financial status. It is crucial that classification algorithms used for decision-making be robust to such strategic manipulation.
Besides the strategic feature manipulation, another common challenge in classification-based decision-making is that the decision-maker often only observes partial feedback. In particular, the decision-maker may only observe the true labels of agents who have received the positive decision. For example, the bank can observe the true financial qualification only for applicants who have already been classified as qualified and granted loan approvals, but has no chance to observe the true qualification of rejected applicants. This type of partial feedback is sometimes called one-sided feedback, apple-tasting feedback [e.g., Harris et al., 2023, Helmbold et al., 2000] or selective label feedback [e.g., Lakkaraju et al., 2017, Chen et al., 2025].
In this paper, we study an online strategic classification problem with partial feedback. In this problem, a principal (decision-maker) interacts with sequentially arriving agents. The principal announces a binary linear classifier to each agent and makes the decision according to the agent's reported feature that may differ from the truth due to strategic manipulation. Following the existing literature, we assume that the agents manipulate their features to maximize the net utility from the classification decision and the cost of feature manipulation. We assume that the agents' true featurelabel relationship is characterized by a linear halfspace model with arbitrary feature-dependent but bounded noise (i.e., Massart noise). This model is widely adopted in the learning theory literature (see references in Section 1.1). The principal does not know the true feature-label relationship, but needs to learn accurate binary classifiers from observations of agents' reported features (but not the original true features) and the true labels of only positively classified agents.
Notably, this problem faces the combination of three challenges: agents' strategic feature manipulations, partial label observations, and label noise. First, because of strategic feature manipulation, the agents' true features may not be faithfully observed by the principal, which impedes the learning process. This is particularly a challenge in the online setting, as the agents' strategic behaviors depend on the classifiers announced to them, so their behaviors change over time as the classifiers evolve. Second, the principal can only observe true labels from positively classified agents, without feedback from negatively classified agents. This means that the principal can learn only when a positive classification is made, while the strategic agents are incentivized to manipulate their features to achieve positive classification. Third, the label noise results in noisy feedback, which further complicates the learning process.
Our work contributes to the literature along the following dimensions. First, to the best of our knowledge, our work is the first to study online strategic classification under Massart noise and partial feedback. This advances the literature of learning halfspaces under noise [e.g., Zhang et al., 2020, Diakonikolas et al., 2020] to the strategic setting. Moreover, within the online strategic classification literature, our halfspace model with Massart noise extends the noise-free model of deterministic feature-label relationship in Ahmadi et al. [2021], Shen et al. [2024] and complements the fully adversarial setting [Dong et al., 2018, Chen et al., 2020]. Second, we propose a novel learning algorithm that effectively addresses the aforementioned three key challenges. This algorithm has an initialization-refinement-enhancement pipeline, proceeding in batches and iterations. It features several key components: 1) a localization scheme that iteratively improves the classifiers via online linear optimization, using data within increasingly narrow bands around the classification boundary; 2) a projection-based method to construct proxy features from agents' reported features; 3) a pairwise contrastive inference technique to infer information of the localization bands by contrasting data from pairs of carefully constructed classifiers. Third, we rigorously prove that the proposed algorithm yields classifiers that converge to the clairvoyant optimal one and attains a regret rate of O( √ T ) up to poly-logarithmic and constant factors over T cycles.
this section cite: ['b16', 'b17', 'b19', 'b6', 'b36', 'b11', 'b0', 'b31', 'b14', 'b7']

Section: Related Literature
Strategic Classification Strategic classification, introduced by Hardt et al. [2016], has gained increasing attention. The existing literature has studied strategic classification in both offline settings [e.g., Hardt et al., 2016, Sundaram et al., 2023, Levanon and Rosenfeld, 2021] and online settings.
In online strategic classification, a principal sequentially interacts with strategic agents, aiming to learn accurate classifiers in the presence of strategic feature manipulation. Some literature models agents' strategic behaviors by a manipulation graph that defines agents' feasible feature manipulations [Ahmadi et al., 2023, 2024, Cohen et al., 2024, Shao et al., 2025]. Meanwhile, other literature considers agents that maximize the utility net the cost of feature manipulation. For example, Dong et al. [2018] derive conditions on the manipulation cost function that enable convex optimization techniques to achieve a sublinear regret rate under different fractions of strategic agents. Chen et al. [2020] consider a distance-based manipulation cost function and a zero-one loss function. Our work considers the same cost function and loss function. However, Chen et al. [2020] studies a fully adversarial setting, while our work studies a stochastic setting where agents' true features and labels follow some probability distributions. Our work is closely related to Ahmadi et al. [2021] and Shen et al. [2024], as we study similar models for agents' strategic behaviors and linear classifiers. However, their works focus on the noise-free setting with a deterministic feature-label relationship, while our work tackles label noise.
Notably, nearly all prior studies focus on full feedback settings, whereas our work studies a partial feedback setting. One exception is Harris et al. [2023], where the feedback can be observed also only under a positive decision. However, they consider continuous feedback following a linear regression model with feature-independent noise and target a different objective. In contrast, our work considers binary classification feedback and studies a halfspace model with potentially featuredependent bounded noise, which directly extends the models in Ahmadi et al. [2021], Shen et al. [2024].
Learning Halfspaces with Noise Our paper adopts a halfspace model with the label flipped at a potentially feature-dependent bounded probability, i.e., Massart noise [Massart and Nédélec, 2006].
Recent studies find that even in the absence of strategic agent manipulations, learning halfspaces under Massart noise presents significant challenges [Zhang et al., 2020, Diakonikolas et al., 2019, 2020, 2024]. The key challenge stems from the nonconvexity of the 0-1 loss function that characterizes the misclassification error. A standard approach to overcome the non-convexity of 0-1 loss in classification is to use a convex surrogate loss function [Bartlett et al., 2006]. However, Awasthi et al. [2015] show that popular algorithms such as SVM or hinge loss minimization fails to learn a halfspace that achieves arbitrarily small excess error under Massart noise. More generally, Diakonikolas et al. [2019] show that one cannot achieve non-trivial misclassification error for learning halfspaces under Massart Noise by optimizing convex surrogates. Instead, a "localization" scheme has been proposed to learn halfspaces under a variety of noise models [e.g., Shen, 2021a, Awasthi et al., 2017, Zhang and Li, 2021, Shen, 2021b, Awasthi et al., 2017]. In particular, Zhang et al. [2020] and Diakonikolas et al. [2020] apply localization to learn halfspaces with Massart noise. The core idea is to iteratively improve classification via convex optimization, using data within increasingly narrow bands around the classification boundary. This localization scheme focuses more on data near the classification boundary, as data far away from the boundary tend to be less informative since they can be either easily correctly classified or misclassified mainly due to noise. However, naïvely extending this localization scheme to strategic classification poses significant challenges, because data points close to the classification boundary are the most prone to feature manipulation.
Our work effectively overcomes these challenges by leveraging carefully constructed proxy data and a novel pairwise contrastive inference approach.
this section cite: ['b15', 'b15', 'b32', 'b20', 'b1', 'b8', 'b27', 'b14', 'b7', 'b7', 'b0', 'b31', 'b16', 'b0', 'b31', 'b22', 'b36', 'b10', 'b5', 'b3', 'b3', 'b35', 'b3', 'b36', 'b11']

Section: Notation
We employ the following notation throughout the paper. Boldface letters such as x, r, w denote vectors. The operator • p denotes any ℓ p norm of a vector. The inner product of two vectors is denoted by •, • and the angle between two vectors is represented by θ (•, •), i.e., θ (v 1 , v 2 ) = arccos v1,v2 v1 2 • v2 2 for ∀v 1 , v 2 ∈ R d . Symbols B d and S d denote the d-dimensional Euclidean unit ball and sphere, respectively. B d (R) denotes the ball with radius R > 0. For any positive integer N , [N ] represents the set {1, 2, . . . , N }. The indicator function I(•) gives the value 1 if the event within the parentheses holds and the value 0 otherwise.
this section cite: []

Section: Problem Setup
We consider a setting where a principal repeatedly interacts with sequentially arriving agents (e.g., applicants). Without loss of generality, time is discretized into T cycles, where one agent arrives in each cycle t ∈ [T ]. The agent is characterized by a feature-label pair (x t , y t ), where x t ∈ R d denotes a d-dimensional feature vector and y t ∈ {+1, -1} denotes the agent label (i.e., qualified or not). At the beginning of each cycle t ∈ [T ], the principal announces a classifier ht (•) as the admission rule for the arriving agent. The agent may strategically manipulate and report feature value r t = x t to the principal at some costs, aiming to get admitted (i.e., classified as the positive class, ht (r t ) = +1). The principal observes the reported features r t , makes the classification decision ht (r t ) accordingly, and observes the true label y t only when this agent is admitted. Importantly, the principal has no chance to observe the true label of rejected agents. Based on the data of reported features and admitted agents' labels, the principal aims to learn accurate classifiers over the T cycles.
this section cite: []

Section: Distributional Assumptions
We assume that the feature-label pairs (x 1 , y 1 ), . . . , (x T , y T ) are independently and identically distributed (i.i.d) draws from a common population denoted by (x, y). We need to first impose some distributional assumptions on (x, y).
this section cite: []

Section: Assumption 1 (Halfspace with Massart noise).
There exists a Boolean-valued function h * (x) = sgn( w ⋆ , x ) for a coefficient vector w ⋆ with w ⋆ 2 = 1 and a noise level bound η ∈ [0, 1/2), such that y = h ⋆ (x) with probability 1 -η(x) and y = -h ⋆ (x) with probability η(x), where η(x) characterizes the potentially feature-dependent noise satisfying 0 ≤ η(x) ≤ η almost surely.
Assumption 1 is a standard assumption in the literature of learning halfspaces without strategic manipulation [e.g., Zhang et al., 2020, Diakonikolas et al., 2020, Massart and Nédélec, 2006]. It allows for arbitrary feature-dependent label noise with an upper bound η ∈ [0, 1/2), and relaxes the assumptions in some existing strategic classification literature that assumes a noiseless halfspace model y = h ⋆ (x) and that the positive and negative classes are strictly separated by a margin [e.g., Ahmadi et al., 2021, Shen et al., 2024]. Our assumption can more aptly model real applications where label noises are common and even feature-dependent. Nonetheless, unlike the existing literature, we need to simultaneously handle both the strategic manipulation and label noise. Assumption 2 (Regular feature distribution). Fix constants R, L 1 , L 2 , U 1 , U 2 , δ, Q > 0, and let x V denote the projection of x onto any subspace V ⊆ R d and ϕ V denote its probability density function. The distribution of features x satisfies the following regularity conditions for any 1-dimensional subspace V 1 ⊆ R and any 2-
dimensional subspace V 2 ⊆ R 2 : 1. ϕ V1 (x V1 ) ≥ L 1 and ϕ V2 (x V2 ) ≥ L 2 for any x V1 ∈ V 1 ∩ B 1 (R), x V2 ∈ V 2 ∩ B 2 (R). 2. ϕ V1 (x V1 ) ≤ U 1 and ϕ V2 (x V2 ) ≤ U 2 e -δ x V 2 2 for any x V1 ∈ V 1 , x V2 ∈ V 2 .
3. For any t > 0 and unit vector w ∈ S d , we have that
P[| w, x | ≥ t] ≤ exp(1 -Qt).
In Assumption 2, condition 1 requires that the densities of any 1-dimensional and 2-dimensional projections of feature x are lower bounded around the origin. Condition 2 indicates that these densities have proper upper bounds. Condition 3 requires that the inner product of x with any unit vector w has a sub-exponential tail bound. These conditions generalize the feature distribution conditions in a large body of literature on learning halfspaces with noise [e.g., Diakonikolas et al., 2020, 2021, Zhang et al., 2020, Dasgupta, 2005, Yan and Zhang, 2017, Shen, 2021a, Awasthi et al., 2017]. This existing literature typically assumes that the feature x has an isotropic log-concave distribution, such as a uniform distribution over a unit sphere. In Appendix A.1, we show that Assumption 2 accommodates even non-isotropic log-concave distributions, including many common distributions such as uniform, Gaussian, exponential, logistic distributions, etc. Notably, we impose distributional assumptions on the feature-label pairs, which differ from and complement the fully adversarial setting in the literature [e.g., Dong et al., 2018, Chen et al., 2020, Ahmadi et al., 2024].
this section cite: ['b36', 'b11', 'b22', 'b0', 'b31', 'b11', 'b36', 'b9', 'b34', 'b3', 'b14', 'b7', 'b2']

Section: Agent Feature Manipulation
We assume that each agent gains a utility of +1 for admission (classified as +1) and -1 for rejection (classified as -1). An agent with true feature x may report his feature as r to sway the classifier's decision. Following Shen et al. [2024], Ahmadi et al. [2021], we assume that this misreporting or manipulation incurs a cost Cost(x, r) = 2 xr 2 /γ, where γ > 0 indicates the maximum manipulation distance. Therefore, upon the principal announcing a classifier h(•), the agent's optimal reported feature that maximizes the net utility would be r
⋆ (x, h) = argmax r∈R d h(r) -2 x -r 2 /γ.
Given the linear model in Assumption 1, we restrict the principal's classifier h to linear classifiers parameterized by (w, m) ∈ S d × R, i.e., h(r) = sgn( w, r + m) . In this case, an agent's optimal reported feature is given in the following lemma [Shen et al., 2024, Ahmadi et al., 2021].
Lemma 1. Given an announced classifier h(r) = sgn( w, r + m), the optimal reported feature for an agent with true feature x is r ⋆ (x, h) =
x -( w, x + m)w, -γ ≤ w, x + m < 0;
x, otherwise.
The Clairvoyant Optimal Classifier Under the manipulated feature in Lemma 1, the misclassification rate of a classifier h can be measured by Err( h) := P( h(r ⋆ (x, h)) = y). We hope to characterize a clairvoyant optimal classifier achieving the minimal misclassification rate: h⋆ ∈ argmin h:R d →{±1} Err( h). To this end, we first connect a classifier h under the manipulated feature r with a hypothetical classifier h under the corresponding true feature x.
Proposition 1. For any (w, m) ∈ S d ×R, the output of h(r) = sgn( w, r +m-γ) for r = r ⋆ (x, h) is identical to the output of h(x) = sgn( w, x + m) for any x ∈ R d .
According to Assumption 1, the optimal classifier in absence of manipulation is h * (x) = sgn( w ⋆ , x ). Following Proposition 1, we can achieve the same classification by a corresponding classifier subject to manipulation, which gives the clairvoyant optimal classifier. This structural knowledge of a clairvoyant optimal classifier will guide our algorithm design in Section 3.
Corollary 1. The classifier h⋆ (r) = sgn( w ⋆ , r -γ) minimizes Err( h) = P( h(r ⋆ (x, h)) = y).
Notably, the clairvoyant optimal classifier on the manipulated feature r has a higher threshold to classify an agent into +1 than the corresponding optimal classifier h ⋆ (x) = sgn( w ⋆ , x ) on the true feature x. Indeed, the principal would like to raise the bar for positive classification, in order to avoid errors due to unqualified agents (label -1) who game the classifier by manipulating their features.
Principal's Regret Over the T cycles, the principal learns a sequence of classifiers h = ( h1 , . . . , hT ), where each ht only depends on the observed data of reported features and admitted agents' labels prior to cycle t. The goal is to achieve a small cumulative misclassification rate over all cycles. This is equivalent to achieving a small total suboptimality gap, or regret, relative to the clairvoyant optimal classifier. Formally, the regret is defined as:
Reg( h; T ) := T t=1 Err( ht ) -T × Err( h⋆ ).(1)
This regret corresponds to the "Stackelberg regret" in the strategic classification literature, where the term "Stackelberg" emphasizes that agents consistently choose their best feature manipulation in response to the principal's announced classifiers [Dong et al., 2018, Chen et al., 2020, Ahmadi et al., 2024]. In the next section, we will propose a learning algorithm that effectively tackles the combined challenges of agents' feature manipulations, partial feedback observations, and label noise. We prove that this algorithm achieves a √ T -regret rate up to poly-logarithmic and constant factors.
this section cite: ['b31', 'b0', 'b31', 'b0', 'b14', 'b7', 'b2']

Section: The Algorithm

this section cite: []

Section: Overview of our Algorithm
Algorithm 1: Main-Algorithm
Input: Maximum manipulation distance γ, noise level bound η, lengths {Tinit} ∪ {T k } K k=0 , bandwidths {b k } K k=0 , stepsizes {α k } K k=0 , feature dimension d 1 w0 = Initialization(Tinit) // See Algorithm 2 2 w1 = Refinement( w0, η, T0, b0, α0, d) // See Algorithm 3 3 for k ← 1 to K do 4 w k+1 = Batched-Enhancement(γ, w1, η, k, T k , b k , α k , d) // See Algorithm 4
Our main Algorithm, outlined in Algorithm 1, comprises three sub-algorithms: an Initialization Algorithm (Algorithm 2), a Refinement Algorithm (Algorithm 3) and a Batched Enhancement Algorithm (Algorithm 4). These algorithms are executed sequentially to generate a sequence of coefficient vectors such that the corresponding classifiers converge to the clairvoyant optimal classifier as specified in Corollary 1. Specifically, we partition the horizon of T cycles (one agent arrives in each cycle) into consecutive batches indexed by k ∈ {init, 0, 1, 2, • • • , K}. Index "init" and "0" denote the batches executing the Initialization and Refinement Algorithms, respectively, while indices "1" to "K" represent the K batches that run the Enhancement Algorithm iteratively. Each batch k takes the result of the previous batch k -1 as input. Cycles in each batch k are further grouped into iterations indexed by i ∈ {1, 2, • • • , T k }, where each iteration i performs an update for the coefficient vector w. At the end of batch k, the algorithms output the (normalized) average vectors of the T k iterations in the batch. During Refinement, each iteration consists of only one cycle. In contrast, during Initialization and Enhancement, each iteration contains two cycles, denoted by the superscript j = 1 or 2 to differentiate between the first and second cycles within the same iteration. Note that the indices (k, i, j) can be mapped to the corresponding cycle t, for convenience, we will use these indices in the remainder of this paper.
this section cite: []

Section: Initialization Refinement
Batched Enhancement
w 0 w 1 w 2 w 3 w k w k+1 2T init T 0 2T 1 2T 2 . . . 2T k . . . ⟨w * , w0 ⟩ ≥ Ω(1 -2η) w.h.p. ̸ (w * , w 1 ) ≤ π/4 w.h.p. ̸ (w * , w 2 ) ≤ π/8 w.h.p. ̸ (w * , w 3 ) ≤ π/16 w.h.p. ̸ (w * , w k ) ≤ π/2 k+1 w.h.p. ̸ (w * , w k+1 ) ≤ π/2 k+2 w.h.p.
this section cite: []

Section: Figure 1: Roles of the three sub-algorithms
The roles of the three sub-algorithms are summarized in Figure 1. First, the Initialization Algorithm runs for T init = O ln T /(1 -2η) 2 iterations (with 2T init cycles) to find a coefficient vector w0 such that θ (w ⋆ , w0 ) ≤ π 2 with high probability (see Proposition 2). Second, the Refinement Algorithm takes w 0 = w0 / w0 2 as the initial vector and runs for T 0 = O d ln d ln T /(1 -2η) 8 iterations (with T 0 cycles) to obtain a refined vector w 1 such that θ (w ⋆ , w 1 ) ≤ π 4 with high probability (see Proposition 3). Third, the Batched Enhancement Algorithm runs for K = O log 4 (1 -2η) 4 T /(γd ln d ln T ) batches, where each batch k enhances its initial coefficient vec-
tor w k through T k = O 4 k d ln d ln T /(1 -2η) 4 iterations (with 2T k cycles), yielding a vector w k+1 such that θ (w ⋆ , w k+1 ) ≤ π
2 k+2 with high probability (see Proposition 4). The specification of the algorithms involves absolute constants c 0 to c 7 , which are derived from the parameters in Assumption 2. Detailed calculations are available in Appendix A.5.
this section cite: []

Section: Initialization Algorithm 2: Initialization
Input: Iteration length Tinit 1 for i ← 1 to Tinit do 2 Uniformly draw winit,i ∈ S d 3 for j ← 1 to 2 do 4 Declare h(j) init,i (r) = (-1)
j-1 sgn(⟨winit,i, r⟩), agent (x (j) init,i , y (j) init,i ) arrives and reports r (j) init,i 5 Make classification decision h(j) init,i (r (j) init,i ) and collect label y (j)
init,i if h(j) init,i (r (j) init,i ) = 1 6 return w0 = 1 T init ∑ T init i=1 ∑ 2 j=1 y (j) init,i r (j) init,i I ( (-1) (j-1) ⟨ winit,i, r (j) init,i ⟩ > 0 )
The initialization algorithm runs for T init = O(ln T /(1 -2η) 2 ) iterations. In each iteration, we randomly explore a coefficient vector w init,i ∈ S d and offer two opposing classifiers based on w init,i to two successive agents. Using the reported features r
init,i , r
init,i and true labels y
init,i , y
init,i of all positively classified agents over the T init iterations, we construct an initial coefficient vector w0 .
The design of our initialization algorithm stems from the well-known "averaging" technique for learning halfspaces [Servedio, 2001]. In the non-strategic, noiseless and full feedback setting, y w ⋆ , x = w ⋆ , yx ≥ 0 for all (x, y) ∈ R d × {±1}, so yx forms an acute angle with the optimal normal vector w ⋆ almost surely (since the set {x | w ⋆ , x = 0} is zero-measure). Analogously, in the noisy feedback setting, we have w ⋆ , E[yx] > 0, so E[yx] forms an acute angle with w ⋆ . In a non-strategic and full feedback setting, the literature uses the sample average of yx to approximate E[yx] as an initial estimate of w ⋆ [Zhang et al., 2020]. However, this estimator is unavailable for us because of agents' feature manipulation and the partial feedback setting. Instead, our algorithm declares pairs of opposite classifiers. We collect and average the yr of agents whose reported feature r falls above the hyperplane. Note that these agents report their features truthfully (r = x), so we are able to form w0 from these agents' yr as a proper approximation of E [yx]. We can show that, for large enough T init , this vector w0 forms an acute angle with the optimal w ⋆ with high probability. Proposition 2. For some constants c 0 , c 1 > 0, when Algorithm 2 runs for T init = c 0 ln T /(1 -2η) 2 iterations, its output w0 satisfies w ⋆ , w0 > c 1 (1 -2η) > 0 and θ (w ⋆ , w0 ) ≤ π 2 with probability at least 1 -2/T 2 .
this section cite: ['b25', 'b36']

Section: Refinement of the Initial Coefficient Vector
Algorithm 3: Refinement Input: Initial vector w0, noise level η, iteration length T0, bandwidth b0, step size α0, feature dimension d Initialization :w0,1 = w0/∥ w0∥2 1 for i ← 1 to T0 do 2 Declare classifier h0,i(r) = sgn(⟨w0,i, r⟩), agent (x0,i, y0,i) arrives and reports r0,i 3 Make classification decision h0,i(r0,i) and collect label y0,i if h0,i(r0,i) = 1 4 Compute gradient: g0,i = [-ηr0,iI (y0,i = 1) + (1 -η)r0,iI (y0,i = -1)]I(0 < ⟨w0,i, r0,i⟩ ≤ b0)
5 Set constraint set: W0 = {w | ∥w∥2 ≤ 1, ⟨w, w0⟩ ≥ c1(1 -2η)} 6 Update w: ŵ0,i+1 = arg minw∈W 0 ⟨g0,i, w⟩ + 1 α 0 ∥w-w 0,i ∥ 2 p 2(p-1)
, where p = ln(8d) ln(8d)-1 7 Normalize: w0,i+1 = ŵ0,i+1/∥ ŵ0,i+1∥2
8 Compute mean vector: w1 = 1 T 0 ∑ T 0 i=1 w0,i 9 return w1 = w1/∥ w1∥2
The refinement algorithm adopts a "localization" scheme to refine the output w0 of Algorithm 2 to better approximate w ⋆ . In every iteration i, we consider only data within a band 0 < w 0,i , r ≤ b 0 adjacent to the boundary of the current classifier h0,i (r) = sgn( w 0,i , r ). Agents in this band are positively classified and have no incentives for feature manipulation, allowing us to observe both the true feature x = r and the true label y. Moreover, this effectively probes the localized region D 0,i = {x : 0 < w 0,i , x ≤ b 0 } in the true feature space. Similar "localization" is widely used in the literature of learning half-spaces with label noises (see references in Section 1.1), since data near the classification boundary is the most informative, while data far from the boundary are either correctly classified with ease or are misclassified mainly due to noises, providing little information.
We formulate an online linear optimization problem with constructed losses {w → w, g0,i } T0 i=1 over a proper constraint set W 0 = {w | w 2 ≤ 1, w, w0 ≥ c 1 (1 -2η)}. This constraint set, according to Proposition 2, contains w ⋆ with high probability. We then solve this problem by online mirror descent with a stepsize α 0 and regularizer ww 0,i 2 p /2(p -1) for p = ln(8d)/(ln(8d) -1), perform proper normalization in each iteration, and normalize the average of all iterates to obtain the output w 1 . By focusing on the band {r : 0 < w 0,i , r ≤ b 0 }, we can perfectly observe r 0,i = x 0,i and y 0,i and the gradientsfoot_0 g0,i coincide with the counterparts in Zhang et al. [2020]. As a result, we can follow their analysis to bound the error of the output w 1 .
Proposition 3. For the constant c 1 in Proposition 2 and some constants c 2 , c 3 , c 4 > 0, when the initial vector w0 satisfies w ⋆ , w0 ≥ c 1 (1 -2η) and Algorithm 3 runs with bandwidth b 0 = c 2 (1 -2η) 2 for T 0 = c 3 d ln d(ln T ) 2 /(1 -2η) 8 iterations with step size α 0 = c 4 d ln(d)/( √ T 0 ln T ), then its output w 1 satisfies θ (w ⋆ , w 1 ) ≤ π/4 with probability at least 1 -3/T 2 .
The main idea in proving Proposition 3 is outlined as follows. By the theory of online convex optimization, we can upper bound the cumulative regret for the constructed loss in this stage, i.e., T0 i=1 w 0,i , g0,iw ⋆ , g0,i . This regret bound, together with a bound on T0 i=1 w 0,i , g0,i and a concentration bound on
T0 i=1 w ⋆ , -g 0,i , leads to a high probability upper bound on T0 i=1 E [ w ⋆ , -g 0,i ]. Moreover, it can be shown that E [ w ⋆ , -g 0,i
] is lower bounded by θ (w ⋆ , w 0,i ) up to some proportional factors. This is why we expect to obtain a high probability upper bound on θ (w ⋆ , w 1 ). Importantly, the gradients g0,i are carefully constructed to ensure that | w 0,i , g0,i | is small and meanwhile E [ w ⋆ , -g 0,i ] upper bounds θ (w ⋆ , w 0,i ).
Notably, while Algorithm 3 collects true feature-label data and implements localization by focusing on local bands around the origin-crossing classification hyperplanes h0,i 's, this approach can be costly. According to Lemma 1, unqualified agents with true features x satisfying -γ ≤ w 0,i , x ≤ 0 would manipulate their features to achieve positive classifications, resulting in constant instantaneous regret. Fortunately, Algorithm 3 runs for only Õ(ln T ) cycles, so this refinement algorithm obtains an improved coefficient w 1 for the next stage at the cost of at most only Õ(ln T ) regret.
this section cite: ['b36']

Section: Batched Enhancement: Proxy Features and Pairwise Contrastive Inference
Algorithm 4: Batched Enhancement Input: Maximum manipulation distance γ, initial vector w k , noise level η, batch index k, iteration length T k , bandwidth b k , step size α k , feature dimension d Initialization :
w k,1 = w k 1 for i ← 1 to T k do 2 Construct classifiers h(1) k,i (r) = sgn(⟨w k,i , r⟩ -γ) and h(2) k,i (r) = sgn(⟨w k,i , r⟩ -γ -b k ) 3 for j ← 1 to 2 do 4
Declare classifier h(j) k,i , agent (x (j) k,i , y (j) k,i ) arrives and reports r (j) k,i 5 Make classification decision h(j) k,i (r k,i ) and collect label y (j)
k,i if h(j) k,i (r (j) k,i ) = 1 6
Construct proxy data:
x(j,+)
k,i = Proj + D k,i (r (j) k,i )I(y (j) k,i = 1, r (j) k,i ∈ D(j) k,i ), x(j,-) k,i = Proj - D k,i (r (j) k,i )I(y (j) k,i = -1, r (j) k,i ∈ D(j) k,i ) 7
Use the proxy data to compute the gradient: ĝk,i = -η(x
(1,+) k,i - x(2,+) k,i ) + (1 -η)(x (1,-) k,i - x(2,-) k,i ) 8 Update: ŵk,i+1 ← arg minw∈W k ⟨ĝ k,i , w⟩ + 1 α k ∥w-w k,i ∥ 2 p 2(p-1)
, where p = ln(8d) ln(8d)-1 , the constraint set
W k = {w| ∥w∥2 ≤ 1, ⟨w, w k ⟩ ≥ cos θ k }, starting angle θ k = π 2 k+1 9 Normalize: w k,i+1 = ŵk,i+1 /∥ ŵk,i+1 ∥2 10 Compute mean vector wk+1 = 1 T k ∑ T k i=1 w k.i 11 return w k+1 = wk+1 /∥ wk+1 ∥2
In the non-strategic and full feedback setting, after obtaining the refined coefficient w 1 , Zhang et al. [2020] further improves it by solving a sequence of adaptively constructed online linear optimization problems min w∈W k
T k i=1 w, g k,i with g k,i = [-ηx k,i I (y k,i = 1) + (1 - η)x k,i I (y k,i = -1)]I(-b k < w k,i , x k,i ≤ b k ) via mirror descent over k = 1 . . . , K batches, using local data within increasingly narrow bands {x | -b k < w k,i , x ≤ b k } around the classifi- cation hyperplanes.
This process can geometrically reduce the error of the coefficient estimates, outputting a final classifier that approaches the optimal classifier after enough batches. The key ingredient underlying this guarantee is that the gradients g k,i are well constructed so that | w k,i , g k,i | is small and meanwhile E [ w ⋆ , -g k,i ] upper bounds θ (w ⋆ , w k,i ) (see discussions below Proposition 3). One may consider directly implementing this batched enhancement approach in our strategic classification. In particular, one may again use classifiers hk,i (r) = sgn( w k,i , r ) and focus on the band D k,i = {x | 0 < w k,i , x ≤ b k } in each batch k and iteration i, since this enables us to collect the true feature-label data and probe the localized region D k,i . However, as we discussed at the end of Section 3.3, this approach may result in constant instantaneous regret in every cycle due to unqualified strategic agents, so that O(T ) regret accumulates over the O(T ) cycles in this stage.
To avoid excessive errors due to feature manipulation, we can instead employ classifiers hk,i (r) = sgn( w k,i , r -γ), mimicking the form of the clairvoyant optimal strategic classifier h⋆ and raising the bar for positive classification to tackle strategic behaviors (see Corollary 1). However, this gives rise to new challenges: it is unclear how to construct the gradients g k,i and probe the localized regions D k,i , since both depend on the true features, but all agents in the localized regions D k,i misreport their features. This means that we know neither which agents' true feature values belong to the regions D k,i nor their true feature values. To tackle these challenges, we propose two key ideas: proxy features and pairwise contrastive inference.
Proxy Features Even if we assume, for the sake of argument, that we can identify agents whose true features lie in D k,i , their true feature values remain unobservable, since they all misreport their features to secure positive classification (so their reported feature values fall on the hyperplane of the announced classifier). To resolve this, we construct proxy features from the reported features.
Specifically, consider an agent with true feature value x ∈ D k,i and reported feature value r. This agent will manipulate his feature to get positively classified, and thus we can observe his true label. If his true label is y = +1, then we construct his proxy feature x as the projection of r onto the upper boundary of D k,i , i.e., x = Proj + D k,i (r) := r + (b k -w k,i , r )w k,i . On the contrary, if his true label is y = -1, then we construct his proxy feature x as the projection of r onto the lower boundary of D k,i , i.e., x = Proj - D k,i (r) := rw k,i , r w k,i . As a result, this agent's proxy feature value, like his true feature value, also belongs to D k,i , and the proxy feature value under a positive label (i.e., projection onto the upper boundary of D k,i ) is more aligned with the direction of positive classification than the proxy feature value under a negative label (i.e., projection onto the lower boundary of D k,i ). See the illustration in Figure 2(a).
Using the proxy features, we can approximate the ideal gradient g k,i by a proxy gradient gk,i = [-ηProj + D k,i (r k,i )I (y k,i = 1) + (1 -η)Proj - D k,i (r k,i )I (y k,i = -1)]I(x k,i ∈ D k,i ). Although this may not exactly recover the ideal gradient, it is still effective, in that | w k,i , gk,i | is small and Appendix A.5). Therefore, we can use the proxy gradients gk,i in the algorithm to achieve similar guarantees. Nevertheless, these proxy gradients require knowing whether an agent's true feature value belongs to the localized region D k,i or not, which is still infeasible in our setting. This motivates our second key idea.
E [ w ⋆ , -g k,i ] ≥ E [ w ⋆ , -g k,i ] also upper bounds θ (w ⋆ , w k,i ) (see
this section cite: []

Section: Pairwise Contrastive Inference
We propose to offer two classifiers h(1)
k,i (r) = sgn( w k,i , r -γ) and h(2) k,i (r) = sgn( w k,i , r -γ -b k ) successively in each iteration. Under classifier h(1) k,i (r)
, we consider only agents with reported features in D(1) k,i = {r : γ ≤ w k,i , r ≤ γ + b k }, while under classifier h(2) k,i (r), we consider only agents with reported features in D(2) k,i = {r : w k,i , r = γ +b k }. These agents are all classified into the positive class, so their true labels are observed. Moreover, according to the feature manipulation rule in Lemma 1, these agents have true feature values in D (1)
k,i = {x : 0 ≤ w k,i , x ≤ γ + b k } and D (2) k,i = {x : b k ≤ w k,i , x ≤ γ + b k }, respectively. Since D k,i = D (1) k,i \ D (2)
k,i up to a measure-zero set, we can expect to infer distributional properties of the data within the region D k,i of interest by contrasting the data within D (1) k,i and the data within D
(2) k,i . We call this a pairwise contrastive inference approach, which is illustrated in Figure 2(b). We can use this approach to infer the two key components in the proxy gradient g k,i . Note
E Proj + D k,i (r k,i )I (y k,i = 1, x k,i ∈ D k,i ) = E x(1,+) k,i - x(2,+) k,i
, where x(j,+)
k,i = Proj + D k,i (r (j) k,i )I y (j) k,i = 1, r(j)
k,i ∈ D(j) k,i for j = 1, 2. This means that we can use x(1,+)
k,i - x(2,+) k,i
to unbiasedly infer one key component of gk,i in expectation. Similarly, we can construct x(1,-)
k,i - x(2,-) k,i
to infer the other component. This gives our gradient estimate ĝk,i in Algorithm 4 Line 7, satisfying that E [ w ⋆ , -ĝ k,i ] = E [ w ⋆ , -g k,i ] also upper bounds θ (w ⋆ , w k,i ).
After getting the gradient estimator ĝk,i , we again conduct online mirror decent with a regularizer similar to that in Algorithm 3 and constraint set W k = {w| w 2 ≤ 1, w, w k ≥ cos θ k }, where θ k = π 2 k+1 . Then we output the normalized average coefficient vector w k+1 for the next batch. Our constraint set ensures that θ(w k,i , w k ) ≤ π/2 k+1 for all i ∈ [T k ]. Then, when the input vector
w k satisfies θ (w ⋆ , w k ) ≤ π/2 k+1 , we have θ(w ⋆ , w k,i ) ≤ θ(w ⋆ , w k ) + θ(w k,i , w k ) ≤ π/2 k
by a triangular inequality shown in Appendix A.3, Lemma 12. This statement is critical: First, it controls the expected cumulative error in batch k to be O 1 2 k • T k . Second, the condition that θ(w ⋆ , w k,i ) ≤ π/2 k , together with our localized online mirror descent method, ensures that batch k outputs a vector w k+1 that satisfy θ(w ⋆ , w k+1 ) ≤ π/2 k+2 with high probability (see Proposition 4), which is in turn required by the next batch. Proposition 4. For some constants c 5 , c 6 , c 7 > 0, when Algorithm 4 runs with an initial vector
w k satisfying θ (w ⋆ , w k ) ≤ θ k = π/2 k+1 , bandwidth b k = c 5 (1 -2η)2 -k for T k = c 6 4 k (γ + 1)d ln d(ln T ) 2 /(1 -2η) 4 iterations with step size α k = c 7 √ d ln dθ k /( √ T k ln T ), its output w k+1 satisfies θ (w ⋆ , w k+1 ) ≤ θ k+1 = θ k
2 with probability at least 1 -6/T 2 . Proposition 4 shows that Algorithm 4 enhances its input by reducing the error by half in every batch, generating a sequence of coefficient estimates (w k ) K k=1 with geometrically decaying errors. Notably, we achieve the enhancement by classifiers h(1) k,i , h(2) k,i that use at least γ classification thresholds and are hence more resilient to errors due to strategic classification, which results in only a sublinear regret, as we will show in Theorem 1.
x1 (r1) x2 (r2) w k,i w ⋆ b k γ O (a) Proxy Feature x1 (r1) x2 (r2) w k,i w ⋆ b k γ γ b k O D (1) k,i D (2) k,i D k,i (b) Contrastive Inference
true feature x with label +1 true feature x with label -1 reported feature r with label +1 reported feature r with label +1 proxy feature x with label +1 proxy feature x with label +1 region
{x | ⟨w ⋆ , x⟩ ≥ 0} region {x | ⟨w ⋆ , x⟩ < 0} the band of interest D k,i = {x | 0 ≤ ⟨w k,i , x⟩ ≤ b k } hyperplane of optimal classifier h ⋆ (x) = sgn(⟨w ⋆ , x⟩) hyperplane of h(1) k,i (r) = sgn(⟨w k,i , r⟩ -γ)} hyperplane of classifier h(2) k,i (r) = sgn(⟨w k,i , r⟩ -γ -b k ) 1
Figure respectively, through which we infer the information for agents in the region D k,i of interest.
this section cite: []

Section: Regret Guarantee
We now provide a formal regret guarantee of Algorithm 1, showing that it achieves a sublinear regret dependent on the noise level η and feature dimension d. Theorem 1. For any instance of our online strategic classification problem with noise level η, maximum manipulation distance γ, and feature dimension d, the expected regret of classifiers h from Algorithm 1 over T cycles satisfies
E[Reg( h; T )] = O d ln d × (ln T ) 2 /(1 -2η) 8 + (γ + 1)d ln d × T ln T /(1 -2η) 2 .
We prove the theorem by analyzing the regret incurred by each of the three sub-algorithms in Section 3. The full proof is outlined in Appendix A.6. We also conduct numerical experiments to evaluate our proposed algorithm, with results presented in Appendix A.2.
this section cite: []

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract accurately reflects the paper's contributions and scope by stating that the paper addresses the online strategic classification problem and proposes a novel learning algorithm that converges to the optimal classifier and achieves a regret rate of O( √ T ) (up to poly-logarithmic and constant factors). It also clearly outlines the combined challenges of agents' strategic feature manipulations, partial label observations, and label noises. Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: In Section 5, we acknowledge our limitations and point out some future directions.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: All assumptions are clearly stated in Assumption 1 and Assumption 2. We provide a complete proof in our appendix and provide proof sketches in Section 3 and Section 4.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: We use Python 3.9 to conduct our numerical experiments. All settings and results are listed in Appendix A.2. We guarantee that our results are genuine and credible.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The paper provides the code and data required for the experiments in the supplementary material. The code is well-organized and well-documented, which facilitates the reproduction process.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: All details of our numerical experiment is listed in Appendix A.2. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We present the mean regret rate of all algorithms, calculated across 10 independent experimental replications in Appendix A.2 The experimental findings are consistent with the theoretical assurances we provide in Section 4.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We state in Appendix A.2 that all experiments can be conducted locally using a standard CPU without requiring specialized hardware, making reproduction accessible and straightforward.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: Yes, the research performed in the paper conforms with the NeurIPS Code of Ethics. The paper does not involve human subjects or sensitive personal information, so issues like privacy and consent are not applicable. The research focuses on developing an algorithm for strategic classification under specific noise conditions, and it does not present any foreseeable risks of harm, discrimination, or other unethical consequences as outlined in the Code of Ethics. Our numerical experiment only uses simulated data, so it does not involve any deprecated datasets or copyright violations.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: We discuss in Section 1 that agents' strategic feature manipulation can hurt a certain classification rule, and we design an algorithm to prevent this, which is a potential positive societal impact. This algorithm aims to enhance the fairness and accuracy of automated decision-making systems, potentially reducing financial losses from misclassifications. While the research focuses on foundational aspects and does not directly address all potential negative societal impacts, we acknowledge the importance of considering such implications as the technology evolves.
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
Justification: The research in this paper focuses on developing an algorithm for online strategic classification and does not involve the release of data or models with high misuse risks, such as pretrained language models, image generators, or scraped datasets. Therefore, no specific safeguards for such releases are applicable.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [NA] Justification: The research in this paper focuses on developing a novel algorithm for online strategic classification. We do not use any existing assets such as code, data, or models from external sources. All methods and experiments are designed and implemented by us, so there are no original owners or licenses to credit.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The research in this paper does not release any new assets such as datasets, code, or models. Therefore, no new assets are introduced that would require documentation.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The research in this paper does not involve crowdsourcing experiments or research with human subjects. Therefore, there are no instructions, screenshots, or compensation details to provide.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According
to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This research does not involve crowdsourcing or research with human subjects. Therefore, there are no risks to participants, disclosures, or IRB approvals to report. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core methodology of this research does not involve the use of LLMs as any important, original, or non-standard components. LLMs were not utilized in developing the algorithms or conducting the experiments. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: A Technical Appendices and Supplementary Material

this section cite: []

Section: A.1 Feature Regularity Conditions and General Log-concave Distributions
The literature on learning halfspaces with noise typically assumes that the feature vector x has an isotropic log-concave distribution [Balcan and Long, 2013, Awasthi et al., 2017, Zhang et al., 2020, Shen, 2023]. The log-concave and isotropic log-concave distributions are defined as follows.
Definition 1 (Isotropic log-concave distribution [Lovász and Vempala, 2007]). A random vector z over R d with probability density function ϕ z (•) follows a log-concave distribution if ln ϕ z (•) is concave. Moreover, it is isotropic if E [z] = 0 and E zz T = I.
The following lemma summarizes some important properties of (isotropic) log-concave distributions that have been proved by literature.
Lemma 2. Suppose z ∈ R d with probability density function ϕ z (•) follows a log-concave distribution. Then, the following holds.
(a) (Klivans et al. [2009] Lemma 5.17) For d = 1, assume that E z 2 = C 2 , then for every
t > 0, P (| z | > t) ≤ e -Ct+1 .
Moreover, if z is isotropic, then, (b) (Lovász and Vempala [2007] Lemma 5.2) ϕ z (z) ≥ β 1 (d) for all 0 ≤ z 2 ≤ 1/9, where (Lovász and Vempala [2007] Lemma 5.5) For d = 1, ϕ z (z) ≤ 1.
β 1 (d) = 2 -8d . (c)
(d) (Klivans et al. [2009] Lemma 7 ) For d ≥ 2, ϕ z (z) ≤ β 2 (d)e -β3(d) z 2 , where β 2 (d) = 2 8d d d/2 e and β 3 (d) = 2 -7d 2(d-1)(20(d-1)) (d-1)/2 .
Following Lemma 2, one can show that any mean-zero isotropic log-concave distribution satisfies the regularity conditions in assumption 2. Importantly, in this part, we show that the regularity conditions can hold even for a mean-zero log-concave distribution that is not isotropic. In this case, eigenvalue bounds on the covariance matrix of the distribution determine the corresponding regularity parameters. Lemma 3. Let x ∈ R d (d ≥ 2) have zero mean and a log-concave distribution. Suppose the eigenvalues of its covariance matrix Σ = E xx are all bounded within [λ, λ] for some positive constants λ, λ, then the distribution of x satisfies the regularity conditions in assumption 2, with parameters
L 1 = β1(1) √ λ , L 2 = β1(2) λ , R = 1 9 √ λ, U 1 = 1 √ λ , U 2 = β2(2) λ , δ = β3(2) √ λ , Q = √ λ for β 1 (1), β 1 (2), β 2 (2), β 3 (2)
given in Lemma 2.
this section cite: ['b4', 'b3', 'b36', 'b30', 'b21', 'b18', 'b21', 'b21']

Section: A.2 Numerical Experiments
In this subsection, we conduct numerical experiments to evaluate our proposed algorithm. To highlight the challenges posed by Massart Noise and strategic behavior, and to demonstrate the effectiveness of our algorithm, we compare its regret against two benchmarks: (1) the Strategic Perceptron algorithm from Ahmadi et al. [2021], designed for noiseless online strategic classification, and (2) the PAC learning algorithm for halfspaces with Massart Noise from Zhang et al. [2020], designed for non-strategic classification. Note that these benchmarks are both originally designed for full feedback settings, whereas our work focuses on partial feedback. We evaluate the performance of these benchmark algorithms both when they have access to full feedback (while our algorithm does not) and when they only use partial feedback as our algorithm.
We test the algorithms under two different settings, with key parameters outlined in Table 1. Each setting is replicated 30 times, and we report the average regret for each algorithm. Our analysis includes a performance comparison of the different algorithms and an investigation of how various problem parameters influence our proposed algorithm.
this section cite: ['b0', 'b36']

Section: Benchmark against Strategic Perceptron by Ahmadi et al. [2021]
To understand the impact of Massart Noise, we compare our algorithm with the Strategic Perceptron algorithm from Ahmadi et al. [2021]. This algorithm provably achieves only a finite number of mistakes under a noiseless model where the feature-label relationship is deterministic and the true and negative classes are strictly separated by a positive margin. Ahmadi et al. [2021] modify the classical Perceptron algorithm by setting a higher threshold for classifying an agent as positive and proxy surrogate features to estimate the agents' true features. Their proxy feature is defined as follows.
Definition 2 (x t , proxy feature , Ahmadi et al. [2021]). For a given classifier h(•) = sgn( w, • +m), an agent (x, y) reports his feature as r according to Lemma 1. Then the corresponding proxy feature xt in Ahmadi et al. [2021] is defined as xt = r t -γw w, r t = γ and y t = -1; r t otherwise.
(2)
Algorithm 5: Original Strategic Perceptron with Full Feedback (Ahmadi et al. [2021])
Accept the first agent without declaring any classifier if y1 = 1 then
ŵ2 ← -r1 else ŵ2 ← --r1 w2 ← -ŵ2/∥ ŵ2∥2 for t = 2 • • • , T do Declare classifier ht(r) = sgn(⟨wt, r⟩ -γ), receive agent response rt
Classify the agent as ŷt = ht(rt)
if yt ̸ = ŷt then ŵt+1 ← -wt + yt xt, wt+1 ← -ŵt+1/∥ ŵt+1∥2 else wt+1 ← -wt
Their original algorithm, designed for the full feedback setting, is presented in Algorithm 5. Algorithm 6 below directly adapts this algorithm to our partial feedback setting. Specifically, instead of using all data points that incur misclassifications for update, the refined algorithm uses only positively classified agents with true labels -1 to adjust the coefficient vector.  1. Results are based on 30 independent replications of the experiment.
Figure 3 illustrates the average regret of each algorithm over 50,000 cycles in both settings listed in Table 1. We observe that our algorithm's regret grows sublinearly, while the two benchmarks' regrets may accumulate linearly. The original Strategic Perceptron with full feedback (the dashed red line) demonstrates slightly better performance compared to its partially feedback-modified counterpart (the dashed green line). However, the improvement remains marginal, suggesting that our partial feedback setting is not the primary cause of the Strategic Perceptron's failure. Intuitively, the ineffectiveness of the two benchmarks stems from their sensitivity to noise when updating with data from all admitted agents. When θ(w ⋆ , w t ) is relatively small but w t , r t (equals w t , x t when w t , r t > γ) is large, the algorithm is more likely to admit a 'wrong' agent (y t = -1 but h(r t ) = 1) due to noise rather than an inaccurate classifier. Since the perceptron algorithm updates are based on mistakes (Rosenblatt [1958]), the presence of noise increases the probability of misleading updates for the classifiers. In contrast, our algorithm explores a small band near the decision boundary, whose bandwidth decreases proportionally to θ(w ⋆ , w k ), k = 1, 2, • • • , K across batches. Within this band, wrong admissions are more likely due to suboptimal classifiers than noise, making the update more effective. This enables our algorithm to gradually converge to the optimal decision.
this section cite: ['b0', 'b0', 'b0', 'b0', 'b24']

Section: Benchmark against Non-Strategic Learning under Massart Noise by Zhang et al. [2020]
Next, to highlight the impact of agents' strategic behavior, we compare our algorithm against the algorithm proposed by Zhang et al. [2020] (see Algorithm 7), which is designed for adaptively learning halfspaces with Massart Noise in the non-strategic classification setting. Their algorithm also adopts a localization scheme that focuses on data within an increasingly narrow band near the classification boundary and uses online mirror descent in batches for classifier updates. However, they do not consider the impact of agents' strategic behavior. We test the performance of their algorithm under strategic manipulation in both full feedback and partial feedback settings. The two settings differ in: 1) whether the principal can collect labels of those who are negatively classified (full feedback) or not (partial feedback) and 2) the algorithm chooses different bandwidths for updates, namely, {r | -b k ≤ w k,i , r k,i ≤ b k } for the full feedback setting and {r | 0 < w k,i , r k,i ≤ b k } for the partial feedback setting.
As shown in Figure 4, after a common pure exploration phase, the regret of the non-strategic learning algorithm in both full feedback and partial feedback settings grows linearly. This is because the nonstrategic learning algorithm ignores agents' strategic manipulation. Consequently, true negative agents may misreport their features to be positively labeled. In contrast, our algorithm accounts for agents' strategic behavior and is able to efficiently learn the ground truth distribution.  g0,i = [-ηr0,iI (y0,i = ŷ0,1) + (1 -η)r0,iI (y0,i ̸ = ŷ0,1)]I(-b0 ≤ ⟨w0,i, r0,i⟩ ≤ b0) if F = "partial" then Make classification decision ŷ0,1 = h0,i(r0,i) and collect label y0,i only if ŷ0,i = 1
if F = "full" then collect label yinit,i if F ="partial" then collect label yinit,ionly if hinit,i(rinit,i) = 1 if F = "full" then return w0 = 1 T init ∑ T init i=1 rinit,iyinit,i if F = "partial" then return w0 = 1 T init ∑ T init i=1 rinit,
Compute gradient: g0,i = [-ηr0,iI (y0,i = ŷ0,1) + (1 -η)r0,iI (y0,i ̸ = ŷ0,1)]I(0 < ⟨w0,i, r0,i⟩ ≤ b0)
Set constraint set: W0 = {w | ∥w∥2 ≤ 1, ⟨w, w0⟩ ≥ c1(1 -2η)} Update w: ŵ0,i+1 = arg minw∈W 0 ⟨g0,i, w⟩ + 1 α 0 ∥w-w 0,i ∥ 2 p 2(p-1)
, where p = ln(8d) ln(8d)-1
Normalize: w0,i+1 = ŵ0,i+1/∥ ŵ0,i+1∥2
Compute mean vector: w1 = 1
T 0 ∑ T 0 i=1 w0,i return w1 = w1/∥ w1∥2
Impact of Different Parameters We examine three groups of additional settings to analyze the impact of different parameters. For each group, we test both settings from Table 1. The average regret over 30 independent experiments for each group is depicted in fig. 5 up to 50,000 cycles.
We first examine how different maximum manipulation distances γ = 0.1, 0.2, 0.5 affect the regret of our algorithm. As depicted in Figure 5 (a1) and (a2), larger γ values result in higher regret. Intuitively, a larger γ permits more agents to manipulate their features, so the strategic manipulation problem becomes more severe. This causes all algorithms to have worse performance.
Next, we vary the feature space dimension, setting w ⋆ to be (1, 0) (d = 2), (1, 0, 0, 0) (d = 4), and (1, 0, 0, 0, 0, 0) (d = 6), respectively. The average regret across different time horizons is shown in Algorithm 10: Non-Strategic-Batched Enhancement Input: Feedback setting F , initial vector w k , noise level η, batch index k, iteration length T k , bandwidth b k , step size α k , feature dimension d Initialization :w k,1 = w k 1 for i ← 1 to T k do 2 Declare classifier hk,i (r) = sgn(⟨w k,i , r⟩), agent (x k,i , y k,i ) arrives and reports r k,i 3 if F = "full" then 4 Make classification decision ŷk,i = hk,i (r k,i ) and collect label y k,i Update: ŵk,i+1 ← arg minw∈W k ⟨ĝ k,i , w⟩ + 1
α k ∥w-w k,i ∥ 2 p 2(p-1)
, where p = ln(8d) ln(8d)-1 , the constraint set  1. Results are based on 30 independent replications of the experiment. fig. 5 (b1) and (b2). As expected, the d = 2 setting yields the lowest regret, while d = 6 setting yields the highest, consistent with our regret bound.
W k = {w| ∥w∥2 ≤ 1, ∥w -w k ∥2 ≤ θ k , starting angle θ k = π 2 k+1 10 Normalize: w k,i+1 = ŵk,i+1 /∥ ŵk,i+1 ∥2 11 Compute mean vector ŵk+1 = 1 T k ∑ T k i=1 w k.i 12 return w k+1 = ŵk+1 /∥ ŵk+1 ∥2
We finally investigate the impact of the noise level η on our algorithm's convergence in fig. 5 (c1) and (c2), setting η to 0.1, 0.2 and 0.4. Surprisingly, the impact of the noise level manifests in opposite trends across the two settings. In setting 1, a higher noise level results in greater regret when T is large enough. Conversely, in setting 2, increased noise levels lead to a lower regret rate. This discrepancy might stem from the fact that, as the noise level rises, the learning accuracy of the clairvoyant optimal classifier diminishes. Given that regret is defined as the difference in cumulative error between our algorithm's classifiers and the clairvoyant optimal ones, the noisier environment could potentially narrow this gap.
this section cite: ['b36']

Section: A.3 Technical Lemmas
In this subsection, we list some technical lemmas as instruments for our further proofs. Lemma 4. Suppose that the distribution of a random vector x ∼ D x satisfies the regularity conditions outlined in Assumption 2, then, it has the following properties. (a) For ∀w ∈ S d and ∀ b > 0, L 1 min{R, b} ≤ P (0 ≤ w, x ≤ b) ≤ U 1 b. (b) There exist positive constants c 8 , c 9 > 0, such that for any two unit vectors
this section cite: []

Section: Properties of Regular Distributions
v 1 , v 2 ∈ S d , if 0 ≤ θ (v 1 , v 2 ) ≤ π 2 , then c 8 P (sgn( v 1 , x ) = sgn( v 2 , x )) ≤ θ (v 1 , v 2 ) ≤ c 9 P (sgn( v 1 , x ) = sgn ( v 2 , x )) .
(
Proof. (a) Since w, x forms a projection of x onto a certain 1-dimensional hyperplane, property (a) trivially holds by conditions 1 and 2 in Assumption 2.
(b) Let z := ( v 1 , x , v 2 , x ), which is a projection of x onto a 2-dimensional subspace V 2 spanned by v 1 and v 2 . Let ϕ V2 (•) and D V2 denote its density and distribution, respectively. Let G V2 := {z | sgn(z 1 ) = sgn(z 2 )}, then,
P x∼Dx (sgn( v 1 , x ) = sgn( v 2 , x )) = P z∼D V 2 (z ∈ G V2 ) = z∈G V ϕ V2 (z) dz ≥ z∈G V ∩B 2 (R) L 2 dz ≥ L 2 R 2 θ (v 1 , v 2 ) ,
where the first inequality holds by condition 1 of Assumption 2 that L 2 ≤ ϕ V2 (z) for all z 2 ≤ R.
this section cite: []

Section: The last inequality holds by an observation that z∈G
V 2 ∩B 2 (R) 1 dz ≥ R 2 θ(v 1 , v 2 ).
Hence, we prove the first inequality of (3).
To prove the second inequality of (3), for ∀ϵ > 0, we have
P x∼Dx (sgn( v 1 , x ) = sgn( v 2 , x )) = P z∼D V 2 (z ∈ G V2 ) ≤ P z∼D V 2 (z ∈ G V2 , z 2 ≤ ϵ) + P z∼D V 2 ( z 2 > ϵ) ≤ z∈G V 2 ∩B 2 (ϵ) ϕ V2 (z) + P x∼Dx (| v 1 , x | > ϵ) + P x∼Dx (| v 2 , x | > ϵ) = z∈G V 2 ∩B 2 (ϵ) ϕ V2 (z) + P x∼Dx (| v 1 , x | > ϵ) + P x∼Dx (| v 2 , x | > ϵ) ≤ U 2 θ (v 1 , v 2 ) ϵ 2 + 2 exp(1 -Qϵ),
where the last inequality holds by the fact that ϕ V2 (z) ≤ U 2 exp(-δ z 2 ) ≤ U 2 according to Assumption 2 condition 2 and P x∼Dx (| w, x | > ϵ) ≤ exp(1 -Qϵ) for ∀w ∈ S d according to Assumption 2 Condition 3. Taking ϵ = 1-ln(θ(v1,v2)) Q , then we have
P x∼Dx (sgn( v 1 , x ) = sgn( v 2 , x )) ≤ U 2 Q 2 + 2 θ (v 1 , v 2 ) .
Thus, we complete the proof of the second inequality in (3).
this section cite: []

Section: Probability Tail bounds Definition 3. ((σ, b)-subexponential, Wainwright [2019], Definition 2.7) A random variable X with mean
µ = E [X] is (σ, b)-subexponential, if for ∀λ ∈ -1 b , 1 b , E [exp(λ(X -µ))] ≤ exp σ 2 λ 2 2 .
Lemma 5.
((σ, b)-subexponential tail bound, another form of Wainwright [2019], Proposition 2.9) Suppose X is a (σ, b)-subexponential random variable with mean E [X] = µ, then with probability at least 1 -δ, X ≤ µ + 2σ 2 ln 1 δ + 2b ln 1 δ , also, with probability at least 1 -δ, X ≥ µ -2σ 2 ln 1 δ -2b ln 1 δ . Lemma 6. (A Bernstein-type bound for i.i.d. random variables, another form of Wainwright [2019], Equation (2.18)) Suppose {X} N i=1 is sequence of i.i.d. (σ, b)-subexponential random variables, then, with probability at least 1 -δ, N i=1
X i ≤ N i=1 E [X i ] + σ 2N ln 1 δ + 2b ln 1 δ ,
and, with probability at least 1 -δ,
N i=1 X i ≥ N i=1 E [X i ] -σ 2N ln 1 δ -2b ln 1 δ .
Lemma 7. (A Bernstein-type bound for a martingale difference sequence, another form of Wainwright [2019], Theorem 2.19) Suppose {X} N i=1 is a sequence of conditionally (σ, b)-subexponential random variables adapted from filtration
{F i } N i=1 , i.e., E [exp(λ(X i -E [X i | F i-1 ])) | F i-1 ] ≤ exp σ 2 λ 2 2 , ∀λ ∈ - 1 b , 1 b .
Then, with probability at least 1 -δ,
N i=1 X i ≤ N i=1 E [X i | F i-1 ] + σ 2N ln 1 δ + 2b ln 1 δ ,
and, with probability at least 1 -δ,
N i=1 X i ≥ N i=1 E [X i | F i-1 ] -σ 2N ln 1 δ -2b ln 1 δ .
Lemma 8. (Azuma-Hoeffding's Inequality, another form of Wainwright [2019], Corollary 2.20) Suppose {X} N i=1 is a sequence adapted from filtration {F i } N i=1 such that X i ∈ [a, b], Then, with probability at least 1 -δ,
N i=1 X i ≤ N i=1 E [X i | F i-1 ] + (b -a) 1 2 N ln 1 δ ,
and, with probability at least 1 -δ,
N i=1 X t ≥ N i=1 E [X i | F i-1 ] -(b -a) 1 2 N ln 1 δ .
We show in the following lemma how to determine the parameters (σ, b) prescribed in Definition 3 by a given probability tail bound. Lemma 9. Suppose a random variable satisfies P (|X| ≥ a) ≤ C exp(-a ν ) for given C, ν > 0, then X is (6ν
√ 1 + 2C, 6ν)-subexponential. Also, if Y is a random variable that satisfy |Y | ≤ M , then, XY is (6M ν √ 1 + 2C, 6M ν)-subexponential.
Proof. First, consider |X|'s moment generating function E e λ|X| , for ∀λ > 0, we have
E e λ|X| = +∞ 0 P e λ|X| ≥ u du ≤ 1 + +∞ 1 P |X| ≥ ln u λ du ≤ 1 + C +∞ 1 u -1 λν du.
From the above inequality, we get that
E e λ|X| ≤ 1 + Cλν 1-λν < ∞ if 0 < λ < 1 ν . Set λ = 2 3ν , as E e 2 3ν |X| = ∞ i=0 E[|X| i ] ( 3 2 ν) i i! , we have: E |X| i ( 3 2 ν) i i! ≤ E e 2 3ν |X| ≤ 1 + 2C.(4)
Now we introduce a new random variable X that is an independent copy of X, then we can bound
E [exp(λ(X -E [X]))] by Jensen's inequality, E [exp(λ(X -E [X]))] ≤ E [exp(λ(X -X ))]. Therefore, we only need to bound E [exp(λ(X -X ))]. For ∀λ ∈ -1 6ν , 1 6ν , E [exp(λ(X -X ))] = ∞ i=0 E (X -X ) i λ i i! = ∞ i=0 E (X -X ) 2i λ 2i (2i)! ≤ 1 + ∞ i=1 E |X| 2i 2 2i λ 2i (2i)! ≤ 1 + (1 + 2C) ∞ i=1 3 2 ν 2i 2 2i λ 2i = 1 + (1 + 2C) ∞ i=1 (3νλ) 2i ≤ 1 + 2(1 + 2C)(3νλ) 2 ≤ exp(2(1 + 2C)(3νλ) 2 ) = exp (6ν √ 1 + 2C) 2 λ 2 2 ,
where the first equality holds by Taylor expansion, the second equality holds since E (X -X ) i = 0 for all i's that are odd. The first inequality holds by the fact that |x -
x | i ≤ 2 i-1 (|x| i + |x | i
) for all i ≥ 1, and that X and X have the same distribution. The second inequality holds by (4). The third inequality holds since
∞ i=1 (3νλ) 2i = (3νλ) 2 1-(3νλ) 2 ≤ 2 × (3νλ) 2 when λ ∈ -1 6ν , 1 6ν .
The last inequality holds by the fact that 1 + x ≤ e x for all x ∈ R.
Thus, by definition of (σ, b)-subexponential, we conclude that X is (6ν √ 1 + 2C, 6ν)subexponential. Now we prove the subexponential property of XY . Since |Y | ≤ M and P (|X| ≥ a) ≤ C exp(-a ν ),
P (|XY | ≥ a) ≤ P |X| ≥ a M ≤ C exp - a M ν .
Replacing ν by M ν, we conclude that XY is (6M ν √ 1 + 2C, 6M ν)-subexponential.
Lemma 9 directly accommodates our regularity assumption and leads to the following corollary.
Corollary 2. (Subexponential property of regular distributions) Suppose x is a random variable that satisfies Assumption 2, then, for ∀ w ∈ B d , w, x is ( 16 Q , 6 Q )-subexponential.
Proof. By Assumption 2 Condition 3, P[| w, x | ≥ t] ≤ exp(1 -Qt) for ∀ t > 0. Then by Lemma 9, set C = e, ν = 1 Q , then, 6ν
√ 1 + 2C = 6 √ 1+2e Q < 16 Q , 6ν = 6 Q , we conclude that x is 16 Q , 6 Q -subexponential.
The relationship between E [ w ⋆ , -g k,i ] and θ (w k,i , w ⋆ ) Recall that in the non-strategic setting, we shall adjust the coefficient by solving a sequence of adaptively constructed online regret minimization problems min w∈W k
T k i=1 w, g k,i - T k i=1 w * , g k,i
with g k,i = [-ηx k,i I (y k,i = 1)+ (1 -η)x k,i I (y k,i = -1)]I(x k,i ∈ D k,i ) via mirror descent over k = 0 . . . , K batches, using local data within increasingly narrow bands D k,i = {x : 0 < w k,i , x ≤ b k }. The key ingredient underlying this guarantee is that the gradients g k,i are well constructed so that | w k,i , g k,i | is small and meanwhile E [ w ⋆ , -g k,i ] upper bounds θ (w ⋆ , w k,i ). Here, we show the relationship between E [ w ⋆ , -g k,i ] and θ (w k,i , w ⋆ ) for k = 0, 1, • • • , K, which is critical in the guarantees of Algorithm 3 and Algorithm 4. Fix batch k and iteration i, to connect E [ w ⋆ , -g k,i ] and θ (w k,i , w ⋆ ) we introduce a new variable f k,i (w k,i ) in (5). Later, we will show how E [ w ⋆ , -g k,i ] upper bounds f k,i (w k,i ) and how f k,i (w k,i ) approximates θ (w k,i , w ⋆ ).
f k,i (w k,i ) := E [| w ⋆ , x | I ( w ⋆ , x < 0) | x ∈ D k,i ] .(5)
Now, we show that E [ w ⋆ , -g k,i ] can upper bound f k,i (w k,i ) by the following lemma.
Lemma 10. Given a unit vector w k,i ∈ S d and an agent with true feature-label pair (x k,i , y k,i ).
For g k,i = [-ηx k,i I (y k,i = 1) + (1 -η)x k,i I (y k,i = -1)]I(x k,i ∈ D k,i ) and f k,i (w k,i ) defined in (5). The following holds.
E [ w ⋆ , -g k,i ] ≥ (1 -2η)f k,i (w k,i )P (x ∈ D k,i ) .
Proof. First, for convenience, we rewrite g k,i as the following.
g k,i = [-ηx k,i I (y k,i = 1) + (1 -η)x k,i I (y k,i = -1)]I(x k,i ∈ D k,i ) = - 1 2 y k,i + 1 2 -η x k,i I(x k,i ∈ D k,i ).
Then, we have
E [ w ⋆ , -g k,i ] = E w ⋆ , 1 2 y k,i - 1 2 -η x k,i I (x k,i ∈ D k,i ) = E w ⋆ , 1 2 y k,i - 1 2 -η x k,i x k,i ∈ D k,i P (x k,i ∈ D k,i ) + 0 = 1 2 E [ w ⋆ , x k,i E [y k,i | x k,i ] | x k,i ∈ D k,i ] P (x ∈ D k,i ) - 1 2 -η E [ w ⋆ , x k,i | x k,i ∈ D k,i ] P (x ∈ D k,i ) ≥ 1 2 -η E [| w ⋆ , x k,i | | x k,i ∈ D k,i ] P (x ∈ D k,i ) - 1 2 -η E [ w ⋆ , x k,i | x k,i ∈ D k,i ] P (x ∈ D k,i ) = (1 -2η) E [| w ⋆ , x k,i | I ( w ⋆ , x k,i < 0) | x k,i ∈ D k,i ] P (x ∈ D k,i ) = (1 -2η) f k,i (w k,i )P (x ∈ D k,i ) ,
where the second and third equality hold by the law of iterated expectations, and the inequality holds as the following.
E [y k,i | x k,i ] = (1 -η(x k,i ))sgn( w ⋆ , x k,i ) -η(x k,i )sgn( w ⋆ , x k,i ) = (1 -2η(x k,i ))sgn( w ⋆ , x k,i ) ≥ (1 -2η)sgn( w ⋆ , x k,i ).
Next, in the following lemma, we show that f k,i (w k,i ) measures the closeness of w ⋆ and w k,i .
Lemma 11. For fixed batch k and iteration i, if θ (w ⋆ , w k,i ) = ϕ, then the following holds.
1. When 0 < b k ≤ R 4 and ϕ ∈ 4b k R , π 2 , we have
f k,i (w k,i ) ≥ L 2 32U 1 R 2 ϕ. 2. When 0 < b k ≤ R 4 and ϕ ∈ π 2 , π -4b k R , we have f k,i (w k,i ) ≥ L 2 32U 1 R 2 (π -ϕ).
Proof. We prove the two cases respectively. For Case 1, define the region 6 as an illustration. We have the following
G 1 := x | 0 ≤ w k,i , x ≤ b k , -1 2 R sin ϕ ≤ w ⋆ , x ≤ -1 4 R sin ϕ , see Figure
E [| w ⋆ , x | I ( w ⋆ , x < 0) I (0 ≤ w k,i , x ≤ b k )] ≥ E [| w ⋆ , x | I (x ∈ G 1 )] ≥ 1 4 R sin ϕE [I (x ∈ G 1 )] ≥ 1 8 RϕP (x ∈ G 1 ) ≥ 1 32 L 2 R 2 ϕb k ,
where the first inequality holds since
G 1 ⊆ {x | 0 ≤ w k,i , x ≤ b, w ⋆ , x < 0}.
The third inequality holds by the fact that sin ϕ ≥ ϕ 2 for 0 ≤ ϕ ≤ π 2 . And the last inequality holds by the claim that P (x ∈ G 1 ) ≥ 1 4 L 2 Rb k , which we will show later.
x1 x2 w w * O A B C D ϕ ϕ Figure 6: Illustration of region G 1 (the red region) in Case 1. Which satisfies G 1 = x | 0 ≤ w, x ≤ b k , -1 2 R sin ϕ ≤ w ⋆ , x ≤ -1 4 R sin ϕ .
Hence, we can establish the lower bound of f k,i (w k,i ) by:
f k,i (w k,i ) = E [| w ⋆ , x | I ( w ⋆ , x < 0) | 0 ≤ w k,i , x ≤ b k ] = E [| w ⋆ , x | I ( w ⋆ , x < 0) I (0 ≤ w k,i , x ≤ b k )] P (0 ≤ w k,i , x ≤ b k ) ≥ 1 32 L 2 R 2 ϕb k P (0 ≤ w k,i , x ≤ b k ) ≥ L 2 32U 1 R 2 ϕ,
where the last inequality holds by Lemma 4, property (a).
Now we show the claim that P (x ∈ G 1 ) ≥ 1 4 L 2 Rb k . For a given vector x, we first project x down to the subspace V 2 ⊆ R 2 spanned by w ⋆ and w k,i and denote the projected value z := ( w ⋆ , x , w, x ).
Without loss of generality, let w = (0, 1) and w ⋆ = (sin ϕ, cos ϕ). As illustrated in Figure 6, the parallelogram ABCD denotes the region G 1 , where
A = 1 4 R, 0 , B = 1 2 R, 0 , C = 1 2 R + b k tan ϕ , b k , D = 1 4 R + b k tan ϕ , b k .
Since C is the farthest point to the origin with respect to the Euclidean Norm, and
1 2 R + b k tan ϕ , b k 2 ≤ 1 2 R + b k tan ϕ , b k 1 = 1 2 R + b k tan ϕ + b k ≤ 1 2 R + b k ϕ + b k ≤ R, then for all z ∈ z = (z 1 , z 2 ) | -1 2 R sin ϕ ≤ z 1 ≤ -1 4 sin ϕ, 0 ≤ z 2 ≤ b k , we have z 2 ≤ R. Also, the area of parallelogram ABCD is b k • 1 4 R = 1 4 Rb k . In addition, by Assumption 2, condition 1, the density ϕ V2 (z) of projected value z satisfies ϕ V2 (z) ≥ L 2 for all z ∈ V 2 ∩ B 2 (R). Hence, we can lower bound P (x ∈ G 1 ) by P (x ∈ G 1 ) ≥ L 2 • 1 4 Rb k = 1 4 L 2 Rb k .
The proof of Case 2 is similar to that of Case 1.
Define the region
G 2 = x | 0 ≤ w k,i , x ≤ b k , -1 2 R sin(π -ϕ) ≤ w ⋆ , x ≤ -1 4 R sin(π -ϕ) (see Figure 7). We re- place ϕ in the proof of Case 1 by π -ϕ, by choosing A = -1 4 R, 0 , B = -1 2 R, 0 , C = -1 2 R + b k tan(π-ϕ) , b k , D = -1 4 R + b k
tan(π-ϕ) , b k and then we complete the proof.
Lemma 11 directly leads to the following corollary.
Corollary 3. If θ(w ⋆ , w k,i ) ≤ π 2 , and f k,i (w k,i ) ≤ L2 160U1 R 2 θ k , then θ(w ⋆ , w k,i ) ≤ θ k 5 .
Proof. We conduct a case analysis.
1. If θ (w ⋆ , w k,i ) < 4b k R , then by our setting of b k in Algorithm 3 and Algorithm 4, θ (w ⋆ , w k,i ) < 4b k R ≤ θ k 5 . x1 x2 w w * O A B C D ϕ ϕ Figure 7: Illustration of region G 2 (the red region) in Case 2. Which satisfies G 2 = x | 0 ≤ w k,i , x ≤ b k , -1 2 R sin(π -ϕ) ≤ w ⋆ , x ≤ -1 4 R sin(π -ϕ) . 2. If 4b k R ≤ θ (w ⋆ , w k,i ) ≤ π 2 , then by Lemma 11, we have f k,i (w k,i ) ≥ L2 32U1 R 2 θ (w ⋆ , w k,i ), combing it with the condition that f k,i (w k,i ) ≤ L2 160U1 R 2 θ k , we have θ(w ⋆ , w k,i ) ≤ θ k 5 .
Other lemmas We also outline some other lemmas that are used in the subsequent proof. Lemma 12. (Triangular inequality of angles) Suppose vectors x, y, z ∈ R d satisfy 0 ≤ θ (x, y) ≤ π 2 and 0 ≤ θ (x, z) ≤ π 2 , then,
| θ (x, y) -θ (x, z) | ≤ θ (y, z) ≤ θ (x, y) + θ (x, z) .
Proof. Without loss of generality, assume that x 2 = y 2 = z 2 = 1. First, we decompose vectors y and z into components along the direction of x and perpendicular to x, respectively, as:
y = cos(θ (x, y))x + sin(θ (x, y))y ⊥x , z = cos(θ (x, z))x + sin(θ (x, z))z ⊥x ,
where y ⊥x and z ⊥x are unit vectors that are perpendicular to x. Hence, we have cos(θ (y, z)) = y, z = cos(θ (x, y)) cos(θ (x, z)) + sin(θ (x, y)) sin(θ (x, z)) y ⊥ , z ⊥ , where the second equality holds since x, y ⊥ = x, z ⊥ = 0. Since y ⊥ and z ⊥ are both unit vectors, by Cauchy-Schwarz inequality, -1 ≤ y ⊥ , z ⊥ ≤ 1. Also, since 0 ≤ θ (x, y) ≤ π 2 and 0 ≤ θ (x, z) ≤ π 2 , we have sin(θ (x, y)) sin(θ (x, z)) ≥ 0. Putting all together, we have:
cos(θ (y, z)) ≤ cos(θ (x, y)) cos(θ (x, z)) + sin(θ (x, y)) sin(θ (x, z)) = cos(θ (x, y) -θ (x, z)),(6)
and
cos(θ (y, z)) ≥ cos(θ (x, y)) cos(θ (x, z)) -sin(θ (x, y)) sin(θ (x, z)) = cos(θ (x, y) + θ (x, z)).(7)
Since cos(x) is decreasing in x ∈ [0, π], and cos(x) = cos(-x) for all x ∈ R, we get θ (y, z) ≥ | θ (x, y) -θ (x, z) | from ( 6), and θ (y, z) ≤ θ (x, y) + θ (x, z) from ( 7).
Lemma 13. Given a random vector x ∼ D x that satisfy Assumption 2, then with probability at least 1 -δ,
x ∞ ≤ 1 Q 1 + ln d δ .
Proof. We bound x ∞ element-wisely. Given x ∼ D x and j ∈ [d], let x j be the j-th coordinate of x. Let e [j] ∈ R d denote the unit vector whose j'th coordinate is 1 while other coordinate is 0, then, by Assumption 2, condition 3, for ∀ a > 0
P (|x j | ≥ a) = P e [j] , x ≥ a ≤ exp(1 -Qa).
Taking union bound over all coordinates, we have
P ( x ∞ ≥ a) ≤ d exp(1 -Qa).
Taking a = 1 Q 1 + ln d δ in the above inequality and hence we complete the proof.
this section cite: []

Section: A.4 Proofs for Section 2
In this subsection, we outline some important results to describe the relationships between classifier h(•) for unmanipulated features x and classifier h for reported features r = r ⋆ (x, h), which is critical for the subsequent algorithm design.
this section cite: []

Section: Proof of Proposition 1
Proposition 1. For any (w, m) ∈ S d ×R, the output of h(r) = sgn( w, r +m-γ) for r = r ⋆ (x, h) is identical to the output of h(x) = sgn( w, x + m) for any x ∈ R d .
Proof. For fixed w ∈ S d and m ∈ R, we categorize the agent population into three classes according to their true features x: {x | w, x + m < 0}, {x | 0 ≤ w, x + m < γ} and {x | w, x + m ≥ γ}. Then, we discuss their classification output by h(x) and h(r) with respect to r = r ⋆ (x, h), respectively.
1. When w, x + m < 0, h(x) = -1. At the same time, w, x + m -γ < -γ, by Lemma 1, agent in this region will report his feature truthfully, i.e., r = x. Thus, h(r
) = sgn( w, r + m -γ) = sgn( w, x + m -γ) = -1. Hence, we have h(x) = h(r) = -1 for ∀x ∈ {x | w, x + m < 0} .
2. When 0 ≤ w, x + m < γ, h(x) = 1. At the same time, -γ ≤ w, x + m -γ < 0, by Lemma 1, agent in this region will manipulate his feature as r = x + (γ -m -w, x )w.
Thus, h(r) = sgn( w, r + m -γ) = sgn( w, x + (γ -m -w, x )w + m -γ) = 1.
Hence, we have h(x) = h(r) = 1 for ∀x ∈ {x | 0 ≤ w, x + m < γ} .
3. When w, x + m ≥ γ, h(x) = 1. At the same time, w, x + m -γ ≥ 0, by Lemma 1, agent in this region will report his feature truthfully, i.e., r = x. Thus, h(r) = sgn( w, r + m -γ) = sgn( w, x + m -γ) = 1. Hence, we have h(x) = h(r) = 1 for ∀x ∈ {x | w, x + m ≥ γ} .
Inferring agents' true features from their reported features Proposition 1 directly leads to the following corollary, enabling us to infer an agent's true features x given a classification rule h(•) and his corresponding reported features r.
Corollary 4. For given announced classifier h(•) = sgn ( w, • + m) and agent response r, then, his true features x satisfy the following.
1. if w, r + m = 0, then x = r; 2. if w, r + m = 0, then -γ ≤ w, x + m ≤ 0.
this section cite: []

Section: A.5 Proofs for Section 3
In this subsection, we show the theoretical guarantees of Algorithm 2, Algorithm 3 and Algorithm 4, respectively.
this section cite: []

Section: Theoretical Guarantees of Algorithm 2
A key observation in the non-strategic and noiseless classification scenario is that y w ⋆ , x > 0 holds for all (x, y). Consequently, xy always forms an acute angle with the optimal normal vector w ⋆ . Considering agents' strategic responses and the bandit feedback setting, we introduce Lemma 14 to show that, under our construction in Algorithm 2, w0 is an unbiased estimator of E[xy].
Lemma 14. In Algorithm 2, we have
E[ w0 ] = E[xy].
Proof. In Algorithm 2, at iteration i, the principal declares classifier h(1) init,i (r) = sgn( w init,i , r ) and h(2) init,i (r) = sgn( -w init,i , r ), and receives response r
init,i and r
init,i respectively. By Corollary 4, we get that r (1)
init,i I w init,i , r (1) init,i > 0 = x (1
)
init,i I w init,i , x(1)
init,i > 0 and r (2)
init,i I -w init,i , r (2) init,i > 0 = x (2
)
init,i I -w init,i , x(2)
init,i > 0 . Also, x(1)
init,i , y (1) init,i and x (2) init,i , y (2) init,i are i.i.d. drawn from D, hence, E r (1) init,i y (1) init,i I w init,i , r
init,i > 0 + r ((1)
)2
init,i y (2
)
init,i I -w init,i , r(2)
init,i > 0 = E x (1
)
init,i y (1
)
init,i I w init,i , x(1)
init,i > 0 + x (2) init,i y (2
)
init,i I -w init,i , x(2)
init,i > 0 = E [xyI ( w init,i , x > 0)] + E [xyI ( w init,i , x < 0)] = E [xy] .
Thus, we have
E [ w0 ] = 1 T init Tinit i=1 E r (1
)
init,i y (1) init,i I w init,i , r(1)
init,i > 0 + r (2
)
init,i y (2
)
init,i I -w init,i , r(2)
init,i > 0 = E [xy] .
Now we show that w0 constructed by Algorithm 2 has a positive inner product with the optimal coefficient w * with high probability.
Proposition 2. For some constants c 0 , c 1 > 0, when Algorithm 2 runs for T init = c 0 ln T /(1 -2η) 2 iterations, its output w0 satisfies w ⋆ , w0 > c 1 (1 -2η) > 0 and θ (w ⋆ , w0 ) ≤ π 2 with probability at least 1 -2/T 2 . Proof. First, considering the non-strategic classification problem, we establish a lower bound of E[ w ⋆ , x y] as the following.
E[ w ⋆ , x y] = E [E[y| x] w ⋆ , x ] = E [[(1 -η(x))sgn( w ⋆ , x ) -η(x)sgn( w ⋆ , x )] w ⋆ , x ] = E [(1 -2η(x))| w ⋆ , x |] ≥ (1 -2η)E [| w ⋆ , x |] ≥ (1 -2η)L 1 R 2 ,(8)
where the first equality holds by the law of iterated expectations. The second equality holds by the definition of Massart Noise. The third equality holds since sgn(
w * , x w * , x ) = | w * , x |. The first inequality holds because η(x) ≤ η, ∀x ∈ R d . Now we prove E [| w ⋆ , x |] ≥ L 1 R 2 for the last inequality as the following: let x V1 := w ⋆ , x denote a 1-dimensional projection of x. Then, by Assumption 2, condition 1, ϕ V1 (x V1 ) ≥ L 1 for all -R ≤ x V1 ≤ R. Hence, we have E[| w ⋆ , x |] ≥ 2 R 0 x V1 ϕ V1 (x V1 ) dx V1 ≥ L 1 × 2 R 0 xdx = L 1 R 2 .
Second, considering agents' strategic response, we find an unbiased estimator of E [ w ⋆ , x y]
through samples. By Lemma 14, w0 =
1 Tinit Tinit i=1 r (1) init,i y (1) init,i I w init,i , r (1) init,i > 0 + r (2) init,i y (2) init,i I -w init,i , r (2) init,i > 0 is an unbiased estimator of E [xy]. Therefore, E [ w ⋆ , w0 ] = E [ w ⋆ , x y] ≥ (1 -2η)L 1 R 2 .
Finally, using the results of concentration inequalities, we establish the high probability bound of w ⋆ , w0 .
By Corollary 1,
w ⋆ , r(1)
init,i I w init,i , r
init,i > 0 = w ⋆ , x(1)
init,i I w init,i , x
init,i > 0 and w ⋆ , r
init,i I -w init,i , r
init,i > 0 = w ⋆ , x(2)
init,i I -w init,i , x(2)
init,i > 0 . Then, by Assumption 2 condition 3 and Lemma 9, we get that both w ⋆ , r
init,i y (1) init,i I w init,i , r (1) init,i > 0 and w ⋆ , r (2) init,i y (2) init,i I -w init,i , r(1)
init,i > 0 are 16 Q , 6 Q -subexponential. Thus, by Lemma 6, we have that with probability at least 1 -1 T 2 ,
1 T init Tinit i=1 w ⋆ , r(1)
init,i y (1) init,i I w init,i , r(1)
init,i > 0 ≥ 1 T init Tinit i=1 E [ w ⋆ , x yI( w init,i , x > 0)] - 32 Q ln T T init - 24 Q ln T T init ,(9)
and with probability 1
-1 T 2 , 1 T init Tinit i=1 w ⋆ , r(2)
init,i y
init,i I -w init,i , r(2)
init,i > 0 ≥ 1 T init Tinit i=1 E [ w ⋆ , x yI( w init,i , x < 0)] - 32 Q ln T T init - 24 Q ln T T init .(2)
Taking the union bound for ( 9) and ( 10), then, with probability at least 1 -2 T 2 ,
w ⋆ , w0 = 1 T init Tinit i=1 w ⋆ , r(1)
init,i y (1) init,i I w init,i , r(1)
init,i > 0 + 1 T init Tinit i=1 w ⋆ , r(2)
init,i y (2) init,i I -w init,i , r(2)
init,i > 0 ≥ 1 T init Tinit i=1 E [ w ⋆ , x yI( w init,i , x > 0)] - 32 Q ln T T init - 24 Q ln T T init + 1 T init Tinit i=1 E [ w ⋆ , x yI( w init,i , x < 0)] - 32 Q ln T T init - 24 Q ln T T init = E [ w ⋆ , x y] - 64 Q ln T T init - 48 Q ln T T init ≥ (1 -2η)L 1 R 2 - 64 Q ln T T init - 48 Q ln T T init ≥ 1 2 (1 -2η)L 1 R 2 .
The first inequality holds by ( 9) and (10). The second inequality holds by Lemma 14. The last inequality holds by setting
T init = 20736 ln T (1-2η) 2 L 2 1 Q 2 R 4 , then 64 Q ln T Tinit ≤ 4 9 (1 -2η)L 1 R 2 and 48 Q ln T Tinit 1 18 (1 -2η)L 1 R 2
for large enough T . Thus, we complete the proof of Proposition 2.
this section cite: []

Section: Theoretical Guarantees of Algorithm 3
Recall that in the i'th iteration of Algorithm 3, we declare classifier h0,i (r) = sgn( w 0,i , r ) and construct the gradient as g0,i = [-ηr 0,i I (y
0,i = 1) + (1 -η)r 0,i I (y 0,i = -1)]I(0 < w 0,i , r 0,i ≤ b 0 ).
In Lemma 15, we establish the high probability upperbound of
T0 i=1 E[ w ⋆ , -g 0,i )| F 0,i-1 ] by T0 i=1 w ⋆ , -g 0,i .
Lemma 15. In Algorithm 3, with probability at least 1 -1/T 2 , we have
T0 i=1 E[ w ⋆ , -g 0,i )| F 0,i-1 ] ≤ T0 i=1 w ⋆ , -g 0,i + 24 δ 1 + 2ρb 0 T 0 ln T + 48 δ ln T, where ρ = max U 1 exp(δ), U2 exp(δ) δ . Proof. Since w ⋆ , -g 0,i = w ⋆ , x 0,i I (0 < w 0,i , x 0,i ≤ b 0 ) 1 2 y 0,i -1 2 -η , we first estab- lish the probability tail bound of w ⋆ , x 0,i I(0 < w 0,i , x 0,i ≤ b 0 ).
We partition x 0,i into two orthonormal vectors, for notational convenience, we omit the subindex 0, i of x 0.i and let x w denote the ingredient of x 0,i that is parallel to w 0,i , i.e., x w = w 0,i , x 0,i w 0,i and x ⊥w denote the ingredient of x 0,i that is vertical to w 0,i , i.e., x ⊥w = x 0,i -x w . Then,
w ⋆ , x 0,i I(0 ≤ w 0,i , x 0,i ≤ b 0 ) = w ⋆ , x w I(0 ≤ w 0,i , x 0,i ≤ b 0 ) (a) + w ⋆ , x ⊥w I(0 ≤ w 0,i , x 0,i ≤ b 0 ) (b) . (11
)
Thus, we have to bound part (a) and (b) in (11), respectively. First, we bound part (a) as
w ⋆ , x w I(0 ≤ w 0,i , x 0,i ≤ b 0 ) = w ⋆ , w 0.i , x 0,i w 0,i I(0 ≤ w 0.i , x 0,i ≤ b 0 ) ≤ b 0 w ⋆ , w 0,i ≤ b 0 , (12
)
where the first equality holds since x w = w 0,i , x 0,i w 0,i . The last inequality holds by the Cauchy-Schwarz Inequality. Next, we bound part (b) in (11). For w 0,i -w ⋆ 2 ≤ r 0 and a > b 0 , we have
P (| w ⋆ , x ⊥w I(0 ≤ w 0,i , x 0,i ≤ b 0 )| ≥ a -b 0 ) = P (| w ⋆ , x ⊥w | ≥ a -b 0 , 0 ≤ w 0,i , x 0,i ≤ b 0 ) = P (| w ⋆ ⊥w , x 0,i | ≥ a -b 0 , 0 ≤ w 0,i , x 0,i ≤ b 0 ) ,
where we prove the second equality as follows
w ⋆ , x ⊥w = w ⋆ ⊥w , x ⊥w + w ⋆ w , x ⊥w , w ⋆ ⊥w , x 0,i = w ⋆ ⊥w , x ⊥w + w ⋆ ⊥w , x w . since w ⋆ w , x ⊥w = w ⋆ ⊥w , x w = 0, we have w ⋆ , x ⊥w = w ⋆ ⊥w , x 0,i . Denote X := w ⋆ ⊥w w ⋆ ⊥w 2 , x 0,i and Y := w 0,i , x 0,i . Then, (X, Y ) forms a projection of x 0,i onto a 2-dimensional subspace V 2 spanned by w * ⊥w w * ⊥w 2 and w 0,i . Let ϕ V2 denote the density of (X, Y ). By Assumption 2 condition 2, we have ϕ V2 (X, Y ) ≤ U 2 exp(-δ (X, Y ) 2 ) = U 2 exp(-δ √ X 2 + Y 2
). Hence, we can bound the above probability by
P (| w ⋆ , x ⊥w I(0 ≤ w 0,i , x 0,i ≤ b 0 ) | ≥ a -b 0 ) = P w ⋆ ⊥w w ⋆ ⊥w 2 , x 0,i ≥ a -b 0 w ⋆ ⊥w 2 , 0 ≤ w 0,i , x 0,i ≤ b 0 = +∞ a-b 0 ∥w ⋆ ⊥w ∥ 2 b0 0 ϕ V2 (X, Y )dXdY ≤ U 2 +∞ a-b 0 ∥w ⋆ ⊥w ∥ 2 b0 0 exp -δ X 2 + Y 2 dXdY ≤ U 2 b 0 +∞ a-b 0 ∥w ⋆ ⊥w ∥ 2 exp(-δX)dXdY = U 2 δ b 0 exp -δ a -b 0 w ⋆ ⊥w 2 ≤ U 2 exp (δ) δ b 0 exp -δ a r 0 . (13
)
The last inequality holds by the fact that w ⋆ -w 0,i 2 ≤ r 0 , which implies w ⋆ ⊥w 2 ≤ r 0 , and that b 0 < r 0 = 2, which implies exp δ b0 r0 < exp (δ). Combing ( 12) and ( 13), for a > b 0 ,
P ( w ⋆ , x 0,i I(0 ≤ w 0,i , x 0,i ≤ b 0 ) > a) ≤ P (| w ⋆ , x ⊥w I(0 ≤ w 0,i , x 0,i ≤ b 0 )| ≥ a -b 0 ) ≤ U 2 exp(δ) δ b 0 exp -δ a r 0 . (14) For 0 < a ≤ b 0 , P ( w ⋆ , x 0,i I(0 ≤ w 0,i , x 0,i ≤ b 0 ) > a) ≤ P (| w ⋆ , x 0,i I(0 ≤ w 0,i , x 0,i ≤ b 0 )| > 0) ≤ P (0 ≤ w 0,i , x 0,i ≤ b 0 ) ≤ U 1 b 0 ≤ U 1 b 0 exp δ b 0 r 0 exp -δ a r 0 ≤ U 1 exp(δ)b 0 exp -δ a r 0 . (15
)
The third inequality holds by Lemma 4 property (a). The fourth inequality holds for 0 < a < b 0 .
The last inequality holds since by our construction, b 0 < r 0 = 2.
Let ρ = max U 1 exp(δ), U2 exp(δ) δ
, by ( 14) and ( 15), we conclude that for ∀ a > 0,
P (I(0 ≤ w 0,i , x ≤ b 0 ) w ⋆ , x > a) ≤ ρb 0 exp -δ a r 0 . (16
)
Thus, P (| w ⋆ , -g 0,i | ≥ a) = P w ⋆ , x 0,i I(0 ≤ w 0,i , x 0,i ≤ b 0 ) 1 2 y 0,i - 1 2 -η ≥ a ≤ P (| w ⋆ , x 0,i I(0 ≤ w 0,i , x 0,i ≤ b 0 ) | ≥ a) ≤ρb 0 exp -δ a 2 ,
where the first inequality holds since 1 2 y 0,i -( 1 2 -η) ≤ 1. In the last inequality, since w 0,iw ⋆ 2 ≤ w 0,i 2 + w ⋆ 2 = 2, we take r 0 = 2 in (16) and get the upper bound.
By Lemma 9, w ⋆ , -g 0,i is 12 δ √ 1 + 2ρb 0 , 12 δ -subexponential.
By Lemma 7, we can get that with probability at least (1 -1/T 2 ),
T0 i=1 E[ w ⋆ , -g 0,i )| F 0,i-1 ] ≤ T0 i=1 w ⋆ , -g 0,i + 24 δ 1 + 2ρb 0 T 0 ln T + 48 δ ln T.
The following lemma establishes a high-probability upper bound for the average of
f 0,i (w 0,i ) over T 0 iterations. Lemma 16. In Algorithm 3, if w ⋆ , w0 ≥ 1 2 (1 -2η)L 1 R 2 , then there exist some con- stants c 2 , c 3 , c 4 > 0, when setting bandwidth b 0 = c 2 (1 -2η) 2 , iteration number T 0 = c 3 1 (1-2η) 8 d ln d(ln T ) 2 , step size α 0 = c 4 √ d ln d √ T0 ln T , then with probability at least 1 -3/T 2 , we have 1 T 0 T0 i=1 f 0,i (w 0,i ) ≤ min 1, L 1 R 2 π(1 -2η)L 2 R 2 2880U 1 . Proof. Recall that in the non-strategic setting, g k,i = [-ηx 0,i I (y 0,i = 1) + (1 - η)x 0,i I (y 0,i = -1)]I (x 0,i ∈ D 0,i ), where D 0,i = {x | 0 ≤ w 0,i , x ≤ b 0 } is just the localization region.
Since the announced classifier is h0,i (r) = sgn( w 0,i , r ), by the construction of g0,i in Algorithm 3 and Corollary 4, we have
g0,i = [-ηr 0,i I (y 0,i = 1) + (1 -η)r 0,i I (y 0,i = -1)]I (0 < w 0,i , r 0,i ≤ b 0 ) = [-ηx 0,i I (y 0,i = 1) + (1 -η)x 0,i I (y 0,i = -1)]I (x 0,i ∈ D 0,i ) = g 0,i . Then, by Lemma 10, we have E [ w ⋆ , -g 0,i ] = E [ w ⋆ , -g 0,i ] ≥ (1 -2η)f 0,i (w 0,i )P (x ∈ D 0,i ) .(17)
We proceed to establish the high probability bound of
T0 i=1 E [ w ⋆ , -g 0,i ].
By Lemma 15, with probability at least 1 -1/T 2 , we have
T0 i=1 E[ w ⋆ , -g 0,i )| F 0,i-1 ] ≤ T0 i=1 w ⋆ , -g 0,i + 24 δ 1 + 2ρb 0 T 0 ln T + 48 δ ln T,(18)
where
ρ = max U 1 exp(δ), U2 exp(δ) δ .
Next, we move on to upper bound Shalev-Shwartz [2007]]. From the analysis of online mirror descent [see Orabona [2023], Lemma 6.9], with step size α 0 , we have
T0 i=1 w ⋆ , -g 0,i through a nonstandard regret analysis of online mirror decent. Let B(v 1 , v 2 ) := 1 2(p-1) v 1 -v 2 2 p denote the Bregman divergence w.r.t. 1 2(p-1) • 2 p , where p = ln (8d) ln (8d)-1 . In each iteration i, the regularizer, B(•, w 0,i-1 ) is 1-strongly convex with respect to • p [see
α 0 ĝ0,i , w 0,i -w ⋆ ≤ B(w ⋆ , w 0,i ) -B(w ⋆ , w 0,i+1 ) + α 2 0 2 g0,i 2 q , where q = ln(8d) > 2. Summing the above equality over i ∈ [T 0 ], we get T0 i=1 α 0 g0,i , w 0,i -w ⋆ ≤ B(w ⋆ , w 0,1 ) -B(w ⋆ , w 0,T0+1 ) + α 2 0 2 T0 i=1 g0,i 2 q .
Dividing both sides by α 0 , and moving T0 i=1 w 0,i , g0,i to RHS, we get
T0 i=1 w ⋆ , -g 0,i ≤ 1 α 0 [B(w ⋆ , w 0,1 ) -B(w ⋆ , w 0,T0+1 )] + T0 i=1 w 0,i , -g 0,i + α 0 2 T0 i=1 g0,i 2 q ≤ 1 α 0 B(w ⋆ , w 0,1 ) + T0 i=1 w 0,i , -g 0,i + α 0 2 T0 i=1 g0,i 2 q . (19
)
Now we need to bound the three terms in the RHS of ( 19) respectively.
First, we bound B(w
⋆ , w 0,1 ) = B(w ⋆ , w 0 ) B(w ⋆ , w 0,1 ) = w ⋆ -w 0 2 p 2(p -1) (a) ≤ w ⋆ -w 0 2 1 2(p -1) (b) ≤ d w ⋆ -w 0 2 2 2(p -1) (c) ≤ 2d ln(8d),(20)
where inequality (a) holds by the fact that x p ≤ x 1 for all p > 1 and
x ∈ R d . Inequality (b) holds since x 1 ≤ √ d x 2 for all x ∈ R d . Inequality (c) holds since w ⋆ -w 0 2 ≤ w ⋆ 2 + w 0 2 = 2 and 1 p-1 ≤ ln(8d) -1 < ln(8d).
Next, we bound
T0 i=1 -w 0,i , g0,i . Since -w 0,i , g0,i = w 0,i , x 0,i 1 2 y 0,i -1 2 -η I (0 < w 0,i , x 0,i ≤ b 0 ), then | -w 0,i , g0,i | ≤ b 0 and E [ -w 0,i , g0,i ] ≤ b 0 P (x ∈ D 0,i ), by Lemma 8, with probability at least 1 -1/T 2 , we have T0 i=1 -w 0,i , g0,i ≤ T0 i=1 E [ -w 0,i , g0,i | F 0,i-1 ] + b 0 T 0 ln T ≤ b 0 T0 i=1 Pr(x ∈ D 0,i ) + b 0 T 0 ln T . (21) Finally, we bound T0 i=1 g0,i 2 q . Since g 0,i q ≤ 2 g 0,i ∞ , we only need to upper bound T0 i=1 g 0,i 2 ∞ , which satisfies g 0,i ∞ = I(0 ≤ w 0,i , x 0,i ≤ b 0 ) - 1 2 y 0,i + 1 2 -η x 0,i ∞ ≤ x 0,i ∞ .
By Lemma 13, we have with probability at least
1 -1/T 0 T 2 , x 0,i ∞ ≤ 1 Q 1 + ln(dT 0 T 2 ) ≤ 3 Q ln T.
Thus, taking the union bound over i ∈ [T 0 ], we have with probability at least 1 -1/T 2 ,
T0 i=1 g 0,i 2 ∞ ≤ T 0 × 3 Q ln T 2 = 9 Q 2 T 0 (ln T ) 2 .
Hence, with probability at least
1 -1/T 2 , T0 i=1 g0,i 2 q ≤ 4 T0 i=1 g 0,i 2 ∞ ≤ 36 Q 2 T 0 (ln T ) 2 .(22)
Combining ( 18), ( 20), ( 21) and ( 22) together, and taking the union bound, we get with probability at least
1 -3/T 2 , (1 -2η) T0 i=1 f 0,i (w 0,i )P (x ∈ D 0,i ) ≤ T0 i=1 E [ w ⋆ , -g 0,i ] ≤ T0 i=1 w ⋆ , -g 0,i + 24 δ 1 + 2ρb 0 T 0 ln T + 48 δ ln T ≤ 2d ln(8d) α 0 + b 0 T0 i=1 Pr(x ∈ D 0,i ) + b 0 T 0 ln T + 18α 0 Q 2 T 0 (ln T ) 2 + 24 δ 1 + 2ρb 0 T 0 ln T + 48 δ ln T,(23)
where ρ = max U 1 exp(δ), U2 exp(δ) δ . By Lemma 4, property (a), and the fact that b 0 < R, we
have L 1 b 0 ≤ P (x ∈ D 0,i ) ≤ U 1 b 0 , dividing both sides of (23) by (1 -2η)L 1 b 0 T 0 , we get 1 T 0 T0 i=1 f 0,i (w 0,i ) ≤ 1 α 0 2d ln(8d) (1 -2η)L 1 b 0 T 0 + U 1 (1 -2η)L 1 b 0 + 1 (1 -2η)L 1 ln T T 0 + 18α 0 (1 -2η)Q 2 L 1 b 0 (ln T ) 2 + 24 (1 -2η)δL 1 b 0 1 + 2ρb 0 ln T T 0 + 48 (1 -2η)δL 1 b 0 ln T T 0 . By our setting, b 0 = min 1, L 1 R 2 (1-2η) 2 L1L2R 2 2880U 2 1 = c 2 (1 -2η) 2 , T 0 = 576(1+2ρb0)d ln(8d)(ln T ) 2 U 2 1 δ 2 b 4 0 = c 3 1 (1-2η) 8 d ln d(ln T ) 2 , α 0 = Q √ d ln(8d) 3 √ T0 ln T = c 4 √ d ln d √
T0 ln T , then we finish the proof.
The following lemma established by Zhang et al. [2020] indicates that by the construction of the constraint set W 0 , any two vectors in W 0 form an angle that is no bigger than π -1 2 (1 -2η)L 1 R 2 . Lemma 17. (Zhang et al. [2020], Lemma 19) For any two vectors u, v ∈ W
0 = w | w 2 ≤ 1, w, w0 ≥ 1 2 (1 -2η)L 1 R 2 , we have θ (u, v) ≤ π -1 2 (1 -2η)L 1 R 2 .
We use the following corollary to show that, in the i'th iteration, a small value of f 0,i (w 0,i ) indicates that w 0,i and w * are close.
Corollary 5. If w * ∈ W 0 and f 0,i (w 0,i ) < min 1, L 1 R 2 π(1-2η)L2R 2 320U1
, then θ (w ⋆ , w 0,i ) ≤ π 10 .
Proof. We first exclude the case that θ (w ⋆ , w 0,i ) > π 2 , which we prove by contradiction. Suppose θ (w ⋆ , w 0,i ) > π 2 . By Lemma 17 and our choice of b 0 , θ (w ⋆ , w 0,i
) ≤ π -1 2 (1-2η)L 1 R 2 < π -b 0 . From Lemma 11, we get f k,i (w k,i ) ≥ L2 32U1 R 2 (π -θ (w ⋆ , w k,i )).
Together with the condition that
f 0,i (w 0,i ) < π(1-2η)L1L2R 4 320U1 , we have θ (w ⋆ , w 0,i ) > π -π 10 (1 -2η)L 1 R 2 > π -1 2 (1 -2η)L 1 R 2
, which is a contradiction. Thus, we conclude that θ (w ⋆ , w 0,i ) ≤ π 2 .
Next, since f 0,i (w
0,i ) < π(1-2η)L2R 2 320U1 ≤ L2R 2 160U1 • π 2 , by Corollary 3, setting θ 0 = π 2 , then θ (w ⋆ , w 0,i ) ≤ θ0 5 = π 10 .
Putting all pieces together, now we are able to show the main theoretical guarantee of Algorithm 3.
Proposition 3. For the constant c 1 in Proposition 2 and some constants c 2 , c 3 , c 4 > 0, when the initial vector w0 satisfies w ⋆ , w0 ≥ c 1 (1 -2η) and Algorithm 3 runs with bandwidth b 0 = c 2 (1 -2η) 2 for T 0 = c 3 d ln d(ln T ) 2 /(1 -2η) 8 iterations with step size α 0 = c 4 d ln(d)/( √ T 0 ln T ), then its output w 1 satisfies θ (w ⋆ , w 1 ) ≤ π/4 with probability at least 1 -3/T 2 . Proof. In Algorithm 3, the constraint set we choose for gradient update is
W 0 = {w| w 2 ≤ 1, w, w0 ≥ c 1 (1 -2η)}, where c 1 = 1 2 L 1 R 2 , since w ⋆ , w0 ≥ c 1 (1 -2η), we can conclude that w ⋆ ∈ W 0 .
Next, Lemma 16 shows that with probability at least 1 -3/T 2 ,
1 T 0 T0 i=1 f 0,i (w 0,i ) ≤ min 1, L 1 R 2 π(1 -2η)L 2 R 2 2880U 1 . (24
)
Let A 0 denote the set i | f 0,i (w 0,i ) > min 1, L 1 R 2 π(1-2η)L2R 2 320U1
. Combing (24), we have:
min 1, L 1 R 2 π(1 -2η)L 2 R 2 2880U 1 ≥ 1 T 0 T0 i=1 f 0,i (w 0,i ) ≥ |A 0 | T 0 min 1, L 1 R 2 π(1 -2η)L 2 R 2 320U 1 .
Solve the above inequality and we get |A0| T0 ≤ 1 9 . From Corollary 3, set θ 0 = π 2 , we know that when i ∈ Ā0 , θ(w ⋆ , w 0,i ) ≤ π 10 . Thus,
1 T 0 T0 i=1 cos(θ (w ⋆ , w 0,i )) ≥ Ā0 T 0 cos π 10 - |A 0 | T 0 ≥ 1 - |A 0 | T 0 1 - 1 2 π 10 2 - |A 0 | T 0 ≥ 1 - 1 9 1 - 1 2 π 10 2 - 1 9 ≥ cos π 4 ,
where the first inequality holds since cos x is decreasing in x ∈ [0, π] and cos x ≥ -1. The second inequality holds since cos x ≥ 1 -1 2 x 2 for all x ∈ [0, π]. The last inequality holds since
1 -1 9 1 -1 2 π 10 2 -1 9 ≈ 0.73 > 0.71 ≈ cos π 4 .
By the concavity of cos(θ (w ⋆ , •)), using Jensen's inequality, we conclude that when the above inequality holds, we have
cos θ w ⋆ , 1 T 0 T0 i=1 w 0,i ≥ 1 T 0 T0 i=1 cos(θ (w ⋆ , w 0,i )) ≥ cos π 4 .
Thus, we can get that with probability at least 1 -3/T 2 , Algorithm 3 returns a vector w 1 such that θ(w ⋆ , w 1 ) ≤ π 4 .
this section cite: ['b26', 'b23', 'b36', 'b36']

Section: Theoretical Guarantees of Algorithm 4
The following lemma shows that if, in batch k, iteration i, we can identify agents whose true features lie in the localization region D k,i , and we use proxy features to construct a proxy gradient gk,i , then
E [ w ⋆ , -g k,i ] upper bounds E [ w ⋆ , -g k,i
] (and hence upper bounds θ (w ⋆ , w k,i )).
Lemma 18. Given a classification rule h(r) = sgn( w k,i , r + m k,i ) with fixed w k,i ∈ S d and arbitrary m k,i < 0, an agent (x k,i , y k,i ) reports his feature as r k,i according to Lemma 1. Construct proxy data as
x+ k,i := r k,i + (b k -w k,i , r k,i )w k,i I (y k,i = 1, x k,i ∈ D k,i ) ,
xk,i := (r k,i -w k,i , r k,i w k,i )I (y k,i = -1, x k,i ∈ D k,i ) , and define the gradient as gk,i := -ηx + k,i + (1 -η)x - k,i I (x k,i ∈ D k,i ) .
Then, we have
E [ w ⋆ , -g k,i ] ≥ E [ w ⋆ , -g k,i ] .
Proof. First, by Lemma 1, given w k,i ,m k,i , for ∀ x k,i ∈ R d , we have
r k,i + (b k -w k,i , r k,i )w k,i = x k,i + (b k -w k,i , x k,i )w k,i , and r k,i -w k,i , r k,i w k,i = x k,i -w k,i , x k,i w k,i .
Therefore, when x k,i ∈ D k,i = {x| 0 ≤ w k,i , x ≤ b k }, we have the following
w ⋆ , x+ k,i = w ⋆ , r k,i + (b k -w k,i , r k,i )w k,i I (y k,i = 1, x k,i ∈ D k,i ) = w ⋆ , x k,i + (b k -w k,i , x k,i )w k,i I (y k,i = 1, x k,i ∈ D k,i ) = [ w ⋆ , x k,i + (b k -w k,i , x k,i ) w ⋆ , w k,i ] I (y k,i = 1, x k,i ∈ D k,i ) ≥ w ⋆ , x k,i I (y k,i = 1, x k,i ∈ D k,i ) ,(25)
where the inequality holds since w ⋆ , w k,i > 0 and w k,i , x k,i I (x k,i ∈ D k,i ) ≤ b k .
Similarly,
w ⋆ , x- k,i = w ⋆ , r k,i -w k,i , r k,i w k,i I (y k,i = -1, x k,i ∈ D k,i ) = w ⋆ , x k,i -w k,i , x k,i w k,i I (y k,i = -1, x k,i ∈ D k,i ) = [ w ⋆ , x k,i -w k,i , x k,i w ⋆ , w k,i ] I (y k,i = -1, x k,i ∈ D k,i ) ≤ w ⋆ , x k,i I (y k,i = -1, x k,i ∈ D k,i ) . (26
)
where the inequality holds since w ⋆ , w k,i > 0 and w k,i , x k,i I (x k,i ∈ D k,i ) ≥ 0. Combing (25) and ( 26), we have
E [ w ⋆ , -g k,i ] = ηE w ⋆ , x+ k,i -(1 -η)E w ⋆ , x- k,i ≥ ηE [ w ⋆ , x k,i I (y k,i = 1, x k,i ∈ D k,i )] -(1 -η)E [ w ⋆ , x k,i I (y k,i = -1, x k,i ∈ D k,i )] = E [ w ⋆ , -g k,i ] .
The following lemma shows that by pairwise comparing agents' responses under two different declared classifiers designed in Algorithm 4, we can unbiasedly estimate the proxy data desired by Lemma 18 and hence construct a gradient estimator ĝk,i , accordingly.
Lemma 19. Given x+ k,i , xk,i and gk,i defined in Lemma 18, for the proxy data x(1,+) k,i , x(1,-) k,i , x(2,+) k,i , x(2,-) k,i and gradient estimator ĝk,i defined in Algorithm 4, we have E
x(1,+)
k,i - x(2,+) k,i = E x+ k,i , and E x(1,-) k,i - x(2,-) k,i = E x- k,i . Moreover, E [ w ⋆ , -ĝ k,i ] = E [ w ⋆ , -g k,i ] .
Proof. For fixed normal vector w k,i and bandwidth b k , recall that D (1)
k,i = {x | 0 ≤ w k,i , x ≤ γ + b k } and D (2
)
k,i = {x | b k ≤ w k,i , x ≤ γ + b k }. Then, we can ver- ify that D k,i = D (1) k,i /D (2) k,i . By Corollary 4, we have x+ k,i = [r k,i + (b k -w k,i , r k,i )w k,i ]I (y k,i = 1, x k,i ∈ D k,i ) = [x k,i + (b k -w k,i , x k,i )w k,i ]I (y k,i = 1, x k,i ∈ D k,i ) , x- k,i = [r k,i -w k,i , r k,i w k,i ] I (y k,i = -1, x k,i ∈ D k,i ) = [x k,i -w k,i , x k,i w k,i ]I (y k,i = -1, x k,i ∈ D k,i ) , x(1,+) k,i = r (1
)
k,i + b k -w k,i , r (1) k,i w k,i I y (1
)
k,i = 1, γ ≤ w k,i , r(1)
k,i ≤ γ + b k = x (1
)
k,i + b k -w k,i , x (1) k,i w k,i I y (1
)
k,i = 1, x(1)
k,i ∈ D (1
)
k,i , x(1,-) k,i = r (1
)
k,i -w k,i , r(1)
k,i w k,i I y (1
)
k,i = -1, γ ≤ w k,i , r(1)
k,i ≤ γ + b k = x (1
)
k,i -w k,i , x(1)
k,i w k,i I y (1
)
k,i = -1, x(1)
k,i ∈ D (1
)
k,i , x(2,+) k,i = r (2
)
k,i + b k -w k,i , r (2) k,i w k,i I y (2
)
k,i = 1, w k,i , r(2)
k,i = γ + b k = x (2
)
k,i + b k -w k,i , x (2) k,i w k,i I y (2
)
k,i = 1, x(2)
k,i ∈ D (2
)
k,i , x(2,-) k,i = r (2
)
k,i -w k,i , r(2)
k,i w k,i I y (2
)
k,i = -1, w k,i , r(2)
k,i = γ + b k = x (2
)
k,i -w k,i , x(2)
k,i w k,i I y (2
)
k,i = -1, x(2)
k,i ∈ D (2) k,i . Also, (x(1)
k,i , y (1) k,i ) and (x (2) k,i , y (2) k,i ) are drawn i.i.d. from D, thus, E x(1,+) k,i -x(2,+)
k,i = E x (1) k,i + b k -w k,i , x (1) k,i w k,i I y (1
)
k,i = 1, x(1)
k,i ∈ D (1) k,i -E x (2) k,i + b k -w k,i , x (2) k,i w k,i I y (2
)
k,i = 1, x(2)
k,i ∈ D (2) k,i = E [x + (b k -w k,i , x )w k,i ]I y = 1, x ∈ D (1) k,i -E [x + (b k -w k,i , x )w k,i ]I y = 1, x ∈ D (2) k,i = E [[x + (b k -w k,i , x )w k,i ]I (y = 1, x ∈ D k,i )] = E x+ k,i .
Similarly, we can show that E x(1,-)
k,i - x(2,-) k,i = E x- k,i . Therefore, E [ w ⋆ , -ĝ k,i ] = ηE w ⋆ , x(1,+) k,i - x(2,+) k,i -(1 -η)E w ⋆ , x(1,-) k,i - x(2,-) k,i = ηE w ⋆ , x(+) k,i -(1 -η)E w ⋆ , x(-) k,i = E [ w ⋆ , -g k,i ] .
Next, we establish the high probability bound of
T k i=1 E[ w ⋆ , -ĝ k,i | F k,i-1 ] by T k i=1 w ⋆ , -ĝ k,i . Lemma 20. At batch k of Algorithm 4, when w ⋆ -w k,i 2 ≤ r k for ∀i ∈ [T k ]
, then with probability at least 1 -2/T 2 , we have
T k i=1 E[ w ⋆ , -ĝ k,i | F k,i-1 ] ≤ T k i=1 w ⋆ , -ĝ k,i + 24 δ 1 + 2ρ(γ + b k )r k T k ln T + 48 δ r k ln T, where ρ = max U 1 exp(δ), U2 exp(δ) δ .
Proof. First, we partition w ⋆ , -ĝ k,i into two parts, as the following
w ⋆ , -ĝ k,i = w ⋆ , η(x (1,+) k,i - x(2,+) k,i ) -(1 -η)(x (1,-) k,i - x(2,-) k,i ) = w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i (a) -w ⋆ , ηx (2,+) k,i -(1 -η)x (2,-) k,i (b) . (27
)
Since the randomness in part (a) arises from one sample, while the randomness in part (b) arises from another independent sample, parts (a) and (b) are independent. Therefore, we can bound them separately. We first discuss the high-probability tail bound of part (a).
We partition ηx
(1,+) k,i -(1 -η)x (1,-) k,i
into two orthonormal vectors. For notational convenience, we omit the subindex k, i and let x w denote the component of x that is parallel to w k,i , i.e., x w = w k,i , x w k,i , and x ⊥w denote the component of x that is orthogonal to w k,i , i.e., x ⊥w = xx w . Then,
w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i = w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i w (a1) + w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i ⊥w (a2
) . (28
)
Hence, we have to bound part (a1) and ( a2) in ( 28), respectively.
To bound part (a1) in ( 28), we have
w ⋆ , x(1,+) w = w ⋆ , r(1)
w + b k -w k,i , r (1) k,i w k,i I y (1
)
k,i = 1, γ ≤ w k,i , r(1)
k,i ≤ γ + b k = w ⋆ , r(1)
w + b k w k,i -r (1) w I y (1
)
k,i = 1, γ ≤ w k,i , r(1)
k,i ≤ γ + b k = w ⋆ , b k w k,i I y (1
)
k,i = 1, γ ≤ w k,i , r(1)
k,i ≤ γ + b k ≤b k , (29
) where the last inequality holds because w ⋆ , w k,i ≤ w ⋆ 2 w k,i 2 ≤ 1. Similarly,
w ⋆ , x(1,-) w = w ⋆ , r(1)
w -w k,i , r(1)
k,i w k,i I y (1
)
k,i = -1, w k,i , r(1)
k,i = γ + b k = w ⋆ , r(1)
w -r (1
) w I y (1
)
k,i = -1, w k,i , r(1)
k,i = γ + b k =0.
(30) Combing ( 29) and (30), we can bound part (a1) as
w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i w ≤ b k . (31
)
Next, we bound part (a2) in ( 28). From Lemma 1, we get that r (1)
⊥w = r (1
)
k,i + b k -w k,i , r (1) k,i w k,i ⊥w = r (1
)
k,i -w k,i , r(1)
k,i w k,i ⊥w = x (1) ⊥w , hence, w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i ⊥w (a2) = η w ⋆ , r(1)
k,i + b k -w k,i , r (1) k,i w k,i ⊥w I y (1
)
k,i = 1, γ ≤ w k,i , r(1)
k,i ≤ γ + b k -(1 -η) w ⋆ , r(1)
k,i -w k,i , r(1)
k,i w k,i ⊥w I y (1
)
k,i = -1, γ ≤ w k,i , r(1)
k,i ≤ γ + b k = η w ⋆ , x(1)
⊥w I y (1
)
k,i = 1, x(1)
k,i ∈ D (1
)
k,i -(1 -η) w ⋆ , x(1)
⊥w I y (1
)
k,i = -1, x(1)
k,i ∈ D (1) k,i = w ⋆ , x (1) ⊥w 1 2 y (1
)
k,i - 1 2 -η I x (1) k,i ∈ D (1
)
k,i , where the second equality holds by Corollary 4.
Since 1 2 y
(1) k,i -1 2 -η ≤ 1, we only need to establish the high probability bound of
w ⋆ , x (1
) ⊥w I x (1) k,i ∈ D (1
)
k,i , for a > b k we have P w ⋆ , x (1
) ⊥w I x (1) k,i ∈ D (1) k,i ≥ a -b k = P w ⋆ , x (1) ⊥w ≥ a -b k , 0 ≤ w k,i , x(1)
k,i ≤ γ + b k = P w ⋆ ⊥w , x (1) k,i ≥ a -b k , 0 ≤ w k,i , x(1)
k,i ≤ γ + b k ,
where the last inequality holds because w ⋆ , x
⊥w = w ⋆ ⊥w , x (1) ⊥w + w ⋆ w , x (1) ⊥w = w ⋆ ⊥w , x(1)
⊥w = w ⋆ ⊥w , x(1)
⊥w + w ⋆ ⊥w , x(1)
w = w ⋆ ⊥w , x(1)
⊥w I x ∈ D (1) k,i ≥ a -b k = +∞ a-b k ∥w ⋆ ⊥w ∥ 2 γ+b k 0 ϕ(X, Y )dXdY ≤ U 2 +∞ a-b k ∥w ⋆ ⊥w ∥ 2 γ+b k 0 exp(-δ X 2 + Y 2 )dXdY ≤ U 2 (γ + b k ) +∞ a-b k ∥w ⋆ ⊥w ∥ 2 exp(-δX)dXdY = U 2 (γ + b k ) δ exp -δ a -b k w ⋆ ⊥w 2 ≤ U 2 (γ + b k ) δ exp -δ a -b k r k ≤ U 2 exp(δ)(γ + b k ) δ exp -δ a r k ,(1)
where the third inequality holds since w ⋆ -w k,i ≤ r k < π 2 , which implies w ⋆ ⊥w 2 ≤ r k . The last inequality holds since b k < r k by our setting. Since 1 2 y
(1)
k,i -1 2 -η ≤ 1, for a ≥ b k , we have P w ⋆ , x (1) ⊥w 1 2 - 1 2 -η y (1) k,i I x (1) k,i ∈ D (1) ≥ a -b k ≤ P w ⋆ , x(1)
⊥w I x (1) k,i ∈ D (1) k,i ≥ a -b k ≤ U 2 exp(δ)(γ + b k ) δ exp -δ a r k .(32)
Combing ( 32) and ( 31), we get that for a ≥ b k ,
P w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i ≥ a ≤ P w * , ηx (1,+) k,i -(1 -η)x (1,-) k,i ⊥w ≥ a -b k ≤ U 2 exp(δ)(γ + b k ) δ exp -δ a r k . (33
) For 0 < a < b k , P w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i ≥ a ≤ P w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i > 0 ≤ P x (1) k,i ∈ D (1) k,i ≤ U 1 (γ + b k ) ≤ U 1 exp(δ)(γ + b k ) exp -δ a r k . (34
)
Where the third inequality holds by Lemma 4 property (a) and the last equality holds since a < b k < r k .
Combing (33) and ( 34), we establish probability tail bound of (a) in ( 27): for ∀a > 0 and ρ = max U 1 exp(δ), U2 exp(δ) δ , we have
P w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i ≥ a ≤ ρ(γ + b k ) exp -δ a r k .
Following the same technique, we can bound part (b) of ( 27) for a > 0 by:
P w ⋆ , ηx (2,+) k,i -(1 -η)x (2,-) k,i ≥ a ≤ ργ exp -δ a r k .
By Lemma 9, part (a) and part (b) in ( 27) are 6 δ 1 + 2ρ(γ + b k )r k , 6 δ r k -subexponential, thus, by Lemma 7, we get that with probability at least 1 -1/T 2 ,
T k i=1 E w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i F k,i-1 ≤ T k i=1 w ⋆ , ηx (1,+) k,i -(1 -η)x (1,-) k,i + 12 δ 1 + 2ρ(γ + b k )r k T k ln T + 24 δ r k ln T,(35)
and with probability at least 1 -1/T 2 ,
T k i=1 E w ⋆ , ηx (2,+) k,i -(1 -η)x (2,-) k,i F k,i-1 ≥ T k i=1 w ⋆ , ηx (2,+) k,i -(1 -η)x (2,-) k,i - 12 δ 1 + 2ρ(γ + b k )r k T k ln T - 24 δ r k ln T. (36
)
Taking the union bound of ( 35) and ( 36), we get that with probability at least 1 -2/T 2 ,
T k i=1 E[ w ⋆ , -ĝ k,i )| F k,i-1 ] ≤ T k i=1 w ⋆ , -ĝ k,i + 24 δ 1 + 2ρ(γ + b k )r k T k ln T + 48 δ r k ln T.
We then show that by our construction of ĝk,i ,
T k i=1 -w k,i , ĝk,i also has a high probability upper bound.
Lemma 21 (High probability bound of T k i=1 -w k,i , ĝk,i ). At batch k, with probability at least 1 -2/T 2 , we have
T k i=1 -w k,i , ĝk,i ≤ ηb k T k i=1 P (x ∈ D k,i ) + 2ηb k T k ln T . Proof. Since ĝk,i = -η(x (1,+) k,i - x(2,+) k,i ) + (1 -η)(x (1,-) k,i - x(2,-) k,i
), and we have the
w k,i , x(1,+) k,i = b k I(y(1)
k,i = 1)I(x (1) k,i ∈ D (1) k,i ), w k,i , x(2,+) k,i = b k I(y(2)
k,i = 1)I(x(2)
k,i ∈ D (2) k,i ), w k,i , x(1,-) k,i = w k,i , x(2,-) k,i = 0. Thus, E[ w k,i , -ĝ k,i ] = ηb k E I y = 1, x ∈ D (1) k,i -I y = 1, x ∈ D (2) k,i = ηb k P (y = 1, x ∈ D k,i ) = ηb k P (y = 1 | x ∈ D k,i ) P (x ∈ D k,i ) .
Also, since x(1,+)
k,i and x(1,-) k,i are calculated by one sample while x(2,+) k,i and x(2,-) k,i
are calculated by another independent sample, we can reformulate w k,i , -ĝ k,i as the following:
w k,i , -ĝ k,i = w k,i , ηx (1,+) k,i -(1 -η)x (1,-) k,i (a) -w k,i , ηx (2,+) k,i -(1 -η)x (2,-) k,i (b) ,(37)
where (a) and (b) in (37) are independent. Thus, we establish the high probability bound of parts (a) and (b), respectively.
For part (a),
w k,i , ηx (1,+) k,i -(1 -η)x (1,-) k,i
≤ ηb k , so by Lemma 8, with probability at least
1 -1/T 2 , T k i=1 w k,i , ηx (1,+) k,i -(1 -η)x (1,-) k,i ≤ T k i=1 E w k,i , ηx (1,+) k,i -(1 -η)x (1,-) k,i F k,i-1 + ηb k T k ln T .
Similarly, for part (b), with probability at least 1 -1/T 2 ,
T k i=1 w k,i , ηx (2,+) k,i -(1 -η)x (2,-) k,i ≥ T k i=1 E w k,i , ηx (2,+) k,i -(1 -η)x (2,-) k,i F k,i-1 -ηb k T k ln T . (38
)
Combing part (a) and (b) above, and taking the union bound, we get that with probability at least 1 -2/T 2 ,
T k i=1 -w k,i , ĝk,i ≤ E T k i=1 -w k,i , ĝk,i + 2ηb k T k ln T = ηb k T k i=1 Pr(y = 1| x ∈ D k,i ) Pr(x ∈ D k,i ) + 2ηb k T k ln T ≤ ηb k T k i=1 Pr(x ∈ D k,i ) + 2ηb k T k ln T . (39
)
The following lemma shows a high probability upper bound of
T k i=1 ĝk,i 2 q . Lemma 22 (High probability bound of T k i=1 ĝk,i 2 q
). In Algorithm 4, with probability at least 1 -2/T 2 ,
T k i=1 ĝk,i 2 ∞ ≤ 144 Q 2 T k (ln T ) 2 .
Proof. For q = ln(8d) > 2, ĝk,i q ≤ 2 ĝk,i ∞ , hence, we only need to establish the high probability bound of ĝk,i ∞ . By our construction of ĝk,i ,
ĝk,i ∞ = -η(x (1,+) k,i - x(2,+) k,i ) + (1 -η)(x (1,-) k,i - x(2,-) k,i ) ∞ ≤ η x(1,+) k,i ∞ + (1 -η) x(1,-) k,i ∞ + η x(2,+) k,i ∞ + (1 -η) x(2,-) k,i ∞ = η [x (1) k,i + (b k -w k,i , x(1)
k,i )w k,i ]I(y k,i = 1)I(x (1) k,i ∈ D (1) k,i ) ∞ + (1 -η) (x (1) k,i -w k,i , x(1)
k,i w k,i )I(y k,i = -1)I(x (1) k,i ∈ D (1) k,i ) ∞ + η [x (2) k,i + (b k -w k,i , x(2)
k,i )w k,i ]I(y k,i = 1)I(x (2) k,i ∈ D (2) k,i ) ∞ + (1 -η) (x (2) k,i -w k,i , x(2)
k,i w k,i )I(y k,i = -1)I(x (2) k,i ∈ D k,i ) ∞ ≤ x (1) k,i ∞ + x (2) k,i ∞ + 2(γ + b k ).
From Lemma 13, we get that ,with probability at least 1 -1
T k T 2 , x (1) k,i ∞ ≤ 1 Q 1 + ln(dT k T 2 ) ,(40)
and with probability at least 1 -1
T k T 2 , x (2) k,i ∞ ≤ 1 Q 1 + ln(dT k T 2 ) .(41)
Taking the union bound of ( 40),( 41), we get that with probability at least 1 -2
T k T 2 , ĝk,i ∞ ≤ 2 Q + 2 Q ln(dT k T 2 ) + 2(γ + b k ).
Taking union bound over all iterations i ∈ [T k ], we have that with probability at least 1 -2/T 2 ,
T k i=1 ĝk,i 2 ∞ ≤ T k 2 Q + 2 Q ln(dT k T 2 ) + 2(γ + b k ) 2 ≤ T k 2 Q ln(T 3 ) 2 = 36 Q 2 T k (ln T ) 2 .
Thus, we conclude that with probability at least 1 -2/T 2 ,
T k i=1 ĝk,i 2 q ≤ 4 T k i=1 ĝk,i 2 ∞ ≤ 144 Q 2 T k (ln T ) 2 .
Given the starting angle of batch k as θ k = π 2 k+1 , the following lemma establishes the high probability upper bound of the average of f k,i (w k,i ). Lemma 23. In Algorithm 4, at every batch k ∈ {1, 2, • • • , K}, if θ(w ⋆ , w k ) ≤ θ k , there exists some constants c 5 , c 6 , c 7 > 0, when setting bandwidth b
k = c 5 1-2η 2 k , iteration number T k = c 6 (γ+1)d ln d ln T (1-2η) 4 • 4 k , step size α k = c 7 √ dθ k
T k ln T , then with probability at least 1 -6 T 2 , the following holds 1 T k
T k i=1 f k,i (w k,i ) ≤ L 2 R 2 θ k 12800U 1 . (42
)
Proof. Combing Lemma 10, Lemma 18 and Lemma 19, we have (1 -2η)f k,i (w k,i )P (x ∈ D k,i ) ≤ E [ w ⋆ , -ĝ k,i ], hence, it suffices to upper bound
T k i=1 E [ w ⋆ , -ĝ k,i ]. First, we upper bound T k i=1 E [ w ⋆ , -ĝ k,i ] by T k i=1 w ⋆ , -ĝ k,i . By our setting of constraint set, cos(θ (w k,i , w k )) = w k,i , w k ≥ cos θ k , since cos θ is decreasing in θ ∈ [0, π], hence, we have θ (w k,i , w k ) ≤ θ k , thus, w k,i -w k 2 ≤ θ (w k,i , w k ) ≤ θ k . Also, w k -w ⋆ 2 ≤ θ (w k , w ⋆ ) ≤ θ k , by Lemma 12, w k,i -w ⋆ 2 ≤ 2θ k .
According to Lemma 20, set r k = 2θ k , then with probability at least 1 -2/T 2 , the following holds:
T k i=1 E[ w ⋆ , -ĝ k,i )| F k,i-1 ] ≤ T k i=1 w ⋆ , -ĝ k,i 48 δ 1 + 2ρ(γ + b k )θ k T k ln T + 96 δ θ k ln T.
(43) Next, we move on to upper bound
T k i=1 w ⋆ , -ĝ k,i through a nonstandard regret analysis of online mirror decent. Let B(v 1 , v 2 ) := 1 2(p-1) v 1 -v 2 2 p denotes the Bregman divergence w.r.t. 1 2(p-1) • 2 p , where p = ln (8d) ln (8d)-1 .
In each iteration i, the regularizer, B(•, w k,i-1 ) is 1-strongly convex with respect to • p [see Shalev-Shwartz [2007]]. From the analysis of online mirror descent [see Orabona [2023], Lemma 6.9], with step size α k , we have
α k ĝk,i , w k,i -w ⋆ ≤ B(w ⋆ , w k,i ) -B(w ⋆ , w k,i+1 ) + α 2 k 2 ĝk,i 2 q .
Where q = ln(8d) > 2. Summing the above equality over i ∈ [T k ], we get
T k i=1 α k ĝk,i , w k,i -w ⋆ ≤ B(w ⋆ , w k,1 ) -B(w ⋆ , w k,T k +1 ) + α 2 k 2 T k i=1 ĝk,i 2 q .
Dividing both sides by α k , and moving T k i=1 w k,i , ĝk,i to RHS, we get
T k i=1 w ⋆ , -ĝ k ≤ 1 α k [B(w ⋆ , w k,1 ) -B(w ⋆ , w k,T k +1 )] + T k i=1 w k,i , -ĝ k,i + α k 2 T k i=1 ĝk,i 2 q ≤ 1 α k B(w ⋆ , w k,1 ) + T k i=1 w k,i , -ĝ k,i + α k 2 T k i=1 ĝk,i 2 q .
(44) Now we need to bound the three terms in the RHS of (44) respectively.
First, we bound B(w ⋆ , w k,1 ) = B(w ⋆ , w k ).
B(w ⋆ , w k,1 ) = w ⋆ -w k 2 p 2(p -1) (a) ≤ w ⋆ -w k 2 1 2(p -1) (b) ≤ d w ⋆ -w k 2 2 2(p -1) (c) ≤ d ln(8d)θ 2 k 2 . (45
)
Where inequality (a) holds by the fact that x p ≤ x 1 for all p > 1 and
x ∈ R d . Inequality (b) holds since x 1 ≤ √ d x 2 for all x ∈ R d . Inequality (c) holds since w ⋆ -w k 2 ≤ θ (w ⋆ , w k ) ≤ θ k .
Next, we bound T k i=1 -w k,i , ĝk,i . By Lemma 21, we have that with probability at least 1 -2/T 2 ,
T k i=1 -w k,i , ĝk,i ≤ ηb k T k i=1 P (x ∈ D k,i ) + 2ηb k T k ln T .(46)
Finally, we bound
T k i=1 ĝk,i 2 q
. By Lemma 22, we have with probability at least 1 -2/T 2 ,
T k i=1 ĝk,i 2 ∞ ≤ 144 Q 2 T k (ln T ) 2 . (47
)
Combining ( 43), ( 45), ( 46) and ( 47), and taking the union bound, we get that with probability at least 1 -6/T 2 , (1 -2η)
T k i=1 f k,i (w k,i )P (x ∈ D k,i ) ≤ T k i=1 E [ w ⋆ , -ĝ k,i ] ≤ 1 α k d ln(8d)θ 2 k 2 + ηb k T k i=1 P (x ∈ D k,i ) + 2ηb k T k ln T + 72 Q 2 α k T k (ln T ) 2 + 48 δ 1 + 2ρ(γ + b k )θ k T k ln T + 96 δ θ k ln T.(48)
By Lemma 4, property (a), and the fact that b
k ≤ b 1 < R, we have L 1 b k ≤ P (x ∈ D k,i ) ≤ U 1 b k , dividing both sides of (48) by (1 -2η)L 1 T k b k , we get 1 T k T k i=1 f k,i (w k,i ) ≤ 1 α k d ln(8d)θ 2 k 2(1 -2η)L 1 T k b k + U 1 η (1 -2η)L 1 b k + 2η (1 -2η)L 1 ln T T k + 72α k Q 2 (1 -2η)L 1 b k (ln T ) 2 + 48 δ(1 -2η)L 1 1 + 2ρ(γ + b k ) θ k b k ln T T k + 96 δ(1 -2η)L 1 θ k b k ln T T k . (49) By our setting, b k = (1-2η)L1L2R 2 38400U 2 1 θ k = c 5 1-2η 2 k , T k = 576π 2 (1+2ρ(γ+b1))d ln(8d)(ln T ) 2 δ 2 c 2 5 (1-2η) 2 b 2 k = c 6 (γ+1)d ln d(ln T ) 2 (1-2η) 4 4 k , α k = Q √ d ln(8d)θ k 12 √ T k ln T = c 7 √ d ln dθ k √
T k ln T , then we get our proof.
Based on Lemma 23, we establish the main theoretical guarantee of Algorithm 4 in Proposition 4.
Proposition 4. For some constants c 5 , c 6 , c 7 > 0, when Algorithm 4 runs with an initial vector w k satisfying θ (w
⋆ , w k ) ≤ θ k = π/2 k+1 , bandwidth b k = c 5 (1 -2η)2 -k for T k = c 6 4 k (γ + 1)d ln d(ln T ) 2 /(1 -2η) 4 iterations with step size α k = c 7 √ d ln dθ k /( √ T k ln T ), its output w k+1 satisfies θ (w ⋆ , w k+1 ) ≤ θ k+1 = θ k
2 with probability at least 1 -6/T 2 .
Proof. For the given unit vector w k that satisfy θ(w ⋆ , w k ) ≤ θ k ≤ π 4 , we have
w ⋆ -w k 2 ≤ 2 sin θ(w ⋆ , w k ) 2 ≤ 2 sin θ k 2 ≤ θ k .
The first inequality holds since w ⋆ 2 = w k 2 = 1. The second inequality holds since sin x is increasing in x ∈ [0, π 2 ], the last inequality holds since sin x ≤ x for all 0 ≤ x ≤ π 2 . By our choice of W k , for every iteration i, we have cos(θ (w k,i , w k )) = w k,i , w k ≥ cos(θ k ), thus, since cos x is decreasing for 0 ≤ x ≤ π, then θ (w k,i , w k ) ≤ θ k , hence,
θ(w ⋆ , w k,i ) ≤ θ (w ⋆ , w k ) + θ (w k , w k,i ) ≤ 2θ k ,(50)
where the first inequality holds by Lemma 12. By Lemma 10, with probability at least 1 -6 T 2 ,
1 T k T k i=1 f k,i (w k,i ) ≤ L2R 2 θ k 12800U1 . Let A k := {i ∈ [T k ]| f k,i (w k,i ) ≥ L2 160U1 R 2 θ k }. Thus, L 2 R 2 θ k 12800U 1 ≥ 1 T k T k i=1 f k,i (w k,i ) ≥ L 2 R 2 θ k 160U 1 |A k | T k .
From the above inequality and we get |A k | T k ≤ 1 80 , and thus | Āk | T k ≥ 79 80 . By Corollary 3, the iterations i ∈ Āk satisfy θ(w ⋆ , w k,i ′ ) ≤ θ k 5 . Other iterations i ∈ A k satisfy θ (w ⋆ , w k,i ) ≤ 2θ k by (50). Therefore,
1 T k T k i=1 cos(θ (w ⋆ , w k.i )) ≥ cos 1 5 θ × | Āk | T k + cos(2θ) × |A k | T k ≥ 1 - 1 50 θ 2 × 79 80 + 1 - 1 2 × 4θ 2 × 1 80 ≥ 1 - 1 50 θ 2 - 1 2 × 1 20 θ 2 ≥ 1 - 1 20 θ 2 = 1 - 1 5 × 1 2 θ 2 ≥ cos θ 2 ,
where the second inequality utilizes the fact that cos x ≥ 1 -1 2 x 2 and the last inequality holds since cos x ≤ 1 -1 5 x 2 for 0 ≤ x ≤ π 2 . By the concavity of cos(θ (w ⋆ , •)) when θ(w ⋆ , w k,i ) ≤ π 2 , we have
cos(θ (w ⋆ , w k+1 )) = cos θ w ⋆ , 1 T k T k i=1 w k,i ≥ 1 T k T k i=1 cos(θ (w ⋆ , w k.i )) ≥ cos θ k 2 .
Since cos x is decreasing in x ∈ [0, π], we have θ(w ⋆ , w k+1 ) ≤ θ k 2 .
this section cite: ['b26', 'b23']

Section: A.6 Proofs for Section 4
In this section, we outline the proof of Theorem 1, which is the key theorem of this paper.
Theorem 1. For any instance of our online strategic classification problem with noise level η, maximum manipulation distance γ, and feature dimension d, the expected regret of classifiers h from Algorithm 1 over T cycles satisfies E[Reg( h; T )] = O d ln d × (ln T ) 2 /(1 -2η) 8 + (γ + 1)d ln d × T ln T /(1 -2η) 2 .
Proof. To derive the regret bound in Theorem 1, we decompose the total regret Reg( h; T ) as defined in (1) into two parts according to pure exploration phase versus exploration-exploitation phase, and then we move on to decompose the regret in the exploration-exploitation phase according to certain events. We upper bound each of these parts separately. First, we define the two phases and the events used in the regret decomposition.
Definition 4. Define the set T PE := {t ∈ [T ] | 0 < t ≤ 2T init + T 0 } as the pure exploration phase, where T init and T 0 are number of iterations in Algorithm 2 and Algorithm 3. Define the set T EE := {t ∈ [T ] | 2T init + T 0 < t ≤ T } as the exploration-exploitation phase. Define the event ε init := { w0 , w ⋆ ≥ c 0 (1 -2η)}, where c 0 is a constant defined in Proposition 2. For k ∈ {0, 1, 2, • • • , K}, define the event ε k := {θ (w k+1 , w ⋆ ) ≤ π 2 k+2 }. Define the event ε := ε init ε 0 k∈[K] ε k as the "clean event".
In Definition 4, the pure exploration phase T P E corresponds to all cycles in the Initialization and Refinement Algorithm, and the exploration-exploitation phase T EE corresponds to all cycles in the Enhancement Algorithm. By Proposition 2, Proposition 3 and Proposition 4, the events defined in Definition 4 satisfy the following properties:
P (ε init ) ≥ 1 - 2 T 2 ,(51)
P (ε 0 | ε init ) ≥ 1 - 3 T 2 , (52
) P (ε k | ε k-1 , ε k-2 , • • • ε 0 , ε init ) = P (ε k | ε k-1 ) ≥ 1 - 6 T 2 , ∀k ∈ [K] .(53)
Hence, taking the union bound by (51),( 52) and ( 53), we get that the probability of clean event satisfy
P (ε) = P   ε init ε 0 k∈[K] ε k   ≥ 1 - 2 T 2 × 1 - 3 T 2 × 1 - 6 T 2 K ≥ 1 - 6 T .
Then, we decompose the total regret as
Reg( h; T ) = Reg( h; T P E ) + Reg( h; T EE , ε) + Reg( h; T EE , ε),(54)
where ε denotes the complement of ε, and the three parts of ( 54) is represented as the following:
Reg( h; T P E ) = t∈T P E Err( ht ) -| T P E | × Err( h⋆ ) , Reg( h; T EE , ε) = t∈T EE Err( ht ) -Err( h⋆ ) I (ε) , Reg( h; T EE , ε) = t∈T EE Err( ht ) -Err( h⋆ ) I (ε) .
The first term in (54) denotes the expected regret incurred during the pure exploration phase. The second term captures the expected regret incurred during the exploration-exploitation phase, given that the clean event holds. The last term characterizes the expected regret incurred during the explorationexploitation phase, given that the clean event does not hold.
Now we upper bound the three parts of ( 54) respectively. For the first term Reg( h; T P E ), the regret incurred in a single cycle is at most 1, and the length of pure exploration phase is |T P E | = 2T init +T 0 , then, the expected total regret during these time can be upper bounded by
E Reg( h; T P E ) ≤ t∈T P E 1 ≤ 2T init + T 0 = O 1 (1 -2η) 8 d ln d(ln T ) 2 ,(55)
where the last equality holds by our setting that T init = O 1 (1-2η) 2 ln T and T 0 = O 1 (1-2η) 8 d ln d(ln T ) 2 . Then, we upper bound the expectation of the second term Reg( h; T EE , ε), which is the cumulative regret incurred under the clean event during the Enhancement procedure. Let Reg k ( h; T EE , ε) denote the regret in each batch k ∈ {1, 2, • • • , K} during this procedure under "clean event" i.e., Reg( h; T EE , ε) = K k=1 Reg k ( h; T EE , ε), then we only need to upper bound E Reg k ( h; T EE , ε) for each batch k ∈ {1, 2, • • • , K}, which is characterized as
E Reg k ( h; T EE , ε) = T k i=1 2 j=1 E Err( h(j) k,i ) -Err( h⋆ ) ε P (ε) ≤ T k i=1 2 j=1 E Err( h(j) k,i ) -Err( h⋆ ) ε ≤ T k i=1 E I sgn w k,i , r(1)
k,i -γ = y (1
)
k,i -I sgn w ⋆ , r (1, * ) k,i -γ = y (1) k,i ε + T k i=1 E I sgn w k,i , r(2)
k,i -γ -b k = y (2
)
k,i -I sgn w ⋆ , r (2, * ) k,i -γ = y (2) k,i ε = T k i=1 E I sgn w k,i , x(1)
k,i = y (1
)
k,i -I sgn w ⋆ , x(1)
k,i = y (1) k,i ε + T k i=1 E I sgn w k,i , x(2)
k,i -b k = y (2
)
k,i -I sgn w ⋆ , x(2)
k,i = y (2) k,i ε . (56) Where r (1, * ) k,i and r (2, * ) k,i are the counterfactual agent responses under the optimal classifier. The first equality holds by the fact that E [XI (ε)
] = E [X | I (ε) = 1] P (ε) + E [0 | I (ε) = 0] P (ε) = E [X | ε] P (ε).
The inequality holds since 0 ≤ P (ε) ≤ 1. The last equality holds by Proposition 1.
For the first term within the summation in the RHS of (56), we have
E I sgn w k,i , x (1) k,i = y (1) k,i -I sgn w ⋆ , x (1) k,i = y (1) k,i ε ≤ P (sgn ( w ⋆ , x ) = sgn ( w k,i , x ) | ε) ≤ c 10 2 k , (57
)
where c 10 is a positive constant. The first inequality holds by the triangular inequality, the last equality holds since by Lemma 4 property (b), and Proposition 4.
For the second term, we have
E I sgn w k,i , x(2)
k,i -b k = y (2
)
k,i -I sgn w ⋆ , x(2)
k,i = y (2) k,i ε = E I sgn w k,i , x(2)
k,i -b k = y (2
)
k,i -I sgn w k,i , x(2)
k,i = y (2) k,i ε + E I sgn w k,i , x(2) k,i = y (2)
k,i -I sgn w ⋆ , x(2)
k,i = y (2) k,i ε ≤ E I sgn w k,i , x(2)
k,i -b k = sgn w k,i , x(2)
k,i ε + E I sgn w k,i , x(2)
k,i = sgn w ⋆ , x(2)
k,i ε = P 0 ≤ w k,i , x(2)
k,i < b k + P (sgn ( w ⋆ , x ) = sgn ( w k,i , x ) | ε) ≤ U 1 b k + c 10 2 k ≤ c 11 (1 -2η) 2 k + c 10 2 k ,(58)
where c 11 > 0 is a positive constant. The first inequality holds by the triangular inequality, the last inequality holds by Lemma 4 property (a) and (b), and Proposition 4.
Summing ( 57) and ( 58) over [T k ], and then we can upper bound ( 56) by . Then, we can upper bound the cumulative regret during the exploration-exploitation phase under "clean event" as
E Reg k ( h; T EE , ε) ≤ T k i=1 O(1) • 1 2 k + O(1) • 1 -2η 2 k = O(1) • T k 2 k + O(1) • (1 -2η)T k 2 k = 2 k O 1 (1 -2η) 4 (γ + 1)d ln d(ln T ) 2 . Since T = |T P E | + |T EE |
E Reg( h; T EE , ε) = K k=1 E Reg k ( h; T EE , ε) = log 4 ( O ( (1-2η) 4 T (γ+1)d ln d(ln T ) 2 )) k=1 2 k O 1 (1 -2η) 4 (γ + 1)d ln d(ln T ) 2 = O 1 (1 -2η) 2 (γ + 1)d ln dT ln T .(59)
Finally, we upper bound the third term in ( 54) as
E Reg h; T EE , ε = t∈T EE E Err( ht ) -Err( h⋆ ) I (ε) ≤ E t∈T EE I (ε) = |T EE | P (ε) ≤ T • 6 T = 6.(60)
By combining the upper bounds of (55), (59), and (60), we finish the proof.
this section cite: []

Section: A.7 Proofs for Appendix A.1
Before proving Lemma 3, we need to first prove some intermediate lemmas. In the following lemma, we show that any log-concave distributed random vector with zero mean and positive definite covariance matrix can be linearly transformed into a new random vector with an isotropic log-concave distribution.
Lemma 3. Let x ∈ R d (d ≥ 2) have zero mean and a log-concave distribution. Suppose the eigenvalues of its covariance matrix Σ = E xx are all bounded within [λ, λ] for some positive constants λ, λ, then the distribution of x satisfies the regularity conditions in assumption 2, with parameters
L 1 = β1(1) √ λ , L 2 = β1(2) λ , R = 1 9 √ λ, U 1 = 1 √ λ , U 2 = β2(2) λ , δ = β3(2) √ λ , Q = √ λ for β 1 (1), β 1 (2
), β 2 (2), β 3 (2) given in Lemma 2. Proof for Lemma 3. For arbitrary 1-dimensional subspace V 1 ⊆ R 1 and 2-dimensional subspace V 2 ⊆ R 2 , let x V1 and x V2 denote the projected vectors on V 1 and V 2 with covariance matrices Σ V1 and Σ V2 , respectively. Let λV1 and λV2 denote the maximum eigenvalues of Σ V1 , Σ V2 and λ V1 and λ V2 denote the minimum eigenvalues of Σ V1 , Σ V2 , respectively. Then by Lemma 25, we have
λ ≥ λV1 , λ ≥ λV2 , λ ≤ λ V1 and λ ≤ λ V2 . Let z V1 = Σ -1 2 V1 x V1 , z V2 = Σ -1 2 V2 x V2 .
Then, z V1 and z V2 have isotropic log-concave densities, denoted as ϕ z V 1 and ϕ z V 2 , respectively.
We first determine L 1 , L 2 and R prescribed in Assumption 2, Condition 1. For all
x V1 2 ≤ 1 9 √ λ, x V1 's probability density function ϕ x V 1 (•) satisfies ϕ x V 1 (x V1 ) = ϕ z V 1 Σ -1 2 V1 x V1 det Σ -1 2 V1 ≥ β 1 (1) λV1 ≥ β 1 (1) √ λ . (61
) Now z V1 2 = Σ -1 2 V1 x V1 2 ≤ 1 √ λ V 1 x V1 2 ≤ 1 √ λ • 1 9 √ λ = 1 9 .
Then, the first inequality in Similarly, for all Combining (61) and (62), we have
x V2 2 ≤ 1 9 √ λ, x V2 's probability density function ϕ V2 (•) satisfies ϕ x V 2 (x V2 ) = ϕ z V 2 Σ -1 2 V2 x V2 det Σ -1 2 V2 ≥ β 1 (2) λV2 ≥ β 1 (2) λ . (62
) Now z V2 2 = Σ -1 2 V2 x V2 2 ≤ 1 √ λ V 2 x V2 2 ≤ 1 √ λ • 1 9 √ λ = 1 9 .
L 1 = β1(1) √ λ , L 2 = β1(2) λ , R = 1 9 √ λ.
Then, we determine U 1 , U 2 and δ prescribed in Assumption 2, Condition 2. For ϕ x V 1 (•), we have
ϕ x V 1 (x V1 ) = ϕ z V 1 Σ -1 2 V1 x V1 det Σ -1 2 V1 ≤ 1 λ V1 ≤ 1 √ λ ,
where the first the inequality holds by Lemma 2, property (c) and det Σ
-1 2 V1 ≤ 1 √ λ V 1
. The second inequality holds by Lemma 25. Thus, U 1 = 1 √ λ .
For ϕ x V 2 (•), we have
ϕ x V 2 (x V2 ) = ϕ z V 2 Σ -1 2 V2 x V2 det Σ -1 2 V2 ≤ β 2 (2) exp -β 3 (2) Σ -1 2 V2 x V2 2 λ V2 ≤ β 2 (2) exp -β3(2) √ λV 2 x V2 2 λ V2 ≤ β 2 (2) exp -β3(2) √ λ x V1 2 λ ,
where the first inequality holds by Lemma 2, property (d) and det Σ
-1 2 V2 ≤ 1 λ V 2 . The second inequality holds by Σ -1 2 V2 x V2 2 ≥ 1 √ λV 2
x V2 2 . The last inequality holds by Lemma 25.
Thus, we have U 2 = β2(2) λ , δ = β3(2) √ λ .
Finally, we determine Q prescribed in Assumption 2, Condition 3. For arbitrary w ∈ B d , w, x forms a 1-dimensional random variable whose probability density function is log-concave. Let x V ′ 1 := w, x denote the projected random variable and σ 2
V ′ 1 = E x T V ′ 1 x V ′
1 denote its variance. Then, by Lemma 2 property (a), for every t > 0,
P (| z | > t) ≤ e -σ V ′ 1 t+1 ≤ e - √ λt+1 ,
where the second inequality holds by σ 2
V ′ 1 ≥ λ V ′ 1 ≥ λ. Thus Q = √ λ.
this section cite: []

Section: 
Lemma 24. For any random vector x ∈ R d that follows a log-concave distribution with E [x] = 0 and E xx T = Σ, where Σ is positive definite, the transformed random vector z = Σ -1 2 x that follows an isotropic log-concave distribution.
Proof. We first prove that z
= Σ -1 2 x is isotropic. Since E xx T = Σ, Σ is positive definite, then E zz T = Σ -1 2 E xx T Σ -1 2 T = Σ -1 2 Σ Σ -1 2 T = I. Also, since E [x] = 0, then E [z] = Σ -1 2 E [x] = 0.
Next, we show that the probability density function of z is log-concave. Suppose the corresponding probability density functions of x and z are ϕ x (•) are ϕ z (•), respectively. Since ϕ x (•) is logconcave, then for ∀ α ∈ [0, 1] and
∀ x 1 , x 2 ∈ R d , α ln (ϕ x (x 1 )) + (1 -α) ln (ϕ x (x 2 )) ≤ ln (ϕ x (αx 1 + (1 -α) x 2 )) . Then, for ∀ α ∈ [0, 1], z 1 , z 2 ∈ R d , α ln ((ϕ z (z 1 ))) + (1 -α) ln ((ϕ z (z 2 ))) = α ln det Σ 1 2 ϕ x Σ 1 2 z 1 + (1 -α) ln det Σ 1 2 ϕ x Σ 1 2 z 2 = α ln ϕ x Σ 1 2 z 1 + (1 -α) ln ϕ x Σ 1 2 z 2 + ln det Σ 1 2 ≤ ln ϕ x αΣ 1 2 z 1 + (1 -α) Σ 1 2 z 2 + ln det Σ 1 2 = ln (ϕ z (αz 1 + (1 -α) z 2 )) . Thus, ϕ z (•) is isotropic log-concave.
The next lemma outlines the relationship between the eigenvalues of the covariance matrices for a high-dimensional random variable before and after it is projected onto a lower-dimensional subspace.
Lemma 25. Let x be an arbitrary d-dimensional random variable with a positive definite covariance matrix Σ, whose maximum and minimum eigenvalues are λ and λ, respectively. Let V d ′ an arbitrary d -dimensional subspace with d ≤ d. Let x V d ′ denote the projection of x onto V d ′ with covariance matrix Σ V d ′ whose maximum and minimum eigenvalues are λV d ′ and λ V d ′ , respectively. Then,
λV d ′ ≤ λ and λ V d ′ ≥ λ. Proof. Let P ∈ R d×d ′
denote the projection matrix of x, i.e., P T P = I and P T x = x V d ′ , then, Σ V d ′ = P T ΣP . Hence, by definition of maximum eigenvalue, we have
λV d ′ = max v =0, v∈R d ′ v T Σ V d ′ v v T v = max v =0, v∈R d ′ v T P T ΣP v v T v = max v =0, v∈R d ′ v T P T ΣP v v T P T P v ≤ max u =0, u∈R d u T Σu u T u = λ, where u = P v. λ V d ′ = min v =0, v∈R d ′ v T Σ V d ′ v v T v = min v =0, v∈R d ′ v T P T ΣP v v T v = min v =0, v∈R d ′ v T P T ΣP v v T P T P v ≥ min u =0, u∈R d u T Σu u T u = λ,
where u = P v.
Now we are ready to prove Lemma 3.
this section cite: []

Section: References
Ref_id:b0 Title: Avrim Blum, and Keziah Naggita. The strategic perceptron Year: (2021)
Ref_id:b1 Title: Fundamental bounds on online strategic classification Year: (2023)
Ref_id:b2 Title: Strategic littlestone dimension: Improved bounds on online strategic classification Year: (2024)
Ref_id:b3 Title: The power of localization for efficiently learning linear separators with noise Year: (2015)
Ref_id:b4 Title: Active and passive learning of linear separators under logconcave distributions Year: (2013)
Ref_id:b5 Title: Convexity, classification, and risk bounds Year: (2006)
Ref_id:b6 Title: Learning with selectively labeled data from multiple decision-makers Year: (2025)
Ref_id:b7 Title: Learning strategy-aware linear classifiers Year: (2020)
Ref_id:b8 Title: Learnability gaps of strategic classification Year: (2024)
Ref_id:b9 Title: Coarse sample complexity bounds for active learning Year: (2005)
Ref_id:b10 Title: Distribution-independent pac learning of halfspaces with massart noise Year: (2019)
Ref_id:b11 Title: Learning halfspaces with massart noise under structured distributions Year: (2020)
Ref_id:b12 Title: Efficiently learning halfspaces with tsybakov noise Year: (2021)
Ref_id:b13 Title: Online learning of halfspaces with massart noise Year: (2024)
Ref_id:b14 Title: Strategic classification from revealed preferences Year: (2018)
Ref_id:b15 Title: Strategic classification Year: (2016)
Ref_id:b16 Title: Strategic apple tasting Year: (2023)
Ref_id:b17 Title:  Year: (2000)
Ref_id:b18 Title: Baums algorithm learns intersections of halfspaces with respect to log-concave distributions Year: (2009)
Ref_id:b19 Title: The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables Year: (2017)
Ref_id:b20 Title: Strategic classification made practical Year: (2021)
Ref_id:b21 Title: The geometry of logconcave functions and sampling algorithms Year: (2007)
Ref_id:b22 Title: Risk bounds for statistical learning Year: (2006)
Ref_id:b23 Title: A modern introduction to online learning Year: (2023)
Ref_id:b24 Title: The perceptron: a probabilistic model for information storage and organization in the brain Year: (1958)
Ref_id:b25 Title: Efficient algorithms in computational learning theory Year: (2001)
Ref_id:b26 Title: Online learning: Theory, algorithms, and applications Year: (2007)
Ref_id:b27 Title: Should decision-makers reveal classifiers in online strategic classification? Year: (2025)
Ref_id:b28 Title: On the power of localized perceptron for label-optimal learning of halfspaces with adversarial noise Year: (2021)
Ref_id:b29 Title: Sample-optimal pac learning of halfspaces with malicious noise Year: (2021)
Ref_id:b30 Title: Pac learning of halfspaces with malicious noise in nearly linear time Year: (2023)
Ref_id:b31 Title: Mistake, manipulation and margin guarantees in online strategic classification Year: (2024)
Ref_id:b32 Title: Pac-learning for strategic classification Year: (2023)
Ref_id:b33 Title: High-dimensional statistics: A non-asymptotic viewpoint Year: (2019)
Ref_id:b34 Title: Revisiting perceptron: Efficient and label-optimal learning of halfspaces Year: (2017)
Ref_id:b35 Title: Improved algorithms for efficient active learning halfspaces with massart and tsybakov noise Year: (2021)
Ref_id:b36 Title: Efficient active learning of sparse halfspaces with arbitrary bounded noise Year: (2020)
