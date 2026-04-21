Title: THE COVERAGE PRINCIPLE: HOW PRE-TRAINING ENABLES POST-TRAINING
Abstract: Language models demonstrate remarkable abilities when pre-trained on large text corpora and fine-tuned for specific tasks, but how and why pre-training shapes the success of the final model remains poorly understood. Notably, although pretraining success is often quantified by cross-entropy loss, cross-entropy can be a poor predictor of downstream performance. Instead, we provide a theoretical perspective on this relationship through the lens of coverage, which quantifies the probability mass the pre-trained model places on high-quality responses and which is necessary and sufficient for post-training and test-time scaling methods such as Best-of-N to succeed. Our main results develop an understanding of the coverage principle, a phenomenon whereby next-token prediction (more generally, maximum likelihood) implicitly optimizes toward a model with good coverage. In particular, we uncover a mechanism that explains the power of coverage in predicting downstream performance: coverage generalizes faster than crossentropy, avoiding spurious dependence on problem-dependent parameters such as the sequence length. We also study practical algorithmic interventions with provable benefits for improving coverage, including (i) model/checkpoint selection procedures, (ii) gradient normalization schemes, and (iii) test-time decoding strategies.Published as a conference paper at ICLR 2026 0.0 0.2 0.4 0.6 0.8 Cov2.0 0.8 0.9 1.0 1.0 Pass@ N Pass@4 Selected model Model w/ min. KL 0.0 0.1 0.2 Cov4.0 0.92 0.96

Section: INTRODUCTION
The remarkable capabilities of language models stem from a two-stage training process: (1) large-scale pre-training via next-token prediction with the cross-entropy loss (predicting what token should follow a prefix) and (2) targeted post-training-typically via reinforcement learning-to adapt the model to specific domains and tasks. Investing more compute and data into pre-training often enables post-training to produce a stronger model, but theoretical understanding of how these stages interact is limited. Indeed, despite substantial investment into scaling pre-training (Gadre et al., 2025;Sardana et al., 2024;Hoffmann et al., 2022), several works have demonstrated that starting post-training from a better next-token predictor does not ensure stronger performance on downstream tasks (Liu et al., 2022;Zeng et al., 2025;Chen et al., 2025;Lourie et al., 2025). Motivated by this disconnect, we theoretically investigate the connection between pre-training objectives and downstream success, asking: Can we precisely characterize the relationship between the next-token prediction loss and downstream performance? What metrics are most predictive of downstream success?
Motivated by the recent interest in test-time scaling, we focus our attention on post-training via Best-of-N (BoN) sampling or reinforcement learning with verifiable rewards. For a prompt x, Bestof-N draws N responses y from the model and returns the best response according to a task-specific Although KL and Cov N exhibit comparable predictive power for small N , Cov N is a better predictor for large N . Also visualized are checkpoints selected via the tournament procedure of Eq. ( 14) (marked ♢) and by minimizing KL (marked red), demonstrating that the former selects better models for Pass@N .
reward. Several prior works have demonstrated that the performance of BoN is strongly indicative of how well the model will perform after post-training via reinforcement learning (Yue et al., 2025;Wu et al., 2025).
Our starting point is the observation that cross-entropy alone cannot provide meaningful answers to the questions above; see Figure 1, which illustrates that cross-entropy can be anti-correlated with BoN performance, echoing Chen et al. (2025). Instead, we show that the missing link is the coverage profile, a refinement of cross-entropy that explicitly quantifies the model's ability to assign sufficient probability to rare but high-quality responses.
Definition 1.1 (Coverage profile). The coverage profile of a model π for a distribution π is
Cov N (π π) := P x∼µ,y∼π(•|x) π(y | x) π(y | x) ≥ N , (1
)
where N ≥ 1 is the number of Best-of-N sampling attempts.
Here, y is the full response when prompted with x, π represents the pre-training data distribution, which we presuppose covers downstream tasks of interest, and π is the pre-trained model. We prove that a good coverage profile is necessary and sufficient for Best-of-N to succeed (see Section 2, as well as Propositions F.6 and F.7). This is highlighted in Figure 1, where we find that the coverage profile is correlated with downstream performance for Best-of-N (which is exactly Pass@N ), even when cross-entropy is not. 1 Motivated by this characterization of BoN performance, we ask: When, and through what mechanism, does next-token prediction produce a model π with good coverage?
1.1 CONTRIBUTIONS We develop a theoretical understanding of the coverage principle, whereby next-token prediction implicitly optimizes toward a model with good coverage, inheriting the training corpus' coverage over tasks of interest.
Cross-entropy: Scaling laws and limitations (Section 3). We begin by deriving provable scaling laws that link cross-entropy-specifically, a certain sequence-level notion-to coverage and hence downstream performance, but show that cross-entropy can be sensitive to sequence length and other problem parameters, leading to vacuous predictions; this motivates our main results.
Next-token prediction implicitly optimizes coverage (Section 4). The first of our main theoretical results (Theorem 4.1) is a new generalization analysis for next-token prediction (more generally, maximum likelihood) that exploits the unique structure of the logarithmic loss to show that coverage can generalize faster than cross-entropy; we refer to this as the coverage principle. Concretely, our analysis shows that the coverage profile for models learned with next-token prediction (i) avoids spurious dependence on problem-dependent parameters such as sequence length (in contrast to crossentropy), and (ii) converges faster still as the tail parameter N is increased. Our analysis-which is similar in spirit to Mendelson's small ball method (Mendelson, 2014;2017)-can be viewed as giving a new, fine-grained understanding of maximum likelihood.
Stochastic gradient descent through the lens of coverage (Section 5). The preceding results apply to general model classes Π, but consider the empirical maximizer of the next-token prediction (maximum likelihood) objective, in the vein of classical techniques in learning theory. For the second of our main results, we focus on a specific model class-overparameterized autoregressive linear models (3)-but take a more realistic approach and analyze stochastic gradient descent (SGD) on the next-token prediction objective, in the one-pass ("compute-optimal") regime. We show that while SGD provably optimizes the coverage profile, it experiences suboptimal dependence on the sequence length H. We then show that gradient normalization (which is loosely connected to Adam-like updates (Bernstein & Newhouse, 2024)) provably improves coverage, removing dependence on the sequence length.
Interventions for better coverage (Section 6). Finally, we look beyond standard next-token prediction and explore families of new interventions aimed at improving coverage in theory.
(i) Test-time (Section 6.1). We show that for standard token-level SGD, a decoding strategy inspired based on test-time training (Krause et al., 2019;Sun et al., 2024;Akyürek et al., 2025) provably improves coverage.
(ii) Model/checkpoint selection (Section 6.2). For selecting the best model (or checkpoint) from a small number of candidates, we give tournament procedures that enjoy significantly better coverage profile (particularly with respect to the tail parameter N ) than naïve validation with cross-entropy.
Additional results (Appendix G). Beyond the results above, we show that: (1) MLE can find models with low coverage even in the presence of severe misspecification; (2) coverage can generalize better under additional structural properties of the model class such as convexity (Appendix G.2).
In summary, we believe that coverage offers a new perspective on the connection between pretraining objectives and downstream post-training success. Our results demonstrate that this perspective is mathematically rich and fundamental, opening the door to a deeper understanding; cf. Appendix A.
this section cite: ['b53', 'b19', 'b33', 'b37', 'b57', 'b67', 'b39', 'b40', 'b10', 'b32', 'b58', 'b1']

Section: PROBLEM SETUP
We now introduce the formal problem setup for the remainder of the paper.
Next-token prediction and maximum likelihood. We work in the following setting, which subsumes next-token prediction: X is the prompt space, Y is the response space, and π D : X → ∆(Y) is the data distribution. We are given a dataset D = {(x i , y i )} n i=1 where x i ∼ µ and y i ∼ π D (• | x i ). We consider the maximum likelihood objective
L n (π) := n i=1 log π(y i | x i ).
(
and refer to π := arg max π∈Π L n (π) as the maximum likelihood estimator for a user-specified model class Π. This is a generalization of the next-token prediction, where Y = V H is a token sequence and π(y | x) = H h=1 π(y h | x, y 1:h-1 ) is explicitly autoregressive, so that L n (π) = n i=1 H h=1 log π(y i h | x i , y i 1:h-1 ). We specialize to next-token prediction at certain points but otherwise focus on the general setting. We make the following realizability assumption throughout.
Assumption 2.1 (Realizability). The data distribution π D is realizable by some model π ∈ Π.
This formulation captures pre-training and SFT, with some caveats; see Appendix A.1.
Post-training and the coverage profile. Given a reward function r T (x, y) ∈ {0, 1} representing success at a downstream task T, the goal is to fine-tune π-through reinforcement learning or test-time scaling-to obtain near-optimal reward. We show (Propositions F.6 and F.7) that for any task-specific comparator policy π T : X → ∆(Y), Best-of-N sampling with Θ(N ) samples satisfies E x∼µ [r T (x, π T (x)) -r T (x, π BoN (x))] Cov N (π T π), so a good coverage profile for π T is sufficient for high reward. Further, while less well understood, some form of coverage is thought to be necessary for the success of post-training methods like GRPO (Yue et al., 2025).
Returning to pre-training, it is clear that there is little hope that next-token prediction will produce a model π with good coverage with respect to a downstream task unless the data distribution π D itself has reasonable coverage with respect to this task. We therefore posit that the data distribution covers such a downstream task, in the sense that it includes high-reward responses with some boundedbelow probability. Since coverage satisfies a transitivity property, it follows that coverage with respect to π D implies coverage with respect to the optimal policy for the downstream task. For example, if π D has a 10% chance of generating a correct response, and Cov N/10 (π D π) = ε, then we get 10ε error. 2 Thus, going forward, we focus on understanding when next-token prediction achieves good coverage Cov N (π D π) relative to the data distribution π D itself, and avoid concerning ourselves with specific details of the task policy π T or the specific relationship between π T and π D .
this section cite: []

Section: Autoregressive linear models.
We analyze next-token prediction and maximum likelihood for general model classes Π, but our running example throughout the paper will be the class Π of autoregressive linear models, defined by a known feature map ϕ :
X × V ⋆ → R d . For each parameter θ ∈ Θ ⊂ R d , the model π θ = (π θ ) H
h=1 is defined by π θ (y h | x, y 1:h-1 ) ∝ exp( θ, ϕ(x, y 1:h ) ).
(
In practice, autoregressive sequence models-such as those based on transformers-generate each token by sampling from a softmax distribution whose logits are given by a linear combination of learned features (Radford et al., 2019). Eq. ( 3) simplifies this by freezing the feature map, yet remains expressive enough to model complex non-Markovian dependencies, depending on the choice of features.
Assumption 2.2. We assume Θ ⊆ {θ : θ ≤ 1} is convex, and sup h,x,y 1:h ϕ(x, y 1:h ) ≤ B.
this section cite: ['b47']

Section: CROSS-ENTROPY AND COVERAGE: SCALING LAWS AND LIMITATIONS
A natural approach to understanding when next-token prediction achieves good coverage is to appeal to cross-entropy-perhaps first showing that next-token prediction achieves low cross-entropy (which is true asymptotically), and then relating cross-entropy to coverage. In this section we motivate our main results by showing that while this is possible in a weak sense, it does not yield predictive guarantees for downstream performance in the finite-sample regime.
Define the sequence-level cross-entropy for π as D CE (π D π) := E πD H h=1 log 1 π(y h |x,y 1:h-1 ) . Since E D i.i.d.
this section cite: []

Section: ∼ πD
L n (π) = -n • D CE (π D π), one expects that as we scale up compute, number of samples n, and model capacity Π, D CE (π D π) → D CE (π D π D ), or equivalently D KL (π D π) → 0, where D KL (π D π) := E πD H h=1 log πD(y h |x,y 1:h-1 ) π(y h |x,y 1:h-1 ) is the sequence-level KL divergence.
A simple scaling law for cross-entropy. We show below that if the model π has reasonable KL divergence to the data distribution, the coverage profile can be bounded:
Proposition 3.1 (KL-to-coverage; see Proposition F.1). For all N ≥ e, Cov N (π D π) ≤ D KL (πD ∥ π) log(N/e) . Combining Proposition 3.1 with Proposition F.6 and our assumption that π D has good coverage with respect to the downstream task yields a simple "scaling law" for test-time compute with BoN: Consider a task of interest with reward r T (x, y), and suppose the data distribution π D itself has constant probability of success (i.e., sampling y ∼ π D (• | x) with r T (x, y) = 1). To achieve sub-optimality ε with Best-of-N, it suffices to choose the compute budget N as
N ≈ exp D KL (π D π) ε . (4
) 0k 10k 20k 30k 40k Iterations 0 5 10 15 20 KL H=8 H=16 H=24 0k 10k 20k 30k 40k Iterations 0.00 0.25 0.50 0.75 1.00 Cov N = 16 H=8 H=16 H=24 0k 10k 20k 30k 40k Iterations 0 50 100 150 200 Ratio KL CovN=16 H=8 H=16 H=24
Figure 2: The coverage profile avoids spurious dependence on sequence length. We train models in a graph reasoning task and record their KL divergence and coverage profile, measured w.r.t. π D as we vary the problem horizon (sequence length); see Appendix D for details. Left: Convergence of KL over training for three horizons H, demonstrating that KL at convergence scales linearly in the horizon H. Center: Convergence of Cov N over training, manifesting no dependence on H at convergence. Right: Ratio of KL over Cov N , showing that Proposition 3.1 can be overly conservative.
That is, for a fixed model π and KL-divergence level D KL (π D π) ≤ D CE (π D π), Eq. ( 4) predicts that test-time compute should increase exponentially with the desired accuracy ε. 3Insufficiency of cross-entropy. At first glance, this seems to be in line with empirical test-time scaling laws (OpenAI, 2024), but there is an issue: While token-level cross-entropy has been observed to be modest in contemporary language models (Kaplan et al., 2020;Hoffmann et al., 2022;Xia et al., 2022), the sequence-level cross-entropy (and KL-divergence) generally grows with the length H of the sequence, so that Eq. ( 4) predicts exponential test-time scaling in the sequence length. Moreover, such a law cannot hold if we only assume token-level cross-entropy is bounded; see Proposition F.7.
Is this the end of the story? On the one hand, it is simple to show (Proposition F.2) that Proposition 3.1 is tight for a worst-case pair of models. Moreover, even for the autoregressive linear model in Eq. ( 3), sequence-level KL divergence scales linearly with the sequence length H, as shown in the next result.
Proposition 3.2. Fix H ∈ N and d = 1. There exists ϕ : X × V ⋆ → [-1, 1] and induced autoregressive linear class Π with parameter space Θ = [-1, 1], distribution µ over X , such that for any proper estimator π = π(D) ∈ Π, there exists data distribution π D ∈ Π such that w.p. at least 0.25, D KL (π D π) ≥ H 4n . This behavior is reflected empirically in Figure 2 for a graph reasoning task. Yet, for this task, we find (Figure 2) that in spite of large cross-entropy/KL, next-token prediction learns a model π with a good coverage profile across a range of sequence lengths and that downstream Best-of-N succeeds. Why is this happening? In light of the discussion above, it must be related to specific inductive bias of the next-token prediction objective itself.
A glimmer of hope: Case study in Bernoulli models. To see why large cross-entropy may not be a barrier to coverage, consider perhaps the simplest setting, Bernoulli models, where X = {⊥}, Y = {0, 1}, Π = {Ber(p)} p∈(0,1/2) , and π D = Ber(p ⋆ ) for some small p ⋆ ∈ (0, 1/2).
The maximum likelihood model is π = Ber( p), where p is the empirical frequency of y = 1 in the dataset. We observe that with positive probability (and constant probability if n ≤ 1 /p ⋆ ), the dataset D will only contain examples where y = 0, so that the maximum likelihood model is π = Ber(0). This implies that expected KL divergence is infinite: E[D KL (π D π)] = +∞. However, the coverage profile turns out to be well-behaved; a direct calculation shows that Cov N (π D π) ≲ log(δ -1 ) n with probability at least 1 -δ for all N ≥ 2; this gives hope that even though cross-entropy itself is infinite, maximum likelihood may actually learn a model with good coverage in the background. In what follows, we will show that this is not a fluke, but a general phenomenon.
Remark 3.1 (Missing mass). The underlying issue in both of the preceding examples is missing mass: there are responses that even a well-generalizing learner will fail to cover, and for these we may incur a large contribution to the KL-divergence. More generally, KL-divergence and crossentropy are susceptible to contributions of the scale log W max where W max = max π∈Π πD π ∞ (which could be as large as H, as in Proposition 3.2) when the model does not have enough information to generalize/extrapolate. This phenomenon is particularly pronounced when the prompt distribution is heterogeneous.
this section cite: ['b29', 'b19', 'b68']

Section: NEXT-TOKEN PREDICTION IMPLICITLY OPTIMIZES COVERAGE
We now present our main result (Theorem 4.1): due to the unique structure of the logarithmic loss, maximum likelihood can learn models with a good coverage profile even when the cross-entropy is vacuously large. Henceforth, we abbreviate Cov N (π) := Cov N (π D π). We make use of the following covering number.
Definition 4.1. For a class Π and α ≥ 0, we let N ∞ (Π, α) denote the size of the smallest cover Π ′ such that for all π ∈ Π, there exists
π ′ ∈ Π ′ such that sup x∈X ,y∈Y |log π(y | x)-log π ′ (y | x)| ≤ α.
Theorem 4.1 (Fast generalization for coverage). Fix N ≥ 8 and let c > 0 be an absolute constant. Suppose Assumption 2.1 holds. With probability at least 1-δ, the maximum likelihood estimator has
Cov N ( π) ≲ 1 log N • inf ε>0 log N ∞ (Π, ε) n + ε =: Cfine(Π,n) + log N ∞ (Π, c log N ) + log(δ -1 ) n =: Ccoarse(Π,N,n) . (5
)
Eq. ( 5) has a fine-grained term C fine (Π, n) and coarse-grained term C coarse (Π, N, n); we interpret each below.
Fine-grained term. C fine (Π, n) evaluates the covering number N ∞ (Π, ε) at a small scale ε (typically ε ≈ poly(1/n)), which matches typical bounds for conditional density estimation (e.g., Bilodeau et al. (2023)) in KL divergence; however, unlike KL-based bounds this term has no explicit dependence on sequence length H or density ratios log W max . The term is further scaled by 1/ log N , which implies that coverage enjoys faster convergence as we move further into the tail by increasing N ; this reflects the unique structure of the logarithmic loss, and may be viewed as a new form of implicit bias.
Summarizing, the fine-grained term in Eq. ( 5) witnesses the phenomenon we term the coverage principle: the coverage profile enjoys faster generalization than cross-entropy; roughly, the rate is what we would expect (via Proposition 3.1) if we could somehow control KL without paying for the sequence length H or density ratio log W max . See Appendix C for a detailed comparison to standard (asymptotic and non-asymptotic) generalization bounds for maximum likelihood based on Hellinger distance and KL-divergence.
Coarse-grained term. The coarse-grained term C coarse (Π, N, n) captures the missing mass phenomenon exemplified by the Bernoulli example in the prequel. This term is not explicitly normalized by 1/ log N (compared to the fine-grained term), but depends on the covering number N ∞ (Π, α) only at a very large scale α ≈ log N . As such, the dependence on the complexity/richness of Π in this term vanishes as we increase N .
Overall, while the guarantee in Eq. ( 5) might look surprising at first glance (particularly the coarse term, as we are not aware of any existing generalization bounds with dependence on covering numbers at such a large scale), we show in Proposition G.1 (Appendix J) that both terms are tight in general.
this section cite: ['b11']

Section: Overview of analysis.
The proof of Theorem 4.1 is given in Appendix J (with a high-level sketch in Appendix J.1). The basic idea is to interpret the condition Cov N (π) ≥ ε as a small ball-like anti-concentration condition in the vein of Mendelson (2014;2017). That is, for models π where coverage is large, the condition Cov N (π) ≥ ε witnesses a one-sided bound which implies that the empirical likelihood of π is not too large with high probability, and thus π cannot be a maximum-likelihood solution.
The coarse-grained term C coarse (Π, N, n) enters because we only need to show that the coverage profile concentrates, not the log loss itself. The fine-grained term C fine (Π, n) enters from one-sided concentration of the empirical likelihood, with the 1/ log N scaling arising from the following form of implicit bias: If an example (x i , y i ) has πD(y i |x i ) /π(y i |x i ) ≥ N , this witnesses a negative contribution of order log N to the difference L n (π) -L n (π D ).
Discussion. We emphasize that while covering numbers are a fundamental and widely used noton of capacity in statistical learning and estimation (van de Geer, 2000;Zhang, 2002;Rakhlin & Sridharan, 2012;Bilodeau et al., 2023), they are conservative from a modern generalization perspective. Nonetheless, Theorem 4.1 shows that they are sufficient to capture rich aspects of generalization for coverage, and we expect that our core analysis techniques can be combined with contemporary advances in generalization theory for overparameterized models (Belkin et al., 2019;Bartlett et al., 2020).
this section cite: ['b39', 'b40', 'b61', 'b48', 'b11', 'b9', 'b8']

Section: EXAMPLES
To build intuition, we analyze the behavior of Theorem 4.1 under a growth assumption on the covering number, then specialize to autoregressive linear models, showing how they exemplify the coverage principle.
Corollary 4.1. (i) Parametric regime: Suppose that there are parameters d ≥ 2 and C ≥ 2 such that log N ∞ (Π, α) ≤ d log(C/α) for α ∈ (0, C/2]. Then for any N ≥ 8, with probability at least
1 -δ, Cov N ( π) ≲ d[[log(C/ log N )] + + log(Cn) log N ]+log(1/δ) n .
(ii) Nonparametric regime: Suppose that there are parameters C ≥ 2 and p > 0 such that log N ∞ (Π, α) ≤ (C/α) p for α ∈ (0, C/2]. Then for any N ≥ 8 and n ≥ log 1/p N • (C/ log N ) p , with probability at least
1 -δ, Cov N ( π) ≲ 1 log N C p n 1 p+1 + log(1/δ) n .
This result shows that for sufficiently rich classes (e.g., when p > 0), the fine-grained term dominates the coarse-grained term for n sufficently large. On the other hand, for simple classes (e.g., when p = 0), the coarse-grained term can dominate the fine-grained term.
Autoregressive linear models: Low dimension. We now consider the autoregressive linear model in Eq. ( 3). When the dimension d is small, this class satisfies log N ∞ (Π, α) d log(BH/α) (corresponding to the parametric regime in Corollary 4.1), and so, coverage generalizes in a (nearly) horizon-independent fashion, in stark contrast to the cross-entropy lower bound in Proposition 3.2. The only drawback (which is fundamental) is that since the class has low capacity, the coarse-grained term dominates for most parameter regimes, and the improvement as N scales is quite modest.
Autoregressive linear models: High dimension. As a more interesting example, we next look at the behavior of next-token prediction for autoregressive linear models in an "overparameterized" regime where the dimension d is arbitrarily large (Zhang, 2002;Neyshabur et al., 2015;Bartlett et al., 2017). Here, we control the richness of the class Π by the norm parameter B. In this regime, it turns out that in the worst-case, the capacity log N ∞ (Π, α) scales polynomially in H. To address, this we prove a refined version of Theorem 4.1 that adapts to the variance in the data distribution π D , avoiding explicit dependence on sequence length.
Define the inherent variance for the data distribution as
σ 2 ⋆ := E πD H h=1 ϕ(x, y 1:h ) -ϕ πD (x, y 1:h-1 ) 2 , (6
)
where ϕ πD (x, y 1:h-1 ) := E y h ∼πD(•|x,y 1:h-1 ) [ϕ(x, y 1:h )] is the average feature vector given the prefix (x, y 1:h-1 ). We can interpret the inherent variance σ 2 ⋆ as a notion of effective sequence length; it captures the number tokens that are "pivotal" in the sense that they have high variation conditioned on the prefix; the name reflects a noted phenomenon in language modeling that most tokens are neardeterministic and easy to predict given their prefix, with only a few having high entropy (Abdin et al., 2024). Thus, while σ 2 ⋆ can be as large as B 2 H in the worst case, we expect it to be smaller in general. Theorem 4.2 (Overparameterized autoregressive linear models). Consider the autoregressive linear model (3), and suppose Assumptions 2.1 and 2.2 hold. For any N ≥ 2, next-token prediction achieves
E[Cov N ( π)] ≲ σ 2 ⋆ n•log N + B 2 n . (7
)
Similar to Theorem 4.1, the first term in Eq. ( 7) can be viewed as "fine-grained" and the second term as "coarse-grained"; the former is typically larger, but decreases with the tail parameter N , while the latter does not decrease with N but is typically smaller to begin with. We prove (details Published as a conference paper at ICLR 2026 in Proposition K.1) that this result is tight in the sense that if σ 2 ⋆ H, n ≥ H is indeed necessary to achieve good non-trivial coverage in the overparameterized regime.
We view the introduction of the inherent variance σ 2 ⋆ as an instance-dependent notion of complexity for autoregressive models to be a non-trivial conceptual contribution, which may find broader use.
this section cite: ['b44', 'b7', 'b0']

Section: STOCHASTIC GRADIENT DESCENT THROUGH THE LENS OF COVERAGE
The coverage-based generalization guarantees for next-token prediction in the prequel apply to general model classes Π, but consider the empirical maximizer π = arg max π∈Π L n (π) of the next-token prediction (maximum likelihood) objective, in the vein of classical techniques in learning theory. For our second set of main results, we focus on autoregressive linear models (3) but take a more realistic approach and analyze stochastic gradient descent (SGD) in the single-pass regime. This setup is motivated by contemporary ("compute-optimal") language model training, which typically uses one or fewer passes over the training corpus (Kaplan et al., 2020;Hoffmann et al., 2022).
this section cite: ['b29', 'b19']

Section: STOCHASTIC GRADIENT DESCENT HAS SUBOPTIMAL COVERAGE
For the next-token prediction objective, single-pass stochastic gradient descent (SGD) takes the formfoot_3
θ t+1 ← Proj Θ (θ t + η∇ log π θ t (y t | x t )),(8)
for x t ∼ µ and y t ∼ π D (• | x t ), where η > 0 is the learning rate. As the next-token prediction loss L(θ) := E πD [-log π θ (y | x)] is convex under the parameterization (3), we can show that SGD converges to π D in KL divergence. This implies a coverage bound, albeit a suboptimal one. Proposition 5.1 (SGD for autoregressive linear models). Upper bound: Suppose Assumptions 2.1 and 2.2 hold. As long as η ≤ 1 2HB 2 , it holds that E 1 T T t=1 D KL (π D π θ t ) ≤ 4 ηT + 2ησ 2 ⋆ . Choosing η to minimize this bound gives
E 1 T T t=1 Cov N (π θ t ) ≲ 1 log N • σ 2 ⋆ T + B 2 H T . (9
)
Lower bound: Suppose that B ≥ c • log 2 (T H). Then there exists an autoregressive linear class Π such that for any constant step size η > 0, there exists an instance π D ∈ Π with σ ⋆ ≤ 1 such that with probability at least 0.5, the SGD iterates satisfy Cov N (π D π θ t ) ≥ c • min H T log N , 1 for any t ∈ [T ]. The coverage bound in Eq. ( 9) (which follows by passing from KL to coverage through Proposition 3.1) is similar to Theorem 4.2, except that the second term B 2 H T has an unfortunate dependence on the sequence length H. The lower bound shows that this dependence is tight, and SGD can indeed experience poor coverage. This failure of SGD is related to heterogeneity across prompts: there are some prompts for which the effective scale of the gradient in Eq. ( 8) grows with H, leading to divergence unless we use a small learning rate η ≲ 1 HB . Yet for other prompts, the effective gradient range is small, leading to slow convergence (on the order of Ω(H) steps) unless η 1 HB . Remark 5.1 (Sequence-level SGD). The update in Eq. ( 8) can be interpreted as a "sequencelevel" form of SGD, since we perform a single gradient step for each full sequence y t (note that
∇ log π θ t (y t | x t ) = H h=1 ∇ log π θ t (y t h | x t , y t 1:h-1 )).
We view this as a model for what is done in practice, whereby one performs SGD on sequences of tokens spanning some fixed context window. While this context window may be shorter than the full training example (e.g., a long article), understanding the implications of a limited context window is beyond the scope of this work.
this section cite: []

Section: GRADIENT NORMALIZATION IMPROVES COVERAGE
To address the suboptimality of SGD, we consider gradient normalization as a simple intervention. For a mini-batch D = {(x i , y i )} K i=1 of K samples from π D , define the batch stochastic gradient as g(θ; D) = 1 |D| (x,y)∈D ∇ log π θ (y | x). We consider the following normalized SGD update:
θ t+1 ← Proj Θ θ t + η • g(θ t ;D t ) λ+∥ g(θ t ;D t )∥ ;(10)
here D t is a mini-batch with K fresh samples drawn i.i.d. from π D , and λ > 0 is a regularization parameter for numerical stability. We show that this update achieves a horizon-independent coverage bound.
Theorem 5.1. Suppose Assumption 2.1 and Assumption 2.2 hold. Let T, K ≥ 1, N ≥ 3 be given. For an appropriate choice of η, λ > 0, the normalized SGD update (10) achieves the following bound:
E 1 T T t=1 Cov N (π θ t ) ≲ σ 2 ⋆ T •log N + B 2 T + B K•log N . (11
)
To achieve E[Cov N ( π)] ≤ ε for a target level ε > 0, it suffices to choose
T = O σ 2 ⋆ ε 2 log N + B 2 ε , K = O B ε log N + 1 , giving total sample complexity n = T K = O σ 2 ⋆ B ε 3 log 2 N + B 3 +σ 2 ⋆ ε 2 log N + B 2 ε .
Theorem 5.1 shows that gradient normalization achieves horizon-independent coverage with a qualitatively similar rate to the guarantee for next-token prediction in Theorem 4.2: To achieve coverage ε, both rates scale as poly σ 2 ⋆ log N , B, ε -1 , though the dependence on ε for Theorem 5.1 is worse. We view this as another instance of the coverage principle, as the rate achieved by gradient normalization goes beyond what can be achieved by passing through KL divergence. We emphasize that minibatching alone is not enough to achieve this result; rather, minibatching is necessary to avoid excessive bias once we introduce gradient normalization.
As a remark, the normalized SG update in (10) is closely related to SignSGD (Balles & Hennig, 2018) and Adam (Kingma & Ba, 2015) as shown by Bernstein & Newhouse (2024). We believe that similar coverage guarantees could potentially be shown for these methods using our techniques.
Distillation. As an additional result, we show (Theorem G.2 in Appendix G.4) that for a distillation setting, where π D corresponds to a teacher model and we have access to its per-token logits, we can derive an improved gradient normalization scheme that fully closes the gap with Theorem 4.2.
this section cite: ['b4', 'b10']

Section: INTERVENTIONS FOR BETTER COVERAGE
In this section, we develop new interventions that improve coverage (and downstream performance) beyond the conventional algorithms analyzed in Sections 4 and 5. We view these results as promising proofs of concept for further research into interventions driven by coverage.
this section cite: []

Section: IMPROVING COVERAGE AT TEST TIME
In this section, we show that a modified decoding strategy based on test-time training (or, dynamic evaluation) (Mikolov et al., 2010;Krause et al., 2018;2019;Sun et al., 2024;Akyürek et al., 2025) leads to improved coverage when combined with token-level SGD.
We focus on autoregressive linear models, but depart from Eq. ( 8) by learning models with a tokenlevel SGD update, defined as
θ t,h+1 = Proj Θ θ t,h + η∇ log π θ t,h (y t h | x t , y t 1:h-1 ) , for h = 0, • • • , H -1,(12)
and θ t+1 ≡ θ t+1,0 := θ t,H for t ∈ [T ], and where (x t , y t 1:H ) ∼ π D . We will show that-when combined with a test-time training-like update that performs token-level gradient updates during test time-the updates in Eq. ( 12) can circumvent the H-dependence in the lower bound of Proposition 5.1.
Concretely, we consider a distribution π TTT θ : X → ∆(Y H ) formally introduced in Appendix K.6, which can be interpreted as an augmented version of the autoregressive linear model π θ that uses test-time training to sample. Given a prompt x, we first sample
y 1 ∼ π θ (• | x), then perform a gradient step θ ′ ← Proj Θ (θ + η∇ log π θ (y 1 | x))
to increase the probability of the token we just sampled. We then sample y 2 ∼ π θ ′ (• | x, y 1 ), update θ ′′ ← Proj Θ (θ ′ + η∇ log π θ ′ (y 2 | x, y 1 )), and so on. Once the full sequence y 1:H is sampled, we reset back to θ (so that we can process the next test-time example). This bears similarity to many test-time training methods in the literature, and specifically coincides with the method used in Krause et al. (2019); Rannen-Triki et al. (2024). We show that when augmented with this test-time sampling scheme, token-level SGD achieves a horizon-independent coverage bound that matches and even slightly improves upon the bound for next-token prediction in Theorem 4.2.
Theorem 6.1 (Token-level SGD with test-time training). Suppose Assumption 2.1 and Assumption 2.2 hold. For a suitably chosen parameter η > 0, token-level SGD (12) achieves E 1 T T t=1 D KL (π D π TTT θ t ) ≲ σ 2 ⋆ T + B 2 T , and thus E 1 T T t=1 Cov N (π TTT θ t ) ≲ 1 log N σ 2 ⋆ T + B 2 T .
This improves Theorem 4.2 by a factor of 1/ √ log N on the leading term and a factor of 1/ log N on the second term. Furthermore, the algorithm bypasses the lower bound on KL divergence for proper methods in Proposition 3.2, demonstrating a provable benefit of being improper.
this section cite: ['b41', 'b31', 'b63', 'b58', 'b1', 'b32', 'b49']

Section: SELECTING FOR COVERAGE
We last consider the problem of selecting a model (e.g., checkpoint) from a small number of candidates to achieve the best coverage. We introduce a tournament-like procedure that improves upon maximum likelihood in that it removes the requirement that π D ∈ Π; it is guaranteed to find a model in the class with good coverage if one exists, even if π D itself is not in the class. As an algorithmic intervention, we envision using this procedure to select a single training checkpoint or hyperparameter configuration to use for RL fine-tuning or test-time scaling. Indeed, as demonstrated in Figure 1, using cross-entropy as a selection criterion-as is standard-may result in poor coverage, while these procedures can select better checkpoints. Our results here concern the general setting in Section 2, and are not restricted to autoregressive linear models.
While their main motivation is model/checkpoint selection with a finite class Π, both estimators can also be applied to general, infinite classes Π. In this case, they improve upon the coverage achieved by the maximum likelihood estimator in Theorem 4.1, even in the well-specified case where π D ∈ Π; informally, the tournament estimators allow us to remove the fine-grained term in Theorem 4.1, leaving only a coarse-grained term.
A simple tournament for maximizing coverage. Given a dataset D = {(x i , y i )} i∈[n] , define
Cov N (π ′ π) := 1 n i ∈ [n] : π ′ (y i |x i ) π(y i |x i ) ≥ N ,(13)
which can be interpreted as an empirical version of the coverage profile Cov N (π ′ π) in Eq. ( 1) when π ′ = π D (see Lemma J.2). For N ≥ 1, we consider the estimator
π := arg min π∈Π max π ′ ∈Π Cov N (π ′ π). (14
)
Informally, this estimator chooses the model π that minimizes the maximum coverage against any other model π ′ in the class Π. When Π is small, we can implement this tournament by simply evaluating the empirical coverage in Eq. ( 13) for each pair. The main guarantee for this estimator is as follows.
Theorem 6.2. Let N ≥ 1 be given. Then, for any a ∈ [0, 1], with probability at least 1 -δ, the tournament estimator (14) achieves
Cov N 1+a ( π) ≲ min π∈Π Cov N a (π) + 1 N 1-a + log(|Π|/δ) n . (15
)
This shows that the tournament achieves a coverage profile nearly as good as the best-in-class, except for a small polynomial blow up, in that we bound the coverage at level N 1+a in terms of the coverage for the best-in-class at level N a .
Infinite class and improving the tournament. Eq. ( 14) can also be applied to general, infinite classes Π. In this case, it turns out that it improves upon the coverage achieved by the maximum likelihood estimator in Theorem 4.1 (see Theorem 6.2 ′ ). Furthermore, in Appendix G.5, we describe an improved tournament estimator that is able to remove the 1/N
1-a term from Theorem 6.2, thereby achieving nontrivial guarantees even when the coverage parameter N is constant. DISCUSSION AND FUTURE WORK See Appendix A for discussion and open problems, and Appendix G for additional results.
this section cite: []

Section: REPRODUCIBILITY STATEMENT
We provide full proofs for all theoretical results in the appendix. Appendix D includes extensive experiment setup and implementation details for all empirical results. The source code is included in the supplementary material, along with the plotting scripts and data to reproduce Figure 1 and Figure 2.
Fan Chen, Dylan J Foster, Yanjun Han, Jian Qian, Alexander Rakhlin, and Yunbei Xu. Assouad, fano, and le cam with interaction: A unifying lower bound framework and characterization for bandit learnability. Advances in Neural Information Processing Systems, 37:75585-75641, 2024a.
Feng Chen, Allan Raventos, Nan Cheng, Surya Ganguli, and Shaul Druckmann. Rethinking finetuning when scaling test-time compute: Limiting confidence improves mathematical reasoning. arXiv preprint arXiv:2502.07154, 2025.
Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning.
In International conference on machine learning, pp. 1042-1051. PMLR, 2019.
Yangyi Chen, Binxuan Huang, Yifan Gao, Zhengyang Wang, Jingfeng Yang, and Heng Ji. Scaling laws for predicting downstream performance in llms.
Transactions on Machine Learning Research, 2024b. Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=dYur3yabMj. Rick Durrett. Probability: theory and examples, volume 49. Cambridge university press, 2019. Amir-massoud Farahmand, Csaba Szepesvári, and Rémi Munos. Error propagation for approximate policy and value iteration. Advances in Neural Information Processing Systems, 2010. Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. Talk like a graph: Encoding graphs for large language models. In The Twelfth International Conference on Learning Representations, 2024. Marc Finzi, Sanyam Kapoor, Diego Granziol, Anming Gu, Christopher De Sa, J Zico Kolter, and Andrew Gordon Wilson. Compute-optimal llms provably generalize better with scale. arXiv preprint arXiv:2504.15208, 2025. Dylan J Foster and Alexander Rakhlin. Foundations of reinforcement learning and interactive decision making. arXiv:2312.16730, 2023. Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv:2112.13487, 2021. Dylan J Foster, Akshay Krishnamurthy, David Simchi-Levi, and Yunzong Xu. Offline reinforcement learning: Fundamental barriers for value function approximation. In Conference on Learning Theory, pp. 3489-3489. PMLR, 2022. Dylan J Foster, Adam Block, and Dipendra Misra. Is behavior cloning all you need? understanding horizon in imitation learning. arXiv preprint arXiv:2407.15007, 2024. Dylan J Foster, Zakaria Mhammedi, and Dhruv Rohatgi. Is a good foundation necessary for efficient reinforcement learning? the computational role of the base model in exploration. Conference on Learning Theory (COLT), 2025. Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, et al. Language models scale reliably with over-training and on downstream tasks. In The Thirteenth International Conference on Learning Representations, 2024. Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, Rui Xin, Marianna Nezhurina, Igor Vasiljevic, Luca Soldaini, Jenia Jitsev, Alex Dimakis, Gabriel Ilharco, Pang Wei Koh, Shuran Song, Thomas Kollar, Yair Carmon, Achal Dave, Reinhard Heckel, Niklas Muennighoff, and Ludwig Schmidt. Language models scale reliably with over-training and on downstream tasks. In The Thirteenth International Conference on Learning Representations, 2025. URL https:  //openreview.net/forum?id=iZeQBqJamf. Tengyang Xie and Nan Jiang. Q* approximation schemes for batch reinforcement learning: A theoretical comparison. In Conference on Uncertainty in Artificial Intelligence, 2020. Yuhong Yang and Andrew R Barron. An asymptotic property of model selection criteria. IEEE Transactions on Information Theory, 44(1):95-116, 1998. Gilad Yehudai, Noah Amsel, and Joan Bruna. Compositional reasoning with transformers, rnns, and chain of thought. arXiv preprint arXiv:2503.01544, 2025. Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025. Hansi Zeng, Kai Hui, Honglei Zhuang, Zhen Qin, Zhenrui Yue, Hamed Zamani, and Dana Alon. Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?, 2025. Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In International Conference on Learning Representations (ICLR), 2017. Tong Zhang. Covering number bounds of certain regularized linear function classes. Journal of Machine Learning Research, 2(Mar):527-550, 2002. Tong Zhang. From ϵ-entropy to KL-entropy: Analysis of minimum information complexity density estimation. The Annals of Statistics, 2006. CONTENTS OF APPENDIX I Additional Discussion and Results A Discussion and Future Work A.1 Simplifications in the Problem Formulation . . . . . . . . . . . . . . . . . . . . . A.2 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B Related Work C Comparison to Classical Generalization Bounds for MLE D Experiments D.1 Graph Reasoning Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . D.2 Experiment Details for Figure 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . D.3 Experiment Details for Figure 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . E Properties of the Coverage Profile F Supporting Results F.1 Properties of the Coverage Profile . . . . . . . . . . . . . . . . . . . . . . . . . . F.2 Analysis of Best-of-N Sampling under a Good Coverage Profile . . . . . . . . . . F.3 Properties of Maximum Likelihood . . . . . . . . . . . . . . . . . . . . . . . . . . F.4 Autoregressive Models: Coverage and Stopped KL-Divergence . . . . . . . . . . . G Additional Results G.1 Tightness of Theorem 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . G.2 Maximum Likelihood: Better Coverage for Convex Classes . . . . . . . . . . . . . G.3 Lower Bound for Maximum Likelihood under Misspecification . . . . . . . . . . . G.4 Stochastic Gradient Descent: Improved Gradient Normalization for Distillation . . G.5 An improved tournament via on-policy generation . . . . . . . . . . . . . . . . . . II Proofs H Technical Tools H.1 Concentration Inequalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . H.2 Information-Theoretic Inequalities . . . . . . . . . . . . . . . . . . . . . . . . . . I Proofs from Section 3 J Proofs from Section 4 J.1 Proof Sketch for Theorem 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . J.2 Proof of Theorem 4.1 (Coverage for MLE) . . . . . . . . . . . . . . . . . . . . . . J.3 Proof of Theorem G.1 (Coverage for MLE with Convex Classes) . . . . . . . . . . J.4 Proofs for Supporting Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . K Proofs for Autoregressive Linear Models K.1 Organization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . K.2 Proof of Theorem 4.2 (Coverage for MLE for Autoregressive Linear Models) . . . K.3 Proof of Proposition 5.1 (Vanilla SGD: Coverage Upper Bound) . . . . . . . . . . K.4 Proof of Proposition 5.1 (Vanilla SGD: Coverage Lower Bound) . . . . . . . . . . K.5 Proof of Theorem 5.1 (Coverage for Normalized SGD) . . . . . . . . . . . . . . . K.6 Proof of Theorem 6.1 (Test-Time Training) . . . . . . . . . . . . . . . . . . . . . K.7 Proof of Theorem G.2 (Gradient Normalization for Distillation) . . . . . . . . . . K.8 Necessity of Variance Dependence in High Dimension . . . . . . . . . . . . . . . L Proofs from Section 6 L.1 Proof of Theorem 6.2 (Simple Tournament) . . . . . . . . . . . . . . . . . . . . .
this section cite: []

Section: Part I Additional Discussion and Results

this section cite: []

Section: A DISCUSSION AND FUTURE WORK
Our work, through the lens of coverage, takes a first step toward clarifying the mechanisms through which pre-training with next-token prediction leads to models for which post-training is effective.
this section cite: []

Section: A.1 SIMPLIFICATIONS IN THE PROBLEM FORMULATION
In the course of the paper we have made various simplifying assumptions. Some of these can be relaxed in a straightforward fashion, while others are more fundamental.
• In language model pre-training, the pre-training corpus consists of sequences y with varying lengths H, and does not typically split examples into prompts and responses. Our formulation in Section 2 is a simplification (one that is closer in spirit to supervised fine-tuning), but we expect that the insights derived here can extend to the general setting.
• Much of our analysis focuses on the realizable/well-specified setting where π D ∈ Π. We give evidence in Appendix G that the coverage profile is more tolerant to misspecification than KLdivergence, but we leave a deeper investigation for future work.
• Our treatment assumes the distribution over prompts µ is the same for pre-training and posttraining. This is straightforward to relax at the cost of introducing an additional coverage or distribution shift coefficient to handle the mismatch between the two distributions.
• We show that a good coverage profile is necessary for BoN to succeed on downstream tasks. While there is ample evidence current RL techniques can fail in the absence of coverage (Yue et al., 2025;Gandhi et al., 2025;Wu et al., 2025), it is not clear what the minimal conditions required for RL are.
• Our results focus on coverage at the sequence level. For reasoning tasks, it is natural to explicitly factorize the response y = (y cot , y ans ) into a chain-of-thought (reasoning trajectory) component y cot and an answer component y ans . For this setting, a weaker notion coverage is the following answer-level coverage profile: Cov ans N (π D π) := P πD πD(yans|x) π(yans|x) ≥ N . The answer-level coverage profile is sufficient for downstream BoN success for tasks where it is only important to produce the right answer, not a correct reasoning trace. We have Cov ans N (π D π) ≤ Cov N (π D π), but the former can be strictly smaller in general.
this section cite: ['b57', 'b15', 'b67']

Section: A.2 FUTURE WORK Our results open several new directions for future research.
Interventions for coverage. There is much to be done in understanding and improving existing algorithms such as optimizers through the lens of coverage. Our results in Section 6 show initial promise for using coverage to guide design of optimizers and model selection schemes, but the algorithm design space remains opaque, and there may be significant room for futher improvement. More ambitiously, one could imagine re-structuring the entire language modeling pipeline itself around coverage.
this section cite: []

Section: Semantic coverage.
The notion of coverage we focus on, the coverage profile, is mathematically convenient but may be conservative in regard to downstream performance, since it only depends on the model through its predicted probabilities. An important direction for future work is to understand pre-training and post-training through fine-grained "semantic" notions of coverage that more explicitly account for the representations learned by next-token prediction.
this section cite: []

Section: B RELATED WORK
Related empirical observations. On the empirical side, our results are connected to a line of work that studies scaling laws for zero-shot downstream performance based on pre-training metrics such as cross-entropy (Gadre et al., 2024;Huang et al., 2024;Chen et al., 2024b;Sardana et al., 2024). Several empirical works have also investigated how specific capabilities scale with additional pre-training, including machine translation (Ghorbani et al., 2022), knowledge capacity and memorization (Allen-Zhu & Li, 2025;Lu et al., 2024), and multi-hop reasoning (Wang et al., 2025). Our findings are consistent with Liu et al. (2022 2025), who observe that cross-entropy is not always sufficient for predicting downstream performance, and in some cases can be anti-correlated.
Perhaps most closely related, Chen et al. (2025) show empirically that decreasing cross-entropy in pre-training does not necessarily lead to better Pass@N performance, and that Pass@N can even degrade as pre-training proceeds-a finding similar to Figure 1. 5 Our results can be viewed as placing their findings on stronger theoretical footing; conversely, their empirical results provide strong motivation for our theoretical treatment. Chen et al. (2025) also study a modification to the maximum likelihood objective aimed at improving coverage (in the spirit of Section 6); their approach targets the structure of outcome-based reward, whereas our notion of coverage profile and results are agnostic to the downstream task/reward structure.
We mention in passing some additional works. Chu et al. (2025) explored the different (synergistic) roles that supervised fine-tuning (SFT) and RL play in language model development, and subsequent work observed that the best checkpoint to start RL from can sometimes be in the middle of SFT training (Jin et al., 2025). Bansal et al. (2025) empirically identified the coverage of teacher-generated synthetic data as an important indicator for how effective distillation can be for reasoning tasks. Several papers have also investigated empirical tradeoffs between model size and reasoning performance under best-of-N sampling (Snell et al., 2025;Brown et al., 2025).
Coverage in post-training. Coverage metrics similar to coverage profile play a central role in theoretical literature on post-training and test-time algorithms (Huang et al., 2025a;b;c;Foster et al., 2025;Liu et al., 2024;Song et al., 2024;Gao et al., 2024;Liu et al., 2024;Ji et al., 2024), which analyze algorithms under the assumption that the base model has good coverage; our work can be viewed as providing theoretical motivation for this assumption. Formally, one can use Markov's inequality to bound the coverage profile by the L p -like coverage quantities considered in these works.
Various notions of coverage similar to coverage profile have also appeared in the more classical literature on offline reinforcement learning (Farahmand et al., 2010;Chen & Jiang, 2019;Xie & Jiang, 2020;Jin et al., 2021;Foster et al., 2022;Jiang & Xie, 2024); here coverage is typically used to quantify the quality of an offline dataset rather than a model/policy itself.
Generalization in deep learning. Understanding the generalization behavior of deep learning models has been a central focus of the theory community for the last decade (Neyshabur et al., 2015;Zhang et al., 2017;Bartlett et al., 2017;Jacot et al., 2018;Belkin et al., 2019;Nagarajan & Kolter, 2019;Bartlett et al., 2020;Bartlett & Montanari, 2021). Our approach is somewhat complementary, in the sense that it focuses on the specific objective of next-token prediction with the logarithmic loss, and aims to understand when minimizing this loss leads to generalization for an alternative objective, coverage profile. We expect that our techniques can be combined with these contemporary generalization results to provide a more refined understanding of generalization for the coverage profile with deep models.
From this line of work, perhaps most closely related are Lotfi et al. (2023;2024);Finzi et al. (2025), which aim to provide non-vacuous generalization bounds for the cross-entropy loss itself for autoregressive models.
Analysis of maximum likelihood. Our theoretical results are closely related to a classical line of work in statistics (Wong & Shen, 1995;van de Geer, 2000;Zhang, 2006), which shows that maximum likelihood can converge to the true model in Hellinger distance (or other Renyi divergences) under minimal assumptions, even when KL divergence is poorly behaved (large or infinite); see Appendix C below for a detailed comparison. Our results in Section 4 are similar in spirit, but provide a more fine-grained perspective, showing that the coverage profile can converge even faster than these results might suggest, particularly as one ventures further into the tail. Our analysis has some conceptual similarity to the small ball method of Mendelson (2014;2017), which we elaborate on in Appendix J.1.
Our techniques are also related to recent work of Foster et al. (2024); Rohatgi et al. (2025), which specializes the general techniques above to autoregressive models (e.g., under Hellinger distance).
this section cite: ['b23', 'b53', 'b17', 'b2', 'b38', 'b65', 'b27', 'b5', 'b54', 'b13', 'b29', 'b34', 'b55', 'b16', 'b34', 'b25', 'b28', 'b26', 'b44', 'b7', 'b24', 'b9', 'b42', 'b8', 'b6', 'b35', 'b45', 'b66', 'b61', 'b39', 'b40', 'b50']

Section: C COMPARISON TO CLASSICAL GENERALIZATION BOUNDS FOR MLE
In this section we briefly compare our main coverage-based generalization bound for maximum likelihood to classical generalization bounds for maximum likelihood based on Hellinger distance and KL-divergence.
Comparison to KL concentration. For general model classes Π, the best non-asymptotic KLbased generalization bound we are aware of is Proposition F.9 (Appendix F), which scales as roughly
D KL (π D π) ≲ log W max • C fine (Π, n)
under the assumption that all π ∈ Π obey a sequence-level density ratio bound πD π ∞ ≤ W max . Note that for the autoregressive linear class, we have log W max = BH, matching Proposition 3.2. Combining such a guarantee with Proposition 3.1 gives a coverage bound of roughly
Cov N ( π) ≲ log W max log N • C fine (Π, n);
this is rather uninteresting since Cov N ( π) = 0 for N ≥ W max ; in other words, we do not get a meaningful improvement as we scale N .
Comparison to Hellinger concentration. The Hellinger distance is a standard metric of distribution estimation, defined via
D 2 H (P, Q) = 1 2 ( √ P - √ Q) 2 .
The guarantees of maximum likelihood estimation (Wong & Shen, 1995;Van der Vaart, 2000;Zhang, 2006) also imply convergence in Hellinger distance. For general model classes Π, the best non-asymptotic Hellinger-based generalization bound we are aware of is Proposition F.8 (Appendix F), which scales as roughly
D 2 H (π D , π) ≲ C fine (Π, n)
Combining such a guarantee with Proposition 3.1 gives a coverage bound of Cov N ( π) ≲ C fine (Π, n) for all N ≥ 2. Compare to the KL-based result above, this result gives a non-trivial bound on coverage when N is constant (comparable to Theorem 4.1), but the issue is that it gives no further improvement as we scale N .
Asymptotic bounds for maximum likelihood. We also note that the classical theory of maximum likelihood (e.g., Van der Vaart (2000)) provides asymptotic convergence rates for d-dimensional parametric classes Π which have the following form:
D KL (π D π) ≲ d n ≲ C fine (Π, n), as n → +∞.
While this upper bound does not scale with log W max , it can only be attained with n ≥ n 0 for a sufficiently large burn-in cost n 0 , which itself will typically scale with log W max or similar problemdependent parameters; see, e.g., Spokoiny (2012) for non-asymptotic bounds of this type. Our lower bounds (e.g., Proposition 3.2) imply that there is no hope of removing such a burn-in cost in general.
this section cite: ['b66', 'b62', 'b62', 'b56']

Section: D EXPERIMENTS
This section presents details for the experiments in Figure 1 and D.1 GRAPH REASONING TASK We evaluate our theoretical predictions using experiments in graph reasoning tasks, in which transformer models are trained to find paths between source and target nodes in graphs. Both graph reasoning benchmarks and synthetic datasets have seen increasing use as abstractions for reasoning problems and for probing language modeling phenomena (Sanford et al., 2024;Nagarajan et al., 2025;Saparov et al., 2025;Bachmann & Nagarajan, 2024;Yehudai et al., 2025;Taylor et al., 2024;Wang et al., 2023;Fatemi et al., 2024;Tang et al., 2025). These tasks provide minimal abstractions of core reasoning problems, yet are expressive enough to capture pre-training and fine-tuning phenomena. They also offer flexibility in problem structure and difficulty: by specifying different graph topologies and path depths, we can modulate difficulty and expose sources of hardness.
D.1.1 GRAPH SEARCH TASK DESCRIPTION The graph search tasks for all of our experiments in Appendix D.2 and Appendix D.3 share the same high-level components, and are comprised of • Problem instances. A set of graph search problems G that map bijectively to a set of prompts X .
• Data distribution. A distribution over the prompts µ ∈ ∆(X ). and a data collection policy π D : X → ∆(Y)
• Dataset. The training dataset D = {(x, y)} is comprised of prompts x ∼ µ and y ∼ π D (x).
Next, we describe the general details of the graph search task common to all experiments, as well as how the graph search task is converted to a sequence modeling problem for language models.
Graph problem instances. Each graph search problem in G ∈ G is specified by a tuple G = (G, s, t).
Here, G = (V, E) is a graph structure with nodes (or vertices) V and edges E = {(u, v) : u, v ∈ V, u = v}, s ∈ V is the source node, and t is the target node. The nodes V are represented as integers, so that V ⊂ [m] for some fixed m ∈ Z.
For all experiments, we utilize a layered directed acyclic graph (layered DAG) for each graph structure (G, _, _) ∈ G, in which nodes are organized into sequential layers with edges flowing only from one layer to the next. The graph G = (V, E) has L + 2 layers with disjoint sets of nodes, so that V = i∈{1,...,L+2} V i where V i denotes the set of nodes in layer i. The first and last layers contain only the source and target nodes, respectively, so that V 1 = {s} and V L+2 = {t}.
The edge structure E connects only a subset of nodes in each layer to the next. We refer to this subset in each layer i ∈ {1, . . . , L + 2} as its passable nodes V i * ⊆ V i , or the set of nodes with non-zero out-degree,
V i * = v ∈ V i : deg + (v) > 0 .
The passable nodes in layer i are fully connected to all nodes in the next layer, that is,
E = (u, v) : u ∈ V i * , v ∈ V i+1 , i ∈ {1, . . . , L + 1} . The remaining nodes in V i \ V i
* have no outgoing edges, and are thus nodes the model must learn to avoid in order to output valid paths.
Data distribution. The model's task is to imitate the data collection policy π D , which samples only a subset of the (potentially many) valid paths from source to target based on global features of the graph. A valid path from s to t is a list of nodes of the form (s, v 2 , . . . , v L+1 , t) where v i ∈ V i * for each i ∈ {2, . . . , L + 1}; that is, the path must start with the source node s and end with the target node t, and each intermediate node in the path must be a passable node from its respective layer. A graph may have many valid paths, specifically, i∈[L+2] |V i * | many. In order for a model to learn valid paths, learning a simple local rule suffices: it can output any node in the next layer with > 0 out-degree, which is representable by a fairly shallow transformer.
However, imitating π D is a much harder problem. The data collection policy π D samples a subset of these valid paths determined via global rules, or complex functions computed over features of the entire graph that go beyond those required for path validity alone. By varying the complexity of these rules, we can modulate both the difficulty and the nature of the learning problem. This structure naturally maps onto reasoning tasks: following passable nodes corresponds to taking "reasoning steps" that make progress towards the solution, while selecting non-passable nodes corresponds to reasoning errors that lead to invalid solutions. Moreover, when π D selects among valid paths via such global rules, this corresponds to learning high-quality solutions that accurately reflect desired properties for the problem.
this section cite: ['b51', 'b43', 'b52', 'b3', 'b60', 'b64', 'b51', 'b59']

Section: Dataset.
Recall that the model learns to imitate π D from a dataset D = {(x, y)}, where each prompt x corresponds to a graph search problem G = (G, s, t) ∈ G, and each response y ∼ π D (• | x) is an expert response, formatted as follows.
We convert a given graph search problem G = (G, s, t) ∈ G with graph structure G = (V, E) to a prompt x by concatenating the edge list E, the source node s, and the target node t, formatted as
x : u_1 v_1 | u_2 v_2 | . . . | u_k v_k / s t =
where (u i , v i ) ∈ [m] 2 are the vertices of the i-th edge in the edge set E. For formatting, the special character | separates two edges, the character / separates the adjacency list from the source and target nodes, while the character = marks the end of the prompt.
As an example, for edge set E = {(10, 23), (86, 47), . . . , (45, 32)}, the prompt is x : 10 23 | 86 47 | . . . | 45 32 / 10 45 = .
Next, each response y encodes the path from the source to the target node in G as a sequence of nodes. That is, the response takes the form of a string y : v_1 v_2 v_2 v_3 . . . v_H-1 v_H
where v i ∈ [m] is the i'th nodes in the path for each i ∈ [H], and v 1 = s while v H = t. Here, the horizon H corresponds to the path length in G, and in the layered DAG we have H = L + 2.
Summary: Graph search to sequence modeling problem. In summary, a graph search task with set of problem instances G induces an autoregressive sequence modeling problem with a vocabulary space V = [m] ∪ {|, /, =}, prompts X ⊆ V * corresponding to search problems in a layered DAG graph structure with L + 2 layers, and responses Y ⊆ V H corresponding to paths with length H = L + 2. In addition, the task is equipped with µ ∈ ∆(X ) and π D : X → ∆(Y) that is used to collect the training dataset D = {(x, y)}, where x ∼ µ and y ∼ π D (x).
D.1.2 MODEL DETAILS Next, we describe the common implementation details for the models we train to solve the graph search task.
Tokenizer. We use a numeral tokenizer, which is standard for graph reasoning tasks (Sanford et al., 2024;Bachmann & Nagarajan, 2024). Each node v ∈ [m] is tokenized as its integer node value, and the special characters |, /, and = are tokenized as m + 1, m + 2, m + 3, respectively.
Transformer model. We train causally-masked GPT2-like transformer models to minimize the cross-entropy loss using the Adam optimizer with fixed learning rate, and perform a grid search over the parameters displayed in Table 1. Parameters with fixed values were chosen based on related papers such as Bachmann & Nagarajan (2024). In both experiments, the model architecture with 4 heads, 6 hidden layers, and 384 hidden dimensions worked best. We use absolute positional encodings. Training iterations and grid search values for the learning rate are different for each experiment, and discussed further below.
this section cite: ['b51', 'b3', 'b3']

Section: D.2 EXPERIMENT DETAILS FOR FIGURE 1
The graph search task for Figure 1 exposes natural properties of pre-training data under which crossentropy reduction comes at the cost of a worse coverage profile. The key idea is that because the pre-training data is diverse (with multiple distinct modes or graph classes), the model is unable to perfectly fit the distribution. As a result, when one mode of behavior is better-represented than another, cross-entropy minimization, which is an average-case distribution-matching metric, can sacrifice coverage across the different modes in order to increase performance on a single mode. Concretely, the graph search task for Figure 1 is a mixture of two classes of graph structures. Due to representational and finite-sample constraints, the model is unable to fit both perfectly during training, and, in particular, fitting one class well (in the sense of cross-entropy loss) comes at the cost of worse performance on the other. The checkpoint with the best coverage arises at some middle point in training when the model learns both classes of graphs equally well, and has good coverage over both classes (the dip Cov N in the leftmost subplot of Figure 1). Further reduction of cross-entropy loss over the latter half of training requires the model to lose coverage over π D in the less-represented graph class (observed as the increase in Cov N in the latter half of training iterations).
Even though the task cannot be learned perfectly from the supervised learning feedback, the model can still learn a policy that always samples a correct path matching π D 's with N = O(1) Best-of-N sampling attempts, which means that it leads to efficient downstream post-training (e.g., on one of the modes or with reward-based feedback), and also achieves optimal performance with test-time scaling methods.
For the experiments in Figure 1, we first pre-train a model on a larger set of graph structure classes so that it learns a diverse set of behaviors, then finetune its behavior on two. The performance on the fine-tuning task is displayed in Figure 1, and we first describe the fine-tuning dataset, followed by the pre-training dataset.
this section cite: []

Section: D.2.1 TASK DESCRIPTION
All graphs in G follow the layered DAG structure described in Appendix D.1 with L = 8 intermediate layers that each have 4 nodes, i.e., |V i | = 4 for layers i ∈ {2, . . . , 9} (recall the first and last layers contain only s and t, respectively).
Recall that in a layer i,
V i * = v ∈ V i : deg + (v) > 0 denotes the set of passable nodes. For each graph problem G = (G, s, t) ∈ G with graph structure G = (V, E),
Data distribution. The set of problem instances G = G 1 G 2 is comprised of two disjoint classes of problems, G 1 and G 2 .
The prompt distribution in the fine-tuning task is a skewed mixture over the two classes with µ ∈ ∆({1, 2}) denoting the probability of each class in the data; within each class, the graphs are drawn uniformly at random (described at the end of this section). Although there are 4 valid paths from source to target, in each class G 1 or G 2 the policy π D chooses one path based on a different global rule, described below.
Class G 1 (probability µ(1) = 0.9). For an integer j ∈ Z, let the function p(j) = (j mod 2) denote its parity. For layers i with |V i * | = 1, π D deterministically selects the unique passable node. For layers i ∈ I 2 (where
|V i * | = 2)
, the set V i * contains one even and one odd node, and π D deterministically chooses the node v ∈ V i * such that p(v) = p(i); that is, the node whose parity matches the parity of the layer index.
Class G 2 (probability µ(2) = 0.1). For layers i with |V i * | = 1, π D deterministically selects the unique passable node. For layers i ∈ I 2 (where
|V i * | = 2), π D chooses the node v ∈ V i * such that p(v) = 1 ⊕ p(i);
that is, the node whose parity is opposite to the parity of the layer index.
The class of a graph is technically identifiable from the prompt by computing a parity-based feature over a randomly selected subset of the nodes, but this problem is too difficult for the model to learn in the fine-tuning stage. Let V ′ ⊆ V be a fixed subset of nodes whose cardinality is half the total number of nodes in the graph (i.e., |V ′ | = |V |/2). Then all graphs in G 1 satisfy 1 = u∈V ′ p(u), while all graphs in G 2 satisfy 0 = u∈V ′ p(u). However, determining which nodes belong to V ′ requires complex reasoning over the graph structure.
Dataset. Each sample in the dataset D = {(x, y)} is then generated via the following procedure.
this section cite: []

Section: First sample an index i ∼ µ.
2. Sample G ∈ G i by randomly drawing V ⊂ [m] without replacement, and instantiate the edges according to the description for each class above.
3. Format the prompt x per Appendix D.1.
4. Draw y ∼ π D (• | x) according to description for each class above.
this section cite: []

Section: D.2.2 PRE-TRAINING DESCRIPTION
The graph problem instances in the pre-training task, G pre , are a superset of the graphs in the finetuning task, that is, ∪ i∈[K] G i = G pre with K = 3, and G 1 and G 2 defined as in the previous section for the finetuning dataset. The data distribution is a uniform mixture of these 3 classes, µ(i) = 1 K for each i ∈ [K], and the third class G 3 shares the same layered DAG structure as G 1 and G 2 (with L = 8 intermediate layers, where two layers are randomly chosen to have multiple passable nodes). However, in G 3 , π D is a stochastic policy and samples one of the 2 2 = 4 valid paths at random. The dataset is then drawn using the same data generation procedure described for the fine-tuning task above.
this section cite: []

Section: D.2.3 TASK-SPECIFIC IMPLEMENTATION DETAILS
The transformer model is first pre-trained on a fixed dataset drawn from the pre-training distribution, with 8 × 64, 000 prompts in total, using a learning rate of 1e-4 for 200k iterations, which was chosen based on a grid search over learning rates {5e-5, 1e-4, 5e-4}.
The final checkpoint is then finetuned for 50k iterations in an online fashion, where fresh samples are drawn for each batch (this is equivalent to offline training with a dataset that has an equivalent number of samples). The learning rate is 5e-6, which was chosen based on a grid search over learning rates {5e-6, 1e-5}.
this section cite: []

Section: D.3 EXPERIMENT DETAILS FOR FIGURE 2
For Figure 2, we consider a family of tasks that is parameterized by the horizon H, in order to expose the fact that cross-entropy is sensitive to horizon, but the coverage profile is not. This construction leverages the intuition from Remark 3.1. The training data is heterogeneous, with a fraction consisting of difficult graph problems that the model cannot learn to cover with the given number of training samples. This un-learnable subset of the data contributes to the large KL-divergence, but does not affect the coverage profile.
this section cite: []

Section: D.3.1 TASK DESCRIPTION
For Figure 2, we devise a family of tasks parameterized by the number of intermediate layers H ∈ {8, 16, 24}. For a fixed H, each task G H utilizes the layered DAG graph structure described in Appendix D.1 with L = H intermediate layers, each containing 4 nodes, so that each graph has H + 2 total layers (including source and target). The response space is Y = V H+2 , corresponding to paths of length H + 2 (including the source and target nodes).
this section cite: []

Section: Data distribution.
The task is a heterogeneous mixture over 3 classes of graphs described below that we refer to as
G H,1 ∪ G H,2 ∪ G H,3 = G H .
The classes G H,2 and G H,3 are significantly harder to learn and the model will fail to do so with the given number of training samples, even though G H,1 is learned quickly (and also provides useful features for learning the other two tasks). The distribution over these 3 classes is fixed for all H and specified by µ ∈ ∆({1, 2, 3}).
Class G H,1 (probability µ(1) = 0.94). All H intermediate layers have only 1 passable node each (i.e., |V i * | = 1 for all i ∈ {2, . . . , H + 1}), so each G ∈ G H,1 has only one valid path from source to target. For prompts corresponding to graphs in this class, π D deterministically selects the unique valid path.
Class G H,2 (probability µ(2) = 0.05). For each graph, half of the intermediate layers (or H/2) are randomly selected to have two passable nodes, while the rest have one. More formally, a subset I H/2 ⊂ {2, . . . , H + 1} with |I H/2 | = H/2 is randomly selected, such that |V i * | = 2 for i ∈ I H/2 and
|V i * | = 1 for i ∈ {2, . . . , H + 1} \ I H/2 .
There are 2 H/2 valid paths from source to target, and π D deterministically selects one of them. For layers i with |V i * | = 1, π D selects the unique passable node. For layers i ∈ I H/2 (where
|V i * | = 2), π D selects the node v ∈ V i
* by following a difficult, deterministic rule. This rule requires π D to select the node v whose parity matches the parity of the layer index, XOR'ed with the parity of each passable node in the entire graph. More specifically, recall that p(j) denotes the parity of an integer j ∈ [m], and let V * := H+1 i=2 V i * denote the set of all passable nodes across all intermediate layers (including those with just one passable node). Then in layer i ∈ Note that prompts/graphs from each class are distinguishable from each other (or, identifiable) based on prompt features alone, so a powerful-enough model can achieve perfect performance across all of them simultaneously. G H,2 , for example, has more edges and thus a longer prompt than G H,1 ; similar statements apply to G H,3 . Dataset generation occurs in the same manner as described in Appendix D.2.
I H/2 , π D selects the node v ∈ V i * such that p(v) = p(i) ⊕ u∈V * p(u) . Class G H,3 (probability µ(3) = 0.01).
this section cite: []

Section: D.3.2 TASK-SPECIFIC IMPLEMENTATION DETAILS
Lastly, we describe experiment-specific implementation details on top of those previously described in Appendix D.1, which are common to all experiments. In addition to a grid search over the parameters in Table 1, we perform a search over learning rates {5e-5, 1e-4, 5e-4}, for which the learning rate of 1e-4 exhibited the best validation performance. The model is trained for 40k iterations over a fixed dataset of 8 × 64, 000 samples.
The results in Figure 2 are computed from evaluations of training checkpoints on per-class validation datasets of 1024 prompts from each G H,i for i ∈ [3]; these metrics are then averaged according to the probabilities in µ to obtain the final result. In total we ran 16 seeds, and plot their median. The shaded region in Figure 2 displays the region between the 1 16 quantile and 15 16 quantile.
this section cite: []

Section: E PROPERTIES OF THE COVERAGE PROFILE
Before proceeding, we briefly discuss some properties of the coverage profile that will be helpful to keep in mind.
Remark E.1 (Coverage profile as a refinement of cross-entropy). The coverage profile can be viewed as a fine-grained, inference budget-sensitive refinement of cross-entropy. Concretely, if we write
Cov N (π D π) = P πD log π D (y | x) π(y | x) ≥ log N , (16
)
it becomes clear that the coverage profile is simply the cumulative distribution function (CDF) of the log density ratio X := log πD(y|x) π(y|x) , while KL-divergence corresponds to the mean:
E πD [X].
It is well known that the CDF of a random variable is a more informative statistic than its mean (Durrett, 2019); the former can be much more sensitive to the model's behavior at the tail than the latter. Indeed, the coverage profile can behave very differently across scales, as shown by Figure 1 . 6   Remark E.2 (KL divergence and coverage profile are not estimable). We emphasize that KLdivergence and the coverage profile are not estimable quantities in general, due to the fact both depend on the unknown density π D (y | x) for the data distribution. This motivates the use of crossentropy in practice, as the former is an estimable upper bound on D KL (π D π). Analogously, we show in Section 6.2 that various estimable proxies for the coverage profile can be used to select models with good coverage. One exception is the expert distillation setting (see Appendix G.4), where π D is a teacher network for which the log-probabilities log π D (y | x) are available.
this section cite: []

Section: F SUPPORTING RESULTS
This section presents technical results used throughout the paper. Appendix F.1 presents basic properties of the coverage profile. Appendix F.2 analyzes the performance of the Best-of-N algorithm under coverage. Appendix F.3 presents properties of the maximum likelihood estimator, and Appendix F.4 presents structural results relating the coverage profile to a "stopped" KL-divergence, which are useful for analyzing autoregressive models.
this section cite: []

Section: F.1 PROPERTIES OF THE COVERAGE PROFILE
This section presents elementary properties of the coverage profile.
Proposition F.1 (KL-to-coverage conversion). For all models π D and π and M ≥ 2, we have
Cov N (π) ≤ D KL (π D π) log N -1 + 1 N .
Proof of Proposition F.1. Lemma 27 of Block & Polyanskiy (2023) states that for any N > 1 and any convex f : [0, ∞] → [0, ∞] with f (1) = f ′ (1) = 0,
Cov N (π) = P πD π D (y | x) π(y | x) > N ≤ N D f (π D π) f (N ) , (17
)
where
D f (π D π) := E π f dπD dπ .
Applying this with KL-divergence, which corresponds to f (x) = x log x -x + 1 with f ′ (x) = log x, we have that
N f (N ) = 1 log N -1 + 1/N , (18
)
which gives the result.
Proposition F.2 (Tightness of KL-to-coverage conversion). For any N ≥ 2, there exist models π D and π such that
Cov N ( π) ≥ D KL (π D π) log N -1 2 + 1 2N .
Proof of Proposition F.2. Consider π D = Ber(p) and π = Ber(p/N ) with p ≤ 1 2 . Then Cov N ( π) = p and
D KL (π D π) = p log N + (1 -p) log 1 -p 1 -p N ≤ p log N + (1 -p) 1 -p 1 -p N -1 = p log N -(1 -p) 1 -1 N 1 -p N ≤ p • log N - 1 2 + 1 2N .
This is the desired result.
Proposition F.3 (Uniform coverage decay implies bounded KL). Given π, π D : X → ∆(Y), define W max := sup x,y πD(y|x) π(y|x) and C := sup
N ≥1 {Cov N (π) • log N },
where we note that C ≤ log W max . It holds that
D KL (π D π) ≤ C • (1 + log(log(W max )/C)).(19)
Proof of Proposition F.3. Let δ > 0 a fixed parameter, and define X := π D /π. Then we have
D KL (π D π) = E πD [log(X)] ≤ E πD [log(X)I{log(X) > δ}] + δ. (20
)
Since X ≤ W max almost surely, we can write
E πD [log(X)I{log(X) > δ}] = log(Wmax) δ P πD [log(X) > t]dt (21
) = log(Wmax) δ P πD X > e t dt (22
) ≤ C log(Wmax) δ 1 t dt (23) = C log log(W max ) δ . (24
)
The result now follows by setting δ = C.
Proposition F.4 (Hellinger-to-coverage conversion). For all models π D and π and N > 1, we have
Cov N (π D π) ≤ 2N ( √ N -1) 2 • D 2 H (π D , π).
Proof of Proposition F.4. Without loss of generality, we assume Y is discrete in the following proof. By definition,
D 2 H (π D , π) = 1 2 E x∼πD y π D (y | x) -π(y | x) 2 ≥ 1 2 E x∼πD y π D (y | x) 1 - 1 √ N 2 I π(y | x) ≤ 1 N π D (y | x) = 1 2 1 - 1 √ N 2 P πD π D (y | x) π(y | x) > N ,
where the inequality follows from the fact that π
D (y | x) -π(y | x) ≥ 1 -1 √ N π D (y | x) is implied by π(y | x) ≤ 1 N π D (y | x).
Re-organizing completes the proof.
Proposition F.5 (Chain rule for coverage profile). For any models π D , π T , and π, and any M 1 , M 2 ≥ 2, we have
Cov M1 (π T π) ≤ M 2 • Cov M1/M2 (π D π) + Cov M2 (π T π D ).(25)
Proof of Proposition F.5. We can write
Cov M1 (π T π) = P πT π T (y | x) π(y | x) > M 1 = P πT π T (y | x) π(y | x) > M 1 , π T (y | x) π D (y | x) ≤ M 2 + P πT π T (y | x) π(y | x) > M 1 , π T (y | x) π D (y | x) > M 2 ≤ M 2 P πD π D (y | x) π(y | x) > M 1 /M 2 + P πT π T (y | x) π D (y | x) > M 2 = M 2 Cov M1/M2 (π D π) + Cov M2 (π T π D ).
this section cite: ['b12']

Section: F.2 ANALYSIS OF BEST-OF-N SAMPLING UNDER A GOOD COVERAGE PROFILE
In this section we analyze the performance of the Best-of-N algorithm under a good coverage profile. Let a base model π be given, and let a reward function r T (x, y) ∈ [0, 1] be given. Let π T : X → ∆(Y) denote an arbitrary task-specific comparator policy.
We let π BoN N (x) denote the distribution of the Best-of-N algorithm with parameter N , which draws N responses y 1 , . . . , y N i.i.d.
∼ π(• | x) and returns y = arg max yi r T (x, y i ).
Proposition F.6 (Coverage implies success for BoN). Let M ≥ 1 be given. For any ε > 0, if N ≥ 2M log(ε -1 ) and Cov M (π T π) ≤ 1 2 , then we are guaranteed that
E x∼µ r T (x, π T (x)) -r T (x, π BoN N (x)) ≤ Cov M (π T π) + ε. (26
)
Proof of Proposition F.6. This is an immediate consequence of Lemma F.1 in Huang et al. (2025b), noting that we can bound E M (π T π) ≤ Cov M (π T π).
Proposition F.7 (Coverage is necessary for BoN). For any model π and reference π T , and for any N ≥ 2, there exists a reward function r T (x, y) ∈ {0, 1} such that
E x∼µ r T (x, π T (x)) -r T (x, π BoN N (x)) ≥ 1 2 Cov 2N (π T π). (27
)
Proof of Proposition F.7. For any x ∈ X , we define S x := y ∈ Y : πT(y|x) π(y|x) ≥ 2N and let r T (x, y) = I{y ∈ S x }.
By definition, for any fixed x ∈ X , it holds that
r T (x, π BoN N (x)) = P y∼ π BoN N (x) (y ∈ S x ) = P y 1 ,...,y N i.i.d. ∼ π(•|x) (∃i ∈ [N ], y i ∈ S x ) = 1 -1 -P y∼ π(•|x) (y ∈ S x ) N ≤ N • P y∼ π(•|x) (y ∈ S x ) = N • y∈Sx π(y | x) ≤ N • y∈Sx 1 2N π T (y | x) = 1 2 P y∼πT(•|x) (S x ),
where we use the fact that π(y | x) ≤ 1 2N π T (y | x) for any y ∈ S x . We also note that
P x∼µ,y∼πT(•|x) (y ∈ S x ) = Cov 2N (π T π). Therefore, E x∼µ r T (x, π T (x)) -r T (x, π BoN N (x)) ≥ 1 2 Cov 2N (π T π).
this section cite: []

Section: F.3 PROPERTIES OF MAXIMUM LIKELIHOOD
In this section, we specialize standard guarantees for maximum likelihood (Wong & Shen, 1995;van de Geer, 2000;Zhang, 2006) to derive bounds on the coverage profile; as discussed in Appendix C, these results are not tight compared to Theorem 4.1.
Proposition F.8 (Convergence of maximum likelihood in Hellinger distance). Assume that π D ∈ Π.
With probability at least 1 -δ, the maximum likelihood estimator π := arg max π∈Π L n (π) satisfies,
D 2 H (π D , π) ≲ inf ε>0 log N ∞ (Π, ε) n + ε , (28
)
and consequently
Cov M ( π) ≲ inf ε>0 log N ∞ (Π, ε) n + ε . (29
)
for all M ≥ 2.
Proof of Proposition F.8. The first bound follows from Proposition B.2 of Foster et al. (2024). The second bound follows from applying Proposition F.4.
Proposition F.9 (Convergence of maximum likelihood in KL). Assume that π D ∈ Π, and that all π ∈ Π satisfy πD π ∞ ≤ W max . With probability at least 1 -δ, the maximum likelihood estimator π := arg max π∈Π L n (π) satisfies,
D KL (π D π) ≲ log W max • inf ε>0 log N ∞ (Π, ε) n + ε , (30
)
and consequently
Cov M ( π) ≲ log W max log M • inf ε>0 log N ∞ (Π, ε) n + ε , (31
)
for all M ≥ 2.
We remark that the log(W max )-factor in Eq. ( 30) can be tight in general. For example, for the class Π considered in Proposition 3.2, it holds that log N ∞ (Π, ε) ≲ log(1/ε) ∨ 1 and πD π ∞ ≤ e 2H . Proof of Proposition F.9. By Lemma 4 of Yang & Barron (1998), it holds that
D KL (π D π) ≤ (2 + log(W max ))D 2
H (π D , π). Therefore, the first bound then follows from Eq. ( 28). The second bound follows from applying Proposition F.1.
this section cite: ['b66', 'b61']

Section: F.4 AUTOREGRESSIVE MODELS: COVERAGE AND STOPPED KL-DIVERGENCE
This section shows that we can relate the coverage profile to a "stopped" KL-divergence defined in Eq. ( 32). This is a useful result in the context of autoregressive models because the stopped KL-divergence is always bounded, even when KL-divergence itself may not be.
Proposition F.10. Define the stopped KL-divergence for parameter N as
D seq,N (π D π) = E (x,y 1:H )∼πD min log N, H h=1 D KL (π D (• | x, y 1:h-1 ) π(• | x, y 1:h-1 )) . (32
)
Then as long as N > e, it holds that
Cov N (π D π) ≤ 2 log N -1 D seq,N (π D π).(33)
Proof of Proposition F.10. Consider the stopping time
τ := min    h : h = H or j≤h D KL (π D (y j+1 = • | x, y 1:j ) π(y j+1 = • | x, y 1:j )) > log N    .
Then, for the process Y τ = (x, y 1:τ ), we have the chain rule:
D KL (π D (Y τ = •) π(Y τ = •)) = E πD τ h=1 D KL (π D (y h = • | x, y 1:h-1 ) π(y h = • | x, y 1:h-1 )) ≤ E πD min log N, H h=1 D KL (π D (y h = • | x, y 1:h-1 ) π(y h = • | x, y 1:h-1 )) ,
where the inequality uses j<τ D KL (π D (y j+1 = • | x, y 1:j ) π(y j+1 = • | x, y 1:j )) ≤ log N , which follows from the definition of τ . Therefore, by Proposition F.1, we have
P πD π D (Y τ ) π(Y τ ) ≥ log N ≤ D KL (π D (Y τ = •) π(Y τ = •)) log N -1 + 1/N .
Published as a conference paper at ICLR 2026 Finally, we bound
P πD π D (y 1:H | x) π(y 1:H | x) ≥ N ≤ P πD (τ < H) + P πD π D (Y τ ) π(Y τ ) ≥ log N .
By Markov's inequality,
P πD (τ < H) ≤ P πD H h=1 D KL (π D (• | x, y 1:h-1 ) π(• | x, y 1:h-1 )) > log N ≤ 1 log N E πD min log N, H h=1 D KL (π D (• | x, y 1:h-1 ) π(• | x, y 1:h-1 )) .
Combining the inequalities above completes the proof.
The following result is a sort of partial converse to Proposition F.10, showing that the coverage profile can be lower bounded in terms of the tail behavior for a sum of step-wise Hellinger distances.
Proposition F.11. For any N ≥ 1 and δ ∈ (0, 1), it holds that
Cov N (π D π) ≥ P πD H h=1 D 2 H (π D (• | x, y 1:h-1 ), π(• | x, y 1:h-1 )) ≥ log(N/δ) -δ.
Proof of Proposition F.11. By definition,
E y h ∼πD(•|x,y 1:h-1 ) exp - 1 2 log π D (y h | x, y 1:h-1 ) π(y | x, y 1:h-1 ) = y h ∈Y π D (y h | x, y 1:h-1 ) • π(y | x, y 1:h-1 ) = 1 -D 2 H (π D (• | x, y 1:h-1 ), π(• | x, y 1:h-1 )) ≤ exp -D 2 H (π D (• | x, y 1:h-1 ), π(• | x, y 1:h-1 )
) . Therefore, it holds that
E πD exp H h=1 D 2 H (π D (• | x, y 1:h-1 ), π(• | x, y 1:h-1 )) - 1 2 log π D (y h | x, y 1:h-1 ) π(y | x, y 1:h-1 ) ≤ 1.
By Markov inequality, this implies
P πD 1 2 log π D (y 1:H | x) π(y 1:H | x) ≤ H h=1 D 2 H (π D (• | x, y 1:h-1 ), π(• | x, y 1:h-1 )) -log(1/δ) ≤ δ.
To conclude, we note that
P πD H h=1 D 2 H (π D (• | x, y 1:h-1 ), π(• | x, y 1:h-1 )) ≥ log(N/δ) ≤ P πD H h=1 D 2 H (π D (• | x, y 1:h-1 ), π(• | x, y 1:h-1 )) ≥ 1 2 log π D (y 1:H | x) π(y 1:H | x) + log(1/δ) + P πD 1 2 log π D (y 1:H | x) π(y 1:H | x) + log(1/δ) ≥ log(N/δ) ≤ δ + Cov N (π D π).
Re-organizing gives the desired result.
this section cite: []

Section: G ADDITIONAL RESULTS

this section cite: []

Section: G.1 TIGHTNESS OF THEOREM 4.1
To conclude, we show that the coarse and fine-grained terms in Theorem 4.1 are both tight in general.
Proposition G.1. The following lower bounds on coverage hold for the maximum likelihood estimator.
(a) Coarse rate: For any n, d ≥ 1 and B ≥ log(5n), there exists a class Π with log N ∞ (Π, α) ≲ d log(B/α) ∨ 1 and π D ∈ Π such that with probability at least 0.5, it holds that for any N ≤ e B ,
Cov N ( π) ≥ c • d n .
(b) Fine rate: For any n ≥ d ≥ 1, N ≥ 1, there exists a class Π and π D ∈ Π such that |Π| = 2 d + 1
and N ∞ (Π, α) ≤ 2 for any α ≥ d n , and with probability at least 0.5, it holds that
Cov N ( π) ≥ c • d n • log N .
Informally, case (a) shows that for the class Π under consideration, the coverage does not decrease with log N until N is trivially large such that log N ∞ (Π, log N ) = 0; this is precisely the behavior of the coarse term in Theorem 4.1, so this implies there is no hope of removing this term. Meanwhile, case (b) can be interpreted as showing that there is no hope of replacing the high-precision covering number found in the fine-grained term in Theorem 4.1 with a coarser notion (e.g, at the scale in the coarse-grained term), since the rate grows with d ≈ log|Π| even though log N (Π, α) is constant for α ≥ d n . We note that Proposition G.1 is an algorithm-specific lower bound, not an informationtheoretic lower bound; we show in Section 6.2 is that it is possible to improve over Theorem 4.1 with algorithms explicitly designed to optimize for coverage. G.2 MAXIMUM LIKELIHOOD: BETTER COVERAGE FOR CONVEX CLASSES In this section, we give an extension to Theorem 4.1 which shows that maximum likelihood can achieve a faster convergence rate for coverage-as well as strong tolerance to misspecificationwhen the model class is convex.
Assumption G.1 (Convex model class). The class Π satisfies Π = {π θ : θ ∈ Θ} for a convex, compact parameter space Θ, and the mapping θ → π θ (y | x) is concave for all x ∈ X , y ∈ Y.
Theorem G.1 (Fast convergence of coverage for convex classes). Let α ≥ 0, N ′ ≥ 1, N ≥ 2e 2α N ′ be given, and suppose that Assumption G.1 holds. Let
θ ⋆ ∈ arg min θ∈Θ D KL (π D π θ ).
With probability at least 1 -δ, the maximum likelihood estimator π := arg max π∈Π L n (π) satisfies
Cov N ( π) ≤ Cov N ′ (π θ ⋆ ) + C log N ∞ (Π, α) + log(δ -1 ) n + Ce 2α N ′ N • inf ε>0 log N ∞ (Π, ε) n + ε , (34
)
where C > 0 is an absolute constant.
Note that we allow for misspecification here, as Eq. ( 34) shows that the coverage of π can be upper bounded by the coverage of π θ ⋆ , the best-in-class approximator of π D with respect to KL-divergence.
In the well-specified case where π D ∈ Π, the bound simplifies to
Cov N ( π) ≲ 1 N 1-2c • inf ε>0 log N ∞ (Π, ε) n + ε + log N ∞ (Π, c log N ) + log(δ -1 ) n = C fine (Π, n) N 1-2c + C coarse (Π, N, n), which improves upon the rate Cov N ( π) ≲ Cfine(Π,n) log N + C coarse (Π, N, n) in Theorem 4.1. The proof of Theorem G.1 is presented in Appendix J.3. G.3 LOWER BOUND FOR MAXIMUM LIKELIHOOD UNDER MISSPECIFICATION
In the following proposition, we show that without a well-specified model class (Assumption 2.1), maximum likelihood may have coverage profile scaling with 1 log M min π∈Π D KL (π D π) (cf. Proposition F.3), even when there exists π ∈ Π such that Cov N (π) = 0.
Proposition G.2 (MLE under misspecification). For any α ∈ [0, 1], M > e α , there exists a problem instance π D and class Π = {π 1 , π 2 } such that
sup x,y |log π D (y | x) -log π 1 (y | x)| ≤ α, Cov N (π 2 ) ≥ cα 2 log M ,
and for any n ≥ 1, it holds that with probability at least 1 4 , the MLE π = π 2 , i.e.,
Cov N ( π) = Ω α 2 log M .
Proof of Proposition G.2. Let p = α 32 log M . Consider X = {+, -}, Y = {0, 1}, ρ(-) = p, ρ(+) = 1 -p, and π D is given by
π D (• | +) = π D (• | +) = Ber 1 2 .
We construct the class Π = {π 1 , π 2 } as
π 1 (•|+) = Ber 1 2e α , π 2 (•|-) = Ber 1 2 , π 2 (•|+) = Ber 1 2 , π 2 (•|-) = Ber 1 2M .
Given the dataset D = {(x t , y t )} t∈[n] sampled from π D , we define N (x, y) = #{t ∈ [n] : (x t , y t ) = (x, y)} and N (x) = N (x, 0) + N (x, 1). Then
L n (π 2 ) -L n (π 1 ) = N (+, 1) • α + N (+, 0) • log e α 2e α -1 -N (-, 1) • log M + N (-, 0) • log 2 - 1 M .
By symmetric, it holds that P(N (+, 1) ≥ N (+, 0)) ≥ 1 2 . Further, by Markov's inequality, it holds that P(N (-) ≥ 4np) ≤ 1 4 . Therefore, for the event E = {N (+, 1) ≥ N (+, 0), N (-) ≤ 4np}, we have P(E) ≥ 1 4 . In the following, we show that L n (π 2 ) -L n (π 1 ) > 0 under E.
We condition on E. We first note that under this event, we have N (+, 1) ≥ 1 2 N (+), N (+, 0) ≤ 1 2 N (+). Hence,
L n (π 2 ) -L n (π 1 ) ≥ N (+) 1 2 • α + 1 2 • log e α 2e α -1 -N (-) • log M = N (+) • D KL Ber 1 2 Ber 1 2e α -N (-) • log M ≥ N (+) • (1 -e -α ) 2 -N (-) • log M.
Finally, using the fact that 1 -e -α > 1 2 α and N (+) ≥ (1 -4p)n ≥ 1 2 n under E, we have
L n (π 2 ) -L n (π 1 ) > α 2 8 -4p log M n = 0.
Hence, under the event E, we have π = π 2 . However, it is clear that
Cov N (π 2 ) = p, Cov N (π 1 ) = 0.
This completes the proof.
G.4 STOCHASTIC GRADIENT DESCENT: IMPROVED GRADIENT NORMALIZATION FOR DISTILLATION In this section, we focus on autoregressive linear models (3), and consider a variant of our setting inspired by distillation . We assume that for each example (x i , y i 1:H ), for each h = 1, . . . , H, we have access to the true next-token probabilities π D (y h | x i , y i 1:h-1 ) for all y h ∈ V. This is an unrealistic assumption for general pre-training, but it is natural for distillation, where π D corresponds to a teacher model (in particular, the next-token probabilities are already computed as part of a standard forward pass through the teacher model).
For the distillation setting, we give an improved gradient normalization scheme that improves upon the rate achieved by Theorem 5.1, closing the gap between SGD and maximum likelihood by matching the guarantee for Theorem 4.2.
Define ϵ θ (x, y 1:h-1 ) := D KL (π D (• | x, y 1:h-1 ) π θ (• | x, y 1:h-1 )); note that for the distillation setting, we can compute this quantity in closed form for any prefix x, y 1:h-1 in the training corpus. We consider the following (single-sample) truncated/normalized stochastic gradient estimator:
g θ (y | x) = H h=1 α θ (x, y 1:h-1 )∇ log π θ (y h | x, y 1:h-1 ),(35)
where A := log N , and where
α θ (x, y 1:h-1 ) =      1, j≤h-1 ϵ θ (x, y 1:j ) ≤ A, 0, j<h-1 ϵ θ (x, y 1:j ) > A, A-j<h-1 ϵ θ (x,y1:j ) ϵ θ (x,y 1:h-1 ) , otherwise.(36)
With this definition, we define the following normalized SGD update:
θ t+1 = Proj Θ (θ t + η g θ t (y t | x t )).(37)
Intuitively, the idea behind the update in Eq. ( 35) is to truncate the gradient at the point where the KL divergence between the teacher and student model is too large, and then normalize the gradient by the KL divergence; this is inspired by the structural result Proposition F.10 in Appendix F.4, where we show a close connection between the coverage profile and a certain "stopped" variant of KL divergence.
Theorem G.2. Let T, N ≥ 1 be given. With a suitably chosen stepsize η > 0, the normalized SGD update (10) achieves the following coverage bound:
E 1 T T t=1 Cov N (π θ t ) ≲ σ 2 ⋆ T log N + B 2 T . (38
)
This guarantee matches the rate of Theorem 4.2 for the maximum likelihood estimator. The proof is presented in Appendix K.7.
this section cite: []

Section: G.5 AN IMPROVED TOURNAMENT VIA ON-POLICY GENERATION
We describe an improved tournament estimator that is able to remove that 1/N 1-a term from Theorem 6.2, meaning it achieves nontrivial guarantees even when the coverage parameter N is a constant.
Note that the term 1/N 1-a of Eq. ( 15) comes from the fact that P πD ( π(y|x) πD(y|x) ≥ N ) can be as large as 1/N in the worst case, implying that the π produced by Eq. ( 14) may at best achieve a coverage of 1/N . To overcome this, we introduce an offset term:
π := arg min π∈Π max π ′ ∈Π { Cov N (π ′ π) -2N a • Cov π N (π ′ π)} ,(39)
where we define
Cov π N (π ′ π) := 1 n n i=1 P y∼π(•|x i ) π ′ (y|x i )
π(y|x i ) ≥ N for models π, π ′ , π. This estimator augments the simple tournament in Eq. ( 14) with an "offset" term that accounts for the fact that some of the models might be quite far from π D . The main guarantee is as follows.
Theorem G.3. Fix N ≥ 1, a > 0 such that N 1-2a ≥ 4. Suppose that there exists π ∈ Π such that |log π D (y | x) -log π(y | x)| ≤ a log N for any x ∈ X , y ∈ Y. Then with probability 1 -δ, the tournament estimator (39) achieves Cov 2N 1+a ( π) ≲ log(|Π|/δ) n . Compared to Theorem 6.2, this tournament eliminates the additive 1/N 1-a term. It does, however, require a stronger condition on the best-in-class model π that |log π D (y | x) -log π(y | x)| ≤ a log N , which implies in particular that Cov N a (π) = 0.
Infinite classes: Beating maximum likelihood. While we motivated the tournament estimators through model/checkpoint selection with a finite class Π, both estimators can also be applied to general, infinite classes Π. In this case, it turns out that they both improve upon the coverage achieved by the maximum likelihood estimator in Theorem 4.1, even in the well-specified case where π D ∈ Π; informally, the tournament estimators allow us to remove the fine-grained term in Theorem 4.1, leaving only a coarse-grained term. See Theorem 6.2 ′ and Theorem G.3 ′ for the formal statements.
Part II Proofs H TECHNICAL TOOLS Notation. We denote by B d 2 (R) := v ∈ R d : v ≤ R the d-dimensional Euclidean ball of radius R. We drop the superscript when the dimension d is clear from context.
this section cite: []

Section: H.1 CONCENTRATION INEQUALITIES
Lemma H.1 (Freedman's inequality). Let (Z i ) i≤n be a real-valued martingale difference sequence adapted to a filtration (F i ) i≤n . If |Z i | ≤ R almost surely, then for any η ∈ (0, 1/R), with probability at least 1 -δ, for all n ′ ≤ n,
n ′ i=1 Z i ≤ η n ′ i=1 E i-1 (Z i ) 2 + log(δ -1 ) η .
The next result is a standard consequence of Lemma H.1 (e.g., Foster et al. ( 2021)).
Lemma H.2. Let (Z i ) i≤n be a sequence of random variables adapted to a filtration (F i ) i≤n . If 0 ≤ Z i ≤ R almost surely, then with probability at least 1 -δ, for all n ′ ≤ n,
n ′ i=1 Z i ≤ 3 2 n ′ i=1 E i-1 [Z i ] + 4R log(2δ -1 ),(40)
and
n ′ i=1 E i-1 [Z i ] ≤ 2 n ′ i=1 Z i + 8R log(2δ -1 ). (41
)
The following lemma is a uniform version of, e.g., Lemma 23 in Foster & Rakhlin (2023).
Lemma H.3. Suppose that µ is a distribution over Z, and let F ⊆ (Z → R) be a function class. We let N (F, ϵ; • ∞ ) be the ϵ-covering number of F under the norm ρ(f, f ′ ) := sup z∈Z |f (z) -f ′ (z)|.
Let D = {Z 1 , • • • , Z n } be drawn i.i.d. from µ.
Then the following holds with probability at least 1 -δ:
n i=1 f (Z i ) ≤ n log E µ [exp(f (Z))] + log(1/δ) + inf ϵ≥0 {log N (F, ϵ; • ∞ ) + 2nϵ}, ∀f ∈ F .
Proof of Lemma H.3. Fix ϵ ≥ 0 attaining the minimum of log N (F, ϵ; • ∞ ) + 2nϵ, and let
f 1 , • • • , f J be an ϵ-covering of F of size J = N (F, ϵ; • ∞ ). For each j ∈ [J], we define g j (z) := f j (z) -log E µ [exp(f j (Z))]. Then, it is clear that E µ e gj (Z) = 1, and hence E exp n i=1 g j (Z i ) = 1, ∀j ∈ [J].
By Markov's inequality and the union bound, it holds that with probability at least 1 -δ,
n i=1 g j (Z i ) ≤ log(J/δ), ∀j ∈ [J].(42)
Note that for any f ∈ F , there exists j ∈ [J] such that ρ(f, f j ) ≤ ϵ, and in particular
f (Z i ) -log E µ [exp(f (Z))] ≤ 2ϵ + f j (Z i ) -log E µ [exp(f j (Z))] = 2ϵ + g j (Z i ), ∀i ∈ [n],
and hence Eq. ( 42) implies that n i=1 f (Z i ) ≤ n log E µ [exp(f (Z))] + log(J/δ) + 2nϵ. By the arbitrariness of f , the proof is hence completed.
this section cite: []

Section: H.2 INFORMATION-THEORETIC INEQUALITIES
Lemma H.4. For distribution P, Q ∈ ∆(X ), function f : X → [-B, B], it holds that
|E P [f ] -E Q [f ]| ≤ 4 Var Q [f ] • D 2 H (P, Q) + 8BD 2 H (P, Q).
More generally, for any g : X → B 2 (B), it holds that
E P [g] -E Q [g] ≤ 4 E Q g -E Q [g] 2 • D H (P, Q) + 8BD 2 H (P, Q). (43
)
and
E P g -E P [g] 2 ≤ 3 E Q g -E Q [g] 2 + 16B 2 D 2 H (P, Q). (44
)
Proof of Lemma H.4. We denote P (x) (resp. Q(x)) to be the density function of P (resp. Q).
Then for any function f : X → R,
|E P [f ] -E Q [f ]| 2 = X (f (x) -E Q [f ])(P (x) -Q(x))dx 2 ≤ X (f (x) -E Q [f ]) 2 ( P (x) + Q(x)) 2 dx • X ( P (x) -Q(x)) 2 dx ≤ 4D 2 H (P, Q) • Var Q [f ] + E P (f -E Q [f ]) 2 .
In particular, when h : X → [0, M ], the inequality above implies that
|E P [h] -E Q [h]| ≤ 2D H (P, Q) M (E P [h] + E Q [h]) ≤ 1 2 (E P [h] + E Q [h]) + 2M D 2 H (P, Q),
and hence it holds that
E P [h] ≤ 3 E Q [h] + 4M D 2 H (P, Q). Now, suppose that f : X → [-B, B]. Applying the above inequality to h(x) = (f -E Q [f ])
2 ∈ [0, 4B 2 ] gives
E P (f -E Q [f ]) 2 ≤ 3 E Q (f -E Q [f ]) 2 + 16B 2 D 2 H (P, Q). (45
)
Combining the above inequalities implies that
|E P [f ] -E Q [f ]| ≤ 4 Var Q [f ] • D 2 H (P, Q) + 8BD 2 H (P, Q).
To prove the upper bound for a vector-valued function g : X → B 2 (B), we can apply the above inequality with f v (x) := v, g(x) and take the maximum over v ∈ B 2 (1). The second upper bound follows similarly by applying Eq. ( 45).
Lemma H.5. Suppose that ϕ : Y → B 2 (B) with B ≥ 1, and for any θ ∈ B 2 (1), π θ ∈ ∆(Y) is defined as π θ (y) ∝ exp( ϕ(y), θ ). Then for any θ ⋆ , θ ∈ B 2 (1), it holds that
E y∼π θ ⋆ ϕ(y) -E π θ ⋆ [ϕ], θ -θ ⋆ 2 ≤ 15BD KL (π θ ⋆ π θ ).
Proof of Lemma H.5. Denote ϕ(y) := ϕ(y) -E π θ ⋆ [ϕ]. By definition,
D KL (π θ ⋆ π θ ) = log E y∼π θ ⋆ exp ϕ(y), θ -θ ⋆ ≥ B log E y∼π θ ⋆ exp 1 B ϕ(y), θ -θ ⋆ .
Note that for x ≥ -4, we have e x ≥ 1 + x + 1 10 x 2 . Therefore, we have
1 B D KL (π θ ⋆ π θ ) ≥ log 1 + 1 10B 2 E y∼π θ ⋆ ϕ(y), θ -θ ⋆ 2 ≥ 1 15B 2 E y∼π θ ⋆ ϕ(y), θ -θ ⋆ 2 ,
where we use log(1 + x) ≥ 3 4 x for all x ∈ [0, 8 5 ].
this section cite: []

Section: I PROOFS FROM SECTION 3
Proof of Proposition 3.2. Consider the setting where d = 1, X = {0, 1}, V = {-1, 1}, the distribution µ is given by µ(1) = 1 -µ(0) = 1 2n , and the feature map ϕ : X × V ⋆ → [-1, 1] is given by ϕ(0, •) = 0, and ϕ(1, y 1:h ) = y h .
In the following, we fix any algorithm Alg : (X × Y) n → ∆(Π). Let P π θ ,Alg be the probability distribution of (D = {(x t , y t )} t∈[n] , π) where x t ∼ µ, y t ∼ π θ (• | x t ) are sampled i.i.d. and π ∼ Alg(D).
Note that under this construction, P π θ ,Alg (x t = 0 ∀t ∈ [T ]) ≥ 1nµ(1) = 1 2 . Consider the event E = {x t = 0 ∀t ∈ [T ]}. Then, for any θ ⋆ ∈ [-1, 1], event A, it holds that
P π θ ⋆ ,Alg (A | E) = E π 0 ,Alg (A | E), because for any θ ∈ Θ, the distribution π θ (y 1:H = • | 0) = Ber 1 2 ⊗H is a product of H Bernoulli
distributions and does not depend on θ. Furthermore, for any θ ∈ [-1, 1],
D KL (π θ ⋆ π θ ) = µ(1) • D KL (π θ ⋆ (y 1:H = • | x = 1) π θ (y 1:H = • | x = 1)) = Hµ(1) • D KL Ber e θ ⋆ e θ ⋆ + e -θ ⋆ Ber e θ e θ + e -θ
,
and hence θ → D KL (π 1 π θ ) + D KL (π -1 π)
is minimized at θ = 0, i.e., for any π ∈ Π,
D KL (π 1 π) + D KL (π -1 π) ≥ H 2n • 2D KL Ber e e + e -1 Ber 1 2 ≥ H 2n .
Therefore, consider the event A θ := D KL (π θ π) ≥ H 4n , and we have shown that A c 1 ⊆ A -1 . Hence, we can lower bound
P π 1 ,Alg (A 1 ) + P π -1 ,Alg (A -1 ) ≥ P π 1 ,Alg (E)P π 1 ,Alg (A 1 | E) + P π -1 ,Alg (E)P π -1 ,Alg (A -1 | E) ≥ 1 2 E π 0 ,Alg [A 1 | E] + 1 2 E π 0 ,Alg [A -1 | E] ≥ 1 2 .
This gives max θ ⋆ ∈{-1,1} P π θ ⋆ ,Alg D KL (π θ ⋆ π) ≥ H 4n ≥ 1 4 , and the desired result follows immediately.
As a remark, we note that the construction above can be modified so that the variance σ 2 ⋆ (defined in Section 4.1) can be bounded as σ 2 ⋆ ≲ He -2B n . In particular, as long as B ≳ log H, it holds that σ ⋆ ≤ 1, implying that KL can converge slowly even when the "inherent variance" σ ⋆ is small.
this section cite: []

Section: J PROOFS FROM SECTION 4
J.1 PROOF SKETCH FOR THEOREM 4.1 The basic idea behind the proof of Theorem 4.1 is to interpret the condition Cov N (π) ≥ ε as an small-ball like anti-concentration condition in the vein of Mendelson (2014;2017). That is, for models π ∈ Π where coverage is large, the condition Cov N (π) ≥ ε witnesses a one-sided tail bound which implies that the empirical likelihood of π is not too large with high probability, and hence π cannot be a maximum-likelihood solution.
Let c ∈ (0, 1/2) be the absolute constant in Theorem 4.1, and let C ≥ log 4 be another absolute constant. Fix N such that log N ≥ 4C. For each model π ∈ Π, let S N (π) :
= 1 n | i ∈ [n] | πD(y i |x i ) π(y i |x i ) ≥ N 1-2c
| denote the empirical probability that π fails to cover π D . Our first step is to show via covering and concentration that with high-probability, all π ∈ Π satisfy
S N (π) ≥ 1 2 Cov N (π) -C coarse (Π, N, n). (46
)
That is, a large coverage profile implies that the number of points in the data where π fails to cover π D is large. This argument only depends on the covering number at a coarse log N scale-leading to the coarse-grained term in Theorem 4.1-because we only need to show that coverage concentrates, not the log-loss itself. 7We now argue that models with large coverage profile must have low log-likelihood compared to π D . In particular, using Eq. ( 46), we have
L n (π) -L n (π D ) = - n i=1 log π D (y i | x i ) π(y i | x i ) -C + + n i=1 log π(y i | x i ) π D (y i | x i ) ∨ (-C) (⋆) ≤ -|S N (π)|((1 -2c) log N -C) + n i=1 log π(y i | x i ) π D (y i | x i ) ∨ (-C) ≤ - n 4 log N • Cov N (π) + C coarse (Π, N, n) • O(n log N ) + n i=1 log π(y i | x i ) π D (y i | x i ) ∨ (-C),(47)
as long as c ≤ 1/8 and log N ≥ 4C. We view step (⋆) as using a form of implicit bias in the logarithmic loss: If an example (
x i , y i ) has πD(y i |x i ) /π(y i |x i ) ≥ N (i.e.
, π fails to cover π D on this example), this witnesses a negative contribution of order log N to the difference L n (π) -L n (π D ).
Next, using a variation of a standard one-sided tail bound for the logarithmic loss (van de Geer, 2000;Zhang, 2006), 8 we show that with high probability, all π ∈ Π satisfy
n i=1 log π(y i | x i ) π D (y i | x i ) ∨ (-C) ≲ C fine (Π, n) • n, (48
)
as long as C ≥ log 4. Combining Eq. ( 47) and Eq. ( 48), we conclude that all π ∈ Π have
Cov N (π) ≲ L n (π D ) -L n (π) + C fine (Π, n) • n n log N + C coarse (Π, N, n). (49
)
Since the maximum likelihood estimator π has L n (π D ) -L n ( π) ≤ 0, the result follows.
To summarize the key ideas as they relate to the final guarantee in Theorem 4.1: The coarse-grained term C coarse (Π, N, n) enters because we only need to show that the coverage profile concentrates, not the log loss itself. The fine-grained term C fine (Π, n) enters concentration of the empirical likelihood, with the 1/ log N scaling arising from implicit bias. The reason this argument avoids dependence on the sequence length H or other spurious parameters that would otherwise affect cross-entropy is that the argument is fundamentally one-sided: the conclusion Eq. ( 49) only shows that models with large coverage profile have low log-likelihood compared to π D .
this section cite: ['b39', 'b40', 'b61']

Section: J.2 PROOF OF THEOREM 4.1 (COVERAGE FOR MLE) Theorem 4.1 ′ (General version of Theorem 4.1).
Let N ≥ 8 be given. With probability at least 1 -δ, any approximate maximum likelihood estimator π with L n ( π) ≥ max π∈Π L n (π) -nε apx satisfies
Cov N ( π) ≲ log N ∞ (Π, c log N ) + log(δ -1 ) n + 1 log N inf ε>0 log N ∞ (Π, ε) n + ε + ε apx , (50
) where c > 0 is an absolute constant.
In the following, for a fixed threshold C ≥ log 4, we define the clipped log loss as
L + C (π) := n i=1 max log π(y i | x i ) π D (y i | x i ) , -C , (51
)
L - C (π) := n i=1 max 0, log π D (y i | x i ) π(y i | x i ) -C . (52
)
Note that L n (π) -L n (π D ) = L + C (π) -L - C (π). Furthermore, since π D ∈ Π, the approximate maximum likelihood estimator satisfies L n ( π) ≥ L n (π D ) -nε apx , and hence
L - C ( π) ≤ L + C ( π) + nε apx .
In the following, we show that L + C (π) can be bounded by a one-sided uniform convergence argument, and show that L - C (π) upper bounds the coverage profile Cov N (π) for any π ∈ Π and log N > C. Proposition J.1. Suppose that C ≥ log 4. Then, with probability at least 1 -δ, it holds that for any π ∈ Π,
L + C (π) ≤ log(1/δ) + 2 inf ϵ≥0 {log N ∞ (Π, ϵ) + nϵ}.
Proposition J.2. Fix any α ∈ (0, log N -C 2 ). Then, with probability at least 1 -δ, it holds that
Cov N (π) ≤ 2 log N -C -2α • L - C (π) + 16 log(2N ∞ (Π, α)/δ) n .
The proof of Theorem 4.1 and Theorem 4.1 ′ is completed by combining the propositions above and setting α = 1 4 log N . In what follows, we prove the propositions. Proof of Proposition J.1. This is a direct corollary of Lemma H.3. For each π ∈ Π, we let f π (x, y) := 1 2 max log π(y |x) πD(y |x) , -C and consider the function class F = {f π : π ∈ Π}. Then, N (F, ϵ; • ∞ ) ≤ N ∞ (Π, 2ϵ) for any ϵ ≥ 0. Applying Lemma H.3 with Lemma J.1 (stated and proved below) gives the desired upper bound.
Lemma J.1. As long as C ≥ log 4, it holds that
E (x,y)∼πD exp 1 2 max log π(y | x) π D (y | x) , -C ≤ 1. (53
)
Proof of Lemma J.1. We denote u = e -C and E := (x, y) : π(y |x) πD(y |x) ≥ u . Then it holds that
E (x,y)∼πD exp 1 2 max log π(y | x) π D (y | x) , -C = E (x,y)∼πD π(y | x) π D (y | x) I{(x, y) ∈ E} + √ u I{(x, y) ∈ E} = E x∼πD   y:(x,y)∈E π(y | x)π D (y | x)   + √ u P πD (E c ).
For x ∈ X , denote E x := {y : (x, y) ∈ E}. By the Cauchy-Schwarz inequality, we have
y:(x,y)∈E π(y | x)π D (y | x) ≤ y∈Ex π(y | x) • y∈Ex π D (y | x) ≤ P y∼πD(•|x) (E x ).
Therefore, as long as u ≤ 1 4 (or equivalently, C ≥ log 4), it holds that
E (x,y)∼πD exp 1 2 max log π(y | x) π D (y | x) , -C ≤ P πD (E) + 1 2 P πD (E c ) ≤ 1,
where we use
1 -p = (1 + √ p)(1 - √ p) ≤ 2(1 - √ p) for any p ∈ [0, 1].
Proof of Proposition J.2. Fix any N ≥ 1, α ≥ 0. By definition, for any π ∈ Π,
L - C (π) = n i=1 max 0, log π D (y i | x i ) π(y i | x i ) -C ≥ (log N -C) i ∈ [n] : log π D (y i | x i ) π(y i | x i ) ≥ log N = n(log N -C) • Cov N (π D π),
where we recall that (see Eq. ( 13))
Cov N (π D π) = 1 n t ∈ [n] : π D (y t | x t ) π(y t | x t ) ≥ N .
Then, by Lemma J.2 (stated and proved below), it holds that with probability at least 1 -δ, for any π ∈ Π,
Cov N (π D π) ≥ 1 2 Cov e 2α N (π D π) - 8 log(2N ∞ (Π, α)/δ) n .
Rescaling N ← e -2α N and reorganizing completes the proof.
Lemma J.2. For any model π, π ′ , we consider the quantities
Cov N (π ′ π) = 1 n t ∈ [n] : π ′ (y t | x t ) π(y t | x t ) ≥ N , Cov πD N (π ′ π) = P πD π ′ (y | x) π(y | x) ≥ M .
Fix α ≥ 0 and model π. With probability at least 1δ, for any π ∈ Π, it holds that
Cov N (π π) ≥ 1 2 Cov πD e 2α N (π π) - 8 log(2N ∞ (Π, α)/δ) n .
Similarly, with probability at least 1 -δ, for any π ∈ Π, it holds that
Cov N (π π) ≤ 2 Cov πD e -2α N (π π) + 8 log(2N ∞ (Π, α)/δ) n .
Proof of Lemma J.2. We only prove the first inequality. Let Π ′ ⊆ Π be an α-covering of Π with |Π ′ | = N ∞ (Π, α). Then, by Freedman's inequality (Lemma H.2) and union bound, it holds that with probability at least 1 -δ, for any π ′ ∈ Π ′ ,
Cov e α N (π π ′ ) ≥ 1 2 Cov πD e α N (π π ′ ) -ε stat , where we denote ε stat = 8 log(2|Π ′ |/δ) n .
Then, note that for any π ∈ Π, there exists
π ′ ∈ Π ′ such that | log π(y | x) -log π ′ (y | x)| ≤ α for ∀x, y, we know t ∈ [n] : π(y t | x t ) π ′ (y t | x t ) ≥ e α N ⊆ t ∈ [n] : π(y t | x t ) π(y t | x t ) ≥ N and hence Cov e α N (π π ′ ) ≤ Cov N (π π). Similarly, Cov πD e α N (π π ′ ) ≥ Cov πD e 2α N (π π).
Hence, under the above event, it holds that
Cov N (π π) ≥ Cov e α N (π π ′ ) ≥ 1 2 Cov πD e α N (π π ′ ) -ε stat ≥ 1 2 Cov πD e 2α N (π π) -ε stat .
Since π ∈ Π is arbitrary, the proof is hence completed.
this section cite: []

Section: J.3 PROOF OF THEOREM G.1 (COVERAGE FOR MLE WITH CONVEX CLASSES)
Let α ≥ 0, N ′ ≥ 1, N ≥ 2e 2α N ′ be fixed. By definition and concavity of θ → π θ (y | x), we know θ ⋆ is an optimal solution of the following concave problem
θ ⋆ ∈ arg max θ∈Θ E (x,y)∼πD [log π θ (y | x)].
Hence, the optimality of θ ⋆ implies that
θ -θ ⋆ , -E πD [∇ log π θ ⋆ (y | x)] ≥ 0, ∀θ ∈ Θ. Consider the function F (θ) = E πD π θ (y |x) π θ ⋆ (y |x) -1, which is also concave by Assumption G.1. For any θ ∈ Θ, θ -θ ⋆ , -∇F (θ ⋆ ) = θ -θ, -E πD ∇π θ ⋆ (y | x) π θ ⋆ (y | x) = θ -θ, -E πD [∇ log π θ ⋆ (y | x)] ≥ 0.
Therefore, F attains its maximum over Θ at θ ⋆ , i.e., F (θ) ≤ F (θ ⋆ ) = 0 for any θ ∈ Θ.
Similarly, it is also clear that θ → n i=1 log π θ (y i | x i
) is concave, and hence π = π θ , where θ ∈ Θ satisfies
θ -θ, n i=1 -∇ log π θ (y i | x i ) ≥ 0, ∀θ ∈ Θ.
In particular, we consider the function
F (θ) := n i=1 π θ (y i | x i ) π θ (y i | x i )-1 .
Under Assumption G.1, F is concave, and for any θ ∈ Θ,
θ -θ, -∇ F ( θ) = θ -θ, - n i=1 ∇π θ (y i | x i ) π θ (y i | x i ) = θ -θ, n i=1 -∇ log π θ (y i | x i ) ≥ 0.
Therefore, F attains its maximum over Θ at θ, and in particular,
F (θ ⋆ ) ≤ F ( θ) = 0. This implies n i=1 π θ ⋆ (y i | x i ) π(y i | x i ) -log π θ ⋆ (y i | x i ) π(y i | x i ) -1 ≤ n i=1 log π(y i | x i ) - n i=1 log π θ ⋆ (y i | x i ).(54)
In the following, we use that N ≥ 2. Note that x-log x-1 ≥ 0 for any x > 0, and x → x-log x-1 is increasing for x ≥ 1. Therefore, Eq. ( 54) implies that
(N -log N -1) • n • Cov N (π θ ⋆ π) ≤ L n ( π) -L n (π θ ⋆ ).(55)
Then, by Lemma J.2, we have with probability at least 1 -δ, for all π ∈ Π,
Cov N (π θ ⋆ π) ≥ 1 2 • P πD π θ ⋆ (y | x) π(y | x) ≥ e 2α N - log(N ∞ (Π, α)/δ) n , ∀π ∈ Π.
Further, by Lemma H.3, the following holds with probability at least 1 -δ: For any θ ∈ Θ,
L n (π θ ) -L n (π θ ⋆ ) = n i=1 log π θ (y i | x i ) π θ ⋆ (y i | x i ) ≤ n log E πD π θ (y | x) π θ ⋆ (y | x) + inf ϵ≥0 {log(N ∞ (Π, ϵ)/δ) + 2nϵ} ≤ inf ϵ≥0 {log(N ∞ (Π, ϵ)/δ) + 2nϵ}, where we use E πD π θ (y |x) π θ ⋆ (y |x) = F (θ) + 1 ≤ 1 for any θ ∈ Θ.
By union bound, we have shown that with probability at least 1 -2δ,
P πD π θ ⋆ (y | x) π(y | x) ≥ e 2α N ≲ log(N ∞ (Π, α)/δ) n + 1 N inf ϵ≥0 log N ∞ (Π, ϵ) n + ϵ .
Published as a conference paper at ICLR 2026 Note that
Cov e 2α N N ′ ( π) = P πD π D (y | x) π(y | x) ≥ e 2α N N ′ ≤ P πD π θ ⋆ (y | x) π(y | x) ≥ e 2α N + P πD π D (y | x) π θ ⋆ (y | x) ≥ N ′ .
Therefore, the proof is completed by rescaling N ← N e -2α /N ′ , δ ← δ 2 and combining the inequalities above.
this section cite: []

Section: J.4 PROOFS FOR SUPPORTING RESULTS

this section cite: []

Section: Proof of Proposition
G.1 (a). Assume that B ≥ log(5n) and n ≥ d ≥ 2. Consider X =⊥, Y = [d]
and let the feature map be given by ϕ(y) = Be y for y ∈ Y, where (e 1 , . . . , e d ) is the coordinate basis of R d . We consider Θ = θ ∈ R d : θ ∞ ≤ 1 , and we set
θ ⋆ = log(4n) 2B •   e 1 - d j=2 e j   .
Then it holds that
π D (1) = 4n d -1 + 4n , π D (y) = 1 d -1 + 4n , ∀y > 1.
Given the dataset D = {y 1 , • • • , y n }, we consider the random variables n y = |{i ∈ [n] :
y i = y}|. Note that under D ∼ π D , it holds that E y>1 n y = E n t=1 I{y t = 1} ≤ n(d -1) d -1 + 4n ≤ d -1 4 .
In particular, with probability at least 0.5, it holds that y>1 n y ≤ d-1 2 , i.e., the set Y 0 := {y ∈ [d] : n y = 0} has cardinality at least d-1 2 .
In the following, we condition on this event analyze the MLE θ. By the definition of MLE,
θ ∈ arg max θ∈Θ -n log   y∈[d] e Bθy   + B y∈[d] n y θ y .
We denote p y := π θ (y) = e B θy i∈[d] e B θ i . Then, the KKT conditions imply that for each y ∈ [d], either p y = ny n , or θ y = -1 and p y ≥ ny n , or θ y = 1 and p y ≤ ny n . In particular, for any y ∈ Y 0 , p y > 0 = ny n , and hence it must hold that θ y = -1. Then, because y∈[d] p y = 1 = y∈[d] ny n , there must exist j ∈ [d] such that p j < nj n , and by the KKT condition we have θ j = 1. Therefore, for any
y ∈ Y 0 , it holds that p y ≤ e -B e -B +e B ≤ 1 e 2B , and in particular πD(y) π θ (y) ≥ e 2B 4n+d-1 ≥ e B . This implies that Cov e B (π θ ) = P πD π D (y) π θ (y) ≥ e B ≥ P πD (Y 0 ) ≥ d -1 2(d -1 + 4n) ≥ d -1 10n .
This is the desired lower bound.
Proof of Proposition G.1 (b). Let ϵ = c 0 d n and p = c0ϵ 2 log N for a sufficiently small absolute constant c 0 > 0, X = {0, 1, • • • , d}, Y = {0, 1}
, and the distribution µ be given by µ(0
) = p, µ(1) = • • • = µ(d) = 1-p d . Let the data distribution π D be π D (• | i) = Ber(1/2) for i ∈ [d] and π D (1 | 0) = 1. For any θ ∈ Θ := {+1, -1} d , we define π θ as π θ (• | 0) = Ber 1 N , π θ (• | i) = Ber 1 + ϵθ i 2 , ∀i ∈ [d].
Consider the model class Π = {π D } ∪ {π θ : θ ∈ Θ}. Note that for any θ ∈ Θ,
Cov N (π D π θ ) ≥ µ(0) = p.
Then, we can calculate
L n (π θ ) -L n (π D ) = -C(0, 1) log N + i∈[d] [C(i, 1) log(1 + ϵθ i ) + C(i, 0) log(1 -ϵθ i )],
where we denote C(x, y) = |{t ∈ [n] : (x t , y t ) = (x, y)}|. We further write C(x) = C(x, 0) + C(x, 1). Taking maximum over θ ∈ Θ = {-1, 1} d gives
max θ∈Θ L n (π θ ) -L n (π D ) = -C(0) log N + 1 2 i∈[d] |C(i, 1) -C(i, 0)| log 1 + ϵ 1 -ϵ + C(i) log(1 -ϵ 2 ) ≥ -C(0) log N -nϵ 2 + ϵ 2 i∈[d] |C(i, 0) -C(i, 1)|,
In the following, we denote ∆ i = C(i, 1) -C(i, 0) and ∆ := i∈[d] ∆ i . Note that for any i ∈ [d], condition on C(i), ∆ i is a sum of C(i) i.i.d. random variables drawn from Unif({-1, 1}), and hence
E[(∆ i ) 2 | C(i)] = C(i), E[|∆ i | | C(i)] ≥ C(i) 2 ,
where we apply Khintchine's inequality. In addition, we note that C(i) ∼ B(n, q) is a binomial random variable, where q = 1-p d . Hence, E[C(i)] = nq, and to lower bound E C(i), we invoke Lemma J.3 (stated and proven in the sequel) to show that E C(i
) ≥ √ nq 1 -1-q 2nq ≥ √ nq 2
(because n ≥ 2d and hence nq ≥ 1). Therefore,
E[∆] = i∈[d] E[|∆ i |] ≥ 1 √ 2 i∈[d] E[ C(i)] ≥ d √ nq 2 √ 2 ,
and we can also bound E(∆
) 2 ≤ d i∈[d] E(∆ i ) 2 = d i∈[d] E[C(i)] = dn(1 -p) = d 2 nq.
Then, by Paley-Zygmund inequality, it holds that
P(∆ > b E[∆]) ≥ (1 -b) 2 (E[∆]) 2 E[∆ 2 ] ≥ (1 -b) 2 8 , ∀b ∈ [0, 1].
We choose b = 1 -√ 0.88 to be a numeric constant so that P(∆ > b E[∆]) ≥ 0.11. By Markov's inequality, it also holds that P(C(0) ≥ 100np) ≤ 0.01. In the following, we condition on the event
E = {∆ > b E[∆]} ∩ {C(0) ≤ 100np} (note that P(E) ≥ 0.1). Then, we have max θ∈Θ L n (π θ ) -L n (π D ) ≥ -C(0) log N -nϵ 2 + ϵ 2 ∆ > bϵ √ nd 8 -100np log N -nϵ 2 ≥ 0,
as long as c 0 ≤ 10 -4 . This implies that there exists θ ∈ Θ such that π = π θ , and hence Cov N ( π) ≥ p. This is the desired lower bound.
Lemma J.3. For non-negative random variable Z, it holds that E[ √ Z] ≥ E[Z] 1 -Var[Z] 2(E[Z]) 2 .
Proof of Lemma J.3. Note that the inequality
√ u ≥ 3u-u 2 2 holds for u ≥ 0. Setting u = Z E[Z]
and taking expectation completes the proof.
this section cite: []

Section: K PROOFS FOR AUTOREGRESSIVE LINEAR MODELS
K.1 ORGANIZATION This section contains proofs for all of the results in Sections 4 to 6 concerning autoregressive linear models (3). We begin with the proof of Theorem 4.2 (MLE for autoregressive linear models).
We then present the proofs for various SGD methods, starting with vanilla SGD (Proposition 5.1; upper and lower bounds), followed by normalized SGD (Theorem 5.1), test-time training (Theorem 6.1), and expert-guided gradient normalization (Theorem G.2). The final subsection provides an additional lower bound, showing that the dependence on the parameter σ 2 ⋆ is necessary in high dimension.
Throughout this section, all upper bounds are derived under Assumptions 2.1 and 2.2, i.e., we assume that Θ ⊆ B 2 (1), ϕ : X × V ⋆ → B 2 (R), and π D = π θ ⋆ is realized by some parameter θ ⋆ ∈ Θ.
Notation and preliminaries. For any f : X × V ⋆ → R and dataset D = {(x i , y i 1:H )} i∈[n] , we write
E D [f ] := 1 n n i=1 f (x i , y i 1:H ),
For notational simplicity, we denote
ϕ θ (x, y 1:h-1 ) = E y h ∼π θ (•|x,y 1:h-1 ) [ϕ(x, y 1:h )],and
ϕ ⋆ (x, y 1:h ) := ϕ(x, y 1:h ) -ϕ θ ⋆ (x, y 1:h-1 ), Var πD (x, y 1:h-1 ) := E y h ∼π θ (•|x,y 1:h-1 ) ϕ ⋆ (x, y 1:h ) 2 .
Then, by definition,
∇ log π θ (y 1:H | x) = H h=1 ϕ(x, y 1:h ) -ϕ θ (x, y 1:h-1 ) = H h=1 ϕ ⋆ (x, y 1:h ) + H h=1 ϕ θ ⋆ (x, y 1:h-1 ) -ϕ θ (x, y 1:h-1 ) ,(56)
and it holds that
σ 2 ⋆ = E πD H h=1 Var πD (x, y 1:h-1 ) .
In addition, we write
ϵ θ (x, y 1:h-1 ) = D KL (π D (• | x, y 1:h-1 ) π θ (• | x, y 1:h-1 )).(57)
For any θ ∈ Θ, the key quantity of interest is D seq,N (π D π θ ), defined via
D seq,N (π D π θ ) = E πD min log N, H h=1 D KL (π D (• | x, y 1:h-1 ) π θ (• | x, y 1:h-1 )) = E πD min log N, H h=1 ϵ θ (x, y 1:h-1 ) . By Proposition F.10, it holds that Cov N (π θ ) ≤ 2 log N -1 D seq,N (π D π θ )
. Further, by concavity, we have
ϵ θ (x, y 1:h-1 ) ≤ ϕ θ (x, y 1:h-1 ) -ϕ θ ⋆ (x, y 1:h-1 ), θ -θ ⋆ .
(58) By Lemma H.4, it holds that
ϕ θ ⋆ (x, y 1:h-1 ) -ϕ θ (x, y 1:h-1 ) ≤ 4 Var πD (x, y 1:h-1 ) • ϵ θ (x, y 1:h-1 ) + 8Bϵ θ (x, y 1:h-1 ).(59)
K.2 PROOF OF THEOREM 4.2 (COVERAGE FOR MLE FOR AUTOREGRESSIVE LINEAR MODELS) We prove the following slightly stronger result. Theorem 4.2 follows immediately by combining Theorem K.1 and Proposition F.10. Theorem K.1. Suppose that Assumption 2.2 holds. Then the MLE π achieves
E D [D seq,N (π D π)] ≲ σ 2 ⋆ log N n + B 2 log N n ,
for any parameter N ≥ 2, where the divergence D seq,N (• •) is defined in Proposition F.10.
We begin with two central technical lemmas, which are proven in the sequel. The first lemma is a consequence of the fact that the MLE π = π θ maximizes the empirical likelihood, i.e.,
θ = arg max θ∈Θ E D [log π θ (y 1:H | x)],(60)
where we recall that for any dataset D = {(
x i , y i 1:H )} i∈[n] , we write E D [f ] := 1 n n i=1 f (x i , y i 1:H ) for any f : X × V ⋆ → R.
Lemma K.1 shows that in expectation, a sum of per-step conditional KL divergences between π D and π is bounded (this does not imply a bound on sequence-level KL divergence, since θ is dependent on the data D).
Lemma K.1. Recall that we denote ϵ θ (x, y 1:h-1 ) = D KL (π D (• | x, y 1:h-1 ) π θ (• | x, y 1:h-1 )). Fur- ther, define E 1 := E D H h=1 ϵ θ (x, y 1:h-1 )(61)
Then it holds that E[E 1 ] ≤ 2σ⋆ √ n . Define A := log N .
The next lemma is a uniform convergence-like argument which shows that the quantity E 1 above-when truncated at a certain level A-concentrates around its expectation up to a multiplicative factor. This argument is inspired by the fractional covering method introduced in Chen et al. (2024a); Chen & Rakhlin (2025).
Lemma K.2. Fix any ∆ ∈ (0, 1 200B ], δ ∈ (0, 1), and let J = exp 1 ∆ 2 + 2 log(1/δ). Let Θ ′ := {θ 1 , • • • , θ J }, where θ 1 , • • • , θ J ∼ N (0, ∆ 2 I) are sampled i.i.d. Then the following holds with probability at least 1 -δ over the randomness of Θ ′ and D:
(1) For any j ∈ [J], it holds that
E πD min A, H h=1 ϵ θj (x, y 1:h-1 ) ≤ 2 E D min A, H h=1 ϵ θj (x, y 1:h-1 ) + 8A log(4J/δ) n .
(2) There exists j ∈ [J] such that
E πD min A, H h=1 ϵ θ (x, y 1:h-1 ) ≤ 2 E πD min A, H h=1 ϵ θj (x, y 1:h-1 ) + C∆ 2 σ 2 ⋆ , (62
)
and
E D min A, H h=1 ϵ θj (x, y 1:h-1 ) ≤ 2 E D min A, H h=1 ϵ θ (x, y 1:h-1 ) + C∆ 2 E D H h=1 Var πD (x, y 1:h-1 ) ,(63)
where C = 1000 is a numeric constant.
Above, the distribution of π θ under θ ∼ N (0, ∆ 2 I) can be viewed as a fractional cover for Π in the sense of Chen et al. (2024a). In particular, working with the fractional cover offers the following technical advantages:
• The fractional cover N (0, ∆ 2 I) incurs error σ 2 ⋆ ∆ 2 (see Lemma K.3) that depends only on the variance at the ground-truth parameter θ ⋆ . This contrasts with classical coverings, which enforce a uniform bound for all θ ∈ Θ.
• For Θ = B d 2 (1), the L ∞ covering number of Π (cf. Definition 4.1) scales with the dimension d. A standard approach to deriving dimension-independent bounds is to apply symmetrization techniques and use a data-dependent L 2 covering to show uniform convergence. In contrast, our fractional-covering approach avoids the (technically subtle) symmetrization step because the cover {θ 1 , . . . , θ J } ∼ N (0, ∆ 2 I) is drawn independently of the dataset D.
Completing the proof. Equipped with the lemmas above, we complete the proof as follows. First, we condition on the success event E of Lemma K.2, and let j ∈ [J] be an index such that (62) and ( 63) hold. Then, we can upper bound (recall that A = log N and D seq,N (• •) is defined in Proposition F.10)
D seq,N π D π θ = E πD min A, H h=1 ϵ θ (x, y 1:h-1 ) ≤ 2 E πD min A, H h=1 ϵ θj (x, y 1:h-1 ) + C∆ 2 σ 2 ⋆ ≤ 4 E D min A, H h=1 ϵ θj (x, y 1:h-1 ) + 16A log(4J/δ) n + C∆ 2 σ 2 ⋆ ≤ 8 E D min A, H h=1 ϵ θ (x, y 1:h-1 ) + 4C∆ 2 E D H h=1 Var πD (x, y 1:h-1 ) + 16A log(4J/δ) n + C∆ 2 σ 2 ⋆ .
where the first inequality uses (62), the second inequality uses Lemma K.2 (1), and the third inequality uses (63). Therefore, we denote σ 2 (D) := E D H h=1 Var πD (x, y 1:h-1 ) , and we have shown that for any δ ∈ (0, 1), any ∆ ∈ (0, 1 200B ], it holds that
P D∼πD D seq,N π D π θ ≥ C 1 E 1 + ∆ 2 σ 2 (D) + ∆ 2 σ 2 ⋆ + A n 1 ∆ 2 + log(1/δ) ≤ δ,
where C 1 > 0 is an absolute constant.
Since δ ∈ (0, 1) is arbitrary, integrating the tail inequality above yields the following bound on the expected value:
E D seq,N π D π θ ≤ C 1 E[E 1 ] + ∆ 2 E[σ 2 (D)] + ∆ 2 σ 2 ⋆ + A n 1 ∆ 2 + 1 ≤ 2C 1 σ 2 ⋆ n + ∆ 2 σ 2 ⋆ + A n∆ 2 , ∀0 < ∆ ≤ 1 200B . Choosing ∆ = min 1 200B , A σ 2 ⋆ n 1/4
completes the proof. The coverage upper bound follows immediately from Proposition F.10.
K.2.1 PROOFS FOR SUPPORTING LEMMAS Proof of Lemma K.1. Recall that π = π θ , where θ = arg max θ∈Θ E D [log π θ (y 1:H | x)].
Then by concavity of the log-likelihood, we have that
E D ∇ log π θ (y 1:H | x) , θ -θ ≤ 0, ∀θ ∈ Θ.
Using the expression (56) and θ ⋆ ∈ Θ, we know
E D H h=1 ϕ(x, y 1:h ) -ϕ θ (x, y 1:h-1 ) , θ ⋆ -θ ≤ 0.
Therefore, combining the inequality above with Eq. ( 58), we have
E D H h=1 ϵ θ (x, y 1:h-1 ) = E D H h=1 D KL (π D (• | x, y 1:h-1 ) π(• | x, y 1:h-1 )) ≤ E D H h=1 ϕ θ ⋆ (x, y 1:h-1 ) -ϕ θ (x, y 1:h-1 ), θ ⋆ -θ ≤ E D H h=1 ϕ θ ⋆ (x, y 1:h-1 ) -ϕ(x, y 1:h ), θ ⋆ -θ ≤ 2 E D H h=1 ϕ ⋆ (x, y 1:h ) =: E ′ 1 ,
where we recall that ϕ ⋆ (x, y 1:h ) := ϕ(x, y 1:h ) -ϕ θ ⋆ (x, y 1:h-1 ). By definition, it holds that E πD [ϕ ⋆ (x, y 1:h ) | x, y 1:h-1 ] = 0, and hence
E(E ′ 1 ) 2 = E E D H h=1 ϕ ⋆ (x, y 1:h ) 2 = 1 n E πD H h=1 ϕ ⋆ (x, y 1:h ) 2 = 1 n E πD H h=1 ϕ ⋆ (x, y 1:h ) 2 = σ 2 ⋆ n .
This gives the desired upper bound.
Proof of Lemma K.2. By Freedman's inequality (Lemma H.2) and the union bound, it follows that (1) holds with probability at least 1 -δ 2 . In the remainder of the proof, we prove (2). Define the following weight function α = α θ :
X × V ⋆ → [0, 1]: 9 α θ (x, y 1:h-1 ) =      1, j≤h-1 ϵ θ (x, y 1:j ) ≤ A, 0, j<h-1 ϵ θ (x, y 1:j ) ≥ A, A-j<h-1 ϵ θ (x,y1:j ) ϵ θ (x,y 1:h-1 )
, otherwise.
We also define F(a, b) = |a -b| -1 2 a. The properties of F(•, •) and the weight function α are summarized in Lemma K.4 (stated and proven in the sequel).
Then, by Lemma K.4, it holds that for any θ ∈ Θ,
E πD min A, H h=1 ϵ θ (x, y 1:h-1 ) ≤ 2 E πD min A, H h=1 ϵ θ (x, y 1:h-1 ) + 2 E πD H h=1 α(x, y 1:h-1 )F ϵ θ (x, y 1:h-1 ), ϵ θ (x, y 1:h-1 ) ,
and
E D min A, H h=1 ϵ θ (x, y 1:h-1 ) ≤ 2 E D min A, H h=1 ϵ θ (x, y 1:h-1 ) + E D H h=1 α(x, y 1:h-1 )F ϵ θ (x, y 1:h-1 ), ϵ θ (x, y 1:h-1 ) ,
Therefore, it remains to control the error H h=1 α(x, y 1:h-1 )F ϵ θ (x, y 1:h-1 ), ϵ θ (x, y 1:h-1 ) under both E πD [•] and E D [•]. We next state the following lemma (proven in the sequel), which leverages the structure of Gaussian distribution. This result can be viewed as a fractional covering number bound (Chen et al., 2024a) and hence generalizes the argument of Chen & Rakhlin (2025, Proposition C.4).
Lemma K.3. For any K ≥ 1, ∆ ∈ (0, 1 100KB ], θ ∈ B 2 (1), distributions ρ 1 , • • • , ρ K over Z := X × V ⋆ , and weight function α : Z → [0, 1], it holds that
-log P θ ′ ∼N (0,∆ 2 ) ∀i ∈ [K], E z∼ρi α(z)F(ϵ θ (z), ϵ θ ′ (z)) ≤ 70K 2 ∆ 2 E z∼ρi Var πD (z) ≤ 1 ∆ 2 + 2,
where we recall that F(a, b) = |a -b| -1 2 a.
In the following, we apply Lemma K.3 with K = 2, parameter θ = θ, weight function α, and the distributions ρ 1 , ρ 2 defined as follows:
• Let ρ 1 be the distribution of
x ′ = (x, y 1:h-1 ) under x ∼ µ, y 1:H ∼ π D (• | x)
and h ∼ Unif([H]).
• Let ρ 2 be the distribution of x ′ = (x t , y t 1:h-1 ) under t ∼ Unif([n]) and h ∼ Unif([H]). By definition, it holds that
E z∼ρ1 α(z)F(ϵ θ (z), ϵ θ ′ (z)) = 1 H E πD H h=1 α(x, y 1:h-1 )F ϵ θ (x, y 1:h-1 ), ϵ θ (x, y 1:h-1 ) , E z∼ρ1 Var πD (z) = 1 H E πD H h=1 Var πD (x, y 1:h-1 ) = σ 2 ⋆ H , E z∼ρ2 α(z)F(ϵ θ (z), ϵ θ ′ (z)) = 1 H E D H h=1 α(x, y 1:h-1 )F ϵ θ (x, y 1:h-1 ), ϵ θ (x, y 1:h-1 ) , E z∼ρ2 Var πD (z) = 1 H E D H h=1
Var πD (x, y 1:h-1 ) .
Now, consider the following set for any θ ∈ Θ:
Θ + θ := ∀i ∈ {1, 2}, E z∼ρi α(z)F(ϵ θ (z), ϵ θ ′ (z)) ≤ 300∆ 2 E z∼ρi Var πD (z) . By Lemma K.3, it holds that q(θ) := P θ ′ ∼N (0,∆ 2 I) (θ ′ ∈ Θ + θ ) ≥ exp - 1 ∆ 2 -2 , ∀θ ∈ Θ, ∀∆ ∈ (0, 1 200B ].
Therefore, we have
P ∀j ∈ [J], θ j ∈ Θ + θ | θ = P θ1,••• ,θ J ∼N (0,∆ 2 I) ∀j ∈ [J], θ j ∈ Θ + θ ≤ (1 -q( θ)) J ≤ exp -Jq( θ) ≤ δ 2 ,
and hence P ∃j ∈ [J], θ j ∈ Θ + θ ≥ 1 -δ 2 . The proof of Lemma K.2 (2) is thus completed, as Eq. ( 62) and Eq. ( 63) hold for any j ∈ [J] such that θ j ∈ Θ + θ .
Proof of Lemma K.3. We first fix any h ∈ [H] and z = (x, y 1:h-1 ) ∈ X × V h-1 and analyze the behavior of log π θ ′ (y h | z) under θ ′ ∼ N (θ, ∆ 2 I).
this section cite: []

Section: By definition, we have π
θ ′ (y h | z) ∝ y h π θ (y h | z) • exp( θ ′ -θ, ϕ(z, y h ) ), i.e., log π θ ′ (y h | z) -log π θ (y h | z) = θ ′ -θ, ϕ(z, y h ) -log E y h ∼π θ (•|z) exp( θ ′ -θ, ϕ(z, y h ) ).
Therefore,
ϵ θ (z) -ϵ θ ′ (z) = D KL (π D (y h = • | z) π θ (y h = • | z)) -D KL (π D (y h = • | z) π θ ′ (y h = • | z)) = E πD(•|z) θ ′ -θ, ϕ(z, y h ) -log E y h ∼π θ (•|z) exp( θ ′ -θ, ϕ(z, y h ) ) = θ ′ -θ, ϕ θ ⋆ (z) -ϕ θ (z) -log E y h ∼π θ (•|z) exp θ ′ -θ, ϕ(z, y h ) -ϕ θ (z) , where we recall that ϕ θ (z) = E y h ∼π θ (•|z) [ϕ(z, y h )].
In the following, we denote ϕ θ (z, y h ) := ϕ(z, y h ) -ϕ θ (z), and
E + θ ′ (z) := log E y h ∼π θ (•|z) exp( θ ′ -θ, ϕ θ (z, y h ) ), E - θ ′ (z) := θ ′ -θ, ϕ θ ⋆ (z) -ϕ θ (z) .
We first bound E + θ ′ (z). By definition, we have
E + θ ′ (z) = D KL (π θ (• | z) π θ ′ (• | z)) ≥ 0.
Further, using Jensen's inequality, for any z ∈ Z, we have
E θ ′ ∼N (θ,∆ 2 I) E + θ ′ (z) ≤ log E θ ′ ∼N (θ,∆ 2 I) E y h ∼π θ (•|z) [exp( θ ′ -θ, ϕ θ (z, y h ) )] = log E y h ∼π θ (•|z) exp 1 2 ∆ 2 ϕ θ (z, y h ) 2 ≤ ∆ 2 E y h ∼π θ (•|z) ϕ θ (z, y h ) 2 ,
where the last inequality follows from e t ≤ 1 + 2t for t ∈ [0, 1]. Further, using Lemma H.4, we have
E y h ∼π θ (•|z) ϕ θ (z, y h ) 2 = E y h ∼π θ (•|z) ϕ(z, y h ) -ϕ θ (z) 2 ≤ 3 E y∼πD(•|z) ϕ(z, y h ) -ϕ θ ⋆ (z) 2 + 16B 2 D KL (π D (• | z) π θ (• | z)) = 3Var πD (z) + 16B 2 ϵ θ (z).
Next, we bound
|E - θ ′ (z)|. Under θ ′ ∼ N (θ, ∆ 2 I), it is clear that θ ′ -θ, ϕ θ ⋆ (z) -ϕ θ (z) ∼ N (0, ∆ 2 ϕ θ ⋆ (z) -ϕ θ (z) 2
) for any fixed z. Therefore, it holds that
E θ ′ ∼N (θ,∆ 2 I) E - θ ′ (z) = 2 π ∆ • ϕ θ ⋆ (z) -ϕ θ (z) ≤ ∆ • 4 Var πD (z) • ϵ θ (z) + 8Bϵ θ (z) ≤ 1 8K + 8B∆ ϵ θ (z) + 32K∆ 2 Var πD (z),
where the second line uses Eq. ( 59).
Combining the inequalities above and taking expectation of z ∼ ρ i , we know that for i ∈ [K], it holds that
E θ ′ ∼N (θ,∆ 2 I) E z∼ρi α(z)E + θ ′ (z) ≤ ∆ 2 E z∼ρi 3Var πD (z) + 16B 2 α(z)ϵ θ (z) , E θ ′ ∼N (θ,∆ 2 I) E z∼ρi α(z) E - θ ′ (z) ≤ E z∼ρi 32K∆ 2 Var πD (z) + 1 8K + 8B∆ α(z)ϵ θ (z) ,
and hence by Markov's inequality and ∆ ≤ 1 100KB , it holds that p := P θ ′ ∼N (θ,∆ 2 I) (θ ′ ∈ Θ -) ≥ 1 2 , where we denote Θ -= ∪ i∈[K] Θ - i , and
Θ - i := θ ′ ∈ R d : E z∼ρi α(z)|ϵ θ (z) -ϵ θ ′ (z)| ≥ E z∼ρi (6K + 64K 2 )∆ 2 Var πD (z) + 1 2 α(z)ϵ θ (z) . Note that D KL N (θ, ∆ 2 I) N (0, ∆ 2 I) = ∥θ∥ 2 2∆ 2 ≤ 1 2∆ 2 .
Hence, by data-processing inequality, we can bound q :=
P θ ′ ∼N (0,∆ 2 I) (θ ′ ∈ Θ -) as 1 2∆ 2 ≥ D KL N (θ, ∆ 2 I) N (0, ∆ 2 I) ≥ D KL (Ber(p) Ber(q)) = p log p q + (1 -p) log 1 -p 1 -q ≥ 1 2 log(1/q) -log 2.
This implies that -log q ≤ 1 ∆ 2 + 2, giving the desired result.
Lemma K.4. Suppose that a
1 , • • • , a H , b 1 , • • • , b H ≥ 0, A ≥ 0. Define F(a, b) = |a -b| -1 2 a. Let α h =      1, j≤h a j ≤ A, 0, j<h a j > A, A-j<h aj a h , otherwise.
Then clearly α h ∈ [0, 1] ∀h ∈ [H], and it holds that H h=1 α h a h = min A, H h=1 a h , and
min A, H h=1 a h ≤ 2 min A, H h=1 b h + 2 H h=1 α h F(a h , b h ),and
min A, H h=1 b h ≤ 2 min A, H h=1 a h + H h=1 α h F(a h , b h ).
Proof of Lemma K.4. Fix the sequence a 1 , • • • , a H . We first prove that
H h=1 α h a h = min A, H h=1 a h . (64
)
To do so, we consider two cases.
Case 1:
H h=1 a h ≤ A.
In this case, α h = 1∀h ∈ [H], and the equation holds trivially.
Case 2:
H h=1 a h > A.
In this case, we let ℓ ∈ [H] be the maximal index such that α ℓ > 0. Then, by definition, j<ℓ a j ≤ A and j≤ℓ a j > A, and
α ℓ = A-j<ℓ aj a ℓ . Hence, H h=1 α h a h = ℓ h=1 α h a h = j<ℓ a j + α ℓ a ℓ = A.
We also note that from the proof above, we also know that for any sequence (c 1 ,
• • • , c H ) such that c h ≥ a h for h ∈ [H], we have min A, H h=1 c h ≤ H h=1 α h c h . (65
)
Equipped with these results, we prove the inequalities in the lemma statement. We note that
H h=1 α h F(a h , b h ) = H h=1 α h |a h -b h | - 1 2 H h=1 α h a h , or equivalently, H h=1 α h |a h -b h | = H h=1 α h F(a h , b h ) + 1 2 min A, H h=1 a h .
Therefore,
min A, H h=1 a h = H h=1 α h a h ≤ min A, H h=1 b h + H h=1 α h |a h -b h | = min A, H h=1 b h + H h=1 α h F(a h , b h ) + 1 2 min A, H h=1 a h .
Re-organizing yields the first inequality. Similarly, we have
min A, H h=1 b h ≤ min A, H h=1 (a h + |a h -b h |) ≤ H h=1 α h (a h + |a h -b h |) = 3 2 min A, H h=1 a h + H h=1 α h F(a h , b h ).
The proof is hence completed.
this section cite: []

Section: K.3 PROOF OF PROPOSITION 5.1 (VANILLA SGD: COVERAGE UPPER BOUND)
We first invoke the following standard lemma.
Lemma K.5. Suppose that the sequence (θ t , g t ) t≥1 satisfies θ t+1 = Proj Θ (θ t + ηg t ) for t ≥ 1.
Then it holds that for any θ ⋆ ∈ Θ, T ≥ 1,
T t=1 -g t , θ t -θ ⋆ ≤ θ ⋆ -θ 0 2 2η + η 2 T t=1 g t 2 . (66
)
Specializing Lemma K.5 to the SGD update (8) and taking expectation, we have
E T t=1 -∇ log π θ t (y t | x t ), θ t -θ ⋆ ≤ 2 η + η 2 E T t=1 ∇ log π θ t (y t | x t ) 2 . (67
)
Note that (x t , y t ) | θ t ∼ π D , and hence
E[∇ log π θ t (y t | x t ) | θ t ] = E (x,y)∼πD [∇ log π θ t (y | x)] = ∇ θ D KL (π D π θ )| θ=θ t .
Further, by convexity, it holds that for any θ ∈ Θ,
G(θ) := E πD [ -∇ log π θ (y | x), θ -θ ⋆ ] = ∇ θ D KL (π D π θ ), θ -θ ⋆ ≥ D KL (π D π θ ).
Therefore, we have
E T t=1 D KL (π D π θ t ) ≤ E T t=1 G(θ t ) ≤ 2 η + η 2 E T t=1 E (x,y)∼πD ∇ log π θ t (y | x) 2 .
On the other hand, using the fact that log π θ (y | x) is concave and (HB 2 )-smooth (i.e., -HB
2 I ∇ 2 log π θ (y | x) 0), ∇ log π θ (y | x) -∇ log π θ ⋆ (y | x) 2 ≤ HB 2 • θ -θ ⋆ , ∇ log π θ ⋆ (y | x) -∇ log π θ (y | x)
Taking expectation of (x, y) ∼ π D and using the fact that E πD [∇ log π θ ⋆ (y | x)] = 0, we have
E πD ∇ log π θ (y | x) -∇ log π θ ⋆ (y | x) 2 ≤ HB 2 • G(θ), ∀θ ∈ Θ. Further, note that E πD ∇ log π θ ⋆ (y | x) 2 = σ 2 ⋆ , it holds that E πD ∇ log π θ (y | x) 2 ≤ 2σ 2 ⋆ + 2HB 2 • G(θ), ∀θ ∈ Θ.(68)
Combining the inequalities above, we can conclude that
E T t=1 G(θ t ) ≤ 2 η + ηHB 2 E T t=1 G(θ t ) + ηT σ 2 ⋆ .
We conclude that as long as η ≤ 1 2HB 2 , it holds
4 η + 2ηT σ 2 ⋆ ≥ E T t=1 G(θ t ) ≥ E T t=1 D KL (π D π θ t ) .
This is the desired upper bound.
Proof of Lemma K.5. A standard result (e.g., Hazan (2016)) is that because the projection operator Proj Θ is an contraction, we have that for all t ∈ [T ], the update satisfies
θ t -θ ⋆ 2 -θ t+1 -θ ⋆ 2 ≥ θ t -θ ⋆ 2 -θ t + ηg t -θ ⋆ 2 = 2η -g t , θ t -θ ⋆ -η 2 g t 2 . (69
)
Summing this inequality across steps t = 1, 2, • • • , T , telescoping, and taking expectation, we have
T t=1 -g t , θ t -θ ⋆ ≤ θ ⋆ -θ 0 2 -θ ⋆ -θ T +1 2 2η + η 2 T t=1 g t 2 . (70
)
This gives the desired upper bound.
K.4 PROOF OF PROPOSITION 5.1 (VANILLA SGD: COVERAGE LOWER BOUND) In the following, we construct X = [ 8 HB , +∞) {-, +}, V = {-1, 0, 1} and Θ = B 2 (1) with d = 2. We fix parameters B ≥ B ≥ 1.
this section cite: ['b18']

Section: Construction of ϕ.
We first construct a map v : X × V → R 2 as follows. For any η ≥ 8 HB , we define α η = ηHB 2(ηHB-1) ≤ 5 8 and let
v(η, 0) = [1; 0], v(η, 1) = [α η ; 1 -α 2 η ], v(η, -1) = [α η ; -1 -α 2 η ].
We further define
v(+, a) = 1 B [Ba; 0], v(-, a) = 1 B [0; Ba] ∀a ∈ V = {-1, 0, 1}.
For x ∈ X , y 1:h ∈ V h , we define ϕ(x, y 1:h ) = Bv(x, y h ). 10   Under this construction of ϕ, we then prove the lower bound by considering two cases based on the value of η.
Lemma K.6. Suppose that η ≥ 8 HB , log N ≤ HB 8 , and B ≥ c B log(T H) for a large constant c B > 1. Then, with the distribution µ being supported on x = η and θ ⋆ = [1; 0], the following holds.
(1) The variance of such an instance is bounded: σ ⋆ ≤ 1.
(2) There exists θ 0 ∈ Θ such that with probability at least 0.5, the SGD sequence (θ t ) satisfies Cov N (π θ t ) ≥ 1 -1 2T for all t ∈ [T ]. Lemma K.7. Suppose that η ≤ 8 HB , log N ≤ HB 8 , and B ≥ B ≥ c B log(T H) for a large constant c B > 1. Then, there exists distribution µ and θ ⋆ ∈ Θ such that the following holds.
(1) The variance of such an instance is bounded: σ ⋆ ≤ 1.
(2) There exists θ 0 ∈ Θ such that with probability at least 0.5, the SGD sequence (θ t ) satisfies
Cov N (π θ t ) ≥ c min 1, HB T • B 2 log N , ∀t ∈ [T ].
The proof of Proposition 5.1 (lower bound) is then completed by combining Lemma K.6 and Lemma K.7.
Proof of Lemma K.6. Fix the parameter η ≥ 8 HB . We denote η := η • HB and α = α η = η 2(η-1) ≤ 5 8 . Denote Under our construction, we have
v 0 = [1; 0], v 1 = [α; 1 -α 2 ], v -1 = [α; -1 -α 2 ].
π θ (y h | η, y 1:h-1 ) = exp(B θ, v y h ) a∈V exp(B θ, v a ) =: P θ (y h ).
We study the SGD update starting from θ 0 = v 1 . By definition, ϕ(η, y 1:h ) = Bv(η, y h ), and hence
∇ log π θ (y 1:H | η) = H h=1 Bv(η, y h ) -E a∼P θ [Bv(η, a)] = B H h=1 v y h -E a∼P θ [v a ] .
In the following, we denote
F (y 1:H ) := 1 H H h=1 v y h , F (θ) := E a∼π θ [v a ] = a∈V a exp(B θ, v a ) a∈V exp(B θ, v a )
.
Then, the SGD update can be written as
u t = θ t + η F (y t 1:H ) -F (θ t ) , θ t+1 = Proj Θ (u t ).
We make the following claims.
Claim 1. For a ∈ {-1, 0, 1} and θ -v a ≤ 1 16 , it holds that 1 -P θ (a) ≤ 2e -B/4 =: ϵ 1 and hence
F (θ) -v a ≤ 2ϵ 1 . Claim 2. Suppose that ϵ 1 ≤ min 1 4T H , 1 5HB 2 .
Then it holds that σ ⋆ ≤ 1. Further, with probability at least 0.5, it holds that F (y t 1:H ) = e 0 for all t ∈ [T ]. In the following, we condition on this event.
Claim 3. By definition, for a ∈ {-1, 1}, we have v a + η(v 0 -v a ) = η -1 and v -1 + v 1 = η η-1 v 0 . Claim 4. Let ϵ = 16ϵ 1 . Suppose that ϵ ≤ 1 16 . Then for a ∈ {-1, 1}, if θ t -v a ≤ ϵ, then it holds that θ t+1 -v -a ≤ ϵ.
Claim 5. Suppose that ϵ 1 ≤ 1 2T H and log N ≤ HB 8 . Then
Cov N (π D π θ ) ≥ 1 -1 2T for θ ∈ Θ such that min{ θ -v 1 , θ -v -1 } ≤ 1
16 . Combining the above claims, we know that there is a constant C such that as long as B ≥ c B log(T H), it holds that σ ⋆ ≤ 1. Further, under the success event of claim 2, it holds that for a ∈ {-1, 1}, θ t -v a ≤ 1 16 for all t ∈ [T ] such that 2 | t -a. Therefore, by Claim 5, this gives Cov N (π θ t ) ≥ 1 2 as long as log N ≤ HB 8 . Proof for Claims 1-5. To prove Claim 1, we note that θ, v a ≥ 1 -θ -v a ≥ 15 16 and for i = a, θ,
v i ≤ v a , v i + θ -v a ≤ α + 1
16 ≤ 11 16 . Therefore,
1 -P θ (a) ≤ i̸ =a e B⟨θ,vi⟩ e B⟨θ,va⟩ ≤ 2 e B/4 = ϵ 1 .
This completes the proof of Claim 1.
Next, we prove Claim 2. Recall that θ ⋆ = [1; 0] = v 0 . By Claim 1, we know 1 -P θ ⋆ (0) ≤ ϵ 1 , and hence
Var a∼P θ ⋆ [v a ] ≤ 5ϵ 1 . This implies σ 2 ⋆ = HB 2 Var a∼P θ ⋆ [v a ] ≤ 5HB 2 ϵ 1 ≤ 1. We also know P πD (y h = 0 ∀h ∈ [H]) = P θ ⋆ (0) H ≥ (1 -ϵ 1 ) H ≥ 1 -Hϵ 1 .
Therefore, taking the union bound, we know P(y t h = 0 ∀h ∈ [H], t ∈ [T ]) ≥ 1 -T Hϵ 1 ≥ 1 2 . This completes the proof of Claim 2. Furthermore, for any θ such that min{ θ -v 1 , θ -v -1 } ≤ 1 16 , as long as log N ≤ H(log(1ϵ 1 ) -log(ϵ 1 )), we have
Cov N (π D π θ ) ≥ 1 - 1 2n I{H log π D (0) -H log π θ (0) ≥ log N } ≥ 1 - 1 2n .
Published as a conference paper at ICLR 2026
In particular, this is ensured when log N ≤ HB 8 . This completes the proof of Claim 5. Claim 3 follows immediately from the definition of α, v 0 , v 1 and v -1 .
Finally, we prove Claim 4. Recall that u t := θ t + η F (y t 1:H ) -F (θ t ) . Then it holds that
u t -(η -1)v -a = u t -ηv 0 + (η -1)v a ≤ θ t -v a + η F (y t 1:H ) -v 0 + η F (θ t ) -v a ≤ ϵ + 2ηϵ 1 =: ϵ ′ . In particular, it holds that | u t -(η -1)| ≤ ϵ ′ and hence u t ≥ η -1 -ϵ ′ = (1 -2ϵ 1 )η -1 -ϵ ≥ η 2 ≥ 1. Therefore, θ t+1 = Proj Θ (u t ) = u t
∥u t ∥ , and we can bound
θ t+1 -v -a = u t -(η -1)v -a u t + v -a η -1 u t -1 ≤ u t -(η -1)v -a u t + |η -1 -u t | u t ≤ 2ϵ ′ u t ≤ 4ϵ ′ η = 4 η ϵ + 8ϵ 1 ≤ ϵ.
Proof of Lemma K.7. We again denote η = HBη ≤ 8. We choose θ ⋆ = [ 1 2 ; 1 2 ], and let the distribution µ be supported on {-, +}:
µ(+) = 1 -µ(-) = min 1, BH 512enB 2 log N . Note that for x ∈ {-, +}, π D (1 | x) = e B/2
e -B/2 +1+e B/2 , and hence 1 -π D (y 1 = 1 | x) ≤ 2e -B/2 . Therefore, similar to Case 1, we have the following claims. In the following, we condition on this event. We choose r ≤ 1 2 such that e rB = H 4 log N , and we let
θ 0 = [r -1 B ; 1 4 ].
Claim 2. For any θ ∈ Θ ⊂ R 2 , it holds that 1 -P θ (1 | +) ≤ 2 e θ[1]B (where w[1] denotes the first coordinate of a vector w ∈ R 2 ). Hence, when x t = +, using y t h ≡ 1, we have ∇ log π θ (y t | x t )[2] = 0 and
0 ≤ ∇ log π θ (y t | x t )[1] = HB(1 -E a∼P θ (•|+) [a]) ≤ 2HB(1 -P θ (1 | +)) ≤ 4HB e θ[1]B .
Similarly, when x t = -, we have
∇ log π θ (y t | x t )[1] = 0, 0 ≤ ∇ log π θ (y t | x t )[2] ≤ 4HB e θ[2]B .
Then, combining the inequalities above with Claim 1, we can inductively show that for any t ∈ [T ],
θ t [1] -θ 0 [1] ≤ T t=1 I{x t = +} • 4ηHB e θ 0 [1]B ≤ T • µ(+) 16eηB Be rB ≤ µ(+) • 512eBn log N BH ≤ 1 B .
Therefore, we have θ t [1] ≤ r for t ≤ T . It remains to prove the following claim.
Claim 3. Suppose that e θ[1]B ≤ H 4 log N . Then it holds that Cov N (π θ ) ≥ µ(+) 2 .
To prove Claim 3, we note that similar to Claim 2,
P πD (y h = 1 ∀h ∈ [H] | x = +) ≥ 1 2 . Further, log π D (y 1 = 1 | +) ≥ log(1 -2e -B/2 ) ≥ -3e -B/2 and log π θ (y 1 = 1 | +) ≤ -1 3e θ[1]B . Hence, for y ⋆ ∈ V H being y ⋆ h = 1 for h ∈ [H], it holds that log π D (y ⋆ | +) -log π θ (y ⋆ | +) ≥ H • 1 3e θ[1]B -3e -B/2 ≥ log N.
The immediately yields
Cov N (π θ ) ≥ µ(+) • P πD (y = y ⋆ | x = +) ≥ µ(+) 2 . K.5 PROOF OF THEOREM 5.1 (COVERAGE FOR NORMALIZED SGD)
We denote M := log N . We analyze the normalized SGD iterates assuming λ ≥ 8BM and λη M ≤
1 16 . Denote g(θ; D) := g(θ; D) λ + g(θ; D) .
Then the normalized SGD update can be rewritten as θ t+1 = Proj Θ (θ + η g(θ t ; D t )). Specializing Lemma K.5 to the normalized SGD update and using Θ ⊆ B 2 (1) yields
T t=1 -g(θ t ; D t ), θ t -θ ⋆ ≤ 2 η + η T t=1 g(θ t ; D t ) 2 .
Taking an expectation on both sides and noting that D t ∼ π D is generated independently, we have
E T t=1 E D∼πD -g(θ t ; D), θ t -θ ⋆ ≤ 2 η + η E T t=1 E D∼πD g(θ t ; D) 2 . (71
)
In what follows, we prove a number of upper and lower bounds for the expressions involving g(θ; D) above, then combine them with Eq. ( 71) to complete the proof.
this section cite: []

Section: Intermediate bounds.
Recall that we write
ϵ θ (x, y 1:h-1 ) = D KL (π D (• | x, y 1:h-1 ) π θ (• | x, y 1:h-1 )).
Also recall that we adopt the notation that for any function f and dataset D, we write
E D [f ] := 1 |D| (x,y 1:H )∈D f (x, y 1:H ). Denote (recall that D seq,N (• •) is defined in Proposition F.10) ϵ θ (D) := E D H h=1 ϵ θ (x, y 1:h-1 ) , ∆ θ := E πD min{M, ϵ θ (D)}.
Using the key structural result in Proposition F.10 (recall M := log N ), we can bound the coverage in terms of the expected sum of stopped KL divergences as follows:
Cov N (π θ ) ≤ 2 M -1 D seq,N (π D π θ ) = 2 M -1 E πD min M, H h=1 ϵ θ (x, y 1:h-1 ) ≤ 2 M -1 E D∼πD min M, E D H h=1 ϵ θ (x, y 1:h-1 ) = 2 M -1 ∆ θ .(72)
Therefore, it remains to derive upper bounds on ∆ θ for θ ∈ {θ 1 , • • • , θ T }.
Lemma K.8. Suppose that λ ≥ 8BM . It holds that for any θ ∈ Θ,
E πD g(θ; D) 2 ≤ 2∆ θ M + 4M σ 2 ⋆ λ 2 + σ ⋆ λ √ K .
Lemma K.9. Suppose that λ ≥ 8BM . Denote Λ θ := -g(θ; D), θ -θ ⋆ . Then it holds that for any θ ∈ Θ,
∆ θ ≤ 8λΛ θ + 240B K + 8 M σ ⋆ λ 2 .
Putting everything together. Under the notation of Lemma K.8 and Lemma K.9, Eq. ( 71) can be rewritten as
E T t=1 Λ θ t ≤ 2 η + η 2 E T t=1 E D∼πD g(θ t ; D) 2 . (73
)
Applying Lemma K.8 and Lemma K.9, we have
1 T E T t=1 ∆ θ t - 240B K -8 M σ ⋆ λ 2 ≤ 8λ T E T t=1 Λ θ t ≤ 16λ T η + 4ηλ T E T t=1 E D∼πD g(θ t ; D) 2 ≤ 16λ T η + 8ηλ M T E T t=1 ∆ θ t + 16ηM σ 2 ⋆ λ + 4ησ ⋆ √ K ,
where the first inequality uses Lemma K.9, the second inequality follows from Eq. ( 73), and the last inequality uses Lemma K.8. Therefore, as long as ηλ ≤ M 16 , it holds that
1 T E T t=1 ∆ θ t ≲ B K + M σ ⋆ λ 2 + λ T η + ηM σ 2 ⋆ λ + ησ ⋆ √ K .
Simplifying the upper bound. In the following, we require η ≤ 1 128B and choose λ = M 16η . Then, it holds that
1 T E T t=1 ∆ θ t ≲ B K + (ησ ⋆ ) 2 + M T η 2 + ησ ⋆ √ K ≲ B K + (ησ ⋆ ) 2 + M T η 2 ,
where we use AM-GM inequality and B ≥ 1. Finally, we may choose η = min
1 128B , M σ 2 ⋆ T 1/4
.
Recall that M = log N , and hence our choice of η gives
1 T E T t=1 D seq,N (π D π θ t ) ≤ 1 T E T t=1 ∆ θ t ≲ σ 2 ⋆ log N T + B 2 log N T + B K ,
which implies (by Eq. ( 72))
1 T E T t=1 Cov N (π θ t ) ≤ 1 T E T t=1 2 log N -1 D seq,N (π D π θ t ) ≲ σ 2 ⋆ T log N + B 2 T + B K log N .
This is the desired upper bound.
Remark K.1 (Comparison to standard convex optimization analyses). On a technical level, we find the proof of Theorem 5.1 to be interesting because it does not pass through KL divergence as an intermediate quantity. More broadly, we do not know how to derive the result as an application of standard analysis techniques in optimization (e.g., via a gradient dominance or PL-type condition), but it would be interesting to see if there is a connection. 11Proof of Lemma K.8. Note that g(θ;
D) ≤ min 1, ∥ g(θ ;D )∥ λ . Recall that g(θ; D) = E D [∇ log π θ (y | x)], ∇ log π θ (y | x) = H h=1 ϕ(x, y 1:h ) -ϕ θ (x, y 1:h-1 ) ,
with the notation introduced at the beginning of Appendix K.
We decompose g(θ; D) by introducing
g θ (D) := E D H h=1 ϕ θ (x, y 1:h-1 ) -ϕ θ ⋆ (x, y 1:h-1 ) ,(74)
and
z(D) := E D H h=1 ϕ ⋆ (x, y 1:h ) = E D H h=1 ϕ(x, y 1:h ) -ϕ θ ⋆ (x, y 1:h-1 ) . (75
)
Then, by definition, -g(θ; D) = g θ (D) -z(D). In the following, we first analyze g θ (D) and z(D) separately under D = {(x i , y i 1:H )} i∈[K] ∼ π D and summarize the corresponding upper bounds on in Lemma K.10 (stated and proven in the sequel).
Now, using g(θ; D) ≤ min 1, ∥ g(θ ;D )∥ λ , we know g(θ; D) 2 ≤ I{ϵ θ (D) > M } + I{ϵ θ (D) ≤ M } • g(θ; D) λ ≤ I{ϵ θ (D) > M } + I{ϵ θ (D) ≤ M } λ • 4 σ 2 (D) • ϵ θ (D) + 8Bϵ θ (D) + 1 λ z(D) ≤ 1 M min{M, ϵ θ (D)} + 4 λ σ 2 (D) • min{M, ϵ θ (D)} + 1 λ z(D) ,
where the second inequality uses g(θ; D) ≤ g θ (D) + z(D) and Lemma K.10 (2), and the last inequality uses λ ≥ 8BM and 1 M min{M, ϵ θ (D)} = 1 when ϵ θ (D) > M . Taking expectation of D ∼ π D , we have
E πD g(θ; D) 2 ≤ 1 M E πD min{M, ϵ θ (D)} + 4 λ E πD σ 2 (D) • min{M, ϵ θ (D)} + σ ⋆ λ √ K ≤ 1 M E πD min{M, ϵ θ (D)} + 4σ ⋆ λ E πD min{M, ϵ θ (D)} + σ ⋆ λ √ K = ∆ θ M + 4σ ⋆ λ ∆ θ + σ ⋆ λ √ K .
where the second inequality follows from Cauchy-Schwarz inequality, Lemma K.10 (3) and the fact that E[σ 2 (D)] = σ 2 ⋆ . By AM-GM inequality, it holds that 4σ⋆
λ √ ∆ θ ≤ ∆ θ M + 4M σ 2 ⋆ λ 2
, and the desired upper bound follows immediately.
Lemma K.10. For any θ ∈ Θ, the following holds:
(1) It holds that
g θ (D), θ -θ ⋆ ≥ E D H h=1 ϵ θ (x, y 1:h-1 ) =: ϵ θ (D) (2) Denote σ 2 (D) := E D H h=1 Var πD (x, y 1:h-1 ) . Then g θ (D) ≤ 4 σ 2 (D) • ϵ θ (D) + 8Bϵ θ (D). (3) It holds that E D∼πD z(D) 2 = σ 2 ⋆ K and E D∼πD z(D), θ -θ ⋆ - 1 2 ϵ θ (D) + ≤ 30B K =: α.
Proof of Lemma K.10. Lemma K.10 (1) follows immediately from Eq. ( 58):
g θ (D), θ -θ ⋆ = E D H h=1 ϕ θ (x, y 1:h-1 ) -ϕ θ ⋆ (x, y 1:h-1 ), θ -θ ⋆ ≥ E D H h=1 ϵ θ (x, y 1:h-1 ) =: ϵ θ (D).
Lemma K.10 (2) follows immediately from Eq. ( 59):
g θ (D) ≤ E D H h=1 ϕ θ (x, y 1:h-1 ) -ϕ θ ⋆ (x, y 1:h-1 ) ≤ E D H h=1 4 Var πD (x, y 1:h-1 ) • ϵ θ (x, y 1:h-1 ) + 8Bϵ θ (x, y 1:h-1 ) ≤ 4 σ 2 (D) • ϵ θ (D) + 8ϵ θ (D). It remains to prove Lemma K.10 (3). Note that K • z(D) = K • E D H h=1 ϕ ⋆ (x, y 1:h ) = K i=1 H h=1 ϕ ⋆ (x i , y i 1:h ) is a sum of the martingale difference sequence {ϕ ⋆ (x i , y i 1:h )} i∈[K],h∈[H] . Therefore, we can calculate E z(D) 2 = 1 K E πD H h=1 ϕ ⋆ (x, y 1:h ) 2 = σ 2 ⋆ K .
Furthermore, by Freedman's inequality (Lemma H.1), for any fixed vector v, parameter γ ∈ (0, 1 B ) and δ ∈ (0, 1), it holds that
P K i=1 H h=1 ϕ ⋆ (x i , y i 1:h ), v -γ E ϕ ⋆ (x i , y i 1:h ), v 2 | x i , y i 1:h-1 ≥ γ -1 log(1/δ) ≤ δ.
Note that for v = θ -θ ⋆ , by Lemma H.5, we have
E ϕ ⋆ (x i , y i 1:h ), v 2 | x i , y i 1:h-1 = E y h ∼πD(•|x i ,y i 1:h-1 ) ϕ(x i , y i 1:h-1 , y h ) -ϕ θ ⋆ (x i , y i 1:h-1 , y h ), θ -θ ⋆ 2 ≤ 15BD KL π D (• | x i , y i 1:h-1 ) π θ (• | x i , y i 1:h-1 ) = 15Bϵ θ (x i , y i 1:h-1 ).
Therefore, setting γ = 1 30B , we have shown that for any δ ∈ (0, 1), it holds that
P πD z(D), θ -θ ⋆ ≥ 1 2 E D H h=1 ϵ θ (x, y 1:h-1 ) + 30B log(1/δ) K ≤ δ.
Recall that we denote ϵ θ (D) := E D H h=1 ϵ θ (x, y 1:h-1 ) . Then, for the random variable V := K 30B z(D), θ -θ ⋆ -1 2 ϵ θ (D) , the above inequality implies that for any u > 0, P(V ≥ u) ≤ e -u , and hence P((V ) + ≥ u) ≤ e -u . Therefore, integrating out the above inequality gives E[(V ) + ] ≤ 1, or equivalently,
E πD z(D), θ -θ ⋆ - 1 2 ϵ θ (D) + ≤ 30B K =: α.
Proof of Lemma K.9. Recall that we can decompose -g(θ; D) = g θ (D) -z(D), where g θ (D) and z(D) are defined in Eq. ( 74) and Eq. ( 75), respectively. Then, we know
Λ θ := E πD -g(θ; D), θ -θ ⋆ = E πD g θ (D), θ -θ ⋆ -z(D), θ -θ ⋆ λ + g θ (D) ≥ E πD ϵ θ (D) -z(D), θ -θ ⋆ λ + g θ (D) ≥ 1 2 E πD ϵ θ (D) λ + z(D) + g θ (D) - 1 λ E πD z(D), θ -θ ⋆ - 1 2 ϵ θ (D) + ≥ 1 2 E πD ϵ θ (D) λ + z(D) + g θ (D) - α λ ,
where the first inequality uses Lemma K.10 (1) and the last inequality uses Lemma K.10 (3) and we recall that α = 30B K . Note that by Lemma K.10 (2), λ + z(D) + g θ (D)
≤ λ + z(D) + 4 σ 2 (D) • ϵ θ (D) + 8Bϵ θ (D) ≤ max{M, ϵ θ (D)} M • 2λ + z(D) + 4M σ 2 (D) min{M, ϵ θ (D)} ,
where we use min{M, x} max{M, x} = M x, and λ ≥ 8BM . Combining these two inequalities, we have
2Λ θ + 2α λ ≥ E πD ϵ θ (D) λ + z(D) + g θ (D) ≥ E πD min{M, ϵ θ (D)} 2λ + z(D) + 4M σ 2 (D)/ min{M, ϵ θ (D)} ≥ (E πD min{M, ϵ θ (D)}) 2 E πD min{M, ϵ θ (D)}(2λ + z(D) ) + 4M σ 2 (D) • min{M, ϵ θ (D)} ≥ (E πD min{M, ϵ θ (D)}) 2 2λ E πD min{M, ϵ θ (D)} + M σ 2 ⋆ /K + 4M σ 2 ⋆ E πD min{M, ϵ θ (D)} = ∆ 2 θ 2λ∆ θ + M σ ⋆ 1 √ K + 4 √ ∆ θ ,
where the last two inequalities follow from Cauchy-Schwarz inequality. Therefore, there are two cases: (a) ∆ θ ≤ 1 K , and the desired upper bound is trivially true. (b) ∆ θ ≥ 1 K , and then it holds that
2λ∆ θ + M σ ⋆ 1 √ K + 4 ∆ θ ≤ 2λ∆ θ + 5M σ ⋆ ∆ θ ≤ 3λ∆ θ + 8(M σ ⋆ ) 2 λ ,
where we use AM-GM inequality. Hence, it holds that
∆ 2 θ ≤ 8(λΛ θ + α) max ∆ θ , 8 M σ ⋆ λ 2 ,
and reorganizing yields
∆ θ ≤ 8 max    (λΛ θ + α), (λΛ θ + α) M σ ⋆ λ 2    ≤ 8(λΛ θ + α) + 8 M σ ⋆ λ 2 .
This is the desired result. K.6 PROOF OF THEOREM 6.1 (TEST-TIME TRAINING) Recall (from Eq. ( 12)) that we consider the token-level SGD iterates defined as
θ t,h+1 = Proj Θ θ t,h + η∇ log π θ t,h (y t h | x t , y t 1:h-1 ) , for h = 0, • • • , H -1,(76)
and θ t+1 ≡ θ t+1,0 := θ t,H for t ∈ [T ], where (x t , y t 1:H ) ∼ π D . To define the guarantee on θ t which we are able to derive, we next define the following test-time parameter update ϑ TTT (x, y 1:h ; θ), for a parameter θ and prompt x. It is defined recursively for h = 0, 1, • • • , H -1: ϑ TTT (x, y 1:h ; θ) := Proj Θ ϑ TTT (x, y 1:h-1 ; θ) + η∇ log π ϑ TTT (x,y 1:h-1 ;θ) (y h | x, y 1:h-1 ) . (77)
We then define a distribution π TTT θ :
X → ∆(Y H ) as π TTT θ (• | x, y 1:h-1 ) := π ϑ TTT (x,y 1:h-1 ;θ) (• | x, y 1:h-1 ).(78)
The distribution π TTT θ can be interpreted as an augmented version of the autoregressive linear model π θ that performs test-time training during sampling.
Proof. While the algorithm in Theorem 6.1 might seem somewhat complicated and mysterious, the proof is actually a based on a fairly simple online-to-batch conversion argument. We use a number of basic inequalities already found in the proof of Proposition 5.1 (cf. Appendix K.3).
We first note that we can specialize Lemma K.5 to the token-level SGD update (12), and taking expectation gives
E T t=1 H h=1 -∇ log π θ t,h (y t h | x t , y t 1:h-1 ), θ t,h -θ ⋆ ≤ 2 η + η 2 E T t=1 H h=1 ∇ log π θ t,h (y t h | x t , y t 1:h-1 ) 2 . (79
)
In the following, we denote
ϵ t,h := E -∇ log π θ t,h (y t h | x t , y t 1:h-1 ), θ t,h -θ ⋆ . By triangle inequality, ∇ log π θ (y h | x, y 1:h-1 ) 2 ≤ 2 ∇ log π θ ⋆ (y h | x, y 1:h-1 ) 2 + 2 ∇ log π θ (y h | x, y 1:h-1 ) -∇ log π θ ⋆ (y h | x, y 1:h-1 ) 2 .
Using the fact that θ → log π θ (y h | x, y 1:h-1 ) is concave and B 2 -smooth, it holds that for any θ,
∇ log π θ (y h | x, y 1:h-1 ) -∇ log π θ ⋆ (y h | x, y 1:h-1 ) 2 ≤ B 2 • θ -θ ⋆ , ∇ log π θ ⋆ (y h | x, y 1:h-1 ) -∇ log π θ (y h | x, y 1:h-1 ) .
Combining the two inequalities above gives that for all t ∈ [T ], h ∈ [H],
∇ log π θ t,h (y t h | x t , y t 1:h-1 ) 2 ≤ 2 ∇ log π θ ⋆ (y t h | x t , y t 1:h-1 ) 2 + 2B 2 ∇ log π θ ⋆ (y t h | x t , y t 1:h-1 ) -∇ log π θ t,h (y t h | x t , y t 1:h-1 ), θ t,h -θ ⋆ Note that the conditional distribution of y t h | (x t , y t 1:h-1 , θ t,h ) is given by y t h ∼ π D (• | x t , y t 1:h-1
). Hence, taking the expectation over the entire learning process, we have
E ∇ log π θ t,h (y t h | x t , y t 1:h-1 ) 2 ≤ 2 E πD ∇ log π θ ⋆ (y h | x, y 1:h-1 ) 2 + 2B 2 E -∇ log π θ t,h (y t h | x t , y t 1:h-1 ), θ t,h -θ ⋆ = 2 E πD [Var πD (x, y 1:h-1 )] + 2B 2 ϵ t,h .
Plugging the above inequality to Eq. ( 79) yields
T t=1 H h=1 ϵ t,h ≤ 2 η + η 2 E T t=1 H h=1 ∇ log π θ t,h (y t h | x t , y t 1:h-1 ) 2 ≤ 2 η + ηT E πD H h=1 Var πD (x, y 1:h-1 ) + ηB 2 T t=1 H h=1 ϵ t,h . Therefore, as long as η ≤ 1 2B 2 , it holds that T t=1 H h=1 ϵ t,h ≤ 4 η + 2ηT E πD H h=1
Var πD (x, y 1:h-1 ) = 4 η + 2ηT σ 2 ⋆ .
By Eq. ( 58), it also holds that
ϵ t,h = E -∇ log π θ t,h (y t h | x t , y t 1:h-1 ), θ t,h -θ ⋆ ≥ E D KL π D (• | x t , y t 1:h-1 ) π θ t,h (• | x t , y t 1:h-1 ) .
Combining the inequalities above, as long as η ≤ 1 2B 2 , we have that
E T t=1 H h=1 D KL π D (• | x t , y t 1:h-1 ) π θ t,h (• | x t , y t 1:h-1 ) ≤ T t=1 H h=1 ϵ t,h ≤ 4 η + 2ηT σ 2 ⋆ . (80
)
Finally, we note that θ t,h = ϑ TTT (x t , y t h-1 ; θ t ), and that for all t and h, x t , y t h-1 | θ t ∼ π D . Therefore, we have the following key identity:
E D KL π D (• | x t , y t 1:h-1 ) π θ t,h (• | x t , y t 1:h-1 ) | θ t = E (x,y)∼πD D KL π D (• | x, y 1:h-1 ) π ϑ TTT (x,y 1:h-1 ;θ t ) (• | x, y 1:h-1 ) = E (x,y)∼πD D KL π D (• | x, y 1:h-1 ) π TTT θ t (• | x, y 1:h-1
) . Combined with Eq. ( 80), this implies that
4 η + 2ηT σ 2 ⋆ ≥ E T t=1 H h=1 D KL π D (• | x t , y t 1:h-1 ) π θ t,h (• | x t , y t 1:h-1 ) = E T t=1 E H h=1 D KL π D (• | x t , y t 1:h-1 ) π θ t,h (• | x t , y t 1:h-1 ) θ t = E T t=1 E πD H h=1 D KL π D (• | x, y 1:h-1 ) π TTT θ t (• | x, y 1:h-1 ) = E T t=1 D KL π D π TTT θ t ,
where the last equality uses the chain rule for KL divergence.
In particular, we may choose η = min 1 2B 2 , 1
σ 2 ⋆ T 1/2 to derive 1 T E T t=1 D KL (π D π TTT θ t ) ≲ σ 2 ⋆ T + B 2 T . K.7 PROOF OF THEOREM G.2 (GRADIENT NORMALIZATION FOR DISTILLATION)
Specializing Lemma K.5 to the update (35) and taking expectation gives
E T t=1 -E (x,y)∼πD [ g θ t (y | x)], θ t -θ ⋆ ≤ 2 η + η 2 E T t=1 E (x,y)∼πD g θ t (y | x) 2 . (81
)
In the following, we analyze -E (x,y)∼πD [ g θ (y | x)], θ t -θ ⋆ and E (x,y)∼πD g θ t (y | x) 2 for any θ ∈ Θ, following the proof of Proposition 5.1 (cf. Appendix K.3).
Relating the gradient to stopped KL divergence. Recall that the estimator g is defined in Eq. ( 35):
g θ (y | x) = H h=1 α θ (x, y 1:h-1 )∇ log π θ (y h | x, y 1:h-1 ),
and the weight function α θ is defined in Eq. ( 36).
We first recall an elementary property of the quantity α θ . By Lemma K.4, we have H h=1 α θ (x, y 1:h-1 )ϵ θ (x, y 1:h-1 ) = min A,
H h=1 ϵ θ (x, y 1:h-1 ) ,(82)
and hence
E (x,y)∼πD H h=1 α θ (x, y 1:h-1 )ϵ θ (x, y 1:h-1 ) = E (x,y)∼πD min A, H h=1 ϵ θ (x, y 1:h-1 ) = D seq,N (π D π θ ),(83)
where we recall that D seq,N (π D π θ ) is defined in Proposition F.10 and we denote A = log N . Hence,
-E (x,y)∼πD [ g θ (y | x)], θ -θ ⋆ = E (x,y)∼πD H h=1 α θ (x, y 1:h-1 ) ϕ θ (x, y 1:h-1 ) -ϕ θ ⋆ (x, y 1:h-1 ), θ -θ ⋆ ≥ E (x,y)∼πD H h=1 α θ (x, y 1:h-1 )ϵ θ (x, y 1:h-1 ) = D seq,N (π D π θ ),(84)
where the inequality uses Eq. ( 58).
In addition, the following lemma shows that E (x,y)∼πD g θ (y | x) 2 is well-controlled.
Lemma K.11 (Gradient error bound). For any θ ∈ Θ, it holds that
E (x,y)∼πD g θ (y | x) 2 ≤ (64A + 2)σ 2 ⋆ + 256AB 2 D seq,N (π D π θ ).
Putting everything together. Finally, combining the inequalities above, we know that
E T t=1 D seq,N (π D π θ t ) ≤ E T t=1 -E (x,y)∼πD [ g θ t (y | x)], θ t -θ ⋆ ≤ 2 η + η 2 E T t=1 E (x,y)∼πD g θ t (y | x) 2 ≤ 2 η + ηT (32A + 1)σ 2 ⋆ + 128AB 2 E T t=1 D seq,N (π D π θ t ) ,
where the first inequality uses Eq. ( 84), the second inequality follows from Eq. ( 81), and the third inequality uses Lemma K.11. Therefore, as long as η ≤ 1 2(32A+1)B 2 , it holds that
E T t=1 D seq,N (π D π θ t ) ≲ 1 η + ηT Aσ 2 ⋆ .
In particular, we may choose η = min
1 (64 log N +2)B 2 , 1 T σ 2 ⋆ log N 1/2
and derive
E 1 T T t=1 D seq,N (π D π θ t ) ≲ σ 2 ⋆ log N T + B 2 log N T .
By Proposition F.10, this implies
E 1 T T t=1 Cov N (π θ t ) ≲ σ 2 ⋆ T log N + B 2 T .
Proof of Lemma K.11. Fix any θ ∈ Θ. By triangle inequality, it holds that
g θ (y | x) -g θ ⋆ (y | x) ≤ H h=1 α θ (x, y 1:h-1 ) ϕ θ ⋆ (x, y 1:h ) -ϕ θ (x, y 1:h-1 ) ≤ H h=1 α θ (x, y 1:h-1 ) 4 Var πD (x, y 1:h-1 ) • ϵ θ (x, y 1:h-1 ) + 8Bϵ θ (x, y 1:h-1 ) ≤ 4 H h=1 α θ (x, y 1:h-1 )Var πD (x, y 1:h-1 ) 1/2 H h=1 α θ (x, y 1:h-1 )ϵ θ (x, y 1:h-1 ) 1/2 + 8B H h=1 α θ (x, y 1:h-1 )ϵ θ (x, y 1:h-1 ) ≤ 4 A • H h=1
Var πD (x, y 1:h-1 ) + 8B min A, H h=1 ϵ θ (x, y 1:h-1 ) .
where the second inequality follows from Eq. ( 59), the third inequality follows from Cauchy-Schwarz inequality, and the final lines follow from the property (82) of the weight function α θ ∈ [0, 1]. Hence, using (a + b) 2 ≤ 2a 2 + 2b 2 , we have
g θ (y | x) -g θ ⋆ (y | x) 2 ≤ 32A H h=1
Var πD (x, y 1:h-1 ) + 128AB 2 min A, H h=1 ϵ θ (x, y 1:h-1 ) .
Therefore, taking expectation of (x, y) ∼ π D and using E πD g θ ⋆ (y | x) 2 ≤ σ 2 ⋆ and Eq. ( 83), it holds that
E (x,y)∼πD g θ (y | x) 2 ≤ (64A + 2)σ 2 ⋆ + 256AB 2 D seq,N (π D π θ )
. This is the desired upper bound.
this section cite: []

Section: K.8 NECESSITY OF VARIANCE DEPENDENCE IN HIGH DIMENSION
We generalize Proposition 3.2 to show that in the worst case (where σ 2 ⋆ HB 2 ), the scaling Cov N ( π) = Ω( H n log N ) can be unavoidable for autoregressive linear model. This implies that the dependence on σ 2 ⋆ is generally necessary to achieve upper bounds that do not explicitly scale with H.
Proposition K.1. Let H, B, N, n ≥ 1, and assume log N ≤ c min{H, B 2 } for a sufficiently small constant c > 0. There exists an instance of the autoregressive linear model class Π with d = H, ϕ : X × V ⋆ → B 2 (B), and Θ = B 2 (1), such that for any proper algorithm Alg with output π = π θ for θ ∈ Θ, there exists π D ∈ Π, such that under π D , it holds that
E π D ,Alg [Cov N (π D π)] ≥ c • min 1, H n • log N .
Proof of Proposition K.1. We consider X = {+, -}, V = {0, 1}, and the distribution µ be given by µ(+) = 1 -µ(-) = p, where p ∈ [0, 1] is a parameter to be chosen later. Let the feature map ϕ be given by ϕ(-, y 1:h ) = 0, ϕ(+, y 1:h ) = By h e h , where (e 1 , • • • , e H ) is a fixed orthonormal basis of R H . Note that with this construction, we have π θ (y h = • | -, y 1:h-1 ) = Ber(1/2), and
π θ (y h = • | +, y 1:h-1 ) = Ber e Bθ h 1 + e Bθ h =: π θ,h .
Note that for any h ∈ [H], we can bound
C 0 B|θ h -θ ′ h | ≤ D H (π θ,h , π θ ′ ,h ) ≤ C 1 B|θ h -θ ′ h |, as long as θ h ∈ [-1 B , 1 B ]
. We fix ϵ ∈ [0, 1 max{ √ H,B} ] to be determined later, and for any v ∈ {-1, 1} H , we let θ v := ϵ H h=1 v h e h , and Θ 0 := θ v : v ∈ {-1, 1} H ⊂ B 2 (1), Π 0 := {π θ : θ ∈ Θ 0 }.
Then a direct argument (see e.g., (Wainwright, 2019, Section 15.3)) shows that when pn ≤ c0 B 2 ϵ 2 for a sufficiently small constant c 0 , there exists θ ⋆ ∈ Θ 0 such that under π D = π θ ⋆ , it holds that
H h=1 P π D ,Alg | θ h -θ ⋆ h | ≥ ϵ ≥ cH.
Therefore, with probability at least c 2 , it holds that
H h=1 I | θ h -θ ⋆ h | ≥ ϵ ≥ cH 2 ,
and this in turn implies
H h=1 D 2 H π θ ⋆ ,h , π θ,h ≥ c 1 HB 2 ϵ 2 .
Then, by Proposition F.11, we know that under the above event, as long as log N ≤ c1HB 2 ϵ 2 2 , we have Cov N ( π) ≥ p 2 . Choosing ϵ = 4 log N c1HB 2 and p = min 1, c0 nB 2 ϵ 2
gives the desired lower bound.
this section cite: []

Section: L PROOFS FROM SECTION 6
L.1 PROOF OF THEOREM 6.2 (SIMPLE TOURNAMENT) Below we state and prove a generalization of Theorem 6.2 which holds when the data distribution π D is not necessarily in the model class Π.
Theorem 6.2 ′ (General version of Theorem 6.2). Fix N ≥ 1, and consider the estimator π from Eq. ( 14):
π := arg min π∈Π max π ′ ∈Π Cov N (π ′ π).(85)
For any δ ∈ (0, 1), parameter a, c ≥ 0, with probability at least 1 -δ, it holds that
Cov N 1+a+2c ( π) ≲ min π∈Π Cov N a (π) + 1 N 1-a-2c + log N ∞ (Π; c log N ) + log δ -1 n . (86
)
Proof of Theorem 6.2 ′ . Fix N, N ′ ≥ 1, α > 0, and let π ∈ arg min π∈Π Cov N ′ (π D π). We study the estimator
π := arg min π∈Π max π ′ ∈Π Cov N (π ′ π). (87
)
Recall that we denote Cov πD N (π ′ π) = P πD π ′ (y|x) π(y|x) ≥ N (cf. Lemma J.2). By Lemma J.2, with probability at least 1 -δ 2 , it holds that
Cov N (π π) ≥ 1 2 Cov πD e 2α N (π π) -ε stat , ∀π ∈ Π,(88)
where ε stat = 8 log(4N∞(Π,α)/δ) n . Next, again by Lemma J.2, with probability at least 1 -δ 2 , it holds that
Cov N (π π) ≤ 2Cov πD e -2α N (π π) + ε stat , ∀π ∈ Π.(89)
In the following, we condition on the success event of Eq. ( 88) and Eq. ( 89). Then, we can bound
1 2 Cov πD e 2α N (π π) -ε stat ≤ Cov N (π π) ≤ max π ′ ∈Π Cov N (π ′ π) = min π∈Π max π ′ ∈Π Cov N (π ′ π) ≤ max π ′ ∈Π Cov N (π ′ π) ≤2 max π ′ ∈Π Cov πD e -2α N (π ′ π) + ε stat .
this section cite: []

Section: Reorganizing yields
Cov πD e 2α N (π π) ≤ 4 max π∈Π
Cov πD e -2α N (π π) + 4ε stat .(90)
Note that for any N ′′ and models π, π ′ , π ′′ ,
Cov πD N ′ N ′′ (π ′ π) ≤ Cov πD N ′ (π ′ π ′′ ) + Cov πD N ′′ (π ′′ π).(91)
Hence, for any model π ∈ Π,
Cov πD e 2α N N ′ (π D π) ≤ Cov πD N ′ (π D π) + Cov πD e 2α N (π π), (92
)
Cov πD e -2α N (π π) ≤ Cov πD N ′ (π π D ) + Cov πD e -2α N/N ′ (π D π).(93)
Therefore, combining the inequalities above, we see that
Cov e 2α N N ′ ( π) = Cov πD e 2α N N ′ (π D π) ≤ Cov πD N ′ (π D π) + Cov πD e 2α N (π π) ≤ Cov πD N ′ (π D π) + 4 max π∈Π Cov πD e -2α N (π π) + 4ε stat ≤ 5Cov πD N ′ (π D π) + 4 max π∈Π Cov πD e -2α N/N ′ (π π D ) + 4ε stat ≤ 5Cov πD N ′ (π D π) + e 2α N ′ N + 4ε stat ,
where the first inequality uses Eq. ( 92), the second inequality uses Eq. ( 90), the third inequality uses Eq. ( 93), and the last inequality follows from the fact that
Cov πD A (π π D ) = P πD π(y|x) πD(y|x) ≥ A ≤ 1 A .
The claimed bound (86) follows by setting α = c log N , and N ′ = N a .
this section cite: []

Section: L.2 PROOF OF THEOREM G.3 (OFFSET TOURNAMENT)
Divergence. For distributions P, Q ∈ ∆(Y), we define the following divergence for N ≥ 1:
12 E N (P Q) := max E y∼P dQ dP -N + , E y∼Q dP dQ -N + ∈ [0, 1].
Then, for models π, π ′ : X → ∆(Y), we further define
E N,µ (π π ′ ) := E x∼µ E N (π(• | x) π ′ (• | x)).
Under this divergence, it holds that for any event E,
P µ,π (E) ≤ N • P µ,π ′ (E) + E N,µ (π π ′ ),(94)
P µ,π ′ (E) ≤ N • P µ,π (E) + E N,µ (π π ′ ),(95)
where P µ,π is the probability under x ∼ µ and y ∼ π(• | x). Furthermore, we can bound
Cov 2N (π) = P µ,πD π D (y | x) π(y | x) ≥ 2N ≤ E N,µ (π D π).(96)
In particular, we know L( π, π) ≤ ε stat + ε ′ apx . Then, we can bound
P n (C N (π, π)) -L( π, π) = 2γP µn, π (C N (π, π)) ≤ 2γ N P µn,π (C N (π, π)) ≤ 2γ N γP µn,πD (C N (π, π)) + ε ′ apx ≤ 2γ N 2γ P n (C N (π, π)) + ε stat + ε ′ apx ,
where the first inequality follows from the fact that π(y | x) ≥ N π(y | x) for (x, y) ∈ C N (π, π), the second inequality uses Eq. ( 95): P µn,π (E) -γP µn,πD (E) ≤ E γ,µn (π D π) = ε ′ apx for any event E, and the third inequality uses Eq. ( 101). Therefore, using N ≥ 8γ 2 , we know P n (C N (π, π)) ≤ 5ε stat + 2ε ′ apx . Then, using Eq. ( 100), we have
Cov πD N (π π) = P µ,πD (C N (π, π)) ≤ 2 P n (C N (π, π)) + 2ε stat ≤ 12ε stat + 4ε ′ apx . By Eq. (91), it holds that Cov 2N γ ( π) = Cov πD 2N γ (π D π) ≤ Cov πD 2γ (π D π) + Cov πD N (π π),
and we also have 96). Combining the inequalities above, we can conclude that
Cov πD 2γ (π D π) = Cov 2γ (π) ≤ E γ,µ (π D π) = ε apx by Eq. (
Cov 2N γ ( π) ≤ Cov πD N (π π) + ε apx ≤ 12ε stat + 4ε ′ apx + ε apx .
Finally, using Lemma H.2, we have ε ′ apx ≤ 2ε apx + ε stat . This is the desired upper bound.
this section cite: []

Section: 
Published as a conference paper at ICLR 2026 Theorem G.3 ′ (General version of Theorem G.3). Fix N, γ ≥ 1 such that N ≥ 8γ 2 . Consider the estimator π := arg min π∈Π max
π ′ ∈Π { Cov N (π ′ π) -2γ • Cov π N (π ′ π)} . (97
)
Then with probability 1 -δ, it holds that
Cov 2N γ ( π) ≲ min π∈Π E γ (π D π) + log(|Π|/δ) n . Note that E γ (π D π) = 0 when |log π D (y | x) -log π(y | x)| ≤ log γ for any x ∈ X , y ∈ Y.
Therefore, Theorem G.3 is an immediate corollary by setting γ = N a .
Proof of Theorem G.3 ′ . For π, π ′ ∈ Π, we define the set
C N (π, π ′ ) = (x, y) | π(y | x) π ′ (y | x) ≥ N . Suppose an i.i.d. dataset D = {(x i , y i )} i∈[n] ∼ π D is drawn. We write P n = 1 n n i=1 δ (x i ,y i ) and µ n = 1 n n i=1 δ x i
to denote the empirical measures (i.e., P n is the uniform distribution over D), and let P µn,π be the probability under the distribution x ∼ µ n , y ∼ π(• | x). Under this notation, we have Cov N (π ′ π) = P n (C N (π ′ , π)) and we also recall that
Cov π N (π ′ π) := 1 n n i=1 P y∼π(•|x i ) π ′ (y | x i ) π(y | x i ) ≥ N = P µn,π (C N (π ′ , π)).
Thus, the tournament estimator in Eq. ( 97) can be expressed as
π := arg min π∈Π max π ′ ∈Π L(π, π ′ ),(98)
where
L(π, π ′ ) := P n (C N (π ′ , π)) -2γ • P µn,π (C N (π ′ , π)). (99
)
As an immediate consequence of Lemma H.2 and the union bound, we have the following lemma.
Lemma L.1. Fix δ ∈ (0, 1), and define ε stat = 16 log(16|Π|/δ) n . With probability 1 -δ, the following bounds hold simultaneously:
(1) For all π, π ′ ∈ Π, it holds that
2P µ,πD (C N (π ′ , π)) + ε stat ≥ P n (C N (π ′ , π)) ≥ 1 2 P µ,πD (C N (π ′ , π)) -ε stat ,(100)
2P µn,πD (C N (π ′ , π)) + ε stat ≥ P n (C N (π ′ , π)) ≥ 1 2 P µn,πD (C N (π ′ , π)) -ε stat . (101
) (2) For any π ∈ Π, it holds that E γ,µn (π D π) ≤ 2E γ,µ (π D π) + ε stat .
In the following, we fix δ ∈ (0, 1) and condition on the success event of Lemma L.1. Let π ∈ arg min π∈Π E γ,µ (π D π). We denote ε apx = E γ,µ (π D π) and ε ′ apx = E γ,µn (π D π). Note that by Lemma L.1, we have ε ′ apx ≤ 2ε apx + ε stat . Then, for any π ′ ∈ Π,
L(π, π ′ ) ≤ 2P µn,πD (C N (π ′ , π)) -2γP µn,π (C N (π ′ , π)) + ε stat ≤ 2E γ,µn (π D π) + ε stat = ε ′ apx + ε stat .
where the first inequality uses Eq. ( 101), and the second inequality uses Eq. ( 94).
Therefore, we have
max π ′ ∈Π L( π, π ′ ) = min π∈Π max π ′ ∈Π L(π, π ′ ) ≤ max π ′ ∈Π L(π, π ′ ) ≤ ε stat + ε ′ apx .
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2024)
Ref_id:b1 Title: The surprising effectiveness of test-time training for few-shot learning Year: (2025)
Ref_id:b2 Title: Physics of language models: Part 3.3, knowledge capacity scaling laws Year: (2025)
Ref_id:b3 Title: The pitfalls of next-token prediction Year: (2024)
Ref_id:b4 Title: Dissecting adam: The sign, magnitude and variance of stochastic gradients Year: (2018)
Ref_id:b5 Title: Smaller, weaker, yet better: Training LLM reasoners via compute-optimal sampling Year: (2025)
Ref_id:b6 Title: Deep learning: A statistical viewpoint Year: (2021)
Ref_id:b7 Title: Spectrally-normalized margin bounds for neural networks Year: (2017)
Ref_id:b8 Title: Benign overfitting in linear regression Year: (2020)
Ref_id:b9 Title: Reconciling modern machinelearning practice and the classical bias-variance trade-off Year: (2019)
Ref_id:b10 Title: Old optimizer, new norm: An anthology Year: (2024)
Ref_id:b11 Title: Minimax rates for conditional density estimation via empirical entropy Year: (2023)
Ref_id:b12 Title: The sample complexity of approximate rejection sampling with applications to smoothed online learning Year: (2023)
Ref_id:b13 Title: Large language monkeys: Scaling inference compute with repeated sampling Year: (2025)
Ref_id:b14 Title: Decision making in changing environments: Robustness, querybased learning, and differential privacy Year: ()
Ref_id:b15 Title: Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars Year: (2025)
Ref_id:b16 Title: REBEL: Reinforcement learning via regressing relative rewards Year: (2024)
Ref_id:b17 Title: Scaling laws for neural machine translation Year: (2022)
Ref_id:b18 Title: Introduction to online convex optimization Year: (2016)
Ref_id:b19 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b20 Title: Self-improvement in language models: The sharpening mechanism Year: ()
Ref_id:b21 Title: Is best-of-n the best of them? coverage, scaling, and optimality in inference-time alignment Year: (2025)
Ref_id:b22 Title: Correcting the mythos of kl-regularization: Direct alignment without overoptimization via chi-squared preference optimization Year: (2025)
Ref_id:b23 Title: Compression represents intelligence linearly Year: (2024)
Ref_id:b24 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b25 Title: Self-play with adversarial critic: Provable and scalable offline alignment for language models Year: (2024)
Ref_id:b26 Title: Offline reinforcement learning in large state spaces: Algorithms and guarantees Year: (2024)
Ref_id:b27 Title: Rl fine-tuning heals ood forgetting in sft Year: (2025)
Ref_id:b28 Title: Is pessimism provably efficient for offline rl Year: (2021)
Ref_id:b29 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b30 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b31 Title: Dynamic evaluation of neural sequence models Year: (2018)
Ref_id:b32 Title: Dynamic evaluation of transformer language models Year: (2019)
Ref_id:b33 Title: Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models Year: (2022)
Ref_id:b34 Title: Provably mitigating overoptimization in RLHF: Your SFT loss is implicitly an adversarial regularizer Year: (2024)
Ref_id:b35 Title: Non-vacuous generalization bounds for large language models Year: (2023)
Ref_id:b36 Title: Unlocking tokens as data points for generalization bounds on larger language models Year: (2024)
Ref_id:b37 Title: Scaling laws are unreliable for downstream tasks: A reality check Year: (2025)
Ref_id:b38 Title: Scaling laws for fact memorization of large language models Year: (2024-11)
Ref_id:b39 Title: Learning without Concentration Year: (2014)
Ref_id:b40 Title: Extending the scope of the small-ball method Year: (2017)
Ref_id:b41 Title: Recurrent neural network based language model Year: (2010)
Ref_id:b42 Title: Uniform convergence may be unable to explain generalization in deep learning Year: (2019)
Ref_id:b43 Title: Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction Year: (2025)
Ref_id:b44 Title: Norm-based capacity control in neural networks Year: (2015)
Ref_id:b45 Title: Introducing openai o1. Blog Year: (2024)
Ref_id:b46 Title: Channel coding: Non-asymptotic fundamental limits Year: (2010)
Ref_id:b47 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b48 Title: Statistical learning and sequential prediction Year: (2012)
Ref_id:b49 Title: Revisiting dynamic evaluation: Online adaptation for large language models Year: (2024)
Ref_id:b50 Title: Computational-statistical tradeoffs at the next-token prediction barrier: Autoregressive and imitation learning under misspecification Year: (2025)
Ref_id:b51 Title: Understanding transformer reasoning capabilities via graph algorithms Year: (2024)
Ref_id:b52 Title: Najoung Kim, and He He. Transformers struggle to learn to search Year: (2025)
Ref_id:b53 Title: Beyond chinchilla-optimal: Accounting for inference in language model scaling laws Year: (2024)
Ref_id:b54 Title: Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning Year: (2025)
Ref_id:b55 Title: The importance of online data: Understanding preference fine-tuning via coverage Year: (2024)
Ref_id:b56 Title: Parametric estimation. finite sample theory. The Annals of Statistics Year: (2012)
Ref_id:b57 Title: Overtrained language models are harder to fine-tune Year: (2025)
Ref_id:b58 Title: Learning to (learn at test time): Rnns with expressive hidden states Year: (2024)
Ref_id:b59 Title: Grapharena: Evaluating and exploring large language models on graph computation Year: (2025)
Ref_id:b60 Title: Are large-language models graph algorithmic reasoners? arXiv preprint Year: (2024)
Ref_id:b61 Title: Empirical Processes in M-Estimation Year: (2000)
Ref_id:b62 Title:  Year: (2000)
Ref_id:b63 Title: High-dimensional statistics: A non-asymptotic viewpoint Year: (2019)
Ref_id:b64 Title: Can language models solve graph problems in natural language? Year: (2023)
Ref_id:b65 Title: Do larger language models imply better generalization? a pretraining scaling law for implicit reasoning Year: (2025)
Ref_id:b66 Title: Probability inequalities for likelihood ratios and convergence rates of sieve mles Year: (1995)
Ref_id:b67 Title: The Invisible Leash: Why RLVR May Not Escape Its Origin Year: (2025)
Ref_id:b68 Title: Training trajectories of language models across scales Year: (2022)
