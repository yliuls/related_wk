Title: Beyond Self-Repellent Kernels: History-Driven Target Towards Efficient Nonlinear MCMC on General Graphs
Abstract: We propose a history-driven target (HDT) framework in Markov Chain Monte Carlo (MCMC) to improve any random walk algorithm on discrete state spaces, such as general undirected graphs, for efficient sampling from target distribution µ. With broad applications in network science and distributed optimization, recent innovations like the self-repellent random walk (SRRW) achieve near-zero variance by prioritizing undersampled states through transition kernel modifications based on past visit frequencies. However, SRRW's reliance on explicit computation of transition probabilities for all neighbors at each step introduces substantial computational overhead, while its strict dependence on timereversible Markov chains excludes advanced nonreversible MCMC methods. To overcome these limitations, instead of direct modification of transition kernel, HDT introduces a history-dependent target distribution π[x] to replace the original target µ in any graph sampler, where x represents the empirical measure of past visits. This design preserves lightweight implementation by requiring only local information between the current and proposed states and achieves compatibility with both reversible and non-reversible MCMC samplers, while retaining unbiased samples with target distribution µ and near-zero variance performance. Extensive experiments in graph sampling demonstrate consistent performance gains, and a memory-efficient Least Recently Used (LRU) cache ensures scalability to large general graphs.

Section: Introduction
Random walks on general graphs are fundamental tools across diverse disciplines, including physics, statistics, machine learning, biology, and social sciences (Robert & Casella, 2013;Grover & Leskovec, 2016;Masuda et al., 2017;Kim et al., 2024), powering applications like online social networks (Xie et al., 2021), web crawling (Olston et al., 2010), robotic exploration (Placed et al., 2023), and mobile networks (Triastcyn et al., 2022;Ayache et al., 2023). Starting from any state (node), a walker transitions to one of its neighbors along the connected edges using only local information, combining simplicity with scalability. This makes random walks powerful for exploring large networks and performing graph-based Markov chain Monte Carlo (MCMC) tasks to generate samples from target distributions µ on the graph (e.g., uniform, degree-based, or energybased). They also enable distributed optimization in networked systems (Sun et al., 2018;Hu et al., 2022;Even, 2023;Hendrikx, 2023). Simple random walks (Gjoka et al., 2011;Perozzi et al., 2014) and Metropolis-Hastings (MH) random walks (Metropolis et al., 1953;Hastings, 1970;Xia et al., 2019) are foundational methods for these tasks, though they can suffer from slow mixing and local trapping.
this section cite: ['b56', 'b26', 'b46', 'b35', 'b66', 'b50', 'b53', 'b4', 'b59', 'b31', 'b21', 'b28', 'b23', 'b52', 'b47', 'b27', 'b64']

Section: Recent Advances in Graph Sampling
In social networks, e-commerce, recommendation systems and other domains, practitioners need to estimate graph size, node degree distribution, and label distribution (Xie et al., 2021). These applications include detecting bot populations, identifying high-value user segments or scarce features (Nakajima & Shudo, 2023), and gradient of local loss for token algorithms in distributed optimizations (Sun et al., 2018;Even, 2023), especially with limited graph access. However, a standard random walk can be slow in discovering underrepresented regions or rare features, forcing more samples to achieve acceptable levels of accuracy. To improve the sampling efficiency of the MH algorithm, MH with Delayed Rejection reduces persistent rejections to facilitate exploration (Green & Mira, 2001). Another variant is the Multiple-Try Metropolis (MTM), which proposes multiple candidates at each step and selects one based on their weights (Liu et al., 2000;Pandolfi et al., 2010).  The choice of weights is further refined by Chang et al. (2022) with the integration of locally balanced functions in Zanella (2020), quantifying the improvement in mixing time. Further improvements have been achieved via nonreversible random walk techniques. These include lifted Markov chains, which augment the state space with directional information (Chen et al., 1999;Diaconis et al., 2000;Apers et al., 2017); 2-cycle MCMC methods, which alternate between two reversible chains (Maire et al., 2014;Ma et al., 2016;Andrieu & Livingstone, 2021); and modifications from a reversible chain by adding antisymmetric perturbations (Suwa & Todo, 2010;Chen & Hwang, 2013;Bierkens, 2016;Thin et al., 2020). Non-backtracking random walks, which avoid revisiting the immediate past, have also improved exploration (Alon et al., 2007;Lee et al., 2012;Hermon, 2019).
In fields such as statistical physics (e.g., Ising or Potts models (Grathwohl et al., 2021;Zhang et al., 2022)), machine learning (e.g., energy-based models for image tasks or Bayesian methods for variable selection (Hinton, 2012;Sun et al., 2023a)), and econometrics (e.g., MCMC-MLE for random graph models (Bhamidi et al., 2011;Byshkin et al., 2016)), a discrete state space can be represented as a graph. Here, nodes denote configurations, which are linked by edges via predefined Hamming distance. Efficient exploration yields quicker, more accurate system representation and inference. Specifically, informed proposals (Zanella, 2020) utilize local data to optimize acceptance-exploration balance but struggle with large-scale applicability. Gradientbased methods improve these proposals through techniques like Taylor series, Metropolis-adjusted Langevin, and Newton approximations (Grathwohl et al., 2021;Rhodes & Gutmann, 2022;Zhang et al., 2022;Sun et al., 2023b;Xiang et al., 2023). Nonetheless, they are designed for structured spaces and specific energy functions, unlike our focus on general graphs and arbitrary distributions µ, which renders gradient-based MCMC methods inapplicable.
All the aforementioned works are still rooted in Markov chains. A recent breakthrough is the self-repellent random walk (SRRW) over general graphs (Doshi et al., 2023), a nonlinear Markov chain that modifies a time-reversible Markov chain with µ-invariant transition kernel P , as shown by Box ① in Figure 1(a). 1 Unlike non-backtracking random walks relying solely on the most recent history (Alon et al., 2007;Lee et al., 2012), SRRW incorporates the entire past trajectory to adaptively adjust transition probabilities towards less-visited states while preserving convergence to µ. Formally, let G = (X , E) be a connected, undirected graph with node set X and edge set E, and let N (i) ≜ {j ∈ X | (i, j) ∈ E} be the neighbor set of state i.
this section cite: ['b66', 'b48', 'b59', 'b21', 'b25', 'b42', 'b51', 'b14', 'b68', 'b15', 'b19', 'b2', 'b45', 'b43', 'b1', 'b60', 'b16', 'b8', 'b61', 'b0', 'b37', 'b29', 'b24', 'b69', 'b30', 'b7', 'b13', 'b68', 'b24', 'b55', 'b69', 'b65', 'b20', 'b0', 'b37']

Section: Its cardinality is denoted |N (i)|.
Let N (i) ≜ N (i) ∪ {i} be the 'expanded' neighborhood of state i. Denote by x i ∈ (0, 1) the visit frequency of state i ∈ X . Given a timereversible kernel P with µ i P ij = µ j P ji where P ij > 0 only if (i, j) ∈ E, SRRW constructs a nonlinear kernel K[x], parametrized by x ≜ [x i ] i∈X , such that, for j ∈ N (i),
K ij [x] = Z -1 i P ij (x j /µ j ) -α ,(1)
where α ≥ 0 is the strength of self-repellency, and the normalizing constant Z i ≜ k∈N (i) P ik (x k /µ k ) -α . Note that K ij [x] = 0 for j / ∈ N (i), conforming to the graph structure. The transition probability to state j is dynamically enlarged or reduced from the original probability P ij based on the history so far via the ratio (x j /µ j ) -α between empirical measure and target distribution of state j (Box ② in Figure 1). The kernel (1) is also known to be 'scale-invariant' in the sense that it remains unchanged under any constant multiple of x or µ. This self-repellent scheme shows remarkable success, achieving near-zero variance for large α at a rate of O(1/α) for graph sampling (Doshi et al., 2023) and O(1/α 2 ) for distributed optimization (Hu et al., 2024a), and can even outperform i.i.d. sampling (i.e., a random jumper) while still walking on the graph.
this section cite: ['b20']

Section: Limitations of SRRW
Computational Issue. Despite its theoretical appeal and general applicability to reversible MCMC samplers, SRRW suffers from critical computational overhead that undermines its practicality, a shortfall overlooked in Doshi et al. (2023). To fully understand these challenges, it is important to first clarify the role of self-transition probability P ii in the time-reversible chain P and how it is used in SRRW. In the MH algorithm, from state i, another neighbor state j ∈ N (i) is proposed with probability Q ij and accepted with probability A ij ≜ min{1, µ j Q ji /µ i Q ij }. If accepted, the walker moves to j, making the transition probability P ij = Q ij A ij , keeping it reversible with respect to the given target µ. If rejected, the chain remains at i, contributing to the self-transition probability
P ii = 1-j̸ =i Q ij A ij .
Note that P ii > 0 unless A ij = 1 for all j ̸ = i, a rare case where the proposal Q is already µ-invariant. Importantly, P ii is just a byproduct out of the accept-reject mechanism, whose value is never explicitly pre-computed in standard MH algorithms. In addition, the acceptance ratio A ij is evaluated only for the proposed state j, ensuring a lightweight sequential implementation.
However, evaluating K ij [x] in (1) demands pre-computation of the normalizing constant Z i , which necessitates knowledge of P ij for all j ∈ N (i), including the self-transition probability P ii when j = i, implying that all A ij must be pre-calculated. This modification completely destroys the essence of MH, where the acceptance ratio is evaluated on demand only for a proposed state one at a time and is never pre-quantified for all possible j ∈ N (i). This issue becomes even more pronounced in scenarios with a large neighborhood size |N (i)|, where the computational cost scales accordingly. On the other hand, sampling j directly from K (i, * ) [x] ∝ P (i, * ) (x * /µ * ) -α as a target distribution over N (i) might be deemed as an alternative to bypass direct computation of the normalizing constant Z i . However, such approach also fails because it still requires knowledge of the target distribution up to a multiplicative constant, including P ii , defying the purpose of lightweight sampling. In short, P ii becomes a requirement in SRRW rather than a natural outcome of MH, translating to equivalent computational costs to sample from K (i, * ) [x], and thus offering no benefits.
this section cite: ['b20']

Section: Requirement of Time-Reversibility.
Another significant drawback of SRRW is its strict reliance on time-reversible Markov chains. While this reversibility ensures a welldefined stationary distribution for the modified kernel K[x] (Doshi et al., 2023, Proposition 2.1), it inherently excludes non-reversible MCMC techniques that still converge to the same target distribution µ with better performance (Neal, 2004;Suwa & Todo, 2010;Lee et al., 2012;Chen & Hwang, 2013;Maire et al., 2014;Ma et al., 2016;Bierkens, 2016;Andrieu & Livingstone, 2021).
this section cite: ['b20', 'b49', 'b60', 'b37', 'b16', 'b45', 'b43', 'b8', 'b1']

Section: Memory Constraints.
Despite being technically viable, SRRW encounters memory issues in large graphs and configuration spaces where the empirical measure x shares the dimensionality with the size of the state space. Retaining complete historical data through x at each step can exceed memory capacity in simulations. Thus, it is natural to ask:
Can we design a universal method to harness SRRW benefits, while tackling these challenges: (i) maintain computational efficiency, (ii) leverage both reversible and non-reversible MCMC samplers, and (iii) reduce memory usage?
this section cite: []

Section: Our Approach and Contributions
In this work, we tackle the three aforementioned challenges arising from nonlinear kernels of SRRW. Specifically, we theoretically resolve two of them: computational issue and incompatibility with non-reversible Markov chains, and propose a heuristic scheme that reduces storage requirements for memory issue. Comprehensive applications to exponentially large state spaces, i.e., high-dimensional problems, are deferred to future work.
In SRRW, the self-repellent mechanism is embedded directly into the transition kernel K[x], leading to significant computational costs, as discussed earlier. However, the essence of a history-dependent scheme need not reside in the kernel itself. As depicted in Box ③ of Figure 1(b), we shift this scheme to a family of 'history-driven' target (HDT) distributions. Specifically, we replace the original target µ, a single point in the probability simplex, with an adaptive distribution π[x] parameterized by the empirical measure x, which evolves dynamically at each step based on the 'feedback' from past visits. The exact formulation of π[x] in (6) ensures that less/more-visited state i is assigned higher/lower probabilities than the original target µ i , dynamically over time. Our approach allows for seamless integration with any MCMC technique, using the target π[x] for a given x. Unlike SRRW, in our framework, we can reuse any advanced MCMC technique by simply replacing target µ by our design π[x] in both reversible and nonreversible samplers, retaining the lightweight, sequential, and adaptive nature of MCMC methods, as shown in Box ④ of Figure 1. By decoupling the self-repellent mechanism from the transition kernel, we overcome the computational limitations of SRRW while achieving near-zero variance. This paradigm shift offers the best of both worlds: computational efficiency and broad compatibility with advanced MCMC samplers. Our contributions are as follows.
1. A General Self-Repellent Framework: We introduce the HDT framework, which replaces the target µ by the history-driven target distribution π[x] via rigorous design. This formulation is compatible with both reversible and non-reversible MCMC samplers while eliminating the computational overhead in SRRW. 2. Theoretical Guarantees: We prove that our HDT framework converges almost surely to the original target µ and achieves O(1/α) variance reduction compared to the base sampler in the central limit theorem (CLT). Moreover, we establish a cost-based CLT, show-ing an additional O(1/E[|N (i)|]) variance reduction relative to SRRW by the average neighborhood size. 3. Empirical Evaluation: We perform extensive graph sampling simulations across various settings, including real-world graphs and both reversible and nonreversible MCMC samplers, and showcase the consistent efficiency and adaptability of our HDT framework. 4. Scalable Implementation for Large Graphs: We implement an LRU (Least Recently Used) cache scheme to manage x with limited memory. For example, even when maintaining visit frequencies for only 10% of states, our method leads to more than a 10% reduction in total variation distance over the original MCMC method, highlighting HDT's effectiveness in limitedmemory environments.
this section cite: []

Section: Preliminaries
Basic Notations. Let X denote a finite discrete state space. Vectors are denoted by lower-case bold letters, e.g., v ≜ [v i ] i∈X , and matrices by upper-case bold, e.g., M ≜ [M ij ] i,j∈X . The diagonal matrix D v is constructed from the vector v, with its components placed along the main diagonal. We denote by 1, 0 vectors of all ones and zeros with proper dimensions, respectively. Denote by Σ the |X |-dimensional probability simplex over X , with Int(Σ) being its interior, i.e., x ∈ Int(Σ) implies that x i ∈ (0, 1), ∀i ∈ X . For a probability vector x ∈ Σ, we write x to denote any of its unnormalized counterparts, i.e., x = x/(1 T x) ∝ x. Define δ i as the canonical vector whose components are all zero, except the i-th entry being one. Let N (0, V ) represent the multivariate Gaussian distribution with zero mean and covariance matrix V . We use ---→ dist.
for weak convergence.
this section cite: []

Section: Ergodic Markov Chains.
Consider an ergodic Markov chain with transition kernel P ∈ R |X |×|X | and target distribution µ ∈ Int(Σ), satisfying P 1 = 1 and µ T P = µ T . In MCMC, the transition kernel P is defined by the target µ.
Throughout this work, we assume that P is full-rank and continuous in µ. Denote by (λ i , u i , v i ) the eigenpair of P , comprising the eigenvalues as well as the corresponding left and right eigenvectors. Here, the Perron-Frobenius eigenvalue λ 1 = 1 with its corresponding eigenvectors u 1 = µ and v 1 = 1. In addition, u T i v i = 1 and u T i v j = 0 for all i, j ∈ [2, |X |] with i ̸ = j. Consider the sample path {X s } s≥1 driven by an ergodic Markov chain on X , with δ(•) as the indicator function. The cumulative visit count to state i by time n is expressed as
xn (i) ≜ n s=1 δ(X s = i), ensuring i∈X xn (i) = n. The empirical measure x n ≜ [x n (i)] i∈X is the normalized version of xn , hence x n = xn /(1 T xn ) = xn /n.
Alternatively, we can express x n iteratively as follows:
x n+1 = x n + 1 n + 1 (δ Xn+1 -x n ).(2)
The ergodic theorem for Markov chains (Brémaud, 2013, Theorem 3.3.2) states that the empirical measure x n of an ergodic Markov chain, updated via (2), almost surely converges to µ as n → ∞. In addition, the multivariate CLT (Brooks et al., 2011, Chapter 1.8.1) shows that √ n(x n -µ) weakly converges to N (0, V ), where the covariance matrix
V = lim t→∞ 1 t E[( t s=1 (δ Xs -µ))( t s=1 (δ Xs -µ)) T ], (3)
which serves as the covariance matrix of both reversible and non-reversible MCMC samplers in Theorem 3.3. By Brémaud (2013, Chapter 6.3.3), V can be rewritten as
V = |X | i=2 1 + λ i 1 -λ i D µ v i u T i . (4
)
When P is reversible, the property
u i = D µ v i gives V = |X | i=2 1+λi 1-λi u i u T i .
See Appendix A for the derivation of (4). Properties of SRRW. The SRRW algorithm in Doshi et al. (2023) includes two steps at each time n: First, sample X n+1 ∼ K (Xn,•) [x n ] in (1) using the current empirical measure x n and state X n ; Second, update x n+1 via (2). This ensures that the SRRW kernel K[x] is adapted to the updated empirical measure x n at each time n, making it a nonlinear Markov chain. A notable property of SRRW is 'scale-invariance', i.e., for any non-zero scalar C,
K ij [Cx] = K ij [x], ∀i, j ∈ X , such that computing K ij [x]
only requires knowing vectors x, µ up to some constant multiples. It has been theoretically proved that the empirical measure x n → µ almost surely as n → ∞, and the scaled error √ n(x n -µ) n→∞ ----→ dist.
N (0, V SRRW (α)), where
V SRRW (α) = |X | i=2 1 2α(λ i + 1) + 1 • 1 + λ i 1 -λ i u i u T i , (5
)
and λ i , u i are eigenvalues and left eigenvectors of the timereversible base MCMC kernel P leveraged by SRRW.
this section cite: ['b20']

Section: Main Results
We present our theoretical findings by first outlining the design principles for a HDT distribution that balances exploration with computational efficiency. Next, we examine the convergence and statistical properties of the HDT-MCMC algorithm. Lastly, we evaluate its computational cost against SRRW within a fixed budget using a cost-based CLT, showcasing the performance benefits of HDT-MCMC.
this section cite: []

Section: Design of History-Driven Target Distribution
The foundation of HDT-MCMC is in the design of a historydriven target π[x, µ] that relies solely on the visit count x and an unnormalized target μ (as an input), while encapsulating self-repellent behavior in the resulting kernel. Our goal is to design the HDT π[x, µ] to satisfy the following four conditions:
C1 (Scale Invariance). π i [x, µ] = π i [x, μ].
this section cite: []

Section: C2 (Local Dependence). The unnormalized term πi [x, µ]
depends only on µ i and x i , and is continuous in x i , µ i .
Henceforth, we suppress µ and simply write π[x] for brevity, whenever its dependence on µ is clear from the context.
C3 (Fixed Point). π[µ] = µ.
this section cite: []

Section: C4 (History Dependence). π[x]
promotes/deters exploration of under/over-sampled states.
C1 preserves the key property that our HDT-MCMC algorithm only needs the unnormalized terms μ, and x when, for instance, computing an acceptance ratio of MH. Examples of the unnormalized target μ include: μi = 1 for uniform target, μi = |N (i)| for degree-based target, and μi = e -H(i) for energy-specific target (Gjoka et al., 2011;Lee et al., 2012;Grathwohl et al., 2021;Pynadath et al., 2024). C2 ensures that the sampler uses only minimal local information, i.e., xi and μi , in its update. By eliminating neighbor information, the algorithm avoids prohibitive computational costs associated with increasing neighborhood size, e.g., in large or complete graphs. C3 indicates that if empirical measure x converges to the target µ, the HDT π[µ] remains µ itself. Indeed, if π[µ] ̸ = µ, then an MCMC sampler with its target π[x] would converge to a different distribution, contradicting our goal of preserving µ as the true target distribution. C4 is to facilitate exploration, ensuring that the sampler adaptively 'pushes' itself away from states already visited more often than µ i , thereby mimicking the key self-repellent effect of SRRW in Doshi et al. (2023). Lemma 3.1. Conditions C1 -C4 hold if and only if
π i [x] ∝ µ i (x i /µ i ) -α for any α > 0. (6
)
The proof of Lemma 3.1 can be found in Appendix B. It reveals that an HDT distribution π[x] satisfying all four conditions C1 -C4 must take the simple form of (6). A special case α = 0 reduces π[x] to the original target µ, which becomes history-independent. Lemma 3.1 also demonstrates that our design (6) suffices to work with unnormalized quantities μ and x in place of µ and x, i.e.,
π i [x] ∝ πi [x] = μi (x i /μ i ) -α .(7)
Following Lemma 3.1, Algorithm 1 shows the steps in our HDT-MCMC framework. 2 As illustrated by Box ④ in Figure 1, the base MCMC sampler for graph sampling in our framework can either be time-reversible, i.e., MH (Metropolis et al., 1953;Hastings, 1970), MHDR (Green & Mira, 2001), MTM (Liu et al., 2000;Chang et al., 2022), or nonreversible, i.e., MHDA (Lee et al., 2012), 2-cycle Markov chains (Maire et al., 2014;Andrieu & Livingstone, 2021), and non-reversible MH (Bierkens, 2016;Thin et al., 2020).
Algorithm 1 HDT-MCMC: Graph Sampling Framework Input: Graph G(X , E), parameter α ≥ 0, unnormalized target μ, number of iterations T , a base MCMC sampler (Bring Your Own MCMC). Initialization: state X 0 ∈ X , visit count x(i) > 0, ∀i ∈ X . for n = 0 to T -1 do Step 1: Use the base sampler (reversible or nonreversible) to draw X n+1 with history-driven target π[x] from (7). Step 2: Update visit count x(X n+1 ) ← x(X n+1 )+ 1; end for Output: A set of samples {X n } T n=1 . As an example, we here illustrate our HDT-MCMC if we use the standard MH algorithm as the base MCMC sampler (Step 1) in Algorithm 1.foot_2 At current state X n = i, a candidate j ∈ N (i) is selected with probability Q ij , then the acceptance ratio is calculated through
A ij [x]= min 1, π j [x]Q ji π i [x]Q ij = min 1, μj (x j /μ j ) -α Q ji μi (x i /μ i ) -α Q ij ,(8)
where only the unnormalized terms μi , μj , xi , and xj are required, keeping the same computational cost as standard MH with true target µ. Then, the sampler accepts state j with probability A ij [x] and sets X n+1 = j, or rejects it with probability 1 -A ij [x] upon which X n+1 = i and repeats the procedure. Note that we recover the acceptance ratio A ij of the standard MH with target µ, when α = 0. This manner alters the target µ of the standard MH algorithm by HDT π[x] while preserving the lightweight, on-demand nature of MH, in contrast to SRRW in which P ij must be evaluated for all possible j ∈ N (i) for a sample X n+1 .
In addition to computational efficiency, A ij [x] inherently embeds the 'self-repellent' effect. If state j is relatively less-visited than state i, i.e., xj /μ j < xi /μ i , it then follows that A ij [x] ≥ A ij , implying that state j is more likely to be accepted than the case with standard MH, and vice versa. Remark 3.1. In Doshi et al. (2023), SRRW directly modifies a time-reversible Markov chain P to incorporate selfrepellency into the kernel K[x] as in (1), which is then shown to be, for any given x ∈ Int(Σ), reversible w.r.t.
π SRRW i [x] ∝ µ i (x i /µ i ) -α j∈N (i) P ij (x j /µ j ) -α , ∀i ∈ X , whose proof in Doshi et al. (2023, Appendix A) critically de- pends on the reversibility of P w.r.t. µ. Note that π SRRW i [x]
is the byproduct of the constructed kernel K[x] as in (1), thus inheriting the same neighborhood dependency. Simply adopting π SRRW [x] in our Algorithm 1 violates C2 and would incur high computational cost for resulting nonlinear kernels. In contrast, our HDT (6) decouples neighbors in the target distribution itself, which eliminates the need to evaluate transition probabilities for all neighbors at each step, offering substantial computational savings and compatibility with both reversible and non-reversible samplers.
this section cite: ['b23', 'b37', 'b24', 'b54', 'b20', 'b47', 'b27', 'b25', 'b42', 'b14', 'b37', 'b45', 'b1', 'b8', 'b61', 'b20']

Section: Performance of HDT-MCMC
We next analyze HDT-MCMC regarding (i) almost-sure convergence of the empirical measure x n to µ, and (ii) an O(1/α) variance reduction relative to the base MCMC sampler with true target µ.
Observe that (2) allows us to decompose x n+1 as
x n+1 = x n + 1 n + 1 [(π[x n ]-x n ) deterministic drift +(δ Xn+1 -π[x n ]) noise term ],
which is the standard step in stochastic approximation (SA) with controlled Markovian dynamics (Kushner & Yin, 2003;Benveniste et al., 2012;Borkar, 2022). It can be viewed as combining a deterministic drift π[x n ] -x n towards the solution of the ODE
ẋ(t) = π[x(t)] -x(t)(9)
and a noise term δ Xn+1 -π[x n ]. The ODE viewpoint clarifies the global asymptotic stability of µ, while the noise term characterizes fluctuations around µ. Lemma 3.2. For π[x] in (6), the ODE (9) has a unique fixed point µ, which is globally asymptotically stable.
Proofs follow by showing that target µ is the unique stable equilibrium and employing standard Lyapunov stability theory, as detailed in Appendix D. For the iteration x n in (2), we need to additionally account for the noise term driven by a history-dependent MCMC from Algorithm 1. We impose the following assumption on x n throughout the paper. Assumption 1. x n ∈ Int(Σ) for all n ≥ 0 almost surely.
Assumption 1 guarantees x n (i) > 0 almost surely for all i ∈ X and n ≥ 0, keeping π[x] well-defined at each step. This type of assumption is standard in the SA literature (Fort, 2015;Borkar, 2022;Li et al., 2023). In practice, it can be enforced using truncation-based methods within (2), as discussed in Doshi et al. (2023, Remark 4.5 and Appendix E). These methods effectively prevent any component x n (i) from approaching zero, thus maintaining x n ∈ Int(Σ). Under this assumption, we obtain the following: Theorem 3.3 (Ergodicity and CLT). HDT-MCMC in Algorithm 1 satisfies
(a) x n → µ almost surely as n → ∞. (b) √ n(x n -µ) n→∞ ----→ dist.
N (0, V HDT (α)), where
V HDT (α) = 1 2α + 1 V base ,(10)
and V base is the limiting covariance of the base MCMC sampler (reversible or non-reversible) with target µ in (3).
The full proof of Theorem 3.3 is in Appendix E, where we leverage the existing asymptotic analysis from the SA literature (Delyon et al., 1999;Fort, 2015), similarly used in Doshi et al. (2023, Appendix C). However, the primary technical challenge arises from handling non-reversible Markov chains, which is excluded from Doshi et al. (2023) by the nature of their kernel design. We proceed with our analysis that is specifically tailored to the augmented state space on which the non-reversible Markov chain is defined, and solve a mismatch issue between the augmented space and the original space by only tracking the marginal empirical measure x ∈ Int(Σ) in the original space X . Theorem 3.3 highlights two appealing features of HDT-MCMC: (i) It preserves the unbiased sampling by converging to the true target µ. (ii) It offers an O(1/α) variance reduction, achieving a similar near-zero variance phenomenon of SRRW but without additional computational overhead or the need for time-reversibility. When α = 0, we recover the baseline scenario π[x] ≡ µ and V HDT (0) = V base , as expected.
Moreover, Theorem 3.3 leads to an instant result as follows.
Corollary 3.4. Suppose two MCMC samplers S 1 and S 2 converge to µ with limiting covariances V S1 and
V S2 sat- isfying V S1 ⪯ V S2
. 4 Applying HDT framework to both, yielding V S 1 -HDT (α) and V S 2 -HDT (α), preserves the ordering:
V S 1 -HDT (α) ⪯ V S 2 -HDT (α), ∀α ≥ 0.
The proof is straightforward from ( 10). Hence, any known covariance orderings between reversible and non-reversible samplers (see Lee et al. (2012); Maire et al. (2014); Bierkens (2016); Andrieu & Livingstone (2021)) carry over to our HDT-MCMC framework, whereas SRRW cannot accommodate non-reversible Markov chains.
this section cite: ['b36', 'b6', 'b10', 'b22', 'b10', 'b40', 'b18', 'b22', 'b20', 'b37', 'b45']

Section: Comparative Analysis of Computational Costs
We now compare the performance of our HDT-MCMC to SRRW (Doshi et al., 2023) under a fixed total computational budget B. Although both methods achieve an O(1/α) variance reduction, HDT-MCMC requires significantly less computation per sample.
Because SRRW mandates a time-reversible base MCMC sampler, we restrict our comparison to the same reversible chain (as illustrated in Boxes ② and ④ of Figure 1). Let a i (resp. b i ) ∈ (0, ∞) be the computational cost of the i-th sample in HDT-MCMC (resp. SRRW). Define:
T HDT (B) ≜ max{k | a 1 + a 2 + • • • + a k ≤ B} T SRRW (B) ≜ max{k ′ | b 1 + b 2 + • • • + b k ′ ≤ B} so that T HDT (B) (resp. T SRRW (B)
) represents the total number of samples that HDT-MCMC (resp. SRRW) can generate before hitting the budget B. Intuitively, under the same budget, SRRW's higher per-sample cost yields fewer total samples. To quantify this effect, we now compare these two frameworks under the same but large amount of total budget B instead of the number of samples as done in Section 3.2. Then, we have
√ B(x T HDT (B) -µ) B→∞ ----→ dist. N (0, C HDT V HDT (α)) (11) √ B(y T SRRW (B) -µ) B→∞ ----→ dist. N (0, C SRRW V SRRW (α)) (12
)
where V HDT (α) is given by (10), and V SRRW (α) by (5).
We leverage the random-change-of-time theory (Billingsley, 2013) and Slutsky's theorem (Ash & Doléans-Dade, 2000) to transform our time-based CLT (Theorem 3.3) to the costbased CLT (Theorem 3.5), with details in Appendix F.
In practice, HDT-MCMC's per-sample cost C HDT can be significantly smaller than C SRRW because the latter must precompute the transition probability K ij [x] at each step. As a concrete example, we focus on the MH framework as the base sampler utilized in many advanced MCMC schemes (Liu et al., 2000;Green & Mira, 2001;Lee et al., 2012;Bierkens, 2016;Zanella, 2020;Chang et al., 2022). Assuming that computing the proposal probability Q ij and inquiring μi incurs c units of cost per each pair (i, j) ∈ E, it requires 2c costs for A ij [x] in (8). Thus, HDT-MCMC spends 2c per sample, whereas SRRW incurs 2c|N (i)| subject to state i, as discussed in Section 1.2. The following lemma shows the ordering of their cost-based covariances:
Lemma 3.6. The cost-based covariances between SRRW and HDT-MCMC in (11) and (12) are ordered as follows:
C HDT V HDT (α) ⪯ (2/E i∼µ [|N (i)|]) • C SRRW V SRRW (α).
See Appendix G for the proof. Lemma 3.6 implies that the cost-based covariance of HDT-MCMC is at least a factor of 2/E[|N (i)|] times smaller than that of SRRW in Loewner ordering for each α, suggesting a universal advantage. This factor becomes more pronounced in dense or nearly complete graphs, where the average neighborhood size is
E[|N (i)|] ≫ 2.
For unbiased graph sampling, at each time T , the sampling agent records state X T , obtains the value f (X T ) ∈ R, and updates the unbiased MCMC estimator ψ T (f ), aiming to approximate the ground truth f , such as a global attribute of the unknown graph, and their expressions are defined in the following:
ψ T (f ) ≜ 1 T T s=1 f (X s ), f ≜ i∈X µ i f (i).
Equivalently, we can rewrite the MCMC estimator as ψ T (f ) = f T x T since x T = 1 T T s=1 δ Xs and f T δ i = f (i), where f ≜ [f (i)] i∈X ∈ R |X | . Thus, the cost-based variance of ψ T (f ) for HDT-MCMC and SRRW is derived by left multiplying f T into (11) and ( 12), yielding
√ T (ψ HDT T (f ) -f ) T →∞ ----→ dist. , N (0, Var HDT (α)), (13
) √ T (ψ SRRW T (f ) -f ) T →∞ ----→ dist. N (0, Var SRRW (α)),(14)
where
Var HDT (α) = C HDT f T V HDT (α)f , and Var SRRW (α) = C SRRW f T V SRRW (α)f
, respectively. By Lemma 3.6 and the Loewner ordering in footnote 4, we have for any α > 0,
Var HDT (α) ≤ (2/E[|N (i)|]) • Var SRRW (α),
which translates into at least a factor of 2/E[|N (i)|] times smaller cost-based variance of the MCMC estimator ψ T (f ) than the case with SRRW. In addition, unlike SRRW, HDT-MCMC also accommodates non-reversible base samplers (see Corollary 3.4), further enhancing efficiency in many applications.
this section cite: ['b20', 'b9', 'b3', 'b42', 'b25', 'b37', 'b8', 'b68', 'b14']

Section: Simulations
We design a series of experiments to evaluate the performance of HDT-based MCMC methods in graph sampling tasks. Our goal is to compare HDT-MCMC with various advanced MCMC algorithms, including both reversible and non-reversible Markov chains, and compare HDT-MCMC with SRRW within the same total computational budget.
this section cite: []

Section: Simulation Setup
We conduct experiments on two real-world graphs, i.e., facebook (4039 nodes with 88234 edges) and p2p-Gnutella04 (10876 nodes with 39994 edges) from SNAP (Leskovec & Krevl, 2014). We use a uniform target distribution µ = 1 |X | 1 throughout this section while deferring the experiments of non-uniform target to Appendix H.6. In the reversible setting, we apply the standard MH algorithm (MHRW) and MTM with locally balanced weights and K = 3 proposed candidates (Chang et al., 2022). In the non-reversible setting, we adopt MHDA (Lee et al., 2012). Additional experiments on WikiVote, p2p-Gnutella08, and non-reversible 2-cycle Markov chains appear in Appendix H. To assess conver- gence, we use the total variation distance
TVD(x n , µ) ≜ 1 2 ∥x n -µ∥ 1
for the distance between the empirical measure x n of the collected samples and the target µ. We also evaluate normalized root mean squared error
NRMSE(ψ n , ψ) = E[(ψ n (f )-ψ) 2 ]/ ψ
for graph-based group-size estimation with test function f defined in Appendix H.2. Each experiment consists of 1000 independent runs, and one-third of the total iterations is used as the burn-in period.
this section cite: ['b38', 'b14', 'b37']

Section: Comparison of Base MCMC and its HDT Version
We first compare each base MCMC algorithm with its HDTenhanced version, setting α = 5 in the target π[x] in (6).
Figure 2 shows the average TVD and NRMSE for MTM and MHDA, with MHRW serving as a benchmark. In both cases, MTM and MHDA outperform MHRW when targeting µ, which aligns with the theoretical results in Chang et al. (2022) and Lee et al. (2012). Moreover, the HDT-enhanced versions (HDT-MTM and HDT-MHDA) consistently attain lower TVD and NRMSE than their respective base algorithms, indicating faster convergence to µ, consistent with Theorem 3.3. A similar trend is observed in other graphs using TVD and NRMSE metrics (see Appendix H.2). Notably, SRRW (with MHRW as its base) achieves the lowest TVD and NRMSE among all methods but entails substantially higher computational overhead to obtain one sample, whose effect is not reflected in the number of steps. We shall examine SRRW's performance under fixed budget in the next experiment. We do not combine SRRW with MTM in this experiment due to its heavy computation in P ij for all possible combinations of K intermediate proposed candidates, whereas HDT integrates seamlessly into MTM without additional cost. Moreover, we conduct the experiment on the effect of different α values influence HDT-MHRW algorithm in Appendix H.4 and observe consistent improvement with smaller TVD using larger α.
this section cite: ['b14', 'b37']

Section: Robustness to Different Initializations
To demonstrate that HDT-MCMC is robust to different initializations, we evaluate its performance when initialized at various nodes, using base chains MHRW, MTM, and MHDA for the HDT-enhanced versions. We show the experimental results for the Facebook graph in Table 1 and include the results for other graphs in Appendix H.3. In particular, we examine the effects of both initial state X 0 and the fake visit counts x 0 . The initiate state X 0 is randomly chosen from either low-degree group (where the node's degree is smaller than the average degree) or high-degree group. This is to test the sensitivity of our algorithm starting from sparse or dense regions. On the other hand, fake visit count x 0 is initialized using one of the following settings: 'Deg' (proportional to node degree, e.g., xi = |N (i)| for all i ∈ [N ]), 'Non-unif' (a non-uniform draw from a Dirichlet distribution with default hyperparameter 0.5), and 'Unif' (same initial count across all nodes, e.g., xi = 1 for all i ∈ [N ]). In Table 1, both TVD and NRMSEfoot_4 show consistent performance when starting from either the low-degree group or the high-degree group. Similarly, a robust result is observed when using different settings of fake visit counts.
this section cite: []

Section: Computational Cost Comparison with SRRW
We compare HDT-MHRW and SRRW under a fixed computational budget B. At each iteration, SRRW needs to compute or retrieve transition probabilities for every neighbor due to the nature of the kernel design (1), whereas HDT only updates the proposal for a single candidate. This com-putational cost aligns with the concerns of the O(N (i)) evaluations in the proposal distribution highlighted in Zanella (2020); Grathwohl et al. (2021) in high-dimensional spaces.
In graph sampling, simply counting samples to assess performance can be misleading under rate-limited API constraints, e.g., online social network sampling (Xu et al., 2017;Li et al., 2019). The performance gap between HDT-MHRW and SRRW in budget B increases with the degree of the node, as proved in Lemma 3.6. Figure 3 shows that under the same total budget, HDT-MHRW consistently achieves lower TVD and NRMSE than SRRW. The discrepancy is even more pronounced in the Facebook graph in both plots, where the average degree (43.6) far exceeds that of p2p-Gnutella04 (7.4), supporting our discussion after Lemma 3.6, where larger neighborhood size leads to more performance advantage of HDT-MCMC compared to SRRW.  The essential idea is to track only recently visited states, discarding the least-recently used when capacity in cache C is reached. This leverages temporal locality, as nonneighboring states do not affect self-repellency. Unlike sparsification methods (Verma et al., 2024;Barnes et al., 2024), which ignore the walker's position, LRU's local simplicity aligns with our HDT-MCMC. For a neighbor j / ∈ C of current state i, we approximate its frequency via
xj = μj |N (i) ∩ C| -1 k∈N (i)∩C xk /μ k .(15)
This approach approximates xj /μ j , which mimics (6) via temporal closeness among states frequently visited around i, allowing the sampler to maintain self-repellency without true visit count. We examine HDT-MCMC using an LRU cache with size |C| = r|X | where 0 < r < 1. In Figure 4, the average TVD of HDT-MHRW with LRU outperforms MHRW even without exact empirical measure due to it limited capacity. In most scenarios, HDT-MHRW with LRU leads to 10% smaller TVD than the base MHRW with over 90% memory reduction. The performance of HDT-MHRW with LRU is robust to the choice of r in most cases. Due to space constraint, we defer more result of LRU scheme in plus-combined graph (over 100K nodes) to Appendix H.5.
2000 4000 6000 8000 100001200014000 Steps 0.4 0.5 0.6 0.7 0.8 0.9 1.0 TVD facebook Graph MHRW HDT-MHRW ( = 5) with r=0.01 HDT-MHRW ( = 5) with r=0.05 HDT-MHRW ( = 5) with r=0.10 HDT-MHRW ( = 5) with r=0.15 HDT-MHRW ( = 5) with r=0.20 HDT-MHRW ( = 5) with r=1 5000 10000 15000 20000 25000 30000 Steps 0.4 0.5 0.6 0.7 0.8 0.9 1.0 TVD p2p-Gnutella04 Graph MHRW HDT-MHRW ( = 5) with r=0.01 HDT-MHRW ( = 5) with r=0.05 HDT-MHRW ( = 5) with r=0.10 HDT-MHRW ( = 5) with r=0.15 HDT-MHRW ( = 5) with r=0.20 HDT-MHRW ( = 5) with r=1
this section cite: ['b24', 'b67', 'b41', 'b63', 'b5']

Section: Conclusion
In this paper, we propose a history-driven target (HDT) framework for MCMC sampling on general graphs. By embedding self-repellency in the target rather than the transition kernel, HDT maintains unbiased sampling with a lightweight design and provides an O(1/α) variance reduction without high computational cost or time-reversibility constraints from SRRW. Our theoretical analysis covers both reversible and non-reversible MCMC samplers, while empirical results on real-world graphs show robust performance gains and reduced computational overhead. To handle memory limitations, we introduce a Least Recently Used (LRU) cache scheme, enabling partial tracking of the empirical measure without loss in sampling efficiency. Future directions include more refined memory approximations for exponentially large configuration spaces and applications to high-dimensional statistical inference and network analysis.
this section cite: []

Section: References
Ref_id:b0 Title: Nonbacktracking random walks mix faster Year: (2007)
Ref_id:b1 Title: Peskun-tierney ordering for markovian monte carlo: beyond the reversible scenario Year: (2021)
Ref_id:b2 Title: Lifting markov chains to mix faster: limits and opportunities Year: (2017)
Ref_id:b3 Title: Probability and measure theory Year: (2000)
Ref_id:b4 Title: Walk for learning: A random walk approach for federated learning from heterogeneous data Year: (2023)
Ref_id:b5 Title: Efficient unbiased sparsification Year: (2024)
Ref_id:b6 Title: Adaptive algorithms and stochastic approximations Year: (2012)
Ref_id:b7 Title: Mixing time of exponential random graphs Year: (2011)
Ref_id:b8 Title: Non-reversible metropolis-hastings Year: (2016)
Ref_id:b9 Title: Convergence of probability measures Year: (2013)
Ref_id:b10 Title: Stochastic Approximation: A Dynamical Systems Viewpoint: Second Edition. Texts and Readings in Mathematics Year: (2022)
Ref_id:b11 Title: Markov chains: Gibbs fields, Monte Carlo simulation, and queues Year: (2013)
Ref_id:b12 Title: Handbook of Markov Chain Monte Carlo Year: (2011)
Ref_id:b13 Title: Auxiliary parameter mcmc for exponential random graph models Year: (2016)
Ref_id:b14 Title: Rapidly mixing multiple-try metropolis algorithms for model selection problems Year: (2022)
Ref_id:b15 Title: Lifting markov chains to speed up mixing Year: (1999)
Ref_id:b16 Title: Accelerating reversible markov chains Year: (2013)
Ref_id:b17 Title: Stochastic approximation with decreasing gain: Convergence and asymptotic theory Year: (2000)
Ref_id:b18 Title: Convergence of a stochastic approximation version of the em algorithm Year: (1999)
Ref_id:b19 Title: Analysis of a nonreversible markov chain sampler Year: (2000)
Ref_id:b20 Title: Self-repellent random walks on general graphs-achieving minimal sampling variance via nonlinear markov chains Year: (2023)
Ref_id:b21 Title: Stochastic gradient descent under markovian sampling schemes Year: (2023)
Ref_id:b22 Title: Central limit theorems for stochastic approximation with controlled markov chain dynamics Year: (2015)
Ref_id:b23 Title: Practical recommendations on crawling online social networks Year: (2011)
Ref_id:b24 Title: Oops i took a gradient: Scalable sampling for discrete distributions Year: (2021)
Ref_id:b25 Title: Delayed rejection in reversible jump metropolis-hastings Year: (2001)
Ref_id:b26 Title: node2vec: Scalable feature learning for networks Year: (2016)
Ref_id:b27 Title: Monte carlo sampling methods using markov chains and their applications Year: (1970)
Ref_id:b28 Title: A principled framework for the design and analysis of token algorithms Year: (2023)
Ref_id:b29 Title: Reversibility of the non-backtracking random walk Year: (2019)
Ref_id:b30 Title: A practical guide to training restricted boltzmann machines Year: (2012)
Ref_id:b31 Title: Efficiency ordering of stochastic gradient descent Year: (2022)
Ref_id:b32 Title: Accelerating distributed stochastic optimization via self-repellent random walks Year: (2024)
Ref_id:b33 Title: Does worst-performing agent lead the pack? analyzing agent dynamics in unified distributed sgd Year: (2024)
Ref_id:b34 Title: Nonlinear Systems Year: (2002)
Ref_id:b35 Title: Revisiting random walks for learning on graphs Year: (2024)
Ref_id:b36 Title: Stochastic approximation and recursive algorithms and applications Year: (2003)
Ref_id:b37 Title: Beyond random walk and metropolis-hastings samplers: why you should not backtrack for unbiased graph sampling Year: (2012)
Ref_id:b38 Title: Stanford large network dataset collection Year: (2014-06)
Ref_id:b39 Title: Graphs over time: densification laws, shrinking diameters and possible explanations Year: (2005)
Ref_id:b40 Title: Online statistical inference for nonlinear stochastic approximation with markovian data Year: (2023)
Ref_id:b41 Title: Walking with perception: Efficient random walk sampling via common neighbor awareness Year: (2019)
Ref_id:b42 Title: The multiple-try method and local optimization in metropolis sampling Year: (2000)
Ref_id:b43 Title: A unifying framework for devising efficient and irreversible mcmc samplers Year: (2016)
Ref_id:b44 Title: The internet as-level topology: three data sources and one definitive metric Year: (2006)
Ref_id:b45 Title: Comparison of Asymptotic Variances of Inhomogeneous Markov Chains with Application to Markov Chain Monte Carlo Methods Year: (2014)
Ref_id:b46 Title: Random walks and diffusion on networks Year: (2017)
Ref_id:b47 Title: Equation of state calculations by fast computing machines Year: (1953)
Ref_id:b48 Title: Random walk sampling in social networks involving private nodes Year: (2023)
Ref_id:b49 Title: Improving asymptotic variance of mcmc estimators: Non-reversible chains are better Year: (2004)
Ref_id:b50 Title: Web crawling Year: (2010)
Ref_id:b51 Title: A generalization of the multiple-try metropolis algorithm for bayesian estimation and model selection Year: (2010)
Ref_id:b52 Title: Online learning of social representations Year: (2014)
Ref_id:b53 Title: A survey on active simultaneous localization and mapping: State of the art and new frontiers Year: (2023)
Ref_id:b54 Title: Gradient-based discrete sampling with automatic cyclical scheduling Year: (2024)
Ref_id:b55 Title: Enhanced gradient-based mcmc in discrete spaces Year: (2022)
Ref_id:b56 Title: Monte Carlo statistical methods Year: (2013)
Ref_id:b57 Title: Any-scale balanced samplers for discrete space Year: (2023)
Ref_id:b58 Title: Discrete langevin samplers via wasserstein gradient flow Year: (2023)
Ref_id:b59 Title: On markov chain gradient descent Year: (2018)
Ref_id:b60 Title: Markov chain monte carlo method without detailed balance Year: (2010)
Ref_id:b61 Title: Nonreversible mcmc from conditional invertible transforms: a complete recipe with convergence guarantees Year: (2020)
Ref_id:b62 Title: Decentralized learning with random walks and communication-efficient adaptive optimization Year: ()
Ref_id:b63 Title: Sparsifying count sketch Year: (2024)
Ref_id:b64 Title: Random walks: A review of algorithms and applications Year: (2019)
Ref_id:b65 Title: Efficient informed proposals for discrete distributions via newton's series approximation Year: (2023)
Ref_id:b66 Title: Optimizing random walk based statistical estimation over graphs via bootstrapping Year: (2021)
Ref_id:b67 Title: Challenging the limits: Sampling online social networks with cost constraints Year: (2017)
Ref_id:b68 Title: Informed proposals for local mcmc in discrete spaces Year: (2020)
Ref_id:b69 Title: A langevin-like sampler for discrete distributions Year: (2022)
