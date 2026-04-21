Title: Adjoint Schrödinger Bridge Sampler
Abstract: Computational methods for learning to sample from the Boltzmann distributionwhere the target distribution is known only up to an unnormalized energy functionhave advanced significantly recently. Due to the lack of explicit target samples, however, prior diffusion-based methods, known as diffusion samplers, often require importance-weighted estimation or complicated learning processes. Both trade off scalability with extensive evaluations of the energy and model, thereby limiting their practical usage. In this work, we propose Adjoint Schrödinger Bridge Sampler (ASBS), a new diffusion sampler that employs simple and scalable matchingbased objectives yet without the need to estimate target samples during training. ASBS is grounded on a mathematical model-the Schrödinger Bridge-which enhances sampling efficiency via kinetic-optimal transportation. Through a new lens of stochastic optimal control theory, we demonstrate how SB-based diffusion samplers can be learned at scale via Adjoint Matching and prove convergence to the global solution. Notably, ASBS generalizes the recent Adjoint Sampling (Havens  et al., 2025)  to arbitrary source distributions by relaxing the so-called memoryless condition that largely restricts the design space. Through extensive experiments, we demonstrate the effectiveness of ASBS on sampling from classical energy functions, amortized conformer generation, and molecular Boltzmann distributions. Codes are available at https://github.com/facebookresearch/adjoint_samplers.

Section: Introduction
Sampling from Boltzmann distributions is a fundamental problem in computational science, with widespread applications in Bayesian inference, statistical physics, and chemistry (Box and Tiao, 2011;Binder et al., 1992;Tuckerman, 2023). Mathematically, we aim to sample from a target distribution ν(x) known up to a unnormalized, often differentiable, energy function E(x) : X ⊆ R d → R, ν(x) := e -E(x) Z , where Z :=
X e -E(x) dx (1
)
is an intractable normalization constant. For instance, the energy function E(x) of a molecular system quantifies the stability of a chemical structure based on the 3D positions of particles. A lower energy indicates a more stable structure and hence a higher likelihood of its occurrence, i.e., ν(x) ∝ e -E( x) .
Classical methods that generate samples from ν(x) rely on Markov Chain Monte Carlo algorithms, which run a Markov chain whose stationary distribution is ν(x) (Metropolis et al., 1953;Neal, 2001;Del Moral et al., 2006). These methods, however, tend to suffer from slow mixing time and require extensive evaluations of energy function, limiting their practical usages due to prohibitive complexity.
To improve sampling efficiency, modern samplers focus on learning better proposal distributions (Noé et al., 2019;Midgley et al., 2023). Among those, recent advances in diffusion-based generative models (Song et al., 2021;Ho et al., 2020) have given rise to a family of Diffusion Samplers, which Table 1: Compared to prior diffusion samplers, Adjoint Schrödinger Bridge Sampler (ASBS) offers the most flexible design for diffusion samplers (2), while learning the drift u θ t via scalable matching objectives that do not rely on computation of importance weights (IWs).
this section cite: ['b9', 'b7', 'b77', 'b49', 'b51', 'b20', 'b55', 'b31']

Section: Design condition for (2)
Learning method for u θ t Method Non-memoryless Arbitrary prior Matching objective 1 No reliance on IWs PIS (Zhang and Chen, 2022) DDS (Vargas et al., 2023) ✗ ✗ ✗ ✓ LV-PIS & LV-DDS (Richter and Berner, 2024) ✗ ✗ ✗ ✗ PDDS (Phillips et al., 2024) iDEM (Akhound-Sadegh et al., 2024) ✗ ✗ ✓ ✗ AS (Havens et al., 2025) ✗ ✗ ✓ ✓ Sequential SB (Bernton et al., 2019) ✓ ✓ ✗ ✗ Adjoint Schrödinger Bridge Sampler (Ours) ✓ ✓ ✓ ✓ consider stochastic differential equations (SDEs) of the following form:
dX t = f t (X t ) + σ t u θ t (X t ) dt + σ t dW t , X 0 ∼ µ(X 0 ),(2)
where f t (x) : [0, 1] × X → X the base drift, σ t : [0, 1] → R >0 the noise schedule, and µ(x) the initial source distribution. Given (f t , σ t , µ), the diffusion sampler learns a parametrized drift u θ t (x) transporting samples to the target distribution ν(x) at the terminal time t = 1.
Computational methods for learning diffusion samplers have grown significantly recently (Zhang and Chen, 2022;Vargas et al., 2023;Berner et al., 2024;Chen et al., 2025). Due to the distinct problem setup in (1), the target distribution is defined exclusively by its energy E(x), rather than by explicit target samples. This characteristic renders modern generative modeling techniques for scalabilityparticularly the score matching objectives 1 -less applicable. As such, prior matching-based diffusion samplers (Phillips et al., 2024;Akhound-Sadegh et al., 2024;De Bortoli et al., 2024) often require computationally intensive estimation of target samples via importance weights (IWs).
Recently, Havens et al. (2025) introduced Adjoint Sampling (AS), a new class of diffusion samplers whose matching objectives rely only on on-policy samples, thereby greatly enhancing scalability. By incorporating stochastic optimal control (SOC) theory (Kappen, 2005;Todorov, 2007), AS facilitates the use of Adjoint Matching (Domingo-Enrich et al., 2025), a novel matching objective that imposes self-consistency in generated samples, effectively eliminating the needs for target samples.
The efficiency of AS, however, is achieved through a specific instantiation of the SDE (2) to satisfy the so-called memoryless condition. This condition-formally discussed in Section 2-restricts its source distribution to be Dirac delta µ(x) := δ, precluding the use of common priors such as Gaussian or domain-specific priors such as the harmonic oscillators in molecular systems (Jing et al., 2023). Notably, the memoryless condition underlies all previous matching-based diffusion samplers, restricting the design space of (2) from other choices known to enhance transportation efficiency (Shaul et al., 2023). While the condition has been relaxed in non-matching-based methods at extensive computational complexity (Richter and Berner, 2024;Bernton et al., 2019), no existing diffusion sampler-to our best understanding-has successfully combined matching objectives with non-memoryless condition. Table 1 summarizes the comparison between prior diffusion samplers.
In this work, we propose Adjoint Schrödinger Bridge Sampler (ASBS), a new adjoint-matchingbased diffusion sampler that eliminates the requirement for memoryless condition entirely. Formally, ASBS recasts learning diffusion sampler as a distributionally constrained optimization, known as the Schrödinger Bridge (SB) problem (Schrödinger, 1931(Schrödinger, , 1932;;Léonard, 2013;Chen et al., 2016):
min u D KL (p u ||p base ) = E X∼p u 1 0 1 2 ∥u θ t (X t )∥ 2 dt , (3a
) s.t. dX t = f t (X t ) + σ t u θ t (X t ) dt + σ t dW t , X 0 ∼ µ(X 0 ), X 1 ∼ ν(X 1 ).(3b)
Here, p u denotes the path distribution induced by the SDE in (3b), whereas p base := p u:=0 denotes the path distribution induced by the "base" SDE when u t := 0. By minimizing their KL divergence, the SB problem (3) seeks the kinetic-optimal drift u ⋆ t -an optimality structure well correlated with sampling efficiency in generative modeling (Finlay et al., 2020;Liu et al., 2023). Since the SOC problem in AS corresponds to a specific case of the SB problem with (f t , µ) := (0, δ), ASBS extends AS to handle non-memoryless conditions by solving more general SB problems (see Theorem 3.1). Computationally, ASBS retains all scalability advantages from AS by utilizing an adjoint-matching objective that removes the need for estimating target samples. It also introduces a corrector-matching objective to correct nontrivial biases arising from non-memoryless conditions. We prove that alternating optimization between the two matching objectives is equivalent to executing the Iterative Proportional Fitting algorithm (Kullback, 1968), ensuring global convergence of ASBS to u ⋆ t (see Theorem 3.2). Though extensive experiments, we show superior performance of ASBS over prior diffusion samplers across various benchmarks on sampling multi-particle energy functions.
In summary, we present the following contributions:
• We introduce ASBS, an SB-based diffusion sampler capable of sampling target distributions using only unnormalized energy functions, by solving general SB problems with arbitrary priors.
• We base ASBS on a new SOC framework that removes the restrictive memoryless condition, develop a scalable matching-based algorithm, and prove theoretical convergence to global solution.
• We show ASBS's superior performance over prior methods on sampling Boltzmann distributions of classical energy functions, alanine dipeptide molecule and amortized conformer generation.
this section cite: ['b79', 'b64', 'b6', 'b79', 'b64', 'b19', 'b34', 'b76', 'b33', 'b64', 'b6', 'b68', 'b69', 'b43', 'b15', 'b24', 'b39']

Section: Preliminary
We revisit the memoryless condition introduced by Domingo-Enrich et al. (2025) and examine its impact on the constructions of SOC-based diffusion samplers (Zhang and Chen, 2022;Havens et al., 2025), which are closely related to our ASBS. Additional review can be found in Appendix A.
this section cite: []

Section: Stochastic Optimal Control (SOC)
The SOC problem (4) studies an optimization problem:
min u E X∼p u 1 0 1 2 ∥u t (X t )∥ 2 dt + g(X 1 ) s.t. (2),(4)
which, unlike the SB problem (3), includes an additional terminal cost g(x) : X → R at the terminal time t = 1 and considers the SDE without the terminal constraint X 1 ∼ ν. The primary reason for studying this specific optimization problem is that the optimal distribution is known analytically byfoot_2
p ⋆ (X 0 , X 1 ) = p base (X 0 , X 1 )e -g(X1)+V0(X0) , where V 0 (x) = -log p base 1|0 (y|x)e -g(y) dy (5
)
is the initial value function. That is, the optimal distribution p ⋆ is an exponentially tilted version of the base distribution, p base := p u:=0 . Specifically, p base is tilted by the terminal cost "-g(X 1 )" and the initial value function V 0 (X 0 ), which is intractable. Consequently, to ensure its marginal p ⋆ (X 1 ) follows the target distribution ν(X 1 ), we must eliminate the initial value function bias from V 0 (X 0 ).
this section cite: []

Section: Memoryless condition & SOC-based diffusion sampler
A common approach to eliminate the aforementioned initial value function bias, adopted by most diffusion samplers, is to restrict the class of base processes to be memoryless. Formally, the memoryless condition assumes statistical independency between X 0 and X 1 in the base distribution:
p base (X 0 , X 1 ) memoryless := p base (X 0 )p base (X 1 ).(6)
This memoryless condition (6) simplifies the optimal distribution at the terminal time t = 1 and, upon choosing a proper terminal cost g(x), recovers the target distribution ν,
p ⋆ (X 1 ) memoryless = p base (X 0 )p base (X 1 )e -g(X1)+V0(X0) dX 0 ∝ p base (X 1 )e -g(X1) = ν(X 1 ),
where the last equality is due to setting the terminal cost to g(x) := log p base 1 (x) ν(x) . Typically, the memoryless condition (6) is enforced by a careful design of the base distribution p base or, equivalently, the parameters (f t , σ t , µ) in (2). For instance, the variance-preserving process (VP; Song et al., 2021) considers a linear base drift f t , a noise schedule σ t that grows significantly with time, and a Gaussian prior µ; see Figure 1. Alternatively, one could implement (6) with Dirac delta prior µ(x) := δ 0 (x) and f t := 0, leading to the following SOC problem (Zhang and Chen, 2022):
min u E X∼p u 1 0 1 2 ∥u t (X t )∥ 2 dt + log p base 1 (X 1 ) ν(X 1 ) s.t. dX t = σ t u t (X t )dt + σ t dW t , X 0 =0. (7)
Based on the aforementioned reasoning, solving (7) results in a diffusion sampler that transports samples to the target distribution at t=1, with Adjoint Sampling (Havens et al., 2025) as the only scalable method of this class. Despite encouraging, the SOC problem in ( 7) is nevertheless limited by its trivial source, precluding potentially more effective options for sampling Boltzmann distributions.
this section cite: []

Section: Adjoint Schrödinger Bridge Sampler
We introduce a new diffusion sampler by solving the SB problem (3), where the target distribution ν(x) is given by its energy function E(x) rather than explicit samples. All proofs are left in Appendix B.
this section cite: []

Section: SOC Characteristics of the SB Problem
The SB problem (3)-as an optimization problem with distribution constraints-is widely explored in optimal transport, stochastic control, and recently machine learning (Léonard, 2012;Chen et al., 2021;De Bortoli et al., 2021). Its kinetic-optimal drift u ⋆ satisfies the following optimality equations:
u ⋆ t (x) = σ t ∇ log φ t (x), where      φ t (x) = p base 1|t (y|x)φ 1 (y)dy, φ 0 (x) φ0 (x) = µ(x) φt (x) = p base t|0 (x|y) φ0 (y)dy, φ 1 (x) φ1 (x) = ν(x)(8a) (8b)
and p base t|s (y|x) := p base (X t =y|X s =x) is the transition kernel of the base process for observing y at time t given x at time s. The SB potentials φ t (x), φt (x) ∈ C 1,2 ([0, 1], R d ) are then defined (up to some multiplicative constant) as solutions to forward and backward time integrations w.r.t. p base t|s . Equation ( 8) are computationally challenging to solve-even when p base t|s has an analytical solutiondue to the intractable integration and coupled boundaries at t = 0 and 1. Our key observation is that the first equation (8a) resembles the optimality condition of the SOC problem (4) (see Appendix A.1). This implies that the optimality conditions of SB hints an SOC reinterpretation, which, as we will demonstrate, is more tractable than solving (8) directly. We formalize our finding below.
Theorem 3.1 (SOC characteristics of SB). The kinetic-optimal drift u ⋆ t in (8) solves an SOC problem
min u E X∼p u 1 0 1 2 ∥u t (X t )∥ 2 dt + log φ1 (X 1 ) ν(X 1 ) s.t. (2).(9)
Theorem 3.1 suggests that every SB problem (3) can be solved like an SOC problem (4) with the terminal cost g(x) := log φ1 (x) ν(x) . Comparing to the formulation in Adjoint Sampling (Havens et al., 2025), the two SOC problems, namely (7) and ( 9), differ in their terminal costs-where p base 1 is replaced by φ1 -and the relaxation of the source distribution from Dirac delta X 0 = 0 to general source µ(X 0 ).
How φ1 (•) debiases non-memoryless SOC problems Taking a closer look at the effect of φ1 , notice that the optimal distribution of the SB problem-according to Theorem 3.1 and (5)-follows
p ⋆ (X 0 , X 1 ) = p base (X 0 , X 1 ) exp -log φ1(X1) ν(X1) -log φ 0 (X 0 ) ,(10)
where "-log φ 0 " is the equivalent initial value function. One can verify that the marginal at the terminal time t = 1 indeed satisfies the target distribution,
p ⋆ (X 1 ) = p ⋆ (X 0 , X 1 )dX 0 (10
) = ν(X1) φ1(X1) p base (X 0 , X 1 ) 1 φ0(X0) dX 0 (8a) = ν(X1) φ1(X1) p base (X 1 |X 0 ) φ0 (X 0 )dX 0 (8b) = ν(X 1 ).(11)
That is, the optimality equations in (8), in their essence, construct a specific function φ1 (•) that eliminates the initial value function bias associated with any non-memoryless processes, thereby ensuring that the optimal distribution satisfies the target ν at t = 1.
this section cite: ['b42', 'b16']

Section: Adjoint Sampling with General Source Distribution
We now specialize Theorem 3.1 to sampling Boltzmann distributions (1), where ν(x) ∝ e -E(x) , and hence the terminal cost of the new SOC problem in (9) becomes log φ1 (x) ν(x) = E(x) + log φ1 (x). To encourage minimal transportation cost (Chen and Georgiou, 2015;Peyré and Cuturi, 2017), we consider the Brownian-motion base process with a degenerate base drift f t := 0. Applying Adjoint Matching (AM; Domingo-Enrich et al., 2025) to the resulting SOC problem leads to
u ⋆ = arg min u E p base t|0,1 p ū 0,1 ∥u t (X t ) + σ t (∇E + ∇ log φ1 ) (X 1 )∥ 2 , ū = stopgrad(u). (12)
Note that the AM objective in (12) functions as a self-consistency loss-in that both the regression and its expectation depend on the optimization variable u. This makes (12) particularly suitable for learning SB-based diffusion samplers, unlike previous matching-based SB methods (Shi et al., 2023;Liu et al., 2024), which all require ground-truth target samples from X 1 ∼ ν.
Computing the AM objective in (12) requires knowing ∇ log φ1 (x), which, as we discussed in (11), serves as a corrector that debiases the optimization toward the desired target. Notably, this corrector function ∇ log φ1 (x) also admits a variational form (Peluchetti, 2022(Peluchetti, , 2023;;Shi et al., 2023
): 3 ∇ log φ1 = arg min h E p u ⋆ 0,1 ∥h(X 1 ) -∇ x1 log p base (X 1 |X 0 )∥ 2 .(13)
To summarize, Equations ( 12) and (13) characterize two distinct matching objectives that any kineticoptimal drift u ⋆ t of SBs must satisfy. When the source distribution degenerates to Dirac delta X 0 := 0, (13) is minimized at ∇ log p base 1 , and (12) simply recovers the objective used in Adjoint Sampling (Havens et al., 2025). In other words, ( 12) and (13) should be understood as a generalization of Adjoint Sampling to handle arbitrary-including non-memoryless-source distributions.
this section cite: ['b14', 'b60', 'b58', 'b59']

Section: Alternating Optimization with Adjoint and Corrector Matching
Building upon the theoretical characterization in Section 3.2, we aim to design a learning algorithm that finds a diffusion sampler satisfying (12) and (13), which correspond to two simple matchingbased objectives. However, these matching objectives cannot be naively implemented due to their interdependency: Solving (12) for the kinetic-optimal drift u ⋆ requires knowing ∇ log φ1 . Likewise, solving (13) for the corrector function ∇ log φ1 requires samples from u ⋆ . We relax the interdependency with an alternating optimization scheme. Specifically, given an approximation of ∇ log φ1 ≈ h (k-1) from the previous stage k -1, we first update the drift u (k) with the AM objective:  14) and the Corrector Matching (CM) objective (15), ASBS progressively learns a better corrector h (k)   ϕ that debiases the SOC problem for the control u (k)  θ . Note that since the corrector is initialized with h (0)  ϕ := 0, the first AM stage simply regresses u (1) θ to the energy gradient ∇E.
u (k) := arg min u E p base t|0,1 p ū 0,1 ∥u t (X t ) + σ t (∇E + h (k-1) )(X 1 )∥ 2 , ū = stopgrad(u). (14
)
Then, we use the resulting drift u (k) to update h (k) by minimizing the following matching objective, which-in light of the corrector role of ∇ log φ1 -we refer to as the Corrector Matching objective:
h (k) := arg min h E p u (k) 0,1 ∥h(X 1 ) -∇ x1 log p base (X 1 |X 0 )∥ 2 .(15)
Equation ( 15) should be distinguish from the bridge-matching objectives in data-driven SB methods (Shi et al., 2023;Somnath et al., 2023), where X 1 must be drawn from the target distribution ν. In contrast, the matching objectives in ( 14) and ( 15) depend only on model samples at the current stage
X 1 ∼ p u (k) θ (X 1 |X 0 )
, hence can be used to learn SB-based diffusion samplers at scale.
The alternating optimization between ( 14) and ( 15) creates a sequence of updates, (u
(0) , h (0) ) → • • • (u (k) , h (k) ) → • • •
, that may be thought of as running coordinate descent between the control u and the corrector h. Intuitively, at each stage k, we first find the control u (k) that best aligns with the corrector from previous stage, h (k-1) , then update the corrector h (k) accordingly to reflect the "memorylessness" of the current control u (k) . We summarize our method, Adjoint Schrödinger Bridge Sampler (ASBS), in Algorithm 1, while leaving the full details with additional components, such as replay buffers, in Appendix C. Finally, we prove that this alternating optimization indeed converges to the kinetic-optimal drift u ⋆ in (8).
Theorem 3.2 (Global convergence of ASBS). Algorithm 1 converges to the Schrödinger bridge solution of (3), provided all matching stages achieve their critical points, i.e.,
lim k→∞ u (k) = u ⋆ .
this section cite: []

Section: Theoretical Analysis
We provide the proof of Theorem 3.2 and highlight theoretical insights throughout. While ASBS is specialized to a degenerate base drift f t := 0, all theoretical results here apply to general f t . To simplify notation, we omit the parameters θ, ϕ and reparametrize the corrector by h (k) = ∇ log h(k) .
All proofs are left in Appendix B.
Our first result presents a variational characteristic to the solution of the AM objective in (14).
Theorem 4.1 (Adjoint Matching solves a forward half bridge). Let p u (k) be the path distribution induced by the drift u (k) in (14) at stage k. Then, p u (k) solves the following variational problem:
p u (k) = arg min p D KL (p||q h(k-1) ) : p 0 = µ ,(16)
where q h(k-1) is the path distribution induced by a "backward" SDE on the reversed time coordinate s := 1 -t, defined by the corrector from the previous stage h(k-1) :
dY s = -f s (Y s ) + σ 2 s ∇ log ϕ s (Y s ) ds + σ s dW s , ϕ s (y) = p base 1-s|0 (y|z)ϕ 1 (z)dz,(17)
with the boundary conditions Y 0 ∼ ν and ϕ 0 (y) = h(k-1) (y).
Theorem 4.1 suggests that any SOC problems with the terminal cost g(x) := log h(k) (x) ν(x) can be reinterpreted as KL minimization w.r.t. a specific backward SDE (17) that is fully characterized by ν-which serves as its source distribution-and h(k) -which defines its drift through the function ϕ s (y). The objective in ( 16) differs from the one in the original SB problem (3) by disregarding the target boundary constraint, X 1 ∼ ν. Consequently, ( 16) only solves a forward half bridge.
Next, we show that the CM objective (15) admits a similar variational form, except backward in time. Theorem 4.2 (Corrector Matching solves a backward half bridge). Let h(k) be the corrector in (15) at stage k. Then, the path distribution q h(k) solves the following variational problem:
q h(k) = arg min q D KL (p u (k) ||q) : q 1 = ν (18
)
Unlike ( 16), the objective in (18) disregards the source boundary constraint µ instead, thereby solving a backward half bridge. Theorems 4.1 and 4.2 imply that our ASBS in Algorithm 1 implicitly employs an optimization scheme that alternates between solving forward and backward half bridges, thereby instantiating the celebrated Iterative Proportional Fitting algorithm (IPF; Fortet, 1940;Kullback, 1968). Combining with the analysis by (De Bortoli et al., 2021) leads to our final result in Theorem 3.2.
this section cite: ['b25', 'b39']

Section: Related Works
We provide additional clarification on SB-related works and leave the full review to Appendix A.3.
this section cite: []

Section: Data-driven Schrödinger Bridges
The SB problem has attracted notable interests in machine learning due to its connection to diffusion-based generative models (Wang et al., 2021). Earlier methods implemented classical IPF algorithms (De Bortoli et al., 2021;Vargas et al., 2021;Chen et al., 2022), with scalability later enhanced by bridge matching-based methods (Shi et al., 2023;Liu et al., 2024). Unlike ASBS, all of them focus on generative modeling and assume access to extensive target samples during training, making them unsuitable for sampling from Boltzmann distributions.
this section cite: ['b78']

Section: SB-inspired Diffusion Samplers
Notably, in the context of diffusion samplers, the SB formulation has been constantly emphasized as a mathematically appealing framework for both theoretical analysis and method motivation (Zhang and Chen, 2022;Vargas et al., 2024;Richter and Berner, 2024;Havens et al., 2025). None of the prior methods, however, offers general solutions to learning SB-based diffusion samplers, instead specializing to either the memoryless condition or non-matching-based objectives, which largely complicate the learning process (see Table 1). Conceptually, our ASBS stands closest to SSB (Bernton et al., 2019) by learning general SB samplers. However, the two methods differ fundamentally in scalability: SSB is a Sequential Monte Carlo-based method (Chopin, 2002) augmented with learned transition kernels using Gaussian-approximated SB potentials. As with many MCMC-augmented samplers (Gabrié et al., 2022;Matthews et al., 2022), SSB requires extensive evaluations on the energy E(x), in contrast to ASBS, which is much more energy-efficient.
Table 2: Results on the synthetic energy functions for n-particle bodies with their corresponding dimensions d. Following (Chen et al., 2025;Havens et al., 2025), we report Sinkhorn for MW-5 and the Wasserstein-2 distances w.r.t samples, W 2 , and energies, E(•)W 2 , for the rest. All values are averaged over three random trials. Best results are highlighted.
this section cite: ['b8', 'b64', 'b6', 'b17', 'b27']

Section: MW-5 (d=5) DW-4 (d = 8)
LJ-13 (d = 39) LJ-55 (d = 165)
Method Sinkhorn ↓ W 2 ↓ E(•) W 2 ↓ W 2 ↓ E(•) W 2 ↓ W 2 ↓ E(•) W 2 ↓
PDDS (Phillips et al., 2024) -0.92±0.08 0.58±0.25 4.66±0.87 56.01±10.80 --SCLD (Chen et al., 2025) 0.44±0.06 1.30±0.64 0.40±0.19 2.93±0.19 27.98± 1.26 --PIS (Zhang and Chen, 2022) 0.65±0.25 0.68±0.28 0.65±0.25 1.93±0.07 18.02± 1.12 4.79±0.45 228.70±131.27 DDS (Vargas et al., 2023) 0.63±0.24 0.92±0.11 0.90±0.37 1.99±0.13 24.61± 8.99 4.60±0.09 173.09± 18.01 LV-PIS ( Richter and Berner, 2024) -1.04±0.29 1.89±0.89 ----iDEM (Akhound-Sadegh et al., 2024) -0.70±0.06 0.55±0.14 1.61±0.01 30.78±24.46 4.69±1.52 93.53± 16.31 AS (Havens et al., 2025) 0.32±0.06 0.62±0.06 0.55±0.12 1.67±0.01 2.40± 1.25 4.04±0.05 30.83± 8.19 ASBS (Ours) 0.15±0.02 0.43±0.05 0.20±0.11 1.59±0.03 1.99± 1.01 4.00±0.03 28.10± 8.15 -26 -22 -18 -14 Energy E(x) .0 .1 .2 .3 Normalized Density DW-4 ASBS (ours) Ground Truth -60 -45 -30 -15 Energy E(x) .00 .02 .04 .06 LJ-13 10 3 10 2 10 1 10 0 10 1 10 2 10 3 Average NFE on Energy 10 0 10 1 10 2 10 3 Average NFE on Model ASBS AS PIS, DDS iDEM Complexity per Grad. Update Figure 4: Complexity w.r.t. the number of function evaluation (NFE) on LJ-13 potential.
6 Experiments Benchmarks We evaluate our ASBS on three classes of multi-particle energy functions E(x).
• Synthetic energy functions These are classical potentials based on pair-wise distances of an n-particle system, where E(x) is known analytically. Following (Akhound-Sadegh et al., 2024;Chen et al., 2025), we consider a 2D 4-particle Double-Well potential (DW-4), a 1D 5-particle Many-Well potential (MW-5), a 3D 13-particle Lennard-Jones potential (LJ-13) and a 3D 55particle Lennard-Jones potential (LJ-55). For the ground-truth samples, we sample analytically from MW-5 and use the MCMC samples from (Klein et al., 2023) for the rest of three potentials.
• Alanine dipeptide This is a molecule consisting of 22 atoms in 3D. Specifically, we consider the alanine dipeptide in an implicit solvent and aim to sample from its Boltzmann distribution at a temperature 300K. Following prior methods (Zhang and Chen, 2022;Wu et al., 2020), we use the energy function E(x) from the OpenMM library (Eastman et al., 2017) and consider a more structural internal coordinate with the dimension d = 60. The ground-truth samples contain 10 7 configurations, simulated from Molecular Dynamics (Midgley et al., 2023).
• Amortized conformer generation Finally, we consider a new benchmark proposed in (Havens et al., 2025) for large-scale conformer generation. Conformers are locally stable configurations located at the local minima of the molecule's potential energy surface (Hawkins, 2017). Sampling conformers is essentially a conditional generation task, targeting a Boltzmann distribution ν(x|g) ∝ e -1 τ E(x|g) at a low temperature τ ≪ 1, conditioned on the molecular topology g ∈ G. The training set G train contains 24,477 molecular topologies from SPICE (Eastman et al., 2023), represented by the SMILES strings (Weininger, 1988), whereas the test set G test contains 80 topologies from SPICE and another 80 from GEOM-DRUGS (Axelrod and Gomez-Bombarelli, 2022). As with (Havens et al., 2025), we consider E(x|g) a foundation model eSEN from (Fu et al., 2025), which predicts energy with density-functional-theory accuracy at a much lower computational cost. We use CREST conformers (Pracht et al., 2024) as the ground-truth samples.
this section cite: ['b79', 'b83', 'b22', 'b30', 'b23', 'b82', 'b3']

Section: Baselines and evaluation
We compare ASBS with a wide range of diffusion samplers, including PIS (Zhang and Chen, 2022), DDS (Vargas et al., 2023), PDDS (Phillips et al., 2024), SCLD (Chen Table 3: Comparison between diffusion samplers on sampling the molecular Boltzmann distribution of the alanine dipeptide. We report the KL divergence D KL for the 1D marginal across five torsion angles and the Wasserstein-2 W 2 on jointly (ϕ, ψ), known as Ramachandran plots (see Figure 5). Best results are highlighted.
D KL on each torsion's marginal ↓ W2 on joint ↓ Method ϕ ψ γ 1 γ 2 γ 3 (ϕ, ψ)
PIS ( Zhang and Chen, 2022 , 2025). For the conformer generation task, we include additionally a domain-specific baseline, RDKit ETKDG (Riniker and Landrum, 2015), which relies on chemistrybased heuristics. The evaluation pipelines are consistent with prior methods, where we adopt the SCLD setup for MW-5, the PIS setup for alanine dipeptide, and the AS setup for all the rest; see Appendix D for details.
ASBS models For all tasks, we consider a degenerate base drift f t := 0, as discussed in Section 3.2, and set σ t a geometric noise schedule. For energy functions that directly take particle systems as inputs-such as DW, LJ, and eSEN-we parametrize the models u θ , h ϕ with two Equivariant Graph Neural Networks (Satorras et al., 2021) and consider a domain-specific source distribution-the harmonic prior (Jing et al., 2023). Formally, for an n-particle system x = {x i } n i=0 , the harmonic prior µ harmonic (x) is a quadratic potential that can be sampled analytically from an anisotropic Gaussian:
µ harmonic (x) ∝ exp(-α 2 i,j ∥x i -x j ∥ 2 ).(19)
For other energy functions, we use standard fully-connected neural networks and consider Gaussian priors. All models are trained with Adam (Kingma and Ba, 2015) and, following standard practices (Havens et al., 2025;Akhound-Sadegh et al., 2024), utilize replay buffers; see Appendix C for details.
Results Table 2 presents the results on synthetic energy functions. Notably, ASBS consistently outperforms prior diffusion samplers across all energy functions. In Figure 3, we compare the energy histograms of DW-4 and LJ-13 potentials between the ground-truth MCMC samples and those from ASBS. It is evident that ASBS generates samples that closely resemble the target Boltzmann distribution ν(x) ∝ e -E(x) , resulting in energy profiles E(x) that are almost indistinguishable from the ground truth. Computationally, Figure 4 shows the average number of evaluation required on the energy E(x) and the model u θ (t, x) for each gradient update. ASBS is much more efficient than most diffusion samplers, with a slight overhead compared to AS due to the additional network h ϕ (x).
Table 3 summarizes the results for alanine dipeptide. Following standard pipeline (Zhang and Chen, 2022), we generate model samples X 1 ∈ R 60 and extract five torsion angles-including the backbone  0 0.5 1.0 1.5 2.0 Threshold (Ångström) 0 20 40 60 80 100 Coverage Recall (%) SPICE RDKit AS ASBS gauss ASBS harmonic 0 0.5 1.0 1.5 2.0 Threshold (Ångström) 0 20 40 60 80 100 SPICE + relax 0 0.5 1.0 1.5 2.0 Threshold (Ångström) 0 20 40 60 80 100 GEOM-DRUG 0 0.5 1.0 1.5 2.0 Threshold (Ångström) 0 20 40 60 80 100 GEOM-DRUG + relax Figure 7: Recall coverage curves on amortized conformer generation on the SPICE and GEOM-DRUGS test sets without RDKit warm-start. Note that Table 4 reports the recall coverages at the threshold 1.0Å.
angles ϕ, ψ and methyl rotation angles γ 1 , γ 2 , γ 3 -all of them exhibit multi-modal distributions. Notably, ASBS achieves lowest KL divergence to the ground-truth marginals across all five torsions. Figure 5 further compares the joint distributions of (ϕ, ψ), known as the Ramachandran plots (Spencer et al., 2019), between ground-truth and ASBS. While ASBS identifies all high-density modes in the region ϕ ∈ [-π, 0], it misses few low-density modes. This mode-seeking behavior, inherit in all SOC-based diffusion samplers, could be improved with important weighting. We provide further discussions in Appendix D.4.
Table 4 presents the recall for amortized conformer generation compared to ground-truth samples. For prior diffusion samplers, we primarily compare to AS (Havens et al., 2025) due to the benchmark's scale. Following AS, we ablate a warm-start stage using RDKit conformers, which are close but not identical to ground-truth samples, and include results with relaxation for post-generation optimization. Since AS is a specific instance of ASBS with a Dirac delta prior-as discussed in Section 3.2-any performance improvements from AS to ASBS highlight the added capability to handle arbitrary priors and, consequently, non-memoryless processes. Remarkably, without any warm-start, ASBS with the harmonic prior (19) already matches and, in many cases, surpasses the RDKit-warm-up AS. With warm-start, ASBS achieves best performance across most metrics. This highlights the significance of domain-specific priors, aiding exploration as effectively as warm-start with additional data, which may not always be available. Finally, we visualize the generation process of ASBS with harmonic prior (19) in Figure 6 and report the recall curves in Figure 7. In practice, we observe that ASBS achieves slightly better results with a harmonic prior compared to a Gaussian prior, with both significantly outperforming AS (Havens et al., 2025). See Appendix D.4 for further ablation studies.
this section cite: ['b79', 'b65', 'b33', 'b74']

Section: Conclusion and Limitation
We introduced Adjoint Schrödinger Bridge Sampler (ASBS), a new diffusion sampler for Boltzmann distributions that solves general SB problems given only target energy functions. ASBS is based on a scalable matching framework, converges theoretically to the global solution, and performs superiorly across various benchmarks. Despite these encouraging results, further enhancement with importance sampling techniques is worth investigating to mitigate the mode collapse inherent in SOC-inspired diffusion samplers. Exploring its effectiveness in sampling amortized Boltzmann distributions would also be valuable.
this section cite: []

Section: References
Ref_id:b0 Title: Iterated denoising energy matching for sampling from boltzmann densities Year: ()
Ref_id:b1 Title: Nets: A non-equilibrium transport sampler Year: (2024)
Ref_id:b2 Title: Annealed flow transport monte carlo Year: ()
Ref_id:b3 Title: GEOM, energy-annotated molecular conformations for property prediction and molecular generation Year: (2022)
Ref_id:b4 Title: The theory of dynamic programming Year: (1954)
Ref_id:b5 Title: An optimal control perspective on diffusion-based generative modeling Year: ()
Ref_id:b6 Title: Schrödinger bridge samplers Year: (2019)
Ref_id:b7 Title: Monte Carlo simulation in statistical physics Year: (1992)
Ref_id:b8 Title: Beyond elbos: a large-scale evaluation of variational methods for sampling Year: (2024)
Ref_id:b9 Title: Bayesian inference in statistical analysis Year: (2011)
Ref_id:b10 Title: JAX: composable transformations of Python+NumPy programs Year: (2018)
Ref_id:b11 Title: Sequential controlled langevin diffusions Year: ()
Ref_id:b12 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b13 Title: Likelihood training of Schrödinger bridge using forward-backward SDEs theory Year: ()
Ref_id:b14 Title: Stochastic bridges of linear systems Year: (2015)
Ref_id:b15 Title: On the relation between optimal transport and schrödinger bridges: A stochastic control viewpoint Year: (2016)
Ref_id:b16 Title: Stochastic control liaisons: Richard sinkhorn meets gaspard monge on a schrödinger bridge Year: (2021)
Ref_id:b17 Title: A sequential particle filter method for static models Year: (2002)
Ref_id:b18 Title: Diffusion Schrödinger bridge with applications to score-based generative modeling Year: ()
Ref_id:b19 Title: Target score matching Year: (2024)
Ref_id:b20 Title: Sequential monte carlo samplers Year: (2006)
Ref_id:b21 Title: Adjoint Matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control Year: ()
Ref_id:b22 Title: OpenMM 7: Rapid development of high performance algorithms for molecular dynamics Year: (2017)
Ref_id:b23 Title: SPICE, a dataset of drug-like molecules and peptides for training machine learning potentials Year: (2023)
Ref_id:b24 Title: How to train your neural ODE: The world of jacobian and kinetic regularization Year: (2020)
Ref_id:b25 Title: Résolution d'un système d'équations de M. Schrödinger Year: (1940)
Ref_id:b26 Title: Learning smooth and expressive interatomic potentials for physical property prediction Year: ()
Ref_id:b27 Title: Adaptive monte carlo augmented with normalizing flows Year: (2022)
Ref_id:b28 Title: Monte carlo sampling methods using markov chains and their applications Year: (1970)
Ref_id:b29 Title: Adjoint Sampling: Highly scalable diffusion samplers via Adjoint Matching Year: ()
Ref_id:b30 Title: Conformation generation: The state of the art Year: (2017)
Ref_id:b31 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b32 Title: On stochastic differential equations Year: (1951)
Ref_id:b33 Title: EigenFold: Generative protein structure prediction with diffusion models Year: (2023)
Ref_id:b34 Title: Path integrals and symmetry breaking for optimal control theory Year: (2005)
Ref_id:b35 Title: Elucidating the design space of diffusionbased generative models Year: ()
Ref_id:b36 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b37 Title: Timewarp: Transferable acceleration of molecular dynamics by learning time-coarsened dynamics Year: ()
Ref_id:b38 Title: Equivariant flows: exact likelihood generative learning for symmetric densities Year: (2020)
Ref_id:b39 Title: Probability densities with given marginals Year: (1968)
Ref_id:b40 Title: Rdkit: Open-source cheminformatics Year: (2006)
Ref_id:b41 Title: Brownian motion, martingales, and stochastic calculus Year: (2016)
Ref_id:b42 Title: From the schrödinger problem to the monge-kantorovich problem Year: (2012)
Ref_id:b43 Title: A survey of the Schrödinger problem and some of its connections with optimal transport. Discrete and Continuous Dynamical Systems Year: (2013)
Ref_id:b44 Title: Reciprocal processes. A measuretheoretical point of view Year: (2014)
Ref_id:b45 Title: The open molecules 2025 (omol25) dataset, evaluations, and models Year: (2025)
Ref_id:b46 Title: I 2 SB: Image-to-Image Schrödinger bridge Year: ()
Ref_id:b47 Title: Generalized Schrödinger bridge matching Year: ()
Ref_id:b48 Title: Continual repeated annealed flow transport monte carlo Year: ()
Ref_id:b49 Title: Equation of state calculations by fast computing machines Year: (1953)
Ref_id:b50 Title: Flow annealed importance sampling bootstrap Year: ()
Ref_id:b51 Title: Annealed importance sampling Year: (2001)
Ref_id:b52 Title: The orca program system Year: (2012)
Ref_id:b53 Title: Action matching: A variational method for learning stochastic dynamics from samples Year: ()
Ref_id:b54 Title: Dynamical theories of Brownian motion Year: (2020)
Ref_id:b55 Title: Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning Year: (2019)
Ref_id:b56 Title: Stochastic differential equations Year: (2003)
Ref_id:b57 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b58 Title: Non-Denoising forward-time diffusions Year: (2022)
Ref_id:b59 Title: Diffusion bridge mixture transports, Schrödinger bridge problems and generative modeling Year: (2023)
Ref_id:b60 Title: Computational optimal transport. Center for Research in Economics and Statistics Working Papers Year: (2017)
Ref_id:b61 Title: Computational optimal transport: With applications to data science Year: (2019)
Ref_id:b62 Title: Particle denoising diffusion sampler Year: ()
Ref_id:b63 Title: CREST-A program for the exploration of low-energy molecular chemical space Year: ()
Ref_id:b64 Title: Improved sampling via learned diffusions Year: (2024)
Ref_id:b65 Title: Better informed distance geometry: using what we know to improve conformation generation Year: (2015)
Ref_id:b66 Title: Applied stochastic differential equations Year: (2019)
Ref_id:b67 Title: E(n) equivariant graph neural networks Year: ()
Ref_id:b68 Title: Über die Umkehrung der Naturgesetze Year: (1931)
Ref_id:b69 Title: Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique Year: (1932)
Ref_id:b70 Title: On kinetic optimal probability paths for generative models Year: ()
Ref_id:b71 Title: Diffusion Schrödinger bridge matching Year: ()
Ref_id:b72 Title: Aligned diffusion Schrödinger bridges Year: ()
Ref_id:b73 Title: Score-based generative modeling through stochastic differential equations Year: ()
Ref_id:b74 Title: Stereochemistry of polypeptoid chain configurations Year: (2019)
Ref_id:b75 Title: Resampling base distributions of normalizing flows Year: ()
Ref_id:b76 Title: Linearly-solvable Markov decision problems Year: (2007)
Ref_id:b77 Title: Statistical mechanics: theory and molecular simulation Year: (2023)
Ref_id:b78 Title: Solving Schrödinger bridges via maximum likelihood Year: (2021)
Ref_id:b79 Title: Denoising diffusion samplers Year: (2023)
Ref_id:b80 Title: Transport meets variational inference: Controlled monte carlo diffusions Year: ()
Ref_id:b81 Title: Deep generative learning via Schrödinger bridge Year: ()
Ref_id:b82 Title: Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules Year: (1988)
Ref_id:b83 Title: Stochastic normalizing flows Year: (2020)
Ref_id:b84 Title: Path integral sampler: A stochastic control approach for sampling Year: ()
