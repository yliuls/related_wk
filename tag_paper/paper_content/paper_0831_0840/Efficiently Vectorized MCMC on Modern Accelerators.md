Title: Efficiently Vectorized MCMC on Modern Accelerators
Abstract: With the advent of automatic vectorization tools (e.g., JAX's vmap), writing multi-chain MCMC algorithms is often now as simple as invoking those tools on single-chain code. Whilst convenient, for various MCMC algorithms this results in a synchronization problem-loosely speaking, at each iteration all chains running in parallel must wait until the last chain has finished drawing its sample. In this work, we show how to design single-chain MCMC algorithms in a way that avoids synchronization overheads when vectorizing with tools like vmap, by using the framework of finite state machines (FSMs). Using a simplified model, we derive an exact theoretical form of the obtainable speed-ups using our approach, and use it to make principled recommendations for optimal algorithm design. We implement several popular MCMC algorithms as FSMs, including Elliptical Slice Sampling, HMC-NUTS, and Delayed Rejection, demonstrating speed-ups of up to an order of magnitude in experiments.

Section: Introduction
Automatic vectorization is the act of transforming a function to handle batches of inputs without user intervention. Implementations of automatic vectorization algorithms are now available in many mainstream scientific computing libraries, and have dramatically simplified the task of running multiple instances of a single algorithm concurrently. They are routinely used to train neural networks (Flax, 2023) and in other scientific applications, e.g., Schoenholz and Cubuk (2021); Oktay et al. (2023); Pfau et al. (2020).
This paper focuses on the use of automatic vectorization for Markov chain Monte Carlo (MCMC) algorithms. Tools like JAX's vmap provide a convenient way to run multiple Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). MCMC chains in parallel: one can simply write single-chain MCMC code, and call vmap to turn it into vectorized, multichain code that can run in parallel on the same processor. Many state-of-the-art MCMC libraries have consequently adopted machine learning frameworks with automatic vectorization tools as their backend.
One limitation with automatic vectorization tools is how they handle control flow. Since all instructions must be executed in lock-step, if the algorithm has a while loop, all chains must wait until the last chain has finished its iterations. This can lead to large inefficiencies for MCMC algorithms that generate each sample using variable-length while loops. Roughly speaking, if vectorization executes, say, 100 chains in parallel, all but one finish after at most 10 steps, and the remaining chain runs for 1000 steps, then about 99% of the GPU capacity assigned to vmap is wasted (and our simulations show the effect can indeed be this drastic). For the No-U-Turn Sampler (HMC-NUTS) (Hoffman et al., 2014), this problem is well-documented (BlackJax, 2019;Sountsov et al., 2024;Radul et al., 2020). It also affects various other algorithms, such as variants of slice sampling (Neal, 2003;Murray et al., 2010;Cabezas and Nemeth, 2023), delayed rejection methods (Mira et al., 2001;Modi et al., 2024) and unbiased Gibbs sampling (Qiu et al., 2019).
In this work, we show how to transform MCMC algorithms into equivalent samplers that avoid these synchronization barriers when using vmap-style vectorization. In particular, 1. We develop a novel approach to transform MCMC algorithms into finite state machines (FSMs), that can avoid synchronization barriers with vmap.
this section cite: ['b9', 'b29', 'b23', 'b24', 'b12', 'b2', 'b30', 'b27', 'b22', 'b20', 'b5', 'b18', 'b19', 'b26']

Section: 2.
We analyze the time complexity of our FSMs against standard MCMC implementations and derive a theoretical bound on the speed-up under a simplified model.
3. We use our analysis to develop principled recommendations for optimal FSM design, which enable us to nearly obtain the theoretical bound in speed-ups for certain MCMC algorithms.
4. We implement popular MCMC algorithms as FSMs, including Elliptical Slice Sampling, HMC-NUTS, and Delayed-Rejection MH -demonstrating speed-ups of up to an order of magnitude in experiments.
this section cite: []

Section: Background and Problem Setup
In this section, we briefly review how MCMC algorithms are vectorized, explain the synchronization problems that can arise, and formalize the problem mathematically.
this section cite: []

Section: MCMC Algorithms and Vectorization
MCMC methods aim to draw samples from a target distribution π (typically on a subset of R d ) which is challenging to sample from directly. To do so, they generate samples {x 1 , ..., x n } ∈ R d×n from a Markov chain with transition kernel P and invariant distribution π, by starting from an initial state x 0 ∈ R d and iteratively sampling x i+1 ∼ P (•|x i ).
For aperiodic and irreducible Markov chains, as n → ∞ the samples will converge in distribution to π (Brooks et al., 2011). In practice, P is implemented by a deterministic function sample, which takes in the current state x i and a pseudo-random state r i ∈ N, and returns new states x i+1 and r i+1 . This procedure is given in Algorithm 1.
Algorithm 1 MCMC algorithm with sample function 1: Inputs: sample x 0 , seed r 0 2: for i ∈ {1, ..., n} do 3: generate x i , r i ← sample(x i-1 , r i-1 ) 4: end for 5: return x 1 , . . . , x n Practitioners commonly run Algorithm 1 on m different initializations, producing m chains of samples. Given an implementation of sample: R d × N → R d × N, one way to do this is to use an automatic vectorization tool like JAX's vmapfoot_2 . vmap takes sample as an input and returns a new program, vmap(sample): R d×m × N m → R d×m × N m , which operates on a batch of inputs collected into tensors xi-1 := (x i-1,1 , . . . , x i-1,m ) ∈ R d×m ri-1 := (r i-1,1 , . . . , r i-1,m ) ∈ N m and returns the corresponding outputs from sample. One can therefore turn Algorithm 1 into a multi-chain algorithm by simply replacing sample with vmap(sample) and replacing (x 0 , r 0 ) by ( x0 , r0 ) (see Algorithm 6 in Appendix B). Under the hood, vmap transforms every instruction in sample (e.g., a dot product) into a corresponding instruction operating on a batch of inputs (e.g., matrix-vector multiplication); that is, it 'vectorizes' sample. These instructions are executed in lock-step across all chains. Using vmap usually yields code that performs as well as manuallybatched code. For this reason, as well as its simplicity and composability with other transformations like grad (for automatic differentiation) and jit (for Just-In-Time compi-lation), vmap has been adopted by major MCMC libraries such as NumPyro and BlackJAX.
this section cite: ['b3']

Section: Synchronization Problems with While Loops
Control flow (i.e., if/else, while, for, etc.) poses a challenge for vectorization, because different batch members may require a different sequence of instructions. vmap solves this by executing all instructions for all batch members, and masking out irrelevant computations. As a consequence, if sample contains a while loop, then vmap(sample) will execute the body of this loop for all chains until all termination conditions are met. Until then, no further instruction can be executed. As a result, if there is high variation in the number of loop iterations across chains, running Algorithm 1 with vmap(sample) introduces a synchronization barrier across all chains: at every iteration, each chain has to wait for the slowest sample call.
This issue arises in practice, because a number of important MCMC algorithms have while loops in their sample implementations: such as variants of slice sampling (Neal, 2003;Murray et al., 2010;Cabezas and Nemeth, 2023), delayed rejection methods (Mira et al., 2001;Modi et al., 2024), the No-U-Turn sampler (Hoffman et al., 2014), and unbiased Gibbs sampling (Qiu et al., 2019).
this section cite: ['b22', 'b20', 'b5', 'b18', 'b19', 'b12', 'b26']

Section: Formalizing The Problem.
Here we formalize the problem through a series of short derivations. These will be made precise in Section 4. Suppose we run a vmap'ed version of Algorithm 1 using a sample function that has a while loop. Let N i,j denote the number of iterations required by the jth chain to obtain its ith sample. If the while loop has a variable length, N i,j is a random number. Due to the synchronization problem described above, the time taken to run vmap(sample) at iteration i is approximately proportional to the largest N i,j out of the m chains, max j≤m N i,j . If we ignore overheads, the total runtime after n samples, C 0 (n), is then roughly proportional to
C 0 (n) ∝ ∼ n i=1 max j≤m N i,j(1)
By contrast, if the chains could be run without any synchronization barriers, the time taken would instead be
C * (n) ∝ ∼ max j≤m n i=1 N i,j(2)
The key difference is that the maximum is now outside the sum. This reflects the fact that when running independently, we only have to wait at the end for the slowest chain to collect its n samples, rather than waiting at every iteration. Clearly C * (n) ≤ C 0 (n). Significant speed-ups are obtainable by de-synchronizing the chains when C * (n) ≪ C 0 (n).
If each N i,j converges in distribution to some P N as we draw more samples, and an appropriate central limit theorem holds (so that the sums in ( 1) and ( 2) behave like scaled means), we can expect for large enough n that:
C 0 (n) = O p (n E max j≤m N ∞,j )(3)
C * (n) = O p (max j≤m n EN ∞,j ) = O p (n EN ∞,1 ) (4)
where the last equality assumes (N ∞,1 , ..., N ∞,m )
iid ∼ P N . De-synchronizing the chains will result in large speedups if
E max j≤m N ∞,j ≫ EN ∞,1(5)
We will see that Equation ( 5) holds in various situations.
Example. Consider the elliptical slice sampling algorithm (Murray et al., 2010) which samples from distributions which admit a density with Gaussian components, p(x) ∝ f (x)N (x|0, Σ). Its transition kernel (see Algorithm 8 in Appendix B) draws each sample by (i) generating a random ellipse of permitted moves and an initial proposal, and (ii) iteratively shrinking the set of permitted moves and resampling the proposal from this set until it exceeds a loglikelihood threshold. The second stage uses a while loop which requires a random number of iterations.
On Figure 1 we display results when implementing this algorithm in JAX to sample from the hyperparameter posterior of a Gaussian process implemented on a regression task using a real dataset from the UCI repository (details are in Section 7). On the LHS we can see the distribution of the number of while loop iterations (i.e. slice shrinks) needed to generate a sample. While the average is ∼6, the average of the maximum across 1024 chains is >18, which implies E max j≤1024 N n,j ≈ 3EN n,1 . On the RHS, we can see that the differences across the chains do balance out as more samples are drawn, which suggests a certain central limit theorem may hold. In particular, after just 100 samples the distribution of the average number of iterations per chain is contained in the interval [5, 8], and after 10,000 samples this shrinks to [6.2, 6.4]. This implies that if we could vectorize the algorithm without incurring synchronization barriers, we could improve the Effective Sample Size per Second (ESS/Sec) by up to 3-fold. We will see that for other algorithms (e.g., HMC-NUTS and delayed rejection) the potential speed-ups are much larger than this.
this section cite: ['b20']

Section: Finite State Machines for MCMC
In this section we present an approach to implementing MCMC algorithms, which avoids the above synchronization problems when vectorizing with vmap. The basic idea is to break down the transition kernel (sample) into a series of smaller 'steps' which avoid iterative control flow like while loops and have minimal variance in execution time. We then define a runtime procedure that allows chains to progress through their own step sequences in de-synchronized fashion. To do this in a principled manner, we adapt the framework of finite state machines (FSMs) (Hopcroft et al., 2001).
An FSM is a 5-tuple (S, Z, δ, B, F), where S is a finite set of states, Z is a finite input set, δ : S ×Z → S is a transition function, B ∈ S is an initial state, and F is the set of final states. We deviate from the usual definition by allowing each state to be a function S : Z → Z that updates inputs dynamicallyfoot_3 . Below we show how to represent different sample functions as an FSM.
this section cite: ['b14']

Section: sample-to-FSM conversion
At a high-level, we construct the FSM of an MCMC algorithm as follows. The states S = (S 1 , ..., S K ) are chosen as functions Z → Z which execute contiguous code blocks of the algorithm. The boundaries of each code block are delineated by the start and end of any while loops. For example, S 1 executes all lines of code before the first while loop, S 2 executes all lines of code from the beginning of the first while loop body to either the start of the next while loop or the end of the current while loop, and so on. The inputs z ∈ Z are all variables used in each code block. (e.g. current sample x, proposal x ′ , log-likelihood log p(x)) 3 , whilst the transition function δ selects the next code block to run according to the relevant loop condition and the received output z ′ after executing the current block (e.g. if a while loop starts after the block S 1 executes, δ will check this loop's condition using the output z ′ = S 1 (z)). Below we make this construction procedure more precise for three kinds of sample functions which cover all MCMC algorithms considered in the present work: (i) functions with a single while loop, (ii) functions with two sequential while loops, and (iii) functions with two nested while loops. An automated construction process for more general programs using a toy language is given in Appendix B. The Single While Loop Case. The simplest case is a sample method with a single while loop. This covers (for example) elliptical slice sampling (Murray et al., 2010) and symmetric delayed rejection Metropolis-Hastings (Mira et al., 2001). In this case, we break sample down into three code blocks B 1 , B 2 , B 3 (one before the while loop, one for the body of the while loop, and one after the while loop) and the termination condition of the loop. This is shown on the LHS of Figure 2. Using these blocks, we define the FSM as (S, Z, δ, B, F), where (1) Z is the set of possible values for the local variables of sample (e.g., the current sample x, seed r, and proposal etc.), (2) S = {S 1 , S 2 , S 3 } is a set of three functions Z → Z, where for each k ∈ {1, 2, 3}, S k (z) runs B k on local variables z and returns their updated value, (3) for each k ∈ {0, 1} and z ∈ Z, the transition function δ(S k , z) checks the while loop termination condition using z, and returns S 2 if False and S 3 if True, (4) B = S 1 and, finally, (5) F = S 3 . The RHS of Figure 2 illustrates the resulting FSM diagram. Note there is an edge S k → S k between states if and only if δ(S k , z) = S k for some z.
Two Sequential While Loops. We break down sample into two blocks: B 1 contains all the code up to the second while loop. B 2 contains the remaining code. In this case, B 1 and B 2 are now single while loop programs, and thus can both be represented by FSMs F 1 and F 2 using the above rule.
The FSM representation of sample can then be obtained by "stitching together" F 1 and F 2 . The full construction process is in Appendix B. The resulting FSM is provided for the case of the Slice Sampler (Figure 3, top-right panel), which contains twofoot_5 while loops: one for expanding the slice, and one for contracting the slice.
Two Nested While Loops. In the case of two nested while loops, we break down sample into B 1 , B 2 , B 3 , where B 1 (resp. B 3 ) is the code before (resp. after) the outer while loop and B 2 is the outer while loop body. As B 2 is a singlewhile loop program, it admits its own FSM F i . Building the final FSM of sample then informally consists in obtaining a first, "coarse" FSM by treating B 2 as opaque, and then refining it by replacing B 2 with its own FSM. The full construction process is given in Appendix B. The resulting FSM is provided for the case of NUTS (Figure 3, bottom-right panel), which-in its iterative form-uses the outer while loop to determine whether to keep expanding a Hamiltonian trajectory, and the inner while loop to determine whether to keep integrating along the current trajectory.
this section cite: ['b20', 'b18']

Section: Defining the FSM Runtime
Going forward, for convenience we assume the transition function δ takes in and returns the label k of each block S k , rather than S k itself. Now, given a constructed FSM, in Algorithm 2 we define a function step, which when executed performs a single transition along an edge in the FSM graph. For reasons made clear shortly, we augment step to return a flag indicating when the final block is run.
Algorithm 2 step function for FSM Inputs: algorithm state k, variables z
1: set isSample ← 1{k = K} 2: z ← switch(k, [{run S 1 (z)}, ..., {run S K (z)}]) 3: k ← switch(k, [{run δ(1, z)}, ..., {run δ(K, z)}]) 4: return (k, z, isSample)
Here switch uses the value of k to determine which branch to run. If we start from some input (k, z) = (0, z 0 ) and call step repeatedly, we transition through a sequence of blocks of sample, until we eventually reach the terminal state. At this point, a sample is obtained (as indicated by isSample=True). We can use this function to draw n samples, by (i) adding a transition from the terminal state F = S K back to the initial state B = S 0 , and (ii) defining a wrapper function which iteratively calls step until isSample=True is obtained n-times (see Algorithm 3).
Algorithm 3 FSM MCMC algorithm 1: input: initial value x 0 , # samples n 2: initialize: z = init(x 0 ), X = list(), B = list() 3: Set t = 0 and k = 0 4: while t < n do 5:
(k, z, isSample) ← step(k, z)
6: append current sample value x stored in z to X 7: append isSample to B 8: update sample counter t ← t + isSample 9: end while 10: return X[B]
1 INIT S 2 PROPOSE S 3 DONE Delayed Rejection MH S 1 S 2 S 3 INIT SHRINK DONE Elliptical Slice Sampling S 1 S 2 S 3 S 4 S 5 INIT E EXPAND INIT S SHRINK DONE Slice Sampling S 1 S 2 S 3 S 4 S 5 INIT DOUBLE INTEGRATE CHECK DONE HMC-NUTS Figure 3.
Finite state machines for the sample function of different MCMC algorithms: The symmetric delayed-rejection Metropolis-Hastings algorithm (Mira et al., 2001), elliptical slice sampling (Murray et al., 2010), (vanilla) slice sampling (with single slice expansion loop) (Neal, 2003), the No-U-Turn sampler for Hamiltonian Monte Carlo (Hoffman et al., 2014).
Both Algorithm 3 and Algorithm 1 call sample n-times and can be easily vectorized with vmap. For Algorithm 3, we just call vmap on step and modify the outer loop to terminate when all chains have collected n samples each (see Algorithm 7 in Appendix B). The crucial difference is: (i) by defining a 'step' to be agnostic to which block each chain is currently on, Algorithm 3 enables chains to simultaneously progress their independent block sequences, and (ii) by moving all while loops to the outer layer, Algorithm 3 only requires chains to synchronize after n samples.
this section cite: ['b18', 'b20', 'b22', 'b12']

Section: Time Complexity Analysis of FSM-MCMC
One limitation with our FSM design is the step function relies on a switch to determine which block to run. When vectorized with vmap or an equivalent transformation, all branches in switch are evaluated for all chains, with irrelevant results discarded. This means the cost of a single call of step is the cost of running all state functions S 1 , ..., S K . To obtain a speedup, the FSM must therefore sufficiently decrease the expected number of steps to obtain n samples from each chain. In this section we quantitatively derive conditions under which this occurs in the simplified setting of an MCMC algorithm with a single while loop. This enables us to subsequently optimize the FSM design in Section 5.
this section cite: []

Section: Long Run Costs of FSM and Non-FSM MCMC
To this end, consider a sample function with a single while loop. Suppose it is broken down into K blocks {S 1 , ...S K }, by our FSM construction procedure, where S k (for some k ∈ [K]) executes the body of the loop. Note our procedure yields K ≤ 3 for one while loop, but we relax this here to analyze the effect of the number of blocks on performance.
A single call of vmap(sample) executes S l once if l ̸ = k, and max j N j -times if l = k, where N j = # loop iterations for chain j. We assume the cost of executing each block S l with vmap is c l (m) for m chains (here the dependence on m reflects underlying GPU vectorization efficiency). The average cost per sample of Algorithm 1 after n samples, when vectorized for m chains, is then
C 0 (m, n) := A 0 (m) + B 0 (m) 1 n n i=1 max j∈[m] N i,j (6
)
where A 0 (m) := j̸ =k c j (m), B 0 (m) = c k (m), and N i,j = # calls of the loop body S k for chain j and sample i. The max reflects the synchronization barrier across chains.
In this case, for the FSM (vmap'ed Algorithm 3), the cost of calling each block S l is K j=1 c j (m), since vmap(step) executes all switch branches for all chains. We will assume this cost can be scaled by some α ∈ [max k∈[K] c k (m)/ K k=1 c k (m), 1], since we will later redesign step to reduce α < 1. This gives an average cost per sample for m chains after n samples of,
C F (m, n) = A F (m) + B F (m) max j∈[m] 1 n n i=1 N i,j (7) where A F (m) = α(K -1)(c ¬k (m) + c k (m)), B F (m) = α(c k (m) + c ¬k (m)), and c ¬k (m) := j̸ =k c j (m).
Let X i,j be the i th random sample obtained by chain j. If the joint sequence (X i,j , N i,j ) i≥1 is an appropriate Markov chain, we have the following concentration result for C 0 , C F , which formalizes our derivations in Section 2.2. Theorem 4.1 (Long Run Costs). Let N i,j ∈ [0, B], X i,j ∈ X ⊆ R d , (X i,j , N i,j ) i≥1 be a Markov Chain with stationary distribution π with absolute spectral gap 1 -λ ∈ [0, 1). Then with probability 1 -δ we have the inequalities:
|C 0 (m, n) -C 0 (m)| ≤ M B 0 (m)n -1 2 ln(2/δ)(8)
|C F (m, n) -C F (m)| ≤ M B F (m)n -1 2 ln(2m/δ) (9
)
Where, for some (N 1 , .., N m )
iid ∼ π N , the long run costs are N j (10)
C F (m) := A F (m) + B F (m)E π N 1 (11
)
The result says that as n → ∞, the cost per sample of the standard MCMC design converges to the expected time for the slowest chain to draw a sample, whilst the FSM design converges to the expected time for a single chain to draw a sample, scaled by the additional cost of control-flow in step (i.e. A F (m) and B F (m)). Both rates are O(n -1 2 ).
this section cite: []

Section: The Relative Long-Run Cost of the FSM
Using the form of the constants in Theorem 4.1, the ratio of the long-run costs, E(m) := C 0 (m)/C F (m), is given by
E(m) = c ¬k (m) + c k (m)E max j∈[m] N j α(c ¬k (m) + c k (m))(K -1 + EN 1 )(12)
In Proposition A.3 in Appendix A we prove that:
E(m) ≤ R(m) := E max j∈[m] N j EN 1(13)
and that this bound is tight. We refer to R(m) as the 'theoretical efficiency bound' for the FSM. Note from Equation ( 12) that minimizing α and K improves the efficiency E(m) of the FSM. In Section 5 we introduce two techniques to minimize α and K in practice, which enable us to nearly obtain the efficiency bound R(m) for certain MCMC algorithms.
this section cite: []

Section: Scale of Potential Speed-ups.
The size of R(m) depends (only) on the underlying distribution of N 1 (since (Vershynin, 2018). Although this implies a slow rate of increase in m, R(m) can be still be very large for small values of m. For example, if N j /B ∼ Bern(p) (i.e., one either needs zero or B iterations to get a sample), then R(m) = (1 -(1 -p) m )/p and converges to 1/p exponentially fast as m increases. For small p this can be very large even for small m. In general R(m) is sensitive to the skewness of the distribution: distributions on [0, B] with zero skewness have R(m) = 2, whilst distributions with skew of 10 can have R(m) ≈ 100; see Figure 4. Intuitively, these are the distributions where chains are slow only occasionally, but at least one chain is slow often. In such cases, our FSM-design can lead to enormous efficiency gains, as we show in experiments.
N i = d N j ∀i, j ∈ [m]). Whenever N 1 is sub-exponential, it is known that R(m) = O(ln(m))
this section cite: ['b33']

Section: Optimal FSM design for MCMC algorithms
In this section we provide two strategies to modify the function step to (effectively) reduce α and K. These strategies enable us to develop MCMC implementations which nearly obtain the theoretical bound R(m) in some experiments.
this section cite: []

Section: Step Bundling to Reduce K
Given an FSM with step defined by code blocks S 1 , ..., S K and transition function δ, one can 'bundle' multiple steps together using a modified step function, bundled_step, which replaces the switch over the algorithm state k in step with a series of separate conditional statements. This is shown in Algorithm 4 for an example with two state functions. Note that under the "run all branches and mask" behaviour of vmap, vmap(step) and vmap(bundled_step) have the same cost. However, whenever bundled_step runs S 1 and δ returns k = 2, it immediately also runs S 2 . This reduces the 'effective' number of states K and/or the overall number of steps needed to recover a sample, increasing efficiency. In principle, the block ordering can be optimized for sequences that are expected to occur with higher probability. However, we found chronological order a surprisingly effective heuristic.
Algorithm 4 bundled step for FSM with S 1 , S 2 Inputs: algorithm state k, variables z 1: set isSample ← 1{k = 2} 2: if k = 1 then 3: run block S 1 with local variables z 4: update state k ← δ(1, z) 5: end if 6: if k = 2 then 7: run block S 2 with local variables z 8: update state k ← δ(2, z) 9: end if 10: return (k, z, isSample)
this section cite: []

Section: Cost Amortization to Reduce α
If a function g is called on a variable θ ∈ z inside M ≤ K state functions, a single call of vmap(step) (or vmap(bundled_step)) will execute g M -times. To prevent this, we (i) augment step to return a flag doComputation indicating if g executed in the next code block, and (ii) define a new step function amortized_step around step which calls step once, and executes g if doComputation=True. The resulting step function is shown in Algorithm 5 (note if g is called in step S 1 , we assume it is already pre-computed in z).
Note the function g is now only called once per step when using vmap on amortized_step. In particular, if the executions of g cost (in total) β ∈ [0, 1] fraction of the total cost of all blocks, amortization will reduce this fraction to β M + (1 -β) We indeed observe this when amortizing g = log p for the elliptical slice sampler in Section 7.1. In that scenario, M = 2 and β ≈ 1 as the log-likelihood is very expensive, resulting in a 2× speed-up over no amortization. One issue is that amortization only allows steps to be bundled which do not require calls of g. We adapt the algorithm to handle this in Appendix B.
Algorithm 5 amortized step for FSM with function g 1: Input: Algorithm state k, variables z 2: (k, z, isSample, doComputation)← step(k, z) 3: if doComputation then
4: Unpack state (z ′ , θ) = z 5: Do computation θ ← g(z) 6:
Re-pack state z ← (z ′ , θ) 7: end if 8: Return (k, z, isSample) 6. Related work FSMs in Machine Learning. Previous work in machine learning has used the framework of FSMs to design imagebased neural networks (Ardakani et al., 2020) and Bayesian nonparametric time series models (Ruiz et al., 2018), as well as extract representations from Recurrent-Neural-Networks (RNNs) (Muškardin et al., 2022;Cechin et al., 2003;Tiňo et al., 1998;Zeng et al., 1993;Koul et al., 2018;Svete and Cotterell, 2023). To our knowledge, our work is the first that uses FSMs as a framework to represent MCMC algorithms, and design novel implementations.
Efficient MCMC on Modern Hardware. Given the recursive nature of HMC-NUTS, previous work has reformulated the algorithm for compatibility with machine learning frameworks that cannot naively support recursion (Abadi et al., 2016;Phan et al., 2019;Lao et al., 2020). However, these implementations do not address synchronization inefficiencies caused by automatic vectorization tools. Our work bears some similarities with a general-purpose algorithm proposed in the High-Performance-Computing literature for executing batched recursive programs (Radul et al., 2020). Both their method and ours breaks programs down into smaller blocks for efficient vectorization, but there are several major differences. Our approach provides a recipe for implementing single-chain iterative MCMC algorithms and is fully compatible with automatic vectorization tools like vmap. In contrast, their algorithm is designed for recursive programs and requires code which is already batched. Crucially, to avoid synchronization barriers due to while loops, this code cannot have been batched with a vmap-equivalent vectorization tool, since vmap converts while loops into a single batched primitive that cannot be broken down by their algorithm. Additionally, our method uses the fewest (while-loop-free) blocks possible and allows for control flow within blocks (which we showed is crucial for optimal performance under the "run and mask" paradigm). By contrast, their algorithm does not allow blocks to contain any controlflow, yielding many small blocks. We also obtain provable speed-ups for MCMC via theoretical analysis.
this section cite: ['b1', 'b28', 'b21', 'b6', 'b32', 'b35', 'b15', 'b31', 'b0', 'b25', 'b16', 'b27']

Section: Experiments
The following experiments evaluate FSMs for different MCMC algorithms with while loops against their standard (non-FSM) implementations, as well as other MCMC methods in one experiment. All methods consist of single chain MCMC algorithms written in JAX, turned into multi-chain methods with vmap, and compiled using jit.
All experiments are run in JAX on an NVIDIA A100 GPU with 32GB CPU memory. Code can be found at https:  //github.com/hwdance/jax-fsm-mcmc.
this section cite: []

Section: Delayed-Rejection MH on a Simple Gaussian
We first illustrate basic properties of the FSM conversion on a toy example. The MCMC algorithm used is symmetric Delayed-Rejection Metropolis Hastings (DRMH) (Mira et al., 2001), which is a simple example of a delayed rejection method. Symmetric DRMH modifies the Random-Walk Metropolis-Hastings algorithm by iteratively re-centering the proposal distribution on the rejected sample and resampling until either acceptance occurs or a maximum number of tries M is reached. To ensure detailed balance, the acceptance probability is adjusted to account for past rejections.
Experimental setup. We implement symmetric DRMH using (vmap'ed) Algorithm 1 (baseline) and Algorithm 3 (ours) to sample from a univariate Gaussian N (0, 1), varying the number of chains. We use a N (x, 0.1) proposal distribution with M = 100 tries per sample and draw 10,000 samples per chain. Although DRMH has 3 state functions by default (see Figure 3), the INIT and DONE states are essentially empty, and so just a single state function can be used for the while loop body. This means an appropriate FSM implementation should be able to get close to R(m), up to overheads. To illustrate the importance of designing FSMs with as few (effective) state functions as possible, we implement an FSM which unrolls the while loop body into four different state functions, as well as a (condensed) FSM with a single state function (i.e. effectively uses bundling).
Results. As the number of chains m increases, the FSM implementations increasingly outperform the standard implementation (see Figure 6). This reflects the increasing synchronization cost of waiting for the slowest chain. The speedups are near an order of magnitude when m = 1024  for the condensed FSM. Bundling sees nearly a 3x efficiency gain and enables no performance loss when m = 1 and there is no synchronization barrier. Note Standard DR tracks the profile of the estimated E[max j∈[m] N 1,j ], whilst the FSMs track the profile of E[N 1,j ] (which is flat). This implies the condensed FSM (i.e. with step bundling) has been able to approximately obtain R(m) up to a constant factor.
this section cite: ['b18']

Section: Elliptical Slice Sampling on Real Estate Data
The elliptical slice sampler (introduced in the example in Section 2.2) has a single while loop, resulting in three state functions (see Figure 3). We compare BlackJAX's implementation to our FSM implementation.
this section cite: []

Section: Experimental setup.
We apply the sampler to infer posteriors on covariance hyperparameters in Gaussian Process Regression, using the UCI repository Real Estate Valuation dataset (Yeh, 2018). This dataset is comprised of n = 414 input and output pairs D n = {x i , y i } n i=1 , where y i is the house price of area i, and x i ∈ R 6 are house price predictors including house age, spatial co-ordinates, and number of nearby convenience stores. We model y = f (x) + ϵ and assume ϵ ∼ N (0, σ 2 ), f ∼ GP(0, k) with kernel k(x, x ′ ) = τ 2 exp(-λ 2 |x -x ′ ∥ 2 ). We use Normal priors σ, τ, λ ∼ N (0, 1), (so the ellipse is drawn using N (0, I)) and use the sampler to draw 10k samples per chain from the posterior p(σ, τ, λ|D n ), for varying # chains m.
Results are shown in Figure 5. As expected, the BlackJAX implementation suffers from synchronization barriers at every iteration due to using vmap with while loops: its average number of iterations per sample increases roughly logarithmically from 6 (1 chain) to 18 (1024 chains), whereas the FSM implementation remains constant. As a result, the FSM significantly improves walltime and ESS/second performance (Figure 5/middle). For instance, when 1024 chains are used, the FSM reduces the time to draw 10k samples per chain from over half an hour to about 10 minutes. The efficiency gain can be measured by the ratio (BlackJAX/FSM) of wall-times (shown in Figure 5/right). As expected, FSM efficiency increases with the number of chains. The greatest efficiency gain occurs precisely where the best ESS/second can be obtained via GPU parallelism. The analysis in Section 4 shows that the efficiency gain is upper-bounded by the ratio of average # iterations per sample for both methods (i.e., R(m)). This bound is almost achieved here by amortizing log-pdf calls. This is because (i) roughly 80% of the time is spent in the iterative 'SHRINK' state, and (ii) log-pdf calls dominate computational cost here, so amortizing them ensures the state function cost is similar to the standard implementation. Since the log-pdf is needed in two states, not amortizing it results in two log-pdf calls per step, and so we lose roughly a factor 2 in relative performance (orange line in Figure 5). These results change with data set size, which determines the cost of log-pdf calls (see Appendix B).
this section cite: ['b34']

Section: HMC-NUTS on High-Dimensional Correlated MoN
The NUTS variant of Hamiltonian Monte Carlo (Hoffman et al., 2014) adaptively chooses how many steps of Hamiltonian dynamics to simulate when drawing a sample, by The FSM achieves speed ups of nearly an order of magnitude for 100 chains, and more than half an order of magnitude for 500 chains.
checking whether the trajectory has turned back on itself or has diverged due to numerical error. Its iterative implementation in BlackJAX involves two nested while loops: An outer loop that expands the proposal trajectory, and an inner loop that monitors for U-turns and divergence. Converting these while loops into an FSM using our procedure results in five states (Figure 3). We again compare BlackJAX's implementation to our own (using vmap for both methods).
Experimental setup. We implement NUTS on a 100dimensional correlated mixture of Gaussians (ρ = 0.99), with the mixture modes placed along the principal direction at (-10 • 1, 0, 10 • 1). We use a pre-tuned step-size with acceptance rate ∼ 0.85 and identity mass matrix. We draw n = 1000 samples per chain and vary # chains m.
Results. The trajectory of a single NUTS chain (1000 samples) are displayed on the LHS of Figure 7 (one dot = one sample). The typical distance traveled by NUTS is small (few integration steps), with the occasional large jump (many integration steps) when the momentum sample aligns with the principal direction. In particular, the probability a sample requires less than 20 integration steps is ∼ 0.95 and needs > 1000 steps is ∼ 0.01, but the probability that at least one chain needs more than 1000 steps is ∼ 0.99 (Middle Figure 7). This results in FSM speed-ups of nearly an order of magnitude for m = 100, and about half an order of magnitude for m = 500 (RHS Figure 7). Note that one can avoid sychronization barriers and obtain very high ESS/Sec using a simpler algorithm like MALA, but this fails to explore the distribution (LHS Figure 7).
this section cite: ['b12']

Section: Transport Elliptical Slice Sampling and NUTS on Distributions with Challenging Local Geometry
Transport elliptical slice sampling (TESS), due to Cabezas and Nemeth ( 2023), is a state-of-the-art variant of elliptical slice sampling designed for challenging local geometries. It uses a normalizing flow T to 'precondition' the distribution π into an approximate Gaussian, does elliptical slice sam- pling on the transformed distribution T # π, and then pushes generated samples through T -1 to recover samples from π. TESS achieves particularly good results on distributions with 'funnel' geometries, on which gradient-based methods like HMC-NUTS tend to struggle (Gorinova et al., 2020). TESS is similar to the elliptical slice sampler and so the FSM has the same 'single loop' structure (see Figure 3).
this section cite: ['b10']

Section: Experimental setup
We examine the speed-ups using FSMs for TESS and NUTS on the four benchmark distributions in Cabezas and Nemeth (2023), which are chosen for their challenging geometries. As baselines we use two state-of-the-art adaptive HMC variants (MEADS (Hoffman and Sountsov, 2022), and CHEES (Hoffman et al., 2021)). We expect NUTS to perform poorly on these problems (as observed in Cabezas and Nemeth (2023)), but implementing it lets us at least examine the FSM speed-ups. For all methods, we average results over 128 chains of 1000 samples, with hyperparameters pre-tuned over 400 warm-up steps.
Results are in Table 1. TESS-FSM improved ESS/sec over TESS in all cases (in 2/4 cases by ∼ 2×) and in 3/4 cases improved the best ESS/sec. NUTS performed poorly as expected, but the FSM improved ESS/sec in 3/4 cases, and > 3× in one case. Speed-ups for NUTS and TESS were larger for tasks with an expensive log-pdf (PP, GS). This is because most time is spent in the loop states where the log-pdf is called, and a more costly log-pdf reduces the gap between the cost of executing these states for non-FSMs (c k ) and FSMs ( m j=1 c j ) (see Section 4), improving efficiency.
this section cite: ['b5', 'b13', 'b11', 'b5']

Section: A. Mathematical Appendix
A.1. Proof of Theorem 4.1
Proof. The result broadly follows from known Hoeffding bounds for Markov chains. For clarity we restate the relevant result belowfoot_6 , using our notation and set-up.
We first define formally the notion of an absolute spectral gap, from (Fan et al., 2021).
Definition A.1 (Absolute Spectral Gap). Let Z and π denote the state space and the invariant measure of a Markov chain {Z i } i≥1 . For a function f : Z → R, write π(h) := h(z) π(dz). Let L 2 (π) = { h : π(h 2 ) < ∞} be the Hilbert space consisting of π-square-integrable functions, and L 0 2 (π) = { h ∈ L 2 (π) : π(h) = 0} be its subspace of π-meanzero functions. The transition probability kernel of the Markov chain, denoted by P , is viewed as an operator acting on L 2 (π). Let λ ∈ [0, 1] be the operator norm of P acting on L 0 2 (π). Then, 1 -λ is the absolute spectral gap of the Markov chain. Proposition A.2 (Hoeffding Bound for Markov Chains -Theorem 1 in Fan et al. (2021)). Let (Z i ) i≥1 be a Markov Chain with measurable state space Z, stationary distribution π, and absolute spectral gap 1 -λ ∈ (0, 1]. Let f : Z → [0, B] be measurable and bounded. Then, for any ϵ > 0,
P π n i=1 f (Z i ) -nE π f (Z 1 ) ≥ ϵ ≤ 2 exp - 1 -λ 1 + λ 2ϵ 2 B 2(14)
where 1 -λ is the absolute spectral gap of π.
Now we are ready to prove our results. We start with C 0 (m, n). First, note that since (X i,j , N i,j ) i≥1 is a Markov chain with stationary distribution π for every chain j, then ({X i,j , N i,j } m j=1 ) i≥1 is also a Markov chain with stationary distribution π m := π ⊗ π ⊗...⊗ m-1 π, because the chains are independent. Therefore, if we set Z i := {X i,j , N i,j } m j=1 and f (Z i ) := max j∈[m] (N i,j ) ∈ [0, B], we get by an application of Proposition A.2 that
P π n i=1 max j∈[m] (N i,j ) -nE π max j∈[m] (N 1,j ) ≥ ϵ ≤ 2 exp - 1 -λ 1 + λ 2ϵ 2 nB 2 (15
)
Setting δ = 2 exp -1-λ 1+λ 2ϵ 2
nB 2 and re-arranging, we get that with probability 1 -δ (under the stationary distribution π),
n i=1 max j∈[m] (N i,j ) -nE π max j∈[m] (N 1,j ) ≤ B n(1 + λ) 2(1 -λ) ln 2 δ(16)
Multiplying by B 0 (m) and dividing by n on both sides, and adding and subtracting A 0 (m) from the LHS gives us the result for C 0
B 0 (m) 1 n n i=1 max j∈[m] (N i,j ) -B 0 (m)E π max j∈[m] (N 1,j ) ≤ B 0 (m)B (1 + λ) 2n(1 -λ) ln 2 δ (17
) B 0 (m) 1 n n i=1 max j∈[m] (N i,j ) ± A 0 (m) -B 0 (m)E π max j∈[m] (N 1,j ) ≤ B 0 (m)B (1 + λ) 2n(1 -λ) ln 2 δ (18
)
C 0 (m, n) -A 0 (m) -B 0 (m)E π max j∈[m] (N 1,j ) ≤ B 0 (m)B (1 + λ) 2n(1 -λ) ln 2 δ (19
)
Now we follow similar steps for C F (m, n). To start, we bound the distance from max j∈m 1 n n i=1 N i,j and E π [N 11 ] in terms of a sum of individual distances using the union bound.
P π max j∈m 1 n n i=1 N i,j -E π N 11 ≥ ϵ = P π max j∈m 1 n n i=1 N i,j -E π N 11 ≥ ϵ + P π max j∈m 1 n n i=1 N i,j -E π N 11 ≤ -ϵ (20
) = P π   m j=1 1 n n i=1 N i,j -E π N 11 ≥ ϵ   + P π   m j=1 1 n n i=1 N i,j -E π N 11 ≤ -ϵ   (21
) ≤ m j=1 P π 1 n n i=1 N i,j -E π N 11 ≥ ϵ + P π 1 n n i=1 N i,j -E π N 11 ≤ -ϵ (22
) = m j=1 P π 1 n n i=1 N i,j -E π N 11 ≥ ϵ (23
) = mP π 1 n n i=1 N i,1 -E π N 11 ≥ ϵ (24
) = mP π n i=1 N i,1 -nE π N 11 ≥ nϵ (25
) Applying Proposition A.2 on the Markov Chain (X 1,i , N 1,i ) i≥1 with f (X 1,i , N 1,i ) = N 1,i ∈ [0, B]
and following the same steps as for C 0 (m, n), we similarly recover
C F (m, n) -A F (m) -B F (m)E π max j∈[m] (N 1,j ) ≤ B F (m)B (1 + λ) 2n(1 -λ) ln 2m δ(26)
which is the result in the Theorem.
Proposition A.3. Fix m, K ∈ N\{0} and let P N be a probability measure on R + strictly positive first moment. Suppose (i)
N 1 , ..., N m iid ∼ P N , (ii) c 1 (m), ..., c K (m) ≥ 0 and (iii) α ∈ [max j∈[K] c j (m)/ j∈[K] c j (m), 1]. Then, we have E(m) := c ¬k (m) + c k (m)E max j∈[K] N j α(c ¬k (m) + c k (m))(K -1 + EN 1 ) ≤ E max j∈[K] N j EN 1 =: R(m)(27)
where c ¬k (m) = j̸ =k c j (m). The bound is tight.
Proof. Note a+b c+d = a c γ + b d (1 -γ) where γ = c c+d for any a, b, c, d ∈ R. Applying this to our case, we get
E(m) = c ¬k (m) α(c ¬k (m) + c k (m))(K -1) w + c k (m) α(c ¬k (m) + c k (m)) R(m)(1 -w)(28)
where
w = α(c ¬k (m)+c k (m)) α(c ¬k (m)+c k (m))(K-1+EN1) ∈ [0, 1].
Now we split into two cases for K = 1 and K > 1. For the case K = 1 we only have a single iterative state and so C ¬k (m) = 0, α = 1. In this case we trivially have E(m) = R(m). Now suppose K > 1. In this case, since α(c ¬k (m) + c k (m)) ≥ max j∈[K] c j (m), we have
c k (m) α(c ¬k (m) + c k (m)) ≤ c k (m) max j∈[K] c j (m) ≤ 1(29)
Which means we can bound E(m) by removing the term in front of R(m),
E(m) ≤ c ¬k (m) α(c ¬k (m) + c k (m))(K -1) w + R(m)(1 -w)(30)
By the same logic, we have
c ¬k (m) α(c ¬k (m) + c k (m))(K -1) ≤ c ¬k (m) max j∈[K] c j (m)(K -1) = j̸ =k c j (m) max j∈[K] c j (m)(K -1) ≤ 1(31)
which means
E(m) ≤ w + R(m)(1 -w) (32) ≤ R(m)(33)
Where the last line uses the fact that R(m) ≥ 1 since N 1 ≥ 0. The bound is tight because when K = 1 we have E(m) = R(m).
This completes the proof B. Additional Details B.1. Algorithms Note here we use x to denote a batch of inputs [x 1 , ..., x m ] for m different chains, and the same for other variables. Algorithm 6 Vectorized MCMC algorithm with vmap(sample) function 1: Inputs: sample x0 , seed r0 2: for i ∈ {1, ..., n} do 3: generate xi , ri ← vmap(sample)( xi-1 , ri-1 ) 4: end for 5: return x1 , . . . , xn Algorithm 7 Vectorized FSM MCMC algorithm with vmap(step) function 1: input: initial value x0 , # samples n 2: initialize: z = vmap(init)( x0 ), X = list(), B = list() 3: Set t = 0 and k = 0 4: while min ti∈ t{ ti } < n do 5: ( k, z, ĩsSample) ← vmap(step)( k, z) 6: append current sample value x stored in z to X 7: append ĩsSample to B 8: update sample counter t ← t + ĩsSample 9: end while 10: return X[ B]
Algorithm 8 Transition kernel for elliptical slice sampler with log-pdf log p, covariance matrix Σ.
1: Input: Sample x 2: Choose ellipse ν ∼ N (0, Σ) 3: Set threshold log y ← log p(x) + log u : u ∼ U[0, 1] 4: Set bracket [θ min , θ max ] ← [θ -2π, θ] : θ ∼ U[0, 2π] 5: Make proposal x ′ ← x cos θ + ν sin θ 6: while log p(x ′ ) > log y do 7:
Shrink bracket and update proposal: 8:
if θ < 0 then 9:
θ min ← θ 10: else 11: θ max ← θ 12:
end if 13:
x ′ ← x cos θ + ν sin θ : θ ∼ U[θ min , θ max ] 14: end while 15: Return x ′
Algorithm 9 bundled step as input to amortized step.
1: Input: Algorithm state k, variables z 2: doComputation = False 3: while not doComputation do 4:
(k, z, isSample, doComputation)← step(k, z) 5: end while 6: Return (k, z, isSample, doComputation)
this section cite: ['b8', 'b8']

Section: B.2. FSM Construction Procedure for Programs Considered in Main Text
Detailed Construction in the two sequential while loops case Here, we describe the FSM construction for programs with two sequential while loops in detail. Following the constructions introduced in the main body, let us note • F 1 := ({S 11 , S 12 , S 13 }, Z, δ 1 , S 11 , S 13 ) the FSM associated to B 1 • F 2 := ({S 21 , S 22 , S 23 }, Z, δ 2 , S 21 , S 23 ) the FSM associated to B 2 By construction, S 21 is empty. Both FSMs share the same input space, which is the set of local variables values associated to the original sample function.
Then, the resulting FSM representation of sample is (S, Z, δ, S 11 , S 23 ), where • S = {S 11 , S 12 , S 13 , S 22 , S 23 }.
• Transition function δ defined as
δ(S, z) =      δ 1 (S, z) if S ∈ {S 11 , S 12 } δ 2 (S 21 , z) if S = S 13 δ 2 (S, z) if S = {S 22 , S 23 },(34)
whose construction illustrates the "FSM stitching" operation performed.
Detailed Construction in the two nested while loop case Here, we describe the FSM construction for programs with two nested while loops. Following the constructions introduced in the main body, let us note
F i := ({S i1 , S i2 , S i3 }, Z, δ i , S i1 , S i3 ) the (inner) FSM associated to B 2 .
Then, the resulting FSM representation of sample is (S o , Z, δ o , S 1 , S 3 ), where
• S o = {S 1 , S i1 , S i2 , S i3 , S 3 }.
• Transition function δ o defined as δ o (S 1 , z) = δ o (S i3 , z) runs the outer while loop condition on z, goes to S i1 if True, and S 3 otherwise.
δ o (S, z) = δ i (S, z) if S ∈ {S i1 , S i2 }
this section cite: []

Section: B.3. FSM Construction Procedure for General Programs.
The FSM construction procedures in ?? for specific programs can be automated to handle programs with any finite number of sequential and/or nested while loops. We show how this works using a simple programming language defined by the following grammar:
Operation op ::= op 1 | op 2 | • • • | op n Simple Block SB ::= ⟨op⟩ | ⟨op⟩ ⟨SB⟩ While Block WB ::= W { ⟨Bs⟩ } Block B ::= ⟨SB⟩ | ⟨WB⟩ Blocks Bs ::= ⟨B⟩ | ⟨Bs⟩ ⟨B⟩
This grammar contains just enough to describe non-trivial programs containing any finite number of (possibly nested) while loops. Given a program generated with this grammar, we construct an FSM variant of it by:
1. Parsing the program into a parse tree.
2. "Coarsening" the parse tree into a tree where each node is a block.
this section cite: []

Section: Applying a recursively-defined function which transforms the program into an FSM.
4. Collapsing any spurious empty states.
this section cite: []

Section: B.3.1. STEP 1: PARSING THE PROGRAM INTO A PARSE TREE.
First, note that as this grammar is context-free, every program P can be represented by the yield of a parse tree (e.g. Theorem 5.12 (Hopcroft et al., 2001)). Consequently, one can use any context-free parser to obtain a parse tree of the program. We choose the look-ahead, left-to-right, rightmost derivation (LALR) parser (DeRemer, 1969). Figure 8 shows the resulting parse tree associated with the program "op1W{op1W{op1}}".
Token('OP1', 'op1') operation simple_block block blocks Token('W', 'W') Token('LBRACE', '{') Token('OP1', 'op1') operation simple_block block blocks Token('W', 'W') Token('LBRACE', '{') Token('OP1', 'op1') operation simple_block block blocks Token('RBRACE', '}') while_block block blocks Token('OP1', 'op1') operation simple_block block blocks Token('RBRACE', '}') while_block block blocks Token('OP1', 'op1') operation simple_block block blocks B.3.2. STEP 2: COARSENING THE PARSE TREE.
To convert the parse tree into an FSM, we first derive a simplified version of it with: (i) no operation nodes (e.g. content of simple blocks), (ii) no block markers (e.g. While tokens and braces), (iii) no "blocks" and "block" nodes (these are helpful to clarify the grammar's definition, but do not carry information about a given program). The first two properties can be achieved by simply pruning the tree of all nodes that are not blocks, which is a well-known operation and not shown here. The output of this procedure is guaranteed to be a tree, as all such nodes (i) and (ii) are terminal. The second property can be achieved by applying the procedure in Algorithm 11.
Algorithm 10 Gathering meaningful children 1: function GATHER CHILDREN(node) returns list of Node 2: all children ← ∪ MAP(GATHER CHILDREN, node.children) 3: all children ← ∪ FILTER(NON TERMINAL, all children) {Discard "op" nodes and markers} 4: if node.type is "block" or "blocks" then 5: return all children {Discard the node, and only return the children} 6: else 7: return [Node(node.type, all children)] 8: end if 9: end function Algorithm 11 Coarsening the parse tree 1: function COARSEN TREE(node) returns Node 2: all children ← ∪ MAP(GATHER CHILDREN, children(node)) 3:
return Node("blocks", all children) 4: end function After applying the coarsening procedure to the parse tree of the same program "op1W{op1W{op1}}", we obtain the coarsened parse tree shown in Figure 9. The algorithm has been modified to return block labels in order to track the graph manipulation operations done in the next steps.
this section cite: ['b14', 'b7']

Section: B.3.3. STEP 3: TRANSFORMING THE COARSENED PARSE TREE INTO AN FSM.
Now we transform the coarsened parse tree into an FSM. Our approach leverages the fact that any subtree of the parse tree is itself a valid (sub)programfoot_7 , and should thus be handle-able by the conversion procedure we define. Each ( sub is either a single simple block, an arbitrary while block (looping over a sequence of blocks), or the root program (e.g. a sequence of blocks). This suggests a recursive conversion procedure which computes a single-state FSM for the block at the leaf nodes, and then (recursively) combines these FSMs into larger FSMs at each layer of the parse tree, until we reach the root node, at which point the final FSM is returned. The FSM at leaf nodes (i.e. simple blocks) is a simple single state FSM. The FSM at non-leaf nodes (e.g. while blocks or the entire program) is constructed by (i) connecting the terminal state of the i th child's FSM to the initial state of the (i + 1) th child's FSM, for every child i of the non-leaf node except the last, and (ii) if the node is a while block, (a) adding an edge from the last child's FSM to the first child's FSM, and (b) assigning the initial and terminal state of the current node's FSM to empty states with an edge between them. Additions (a) and (b) account for the fact that unlike standard blocks, while blocks have bodies that may be looped over, or skipped.
Using empty states as initial and terminal FSM of while blocks allows our FSM conversion procedure to be "context-free"each subFSM is obtained by assembling sub-subFSMs, without knowledge about the nature of the blocks being connected. Without these empty states, we would require more complicated logic to perform the connection. For instance, in a program of the form "op1W{op1}W{op1}op2", one needs to connect the first op1 to the last op2 to encode the fact that both while loops may be skipped.
The full algorithm is given in Algorithm 12. The resulting FSM for the program "op1W{op1W{op1}}" is shown in Figure 10. The nodes starting with an "I" (resp. "T") are the (empty) initial (resp. terminal) states of each while block. The transitions are labeled using the following convention:
• "E⟨n⟩" means "enter loop number n"
• "C⟨n⟩" means "continue loop number n"
• "S⟨n⟩" means "skip loop number n"
• "F⟨n⟩" means "exit loop number n" The empty states produced by the step above are side effects of the conversion procedure used: the final FSM should not contain them. To this end, we next describe a post-processing, iterative procedure which removes them from the FSM. At each iteration, the procedure selects (if any) an empty state, and-unless it transitions into itself-(i)removes from the list of FSM nodes and (ii) adds edges connecting the parents of the empty state to its children.
The labels of the new edges are obtained by concatenating the "parent-to-empty state" and "empty state-to-children" edges. The procedure is given in Algorithm 13, and the resulting FSM for the program "op1W{op1W{op1}}" is shown in Figure 11. The procedure results in FSMs that agree those in the main text for single while loops, two nested while loops, and two sequential while loops. There are two minor caveats, which we discuss below. Remark B.1 (Handling empty states with self-transitions). The procedure above removes all empty states, apart from those with self-transitions. These self-transitions are not present before collapsing the empty states, and arise in the midst of the procedure, after collapsing a subset out of all the FSM empty state. Importantly, such self-transitions are pathological: as the state of the program does not change when performing a self-transition into an empty node, if such a self-transition occurs, it must keep occurring indefinitely afterwards. Thus, programs resulting into FSM with self-recurring empty states form a class of "potentially non-halting" programs. Remark B.2 (Handling impossible transitions). The node collapsing process results in the creation of "multistep" edges. These transitions involve evaluating multiple conditions in a single step. Sometimes, these conditions are mutually exclusive: in that case, this multistep transition is impossible, and can be removed from the graph. By removing such transitions after each step of the procedure, the routine can be shown to not add any duplicated edges to the FSM, as proved in the next proposition.
Algorithm 13 Collapsing empty states 1: function COLLAPSE EMPTY STATES(fsm) returns FSM 2: edges ← fsm.edges 3: labels ← fsm.labels 4: for all node in FILTER(EMPTY | NOT INITIAL | NOT FINAL, fsm.nodes) do 5: if node ∈ node.children then 6: continue 7: end if 8: for all parent, child in PRODUCT(node.parents, node.children) do 9: new edge ← (parent, child) 10: if new edge ∈ edges then 11: error {Edge already exists} 12: end if 13: new label ← AND(parent.label, child.label) 14: if IMPOSSIBLE(new label) then 15: continue 16: end if 17: edges ← edges ∪ {new edge} 18: labels ← labels ∪ {new label} 19: end for 20: end for 21: nodes ← FILTER(NON EMPTY, fsm.nodes) 22: return FSM(nodes, edges, labels) 23: end function B.4. Implementation Details
this section cite: []

Section: FSM Step Function Design.
At each step of the FSM, one always executes a block S k : z → z ′ which updates the inputs, before calling the transition function δ on (k, S k (z)) to determine which block should be run next. One can therefore combine these into a single (state) function T : (k, z) → (δ(k, S k (z)), S k (z)), or equivalently, a collection of functions T 1 , ..., T K where T k : z → (δ(k, S k (z)), S k (z)). In practice, our implementation of Algorithm 2 calls a single switch over these composite state functions to streamline the code. The same composite state functions are also used in our implementation of bundled_step in Algorithm 4.
this section cite: []

Section: FSM Runtime Design.
In our experiments, we use a native Python while loop in Algorithm 3 (and its vmap-ed variant in Algorithm 7). Inside the loop, we use a (JIT compiled) jax.lax.scan to run the FSM step function for blocks of t = 100 steps, when drawing n > 100 samples. This gives us the flexibility to store the results in dynamically shaped lists/arrays and transport to the CPU for faster array slicing when CPU memory is available, whilst still reaping the benefits of JIT compilation.
Compilation. We JIT compile both vmap(step) and vmap(sample) functions for each MCMC algorithm implementation (here step refers to any of the basic variant Algorithm 2, bundled variant Algorithm 4, amortized variant Algorithm 5, or the bundled and amortized variant Algorithm 9). For Delayed Rejection and the Elliptical Slice Sampler (with n = 25) we remove compilation time to get more accurate results or the runs with small numbers of chains m, due to the low cost of the computations involved.
this section cite: []

Section: JAX implementation.
When comparing to non-FSM implementations, we used BlackJAX (Cabezas et al., 2024) for fair comparison with our method, since we use BlackJAX primitives for the key computations in some of our algorithms (e.g. HMC-NUTS). Where a BlackJAX implementation was not available (e.g. Delayed Rejection), we wrote our own for fair comparison with our FSM implementation.
this section cite: ['b4']

Section: Bundling with Amortized Step.
As discussed in Section 5, one cannot non-trivially use bundled_step as the step function inside amortized_step, because a given sequence of states may require the amortized computation to be updated in the middle, and nothing in bundled_step flags this. To reap some of the benefits of step bundling when amortizing an expensive computation g (e.g. expensive log-pdf calculations), we use another step function defined in Algorithm 9, as the 'step' function called inside amortized_step. This step function iteratively runs step (modified to return the doComputation flag) until doComputtion = True -ironically, using a while loop. When vmap-ed, each 'step' inside amortized_step now executes sequences of cheap states that do not require g, until g is required again for all chains. This works effectively when g is called inside the iterative state, which is typically the case when g = log p.
this section cite: []

Section: B.5. Additional Results
Figure 12. Efficiency Ratio of our elliptical slice FSM against BlackJAX's elliptical slice algorithm (as measured by estimated R(m) = E[max j∈[m] Nj]/E[N1] (i.e. iters per sample) and walltime) on the Real Estate Dataset described in Section 7 when restricting the dataset to the first n ∈ {25, 100, 400} datapoints. The relative efficiency of the FSM improves as the number of chains used increase, and as the log-likelihood cost increases. When n = 400, we almost achieve the theoretical bound R(m) in speed-ups.
Figure 13. Walltimes and ESS per second using the Elliptical Slice Sampler (non-FSM vs FSM implementation) on the Real Estate Dataset described in Section 7, when restricting the dataset to the first n ∈ {25, 100, 400} datapoints. For each dataset size, the best walltime and ESS/second is obtained by both implementations when using m = 1024 chains. Our FSM implementation can obtain the greatest efficiency for all dataset sizes. As the log-likelihood cost increases (the log-likelihood in GPR regression costs O(n 3 )), we see the FSM efficiency gain increase, reflecting the benefits of amortization.
Distribution NUTS-FSM Standard NUTS FSM Speed-up Real-Estate GPR 863.6 274.1 3.15x Soil 0.014 0.006 2.43x Pilots 3.538 3.869 0.91x
Table 2. ESS/sec comparison for NUTS with and without FSM acceleration on Real-Estate GPR sampling problem considered in Section 7.2, and two sampling problems (Soil and Pilots) from the PosteriorDB database (Magnusson et al., 2025). Results are averaged over 1000 samples and 128 chains. As with Section 7.4 in the main text, we use 400 warm-up steps to tune the mass matrix and step size of our FSM implementation, and the BlackJAX (non-FSM) implementation. We find substantial speed-ups in 2/3 problems. Both of these problems (Real-estate GPR and Soil) have an expensive log-pdf (and gradient) whilst the log-pdf in Pilots is much cheaper. In general, when the number of integration steps taken by NUTS is moderate to high on average (so that most time is spent integrating along the trajectory), an expensive log-pdf makes the additional cost of executing all other blocks in the FSM step function (when integrating) relatively smaller. That is, the gap between E(m) and R(m) (as defined in Section 4) is smaller, hence increasing the relative efficiency of the FSM. The opposite happens when the log-pdf is cheap (as in Pilots). We also note that computational resources did not balance out across chains in Soil and Pilots after 1000 samples. We therefore would expect to see larger speed-ups when running for longer.
this section cite: ['b17']

Section: Efficiently Vectorized MCMC on Modern Accelerators
Algorithm 12 FSM creation from a coarsened parse tree 1: function COARSE TREE TO FSM(node) returns FSM 2: if node.type is "while block" or "blocks" then 3: c fsms ← MAP(COARSE TREE TO FSM, node.children) 4: nodes ← ∪ MAP(GET NODES, c fsms) 5: edges ← ∪ MAP(GET EDGES, c fsms) 6: labels ← ∪ MAP(GET LABELS, c fsms) 7: // Connect the different subFSMs 8: conn edges ← {(c fsms[i-1].terminal, c fsms[i].initial) for i = 1, . . . , len(c f sms) -1} 9: conn labels ← {"N", . . . , "N"} 10: edges, labels ← edges ∪ conn edges, labels ∪ conn labels 11: i ← COUNTER() 12: if node.type == "while block" then 13: terminal state, initial state ← Node("terminal"), Node("initial") 14: enter loop edge ← (initial state, c fsms[0].initial) 15: exit loop edge ← (c fsms[-1].terminal, terminal state) 16: continue loop edge ← (c fsms[-1].terminal, c fsms[0].initial) 17: skip loop edge ← (terminal state, initial state) 18: nodes ← nodes ∪ {terminal state, initial state} 19:
edges ← edges ∪ {continue loop edge, skip loop edge, enter loop edge, exit loop edge} 20: labels ← labels ∪ {"E" + i, "F" + i, "C" + i, "S" + i}
21: return FSM(nodes, edges, labels) 22: else 23: return FSM([simple block], [ ], [ ]) 24: end if 25: end if 26: end function
S1 S2 S5 S3 S4 C2 C1 (N + E1) (N + E2) (N + S2 + N) (F2 + N) (N + S1 + N) (F1 + N)
Figure 11. FSM for the program "op1W{op1W{op1}}", with empty states collapsed, using the automatic procedure. Note that this matches the structure of the FSM for the No-U-Turn sampler in Figure 2, which was produced using (essentially) a special case of this procedure for programs with two nested while loops.
this section cite: []

Section: References
Ref_id:b0 Title: Tensorflow: Large-scale machine learning on heterogeneous distributed systems Year: (2016)
Ref_id:b1 Title: Training linear finite-state machines Year: (2020)
Ref_id:b2 Title: Sample with multiple chains in parallel Year: (2019)
Ref_id:b3 Title: Handbook of markov chain monte carlo Year: (2011)
Ref_id:b4 Title: Blackjax: Composable bayesian inference in jax Year: (2024)
Ref_id:b5 Title: Transport elliptical slice sampling Year: (2023)
Ref_id:b6 Title: State automata extraction from recurrent neural nets using k-means and fuzzy clustering Year: (2003)
Ref_id:b7 Title: Practical translators for LR (k) languages Year: (1969)
Ref_id:b8 Title: Hoeffding's inequality for general markov chains and its applications to statistical learning Year: (2021)
Ref_id:b9 Title: Flax: A neural network library for jax Year: (2023)
Ref_id:b10 Title: Automatic reparameterisation of probabilistic programs Year: (2020)
Ref_id:b11 Title: An adaptive-mcmc scheme for setting trajectory lengths in hamiltonian monte carlo Year: (2021)
Ref_id:b12 Title: The no-u-turn sampler: adaptively setting path lengths in hamiltonian monte carlo Year: (2014)
Ref_id:b13 Title: Tuning-free generalized hamiltonian monte carlo Year: (2022)
Ref_id:b14 Title: Introduction to automata theory, languages, and computation Year: (2001)
Ref_id:b15 Title: Learning finite state representations of recurrent policy networks Year: (2018)
Ref_id:b16 Title: tfp. mcmc: Modern markov chain monte carlo tools built for modern hardware Year: (2020)
Ref_id:b17 Title: \texttt {posteriordb}: Testing, benchmarking and developing bayesian inference algorithms Year: (2025)
Ref_id:b18 Title: On metropolis-hastings algorithms with delayed rejection Year: (2001)
Ref_id:b19 Title: Delayed rejection hamiltonian monte carlo for sampling multiscale distributions Year: (2024)
Ref_id:b20 Title: Elliptical slice sampling Year: (2010)
Ref_id:b21 Title: Learning finite state models from recurrent neural networks Year: (2022)
Ref_id:b22 Title: Slice sampling. The annals of statistics Year: (2003)
Ref_id:b23 Title: Neuromechanical autoencoders: Learning to couple elastic and neural network nonlinearity Year: (2023)
Ref_id:b24 Title: Ab initio solution of the many-electron schrödinger equation with deep neural networks Year: (2020)
Ref_id:b25 Title: Composable effects for flexible and accelerated probabilistic programming in numpyro Year: (2019)
Ref_id:b26 Title: Unbiased contrastive divergence algorithm for training energy-based latent variable models Year: (2019)
Ref_id:b27 Title: Automatically batching controlintensive programs for modern accelerators Year: (2020)
Ref_id:b28 Title: Infinite factorial finite state machine for blind multiuser channel estimation Year: (2018)
Ref_id:b29 Title: Jax, md a framework for differentiable physics Year: (2021)
Ref_id:b30 Title: Running markov chain monte carlo on modern hardware and software Year: (2024)
Ref_id:b31 Title: Recurrent neural language models as probabilistic finite-state automata Year: (2023)
Ref_id:b32 Title: Finite state machines and recurrent neural networks-automata and dynamical systems approaches Year: (1998)
Ref_id:b33 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b34 Title: Real Estate Valuation. UCI Machine Learning Repository Year: (2018)
Ref_id:b35 Title: Learning finite state machines with self-clustering recurrent networks Year: (1993)
