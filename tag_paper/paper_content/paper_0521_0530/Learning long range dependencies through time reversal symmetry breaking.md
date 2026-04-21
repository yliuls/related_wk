Title: Learning long range dependencies through time reversal symmetry breaking
Abstract: Deep State Space Models (SSMs) reignite physics-grounded compute paradigms, as RNNs could natively be embodied into dynamical systems. This calls for dedicated learning algorithms obeying to core physical principles, with efficient techniques to simulate these systems and guide their design. We propose Recurrent Hamiltonian Echo Learning (RHEL), an algorithm which provably computes loss gradients as finite differences of physical trajectories of non-dissipative, Hamiltonian systems. In ML terms, RHEL only requires three "forward passes" irrespective of model size, without explicit Jacobian computation, nor incurring any variance in the gradient estimation. Motivated by the potential to implement our algorithm in non-digital physical systems, we first introduce RHEL in continuous time and demonstrate its formal equivalence with the continuous adjoint state method. To facilitate the simulation of Hamiltonian systems trained by RHEL, we propose a discrete-time version of RHEL which is equivalent to Backpropagation Through Time (BPTT) when applied to a class of recurrent modules which we call Hamiltonian Recurrent Units (HRUs). This setting allows us to demonstrate the scalability of RHEL by generalizing these results to hierarchies of HRUs, which we call Hamiltonian SSMs (HSSMs). We apply RHEL to train HSSMs with linear and nonlinear dynamics on a variety of time-series tasks ranging from mid-range to long-range classification and regression with sequence length reaching ∼ 50k. We show that RHEL consistently matches the performance of BPTT across all models and tasks 3 . This work opens new doors for the design of scalable, energy-efficient physical systems endowed with self-learning capabilities for sequence modelling.

Section: Introduction
The resurgence of Recurrent Neural Networks (RNNs) for sequence modeling [1], particularly when integrated with State Space Models (SSMs) [1][2][3][4][5][6], reopens the debate about the "hardware lottery". This raises a critical question with high stakes [7]: should future hardware development continue to optimize for the long-dominant GPU/TPU-Transformer-backprop paradigm [8], or should it explore alternative computational approaches alongside novel training algorithms?
This paper embraces a distinctive standpoint in the realm of AI hardware by not distinguishing algorithms from hardware [9] and posits that some SSMs could be physically realized in dynamical physical systems and, when endowed with bespoke credit assignment mechanisms, be turned into "self-learning" machines [10,11] -see Section 5 for a detailed discussion. To this end, such algorithms should fulfill at least two requirements: i) use "forward passes" only (i.e. no backward passes), ii) do not use explicit state-Jacobian (i.e. Jacobian of the system's dynamics). The vast majority of existing algorithms endowed with these features revolve around forward-mode automatic differentiation (AD) [12]. The standard forward-mode AD algorithm for temporal processing is Recurrent Real-Time Learning [13] (RTRL). Due to its cubic memory complexity with respect to the number of neurons, RTRL can only be exactly implemented on architectures of limited size [14] and either requires low-rank approximations [15], or hardcoded [16] or metalearned [17] heuristics.
Figure 1: HEB core mechanics [10]. After following a free forward trajectory (blue curve) and undergoing "rebound", the neurons exactly travel backward (grey curve). Instead, HEB prescribes nudging the "echo" trajectory (red curve) closer to y before rebound, with the resulting position gap encoding the error gradient with respect to momentum (the left green arrow).
RTRL fails to satisfy criteria ii) because it explicitly uses the Jacobian to compute directional derivatives of the loss L along each direction v i of the canonical basis of parameters θ as a supplementary computation during the forward pass. However, it can be reformulated as a zeroth-order procedure by approximating the directional derivatives via ∇ θ L • v = lim ϵ→0 ϵ -1 (L(θ + ϵv) -L(θ)). This recipe theoretically aligns with our three algorithmic requirements but incurs a prohibitive complexity cost by requiring separate forward passes for each perturbation along every parameter direction. Alternative methods attempt to circumvent this by sampling random directions from the parameter space [18][19][20][21] rather than exhaustively computing gradients along the entire canonical basis. However, these approaches suffer from either excessive variance [22] or high bias [23], limiting their effectiveness to small-scale applications [24] or fine-tuning tasks [25]. To avoid the pitfalls of RTRL and other forward-mode AD proxies, a better approach would be to instead emulate backward-mode AD, forward in time [26]. Yet, such techniques have only developed to emulate backwardmode implicit differentiation on energy-based models on static inputs [27] and have not been extended out of equilibrium to sequential data.
In this work, we propose an alternative approach that emulates backward-mode AD forward in time, inspired by the Hamiltonian Echo Backprop (HEB) algorithm [10], illustrated in Fig. 1 for a single neuron. Consider a system of neurons described by their position ϕ i and momentum π i which do not dissipate energy. This means, for instance for coupled oscillators (Eq. ( 8)), that the initial energy provided to this system is preserved and transferred across oscillators from kinetic energy to elastic potential or vice versa. Such systems can be modelled by the Hamiltonian formalism (section 2.1). Let us say that we want to learn the initial conditions on this system such that it reaches some target y after some time. HEB proceeds in three steps. For the first step, the system evolves freely, but does not reach y. For the second step, HEB perturbs the trajectory for a brief duration to drive the system slightly closer to y per some metric L. Then in the last step, the neurons are "bounced" backwards, causing the system to evolve backward. If the perturbation was omitted, the system would exactly retrace its previous trajectory backward in time, a property which is called time-reversal symmetry. Because the perturbation breaks time-reversal symmetry, the system no longer retraces its initial trajectory exactly-the resulting deviation encodes the gradient of L with respect to its initial state.
In spite of its significant implications for physical learning, HEB has multiple features which may hinder its broader investigation within the ML community: i) HEB is difficult to compare to standard ML algorithmic baselines as it was derived with dedicated theoretical physics tools and is entangled with fine-grained physics of the systems being trained; ii) HEB theory assumes that model parameters are also dynamical variables and that only their initial state is learnable, which is in stark contrast with standard RNN parametrization; iii) it is unclear how HEB would extend to SSMs, i.e. in discrete time, on sequential data with losses defined at all timesteps, on hierarchical recurrent units; iv) finally, HEB was only evaluated on small static problems (e.g. XOR, MNIST) and not proved at larger scale.
Taking inspiration from HEB and using standard ML tools, we propose Recurrent Hamiltonian Echo Learning (RHEL) as a simple, general and scalable forward-only proxy of backward-mode AD applying to a broad class of dissipative-free Hamiltonian models with the following key contributions:
• With the primary goal in mind to inform the design of self-learning physical systems and to facilitate its comparison to HEB, we first introduce RHEL in continuous time. We show that RHEL generalizes HEB to a broader class of models and problems and demonstrate its equivalence, at every timestep, with the continuous adjoint state method [28] in the limit of small trajectory perturbations (Theorem 3.1). We numerically highlight this property on a toy physical model (Eq. ( 8), Fig. 2).
• To efficiently simulate this algorithm, we extend RHEL to discrete time and demonstrate its equivalence with Backpropagation Through Time (BPTT) (Theorem 3.2) on Hamiltonian Recurrent Units (HRUs) -which are discrete-time, symplectic integrators of separable Hamiltonian dynamics. We further generalize this result to learning a hierarchy of HRUs, which we call Hamiltonian State Space Models (HSSMs, Fig. 2), and propose a RHL chaining procedure accordingly (Theorem 3.3).
• With this simulation toolbox in hand, we finally demonstrate the effectiveness of the proposed approach on HSSMs with linear [29] and nonlinear [30] recurrent units. We show that: i) gradients estimates produced by RHEL near perfectly match gradients computed by end-to-end AD (Fig. 4), ii) RHEL remains on par with AD in terms of resulting model performance across classification and regression long-range tasks (Tables 12).
2 Problem statement
this section cite: ['b0', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b9', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b9', 'b27', 'b28', 'b29']

Section: Notations & model description.
Notations. Given a differentiable mapping H : R d → R, we denote ∇ v1 H ∈ R dv 1 ×1 and ∇ 2 v1,v2 H ∈ R dv 1 ×dv 2 the gradient and Hessian of H respectively. When necessary, we use the Leibniz notation to denote as ∇ i H the gradient of H with respect to the i th variable. When considering a differentiable mapping M : R d → R n , we denote its Jacobian matrix as ∂M ∈ R n×d . We also use the notation d v1 to denote a total derivative with respect to some variable v 1 which accounts for both direct and indirect effects of the variable v 1 on the function being differentiated.
this section cite: []

Section: Continuous Hamiltonian model.
Following [10], we model neurons as a vector Φ ∈ S ≡ R d Φ comprising a position vector ϕ ∈ R d Φ 2 and a momentum vector π ∈ R d Φ 2 such that Φ := (ϕ ⊤ , π ⊤ ) ⊤ . We define Σ z ∈ R d Φ ×d Φ such that Σ z • Φ = (ϕ, -π) and Σ x ∈ R d Φ ×d Φ such that Σ x • Φ = (π, ϕ). Denoting θ ∈ Θ ≡ R d θ and u ∈ U ≡ R du the model parameters and inputs, we define the Hamiltonian associated to the model as a mapping H : S × Θ × U → R. Taking -T as the origin of time and starting from Φ(-T ) = x, we say that Φ follows Hamiltonian dynamics under a sequence of inputs t → u(t) and with some parameters θ if it satifies the ordinary differential equation (ODE): ∀t ∈ [-T, 0] : ∂ t Φ(t) = J • ∇ Φ H[Φ(t), θ, u(t)], J := 0 I -I 0 .
Assumptions. The most fundamental requirement of the whole proposed approach for the Hamiltonian models at use is time-reversal symmetry. Heuristically, if we were recording the dynamics of a conversative system described by Eq. ( 1) and playing the resulting recording forward and backward, we could not distinguish them as both are physically feasible (Lemma A.3). A direct consequence of this fact is that if neurons are let to evolve for some time and then bounced back, i.e. their momenta are reversed, they will exactly travel back to their initial state (Corollary A.3). Namely (see Fig. 1):
Φ(-T ) -→ Eq. (1) Φ(0) ⇒ Φ ⋆ (0) := Σ z • Φ(0) -→ Eq. (1) Φ ⋆ (-T )(2)
this section cite: ['b9']

Section: Learning as constrained optimization
Problem formulation. Our goal can be framed as a constrained optimization problem where we aim to find a set of model parameters θ such that, under some input sequence t → u(t), the model trajectory approaches as much as possible some target trajectory, as measured by a loss function L, under the constraint that the model trajectory is physically feasible in the sense of satifying Eq. ( 1):
min θ L := 0 -T dtℓ[Φ(t), t] s.t. ∀t ∈ [-T, 0] : ∂ t Φ(t) = J • ∇ Φ H[Φ(t), θ, u(t)],(3)
... where L reads as the sum of cost functions t → ℓ(•, t). The most common approach to solve Eq. ( 3) is by gradient descent or variants thereof such that the problem boils down to computing d θ L. Note that the problem defined by Eq. ( 3) is more general than the one solved by the seminal HEB work [10] as: i) the loss function L is defined over the whole trajectory, ii) the Hamiltonian is time-dependent (through t → u(t)) and parametrized by θ which is shared across the whole computational graph.
this section cite: ['b9']

Section: Algorithmic baseline.
One standard approach to compute d θ L is the continuous-adjoint state method (ASM) [28], which can be simply regarded as the continuous counterpart of backward-mode AD. Given some forward trajectory {Φ(t)} t∈[-T,0] spanned by Eq. ( 1), this method prescribes solving the following backward ODE:
λ(0) = 0, ∂ t λ(t) = ∇ 2 1 H[Φ(-t), θ, u(-t)] • J ⊤ • λ(t) + ∇ 1 ℓ[Φ(-t), -t],(4)
with the gradient of the loss with respect to the initial state of the neurons x and the model parameters θ given by (Theorem A.1):
d x L = λ(T ), d θ L = T 0 g θ (t)dt, with g θ (t) := ∇ 2 1,2 H[Φ(-t), θ, u(-t)] • J ⊤ • λ(t) (5)
Note that in theory, the forward trajectory {Φ(t)} t∈[-T,0] can either be stored or recomputed backwards alongside λ yielding O(1) memory cost -a property which holds beyond Eq. ( 1) specifically [31]. However in practice, recomputing variables backward through a discrete computational graph is generally inexact and induces bias in the gradient estimation [32], unless a reversible discretization scheme is used. Fortunately, symplectic integrators associated with Hamiltonian flows are reversible [33], a property which has been leveraged to yield memory savings in neural networks [30]. Therefore, the continuous ASM as well as our proposed algorithm (Alg. 1) also naturally inherit this memory efficiency as a model feature rather than a feature of the training algorithms themselves.
3 Recurrent Hamiltonian Echo Learning (RHEL)
this section cite: ['b27', 'b30', 'b31', 'b32', 'b29']

Section: Definition in continuous time & equivalence with continuous ASM
We are now equipped to introduce Recurrent Hamiltonian Echo Learning (RHEL). In comparison to the continuous ASM, RHEL does not require solving a separate adjoint ODE akin to Eq. ( 4). Instead, RHEL simply prescribes running multiple additional times the forward trajectory Eq. ( 1) with three modifications: i) the neurons are first conjugated, i.e. Φ → Φ ⋆ ; ii) the inputs t → u(-t) are processed backwards; iii) the trajectory of the neurons is slightly nudged, with a strength ϵ, towards direction of decreasing loss values with cost functions t → ℓ[•, -t] processed backwards too. More precisely, let us assume a "free" forward trajectory Eq. ( 1) has been executed between -T and 0 yielding some neuron state Φ(0). Then, we define the echo dynamics of the neurons Φ e through:
Φ e (0) = Φ ⋆ (0), ∂ t Φ e (t, ϵ) = J ∇ Φ e H[Φ e (t, ϵ), θ, u(-t)]-ϵJ ∇ Φ e ℓ[Φ e (t, ϵ), -t] ∀t ∈ [0, T ](6)
A crucial observation is that when ϵ = 0, the echo trajectory simply matches the forward trajectory in reverse: Φ e (t, ϵ = 0) = Φ ⋆ (-t) for t ∈ [0, T ] (Lemma A.3), reflecting time-reversal symmetry. When ϵ ̸ = 0, this symmetry is broken: the resulting trajectory difference Φ e (t, ϵ) -Φ e (t, ϵ = 0) implicitly encodes for the error signals carried by the continuous ASM. We introduce this result more formally in Theorem 3.1 with the help of the following quantities capturing the essence of RHEL as a forward-only, difference-based gradient estimator:
∆ RHEL θ (t, ϵ) := -1 2ϵ (∇ 2 H[Φ e (t, ϵ), θ, u(-t)] -∇ 2 H[Φ e (t, -ϵ), θ, u(-t)]) , ∆ RHEL Φ (t, ϵ) := 1 2ϵ Σ x • (Φ e (t, ϵ) -Φ e (t, -ϵ)) ,(7)
Theorem 3.1 (Informal). Under mild assumptions on the Hamiltonian function and with t → λ(t) and t → g θ (t) defined by Eq. ( 4)-( 5), the continuous ASM and RHEL are equivalent at all times in the sense that: Example. We numerically verify Theorem 3.1 on a toy model of six coupled harmonic oscillators i (Fig. 2) with masses m i and spring parameters k i , k ij . The system is driven by an external input force u(t) applied to oscillator 1 and produces an output on oscillator 4, which is nudged toward a target trajectory y(t) during the echo phase. The corresponding Hamiltonian reads:
∀t ∈ [0, T ], λ(t) = lim ϵ→0 ∆ RHEL Φ (t, ϵ), g θ (t) = lim ϵ→0 ∆ RHEL θ (t, ϵ) Proof sketch. Defining ∆ RHEL Φ (t) := lim ϵ→0 ∆ RHEL Φ (t, ϵ) and noticing that ∆ RHEL Φ (t) = Σ x • ∂ ϵ (Φ e (t,
H[Φ, θ, u] = 6 i=1 π 2 i 2m i + 1 2 6 i=1 k i ϕ 2 i + 1 2 6 i=1 6 j>i k ij (ϕ j -ϕ i ) 2 + uϕ 1 ,(8)
with parameters θ = {m i , k i , k ij } i,j . This yields the following dynamics and associated RHEL gradient estimators (see App. A.4.1):
     ∀i ∈ 1, 6 , ∂ t ϕ i = π i /m i , ∂ t π i = -k i ϕ i + j̸ =i k ij (ϕ j -ϕ i ) + δ i1 u + δ i4 δ e ϵ(ϕ 4 -y), g kij (t) = lim ϵ→0 -1 2ϵ ϕ e j (t, ϵ) -ϕ e i (t, ϵ) 2 -ϕ e j (t, -ϵ) -ϕ e i (t, -ϵ) 2 ,
where u(t) denotes the external driving force applied to oscillator 1, y(t) the target trajectory associated with the output oscillator 4, δ ij the Kronecker delta, and δ e an indicator equal to 1 during the echo phase and 0 otherwise.
this section cite: []

Section: Extension to discrete time & equivalence with BPTT
In this section, we propose a discrete-time version of RHEL by first defining a family of discrete models as integrators of the Hamiltonian flow which exactly preserve time-reversal symmetry, defining an associated surrogate problem and then solving it. This "discretize-then-optimize" approach ensures a better gradient estimation than directly discretizing RHEL theory in continuous time [32]. Hamiltonian Recurrent Units (HRUs). We now assume that the neurons Φ obey the following discrete-time dynamics, starting from Φ -K = x and:
Φ k+1 = M H,δ [Φ k , θ, u k ] ∀k = -K • • • -1,(9)
where M H,δ denotes the integrator associated with the Hamiltonian function H with time discretization δ. We classically pick M H,δ as the Leapfrog integrator [33] and restrict ourselves to separable Hamiltonians of the form H[Φ] = T [π] + V [ϕ] to yield a reversible and explicit integration scheme. One possible parametrization of the Leapfrog integrator in this case is as the composition of three explicit Euler integrators, which alternate between updates of the position ϕ (first and third steps) and of the momentum (second step) -see Def. A.3 for a detailed description and Fig. 3 for the associated computational graph. As these two intermediate integrator time steps are needed to accurately define the RHEL learning rule in discrete time, we explicitly denote them as fractional time units:
M H,δ : Φ k -→ Φ k+1/3 -→ Φ k+2/3 -→ Φ k+1(10)
We call such models Hamiltonian Recurrent Units (HRUs). Therefore by design, the time-reversal symmetry property defined in Eq. ( 2) extends in discrete time to HRUs (Corollary A.6).
this section cite: ['b31', 'b32']

Section: Surrogate learning problem.
Given this modelling choice, the continuous-time problem introduced in Eq. ( 3) naturally translates here in discrete time as:
min θ L := 0 k=-K+1 ℓ[Φ k , k] s.t. Φ k+1 = M H,δ [Φ k , θ, u k ] ∀k = -K • • • -1(11)
Algorithmic baseline. Eq. ( 11) defines a classical RNN learning problem where Backpropagation Through Time (BPTT), i.e. the instantiation of backward-mode AD in this context, is a natural algorithmic baseline. BPTT simply amounts to apply the "chain rule" backward through the computational graph spanned by the forward pass of a HRU (Eq. ( 9)). Alternatively, it can also be regarded as the discrete counterpart of the continuous ASM previously introduced (Theorem A.3). For this reason, we re-use the same notations as above and define the following quantities associated to BPTT:
d θ L = K-1 k=0 g θ (k), g θ (k) := d θ k L, g u (k) := d u -k L, λ k := d Φ -k L,(12)
where g θ (k), g u (k) and λ k denote the "sensitivity" of the loss L to θ at time step k, u -k and Φ -k respectively -see App. A.2.2 for a more detailed definition and derivation of BPTT.
RHEL in discrete time. Finally, we extend RHEL in discrete-time by defining the echo dynamics on HRUs as:
Φ e 0 (ϵ) = Φ ⋆ 0 + ϵΣ x • ∇ Φ ℓ[Φ 0 , 0], Φ e k+1 (ϵ) = M H,δ [Φ e k (ϵ), θ, u -(k+1) ] -ϵJ • ∇ Φ e ℓ[Φ e k+1 , -(k + 1)] ∀k = 0, • • • , K -1 (13
)
Denoting:
H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] := 1 2 H[Φ e k+1/3 (ϵ), θ, u -(k+1) ] + H[Φ e k+2/3 (ϵ), θ, u -(k+1) ] , (14
)
we can define the discrete-time counterpart of Eq. ( 7) as:
   ∆ RHEL θ (k, ϵ) := -δ 2ϵ ∇ 2 H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] -∇ 2 H 1/2 [Φ e k (-ϵ), θ, u -(k+1) ] ∆ RHEL u (k, ϵ) := -δ 2ϵ ∇ 3 H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] -∇ 3 H 1/2 [Φ e k (-ϵ), θ, u -(k+1) ] ∆ RHEL Φ (k, ϵ) := 1 2ϵ Σ x • (Φ e k (ϵ) -Φ e k (-ϵ))
, and consequently extend Theorem 3.1 in discrete time as well.
this section cite: []

Section: Theorem 3.2 (Informal).
Under mild assumptions on the Hamiltonian function and with (λ k ) k∈[0,K-1] , (g θ (k)) k∈[0,K-1] and (g u (k)) k∈[0,K-1] defined by Eq. ( 12), BPTT and RHEL are equivalent at all times in the sense that ∀k = 0, • • • , K:
λ k = lim ϵ→0 ∆ RHEL Φ (k, ϵ), g θ (k) = lim ϵ→0 ∆ RHEL θ (k, ϵ), g u (k) = lim ϵ→0 ∆ RHEL u (k, ϵ).
The full formal statement and proof are provided in Appendix A.2.3.
this section cite: []

Section: Learning stacks of HRUs via RHL chaining
Vectorized notations. Given a discrete vector field
(V k ) k∈ -K,0 , we denote V := (V ⊤ -K , • • • , V ⊤ 0 ) ⊤ ∈ R T ×d V and V := (V ⊤ 0 , • • • , V ⊤ -K ) ⊤ ∈ R T ×d V
the forward and backward trajectories associated with V . We define the vectorized operator M H,δ such that the dynamics of a single HRU unit as defined in Eq. ( 9) and the the nudged dynamics prescribed by RHEL as given in Eq. ( 13) rewrite more compactly as:
Φ = M H,δ [Φ, θ, u], Φ e = M H,δ [Φ e , θ, u] -ϵJ • ∇ Φ e L[Φ e ].
Learning Hamiltonian State Space Models (HSSMs) as multi-level optimization. We now consider hierarchical models reading as a composition of N HRU units of the form:
Φ (0) := u, ∀ℓ ∈ 0, N -1 : Φ (ℓ+1) = M (ℓ) H (ℓ) ,δ [Φ (ℓ+1) , θ (ℓ) , Φ (ℓ) ],(15)
where the trajectory of the (ℓ)-th HRU unit is fed as an input sequence into the (ℓ + 1)-th HRU unit (Fig. 2). We call such hierarchies of HRUs Hamiltonian State Space Models (HSSMs). The corresponding optimization problem reads: ℓ) ] ∀ℓ ∈ 0, N -1 (16) Alg. 1 prescribes an intuitive receipe to compute d θ L for the above optimization problem (Eq. ( 16)), chaining RHEL backward through HRUs. Namely, the echo dynamics of the top-most HRU read as Eq. ( 13) using the initial learning signal ∇ Φ L to nudge its trajectory. On top of estimating its parameter gradients, we also estimate its input gradients which are used to nudge the echo dynamics of the preceding HRU. This procedure is repeated until reaching the first HRU -see Fig. 2.
min θ L := 0 k=-K+1 ℓ[Φ (N ) k , k] s.t. Φ (ℓ+1) = M H (ℓ) ,δ [Φ (ℓ+1) , θ (ℓ) , Φ(
Algorithm 1 Recurrent Hamiltonian Echo Learning (RHEL) on a single HRU Inputs: Φ 0 (final state of the forward trajectory), ∆ Φ (incoming gradient), ϵ (nudging strength), δ (timestep) Outputs: ∆θ (parameter gradient estimate), ∆ u (input gradient estimate) 1:
∆θ ← 0 d θ 2: ∆ u := (∆ u -K , • • • , ∆ u0 ) ← 0 K×d Φ 3: Φ e ±ϵ ← Φ ⋆ 0 ± ϵΣ x • ∆ Φ,0 ▷ Compute Φ e ±ϵ in parallel 4: for k in 0, • • • , K do 5: Φ e ±ϵ ← M H,δ [Φ e ±ϵ , θ, u -(k+1) ] ± ϵΣ x • ∆ Φ,-(k+1) 6: ∆θ ← ∆θ -δ 2ϵ ∇ 2 H 1/2 [Φ e ϵ , θ, u -(k+1) ] -∇ 2 H 1/2 [Φ e -ϵ , θ, u -(k+1) ] ▷ Eq. (14
) 7: ∆ u -k ← -δ 2ϵ ∇ 3 H 1/2 [Φ e ϵ , θ, u -(k+1) ] -∇ 3 H 1/2 [Φ e
-ϵ , θ, u -(k+1) ] 8: end for 9: return ∆θ, ∆ u Theorem 3.3 (Informal). Given u an input sequence and a HSSM with layer-wise Hamiltonians {H (ℓ) } k∈[1,N ] , applying Alg. 1 recursively backward from the top-most HRU solves the multilevel optimization problem defined in Eq. ( 16) in the sense that:
∀ℓ = 0, • • • , N : d θ (ℓ) L = lim ϵ→0 ∆θ (ℓ) (ϵ)
The full formal statement and proof are provided in Appendix A.4.
this section cite: []

Section: Experiments
Foreword. We first introduce the two types HRUs at use, empirically assess the validity of Theorem 3.3 by statically comparing gradients computed by RHEL and BPTT on HSSMs made up of these two types of HRUs, then perform training experiments across classification and regression sequence tasks. Importantly, the goal of the proposed experiments is not to improve the SOTA performance on these tasks. Rather we want to show, on a given model satisfying the requirements of RHEL (i.e. being a HRU or a stack thereof), that RHEL maintains training performance with respect to BPTT.
Linear HRU block. Following recent work [29], we introduce the linear HRU block with the parametrization θ (ℓ) = {A (ℓ) , B (ℓ) , W (ℓ) } and Hamiltonian:
H (ℓ) [Φ (ℓ) , θ (ℓ) , Φ (ℓ-1) ] = 1 2 ∥π (ℓ) ∥ 2 + 1 2 ϕ (ℓ) ⊤ • A (ℓ) • ϕ (ℓ) -ϕ (ℓ) ⊤ • B (ℓ) • u (ℓ) , u (ℓ) = F [ϕ (ℓ-1) , W (ℓ) ],(17)
where
A ∈ R (d Φ /2)×(d Φ /2
) is assumed to be diagonal and to have positive entries, B (ℓ) ∈ R (d Φ /2)×du , α ∈ R, and F is the nonlinear spatial building block parametrized by W (ℓ) (see App. A.4.3). Under these assumptions, it can be shown that such HRUs have a bounded spectrum and yield HSSMs which are universal approximators of continuous and causal operators between time-series [29].
Nonlinear HRU block. To demonstrate the validity of our approach beyond linear HRUs, we also introduce a nonlinear HRU block [30] with θ (ℓ) = {A (ℓ) , W (ℓ) , B (ℓ) , b (ℓ) , α (ℓ) } and:
     H (ℓ) [Φ (ℓ) , θ (ℓ) , Φ (ℓ-1) ] = 1 2 ∥π (ℓ) ∥ 2 + α (ℓ) 2 ∥ϕ (ℓ) ∥ 2 + A (ℓ) -⊤ • log cosh A (ℓ) • ϕ (ℓ) + B (ℓ) • u (ℓ) + b (ℓ) , u (ℓ) = F [ϕ (ℓ-1) , W (ℓ) ],(18)
where
A ∈ R (d Φ /2)×(d Φ /2) is also assumed to be diagonal, B (ℓ) ∈ R (d Φ /2)×du , b (ℓ) ∈ R (d Φ /2)
, α ∈ R, and F is the nonlinear spatial building block parametrized by W (ℓ) (see App. A.4.5). It has been shown that these HRUs mitigate by design vanishing and exploding gradients [30]; we provide empirical validation in App. A.5.5..
this section cite: ['b28', 'b28', 'b29', 'b29']

Section: Static gradient comparison
As a sanity check of Theorem 3.3 and preamble to training experiments, we first check that RHEL gradients are computed correctly. Given a HSSM model, we randomly sample a tuple (u i , y i ) ∼ D from the SCP1 dataset (see App. A.5.1 for details on this dataset) and pass u i through the model. Given y, we then run BPTT and RHEL through the model and compare them in terms of cosine similarity and norm ratio of the resulting layer-wise parameter gradients. We run this experiment on a HSSM made up of six linear HRU blocks (which we call "linear HSSM") and another HSSM comprising six nonlinear HRU blocks (resp. "nonlinear HSSM") and obtain Fig. 4. We observe that in terms of these two comparison metrics, RHEL and BPTT parameter gradients are near-perfectly aligned for all parameters and across all HRU blocks, for both the linear and nonlinear HSSMs.
this section cite: []

Section: Training experiments
Classification. To further check our theoretical guarantees, we now perform training experiments with BPTT and RHEL across six multivariate sequence classification datasets with various sequence lengths (from ∼ 400 to ∼ 18k) and number of classes, on linear and nonlinear HSSMs (see App. A.5.1 for details on these datasets) and display our results in Table 1. This series of tasks was recently introduced [34] as a subset of the University of East Anglia (UEA) datasets [35] with the longest sequences for increased difficulty and recently used to benchmark the linear HSSM previously introduced [29]. We observe that models trained by RHEL almost match, on average across the datasets, those trained by BPTT in terms of resulting performance on the test dataset.
this section cite: ['b33', 'b34', 'b28']

Section: Regression & scalability.
Finally, to assess the applicability of RHEL beyond classification and scalability to longer sequences, we run training experiments on the PPG-DaLiA dataset, a multivariate  17)) and a nonlinear HSSM, Eq. ( 18)).
T a s k s W o r m s S C P 1 S C P 2 E t h a n o l H e a r b e a t M o t o r Seq. length 17,984 896 1,152 1,751 405 3,000 # classes 5 2 2 4 2 2 Avg Lin BPTT 78.3 ±7.5 86.4 ±1.8 63.9 ±7.3 29.9 ±0.6 73.9 ±0.6 48.1 ±5.7 63.4 ±3.9 RHEL 75.0 ±9.9 86.1 ±2.9 61.4 ±9.4 29.9 ±0.6 73.5 ±1.6 51.6 ±5.0 62.9 ±4.9 Nonlin BPTT 51.1 ±7.2 86.8 ±3.2 54.0 ±4.9 29.9 ±0.6 74.5 ±2.4 56.5 ±7.6 58.8 ±4.3 RHEL 50.6 ±6.7 85.6 ±4.4 54.0 ±2.0 29.9 ±0.6 73.9 ±4.3 53.0 ±5.7 57.8 ±4.0 Table 1: Test mean accuracy (%, higher is better) across five different seeds (± indicates standard deviation) using RHEL and BPTT, nonlinear and linear HSSMs, on six UEA time series classification datasets with various sequence length and number of classes.
time series regression dataset designed for heart rate prediction using data collected from a wrist-worn device [36]. With sequence length of ∼ 50k, this task is considered to be a difficult "long-range" benchmark [29]. We display in Table 2 the results obtained when training linear and nonlinear HSSMs with RHEL and BPTT on this dataset. Here again, we observe that on average and with both models, the performance of RHEL matches that of BPTT.
5 Discussion PPG-DaLiA Seq. length 49,920 Input/output dim. 6/1 Lin BPTT 9.1 ±1.1 RHEL 9.5 ±1.0 Nonlin BPTT 7.8 ±0.5 RHEL 8.4 ±0.5
Table 2: Test average mean-square error (×10 -2 , lower is better) across five different seeds (± indicates standard deviation) applying RHEL and BPTT to train nonlinear and linear HSSMs on the PPG-DaLiA dataset.
Limitations. We observe on Tables 1-2 that RHEL slightly underperforms BPTT, though this gap is statistically significant only for nonlinear HSSMs on regression tasks. This performance difference can be attributed to: i) the approximation bias introduced by finite nudging in our method, a known issue in related works [37], and ii) numerical precision that requires some careful selection of the nudging strength (see App. A.5.4).
"Physical computing"? Every logically irreversible computation, be it on a digital accelerator or any alternative substrate, is sustained by a physical process carried by wires and transistors which fundamentally generates heat [38]. Therefore the very notion of "physical computing" may sound pleonastic. However, digital hardware designs abstract core physics away to enable idealized boolean logic relying (among many other things) upon statelessness, unidirectionality, determinism and synchro-nization [39]. Compilation pipelines are then typically designed to be as hardware-agnostic and as general-purpose as possible [40]. In this generalistic computing paradigm, one of the most popular practices of "hardware-software" codesign is to write bespoke low-level kernels for specific workloads to mitigate their compute, memory costs and off-chip memory accesses [6,41,42], yet remaining largely physics-agnostic. Conversely, physical computing firstly targets Application-Specific Integrated Circuits (ASICs) rather than general-purpose machines [39]. Secondly, the aforementioned digital design constraints may be relaxed and the resulting analog physics leveraged. Supply voltages may be decreased and the resulting circuits become non-deterministic and be leveraged for stochastic algorithms [43]. Continuous-valued currents may be used for vector-matrix multiplication [44], matrix inversion [45], nearest neighbor search [46] within resistive systems. Constrained [47] or combinatorial [48] optimization problems may be embodied into a physical system and subsequently solved as the system settles to equilibrium-see more examples in recent review works [11,39].
Self-learning machines and Hamiltonian systems in ML. As a particular instantiation of physical computing, self-learning machines are physical embodiments of neural networks whose inference and gradient computation are, for instance, carried out by relaxing to equilibrium [49,50]. In this regard, RHEL extends the design of scalable self-learning machines beyond dissipative systems emulating implicit differentiation [51][52][53][54][55]. In comparison to these algorithms, RHEL crucially avoids prohibitively long simulation times due to the use of lengthy root-finding algorithms, extends their core mechanics to the broader domain of sequence modelling and possibly larger-scale tasks. Finally, on top of its direct connection with HEB [10], RHEL belongs to a large body Hamiltonian-based ML algorithms dedicated to physics-informed neural networks [56][57][58], memory-efficient models [30], generative models [59], sampling [60][61][62] and optimization [63] algorithms.
Forward-only learning. In mainstream ML, zeroth-order optimization (ZO) techniques are pursued for their memory efficiency compared to vanilla backprop with gradient checkpointing [64], which can be further enhanced with appropriate quantization schemes [65]. The relatively recent exploration of ZO techniques on Large Language Models (LLMs) finetuning [25] has spurred a revival of "forwardonly" techniques for memory-cheap gradient computation in LLMs on a variety of applications. However, the quest for forward-only algorithms in physical computing is not primarily motivated by memory efficiency. In our context, "forward-only" means that both the forward and backward passes obey the same physical principle, for instance obeying to the same Hamiltonian dynamics as required by RHEL.
Future work. The restriction of RHEL to non-dissipative systems limits model expressivity [29].
While inducing artificial dissipation through the use dissipative integrators [29] or by embedding a dissipative system within a larger conservative one [10] are convenient workarounds, there is ultimately a need for a temporal credit assignment algorithm tailored for truly dissipative systems. In Appendix A.3.1, we show that even when assuming that the time-reversed trajectory of a dissipative system is physically feasible, perturbations of this trajectory do not encode ASM error signals.
Additionally and in the spirit of HEB, an idealized version of RHEL should be entirely "black-box" as it still requires explicit Hamiltonian parameter gradients to implement its learning rule (Alg. 1). This could possibly be achieved for instance using homeostatic control via control knobs acting on the model parameters [66] -we sketch inside Appendix A. 3.2 what such a procedure would look like. Another exciting direction of work is to have RHEL operate online -it currently requires one forward pass and subsequent ones revisiting the inputs in reverse order. Finally, while HRU hidden units can be recomputed from their final state [30], it still requires storing the input sequence of the HRU so that memory gains are not evident within a stack of HRUs. Looking beyond analog physical systems, endowing RHEL with better memory efficiency and online mechanics could potentially yield a compelling alternative to BPTT on digital accelerators like GPUs [14].
Conclusion. While we simulated Hamiltonian dynamics alongside RHEL on GPUs, the greatest potential of RHEL lies within real Hamiltonian systems, namely on-chip photonic circuits [67], superconducting circuits [68], or spintronic systems [69]. A possible physical implementation of a HSSM would map its feedforward components onto digital circuits (running backprop) and its recurrent components onto analog circuits (running RHEL) [54] -see Remark 8 in Appendix. We hope this work will incentivize the search for alternative analog hardware capable of training sequence models with much higher energy efficiency, as well as alternative algorithms for digital hardware.
this section cite: ['b35', 'b28', 'b36', 'b37', 'b38', 'b39', 'b5', 'b40', 'b41', 'b38', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b10', 'b38', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b9', 'b55', 'b56', 'b57', 'b29', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b24', 'b28', 'b28', 'b9', 'b65', 'b2', 'b29', 'b13', 'b66', 'b67', 'b68', 'b53']

Section: A.1 Theoretical results in continuous time
Summary. In this section, we present all the results derived in continuous time. More precisely:
• We formally define our model (Def. A.1) and constrained optimization problem (Def. A.2) in continuous time.
• We state and prove our algorithm baseline, the continuous adjoint state method (ASM) for this constrained optimization problem (Theorem A.1). To ease the comparison of the continuous ASM with Recurrent Hamiltonian Echo Learning (RHEL) in continuous time, we state re-parametrized version of the continuous ASM where time is indexed backwards (Corollary A.1). Finally, we also state a variant of the continuous ASM when the loss function is only defined at the final timestep (Corollary A.2) to ease the comparison between the continuous ASM and Hamiltonian Echo Backprop (HEB, [10]).
• We introduce two technical results (Lemma A.1-A.2) which enable us to prove the timereversal invariance property of our model (Lemma A.3). We then show that the direct consequence of this property is the time-reversibily of our model upon momentum flipping (Corollary A.3), the key mechanics which fundamentally underpins our algorithm. All these intermediate results allow us to finally introduce RHEL in continuous time and prove its equivalence with the continuous ASM (Theorem A.2).
• Lastly, we connect HEB [10] with the continuous ASM when the loss is defined only at the final time step (Theorem A.4). We highlight some key differences between HEB and RHEL.
this section cite: ['b9', 'b9']

Section: A.1.1 Definitions & assumptions
Definition A.1 (Continuous Hamiltonian model). Given θ ∈ R d θ , T ∈ R ⋆ + and an input sequence t → u(t) ∈ R du [-T,0] , the continuous Hamiltonian model prediction t → Φ(t) ∈ R d Φ [-T,0] is, by definition, implicitly given as the solution of the following ODE: Φ(-T ) = x, ∀t ∈ [-T, 0] : ∂ t Φ(t) = J • ∇ Φ H[Φ(t), θ, u(t)], J := 0 I -I 0 .
We assume that: 1. H is time-reversal invariant:
∀Φ ∈ R d Φ , ∀θ ∈ R d θ , ∀u ∈ R du : H[Φ, θ, u] = H[Σ z •Φ, θ, u], Σ z := I 0 0 -I 2. Φ → H[Φ, •, •] is twice continuously differentiable, 3. θ → H[•, θ, •] is differentiable, 4. ∇ 2 1,2 H
exists and is continuous with respect to t,
5. u → H[•, •, u] is continuous, 6. Φ → ∇ 1 H[Φ, •, •] and Φ → ∇ 2,1 H[Φ, •, •] are Lipschitz continuous.
this section cite: []

Section: Remark 1.
While some of these assumptions will be used explicitly in our derivations, they are all needed to guarantee the existence of partial derivatives of s as an implicit function of x and θ through Eq. ( 1) and we refer to [28] for such claims given these assumptions.
Definition A.2 (Continuous constrained optimization optimization problem). Given a continuous Hamiltonian model (Def. A.1), we consider the following constrained optimization problem:
min θ L := 0 -T dtℓ[t, Φ(t), θ] s.t. ∀t ∈ [-T, 0] : ∂ t Φ(t) = J • ∇ Φ H[Φ(t), θ, u(t)],
where we assume that:
1. ℓ is time-reversal invariant:
∀Φ ∈ R d Φ , ∀θ ∈ R d θ , ∀t ∈ [-T, 0] : ℓ[t, Φ, θ] = ℓ[t, Σ z • Φ, θ] 2. t → ℓ[t, •, •] is continuous, 3. θ → ℓ[•, •, θ] is differentiable, 4. t → ∇ θ ℓ[t, •, •] is continuous. 5. Φ → ℓ[•, Φ, •] is twice differentiable, Remark 2.
Note that while we did not assume in the main part of this manuscript that ℓ depended on θ, we assume it in the appendix for the generality of our derivations.
this section cite: ['b27']

Section: A.1.2 Proof of the continuous adjoint state method (ASM)
Theorem A.1 (Continuous adjoint state method ( [28])). Given assumptions A.1-A.2, the gradients of L with respect to θ and x are given by:
d θ L = 0 -T g θ (t)dt, d x L = λ(0
)
with t → g θ ∈ R d θ [-T,0] defined as:
g θ (t) := ∇ θ ℓ[t, Φ(t), θ] + ∇ 2 Φ,θ H[Φ(t), θ, u(t)] • J ⊤ • λ(t)
∀t ∈ [-T, 0] and λ solving for the adjoint ODE:
λ(0) = 0 ∂ t λ(t) = -∇ 2 Φ H[Φ(t), θ, u(t)] • J ⊤ • λ(t) -∇ Φ ℓ[t, Φ(t), θ]
Proof of Theorem A.1. With slight adaptations, our proof mostly follows that of [32] and also assume the existence of the partial derivatives of Φ = Φ(t, x, θ) as an implicit function of t, x and θ -we defer to [28] for the proof of this claim.
We start off defining the Lagrangian associated with the constraint optimization problem:
L(Φ, λ, θ, u) := 0 -T dt ℓ[t, Φ(t, θ), θ] + λ ⊤ (t) • (J • ∇ Φ H[Φ(t, θ), θ, u(t)] -∂ t Φ(t, θ)) ,
where t → λ(t) ∈ R d Φ [-T,0] denotes the Lagrangian multiplier associated to the constraint.
this section cite: ['b27', 'b31', 'b27']

Section: Derivation of d θ L.
For readability, we emphasize the dependence of Φ on t and θ, as we will leverage the existence of its partial derivatives with respect to these variables. Then, given Assumptions A.1, the total derivative of the Lagrangian with respect to θ exists and reads:
d θ L = 0 -T dt ∂ θ Φ(t, θ) ⊤ • ∇ Φ ℓ[t, Φ(t, θ), θ] + ∇ θ ℓ[t, Φ(t, θ), θ] + 0 -T dt ∂ θ Φ(t, θ) ⊤ • ∇ 2 Φ H[Φ(t, θ), θ, u(t)] • J ⊤ + ∇ 2 θ,Φ H[Φ(t, θ), θ, u(t)] ⊤ • J ⊤ -∂ 2 θ,t Φ(t, θ) ⊤ • λ(t).
We can transform the last term of the integrand with ∂ θ,t Φ(t, θ) by applying Schwartz's theorem and and integration by parts as:
- 0 -T dt∂ θ,t Φ(t, θ) ⊤ • λ(t) = - 0 -T dt∂ t,θ Φ(t, θ) ⊤ • λ(t) = -∂ θ Φ(t, θ) ⊤ • λ(t) 0 -T + 0 -T dt∂ θ Φ(t, θ) ⊤ • ∂ t λ(t) = -∂ θ Φ(0, θ) ⊤ • λ(0) + 0 -T dt∂ θ Φ(t, θ) ⊤ • ∂ t λ(t)
where the contribution of the first term at t = -T vanishes because: Φ(-T, θ) = x ⇒ ∂ θ Φ(0, θ) = 0 Plugging this back into d θ L yields:
d θ L = -∂ θ Φ(0, θ) ⊤ • λ(0) + 0 -T dt∂ θ Φ(t, x, θ) ⊤ • ∇ Φ ℓ[t, Φ(t, θ), θ] + ∇ 2 Φ H[Φ(t, x, θ), θ] • J ⊤ • λ(t) + ∂ t λ(t) + 0 -T dt ∇ θ ℓ[t, Φ(t, θ), θ] + ∇ 2 Φ,θ H[Φ(t, x, θ), θ] • J ⊤ • λ(t)
Denoting Φ * the solution of the (primal) ODE and by defining λ * as the solution of the adjoint ODE:
∂ t λ * (t) = -∇ 2 Φ H[Φ * (t, θ), θ, u(t)] • J ⊤ • λ * (t) -∇ Φ ℓ[t, Φ * (t, θ), θ]
, λ * (0) = 0 we have:
d θ L := d θ 0 -T dtℓ[t, Φ(t, θ), θ] = d θ L(Φ * , λ * , θ) = 0 -T dt ∇ θ ℓ[t, Φ * (t, θ), θ] + ∇ 2 Φ,θ H[Φ * (t, x, θ), θ] • J ⊤ • λ * (t) Derivation of d x L.
Similarly, we can prove that:
d x L = ∂ x Φ(0, x) ⊤ • [∇ℓ[Φ(0, x)] -λ(0)] + 0 -T dt∂ x Φ(t, x) ⊤ • ∇ Φ ℓ[t, Φ(t, θ), θ] + ∇ 2 Φ H[Φ(t, x), θ, u(t)] • J ⊤ • λ(t) + ∂ t λ(t) + λ(0)
Using the same Φ * and λ * , we get d x L = λ(0).
this section cite: []

Section: Reparametrization of the continuous adjoint method.
To ease the comparison of the continuous adjoint state method with our algorithm, we slightly reparametrize the variables introduced in Theorem A.1. Corollary A.1. Under the same assumptions as Theorem A.1, the gradients of L with respect to θ and x are given by:
d θ L = T 0 g θ (t)dt, d x L = λ(T )
with t → g θ ∈ R d θ [0,T ] defined as: g θ (t) := ∇ θ ℓ[-t, Φ(-t), θ] + ∇ 2 Φ,θ H[Φ(-t), θ, u(-t)] • J ⊤ • λ(t) ∀t ∈ [0, T ] and λ solving for the adjoint ODE:
λ(0) = 0 ∂ t λ(t) = ∇ 2 Φ H[Φ(-t), θ, u(-t)] • J ⊤ • λ(t) + ∇ Φ ℓ[-t, Φ(-t), θ]
Proof of Corollary A.1. Immediately stems from Theorem A.1.
Edge case: loss at the final timestep only. The backward ODE can be differently parametrized when a loss is defined only at the last time step. We need this formulation for later convenience.
Corollary A.2. Under the same assumptions as Theorem A.1, and assuming:
ℓ[t, •, •] = 0 ∀t ∈ [0, T ), ℓ[T, •, •] := ℓ T ,
the gradients of ℓ T with respect to θ and x are given by:
d θ ℓ T [Φ(0), θ] = ∇ θ ℓ T [Φ(0), θ] + T 0 g θ (t)dt, d x L = λ(T )
with t → g θ ∈ R d θ [0,T ] defined as:
g θ (t) := ∇ 2 Φ,θ H[Φ(-t), θ, u(-t)] • J ⊤ • λ(t)
∀t ∈ [0, T ] and λ solving for the adjoint ODE:
λ(0) = ∇ Φ ℓ T [Φ(0), θ] ∂ t λ(t) = ∇ 2 Φ H[Φ(-t), θ, u(-t)] • J ⊤ • λ(t)
Proof of Corollary A.2. Starting from the Lagrangian:
L(Φ, λ, θ, u) := ℓ T [Φ(0, θ), θ] + 0 -T dt λ ⊤ (t) • (J • ∇ Φ H[Φ(t, θ), θ, u(t)] -∂ t Φ(t, θ)) ,
the proof reads in the exact same fashion as that of Theorem A.1.
this section cite: []

Section: A.1.3 Proof of Theorem 3.1
Technical Lemmas. We first introduce two technical Lemmas which will be needed for the derivation of our main result.
Lemma A.1 (Block-wise Pauli matrices and associated properties). Defining Σ x , Σ y , Σ z ∈ R d Φ ×d Φ as:
Σ x := 0 I d Φ /2 I d Φ /2 0 , Σ y := 0 -iI d Φ /2 iI d Φ /2 0 , Σ z := I d Φ /2 0 0 -I d Φ /2
where i denotes the imaginary unit, the following equalities hold:
1. J = iΣ y 2. Σ 2 x = Σ 2 y = Σ 2 z = I d Φ /2 , 3. Σ x • Σ y = iΣ z , Σ y • Σ z = iΣ x , Σ z • Σ x = iΣ y , 4. Σ i • Σ j = -Σ j • Σ i for any i ̸ = j ∈ {x, y, z}.
Proof of Lemma A.1. Because of the block-wise structure of Σ x , Σ y , Σ z , these equalities can be easily checked.
Lemma A.2. Under the assumptions of Def. A.1, the following equalities hold for all Φ, θ, u:
∇ Φ H[Φ, θ, u] = Σ z • ∇ Φ ⋆ H[Φ ⋆ , θ, u], ∇ 2 Φ H[Φ, θ, u] = Σ z • ∇ 2 Φ ⋆ H[Φ ⋆ , θ, u] • Σ z , ∇ Φ,θ H[Φ, θ, u] = ∇ Φ ⋆ ,θ H[Φ ⋆ , θ, u] • Σ z
Proof of Lemma A.2. The above equalities can be simply obtained by differentiating through the time-reversal invariance hypothesis -which is possible because of the differentiability of H with respect to Φ and θ -and using the chain rule. Namely, given some Φ, θ, u:
∇ Φ H[Φ, θ, u] = ∇ Φ (H[Φ ⋆ , θ, u]) = (∂ Φ Φ ⋆ ) ⊤ • ∇ Φ ⋆ H[Φ ⋆ , θ, u] = Σ ⊤ z • ∇ Φ ⋆ H[Φ ⋆ , θ, u] = Σ z • ∇ Φ ⋆ H[Φ ⋆ , θ, u], since Φ ⋆ := Σ z • Φ.
The other equalities are derived in the same way.
Time-reversal invariance. We highlight here how the assumption H[Φ, •, •] = H[Σ z • Φ, •, •] given inside Def. A.1 entails time-reversal invariance of the dynamics -up to time-reversal the input sequence.
Lemma A.3 (Time-reversal invariance of the dynamics). Under the assumptions given in Def. A.1, if Φ the solution of the ODE:
∂ t Φ(t) = J • ∇ Φ H[Φ(t), θ, u(t)], the function Φ ⋆ : t → Σ z • Φ(-t) = [ϕ ⊤ (-t)
, -π ⊤ (-t)] ⊤ is solution of the same ODE with the time-reversed input sequence t → u(t) := u(-t).
Proof. Let t ∈ [-T, 0]. We have:
∂ t Φ ⋆ (t) = -Σ z • ∂ t Φ(-t) (Def. of Φ ⋆ and chain rule) = -Σ z • J • ∇ Φ H[Φ(-t), θ, u(-t)]
(by assumption) = +J • Σ z • ∇ Φ H[Φ(-t), θ, u(-t)]
(Lemma A.1)
= J • Σ 2 z • ∇ Φ ⋆ H[Φ ⋆ (-t), θ, u(-t)] (Lemma A.2) = J • ∇ Φ ⋆ H[ Φ ⋆ (t)
, θ, u(-t)] (Lemma A.1) Time reversibility. A direct consequence of the time-reversal of the dynamics under consideration (Lemma A.3) is time reversibility: upon flipping the momentum of Φ at time t = 0 (π(0) ← -π(0)) and presenting the input sequence in reversed order, the system evolves backward to its initial state.
this section cite: []

Section: Corollary A.3 (Reversibility of the dynamics).
Under the same assumptions as Lemma A.3, we define Φ e as the solution of the ODE:
Φ e (0) = Φ ⋆ (0), ∂ t Φ e (t) = J • ∇ Φ e H[Φ e (t)
, θ, u(-t)] ∀t ∈ [0, T ].
Then: ∀t ∈ [0, T ] : Φ e (t) = Φ ⋆ (t)
Proof. Φ e and Φ ⋆ satisfy: i) the same initial conditions ( Φ ⋆ (0) = Φ ⋆ (0) = Φ e (0)), ii) the same ODE (Lemma A.3), therefore by unicity of the solution of the ODE, they are equal at all time over the domain of definition of Φ.
this section cite: []

Section: Main result.
We are now ready to state and demonstrate our main result in continuous time.
this section cite: []

Section: Theorem A.2 (Equivalence between RHEL and the continuous ASM).
Under the assumptions of Def. A.1 and Def. A.2, let Φ be the solution of the ODE for t ∈ [-T, 0]:
Φ(-T ) = x, ∂ t Φ(t) = J • ∇ Φ H[Φ(t), θ, u(t)].
Given Φ, let Φ e be defined as the solution of the other ODE for t ∈ [0, T ]:
Φ e (0) = Φ ⋆ (0), ∂ t Φ e (t, ϵ) = J ∇ Φ e H[Φ e (t, ϵ), θ, u(-t)] -ϵJ ∇ Φ e ℓ[-t, Φ e (t, ϵ), θ].
Defining:
∆ RHEL θ (t, ϵ) := ∇ θ ℓ[-t, Φ e (t, ϵ), θ] + 1 2ϵ (∇ θ H[Φ e (t, ϵ), θ, u(-t)] -∇ θ H[Φ e (t, -ϵ), θ, u(-t)]) , ∆ RHEL Φ (t, ϵ) := 1 2ϵ Σ x • (Φ e (t, ϵ) -Φ e (t, -ϵ)) , we have: ∀t ∈ [0, T ], λ(t) = lim ϵ→0 ∆ RHEL Φ (t, ϵ), g θ (t) = lim ϵ→0 ∆ RHEL θ (t, ϵ)
where λ and g θ are defined in Corollary A.1.
Proof. Defining ∆ RHEL Φ (t) := lim ϵ→0 ∆ RHEL Φ (t, ϵ) and ∆ RHEL θ (t) := lim ϵ→0 ∆ RHEL θ (t, ϵ), note that:
∆ RHEL Φ (t) = Σ x • ∂ ϵ Φ e (t, ϵ)| ϵ=0 ∆ RHEL θ (t) = ∇ θ ℓ[-t, Φ e (t, 0), θ] + ∂ ϵ (∇ θ H[Φ e (t, ϵ), θ, u(-t)]) | ϵ=0 Derivation of λ(t) = lim ϵ→0 ∆ RHEL Φ (t, ϵ).
Given t ∈ [0, T ], differentiating the ODE satisfied by Φ e with respect to ϵ at ϵ = 0 yields:
∂ t (∂ ϵ Φ e (t, ϵ)| ϵ=0 ) = ∂ ϵ (∂ t Φ e (t, ϵ))| ϵ=0 (Schwartz Theorem) = ∂ ϵ (J • ∇ Φ e H[Φ e (t, ϵ), θ, u(t)] -ϵJ ∇ Φ e ℓ[t, Φ e (t, ϵ), θ])| ϵ=0 = J • ∇ 2 Φ e H[Φ e (t, 0)] • ∂ ϵ Φ e (t, ϵ)| ϵ=0 -J ∇ Φ e ℓ[t, Φ e (t, 0), θ]. By Lemma A.3: Φ e (t, 0) = Φ ⋆ (t) ∀t ∈ [0, T ], therefore: ∂ t (∂ ϵ Φ e (t, ϵ)| ϵ=0 ) = J • ∇ 2 Φ ⋆ H[Φ ⋆ (-t)] • ∂ ϵ Φ e (t, ϵ)| ϵ=0 -J ∇ Φ ⋆ ℓ[t, Φ ⋆ (-t), θ] = J • Σ z • ∇ 2 Φ H[Φ(-t, 0)] • Σ z • ∂ ϵ Φ e (t, ϵ)| ϵ=0 -J • Σ z • ∇ Φ ℓ[t, Φ(-t), θ] (Lemma A.2)
Additionally, note that we have by Lemma A.1:
J • Σ z = -Σ x , Σ z = J • Σ x ,
so that:
∂ t (∂ ϵ Φ e (t, ϵ)| ϵ=0 ) = -Σ x •∇ 2 Φ H[Φ(-t, 0)]•(J •Σ x )•∂ ϵ Φ e (t, ϵ)| ϵ=0 +Σ x •∇ Φ ℓ[t, Φ(-t), θ] (19
)
Left multiplying Eq. ( 19) on both sides by Σ x yields:
∂ t (Σ x • ∂ ϵ Φ e (t, ϵ)| ϵ=0 ) = -Σ 2 x • ∇ 2 Φ H[Φ(-t, 0)] • J • (Σ x • ∂ ϵ Φ e (t, ϵ)| ϵ=0 ) + Σ 2 x • ∇ Φ ℓ[t, Φ(-t), θ] = -∇ 2 Φ H[Φ(-t, 0)] • J • (Σ x • ∂ ϵ Φ e (t, ϵ)| ϵ=0 ) + ∇ Φ ℓ[t, Φ(-t), θ] (Lemma A.1) = ∇ 2 Φ H[Φ(-t, 0)] • J ⊤ • (Σ x • ∂ ϵ Φ e (t, ϵ)| ϵ=0 ) + ∇ Φ ℓ[t, Φ(-t), θ] (J ⊤ = -J )
Finally, note that because Φ e (0) = Φ ⋆ does not depend on ϵ, we have that:
∂ t (Σ x • ∂ ϵ Φ e (0, ϵ)| ϵ=0 ) = 0,
so that all in all, ∆ RHEL Φ satisfies:
∆ RHEL Φ (0) = 0 ∂ t ∆ RHEL Φ (t) = ∇ 2 Φ H[Φ(-t), θ, u(-t)] • J ⊤ • ∆ RHEL Φ (t) + ∇ Φ ℓ[-t, Φ(-t), θ]
Therefore ∆ RHEL Φ and λ (as defined in Corollary A.1) satisfy the same initial conditions and the same ODE, therefore they are equal at all times.
Derivation of g θ (t) = lim ϵ→0 ∆ RHEL θ (t, ϵ). Note that by Lemma A.3 and time-reversal invariance of ℓ:
∆ RHEL θ (t) = ∇ θ ℓ[-t, Φ ⋆ (-t, 0), θ] + ∂ ϵ (∇ θ H[Φ e (t, ϵ), θ, u(-t)]) | ϵ=0 = ∇ θ ℓ[-t, Φ(-t, 0), θ] + ∂ ϵ (∇ θ H[Φ e (t, ϵ), θ, u(-t)]) | ϵ=0
As the first term of ∆ RHEL (t) and g θ (t) coincide, the remainder of the derivation focuses on the second term of ∆ RHEL θ (t). Given t ∈ [0, T ], we have:
∇ 2 Φ,θ H[Φ(-t), θ, u(t)] • J ⊤ • λ(t) = ∇ 2 Φ e ,θ H[Φ e (t, 0), θ, u(t)] • Σ z • J ⊤ • λ(t) (Lemma A.2) = -∇ 2 Φ e ,θ H[Φ e (t, 0), θ, u(t)] • Σ z • iΣ y • λ(t) (J = iΣ y ) = +∇ 2 Φ e ,θ H[Φ e (t, 0), θ, u(t)] • iΣ y • Σ z • λ(t) (Lemma A.1) = -∇ 2 Φ e ,θ H[Φ e (t, 0), θ, u(t)] • Σ x • λ(t) (Lemma A.1) = -∇ 2 Φ e ,θ H[Φ e (t, 0), θ, u(t)] • Σ 2 x • ∂ ϵ Φ e (t, ϵ)| ϵ=0 (λ = ∆ RHEL Φ ) = -∇ 2 Φ e ,θ H[Φ e (t, 0), θ, u(t)] • ∂ ϵ Φ e (t, ϵ)| ϵ=0 (Lemma A.1) = -∂ ϵ (∇ θ H[Φ e (t, ϵ), θ, u(t)]) | ϵ=0 , which finishes to prove g θ (t) = ∆ RHEL θ (t) for t ∈ [0, T ].
this section cite: []

Section: A.1.4 Connection to Hamiltonian Echo Backpropagation (HEB)
Remark 3. The above setup and implementation of RHEL is not exactly that of Hamiltonian Echo Backprop (HEB, [10]). In particular:
• the loss function in HEB is only defined at the final time step,
• the interaction with ℓ does not happen simultaneously with H,
• finally, we would like to recover the HEB formula giving the gradient estimate of the loss with respect to the initial state of the neurons.
In the following corollary, we make slight algorithmic adjustments to match the seminal HEB implementation as much as possible while preserving the generality of the sequence modelling setting, i.e. dependence of H with θ and u. Note that in the seminal HEB work, H does not depend on a static set of parameters θ nor on an input sequence. u.
Corollary A.4. Under the assumptions of Def. A.1 and Def. A.2, and assuming additionally:
ℓ[t, •, •] = 0 ∀t ∈ [0, T ), ℓ[T, •, •] := ℓ T ,
let Φ be the solution of the ODE, for t ∈ [-T, 0]:
Φ(-T ) = x, ∂ t Φ(t) = J • ∇ Φ H[Φ(t), θ, u(t)],
and the solution of another ODE, for t ∈ [0, ϵ]:
∂ t Φ(t) = J ∇ Φ ℓ T [Φ(t), θ].
Let Φ e be the solution of the following ODE, for t ∈ [ϵ, T ]:
Φ e (0, ϵ) = Φ(ϵ) ⋆ , ∂ t Φ e (t, ϵ) = J ∇ Φ e H[Φ e (t, ϵ), θ, u(-t)],
Defining:
∆ RHEL θ (t, ϵ) := 1 2ϵ (∇ θ H[Φ e (t, ϵ), θ, u(-t)] -∇ θ H[Φ e (t, -ϵ), θ, u(-t)]) , ∆ RHEL Φ (t, ϵ) := 1 2ϵ Σ x • (Φ e (t, ϵ) -Φ e (t, -ϵ)) , we have: ∀t ∈ [0, T ], λ(t) = lim ϵ→0 ∆ RHEL Φ (t, ϵ), g θ (t) = lim ϵ→0 ∆ RHEL θ (t, ϵ)
where λ and g θ are defined in Corollary A.2. In particular:
-iϵd w x ⋆ ℓ T [Φ(0), θ] = Φ e (T, ϵ) ⋆ -Φ(-T ) + O(ϵ 2 ) where d w
x ⋆ ≡ d Φ denotes the total Wirtinger derivative with respect to x ⋆ and i the imaginary unit.
Proof of Corollary A.4. The derivation is almost exactly similar to that of Theorem 3.1 with two key differences:
• the version of the continuous ASM against which this version of RHEL is compared is different (Corollary A.2),
• the interaction of Φ with ℓ and H do not happen simultaneously but on disjoint intervals.
We will simply show that the interaction with ℓ and conjugation Φ → Φ ⋆ yields the correct initial conditions and defer to the proof of Theorem 3.1 for the remainder. We will also use the same notations and denote
∆ RHEL Φ (t) := lim ϵ→0 ∆ RHEL Φ (t, ϵ), ∆ RHEL θ (t) := lim ϵ→0 ∆ RHEL θ (t, ϵ). Derivation of λ(t) = lim ϵ→0 ∆ RHEL Φ (t, ϵ).
Integrating the ODE satisfied by Φ between 0 and T yields:
Φ(ϵ) = Φ(0) + ϵ 0 dtJ • ∇ Φ ℓ[Φ(t), θ] = Φ(0) + ϵJ • ∇ Φ ℓ[Φ(0), θ] + O(ϵ 2 )
Therefore, the initial state of Φ e can be written as:
Φ e (ϵ) = Φ ⋆ (0) + ϵΣ z • J • ∇ Φ ℓ[Φ(0), θ] + O(ϵ 2 ) = Φ ⋆ (0) + ϵΣ x • ∇ Φ ℓ[Φ(0), θ] + O(ϵ 2 ) (Lemma A.1).
By differentiating the last equality with respect to ϵ at ϵ = 0, we obtain:
∆ RHEL Φ (0) = ∇ Φ ℓ[Φ(0), θ].
Proceeding exactly as in the proof of Theorem A.2, we obtain that:
∆ RHEL Φ (0) = ∇ Φ ℓ[Φ(0), θ], ∂ t ∆ RHEL Φ (t) = ∇ 2 Φ H[Φ(-t), θ, u(-t)] • J ⊤ • ∆ RHEL Φ (t)
Therefore ∆ RHEL Φ and λ (as defined in Corollary A.2) satisfy the same initial conditions and the same ODE, therefore they are equal at all times.
Derivation of g θ (t) = lim ϵ→0 ∆ RHEL θ (t, ϵ). See proof of Theorem A.2.
Connection to HEB formula. In particular, we have:
d x ℓ T [Φ(0), θ] = λ(T ) = Σ x • ∂ ϵ (Φ(T, ϵ) | ϵ=0 = 1 ϵ Σ x (Φ e (T, ϵ) -Φ e (T, 0)) + O(ϵ) = 1 ϵ Σ x (Φ e (T, ϵ) -Φ ⋆ (-T )) + O(ϵ) (Lemma A.3) = - i ϵ Σ y • Σ z • (Φ e (T, ϵ) -Φ ⋆ (-T )) + O(ϵ) (Lemma A.1) = - i ϵ Σ y • (Φ e (T, ϵ) ⋆ -Φ(-T )) + O(ϵ)
Left multiplying on both sides by iϵΣ y and noticing that id w Φ ⋆ ≡ -iΣ y • d Φ , we finally obtain:
-iϵd w x ⋆ ℓ T [Φ(0), θ] = Φ e (T, ϵ) ⋆ -Φ(-T ) + O(ϵ 2 )
A.2 Theoretical results in discrete time Summary. In this section, we introduce all the results derived in discrete time. More precisely:
• We first formally define Hamiltonian Recurrent Units (HRUs, Definition A.3). HRUs can be regarded as the discrete-time counterpart of the continuous model introduced in the previous section (Definition A.1), namely as an explicit and symplectic integrator of the continuous Hamiltonian model which preserves the time-reversal invariance and timereversibility properties in discrete time. We also introduce the constrained optimization problem naturally associated with HRUs (Definition A.4), which is the discrete time counterpart of the constrained continuous optimization problem introduced in the previous section (Definition A.2).
• We then formally define Hamiltonian State Space Models (HSSMs) as stacks of HRUs (Definition A.5) and the multilevel constrained optimization problem which is naturally associated to these models (Definition A.6).
• We state and prove our algorithmic baseline, Backpropagation Through Time (BPTT), through the lens of the Lagrangian formalism to establish a clear connection with the continuous ASM. We first introduce and derive BPTT in its general form (Theorem A.3) and then apply it more specifically to a HRU as defined in Definition A.3 (Corollary A.5).
• As we did in continuous time, we introduce a series of technical Lemmas needed to extend RHEL in discrete time. We first demonstrate the time-reversibility of HRUs on a single time step (Lemma A.4), which then enables us to extend the time reversibility property derived in continuous time (Corollary A.3) to discrete time (Corollary A.6). After introducing one last technical result (Lemma A.5), we then state and prove RHEL in discrete time when applied to HRUs (Corollary A.7). As the algorithm prescribed by Corollary A.7 includes solving an implicit equation, we finally introduce a slight practical (i.e. fully explicit) variant of RHEL in discrete time (Corollary A.8).
• Lastly, we show how to estimate gradients end-to-end in HSSMs by using RHEL-chaining (Theorem A.4). We also highlight that in practice, when using feedforward transformations across HRUs, the algorithm prescribed by Theorem A.4 implicitly requires to chain RHEL through HRUs and automatic differentiation through these feedforward transformations (Remark 8). This remark fundamentally underpins the actual algorithmic implementation of RHEL which was used throughout our experiments.
this section cite: ['b9']

Section: A.2.1 Definitions & assumptions
Definition A.3 (Hamiltonian Recurrent Unit). Given θ ∈ R d θ , K ∈ N ⋆ and an input sequence (u k ) k∈[-K,0] ∈ R du K , the Hamiltonian Recurrent Unit (HRU) prediction is given by:
Φ k+1 = M H,δ [Φ k , θ, u k ] ∀k = -K • • • -1,
with H := T + V and:
M H,δ := M T,δ/2 • M V,δ • M T,δ/2 , M T,δ := Φ + δJ • ∇ Φ T, M V,δ := Φ + δJ • ∇ Φ V
We assume that: 1. H is separable, i.e. V and T only depend on ϕ and π respectively:
V [Φ, θ, u] = V [ϕ, θ, u], T [Φ, θ, u] = T [π, θ, u]
2. T and V are time-reversal invariant:
∀Φ ∈ R d Φ , ∀θ ∈ R d θ , ∀u ∈ R du : T [Φ, θ, u] = T [Σ z • Φ, θ, u], V [Φ, θ, u] = V [Σ z • Φ, θ, u]
3. T and V are twice differentiable with respect to Φ, θ and u.
Remark 4. Note that M H,δ is simply a Leapfrog integrator associated with H. We justify each of our design choices below:
• 3 steps-parametrization. We write the Leapfrog integrator in a three-steps fashion to yield a reversible integrator.
• Separability of the Hamiltonian. In the case where ϕ and ϕ can be separated out in the Hamiltonian function, the Leapfrog integrator becomes explicit [33].
• T and V as functions of Φ. Although T and V only depend on π and ϕ respectively, we choose to define them as functions of Φ so that the proof of RHEL in the continuous case seamlessly translates to the discrete case.
Definition A.4 (Constrained optimization optimization problem in discrete time). Given a continuous Hamiltonian model (Def. A.1), we consider the following constrained optimization problem:
min θ L := 0 k=-K+1 ℓ[Φ k , k] s.t. Φ k+1 = M H,δ [Φ k , θ, u k ] ∀k = -K • • • -1
where we assume that: 1. ℓ is time-reversal invariant:
∀Φ ∈ R d Φ , ∀θ ∈ R d θ , ∀k = -K, • • • , 0 : ℓ k [Φ, θ] = ℓ k [Σ z • Φ, θ]
2. ℓ is twice differentiable with respect to Φ and θ.
Definition A.5 (Hamiltonian State Space Models). Given (θ (1) , • • • , θ (N ) ) ∈ R d θ N , K ∈ N ⋆ and an input sequence (u k ) k∈[-K,0] ∈ R du K , a Hamiltonian State Space Model (HSSM) is defined as the composition of HRUs defined in Def. A.3 as:
Φ (0) := u, ∀ℓ ∈ 0, N -1 , ∀k ∈ -K, 0 : Φ (ℓ+1) k+1 = M (ℓ) H (ℓ) ,δ [Φ (ℓ+1) k , θ (ℓ) , Φ (ℓ) k ],
or in a vectorized fashion as:
Φ (0) := u, ∀ℓ ∈ 0, N -1 : Φ (ℓ+1) = M (ℓ) H (ℓ) ,δ [Φ (ℓ+1) , θ (ℓ) , Φ (ℓ) ],
Definition A.6 (Multilevel optimization problem in discrete time). Given a HSSM (Def. A.5), we consider the following constrained optimization problem:
min θ L := 0 k=-K+1 ℓ[Φ k , k] s.t. Φ (0) := u, ∀ℓ ∈ 0, N -1 : Φ (ℓ+1) = M (ℓ) H (ℓ) ,δ Φ (ℓ+1) , θ (ℓ) , Φ(ℓ)
where we assume that ℓ satisfies the same assumptions as in Def. A.4.
this section cite: ['b32']

Section: A.2.2 Backpropagation Through Time (BPTT)
General form. We first state and prove Backpropagation Through Time (BPTT) for any integrator M H,δ .
this section cite: []

Section: Theorem A.3 (Backpropagation Through Time (BPTT)).
Given assumptions in Def. A.3-A.4, the gradients of the loss with respect to the parameters θ and the inputs u -k are given by:
d θ L = K-1 k=0 g θ (k), d u -(k+1) L = g u (k) ∀k ∈ 0, K -1 , with: g θ (k) = ∇ 2 ℓ[Φ -k , θ, -k] + ∂ 2 M H,δ [Φ -(k+1) , θ, u -(k+1) ] ⊤ • λ k g u (k) = ∂ 3 M H,δ [Φ -(k+1) , θ, u -(k+1) ] ⊤ • λ k ,
and where (λ k ) satisfy the following recursion relationship:
λ 0 = ∇ 1 ℓ[Φ 0 , 0], λ k+1 = ∂ 1 M H,δ (Φ -(k+1) , θ, u -(k+1) ) ⊤ • λ k + ∇ 1 ℓ[Φ -(k+1) , -(k + 1)] ∀k = 0, • • • , K -1
Proof of Theorem A.3. BPTT can classically be derived through the application of the "chain rule" backward through the inference computational graph defined in Def. A.3. Another useful viewpoint though, which directly connects BPTT as the discrete counterpart of the continuous ASM and will be useful later in the appendix, is to derive it through the method of Lagrangian multipliers. Namely, the Lagrangian associated to the constrained optimization problem in Def. A.4 reads as:
L(Φ, λ, θ, u) = K-1 k=0 ℓ[Φ -k , θ, -k] + λ ⊤ k • M H,δ Φ -(k+1) , θ, u -(k+1) -Φ -k
Extremizing L with respect to Φ and λ yield Φ k, * and λ k, * :
∀k = 0, • • • , K -1 : ∂ λ k L(Φ * , λ * , θ, u) = M H,δ Φ -(k+1), * , θ, u -k -Φ -k, * = 0, ∂ Φ0 L(Φ * , λ * , θ, u) = ∇ 1 ℓ[Φ 0, * , θ, 0] -λ 0, * = 0 ∀k = 1, • • • , K -1 : ∂ Φ -k L(Φ * , λ * , θ, u) = ∇ 1 ℓ[Φ -k , θ, -k] + ∂ 1 M H,δ [Φ -k , θ, u -k ] ⊤ • λ k-1 -λ k = 0,
Finally, the total derivative of L with respect to θ reads as:
d θ L = d θ L(Φ * , λ * , θ, u) = ∂ θ L(Φ * , λ * , θ, u) + ∂ θ Φ ⊤ * • ∂ Φ L(Φ * , λ * , θ, u) =0 +∂ θ λ ⊤ * • ∂ λ L(Φ * , λ * , θ, u) =0 = K-1 k=0 ∇ 2 ℓ[Φ -k , θ, -k] + ∂ 2 M H,δ [Φ -(k+1) , θ, u -(k+1) ] ⊤ • λ k
The total derivative of L with respect to u -k is derived in the exact same fashion.
Remark 5. Note that using the vectorized notations introduced in subsection 3.3, the Lagrangian of the constrained optimization problem defined in Def. A.4 re-writes:
L = 1 ⊤ • ℓ[ Φ, θ] + Tr M H,δ [ Φ, θ, u] -Φ • λ ⊤
with Tr denoting the trace matrix operator and:
1 :=    1 . . . 1    ∈ R K×1 , ℓ[ Φ, θ] :=    ℓ[Φ 0 , θ, 0] . . . ℓ[Φ -(K-1) , θ, -(K -1)]    ∈ R K×1 , M H,δ [ Φ, θ, u] :=    M H,δ [Φ 0 , θ, u] . . . M H,δ [Φ -(K-1) , θ, u]    ∈ R K×d Φ
Detailed BPTT. For the needs of our derivation of RHEL in discrete time, we now introduce a finer-grained version of BPTT given model assumptions given in Def. A.3.
this section cite: []

Section: Corollary A.5 (Detailed BPTT).
Given assumptions in Def. A.3-A.4, the gradients of the loss with respect to the parameters θ and the inputs u -k are given by:
d θ L = K-1 k=0 g θ (k), d u -k L = g u (k),
with:
g θ (k) := ∇ 2 ℓ[Φ -k , θ, -k] + δ 2 ∇ 2 1,2 T [Φ -(k+1/3) , θ, u -(k+1) ] • J ⊤ • λ k + δ∇ 2 1,2 V [Φ -(k+2/3) , θ, u -(k+1) ] • J ⊤ • λ k+1/3 + δ 2 ∇ 2 1,2 T [Φ -(k+1) , θ, u -(k+1) ] • J ⊤ • λ k+2/3 , g u (k) := δ 2 ∇ 2 1,3 T [Φ -(k+1/3) , θ, u -(k+1) ] • J ⊤ • λ k + δ∇ 2 1,3 V [Φ -(k+2/3) , θ, u -(k+1) ] • J ⊤ • λ k+1/3 + δ 2 ∇ 2 1,3 T [Φ -(k+1) , θ, u -(k+1) ] • J ⊤ • λ k+2/3 ,
and where (λ k ) satisfy the following recursion relationship, with λ 0 = ∇ Φ ℓ[Φ 0 ] and ∀k ∈ [0, K -1]:
   λ k+1/3 = λ k + δ 2 ∇ 2 1 T [Φ -(k+1/3) , θ, u -(k+1) ] • J ⊤ • λ k λ k+2/3 = λ k+1/3 + δ∇ 2 1 V [Φ -(k+2/3) , θ, u -(k+1) ] • J ⊤ • λ k+1/3 λ k+1 = λ k+2/3 + δ 2 ∇ 2 1 T [Φ -(k+1) , θ, u -(k+1) ] • J ⊤ • λ k+2/3 + ∇ 1 ℓ[Φ -(k+1) , θ]
Proof of Corollary A.5. Direct application of Theorem A.3 with the inference computational graph details defined inside Def. A.3.
this section cite: []

Section: A.2.3 Proof of Theorem 3.2
Time reversibility in discrete time. We first derive the discrete counterpart of Lemma A.3 as a technical pre-requisite for the extension of RHEL to the discrete-time setting. Lemma A.4 (Reversibility of M H,δ ). Under the assumptions of Def. A.3-A.4:
∀Φ k ∈ R d Φ , ∀θ ∈ R d θ , ∀u ∈ R du : Φ k+1 = M H,δ [Φ k , θ, u] ⇒ M H,δ [Φ ⋆ k+1 , θ, u] = Φ ⋆ k
Proof of Lemma A.4. Let Φ k , Φ k+1 ∈ R d Φ be such that:
Φ k+1 = M H,δ [Φ k , θ, u]
which rewrites, given Def. A.3:
M H,δ :              ϕ k+1/3 = ϕ k + δ 2 ∇ π T [π k , θ, u] π k+1/3 = π k ϕ k+2/3 = ϕ k+1/3 π k+2/3 = π k+1/3 -δ∇ ϕ V [ϕ k+1/3 , θ, u] ϕ k+1 = ϕ k+2/3 + δ 2 ∇ π T [π k+2/3 , θ, u] π k+1 = π k+2/3 (20
)
It becomes apparent from Eq. ( 20) that M H,δ is invertible with respect to its first argument and that inverting M H,δ amounts to change δ to -δ:
M -1 H,δ :              ϕ k+2/3 = ϕ k+1 -δ 2 ∇ π T [π k+1 , θ, u] π k+2/3 = π k+1 ϕ k+1/3 = ϕ k+2/3 π k+1/3 = π k+2/3 + δ∇ ϕ V [ϕ k+2/3 , θ, u] ϕ k = ϕ k+1/3 -δ 2 ∇ π T [π k+1/3 , θ, u] π k = π k+1/3(21)
and therefore:
Φ k = M -1 H,δ [Φ k+1 , θ, u] = M H,-δ [Φ k+1 , θ, u],
Denoting π ⋆ := -π, note that by time-reversal invariance hypothesis in Def. A.3 and Lemma A.2, we have that ∇ π T [π, θ, u] = -∇ π ⋆ T [π ⋆ , θ, u]. Therefore, Eq. ( 21) rewrites as:
               ϕ k+2/3 = ϕ k+1 + δ 2 ∇ π ⋆ T [π ⋆ k+1 , θ, u] π ⋆ k+2/3 = π ⋆ k+1 ϕ k+1/3 = ϕ k+2/3 π ⋆ k+1/3 = π ⋆ k+2/3 -δ∇ ϕ V [ϕ k+2/3 , θ, u] ϕ k = ϕ k+1/3 + δ 2 ∇ π ⋆ T [π ⋆ k+1/3 , θ, u] π ⋆ k = π ⋆ k+1/3 ,(22)
where equations bearing on π have been multiplied on both sides by -1. Finally note that Eq. ( 22) simply rewrites as:
M H,δ [Φ ⋆ k+1 , θ, u] = Φ ⋆ k
Corollary A.6. Under the assumptions of Def. A.3, if Φ satisfies the following recursive equations:
Φ -K = x, ∀k = -K, • • • , -1 : Φ k+1 = M H,δ [Φ k , θ, u k ]
and Φ e is subsequently defined as:
Φ e 0 = Φ ⋆ 0 , ∀t = 0, • • • , K -1 : Φ e k+1 = M H,δ [Φ e k , θ, u -(k+1) ] Then: ∀t = 0, • • • , K -1 : Φ e k = Φ ⋆ -k
Proof of Corollary A.6. This result is immediately obtained by iterating Lemma A.4 over the whole trajectory.
A technical pre-requisite. Finally, we need one last technical Lemma to handle subtleties pertaining to Jacobian evaluation which only occur in discrete time.
Lemma A.5. Under the assumptions of Def. A.3, if we have, for some θ ∈ R d θ and u ∈ R du :
Φ k+1 = M H,δ [Φ k , θ, u], then:    T [Φ k+1/3 , θ, u] = T [Φ k , θ, u], V [Φ k+2/3 , θ, u] = V [Φ k+1/3 , θ, u] T [Φ k+1 , θ, u] = T [Φ k+2/3 , θ, u]
Proof. This can be seen by simply writing M H,δ explicitly for ϕ and π:
M H,δ :              ϕ k+1/3 = ϕ k + δ 2 ∇ π T [π k , θ, u] π k+1/3 = π k ϕ k+2/3 = ϕ k+1/3 π k+2/3 = π k+1/3 -δ∇ ϕ V [ϕ k+1/3 , θ, u] ϕ k+1 = ϕ k+2/3 + δ 2 ∇ π T [π k+2/3 , θ, u] π k+1 = π k+2/3
Discrete-time RHEL. We are now ready to state the main result of this section.
Corollary A.7. Under the assumptions of Def. A.3-A.4, let (Φ k ) k satisfy the recursive equation:
Φ -K = x, Φ k+1 = M H,δ [Φ k , θ, u k ] ∀k = -K, • • • , -1,
and let Φ e satisfy:
           Φ e 0 (ϵ) = Φ ⋆ 0 + ϵΣ x • ∇ Φ ℓ[Φ 0 , θ, 0], ∀k = 0, • • • , K -1 : Φ e k+1/3 (ϵ) = M T,δ/2 [Φ e k (ϵ), θ, u -(k+1) ] Φ e k+2/3 (ϵ) = M V,δ [Φ e k+1/3 (ϵ), θ, u -(k+1) ] Φ e k+1 (ϵ) = M T,δ/2 [Φ e k+2/3 (ϵ), θ, u -(k+1) ] -ϵJ • ∇ Φ e ℓ[Φ e k+1 , θ, -(k + 1)],
Then defining:
H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] := 1 2 H[Φ e k+1/3 (ϵ), θ, u -(k+1) ] + H[Φ e k+2/3 (ϵ), θ, u -(k+1) ] , ∆ RHEL θ (k, ϵ) := ∇ 2 ℓ[Φ e k (ϵ), θ, -k] - δ 2ϵ ∇ 2 H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] -∇ 2 H 1/2 [Φ e k (-ϵ), θ, u -(k+1) ] , ∆ RHEL u (k, ϵ) := - δ 2ϵ ∇ 3 H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] -∇ 3 H 1/2 [Φ e k (-ϵ), θ, u -(k+1) ] , ∆ RHEL Φ (k, ϵ) := 1 2ϵ Σ x • (Φ e k (ϵ) -Φ e k (-ϵ)) ,
we have:
∀k = 0, • • • , K -1 : λ k = lim ϵ→0 ∆ RHEL θ (k, ϵ), g θ (k) = lim ϵ→0 ∆ RHEL θ (k, ϵ), g u (k) = lim ϵ→0 ∆ RHEL u (k, ϵ),
where (λ k ) k∈ 0,K , (g θ (k)) k∈ 0,K-1 and (g u (k)) k∈ 0,K-1 are defined inside Corollary (A.5).
Proof of Corollary A.7. Let k ∈ [0, K -1]. We define:
∆ RHEL Φ (k) := lim ϵ→0 ∆ RHEL Φ (k, ϵ) = Σ x • ∂ ϵ Φ(k, ϵ)| ϵ=0 ∆ RHEL θ (k) := lim ϵ→0 ∆ RHEL θ (k, ϵ) = ∇ 2 ℓ[Φ e k (0), θ, -k] -δ∂ ϵ ∇ 2 H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] | ϵ=0 ∆ RHEL u (k) := lim ϵ→0 ∆ RHEL u (k, ϵ) = -δ∂ ϵ ∇ 3 H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] | ϵ=0
Derivation of λ k = lim ϵ→0 ∆ RHEL θ (k, ϵ). We proceed exactly as in Theorem A.2 with some subtle adaptations which we highlight. Differentiating the dynamics of Φ e between k and k + 1/3 and proceeding as in the proof of Theorem A.2 using Lemma A.1, Lemma A.2 and Corollary A.6 (as the discrete counterpart of Lemma A.3 which was used for Theorem A.2), we obtain:
∆ RHEL Φ (k + 1/3) = ∆ RHEL Φ (k) + δ 2 ∇ 2 Φ T [Φ -k , θ, u -(k+1) ] • J ⊤ • ∆ RHEL Φ (k)
However note that this does not correctly match the dynamics satisfied by λ inside Corollary A.5 between k and k + 1/3: the Hessian ∇ 2 Φ T should instead be evaluated at Φ -(k+1/3) . Fortunately, using Lemma A.5:
∇ 2 Φ T [Φ -k , θ, u -(k+1) ] = ∇ 2 Φ T [Φ -(k+1/3) , θ, u -(k+1)
] Therefore we get:
∆ RHEL Φ (k + 1/3) = ∆ RHEL Φ (k) + δ 2 ∇ 2 Φ T [Φ -(k+1/3) , θ, u -(k+1) ] • J ⊤ • ∆ RHEL Φ (k)
Proceeding the same way on Φ e k+2/3 and Φ e k+1 , we get altogether:
       ∆ RHEL Φ (k + 1/3) = ∆ RHEL Φ (k) + δ 2 ∇ 2 Φ T [Φ -(k+1/3) , θ, u -(k+1) ] • J ⊤ • ∆ RHEL Φ (k) ∆ RHEL Φ (k + 2/3) = ∆ RHEL Φ (k + 1/3) + δ∇ 2 Φ V [Φ -(k+2/3) , θ, u -(k+1) ] • J ⊤ • ∆ RHEL Φ (k + 1/3) ∆ RHEL Φ (k + 1) = ∆ RHEL Φ (k + 2/3) + δ 2 ∇ 2 Φ T [Φ -(k+1) , θ, u -(k+1) ] • J ⊤ • ∆ RHEL Φ (k + 2/3) +∇ Φ ℓ[Φ -(k+1) , θ, -(k + 1)] ∆ RHEL Φ
satisfying the same equations as (λ k ) k given by BPTT, together with same initial conditions:
∆ RHEL Φ (0) = ∇ Φ ℓ[Φ 0 , θ, 0] yields the desired equality. Derivation of g θ (k) = lim ϵ→0 ∆ RHEL θ (k, ϵ).
Proceeding in the same way as in the derivation of Theorem A.2, starting from the expression of g θ (k) derived in Corollary A.5, using ∆ RHEL Φ (k) = λ k and paying attention to evaluating Jacobian at the right places using Lemma A.5, we obtain ∀k = 0, • • • , K -1:
g θ (k) = ∇ 2 ℓ[Φ e k (0), θ, -k] - δ 2 ∂ ϵ ∇ θ T [Φ e k (ϵ), θ, u -(k+1) ] + 2∇ θ V [Φ e k+1/3 (ϵ), θ, u -(k+1) ] + ∇ θ T [Φ e k+2/3 (ϵ), θ, u -(k+1) ] ϵ=0 .
There again, using Lemma A.5:
∇ θ T [Φ e k (ϵ), θ, u -(k+1) ] = ∇ θ T [Φ e k+1/3 (ϵ), θ, u -(k+1) ], ∇ θ V [Φ e k+1/3 (ϵ), θ, u -(k+1) ] = ∇ θ V [Φ e k+2/3 (ϵ), θ, u -(k+1) ], therefore: g θ (k) = ∇ 2 ℓ[Φ e k (0), θ, -k] - δ 2 ∂ ϵ ∇ θ T [Φ e k+1/3 (ϵ), θ, u -(k+1) ] + ∇ θ V [Φ e k+1/3 (ϵ), θ, u -(k+1) ] +∇ θ V [Φ e k+2/3 (ϵ), θ, u -(k+1) ] + ∇ θ T [Φ e k+2/3 (ϵ), θ, u -(k+1) ] ϵ=0 . = ∇ 2 ℓ[Φ e k (0), θ, -k] - δ 2 ∂ ϵ H[Φ e k+1/3 (ϵ), θ, u -(k+1) ] + H[Φ e k+2/3 (ϵ), θ, u -(k+1) ] ϵ=0 = ∇ 2 ℓ[Φ e k (0), θ, -k] -δ∂ ϵ H 1/2 [Φ e k (ϵ), θ, u -(k+1) ] ϵ=0 = lim ϵ→0 ∆ RHEL θ (k, ϵ) Derivation of g u (k) = lim ϵ→0 ∆ RHEL u (k, ϵ).
Strictly identical to the above paragraph.
Remark 6. Note that the echo dynamics prescribed by Corollary A.7 are implicit:
           Φ e 0 (ϵ) = Φ ⋆ 0 + ϵΣ x • ∇ Φ ℓ[Φ 0 , θ, 0], ∀k = 0, • • • , K -1 : Φ e k+1/3 (ϵ) = M T,δ/2 [Φ e k (ϵ), θ, u -(k+1) ] Φ e k+2/3 (ϵ) = M V,δ [Φ e k+1/3 (ϵ), θ, u -(k+1) ] Φ e k+1 (ϵ) = M T,δ/2 [Φ e k+2/3 (ϵ), θ, u -(k+1) ] -ϵJ • ∇ Φ e ℓ[Φ e k+1 , θ, -(k + 1)],
where Φ e k+1 appears, as highlighted in red, on both sides of the last step. A straightforward way to make the echo dynamics explicit while preserving the theoretical guarantees of Corollary A.7 is to linearize the nudging signal, namely using instead the following set of equations:
           Φ e 0 (ϵ) = Φ ⋆ 0 + ϵΣ x • ∇ Φ ℓ[Φ 0 , θ, 0], ∀k = 0, • • • , K -1 : Φ e k+1/3 (ϵ) = M T,δ/2 [Φ e k (ϵ), θ, u -(k+1) ] Φ e k+2/3 (ϵ) = M V,δ [Φ e k+1/3 (ϵ), θ, u -(k+1) ] Φ e k+1 (ϵ) = M T,δ/2 [Φ e k+2/3 (ϵ), θ, u -(k+1) ]+ϵΣ x • ∇ Φ e ℓ[Φ -(k+1)
, θ, -(k + 1)], Note that the ϵJ of the original implicit equation becomes -ϵΣ x in its linearized counterpart.
this section cite: []

Section: This remark leads us to a slight variant of Corollary A.7.
Corollary A.8. Under the assumptions of Def. A.3-A.4, let (Φ k ) k satisfy the recursive equation:
Φ -K = x, Φ k+1 = M H,δ [Φ k , θ, u k ] ∀k = -K, • • • , -1,
and let Φ e satisfy:
           Φ e 0 (ϵ) = Φ ⋆ 0 + ϵΣ x • y 0 , ∀k = 0, • • • , K -1 : Φ e k+1/3 (ϵ) = M T,δ/2 [Φ e k (ϵ), θ, u -(k+1) ] Φ e k+2/3 (ϵ) = M V,δ [Φ e k+1/3 (ϵ), θ, u -(k+1) ] Φ e k+1 (ϵ) = M T,δ/2 [Φ e k+2/3 (ϵ), θ, u -(k+1) ] + ϵΣ x • y -(k+1) ,
where y ∈ R K×d Φ does not depend on Φ. Then the same conclusions as Corollary A.7 hold, with (λ k ) k∈ 0,K-1 satisfying λ 0 = y 0 and ∀k ∈ [0, K -1]:
   λ k+1/3 = λ k + δ 2 ∇ 2 1 T [Φ -(k+1/3) , θ, u -(k+1) ] • J ⊤ • λ k λ k+2/3 = λ k+1/3 + δ∇ 2 1 V [Φ -(k+2/3) , θ, u -(k+1) ] • J ⊤ • λ k+1/3 λ k+1 = λ k+2/3 + δ 2 ∇ 2 1 T [Φ -(k+1) , θ, u -(k+1) ] • J ⊤ • λ k+2/3 + y -(k+1)
Proof Corollary A.8. Identical to the proof of Corollary A.7.
this section cite: []

Section: A.2.4 Proof of Theorem 3.3
Theorem A.4. Assuming a HSSM (Def. A.5) and the optimization problem depicted in Def. A.6, we have:
∀ℓ = 0, • • • , N -1 : d θ (ℓ) L = lim ϵ→0 ∆θ (ℓ) (ϵ),
where ∆θ (ℓ) (ϵ) can be recursively computed backwards, starting from the top-most block as:
Φ e,(N ) (ϵ) = M H (N -1) ,δ [Φ e,(N ) (ϵ), θ (N -1) , Φ (N -1) ] + ϵΣ x • ∇ Φ (N ) L ∆θ (N -1) (ϵ) = K-1 k=0 ∇ 2 ℓ[Φ e k , θ (L-1) , -k] - δ 2ϵ K-1 k=0 ∇ 2 H 1/2 [Φ e,(L) k (ϵ), θ (N -1) , Φ (N -1) -k ] -∇ 2 H 1/2 [Φ e,(N ) k (-ϵ), θ (N -1) , Φ (N -1) -k ] , ∆ Φ (N -1) (ϵ) = - δ 2ϵ ∇ 3 H 1/2 [Φ e,(N ) (ϵ), θ (N -1) , Φ (N -1) ] -∇ 3 H 1/2 [Φ e,(L) (-ϵ), θ (N -1) , Φ (N -1) ]
and subsequently for upstream blocks, i.e. ∀ℓ = N -2, • • • , 0:
Φ e,(ℓ+1) (ϵ) = M H (ℓ) ,δ [Φ e,(ℓ+1) (ϵ), θ (ℓ) , Φ (ℓ) ] + ϵΣ x • ∆ Φ (ℓ+1) (ϵ) ∆θ (ℓ) (ϵ) = - δ 2ϵ K-1 k=0 ∇ 2 H 1/2 [Φ e,(ℓ+1) k (ϵ), θ (ℓ) , Φ (ℓ) -k ] -∇ 2 H 1/2 [Φ e,(ℓ+1) k (-ϵ), θ (ℓ) , Φ (ℓ) -k ] ∆ Φ (ℓ) (ϵ) = - δ 2ϵ ∇ 3 H 1/2 [Φ e,(ℓ+1) (ϵ), θ (ℓ) , Φ (ℓ) ] -∇ 3 H 1/2 [Φ e,(ℓ+1) (-ϵ), θ (ℓ) , Φ (ℓ) ]
Proof. Re-using the vectorized notations introduced in subsection 3.3 and used in Remark 5, the Lagrangian associated to the optimization problem depicted in Def. A.6 reads:
L = 1 ⊤ • ℓ[ Φ (L) , θ (L-1) ] + L-1 ℓ=0 Tr M H (ℓ),δ [ Φ (ℓ+1) , θ (ℓ) , u (ℓ) ] -Φ (ℓ+1) • λ (ℓ+1) ⊤ = K-1 k=0 ℓ[Φ (L) -k , θ (L) , -k] + L-1 ℓ=0 λ (ℓ+1) k ⊤ • M H (ℓ) ,δ [Φ (ℓ+1) -(k+1) , θ (ℓ) , Φ (ℓ) -(k+1) ] -Φ (ℓ+1) -k
We proceed by induction on the block index starting from ℓ = L.
Initialization (ℓ = L). Let Φ (N ) k
and λ (N ) k for k ∈ 0, K -1 be the critical points of L. By Theorem A.3:
d θ (N -1) L = K-1 k=0 ∇ 2 ℓ Φ (N ) -k , θ (N -1) , -k + ∂ 2 M H (N -1) ,δ Φ (N ) -(k+1) , θ (L-1) , Φ (N -1) -(k+1) ⊤ • λ (N ) k , with (λ (N ) k ) k∈ 0,K-1 satisfying the following recursion relationship:      λ (N ) 0 = ∇ 1 ℓ[Φ (N ) 0 , θ (N -1) , 0], ∀k = 0, • • • , K -1 : λ (N ) k+1 = ∂ 1 M H (N -1) ,δ Φ (N ) -(k+1) , θ (N -1) , Φ (N -1) -(k+1) ⊤ • λ (N ) k + ∇ 1 ℓ Φ (N ) -(k+1) , θ, -(k + 1)
Given the definition of the dynamics of Φ e,(N ) by hypothesis, we can directly apply Corollary A.7 to obtain:
d θ (N -1) L = K-1 k=0 ∇ 2 ℓ Φ (N ) -k , θ (N -1) , -k -δ ∂ ϵ ∇ 2 H (N -1),1/2 Φ e,(N ) k (ϵ), θ (L-1) , Φ (N -1) -(k+1) ϵ=0 , d Φ (N -1) k-1 L = ∂ 3 M H (N -1) ,δ [Φ (N ) -(k+1) , θ (N -1) , Φ (N -1) -(k+1) ] ⊤ • λ (N ) k = -δ ∂ ϵ ∇ 3 H (N -1),1/2 Φ e,(N ) k (ϵ), θ (L-1) , Φ (N -1) -(k+1)ϵ=0
Induction (ℓ + 1 → ℓ). Let us assume that the desired property is satisfied at layer ℓ + 1. We denote again Φ
k and λ
(ℓ) k for k ∈ 0, K -1 the critical point of L. We have, for k ∈ 0, K -1 :        λ (ℓ) 0 = ∂ 3 M H (ℓ) ,δ Φ (ℓ+1) 0 , θ (ℓ) , Φ (ℓ) 0 ⊤ • λ (ℓ+1) 0 ∀k = 0, • • • , K -1 : λ (ℓ) k+1 = ∂ 1 M H (ℓ-1) ,δ Φ (ℓ) -(k+1) , θ (ℓ-1) , Φ (ℓ-1) -(k+1) ⊤ • λ (ℓ) k + ∂ 3 M H (ℓ) ,δ Φ (ℓ+1) -(k+1) , θ (ℓ) , Φ (ℓ) -(k+1) ⊤ • λ (ℓ+1) k
Using the induction hypothesis at layer (ℓ + 1):
∂ 3 M H (ℓ) ,δ Φ (ℓ+1) -(k+1) , θ (ℓ) , Φ (ℓ) -(k+1) ⊤ • λ (ℓ+1) k = -δ ∂ ϵ ∇ 3 H (ℓ),1/2 Φ e,(ℓ+1) k (ϵ), θ (ℓ) , Φ (ℓ) -(k+1) ϵ=0 = lim ϵ→0 ∆ Φ (ℓ) (k+1)(ϵ)
Therefore on the one hand, denoting
∆ Φ (ℓ) := lim ϵ→0 ∆ Φ (ℓ) (ϵ) ∈ R K×d Φ , the dynamics on λ rewrite:        λ (ℓ) 0 = ∆ Φ (ℓ) 0 ∀k = 0, • • • , K -1 : λ (ℓ) k+1 = ∂ 1 M H (ℓ-1) ,δ Φ (ℓ) -(k+1) , θ (ℓ-1) , Φ (ℓ-1) -(k+1) ⊤ • λ (ℓ) k + ∆ Φ (ℓ)(k+1)
On the other hand, the dynamics of Φ e,(ℓ) read by hypothesis:
       Φ (ℓ) 0 = Φ (ℓ) 0 ⋆ + ϵΣ x • ∆ Φ (ℓ) 0 (ϵ) ∀k = 0, • • • , K -1 : Φ e,(ℓ) k+1 = M H (ℓ-1) ,δ Φ e,(ℓ) k+1 , θ (ℓ-1) , Φ (ℓ-1) -(k+1) + ϵΣ x • ∆ Φ (ℓ)(k+1)
using Corollary A.8 with y = ∆ Φ (ℓ) , we conclude that:
d θ (ℓ) L = lim ϵ→0 ∆θ (ℓ) (ϵ)
Remark 7. Theorem A.4, and more generally our definition of HSSMs (Def. A.5) assume that the connectivity pattern of the HRU units is a linear chain. Note that while we chose this hypothesis for the sake of clarity of our results and their derivations, Theorem A.4 could be seamlessly extended to any Directed Acyclic Graph (DAG) of HRUs. This allows, for instance as a simple and realistic case, to use skip connections across HRUs within HSSMs. Remark 8. Note that RHEL chaining as prescribed by Theorem A.4 implicitly chains RHEL and automatic differentiation. Indeed, if H explicitly parametrizes feedforward mappings across HRUs as:
H (ℓ) Φ e,(ℓ+1) k (ϵ), θ (ℓ) , Φ(ℓ)
-(k+1) = H (ℓ) Φ e,(ℓ+1) k (ϵ), θ (ℓ) α , F Φ (ℓ) -(k+1) , θ (ℓ) β ,(23)
then, denoting u (ℓ) := F Φ (ℓ)
-(k+1) , θ(ℓ) β
, we have:
∂ ϵ ∇ 3 H (ℓ),1/2 Φ e,(ℓ+1) k (ϵ), θ (ℓ) , Φ (ℓ) -(k+1) ϵ=0 = ∂ 1 F Φ (ℓ) -(k+1) , θ (ℓ) β ⊤ • ∂ ϵ ∇ 3 H (ℓ),1/2 Φ e,(ℓ+1) k (ϵ), θ (ℓ) α , u ℓ ϵ=0 ≈ ∂ 1 F Φ (ℓ) -(k+1) , θ (ℓ) β ⊤ • 1 2ϵ ∇ 3 H (ℓ),1/2 Φ e,(ℓ+1) k (ϵ), θ (ℓ) α , u ℓ -∇ 3 H (ℓ),1/2 Φ e,(ℓ+1) k (-ϵ), θ (ℓ) α , u ℓ
The red part is done by automatic differentiation and the blue part by RHEL. This underpins the implementation of RHEL chaining we used in our own code.
this section cite: []

Section: A.3 Looking ahead A.3.1 Does RHEL break with dissipative dynamics?
Foreword. The applicability of RHEL is fundamentally limited by its restriction to conversative systems. Whenever dissipation is introduced, the system is no longer time-reversal invariant so that RHEL does not readily apply. One interesting question though is whether RHEL could be extended to the case where the time-reversed trajectory is physically feasible. Namely: the energy which is lost during the forward trajectory could be exactly pumped back into the system during the echo trajectory. Although the system may no longer be time-reversal invariant, we can ask ourselves whether perturbations of the time-reversed trajectory carry relevant error signals.
this section cite: []

Section: Dissipative Hamiltonian model.
For this purpose, we introduce dissipation inside Hamiltonian dynamics as [70]:
Φ(-T ) = x, ∀t ∈ [-T, 0] : ∂ t Φ(t) = (J -R) • ∇ Φ H[Φ(t), θ, u(t)],
with R symmetric and positive definite. We set R = I for simplicity -the loss of generality of this analysis is not an issue here as we state a negative result below.
Time-reversed dynamics. With this model, the corresponding time-reversed dynamics read:
∂ t Φ ⋆ (t) = -Σ z • ∂ t Φ(-t) = J ∇ Φ ⋆ H[ Φ ⋆ (t), θ, u(-t)] + Σ z • Σ z • ∇ Φ ⋆ H[ Φ ⋆ (t), θ, u(-t)] = J ∇ Φ ⋆ H[ Φ ⋆ (t), θ, u(-t)] + ∇ Φ ⋆ H[ Φ ⋆ (t), θ, u(-t)](24)
Therefore as expected: i) time-reversal invariance no long holds, ii) the sign of the dissipation in the forward trajectory is switched in the time-reversed trajectory.
ASM with dissipative dynamics. For this model, the ASM gradient estimate at time t reads ∀t ∈ [0, T ]:
g θ (t) := ∇ θ ℓ[-t, Φ(-t), θ] + ∇ 2 Φ,θ H[Φ(-t), θ, u(-t)] • J ⊤ • λ(t) -∇ 2 Φ,θ H[Φ(-t), θ, u(-t)] • λ(t)
with λ solving for the adjoint ODE:
λ(0) = 0 ∂ t λ(t) = ∇ 2 Φ H[Φ(-t), θ, u(-t)] • J ⊤ • λ(t) -∇ 2 Φ H[Φ(-t), θ, u(-t)] • λ(t) + ∇ Φ ℓ[-t, Φ(-t), θ](25)
We highlight in blue the extra term in the adjoint dynamics which is induced by the dissipation. RHEL procedure with dissipative dynamics. Following the intuition conveyed earlier, we now define the echo dynamics as perturbations of the time-reversed trajectory (Eq. ( 24)): Φ e (0) = Φ ⋆ (0) ∀t ∈ [0, T ] : ∂ t Φ e (t, ϵ) = J ∇ Φ e H[Φ e (t, ϵ), θ, u(-t)] + ∇ Φ e H[Φ e (t, ϵ), θ, u(-t)] -ϵJ ∇ Φ e ℓ[Φ e (t, ϵ), -t] Proceeding exactly as we did for the derivation of Theorem A.2, we differentiate the above echo dynamics with respect to ϵ at ϵ = 0, yielding:
∂ t (∂ ϵ Φ e (t, ϵ)| ϵ=0 ) = -Σ x • ∇ 2 Φ H[Φ(-t, 0)] • (J • Σ x ) • ∂ ϵ Φ e (t, ϵ)| ϵ=0 + Σ z • ∇ 2 Φ H[Φ(-t, 0)] • Σ z • ∂ ϵ Φ e (t, ϵ)| ϵ=0 + Σ x • ∇ Φ ℓ[t, Φ(-t), θ]
Multiplying by Σ x on both sides and reusing the notation ∆ RHEL Φ (t) := Σ x • ∂ ϵ Φ e (t, ϵ)| ϵ=0 , we have that:
∆ RHEL Φ (0) = 0 ∂ t ∆ RHEL Φ (t) = ∇ 2 Φ H[Φ(-t, 0)] • J ⊤ • ∆ RHEL Φ (t) -J • ∇ 2 Φ H[Φ(-t, 0)] • J • ∆ RHEL Φ (t) + ∇ Φ ℓ[t, Φ(-t), θ](26)
where we have highlighted in red the extra term induced by dissipation.
Comparing Eq. ( 25) and Eq. ( 26), we see that the red term inside the perturbed echo dynamics and the blue term inside the ASM dynamics differ. Therefore, it cannot be concluded that λ and ∆ RHEL Φ are equal at all times, as it is the case for time-reversal invariant systems.
this section cite: ['b69']

Section: Conclusion.
Using our methodology based on the equivalence with the continuous ASM, the echo dynamics cannot be seamlessly adapted to the case where the system is subject to dissipation, even when "pumping energy" back into the system". However, this does not mean that developing forward-only, perturbation-based methods for temporal credit assignment is impossible in general when systems exhibit dissipation. In a study released shortly after our submission [71], it was shown that a class of linear dissipative systems could be described with a Lagrangian formalism (which is closely related to our Hamiltonian formalism) and gradients estimated as finite differences of two dissipative trajectories, provided that the system is subject to periodic boundary conditions.
this section cite: ['b70']

Section: References
Ref_id:b0 Title: Resurrecting recurrent neural networks for long sequences Year: (2023)
Ref_id:b1 Title: Combining recurrent, convolutional, and continuous-time models with linear state space layers Year: (2021)
Ref_id:b2 Title: Efficiently modeling long sequences with structured state spaces Year: (2021)
Ref_id:b3 Title: Diagonal state spaces are as effective as structured state spaces Year: (2022)
Ref_id:b4 Title: Simplified state space layers for sequence modeling Year: (2022)
Ref_id:b5 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b6 Title: The hardware lottery Year: (2021)
Ref_id:b7 Title: Attention is all you need Year: (2017)
Ref_id:b8 Title: The hardware is the software Year: (2024)
Ref_id:b9 Title: Self-learning machines based on hamiltonian echo backpropagation Year: (2023)
Ref_id:b10 Title: Training of physical neural networks Year: (2024)
Ref_id:b11 Title: Automatic differentiation in machine learning: a survey Year: (2018)
Ref_id:b12 Title: A learning algorithm for continually running fully recurrent neural networks Year: (1989)
Ref_id:b13 Title: Online learning of long-range dependencies Year: (2023)
Ref_id:b14 Title: Unbiased online recurrent optimization Year: (2017)
Ref_id:b15 Title: Robert Legenstein, and Wolfgang Maass. A solution to the learning dilemma for recurrent networks of spiking neurons Year: (2020)
Ref_id:b16 Title: A truly sparse and general implementation of gradient-based synaptic plasticity Year: (2025)
Ref_id:b17 Title: Accelerating stochastic gradient descent using predictive variance reduction Year: (2013)
Ref_id:b18 Title:  Year: (2019)
Ref_id:b19 Title: Multivariate stochastic approximation using a simultaneous perturbation gradient approximation Year: (1992)
Ref_id:b20 Title: Model of birdsong learning based on gradient estimation by dynamic perturbation of neural conductances Year: (2007)
Ref_id:b21 Title: Scaling forward gradient with local losses Year: (2022)
Ref_id:b22 Title: Learning by directional gradient descent Year: (2021)
Ref_id:b23 Title: Can forward gradient match backpropagation Year: (2023)
Ref_id:b24 Title: Fine-tuning language models with just forward passes Year: (2023)
Ref_id:b25 Title: A cookbook for hardware-friendly implicit learning on static data Year: (2024)
Ref_id:b26 Title: Equilibrium propagation: Bridging the gap between energy-based models and backpropagation Year: (2017)
Ref_id:b27 Title: Mathematical theory of optimal processes Year: (1985)
Ref_id:b28 Title: Oscillatory state-space models Year: (2024)
Ref_id:b29 Title: Unicornn: A recurrent model for learning very long time dependencies Year: (2021)
Ref_id:b30 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b31 Title: The elements of differentiable programming Year: (2024)
Ref_id:b32 Title: Numerical methods for ordinary differential equations Year: (2018)
Ref_id:b33 Title: Log neural controlled differential equations: The lie brackets make a difference Year: (2024)
Ref_id:b34 Title: The uea multivariate time series classification archive Year: (2018)
Ref_id:b35 Title: Deep ppg: Large-scale heart rate estimation with convolutional neural networks Year: (2019)
Ref_id:b36 Title: Scaling equilibrium propagation to deep convnets by drastically reducing its gradient estimator bias Year: (2021)
Ref_id:b37 Title: The stochastic thermodynamics of computation Year: (2019)
Ref_id:b38 Title: Discusses relaxing digital constraints and leveraging analog physics:contentReference Year: (2025-07)
Ref_id:b39 Title: Mlir: Scaling compiler infrastructure for domain specific computation Year: (2021)
Ref_id:b40 Title: Fast transformer decoding: One write-head is all you need Year: (2019)
Ref_id:b41 Title: Flashattention: Fast and memory-efficient exact attention with io-awareness Year: (2022)
Ref_id:b42 Title: Proposes a thermodynamic framework for AI hardware and algorithms:contentReference Year: (2023)
Ref_id:b43 Title: Neuromorphic computing using non-volatile memory Year: (2017)
Ref_id:b44 Title: An optimization network for matrix inversion Year: (1987)
Ref_id:b45 Title: In-memory hyperdimensional computing Year: (2020)
Ref_id:b46 Title: Demonstrates that physical systems implement Lagrange multiplier optimization:contentReference Year: (2020)
Ref_id:b47 Title: Analog optical computer for ai inference and combinatorial optimization Year: (2025)
Ref_id:b48 Title: Supervised learning in physical networks: From machine learning to learning machines Year: ()
Ref_id:b49 Title: Introduces equilibrium propagation for training analog neural networks:contentReference Year: (2020)
Ref_id:b50 Title: Supervised learning in physical networks: From machine learning to learning machines Year: (2021)
Ref_id:b51 Title: Holomorphic equilibrium propagation computes exact gradients through finite size oscillations. Advances in neural information processing systems Year: (2022)
Ref_id:b52 Title: Energy-based learning algorithms for analog computing: a comparative study Year: (2023)
Ref_id:b53 Title: Towards training digitally-tied analog blocks via hybrid gradient computation Year: (2024)
Ref_id:b54 Title: A fast algorithm to simulate nonlinear resistive networks Year: (2024)
Ref_id:b55 Title: Symplectic recurrent neural networks Year: (2019)
Ref_id:b56 Title: Hamiltonian neural networks Year: (2019)
Ref_id:b57 Title: Lagrangian neural networks Year: (2020)
Ref_id:b58 Title: Sébastien Racanière, Aleksandar Botev, and Irina Higgins. Hamiltonian generative networks Year: (2019)
Ref_id:b59 Title: Hybrid monte carlo Year: (1987)
Ref_id:b60 Title: Mcmc using hamiltonian dynamics. Handbook of markov chain monte carlo Year: (2011)
Ref_id:b61 Title: Magnetic hamiltonian monte carlo Year: (2017)
Ref_id:b62 Title: Memory-efficient optimization with factorized hamiltonian descent Year: (2024)
Ref_id:b63 Title: Training deep nets with sublinear memory cost Year: (2016)
Ref_id:b64 Title: Quzo: Quantized zeroth-order fine-tuning for large language models Year: (2025)
Ref_id:b65 Title: Agnostic physicsdriven deep learning Year: (2022)
Ref_id:b66 Title: A photonics perspective on computing with physical substrates Year: (2024-12)
Ref_id:b67 Title: SuperMind: A survey of the potential of superconducting electronics for neuromorphic computing Year: (2022-03)
Ref_id:b68 Title: Neuromorphic spintronics Year: (2020-07)
Ref_id:b69 Title: Learning dissipative hamiltonian dynamics with reproducing kernel hilbert spaces and random fourier features Year: (2024)
Ref_id:b70 Title: Equilibrium propagation for periodic dynamics Year: (2025)
Ref_id:b71 Title: Gaussian Error Linear Units (GELUs) Year: (2023-06)
Ref_id:b72 Title: Language Modeling with Gated Convolutional Networks Year: (2017-07)
Ref_id:b73 Title: JAX: composable transformations of Python+NumPy programs Year: (2018)
