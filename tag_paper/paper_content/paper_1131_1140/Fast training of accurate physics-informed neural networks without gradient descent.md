Title: FAST TRAINING OF ACCURATE PHYSICS-INFORMED NEURAL NETWORKS WITHOUT GRADIENT DESCENT
Abstract: Solving time-dependent Partial Differential Equations (PDEs) is one of the most critical problems in computational science. While Physics-Informed Neural Networks (PINNs) offer a promising framework for approximating PDE solutions, their accuracy and training speed are limited by two core barriers: gradient-descentbased iterative optimization over complex loss landscapes and non-causal treatment of time as an extra spatial dimension. We present Frozen-PINN, a novel PINN based on the principle of space-time separation that leverages random features instead of training with gradient descent, and incorporates temporal causality by construction. On nine PDE benchmarks, including challenges like extreme advection speeds, shocks, and high-dimensionality, Frozen-PINNs achieve superior training efficiency and accuracy over state-of-the-art PINNs, often by several orders of magnitude. Our work addresses longstanding training and accuracy bottlenecks of PINNs, delivering quickly trainable, highly accurate, and inherently causal PDE solvers, a combination that prior methods could not realize. Our approach challenges the reliance of PINNs on stochastic gradient-descent-based methods and specialized hardware, leading to a paradigm shift in PINN training and providing a challenging benchmark for the community.

Section: INTRODUCTION
Partial Differential Equations (PDEs) provide a unifying framework for modeling complex dynamical systems across physics, biology, and engineering, yet developing efficient methods to solve them remains a longstanding challenge (Farlow, 1993). Deep neural networks have recently shown significant promise for approximating solutions of PDEs because of the mesh-free construction of basis functions, high expressivity of neural networks (Rudi & Rosasco, 2021), their ability to represent functions in high dimensions (E, 2020;Wu & Long, 2022;Han et al., 2018), and powerful software for automatic differentiation (e.g., Pytorch (Paszke et al., 2017), TensorFlow (Abadi et al., 2015), DeepXDE (Lu et al., 2021b)). Earlier work on solving PDEs using neural networks (Dissanayake & Phan-Thien, 1994;Lagaris et al., 1998) was recently popularized in the form of Physics-informed neural networks (PINNs) (Raissi et al., 2019;Karniadakis et al., 2021;Sirignano & Spiliopoulos, 2018). PINNs incorporate physical constraints by minimizing a loss function involving the PDE, boundary condition, and initial condition residuals during training. Despite their promise, we identify two root causes limiting the performance of PINNs in terms of accuracy and training time.
1. Inherent challenges posed by the PINN optimization problem: Many studies (Wang et al., 2021;2022) show that even in very simple settings, the PINN loss is quite challenging to minimize using iterative gradient-descent-based optimization methods leveraging the classical back-propagation algorithm (Rumelhart et al., 1986). Krishnapriyan et al. (2021) show that incorporating PDE-based soft constraints into the PINN loss function yields a highly nontrivial loss landscape, rendering optimization particularly challenging. Wang et al. (2022) analyze PINN training dynamics via the Neural Tangent Kernel (NTK) and highlight issues with spectral bias and different convergence rates across different loss components. Rathore et al. (2024) show that differential operators in the PDE residual loss induce "ill-conditioning", characterized by steep and shallow gradients in different directions near the optimum, complicating the optimization.
Efforts to improve PINN training, such as balancing loss terms (Yao et al., 2023), effective regularization (Lu et al., 2021c;Yu et al., 2022), architectural innovations (Wang et al., 2024b), and improved optimizers (Müller & Zeinhofer, 2023;Liu et al., 2024), have been explored. We assert that such approaches address the symptoms rather than the root cause that makes training PINNs extremely challenging: the PINN optimization problem is high-dimensional (large number of trainable parameters), multi-objective (simultaneous minimization of PDE, and initial and boundary condition losses), and non-convex, with inherently conflicting loss terms (Liu et al., 2024) and further complicated by treating time as an additional dimension in space.
2. Non-causal treatment of time as an extra spatial dimension: The temporal structure of initial value PDEs is inherently Markovian as the solution at each subsequent time step depends solely on the solution at the preceding time step. Most PINN-based approaches fail to incorporate temporal causality explicitly, and time is treated as an extra dimension in the input layer. This leads to neural bases spanning the entire space-time domain, exacerbating the optimization. Such approaches struggle to capture high-frequency temporal dynamics (Krishnapriyan et al., 2021), and solving PDEs over a long time horizon, without resorting to domain decomposition techniques (Meng et al., 2020).
Previous studies have sought to enforce temporal causality by progressively penalizing residuals in time (Wang et al., 2024d), training distinct models across disjoint intervals with integral-form losses within each interval (Jung et al., 2024), or applying implicit time-differencing with transfer learning to sequentially update PINNs on each interval (Li et al., 2024). Nonetheless, such approaches are difficult to implement, require precise tuning of temporal windows and weight scheduling, and remain computationally demanding (Kim & Son, 2025;Li et al., 2024;Penwarden et al., 2023). See Appendix A for an extended literature review and Appendix B.1 for a detailed discussion on PINNs.
Time-dependent output layer coefficients Time (1) Space-time separation (2) Decouple the PINN loss and optimize each term separately Space Frozen-PINN solution Basis functions in space: sampled and frozen Total loss: Classical ODE solvers with step-size control Impose hard constraints Solve a least squares problem . To address the root causes of accuracy and training bottlenecks of PINNs rather than the symptoms, we investigate: How can the PINN optimization problem be simplified while enforcing temporal causality for time-dependent PDEs? We propose "Frozen-PINN" based on space-time separation -a novel approach that simplifies the PINN optimization problem and enforces temporal causality by construction. We achieve this by: (a) sampling and freezing space-dependent hidden layer parameters to reduce the dimensionality, (b) decoupling the PINN loss and optimizing each term separately, and (c) computing time-dependent output layer parameters using least squares and adaptive Ordinary Differential Equation (ODE) solvers, replacing gradient-descent-based training (see Figure 1). In Figure 8, we contrast Frozen-PINNs with classical PINNs. Our key contributions are:
1. Training algorithm: Frozen-PINNs break the longstanding training and accuracy bottlenecks of PINNs, making PINNs rapidly trainable, temporally causal, and highly accurate, a combination realized for the first time, defining a new state-of-the-art, to our knowledge. 2. Extensive empirical evaluation: Across nine challenging PDE benchmarks and rigorous ablation studies, we show that Frozen-PINNs achieve up to 4-5 orders of magnitude faster training than state-of-the-art (SOTA) PINNs, attain high-precision accuracies that are comparable to efficient mesh-based methods in low dimensions, which most SOTA neural PDE solvers fail to match, and scale efficiently to high-dimensional problems where mesh-based solvers fail.
this section cite: ['b37', 'b97', 'b35', 'b118', 'b44', 'b84', 'b0', 'b26', 'b64', 'b94', 'b104', 'b112', 'b85', 'b98', 'b63', 'b113', 'b95', 'b121', 'b123', 'b82', 'b72', 'b72', 'b63', 'b79', 'b54', 'b69', 'b60', 'b69', 'b86']

Section: Adaptive solution-driven network parameters:
We use solution data from previous time-steps to compute efficient neural network parameters. This extends previous work on random feature methods (Bolager et al., 2023) for self-supervised PDE learning tasks.
this section cite: ['b7']

Section: Model compression:
We introduce an SVD layer that reduces the number of neurons in the last hidden layer of the network by up to 20 times and speeds up training up to 75 times.
this section cite: []

Section: SOLVING TIME-DEPENDENT PDES USING FROZEN-PINNS
In this section, we discuss the theoretical details of Frozen-PINNs.
this section cite: []

Section: FROZEN-PINN ANSATZ
In this work, we consider time-dependent PDEs on domain Ω ⊂ R d for space dimension d with boundary ∂Ω, seeking solutions u : Ω × R → R of PDEs defined by linear operators L and B that only involve derivative operators in space, forcing f : Ω → R, boundary g : ∂Ω → R, initial condition u 0 : Ω → R, and a nonlinear operator γN for γ ∈ R (γ = 0 for linear PDEs):
u t (x, t) + Lu(x, t) + γN (u)(x, t) = f (x), x ∈ Ω, t ∈ [0, T ],
where u t denotes the time derivative of u, with boundary and initial conditions given by Bu(x, t) = g(x), x ∈ ∂Ω, t ∈ [0, T ], and, u(x, 0) = u 0 (x), x ∈ Ω,
respectively. We parameterize the approximation of the solution to the PDE (Equation ( 1)) with a Frozen-PINN having a single hidden layer with M neurons and activation function σ = tanh as û(x, t) = C(t) [Φ(x), 1] = c(t)σ(W x ⊤ + b) + c 0 (t).
(2)
Here, c(t) ∈ R 1×M and c 0 (t) ∈ R are time-dependent parameters, W ∈ R M ×d and b ∈ R M ×1 are space-dependent parameters, and C := [c, c 0 ] ∈ R 1×(M +1) . The activation functions are stacked in Φ = [ϕ 1 , . . . , ϕ M ], where ϕ m (x) = σ(w m x ⊤ + b m ). Note that our approach does not require the PDE solution to be separable in space and time. We next discuss how to sample parameters W and b.
this section cite: []

Section: COMPUTING HIDDEN LAYER PARAMETERS WITHOUT GRADIENT DESCENT
We sample space-dependent hidden layer parameters in Frozen-PINNs using either ELM or SWIM.
Hidden layer parameters are frozen (kept independent of time) after sampling (except Section 3.4).
this section cite: []

Section: ELM (Data-agnostic):
In the Extreme Learning Machine (ELM) approach (Huang et al., 2006), the weights are sampled from a Gaussian distribution, and biases are sampled from a uniform distribution in [-η, η] for each hidden layer, where η is a hyper-parameter.
this section cite: ['b48']

Section: SWIM (Data-dependent):
The Sample Where It Matters (SWIM) approach follows Bolager et al. (2023) and samples weights and biases using a data-dependent distribution. Each pair (w m , b m ) is computed using two collocation points x (1) , x (2) ∈ Ω: w m = s 1
x (2) -x (1)
∥x (2) -x (1) ∥ 2 , b m = -⟨w m , x (1) ⟩+ s 2 , where s 1 , s 2 ∈ R depend on the activation function. In the unsupervised setting, one can choose pairs of collocation points from a uniform distribution over all possible pairs of collocation points, which is the default setting in this paper, as we do not know the solution of the PDE beforehand. In the supervised setting (Section 3.4, Section 3.7), collocation pairs (x (1) , x (2) ) are sampled with density ∥f (x (2) ) -f (x (1) )∥/∥x (2) -x (1) ∥. Neuron weights and biases are set so that the tanh output is -0.5 at x (1) and +0.5 at x (2) , ensuring centers of activations tanh lie inside the domain and are aligned with the direction x (1) → x (2) , unlike ELM. The suitability of each of the proposed approaches depends on the true PDE solution's gradient distribution. See Appendix B.2.1 for details. In Figure 2, we illustrate the difference between the basis functions sampled with ELM and SWIM.
this section cite: ['b7']

Section: Data-driven sampling
Data-agnostic sampling
this section cite: []

Section: SOLVING TIME-DEPENDENT PDES USING FROZEN-PINNS BY SEPARATION OF VARIABLES
We now discuss the computation of time-dependent output layer parameters c(t). We insert the ansatz (Equation ( 2)) into the PDE Equation (1a), reformulating it as an ODE for c(t), preserving the inherent causal structure of time-dependent PDEs, thereby enforcing temporal causality by design. We assemble N c collocation points in X ∈ R Nc×d , sample weights and biases of M neurons, compute hidden layer output Φ(X), and obtain the ODE
C t (t) = R(X, C(t))[Φ(X), 1] + , where R(X, C(t)) = -C(t)L[Φ(X), 1] -γN (C(t)[Φ(X), 1]) + [f (X)] ⊤ ,(3)
where [Φ(X), 1] ∈ R (M +1)×Nc and the pseudo-inverse is denoted by • + . The initial condition is computed via a least squares solution: C(0) = u(X, 0) ⊤ [Φ(X), 1] + , which decouples the initial condition loss from PDE and boundary losses, simplifying the optimization problem. We compute C(t) via ODE solvers with step-size control (e.g., RK45 (Dormand & Prince, 1980), LSODA (Petzold, 1983)) instead of gradient descent, and interpolate solutions at test points. See Appendix B.2.2, Appendix B.2.3 for detailed derivations of PDE-to-ODE reformulations for all PDEs considered here.
this section cite: ['b87']

Section: APPROACHES FOR SATISFYING BOUNDARY CONDITIONS FOR FROZEN-PINNS
We propose two different strategies to satisfy boundary conditions for Frozen-PINNs: the first utilizes a boundary-compliant layer, and the second augments the reformulated ODE.
Boundary-compliant layer: Certain boundary conditions can be enforced via a linear map A ∈ R M b ×Ms (M s := M ) applied after the sampled hidden layer, forming a boundary-compliant layer (see Figure 3). Defining Φ A := [AΦ, 1] and C(t) ∈ R 1×(M b +1) , we rewrite Equation (3) to
C t (t) = R(X, C(t))Φ A (X) + , where R(X, C(t)) = -C(t)LΦ A (X) -γ N (C(t)Φ A (X)) + [f (X)] ⊤ .(4)
Boundary conditions defined by B and g determine the construction of A; see Appendix B.2.4 for details. With a boundary-compliant layer, boundary conditions are satisfied by construction, fully decoupling the PINN loss so that the ODE solver minimizes only the PDE residual. The rationale for outer basis functions is discussed in Appendix B.2.1.
Augmented ODE: This strategy eliminates the need for a boundary-compliant layer by augmenting the ODE with a correction term enforcing boundary conditions. For Dirichlet boundary condition u(x) = g(x), we add ût (x) = -κ(û(x) -g(x)) for x ∈ ∂Ω and solve the augmented system:
C t (t) = [R(X, C(t)), -κ(C(t)Φ A (X b )-g(X b ) ⊤ )] ∈R 1×(Nc +N b ) Φ A ([X, X b ]) + ∈R (Nc +N b )×(M b +1) ,(5)
where κ > 0 is a fixed parameter, X are the N c collocation points and X b ∈ R N b ×d is a collection of N b points on the boundary ∂Ω. For consistency of notation, we set A = I in Equation (4) when using the augmented ODE. In practice, we skip the boundary-compliant layer if we adopt this approach. The intuition behind this technique is that the augmented ODE (Equation ( 5)) corrects the solution by steering û(x, t) toward g(x) for x ∈ ∂Ω at rate κ(û -g), with κ = 10 5 as a default value. We empirically investigate the effect of κ on the boundary loss and the time to solution (see Figure 15). This still partially decouples the PINN loss, with the initial condition treated separately. Depending on the PDE, domain, and boundary type, either strategy can be applied (see Appendix B.2.5).
this section cite: []

Section: SVD LAYER
As the last step in the Frozen-PINN architecture, we add a linear layer to reduce the stiffness of the associated ODE (Equation ( 4)) and the size of the ODE system. To achieve this, we propose orthogonalizing the basis functions using an SVD layer. We compute a truncated singular value decomposition of AΦ(X) ∈ R M b ×Nc to obtain matrices V r , Σ r , and
U r with r ≤ M b such that V r Σ r U ⊤ r = AΦ(X) + O(Σ r+1
). We then define A r := V ⊤ r A and use it instead of the matrix A and C(t) ∈ R 1×(r+1) . This ensures A r Φ(X) are orthogonal functions on the data X, and the matrix A r Φ(X) has a bounded condition number. The SVD layer accelerates computation by up to 75 times while reducing the ODE system dimension 20 times, as validated by an extensive ablation study (see Appendix C). Figure 3 visualizes the complete Frozen-PINN architecture.
this section cite: []

Section: SUMMARY OF THE TRAINING ALGORITHM FOR FROZEN-PINNS
We summarize our training process in Algorithm 1, where ϵ SV D is the SVD threshold that governs the SVD-layer width. See Appendix B.2 for additional methodological details, and Appendix B.2.1 for extended discussion on PINN vs. Frozen-PINN training, comparison between sampling strategies, influence of random sampling, rationale for outer bases, and the Kolmogorov n-width barrier.
this section cite: []

Section: SVD layer
Output layer ELM/ SWIM
this section cite: []

Section: Boundarycompliant layer

this section cite: []

Section: Input layer
Figure 3: Architecture of Frozen-PINNs trained with a gradientdescent-free training algorithm.
this section cite: []

Section: Algorithm 1 Frozen-PINN training algorithm
Input: PDE (Equation ( 1)), test grid points Xtest × Ttest Output: PDE Solution on the test grid points û(Xtest, Ttest)
Parameters: Nc, Ms, M b ∈ N, ϵ SV D ∈ R 1: Sample Nc collocation points: X ∈ R Nc ×d 2: Construct hidden layer params {wm, bm} Ms m=1 (SWIM/ELM) ▷ Section 2.2 3: Compute hidden layer output Φ(X) ∈ R Ms ×Nc 4: Construct boundary-compliant layer: AΦ(X) ∈ R M b ×Nc ▷ Section 2.4 5: Compute truncated SVD: VrΣrU ⊤ r = AΦ(X) and SVD layer output V ⊤ r AΦ(X) = ArΦ(X)
6: Compute neural bases: Φ Ar (X) := (ArΦ(X), 1) ⊤ ∈ R (r+1)×Nc 7: Initialize output-layer params (least-squares):
C(0) = u(X, 0) ⊤ Φ Ar (X) + 8: Solve ODE for C(t) ∈ R 1×(r+1) using Φ Ar ▷ Equation (4) 9: Evaluate û(Xtest, Ttest) = C(Ttest)Φ Ar (Xtest) ▷ Equation (2) 3 EMPIRICAL RESULTS
In this section, with a comprehensive empirical study across nine challenging low-and highdimensional PDE benchmarks, we demonstrate that Frozen-PINNs consistently outperform existing state-of-the-art neural PDE solvers with orders-of-magnitude faster training in all cases and higher accuracy in almost all cases without requiring specialized hardware like GPUs. Moreover, our work includes rigorous evaluation against the classical SOTA approaches like IGA-FEM (see Appendix B.3) (Hughes et al., 2005;Cottrell et al., 2006;2009) or FEM for low-dimensional PDEs, bridging a gap not sufficiently addressed in the literature between neural and mesh-based solvers.
Appendix C contains details of the PDEs, important ablation studies for our experiments (for the SVD layer and the width of the network), metrics used for comparison, train and test data, software and hardware environments, the absolute error plots on test points, and elaborate explanations of results. Figure 12 visually summarizes all the PDE benchmarks used for evaluation, identifies the specific challenges posed by each PDE, and shows true solutions. We perform all experiments with three seeds and report the mean and standard deviation. The code to reproduce experiments from the paper, and a refactored version that will be actively maintained are available at:
https://gitlab.com/felix.dietrich/swimpde-paper, https://gitlab.com/fd-research/swimpde.
To ensure fair comparisons, we follow the two rules outlined by McGreivy & Hakim (2024): (i) we benchmark at (almost) equal accuracy, defining low-precision (1e-2 to 1e-4) and high-precision (1e-5 to 1e-10) regimes, configuring Frozen-PINNs to marginally outperform the best PINN baselines in the low-precision regime and aligning FEM/IGA-FEM fidelity with Frozen-PINNs in the high-precision regime; (ii) we compare against efficient numerical methods, including SOTA IGA-FEM or classical FEM for low-dimensional PDEs, while highlighting neural solvers' scalability in high-dimensional benchmarks where FEM and IGA-FEM suffer from the curse of dimensionality.
this section cite: ['b50', 'b21']

Section: HIGH ADVECTION SPEEDS, FAST CONVERGENCE, AND LONG-TIME SIMULATION
We benchmark the linear advection equation to demonstrate how Frozen-PINNs resolve three important well-known challenges for PINNs: (1) handling high advection speeds (Krishnapriyan et al., 2021), (2) achieving fast convergence with increasing width (Cuomo et al., 2022), and (3) long-time simulations (Lippe et al., 2024;Kapoor et al., 2024a). We describe all details in Appendix C.1.
this section cite: ['b63', 'b23', 'b71']

Section: High advection speeds:
We solve the advection equation for increasing advection coefficients, denoted by β. Figure 4 (Left) shows that approaches using basis functions in the entire spatiotemporal domain, such as PINNs, ELM, and SWIM, completely fail as the flow velocity β increases beyond 40.
In contrast, Frozen-PINNs can accurately solve the PDE, even for extremely high values of β (as high as 10 4 ) with relative L 2 errors less than 10 -4 . Table 1 shows that for β = 40, Frozen-PINNs train 45 to 533 times faster than other alternatives at similar accuracy in the low-precision regime. With the exception of Frozen-PINNs, none of the neural PDE solvers evaluated here attain high-precision accuracy. Frozen-PINNs outperform existing neural PDE solvers by over six orders of magnitude in accuracy and approach the fidelity of IGA-FEM, which unsurprisingly is the most accurate solver.
Fast convergence (error decay with hidden layer width): For a low value of advection coefficient β = 10, Figure 4 (Middle) shows that errors with classical PINNs do not decay quickly with width, primarily due to the difficulties in training. In contrast, the relative L 2 error decays exponentially with hidden layer width for Frozen-PINNs, ultimately plateauing at a value more than four orders of magnitude smaller than that obtained with PINNs.
Long-time simulation: Neural PDE solvers employing joint space-time basis functions, like vanilla PINNs, encounter substantial challenges in accurately approximating dynamics over extended time spans. Here, we consider the advection equation with the advection coefficient β = 1. As shown in Figure 4 (Right), Frozen-PINNs can simulate the advection equation for 1000 seconds with a relative L 2 error under 0.001% in just 0.94 seconds.
this section cite: []

Section: HIGHER-ORDER DERIVATIVES IN SPACE AND TIME
We consider two variants of the Euler-Bernoulli beam equation -classical Euler-Bernoulli beam equation and its extension with a Winkler foundation. See Appendix C.2 for details. The main challenge posed by both PDEs for PINNs is the higher-order differential terms (fourth-and secondorder derivatives in space and time, respectively). Frozen-PINNs eliminate expensive evaluation of higher-order derivatives via backpropagation, cutting training cost by four orders of magnitude in the low-precision regime, while achieving IGA-FEM-level accuracy that is more than six orders of magnitude accurate compared to other SOTA PINN benchmarks considered here (see Table 1).
this section cite: []

Section: MULTI-SCALE SOLUTIONS
To demonstrate the capability of our method to solve PDEs with multi-scale solutions, we consider a Wave equation benchmark (Hao et al., 2024). We examine two settings: one with two distinct frequencies and another with three well-separated frequencies, which increases the spatial complexity and significantly broadens the range of scales in the solution. For the two-frequency setup, we compare the performance against prior PINN baselines in Table 1 and observe that CPU-trained Frozen-PINNs achieve 625 to 5500 times faster training than GPU-trained competing PINN variants, while simultaneously being four to five orders of magnitude more accurate. Frozen-PINNs also solve the wave equation in the three-frequency scenario (illustrated in Figure 19) extremely quickly and with high precision, reinforcing their potential for solving PDEs with complex, multiscale dynamics. Additional implementation details and extended results are provided in Appendix C.3.
this section cite: ['b45']

Section: NON-LINEARITY AND SHOCKS
In this example, we highlight how using pairs of data points to sample neural basis functions using the SWIM algorithm can be leveraged to resolve locally steep gradients in the solution of the non-linear viscous Burgers' equation, as shown in Figure 2 (Left). See Appendix C.4 for details.
Frozen-PINN-swim creates numerous basic functions with steep gradients, accurately placing them near the location of the shock, leveraging the SWIM algorithm and solutions from previous time-steps to fit neural basis functions, given enough collocation points in the domain's center (see Figure 20a (Left)). To concentrate collocation points near the shock in the domain's center, we resample them periodically after a set number of time steps, guided by a probability distribution that leverages the gradient of the approximate solution (see Figure 5a
this section cite: []

Section: NON-LINEARITY AND COMPLICATED DOMAIN GEOMETRY
In this example, we consider a non-linear diffusion equation on a complicated domain geometry. See Appendix C.5 for details. For mesh-based methods, meshing can be resource-intensive and technically demanding (see Figure 24), unlike neural PDE solvers. As shown in Table 1, Frozen-PINNs are 145 to 456 times faster than PINNs and 4.83 times faster than FEM at comparable low-precision accuracy, and can achieve over 1000 times better accuracy than other PINNs. Notably, Frozen-PINNs require only 350 basis functions versus around 2000 finite elements in FEM for similar accuracy (see Table 18), mainly due to the global support of neural bases. For fairness, the FEM grid points are reused as collocation points for minimizing the PDE residual in Frozen-PINNs.
this section cite: []

Section: CHAOS AND STRONG NON-LINEARITY
We tackle the highly nonlinear Kuramoto-Sivashinsky equation, which models laminar flame-front instabilities that exhibit spatiotemporal chaos. As shown in Figure 6, our Frozen-PINN captures the characteristic chaotic pattern over a long-time horizon t ∈ [0, 5], with an average training time of only 6.9 seconds on CPU (averaged over 5 seeds). Further experimental details are provided in Appendix C.7. Since chaotic dynamics amplify small numerical differences, trajectory-level errors are not meaningful, and we assess performance based on the qualitative spatiotemporal patterns.
this section cite: []

Section: HIGH-DIMENSIONAL PDES WITH LOW-DIMENSIONAL SOLUTION MANIFOLDS
In this benchmark (Zang et al., 2020), we solve a five-dimensional non-linear reaction-diffusion equation, where the solution only changes in two dimensions that are a priori unknown. We construct SWIM basis functions aligned with the two intrinsic dimensions of variation, directly embedding directional information unlike in PINNs and ELMs, by using spatial coordinates projected onto the gradient of the initial solution to sample SWIM basis functions, as shown in Figure 5b. See Appendix C.6 for further details.
Table 1 shows that Frozen-PINN-swim is over 3400 times faster than other PINNs at comparable low-precision accuracy. It is the only method to reach the high-precision regime, achieving 2-3 orders of magnitude higher accuracy than other PINN variants and weak adversarial networks (Zang et al., 2020). These results confirm that explicitly embedding informative basis functions yields far more efficient and accurate models than relying on iterative optimization to learn them implicitly.
this section cite: ['b124', 'b124']

Section: HIGH-DIMENSIONALITY
High-dimensional PDEs, such as the 100-dimensional heat equation, are computationally prohibitive for grid-based methods, which require more than 10 30 grid points, considering only two points per dimension. The following examples demonstrate Frozen-PINNs' ability to solve such PDEs efficiently and accurately. We evaluate our approach on two established benchmarks: one introduced in Wang & Dong (2024), which addresses the heat equation in up to 10 dimensions on a unit hypercube, and another introduced in He et al. (2023), which focuses on a 100-dimensional variant of the heat equation on a unit ball. We discuss all details in Appendix C.8.
Frozen-PINN-elm is consistently 10-1000 times more accurate than classical PINNs for up to 100dimensional PDEs Figure 7 (top), with error decaying rapidly with network width until saturation Figure 7 (bottom). For the 10-d heat equation, Frozen-PINN-elm trains 100 -1000 times faster than other PINNs while achieving higher accuracy. For the 100-d heat equation, CPU-trained Frozen-PINNs remain hundreds of times faster than GPU-trained PINNs while delivering an orderof-magnitude better accuracy (Table 1), underscoring both their computational efficiency and high accuracy. Table 2 summarizes the advantages of our algorithm over classical mesh-based and physics-informed methods based on iterative gradient-descent-based methods.
this section cite: ['b46']

Section: CONCLUSION
Frozen-PINNs directly address the longstanding training and accuracy bottlenecks of PINNs by fundamentally simplifying the optimization problem and enforcing temporal causality by construction, leveraging the idea of space-time separation. Our extensive empirical analysis reveals that Frozen-PINNs consistently realize extremely fast training and high precision (often several orders of magnitude better than SOTA PINNs), and preserve temporal causality on a broad range of PDEs involving challenges such as extreme flow velocities, long-time simulation, higher-order spatial and temporal derivatives, complicated spatial domains, non-linearities, shocks, and high-dimensionality, Within the scope of the empirical study in this work, in low dimensions, Frozen-PINNs match the accuracy of classical mesh-based solvers while retaining advantages such as mesh-free basis functions, ease of implementation, the ability to handle complex domains, spectral convergence for PDEs with smooth solutions, and scalability for high-dimensional PDEs where mesh-based approaches struggle. PDE setting IGA-FEM/ PINNs Frozen-PINNs FEM Solutions with shocks ✓ ✓ ✓ (SWIM) Complex domains mesh Easy Easy High dimensionality ✗ (CoD) ✓ ✓ Performance/features Accuracy/Precision High Often low High Speed Fast Slow (training) Fast Temporal causality ✓ ✗ (soft constraint) ✓ Table 2: Comparison of Frozen-PINNs with mesh-based FEM and classical PINNs in different problem settings presented in this paper: The comparison is grounded in results reported in Section 3 for the PDEs and solvers studied. ✓ denotes compatibility, and ✗ denotes either incompatibility or the need for substantial modifications. Curse of Dimensionality is abbreviated as CoD. 3 5 7 10 100 Dimension of the PDE 10 3 10 4 10 5 10 6 Relative 2 error PINN Frozen-PINN-elm 10 2 10 3 Width (before SVD layer) 10 5 10 3 10 1 Relative 2 error d = 3 d = 3 d = 5 d = 5 d = 7 d = 7 d = 10 d = 10 d = 100 d = 100 Limitations and future work: Our method assumes knowledge of the PDE, but its speed makes it well-suited for inverse problems via fast forward solves. While Frozen-PINNs efficiently deal with extreme temporal complexity, as shown in the advection equation with extreme flow velocities, dealing with spatial complexity encountered while solving PDEs like Navier-Stokes is an exciting next step, where one could leverage domain decomposition to deal with the added complexity (Moseley et al., 2023;Howard et al., 2024). Finally, universal approximation properties concerning specific PDE settings and understanding the role of re-sampling network parameters in overcoming the Kolmogorov n-width barrier (Peherstorfer, 2022) are some of the most challenging, yet important theoretical open areas of investigation, beyond the scope of this paper.
Frozen-PINNs take a decisive step toward practical neural PDE solvers through a lightweight optimization process and extremely fast training without GPUs, promoting low-carbon AI development (Verdecchia et al., 2023), advancing state-of-the-art performance, and establishing a formidable benchmark for the community to build upon in advancing rapid and accurate neural PDE solvers.
Reproducibility statement: The source code used to reproduce the experimental results, along with comprehensive reproducibility instructions, is included in the supplementary material and publicly available as an open-source repository. All experiments are run with multiple seeds, and the corresponding seed values are stored in the repository to ensure reproducibility.
Ethics statement: Neural networks are inherently dual-use technologies, and ethical considerations are essential for any new machine learning approach. Frozen-PINNs are grounded in classical scientific computing principles, which offer well-understood behavior and interpretability. By bridging neural PDE solvers with classical numerical methods, our framework enables clearer analysis of robustness, failure modes, and reproducibility. We believe this transparency reduces the risk of misuse and enhances controllability, making Frozen-PINNs safe and interpretable. Thus, we believe that the benefits of our approach far outweigh the potential downsides of misuse because a system that is better understood can also be controlled more straightforwardly. Physics-informed neural networks are widely used to solve PDEs with neural networks. In this work, we benchmark our approach against various PINN variants such as adaptive activation PINNs (Jagtap et al., 2020), self-adaptive PINNs (McClenny & Braga-Neto, 2023), wavelet PINNs (Uddin et al., 2023), and causal PINNs (Wang et al., 2024c), among others. For high-frequency temporal variations in the PDE solutions, Krishnapriyan et al. (2021) propose curriculum learning with gradually increasing advection coefficients. Compared to curriculum learning, our approach with space-time separation is much easier to implement, computationally efficient, and accurate, as we demonstrate in Section 3.1. Subramanian et al. (2023) propose using adaptive self-supervision of PINNs for sampling collocation points using the gradient of the loss function. We instead use the solution gradient to capture locally sharp features in the solution (see Section 3.4). Many specialized approaches based on PINNs (Cho et al., 2024;Meng et al., 2020;Sharma & Shankar, 2022;Chiu et al., 2022), methods based on hash-encoding (Huang & Alkhalifah, 2024;Wang et al., 2024a), and transfer learning (Kapoor et al., 2024b) have been proposed, but are still based on gradient-based iterative optimization and back-propagation, unlike ours.
Other recent advances of PINNs include methods that model the PDE system as pseudo-sequences. For instance, PINNsFormer employs a Transformer-based architecture that constructs pseudosequences from spatio-temporal samples and uses self-attention to model long-range temporal dependencies (Zhao et al., 2024). Another work, PINNMamba, is based on State Space Models (SSMs) and sub-sequence alignment, enabling continuous-discrete temporal modeling and improved propagation of initial-condition information (Xu et al., 2025). Although these methods model PDE systems as pseudo-sequences, these architectures often lead to more computational time and out-of-memory issues owing to their architecture, as presented by Xu et al. (2025).
Physics-informed approaches using randomized neural networks for solving PDEs have mostly been studied by combining Extreme Learning Machines (ELMs) with the self-supervised setting of PINNs (Chen et al., 2024a;Wang & Dong, 2024;Shang & Wang, 2024;Sun et al., 2024). For instance, Dwivedi & Srinivasan (2020) propose a physics-informed extreme learning machine (PIELM) to efficiently solve linear PDEs, while Calabrò et al. (2021); Galaris et al. (2022) employ ELMs to learn invariant manifolds as well as PDEs from data. Dong & Yang (2022) show that given a fixed computational budget, ELMs achieve substantially higher accuracy compared to classical second-order FEM and slightly higher accuracy compared to higher-order FEM. For static, nonlinear PDEs, ELMs can be used together with nonlinear optimization schemes (Fabiani et al., 2021). On larger spatiotemporal domains, Dong & Li (2021) and Dwivedi et al. (2021) propose using multiple distributed ELMs on multiple subdomains. Although the aforementioned methods simplify the optimization problem by randomly sampling hidden layer parameters and fixing them, they treat time as merely another spatial dimension. As a result, their neural basis functions span the full spatiotemporal domain, which limits their accuracy on PDEs exhibiting high-frequency temporal dynamics, unlike our approach.
While the problem setting is restricted to Hamiltonian systems, Rahma et al. (2024;2025) discuss how to train Hamiltonian neural networks and Hamiltonian graph neural networks using ELM and SWIM approaches, and demonstrate how random sampling can be leveraged to significantly speed up training compared to gradient-based iterative optimization. In this work, we show how random sampling can speed up training and resolve optimization challenges of PINNs for time-dependent PDEs.
Neural Galerkin schemes (Finzi et al., 2023;Aghili et al., 2024;Berman et al., 2024;Bruna et al., 2024) offer an alternative to the full spatiotemporal approach of the randomized neural networks and PINNs. These approaches treat all or sparse subsets of network parameters, beyond just the last layer's parameters, as time-dependent. This leads to a much larger system of ODEs compared to our approach. The work on neural implicit representations (Chen et al., 2023;Yin et al., 2023) also uses neural basis functions to represent only the space component, but relies on gradient-based iterative optimization via back-propagation, unlike our approach.
Spectral methods for solving PDEs promise fast convergence with much fewer basis functions. Meuris et al. (2023) present a method to extract hierarchical spatial basis functions from a trained DeepONet and employ it in a spectral method to solve the given PDE. Xia et al. (2023) integrate adaptive techniques into PINN-based PDE solvers to obtain numerical solutions of unbounded domain problems that standard PINNs cannot efficiently approximate. Lange et al. (2021) propose spectral methods that fit linear and nonlinear oscillators to data and facilitate long-term forecasting of temporal signals. Dresdner et al. (2022) demonstrate spectral solvers that provide sub-grid corrections to classical spectral methods to improve their accuracy. Du et al. (2023) use fixed orthogonal bases to learn PDE solutions as a map between spectral coefficients and introduce a training strategy based on spectral loss. These methods differ from ours in problem setting, architecture, and training.
Neural operator frameworks (Lu et al., 2021a;Kovachki et al., 2021;Li et al., 2020;Pfaff et al., 2021) are promising but are typically trained with PDE solutions with different initial conditions, spatial domains (geometries), or parameter settings. Datar et al. (2025) have demonstrated how continuous-time neural networks can be constructed for linear operator approximation for linear and time-invariant systems. Instead, in our setting here, we solve the PDE using given coefficients, domain, and initial conditions without relying on any training data. The ease of implementation, rapid training, and high accuracy of our backpropagation-free approach can be leveraged to generate PDE solution data for training operator networks.
Mesh-free methods are typically based on radial basis functions (RBFs, (Powell, 1992;Chen et al., 2014)) or Moving Least Squares (MLS) (Shepard, 1968;Lancaster & Salkauskas, 1981). These often do not have user-friendly software or are only applicable in specialized settings (e.g., smoothed particle hydrodynamics, (Lucy, 1977;Gingold & Monaghan, 1977;Shadloo et al., 2016)). Moreover, despite the ease of dealing with complicated geometries, these methods typically suffer from many challenges, such as the choice of kernel, imposing boundary conditions, and convergence issues. These methods are not the focus of this work.
Classical numerical methods such as finite elements, finite volumes, and finite differences have been used to solve PDEs for decades. They often have a rich theoretical grounding and high accuracy. Isogeometric analysis (IGA) is one such method, in which spline-based basis functions are defined over a structured grid (Hughes et al., 2005;Cottrell et al., 2009;2006). Mesh-based methods often entail a time-consuming setup phase, especially when mesh generation is challenging. Methods like sparse grids enable adaptivity through hierarchical bases but pose significant implementation challenges, particularly for irregular domains (Bungartz & Griebel, 2004). In this work, we benchmark our results against IGA and finite-element-based methods.
this section cite: ['b81', 'b47', 'b85', 'b109', 'b77', 'b107', 'b63', 'b105', 'b18', 'b79', 'b102', 'b17', 'b49', 'b125', 'b120', 'b120', 'b101', 'b106', 'b33', 'b12', 'b39', 'b28', 'b36', 'b27', 'b34', 'b92', 'b38', 'b1', 'b10', 'b14', 'b122', 'b80', 'b119', 'b67', 'b30', 'b32', 'b62', 'b70', 'b88', 'b93', 'b90', 'b15', 'b103', 'b66', 'b76', 'b41', 'b100', 'b50', 'b20', 'b11']

Section: References
Ref_id:b0 Title: TensorFlow: Large-scale machine learning on heterogeneous systems Year: (2015)
Ref_id:b1 Title: A Dynamical Neural Galerkin Scheme for Filtering Problems Year: (2024-01)
Ref_id:b2 Title: DOLFINx: the next generation FEniCS problem solving environment Year: (2023)
Ref_id:b3 Title: Paolo Orlandi, and AT0612 Patera. Spectral and finite difference solutions of the burgers equation Year: (1986)
Ref_id:b4 Title: Randomized sparse neural galerkin schemes for solving evolution equations with deep networks Year: (2024)
Ref_id:b5 Title: Neural Galerkin schemes for sequentialin-time solving of partial differential equations with deep networks Year: (2024)
Ref_id:b6 Title: Boundary element methods in dynamic analysis Year: (1987)
Ref_id:b7 Title: Sampling weights of deep neural networks Year: (2023)
Ref_id:b8 Title: A systematic literature review of burgers' equation with recent advances Year: (2018)
Ref_id:b9 Title: Reaction-diffusion equations and their applications to biology Year: (1986)
Ref_id:b10 Title: Neural Galerkin schemes with active learning for high-dimensional evolution equations Year: (2024-01)
Ref_id:b11 Title: Sparse grids. Acta numerica Year: (2004)
Ref_id:b12 Title: Extreme learning machine collocation for the numerical solution of elliptic PDEs with sharp gradients Year: (2021-12)
Ref_id:b13 Title: Optimization of random feature method in the high-precision regime Year: (2024)
Ref_id:b14 Title: CROM: Continuous reduced-order modeling of PDEs using implicit neural representations Year: (2023)
Ref_id:b15 Title: Recent Advances in Radial Basis Function Collocation Methods Year: (2014)
Ref_id:b16 Title: Self-adaptive weights based on balanced residual decay rate for physics-informed neural networks and deep operator networks Year: (2024)
Ref_id:b17 Title: Can-pinn: A fast physics-informed neural network based on coupled-automatic-numerical differentiation method Year: (2022)
Ref_id:b18 Title: Separable physics-informed neural networks Year: (2024)
Ref_id:b19 Title: Diffusion maps Year: (2006)
Ref_id:b20 Title: Isogeometric Analysis: Toward Integration of CAD and FEA Year: (2009)
Ref_id:b21 Title: Isogeometric analysis of structural vibrations Year: (2006)
Ref_id:b22 Title: The Numerical Evaluation of B-Splines* Year: (1972)
Ref_id:b23 Title: Scientific machine learning through physics-informed neural networks: Where we are and what's next Year: (2022)
Ref_id:b24 Title: Systematic construction of continuous-time neural networks for linear dynamical systems Year: ()
Ref_id:b25 Title: On calculating with b-splines Year: (1972)
Ref_id:b26 Title: Neural-network-based approximations for solving partial differential equations Year: (1994)
Ref_id:b27 Title: Local extreme learning machines and domain decomposition for solving linear and nonlinear partial differential equations Year: (2021)
Ref_id:b28 Title: On computing the hyperparameter of extreme learning machines: Algorithm and application to computational pdes, and comparison with classical and high-order finite elements Year: (2022)
Ref_id:b29 Title: A family of embedded runge-kutta formulae Year: (1980)
Ref_id:b30 Title: Learning to correct spectral methods for simulating turbulent flows Year: (2022)
Ref_id:b31 Title: Evolutional deep neural network Year: (2021)
Ref_id:b32 Title: Neural spectral methods: Self-supervised learning in the spectral domain Year: (2023)
Ref_id:b33 Title: Physics informed extreme learning machine (pielm)-a rapid method for the numerical solution of partial differential equations Year: (2020)
Ref_id:b34 Title: Distributed learning machines for solving forward and inverse problems in partial differential equations Year: (2021)
Ref_id:b35 Title: Towards a Mathematical Understanding of Neural Network-Based Machine Learning: What We Know and What We Don't Year: (2020-06)
Ref_id:b36 Title: Numerical solution and bifurcation analysis of nonlinear partial differential equations with extreme learning machines Year: (2021-11)
Ref_id:b37 Title: Partial Differential Equations for Scientists and Engineers Year: (1993)
Ref_id:b38 Title: A stable and scalable method for solving initial value pdes with neural networks Year: (2023)
Ref_id:b39 Title: Numerical Bifurcation Analysis of PDEs From Lattice Boltzmann Model Simulations: A Parsimonious Machine Learning Approach Year: (2022-08)
Ref_id:b40 Title: Gmsh: A 3-d finite element mesh generator with built-in pre-and post-processing facilities Year: (2009-09)
Ref_id:b41 Title: Smoothed particle hydrodynamics: theory and application to non-spherical stars Year: (1977)
Ref_id:b42 Title: Understanding the difficulty of training deep feedforward neural networks Year: (2010)
Ref_id:b43 Title: On the gibbs phenomenon and its resolution Year: (1997)
Ref_id:b44 Title: Solving high-dimensional partial differential equations using deep learning Year: (2018)
Ref_id:b45 Title: Pinnacle: a comprehensive benchmark of physics-informed neural networks for solving pdes Year: (2024)
Ref_id:b46 Title: Learning physics-informed neural networks without stacked back-propagation Year: (2023)
Ref_id:b47 Title: Finite basis kolmogorov-arnold networks: domain decomposition for data-driven and physics-informed problems Year: (2024)
Ref_id:b48 Title: Extreme learning machine: theory and applications Year: (2006)
Ref_id:b49 Title: Efficient physics-informed neural networks using hash encoding Year: (2024)
Ref_id:b50 Title: Isogeometric analysis: Cad, finite elements, nurbs, exact geometry and mesh refinement Year: (2005)
Ref_id:b51 Title: Viscous fingering in multiport hele shaw cell for controlled shaping of fluids Year: (2017)
Ref_id:b52 Title: Adaptive activation functions accelerate convergence in deep and physics-informed neural networks Year: (2020)
Ref_id:b53 Title: Applied diffusion processes from engineering to finance Year: (2013)
Ref_id:b54 Title: Ceens: Causality-enforced evolutional networks for solving time-dependent partial differential equations Year: (2024)
Ref_id:b55 Title: Physics-informed neural networks for solving forward and inverse problems in complex beam systems Year: (2023)
Ref_id:b56 Title: Neural oscillators for generalization of physics-informed machine learning Year: (2024)
Ref_id:b57 Title: Transfer learning for improved generalizability in causal physics-informed neural networks for beam simulations Year: (2024)
Ref_id:b58 Title: Physics-informed machine learning Year: (2021)
Ref_id:b59 Title: Positional embeddings for solving pdes with evolutional deep neural networks Year: (2024)
Ref_id:b60 Title: Causality-aware training of physics-informed neural networks for solving inverse problems Year: (2025)
Ref_id:b61 Title: Which optimizer works best for physics-informed neural networks and kolmogorov-arnold networks? arXiv preprint Year: (2025)
Ref_id:b62 Title: Neural operator: Learning maps between function spaces Year: (2021)
Ref_id:b63 Title: Characterizing possible failure modes in physics-informed neural networks Year: (2021)
Ref_id:b64 Title: Artificial neural networks for solving ordinary and partial differential equations Year: (1998)
Ref_id:b65 Title: Introduction to reaction-diffusion equations: Theory and applications to spatial ecology and evolutionary biology Year: (2022)
Ref_id:b66 Title: Surfaces generated by moving least squares methods Year: (1981)
Ref_id:b67 Title: From fourier to koopman: Spectral methods for long-term time series prediction Year: (2021)
Ref_id:b68 Title:  Year: (2001)
Ref_id:b69 Title: Causality-enhanced discreted physics-informed neural networks for predicting evolutionary equations Year: (2024)
Ref_id:b70 Title: Fourier neural operator for parametric partial differential equations Year: (2020)
Ref_id:b71 Title: Pderefiner: Achieving accurate long rollouts with neural pde solvers Year: (2024)
Ref_id:b72 Title: Config: Towards conflict-free training of physics informed neural networks Year: (2024)
Ref_id:b73 Title: Learning nonlinear operators via deeponet based on the universal approximation theorem of operators Year: (2021)
Ref_id:b74 Title: DeepXDE: A deep learning library for solving differential equations Year: (2021)
Ref_id:b75 Title: Physics-informed neural networks with hard constraints for inverse design Year: ()
Ref_id:b76 Title: A numerical approach to the testing of the fission hypothesis Year: (1977-12)
Ref_id:b77 Title: Self-adaptive physics-informed neural networks Year: (2023)
Ref_id:b78 Title: Nick McGreivy and Ammar Hakim. Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations Year: (2023)
Ref_id:b79 Title: Ppinn: Parareal physicsinformed neural network for time-dependent pdes Year: (2020)
Ref_id:b80 Title: Machine-learning-based spectral methods for partial differential equations Year: (2023)
Ref_id:b81 Title: Finite basis physics-informed neural networks (fbpinns): a scalable domain decomposition approach for solving differential equations Year: (2023)
Ref_id:b82 Title: Achieving high accuracy with pinns via energy natural gradient descent Year: (2023)
Ref_id:b83 Title: Schrödinger equations and diffusion theory Year: (2012)
Ref_id:b84 Title:  Year: (2017)
Ref_id:b85 Title: Breaking the kolmogorov barrier with nonlinear model reduction Year: (2022)
Ref_id:b86 Title: A unified scalable framework for causal sweeping strategies for physics-informed neural networks (pinns) and their temporal decompositions Year: (2023)
Ref_id:b87 Title: Automatic selection of methods for solving stiff and nonstiff systems of ordinary differential equations Year: (1983)
Ref_id:b88 Title: Learning mesh-based simulation with graph networks Year: (2021)
Ref_id:b89 Title: The NURBS book Year: (1997)
Ref_id:b90 Title: The Theory of Radial Basis Function Approximation in 1990 Year: (1992-04)
Ref_id:b91 Title: A priori error estimate for the baumann-oden version of the discontinuous galerkin method Year: (2001)
Ref_id:b92 Title: Training hamiltonian neural networks without backpropagation Year: (2024)
Ref_id:b93 Title: Rapid training of hamiltonian graph networks without gradient descent Year: (2025)
Ref_id:b94 Title: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations Year: (2019)
Ref_id:b95 Title: Challenges in training pinns: A loss landscape perspective Year: (2024)
Ref_id:b96 Title: Numerical advection algorithms and their role in atmospheric transport and chemistry models Year: (1987)
Ref_id:b97 Title: Generalization Properties of Learning with Random Features Year: (2021-04)
Ref_id:b98 Title: Learning representations by back-propagating errors Year: (1986)
Ref_id:b99 Title: Geometric partial differential equations and image analysis Year: (2001)
Ref_id:b100 Title: Smoothed particle hydrodynamics method for fluid flows, towards industrial applications: Motivations, current state, and challenges Year: (2016)
Ref_id:b101 Title: Randomized Neural Networks with Petrov-Galerkin Methods for Solving Linear Elasticity and Navier-Stokes Equations Year: (2024-04)
Ref_id:b102 Title: Accelerated training of physics-informed neural networks (pinns) using meshless discretizations Year: (2022)
Ref_id:b103 Title: A two-dimensional interpolation function for irregularly-spaced data Year: (1968)
Ref_id:b104 Title: Dgm: A deep learning algorithm for solving partial differential equations Year: (2018)
Ref_id:b105 Title: Adaptive selfsupervision algorithms for physics-informed neural networks Year: (2023)
Ref_id:b106 Title: Local randomized neural networks with discontinuous Galerkin methods for partial differential equations Year: (2024-08)
Ref_id:b107 Title: Wavelets based physics informed neural networks to solve non-linear differential equations Year: (2023)
Ref_id:b108 Title: Unveiling the optimization process of physics informed neural networks: How accurate and competitive can pinns be Year: (2025)
Ref_id:b109 Title: A systematic review of green ai Year: (2023)
Ref_id:b110 Title: Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python Year: (2020)
Ref_id:b111 Title: Neural physical simulation with multi-resolution hash grid encoding Year: (2024)
Ref_id:b112 Title: Understanding and mitigating gradient flow pathologies in physics-informed neural networks Year: (2021)
Ref_id:b113 Title: When and why pinns fail to train: A neural tangent kernel perspective Year: (2022)
Ref_id:b114 Title: Piratenets: Physics-informed deep learning with residual adaptive networks Year: (2024)
Ref_id:b115 Title: Respecting causality for training physicsinformed neural networks Year: (2024)
Ref_id:b116 Title: Respecting causality for training physicsinformed neural networks Year: (2024)
Ref_id:b117 Title: An extreme learning machine-based method for computational pdes in higher dimensions Year: (2024)
Ref_id:b118 Title: A Spectral-Based Analysis of the Separation between Two-Layer Neural Networks and Linear Methods Year: (2022-01)
Ref_id:b119 Title: Spectrally adapted physics-informed neural networks for solving unbounded domain problems Year: (2023)
Ref_id:b120 Title: Sub-sequential physics-informed learning with state space model Year: (2025)
Ref_id:b121 Title: Multiadam: Parameterwise scale-invariant optimizer for multiscale training of physics-informed neural networks Year: (2023)
Ref_id:b122 Title: Continuous pde dynamics forecasting with implicit neural representations Year: (2023)
Ref_id:b123 Title: Gradient-enhanced physics-informed neural networks for forward and inverse pde problems Year: (2022)
Ref_id:b124 Title: Weak adversarial networks for highdimensional partial differential equations Year: (2020)
Ref_id:b125 Title: Pinnsformer: A transformer-based framework for physics-informed neural networks Year: (2024)
Ref_id:b126 Title:  Year: ()
Ref_id:b127 Title:  Year: ()
Ref_id:b128 Title:  Year: ()
Ref_id:b129 Title:  Year: ()
Ref_id:b130 Title:  Year: ()
Ref_id:b131 Title:  Year: ()
