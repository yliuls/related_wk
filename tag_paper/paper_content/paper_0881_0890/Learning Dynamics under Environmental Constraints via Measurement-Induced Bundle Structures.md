Title: Learning Dynamics under Environmental Constraints via Measurement-Induced Bundle Structures
Abstract: Learning unknown dynamics under environmental (or external) constraints is fundamental to many fields (e.g., modern robotics), particularly challenging when constraint information is only locally available and uncertain. Existing approaches requiring global constraints or using probabilistic filtering fail to fully exploit the geometric structure inherent in local measurements (by using, e.g., sensors) and constraints. This paper presents a geometric framework unifying measurements, constraints, and dynamics learning through a fiber bundle structure over the state space. This naturally induced geometric structure enables measurement-aware Control Barrier Functions that adapt to local sensing (or measurement) conditions. By integrating Neural ODEs, our framework learns continuous-time dynamics while preserving geometric constraints, with theoretical guarantees of learning convergence and constraint satisfaction dependent on sensing quality. The geometric framework not only enables efficient dynamics learning but also suggests promising directions for integration with reinforcement learning approaches. Extensive simulations demonstrate significant improvements in both learning efficiency and constraint satisfaction over traditional methods, especially under limited and uncertain sensing conditions.

Section: Introduction
Learning unknown dynamics under measurement constraints is fundamental to many applications, from manipulators with force sensing to autonomous vehicles with range detection. Control barrier functions (CBFs) (Ames et al., 2016) have emerged as a powerful tool for ensuring constraint satisfaction. However, the classical CBF framework treats measurements as external observations rather than integral components of the system's geometric structure, limiting the ability to fully exploit measurement information for both constraint verification and dynamics learning.
The fundamental challenge lies in the geometric relationship between state space, measurements, and constraint manifolds. Traditional approaches often require complete knowledge of constraints, making them impractical when only local sensors' measurements are available. Consider a robotic arm with force sensors or a drone with range detectors -even local measurements contain sufficient geometric information about both constraints and underlying dynamics, suggesting global knowledge may be unnecessary if we properly exploit this local structure.
Our key insight is that measurement uncertainty naturally induces a fiber bundle structure that unifies measurements, dynamics, and constraints. This geometric perspective reveals how measurements and dynamics are fundamentally intertwined through the bundle's connection, enabling Neural ODEs (Chen et al., 2018) to learn continuous-time dynamics that naturally respect the system's physical behavior. By leveraging this structure, our approach provides a new paradigm for machine learning to understand environmental dynamics -instead of treating measurements as simple inputs, we exploit their inherent geometric information to guide the learning process. The framework allows control strategies to automatically adapt based on measurement quality -becoming more conservative in regions of high uncertainty while allowing more aggressive behavior where measurements are reliable.
The main contributions of this work are: 1) Proposes a novel geometric framework that unifies measurement uncertainty, system dynamics, and constraints within a fiber bundle structure, providing principled information for Neural ODEs while maintaining safety guarantees. 2) Introduces adaptive measurement-aware safety certificates (mCBFs, defined in Section 3.6) that automatically adjust conservative margins based on local measurement quality. 3) Demonstrates enhanced generalization capabilities across different scenarios without requiring global information through experimental validation.
The practical significance lies in learning safely from local measurements without complete constraint knowledge. By working directly with the bundle structure induced by measurements, we prove that Neural ODEs trained within this framework naturally preserve physical constraints while learning continuous-time dynamics. This geometric approach provides machine learning algorithms with a structured way to understand system dynamics through the lens of measurement geometry, leading to more efficient and interpretable learning. Moreover, this framework offers important insights for reinforcement learning by providing a principled way to handle partial observations and measurement uncertainties in the learning process. Experimental results demonstrate the effectiveness in real-world applications, where the learned dynamics model successfully captures both the control-to-trajectory relationships and constraint requirements under measurement uncertainties. Our implementation is publicly available at https://gith  ub.com/ContinuumCoder/Measurement-Induc  ed-Bundle-for-Learning-Dynamics/.
this section cite: ['b0', 'b7']

Section: Related Work

this section cite: []

Section: Safety-Critical Control and Learning
Our work builds upon fundamental theoretical advances in differential geometry and control theory. The fiber bundle framework we employ originates from Ehresmann's seminal work (Ehresmann, 1950) on geometric connections, later developed by (Kobayashi & Nomizu, 1996) for control applications. Early work in safety-critical control focused on analytical safety certificates through Control Barrier Functions (CBFs). (Ames et al., 2019) introduced the foundational CBF framework providing formal guarantees for constraint satisfaction in known dynamical systems, extended by (Jankovic, 2018;Das ¸& Murray, 2022;Choi et al., 2021) to handle bounded disturbances through robust CBFs. Learning-based approaches emerged to address model uncertainty while maintaining safety guarantees. (Cheng et al., 2019) proposed Neural CBFs that learn safety certificates directly from data, while (Taylor et al., 2020) developed the SafeLearn framework combining Gaussian processes with CBFs for safe exploration. However, these methods treat measurements as perfect observations rather than uncertain quantities, limiting their real-world applicability.
this section cite: ['b14', 'b20', 'b1', 'b18', 'b11', 'b9', 'b8']

Section: Geometric Learning and Bundle Theory
Geometric structure preservation in learning control has seen significant development, building on (Marsden & Weinstein, 1974)'s theoretical foundation for geometric mechanics and symmetry reduction. (Ratliff et al., 2018) intro-duced Riemannian Motion Policies respecting the underlying manifold structure, while (Chen et al., 2018) proposed Neural ODEs that opened new possibilities for learning dynamics with geometric properties. The bundle-theoretic perspective was pioneered by (Lewis, 1998) establishing connections between mechanical systems and principal bundles, with (Montgomery, 1993) developing gauge-theoretic approaches to mechanical control. Building on this foundation, (Bronstein et al., 2017;Cohen et al., 2019) advanced geometric methods for learning on manifolds, and (Hansen-Estruch et al., 2021) developed frameworks for control on Lie groups. However, these approaches typically require global geometric information and struggle with local measurement uncertainty.
this section cite: ['b25', 'b27', 'b7', 'b22', 'b26', 'b6', 'b10', 'b16']

Section: Measurement-Aware Control
The geometric treatment of measurement uncertainty draws inspiration from (Hsu, 2002) on stochastic differential geometry and (Diaconis et al., 1988) on geometric filtering theory. Early approaches like (Kalman, 1960) developed robust control using filtering-based state estimation, while (Berkenkamp et al., 2017) proposed learning control under measurement noise. Recent advances by (Wu et al., 2015) introduced geometric numerical integration methods for uncertainty quantification, while (Boumal, 2023) developed comprehensive tools for optimization and estimation on manifolds. However, these works often treat measurement uncertainty as an external disturbance rather than an intrinsic geometric property of the system. Our framework addresses these limitations by unifying measurements, constraints, and learning objectives through the natural fiber bundle structure induced by the measurement process, enabling more efficient learning while maintaining rigorous safety guarantees.
this section cite: ['b17', 'b12', 'b19', 'b2', 'b33', 'b5']

Section: Theoretical Foundations and System Modeling
Through fiber bundle structures and measurement-adapted barrier functions, this work establishes a geometric framework that unifies the challenge of maintaining safety guarantees while adapting to uncertainties in both system dynamics and measurements in safe learning control.
this section cite: []

Section: System Model and Measurement Structure
Let T x M denote the tangent space at point x ∈ M, which is the vector space of all tangent vectors at x, and T M = x∈M T x M be the tangent bundle. Let T * x M denote the cotangent space at x and T * M = x∈M T *
x M represents the cotangent bundle.
Consider a controlled dynamical system with state x ∈ M on a smooth manifold M and control input u ∈ U ⊆ R m . The system evolution and measurement process are described by
ẋ = f (x, u) + g(x)w, x(0) = x 0 y = h(x) + v(1)
where f : M × U → T M represents the nominal dynamics, g : M → T * M characterizes model uncertainty, and h : M → Y is the measurement map. The process noise w and sub-Gaussian measurement noise v are bounded with |w| ≤ δ w and |v| ≤ δ v , respectively.
The measurement space Y ⊆ R k carries a natural metric structure induced by the Euclidean norm:
d Y (y 1 , y 2 ) = k i=1 (y 1,i -y 2,i ) 2
. This metric quantifies the uncertainty in measurement space and plays a crucial role in safety analysis.
this section cite: []

Section: Fiber Bundle Framework
The relationship between states and their measurements induces a natural fiber bundle structure π : E → M where E = M × Y is the total space. For each state x ∈ M, the fiber π -1 (x) characterizes the set of possible measurements:
π -1 (x) = {(x, y) ∈ E : y = h(x) + v, ∥v∥ ≤ δ v } (2)
This bundle is equipped with a connection ∇ that describes the geometric relationship between system trajectories and measurement evolution:
∇ X Y = π -1 * (∇ π * X (π * Y )) + K(x) y -h(x)(3)
where ∇ X Y represents the covariant derivative of the vector field Y ∈ X(E) along the vector field X ∈ X(E) (here, X(E) denotes the space of smooth vector field on E), π * refers to the pushforward map of the projection π, K : M → L(Y, T M) is the measurement feedback gain operator that couples state and measurement dynamics. Here, L(Y, T M) denotes the space of bounded linear operators from Y to T M.
this section cite: []

Section: Bundle-Based Safety Certificates
Safety constraints are formalized through a smooth bundle map Φ : E → R over the fiber bundle π : E → M satisfying three fundamental properties:
1. Φ(x, h(x)) > 0 for all x ∈ S 0 2. The bundle derivative dΦ(X) > 0 for all X ∈ A(E),
where dΦ(X) := ⟨∇ E Φ, X⟩ 3. Φ(x, y) ≥ γ(∥y -h(x)∥) for some γ ∈ K ∞ (4) Here, S 0 denotes the nominal safe set, A(E) is the space of admissible vectors on the total space E, ∇ E is the covariant derivative on E, and γ is a class K ∞ function that captures the degradation of safety guarantees with measurement uncertainty.
this section cite: []

Section: Uncertainty Propagation
The propagation of uncertainties through the bundle structure follows from the differential geometry of the fiber bundle. The key relationships are:
dπ * (X f ) = f (x, u) dπ * (X g ) = g(x)w dy = dh(x) + dv (5)
where dπ * represents the pushforward of the vector fields X f , X g along the projection π. These relationships induce an uncertainty tube T ε (x, t) around nominal trajectories:
T ε (x, t) = {y : d Y (y, h(ϕ t (x))) ≤ ε(t)} (6
)
where ϕ t denotes the flow of the nominal system and ε(t) characterizes the growth of uncertainty over time.
this section cite: []

Section: Compatible Group Actions
The system often exhibits symmetries that can be exploited for, for example, dimensional reduction. These symmetries are captured by compatible Lie group actions. Let G be a Lie group acting on both M and Y through smooth maps.
Then,
Ψ M : G × M → M Ψ Y : G × Y → Y (7
)
The compatibility conditions for these actions are
f (Ψ M (g, x), u) = dΨ M (g, •)f (x, u) h(Ψ M (g, x)) = Ψ Y (g, h(x))(8)
for all g ∈ G, where dΨ M (g, •) represents the differential of the map Ψ M (g, •) with respect to the state. These conditions ensure that the symmetries respect both the dynamics and measurements.
this section cite: []

Section: Measurement-Adapted Control Barrier Functions
The cornerstone of our safety framework is the concept of measurement-adapted Control Barrier Functions (mCBFs).
A smooth function b : E → R qualifies as an mCBF if it satisfies
1. b(x, y) ≥ 0 =⇒ x ∈ S 0 2. inf u∈U L f b + (L g b)w + α(b) ≥ 0 3. |b(x, y 1 ) -b(x, y 2 )| ≤ L b d Y (y 1 , y 2 )(9)
where L f and L g denote the Lie derivatives along vector fields f and g respectively, α is a class K ∞ function, and L b > 0 is the Lipschitz constant of b with respect to measurements (recall that S 0 denotes the nominal safe set).
this section cite: []

Section: Safety Guarantees
The culmination of this geometric framework is captured in the following fundamental theorem:
Theorem 3.1. Given an mCBF b satisfying the preceding conditions, if b(x(0), y(0)) ≥ 0, then for any admissible noise sequences w(•), v(•):
P(x(t) ∈ S 0 for all t ≥ 0) ≥ 1 -exp(-c/δ 2 v )(10)
where c > 0 is a constant depending on system parameters.
Proof Sketch. The proof proceeds through three key steps. First, we establish that the bundle connection preserves safety certificates along fibers, utilizing the compatibility conditions between the connection and barrier function. Second, we demonstrate that uncertainty propagation remains bounded within the tube T ε , leveraging the Lipschitz properties of the system dynamics. Finally, we show that the Lipschitz condition on b ensures controlled variation of safety certificates under measurement uncertainty, leading to the probabilistic bound (1 -exp(-c/δ v )). The technical proof is provided in Appendix A.
The geometric framework developed in this section establishes three key theoretical advances. First, the fiber bundle structure provides a natural setting for handling measurement uncertainty, enabling precise tracking of error propagation through system dynamics. Second, the compatible group actions facilitate systematic dimension reduction while preserving safety properties through quotient space dynamics. Third, the measurement-adapted Control Barrier Functions yield robust safety guarantees that degrade gracefully with measurement noise, thanks to their Lipschitz continuity properties.
This theoretical foundation directly enables practical learning algorithms that maintain safety under realistic sensing conditions, as we will demonstrate in the subsequent section.
this section cite: []

Section: Learning Framework under Measurement Uncertainties
Building on the geometric foundations, we now develop a learning framework that actively incorporates measurement uncertainty. The key idea is to learn both the dynamics and safety certificates on the bundle E.
this section cite: []

Section: Bundle-Valued Learning Operators
Define the bundle-valued learning operator L :
C ∞ (E) → Γ(T E): L(Φ)(x, y) = ∇ E Φ(x, y) + λR(x, y)(11)
where C ∞ (E) denotes the space of smooth functions defined on the total space E, Γ(T E) stands for the space of sections of the tangent bundle T E, L(Φ)(x, y) represents the operator L acting on Φ, evaluated at the point (x, y), ∇ E is the connection defined on E, and R provides a regularization function preserving the fiber structure of the bundle.
The learning dynamics on the bundle take the form
ḟ = -L 1 ( f -f ) Φ = -L 2 (Φ -Φ * )(12)
where L 1 , L 2 are compatible bundle-valued operators. f denotes the learned estimate of the true system dynamics f , while Φ * represents the optimal barrier function that ensures safety guarantees, both serving as target values in the learning dynamics governed by bundle-valued operators.
this section cite: []

Section: Measurement-Adapted Safety Certificates
Let Φ 0 : E → R denote the nominal safety certificate that characterizes system safety under ideal measurements. The safety certificate Φ : E → R adapts to measurement uncertainty through:
Φ(x, y) = Φ 0 (x, y) -α(∥y -h(x)∥) L f Φ + (L g Φ)w ≥ -β(Φ) along solutions (13
)
where Φ 0 is the nominal certificate, α ∈ K ∞ , and β is a class K function.
this section cite: []

Section: Uncertainty-Aware Learning Algorithm
The learning process incorporates measurement uncertainty through:
θ = -Λ∇ θ T ( fθ , D) T ( f , D) = N i=1 ∥ f (x i , u i ) -ẋi ∥ 2 Σ -1 i (14
)
where Σ i captures measurement uncertainty in data point i.
The learning rate matrix Λ guides parameter updates, while
| • | 2 Σ -1 i
denotes the uncertainty-weighted norm using the inverse covariance matrix Σ -1 i , and D contains N triplets of state, input, and state derivative measurements.
this section cite: []

Section: Safety-Constrained Policy Updates
Let Θ ∈ R d denote the parameters of a policy π Θ : M → U that maps state to control inputs. The policy update law preserves safety through:
Θ = Π S [-∇ Θ J(Θ)] S = {Θ : Φ(x, y) ≥ 0 for all (x, y) ∈ E}(15)
where Π S denotes projection onto the safe policy set S, and J(Θ) = Ex 0 [ ∞ t=0 γ t c(x t , Θ(x t ))] represents the expected discounted cumulative cost under policy Θ, with immediate cost c(x, Θ(x)), discount factor γ ∈ (0, 1), and initial state distribution x 0 .
this section cite: []

Section: Convergence and Safety Guarantees
Building on Theorem 3.1, we establish the convergence properties of our learning framework: Theorem 4.1. Under the proposed learning dynamics, we have
∥ f -f ∥ E ≤ c 1 exp(-λ 1 t) + c 2 δ v P(x(t) ∈ S 0 ) ≥ 1 -exp(-c 3 /δ 2 v )(16)
where c 1 , c 2 , c 3 , λ 1 > 0 are constants.
A detailed proof of Theorem 4.1 is in Appendix B. The first inequality shows exponential convergence of the learned dynamics with a residual error bounded by measurement uncertainty, while the second preserves the safety guarantees during learning.
this section cite: []

Section: Experimental Design
We design a comprehensive experimental framework to evaluate our proposed method against state-of-the-art approaches, focusing on three interconnected research directions: Learning-based safety control, geometric structure learning, and safe control under uncertainty. The experiments are constructed to highlight key methodological differences while ensuring fair comparison through standardized implementations and evaluation protocols.
this section cite: []

Section: Baseline Methods
For detailed discussions on comparisons with mainstream advanced manifold learning methods, we refer readers to Appendix F. While direct numerical comparisons with recent geometric deep learning approaches may seem natural, there are several fundamental differences that make such direct benchmarking potentially misleading. We evaluate our method against state-of-the-art approaches spanning different technical directions in safe learning control. Our baseline selection aims to comprehensively compare with methods that address various aspects of our proposed framework:
Learning-based Safety Certification: We implement Neural-CBF (Liu et al., 2023), BayesSafe (Berkenkamp et al., 2023), and StructCBF (Taylor et al., 2020) as fundamental approaches using neural networks and Bayesian optimization for safety certification. Recent advances like SafetyNet (Vitelli et al., 2022) and SafeTrack (Li et al., 2024) enhance these guarantees through adaptive barriers and system-level guards, though they still lack explicit handling of measurement uncertainty. Physics-Informed and Geometric Methods: To evaluate our physical consistency, we compare against PNDS (Djeumou et al., 2022) and GEM (Hansen-Estruch et al., 2021), which encode physical laws through specialized neural architectures. We also include GeoPath (Zhang et al., 2015) that leverages geometric principles for control design, though without addressing measurement uncertainty. Robust and Adaptive Control: Several approaches address system robustness through different theoretical frameworks. RobustSafe (Gurriet et al., 2020) provides safety-critical control using fixed uncertainty bounds, while DataFilter (Wabersich et al., 2023) leverages data-driven safety filters for handling uncertainties, and AdaptSafe (Taylor & Ames, 2020) introduces measurementdependent barrier functions. Uncertainty-Aware Predictive Control: For handling uncertain dynamics, GPMPC (Bonzanini et al., 2021) and ALMPC (Saviolo et al., 2023) employ Gaussian processes and active learning for uncertainty quantification. SafeRL (Cheng et al., 2019) combines reinforcement learning with barrier functions, demonstrating strong performance in handling model uncertainty through probabilistic frameworks, though these methods lack formal geometric safety certificates.
this section cite: ['b24', 'b3', 'b31', 'b23', 'b13', 'b16', 'b15', 'b32', 'b4', 'b28', 'b8']

Section: Experimental Tasks
We implement three tasks in a simulation environment built on Genesis physics engine (Xian et al., 2023). The first task examines a soft-body worm robot (0.1m per segment) navigating through obstacles to reach a target, using a fixedstep forward Euler integrator (dt = 5e-4s). The worm is modeled using the Material Point Method (MPM) with the neo-Hookean material model. Let x ∈ R 3 denote the position field and ρ the material density, the dynamics follow:
ρẍ = ∇ • P + b (17
)
where P is the first Piola-Kirchhoff stress tensor and b represents body forces. For neo-Hookean materials:
P = µ(F -F -T ) + λ log(J)F -T (18
)
Here F is the deformation gradient, J = det(F), and material parameters µ, λ are unknown.
The control input consists of four muscle actuation signals u = [u uf , u uh , u lf , u lh ] ⊤ ∈ [0, 1] 4 , where subscripts indicate upper-fore, upper-hind, lower-fore, and lower-hind muscle groups respectively. The system must maintain safe distances from obstacles through constraints h i (x) = ∥xx obs,i ∥ 2 -r saf e ≥ 0 for all obstacles i = 1, . . . , N obs . Six visual sensors (two on head/tail, four on sides) provide local measurements y i = [x -x obs , ∥x -x obs ∥ 2 ] ⊤ + v i with uncertainty bound ∥v i ∥ ≤ α∥x -x i ∥ increasing with distance from sensor location x i .
The second task involves a 7-DOF Franka arm performing obstacle-aware manipulation, integrated with dt = 1e-2s. Let q ∈ R 7 denote the joint angles and M(q) the inertia matrix, the system dynamics with control input τ follow:
M(q)q + C(q, q) q + g(q) + f ( q) = τ(19)
where C(q, q) represents Coriolis terms, g(q) gravity, f ( q) joint friction, and τ control torques. The system operates under joint limits h q (q) = q max -|q| ≥ 0 and obstacle avoidance constraints h o (q) = ∥p ee (q) -p obs ∥ 2 -d saf e ≥ 0, where p ee (q) and p obs denote the end-effector and obstacle positions, respectively.
The third task features a quadrotor drone navigating through 3D space, integrated with dt = 2e-3(sec). Let p ∈ R 3 denote position, v ∈ R 3 velocity, R ∈ SO(3) orientation, and ω ∈ R 3 angular velocity, the dynamics are described by
    ṗ v Ṙ ω    =     v 1 m Rf -ge 3 -D(v) R[ω] × J -1 (τ -ω × Jω)     (20
)
where m is mass, g unknown gravity, D(v) unknown aerodynamic drag, J inertia matrix, and  For all three tasks, the workspace is configured as a 2m × 2m × 2m arena with randomly placed obstacles. The obstacles' positions are sampled uniformly within the workspace while maintaining minimum separation distances. Initial and goal states are sampled to ensure feasible paths exist while providing sufficient challenge for evaluating the learning and control performance.
this section cite: []

Section: Implementation Details
All experiments are implemented in Python using PyTorch, with Soft Actor-Critic (SAC) as our base reinforcement learning framework. For fair comparisons, we maintain the original implementations for model-based baselines (GPMPC, RobustSafe, ALMPC) and learning-based baselines (Neural-CBF, SafetyNet, DataFilter). Our neural architectures use three hidden layers (128-64-32 units) with ReLU activations, while barrier functions add a tanh activation in the output layer for boundedness. All networks are trained with Adam optimizer using mixed precision training on an NVIDIA RTX 3090 GPU. Detailed implementation specifications, including hyperparameters, network architectures, and optimization techniques, are provided in Appendix C.
For the soft worm task, uncertainties include MPM material parameters (µ, λ variations of ±10%), actuation response (±5% muscle force scaling), and sensor noise proportional to distance (0.5-2% of measured distance). The Franka arm experiments incorporate joint friction variations (±8%), payload changes (0-200g), and measurement uncertainties in joint angles (±0.02 rad) and end-effector pose (±2cm). The quadrotor tests feature mass variations (±5%), aerodynamic disturbances (up to 0.2N), and depth measurement noise scaling with distance (1-3% of measured depth).
this section cite: []

Section: Evaluation Metrics and Protocol
We evaluate both motion quality and safety performance using comprehensive metrics including success rate (SR), path efficiency measures, safety margins, and control quality indicators. Detailed definitions and calculations of these metrics are provided in Appendix C.1. All experiments are conducted in a 2m × 2m × 2m workspace with randomly generated start/goal positions and obstacle placements. We test the worm robot (500 trials), Franka arm (400 trials), and quadrotor (300 trials) under various task scenarios. Complete experimental settings and success criteria are detailed in Appendix C.2.
this section cite: []

Section: Results and Analysis
We evaluate our method across three robotic control tasks to demonstrate generalization of safety constraints under novel obstacle configurations with local observations. Table 1 presents the quantitative results.
While learning-based safety approaches achieve limited success rates (82%-86%) with fixed barrier functions that cannot adapt to new configurations, and physics-informed methods maintain geometric properties but achieve only 73%-76% success in dynamic environments, our method's measurement-induced bundle structure enables 96.3% success rate. Traditional uncertainty-aware MPC methods achieve high constraint satisfaction (99.7%) but produce overly conservative trajectories (26-27m vs our 18.5m), lacking our geometric framework for measurement uncertainty.
The key advantage of our approach lies in the unified geometric treatment of measurement uncertainty and safety constraints. Unlike recent adaptive methods that handle uncertainty estimation and safety certification separately (88%-89% success), our framework enables simultaneous adaptation of safety bounds and uncertainty estimation through the fiber bundle structure. This fundamental integration of measurement uncertainty into the geometric safety constraints allows our method to achieve superior performance (96.3% success, 18.5m paths with 99.3% constraint satisfaction) while maintaining robust safety guarantees across novel environments.
this section cite: []

Section: Performance Convergence Analysis
Figure 2 showcases the training convergence trends of our proposed method in comparison with selected baseline approaches across three distinct tasks: Soft Worm Peristaltic Navigation, Franka Arm Joint Motion, and Quadrotor Propeller Control. Across all tasks, our method demonstrates a significantly faster convergence rate, reaching optimal performance metrics within fewer training episodes than the baseline methods. Additionally, the shaded regions representing standard deviation are noticeably narrower for our method, indicating reduced performance variance and enhanced stability during training. In the Soft Worm Peristaltic Navigation task, our approach achieves higher average returns more swiftly, highlighting its efficiency in simpler navigation scenarios. For the more complex Franka Arm Joint Motion and Quadrotor Propeller Control tasks, our method not only converges rapidly but also maintains higher final performance levels with minimal fluctuations, underscoring its robustness and reliability in handling intricate control dynamics.
this section cite: []

Section: Noise Robustness Performance Experiment
Our approach demonstrates remarkable stability across all noise levels (σ=0.1-0.3) for all three tasks. The averaged success rates stay above 91% with minimal variance, while baseline methods like Neural-CBF and SafetyNet show significant degradation under higher noise conditions (Figure 3). This robust performance demonstrates how the measurement-induced bundle structure naturally handles uncertainties in different scenarios, validating that our geometric safety certificates effectively preserve constraints regardless of the underlying task complexity.
this section cite: []

Section: Ablation Study
To validate our design choices and understand the interplay between different components, we conduct comprehensive ablation studies focusing on three key architectural elements: (1) measurement-induced bundle structure, (2) measurement-aware CBFs (mCBFs), and (3) Lie group symmetry.
For each ablation variant, we perform 10 independent runs on each of the three experimental tasks (soft robot navigation, Franka arm manipulation, and quadrotor control). The success rate (SR) represents the percentage of successful task completions averaged across all runs and tasks. Table 2 presents the comparative results.
Further ablation experiments examining the impact of different measurement uncertainty patterns (e.g., Gaussian noise, bias, delays) and bundle structure variations (e.g., fixed vs. adaptive dimension) are presented in Appendix E.
The measurement-induced bundle structure provides a geometric framework for handling system uncertainties, leading to more efficient trajectories and improved success rates. Without this structure, the success rate drops significantly from 96.3% to 62.7%, while path length and control magnitude increase substantially, indicating degraded performance and efficiency.
Removing mCBFs reduces our framework to its underlying Neural ODE architecture, which focuses solely on learning dynamics without safety constraints. This leads to the most severe performance degradation, with the success rate dropping to 45.7%. The dramatic reduction in constraint satisfaction rate (from 99.3% to 72.8%) demonstrates that while Neural ODEs can effectively learn system dynamics, they struggle to maintain safety constraints without the geometric safety certificates provided by mCBFs. The substantially increased path length (48.2m vs 18.5m) and control magnitude (0.85 vs 0.17) further suggest that pure dynamics learning leads to inefficient and potentially unsafe trajectories.
The Lie group symmetry enables dimension reduction and invariant control synthesis, which is particularly beneficial in tasks involving rotational and translational symmetries. Its removal shows relatively mild performance degradation (success rate of 85.7%), suggesting its role as a complementary enhancement to the core geometric framework rather than a critical component.
this section cite: []

Section: Conclusion
This paper presents a novel geometric framework that unifies measurements, constraints, and dynamics learning through a fiber bundle structure. Our framework provides fundamental insights into how dynamical systems can learn and   adapt under environmental constraints through local observations, bridging the gap between theoretical control guarantees and modern robotics (or even practical embodied intelligence). The measurement-induced bundle structure naturally captures how autonomous agents perceive and in-
Metrics Full Model w/o Bundle w/o mCBF w/o Lie Group SR (%) 96.3 62.7 45.7 85.7 PL (m) 18.5±0.7 35.3±3.9 48.2±5.1 22.5±1.5 GRS 383±18 712±82 935±108 465±32 FSE 0.05±0.01 0.15±0.03 0.21±0.04 0.08±0.02 MMC (m) 0.24±0.03 0.18±0.04 0.15±0.05 0.22±0.03 AMC (m) 0.26±0.03 0.20±0.03 0.17±0.04 0.24±0.02 CSR (%) 99.3±0.2 88.2±2.4 72.8±3.6 96.7±1.0 ACM 0.17±0.02 0.51±0.12 0.85±0.18 0.26±0.05 CS 0.03±0.004 0.07±0.008 0.09±0.010 0.04±0.005
Table 2. Ablation study results across three experimental tasks.
teract with their environment through local sensing, while the measurement-aware Control Barrier Functions enable adaptive safety certificates that emerge from direct environmental interactions rather than prescribed global knowledge.
Limitations and Future Work Despite these advances, our current implementation has limitations in handling highly stochastic dynamics and complex environmental uncertainties. Future work could explore richer representations of environment-agent interactions and investigate more sophisticated uncertainty quantification methods for embodied learning. Additionally, the framework could be extended to better understand how local observations can build towards a global understanding of environmental constraints and dynamics, potentially offering new perspectives on embodied intelligence and adaptive control. These results establish a promising direction for understanding physical systems learning through environmental interaction, offering insights into, for example, both dynamical systems theory and embodied intelligence principles.
this section cite: []

Section: References
Ref_id:b0 Title: Control barrier function based quadratic programs for safety critical systems Year: (2016)
Ref_id:b1 Title: Control barrier functions: Theory and applications Year: (2019)
Ref_id:b2 Title: Safe model-based reinforcement learning with stability guarantees Year: (2017)
Ref_id:b3 Title: Bayesian optimization with safety constraints: safe and automatic parameter tuning in robotics Year: (2023)
Ref_id:b4 Title: Fast approximate learning-based multistage nonlinear model predictive control using gaussian processes and deep neural networks Year: (2021)
Ref_id:b5 Title: An introduction to optimization on smooth manifolds Year: (2023)
Ref_id:b6 Title: Geometric deep learning: going beyond euclidean data Year: (2017)
Ref_id:b7 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b8 Title: End-to-end safe reinforcement learning through barrier functions for safety-critical continuous control tasks Year: (2019)
Ref_id:b9 Title: Robust control barrier-value functions for safety-critical control Year: (2021)
Ref_id:b10 Title: Gauge equivariant convolutional networks and the icosahedral cnn Year: (2019)
Ref_id:b11 Title: Robust safe control synthesis with disturbance observer-based control barrier functions Year: (2022)
Ref_id:b12 Title: Geometric aspects of diffusions on manifolds Year: (1985)
Ref_id:b13 Title: Neural networks with physics-informed architectures and constraints for dynamical systems modeling Year: (2022)
Ref_id:b14 Title: Les connexions infinitésimales dans un espace fibré différentiable Year: (1950)
Ref_id:b15 Title: A scalable safety critical control framework for nonlinear systems Year: (2020)
Ref_id:b16 Title: Group enhanced model for learning dynamical control systems Year: (2021)
Ref_id:b17 Title: Stochastic analysis on manifolds Year: (2002)
Ref_id:b18 Title: Robust control barrier functions for constrained stabilization of nonlinear systems Year: (2018)
Ref_id:b19 Title: A new approach to linear filtering and prediction problems Year: (1960)
Ref_id:b20 Title: Foundations of differential geometry Year: (1996)
Ref_id:b21 Title: Crafting papers on machine learning Year: (2000)
Ref_id:b22 Title: Affine connections and distributions with applications to nonholonomic mechanics Year: (1998)
Ref_id:b23 Title: System-level safety guard: Safe tracking control through uncertain neural network dynamics models Year: (2024)
Ref_id:b24 Title: Safe control under input limits with neural control barrier functions Year: (2023)
Ref_id:b25 Title: Reduction of symplectic manifolds with symmetry Year: (1974)
Ref_id:b26 Title: Gauge theory of the falling cat Year: (1090)
Ref_id:b27 Title: Riemannian motion policies Year: (2018)
Ref_id:b28 Title: Active learning of discrete-time dynamics for uncertaintyaware model predictive control Year: (2023)
Ref_id:b29 Title: Learning for safety-critical control with control barrier functions Year: (2020)
Ref_id:b30 Title: Adaptive safety with control barrier functions Year: (2020)
Ref_id:b31 Title: Safetynet: Safe planning for real-world selfdriving vehicles using machine-learned policies Year: (2022)
Ref_id:b32 Title: Datadriven safety filters: Hamilton-jacobi reachability, control barrier functions, and predictive methods for uncertain systems Year: (2023)
Ref_id:b33 Title: Structure-preserving algorithms for oscillatory differential equations II Year: (2015)
