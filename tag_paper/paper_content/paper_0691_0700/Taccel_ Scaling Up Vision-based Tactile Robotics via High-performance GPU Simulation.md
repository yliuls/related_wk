Title: Taccel: Scaling Up Vision-based Tactile Robotics via High-performance GPU Simulation
Abstract: Tactile sensing is crucial for achieving human-level robotic capabilities in manipulation tasks [54]. As a promising solution, Vision-based tactile sensors (VBTSs) [61,37] offer high spatial resolution and cost-effectiveness, but present unique challenges in robotics for their complex physical characteristics and visual signal processing requirements. The lack of efficient and accurate simulation tools for VBTSs has significantly limited the scale and scope of tactile robotics research [51,8]. We present Taccel, a high-performance simulation platform that integrates Incremental Potential Contact (IPC) and Affine Body Dynamics (ABD) to model robots, tactile sensors, and objects with both accuracy and unprecedented speed, achieving a total of 915 FPS with 4096 parallel environments. Unlike previous simulators that operate at sub-real-time speeds with limited parallelization, Taccel provides precise physics simulation and realistic tactile signals while supporting flexible robot-sensor configurations through user-friendly APIs. Through extensive validation in object recognition, robotic grasping, and articulated object manipulation, we demonstrate precise simulation and successful sim-to-real transfer. These capabilities position Taccel as a powerful tool for scaling up tactile robotics research and development, potentially transforming how robots interact with and understand their physical environment. accel (Ours) FEM ABD IPC DNN Any 4096 / 64 18.30ˆ/ 0.252

Section: Introduction
The ability to physically interact with the environment through touch is fundamental to robotic manipulation [3,13]. While vision provides global scene understanding, tactile sensing captures crucial local contact information [57] essential for precise manipulation. Among various tactile sensing technologies [64,25,38,26], vision-based tactile sensors (VBTSs) such as GelSight [61] and 9DTact [37] have emerged as a central focus in tactile research. Their ability to provide highresolution tactile feedback through camera-captured deformation patterns of elastic gel pads, combined with cost-effectiveness, has driven significant advances in robotics [66,35,47,8,65].
The primary challenge in scaling up VBTS-equipped robot simulation lies in accurately modeling the hyperelastic soft gel pad and its contact [66,12]. Current approaches follow two main directions: rigid-body approximations [51,56] and soft-body simulations [47,8,12,21,30,66]. While rigidbody methods efficiently support basic tasks like pick-and-place [1,51], they cannot capture the fine-grained contact and elastomer deformations essential for complex manipulation tasks [66,12] and detailed force distribution analysis [40,47]. Soft-body simulations offer higher fidelity but face significant computational challenges that limit their practical application in large-scale experiments.
Simulated Tactile Sensor Mahjong Tile 0.5 Deformation / mm 3.5 An ideal VBTS simulator must simultaneously achieve:
• Precision: Precise modeling of robots, sensors, and objects with physically valid solutions, particularly maintaining inversion-free and intersection-free states during complex contact interactions; generation of realistic tactile signals across multiple resolutions, from high-resolution RGB patterns and depth maps to lowresolution marker movements.
• Scalability: Capability for large-scale parallelization for extensive simulation data generation.
• Flexibility: Support for diverse robotic platforms and sensor configurations, from parallel grippers to multifinger hands with varying sensor arrangements.
As detailed in Tab. 1, existing solutions often compromise on precision, scalability, or flexibility. They typically produce suboptimal physics, operate slower than real-time with limited parallel environments, or focus on specific sensor setups or simple grippers. These limitations significantly impede the broader application of tactile robotics.
To address these challenges, we present Taccel, a high-performance simulation platform for scaling up robots with VBTS-integration. Built on state-of-the-art simulation techniques, Taccel provides dedicated components for simulating robots (Sec. 4), tactile sensors (Sec. 4.2), and tactile signal generation (Sec. 4.3). Comprehensive evaluations (Sec. 5) demonstrate its characteristics:
• Precision: Taccel leverages advanced solid material simulation techniques (IPC [32] and ABD [31]) for physical accuracy. IPC guarantees inversion-and intersection-free contact solutions, while the integration of ABD allows for efficient and precise simulation.
• Scalability: With an efficient ABD-IPC implementation with NVIDIA Warp [41], Taccel achieves unprecedented parallelization. On a single H100 GPU, it reaches over 900 FPS in total (4096 environments, 18ˆwallclock time) for a peg-insertion task with dual sensors and 12.67 FPS (256 environments, 0.25ˆwallclock time) in a dexterous grasping task with full-hand tactile sensing.
• Flexibility: Taccel provides user-friendly APIs for seamless integration of diverse robotic platforms and sensor configurations. Users can easily load and configure robots through Unified Robot Description Format (URDF) with auxiliary configurations, supporting applications from simple grippers to complex manipulation tasks, like the mahjong tile sensing task in Fig. 1.
Taccel's utility is validated through three fundamental tactile-informed robotic tasks (Sec. 6). In object classification, models trained solely on Taccel's synthetic tactile signals demonstrate strong generalization to real-world data without adaptation. In grasping experiments across four robotic hand designs, we showcase the platform's versatility in handling diverse robot configurations and
this section cite: ['b2', 'b12', 'b56', 'b63', 'b24', 'b37', 'b25', 'b60', 'b36', 'b65', 'b34', 'b46', 'b7', 'b64', 'b65', 'b11', 'b50', 'b55', 'b46', 'b7', 'b11', 'b20', 'b29', 'b65', 'b0', 'b50', 'b65', 'b11', 'b39', 'b46', 'b31', 'b30', 'b40']

Section: Related Work
Early VBTS simulators focused on normal deformation scenarios, approximating hyperelastic behavior through geometric computations and surface modifications [17,51,56,1,46]. While efficient in generating high-resolution tactile signals through physics-based rendering or look-up tables, these approaches inadequately capture elastomer dynamics during complex manipulation tasks, especially those involving tangential forces and continuous interactions [66].
Recent approaches have achieved higher physical fidelity by incorporating advanced solid material simulation techniques. Methods using Material Point Methods (MPM) [49,22,23] and Finite Element Methods (FEM) [32,31] better model elastomer properties through time-integrated deformation computations [10,8,47,12]. Notable improvements include the adoption of IPC [32] by several simulators [12,8], providing robust contact handling with guaranteed inversion-and intersectionfree solutions. Tab. 1 compares key features of representative approaches.
Taccel builds on these advances by combining IPC and ABD in a unified platform, achieving both physical accuracy and computational efficiency while supporting diverse robot configurations and enabling large-scale parallel simulation for robot learning applications.
A more comprehensive review on related works is in Sec. B concerning the VBTSs and tactileinformed robotic tasks.
this section cite: ['b16', 'b50', 'b55', 'b0', 'b45', 'b65', 'b48', 'b21', 'b22', 'b31', 'b30', 'b9', 'b7', 'b46', 'b11', 'b31', 'b11', 'b7']

Section: Unified IPC Simulation in Taccel
This section presents the unified IPC simulation framework in Taccel, detailing its mathematical foundations. For complete derivations, we refer readers to Sec. 3 and the original works [31,32,9].
this section cite: ['b30', 'b31', 'b8']

Section: Problem Formulation and Soft Body Dynamics
We consider n s tetrahedralized soft bodies discretized into N s vertices with positions x 1 , x 2 , ..., x Ns in Cartesian space. The system state is represented by the stacked position vector x " rx T 1 , x T 2 , ..., x T Ns s T P R 3Ns . Following Lagrangian mechanics, we express the system's Lagrangian as Lpx, 9
xq " T px, 9 xq ´V pxq, where T px, 9 xq " 1 2 9
x T M 9 x represents kinetic energy with mass matrix M P R 3Nsˆ3Ns . The potential energy V pxq comprises two terms: an elastic energy Φpxq utilizing the Neo-Hookean constitutive model for hyperelastic materials (characterized by Young's modulus E and Poisson's ratio ν), and external forces E ext pxq.
this section cite: []

Section: Frictional Contact
The Euler --Lagrange equation BL Bx px, 9 xq ´d dt BL B 9
x px, 9 xq " 0 is equivalent to the following Incremental Potential (IP) energy minimization problem [32] under the backward Euler integration scheme:
E IP pxq " 1 2 px ´xn ´∆t 9 x n q T Mpx ´xn ´∆t 9 x n q `∆t 2 V pxq, x n`1 " arg min x E IP pxq,(1)
where time is discretized into steps tt n " n∆t : n P Nu with a fixed step size ∆t ą 0, and x n " xpt n q.
To ensure intersection-free trajectories, IPC augments the objective with log barrier functions bpd k pxqq, which diverge to `8 as the distance d k pxq between contact primitive pair k approaches zero. Additionally, IPC introduces an approximate frictional potential energy D k px, x n q that captures frictional forces through its gradient. The full simulation thus minimizes the IPC energy:
E IPC pxq " E IP pxq `∆t 2 Bpxq `∆t 2 Dpx, x n q.(2)
with Bpxq " κ ř kPB A k bpd k pxqq, Dpx, x n q " ř kPB D k px, x n q, where κ ą 0 controls contact stiffness.
this section cite: ['b31']

Section: ABD and Unified Simulation
For n a affine bodies, we introduce a reduced coordinate space y P R 12na with an embedding map ϕ : R 12na Ñ R 3Na that projects reduced coordinates to full-space vertices ϕpyq [31], where N a denotes the total vertex count of affine bodies' surface meshes. Each affine body uses 12 Degree of Freedom (DoF): three for translation (R 3 ) and nine for affine deformation (R 3ˆ3 ).
T py, 9 yq "
1 2 9 x T M 9 x " 1 2 9 ϕpyq T M 9 ϕpyq " 1 2 pJ 9 yq T MpJ 9 yq " 1 2 9 y T pJ T MJq 9 y " 1 2 9 y T M y 9 y, (3
)
where J " Bϕ By P R 3Naˆ12na is the Jacobian, M is the full-space mass matrix, and M y " J T MJ is the reduced-space mass matrix. The potential energy V pyq includes an As-Rigid-As-Possible (ARAP) term Φ y pyq " Φ x pϕpxqq with large κ s to limit deformation, plus external forces E ext pyq.
Combining with Eq. ( 2) yields the unified IPC energy [9] for the full system state ty; xu P R 12na`3Ns :
E IPC py; xq " E IP pxq `EIP pyq `∆t 2 Bpϕpyq; xq `∆t 2 Dpϕpyq; x, ϕpy n q; x n q, E IP pyq " 1 2 py ´yn ´∆t 9 y n q T M y py ´yn ´∆t 9 y n q `∆t 2 V pyq.(4)
The next timestep's configuration follows from minimizing the following barrier-augmented IP:
y n`1 ; x n`1 " arg min y;x E IPC py; xq.(5)
this section cite: ['b30', 'b8']

Section: Kinematic Constraints
We express kinematic constraints as S x x " s x and S y y " s y , where S x P R c x ˆ3Ns , s x P R 3Ns for soft bodies, and S y P R c y ˆ12na , s y P R 12na for affine bodies. To enforce the constraints, we augment E IPC using the Augmented Lagrangian method with Lagrangian multipliers λ x P R c x , λ y P R c y :
E AL IPC py; xq " E IPC py; xq `}pS x x ´sx q T λ x } 2 2 `}pS y y ´sy q T λ y } 2 2 .(6)
4 Robot and VBTS Simulation in Taccel Building upon the unified ABD-IPC, Taccel implements robot and VBTS simulation through a modular design that leverages the complementary strengths of affine-and soft-body dynamics.
this section cite: []

Section: Robot and Sensor Simulation
Robot Modeling For a robot with D-DoF, L links, and N vision-based tactile sensors (VBTSs), Taccel constructs its kinematic model from a URDF specification, loading visual and collision meshes as affine bodies (Sec. 3.3). Given the robot's global transformation T r and joint configuration q P Q, forward kinematics yields the transformation of each link j in the world frame, r lj T pqq. Tactile Sensors Modeling Each VBTS is simulated by a tetrahedral mesh as a soft volumetric body (Sec. 3) for its gel pad, attached to its corresponding robot link. For the i-th sensor's gel pad G i attached to link l j with local transformation lj Gi T , we denote its outer surface as S i " BG. The surface comprises a reflective-coated region B `Gi and a sensor-attached region B ´G. Contact interactions during simulation cause gel pad deformation, transforming the coated surface B `G to B `G and marker positions to Pi . These deformed quantities serve as the foundation for generating multiple types of tactile signals, each suited for different robotic applications.
Wrong collision Wrong collision 29K nodes (bolt and nut) 6K nodes (bolt and nut) Penetration ≈64 envs / ≈4 envs >4096 envs / >256 envs Inaccurate physics with rigid sim PyBullet (a) a bolt-and-nut test (b) a so block pressing test (d) a multi-environment parallellization test (low-res / high-res) SAPIEN Taccel (Ours) w/o ABD Taccel (Ours) Di Tactile SAPIEN-IPC Taccel (Ours) (c) an articulated object manipulation task #switch error > 600% #switch error = 1.1% Isaac Sim Implementation Taccel (Ours) Taccel (Ours) Accurate physics with so body sim Penetration-free
this section cite: []

Section: Robot Actions
For scene initialization, we compute affine states through forward kinematics for robot links and explicit state specification for stiff objects. Gel pad node positions are transformed to the world frame, with all states written to x, y, and velocities 9
x, 9 y initialized to zero. Robot actions are implemented through kinematic constraints. From joint space targets, we compute affine state targets for links and node position targets for gel pad attached surfaces B ´G, assembled as s x , s y . Selection matrices S x , S y apply these constraints, with remaining states solved via time stepping.
this section cite: []

Section: Tactile Signal Simulation in Taccel
While many works directly generate tactile signals from the object geometry and frictional forces during rigid body simulation for efficiency [51,56,1], this approach can lead to inauthentic signals in dynamic scenarios. Instead, Taccel supports accurate simulation of high-resolution soft bodies to fully capture the fine-grained contact and deformation patterns.
this section cite: ['b50', 'b55', 'b0']

Section: RGB Images and Depth Maps
High-resolution tactile signals, including RGB images and depth maps, are essential when fine-grained details like object texture and local geometries are required. The signal generation process proceeds in two stages. First, we extract the depth and normal maps d pu,vq , n pu,vq for pixel coordinates pu, vq from the deformed coated surface B `G. Next, following the method of Si et al. [47], we apply a Deep Neural Network (DNN) to generate RGB tactile signals from the depth information. Specifically, for each pixel coordinate pu, vq, a pixel-to-pixel DNN parameterized by θ maps the inputs to the pixel colors relative to a reference image (RGB signal of undeformation gel pad): f θ `γpu, vq, n pu,vq ˘Þ Ñ ∆σ pu,vq , which are added to the reference image to obtain the final RGB image. γp¨, ¨q provides the 2D positional encoding of the pixel coordinate. The model is trained on patches from 200 real tactile images and corresponding depth map annotations.
Markers Within B `Gi , m i markers are positioned at locations Gi P i " tp piq k P R 3 , k " 1, . . . , m i u N
i"1 , each defined by barycentric coordinates in its triangle:
p piq k " ř 3 u"1 α u x pi,kq u
, where ř 3 u"1 α u " 1, α u P r0, 1s. Low-resolution tactile signals primarily track marker positions and their movements by computing their new positions throughout the simulation (at time step t):
ppiq k ptq " ř 3 u"1 α u xpi,kq u
ptq, and projecting them on the tactile image. The marker flows, representing local deformation patterns, are then computed as:
∆P i " tp piq k ´ppiq k u " t∆p piq k u.
3D Tactile Signals The depth map and marker positions can be transformed into a dense or sparse 3D point cloud in the world frame using robot kinematics and sensor configurations. These 3D tactile signals provide crucial spatial information for robotic manipulation tasks. We demonstrate their effectiveness through the simulation of Tac-Man framework [66] in Sec. 6.3.
We further demonstrate Taccel's capability to scale up synthetic data generation of tactile signals via robotic grasping simulations and explore object recognition model learning (Secs. 6.1 and 6.2).
this section cite: ['b46', 'b65']

Section: API Designs in Taccel
Taccel provides intuitive Python APIs designed to make tactile robotics simulation accessible to researchers while maintaining high performance through NVIDIA Warp [41]. The APIs allows for seamless loading of robots from URDF files with automatic parsing, sensor configurations from auxiliary files, and objects from mesh files. Users can efficiently reset simulation states or apply kinematic targets to control the robots in familiar formats (NumPy arrays, PyTorch tensors). Further, Taccel supports parallel simulation of multiple environments similar to Isaac Gym [43].
To foster community development, we will release Taccel codebase and documentations, while maintaining active collaboration with researchers to incorporate feedback, add features, and expand capabilities, ensuring its evolution as a comprehensive tool for tactile robotics research.
this section cite: ['b40', 'b42']

Section: Performance Evaluation of Taccel
We evaluate Taccel through comprehensive benchmarks on its precision and efficiency. Our analysis demonstrates that the combination of ABD and IPC provides significant advantages over existing approaches (Sec. 5.1), generates high-quality tactile signals (Sec. 5.2), ensures precise frictions and deformation solving for the gel pad (Sec. 5.3), and enables efficient scaling (Sec. 5.4).
this section cite: []

Section: Overall Comparisons
As illustrated in Fig. 2, Taccel achieves superior precision and efficiency through its unified ABD and IPC framework, demonstrated across four challenging scenarios.
First, the bolt screwing task (Fig. 2a) demonstrates Taccel's ability to handle complex physical interactions between highly non-convex objects, where conventional simulators including PyBullet and SAPIEN [55] fail to handle. This capability stems from our ABD formulation, which provides an efficient approach to simulating dense rigid-soft body interactions. Alternative approaches either model stiff objects as soft bodies, introducing excessive DoFs and computational overhead (Fig. 2a, ours w/o ABD), or rely on RBD [14], which requires expensive nonlinear Continuous Collision Detection (CCD) calculations to avoid intersection, significantly degrading performance.
Next, physical realism in Taccel is demonstrated through the soft block pressing scenario (Fig. 2b), where our collision-free and intersection-free guarantees produce notably more realistic deformations compared to penalty-based approaches. This precision extends to practical robotics applications, as shown in the Tac-Man microwave manipulation task (Fig. 2c). Here, Taccel's accurate contact force solving enables faithful reproduction of gel pad-object handle interactions, closely matching real-world execution patterns (detailed analysis in Sec. 6.3).
Finally, the computational efficiency of Taccel emerges from our optimized implementation of the ABD and IPC algorithms, enabling unprecedented scaling capabilities. In a peg insertion task, Taccel achieves parallel simulation of over 4096 environments on a single GPU with 80GB VRAM-representing a 64-fold improvement over SAPIEN-IPC [8]. This opens new possibilities for large-scale robotics simulation and learning; see also the comprehensive analysis in Sec. 5.4.
this section cite: ['b54', 'b13', 'b7']

Section: Tactile Signal Simulation
We evaluate the fidelity of tactile signals generated by Taccel with real-world samples. We press a calibrated GelSight-type sensor perpendicullarly on 18 objects from a standard tactile shape testing dataset [16] on a real-world setup (Fig. 3(a)) and in Taccel; see Sec. D.1 for setup details.
The qualitative and quantitative comparisons shown in Fig. 3(a) demonstrate Taccel's ability to produce highly realistic tactile patterns, achieving an average SSIM of 0.93 across all test objects. Minor variations between simulated and real signals primarily stem from manufacturing tolerances in the 3D-printed objects and challenges in precise camera calibration. Despite these practical limitations, the results establish Taccel's capability to generate high-fidelity tactile signals suitable for VBTSs, with simulated patterns closely matching experimental measurements.
this section cite: ['b15']

Section: Precision on Frictions and Shear Deformation
We further investigate Taccel's precision on solving frictions and shear deformation. Fig. 3(b) shows the evaluation setup: A VBTS-gripper grasps a fixed bar with various forces (causes tactile depth d) then pulls back at 2 mm{s, recording gel deformation represented by the marker deformations. We calibrate the object's friction coefficient with one record and use two others with various ds to compare the errors between simulated and real-world trajectories. Small sim-real errors (avg 28 µm) tested on two bars with various frictions highlights Taccel's fidelity on gel deformation and frictions in both static and slipping cases.
this section cite: []

Section: Multi-environment Simulation
Efficient and stable parallel simulation of multiple environments is crucial for scaling up synthetic data collection across diverse downstream tasks. To evaluate Taccel's capabilities in this regard, we designed three test cases of increasing complexity: (i) a dual-sensor (139 nodes per gel pad) peg-insertion task adapted from SAPIEN-IPC [8], (ii) the peg-insertion task with higher solution (1.5k nodes per gel pad), and (iii) a grasping task using a customized five-fingered dexterous hand equipped with 17 gel pads wrapped around the hand links (5k nodes). These tasks involve continuous contact and gel pad deformation, shedding light on how the simulator would perform on various robotic tasks. Sec. D.2 provide more details on these tasks.
With a single NVIDIA H100 80G GPU, we benchmarked Taccel against SAPIEN-IPC. The results in Fig. 4 showcase Taccel's superior performance: in the low-resolution peg-insertion task, Taccel achieves 915 FPS (18.30ˆwallclock time) while managing over 4096 parallel environments-a 16-fold improvement over the baseline using the same GPU memory, enabled by our efficient parallelization implementation. The high-resolution test further demonstrates Taccel's advantages, maintaining both stability and precision while consistently outperforming the baseline. Even in the complex dexterous hand scenario, Taccel efficiently handles full-hand tactile sensing, achieving 12.67 FPS across 256 environments. We observed that SAPIEN-IPC's use of FP32 precision leads to convergence issues when solving contact forces in Eq. ( 2), particularly in the logarithmic barrier energy term calculations, as indicated by red outlines in Fig. 4. We further investigate this gap in the low-resolution peg-insertion benchmark. We set the maximum Newton iterations to 50 and optimization residue tolerance set to 0.01m/s, i.e. the IPC solver performs conjugate gradient descent step until (i) the 50-step limit is exceeded, or (ii) the optimization residue is below 0.01m/s. We report the key quantities including the Newton iterations used, its optimization residue, total time spent for the simulation, and total FPS in Tab. 2. Although each step takes longer for FP64, the convergence is much faster due be the better precision, and thus the better simulation speed.
Noteworthy, while Taccel's FP64 precision is ideal for HPC GPUs (which have a high 1:2 FP32:FP64 FLOPS ratio), it remains highly performant on more accessible consumer cards like RTX 3090 and 4090; see Sec. C.2 for more details.
this section cite: ['b7']

Section: Applications in Tactile Robotics
We demonstrate Taccel's capabilities across three tactile-informed robotic tasks: training object classification models with synthetic data (Sec. 6.1), generating a large-scale dataset through parallel grasp simulations (Sec. 6.2), and manipulating articulated objects (Sec. 6.3). These applications showcase how Taccel enables precise robotic simulation and scalable synthetic data generation.
this section cite: []

Section: Learning Object Classification Models
To demonstrate Taccel's ability to generate synthetic training data that generalizes to real-world scenarios, we developed a tactile-based object classification system. As shown in Fig. 5, our approach trains a DNN to classify objects using high-resolution tactile signals.   Following Yang et al. [58], we selected 10 mechanical parts with distinct fine-grained geometries (illustrated in Fig. 5, top). We collect a training dataset 4K tactile depth maps of a tactile sensor pressing on them within Taccel, each with object pose and tactile depth randomly sampled. We also collect a real-world test set that consists of 160 depth maps for testing. Sec. D.3 illustrates more details for the data collection protocol. We trained a ResNet-18 model [20] for 10-category object classification using the simulated depth images. To enhance robustness, we augmented the depth maps through random affine transformations, morphological operations (erosion and dilation), and Gaussian filtering. The model is then directly evaluated them on the real-world samples, with each sample tested 4 times with random shear deformation. As shown in Tab. 3, our model achieved 86.50% accuracy on the synthetic test set and 70.94% on real-world samples without any domain adaptation. This modest sim-to-real gap demonstrates Taccel's capability to generate precise tactile signals that enable data-efficient training of transferable tactile perception models.
this section cite: ['b57', 'b19']

Section: Robotic Grasping with Tactile Sensors
We investigate robotic grasping across different hand configurations with varying tactile sensor arrangements. We first extend the DFC algorithm [39] for generating contact-oriented grasping poses for four robotic hands, then simulate their tactile responses within Taccel, as illustrated in Fig. 6. We generate grasps on 10 diverse objects from ContactDB [4], YCB [7], and adversarial object [42] datasets. For each object, we generated grasps using 4 different robotic hands, producing "14k total grasps. These were simulated in Taccel to generate tactile signals, with key metrics summarized in Tab. 3. Fig. A1 provides an additional visualizations for the grasps across 4 robotic hands. Sec. D.4 explains the details of the modification to DFC.
Simulating robotic grasps with tactile sensors yields synthetic data extending the existing robotic grasping datasets with tactile perception capabilities, serving as a foundation for various robotic tasks. To demonstrate its utility, we implemented the object classification task described in Sec. 6.1, adapting it to use the deformed coat's point cloud as input and PointNet as the feature extractor.
The classification performances in Tab. 3 reveals an interesting trade-off: while robots differ in sensor count and sensing area, higher dexterity (Allegro Hand) enables better object contact despite fewer sensors. This enhanced contact leads to superior classification accuracy, highlighting the balance between sensor count and dexterity in tactile hand design. These findings demonstrate Taccel's value in validating robotic hand designs before physical fabrication.
this section cite: ['b38', 'b3', 'b6', 'b41']

Section: Articulated Object Manipulation
We consider articulated object manipulation, a challenging task where tactile perception provides critical feedback on hand-object contact, informing object articulation and guiding robot actions [26,66]. Sec. D.5 detailedly explain the algorithm.
Tac-Man's effectiveness relies heavily on gel pad deformation for motion adaptation and tactile feedback. While the original implementation by Zhao et al. used rigid body simulation with gripper compliance approximations, it couldn't authentically replicate gel pad deformation, contact dynamics, and tactile feedback, resulting in significant sim-to-real gaps during large-scale verification. Taccel overcomes these limitations through accurate simulation of sensor-object contact and gel pad deformation. We simulate Tac-Man on three types of articulated objects: drawers with prismatic joints, cabinets with revolute joints, and bolt-nut pairs with helical joints. Fig. 7(a) shows the manipulation sequences and corresponding tactile signals.
To evaluate sim-real correspondence, we compared our simulation against real-world Tac-Man execution using a microwave (revolute joint) and drawer (prismatic joint), shown in Fig. 7(b-c). For comprehensive comparison, we also tested Tac-Man's official implementation on these objects. We use the "execution-recovery switch count" as our key metric, which tallies how often the agent switch between the execution and recovery state when executing the Tac-Man algorithm (see Zhao et al. [66] and Sec. D.5 for details). As demonstrated in Fig. 7, Taccel faithfully reproduces real-world execution patterns, with execution-recovery switch counts averaging 68.75 and 0.0 for revolute and prismatic settings, respectively-remarkably matching real-world observations. While Isaac Sim successfully simulated the manipulation, its dynamics gap resulted in substantially higher switch counts. These results highlight Taccel's capability to authentically replicate physical interactions in manipulation scenarios. We present Taccel, a flexible and high-performance simulator for VBTS-integrated robots. Its userfriendly APIs enables precise and efficient simulation of complex tactile robotic tasks with realistic tactile signals. Taccel excels in capturing intricate deformation and contact dynamics of soft gel pads with unprecedented stability, while supporting thousands of parallelized environments. These capabilities also position Taccel as a powerful tool for hand-sensor validating before fabrication, potentially reducing development time and costs.
Limitations Despite the high performance of Taccel, its computational demands of large-scale simulation remain challenging, with a major bottleneck being the PCG-based linear system solving. Possible remedies include carefully relaxing convergence tolerances, simplifying simulation protocols, and using larger timesteps, enabled by IPC's unconditionally stable solver. Besides, the Neo-Hookean model itself may still be insufficient in capturing all necessary physical characteristics of the gel pad. Further, the data-driven tactile signal generation method is still trained from limited data and thus may not generalize to all contact conditions. These limiations are to be addressed in future studies.
Broader Impact This paper focuses on fundamental research, with limited direct societal impact.
this section cite: ['b25', 'b65', 'b65']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: The abstract and introduction clearly state the claims made.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: Section Sec. 7 includes a "Limitations" paragraph.
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
Answer: [Yes] Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The codebase and dataset are ready and will be released upon acceptance for better reproducibility.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: Most of the key details are provided within the text. Full details will be provided within the code.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: The simulation speed test in Sec. 5 (see Fig. 4) includes the error bars. We also report the standard deviation for the Tac-Man simulation in Fig. 7.
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
Answer: [Yes]
Justification: The paper includes necessary information on the computer resources.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: The research conducted in the paper conforms with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: The broader impact is discussed in Sec. 7.
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
Answer: [NA] Justification: The paper does now contain data or models with a high rist for misuse.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes]
Justification: The authors of the assets are cited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
this section cite: []

Section: A Unified ABD-IPC in Taccel
In Taccel, robot links are efficiently modeled as affine bodies to capture their primarily rigid motion, while VBTSs are simulated as soft bodies to accurately represent their deformation mechanics. This natural division allows Taccel to balance computational efficiency with physical accuracy while maintaining consistent contact handling through IPC.
We provide a more detailed derivation of the unified ABD-IPC algorithm underlying Taccel, as previously briefed in Sec. 3.
this section cite: []

Section: A.1 Soft Body Dynamics
Recall that we consider n s tetrahedralized soft bodies with N s vertices: x 1 , x 2 , ..., x Ns . The stacked position vector x " rx T 1 , x T 2 , ..., x T Ns s T P R 3Ns represents the system state, and thus the system's Lagrangian is Lpx, 9
xq " T px, 9 xq ´V pxq with the kinetic energy T px, 9 xq " 1 2 9
x T M 9 x, given the mass matrix M P R 3Nsˆ3Ns . The potential energy V pxq is composed of the elastic energy Φpxq utilizing the Neo-Hookean constitutive model for hyperelastic materials (characterized by Young's modulus E and Poisson's ratio ν), and the external forces E ext pxq. The elastic energy is defined as Φpxq " ş Ω Ψpxqdx, where Ψpxq denotes elastic energy density over the volume region Ω of all objects in rest configuration.
this section cite: []

Section: A.2 Time Stepping

this section cite: []

Section: Substituting Lpx, 9
xq into the Euler-Lagrange equation BL Bx px, 9 xq ´d dt BL B 9
x px, 9 xq " 0 yields the governing dynamics:
M: x " ´dV dx pxq.(A1)
We temporally discretize Eq. (A1) using backward Euler:
x n`1 ´xn ∆t " 9 x n`1 , Mp 9 x n`1 ´9 x n q ∆t " ´dV dx px n`1 q, (A2
)
where time is discretized into steps tt n " n∆t : n P Nu with step size ∆t ą 0, and x n " xpt n q. Under this discretization, Eq. (A1) can be formulated as:
d dx pE IP px n qq " 0.(A3)
If we define the incremental potential energy of the constrained system as:
E IP pxq " 1 2 px ´xn ´∆t 9 x n q T Mpx ´xn ´∆t 9 x n q `∆t 2 V pxq,(A4)
then the general simulation problem in a conservative system can be reformulated as the minimization problem:
x n`1 " arg min x E IP pxq.(A5)
this section cite: []

Section: A.3 Frictional Contact
We employ IPC [32] to handle contact interactions. The method operates on surface contact pairs B, comprising point-triangle and edge-edge pairs from the surface meshes of soft and affine objects. For each contact pair k P B with distance d k ą 0, IPC defines two key energy terms. First, a barrier energy that prevents interpenetration:
bpd k pxqq " ´´d k ´d ¯2 logp d k d qI td k Pp0, dqu pd k q,(A6)
where d ą 0 is the distance threshold for contact force activation and Ip¨q is the indicator function.
training data with strong real-world transfer, and (iii) articulated object manipulation including drawers, cabinets, and bolt-nut assembly tasks, extending the Tac-Man framework [66].
this section cite: ['b31', 'b65']

Section: C Additional Tests on Simulation Speed

this section cite: []

Section: C.1 VRAM Usage
Tab. A1 report the VRAM occupancy of the low-resolution peg-insertion test (Sec. 5.4). The GPU memory (VRAM) scales linearly with the number of environments, enabling efficient scaling until memory saturation.
Table A1: VRAM Usage of Taccel in the low-resolution peg-insertion test. # Envs 1 16 256 1024 4096 VRAM / GiB 4.3 4.3 7.2 16.7 54.0 C.2 Simulation Speed on Various GPUs Tab. A2 reports the simulation speed of the low-resolution peg-insertion test on various GPUs (Sec. 5.4). While HPC GPUs with high FP32:FP64 ratio (1:2) is ideal for optimal performance, desktop GPUs like the 3090 (FP32:FP64 ratio = 1:64) also deliver scalable performance, achieving real-time performance (50FPS) with around 100 environments.
Table A2: Simulation speed of Taccel in the low-resolution peg-insertion task on various GPUs. GPU H100 RTX 4090 RTX 3090 RTX 3090 RTX 3090 # Envs 256 256 256 128 64 Total FPS Ò 185.52 129.12 121.38 74.81 43.40
this section cite: []

Section: D Experiments Details

this section cite: []

Section: D.1 Real-world Data Collection for Tactile Signal Evaluation
For real-world data collection, our experimental setup consisted of a calibrated GelSight-type sensor, mounted on a vertical rack for precise movement control, as shown in Fig. 3(a). A white mount is fixed in the center of the platform to hold the test objects.
We 3D-print the 18 objects from a standard tactile shape testing dataset [16] at 0.2mm layer height.
For each object, we first fix it on the mount, slide the sensor along the rack to press its gel pad on the object, and put a 500 g weight on the sensor to ensure an appropriate amount of gel deformation. We record the RGB tactile patterns five times and compute their mean image to partially remove the noise. We then replicated this pressing sequence in Taccel using a high-resolution soft body gel pad (maximal cell volume V max « 10 ´12 m 3 ) to ensure signal fidelity. Sec. 5.2 compares the similarity between the real-world and simulated tactile signals.
this section cite: ['b15']

Section: D.2 Multi-environment Simulation Test
Our multi-environment simulation test involves three scripted tasks. First, we implemented a peginsertion task adapted from SAPIEN-IPC [8], where two gel pads (139 nodes and 317 cells each) follow a scripted trajectory. The trajectory involves 200 simulation steps, where the sensors squeeze the peg and manipulate it around the hole. Next, we scaled this task to a higher resolution (1,533 nodes and 5,360 cells each), which results in a larger system to solve. Finally, to demonstrate Taccel's potential for advanced robotics research, we created a scripted grasping task with a customized fivefingered dexterous hand equipped with 17 gel pads covering the entire hand, totaling 5,157 nodes and 14,311 cells. The trajectory also involves 200 simulation steps, with the hand approaches, grasps, lifts, maneuvers, and releases a stiff cylinder.
this section cite: ['b7']

Section: D.3 Data Collection and Training Details for Learning Object Classification
In the object classification sim-to-real experiment, we collect tactile signals of grasping 10 3Dprinted mechanical parts in both simulation and the real world. We also train an object classification model with the simulated data and evaluate it on the real data.
For each object, we simulated 200 grasp trials using a parallel gripper with randomized grasping poses. The gripper closes towards the object until the depth deformation exceeds a threshold randomly sampled from τ d " U r0.5, 1.5smm, and with each grasp yields 2 depth map samples after grasping. To ensure the granularity of the depth map, high-resolution gel pad models max V ě 10 ´12 m 3 . The depth maps d pu,vq extracted from these simulated tactile signals yielded approximately 4,000 training samples. We split the samples into a training set (85%) and a validation set (15%). The former is used to train the classification model, supervised by the NLL Loss, using the Adam optimizer with a learning rate 1e-4 for 100 epochs. During training, we apply an exponential LR scheduler.
For real-world validation, we collected tactile signals using a RobotiQ-2F85 parallel gripper equipped with GelSight-type sensors. The test objects were 3D printed at 0.2mm layer height to maintain high geometric fidelity. We gathered 8 grasps (16 depth maps) per object.
this section cite: []

Section: D.4 Dexterous Grasping
Our key modification to the DFC algorithm promotes perpendicular contact between gel pads and object surfaces, optimizing for downstream tasks that rely on tactile perception. Following Liu et al. [39], we synthesize grasping poses in the robot's joint space q P Q relative to the object frame by minimizing a modified Gibbs energy:
EpO, q, T q " E DFC `λcontact E contact pO, q, T q. (A18
)
The force-closure term E DFC maintains its original formulation [39], with added constraints to ensure gel pad penetration depth ϵ (typically 0.5 mm). We introduce a new contact term E contact that aligns the normals of gel pad contacts c i P B `Gi with their corresponding object surface normals o i " arg min oPBO }o ´ci }:
E contact pO, q, T q " 1 ´ c K i , o K i ,(A19)
where p¨q K represents the surface normals of the gel pad and object surfaces.
Fig. A1 shows the examples of the synthesized grasps and the simulated tactile signals in both 2D and 3D.
this section cite: ['b38', 'b38']

Section: 
Justification: The derivation of the unified IPC-ABD algorithm is included in Sec. 3.
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
Answer: [Yes] Justification: The paper discloses all the required information for the simulation system and the experiments. The codebase is ready and will be released upon acceptance for better reproducibility.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: 5.
Open access to data and code
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/  datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not contain crowdsourcing experiments or research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not contain research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: LLMs are only used for editing the manuscript. Thus the declaration is not needed. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/  2025/LLM) for what should or should not be described.
Second, an approximated friction potential energy:
D k px, x n q " µλ n k f 0 p∥u k ∥q,(A7)
where x n represents the configuration at the previous timestep t n , λ n k is the magnitude of the lagged normal contact force, and u k P R 2 denotes the tangential relative displacement in the local contact frame.
The friction transition function f 0 pxq " ş x ϵv∆t f 1 pyqdy `ϵv ∆t uses:
f 1 pyq " # ´y2 ϵ 2 v ∆t 2 `2y ϵv∆t , y P p0, ∆tϵ v q, 1, y ě ∆tϵ v ,(A8)
where ϵ v ą 0 serves as a velocity threshold distinguishing between static and dynamic friction regimes.
These contact and friction terms augment our incremental potential energy:
E IPC pxq " E IP pxq `∆t 2 Bpxq `∆t 2 Dpx, x n q,(A9)
with Bpxq " κ
ř kPB A k bpd k pxqq, Dpx, x n q " ř kPB D k px, x n q,
where κ ą 0 controls contact stiffness.
this section cite: ['b15']

Section: A.4 ABD and Unified Simulation
For n a affine bodies, we introduce a reduced coordinate space y P R 12na with an embedding map ϕ : R 12na Ñ R 3Na that projects reduced coordinates to full-space vertices ϕpyq [31], where N a denotes the total vertex count of affine bodies' surface meshes. Each affine body uses 12 DoF: three for translation (R 3 ) and nine for affine deformation (R 3ˆ3 ).
T py, 9 yq "
1 2 9 x T M 9 x " 1 2 9 ϕpyq T M 9 ϕpyq " 1 2 pJ 9 yq T MpJ 9 yq " 1 2 9 y T pJ T MJq 9 y " 1 2 9 y T M y 9 y,(A10)
where J " Bϕ By P R 3Naˆ12na is the Jacobian, M is the full-space mass matrix, and M y " J T MJ is the reduced-space mass matrix. The potential energy V pyq includes an As-Rigid-As-Possible (ARAP) term Φ y pyq " Φ x pϕpxqq with high stiffness κ s to limit deformation, plus external forces E ext pyq.
this section cite: ['b30']

Section: Combining with Eq. (A9
), we obtain the unified affine-deformable coupled IPC energy [9] for the full system state ty; xu P R 12na`3Ns :
E IPC py; xq " E IP pxq `EIP pyq `∆t 2 Bpϕpyq; xq `∆t 2 Dpϕpyq; x, ϕpy n q; x n q,(A11)
where E IP pyq is defined as:
E IP pyq " 1 2 py ´yn ´∆t 9 y n q T M y py ´yn ´∆t 9 y n q `∆t 2 V pyq.(A12)
The next timestep's configuration follows from minimizing this barrier-augmented incremental potential:
y n`1 ; x n`1 " arg min y;x E IPC py; xq.(A13)
this section cite: ['b8']

Section: A.5 Kinematic Constraints
The kinematic constraints is expressed as
S x x " s x , S y y " s y ,(A14)
where S x P R c x ˆ3Ns , s x P R 3Ns for soft bodies, and S y P R c y ˆ12na , s y P R 12na for affine bodies. The constraints are applied by selecting the constrained DoFs of the state vectors and specifying the constraint values with. To enforce these constraints, we employ the Augmented Lagrangian method by augmenting E IPC to:
E AL IPC py; xq " E IPC py; xq `}pS x x ´sx q T λ x } 2 2 `}pS y y ´sy q T λ y } 2 2 ,(A15)
where λ x P R c x and λ y P R c y are Lagrangian multipliers.
Optimizing E AL IPC py; xq yields the solution to the constrained system:
y n`1 ; x n`1 " arg min y;x E IPC py; xq,(A16)
s.t. S x x " s x and S y y " s y .
this section cite: []

Section: A.6 Guides on Setting the Materials Parameters for Gel Pads
We provide guidance on properly choosing the parameters of the gel pads, which plays a vital role in vision-based tactile sensing. Emperically, the pad's Poisson's ratio ν is typically in r0.3, 0.45s, within which the response variation is subtle. The Young's modulus typically falls in r0.01, 10sMPa, and the response remains almost constant within the same order of magnitude. For simulations without measuring the parameters of the material, our practice is to start within the typical range and carefully adjust them by inspecting the simulation scenes.
While the simulation is not overly sensitive to these parameters, we still recommend measuring these parameters using standard techniques. For example, the sliding experiments for friction coefficients [62] or tensile tests for elastic moduli [18,27,33].
this section cite: ['b61', 'b17', 'b26', 'b32']

Section: B Additional Related Work

this section cite: []

Section: B.1 Robot Tactile Sensors
Tactile sensing plays a fundamental role in precise manipulation, as established by neuroscientific studies [54,29,28,3]. This understanding has driven the development of artificial tactile sensing systems for robots [44]. Among these, VBTSs have gained prominence by offering high-resolution sensing with cost-effectiveness and operational simplicity [61,53,37,35]. While these sensors have advanced robotic manipulation [45,40,66], their development remains constrained by the reliance on physical hardware experimentation. Taccel addresses this limitation by providing a comprehensive simulation platform to accelerate research and development in tactile robotics.
this section cite: ['b53', 'b28', 'b27', 'b2', 'b43', 'b60', 'b52', 'b36', 'b34', 'b44', 'b39', 'b65']

Section: B.2 Tactile-Informed Robotic Tasks
Tactile sensing enhances robotic capabilities across three fundamental domains through precise contact interaction measurements: Perception Tactile feedback enables sophisticated object understanding through contact-based sensing. Applications include shear and slip detection [63,11], object classification and pose estimation [34,58,50,2], material property inference [19,24], and interaction reconstruction [50,60,57]. These perceptual capabilities form the foundation for advanced manipulation algorithms.
Grasping Stable grasping requires precise control of contact forces to balance external loads [15,48]. Tactile sensing provides direct force-torque feedback essential for diverse grasping strategies [39,36,59]. This tactile information complements vision-based approaches by enabling finegrained contact monitoring and in-hand adjustments [6,5].
Manipulation Tactile feedback enables complex manipulation beyond basic pick-and-place operations. Applications include precision tasks like peg insertion [8], object pivoting [21], and articulated object manipulation [67,3]. Systems such as Tac-Man [66] and DoorBot [52] demonstrate how tactile sensing guides contact geometry understanding and articulation control. This sensing modality is particularly crucial for high-frequency object tracking during dexterous manipulation.
We validate Taccel's capabilities through three representative applications: (i) multi-platform robotic grasping with both rigid and soft objects, (ii) object classification using purely synthetic
this section cite: ['b62', 'b10', 'b33', 'b57', 'b49', 'b1', 'b18', 'b23', 'b49', 'b59', 'b56', 'b14', 'b47', 'b38', 'b35', 'b58', 'b5', 'b4', 'b7', 'b20', 'b66', 'b2', 'b65', 'b51']

Section: D.5 Articulated Object Manipulation with Tac-Man
In the Tac-Man framework [66], the system alternates between execution and recovery phases. During execution, the system performs coarse manipulation actions (e.g., pulling backward) to gradually move the articulated object part. When the actual motion deviates from intended trajectories due to articulation constraints, the gel pad deforms, creating contact deviation reflected in marker flow magnitudes. Once these flows exceed threshold δ 0 , the system enters recovery mode to restore stable contact by reducing deviation, before resuming execution. This execution-recovery cycle typically requires tens of iterations to finish manipulation.
For our sim-real comparison, we manually created URDF models for the microwave oven and the drawer to match real-world geometries and kinematics, maintaining identical initial grasping poses across simulation and physical setups. Following Tac-Man's implementation, we set δ 0 " 0.4 mm and α " 0.6.
this section cite: ['b65']

Section: References
Ref_id:b0 Title: Tacsl: A library for visuotactile sensor simulation and learning Year: (2025)
Ref_id:b1 Title: Tac2pose: Tactile object pose estimation from the first touch Year: (2023)
Ref_id:b2 Title: Trends and challenges in robot manipulation Year: (2019)
Ref_id:b3 Title: Contactdb: Analyzing and predicting grasp contact via thermal imaging Year: (2019)
Ref_id:b4 Title: Texterity: Tactile extrinsic dexterity Year: ()
Ref_id:b5 Title: More than a feeling: Learning to grasp and regrasp using vision and touch Year: (2018)
Ref_id:b6 Title: The ycb object and model set: Towards common benchmarks for manipulation research Year: (2015)
Ref_id:b7 Title: General-purpose sim2real protocol for learning contact-rich manipulation with marker-based visuotactile sensors Year: ()
Ref_id:b8 Title: A unified newton barrier method for multibody dynamics Year: ()
Ref_id:b9 Title: Fuchun Sun, and Bin Fang. Tacchi: A pluggable and low computational cost elastomer deformation simulator for optical tactile sensors Year: (2023)
Ref_id:b10 Title: Improved gelsight tactile sensor for measuring geometry and slip Year: ()
Ref_id:b11 Title: Tacipc: Intersection-and inversion-free fem-based elastomer simulation for optical tactile sensors Year: (2024)
Ref_id:b12 Title: A tale of two explanations: Enhancing human trust by explaining robot behavior Year: (2019)
Ref_id:b13 Title: Intersection-free rigid body dynamics Year: ()
Ref_id:b14 Title: Planning optimal grasps Year: ()
Ref_id:b15 Title: Generation of gelsight tactile images for sim2real learning Year: (2021)
Ref_id:b16 Title: Gelsight simulation for sim2real learning Year: (2019)
Ref_id:b17 Title:  Year: (1994)
Ref_id:b18 Title: Estimating properties of solid particles inside container using touch sensing Year: ()
Ref_id:b19 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b20 Title: Tactile dexterity: Manipulation primitives with tactile feedback Year: ()
Ref_id:b21 Title: A moving least squares material point method with displacement discontinuity and two-way rigid body coupling Year: (2018)
Ref_id:b22 Title: Chainqueen: A real-time differentiable physical simulator for soft robotics Year: (2019)
Ref_id:b23 Title: Understanding dynamic tactile sensing for liquid property estimation Year: ()
Ref_id:b24 Title: Hierarchical fall avoidance strategy for small-scale humanoid robots Year: ()
Ref_id:b25 Title: Control framework for dexterous manipulation using dynamic visual servoing and tactile sensors' feedback Year: (2014)
Ref_id:b26 Title: Quantifying cell-generated forces: Poisson's ratio matters Year: ()
Ref_id:b27 Title: Capturing forceful interaction with deformable objects using a deep learning-powered stretchable tactile array Year: ()
Ref_id:b28 Title: Coding and use of tactile signals from the fingertips in object manipulation tasks Year: (2009)
Ref_id:b29 Title: Texterity-tactile extrinsic dexterity: Simultaneous tactile estimation and control for extrinsic dexterity Year: ()
Ref_id:b30 Title: Affine body dynamics: fast, stable and intersection-free simulation of stiff materials Year: ()
Ref_id:b31 Title: Incremental potential contact: intersectionand inversion-free, large-deformation dynamics Year: ()
Ref_id:b32 Title: Easycalib: Simple and low-cost in-situ calibration for force reconstruction with vision-based tactile sensors Year: ()
Ref_id:b33 Title: Localization and manipulation of small parts using gelsight tactile sensing Year: ()
Ref_id:b34 Title: Minitac: An ultra-compact 8 mm vision-based tactile sensor for enhanced palpation in robot-assisted minimally invasive surgery Year: (2024)
Ref_id:b35 Title: Grasp multiple objects with one hand Year: (2024)
Ref_id:b36 Title: 9dtact: A compact vision-based tactile sensor for accurate 3d shape reconstruction and generalizable 6d force estimation Year: ()
Ref_id:b37 Title: Tactile image based contact shape recognition using neural network Year: (2012)
Ref_id:b38 Title: Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator Year: ()
Ref_id:b39 Title: Pose-and-shear-based tactile servoing Year: (2024)
Ref_id:b40 Title: Warp: A high-performance python framework for gpu simulation and graphics Year: (2022-03)
Ref_id:b41 Title: Dex-net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics Year: (2017)
Ref_id:b42 Title: Isaac gym: High performance gpu-based physics simulation for robot learning Year: (2021)
Ref_id:b43 Title: Recent progress in advanced tactile sensing technologies for soft grippers Year: ()
Ref_id:b44 Title: Cable manipulation with a tactile-reactive gripper Year: (2021)
Ref_id:b45 Title: Taxim: An example-based simulation model for gelsight tactile sensors Year: (2022)
Ref_id:b46 Title: Difftactile: A physics-based differentiable tactile simulator for contact-rich robotic manipulation Year: (2005)
Ref_id:b47 Title: Robotics and the handbook Year: (2016)
Ref_id:b48 Title: A particle method for history-dependent materials Year: (1994)
Ref_id:b49 Title: Neuralfeels with neural fields: Visuotactile perception for in-hand manipulation Year: ()
Ref_id:b50 Title: Tacto: A fast, flexible, and open-source simulator for high-resolution vision-based tactile sensors Year: (2022)
Ref_id:b51 Title: Doorbot: Closed-loop task planning and manipulation for door opening in the wild with haptic feedback Year: ()
Ref_id:b52 Title: The tactip family: Soft optical tactile sensors with 3d-printed biomimetic morphologies Year: (2018)
Ref_id:b53 Title: Factors influencing the force control during precision grip Year: (1984)
Ref_id:b54 Title: Sapien: A simulated part-based interactive environment Year: (2020)
Ref_id:b55 Title: Efficient tactile simulation with differentiability for robotic manipulation Year: (2023)
Ref_id:b56 Title: Visual-tactile sensing for in-hand object reconstruction Year: ()
Ref_id:b57 Title: In-hand object classification and pose estimation with sim-to-real tactile transfer for robotic manipulation Year: (2023)
Ref_id:b58 Title: Exploiting kinematic redundancy for robotic grasping of multiple objects Year: (1982)
Ref_id:b59 Title: Dynamic reconstruction of hand-object interaction with distributed force-aware contact representation Year: ()
Ref_id:b60 Title: Gelsight: High-resolution robot tactile sensors for estimating geometry and force Year: (2017)
Ref_id:b61 Title: Tactile measurement with a gelsight sensor Year: (2014)
Ref_id:b62 Title: Measurement of shear and slip with a gelsight tactile sensor Year: (2015)
Ref_id:b63 Title: Tactile sensing in dexterous robot hands-review Year: (2015)
Ref_id:b64 Title: Embedding high-resolution touch across robotic hands enables adaptive human-like grasping Year: (2025)
Ref_id:b65 Title: Tactile-informed prior-free manipulation of articulated objects Year: (2009)
Ref_id:b66 Title: Dark, beyond deep: A paradigm shift to cognitive ai with humanlike common sense Year: (2020)
