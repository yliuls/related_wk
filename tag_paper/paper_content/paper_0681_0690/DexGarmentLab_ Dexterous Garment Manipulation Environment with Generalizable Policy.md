Title: DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy
Abstract: Garment manipulation is a critical challenge due to the diversity in garment categories, geometries, and deformations. Despite this, humans can effortlessly handle garments, thanks to the dexterity of our hands. However, existing research in the field has struggled to replicate this level of dexterity, primarily hindered by the lack of realistic simulations of dexterous garment manipulation. Therefore, we propose DexGarmentLab, the first environment specifically designed for dexterous (especially bimanual) garment manipulation, which features large-scale high-quality 3D assets for 15 task scenarios, and refines simulation techniques tailored for garment modeling to reduce the sim-to-real gap. Previous data collection typically relies on teleoperation or training expert reinforcement learning (RL) policies, which are labor-intensive and inefficient. In this paper, we leverage garment structural correspondence to automatically generate a dataset with diverse trajectories using only a single expert demonstration, significantly reducing manual intervention. However, even extensive demonstrations cannot cover the infinite states of garments, which necessitates the exploration of new algorithms. To improve generalization across diverse garment shapes and deformations, we propose a Hierarchical gArment-manipuLation pOlicy (HALO). It first identifies transferable affordance points to accurately locate the manipulation area, then generates generalizable trajectories to complete the task. Through extensive experiments and detailed analysis of our method and baseline, we demonstrate that HALO consistently outperforms existing methods, successfully generalizing to previously unseen instances even with significant variations in shape and deformation where others fail. Our project

Section: Introduction
The ability to manipulate various objects is critical for general robots. Despite advancements in the manipulation of rigid [12] and articulated [33] objects, deformable objects, and garments in particular, continue to pose substantial challenges [34,36,35] due to their highly variable geometries and intricate deformations. Despite this, humans handle garments with remarkable ease using dexterous hands, thanks to their superior adaptability, larger manipulation area, and coordinated finger control, highlighting the importance of equipping robots with similar capabilities. Dexterous (especially bimanual) hands enable stable and precise actions (such as catching, cradling, pinching, and smoothening, as shown in Fig. 1), and excel in complex tasks like tie knotting and assisted dressing, where multi-finger coordination ensures accurate and adaptive manipulation.
Dexterous garment manipulation faces three key challenges: (i) Data: The high-dimensional action space of dexterous hands and the complex nature of garments make policy learning data-intensive [14,6]. What's more, different garment manipulation tasks require different hand grasp poses to ensure smooth manipulation and make garments maintain the desired condition. Directly collecting real-world data is impractical due to high cost. Thus, researchers have pursued simulators for garment manipulation [23,26,39,16,25], but these simulators often rely on labor-intensive teleoperation
Single Expert Demo Dress 15 Task Scenes 2500+ Garment Glove Baseball Cap Trousers Scarf Bowl Hat Tops Coat Garment-Hands Interaction Catch Cradle Pinch Smoothen Task ...... Fling Hang Fold ... Diverse Positions ... Diverse Deformations ... Diverse Garment Shapes or expert RL policies to collect demonstrations, which are inefficient. (ii) Environment: Effective garment manipulation involves interactions not only with garments but also with rigid and articulated objects (such as hangers, the human body, etc.). Current simulators [23,39] lack the necessary level of realism and physical accuracy for complex interactions, particularly with dexterous hands. (iii) Algorithm: Garment manipulation requires understanding diverse geometries, complex states, and difficult goals. Existing reinforcement learning (RL) [44,23] or imitation learning [1,6] approaches often require intricate task-specific reward designs or extensive demonstrations, limiting their scalability to real-world applications.
To address these challenges, we introduce DexGarmentLab (Fig. 1), an environment built upon Isaac Sim and specially designed for dexterous (especially bimanual) garment manipulation, featuring:
(1) Diverse and Realistic Environments: A large-scale dataset of more than 2,500 garments in 8 categories from ClothesNet [45], high-quality 3D assets for 15 task scenarios, paired with advanced simulation techniques to reduce sim-to-real gap. (2) Automated Data Collection Pipeline: An automated pipeline that generates diverse demonstrations based on garment structural correspondence with the help of a single expert demonstration, facilitating the generation of large and varied datasets without requiring manual intervention. (3) Generalizable Policy: a Hierarchical gArment-manipuLation pOlicy (HALO) which leverages affordance (for locating garment manipulation areas) and diffusion method (for generating trajectories based on garment and scene), achieving better generalization in garment manipulation than previous imitation learning algorithms.
In summary, our contributions include:
• We introduce DexGarmentLab Environment, the first simulation environment for dexterous (especially bimanual) garment manipulation, featuring a wide range of task scenarios, high-quality assets, and realistic physical interactions.
• We propose a pipeline for automated data collection, generating diverse demonstrations in various task scenarios and reducing the need for manual intervention.
• We propose Hierarchical gArment-manipuLation pOlicy (HALO), a novel hierarchical framework that leverages affordance and diffusion method to enable generalizable manipulation of diverse garments.
• Extensive experiments and detailed analysis of our approach and baseline in both simulation and real-world settings, demonstrating its data efficiency and generalization ability significantly outperforming baseline methods.
2 Related Work
this section cite: ['b11', 'b32', 'b33', 'b35', 'b34', 'b13', 'b5', 'b22', 'b25', 'b38', 'b15', 'b24', 'b22', 'b38', 'b43', 'b22', 'b0', 'b5', 'b44']

Section: Deformable Object Simulation
Current deformable simulations [18,23,39,21] are limited in the types of objects they support and lack realistic physical interactions, which hinders dexterous garment manipulation research. For instance, softgym [23] is confined to simulating tops and trousers while fluidlab [39] can only simulate fluids. Although Lu et al. [25] extends to various object types, it relies on attaching invisible cubes to garments, failing to simulate realistic physical interactions. In contrast, we introduce DexGarmentLab, which incorporates extra physical parameters (adhesion, friction, particle-adhesion/friction-scale, etc.) to guarantee stable and realistic interaction between garments and robots. Additionally, simulators [23,26,39,16,25] often rely on labor-intensive teleoperation or expert RL policies to collect demonstrations, which are inefficient. Thus, we propose an automated pipeline that exploits structural correspondence across garment categories to collect demonstrations, eliminating manual effort. Table 1 shows comparisons between DexGarmentLab and other environments.
Softgym × ✓ ✓ × × Manual × × PyBullet × ✓ × × × Manual × × Fluidlab × × ✓ × × Manual × × Dexterous Gym × × × × × Manual × × Sapien ✓ × ✓ ✓ × Manual × × GarmentLab ✓ ✓ ✓ × × Manual × × DexGarmentLab ✓ ✓ ✓ ✓ ✓ Automated ✓ ✓
this section cite: ['b17', 'b22', 'b38', 'b20', 'b22', 'b38', 'b24', 'b22', 'b25', 'b38', 'b15', 'b24']

Section: Dexterous Manipulation
Dexterous manipulation has emerged as a critical research frontier, with applications in tasks like inhand manipulation [40,15,29,37,8], articulated object manipulation [4,19,9,42] and deformable object manipulation [21,38,43]. However, most researches focus on single-handed manipulation, overlooking bimanual dexterity, which is essential for tasks like cradling garments and pinching gloves (Shown in Fig. 1). While recent efforts have demonstrated bimanual capabilities in specialized contexts including lid manipulation [22], dynamic object interception [17], and articulated object manipulation [42], these approaches remain fundamentally limited to rigid-body dynamics. Our work performs the first investigation on the learning-based bimanual dexterous manipulation of garment and build the pioneering environment with diverse scenarios covering different garment categories.
this section cite: ['b39', 'b14', 'b28', 'b36', 'b7', 'b3', 'b18', 'b8', 'b41', 'b20', 'b37', 'b42', 'b21', 'b16', 'b41']

Section: Garment and Deformable Object Manipulation
While much research has focused on manipulating simple deformable objects like square-shaped cloths [34,38,24], ropes and cables [31,34,38], and bags [2,7], garment manipulation presents a substantial challenge. Garment manipulation involve diverse geometries, complex deformations, and fine-grained actions. Many existing studies on dexterous garment manipulation rely on optimize method [3,13], which struggles with the high freedom of bimanual dexterous hands. Zhaole et al. [44], Lin et al. [23] attempt to solve bimanual dexterous manipulation tasks with reinforcement learning (RL). However, they require intricate reward designs tailored to specific manually designed tasks. Avigal et al. [1], Canberk et al. [6] rely on large-scale annotated data, which is labor-intensive and time-consuming, hindering the scalability in the scenarios of real-world applications. In this paper, we introduce Hierarchical gArment-manipuLation pOlicy (HALO) which leverages affordance and diffusion method to facilitate manipulating diverse unseen category-level garments in multiple tasks with different scene configurations.
this section cite: ['b33', 'b37', 'b23', 'b30', 'b33', 'b37', 'b1', 'b6', 'b2', 'b12', 'b43', 'b22', 'b0', 'b5']

Section: DexGarmentLab Environment
In this section, we present the construction of DexGarmentLab, the first environment specifically designed for dexterous (especially bimanual) garment manipulation and built upon IsaacSim 4.5.0.
this section cite: []

Section: DexGarmentLab Environment
Setup Observation Space. The observation space includes both proprioceptive and visual data. Proprioceptive data comprises robot joint positions, end-effector 6D poses, and other kinematic information. Grasp with AttachmentBlock (from GarmentLab) Grasp with AttachmentBlock (DexGarmentLab) Grasp with Our Precise Simulation (DexGarmentLab) TRANSFER Red Objects in Pictures Mean AttachmentBlock AttachementBlock method to dexterours hands and set red block at the tip of each finger (ten blocks totally). The performance is not so good, as described in 3.2. Right: Our method (DexGarmentLab) can make the interactions between dexterous hands and garment more natural. Visual data includes point clouds captured by depth camera and RGB images. The point cloud is cropped to the robot's workspace and down-sampled for efficiency.
this section cite: []

Section: Tops

this section cite: []

Section: GarmentLab

this section cite: []

Section: Trousers

this section cite: []

Section: Action Space.
The action space is a 60-dimensional vector: 6 DoF for each arm and 24 DoF for each Shadow Hand. The UR10e arms can be controlled using both Inverse-Kinematics (IK) Controller given end-effector 6D poses and Proportional-Derivative (PD) Controller given joint positions, while Shadow Hands are controlled using PD controller based on joint positions.
this section cite: []

Section: DexGarmentLab Physcial Simulation
Simulation Method. To achieve realistic simulation, we employ methods tailored to the physical properties of garments. Large garments (e.g., tops, dresses, trousers, etc.) are simulated using Position-Based Dynamics (PBD) [27], while small, elastic items (e.g., gloves, hats) are modeled via the Finite Element Method (FEM) [5]. We provide detailed introduction and selection reason about PBD and FEM in Appendix I.1. Human avatars are represented by articulated skeletons with rotational joints and a skinned mesh for lifelike rendering.
this section cite: ['b26', 'b4']

Section: Key Design for Physical Garment Simulation.
PBD is widely used for simulating most garments, but its loosely connected particles often allow grippers to penetrate the garment without achieving effective lifting. GarmentLab introduces attach blocks to address this, enabling garment-gripper attachment (Fig. 2, left). However, this approach fails to capture realistic interactions, resulting in unnatural sagging when applied to dexterous hands (Fig. 2, middle). Moreover, even minimal contact-such as a single finger block touching the garment-can establish attachment and lift the garment, which is clearly unreasonable.
Therefore, we introduce adhesion (between particle and rigid), friction (between particle and rigid) and particle-scale (between particles) parameters to enhance realism. Benefiting from friction and adhesion, dexhands can grasp and lift garments based on physical force without attach blocks (Fig. 2, right), while particle-adhesion (or -friction)-scale stabilizes the particle system, preventing excessive self-collisions between particles which cause garments to become disorganized (Fig. 3). We provide more details in Appendix A.
Fling Tops Fling Dress Fling Trousers Fold Tops Fold Dress Hang Dress Wear Baseball Cap Fold Trousers Hang Tops Hang Trousers Hang Coat Wear Scarf Wear Bowl Hat Store Tops Wear Glove
Figure 4: DexGarmentLab Simulation Environment. We introduce 15 garment manipulation tasks across 8 categories, encompassing both garment-self-interaction (e.g., Fling, Fold) and garmentenvironment-interaction (e.g., Hang, Wear, Store) scenarios. In garment-self-interaction tasks, key variables include garment position, orientation, and shape. In garment-environment-interaction tasks, environment-interaction assets positions (e.g., hangers, pothooks, humans, etc.) are also considered.
this section cite: []

Section: Asset Selection and Annotation
We use garment models from the ClothesNet dataset [45], which contains over 2,500 garments across 8 categories (e.g., tops, coats, trousers, dresses, etc.), and build environment-interaction assets (such as hangers, pothooks, humans, etc.). We provide plain meshes customizable with colors and textures for garments to support both realistic and controlled experimental setups. Controlled randomness in placement for both garments and environment-interaction assets-through limited rotations and translations-maintains task feasibility while enhancing generalization in policy learning.
this section cite: ['b44']

Section: DexGarmentLab Tasks
Dexterous (especially bimanual) garment manipulation is vital for domestic applications, yet it has not been thoroughly explored in existing research. To address this, we introduce 15 tasks across 8 garment categories (Fig. 4). Further details on these tasks are available in Appendix K.
this section cite: []

Section: Automated Data Collection
Collecting data through teleoperation or RL is highly labor-intensive, especially for dexterous garment manipulation tasks, due to the diverse shapes and deformations of garments and the highdimensional action space of dexterous hands. This makes automated data collection essential, with the key challenges being: 1) identifying appropriate manipulation points across different garment configurations, and 2) generating task-specific hand poses accordingly.
In our proposed automated data collection pipeline, for a given task, we begin with a single expert demonstration to extract key information: hand grasp poses, task sequences, and demo grasp points on the garment. Leveraging the Garment Affordance Model (Refer Sec. 4.1), we use affordance to identify target grasp points on novel garments with diverse deformations corresponding to demo grasp points. Then, the pipeline executes the task sequence based on inferred points and hand grasp poses, thereby enabling efficient and scalable data collection. Sec. 4.2 explains the whole procedure.
this section cite: []

Section: Garment Affordance Model (GAM)
Built upon the UniGarmentManip [35] framework, GAM leverages structural and correspondence consistency across category-level garments, enabling the identification of target grasp points on category-level novel garments. For training process, as shown in Fig. 5 (Red Part), we employ a Skeleton Merger [32] network architecture to obtain the skeleton point correspondences between flat garments while adopting the point tracing method in simulation to establish correspondences between the flat garment and its deformed version. Using InfoNCE [20] loss function, we train PointNet++ [28] to pull features of positive corresponding point correspondences closer while pushing apart negative corresponding pairs, which enhances dense visual correspondence by enabling alignment across different garments in various states. Please refer [35] for more details.
POINTNET++ Skeleton Merger InfoNCE Loss Corresponding Pairs Positive Pair Negative Pair Back Propogation Training Process Execution STEP Ⅰ STEP Ⅱ STEP Ⅲ Garment Affordance Model POINTNET++ Demo Point Cloud Operated Point Cloud Demo Points Extract …… Demo Points Features Match …… Target Points Features
this section cite: ['b34', 'b31', 'b19', 'b27', 'b34']

Section: Inference Process

this section cite: []

Section: Garment Affordance Model

this section cite: []

Section: Single Expert Demonstration
Demo Task Configuration Demo Hand Grasp Pose
Demo Points Demo Task Sequence LeapMotion teleoperate Close Open Demo Grasp Pose One Shot Transfer It's worth mentioning that, to enable GAM to handle point clouds with translation and scale invariance, we pre-normalize the input demo/operated point cloud into a canonical space. This ensures that GAM maintains generalization ability when faced with garments undergoing different translations and scales. As for the issue of rotational invariance, we consider it as the garment's deformation state. By generating sufficient samples with varying rotations, we enable GAM to effectively learn the correspondences of garments under different rotations.
For inference process, as shown in Fig. 5 (Green Part), with pretrained GAM, given one demo garment point cloud O, demo grasp points (p 1 , p 2 , ...) and one operated garment point cloud O ′ , we can obtain the demo grasp points features (f p1 , f p2 , ...) and operated garment observation features. By dotting demonstration grasp points features and operated garment observation features to get similarities and selecting features with biggest similarity scores, we can get corresponding grasp points (p ′ 1 , p ′ 2 , ...) on O ′ .
this section cite: []

Section: Automated Data Collection Pipeline
Here we give a detailed description about our automated data collection pipeline shown in Fig. 5.
Firstly, We obtain demo grasp points, demo task sequences, and demo hand grasp poses from a single expert demonstration. In our actual operation process, while grasp points and task sequences are manually defined, hand poses are generated using the LeapMotion solution (see Appendix B).
Once target grasp points are identified on the operated garment using GAM, with demo task sequences, we control the robotic arms via inverse kinematics (IK) to execute sequential operations while controlling dexterous hands using PD controller based on the joint positions from demo hand grasp poses. It is important to note that the task trajectories are not fixed but are adapted based on the garment's shape and length. For example, in garment-self-interaction tasks, such as folding, the lifting height is adjusted according to the sleeves and overall garment length to ensure proper folding.
In garment-environment-interaction tasks, such as hanging, both the lifting height and placement position are adapted to align the garment's center with the hanger, preventing slippage. These adjustments reflect common and reasonable actions in real-world scenarios, and introducing such variations increases the task difficulty. The details about tasks can be found in Appendix K.
During task execution, we can simultaneously record various information (such as images, point clouds, robot joint states, etc.) within the simulation environment. These serve as expert demonstration data for subsequent offline training of the policy. Details about recorded information can be found in Appendix D.
this section cite: []

Section: Generalizable Policy
When dealing with garments, which exhibit highly complex deformation states, current mainstream imitation learning (IL) algorithms (e.g. Diffusion Policy [10], Diffusion Policy 3D [41]) show relatively poor generalization (as evidenced by our experimental results shown in Tab. 2). The main issue is that IL-based trajectories fail to accurately reach the target manipulation points on garments with new shapes and deformations, while also being unable to generate suitable trajectories based on the garment's own shape and structure, ultimately leading to manipulation failures.
To address this, we propose Hierarchical gArment manipuLation pOlicy (HALO), a generalizable policy to solve the manipulation of garments with complex deformations and uncertain states. HALO is decomposed into two major stages, as shown in Fig. 6 Interaction-Object Point Cloud
(optional) Hanger Human Pothook STAGE Ⅰ Generalizable Affordance Points Demo Point Cloud Demo Points Operated Point Cloud Garment Affordance Model POINTNET++ concatenate STAGE Ⅱ Generalizable Trajectories Structure-Aware Diffusion Policy Environment Perception Decision INPUT CALCULATE LOSS INPUT EXECUTE Train Process Evaluation Process Compact Scene Feature Robot State Feature conditioning Noised Action Input Compact Scene Feature Final Action Output P E R C E P T I O N D E C E S I O N Expert Demonstrations Max Pooling Max Pooling Environment Point Cloud Feature demo point for left hand demo point for right hand Target Point (Left) Affordance Heatmap Target Point (Right) Affordance Heatmap Generate Guide Move to Target Manipulation Area Operated Garment Point Cloud POINTNET++ MLP target point (left) affordance target point (right) affordance Due to the poor generalization ability of current mainstream methods such as DP and DP3 for complex and variable garment manipulation scenarios, we propose Structure-Aware Diffusion Policy (SADP), a garment-environment-generalizable diffusion policy that improves the generalization for different garment shapes and scene configurations, thereby enabling the smooth generation of subsequent trajectories after moving to the target manipulation area guided by GAM.
SADP fundamentally follows the framework of Diffusion Policy [10], with the primary distinction lying in its observation representation, denoted as s, which is elaborated below.
With operated garment point cloud and left / right target point affordances generated by GAM, we concatenate them together and use PointNet++ [28] to extract garment feature F garment , while using MLP-based Feature Extractors to extract interaction-object feature F object . F garment and F object are concatenated into a compact scene feature F scene . At each timestep, the full environment point cloud O environment and the robot state O state are encoded using MLP and fused with F scene to form the denoising condition s for SADP. As for Garment-Self-Interaction tasks without interaction-object point cloud, we only use F garment to be F scene , which means interaction-object point cloud is optional. Here, F garment captures current garment state (position, shapes, structure, etc.), while F object reflects current interaction-object state (position, etc.).
Through experimental validation, SADP exhibits better generalization capabilities. We will further illustrate this advantage with experimental results in Sec. 6.2. Training details can be found in Appendix H. 6 Experiment 6.1 Environment Setup Tasks and Environments. We evaluate our method on 14 garment manipulation tasks with varying deformation characteristics. Detailed environment specifications including scene randomization parameters, success metrics, and train/test configurations are provided in the Appendix K.
this section cite: ['b9', 'b40', 'b9', 'b27']

Section: Demonstration Collection.
Using our automated data collection pipeline (Sec. 4), we acquired 100 demonstrations per task, with 30-100 environment steps in each demonstration. The time required to collect a demonstration varies between approximately 30-80 seconds depending on the task, which is significantly faster than data collection through teleoperation. We make comparison between autonomously collected data and teleoperation data in Appendix F.
Baselines. We compare against two state-of-the-art diffusion-based approaches across all the garment manipulation tasks: 1) Diffusion Policy (DP) [10], which utilizes images as observations to generate actions via diffusion. 2) 3D Diffusion Policy (DP3) [41], which replaces image inputs in the Diffusion Policy with point clouds. What's more, we have also additionally included four new baselines for comparison: ACT (IL), pi0 (VLA), RDT (VLA), and Eureka (RL+VLM) and select representative tasks for evaluation, including Fling Dress, Fold Trousers, Hang Coat, and Wear Bowlhat in simulation, as well as Fold Tops in real world, which can be found in Appendix G.
this section cite: ['b9', 'b40']

Section: Ablations.
To analyze components, we evaluate two ablated variants: 1) w/o GAM: Excludes the Garment Affordance Model (GAM), using SADP for trajectory execution. 2) w/o SADP: Removes Structure-Aware Diffusion Policy (SADP), using GAM + DP3 for trajectory execution.
Metrics. Each task is evaluated over 50 episodes with three different seeds. We report success rates as M ean ± Std across all trials.  Through Tab. 2, we find that the performance of HALO markedly decreases when GAM is excluded. The integration of GAM leverages dense visual correspondence, thereby enhancing the model's performance to locate precise manipulation area (shown in Fig. 7), particularly for tasks with significant variability in garment shapes and deformations. Affordance Points Heatmap Affordance Points Heatmap move to target manipulation area move to target manipulation area Initial State Different Configuration Garment Affordance Model (GAM) Generate Generalizable Points for Pre-Movement Structure-Aware Diffusion Policy (SADP) Generate Generalizable Trajectories based on Garment 0 T lift height lift height final state lift procedure final state lift procedure Figure 8: HALO's whole procedure.
Using "Hang Coat" as an example. GAM first infers target manipulation points for robot's movement, enabling generalization across different garments. Then, SADP generates trajectories based on the garment and scene configurations. Despite variations in garment shape, length, and pothook positions across scenes, SADP adapts accordingly, moving to accurate positions and lifting coats to appropriate heights to successfully hang them on the pothook. the tasks. For instance, as shown in Fig. 7, in the "Hang Trousers" task, the shape of the trousers determines the appropriate hanging height and forward-moving distance; the HALO can adjust its trajectory to complete the task. In the "Fold Tops" task, the HALO can adjust the folding position according to the garment structure, making the folds neater.
HALO enables precise manipulation point inference and generates robust policies for diverse garment and scene configurations. Figure 8 further illustrates this.
this section cite: []

Section: Real-World Experiments
There are two ways using HALO for garment manipulation tasks:
1) transfer the data collection pipeline from simulation to real world, conduct automated data collection and policy training directly in the real world.
2) train the policy in the simulation and transfer the policy to the real world.
this section cite: []

Section: Experiments and Results Analysis for Way 1
we adopt way 1 to demonstrate the effectiveness of HALO. In the Appendix F Tab. 5, we report the efficiency and success rate of real-world data collection, highlighting the strong sim-to-real performance of GAM.
Fold Tops Hang Tops Wear Scarf Wear Hat Tops Ⅰ | Step 1 Tops Ⅱ | Step 2 Tops Ⅲ | Step 3 Tops Ⅰ Tops Ⅱ Tops Ⅲ Scarf Ⅰ Scarf Ⅱ Scarf Ⅲ Hat Ⅰ Hat Ⅱ Hat Ⅲ Test Sample Initial State Execution Initial State Execution Initial State Execution affordance affordance affordance affordance affordance affordance affordance affordance affordance affordance affordance affordance Setup. Our setup comprises two RealMan RM75-6F (Arms) with Psibot G0-R (Dexterous Hands) and a RealSense D435 camera (As shown in Appendix E). We use Segment-Anything-2 [30] to segment the garment and interaction object from the scene and obtain corresponding point clouds. Evaluation. We evaluate our proposed method on 4 tasks: Fold Tops, Hang Tops, Wear Scarf, and Wear Hat. For each task, we have 3 distinct garments per category, each with 5 initial deformations. Shown in Tab. 3, our method outperforms all baselines. Fig. 9 demonstrates the excellent performance of our proposed method.
this section cite: ['b29']

Section: Experiments and Results Analysis for Way 2
Sim-to-real transfer from the simulation environment remains an important aspect of our study. To this end, as shown in Fig. 10, we aligned the settings of both the simulation and real-world environments by using the same hardware setup-Shadow Hand and UR10e-and selected two tasks, Hang Trousers and Wear Hat, for policy-level sim-to-real transfer, which means way 2. The evaluation criteria follow those used in the previous real-world experiments.
It is worth noting that sim-to-real performance is more sensitive to point cloud noise. To address the limited precision of the Realsense D435 in this context, we employed a Kinect camera for more accurate point cloud acquisition. Table 4: Performance Impact of Adding Real-World Data to Simulation Data Task Name Only Simulation Data Simulation Data + 15 Real-World Data Hang Trousers 8 / 15 (53.3%) 13 / 15 (86.7%) Wear Hat 9 / 15 (60.0%) 13 / 15 (86.7%)
Experimental results in Tab. 4 show that due to the gap between simulation and the real world, training the policy solely on simulated data leads to a drop in sim-to-real performance. Incorporating a small amount of real-world data into the training process can effectively enhance the policy's generalization ability.
this section cite: []

Section: Conclusion
In this paper, we introduce DexGarmentLab, the first simulation environment designed to address the challenges of dexterous (especially bimanual) garment manipulation. Our work mainly makes three key contributions, including DexGarmentLab Environment, Automated Data Collection and Generalizable Policy. Through extensive experiments, we demonstrate that our approach can effectively learn complex manipulation tasks with minimal supervision and generalize across a wide range of garment shapes and deformation states in both simulation and real-world environments. The limitation of our work is discussed in Appendix I.
this section cite: []

Section: References
Ref_id:b0 Title: Learning efficient bimanual folding of garments Year: (2022)
Ref_id:b1 Title: Bag all you need: Learning a generalizable bagging strategy for heterogeneous objects Year: (2023)
Ref_id:b2 Title: Dexterous manipulation of cloth Year: (2016)
Ref_id:b3 Title: Dexart: Benchmarking generalizable dexterous manipulation with articulated objects Year: (2023-06)
Ref_id:b4 Title: Finite element method. Wiley encyclopedia of computer science and engineering Year: (2007)
Ref_id:b5 Title: Cloth funnels: Canonicalized-alignment for multi-purpose garment manipulation Year: (2022)
Ref_id:b6 Title: Autobag: Learning to open plastic bags and insert objects Year: (2023)
Ref_id:b7 Title: Bi-dexhands: Towards human-level bimanual dexterous manipulation Year: (2024)
Ref_id:b8 Title: Object-centric dexterous manipulation from human motion data Year: (2024)
Ref_id:b9 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b10 Title: Bunny-visionpro: Real-time bimanual dexterous teleoperation for imitation learning Year: (2024)
Ref_id:b11 Title: Learning part motion of articulated objects using spatially continuous neural implicit representations Year: (2023)
Ref_id:b12 Title: Fem-based deformation control for dexterous manipulation of 3d soft objects Year: (2018)
Ref_id:b13 Title: Flingbot: The unreasonable effectiveness of dynamic manipulation for cloth unfolding Year: (2021)
Ref_id:b14 Title: Dieter Fox, and Gavriel State. Dextreme: Transfer of agile in-hand manipulation from simulation to reality Year: (2022)
Ref_id:b15 Title: Difftaichi: Differentiable programming for physical simulation Year: (2020)
Ref_id:b16 Title: Dynamic handover: Throw and catch with bimanual hands Year: (2023)
Ref_id:b17 Title: A soft-body manipulation benchmark with differentiable physics Year: (2021)
Ref_id:b18 Title: Dexsim2real 2 : Building explicit world model for precise articulated object dexterous manipulation Year: (2024)
Ref_id:b19 Title: Contrastive predictive coding based feature for automatic speaker verification Year: (2019)
Ref_id:b20 Title: Dexterous deformable object manipulation with human demonstrations and differentiable physics Year: (2023)
Ref_id:b21 Title: Twisting lids off with two hands Year: (2024)
Ref_id:b22 Title: Softgym: Benchmarking deep reinforcement learning for deformable object manipulation Year: (2021)
Ref_id:b23 Title: Learning visible connectivity dynamics for cloth smoothing Year: (2022)
Ref_id:b24 Title: Garmentlab: A unified simulation and benchmark for garment manipulation Year: (2024)
Ref_id:b25 Title: Unified particle physics for real-time applications Year: (2014-07)
Ref_id:b26 Title: Position based dynamics Year: (2007)
Ref_id:b27 Title: Pointnet++: Deep hierarchical feature learning on point sets in a metric space Year: (2017)
Ref_id:b28 Title: In-Hand Object Rotation via Rapid Motor Adaptation Year: ()
Ref_id:b29 Title: Segment anything in images and videos Year: (2024)
Ref_id:b30 Title: Learning to rearrange deformable cables, fabrics, and bags with goal-conditioned transporter networks Year: (2023)
Ref_id:b31 Title: Skeleton merger: an unsupervised aligned keypoint detector Year: (2021)
Ref_id:b32 Title: Vat-mart: Learning visual action trajectory proposals for manipulating 3d articulated objects Year: (2022)
Ref_id:b33 Title: Learning foresightful dense visual affordance for deformable object manipulation Year: (2023)
Ref_id:b34 Title: Unigarmentmanip: A unified framework for category-level garment manipulation via dense visual correspondence Year: (2024)
Ref_id:b35 Title: Garmentpile: Pointlevel visual affordance guided retrieval and adaptation for cluttered garments manipulation Year: (2025)
Ref_id:b36 Title: Canonical representation and force-based pretraining of 3d tactile for dexterous visuo-tactile policy learning Year: (2024)
Ref_id:b37 Title: Learning to manipulate deformable objects without demonstrations Year: (2020)
Ref_id:b38 Title: Fluidlab: A differentiable environment for benchmarking complex fluid manipulation Year: (2023)
Ref_id:b39 Title: Rotating without seeing: Towards in-hand dexterity through touch Year: (2023)
Ref_id:b40 Title: d diffusion policy Year: (2024)
Ref_id:b41 Title: Artigrasp: Physically plausible synthesis of bi-manual dexterous grasping and articulation Year: (2024)
Ref_id:b42 Title: Dexdlo: Learning goal-conditioned dexterous policy for dynamic manipulation of deformable linear objects Year: (2024)
Ref_id:b43 Title: Learning goal-conditioned dexterous policy for dynamic manipulation of deformable linear objects Year: (2024)
Ref_id:b44 Title: Clothesnet: An information-rich 3d garment model repository with simulated clothes environment Year: (2023)
