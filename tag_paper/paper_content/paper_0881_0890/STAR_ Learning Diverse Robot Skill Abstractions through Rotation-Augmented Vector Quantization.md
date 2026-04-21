Title: STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization
Abstract: Transforming complex actions into discrete skill abstractions has demonstrated strong potential for robotic manipulation. Existing approaches mainly leverage latent variable models, e.g., VQ-VAE, to learn skill abstractions through learned vectors (codebooks), while they suffer from codebook collapse and modeling the causal relationship between learned skills. To address these limitations, we present Skill Training with Augmented Rotation (STAR), a framework that advances both skill learning and composition to complete complex behaviors. Specifically, to prevent codebook collapse, we devise rotation-augmented residual skill quantization (RaRSQ). It encodes relative angles between encoder outputs into the gradient flow by rotation-based gradient mechanism. Points within the same skill code are forced to be either pushed apart or pulled closer together depending on gradient directions. Further, to capture the causal relationship between skills, we present causal skill transformer (CST) which explicitly models dependencies between skill representations through an autoregressive mechanism for coherent action generation. Extensive experiments demonstrate the superiority of STAR on both LIBERO benchmark and realworld tasks, with around 12% improvement over the baselines.

Section: Introduction
The challenge of modeling multitask visuomotor policy has long been a central problem in robotic manipulation (Levine Figure 1. Comparison between naive VQ and our RaRSQ approach in the skill learning process. Top: Overview of skill quantization framework. Bottom: Visualization of gradient flow and codebook updates across three stages (before update, during gradient flow, and after update), where RaRSQ maintains geometric relationships between embeddings, leading to more diverse skills. et al., 2016;Zhu et al., 2018). Individual manipulation tasks already pose significant challenges like multimodal action distributions (Mandlekar et al., 2021), while these challenges are substantially amplified in the multitask setting. This results in a highly entangled action space where characteristics of different tasks interact and overlap, making it challenging to learn complex manipulation behaviors (Lv et al., 2024).
An intuitive approach to alleviate this problem is to learn structured representations of manipulation behaviors by decomposing complex actions into simpler, reusable skill abstractions (Fu et al., 2024;Sharma et al., 2019). This hierarchical framework reflects the compositional structure inherent to manipulation tasks, enabling the systematic decomposition of complex behaviors into sequences of skill abstractions. Recent studies (Ju et al., 2024;Wu et al., 2024) have demonstrated promising results using latent variable models (LVM) to discretize continuous action spaces into learned skills. These methods enable a more efficient representation and composition of complex behaviors. However, while this discretization paradigm transforms continuous actions into skills and provides a structured representation for complex behaviors, existing LVM-based methods face two critical limitations in skill learning and composition (Garg et al., 2022;Lee et al., 2024). First, techniques like VQ-VAE suffer from codebook collapse (Mentzer et al., 2023;Roy et al., 2018). Most codebook vectors remain unused during training, with only a small subset being frequently utilized for encoding diverse manipulation skills. This severely limits the capacity to capture the rich variety of robot behaviors. We argue that this limitation stems from the straight-through gradient estimator (STE) in VQ-VAE (Van Den Oord et al., 2017). During training, as illustrated in the middle column of Fig. 1, STE assigns identical gradients to all encoder embeddings (hollow circles) that are quantized to the same codebook vector (orange solid circles). This oversimplified gradient assignment ignores the inherent geometric relationships between different embeddings within the same partition (regions separated by curved blue decision boundaries), leading to suboptimal codebook updates and eventual collapse of the representation space.
Second, existing approaches struggle with effective skill composition, particularly in complex, long-horizon tasks that require precise coordination of multiple skills (Mete et al., 2024). While some methods adopt residual quantization (RQ) (Zeghidour et al., 2021) to decompose skills into multiple levels for more precise representation, they fail to model the dependencies between different skill abstractions (Lee et al., 2024). This makes it difficult to generate coherent and temporally consistent actions for multi-stage manipulation tasks, where skills need to be carefully sequenced and composed.
To address these fundamental challenges, we propose Skill Training with Augmented Rotation (STAR), a novel framework that advances both skill learning and composition for robot manipulation. Our key insight is that encoding geometric relationships between action sequences into the residual quantization process is crucial for learning diverse and reusable skills, rather than relying on the oversimplified gradient assignment of straight-through estimation. Specifically, (1) to prevent codebook collapse, we devise rotationaugmented residual skill quantization (RaRSQ), which combines multi-level residual encoding with rotation-based gradient mechanisms. The residual structure progressively captures skills at different abstraction levels, while the rotationaugmented gradient flow enables points within the same skill code to be either pushed apart or pulled closer together based on their geometric relationships. Compared to naive VQ-VAE where similar embeddings are forced to have identical gradients, Fig. 1 demonstrates that RaRSQ prevents embeddings from collapsing to the same codebook vector, leading to more diverse skill representations; and (2) for effective skill composition, we present causal skill transformer (CST) which explicitly models dependencies between skill representations through an autoregressive mechanism for coherent action generation. By leveraging the hierarchical nature of residual quantization, CST sequentially predicts skill codes from coarse to fine levels and incorporates an offset prediction mechanism from BeT (Shafiullah et al., 2022) to bridge the gap between discrete skills and continuous control. This combination of hierarchical prediction and continuous refinement enables precise control throughout extended sequences, making it particularly effective for long-horizon manipulation tasks. To summarize, our main contributions are as follows:
• A rotation-augmented residual skill quantization (RaRSQ) mechanism that maintains diverse skill representations by encoding relative angular relationships in gradient updates, while achieving precise skill abstraction through hierarchical residual encoding.
• A causal skill transformer (CST) that models skill dependencies through autoregressive prediction and enhances action precision via action refinement.
• Comprehensive experimental validation across multiple benchmarks and real-world tasks, demonstrating substantial improvements in both skill learning efficiency and task performance.
this section cite: ['b15', 'b42', 'b22', 'b20', 'b7', 'b28', 'b11', 'b35', 'b8', 'b14', 'b23', 'b26', 'b31', 'b24', 'b39', 'b14', 'b27']

Section: Related Work
Multi-task Imitation Learning. Multi-task robotic learning has been approached through various methods including supervised pre-training (Sun et al., 2023;Wu et al., 2023;Li et al.) and large-scale demonstration learning (Vuong et al., 2023;Brohan et al., 2023;Li et al., 2025). Recent advances have explored the use of generative models, with frameworks like diffusion models (Chi et al., 2023;Ze et al., 2024;Lv et al., 2025) and transformer-based approaches (Pertsch et al., 2025;Bharadhwaj et al., 2024) showing promising results in handling multimodal action distributions. The Behavior Transformer (BeT) (Shafiullah et al., 2022) demonstrated that policies operating in discretized action spaces can effectively model diverse behaviors, and introduced an offset prediction mechanism to handle the discretization-induced precision loss. Action Chunking Transformer (ACT) (Zhao et al., 2023) further addressed temporal correlations by predicting action chunks. Unlike existing methods, STAR advances this line of work by introducing novel mechanisms for learning structured skill representations while preserving the geometric relationships inherent in continuous action sequences.
Robotic Manipulation in Learned Latent Spaces. Latent Variable Models (LVMs) have emerged as powerful tools for learning structured representations in robotics (Yang et al., 2024;Luo et al., 2023;Bharadhwaj et al., 2023), particularly for offline imitation learning and skill abstractions. Among them, methods like LAPA (Ye et al., 2024), IGOR (Chen et al., 2024), leverage internet-scale human videos to learn transferable manipulation skills. Several works have explored discrete latent spaces for skill representation. PRISE (Zheng et al., 2024) employs BPE tokenization for temporal abstraction but struggles to effectively encode varied action distributions across tasks. TAP (Jiang et al., 2022) and H-GAP (Jiang et al., 2023) use self-supervised autoencoders for skill learning but rely heavily on state-based model predictive control, limiting their real-world applicability. QueST (Mete et al., 2024) learns discrete latent skills with temporal dependencies, but struggles to learn diverse skill representations. VQ-BeT (Lee et al., 2024) shares our motivation of using discrete latent skills for transformer-based policies, but their standard quantization approach suffers from codebook collapse and fails to capture temporal dependencies between skills. Unlike existing approaches, STAR addresses both the representational and temporal challenges through a two-stages framework that combines rotation-augmented quantization for preventing codebook collapse with explicit modeling of skill dependencies for coherent behavior generation.
this section cite: ['b29', 'b34', 'b3', 'b3', 'b17', 'b5', 'b38', 'b21', 'b25', 'b2', 'b27', 'b40', 'b36', 'b19', 'b1', 'b37', 'b4', 'b41', 'b9', 'b10', 'b24', 'b14']

Section: Method

this section cite: []

Section: Preliminaries
Residual VQ-VAE and STE. VQ-VAE with residual quantization transforms continuous data into hierarchical discrete representations through a multi-stage quantization process (Adiban et al., 2022). It consists of three key components: an encoder E, a decoder D, and multiple codebooks C i . Given an input x ∈ R n , the encoder E first maps it to a continuous latent code e ∈ R m . Then, e are quantized using D codebooks, where each codebook C i = {e (i,1) , ..., e (i,K) } contains K learnable vectors.
Starting with the initial residual r 0 = e, residual quantization iteratively performs nearest neighbor lookup and residual computation:
k d = Q(r d-1 ; C d ) = arg min k∈{1,...,K} ∥r d-1 -e (d,k) ∥ 2 2 (1) r d = r d-1 -e (d,k d )(2)
where k d is the selected code index at depth d, and r d is the remaining residual to be quantized by subsequent codebooks. The final quantized representation ê is obtained by summing the selected code vectors:
ê = D d=1 e (d,k d )(3)
The decoder D then reconstructs the input: x = D(ê). The model is trained with a combination of reconstruction and
Algorithm 1 Rotation-augmented Residual Skill Quantization (RaRSQ) Require: action sequence at:t+T , codebooks {C d } D d=1 1: z ← Encoder(at:t+T ) 2: r0 ← z 3: for d = 1 to D do 4: // Quantize current residual 5: k d ← arg min k ∥r d-1 -e (d,k) ∥ 2 2 6: // Compute rotation matrix that aligns r d-1 to e (d,k d ) 7: R d ← ComputeRotation(r d-1 , e d,k d ) 8: // Apply rotation with stop-gradient to preserve geometric structure 9: qd ← sg ∥e (d,k d ) ∥ ∥r d-1 ∥ R d r d-1 10: // Update residual for next level 11: r d ← r d-1 -qd 12: end for 13: ẑ ← D d=1 qd 14: â ← Decoder(ẑ) 15: return reconstructed action â, codes {k d } D d=1 commitment losses:
L = ∥x -D(ê)∥ 2 2 + ∥ sg(e) -ê∥ 2 2 + β∥e -sg(ê)∥ 2 2 (4
)
where sg(•) denotes the stop-gradient operator and β is a hyperparameter scaling the commitment loss for multi-stage residual learning stability (Lee et al., 2022).
Due to the non-differentiability of the quantization operation Q(•), the straight-through estimator (STE) is employed for backpropagation by simply copying gradients from ê to e through setting ∂ê/∂e = I. This residual approach enables more precise approximation than standard VQ-VAE -with codebook size K and depth D, it can effectively represent K D distinct quantization outputs while maintaining better computational efficiency and training stability.
this section cite: ['b0', 'b13']

Section: Rotation Trick.
To address the limitations of STE in preserving geometric relationships during gradient propagation, recent work (Fifty et al., 2024) proposes the rotation trick that transforms encoder outputs to codebook vectors via rotation and rescaling. For encoder output e and codebook vector q, it computes:
q = ∥q∥ ∥e∥ • R • e (5
)
where R is the rotation matrix that aligns e with q. During backpropagation, the rotation transformation preserves relative angles between gradients and vectors, enabling different points within the same quantization region to receive varying gradient updates based on their geometric relationships. This mechanism helps prevent codebook collapse and maintain diverse vector representations by encouraging appropriate exploration of the latent space.
this section cite: ['b6']

Section: ROTATION-AUGMENTED RESIDUAL SKILL QUANTIZATION
To learn expressive and reusable skill representations from continuous action sequences, we propose Rotationaugmented Residual Skill Quantization (RaRSQ). Our approach addresses two key limitations of standard VQ-VAE: codebook collapse and inefficient skill representation. By integrating rotation transformations with residual quantization, RaRSQ preserves geometric relationships between action sequences while enabling hierarchical skill encoding.
this section cite: []

Section: Skill Encoding Process.
Given an action sequence a t:t+T , we first encode it into a latent vector z = ϕ(a t:t+T ) through an encoder network ϕ (see Algorithm 1). RaRSQ then discretizes z through an iterative process as follows:
Starting with the initial residual r 0 = z, for each depth d = {1, ..., D}, we quantize and rotate the residual:
k d = arg min k ∥r d-1 -e (d,k) ∥ 2 2 (6) qd = sg ∥e (d,k d ) ∥ ∥r d-1 ∥ R d r d-1 (7) r d = r d-1 -qd (8
)
where e d,k d represents the k d -th vector in codebook C d , and sg[•] denotes the stop-gradient operator. The rotation matrix R d is computed as:
R d = I -2r d r T d + 2q d rT d-1(9)
where
qd = e (d,k d ) /∥e (d,k d ) ∥, rd-1 = r d-1 /∥r d-1 ∥ (10) rd = rd-1 + qd ∥r d-1 + qd ∥ (11
)
The final skill representation is obtained by summing the rotated quantized vectors:
ẑ = D d=1 qd (12
)
During backpropagation, gradients flow through the rotation matrices:
∂ẑ ∂r d-1 = ∥e (d,k d ) ∥ ∥r d-1 ∥ R d (13
)
This rotation-based gradient mechanism enables different points within the same quantization region to receive varying updates based on their geometric relationships, effectively preventing codebook collapse.
Theoretical Benefits. Our formulation offers three key advantages:
1. Improved Skill Diversity: The rotation-based gradient mechanism prevents codebook collapse by preserving geometric relationships between actions and corresponding skills. Our rotation transformation enables varied updates based on relative angular relationships, preventing skills from collapsing to a small subset of codes. 2. Hierarchical Skill Structure: The residual quantization naturally decomposes actions into a hierarchy, where k 1 captures coarse primitives while subsequent codes encode finer details, matching the inherent structure of manipulation tasks. 3. Enhanced Representation Capacity: The combination of residual quantization and rotation-augmented gradients enables representing K D distinct skills with only K codes per level, while maintaining low quantization errors compared to naive VQ approaches.
Training Objective. We train RaRSQ using a combination of reconstruction and commitment losses:
L = L recon + L commit(14)
L recon = ∥a t:t+T -ψ(ẑ)∥ 2 2 (15
)
L commit = β D d=1 ∥ sg[r d-1 ] - ∥e d,k d ∥ ∥r d-1 ∥ R d r d-1 ∥ 2 2 (16
)
where ψ is the decoder, sg[•] denotes stop-gradient, and β is a weighting coefficient. The reconstruction loss L recon ensures accurate action reconstruction, while the commitment loss L commit encourages the residuals to stay close to their corresponding rotated and scaled skill abstractions, maintaining geometric relationships during quantization.
this section cite: []

Section: CAUSAL SKILL TRANSFORMER
To effectively compose learned skills for sequential manipulation tasks, we propose the Causal Skill Transformer (CST) that combines autoregressive skill prediction with adaptive refinement. Our framework explicitly models the hierarchical dependencies between skills while enabling precise action generation through refinement.
this section cite: []

Section: Input Representation.
Given a sequence of observations o t-h:t = {(i k , p k )}, where i k and p k represent visual and proprioceptive inputs at timestep k, and a task instruction τ , we first encode the multimodal inputs using:h k = [f vis (i k ); f prop (p k )], where f vis and f prop are vision and proprioceptive encoders respectively. The transformerbased policy π θ then processes these encodings along with the task embedding to generate contextual features:
g t = π θ ([τ ; h t-h:t ]).(17)
Hierarchical Skill Prediction. Building on our residual quantization framework, CST models skill selection as a hierarchical process where each skill depends on previous choices:
P (k 1 , ..., k D |o t-h:t , τ ) = D d=1 P (k d |k <d , g t )(18)
where k <d represents all previously predicted skill codes, and k d ∈ {1, ..., K} denotes the index selected from the dth codebook. For each depth d, a prediction head ζ d outputs a categorical distribution over the K possible codebook indices, enabling the model to select appropriate skills at each level of abstraction. This autoregressive formulation is crucial as it captures the natural dependency structure in our residual skill space -coarse movement primitives must be selected before fine-grained adjustments.
Action Refinement. While the predicted codes can be directly decoded to actions using the decoder in RaRSQ, discretization of the continuous action space inevitably leads to some loss of fidelity (Shafiullah et al., 2022). Following BeT, we introduce a refinement mechanism through an offset prediction head to bridge this gap. Specifically, we add an offset head ζ ref to predict continuous refinements to the discretized actions. The final action is computed as:
ât = ψ( D d=1 R d e d,k d ) + ζ ref (g t ) (19
)
where ψ is the RaRSQ decoder from Section 3.2.2, e d,k d is the selected codebook vector at depth d, and R d is the corresponding rotation matrix.
Training Objective. We optimize our framework using a combination of skill prediction and refinement losses:
L = - D d=1 log P (k * d |k <d , g t ) + λ∥a t -ât ∥ 2(20)
where k * d are the ground truth codes from RaRSQ encoding of the expert action a t , and λ is a trade-off coefficient.
this section cite: ['b27']

Section: INFERENCE PROCESS
At inference time, our framework generates actions through an efficient two-stage process that balances exploration and precision. Given the current observation context o t-k:t and task instruction ℓ, we perform: Hierarchical Skill Selection. We first sample skill codes (k 1 , ..., k D ) autoregressively using nucleus sampling with temperature τ and threshold p:
k d ∼ NucleusSample(P (k d |k <d , g t ), p, τ )(21)
Action Generation and Execution. The sampled skill codes are mapped to their corresponding codebook vectors and combined with predicted refinements to generate actions:
ât:t+h = ψ( D d=1 e d,k d ) + ζ offset (g t )(22)
The system executes the generated action sequence and updates observations before re-planning. This rolling horizon approach allows our framework to adapt to environment dynamics while maintaining behavioral consistency through the learned skill space.
this section cite: []

Section: Experiment

this section cite: []

Section: Setup and Baselines
We evaluate STAR on two comprehensive manipulation benchmarks: LIBERO (130 tasks across five suites) and MetaWorld MT50 (50 distinct manipulation tasks), plus two real-world long-horizon tasks. Success Rate (SR) is measured over 50 episodes per task with three random seeds.
Detailed descriptions are in Appendix A. We compare STAR against three categories of state-of-the-art methods: (1) Discrete LVM approaches, (2) End-to-end imitation learning, (3) Large-scale VLA models. Detailed descriptions of these baselines can be found in Appendix A.5.1.
this section cite: []

Section: Overall Performance
As shown in
Table 1, STAR significantly outperforms all baselines across different LIBERO task suites, achieving 93.6% overall success rate and surpassing the previous state-of-the-art QueST by 12.1% (81.5%). The performance improvement is particularly pronounced on LIBERO-Long tasks (88.5% vs. 69.1%, +19.4%) and complex manipulation scenarios like LIBERO-Object (98.3% vs. 90.0%, +8.3%). Compared with other baselines, STAR demonstrates consistent improvements across all task categories. For basic manipulation tasks in LIBERO-Object and LIBERO-Spatial, our method achieves 7.2%-12.6% higher success rates. The improvement margins expand significantly to 18.3%-33.9% on more challenging LIBERO-Goal and LIBERO-Long tasks. This larger gap on complex tasks stems from the codebook ResNet-T Diffusion Policy ACT VQ-BeT QuesT STAR 80 82 84 86 88 90 92 94 Mean Success Rate (%) 87.3 88.4 88.2 87.7 90.6 92.7 Figure 3. Performance comparison on the MetaWorld MT50 benchmark. STAR achieves consistently superior performance (92.7%) compared to baseline methods across 50 manipulation tasks.
collapse issue in discrete latent approaches like VQ-BeT and QueST. Notably, STAR even outperforms large-scale models like Octo and OpenVLA despite their access to significantly more training data.
To further validate the capability of our approach across different manipulation scenarios, we evaluate STAR on the MetaWorld MT50 benchmark. As demonstrated in Fig. 3, the strong performance extends to the MetaWorld MT50 benchmark, where STAR achieves 92.7% average success rate across all 50 tasks, outperforming existing methods with a margin of 2.1%-5.4%. The consistent improvements across both manipulation benchmarks demonstrate the effectiveness of STAR as a general framework for learning diverse robot skills.
this section cite: []

Section: Analysis of Learned Skill Diversity
To evaluate the effectiveness of STAR in preventing codebook collapse, we conduct quantitative analysis of the learned skill representations. Fig. 4   utilization with all 16 codes being actively engaged in skill representation, while naive VQ-VAE exhibits severe collapse, utilizing only 43.8% of its codebook capacity (7 out of 16 codes). This comprehensive utilization indicates that RaRSQ successfully maintains diverse skill abstractions throughout the learning process. Second, beyond mere utilization, RaRSQ demonstrates significantly more balanced skill distribution across its codebook. The mean utilization frequency per code is 6.25%, approaching the theoretical optimal uniform distribution. In contrast, VQ-VAE shows a highly skewed distribution with 14.29% mean utilization per active code, indicating overreliance on a limited subset of representations. RaRSQ crucially maintains this healthy variation across its entire codebook rather than concentrating it in a small subset of active codes.
this section cite: []

Section: Ablation Study
To evaluate the contribution of each key component in STAR, we conduct ablation studies by removing critical modules. The results are shown in
Table 2. We compare the following variants: (1) w/o AR: Removes the autoregressive prediction in CST, directly predicting all skill codes independently; (2) w/o Rotation: Removes the rotation-augmented gradient in RaRSQ, using standard straight-through estimation; (3) w/o Rotation and AR: Removes both components.
The results demonstrate that both components are essential for strong performance. First, removing the autoregressive prediction (w/o AR) significantly impacts the ability to capture skill dependencies, leading to a substantial performance drop (89.5% vs 93.6% overall). This degradation is particularly severe on tasks requiring precise skill sequencing, such as LIBERO-Goal (-6.9%) and LIBERO-Long (-5.2%). Without autoregressive prediction, the model struggles to maintain temporal coherence between selected skills, resulting in fragmented or inconsistent behavior sequences.
The rotation-augmented gradient also proves crucial for effective skill learning. Removing this component (w/o Rotation) leads to degraded performance (91.0% overall), with the impact most pronounced on LIBERO-Object (-4.6%). This aligns with our theoretical analysis -without rotation-augmented gradients, the model suffers from codebook collapse and fails to maintain diverse skill representations. The performance decline is especially noticeable in tasks requiring varied manipulation skills, where having a rich repertoire of distinct skills is essential.
When both components are removed (w/o Rotation and AR), we observe the most severe performance degradation (87.8% overall). This synergistic effect demonstrates how our two-stage approach -first learning diverse skills through rotation-augmented quantization, then capturing their causal relationships through autoregressive prediction -is crucial for effective skill learning and composition. The substantial performance gap (5.8% lower than STAR) validates our design principle of combining geometric structure preservation for skill diversity with explicit temporal dependency modeling for skill composition.  Method Task Completion Overall Cube→Plate Toy→Box Success VQ-BET 5/10 3/10 3/10 QueST 6/10 4/10 4/10 Ours 8/10 6/10 6/10
Table 4. Performance on sequential manipulation task (sequential object placement). Results show successful trials out of 10 attempts for each subtask and overall completion.
this section cite: []

Section: STAR on Real-World Robots
To validate the effectiveness of STAR beyond simulation, we evaluate our approach on two challenging real-world manipulation tasks: a sequential object placement task ("Pick the cube into the plate and pick the toy into the box") and a structured drawer manipulation sequence ("Open the drawer, place the toy in the drawer and then close it"). These tasks mirror the complexity of LIBERO-Long tasks while introducing real-world challenges like lighting variations and physical dynamics.
For the drawer manipulation task, as shown in
Table 3, the results highlight a key advantage of the hierarchical skill decomposition. While all methods can initiate basic actions like drawer opening (60% success rate for STAR), performance degrades through complex sequences. STAR maintains higher success rates across stages, achieving 30% complete task success compared to 10% for VQ-BeT and 0% for QueST. The performance degradation from opening (60%) to complete execution (30%) reveals the compounding difficulty of maintaining precise control through extended sequences, though the degradation of STAR is notably less severe than the baselines.
As show in Table 4, the sequential object placement results further demonstrate the effectiveness of STAR in handling varied manipulation skills. STAR achieves 60% success rate for complete task execution, significantly outperforming VQ-BeT and QueST. The performance difference between initial cube placement and the more challenging toy placement aligns with task complexity, as the second placement requires more precise control given the confined space of the box. These results, visualized in Fig. 5, demonstrate that the improvements of STAR in skill learning and composition are effective in real-world scenarios.
this section cite: []

Section: Conclusion
In this paper, we presented STAR, a framework for learning and composing diverse robot skills through rotationaugmented vector quantization. Through RaRSQ and CST, our approach effectively prevents codebook collapse while enabling precise skill composition. Extensive experiments demonstrate the strong performance of STAR across manip-ulation benchmarks, significantly outperforming state-ofthe-art methods in both simulation and real-world settings.
Limitations. While STAR demonstrates strong performance across benchmarks, our approach requires predefined codebook sizes and quantization depths, which must be manually tuned for different task domains. Further, as an imitation learning approach, STAR depends on the quality of available expert demonstrations, which may limit its applicability when such data is scarce.
• For the drawer manipulation task, we measured success rates across three sequential stages: drawer opening, toy placement, and drawer closing.
A trial was considered successful only if all stages of the task were completed in the correct sequence. This evaluation protocol allowed us to identify potential bottlenecks in the manipulation sequence and assess the overall robustness of the learned policies.
this section cite: []

Section: References
Ref_id:b0 Title: Hierarchical residual learning based vector quantized variational autoencorder for image reconstruction and generation Year: (2022)
Ref_id:b1 Title: Visual affordance prediction for guiding robot exploration Year: (2023)
Ref_id:b2 Title: Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking Year: (2024)
Ref_id:b3 Title:  Year: (2023)
Ref_id:b4 Title: Image-goal representations are the atomic control units for foundation models in embodied ai Year: (2024)
Ref_id:b5 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b6 Title: Restructuring vector quantization with the rotation trick Year: (2024)
Ref_id:b7 Title: Language-guided skill learning with temporal variational inference Year: (2024)
Ref_id:b8 Title: Learning interpretable skill abstractions from language Year: (2022)
Ref_id:b9 Title: Efficient planning in a compact latent action space Year: (2022)
Ref_id:b10 Title: Humanoid control with a generalist planner Year: (2023)
Ref_id:b11 Title: Rethinking mutual information for language conditioned skill discovery on imitation learning Year: (2024)
Ref_id:b12 Title: An open-source vision-languageaction model Year: (2024)
Ref_id:b13 Title: Autoregressive image generation using residual quantization Year: (2022)
Ref_id:b14 Title: Behavior generation with latent actions Year: (2024)
Ref_id:b15 Title: End-to-end training of deep visuomotor policies Year: (2016)
Ref_id:b16 Title: Optimus-1: Hybrid multimodal memory empowered agents excel in long-horizon tasks Year: ()
Ref_id:b17 Title: Optimus-2: Multimodal minecraft agent with goalobservation-action conditioned policy Year: (2025)
Ref_id:b18 Title: Benchmarking knowledge transfer for lifelong robot learning Year: (2024)
Ref_id:b19 Title: Action-quantized offline reinforcement learning for robotic skill learning Year: (2023)
Ref_id:b20 Title: Robomp2: a robotic multimodal perception-planning framework with multimodal large language models Year: (2024)
Ref_id:b21 Title: Spatial-temporal graph diffusion policy with kinematic modeling for bimanual robotic manipulation Year: (2025)
Ref_id:b22 Title: What matters in learning from offline human demonstrations for robot manipulation Year: (2021)
Ref_id:b23 Title: Finite scalar quantization: Vq-vae made simple Year: (2023)
Ref_id:b24 Title: Quest: Self-supervised skill abstractions for learning continuous control Year: (2024)
Ref_id:b25 Title: Efficient action tokenization for vision-language-action models Year: (2025)
Ref_id:b26 Title: Theory and experiments on vector quantized autoencoders Year: (2018)
Ref_id:b27 Title: Behavior transformers: Cloning k modes with one stone Year: (2022)
Ref_id:b28 Title: Dynamics-aware unsupervised discovery of skills Year: (2019)
Ref_id:b29 Title: Self-supervised multi-task pretraining with control transformers Year: (2023)
Ref_id:b30 Title: An open-source generalist robot policy Year: (2024)
Ref_id:b31 Title: Neural discrete representation learning Year: (2017)
Ref_id:b32 Title:  Year: ()
Ref_id:b33 Title: Open x-embodiment: Robotic learning datasets and RT-x models Year: (2023)
Ref_id:b34 Title: Unleashing large-scale video generative pre-training for visual robot manipulation Year: (2023)
Ref_id:b35 Title: Discrete policy: Learning disentangled action space for multi-task robotic manipulation Year: (2024)
Ref_id:b36 Title: Vq-ace: Efficient policy search for dexterous robotic manipulation via action chunking embedding Year: (2024)
Ref_id:b37 Title: Latent action pretraining from videos Year: (2024)
Ref_id:b38 Title: d diffusion policy Year: (2024)
Ref_id:b39 Title: Soundstream: An end-to-end neural audio codec Year: (2021)
Ref_id:b40 Title: Learning fine-grained bimanual manipulation with low-cost hardware Year: (2023)
Ref_id:b41 Title: Learning temporal action abstractions as a sequence compression problem Year: (2024)
Ref_id:b42 Title: Reinforcement and imitation learning for diverse visuomotor skills Year: (2018)
