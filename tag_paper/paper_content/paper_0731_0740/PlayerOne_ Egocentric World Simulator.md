Title: PlayerOne: Egocentric World Simulator
Abstract: Figure 1: Simulated videos of our PlayerOne. Given an egocentric image as the scene to be explored, we can simulate egocentric immersive videos that are accurately aligned with the user's motion sequence captured by an exocentric camera. All the users have been anonymized and action videos are shot with the front camera.

Section: Introduction
World models [36,28,35,13,1,34] have undergone extensive research due to their ability to model environmental dynamics and predict long-term outcomes. Recent breakthroughs in video diffusion models [30,12,17] have revolutionized this domain, enabling the synthesis of high-fidelity, action-conditioned simulations that forecast intricate future states. These advancements empower applications ranging from autonomous navigation in dynamic real-world environments to the creation of immersive, responsive virtual worlds in AAA game development. By bridging the gap between predictive modeling and interactive realism, world simulators are emerging as critical infrastructure for next-generation autonomous systems and game engines, particularly in scenarios requiring real-time adaptation to complex, evolving interactions.
Despite significant progress, this topic remains underexplored in existing research. Prior studies [9,36,7] predominantly focused on simulations within game-like environments, falling short of replicating realistic scenarios. Additionally, in their simulated environments, users are limited to performing predetermined actions (i.e., directional movements). Operating within the confines of a constructed world restricts the execution of unrestricted movements as in real-world scenarios. While some initial efforts [19,26,1] have been made toward real-world simulation, they mainly contribute to world-consistent generation without human movement control. Consequently, users are reduced to passive spectators within the environment, rather than being active participants. This limitation significantly impacts the user experience, as it prevents the establishment of a genuine connection between the user and the simulated environment.
Faced with these challenges, we aim to design an egocentric world foundational framework that enables the user being a freeform adventurer. Given a user-provided egocentric image as the world to be explored, it can enable the user to perform unrestricted human movements real-time captured by an exocentric camera and consistent 4D scene modeling in the simulated world. Specifically, we propose the first realistic egocentric world simulator termed PlayerOne. Starting from a diffusion transformer (DiT) model [23], we first extract the latent of an egocentric user-input image. Meanwhile, we select the real-world human motions (i.e., human pose or keypoints) as our motion representation. Considering the varying importance of different body parts in our task, the human motion sequence is partitioned into three groups (i.e., head, hands, feet and body) and fed into our part-disentangled motion injection to generate latents that can enable precise part-wise control. Additionally, we developed a joint scene-frame reconstruction framework that can progressively complete scene point maps during the video generation process to enable scene-consistent generation. The DiT model takes the concatenation of the first frame latent, motion latent, video latent, and the point map latent as input and conducts noising and denoising on both the video and point map latent. Notably, the point map sequence is not required during inference, ensuring practical efficiency. Moreover, to overcome the absence of publicly available datasets, we curate required motion-video pairs from existing egocentric-exocentric datasets using an automated pipeline designed to filter and retain high-quality data. A coarse-to-fine training strategy is also designed to compensate for the data scarcity. The base model is fine-tuned on large-scale egocentric text-video data for coarse-level generation, then refined on our curated dataset to achieve precise motion control and scene modeling. Finally, we distill our trained model [38] to achieve real-time generation. By integrating these innovations, PlayerOne advances the field of dynamic world modeling. Our contributions are summarized as follows:
enable text-to-video generation to compensate for the limited availability of high-quality video-text datasets. Subsequently, diffusion transformers based frameworks [37,23,17,12,30,29] are proposed. When scaling-up training, they enable more highly realistic and temporally coherent generation results. Among them, HunyuanVideo [17] substitutes T5 with a Multimodal Large Language Model. LTX-Video [12] modifies the VAE decoder to handle the final denoising step and convert latents into pixels. Wan [30] introduces a full spatial-temporal attention to ensure computational efficiency.
World models. Existing world models [13,35,36,7,1,34] can be roughly divided into two categories: 1) Agent learning targeted models, 2) World simulation models. For the former, they [13,35,34] aim at enhancing policy learning within simulated environments. Among them, Dreamer [13] and DayDreamer [35] solve long-horizon tasks from images purely by latent imagination. MuZero [34] runs the self-play of Monte Carlo tree search to build world models for Atari. Distinct from this direction, world simulation aims to model an environment by predicting the next state given the current state and action. These works focus on human interaction with neural networks through high-quality rendering, robust control, and strong domain generalization to real-world scenarios. With advances in video generation, high-quality world simulation with robust control has become feasible, leading to numerous works focusing on interactive world simulation [9,26,19,36,7,1,28,6]. Among these works, WORLDMEM [36]. The Matrix [7] proposes the first world simulator capable of generating infinitely long real-scene video streams with real-time, responsive control. Matrix-Game [43] redefines video generation as an interactive process of exploration and creation. Cosmos [1] presents a general-purpose world model and a pre-training-then-post-training scheme. Aether [28] designs a unified framework with synergistic knowledge sharing across reconstruction, prediction, and planning objectives. However, these methods primarily focus on virtual game scenarios and are limited to specific directional actions, rather than facilitating high-degree-of-freedom motion control in real-world environments. To address these limitations, we target at developing a human motion driven realistic world simulator. Given an egocentric image, we can construct a real-scene world that immerses users as freeform adventurers with precise and unrestricted human motion control.
this section cite: ['b35', 'b27', 'b34', 'b12', 'b0', 'b33', 'b29', 'b11', 'b16', 'b8', 'b35', 'b6', 'b18', 'b25', 'b0', 'b22', 'b37', 'b36', 'b22', 'b16', 'b11', 'b29', 'b28', 'b16', 'b11', 'b29', 'b12', 'b34', 'b35', 'b6', 'b0', 'b33', 'b12', 'b34', 'b33', 'b12', 'b34', 'b33', 'b8', 'b25', 'b18', 'b35', 'b6', 'b0', 'b27', 'b5', 'b35', 'b6', 'b42', 'b0', 'b27']

Section: Method
In this section, we detail the methodology of PlayerOne. Sec. 3.1 introduces the relevant preliminaries and the overall pipeline. Sec. 3.2 presents the core of our proposed model, followed by Sec. 3.3 introducing our dataset construction and training strategy.
this section cite: []

Section: Overview
Video diffusion models [23,15] consist of two key processes: a forward (noising) process and a reverse (denoising) process. The forward process gradually adds Gaussian noise, denoted as ϵ ∼ N (0, I), to a clean latent sample z 0 ∈ R k×c×h×w , where k, c, h, and w represent the dimensions of the video latents. This transforms z 0 into a noisy latent z t . In the reverse process, a learned denoising model ϵ θ progressively removes the noise from z t to reconstruct the original latent representation.
As shown in Fig. 2, our method comprises two core modules: Part-disentangled Motion Injection (PMI) and Scene-frame Reconstruction (SR). In PMI, we use real-scene human motion as the motion condition to enable free action control for the user. The first frame is converted into z f rame via a 3D VAE encoder. The human motion sequence is split into three parts based on varying importance, and each part is fed into a 3D motion encoder to obtain latents. These motion latents are concatenated into z motion ∈ R k×3×h×w . To improve view alignment, we transform the head parameters of the human motion sequence into a camera sequence, which is then fed into a camera encoder. The output is added to the noised video latents z video to inject view-change signals. In SR, we jointly reconstruct video frames and 4D scenes to ensure world-consistent generation in the context of long video generation. We render a point map sequence from the ground truth video and feed it into a point map encoder with an adapter to obtain z point ∈ R k×64×h×w . Finally, all latents and conditions are concatenated channel-wise. The training objective of our method can be expressed as:
L = E t∼U (0,1),ϵ∼N (0,I) ∥ϵ -ϵ θ (z t , t)∥ 2 2 , where z t = α t z 0 + δ t ϵ(1)
Where t = 1, . . . , T , α t 2 + δ t 2 = 1. Since we only add noise to point map latents and video latents, thus z 0 = z video ⊗ z point . ⊗ denotes the channel-wise concatenation operation, U(•) represents a uniform distribution, and T denotes the denoising steps.
this section cite: ['b22', 'b14']

Section: 3D VAE Encoder

this section cite: []

Section: Human Motion Sequence
First Frame
this section cite: []

Section: Noised Video Latent

this section cite: []

Section: ❄

this section cite: []

Section: Motion Latent
Noised Point Map Latent
this section cite: []

Section: ❄ 🔥 Trainable

this section cite: []

Section: Frozen

this section cite: []

Section: Part-disentangled Motion Injection 🔥

this section cite: []

Section: First Frame Latent

this section cite: []

Section: Diffusion Transformer

this section cite: []

Section: 🔥

this section cite: []

Section: Scene-frame Reconstruction 🔥

this section cite: []

Section: Denoised Latent
Video/Point Map Decoder
this section cite: []

Section: ❄

this section cite: []

Section: Ground Truth

this section cite: []

Section: Point Map Encoder

this section cite: []

Section: Point Map Sequence
Noised Point Map Latent Head Hands Body&Feet
this section cite: []

Section: Decompose

this section cite: []

Section: Camera Sequence

this section cite: []

Section: Camera
Encoder 🔥
this section cite: []

Section: Motion
Encoder 🔥
this section cite: []

Section: Noised Video Latent

this section cite: []

Section: ✚

this section cite: []

Section: Motion Latent

this section cite: []

Section: Part-disentangled Motion Injection Scene-frame Reconstruction
Part-wise tokens. The human motion sequence is split into groups and fed into the motion encoders respectively to generate part-wise motion latents, with the head parameters converted into a rotation-only camera sequence. This camera sequence is then encoded via a camera encoder, and its output is injected into noised video latents to improve view-change alignment. Next, we render a 4D scene point map sequence with the ground truth video, which is then processed by a point map encoder with an adapter to produce scene latents. Then we input the concatenation of these latents into the DiT Model and perform noising and denoising on both the video and scene latents to ensure world-consistent generation. Finally, the denoised latents are decoded by VAE decoders to produce the final results. Note that only the first frame and the human motion sequence are needed for inference.
this section cite: []

Section: Model Components
Part-disentangled motion injection. Prior studies [19,26,7,43] typically utilize camera trajectories as motion conditions or are constrained to specific directional movements. These restrictions confine users to passive "observer" roles, preventing meaningful user interaction. In contrast, our approach empowers users to become active "participants" by adopting real-world human motion sequences (i.e., human pose or keypoints) as motion conditions, allowing for more natural and unrestricted movement. However, our empirical analysis reveals that extracting latent representations holistically from human motion parameters complicates precise motion alignment. To address this challenge, we introduce a part-disentangled motion injection strategy that recognizes the distinct roles of various body parts. Specifically, hand movements are essential for interacting with objects in the environment, while the head plays a crucial role in maintaining egocentric perspective alignment. Accordingly, we categorize the human motion parameters into three groups: body and feet, hands, and head. Each group is processed through its own dedicated motion encoder, comprising eight layers of 3D convolutional networks, to extract the relevant latent features. This specialized processing ensures accurate and synchronized motion alignment. These latents are subsequently concatenated along the channel dimension to form the final part-aware motion latent representation z motion ∈ R k×3×h×w .
To further enhance the egocentric view alignment, we solely transform the head parameters of the human motion sequence into a sequence of camera extrinsics with only rotation values. We zero out the translation values in the camera extrinsics, assuming the head parameters are at the camera coordinate system's origin. Specifically, suppose the head parameter v = (θ x , θ y , θ z ), we first normalize the rotation axis as follows:
u = v ∥v∥ , θ = ∥v∥(2)
Then we construct the rotation matrix as follows:
R = I + sin θ • [u] × + (1 -cos θ) • [u] 2 × (3
)
Exocentric Video Egocentric Video Video Pool Detection Model Low Confidence ❌ SMPL Model Automatic Data Filtering 3D Mesh Camera Sequence SMPL-X Exocentric Video OpenPose 2D Keypoints 2D Keypoints Projection Keypoint Distance ❌ ✓ Small Large SMPL Sequence
Figure 3: The overall pipeline of the dataset construction. By seamlessly integrating detection and human pose estimation models, we can extract motion-video pairs from existing egocentric-exocentric video datasets while retaining high-quality data through our automatic filtering scheme.
Where u × is the cross product matrix of u, which can be denoted as follows:
[u] × = 0 -u z u y u z 0 -u x -u y u x 0 (4
)
Then we use Plücker ray [40] to parameterize the camera extrinsics and then feed the output to an extra camera encoder, which shares a similar structure with the motion encoder. Then the latents from this encoder are added to the noised video latents to inject the view-change information.
Scene-frame reconstruction. While PMI enables precise control over egocentric perspective and motion, it does not guarantee scene consistency within the generated world. To address this limitation, we introduce a joint reconstruction framework that simultaneously models the 4D scene and video frames, ensuring scene coherence and continuity throughout the video. Specifically, it begins by employing CUT3R [31] to generate a point map for each frame based on ground truth video data, reconstructing the n-th frame's point map using information from frames 1 through n. These point maps are then compressed into latent representations using a specialized point map encoder [16].
To integrate these latents with video features, we implement an adapter composed of five 3D convolutional layers. This adapter aligns the point map latents with video latents and projects them into a shared latent space, facilitating seamless integration of motion and environmental data. Finally, we concatenate the latent representations from the first frame, the human motion sequence, the noised video latents, and corresponding noised point map latents. This comprehensive input is then fed into a diffusion transformer for denoising, resulting in a coherent and visually consistent world. Importantly, point maps are only required during the training phase. During inference, the system simplifies the process by utilizing only the first frame and the corresponding human motion sequence to generate world-consistent videos. This streamlined approach enhances generation efficiency while ensuring that the resulting environment remains stable and realistic throughout the entire video.
this section cite: ['b18', 'b25', 'b6', 'b42', 'b39', 'b30', 'b15']

Section: Training Strategy

this section cite: []

Section: Dataset preparation.
The ideal training samples for our task are egocentric videos paired with corresponding motion sequences. However, no such dataset currently exists in publicly available repositories. As a substitute, we derive these data pairs from existing egocentric-exocentric video datasets through an automatic pipeline. Specifically, for each synchronized egocentric-exocentric video pair, we first employ SAM2 [25] to detect the largest person in the exocentric view. The background-removed exocentric video is then processed using SMPLest-X [39] to extract the SMPL parameters of the identified individual as the human motion. To enhance optimization stability, an L2 regularization prior is incorporated. We then evaluate the 2D reprojection consistency to filter out low-quality SMPL data. This involves generating a 3D mesh from the SMPL parameters using SMPLX [22], projecting the 3D joints onto the 2D image plane with the corresponding camera parameters, and extracting 2D key points via OpenPose [4]. The reprojection error is calculated by measuring the distance between the SMPL-projected 2D key points and those detected by OpenPose. Data pairs with reprojection errors in the top 10% are excluded, ensuring a final dataset of high-quality motion-video pairs. The refined SMPL parameters are decomposed into body and feet (66 dimensions), head orientation (3 dimensions), and hand articulation (45 dimensions per hand) components for each frame. These components are fed into their respective motion encoders. The dataset construction pipeline is illustrated in Fig. 3. As detailed in Tab. 1, our training dataset combines multiple publicly available datasets to ensure comprehensive coverage of diverse environmental contexts, action types, and intensity levels, thereby enhancing model generalization.  Coarse-to-fine training. Though we can extract high-quality motion-video training data with our automatic pipeline, the limited scale of this dataset is insufficient for training video generation models to produce high-quality egocentric videos. To address this, we harness the extensive egocentric text-video datasets (i.e., Egovid-5M [33]). Specifically, we first fine-tune the baseline model using LoRA on large-scale egocentric text-video data pairs, enabling egocentric video generation with coarse-level motion alignment. Then we freeze the trained LoRA and fine-tune the last six blocks of the model with our constructed high-quality dataset to enhance fine-grained human motion alignment and view-invariant scene modeling, which can effectively address the scarcity of pair-wise data.
Finally, we adopt an asymmetric distillation strategy that supervises a causal student model with a bidirectional teacher [38] to achieve real-time generation and long-duration video synthesis.
this section cite: ['b24', 'b38', 'b21', 'b3', 'b32', 'b37']

Section: Experiments

this section cite: []

Section: Experimental Setting
Implementation details. We choose Wanx2.1 1.3B [30] as the base generator. We set the LoRA rank and the update weight of the matrices as 128 and 4 respectively and initialize its weight following [30]. The inference step and the learning rate are set as 50 and 1 × 10 -5 respectively, where the Adam optimizer and mixed-precision bf16 are adopted. The cfg of 7.5 is used. We train our model for 100,000 steps on 8 NVIDIA A100 GPUs with a batch size of 56 and sample resolution of 480×480. The generated video runs at eight frames per second, and we utilize 49 video frames (6 seconds) for training. After distillation, our method can achieve 8 FPS to generate the desired results. All the action videos in this paper are shot with the front camera. Benchmark. Since there is no publicly available benchmark for our task, we construct a benchmark with 100 videos collected from Nymeria [21] dataset, which is not included for training. It consists of coarse-level motion descriptions for each sample and covers diverse realistic scenarios. Considering the information gap between the human motion sequence and the text, we further use Qwen2.5-VL [2] to enrich the caption to generate videos for the competitors for more fair comparisons. Metrics. On our constructed benchmark, for evaluation of alignment with the given text descriptions, we calculate both CLIP-Score and DINO-Score, where LPIPS [42] is employed to evaluate the video fidelity of the generated video. Besides, we calculate the frame consistency to evaluate the temporal coherence and consistency of the generated video frames over time. We further utilize a 3D hand pose estimation model [24] to estimate the hand pose of the generated videos and use the results of  the ground truth video as the labels. Afterward, we follow [24] to calculate two metrics: (1) Mean Per-Joint Position Error (MPJPE): the L2 distance between the predicted and ground truth joints for each hand after subtracting the root joint. (2) Mean Relative-Root Position Error (MRRPE): the metric distance between the root joints of the left hand and right hand.
this section cite: ['b29', 'b29', 'b20', 'b1', 'b41', 'b23', 'b23']

Section: Ablation Study Investigation on coarse-to-fine training.
We first evaluate several variants of our coarse-to-fine training scheme, as depicted in Fig. 4. Specifically, when inputting action descriptions into the baseline model without fine-tuning, the generated results exhibit noticeable flaws, such as hand distortions or the unexpected appearance of individuals. Similar issues can be observed when training with only motion-video pairs. We also explore jointly training with both large-scale egocentric videos and motion-video pairs. Specifically, when inputting egocentric videos, we set the motion latent values to zero and extract the latents of the text description to serve as the motion condition, where a balanced-sampling strategy is used as well. Despite this variant being capable of generating egocentric videos, it fails to produce results accurately aligned with the given human motion conditions. In contrast, our coarse-to-fine training scheme delivers much better outcomes compared to these variants.
Investigation on part-disentangled motion injection. Next, we conduct a detailed analysis of our PMI module. Specifically, three variants are included: ControlNet-based [41] motion injection, inputting motion sequences as a unified entity (the "Entangled" scheme), and removing our camera encoder. As shown in Fig. 5, the ControlNet-based scheme suffers from information loss, preventing it from producing results that accurately align with the specified motion conditions. Similarly, the entangled scheme demonstrates comparable shortcomings. Furthermore, removing the camera encoder leads to the model's inability to generate view-accurate alignments. As depicted in Fig. 5, this variant fails to produce the corresponding perspective change associated with crouching. Ultimately, our PMI module successfully generates outcomes that are both view-aligned and action-aligned. Investigation on scene-frame reconstruction. Additionally, we conducted a detailed analysis of the SR module, exploring three variants: omitting reconstruction, removing the adapter within the SR
this section cite: ['b40']

Section: Ours Action
No Recon
this section cite: []

Section: DUSt3R No Adapter
First Frame First Frame Action Action First Frame Result Result Result First Frame First Frame First Frame First Frame First Frame Figure 7: Qualitative evaluation on the motion alignment. We generate simulated videos based on the same first frame but different motion sequences. Results show that we can achieve accurate motion alignment.
module, and substituting CUT3R [31] with DUStR [32] for point map rendering. As illustrated in Fig. 6, the absence of reconstruction results in the model's inability to generate consistently simulated results. Moreover, due to the distribution gap between latents of frames and point maps, training without the adapter leads to difficulty in loss convergence, causing noticeable distortions. Furthermore, after replacing CUT3R [31] with DUStR [32], our PlayerOne can also produce scene-consistent outputs, demonstrating its robustness to different point map rendering techniques. Motion alignment. To verify the alignment capability with the given motion condition, we conduct experiments by generating world-simulated videos with the same first frame but different human motion sequences. Fig. 7 shows that our PlayerOne can accurately generate corresponding results according to different conditions and produce reasonable interactive changes. Quantitative comparisons. We provide quantitative results on the core components of our PlayerOne in Tab. 2. All numerical results concur with the visualization outcomes. A significant performance improvement is observed when the model undergoes pre-training on large-scale egocentric text-video datasets. The introduction of PMI yields an additional accuracy boost, and it outperforms all of its variants. In addition, our designed filtering strategy maximizes performance as well by filtering noisy motion-video pairs. After removing the adapter, the performance suffers from a notable degradation due to the distribution gap between latents of video frames and point maps. By introducing our joint scene-frame reconstruction scheme, we achieve superior results across all metrics.
this section cite: ['b30', 'b31', 'b30', 'b31']

Section: Comparison with State-of-the-arts
Quantitative comparison. Since there is no method sharing the same setting as ours, we selected two potential competitors for comparison: Cosmos [1] and Aether [28]. As shown in Tab. 3, our   Table 4: User study on our PlayerOne and existing alternatives. "Quality", "Fidelity", "Smooth", and "Alignment" measure synthesis quality, object identity preservation, motion consistency, and alignment with the text descriptions, respectively. Each metric is rated from 1 (worst) to 4 (best).
Quality (↑) Fidelity (↑) Smooth (↑) Alignment (↑) Aether [28] 1.32 1.30 1.31 1.34 Cosmos(Diff-7B) [1] 2.07 2.13 2.05 2.09 Cosmos(Diff-14B) [1] 3.02 2.94 2.98 2.71 PlayerOne(ours) 3.59 3.63 3.65 3.86
PlayerOne outperforms all the baselines by large margins, especially on the metrics of motion alignment. Notably, Cosmos [1] exhibits better generalization ability than Aether [28] by explicitly capturing general knowledge of real-world physics and natural behaviors. Besides qualitative results, we provide visualization comparisons in Fig. 8, where consistent superiority can be observed in diverse scenarios for both user interaction and world modeling. User study. In Tab. 4, we report the comparison results of human preference rates. We let 20 annotators rate 25 groups of videos, where each group contains the generated video of each method and text description. And we provide detailed regulations to rate the results for scores of 1-4 from four views: "Quality", "Smooth", "Fidelity", "Alignment". "Quality" counts for whether the result is harmonized without considering fidelity. "Smooth" assesses the motion consistency across the video. "Fidelity" measures ID preservation and distortions within the video, while we use "Alignment" to measure the alignment with the given text descriptions. It can be noted that our model demonstrates significant superiority across all the metrics, especially for "Alignment", and "Smooth".
this section cite: ['b0', 'b27', 'b0', 'b27']

Section: Conclusion
In conclusion, PlayerOne represents a significant advancement in interactive and realistic world modeling for video generation. Unlike conventional models that are restricted to particular game scenarios or actions, our PlayerOne can capture the complex dynamics of general-world environments and enable free motion control within the simulated world. By formulating world modeling as a joint process of videos and 4D scenes, our PlayerOne ensures coherent world generation and enhances motion and view alignment with the given conditions through part-disentangled motion injection.
Experimental results demonstrate our superior performance across diverse scenarios.
Limitations. Despite the compelling outcomes, our performance in game scenarios is slightly inferior to realistic ones, likely due to the imbalanced distribution between realistic and game training data. It can be addressed by incorporating more game-scenario datasets in future research.
this section cite: []

Section: References
Ref_id:b0 Title: Cosmos world foundation model platform for physical ai Year: (2009)
Ref_id:b1 Title:  Year: (2023)
Ref_id:b2 Title: Retrieval-augmented diffusion models Year: (2022)
Ref_id:b3 Title: Openpose: Realtime multi-person 2d pose estimation using part affinity fields Year: (2019)
Ref_id:b4 Title: Control-a-video: Controllable text-to-video generation with diffusion models Year: (2023)
Ref_id:b5 Title: Animegamer: Infinite anime life simulation with next game state prediction Year: (2025)
Ref_id:b6 Title: The matrix: Infinite-horizon world generation with real-time moving control Year: (2024)
Ref_id:b7 Title: Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives Year: (2024)
Ref_id:b8 Title: Mineworld: a real-time and open-source interactive world model on minecraft Year: (2025)
Ref_id:b9 Title: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning Year: (2024)
Ref_id:b10 Title: Ft-hid: A large scale rgb-d dataset for first and third person human interaction analysis Year: (2022)
Ref_id:b11 Title: Ltx-video: Realtime video latent diffusion Year: (2024)
Ref_id:b12 Title: Dream to control: Learning behaviors by latent imagination Year: (2019)
Ref_id:b13 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b14 Title: Video diffusion models Year: ()
Ref_id:b15 Title: Geo4d: Leveraging video generators for geometric 4d scene reconstruction Year: (2025)
Ref_id:b16 Title: A systematic framework for large video generative models Year: (2024)
Ref_id:b17 Title: Egoexo-fitness: towards egocentric and exocentric full-body action understanding Year: (2024)
Ref_id:b18 Title: Navigating 3d scenes from a single image Year: (2024)
Ref_id:b19 Title: Nymeria: A massive collection of multimodal egocentric daily motion in the wild Year: (2024)
Ref_id:b20 Title: Nymeria: A massive collection of multimodal egocentric daily motion in the wild Year: (2024)
Ref_id:b21 Title: Expressive body capture: 3d hands, face, and body from a single image Year: (2019)
Ref_id:b22 Title: Scalable diffusion models with transformers Year: (2022)
Ref_id:b23 Title: 3d hand pose estimation in everyday egocentric images Year: (2024)
Ref_id:b24 Title: Sam 2: Segment anything in images and videos Year: (2025)
Ref_id:b25 Title: d-informed worldconsistent video generation with precise camera control Year: (2025)
Ref_id:b26 Title: High-resolution image synthesis with latent diffusion models Year: (2021)
Ref_id:b27 Title: Geometric-aware unified world modeling Year: (2009)
Ref_id:b28 Title:  Year: (2024)
Ref_id:b29 Title: Wan: Open and advanced large-scale video generative models Year: (2006)
Ref_id:b30 Title: Continuous 3d perception model with persistent state Year: (2025)
Ref_id:b31 Title: Dust3r: Geometric 3d vision made easy Year: (2024)
Ref_id:b32 Title: Egovid-5m: A large-scale video-action dataset for egocentric video generation Year: (2024)
Ref_id:b33 Title: Muzero general: Open reimplementation of muzero Year: (2019)
Ref_id:b34 Title: Daydreamer: World models for physical robot learning Year: (2022)
Ref_id:b35 Title: Long-term consistent world simulation with memory Year: (2025)
Ref_id:b36 Title: Efficient high-resolution image synthesis with linear diffusion transformer Year: (2024)
Ref_id:b37 Title: From slow bidirectional to fast autoregressive video diffusion models Year: (2025)
Ref_id:b38 Title: Smplest-x: Ultimate scaling for expressive human pose and shape estimation Year: (2025)
Ref_id:b39 Title: Cameras as rays: Pose estimation via ray diffusion Year: (2024)
Ref_id:b40 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b41 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b42 Title: Matrix-game: Interactive world foundation model Year: (2025)
