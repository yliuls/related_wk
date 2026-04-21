Title: MOSPA: Human Motion Generation Driven by Spatial Audio
Abstract: Figure 1: We introduce a novel human motion generation task centered on spatial audio-driven human motion synthesis. Top row: We curate a novel Spatial Audio-Driven Human Motion (SAM) dataset, including diverse spatial audio signals and high-quality 3D human motion pairs. Bottom row: We develop a generative framework for human MOtion generation driven by SPatial Audio (MOSPA) to produce high-quality, responsive human motion driven by spatial audio. We note that the motion generation results are both realistic and responsive, effectively capturing both the spatial and semantic features of spatial audio inputs.

Section: Introduction
Humans exhibit varying responses to different auditory inputs within a given space. For instance, when exposed to sharp, piercing sounds, individuals are likely to cover their ears and move away in the direction opposite to the sound source. Conversely, when the sound is soft and soothing, they may approach it out of curiosity or to investigate further. Therefore, generating realistic human motion for virtual characters to respond realistically to a variety of sounds in their environment is both a highly sought-after feature and is crucial for applications such as virtual reality, human-computer interaction, robotics, etc.
Unfortunately, while previous studies have extensively explored motion generation from action label [85,29], text [93,100,77], music [78,91,46,1], and speech [2,97,3,102], human motion generation driven by spatial audio remains unexplored to the best of our knowledge. Unlike pure audio signals, e.g., music [70,50,78], speech [3,97], the spatial audio signals not only does it encode semantics, but it also captures spatial characteristics that significantly influence body movements, requiring a specialized framework to accurately model motion responses to spatial audio stimuli.
To address this overlooked aspect, we propose to model the complex interactions between spatial audio inputs and human motion using a generative model. Since there is no such dataset tailored for this task, we first introduce the SAM dataset (Spatial Audio Motion dataset), which captures diverse human responses to various spatial audio conditions. This dataset is meticulously curated to include a wide range of spatial audio scenarios, enabling the study of motion conditioned on sound field variations. The SAM dataset has a total of more than 9 hours of motion, covering 27 common spatial audio scenarios and more than 70 audio clips. To ensure the diversity of the spatial audio, around 480 seconds of motion were captured for each audio clip at different positions in the character space. To ensure diverse motion responses to spatial audio, we introduce 20 distinct motion types (excluding motion genres) and 49 in total when including motion genres. We visualize samples from SAM in the top row of Fig. 1. See Appendix A for detailed statistics.
We further conduct benchmarking experiments on the proposed dataset, revealing the limitations of existing methods in this setting. To enable spatial audio-driven human motion generation, we introduce MOSPA, a simple yet effective framework tailored for this task. In real-world scenarios, human responses to sound are inherently influenced by spatial perception, intensity variations, directional cues, temporal dynamics, etc. Motivated by this, we generate motion by incorporating features extracted from the input spatial audio signals using [61]. Specifically, to capture intrinsic features across both temporal and spatial dimensions, we mainly utilize Mel-Frequency Cepstral Coefficients (MFCCs) [17] and Tempograms [28] to model the temporal characteristics of the audio. Additionally, we characterize the spatial audio by analyzing the root mean square (RMS) [61] energy, which quantifies signal intensity in audio processing.
These features enhance the effective modeling of the spatial and intensity variations of the spatial audio. To capture the distribution of spatial audio features and human motion dynamics effectively, we employ a diffusion-based generative model that ensures strong alignment between the two modalities-human motion and spatial audio signals. Leveraging diffusion models, MOSPA excels at modeling the complex interplay between spatial audio features and human motion. Besides, a residual feature fusion mechanism is employed to model the subtle influences of spatial audio on human movement.
Extensive evaluations on the SAM demonstrate that MOSPA achieves state-of-the-art performance on this task, outperforming existing baselines in generating realistic and diverse motion responses to spatial audio. Our contributions are summarized as follows:
• We introduce a novel task of spatial audio-conditioned motion generation and present the first comprehensive dataset SAM with over 9 hours of motion across diverse scenarios.
• We conduct extensive benchmarking and propose MOSPA, a diffusion-based generative framework tailored for modeling and generating diverse human motions from spatial audio.
• We achieve the SOTA performance on motion generation conditioned on spatial audio. Our dataset, code, and models will be publicly released for further research.
2 Related Work Spatial Audio. Many studies have explored spatial audio modeling [98,24,99,74,37,87,44,45]. For instance, [98] utilizes the natural synchronization between visual and audio modalities to learn models that jointly parse sounds and images without manual annotations. [24] leverages unlabeled audiovisual data to localize objects, such as moving vehicles, using only stereo sound at inference time. [99] reason about spatial sounds with large language models. Recently, spatial audio generation has been explored from text [74] and video [42]. [87] propose a method to model 3D spatial audio from body motion and speech. [37] presents a framework for spatial audio generation, capable of rendering 3D soundfields generated by human actions, including speech, footsteps, and hand-body interactions. Despite progress on spatial-audio tasks, generating human motion from spatial audio remains largely underexplored.
this section cite: ['b84', 'b28', 'b92', 'b99', 'b76', 'b77', 'b90', 'b45', 'b0', 'b1', 'b96', 'b2', 'b101', 'b69', 'b49', 'b77', 'b2', 'b96', 'b60', 'b16', 'b27', 'b60', 'b97', 'b23', 'b98', 'b73', 'b36', 'b86', 'b43', 'b44', 'b97', 'b23', 'b98', 'b73', 'b41', 'b86', 'b36']

Section: Conditional Motion Generation.
Extensive efforts have been made into motion synthesis conditioning on user control signals [36,72,11,79,92], text [93,100,77,30,51,53,14], action [85,29,10,39], music [78,91,46,1], speech [2,68,38,97,3,102], past trajectories [23,9,4,90,73], etc. We refer readers to [103] for a detailed survey of motion generation. Text-to-Motion. Text-to-motion generation has recently gained popularity as an intuitive and userfriendly approach to synthesizing diverse human body motions. Generative pre-trained transformer frameworks have been utilized for text-to-motion generation [93,57]. Subsequently, various generation techniques have been explored, including diffusion models [77], latent diffusion models [12], autoregressive diffusion model [69,11], denoising diffusion GANs [100], consistency models [16], and generative masked modeling [31]. Recent advancements include the integration of motion generation with large language models [41,94] and investigations into the scaling laws for motion generation [58,22]. Recently, controllable text-to-motion generation has gained attention, enabling motion synthesis conditioned on both text prompts and control signals, e.g., target control points [84,79].
this section cite: ['b35', 'b71', 'b10', 'b78', 'b91', 'b92', 'b99', 'b76', 'b29', 'b50', 'b52', 'b13', 'b84', 'b28', 'b9', 'b38', 'b77', 'b90', 'b45', 'b0', 'b1', 'b67', 'b37', 'b96', 'b2', 'b101', 'b22', 'b8', 'b3', 'b89', 'b72', 'b102', 'b92', 'b56', 'b76', 'b11', 'b68', 'b10', 'b99', 'b15', 'b30', 'b40', 'b93', 'b57', 'b21', 'b83', 'b78']

Section: Music-to-Motion.
Recent advancements in Music-to-Motion generation have been made [21,70,48,78,27,88,71]. DanceFormer [47] adopts a two-stage approach, generating key poses for beat synchronization followed by parametric motion curves for smooth, rhythm-aligned movements. Bailando [70] utilizes a VQ-VAE to encode motion features via a choreographic memory module.
[1] introduces a diffusion-based probabilistic model for motion generation, using a Conformer-based architecture. EDGE [78] also applies a diffusion model for dance generation and editing. Furthermore, multimodal approaches incorporating language and music enhance generation quality [27,88,15]. Speech-to-Motion. We mainly review studies on audio-driven motion (gesture) generation [97,13,3,2,102]. Early works are mostly based on GAN models [26,54,65,89], while the recent attempts are mainly based on the generative diffusion model [102]. For instance, [97] proposes a generative retrieval framework leveraging a large language model to efficiently retrieve semantically appropriate gesture candidates from a motion library in response to input speech. [2] introduces a co-speech gesture synthesis method by employing a segmentation pipeline for temporal alignment and disentangling speech-motion embeddings to capture both semantics and subtle variations.
While audio signals have been widely used in music-and speech-to-motion tasks, human motion synthesis driven by spatial audio remains largely unexplored. As a result, data-driven methods are highly constrained by limited paired data. The goal of this paper is to develop a comprehensive dataset and a novel approach for high-quality spatial audio-driven motion synthesis.
this section cite: ['b20', 'b69', 'b47', 'b77', 'b26', 'b87', 'b70', 'b46', 'b69', 'b77', 'b26', 'b87', 'b14', 'b96', 'b12', 'b2', 'b1', 'b101', 'b25', 'b53', 'b64', 'b88', 'b101', 'b96', 'b1']

Section: SAM Dataset
We first introduce the Spatial Audio-driven Motion (SAM) dataset designed for human motion synthesis conditioned on spatial audio. We focus on binaural audio, a common form of spatial audio that aligns with human and (most) animal perception and can be readily applied to robotic platforms. SAM consists of more than 9 hours of human motions with corresponding binaural audio, and more than 4M frames, covering 27 common spatial audio scenarios and 20 common reaction types in daily life without counting the motion genres. The majority of the audio clips are sourced from the AudioSet [25], while only a small portion is extracted from publicly available YouTube videos   or through manual recording. More detailed information can be found in Tab. 1 and Appendix A. Visualization results are in Fig.   Data Capture Settings. We utilize a Vicon motion capture system [56] to collect motion data and spatial audio signals. The motion capture is performed in a semi-open cage having a space of approximately 5m × 10m × 3m, a structure covered by rope nets, with 28 mocap cameras mounted on the ropes and vertical supports recording at a frame rate of 120 Hz; See Fig. 3. The surrounding walls are standard painted concrete, resulting in a setting that resembles a typical indoor environment.
In SAM, each audio clip is associated with 16 randomly sampled relative sound source locations, defined by combinations of different speakers and spatial positions relative to the subject. For each location, we capture three motion sequences corresponding to different reaction intensities: dull, neutral, and sensitive, resulting in a total of 48 motion sequences, each lasting 10 seconds. Fig. 4 shows the statistics of the approximate duration of the actions. The three motion genres define the varying degrees of responsiveness, decreasing from sensitive to dull. For instance, upon hearing an explosion, a dull individual might remain largely unreactive, whereas a sensitive one may immediately flee from the sound source. The total number of action types is 49. The percentage of action types covered within the dull, neutral, and sensitive motion genres are 28.57%, 34.69%, and 36.73%, respectively. To capture the binaural sound heard at the position of the actor, we employ two microphones to record the audio at the ear positions of the actors separately; See the inset. The two microphones are connected to a Deity PR-2 recorder [18] that has been synchronized with the Vicon mocap system in advance using a timecode with a frame rate of 30 FPS. With this setting, the stereo sound at the position of the actor can be recorded and has an accurate alignment with the corresponding motion.
this section cite: ['b24', 'b55', 'b17']

Section: Data Processing.
All motion and audio clips are precisely aligned. The motions are re-targeted and converted from the original BVH format in Vicon to the SMPL-X [63] format. SMPL-X is a parametric 3D human body model that encompasses the body, hands, and face, comprising N = 10, 475 vertices and K = 55 joints. Given shape parameters β and pose parameters θ, the SMPL-X model generates the corresponding body shape and pose through forward dynamics. We extract the locations of the sound sources in each motion clip. The sound source locations are then transformed into the local space of the character aligned with the SMPL-X local coordinate system (a.k.a the local frame).
this section cite: ['b62']

Section: Method
We introduce MOSPA, a diffusion-based probabilistic model that serves as a baseline for this novel task of spatial audio-driven human motion generation. First, we extract spatial audio features a using a feature extractor [61]. During motion generation, the extracted spatial audio feature a is combined with the sound source location s and the motion genre g as conditioning inputs. These inputs are passed to a denoiser G, which is trained to reconstruct the original clean motion vector x0 by denoising the given noisy motion vector x t at time step t. Mathematically, we have x0 = G(x t , t; a, s, g).
this section cite: ['b60']

Section: Feature Representation
The two key vectors are the audio feature vector a and the motion vector p. We carefully designed the structure of the two vectors in MOSPA.
this section cite: []

Section: Spatial Audio Feature Extraction.
We first extract a range of audio features that capture intensity, temporal dynamics, and spatial characteristics. Inspired by [70], our feature set primarily includes Mel-frequency cepstral coefficients (MFCCs), MFCC delta, constant-Q chromagram, short-time Fourier transform (STFT) of the chromagram, onset strength, tempogram, and beats [17,67,28,61,7,20]. On top of these audio features, we additionally add the root mean square (RMS) energy E rms of the audio [61], and the active frames F active defined as F active = E rms > 0.01 to capture the distance information of the audio. The dimension of the audio feature vector for each ear is 1136. By concatenating the features from both ears, we obtain a combined feature vector a of dimension 2272. The detailed construction of the audio vector can be viewed in Appendix B.1. Motion Representation. In this paper, we focus on body motion and leave the modeling of detailed finger movements to future work. Therefore, we exclude all the finger joints and retain only the first J = 25 body joints of the SMPL-X model [63]. In addition to the essential translation and joint rotations required for human pose representation, we introduce the residual feature fusion mechanism [30] to incorporate the global joint positions and the velocity of the joints to capture the nuanced difference in audio and further improve the accuracy of the generated samples. Each motion vector x is thus composed of the global positions p ∈ R T ×(J×3) , the local rotations r ∈ R T ×(J×6) and the velocities v ∈ R T ×(J×3) of the joints (including the root), where T = 240 represents the number of frames in each motion sequence. The joint rotations are represented in the 6d format [101] to guarantee the continuity of the change (x 0 = (p 0 , r 0 , v 0 ), x ∈ R T ×(J×12) ). The dimension of each motion vector is therefore 300.
this section cite: ['b69', 'b16', 'b66', 'b27', 'b60', 'b6', 'b19', 'b60', 'b62', 'b29', 'b100']

Section: Framework
Figure 5: The framework of MOSPA. We perform diffusion-based motion generation given spatial audio inputs. Specifically, Gaussian noise is added to the clean motion sample x 0 , generating a noisy motion vector x t , modeled as q(x t |x t-1 ). An encoder transformer then predicts the clean motion from the noisy motion x t , guided by extracted audio features a, sound source location (SSL) s, motion genre g, and timestep t.
Following [77,11], the diffusion is modeled as a Markov chain process which progressively adds noise to clean motion vectors x 0 in t time steps, i.e.
q(x t |x t-1 ) = N ( √ α t x t-1 , (1-α t )I)(1)
where α t ∈ (0, 1). The model then learns to gradually denoise a noisy motion vector x t in t time steps, i.e. p(x t-1 |x t ). We directly predict the clean sample x0 in each diffusion step x0 = G(x t , t; a, s, g), where a is the audio features, s is the sound source location and g is the motion genre. This strategy, employed by [77,11,66,84], has been proved to be more efficient and accurate than predicting the noise ϵ t , suggested by [35]. We employ an encoder-only Transformer to reverse the diffusion process and predict the clean samples. The timestep, motion, and conditioning signals are each projected into the same latent dimension using separate feed-forward networks. Random masks are applied to the audio features a and the sound source location (SSL) s, after which all components are concatenated to form the complete token sequence z. The tokens are positionally embedded afterward and input into a transformer to get the output ẑ. The predicted clean sample is thus extracted from the last T tokens of ẑ by inputting it to another feed-forward network, where T = 240 is the length of the motion; see Fig. 5.
this section cite: ['b76', 'b10', 'b76', 'b10', 'b65', 'b83', 'b34']

Section: Loss Functions
We train MOSPA using the following loss functions. A simple mean squared error (MSE) loss is applied to the original clean sample and the predicted clean sample as the main objective: E∥ x0x 0 ∥ 2 2 . To guarantee the smooth variation on the predicted clean sample across frames, we also apply MSE loss to the rate of change of the vectors across frames: E∥δ x0 -δx 0 ∥ 2 2 . Combining the two simple losses we have L data = E∥ x0 -x 0 ∥ 2 2 + E∥δ x0 -δx 0 ∥ 2 2 . Geometric losses, encompassing position loss and velocity loss, are also incorporated, as we rely solely on joint We visualize motion results from five cases. MOSPA produces high-quality movements that closely correspond to the input spatial audio. We provide Expected Motion as a description for reference.
rotations and translations in motion vectors to represent poses:
L geo = E∥F K( x0 ) -F K(x 0 )∥ 2 2 + E∥δF K( x0 ) -δF K(x 0 )∥ 2 2 .
Furthermore, foot sliding is prevented by introducing the foot contact loss L f oot that measures the inconsistency in the velocities of the foot joints between the ground truth and the predicted motions.
We also incorporate trajectory loss and joint rotation loss to underscore their importance in achieving the training objectives and accelerate the convergence of the model, defined as L traj = E∥ traj 0traj 0 ∥ 2 2 + E∥δ traj 0 -δtraj 0 ∥ 2 2 and L rot = E∥r 0 -r 0 ∥ 2 2 + E∥δr 0 -δr 0 ∥ 2 2 respectively, where traj is the trajectory vector of the motion sequence and r is the joint rotations represented in the 6d format [101]. Given that trajectory and joint rotations are inherently encoded within the motion vectors, these supplementary losses represent an overlap with the existing loss terms, effectively amplifying the emphasis on trajectory and joint rotation accuracy through increased weighting. Empirically, we observe that this implementation accelerates model convergence and facilitates correct displacement direction generation in motion sequences. In sum, the total loss is given by:
L = λ data L data + λ geo L geo + λ f oot L f oot + λ traj L traj + λ rot L rot (2
)
All loss weights (λ) are initialized set to 1. At epoch 5,000 of the total 6,000 training epochs, λ traj and λ rot are increased to 3, thereby intensifying the emphasis on trajectory and rotation accuracy.
this section cite: ['b100']

Section: Implementation Details
In MOSPA, the diffusion model is a transformer-based diffusion network [11,77,100]. The encoder transformer is configured with a latent dimension of 512, 8 heads, and 4 layers. We employ AdamW [55] as the optimizer with an initial value of 1 × 10 -4 . The number of denoising steps used is 1000, and the noise schedule is cosine. The training phase concludes after 6, 000 epochs. Exceeding these recommended epoch counts may degrade model quality due to overfitting. The 5 Experiments Experiment Setup. We use our SAM dataset to evaluate the spatial audio-driven motion generation task. As detailed in Sec. 3, it contains 9 hours of human motion with paired binaural audio and corresponding sound source locations, covering 27 common spatial audio scenarios and 20 common reaction types. The dataset is split into training, validation, and test sub-datasets at a common ratio of 8:1:1. Consequently, the training sub-dataset comprises 2,400 motion sequences, while the validation and test sub-datasets each contain approximately 300 motion sequences. To keep fair setting [10], the motions and the audio clips are both downsampled to the frame rate of 30 FPS. The character is rotated to face the negative y-axis and initially translated to the origin in the world space in all motion sequences, and the sound source locations (SSL) are transformed to the local space of the character in every single frame.
Baselines and Metrics. Our system is the first work to receive spatial audio as input to generate human motion results. To our best knowledge, as there is no other system achieving this, we made adaptations on other audio2motion methods, such as EDGE [78], POPDG [60], LODGE [50] and Bailando [70] by replacing their original audio input with our spatial audio feature as input. We evaluated four metrics, focusing on motion quality and diversity: 1) R-precision, FID, Diversity These three metrics are calculated using the same setup proposed by [30]. Two bidirectional GRU are trained with a hidden size of 1024 for 1,500 epochs with a batch size of 64 to extract the audio features and the corresponding motion features, as suggested by [30]. Detailed implementation details of the feature extractor are provided in Appendix B.2. 2) APD [19,33] is calculated by
AP D(M ) = 1 N (N -1) N i=1 N j=1 j̸ =i L t=1 ∥s i t -s j t ∥ 2 1 2
, where M = { xi } is the set of generated motion sequences, N is the number of motion sequences in the set M , L is the number of frames of each motion sequence, and s i t ∈ xi is a state in the motion sequence xi .
this section cite: ['b10', 'b76', 'b99', 'b54', 'b9', 'b77', 'b59', 'b49', 'b69', 'b29', 'b29', 'b18', 'b32']

Section: Comparisons
Qualitative Results. We demonstrate the qualitative comparison in Fig. 6. For the same input spatial audio, our methods show the superiority of producing high-quality and realistic response motion. Other methods often exhibit various limitations due to their unique model characteristics. EDGE [78] and POPDG [60] demonstrate relatively strong performance among the four baselines, sharing a diffusion-based foundation with MOSPA, despite differences in their encoding and decoding mechanisms. Their shortcomings in generated samples can primarily be attributed to model size and their strong focus on music-like audio. The bad performance of LODGE [50] is likely due to its specialization in long-term music-like audio, resulting in deficiencies when handling short-term audio information with abrupt feature changes. Similarly, Bailando [70] faces challenges in processing rapidly changing spatial audio. More critically, due to its separate training process for upper and lower body parts, Bailando occasionally produces distorted or disjointed motions when encountering sudden changes in spatial audio. Please watch our supplementary video for more results. Furthermore, we test MOSPA on out-of-distribution audio-source configurations. As shown in Fig. 8, it maintains motion quality and intent alignment, demonstrating robustness to unseen spatial setups.
Quantitative Results. The quantitative results are reported in Tab. 2. MOSPA achieves the best performance as shown by the lowest FID value and the highest R-precision values. Also, our generated motions exhibit the closest diversity and APD [19] values compared with the Real Motion, demonstrating the effectively balanced variation and precision. Bailando [70] has the worst performance among the four baselines in practice, as illustrated by the extremely high FID. The model possibly lacks the ability to perceive commonly heard sounds other than music and also the spatial information of the audio. Our method, overall speaking, still demonstrates competitive performance in spatial audio conditioned motion generation, which is proved by the low values in precision-related metrics and the high values in diversity-related metrics.
this section cite: ['b77', 'b59', 'b49', 'b69', 'b18', 'b69']

Section: User Study.
We conducted a user study with 25 participants to assess the perceptual quality of motion generation. Participants evaluated five models (MOSPA, EDGE, POPDG, LODGE, Bailando) alongside ground truth (GT), selecting the best motion for: 1) Human Intent Alignment: Does the motion align with real-world intent? 2) Motion Quality: Which has the highest movement quality? 3) GT Similarity:Which best matches the GT motion? We provided GT motion and a textual description for reference. As shown in Fig. 7, MOSPA outperforms all baselines across all criteria, while LODGE and Bailando received the fewest selections, indicating limitations in generating realistic, semantically meaningful motions. See more details in Appendix C.
this section cite: []

Section: Ablation Study
We conducted ablation studies on the latent dimension, the number of attention heads, the diffusion step number, and the masking of motion genre, with results summarized in Tab. 3. All ablation experiments maintained consistent training epoch counts throughout.
this section cite: []

Section: Latent Dimension.
The default latent dimension of MOSPA's encoder transformer is 512. In our study, reducing it to 256 slightly increases the APD [19] value but also degrades the R-precision and the FID, leading to an overall decline in model performance, as seen in row 1 and row 2 in Tab. 3.
this section cite: ['b18']

Section: Number of Attention Heads.
We reduced the number of attention heads in MOSPA 's encoder transformer from 8 to 4, observing degradation in almost all of the metrics except a slight improvement in APD [19]. This reduction compromises overall model performance without yielding significant improvements in training efficiency, as seen in row 1 and row 3 in Tab. 3.
this section cite: ['b18']

Section: Number of Diffusion Steps.
We evaluated MOSPA with varying diffusion step numbers, reducing it from 1000 to 100 and further to 4, as detailed in rows 1, 4, and 5 of Tab. 3. Fewer steps slightly degrade the performance as shown by the increase in FID and degradation in diversity, thereby lowering the upper limit of the power of the model.   We evaluate the contribution of the extracted audio features by conducting an ablation study on the effectiveness of MFCC [17] and tempogram [28] features. As shown in Tab. 4, improvements in FID and Rprecision-two key metrics for assessing generative quality and correspondence-demonstrate their significance in model.
this section cite: ['b16', 'b27']

Section: Conclusion
This introduces a novel task for enabling virtual humans to respond realistically to spatial auditory stimuli. We present a comprehensive SAM dataset, capturing human movement in response to spatial audio, and propose MOSPA, a diffusion-based generative model with an attention-based fusion mechanism. Once trained, MOSPA synthesizes diverse, high-quality motions that adapt to varying spatial audio inputs with binaural recording. Extensive evaluations show MOSPA achieves state-of-the-art performance on this task. Limitations and Future Works. Physical Correctness: While MOSPA generates diverse and semantically plausible motions, it lacks physical constraints, which may lead to physically implausible artifacts. Integrating physics-based control methods [19,59,76,40,95,96,62,82] could improve motion realism and embodiment fidelity (see Fig. 9 for spatial audio-driven humanoid robot control). Body Modeling: This work focuses on body motion and omits finer-grained components such as hand gestures and facial expressions supported by SMPL-X [63]. Extending the model to full-body motion generation [57, 52, 86, 64, 5]-including hand motions-remains an important direction for future research. Scene Awareness: The current framework does not incorporate awareness of surrounding environments or physical scene geometry, limiting its ability to produce scene-consistent or contact-aware motions. Future extensions could integrate scene representations or affordance prediction [14,80,83,8,81] with spatial audio signals to enhance human motion generation.
this section cite: ['b18', 'b58', 'b75', 'b39', 'b94', 'b95', 'b61', 'b81', 'b62', 'b13', 'b79', 'b82', 'b7', 'b80']

Section: References
Ref_id:b0 Title: Listen, denoise, action! audio-driven motion synthesis with diffusion models Year: (2023)
Ref_id:b1 Title: Rhythmic gesticulator: Rhythmaware co-speech gesture synthesis with hierarchical neural embeddings Year: (2022)
Ref_id:b2 Title: Gesturediffuclip: Gesture diffusion model with clip latents Year: (2023)
Ref_id:b3 Title: Belfusion: Latent diffusion for behavior-driven human motion prediction Year: (2023)
Ref_id:b4 Title: Motioncraft: Crafting whole-body motion with plug-and-play multimodal controls Year: (2025)
Ref_id:b5 Title: The measurement of power spectra dover publications Year: (1958)
Ref_id:b6 Title: Maximum filter vibrato suppression for onset detection Year: (2013-09)
Ref_id:b7 Title: Generating human motion in 3d scenes from text descriptions Year: (2024)
Ref_id:b8 Title: Humanmac: Masked motion completion for human motion prediction Year: (2023)
Ref_id:b9 Title: Pay attention and move better: Harnessing attention for interactive motion generation and training-free editing Year: (2024)
Ref_id:b10 Title: Taming diffusion probabilistic models for character control Year: (2024)
Ref_id:b11 Title: Executing your commands via motion diffusion in latent space Year: (2023)
Ref_id:b12 Title: Siggesture: Generalized co-speech gesture synthesis via semantic injection with large-scale pre-training diffusion models Year: (2024)
Ref_id:b13 Title: Laserhuman: language-guided scene-aware human motion generation in free environment Year: (2024)
Ref_id:b14 Title: Mofusion: A framework for denoising-diffusion-based motion synthesis Year: (2023)
Ref_id:b15 Title: Motionlcm: Real-time controllable motion generation via latent consistency model Year: (2024)
Ref_id:b16 Title: Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences Year: (1980)
Ref_id:b17 Title:  Year: (2025-02-05)
Ref_id:b18 Title: Learning conditional adversarial skill embeddings for physics-based characters Year: (2023)
Ref_id:b19 Title: Beat tracking by dynamic programming Year: (2007)
Ref_id:b20 Title: A bi-directional attention guided cross-modal network for music based dance generation Year: (2022)
Ref_id:b21 Title: Go to zero: Towards zero-shot motion generation with million-scale data Year: (2025)
Ref_id:b22 Title: Motionwavelet: Human motion prediction via wavelet manifold learning Year: (2024)
Ref_id:b23 Title: Self-supervised moving vehicle tracking with stereo sound Year: (2019)
Ref_id:b24 Title: Audio set: An ontology and human-labeled dataset for audio events Year: (2017)
Ref_id:b25 Title: Learning individual styles of conversational gesture Year: (2019)
Ref_id:b26 Title: Tm2d: Bimodality driven 3d dance generation via music-text integration Year: (2023)
Ref_id:b27 Title: Cyclic tempogram-a mid-level tempo representation for musicsignals Year: (2010)
Ref_id:b28 Title: Action2motion: Conditioned generation of 3d human motions Year: (2020)
Ref_id:b29 Title: Generating diverse and natural 3d human motions from text Year: (2022)
Ref_id:b30 Title: Momask: Generative masked modeling of 3d human motions Year: (2024)
Ref_id:b31 Title: Dimensionality reduction by learning an invariant mapping Year: (2006)
Ref_id:b32 Title: Stochastic scene-aware motion prediction Year: (2021)
Ref_id:b33 Title: Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills Year: (2025)
Ref_id:b34 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b35 Title: Phase-functioned neural networks for character control Year: (2017)
Ref_id:b36 Title: Modeling and driving human body soundfields through acoustic primitives Year: (2024)
Ref_id:b37 Title: Interact: Capture and modelling of realistic, expressive and interactive activities between two persons in daily scenarios Year: (2024)
Ref_id:b38 Title: Como: Controllable motion generation through language guided pose code editing Year: (2024)
Ref_id:b39 Title: Modskill: Physical character skill modularization Year: (2025)
Ref_id:b40 Title: Motiongpt: Human motion as a foreign language Year: (2023)
Ref_id:b41 Title: Visage: Video-to-spatial audio generation Year: (2025)
Ref_id:b42 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b43 Title: Acoustic volume rendering for neural impulse response fields Year: (2024)
Ref_id:b44 Title: Resounding acoustic fields with reciprocity Year: (2025)
Ref_id:b45 Title: Controllable group choreography using contrastive diffusion Year: (2023)
Ref_id:b46 Title: Danceformer: Music conditioned 3d dance generation with parametric motion transformer Year: (2022)
Ref_id:b47 Title: Ai choreographer: Music conditioned 3d dance generation with aist++ Year: (2021)
Ref_id:b48 Title: Finedance: A fine-grained choreography dataset for 3d full body dance generation Year: (2023)
Ref_id:b49 Title: Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives Year: (2024)
Ref_id:b50 Title: Rmd: A simple baseline for more general human motion generation via training-free retrieval-augmented motion diffuse Year: (2024)
Ref_id:b51 Title: Motionx: A large-scale 3d expressive whole-body human motion dataset Year: (2023)
Ref_id:b52 Title: Yansong Tang, and Xin Tong. Plan, posture and go: Towards open-world text-to-motion generation Year: (2023)
Ref_id:b53 Title: Learning hierarchical cross-modal association for co-speech gesture generation Year: (2022)
Ref_id:b54 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b55 Title: Vicon Motion Systems Ltd. Vicon motion capture systems Year: (2025-02-05)
Ref_id:b56 Title: Humantomato: Text-aligned whole-body motion generation Year: (2023)
Ref_id:b57 Title: Exploring the scaling law in autoregressive motion generation model Year: (2024)
Ref_id:b58 Title: Perpetual humanoid control for real-time simulated avatars Year: (2023)
Ref_id:b59 Title: Popdg: Popular 3d dance generation with popdanceset Year: (2024)
Ref_id:b60 Title: Eric Battenberg, and Oriol Nieto. librosa: Audio and music signal analysis in python Year: (2015)
Ref_id:b61 Title: Unified synthesis of physical human-scene interactions through task tokenization Year: (2025)
Ref_id:b62 Title: Expressive body capture: 3d hands, face, and body from a single image Year: (2019)
Ref_id:b63 Title: Coda: Coordinated diffusion noise optimization for whole-body manipulation of articulated objects Year: (2025)
Ref_id:b64 Title: Speech drives templates: Co-speech gesture synthesis with learned templates Year: (2021)
Ref_id:b65 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b66 Title: Constant-q transform toolbox for music processing Year: (2010)
Ref_id:b67 Title: It takes two: Real-time co-speech two-person's interaction generation via reactive auto-regressive diffusion model Year: (2024)
Ref_id:b68 Title: Interactive character control with auto-regressive motion diffusion models Year: (2024)
Ref_id:b69 Title: Bailando: 3d dance generation by actor-critic gpt with choreographic memory Year: (2022)
Ref_id:b70 Title: Duolando: Follower gpt with off-policy reinforcement learning for dance accompaniment Year: (2024)
Ref_id:b71 Title: Deepphase: Periodic autoencoders for learning motion phase manifolds Year: (2022)
Ref_id:b72 Title: Comusion: Towards consistent stochastic human motion prediction via motion diffusion Year: (2024)
Ref_id:b73 Title: Both ears wide open: Towards language-driven spatial audio generation Year: (2024)
Ref_id:b74 Title: Dance with melody: An lstm-autoencoder approach to musicoriented dance synthesis Year: (2018)
Ref_id:b75 Title: Maskedmimic: Unified physics-based character control through masked motion inpainting Year: (2024)
Ref_id:b76 Title: Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model Year: (2023)
Ref_id:b77 Title: Edge: Editable dance generation from music Year: (2023)
Ref_id:b78 Title: Tlcontrol: Trajectory and language control for human motion synthesis Year: (2023)
Ref_id:b79 Title: Scene-aware generative network for human motion synthesis Year: (2021)
Ref_id:b80 Title: Towards diverse and natural scene-aware 3d human motion synthesis Year: (2022)
Ref_id:b81 Title: Simulating stylized human-scene interactions with retrieval-augmented script generation Year: (2024)
Ref_id:b82 Title: Move as you say interact as you can: Language-guided human motion generation with scene affordance Year: (2024)
Ref_id:b83 Title: Omnicontrol: Control any joint at any time for human motion generation Year: (2023)
Ref_id:b84 Title: Actformer: A gan-based transformer towards general actionconditioned 3d human motion generation Year: (2023)
Ref_id:b85 Title: Intermimic: Towards universal whole-body control for physics-based human-object interactions Year: (2025)
Ref_id:b86 Title: Sounding bodies: modeling 3d spatial sound of humans using body pose and audio Year: (2023)
Ref_id:b87 Title: Unified text, music and motion generation Year: (2024)
Ref_id:b88 Title: Speech gesture generation from the trimodal context of text, audio, and speaker identity Year: (2020)
Ref_id:b89 Title: Diverse trajectory forecasting with determinantal point processes Year: (2019)
Ref_id:b90 Title: Bidirectional autoregessive diffusion model for dance generation Year: (2024)
Ref_id:b91 Title: Mode-adaptive neural networks for quadruped motion control Year: (2018)
Ref_id:b92 Title: Generating human motion from textual descriptions with discrete representations Year: (2023)
Ref_id:b93 Title: Large motion model for unified multi-modal motion generation Year: (2024)
Ref_id:b94 Title: Physpt: Physics-aware pretrained transformer for estimating human dynamics from monocular videos Year: (2024)
Ref_id:b95 Title: Incorporating physics principles for precise human motion prediction Year: (2024)
Ref_id:b96 Title: Semantic gesticulator: Semantics-aware co-speech gesture synthesis Year: (2024)
Ref_id:b97 Title: The sound of pixels Year: (2018)
Ref_id:b98 Title: Learning to reason about spatial sounds with large language models Year: (2024)
Ref_id:b99 Title: Emdm: Efficient motion diffusion model for fast and high-quality motion generation Year: (2025)
Ref_id:b100 Title: On the continuity of rotation representations in neural networks Year: (2019)
Ref_id:b101 Title: Taming diffusion models for audio-driven co-speech gesture generation Year: (2023)
Ref_id:b102 Title: Human motion generation: A survey Year: (2023)
Ref_id:b103 Title: Justification: Editing (e.g., grammar, spelling, word choice) Guidelines: • The answer NA means that the core method development in this Year: (2022)
Ref_id:b104 Title: ) for what should or should not be described Year: ()
