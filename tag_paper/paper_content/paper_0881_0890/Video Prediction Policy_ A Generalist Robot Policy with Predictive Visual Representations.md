Title: Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
Abstract: Visual representations play a crucial role in developing generalist robotic policies. Previous vision encoders, typically pre-trained with singleimage reconstruction or two-image contrastive learning, tend to capture static information, often neglecting the dynamic aspects vital for embodied tasks. Recently, video diffusion models (VDMs) demonstrate the ability to predict future frames and showcase a strong understanding of physical world. We hypothesize that VDMs inherently produce visual representations that encompass both current static information and predicted future dynamics, thereby providing valuable guidance for robot action learning. Based on this hypothesis, we propose the Video Prediction Policy (VPP), which learns implicit inverse dynamics model conditioned on predicted future representations inside VDMs. To predict more precise future, we fine-tune pre-trained video foundation model on robot datasets along with internet human manipulation data. In experiments, VPP achieves a 18.6% relative improvement on the Calvin ABC-D generalization benchmark compared to the previous state-of-the-art, and demonstrates a 31.6% increase in success rates for complex real-world dexterous manipulation tasks. Videos and code are available at https:   //video-prediction-policy.github.io.

Section: Introduction
Building generalist robot policies capable of solving a variety of tasks is a rapidly advancing area of research (Brohan
this section cite: []

Section: …
Figure 1. Visual representations inside video prediction models explicitly express both current and future frames, providing valuable future information for embodied agent. Previous vision encoders did not have explicit future representations. et al., 2023;Team et al., 2024;Wu et al., 2023a;Guo et al., 2025;Cui et al., 2025;Ding et al., 2024;2025;Shi et al., 2025;Zhao et al., 2025). A crucial component in these generalist policies is the vision encoder, which captures visual information from pixel observations. Many studies have focused on optimizing vision representations for embodied agents, often leveraging internet video datasets (Ebert et al., 2021;Grauman et al., 2022) and self-supervised techniques such as single-image reconstruction (Majumdar et al., 2023;Karamcheti et al., 2023;Gupta et al., 2024), two-image contrastive learning, and image-text contrastive learning (Nair et al., 2022;Ma et al., 2022). Although these visual pretraining methods have demonstrated success for embodied tasks, they may not fully exploit the dynamic information encoded in sequential video datasets, as they typically operate on only one or two sampled images.
Recently, powerful video diffusion models (VDMs) (Ho et al., 2022;Blattmann et al., 2023a;Hong et al., 2022;Yang et al., 2024) have achieved impressive results in video generation tasks. Instead of performing pre-training operation on single image or pairs of images, VDMs directly model entire video sequences. Text-guided video prediction models (TVPs) (Gu et al., 2023;Chen et al., 2023) can even predict future frames based on current observations and instructions, demonstrating a good understanding of the physical dynamics.
Inspired by the strong prediction capabilities of TVP models, we hypothesize that they inherently contain valuable physical dynamics knowledge and can produce more effective visual representations for embodied agent. We take a deeper look at the visual representation inside TVP models. These representations are typically structured as a tensor with dimensions (T, H, W ), explicitly representing 1 current step and (T -1) predicted future steps (Blattmann et al., 2023a), where H and W correspond to the height and width of the image representation. In contrast, previous vision encoders do not explicitly capture future representations, as shown in Figure 1. Based on this distinction, we refer to these latent variables within the video diffusion model as "predictive visual representations".
Our key insight is that the downstream policy can implicitly learn the inverse dynamics model by tracking the robot's movements within the predictive representation. As long as the video model accurately predicts future scenarios for diverse tasks, the policy can generate appropriate actions by tracking robot arm's position implicitly. In this way, we can transfer the generalization capabilities of the video prediction model to robotic policy. We only need few demonstrations to align the robot's action space with the visual space.
Building on this insight, we introduce the Video Prediction Policy (VPP), which employs a two-stage learning process: First, we fine-tune a general-purpose video diffusion model into a text-guided video prediction (TVP) model using internet human and robot manipulation data (Goyal et al., 2017;O'Neill et al., 2023). This step aims to develop a controllable video generation model that improves prediction capabilities in the manipulation domain. In the second stage, we learn a inverse dynamics model conditioned on the predictive representations from the TVP model. Since we direct use the internal representation and avoid the need for multiple denoising steps as in previous work (Black et al., 2023;Du et al., 2024), VPP can operate at high frequency in a closed-loop manner. We also visualize the representations within the VDM and confirm that they effectively capture key information about future evolution.
In experiments, VPP consistently outperform other baseline algorithms across two simulated (Mees et al., 2022;Yu et al., 2020) and two real-world settings, demonstrating the effectiveness of our approach. Notably, the VPP achieves a 41.5% improvement in the Calvin ABC→D benchmark (Mees et al., 2022) compared to the previous SOTA method (Wu et al., 2023a). In real-world experiments, VPP shows a 31.6% improvement in success rate over the strongest baseline on high-dimensional dexterous hand manipulation tasks.
this section cite: ['b51', 'b26', 'b17', 'b18', 'b49', 'b63', 'b21', 'b23', 'b39', 'b33', 'b27', 'b41', 'b38', 'b30', 'b31', 'b59', 'b24', 'b13', 'b22', 'b42', 'b4', 'b20', 'b40', 'b61', 'b40']

Section: Related Works
Visual Representation Learning for Robotics. Selfsupervised learning (SSL) techniques, such as contrastive (Chen et al., 2021;2020), distillation-based (Baevski et al., 2022;Caron et al., 2020), and reconstructive (He et al., 2022;Bao et al., 2021), have achieved significant advancements in visual representation learning. Prior research has shown that these SSL techniques enable vision encoders to produce effective representations for embodied AI tasks (Yadav et al., 2023b;a;Parisi et al., 2022;Radosavovic et al., 2023;Chen et al., 2024a), capturing both high-level semantic and low-level spatial information. Notably, methods like R3M (Nair et al., 2022), vip (Ma et al., 2022), VC-1 (Majumdar et al., 2023), and Voltron (Karamcheti et al., 2023) have specifically focused on embodied tasks by innovating pre-training approaches on human manipulation video datasets (Goyal et al., 2017;Grauman et al., 2022). However, regardless of the training objective, the learned vision encoders primarily focus on extracting pertinent information from current observations without explicitly predicting future states. In contrast, our Video Prediction Policy leverages predictive representations within video prediction models to explicitly encapsulate both current and predicted future frames.
Future Prediction for Embodied Control Tasks. Existing research also explores the use of future prediction to enhance policy learning (Bharadhwaj et al., 2024;Chen et al., 2024b;Ye et al., 2024;Guo et al., 2024;Zhang et al., 2025;Song et al., 2025). For example, SuSIE (Black et al., 2023) conditions its control policy on a predicted future keyframe generated by InstructPix2Pix (Brooks et al., 2023), while UniPi (Du et al., 2024) learns the inverse dynamics between two generated frames. These methods rely on a single future prediction step to determine actions, which may not accurately capture the complexities of physical dynamics. Additionally, they denoise the final future image which is time-cosuming and lead to low control frequency. GR-1 (Wu et al., 2023a) generates subsequent frames and actions autoregressively. However, it only generates one image per forward pass, and its prediction quality lags behind that of diffusion-based methods. Furthermore, GR-1 does not leverage pre-trained video foundation models. In contrast, VPP leverages representation fine-tuned from video foundation model, and predict a sequence of future frames to more effectively inform policy learning.
this section cite: ['b14', 'b1', 'b10', 'b29', 'b2', 'b43', 'b46', 'b41', 'b38', 'b39', 'b33', 'b22', 'b23', 'b3', 'b60', 'b25', 'b62', 'b50', 'b4', 'b9', 'b20']

Section: Visual Representation inside Diffusion Models.
Diffusion models have achieved remarkable success in the image and video generation tasks (Rombach et al., 2022;Blattmann et al., 2023a). Although diffusion models are trained as denoisers, researches have shown that image diffusion models can also function effectively as vision encoders, generating meaningful visual representations that is linear-separable for discrimination tasks (Xiang et al., 2023) and invaluable for semantic segmentation (Luo et al., 2024). Gupta et al. (2024) also point out that representation inside image diffusion are versatile for embodied tasks. However, the capabilities of representations within video diffusion models have not been extensively explored. He et al. (2024) try to use latent representation inside discrete VDMs to assist policy learning, however it need not leverage pre-trained video foundation models and train from scratch. Our findings suggest that representation within pretrained VDMs have a unique predictive property, making them especially useful for sequential embodied control tasks.
this section cite: ['b48', 'b56', 'b36', 'b27', 'b28']

Section: Preliminaries
Video Diffusion Models. The core idea of diffusion models is to continuously add Gaussian noise to make video sequences a Gaussian and leverage the denoising process for generating videos. Let x 0 represent a real video sample, the forward process aims to add Gaussian noise and result in a set of noisy data, i.e., q(x t |x t-1 ) = N (x t ;
√ α t x t-1 , (1α t )I) , where x t and α t indicate the noisy data and noise amplitude at the timestep t. Let ᾱt = t i=1 α i , the above process can be simplified as:
x t = √ ᾱt x 0 + √ 1 -ᾱt ϵ t .(1)
The reverse process starts from the most noisy sample x T can be described in a variational approximation of the probabilities q(x t-1 |x t ), as follows:
p(x t-1 |x t ) = N (x t-1 ; √ ᾱt-1 µ θ (x t , t), (1 -ᾱt-1 )I).(2)
where µ θ (x t , t) = (x t -√ 1 -ᾱt ϵ θ (x t , t))/ √ ᾱt is a learnable neural network to estimate x t-1 . Further, in textguided video generation, the denoising process learns the noise estimator ϵ θ (x t , c) to approximate the score function √ 1 -ᾱt ∇ xt log p ψ (x t |c), controlling the video generation based on the initial frame and language prompt.
this section cite: []

Section: Diffusion Policy.
The diffusion model has also proven effective in action learning, known as diffusion policy (Chi et al., 2023). The diffusion policy aims to denoise the action sequence a i = ( a i , a i+1 , ..., a i+m ) based on observations s i and instruction. Chi et al. (Chi et al., 2023) point out that diffusion policy is capable of expressing complex multimodal action distributions and stabilizing training. Recent work (Reuss et al., 2024) further enhances the diffusion policy by incorporating the advanced diffusion transformer (DiT) block (Peebles & Xie, 2023), a technique we also adopt in the Video Prediction Policy to improve performance.
this section cite: ['b16', 'b16', 'b47', 'b44']

Section: Video Prediction Policy
In this section, we describe the two-stage learning process of the Video Prediction Policy, shown in Figure 2. Initially, we train the Text-guided Video Prediction (TVP) model across diverse manipulation datasets to harness physical knowledge from internet data; subsequently, we design networks to aggregate predictive visual representations inside the TVP model and output final robot actions.
this section cite: []

Section: Text-guided Video Prediction (TVP) Model for Robot Manipulation.
Recent advancements have focused on training general video generation models using extensive online video datasets, which encode abundant prior knowledge about the physical world's dynamics. However, we notice that these models are not fully controllable and fail to yield optimal results in specialized domains such as robot manipulation.
To address this, we fine-tune the general video generation model into a specialized "Manipulation TVP Model" to enhance prediction accuracy.
We chose the open-sourced Stable Video Diffusion (SVD) model (Blattmann et al., 2023a) with 1.5 billion parameters as our foundation. we observe that the open-sourced SVD model conditions only on initial-frame images s 0 . We augment the model to incorporate CLIP (Radford et al., 2021) language feature l emb using cross-attention layers. Furthermore, we adjust the output video resolution to 16×256×256 to improve training and inference efficiency. Despite these modifications, we preserve the other components of the original pre-trained SVD framework to retain its core capabilities. We denote this modified version as V θ . In this setup, the initial observation s 0 is concatenated channelwise with each predicted frame as a condition. Then model V θ is trained with diffusion objective, reconstructing the full video sequence
x 0 = s 0:T in dataset D from noised samples x t = √ ᾱt x 0 + √ 1 -ᾱt ϵ: L D = E x0∼D,ϵ,t ∥V θ (x t , l emb , s 0 ) -x 0 ∥ 2(3)
The video prediction objective offers a unified interface that directly generates future visual sequences, enabling the TVP model to harness physical knowledge from diverse datasets. These include internet human manipulation datasets D H , internet robot manipulation data D R , and also self-collected datasets D C . Given the varying quality and scale of these datasets, we introduce specific coefficients λ to appropriately balance the influence of different dataset types:
L video = λ H L D H + λ R L D R + λ C L D C (4
)
Then we froze the fine-tuned manipulation TVP models in downstream action learning.
this section cite: ['b45']

Section: Action Learning Conditioned on Predictive Visual Representation
TVP Model as Vision Encoder. After training the TVP model specifically for manipulation tasks, it can accurately Tokenized and Time Position Embedding Text Embedding Action Denoise … … … … Interpolate & Stack ⊕ ⊕ Figure 2. In the first stage, VPP fine-tunes a general-purpose video foundation model into a manipulation-focused Text-guided Video
Prediction (TVP) model using robot and internet manipulation datasets. In the second stage, we use video-former to aggregate the representations from the TVP model during the first forward pass, followed by the diffusion policy head. This approach enables VPP to learn an implicit inverse dynamics model from the predicted future while maintaining a high control frequency.
predict future sequences based on image observations and instructions. However, denoising an entire video sequence is highly time-consuming and may lead to open-loop control issues, as discussed in (Du et al., 2024). Moreover, videos in their original pixel format often contain excessive, irrelevant information that can interfere with effective decision-making.
To address these concerns, we employ the video diffusion model primarily as a "vision encoder" rather than a "denoiser" by performing only a single forward step. Our insight is that the first forward step, while not yielding a clear video, still provides a rough trajectory of future states and valuable guidance. This insight is verified in our experiment section and shown in Fig 4. Specifically, we concatenate the current image s 0 with the final noised latent q(x t ′ |x 0 ) (typically white noise) and input this combination into the TVP model. We then directly leverage the latent features. Previous work (Xiang et al., 2023) highlights that up-sampling layers in diffusion models yield more effective representations. The feature at the m th up-sampling layer, with width W m and height H m , is expressed as:
L m = V θ (x t ′ , l emb , s 0 ) (m) , L m ∈ R T ×Cm×Wm×Hm
To effectively aggregate features from the up-sampling layers and eliminate the need for manual layer selection, we propose an automatic method for aggregating features across different layers. First, we linearly interpolate each layer's feature map to the same height and width W p × H p :
L ′ m = Interpolation(L m ), L ′ m ∈ R T ×Cm×Wp×Hp
We then stack the features along the channel dimension. The final predictive visual representation F p ∈ R T ×( m Cm)×Wp×Hp is given by:
F p = concate((L ′ 0 , L ′ 1 , . . . , L ′ m ), dim = 1)
For a robot with multiple camera views, such as a third-view and a wristed camera, we predict the future for each view independently, denoted as F static p , F wrist p .
Video Former. These predictive representations within the video diffusion model are still high-dimensional, as they express a sequence of image features. To efficiently aggregate representations across spatial, temporal, and multi-view dimensions, we design a Video Former to consolidate this information into a fixed number of tokens. The Video Former initializes learnable tokens Q [0:T,0:L] with fixed length T × L, performing spatial-temporal attention (Blattmann et al., 2023b) on each corresponding frame, followed by feed-forward layers. Formally, this branch can be expressed as follows where i is the index of frame:
Q ′ = {Spat-Attn(Q[i], (F static p [i], F wrist p [i]))} T i=0 Q ′′ = FFN(Temp-Attn(Q ′ )).(5)
Action Generation. After the Video-Former aggregates the Predictive feature into learnable tokens Q ′′ , a diffusion policy is employed as the action head to generate the action sequence a 0 ∈ A based on Q ′′ . We integrate the aggregated presentation Q ′′ into diffusion transformer blocks using cross-attention layers. The diffusion policy aims to reconstruct the original actions a 0 from noised action a k = βk a 0 + 1 -βk ϵ, where ϵ represents white noise, and βk is the noisy coefficient at step k. Table 2. Multi-task success rate on Metaworld. We use a single languageconditioned policy to solve all 50 tasks.
noise ϵ and minimize the following loss function:
L diff (ψ; A) = E a0,ϵ,k ∥D ψ (a k , l emb , Q ′′ ) -a 0 ∥ 2 (6)
this section cite: ['b20', 'b56']

Section: Experiments
In this section, we conduct extensive experiments on both simulated and real-world robotic tasks to evaluate the performance of the video prediction policy (VPP). We aim to answer the following questions:
1. Can VPP achieve a higher success rate in manipulation tasks with predictive visual representations? 2. How do the video pre-training and internet manipulation datasets enhance the performance of VPP? 3. How does predictive representation compare to previous visual representations? 4. Which layer of the video diffusion model provides the most effective predictive visual representations?
this section cite: []

Section: Simulation Setups and Baselines
CALVIN Benchmark. CALVIN (Mees et al., 2022) is a widely used benchmark designed to assess the instructionfollowing capability of robotic policies in long-horizon manipulation tasks. We focus on the challenging ABC→D setting, where the agent is trained in the ABC environment and evaluated in the unseen D environment, as illustrated in Figure 3. We use settings same as GR1 (Wu et al., 2023a) which only use the language-annotated ABC datasets for training.
MetaWorld Benchmark. Metaworld (Yu et al., 2020) features a Sawyer robot performing various manipulation tasks and is widely used to evaluate the precision and dexterity of robotic policies. As shown on the right of Figure 3, it includes 50 tasks with a rich array of operating objects at different levels of difficulty (Radosavovic et al., 2023). We use official Oracle policy to collect 50 trajectories for each task as our training dataset.
this section cite: ['b40', 'b61', 'b46']

Section: VPP Training Details.
As outlined in Sec. 4, we use a twostage training process. In the first stage, we fine-tune a video foundation model into a manipulation-focused TVP model. The videos used in this stage include 193,690 human manipulation trajectories (Goyal et al., 2017) and 179,074 robotic manipulation trajectories (O'Neill et al., 2023), along with downstream task videos, such as the official Calvin ABC videos, the MetaWorld videos, and real-world videos. Given the varying scales and quality of these datasets, we apply different sampling ratios, following the approach in Octo (Team et al., 2024). Detailed dataset scales and sampling ratios can be found in Appendix B. Fine-tuning the video model takes 2-3 days on eight NVIDIA A100 GPUs.
In the second stage, we train a generalist policy with Calvin or Metaworld dataset, which requires approximately 6-12 hours on four NVIDIA A100 GPUs.
Policy Roll-out Details. Previous works choose to denoise high-precision videos, a process that is time-consuming and results in low-frequency (Black et al., 2023), or even openloop control (Du et al., 2024). In contrast, our approach uses the TVP model as an encoder rather than a denoiser, ensuring that each observation is processed through the TVP model only once, which takes less than 160 ms. Then downstream policy generate action conditioned on the predictive representation. This modification allows us to achieve a significantly higher frequency of 7-10 Hz with consumerlevel NVIDIA RTX 4090 GPU. Additionally, we implement action chunking (Chi et al., 2023) with 10 steps to further improve the control frequency.
Comparisons. Generalist robot policy has been widely explored in previous studies. In our experiments, we opted to compare against a representative subset of prior methods that have either achieved state-of-the-art performance or share a similar approach with our methods.
• RT-1 (Brohan et al., 2022). A direct action learning robot policy that integrates semantic information using Efficient-Net with FiLM-conditioning, followed by token learners for action learning.
• Diffusion Policy (Chi et al., 2023). A direct action learning policy with novel action diffusers.
• Robo-Flamingo (Li et al., 2023). A direct action learning policy that leverages a pre-trained LLM, incorporating visual information into each layer in a flamingo style (Alayrac et al., 2022).
• Uni-Pi (Du et al., 2024). Begins by learning a video prediction model to generate future sequences and then learns an inverse kinematics model between two frames to determine actions.
• MDT (Reuss et al., 2024). Learns a diffusion transformer policy along with an auxiliary mae loss to reconstruct one masked future frame.
• Susie (Black et al., 2023). Uses a fine-tuned Instruct-Pix2Pix (Brooks et al., 2023) model to generate a goal image and learns a downstream diffusion policy conditioned on the goal image.
• GR-1 (Wu et al., 2023a). Learns video and action sequences jointly using an auto-regressive transformer. During policy execution, GR-1 outputs one future frame followed by one action.
• Robo-Uniview (Liu et al., 2024). Learns a 3d-aware visual encoder with 3d occupation loss to assist policy learning.
• Vidman (Wen et al., 2024). Pre-trained on the Open X-Embodiment dataset (OXE) video datasets and use a layer-wise self-attention adapter to transform video representation into policy model. However, Vidman did not finetune video model on down-stream tasks which lead to sub-optimal performance.
Quantitative Results. The comparisons on the Calvin benchmark are shown in Table 1. Results for Robo-Flamingo, Susie, GR-1, and 3D Diffuser Actors are recorded from their original papers. The MDT result is run on official implementation. The RT-1 result is sourced from (Li et al., 2023) and the Uni-Pi result from (Black et al., 2023). We also ran the Diffusion Policy based on the official opensource codebase with CLIP language conditions. Our proposed Video Prediction Policy significantly improved the previous state-of-the-art result from an average task completion length of 3.65 to 4.33. Even with only 10% of the annotated Calvin ABC data used for training, our method still achieved a length of 3.25, which exceeds the results of related methods using full data. Furthermore, the Video Prediction Policy also achieved the best performance in the MetaWorld benchmark with 50 tasks, outperforming the similar strongest GR-1 baseline by 10.8% in average success rate.
Visualizations of Predictive Representations. Since we use the video prediction model as a vision encoder and perform a single forward pass to obtain predictive representations, we are curious about the quality of these representations. In Figure 4 , we visualize the ground truth future, single-step predictions, and 30-step denoised predictions. We can observe that single-step representation already conveys valuable information, such as the movement of objects and the robot arm, which effectively supports downstream action learning.
this section cite: ['b22', 'b51', 'b4', 'b20', 'b16', 'b7', 'b16', 'b34', 'b0', 'b20', 'b47', 'b4', 'b9', 'b35', 'b53', 'b34', 'b4']

Section: Ablation Study
VPP achieves significant improvements in simulated experiments. In this section, we conduct ablation studies to identify the effectiveness of different components of VPP.
All ablation study are performed on Calvin ABC-D benchmark and evaluated with average task completion length.
this section cite: []

Section: Effectiveness of Predictive Visual Representations.
To verify the effectiveness of representation inside VDM, we replace the VDM vision encoder with several other pretrained vision encoders designed for embodied tasks, while keeping all other components and settings unchanged.
1. Stable-VAE (Blattmann et al., 2023a), pre-trained with a VAE image reconstruction loss. Since the VAE encoder-decoder already performs well in reconstructing images from video datasets, we did not perform further fine-tuning. The input 256×256 images are encoded into 32×32 features with VAE, which are then Input Ground Truth 30 Steps Denoise Prediction 1 Step Direct Prediction "Place the grasped object in the drawer." "Place the orange to blue plate." Figure 4. Visualization of one-step forward visual representations. We can observe that one-step representation already provide valuable information on physical evolution, although the textures and details are not precise. Encoder Pre-training Type Avg. Length ↑ VDM (ours) Video Generation 4.33 Stable-VAE VAE Reconstruction 2.58 VC-1 MAE Reconstruction 1.23 Voltron MAE Reconstruction+ Language Generation 1.54
this section cite: []

Section: ∼140ms
Table 4. Ablation study on video pre-training and architecture.
resampled into 256 tokens via resampler (Jaegle et al., 2021) before passing to the diffusion policy, consistent with VPP.
2. VC-1 (Majumdar et al., 2023), pre-trained with a masked autoencoder loss. The authors note that finetuning vc-1 encoder with MAE loss on downstream task datasets can significantly improve performance. For a fair comparison, we first fine-tuned the model on the same video datasets used in VPP. The vc-1 features are resampled into 256 tokens with resampler and pass to policy head.
3. Voltron (Karamcheti et al., 2023), pre-trained with both MAE future reconstruction and language generation tasks. We also fine-tuned the model on our video datasets and resampled the features into 256 tokens.
The results, presented in Table 3, indicate that replacing our predictive visual representations leads to a clear decline in performance.
this section cite: ['b32', 'b39', 'b33']

Section: Effectiveness of Video Pre-training and Internet Manipulation Datasets.
A significant advantage of the VPP is its ability to leverage the physical knowledge encoded in pre-trained video generation models and Internet manipulation datasets. We conducted experiments to verify the effectiveness of these two components. As shown in Table 4, removing the co-trained Internet manipulation data resulted in a performance decrease from 4.33 to 3.97. Further removing the pre-trained SVD model and training the video prediction model from scratch on the Calvin dataset led to a substantial performance drop. Notably, removing the video pretraining on Calvin alone also caused a significant decline.
Effectiveness of Video Former. The Video Former module plays a pivotal role in extracting predictive representations from the TVP model. To evaluate its effectiveness, we conduct an ablation study by removing the Video Former and directly connecting the TVP features to the diffusion policy. The results, presented in Table 5, are obtained by evaluating the complete VPP model on a single NVIDIA RTX 4090 GPU. The VPP score decreases from 4.33 to 3.86, while the inference time nearly triples. These findings indicate that the absence of the Video Former leads to a substantial degradation in both accuracy and computational efficiency compared to the full model.
this section cite: []

Section: Effectiveness of Feature Aggregation Module.
Many previous works (Black et al., 2023;Wu et al., 2023a) directly use the final predicted image to learn policies. However, the image from the final layer often contains many irrelevant details that are not beneficial for the task. In contrast, we adopt a feature aggregation mechanism to leverage multiple layers of features within the up-sampling layers. We replace aggregated features with final layer features while keeping the other layers unchanged. This process lead to a decrease in the average task completion length on the Calvin benchmark, from 4.33 to 4.05. More ablations on different layers can be found at Appendix C.2.
this section cite: ['b4']

Section: Real World Experiments
We further verified the Video Prediction Policy on two realworld hardware platforms.  Franka Panda Robot Arm. On the Franka Panda platform, we collected 2,000 trajectories for over 30 tasks in 6 categories: picking, placing, pressing, routing, opening, and closing. We divided the tasks into seen and unseen categories. A task is considered unseen if the operated object is new or the background scene is new.
Xarm with 12-degree Xhand Dexterous Hand. On the dexterous hand platform, we collected 4,000 trajectories over 100+ tasks in 13 categories, including picking, placing, cup-upright, relocating, stacking, passing, pressing, unplugging, opening, closing, pouring, suction, and knocking. We also define a task as unseen if the operated object is new or the background scene is new. Additionally, we included four challenging tool-use tasks, including the use of a spoon, hammer, electrical drill, and pipette for chemistry tasks.
More task details can be found in Appendix A.
Training and Rollout Details. We employ the same textguided video prediction (TVP) model as in our simulated experiments, trained on both internet datasets and collected real-world data. Then a generalist robot policy is learned to solve all tasks in the domain conditioned on instructions.
The hardware platform and visualizations of some selected tasks are shown in Figure 5.
Franka Panda DP Susie GR-1 VPP(ours) Seen Tasks 0.42 0.56 0.52 0.85 Unseen Tasks 0.25 0.46 0.38 0.73 Dexterous Hand DP Susie GR-1 VPP(ours) Seen Tasks 0.28 0.45 0.32 0.75 Unseen Tasks 0.11 0.28 0.15 0.60 Tool-use Tasks 0.05 0.23 0.15 0.68
Table 5. Success rates on real-world tasks. Due to space limit, we only show the average success rate on each category. Detailed success rate can be found at Appendix A
Quantitative Results. Due to the complexity of deploying methods on real-world hardware, we select the strongest baseline models-GR-1, Susie, and the widely-used diffusion policy-as our baselines. For evaluation, we perform 200+ rollouts for Panda arm manipulation tasks and 500+ rollouts for dexterous hand manipulation tasks. The comparisons are in the Table 5, which indicate VPP outperforms all the baselines with a clear margin in both seen tasks, unseen tasks and tool-use tasks.
Generalization Analysis. we take three unseen tasks as case studies: picking up a tennis ball, pouring Coca-Cola, and using a spoon. Notably, none of these objects-tennis ball, Coca-Cola, or spoon-appear in our collected dataset.
As illustrated in Figure 6, the video prediction model forecast reasonable future states even on unseen tasks. Moreover, we observe that the actual execution trajectory closely aligns with the predicted future state. We interpret the generalization mechanism of the VPP model in two key aspects: First, video models can make correct visual predictions even on unseen tasks due to internet-scale pre-training; Second, the low-level policy learns a robust inverse dynamics model that only needs to implicitly track the movement of the robot in the predicted future, without the need to focus on new objects or backgrounds. In this way, the VPP model successfully generalizes to a wide range of unseen tasks.
this section cite: []

Section: Conclusion
We introduce Video Prediction Policy (VPP), a novel approach for learning a generalist robot policy. VPP learns an implicit inverse dynamics model conditioned on predictive representations inside VDMs and yields consistent improvements across both simulated and real-world tasks. As video generation models are more and more powerful these days, we aim to fully unlock the power of video model in building physical intelligence and highlight the potential of video generation models in embodied tasks.
Seen Tasks Diffusion Policy Susie GR-1 VPP Pick 0.38 0.61 0.48 0.83 Pick&Place 0.35 0.55 0.40 0.79 Cup-upright 0.00 0.00 0.00 0.64 Relocate 0.28 0.44 0.16 0.80 Stack 0.00 0.08 0.00 0.64 Pass 0.040 0.00 0.00 0.48 Press 0.68 0.96 0.64 0.96 Unplug 0.00 0.00 0.00 0.52 Drawer 0.40 0.64 0.48 0.72 Average 0.287 0.450 0.319 0.749 Unseen Tasks Diffusion Policy Susie GR-1 VPP Pick 0.12 0.42 0.26 0.75 Pick&Place 0.08 0.32 0.20 0.68 Cup-upright 0.00 0.00 0.00 0.40 Relocate 0.12 0.32 0.12 0.76 Stack 0.00 0.00 0.00 0.56 Pass 0.00 0.00 0.00 0.32 Press 0.44 0.76 0.40 0.88 Unplug 0.00 0.00 0.00 0.20 Drawer 0.28 0.44 0.24 0.56 Average 0.110 0.328 0.159 0.605 Tool-use Tasks Diffusion Policy Susie GR-1 VPP Spoon 0.0 0.4 0.3 0.9 Hammer 0.2 0.2 0.1 0.6 Drill 0.0 0.1 0.2 0.8 Pipette 0.0 0.0 0.0 0.4 Average 0.05 0.23 0.15 0.68
Table 7. Specific success rate at category level. In seen tasks, We evaluate pick and place tasks 100 times and other tasks 25 times respectively. In unseen tasks, we evaluate pick and place tasks 50 times and other tasks 20 times respectively. We evaluate each tool-use task for 10 times.
this section cite: []

Section: References
Ref_id:b0 Title: Flamingo: a visual language model for fewshot learning Year: (2022)
Ref_id:b1 Title: Data2vec: A general framework for self-supervised learning in speech, vision and language Year: (2022)
Ref_id:b2 Title: Bert pre-training of image transformers Year: (2021)
Ref_id:b3 Title: Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation Year: (2024)
Ref_id:b4 Title: Zero-shot robotic manipulation with pretrained image-editing diffusion models Year: (2023)
Ref_id:b5 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b6 Title: Align your latents: Highresolution video synthesis with latent diffusion models Year: (2023)
Ref_id:b7 Title: Rt-1: Robotics transformer for real-world control at scale Year: (2022)
Ref_id:b8 Title: Rt-2: Vision-language-action models transfer web knowledge to robotic control Year: (2023)
Ref_id:b9 Title: Instructpix2pix: Learning to follow image editing instructions Year: (2023)
Ref_id:b10 Title: Unsupervised learning of visual features by contrasting cluster assignments Year: (2020)
Ref_id:b11 Title: Endowing vision-language models with spatial reasoning capabilities Year: (2024)
Ref_id:b12 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b13 Title: Control-a-video: Controllable textto-video generation with diffusion models Year: (2023)
Ref_id:b14 Title: An empirical study of training self-supervised vision transformers Year: (2021)
Ref_id:b15 Title: Image-goal representations are the atomic control units for foundation models in embodied ai Year: (2024)
Ref_id:b16 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b17 Title: Openhelix: A short survey, empirical analysis, and open-source dualsystem vla model for robotic manipulation Year: (2025)
Ref_id:b18 Title: Quar-vla: Vision-languageaction model for quadruped robots Year: (2024)
Ref_id:b19 Title: Humanoid-vla: Towards universal humanoid control with visual integration Year: (2025)
Ref_id:b20 Title: Learning universal policies via text-guided video generation Year: (2024)
Ref_id:b21 Title: Bridge data: Boosting generalization of robotic skills with crossdomain datasets Year: (2021)
Ref_id:b22 Title: The" something something" video database for learning and evaluating visual common sense Year: (2017)
Ref_id:b23 Title: Ego4d: Around the world in 3,000 hours of egocentric video Year: (2022)
Ref_id:b24 Title: Seer: Language instructed video prediction with latent diffusion models Year: (2023)
Ref_id:b25 Title: Prediction with action: Visual policy learning via joint denoising process Year: (2024)
Ref_id:b26 Title: Improving vision-language-action model with online reinforcement learning Year: (2025)
Ref_id:b27 Title: Pre-trained text-to-image diffusion models are versatile representation learners for control Year: (2024)
Ref_id:b28 Title: Learning an actionable discrete diffusion policy via large-scale actionless video pre-training Year: (2024)
Ref_id:b29 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b30 Title: Video diffusion models Year: (2022)
Ref_id:b31 Title: Large-scale pretraining for text-to-video generation via transformers Year: (2022)
Ref_id:b32 Title: General perception with iterative attention Year: (2021)
Ref_id:b33 Title: Language-driven representation learning for robotics Year: (2023)
Ref_id:b34 Title: Vision-language foundation models as effective robot imitators Year: (2023)
Ref_id:b35 Title: Visual-language model with unified view representation for robotic manipulation Year: (2024)
Ref_id:b36 Title: Diffusion hyperfeatures: Searching through time and space for semantic correspondence Year: (2024)
Ref_id:b37 Title: Videofusion: Decomposed diffusion models for high-quality video generation Year: (2023)
Ref_id:b38 Title: Towards universal visual reward and representation via value-implicit pre-training Year: (2022)
Ref_id:b39 Title: Where are we in the search for an artificial visual cortex for embodied intelligence? Year: (2023)
Ref_id:b40 Title: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks Year: (2022)
Ref_id:b41 Title: R3m: A universal visual representation for robot manipulation Year: (2022)
Ref_id:b42 Title: Open x-embodiment: Robotic learning datasets and rt-x models Year: (2023)
Ref_id:b43 Title: The unsurprising effectiveness of pre-trained vision models for control Year: (2022)
Ref_id:b44 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b45 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b46 Title: Real-world robot learning with masked visual pre-training Year: (2023)
Ref_id:b47 Title: Multimodal diffusion transformer: Learning versatile behavior from multimodal goals Year: (2024)
Ref_id:b48 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b49 Title: Hi robot: Open-ended instruction following with hierarchical vision-language-action models Year: (2025)
Ref_id:b50 Title: Accelerating vision-languageaction model integrated with action chunking via parallel decoding Year: (2025)
Ref_id:b51 Title: An open-source generalist robot policy Year: (2024)
Ref_id:b52 Title: Towards accurate generative models of video: A new metric & challenges Year: (2018)
Ref_id:b53 Title: Exploiting implicit dynamics from video diffusion model for effective robot manipulation Year: (2024)
Ref_id:b54 Title: Unleashing large-scale video generative pre-training for visual robot manipulation Year: (2023)
Ref_id:b55 Title: Tune-avideo: One-shot tuning of image diffusion models for text-to-video generation Year: (2023)
Ref_id:b56 Title: Denoising diffusion autoencoders are unified self-supervised learners Year: (2023)
Ref_id:b57 Title: Ovrlv2: A simple state-of-art baseline for imagenav and objectnav Year: (2023)
Ref_id:b58 Title: Offline visual representation learning for embodied navigation Year: (2023)
Ref_id:b59 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2024)
Ref_id:b60 Title: Latent action pretraining from videos Year: (2024)
Ref_id:b61 Title: Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning Year: (2020)
Ref_id:b62 Title: Gevrm: Goal-expressive video generation model for robust visual manipulation Year: (2025)
Ref_id:b63 Title: Vision-language-action model with speech instructions for customized robot manipulation Year: (2025)
