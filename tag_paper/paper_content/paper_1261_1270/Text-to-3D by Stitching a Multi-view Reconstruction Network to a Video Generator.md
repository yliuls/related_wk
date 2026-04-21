Title: TEXT-TO-3D BY STITCHING A MULTI-VIEW RECON-STRUCTION NETWORK TO A VIDEO GENERATOR
Abstract: A golden retriever with a blue bowtie" "An alpinist scaling a dramatic, snow-covered mountain face ... the climber is captured midascent, emphasizing scale and solitude against the immense landscape.""A majestic view of the Matterhorn at sunrise, ... the scene is surrounded by snowy slopes and serene winter atmosphere.""A wooden rocking horse in a child's playroom" (a) Text-to-3DGS (b) Text-to-Pointmap "A gleaming golden trophy with intricate engravings stands too tall to fit inside a small, worn brown leather suitcase... "A bedroom scene features a large bed adorned with white linens. Two lamps sit on nightstands…"Figure 1: Text-to-3D generation with VIST3A. Video models excel at generating latent visual content from text prompts, whereas 3D foundation models shine when it comes to decoding such a latent representation into consistent scene geometry. By stitching a video generator and a 3D reconstruction network together and aligning their latents, we obtain an end-to-end model that produces high-quality Gaussian splats (a) or point maps (b) from text prompts.

Section: INTRODUCTION
With image and video generators now a commodity, text-to-3D models that produce 3D scenes from text prompts have become a new research frontier, with applications in AR/VR, gaming, robotics, and simulation. Early methods for 3D generation adopt Score Distillation Sampling (SDS) (Poole et al., 2023;Tang et al., 2024b;Wang et al., 2023b;Chen et al., 2024b) to optimize a 3D representation, e.g. a NeRF (Mildenhall et al., 2021;Müller et al., 2022) or 3D Gaussian Splats (3DGS, Kerbl et al., 2023) under a pretrained 2D diffusion prior (Rombach et al., 2022). A drawback these methods have in common is the need for slow per-scene optimization. Another line of work uses multi-stage pipelines that first synthesize images and then lift them to 3D with a separate model (Tang et al., 2024a;Xu et al., 2024b;Zhang et al., 2024b) or with per-scene optimization (Gao et al., 2024;Wu et al., 2024a;Yu et al., 2025b); employ progressive warping and refinement (Shriram et al., 2025;Yu et al., 2025a;2024); or sequentially chain multiple generative modules (Yang et al., 2025b;Engstler et al., 2025). The multi-stage design not only increases model complexity and engineering effort, but also makes such models prone to error accumulation (Lin et al., 2025;Meng et al., 2025).
this section cite: ['b68', 'b57', 'b61', 'b37', 'b71', 'b21', 'b80', 'b14', 'b49', 'b56']

Section: 3D Decoder

this section cite: []

Section: 2D Generative Model
Multi-View Latent 3D Representation
this section cite: []

Section: 3D Representation

this section cite: []

Section: Feedforward 3D Model

this section cite: []

Section: Aligned Latent
Model Stitching A recent trend is to directly generate the 3D representation with end-to-end latent diffusion models (LDMs, Schwarz et al., 2025;LAN et al., 2025;Li et al., 2025b;a).
A prominent line of work starts from pretrained 2D image (Esser et al., 2024;Rombach et al., 2022) or video models (Team, 2024;Yang et al., 2025e) and finetunes them to output multi-view 2D latents, reusing the pretrained priors (Szymanowicz et al., 2025;Liang et al., 2025;Schwarz et al., 2025;Lin et al., 2025;Yang et al., 2025c;Go et al., 2025a;b). Subsequently, a VAE-style decoder is trained to decode those latents into the desired 3D representation, see Fig. 2. The LDM-like design unifies 2D generation and multi-view reconstruction within the latent space and enables efficient 3D scene generation with a compact, wellamortized decoder.
Still, two key limitations remain. First, we argue that the Achilles heel of existing 2D-to-3D diffusion models is the decoder. By simply repurposing the 2D VAE to produce 3D outputs, the network must learn 3D reconstruction more or less from scratch, which requires extensive training and large datasets that are hard to obtain (Yang et al., 2025c;Szymanowicz et al., 2025;Go et al., 2025b). This practice becomes increasingly problematic as new, better 3D foundation models emerge (Wang et al., 2026;2025b;2024;Zhang et al., 2025) and the ad-hoc trained decoders of text-to-3D models fall further behind the state of the art in 3D vision.
Second, the prevalent training scheme tends to suffer from weak alignment between the generative model and the VAE decoder. Typically, the former is finetuned on multi-view datasets with a generative objective like a diffusion loss (Song et al., 2021;Sohl-Dickstein et al., 2015;Ho et al., 2020) or flow matching (Liu et al., 2023;Lipman et al., 2023;Albergo & Vanden-Eijnden, 2023), which only indirectly promotes 3D-consistent latents. Moreover, the separate training may cause the latents, even if 3D-consistent, to be out of domain from the perspective of the decoder. To mitigate that misalignment, it has been proposed to add rendering losses that promote decodable latents (Lin et al., 2025). However, the resulting objective is based on single-step sampling and does not sufficiently take into account the denoising trajectory, leading to weak alignment at inference.
We introduce VIST3A: VIdeo VAE STitching and 3D Alignment. The proposed method consists of two complementary components that address the above-mentioned limitations, see Fig. 2. First, we resort to the concept of model stitching (Pan et al., 2023;Lenc & Vedaldi, 2015;Bansal et al., 2021;Csiszárik et al., 2021;Yang et al., 2022) to leverage powerful, pretrained feedforward 3D models for decoding, rather than start from scratch. The idea is to attach the relevant part of a 3D reconstruction network as a "decoder" to the latent space of a video VAE. For this to work, there needs to be one or more layers in the 3D model whose activations are similar (up to a linear transformation) to those in the VAE's latent space, despite their independent pretraining. Perhaps surprisingly, this turns out to be the case. For the 3D model, we identify the layer with the most linear relation to the LDM latents, slice the network before that layer, and retain the downstream portion as 3D decoder. After fitting a single, linear stitching layer (in closed form), the VAE latent space already matches the expected input of the 3D decoder well, such that subsequent fine-tuning will be minor and not degrade the respective generative and 3D reasoning capabilities of the two base models.
Second, we further improve alignment between the generative model and the stitched decoder through direct reward finetuning (Clark et al., 2024;Xu et al., 2023;Prabhudesai et al., 2024;Wu et al., 2024c;Shen et al., 2025). In that technique, commonly used to align diffusion models with human preferences, reward signals are defined based on the "goodness" of the VAE output -in our setting, the visual quality and 3D consistency of the decoded 3D representations. Maximizing these rewards encourages the LDM to produce latents that are 3D-consistent and lie within the decoder's input domain, ensuring high-quality outputs. Importantly, our alignment compares video model outputs and images rendered from the generated 3D scenes, hence it does not require labels.
In our experiments, we show that the proposed stitching scheme is applicable across a range of video generative models and also across several different feedforward 3D models. VIST3A's direct 3D decoding consistently outperforms prior text-to-3DGS methods, and additionally offers high-quality pointmap generation from text prompts.
this section cite: ['b76', 'b40', 'b15', 'b71', 'b84', 'b48', 'b76', 'b49', 'b84', 'b98', 'b122', 'b82', 'b81', 'b29', 'b54', 'b51', 'b0', 'b49', 'b65', 'b42', 'b1', 'b11', 'b111', 'b10', 'b105', 'b69', 'b77']

Section: RELATED WORKS

this section cite: []

Section: 3D generation.
Recent works have explored various 3D representations for generative modelling, including point clouds (Mo et al., 2023;Nichol et al., 2022;Vahdat et al., 2022), meshes (Xu et al., 2024a), voxel grids (Sanghi et al., 2023), NeRFs (Chen et al., 2023;Müller et al., 2022;Mildenhall et al., 2021), and 3DGS (Henderson et al., 2024;Zhang et al., 2024a;Kerbl et al., 2023). Score distillation using 2D diffusion models is time-consuming, as it requires per-scene test time optimization (Wang et al., 2023a;Shi et al., 2024;Wang et al., 2023b), while multi-stage pipelines (Yu et al., 2025b;Liu et al., 2024;Zheng et al., 2025) lack robustness and create significant engineering overhead. For further details on multi-stage pipelines, please refer to Appendix A.
More recently, the field has shifted towards end-to-end latent diffusion models, where the generator operates in the latent space of a VAE, and the latter directly decodes the resulting latents to 3D outputs. Many of these works focus on object-centric asset generation (Wu et al., 2024b;Zhao et al., 2023;Lin et al., 2025) and train the LDM on curated datasets such as Objaverse (Deitke et al., 2023), with single objects or bounded scenes, and controlled camera paths. Consequently, they are unable to handle real-world challenges like strongly varying scene scale, variable lighting, etc.
To tackle such situations, recent methods (Szymanowicz et al., 2025;Liang et al., 2025;Schwarz et al., 2025;Lin et al., 2025;Yang et al., 2025c;Go et al., 2025a;b) repurpose the comprehensive knowledge of the visual world that is implicit in 2D image generators. The general strategy is to finetune a pretrained 2D model on multi-view data, by using generative losses to enforce crossview consistency. In many cases training is further supported by additional 3D cues like camera poses (Li et al., 2024b;Go et al., 2025b), depthmaps (Go et al., 2025a;Yang et al., 2025c), or pointmaps (Szymanowicz et al., 2025). The resulting multi-view latents are decoded to 3D scenes with a dedicated VAE-style decoder, meaning that 3D reasoning capabilities must be rebuilt from scratch, and that they are only weakly aligned with the generator output -limitations which we address with VIST3A.
this section cite: ['b60', 'b63', 'b90', 'b72', 'b6', 'b61', 'b57', 'b26', 'b37', 'b78', 'b52', 'b124', 'b123', 'b49', 'b13', 'b84', 'b48', 'b76', 'b49', 'b84']

Section: Learned 3D reconstruction.
A notable trend in 3D computer vision is the trend to move away from multi-stage pipelines and iterative optimization towards end-to-end, feedforward 3D modelling. Classical reconstruction pipelines based on SfM (Hartley & Zisserman, 2003;Schönberger & Frahm, 2016) and MVS (Furukawa et al., 2015;Schönberger et al., 2016) require incremental, iterative optimization, whereas recent advances like DUSt3R (Wang et al., 2024) and MASt3R (Leroy et al., 2024) directly predict 3D point maps in one forward pass. Several follow-up works have further reduced test-time optimization (Tang et al., 2025;Wang et al., 2025c;Yang et al., 2025a). Likewise, 3D Gaussian splatting has evolved from per-scene optimization to feedforward prediction (Charatan et al., 2024;Chen et al., 2024a;Ye et al., 2025). Once more, data scaling has been a critical factor (Wang et al., 2025b;2026). Consequently, replicating the 3D capabilities of recent feedforward models as part of VAE training would be difficult and costly. VIST3A offers a solution by reusing, rather than rebuilding, models like AnySplat (Jiang et al., 2025), VGGT (Wang et al., 2025b), or MVDUSt3R (Tang et al., 2025).
this section cite: ['b24', 'b20', 'b96', 'b43', 'b87', 'b4', 'b115', 'b35', 'b87']

Section: 1) Search for stitching layer
Find best transferable layer 𝑘 ⋆ through:
2) Stitch and finetune
Unlabeled Video Stitched Parts 𝐹 𝑘 ⋆ +1:𝑙 Feedforward 3D Model 𝐹 1:𝑙 Feedforward 3D Model 𝐹 1:𝑙 ℓ 1 -loss Direct reward finetuning for alignment Stitched Parts 𝐹 𝑘 ⋆ +1:𝑙 "Prompt" 3D representation Rendered images 𝑧 1 𝑧 0 Decoded images (3) Multi-view image quality (1) 3D representation quality (2) 3D consistency Reward Backprop : Latent generative model : Gradient backpropagation
this section cite: []

Section: Model stitching.
Recomposing the heads and tails of two different networks was initially studied as a way to assess the equivariance of neural representations (Lenc & Vedaldi, 2015), and as an experimental tool to compare two different representations (Csiszárik et al., 2021;Bansal et al., 2021). To ensure invariance against trivial affine transformations, the head of some trained network A is normally attached to the tail of another network B via a linear, trainable stitching layer. Besides revealing similarities between networks that common metrics like CKA (Kornblith et al., 2019) would miss, it was also found that different architectures that were trained on the same data can often be stitched into a new, hybrid model with minimal degradation (Bansal et al., 2021). This has opened the door for practical uses of stitching, e.g. DeRy (Yang et al., 2022) for resourceconstrained reassembly of pretrained models and SN-Net (Pan et al., 2023) to build networks with varying scales. Going one step further, we demonstrate that strong 3D VAEsfoot_0 can be obtained by stitching a foundational 3D model to the latent space of a video VAE as its decoder, even if they were trained independently on different data.
this section cite: ['b42', 'b11', 'b1', 'b39', 'b1', 'b111', 'b65']

Section: METHODOLOGY
VIST3A consists of two key components, see Fig. 3: (1) model stitching to optimally attach (part of) a foundational 3D model as the decoder for the latent, and (2) direct reward finetuning to optimize the alignment of the (latent) generative model with that new decoder.
this section cite: []

Section: MODEL STITCHING FOR 3D VAE CONSTRUCTION
Our objective is to build a 3D VAE by seamlessly combining the encoder of a video LDM and a feedforward 3D reconstruction model. Note that, for stitching purposes, one can skip the denoising loop, since feeding images into the encoder already gives clean latents. Let E denote the encoder and D the decoder of the VAE, and let
F 1:l (x) = f l • • • • • f 1 (x) =
y be the feedforward 3D network that maps a set of views x to a 3D output y, with l the total number of layers in that feedforward model. As shown in Fig. 3, we cut the feedforward model at layer k * and stitch the downstream part
F k ⋆ +1:l = f l • • • • • f k ⋆ +1
to the output layer of the encoder E, with the help of a linear stitching layer S. In doing so, we obtain a new 3D VAE M stitched that outputs the same representation ŷ as the original 3D model:
M stitched = F k ⋆ +1:l • S • E(x) = ŷ, D stitched = F k ⋆ +1:l • S(1)
The front portion F 1:k ⋆ of the 3D model is discarded -but if the clean encoder latents, after the affine warping S, are (almost) the same as the activations f k ⋆ , then the back portion will still produce the same output, ŷ ≈ y. In other words, the stitched VAE M stitched is an approximation of the original 3D model F . It retains much of the ability to map multi-view images to a 3D reconstruction and only requires a little fine-tuning to restore that ability.
this section cite: []

Section: Step 1: Finding the stitching index and initialization.
To identify the layer k * in the 3D model whose representation is most compatible with the VAE latent, we first push a set of N samples through the encoder E to obtain their latents B ∈ R N ×D E . Here, D E denotes the dimensionality of the encoder latent space, and D k F denotes the dimensionality of the feature (activation) at layer k. Then, we scan over candidate layers k ∈ {1, ..., l -1} of the 3D model and, for each layer in turn, extract the activations A k ∈ R N ×D k F and fit the linear stitching layer S * k ∈ R D E ×D k F that best recovers the activations of the 3D model at layer k, by solving a least-squares problem:
S * k = arg min S k ∥BS k -A k ∥ 2 F = B ⊤ B -1 B ⊤ A k .(2)
Finally, we select the stitching layer k ⋆ that leads to the smallest (mean squared) error, k ⋆ = arg min k ∥BS * k -A k ∥ 2 F , and assemble the 3D VAE by concatenating E, S * k ⋆ and F k ⋆ +1:l . Empirically, we find that many combinations of foundational VAEs and 3D feedforward models can be stitched in this manner, with minimal performance loss.
this section cite: []

Section: Step 2: Stitched decoder finetuning.
To further reduce the remaining discrepancies between the newly assembled 3D VAE and the original 3D model, we finetune S and F k ⋆ +1:l to reproduce the predictions of the original 3D model y, using them as pseudo-targets. Practical feedforward models produce multiple outputs (e.g., point maps, depth, poses), so we optimize a weighted sum of ℓ 1 losses for all of them. Note that the fine-tuning step is self-supervised and does not require labels. In our implementation, we restrict the stitching layer to a 3D convolution and employ LoRA (Hu et al., 2022) for updating F k ⋆ +1:l , to prevent large deviations from the pretrained weights. For further details, see Appendix B.1.
this section cite: ['b31']

Section: ALIGNMENT VIA DIRECT REWARD FINETUNING
So far, we have assembled a 3D VAE with a strong, pretrained 3D decoder. However, during text-to-3D inference, the latents are not obtained from the encoder but generated from noise by the denoising loop conditioned on the text prompt. Therefore, we must also align the generative model itself with the 3D decoder, such that it produces decodable latents.
Previous work finetunes the generative network by minimizing generative losses over some multiview dataset. Unfortunately, that strategy does not ensure 3D-consistent latents. Even if it did, the finetuning bypasses the decoder, hence there is no guarantee that the generated latents fall within the distribution expected by the 3D VAE and can be decoded to meaningful outputs.
To address the disconnect between the denosing loop and the 3D VAE, we adopt direct reward finetuning to align the two. In other words, we extend conventional, generative multi-view finetuning with reward maximization. The conventional generative loss L gen uses paired data, i.e., multi-view images and corresponding prompts. In contrast, the proposed reward term r(•, c) relies only on the text prompt and requires no ground-truth images. Our total loss is defined as
L total = L gen -r z 0 (θ, c, z T ), c ,(3)
where θ are the parameters of the video generative model, c represents the text prompt, z T is the initial noise, and z 0 (θ, c, z T ) is the final latent produced by the denoising loop.
Reward. The proposed reward function consists of three components that ensure high-quality and 3D-consistent generation. (1) Multi-view Image Quality: As we keep the encoder frozen, the generated latents can be decoded by the original video decoder D to obtain multi-view images. We evaluate these images against the input prompt using CLIP-based (Fang et al., 2024) and HPSv2 human preference scores (Wu et al., 2023) to promote prompt adherence and visual quality, similar to DanceGRPO (Xue et al., 2025).
(2) 3D Representation Quality: To encourage high-quality 3D outputs after decoding with D stitched , we render the generated 3D scenes (pointmaps and/or 3DGS) back into 2D views and apply the same (CLIP + HPSv2) metrics to them as above.
(3) 3D Consistency: To enforce 3D consistency, we render the 3D representation from the same viewpoints as the multi-view images reconstructed by the video decoder D, using the camera poses predicted by the feedforward 3D model. We then compute a combination of ℓ 1 -loss and LPIPS (Zhang et al., 2018) for each pair of decoded and rendered images belonging to the same viewpoint. The final (negative) reward is a weighted sum of these three losses. For further details, see Appendix B.2.
this section cite: ['b17', 'b102', 'b107', 'b121']

Section: Alignment algorithm.
To optimize the generative model according to the reward function above, we employ direct reward finetuning (Clark et al., 2024;Xu et al., 2023;Prabhudesai et al., 2024;Wu et al., 2024c;Shen et al., 2025). I.e., the model generates samples by unfolding the full denoising path, and the rewards computed from these samples are then backpropagated through the denoising chain. While the algorithm benefits from gradient-based feedback, it can also suffer from exploding gradient norms. To stabilize the optimization, we generalize the idea of DRTune (Wu et al., 2024c): gradients are detached from the inputs to the generative model, but retained during the update step to the next denoising state. In this way, reward propagation remains stable even at early denoising steps. Furthermore, we modify the optimizer for better computational efficiency by (i) randomized sampling, using fewer timesteps than during inference, and (ii) randomizing the subset of denoising steps where gradients are backpropagated, such that the model learns from diverse denoising trajectories. For further details, see Appendix B.2.
In summary, we perform joint, end-to-end alignment of the VAE and the generative model, unlike conventional multi-view fine-tuning that keeps them separate. Reward tuning ensures that, throughout the iterative denoising process, the generative model remains aligned with our 3D VAE and generates latents that suit the stitched decoder.
this section cite: ['b10', 'b105', 'b69', 'b77']

Section: EXPERIMENTAL RESULTS
In what follows, we demonstrate VIST3A's text-to-3D generation performance. The main findings are that VIST3A clearly outperforms existing feedforward text-to-3DGS approaches and also offers high-quality text-to-pointmap generation. Moreover, we experimentally analyze our two core components, self-supervised model stitching and alignment finetuning.
this section cite: []

Section: EXPERIMENTAL SETUPS
We provide a high-level overview of the experimental setup. A complete description of evaluation protocols and training details can be found in Appendix C.
this section cite: []

Section: Target 3D models.
We target last-generation foundational 3D vision models that have been trained on large-scale datasets, have demonstrated generality and reliable performance across diverse domains, and require only images as input. For our experiments, we select three representative state-of-the-art models: (1) MVDUSt3R (Tang et al., 2025) predicts pointmaps and Gaussian splats, (2) VGGT (Wang et al., 2025b) predicts pointmaps, depth maps and camera poses, and (3) AnySplat (Jiang et al., 2025) predicts Gaussian splats and camera poses.
this section cite: ['b87', 'b35']

Section: Target video generators.
Our primary video model is Wan 2.1 T2V large (Wan et al., 2025), a state-of-the-art text-to-video generator. To demonstrate the generality of VIST3A across different architectures, we additionally use several other latent video models, including CogVideoX (Yang et al., 2025e), SVD (Blattmann et al., 2023), and HunyuanVideo (Kong et al., 2024).
Training data. We finetune stitched VAEs on DL3DV-10K (Ling et al., 2024) and ScanNet (Dai et al., 2017), without 3D labels. To align the video generator in latent space, we utilize DL3DV-10K to compute the generative loss, with prompts from the HPSv2 training set (Wu et al., 2023).
this section cite: ['b91', 'b3', 'b38', 'b50', 'b12', 'b102']

Section: MAIN RESULTS: 3D GENERATION
Stitching Wan to the 3D models listed in Section 4.1 yields two types of generative models: (i) Textto-3DGS when using AnySplat or MVDUSt3R as decoder; and (ii) Text-to-Pointmap when using VGGT or MVDUSt3R. Both variants are evaluated in the following.
Baselines. Important baselines for text-to-3DGS are SplatFlow (Go et al., 2025a), Director3D (Li et al., 2024b), Prometheus3D (Yang et al., 2025c), and VideoRFSplat (Go et al., 2025b). Additionally, we include Matrix3D-omni (Yang et al., 2025d), to our knowledge, the only other model that unifies generation and reconstruction in latent space.
this section cite: []

Section: Evaluation protocol.
We evaluate text-to-3DGS models on three benchmarks: T3bench (He et al., 2023) for object-centric generation, SceneBench (Yang et al., 2025c) for scene-level synthesis, and DPG-bench (Hu et al., 2024) to assess adherence to long, detailed prompts. On T3bench and SceneBench, we render images and compute Imaging Quality and Aesthetic Quality scores as defined by VBench (Huang et al., 2024) to assess visual fidelity, CLIP score (Hessel et al., 2021) for text-prompt alignment, and Alignment, Coherence, and Style scores according to Wang et al. (2025d) as comprehensive quality metrics. We prefer to avoid traditional no-reference metrics like NIQE (Mittal et al., 2012b) and BRISQUE (Mittal et al., 2012a) that have sometimes been used in the context of 3D generation, but lack a meaningful connection to the conditional generation task (e.g., they can be gambled by always returning the same sharp and colorful, high-scoring image, independent of the prompt). For DPG-bench, we follow the suggested protocol (Hu et al., 2024), but upgrade from the originally proposed language models to the more capable, UnifiedReward LLM (based on Qwen 7B). Text-to-pointmap models are evaluated qualitatively, as no established benchmarks or baselines exist.
Quantitative Results. Tables 1 and 2 show the results for the three text-to-3DGS benchmarks. Notably, both tested VIST3A variants exhibit superior performance across all datasets and evaluation metrics. On T3bench, both Wan+AnySplat and Wan+MVDUSt3R consistently outperform all baselines, with particularly large margins in Imaging Quality and Coherence score. For the more complex scene-level synthesis of SceneBench, our models reach Imaging Quality scores >60 and Coherence scores >3.8, again a marked improvement over prior art. On DPG-bench, our models greatly outperform the baselines, mostly scoring >75 (often even ≈85), values that previously seemed out of reach. The consistent gains on T3bench, SceneBench, and DPG-bench demonstrate the effectiveness and versatility of our stitching approach for text-based 3D scene generation. We attribute these results to the power of foundational contemporary video and 3D models, which our stitching and fine-tuning scheme unlocks for the purpose of 3D generative modeling.
this section cite: ['b25', 'b32', 'b33', 'b27', 'b32']

Section: Human evaluation.
To further validate the effectiveness of VIST3A, we conduct a user study where we compare it against four other methods: Director3D, SplatFlow, Prometheus3D, and Vide-oRFSplat. A total of 28 participants evaluated 14 randomly selected samples drawn from T3Bench, SceneBench, and DPG-Bench, ranking each method according to two criteria: (1) Text Alignment and (2) Visual Quality of videos rendered from the generated 3DGS. As shown in Table 4, VIST3A achieves the best performance (lowest average rank) on both criteria. Notably, participants rank VIST3A as the top method in >68% of cases for text alignment and >87% for visual quality, underscoring its superiority in generating high-fidelity, semantically consistent 3D scenes.
Qualitative Results. Figure 4 qualitatively compares VIST3A (Wan+AnySplat) to several baselines. In line with the quantitative results, VIST3A produces superior, visually compelling, and geometrically coherent renderings that closely follow the input prompts; whereas previous methods tend to exhibit artifacts, structural distortions, and poor text alignment. Further qualitative results, including Wan+MVDUst3R and Wan+AnySplat variants of VIST3A, as well as text-to-pointmap  examples, can be found in Appendix E. Interestingly, we find that, even without specific training on very long image sequences, VIST3A can generate coherent large-scale scenes by extending the number of frames generated by the LDM. This demonstrates that our framework preserves the ability of video generator and the 3D decoder to handle long sequences. Examples are depicted in Fig. 16. Evaluation protocol. For 3DGS models, we evaluate novel-view synthesis on RealEstate10K (Zhou et al., 2018), with 8 source and 4 target images. For 3D reconstruction models, we follow Pi3 (Wang et al., 2026) and assess pointmap quality on 7Scenes (Shotton et al., 2013) and ETH3D (Schöps et al., 2017), and camera pose estimation on RealEstate10K and ScanNet (Dai et al., 2017). Specifically, Accuracy (Acc.), Completion (Comp.), and Normal Consistency (N.C.) are used for pointmap estimation, while camera pose estimation is evaluated with Relative Rotation Accuracy (RRA) and Relative Translation Accuracy (RTA) at 5°and their AUC up to 30°.
this section cite: ['b125', 'b98', 'b79', 'b75', 'b12']

Section: Novel view synthesis.
Table 3 reports results on RealEstate10K. Stitching AnySplat onto any video model always improves over using AnySplat alone. We attribute the gains to the richer appearance representation of video VAE latents. The experiment is consistent with the results of Wonderland (Liang et al., 2025), where operating in latent space rather than RGB space also benefits 3DGS. Moreover, our stitched VAEs outperform the earlier VAE-based approaches. Remarkably, we surpass Prometheus3D and VideoRFSplat despite their use of camera poses and large-scale training data, showing that stitching high-performance 3D models is indeed an effective strategy to obtain powerful 3D VAEs.
this section cite: ['b48']

Section: Pointmap reconstruction results.
Table 5 shows that stitching preserves the accuracy and completeness of the original 3D foundation models: both pointmap quality and camera pose accuracy barely change when using video encoder latents as input. The results confirm that stitching achieves its goal, to take advantage of the pretrained models' 3D reconstruction capabilities and repurpose them for generative modeling, without relying on large training datasets or labels.
this section cite: []

Section: ABLATIONS
Effectiveness of MSE for finding stitching layer (Sec 3.1). We pick the best layer for stitching according to a fairly simple criterion, namely the one that best supports a linear transfer of the encoder latents. To analyze the impact of this design, we train stitched decoders for the combination (Wan+VGGT) while varying the stitching index. In Fig. 5, we see that layers with lower stitching residual indeed yield better pointmaps, supporting the MSE of the linear stitching layer as our selection criterion.
This empirical trend is also consistent with existing theory: Theorem 1 in Insulla et al. (2025) shows that the stitching risk of the hybrid network, obtained by connecting the source model's early layers f 1 with the target model's latter layers g 2 via a linear map S 1,2 , is upper-bounded by the MSE at the stitching layer,
E ∥g 2 (S 1,2 f 1 )(x) -g 2 (f 2 )(x)∥ 2 ≤ κ 2 2 E ∥S 1,2 f 1 (x) -f 2 (x)∥ 2 ,(4)
where κ 2 is the Lipschitz constant of g 2 . Thus, an MSE is related to the upper bound on the stitching error, supporting our use of MSE. Furthermore, motivated by the observation of Insulla et al. (2025) that the right-hand side of Eq. 4 takes a similar form of kernel alignment, we investigate whether CKA (Kornblith et al., 2019) can track the trend of final performance. Figure 6 reports the CKA between the latent representation of the Wan VAE and the representations at different layer indices of VGGT (larger values indicate greater similarity). As shown, CKA captures the overall degradation in performance as the layer index increases; however, it is less precise than MSE in identifying the best layer, failing to capture that the best performance is achieved at layer 2. These results suggest that, in our setting, MSE is a more reliable indicator of transferability than CKA.
Impact of direct reward finetuning (Sec 3.2). As shown in Appendix D.1, direct reward finetuning is more effective than a pretrained video model on its own, as well as that same model finetuned on multi-view data, with each reward component contributing to the overall performance.
Benefits of integrated vs. sequential 3D generation. In Appendix D.2, we observe that an integrated approach is more robust to noise in the latent space, which suggests it may lead to more consistent 3D reconstruction from noise in the generation process.
Additional results in Appendix E demonstrate that VIST3A inherits prompt-based camera control from the video backbone (e.g., responding to instructions like "aerial droneshot"), that the stitching analysis generalizes across multiple VAE architectures, and that our finetuning does not degradeand even slightly improves -video generation quality as measured by VBench.
this section cite: ['b34', 'b34', 'b39']

Section: Additional results.
In Appendix E, we further show that VIST3A inherits prompt-based camera control from the video backbone, and the stitching layer analysis generalizes across multiple VAE architectures, and our finetuning does not degrade video generation quality on VBench.
this section cite: []

Section: CONCLUSION
We have presented VIST3A, a framework for training latent diffusion models that generate 3D content from text prompts. Our key idea is to employ model stitching as a way to integrate the generative abilities of modern video models with the 3D understanding of recent feedforward 3D models. We found that this strategy indeed leads to high-quality 3D VAEs, while not requiring labeled data or massive training runs. To then align a latent-space video generator with the stitched 3D decoder it feeds into, we design a reward-based finetuning strategy. Together, these two measures yield a family of text-to-3D models with high-quality, geometrically consistent 3D outputs. In passing, they extend 3D generation to other outputs of foundational 3D models, such as pointmaps and depthmaps. More broadly, we see great potential for model stitching as a general tool to combine two or more foundational neural networks, including latent generative models, into powerful end-to-end solutions.
this section cite: []

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2023)
Ref_id:b1 Title: Revisiting model stitching to compare neural representations Year: (2021)
Ref_id:b2 Title: Training diffusion models with reinforcement learning Year: (2024)
Ref_id:b3 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b4 Title: pixelSplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction Year: (2024)
Ref_id:b5 Title: Aligning visual foundation encoders to tokenizers for diffusion models Year: (2026)
Ref_id:b6 Title: Singlestage diffusion NeRF: A unified approach to 3d generation and reconstruction Year: (2023)
Ref_id:b7 Title: FlexWorld: Progressively expanding 3d scenes for flexible-view synthesis Year: (2025)
Ref_id:b8 Title: MVSplat: Efficient 3d gaussian splatting from sparse multi-view images Year: (2024)
Ref_id:b9 Title: Text-to-3d using gaussian splatting Year: (2024)
Ref_id:b10 Title: Directly fine-tuning diffusion models on differentiable rewards Year: (2024)
Ref_id:b11 Title: Similarity and matching of neural network representations Year: (2021)
Ref_id:b12 Title: ScanNet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017)
Ref_id:b13 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b14 Title: Syncity: Training-free generation of 3d worlds Year: (2025)
Ref_id:b15 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b16 Title: Reinforcement learning for finetuning text-to-image diffusion models Year: (2023)
Ref_id:b17 Title: Data filtering networks Year: (2024)
Ref_id:b18 Title: WonderVerse: Extendable 3d scene generation with video generative models Year: (2025)
Ref_id:b19 Title: SceneScape: Text-driven consistent scene generation Year: (2023)
Ref_id:b20 Title: Multi-view stereo: A tutorial Year: (2015)
Ref_id:b21 Title: CAT3D: Create anything in 3d with multi-view diffusion models Year: (2024)
Ref_id:b22 Title: Splat-Flow: Multi-view rectified flow model for 3d gaussian splatting synthesis Year: (2025)
Ref_id:b23 Title: Videorfsplat: Direct scene-level text-to-3d gaussian splatting generation with flexible pose and multi-view joint modeling Year: (2025)
Ref_id:b24 Title: Multiple view geometry in computer vision Year: (2003)
Ref_id:b25 Title:  Year: (2023)
Ref_id:b26 Title: Sampling 3d gaussian scenes in seconds with latent diffusion models Year: (2024)
Ref_id:b27 Title: CLIPScore: A reference-free evaluation metric for image captioning Year: (2021)
Ref_id:b28 Title: Classifier-free diffusion guidance Year: (2021)
Ref_id:b29 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b30 Title: LRM: Large reconstruction model for single image to 3d Year: (2024)
Ref_id:b31 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b32 Title: ELLA: Equip diffusion models with LLM for enhanced semantic alignment Year: (2024)
Ref_id:b33 Title: VBench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b34 Title: Towards a learning theory of representation alignment Year: (2025)
Ref_id:b35 Title: Anysplat: Feed-forward 3d gaussian splatting from unconstrained views Year: (2025)
Ref_id:b36 Title: Musiq: Multi-scale image quality transformer Year: (2021)
Ref_id:b37 Title: 3d Gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b38 Title: HunyuanVideo: A systematic framework for large video generative models Year: (2024)
Ref_id:b39 Title: Similarity of neural network representations revisited Year: (2019)
Ref_id:b40 Title: Gaussiananything: Interactive point cloud flow matching for 3d generation Year: (2025)
Ref_id:b41 Title: Aligning text-to-image models using human feedback Year: (2023)
Ref_id:b42 Title: Understanding image representations by measuring their equivariance and equivalence Year: (2015)
Ref_id:b43 Title: Grounding image matching in 3d with MASt3R Year: (2024)
Ref_id:b44 Title: Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model Year: ()
Ref_id:b45 Title: CraftsMan3D: High-fidelity mesh generation with 3d native diffusion and interactive geometry refiner Year: (2025)
Ref_id:b46 Title: Director3D: Real-world camera trajectory and 3d scene generation from text Year: (2024)
Ref_id:b47 Title: Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models Year: (2025)
Ref_id:b48 Title: Wonderland: Navigating 3d scenes from a single image Year: (2025)
Ref_id:b49 Title: DiffSplat: Repurposing image diffusion models for scalable gaussian splat generation Year: (2025)
Ref_id:b50 Title: DL3DV-10K: A large-scale scene dataset for deep learning-based 3d vision Year: (2024)
Ref_id:b51 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b52 Title: ReconX: Reconstruct any scene from sparse views with video diffusion model Year: (2024)
Ref_id:b53 Title: Flow-GRPO: Training flow matching models via online RL Year: (2025)
Ref_id:b54 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b55 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b56 Title: Zero-1-tog: Taming pretrained 2d diffusion model for direct 3d generation Year: (2025)
Ref_id:b57 Title: NeRF: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b58 Title: No-reference image quality assessment in the spatial domain Year: (2012)
Ref_id:b59 Title: Making a "completely blind" image quality analyzer Year: (2012)
Ref_id:b60 Title: DiT-3D: Exploring plain diffusion transformers for 3d shape generation Year: (2023)
Ref_id:b61 Title: Instant neural graphics primitives with a multiresolution hash encoding Year: (2022)
Ref_id:b62 Title: Generating interactive 3d world in 0.72 seconds Year: (2025)
Ref_id:b63 Title: Point-E: A system for generating 3d point clouds from complex prompts Year: (2022)
Ref_id:b64 Title: LSD-3D: Large-scale 3d driving scene generation with geometry grounding Year: (2025)
Ref_id:b65 Title: Stitchable neural networks Year: (2023)
Ref_id:b66 Title: Steerx: Creating any camera-free 3d and 4d scenes with geometric steering Year: (2025)
Ref_id:b67 Title: Advantage-weighted regression: Simple and scalable off-policy reinforcement learning Year: (2019)
Ref_id:b68 Title: DreamFusion: Text-to-3d using 2d diffusion Year: (2023)
Ref_id:b69 Title: Video diffusion alignment via reward gradients Year: (2024)
Ref_id:b70 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b71 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b72 Title: CLIP-Sculptor: Zero-shot generation of high-fidelity and diverse shapes from natural language Year: (2023)
Ref_id:b73 Title: Structure-from-motion revisited Year: (2016)
Ref_id:b74 Title: Pixelwise view selection for unstructured multi-view stereo Year: (2016)
Ref_id:b75 Title: A multi-view stereo benchmark with high-resolution images and multi-camera videos Year: (2017)
Ref_id:b76 Title: Generative gaussian splatting: Generating 3d scenes with video diffusion priors Year: (2025)
Ref_id:b77 Title: Directly aligning the full diffusion trajectory with finegrained human preference Year: (2025)
Ref_id:b78 Title: MVDream: Multi-view diffusion for 3d generation Year: (2024)
Ref_id:b79 Title: Scene coordinate regression forests for camera relocalization in RGB-D images Year: (2013)
Ref_id:b80 Title: RealmDreamer: Text-driven 3d scene generation with inpainting and depth diffusion Year: (2025)
Ref_id:b81 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b82 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b83 Title: Dimensionx: Create any 3d and 4d scenes from a single image with decoupled video diffusion Year: (2025)
Ref_id:b84 Title: Bolt3d: Generating 3d scenes in seconds Year: (2025)
Ref_id:b85 Title: LGM: Large multi-view gaussian model for high-resolution 3d content creation Year: (2024)
Ref_id:b86 Title: DreamGaussian: Generative Gaussian splatting for efficient 3d content creation Year: (2024)
Ref_id:b87 Title: MV-DUSt3R+: Single-stage scene reconstruction from sparse views in 2 seconds Year: (2025)
Ref_id:b88 Title:  Year: (2024)
Ref_id:b89 Title: Qwen2.5-vl Year: (2025-01)
Ref_id:b90 Title: LION: Latent point diffusion models for 3d shape generation Year: (2022)
Ref_id:b91 Title: Wan: Open and advanced large-scale video generative models Year: (2025)
Ref_id:b92 Title: Vistadream: Sampling multiview consistent images for single-view scene reconstruction Year: (2025)
Ref_id:b93 Title: Score Jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation Year: (2023)
Ref_id:b94 Title: VGGT: Visual geometry grounded transformer Year: (2025)
Ref_id:b95 Title: Continuous 3d perception model with persistent state Year: (2025)
Ref_id:b96 Title: DUSt3R: Geometric 3d vision made easy Year: (2024)
Ref_id:b97 Title: Unified reward model for multimodal understanding and generation Year: (2025)
Ref_id:b98 Title: $\piˆ3$: Permutation-equivariant visual geometry learning Year: (2026)
Ref_id:b99 Title: Prolific-Dreamer: high-fidelity and diverse text-to-3d generation with variational score distillation Year: (2023)
Ref_id:b100 Title: ReconFusion: 3d reconstruction with diffusion priors Year: (2024)
Ref_id:b101 Title: Direct3D: Scalable image-to-3d generation via 3d latent diffusion transformer Year: (2024)
Ref_id:b102 Title: Human preference score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis Year: (2023)
Ref_id:b103 Title: Deep reward supervisions for tuning text-to-image diffusion models Year: (2024)
Ref_id:b104 Title: InstantMesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models Year: (2024)
Ref_id:b105 Title: ImageReward: Learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b106 Title: GRM: Large Gaussian reconstruction model for efficient 3d reconstruction and generation Year: (2024)
Ref_id:b107 Title: DanceGRPO: Unleashing GRPO on visual generation Year: (2025)
Ref_id:b108 Title: Fast3R: Towards 3d reconstruction of 1000+ images in one forward pass Year: (2025)
Ref_id:b109 Title: Using human feedback to fine-tune diffusion models without any reward model Year: (2024)
Ref_id:b110 Title: LayerPano3D: Layered 3d panorama for hyper-immersive scene generation Year: (2025)
Ref_id:b111 Title: Deep model reassembly Year: (2022)
Ref_id:b112 Title: Prometheus: 3d-aware latent diffusion models for feed-forward text-to-3d scene generation Year: (2025)
Ref_id:b113 Title: Matrix-3d: Omnidirectional explorable 3d world generation Year: (2025)
Ref_id:b114 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2025)
Ref_id:b115 Title: No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images Year: (2025)
Ref_id:b116 Title: Going from anywhere to everywhere Year: (2024)
Ref_id:b117 Title: WonderWorld: Interactive 3d scene generation from a single image Year: (2025)
Ref_id:b118 Title: Viewcrafter: Taming video diffusion models for highfidelity novel view synthesis Year: (2025)
Ref_id:b119 Title: Gaussiancube: A structured and explicit radiance representation for 3d generative modeling Year: ()
Ref_id:b120 Title: GS-LRM: Large reconstruction model for 3d gaussian splatting Year: (2024)
Ref_id:b121 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b122 Title: FLARE: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views Year: (2025)
Ref_id:b123 Title: Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation Year: (2023)
Ref_id:b124 Title:  Year: (2025)
Ref_id:b125 Title: Stereo magnification: learning view synthesis using multiplane images Year: (2018)
