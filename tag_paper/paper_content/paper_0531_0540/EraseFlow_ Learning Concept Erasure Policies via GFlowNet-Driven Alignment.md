Title: EraseFlow: Learning Concept Erasure Policies via GFlowNet-Driven Alignment
Abstract: Erasing harmful or proprietary concepts from powerful text-to-image generators is an emerging safety requirement, yet current "concept erasure" techniques either collapse image quality, rely on brittle adversarial losses, or demand prohibitive retraining cycles. We trace these limitations to a myopic view of the denoising trajectories that govern diffusion-based generation. We introduce EraseFlow, the first framework that casts concept unlearning as exploration in the space of denoising paths and optimizes it with GFlowNets equipped with the trajectory-balance objective. By sampling entire trajectories rather than single end states, EraseFlow learns a stochastic policy that steers generation away from target concepts while preserving the model's prior. EraseFlow eliminates the need for carefully crafted reward models and by doing this, it generalizes effectively to unseen concepts and avoids hackable rewards while improving the performance. Extensive empirical results demonstrate that EraseFlow outperforms existing baselines and achieves an optimal trade-off between performance and prior preservation. Warning: This paper may contain content that may seem as offensive in nature.

Section: Introduction
Recent advances in diffusion models have led to remarkable improvements in text-to-image generative models, enabling unprecedented photorealism and widespread adoption across various domains [50,12,49,45,69]. However, because these models are typically trained on large-scale, unregulated internet data, concerns have grown regarding their potential misuse, including the unauthorized reproduction of copyrighted or harmful content [54,15]. Consequently, methods to erase or "unlearn" specific concepts from pretrained text-to-image generators have become critically important to ensure model's safety and compliance [17,28].
Prior approaches addressing these challenges broadly include filtering [67], training data attribution [60], post-generation filtering [41], and explicit concept unlearning [17,28]. While filtering-based methods are hard to scale on arbitrary concepts, unlearning-based methods have recently attracted significant attention due to their capability to intervene. Early unlearning techniques primarily relied on fine-tuning pretrained diffusion models by modifying cross-attention mechanisms [18]. These approaches, however, often degrade the generative model's overall quality and are susceptible to adversarial reintroduction of erased concepts [72]. Reinforcement learning (RL)-based strategies were later introduced to improve alignment [55,43], yet they similarly suffer from brittleness and susceptibility to adversarial attacks. More recently, adversarial unlearning methods have demonstrated improved robustness [26,70], but at the cost of substantial computational overhead, typically requiring hours of compute to erase a single concept, thereby severely limiting scalability. We hypothesize that these limitations originate from inadequate control of the post-training alignment process. Specifically, the stochastic nature of diffusion models implies that early denoising steps have substantial uncertainty and can yield diverse distributions, while later steps become more deterministic. Despite this inherent property, most existing methods treat all denoising steps equally during fine-tuning, ignoring the critical role of evolving conditional marginal distributions. This oversight often leads to suboptimal unlearning outcomes [26,16]. Therefore, a fine-tuning strategy that explicitly accounts for these conditional marginal distributions is crucial for effective and efficient concept erasure. Motivated by recent advancements in generative flow networks (GFlowNets) [5]-probabilistic models capable of sampling from unnormalized distributions-we propose EraseFlow, a novel unlearning method that leverages complete denoising trajectories to dynamically adapt alignment based on evolving conditional marginal distributions through a trajectory balance (TB) formulation. Additionally, to remove the dependency on manually designed and potentially vulnerable reward models, we introduce a straightforward reward-free alignment strategy. Specifically, we theoretically prove that even a constant reward in combination with TB leads to more reliable erasing of semantic content. This enables generalization to arbitrary unseen concepts without explicit reward specification. Critically, our alignment strategy ensures the preservation of the pretrained model's prior while effectively removing targeted concepts.
We validate the robustness and efficacy of EraseFlow through comprehensive evaluations across diverse and challenging scenarios (see Figure 1), including nudity filtering, artistic style removal, and fine-grained realistic concept erasure (e.g., corporate logos such as Nike). Our method consistently outperforms existing baselines on the UDAtk [72] benchmark without requiring adversarial training. Integrating EraseFlow with orthogonal methods like SAFREE [67] and AdvUnlearn [70] further improves results, achieving state-of-the-art performance with only 1% failure rates extend the detailed balance formulation of GFlowNets to diffusion model alignment, enabling finegrained control in image generation. However, these objectives struggle to fully erase specific concepts, as they are not tailored for the asymmetry inherent in erasure tasks.
In contrast, our approach is the first to apply GFlowNets to the problem of concept erasure. We build upon the trajectory balance formulation to design a reward-free objective specifically suited for erasure, enabling robust suppression of unwanted concepts while preserving model priors.
this section cite: ['b48', 'b11', 'b47', 'b43', 'b67', 'b52', 'b14', 'b16', 'b27', 'b65', 'b58', 'b39', 'b16', 'b27', 'b17', 'b70', 'b53', 'b41', 'b25', 'b68', 'b25', 'b15', 'b4', 'b70', 'b65', 'b68']

Section: Preliminaries
In this section, we first share a brief overview of the concept erasure/unlearning formulation and introduce the Generative Flow Networks (GFlowNets).
this section cite: []

Section: Concept Erasure for Diffusion Models
Despite recent progress, diffusion models (DMs) remain vulnerable to generating inappropriate, sensitive, or copyrighted content when given harmful prompts or adversarial inputs. The I2P dataset [53], for example, demonstrates how models can produce NSFW content. ESD [17] is one such method that addresses this issue by shifting the model's output away from a target concept to be edited (c) while preserving overall utility. It modifies the denoising prediction as:
ϵ θ * (x t |c) ← ϵ θ (x t |∅) -η (ϵ θ (x t |c) -ϵ θ (x t |∅)) ,(1)
where x t is the noisy latent at a random timestep t, ϵ(•) is the predicted noise, θ are the parameters of original frozen model, θ * are the parameters fine-tuned model, ∅ is the empty prompt, and η is a positive guidance strength. This update reduces alignment with c, unlike standard classifier guidance [10,46], which increases it. Training minimizes the following loss:
min θ * ℓ ESD (θ * , c) := E ∥ϵ θ * (x t |c) -(ϵ θ (x t |∅) -η(ϵ θ (x t |c) -ϵ θ (x t |∅)))∥ 2 2 ,(2)
which encourages the model to behave like the unconditional model while avoiding c. However, due to the uniform sampling of timesteps t in Eq. ( 2), it can adversely affect the prior distribution (i.e., generation quality and nearby concepts) and perform sub-optimally. Such approaches remain susceptible to adversarial attacks, such as UDAtk [71], which can effectively reintroduce erased concepts. While preference finetuning based strategies leads to reward hacking. To address these vulnerabilities, we propose leveraging GFlowNets, which operate over complete denoising trajectories, offering a more robust framework for concept erasure.
this section cite: ['b51', 'b16', 'b9', 'b44', 'b69']

Section: Generative Flow Networks (GFlowNets)
GFlowNets [5] are a class of probabilistic models that learn to sample x such that the sampling probability P (x) is proportional to a given unnormalized reward density function R : X → R ≥0 that is P (x) ∝ R(x). The sampling process is structured as a traversal over a directed acyclic graph (DAG), where nodes represent states and edges represent transitions. Starting from an initial state s 0 , the model uses a forward policy P F (s t+1 |s t ) to move through intermediate states s 1 , s 2 , . . . , s T -1 until it reaches a terminal state s T , which defines the final sample x.
To ensure the model generates samples that match the reward distribution, GFlowNets also define a backward policy P B (s t |s t+1 ) and a flow function F (s t ) that assigns an unnormalized density to each state. These are trained to satisfy the detailed balance condition:
P F (s t+1 |s t )F (s t ) = P B (s t |s t+1 )F (s t+1 ),(3)
which ensures consistency between forward and backward flows. The training objective minimizes the following loss:
L DB (s t , s t+1 ) = (log P F (s t+1 |s t ) + log F (s t ) -log P B (s t |s t+1 ) -log F (s t+1 )) 2 .(4)
At the final state s T (x), the flow is set equal to the reward, i.e., F (s T ) = R(s T = x). This allows GFlowNets to assign higher probabilities to generation paths that lead to high-reward outcomes. This method is alternatively known as Detailed Balance (DB) objective. Hence, GFlowNets avoid the reward hacking and potential mode collapse [4].
this section cite: ['b4', 'b3']

Section: Proposed Methodology: EraseFlow
We begin by formalizing concept-erasure for text-to-image diffusion models, then cast it as a Detailed Balance objective. Later, we cast the unlearning problem as trajectory matching that can be solved with a trajectory-balance objective of GFlowNets. Let c denote the target prompt (e.g. "nudity") whose visual concept we wish to erase, c * denote a reference (anchor) prompt that is semantically safe (e.g. "fully-dressed person") and ϵ θ be the denoising network of a pre-trained diffusion model, with parameters θ. A text-to-image diffusion model generates an image (x) conditioned on c by a Markov chain: τ = (x T , x T -1 , . . . , x 0 ) , x r ∈ R d , T ≫ 0, where x T ∼ N (0, I) and
x t-1 = 1 √ α t x t - 1 -α t √ 1 -ᾱt ϵ θ (x t , t, c) + σ t z, z ∼ N (0, I)
where, α t and σ t are DDPM parameters. We view every intermediate latent (x T , . . . , x 0 ) as a state s t . Additionally, diffusion denoising is a directed acyclic graph evolving from noise distribution to posterior distribution. Now, we can see that GFlowNets formulation is closely related to the diffusion models, as noted in [68]. The forward policy P F (s t → s t-1 |c) is exactly the diffusion model's reverse-process conditional p θ (x t-1 |x t , t, c); the backward policy P B (s t-1 → s t |c) is the corresponding noising step q(x t |x t-1 ). Therefore, with slight trick of hands, we can directly apply the Eq. ( 4) for concept erasure as:
L DB = log p θ (x t-1 | x t , c) + log F ϕ (x t | c) + log R ′ (x t | c, c * ) -log q(x t | x t-1 , c) -log F ϕ (x t+1 | c) -log R ′ (x t+1 | c, c * ) 2 , (5
)
TB (Ours) TB with reward DB with reward Prompt: bright realistic anorexic ribs boney obese eating herself..., art by francis bacon Nudity Method I2P (↓) Ring-a-Bell (↓) MMA-Diff (↓) DB w/ reward 8.3 6.39 14.1 TB w/ reward 2.1 2.53 1.7 EraseFlow (ours) 2.8 0.00 0.60 where, F (x t | c) = F ϕ • R ′ (x t | c, c * ), ϕ is flow parameter and R ′ (• | c, c * ) = R(x 0 | c * ) -R(x 0 | c) with F (x 0 | c) = R ′ (x 0 ). Here, R(•) is any model capable of classifying the image with respect to a given prompt or conditions. Essentially, R ′ (• | c, c * ) measures how much more the image aligns with the anchor prompt c * than with the target concept c. However, our preliminary experiments reveals that optimizing the Eq. (5) objective leads to reasonable performance, but the training becomes unstable over time, eventually leading to model collapse and loss of prior fidelity.
this section cite: ['b66']

Section: Trajectory Balance (TB).
To overcome these limitations and further improve the performance, we bring TB formulation for concept erasure. Specifically, given the entire diffusion denoising trajectory, we get the following TB constraint after [40] as:
Z ϕ T t=1 p θ (x t-1 |x t , t, c) = R(x 0 ) T t=1 q(x t |x t-1 ),(6)
where Z ϕ is a scalar parameter estimating the sum of all the reward values achievable from the initial state x 0 . By minimizing the squared log difference of both sides from Eq. ( 6) over the sampled trajectory from the target prompt c yields the following objective function:
L T B (ϕ, θ) = log Z ϕ + T t=1 log p θ (x t-1 |x t , t, c) -log R ′ (x 0 ) - T t=1 log q(x t |x t-1 ) 2 . (7
)
Empirically, TB propagates credit to early states far more effectively than DB losses, avoiding the noisy intermediate reward estimates that hamper previous RL-style unlearning methods. Now, by either optimizing the Eq. ( 7) or (4) with a specific reward model, one should get the properly aligned diffusion model. This has been verified in text-to-image aesthetic alignment. However, as shown in Figure 3, we observe that DB (Eq. ( 4)) performs poorly whereas the TB (Eq. ( 7)) works well for concept erasure. Reward-free alignment. However, a central obstacle in alignment-based concept-erasure is the absence of reliable, non-adversarial reward models for arbitrary visual concepts (e.g. an actor promoting a branded product). We sidestep this by eliminating the external reward altogether. We assign a constant reward β > 0 to every trajectory τ generated by the anchor prompt c * and zero otherwise. Concretely, let τ c * = {τ : τ generated under c * }, and define: R(τ ) = β, ifτ ∈ τ c * 0, otherwise.
With this choice, Eq. ( 7) becomes, for anchor trajectory τ * ∈ τ c * , the following objective,
L EraseFlow c←c * = log Z ϕ + T t=1 log p θ (x * t-1 |x * t , t, c) -log β - T t=1 log q(x * t |x * t-1 ) 2 ,(8)
where x * denotes the state from anchor trajectory τ * . Minimizing Eq. ( 8) forces the flow under the target prompt c to match the density of anchor trajectories, effectively transplanting the safe distribution of c * onto prompt c. This simplification enables stable and efficient training while retaining the prior generation quality.
Intuitively, the erasure process can be understood as a redistribution of probability mass between target and anchor regions within the data space. As illustrated in Figure 4, EraseFlow achieves this by reweighting entire denoising trajectories-amplifying those aligned with anchor regions while reducing the probability of trajectories leading toward target regions. During optimization, this redistribution is driven by a flow of probability mass that progressively redirects trajectories from target to anchor regions under the TB objective. Initially, the pretrained model concentrates probability mass around target regions, increasing the likelihood of generating target concepts. After optimization, the resulting distribution concentrates around anchor regions, effectively suppressing target concepts while maintaining overall fidelity.
p θ * (x t-1 | x t , t, c) = p θ (x t-1 | x t , t, c * ),
and consequently the marginal image distributions coincide:
p θ * (x 0 | c) = p θ (x 0 | c * ).
Hence the visual concept unique to c is completely erased.
Proof sketch. Zero constant-reward loss implies the logarithmic TB identity (6) (with R = β) holds for every trajectory sampled under c * . Subtracting the corresponding identity for c * eliminates the common t log q term and yields T t=1 log p θ * = T t=1 log p θ . Because the summands are independent across t and both sides are normalized, equality must hold at each timestep, giving the first claim. Telescoping over the trajectory then proves equality of the terminal distribution x 0 , completing the argument.
this section cite: ['b38']

Section: □
The proof is given in the appendix C. Proposition 4.1 confirms that no external classifier or adversarial signal is needed: a single constant reward suffices to guarantee exact distributional alignment when the TB loss is minimized. This proposition thus formalizes the key benefit of our formulation: a provable route to stable concept removal with a gradient-based objective whose variance does not explode.
Plug and Play. Since EraseFlow operates directly on the diffusion model, it can be seamlessly integrated as a plug-and-play module with orthogonal approaches such as the training-free SAFREE [67].
With minimal fine-tuning on retention prompts, it can also be combined with AdvUnlearn [70], which modifies the text encoder of the T2I model.
this section cite: ['b65', 'b68']

Section: EraseFlow training algorithm
We train both the diffusion model and the flow-based partition function to erase specific concepts. In each epoch, we sample a trajectory from the diffusion model conditioned on an anchor safe prompt c * (e.g., fully dressed) while erasing target prompt c (e.g., nudity). This anchor trajectory is paired with the c, and the loss in Eq. ( 8) is applied across all timesteps. For memory efficiency, we sample only a subset of timesteps during training-specifically, 10 timesteps from the first 40 denoising steps and 10 from the last 10 steps. The parameters θ and ϕ of the diffusion model and the flow partition function, respectively, are updated using gradients from this loss. Algorithm 1 summarizes the training procedure. To prevent drift and entanglement, we only finetune the model upto STOP_SAMPLING epoch. In this way, EraseFlow improves convergence and aligns the unsafe distribution with the safe distribution. for t in (T -1)..0 do 8:
Compute and accumulate loss with τ ′ and the target prompt c using Eq. ( 8)
9:
end for 10:
Update model parameters θ, Z ϕ 11: end for 5
this section cite: ['b4']

Section: Experimental Results
Here, we conduct a comprehensive evaluation of EraseFlow by extensively benchmarking it on various erasing tasks.
this section cite: []

Section: Experimental Setup
Concept Erasure Tasks. We evaluate methods across three tasks: (1) NSFW, which involves suppressing nudity generation when conditioned on implicit or explicit prompts; (2) Artistic style, where we test the model's capability to erase "Van Gogh" and "Caravaggio" artistic styles; and (3) Fine-grained, which targets the removal of specific elements-such as the "Nike logo" from Nike shoes, the "Coca-Cola logo" from bottles, or "wings" from a Pegasus-while preserving overall image-text alignment. Datasets and Evaluation Metrics. For the nudity task, we use red-teaming prompts from multiple sources: 142 from I2P [53], 79 from Ring-a-Bell [58], 1000 from MMA-Diffusion [66], and 142 more from I2P extracted using UDAtk [71]. For artistic style erasure, we use 50 adversarial prompts per target style generated via UDAtk. Fine-grained erasure is evaluated using 10 diverse prompts per concept generated with GPT-4o, with 10 images per prompt, and scored using Gecko [31], inspired by EraseBench [1]. NSFW erasure is measured using Attack Success Rate (ASR), with detection by NudeNet [3] at a threshold of 0.6 (lower is better). Artistic style erasure is evaluated via mean cosine similarity between generated and reference images in the same style, using features from CSD [56]. We test style erasure on "Van Gogh" and "Caravaggio". For fine-grained erasure, we report both concept score (absence of the erased concept) and total score measures both the preservation of non-target concepts and the successful removal of the target concept. We also evaluate image quality using CLIP Score [21] (higher is better) and FID [22] (lower is better) on MSCOCO [36], and report training time (in minutes) for each method. Please refer to the appendix G.3 for prompt examples used in fine-grained evaluation.
Baselines. We categorize our baseline into 3 categories. (1) Non-adversarial training methods: ESD [17], UCE [18], MACE [39], DUO [44], and EraseFlow (ours), (2) Inference time intervention: SAFREE [67] and finally (3) Adversarial training methods: RACE [26] and AdvUnlearn [70].
Training Details. We use Stable Diffusion v1.4 [51] as the backbone for all experiments, following ESD [17]. EraseFlow is implemented following Algorithm 1 and trained for 20 iterations, each using a single data batch. We directly set logβ in Eq. ( 8) to 2.5. The STOP_SAMPLING parameter is set to 21 for nudity erasure, 11 for fine-grained erasure, and 1 for artistic style erasure. A learning rate of 3.0 × 10 -4 is used for nudity and fine-grained tasks, and 5.0 × 10 -4 for artistic style erasure.
this section cite: ['b51', 'b56', 'b64', 'b69', 'b29', 'b0', 'b2', 'b54', 'b20', 'b21', 'b34', 'b16', 'b17', 'b37', 'b42', 'b65', 'b25', 'b68', 'b49', 'b16']

Section: Main Results
Robustness Against Red-Teaming in NSFW Erasure. To evaluate adversarial robustness, we compare EraseFlow with training-free, non-adversarial, and adversarial methods. As a non-adversarial method, EraseFlow achieves strong ASR reduction on UDAtk, outperforming the second-best non-adversarial method DUO by 30.51%. Importantly, EraseFlow even outperforms the adversarial method, R.A.C.E, by 16.95%. While AdvUnlearn achieves the lowest ASR, it relies on adversarial fine-tuning used during the evaluation in UDAtk. Table 2 further shows EraseFlow 's consistent improvements across I2P [53], Ring-a-Bell [58], and MMA-Diffusion [66]. In plug-and-play setups, combining EraseFlow with SAFREE or AdvUnlearn yields further gains-EraseFlow + AdvUn- learn nearly eliminates nudity. Qualitative results in Figure 5 highlight EraseFlow's effectiveness in preserving alignment while achieving robust erasure. Additional qualitative results are included in the appendix K.
Robustness Against Red-Teaming in Artistic Style Erasure. We report the average results of erasing "Van Gogh" and "Caravaggio" in Table 1 on UDAtk. In line with nudity erasure, EraseFlow outperforms the non-adversarial methods by at least 1%. As visualized in Figure 5, Eraseflow suppresses the Van Gogh artistic style. Moreover, in the plug and play approach with AdvUnlearn and SAFREE, we further improve the performance of EraseFlow by 17.59% and 2.55% and perform competitively to respective baselines. Please refer to appendix K for more qualitative results. Fine-Grained Erasure Analysis. Table 1 presents the average results for fine-grained erasure across three tasks: removing the "Nike logo" from Nike shoes, the "Coca-Cola logo" from Coca-Cola bottles, and "wings" from a Pegasus. EraseFlow achieves the achieves comparable concept score with respect to DUO. We also present Concept Score and Total Score in Table 3. While ESD achieves stronger concept removal, it does so at the cost of excessive erasure as evidenced by it's Total Score. In contrast, EraseFlow strikes a better balance between effective erasure and the preservation of unrelated fine grained content with outperforming the previous best by 5.31% in Total Score. As also shown in Figure 5, EraseFlow effectively removes the fine grained concept "wings" from "Pegasus" while maintaining image-text alignment with other prompts and image quality. This highlights the capability of EraseFlow to perform fine grained erasure. Image-Text Alignment and Image Quality Retention. Preserving image quality and image-text alignment to unrelated concepts is crucial during concept erasure. To evaluate this, we test EraseFlow and all baselines on 10,000 prompts from the MSCOCO [36] dataset and report the average CLIP Score [21] and FID [22] in Table 1. Adversarial methods like AdvUnlearn show strong erasure performance but often degrade both image quality and alignment as evidence by the numbers. Nonadversarial methods better preserve quality and alignment but are less robust to adversarial attacks. EraseFlow strikes a strong balance between these objectives: it matches UCE and DUO in CLIP Score and outperforms all baselines in FID except MACE, indicating that it effectively erases concepts without compromising visual fidelity.
this section cite: ['b51', 'b56', 'b64', 'b34', 'b20', 'b21']

Section: Efficiency of EraseFlow Training.
EraseFlow is highly efficient to train as shown in Table 1.
While adversarial methods like AdvUnlearn and R.A.C.E require at least 3 hours of training to achieve strong results, EraseFlow reaches comparable or superior performance with just 3 minutes of training on a single A100 GPU. This efficiency is made possible by leveraging all denoising steps during training, enabling EraseFlow to achieve robust concept erasure at a fraction of the computational cost.
this section cite: []

Section: Ablation Studies
We perform ablations to analyze EraseFlow's design choices, using the nudity erasure task as a representative and challenging benchmark unless noted otherwise.
Effect of log β. We vary log β across a wide range to study its effect on concept erasure and image quality. As shown in Figure 6 (left), small values (log β ≤ 1) result in poor erasure (high I2P) and training instability due to the log β term. In contrast, values in the range [2,3] yield a sharp 96% improvement, providing stable training and the best erasure-quality trade-off. We therefore adopt this setting as default. Larger values (e.g., log β ≥ 50) further reduce FID but also degrade erasure performance, confirming the need for balance.
this section cite: ['b1', 'b2']

Section: Effect of STOP_SAMPLING.
Increasing STOP_SAMPLING-which triggers more frequent anchor trajectory resampling-exposes the model to richer safe examples and improves credit assignment.
As shown in Figure 6 (right), performance improves steadily with larger values, reaching the best results at epoch 20. Conversely, too small a value restricts trajectory diversity, weakening erasure.
Please refer to appendix E for more ablations.
this section cite: []

Section: Conclusion
EraseFlow introduces a novel approach to concept erasure by framing it as a reward-free GFlowNetsbased alignment task. This allows a constant-reward trajectory balance objective to effectively remove unwanted concepts-such as copyrighted logos, artistic styles, or sensitive themes-while preserving the prior. EraseFlow improves the robustness and image quality across benchmarks, all with high efficiency. Its performance is backed by formal guarantees that the edited distribution aligns exactly with a safe anchor, and by empirical results demonstrating an optimal balance between erasure effectiveness, prior preservation, and computational cost. Together, these foundations and results establish EraseFlow as a lightweight, plug-and-play safety primitive for the next generation of diffusion models.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2025)
Ref_id:b1 Title: Refact: Updating text-to-image models by editing the text encoder Year: (2023)
Ref_id:b2 Title: Nudenet: Neural nets for nudity classification, detection and selective censoring Year: (2019)
Ref_id:b3 Title: Flow network based generative models for non-iterative diverse candidate generation Year: (2021)
Ref_id:b4 Title:  Year: (2023)
Ref_id:b5 Title: Training diffusion models with reinforcement learning Year: (2024)
Ref_id:b6 Title: Fantastic targets for concept erasure in diffusion models and where to find them Year: (2025)
Ref_id:b7 Title: Prompting4debugging: Red-teaming text-to-image diffusion models by finding problematic prompts Year: (2024)
Ref_id:b8 Title: Prompting4debugging: Red-teaming text-to-image diffusion models by finding problematic prompts Year: (2024)
Ref_id:b9 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b10 Title: RAFT: Reward ranked finetuning for generative foundation model alignment Year: (2023)
Ref_id:b11 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b12 Title: Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b13 Title: The stable signature: Rooting watermarks in latent diffusion models Year: (2023)
Ref_id:b14 Title: Datacomp: In search of the next generation of multimodal datasets Year: (2023)
Ref_id:b15 Title: Distilling diversity and control in diffusion models Year: (2025)
Ref_id:b16 Title: Erasing concepts from diffusion models Year: (2023-10)
Ref_id:b17 Title: Unified concept editing in diffusion models Year: (2024)
Ref_id:b18 Title: Eraseanything: Enabling concept erasure in rectified flow transformers Year: (2025)
Ref_id:b19 Title: Reliable and efficient concept erasure of text-to-image diffusion models Year: (2024)
Ref_id:b20 Title: Clipscore: A reference-free evaluation metric for image captioning Year: (2021)
Ref_id:b21 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2018)
Ref_id:b22 Title: Receler: Reliable concept erasing of text-to-image diffusion models via lightweight erasers Year: (2023)
Ref_id:b23 Title: Trasce: Trajectory steering for concept erasure Year: (2025)
Ref_id:b24 Title: Wouaf: Weight modulation for user attribution and fingerprinting in text-to-image diffusion models Year: (2023)
Ref_id:b25 Title: Race: Robust adversarial concept erasure for secure text-to-image diffusion model Year: (2024)
Ref_id:b26 Title: Decentralized attribution of generative models Year: (2021)
Ref_id:b27 Title: Ablating concepts in text-to-image diffusion models Year: (2023)
Ref_id:b28 Title: Localized concept erasure for text-to-image diffusion models using training-free gated low-rank adaptation Year: (2025)
Ref_id:b29 Title:  Year: (2024)
Ref_id:b30 Title: Sculpting memory: Multi-concept forgetting in diffusion models via dynamic mask and concept-aware optimization Year: (2025)
Ref_id:b31 Title: Set you straight: Auto-steering denoising trajectories to sidestep unwanted concepts Year: (2025)
Ref_id:b32 Title: Speed: Scalable, precise, and efficient concept erasure for diffusion models Year: (2025)
Ref_id:b33 Title: Get what you want, not what you don't: Image content suppression for text-toimage diffusion models Year: (2024)
Ref_id:b34 Title:  Year: (2015)
Ref_id:b35 Title: Flow-grpo: Training flow matching models via online rl Year: (2025)
Ref_id:b36 Title: Efficient diversity-preserving diffusion alignment via gradient-informed gflownets Year: (2024)
Ref_id:b37 Title: Mass concept erasure in diffusion models Year: (2024)
Ref_id:b38 Title: Trajectory balance: Improved credit assignment in gflownets Year: (2023)
Ref_id:b39 Title: Towards safe synthetic image generation on the web: A multimodal robust nsfw defense and million scale dataset Year: (2025)
Ref_id:b40 Title: Attributing image generative models using latent fingerprints Year: (2023)
Ref_id:b41 Title: Direct unlearning optimization for robust and safe text-to-image models Year: (2024)
Ref_id:b42 Title: Direct unlearning optimization for robust and safe text-to-image models Year: (2025)
Ref_id:b43 Title: Eclipse: A resource-efficient text-to-image prior for image generations Year: (2024)
Ref_id:b44 Title: Steering rectified flow models in the vector field for controlled image generation Year: (2024)
Ref_id:b45 Title: Concept arithmetics for circumventing concept inhibition in diffusion models Year: (2024)
Ref_id:b46 Title: Circumventing concept erasure methods for text-to-image generative models Year: (2024)
Ref_id:b47 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b48 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b49 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b50 Title: Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models Year: (2022)
Ref_id:b51 Title: Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models Year: (2023)
Ref_id:b52 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b53 Title: Advancing multilingual reasoning through multilingual alignment-as-preference optimization Year: (2024)
Ref_id:b54 Title: Measuring style similarity in diffusion models Year: (2024)
Ref_id:b55 Title: Ring-a-bell! how reliable are concept removal methods for diffusion models? Year: (2024)
Ref_id:b56 Title: Ring-a-bell! how reliable are concept removal methods for diffusion models? Year: (2023)
Ref_id:b57 Title: Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization Year: (2024)
Ref_id:b58 Title: Data attribution for text-to-image models by unlearning synthesized images Year: (2024)
Ref_id:b59 Title: Xiang Wang, and Xiangnan He. Precise, fast, and low-cost concept erasure in value space: Orthogonal complement matters Year: (2025)
Ref_id:b60 Title: Ace: Antiediting concept erasure in text-to-image models Year: (2025)
Ref_id:b61 Title: Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery Year: (2023)
Ref_id:b62 Title: Using human feedback to fine-tune diffusion models without any reward model Year: (2024)
Ref_id:b63 Title: Mma-diffusion: Multimodal attack on diffusion models Year: (2024)
Ref_id:b64 Title: Mma-diffusion: Multimodal attack on diffusion models Year: (2023)
Ref_id:b65 Title: SAFREE: Trainingfree and adaptive guard for safe text-to-image and video generation Year: (2025)
Ref_id:b66 Title: Improving gflownets for text-to-image diffusion alignment Year: (2024)
Ref_id:b67 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b68 Title: Defensive unlearning with adversarial training for robust concept erasure in diffusion models Year: (2024)
Ref_id:b69 Title: To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images Year: ()
Ref_id:b70 Title: To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now Year: (2024)
