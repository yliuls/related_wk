Title: COMPOSITIONAL DIFFUSION WITH GUIDED SEARCH FOR LONG-HORIZON PLANNING
Abstract: Generative models have emerged as powerful tools for planning, with compositional approaches offering particular promise for modeling long-horizon task distributions by composing together local, modular generative models. This compositional paradigm spans diverse domains, from multi-step manipulation planning to panoramic image synthesis to long video generation. However, compositional generative models face a critical challenge: when local distributions are multimodal, existing composition methods average incompatible modes, producing plans that are neither locally feasible nor globally coherent. We propose Compositional Diffusion with Guided Search (CDGS), which addresses this mode averaging problem by embedding search directly within the diffusion denoising process.Our method explores diverse combinations of local modes through populationbased sampling, enforces global consistency through iterative resampling between overlapping segments, and prunes infeasible candidates using likelihood-based filtering. CDGS matches oracle performance on seven robot manipulation tasks, outperforming baselines that lack compositionality or require long-horizon training data. The approach generalizes across domains, enabling coherent text-guided panoramic images and long videos through effective local-to-global message passing. More details: https://cdgsearch.github.io/

Section: 
Synthesizing coherent long sequences is a crucial and challenging task, requiring reasoning over extended horizons. This task arises naturally in various domains: robotic actions must enable future steps, parts of a panorama must align semantically, and subjects in a video must remain consistent across hundreds of frames.
Recent work leverages generative models to learn long sequence distributions [25,3], with diffusion models [53,21] gaining popularity for modeling multi-modal data [9,20]. However, full-sequence data is expensive to acquire, and monolithic models fail to generalize beyond training horizons [10]. As an alternative, compositional generation effectively combines short-horizon local distributions to sample long-horizon global plans [66,44,40]-e.g., chaining skills for task planning, connecting images into panoramas, or stitching clips into videos. While this improves data-efficiency and allows extrapolation beyond training data, it intro- duces a critical challenge: as local plan distributions become highly multimodal, the distribution of global plans inherits combinatorial multi-modality. For example, in the robotics scenario in Fig. 2, because the robot has a large combination of actions and objects it can act on, the search space of possible plans grows exponentially with the length of the planning horizon.
Existing methods for compositional generation offer a promising approach, using score-averaging to compose modes of local distributions into a global distribution [66,44]. However, these methods have an important limitation: their inability to handle the combinatorial multi-modality leads them to average incompatible local modes (mode-averaging), ultimately producing invalid global plans. Addressing such complex multi-modal distributions requires inference methods that jointly reason about compatibility between local modes and effectively navigate the exponentially large search space.
To address the challenge and overcome the limitation, we aim to identify compatible sequences of local modes that compose into a globally coherent plan. Given the diversity and multi-modality of the search space, we take inspiration from classical search techniques and introduce Compositional Diffusion with Guided Search (CDGS), a guided search mechanism integrated into the diffusion denoising process as illustrated in Fig. 1. To facilitate the search during inference, at each diffusion timestep, our method introduces two key components: (i) iterative resampling to enhance localglobal message passing in compositional diffusion to propose globally plausible candidates, and (ii) likelihood-based pruning to remove incoherent candidates that fall into low-likelihood regions due to mode-averaging. Together, these components enable CDGS to efficiently sample coherent long-horizon plans. For robotics tasks, our method outperforms or is on par with baselines that lack compositionality or use long-horizon data for training, respectively. We also show the efficacy of our method in long text-to-image and text-to-video tasks (Fig. 2), producing more coherent and consistent generations.
this section cite: ['b24', 'b2', 'b52', 'b20', 'b8', 'b19', 'b9', 'b65', 'b43', 'b39', 'b65', 'b43']

Section: BACKGROUND
Problem formulation. A long-horizon plan generation problem is characterized by the task of constructing a global plan ! = (x 1 ,. ..,x N ) as a sequence of variables x i , by sampling from the joint distribution p(!). The problem becomes goal-directed if ! must connect a given start x 1 = x s to a desired goal x N = x g . Such problems arise in diverse domains: long-horizon manipulation planning, panoramic images, and long videos. While modeling the full joint distribution p(!) would directly model all dependencies between any x i , it usually entails end-to-end learning from long-horizon data, which can be infeasible or expensive to obtain. In the absence of long-horizon data, a promis-  ing strategy is to approximate the joint distribution p(!) with a factor graph of overlapping local distributions that can be learned from short-horizon data. For the joint variable ! = (x 1 , x 2 ,. .., x N ), we construct a factor graph [30] connecting variable nodes {x i } N i=1 and factor nodes {y j } M j=1 , where each factor y j represents the joint distribution of contiguous subsequences of !. For example, we represent ! = (x 1 , x 2 ,. ..,x 5 ) with factors y 1 = (x 1 , x 2 , x 3 ), y 2 = (x 3 , x 4 , x 5 ). With this, we construct the joint distribution p(!) using the Bethe approximation [64]:
p(!) := p(x 1 , x 2 , x 3 )p(x 3 , x 4 , x 5 ) . .. p(x 3 ) . .. = ! M j=1 p(y j ) ! N i=1 p(x i ) d i →1(1)
where d i is the degree of the variable node x i . This representation enables sampling from the longhorizon distribution p(!) using only samples drawn from a short-horizon distribution p(y).
this section cite: ['b29', 'b63']

Section: Diffusion models.
Diffusion models are defined by a forward process that progressively injects noise into the data distribution p(y
(0) ) and a reverse diffusion process that iteratively removes the noise by approximating ∀ log p to recover the original data distribution. For a given noise injection schedule ∀ t , forward noising adds a Gaussian noise # to clean samples s.t. y (t) = ↑ ∀ t y (0) + ↑ 1 → ∀ t #. With p t being the distribution of noisy samples, the denoising is performed using the score function ∀ y (t) log p t (y (t) ) often estimated by a neural network # ∃ (y (t) ,t) learned via minimizing the score matching loss [23] given by E t,y (0) [|# → # ∃ (y (t) ,t)| 2 ]. Such a score function allows denoising the noise samples via sampling from p(y (t→1) |y (t) ) = N y (t→1) ; ↑ ∀ t→1 ŷ(t) 0 + 1 → ∀ t→1 → % 2 t #(y (t) ,t), % 2 t I (2) where ŷ(t) 0 = y (t) → ↑ 1→∀ t # ∃ (y (t) ,t) ↑ ∀ t
is the Tweedie estimate of the clean sample distribution at denoising step t and % t controls stochasticity [54]. Several works have leveraged the flexibility of the denoising process in performing post-hoc guidance [20] and plug-and-play generation [37,12].
Compositional sampling with diffusion models. Under the diffusion model formulation, we can compositionally sample [11,66] from the factor graph representation of p(!) by calculating the score ∀ log p(!) as a sum of factor and variable scores following Eq. 1:
∀ log p(!) := M # j=1 ∀ log p(y j ) + N # i=1 (1 → d i )∀ log p(x i )(3)
In practice, our factor graph is a chain, so overlapping variables (i.e., the ones shared between neighboring factors y j and y j+1 ) have degree d i = 2 while non-overlapping ones have d i = 1 (i.e. their marginals have no contribution to ∀ log p(!)). For overlapping variables, we approximate the marginal scores using the average of the conditional score:
∀ log p(x i ) ↓ 1 2 ∀ log p y j (x i | ...) + ∀ log p y j+1 (x i |...)
where x i ↔ y j ↗ y j+1 denotes the overlapping variable between y j and y j+1 . This, along with the scores of p(y j ) computed from local distribution, allows us to formulate the global compositional score ∀ log p(!) using Eq. 3. While this formulation enables generalization beyond the lengths seen during training, it comes with limitations described in Sec. 3.
! (") " $ (") " % (") " % (") Local Plan Diffusion Model # & (" ' ( , &) # & (" ) ( , &) # & (" * ( , &) #(! (() , &) Denoising Diffusion ! ("+$) Resampling (× #) Compositional Score Sampling for Global Coherence ! (,) " $ (,) " % (,) " % (,) " $ (-+$) " % (-+$) " % (-+$) DDIM Inversion $!(& (#$%) , () " $ (-) " % (-) " % (-) * + ' = --.$!(& ( (#$%) , () ./ ) * +,' ( ! (.) ... ... Local feasibility metric with DDIM Inversion for Pruning !(0, %) ... ... ... ... ... ((0, *) Batch ! (.) ... Batch ! (") ... Batch ! ("+$) Predicted Clean Sample Local feasibility metric +(⋅) ... Re-populate Batch ! ("+$) Elite Batch ! ("+$) min +(⋅) ...
this section cite: ['b53', 'b19', 'b36', 'b11', 'b10', 'b65']

Section: METHOD
Challenge: Compositional sampling with multi-modal distributions. Solving long-horizon tasks requires constructing a coherent global plan distribution that induces an exponentially large search space and requires reasoning about long-horizon dependencies. Data scarcity prohibits directly learning the target global plan distribution p(!), so a convincing alternative is to approximate it as a composition of local plan distribution p(y) (using Eq. 1). Thus, one can sample short-horizon local plans y 1:M ↘ p(y) and compose them with suitable overlaps to form a coherent !. However, as the diversity of feasible local behaviors increases, p(y) becomes highly multi-modal and composing such distributions causes p(!) to inherit combinatorial multi-modality-where each mode of the global plan distribution corresponds to a distinct sequence of modes from the local plan distribution. In this setting, naïve compositional methods ( [43]) that merge distributions y 1:M ↘ p(y) via score averaging (Eq. 3) often fail due to the mode-averaging issue: selecting high-likelihood local segments that, while individually plausible, result in incompatible mode sequences-leading to inconsistent overlaps and incoherent global plans. A natural way to address multi-modality is to explore diverse modes during sampling, an idea recently explored by inference-time scaling approaches [42,69]. However, these methods are limited to sampling from standalone distributions and not a composed sequence of distributions. The key challenge is to generate a feasible sequence of local plans that collectively form a coherent global plan-requiring a sampling algorithm that reasons over structured combinations of modes rather than collapsing into incoherent averages.
Our method: Compositional Diffusion with Guided Search (CDGS). CDGS is a structured inference-time algorithm designed to identify coherent sequences of local modes that form valid global plans. Specifically, CDGS employs a population-based search to explore and select promising mode sequences beyond naïve sampling. To facilitate the search, it: (i) incorporates iterative resampling into the compositional score calculation to enhance information exchange across distant segments, leading to potentially coherent global plan candidates, and (ii) prunes the incoherent candidates by evaluating the likelihood of their local segments with a ranking objective. Note that this is all within a standard denoising diffusion process, making CDGS a plug-and-play sampler applicable across domains, including robotics planning, panorama image generation, and long video generation. In the following sections, we detail each of these components and demonstrate how their integration enables efficient navigation of the complex multi-modal search space to produce coherent long-horizon plans.
this section cite: ['b42', 'b41', 'b68']

Section: COMPOSITIONAL DIFFUSION WITH GUIDED SEARCH
A key challenge with multi-modal distributions is that naïve compositional sampling can lead to incoherent global plans: since each segment is independently sampled from p(y), they may not align well at their overlaps and potentially lead to mode-averaging issues-where high-likelihood local plans do not combine to form a feasible global plan.
#(! (t) ,t) = ComposedScore(! (t) ,t, # ∃ , x s , x g ) 5: !(t) 0 = (! (t) → 1 → ∀ t #(! (t) ,t))/ ↑ ∀ t 6:
Rank plans using J( !(t) 0 ) Eq. 5   7:
Select best-K global plans 8:
Repopulate candidates using filtered plans 9:
! (t→1) ↘ p(! (t→1) |! (t) , !(
t) 0 ) Eq. 2 10: end for 11: return ! (0) Algorithm 2 ComposedScore Require: Noisy sample ! (t) , denoising timestep t, pretrained local plan score function # ∃ Require: Start and goal: x s , x g Require: Number of resampling steps U 1: for u = 1,...,U do 2: Calculate #(! (t) ,t) using Eq. 3 3: if u < U then 4: Calculate ! (t→1) using Eq. 2 5:
Add noise to x s /x g : 6:
x (t→1) s/g ↘ N( ↑ ∀ t→1 x s/g , (1 → ∀ t→1 )I) 7:
Inpaint noisy start and goal in ! (t→1) 8:
Resampling:
! (t) ↘ p(! (t) |! (t→1) ) 9:
end if 10: end for 11: return #(! (t) ,t)
To address this, our approach leverages a guided search procedure that explores promising sequences of local modes while filtering out ones that are more likely to result in incoherent global plans.
this section cite: []

Section: Method formulation.
At each diffusion timestep t, given a noisy global plan ! (t) , our goal is to sample from an improved next-step distribution over ! (t→1) , that is more likely to yield a coherent global plan. To achieve this, we define a modified sampling distribution:
p J ! (t→1) |! (t) ∃ p ! (t→1) |! (t) exp → J !(t→1) 0 /& t ,
where
(i) p(! (t→1) |! (t)
) is the original diffusion transition realized using the compositional score function #(! (t) ,t), (ii) !(t→1) 0 is the Tweedie-estimate of the clean global plan at timestep t → 1, (iii) J(•) is a plan ranking metric we define below, and (iv) & t controls the exploration-exploitation tradeoff. We approximate sampling from this distribution using a Monte Carlo search procedure resembling the cross-entropy method: draw a batch of noisy global plans from p(! (t→1) |! (t) ), rank them using J and retain a subset of elite global plans that minimizes the evaluation metric J(•) as illustrated in Algorithm 1. The number of elites K is a tunable parameter of our algorithm, enabling exploration of many possibilities in parallel when the planning problem is very large/difficult. Now, we just need to ensure that (i) the global plans are ranked appropriately and (ii) the candidate samples proposed by compositional sampling contain informative, globally coherent mode-sequences to pursue.
this section cite: []

Section: Ranking global plans via local feasibility.
To guide the search effectively, we require a mechanism to evaluate the feasibility of candidate plans. Our key insight is that a global plan is feasible iff all of its local transitions are feasible. Since the local model p(y) is trained to model feasible short-horizon behavior, high-likelihood local plans are strong indicators of local feasibility. Therefore, a globally feasible plan should consist of high-likelihood local-plan segments throughout. However, computing exact likelihoods in diffusion models is computationally expensive [55], often intractable.
To address this, we leverage DDIM inversion [54] to approximate the likelihoods of local plan segments y. Each local segment y of a sampled global plan ! goes through forward diffusion using the learned score network (# ∃ ) such that:
y (t) ↑ ∀ t = y (t→1) ↑ ∀ t→1 + 1 → ∀ t ∀ t → 1 → ∀ t→1 ∀ t→1 # ∃ (y (t→1) ,t)(4)
A high-likelihood sample follows a low-curvature path, whereas low-likelihood samples exhibit high curvature to bring noisy latents in-distribution when forward noised [18] (refer App. D). Specifically, we define a smoothness measure based on the curvature of the diffusion trajectory during inversion:
g y (0) = T # i=1 ∋ # ∃ (y (i→1) , i) ∋ i 2 , J(! (0) ) = M ! m=1 exp → g y (0) m(5)
where g(y (0) ) measures closeness of y (0) to the nearest mode of p(y), intuitively. A higher value of g(y (0) ) corresponds to lower-likelihood local plans. We aggregate g(y (0) ) over all local plan segments y
1:M in ! (0) to define the global plan ranking metric J(! (0) ) to measure plan feasibility. Low-quality plans have high J values, making their denoising paths more likely to be pruned.
this section cite: ['b54', 'b53', 'b17']

Section: ITERATIVE RESAMPLING
To ensure the effectiveness of the guided search, it is not enough to rank global plans correctly-we must also promote globally coherent candidate plans. However, standard compositional sampling fails to propagate long-horizon dependencies across overlapping local plans. Consider the running example in Fig. 3. After one denoising step, due to independent sampling of local plans, y 1 has no information about y 6 , and vice versa.
To address this, we apply iterative resampling [39]: repeatedly alternating between forward noising
! (t) ↘ p(! (t) |! (t→1)
) and denoising steps. This procedure enables the score network's predictions for each segment to incorporate information from distant neighbors via overlapping variables, encouraging global consistency. Mathematically, this process resembles belief propagation on a chain of factors where each local plan y m ↔ y 1:M in ! depends on its neighbors y m→1 and y m+1 through the respective overlaps (y m ↗ y m→1 and y m ↗ y m+1 ). During resampling, the belief of y m is updated as: p(y m |y m→1 , y m+1 ) ∃ p(y m )p(y m |y m ↗ y m→1 )p(y m |y m ↗ y m+1 ) Following Algorithm 2, after U iterations, this iterative resampling ensures that information propagates across the entire long-horizon sequence, producing a more globally coherent plan.
this section cite: ['b38']

Section: Summary of CDGS.
We propose a guided-search algorithm by integrating a population-based pruning strategy within compositional sampling. Given a local plan score function, our approach samples potentially coherent global plan candidates and filters out plans with locally inconsistent segments. Repeating this throughout the denoising process improves the probability that the retained candidates satisfy local feasibility at every segment and are therefore globally feasible plans. Our algorithm benefits from adaptive compute at inference time, with the flexibility to scale the batch size B and the number of resampling steps U for problems with longer horizons and larger search spaces.
this section cite: []

Section: EXPERIMENTAL RESULTS: ROBOTIC PLANNING
In this section, we evaluate the performance of CDGS for long-horizon robotic planning. For all the experiments, we represent inputs with a low-dimensional state-space of the system comprising the pose of the end-effector and the objects in the scene in the global frame of reference. For real-world evaluations, we obtain the pose of the objects through perception, more details in App. H.
CDGS can solve learning from play and stitching problems efficiently. We evaluate CDGS for sequential-decision making tasks using the OGBench Maze and Scene task suite [48], which includes PointMaze and AntMaze along with five tasks for Scene where a robot must manipulate objects (a drawer, sliding window, and cube) to reach a goal state. The primary challenge is learning from small maze trajcetories or unstructured play data during training, which does not directly solve the target tasks. The diversity of the unstructured plans makes the local distributions highly multimodal. We hypothesize that CDGS is an ideal method for this problem statement because it can compose short-horizon plans into meaningful long-horizon plans.
CDGS uses a Diffuser [24] to learn the distribution of local plans (up to 4 secs of trajectory at 20 Hz) represented as a sequence of states and actions y = {s 1 , a 1 ,...,s h , a h } and then composes them at inference for a given goal state to sample up to 10 secs of motion plans ! = {s i , a i } H i=1 (h < H). We compare the performance of CDGS with inverse reinforcement learning baselines from OGBench, including GCBC [41,17], GCIVL, GCIQL [31], and HIQL [49], with results presented in Tab. 1. In addition we also include compositional generative baselines like GSC [43] and CompDiffuser [40]. It should be noted that CDGS with resampling and pruning can scale the performance of naïve compositional sampling (GSC), in a training-free manner, to an extent that beats baselines like Com-pDiffuser [40] that use overlap information while training and learn an overlap conditioned score function. Finally, we also validate CDGS on composite ball reaching and ball carrying trajectory stitching of AntSoccer in OGBench and show the results in Tab. 2.
CDGS can solve hybrid-planning problems. Task and Motion Planning (TAMP) decomposes robotic planning into a symbolic search for a sequence of discrete high-level skills (e.g., pick, place, pull) followed by low-level motion planning for each skill [14]. Specifically, we formu-    3. Note that while GSC (Original) [43] leverages skill-level expert diffusion models and oracle task plan, in our case it represents naïve compositional sampling with a unified model (w/o oracle task plan).
CDGS's performance scales with compute. We hypothesize that CDGS has adaptive inferencetime compute, meaning that it benefits from more compute on harder problems. We validate this hypothesis on our most challenging TAMP tasks with a planning horizon of 7. We find that increasing batch size (B) and number of resampling steps (U) increases the task planning success Fig. 5(c) and motion planning success Fig. 5(d) of CDGS. Interestingly, we find that neither increasing B nor U on their own is sufficient for overall motion planning success. Thus, both resampling and pruning are essential for long-horizon tasks, as evidenced by the significant improvement of CDGS ( Tab. 3).
this section cite: ['b47', 'b23', 'b40', 'b16', 'b30', 'b48', 'b42', 'b39', 'b39', 'b13', 'b42']

Section: CDGS FOR LONG CONTENT GENERATION
We formulate CDGS with specific design choices that enable (i) efficient message passing for global consistency and (ii) pruning denoised paths that lead to incoherent sequences. While these mechanisms are essential for long-horizon planning, we investigate their broader applicability, particularly in long-content generation tasks such as text-to-image (T2I) and text-to-video (T2V), which require spatial and temporal coherence over extended horizons. Our framework demonstrates effective improvement in long-horizon content generation.
CDGS enables coherent panoramic image generation via stitching. We evaluate CDGS on panoramic synthesis by composing multiple image patches. A panorama ! is represented as a sequence of small images y, each split into three overlapping patches y = (x 1 , x 2 , x 3 ). Using Stable Diffusion-2.0 [51], we generate up to 512 ⇐ 4608 panoramas by stitching 512 ⇐ 512 images. We compare against (i) Multi-Diffusion (MD) [4], which averages scores across overlaps (image-domain analogue of GSC [43]), and (ii) Sync-Diffusion (SD) [33], which enforces LPIPS-based perceptual guidance [67]. As shown in Tab. 4, CDGS matches SD without explicit perceptual loss, indicating effective message passing for global style and perceptual transfer while maintaining prompt alignment (CLIP [50]). Qualitative samples are shown in Fig. 6, with more details in App. A.
CDGS can sample temporally-consistent longer videos. We follow a setup similar to panorama generation, composing shorter clips along the temporal axis for long-video generation. When short   4] and SD [33]. We show qualitative intuition behind global coherence and local feasibility: while SD generates smooth panoramas, they fail to satisfy the global context (mountain peak with skiers), on the other hand, MD follows the global context (beach in La La Land style) but fails to exhibit local consistency. CDGS excels at both.
Metric GSC/Multi-Diffusion Sync-Diffusion CDGS w/o PR CDGS Intra-LPIPS ⇒ 0.72 ±0.08 0.58 ±0.06 0.61 ±0.08 0.59 ±0.04 Intra-Style-L(⇐10 →2 ) ⇒ 2.96 ±0.24 1.39 ±0.12 1.97 ±0.08 1.38 ±0.03 Mean-CLIP-S ⇑ 31.77 ±2.14 31.77 ±2.14 31.71 ±2.34 32.51 ±2.66
Table 4: Quantitative comparison of panorama generation. We generate 1000 panoramas of dimensions 512 ⇐ 4608 using 14 prompts and compare different methods based on their perceptual similarity (LPIPS [67]), style similarity (Style-loss [15]), and prompt alignment (CLIP score [50]).
sequences of frames are stitched to make a long video, a key challenge is maintaining subject consistency and minimizing temporal artifacts. We use CogVideoX-2B [62] as the base model, capable of generating ↘ 50-frame videos, and extend it to up to 350 frames at 720p resolution. We use six prompts to generate videos with naïve composition (GSC/Gen-L-Video [56] equivalent), compositional diffusion with resampling, and CDGS. The results are evaluated with VBench [22] for temporal consistency, subject fidelity, visual quality, and alignment with the prompt (refer Tab. 5). Qualitative analysis in Fig. 7 clearly shows the multimodal problem where multiple local plans allow satisfying the global context, but with CDGS's effect local-to-global message passing, we see an improvement in subject consistency and temporal smoothness. This comes at a minor aesthetic degradation-a tradeoff commonly observed in long-video generation models.
this section cite: ['b50', 'b3', 'b42', 'b32', 'b66', 'b49', 'b3', 'b32', 'b66', 'b14', 'b49', 'b61', 'b55', 'b21']

Section: RELATED WORK

this section cite: []

Section: Long-horizon content generation
There are many approaches to generating long-horizon content like panoramas and long videos [26,38,7,19]. Some assume access to long-horizon training data for end-to-end training [16,5,60,61], while others with weaker assumptions about training data will compose the outputs of short-horizon models through outpainting [59,28] or stitching [66,29,34,32,47,6,40]. Our method belongs to the latter, enabling generalization to longer horizons than seen during training.
Generative planning. Generative models such as diffusion models [53,21] are widely used for planning [25,3,8,35,40], though they struggle with task lengths beyond their training data. Recent works including Diffusion-CCSP [63], GSC [44], and GFC [45] have explored compositional sampling [37,12,66] but they sidestep the mode-averaging problem via additional mode supervision in the form of task skeletons or constraint graphs. In contrast, our approach directly addresses the mode-averaging problem to generate goal-directed long-horizon plans from short-horizon models.
Inference-time compute. Scaling inference-time computation is a powerful strategy for improving the performance of generative models [58,46]. For diffusion models [54,27], recent work has shown the efficacy of scaling inference-time compute through verifier-guided search during the denoising process [42,52,65,68,69]. Our algorithm differs in that it addresses the unique limitation of mode-averaging when sampling from a compositional chain of distributions.
this section cite: ['b25', 'b37', 'b6', 'b18', 'b15', 'b4', 'b59', 'b60', 'b58', 'b27', 'b65', 'b28', 'b33', 'b31', 'b46', 'b5', 'b39', 'b52', 'b20', 'b24', 'b2', 'b7', 'b34', 'b39', 'b62', 'b43', 'b44', 'b36', 'b11', 'b65', 'b57', 'b45', 'b53', 'b26', 'b41', 'b51', 'b64', 'b67', 'b68']

Section: CONCLUSION
We introduce CDGS, a framework integrating compositional diffusion with guided search to generate long-horizon sequences with short-horizon models. By embedding search within the denoising process, CDGS can handle composing highly multimodal distributions and sample solutions that are both globally coherent and locally feasible. Qualitative and quantitative results suggest that CDGS is a general pathway for extending the reach of generative models beyond their training horizons across robotic planning, panoramic images, and video generation.
this section cite: []

Section: LIMITATIONS
While CDGS demonstrates strong performance in long-horizon goal-directed planning, it relies on a few simplifying assumptions that also suggest directions for future work. We assume the ability to specify a goal state, which simplifies planning but can be naturally extended to goal-generation or classifier-guided goal-conditioning methods [13]. Similarly, we generate plans for a fixed horizon, yet the framework can handle arbitrary horizons given the same start and goal, enabling selection among multiple candidate plan lengths. Finally, long-horizon dependencies are communicated through score averaging and resampling between adjacent skills; more sophisticated messagepassing or attention-based mechanisms could improve efficiency and coherence across entire plans. These assumptions keep the problem tractable while providing a flexible foundation for extending CDGS to more general and complex planning scenarios.
this section cite: ['b12']

Section: REPRODUCIBILITY STATEMENT
We are committed to ensuring that all the results presented in this paper are reproducible. To this end, we have provided pseudocodes in the paper and released the official code base through our project website: https://cdgsearch.github.io/. We have also provided the hyperparameters
table for motion planning (refer App. F), for image generation (refer App. J) and video generation (refer App. K). Apart from this our content-generation experiments use open-source models like Stable-Diffusion-2 (refer https:  //huggingface.co/stabilityai/stable-diffusion-2) and CogVideoX-2B (refer https://huggingface.co/zai-org/CogVideoX-2b). For all other robotics setup, we provide more information through appendix and our project website.
this section cite: []

Section: LLM USAGE
LLMs were not used in any manner for conceptualization of the idea, key contributions of the proposed work and finding relevant prior woks.
this section cite: []

Section: References
Ref_id:b0 Title: Taps: Task-agnostic policy sequencing Year: (2022)
Ref_id:b1 Title: Stap: Sequencing taskagnostic policies Year: (2023)
Ref_id:b2 Title: Is conditional generative modeling all you need for decision making? Year: (2023)
Ref_id:b3 Title: Multidiffusion: Fusing diffusion paths for controlled image generation Year: (2023)
Ref_id:b4 Title: Diffusion forcing: Next-token prediction meets full-sequence diffusion Year: (2024)
Ref_id:b5 Title: Extendable long-horizon planning via hierarchical multiscale diffusion Year: (2025)
Ref_id:b6 Title: Seine: Short-to-long video diffusion model for generative transition and prediction Year: (2023)
Ref_id:b7 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: ()
Ref_id:b8 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b9 Title: Position: Compositional generative modeling: A single model is not all you need Year: (2024)
Ref_id:b10 Title: Compositional visual generation with energy based models Year: (2020)
Ref_id:b11 Title: Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc Year: (2023)
Ref_id:b12 Title: Automatic goal generation for reinforcement learning agents Year: (2018)
Ref_id:b13 Title: Integrated task and motion planning Year: (2021)
Ref_id:b14 Title: Image style transfer using convolutional neural networks Year: (2016)
Ref_id:b15 Title: Long video generation with time-agnostic vqgan and time-sensitive transformer Year: (2022)
Ref_id:b16 Title: Learning to reach goals via iterated supervised learning Year: (2019)
Ref_id:b17 Title: Out-of-distribution detection with a single unconditional diffusion model Year: (2024)
Ref_id:b18 Title: Streamingt2v: Consistent, dynamic, and extendable long video generation from text Year: (2025)
Ref_id:b19 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b20 Title: Denoising diffusion probabilistic models. Advances in neural information processing systems Year: (2020)
Ref_id:b21 Title: Vbench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b22 Title: Estimation of non-normalized statistical models by score matching Year: (2005)
Ref_id:b23 Title: Planning with diffusion for flexible behavior synthesis Year: (2022)
Ref_id:b24 Title: Planning with diffusion for flexible behavior synthesis Year: (2022-07)
Ref_id:b25 Title: Cubediff: Repurposing diffusion-based image models for panorama generation Year: (2025)
Ref_id:b26 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b27 Title: Fifo-diffusion: Generating infinite videos from text without training Year: (2024)
Ref_id:b28 Title: Tuningfree multi-event long video generation via synchronized coupled sampling Year: (2025)
Ref_id:b29 Title: Probabilistic graphical models: principles and techniques Year: (2009)
Ref_id:b30 Title: Offline reinforcement learning with implicit q-learning Year: (2021)
Ref_id:b31 Title: State-covering trajectory stitching for diffusion planners Year: (2025)
Ref_id:b32 Title: Syncdiffusion: Coherent montage via synchronized joint diffusions Year: (2023)
Ref_id:b33 Title: Diffstitch: Boosting offline reinforcement learning with diffusion-based trajectory stitching Year: (2024)
Ref_id:b34 Title: Hierarchical diffusion for offline decision making Year: (2023-07)
Ref_id:b35 Title: Text2motion: From natural language instructions to feasible plans Year: (2023)
Ref_id:b36 Title: Compositional visual generation with composable diffusion models Year: (2022)
Ref_id:b37 Title: Freelong: Training-free long video generation with spectralblend temporal attention Year: (2024)
Ref_id:b38 Title: Repaint: Inpainting using denoising diffusion probabilistic models Year: (2022)
Ref_id:b39 Title: Generative trajectory stitching through diffusion composition Year: (2025)
Ref_id:b40 Title: Learning latent plans from play Year: (2020)
Ref_id:b41 Title: Inference-time scaling for diffusion models beyond scaling denoising steps Year: (2025)
Ref_id:b42 Title: Generative skill chaining: Long-horizon skill planning with diffusion models Year: (2023)
Ref_id:b43 Title: Generative skill chaining: Long-horizon skill planning with diffusion models Year: (2023)
Ref_id:b44 Title: Generative factor chaining: Coordinated manipulation with diffusion-based factor graph Year: (2024)
Ref_id:b45 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b46 Title: T-stitch: Accelerating sampling in pre-trained diffusion models with trajectory stitching Year: (2024)
Ref_id:b47 Title: Benchmarking offline goal-conditioned rl Year: (2024)
Ref_id:b48 Title: Hiql: Offline goalconditioned rl with latent states as actions Year: (2024)
Ref_id:b49 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b50 Title: High-resolution image synthesis with latent diffusion models Year: (2022-06)
Ref_id:b51 Title: A general framework for inference-time scaling and steering of diffusion models Year: (2025)
Ref_id:b52 Title: Deep Unsupervised Learning using Nonequilibrium Thermodynamics Year: (2015-11)
Ref_id:b53 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b54 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b55 Title: Gen-l-video: Multi-text to long video generation via temporal co-denoising Year: (2023)
Ref_id:b56 Title: Apriltag 2: Efficient and robust fiducial detection Year: (2016)
Ref_id:b57 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b58 Title: Yuejian Fang, and Nan Duan. Nuwa-infinity: Autoregressive over autoregressive generation for infinite visual synthesis Year: (2022)
Ref_id:b59 Title: Progressive autoregressive video diffusion models Year: (2025)
Ref_id:b60 Title: Long video diffusion generation with segmented cross-attention and content-rich video data curation Year: (2025)
Ref_id:b61 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2024)
Ref_id:b62 Title: Compositional diffusion-based continuous constraint solvers Year: (2023)
Ref_id:b63 Title: Constructing free-energy approximations and generalized belief propagation algorithms Year: (2005)
Ref_id:b64 Title: Doojin Baek, Yoshua Bengio, and Sungjin Ahn. Monte carlo tree diffusion for system 2 planning Year: (2025)
Ref_id:b65 Title: Diffcollage: Parallel generation of large content with diffusion models Year: (2023-06)
Ref_id:b66 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b67 Title: Test-time scalable mctsenhanced diffusion model Year: (2025)
Ref_id:b68 Title: Inference-time scaling of diffusion models through classical search Year: (2025)
Ref_id:b69 Title: Viola: Imitation learning for visionbased manipulation with object proposal priors Year: ()
