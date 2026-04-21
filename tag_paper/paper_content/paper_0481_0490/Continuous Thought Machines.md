Title: Continuous Thought Machines
Abstract: 

Section: 
Biological brains demonstrate complex neural activity, where neural dynamics are critical to how brains process information. Most artificial neural networks ignore the complexity of individual neurons . We challenge that paradigm. By incorporating neuron-level processing and synchronization, we reintroduce neural timing as a foundational element. We present the Continuous Thought Machine (CTM), a model designed to leverage neural dynamics as its core representation. The CTM has two innovations: (1) neuron-level temporal processing, where each neuron uses unique weight parameters to process incoming histories; and
(2) neural synchronization as a latent representation. The CTM aims to strike a balance between neuron abstractions and biological realism. It operates at a level of abstraction that effectively captures essential temporal dynamics while remaining computationally tractable. We demonstrate the CTM's performance and versatility across a range of tasks, including solving 2D mazes, ImageNet-1K classification, parity computation, and more. Beyond displaying rich internal representations and offering a natural avenue for interpretation owing to its internal process, the CTM is able to perform tasks that require complex sequential reasoning. The CTM can also leverage adaptive compute, where it can stop earlier for simpler tasks, or keep computing when faced with more challenging instances. The goal of this work is to share the CTM and its associated innovations, rather than pushing for new state-of-the-art results. To that end, we believe the CTM represents a significant step toward developing more biologically plausible and powerful artificial intelligence systems. We provide an accompanying interactive online  demonstration and an extended technical report.   (b) The CTM looks around to build up its prediction, effectively tracing an intuitive path by synchronizing its neurons to attend dynamically. Arrows trace the average weighting over internal ticks, exemplifying a complex path that emerges without any training signal. We discuss more interesting emergent properties of the CTM in Appendix I. Video demonstrations are here.
this section cite: []

Section: Introduction
Biological brains exhibit complex time-dependent neural dynamics, but artificial neural networks (NNs) intentionally abstract away the precise timing and interplay of neuron interactions to facilitate large-scale deep learning [1,2,3]. While enabling significant advancements over the years, these simplifications deviate from fundamental biological neural computation principles. Emulating the temporal aspects of neural dynamics present in brains remains challenging. Consequently, modern NNs prioritize simplicity and computational efficiency over strict emulation. This abstraction, though task-performant, contributes to a gap between flexible human cognition and current AI capabilities, suggesting missing fundamental components, potentially related to temporal processing [4,5,6].
Despite its outstanding performance, modern AI lacks the flexibility, efficiency, fluidity, generalization capabilities, and common sense of human intelligence, which operates in an open world where learning and adaptation are tied to the arrow of time [5,7,6,8]. We argue that incorporating time as part of neural computation is crucial for advancing AI [9,10]. We introduce the Continuous Thought Machine (CTM), a model explicitly incorporating neural dynamics over time. Our contributions are:
1. The CTM architecture using an internal dimension for modeling the temporal evolution of neural activity, neuron-level models (NLMs) as a more biologically plausible mid-level abstraction of neurons that unfold neural dynamics , and the use of neural synchronization directly as the representation (implemented via temporal correlations between neuron-level activity; Section 3.4) for observation and prediction, making neural dynamics the core operating principle.
2. An exposition of the capabilities unlocked by the CTM, including strong performance on sequential reasoning tasks (Figure 1) , native adaptive compute time, natural and interpretable behaviors such as 'looking around' images before predicting (Figure 2) , and learning algorithmic solutions, opening up opportunities to the AI community for new research.
The CTM learns to use neural synchronization as its latent representation, distinguishing it from existing work that explores synchrony as emergent properties for post-hoc use [11,12]. This representation is distinct from the common static 'snapshot' representations used in most modern NNs as it directly encodes the temporal interplay of neural dynamics.
Recurrence and Reasoning. Recurrence is a strong contender for extending model complexity beyond current scaling limitations [13,14,15]. We posit that recurrence, while essential, is merely one piece of the puzzle. The temporal dynamics unlocked by recurrence are equally crucial. We demonstrate in this paper that neural dynamics can be leveraged to build a new kind of neural network with surprising capabilities. We show how the CTM navigates complex 2D mazes by forming internal maps without positional encodings (Section 4), learns to 'look around' (without any signal to do so) when classifying images and exhibits native adaptive computation time as a side-effect (Section 5), and utilizes its dynamic representations for tasks requiring memory and sequential reasoning (Section 6). These capabilities emerge from the same core architecture applied to different tasks, showcasing its versatility and trainability. We believe that the CTM represents a step towards bridging the gap between powerful modern AI and biological plausibility.
The remainder of this paper details related work (Section 2), describes the CTM (Section 3), evaluates core capabilities on 2D mazes, ImageNet-1K classification, and parity computation (Sections 4 to 6), summarizes further experiments and applications (Section 7), and discusses findings (Section 8).
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b4', 'b6', 'b5', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14']

Section: Related Work
The CTM uses neural timing and synchronization as core computational principles. This positions it relative to, yet distinct from, several lines of research.
Adaptive Computation. Many approaches achieve adaptive computation via explicit mechanisms.
Early-exit networks [16] use intermediate classifiers for early termination. PonderNet [17] and Adaptive Computation Time (ACT) [18] introduce learnable halting modules governing recurrent steps. More recent methods like AdaTape [19] dynamically extend input sequences, while Sparse Universal Transformers (SUT) [20] combine recurrent weight sharing with dynamic halting and Mixture-of-Experts. In contrast, the CTM's adaptive processing (varying internal ticks per input based on certainty and loss dynamics; Section 3.5) emerges naturally from its core architecture, driven by the unfolding of its internal neural dynamics without dedicated halting components.
Iterative and Recurrent Reasoning. The CTM's internal ticks facilitate iterative refinement, akin to models promoting internal computational steps. For instance, Quiet-STaR [21] uses hidden rationale generation in language models, and Recurrent Independent Mechanisms (RIMs) [22] employ modular, asynchronous sub-networks for multi-step reasoning. While Recurrent Models of Visual Attention (RAM) [23] also leveraged recurrence for sequential processing of visual glimpses, the CTM's novelty lies in generating internal neural dynamics from neuron-level histories across a decoupled time dimension and then utilizing the explicit temporal patterns of neural synchronization as its primary representation. This contrasts with RAM's focus on perceptual decision-making from external glimpses or models relying solely on a final recurrent state.
this section cite: ['b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22']

Section: Biologically Inspired Neural Dynamics.
There is growing interest in more biologically plausible neural computation [24]. Examples include Liquid Time-Constant Networks (LTCNs) [25] with neurons governed by time-varying differential equations, and various Spiking Neural Network (SNN) paradigms that inherently use discrete, timed events, with recent work also exploring synchronization mechanisms [26,27]. The CTM draws inspiration from temporal coding and neural synchrony, but uses: (1) neuron-level models (NLMs) to process a history of continuous-valued pre-activations to generate complex dynamics, and (2) neural synchronization as the primary latent representation for attention and output. While inspired by principles like spike-timing and synchrony, CTM abstracts these-focusing on local temporal integration and population-level synchronization-into a tractable, differentiable framework suitable for gradient-based deep learning, rather than replicating detailed biophysics. This situates the CTM alongside, yet distinct from, extensive work on models such as Liquid State Machines [28], and diverse SNNs that exploit precise spike timing for computation or employ specialized learning rules [29,30,31,32,33]. These latter models often emphasize event-driven dynamics, explore non-differentiable computation, or focus on online learning. The CTM offers a complementary direction, retaining inspiration from biological timing while ensuring compatibility with established deep learning training paradigms.
this section cite: ['b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32']

Section: Synchronization.
Reichert & Serre [11] proposed a model where synchronization emerges from interactions among complex-valued neurons, serving as a gating mechanism that modulates information flow and enables post-hoc grouping of neurons for tasks like object segmentation. Unlike CTM, however, their model does not use synchrony as a learned latent representation during computation. Other approaches in complex-valued neural networks [12] employ synchronization from a control-theoretic perspective, aiming to stabilize or coordinate networks via externally enforced synchrony. In contrast, CTM integrates synchronization intrinsically, optimizing neural phase relationships during training to encode task-relevant representations. This positions CTM as a computationally grounded model of synchrony, fundamentally distinct from prior works that treat synchrony as a control objective.
this section cite: ['b10', 'b11']

Section: Method
Figure 3: CTM architecture overview. Key components include: 1 Synapse model generating pre-activations from prior post-activations z t and attention output o t . 2 History of pre-activations A t . 3 Neuron-level models (NLMs) processing A t d to produce 4 post-activations z t+1 d . 5 History of post-activations Z t . 6 Neural synchronization matrix S t computed from Z t . 7 Selected neuron pairs from S t form 8 latent representations used for 9 outputs y t and attention queries q t . 10  Attention output o t is concatenated with z t+1 for the next internal tick. Owing to the inherent difficulty in visualizing a dynamic, time-based architecture, we include the supplementary video 'arch.mp4' (hosted here too) that visualizes functional data flow.
The Continuous Thought Machine (CTM) is a neural network architecture that explicitly incorporates neural dynamics as a core component. Figure 3 1 → 10 and pseudocode in Listing 1 illustrate the CTM's flow. The CTM differs from other recurrent architectures [34,35,36,18,37] in two ways: (1) it applies neuron-level models (NLMs), each with private weights, to histories of pre-activations to produce complex neuron-level activity (Section 5); and (2) it uses neural synchronization directly as the latent representation for modulating data and producing outputs (Section 3.4).
this section cite: ['b33', 'b34', 'b35', 'b17', 'b36']

Section: Continuous Thought: The Internal Sequence Dimension
The CTM uses an internal dimension t ∈ {1, . . . , T }, decoupled from data dimensions. This timeline of internal ticks [34,35,36,37] enables iterative refinement of representations, even for static data. Unlike conventional sequential models that process data-inherent sequences, the CTM along a self-generated timeline of 'thought steps' that unfolds neural dynamics for downstream use.
this section cite: ['b33', 'b34', 'b35', 'b36']

Section: Recurrent Weights: Synapses
A 1 synapse model, f θsyn , interconnects neurons in a shared D-dimensional latent space, z t ∈ R D . We found a U-NET-esque [38] MLP (details in Appendix C.1) performs best, suggesting benefit from deeper and more flexible synaptic computation. It produces pre-activations, a t :
a t = f θsyn (concat(z t , o t )) ∈ R D ,(1)
where o t is attention output (Section 3.4). The M most recent pre-activations form a 2 history A t :
A t = a t-M +1 a t-M +2 • • • a t ∈ R D×M .(2)
Initial pre-activation history and z t=1 are learnable parameters. We found that setting M ≈ 10 -100 was effective during our initial exploration.
this section cite: ['b37']

Section: Privately-Parameterized Neuron-Level Models (NLMs)
Each neuron d ∈ {1, . . . , D} has a 3 privately parameterized NLM, g θ d (depth 1 MLP of width d hidden ), processing its M -dimensional pre-activation history A t d to produce 4 post-activations:
z t+1 d = g θ d (A t d ).(3)
The full set of post-activations z t+1 is 10 concatenated with attention output, o t , and fed into the synapse model f θsyn for the next internal tick, t + 1. See Listing 2 for pseudo-code.
this section cite: []

Section: Neural Synchronization: Modulating Data and Outputs
Synchronization is inspired by biological brains [39]. The CTM modulates data via the synchronization of neural activityfoot_0 . We first collect post-activations into 5 a (non-fixed length) history:
Z t = z 1 z 2 • • • z t ∈ R D×t .(4)
We define neural synchronization is defined as the 6 inner product of the histories of each neuron:
S t = Z t • (Z t ) ⊺ ∈ R D×D .(5)
this section cite: ['b38']

Section: Neuron Pairing: A Sub-sampling Approach
Since S t scales with O(Dfoot_1 ) it can grow very large. We sample (i, j) neurons at the start of training by randomly selecting D out and D action pairs for two synchronization representations, S t out ∈ R Dout and S t action ∈ R Daction . These are projected by W out and W in for outputs y t and attention queries q t :
y t = W out • S t out ,(6)
q t = W in • S t action .(7)
We use standard cross attention [40] for o t :
o t = Attention(Q = q t , KV = FeatureExtractor(data)), (8
)
where a FeatureExtractor (e.g., ResNet [41]) provides keys/values. o t ∈ R dinput is then concatenated with z t+1 . This process, including learnable temporal scaling, is shown in Listing 3.
this section cite: ['b39', 'b40']

Section: Scaling Temporal Dependency.
To modulate the influence of past activity on S t , we introduce learnable exponential decay factors r ij ≥ 0 for each neuron pair ij. The rescaling vector over t is:
R t ij = [exp(-r ij (t -1)) exp(-r ij (t -2)) • • • exp (0)] ⊺ ∈ R t .(9)
The rescaled synchronization is (see Appendix H for efficient recursive computation):
S t ij = (Z t i ) ⊺ • diag(R t ij ) • Z t j t τ =1 R t ij τ .(10)
Higher r ij bias towards recent ticks (r ij = 0 means no decay). Learnable decay rates r ij allow the CTM to modulate synchronization across multiple time scales 2 . Details on neuron-pair sub-sampling strategies, including recovering snapshot dependencies, are in Appendix C.2.
this section cite: []

Section: Loss Function: Optimizing Across Internal Ticks
The CTM produces outputs y t ∈ R C (e.g., class probabilities) at each internal tick t. We compute a loss L t = CrossEntropy(y t , y true ) and certainty C t (1 -normalized entropy) per tick. For each forward pass we select to ticks: 2. the point of maximum certainty: t 2 = argmax(C), to ensure certainty aligns with correctness.
The final loss for optimizing θ syn and θ d=1...D is:
L = L t1 + L t2 2 .(11)
Since t 1 and t 2 are dynamically defined per data point, the CTM can attribute variable compute (internal ticks) to different data points as needed without explicit restrictions on which tick should be used in the loss function. This effectively implements native adaptive computation [18] as opposed to a post-hoc addition. We give pseudo-code in Listing 4.
this section cite: ['b17']

Section: Experimental Evaluation
The following sections present a focused evaluation of the CTM on tasks that highlight its core principles: neuron-level temporal processing and neural synchronization as a direct latent representation. We aim to demonstrate how neural dynamics enables the CTM to implement complex reasoning or adaptive processing, while yielding interpretable strategies. We prioritize depth in three key experiments: 2D maze navigation, ImageNet-1K classification, and parity computation. We also summarize and highlight additional experiments demonstrating the CTM's broader capabilities.
this section cite: []

Section: 2D Mazes: Complex Sequential Reasoning and Internal World Models
In this section we analyze the CTM's capacity for sequential reasoning, planning, and spatial understanding using a challenging phrasing of the 2D maze navigation task. Solving mazes can be easy with the right inductive bias. For example, matching the output dimensions to the input space, a model can perform binary classification at each location. Such a setup is amenable to machines by design, as they can learn iterative algorithmic solutions [37,42], but this is not how humans solve mazes.
Setup. The setup of our maze task deviates from the norm, specifically to necessitate the formation of an internal world model [43] by (1) requiring a direct sequence-of-actions output and (2) disallowing positional embeddings in the visual input. This requires a model to build its own spatial representation via observation (see Appendix D.6 for further discussion). We compare the CTM against LSTM and feed-forward (FF) baselines. For the results that follow, we trained a CTM, LSTMs (1, 2, and 3 layers), and a FF baseline to predict up to 100 steps down the path of 39 × 39 mazes, where predictions took the form of a sequence of classes for left, right, up, down, and wait, using 'wait' to pad instances shorter than 100 steps. For the CTMs and LSTM baselines, we used 75 internal ticks, but LSTM stability issues meant that using 50 internal ticks yielded superior performance, so we report these too. In each case we used a automatic curriculum approach when training (see details in Appendix D.3). Appendices D.2 and D.4 detail hyperparameters for the CTM and baselines.
this section cite: ['b36', 'b41', 'b42']

Section: Results
The CTM significantly outperforms the baselines in solving these mazes, demonstrating superior trainability and generalization to longer paths (Figure 4). The FF model and LSTMs struggled to learn effectively or overfit (see Appendix D.5), whereas the CTM achieved high accuracy. This suggests that the CTM's architecture, particularly its use of neural dynamics and synchronization, is well-suited for tasks requiring robust internal state maintenance and planning.
this section cite: []

Section: Demonstrations and Generalization
Qualitative analysis shows the CTM methodically tracing paths (Figures 1a and 1b; supplementary video 'mazes.mp4'), exhibiting emergent behavior such as continuing to explore paths beyond its training horizon. This suggests the CTM learns a general procedure rather than merely memorizing. Furthermore, the CTM, trained on 39 × 39 mazes, generalizes effectively to longer paths and larger 99 × 99 mazes (Figure 4c) by re-applying its learned policy, as shown in Figure 1c (see supplementary videos 'maze-large1.mp4' to 'maze-large4.mp4' for examples). Crucially, this CTM is not using any positional embedding, meaning that in order for it to follow a path through the maze it must craft the Figure 4: CTM versus baselines on 2D mazes. The CTM demonstrates superior trainability compared to baselines, yielding higher accuracy for longer paths. Using iterative re-applications, we show in (b) that the CTM can generalise to longer paths and bigger mazes. See Appendix D.5 for loss curves.
cross-attention query by 'imagining' the future state of the maze: a process known as 'episodic future thinking' [44] in humans. Appendix I discuss some of the emergent properties we observed.
this section cite: ['b43']

Section: ImageNet-1K Classification: Adaptive Processing and Emergent Dynamics
We evaluate the CTM on ImageNet-1K to understand its internal processing dynamics when trained to solve a standard classification task . We are not yet aiming for state-of-the-art accuracy (with 50 internal ticks and a ResNet-152 backbone: 72.47% top-1, 89.89% top-5 on uncropped data). Since the CTM uses new neural computation principles it would require a thorough hyperparameter search to find the optimal settings, and that is outside the scope of this work. Instead, we focus on how the CTM leverages neural dynamics (setup details in Appendix E.1) as a new mechanism for reasoning.  The CTM exhibits adaptive computation: it can halt internal ticks based on prediction certainty. For instance, setting a certainty threshold of 0.8 (Figure 5a) means that a user could halt compute for the majority of instances after fewer than 10 of 50 internal ticks. This is a consequence of internal recurrence couple with our novel loss function. The CTM also demonstrates excellent calibration (Figure 5b) as an emergent property of its iterative refinement process ( Appendix E.3).
this section cite: []

Section: Adaptive Computation and Calibration

this section cite: []

Section: Reasoning sequentially about static images
The CTM exhibits diverse temporal dynamics (Figure 2a), the synchronization of which is the representation with which it observes data and forms predictions. We show in Figure 2b how the CTM learns to 'look around' an image in order to gather information and make a prediction. It does this entirely without prompting or any guide, implementing computationally beneficial adaptive compute in an intuitive fashion . This internal process can even manifest emergent phenomena like low-frequency traveling waves [45] across UMAP-projected neuron activations (see supplementary video 'umap.mp4'). Unpacking every interesting facet of these attention map progressions is simply infeasible in a static form; we encourage viewing supplementary video 'imagenet.mp4' for demonstrations of the CTM 'gazing' in a manner not quite entirely unlike how humans might look around images. Appendix E.4 has further demos and UMAP visualizations. These observations underscore that the CTM solves classification by leveraging an internal, dynamic reasoning process, a departure from typical feed-forward approaches.
this section cite: ['b44']

Section: Parity: Learning Sequential Algorithms and Interpretable Strategies
To test the CTM's ability to learn algorithmic procedures and develop interpretable strategies, we use a cumulative parity task: given a 64-length binary sequence, predict the parity at each position (Figure 6a). Unlike prior work focusing on final parity [18], our setup requires the model to output sequences at each internal tick, enabling us to examine how the full output evolves across ticks and throughout training. Setup details are in Appendix F.1.
this section cite: ['b17']

Section: Results and Learned Strategies
Input Target 0 50000 100000 150000 200000 Training Iterations 50 60 70 80 90 100 Accuracy (%) 1 Iters. 10 Iters. 25 Iters. 50 Iters. 75 Iters. 100 Iters. CTM LSTM (b) Accuracies during training.
0 25 50 75 100 Internal Ticks 50 60 70 80 90 Accuracy (%) CTM LSTM (c) Accuracies versus ticks.
0 20 40 60 80 Internal Tick 0 10 20 30 40 50 60 Sequence Index 0.00 0.05 0.10 0.15 0.20 0.25 Attention Weight (d) Example attention strategy. The CTM's accuracy improves with more internal ticks, significantly outperforming parametermatched LSTMs, which struggled with stability and performance (Figure 6b). CTMs with 75 and 100 ticks could achieve perfect accuracy in some seeded runs. Figure 6d shows how the attention shifts over the input data, and Figure 7 shows a specific demonstration (4 of 8 attention heads), revealing a distinct and interpretable strategy. Which specific 'style' of solution depends on the configuration and seed, so we show other examples and analyses in Appendix F.2). Crucially, this experiment demonstrates that the CTM can learn to form and follow an internal strategy for an algorithmic task. See supplementary material 'parity.mp4' for video format.
this section cite: []

Section: Other Experiments and Analyses
We also evaluated the CTM in a number of other settings in order to probe its functionality and versatility. Owing to space constraints, we provide the details of these additional experiments in the appendices (referenced below). In summary, these additional experiments investigated:
CIFAR-10 Classification Compared to Humans (Appendix G.1): The CTM, feed-forward, and LSTM baselines were evaluated on CIFAR-10, with results compared against human data for difficulty and uncertainty. The CTM demonstrated good model calibration and alignment with humans.
this section cite: []

Section: CIFAR-100 Ablation Studies (Appendix G.
2): We investigated the impact of model width and the number of internal ticks. We found that the diversity of neural activity are functions of these. Wider models tended to exhibit more varied neural dynamics. Using more internal ticks allowed the CTM to engage in extended processing, sometimes revealing distinct computational phases.
this section cite: []

Section: Neuron-Level Models and Synchronization Ablations (Appendix G.3):
We compared the CTM to parameter-matched variants without NLMs and without synchronization, as well as an LSTM with synchronization. The results show that the combination of neuron-level models and synchronization as a representation is key to the success of the CTM.
Sorting Real Numbers (Appendix G.4): The CTM was tasked with sorting sequences of 30 real numbers, outputting sorted indices sequentially using a Connectionist Temporal Classification (CTC) loss [46]. This experiment showed that the CTM could learn an algorithmic sorting procedure and exhibited adaptive computation by varying its internal processing duration ("wait times") based on characteristics of the input sequence, such as the difference between successive values.
Q&A MNIST (Appendix G.5): In this task, the CTM processed sequences of MNIST digits followed by index and operator embeddings to perform multi-step modular arithmetic. This investigation highlighted the CTM's capacity for memory and retrieval, using its synchronization mechanism to recall digit information beyond the immediate history window of individual neuron-level models, and to generalize to longer computational sequences than seen during training.
Reinforcement Learning (Appendix G.6): The CTM was adapted for reinforcement learning in several partially observable Markov decision processes (POMDPs), including classic control (CartPole, Acrobot) and grid-world navigation (MiniGrid Four Rooms). This demonstrated the CTM's applicability to sequential decision-making in continuous interaction settings, where it achieved performance comparable to LSTM baselines while developing richer internal state dynamics.
this section cite: ['b45']

Section: Discussion and Conclusion
The Continuous Thought Machine (CTM) represents a new perspective, where the temporal dynamics of neural activity are central to artificial cognition. Its core innovations-neuron-level models and synchronization as a latent representation-effectively enable it to both unfold and leverage neural dynamics to solve problems. We showed in this work that such an approach is not only feasible but also leads to unique computational capabilities and emergent properties.
Our experiments demonstrate that the CTM can effectively solve challenging tasks. We trained a CTM to observe, plan, and implement routes through 2D mazes using a setup that necessitated the formation of an internal world model. On ImageNet, the CTM exhibited native adaptive computation, naturally tailoring its processing time to input difficulty, and achieved strong calibration-a desirable property often requiring specialized techniques. On algorithmic tasks like parity checking, the CTM developed interpretable, sequential problem-solving strategies. Notably, the core architecture remained consistent across tasks, highlighting its robustness.
The CTM's NLMs are inspired by the complexity of biological neurons, but are implemented with a level of abstraction appropriate for modern deep learning . The direct use of neural synchronization as a representation is, to our knowledge, a novel approach at this scale, offering benefits such as a high-cardinality representational space and the potential to capture the temporal aspects of 'thought'. While traditional deep learning has abstracted away neural timing for computational efficiency, the CTM shows that reintroducing such dynamics in a structured way can unlock new functionalities.
Limitations. The CTM uses an internal sequence, meaning training times are extended. NLMs also increase parameter counts compared to standard activation functions, but also provide a new avenue for scaling. The experiments in this paper are preliminary and not intended to beat state-of-the-art models tailored for performance, therefore a limitation of this paper is its relatively limited depth of comparison since we favored breadth to investigate the CTM's internal functionality.
Future Work. We plan to apply the CTM to language modeling, self-supervised video understanding, lifelong-learning, biologically-inspired memory and plasticity, multi-modal systems, and more. We believe that, conceptually, synchronization representations have high widespread potential.
this section cite: []

Section: References
Ref_id:b0 Title: Deep learning Year: (2015)
Ref_id:b1 Title: Deep learning Year: (2016)
Ref_id:b2 Title: Emergent abilities of large language models Year: (2022)
Ref_id:b3 Title: Building machines that learn and think like people Year: (2017)
Ref_id:b4 Title: Deep learning: A critical appraisal Year: (2018)
Ref_id:b5 Title: On the measure of intelligence Year: (2019)
Ref_id:b6 Title: The computational limits of deep learning Year: (2020)
Ref_id:b7 Title: Ontology reasoning with deep neural networks Year: (2020)
Ref_id:b8 Title: Time is of the essence: neural codes, synchronies, oscillations, architectures Year: (2022)
Ref_id:b9 Title: On the relevance of time in neural computation and learning Year: (2001)
Ref_id:b10 Title: Neuronal synchrony in complex-valued deep networks Year: (2013)
Ref_id:b11 Title: Complex-valued neural networks: A comprehensive survey Year: (2022)
Ref_id:b12 Title: Perceiver: General perception with iterative attention Year: (2021)
Ref_id:b13 Title: Scaling up test-time compute with latent reasoning: A recurrent depth approach Year: (2025)
Ref_id:b14 Title: Looped transformers are better at learning learning algorithms Year: (2023)
Ref_id:b15 Title: Adaptive neural networks for efficient inference Year: (2017)
Ref_id:b16 Title: Learning to ponder Year: (2021)
Ref_id:b17 Title: Adaptive computation time for recurrent neural networks Year: (2016)
Ref_id:b18 Title: Adaptive computation with elastic input sequence Year: (2023)
Ref_id:b19 Title: Sparse universal transformer Year: (2023)
Ref_id:b20 Title: Quiet-star: Language models can teach themselves to think before speaking Year: (2024)
Ref_id:b21 Title: Yoshua Bengio, and Bernhard Schölkopf. Recurrent independent mechanisms Year: (2019)
Ref_id:b22 Title: Recurrent models of visual attention Year: (2014)
Ref_id:b23 Title: Brain-inspired learning in artificial neural networks: a review Year: ()
Ref_id:b24 Title: Liquid time-constant networks Year: (2021)
Ref_id:b25 Title: Recurrent complex-weighted autoencoders for unsupervised object discovery Year: (2024)
Ref_id:b26 Title: Learning long sequences in spiking neural networks Year: (2024)
Ref_id:b27 Title: Computability in context: computation and logic in the real world Year: (2011)
Ref_id:b28 Title: Superspike: Supervised learning in multilayer spiking neural networks Year: (2018)
Ref_id:b29 Title: Building functional networks of spiking model neurons Year: (2016)
Ref_id:b30 Title: Generating coherent patterns of activity from chaotic neural networks Year: (2009)
Ref_id:b31 Title: Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits Year: (2021)
Ref_id:b32 Title: Robert Legenstein, and Wolfgang Maass. A solution to the learning dilemma for recurrent networks of spiking neurons Year: (2020)
Ref_id:b33 Title: Meta learning backpropagation and improving it Year: (2021)
Ref_id:b34 Title: Structurally flexible neural networks: Evolving the building blocks for general agents Year: (2024)
Ref_id:b35 Title: Introducing symmetries to black box meta reinforcement learning Year: (2022)
Ref_id:b36 Title: Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks Year: (2021)
Ref_id:b37 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b38 Title: Neural synchrony in cortical networks: history, concept and current status Year: (2009)
Ref_id:b39 Title: Attention is all you need Year: (2017)
Ref_id:b40 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b41 Title: End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking Year: (2022)
Ref_id:b42 Title:  Year: (2018)
Ref_id:b43 Title: Episodic future thinking Year: (2001)
Ref_id:b44 Title: Cortical travelling waves: mechanisms and computational principles Year: (2018)
Ref_id:b45 Title: Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks Year: (2006)
Ref_id:b46 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b47 Title: Automated construction of cognitive maps with visual predictive coding Year: (2024)
Ref_id:b48 Title: A path towards autonomous machine intelligence version 0.9 Year: (2022)
Ref_id:b49 Title: Umap: Uniform manifold approximation and projection for dimension reduction Year: (2018)
Ref_id:b50 Title: Artificial kuramoto oscillatory neurons Year: (2024)
Ref_id:b51 Title: Traveling waves integrate spatial information into spectral representations Year: (2025)
Ref_id:b52 Title: Cifar10 to compare visual recognition performance between deep neural networks and humans Year: (2018)
Ref_id:b53 Title: Human uncertainty makes classification more robust Year: (2019)
Ref_id:b54 Title: Deepproblog: Neural probabilistic logic programming. Advances in neural information processing systems Year: (2018)
Ref_id:b55 Title: Augmenting classic algorithms with neural components for strong generalisation on ambiguous and high-dimensional data Year: (2021)
Ref_id:b56 Title: Gradient-based learning applied to document recognition Year: (1998)
Ref_id:b57 Title: Neuronlike adaptive elements that can solve difficult learning control problems Year: (1983)
Ref_id:b58 Title: Generalization in reinforcement learning: Successful examples using sparse coarse coding Year: (1995)
Ref_id:b59 Title: Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks Year: (2023)
Ref_id:b60 Title: Gymnasium: A standard interface for reinforcement learning environments Year: (2024)
Ref_id:b61 Title: Deep recurrent q-learning for partially observable mdps Year: (2015)
Ref_id:b62 Title: Language modeling with gated convolutional networks Year: (2017)
Ref_id:b63 Title: Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms Year: (2022)
Ref_id:b64 Title: Proximal policy optimization algorithms Year: (2017)
