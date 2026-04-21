Title: Measuring and Controlling Solution Degeneracy across Task-Trained Recurrent Neural Networks
Abstract: Task-trained recurrent neural networks (RNNs) are widely used in neuroscience and machine learning to model dynamical computations. To gain mechanistic insight into how neural systems solve tasks, prior work often reverse-engineers individual trained networks. However, different RNNs trained on the same task and achieving similar performance can exhibit strikingly different internal solutions, a phenomenon known as solution degeneracy. Here, we develop a unified framework to systematically quantify and control solution degeneracy across three levels: behavior, neural dynamics, and weight space. We apply this framework to 3,400 RNNs trained on four neuroscience-relevant tasks: flip-flop memory, sine wave generation, delayed discrimination, and path integration, while systematically varying task complexity, learning regime, network size, and regularization. We find that higher task complexity and stronger feature learning reduce degeneracy in neural dynamics but increase it in weight space, with mixed effects on behavior. In contrast, larger networks and structural regularization reduce degeneracy at all three levels. These findings empirically validate the Contravariance Principle and provide practical guidance for researchers seeking to tune the variability of RNN solutions, either to uncover shared neural mechanisms or to model the individual variability observed in biological systems. This work provides a principled framework for quantifying and controlling solution degeneracy in task-trained RNNs, offering new tools for building more interpretable and biologically grounded models of neural computation.

Section: Introduction
Recurrent neural networks (RNNs) are widely used in machine learning and computational neuroscience to model dynamical processes. They are typically trained with standard nonconvex optimization methods and have proven useful as surrogate models for generating hypotheses about the neural mechanisms underlying task performance [1,2,3,4,5,6]. Traditionally, the study of task-trained RNNs has focused on reverse-engineering a single trained model, implicitly assuming that networks trained on the same task would converge to similar solutions, even when initialized or trained differently. However, recent work has shown that this assumption does not hold universally, and the solution space of task-trained RNNs can be highly degenerate: networks may achieve the same level of training loss, yet differ in out-of-distribution (OOD) behavior, internal representations, neural dynamics, and connectivity [7,8,9,10,11,12,13]. For instance, [8] found that while trained RNNs may share certain topological features, their representational geometry can vary widely. Similarly, [7] showed that task-trained networks can develop qualitatively distinct neural dynamics and OOD generalization behaviors.
These findings raise fundamental questions about the solution space of task-trained RNNs: What factors govern the solution degeneracy across independently trained RNNs? When the solution space of task-trained RNNs is highly degenerate, to what extent can we trust conclusions drawn from a single model instance? While feedforward networks have been extensively studied in terms of how weight initialization and stochastic training (e.g., mini-batch gradients) lead to divergent solutions, RNNs still lack a systematic and unified understanding of the factors that govern solution degeneracy [14,15,16,17,18,19,20,21,22,23]. Cao and Yamins [24] proposed the Contravariance Principle, which posits that as the computational objective (i.e., the task) becomes more complex, the solution space should become less dispersed-since fewer models can simultaneously satisfy the stricter constraints imposed by harder tasks. While this principle is intuitive and compelling, it has thus far remained largely theoretical and has not been directly validated through empirical studies.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b7', 'b6', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23']

Section: Factors controlling solution degeneracy
Behavioral degeneracy Dynamical degeneracy Task complexity Learning regime Structural regularization Network size
this section cite: []

Section: Solution degeneracy
Weight degeneracy Figure 1: Key factors shape degeneracy across behavior, dynamics, and weights. Schematic of our framework for analyzing solution degeneracy in task-trained RNNs. We evaluate how task complexity, learning regime, network size, and structural regularization influence degeneracy at three levels: behavior (network outputs), neural dynamics (state trajectories), and weight space (connectivity).
In this paper, we introduce a unified framework for quantifying solution degeneracy at three levels: behavior, neural dynamics, and weight space (Figure 1). Leveraging this framework, we isolate four key factors that control solution degeneracy: task complexity, learning regime, network width, and structural regularization. We apply this framework in a large-scale experiment, training 50 independently initialized RNNs on each of four neuroscience-relevant tasks. By systematically varying task complexity, learning regime, network width, and regularization, we map how each factor shapes degeneracy across behavior, dynamics, and weights. We find that as task complexity increases (whether via more input-output channels, higher memory demand, or auxiliary objectives, or as networks undergo stronger feature learning), their neural dynamics become more consistent, while their weight configurations grow more variable. In contrast, increasing network size or imposing structural regularization during training reduces variability at both the dynamics and weight levels. At the behavioral level, each of these factors reliably modulates behavioral degeneracy; however, the relationship between behavioral and dynamical degeneracy is not always consistent.
Table 1 summarizes how task complexity, learning regime, network size, and regularization affect degeneracy across levels. In both machine learning and neuroscience, the desired level of degeneracy may vary depending on the specific research questions being investigated. This framework offers practical guidance for tailoring training to a given goal, whether encouraging consistency across models [25], or promoting diversity across learned solutions [26,27,28].
Our key contributions are as follows:
• A unified framework for analyzing solution degeneracy in task-trained RNNs across behavior, dynamics, and weights.
• A systematic sweep of four factors: task complexity, feature learning, network size, and regularization, and a summary of their effects across levels (Table 1), with practical guidance for tuning consistency vs. diversity [25,26,27,28].
• A double dissociation: task complexity and feature learning yield contravariant effects on weights vs. dynamics, while network size and regularization yield covariant effects. Here, contravariant means that a factor decreases degeneracy at one level (e.g., dynamics) while increasing it at another (e.g., weights), whereas covariant means both levels change in the same direction.
this section cite: ['b24', 'b25', 'b26', 'b27', 'b24', 'b25', 'b26', 'b27']

Section: Methods

this section cite: []

Section: Model architecture and training procedure
We use discrete-time nonlinear vanilla recurrent neural networks (RNNs), defined by the update rule: h t = tanh (W h h t-1 + W x x t + b) where h t ∈ R n is the hidden state, x t ∈ R m is the input, W h ∈ R n×n and W x ∈ R n×m are the recurrent and input weight matrices, and b ∈ R n is a bias vector. A learned linear readout is applied to the hidden state to produce the model's output at each time step. Networks are trained with Backpropagation Through Time (BPTT) [29], which unrolls the RNN over time to compute gradients at each step. All networks are trained using supervised learning with the Adam optimizer without weight decay. Learning rates are tuned per task (Appendix B).
For each task, we train 50 RNNs with 128 hidden units. Weights are initialized from the uniform distribution U (-1/ √ n, 1/ √ n) and hidden states are initialized to be zeros.
In all experiments, we train networks until them reach a near-asymptotic, task-specific mean-squred error (MSE) threshold on the training set (see Appendix B), after which we allow a patience period of 3 epochs and stop training to measure degeneracy. This early-stopping criterion ensures that networks trained on the same task achieve comparable final losses before any degeneracy analysis.
this section cite: ['b28']

Section: Task suite for diagnosing solution degeneracy
We selected a diverse set of four tasks designed to elicit distinct neural dynamics commonly studied in neuroscience. The N-Bit Flip-Flop task captures pattern recognition and memory retrieval processes, analogous to Hopfield-type attractor networks that store discrete binary patterns and retrieve them from partial cues [30,31]. The Delayed Discrimination task models working memory maintenance in classic delayed-response paradigms [32,33]. The Sine Wave Generation task represents pattern generation, analogous to Central Pattern Generators (CPGs) that produce self-sustaining rhythmic outputs underlying motor control [34], as well as oscillatory activity observed in motor cortex during movement [35]. Finally, the Path Integration task is inspired by hippocampal and entorhinal circuits that build a cognitive map of the environment to track position by integrating self-motion cues [36]. These tasks have also been used in prior benchmark suites for neuroscience-relevant RNN training [37,38,8], underscoring their broad relevance for studying diverse neural computations. Below, we briefly describe the task structure and the typical dynamics required to solve each one. N-Bit Flip-Flop Task Each RNN receives N independent input channels taking values in {-1, 0, +1}, which switch with probability p switch . The network has N output channels that must retain the most recent nonzero input on their respective channels. The network dynamics form 2 N fixed points, corresponding to all binary combinations of {-1, +1} N . The output range of this task is [-1, 1] and we apply an early-stopping training MSE threshold at 0.001.
this section cite: ['b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b7']

Section: Delayed Discrimination Task
The network receives two pulses of amplitudes f 1 , f 2 ∈ [2, 10], separated by a variable delay t ∈ [5,20] time steps, and must output sign(f 2 -f 1 ). In the N -channel variant, comparisons are made independently across channels. The network forms task-relevant fixed points to retain the amplitude of f 1 during the delay period. The output range of this task is [-1, 1] and we apply an early-stopping training MSE threshold at 0.01.
this section cite: ['b4', 'b19']

Section: Sine Wave Generation
The network receives a static input specifying a target frequency f ∈ [1,30] and must generate the corresponding sine wave sin(2πf t) over time. We define N freq target frequencies, evenly spaced within the range [1,30], and use them during training. In the N -channel variant, each input channel specifies a frequency, and the corresponding output channel generates a sine wave at that frequency. For each frequency, the network dynamics form and traverse a limit cycle that produces the corresponding sine wave. The output range of this task is [-1, 1] and we apply an early-stopping training MSE threshold at 0.05.
Path Integration Task Starting from a random position in 2D, the network receives angular direction θ and speed v at each time step and updates its position estimate. In the 3D variant, the network takes as input azimuth θ, elevation ϕ, and speed v, and outputs updated (x, y, z) position. The network performs path integration by accumulating velocity vectors based on the input directions and speeds. After training, the network forms a map of the environment in its internal state space. The output range of this task is [-5, 5] and we apply an early-stopping training MSE threshold at 0.05.
In our task suite, trained RNNs develop distinct stable dynamical objects: fixed-point (N-Bit Flip Flop, Delayed Discrimination), limit cycle (Sine Wave Generation), and attractor manifold (Path Integration). In Appendix E, we extend our task suite to include a next-step prediction task on the Lorenz 96 chaotic attractors [39], where networks exhibit chaotic dynamical regime.
this section cite: ['b0', 'b29', 'b0', 'b29', 'b38']

Section: Multi-level framework for quantifying degeneracy

this section cite: []

Section: Behavioral degeneracy
We define a novel metric for behavioral degeneracy as the variability in network responses to out-ofdistribution (OOD) inputs. We quantify OOD performance as the mean squared error of all converged networks that achieved near-asymptotic training loss under a temporal generalization condition. For the Delayed Discrimination task, we doubled the delay period. For all other tasks, we doubled the length of the entire trial to assess generalization under extended temporal contexts. Behavioral degeneracy is defined as standard deviation of the OOD losses:
σ OOD = 1 N N i=1 L (i) OOD -L OOD 2 ,
where L OOD is the mean OOD loss. While we focus primarily on the temporal generalization condition for behavioral degeneracy since it directly probes RNNs' sequence processing capacities and their ability to generalize across extended temporal horizons, the same metric can be readily applied to other OOD conditions, such as input noise or external perturbations. In the rest of the paper, we use the term behavioral degeneracy [temporal generalization] to explicitly indicate the OOD condition being tested.
this section cite: []

Section: Dynamical degeneracy
We use Dynamical Similarity Analysis (DSA) [40] to compare the neural dynamics of task-trained networks through pairwise analyses. While previous comparison methods mostly focus on geometry of the data [41,42,43,44], RNNs implement computations through time-varying trajectories rather than static representations, and two RNNs exhibiting similar representational geometry can implement distinct dynamical computations, and vise versa. DSA compares the topological structure of the neural dynamics and has been shown to be more robust to noise and better at identifying behaviorally relevant differences than geometry-based comparison method [45]. For a pair of networks X and Y , DSA projects their time series of activities to a higher-dimensional space and identifies a linear dynamic operator for each system via next-step prediction. The DSA distance between two systems is then computed by minimizing the Frobenius norm between the operators, up to an orthogonal transformation (rotation and reflection):
d DSA (A x , A y ) = min C∈O(n) A x -CA y C -1 F ,
where O(n) is the orthogonal group. We define dynamical degeneracy as the average DSA distance across all network pairs. Additional details on the DSA metric are provided in Appendix F. We note that scale of the DSA distance used to quantify dynamical degeneracy can depend on the choice of DSA hyperparameters. To ensure fair comparison across conditions, we keep all DSA hyperparameters fixed for RNNs trained on the same task. To assess if the neural dynamics across different trained networks are statistically different, we also establish a null distribution by comparing neural trajectories sampled from the same underlying network, see Appendix F.3 for details.
We focus on comparing neural dynamics because RNNs implement computations through timeevolving trajectories rather than static input representations. In addition, we assess representational degeneracy using Singular Vector Canonical Correlation Analysis (SVCCA) [41]. As shown in Appendix G, the four factors that influence dynamical degeneracy do not impose the same constraints on representational degeneracy.
this section cite: ['b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b40']

Section: Weight degeneracy
We quantify weight-level degeneracy via a permutation-invariant version of the Frobenius norm, defined as:
d PIF (W 1 , W 2 ) = min P∈P(n) W 1 -P ⊤ W 2 P F
where W 1 and W 2 are the recurrent weight matrices for a pair of RNNs, P(n) is the set of permutation matrices of size n × n, and ∥ • ∥ F denotes the Frobenius distance. See Appendix F.2 for additional details. For comparing d P IF computed on networks of different sizes, we normalize the above norm by the number of parameters in the weight matrix.
this section cite: []

Section: Results

this section cite: []

Section: Task complexity modulates degeneracy across levels
To investigate how task complexity influences dynamical degeneracy, we varied the number of independent input-output channels. This increased the representational load by forcing networks to solve multiple input-output mappings simultaneously. To visualize how neural dynamics vary across networks, we applied two-dimensional Multidimensional Scaling (MDS) to their pairwise distances.
As task complexity increased, network dynamics became more similar, forming tighter clusters in the MDS space (Figure 3A). This contravariant relationship between task complexity and dynamical degeneracy was consistent across all tasks (Figure 3B). Higher task demands constrain the space of viable dynamical solutions, leading to greater consistency across independently trained networks. At the behavioral level, networks trained on more complex tasks consistently showed lower variability in their responses to OOD test inputs (Figure 3D) in the temporal generalization condition. This finding suggests that increased task complexity, by reducing dynamical degeneracy, also leads to more consistent and less degenerate behavior on the temporal generalization condition across networks. Together, the results at the behavioral and dynamical levels support the Contravariance Principle, which posits an inverse relationship between task complexity and the dispersion of network solutions [24].
At the weight level, we found that pairwise distances between converged RNNs' weight matrices increased consistently with task complexity (Figure 3C). This likely reflects increased dispersion of local minima in weight space for harder tasks. This interpretation is consistent with prior work on mode averaging and loss landscape geometry in feedforward networks, showing that harder tasks tend to yield increasingly isolated minima, separated by steeper barriers [46,47,48,49,50,51,52].
A complementary perspective comes from [53] who introduced the intrinsic dimension as the lowestdimensional weight subspace that still contains a solution, which can serve as a proxy for task complexity. As task complexity increases, the intrinsic dimension of the weight space expands and each solution occupies a thinner slice of a higher-dimensional space, leading to minima that lie further apart. In Section 3.2, we propose an additional mechanism: an interaction between task complexity and the network's learning regime that further amplifies weight-space degeneracy.
this section cite: ['b23', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52']

Section: Additional axes of task complexity
A B C Changing memory demand Adding auxiliary loss In earlier experiments, we controlled task complexity by varying the number of independent input-output channels, effectively duplicating the task across dimensions. Here, we explore two alternative approaches: increasing the task's memory demand and adding auxiliary objectives.
Changing memory demand. Of the four tasks, only Delayed Discrimination requires extended memory, as its performance depends on maintaining the first stimulus across a variable delay. See Appendix D for a quantification of each task's memory demand. We increased the memory load in Delayed Discrimination by lengthening the delay period. This manipulation reduced degeneracy at the dynamical and behavioral levels but increased it at the weight level, mirroring the effect of increasing task dimensionality (Figure 4A).
this section cite: []

Section: Adding auxiliary loss.
We next examined how adding an auxiliary loss affects solution degeneracy in the Delayed Discrimination task. Specifically, the network outputs both the sign and the magnitude of the difference between two stimulus values (f 2 -f 1 ), using separate output channels for each. This manipulation added a second output channel and increased memory demand by requiring the network to track the magnitude of the difference between incoming stimuli. Consistent with our hypothesis, this manipulation reduced dynamical and behavioral degeneracy [temporal generalization] while increasing weight degeneracy (Figure 4B). Crucially, the auxiliary loss induced additional line attractors in the network dynamics, further structuring internal trajectories and aligning neural responses across networks (Figure 4C). While the auxiliary loss increases both output dimensionality and temporal memory demand, we interpret its effect holistically as a structured increase in task complexity.
this section cite: []

Section: Feature learning

this section cite: []

Section: Task complexity scales feature learning
In deep learning theory, neural networks can either solve tasks using their random features at initialization, or adapt their weights and internal features to capture task specific structure [54,55,56,57]. These are referred to as the lazy learning regime, where weights and internal features remain largely unchanged during training, and the rich learning, or feature learning regime, where networks reshape their hidden representations and weights to capture task-specific structure [54,58,59,55].
As the complexity of a task grows, the initial random features no longer suffice to solve it, pushing the network beyond the lazy regime and into feature learning, where weights and internal representations adapt more substantially. [60,61]. If more complex task variants, like those in Section 3.1, truly induce greater feature learning, then networks should adapt more from their initializations and traverse a greater distance in the weight space, resulting in more dispersed final weights. We therefore hypothesize that the increased weight degeneracy observed in harder tasks reflects stronger feature learning within the network. To test this idea, we measured feature learning strength in networks trained on different task variants using two complementary metrics [62,58]: Weight-change norm: ∥W T -W 0 ∥ F , where larger values indicate stronger feature learning. Kernel alignment (KA): The geometry of learning under gradient descent can be described by the neural tangent kernel (NTK), which captures how weight updates affect the network outputs. The NTK is defined by K = ∇ W ŷ⊤ ∇ W ŷ where ŷ denotes the network output. KA measures the directional change of the NTK before and after training:
KA K (T ) , K (0) = Tr(K (T ) K (0) ) ∥K (T ) ∥ F ∥K (0) ∥ F . Lower
KA indicates greater NTK rotation and thus stronger feature learning.
We find that more complex tasks consistently drive stronger feature learning and greater dispersion in weight space, as reflected by increasing weight-change norm and decreasing kernel alignment across all tasks (Figure 5).
this section cite: ['b53', 'b54', 'b55', 'b56', 'b53', 'b57', 'b58', 'b54', 'b59', 'b60', 'b61', 'b57']

Section: Controlling feature learning reshapes degeneracy across levels
Our earlier results show that harder tasks induce stronger feature learning, which in turn shapes the dispersion of solutions in the weight space. To test whether feature learning causally affects degeneracy, we used a principled network parameterization known as maximum update parameterization (µP ), which allows stable feature learning across network widths, even in the infinite-width limit [57,54,56,55]. In this setup, a single hyperparameter (γ) controls the strength of feature learning: higher γ values induce a richer feature-learning regime. Under this parameterization, the network update rule, initialization, and learning rate are scaled with respect to network width N . For the Adam optimizer, the output is scaled as
f (t) = 1 γN W readout ϕ(h(t)). The hidden state update is scaled as h(t + 1) -h(t) = τ -h(t) + 1 N Jϕ(h(t)) + U x(t)
, where J ij ∼ N (0, N ) are the recurrent weights and ϕ is the tanh nonlinearity. The learning rate scales as η = γη 0 . A detailed explanation of µP and its relationship to the standard parameterization is in Appendix K and L. For each task, we trained networks with multiple γ values and confirmed that larger γ consistently induces stronger feature learning, as evidenced by increased weight-change norm and decreased kernel alignment (Appendix M).
We observed that stronger feature learning reduced degeneracy at the dynamical level but increased it at the weight level. We see that when γ is high, networks tend to learn similar task-specific features and converge to consistent dynamics and behavior. In contrast, lazy networks (with small γ) rely on their initial random features, leading to more divergent solutions across seeds-even though their weights move less overall (Figure 6). This finding aligns with prior work in feedforward networks, where feature learning was shown to reduce the variance of the neural tangent kernel across converged models [60]. At the behavioral level, however, increasing feature-learning strength leads networks to overfit the training distribution (Appendix J.2). We hypothesize that stronger feature learning exacerbates overfitting, increasing both average OOD loss and the variability of OOD behavior across models (Figure 6) [63,64,65,66]. Although stronger feature learning increases behavioral degeneracy [temporal generalization], this may partially reflect overfitting to the training distribution, an effect we highlight in Appendix J.2. Clarifying the mechanistic link between dynamical and behavioral degeneracy [temporal generalization] remains an important direction for future work. In Appendix I, we demonstrate that the observed effects of feature learning on degeneracy both interpolates smoothly within the range of γ values and extrapolates beyond the range reported in Figure 6.
this section cite: ['b56', 'b53', 'b55', 'b54', 'b59', 'b62', 'b63', 'b64', 'b65']

Section: 3.3
Larger networks yield more consistent solutions across levels Prior work in machine learning and optimization shows that over-parameterization improves convergence by helping gradient methods escape saddle points [67,68,69,70,71,72,16]. We therefore hypothesized that larger RNNs would converge to more consistent solutions across seeds. However, increasing width also tends to push models towards the lazy regime, where feature learning is suppressed [73,59,54,55,56]. To disentangle these competing effects, we again use the µP parameterization, which holds feature learning strength constant (via fixed γ) while scaling width. Although larger networks may yield more consistent solutions via self-averaging, this outcome is not guaranteed without controlling for feature learning. In standard RNNs, increasing width often Across all tasks, larger networks consistently exhibit lower degeneracy at the weight, dynamical, and behavioral levels, producing more consistent solutions across random seeds (Figure 7). Our dense sweep over 12 intermediate network sizes from 32 to 512 on the 3-Bits Flip Flop task in Appendix I further confirms the observed effect of network width on degeneracy. This pattern aligns with findings in vision and language models, where wider networks converge to more similar internal representations [74,75,41,76,77,65]. In recurrent networks, only a few studies have investigated this "convergence-with-scale" effect using representation-based metrics [74,78]. Our results extend these findings by (1) focusing on neural computations across time (i.e., neural dynamics) rather than static representations, and (2) demonstrating convergence-with-scale across weight, dynamical, and behavioral levels in RNNs.
this section cite: ['b66', 'b67', 'b68', 'b69', 'b70', 'b71', 'b15', 'b72', 'b58', 'b53', 'b54', 'b55', 'b73', 'b74', 'b40', 'b75', 'b76', 'b64', 'b73', 'b77']

Section: Structural regularization reduces solution degeneracy
Figure 8: Low-rank and sparsity regularization reduce solution degeneracy across all levels. On the Delayed Discrimination task, both regularizers lower degeneracy in dynamics, weights, and behavior. Shaded area indicates ±1 standard error.
Low-rank and sparsity constraints are widely used structural regularizers in neuroscienceinspired modeling and efficient machine learning [4,79,80,81,82]. A low-rank penalty compresses the weight matrices into a few dominant modes, while an ℓ 1 penalty drives many parameters to zero and induces sparsity. In both cases, task-irrelevant features are pruned, nudging independently initialized networks toward more consistent solutions on the same task. To test this idea, we augmented the task loss with either a nuclear-norm penalty on the recurrent weights L = L task + λ rank r i=1 σ i , where σ i are the singular values of the recurrent matrix, or an ℓ 1 sparsity penalty:
L = L task + λ ℓ1 i |w i |.
We focused on the Delayed Discrimination task to control for baseline difficulty, and observe that both regularizers consistently reduced degeneracy across all levels. Similar effects hold in other tasks (Appendix O, Figure 8) and intermediate regularization strengths (Appendix I ).
this section cite: ['b3', 'b78', 'b79', 'b80', 'b81']

Section: Discussion
In this work, we introduced a unified framework for quantifying solution degeneracy in task-trained recurrent neural networks (RNNs) at three complementary levels: behavior, neural dynamics, and weights. We systematically varied four factors within our generalizable framework: (i) task complexity (via input-output dimensionality, memory demand, or auxiliary loss), (ii) feature learning strength, (iii) network size, and (iv) structural regularization. We then evaluated their effects on solution degeneracy across a diverse set of neuroscience-relevant tasks.
Two consistent patterns emerged from this analysis. First, increasing task complexity or boosting feature learning produced a contravariant effect: dynamical degeneracy decreased while weight degeneracy increased. Second, increasing network size or applying structural regularization reduced degeneracy at both the weight and dynamical levels-that is, a covariant effect. Here, covariant and contravariant refer to the relationship between weight and dynamic degeneracy, not whether degeneracy increases or decreases overall. For example, task complexity and feature learning reduce dynamical degeneracy but increase weight degeneracy, whereas network size and regularization reduce both.
We also observed that the relationship between dynamical and behavioral degeneracy depends on the varying factor. For instance, stronger feature learning leads to more consistent neural dynamics on the training task but greater variability in OOD generalization This suggests that tightly constrained dynamics on the training set do not guarantee more consistent behavior on OOD inputs. This highlights the need for further empirical and theoretical work on how generalization depends on the internal structure of task-trained networks [83,84,85]. This divergence highlights a key open question: how much of behavioral consistency generalizes beyond training-aligned dynamics, and what task or network factors drive this decoupling?
These knobs allow researchers to tune the level of degeneracy in task-trained RNNs to suit specific research questions or application needs. For example, researchers may want to suppress degeneracy to study common mechanisms underlying a neural computation. Conversely, to probe individual differences, they can increase degeneracy to expose solution diversity across independently trained networks [86,87,88,89,90]. Our framework also supports ensemble-based modeling of brain data. By comparing dynamical and behavioral degeneracy across trained networks, it may be possible to match inter-individual variability in models to that observed in animals, helping capture the full distribution of task-solving strategies [91,92,93,94].
Although our analyses use artificial networks, several of the mechanisms we uncover may translate directly to experimental neuroscience. For example, introducing an auxiliary sub-task during behavioral shaping, which mirrors our auxiliary-loss manipulation, could constrain the solution space animals explore, thereby reducing behavioral degeneracy [95]. Finally, our contrasting findings motivate theoretical analysis, e.g., using linear RNNs to understand why some factors induce contravariant versus covariant relationships across behavioral, dynamical, and weight-level degeneracy.
In summary, our work takes a first step toward addressing this classic puzzle in task-driven modeling: What factors shape the variability across independently trained networks? We present a unified framework for quantifying solution degeneracy in task-trained RNNs, identify the key factors that shape the solution landscape, and provide practical guidance for controlling degeneracy to match specific research goals in neuroscience and machine learning.
Limitations and future directions. This work considers networks equivalent if they achieve similar training loss. Future work could extend the framework to tasks with multiple qualitatively distinct solutions, to examine whether specific factors bias the distribution of networks across those solutions. Another open question is the observed decoupling between dynamical and behavioral degeneracy: how much of behavioral consistency generalizes beyond training-aligned dynamics, and what task or network factors drive this divergence.
B.4 Path Integration Training Hyperparameter Value Optimizer Adam Learning rate 0.001 Learning rate scheduler ReduceLROnPlateau Learning rate decay factor 0.5 Learning rate decay patience 40 Max epochs 1000 Steps per epoch 128 Batch size 64 Early stopping threshold 0.05 Patience 3 Time constant (µP ) 0.1
this section cite: ['b82', 'b83', 'b84', 'b85', 'b86', 'b87', 'b88', 'b89', 'b90', 'b91', 'b92', 'b93', 'b94']

Section: References
Ref_id:b0 Title: Neural circuits as computational dynamical systems Year: (2014)
Ref_id:b1 Title: Recurrent network models of sequence generation and memory Year: (2016)
Ref_id:b2 Title: Recurrent neural networks as versatile tools of neuroscience research Year: ()
Ref_id:b3 Title: Linking connectivity, dynamics, and computations in low-rank recurrent neural networks Year: (2018)
Ref_id:b4 Title: Computation through neural population dynamics Year: (2020)
Ref_id:b5 Title: Flexible multitask computation in recurrent networks utilizes shared dynamical motifs Year: (2024-07)
Ref_id:b6 Title: Charting and navigating the space of solutions for recurrent neural networks Year: (2021)
Ref_id:b7 Title: Universality and individuality in neural dynamics across recurrent networks Year: (2019)
Ref_id:b8 Title: Dynamical phases of short-term memory mechanisms in rnns Year: (2025)
Ref_id:b9 Title: Individual variability of neural computations underlying flexible decisions Year: (2025-03)
Ref_id:b10 Title: Symmetries and Continuous Attractors in Disordered Neural Circuits Year: ()
Ref_id:b11 Title: Connectome-constrained networks predict neural activity across the fly visual system Year: (2024-10)
Ref_id:b12 Title: Phase codes emerge in recurrent neural networks optimized for modular arithmetic Year: (2025)
Ref_id:b13 Title: Systematic errors in connectivity inferred from activity in strongly recurrent networks Year: (2020)
Ref_id:b14 Title: The simplicity bias in multi-task rnns: shared attractors, reuse of dynamics, and geometric representation Year: (2024)
Ref_id:b15 Title: Expand-and-cluster: Parameter recovery of neural networks Year: (2024)
Ref_id:b16 Title: Flat channels to infinity in neural loss landscapes Year: (2025)
Ref_id:b17 Title: Deep ensembles: A loss landscape perspective Year: (2019)
Ref_id:b18 Title: Qualitatively characterizing neural network optimization problems Year: (2015)
Ref_id:b19 Title: Visualizing the loss landscape of neural nets Year: (2018)
Ref_id:b20 Title: Three factors influencing minima in sgd Year: (2018)
Ref_id:b21 Title: Entropy-sgd: Biasing gradient descent into wide valleys Year: (2017)
Ref_id:b22 Title: Similarity of neural network representations revisited Year: (2019)
Ref_id:b23 Title: Explanatory models in neuroscience, part 2: Functional intelligibility and the contravariance principle Year: (2024)
Ref_id:b24 Title: Curriculum learning as a tool to uncover learning principles in the brain Year: (2022)
Ref_id:b25 Title: Striatal dopamine reflects individual long-term learning trajectories Year: (2023-12)
Ref_id:b26 Title: Neural representational geometries reflect behavioral differences in monkeys and recurrent neural networks Year: (2024-08)
Ref_id:b27 Title: International Brain Laboratory, JW Pillow, ND Daw, and IB Witten. Pre-existing visual responses in a projection-defined dopamine population explain individual learning trajectories Year: (2024)
Ref_id:b28 Title: Backpropagation through time: what it does and how to do it Year: (1990)
Ref_id:b29 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b30 Title: Exploring flip flop memories and beyond: training recurrent neural networks with key insights Year: (2024)
Ref_id:b31 Title: Mnemonic coding of visual space in the monkey's dorsolateral prefrontal cortex Year: (1989)
Ref_id:b32 Title: Cellular basis of working memory Year: (1995)
Ref_id:b33 Title: Central pattern generators and the control of rhythmic movement Year: (2001)
Ref_id:b34 Title: Neural population dynamics during reaching Year: (2012)
Ref_id:b35 Title: Path integration and the neural basis of the 'cognitive map' Year: (2006)
Ref_id:b36 Title: Task representations in neural networks trained to perform many cognitive tasks Year: (2019)
Ref_id:b37 Title: Winning the lottery with neural connectivity constraints: Faster learning across cognitive tasks with spatially constrained sparse rnns Year: ()
Ref_id:b38 Title: Predictability: A problem partly solved Year: (1995-09-08)
Ref_id:b39 Title: Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis Year: (2023-10)
Ref_id:b40 Title: Svcca: Singular vector canonical correlation analysis for deep learning dynamics Year: (2017)
Ref_id:b41 Title: Representational similarity analysis -connecting the branches of systems neuroscience Year: (2008)
Ref_id:b42 Title: Generalized Shape Metrics on Neural Representations Year: (2022-01)
Ref_id:b43 Title: Brain-score: Which artificial neural network for object recognition is most brain-like? bioRxiv Year: (2020)
Ref_id:b44 Title: Dynamical similarity analysis uniquely captures how computations develop in RNNs Year: (2025)
Ref_id:b45 Title: Qualitatively characterizing neural network optimization problems Year: (2015)
Ref_id:b46 Title: Linear mode connectivity and the lottery ticket hypothesis Year: (2020)
Ref_id:b47 Title: On monotonic linear interpolation of neural network parameters Year: (2021)
Ref_id:b48 Title: Large scale structure of neural network loss landscapes Year: (2019)
Ref_id:b49 Title: Where is the information in a deep neural network? CoRR, abs Year: (1905)
Ref_id:b50 Title: Rethink model re-basin and the linear mode connectivity Year: (2024)
Ref_id:b51 Title: Optimization on multifractal loss landscapes explains a diverse range of geometrical and dynamical properties of deep learning Year: ()
Ref_id:b52 Title: Measuring the intrinsic dimension of objective landscapes Year: (2018)
Ref_id:b53 Title: On lazy training in differentiable programming Year: (2019)
Ref_id:b54 Title: Kernel and rich regimes in overparametrized models Year: (2020-07)
Ref_id:b55 Title: Disentangling feature and lazy training in deep neural networks Year: (2020)
Ref_id:b56 Title: Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks Year: (2022-10)
Ref_id:b57 Title: Lazy vs hasty: linearization in deep networks impacts learning schedule based on example difficulty Year: (2022)
Ref_id:b58 Title: Wide neural networks of any depth evolve as linear models under gradient descent * Year: (2020-12)
Ref_id:b59 Title: Dynamics of finite width Kernel and prediction fluctuations in mean field neural networks * Year: (2024-10)
Ref_id:b60 Title: Grokking as the transition from lazy to rich training dynamics Year: (2023)
Ref_id:b61 Title: How connectivity structure shapes rich and lazy learning in neural circuits Year: ()
Ref_id:b62 Title: Revisiting model stitching to compare neural representations Year: (2021)
Ref_id:b63 Title: Unsupervised model selection for variational disentangled representation learning Year: (2020)
Ref_id:b64 Title: The platonic representation hypothesis Year: (2024)
Ref_id:b65 Title: Convergent learning: Do different neural networks learn the same representations? Year: (2016)
Ref_id:b66 Title: Deep learning without poor local minima Year: (2016)
Ref_id:b67 Title: The loss surface of deep and wide neural networks Year: (2017)
Ref_id:b68 Title: Gradient descent finds global minima of deep neural networks Year: (2018)
Ref_id:b69 Title: A convergence theory for deep learning via over-parameterization Year: (2019)
Ref_id:b70 Title: Stochastic gradient descent optimizes over-parameterized deep relu networks Year: (2018)
Ref_id:b71 Title: Geometry of the loss landscape in overparameterized neural networks: Symmetries and invariances Year: (2021)
Ref_id:b72 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b73 Title: Insights on representational similarity in neural networks with canonical correlation Year: (2018)
Ref_id:b74 Title: Similarity of neural network representations revisited Year: (2019)
Ref_id:b75 Title: Dynamical models of cortical circuits Year: ()
Ref_id:b76 Title: Contrasim -analyzing neural representations based on contrastive learning Year: (2024)
Ref_id:b77 Title: Do wide and deep networks learn the same things? uncovering how neural network representations vary with width and depth Year: (2021)
Ref_id:b78 Title: Shaping dynamics with multiple populations in low-rank recurrent networks Year: (2020)
Ref_id:b79 Title: Emergence of simple-cell receptive field properties by learning a sparse code for natural images Year: (1996)
Ref_id:b80 Title: Learning both weights and connections for efficient neural networks Year: (2015)
Ref_id:b81 Title: Deep sparse rectifier neural networks Year: (2011)
Ref_id:b82 Title: Representations and generalization in artificial and brain neural networks Year: (2024)
Ref_id:b83 Title: Separability and geometry of object manifolds in deep neural networks Year: (2020)
Ref_id:b84 Title: Neural representational geometry underlies few-shot concept learning Year: (2022)
Ref_id:b85 Title: Accounting for variance in machine learning benchmarks Year: (2021)
Ref_id:b86 Title: Sloppy modeling. Knowledge representation and organization in machine learning Year: (2005)
Ref_id:b87 Title: Does the data induce capacity control in deep learning Year: (2022)
Ref_id:b88 Title: Emergent behaviour and neural dynamics in artificial agents tracking odour plumes Year: (2023)
Ref_id:b89 Title: Behavioral and neural variability of naturalistic arm movements Year: ()
Ref_id:b90 Title: Coordinated prefrontal-hippocampal activity and navigation strategy-related prefrontal firing during spatial memory formation Year: (2018)
Ref_id:b91 Title: Mice alternate between discrete strategies during perceptual decision-making Year: (2022)
Ref_id:b92 Title: A reservoir of foraging decision variables in the mouse brain Year: (2023)
Ref_id:b93 Title: Individual variability of neural computations underlying flexible decisions Year: (2025)
Ref_id:b94 Title: Control of variability Year: (2002)
Ref_id:b95 Title: Dynamic mode decomposition and its variants Year: (2022)
Ref_id:b96 Title: A generalized solution of the orthogonal procrustes problem Year: (1966-03)
Ref_id:b97 Title: Nonnegative matrix factorization for combinatorial optimization: Spectral clustering, graph matching, and clique finding Year: (2008)
Ref_id:b98 Title: Procrustes: A python library to find transformations that maximize the similarity between matrices Year: ()
Ref_id:b99 Title: Universally sloppy parameter sensitivities in systems biology models Year: (2007)
Ref_id:b100 Title: Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer Year: ()
Ref_id:b101 Title: Tensor programs vi: Feature learning in infinite-depth neural networks Year: ()
Ref_id:b102 Title: Mathematical equivalence of two common forms of firing rate models of neural networks Year: (2012)
