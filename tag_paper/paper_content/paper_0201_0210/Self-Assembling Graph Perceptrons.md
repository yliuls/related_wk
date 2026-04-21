Title: Self-Assembling Graph Perceptrons
Abstract: Inspired by the workings of biological brains, humans have designed artificial neural networks (ANNs), sparking profound advancements across various fields. However, the biological brain possesses high plasticity, enabling it to develop simple, efficient, and powerful structures to cope with complex external environments. In contrast, the superior performance of ANNs often relies on meticulously crafted architectures, which can make them vulnerable when handling complex inputs. Moreover, overparameterization often characterizes the most advanced ANNs. This paper explores the path toward building streamlined and plastic ANNs. Firstly, we introduce the Graph Perceptron (GP), which extends the most fundamental ANN, the Multi-Layer Perceptron (MLP). Subsequently, we incorporate a self-assembly mechanism on top of GP called Self-Assembling Graph Perceptron (SAGP). During training, SAGP can autonomously adjust the network's number of neurons and synapses and their connectivity. SAGP achieves comparable or even superior performance with only about 5% of the size of an MLP. We also demonstrate the SAGP's advantages in enhancing model interpretability and feature selection.

Section: Introduction
With the exceptional intelligence of the brain, humanity has created a remarkable modern civilization. This intelligence stems from the brain's intricate structure, resulting from long-term environmental adaptation and natural selection. Research shows that the survival environment profoundly impacts shaping the brain, which in turn indirectly shapes human cognitive and emotional abilities [1].
On the other hand, since the 1940s, researchers have been exploring how to simulate the behavior of the biological brain using computers to build artificial intelligence agents. It was only in recent decades that a framework known as artificial neural networks (ANNs) has indeed demonstrated significant potential in this field. ANNs simulate neurons and their synaptic connections in the brain, determining each neuron's output by the strength of the input signals they receive.
Despite the powerful capabilities of ANNs, their design seems to deviate from the initial intent of mimicking the biological brain. In ANNs, the number of neurons and synaptic connection patterns are predefined, resulting in noticeable vulnerabilities when the network faces complex environments. [2] and [3] have demonstrated, from theoretical and experimental perspectives, respectively, the significant impact of the number of hidden layers and neurons on the ability of Multi-Layer Perceptrons (MLPs). Moreover, modern deep models are often over-parameterized, with the most advanced large language models reaching a parameter scale of trillions. In contrast, biological brains continually self-assemble throughout their lifecycle, developing remarkable capabilities. For instance, a nematode can manage all its behaviors with fewer than 500 neurons [4].
Recently, some studies have focused on creating ANNs that can assemble themselves based on input without relying on prior knowledge. [5] and [6] proposed learning genomes that regulate neuronal behavior, enabling self-adjustment of synaptic connection rules without altering the number of neurons. The neural development program (NDP, [7,8]) first suggests regulating neuron growth through genomes, allowing the network to develop from a single neuron and incrementally add new neurons and synapses in response to inputs, eventually growing into a network of a predefined size. However, NDP does not fully realize biological self-assembly. It neglects the most important reason biological neural systems maintain their efficiency and compactness: neuronal apoptosis [9]. Furthermore, since NDP explicitly encodes a genome for generating the network structure and optimizes it by reinforcement learning, the high computational cost prevents it from running on even a typical-scale dataset.
In this paper, we first introduce a generalized version of the Multi-Layer Perceptron (MLP), namely the Graph Perceptron (GP). Building upon this, we present the Self-Assembling Graph Perceptrons (SAGP)-the first model with full self-assembly capability. SAGP begins from the simplest form and autonomously determines when to grow or undergo apoptosis during its developmental cycle. Synaptic connections between neurons are dynamically adjusted. Unlike previous works, SAGP does not explicitly model the genome that controls neuronal behavior and topology. Instead, it achieves self-assembly by establishing assembly rules and simulating the competitive pressures found in nature. This makes SAGP more aligned with the general paradigm of modern deep networks and results in a significant improvement in the assembly speed by more than 10,000 times. We demonstrate that SAGP can achieve a more streamlined perceptron topology than MLP while highlighting the potential of the self-assembly mechanism in enhancing model interpretability and feature selection.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']

Section: Related work

this section cite: []

Section: Neural network bionics

this section cite: []

Section: Structure

this section cite: []

Section: Assembly Connection

this section cite: []

Section: Networked
Self-assembling Sparse Layered Pre-defined Dense Scientists have long sought inspiration from the biological world to develop algorithms for solving real-world problems. Early bio-inspired strategies, such as evolutionary and genetic algorithms [10,11], solve complex optimization problems by simulating the process of biological evolution. Meanwhile, artificial neural networks, which simulate the process of neuronal interactions through synaptic connections [12], have achieved great success in numerous applications. However, ANNs, represented by MLP, are predefined as layered structures with fixed sizes and usually require a large number of neurons and dense synaptic connections for optimal performance, which is contrary to the properties of biological neural networks (Fig. 1). Consequently, a growing body of research is focused on designing networks with more bio-inspired features. Spiking Neural Networks (SNNs) mimic the mechanism of neurons transmitting information through action potentials, enabling asynchronous and low-power information processing [13,14]; Liquid Neural Networks (LNNs) simulate the neural connection of the nematodes, using a small number of neurons to generate continuous-time outputs [15]. Plastic Neural Networks (PNNs) emulate the developmental processes of organisms, dynamically adjusting their structures based on environmental states [16].
this section cite: ['b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15']

Section: Synaptic-level plasticity
The characteristic of plastic ANNs lies in their ability to adjust their structure in response to environmental changes. Initially, this research area focused on synaptic-level plasticity; inspired by early neurobiological theories, researchers developed mechanisms such as the Hebbian rule [17] and Spike-Timing Dependent Plasticity (STDP, [18]) to enable ANNs to selfregulate synaptic strength. These classical methods are unsupervised, while more modern approaches employ backpropagation to learn synaptic rules [19,20] or use meta-learning to obtain genomes that control synaptic behavior [5,6].
Neuron-level plasticity Some methods allow neurons to adjust their state based on the environment, such as by modifying weights [21], changing activation functions [22], or learning rates [23]. Neuroevolution [24,25] encodes the network topology as individuals in a population and finds the optimal ones by evolution. [26] considers parameter pruning based on plastic neurons. Only recently has the concept of self-assembly -the dynamic increase of neurons within a single network (as opposed to a population) -been introduced by Neural Developmental Program (NDP, [7,8]). We achieve a fully self-assembling model with neural competition and apoptosis mechanisms. Compared to NDP, our model improves the assembly speed by over 10 4 times per epoch and shows the ability to integrate with modern deep models. See Appendix A for more related works.
this section cite: ['b16', 'b17', 'b18', 'b19', 'b4', 'b5', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b6', 'b7']

Section: Self-Assembling Graph Perceptrons
This section introduces our approach, the Self-Assembling Graph Perceptrons (SAGP), in detail. Without additional constraints, the free connections between neurons may exhibit a more general connection pattern than the layered connections, namely, a graph-structured connection. In Section 3.1, we explore how information is updated in a graph-structured perceptron model (i.e., GP), which forms the foundation for achieving the network's self-assembly capability. Section 3.2 introduces the approach by which GP achieves self-assembly through establishing assembly rules and simulating competitive pressures. Since the inception of perceptron models, they have been conventionally regarded as layered structures [12,27,28,29]. Although fully connected perceptron models like Hopfield Networks [30] and Boltzmann Machines [31] emerged in the past, they gradually became marginalized due to practical limitations. However, we re-examine this concept inspired by the message-passing mechanism [32].
this section cite: ['b11', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31']

Section: From MLPs to graph perceptrons
Take a simple MLP with 2 input neurons, 2 hidden neurons in 1 hidden layer, and 1 output neuron as an example (Fig. 2). Let x = x 0 x 1 and z = [z 0 ] represent the model's input and output, respectively, then we have:
z = W T 1 • σ(W T 0 • x + b 0 ), (1
)
where
W 0 = w 0 00 w 0 01 w 0 10 w 0 11 , b 0 = b 0 0 b 0 1 ,and
W 1 = w 1 00 w 1 10 .
Now, we number all neurons sequentially in the order of input, hidden, and output layers. In this way, the topology of the MLP and the weights of its edges can be represented by a 5 × 5 adjacency matrix A. If the i-th neuron has a synaptic connection to the j-th neuron, then A ij is the weight of that synapse; otherwise, A ij = 0. Input neurons carry self-loops, meaning they continuously send information to the network. We also construct a vector b of length 5 to represent the biases of all the neurons, where the biases of input and output neurons are set to 0, as follows:
A = I 2×2 W 0 0 2×1 0 2×2 0 2×2 W 1 0 1×2 0 1×2 0 1×1 , b = 0 2 b 0 0 1 .(2)
Here, 0 represents a zero matrix or vector. If we input x at the input neurons and use zero inputs for other neurons, then, after one step of message-passing [32] on the graph described by A, we obtain the same intermediate results at the hidden neurons as we would in MLPs:
σ A T • x 0 2 0 1 + b =   σ(x) σ(W T 0 x + b 0 ) 0 1   .(3)
By repeating this process once again, the output neuron produces the same result as the output of the MLPs:
A T •   σ(x) σ(W T 0 x + b 0 ) 0 1   + b =   σ(x) W T 0 σ(x) + b 0 z   .(4)
Input Hidden Output
this section cite: ['b31']

Section: Growth Competition Apoptosis
Self-assembling cycle (if possible) (if possible)
Figure 3: A schematic of our model (SAGP). It assembles itself by simulating neuron growth, competition, and apoptosis.
Thus, we obtain an alternative representation of z, where we use Python-style indexing [-2:] to denote taking the last two elements of a vector to form a new one:
z = A T • σ A T • x 0 3 + b + b [-2:] . (5
)
It is easy to show that similar conclusions hold for MLPs of any number of hidden layers and sizes (See appendix B).
Based on this observation, we now define a generalized perceptron model, referred to as the Graph Perceptrons (GP). Let the sets of input neurons, hidden neurons, and output neurons be denoted as I, H, and O, respectively. GP has the following adjacency matrix and biases:
A =   I |I|×|I| A I→H A I→O 0 |H|×|I| A H→H A H→O 0 |O|×|I| 0 |O|×|H| 0 |O|×|O|   , b =   0 |I| b H 0 |O|   .(6)
GP allows direct connections from input neurons to output neurons and interconnections between any two hidden neurons. In GP, the feature update process is described as:
∀l ∈ [0, L -1], h l+1 = A x l + b, x l+1 = σ(h l+1 ) where x 0 = x 0 |H|+|O| and z = h L [-|O|:] .(7)
When A I→O = 0, A H→H = I, and L = 2, the GP degenerate to MLPs with 1 hidden layer. L represents the number of message-passing steps. It is worth noting that when a GP has the same topology as the MLP shown in Fig. 2, it can perform arbitrary L (L ≥ 2) message-passing steps, not only 2. When L > 2, this can be interpreted as the output neurons not producing an output immediately upon receiving the first message but rather waiting for further ones. We also provide a neuron-level equivalent representation of equation (7), which is useful in subsequent discussions:
∀l ∈ [0, L -1], ∀i ∈ H ∪ O, h l+1 i = b i + j∈N A ij x l j , x l+1 i = σ(h l+1 i ), where x 0 i = x i , if i ∈ I, 0, else. and z = [h L i ] i∈O .(8)
Where N = I ∪ H ∪ O. The strength of GP lies in its ability to enable perceptrons to work under any topology. It is crucial for building self-assembling neural networks, as neurons' dynamic growth and apoptosis lead to complex connectivity patterns.
this section cite: ['b6']

Section: Growth and apoptosis: Let GP assemble itself
This section introduces how to implement a GP with full self-assembly capabilities, namely SAGP. SAGP is the first to realize a fully self-assembling neural network, capable of growing new neurons and achieving neuronal apoptosis and synaptic pruning. This functionality is realized by setting assembly rules and simulating competitive pressures, thus avoiding direct encoding of the genome.
this section cite: []

Section: Initial state
The SAGP begins its development from the simplest state. GP always has a fixed number of input and output neurons, but at this stage, no hidden neuron, i.e., |H| = 0. It also includes all possible synapses from I to O, totaling |I| × |O|.
this section cite: []

Section: Neuron competition & synaptic competition
During the development of the biological brain, neuron competition [33] and synaptic competition [34] play a crucial role in acquiring cognitive and memory  abilities. We simulate this process by learnable masks with competition loss. Two types of masks are introduced in SAGP: neuron-level masks (denoted as m i ) and synapse-level masks (denoted as m ij , Fig. 4a). Noting that if a hidden neuron has no outgoing synapses, it will never affect the output neurons. Therefore, the node-level mask acts on all the neuron's outgoing edges. The feature update of the GP can be rewritten as:
h l+1 i = b i + j∈N m j • m ij • A ij x l j .(9)
All masks are learnable and can only take values of 0 or 1. It is achieved through a two-step process: first, for a learnable real number m ∈ R, we generate a soft mask m S ∈ [0, 1] by Gumbel reparameterization [35]; then, we use the Straight-Through Estimator (STE, [36]) to produce a hard mask m = m H ∈ {0, 1} that is suitable for back-propagation:
m S = Gumbel_Sigmold( m),(10)
m H = Stop_Grad(1(m S > 0.5) -m S ) + m S ,(11)
m = m H .(12)
However, what truly enables the masks to make a competitive effect is what we refer to as the "Competition Loss":
CompLoss α (m S ) = C(α) • m S (1 -m S ) α . (13
)
Here, α ∈ (0, 1) is a temperature parameter, and C(α) is a scaling factor designed to ensure that CompLoss a has a maximum value of exactly 1 over [0, 1]. As shown in Fig. 4b, SparseLoss weakens the less advantageous soft masks, driving them towards 0. In contrast, only the masks that dominate in competition-i.e., those with larger values-are enhanced, tending towards 1. By summing up the SparseLoss of all masks, we obtain the auxiliary loss:
AuxLoss = γ 1 |H| i∈H CompLoss α (m S i ) + γ 2 |I| + |H| i∈I∪H j∈H∪O CompLoss α (m S ij ).(14)
γ 1 and γ 2 are both weighting parameters. We achieve neuron competition and synaptic competition mechanism via back-propagation by adding AuxLoss to downstream task loss.
Neuronal apoptosis Biological organisms streamline the structure of their nervous systems through synaptic pruning [37] and apoptosis [9]. Similarly, when training SAGP, we introduce the following rules to remove neurons from the network to ensure the efficiency of the structure:
Apoptosis Rule: Remove hidden neuron i from set H and remove all synapses connected to i if m S i < β • Mean j∈H (m S j ). Here, β ∈ (0, 1). The soft mask of a hidden neuron reflects its status in the competition. The Apoptosis Rule removes neurons that are relatively weaker within the overall population, as their influence on the outcome is negligible, thereby reducing computational overhead.
Neuronal growth Organisms tend to reproduce faster under reduced competitive pressure [38]. In SAGP, if no neurons have been pruned over N consecutive training epochs, this may indicate that competition has sufficiently stabilized, with each neuron occupying a corresponding niche. At this point, we add new hidden neurons to increase competitive pressure:
Growth Rule: Add a new neuron to H and add all possible synapses connected to this new neuron to the network if no neurons are pruned for N consecutive epochs.
As the neural network trains, the number of hidden neurons dynamically increases or decreases based on two rules. Synapses or neurons in the network may also become temporarily ineffective due to insufficient competitiveness (corresponding to a small mask value). See Appendix C.3 for implementation details.
this section cite: ['b32', 'b33', 'b34', 'b35', 'b36', 'b8', 'b37']

Section: Assembling dynamics
We visualized the selfassembly process of SAGP (Fig. 5a and Fig. 5b) and its performance variations (Fig. 5c). The results show that the self-assembly undergoes three distinct stages: First, the network rapidly expands to near its maximum size within a short period (approximately 0 to 1000 epochs); next, intense competition occurs between neurons and synapses, with the majority of neuronal apoptosis and synaptic pruning happening during this stage (approximately 1000 to 10000 epochs); finally, the network topology stabilizes, with the number of neurons remaining almost constant while the number of synapses slowly decreases (approximately 10000 to 50000 epochs). Even as the network size decreases, its performance keeps improving, indicating that the information density in the parameters is increasing. Interestingly, this trend is similar to the changes in human cortical volume: the cortex rapidly reaches its maximum volume during childhood. It then gradually decreases in size over a long maturation period, enhancing its functionality [39].
Hyperparametes Several hyperparameters are introduced in SAGP, including L, α, γ 1 , γ 2 , β, and N . We provide a set of empirical hyperparameters, which are consistently applied across all experiments discussed in this paper (See Appendix C.2). While hyperparameter tuning typically yields better performance for specific tasks or datasets, by fixing these parameters, we demonstrate the strong adaptability of SAGP in tackling complex environments.
Complexity Let the batch size be B. The computational complexity of one forward propagation in SAGP is O(BLS), where L and S represent the number of message-passing steps and the number of synapses, respectively. S can be further expressed as λ(n 1 +n 2 +n 3 ) 2 , where n 1 , n 2 , and n 3 represent the numbers of input, hidden, and output neurons, respectively, and λ denotes the density of the adjacency matrix. Therefore, the complexity of SAGP is also expressed as O(λBL(n 1 + n 2 + n 3 ) 2 ).
this section cite: ['b38']

Section: Experiment
Overview In the experiment, we answer three questions: first, the impact of topological structure on the perceptron model; second, the performance of SAGP on deep learning tasks; and third, the inspiration that self-assembling neural networks provide for modern deep learning. Three domains of the dataset are used: text, audio, and images. We also conducted experiments on deep graph models and temporal models. See Appendix C.1 for more dataset details.
this section cite: []

Section: Perceptron topology
Question 1: Is a multi-layered connection like MLP always the best perceptron topology?  With a fixed budget of 128 hidden neurons, we investigated the performance of 8 different topologies for graph perceptrons (Table 1). The results show that no single topology consistently outperforms others across different datasets and metrics. This suggests that mechanisms like self-assembly, which dynamically adjust the perceptron topology, have the potential to deliver better performance than fixed multi-layer topology. We select the FSDD dataset with the smallest performance gap for further investigation. We report the convergence speed of the GP (Fig. 6 left) and the impact of hidden neuron count on performance (Fig. 6 right). We found that the eight topologies can be divided into two categories: The first category includes topologies with strict hierarchical connection structures like (A), (C), (D), and (H), which converge faster but suffer from significant performance degradation at low budgets. The second category consists of topologies with nonstrict hierarchical connections like (B), (E), (F), and (G), which converge slower but perform well even with a small number of hidden neurons. Thus, another potential advantage of general-topology perceptrons is that they may achieve performance comparable to multilayer-connected ones with a much smaller size. However, multilayer perceptrons have a computational efficiency advantage since they can be implemented using straightforward matrix multiplications rather than message-passing mechanisms.  Scale Metric |H| #Synapse F1-micro (Accuracy) F1-macro AUC-micro AUC-macro Photo MLP-b 512 385536 .8687±.0053 .8314±.0064 .9851±.0010 .9867±.0002 SAGP-b 0.20(0.04%) 4196(1.09%) .8704±.0047(100.20%) .8354±.0058(100.48%) .9856±.0011(100.05%) .9881±.0003(100.14%) SAGP-l 0.00(0.00%) 4040(1.05%) .8671±.0015(99.82%) .8335±.0017(100.25%) .9855±.0004(100.04%) .9882±.0001(100.15%) Computers MLP-b 1024 659968 .7820±.0044 .7073±.0058 .9696±.0012 .9769±.0006 SAGP-b 0.00(0.00%) 4376(0.66%) .7687±.0032(98.30%) .6986±.0045(98.77%) .9657±.0013(99.60%) .9752±.0003(99.83%) SAGP-l 0.00(0.00%) 3810(0.58%) .7677±.0040(98.17%) .6992±.0040(98.85%) .9660±.0013(99.63%) .9752±.0003(99.83%) ESC-50 MLP-b 1024 379904 .3297±.0096 .2993±.0102 .8888±.0036 .8841±.0036 SAGP-b 98.20(9.59%) 20874(6.18%) .3190±.0126(96.75%) .3039±.0164(101.54%) .8715±.0027(98.05%) .8650±.0030(97.84%) SAGP-l 106.6(10.41%) 11251(2.96%) .3438±.0153(104.28%) .3305±.0160(110.42%) .8711±.0025(98.01%) .8659±.0027(97.94%) FSDD MLP-b 2048 883712 .9467±.0065 .9473±.0056 .9979±.0003 .9975±.0004 SAGP-b 55.10(2.69%) 6246(0.70%) .9142±.0112(96.57%) .9149±.0111(96.64%) .9925±.0012(99.46%) .9923±.0013(99.48%) SAGP-l 55.00(2.69%) 4312(0.49%) .9120±.0142(96.33%) .9132±.0141(96.40%) .9917±.0010(99.38%) .9919±.0011(99.44%) Fasion MNIST MLP-l 1024 668672 .8668±.0049 .8651±.0046 .9924±.0005 .9894±.0006 SAGP-l 2.90(0.28%) 3247(0.49%) .8321±.0073(96.00%) .8307±.0067(96.02%) .9869±.0009(99.45%) .9818±.0012(99.23%) CIFAR-10 MLP-l 768 920064 .5201±.0027 .5168±.0037 .8975±.0013 .8941±.0011 SAGP-l 65.60(8.54%) 90674(9.86%) .5132±.0068(98.67%) .5095±.0073(98.59%) .8918±.0032(99.42%) .8880±.0035(99.32%) Table 4: We replaced the MLP as the nonlinear transformation layers of several classic models with SAGP and evaluated its performance.
We compared two perceptron models: SAGP and MLP (Table 2). The only previous self-assembling model NDP was excluded from the comparison due to resource constraints (a single training run exceeding 12 hours on a single Nvidia RTX 4090). The suffix "-b" indicates the epoch with the best performance during training, i.e., the epoch that achieves the lowest loss on the validation set. The suffix "-l" refers to the last training epoch when convergence is reached. MLPs report results from the best epoch by default unless the dataset lacks a validation set. SAGP-b exhibits better performance, while SAGP-l has a smaller topology size. Compared to MLP, SAGP uses only 0% ∼ 10.41% of hidden neurons and 0.49% ∼ 9.86% of synapses, achieving performance ranging from 96% ∼ 110.42%. Following the setup in [7], we compared SAGP and NDP on a toy dataset, Digit (Table 3). We found that SAGP's average runtime per epoch improved by over 10,000 times. Due to neuronal apoptosis, SAGP eventually removed all hidden neurons, achieving significantly better performance with only the synapses between input and output neurons. We also experimented with the potential of SAGP as a submodule in other models (Table 4). Specifically, we replaced the MLP used for generating classification outputs in GAMLP [40], LSTM [41], and LeNet [42] with SAGP. The results show that SAGP can achieve comparable performance while maintaining a significantly simplified topology. Notably, reinforcement learning-trained NDP cannot be integrated into these models.
Answer 2: SAGP achieves performance comparable to MLP with a smaller topology, while being far more efficient and flexible than NDP.
this section cite: ['b6', 'b39', 'b40', 'b41']

Section: Inspiration for deep learning
Question 3: In what fields can SAGP show its potential? Model Interpretability A significant challenge of modern deep learning models is the lack of interpretability. We visualize the out-degree of each pixel (input neuron) in the SAGP for CIFAR-10. We found that the pixels with high degrees are concentrated in the image's central region, where the image's main subject typically occupies. When selecting the top 50% of pixels by degree, the central semantics of the image are still preserved (fig. 8). This suggests that even a purely perceptron-based model like SAGP might exhibit some clues about the reasoning behind its judgments, similar to how humans respond.  Feature selection Sometimes we want to determine which data preprocessing method is more effective or select the most critical subset from many features to reduce computation. These issues are collectively referred to as feature selection problems. Our experiments show that after training with SAGP, the out-degree of input features (input neurons) can indicate feature importance. In the FSDD audio dataset, we find that Mel-Frequency Cepstral Coefficients (MFCC) and Mel frequency spectrogram (Mel) features are much more effective than chroma features (Fig. 9). On the Computers datasets, we selected the top 5%, 10%, and 20% of features based on their out-degree, and tested them with backbone models GCN [43] and GAT [44]. The classical non-deep semi-supervised feature selection method XGBoost [45], as well as the state-ofthe-art deep semi-supervised feature selection methods LassoNet [46] and GradEnFS [47], were used for comparison (Table 5). The results show that SAGP significantly outperforms XGBoost and GradEnFS, and is also competitive with LassoNet, despite SAGP not being specifically designed for feature selection tasks.
Answer 3: The out-degree of input neurons of a well-trained SAGP can be used in areas such as model interpretability and feature selection.
this section cite: ['b42', 'b43', 'b44', 'b45', 'b46']

Section: Conclusion
We introduced SAGP, a graph-structured perceptron model with full self-assembly capabilities inspired by the growth process of the human brain. SAGP optimizes its topology and enhances its functionality through neuron growth, competition, and apoptosis. We highlighted the advantages of SAGP in terms of structural simplification and assembly speed and explored its potential in interpretability and feature selection.
Justification: This article only performs question answering and fine-tuning on LLM.
Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Econeurobiology and brain development in children: key factors affecting development, behavioral outcomes, and school interventions Year: (2024)
Ref_id:b1 Title: Bridging the gap between approximation and learning via optimal approximation by relu mlps of maximal regularity Year: (2024)
Ref_id:b2 Title: Multi-layer perceptron trainability explained via variability Year: (2021)
Ref_id:b3 Title: Whole-animal connectomes of both caenorhabditis elegans sexes Year: (2019)
Ref_id:b4 Title: Meta-learning bidirectional update rules Year: (2021)
Ref_id:b5 Title: Meta learning backpropagation and improving it Year: (2021)
Ref_id:b6 Title: Towards self-assembling artificial neural networks through neural developmental programs Year: (2023)
Ref_id:b7 Title: Evolving self-assembling neural networks: From spontaneous activity to experience-dependent learning Year: (2024)
Ref_id:b8 Title: Apoptosis in the nervous system Year: (2000)
Ref_id:b9 Title: Ant system: optimization by a colony of cooperating agents Year: (1996)
Ref_id:b10 Title: A review on genetic algorithm: past, present, and future. Multimedia tools and applications Year: (2021)
Ref_id:b11 Title: The perceptron: a probabilistic model for information storage and organization in the brain Year: (1958)
Ref_id:b12 Title: Spiking neural networks: A survey Year: (2022)
Ref_id:b13 Title: Learning rules in spiking neural networks: A survey Year: (2023)
Ref_id:b14 Title: Liquid time-constant networks Year: (2021)
Ref_id:b15 Title: Artificial evolution of plastic neural networks: a few key concepts Year: (2014)
Ref_id:b16 Title: The organization of behavior. a neuropsychological theory Year: (1949)
Ref_id:b17 Title: Regulation of synaptic efficacy by coincidence of postsynaptic aps and epsps Year: (1997)
Ref_id:b18 Title: Differentiable plasticity: training plastic neural networks with backpropagation Year: (2018)
Ref_id:b19 Title: Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity Year: (2020)
Ref_id:b20 Title: A 'self-referential'weight matrix Year: (1993)
Ref_id:b21 Title: Neural plasticity networks Year: ()
Ref_id:b22 Title: Overcoming catastrophic forgetting by neuron-level plasticity control Year: (2020)
Ref_id:b23 Title: Evolving neural networks through augmenting topologies Year: (2002)
Ref_id:b24 Title: A hypercube-based encoding for evolving largescale neural networks Year: (2009)
Ref_id:b25 Title: Developmental plasticity-inspired adaptive pruning for deep spiking and artificial neural networks Year: (2024)
Ref_id:b26 Title: Information processing in dynamical systems: Foundations of harmony theory Year: (1986)
Ref_id:b27 Title: Beyond regression: New tools for prediction and analysis in the behavioral sciences Year: (1974)
Ref_id:b28 Title: Learning representations by back-propagating errors Year: (1986)
Ref_id:b29 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b30 Title: Learning and relearning in boltzmann machines. Parallel distributed processing: Explorations in the microstructure of cognition Year: (1986)
Ref_id:b31 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b32 Title: Neuronal competition and selection during memory formation Year: (2007)
Ref_id:b33 Title: Synaptic competition in structural plasticity and cognitive function Year: (1633)
Ref_id:b34 Title: Categorical reparameterization with gumbel-softmax Year: (2016)
Ref_id:b35 Title: Estimating or propagating gradients through stochastic neurons for conditional computation Year: (2013)
Ref_id:b36 Title: Synaptic pruning by microglia is necessary for normal brain development Year: (2011)
Ref_id:b37 Title: Competition among native and invasive Impatiens species: the roles of environmental factors, population density and life stage Year: (2015)
Ref_id:b38 Title: Brain charts for the human lifespan Year: (2022)
Ref_id:b39 Title: On graph neural networks versus graph-augmented mlps Year: (2020)
Ref_id:b40 Title: Long short-term memory Year: (1997)
Ref_id:b41 Title: Gradient-based learning applied to document recognition Year: (1998)
Ref_id:b42 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b43 Title: Graph attention networks Year: (2017)
Ref_id:b44 Title: Xgboost: A scalable tree boosting system Year: (2016)
Ref_id:b45 Title: Lassonet: A neural network with feature sparsity Year: (2021)
Ref_id:b46 Title: Supervised feature selection via ensemble gradient information from sparse neural networks Year: (2024)
