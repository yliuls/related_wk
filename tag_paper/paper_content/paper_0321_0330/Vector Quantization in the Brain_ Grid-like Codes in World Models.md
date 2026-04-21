Title: Vector Quantization in the Brain: Grid-like Codes in World Models
Abstract: We propose Grid-like Code Quantization (GCQ), a brain-inspired method for compressing observation-action sequences into discrete representations using grid-like patterns in attractor dynamics. Unlike conventional vector quantization approaches that operate on static inputs, GCQ performs spatiotemporal compression through an action-conditioned codebook, where codewords are derived from continuous attractor neural networks and dynamically selected based on actions. This enables GCQ to jointly compress space and time, serving as a unified world model. The resulting representation supports long-horizon prediction, goal-directed planning, and inverse modeling. Experiments across diverse tasks demonstrate GCQ's effectiveness in compact encoding and downstream performance. Our work offers both a computational tool for efficient sequence modeling and a theoretical perspective on the formation of grid-like codes in neural systems.

Section: Introduction
VQ-VAE [1] introduces discrete latent variables into the autoencoding framework through vector quantization (VQ) [2], allowing the model to compress high-dimensional continuous inputs into discrete, tokenized representations. This capability has led to its widespread application across various domains, including images [3,4], video [5], speech [6], actions [7], and multimodal data [8], demonstrating its versatility in handling complex, diverse inputs. The success of VQ-VAE underscores the utility of compressing inputs into reusable codes as a general computational strategy for preprocessing and organizing data across a wide range of tasks.
Biological systems face the similar challenge: how to process and represent high-dimensional, continuous inputs arising from multiple sensory and motor modalities. In parallel, the brain exhibits grid-like codes (GCs), which serve as general-purpose neural patterns for encoding information. GCs are extensively observed across various brain regions. Initially identified in the medial entorhinal cortex for spatial navigation [9], GCs have since been observed in the neocortex [10,11,12] and associated with representing abstract concepts beyond space, such as time and relational knowledge [10,11,13,14]. This widespread neural activity is characterized by bump-like patterns, periodicity, and typically disentangled representations.
Building on this insight, we propose a brain-inspired VQ method, Grid-like Code Quantization (GCQ), which uses the principles of GCs to structure the codebook. Specifically, we use continuous attractor neural networks (CANNs) [15,16,17] to generate grid-like activity patterns, where each stable state-bump-acts as a codeword. Due to the finite number of neurons, these bumps naturally form a discretized representation [18]. Unlike traditional VQ methods that use a static codebook, GCQ introduces an action-conditioned codebook: a dynamic set of codewords formed by CANN-generated bumps whose transitions are modulated by actions. This enables GCQ to perform quantization not on isolated observations, but on observation-action sequences, allowing the representations to capture temporal dependencies and behavioral context. Moreover, assigning distinct CANNs to different action types naturally yields disentangled representations, facilitating generalization and compositionality.
The overall GCQ pipeline follows an encoder-quantizer-decoder architecture, adapted for actionconditioned sequence compression (Fig. 2). Specifically, the model processes an observation-action sequence, where the action sequence is used to construct an action-conditioned codebook, and the observation sequence is passed through the encoder to produce a corresponding latent sequence. This latent sequence is then quantized via template matching with the action-conditioned codebook. The matched codewords are passed to the decoder, which reconstructs the original observation sequence. Since the codebook is fixed, training requires only a commitment loss and a reconstruction loss. To enable gradient flow through the discrete quantization step, we use a straight-through estimator (STE).
GCQ is a dynamic compression approach that operates on observation-action sequences, and therefore serves as a form of world model [19,20]. Unlike prior world models that rely on a two-stage design to separately compress space and time-typically using models like VQ-VAE for static spatial observations and autoregressive models [21] for temporal dynamics-GCQ performs spatial and temporal compression jointly.
In summary, our contributions are as follows:
• To the best of our knowledge, GCQ is the first model to unify spatial and temporal compression through an action-conditioned quantization process. This enables direct compression of observation-action sequences, offering an integrated alternative to conventional two-stage world models. (Sec. 4)
• GCQ's spatiotemporal compression yields a cognitive map, which supports long-horizon prediction, goal-directed planning, and the derivation of an inverse model. In particular, goal-directed planning becomes computationally simple, as it reduces to finding a sequence of valid bump transitions on the map. (Sec. 5).
• GCQ offers insights into the formation of GCs in the brain, enhancing our understanding of neural representations (Sec. 6).
2 Related Work VQ methods Vanilla VAEs [22] often suffer from posterior collapse in their latent spaces when compressing high-dimensional data [23], impairing downstream tasks. VQ-VAEs [1] address this by enforcing a structured latent space through discretization. Due to their superior compression efficiency and tokenization paradigm, VQ has become a standardized module in single-modal preprocessing pipelines in machine learning [3,4]. In multimodal settings, these compressed tokens further act as a universal interface across modalities [5]. Meanwhile, numerous studies have proposed diverse codebook designs to enhance compression rates [24,25,26]. Unlike most learnable codebooks, FSQ uses a predefined codebook. Similarly, our GCQ utilizes a fixed codebook derived from continuous attractor dynamics. Critically, our method diverges from conventional VQ approaches by performing sequence-to-sequence template matching rather than single-frame matching.
World models [19,20] provide a framework for predicting future observations conditioned on actions. Most world models based on encoder-decoder architectures first compress observations using a VAE, and then model temporal dynamics in the latent space using temporal predictors such as RNNs [27,28], Transformers [29], S4 models [30], or continuous Hopfield networks [31]. These approaches typically follow a two-stage design, with spatial and temporal compression handled separately. In contrast, GCQ is also an encoder-decoder world model, but it performs spatial and temporal compression jointly. There also exist decoder-only world models [32] that skip explicit compression and directly predict future observations. However, these models often struggle with planning due to the high computational cost of operating in the raw observation space. GCQ, by compressing both space and time into a compact latent representation, enables more efficient planning and inference.
Cognitive map with CANNs Unlike classical attractor networks [33]-which store discrete, unstructured patterns-CANNs encode structured patterns organized by metric relationships. This geometric regularity facilitates flexible state transitions through predefined operators [34], enabling operations like metric-based navigation and relational inference. Recent advances [35] have harnessed predefined CANNs as structured latent states for representation learning, empirically validating their ability to model neural population dynamics. Further work [36] proposes that structured latent spaces can map biologically to the entorhinal-hippocampal loop, a core circuit for spatial and episodic memory. However, existing implementations rely on biologically constrained online learning, which limits scalability. Our GCQ framework uses offline learning, enhancing parallelism and enabling application to large-scale datasets.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b9', 'b10', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b0', 'b2', 'b3', 'b4', 'b23', 'b24', 'b25', 'b18', 'b19', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35']

Section: CANNs and Template Matching
In this section, we will briefly introduce CANNs, explain how they can form bumps as attractor states.
In parallel, for VQ, the latent state obtained by the encoder must undergo template matching with codewords. We will demonstrate that CANNs inherently implement template matching between representations and bump states through their intrinsic dynamics. Finally, we will show how transitions between distinct attractor states can be mediated by actions.
GCs can naturally be modeled by bumps in CANNs (Fig. 1E). The formation of CANNs does not require complex optimization but relies on translation-invariant connectivity and periodic boundary conditions. CANNs have been widely used as canonical models to elucidate the encoding of features in neural systems, including, for example, the encoding of orientation [37], head direction [38] and spatial location [17,39]. CANNs can be expressed through various mathematical formulations. Here, we adopt a relatively concise form [16] to demonstrate their principles. We consider N 2 neurons distributed on a toroidal (S 1 × S 1 ) surface. These neurons are indexed by their positions on the torus θ ∈ {θ i } N i=1 and φ ∈ {φ j } N j=1 , where θ i and φ j are uniformly distributed over (-π, π] (Fig. 1A). Let U θ,φ (t) and r θ,φ (t) denote the synaptic input and firing rate, respectively, of the neuron located at (θ, φ) at time t. The dynamics of the CANN are governed by:
τ ∂U θ,φ (t) ∂t = -U θ,φ (t) + ρ θ ′ ,φ ′ W θ,φ (θ ′ , φ ′ )r θ ′ ,φ ′ (t) + I θ,φ (t),(1)
where τ is the synaptic time constant and ρ is the neuronal density. W θ,φ (θ ′ , φ ′ ) is the recurrent neuronal connections weights between neuron (θ, φ) and neuron (θ ′ , φ ′ ),
W θ,φ (θ ′ , φ ′ ) = J 2πa 2 exp - ||θ -θ ′ || 2 S + ||φ -φ ′ || 2 S 2a 2 . (2
)
The norm ∥•∥ S denotes the shortest path between two points on the circle, ensuring periodic boundary and translation-invariant conditions. The parameters J and a control the strength and width of the Gaussian connectivity, respectively. The nonlinear relationship between the firing rate r θ,φ (t) and the synaptic input U θ,φ (t) is implemented by divisive normalization, which is written as,
r θ,φ (t) = U 2 θ,φ (t) 1 + kρ θ ′ ,φ ′ W θ,φ (θ ′ , φ ′ )U 2 θ ′ ,φ ′ (t) ,(3)
where k controls the normalization strength. In reality, divisive normalization could be implemented by shunting inhibition [40].
Previous studies [16,41] have established that the CANN dynamics governed by Eq. ( 1) possess N 2 stationary states (attractors) when the external input I θ,φ (t) = 0 (Fig. 1B). Each state corresponds to a 2D Gaussian bump on the torus, centered at coordinates (θ, φ), with the firing rate of the neuron at position (θ ′ , φ ′ ) given by:
e θ,φ (θ ′ , φ ′ ) = A exp - ∥θ -θ ′ ∥ 2 S + ∥φ -φ ′ ∥ 2 S 2a 2 ,(4)
where
A = 1 + (1 -32πa 2 k/J 2 ρ) 1/2 /(4πa 2 kρ) is the amplitude. When I θ,φ (t)
is a constant input, prior work [42] demonstrated that after its removal, the network converges to an attractor determined by:
θ * , φ * = max θ,φ θ ′ ,φ ′ e θ,φ (θ ′ , φ ′ )I θ ′ ,φ ′ ,(5)
which demonstrates that CANN dynamics effectively perform template matching between the input I and the N 2 attractors according to their inner product. (Fig. 1C).
The bump in CANNs exhibit high mobility, enabling controlled movement through mechanisms such as: anti-symmetric connections [43], negative feedback [44,43], velocity neurons [45]. Such bump displacements correspond to transitions between attractor states. For the toroidal CANN described above, each attractor can undergo local two-dimensional displacements in the θ, φ plane. We define two orthogonal action bases aligned with the θ and φ axes (Fig. 1D),
a ± θ = e θ±∆θ,φ -e θ,φ , a ± φ = e θ,φ±∆φ -e θ,φ .(6)
where ∆φ and ∆θ denote a small displacement step.
this section cite: ['b36', 'b37', 'b16', 'b38', 'b15', 'b39', 'b15', 'b40', 'b41', 'b42', 'b43', 'b42', 'b44']

Section: Grid-like Code Quantization
In this section, we first introduce the action-conditioned codebook in GCQ and the template matching process for sequences. We then describe how GCQ enables bidirectional mapping between real-world actions and latent transitions, and propose a greedy operator for measuring distances on the cognitive map to support inverse modeling and planning.
this section cite: []

Section: Action-conditioned codebook and sequence matching
We first introduce the key difference between GCQ and VQ from a high-level perspective. In the VQ method, the encoder first compresses the observation o into s, which is then matched to the closest codes in the codebook through template matching, producing ŝ. The decoder then reconstructs ô from ŝ. In GCQ, the input consists of an action-observation sequence {o 1 , a 1 , o 2 , a 2 , ..., o n }. The encoder compresses the observation sequence o 1:n = {o 1 , o 2 , ..., o n } into s 1:n , which is then matched to the closest codes in the action-conditioned codebook via template matching, yielding ŝ1:n . The decoder then reconstructs ô1:n from ŝ1:n .
𝑒 ! ⨁𝑎 !:# 𝑒 ! 𝑒 ! + 𝑎 ! 𝑒 ! + 𝑎 !:$ 𝑒 ! + 𝑎 !:% 𝑒 ! + 𝑎 !:# 𝑒 & 𝑒 & + 𝑎 ! 𝑒 & + 𝑎 !:$ 𝑒 & + 𝑎 !:% 𝑒 & + 𝑎 !:# 𝑒 & ⨁𝑎 !:# … 𝑎 ! 𝑜 ! 𝑎 $ 𝑎 % 𝑎 # 𝑠 ! 𝑒 ' 𝑜 $ 𝑜 % 𝑜 # 𝑜 ( 𝑠 $ 𝑠 % 𝑠 # 𝑠 ( ŝ ! ŝ $ ŝ % ŝ # ŝ ( 𝑒 $ … 𝑒 ' + 𝑎 ! 𝑒 ' + 𝑎 !:$ 𝑒 ' + 𝑎 !:% 𝑒 ' + 𝑎 !:# 𝑒 $ + 𝑎 ! 𝑒 $ + 𝑎 !:$ 𝑒 $ + 𝑎 !:% 𝑒 $ + 𝑎 !:# ) 𝑜 ! ) 𝑜 $ ) 𝑜 % ) 𝑜 # ) 𝑜 ( action-conditioned codebook … … … …
In GCQ, each code corresponds to an attractor in the CANN (Fig. 1B). In the previous section, we used θ and φ to index different attractors; for simplicity, we will now use natural numbers as attractor indices. Each code consists of d neurons, and the codebook contains K attractors. A simple implementation sets K = d, where each attractor's center coincides with a single neuron. Alternatively, we can set K > d, causing some attractor centers to fall between two neurons. In practice, different combinations of K and d can be selected. The state representation s i ∈ R m×d , meaning that s i is composed of m codes.
Additionally, we manually define a mapping between the action sequence a i ∈ A from the dataset and the action combinations applied to the CANNs. For notational simplicity, we hereafter use a i to refer to an action in either the original space or the CANN space. In the latter context, a i ∈ R m×d represents the composite action over m bumps, with its component a j i ∈ R d denoting the action applied to the j-th bump in Eq.( 6). Each CANN supports five distinct actions, resulting in up to 5 m possible action combinations across m CANNs. Since this mapping is injective, the discrete action space must satisfy |A| ≤ 5 m . For continuous actions, a CANN can define transitions in two directions, imposing the constraint dim(A) ≤ 2m.
After establishing the mapping, we quantize the latent representation s 1:n = {s j 1:n } m j=1 . This representation consists of a set of m parallel sequences, where each s j 1:n corresponds to a sequence from one of the m CANNs (as depicted by the dashed lines in Fig. 2). The quantization process is performed independently for each of these m sequences. For each latent sequence s j 1:n , we perform a template matching procedure. This involves comparing s j 1:n against a set of K candidate trajectories. Each candidate trajectory is generated by applying the known action sequence a j 1:n-1 to a base bump state e i . We denote this operation as:
e i ⊕ a j 1:n-1 = {e i , e i + a j 1 , . . . , e i + a j 1:n-1 },(7)
where e i + a j 1:n-1 = e i + n-1 t=1 a j t . The index k of the best-matching codeword for the j-th latent sequence is found by minimizing a distance metric (e.g., the L2 norm) between the latent sequence and each of the K candidate trajectories: Finally, the quantized sequence ŝj 1:n is constructed using this optimal codeword e kj . The complete quantized representation ŝ1:n is the collection of these individually quantized sequences:
k j = arg min i∈{1,..,K} ||s j 1:n -(e i ⊕ a j 1:n-1 )|| (8
) (A) Long-horizon prediction (B) Goal-directed planning 𝑎 ! 𝑜 ! 𝑎 " 𝑠 ! 𝑒 # 𝑜 " 𝑜 $ 𝑠 " 𝑠 $ ŝ ! ŝ " ŝ $ 𝑒 " … 𝑒 # + 𝑎 ! 𝑒 # + 𝑎 !:" 𝑒 " + 𝑎 ! 𝑒 " + 𝑎 !:" … … z 𝑒 ! ⨁𝑎 !:" 𝑒 & ⨁𝑎 !:" … ŝ ' ŝ ( 𝑒 # + 𝑎 !:$ 𝑒 # + 𝑎 !:' 𝑒 " + 𝑎 !:$ 𝑒 " + 𝑎 !:' ) 𝑜 ' ) 𝑜 ( … … 𝑎 $ 𝑎 ' 𝑜 ! 𝑠 )*+, ŝ )*+, … z 𝑒 ! 𝑒 & … 𝑜 )*+, 𝑠 ! ŝ ! … ŝ )*+, ⊝ ŝ ! 𝑎 ! 𝑠 " ŝ " … 𝑜 " ŝ )*+, ⊝ ŝ " 𝑎 " 𝑜 $ 𝑠 $ ŝ $ … ŝ )*+, ⊝ ŝ $ end if 𝑎 -== 𝑎 .
ŝj 1:n = e kj ⊕ a j 1:n-1 , and ŝ1:n = {ŝ j 1:n } m j=1(9)
When computing the loss in GCQ using backpropagation (BP), we adopt the same straight-through estimator (STE) as in the VQ method, copying gradients from the decoder input to the encoder output to enable gradient flow to the encoder. GCQ uses two loss terms: a reconstruction loss and a commitment loss:
L = ||o 1:n -ô1:n || 2 + β ∥s 1:n -sg [ŝ 1:n ]∥ 2 (10
)
where sg[•] denotes the stop-gradient operation and β adjusts the strength of the commitment loss.
In GCQ, the encoder and decoder are not designed in the same way as in conventional VQ models. Traditional VQ architectures often use ResNet-based building blocks, which provide each code with only a limited receptive field. As a result, modifying a single code typically leads to only local changes in the reconstructed observation. In contrast, GCQ assigns each code to an action, and altering the action can result in global changes to the observation. This necessitates that each code has access to global information during encoding and decoding. To address this, we explore three architectural variants for the encoder and decoder: (1) ResNet followed by a fully connected layer, (2) ViT [46], and (3) a hybrid of ResNet and ViT. Among these, ViT achieves the best trade-off in terms of parameter efficiency, training stability, and overall performance (Table .1).
this section cite: ['b45']

Section: Operations on cognitive map
GCQ uses a structured latent space, allowing an agent's actions in the real environment to correspond to simple movements of bumps within the latent space. In effect, GCQ constructs a space defined by bump dynamics, which can be interpreted as a cognitive map. By establishing a mapping between observations and this map, actions in the real space can be projected onto the map to determine position changes, and conversely, movements within the map can be mapped back to real-space actions. This bidirectional mapping enables GCQ to support both inverse modeling and goal-directed planning. Specifically, to compute the distance between two states s i and s j , we define an operation on the cognitive map. Since bump movements are action-driven and only valid actions produce feasible transitions, we introduce the following operation:
s i ⊖ s j = arg min a∈A |s j + a -s i |.(11)
This operation represents a greedy step: it selects the best valid action a that moves s j one step closer to s i .
this section cite: []

Section: Experiment
As a spatiotemporal compression model, GCQ is first evaluated in ablation studies to demonstrate its ability to compress and reconstruct observations. We then show that GCQ, when used as a world model, supports long-horizon prediction, goal-directed planning, and inverse modeling. Compared to traditional two-stage models, GCQ exhibits superior performance in long-range prediction tasks.
this section cite: []

Section: Datasets.
We evaluate GCQ on four datasets, all of which contain image-based observations. The 2DMaze [47] dataset is a virtual environment where actions correspond to the agent's movements. Each observation contains a full view of the maze, providing complete information. The Google Street View (GSV) dataset represents real-world environments with partial observations; the actions include both translational movements and rotational head turns in two directions. In the MPI3D [48] and 3DShapes [49] datasets, actions are defined as abstract feature-level changes.
Baselines. We compare GCQ with traditional two-stage world models. VQ-VAE is used in the first stage for spatial compression. The codebook size in GCQ and VQ-VAE is kept the same for a fair comparison. For modeling temporal relationships, we use a UNet that predicts the next latent state s t+1 based on the current latent state s t and action a t . We refer to this baseline as 'VQ+UNet.' For action embedding in the UNet, we follow the approach from LAPO [7]. To further model temporal dependencies, we also adopt a Transformer-based architecture following TransDreamer [29]. We refer to this baseline as 'VQ+Transformer.'
this section cite: ['b46', 'b47', 'b48', 'b6', 'b28']

Section: Evaluation Metrics.
To evaluate the quality of the model-generated observations, we report peak signal-to-noise ratio (PSNR) for pixel-level reconstruction fidelity, and use the Fréchet Inception Distance (FID) [50] to assess the quality of generated images.
this section cite: ['b49']

Section: Ablations
We first conducted ablation experiments on the GSV dataset. Table 1 presents the performance of three different encoder-decoder network building blocks. It can be observed that the ViT and Hybrid models achieve better performance with fewer parameters. However, during the experiments, we found that the Hybrid model was less stable in training and converged more slowly than ViT. Therefore, unless otherwise specified, all subsequent experiments utilized the ViT-structured network. The GCQ exhibits scalability with model size similar to VQ+UNet, both in reconstruction and prediction. (Fig. 4A,B). We also make the bump-like codes in the codebook learnable by using the following loss function:
L = ||o 1:n -ô1:n || 2 + β ∥s 1:n -sg [ŝ 1:n ]∥ 2 + γ ∥sg [s 1:n ] -ŝ1:n ∥ 2 (12
)
However, our experiments show that making the codes learnable actually degrades performance. We attribute this to the fact that, unlike the relatively simple codes in VQ, our codes exhibit more complex dynamic relationships. Allowing the codes themselves to be trained may therefore reduce training stability.
this section cite: []

Section: Model
FIDp↓ PSNRp↑ GCQ (fixed) 43.41 27.77 GCQ (learnable) 47.76 24.48 Table 2: Effect of learnable vs. fixed codes in GCQ.
Long-horizon prediction. After being initialized with an observation-action sequence, GCQ can perform actions directly in the latent space to predict future observations (Fig. 3A). Notably, its prediction performance remains stable regardless of the length of the initialization sequence (Fig. 4C; Fig. 4E, rows 1-2). As shown in Fig. 4D, the performance of the VQ method degrades as the prediction horizon increases, whereas GCQ maintains robust predictive quality due to its stable latent structure (Fig. 4E, rows 2-3). This is a key advantage of GCQ: by constructing a consistent cognitive map, it effectively addresses the instability issues commonly seen in current world models [51]-such as inaccurate predictions after completing a full rotation in the environment (Fig. 4E, rows 4-5). GCQ also demonstrates strong zero-shot prediction capabilities on relatively simple datasets. As shown in Fig. 5, rows 1-2, the model produces reasonable predictions in environments it has never encountered during training. Furthermore, by treating abstract feature transitions as a form of action, GCQ can also be used to predict observation changes driven by abstract-level variations. These predictions likewise exhibit long-range stability (Fig. 5, rows 3-4).
Goal-directed planning. Given a goal and an initial position, the GCQ can utilize the distance in the cognitive map to generate the most desirable action for the current step. After executing the action, a new observation is obtained, and this process is iterated, continuously reducing the distance to the goal in the cognitive map until the goal is reached (Fig. 3B, Fig. 6 rows, 1-2). The computation of the action at each step is of constant complexity.
Inverse model. Given a sequence of observations, the GCQ can first map them onto the cognitive map and then use goal-directed planning to determine the action or sequence of actions between adjacent observations, thus implementing the inverse model. The corresponding action sequence can be applied to the latent representation of another observation, and using the prediction capability, the generated sequence under this set of actions can be obtained (Fig. 6 rows, 3-4). Goal-directed planning Inverse modeling Figure 6: Rows 1-2: Goal-directed planning. The first patch shows the trajectory after planning, with orange indicating the starting point, red indicating the endpoint, and blue representing the planned route. The subsequent red-framed patches represent the endpoint, orange-framed patches represent the starting point, and blue-framed patches represent intermediate observations encountered during the process. Rows 3-4: Inverse modeling. The top row displays the given observation sequence. The bottom row starts with the first patch showing the action trajectory inferred from the observation sequence, with orange indicating the given initial observation, followed by the sequence generated based on the action trajectory.
this section cite: ['b50']

Section: Discussion
In this work, we introduced GCQ, a brain-inspired framework for compressing observation-action sequences into discrete, structured representations. GCQ uses continuous attractor dynamics to generate grid-like codewords, and selects them in an action-conditioned manner to capture both spatial and temporal dependencies. This spatiotemporal quantization process produces compact latent representations that serve as cognitive maps, enabling long-horizon prediction, goal-directed planning, and inverse modeling. Our experiments demonstrate that GCQ supports generalization across tasks while offering interpretability through its structured latent space.
this section cite: []

Section: References
Ref_id:b0 Title: Neural discrete representation learning Year: (2017)
Ref_id:b1 Title: Vector quantization Year: (1984)
Ref_id:b2 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b3 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b4 Title: Align your latents: High-resolution video synthesis with latent diffusion models Year: (2023)
Ref_id:b5 Title: Vector-quantized variational autoencoder for phase-aware speech enhancement Year: (2022)
Ref_id:b6 Title: Learning to act without actions Year: (2023)
Ref_id:b7 Title: Vila-u: a unified foundation model integrating visual understanding and generation Year: (2024)
Ref_id:b8 Title: Microstructure of a spatial map in the entorhinal cortex Year: (2005)
Ref_id:b9 Title: Evidence for grid cells in a human memory network Year: (2010)
Ref_id:b10 Title: Organizing conceptual knowledge in humans with a gridlike code Year: (2016)
Ref_id:b11 Title: Direct recordings of grid-like neuronal activity in human spatial navigation Year: (2013)
Ref_id:b12 Title: Grid cell responses in 1d environments assessed as slices through a 2d lattice Year: (2016)
Ref_id:b13 Title: Mapping of a non-spatial dimension by the hippocampal-entorhinal circuit Year: (2017)
Ref_id:b14 Title: Dynamics of pattern formation in lateral-inhibition type neural fields Year: (1977)
Ref_id:b15 Title: Dynamics and computation of continuous attractors Year: (2008)
Ref_id:b16 Title: Accurate path integration in continuous attractor network models of grid cells Year: (2009)
Ref_id:b17 Title: Maintaining and updating accurate internal representations of continuous variables with a handful of neurons Year: (2024)
Ref_id:b18 Title: Making the world differentiable: On using self-supervised fully recurrent n eu al networks for dynamic reinforcement learning and planning in non-stationary environments Year: (1990)
Ref_id:b19 Title: A path towards autonomous machine intelligence version 0.9 Year: ()
Ref_id:b20 Title: Language modeling is compression Year: (2023)
Ref_id:b21 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b22 Title: Disentangling factors of variation with cycle-consistent variational auto-encoders Year: (2018)
Ref_id:b23 Title: Variable-rate discrete representation learning Year: (2021)
Ref_id:b24 Title: Theory and experiments on vector quantized autoencoders Year: (2018)
Ref_id:b25 Title: Finite scalar quantization: Vq-vae made simple Year: (2023)
Ref_id:b26 Title: A recurrent latent variable model for sequential data Year: (2015)
Ref_id:b27 Title: Learning latent dynamics for planning from pixels Year: (2019)
Ref_id:b28 Title: Transdreamer: Reinforcement learning with transformer world models Year: (2022)
Ref_id:b29 Title: Mastering memory tasks with world models Year: (2024)
Ref_id:b30 Title: Generalisation of structural knowledge in the hippocampal-entorhinal system Year: (2018)
Ref_id:b31 Title: Navigation world models Year: (2024)
Ref_id:b32 Title: Learning patterns and pattern sequences by self-organizing nets of threshold elements Year: (1972)
Ref_id:b33 Title: The motion planning neural circuit in goal-directed navigation as lie group operator search Year: (2024)
Ref_id:b34 Title: Predictive learning in energy-based models with attractor structures Year: (2025)
Ref_id:b35 Title: Episodic and associative memory from spatial scaffolds in the hippocampus Year: (2025)
Ref_id:b36 Title: Theory of orientation tuning in visual cortex Year: (1995)
Ref_id:b37 Title: Representation of spatial orientation by the intrinsic dynamics of the headdirection cell ensemble: a theory Year: (1996)
Ref_id:b38 Title: Path integration and the neural basis of the'cognitive map' Year: (2006)
Ref_id:b39 Title: Shunting inhibition modulates neuronal gain during synaptic excitation Year: (2003)
Ref_id:b40 Title: A moving bump in a continuous manifold: A comprehensive study of the tracking dynamics of continuous attractor neural networks Year: (2010)
Ref_id:b41 Title: Reading population codes: a neural implementation of ideal observers Year: (1999)
Ref_id:b42 Title: Spike frequency adaptation implements anticipative tracking in continuous attractor neural networks Year: (2014)
Ref_id:b43 Title: Attractor dynamics with synaptic depression Year: (2010)
Ref_id:b44 Title: Translation-equivariant representation in recurrent networks with a continuous manifold of attractors Year: (2022)
Ref_id:b45 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b46 Title: Leveraging procedural generation to benchmark reinforcement learning Year: (2019)
Ref_id:b47 Title: On the transfer of inductive bias from simulation to the real world: a new disentanglement dataset Year: (2019)
Ref_id:b48 Title:  Year: (2018)
Ref_id:b49 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b50 Title: Facing off world model backbones: Rnns, transformers, and s4 Year: (2023)
Ref_id:b51 Title: Receptive fields and functional architecture of monkey striate cortex Year: (1968)
Ref_id:b52 Title: Emergence of grid-like representations by training recurrent neural networks to perform spatial localization Year: (2018)
Ref_id:b53 Title: Vectorbased navigation using grid-like representations in artificial agents Year: (2018)
Ref_id:b54 Title: A unified theory for the origin of grid cells through the lens of pattern formation Year: (2019)
Ref_id:b55 Title: Predictive learning as a network mechanism for extracting low-dimensional latent space representations Year: (2021)
Ref_id:b56 Title: The tolman-eichenbaum machine: unifying space and relational memory through generalization in the hippocampal formation Year: (2020)
Ref_id:b57 Title: Disentanglement with biological constraints: A theory of functional cell types Year: (2022)
Ref_id:b58 Title: Predictive sequence learning in the hippocampal formation Year: (2024)
Ref_id:b59 Title: When and why grid cells appear or not in trained path integrators Year: (2022)
Ref_id:b60 Title: No free lunch from deep learning in neuroscience: A case study through models of the entorhinal-hippocampal circuit Year: (2022)
Ref_id:b61 Title: Experience-independent emergence of toroidal and ring manifolds in the entorhinal cortex Year: (2024)
Ref_id:b62 Title: Toroidal topology of population activity in grid cells Year: (2022)
Ref_id:b63 Title: Imagenet: A largescale hierarchical image database Year: ()
