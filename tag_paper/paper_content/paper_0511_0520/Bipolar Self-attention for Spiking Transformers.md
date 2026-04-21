Title: Bipolar Self-attention for Spiking Transformers
Abstract: Harnessing the event-driven characteristic, Spiking Neural Networks (SNNs) present a promising avenue toward energy-efficient Transformer architectures. However, existing Spiking Transformers still suffer significant performance gaps compared to their Artificial Neural Network counterparts. Through comprehensive analysis, we attribute this gap to these two factors. First, the binary nature of spike trains limits Spiking Self-attention (SSA)'s capacity to capture negative-negative and positive-negative membrane potential interactions on Querys and Keys. Second, SSA typically omits Softmax functions to avoid energy-intensive multiplyaccumulate operations, thereby failing to maintain row-stochasticity constraints on attention scores. To address these issues, we propose a Bipolar Self-attention (BSA) paradigm, effectively modeling multi-polar membrane potential interactions with a fully spike-driven characteristic. Specifically, we demonstrate that ternary matrix multiplication provides a closer approximation to real-valued computation on both distribution and local correlation, enabling clear differentiation between homopolar and heteropolar interactions. Moreover, we propose a shift-based Softmax approximation named Shiftmax, which efficiently achieves low-entropy activation and partly maintains row-stochasticity without non-linear operation, enabling precise attention allocation. Extensive experiments show that BSA achieves substantial performance improvements across various tasks, including image classification, semantic segmentation, and event-based tracking. These results establish its potential as a fundamental building block for energy-efficient Spiking Transformers.

Section: Introduction
As the core computational unit of Transformers [5,15,11,32,9], self-attention mechanism dynamically models the global dependencies among sequence elements, overcoming the long-range dependency challenges [33,10]. However, its' O(N 2 d) computational complexity [18,48] incurs an exponential rise during both training and inference, restricting its application in many resourceconstrained environments. Consequently, how to develop energy-efficient and high-performance Transformers remains a critical research focus.
Spiking Neural Networks (SNNs) [23,12] have gained significant attention due to their brain-inspired dynamics [17,24]. Spiking neurons fire discrete spikes only when activated, remaining silent otherwise. Compared with Artificial Neural Networks (ANNs) that rely on multiply-accumulate (MAC) computation, the spike-driven mechanism [61,43,44] in SNNs supports sparse accumulate (AC) operations [2]. Such sparse spike-based computation [20,56] delivers significant power efficiency, particularly on neuromorphic platforms such as Tianjic [26,8,22] and Loihi [6,25]. Recently, numerous researchers focus on developing Spiking Transformers, including Spikformer [70], Spikingformer [67], Spike-driven Transformers [51,49,52], SpikingResformer [30], and QKformer [68]. These approaches enhance the performance ceiling of SNNs in various tasks [64,45,36,31,62,34], demonstrating that Spiking Transformers achieve a trade-off between high performance and efficiency. As a core component of Spiking Transformer architectures, the Spiking Self-ttention (SSA) [70,51] computational paradigm represents the critical factor limiting their performance ceiling. As shown in Fig. 1, Vanilla Self-attention (VSA) in ANNs effectively captures magnitude and multi-polarity information by computing Score ′ through query (Q) and key (K) correlation. Meanwhile, it calculates the Score using the Softmax function under row-stochasticity constraint. In contrast, SSA in SNNs operates with binary spike trains (0 or 1), which not only sacrifices quantization precision but also disregards negative-negative and positive-negative interactions. Furthermore, to maintain energy efficiency, SSA typically omits Softmax function, resulting in severely imbalanced row-stochasticity in Score. These issues limit SSA's ability to effectively compute attention allocation between Q and K, causing SSA more like a simplified Token Mixer [54,55]. Therefore, how to overcome these limitations is crucial for pushing Spiking Transformers beyond their current performance bottlenecks.
In this paper, we propose a Bipolar Self-attention (BSA) paradigm to effectively address these issues. Unlike the SSA paradigm that exclusively captures positive-positive Q-K interactions, BSA employs ternary spiking neurons [14,35] to comprehensively process both homopolar and heteropolar interaction patterns in Q-K correlation computation. We theoretically demonstrate that ternary matrix multiplication more closely approximates real-valued computation in terms of both distributional similarity and local correlation than binary part. Moreover, we propose the innovative Shiftmax method, which approximates Softmax's low-entropy activation characteristics and row-stochasticity constraints through energy-efficient bit-shift operations. Finally, we conduct extensive experiments across diverse tasks including image classification [70,68], semantic segmentation [49,52], and event-based tracking [58,28], consistently demonstrating that BSA delivers substantial performance improvements. The main contributions of our work are outlined as follows:
• We first identify two limitations of Spiking Self-attention: (1) binary matrix products exclusively capture positive-positive correlations while neglecting negative-negative and positive-negative polarity features, leading to a complete loss of polarity information. (2) without Softmax, attention scores across different rows exist on incomparable scales, rendering attention allocation ineffective. These deficiencies prevent SSA from fully harnessing the potential of the self-attention mechanism.
• We propose Bipolar Self-attention (BSA) to overcome these limitations. BSA employs ternary matrix products to extract Q-K correlations, comprehensively process different polarity interaction patterns. In addition, we propose a Shiftmax method to approximate Softmax that achieves low-entropy activation and maintains partial row-stochasticity without non-linear operations, enabling precise attention allocation.
• Extensive experiments demonstrate that BSA achieves significant performance improvements across various advanced Spiking Transformers on ImageNet-1K. Furthermore, our method establishes state-of-the-art performance in both semantic segmentation and eventbased tracking tasks compared to existing SNNs approaches.
this section cite: ['b4', 'b14', 'b10', 'b31', 'b8', 'b32', 'b9', 'b17', 'b47', 'b22', 'b11', 'b16', 'b23', 'b60', 'b42', 'b43', 'b1', 'b19', 'b55', 'b25', 'b7', 'b21', 'b5', 'b24', 'b69', 'b66', 'b50', 'b48', 'b51', 'b29', 'b67', 'b63', 'b44', 'b35', 'b30', 'b61', 'b33', 'b69', 'b50', 'b53', 'b54', 'b13', 'b34', 'b69', 'b67', 'b48', 'b51', 'b57', 'b27']

Section: Related Work
Recently, growing attention has been paid to energy-efficient Spiking Transformers [4,50]. For instance, Spikformer [70] first proposes Spiking Self-attention computation, establishing the first Spiking Transformer. Building on this, Spikingformer [67] introduces a hardware-friendly residual learning architecture that avoids non-spike computations in SNNs. Then, the Spike-driven Transformers (SDT-V1) [51] incorporates the Hadamard product into the self-attention module, achieving a fully spike-driven mechanism. Furthermore, SpikingResformer [30] integrates a Dual Spike selfattention module, enhancing both performance and energy efficiency. Recently, QKformer [68] and Spike-driven Transformer-V3 [52] both significantly elevate the performance ceiling of Spiking Transformers. Nonetheless, these models primarily utilize self-attention as a token mixer [54], without paying attention to exploring effective similarity calculations suited to spike trains [60,1]. Therefore, designing an innovative spiking self-attention mechanism that fully leverages self-correlation computation under spike-driven characteristics is crucial for further advancement.
this section cite: ['b3', 'b49', 'b69', 'b66', 'b50', 'b29', 'b67', 'b51', 'b53', 'b59', 'b0']

Section: Preliminary

this section cite: []

Section: Vanilla Self-attention
The self-attention mechanism effectively captures global dependencies within sequences by dynamically allocating attention across tokens [33,29,13]. Given an input matrix X ∈ R n×d , where n denotes sequence length and d represents embedding dimensionality, the mechanism initially projects X into query Q, key K, and value V matrices via distinct parameter matrices W Q , W K , and
W V : X = LN(X), Q = Linear( X), K = Linear( X), V = Linear( X).(1)
Here, LN denotes layer normalization, and Linear refers to a fully connected layer. To determine contextual relationships, the model computes token-wise similarities through dot-product operations between Q and K, generating an attention score matrix:
Score ′ = Q × K ⊤ , Score = Softmax Score ′ √ d , Attn = Score × V.(2)
The mechanism applies a scaling factor √ d to mitigate gradient instability issues. Subsequently, the Softmax function normalizes these scaled Score to a probability distribution, ensuring attention weights sum to unity while emphasizing relevant tokens and suppressing irrelevant ones.
this section cite: ['b32', 'b28', 'b12']

Section: Spiking Self-attention
Building on VSA, Spikformer [70] introduced SSA. For an input matrix X ∈ R n×d , queries Q, keys K, and values V are first computed via learnable weight matrices and then converted into spike trains by binary spiking neurons for subsequent processing. The dynamics of these neurons are given by:
U [t + 1] = H[t] + X[t + 1],(3)
S[t + 1] = Θ(U [t + 1] -V th ),(4)
H[t + 1] = V reset S[t + 1] + τ U [t + 1](1 -S[t + 1]).
(5) X[t + 1] denotes the input current, while H[t] and U [t] represent the pre-synaptic and post-synaptic membrane potentials, respectively. The Heaviside function Θ(•) is employed for spike generation. If a spike occurs (S[t + 1] = 1), H[t] resets to V reset ; otherwise, U [t + 1] decays with a time constant τ and feeds into H[t + 1]. Here a spiking neuron layer is denoted as SN (•), which takes X[t + 1] as input and produces the spike S[t + 1] as output. Then the Q, K and V in SSA can be decribed as:
Q = SN (BN(Conv 1 (X)), K = SN (BN(Conv 2 (X)), V = SN (BN(Conv 3 (X)),(6)
where Q, K, V ∈ R T ×n×d , BN(•) denotes batch normalization and Conv(•) refers to a convolution operation. Unlike VSA, SSA omits the Softmax operation while retaining scaling to control the magnitude of the attention output. The attention output Attn is obtained by performing a matrix multiplication of the spiking Q and K, scaled by factor s, and then convert to spike trains:
Score = s • Q × K T , Attn = SN (Score × V).(7)
Therefore, SSA provides an energy-efficient self-attention computation paradigm without any nonlinear operations. It allows the ordering of the Q, K, and V matrices to be flexibly adjusted as needed, enabling the Spiking Transformer to capture global dependencies efficiently.
this section cite: ['b69']

Section: Method

this section cite: []

Section: Problem Analysis for Spiking Self-attention
As previously discussed, SSA reduces computational costs by eliminating nonlinear computations and MAC operations. However, there remains a significant performance gap compared to VSA. We thoroughly analyze the reasons and attribute them to the following parts: (1) binary Q-K Matrix multiplication cannot effectively capture polarity correlations, and (2) eliminating Softmax removes the attention score's row-stochasticity constraints. The details are elaborated in the following.
this section cite: []

Section: Incomplete Polarity Information Modeling
In VSA, Q and K are represented by continuous real-valued vectors, enabling matrix multiplication can efficiently calculate the correlation between Q and K. However, SSA typically binarizes the membrane potential information of Q and K, limiting its representation to binary values (0 or 1). This spike-driven approach not only narrows the precision of Q-K interactions, but also undermines the self-attention mechanism's capacity to detect vector polarity relationships. Consider Ω VSA = {(a 1 , a 2 ) | a i ∈ {+, -}, i = 1, 2} representing correlation types in VSA and Ω SSA = {(b 1 , b 2 ) | b i ∈ {0, 1}, i = 1, 2} denoting binary states in SSA, with ϕ : R d → {0, 1} d as a binarization mapping. Let P = {B 0 , B 1 } partition Ω SSA where B 0 = {(0, 0), (0, 1), (1, 0)}, B 1 = {(1, 1)}. We establish a surjection G : Ω VSA → P that fundamentally characterizes the polarity information loss during the binarization transformation process, such that: {(+, -), (-, +), (-, -)
Case 1: G ({(+, +)}) = B 1 , Case 2: G (
}) = B 0 .(8)
Eq.8 demonstrates that SSA completely discards the ability to compute negative-negative polarity or mixed polarity correlations. Due to the sparse activation characteristics of SNNs, the amount of discarded information far exceeds what is retained, resulting in substantial information loss. As shown in Fig. 2(a), the green-boxed region represents the positive polarity information that SSA can preserve, while the red-boxed regions indicate the entirely neglected (-, -) and (+, -) polarity information. More importantly, SSA may display entirely disparate attention regions from VSA as shown in Fig. 2(b). Thus, devising efficient mechanisms to embed polarity information offers a promising avenue for improving SSA's performance ceiling.
this section cite: []

Section: Missing Softmax Operation
SSA not only suffers from a loss of polarity information but also neglects the critical Softmax component. In VSA, Softmax operation [37] serves two pivotal roles: first, amplifying highly relevant through a low-entropy activation effect while suppressing low-value regions; second, ensuring comparability of Score across varying scales via row-stochasticity constraints. These mechanisms ensure that Softmax highlights key features while maintaining the comparability of the Score.
By contrast, SSA's binary matrix multiplication preserves limited low-entropy activation properties (detailed in the Appendix.A) and entirely lacks row-stochasticity constraints. Consequently, attention scores from different query vectors exist on disparate numerical scales. The disparity in scales makes it difficult to effectively compare the relative degree of elements across different rows during the allocation of attention to V . Therefore, how to allocate global attention on comparable scales will be another potential area for improvement.
this section cite: ['b36']

Section: Bipolar Self-attention for Spiking Transformers
To address two critical issues in existing SSA: incomplete polarity correlation and Softmax loss, we propose a novel Bipolar Self-attention (BSA) paradigm. First, we model the different polarity interactions between Querys and Keys using ternary matrix multiplication, ensuring comprehensive representation of polarity information. Furthermore, we introduce the Shiftmax method, which approximates Softmax through energy-efficient bitwise operations while preserving the critical low-entropy activation and row-normalization properties. The details are as follows:
this section cite: []

Section: Ternary Matrix Product for Comprehensive Polarity Correlation
To effectively capture the comprehensive polarity correlation in SNNs, we introduce ternary spike neurons (T SN ) [14] for characterizing the pre-synaptic membrane potential of Q and K. The dynamics of the T SN (•) are defined as follows:
U [t + 1] = H[t] + X[t + 1],(9)
S[t + 1] = Sign (U [t + 1]) • Θ (|U [t + 1]| -V th ) ,(10)
H[t + 1] = V reset S[t + 1] + τ U [t + 1](1 -|S[t + 1]|).(11)
T SN (•) fires signed spikes S[t + 1] and H[t + 1] reset to 0. The membrane potential accumulation and reset processes are similar to those in binary spike neurons. Subsequently, BSA computes Score ′ through matrix multiplication using ternary Q and K. The Ternary-valued Matrix Product (TMP) method effectively preserves the fundamental characteristics of both Binary-valued Matrix Product (BMP) in SSA and Real-valued Matrix Product (RMP) in VSA. First, it preserves the spike-driven characteristics of BMP, enabling full AC operations and achieving energy-efficient neuromorphic computation. Additionally, it retains the polarity information in the membrane potential U [t], thereby capturing polarity-related correlations comparable to RMP operations. To validate this proposition, we conduct a detailed analysis of the three matrix product, examining both their distributional relationships and local correlation. Detailed proofs are presented in Appendices.B. Theorem 1. For independent random vectors q = Q j , k = K j ∈ R d with elements Theorem.1 establishes that TMP preserves the zero-mean statistical property of RMP, while BMP exhibits systematic bias. This bias results from binary activation's exclusion of negative polarity information, corroborating our hypothesis. Furthermore, as shown in Fig. 3(c), at typical operating points (V th = 1, 0.5, corresponding to common SNNs' thresholds), TMP's variance consistently exceeds BMP's, enhancing information capacity and providing a more faithful representation of RMP's statistical characteristics. In summary, TMP approximates RMP's overall distribution more accurately than BMP, consequently enhancing Q-K correlation representation. Nevertheless, global distributional similarity does not guarantee accuracy at individual key-value pairs. Therefore, we further explore whether TMP also demonstrates stronger local correlation with RMP relative to BMP. Theorem 2. For independent variables q = Q j , k = K j ∈ R d , q i , k i ∼ N (0, σ 2 ), p = Φ(-θ σ ) representing the probability P (q > θ), θ represent the threshold for spiking neurons. We define B(•) and T (•) to represent the binary and ternary activation functions, respectively. According to the covariance calculation formulas and Theorem.1, the covariance between qk, B(q)B(k) and T (q)T (k) can be expressed as:
q i , k i ∼ N (0, σ 2 ), p = Φ(-θ σ )
Cov(qk, B(q)B(k)) = dϕ(θ) 2 • σ 2 = dϕ(θ) 2 σ 2 , Cov(qk, T (q)T (k)) = 4dϕ(θ) 2 σ 2 .
Where ϕ(θ) represents the value of the probability density function at the threshold θ.
Based on Theorem.2, we derive the approximate Pearson correlation coefficients between RMP, BMP, and TMP for each q-k pair (R, B, T), which can be formulated as follows:
ρ(R, B) = Cov(R, B) Var(R)Var(B) = ϕ(θ) 2 pσ 1 -p 2 , ρ(R, T ) = Cov(R, T ) Var(R)Var(T ) = 2ϕ(θ) 2 pσ . (12
)
By analyzing the ratio of these correlation coefficients, we obtain: ρ(R,T ) ρ(R,B) = 2 1 -p 2 , where p ranges from (0, 1). For typical spike firing rate (0.1 < p < 0.4), TMP provides approximately 2× local correlation than BMP. In conclusion, both in terms of distribution and local correlation, TMP is more closer to RMP compared to BMP, thereby better capturing the correlation of Q and K.
this section cite: ['b13']

Section: Shiftmax for Energy-efficiency Softmax Alternative
As previously mentioned, the distribution of TMP mirrors the bell-shaped distribution as RMP in ANNs. Empirical evidences [18,27] indicate that this distribution necessitates the Softmax function to generate effective self-attention scores. However, Softmax needs massive computing resources conflicting with the energy efficiency of SNNs. To address this, we propose an energy-efficient Shiftmax method based on bit-shift operations, which partially replicates the Softmax effect while better aligning with SNNs. Given an input vector x = [x 1 , x 2 , . . . , x n ] T ∈ I n , we define the approximate Softmax function as follows:
Shiftmax(x) i = 2 xi-γ(x) , γ(x) = log 2 n i=1 2 xi .(13)
2 xi represents the element-wise power operation. γ(x) is defined as the minimal power of 2 that is greater than or equal to the summation of 2 xi . Shiftmax(•) retains the key advantages of Softmax while introducing unique computational efficiency features. Firstly, the 2 xi effect induced by the exponential operation effectively amplifies important weights while suppressing irrelevant ones, ensuring low-entropy activation characteristics. Secondly, it achieves quasi-normalization through a meticulously structured denominator, explicitly constraining the row-stochasticity constraints of Score within a bounded range, formally expressed as:
1 2 < n i=1 Shiftmax(x) i = n i=1 2 xi 2 ⌈log 2 ( n j=1 2 x j )⌉ ≤ 1.(14)
γ(x) constrains the row-stochasticity constraints of attention scores within the interval (0.5, 1].
Although this constraint does not strictly enforce a row-stochasticity constraint, the limited range ensures consistency across attention distributions, enhancing the overall stability of the attention mechanism. Notably, the Shiftmax(•) converts traditional MAC operations into highly efficient bit-shift operations when multiplied by matrix V, which can be decried as:
Score ′ ▷ V = {a ik = j (v jk ≫ n ij ) | s ij = 2 -nij }.(15)
In summary, the Shiftmax(•) preserves the key advantages of the Softmax while transforming intensive exponentiation and normalization into bit-shift operations. It achieves optimal attention score allocation without incurring significant additional computational overhead.
this section cite: ['b17', 'b26']

Section: Overall Architecture
Through ternary Q-K matrix product and energy-efficient Shiftmax operations, we propose the BSA module. When the input x ∈ R T ×n×d , the computational process of BSA are as follows:
Q, K, V = (BN(Conv(X))), Q, K, V ∈ R T ×n×d ,(16)
Score ′ = T SN (Q) × T SN (K), Score ′ ∈ I T ×n×n ,(17)
Score = Shiftmax(Score ′ ), Score ∈ I T ×n×n ,(18)
Attn = T SN (Score ▷ V), Attn ∈ S T ×n×d . (19
)
Where I denotes integer values and S represents spike trains. Since Shiftmax(•) directly yields powers of 2 (2 n ), we can perform bit-shift operations on the membrane potentials of V without employing MAC operations. This approach simultaneously preserves the rich membrane potential information in V while maintaining minimal energy overhead, as bit-shift operations [16] consume merely 1/20 of the energy required for AC operations.
this section cite: ['b15']

Section: Experiments

this section cite: []

Section: Image Classification
In image classification Task, we evaluate the proposed BSA module on ImageNet-1K [7] using three representative state-of-the-art (SOTA) Spiking Transformer architectures: Spikingformer [67], QKformer [68], and Spike-driven Transformer-V3 [52]. Additionally, we perform comprehensive comparative analyses against recent Spiking Transformers [70,49,30,51]. As shown in Table 1, our BSA module consistently improves performance across all three Spiking Transformer architectures. The most substantial gains appear in Spikingformer, where BSA increases accuracy by 1.35% (D=512) and 0.97% (D=768), reaching 76.14% and 76.82% respectively. For Spike-driven V3, BSA yields improvements ranging from 0.36% to 0.74% across different parameter settings (5M, 10M, 19M). Similarly, QKformer's accuracy increases by 0.41% and 0.35% in its two configurations. These results can demonstrate BSA' versatility across different architectures. Additionally, as shown in Fig. 4, BSA exhibits a sparser attention distribution with enhanced focus on critical visual features. Notably, performance improvements in the QKformer and SDT-V3 architecture are less compared to those in the Spikingformer. We attribute this to two factors. First, QKformer and SDT-V3 already demonstrate strong baseline performance on image classification tasks, leaving limited room for further enhancement. Second, both QKformer and SDT-V3 structurally align with the Pyramid Vision Transformer [21,39,54], which substantiates the conclusion that self-attention modules contribute relatively minor performance gains within Metaformer [55] designs. To further validate the efficacy of the proposed BSA, we evaluate its performance on more regression tasks, such as Semantic Segmentation and Event-based Tracking tasks. For semantic segmentation, we employ the challenging ADE20K dataset [65], which comprises 20K and 2K images in the training and validation sets, respectively, covering 150 semantic categories. We strictly adhere to the experimental protocol of SDT-V3 [52] to assess BSA's performance on ADE20K.
As shown in Table .2, BSA achieves significant improvements of 1.8% and 2.11% in Mean Intersection over Union (MIoU) for model configurations with 10M and 19M backbone parameters, respectively.
Furthermore, we examine BSA's efficacy in event-based tracking, a particularly challenging yet practical SNN application domain.
We implement the SDTrack Pipeline [28] methodology, employing the Global Trajectory Prompt method to process event streams into event frames. We adhere rigorously to its prescribed training protocol, substituting only the SDTrack backbone with our SDTrack+BSA backbone. As shown in Table . 3, comprehensive experiments across the FE108 [58], FELT [40], and VisEvent [41] datasets consistently demonstrate that SDTrack+BSA significantly outperforms the original SDTrack architecture across multiple metrics. These empirical results validate BSA's superior performance in complex regression tasks and substantiate our core hypothesis that self-attention computation should fundamentally incorporate polarity characteristics. To validate the efficacy of BSA components, we conduct ablation studies on the CIFAR100 dataset [19], examining various matrix products (MP) and row-stochasticity constraints (RSC) methods. Our experiments utilize the Spikingformer architecture. As shown in the Table.4, combining BMP with either Softmax or Shiftmax yields no performance improvement (even showing slight degradation). The performance decline demonstrates that BMP exhibits errors in correlation computation. These errors are amplified by Softmax, consequently impairing the network's decision-making performance. Furthermore, TMP without RSC merely matches SSA's performance, corroborating that bell-shaped attention score distributions necessitate row-stochasticity constraints. Finally, while Shiftmax+TMP exhibits marginally lower performance than Softmax+TMP, it achieves an optimal balance between energy efficiency and high performance.
this section cite: ['b6', 'b66', 'b67', 'b51', 'b69', 'b48', 'b29', 'b50', 'b20', 'b38', 'b53', 'b54', 'b64', 'b51', 'b27', 'b57', 'b39', 'b40', 'b18']

Section: Conclusion
In this paper, we identify two fundamental limitations of Spiking Self-attention: the binary matrix product's inability to capture anything beyond positive-positive correlations, and the lack of rowstochasticity constraints leading to incomparable attention scores. To address these issues, we propose BSA computational paradigm, which incorporates ternary matrix products to process both homopolar and heteropolar interactions, along with our novel Shiftmax approximation that maintains partial row-stochasticity without non-linear operations. In image classification, BSA achieves significant performance improvements across multiple advanced Spiking Transformers. Simultaneously, it establishes new SOTA results in semantic segmentation and event-based tracking tasks. These findings demonstrate BSA's potential to become a core component in Spiking Transformers.
this section cite: []

Section: References
Ref_id:b0 Title: Spikeprop: backpropagation for networks of spiking neurons Year: (2000)
Ref_id:b1 Title: Spiking neural networks hardware implementations and challenges: A survey Year: (2019)
Ref_id:b2 Title: Hiptrack: Visual tracking with historical prompts Year: (2024)
Ref_id:b3 Title: Binary event-driven spiking transformer Year: (2025)
Ref_id:b4 Title: Crossvit: Cross-attention multiscale vision transformer for image classification Year: (2021)
Ref_id:b5 Title: Loihi: A neuromorphic manycore processor with on-chip learning Year: (2018)
Ref_id:b6 Title: Imagenet: A largescale hierarchical image database Year: (2009)
Ref_id:b7 Title: Tianjic: A unified and scalable chip bridging spike-based and continuous neural computation Year: (2020)
Ref_id:b8 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b9 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b10 Title: You only look at one sequence: Rethinking transformer in vision through object detection Year: (2021)
Ref_id:b11 Title: Spiking neuron models: Single neurons, populations, plasticity Year: (2002)
Ref_id:b12 Title: Beyond self-attention: External attention using two linear layers for visual tasks Year: (2022)
Ref_id:b13 Title: Ternary spike: Learning ternary spikes for spiking neural networks Year: (2024)
Ref_id:b14 Title: Flatten transformer: Vision transformer using focused linear attention Year: (2023)
Ref_id:b15 Title: 1.1 computing's energy problem (and what we can do about it) Year: (2014)
Ref_id:b16 Title: Simple model of spiking neurons Year: (2003)
Ref_id:b17 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b18 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b19 Title: Towards accurate binary spiking neural networks: Learning with adaptive gradient modulation mechanism Year: (2025)
Ref_id:b20 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b21 Title: Neuromorphic computing chip with spatiotemporal elasticity for multi-intelligent-tasking robots Year: (2022)
Ref_id:b22 Title: Networks of spiking neurons: the third generation of neural network models Year: (1997)
Ref_id:b23 Title: Spike timing dependent plasticity finds the start of repeating patterns in continuous spike trains Year: (2008)
Ref_id:b24 Title: Efficient neuromorphic signal processing with loihi 2 Year: (2021)
Ref_id:b25 Title: Towards artificial general intelligence with hybrid tianjic chip architecture Year: (2019)
Ref_id:b26 Title: Reinventing rnns for the transformer era Year: (2023)
Ref_id:b27 Title: Sdtrack: A baseline for event-based tracking via spiking neural networks Year: (2025)
Ref_id:b28 Title: Self-attention with relative position representations Year: (2018)
Ref_id:b29 Title: Spikingresformer: Bridging resnet and vision transformer in spiking neural networks Year: (2024)
Ref_id:b30 Title: Deep directly-trained spiking neural networks for object detection Year: (2023)
Ref_id:b31 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b32 Title: Attention is all you need Year: (2017)
Ref_id:b33 Title: Training-free ann-to-snn conversion for highperformance spiking transformer Year: (2025)
Ref_id:b34 Title: Ternary spike-based neuromorphic signal processing system Year: (2025)
Ref_id:b35 Title: Spiking vision transformer with saccadic attention Year: (2025)
Ref_id:b36 Title: Linformer: Self-attention with linear complexity Year: (2020)
Ref_id:b37 Title: Internimage: Exploring large-scale vision foundation models with deformable convolutions Year: (2023)
Ref_id:b38 Title: Pyramid vision transformer: A versatile backbone for dense prediction without convolutions Year: (2021)
Ref_id:b39 Title: Long-term frame-event visual tracking: Benchmark dataset and baseline Year: (2024)
Ref_id:b40 Title: Visevent: Reliable object tracking via collaboration of frame and event flows Year: (2023)
Ref_id:b41 Title: Autoregressive visual tracking Year: (2023)
Ref_id:b42 Title: Spatio-temporal backpropagation for training high-performance spiking neural networks Year: (2018)
Ref_id:b43 Title: A simple model for behavioral time scale synaptic plasticity (btsp) provides content addressable memory with binary synapses and one-shot learning Year: (2025)
Ref_id:b44 Title: Rethinking spiking self-attention mechanism: Implementing a-xnor similarity calculation in spiking transformers Year: (2025)
Ref_id:b45 Title: Segformer: Simple and efficient design for semantic segmentation with transformers Year: (2021)
Ref_id:b46 Title: Learning spatio-temporal transformer for visual tracking Year: (2021)
Ref_id:b47 Title: Gated linear attention transformers with hardware-efficient training Year: (2023)
Ref_id:b48 Title: Spike-driven transformer v2: Meta spiking neural network architecture inspiring the design of next-generation neuromorphic chips Year: (2024)
Ref_id:b49 Title: Spike-driven transformer v2: Meta spiking neural network architecture inspiring the design of next-generation neuromorphic chips Year: (2024)
Ref_id:b50 Title: Spikedriven transformer Year: (2023)
Ref_id:b51 Title: Scaling spike-driven transformer with efficient spike firing approximation training Year: (2025)
Ref_id:b52 Title: Joint feature learning and relation modeling for tracking: A one-stream framework Year: (2022)
Ref_id:b53 Title: Metaformer is actually what you need for vision Year: (2022)
Ref_id:b54 Title: Metaformer is actually what you need for vision Year: (2022)
Ref_id:b55 Title: Spike-based neuromorphic model for sound source localization Year: (2024)
Ref_id:b56 Title: Spiking transformers for event-based single object tracking Year: (2022)
Ref_id:b57 Title: Object tracking by jointly exploiting frame and event domain Year: (2021)
Ref_id:b58 Title: Spiking neural networks with adaptive membrane time constant for event-based tracking Year: (2025)
Ref_id:b59 Title: Rectified linear postsynaptic potential function for backpropagation in deep spiking neural networks Year: (2021)
Ref_id:b60 Title: Toward energy-efficient spike-based deep reinforcement learning with temporal coding Year: (2025)
Ref_id:b61 Title: Reconstructing clear image for high-speed motion scene with a retina-inspired spike camera Year: (2021)
Ref_id:b62 Title: Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers Year: (2021)
Ref_id:b63 Title: Highspeed image reconstruction through short-term plasticity for spiking cameras Year: (2021)
Ref_id:b64 Title: Scene parsing through ade20k dataset Year: (2017)
Ref_id:b65 Title: Semantic understanding of scenes through the ade20k dataset Year: (2019)
Ref_id:b66 Title: Spikingformer: Spike-driven residual learning for transformer-based spiking neural network Year: (2023)
Ref_id:b67 Title: Qkformer: Hierarchical spiking transformer using qk attention Year: (2024)
Ref_id:b68 Title: Rethinking semantic segmentation: A prototype view Year: (2022)
Ref_id:b69 Title: Spikformer: When spiking neural network meets transformer Year: (2023)
