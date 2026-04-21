Title: Toward Relative Positional Encoding in Spiking Transformers
Abstract: Spiking neural networks (SNNs) are bio-inspired networks that mimic how neurons in the brain communicate through discrete spikes, which have great potential in various tasks due to their energy efficiency and temporal processing capabilities. SNNs with self-attention mechanisms (spiking Transformers) have recently shown great advancements in various tasks, and inspired by traditional Transformers, several studies have demonstrated that spiking absolute positional encoding can help capture sequential relationships for input data, enhancing the capabilities of spiking Transformers for tasks such as sequential modeling and image classification. However, how to incorporate relative positional information into SNNs remains a challenge. In this paper, we introduce several strategies to approximate relative positional encoding (RPE) in spiking Transformers while preserving the binary nature of spikes. Firstly, we formally prove that encoding relative distances with Gray Code ensures that the binary representations of positional indices maintain a constant Hamming distance whenever their decimal values differ by a power of two, and we propose Gray-PE based on this property. In addition, we propose another RPE method called Log-PE, which combines the logarithmic form of the relative distance matrix directly into the spiking attention map. Furthermore, we extend our RPE methods to a two-dimensional form, making them suitable for processing image patches. We evaluate our RPE methods on various tasks, including time series forecasting, text classification, and patch-based image classification, and the experimental results demonstrate a satisfying performance gain by incorporating our RPE methods across many architectures. Our results provide fresh perspectives on designing spiking Transformers to advance their sequential modeling capability, thereby expanding their applicability across various domains. Our code is available at https://github.com/microsoft/SeqSNN.* The work was conducted during the internship of Changze Lv at Microsoft Research Asia.

Section: Introduction
Spiking Neural Networks (SNNs) [1] are a class of bio-inspired models designed to emulate the communication process of biological neurons, which transmit information through discrete spikes. In contrast to artificial neural networks (ANNs) that operate on continuous values, SNNs process information in the form of spikes occurring at precise moments in time. The temporal characteristics of spikes make SNNs particularly well-suited for tasks involving sequential data or dynamic environments, such as sensory processing [2,3], patch-based image classification [4,5], time-series forecasting [6][7][8], and natural language processing [9][10][11].
In the vanilla Transformer architecture [12], positional encoding serves as a critical mechanism for modeling sequential dependencies in input data. Beyond absolute positional encoding, relative positional encoding (RPE) [13,14] has emerged as an effective approach to represent inter-element distances, enabling models to capture relational patterns within sequences dynamically. Although RPE has demonstrated effectiveness in improving language modeling [13] and visual recognition tasks [15], its integration into SNNs remains underexplored. Existing methodologies for implementing positional encoding in spiking Transformers either suffer from ambiguous spike representations across positions [4,16], or neglect to integrate relative positional relationships entirely [7]. Directly adapting current RPE techniques, such as Attention with Linear Biases (ALiBi) [13] and Rotary Position Embedding (RoPE) [14], to spiking Transformers encounters significant challenges. Specifically, spiking neural architectures exhibit intrinsic difficulty in decoupling relative positional information from their sparse, event-driven representations. This limitation, empirically demonstrated in Section 5.2, underscores the necessity for rethinking RPE integration to align with neuromorphic computing principles, such as temporal sparsity and spike-based communication.
In this paper, we first propose that the Hamming distance [17], which quantifies the number of ones resulting from the XOR operation between two binary strings, serves as an appropriate metric for measuring relative distances when both the query and key matrices are binary. Consequently, we refine the spiking self-attention mechanism [4] by replacing dot-product operations with exclusive-NOR (XNOR) logic operations. Then we present two novel approximation strategies for integrating RPE into spiking Transformers, while strictly preserving the binary activation dynamics inherent to spiking neurons. First, we propose Gray-PE, a method exploiting the properties of Gray Code [18] to binarize relative positional distances. We theoretically prove that encoding relative distances via Gray Code ensures a constant Hamming distance between the binary representations of positional indices whose decimal differences equal 2 n , where n ≥ 0 (See Theorem 1). This property guarantees that any pair of positions separated by a relative distance of 2 n in decimal space exhibits invariant Hamming distances in their Gray Code-encoded representations. Such invariance stabilizes positional relationship modeling for power-of-two intervals, addressing a critical limitation in existing spiking neural architectures. Second, we propose Log-PE, a method adapting insights from ALiBi [13] and Rectified RoPE [19]. Log-PE integrates a non-negative logarithmic transformation of the relative distance map directly into the spiking attention map, inducing a decaying sensitivity to positional relationships akin to windowed attention mechanisms. Moreover, we extend the proposed RPE methods to their two-dimensional form, making them suitable for processing image patches.
To systematically evaluate the efficacy of our proposed RPE methods, we benchmark them across three cross-domain tasks: time series forecasting, text classification, and patch-based image classification. We employ three representative spiking Transformer architectures as backbones: Spikformer [4], the Spike-driven Transformer [5], and QKFormer [20]. Experimental results demonstrate consistent performance gains across all tasks when integrating our RPE approaches, affirming that explicit modeling of relative positional relationships addresses a critical limitation in existing spiking Transformer designs. Furthermore, we conduct experiments on ablation study, long sequence modeling, and sensitivity analysis to validate the inner properties of our proposed RPE method.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b12', 'b14', 'b3', 'b15', 'b6', 'b12', 'b13', 'b16', 'b3', 'b17', 'b12', 'b18', 'b3', 'b4', 'b19']

Section: Related Work
Positional encoding serves as an indispensable mechanism for preserving the order of input elements in sequential modeling tasks. Traditional absolute positional encoding assigns static, predefined embeddings to individual tokens based on their sequential indices. In contrast, relative positional encoding (RPE) dynamically models the pairwise distances between tokens, enabling the selfattention mechanism to prioritize interactions based on their relative proximity. RPE allows the model to generalize across different sequence lengths and better capture relationships between tokens.
Despite the importance of PE in sequence-aware architectures, its application to SNNs is limited.
Existing implementations, such as Spikformer [4] and Spike-driven Transformer [5,16,21], incorporate a combination of convolutional layers, batch normalization, and spiking neuron layers to derive learnable positional encodings. However, we argue that this approach functions more similarly to a spike-element-wise residual connection [3] than to a conventional positional encoding module. A principled PE module should offer unique representations for positions, but the spike-position matrices generated by these methods may lead to identical spike representations for different positions.
CPG-PE, proposed by [7], introduces a spiking absolute positional encoding inspired by central pattern generators [22], generating unique periodic binary spike patterns for each position. However, their approach is based on absolute positional encoding and, thus, does not capture the time-translational invariance property in many sequential modeling problems, which, however, is an important advantage of relative positional encoding methods.
this section cite: ['b3', 'b4', 'b15', 'b20', 'b2', 'b6', 'b21']

Section: Preliminary

this section cite: []

Section: Spiking Neurons
We take the leaky integrate-and-fire (LIF) neuron [1] as our building brick of SNNs, which is governed by the input current I[t], influencing the membrane potential U [t] and the spike output S[t] at each time step t. The dynamic of the LIF neuron is captured by the following system of equations:
U [t] = H[t](1 -S[t]) + U reset S[t], S[t] = Θ(H[t] -U thr ),(1)
H[t] = U [t -1] + 1 τ (I[t] -(U [t -1] -U reset )),(2)
where τ is the membrane time constant. The spike S(t) will be triggered when the membrane potential H(t) exceeds a threshold U thr , right after which U [t] will be reset to U reset .
this section cite: ['b0']

Section: Spiking Self-Attention
Spiking self-attention (SSA) is a spiking version of self-attention [12], which was proposed in Spikformer [4]. The vital design is to utilize discrete spikes to approximate the vanilla self-attention mechanism. It can be written as:
Q, K, V = SN (BN (X • W Q,K,V )) ∈ {0, 1} T ×L×D (3
)
where SN is a spike neuron layer described in Equation 1. The input is denoted as X ∈ {0, 1} T ×L×D , where T is the number of time steps. BN represents batch normalization, and σ is a scaling factor.
The attention map AttnMap is then computed as the dot product between Q and K T :
SSA (Q, K, V) = SN (BN(( Q • K T AttnMap •V * σ) • W)).(4)
As a result, the attention map AttnMap ∈ N T ×L×L 0 , where N 0 denotes the set of non-negative integers. The outputs of the SSA, as well as Q, K, and V, are all spike matrices containing only values of 0 and 1. The parameters W Q , W K , W V , and W are all learnable parameters.
Recent studies, including Spike-Driven Transformer (SDT) [5,16,21], SpikingResFormer [23], and QKFormer [20], have proposed various modifications to the standard SSA mechanism. For our empirical evaluation, we selectively employ architectures demonstrating compatibility with our proposed relative position encoding methods.
this section cite: ['b11', 'b3', 'b4', 'b15', 'b20', 'b22', 'b19']

Section: Relative Positional Encoding
Relative positional encoding (RPE) in Transformers primarily introduces bias terms into the selfattention mechanism that dynamically encode pairwise token distances. A common implementation of RPE, as demonstrated in prior work [15,24], is formalized as follows:
Attention(Q, K, V) = Softmax Q • K T √ d k + R i,j AttnMap •V.(5)
Here, R i,j represents the relative positional bias between the i-th query and the j-th key positions.
Beyond additive bias terms, another widely adopted form of RPE leverages relative positional embeddings directly in the attention computation, where query-position and key-position interactions are parameterized separately. For example, RoPE [14] can be expressed as
Attention(Q, K, V) = Softmax (QR i ) • (KR j ) T √ d k AttnMap •V,(6)
where R i and R j are position-dependent rotation operators applied to the i-th query and j-th key vectors, respectively.
A critical aspect of RPE is its adherence to distance consistency: the magnitude of R i,j is determined exclusively by the relative positional offset |i-j|, ensuring that the model systematically differentiates between proximal and distant tokens. This property enhances the model's capacity to capture longrange dependencies and generalize across variations in sequence length and structure.
this section cite: ['b14', 'b23', 'b13']

Section: Hamming Distance
The Hamming distance [17] between two binary strings of equal length is the number of bit positions at which the corresponding bits are different. Formally, for two binary strings A and B of length m,
d H (A, B) = m i=1 δ(A i , B i ), where δ(A i , B i ) = 1 if A i ̸ = B i , 0 otherwise. (7
)
Hamming distance is suitable for measuring the relative distances when Q and K are spike matrices.
this section cite: ['b16']

Section: Gray Code
Gray Codes [18], also known as reflected binary codes, are binary numbering systems where adjacent values differ by precisely one bit. For a non-negative integer x, the standard binary reflected Gray Code G(x) is defined by the following bitwise operation:
G(x) = x ⊕ (x ≫ 1) ,(8)
where ⊕ denotes the bitwise XOR operation, and ≫ denotes the arithmetic right shift.
Since the preliminary knowledge involved is extensive and loosely connected, we have provided Figure 1 to help readers visually grasp the key concepts of each section.
this section cite: ['b17']

Section: Method

this section cite: []

Section: Design Principles
Relative position encoding (RPE) aims to encode the relative distances between positional indices within a sequence. In many spiking Transformers, such as Spikformer, Spike-Driven Transformer, and QKFormer, both the Q and K matrices are binary. Consequently, their relative distances can be computed using the Hamming distance, which corresponds to the number of ones resulting from the XOR operation between Q and K. To better align with this Hamming distance-based similarity measure, we replace the traditional dot-product spiking self-attention (SSA) mechanism with an XNOR-based SSA. Inspired by RPE strategies in Transformers, we propose two approaches for incorporating relative distance information into spiking attention mechanisms: (1) Gray-PE: Gray-Code-based positional encoding concatenated to Q and K, and (2) Log-PE: logarithmic positional encoding applied directly to the attention map.
this section cite: []

Section: XNOR-Based Spiking Self-Attention
In the original Transformer [12], the attention map is computed via the dot product between the query and key matrices, AttnMap = Q • K T , which effectively captures similarity of Q and K. As mentioned above, in order to capture the relative distances of spiking matrices while effectively measuring the similarity, we design the XNOR-based SSA. Unlike the dot-product operation, XNOR accounts for both spiking state (1) and the resting state (0). Formally, we modify Equation 4 as follows:
AttnMap = D i=0 ¬(Q ⊕ K),(9)
where ¬ denotes the Not operation, ⊕ denotes the XOR operation, and D represents the channel dimension. Note that every token in Q will perform XOR with every token in K, so we sum over the channel dimension D to get AttnMap ∈ N T ×L×L 0 , shown in Figure 2 (a). The scale factor σ in Equation 4 should be set to a smaller value or treated as a learnable parameter, ensuring that the firing rate of SN does not become excessively large. We will empirically demonstrate that this XNOR modification does not negatively impact the performance of the vanilla spiking self-attention.
this section cite: ['b11']

Section: Gray-PE
We propose that the Gray Code can serve as an approximate approach to relative positional encoding for spiking Transformers. This is supported by the following Theorem 1: Theorem 1. (Proof in Appendix A) For two position indices differing by 2 n (n ≥ 0), their Gray Code representations have a consistent Hamming distance. Specifically, ∀ position i, we have:
d H (G(i), G(i + 2 n )) = 1 if n = 0, 2 if n ≥ 1.(10)
As illustrated in Figure 2 (b), the Hamming distance
d H (G(0), G(1)) and d H (G(1), G(2)) both equal 1 because their relative distance is 1, i.e., 2 n , n = 0. Similarly, d H (G(0), G(2)) = d H (G(1), G(3)),and
d H (G(0), G(4)) = d H (G(1), G(5
)), as their relative distances are the power of 2. That said, Gray Code ensures the consistency of relative distance representations for every 2 n (n ≥ 0) relative distance.
For implementation, we concatenate the Gray Code representations of each position index to both the query matrix Q ∈ N T ×L×D 0 and key matrix K ∈ N T ×L×D 0 , leaving the remaining operations unchanged. We use concatenation instead of addition because Q and K are spike matrices, and addition would compromise their binary nature. Formally, the attention map AttnMap will be:
AttnMap = D i=0 ¬([Q ∥ G(l)] ⊕ [K ∥ G(l)]),(11)
where G(•) represents the function that converts integers into their binary Gray Code representations. The vector l denotes an array of position indexes, specifically [0, 1, 2, . . . , L -1], where L is the sequence length of Q and K. ∥ denotes concatenation on the channel dimension D.
Notably, the binary nature of Gray Code (comprising only 0 and 1) aligns intrinsically with the spike-based computation paradigm, avoiding the need for floating-point operations that impose significant implementation overhead on neuromorphic hardware.
this section cite: []

Section: Log-PE
Although Gray-PE can partially capture relative distances, it faces significant challenges when the input sequence is long or when the downstream task is highly sensitive to long-range dependencies. For instance, when L ≥ 10 2 , the distinguishable range of relative distances under Gray-PE becomes constrained by its power-of-two quantization mechanism. To mitigate this, we propose Log-PE that integrates logarithmic positional bias into spiking-based self-attention. Specifically, we simulate Equation 5 and follow ALiBi [13] to directly add a pre-assigned relative position map, denoted as R ij , to the attention map produced by SSA:
AttnMap = D i=0 ¬(Q ⊕ K) + R i,j , where R i,j = [R i,j ] = ⌈log 2 ( L-1 |i-j|+1 )⌉ .(12)
Here, ⌈.⌉ denotes the round-up function, L is the sequence length, and i, j is position indices.
Figure 2 (c) shows an illustration of Log-PE. Since the original AttnMap is a matrix composed of non-negative integers, we aim to ensure accurate relative distance consistency while preserving the effectiveness of spiking self-attention. Theoretically, if we set the R i,j as L-1 |i-j|+1 , we could obtain a complete RPE for the spiking Transformers. However, we choose not to pursue this solution, because for long sequence lengths L, the large values of L-1 |i-j|+1 would catastrophically overshadow the original spiking attention activations (See Appendix B). Therefore, using the logarithmic form R i,j represents a compromise that balances the values between the spiking attention map and complete-RPE, while partially capturing relative position information.
this section cite: ['b12']

Section: Two-Dimensional Form for Image Patches
CNN-based SNN models, such as Spiking VGG [25] and SEW-ResNet [3], do not incorporate the concept of "positional encoding" in their spike representations. Vision Transformer [26] reformulated traditional image classification into a patch-based approach, dividing images into smaller patches. Unlike 1D positional encoding, which only considers the linear sequence of patches, 2D RPE accounts for both the horizontal and vertical positions of the patches in the image grid. This ensures that the model can recognize the relative positions along a single axis and the crucial interactions between patches across both dimensions. We show our 2D form in Figure 2 (d). In our implementation, we assign horizontal and vertical positions with independent dimensions to store the Gray Code. Formally, the attention map AttnMap is:
AttnMap = D i=0 ¬ ([Q ∥ G(h) ∥ G(w)] ⊕ [K ∥ G(h) ∥ G(w)]) .(13)
Here, h is the array of position indices, specifically h = [0, 1, 2, . . . , h -1], where h denotes the maximum patch index along the height axis. Similarly, w is along the width axis. As for the 2D form of Log-PE, we can add R h i,j and R w i,j on AttnMap, replacing the sequence length L in Equation 12with h or w. However, in our pre-experiments, we found that spiking Transformers with Log-2D failed to converge due to the excessive magnitude. Therefore, we abandon the 2D form of Log-PE.
this section cite: ['b24', 'b2', 'b25']

Section: Experiments

this section cite: []

Section: Datasets
To evaluate the RPE capabilities of the compared models, we conduct experiments on two sequential tasks: time-series forecasting and text classification. Following [6], we choose 4 real-world datasets for time-series forecasting: Metr-la [27], Pems-bay [27], Electricity [28], Solar [28]. For text classification, we follow [7] and conduct experiments on six benchmark datasets: Movie Reviews [29], SST-2 [30], SST-5, Subj, ChnSenti, and Waimai. Additionally, to demonstrate the versatility of our RPE method in image processing, we perform patch-based image classification experiments on two static datasets, CIFAR and Tiny-ImageNet, and one neuromorphic dataset, CIFAR10-DVS [2]. The details of these datasets, metrics, and training hyperparameters are provided in Appendix D.
this section cite: ['b5', 'b26', 'b26', 'b27', 'b27', 'b6', 'b28', 'b29', 'b1']

Section: Time-Series Forecasting
We follow the SeqSNN [6] framework to conduct time-series forecasting experiments. Specifically, we take Spikformer [4], Spikingformer [31], Spike-driven Transformer (SDT) V1 [5], and the current visual state-of-the-art (SOTA) model, QKFormer [20], as the backbone architectures. We modify the SSA mechanism as outlined in Section 4.2 to create two variants: Spikformer-XNOR and QKFormer-XNOR. SDT adopts a variant of SSA, which makes it only able to integrate Log-PE but not Gray-PE. We present the performance of the compared SNN models with various positional encoding methods in Table 1. The key findings are as follows:
(1) Directly applying RPE methods to spiking Transformers is ineffective. Specifically, Spikformers that are directly equipped with RoPE or ALiBi exhibit poor performance across all benchmarks. As discussed in Section 1, we argue that this limitation stems from the binary nature of spiking neurons during the computation of Q and K, which makes it difficult to disentangle positional information from sparse spiking activations.
(2) The XNOR modification does not impact the performance of the original SNN models. The average performance of Spikformer with Conv-PE is nearly identical to that of Spikformer-XNOR with Conv-PE. This suggests that our XNOR modification of the SSA does not affect the performance of the original SNN models.
(3) Gray-PE and Log-PE, enable spiking Transformers to achieve the best performance among their variants. CPG-PE is a spiking version of absolute PE designed for SNNs. Spikformer and QKFormer, when equipped with our proposed Gray-PE and Log-PE, consistently outperform all other corresponding variants.
Table 1: Experimental results of time-series forecasting on 4 benchmarks with various prediction lengths 6, 24, 48, 96. "PE" stands for positional encoding. "R" denotes relative PE, while "A" denotes absolute PE. "w/" denotes "with". The best results for each series of spiking Transformers are highlighted in bold font. ↑ (↓) indicates that the higher (lower) the better. Results highlighted with shading are ours. All results are averaged across 3 random seeds.
this section cite: ['b5', 'b3', 'b30', 'b4', 'b19']

Section: Models

this section cite: []

Section: PE
Metric Metr-la (L = 12) Pems-bay (L = 12) Solar (L = 168) Electricity (L = 168) Avg. Spike Type 6 24 48 96 6 24 48 96 6 24 48 96 6 24 48 96 Transformer w/ RoPE ✗ R R 2 ↑ .729 .560 .416 .306 .787 .730 .694 .676 .951 .854 .763 .720 .984 .978 .974 .968 .756 RSE↓ .548 .696 .802 .878 .499 .563 .600 .617 .225 .373 .492 .539 .251 .274 .341 .420 .507 Transformer w/ ALiBi ✗ R R 2 ↑ .725 .558 .409 .293 .782 .727 .690 .677 .924 .845 .741 .665 .984 .980 .976 .968 .747 RSE↓ .556 .700 .814 .885 .507 .569 .606 .615 .281 .393 .527 .602 .250 .271 .339 .422 .521 Transformer w/ Sin-PE ✗ A R 2 ↑ .727 .554 .413 .284 .785 .734 .688 .673 .953 .858 .759 .718 .978 .975 .972 .964 .752 RSE↓ .551 .704 .808 .895 .502 .558 .610 .618 .223 .377 .504 .545 .260 .277 .347 .425 .512 Spikformer w/ Conv-PE (Original) ✓ A R 2 ↑ .713 .527 .399 .267 .773 .697 .686 .667 .929 .828 .744 .674 .959 .955 .955 .954 .733 RSE↓ .565 .725 .818 .903 .514 .594 .606 .621 .272 .426 .519 .586 .373 .371 .379 .382 .541 Spikformer w/ ALIBi R 2 ↑ .665 .483 .380 .104 .760 .644 .348 .064 .080 .080 .080 .080 .710 .710 .710 .710 .413 ✗ R RSE↓ .622 .768 .833 1.02 .529 .709 .870 1.04 1.01 1.01 1.01 1.01 1.03 1.03 1.03 1.03 .909 Spikformer w/ RoPE R 2 ↑ .699 .493 .390 .243 .768 .699 .680 .664 .911 .820 .714 .644 .954 .951 .949 .940 .720 ✗ R RSE↓ .584 .757 .835 .920 .519 .591 .614 .625 .294 .441 .550 .633 .375 .383 .384 .454 .559 Spikformer w/ CPG-PE R 2 ↑ .726 .526 .419 .287 .780 .712 .690 .666 .937 .833 .757 .707 .972 .970 .966 .960 .744 ✓ A RSE↓ .553 .720 .806 .890 .508 .580 .602 .622 .257 .420 .506 .555 .299 .310 .314 .355 .519 Spikformer-XNOR w/ Conv-PE ✓ A R 2 ↑ .718 .531 .405 .269 .771 .693 .690 .665 .928 .829 .740 .669 .960 .957 .955 .953 .733 RSE↓ .559 .721 .813 .910 .518 .599 .613 .628 .273 .421 .527 .595 .365 .371 .376 .384 .542 R 2 ↑ .728 .544 .414 .295 .782 .724 .694 .673 .936 .840 .756 .710 .974 .972 .966 .962 .748 Spikformer-XNOR w/ Gray-PE ✓ R RSE↓ .546 .706 .806 .885 .506 .578 .597 .618 .257 .409 .507 .546 .276 .304 .320 .342 .513 R 2 ↑ .735 .535 .424 .290 .789 .717 .691 .670 .933 .841 .758 .734 .978 .974 .968 .964 .750 Spikformer-XNOR w/ Log-PE ✓ R RSE↓ .543 .719 .799 .876 .496 .575 .601 .620 .265 .408 .504 .525 .272 .300 .314 .340 .509 R 2 ↑ .717 .530 .362 .212 .800 .704 .681 .629 .934 .751 .518 .381 .973 .971 .967 .964 .693 Spikingformer w/o PE (Original) --RSE↓ .560 .720 .842 .936 .483 .587 .611 .659 .258 .500 .694 .788 .299 .305 .325 .340 .557 R 2 ↑ .720 .537 .396 .260 .820 .714 .681 .646 .934 .832 .535 .420 .970 .973 .973 .965 .711 Spikingformer-XNOR w/ Gray-PE ✓ R RSE↓ .558 .712 .819 .907 .459 .578 .610 .643 .257 .421 .663 .768 .305 .293 .294 .338 .539 R 2 ↑ .737 .535 .403 .260 .816 .719 .682 .640 .939 .854 .544 .434 .977 .974 .972 .967 .716 Spikingformer-XNOR w/ Log-PE ✓ R RSE↓ .540 .714 .814 .906 .463 .573 .609 .652 .246 .382 .651 .759 .270 .292 .293 .336 .531 SDT-V1 w/ Conv-PE (Original) R 2 ↑ .689 .517 .409 .253 .769 .700 .647 .630 .917 .819 .723 .655 .956 .952 .949 .950 .721 ✓ A RSE↓ .604 .735 .811 .915 .522 .596 .665 .673 .286 .439 .538 .602 .371 .376 .388 .386 .557 SDT-V1 w/ CPG-PE R 2 ↑ .701 .525 .418 .257 .778 .716 .660 .656 .919 .820 .710 .644 .963 .960 .958 .952 .727 ✓ A RSE↓ .585 .724 .799 .920 .515 .578 .633 .642 .285 .439 .558 .637 .361 .368 .370 .376 .548 R 2 ↑ .714 .531 .415 .265 .784 .709 .672 .654 .921 .820 .730 .674 .972 .968 .963 .957 .734 SDT-V1 w/ Log-PE ✓ R RSE↓ .554 .713 .807 .904 .502 .585 .629 .641 .280 .437 .527 .598 .353 .356 .360 .366 .538 QKFormer w/ Conv-PE (Original) ✓ A R 2 ↑ .717 .513 .376 .246 .767 .706 .681 .654 .920 .748 .512 .416 .970 .967 .963 .958 .695 RSE↓ .561 .735 .832 .917 .521 .586 .609 .635 .289 .515 .716 .784 .306 .319 .355 .367 .565 QKFormer w/ CPG-PE ✓ A R 2 ↑ .740 .554 .419 .276 .783 .714 .702 .660 .922 .754 .702 .604 .977 .969 .968 .963 .732 RSE↓ .536 .704 .803 .896 .503 .578 .589 .633 .285 .520 .581 .645 .266 .312 .315 .332 .531 R 2 ↑ .742 .551 .418 .274 .799 .715 .691 .674 .927 .817 .710 .691 .974 .970 .968 .965 .742 QKFormer-XNOR w/ Gray-PE ✓ R RSE↓ .534 .711 .804 .898 .484 .577 .601 .616 .276 .438 .556 .570 .277 .310 .314 .331 .519 R 2 ↑ .742 .541 .416 .265 .801 .710 .707 .661 .928 .818 .748 .698 .978 .974 .972 .966 .746 QKFormer-XNOR w/ Log-PE ✓ R RSE↓ .535 .715 .805 .903 .482 .581 .585 .629 .274 .437 .515 .564 .264 .285 .296 .328 .514 (4) For long input sequences, Log-PE is more effective than Gray-PE in capturing relative positional information. The input sequence length for Metr-la and Pems-bay is 12, whereas for Solar and Electricity, it is 168. On the long-sequence datasets Solar and Electricity, spiking Transformers equipped with Log-PE consistently outperform those with Gray-PE across nearly all prediction length settings. This result indicates that Log-PE is more effective for processing long input sequences.
this section cite: []

Section: Text Classification
We conduct experiments to assess the efficacy of spiking Transformers with Gray-PE and Log-PE in text classification tasks. By comparing them against alternative PE techniques, we demonstrate their superior ability to model complex linguistic structures and contextual dependencies. Our experimental setup strictly adheres to the methodology outlined in [7], and the results are shown in Table 2. Based on the results in Table 2, it is evident that our proposed Gray-PE and Log-PE significantly outperform the other spiking positional encoding methods across several key benchmarks. Both Gray-PE and Log-PE demonstrate superior accuracy on the English and Chinese datasets, with particularly notable improvements on MR, SST-2, Subj, and ChnSenti. However, the performance of RPE on the Waimai dataset is not as strong as that of CPG-PE. We attribute this to the nature of the dataset, which consists of user reviews often containing informal language, typos, or mixed expressions. This noise can hinder the model's ability to extract meaningful patterns. These results highlight the advantages of our proposed spiking RPE techniques, especially in handling the dependencies and varying word order in text classification tasks. Unlike spiking absolute PE, i.e., CPG-PE, which struggles to adapt to the nuances of language, Gray-PE and Log-PE provide a more flexible and context-sensitive representation, improving the model's ability to classify sentences accurately. In this section, we evaluate ViT-based SNNs, Spikformer, which adopts a patch-splitting processing approach. To enhance compatibility with this framework, we extend Gray-PE into a 2D form and integrate it into the patch-based architecture. The experimental results are summarized in Table 3. We draw conclusions that:
this section cite: ['b6']

Section: Patch-based Image Classification
(1) Gray-PE enhances the performance of Spikformer while maintaining parameter efficiency. Both 1D and 2D variants of Gray-PE consistently improve classification accuracy. Notably, Gray-PE surpasses spiking absolute PE (CPG-PE), indicating its superior ability to model inter-patch dependencies within images, even as an approximation of RPE.
(2) The 2D variant of Gray-PE demonstrates superior performance over its 1D counterpart in processing image patches. Empirical comparisons between Spikformers equipped with Gray-PE 1D and 2D reveal that the two-dimensional form is highly effective. Specifically, Gray-PE 2D achieves an average accuracy improvement of 0.44% over Gray-PE 1D.
Furthermore, we present the image classification performance of the state-of-the-art QKFormer integrated with our proposed RPE methods in Appendix C.
this section cite: []

Section: Capability of Processing Long Sequences
In this section, we assess the effectiveness of our proposed relative positional encoding methods in handling long sequences within spiking Transformers. To this end, we use two text classification datasets characterized by long input samples: AGNEWS [32] and IMDB [33]. Following [34], we fix the sequence max length to 1024 for AGNEWS and 2048 for IMDB. We train the Spikformer model using various positional encoding strategies on these datasets, and present the results in Table 4. As shown in Table 4, although Spikformer models lag behind the finetuned BERT in overall performance, both Log-PE and Gray-PE demonstrate effectiveness when handling long input sequences. Notably, Log-PE yields substantial performance improvements, suggesting its strong suitability for processing long texts. This outcome is expected, as Log-PE is specifically designed to accommodate long-range dependencies. Although traditional SSA benefits from highly optimized matrix multiplication (GEMM) on GPUs, we would like to clarify that our XNOR-based SSA also retains computational efficiency for the following reasons: First, the core of XNOR-based SSA relies on XNOR and bit-count operations, which are natively supported by dig-ital hardware and neuromorphic processors. These are much cheaper than floating-point multiplications and additions in terms of energy and hardware complexity. Secondly, many neuromorphic accelerators (e.g., Loihi [35], TrueNorth [36]) natively support spike-based bitwise logic, making our XNOR mechanism better aligned with the target deployment platform than conventional floatingpoint matrix products. Lastly, while matrix multiplication benefits from BLAS acceleration, XNOR and summation over dimensions are also highly parallelizable, and can be efficiently implemented using tensor intrinsics (e.g., bitwise_xnor, popcount, reduce_sum).
this section cite: ['b31', 'b32', 'b33', 'b34', 'b35']

Section: Discussion on Hardware-Friendliness and Computing Efficiency
We benchmarked both time consumption and GPU memory usage for SNNs in a time-series forecasting task, mainly on the Electricity dataset with 24 of horizon length, as shown in 5. For more analysis on the hardware-friendliness of Log-PE, please refer to the Appendix E.
this section cite: []

Section: Analysis and Ablation
In this section, we analyze the following aspects: (1) The influence of internal properties in Gray-PE, (2) Ablation studies on XNOR and Log-PE (shown in Appendix B). Consider that: If the number of bits used for encoding relative positions in Gray Code is b, then the total number of unique encodings possible is 2 b . We set the maximum sequence length is L, so relative distances range from 0 to L-1. According to the pigeonhole principle, if L -1 > 2 b , there will be at least two distances that are represented identically. This issue can be mitigated by increasing b to cover the range of relative distances up to L -1. From Figure 3 (a), we observe that for long-sequence datasets, such as Solar and Electricity (Length = 168), the number of bits should be at least 7 to avoid Gray-PE missing relative positional information. However, for shorter datasets like Metr-la (Length = 12) and ChnSenti (Length = 32), 5 bits are sufficient.
this section cite: []

Section: Conclusion
In this work, we have designed several RPE methods for spike Transformers. Our approach preserves the spiking nature of SNNs while effectively representing relative positions. Experimental evaluations on time series forecasting, text classification, and image classification demonstrate significant performance improvements. These empirical results, together with theoretical analysis of the proposed RPE methods, highlight the potential to enhance the versatility and applicability of SNNs across various domains. Future work and limitations are discussed in Appendix F.
this section cite: []

Section: References
Ref_id:b0 Title: Networks of spiking neurons: the third generation of neural network models Year: (1997)
Ref_id:b1 Title: Cifar10-dvs: An eventstream dataset for object classification Year: (2017)
Ref_id:b2 Title: Deep residual learning in spiking neural networks Year: (2021)
Ref_id:b3 Title: Spikformer: When spiking neural network meets transformer Year: (2023)
Ref_id:b4 Title: Spikedriven transformer Year: (2023)
Ref_id:b5 Title: Efficient and effective time-series forecasting with spiking neural networks Year: ()
Ref_id:b6 Title: Advancing spiking neural networks for sequential modeling with central pattern generators Year: (2024)
Ref_id:b7 Title: Ts-lif: A temporal segment spiking neuron network for time series forecasting Year: (2025)
Ref_id:b8 Title: SpikeGPT: Generative pre-trained language model with spiking neural networks Year: (2024)
Ref_id:b9 Title: Spiking convolutional neural networks for text classification Year: (2023)
Ref_id:b10 Title: Spikelm: Towards general spike-driven language modeling via elastic bi-spiking mechanisms Year: (2024)
Ref_id:b11 Title: Attention is all you need Year: (2017)
Ref_id:b12 Title: Train short, test long: Attention with linear biases enables input length extrapolation Year: (2017)
Ref_id:b13 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b14 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b15 Title: Spike-driven transformer v2: Meta spiking neural network architecture inspiring the design of next-generation neuromorphic chips Year: (2024)
Ref_id:b16 Title: Coding and information theory Year: (1986)
Ref_id:b17 Title: Pulse code communication. United States Patent Number 2632058 Year: (1953)
Ref_id:b18 Title:  Year: (2023)
Ref_id:b19 Title: QKFormer: Hierarchical Spiking Transformer using Q-K Attention Year: (2024)
Ref_id:b20 Title: Scaling spike-driven transformer with efficient spike firing approximation training Year: (2025)
Ref_id:b21 Title: Central pattern generators and the control of rhythmic movements Year: (2001)
Ref_id:b22 Title: Spikingresformer: bridging resnet and vision transformer in spiking neural networks Year: (2024)
Ref_id:b23 Title: Posmlp-video: spatial and temporal relative position encoding for efficient video recognition Year: (2024)
Ref_id:b24 Title: Going deeper in spiking neural networks: Vgg and residual architectures Year: (2019)
Ref_id:b25 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b26 Title: Diffusion convolutional recurrent neural network: Data-driven traffic forecasting Year: (2018)
Ref_id:b27 Title: Modeling long-and short-term temporal patterns with deep neural networks Year: (2018)
Ref_id:b28 Title: Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales Year: (2005)
Ref_id:b29 Title: Recursive deep models for semantic compositionality over a sentiment treebank Year: (2013)
Ref_id:b30 Title: Spikingformer: Spike-driven residual learning for transformer-based spiking neural network Year: (2023)
Ref_id:b31 Title: Character-level convolutional networks for text classification Year: (2015)
Ref_id:b32 Title: Learning word vectors for sentiment analysis Year: (2011)
Ref_id:b33 Title: Searching for an effective defender: Benchmarking defense against adversarial word substitution Year: (2021)
Ref_id:b34 Title: Loihi: A neuromorphic manycore processor with on-chip learning Year: (2018)
Ref_id:b35 Title: Truenorth: Design and tool flow of a 65 mw 1 million neuron programmable neurosynaptic chip Year: (2015)
Ref_id:b36 Title: Dpt: Deformable patch-based transformer for visual recognition Year: (2021)
Ref_id:b37 Title: Patch-based discriminative learning for remote sensing scene classification Year: (2022)
Ref_id:b38 Title: A patch information supplement transformer for person re-identification Year: (1997)
Ref_id:b39 Title: Adaptive decomposition and shared weight volumetric transformer blocks for efficient patch-free 3d medical image segmentation Year: (2023)
Ref_id:b40 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b41 Title: Spikebert: A language spikformer learned from bert with knowledge distillation Year: (2023)
Ref_id:b42 Title: Decoupled weight decay regularization Year: (2018)
Ref_id:b43 Title: Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection Year: (2024)
Ref_id:b44 Title: Spiking nerf: Representing the real-world geometry by a discontinuous representation Year: (2024)
