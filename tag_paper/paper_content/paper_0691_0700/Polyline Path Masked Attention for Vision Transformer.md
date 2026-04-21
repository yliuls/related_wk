Title: Polyline Path Masked Attention for Vision Transformer
Abstract: Global dependency modeling and spatial position modeling are two core issues of the foundational architecture design in current deep learning frameworks. Recently, Vision Transformers (ViTs) have achieved remarkable success in computer vision, leveraging the powerful global dependency modeling capability of the self-attention mechanism. Furthermore, Mamba2 has demonstrated its significant potential in natural language processing tasks by explicitly modeling the spatial adjacency prior through the structured mask. In this paper, we propose Polyline Path Masked Attention (PPMA) that integrates the self-attention mechanism of ViTs with an enhanced structured mask of Mamba2, harnessing the complementary strengths of both architectures. Specifically, we first ameliorate the traditional structured mask of Mamba2 by introducing a 2D polyline path scanning strategy and derive its corresponding structured mask, polyline path mask, which better preserves the adjacency relationships among image tokens. Notably, we conduct a thorough theoretical analysis on the structural characteristics of the proposed polyline path mask and design an efficient algorithm for the computation of the polyline path mask. Next, we embed the polyline path mask into the self-attention mechanism of ViTs, enabling explicit modeling of spatial adjacency prior. Extensive experiments on standard benchmarks, including image classification, object detection, and segmentation, demonstrate that our model outperforms previous state-of-the-art approaches based on both state-space models and Transformers. For example, our proposed PPMA-T/S/B models achieve 48.7%/51.1%/52.3% mIoU on the ADE20K semantic segmentation task, surpassing RMT-T/S/B by 0.7%/1.3%/0.3%, respectively. Code is available at https://github.com/zhongchenzhao/PPMA.

Section: Introduction
The research of foundational models has long been a cornerstone of deep learning. In computer vision, Convolutional Neural Networks (CNNs) [20,19] and Vision Transformers (ViTs) [46,9,31,22] currently represent the dominant architectures. Notably, ViTs have become the most mainstream architecture in large models through the powerful self-attention mechanism, which can capture the non-local self-similarity within global receptive fields. However, the quadratic complexity of the Transformer, when implementing self-attention, severely limits its application in large image processing models. Moreover, as shown in Fig. 1 (b), classic positional encoding methods [46,31,38] in ViTs lack the explicit modeling capability of spatial distance between image tokens, and largely ignore the important spatial adjacency priors in texture, shape, semantics, and so on. This increases the learning pressure and limits its capability for fine-grained image feature extraction.
Compared to CNNs and Transformers, the recently proposed Mamba [11] achieves linear complexity while maintaining global receptive fields, demonstrating strong potential as the next-generation architecture. Specifically, Mamba follows the State Space Models (SSMs) paradigm and employs the selective scan mechanism with the state transition matrix to recursively propagate dependencies among tokens in a sequence. Building on this foundation, Mamba2 [6] further refines the state transition matrix into a lightweight structured mask and introduces a unified theoretical framework, Structured State Space Duality (SSD), to bridge SSMs and attention variants. Under SSD, the core selective scan mechanism of Mamba2 can be reformulated as a form of structured masked attention, i.e., a Linear Attention [23] element-wise multiplied by the structured mask, as illustrated in Fig. 1 (a). Notably, this structured mask explicitly encodes the sequence adjacency of tokens, enabling Mamba2 to match or surpass Transformers across various natural language processing (NLP) tasks. Following its success in NLP, Mamba [11] has been rapidly adapted to various visual domains, including: high-level tasks (classification, object detection, segmentation [58,30,49,47]), low-level tasks (super-resolution, denoising, deraining [13,12,59]), image generation [43], video analysis [26], point cloud analysis [27,51], and remote sensing images [54].
Although Mamba [11] has demonstrated impressive results on certain vision tasks, empirical results on high-level vision tasks demonstrate that even state-of-the-art (SOTA) Mamba-based backbones [30,49,47,15] still underperform SOTA Transformer-based backbones [57,10] with a substantial performance gap. As shown in Fig. 1 (a), this gap mainly stems from two issues: (I) 1D Scanning Issue. Mamba's 1D scanning strategy arranges the tokens of a 2D image into a 1D sequence, which inevitably disrupts the inherent spatial adjacency within 2D images and limits the effectiveness of its recursive selective scanning mechanism. (II) Weak Global Dependency Modeling Issue. The linear attention in Mamba2 omits the non-linear softmax layer, leading to a decrease in the precision and stability of global dependency modeling of images.
In this paper, we present Polyline Path Masked Attention (PPMA), a brand-new method that effectively combines the advantages of ViTs and Mamba2. Specifically, as illustrated in Fig. 1 (c), to address the 1D Scanning Issue when applying current Mamba to 2D images, we propose a novel 2D polyline path scanning strategy and derive an efficient calculation method for its corresponding structured mask, the polyline path mask. Then, we embed the polyline path mask as an explicit positional encoding into the ViT framework. This not only avoids the Weak Global Dependency Modeling Issue of Mamba2, but also alleviates the positional encoding issue of ViTs. As a result, our method fully leverages the powerful global context modeling capability of the self-attention mechanism in ViTs together with the explicit spatial adjacency modeling capability of the polyline path mask inspired by Mamba2, achieving SOTA performance on mainstream high-level vision tasks.
To the best of our knowledge, this is the first work to integrate Mamba2's structured mask mechanism into ViTs. The main contributions of this study are summarized as follows:
• We propose a 2D polyline path scanning strategy for visual Mamba, which better preserves the inherent 2D spatial structure of images compared to existing scanning strategies. Building on this, we further derive a novel structured mask, termed polyline path mask, which is more suitable for 2D images than the traditional structured mask used in Mamba2.
• We conduct a comprehensive theoretical analysis for the proposed 2D polyline path mask. Specifically, we theoretically prove that it can be decomposed into two 1D structured masks with clear physical meanings (i.e., horizontal and vertical scanning masks). More importantly, by leveraging this decomposability, we derive an efficient algorithm to reduce its computational complexity from O(Nfoot_0 ) in the naive calculation to O(N
• Leveraging PPMA, we construct a hybrid Mamba2-Transformer model. Experimental results demonstrate that our model achieves SOTA performance on standard benchmarks for image classification, object detection, and segmentation.
this section cite: ['b19', 'b18', 'b45', 'b8', 'b30', 'b21', 'b45', 'b30', 'b37', 'b10', 'b5', 'b22', 'b10', 'b57', 'b29', 'b48', 'b46', 'b12', 'b11', 'b58', 'b42', 'b25', 'b26', 'b50', 'b53', 'b10', 'b29', 'b48', 'b46', 'b14', 'b56', 'b9']

Section: Related Work
Vision Transformers. ViTs have become foundational in large-scale vision models such as SAM [24] and Sora [1], primarily due to their self-attention mechanism that effectively captures the long-range dependency. Moreover, the spatial structural information provided by positional encodings (e.g., APE [46], RPE [31], and RoPE [38]) is also crucial to ViTs. However, traditional positional encodings fail to explicitly encode spatial adjacency. Recent works, such as RMT [10] and VVT [39], porpose to incorporate RetNet's input-independent temporal decay mask [40] into ViTs for more explicit spatial modeling based on the Manhattan distance. In comparison, Mamba2's input-dependent selective structured mask not only explicitly encodes the relative positional information in the spatial space but also captures the semantic continuity in the feature space.
this section cite: ['b23', 'b0', 'b45', 'b30', 'b37', 'b9', 'b38', 'b39']

Section: Mamba.
As a state space model, Mamba introduces an input-dependent selection mechanism into the state transition matrix A, achieving Transformer-level performance with linear complexity on NLP tasks. Building on this foundation, Mamba2 [6] further simplifies the matrix A to a scalar a, enabling more hardware-efficient parallelizable training without sacrificing performance. Moreover, Mamba2 [6] demonstrates that its formulation is mathematically equivalent to a 1-semiseparable structured masked attention, and develops the State Space Duality (SSD) framework to connect structured SSMs and attention variants. Furthermore, Mamba2 points out that other potential structured masked attentions can also be integrated into the SSD framework.
In this paper, we introduce a novel structured masked attention, termed polyline path masked attention, tailored for vision tasks. Different from the previous 2D selective SSM framework [53] based on Mamba [11], our Mamba2-based polyline path mask is more lightweight and can be plugged seamlessly into various attention variants. Moreover, compared to MambaVision [17] which naively concatenates Mamba's blocks and ViT self-attention layers, our method more effectively harnesses complementary strengths of both architectures.
this section cite: ['b5', 'b5', 'b52', 'b10', 'b16']

Section: Preliminaries
Mamba2's Recurrent Form. Mamba2 [6] initially adopts a recurrent form with linear complexity for sequence modeling. Specifically, Mamba2 employs the selective state space models to map the input sequence x ∈ R N×C to the output sequence y ∈ R N×C , i.e., for i = 1 : N ,
h i = a i h i-1 + B ⊤ i x i , y i = C i h i ,(1)
where x i , y i ∈ R 1×C , h i ∈ R D×C denotes the hidden state, a i ∈ R and B i , C i ∈ R 1×D are inputdependent parameters learned by multilayer perceptron (MLP) layers, the scalar a i serves as a decay factor bounded in [0, 1], N, C and D denote the sequence length, channel number, and hidden state dimension, respectively.
this section cite: ['b5']

Section: Mamba2's Attention Form.
Leveraging the SSD framework in Mamba2 [6], the recurrent form of Mamba2 in Eq. ( 1) can be reformulated as its equivalent dual form, i.e., structured masked attention, by eliminating the hidden state h i via substitution:
y = CB ⊤ ⊙ L 1D x, L 1D ij = a i:j =    a i × • • • × a j+1 i > j 1 i = j 0 i < j ,(2)
where ⊙ denotes the Hadamard (element-wise) product, B, C ∈ R N ×D , and the 1D structured mask L 1D ∈ R N ×N is a 1-semiseparable matrix which can be efficiently calculated with a complexity of O(N 2 ) by the chunkwise algorithm [6]. Mamba2's attention form (Eq. ( 2)) enables more efficient parallelizable training than its recurrent form (Eq. ( 1)). Notably, parameters C and B in Eq. ( 2) are learned analogously to the query Q and key K in ViTs, respectively. Thus, Eq. ( 2) reveals that the selective state transition function in Mamba2 is equivalent to the Hadamard product of a linear attention map CB ⊤ and a 1D structured mask L 1D . Here, the structured mask can be interpreted as a form of relative positional encoding [6]. In this work, we extend the structured masked attention in Mamba2 from 1D sequences to 2D images. Specifically, we extend the 1D structured mask L 1D to the 2D polyline path mask L 2D by introducing a novel 2D scanning strategy, and propose an efficient algorithm for computing and applying this polyline path mask L 2D . The proposed L 2D can be substituted into Eq. ( 2) for replacing L 1D or adopted in ViTs as the explicit positional encoding.
this section cite: ['b5', 'b5', 'b5']

Section: Method
In this section, we introduce the idea of adapting the structured mask of Mamba2 to 2D scanning and integrating it into the self-attention mechanism of ViTs, achieving an explicit positional encoding. Specifically, we 1) introduce the definition of 2D polyline path mask in Sec. 4.1; 2) analyze the theoretical properties of the proposed polyline path mask and introduce an efficient algorithm for the proposed polyline path mask in Sec. 4.2; 3) apply the polyline path mask to standard self-attention and criss-cross attention of ViTs in Sec. 4.3.
this section cite: []

Section: Definition of Polyline Path Mask
As a sequence autoregressive framework, visual Mamba starts from employing a scanning strategy to flatten a 2D image into a 1D sequence of image tokens. This scanning strategy plays an important role in Mamba's performance, since the order of tokens is determined by it. As illustrated in Fig. 2, previous works [58,30,27] have proposed various scanning strategies for visual Mamba. However, these strategies fail to fully preserve the inherent spatial adjacency of 2D tokens. For example, as shown in Fig. 2 (a) and (b), for two tokens B and C which are close in an image, previous scanning strategies [30,27] may cause them to be significantly farther apart in the 1D scanning path.
𝑥 1,1 𝑥 1,2 𝑥 1,3 𝑥 1,4 𝛼 1,2 𝛼 1,3 𝛼 1,4 𝛽 2,1 𝛽 3,1 𝛽 4,1 𝛼 2,2 𝛼 1,3 𝛼 2,4 𝛼 3,2 𝛼 3,3 𝛼 3,4 𝛼 4,2 𝛼 4,3 𝛼 4,4 𝛽 2,2 𝛽 3,2 𝛽 4,2 𝛽 2,3 𝛽 3,3 𝛽 4,3 𝛽 2,4 𝛽 3,4 𝛽 4,4 𝑥 2,1 𝑥 2,2 𝑥 2,3 𝑥 2,4 𝑥 3,1 𝑥 3,2 𝑥 3,3 𝑥 3,4 𝑥 4,1 𝑥 4,2 𝑥 4,3 𝑥 4,4 𝑥 1,1 𝑥 1,2 𝑥 1,3 𝑥 1,4 𝛼 1,2 𝛼 1,3 𝛼 1,4 𝛽 2,1 𝛽 3,1 𝛽 4,1 𝛼 2,2 𝛼 1,3 𝛼 2,4 𝛼 3,2 𝛼 3,3 𝛼 3,4 𝛼 4,2 𝛼 4,3 𝛼 4,4 𝛽 2,2 𝛽 3,2 𝛽 4,2 𝛽 2,3 𝛽 3,3 𝛽 4,3 𝛽 2,4 𝛽 3,4 𝛽 4,4 𝑥 2,2 𝑥 2,3 𝑥 2,4 𝑥 3,2 𝑥 3,3 𝑥 3,4 𝑥 2,1 𝑥 3,1 𝑥 4,1 𝑥 4,2 𝑥 4,3 𝑥 4,4 Polyline Path Scanning. To address this limitation, we design a 2D polyline path scanning strategy. Specifically, for each token pair (x i,j , x k,l ) in the 2D grid, we define their scanning path as the Lshaped polyline connecting them, as shown in Fig. 2 (c). To ensure symmetry in mutual distances, we set two bidirectional polyline paths: vertical-then-horizontal path (V2H solid lines in Fig. 2 (c)) and horizontal-then-vertical path (H2V dotted lines in Fig. 2 (c)), and use their combination as the final scanning path. In this way, the adjacency relationship of 2D tokens can be strictly maintained under the Manhattan distance 2 . Intuitively speaking, tokens close (or far) to each other will be in close (or far) distance on the scanning path, and vice versa. As the example shown in Fig. 2, polyline scanning strategy better preserves the distance between token B and C compared to the other two strategies. An more intuitive example is shown in Fig. 8.
this section cite: ['b57', 'b29', 'b26', 'b29', 'b26']

Section: Definition of Polyline Path Mask.
Based on the proposed polyline path scanning strategy, we introduce the polyline path mask. As an example shown in Fig. 3, we define the horizontal and vertical decay factors of each input token x i,j as α i,j and β i,j , respectively. In this paper, we employ two MLP layers to learn α i,j and β i,j , respectively. 3 Then, the decay weight of V2H polyline path from x k,l to x i,j is defined as L i,j,k,l , which is the product of all decay factors along that path, i.e., L i,j,k,l = α i,j:l β i:k,l , where
α i,j:l =      l n=j+1 αi,n j < l 1 j = l j n=l+1 αi,n j > l , β i:k,l =      k n=i+1 β n,l i < k 1 i = k i n=k+1 β n,l i > k .(3)
For example, as illustrated in Fig. 3, the V2H polyline path's decay weight from token x 4,4 to x 1,1 is L 1,1,4,4 = α 1,2 α 1,3 α 1,4 β 2,4 β 3,4 β 4,4 . Similarly, the decay weight along the H2V polyline path is defined as Li,j,k,l = α k,j:l β i:k,l . Due to the spatial symmetry, it is evident that Li,j,k,l = L k,l,i,j . By combining the V2H and H2V polyline paths, the final decay weight is
L 2D = L+ L.(4)
Note that L, L and L 2D are all 4D tensors of size R H×W×H×W , where H and W are the height and width of the feature map, respectively. The polyline path mask, a 2D matrix L 2D ∈ R HW×HW , can be obtained by unfolding the decay weight tensor, i.e., for all i, j, k, and l,
L 2D (i-1)×W +j,(k-1)×W +l = L 2D i,j,k,l .(5)
For simplicity, we denote the above tensor-to-matrix unfolding operation as L 2D = unfold(L 2D ), and its inverse operation as L 2D = fold(L 2D ) in the following sections. More details can be found in Appendix A.2.
this section cite: []

Section: Efficient Computation Theory of Polyline Path Mask
According to the definition (3), the direct approach to compute the polyline path mask L 2D is to calculate each element individually. However, the large size of the mask and numerous multiplications for each element lead to a high computational cost in both calculating and applying L 2D . To address this issue, we present a decomposition theorem for matrices structured as L 2D . Based on this, we further design an efficient algorithm for performing multiplication on L 2D . For simplicity, we focus our theoretical study on L, which is similar to the case of L 2D . Complete proofs of the theorems are provided in Appendix A.3 and A.5.
this section cite: []

Section: Theorem 1 (Matrix Decomposition).
For any matrix M ∈ R HW×HW and M = fold (M ), if for ∀i, j, k, l, ∃A i ∈ R W×W and B l ∈ R H×H , s.t., M i,j,k,l = A i j,l × B l i,k , then M can be decomposed as:
M = M A × M B = M A ⊙ M B ,(6)
where M A , M B , M A , M B ∈ R HW×HW , which satisfy
M A = unfold(M A ), M B = unfold(M B ), s.t., M A i,:,k,: = A i k = i 0 k ̸ = i , M B :,j,:,l = B l j = l 0 j ̸ = l ,(7)
M A = unfold( MA ), M B = unfold( MB ), s.t., MA i,:,k,: = A i , MB :,j,:,l = B l .(8)
As defined in Eq. ( 3), the polyline path mask L satisfies the conditions in Theorem 1 with [A i ] j,l = α i,j:l and [B l ] i,k =β i:k,l . Thus, based on Theorem 1, the polyline path mask L can be decomposed as L = L H ×L V = LH ⊙ LV . Moreover, for the complexity of computing L, we have: Corollary 1 (Mask Complexity). The complexity of directly computing polyline path mask L with Eq.( 3) and (5) is O(N 5 2 ), which can be reduced to O(N 2 ) by applying Theorem 1, where N = H×W .
For matrices in the form of Eq. ( 7), when performing multiplication operations, we have: Theorem 2 (Efficient Matrix Multiplication). For matrices M A , M B defined in Eq. (7), ∀x ∈ R HW , the following equation holds:
y = M A ×M B ×x ⇔ Z :,l = B l ×X :,l , Y i,: = A i ×Z i,: ,(9)
where y ∈ R HW , X = unvec(x) ∈ R H×W , Y = unvec(y) ∈ R H×W , Z ∈ R H×W , and the operator vec(•) vectorizes a matrix by stacking its columns and unvec(•) is its inverse operator. Algorithm 1: Efficient Masked Attention Computation. Input: decay factors α, β of L, vector x ∈ R HW ; 1: Compute X = unvec(x) ∈ R H×W ; 2: Compute B l ∈ R H×H , where for l = 1 : W, [B l ] i,k =β i:k,l ; 3: Compute Z ∈ R H×W , where Z :,l = B l ×X :,l ; 4: Compute A i ∈ R W×W , where for i = 1 : H, [A i ] j,l =α i,j:l ; 5: Compute Y ∈ R H×W , where Yi,: = A i ×Zi,:; Output: y = vec(Y );
this section cite: []

Section: 𝒚 = 𝑳 𝟐𝑫 ⋅ 𝒙
Based on Theorem 2, we can design Algorithm 1 for computing the matrix multiplication between polyline path mask L and the vector x. Note that the involved matrices A i and B l are symmetric matrices with lower triangular parts being 1-semiseparable, as defined in Mamba2 [6]. This will lead to a substantial reduction in complexity as stated in the following corollary.
this section cite: ['b5']

Section: Corollary 2 (Masked Attention Complexity).
The computational complexity of the matrix multiplication between polyline path mask and vector x, i.e., y = Lx, can be reduced from O(N 2 ) to O(N 3 2 ) by Algorithm 1, and further reduced to O(N ) by applying the chunkwise algorithm of Mamba2 [6] to steps 3 and 5 in Algorithm 1.
Remarks. Intuitively, as illustrated in Fig. 4, Algorithm 1 shows that the 2D polyline path scanning on 2D tokens (i.e., Lx) can be decomposed as the 1D vertical scanning along each column of X (i.e., Z :,l = B l ×X :,l ) followed by the 1D horizontal scanning along each row of Z (i.e., Y i,: = A i ×Z i,: ). This equivalence offers an intuitive understanding of the physical meaning of the decomposed polyline path mask L = L H L V and enables its natural extension to 3D or higher-dimensional tokens, as detailed in Appendix C.2.
this section cite: ['b5']

Section: Polyline Path Masked Attention
The proposed polyline path mask can be seamlessly integrated into various attention variants in a plug-and-play manner. In this section, we integrate it into two softmax-based self-attention layers: vanilla attention [9] and criss-cross attention [22]. Notably, theorems and algorithm given in Sec. 4.2 guarantee that integration of polyline path mask does not substantially increase the computational complexity of the original attention mechanism. More applications, such as the polyline path masked linear attention with a complexity of O(N ), are provided in Appendix A.7.
this section cite: ['b8', 'b21']

Section: Polyline Path Masked Vanilla Attention.
The polyline path mask L 2D is integrated into vanilla attention via a Hadamard product with the attention map, i.e., for query Q, key K, and value V :
PPMVA (x) = softmax(QK ⊤ ) ⊙ L 2D V ,(10)
where Q, K, V ∈ R HW×C . Based on Corollary 1, Eq. ( 10) maintains the complexity of O(N 2 ).
Polyline Path Masked Criss-Cross Attention. The original criss-cross attention [22] employs the sparse attention over tokens located in the same row or column, achieving a complexity of O(N 3 2 ). In this work, we follow RMT [10] to decompose criss-cross attention into the vertical attention over each column followed by the horizontal attention over each row. The polyline path mask L 2D is applied to the decomposed criss-cross attention through the Hadamard product, that is: (11) where horizontal and vertical attention maps S H , S V ∈ R HW×HW satisfy the form in Eq. ( 7) with A i = softmax(Q i,:,: K ⊤ i,:,: ) and B l = softmax(Q :,l,: K ⊤ :,l,: ), and Q, K ∈ R H×W×C are tensor forms of Q, K, respectively [22]. Based on Theorem 1, we can reformulate the left part of Eq. ( 11) as:
PPMCCA (x) = S H ×S V ⊙L 2D V = S H ×S V ⊙L V + S H ×S V ⊙ L V ,
S H ×S V ⊙L V = S H ×S V ⊙ L H ×L V V = ŜH ⊙ ŜV ⊙ LH ⊙ LV V = ŜH ⊙ LH ⊙ ŜV ⊙ LV V = S H ⊙L H × S V ⊙L V ×V . (12
)
Note that matrices ŜH ⊙ LH and ŜV ⊙ LV also satisfy the form in Eq. ( 7). Thus, the computational complexity of Eq. ( 12) can be reduced to O(N 3 2 ) by Algorithm 1. Similar conclusions can also be derived for the right part of Eq. ( 11). Thus, the complexity of Eq.( 11) maintains O(N 3 2 ).
this section cite: ['b21', 'b9', 'b10', 'b21']

Section: Overall Architecture
Based on the proposed Polyline Path Masked Attention, we construct a hybrid Mamba2-Transformer backbone for vision tasks. As illustrated in Fig. 5, our backbone adopts the four-stage hierarchical architecture. Following RMT [10], we employ Polyline Path Masked Criss-Cross Attention in the first three stages, and Polyline Path Masked Vanilla Attention in the final stage. Moreover, we develop our model in three scales: tiny (PPMA-T), small (PPMA-S), and base (PPMA-B).
this section cite: ['b9']

Section: Experiments
To validate the effectiveness of our method, we conduct a series of experiments on mainstream benchmarks for image classification (Sec. 5.1), object detection and instance segmentation (Sec. 5.2), and semantic segmentation (Sec. 5.3). Comparison methods include advanced CNN-based [34,32,42], SSM-based [30,53,49,47], and Transformer-based backbones [31,8,17,16,57,10]. For a fair comparison, we reproduce the experimental results of RMT [10] with the same experimental settings as ours. We also perform comprehensive ablation studies on the structured mask design in Sec. 5.4. More detailed experimental settings and results can be found in Appendix B.
this section cite: ['b33', 'b31', 'b41', 'b29', 'b52', 'b48', 'b46', 'b30', 'b7', 'b16', 'b15', 'b56', 'b9', 'b9']

Section: Image Classification on ImageNet-1K
Settings. We evaluate the classification performance of our method on ImageNet-1K [7]. Following the same training strategy as in [10,44], we train our models from scratch for 300 epochs with the input size of 224×224. We use the adaptive AdamW optimizer with a cosine decay learning rate scheduler (batch size=1024, initial learning rate=0.001, weight decay=0.05).
this section cite: ['b6', 'b9', 'b43']

Section: Results.
The comparison results presented in Table 1 show that our method achieves state-of-the-art (SOTA) performance compared to other advanced models based on various architectures across tiny, small, and base scales. Specifically, PPMA-S achieves 84.2% top-1 accuracy, surpassing 2DMamba-T [53]
this section cite: ['b52']

Section: Object Detection and Instance Segmentation on COCO
Settings. We evaluate our method for object detection and instance segmentation tasks on MSCOCO2017 [28] using the MMDetection library [2]. Following previous work [35], we initialize the backbone with ImageNet-1K pretrained weights and adopt Mask R-CNN [18] as the basic framework. The models are trained for 12 epochs (1× schedule) and 36 epochs with multi-scale inputs (3× schedule) using AdamW optimizer (batch size=16, learning rate=0.0001, weight decay=0.05).
this section cite: ['b27', 'b1', 'b34', 'b17']

Section: Results.
The results presented in Table 2 show that our model outperforms existing methods on most evaluation metrics. Under the same experimental settings, PPMA-T achieves a box mAP of 47.1% and a mask mAP of 42.4%, surpassing the SOTA Transformer-based backbone RMT-T [10] by 0.4% and 0.3% in the 1× schedule, respectively. Moreover, PPMA-B achieves a box mAP of 51.1% and a mask mAP of 45.5%, surpassing the SOTA SSM-based backbone MLLA-S [15] by 1.9% and 1.3% in the 1× schedule, respectively. Furthermore, PPMA-B maintains its superior performance under the 3× multi-scale training schedule.
this section cite: []

Section: Semantic Segmentation on ADE20K
Settings. We evaluate the semantic segmentation performance of our method on ADE20K [56] using the MMSegmentation library [4]. Following the settings in previous works [35], we initialize the backbone with ImageNet-1K pretrained weights and adopt UPerNet [48] as the basic framework. The input size of images is set to 512 × 512 and all models are trained for 160K iterations with AdamW optimizer (batch size=16, learning rate=6×10 -5 , weight decay=0.05).
this section cite: ['b55', 'b3', 'b34', 'b47']

Section: Results.
The semantic segmentation results are summarized in
Table 3. Our method consistently outperforms previous methods under all settings. Compared to SOTA Transformer-based counterparts, PPMA-T/S/B surpass RMT-T/S/B by 0.7%/1.3%/0.3% mIoU in the Single-Scale (SS) setting and 0.3%/2.3%/0.9% mIoU in the Multi-Scale (MS) setting. Compared to SOTA SSM-based methods, PPMA-T/S/B surpass them by at least 3.6%/2.5%/1.6% in SS mIoU, respectively.
this section cite: []

Section: Ablation Study
Polyline Path Mask Design. To verify the effectiveness of the proposed polyline path mask, we conduct an ablation study on ImageNet-1K and ADE20K using PPMA-T as the backbone. Under the same experimental settings, we compare various structured masks embedded into the softmax-based self-attention layers by Hadamard product, including: no mask (baseline), RMT decay mask [10], cross scan mask [30], Hilbert scan mask [27], V2H polyline path mask, and our final 2D polyline path mask. As shown in Fig. 6, our polyline path mask L 2D , compared to the RMT decay mask, can selectively capture the semantic continuity in the image. Compared to the cross scan mask and Hilbert scan mask, the polyline path mask better preserves the spatial relationships between 2D tokens, alleviating the long-range forgetting issue. Experimental results in Table 4 show that the 2D    polyline path mask L 2D boosts the baseline by 0.32% top-1 accuracy on ImageNet-1K and 0.95% SS mIoU on ADE20K, respectively. Visualization results in Fig. 7 further demonstrate that our 2D polyline path mask L 2D effectively suppresses the falsely highlighted areas in the original attention maps. More visualizations and detailed discussions are provided in Fig. 12 and Sec. C.1.
Horizontal and Vertical Decay Factors. In our model, we employ different decay factors (α i,j ̸ = β i,j ) to capture semantic similarity between adjacent tokens along horizontal and vertical directions, respectively. As illustrated in Fig. 7 (b) and (c), the learned decay factors α and β effectively capture semantic continuity in horizontal and vertical directions, respectively. Table 4 shows that replacing different decay factors with a shared decay factor (α i,j = β i,j ) results in a significant performance drop, highlighting the importance of modeling horizontal and vertical decay factors separately.
this section cite: ['b9', 'b29', 'b26']

Section: Conclusion
In this paper, we argue that the key component of Mamba2 model is its structured mask, which explicitly encodes the spatial distance information through the recursive propagation mechanism and captures the semantic continuity in sequences through the selective mechanism. Building on this insight, we propose to extend the structured mask from 1D text sequences to 2D images. To this end, we propose a novel 2D polyline path scanning strategy with its corresponding structured mask tailed for images. To achieve SOTA performance on high-level vision tasks, we integrate the polyline path mask into the powerful self-attention mechanism of ViTs.
this section cite: []

Section: Limitations.
Although the proposed efficient algorithm optimizes the integration complexity, it inevitably incurs additional GPU memory occupation and lower throughput, as shown in Table 4. We plan to alleviate this limitation through further engineering optimizations, such as CUDA-based or Triton-based implementations, in the future work. There are 81 scanning paths. Each scanning path (red polyline) corresponds to a decay weight in the polyline path mask L.
this section cite: []

Section: A.2 Definition of Polyline Path Mask
For each token pair (x i,j , x k,l ) in the 2D grid, the decay weight of the vertical-then-horizontal (V2H) polyline path from x i,j to x k,l is defined as L i,j,k,l , which is the product of all decay factors along that path, i.e., L i,j,k,l = α i,j:l β i:k,l , where
α i,j:l =      l n=j+1 αi,n j < l 1 j = l j n=l+1 αi,n j > l , β i:k,l =      k n=i+1 β n,l i < k 1 i = k i n=k+1 β n,l i > k , (13
)
where α i,j:l and β i:k,l are horizontal and vertical decay factors bounded in the range [0, 1]. For convenience, we unfold the 4D tensor L ∈ R H×W×H×W into a 2D matrix as the polyline path mask L ∈ R HW×HW , i.e., L = unfold(L).
L=               (14)
               (15
)
An intuitive example illustrating the polyline path scanning on a 3×3 grid is presented in Fig. 8. For the 9 tokens in the 2D grid, there are 81 V2H scanning paths connecting them. The V2H scanning path between each token pair is marked by the red polyline, which corresponds to a decay weight in Theorem 3
this section cite: []

Section: Lemma 1
The complexity of 𝑳 𝟏𝑫 𝑿 can be reduced to 𝑂(𝑁) by chunkwise Algorithm
this section cite: []

Section: Theorem 1
Matrix Decomposition 𝑴 = 𝑴 𝑨 × 𝑴 𝑩 = 𝑴 𝑨 ⊙ 𝑴 𝑩
this section cite: []

Section: Conclusion 3.1
The complexity of Polyline Path Masked Standard self-Attention is 𝒪(𝑁 2 ) PPMVA 𝑿 = softmax 𝑸𝑲 ⊤ ⊙ 𝑳 𝟐𝑫 𝑽
this section cite: []

Section: References
Ref_id:b0 Title: Video generation models as world simulators Year: (2024)
Ref_id:b1 Title: MMDetection: Open mmlab detection toolbox and benchmark Year: (2019)
Ref_id:b2 Title: Conditional positional encodings for vision transformers Year: (2023)
Ref_id:b3 Title: Mmsegmentation, an open source semantic segmentation toolbox Year: (2020)
Ref_id:b4 Title: Randaugment: Practical automated data augmentation with a reduced search space Year: (2020)
Ref_id:b5 Title: Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b6 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b7 Title: Cswin transformer: A general vision transformer backbone with cross-shaped windows Year: (2022)
Ref_id:b8 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b9 Title: Rmt: Retentive networks meet vision transformers Year: (2024)
Ref_id:b10 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b11 Title: Attentive state space restoration Year: (2024)
Ref_id:b12 Title: Mambair: A simple baseline for image restoration with state-space model Year: (2024)
Ref_id:b13 Title: Cmt: Convolutional neural networks meet vision transformers Year: (2022)
Ref_id:b14 Title: Demystify mamba in vision: A linear attention perspective Year: (2024)
Ref_id:b15 Title: Neighborhood attention transformer Year: (2023)
Ref_id:b16 Title: Mambavision: A hybrid mamba-transformer vision backbone Year: (2025)
Ref_id:b17 Title: Mask r-cnn Year: (2017)
Ref_id:b18 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b19 Title: Densely connected convolutional networks Year: (2017)
Ref_id:b20 Title: Localmamba: Visual state space model with windowed selective scan Year: (2024)
Ref_id:b21 Title: Ccnet: Criss-cross attention for semantic segmentation Year: (2020)
Ref_id:b22 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b23 Title: Segment anything Year: (2023)
Ref_id:b24 Title: Mpvit: Multi-path vision transformer for dense prediction Year: (2022)
Ref_id:b25 Title: Videomamba: State space model for efficient video understanding Year: (2024)
Ref_id:b26 Title: Pointmamba: A simple state space model for point cloud analysis Year: (2024)
Ref_id:b27 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b28 Title: Scale-aware modulation meet transformer Year: (2023)
Ref_id:b29 Title: Vmamba: Visual state space model Year: (2024)
Ref_id:b30 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b31 Title: A convnet for the 2020s Year: (2022)
Ref_id:b32 Title: Acceleration of stochastic approximation by averaging Year: (2019)
Ref_id:b33 Title: Designing network design spaces Year: (2020)
Ref_id:b34 Title: Transnext: Robust foveal visual perception for vision transformers Year: (2024)
Ref_id:b35 Title: Multi-scale vmamba Year: (2024)
Ref_id:b36 Title: Inception transformer Year: (2022)
Ref_id:b37 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b38 Title: Vicinity vision transformer Year: (2023)
Ref_id:b39 Title: Retentive network: A successor to Transformer for large language models Year: (2023)
Ref_id:b40 Title: Rethinking the inception architecture for computer vision Year: (2016)
Ref_id:b41 Title: Efficientnet: Rethinking model scaling for convolutional neural networks Year: (2019)
Ref_id:b42 Title: Dim: Diffusion mamba for efficient high-resolution image synthesis Year: (2024)
Ref_id:b43 Title: Training data-efficient image transformers & distillation through attention Year: (2021)
Ref_id:b44 Title: Maxvit: Multi-axis vision transformer Year: (2022)
Ref_id:b45 Title: Attention is all you need Year: (2017)
Ref_id:b46 Title: Spatial-mamba: Effective visual state space models via structure-aware state fusion Year: (2025)
Ref_id:b47 Title: Unified perceptual parsing for scene understanding Year: (2018)
Ref_id:b48 Title: Tree topology is all you need in state space model Year: (2024)
Ref_id:b49 Title: Cutmix: Regularization strategy to train strong classifiers with localizable features Year: (2019)
Ref_id:b50 Title: Voxel mamba: Group-free state space models for point cloud based 3d object detection Year: (2024)
Ref_id:b51 Title: mixup: Beyond empirical risk minimization Year: (2018)
Ref_id:b52 Title: dmamba: Efficient state space model for image representation with applications on giga-pixel whole slide image classification Year: (2025)
Ref_id:b53 Title: Rs-mamba for large remote sensing image dense prediction Year: (2024)
Ref_id:b54 Title: Random erasing data augmentation Year: (2020)
Ref_id:b55 Title: Scene parsing through ade20k dataset Year: (2017)
Ref_id:b56 Title: Biformer: Vision transformer with bi-level routing attention Year: (2023)
Ref_id:b57 Title: Vision mamba: Efficient visual representation learning with bidirectional state space model Year: (2024)
Ref_id:b58 Title: Freqmamba: Viewing mamba from a frequency perspective for image deraining Year: (2024)
