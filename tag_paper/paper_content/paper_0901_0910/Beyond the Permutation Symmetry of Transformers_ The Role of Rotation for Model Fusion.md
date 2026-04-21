Title: Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion
Abstract: Symmetry in the parameter space of deep neural networks (DNNs) has proven beneficial for various deep learning applications. A well-known example is the permutation symmetry in Multi-Layer Perceptrons (MLPs), where permuting the rows of weight matrices in one layer and applying the inverse permutation to adjacent layers yields a functionally equivalent model. While permutation symmetry fully characterizes the equivalence set for MLPs, its discrete nature limits its utility for transformers. In this paper, we introduce rotation symmetry, a novel form of parameter space symmetry for transformers that generalizes permutation symmetry by rotating parameter matrices in self-attention layers. Unlike permutation symmetry, rotation symmetry operates in a continuous domain, thereby significantly expanding the equivalence set for transformers. Based on this property, we propose a theoretically optimal parameter matching algorithm as a plug-and-play module to enhance model fusion. We evaluate our approach using pre-trained transformers across diverse natural language and vision tasks. Experimental results demonstrate that our rotation symmetrybased matching algorithm substantially improves model fusion, highlighting the potential of parameter space symmetry to facilitate model fusion. Our code is available on https://github.   com/zhengzaiyi/RotationSymmetry.

Section: Introduction
Parameter space symmetry is an intriguing property of neural networks that has garnered increasing attention in recent years (Du et al., 2018;Armenta & Jodoin, 2021;Kunin et al., Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). 2021; Simsek et al., 2021;Entezari et al., 2022;Grigsby et al., 2023;Lim et al., 2024b;Ziyin et al., 2024;2025). One of the most studied forms of parameter space symmetry is permutation symmetry (Ainsworth et al., 2023;Entezari et al., 2022). For instance, in a two-layer MLP, permuting the rows of the weight matrix in the first layer and applying the corresponding inverse permutation to the second layer results in a functionally equivalent model, i.e., the outputs of the original and permuted models remain identical for any given input (Ainsworth et al., 2023). All functionally equivalent models corresponding to weight permutations form an equivalence set, which provides theoretical insights into neural network optimization, such as the linear mode connectivity of loss landscapes (Entezari et al., 2022;Zhou et al., 2023c;Ferbach et al., 2024). In addition, permutation symmetry has also proven helpful in advancing neural network applications, such as model fusion (Singh & Jaggi, 2020;Ainsworth et al., 2023) and optimization (Zhao et al., 2024;Zamir et al., 2025).
Although parameter space symmetry has been extensively studied in classical neural network architectures, such as MLPs and CNNs, the understanding of its application in transformers (Vaswani et al., 2017) remains limited. Transformers have seen rapid advancements in recent years, achieving remarkable success in a wide range of applications (Yun et al., 2019;Lewis et al., 2020;Raffel et al., 2020;Clark et al., 2020;Zhou et al., 2021;He et al., 2024;Zhu et al., 2024;Zheng et al., 2024;Zhang et al., 2025;Wei & Liu, 2025). The transformer architecture is built upon two primary submodules: feedforward networks and selfattention layers. The feedforward network, which is structurally similar to MLPs, naturally inherits the permutation symmetry that has been extensively studied in the existing literature. Self-attention layers, on the other hand, involve a unique attention mechanism powered by matrix products of queries, keys, and values, which introduce additional potentials for symmetry beyond permutations. Permutation symmetries limit the equivalence set of neural networks to discrete operations, which aligns well with MLPs due to their element-wise activations (e.g., ReLU (Glorot et al., 2011)). In contrast, the continuous nature of the matrix operations in self-attention layers necessitates more flexible operations to fully characterize their equivalence set.
this section cite: ['b18', 'b1', 'b42', 'b68', 'b25', 'b97', 'b0', 'b0', 'b20', 'b69', 'b0', 'b89', 'b85', 'b75', 'b83', 'b43', 'b63', 'b12', 'b93', 'b30', 'b96', 'b90', 'b86', 'b78', 'b22']

Section: Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion
In this paper, we introduce rotation symmetry, a novel form of parameter space symmetry for self-attention layers in transformers. Specifically, we analyze the query and key matrices jointly and demonstrate that applying a rotation to the query matrix, followed by the corresponding inverse rotation to the key matrix, preserves the query-key product. Additionally, we find that the same rotation rule can be applied to the value and output matrices. Our findings provide novel insights into the functional invariance of the attention mechanism and extend the permutation symmetry (Singh & Jaggi, 2020;Wang et al., 2020;Tatro et al., 2020;Entezari et al., 2022;Ainsworth et al., 2023;Imfeld et al., 2024) from discrete spaces to continuous spaces, which significantly extends the scope of parameter space symmetries for transformers.
To further demonstrate the benefits of rotation symmetry for transformers, we explore its utility in model fusion. The goal of model fusion is to merge multiple well-trained end models in the parameter space to produce a single merged model with improved overall utility. Model fusion is widely adopted across various settings, such as hyperparameter tuning (where end models are trained on the same benchmark) and multi-task learning (where end models are trained with different tasks) (Jin et al., 2023). Unlike ensemble learning (Dietterich, 2000;Lakshminarayanan et al., 2017;Sagi & Rokach, 2018;Dong et al., 2020), model fusion can work in a data-agnostic manner, making it suitable for privacysensitive scenarios such as federated learning (Yurochkin et al., 2019;Wang et al., 2020).
The existing literature has demonstrated that the performance of model fusion is closely tied to the distance between the end models (Wortsman et al., 2022). Inspired by this finding, we propose a parameter matching algorithm that selects functionally equivalent end models from the equivalence class determined by rotation symmetry. This approach ensures that the selected representative models are closer in parameter space, resulting in a smaller inner distance. To achieve this, we formulate the problem of parameter matching as an optimization problem with orthogonal constraints. Leveraging the continuous nature of rotation symmetry, we propose a closed-form solution to this problem. Our parameter matching algorithm is highly efficient, easy to implement, and can be seamlessly incorporated as a plug-and-play module for model fusion. To evaluate its effectiveness, we conduct extensive experiments with pre-trained transformers on real-world NLP and vision tasks. The experimental results demonstrate that incorporating rotation symmetry into parameter matching improves model fusion effectively and efficiently. Furthermore, additional experiments reveal that even matching a subset of parameters can lead to notable performance improvements, highlighting the practical utility of our approach. Our contributions are threefold:
• We introduce a novel rotation symmetry for the attention mechanism in transformers, extending the concept of symmetry to a continuous space.
• Building on rotation symmetry, we propose a theoretically optimal parameter matching algorithm that improves the effectiveness of model fusion in transformers.
• Through extensive experiments, we validate the efficacy of our proposed parameter matching algorithm, demonstrating its potential to advance model fusion through parameter space symmetry.
this section cite: ['b69', 'b77', 'b71', 'b0', 'b34', 'b36', 'b15', 'b66', 'b16', 'b84', 'b77', 'b79']

Section: Preliminaries
To facilitate further discussion, we begin by defining the notations. Let A[i, :], A[:, j], and A[i, j] denote the i-th row vector, the j-th column vector, and the (i, j)-th element of the matrix A, respectively. Additionally, let 1 denote an allones vector. We now illustrate the concept of permutation symmetry in neural networks, using an MLP as an example. Consider an L-layer MLP model defined as:
f W (X) = Z (L) , Z (l) = σ(Z (l-1) (W (l) ) ⊤ + b (l) ), (1
)
where l) is a row vector and can be seen as broadcast over all rows in Equation ( 1)), W = {W (l) , b (l) } l=1,...,L collects all learnable parameters, and σ(•) stands for a nonlinear activation function, such as ReLU (Nair & Hinton, 2010). A loss function L(f W (X), Y ) is used to measure the distance between the model prediction and the ground truth label Y .
Z (0) = X is the input feature matrix, b (l) is the bias vector (b (
To analyze the permutation symmetry in the MLP model, let P ∈ P be a permutation matrix, where P [i, j] ∈ {0, 1} and P [i, :]1 = P [:, j] ⊤ 1 = 1 for any i, j. All permutation matrices are orthogonal (Strang, 1976), satisfying P ⊤ = P -1 . Consequently, for layer l and l + 1, we have: l+1) .
Z (l+1) = σ σ(Z (l-1) (W (l) ) ⊤ + b (l) )(W (l+1) ) ⊤ + b (l+1) = σ σ(Z (l-1) (P ⊤ W (l) ) ⊤ + b (l) P )(W (l+1) P ) ⊤ + b(
(
The third equal sign holds because the element-wise activation function σ is decoupled from column permutation (i.e., being multiplied by P ). Based on Equation (2), it follows that for layers l and l + 1, the mappings l+1) for any input Z (l-1) . For each pair of adjacent layers, a similar mapping exists independently based on a specific permutation matrix, denoted as P (l) for layers l and l + 1. Consequently, for the entire L-layer MLP, there exists a mapping that preserves the model's predictions for any input X: l) . Equivalently, if we define W ′ = {(P
W (l) → P ⊤ W (l) , b (l) → b (l) P , W (l+1) → W (l+1) P preserves the output Z(
W (l) → (P (l) ) ⊤ W (l) P (l-1) , b (l) → b (l) P (
(l) ) ⊤ W (l) P (l-1) , b (l) P (l) } l=1,...,L for any P (l) ∈ P Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion
(l = 1, . . . , L -1) and P (0) = P (L) = I, then we have f W ′ (X) = f W (X) for any X. This phenomenon is referred to as the permutation symmetry of the parameter space (Godfrey et al., 2022;Hecht-Nielsen, 1990;Navon et al., 2023;Rossi et al., 2023;Simsek et al., 2021). Leveraging permutation symmetry allows us to identify an equivalence class of functionally equivalent model parameters, which is known as permutation invariance (Ainsworth et al., 2023;Entezari et al., 2022;Lubana et al., 2023). We denote the equivalence relation induced by permutation invariance as π, where, for example, W ′ = π(W ) represents the equivalence between the original parameters W and the permuted parameters W ′ .
this section cite: ['b56', 'b70', 'b23', 'b31', 'b57', 'b65', 'b68', 'b0', 'b51']

Section: Parameter Space Symmetry of Transformers
Transformers (Vaswani et al., 2017) have revolutionized deep learning with their ability to handle sequential data effectively, particularly in natural language processing (NLP) and other fields (Brown et al., 2020;Devlin et al., 2019;Liu et al., 2021;Radford et al., 2021). Their success is primarily driven by two key components 1 : feedforward networks and self-attention layers (Vaswani et al., 2017). To better understand the parameter space symmetry of transformers, we examine these two core modules individually in the following sections.
this section cite: ['b75', 'b8', 'b14', 'b50', 'b62', 'b75']

Section: Permutation Symmetry of Feedforward Networks
We first look at the feedforward networks. The feedforward network adopted in the transformer blocks is a two-layer MLP model which can be written as
F F N (X) = LN (ReLU (XW ⊤ i + bi)W ⊤ o + bo + X), (3
)
where LN denotes the Layer Normalization operator (Ba et al., 2016). Different from (Vaswani et al., 2017), we include the residual connection (He et al., 2016) and layer normalization modules into the formula of the feedforward network (and the self-attention layer mentioned later). According to Equation ( 2) and the analysis in Preliminaries, the feedforward networks have the permutation symmetry property and the equivalence class determined by permutation invariance is defined as
W i → P ⊤ W i , b i → b i P , W o → W o P , b o → b o , (4
)
where P ∈ P is a permutation matrix. It is worth noting that the permutations of different feedforward networks in a transformer are independent due to the scalable modular design. Consequently, we are able to flexibly compute the permutation invariance equivalence class of each F F N module in a transformer model. 1 We follow the architecture in the original transformer paper (Vaswani et al., 2017).
this section cite: ['b3', 'b75', 'b28', 'b75']

Section: Rotation Symmetry of Self-attention Layers
We then focus on the self-attention layers. In this paper, we introduce the rotation symmetryof self-attention layers. For MLPs, we have to switch the order of multiplying P and passing the activation function σ (the third equal sign in Equation ( 2)), requiring the matrix P to be a permutation matrix. In contrast, the self-attention layer does not contain an element-wise activation function, enabling a wider range of the matrix P in self-attention layers. The self-attention layer can be written as
AT T N (X) = LN Cat H h=1 X h QKV W ⊤ O + bO + X , X h QKV = Sf tmx X h Q (X h K ) ⊤ / d k • X h V ,
where Cat stands for the operator concatenating the outputs of multi-head attention, Sf tmx(•) denotes the softmax operator, H denotes the number of multi-heads, and the subscripts Q, K, V , and O denote Query, Key, Value, and Output, respectively.
We first transform the query and key matrices as
X h Q (X h K ) ⊤ = (X(W h Q ) ⊤ + b h Q )(X(W h K ) ⊤ + b h K ) ⊤ = (X(W h Q ) ⊤ + b h Q )RR ⊤ (X(W h K ) ⊤ + b h K ) ⊤ = (X(R ⊤ W h Q ) ⊤ + b h Q R)(X(R ⊤ W h K ) ⊤ + b h K R) ⊤ ,
where R is a rotation matrix, i.e., RR ⊤ = I. It is worth noting that each multi-head corresponds to a specific rotation matrix R.
Let W O = [W 1 O W 2 O • • • W H O ]
and we then rewrite the concatenating operation (with the product by W O ) as
H h=1 S(• • • )(X(W h V ) ⊤ + b h V )(W h O ) ⊤ .
Similar to the Q-K case, we can transform the value and output matrices as (X(
W h V ) ⊤ + b h V )(W h O ) ⊤ = (X(W h V ) ⊤ + b h V )RR ⊤ (W h O ) ⊤ = (X(R ⊤ W h V ) ⊤ +b h V R)(W h O R) ⊤ for each multi-head h,
where R is a rotation matrix. Finally, we derive an equivalence class of the parameters in a selfattention layer determined by rotation invariance as
W h Q → (R h qk ) ⊤ W h Q , b h Q → b h Q R h qk , W h K → (R h qk ) ⊤ W h K , b h K → b h K R h qk , W h V → (R h vo ) ⊤ W h V , b h V → b h V R h vo , W h O → W h O R h vo , bO → bO,(5)
where R h qk and R h vo are rotation matrices for h = 1, . . . , H. The progress from permutation to rotation extends the symmetry of transformers to a continuous space and enhances our understanding of the parameter space symmetry of attention mechanism. The denseness of the symmetry allows for the choice of better invariant models to analyze transformers' loss landscapes. For better clarity, we provide an intuitive illustration of the rotation symmetry in Figure 1.
this section cite: []

Section: Symmetry for Model Fusion
In this section, we explore the benefit of the rotation symmetry of transformers in model fusion (Li et al., 2023;Matena Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion Q Linear K Linear V Linear Attention Output Linear (𝑾𝑾) 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆 𝑸𝑸𝑹𝑹 � 𝑹𝑹 𝑻𝑻 𝑲𝑲 𝑻𝑻 𝑑𝑑 𝑽𝑽𝑹𝑹 � 𝑹𝑹 𝑻𝑻 𝑾𝑾 𝑹𝑹 𝑹𝑹 𝑻𝑻 𝑹𝑹 𝑹𝑹 𝑻𝑻 𝑹𝑹𝑹𝑹 𝑻𝑻 = 𝑰𝑰 𝑹𝑹: Rotate 𝑹𝑹 𝑻𝑻 : Rotate back Equivalent function for rotation matrix 𝑹𝑹 & Raffel, 2022;Wortsman et al., 2022;Yadav et al., 2023;Jin et al., 2023;Daheim et al., 2024;Yang et al., 2024). Model fusion is proposed to merge multiple given end models trained in different settings (e.g., upon different datasets and hyperparameter settings) in the parameter space to improve model utility and robustness. Compared with ensemble learning (Dietterich, 2000;Sagi & Rokach, 2018;Lakshminarayanan et al., 2017), model fusion has a lower inference-stage complexity without requiring access to the training data. Most existing methods of model fusion conduct a weighted averaging of different end models, e.g., direct averaging (Wortsman et al., 2022), Fisher-weighted averaging (Matena & Raffel, 2022), and regression-mean averaging (Jin et al., 2023). We next show the potential of exploiting the permutation and rotation symmetry as a plugand-play module to improve the model fusion techniques.
this section cite: ['b45', 'b53', 'b79', 'b80', 'b36', 'b13', 'b82', 'b15', 'b66', 'b79', 'b53', 'b36']

Section: Background and Motivation
Let W 1 , . . . , W k denote k different end models (after training or pre-training) with the same architecture. The goal of model fusion is to merge the given k end models in the parameter space and obtain a single model. If the given models are trained over different datasets, we can expect the merged model to have better utility and out-of-distribution robustness (Jin et al., 2023). The theoretical results in previous literature (Wortsman et al., 2022) have shown that strong convexity and closer end models can boost the utility of direct model fusion. Consequently, the primary advantage of permutation (and rotation) symmetry applying to model fusion is that we can substitute the end models with corresponding carefully chosen equivalent models (in the equivalence class determined by permutation and rotation invariance) to make the selected end models more concentrated, i.e., closer to each other. This step is usually called parameter matching (Singh & Jaggi, 2020;Wang et al., 2020;Ainsworth et al., 2023), aka parameter or neuron alignment. Parameter matching, which aims to reduce the distance between end models, can naturally yield closer end models and improve model fusion performance. On the other hand, different end models can lie in the basins of different local optimums regarding the highly non-convex nature of transformers. Previous studies have verified that parameter matching can merge different end models toward a single low-loss basin, i.e., the loss value along the linear interpolation between matched models shows an approxi- mately flat or convex curve (Entezari et al., 2022;Ainsworth et al., 2023). This property is called linear mode connectivity and is regarded as a weak form of convexity (Ainsworth et al., 2023). As a result, parameter matching can also help improve the convexity of the objective in the area adjacent to the end models. We showcase an intuitive example of using parameter space symmetry to improve model fusion in Figure 2 to illustrate the intuition behind parameter matching. We can observe that the merged model after parameter matching (AB') has a lower loss value, i.e., better utility than the naive merged model (AB).
this section cite: ['b36', 'b79', 'b69', 'b77', 'b0', 'b0', 'b0']

Section: Parameter Matching
Next, our primary goal is to develop a practical parameter matching algorithm to minimize the distance between different end models. We begin by merging two models W 1 and W 2 .
this section cite: []

Section: Matching two FFNs.
Let {W i k , b i k , W o k , b o k } k=1,2
denote the model parameters in the two feedforward networks to be merged where k denote the index of the networks. We have already derived the equivalence relation of parameters in the feedforward networks determined by permutation invariance as Equation ( 4). Based on Equation (4), we can formulate parameter matching as an optimization problem as follows.
argmin P1,P2∈S ∥P ⊤ 1 W i1 -P ⊤ 2 W i2 ∥ 2 F + ∥b i1 P 1 -b i2 P 2 ∥ 2 F + ∥W o1 P 1 -W o2 P 2 ∥ 2 F .(6)
As shown in Equation ( 6), the goal is to find the functionally equivalent models from the equivalence classes π(W 1 ) and π(W 2 ) determined by P 1 and P 2 , which has the smallest ℓ-2 distance in the parameter space. Following the method (Ainsworth et al., 2023), we reformulate the optimization problem shown in Equation ( 6) as a linear assignment problem:
argmax P ∈S P , W i1 W ⊤ i2 + b ⊤ i1 b i2 + W ⊤ o1 W o2 F . (7)
Linear assignment problems such as Equation ( 7) have been well-studied in previous literature (Martello & Toth, 1987;Burkard & Cela, 1999) and can be solved precisely by Hungarian Algorithm (Martello & Toth, 1987). After calculating the value of P , we substitute P = P 1 P -1 2 back to Equation ( 4) and let P 2 = I (for simplicity), then we obtain the matched parameter as
W i1 → P ⊤ W i1 , b i1 → b i1 P , W o1 → W o1 P , b o1 → b o1 . Parameters of the sec- ond model {W i2 , b i2 , W o2 , b o2 } remain
unchanged, which makes the second model an anchor model for matching.
this section cite: ['b0', 'b52', 'b9', 'b52']

Section: Matching two ATTNs.
Let the parameters in the two self-attention layers to be merged be
{W i Q k , b i Q k , W i K k , b i K k , W i V k , b i V k , W i O k , b i O k } k=1,2;i=1.
..H where k denote the index of the layers. The equivalence relation of these parameters in the self-attention layers in terms of rotation invariance is shown in Equation ( 5). Similar to matching FFNs, our goal is to match the parameters of one (source) attention layer to the other (target) attention layer where the matched parameters are supposed to be closest to the target parameters while being functionally equivalent when feeding with any input data, i.e., in the equivalence class determined by rotation invariance. The goal can be formulated as an optimization problem.
min R h qk ,R h vo ∈R (W h Q 1 ) ⊤ (W h Q 2 ) ⊤ b h Q 1 b h Q 2 R h qk1 -R h qk2 2 F + (W h K 1 ) ⊤ (W h K 2 ) ⊤ b h K 1 b h K 2 R h qk1 -R h qk2 2 F + (W h V 1 ) ⊤ (W h V 2 ) ⊤ b h V 1 b h V 2 R h vo1 -R h vo2 2 F + W h O 1 W h O 2 0 0 R h vo1 -R h vo2 2 F ,(8)
where R denotes the set of rotation matrices. To solve this problem, we divide Equation ( 8) into two separate optimization problems (the first line in terms of R h qk1 , R h qk2 as the first objective and the second line in terms of R h vo1 , R h vo2 as the second objective). Considering the similar formulation of these two optimization problems, in this paper, we propose the following theorem to solve both problems.
Theorem 4.1. The following optimization problem has a closed-form solution.
min R1,R2∈R W ⊤ Q1 W ⊤ Q2 b Q1 b Q2 R 1 -R 2 2 F + W ⊤ K1 W ⊤ K2 b K1 b K2 R 1 -R 2 2 F . (9
)
The solution is given by
R 1 = U V ⊤ , R 2 = I,(10)
Algorithm 1 Matching two self-attention layers.
Input: Model parameters of the attention layers
{W h Q k ,b h Q k ,W h K k ,b h K k ,W h V k ,b h V k ,W h O k ,b h O k } k=1,2;h=1,...,H . Output:
The optimally matched parameters from source parameters (k = 1) to anchor parameters (k = 2, maintain unchanged).
1: QK solution:
R h qk1 = U h qk (V h qk ) ⊤ , R h qk2 = I, where U h qk Σ h qk (V h qk ) ⊤ = W h Q1 (W h Q2 ) ⊤ + W h K1 (W h K2 ) ⊤ + (b h Q1 ) ⊤ b h Q2 + (b h K1 ) ⊤ b h K2 . 2: QK matching: W h Q1 → (R h qk1 ) ⊤ W h Q1 , W h K1 → (R h qk1 ) ⊤ W h K1 , b h Q1 → b h Q1 R h qk1 , b h K1 → b h K1 R h qk1 . 3: VO solution: R h vo1 = U h vo (V h vo ) ⊤ , R h vo2 = I, where U h vo Σ h vo (V h vo ) ⊤ = W h V1 (W h V2 ) ⊤ + (W h O1 ) ⊤ W h O2 + (b h V1 ) ⊤ b h V2 . 4: VO matching: W h V1 → (R h vo1 ) ⊤ W h V1 , W h O1 → W h O1 R h vo1 , b h V1 → b h V1 R h vo1 .
where I is the identity matrix and
U ΣV ⊤ = W Q1 W ⊤ Q2 + W K1 W ⊤ K2 + b ⊤ Q1 b Q2 + b ⊤ K1 b K2 is the result of eigende- composition.
We leave a detailed proof of Theorem 4.1 in Appendix A. According to Theorem 4.1, we can obtain the algorithm to match two self-attention layers shown in Algorithm 1. Algorithm 1 can be seen as an adaptation of the Kabsch algorithm (Kabsch, 1976;Umeyama, 1991). Without loss of generality, we let the parameters in the second (k = 2) self-attention layer be the anchor and conduct rotation for the first layer (k = 1). The denseness of rotation symmetry helps reduce the distance between the end models after parameter matching as Algorithm 1 shows.
Rescaling Matching. We find that the rescaling symmetry (Neyshabur et al., 2015;Meng et al., 2019;Godfrey et al., 2022;Kalogeropoulos et al., 2024) can be integrated with our proposed rotation symmetry in the parameter matching algorithm. For instance, for the (simplified) Q-K product W Q W K , we can find a rescaling operation that preserves the functionality of the Q-K product aW Q • 1 a W K where a ̸ = 0 is a real number. By adding a scalar variable to each parameter matrix, we can formulate the objective of rescaling symmetry as follows (taking the Q-K product as an example).
min a aW ′ Q1 -W ′ Q2 2 + ab ′ Q1 -b ′ Q2 2 + W ′ K1 /a -W ′ K2 2 + b ′ K1 /a -b ′ K2 2 , (11
)
where a is the rescaling variable and ′ denotes the parameters after rotation symmetry-based matching. Here, we still set model 2 as the anchor model and conduct the rescaling operation on model 1 to align with model 2. We can easily solve the optimality condition of Equation ( 11) as
W ′ Q1 2 + b ′ Q1 2 a 4 -W ′ K1 2 -b ′ K1 2 -W ′ Q1 , W ′ Q2 + b ′ Q1 , b ′ Q2 a 3 + W ′ K1 , W ′ K2 + b ′ K1 , b ′ K2 a = 0. (12
) The roots of Equation ( 12) can be derived using numerical methods as the value of a. We then conduct the rescaling operation to model 1 as
W ′ Q1 → aW ′ Q1 , b ′ Q1 → ab ′ Q1 , W ′ K1 → 1 a W ′ K1 , b ′ K1 → 1 a b ′ K1 .
It is worth noting that the rescaling symmetry-based matching is conducted after the rotation symmetry-based matching. Using the rescaling operation, we extend the rotation matrices to orthogonal matrices with larger norms. Nevertheless, the end models are close to each other in practical scenarios so the value of a is usually close to 1.
Complexity. We next provide a brief analysis of the complexity of our proposed parameter matching algorithm. We let the hidden dimension of the target transformer be d, the layer of the target transformer be L, and the number of attention heads be H. For the parameter matching of feedforward networks, the linear assignment problem can be solved in O(d 3 ) by the Hungarian algorithm (Kuhn, 1955;Martello & Toth, 1987). Additionally, to solve the optimization problem in Equation ( 8), Algorithm 1 requires the complexity of O(d 3 ) for eigendecomposition. Hence, the complexity of our proposed full parameter matching algorithm of a transformer is O(d 3 LH), similar to the complexity of the feedforward. It is worth noting that the complexity can be further reduced in two ways. The first way is to match a subset of layers instead of all. In this way, the complexity of our proposed parameter matching algorithm is O(d 3 L s H) where L s denotes the number of selected layers. The second way is to match each unit module (a single feedforward network or a single attention layer) in parallel. The decoupling of matching different modules makes it easy to implement multiprocessing. Consequently, the overall complexity becomes to O(d 3 LH/p) where p is the number of processes in parallel.
this section cite: ['b37', 'b74', 'b59', 'b55', 'b23', 'b38', 'b41', 'b52']

Section: Experiments
In this section, we conduct experiments to evaluate the effectiveness of our proposed parameter matching algorithm based on rotation symmetry. Specifically, we aim to answer the following research questions: RQ1: Can our proposed parameter matching algorithm enhance the performance of model fusion for transformer-based models? RQ2: How does rotation symmetry contribute to parameter matching for self-attention layers? RQ3: How does parameter matching influence the loss landscape between independently trained models? RQ4: As a plugin module, does our algorithm introduce significant additional computational overhead? RQ5: Is matching all transformer layers equally important? Can we improve efficiency without compromising utility by matching only a subset of layers?
this section cite: []

Section: Experimental Settings
Platform. Our implementation is based on Python 3.10 and Pytorch 1.13. All fine-tuning, parameter matching, and model merging processes are conducted on a cluster equipped with Nvidia A100 80GB GPUs.
this section cite: []

Section: Models.
To evaluate the effectiveness of our approach, we use two widely adopted transformer models: RoBERTa (Liu et al., 2019) and DeBERTa (He et al., 2021). We obtain pretrained models of RoBERTa (roberta-base with 12 attention layers) and DeBERTa-Large (microsoft/deberta-v3-large with 24 attention layers) from the Hugging Face library. For vision transformers (ViTs) (Dosovitskiy et al., 2021), we directly use the pretrained models following (Imfeld et al., 2024). By selecting these three type of models, we assess the effectiveness of our method across different model scales and downstream tasks.
this section cite: ['b48', 'b29', 'b17', 'b34']

Section: Finetuning and Matching.
In the experiments, each pretrained language model is fine-tuned for 20 epochs on each dataset individually. We set the learning rate at 1e-5, the batch size at 16, and the warmup ratio at 0.06 for each model. After fine-tuning, we perform parameter matching and model merging, considering both in-domain (pairwise
Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion 7 8 9 10 11 12 Simple match (Ours) w/o rescaling w/o FFN w/o ATTN 10.19 10.22 11.15 10.45 10.0 12.5 15.0 17.5 20.0 Fisher match (Ours) w/o rescaling w/o FFN w/o ATTN 18.58 18.61 20.21 12.21 12 14 16 Regmean match (Ours) w/o rescaling w/o FFN w/o ATTN 15.35 15.31 12.94 11.89 28 30 32 34 OT-ACTS-EMB match (Ours) w/o rescaling w/o FFN w/o ATTN 32.53 32.5 28.66 32.08 60.0 60.5 61.0 61.5 OT-ACTS match (Ours) w/o rescaling w/o FFN w/o ATTN 61.25 61.23 60.07 61.15 56.0 56.5 57.0 57.5 OT-WTS match (Ours) w/o rescaling w/o FFN w/o ATTN 57.17 57.16 56.16 57.11 Figure 3. Ablation Study of ViT merging over the image classification task. "ATTN" is short for "attention". We compare our matching algorithm with its three variants (w/o ATTN/FFN/rescaling) and the original performance (w/o match) on all six merging baselines. 600 610 620 630 640 650 660 670 Distance w/o match Git Re-Basin OT-Fusion w/o ATTN, rescaling w/o ATTN w/o rescaling match (Ours) 651.57 625.73 626.16 640.01 639.96 613.68 613.63 fine-tuned models) and out-of-domain (grouped models) experiments. Notably, we match only the parameters within the attention layers, while the classifier module is directly copied from the model fine-tuned on the corresponding downstream task. For additional details on datasets and baseline methods, please refer to Appendix C.
this section cite: []

Section: Performance of Model Fusion
To answer RQ1, we compare the performance of three model fusion baselines with and without our proposed parameter matching algorithm.
In-Domain Settings. We evaluate model fusion performance on Emotion Classification and Named Entity Recognition (NER) tasks, where our approach (+match) is integrated as a plugin module for three model fusion baselines. For each task, we fine-tune the language models on each in-domain dataset (5 for Emotion, 6 for NER) respectively and merge them pairwise. The first two columns in Table 1 present the average macro F1-score (for Emotion) and micro F1-score (for NER) of the merged models following (Jin et al., 2023).
this section cite: ['b36']

Section: Out-of-Domain Settings.
To assess the generalization ability, we merge models trained on all in-domain NER datasets and evaluate their performance on CoNLL datasets, which serve as out-of-domain (OOD) test sets. The third column in Table 1 reports the OOD performance of the merged models.
ViT Settings. We employ three OT (optimal transportation)-based methods (OT-ACTS-EMD, OT-ACTS, and OT-WTS) from (Imfeld et al., 2024) as additional baselines. Two pretrained models are merged, and we evaluate their performance on the CIFAR-10 ( Krizhevsky et al., 2009) dataset. Table 2 presents the classification accuracy of the merged models. To improve the flexibility of parameter matching, we match each layer separately before conducting model fusion. The merged model achieving the highest validation performance is selected as the final test model.
From the results in Table 1 and Table 2, we make the following observations: (1) Our parameter matching algorithm consistently improves the performance of different model fusion methods. The reason is that parameter matching helps align distant models, bringing them closer in parameter space. As a result, the merged model is more likely to approach an overall minimum. (2) Compared to RoBERTa-base, our method brings larger improvement for DeBERTa-large, suggesting that larger models benefit more from parameter matching. We explain this as larger models have more parameters that can be aligned, and in highdimensional spaces, models tend to be more spread out. Subsequently, parameter matching helps bridge this gap more effectively. (3) Among all the fusion methods, simple fusion shows the largest improvement after parameter matching. This is likely because simple fusion directly averages model weights, so reducing the distance between model parameters brings a clear benefit. In contrast, advanced fusion methods work by aligning outputs of the end models based on input data, which is equivalent to weighted averaging. Unlike direct averaging, these methods benefit from parameter matching in a more implicit way.
this section cite: ['b34', 'b40']

Section: Ablation Study
To answer RQ2, we conduct ablation studies to evaluate the contributions of rotation symmetry in matching different model components. These experiments are performed on the same pair of pretrained ViT models used in the previous
Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion Loss Barriers aging. Unlike direct averaging, these methods benefit parameter matching in a more implicit way. Ablation Study nswer RQ2, we conduct ablation studies to evaluate the ributions of rotation symmetry in matching different el components. These experiments are performed on ame pair of pretrained ViT models used in the previous ion. We design three ablation settings: (1) w/o ATTN: ches only the FFN modules, excluding self-attention ules. (2) w/o FFN: Matches only the self-attention ules, excluding FFN modules. (3) w/o scaling: Diss the rescaling symmetry in our algorithm. Ablation lts in Figure 3 demonstrate the following observations: n Fisher and RegMean, attention matching contributes e significantly to the fusion process. (2) In contrast, usion benefits more from FFN matching. These distincs demonstrate the nuanced interactions between the fumechanisms and corresponding component alignment egies. (3) w/o scaling results in performance degradaacross most settings, validating the effectiveness of aling symmetry integrated with rotation symmetry.
urther quantify the influence of rotation symmetry, we sure the Euclidean distance between the matched model the anchor model in the parameter space. Instead of agating distances at the layer or module level, we analyze ntire model holistically to provide a more comprehenassessment of alignment. We also compare our method nst two external baselines: Git-Rebasin (Ainsworth ., 2023) and OT-Fusion (Imfeld et al., 2024). Followthe original paper, OT-Fusion is applied only to FFN ules. The results in Figure 4 indicate that our rotation metry-based parameter matching algorithm consistently ces the distance between end models more effectively permutation symmetry-based methods. This improvet can be attributed to the continuous nature of rotation metry, which allows for smoother and more precise ment, especially in self-attention layers. Additionally, rporating rescaling symmetry further refines parameter ment, leading to a greater distance reduction. Furthere, the integration of rescaling symmetry leads to addial distance reductions of end models, further enhancing meter alignment.
this section cite: ['b34']

Section: Loss Landscape Study
ddress RQ3, we examine the loss landscape between arameters of the two pretrained ViT models introduced
Matching 1.59 1.71 3.48 Simple Merging 0.13 0.09 0.22 Fisher Merging 197.47 69.57 83.67 Regmean Merging 137.67 36.44 71.02 along the following linear path between the two original sets of model parameters, denoted as ω 1 and ω 2 :
ω(ε) = ε • ω 1 + (1 → ε) • ω 2 , ε ↑ [0, 1]. (13
)
Afterwards, we separately apply attention matching (w/o FFN), FFN matching (w/o ATTN), and complete matching (w/ match) on the first ViT's parameters ω 1 . Consequently, we construct the linear path and calculate loss values likewise for each matching setting. The empirical results in Figure 6 give rise to the following observations: (1) There exist positive loss barriers (the maximum increase in loss along the linear path starting from ε = 0 or 1) in all model parameter pairs due to the strong non-convexity in ViTs (Park & Kim, 2022).
(2) The parameter pair after complete matching exhibits the lowest loss barrier compared to other matching settings, suggesting that rotation symmetry facilitates smoother connectivity between model pairs. 8 Figure 5. Loss landscapes and barriers between the two pretrained ViT models under four distinct matching settings. "LB" is short for "Loss Barrier". To further quantify the influence of rotation symmetry, we measure the Euclidean distance between the matched model and the anchor model in the parameter space. Instead of aggregating distances at the layer or module level, we analyze the entire model holistically to provide a more comprehensive assessment of alignment. We also compare our method against two external baselines: Git Re-Basin (Ainsworth et al., 2023) and OT-Fusion (Imfeld et al., 2024). Following the original paper, OT-Fusion is applied only to FFN modules. The results in Figure 4 indicate that our rotation symmetry-based parameter matching algorithm consistently reduces the distance between end models more effectively than permutation symmetry-based methods. This improvement can be attributed to the continuous nature of rotation symmetry, which allows for smoother and more precise alignment, especially in self-attention layers. Furthermore, the integration of rescaling symmetry leads to additional distance reductions of end models, further enhancing parameter alignment.
this section cite: ['b60', 'b0', 'b34']

Section: Loss Landscape Study
To address RQ3, we examine the loss landscape between the two pretrained ViT models introduced in Section 5. The results in Figure 5 give rise to the following observations: (1) All model pairs exhibit positive loss barriers, i.e., the range of loss value along the interpolation path, highlighting the strong inherent non-convexity of ViTs (Park & Kim, 2022). (2) Complete matching yields the lowest loss barrier, indicating that aligning parameters through rotation symmetry leads to smoother connectivity between model pairs. (3) While FFN-only matching (w/o ATTN) has been proven effective for MLPs and CNNs (Ainsworth et al., 2023), its effectiveness remains limited for Transformers. Their intricate loss landscape brings additional challenges for parameter matching.
this section cite: ['b60', 'b0']

Section: Complexity Study
To answer RQ4, we evaluate the computational overhead introduced by our parameter matching algorithm. Specifically, we measure the average runtime for fine-tuning (per dataset), matching (per model pair), and merging methods (per model pair) from the main experiments, as shown in Table 3. For ViTs, we only use pre-trained models following (Imfeld et al., 2024) without fine-tuning. The results show that our parameter matching module incurs an overhead of less than 5% compared to Fisher / RegMean merging across all models, with negligible impact on overall complexity.
this section cite: ['b34']

Section: Matching a Subset of Layers
To answer RQ5, we investigate the effect of matching different subsets of layers. We fine-tune DeBERTa models on CoLA and STS-B from the GLUE benchmark and evaluate the performance of matched models under different subsets
Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion of layers. Specifically, we cross-validate the importance of each attention layer in matching through two settings:
Single-Layer Matching. In this setting, we match only a single attention layer while leaving all other layers unchanged. The evaluation loss corresponding to different matched layer indices is shown in Figure 6(a).
Tail-Layers Matching. Here, we match a certain number of trailing attention layers (e.g., when matching three attention layers, we align only the last three layers). The evaluation loss for different numbers of matched layers is presented in Figure 6(b).
The experimental results in Figure 6 demonstrate that matching head layers yields greater improvements in model utility compared to tail layers. Notably, the loss value drops sharply when the first 5 layers are matched. Based on this observation, we can further improve efficiency by dropping tail layers without significantly compromising utility.
this section cite: []

Section: Related Work
Parameter Space Symmetry. Parameter space symmetry refers to a set of models with different parameter values but functionally equivalent. This concept has been extensively studied in the context of deep neural networks, as it plays a crucial role in understanding model behavior and training dynamics. Examples of parameter space symmetries include rescaling symmetry (Neyshabur et al., 2015;Badrinarayanan et al., 2015;Du et al., 2018;Meng et al., 2019), scaling symmetry (Kunin et al., 2021), and translation symmetry (Kunin et al., 2021). These symmetries have been identified in conventional deep neural networks to provide deeper insights into training dynamics and to accelerate the optimization process (Zhao et al., 2022;2023;2024). Another important type of parameter space symmetry is permutation symmetry, which has been shown to closely relate to the manifold of global minima and critical points (Fukumizu & Amari, 2000;Brea et al., 2019;Simsek et al., 2021;Benton et al., 2021;Entezari et al., 2022;Ainsworth et al., 2023). The permutation symmetry can also be used to align (match) the outputs or model parameters of different end models with the same architecture (Singh & Jaggi, 2020;Wang et al., 2020;Ainsworth et al., 2023;Peña et al., 2023;Imfeld et al., 2024;Navon et al., 2024). Complementary to parameter space symmetry, recent advances in neural functionals and metanetworks (Navon et al., 2023;Zhou et al., 2023b;a;Lim et al., 2024a;Tran et al., 2024) explore permutation-equivariant functionals that operate directly on model weights across diverse architectures. Some concurrent works (Liu, 2024;Tran et al., 2024) investigate similar forms of rotation symmetry in neural networks. Liu (2024) shows that the mirror symmetry leads to low-rankness. Meanwhile, Tran et al. (2024) leverages the rotation symmetry to construct transformer-based neural functional networks. In comparison, our study focuses on the role of rotation symmetry in model fusion and proposes a theoretically optimal parameter matching approach based on the properties of rotation symmetries.
this section cite: ['b59', 'b4', 'b18', 'b55', 'b42', 'b42', 'b87', 'b49', 'b21', 'b7', 'b68', 'b5', 'b0', 'b69', 'b77', 'b0', 'b61', 'b34', 'b58', 'b57', 'b73', 'b49', 'b73', 'b49', 'b73']

Section: Model Fusion.
The goal of model fusion (Li et al., 2023) is to merge multiple available end models (with the same architecture) to obtain a stronger model. The scenarios of model fusion can be flexible. When training on the same dataset, model fusion can be used to improve the model utility or generalization by merging models trained with different configurations or in different stages (Izmailov et al., 2018;Gupta et al., 2020;Cha et al., 2021;Wortsman et al., 2022;Rame et al., 2022;Arpit et al., 2022;Huang et al., 2024;Yadav et al., 2024;Hammoud et al., 2024). As a representative method in this setting, ModelSoup (Wortsman et al., 2022) greedily averages the models fine-tuned with different hyperparameter configurations to improve the utility and robustness of the model. In addition, when training on different datasets or tasks, model fusion can be used to improve out-of-domain generalization or multitasking of the model (Matena & Raffel, 2022;Choshen et al., 2022;Li et al., 2022;Jin et al., 2023;Zhou et al., 2024), especially for language models. A state-of-the-art merging algorithm, RegMean (Jin et al., 2023), successfully merges language models fine-tuned over different tasks and improves the model's out-of-distribution generalization. Moreover, model fusion plays a pivotal role in federated learning (Konečnỳ et al., 2016;McMahan et al., 2017;Wang et al., 2020) when the local updates are collected to make a global update. FedAvg (McMahan et al., 2017) is a classical merging algorithm that directly computes the average of the local models as the updated global model. Recent studies propose to incorporate the permutation symmetry to align the neurons of different end models (Wang et al., 2020;Singh & Jaggi, 2020;Ainsworth et al., 2023). However, these methods fail to achieve a desirable performance when tackling transformer-based models (Jin et al., 2023).
this section cite: ['b45', 'b35', 'b26', 'b10', 'b79', 'b64', 'b2', 'b33', 'b81', 'b27', 'b53', 'b11', 'b44', 'b36', 'b94', 'b36', 'b39', 'b54', 'b77', 'b54', 'b77', 'b69', 'b0', 'b36']

Section: Conclusion
In this paper, we introduced rotation symmetry as a novel type of parameter space symmetry for transformers, extending the concept of permutation symmetry to continuous spaces. Building on this foundation, we proposed a theoretically optimal parameter matching algorithm to enhance the fusion of transformer models in a plug-and-play manner. To validate our approach, we conducted extensive experiments on real-world NLP and vision benchmarks. The results demonstrated that incorporating rotation symmetry effectively and efficiently facilitates transformer model fusion, showcasing our method's practical utility.
Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion
this section cite: []

Section: References
Ref_id:b0 Title: Git re-basin: Merging models modulo permutation symmetries Year: (2023)
Ref_id:b1 Title: The representation theory of neural networks Year: (2021)
Ref_id:b2 Title: Ensemble of averages: Improving model selection and boosting performance in domain generalization Year: (2022)
Ref_id:b3 Title: Layer normalization Year: (2016)
Ref_id:b4 Title: Symmetryinvariant optimization in deep networks Year: (2015)
Ref_id:b5 Title: Loss surface simplexes for mode connecting volumes and fast ensembling Year: (2021)
Ref_id:b6 Title: An analysis of annotated corpora for emotion classification in text Year: (2018-08)
Ref_id:b7 Title: Weightspace symmetry in deep networks gives rise to permuta-tion saddles, connected by equal-loss valleys across the loss landscape Year: (2019)
Ref_id:b8 Title: Language models are few-shot learners Year: (2020)
Ref_id:b9 Title: Linear assignment problems and extensions Year: (1999)
Ref_id:b10 Title: Domain generalization by seeking flat minima Year: (2021)
Ref_id:b11 Title: Fusing finetuned models for better pretraining Year: (2022)
Ref_id:b12 Title: Electra: Pre-training text encoders as discriminators rather than generators Year: (2020)
Ref_id:b13 Title: Model merging by uncertainty-based gradient matching Year: (2024)
Ref_id:b14 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b15 Title: Ensemble methods in machine learning Year: (2000)
Ref_id:b16 Title:  Year: (2020)
Ref_id:b17 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b18 Title: Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced Year: (2018)
Ref_id:b19 Title: Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion Entezari Year: (2022)
Ref_id:b20 Title: Proving linear mode connectivity of neural networks via optimal transport Year: (2024)
Ref_id:b21 Title: Local minima and plateaus in hierarchical structures of multilayer perceptrons Year: (2000)
Ref_id:b22 Title: Deep sparse rectifier neural networks Year: (2011)
Ref_id:b23 Title: On the symmetries of deep learning models and their internal representations Year: (2022)
Ref_id:b24 Title: Procrustes problems Year: (2004)
Ref_id:b25 Title: Hidden symmetries of relu networks Year: (2023)
Ref_id:b26 Title: Stochastic weight averaging in parallel: Large-batch training that generalizes well Year: (2020)
Ref_id:b27 Title: Model merging and safety alignment: One bad model spoils the bunch Year: (2024)
Ref_id:b28 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b29 Title: Decodingenhanced bert with disentangled attention Year: (2021)
Ref_id:b30 Title: Explaining graph neural networks with large language models: A counterfactual perspective on molecule graphs Year: (2024)
Ref_id:b31 Title: On the algebraic structure of feedforward network weight spaces Year: (1990)
Ref_id:b32 Title: OntoNotes: The 90% solution Year: (2006-06)
Ref_id:b33 Title: Emr-merging: Tuning-free high-performance model merging Year: (2024)
Ref_id:b34 Title: Transformer fusion with optimal transport Year: (2024)
Ref_id:b35 Title: Averaging weights leads to wider optima and better generalization Year: (2018)
Ref_id:b36 Title: Dataless knowledge fusion by merging weights of language models Year: (2023)
Ref_id:b37 Title: A solution for the best rotation to relate two sets of vectors Year: (1976)
Ref_id:b38 Title: Advances in neural information processing systems Year: (2024)
Ref_id:b39 Title: Federated learning: Strategies for improving communication efficiency Year: (2016)
Ref_id:b40 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b41 Title: The hungarian method for the assignment problem Year: (1955)
Ref_id:b42 Title: Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles Year: (2017)
Ref_id:b43 Title: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension Year: (2020)
Ref_id:b44 Title: Branch-train-merge: Embarrassingly parallel training of expert language models Year: (2022)
Ref_id:b45 Title: Deep model fusion: A survey Year: (2023)
Ref_id:b46 Title: Graph metanetworks for processing diverse neural architectures Year: (2024)
Ref_id:b47 Title: The empirical impact of neural parameter symmetries, or lack thereof Year: (2024)
Ref_id:b48 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b49 Title: Symmetry induces structure and constraint of learning Year: (2024)
Ref_id:b50 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b51 Title: Mechanistic mode connectivity Year: (2023)
Ref_id:b52 Title: Linear assignment problems Year: (1987)
Ref_id:b53 Title: Merging models with fisherweighted averaging Year: (2022)
Ref_id:b54 Title: Communication-efficient learning of deep networks from decentralized data Year: (2017)
Ref_id:b55 Title: Optimizing relu neural networks in its positively scale-invariant space Year: (2019)
Ref_id:b56 Title: Rectified linear units improve restricted boltzmann machines Year: (2010)
Ref_id:b57 Title: Equivariant architectures for learning in deep weight spaces Year: (2023)
Ref_id:b58 Title: Equivariant deep weight space alignment Year: (2024)
Ref_id:b59 Title: Pathsgd: Path-normalized optimization in deep neural networks. Advances in neural information processing systems Year: (2015)
Ref_id:b60 Title: How do vision transformers work? Year: (2022)
Ref_id:b61 Title: Re-basin via implicit sinkhorn differentiation Year: (2023)
Ref_id:b62 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b63 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b64 Title: Diverse weight averaging for Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion out-of-distribution generalization Year: (2022)
Ref_id:b65 Title: On permutation symmetries in bayesian neural network posteriors: a variational perspective Year: (2023)
Ref_id:b66 Title: Ensemble learning: A survey Year: (2018)
Ref_id:b67 Title: A generalized solution of the orthogonal procrustes problem Year: (1966)
Ref_id:b68 Title: Geometry of the loss landscape in overparameterized neural networks: Symmetries and invariances Year: (2021)
Ref_id:b69 Title: Model fusion via optimal transport Year: (2020)
Ref_id:b70 Title: Linear algebra and its applications Year: (1976)
Ref_id:b71 Title: Optimizing mode connectivity via neuron alignment Year: (2020)
Ref_id:b72 Title: Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition Year: (2003)
Ref_id:b73 Title: Equivariant neural functional networks for transformers Year: (2024)
Ref_id:b74 Title: Least-squares estimation of transformation parameters between two point patterns Year: (1991)
Ref_id:b75 Title: Attention is all you need Year: (2017)
Ref_id:b76 Title: Glue: A multi-task benchmark and analysis platform for natural language understanding Year: (2019)
Ref_id:b77 Title: Federated learning with matched averaging Year: (2020)
Ref_id:b78 Title: Are large language models good incontext learners for financial sentiment analysis Year: (2025)
Ref_id:b79 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022)
Ref_id:b80 Title: Ties-merging: Resolving interference when merging models Year: (2023)
Ref_id:b81 Title: What matters for model merging at scale? arXiv preprint Year: (2024)
Ref_id:b82 Title: Adamerging: Adaptive model merging for multi-task learning Year: (2024)
Ref_id:b83 Title: Graph transformer networks Year: (2019)
Ref_id:b84 Title: Bayesian nonparametric federated learning of neural networks Year: (2019)
Ref_id:b85 Title: Improving learning to optimize using parameter symmetries Year: (2025)
Ref_id:b86 Title: Resolving editing-unlearning conflicts: A knowledge codebook framework for large language model updating Year: (2025)
Ref_id:b87 Title: Symmetry teleportation for accelerated optimization Year: (2022)
Ref_id:b88 Title: Symmetries, flat minima, and the conserved quantities of gradient flow Year: (2023)
Ref_id:b89 Title: Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion Year: (2024)
Ref_id:b90 Title: Kg-cf: Knowledge graph completion with context filtering under the guidance of large language models Year: (2024)
Ref_id:b91 Title: Permutation equivariant neural functionals Year: (2023)
Ref_id:b92 Title: Neural functional transformers Year: (2023)
Ref_id:b93 Title: Informer: Beyond efficient transformer for long sequence time-series forecasting Year: (2021)
Ref_id:b94 Title: Merging large language models using model exclusive task arithmetic Year: (2024)
Ref_id:b95 Title: Going beyond linear mode connectivity: The layerwise linear feature connectivity Year: (2023)
Ref_id:b96 Title: Understanding and modeling job marketplace with pretrained language models Year: (2024)
Ref_id:b97 Title: Parameter symmetry and noise equilibrium of stochastic gradient descent Year: (2024)
Ref_id:b98 Title: Remove symmetries to control model expressivity and improve optimization Year: (2025)
