Title: RidgeLoRA: Matrix Ridge Enhanced Low-Rank Adaptation of Large Language Models
Abstract: As one of the state-of-the-art parameter-efficient fine-tuning (PEFT) methods, Low-Rank Adaptation (LoRA) enables model optimization with reduced computational cost through trainable low-rank matrix. However, the low-rank nature makes it prone to produce a decrease in the representation ability, leading to suboptimal performance. In order to break this limitation, we propose RidgeLoRA, a lightweight architecture like LoRA that incorporates novel architecture and matrix ridge enhanced full-rank approximation, to match the performance of full-rank training, while eliminating the need for high memory and a large number of parameters to restore the rank of matrices. We provide a rigorous mathematical derivation to prove that RidgeLoRA has a better upper bound on the representations than vanilla LoRA. Furthermore, extensive experiments across multiple domains demonstrate that RidgeLoRA achieves better performance than other LoRA variants, and can even match or surpass full-rank training.

Section: Introduction
Large Language Models (LLMs) with large number of parameters [1][2][3][4][5][6][7][8] have demonstrated exceptional performance in natural language generation tasks. These models acquire their primary knowledge during the pre-training phase, through training on massive high-quality datasets from both real-world corpora or model-generated synthetic data. To align with real-world scenarios [9], LLMs also require supervised fine-tuning (SFT). Traditionally, this fine-tuning process employs full-parameter (also full-rank) training (FFT) to achieve optimal performance. However, this approach demands substantial computational resources.
When adapting LLMs for downstream tasks, training is often constrained by limited computational resources, calling for efficient and lightweight solutions. Parameter-Efficient Fine-Tuning (10, 11, PEFT) methods achieve comparable performance to full-parameter fine-tuning with minimal cost. Low-rank adaptation (LoRA, 12), as a representative PEFT method, is widely adopted due to the fact that downstream tasks largely rely on the generic capabilities developed in pre-training. To maintain model performance, LoRA's initial state should also align with the original model's output, which we refer to as "the transform-calibrating restriction".
However, vanilla LoRA, while significantly reducing the number of trainable parameters, is often criticized for its lower performance ceiling [13][14][15]. Its low-rank nature results in significantly lower representation capability compared to full-parameter fine-tuning, making it prone to underfitting in downstream tasks [14]. Existing LoRA variants [16][17][18][19][20][21] primarily focus on matrix decomposition or Figure 1: Differences between LoRA (left) and RidgeLoRA (right): As is depicted, though LoRA reduces the number of parameters to be trained, the matrix rank severely shrinks. In order to be comparable with full rank training, the proposed RidgeLoRA introduces a matrix ridge (formalized as λΣ) to complement the rank of the trainable parameters. numerical stability, proposing better parameter initialization methods or updating strategies. However, these methods lack theoretical investigation on full-rank matrix approximation and the fundamental challenges of representation ability in low-rank settings. Additionally, exploring architectural alternatives beyond the vanilla LoRA framework could potentially unlock new opportunities for improvement.
Unlike existing works, RidgeLoRA incorporates a full-rank module to achieve performance comparable to full-parameter training. We redesign the architecture by replacing the parallel connection in vanilla LoRA with a series connection module. Inspired by the Matrix Ridge algorithm [22], we enhance vanilla LoRA by incorporating a ridge term, thus introducing RidgeLoRA. The architectural differences between RidgeLoRA and vanilla LoRA are illustrated in Figure 1. Based on observations from previous works [17,18] and our experiments, appropriate initialization methods can significantly improve low-rank training performance. As RidgeLoRA adopts a series connection architecture, it enables different parameter initialization approaches even under "the transform-calibrating restriction". RidgeLoRA achieves comparable performance to full-parameter training at minimal cost without introducing additional computation and parameters. Our main contributions are summarized as three-fold:
• We propose RidgeLoRA, a novel LoRA variant that replaces parallel connection with series connection, enabling better parameter initialization and enhanced representation capability.
• We introduce a diagonal ridge term alongside the low-rank matrices, inspired by the matrix ridge algorithm [22], which improves approximation flexibility while maintaining computational efficiency, to better represent the high-rank updates of LLM training.
• We provide theoretical analysis of RidgeLoRA's representation capability and demonstrate its superior performance through extensive experiments across multiple datasets.
2 Related Works
this section cite: ['b1', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b12', 'b13', 'b14', 'b13', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b16', 'b17', 'b21']

Section: Matrix Low-Rank Decomposition and LLMs
Matrix decomposition aims at optimizing the following objective:
min ∥W -W r ∥ 2 F , where W ∈ R din×dout is the target matrix, W r represents its low-rank approximation with rank r < min(d in , d out ) and ∥ • ∥ F denotes the Frobenius norm. Given a restricted rank r, Singular value decomposition (SVD, 23) has a bounded (also the minimum) error, which justifies its wide application, whose details can be found in Appendix A. 3. In order to accelerate the inference
this section cite: []

Section: LoRA Method Number of Parameters Weight Initialization Complexity

this section cite: []

Section: Forward Formalization Computation Complexity
FFT din × dout O(1) XinW ′ O(dfoot_0 in dout) LoRA [12] r × (din + dout) O(rdin) Xin(W + α r AB) O(d 2 in dout) DoRA [16] r × (din + dout) + dout O(rdindout)
( ∥W ∥2 ∥W + α r AB∥2 -1)XW + ∥W ∥2 ∥W + α r AB∥2 • XAB • α r O(d 2
in dout) PiSSA [17] r × (din + dout) O[max(din, dout) • min(d 2 in , d 2 out )] Xin(W P res + α r AB) O(d 2 in dout) KaSA [18] r × (din + dout + 1) O[max(din, dout) • min(d 2 in , d 2 out )] Xin(W K res + α r AΣrB) O(d 2 in dout) RidgeLoRA r × 2din + (din + 1) O(r 2 din) Xin(λΣ
+ α r AB)W O(d 2 in dout)
Table 1: Comparisons between FFT and other LoRA methods, here we focus on analyzing the number of parameters from the perspective of a single matrix. The comparison here also considers the complexity of weight initialization, where some methods may conduct SVD or matrix multiplications. W P res and W K res denote different ways to initialize W res in their works. RidgeLoRA is showcased with simple initialization method, small number of parameters, and will not increase the computation and memory requirements during training. speed, SVD-LLM [24] utilizes SVD compression method together with data whitening [25] to minimize ∥XW -XW r ∥ 2 F to compress LLMs. MoDeGPT [26] conducts detailed analysis of each matrix calculation operation inside an LLM and selects the corresponding decomposition method accordingly, namely SVD, Nyström approximation [27] and CR decomposition [28]. As an inspiration of this paper, Matrix Ridge [22] proposes an algorithm that approximates a positive semi-definite matrix using a combination of an incomplete matrix decomposition and a ridge term. This achieves tighter approximation than both incomplete Cholesky decomposition [29] and incomplete spectral decomposition, while ensuring that the condition number of the approximated matrix does not exceed that of the original matrix.
this section cite: ['b11', 'b15', 'b16', 'b17', 'b23', 'b24', 'b25', 'b26', 'b27', 'b21', 'b28']

Section: Parameter-Efficient Fine-Tuning
LoRA and its Variants LoRA and its variants [12,[16][17][18][19][20][21][30][31][32][33][34][35] train extra low-rank weights on top of the model, which makes low-rank matrix decomposition naturally suitable for the improvement of it. Considering "the transform-calibrating restriction", advanced LoRA variants like PiSSA [17], LoRA-GA [30], MiLoRA [31], LoRA-XS [32] and KaSA [18] conduct SVD on the original matrices to endow the low-rank matrices with better initializations, thus achieving better performance. Furthermore, some works focus more on the numerical stability of matrices, rsLoRA [34] and proposes a better setup of the scaling factor from a statistical point of view. To achieve training stability, DoRA [16] decomposes the updates of matrices to magnitude factor and direction factor, hereby adding a scaling factor during training. VeRA [35] adds learnable scaling vectors which can be updated and tune frozen random matrices across layers, which changes a bit of the architecture.
Apart from parameter-based methods, LoRA+ [20] assigns the matrix B initialized with all zeros with larger learning rate, LoRA-Pro [19] updates the matrices from the perspective of transformation invariance. These works all focus on how to deal with the gradients in the optimization stage.
Other PEFT Methods LoRA-based methods and variants can be attributed to one type of PEFT [36] method, which also includes (1) Selective training: BitFit [37]; (2) Soft prompt: Prefix Tuning [38] and P-Tuning [39,40]; (3) Adapter-based method: Serial Adapter 2 [41] and Parallel Adapter [42]. The latter two types also insert extra trainable modules while keeping the pre-trained matrices frozen like LoRA does.
3 RidgeLoRA: Matrix Ridge Enhanced LoRA
this section cite: ['b11', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b16', 'b29', 'b30', 'b31', 'b17', 'b33', 'b15', 'b34', 'b19', 'b18', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41']

Section: Architecture of RidgeLoRA
RidgeLoRA switches the computation graph to the following formalization:
X out = X in (λΣ + α r AB)W,(1)
where Σ is a diagonal matrix (referred to as the Ridge) that requires gradient descent. It has d in learnable parameters on the matrix diagonal, with its matrix rank d in . λ denotes the "Ridge Intensity" which can be updated. Specifically, RidgeLoRA inserts additional trainable ridge beside the matrix and novelly converts the extra modules into series connections, as opposed to the parallel connections used in LoRA variants. Moreover, the newly trained part can also be absorbed back to the original weight as described in the following formalization:
W ′ = (λΣ + α r AB)W,(2)
where the trained matrix W ′ is the product of two full-rank matrices. This absorption recovers the original architecture while endowing the model with brand new optimized weights during inference, which provides the same advantage as LoRA and its variants. Moreover, in order to provide a clear comparison to main-stream tuning methods, we further list the properties of them in Table 1.
this section cite: []

Section: Theoretical Analysis of RidgeLoRA
According to the setups of RidgeLoRA, λΣ is initialized from a part of a diagonal matrix, thus enabling it to dominate the spectrum of λΣ + α r AB and achieve a high rank of d in . Here we conduct derivations on how closely the enhanced RidgeLoRA architecture can approximate the full-rank weight update ∆W . By introducing this ridge term, we effectively increase the model's rank expressiveness and improve the overall performance.
Theorem 3.1. Let K ∈ R d×d be a rank-k matrix (k ≤ d), D ∈ R d×d be a diagonal matrix, M ∈ R d×d be an arbitrary matrix. Given that M has a Singular Value Decomposition (SVD) M = U ΣV ⊤ . Let A ∈ R n(n-1)×k be a matrix whose rows are indexed by ordered pairs (p, q) where p, q ∈ {1, 2, ..., n} and p ̸ = q. Each row a (pq) ∈ R 1×k of A has entries given by U pj V qj for j ∈ {1, 2, ..., k}. Let c ∈ R n(n-1)×1 be a vector whose entries are c pq = n j=k+1 U pj s j V qj , where s j denotes the corresponding entries in Σ. We have that
min K,D ∥K + D -M ∥ 2 F ≤ ∥(I -A(A ⊤ A) † A ⊤ )c∥ 2 F ,(3)
where (•) † denotes the pseudo inverse of matrix.
This demonstrates the advantages of adding Ridge. Specifically, the right side of the above expression is less than or equal to the LoRA case. For the detailed proof, please refer to Appendix A.1. Next, we will elaborate on this point.
In the case of LoRA, D is a zero matrix, and the minimization objective becomes
min K ||K -M || 2 F ,
Let K be a rank-k matrix. According to the Eckart-Young-Mirsky theorem [43], the optimal choice K = U Σ k V ⊤ minimizes the Frobenius norm ∥K -M ∥ 2 F . Substituting this optimal K, we obtain
∥K -M ∥ 2 F = ∥U (Σ -Σ k )V ⊤ ∥ 2 F ,
where Σ k denotes the best rank-k approximation of Σ, obtained by retaining the top-k largest singular values (ordered in descending magnitude) and zeroing out the rest.
In the proof of Theorem 3.1, after introducing the diagonal matrix D, if keeping K fixed as K = U Σ k V ⊤ , the problem reduces to a least squares optimization with respect to the entries of D. Clearly, D = 0 is not the optimal solution. The bound we established demonstrates that our method is more effective, detailed analysis can be found in Appendix A.1 to illustrate a theoretical measure.
this section cite: ['b42']

Section: Detailed Designs of RidgeLoRA
On top of the theoretical analysis of how RidgeLoRA facilitates full-rank training, additional mechanisms are introduced to optimize performance. Specifically, it incorporates a novel weight initialization strategy and an auxiliary loss function for the efficient updates of parameters.
Algorithm 1 Weight Initialization of RidgeLoRA Input: Input dimension d in , Target low rank r, Scaling factor α Output: λ, Σ, A, B 1: Initialize λ ← 1. ▷ Initialize the intensity of ridge term. 2: Sample noise vector N ∈ R r using N = σ[N (µ N , σ 2 N )], where σ(•) is the sigmoid function.
▷ Split a small portion to initialize low-rank matrices. 3: Construct matrix Σ ∈ R din×din as
Σ = diag(N) 0 0 I din-r ,
where diag(N) is an r × r diagonal matrix, and I din-r is an identity matrix. 4: Initialize low-rank matrix A ∈ R din×r with Gaussian Noise N (0, 1/r)
this section cite: []

Section: 5: Conduct SVD on matrix A: U A Σ A V ⊤
A = A. 6: To ensure the condition ▷ This ensures the "transform-calibrating restriction".
λΣ + α r AB = I, B is initialized with B = r α V A Σ -1 A U ⊤ A (I din -λΣ) . ▷ Note that U ⊤ A U A = V ⊤ A V A = I r .
this section cite: []

Section: Weight Initialization
In the field of deep learning, matrices are usually initialized with welldesigned methods [44][45][46] for a better starting point. However, vanilla LoRA is constrained by the "transform-calibrating restriction" [12], which allows only the initialization of matrix A with such methods, while B is initialized as an all-zero matrix. This limitation hinders the ability of LoRA to fully explore the parameter space. With the novel series connection, this restriction can be fulfilled with λΣ + α r AB = I, eliminating the strong constraint that requires B = 0. This ensures good initializations can be adopted on all matrices, endowing them with the possibility to get efficiently updated. In practice, given AB's target low rank r, a small portion (r × r) of the identity matrix is initialized with a diagonal matrix diag(N). This leads to be both part (the Ridge and the low-rank matrices) to be non-zero, allowing for good initialization of AB while ensuring λΣ dominates the spectrum. Subsequently, RidgeLoRA initializes the matrix A (with a Gaussian distribution, by default), and then compute matrix B to ensure the sum of these modules equal to the identity matrix. The whole procedure is also demonstrated in Alg. 1 for a clear presentation.
It is noteworthy that this process also differs from SVD-based methods [17,18] that initialize matrices with the eigenvalues and eigenvectors of the pre-trained matrix. They require massive computation to conduct SVD on the full-rank original matrices and split corresponding portion to initialize the low-rank matrices, while RidgeLoRA only requires SVD on the low-rank matrix A, quantitative comparisons can also be found at Table 1.
Ridge Squashing Loss The initialization of the low-rank modules ensures that their sum equals an identity matrix I, hereby adhering to the "transform-calibrating restriction". Furthermore, to encourage the matrix to explore a broader space, RidgeLoRA adds a loss term L RS , pushing the weight update towards higher rank:
L RS = 1 LB L l=1 B b=1 λ l,b Σ l,b + α r A l,b B l,b -I * , (4
)
where L is the total number of layers in the model, B denotes the number of matrix blocks in each layer, and ∥ • ∥ * is the nuclear norm [47], which measures the rank of the weight updates. Notably, β RS is set to a negative value, which encourages rank updates in the non-diagonal portions, hereby increasing the effectiveness of training the matrices A and B. We further incorporate this loss into the original objective: where L model is the task-related loss for the PEFT-enhanced base model, and β RS is a hyperparameter controlling the influence of the rank-decreasing penalty. Our experimental results and ablation study also indicate that selecting a negative value further improves performance.
L = L model + β RS L RS ,(5)
this section cite: ['b43', 'b44', 'b45', 'b11', 'b16', 'b17', 'b46']

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets In order to showcase the validity of RidgeLoRA and demonstrate its good performance. In comparisons with state-of-the-art low rank methods, we conduct comprehensive experiments across different tasks, which are widely utilized for evaluation in previous works: namely, (i) Commonsense Reasoning, (ii) Math & Code Problems and (iii) Multi-modal Understanding tasks. We adopted all of its training split for fine-tuning for fixed number of steps to ensure fair comparisons. Reported metrics are evaluated on the official test splits.
Baselines Across all of our experiments, we mainly compare RidgeLoRA with low-rank based tuning methods, namely vanilla LoRA [12], DoRA [16] and SVD-based PiSSA [17] and KaSA [18]. To further demonstrate the comparable performance of RidgeLoRA on par with FFT, we also include FFT in our main results. Details of the experiments can be found at Appendix B.4.
this section cite: ['b11', 'b15', 'b16', 'b17']

Section: Detailed Setups
Throughout our experiments, we adopt a cosine learning rate schedule and use AdamW [48] as the optimizer. Unless otherwise specified, all LoRA variants share the same maximum learning rate for a given task. Concretely, we use a learning rate of 2 × 10 -5 for Math & Code tasks, and 3 × 10 -5 for Commonsense tasks. The rank of the low-rank matrices is set to 64 for Math & Code tasks, and 8 for Commonsense tasks. For multi-modal understanding, we follow the configuration proposed in DoRA [16] for a fair comparison.
this section cite: ['b47', 'b15']

Section: Main Experimental Results
Commonsense Reasoning As is showcased in Table 2, across eight commonsense reasoning datasets, the average scores achieved by our proposed RidgeLoRA outperforms most of LoRA variants. When choosing Llama-2-7B as the base model, RidgeLoRA outperforms state-of-the-art baseline, i.e., PiSSA, by an improvement of 0.86% in the average accuracy. When analyzing each single dataset, we observe that RidgeLoRA surpasses most of the tuning methods including FFT by large margins. For example, RidgeLoRA outperforms LoRA by a 6.0% accuracy improvement on WinoGrande with Llama-2-7B as the base model, and is generally better on every dataset than KaSA across different base models. From the results we can observe that only in few datasets that RidgeLoRA may not surpass baselines, with the performance drops usually do not exceed 1%.
As for low-rank methods' performance in calibrating FFT, our proposed RidgeLoRA is the closest to the performance of FFT when selecting Llama-2-7B as the base model, which demonstrates the advantages of full-rank training of RidgeLoRA, with a performance drop of only 0.15%. We also observe that when selecting Llama-3.1-8B and Mistral-v0.3-7B as the base model, the performances of FFT are commonly exceeded by low-rank based methods, which are also observed by many previous works [17,18,15], that base model may find it harder to converge given limited data in some tasks comparing to low-rank methods.
this section cite: ['b16', 'b17', 'b14']

Section: Method

this section cite: []

Section: ↓
Trainable Parameters (%) GSM8K MATH HumanEval (+) MBPP (+) (Acc.) (Acc.) (Pass@1) (Pass@1) Llama-2-7B FFT 100% (6.74B) 66.32 17.72 37.8 (35.4) 45.2 (36.8) LoRA 2.32% (159.9M) 53.44 8.94 22.6 (18.3) 37.0 (31.0) DoRA 2.34% (161.3M) 52.25 8.08 24.4 (20.1) 36.2 (31.0) PiSSA 2.32% (159.9M) 56.29 9.28 25.0 (20.1) 36.8 (29.4) KaSA 2.33% (160.8M) 49.18 7.20 22.0 (19.5) 35.4 (29.4) RidgeLoRA 2.13% (147.0M) 56.44 9.76 26.2 (23.8) 37.3 (29.9) Llama-3.1-8B FFT 100% (8.03B) 77.77 28.84 58.5 (55.5) 64.0 (55.6) LoRA 2.05% (167.8M) 75.97 28.30 51.2 (47.0) 67.5 (56.6) DoRA 2.06% (169.2M) 76.05 27.82 51.2 (48.2) 67.7 (56.6) PiSSA 2.05% (167.8M) 77.92 30.26 53.7 (50.0) 65.1 (56.1) KaSA 2.06% (168.7M) 75.82 27.52 53.0 (50.6) 68.5 (58.2) RidgeLoRA 1.96% (160.7M) 78.07 30.12 53.7 (51.8) 69.8 (59.8) Mistral-v0.3-7B FFT 100% (7.25B) 68.11 21.68 49.4 (47.0) 51.3 (43.1) LoRA 2.26% (167.8M) 71.56 21.56 45.7 (39.0) 62.2 (51.1) DoRA 2.28% (169.2M) 72.83 22.08 45.1 (39.0) 60.8 (51.3) PiSSA 2.26% (167.8M) 72.53 22.96 47.6 (40.9) 62.7 (51.3) KaSA 2.28% (168.7M) 74.33 23.06 47.6 (40.2) 62.2 (50.5) RidgeLoRA 2.17% (160.7M) 73.88 24.02 48.2 (41.5) 63.2 (53.7)
Table 3: Performances of different models on math & code problem benchmarks. For the code benchmarks, (+) denotes the enhanced datasets with more difficult test cases, whose metrics are in the parentheses, best performances of low-rank methods are marked as Bold. Math & Code Problems We also evaluate RidgeLoRA together with its baselines on math and code problem solving datasets, where models are usually required to generate long-form arithmetic reasoning trace (math) and complete executable program (code), to further test the performance. As is demonstrated in Table 3, RidgeLoRA surpasses almost all of its baselines with different LLMs as base models. Taking the math benchmarks with Llama-2-7B as the base model as an example, except for FFT, RidgeLoRA outperforms all of its low-rank baselines, even outperforms KaSA by 7.26% on GSM8K. Similar results hold for Llama-3.1-8B and Mistral-0.3-7B models, where RidgeLoRA outperforms its baselines by 0.56% in average and even surpasses FFT by an improvement of 2.42% in the result accuracy. In few datasets, RidgeLoRA may get surpassed by SVD-based PiSSA or KaSA, but with an average margin of only 0.30%.
Similar results hold for the code benchmarks. On the datasets with original test cases, RidgeLoRA surpasses every low-rank method and outperforms the best of its baseline performances with a 0.65% improvement. As for datasets with enhanced test cases, i.e., the ones with (+), RidgeLoRA outperforms the low-rank methods by a 1.4% improvement, showcasing its good performance and generalizability on hard cases. Like Commonsense Reasoning, we also observe several low-rank methods even achieve better performance than FFT, the reason is that the base models already possess enough capabilities, thus possible to achieve good results by tuning fewer parameters on the training set with less data.
Method ↓ Trainable Parameters (%) GQA SQA VQA T POPE Avg. FFT 100% 61.9 66.8 58.2 85.9 68.20 LoRA 4.61% 62.9 68.4 58.2 86.4 68.98 DoRA 4.63% 62.9 69.9 57.0 87.2 69.25 RidgeLoRA 4.26% 61.9 69.2 58.4 87.6 69.28
Table 4: Multi-modal understanding evaluation results of different tuning methods on 4 vision-language tasks following the setups of DoRA [16], best performances are marked Bold.
Multi-modal Understanding In order to expand the applicable domain of RidgeLoRA, we further evaluate RidgeLoRA's performance in aligning a pre-trained a language model with a multi-modal projector. The evaluation results can be found at Table 4. As is showcased, it is showcased that language model tuned with RidgeLoRA achieves better performances than DoRA, with an improvement of at most 1.4% comparing to DoRA on VQA T . Consistent with the training of language-only models, the ratio of trainable parameters remains lower than low-rank baselines, demonstrating the lightweight advantages of RidgeLoRA.
this section cite: ['b15']

Section: Ablation Study
Apart from comprehensive evaluation with various types of model and different tasks comparing with baselines, which provides solid evidences that the RidgeLoRA surpasses vanilla LoRA together with its state-of-the-art variants, we also conduct ablation study to demonstrate the validity of different parts of RidgeLoRA at a fine-grained level.
Method GSM8K MATH (Acc.) (Acc.) LoRA+Ridge 55.39 (↑1.95) 9.14 (↑0.20) DoRA+Ridge 54.12 (↑1.87) 8.72 (↑0.64) PiSSA+Ridge 56.36 (↑0.07) 9.56 (↑0.28) KaSA+Ridge 51.42 (↑2.24) 7.98 (↑0.78)
Table 6: Performances on math datasets existing LoRA variants enhanced with parallel Ridge.
Ridge Enhanced Parallel Connections Since we modify the connection between the newly trained modules and the original matrix, in order to provide in-depth analysis about the necessity of series connection, we also conduct ablation study with different LoRA variants enhanced with a Ridge alongside. The experimental results can be found in the first group of Table 5 and Table 7: Performances on math datasets of RidgeLoRA with different initializations.
this section cite: []

Section: Weight Initialization Methods
Here we remain the series connection setup and further compare the performances of different initialization methods of matrix A, namely Gaussian [44], Xavier [45], and Kaiming [46] initialization, the results can be found at the second group in Table 5 and Table 7. We also include an initialization where matrix A only has diagonal values before training, termed as Diagonal. From the results we can observe that randomized matrix initialization always leads to better performances, which is demonstrated by the fact that Diagonal gets surpassed by others by a margin of at least 0.78%. As for the normal distribution initialized methods, we can observe that Gaussian, with a higher variance 1 r in our default setup, has better performances than Kaiming and Xavier, with a 0.64% improvement on GSM8K. Moreover, initializing with uniform distributions won't lead to good performances when compared to normal distributions, their performance drop can achieve up to 2.20% on OBQA if taking Kaiming (U) as an example.
this section cite: ['b43', 'b44', 'b45']

Section: Pattern Analysis of Weight Updates
As an auxiliary analysis, we also performed a visualization of the equivalent parameter updates of different methods. As is illustrated in Figure 2, constrained with the low-rank nature, the update norms of other LoRA variants are always lower than those of FFT, especially the vanilla LoRA. Benefit from the full-rank nature of RidgeLoRA, its updates are comparable or sometimes larger than that of FFT, showcasing its good representation capability.
Ridge Squashing Loss Fusion As we fuse Ridge Squashing Loss with the original loss to facilitate the training process, in order to prove its validity, we conduct ablation study with different hyperparameter setups. From the evaluation results from Table 8, we can observe that on all eight datasets, a negative value of β RS tends to lead to better model performances, with a gap of at most 1.21% in average score across two groups. Comparing to the baseline group (w/o L RS ), selecting a negative values always bring about better performance, with its scores all above baseline accuracy. Furthermore, we conduct Student's t-test [49] to showcase that selecting a negative value for β RS significantly improves the performance, which means increasing the rank of weight updates is favorable for good performances. Details can be found at Appendix B.1.
this section cite: ['b48']

Section: Conclusion
In this paper, we propose RidgeLoRA, a novel PEFT approach that enhances the vanilla LoRA architecture through two key innovations: replacing the conventional parallel structure with series connection and incorporating a diagonal ridge term. Combined with the well designed initialization strategy and ridge squashing loss, RidgeLoRA achieves superior capability while maintaining computational efficiency. Through rigorous theoretical analysis and comprehensive experimental evaluations, we demonstrated that RidgeLoRA consistently outperforms existing approaches. Our work opens up new possibilities for PEFT and provides valuable insights into PEFT methods.
this section cite: []

Section: Limitations
While RidgeLoRA is supported by comprehensive experiments and rigorous mathematical derivations. However, several limitations remain: 1) Its potential in scenarios such as continual learning [50,51] and model editing [52] where vanilla LoRA is commonly applied has not yet been explored. Future work could investigate how RidgeLoRA extends to these settings and whether it offers advantages in such contexts. 2) Due to computational constraints, we do not include experiments on very large models (e.g., those with more than 70B parameters). Nonetheless, the reported results, along with detailed ablations, provide a compelling demonstration of the method's effectiveness on a range of models.
this section cite: ['b49', 'b50', 'b51']

Section: A Theorems and Math Derivations

this section cite: []

Section: A.1 Proof of Theorem 3.1
Theorem A.1. Let K ∈ R d×d be a rank-k matrix (k ≤ d), D ∈ R d×d be a diagonal matrix, M ∈ R d×d be an arbitrary matrix. Given that M has a Singular Value Decomposition (SVD) M = U ΣV ⊤ . Let A ∈ R n(n-1)×k be a matrix whose rows are indexed by ordered pairs (p, q) where p, q ∈ {1, 2, ..., n} and p ̸ = q. Each row a (pq) ∈ R 1×k has entries given by U pj V qj for j ∈ {1, 2, ..., k}. Let c ∈ R n(n-1)×1 be a vector whose entries are c pq = n j=k+1 U pj s j V qj , where s j denotes the corresponding entries in Σ. We have that
min K,D ∥K + D -M ∥ 2 F ≤ ∥(I -A(A ⊤ A) † A ⊤ )c∥ 2 F (6
)
Proof. We construct an upper-bound objective function via a relaxation of the original problem. Perform the singular value decomposition (SVD) of M as M = U ΣV ⊤ . Define K = U Σ k V ⊤ , where Σ k is a rank-k diagonal matrix which is left for optimization. Then we have
min K,D ∥K + D -M ∥ 2 F ≤ min Σ k ,D ∥U Σ k V ⊤ + D -U ΣV ⊤ ∥ 2 F = min Σ k ,D ∥D -U (Σ -Σ k )V ⊤ ∥ 2 F ,(7)
which serves as a tractable upper bound of the original objective.
Subsequently, we conduct two steps to find the upper bound of (7). The first step is to get the analystic value of D to minimize (7) while fixing Σ k . Then D can be represented in terms of Σ k . In the second step, we substitute D with its expression in terms of Σ k , and then minimize (7) with respect to Σ k .
Derivation of the optimal D. Fix Σ k , here we denote A = U (Σ -Σ k )V ⊤ . We need to minimize ∥D -A∥ 2 F with respect to D. Since D is diagonal, i.e., D ij = 0 where i ̸ = j. We have that ∥D -A∥ 2 F = i,j
(D i,j -A i,j ) 2 = i (D i,i -A i,i ) 2 + i̸ =j A 2 i,j .
The objective above is minimized when D i,i = A i,i . Therefore, problem (7) reduces to minimizing
i̸ =j U (Σ -Σ k )V ⊤ 2 i,j
, with respect to the rank-k diagonal matrix Σ k .
Derivation of the optimal Σ k . Let Y = Σ -Σ k . Since Σ and Σ k are diagonal matrices, Y is also a diagonal matrix. Let Σ = diag(s 1 , s 2 , ..., s n ) and Σ k = diag(t 1 , t 2 , ..., t n ). Note that there are exactly k non-zero diagonal entries in Σ k . Let J 0 ⊂ {1, ..., n} be the set of n -k indices where t j = 0. Let J k ⊂ {1, 2, ..., n} be the set of k indices where t j ̸ = 0. The target objective can be rewritten as follows
min J K ,{tj } j∈J k p̸ =q [U Y V ⊤ ] pq 2 = min J K ,{tj } j∈J k p̸ =q   n j=1 U pj y j V qj   2 . (8
)
For a fixed choice of the index set J k , this is a linear least square problems with respect to k variables {t j } j∈J k .
Let x denote the k × 1 vector whose elements are s j -t j for j ∈ J k .There are n(n -1) pairs of (p, q) with p ̸ = q. For each pair (p, q), let a (pq),J k be a 1 × k row vector with entries U pj V qj for j ∈ J k , c pq,J0 = j∈J0 U pj s j V qj . Let A J k be an n(n -1) × k matrix whose rows are a (pq),J k , and let c J k be an n(n -1) × 1 vector with entries c pq,J0 .
For a fixed index set J k , the minimization objective (8) can be rewritten as
min J k ,x ∥A J k x + c J k ∥ 2 F .
The solution to this least squares problem is given by
x * = -(A ⊤ J k A J k ) † A ⊤ J k c J k . The minimum value for a fixed J k is G(J k ) = ∥(I -A J k (A ⊤ J k A J k ) † A ⊤ J k )c J k ∥ 2 F .
Each choice of J k is an upper bound of the target objective. The final step is to find the optimal set of k indices J k that minimize G(J k ). There are n k possible choices for the set J k . The overall minimum value of the above objective function is
min J k ⊂{1,2...,n},|J k |=k ∥(I -A J k (A ⊤ J k A J k ) † A ⊤ J k )c J k ∥ 2 F . (9
)
The upper bound in the theorem can be obtained by setting J k = {1, 2, ..., k}.
this section cite: ['b6']

Section: A.2 Weight Updates of FFT
Here we present two perspectives on the weight updates during FFT. While low-rank matrices are usually used to approximate full-rank matrices, we propose enhancing this approximation with a ridge term, which provides better calibration to the original full-rank update scheme. There are two equivalent ways to view the weight updates of the original matrix W :
• Additive update (Parallel connection): W + ∆W ;
• Multiplicative update (series connection): W • ∆W ′ .
The full equivalence of additive update and full-rank training is derived further at Appendix A.5. If freezing the original matrix W , these two representations are equivalent, as we can establish a bi-directional mapping between them:
• Given any additive update ∆W , we can find its multiplicative counterpart as ∆W ′ = W -1 (W + ∆W );
• Given any multiplicative update ∆W ′ , we can find its additive counterpart as ∆W = W • ∆W ′ -W .
In both cases, theoretically W + ∆W = W • ∆W ′ . It is noteworthy that vanilla LoRA can be viewed as a low-rank approximation to the parallel connection, where ∆W is constrained to be the product of two low-rank matrices. We will also demonstrate the experimental results of series and parallel connections with the setup of RidgeLoRA.
this section cite: []

Section: A.3 Singular Value Decomposition with Math Bounded Error
According to Eckart-Young-Mirsky Theorem [43], when conducting low-rank decomposition with SVD on a matrix W to obtain a low-rank matrix W k with rank k. The decomposition error, calculated with Frobenius Norm [53], is equal to the sum of the squares of the singular values of the compressed part of the matrix, which can be formalized as
∥W -W k ∥ 2 F = min(din,dout) i=k+1 σ 2 i ,
where W = U ΣV T and W k = U Σ k V ⊤ , σ i is the singular value calculated with SVD, rearranged by descending order. This theorem also gives a math bound to the error of a low-rank decomposition approximating the original full-rank matrix, any matrix W k with rank k won't better approximate the matrix W . However, if breaking the low-rank restrictions with a full-ranked ridge, though with limited parameters, the approximation error gets way lower compared to SVD-based methods. This justifies the necessity of restoring the rank of low-parameter modules to gain better performances.
this section cite: ['b42', 'b52']

Section: A.4 Weight Updates of Vanilla LoRA
LoRA achieves efficient fine-tuning with the low-rank matrices, which is particularly effective when the target task requires capabilities that are closely related to those acquired during pre-training, necessitating only minimal weight adjustments. In the implementation of vanilla LoRA [12], an input signal X in is computed in the following way:
X out = X in (W + α r AB),(10)
where A ∈ R din×r , B ∈ R r×dout , r denotes the preset low rank of the two matrices and α acts as the scaling hyper-parameter. The low-rank setup freezes the original matrix W , making the weights to be updated lightweight. The architecture enables the extra matrices: A and B to be absorbed by the pre-trained weights, thus feasible to reorganized into a model with exactly the same structure as the original one. The weights can be reorganized as follows:
W ′ = W + α r AB. (11
)
Here we take a look at the weight update ∆W , it is an item-wise weight difference between fine-tuned weight W ′ and the original weight W :
∆W = W ′ -W = α r AB,(12)
where A and B are low-rank matrices, making this module short of representation capabilities. According to Biderman et al. [14], Shuttleworth et al. [15], LoRA can not fully calibrate full-rank training and may entail intruder dimensions for the existence of low-rank matrices.
this section cite: ['b11', 'b13', 'b14']

Section: A.5 Gradient Analysis of Full-Rank Training
Figure 3: Schematic of matrix computation with input matrix X in and output X out .
Here we analyze the matrix computation in detail and the difference between the weight updates of FFT and LoRA training. Specifically, as is depicted in Figure 3, X out is obtained by matrix computation from X in , which results in the following computation path:
   X out = X in W ∂X out ∂W = X ⊤ in (13
)
Theorem A.2. FFT is equivalent to adding a full-rank matrix through parallel connection while freezing the original matrix.
▷ Gradient of FFT: Suppose that the gradient ∂L ∂Xout has been backpropagated [54] from subsequent modules. Our goal is to compute the corresponding gradient w.r.t. the input, ∂L ∂Xin , through applying the chain rule. Specifically, given the transformation X out = X in W , the gradient can be calculated as:
g W = ∂L ∂W = ∂L ∂X out • ∂X out ∂W = ∂L ∂X out X ⊤ in , (14
)
from which we can observe that the gradient is calculated by the product of input activation and gradients from subsequent modules, which can be further utilized by optimizers like AdamW [48] to compute the weight updates.
▷ Gradient of an Additive (Parallel) Matrix: Here we derive how parameters of an extra inserted parallel matrix W ′ update during training, where the computation is transformed to the formalization below:
X out = X in (W + ∆W ), (15
)
where ∆W is full-rank while W is kept frozen during training. The gradient of the inserted matrix can be calculated from:
g ∆W = ∂L ∂∆W = ∂L ∂X out • ∂X out ∂(W + ∆W ) • ∂(W + ∆W ) ∂∆W = ∂L ∂X out • ∂X out ∂(W + ∆W ) • I = ∂L ∂X out X ⊤ in ,(16)
where the gradient is exactly the same as conducting full-rank training. If the original matrix W is frozen during training while the inserted matrix is initialized as (0) din×dout to ensure the output at starting point calibrate the original outputs, every optimization step of parallel added matrix is exactly the same as FFT, hereby leading to the same results.
this section cite: ['b53', 'b47']

Section: B Details and Analysis of Experiments

this section cite: []

Section: B.1 Hypothesis Test of β RS Selection
In order to complement the analysis of our ablation study in Section 4.3, we conduct t-test [49] on the accuracy scores to analyze the impact of β RS and compare the statistical significance to prove the validity of the proposed ridge squashing loss. Data points across different datasets and β RS values are from Table 8. In the hypothesis test, we refer to µ 1 and µ 2 as the means of the two experimental groups, negative value group and positive value group, respectively, while µ 0 represents a reference data point in which ridge squashing loss is not in effect for comparison in our t-test. The results of the hypothesis tests, including the t-statistics and p-values for each dataset, are summarized in Table 9.
β RS BoolQ PIQA SIQA HellaSwag WinoGrande ARC-e ARC-c OBQA Avg. (Acc.) (Acc.) (Acc.) (Acc.) (Acc.) (Acc.) (Acc.) (Acc.) (Acc
.) Negative Values -1.0 71.12 83.68 80.81 93.22 79.87 85.82 73.21 81.40 81.14 -5 × 10 -1 71.43 83.08 81.27 93.24 81.53 86.66 71.76 82.00 81.37 -2 × 10 -1 70.55 82.70 80.50 93.59 81.22 86.28 70.90 81.80 80.94 -1 × 10 -1 71.40 83.41 80.81 93.54 81.45 86.28 72.01 81.60 81.31 -5 × 10 -2 71.00 83.62 80.66 93.59 82.16 85.98 70.82 80.80 81.08 -2 × 10 -2 70.94 83.68 80.30 93.43 80.51 87.04 72.44 81.60 81.24 -1 × 10 -2 71.73 83.03 82.29 93.37 80.03 86.36 72.18 81.40 81.30 -5 × 10 -3 71.43 82.81 80.55 93.52 80.19 86.20 72.35 81.80 81.11 -1 × 10 -3 70.82 83.19 80.55 93.26 79.87 85.82 71.08 79.20 80.47 0 (w/o L RS ) 70.61 82.21 80.25 93.49 80.35 85.77 71.33 82.20 80.78 Positive Values 1 × 10 -3 70.61 82.21 80.04 92.71 80.03 85.48 72.01 79.60 80.34 5 × 10 -3 70.58 82.64 80.25 92.60 80.82 86.28 71.84 79.80 80.60 1 × 10 -2 71.12 82.54 80.04 92.83 80.74 85.56 71.33 80.8 80.62 2 × 10 -2 70.64 82.75 80.19 92.99 80.82 85.35 72.53 80.40 80.71 5 × 10 -2 70.42 81.66 80.45 92.73 80.82 85.56 71.84 80.80 80.54 1 × 10 -1 70.51 82.10 79.94 92.84 80.03 85.77 71.59 81.00 80.47 2 × 10 -1 71.15 81.77 80.14 92.91 79.87 85.56 71.25 80.20 80.36 5 × 10 -1 70.30 81.72 79.68 92.73 80.19 85.19 70.82 80.60 80.15 1.0 69.81 82.21 80.40 92.79 80.27 84.93 71.67 79.80 80.24
Table 8: Ablation study with different β RS values with Llama-2-7B as base model on Commonsense Datasets. Hyper-parameters, except for β RS , are set to align with the main experiment. We evaluate across positive and negative in comparison with the β RS = 0 test, i.e., the one without β RS , to obtain conclusions about how the loss takes effect. Observations from metrics above can be used for hypothesis test.
Test 1: Will fusing losses by a negative β RS lead to better performance?
Here we conduct t-test with accuracy scores from negative value group and µ 0 to prove the validity of fusing losses by a negative β RS . This is a right-tailed mean test, the null hypothesis and the alternative hypothesis can be expressed as
H 0 : µ 1 ≤ µ 0 (Null Hypothesis) H a : µ 1 > µ 0 (Alternative Hypothesis)
where the test statistic is given by t 1 = x1-µ0
this section cite: ['b48']

Section: S1/
√ n1 where n denotes the number of data points and S 2 is the variance, x is the mean of all data points from the same group. The null hypothesis is rejected if t 1 > t α,n1-1 , i.e., p < α, where the degree of freedom df 1 is n 1 -1. We set the significance level at p < 0.10. From the left of Table 9, the differences are significant across five of eight datasets, which proves the validity of L RS and indicates that setting β RS a negative value will improve the performance.
Dataset t 1 (µ 1 > µ 0 ) p 1 -value Significant BoolQ 4.4488 0.0021 PIQA 8.3350 0.0000 SIQA 3.0429 0.0160 HellaSwag -1.4335 0.8104 Winogrande 1.4475 0.0912 ARC-Easy 3.8073 0.0052 ARC-Challenge 1.9867 0.0822 OpenBookQA -3.1967 0.9937 Dataset t 2 (µ 2 < µ 1 ) p 2 -value Significant BoolQ -3.2042 0.0056 PIQA -5.8022 0.0000 SIQA -3.4093 0.0063 HellaSwag -9.8413 0.0000 Winogrande -1.1541 0.2722 ARC-Easy -4.1315 0.0008 ARC-Challenge -0.6627 0.5188 OpenBookQA -2.8795 0.0129
Table 9: Hypothesis test results for µ 1 > µ 0 (left) and µ 2 < µ 1 (right). Across six out of eight datasets, the average accuracy scores from the negative group µ 1 is significantly greater than µ 0 .
Meanwhile, µ 1 is significantly higher than µ 2 in six out of eight datasets, showcasing the necessity of selecting a negative β RS .
Test 2: Will fusing losses by a positive β RS worsen the performance compared to the negative group?
In order to fully perform ablation study to facilitate hyper-parameter selection, we also conduct Welch's t-test [55] to make a comparison between the positive and negative group for the uncertainty in the variance of the two groups. This is a left-tailed mean test, where the hypotheses can be expressed as
H 0 : µ 2 ≥ µ 1 (Null Hypothesis) H a : µ 2 < µ 1 (Alternative Hypothesis)
In this case, using the Welch-Satterthwaite formula [56,55], the t value and the degree of freedom df are calculated as follows:
t 2 = x2 -x1 S 2 2 n2 + S 2 1 n1 , df 2 = ( S 2 1 n1 + S 2 2 n2 ) 2 (S 2 1 /n1) 2 n1-1 + (S 2 2 /n2) 2 n2-1 .(17)
The null hypothesis is rejected if t 2 < -t α,df , i.e., p < α. From the right table of Table 9, six out of eight datasets are witnessed with a high significance, which illustrates selecting a negative β RS can be much better than picking a positive value for it. Furthermore, this proves that increasing the rank of the non-diagonal part of the weight update is more beneficial to obtain better performances.
this section cite: ['b54', 'b55', 'b54']

Section: B.2 Results of Natural Language Understanding Benchmark
We also evaluate RidgeLoRA with BERT [57]-like encoder models on Natural Language Understanding datasets from the GLUE [58] benchmark. We selected encoder-based discriminative models, RoBERTa [59] and DeBERTaV3 [60], to evaluate RidgeLoRA on NLU tasks on the GLUE benchmarks. Here we present the results from the NLU datasets in Table 10.
From the results in the table, we can observe that the performance improvements hold for small encoder models whose model sizes are usually less than 1B. The Ridge-enhanced model achieved the best results in almost all of the datasets, with its average scores also the highest amongst its baselines. On datasets where Ridge doesn't perform that well, the accuracy drop compared to other well-performed LoRA variants is usually less than 1%, showcasing the scalability and good performance of RidgeLoRA.
this section cite: ['b56', 'b57', 'b58', 'b59']

Section: B.3 Resource Consumption of LoRA Variants
In addition to the theoretical analysis as presented in Table 1, we also conduct real-world test of different LoRA variants on its memory cost and time consumed for initialization. The results are as reported in Table 11. As is depicted, the proposed RidgeLoRA hardly increases the computation and memory requirements during training. Consistent with the theoretical results, conducting SVD on original weights in order to initialize LoRA weights (like PiSSA or KaSA) takes way more time than LoRA, DoRA and RidgeLoRA. DoRA always consumes more FLOPs and memory for its decoupling the updates of magnitude and direction.
this section cite: []

Section: B.4 Details Descriptions of Experiments
Base Models As we select language generation tasks for evaluation, we choose our base models from state-of-the-art LLMs and select Llama-2-7B [4], Llama-3.1-8B [6] and Mistral-v0.3-7B [5] models. The reason for this selection is that we want to confirm the validity of RidgeLoRA on more models. Also, since the Llama-2 model follows the Multi-Head Attention (61, MHA) architecture, unlike the Grouped-Query Attention (62, GQA) that is adopted by the latter two models, it is beneficial to further prove the generalizability of our effects considering this variation in the architecture. 3For multi-modal understanding, following DoRA's [16] setup, we adopt vicuna-7b-1.5 [63] as Table 11: Test results of the resources (time, computation and memory) consumed of different LoRA variants. This corroborates the theoretical results from Table 1.
the language model and adopt the CLIP [64]-based vision projector in LLaVA [65] to obtain the representations of images.
this section cite: ['b3', 'b5', 'b4', 'b15', 'b62', 'b63', 'b64']

Section: Datasets
We used comprehensive datasets to evaluate the performance of RidgeLoRA in adapting the language models to different down-stream tasks. Starting from where LoRA is most widely used, we perform supervised fine-tuning (SFT, 9) on the Commonsense Reasoning datasets and evaluate it on eight corresponding datasets. In order to further evaluate the performances of RidgeLoRA on long-form generation tasks, we also include Math & Code Problems, which require the model to have a strong reasoning and generation ability; Furthermore, besides language-only evaluation, we extend our experiments to Natural Language Understanding datasets, where language models are aligned to visual projectors to understand images.
i. For the Commonsense Reasoning datasets, following previous works, we conduct multi-task training with the training split of eight related datasets, namely BoolQ [66], PIQA [67], SocialIQA [68], HellaSwag [69], WinoGrande [70], ARC-Easy, ARC-Challenge [71] and OpenbookQA [72]. These datasets require the large models to fully utilize the real-world commonsense knowledge to answer a question.
ii. As for Math&Code problems, we evaluate LLMs with GSM8K [73] and MATH [74] for math, HumanEval [75] and MBPP [76] for code capability. We conduct supervised fine-tuning with MetaMath [77] and Code-Feedbackfoot_2 for math and code, respectively, to ensure there is no data leakage.
iii. For the multi-modal understanding datasets, we include GQA [78], ScienceQA (79, SQA in Table 4), TextVQA (80, VQA T in Table 4) and POPE [81] to test how well the trained model fits with the vision projector and understands the images.
iv. We also adopt the GLUE benchmark 5 for the NLU tasks, which consists of the following datasets: a) Single Sentence Classification Tasks: SST-2 [82] or the Stanford Sentiment Treebank's goal is to predict the sentiment (positive/negative) of reviews on different movies, which is a binary classification task. CoLA [83] or the Corpus of Linguistic Acceptability consists of sentences each annotated with whether it is a grammatical English sentence. b) Similarity or Paraphrase Tasks: MRPC [84] or the Microsoft Research Paraphrase Corpus is to identify if a sentence pair consists of sentences paraphrases of each other. QQP or Quora Question Pairsfoot_4 is to determine whether two questions are semantically equivalent, question pairs are collected from the website Quora. STS-B [85] or the Semantic Textual Similarity Benchmark is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data. The task is to evaluate how similar two chunks of texts are with a score from 1 to 5. c) Language Entailment Tasks: MNLI [86] or the Multi-Genre Natural Language Inference is a crowdsourced dataset of sentence pairs with entailment annotations, sourced from diverse materials like speech, fiction, and reports, evaluated on both in-domain and cross-domain sections using private labels. QNLI or the Question Natural Language Inference consists of question-paragraph pairs from Wikipedia, originally from the SQuAD [87] and post processed when building GLUE. RTE or the Recognizing Textual Entailment is a binary entailment task with a small training dataset, which consists of sentence pairs from four annual textual entailment challenges [88][89][90][91].
this section cite: ['b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b75', 'b76', 'b77', 'b80', 'b81', 'b82', 'b83', 'b84', 'b85', 'b86', 'b87', 'b88', 'b89', 'b90']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Our claims are mainly presented in the Abstract, which are also summarized as three-fold contributions in Sec. 1.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss the limitations in Section 6.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Details of RidgeLoRA are proposed in Sec. 3, followed by theoretical analysis (extensive proof and theories can also be found in Appendix A).
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The hyper parameter setups are as disclosed in Section 4.1.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
this section cite: []

Section: 
Answer: [Yes] Justification: We use open-source datasets for evaluation and provide links to them in the appendix. Code is attached with the supplementary materials. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: The hyper parameter setups are as disclosed in Section 4.1, which are consistent with the Code affiliated as supplementary materials.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We include error bars in Figure 2 when conducting pattern analysis. We also include a hypothesis testing in Appendix B.1, which serves as a strong indication of the significance of adding squashing loss. Following multiple previous LoRA variants, we report accuracies in our main experiments. Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: The resources consumed by different variants are included in Appendix B.3. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: This research is in compliance with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This research adopts only publicly available datasets and LLMs.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA]
Justification: This research poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: Following major previous works, all open-source datasets and LLMs included in this research are properly cited and discussed.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes] Justification: Code is attached to supplementary materials. Guidance is also included in the repository.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA]
Justification: This research does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This research does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: LLMs are only adopted for grammar checking at sentence level. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Gemini: a family of highly capable multimodal models Year: (2023)
Ref_id:b2 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b3 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b4 Title: Mistral 7b Year: (2023)
Ref_id:b5 Title: The llama 3 herd of models Year: (2024)
Ref_id:b6 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b7 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b8 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b9 Title: Parameter-efficient fine-tuning of large-scale pre-trained language models Year: (2023)
Ref_id:b10 Title: Parameter-efficient finetuning for large models: A comprehensive survey Year: (2024)
Ref_id:b11 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b12 Title: Parameter-efficient fine-tuning methods for pretrained language models: A critical review and assessment Year: (2023)
Ref_id:b13 Title: LoRA learns less and forgets less Year: (2024)
Ref_id:b14 Title: Lora vs full fine-tuning: An illusion of equivalence Year: (2024)
Ref_id:b15 Title: Weight-decomposed low-rank adaptation Year: (2024-07)
Ref_id:b16 Title: PiSSA: Principal singular values and singular vectors adaptation of large language models Year: (2024)
Ref_id:b17 Title: KaSA: Knowledgeaware singular-value adaptation of large language models Year: (2025)
Ref_id:b18 Title: Lora-pro: Are low-rank adapters properly optimized? arXiv preprint Year: (2024)
Ref_id:b19 Title: LoRA+: Efficient low rank adaptation of large models Year: (2024)
Ref_id:b20 Title: Flat-loRA: Low-rank adaptation over a flat loss landscape Year: (2025)
Ref_id:b21 Title: The matrix ridge approximation: algorithms and applications Year: (2014)
Ref_id:b22 Title: Singular value decomposition and least squares solutions Year: (1971)
Ref_id:b23 Title: Svd-llm: Truncation-aware singular value decomposition for large language model compression Year: (2024)
Ref_id:b24 Title: DRONE: Data-aware low-rank compression for large NLP models Year: (2021)
Ref_id:b25 Title: Modegpt: Modular decomposition for large language model compression Year: (2024)
Ref_id:b26 Title: Revisiting the nystrom method for improved large-scale machine learning Year: (2013-06)
Ref_id:b27 Title: Fast monte carlo algorithms for matrices i: Approximating matrix multiplication Year: (2006)
Ref_id:b28 Title: Note sur une méthode de résolution des équations normales provenant de l'application de la méthode des moindres carrés à un système d'équations linéaires en nombre inférieur à celui des inconnues (procédé du commandant cholesky) Year: (1924)
Ref_id:b29 Title: LoRA-GA: Low-rank adaptation with gradient approximation Year: (2024)
Ref_id:b30 Title: Harnessing minor singular components for parameter-efficient llm finetuning Year: (2024)
Ref_id:b31 Title: Lora-xs: Low-rank adaptation with extremely small number of parameters Year: (2024)
Ref_id:b32 Title: HiRA: Parameter-efficient hadamard high-rank adaptation for large language models Year: (2025)
Ref_id:b33 Title: A rank stabilization scaling factor for fine-tuning with lora Year: (2023)
Ref_id:b34 Title: Vector-based random matrix adaptation Year: (2024)
Ref_id:b35 Title: Parameter-efficient fine-tuning for large models: A comprehensive survey Year: (2024)
Ref_id:b36 Title: BitFit: Simple parameter-efficient finetuning for transformer-based masked language-models Year: (2022-05)
Ref_id:b37 Title: Prefix-tuning: Optimizing continuous prompts for generation Year: (2021-08)
Ref_id:b38 Title: P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks Year: (2021)
Ref_id:b39 Title:  Year: (2024)
Ref_id:b40 Title: Parameter-efficient transfer learning for nlp Year: (2019)
Ref_id:b41 Title: Towards a unified view of parameter-efficient transfer learning Year: (2022)
Ref_id:b42 Title: The approximation of one matrix by another of lower rank Year: (1936)
Ref_id:b43 Title: Neural networks: Tricks of the trade Year: (2002)
Ref_id:b44 Title: Understanding the difficulty of training deep feedforward neural networks Year: (2010-05)
Ref_id:b45 Title: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification Year: (2015)
Ref_id:b46 Title: Maximum properties and inequalities for the eigenvalues of completely continuous operators Year: (1951)
Ref_id:b47 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b48 Title: The probable error of a mean Year: (1908)
Ref_id:b49 Title: Federated continual learning via prompt-based dual knowledge transfer Year: (2024)
Ref_id:b50 Title: Sd-lora: Scalable decoupled low-rank adaptation for class incremental learning Year: (2025)
Ref_id:b51 Title: A 3 e: Towards compositional model editing Year: (2025)
Ref_id:b52 Title: Matrix computations Year: (2013)
Ref_id:b53 Title: Learning representations by back-propagating errors Year: (1986)
Ref_id:b54 Title: The generalization of 'student's'problem when several different population varlances are involved Year: (1947)
Ref_id:b55 Title: An approximate distribution of estimates of variance components Year: (1946)
Ref_id:b56 Title: BERT: Pre-training of deep bidirectional transformers for language understanding Year: (2019-06)
Ref_id:b57 Title: GLUE: A multi-task benchmark and analysis platform for natural language understanding Year: (2019)
Ref_id:b58 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b59 Title: Deberta: Decoding-enhanced bert with disentangled attention Year: (2021)
Ref_id:b60 Title: Attention is all you need Year: (2017)
Ref_id:b61 Title: GQA: Training generalized multi-query transformer models from multi-head checkpoints Year: (2023-12)
Ref_id:b62 Title: Judging LLM-as-a-judge with MT-bench and chatbot arena Year: (2023)
Ref_id:b63 Title: Learning transferable visual models from natural language supervision Year: (2021-07)
Ref_id:b64 Title: Visual instruction tuning Year: (2023)
Ref_id:b65 Title: BoolQ: Exploring the surprising difficulty of natural yes/no questions Year: (2019-06)
Ref_id:b66 Title: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b67 Title: Social IQa: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b68 Title: HellaSwag: Can a machine really finish your sentence? Year: (2019-07)
Ref_id:b69 Title: Winogrande: an adversarial winograd schema challenge at scale Year: (2021-08)
Ref_id:b70 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b71 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018-11)
Ref_id:b72 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b73 Title: Measuring mathematical problem solving with the MATH dataset Year: (2021)
Ref_id:b74 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b75 Title: Program synthesis with large language models Year: (2021)
Ref_id:b76 Title: Metamath: Bootstrap your own mathematical questions for large language models Year: (2024)
Ref_id:b77 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b78 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b79 Title: Towards vqa models that can read Year: (2019)
Ref_id:b80 Title: Evaluating object hallucination in large vision-language models Year: (2023-12)
Ref_id:b81 Title: Recursive deep models for semantic compositionality over a sentiment treebank Year: (2013-10)
Ref_id:b82 Title: Neural network acceptability judgments Year: (2019)
Ref_id:b83 Title: Automatically constructing a corpus of sentential paraphrases Year: (2005)
Ref_id:b84 Title: SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation Year: (2017-08)
Ref_id:b85 Title: A broad-coverage challenge corpus for sentence understanding through inference Year: (2018-06)
Ref_id:b86 Title: SQuAD: 100,000+ questions for machine comprehension of text Year: (2016-11)
Ref_id:b87 Title: The pascal recognising textual entailment challenge Year: (2006)
Ref_id:b88 Title: The second pascal recognising textual entailment challenge Year: (2006)
Ref_id:b89 Title: The third PASCAL recognizing textual entailment challenge Year: (2007-06)
Ref_id:b90 Title: The fifth pascal recognizing textual entailment challenge Year: (2009)
Ref_id:b91 Title: Method Rank Total Parameters Trainable Parameters Weight Memory/MB Computation/FLOPs Initialization Year: ()
Ref_id:b92 Title:  Year: (7914)
Ref_id:b93 Title:  Year: ()
Ref_id:b94 Title:  Year: ()
Ref_id:b95 Title:  Year: (2561)
Ref_id:b96 Title:  Year: (1988)
Ref_id:b97 Title:  Year: (1987)
Ref_id:b98 Title:  Year: ()
Ref_id:b99 Title:  Year: ()
