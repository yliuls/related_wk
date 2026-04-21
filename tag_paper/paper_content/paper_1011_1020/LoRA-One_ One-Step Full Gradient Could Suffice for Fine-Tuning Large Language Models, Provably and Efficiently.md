Title: LoRA-One: One-Step Full Gradient Could Suffice for Fine-Tuning Large Language Models, Provably and Efficiently
Abstract: This paper explores how theory can guide and enhance practical algorithms, using Low-Rank Adaptation (LoRA) (Hu et al., 2022) in large language models as a case study. We rigorously prove that, under gradient descent, LoRA adapters align with specific singular subspaces of the onestep full fine-tuning gradient. This result suggests that, by properly initializing the adapters using the one-step full gradient, subspace alignment can be achieved immediately-applicable to both linear and nonlinear models. Building on our theory, we propose a theory-driven algorithm, LoRA-One, where the linear convergence (as well as generalization) is built and incorporating preconditioners theoretically helps mitigate the effects of ill-conditioning. Besides, our theory reveals connections between LoRA-One and other gradient-alignment-based methods, helping to clarify misconceptions in the design of such algorithms. LoRA-One achieves significant empirical improvements over LoRA and its variants across benchmarks in natural language understanding, mathematical reasoning, and code generation. Code is available at: https://github.  com/YuanheZ/LoRA-One.

Section: Introduction
How to efficiently approximate or learn nonlinear models is a central question in large-scale machine learning, especially in the era of large language models (LLMs) (Brown et al., 2020;Thoppilan et al., 2022). Fine-tuning (Dodge et al., 2020) aims to make LLMs perform well on new tasks Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
while retain the knowledge from pre-trained models. For scalability, we expect that fine-tuning can be conducted with low computation/memory cost, i.e., parameter-efficient fine-tuning (PEFT) (Houlsby et al., 2019;Han et al., 2024).
One typical PEFT strategy is Low-Rank Adaptation (LoRA) (Hu et al., 2022), which learns an approximation of the unknown feature shift ∆ by two low-rank matrices A and B with rank r, i.e. ∆ ≈ AB under the following initialization (denoted by index 0):
[A 0 ] ij ∼ N (0, α 2 ) [B 0 ] ij = 0 , α > 0 . (LoRA-init)
To improve the performance in the downstream tasks, various LoRA-based algorithms have been proposed based on, e.g., refined initialization (Li et al., 2025), learning rates (Hayou et al., 2024), efficiency (Kopiczko et al., 2024), and gradient information (Meng et al., 2024;Wang et al., 2024).
Although LoRA is conceptually simple, its optimization dynamics are inherently nonlinear and non-convex. There is few theoretical understanding of its behavior, e.g., optimization from lazy-training regime (Jang et al., 2024;Malladi et al., 2023;Liu et al., 2025) to non-lazy training regime (Kim et al., 2025) and generalization guarantees in some simplified settings (Dayi & Chen, 2024). It still remains unclear how (low-rank) gradient updates in LoRA evolve and which subspaces LoRA will converge to. More importantly, given the application-driven nature of LoRA, a rigorous theoretical understanding should not only explain its behavior but also inform practical algorithm design. The goal of this work is to enhance LoRA's empirical performance through theoretically grounded insights. To this end, we address two key questions at the intersection of theory and practice:
• Q1: How to characterize low-rank dynamics of LoRA and the associated subspace alignment in theory?
• Q2: How can our theoretical results contribute to algorithm design for LoRA in practice?
this section cite: ['b4', 'b46', 'b14', 'b21', 'b16', 'b22', 'b30', 'b17', 'b27', 'b36', 'b52', 'b24', 'b35', 'b32', 'b26', 'b12']

Section: Contributions
In this work, we theoretically investigate the behavior of gradient descent (GD) update of LoRA parameters (A t , B t ) and identify the subspaces they align with. Our theory identifies the optimal initialization strategies in the perspective The mean principle angles within each layer class is estimated for the alignment of fine-tuning a T5 base model (Raffel et al., 2020) on MRPC using LoRA. The principal angle measures the distance between the projection matrices of the top-r (LoRA rank) singular subspace of the LoRA matrices and that of the one-step gradient. Note that the principal angles are approximately 1 at initialization and then decrease for alignment. (c) Comparison of trajectories among full fine-tuning (Full FT), LoRA, and LoRA-One under gradient descent. We use a two-layer neural network pretrained on odd-labeled data and then fine-tune on even-labeled MNIST data. Experimental details are presented in Appendix G.1.
of subspace alignment, and we find that it also performs well on some real-world datasets. We term this initialization as spectral initialization, which leverages the information of one-step full gradient, leading to the theoretical grounded algorithm, LoRA-One. This algorithm incorporated into several architectures achieves promising performance on natural language processing (NLP), reasoning tasks when compared to LoRA and its variants. Our contributions from theory (see Table 1 for
of G ♮ = U G ♮ S G ♮ V ⊤ G ♮ .
The alignment can be directly achieved at the certain initialization strategy, termed as spectral initialization
A 0 = √ γ U G ♮ [:,1:r] S 1/2 G ♮ [1:r] , B 0 = √ γ S 1/2 G ♮ [1:r] V G ♮ ⊤ [:,1:r] , (Spectral-init)
where γ is a tuning parameter. By (Spectral-init), we theoretically ensure that ∥A 0 B 0 -∆∥ F is sufficiently small at beginning, see the theoretical results for linear models in Section 3.2 and nonlinear models in Section 4, respectively.
It demonstrates the sufficiency of using one-step full gradient, which can be numerically verified on several real-world benchmarks, serving as the algorithm principle.
ii) Global convergence and generalization guarantees: Under spectral initialization, continuing gradient descent (GD) updates for (A t , B t ), we further establish the linear convergence rate of ∥A t B t -∆∥ F for both linear and nonlinear models. This linear rate, however, is sensitive to the condition number κ(∆) of ∆, leading to unsatisfactory convergence performance if ∆ is ill-conditioned. To address this issue, we rigorously show that adding preconditioners into the GD update eliminates the dependence on the condition number; see Section 3.2 and Section 4, respectively.
Moreover, our theory aims to clarify certain misunderstandings in prior algorithm designs. Specifically, it identifies the correct subspace for alignment and highlights potential limitations of previous LoRA variants based on gradient alignment-such as LoRA-GA (Wang et al., 2024); see the discussion in Section 5.
iii) Performance improvement in numerical and readworld datasets: Guided by our theory, the spectral initialization strategy (Spectral-init) leads to our theoretically grounded algorithm, LoRA-One. As shown in Fig. 1(c), our numerical results demonstrate that LoRA-One's trajectory is close to the full fine-tuning (Full FT) and obtain lower loss than LoRA.
We conducted experimental comparisons between LoRA-One and standard LoRA-based algorithms across various NLP benchmarks, including natural language understanding (NLU), mathematical reasoning, and code generation tasks. For instance, using only (Spectral-init), it takes just one second on some NLU tasks to achieve performance comparable to LoRA which requires tens of seconds. On the HumanEval benchmark, LLaMA 2-7B fine-tuned with LoRA-One achieves a score of 28.66, outperforming stan- dard LoRA (25.85) by 2.81, while maintaining almost the same time and memory costs.
Notations For a matrix A, let ∥A∥ op denote its operator and ∥A∥ F its Frobenius norm. Let ⊙ denote the Hadamard (i.e., entrywise) matrix product. We use I n to denote the R n×n -valued identity matrix. The notation U A denotes the left singular matrix of the compact SVD of A and U A,⊥ denotes the corresponding orthogonal complement. Similarly, V A denotes the right singular matrix of A and V A,⊥ denotes its orthogonal complement. Let U r * (A) denote the left singular subspace spanned by the r * largest singular values of A and U r * ,⊥ (A) denote the left singular subspace orthogonal to U r * (A). Similarly define V r * (A) and V r * ,⊥ (A) for the right singular subspace. A complete list notations can be found in Table 5 of Appendix A.
this section cite: ['b42', 'b52']

Section: Related Work
Parameter-Efficient Fine-Tuning (PEFT): LoRA (Hu et al., 2022) and its variants have received great attention for downstream applications. The variants of LoRA focus on imbalance stepsize (Hayou et al., 2024), initialization using SVD of pre-trained weights (Meng et al., 2024), gradient approximation (Wang et al., 2024;2025) for better performance, reducing parameters (Kopiczko et al., 2024) efficiency, preconditioned algorithm (Zhang & Pilanci, 2024) for stability.
In theory, the training dynamics and generalization ability of LoRA are rarely discovered. Based on the empirical evidence of kernel behavior of LoRA in Malladi et al. (2023), the global convergence is given by Jang et al. (2024) for LoRA with rank O( √ N ) under the lazy training (Jacot et al., 2018) as well as Liu et al. (2025) on PL* condition. Beyond the lazy training regime, Kim et al. (2025) study the loss landscape of LoRA as well as its implicit bias. For generalization, Dayi & Chen (2024) derive the sample/time complexity by exploring the SGD dynamics of rank-1 LoRA, related to single-index model (Arous et al., 2021). In our work, we study the dynamics of LoRA from the perspective of subspace alignment, which has some overlap with matrix sensing as below.
Matrix Sensing under Gradient Descent: Since LoRA performs fine-tuning using a Burer-Monterio factorization, it admits similarities with matrix sensing problems, including the symmetric matrix problem with r = r * (Li et al., 2018) and r ≥ r * (Stöger & Soltanolkotabi, 2021); asymmetric problem with r ≥ r * (Soltanolkotabi et al., 2023;Xiong et al., 2024). Regarding initialization, small initialization (Ding et al., 2022) and spectral initialization (Ma et al., 2021) help convergence with theoretical guarantees, which is applied to LoRA under certain specific settings (Xu et al., 2025). Besides, adding preconditioner (Zhang et al., 2021;Tong et al., 2021;Xu et al., 2023;Zhang et al., 2023;Giampouras et al., 2024;Zhu et al., 2024) is beneficial to solve the problem of ill-conditioned ground truth matrix.
Technically, for the alignment part, our theory leverages some techniques from Soltanolkotabi et al. (2023). However, the symmetrization technique used in prior work cannot be applied to decouple the GD dynamics of (A t , B t ), posing a challenge in analyzing their individual spectral behaviors. To overcome this limitation, we develop a novel approach that enables a detailed analysis of the distinct spectral dynamics of A t and B t , which is one technical contribution of this work. In fact, one-step gradient information has been used in deep learning theory, demonstrating that it allows for feature learning under different stepsizes (Ba et al., 2022;Moniri et al., 2024;Cui et al., 2024;Dandi et al., 2025). Besides, for the nonlinear model part, dynamical analysis are normally based on classical gradient-based algorithm (Damian et al., 2022;Lee et al., 2024) for feature learning. Nevertheless, how such model behaves under low-rank updates under (Spectral-init) is still unclear to our knowledge.
this section cite: ['b22', 'b17', 'b36', 'b52', 'b27', 'b61', 'b35', 'b24', 'b23', 'b32', 'b26', 'b12', 'b1', 'b31', 'b44', 'b43', 'b56', 'b13', 'b34', 'b58', 'b63', 'b47', 'b57', 'b62', 'b15', 'b65', 'b43', 'b2', 'b38', 'b9', 'b11', 'b10', 'b29']

Section: Problem Settings
In this section, we introduce the problem setting of finetuning pre-trained linear and nonlinear models with the following assumptions for our theory.
this section cite: []

Section: Basic Assumptions
We consider both linear and nonlinear pre-trained models with multiple outputs and thus matrix parameters (instead of vectors), which is consistent with LoRA in practice.
Assumption 2.1 (Pre-trained model). For the input x ∈ R d , we denote by W ♮ ∈ R d×k the known pre-trained parameter matrix. We assume that the pre-trained model can be linear or nonlinear with σ(•) = max{0, • } being the (entry-wise) ReLU activation function.
f pre (x) := (x ⊤ W ♮ ) ⊤ ∈ R k linear σ[(x ⊤ W ♮ ) ⊤ ] ∈ R k nonlinear .
Note that our results can handle large dimension d and k.
For fine-tuning, we assume there exists an unknown lowrank feature shift ∆ on W ♮ that we aim to estimate.
Assumption 2.2. The downstream feature matrix W ♮ := W ♮ + ∆ admits an unknown low-rank feature shift ∆ ∈ R d×k , where Rank (∆) = r * < min{d , k}.
This assumption is widely used in the literature on LoRA analysis and matrix factorization (Zhang et al., 2021;Stöger & Soltanolkotabi, 2021;Soltanolkotabi et al., 2023;Xiong et al., 2024). Next we assume the following data generation process, i.e., label-noiseless and well-behaved data.
Assumption 2.3 (Data generation process for fine-tuning).
Given the unknown W ♮ , the label y is generated by
y := ( x ⊤ W ♮ ) ⊤ ∈ R k , { x i } N i=1 i.i.d. ∼ SG, linear σ[( x ⊤ W ♮ ) ⊤ ], { x i } N i=1 i.i.d. ∼ N (0, I d ) nonlinear ,
where SG denotes the probability distribution for isotropic centered sub-Gaussian random vectors. We assume that we have N i.i.d training data { x i , y i } N i=1 for fine-tuning.
Note that the nonlinear model can be regarded as a special case of multi-index model (Damian et al., 2022;Abbe et al., 2022;Bietti et al., 2023) and Gaussian data is a common assumption in the analysis of single/multi-index models (Damian et al., 2022;Lee et al., 2024;Oko et al., 2024). We additionally assume that d < N , which coincides with practical settings of LoRA for LLaMA 2-7b (Touvron et al., 2023) on real-world datasets, e.g., MetaMathQA (Yu et al., 2024) and Code-Feedback (Zheng et al., 2024), where d = 128 ∼ 4096 and N is on the order of 10 5 .
this section cite: ['b63', 'b44', 'b43', 'b56', 'b10', 'b0', 'b3', 'b10', 'b29', 'b40', 'b48', 'b59', 'b64']

Section: Full Fine-tuning and LoRA
Our goal is to efficiently recover ∆ by fine-tuning on the downstream data. Let the complete SVD of ∆ ∈ R d×k be
∆ = U S * V ⊤ := U U ⊥ S * 0 0 0 V ⊤ V ⊤ ⊥ ,(1)
where U ∈ R d×d and V ∈ R k×k are the left and right singular matrices, and S * ∈ R d×k is a rank-r * diagonal matrix with nonzero singular values
{λ * i } r * i=1 . It admits the compact SVD ∆ = U S * V ⊤ with U ∈ R d×r * , V ⊤ ∈ R r * ×k , and S * ∈ R r * ×r *
. The left/right singular subspaces spanned by U and V play an important role in our analysis.
We write the downstream data in a compact form X = [ x 1 , • • • , x N ] ⊤ ∈ R N ×d and the label matrix Y = [ y 1 • • • y N ] ⊤ ∈ R N ×k is generated by either linear or nonlinear target functions in Assumption 2.3. We introduce the training based on full fine-tuning and LoRA below.
Full Fine-tuning: We consider the following empirical risk minimization with a squared loss
L(W ) := 1 2N      XW -Y 2 F linear, σ( XW ) -Y 2 F nonlinear ,(2)
where the parameter W can be learned by gradient descent (GD) initialized at W ♮ , i.e., W 0 := W ♮ .
LoRA: It updates two low-rank matrices A ∈ R d×r , B ∈ R r×k for efficiency with the following empirical risk
L (A , B) := 1 2N      X(W ♮ +AB)-Y 2 F , linear, σ X(W ♮ +AB) -Y 2 F , nonlinear(3)
which can be minimized using GD with stepsize η > 0
A t+1 = A t -η∇ A L (A t , B t ) , B t+1 = B t -η∇ B L (A t , B t ) .(4)
Since the true rank r * of ∆ is unknown in LoRA, our results will cover two cases: over-ranked (r ≥ r * ) and exact-ranked (r = r * ). 1 Our results allow for large d, k while r, r * = Θ(1), which coincides with common practice.
this section cite: []

Section: Optimization and Generalization:
We are interested in the error ∥A t B t -∆∥ 2 F under the LoRA training dynamics. Bounds on this error also imply generalization performance, because the generalization error for a new data (
x, y) sat- isfies E x y -σ(W ♮ + A t B t ) ⊤ x 2 2 ≤ ∥A t B t -∆∥ 2 F
in the nonlinear setting, with equality in the linear setting.
this section cite: []

Section: Analysis of LoRA under Linear Model
In this section, we establish the alignment between LoRA and one gradient of full fine-tuning. This result guides us to design new strategies for speeding up practical LoRA-based algorithms, which achieve this alignment at initialization.
We formally define the negative gradient of full fine-tuning in Eq. ( 2) for the linear setting after the first step as
G ♮ := -∇ W L(W ♮ ) = 1 N X ⊤ ( Y -XW ♮ ) . (5)
Note that X ⊤ X is a non-singular square matrix (Zeng & Lee, 2024, Lemma 6). Since left multiplication by a non-singular square matrix does not change the rank (Horn & Johnson, 2012, 0.4.6 (b)), we have Rank(G ♮ ) = Rank(∆) = r * . Then, we denote the singular values of G ♮ by {λ i (G ♮ )} r * i=1 in non-increasing order.
this section cite: ['b20']

Section: Alignment under LoRA Initialization
We first present the results for the alignment of B t by recalling the notations V r * (•) and V r * ,⊥ (•).
Theorem 3.1 (Alignment between G ♮ and B t ). Under assumptions in Section 2.1 for the linear setting, consider the LoRA updates (4) with (LoRA-init). We have
V ⊤ r * ,⊥ G ♮ V r * (B t ) op = 0 , ∀t ∈ N + .
One can see that, due to the zero initialization of B 0 in (LoRA-init), after the first GD step, it holds that B 1 = ηA ⊤ 0 G ♮ , which has rank ≤ r * and lies in the right top-r * singular subspace of G ♮ . The subsequent GD dynamics of B t is always restricted to this invariant subspace.
Next we build the alignment for A t with the notations U r * (•), U r * ,⊥ (•) and κ ♮ as the condition number of G ♮ .
this section cite: []

Section: Theorem 3.2 (Alignment between G ♮ and A t . Simplified version of Theorem C.9).
For the r ≥ 2r * case, under assumptions in Section 2.1 for the linear setting, we consider the LoRA updates (4) with [A 0 ] ij ∼ N (0, α 2 ) in (LoRA-init). Then for any constant θ ∈ (0, 1), by taking
α = O θ 3 2 κ ♮ d -3 4 κ ♮ -1 2 ∥G ♮ ∥ 1 2
op , and running gradient descent for t * steps with
t * ≲ ln √ d θ ln (1 + ηλ r * (G ♮ )) ,(6)
we achieve the following the alignment on the left singular subspace between G ♮ and A t * as below
U ⊤ r * ,⊥ (G ♮ ) U r * (A t * ) op ≲ θ ,(7)
with probability at least 1 -C 1 exp(-d) -C 2 exp(-r) -C 3 exp(-N ) for some constants C 1 , C 2 , C 3 .
this section cite: []

Section: Remark:
The result under the r * ≤ r < 2r * case is more complex and we defer this result to Theorem C.9. The choice of α in Theorem 3.2 shows that after t * = Θ
ln d λ r * (G ♮ )
in Eq. ( 6), the alignment can be achieved. Our results can cover the standard Heinitialization (He et al., 2015) if ∥G ♮ ∥ op ≥ Ω(d 3 4 κ ♮ ).
min t ∥U ⊤ r * ,⊥ (G ♮ ) U r * (A t ) ∥ op . More experimental details can be found in Appendix G.1.
Requirement on ∥G ♮ ∥ op can be relaxed under smaller initialization, illustrated by Fig. 2.
The above two theorems characterize the alignment between G ♮ and (A t , B t ). Fig. 2 empirically validates Theorem 3.2 in two folds: i) Smaller initialization (α 2 in the x-axis) encourages better alignment (evaluated by the principal angle), and then better generalization performance of fine-tuning (evaluated by the risk). But in practice smaller initialization would increase the training time for convergence, as a double-edge sword. ii) increasing d leads to longer alignment time, illustrated by Eq. ( 6), and worse alignment performance, illustrated by the formulation of α. Besides, we also verify this alignment in read-world applications by finetuning a T5 base model (Raffel et al., 2020) on MRPC using LoRA, as shown in Fig. 1(b). The mean principle angles within each layer class is computed in a similar way of Fig. 2. Our empirical results demonstrate that the principal angles decrease from around 1 because of Gaussian initialization to the value around 0.2 ∼ 0.4.
this section cite: ['b18', 'b42']

Section: Proof of Sketch:
Here we give a proof of sketch of Theorem 3.2. The dynamics (4) can be written as
A t+1 B ⊤ t+1 =:Zt+1 = I d ηG ♮ ηG ♮ ⊤ I k =:H A t B ⊤ t =:Zt + nonlinear term ,
where H is a time-independent matrix corresponding to the linear part of the dynamic Z lin t := H t Z 0 . By Schur decomposition of H (see Lemma C.1), we can obtain the precise spectral dynamics of Z lin t and derive the alignment between Z lin t and G ♮ , see Lemma C.5 for details. We prove that the nonlinear term is well controlled, i.e., ∥Z t -Z lin t ∥ op ≤ ∥A 0 ∥ op , ∀t ≤ t * , see Lemma C.6. Then
0.4 0.8 1.2 1.6 a 1 1.8 1.2 0.6 0.0 0.6 a2 0.3 0.6 0.9 1.2 1.5 b LoRA init Spectral Init Global Minimizers LoRA init Start LoRA init End Spec. Start Spec. End 0.8 0.0 0.8 1.6 a 1 0.4 0.8 1.2 1.6 a2 0.0 0.4 0.8 1.2 b the alignment between Z lin t and G ♮ can be successfully transferred to that of Z t , see Theorem C.9 for details.
We remark that previous work on matrix sensing (Stöger & Soltanolkotabi, 2021;Soltanolkotabi et al., 2023) via a symmetrization technique cannot be directly applied to our setting. Such symmetrization technique prevents the alignment results decoupling into two factorized matrices. We extend their technique to decouple the alignment for A t and B t individually via Schur decomposition of H.
this section cite: ['b44', 'b43']

Section: Spectral Initialization and Global Convergence
Theorem 3.2 has demonstrated the alignment on the rankr * singular space of G ♮ and (A t , B t ). In other words, if we take the SVD of G ♮ and choose the certain singular subspace for initialization in (Spectral-init), we can directly achieve the alignment at this initialization and recover ∆ to some extent, which is the main target of this work.
By the following standard concentration result for (sub)-Gaussian data: with probability at least 1 -2C exp(-ϵ 2 N ) for some constants C > 0, we have
Σ -I d op ≤ ϵ := min 1 2κ , c κ 3 ≤ 1 2 . (8
)
Recall κ is the condition number of ∆ and λ * r * is the r * -th singular value of ∆, we have the following result at the spectral initialization.
Theorem 3.3. [One-step gradient can suffice] Under assumptions in Section 2.1 for the linear setting via (Spectral-init), taking ϵ in Eq. (8), then with probability at least 1 -2C exp(-ϵ 2 N ) for constant C > 0, we have
∥A 0 B 0 -∆∥ op ≤ ϵ∥∆∥ op ≤ λ * r * 2 .
Theorem 3.3 demonstrates that, after one-step full gradient, i.e., using spectral initialization (Spectral-init), A 0 B 0 is able to recover ∆ with small error. This is still true for nonlinear models, see Lemma D.5 for details. Fig. 3 numerically validates that the starting points initialized by (Spectral-init) are consistently closer to the set of global minimizers, whereas those initialized by (LoRA-init) tend to be farther away across different random seeds.
Moreover, running gradient descent from points initialized by (Spectral-init) requires significantly fewer steps to reach a global minimizer, demonstrating the advantages of (Spectral-init). Due to page limit, we present the global convergence of linear models is deferred to Appendix C.2 and Appendix C.3, respectively. In Appendix C.2, we derive the linear convergence rate ∥A t B t -∆∥ F . Since the convergence will be slow if the downstream feature shift ∆ is ill-conditioned, i.e., κ is large. This motivates us to add preconditioners, then the convergence rate will be independent of κ accordingly, see Appendix C.3 for details.
this section cite: []

Section: Analysis of LoRA under Nonlinear Models
Now we focus on the nonlinear setting described in Section 2, where we consider the exact-rank case r = r * for delivery. We will demonstrate that ∥A 0 B 0 -∆∥ F is still small under the spectral initialization. Besides, the linear convergence rate of ∥A t B t -∆∥ F can still hold.
As an example, we demonstrate the equipment of precondition GD on (A t , B t ) for global convergence
A t+1 = A t -η∇ A L (A t , B t ) B t B ⊤ t -1 , B t+1 = B t -η A ⊤ t A t -1 ∇ B L (A t , B t ) .(9)
Notice that here we use standard matrix inversion since we can prove that A t and B t stay non-singular across all t ≥ 0. By denoting W t := W ♮ + A t B t , we have the gradient
∇ A L (A t , B t ) = -J Wt B ⊤ t , ∇ B L (A t , B t ) = -A ⊤ t J Wt ,
where we denote
J Wt := 1 N X ⊤ σ( X W ♮ ) -σ( XW t ) ⊙ σ ′ ( XW t ) .
To deliver the proof, apart from the above-mentioned assumptions in Section 2.1 for the the nonlinear setting, we also need the following assumption.
Assumption 4.1. We assume that i)
∥ W ♮ ∥op ∥ w ♮ m ∥2 = O (1); ii) max{λ * r * ,∥∆m∥ op } ∥ w ♮ m ∥2 = O 1 κr * for m ∈ [k].
Remark: The condition i) ensures the balance between different neurons within one layer for the downstream teacher model and the task diversity. The condition ii) ensures the signal of downstream feature shift is smaller than the pretrained ones approximately in the order (κr * ) -1 since the signal of adapted weight is generally weaker than the pretrained weight. Two conditions can be empirically observed in Appendix G.5.
Here we can show that, for the nonlinear model, LoRA training can achieve global linear convergence under (Spectral-init) via preconditioned GD in Eq. ( 9).
Theorem 4.2 (Simplified version of Theorem D.10). Under assumptions in Section 2.1 for the nonlinear setting and 4.1, with training conducted by Eq. (9) and initialization via (Spectral-init) with setting γ = 2, we take ϵ = O 1 r * κ √ d
and ρ ≤ 1 20 . Then choosing η ∈ (c η , 1) for a small constant c η > 0, with probability at least 1 -2Cdk exp -ϵ 2 N for a universal constant C > 0, we have
∥A t B t -∆∥ F ≤ 1 - η 4 t ρλ * r * , ∀t ≥ 0 .(10)
Remark: We make three remarks here: i) This theorem is based on ∥A 0 B 0 -∆∥ F ≤ ρλ * r * at initialization, see Lemma D.5 for details, which demonstrates that one-step full gradient can be sufficient.
ii) The convergence rate is independent of condition number κ of downstream feature shift ∆, demonstrating the benefits of adding preconditioners.
Proof of Sketch The complete proof can be found in Appendix D.2. We first compute the expectation of J Wt (see Lemma D.2) and decompose J Wt into 1 2 (A t B t -∆)+Ξ t , where Ξ t is defined in Lemma D.6. The first term is the signal term which can dominate the preconditioned GD dynamics. The second term Ξ t := T 1 + T 2 consists of two parts (details see Lemma D.7): the first part T 1 is the residual term from E x [J Wt ] which vanishes due to pre-training signal dominance. For the second term T 2, it comes from the concentration error of J Wt , which can also controlled by large sample size N .
To handle ∥A t B t -∆∥ F , we explore its recursion relationship in Lemma D.6. The key part is to control
I d -U At U ⊤ At ∆ I k -V Bt V ⊤
Bt F (Lemma D.9) and higher order term (Lemma D.8).
this section cite: []

Section: Algorithm and Discussions
In this section, we present the LoRA-One algorithm and justify the optimality of our initialization over previous gradient alignment based algorithms for fine-tuning.
We present the implementations in Algorithm 1, which is driven by (Spectral-init) (shown in line 3-6). It coincides with the spirit of gradient alignment work, e.g., LoRA-GA (Wang et al., 2024), LoRA-pro (Wang et al., 2025), but the mechanisms for gradient alignment differ significantly, as suggested by our theory. First, LoRA-GA proposes the fol-Algorithm 1 LoRA-One for one specific layer Input: Pre-trained weight W ♮ , batched data {D m } T m=1 , sampled batch data B, LoRA rank r, LoRA alpha α, loss function L, scaling parameter s Initialize:
1
: Compute ∇ W L(W ♮ ) given B 2: U , S, V ← SVD -∇ W L(W ♮ ) 3: S ← S/S [0,0] and γ ← 1/s 4: A 0 ← √ γ • U [:,1:r] S 1/2 [:r,:r] 5: B 0 ← √ γ • S 1/2 [:r,:r] V ⊤ [:,1:r] 6: Clear ∇ W L(W ♮ ) Train:
7: for t = 0 , ... , T -1 do 8:
Compute gradients given D t+1 : lowing initialization strategy (omit the scaling parameters)
G A t+1 ← ∇ A L (A t ,B t ) , G B t+1 ← ∇ B L (A t ,B t ) 9: Update A t+1 , B t+1 ← AdamW G A t+1 , G B
A 0 ← -U G ♮ [:,1:r] , B 0 ← V G ♮ ⊤ [:,r+1:2r]
, which aims to provide the best 2r approximation of G ♮ . However, our theory indicates that B t will align to the right-side rank-r * singular subspace of G ♮ under random initialization. However, LoRA-GA chooses the (r + 1)-th to 2r-th singular values for B 0 , causing the iterates B t to lie outside the desired subspace. As a result, the optimization may remain trapped in an undesirable subspace and fail to converge to an optimal solution, which can numerically verified by Fig. 4. Moreover, this approach subtracts the gradient for non-zero initialization and thus yields a biased estimate of ∆, scaling with the model size; see further discussion in Appendix F.
Secondly, there is one concurrent work (Ponkshe et al., 2024), LoRA-SB, which uses the same singular subspace for initialization but only updates r × r matrix R from the SVD of G ♮ . Their intuition is to project the fine-tuning updates onto the singular subspace of first gradient step G ♮ . However, the singular subspace of G ♮ normally still have distances with the ground truth, which will make their method hard to escape/rotate this subspace with limited degrees of freedom.
Our toy experiments based on (3) in Fig. 4 show that both LoRA-GA and LoRA-SB fail to find global minimizers even in a simple linear setting, whereas LoRA-One demonstrates significantly better generalization. More experimental details are presented in Appendix G.1.
this section cite: ['b52', 'b53', 'b41']

Section: Experiments
In this section, we conduct experiments to compare LoRA-One with typical LoRA based algorithms across multiple NLP benchmarks. In Section 6.1, we evaluate the ability of one-step gradient on real-world fine-tuning tasks to justify our theory on natural language understanding, i.e. Theorem 3.3 and Lemma D.5. In Section 6.2, we evaluate on mathematical reasoning, general knowledge, and code generation tasks, with more data and epochs for further evaluating math reasoning ability in Section 6.3. Furthermore, we compare the time and memory cost across different methods to illustrate our efficiency.
this section cite: []

Section: One-Step Full Gradient Could Suffice in Natural Language Understanding
We fine-tune T5 base model (Raffel et al., 2020) on a subset from GLUE (Wang et al., 2019) -MNLI, SST2, CoLA, QNLI, and MRPC. We evaluate the test performance by accuracy (%). We compare LoRA (Hu et al., 2022), LoRA+ (Hayou et al., 2024), P-LoRA (Zhang & Pilanci, 2024), PiSSA (Meng et al., 2024), LoRA-GA (Wang et al., 2024), LoRA-Pro (Wang et al., 2025), and LoRA-One with rank 8. The hyperparameters are optimized for each method. More experimental details are presented in Appendix G.4.
Before experimental comparison, we first access the capacity of the one-step full gradient with its low-rank components on these real-world fine-tuning tasks. We approximate the one-step full-batch update G ♮ from full fine-tuning by a large sampled batch (2048) as G ♮ B and a best rank-r approximation of G ♮ for r = 8 using a smaller sampled batch (8) as P r (G ♮ b ). We optimize the learning rate, i.e. η * , and update the base model by
W ♮ -η * G ♮ B and W ♮ -η * P r (G ♮ b )
for SST2, CoLA, QNLI, and MRPC, respectively.
The top rows of Table 2 shows that the test performance can be significantly improved over the pre-trained model by the one-step full gradient step with proper selection of stepsize. This improvement is still promising (even better) after taking the best rank-r approximation with smaller sampled batch, which is equivalent to (Spectral-init). We remark that low-rank update with small batch for CoLA and MRPC only costs less than one second but already matches the performance of LoRA which needs tens of seconds. Accordingly, one-step full gradient can suffice for finetuning on small-scale datasets, e.g., CoLA, MRPC.
Besides, Table 2 also shows that LoRA-One outperforms other LoRA-based methods on three tasks out of five and achieves the best in average. Significant gains appear on the smaller benchmarks, i.e. CoLA and MRPC. On large datasets such as MNLI and QNLI, LoRA-Pro performs better but is with more cost. This is because, LoRA-pro (Wang et al., 2025) approximates gradient from full fine-tuning at every training step while LoRA-One only conducts at the first step. LoRA-pro adds 10drfoot_1 + 6kr 2 + 155r 3 /6 more FLOPs than LoRA-One per pair of matrices for time cost. For memory cost, LoRA-pro on MetaMathQA100k with rank 8 costs 43.87 GB while our method only costs 21.7 GB for memory management.
this section cite: ['b42', 'b51', 'b22', 'b17', 'b61', 'b36', 'b52', 'b53', 'b53']

Section: Natural Language Generation
We fine-tune LLaMA 2-7B (Touvron et al., 2023) on: 1) 100K samples from MetaMathQA (Yu et al., 2024) and evaluate the accuracy based on two types of prompting: (a) direct prompting, (b) 8-shot Chain-of-Thought 2 (CoT) (Wei et al., 2022) prompting on GSM8K (Cobbe et al., 2021); 2) Alpaca (Taori et al., 2023) and evaluate on the MMLU (Hendrycks et al., 2021) benchmarks using direct prompting; 3) 100K samples from Code-Feedback (Zheng et al., 2024) and evaluate the PASS@1 on HumanEval (Chen et al., 2021a). We compare LoRA, LoRA-GA, and LoRA-One with rank 8. The learning rate and batch size are optimized for each method. More experimental details are presented in Appendix G.2.
Table 3 shows that LoRA-One consistently outperforms both vanilla LoRA and LoRA-GA across different tasks and prompting methods. LoRA-One achieves 60.44% accuracy under direct prompting-about 1.18 points higher than LoRA-and 55.88% in the few-shot CoT setting, a gain of roughly 2.52 points, indicating it not only strengthens the model's core problem-solving abilities but also its capacity for coherent, multi-step reasoning. On MMLU, LoRA-One also shows superior generalization on the MMLU benchmark (47.24% vs. 45.73% for LoRA), indicating improved knowledge retention across diverse domains. Finally, LoRA-One excels in code generation, it achieves a PASS@1 score of 28.66%, nearly 3 points higher than LoRA's 25.85% and also improving upon LoRA-GA, implying better adaptation to structured code synthesis tasks. Moreover, LoRA-One exhibits noticeably lower run-to-run variability compared to  the baselines, indicating better stability under (Spectral-init). Regarding the time and memory cost, LoRA-One takes almost the same cost as LoRA, as shown in Table 4 across all three datasets. This suggests that LoRA-One delivers its intended benefits-such as improved convergence stability or enhanced adaptability-without imposing any meaningful extra time or memory cost during fine-tuning.
this section cite: ['b48', 'b59', 'b55', 'b8', 'b45', 'b19', 'b64']

Section: Math Reasoning on Full Data and Multiple Epochs
Beyond Section 6.2, we further fine-tune LLaMA 2-7B on the complete MetaMathQA (395K) dataset for 4 epochs to access the maximum capacity of math reasoning. Here we compare LoRA, LoRA+, LoRA-GA, and LoRA-One with rank 8. We evaluate the fine-tuned models on GSM8K with direct prompting. The learning rate and batch size are optimized for each method.
More experimental details are presented in Appendix G.3. Epoch 1 Epoch 2 Epoch 3 Epoch 4 56 58 60 62 64 Accuracy (%) LoRA LoRA+ LoRA-GA LoRA-One Results are reported as mean over 2 runs (higher is better).
Fig. 5 shows that LoRA-One consistently leads other methods over epochs, suggesting that it scales more effectively with additional training and data. In contrast, LoRA-GA only shows marginal gains over LoRA and LoRA+.
this section cite: []

Section: Conclusion
This paper theoretically demonstrates how LoRA can be improved from our theoretical analysis in both linear and nonlinear models: the alignment between LoRA's gradient update (A t , B t ) and the singular subspace of G ♮ , and adding preconditioners. Our theory derives the optimal initialization strategy for LoRA, clarifies some potential issues behind gradient alignment work, and bridge theory to practice with the promising performance of LoRA-One.
this section cite: []

Section: References
Ref_id:b0 Title: The mergedstaircase property: a necessary and nearly sufficient condition for SGD learning of sparse functions on two-layer neural networks Year: (2022)
Ref_id:b1 Title: Online stochastic gradient descent on non-convex losses from high-dimensional inference Year: (2021)
Ref_id:b2 Title: High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation Year: (2022)
Ref_id:b3 Title: On learning gaussian multi-index models with gradient flow Year: (2023)
Ref_id:b4 Title: Language Models are Few-Shot Learners Year: (2020)
Ref_id:b5 Title: Globally Optimal Gradient Descent for a ConvNet with Gaussian Inputs Year: (2017)
Ref_id:b6 Title: Evaluating Large Language Models Trained on Code Year: ()
Ref_id:b7 Title: Spectral Methods for Data Science: A Statistical Perspective. Foundations and Trends® in Machine Learning Year: (2021)
Ref_id:b8 Title: Training Verifiers to Solve Math Word Problems Year: (2021)
Ref_id:b9 Title: Asymptotics of feature learning in two-layer networks after one gradient-step Year: (2024)
Ref_id:b10 Title: Neural Networks can Learn Representations with Gradient Descent Year: (2022)
Ref_id:b11 Title: A Random Matrix Theory Perspective on the Spectrum of Learned Features and Asymptotic Generalization Capabilities Year: (2025)
Ref_id:b12 Title: Gradient dynamics for low-rank fine-tuning beyond kernels Year: (2024)
Ref_id:b13 Title: A Validation Approach to Over-parameterized Matrix and Image Recovery Year: (2022)
Ref_id:b14 Title: Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping Year: (2020)
Ref_id:b15 Title: Guarantees of a Preconditioned Subgradient Algorithm for Overparameterized Asymmetric Low-rank Matrix Recovery Year: (2024)
Ref_id:b16 Title: Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey Year: (2024)
Ref_id:b17 Title: LoRA+: Efficient Low Rank Adaptation of Large Models Year: (2024)
Ref_id:b18 Title: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification Year: (2015)
Ref_id:b19 Title: Measuring Massive Multitask Language Understanding Year: (2021)
Ref_id:b20 Title: Matrix Analysis Year: (2012)
Ref_id:b21 Title: Parameter-Efficient Transfer Learning for NLP Year: (2019)
Ref_id:b22 Title: LoRA: Low-Rank Adaptation of Large Language Models Year: (2022)
Ref_id:b23 Title: Neural Tangent Kernel: Convergence and Generalization in Neural Networks Year: (2018)
Ref_id:b24 Title: LoRA Training in the NTK Regime has No Spurious Local Minima Year: (2024)
Ref_id:b25 Title: Preconditioning Matters: Fast Global Convergence of Nonconvex Matrix Factorization via Scaled Gradient Descent Year: (2024)
Ref_id:b26 Title: LoRA Training Provably Converges to a Low-Rank Global Minimum or It Fails Loudly (But it Probably Won't Fail) Year: (2025)
Ref_id:b27 Title: Vector-based Random Matrix Adaptation Year: (2024)
Ref_id:b28 Title: The mnist database of handwritten digits Year: (1998)
Ref_id:b29 Title: Neural network learns low-dimensional polynomials with SGD near the information-theoretic limit Year: (2024)
Ref_id:b30 Title: On the Crucial Role of Initialization for Matrix Factorization Year: (2025)
Ref_id:b31 Title: Algorithmic Regularization in Over-parameterized Matrix Sensing and Neural Networks with Quadratic Activations Year: (2018)
Ref_id:b32 Title: On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models Year: (2025)
Ref_id:b33 Title:  Year: (2017)
Ref_id:b34 Title: Beyond Procrustes: Balancing-Free Gradient Descent for Asymmetric Low-Rank Matrix Sensing Year: (2021)
Ref_id:b35 Title: A Kernel-Based View of Language Model Fine-Tuning Year: (2023)
Ref_id:b36 Title: PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models Year: (2024)
Ref_id:b37 Title: A riemannian geometry for low-rank matrix completion Year: (2012)
Ref_id:b38 Title: A Theory of Non-Linear Feature Learning with One Gradient Step in Two-Layer Neural Networks Year: (2024)
Ref_id:b39 Title: Simplified Neuron Model as a Principal Component Analyzer Year: (1982)
Ref_id:b40 Title: Pretrained Transformer Efficiently Learns Low-Dimensional Target Functions In-Context Year: (2024)
Ref_id:b41 Title: Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning Year: (2024)
Ref_id:b42 Title: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer Year: (2020)
Ref_id:b43 Title: Implicit Balancing and Regularization: Generalization and Convergence Guarantees for Overparameterized Asymmetric Matrix Sensing Year: (2023)
Ref_id:b44 Title: Small random initialization is akin to spectral learning: Optimization and generalization guarantees for overparameterized low-rank matrix reconstruction Year: (2021)
Ref_id:b45 Title: An Instruction-following LLaMA model Year: (2023)
Ref_id:b46 Title: Language Models for Dialog Applications Year: (2022)
Ref_id:b47 Title: Accelerating Ill-Conditioned Low-Rank Matrix Estimation via Scaled Gradient Descent Year: (2021)
Ref_id:b48 Title: Open Foundation and Fine-Tuned Chat Models Year: (2023)
Ref_id:b49 Title: Introduction to the non-asymptotic analysis of random matrices Year: (2010)
Ref_id:b50 Title: High-Dimensional Probability: An Introduction with Applications in Data Science Year: (2018)
Ref_id:b51 Title: GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding Year: (2019)
Ref_id:b52 Title: Low-Rank Adaptation with Gradient Approximation Year: (2024)
Ref_id:b53 Title: LoRA-Pro: Are Low-Rank Adapters Properly Optimized? Year: (2025)
Ref_id:b54 Title: Perturbation bounds in connection with singular value decomposition Year: (1972)
Ref_id:b55 Title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models Year: (2022)
Ref_id:b56 Title: How Over-Parameterization Slows Down Gradient Descent in Matrix Sensing: The Curses of Symmetry and Initialization Year: (2024)
Ref_id:b57 Title: The Power of Preconditioning in Overparameterized Low-Rank Matrix Sensing Year: (2023)
Ref_id:b58 Title: Understanding the Learning Dynamics of LoRA: A Gradient Flow Perspective on Low-Rank Adaptation in Matrix Factorization Year: (2025)
Ref_id:b59 Title: Bootstrap Your Own Mathematical Questions for Large Language Models Year: (2024)
Ref_id:b60 Title: The Expressive Power of Low-Rank Adaptation Year: (2024)
Ref_id:b61 Title: Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models Year: (2024)
Ref_id:b62 Title: Preconditioned Gradient Descent for Overparameterized Nonconvex Burer-Monteiro Factorization with Global Optimality Certification Year: (2023)
Ref_id:b63 Title: Preconditioned Gradient Descent for Over-Parameterized Nonconvex Matrix Factorization Year: (2021)
Ref_id:b64 Title: Integrating Code Generation with Execution and Refinement Year: (2024)
Ref_id:b65 Title: Imbalance-Regularized LoRA: A Plug-and-Play Method for Improving Fine-Tuning of Foundation Models Year: (2024)
