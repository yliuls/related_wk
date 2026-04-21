Title: Model Immunization from a Condition Number Perspective
Abstract: Model immunization aims to pre-train models that are difficult to fine-tune on harmful tasks while retaining their utility on other non-harmful tasks. Though prior work has shown empirical evidence for immunizing text-to-image models, the key understanding of when immunization is possible and a precise definition of an immunized model remain unclear. In this work, we propose a framework, based on the condition number of a Hessian matrix, to analyze model immunization for linear models. Building on this framework, we design an algorithm with regularization terms to control the resulting condition numbers after pre-training. Empirical results on linear models and non-linear deep-nets demonstrate the effectiveness of the proposed algorithm on model immunization. The code is available at https://github.com/amberyzheng/  model-immunization-cond-num.

Section: Introduction
Model immunization, recently proposed by Zheng & Yeh (2024), studies how to pre-train a model that is more difficult to fine-tune on harmful content, but not others. The aim is to mitigate the risk of misuse (Brundage et al., 2018;Marchal et al., 2024) associated with open-sourced models by immunizing them before they are released to the public. Zheng & Yeh (2024) focus on immunizing text-to-image models, where they formulate immunization as a bi-level optimization. Empirically, they show that pre-trained diffusion models that undergo immunization are more difficult to finetune on a given harmful concept dataset. To quantify this difficulty, they compare the generation quality of models with and without immunization after a fixed number of finetuning iterations. While the empirical results are promising, Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). a definition of an immunized model and the circumstances that make immunization possible remain unclear.
To tackle this issue, we propose a framework to study model immunization using the condition number (Gloub & Van Loan, 1996). The effectiveness of immunization can be characterized by the condition number of the Hessian matrix. When using gradient-based methods during finetuning, a condition number closer to one indicates faster convergence (Boyd & Vandenberghe, 2004), i.e., easier to fine-tune. With this perspective, we observe that the existence of an effective immunization for linear models is related to the angle between the singular vectors of the harmful fine-tuning dataset's covariance matrix and the pretraining dataset's covariance matrix.
From this condition number perspective, we propose an immunization algorithm to find such a model. In detail, we propose two additional terms to regularize the condition number during pre-training. Each of the introduced regularization terms can be shown to ensure a monotonic increase/decrease of the condition number under gradient updates.
Beyond the theoretical results, we empirically validate the proposed algorithm on linear models for regression and image classification tasks. Lastly, we conduct experiments using the proposed algorithm on non-linear models, i.e., deep-nets. Despite the gap in theory, we observe that the proposed approach remains effective at model immunization across ResNet (He et al., 2016) and ViT (Dosovitskiy, 2021).
Our contributions are summarized as follows:
• We introduce a framework based on the condition number to study the task of model immunization. This framework leads to a concrete definition of an immunized model along with a novel experiment setup and evaluation metric to compare the quality of different immunization techniques.
• We propose regularizers to maximize/minimize the condition number, with a guaranteed monotonic increase/decrease when updated with the gradient-based method.
• Together with the task objective and regularizers, we demonstrate that the proposed algorithm effectively immunizes linear models and deep-nets on regression/image classification tasks.
κ(S) ≜ ∥S∥ 2 S † 2 = σ max S /σ min S ,(1)
where † is the pseudoinverse and σ S corresponds to the max/min singular value of S. The condition number is related to the convergence rate of gradient-based algorithms.
Consider an optimization problem min w L(w) where L is strongly convex and has a Hessian ∇ 2 L with max/min singular values denoted as σ max/min . In this case, the constant step-size steepest descent algorithm has a convergence rate (Bubeck, 2015) of the following
∥w t -w * ∥ 2 ≤ 1 - σ min σ max t ∥w 0 -w * ∥ 2 ,(2)
where w * denotes the optimal solution, and w t denotes the steepest descent iterate at step t. We can observe that a larger condition number corresponds to a slower convergence.
Condition number regularization. Nenov et al. (2024) proposed a regularizer for minimizing the condition number of some general matrix S R well (S) = 1 2 ∥S∥ 2 2 -
1 2p ∥S∥ 2 F ,(3)
in which p is the minimum dimension of S, and the norms correspond to the spectral norm and Frobenius norm. They showed that R well (S) is a valid regularizer by proving its nonnegativity, and is an upper bound on log (κ (S)). In addition, they showed that R well (S) is differentiable under some mild conditions, and if updated with gradient descent, it is guaranteed to decrease the condition number monotonically. See Appendix A for the exact statements.
Different from Nenov et al. (2024), we propose a differentiable regularizer that is guaranteed to increase the condition number as an upper bound on 1/log (κ (S)). For model immunization, instead of a general matrix S, we need to consider the regularization of the Hessian of linear models composed of a feature extractor and a classifier, while preserving their differentiability and monotonicity guarantees during gradient updates to the feature extractor.
Transfer learning via linear probing. In this work, we focus on the transfer learning method of linear probing. Given a pre-trained feature extractor f θ : R Din → R Dhid , linear probing learns an a linear classifier h w : R Dhid → R Dout over the target dataset D = {(x, y)} using the frozen feature extractor f θ . This model learning is formulated as the following optimization problem
min w L(D, w, θ) ≜ min w (x,y)∈D ℓ(h w • f θ (x), y)(4)
where ℓ denotes a suitable loss function, e.g., cross-entropy. By keeping θ fixed, the model leverages features learned from pre-training task and transfers them to the target task. This approach is effective when the target dataset is too small to train a model from scratch.
this section cite: ['b8', 'b34', 'b17', 'b6', 'b20', 'b13', 'b9', 'b38', 'b38']

Section: Immunization with Condition Number
The goal of model immunization is to learn a pre-trained model g ω • f θ I , consisting of a classifier g ω and an immunized feature extractor f θ I , such that fine-tuning f θ I on a harmful task is difficult, but not for other tasks. The model should also maintain a good pre-training task performance. Specifically, we study the setting when a bad actor uses linear probing on a pre-trained linear feature extractor with gradient descent.
Immunization setting. We denote a pre-training dataset as D P = {(x, y)} and a harmful dataset as D H = {(x, ỹ)} where x ∈ R Din . The bad actor performs linear probing using D H following Eq. ( 4) with an ℓ 2 loss. We will focus our analysis on linear pre-trained feature extractor without dimensionality reduction, i.e., f θ ≜ x ⊤ θ with θ ∈ R Din×Din .
Definition 3.1. Under this setting, a model is said to be immunized if it satisfies the following:
(a) It is more difficult to apply linear probing on the harmful task D H using the immunized feature extractor f θ I than directly on the input data, i.e.,
κ(∇ 2 w L(D H , w, θ I )) ≫ κ(∇ 2 w L(D H , w, I)),(5)
where I denotes the identity matrix.
(b) It is not more difficult to apply linear probing on other tasks. As there is only one other task D P , an immunized feature extractor should have
κ(∇ 2 ω L(D P , ω, θ I )) ≤ κ(∇ 2 ω L(D P , ω, I)).(6)
Note: we use ω to denote the classifier parameters of the pre-training task and w for the harmful task.
(c) The immunized model should maintain a competitive task performance on the pre-training dataset D P , i.e.,
min ω,θ L(D P , ω, θ) ≈ min ω L(D P , ω, θ I ).(7)
For linear models, as long as θ I is invertible, exact equality can be achieved.
this section cite: []

Section: Analysis on Immunized Linear Models
To provide some intuition on how the feature extractor θ affects the convergence of linear probing, we study the analytical form of the singular values of the Hessian. For readability, we will rewrite linear probing in Eq. ( 4) by considering f θ ≜ x ⊤ θ and a ℓ 2 -loss.
Let X H ∈ R N ×Din and Y H ∈ R N ×Dout denote data from D H stacked into matrices with N ≜ |D H |. When using a ℓ 2 -loss, Eq. ( 4) can be written as
min w L(D H , w, θ) = min w ∥(X H θ)w -Y ∥ 2 2 .(8)
In this case, the Hessian matrix
H H (θ) ≜ ∇ 2 w L(D H , w, θ) = θ ⊤ K H θ,(9)
where K H ≜ X ⊤ H X H is the data covariance matrix. Proposition 3.2. The singular values of the Hessian matrix in Eq. ( 9) are given by
σ i = Din j=1 σ θ,i (u ⊤ θ,i q j ) √ γ j 2 , ∀i ∈ {1, . . . , D in }. (10
)
Here, σ θ,i and u θ,i correspond to the i-th singular value and vector of θ. Next, γ j and q j correspond to the j-th singular value and vector of the covariance K.
Proof sketch. This result can be shown by using the fact that K H is a symmetric positive semi-definite matrix and decomposing via SVD. The complete proof is provided in Appendix B.1. □ From Eq. ( 10), we can see that the singular value of the Hessian depends on the relative angle between the singular vectors between feature extractor θ and the covariance matrix of the data K H . As the feature extractor is shared between the pretrained D P and harmful D H datasets, the strength of the immunization depends on the relative angle between the singular vectors of K P and K H . For example, if the singular vectors (sorted by the singular values) are all perfectly aligned between the two, then no θ can simultaneously maximize κ(∇ 2 w L(D H , w, θ)) and minimize κ(∇ 2 ω L(D P , ω, θ)). With a better understanding of the effect of the feature extractor θ on the condition number, we will next present an algorithm to immunize a model.
this section cite: []

Section: Algorithm for Immunizing a Model
We formulate model immunization as an optimization problem with the following objective:
min ω,θ R ill (H H (θ)) + R well (H P (θ)) + L(D P , ω, θ), (11
)
where R ill , to be defined in Sec. 4.1, denotes our proposed regularizer to maximize the condition number, R well Algorithm 1 Condition number regularized gradient descent for model immunization input Primary task D P = (X P , Y P ), harmful task input X H , supervised loss L, learning rate η, regularizing constants λ P , λ H ∈ R + , model initialization θ 0 , ω 0 1:
K P = X ⊤ P X P 2: K H = X ⊤ H X H 3: for t = 0, 1, . . . , T -1 do 4: ω t+1 = ω t -η∇ ω L(ω t , θ t ; D P ) 5: H P (θ t ) = θ ⊤ t K P θ t , H H (θ t ) = θ ⊤ t K H θ t 6: θ t+1 = θ t -η∇ θ L(ω t , θ t ; X 1 ) -ηλ P K -1 P ∇ θ R well (H P (θ t )) -ηλ H K -1 H ∇ θ R ill (H H (θ t )) 7: end for output Immunized feature extractor θ I ≜ θ T .
in Eq. ( 3) denotes the regularizer to minimize the condition number, H P (θ) ≜ ∇ 2 ω L(D P , ω, θ) = θ ⊤ K P θ is the Hessian matrix of the pre-training task, and L denotes the supervised loss.
Each of the terms encourages the model to satisfy the three immunization requirements in Definition 3.1. For readability, we have dropped the scalar hyperparameters balancing the terms. We propose to solve Eq. ( 11) using a gradientbased method as outlined in Alg. 1.
In the remainder of this section, we will first introduce the novel regularizer to maximize general matrices' condition number and their relevant properties (Sec. 4.1). We then show how to incorporate the regularizers R ill and R well into the immunization setup (Sec. 4.2). Finally, we discuss the provable guarantees with respect to each of the regularizers (Sec. 4.3).
this section cite: []

Section: Regularizer for Maximizing the Condition Number
We analyze the condition number of a general matrix S ∈ R pr×pc , p = min{p r , p c }, and rank (S) = k ≤ p. The compact SVD of S is given by
S = U Diag(σ)V ⊤ , in which σ = [σ 1 , • • • , σ k ] ⊤ such that σ max S = σ 1 ≥ σ 2 ≥ • • • ≥ σ k = σ min S > 0 and u i , v i denotes the i th column vector of U , V for i ∈ [k].
Inspired by the regularizer for minimizing the condition number, we propose its counterpart for maximizing the condition number
R ill (S) = 1 1 2k ∥S∥ 2 F -1 2 (σ min S ) 2 ,(12)
which satisfies the properties in the following theorem.
Theorem 4.1 (Properties of κ-maximizing regularizer R ill (S)).
(1) [Nonnegativity] For any S ∈ R pr×pc , R ill (S) ≥ 0, and R ill (S) = 0 if and only if κ (S) = ∞.
(
) [Upper Bound] 1 log(κ(S)) ≤ (σ max S ) 2 R ill (S), i.e., R ill (S) upper bounds 1 log(κ(S)) when σ max S is reasonably away from ∞. (3) [Differentiability] If σ min S = σ k < σ i for any i < k, i.e., σ min S is unique, then R ill (S) is differentiable and
∇ S R ill (S) = σ k u k v ⊤ k -1 k S 1 2k ∥S∥ 2 F -1 2 (σ min S ) 2 2 . (13
) (4) [Monotonic Increase] If σ min S is unique, update S with ∇ S R ill (S) such that S ′ = S -η 2 ∇ S R ill (S) for 0 < η 2 < k k-1 1 2k ∥S∥ 2 F -1 2 (σ min S ) 2 2
, then κ (S ′ ) > κ (S).
Proof sketch. We provide some intuitive illustrations of the proof and defer the complete version to Appendix B.2.
For (1), as the squared Frobenius norm of a matrix equals the sum of the squares of its singular values, the denominator of R ill (S) is the average of the squared singular values minus their minimum, ensuring it is nonnegative. It can be shown that R ill (S) is inversely related to κ (S), which indicates that R ill (S) = 0 if and only if κ (S) = ∞.
For (2) the upper bound holds by the design of R ill (S) and applying the mean value inequality on
log κ(S) 2 = log (σ max S ) 2 -log σ min S 2 . (14
)
For (3), even though σ min S is not differentiable since it involves taking the minimum of the singular values, its subdifferential is well-defined (Lewis, 1995). When σ min S is unique, its subdifferential reduces to a singleton, i.e., its gradient, making R ill (S) also differentiable. For (4), one key observation is that the closed-form ∇ S R ill (S) shares the same set of singular vectors as S, so that the linear relation in gradient update can be passed on to singular values. By choosing a suitable step size, the increase in condition number can be guaranteed.
this section cite: ['b32']

Section: □
Theorem 4.1 demonstrates that the regularizer R ill (S) introduced is a reasonable upper bound for maximizing condition numbers and indicates that under some mild condition, i.e., the minimum singular value is unique, simple first-order algorithms like gradient descent can be used to minimize the regularizer with guaranteed increase in condition number.
this section cite: []

Section: Incorporating Regularizers into Immunization
Given the immunization setup, we now analyze the regularizer R ill and R well for matrices with the specific structure of feature covariance matrices, and propose the corresponding algorithm for model immunization.
As illustrated in the immunization setup, the feature extractor θ is the trainable parameter. For data X ∈ R N ×Din of the feature extractor, we analyze the condition number of H(θ) ≜ θ ⊤ Kθ ∈ R Din×Din with rank (H) = k, and compact SVD H = U Diag(σ)V ⊤ . Recall, we define K = X ⊤ X to be the covariance matrix of the data.
In the following theorem, we show that under the same conditions, the introduced regularizers R ill (•) and R well (•) are also differentiable w.r.t. θ when applied to θ ⊤ Kθ. Theorem 4.2. For H (θ) = θ ⊤ Kθ, if its maximum and minimum singular values σ 1 and σ k are unique, then
(1) ∇ θ R well (H (θ)) = 2Kθ σ 1 v 1 v ⊤ 1 -1 Din θ ⊤ Kθ , (2) ∇ θ R ill (H (θ)) = 2Kθ(σ k v k v ⊤ k -1 k θ ⊤ Kθ) ( 1 2k ∥θ ⊤ Kθ∥ 2 F -1 2 σ 2 k ) 2 .
Proof sketch. The differentiability follows from the same argument of Theorem 4.1 (3) under the condition that the maximum and minimum singular values are unique. The closed-form gradients are computed with the chain rule in matrix calculus defined by the Frobenius inner product. The complete proof can be found in Appendix B.3. □
With the closed-form gradient of the regularizers w.r.t. θ, we propose our algorithm for model immunization in Alg. 1. Specifically, Alg. 1 employs the general gradient descent framework. Line 4 conducts standard updates for the classifier ω, minimizing the supervised loss L. In lines 5 to 6, the regularizers R ill and R well are applied on the feature covariance H H (θ) of the harmful task and H P (θ) of the pretraining task. This is done by updating the feature extractor θ with the gradients ∇ θ R ill (H H ) and ∇ θ R well (H P ) normalized by their input covariances and the gradient from the supervised loss ∇ θ L.
this section cite: []

Section: Condition Number Guarantees
We show in the following theorem that the condition number decrease/increase guarantees introduced in Theorem A.1 (4) and Theorem 4.1 (4) are preserved for θ ⊤ Kθ even when the gradient updates are taken in θ as in Alg. 1, instead of θ ⊤ Kθ.
Theorem 4.3. For the trainable feature extractor θ, feature covariance H P (θ) = θ ⊤ K P θ of the primary task and H H (θ) = θ ⊤ K H θ of the immunization task with rank (H
P ) = k P , rank (H H ) = k H and compact SVD H P (θ) = U P Diag(σ P )V ⊤ P , H H (θ) = U H Diag(σ H )V ⊤ H , for σ P = [σ P,1 , • • • , σ P,kP ], σ H = [σ H,1 , • • • , σ H,kH ], (1) if σ max HP is unique, i.e., σ max HP = σ P,1 > σ P,2 , update θ such that θ ′ = θ -η P K -1 P ∇ θ R well (H P (θ)) for 0 < η P < min 1 (1-1 D in )σP,1 , √ σP,1σP,2-σP,2 2 D in σ 2 P,2 , then κ θ ′ ⊤ K P θ ′ < κ θ ⊤ K P θ , (2) if σ min HH is unique, i.e., σ min HH = σ H,kH < σ H,kH-1 , update θ such that θ ′ = θ -η H K -1 H ∇ θ R ill (H H (θ)) for 0 < η H < 1 1-2σ min H H /kH 1 2kH θ ⊤ K H θ 2 F -1 2 σ min HH 2 2 , then κ θ ′ ⊤ K H θ ′ > κ θ ⊤ K H θ . Proof sketch.
There is a mismatch between the gradient update on θ and the condition number update, which is observed for H (θ). To address this, we carefully leverage the structure of the problem, noting that H (θ), unlike a general matrix, is symmetric and positive semidefinite, with identical left and right singular vectors. Exploiting this property, along with our algorithm design, ensures that the linearity in singular value updates is preserved when expanding H (θ ′ ) using the closed-form gradient in Theorem 4.2. Consequently, a monotonic increase or decrease in the condition number can be guaranteed by appropriately selecting the step size. The full proof is provided in Appendix B.4. □
this section cite: []

Section: Additional Discussion
Implementation considerations. At a glance, it may seem that to implement Alg. 1 using automatic differentiation packages, e.g., Pytorch (Paszke et al., 2019), one would have to implement a custom optimizer and involve multiple update steps. Instead, we observe that by directly modifying the computation graph, it would only involve a single backward pass. This is done by introducing a "dummy layer" with an identified function as its forward pass and its backward pass multiplies the gradient by the inverse feature covariance matrix. The "dummy layer" implementation is inspired by prior works in gradient estimator (Bengio et al., 2013;Roeder et al., 2017). Pseudo-code is provided in Appendix C.3.
Limitations. The monotonicity guarantees in Theorem 4.3 serve as a theoretical justification for our proposed algorithm, albeit a partial reflection of the application setup. Note that the feature extractor is updated with the gradients of the two regularizers jointly together with that of the supervised loss and the guarantees may not linearly combine as such. In practice, maintaining the balance between κ (H P (θ)) and κ (H H (θ)) requires a proper choice of hyperparameters.
Next, the current framework we analyzed focuses on linear feature extractors and using linear probing for transfer learning. We are aware of the practical limitations of this setting.
To address this, in the experiments, we empirically study the effect of the proposed method on non-linear models, i.e., deep-nets, and demonstrate our method's potential despite the theoretical gap.
this section cite: ['b41', 'b2', 'b46']

Section: Experiments
We evaluate the proposed Alg. 1 on regression and image classification tasks using linear models, and also explored immunizing non-linear models, i.e., deep-nets. Experiment and implementation details are provided in Appendix C.
this section cite: []

Section: Evaluation metrics.
We introduce the relative immunization ratio (RIR) to quantify the effectiveness of the immunization based on the ratio of the condition number of Hessian, defined as follows:
RIR ≜ κ(H H (θ I )) κ(H H (I)) (i) κ(H P (θ I )) κ(H P (I)) (ii)(15)
where I denotes the identity matrix. Each term here measures the ratio between condition numbers with and without the pre-trained feature extractor on the (i) harmful task or (ii) on the pre-training task.
A successful immunization is characterized by:
(i) a large ratio κ(HH(θI)) κ(HH(I)) , i.e., using the immunized feature extractor makes the optimization of linear probing more difficult on the harmful task. (ii) a small ratio κ(HP(θI)) κ(HP(I)) ), i.e., using the pre-trained extractor do not make optimization more difficult on the pre-training task.
To obtain a single metric, we compare (i) and (ii) relative to each other. In other words, an effective immunized model should have a relative immunization ratio RIR ≫ 1.
Baselines. We consider three baselines for comparisons:
• R ill Only immunizes the model by minimizing only the regularizer R ill (H H ) as defined in Eq. ( 12) using gradient descent.
• IMMA (Zheng & Yeh, 2024) 16) vs. Epochs. We visualize the convergence of linear probing of different immunized models using gradient descent with an exact line search. Here, Identity corresponds to not using a feature extractor, i.e., θI = I. Observe that Ours made the convergence faster on DP while slower in DH when compared to the other baselines; consistent with the results in Tab. 1. via gradient descent instead of using our proposed regularizers.
this section cite: []

Section: Experiments on Immunizing Linear Models
Linear regression task. We use the regression task from the House prices dataset (Montoya & DataCanary, 2016). We split the data into D P and D H based on the feature MSZoning. For the pre-training task, we use the target of LotArea and for the harmful task we use the target of SalePrice. Both D P and D H contain input vectors of dimension 79. We immunized the model by running Alg. 1 for 100 epochs with η = 0.005. We choose λ P and λ H by balancing the gradient norm of R well and R ill . The implementation details can be found in Appendix C.2.
In Tab. 1, we present the empirical results of immunizing a linear feature extractor θ. We observe that only Opt κ and our method successfully immunize the model achieving an RIR that's much greater than 1. For R ill Only and IMMA, while they successfully made the harmful task more ill-conditioned, i.e., Eq. (15) (i) went up, however, this is at the cost of making the other task ill-conditioned as well, i.e., Eq. (15) (ii) went up.
Next, we demonstrate how a large condition number slows down the convergence of linear probing on the harmful task by analyzing the norm ratio defined as
∥w t -w ⋆ ∥ 2 2 /∥w 0 -w ⋆ ∥ 2 2 ,(16)
which measures how the classifier weights w t at step t approach the optimal weights w ⋆ during fine-tuning. Note, naively choosing a step size will not reflect the difference in condition number. Hence, we use the exact line search (Boyd & Vandenberghe, 2004) which chooses the step size that minimizes the loss at each iteration.
As illustrated in Fig. 1, both our method and Opt κ slow down convergence in D H compared to Identity while accelerating convergence in D P . Furthermore, our method achieves a stronger immunization effect than Opt κ. In contrast, R ill Only and IMMA slowed the convergence on both the harmful task D H and the pre-training task D P .
this section cite: ['b35', 'b6']

Section: Image classification task.
For image classification, we conduct experiments using MNIST (LeCun, 1998). The MNIST dataset consists of images over 10-digit classes, which can be formulated into 10 independent binary classification tasks. Across all pairs of tasks, we choose one to be the harmful task D H and the other the pre-training D P resulting in a total of 90 experiments. We ran Alg. 1 for 30 epochs with η = 0.005 for these experiments. The implementation details can be found in Appendix C.2.
In Tab. 2, we present the quantitative results on these binary task pairs. For each entry, the values are averaged over all 90 pairs. Based on the averaged results, we observe that our method effectively immunizes the linear feature extractor θ on D H without compromising performance on D P . Although Opt κ achieves comparable RIR with our method, the variances of the metric values are relatively large. This indicates that Opt κ is sensitive to random initialization while our method is robust.
In Fig. 2 we further analyze the results by visualizing the log(RIR) for each digit pair. A blue block indicates successful immunization, while a red block indicates failure. It can R ill Only IMMA Opt κ Ours 0 1 2 3 4 5 6 7 8 9 P 0 1 2 3 4 5 6 7 8 9 H 0.00 2.20 0.76 0.67 0.73 0.66 0.73 1.33 0.57 0.99 1.19 0.00 0.65 0.48 0.64 0.54 0.56 0.61 0.34 0.47 0.53 1.09 0.00 0.36 0.67 0.56 0.42 1.17 0.28 0.83 0.49 1.03 0.34 0.00 0.64 0.27 0.66 0.90 0.19 0.59 0.63 1.33 0.67 0.50 0.00 0.38 0.40 0.53 0.25 0.10 0.22 0.96 0.53 0.07 0.27 0.00 0.40 0.65 0.02 0.32 0.50 1.23 0.48 0.57 0.43 0.56 0.00 1.09 0.37 0.60 0.68 1.07 0.84 0.49 0.30 0.52 0.85 0.00 0.29 0.17 0.55 0.86 0.44 0.24 0.33 0.26 0.51 0.74 0.00 0.34 0.67 1.10 0.73 0.43 0.09 0.37 0.45 0.32 0.19 0.00 0 1 2 3 4 5 6 7 8 9 P 0 1 2 3 4 5 6 7 8 9 H 0.00 1.92 0.59 0.64 0.68 0.52 0.66 1.20 0.51 0.92 1.07 0.00 0.50 0.46 0.59 0.49 0.54 0.58 0.32 0.46 0.49 0.97 0.00 0.32 0.62 0.43 0.37 1.08 0.24 0.77 0.48 0.90 0.27 0.00 0.55 0.23 0.58 0.77 0.18 0.53 0.57 1.17 0.48 0.48 0.00 0.34 0.35 0.44 0.22 0.09 0.23 0.84 0.35 0.06 0.22 0.00 0.36 0.55 0.00 0.27 0.47 1.12 0.35 0.52 0.41 0.46 0.00 0.96 0.34 0.54 0.65 0.96 0.65 0.46 0.26 0.46 0.74 0.00 0.26 0.16 0.51 0.76 0.34 0.24 0.29 0.20 0.47 0.61 0.00 0.31 0.65 0.98 0.54 0.40 0.09 0.33 0.41 0.26 0.17 0.00 0 1 2 3 4 5 6 7 8 9 P 0 1 2 3 4 5 6 7 8 9 H 0.00 4.65 3.38 4.15 4.12 4.00 3.83 4.46 4.56 2.36 5.89 0.00 5.75 4.99 5.97 5.55 5.25 5.62 4.27 5.80 3.60 3.41 0.00 3.09 2.62 3.68 1.74 3.15 3.06 3.60 3.53 3.86 2.45 0.00 3.74 3.46 3.80 2.74 2.43 2.93 4.70 2.70 4.24 3.99 0.00 4.11 3.87 3.53 2.95 2.55 3.23 3.77 3.59 2.28 3.60 0.00 2.85 3.97 2.91 3.72 3.60 2.50 4.70 5.23 4.36 3.80 0.00 5.33 3.75 4.97 4.40 3.08 3.10 5.21 3.55 4.24 4.41 0.00 4.36 2.37 3.77 3.01 3.27 2.81 3.75 2.93 4.13 3.28 0.00 2.63 3.69 3.61 5.01 3.34 1.05 3.46 4.73 3.18 3.90 0.00 0 1 2 3 4 5 6 7 8 9 P 0 1 2 3 4 5 6 7 8 9 H 0.00 4.83 4.84 4.05 4.85 4.56 3.60 4.28 4.35 4.28 6.26 0.00 5.45 5.64 5.84 5.39 5.24 4.72 5.06 4.67 3.62 2.97 0.00 3.14 3.48 3.67 3.20 3.36 2.89 3.26 4.32 3.63 3.21 0.00 3.83 3.24 3.49 3.48 2.75 3.24 4.65 3.54 4.45 4.86 0.00 4.80 3.64 3.83 4.23 2.15 3.58 3.35 3.53 2.17 3.68 0.00 3.04 3.49 3.64 3.07 3.49 3.63 4.70 4.37 3.95 3.81 0.00 4.67 3.86 4.05 4.42 3.46 4.22 4.32 3.24 4.51 4.17 0.00 3.52 2.67 4.71 2.77 4.03 3.07 4.10 3.59 3.54 3.49 0.00 3.09 4.15 3.91 4.38 3.85 3.14 3.87 4.12 3.76 3.11 0.00 be observed that R ill Only fails for all digit pairs, IMMA only succeeds in one pair, and Opt κ fails for 32 out of 90 pairs. In contrast, our method achieves success across all digit pairs demonstrating its effectiveness for immunization.
Thus far, we have conducted experiments strictly following the immunization setting that we have proposed in Sec. 3. However, one limitation of the setting is that the feature extractor is assumed to be linear, which limits its real-world potential. To further study the practicality of our method, despite the theoretical gap, we conduct experiments with non-linear models, i.e., deep-nets, on a larger-scale image classification dataset of ImageNet.
this section cite: ['b31']

Section: Experiments on Immunizing Deep-Nets
Immunization task. In this experiment, we consider a common setup of linear probing on models pre-trained on Ima-geNet (Deng et al., 2009), i.e., ImageNet serves as D P . For D H we experiment with the Stanford Cars Dataset (Krause et al., 2013) and Country211 Dataset (Radford et al., 2021). These datasets have been previously used for studying transfer learning (Radford et al., 2021) for image classification.
More dataset details are deferred to Appendix C.1.
Experiment setup. For non-linear models, we experiment with the architecture of ResNet18 (He et al., 2016) and ViT (Dosovitskiy, 2021). Here we study a practical setting where a given model with parameters θ 0 has already been trained on D P and would undergo immunization to obtain θ I to be released to the public.
Note that as we are now using an initialization of θ 0 and a non-linear feature extractor f θ , we extend the RIR metric to consider those changes. Specifically, we propose
RIR θ0 ≜ κ( HH (θ I )) κ( HH (θ 0 )) (i) κ( HP (θ I )) κ( HP (θ 0 )) (ii)(17)
where we compare the immunized model θ I relative to the initialization model θ 0 . Here, H(θ) denotes the Hessian for linear probing on D H with a non-linear f θ , i.e.,
HH (θ) = ∇ 2 w L(D H , w, θ) = XH (θ) ⊤ XH (θ).(18)
Here, XH (θ) ≜ [f θ (x); ∀x ∈ D H ] ∈ R N ×Dhid denotes the concatenation of the features, with dimensions D hid , extracted from the input data. Due to memory constraints, we approximate Eq. ( 17) by randomly sampling 20 groups of training data, each containing 100 samples, and reporting the average values.
Finally, we also report the task performance after immunization. This is because, as the feature extractor is non-linear we are no longer guaranteed to retain the task performance. For ResNet18, we immunize only the last two convolutional blocks of the trained feature extractor and keep the rest of the parameters frozen as in θ 0 . For ViT, we only immunize the final transformer block. We optimize Eq. ( 11) using SGD with momentum, the default optimizer on ImageNet. Further details are provided in Appendix C.2.
Results. We present the quantitative results of immunizing deep-nets in Tab. 3. On both Cars and Country211 datasets, our method demonstrates strong performance when applied to ResNetg18 and ViT, as indicated by RIR θ0 ≫ 1. In comparison, R ill Only and IMMA did not effectively immunize the models in all evaluated settings. Next, Opt κ also succeeds in immunizing the models but our proposed method outperforms it in RIR θ0 .
Next, we report the test accuracy of the immunized models on D P , i.e., ImageNet1K. On the ResNet18 architecture, we observe a reduction in test-accuracy from the initialization model θ 0 of 68.24% to 62.36% when D H is Cars and 65.01% when D H is Country211. Interestingly, on the ViT architecture the test-accuracy increased from 81.78% to 82.79% for Cars, and 83.17% for Country211. These results suggested that it is possible to immunize a non-linear model against the harmful task without losing the effectiveness of the other task.
Table 3. Quantitative results of immunization of model pre-trained on ImageNet (Deng et al., 2009), computed over 3 random seeds. The DP test accuracy for the off-the-shelf model initialization of θ0 on ResNet18 is 68.24% and that of ViT is 81.78%. We report RIR θ0 to measure the quality of immunization. Test accuracy of DP is reported to ensure the performance on the pre-training task is maintained.
D H Method ResNet18 ViT Eq. (17) (i)↑ Eq. (17) (ii) ↓ RIR θ0 ↑ D P Test Acc. (%) ↑ Eq. (17) (i)↑ Eq. (17) (ii) ↓ RIR θ0 ↑ D P Test Acc. (%) ↑ Cars Init. θ 0 1.0 1.0 1.0 68.24 1.0 1.0 1.0 81.78 R ill Only 1.878 ±0.034 1.786 ±0.025 1.057 ±0.026 63.84 ±0.292 13.121 ±0.038 4.097 ±0.098 3.342 ±0.048 82.21 ±0.035 IMMA 0.866 ±0.002 0.889 ±0.001 0.974 ±0.002 63.57 ±0.234 1.422 ±0.006 2.090 ±0.043 0.702 ±0.007 81.89 ±0.010 Opt κ 1.217 ±0.021 0.798 ±0.005 1.527 ±0.019 63.65 ±0.148 3.598 ±0.510 0.171 ±0.033 26.369 ±2.814 82.51 ±0.085 Ours 2.386 ±0.442 0.699 ±0.062 3.467 ±0.358 62.36 ±0.173 7.945 ±0.247 0.323 ±0.086 34.517 ±0.886 82.79 ±0.200 Country211 R ill Only 20.727 ±0.791 20.675 ±1.685 1.038 ±0.05 62.17 ±1.599 69.291 ±1.198 63.519 ±6.62 1.122 ±0.097 80.73 ±0.129 IMMA 0.791 ±0.005 0.814 ±0.006 0.972 ±0.007 67.03 ±0.146 6.242 ±0.203 7.599 ±0.717 0.845 ±0.048 82.47 ±0.036 Opt κ 1.538 ±0.155 1.053 ±0.091 1.472 ±0.043 66.81 ±0.115 4.589 ±0.079 0.300 ±0.106 16.498 ±5.183 82.79 ±0.023 Ours 3.287 ±0.33 0.399 ±0.034 8.714 ±0.672 65.01 ±0.143 20.894 ±1.425 0.700 ±0.082 41.341 ±0.967 83.17 ±0.075
To further show a larger Eq. ( 17) (i) indicating that a model is better immunized, we report the linear probed (fine-tuned) results on different feature extractors and provide the test accuracy on D H , where D H is the Stanford Cars dataset.
As shown in Fig. 3, our method exhibits the slowest convergence rate on both ResNet18 and ViT, indicated by the lowest test accuracy compared with baselines. In summary, our method remains effective on deep-nets, producing models that satisfy the requirements of an immunized model as in Definition 3.1.
this section cite: ['b11', 'b28', 'b44', 'b44', 'b20', 'b13', 'b11']

Section: Related Work
We briefly discuss related research on AI safety and the condition number.
AI safety, model un/re-learning, and immunization. AI safety has received attention lately, specifically in generative AI, due to the impressive progress. We refer the reader to Brundage et al. (2018); Marchal et al. (2024); Bengio et al. (2025) for a more in-depth discussion on this topic. In the following, we will discuss model unlearning, one of the ways to mitigate the potential of misuse, followed by model immunization, which protects a model against relearning.
Machine unlearning was first introduced by Cao & Yang (2015) to remove a user's private information from a model. Approximate unlearning aims to achieve this by modifying the pre-trained model directly using the specific data samples to erase, without requiring full retraining (Nguyen et al., 2020;Wu et al., 2022;Guo et al., 2019;Sekhari et al., 2021;Neel et al., 2021). In the context of text-to-image models, several methods for concept erasure have been proposed. These include inference-time approaches (Brack et al., 2023;Schramowski et al., 2023), fine-tuning of diffusion models (Gandikota et al., 2023;Kim et al., 2023;Kumari et al., 2023), and direct model editing (Zhang et al., 2024;Gandikota et al., 2024).
While promising, these works still face potential risks of the re-emergence/re-learning of harmful data (Zheng & Yeh, 2024;Zheng et al., 2024;Zhan et al., 2024;Bertran et al., 2024;Xu et al., 2025). To avoid relearning or further finetuning on harmful data, Zheng & Yeh (2024) propose to immunize the text-to-image models against malicious finetuning and Zheng & Yeh (2025) extend model immunization to multi-concept settings. Recent work highlights the importance of preventing re-finetuning or distillation on harmful tasks in language models (Huang et al., 2024;Savani et al., 2025;Rosati et al., 2024a;Tamirisa et al., 2024;Rosati et al., 2024b;Henderson et al., 2023) and encoder probing (Ding et al., 2025), which is closely related to our goal. While we also study the task of model immunization, different from Zheng & Yeh (2024) that primarily focuses on empirical applications on generative tasks, our work aims to provide a more principled understanding of model immunization by analyzing it through the lens of the condition number.
this section cite: ['b8', 'b34', 'b4', 'b10', 'b40', 'b59', 'b19', 'b54', 'b37', 'b7', 'b53', 'b15', 'b26', 'b30', 'b62', 'b16', 'b61', 'b5', 'b60', 'b23', 'b52', 'b55', 'b21', 'b12']

Section: Minimizing Condition Number.
Condition number has been a key factor in the convergence rates and accuracies of iterative methods, e.g., Jacobi method (Arioli & Romani, 1985), steepest descent (Luenberger et al., 1984), conjugate gradient (Hestenes et al., 1952), for solving optimization problems from classic linear systems (Saad, 2003) to those with general nonlinear objectives (Nesterov, 2018) concerning modern machine learning applications. It is widely observed that a small condition number tends to speed up convergence and improve accuracy whereas a large condition number could lead to an unstable optimization procedure (Saarinen et al., 1993;Kress, 2012;Bengio et al., 2017;Guille-Escuret et al., 2021).
As a result, methods to minimize the condition number in various contexts have been proposed. Preconditioning (Evans, 1968), a technique that involves finding a matrix, i.e., the preconditioner, to multiply with the original matrix, resulting in a new matrix with a significantly smaller condition number, is widely used for solving linear systems. The preconditioner can be constructed using methods such as semidefinite programming (Jambulapati et al., 2020;2023;Qu et al., 2024) or matrix equilibration (Van der Sluis, 1969), and has recently found applications in deep learning (Saratchandran et al., 2024).
Most related to this work, Balazs et al. (2024) propose to regularize the condition number of weight matrices by directly adding the condition number term into the optimization objective and applying (sub)gradient descent. Observing that the condition number is discontinuous and nonconvex, Nenov et al. (2024) proposed a differentiable regularizer that minimizes the matrix condition number with a monotonic decrease guarantee if optimized with gradient descent. To the best of our knowledge, no notable effort has been made to increase or maximize the condition number.
this section cite: ['b0', 'b33', 'b22', 'b49', 'b39', 'b50', 'b29', 'b3', 'b18', 'b14', 'b24', 'b43', 'b51', 'b1', 'b38']

Section: Conclusion
We propose a framework for studying model immunization through the condition number of the Hessian matrix. We show that immunization can be achieved by increasing the condition number of harmful datasets while keeping it stable for the pre-training task. To achieve this, we introduce two differentiable regularizers and propose an algorithm that incorporates these regularizers into a gradient-based optimization algorithm. Empirical results on both linear and deep models demonstrate the effectiveness of our approach to model immunization. We believe that our proposed framework is a first step towards a more principled understanding of model immunization and will ultimately make open-sourced models safer.
We observe the following decomposition of M in to two matrices O and D:
M =     . . . . . . . . . . . . σ θ,i (u ⊤ θ,i q j )γ j . . . . . . . . . . . .     =       . . . . . . . . . . . . σ θ,i (u ⊤ θ,i qj )γj j ′ (σ θ,i (u ⊤ θ,i q j ′ )γ j ′ ) 2 . . . . . . . . . . . .            . . . 0 0 0 j ′ (σ θ,i (u ⊤ θ,i q j ′ γ j ) 2 0 0 0 . . .      = OD
where O is an orthonormal matrix, i.e., O ⊤ O = I, and
D = diag(d 1 , . . . , d d ) with d i = j ′ (σ θ,i (u ⊤ θ,i q j ′ )γ j ′
) 2 is a diagonal matrix. As a result, diagonal entries of D 2 are:
d 2 i = d j=1 σ θ,i (u ⊤ θ,i q j )γ j 2 .
Thus, M M ⊤ = (OD)(OD) ⊤ = OD 2 O ⊤ , and the eigenvalues of θ ⊤ Kθ are the diagonal entries of D 2 , given by:
σ i = d 2 i = d j=1 σ θ,i (u ⊤ θ,i q j )γ j 2 , i = 1, . . . , d.
Proof. By definition, R ill (S) = 1 1 2k ∥S∥ 2 F -1 2 (σ min S ) 2 . Denote R ′ ill (S) = 1 2 (σ min S ) 2 -1 k ∥S∥ 2 F , then we have R ill (S) = 1 -R ′ ill (S) , and R ′ ill (S) = 1 2 σ 2 k - 1 k k i=1 σ 2 i = 1 2k k i=1 σ 2 k -σ 2 i ≤ 0, since ∀ i ∈ [k], σ min S = σ k ≤ σ i . As a result, -R ′ ill (S) ≥ 0 and R ill (S) = 1 -R ′ ill (S) ≥ 0, i.e., R ill (S) is non-negative. Also, by definition, σ 1 = κ (S) σ k . Therefore, R ill (S) = 2 1 k k i=1 σ 2 i -(σ min S ) 2 ≤ 2 1 k σ 2 1 + k-1 k (σ min S ) 2 -(σ min S ) 2 = 2 1 k σ 2 1 -(σ min S ) 2 = 2k (κ(S) 2 -1) (σ min S ) 2 . If κ(S) = ∞, R ill (S) ≤ 2k (κ(S) 2 -1)(σ min S ) 2 = 0 for σ min S > 0, which yields R ill (S) = 0 given that R ill (S) ≥ 0.
Similarly, we have
R ill (S) = 2 1 k k i=1 σ 2 i -(σ min S ) 2 ≥ 2 k-1 k σ 2 1 + 1 k (σ min S ) 2 -(σ min S ) 2 = 2 k-1 k σ 2 1 -(σ min S ) 2 = 2k k-1 (κ(S) 2 -1) (σ min S ) 2 . If R ill (S) = 0, we have κ (S) ≥ 2k k-1 Rill(S)(σ min S ) 2 + 1 = ∞ which yields κ (S) = ∞.
this section cite: []

Section: References
Ref_id:b0 Title: Relations between condition numbers and the convergence of the jacobi method for real positive definite matrices Year: (1985)
Ref_id:b1 Title: Trainable signal encoders that are robust against noise Year: (2024)
Ref_id:b2 Title: Estimating or propagating gradients through stochastic neurons for conditional computation Year: (2013)
Ref_id:b3 Title: Deep learning Year: (2017)
Ref_id:b4 Title:  Year: (2025)
Ref_id:b5 Title: Reconstruction attacks on machine unlearning: Simple models are vulnerable Year: (2024)
Ref_id:b6 Title: Convex optimization Year: (2004)
Ref_id:b7 Title: Instructing text-to-image models using semantic guidance Year: (2023)
Ref_id:b8 Title: The malicious use of artificial intelligence: Forecasting, prevention, and mitigation Year: (2018)
Ref_id:b9 Title: Convex optimization: Algorithms and complexity. Foundations and Trends® in Machine Learning Year: (2015)
Ref_id:b10 Title: Towards making systems forget with machine unlearning Year: (2015)
Ref_id:b11 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b12 Title: Probe-me-not: Protecting pre-trained encoders from malicious probing Year: (2025)
Ref_id:b13 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b14 Title: The use of pre-conditioning in iterative methods for solving linear equations with symmetric positive definite matrices Year: (1968)
Ref_id:b15 Title: Erasing concepts from diffusion models Year: (2023)
Ref_id:b16 Title: Unified concept editing in diffusion models Year: (2024)
Ref_id:b17 Title: Matrix computations Year: (1996)
Ref_id:b18 Title: A study of condition numbers for first-order optimization Year: (2021)
Ref_id:b19 Title: Certified data removal from machine learning models Year: (2019)
Ref_id:b20 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b21 Title: Self-destructing models: Increasing the costs of harmful dual uses of foundation models Year: (2023)
Ref_id:b22 Title: Methods of conjugate gradients for solving linear systems Year: (1952)
Ref_id:b23 Title: Harmful fine-tuning attacks and defenses for large language models: A survey Year: (2024)
Ref_id:b24 Title: Fast and near-optimal diagonal preconditioning Year: (2020)
Ref_id:b25 Title: Structured semidefinite programming for recovering structured preconditioners Year: (2023)
Ref_id:b26 Title: Towards safe self-distillation of internet-scale text-to-image diffusion models Year: (2023)
Ref_id:b27 Title: A method for stochastic optimization Year: (2014)
Ref_id:b28 Title: 3d object representations for fine-grained categorization Year: (2013)
Ref_id:b29 Title: Numerical analysis Year: (2012)
Ref_id:b30 Title: Ablating concepts in text-to-image diffusion models Year: (2023)
Ref_id:b31 Title: The MNIST database of handwritten digits Year: (1998)
Ref_id:b32 Title: The convex analysis of unitarily invariant matrix functions Year: (1995)
Ref_id:b33 Title: Linear and nonlinear programming Year: (1984)
Ref_id:b34 Title: Generative AI misuse: A taxonomy of tactics and insights from real-world data Year: (2024)
Ref_id:b35 Title: House prices -advanced regression techniques Year: (2016)
Ref_id:b36 Title: Variational analysis and applications Year: (2018)
Ref_id:b37 Title: Descent-todelete: Gradient-based methods for machine unlearning Year: (2021)
Ref_id:b38 Title: Almost) Smooth Sailing: Towards numerical stability of neural networks through differentiable regularization of the condition number Year: (2024)
Ref_id:b39 Title: Lectures on convex optimization Year: (2018)
Ref_id:b40 Title: Variational bayesian unlearning. Proc. NeurIPS Year: (2020)
Ref_id:b41 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b42 Title: The matrix cookbook Year: (2008)
Ref_id:b43 Title: Optimal diagonal preconditioning Year: (2024)
Ref_id:b44 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b45 Title: Convex analysis Year: (1970)
Ref_id:b46 Title: Sticking the landing: Simple, lower-variance gradient estimators for variational inference Year: (2017)
Ref_id:b47 Title: Representation noising: A defence mechanism against harmful finetuning Year: (2024)
Ref_id:b48 Title: Immunization against harmful fine-tuning attacks Year: ()
Ref_id:b49 Title: Iterative methods for sparse linear systems Year: (2003)
Ref_id:b50 Title: Ill-conditioning in neural network training problems Year: (1993)
Ref_id:b51 Title: Weight conditioning for smooth optimization of neural networks Year: (2024)
Ref_id:b52 Title:  Year: (2025)
Ref_id:b53 Title: Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models Year: (2023)
Ref_id:b54 Title: Remember what you want to forget: Algorithms for machine unlearning Year: (2021)
Ref_id:b55 Title: Tamper-resistant safeguards for open-weight llms Year: (2024)
Ref_id:b56 Title: YFCC100M: The new data in multimedia research Year: (2016)
Ref_id:b57 Title: Condition numbers and equilibration of matrices Year: (1969)
Ref_id:b58 Title:  Year: (2019)
Ref_id:b59 Title: Puma: Performance unchanged model augmentation for training data removal Year: (2022)
Ref_id:b60 Title: Unlearning isn't deletion: Investigating reversibility of machine unlearning in llms Year: (2025)
Ref_id:b61 Title: Removing rlhf protections in gpt-4 via fine-tuning Year: (2024)
Ref_id:b62 Title: Forgetme-not: Learning to forget in text-to-image diffusion models Year: (2024)
Ref_id:b63 Title: Immunizing textto-image models against malicious adaptation Year: (2024)
Ref_id:b64 Title: Multi-concept model immunization through differentiable model merging Year: (2025)
Ref_id:b65 Title: Learning to obstruct few-shot image classification over restricted classes Year: (2024)
Ref_id:b66 Title: Proceedings of the IEEE Year: (2020)
