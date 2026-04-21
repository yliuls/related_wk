Title: Improving Zero-Shot Adversarial Robustness in Vision-Language Models by Closed-form Alignment of Adversarial Path Simplices
Abstract: Vision-Language Models (VLMs) such as CLIP excel at zero-shot classification due to large-scale pre-training but are vulnerable to adversarial examples. Adversarial fine-tuning robustifies zeroshot models by aligning prediction scores of individual adversaries with their clean counterparts, which typically overlooks intermediate adversarial samples along the adversarial trajectory crossing the decision boundary. Such intermediate adversaries and their vicinity produce informative representations capturing the decision boundary in detail. They can be improved by sampling adversarial candidates from simplices formed by joining two consecutive vertices on the adversarial trajectory and their clean counterpart. However, sampling simplices for adversaries is very costly. To train robust VLM, we overcome these limitations by Taylor expansion and formulating an upper-bound of alignment loss that depends on the Jacobian/Hessian obtained at clean samples. As regions between clean and intermediate adversarial samples capture a larger decision landscape, we robustify VLM by plausible adversaries from simplices by our closed-form formulation equivalent to infinite uniform sampling of the simplex. We obtain state-of-the-art robustness across 15 datasets and diverse vision-language tasks.

Section: Introduction
Despite significant advancements driven by Deep Neural Networks (DNNs) in various areas, Szegedy et al. (2014);  1a shows the zero-shot robust performance of our method (against adversarial samples) vs. adversarial fine-tuning approaches (TeCoA (Mao et al., 2023), PMG (Wang et al., 2024), and FARE (Schlarmann et al., 2024)) across diverse downstream tasks. Fig. 1b shows the principle of sampling simplices formed from vertex x and consecutive intermediate adversaries x+δx,i and x+δx,i+1 along the adversarial trajectory obtained with several steps of gradient ascent. However, sampling such simplices and aligning them individually with x is costly. Fig. 1c shows our model which is much faster as it computes closed-form Σx used in our alignment formula based on Jacobian and Hessian Jg(x), (Hg(x))c . Goodfellow et al. (2015); Dong et al. (2024a) have shown their vulnerability to adversarial examples, which are lowlevel perturbations added to legitimate samples to elicit incorrect class predictions. Such adversarial vulnerabilities also affect Vision-Language Models (VLMs) (Zhang et al., 2022a;Zhao et al., 2023a), posing concerns about deploying VLMs in real-life applications (Díaz-Rodríguez et al., 2023).
To counteract malicious adversaries, a growing body of research seeks to strengthen zero-shot adversarial robustness of VLMs through adversarial fine-tuning (Mao et al., 2023;Wang et al., 2024;Dong et al., 2025), with a focus on CLIPbased architectures (Radford et al., 2021). These methods predominantly align single adversarial predictions-derived from feature-level image-text cosine similarity-with either their benign counterparts or the ground-truth labels. However, such an alignment scheme overlooks the broader spectrum of underlying adversaries, especially intermediate  1c is used to form simplices between vertex x and consecutive adversarial vertices pairs (x + δx,i, x + δx,i+1) for i = 1, . . . , m -1. Next, an efficient alignment is performed between x and all points on simplices to robustify the model.
adversarial samples encountered along the adversarial trajectory obtained during iterative adversary generation (i.e., the path that crosses the decision boundary). Although these intermediate adversaries and their variants encode rich information about the class boundaries, limited efforts have been made to incorporate them explicitly during adversarial finetuning due to additional computational cost. Consequently, such an oversight exposes VLMs to unforeseen adversaries.
In this work, we exploit the disruptive effect of augmented or diversified adversaries (Wang et al., 2022a;Lu et al., 2023;Li & Spratling, 2023;Dong et al., 2024c;Gao et al., 2024). Li & Spratling (2023) explored data augmentation to improve adversarial diversity. Lu et al. (2023) obtained intermediate adversaries along the adversary generation trajectory to achieve cross-VLM attacks. Gao et al. (2024) extended such a mechanism to a triangular region with the vertices of two consecutive adversarial samples and their clean counterpart. Wang et al. (2022a) exploited such geometric information search for adversaries in the black-box setting. However, the above works focus on conducting attacks with such adversarially diverse sets. Their associated computational cost makes them inapplicable to fine-tuning.
Thus, to effectively exploit the rich structure of decision boundaries and adversarial diversity of samples contained by simplices between a sample vertex and consecutive pairs of intermediate adversary samples, we depart from the traditional point-wise alignment of prediction scores. Specifically, Gao et al. (2024) sampled adversarial candidates from 2D simplices with vertices (x, x+δ x,i , x+δ x,i+1 ) for adversarial trajectory (with indices i = 1, . . . , m-1) obtained by the iterative gradient ascent, and used a small number of such samples for attacks. However, explicit sampling of 2D simplices to robustify VLM is prohibitively costly. Further additional cost is due to passing such samples via
1 2 3 4 Perturbation Radius (255⋅ε) 0 10 20 30 40 50 60 Robust Accuracy (%) Robust CLIP (TeCoA) Robust CLIP (PMG) Robust CLIP (FARE) Ours (a) worst-case adversary 1 2 3 4 Perturbation Radius (255⋅ε) 40 50 60 70 80 90 100 Transfer Attack Success Rate (%) Robust CLIP (TeCoA) Robust CLIP (PMG) Robust CLIP (FARE) Ours (b) most transferable adv.  1a but at test time. The CE loss is used to choose the worst-case adversary. Fig. 3b shows the average transfer attack success rate for the most transferable adversaries. For each model, we choose the most transferable attack from the simplices of remaining models.
the backbone to minimize
min θ x∈X ℓ g θ (x), y x + 1 κ δx∈∆ X ∥g θ (x+δ x )-g θ (x)∥ 2 2 Ω(x)
.
(1)
Here, ℓ(•, •) can be cross-entropy (CE) loss, κ = |∆ X | is the number of adversary candidates sampled from a simplex associated with x. Thus, for a dataset with 1M images, and κ = 10 and ascent steps m = 10, one obtains prohibitive 91M images. Such a naive setting is shown in Fig. 1b.
To train robust VLM, we overcome these limitations by a Taylor expansion of g θ (x+δ x ) around x and formulating an upper-bound of approximated alignment loss Ω(•) that depends on the closed-form Σ x instead of naive computations of costly second-order matrix Σx = 1 κ δx∈∆ X δ x δ ⊤ x . We also require easily obtainable Jacobian and Hessian-vector product J g (x), (H g (x)) c p of g θ (•) evaluated at a clean sample x. The entire pipeline of our model, called Adversarial Simplex (AdvSimplex), is shown in Figure 2. Fig. 1c illustrates our closed-form formulation. Fig. 1a demonstrates excellent zero-shot robustness of Ad-vSimplex against existing state-of-the-art adversarial finetuning approaches on several benchmarks. Moreover, Fig. 3a shows an experiment where a CE loss is used at the test time to select the most disruptive adversarial candidate sample from simplices sharing vertex x. Our approach is the most robust model against such adversaries. As TeCoA (Mao et al., 2023), PMG (Wang et al., 2024) and FARE (Schlarmann et al., 2024) perform poorly, this is an indirect proof that such "adversarial simplices" indeed contain adversarial samples. While their strength may differ, they clearly are harmful and thus can be used for robustification. Fig. 3b shows a similar experiment where the most disruptive candidate adversaries were sampled from simplices of all methods other than the tested method to verify cross-model transferability. The figure shows that the attacks sampled from "adversarial simplices" are both strongly adversarial and highly transferable, highlighting their universality.
In summary, our main contributions are as follows:
1. In contrast to existing adversarial fine-tuning approaches that employ sample-wise prediction alignment of adversaries to clean samples, we employ simplex regions formed from vertex x and consecutive adversarial pairs (x + δ x,i , x + δ x,i+1 ) : i = 1, . . . , m-1 from the msteps gradient ascent. While similar schemes were used to create attacks, due to their high computational cost due to sampling, they have not been used in robustification.
2. To alleviate the computational burden of explicit sampling from "adversarial simplices", we derive an upper bound of alignment term Ω(•) in Eq. ( 1) by the use of Taylor expansion, and a scalable upper bound that employs closed-form second-order statistic of points contained within simplices, equivalent to infinitely dense uniform sampling strategy. This alleviates the need to pass numerous candidate adversaries through the backbone. Moreover, minimizing our upper bound is shown as minimizing the upper bound of robust risk.
3. We conduct experiments across 15 datasets and diverse scenarios (e.g., Fig. 1), showing that our method outperforms state-of-the-art adversarial fine-tuning approaches.
this section cite: ['b64', 'b56', 'b69', 'b62', 'b31', 'b22', 'b80', 'b15', 'b56', 'b69', 'b27', 'b61', 'b71', 'b53', 'b50', 'b24', 'b30', 'b50', 'b53', 'b30', 'b71', 'b30', 'b56', 'b69', 'b62']

Section: Background
Related works are in Section 4. CLIP (Radford et al., 2021) enjoys great performance on zero-shot tasks. Its architecture consists of image and text encoders, parameterized by θ and θ ′ . The image encoder f θ : X → R d projects input images x ∈ X into a d-dimensional feature space. Similarly, the text encoder f θ ′ : T → R d maps input textual descriptions t ∈ T into d-dimensional embeddings. By jointly encoding image-text pairs (x, t), CLIP aligns two modalities by the cosine similarity. The probability of assigning an image x to a specific category c ∈ {1, . . . , C} is given as softmax:
g θ (x) c = exp sim(f θ (x), f θ ′ (t c )) C i=1 exp cos(f θ (x), f θ ′ (t i )) ,(2)
where exp(•) is the exponential function, and sim(•, •) represents the cosine similarity. Each text prompt t c = "[Context][CLASS c ]" (e.g., "This is a photo of a [CLASS c ]") is tokenized and embedded by f θ ′ (•), serving as the alignment reference. The predicted probabilities of sample x for all C classes can be represented as
g θ (x) = g θ (x) 1 , . . . , g θ (x) C ⊤ ∈ 0, 1 C .
To enhance zero-shot robustness, adversarial fine-tuning adaptively integrates adversarial samples into the optimization process, aligning their predictions either with their clean Predictions of the CLIP model. J g (•)
Jacobian matrix of g θ (•) (Jacobians stacked for C classes). H g (•) c Hessian matrix of g θ (•) for class c.
this section cite: ['b61']

Section: Ω(•)
The Euclidean alignment loss.
this section cite: []

Section: Ω(•)
Upper bound of Ω(•) with the cross-product term. Ω(•)
Upper bound of Ω(•) without the cross-product term.
counterparts or one-hot ground-truth labels. Given a set X , the standard adversarial fine-tuning approach (i.e., TeCoA (Mao et al., 2023)) is formulated as a minimax optimization:
min θ E x∼X max ∥δx∥ ∞ ≤ϵ ℓ g θ (x + δ x ), y x ,(3)
where δ x is an image-level adversarial perturbation constrained within an ℓ ∞ -norm ball of radius ϵ, and y x represents the class of x. The inner maximization of the crossentropy loss generates adversarial examples by perturbing the prediction, while the outer minimization reduces the empirical risk over these adversaries. Following Mao et al. (2023); Wang et al. (2024), adversaries are generated by the iterative Projected Gradient Descent (PGD) (Madry et al., 2018), which updates perturbed input x(i+1) = x+δ x,i as:
x(i+1) = Π B(x,ϵ) x(i) +α•sign ∇ x(i) ℓ g θ (x (i) ), yx ,(4)
where α denotes the step size, and sign(•) is the sign function. Π B(x,ϵ) ensures the perturbation remains within the ℓ ∞ -norm ball. Adversarial initialization begins with a random perturbation x(0) ∼ x + 0.001 • N (0, I). After m iterations, the final adversarial sample x = x(m) is obtained. The set of the final and intermediate adversarial samples is denoted as
I x = x(i) m i=1 .
Problem definition. Moving beyond traditional robustness evaluations on in-distribution adversarial examples (Croce et al., 2021), we address the more complex zeroshot robustness setting (Mao et al., 2023). In this scenario, adversarial examples are generated with unlimited access to data from previously unseen datasets during inference. The goal for defenders, including our approach, is to maintain robustness against these novel security threats, despite having no prior exposure to such data. From a practical defense view, we assume that text prompts, stored in multi-modal systems, remain unchanged during inference, so we do not attack text.
this section cite: ['b56', 'b56', 'b69', 'b54', 'b13', 'b56']

Section: Proposed Method
Below, we introduce our proposed adversarial fine-tuning approach, enhancing zero-shot adversarial robustness. Our method leverages "adversarial simplices" formed from vertices (x, x+δ x,i , x+δ x,i+1 ) for consecutive i = 1, . . . , m-1
given the m-steps gradient ascent. Table 1 lists common symbols and their explanations.
this section cite: []

Section: Upper-bounding Ω(x)
Below we start with Taylor expansion of g θ (x+δ x ) around x, i.e., g(x+δ
x ) = g(x)+J g (x) δ x + 1 2 δ T x (H g (x)) c δ x C c=1
+ O ∥δ x ∥ 3 , where J g (x) ∈ R C×wh is the Jacobian matrix evaluated for g θ (•) at vertex x. We also have C Hessian matrices H g (x) 1 , . . . , H g (x) C ∈ R wh×wh where wh is width × height of an image. Notice that perturbations ∥δ x ∥ ∞ ≤ ϵ. Thus, we assume that for a sufficiently small ϵ, the remainder term O ∥δ x ∥ 3 of expansion around vertex x is negligible. Moreover, observe that g(x
+ δ x )-g(x) 2 2 ≈ J g (x) δ x + 1 2 δ T x (H g (x)) c δ x C c=1 2 2
, so for a set ∆ X with κ adversarial perturbations we obtain:
Ω(x) ≈ 1 κ δx∈∆ X J g (x) δ x + 1 2 δ T x (H g (x)) c δ x C c=1 2 2 (5) = 1 κ δx∈∆ X J g (x) δ x 2 2 α(x,δx) + 1 4 δ T x (H g (x)) c δ x C c=1 2 2 β(x,δx) + J g (x) δ x , δ T x (H g (x)) c δ x C c=1 γ(x,δx)(6)
≤ 1 κ δx∈∆ X 2α(x, δ x ) + 1 2 β(x, δ x ).(7)
Eq. ( 7) is derived from inequality |x + y| p ≤ 2 p-1 (|a| p + |a| p ) for p ≥ 1 due to convexity. See Appendix C.1.
The upper-bound in Eq. ( 7) does not solve our issue of aggregating over elements of the large set
∆ X . Notice α(x, δ x ) = C c=1 δ x , (J g (x)) ⊤ c,: 2 = C c=1 δ x δ ⊤ x , (J g (x)) ⊤ c,: (J g (x)) c,: .
As the inner product is linear in its argument, we have:
1 κ δx∈∆ X 2α(x, δ x ) = 2 C c=1 1 κ δx∈∆ X δ x δ ⊤ x Σx , (J g (x)) ⊤ c,: (J g (x)) c,: Jg(x,c)(8)
We have now arrived at Eq. ( 8) which depends on Σx (later to be replaced with closed-form Σ x ) and J g (x). The case of expanding β(x, δ x ) is more complex and thus we will upper-bound this term. Notice that β(x, δ
x ) = C c=1 δ T x (H g (x)) c δ x 2 = C c=1 δ x δ ⊤ x , (H g (x)) c 2 βc(x)
.
However, ⟨•, •⟩ 2 is not linear in its arguments, yet we can replace squaring by the Kronecker product ⊗ as follows:
β c (x, δ x ) = vec(δ x δ ⊤ x )⊗vec(δ x δ ⊤ x ), vec((H g (x)) c )⊗vec((H g (x)) c ) (Hg(x))c . (9)
Here, vec(•) simply vectorizes the matrix-shaped input. As each argument of the above inner product contains (wh) 4 elements, this is prohibitive. Thus, we use upper-bound:
1 κ δx∈∆ X 1 2 β(x, δ x ) = 1 2 C c=1 1 κ δx∈∆ X vec(δ x δ ⊤ x )⊗vec(δ x δ ⊤ x ), H g (x, c) ≤ 1 2 C c=1 κ 1 κ δx∈∆ X vec(δ x δ ⊤ x )⊗ 1 κ δx∈∆ X vec(δ x δ ⊤ x ), H g (x, c) = 1 2 κ C c=1 1 κ δx∈∆ X δ x δ ⊤ x Σx , (H g (x)) c 2 . (10
)
Putting together Eq. ( 8) & ( 10), under negligible O ∥δ x ∥ 3 , we readily obtain Ω(x) ≤ Ω(x) ≤ Ω(x):
Ω(x; Σx )= C c=1 Σx , J g (x, c) + 1 4 κ Σx , (H g (x) c 2 +γ terms , Ω(x; Σx )= C c=1 2 Σx , J g (x, c) + 1 2 κ Σx , (H g (x) c 2 . (11
)
Having developed two upper bounds of Ω(x), we now define a closed-form expression for Σ x .
this section cite: []

Section: The ∞-dense sampling of simplex (closed-form Σ x )
To avoid aggregating empirical Σx over κ elements of set ∆ X , we propose the following theorem.
Theorem 3.1. The closed-form expression for Σ x = E pp T over all p in simplex with vertices (x, y, z) is
Σ x = 1 12 x + y + z x + y + z T + 1 12 xx T + yy T + zz T .(12)
Moreover, let Q be the number of vertices (z 1 , . . . , z Q ) of a simplex. The closed-form expression for Σ x = E pp T for higher-order simplices, e.g., tetrahedron (Q = 4 vertices) or pentachoron (Q = 5 vertices) is given as:
Σ x = 1 Q(Q+1) Q i=1 z i z ⊤ i + Q i=1 z i Q i=1 z i ⊤ . (13) Proof. Parameterize a point p ∈ R wh over vertices (x, y, z) as p = αx+βy +γz, α, β, γ ≥ 0, α+β +γ = 1. Expand pp T = (αx+βy+γz)(αx+βy+γz) T and note that E pp T = E α 2 xx T +E β 2 yy T +E γ 2 zz T +E αβ (xy T +yx T )+ E αγ (xz T + zx T ) + E βγ (yz T + zy T ). For a uniform distribution on the simplex {α, β, γ ≥ 0, α + β + γ ≤ 1}, we have E[α] = E[β] = E[γ] = 1 3 , E[α 2 ] = E[β 2 ] = E[γ 2 ] = 1 6 , E[αβ] = E[αγ] = E[βγ] = 1 12 .
Substitute expectations into the expansion to conclude the proof for Q = 3. For higher-order simplices (i.e., Q > 3), one can parameterize p = Q i=1 α i z i , α i ≥ 0, Q i=1 α i = 1, and obtain:
E α i = 1 Q , E α 2 i = 2 Q(Q+1)
, and
E α i α j = 1 Q(Q+1) , i ̸ = j for the underlying Dirichlet distribution. Then one ex- pands E pp T for pp T = Q i,j=1 α i α j z i z ⊤ j .
Eq. ( 12) requires mere aggregation over four outer products of vectors. As long as κ > 4, using the closed form is more efficient and equivalent to evaluating for κ = ∞ set ∆ X .
For efficiency, we use the Hessian-vector product (HVP) in Ω and Ω, i.e., (H g • p) as follows:
E pp T , H g = E p ⊤ (H g •p) (14
) = 1 Q(Q+1) Q i=1 z ⊤ i H g • z i + Q i=1 z i ⊤ H g • Q i=1 z i . Thus, we set Σx , (H g (x)) c 2 = E p ⊤ (H g (x)) c • p 2 .
For a simplex with Q = 3 vertices (one vertex equals 0), this requires only three Hessian-vector products. One HPV evaluation costs the same as 2-4 Jacobian evaluations.
this section cite: []

Section: Our loss function
Following Fig. 1c, we have to align a set of m-1 simplices with the Jacobian and Hessian statistics. Thus, our loss takes the form below:
min θ x∈X ℓ g θ (x), y x + λ m-1 i=1 ω i (x) Ω(x; Σ x,i ),(15)
and Σ x,i is evaluated on vertices (0, δ x,i , δ x,i+1 ) by Eq. ( 12), λ ≥ 0 controls the impact of minimizing the upper bound. Additionally, we can reweight the impact of each simplex by a simple perturbance impact measure of intermediate adversarial vertex x(i) = x+δ x,i :
ω i (x) = 1 τ g θ (x) yx -g θ x(i) yx ,(16)
where
τ = max j∈B g θ (x j ) yx j -g θ x(i) j yx j
simply measures the biggest perturbance for a batch B of samples x.
this section cite: []

Section: Bounding the robust risk
Generally, there exists an inevitable trade-off between natural performance and adversarial robustness (Wang et al., 2024). Below, we study the implications of our design for robust risk (Zhang et al., 2019). Generally, the following three risks are known in adversarial learning:
R nat (g) := E x∼X ℓ g(x), y x , (17
) R rob (g) := E x∼X max ∥δx∥≤ϵ ℓ g(x+δ x ), y ,(18)
R boundary (g; ϵ) := P x∼X ∃ δ x : ∥δ x ∥ ≤ ϵ, g(x) ̸ = g(x+δ x ) ,(19)
where for say the 0-1 loss ℓ(•, •), R nat (g) (the natural risk) is the probability of misclassification on clean data, R rob (g) (the robust risk) is the probability that the strongest perturbation in ∥δ x ∥ ≤ ϵ will cause misclassification. R boundary (g; ϵ) quantifies the fraction of points "within ϵ" of the classifier's decision boundary, i.e., the set of points that can be flipped by a perturbation of size ϵ. The prior knowledge also states that the following bound holds R rob (g) ≤ R nat (g)+R boundary (g; ϵ).
For a single adversarial simplex, our boundary risk becomes:
R boundary (g; ∆ X ) := P x∼X ∃ δ x ∈ ∆ X , g(x) ̸ = g(x+δ x ) ,(20)
We can upper-bound this risk by our defined boundary counter risk, which not only captures decision flips on x but also counts in how many different ways x can be perturbed to cause a decision flip:
R counter (g; ∆ X ) := E x∼X δx∈∆ X g(x) ̸ = g(x+δ x ) . (21
) It is clear that R counter (g; I x ) ≤ R counter (g; ∆ X ) where I x = {x (i) } m i=1
is the set of the final and intermediate adversarial samples of x. As the following holds
R rob (g) ≤ R nat (g)+R boundary (g; ∆ X ) (22) ≤ R nat (g)+R counter (g; I x ) (23) ≤ R nat (g)+R counter (g; ∆ X ),(24)
we are optimizing the upper bound of the robust risk, taking into account counts of successful perturbations per x.
this section cite: ['b69', 'b78']

Section: Related Works
Prediction alignment. As a means of enforcing consistency between model outputs, prediction alignment is widely adopted in machine learning. Originally explored in knowledge distillation (Hinton et al., 2015), where the student network is trained to align its soft predictions with those of a teacher, this concept has since been extended to various domains, including semi-supervised learning (Sohn et al., 2020), unsupervised learning (He et al., 2020), and domain adaptation (Tzeng et al., 2017). Noteworthy are also feature alignment-based domain adaptation (Tas & Koniusz, 2018), few-shot detection and segmentation (Zhang et al., 2022b;Kang et al., 2023;Lu et al., 2024), contrastive learning (Zhang et al., 2025), and misalignment-based anomaly detection (Ding et al., 2025).
In the context of VLMs, prediction alignment plays a critical role in promoting modality consistency and label-space agreement, where recent works leverage alignment losses to improve cross-modal generalization (Jia et al., 2021; Yang  , 2022;Dong et al., 2025). Departing from conventional point-wise alignment, our method introduces robust alignment over adversarial path simplices (sets of adversarial points), promoting stronger robust generalization.
Uni-modal adversarial robustness. The growing use of DNNs in both vision and language tasks (Khan et al., 2022;Wang et al., 2022b;Hu et al., 2024;Zhao & Zhang, 2024) has heightened awareness of their vulnerability to adversarial inputs, stimulating research on defense mechanisms (Bai et al., 2021;Aldahdooh et al., 2022;Xie & Yuille, 2020), among which adversarial training is a very effective paradigm. By iteratively optimizing models against worst-case perturbations, it enhances robustness under attacks (Madry et al., 2018;Zhang et al., 2019;Dong et al., 2024d;b). In this paper, we extend adversarial robustness in the context of multi-modal zero-shot generalization.
Multi-modal adversarial robustness. The computational cost impedes scaling to large VLMs such as CLIP (Radford et al., 2021). To address this limitation, adversarial fine-tuning (Mao et al., 2023), often leveraging Parameter-Efficient Fine-tuning (PEFT) (Jia et al., 2022;Zhou et al., 2022;Ni et al., 2024;Zhu et al., 2025), has attracted increasing attention. Mao et al. (2023) introduced adversarial fine-tuning through text-guided contrastive learning, aligning image-text embeddings for adversarial robustness. To mitigate the potential over-fitting to fine-tuning datasets, Wang et al. (2024) designed a prediction-level regularization guided by natural CLIP, while Schlarmann et al. (2024) proposed an unsupervised adversarial framework. Despite their efficacy, existing approaches adopt a point-wise align-ment that integrates a single adversarial counterpart per clean sample, overlooking the broader spectrum of plausible adversaries in the vicinity of the decision boundary, thus compromising zero-shot robustness against unforeseen adversaries. In contrast, we "incorporate" entire "adversarial simplices" into the robustification process.
this section cite: ['b36', 'b63', 'b34', 'b67', 'b65', 'b72', 'b43', 'b52', 'b82', 'b16', 'b41', 'b27', 'b44', 'b72', 'b37', 'b83', 'b3', 'b2', 'b75', 'b54', 'b78', 'b61', 'b56', 'b86', 'b57', 'b87', 'b56', 'b69', 'b62']

Section: Experiments
Below, we provide our experimental configurations and present our comparisons between our AdvSimplex and other adversarial fine-tuning models across 15 datasets.
this section cite: []

Section: Datasets.
We adopt the setup from Mao et al. (2023); Wang et al. (2024); Schlarmann et al. (2024), where CLIP is adversarially fine-tuned on the ImageNet training set (Deng et al., 2009). Then we assess the zero-shot results of the fine-tuned CLIP on the ImageNet val set and 14 novel datasets. We also investigate our method in medical image analysis and vision-text understanding. See settings in Appendix B.1.
Implementation details. Unless specified otherwise, we use CLIP (Radford et al., 2021) based on ViT-Base/32 (Dosovitskiy et al., 2021), as per studies (Mao et al., 2023;Wang et al., 2024;Schlarmann et al., 2024). For adversary generation during fine-tuning, we employ PGD (Madry et al., 2018) with m = 10 iterations under the ℓ ∞ -norm threat model, the perturbation radius ϵ = 2/255 and the step size α = 1/255. The weighting factor is set to λ = 0.6. During evaluations, we assess the natural and robust performance under three strong white-box adversarial attacks: PGD (Madry et al., 2018) with 20 iterations, CW (Carlini & Wagner, 2017), and Auto-Attack (AA) (Croce & Hein, 2020). All evaluations use adaptive attack schemes for fair comparison. See implementation details in Appendix B.2.
this section cite: ['b56', 'b69', 'b62', 'b14', 'b61', 'b28', 'b56', 'b69', 'b62', 'b54', 'b54', 'b12']

Section: Main Results
Evaluations across 15 datasets. We compare our AdvSimplex with TeCoA (Mao et al., 2023), PMG-FT (Wang et al., 2024), and FARE (Schlarmann et al., 2024) in Tables 2 & 3, where we also provide zero-shot inference on additional 14 datasets, reporting natural performance and robustness against 20-step PGD attacks. AdvSimplex consistently outperforms other models in clean accuracy, with an average improvement of 3.5%, thus approaching standard CLIP (Table 2). While the standard CLIP has nearly zero adv. robustness (Table 3), our AdvSimplex enjoys an average improvement of 4.7% across all datasets in comparison to FARE.
Adversarial fine-tuning of diverse architectures. Table 4 reports zero-shot results on various clip architectures for clean samples and adversarial counterparts across three attack types (ϵ = 2/255): PGD (Madry et al., 2018) (20 steps), CW (Carlini & Wagner, 2017), and Auto-Attack (Croce & Hein, 2020). Our AdvSimplex outperforms other adversarial fine-tuning models under all architectures. Robustness on text-level and bi-level attacks. In addition to image-level attacks, we assess text-level and bi-level attacks. Table 6 reports robust accuracy under these settings. Text-level attacks are evaluated using BERT-Attack (Li et al., 2020) and Gradient-Based Distributional Attack (GBDA) (Guo et al., 2021), while bi-level adversaries are tested using Collaborative Multi-modal Adversarial Attack (Co-Attack) (Zhang et al., 2022a) and Set-level Guidance Attack (SGA) (Lu et al., 2023). Our AdvSimplex outperforms existing adversarial fine-tuning methods on all attack types.
Efficient fine-tuning with VPT. Fine-tuning the full parameter space is computationally expensive for VLMs. Thus, we investigate adversarial fine-tuning with Visual Prompt Tuning (VPT) (Jia et al., 2022), a parameter-efficient strategy with learnable parameters in the token embedding layer. Table 7 shows the zero-shot performance of AdvSimplex under various adversarial configurations, comparing it with other techniques that also use VPT. Our AdvSimplex, even when using VPT for efficiency, outperforms previous works.
this section cite: ['b56', 'b69', 'b62', 'b54', 'b7', 'b12', 'b51', 'b33', 'b80', 'b53']

Section: Extensions to Other Architectures and Tasks
BLIP: Vision-text Understanding. Beyond standard CLIP (Radford et al., 2021), we further explore zero-shot robust-Table 8. BLIP Extension for Vision-Text Understanding. Evaluations on clean and PGD-20 adversarial samples. TR and IR represent the recall@1 for text and image retrieval, respectively. CIDEr measures the similarity of a generated sentence against a set of ground truth sentences for image captioning evaluations.
this section cite: ['b61']

Section: Method
Image-Text Retrieval Image ness under alternative VLM architectures and downstream tasks using BLIP (Li et al., 2022), which combines multiple vision-language understanding tasks. We consider two cross-modal tasks: (i) image-text retrieval using Flickr30k (Plummer et al., 2015), and (ii) image captioning using Nocaps (Agrawal et al., 2019). For adversarial fine-tuning, we adversarially optimize the Image-Text Contrastive (ITC) learning, Image-Text Matching (ITM), and Language Modeling (LM) modules (Li et al., 2021), instead of performing alignment as in Eq. ( 3). Table 8 shows zero-shot results on clean samples and PGD-based adversarial attacks (20 steps) with ϵ = 1/255, where AdvSimplex enjoys great adaptability.
Captioning Clean TR Robust TR Clean IR Robust IR Clean CIDEr Robust CIDEr
Medical CLIP: Medical Diagnosis. Below, we investigate AdvSimplex in the medical imaging domain, where adversarial threats may affect computer-aided diagnostics (Zhao et al., 2023b). We employ a radiology-oriented CLIP variant under the CheXzero paradigm (Tiu et al., 2022) with a ViT-B backbone, again applying adversarial fine-tuning. We measure zero-shot performance on three multi-label radiology benchmarks: ChestX-ray14 (Wang et al., 2017), CheXpert (Irvin et al., 2019), and PadChest (Bustos et al., 2020). We report the Area Under the Curve (AUC) on clean and adversarial samples, where adversaries are generated via 20-step PGD under ϵ = 1/255. Table 9 shows that our method consistently obtains higher AUC scores than prior approaches in clean and adversarial samples. Our method enjoys superior robustness on PadChest, which includes 192 disease categories and numerous uncommon pathologies.
this section cite: ['b49', 'b60', 'b1', 'b48', 'b85', 'b66', 'b70', 'b38', 'b6']

Section: Further Analyses
Below, we analyze the effectiveness and generalizability of our AdvSimplex across diverse settings. Higher-order simplices. According to Theorem 3.1 for simplices with Q > 3 vertices, i.e., a tetrahedron (Q = 4 vertices), Table 10 shows gains for (x, x+δ x,i , x+δ x,i+1 , x+ δ x,i+2 ) for consecutive i = 1, . . . , m -2 simplices from gradient ascent steps over AdvSimplex (Q = 3 vertices).
Cross-product term in the upper bound Ω(x). Recall that we have a cross-product term γ (last term in approx. Ω(x) in Eq. ( 6)) for a better estimation of the upper bound of the prediction gap. Table 11 analyzes its impact on adversarial fine-tuning. A more precise estimation with the cross-product term improves the zero-shot adversarial robustness. The cross-term product is derived using Kronecker operations, resulting in third-order statistics. Such statistics also have a closed-form solution, and the dimensionality of tensors is tensor-sketched to keep calculations fast (Weinberger et al., 2009). See Appendix C.3 for derivations.
Impact of each module. Below we ablate two key components of AdvSimplex: (i) Ω, and (ii) weighting in Eq. ( 16). Table 12 shows that our baseline (first row) follows the surrogate optimization of robust risk (i.e., TRADES (Zhang et al., 2019)) by extending the point-wise cleanadversarial prediction alignment with intermediate adversaries. Despite its simplification, our baseline approach already achieves competitive performance compared to prior adversarial fine-tuning methods. Incorporating "adversarial simplices" yields further gains in both clean and robust accuracy. Our adaptive re-weighting emphasizes that not every simplex is equally adversarial.
Accuracy-robustness trade-off. Striking a balance between natural performance and adversarial robustness is known from uni-modal adversarial learning (Zhang et al., 2019;Dong et al., 2023a). Below, we explore it in the multi-  modal CLIP and zero-shot scenario. We analyze the effect of hyper-parameter λ, which controls the relative weighting of clean sample classification vs. adversarial-clean prediction alignment. Figure 4a shows that increasing λ enhances adversarial robustness yet reduces clean accuracy. Conversely, lowering λ improves zero-shot performance on benign inputs at the cost of reduced robustness. Such a trade-off stems from the optimization of natural and boundary risks.
Performance of sampling vs. closed-form solution. Recall that we introduce an efficient upper bound derived with a closed-form solution replacing sampling "adversarial simplices". Below, we compare our "upper-bound closed-form model" against explicit sampling of "adversarial simplices" for the standard alignment loss. Figure 4b shows that although increasing the sampling amount leads to a gradual improvement in adversarial robustness, it stabilizes around 70 samples and saturates around 100 samples, posing substantial computational training time of 13.6 hours per epoch. In contrast, our AdvSimplex requires merely 4.1 hours per epoch while attaining comparable robustness.
Re-weighting mechanisms for "adv. simplices". Below we compare the use of weights from Eq. ( 16) with (i) uniform weighting vs. (2) linear weighting (i/m) that place greater emphasis on adversaries from later iteration steps. Table 13 shows that our Eq. ( 16) outperforms other variants.
Closed-form vs. sampled "adversarial simplex". Below, we analyze the average robust accuracy of the closed-form "adversarial simplices" vs. sampled "adversarial simplices" (70 samples per simplex). We attack both methods with an index i adversary from PGD-20. Table 14 shows that the closed-form solution enjoys greater zero-shot robustness against adversarial attacks of various step numbers.
Performance w.r.t. sampled "adversarial simplices". In addition to attacks along the generation path, we also evaluate the adversarial robustness against both the worst-case and the most transferable adversaries sampled from "adversarial simplices". Following the setup from Figure 3, worst-case adversaries are obtained from the target CLIP model, and the most transferable adversaries from three other CLIP models. Table 15 shows that our derived closedform upper-bound model enjoys greater robustness against adversaries from "adversarial simplices".
this section cite: ['b73', 'b78', 'b78', 'b19']

Section: Conclusion
Motivated by our analysis of the robustness degradation against underlying adversaries from "adversarial simplices", we have uncovered that the point-wise prediction alignment in robust VLMs leads to weak robustness generalization. Thus, we have explored recent attack strategies to formulate simplices between clean vertex x and consecutive adversarial samples on the gradient ascent path. While sampling such simplices is prohibitive, and aligning such adversarial candidate points is also prohibitive, one may reformulate the problem by minimizing an upper bound of the alignment loss. Our upper bound employs closed-form statistics obtained from the vertices of simplices, the Jacobian and Hessian matrices. We only pass clean samples via the encoder, reducing time complexity, and we achieve "infinite sampling" effect with our formulation during fine-tuning.
Any opinions, findings, conclusions, or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore, and Infocomm Media Development Authority. Piotr Koniusz and Hao Zhu are supported by CSIRO's Science Digital.
this section cite: []

Section: References
Ref_id:b0 Title: Efficient and effective augmentation strategy for adversarial training Year: (2022)
Ref_id:b1 Title: Nocaps: Novel object captioning at scale Year: (2019)
Ref_id:b2 Title: Adversarial example detection for dnn mod-els: A review and experimental comparison Year: (2022)
Ref_id:b3 Title: Recent advances in adversarial training for adversarial robustness Year: (2021)
Ref_id:b4 Title: Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures Year: (2013)
Ref_id:b5 Title: Food-101mining discriminative components with random forests Year: (2014)
Ref_id:b6 Title: Padchest: A large chest x-ray image dataset with multi-label annotated reports Year: (2020)
Ref_id:b7 Title: Towards evaluating the robustness of neural networks Year: (2017)
Ref_id:b8 Title: On evaluating adversarial robustness Year: (2019)
Ref_id:b9 Title: Mind the trojan horse: Image prompt adapter enabling scalable and deceptive jailbreaking Year: (2025)
Ref_id:b10 Title: Describing textures in the wild Year: (2014)
Ref_id:b11 Title: An analysis of singlelayer networks in unsupervised feature learning Year: (2011)
Ref_id:b12 Title: Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks Year: (2020)
Ref_id:b13 Title: Robustbench: a standardized adversarial robustness benchmark Year: (2021)
Ref_id:b14 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b15 Title: Connecting the dots in trustworthy artificial intelligence: From ai principles, ethics, and key requirements to responsible ai systems and regulation Year: (2023)
Ref_id:b16 Title: Learnable expansion of graph operators for multimodal feature fusion Year: (2025)
Ref_id:b17 Title: Visually maintained image disturbance against deepfake face swapping Year: (2021)
Ref_id:b18 Title: Improving adversarially robust few-shot image classification with generalizable representations Year: (2022)
Ref_id:b19 Title: The enemy of my enemy is my friend: Exploring inverse adversaries for improving adversarial training Year: (2023-08)
Ref_id:b20 Title: Restricted black-box adversarial attack against deepfake face swapping Year: (2023)
Ref_id:b21 Title: Toward intrinsic adversarial robustness through probabilistic training Year: (2023)
Ref_id:b22 Title: Survey on adversarial attack and defense for medical image analysis: Methods and challenges Year: (2024-01)
Ref_id:b23 Title: Adversarially robust distillation by reducing the student-teacher variance gap Year: (2024-06)
Ref_id:b24 Title: Robust distillation via untargeted and targeted intermediate adversarial samples Year: (2024-02)
Ref_id:b25 Title: Adversarially robust few-shot learning via parameter codistillation of similarity and class concept learners Year: (2024)
Ref_id:b26 Title: Generalizable and discriminative representations for adversarially robust few-shot learning Year: ()
Ref_id:b27 Title: Stabilizing modality gap & lowering gradient norms improve zeroshot adversarial robustness of vlms Year: (2025)
Ref_id:b28 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b29 Title: Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories Year: (2004)
Ref_id:b30 Title: Boosting transferability in vision-language attacks via diversification along the intersection region of adversarial trajectory Year: (2024)
Ref_id:b31 Title: Explaining and harnessing adversarial examples Year: (2015)
Ref_id:b32 Title: Caltech-256 object category dataset Year: (2007)
Ref_id:b33 Title: Gradientbased adversarial attacks against text transformers Year: (2021)
Ref_id:b34 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b35 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b36 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b37 Title: Longrecipe: Recipe for efficient long context generalization in large language models Year: (2024)
Ref_id:b38 Title: Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison Year: (2019)
Ref_id:b39 Title: Scaling up visual and vision-language representation learning with noisy text supervision Year: ()
Ref_id:b40 Title: Visual prompt tuning Year: ()
Ref_id:b41 Title:  Year: (2022)
Ref_id:b42 Title: Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports Year: (2019)
Ref_id:b43 Title: Distilling self-supervised vision transformers for weaklysupervised few-shot classification & segmentation Year: (2023)
Ref_id:b44 Title: Transformers in vision: A survey Year: (2022)
Ref_id:b45 Title: 3d object representations for fine-grained categorization Year: (2013)
Ref_id:b46 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b47 Title: Biobert: a pre-trained biomedical language representation model for biomedical text mining Year: (2020)
Ref_id:b48 Title: Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems Year: (2021)
Ref_id:b49 Title: Bootstrapping language-image pre-training for unified vision-language understanding and generation Year: (2022)
Ref_id:b50 Title: Data augmentation alone can improve adversarial training Year: (2002)
Ref_id:b51 Title: BERT-ATTACK: adversarial attack against BERT using BERT Year: (2020)
Ref_id:b52 Title: Opening prompt diversity for zero-and few-shot keypoint detection Year: (2024)
Ref_id:b53 Title: Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models Year: (2023)
Ref_id:b54 Title: Towards deep learning models resistant to adversarial attacks Year: (2018)
Ref_id:b55 Title: Fine-grained visual classification of aircraft Year: (2013)
Ref_id:b56 Title: Understanding zero-shot adversarial robustness for largescale models Year: (2023)
Ref_id:b57 Title: PACE: Marrying generalization in parameter-efficient fine-tuning with consistency regularization Year: (2024)
Ref_id:b58 Title: Automated flower classification over a large number of classes Year: (2008)
Ref_id:b59 Title: Cats and dogs Year: (2012)
Ref_id:b60 Title: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models Year: (2015)
Ref_id:b61 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b62 Title: Robust CLIP: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models Year: (2024-07-01)
Ref_id:b63 Title: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems Year: (2020)
Ref_id:b64 Title: Intriguing properties of neural networks Year: (2014)
Ref_id:b65 Title: CNN-based action recognition and supervised domain adaptation on 3d body skeletons via kernel feature maps Year: (2018)
Ref_id:b66 Title: Expert-level detection of pathologies from unannotated chest x-ray images via self-supervised learning Year: (2022)
Ref_id:b67 Title: Adversarial discriminative domain adaptation Year: (2017)
Ref_id:b68 Title: Rotation equivariant cnns for digital pathology Year: (2018)
Ref_id:b69 Title: Pre-trained model guided fine-tuning for zero-shot adversarial robustness Year: (2024)
Ref_id:b70 Title: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases Year: (2017)
Ref_id:b71 Title: Triangle attack: A query-efficient decision-based adversarial attack Year: (2022-02)
Ref_id:b72 Title: Learning bi-directional feature propagation with latent layout modeling for group re-identification Year: (2022-06)
Ref_id:b73 Title: Feature hashing for large scale multitask learning Year: (2009)
Ref_id:b74 Title: Sun database: Large-scale scene recognition from abbey to zoo Year: (2010)
Ref_id:b75 Title: Intriguing properties of adversarial training at scale Year: (2020)
Ref_id:b76 Title: Vision-language pre-training with triple contrastive learning Year: (2022)
Ref_id:b77 Title: Text-guided attention is all you need for zero-shot robustness in vision-language models Year: (2018)
Ref_id:b78 Title: Theoretically principled trade-off between robustness and accuracy Year: (2019)
Ref_id:b79 Title: Attacks which do not kill training make adversarial learning stronger Year: ()
Ref_id:b80 Title: Towards adversarial attack on vision-language pre-training models Year: (2007)
Ref_id:b81 Title: Time-rEversed DiffusioN tEnsor Transformer: A New TENET of Few-Shot Object Detection Year: (2022)
Ref_id:b82 Title: Understanding and mitigating hyperbolic dimensional collapse in graph contrastive learning Year: (2025)
Ref_id:b83 Title: Large language model is not a (multilingual) compositional relation reasoner Year: (2024)
Ref_id:b84 Title: On evaluating adversarial robustness of large vision-language models Year: ()
Ref_id:b85 Title: Clip in medical imaging: A comprehensive survey Year: (2023-08)
Ref_id:b86 Title: Learning to prompt for vision-language models Year: (2022)
Ref_id:b87 Title: Almost-orthogonal parameter spaces for continual learning Year: (2025)
