Title: Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness
Abstract: Adversarial examples reveal critical vulnerabilities in deep neural networks by exploiting their sensitivity to imperceptible input perturbations. While adversarial training remains the predominant defense strategy, it often incurs significant computational cost and may compromise clean-data accuracy. In this work, we investigate an architectural approach to adversarial robustness by embedding group-equivariant convolutions-specifically, rotation-and scale-equivariant layers-into standard convolutional neural networks (CNNs). These layers encode symmetry priors that align model behavior with structured transformations in the input space, promoting smoother decision boundaries and greater resilience to adversarial attacks. We propose and evaluate two symmetry-aware architectures: a parallel design that processes standard and equivariant features independently before fusion, and a cascaded design that applies equivariant operations sequentially. Theoretically, we demonstrate that such models reduce hypothesis space complexity, regularize gradients, and yield tighter certified robustness bounds under the CLEVER (Cross Lipschitz Extreme Value for nEtwork Robustness) framework. Empirically, our models consistently improve adversarial robustness and generalization across CIFAR-10, CIFAR-100, and CIFAR-10C under both FGSM and PGD attacks, without requiring adversarial training. These findings underscore the potential of symmetry-enforcing architectures as efficient and principled alternatives to data augmentation-based defenses.

Section: Introduction
Adversarial robustness, defined as the capacity of deep neural networks to produce consistent predictions under small and often imperceptible input perturbations, remains a fundamental and unresolved challenge in modern machine learning. Adversarial attacks exploit a model's sensitivity to small, norm-bounded input perturbations, leading to incorrect and often high-confidence predictions [1]. These perturbations typically exploit the model's reliance on spurious, non-semantic features that do not align with the true data-generating process [2]. A key contributing factor to this vulnerability is insufficient training data: when datasets are limited in size or diversity, models tend to overfit to superficial statistical patterns, such as background textures or local pixel correlations, rather than learning robust and generalizable representations [3,4].
Adversarial training has become a dominant approach to mitigate this vulnerability. It enhances model resilience by explicitly injecting adversarial examples into the training process, thereby guiding the model to focus on more discriminative, semantically grounded features [5,6]. These adversarial examples expand the effective support of the training distribution, allowing models to develop wider decision margins and improved generalization to perturbed inputs [2,7]. Despite its success, adversarial training is not without limitations: it is computationally expensive, may degrade performance on clean data, and is often specialized to the attack types seen during training. Moreover, it addresses robustness reactively by modifying data rather than proactively by redesigning the model architecture.
This motivates a fundamental question: Can architectural priors alone improve adversarial robustness by encouraging models to align more closely with the geometric structure of data? In this work, we explore this question through the lens of equivariance, the principle that model outputs should transform predictably under known input transformations. In particular, we investigate whether embedding symmetry priors via group-equivariant convolutions can enhance adversarial robustness in convolutional neural networks (CNNs) even in the absence of adversarial training.
Equivariance provides a principled mechanism for enforcing inductive biases that align with underlying symmetries in data. While standard CNNs are translation-equivariant by design, they are not inherently equivariant to other common transformations such as rotations and scalings. Group-equivariant convolutions generalize standard convolutions to be equivariant under larger transformation groups, such as the discrete rotation group P4 or scale groups [8][9][10]. These architectures encode transformation consistency directly into the weight-sharing scheme of the network, allowing the model to process rotated or rescaled inputs without relying on data augmentation. As a result, equivariant CNNs have demonstrated improved sample efficiency, stronger generalization, and greater interpretability across domains such as medical imaging, remote sensing, and physics-informed learning [11,12].
Despite their success in structured learning tasks, the relationship between equivariance and adversarial robustness remains underexplored. Intuitively, adversarial perturbations often introduce changes that lie off the data manifold or violate known symmetries. By constraining the model to respond consistently along group-induced orbits and suppressing sensitivity to off-orbit perturbations, equivariant architectures may provide a natural defense mechanism. This raises a key research question: How does architectural equivariance influence a model's resilience to adversarial perturbations, both theoretically and empirically?
In this paper, we bridge the gap between symmetry enforcement and adversarial robustness by conducting a systematic study of CNN architectures that integrate standard, rotation-equivariant, and scale-equivariant convolutions. We propose two model designs to incorporate equivariant layers and evaluate their robustness properties across a spectrum of adversarial and natural corruption settings. Our main contributions are summarized as follows:
• We present a theoretical analysis demonstrating that equivariant architectures contract the hypothesis space, regularize gradient behavior, and admit tighter certified robustness bounds under the CLEVER (Cross Lipschitz Extreme Value for nEtwork Robustness) framework.
• We propose and compare two symmetry-aware CNN architectures parallel and cascaded that integrate standard, rotation-equivariant, and scale-equivariant convolutional layers. We show that the parallel design better preserves complementary feature spaces and achieves superior robustness. We explore two fusion strategies simple concatenation and weighted summation for combining features from multiple symmetry branches. Our findings indicate that concatenation consistently outperforms weighted fusion in adversarial settings.
• We validate our approach through extensive experiments on CIFAR-10, CIFAR-10C, and CIFAR-100 datasets, using FGSM and PGD attacks. Our results show that equivariant CNNs, particularly the parallel design with combined rotation and scale branches, significantly outperform standard CNNs in adversarial accuracy without requiring adversarial training.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b1', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11']

Section: Related Works
Trustworthy machine learning, which focuses on developing and deploying machine learning models that are not only accurate but also robust, private, fair, and explainable, has attracted active research in recent years . Adversarial robustness, a core pillar of trustworthy ML, addresses the vul-nerability of neural networks to imperceptible perturbations. In this work, we explore the intersection of symmetry-aware architectures and adversarial robustness, drawing from two key research areas.
this section cite: []

Section: Equivariant Neural Networks
Equivariance in neural networks ensures that transformations applied to input data lead to predictable and consistent transformations in the learned representations, aligning models with the inherent symmetries in the data. A landmark advancement in this field is the introduction of Group Equivariant Convolutional Networks (G-CNNs) [8], which extended traditional convolutional operations to group transformations, including rotations [87][88]. These networks demonstrated significant gains in performance and efficiency on symmetry-rich datasets such as MNIST and CIFAR-10. Subsequent progress led to the development of Harmonic Networks, which employed circular harmonics to achieve continuous rotational equivariance [9]. By eliminating the need for discrete approximations, these models improved the flexibility of equivariant frameworks [89]. Further extending these ideas, Scale-Equivariant Steerable Networks were introduced to address scale transformations, enabling the processing of multi-scale inputs without explicit data augmentation [90]. Steerable CNNs provided a versatile framework for handling equivariance under a range of transformations, paving the way for applications in domains such as medical imaging, astrophysics, and 3D object recognition [91].
More recently, spherical equivariance has garnered attention, with spherical CNNs being developed to handle data defined on spherical domains [12][92] [93]. These models have found use in global-context tasks, including climate modeling and astrophysics [11]. Additionally, equivariant networks have been applied in molecular biology, utilizing molecular symmetries to predict chemical properties [94]. Despite these advancements, the susceptibility of equivariant models to adversarial perturbations remains an open area of investigation.
this section cite: ['b7', 'b8', 'b88', 'b89', 'b90', 'b11', 'b92', 'b10', 'b93']

Section: Adversarial Robustness
The discovery of adversarial examples exposed a significant limitation in neural networks, revealing their vulnerability to small, carefully designed perturbations [1]. These adversarial inputs exploit weaknesses in CNN architectures, leading to incorrect predictions. The Fast Gradient Sign Method (FGSM) formalized this issue as a single-step attack based on the direction of the loss gradient [95]. Later, Projected Gradient Descent (PGD) was introduced as a stronger iterative attack method, becoming a benchmark for adversarial testing [5]. Efforts to defend against such attacks have primarily focused on adversarial training, where models are trained using adversarially perturbed data to improve robustness [5][96] [97]. However, this approach often results in reduced accuracy on clean data [2]. To mitigate these trade-offs, researchers have proposed architectural innovations, such as feature denoising modules [98] and preprocessing techniques like compression, resizing, and randomization [99][100] [101], which aim to diminish the effect of adversarial perturbations.
Advanced defenses have also leveraged model uncertainty and interpretability. Randomized smoothing has emerged as a certified defense strategy [102][103] [104], while ensemble methods have demonstrated improved robustness by combining multiple decision boundaries [105] [64]. Despite these promising developments, the potential to integrate the symmetry-preserving principles of equivariant networks into adversarial defense strategies remains largely untapped, leaving a valuable avenue for future research.
this section cite: ['b0', 'b94', 'b4', 'b4', 'b96', 'b1', 'b97', 'b98', 'b100', 'b101', 'b103']

Section: Equivariance and Adversarial Robustness
Neural networks are known to be vulnerable to adversarial perturbations: small, human-imperceptible modifications to input data that cause incorrect predictions with high confidence. This phenomenon is often attributed to the model's overreliance on non-semantic features and irregular decision boundaries. One principled approach to mitigating this sensitivity is to enforce architectural inductive biases aligned with known symmetries of the data distribution. Among such biases, equivariance has emerged as a theoretically grounded and empirically effective mechanism.
this section cite: []

Section: Equivariance in Neural Networks
Definition 1 (Equivariant Function). Let G act on X via T g and on Y ⊆ R k via a representation ρ(g) ∈ Aut(Y). A function f : X → Y is said to be G-equivariant if:
f (T g x) = T g f (x), ∀g ∈ G, x ∈ X .
Standard CNNs are translation-equivariant due to weight sharing across spatial positions. However, they lack equivariance to transformations like rotation and scaling. Group-equivariant CNNs (G-CNNs) generalize convolution to act equivariantly under more general groups G, such as C n , SO(2), or dilation groups, thereby promoting symmetry-aligned representations.
Formally, let G be a group with an associated action on the input space X ⊂ R d , and let ρ : G → Aut(R k ) be a linear representation of G acting on the feature space. A function f : X → R k is said to be equivariant with respect to the group G if it satisfies:
f (g • x) = ρ(g)f (x), ∀g ∈ G,(1)
where g • x denotes the transformed input under the action of g ∈ G. Equivariance ensures that applying a transformation to the input leads to a predictable transformation in the output, thereby promoting stability and consistency in feature representations.
this section cite: []

Section: Adversarial Robustness and Margin Bounds
Definition 2 (Adversarial Robustness). A classifier f : R d → R k is said to be (ε, p)-robust at input x ∈ R d if: f (x + δ) = f (x), ∀δ ∈ R d , ∥δ∥ p ≤ ε. Definition 3 (Margin Function). Let f c (x) denote the logit score of the predicted class c, and f j (x) the score of class j ̸ = c. The class margin is:
g c,j (x) := f c (x) -f j (x).
Definition 4 (Certified Robustness via Lipschitz Bound). Let g c,j be locally Lipschitz with constant L > 0 near x. Then for any δ ∈ R d such that ∥δ∥ p ≤ ε, we have:
|g c,j (x + δ) -g c,j (x)| ≤ L∥δ∥ p .
Consequently, robustness against class j is certified if:
ε (p) c→j ≥ g c,j (x) L .
Definition 5 (CLEVER Bound [106]). Let g c,j be the margin function as above. The CLEVER bound estimates a certified perturbation radius as:
ϵ (p) min (x) := min j̸ =c g c,j (x) L (j) q ,
where 1/p + 1/q = 1, and L (j) q := sup
x ′ ∈Bp(x,r) ∥∇g c,j (x ′ )∥ q
is a data-dependent estimate of the local Lipschitz constant of g c,j in the dual norm.
this section cite: ['b105']

Section: Theoretical Analysis of Adversarial Robustness with Equivariant Convolutions
This section presents a comprehensive theoretical framework for analyzing the adversarial robustness of group-equivariant neural networks. We develop the mathematical foundations necessary for understanding the relationship between equivariance and model sensitivity, formalize certified robustness bounds under Lipschitz constraints, and show how equivariant architectures induce smoother gradients and larger certified margins.
this section cite: []

Section: Mathematical Preliminaries and Equivariant Structures
Definition 6 (Input Space and Model). Let X ⊂ R d be the input space, and let f : X → R k be a neural network mapping inputs to logit vectors. We assume f is differentiable almost everywhere. Definition 7 (Orbit and Quotient Space). The orbit of a point
x ∈ X under G is the set [x] G := {g • x | g ∈ G}.
The quotient space X /G is the collection of distinct orbits in X .
this section cite: []

Section: Definition 8 (Jacobian and Lipschitz Constant).
If f is differentiable at x, the Jacobian is J f (x) := ∇f (x) ∈ R k×d , and the local Lipschitz constant is L(x) := ∥J f (x)∥ 2 .
this section cite: []

Section: Definition 9 (Adversarial Perturbation).
A vector δ ∈ R d is an adversarial perturbation at x if f (x + δ) ̸ = f (x) and ∥δ∥ p ≤ ε.
this section cite: []

Section: Jacobian Structure and Lipschitz Regularity
Definition 10 (Global Lipschitz Continuity). A function f is globally Lipschitz if there exists L > 0 such that:
∥f (x 1 ) -f (x 2 )∥ ≤ L • ∥x 1 -x 2 ∥, ∀x 1 , x 2 ∈ R d . Definition 11 (Jacobian under Equivariance). If f (g • x) = ρ(g)f (x)
, then the Jacobian transforms as:
J f (g • x) = ρ(g)J f (x)Dg -1
, where Dg -1 is the Jacobian of the inverse transformation. Lemma 1 (Jacobian Norm Invariance [107]). If ρ(g) and Dg -1 are orthogonal matrices, then:
∥J f (g • x)∥ 2 = ∥J f (x)∥ 2 .
this section cite: ['b106']

Section: Certified Robustness via CLEVER Bounds
We analyze the certified adversarial robustness of group-equivariant networks through the CLEVER framework [106], which provides a lower bound on the minimum input perturbation required to induce misclassification. We show that equivariance yields gradient invariance over group orbits, leading to consistent and stronger robustness certification. Lemma 2 (Transformation of Margins under Group Equivariance [108]). Let f be G-equivariant, i.e.,
f (g • x) = ρ(g)f (x), where ρ : G → GL(R k ) is a linear representation. Then for any j ̸ = c, g c,j (g • x) = ρ cc (g)f c (x) -ρ jj (g)f j (x)
. Lemma 3 (Gradient Transformation of Margin Function [107]). Differentiating the margin function under the group action yields: ∇g c,j (g • x) = ρ(g)∇g c,j (x)Dg -1 , where Dg -1 ∈ R d×d is the Jacobian of the inverse group action. Theorem 1 (Orbit-Invariance of Margin Gradient Norm). If both ρ(g) and Dg -1 are norm-preserving (e.g., orthogonal matrices), then for all g ∈ G, ∥∇g c,j (g • x)∥ q = ∥∇g c,j (x)∥ q . As a result, the Lipschitz constant of the margin function is invariant across the group orbit:
L (j) q = sup x ′ ∈Bp([x] G ,r) ∥∇g c,j (x ′ )∥ q , where [x] G := {g • x | g ∈ G}.
The orbit-invariance of both the classification margin and its gradient norm establishes a compelling theoretical foundation for the robustness of group-equivariant networks. These models are not merely robust at isolated input points but offer uniform guarantees across entire equivalence classes of inputs linked by symmetry transformations. By construction, group-equivariant architectures preserve margin values consistently under group actions, ensuring that the discriminative separation between classes remains stable across symmetrically transformed instances. Furthermore, they inherently suppress gradient sensitivity in directions aligned with the symmetry structure of the data, effectively filtering out perturbations that respect these invariances. As a result, equivariant networks exhibit tighter and more reliable CLEVER-certified robustness bounds throughout the input space. Detailed proof is provided in Appendix A.1.
this section cite: ['b105', 'b107', 'b106']

Section: Equivariance-Induced Gradient Smoothing
A core source of adversarial vulnerability in neural networks is the irregularity of their input-output mappings, often reflected in sharp or non-smooth gradients. Group-equivariant networks mitigate this by imposing geometric constraints that smooth gradients over symmetric input transformations. The idea of this section is to analyze how group equivariance suppress adversarial vulnerability by smoothing the gradients of the network with respect to its inputs. Specifically, we show that group symmetries induce consistent, low-variance gradient fields along symmetric transformations, and reduce sensitivity to adversarial perturbations that deviate from these structured directions. A key source of adversarial fragility in neural networks stems from the irregularity of their inputoutput mappings, often manifesting as sharp or non-smooth gradients. Group-equivariant models mitigate this issue by embedding geometric constraints that align the model's behavior with inherent symmetries in the data, leading to smoother gradients and more stable decision boundaries.
Definition 12 (Logit Gradient and Jacobian Matrix). Let f : R d → R k be a differentiable classifier, and let f j (x) denote the logit for class j. The Jacobian of f at x is:
J f (x) := ∇f (x) =    ∇f 1 (x) ⊤ . . . ∇f k (x) ⊤    ∈ R k×d ,
where ∇f j (x) ∈ R d denotes the gradient of the j-th logit with respect to the input. Each row of J f (x) characterizes how sensitive a particular output is to infinitesimal changes in different input directions.
Lemma 4 (Gradient Transformation under Group Equivariance [109]). Let f be a G-equivariant function, i.e., f (g • x) = ρ(g)f (x), with ρ(g) a representation and Dg -1 the Jacobian of the inverse group action. Then:
∇f (g • x) = ρ(g) • ∇f (x) • Dg -1 . Definition 13 (Orbit-Averaged Gradient Field).
Define the per-logit gradient vector as ϕ j (x) := ∇f j (x). Then, the orbit-averaged gradient is:
ϕ j (x) := 1 |G| g∈G ∇f j (g • x) = 1 |G| g∈G ρ(g)∇f j (x)Dg -1 .
Lemma 5 (Smoothing via Orbit Averaging [108]). Let δ ∈ R d be a small perturbation. Then:
∥ϕ j (x) -ϕ j (x)∥ ≪ ∥ϕ j (x + δ) -ϕ j (x)∥,
especially when δ is orthogonal to the group orbit [x] G . Thus, orbit-averaging suppresses highfrequency variations in the gradient field. Theorem 2 (Directional Suppression of Off-Orbit Perturbations). Let f : R d → R k be a differentiable function that is equivariant under the action of a compact group G, i.e.,
f (g • x) = ρ(g)f (x) for all g ∈ G,
where ρ(g) ∈ GL(R k ) is a linear representation and g • x is the group action on the input space. Suppose a perturbation vector δ ∈ R d can be decomposed as:
δ = δ G + δ ⊥ , where δ G ∈ T x ([x] G
) lies in the tangent space of the group orbit at x, and δ ⊥ ⊥ T x ([x] G ). Then:
∥∇f (x + δ ⊥ ) -∇f (x)∥ 2 ≫ ∥∇f (x + δ G ) -∇f (x)∥ 2 . In particular, if f is orbit-averaged over G, then ∥∇f (x + δ G ) -∇f (x)∥ 2 → 0,
this section cite: ['b108', 'b107']

Section: and off-orbit perturbations dominate gradient variability.
This theorem formalizes the observation that equivariant networks inherently suppress gradient sensitivity along symmetry-respecting directions, while remaining susceptible to orthogonal, adversarial ones. This anisotropy in gradient variability contributes to improved adversarial robustness and smoother decision boundaries. Equivariance not only constrains functional outputs under transformations but also regularizes local geometry of the model's decision surface. Gradient smoothing and directional suppression enhance robustness by reducing sensitivity to adversarial perturbations, especially those orthogonal to the data manifold's symmetric structure. Detailed proof is provided in Appendix A.2.
this section cite: []

Section: Robustness Analysis of Scale Equivariance
Scale-equivariant neural networks do not satisfy the same assumptions required for certified robustness under isometric transformations such as rotations. Specifically, scale transformations alter the norm of the input, violating the orthogonality condition required in Lemma 1 and Theorem 1. Nevertheless, scale-equivariant architectures contribute to adversarial robustness through a different mechanism namely, gradient smoothing via multi-scale orbit averaging. Let x ∈ R d be an input and G s a finite group of scaling transformations. The orbit of x under G s is defined as:
O s (x) = {T s (x) | s ∈ G s },
where T s (x) = s • x. We define the orbit-averaged gradient of a class logit function ϕ j as:
∇ϕ j (x) = 1 |G s | s∈Gs ∇ϕ j (T s x).
Unlike in the rotation-equivariant case, the norms ∥∇ϕ j (T s x)∥ are not preserved across the orbit, but the averaging process acts as a form of regularization. It reduces the gradient variance across local neighborhoods, thereby smoothing the decision boundary and dampening sensitivity to adversarial perturbations that rely on sharp gradients. Scale-equivariant convolutional neural networks (CNNs) achieve robustness by explicitly enforcing equivariance across multiple spatial scales. This is typically done using scale-group convolutions, defined as:
[Φf ](x) = s∈Gs ψ s * f (T -1 s x),
where ψ s is the filter bank corresponding to scale s, * denotes convolution, and
T s (x) = s • x.
The output is a scale-indexed feature map that captures the input structure across different resolutions.
This structure induces a smoothing effect both in the feature and gradient spaces. Specifically, consider the aggregated feature response at a given layer:
h(x) = s∈Gs w s • ϕ s (x), with ϕ s (x) = ψ s * f (T -1 s x),
and its gradient with respect to the input:
∇h(x) = s∈Gs w s • ∇ϕ s (x).
Although the gradient norms ∥∇ϕ s (x)∥ scale with s, their aggregation smooths the overall gradient field by suppressing high-frequency components. This is analogous to a low-pass filter in the frequency domain and reduces the model's vulnerability to adversarial perturbations that exploit sharp local gradient changes. While scale-equivariant models fall outside the domain of certified robustness guarantees derived under norm-preserving assumptions, they introduce robustness via a complementary mechanism: smoothing the activation and gradient fields through multi-scale aggregation. This mechanism stabilizes the model's output under input perturbations and effectively regularizes its sensitivity, promoting robustness in practice.
this section cite: []

Section: Equivariance Enhanced Architectural Designs

this section cite: []

Section: Group Equivariant Convolutions
Group Equivariant Convolutional Networks (G-CNNs) generalize standard convolutional architectures by incorporating symmetry priors directly into the model design [8]. These networks are constructed to preserve equivariance under transformations defined by a group G, such as translations, rotations, or scalings. To achieve this, the conventional convolution operation is replaced by a group convolution, which aggregates features across group-transformed versions of both the input and the filters. In this work, we focus on two widely applicable instances: rotation-equivariant and scale-equivariant convolutions. Detailed formulations of these operations are provided in Appendix C.1.
this section cite: ['b7']

Section: Equivariance-Enforced Architectural Designs
We investigate two architectural strategies that integrate standard, rotation-equivariant, and scaleequivariant convolutions to enhance robust feature extraction: the parallel design and the cascaded design. Each approach offers a distinct balance between representational diversity and computational efficiency. Comprehensive architectural details are provided in Appendix C.2.
this section cite: []

Section: Experiments and Discussion
This section details the experimental setup used to evaluate the impact of adding rotation-and scale-equivariant convolutions on adversarial robustness and generalization. The experiments were conducted using three widely recognized datasets CIFAR-10, CIFAR-100, and CIFAR-10C to ensure a comprehensive evaluation of adversarial robustness, and generalization under natural corruptions. CIFAR-10C [110] is a variant of CIFAR-10 designed to evaluate corruption robustness. It includes 19 types of natural corruptions (e.g., Gaussian noise, motion blur, fog, and pixelation) applied at five levels of severity.
this section cite: ['b109']

Section: Models
To investigate the impact of equivariance on adversarial robustness, we designed and evaluated five CNN architectures with varying symmetry-aware modifications.
Baseline Standard CNN serves as the benchmark model, implemented with either 4 or 10 convolutional layers. Parallel GCNN replaces the first convolutional layer with two parallel branches: a standard convolution branch and a rotation-equivariant branch based on the discrete group P4.
Parallel GCNN with Rotation-and Scale-Equivariant Branches extends the above by introducing a third scale-equivariant branch, enabling the model to process inputs across multiple geometric transformations. Cascaded GCNN adopts a sequential structure, where the input is first processed by a rotation-equivariant layer, followed by standard convolutions. Weighted Parallel GCNN uses the same three-branch structure as the previous parallel design but replaces feature concatenation with learnable fusion weights optimized during training.
this section cite: []

Section: Comparison of Adversarial Robustness under FGSM and PGD Attacks
We evaluated five models Baseline Standard CNN, Parallel GCNN, Parallel GCNN with Rotationand Scale-Equivariant Branch, Cascaded GCNN, and Weighted Parallel GCNN on CIFAR-10 and CIFAR-100 datasets under FGSM and PGD attacks. To ensure a comprehensive understanding of the impact of network depth on robustness, we experimented with both 4-layer and 10-layer CNN architectures for all models.
In Figure 1, we present the adversarial robustness comparison of five models using 4-layer architectures on CIFAR-10 and CIFAR-100. The evaluation considers adversarial accuracies under FGSM and PGD attacks across a range of perturbation magnitudes (ϵ). For CIFAR-10, the Parallel GCNN with Rotation and Scale Branch exhibited the highest robustness. The Parallel GCNN also performed well, but its robustness declined more rapidly compared to the combined model. On CIFAR-100, the overall robustness was lower due to increased class diversity and complexity. The Parallel GCNN with Rotation-and Scale-Equivariant Branch remained the most robust particularly at higher perturbations. The Parallel GCNN showed competitive performance at low perturbations but lagged behind the combined model, particularly at higher perturbations.
In Figure 2, we consider 10-layer architectures on CIFAR-10 and CIFAR-100 datasets. On both CIFAR-10 and CIFAR-100, the Parallel GCNN with Rotation and Scale-Equivariant Branch maintained the highest adversarial robustness across all perturbation levels. The Parallel GCNN with Rotation-Equivariant Branch also demonstrated strong robustness, though it was consistently outperformed by the combined model. Both the Cascaded GCNN and Weighted Parallel GCNN showed limited robustness, with adversarial accuracies dropping below 15% for FGSM and almost negligible for PGD attacks at higher perturbation levels.
To validate our theoretical framework under strict symmetry constraints, we evaluated fully equivariant architectures where all convolutional layers are equivariant, without any standard convolution branches. The 10-layer fully equivariant model achieves 73.01% FGSM and 64.96% PGD accuracy  at ε = 0.01 on CIFAR-10, confirming that orbit-invariant gradient regularization compounds beneficially when symmetry is enforced end-to-end. Complete results for fully equivariant architectures are provided in Appendix D.
Our equivariant models achieve these robustness improvements without adversarial training. For context, we compare our 10-layer equivariant model against a standard CNN trained with PGD adversarial training in Appendix E. Table 1 presents the performance analysis of various models on CIFAR-10C under a range of corruption types and perturbation levels. The Parallel GCNN model consistently achieved the best performance across most corruption types, especially for lower perturbation thresholds. Parallel GCNN with Rotation and Scale Equivariance does not perform so well compared with baseline model under the data corruption. We assess the robustness of the proposed Parallel GCNN models using the maximum invariant perturbation metric [111], which quantifies the largest input perturbations a model can tolerate without altering the model's prediction. To further understand the contribution of each equivariant convolutional module, we perform ablation studies under default settings. The experimental details and visualizations of these studies are provided in Appendix F.
this section cite: ['b110']

Section: Conclusion
In this work, we conducted a systematic investigation into the role of architectural symmetry enforcement in improving adversarial robustness. By incorporating rotation-and scale-equivariant convolutions into standard CNNs, we demonstrated that symmetry-aware models could achieve improved resilience against adversarial attacks without relying on adversarial training or extensive data augmentation. Our theoretical analysis showed that equivariant architectures reduced hypothesis space complexity, regularized gradient behavior, and yielded tighter CLEVER-certified robustness bounds. These models consistently preserved classification margins under group transformations and suppressed gradient sensitivity in directions aligned with the data manifold's symmetry structure. Future work could extend these insights to larger-scale datasets, broader threat models, and more expressive network architectures.
this section cite: []

Section: References
Ref_id:b0 Title: Intriguing properties of neural networks Year: (2013)
Ref_id:b1 Title: Robustness may be at odds with accuracy Year: (2019)
Ref_id:b2 Title: Improving alignment and robustness with circuit breakers Year: (2024)
Ref_id:b3 Title: Rethinking model scaling for convolutional neural networks Year: (2019)
Ref_id:b4 Title: Towards deep learning models resistant to adversarial attacks Year: (2018)
Ref_id:b5 Title: Neural polarizer: A lightweight and effective backdoor defense via purifying poisoned features Year: (2023)
Ref_id:b6 Title: Adversarially robust distillation by reducing the student-teacher variance gap Year: (2025)
Ref_id:b7 Title: Group equivariant convolutional networks Year: (2016)
Ref_id:b8 Title: Harmonic networks: Deep translation and rotation equivariance Year: (2017)
Ref_id:b9 Title: Diffusion models for imperceptible and transferable adversarial attack Year: (2024)
Ref_id:b10 Title: 3d object classification and retrieval with spherical cnns Year: (2018)
Ref_id:b11 Title: Spherical cnns Year: (2018)
Ref_id:b12 Title: Flexible, efficient, and stable adversarial attacks on machine unlearning Year: (2025)
Ref_id:b13 Title: Explainability-guided defense: Attribution-aware model refinement against adversarial data attacks Year: (2025)
Ref_id:b14 Title: Efficient federated learning with timely update dissemination Year: ()
Ref_id:b15 Title: Efficient federated learning with heterogeneous data and adaptive dropout Year: (2025)
Ref_id:b16 Title: Privacy-preserving publishing of multilevel utility-controlled graph datasets Year: (2018)
Ref_id:b17 Title: Robust meta network embedding against adversarial attacks Year: (2020)
Ref_id:b18 Title: Adversarial attacks on deep graph matching Year: (2020-12)
Ref_id:b19 Title: Robust network alignment via attack signal scaling and adversarial perturbation elimination Year: (2021)
Ref_id:b20 Title: Expressive 1-lipschitz neural networks for robust multiple graph learning against adversarial attacks Year: (2021)
Ref_id:b21 Title: Integrated defense for resilient graph matching Year: (2021)
Ref_id:b22 Title: Adversarial attack against cross-lingual knowledge graph alignment Year: (2021-11)
Ref_id:b23 Title: Validating the lottery ticket hypothesis with inertial manifold theory Year: (2021)
Ref_id:b24 Title: Cross-lingual entity alignment with adversarial kernel embedding and adversarial knowledge translation Year: (2021)
Ref_id:b25 Title: Unsupervised adversarial network alignment with reinforcement learning Year: (2022)
Ref_id:b26 Title: Input-agnostic certified group fairness via gaussian parameter smoothing Year: (2022)
Ref_id:b27 Title: Prompt certified machine unlearning with randomized gradient smoothing and quantization Year: (2022-12-09)
Ref_id:b28 Title: Accelerated federated learning with decoupled adaptive optimization Year: (2022)
Ref_id:b29 Title: Federated fingerprint learning with heterogeneous architectures Year: (2022-12-01)
Ref_id:b30 Title: From distributed machine learning to federated learning: A survey Year: (2022)
Ref_id:b31 Title: Fast federated machine unlearning with nonlinear functional theory Year: (2023)
Ref_id:b32 Title: Dimension-independent certified neural network watermarks via mollifier smoothing Year: (2023)
Ref_id:b33 Title: Federated learning of large language models with parameter-efficient prompt tuning and adaptive optimization Year: (2023-10)
Ref_id:b34 Title: Multi-job intelligent scheduling with cross-device federated learning Year: (2023)
Ref_id:b35 Title: FedASMU: Efficient asynchronous federated learning with dynamic staleness-aware model update Year: (2024)
Ref_id:b36 Title: AEDFL: Efficient asynchronous decentralized federated learning with heterogeneous devices Year: (2024)
Ref_id:b37 Title: Effective federated graph matching Year: (2024)
Ref_id:b38 Title: Advancing certified robustness of explanation via gradient quantization Year: (2024)
Ref_id:b39 Title: Fisher information-based efficient curriculum federated learning with large language models Year: (2024)
Ref_id:b40 Title: Efficient federated learning using dynamic update and adaptive pruning with momentum on shared server data Year: (2024)
Ref_id:b41 Title: Clustering large attributed graphs: An efficient incremental approach Year: (2010)
Ref_id:b42 Title: Graph clustering based on structural/attribute similarities Year: (2009)
Ref_id:b43 Title: Clustering large attributed graphs: A balance between structural and attribute similarities Year: (2011)
Ref_id:b44 Title: Clustering analysis in large graphs with rich attributes Year: (2012)
Ref_id:b45 Title: Clustering large attributed information networks: An efficient incremental computing approach Year: (2012)
Ref_id:b46 Title: Representation learning and nature encoded fusion for heterogeneous sensor networks Year: (2019)
Ref_id:b47 Title: Congestion aware dynamic user association in heterogeneous cellular network: A stochastic decision approach Year: (2014)
Ref_id:b48 Title: Enhanced robustness by symmetry enforcement Year: (2024)
Ref_id:b49 Title: Social influence based clustering of heterogeneous information networks Year: (2013)
Ref_id:b50 Title: Trust management in service provision networks Year: (2013-07-02)
Ref_id:b51 Title: Residency aware inter-VM communication in virtualized cloud: Performance measurement and analysis Year: (2013-07-02)
Ref_id:b52 Title: Activity-edge centric multi-label classification for mining heterogeneous information networks Year: (2014)
Ref_id:b53 Title: Reliable and resilient trust management in distributed service provision networks Year: (2015)
Ref_id:b54 Title: Social influence based clustering and optimization over heterogeneous information networks Year: (2015)
Ref_id:b55 Title: Integrating vertex-centric clustering with edge-centric clustering for meta path graph analysis Year: (2015)
Ref_id:b56 Title: Policy-driven autonomic configuration management for nosql Year: (2015-07-02)
Ref_id:b57 Title: Fast iterative graph computation with resource aware graph parallel abstractions Year: (2015)
Ref_id:b58 Title: Analyzing enterprise storage workloads with graph modeling and clustering Year: (2016)
Ref_id:b59 Title: Innovative Mining, Processing, and Application of Big Graphs Year: (2017)
Ref_id:b60 Title: Density-adaptive local edge representation learning with generative adversarial network multi-label edge classification Year: (2018)
Ref_id:b61 Title: Density-aware local siamese autoencoder network embedding with autoencoder graph clustering Year: (2018)
Ref_id:b62 Title: Explaining the behavior of neuron activations in deep neural networks Year: (2021)
Ref_id:b63 Title: Improving robustness of deep neural networks via large-difference transformation Year: (2021)
Ref_id:b64 Title: Looking beyond content: Modeling and detection of fake news from a social context perspective Year: (2022)
Ref_id:b65 Title: Dual adversarial learning based network alignment Year: (2019-11)
Ref_id:b66 Title: Enhancing collaborative filtering with multilabel classification Year: (2019)
Ref_id:b67 Title: Integrating local vertex/edge embedding via deep matrix fusion and siamese multi-label classification Year: (2019-12)
Ref_id:b68 Title: Semi-supervised classificationbased local vertex ranking via dual generative adversarial nets Year: (2019-12)
Ref_id:b69 Title: Approximate deep network embedding for mining large-scale graphs Year: (2019)
Ref_id:b70 Title: Diverse and informative dialogue generation with context-specific commonsense knowledge awareness Year: (2020-10)
Ref_id:b71 Title: TopicKA: Generating commonsense knowledgeaware dialogue responses towards the recommended topic fact Year: (2021)
Ref_id:b72 Title: Unsupervised multiple network alignment with multinominal GAN and variational inference Year: (2020)
Ref_id:b73 Title: Layer-wise entropy analysis and visualization of neurons activation Year: (2019)
Ref_id:b74 Title: Dense cross-connected ensemble convolutional neural networks for enhanced model robustness Year: (2024)
Ref_id:b75 Title: Bridging interpretability and robustness using lime-guided model refinement Year: (2024)
Ref_id:b76 Title: Enhancing adversarial robustness of deep neural networks through supervised contrastive learning Year: (2024)
Ref_id:b77 Title: Explainabilitydriven defense: Grad-cam-guided model refinement against adversarial threats Year: (2025)
Ref_id:b78 Title: Multiscale unrectified push-pull with channel attention for enhanced corruption robustness Year: (2025)
Ref_id:b79 Title: Expert-guided explainable few-shot learning for medical image diagnosis Year: (2025)
Ref_id:b80 Title: Improving collaborative filtering with social influence over heterogeneous information networks Year: (2020)
Ref_id:b81 Title: Towards a better understanding of linear models for recommendation Year: (2021)
Ref_id:b82 Title: Promoting shape bias in cnns: Frequency-based and contrastive regularization for corruption robustness Year: (2025)
Ref_id:b83 Title: Knowledge-aware dialogue generation via hierarchical infobox accessing and infobox-dialogue interaction graph network Year: (2021)
Ref_id:b84 Title: More is better: Enhancing opendomain dialogue generation via multi-source heterogeneous knowledge Year: (2021-11)
Ref_id:b85 Title: Maximal directed quasi-clique mining Year: (2022)
Ref_id:b86 Title: Continuous rotation group equivariant network inspired by neural population coding Year: (2024)
Ref_id:b87 Title: Symmetry and generalisation in machine learning Year: (2025)
Ref_id:b88 Title: On the utility of equivariance and symmetry breaking in deep learning architectures on point clouds Year: (2025)
Ref_id:b89 Title: Deep scale-spaces: Equivariance over scale Year: (2019)
Ref_id:b90 Title: General e (2)-equivariant steerable cnns Year: (2019)
Ref_id:b91 Title: Current symmetry group equivariant convolution frameworks for representation learning Year: (2024)
Ref_id:b92 Title: Ads-gnn-a conformally equivariant graph neural network Year: ()
Ref_id:b93 Title: Schnet: A continuous-filter convolutional neural network for modeling quantum interactions Year: (2017)
Ref_id:b94 Title: Explaining and harnessing adversarial examples Year: (2014)
Ref_id:b95 Title: Adversarial attacks in explainable machine learning: A survey of threats against models and humans Year: (2025)
Ref_id:b96 Title: Race: Robust adversarial concept erasure for secure text-to-image diffusion model Year: (2024)
Ref_id:b97 Title: Feature denoising for improving adversarial robustness Year: (2019)
Ref_id:b98 Title: Countering adversarial images using input transformations Year: (2018)
Ref_id:b99 Title: Black-box access is insufficient for rigorous ai audits Year: (2024)
Ref_id:b100 Title: Navigating the safety landscape: Measuring risks in finetuning large language models Year: (2024)
Ref_id:b101 Title: Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples Year: (2018)
Ref_id:b102 Title: Enhancing the transferability of adversarial attacks via multi-feature attention Year: (2025)
Ref_id:b103 Title: Adversarially robust object detection via deviation calibration and content preservation Year: (2025)
Ref_id:b104 Title: Ensemble adversarial training: Attacks and defenses Year: (2018)
Ref_id:b105 Title: Evaluating the robustness of neural networks: An extreme value theory approach Year: (2018)
Ref_id:b106 Title: Symmetryadapted representation learning Year: (2019)
Ref_id:b107 Title: On invariance and selectivity in representation learning Year: (2016)
Ref_id:b108 Title: Unsupervised learning of invariant representations Year: (2016)
Ref_id:b109 Title: Benchmarking neural network robustness to common corruptions and perturbations Year: (2019)
Ref_id:b110 Title: Maximally invariant data perturbation as explanation Year: (2018)
Ref_id:b111 Title: The national research platform: Stretched, multi-tenant, scientific kubernetes cluster Year: (2025)
