Title: Robust SuperAlignment: Weak-to-Strong Robustness Generalization for Vision-Language Models
Abstract: Numerous well-established studies have demonstrated the superhuman capabilities of modern Vision-Language Models (VLMs) across a wide range of tasks. However, growing is the doubt about the continuing availability of reliable high-quality labeling (supervision) from human annotators, leading to stagnation of the model's performance. To address this challenge, "superalignment" employs the so-called weak-to-strong generalization paradigm, where the supervision from a weak model can provide generalizable knowledge for a strong model. While effective in aligning knowledge for clean samples between the strong and weak models, the standard weak-to-strong approach typically fails to capture adversarial robustness, exposing strong VLMs to adversarial attacks. This inability to transfer adversarial robustness is because adversarial samples are normally missing in the superalignment stage. To this end, we are the first to propose the weak-to-strong (adversarial) robustness generalization method to elicit zero-shot robustness in large-scale models by an unsupervised scheme, mitigating the unreliable information source for alignment from two perspectives: alignment re-weighting and source guidance refinement. We analyze settings under which robustness generalization is possible. Extensive experiments across various vision-language benchmarks validate the effectiveness of our method in numerous scenarios, demonstrating its plug-and-play applicability to large-scale VLMs.* The research work was done during Junhao's internship at TikTok Singapore.

Section: Introduction
Vision-Language Models (VLMs) enjoy remarkable prediction capabilities, frequently surpassing human performance on diverse multimodal tasks, ranging from zero-shot recognition to multimodal reasoning [60,42,49] to task unlearning [24]. Despite these outstanding achievements, they suffer from an emerging issue: as models become highly capable, human annotators fail to reliably guide and/or evaluate their outputs, especially given the complexity and ambiguity inherent in visionlanguage paired data due to inherent limitations in cognitive capacity and scalability [7,4]. This fundamental limitation has prompted the super-intelligence alignment (superalignment) [41], a conceptualized framework to align (future) superhuman models with human values and goals based on limited human supervision. One seminal approach toward superalignment is weak-to-strong generalization [5], which explores supervision from a weak (lightweight) teacher model to guide a strong (large-scale) student model, enabling the student to generalize beyond the original limitations of its supervision source (reference model) on unforeseen data.
However, while existing weak-to-strong generalization methods focus on distilling knowledge from a teacher model, we demonstrate that robustness, a distinct form of knowledge, cannot be transferred into a VLM even when guided by a robust teacher 1 . This limitation inherently exacerbates the vulnerability to adversarial examples-original images with added imperceptible artifacts leading to erroneous predictions with high confidence [65]. In addition to single-modal backbones [21,20,23,15,22,14], these vulnerabilities are especially critical in VLMs, where the correctness of language reasoning heavily relies on visual inputs: poisoned images may compound errors and propagate catastrophic failures through the model's output, fundamentally compromising its reliability for real-world deployment [72,51,46,47].
Through systematic analysis, we are the first to identify the root cause of robustness transfer failure presenting as a mismatch in learning objectives arising from the data employed in weak-to-strong generalization models. Our analysis in Section 3.2 reveals that integrating adversarial examples-rather than their clean counterparts-into learning can induce robustness in the strong student model, even without explicit supervision from an adversarially robust teacher. Motivated by this insight, we propose the first adversarially robust weak-to-strong generalization framework, termed Adv-W2S, designed to elicit robust knowledge from the strong student model in an unsupervised scheme. Specifically, our method mitigates unreliable supervision signals from the weak teacher model through two complementary mechanisms: alignment re-weighting and source guidance refinement. To quantify the reliability of guidance of the weak teacher, we introduce an instance-wise re-weighting mechanism based on prediction entropy, adaptively modulating the emphasis placed on robust knowledge alignment between the teacher and student VLMs. Additionally, in order to further reduce reliance on potentially erroneous guidance signals, we design an adaptive refinement scheme for prediction reference, generating benign inputs that can be considered as improved guidance for robust weak-tostrong generalization. Our theoretical analysis establishes two core claims: (i) low prediction entropy enforced on the student promotes larger classification margins and robustness, and (ii) alignment toward refined source guidance strictly improves robustness generalization during superalignment.
Extensive experiments conducted across diverse network architectures and evaluation scenarios demonstrate that our proposed Adv-W2S framework achieves superior zero-shot classification performance in both clean and robust accuracy compared to state-of-the-art adversarial fine-tuning methods. Moreover, we show that the robustness induced by Adv-W2S can effectively transfer to a broad array of downstream vision-language tasks-including image captioning, visual question answering, hallucination mitigation, and chain-of-thought reasoning-via a plug-and-play replacement of the original vision encoder with our robustly aligned encoder. Our empirical analyses further reveal that the in-distribution robustness initially obtained via unsupervised alignment can be further improved through supervised fine-tuning. To our best knowledge, this work represents the first systematic exploration of robust superalignment, providing a promising pathway toward building robust and aligned foundation VLMs for artificial general intelligence applications.
Our core contributions are summarized as follows:
1. We reveal that standard weak-to-strong generalization schemes fail to transfer VLM robustness.
Through systematic analyses, we identify-for the first time-that the root cause of this failure is the absence of adversarial examples in alignment objectives of weak-to-strong generalization.
2. In contrast to adversarial fine-tuning from scratch, we propose Adv-W2S, the first adversarially robust weak-to-strong generalization framework to elicit robust knowledge from a weak student VLM by an unsupervised scheme. We also investigate unreliable source guidance mitigation from two complementary perspectives: alignment re-weighting and source guidance refinement. Theoretical analyses characterize the conditions under which robustness generalization is possible.
3. We conduct extensive experiments across 20 datasets spanning diverse vision-language tasks (including visual question answering and captioning) across various scenarios, demonstrating that our Adv-W2S consistently outperforms state-of-the-art adversarial fine-tuning approaches in terms of natural performance and zero-shot robustness, while supporting plug-and-play integration.
2 Related Works Foundation VLMs. Vision-language pre-training learns joint visual-textual representations to improve downstream task performance [73]. CLIP [60] pioneered large-scale vision-language contrastive learning, enabling zero-shot transfer to a wide range of vision tasks. Subsequent foundation VLMs, e.g., BLIP [43], OpenFlamingo [2], and LLaVA [49,48], have demonstrated strong multimodal understanding and generalization. Our study concentrates on improving the robustness of the widely adopted CLIP model, generalizing robustness to other foundation VLMs via a plug-and-play replacement of their vision encoders with our robust counterpart. To mitigate computational costs and potential overfitting associated with full fine-tuning, we explore Parameter-Efficient Fine-Tuning (PEFT) [36,35,56,76,74] within our robust weak-to-strong generalization framework.
Weak-to-strong generalization for superalignment. Inspired by the challenge of aligning superhuman models via weaker supervision (superalignment), Burns et al. [5] first explored fine-tuning large-scale models using weak-model supervision. This paradigm, termed weak-to-strong generalization, improves the generalizability of the strong student model via a weak teacher model, distinct from standard knowledge distillation [34]. Lang et al. [40] established the error bound for weak supervision, highlighting the significance of pseudolabel correction. Our study explores predictionbased pseudolabel refinement by generating more benign inputs for improved supervision via an inverse procedure of adversarial generation. Subsequent works studied the inherent reliability of weak teacher supervision [29,30]. However, prior works focus on single-modal vanilla knowledge transfer for specific tasks. In contrast, we address an underexplored problem: robust weak-to-strong generalization for VLMs, generalizing natural and robust knowledge across diverse multimodal tasks.
Adversarial robustness of VLMs. The increasing security risks posed by adversarial examples [37,70] have spurred extensive research on defense schemes for VLMs [63,75,3,17,18,16,19]. Adversarial fine-tuning, a leading defense paradigm, augments training data with adversaries based on naturally pre-trained VLMs, e.g., CLIP [60]. Mao et al. [55] first introduced adversarial fine-tuning within a contrastive learning framework to align adversarially perturbed image embeddings with their text counterparts. Schlarmann et al. [63] developed an unsupervised adversarial fine-tuning strategy by preserving the features of the original CLIP model. Unlike prior adversarial fine-tuning approaches that exclusively rely on training data, we focus on a weak-to-strong generalization framework guided by a weak VLM to elicit the strong robustness generalization capability from a strong VLM across diverse downstream tasks, offering a promising solution toward the superalignment challenge for foundation VLMs.
this section cite: ['b59', 'b41', 'b48', 'b23', 'b6', 'b3', 'b40', 'b4', 'b64', 'b20', 'b19', 'b22', 'b14', 'b21', 'b13', 'b71', 'b50', 'b45', 'b46', 'b72', 'b59', 'b42', 'b1', 'b48', 'b47', 'b35', 'b34', 'b55', 'b75', 'b73', 'b4', 'b33', 'b39', 'b28', 'b29', 'b36', 'b69', 'b62', 'b74', 'b2', 'b16', 'b17', 'b15', 'b18', 'b59', 'b54', 'b62']

Section: Robust Weak-to-strong Generalization for Superalignment
Below, we propose Adv-W2S, the first robust weak-to-strong generalization framework to elicit zeroshot robustness from large VLMs across a variety of vision-language tasks without extra supervision.
this section cite: []

Section: Revisiting Adversarial Fine-tuning and Weak-to-strong Generalization
Adversarial Fine-tuning. CLIP [60] consists of an image encoder f θI : X → R d and a text encoder f θT : T → R d , parameterized by θ = (θ I , θ T ). These encoders map image-text pairs (x, t) into ddimensional features. Classification is performed via computing the probability that input x belongs to class c ∈ {1, . . . , C} via softmax over the image-text feature cosine similarity:
p c (x) = exp cos(f θI (x), f θT (t c )) C c ′ =1 exp cos(f θI (x), f θT (t c ′ )) ,(1)
where exp(•) and cos(•) denote the exponential and cosine similarity functions. Each text prompt is tokenized by g(•) and embedded by f θT (•), denoted as t c ′ = g("[Context][CLASS c ′ ]"). A typical template is "This is a photo of[CLASS c ′ ]". We define p(x) = [p 1 (x), . . . , p C (x)] ⊤ ∈ [0, 1] C as the prediction for input x across C categories. Given image-text pairs D, standard adversarial fine-tuning (TeCoA) [55] is framed as a minimax optimization to improve CLIP robustness:
min θI E (x,c)∼D max ∥δ∥ ∞ ≤ϵ L CE p(x + δ), y(c) ,(2)
where y(c) = [1(c = 1), . . . , 1(c = C)] ⊤ ∈ {0, 1} C is a one-hot label for class c, and L CE is the Cross-Entropy (CE) loss. The adversarial example x = x+δ lies within an ℓ ∞ -norm hyperball of radius ϵ around x. Further details about adversarial learning are in Appendix A. CLIP parameters are optimized by empirical risk on adversaries, generated via Projected Gradient Descent (PGD) [6].
x(i+1) = Π B(x,ϵ) x(i) + α • sign ∇ x(i) L CE p(x (i) ), y(c) ,(3)
where sign(•) represents the sign function, and the scalar α denotes the step size. The projection operator Π B(x,ϵ) restricts the adversary in ℓ ∞ -norm hyperball with ϵ-radius around x. The starting point is randomly initialized x(0) ∼ x+0.001•N (0, I) with the final adversary x = x(m) after m steps.
Weak-to-strong Generalization. In analogy to superalignment, Burns et al. [5] proposed weakto-strong generalization via fine-tuning a strong student model using soft-label supervision from a weak teacher. We extend this idea to VLMs, denoting the weak teacher VLM as [f θ T I (•), f θ T T (•)] and the strong student VLM as [f θ S I (•), f θ S T (•)]. Based on Eq. ( 1), let p T (x) and p S (x) denote the prediction of the teacher and student on input x. The standard weak-to-strong generalization approach is formulated as an auxiliary confidence loss:
L AuxConf = (1 -β) • L CE p S (x), p T (x) + β • L CE p S (x), M(p S (x)) ,(4)
where M(•) produces one-hot labels based on strong model predictions, and β controls the trade-off between teacher-student prediction alignment (first term) and student self-refinement (second term).
Problem definition. Unlike standard robustness evaluations that assume in-distribution adversarial attacks [11], we study a more challenging zero-shot robustness scenario [55], where adversaries originate from unseen distributions during inference. Under practical defense conditions, we presume textual prompts are fixed and safeguarded, as they typically reside within multimodal systems and are thus not subjected to manipulation. Within this demanding zero-shot robustness context, we further explore how weak-to-strong generalization contributes to improved robustness elicitation across different vision-language model settings and downstream-task generalization.
this section cite: ['b59', 'b54', 'b5', 'b4', 'b10', 'b54']

Section: Can Vanilla Weak-to-strong Generalization Elicit Robustness?
Table 1: Performance (%) of fine-tuned teacher CLIP models, and their weak-to-strong generalizations, where robustness is w.r.t. Auto-Attack [12].
this section cite: ['b11']

Section: Vanilla-W2S

this section cite: []

Section: -------→ and

this section cite: []

Section: Adversarial-W2S
---------→ denote vanilla and adversarial weak-to-strong generalization.
Method ImageNet Avg. 13 Datasets Clean Robust Clean Robust Natural Fine-Tuning 76.06 0.00 58.64 0.01 Robust Fine-Tuning 64.96 39.74 48.27 32.92 Natural FT Vanilla-W2S -------→ 79.54 0.00 65.50 0.26 Robust FT Vanilla-W2S -------→ 72.72 5.60 55.37 9.20 Natural FT Adversarial-W2S ---------→ 76.10 50.82 63.28 40.97 Robust FT Adversarial-W2S ---------→ 74.95 51.97 62.57 41.70
While vanilla weak-to-strong generalization improves natural performance in single modality tasks [5], its effectiveness in enhancing VLM robustness remains underexplored. We investigate whether the vanilla weak-tostrong generalization scheme (Eq. ( 4)) can elicit robustness from the strong student model. As shown in Table 1, standard weak-to-strong generalization primarily improves natural performance, but fails to yield robustness in large-scale student VLMs. This contrasts with vanilla knowledge distillation, where robustness has been successfully transferred from robust teachers [27]. Intriguingly, even substituting the weak guidance with an adversarially robust VLM (e.g., TeCoA [55]) does not facilitate robustness elicitation. We attribute this to a mismatch in model capacity and learning objectives. Representations from a limited-capacity VLM trained on clean data poorly generalize to a high-capacity VLM tasked with handling unforeseen adversaries.
To validate our claim, we explore an adversarial variant of vanilla weak-to-strong generalization by shifting the data perspective of the strong student VLM to adversarial samples (colored in red):
L Adv-AuxConf = (1 -β) • L CE p S (x + δ), p T (x) + β • L CE p S (x + δ), M(p S (x)) . (5
)
Integrating adversaries into weak-to-strong generalization elicits robustness even with guidance from a non-robust weak VLM (Table 1). The necessity of employing adversaries arises from their ability to represent worst-case input distributions, enabling the strong student VLM to better mimic the weak teacher's robust behavior and its own self-knowledge refinement. Thus, we focus on adversary-driven weak-to-strong generalization for transferable robustness. To mitigate potentially unreliable supervision from weak teacher VLMs, we investigate two complementary mechanisms: alignment re-weighting and source guidance refinement, supported by theoretical analyses in the following sections.
this section cite: ['b4', 'b26', 'b54']

Section: Entropy-guided Uncertainty Re-weighting
Recall that in weak-to-strong generalization, the trade-off factor β balances teacher-student alignment and student self-refinement. However, β is either fixed or follows a predefined warm-up schedule, neglecting the potential prediction errors from the weak teacher model. Thus, we propose an adaptive re-weighting mechanism guided by the teacher's prediction uncertainty, quantified via entropy:
Teacher entropy. We define teacher prediction entropy as H T (x) =c p T (x) c log p T (x) c , capturing uncertainty in an unsupervised scheme: higher entropy H T (x) indicates greater uncertainty (predictions are close to uniform distributions), whereas low entropy reflects more confident predictions. By leveraging prediction entropy as a reliability quantification, we adaptively emphasize the low-entropy guidance from the teacher model, enhancing robust knowledge superalignment.
We seek an instance-wise re-weighting function w(•) that maps prediction entropy to weights while preserving relative differences in uncertainty across samples. To avoid uncertainty scale distortions from ad hoc normalization, w is required to be monotonically increasing in H T (so that more uncertain examples receive higher weight). A suitable choice is a soft normalization:
w(x) = H T (x) H T (x) + κ H ,(6)
where κ H is obtained by an adaptive scaling strategy as the median entropy of the teacher's prediction over the dataset: κ H = median {H T (x)|x ∈ D} . The median entropy serves as a robust central tendency measure against outliers, allowing confident teacher predictions to guide alignment, while uncertain cases prioritize the student self-refinement. Robust weak-to-strong generalization (Eq. ( 5)) can be reformulated by replacing fixed β with our entropy-guided weighting w(x) per input x.
Theorem 1 (Adapted from [25]). Let the weights of the classifier, i.e., in our case these are textual VLM embeddings denoted as ψ c = f θT (t c ), and the vision feature extractor be given by ϕ(•). Let H[•] be the Shannon entropy. Then for the conditional entropy over predictions it holds true that:
∥ϕ(x)∥ 2 ≥ log(C) -H p(• | ϕ(x), ψ 1 , . . . , ψ C ) 2 max i=1,...,C ∥ψ i ∥ 2 .(7)
Proof. See [25].
Hence, with fixed textual embeddings (denominator), lower prediction entropy tightens the model selection space for the vision encoder in VLMs by favoring models with larger weights. The vision encoder is thus constrained to a more stable optimization trajectory and is less prone to overfitting.
Definition 1 (Classification margin in VLMs). Consider a VLM of the form ϕ(•), {ψ c } C c=1 , where ϕ(x) ∈ R d is the vision feature for input x, and each ψ c ∈ R d is a text embedding representing class c. For input x with ground-truth label y, the classification margin is defined as:
γ(y|x) := ψ ⊤ y ϕ(x) -max c̸ =y ψ ⊤ c ϕ(x).(8)
The input x is correctly classified as y when the classification margin γ(y|x) > 0, with larger γ(y|x) indicating a wider separation from the nearest decision boundary.
Theorem 2. Let ϕ(•), {ψ c } C c=1 be the same VLM setup as in Definition 1. Assume ϕ(•) is L- Lipschitz continuous w.r.t. the input norm, i.e., ∥ϕ(x)-ϕ(x ′ )∥ 2 ≤ L∥x-x ′ ∥ for all x, x ′
∈ X . Suppose input x satisfies Theorem 1 (i.e., low prediction entropy enforces large feature norm ∥ϕ(x)∥ 2 ). Under the mild alignment assumption that ϕ(x) is not close-to-orthogonal to the ground-truth textual embedding ψ y of class y, increasing the vision feature norm ∥ϕ(x)∥ 2 expands the margin γ(x). If γ(x) > 0, the prediction for (x+δ) cannot be altered away from the ground-truth label y for any perturbation δ ∈ X with:
∥δ∥ ∞ ≤ ∥δ∥ 2 < γ(x) L max c̸ =y ∥ψ c -ψ y ∥ 2 . (9
)
Proof. See Appendix C.1.
Combined Theorems 1 & 2, and Definition 1 indicate that low-entropy prediction enforces a large vision feature norm, which in turn amplifies the classification margin. Under a mild Lipschitz assumption, this margin translates to robustness against input perturbations. In other words, a model with highly confident predictions naturally places each sample far from the decision boundary in feature space, making it less susceptible to small adversarial or noisy modifications.
this section cite: ['b24', 'b24']

Section: Teacher Guidance Refinement via Inverse Adversarial Examples
Beyond re-weighting uncertain teacher predictions, we investigate an adaptive prediction refinement scheme to reduce reliance on erroneous supervision. In other words, we focus on generating more benign inputs to improve guidance for robust weak-to-strong generalization model. Instead of generating adversaries that reduce confidence (i.e., loss-input gradient ascent), we consider an inverse procedure by reversing the gradient to maximize the likelihood in the neighborhood region of clean samples. Thus, this input perturbation enhances teacher prediction confidence. Such refined samples highlight class-specific features, offering more reliable guidance for the robust weak-to-strong generalization framework.
Inverse adversarial perturbation formulation. Given an input x, we aim to generate a benignly perturbed sample x = x+ δ within an ℓ ∞ -bounded region (∥ δ∥ ∞ ≤ ε), which maximizes the teacher's prediction confidence. Formally, this inverse adversarial example is defined by:
x(i+1) = Π B(x,ε) x(i) -α • sign ∇ x(i) L CE p T (x (i) ), M(p T (x)) .(10)
Equivalently, the perturbation δ shifts the teacher's prediction p T (x) closer to a one-hot distribution based on the teacher-derived pseudolabels M(p T (x)). This process contrasts standard adversary generation Eq. ( 3) by reinforcing, rather than undermining, the teacher's initial decision, creating an "inverse adversary". Consequently, the resulting inverse adversary x = x + δ enhances class-specific confidence and refines teacher supervision, supporting robust weak-to-strong generalization. We reformulate L CE p S (x+δ), p T (x) into its refined counterpart L CE p S (x+δ), p T (x) by aligning adversarial predictions with high-confidence teacher outputs. To justify this refinement, Theorem 3 formalizes the unique role of inverse adversaries as effective guidance distinct from standard inputs.
Theorem 3. Let p T (x) and p S (x) denote the softmax predictions over C categories w.r.t. the teacher and student VLM during weak-to-strong generalization, respectively. Suppose x = x + δ is an adversarial example for student, while x = x + δ is an inverse adversarial example for the teacher with high confidence in the ground-truth class. Then the following cross-entropy inequality holds:
L CE p S (x), p T (x) ≥ L CE p S (x), p T (x) ,(11)
where L CE (q, p)= -C i=1 p i log(q i ) is the cross-entropy of q w.r.t. target p. Proof. See Appendix C.2.
Theorem 3 shows a unique and beneficial property of using our proposed unsupervised inverse adversaries as guidance during robust weak-to-strong generalization. Reducing the prediction gap between p T (x) and p S (x) also enhances the standard teacher-student prediction alignment for clean samples, thus enhancing natural generalization to unseen data.
this section cite: []

Section: Objective function.
We formulate Adv-W2S as a simple min-max optimization by replacing the learning objectives in standard weak-to-strong generalization (Eq. ( 4)) with adversarial and inverse adversarial examples. Under the unsupervised setting, we conduct both adversarial perturbation δ and inverse adversarial perturbation δ generation based on the pseudolabels from the teacher VLM:
δ = arg max ∥δ∥ ∞ ≤ϵ L CE p S (x + δ), p T (x) and δ = arg min ∥ δ∥ ∞ ≤ε L CE p T (x + δ), M(p T (x)) . (12
)
We can thus form adversarial example x = x+δ and its inverse adversarial counterpart x = x+ δ based on the maximization above. By reformulating the adversarial variant of weak-to-strong generalization (Eq. ( 5)) with our entropy-guided uncertainty re-weighting (Eq. ( 6)), our Adv-W2S minimizes:
L W-Inv Adv-AuxConf = 1-w(x) • L CE p S (x), p T (x) + w(x) • L CE p S (x), M(p S (x)) .(13)
During the inference stage, we directly use the CLIP model fine-tuned via our Adv-W2S method for robustness evaluations and further cross-task generalization without modifying VLM architectures.
this section cite: []

Section: Experiments
In this section, we compare our Adv-W2S with state-of-the-art adversarial fine-tuning approach across various downstream tasks and scenarios. Below, we detail our experimental configurations. Standard CLIP 0.00 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.03 0.01 0.01 0.09 0.01 TeCoA [55] 61.74 86. 34 61.99 35.82 18.62 70.57 68.22 27.27 26.17 12.37 5.43 26.93 59.57 44.56 4326 PMG [69] 60.02 88.21 64.12 37.14 23.68 72.47 70.92 28.20 26.33 9.07 5.79 47.06 62.24 45.08 45.74 FARE [63] 43.56 88.55 61.82 34.89 23.74 70.88 67.70 32.95 25.69 3.76 5.31 49.39 56.47 36.86 42.97 TGA [71] 61.46 88.56 63.21 35.44 21.60 71.16 68.52 26.15 26.70 11.37 5.76 47.88 60.32 44.46 45.19 Adv-W2S 58.30 90.49 69.56 41.08 29.62 73.03 73.24 34.33 29.63 11.06 7.53 48.76 65.32 45.43 48.38 Datasets. In line with prior works [55,63], we conduct adversarial learning on the ImageNet training set [13], with zero-shot classification evaluations on its test set and other 13 datasets. We further explore downstream task generalization across various datasets w.r.t. image captioning, visual question answering, object hallucination, and science question answering (see Appendix B.1).
this section cite: ['b54', 'b33', 'b54', 'b62', 'b12']

Section: Implementation details.
Unless specified otherwise, we adopt CLIP [60] with the ViT-Large/14 architecture, as in previous studies [55,63]. During adversarial weak-to-strong generalization, we conduct adversary generation via 10-step PGD with perturbation radius ϵ = 2/255 and step size α = 1/255 in an unsupervised scheme. In analogy to superalignment, we consider a relatively weak teacher of the ViT-Base/32 architecture, pre-trained on the ImageNet training set using TeCoA [55]. Zero-shot robustness is assessed under Auto-Attack [12], an ensemble adversarial attack method for reliable evaluations. We achieve downstream task generalization based on two large-scale VLM frameworks, LLaVA 1.5 7B [48] and OpenFlamingo 9B [2], by replacing vision encoders with our robust counterparts. For fairness, evaluations are under adaptive attacks. Details are in Appendix B.2.
this section cite: ['b59', 'b54', 'b62', 'b54', 'b11', 'b47', 'b1']

Section: Main Results (Zero-shot Classification)
Zero-shot classification performance. Table 2 compares Adv-W2S with state-of-the-art adversarial fine-tuning methods using CLIP ViT-L. We report both clean and robust accuracy (Auto-Attack [12]) across 14 datasets. Our Adv-W2S achieves the best zero-shot performance with an average improvement of ~5% in clean accuracy and ~5.8% in robustness. While in-distribution accuracy on ImageNet drops slightly due to the unsupervised learning scheme [63], Section 4.3 shows this gap can be mitigated by incorporating a supervised objective into robust weak-to-strong generalization.
Robustness across diverse CLIP backbones. In addition to robustness with ViT-L, we here apply our Adv-W2S method on ResNet101 and ViT-Base/16 based on the weak teacher VLM of ResNet50 and ViT-Base/32, respectively. Table 3 shows that our Adv-W2S consistently outperforms other adversarial learning approaches in both average clean and robust accuracy across 14 datasets.
Adversarial learning with diverse perturbation radii. Beyond the default perturbation radius ϵ = 2/255, we explore adversaries of larger ℓ ∞ -norm perturbation radii (ϵ = 3/255, 4/255) during both fine-tuning and robustness evaluations for fair comparisons. As shown in Table 4, we observe that our Adv-W2S surpasses other methods across diverse perturbation radii in zero-shot scenarios. Extension with adversarial PEFT. Fully fine-tuning large-scale VLMs typically introduces significant computational costs. We therefore extend adversarial VLM learning with LoRA [35], a PEFT strategy using trainable low-rank matrices for efficient adaptation. We report both clean and robust accuracy in the zero-shot setting of our LoRA-based Adv-W2S as well as other approaches with the LoRA strategy (see Table 5). The results indicate that our Adv-W2S can still outperform other approaches, even with the LoRA strategy for efficiency.
this section cite: ['b11', 'b62', 'b34']

Section: Zero-Shot Downstream Task Generalization
Image captioning extension. We here extend Adv-W2S to image captioning by replacing the vision encoders of large-scale VLMs (e.g., LLaVA and OpenFlamingo) with our robust versions. We report the CIDEr score [66] for both COCO and Flickr30k datasets (see Table 6). We observe that our Adv-W2S achieves the best captioning performance in terms of both clean and adversarial examples compared to other adversarial learning approaches. In addition to quantitative results, we further provide image captioning visualizations against unforeseen adversaries in Figure 1 in Appendix F.
this section cite: ['b65']

Section: Visual Question Answering (VQA) extension.
Table 6 reports VQA accuracy [1] across three standard VQA datasets using different VLMs. Our Adv-W2S method receives a large gain in zeroshot robustness while maintaining comparable natural performance with standard CLIP. Note that our method can even outperform standard CLIP in clean accuracy on Vizwiz, leading to lossless robustness enhancement. The corresponding visualizations are provided in Figure 2 in Appendix F.
Table 7: Hallucination evaluations (F1-score) using POPE for adversarial VLM learning (ViT-L).
this section cite: ['b0']

Section: Method

this section cite: []

Section: POPE Sampling
Avg. Score Random Popular Adversarial TeCoA [55] 79.8 79.1 75.2 78.0 PMG [69] 81.7 80.9 76.3 79.6 FARE [63] 82.2 81.5 78.6 80.8 TGA [71] 80.4 79.8 76.0 78.7 Adv-W2S 85.6 84.9 81.0 83.8
Object hallucination extension. Foundation VLMs are vulnerable to object hallucinations (i.e., erroneously recognizing objects that do not exist in inputs) [61]. POPE [44] served as an object hallucination evaluation benchmark based on VQA across diverse question sampling scenarios (details in Appendix B.3). We extend adversarial VLM learning with hallucination evaluations in Table 7 (see also visualizations in Figure 3 in Appendix F). Notably, our Adv-W2S shows reduced hallucination rates, benefiting from our inherent regularization to over-confident or overly aligned predictions by balancing teacher-student alignment with model uncertainty. 51.9 52.0 51.6 51.8 FARE [63] 52.5 52.2 52. 4 52.4 TGA [71] 52.1 51.9 51.8 51.9 Adv-W2S 53.8 53.9 53.6 53.8
this section cite: ['b54', 'b68', 'b62', 'b70', 'b60', 'b43', 'b3', 'b70']

Section: Single Choice
TeCoA [55] 67.4 67.6 67.1 67.4 PMG [69] 68.2 68.0 67.7 68.0 FARE [63] 68.1 67.9 67.7 67.9 TGA [71] 67.5 67.8 67.4 67.6 Adv-W2S 69.3 69.5 69.0 69.3 Science question answering extension w/ CoT. Chain of Thought (CoT) has been widely explored to elicit intermediate reasoning steps from large-scale VLMs by guiding them to generate multi-step rationales before producing a final answer [8]. Science Question Answering [52] has been recognized as a standard evaluation benchmark for CoT due to its large quantity of multi-option questions with multimodal contexts from the science curriculum. As shown in Table 8, we extend diverse adversarial VLM learning methods to science question answering across different settings to evaluate their CoT capability. Further details of diverse configurations are in Appendix B.3. Our method consistently achieves the best science question answering performance across diverse settings, which demonstrates that our robust weak-to-strong generalization potentially enhances the CoT capability of large-scale VLMs. Visual examples are in Figure 4 in Appendix F.
this section cite: ['b54', 'b68', 'b62', 'b70', 'b7', 'b51']

Section: Further Analyses (Why Adv-W2S is Effective)
Below we conduct systematic analyses of our proposed Adv-W2S framework and its component modules to justify its efficacy and generalization capability across diverse configurations.
Table 9: Ablation study of key components in our Adv-W2S for average clean and robust accuracy (%) on 14 datasets. EUR IAR Clean Robust 1 63.45 42.43 2 ✓ 66.42 47.35 3 ✓ 67.09 46.79 4 ✓ ✓ 68.75 48.38 Impact of component modules. We analyze the contribution of two main components of our Adv-W2S: (i) Entropy-guided Uncertainty Re-weighting (EUR) from Eq. (6), and (ii) Inverse Adversarial Refinement (IAR) in Eq. (10). Table 9 reports the average clean and robust accuracy across 14 datasets in the zero-shot setting. We adopt the adversarial variant of weak-to-strong generalization from Eq. ( 5) as the baseline (first row in Table 9). Enforcing instance-wise re-weighting to balance prediction alignment and self-refinement contributes to improving both natural performance and adversarial robustness. Refining the teacher guidance with inverse adversarial examples also enhances natural generalization to unseen data, yielding further gains in the zero-shot accuracy. Impact of teacher-student setups. In the standard weak-to-strong generalization paradigm, the weak teacher model is typically pre-trained using task-specific supervised learning under natural (non-adversarial) conditions. To provide further insights into the teacher-student configurations employed in our adversarial weak-tostrong generalization (Adv-W2S) framework, we summarize the key setups (i) whether the weak teacher is pre-trained under adversarial or natural conditions, and (ii) whether an unsupervised adversarial fine-tuning (i.e., FARE [63]) is employed during the initial warm-up stage of Adv-W2S. Table 10 shows that an adversarially pre-trained teacher with unsupervised adversarial fine-tuning for student warm-up enjoys greater zero-shot robustness.
this section cite: ['b62']

Section: Impact of adversary generation schemes.
Below, we analyze the average zero-shot classification performance of diverse adversary generation schemes (i.e., objective functions for generating adversarial examples x = x+δ) in Table 11. We observe that adversary generation by maximizing the teacher-student prediction gap enforces a better robustness transfer in the context of weak-to-strong generalization. More details are in Appendix D.1.
this section cite: []

Section: Impact of inverse adversary generation schemes.
In addition to diverse adversary generation strategies, we also investigate a range of inverse adversary generation approaches (details in Appendix D.2) in Table 12. The results indicate that using the pseudolabel cross-entropy loss leads to inverse adversaries of more benign guidance for robust weak-to-strong generalization.
this section cite: []

Section: Auxiliary ground-truth supervision.
Recall that we primarily focus on robust weak-to-strong generalization in an unsupervised scheme following the standard setup [5]. Despite its improved zero-shot performance, its indistribution performance on the finetuned dataset is still lower than supervised adversarial fine-tuning at the same VLM backbone. Thus, we investigate the underlying effect of appending an auxiliary ground-truth supervision in our Adv-W2S method (see Table 13). Details of this auxiliary branch are in Appendix E. We observe an inherent trade-off between in-distribution performance on ImageNet and out-of-distribution (zero-shot) performance on other datasets.
this section cite: ['b4']

Section: Hyper-parameter sensitivity analyses
We further provide a systematic analysis of key hyperparameters involved in our proposed Adv-W2S method in Appendix G.
this section cite: []

Section: Conclusions
Motivated by our analysis of standard weak-to-strong generalization paradigm in eliciting VLM robustness, we have uncovered that neglecting adversarial examples in alignment objectives leads to robustness degradation, even when leveraging source guidance from an adversarially pre-trained VLM. Thus, we have investigated an adversarial adaptation of standard weak-to-strong generalization, explicitly integrating adversarial examples to elicit robust knowledge in an unsupervised scheme.
Recognizing that supervision from a weak-capacity VLM may be inherently unreliable, we introduce two complementary strategies: uncertainty-based alignment re-weighting and source guidance refinement via inverse adversarial examples. Further theoretical analyses characterize the robustness elicitation efficacy of our method, demonstrating enhanced generalization against subtle adversarial or noise modifications.
TGA [71]. Motivated by the empirical observation that adversarial perturbations induce shifts in text attention maps, TGA [71] has been proposed to incorporate test-guided attention into adversarial fine-tuning, improving the zero-shot adversarial robustness of VLMs:
min θ I E (x,c)∼D max ∥δ∥ ∞ ≤ϵ LCE p(x+δ), y(c) +λ1 • gorig(x) -g(x+δ) +λ2 • gorig(x) -g(x) ,(16)
where g(•) extracts the text-guided attention map. While the original TGA mechanism relies on the global [CLS] token available in ViT-based CLIP models to compute text-guided attention maps, this approach is not directly applicable to ResNet-based CLIP architectures due to the absence of an explicit global pooling token. To address this limitation, we adapt the TGA computation by operating on the final spatial feature maps extracted from the ResNet-101 backbone (as shown in Table 3).
this section cite: ['b70', 'b70']

Section: References
Ref_id:b0 Title: Vqa: Visual question answering Year: (2015)
Ref_id:b1 Title: Openflamingo: An opensource framework for training large autoregressive vision-language models Year: (2023)
Ref_id:b2 Title: Improving adversarial robustness in vision-language models with architecture and prompt design Year: (2024)
Ref_id:b3 Title: Measuring progress on scalable oversight for large language models Year: (2022)
Ref_id:b4 Title: Weak-to-strong generalization: Eliciting strong capabilities with weak supervision Year: (2009)
Ref_id:b5 Title: Projected gradient methods for linearly constrained problems Year: (1987)
Ref_id:b6 Title: Supervising strong learners by amplifying weak experts Year: (2018)
Ref_id:b7 Title: Navigate through enigmatic labyrinth A survey of chain of thought reasoning: Advances, frontiers and future Year: (2024)
Ref_id:b8 Title: Describing textures in the wild Year: (2014)
Ref_id:b9 Title: An analysis of single-layer networks in unsupervised feature learning Year: (2011)
Ref_id:b10 Title: Robustbench: a standardized adversarial robustness benchmark Year: (2021)
Ref_id:b11 Title: Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks Year: (2020)
Ref_id:b12 Title: Imagenet: A largescale hierarchical image database Year: (2009)
Ref_id:b13 Title: Adversarially robust distillation by reducing the student-teacher variance gap Year: (2024)
Ref_id:b14 Title: Robust distillation via untargeted and targeted intermediate adversarial samples Year: (2024)
Ref_id:b15 Title: Robustifying zero-shot vision language models by subspaces alignment Year: (2003)
Ref_id:b16 Title: Stabilizing modality gap & lowering gradient norms improve zero-shot adversarial robustness of vlms Year: (2025)
Ref_id:b17 Title: Improving zero-shot adversarial robustness in vision-language models by closed-form alignment of adversarial path simplices Year: (2025-07)
Ref_id:b18 Title: Confound from all sides, distill with resilience: Multi-objective adversarial paths to zero-shot robustness Year: (2003)
Ref_id:b19 Title: The enemy of my enemy is my friend: Exploring inverse adversaries for improving adversarial training Year: (2002)
Ref_id:b20 Title: Improving adversarially robust few-shot image classification with generalizable representations Year: (2022)
Ref_id:b21 Title: Generalizable and discriminative representations for adversarially robust few-shot learning Year: (2024)
Ref_id:b22 Title: Toward intrinsic adversarial robustness through probabilistic training Year: (2023)
Ref_id:b23 Title: Machine unlearning via task simplex arithmetic Year: (2025)
Ref_id:b24 Title: Maximum-entropy fine grained classification Year: (2018)
Ref_id:b25 Title: Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories Year: (2004)
Ref_id:b26 Title: Adversarially robust distillation Year: (2020)
Ref_id:b27 Title: Making the v in vqa matter: Elevating the role of image understanding in visual question answering Year: (2017)
Ref_id:b28 Title: Vision superalignment: Weak-to-strong generalization for vision foundation models Year: (2024)
Ref_id:b29 Title: Improving weak-to-strong generalization with reliability-aware alignment Year: (2024)
Ref_id:b30 Title: Vizwiz grand challenge: Answering visual questions from blind people Year: (2018)
Ref_id:b31 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b32 Title: The many faces of robustness: A critical analysis of out-of-distribution generalization Year: (2021)
Ref_id:b33 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b34 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b35 Title: Visual prompt tuning Year: (2022)
Ref_id:b36 Title: Roast: Robustifying language models via adversarial perturbation with selective training Year: (2023)
Ref_id:b37 Title: 3d object representations for finegrained categorization Year: (2013)
Ref_id:b38 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b39 Title: Theoretical analysis of weak-tostrong generalization Year: (2024)
Ref_id:b40 Title:  Year: (2023)
Ref_id:b41 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: ()
Ref_id:b42 Title: Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation Year: ()
Ref_id:b43 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b44 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b45 Title: A survey of attacks on large vision-language models: Resources, advances, and future trends Year: (2024)
Ref_id:b46 Title: Pandora's box: Towards building universal attackers against real-world large visionlanguage models Year: (2024)
Ref_id:b47 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b48 Title: Visual instruction tuning Year: (2023)
Ref_id:b49 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b50 Title: Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models Year: (2023)
Ref_id:b51 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b52 Title: Towards deep learning models resistant to adversarial attacks Year: (2018)
Ref_id:b53 Title: Finegrained visual classification of aircraft Year: (2013)
Ref_id:b54 Title: Understanding zeroshot adversarial robustness for large-scale models Year: (2009)
Ref_id:b55 Title: PACE: marrying the generalization of PArameterefficient fine-tuning with consistency regularization Year: (2024)
Ref_id:b56 Title: Automated flower classification over a large number of classes Year: (2008)
Ref_id:b57 Title: Cats and dogs Year: (2012)
Ref_id:b58 Title: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models Year: (2015)
Ref_id:b59 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b60 Title: A comprehensive survey of hallucination in large language, image, video and audio foundation models Year: (2024)
Ref_id:b61 Title: On the adversarial robustness of multi-modal foundation models Year: (2023)
Ref_id:b62 Title: Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models Year: (2009)
Ref_id:b63 Title: Towards vqa models that can read Year: (2019)
Ref_id:b64 Title: Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks Year: (2014)
Ref_id:b65 Title: Cider: Consensus-based image description evaluation Year: (2015)
Ref_id:b66 Title: Rotation equivariant cnns for digital pathology Year: (2018)
Ref_id:b67 Title: Learning robust global representations by penalizing local predictive power Year: (2019)
Ref_id:b68 Title: Pre-trained model guided fine-tuning for zero-shot adversarial robustness Year: (2024)
Ref_id:b69 Title: An LLM can fool itself: A prompt-based adversarial attack Year: (2024)
Ref_id:b70 Title: Text-guided attention is all you need for zero-shot robustness in vision-language models Year: (2024)
Ref_id:b71 Title: Towards adversarial attack on vision-language pretraining models Year: (2022)
Ref_id:b72 Title: Vision-language models for vision tasks: A survey Year: ()
Ref_id:b73 Title: CrossSpectra: exploiting cross-layer smoothness for parameter-efficient fine-tuning Year: (2025)
Ref_id:b74 Title: Pip: Detecting adversarial examples in large vision-language models via attention patterns of irrelevant probe questions Year: (2024)
Ref_id:b75 Title: BiLoRA: almost-orthogonal parameter spaces for continual learning Year: (2003)
