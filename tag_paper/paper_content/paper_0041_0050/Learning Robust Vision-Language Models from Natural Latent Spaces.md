Title: Learning Robust Vision-Language Models from Natural Latent Spaces
Abstract: Pre-trained vision-language models (VLMs) exhibit significant vulnerability to imperceptible adversarial perturbations. Current advanced defense strategies typically employ adversarial prompt tuning to improve the adversarial robustness of VLMs, which struggle to simultaneously maintain generalization across both natural and adversarial examples under different benchmarks and downstream tasks. We propose a collaborative adversarial prompt tuning (CoAPT) approach from pre-trained VLMs to target robust VLMs. Inspired by the image mask modeling, we adopt an improved real-time total variation algorithm to suppress and eliminate high-frequency details from images while preserving edge structures, thereby disrupting the adversarial perturbation space. Subsequently, guided by the high-level image and text representations in the latent space of the pre-trained VLMs, the corrupted natural features are restored while inheriting the superior generalization capability. Experiments on four benchmarks demonstrate that CoAPT achieves an excellent trade-off among natural generalization, adversarial robustness, and task-specific adaptation compared to state-of-the-art methods.

Section: Introduction
Vision-language models (VLMs) such as CLIP [1] and ALBEF [2] have shown significant potential for application in multiple industry ecosystems in recent years. However, recent studies [3,4] have revealed that VLMs exhibit a range of concerning vulnerabilities in real-world deployment. When confronted with distributional biases, adversarial samples, or semantic ambiguities, they often display reasoning biases that deviate from human cognition. As an increasing number of downstream applications built upon VLMs as foundational models emerge, the chain reactions triggered by the vulnerability of VLMs pose serious threats to the security and reliability of multimodal downstream tasks. In this paper, we holistically investigate the vulnerabilities of VLMs and their adversarial robustness, with a particular focus on the typical base model CLIP.
Current adversarial robustness strategies for VLMs primarily include model fine-tuning and adversarial prompt tuning. During adversarial training, model fine-tuning [5,6] relearns the entire set of model parameters to adapt to adversarial examples. This process disrupts the natural data distribution captured by the pre-trained model, leading to a contradiction between robustness and generalization.
Adversarial prompt tuning [7,8,9] improves the robust adaptability of VLMs by guiding the pretrained models to efficiently adapt to adversarial data distributions, without altering the pre-trained model parameters. Textual adversarial prompt tuning [10,11] employs learnable prompts in the language branch to match and counteract adversarial attacks from the visual branch. In contrast, visual adversarial prompt tuning C-AVP [12] directly recognize and refine the adversarial images to allow the pre-trained models to make more accurate predictions. More promising multimodal adversarial prompt methods [7,13,14,8] simultaneously introduce deep learnable prompts into the visual and language branches to achieve more comprehensive adversarial robustness. Although adversarial prompt tuning preserves the generalized feature representations of pre-trained VLMs, excessive reliance on in-distribution adversarial samples causes degradation of their natural generalization distribution during the adaptation process. Out-of-distribution (OOD) or unseen tasks further challenge the natural generalization and robustness of prompt-tuned VLMs [10].
Pre-trained VLMs retain generalizable knowledge for unseen tasks, while adversarial prompts can guide the shift of natural distribution toward adversarial-robust distributions or downstream taskspecific distributions [15]. Therefore, we propose to leverage adversarial prompt tuning to identify a shared latent distribution that effectively balances natural generalization, adversarial robustness, and task-specific adaptation. Due to the inherent discrepancies among different distributions, directly training models with a mixture of natural and adversarial samples to fit the latent distribution leads to suboptimal solutions. Recent findings [16,17] indicate that masked image modeling (MIM) enables models to learn more generalizable and robust representations, which significantly enhances their capacity to adapt to input distribution variations and improve fine-tuning performance in downstream vision tasks. The success of MIM is due to masked image input and image-level reconstruction objectives. However, this paradigm directs the model to pay more attention to high-frequency (HF) components where adversarial perturbations are concentrated, thus failing to effectively improve adversarial robustness [9]. We propose a collaborative adversarial prompt tuning (CoAPT) in which pre-trained CLIP collaborates with a target robust CLIP to address this issue. We convert the patch-level image masking from MIM to pixel-level image corruption for model inputs. An improved real-time total variation (TV) regularization method is employed to suppress the adversarial perturbation space by drastically smoothing the high-frequency details of the input images while preserving the image edge structures. To mitigate the cost of sacrificing natural high-frequency features, we shift the reconstruction objective from the pixel space of the target robust CLIP to the latent representation space of the natural CLIP. The corrupted natural detail features are restored under the guidance of high-level features of natural CLIP images and texts, thereby inheriting their excellent generalization ability. Overall, the fine-tuned adversarial prompts work in synergy with the frozen weights of the original pre-trained CLIP to support the target robust CLIP. They achieve a good balance between (a) improving adversarial robustness while maintaining natural performance on indistribution tasks, and (b) maintaining natural generalization while enhancing the robust adaptability of the original VLMs on OOD or unseen tasks. Our contributions are threefold:
• We propose a novel paradigm for adversarial prompt tuning that learns robust CLIP from the latent space of natural CLIP. CoAPT weakens high-frequency details of input images to suppress the adversarial perturbation space. Guided by natural CLIP, corrupted generalization features are restored in the latent space. We introduce Rényi divergence to minimize the discrepancy between the similarity distributions of adversarial and natural examples.
• We design a real-time adaptive TV regularization method to efficiently suppress the perturbation space. It addresses the slow convergence and residual adversarial perturbations of traditional TV regularization by combining a spatially adaptive regularization strategy based on edge strength response and an accelerated gradient method with adaptive restart.
• An optimal trade-off among natural generalizability, adversarial robustness, and task-specific adaptation is achieved. Without benchmark-specific or dataset-specific hyperparameter tuning, we improve natural and adversarial robustness performance on 15 datasets across four benchmarks by an average of 9.83% and 24.16%, respectively.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b6', 'b12', 'b13', 'b7', 'b9', 'b14', 'b15', 'b16', 'b8']

Section: Related Work
Adversarial attacks on VLMs. Adversarial attacks induce incorrect decisions in VLMs by applying elaborate and imperceptible perturbations to the input texts or images [18,19,20,21,22]. Textbased attacks [23,24,25,26] mislead models into generating incorrect outputs through synonym substitution, rewriting, or character-level perturbations. FGSM [27], PGD [28], AutoAttack [29], and C&W [30] are classical image-based white-box attacks that construct adversarial images by accessing model parameters and gradient information. In terms of multimodal attacks, Co-Attack [31] is a white-box attack method designed for VLMs, while more works focus on building transferable adversarial black-box attack frameworks [32,33,34,35,36,37].
this section cite: ['b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36']

Section: General adversarial robustness.
Researchers have proposed multiple robustness strategies to enhance the reliability of models in adversarial settings [38,39]. Detector-based approaches [40,41] defend against adversarial attacks by detecting and filtering anomalous patterns within input samples. Purification methods [42,43,44] utilize techniques such as image transformations [45,46] and denoising filters [47] to disrupt or remove potential adversarial perturbations from the input, yet they run the risk of weakening normal sample characteristics. Certified robustness approaches [48,49,50] provide theoretical and verifiable guarantees for model robustness, though they are typically applicable only to simple threat models with small certified radii. Adversarial training [51,52,53,54] addresses model vulnerabilities by mining potential adversarial examples in the dataset and adapting the model to withstand adversarial attacks during the training process.
Adversarial robustness of VLMs. Numerous studies have explored the robustness of VLMs under adversarial attacks, mainly including defense strategies based on model fine-tuning and adversarial prompt tuning. TeCoA [5] and LAAT [6] enhance zero-shot adversarial robustness by leveraging the semantic consistency of the text encoder to guide fine-tuning of the image encoder. PMG-AFT [55] and FARE [56] leverage the generalization features of the original pre-trained model to improve the adversarial robustness of the CLIP visual encoder on downstream tasks while preserving natural generalizability. Prompt tuning serves as a lightweight adaptation approach that facilitates the efficient transfer of pretrained models toward the target task distribution [57,15,58,59]. Recent studies [7,8,9] have shown that adversarial prompt tuning can efficiently enhance the robust adaptability of VLMs. APT [10] and AdvPT [11] approaches improve model robustness by introducing learnable textual prompts into the language branch of CLIP to align with adversarial image embeddings. Correspondingly, C-AVP [12] and TeCoA [5] incorporate learnable visual prompts to defend against adversarial attacks. Recent multimodal adversarial prompt methods [7,13,14,8] enhance the consistency between visual and language features of adversarial examples under the guidance of pre-trained CLIP, thereby balancing natural generalization and robust adaptation.
this section cite: ['b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b5', 'b54', 'b55', 'b56', 'b14', 'b57', 'b58', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b6', 'b12', 'b13', 'b7']

Section: Proposed Method
Although prompt learning preserves the general representations of pre-trained VLMs, the adapted prompts lead to overfitting on specific supervised tasks. We propose architectural refinements to enhance VLMs for achieving robustness in both in-distribution and OOD scenarios. Figure 1 provides an overview of our proposed approach, with further details presented in the following sections.
this section cite: []

Section: Preliminaries
CLIP recap. Let V θv (•) and T θt (•) denote the image encoder and text encoder of CLIP, respectively, where θ v and θ t represent the corresponding pre-trained weights. Given a natural image v, the input sequence for the visual branch is constructed as ṽ = {v cls , v 1:M }, where v 1:M are the patch-level linearly projections of the image, and v cls is a learnable vector aggregating global features. Given a manually designed fixed text template t, the input sequence for the language branch is constructed as t = {t sos , t 1:N , t c , t eos }, where t 1:N and t c represent the word embeddings of the template text and the class label, respectively. t sos and t eos are non-parametric start and end tokens. The input sequences from the visual and language branches are encoded by CLIP in the latent space into image embeddings V θv (ṽ) and text embeddings T θt ( t), respectively. During zero-shot inference, the similarity between V θv (ṽ) and the text embeddings of all candidate categories {T θt ( tc )} C c=1 is computed as
exp(sim(V θv (ṽ),T θ t ( t))/ϑ) C c=1 exp(sim(V θv (ṽ),T θ t ( tc))/ϑ)
, where sim(•, •) denotes the cosine similarity function, ϑ is the temperature parameter, and C is the total number of classes.
Adversarial attacks against CLIP. Given a natural image v with ground-truth label y, adversaries construct a perceptually imperceptible adversarial example v adv = v+δ by optimizing the perturbation δ within a q-norm ball of radius ϵ. A successful attack must satisfy the following criteria:
arg max c∈{1,...,C} sim(V θv (ṽ adv ), T θt ( tc )) ̸ = y, s.t. ∥v adv -v∥ q ≤ ϵ.(1)
Adversarial prompt tuning. APT enhances the adversarial adaptability of pre-trained VLMs for specific or novel downstream tasks by optimizing visual or textual prompts through adversarial training. Given the prompts ϕ = {ϕ 1:V v , ϕ 1:T t } to be optimized during adversarial training, where V and T represent the number of trainable tokens within the visual and textual prompts, respectively. Adversarial visual-only and text-only prompting [10,11,12] typically employs shallow prompting, where prompts are inserted solely into the input sequences. Specifically, the visual and textual input sequences are updated as ṽ = {v cls , ϕ 1:V v , v 1:M } and t = {t sos , ϕ 1:T t , t c , t eos }. Building upon shallow prompting, both independent and joint vision-language adversarial prompting [7,8] incorporate deep prompts into multiple layers within the visual and language transformer architectures.
We aim to develop joint vision-language adversarial prompts that learn adversarial transformationinvariant features during training, strengthening the adversarial robustness of the CLIP visual branch. We still denote the adversarial deep prompts as ϕ. Given a downstream dataset D, ϕ is optimized jointly with the frozen parameters θ on adversarial examples. Focusing on the ℓ ∞ threat model, the adversarial optimization process for obtaining the optimal parameters of robust prompts ϕ * can be formalized as:
ϕ * = arg min ϕ E (v,y)∼D max ∥vadv-v∥∞≤ϵ L(V θv,ϕv (ṽ adv ), T θt,ϕt ( tc )) .(2)
this section cite: ['b9', 'b10', 'b11', 'b6', 'b7']

Section: Real-Time Total Variation Regularization for High-Frequency Suppression
Background on total variation. Total variation regularization is implemented in the continuous and discrete settings by solving an unconstrained convex optimization problem in its penalized form:
min u∈U 1 2λ ∥u -v (adv) ∥ 2 + ∥u∥ TV ,(3)
where u ∈ U = R m×n denotes the image to be restored, v (adv) ∈ U represents either a natural or adversarial image. For simplicity, v is used uniformly in this section. ∥•∥ TV represents the discrete total variation of the image gradient, and λ > 0 balances the fidelity and regularization terms. Chambolle [60] transforms Eq. ( 3) into a nonlinear projection problem on a constrained space via dual formulation. However, this method lacks real-time capability and is prone to over-smoothing image details and residual adversarial perturbations. We design adaptive-FGP, a fast gradient projection (FGP) method with an adaptive restart mechanism and a spatially adaptive regularization strategy.
Accelerated gradient method with adaptive restart mechanism. We obtain the optimal solution from a norm-constrained dual vector field, thereby recovering v in the form:
min w∈W f (w k ) := v -γ(v) • div(w k ) 2 ,(4)
where k denotes the current time step, and W ⊆ R (m-1)×n × R m×(n-1) is the unit-ball constraint set for the gradient dual components w k i,j = (p k,x i,j , p k,y i,j ) ⊤ . If the gradient vector is defined in both horizontal and vertical directions, it satisfies ∥w k i,j ∥ ≤ 1. Otherwise, only the single-direction constraint remains, satisfying ∥p k,x i,n ∥ ∞ ≤ 1 and ∥p k,y m,j ∥ ∞ ≤ 1. div(•) denotes the discrete divergence operator, which maps the dual variables w from the vector field W to the image domain U . The gradient of f (w k ) can be computed as
∇ w k f (w k ) = -2 • γ(v) • div * v -γ(v) • div(w k ) .
Using a step size of 1/L, where L denotes the Lipschitz constant of f (w k ) with its upper bound derived as 16γ 2 (v) in the Appendix B. The dual variable update rule can be expressed as:
w k = Π W wk - ∇(v -γ(v) • div( wk )) 8 • γ(v) ,(5)
where Π W represents the projection operator. The update of w is performed as follows:
wk+1 = w k + (τ k -1) • w k -w k-1 /τ k+1 , if θ k < θ th , w k , otherwise,(6)
when θ k meets the predefined threshold θ th , the Nesterov [61] time-scale variable is updated with τ k+1 = 1 + 1 + 4τ 2 k /2; otherwise, it is reset to 1.0. The solution to the objective function is denoted as u k = v -γ(v) • div( wk-1 ). The solution increments at two consecutive time steps are defined as σ k = u k -u k-1 and σ k-1 = u k-1 -u k-2 . Whether the current momentum accumulation benefits the variable update is determined utilizing a cosine similarity-based adaptive restart criterion:
cos(θ k ) = ⟨σ k , σ k-1 ⟩ ∥σ k ∥ • ∥σ k-1 ∥ + ζ , (7
)
where ζ is a numerical stabilization term. When the angle between directions exceeds 90 • , it signals a sharp deviation or reversal between momentum and update, indicating trajectory discontinuity. We then reset the temporal scaling and disable momentum to avoid overshooting.
Spatially adaptive regularization strategy. The regularization map γ(v) ∈ R m×n + is given by:
γ(v) = µ base • (1 + µ gain • Φ(v)) ,(8)
where µ base , µ gain ∈ R + represent the base regularization strength and the sensitivity of the adjustment factor, respectively. The edge magnitude response function Φ(v) ∈ R m×n + is estimated using Sobel convolution kernels as (v * K x ) 2 + (v * K y ) 2 , where K x and K y denote the horizontal and vertical Sobel operators respectively. This adaptive regularization strategy automatically reduces the regularization strength in edge regions while enhancing it in flat regions, thereby preserving structural image details and effectively suppressing adversarial perturbations.
Convergence criterion. The relative change in update is measured through the Frobenius norm:
max i∈{k, k-1, ..., k-s} ∥σ i ∥ F ∥u i ∥ F + ζ < ξ. (9
)
If the convergence tolerance threshold ξ > 0 is satisfied for s consecutive iterations, the projection optimization problem is considered to have converged. Based on the optimal solution w k⋆ (v), the optimal image estimate for the original problem can be recovered as
ρ(v) = v -γ(v) • div(w k⋆ (v)).
this section cite: ['b59', 'b60']

Section: Natural-Latent-Guided Adversarial Prompt Learning
Reconstruction of natural generalization representations. CoAPT employs deep contextual multimodal prompts and refines visual prompts through linear projection onto language prompts to foster synergy between visual-language prompts. As illustrated in Figure 1, we efficiently learn generalizable knowledge from the natural CLIP by aligning its clean vision-language embeddings with adversarial embeddings from the robust CLIP in the latent space. Notably, Vanilla CLIP employs fixed text templates, which limit its ability to capture the semantic diversity required for generalization effectively during fine-tuning. A Gaussian radial basis function (RBF) is used to measure the embedding similarity between the natural CLIP and the robust CLIP in the latent space. Compared to cosine similarity, which primarily captures angular differences of vectors, Gaussian RBF highlights feature shifts caused by small-scale perturbations, allowing more sensitive detection of subtle distributional changes. In particular, we align both the visual and language branches:
L recon = 2 -exp -β ∥V θv,ϕv,φv (ρ(ṽ adv )) -V θv (ṽ)∥ 2 2 + ∥T θt,ϕt,φt ( t) -T θt ( t)∥ 2 2 , (10
)
where the parameter β = (2σ 2 ) -1 controls the sensitivity of distance variation to similarity. φ v and φ t are the low-rank residual modules introduced next. The learnable prompts in both the language and visual branches can adapt the data distribution of Vanilla CLIP to that of specific downstream adversarial tasks, while preserving and enhancing generalization and robustness to OOD tasks.
Low-rank residual module. Directly imposing consistency constraints in the latent space is equivalent to introducing a strong supervisory signal, which lacks the flexibility to adapt to taskspecific requirements and interpretable deviations. Inspired by LoRA [62], we introduce two low-rank matrices as an intermediate learnable bottleneck structure. This design allows the model to preserve the backbone features while selectively capturing fine-grained task-specific shifts within a compact subspace. Specifically, we incorporate an additional update term through low-rank reparameterization:
V θ,ϕ,φ = (I + η • BA) V θ,ϕ ,(11)
where η is the scaling factor, B ∈ R d×r , A ∈ R r×d , and r ≪ d. The initial parameter perturbation is controlled by initializing the matrices as A ∼ N (0, 1/r) and B ∼ δ(0).
this section cite: ['b61']

Section: Rényi regularization.
Let P and Q denote the predicted probability distributions of natural and adversarial samples in the vision-language space of robust CLIP, respectively. Since adversarial samples are derived from minor perturbations of natural samples, P is considered absolutely continuous with respect to Q. We introduce a regularization loss based on the α-order Rényi divergence [63] to reduce the discrepancy between the natural and adversarial predictive distributions in robust CLIP:
L rényi = 1 α -1 log E P dP dQ α-1 , α ∈ [0, ∞),(12)
where dP dQ is the Radon-Nikodym derivative of P with respect to Q. α explicitly controls the sensitivity to distributional differences. Higher orders (α > 1) enhance the ability of the model to suppress spurious correlations. This mechanism corrects potential discriminative boundary ambiguities and reduces overfitting risks by preserving task-beneficial generalized features. Correspondingly, the supervised loss for downstream classification tasks can be expressed with the Rényi cross-entropy [64]:
L rce = α 1 -α log i P (i) • Q(i) α-1 α , α ∈ [0, ∞).(13)
Note that the Rényi cross entropy degenerates into Shannon cross entropy when the dataset labels are represented in one-hot coding. The overall training objective of CoAPT can be expressed as follows:
L coapt = κ 1 L recon + κ 2 L rényi + κ 3 L rce ,(14)
κ 1 , κ 2 , κ 3 are hyperparameters weighting contributions of individual losses to the overall objective.
Overview of proposed method. Algorithm 1 illustrates the adversarial prompt optimization procedure adopted by CoAPT. In each training iteration, a batch of image-label pairs (v, y) is sampled from the downstream dataset D. Subsequently, the visual and textual sequences are constructed and accompanied by trainable deep prompts. Perceptually invisible adversarial examples v adv are crafted under ℓ ∞ norm constraints to induce erroneous model predictions (Lines 2∼4). These sequences are then fed into the natural CLIP and the robust CLIP equipped with low-rank residual modules φ v and φ t to obtain the corresponding visual and language representations (Lines 5∼9). CoAPT integrates
this section cite: ['b62', 'b63']

Section: Algorithm 1 Natural-Latent-Guided Adversarial Prompt Learning
Input: Dataset D, frozen CLIP encoders V θv , T θt , prompt parameters ϕ = {ϕ v , ϕ t }, low-rank modules φ = {φ v , φ t }, loss weights κ 1 , κ 2 , κ 3 , adversarial budget ϵ Output: Optimized robust prompts ϕ ⋆ 1: for each minibatch (v, y) ∼ D do 2:
Set the real-time total variation regularization parameters 3:
Construct input sequences ṽ, t and deep prompts ϕ 4:
Generate adversarial example v adv under ℓ ∞ constraint: ∥v adv -v∥ ∞ ≤ ϵ 5:
Generate visual and textual representations for natural CLIP and robust CLIP:
6: V nat ← V θv (ṽ) 7: T nat ← T θt ( t) 8: V adv ← V θv,ϕv,φv (ρ(ṽ adv )) 9:
T adv ← T θt,ϕt,φt ( t)
10: Compute reconstruction loss L recon ← 2 -exp -β(∥V adv -V nat ∥ 2 2 + ∥T adv -T nat ∥ 2 2 ) 11: Compute visual-textual representation similarity P = scale • V nat • T ⊤ nat , Q = scale • V adv • T ⊤ adv 12: Compute Rényi divergence loss L rényi ← 1 α-1 log E P [( dP dQ ) α-1 ] 13:
Compute Rényi cross-entropy loss:
L rce ← α 1-α log i P (i) • Q(i) α-1 α 14: Take gradient step on ∇ ϕ,φ (κ 1 L recon + κ 2 L rényi + κ 3 L rce ) 15:
ϕ, φ ← Backward(∇ ϕ,φ ) 16: end for three losses, including a reconstruction loss for recovering generalization, a Rényi divergence loss to quantify prediction discrepancies between natural and adversarial samples, and a cross-entropy loss for classification (Lines 10∼13). Finally, only the prompt parameters ϕ and the low-rank module parameters φ are updated via gradient descent. Adversarial prompt learning significantly improves the robust generalization of the model under image perturbations and distributional shifts, and exhibits strong cross-task transferability (Lines 14∼15).
this section cite: []

Section: Experiments

this section cite: []

Section: Evaluation Settings
Datasets and benchmark settings. We conduct a comprehensive evaluation of the proposed CoAPT method across four benchmark settings on 15 datasets spanning diverse vision tasks. For the evaluation of few-shot learning, base-to-novel class generalization, and zero-shot benchmarks, we adopt 11 image classification datasets, including EuroSAT [65] for satellite imagery, UCF101 [66] for action recognition, DTD [67] for texture classification, SUN397 [68] for scene recognition, Caltech101 [69] and ImageNet [70] for general object recognition, and FGVC Aircraft [71], Flowers102 [72], OxfordPets [73], Food101 [74], and StanfordCars [75] for fine-grained classification tasks. For the OOD benchmark, we select four variants of ImageNet, ImageNet-A [76], ImageNet-R [77], ImageNet-Sketch [78], and ImageNetV2 [79], as the domain generalization test sets. Notably, both zero-shot and OOD utilize the training set of ImageNet as the source dataset.
Adversarial training and evaluation. The attack settings of baseline methods TeCoA [5] and FAP [7] are adopted to ensure fair comparison. During adversarial training, we adopt a two-step PGD attack with a maximum perturbation magnitude ℓ ∞ = 1/255 and step size α = 1/255. For robustness evaluation, we employ a 100-step PGD attack under the same constraints to thoroughly assess the defense capability of the model under strong attacks.
this section cite: ['b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b75', 'b76', 'b77', 'b78', 'b6']

Section: Implementation details.
Our method is built upon the ViT-B/32 architecture of Vanilla CLIP. Each experiment is conducted three times with different random seeds, and the average results are reported. The convergence tolerance threshold in Adaptive-FGP is set to ξ = 1e -3 , s = 3, and the maximum number of iterations is 30. The parameters of the regularization factor map γ(v) are set to µ base = 0.1 and µ gain = 1.2. We employed 2.5-order Rényi divergence regularization, with L coapt coefficients set to κ 1 = 8, κ 2 = 1, κ 3 = 1. Adversarial prompts with a length of 4 and a depth of 9 are applied to both the visual and textual branches. The RAdam optimizer with an initial learning rate of 0.00735 is adopted, and the batch size is set to 64. In contrast to the existing research work, we do not set proprietary hyperparameters for any of the benchmarks and datasets, in order to prove the generality of the proposed CoAPT. Under few-shot settings we compare with FAP and baselines from its paper.
this section cite: []

Section: Adversarial Few-Shot Learning
The robust generalization capability of each model to specific tasks is evaluated under the condition of only a few identically distributed samples. As shown in Figure 2, CoAPT demonstrates consistently superior performance compared to all baseline methods. CoAPT exhibits robust learning ability with near-linear steady improvement in natural and adversarial accuracy as the number of shots increases. In contrast, the baseline methods show significant performance fluctuations across different shot counts. Furthermore, our approach achieves superior control over the trade-off between natural accuracy and adversarial robustness. In most of the datasets, CoAPT is able to match the natural accuracy of Vanilla CLIP with only 1-shot learning. On six datasets, including Caltech101, our robust accuracy is even higher than the natural accuracy of the baseline method. The robust accuracy of CoAPT on five datasets, including DTD, can be improved to higher than the natural accuracy of Vanilla CLIP by few-shot learning.
this section cite: []

Section: Adversarial Base-to-New Generalization
We assess the ability of the models to balance robust adaptation to specific class distributions and robust generalization to unseen class distributions. Specifically, the models are trained on base classes with a 16-shot setting and jointly evaluated on the base classes and the novel unseen classes. As shown in Table 1, our method outperforms state-of-the-art approaches on all datasets. While improving the average harmonic mean (HM) of robustness by 32.39%, the natural generalization performance of the model also achieves an average gain of 13.09%. Notably, the harmonic mean of robustness for novel classes reaches a maximum of 51.57% on the OxfordPets dataset. These results demonstrate that the robust prompts learned by CoAPT not only adapt to category-specific distributional shifts and distributional discrepancies between natural and adversarial examples but also effectively preserve the natural generalization capability of the original pretrained model. The generalization ability of the models across datasets is explored. CoAPT is trained on Ima-geNet as the source dataset and then evaluated on ten different types of downstream target datasets. The evaluation for each dataset and the corresponding statistical results are presented in Figure 3 and Table 2, respectively. Compared to the FAP method, our approach achieves significant improvements across all metrics on all datasets, particularly in adversarial robustness. We attain a better trade-off between natural and adversarial generalization. Relative to Vanilla CLIP, we sacrifice only 7.83% in natural generalization accuracy while achieving absolute gains of 49.61% and 39.37% in robustness on the source and target datasets.
this section cite: []

Section: Zero-Shot Performance

this section cite: []

Section: Out-of-Distribution Performance
We test the natural generalization and adversarial robustness of the model under domain distribution shift. While maintaining ImageNet as the source dataset, we conduct direct evaluations on four representative variant datasets that share the same set of categories. As shown in Table 3, our method achieves superior natural generalization and robust adaptation across all target datasets compared to the comparison methods.
this section cite: []

Section: Ablation Analysis
As shown in Table 4, we progressively ablate CoAPT components to evaluate their generalizability and importance across the four benchmarks. CoAPT with all components achieves the best performance on all benchmarks. We first remove the adaptive restart mechanism. Most metrics exhibited varying degrees of degradation, with 16-shot and OOD robust accuracy declining by 1.85% and 1.69%, respectively. This mechanism restores optimal convergence without prior knowledge of function parameters and enhances stability near the optimum. We replace the spatially adaptive regularization strategy with a fixed global regularization factor. The ablated model ignores the diversity of image spatial structures, leading to structural blurring and loss of details, with an average drop of 6.00% in clean accuracy across the four benchmarks. We subsequently remove the entire adaptive-FGP method, thereby eliminating the adversarial space compression. During high-level feature recovery in the natural CLIP latent space, the model places greater emphasis on high-frequency components where adversarial perturbations are concentrated, resulting in a degradation in adversarial robustness. However, even with full natural images, the ablated model yields lower natural accuracy than full CoAPT across all benchmarks. Removing the low-rank residual module leads to drops in few-shot-16 robustness and base-to-novel accuracy. As it is sensitive to dataset-specific hyperparameters and was not fine-tuned, its effectiveness is limited. However, due to its potential on certain datasets, the module is retained. When we remove Rényi regularization, the overall performance of the model decreases. Rényi regularization facilitates early detection and correction of boundary ambiguities, and mitigates overfitting by preserving task-relevant generalizable features. CoAPT reduces to a TeCoA-like approach when the final reconstruction loss is removed. The performance drop on unseen tasks is due to the reconstruction loss guiding prompts toward task-irrelevant generalization. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: For datasets, we only use open-source datasets that are publicly available. For codes, we list the original paper of baseline methods in the appendix with access to their respective code repositories.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: In the experimental section, we give all details concerning the experiment settings, parameter values, optimizer, etc.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: We report the average performance and standard deviations across multiple runs in the experimental results section and the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: Details on compute resources are provided in the appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: We have carefully read the NeurIPS Code of Ethics and checked the anonymity of our submission.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: We discuss the boarder impact of our paper in Appendix.
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
Answer: [NA] Justification: Our paper does not include generative models and typically uses open-source datasets for training and evaluation.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes]
Justification: The creator of assets used in our paper states the license in their repository (MIT License).
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: Although we will submit the code in the supplementary materials, we will continue to improve the codebase and make it publicly available after the paper is officially accepted. Currently, we have not released any new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
u ⋆ of the output image is obtained. The adaptive-FGP method exhibits strong parallelizability and efficient acceleration mechanisms, significantly enhancing model robustness while keeping the computational overhead below 10% (Lines 15∼17).
this section cite: []

Section: B Upper Bound Analysis of the Lipschitz Constant
Since the gradient ∇f (w) of the objective function f (w) is Lipschitz continuous, there exists a constant L > 0 such that for any w 1 , w 2 , the following inequality holds:
∥∇f (w 1 ) -∇f (w 2 )∥ ≤ L∥w 1 -w 2 ∥. (15
)
The gradient difference can be computed as:
∇f (w 1 ) -∇f (w 2 ) = -2γ(v) • ∇ div * [(v -γ(v) • div(w 1 )) -(v -γ(v) • div(w 2 ))] = 2γ(v) 2 • ∇ div * [div(w 1 ) -div(w 2 )] (16
) = 2γ(v) 2 • ∇ div * • div(w 1 -w 2 ).
Thus, the norm is bounded by:
∥∇f (w 1 ) -∇f (w 2 )∥ ≤ 2γ(v) 2 • ∥∇ div T • div ∥ • ∥w 1 -w 2 ∥ ≤ 2γ(v) 2 • ∥ div ∥ 2 • ∥w 1 -w 2 ∥.(17)
Analogous to the spectral norm bound of the discrete gradient operator in the TV regularization term, if the operator norm of the discrete divergence operator satisfies | div | ≤ √ 8, we obtain:
∥∇f (w 1 ) -∇f (w 2 )∥ ≤ 16γ(v) 2 • ∥w 1 -w 2 ∥.(18)
Therefore, the upper bound of the Lipschitz constant L(f ) for the objective function f (w) is given by:
L(f ) ≤ 16γ(v) 2 . (19
)
this section cite: []

Section: C Additional Experimental Results

this section cite: []

Section: C.1 Sensitivity Analysis of PGD Attack Hyperparameters
Table 5 systematically evaluates the impact of different configurations on natural and robust accuracy across five datasets (Caltech101 [69], DTD [67], EuroSAT [65], FGVC-Aircraft [71], OxfordPets [73]) under the 16-shot setting and varying perturbation budgets ϵ = {1/255, 2/255, 4/255}. Specifically, it assesses the sensitivity to different numbers of attack iterations ι = {2, 4, 8} and step sizes ς = {ϵ/ι, 2ϵ/ι, 4ϵ/ι}. During the robustness evaluation phase, a 100-step PGD attack with the same perturbation budget and step size as in the training phase is employed to fully examine the defense capability of the model under strong attacks. We aim to determine the optimal combination of hyperparameters to more efficiently perform the next adversarial robustness tests under stronger attacks.
As evidenced in Table 5, employing larger attack step counts and step sizes during training (ι = 8, ς = 4ϵ/ι) does not enhance adversarial robustness during evaluation. Adversarial examples generated by PGD-8 tend to deviate significantly from the true data distribution, potentially causing the model to overfit the distribution of adversarial samples encountered during training rather than learning generalizable robust features. The model achieves higher natural accuracy when trained with a larger number of attack steps and a smaller step size (ι = 8, ς = ϵ/ι), as the resulting adversarial examples remain in close proximity to the original data manifold. The model demonstrates the capability to learn robust features while preserving discriminative power for natural samples. Across all perturbation budget settings, the combination of two attack iterations with a step size of 4ϵ/ι consistently achieves optimal robust accuracy and high clean accuracy. Therefore, we adopt this hyperparameter configuration for subsequent experiments involving varying perturbation budgets and different adversarial attack methods.  58.92 38.36 79.54 57.63 32.67 18.24 74.79 39.60 67.59 46.85  4ϵ/ι 92.58 83.20 59.63 45.27 79.49 58.51 33.33 19.80 79.45 49.03 68.90 51.16 4 ϵ/ι 92.74 81.30 61.82 40.07 80.12 64.15 33.57 17.76 79.50 41.40 69.55 48.94 2ϵ/ι 92.01 80.20 59.69 40.31 78.70 63.54 31.53 17.85 75.61 39.55 67.51 48.29 4ϵ/ι 92.33 80.45 59.04 38.42 79.57 59.01 33.03 18.54 76.04 39.11 68.00 47.11 8 ϵ/ι 93.10 81.99 61.76 40.07 82.06 60.85 35.07 18.15 78.30 40.47 70.06 48.31 2ϵ/ι 91.60 79.63 59.46 40.19 78.99 61.49 32.37 18.60 75.28 38.35 67.54 47.65 4ϵ/ι 91.85 80.81 58.92 40.60 78.51 63.32 32.82 19.50 76.02 40.94 67.62 49.03
this section cite: ['b68', 'b66', 'b64', 'b70', 'b72']

Section: C.2 Impact of Perturbation Budget on Model Performance
We document the performance of CoAPT under four benchmark settings with three perturbation budgets ϵ = {1/255, 2/255, 4/255}. The case of ϵ = 1/255 corresponds to the results presented in the main text of the paper. As shown in Table 6 under the base-to-novel benchmark, the robust HM metrics decrease by 5.72% and 9.73% as the perturbation budgets increase, remaining within acceptable thresholds overall. The natural HM metrics decrease by only 2.27% and 4.59%, respectively, demonstrating the effectiveness of CoAPT in preserving natural generalization. Table 7 reports the natural and robust accuracy of CoAPT under the 16-shot setting across different perturbation budgets. Compared to the base-to-novel setup, the few-shot scenario provides more training samples, enabling the model to exhibit greater stability when confronted with increased perturbations. Specifically, as the perturbation budgets increase, the robust accuracy declines by
this section cite: []

Section: C.3 Robustness Evaluation under Varying Attacks
We evaluate our method using attack types based on different perturbation mechanisms. The CW attack is an optimization-based method designed to generate adversarial perturbations that are minimal in magnitude yet highly effective in misleading the model. It has demonstrated strong attack performance across various tasks. The TPGD attack is a targeted variant of the PGD attack that misdirects samples toward specific target classes. AutoAttack is an ensemble-based, parameterfree robustness evaluation framework that integrates multiple strong attack algorithms to provide reliable adversarial assessment results. Specifically, we evaluate CW, TPGD, and AutoAttack attacks under the zero-shot benchmark, while only CW and TPGD are evaluated under the base-to-novel benchmark. We adopt PGD attack with the hyperparameter configuration ϵ = 4/255, ι = 2, ς = 4ϵ/ι for adversarial training. During the robustness evaluation phase, both CW and TPGD attacks are applied with the same perturbation budget and step size, while the number of attack steps is uniformly set to 100. For AutoAttack, we use the same perturbation budget (ϵ = 4/255), and its attack process does not rely on hyperparameters such as step size or the number of steps. Overall, the robustness advantage of our method is not a result of overfitting to any specific attack. As can be seen from the experimental results under the base-to-novel benchmark in Table 10, our approach exhibits strong robust generalization capabilities when confronted with different types of adversarial attacks. Overall, the CW attack is more destructive. Although it induces significant accuracy degradation on novel classes, the performance remains within acceptable range. In contrast, under the TPGD attack, the model maintains relatively high natural and robust accuracy, further validating the stable performance of CoAPT across different types of adversarial attacks.
Figure 4 presents the robust accuracy of the model under CW, AutoAttack, and TPGD attacks across 11 datasets in the zero-shot benchmark. In terms of overall trends, the model demonstrates the strongest robustness under TPGD attacks, achieving the highest robust accuracy across nearly all datasets. In contrast, CW attacks are more destructive, particularly showing stronger attack effectiveness on complex datasets such as ImageNet and StanfordCars. AutoAttack, as an ensemble-based evaluation framework, displays intermediate attack strength between CW and TPGD. Moreover, significant robustness variations exist across different datasets. The model maintains relatively high robust accuracy on Caltech101, Flowers102, and OxfordPets, while showing noticeably lower performance on FGVCAircraft and EuroSAT. To evaluate the impact of ℓ 2 -norm adversarial attacks on robust VLMs, we designed and conducted an experiment based on ℓ 2 -norm perturbations. The training weights were derived from the ℓ ∞ -based PGD attack, and the evaluation settings remained consistent. Table 11 presents the experimental results of our approach across five datasets under varying perturbation budgets. It can be observed that as the perturbation budget increases, the model's classification accuracy experiences a moderate decline. Nevertheless, our approach significantly improves the model's robustness against ℓ 2 -norm attacks, even under the ℓ ∞ -norm threat model.
, P D J H 1 H W & D O W H F K ' 7 ' ( X U R 6 $ 7 ) * 9 & $ L U F U D I W ) R R G ) O R Z H U V 2 [ I R U G 3 H W V 6 W D Q I R U G & D U V
this section cite: []

Section: C.4 Sensitivity Analysis of Prompt Length and Depth in Multimodal Prompting
Prompt depth and prompt length. We conduct ablation studies on prompt depth and prompt length under the base-to-novel setting across 10 datasets, excluding ImageNet and its variants. Figure 5 summarizes the average results over these datasets. As shown in the left panel of Figure 5, model performance steadily improves with increasing adversarial prompt depth. However, performance gains plateau when the depth exceeds nine layers, showing diminishing returns. To avoid introducing excessive trainable parameters, we ultimately set the prompt depth to 9.
The right panel of Figure 5 illustrates the impact of prompt length on model performance. As the number of prompt tokens increases, the natural and robust performance on base classes remains relatively stable, whereas the natural and robust performance of the novel classes exhibits a declining trend. This indicates that excessive trainable prompt tokens are prone to overfit task-specific features, thereby undermining the task-agnostic generalization capability of VLMs. Similar performance trends have also been reported in the literature [57]. The model achieves optimal performance when the prompt length is set to 4.
underexplored. The current framework assumes that adversarial noise originates solely from the visual modality, which limits its applicability in scenarios involving adversarial manipulations in textual inputs. Although the proposed latent space reconstruction method shows strong generalization in experiments, its specific impact on generalization behavior and the theoretical analysis for its superiority over other techniques remain unexplained. The influence of latent space structure and distribution on model robustness and generalization requires further theoretical exploration. We leave these limitations as essential directions for future investigation.
this section cite: ['b56']

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: The abstract and introduction clearly explain the scope and importance of the work, and the main contributions are summarized at the end of the introduction.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: The limitations of this paper are discussed in the appendix. Guidelines:
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: Only a small portion of this work requires theoretical justification, which has been rigorously proven. The remaining contributions focus on improving adversarial prompt learning from an empirical perspective.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We provide detailed implementation information in Section 4.1 and the appendix to support the reproduction of our experimental results. The corresponding code will also be included in the supplemental material.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: There are no crowdsourcing experiments and research with human subjects under adversarial prompt learning settings. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: There are no crowdsourcing experiments and research with human subjects under adversarial prompt learning settings. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: This paper does not involve LLM. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b1 Title: Align before fuse: Vision and language representation learning with momentum distillation Year: (2021)
Ref_id:b2 Title: Strong transferable adversarial attacks via ensembled asymptotically normal distribution learning Year: (2024)
Ref_id:b3 Title: Improving transferable targeted adversarial attacks with model self-enhancement Year: (2024)
Ref_id:b4 Title: Understanding zero-shot adversarial robustness for large-scale models Year: (2023)
Ref_id:b5 Title: Language-driven anchors for zero-shot adversarial robustness Year: (2024)
Ref_id:b6 Title: Few-shot adversarial prompt learning on vision-language models Year: (2024)
Ref_id:b7 Title: Tapt: Test-time adversarial prompt tuning for robust inference in vision-language models Year: (2024)
Ref_id:b8 Title: Improving adversarial robustness of masked autoencoders via test-time frequency-domain prompting Year: (2023)
Ref_id:b9 Title: One prompt word is enough to boost adversarial robustness for pre-trained vision-language models Year: (2024)
Ref_id:b10 Title: Adversarial prompt tuning for vision-language models Year: (2024)
Ref_id:b11 Title: Visual prompting for adversarial robustness Year: (2023)
Ref_id:b12 Title: Adversarial prompt distillation for vision-language models Year: (2024)
Ref_id:b13 Title: Revisiting the robust generalization of adversarial prompt tuning Year: (2024)
Ref_id:b14 Title: Self-regulating prompts: Foundational model adaptation without forgetting Year: (2023)
Ref_id:b15 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b16 Title: Toward richer visual representations by integrating pixel and latent masked image modeling Year: (2025)
Ref_id:b17 Title: Intriguing properties of neural networks Year: (2014)
Ref_id:b18 Title: Deepfool: a simple and accurate method to fool deep neural networks Year: (2016)
Ref_id:b19 Title: Explaining and harnessing adversarial examples Year: (2015)
Ref_id:b20 Title: Las-at: adversarial training with learnable attack strategy Year: (2022)
Ref_id:b21 Title: Towards compositional adversarial robustness: Generalizing adversarial training to composite semantic perturbations Year: (2023)
Ref_id:b22 Title: Is bert really robust? a strong baseline for natural language attack on text classification and entailment Year: (2020)
Ref_id:b23 Title: Black-box generation of adversarial text sequences to evade deep learning classifiers Year: (2018)
Ref_id:b24 Title: Generating natural language adversarial examples through probability weighted word saliency Year: (2019)
Ref_id:b25 Title: BERT-ATTACK: adversarial attack against BERT using BERT Year: (2020)
Ref_id:b26 Title: Explaining and harnessing adversarial examples Year: (2015)
Ref_id:b27 Title: Towards deep learning models resistant to adversarial attacks Year: (2018)
Ref_id:b28 Title: Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks Year: (2020)
Ref_id:b29 Title: Towards evaluating the robustness of neural networks Year: (2017)
Ref_id:b30 Title: Towards adversarial attack on vision-language pre-training models Year: (2022)
Ref_id:b31 Title: Vlattack: Multimodal adversarial attacks on vision-language tasks via pre-trained models Year: (2023)
Ref_id:b32 Title: Ot-attack: Enhancing adversarial transferability of vision-language models via optimal transport optimization Year: (2023)
Ref_id:b33 Title: Transferable multimodal attack on vision-language pretraining models Year: (2024)
Ref_id:b34 Title: Sa-attack: Improving adversarial transferability of vision-language pre-training models via self-augmentation Year: (2023)
Ref_id:b35 Title: On evaluating adversarial robustness of large vision-language models Year: (2023)
Ref_id:b36 Title: Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models Year: (2023)
Ref_id:b37 Title: Semantically consistent visual representation for adversarial robustness Year: (2023)
Ref_id:b38 Title: Stylized adversarial defense Year: (2022)
Ref_id:b39 Title: Libre: A practical bayesian approach to adversarial detection Year: (2021)
Ref_id:b40 Title: Slowlidar: Increasing the latency of lidar-based detection using adversarial examples Year: (2023)
Ref_id:b41 Title: Densepure: Understanding diffusion models for adversarial robustness Year: (2023)
Ref_id:b42 Title: Diffusion models for adversarial purification Year: (2022)
Ref_id:b43 Title: Disco: Adversarial defense with local implicit functions Year: (2022)
Ref_id:b44 Title: Augmax: Adversarial composition of random augmentations for robust training Year: (2021)
Ref_id:b45 Title: Improving adversarial robustness with adversarial augmentations Year: (2023)
Ref_id:b46 Title: Countering adversarial images using input transformations Year: (2018)
Ref_id:b47 Title: Provable defense against adversarial attacks to multi-modal models Year: (2024)
Ref_id:b48 Title: (certified!!) adversarial robustness for free Year: (2023)
Ref_id:b49 Title: Exploring and exploiting decision boundary dynamics for adversarial robustness Year: (2023)
Ref_id:b50 Title: Improving accuracy-robustness trade-off via pixel reweighted adversarial training Year: (2024)
Ref_id:b51 Title: Boosting adversarial training with hypersphere embedding Year: (2020)
Ref_id:b52 Title: Improving adversarial robustness with self-paced hard-class pair reweighting Year: (2023)
Ref_id:b53 Title: Efficiently boosting the robustness of pretrained vision transformers Year: (2024)
Ref_id:b54 Title: Pre-trained model guided fine-tuning for zero-shot adversarial robustness Year: (2024)
Ref_id:b55 Title: Robust CLIP: unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models Year: (2024)
Ref_id:b56 Title: Multi-modal prompt learning Year: (2023)
Ref_id:b57 Title: Consistency-guided prompt learning for vision-language models Year: (2024)
Ref_id:b58 Title: Learning to prompt for vision-language models Year: (2022)
Ref_id:b59 Title: An algorithm for total variation minimization and applications Year: (2004)
Ref_id:b60 Title: A method for solving the convex programming problem with convergence rate o (1/k2) Year: (1983)
Ref_id:b61 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b62 Title: α-leakage interpretation of sibson mutual information and rényi capacity Year: (2025)
Ref_id:b63 Title: A cross entropy interpretation of renyi entropy for α-leakage Year: (2024)
Ref_id:b64 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b65 Title: Ucf101: A dataset of 101 human actions classes from videos in the wild Year: (2012)
Ref_id:b66 Title: Describing textures in the wild Year: (2014)
Ref_id:b67 Title: Sun database: Large-scale scene recognition from abbey to zoo Year: (2010)
Ref_id:b68 Title: Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories Year: (2004)
Ref_id:b69 Title: Imagenet: A large-scale hierarchical image database Year: ()
Ref_id:b70 Title: Fine-grained visual classification of aircraft Year: (2013)
Ref_id:b71 Title: Automated flower classification over a large number of classes Year: (2008)
Ref_id:b72 Title: Cats and dogs Year: (2012)
Ref_id:b73 Title: Food-101-mining discriminative components with random forests Year: (2014)
Ref_id:b74 Title: 3d object representations for fine-grained categorization Year: (2013)
Ref_id:b75 Title: Natural adversarial examples Year: (2021)
Ref_id:b76 Title: The many faces of robustness: A critical analysis of out-of-distribution generalization Year: (2021)
Ref_id:b77 Title: Learning robust global representations by penalizing local predictive power Year: (2019)
Ref_id:b78 Title: Do imagenet classifiers generalize to imagenet Year: (2019)
Ref_id:b79 Title: Table 7: Performance of CoAPT under varying perturbation budgets on the few-shot benchmark across 11 datasets. ϵ Metric Caltech101 DTD EuroSAT FGVCAircraft Food101 Year: ()
Ref_id:b80 Title: As shown in the evaluation results under the zero-shot settings in Table 8, our model consistently demonstrates strong natural generalization, adversarial robustness, and stability across different perturbation budgets. Specifically, under the zero-shot scenario, the average robust accuracy decreases by 3.93% and 7.35% with increasing perturbation budgets, while the average natural accuracy declines by only 1.63% and 4.21%. The results indicate that the model maintains strong perturbation resistance even under extreme generalization conditions. The evaluation results under the out-of-distribution settings in Table 9 exhibit a similar trend. Table 8: Performance of CoAPT under varying perturbation budgets on the zero-shot benchmark across 11 datasets Year: ()
