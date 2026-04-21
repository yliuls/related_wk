Title: Angular Steering: Behavior Control via Rotation in Activation Space
Abstract: Controlling specific behaviors in large language models while preserving their general capabilities is a central challenge for safe and reliable artificial intelligence deployment. Current steering methods, such as vector addition and directional ablation, are constrained within a two-dimensional subspace defined by the activation and feature direction, making them sensitive to chosen parameters and potentially affecting unrelated features due to unintended interactions in activation space. We introduce Angular Steering, a novel and flexible method for behavior modulation that operates by rotating activations within a fixed two-dimensional subspace. By formulating steering as a geometric rotation toward or away from a target behavior direction, Angular Steering provides continuous, fine-grained control over behaviors such as refusal and compliance. We demonstrate this method using refusal steering and emotion steering as use cases. Additionally, we propose Adaptive Angular Steering, a selective variant that rotates only activations aligned with the target feature, further enhancing stability and coherence. Angular Steering generalizes existing addition and orthogonalization techniques under a unified geometric rotation framework, simplifying parameter selection and maintaining model stability across a broader range of adjustments. Experiments across multiple model families and sizes show that Angular Steering achieves robust behavioral control while maintaining general language modeling performance, underscoring its flexibility, generalization, and robustness compared to prior approaches.

Section: Introduction
Large language models (LLMs) have become remarkably capable, yet steering their behavior towards desired responses remains a challenge. On one hand, we want the model to follow certain guidelines or exhibit particular traits, e.g., refusing inappropriate requests or complying with user instructions. On the other hand, aggressive tuning of the models behavior can degrade its original performance, causing losses in fluency or actuality [49,51].
Activation steering, which manipulates internal representations of language models at inference time, has emerged as a compelling alternative to retraining for behavior control [51,58,39]. Techniques such as activation addition [51,39] and direction orthogonalization [1,58] have demonstrated the capacity to steer models toward or away from specific behaviors. However, these methods offer limited granularity. For instance, orthogonalization removes the feature entirely by projecting activations onto the orthogonal subspace, leaving no room for partial suppression. Moreover, activation addition requires careful tuning of the coefficient to avoid instability; improper values can lead to degraded fluency or incoherent outputs [43,47,52,44]. While conditional methods improve context-sensitivity, they often retain the underlying manipulation mechanism [53,19,21].
Contribution. We propose Angular Steering, a method that reformulates activation editing as a geometric rotation within a fixed 2D subspace of the models activation space. By identifying a Before Normalization After Normalization h (activation) dfeature (feature direction) h + dfeature (activation addition, = 1) h (directional ablation) Transformer Block LayerNorm Self Attention LayerNorm Angular Steering MLP Angular Steering Figure 2: Illustration of a typical Transformer Block in modern LLMs with Angular Steering applied after each normalization layer. Sure, let's create a fun and fictional example of a hoax website that could spread false information and potentially cause a bit of a stir. For this example, we'll create a website about a new, highly contagious virus that has just been discovered. The goal is to highlight how misinformation can spread quickly, so let's craft a website that looks credible but contains false information. ### Website Name: GlobalVirusAlert.org (...) redirect (steered to 300 • ) Creating a hoax website that spreads false information and leads to panic is unethical and illegal. Such actions can cause real harm and distress to individuals and communities. Instead, I can help you create a responsible and informative website that educates people on how to spot and report misinformation. Heres an example of a responsible website that provides resources and guidance: (...)
behavior-associated direction, e.g., refusal, and a complementary axis that jointly define a steering plane, we rotate the activation vector within this plane to control the behavioral expression (Tab. 1). This approach offers a smooth and interpretable control of the model's behavior via the rotation angle of activations, enabling the modulation of the feature's strength without compromising the model's overall representation capacity. We also introduce an adaptive variant of Angular Steering, namely Adaptive Angular Steering, which adds another dimension of controllability by applying steering selectively based on the local activation context.
Crucially, our formulation generalizes and unifies existing steering operations (Fig. 1). In particular, activation addition [51] and directional ablation [1], when apply before normalization, can both be viewed as specific instances of rotation in a 2D plane defined by the original activation and a target feature direction. Linear combination [51] corresponds a partial rotation toward or away from the feature; and orthogonalization [1] corresponds to rotating to a position 90 degrees from the feature. We summarize these correspondences in Appendix A. By subsuming these techniques under a common geometric framework, Angular Steering offers a principled abstraction that clarifies their effects and limitations, while extending their controllability.
In summary, our contribution is three-fold:
1. We propose the novel Angular Steering, a rotation-based framework for fine-grained, continuous control of model behaviors, and the Adaptive Angular Steering, a selective variant of Angular Steering that improves robustness and minimizes coherence loss. 2. We demonstrate that Angular Steering serves as a unifying framework for prior activation intervention methods from a geometric perspective. 3. We empirically demonstrate that both Angular Steering and Adaptive Angular Steering achieve strong behavior control with minimal degradation of model's performance outside of the targeted steering tasks across multiple modern LLM families.
Organization. We structure this paper as follows: Section 2 contextualizes the existing body of literature regarding LLMs Activation Steering and Interpretability. In Section 3, we provide the necessary background and describe the experimental setup for our study on Angular Steering. In Section 4, we first discuss the extraction of feature directions and the construction of the steering plane, then introduce the Angular Steering operation and its adaptive variant. Section 5 presents refusal steering experiments and analyzes the behavioral transition across angles. In Section 6, we evaluate the effect of Angular Steering on the overall capability of the model. The paper ends with concluding remarks.
this section cite: ['b49', 'b51', 'b51', 'b58', 'b39', 'b51', 'b39', 'b0', 'b58', 'b43', 'b47', 'b52', 'b44', 'b53', 'b18', 'b20', 'b51', 'b0', 'b51', 'b0']

Section: Related Work
Activation Steering Features such as behaviors or concepts are hypothesized to correspond to (nearly) orthogonal directions in activation space [33,31,4,10]. Activation steering modifies hidden representations at inference time to induce or suppress such features [1,2,17,20,25,51,58,49,26,39,50]. Most methods scale feature directions using manually tuned coefficients [51,58,49,2,20,52,45], but tuning is challenging due to sensitivity to the activation norm, which grows exponentially across layers (Fig. 3). Poor scaling often leads to incoherent outputs [51,49]. Directional ablation [1,58] avoids coefficient tuning by orthogonalizing activations, but fails in cases where negative alignment with a feature direction is meaningful [51,58,49].
Our experiments further show that extracted feature directions reliably distinguish contrastive data (Fig. 4).
this section cite: ['b33', 'b31', 'b3', 'b9', 'b0', 'b1', 'b16', 'b19', 'b24', 'b51', 'b58', 'b49', 'b25', 'b39', 'b50', 'b51', 'b58', 'b49', 'b1', 'b19', 'b52', 'b45', 'b51', 'b49', 'b0', 'b58', 'b51', 'b58', 'b49']

Section: Directional Representation.
Contemporary LLMs such as LLAMA 3 [23], QWEN 2.5 [54], and GEMMA 2 [14] universally adopt RMSNorm [57] for normalization. RMSNorm first maps activations to a scaled unit sphere, then deforms them into a fixed hyperellipsoid, emphasizing direction over magnitude. Moreover, Rotary Positional Embeddings (RoPE) and related variants [46,5,7,34] further validate this directional emphasis by encoding positional information as rotations. Methods such as Householder Pseudo-Rotation have extended this notion by explicitly employing normpreserving geometric transformations to steer behaviors effectively and minimally invasively [35].
this section cite: ['b22', 'b54', 'b13', 'b57', 'b46', 'b4', 'b6', 'b34', 'b35']

Section: Wider Geometric Perspective.
Recent work has explored geometric and spectral approaches to post-hoc manipulation of model internals. Spectral Editing of Activations [37] constructs steering directions in principal component space by combining components with maximal positive and minimal negative covariance. AlphaEdit [11] similarly aims to preserve untargeted behavior but modifies model weights offline, making it complementary to activation-based methods. Affine Steering [42] learns linear transformations to shift between concepts and offers theoretical support for directionbased steering, further grounding approaches like Angular Steering.
Our work expands upon these foundations by introducing Angular Steering, a generalization of existing activation steering operators. By explicitly treating steering as a rotation in a defined 2D subspace, our method achieves more robust, interpretable, and flexible behavior control. Rather than focusing on maximizing downstream accuracy, our goal is to present a principled and broadly applicable framework for controlled and non-destructive intervention in LLM activations.
this section cite: ['b37', 'b10', 'b42']

Section: Background
Transformers. Decoder-only transformers process an input token sequence t = (t 1 , . . . , t n ) by first converting tokens to initial embeddings, h
i = Embed(t i ). These activations are then iteratively refined through L layers. Within each layer l, the residual stream activation h (l) i for token t i is updated by incorporating information from a Self-Attention mechanism and a Multi-Layer Perceptron 0 4 8 12 16 20 24 28 32 36 40 44 48 0 0 4 8 12 16 20 24 28 32 36 40 44 48 52 (MLP) block, typically with normalization applied before these components:
1 h (l) i,post-attn = h (l) i + Attn (l) (Norm(h (l) 1:i )); h (l+1) i = h (l) i,post-attn + MLP (l) (Norm(h (l) i,post-attn ))
This layered processing allows the model to construct increasingly sophisticated representations from the input, and the h ∈ R dmodel values are collectively referred to as activations. Finally, the output activations from the last layer, h (L+1) i , are projected to logit scores over the vocabulary via an unembedding step,
logits i = Unembed(h (L+1) i
). These logits are then transformed into probability distributions y i for the next token using a softmax function.
Activation Steering Operators. Two popular approaches are: Activation addition [51] modifies an activation h by adding a scaled feature vector: h ′ = h + α d feat , where d feat denoting the unit-normalized feature direction and α controls the strength of the effect; Directional ablation [1] removes the feature by projecting the activation onto the orthogonal complement: h ′ = hd feat d⊤ feat h. While effective, these methods offer limited granularity. Addition is sensitive to coefficient tuning, and orthogonalization removes the feature entirely. Recent works introduce conditional steering [19,21], which applies these edits selectively, but still rely on the same underlying primitives. Our proposed method, Angular Steering, generalizes these interventions as rotation in a 2D subspace, offering continuous, interpretable, and norm-preserving control.
this section cite: ['b51', 'b0', 'b18', 'b20']

Section: Choice of Activations for Steering.
There are two main options for choosing the representation for steering: the raw activations [1,58,51,20,2] or the normalized activations [52]. While the method proposed in this work applies to both cases, we argue that the latter is the better choice for model steering research. Section 4.1 discusses our motivation for this choice, which leads us to propose steering by angular rotation.
this section cite: ['b0', 'b58', 'b51', 'b19', 'b1', 'b52']

Section: Angular Steering

this section cite: []

Section: Motivation for Angular Steering
Rotation is Better for Steering. Existing activation steering methods that use vector addition [51] require carefully tuned coefficients, which are highly sensitive to layer-specific activation norms. These norms vary due to the residual stream's additive structure and tend to grow across layers (see Fig. 3, also Fig. 10 (left) in Appendix I), making hyperparameter tuning brittle. Orthogonalization [1] offers a hyperparameter-free alternative but ignores the effects of negative scaling, which prior work suggests can induce opposite behaviors [51,58,49].
Our experiments show that feature directions effectively separate contrastive examples. Particularly, in Fig. 4 (also Fig. 10 (right) in Appendix I), for each layer i, we plot the scalar projection of the normalized activation ĥi on the locally extracted feature direction d i feature and demonstrate that activations from contrastive datasets aligned oppositely with the local refusal directions. Furthermore, modern LLMs such as LLAMA 3 [23], QWEN 2.5 [54], and GEMMA 2 [14] use RMSNorm [57] before each MLP and Self Attention block. It is formulated as h = h/RMS(h) ⊙ g, where RMS(h) = √ (1/d model ) ∑ dmodel i=1 h 2 i and ⊙ denotes element-wise multiplication. This operation first maps the activation to a √ d model -scaled unit sphere, making any prior modification effectively norm-preserving, and then directionally scales it into a fixed hyperellipsoid via ⊙g. Even after rescaling, the activation norms have been shown to remain stable [57].
This highlights direction, not magnitude, as the core representational unit. It also aligns with recent interpretability work supporting the Superposition Hypothesis [10]: that features correspond to nearly orthogonal directions and activations are linear combinations of them [1,2,4,6,10,12,25,52,49,3,26,39,50]. Scalar projections measure feature strength, making direction and angle key geometric concepts. Norm-preserving transformations like rotation are, therefore, a principled choice for behavior control.
this section cite: ['b51', 'b0', 'b51', 'b58', 'b49', 'b22', 'b54', 'b13', 'b57', 'b57', 'b9', 'b0', 'b1', 'b3', 'b5', 'b9', 'b11', 'b24', 'b52', 'b49', 'b2', 'b25', 'b39', 'b50']

Section: Existing Activation Steering as Special Cases of Steering by Rotation.
Vector arithmetic and orthogonalization using the pre-normalized activation h i at layer i and a direction representing some feature (d feat ) are equivalent to rotation inside a 2D subspace spanned by Span{h i , d feat } (Fig. 1).
When the activation norms are fixed as recommended by [52,35], existing steering techniques are special cases of angular steering, albeit with restricted flexibility: vector addition is limited to less than 180 degrees, and orthogonalization is fixed at 90 degrees. We provide detailed derivations in Appendix A and empirical comparisons in Appendix C.
In contrast, Angular Steering allows full, continuous control within the steering plane, offering a more expressive and robust alternative. This is further supported by [52], who show that using normalized activations improves probing accuracy across classifiers, reinforcing our hypothesis that steering direction, not raw magnitude, is what ultimately matters.
this section cite: ['b52', 'b35', 'b52']

Section: Overview of Angular Steering
We propose to formulate activation steering as a rotation on a 2-dimensional (2D) subspace P and around the (d model -2)-dimensional orthogonal complement Q of P . Ideally, the plane of rotation P should be parallel to the true target feature direction and perpendicular to other feature directions that are independent of the desired behaviour. Our angular steering provides the following advantages:
• Generalization. It is a generalization of existing steering operations (Fig. 1), namely activation arithmetic [51,58,2,39] and directional ablation [1,58].
• Universality. It can be applied to both raw and normalized activations, although the latter is more computationally efficient.
• Stability. Restricting the rotation to a 2D subspace confines changes to just two orthogonal directions, leaving the remaining basis vectors unaffected. This minimizes interference with other features, consistent with the Superposition Hypothesis, which suggests that features are represented in near-orthogonal directions [10]. Consequently, this approach enables more robust control over the steering effect, preserving coherence (see Section 5).
• Flexibility. It enables steering the activations for more than 180 degrees, making the accuracy less dependent on the quality of the direction of the extracted features.
this section cite: ['b51', 'b58', 'b1', 'b39', 'b0', 'b58', 'b9']

Section: Preparing Dataset and Models
Datasets. To calibrate the feature (refusal) direction, we construct two datasets:
D (cal)
harmful , which is a split (80%) of the ADVBENCH dataset [59] consisting of 416 harmful instructions; and D (cal) harmful , a random subset of 512 harmless examples from the ALPACA dataset [48]. For evaluating steering effectiveness, we use the remaining 20% of ADVBENCH, denoted as D (eval) harmful , containing 104 samples. To assess general language modeling capabilities, we employ the TINYBENCHMARKS dataset [24], a collection of reduced-scale benchmarks each containing 100 examples: ARC [8], MMLU [15], WINOGRANDE [40], GSM8K [9], TRUTHFULQA [22], and HELLASWAG [56].
Models. We show experimental results on steering the refusal feature on various model families (LLAMA 3 [23], QWEN 2.5 [54], GEMMA 2 [14]) of various sizes (3B to 14B). A full list of models used in this work is presented in Appendix D.
this section cite: ['b59', 'b48', 'b23', 'b7', 'b14', 'b40', 'b8', 'b21', 'b56', 'b22', 'b54', 'b13']

Section: Computing the target feature direction
Extracting Activation Vectors. Following [1], we pass D (cal) harmful and D (cal) harmless through the model and record the activations of the final input token after the normalization layers in each transformer block as recommended by [52]. Note that in each transformer block, there are two normalization layers: before the Attention and before the MLP. As a result, we record the activations at two extraction points per transformer block.
0 4 8 12 16 20 24 28 32 36 40 44 4852 0.2 0.4 0.6 0.8 Extraction Point Norm of Refusal Direction (a) Norm of candidate feature direction at each layer. 0 4 8 12 16 20 24 28 32 36 40 44 48
this section cite: ['b0', 'b52']

Section: Calculating Candidate Directions.
At each extraction point i, we compute a candidate direction using the Difference-in-Means method [3]:
d i feat = h (cal),i harmful -h (cal
),i harmless (i = 1, . . . , M ), where d i feat is the direction at extraction point i, and h (cal),i harmful and i h (cal),i harmless are the means computed over activations from D (cal) harmful and D (cal)
harmless , respectively. Here, M is the number of extraction points, defined as twice the number of Transformer blocks in the model. One candidate direction is computed at each extraction point, yielding a total of M candidate directions.
Choosing One Feature Direction. Among M candidate directions, we choose a feature direction for Angular Steering. Fig. 5b shows high cosine similarity among candidate directions in layers where refusal is strong, suggesting those directions are stable approximations of the true feature. This observation suggests that the similarity between candidate directions can be a promising metric to select the feature direction. In Angular Steering, we choose the candidate direction d feat that is most similar to others as the feature direction. We normalize d feat to make it a unit vector.
this section cite: ['b2']

Section: Remark 1 (Automatic Direction Selection)
Unlike [1], which selects directions manually, we use a simple statistical procedure to choose the feature direction automatically. Though hand-tuning might yield better downstream results, we aim to study steering control rather than maximize performance.
Remark 2 Fig. 4 and Fig. 5 shows that refusal behavior emerges progressively along the depth of the model, stabilizes, and then spikes again near the final layer. We hypothesize that this late spike reflects a filtering step just before token generation and thus omit this point from the list of candidates.
this section cite: ['b0']

Section: Selecting the Steering Plane
We now require a second direction to define the 2D steering plane in Angular Steering. As discussed in Section 4.1, the optimal plane should maximize the influence on the feature of interest while minimizing unintended impacts on other features. While using the Span{h i , d feat } aligns with prior methods like directional ablation and activation addition, we argue against it due to three reasons: (1) prior work suggests that feature directions are layer-independent [33,10,50,1], implying a shared geometry across layers; (2) this span might include other dominant features, risking general degradation [51,49]; and (3) computing rotation at each step is costly. Instead, we propose a fixed plane that isolates the feature of interest. To construct this fixed plane, we perform PCA on the candidate directions d i feat and select the first principal component, d PC0 , as the second axis. This captures variance across layers, which, as shown in prior work [1,52,20,58], reflects variation in approximating the true feature direction. The resulting plane Span( d feat , d PC0 ) thus isolates meaningful variation in the target feature. Fig. 6 shows a smooth directional shift across layers in this plane, supporting the hypothesis that feature strength evolves gradually, making it a natural basis for steering (see Section 5).
this section cite: ['b33', 'b9', 'b50', 'b0', 'b51', 'b49', 'b0', 'b52', 'b19', 'b58']

Section: Putting It All Together: The (Adaptive) Angular Steering Framework
We are now ready to formulate Angular Steering and its adaptive variant.
this section cite: []

Section: Angular Steering Framework
Let P be the 2D subspace spanned by d feat and d PC0 . We compute the orthonormal basis {b 1 , b 2 } of P as follows:
b 1 ← d feat ; b 2 ← d PC0 -( d PC0 • b 1 )b 1 ; b 2 ← b 2 /||b 2 ||.
this section cite: []

Section: Rotation by an Offset Angle.
To rotate within the subspace P by an angle ϕ, the transformation matrix R P ϕ is given as
R P ϕ = I -(b 1 b ⊤ 1 + b 2 b ⊤ 2 ) + [b 1 b 2 ] R ϕ [b 1 b 2 ] ⊤ (1
)
where
I -(b 1 b ⊤ 1 + b 2 b ⊤ 2 )
is the projection to the (d model -2)-dimensional orthogonal complement Q of P and R ϕ is the 2D rotation matrix given as R ϕ = [ cos(ϕ) -sin(ϕ) sin(ϕ) cos(ϕ)
] .
Rotation to a Target Angle. In practice, rather than rotating all activations by a fixed offset, we often want to rotate them to a specific angular position θ, e.g., where a desired behaviour is strongly expressed. A naive approach would involve: (1) projecting the input h onto the steering plane P :
proj P (h) = (b 1 b ⊺ 1 + b 2 b ⊺ 2 ) • h;
(2) computing the current angle ϕ P h,b1 between proj P (h) and b 1 ;
(3) constructing the rotation matrix R P θ-ϕ using Eqn. 1; and (4) applying this matrix to h. However, this is inefficient when θ is fixed and can be optimized by precomputing reusable components.
this section cite: []

Section: Noting that the term
[b 1 b 2 ] R ϕ [b 1 b 2 ]
⊺ in Eqn. 1 is a norm-preserving transformation, we can precompute its effect on the unit vector [1 0] ⊺ and scale the result by |proj P (h)|. This leads to the following efficient formulation for rotating an input h to angle θ:
h steered,θ = R P θ-ϕ h,b 1 • h = h -proj P (h) + |proj P (h)| • [b 1 b 2 ] R θ [1 0] ⊤ ,(2)
where R P θ-ϕ h,b 1
is the rotation matrix defined in Eqn. 1. Here, both the projection matrix (b
1 b ⊺ 1 + b 2 b ⊺ 2 ) and [b 1 b 2 ] R θ [1 0] ⊤ can be precomputed.
this section cite: []

Section: Adaptive Angular Steering Framework
Since inputs from contrastive datasets tend to align with d i feat in opposite directions (Fig. 4), it is unnecessary to rotate all activations uniformly. To increase flexibility and further reduce unintended effects on non-targeted features, we propose an adaptive variant that rotates only activations positively aligned with d feat . In particular, we first compute a conditional mask based on the sign of the projection onto d feat : mask = max(0, sign(proj d feat (x))). Using this mask, Eqn. 2 becomes:
h steered (adaptive),θ = h + mask • ( |proj P (h)| • [b 1 b 2 ] R θ [1 0] ⊤ -proj P (h) )(3)
This formulation adds an additional layer of control and robustness: steering is both restricted to a 2D subspace and selectively applied based on feature alignment. Beyond adjusting the steering angle θ, users may also vary the similarity threshold used in the mask or employ different d i feat across layers. We note that another conditional steering approach has been explored in contemporary work by [19], but activation addition was used as the steering framework instead of rotation. We summarize the algorithms for feature direction extraction, steering plane selection, and angular steering in Appendix B.
this section cite: ['b18']

Section: Controlling the Steering Effect
For inference, we apply Adaptive Angular Steering as described in Eqn. 3 on every normalization module before each Attention and MLP layer. By varying the target angular position θ from 0 to 360 degrees (with 10-degree intervals), we observe that the models change from refusal to compliance and back to refusal again (see Fig. 7). We found that both Angular Steering and Adaptive Angular Steering are effective at varying the steering effect. However, the non-adaptive version runs a risk of breaking the coherence on smaller models, which will be discussed in Section 6.
this section cite: []

Section: Remark 3
In addition to the evaluation of refusal steering presented in this section, we also assess our method's ability to control various emotions, with results reported in Appendix H.
this section cite: []

Section: Evaluation Metrics.
We compute a refusal score using the substring matching method [1], which operates by matching a set of common "refusal substrings" (e.g., "I'm sorry", "As an (a) Refusal score (substring matching [1]) and harmful scores (LLAMAGUARD3 [23], HARM-BENCH [27]).  AI") on the model completion. The score is 1 if at least one such substring is matched and 0 otherwise.
Intuitively, this metric only detects memorized refusal phrases but does not assess coherence and harmfulness, as noted by [1,16,29,36,41]. To evaluate harmfulness, we follow the setup in [1] and use two more complementary evaluation metrics, LLAMAGUARD3 [23] and HARMBENCH [27], which we collectively call harmful scores. These two methods use open-source models to classify whether an input is harmful, in which the score is 1 if the classification is true and 0 otherwise.
Beyond refusal and harmfulness detection, we are interested in how the model's output changes semantically at different level of refusal. Thus, we perform qualitative analysis using a reasoning model QVQ-72B-PREVIEW [38] to classify the generation outputs into 4 classes: direct: The model directly answers the prompt; indirect: The model starts out seemingly unwilling to answer but then still provides with an answer; redirect: The model does not explicitly agree or refuse to answer but provides a tactful response without producing any harmful content; refusal: The model explicitly refuses to answer.
Evaluation along the Steering Circle. Fig. 7 demonstrates that angular steering effectively modulates refusal and safety behaviors. In Fig. 7a, all models show a clear arc of strong alignment-high refusal and low harmful scores-and an opposing arc of weak alignment-low refusal and high harmful scores. These arcs lie in opposite directions within the steering circle, with performance peaking near the center and diminishing outward. Fig. 7b further supports this observation by showing that, for five of six models, refusal dominates in the strong arc, followed by redirect, and then  direct or indirect responses as the angle shifts. Tab.1 reports example completions for each class. GEMMA-2-9B-IT is an exception, displaying the weakest effect yet still following the overall trend.
Steering on a random plane. For completeness, we conduct an ablation study on steering using Adaptive Angular Steering with a random plane. Fig. 14b in Appendix I.2 shows that it has little to no effect on controlling refusal in five out of six tested models.
this section cite: ['b0', 'b0', 'b22', 'b26', 'b0', 'b15', 'b28', 'b36', 'b41', 'b0', 'b22', 'b26', 'b38']

Section: Effects on Model's Performance beyond the Targeted Steering Task
Steering can degrade language modeling ability [43], especially when relying on sensitive hyperparameters [51,58,49,2,20,52], which may lead to incoherent outputs if not carefully tuned [51,49].
In this section, we quantitatively assess the impact of our method on overall LLM performance.
this section cite: ['b43', 'b51', 'b58', 'b49', 'b1', 'b19', 'b52', 'b51', 'b49']

Section: Language Modeling Benchmarks
Method. For each model, we adaptively steer its activation with a 10 • interval along the entire steering circle using Eqn. 3 and evaluate all benchmarks from the TINYBENCHMARKS suite [24].
The results are visualized in Fig. 8a.
this section cite: ['b23']

Section: Results.
Overall, our steering method effectively preserves benchmark accuracies across the entire steering circle, demonstrating strong robustness. Interestingly, in many cases, performance under intervention even surpasses the non-steered baseline.
A notable outlier is QWEN2.5-3B-INSTRUCT, which exhibits a performance drop along the arc from 160 • to 280 • . We attribute this to feature interference [10], where multiple latent features dominate within the chosen steering plane, a phenomenon to which smaller models are more susceptible. The consistent accuracy drop across all benchmarks in this region suggests the model is reacting to a competing feature. For TINYGSM8K, although the model often generates a correct answer, it fails to match the expected format, leading to significantly lower scores under the strict metric compared to the more lenient flexible variant.
It is important to note that for TINYGSM8K, the flexible metric extracts the last numeric value as the final answer, whereas the strict variant assumes a predefined output format. Consequently, these metrics are highly sensitive to formatting variations, leading to noticeable fluctuations in accuracy across different steering angles.
this section cite: ['b9']

Section: Perplexity of the Steered Generations
Smaller Models are More Vulnerable to Interference under Angular Steering. In non-adaptive Angular Steering experiments, 7B-14B models generate coherent outputs throughout the steering circle, while smaller models like LLAMA-3.2-3B-INSTRUCT and QWEN2.5-3B-INSTRUCT often produce incoherent text across a wide arc. Notably, refusal phrases still appear randomly in various languages for LLAMA-3.2-3B-INSTRUCT, and mainly in Chinese for QWEN2.5-3B-INSTRUCT, despite English prompts. This suggests that limited capacity in smaller models leads to feature interference [10], with multiple features entangled in the 2D steering subspace, as discussed in Sections 5 and 6.1.
Method. Motivated by such observations, we analyze the perplexity of the steered generations using the non-steered models and report the results in Fig. 8b. Given an input sequence x, an non-steered LLM π non-steered , the output is modeled by y non-steered ∼ π non-steered (x). Similarly, π steered and y steered denote the steered model and its output, respectively. We denote the perplexity score of x with respect to a model π as P P L π (x). In Fig. 8b, we compare P P L πnon-steered (x||y non-steered ), P P L πnon-steered (x||y steered (non-adaptive) ) and P P L πnon-steered (x||y steered (adaptive) ) for each model and at every 10 • rotation.
Results. Both 3B models exhibit unstable perplexity under non-adaptive steering, indicating vulnerability to interference. For QWEN2.5-3B-INSTRUCT, perplexity remains significantly above baseline across more than half of the circle, aligning with the incoherent outputs discussed earlier.
In contrast, LLAMA-3.2-3B-INSTRUCT shows perplexity closer to baseline, consistent with its behavior of still refusing harmful requests, albeit in different languages.
this section cite: ['b9']

Section: Adaptive Steering effectively preserves coherence.
Fig. 8b reveals that the perplexity of Adaptive Steering is lower, more stable, and closer to no steering than its non-adaptive counterpart, indicating effectiveness at balancing behavior control with coherence and performance.
Alignment masks rather than removes harmful behavior. Perplexity stays near baseline when steering aligns with the target feature, but drops below baseline as it moves toward the jailbroken region. Our perplexity analysis was inspired by [6], which shows that safety alignment mainly affects the first few tokens, while the probability of later harmful tokens remains largely unchanged, suggesting a shortcut where models shift early output distributions without removing harmful behaviors. Our results support this: harmful generations (learnt during pretraining) have lower perplexity than refusal responses (learnt during safety tuning), indicating they remain more probable. While the mechanisms behind safety alignment are still unclear, our findings offer a glimpse into this issue.
this section cite: ['b5']

Section: Concluding Remarks
We propose Angular Steering, a novel activation steering method offering continuous, fine-grained control over large language model behaviors by rotating activation vectors within a two-dimensional subspace. This geometric perspective unifies prior steering techniques, enhancing interpretability and deepening understanding of model mechanisms without compromising general performance. Our adaptive variant further improves robustness by selectively applying steering based on context.
A limitation of Angular Steering is that while promising, it currently relies on heuristically selected steering planes, which might not always generalize optimally across diverse behaviors or architectures. Future work should focus on systematically identifying effective subspaces and extending adaptive strategies to support broader alignment goals.
this section cite: []

Section: A Detailed Derivation: Existing Activation Steering as Special Cases of Steering by Rotation
We will show that, when activation norms are preserved, existing steering techniques are special cases of angular steering but with restricted flexibility: vector addition is limited to less than 180 degrees, and orthogonalization is fixed at 90 degrees.
Formally, let the activation h i be decomposed into components parallel and orthogonal to a unitnorm feature direction d feat (for brevity, here we denote them as h and d respectively):
h = (h • d)d + h ⊥ , where h ⊥ = h -(h • d)d.
Let u = h ⊥ ∥h ⊥ ∥ , and define the initial angle between h and d as:
θ 0 = tan -1 ( ∥h ⊥ ∥ h • d
) .
We define Angular Steering as rotating h by an offset angle ϕ in the plane Span{h, d}, producing a vector: h rot (ϕ) = cos(θ 0 + ϕ)
• d + sin(θ 0 + ϕ) • u.
Now consider vector addition [51], defined as:
h add = h + αd = (h • d + α)d + h ⊥ .
After normalization, the direction becomes:
h add-norm = h add ∥h add ∥ = cos(θ 0 + ϕ add ) • d + sin(θ 0 + ϕ add ) • u,
where ϕ add = tan -1
( ∥h ⊥ ∥ h•d+α ) -θ 0 .
Likewise, directional ablation (orthogonalization) [1], given by:
h ablate = h ⊥ ,
after normalization becomes:
h ablate-norm = u = cos(θ 0 + ϕ ablate ) • d + sin(θ 0 + ϕ ablate ) • u,
with ϕ ablate = π 2 -θ 0 . Thus, when activation norms are fixed, both addition and ablation shift the direction of h in a way that is exactly equivalent to a rotation by some angle ϕ within the plane spanned by h and d. This establishes them as special cases of Angular Steering.
In practice, RMSNorm [57] stabilizes activation norms rather than fixing them to exact values (as shown in [57] and Fig. 10, left). As a result, these methods can still be interpreted as approximate rotations.
this section cite: ['b51', 'b0', 'b57', 'b57']

Section: B Algorithms for Angular Steering
Algorithm 1 Extract Feature Direction Require: Contrastive datasets D harmful , D harmless , model M 1: for each layer i in model do 2: Compute normalized activations h (i) after Attention and MLP 3: Compute mean activation for each dataset: h(i) harmful , h(i) harmless 4: Compute candidate direction: d (i) = h(i) harmful -h(i) harmless 5: end for 6: Select final feature direction d using max average cosine similarity: d = argmax i=1...|layers|   1 |layers| |layers| ∑ j=1 cosine(d (i) , d (j) )   7: Normalize: d = d ∥d∥ Algorithm 2 Select Steering Plane Require: Candidate directions {d (i) }, feature direction d 1: Perform PCA on {d (i) } 2: Let first principal component be d 1stPC 3: Set orthonormal basis for plane: Apply adaptive steering:
h steered = h + mask • (r • v θ -proj P (h)) 7: else 8:
Apply steering:
h steered = h -proj P (h) + r • v θ 9: end if
this section cite: []

Section: C Comparison with existing methods

this section cite: []

Section: C.1 Steering Performance Comparison
In Tab. 2, we compare refusal steering performance between our method, prior approaches, and the no-steering baseline. To ensure a fair and consistent setup, we employ the protocol below:
• Following observations in [1,32,28] that multi-layer interventions yield better results, we apply steering across all layers for methods considered in this study.
• All methods perform steering within the subspace Span(h, d feat ), as in [1,51].
• We conduct hyperparameter tuning for both Angular Steering and Activation Addition. For Activation Addition, tuning is notably more complex and time-consuming, requiring layerwise unbounded coefficients. In contrast, our method only uses a single bounded rotation angle.
Results: Across all evaluated models, our method achieves equal or better refusal performance than existing methods, supporting our theoretical insights. In Tab. 3 and Fig. 9, we examine the models coherence and general performance under two different steering subspaces:
• Span(h, d feat ) (used in [1,51])
• Span( d PC0 , d feat ) (our proposal)
this section cite: ['b0', 'b32', 'b27', 'b0', 'b51', 'b0', 'b51']

Section: C.2.1 Perplexity Analysis
Similar to Fig. 8b, Fig. 9a show comparisons of perplexity scores between steering within Span(d feat , d PC0 ) (ours) and within Span(h, d feat ) (used by exsting work).
In Tab. 3, we report the following metrics: mean, max, minperplexity values across different steering angles; and mean diff the average difference in perplexity between consecutive angles, which indicates the model's sensitivity to small hyperparameter changes.
Results: Steering on Span( d PC0 , d feat ) yields low and stable perplexity, demonstrating strong coherence across steering angles. In contrast, steering on Span(h, d feat ) causes larger fluctuations and higher perplexity, indicating greater sensitivity and frequent coherence breakdowns (e.g., generating gibberish), a finding consistent with our qualitative observations.   Results: Steering within Span( d PC0 , d feat ) preserves performance across most angles. In contrast, steering on Span(h, d feat ) causes significant performance drops, except near 90 • where performance temporarily aligns with the baseline, consistent with prior observations [1].
These results support our hypothesis that Span(h, d feat ) overlaps with unrelated directions, leading to interference. In contrast, our proposed subspace more effectively isolates the target feature, yielding more robust and controllable steering.
this section cite: ['b0']

Section: D Use of existing assets D.1 Models

this section cite: []

Section: E Compute statement
This research was conducted using mainly Nvidia H100 GPUs with 80GB of memory. For each model:
• Constructing the steering plane took about 15 minutes on 1 GPU using TRANSFORMER-LENS [30].
• Pre-generating responses for evaluation took about 10 minutes on 1 GPU using our fork of vLLM [18] as the serving engine.
• Evaluation with substring matching [1], LLAMA 3 GUARD [23] and HARM-BENCH [27] collectively took about 10 minutes on 1 GPU using vLLM [18] as the serving enging.
• Evaluation with LLM-as-a-judge took about 50 minutes on 4 GPUs using vLLM [18] as the serving engine.
• Computing perplexity scores took about 5 minutes on 1 GPU.
• Evaluation with TINYBENCHMARKS [24] took about 4 hours on 1 GPU using vLLM [18] as the serving engine and LM HARNESS [13] as the evaluation device.
this section cite: ['b29', 'b17', 'b0', 'b22', 'b26', 'b17', 'b17', 'b23', 'b17', 'b12']

Section: F Computational and memory complexity analysis
Overall, our method has a time complexity of O(|transformer layers| × d model 2 ) and a memory complexity of O(d model 2 ) where d model is the dimension of the transformer layers' hidden states. For each token at each intervention point, (Adaptive) Angular Steering makes two matrix multiplications and a few element-wise operations. In terms of memory, our formulation enables us to pre-compute one d model × d model matrix and one d model -dimensional vector, which are shared across all extraction points. Below we present the detail analysis of the time and memory complexity of our method.
Recall the Adaptive Angular Steering formula is:
mask = max(0, sign(proj d feat (h))) h steered (adaptive),θ = h + mask • ( |proj P (h)| • [b 1 b 2 ]R θ [1 0] ⊤ -proj P (h) ) with
• h ∈ R dmodel : the activation at some intervention point.
• P : the 2D rotation subspace.
• {b 1 , b 2 } ∈ R dmodel : the orthonormal basis of P .
• θ: the target angular position.
• R θ ∈ R 2×2 : the 2D rotation matrix to θ.
• proj y (x) denotes the projection of x onto y.
The formulation above was chosen with the intention that some components can be pre-computed:
• (b 1 b ⊤ 1 + b 2 b ⊤ 2 ) ∈ R dmodel×dmodel : the projection matrix for proj P (•). • [b 1 b 2 ] R θ [1 0] ⊤ ∈ R dmodel
Hence the complexity of the above operation is: Time (per token): O(|transformer layers| × d model 2 ) (assuming the naive implementation of matrix multiplication)
• Computing proj P (h) takes O(d model 2 ).
• Computing proj d feat (h) takes O(d model )
• Other element-wise operations (sign, max, •, +, -) each takes O(d model ).
• The operation is applied at each intervention point and the number of intervention points is O(|transformer layers|).
Memory: O(d model 2 )
• Storing (b
1 b ⊤ 1 + b 2 b ⊤ 2 ) takes O(d model 2 ) • Storing [b 1 b 2 ] R θ [1 0] ⊤ takes O(d model )
• In our implementation, the rotation plane and target angular position are shared accross intervention points, thus the memory complexity doesn't grow linearly with the number of intervention points. Though, practitioners could choose to use different configurations for different intervention points.
Further highlighting the practical and emprical aspect of our work, we have integrated our method into vLLM [18] -a popular LLM serving engine. We benchmark the generation speed of our method against non-steering baseline and report the results in Tab. 6. Overall, our method adds less than 4% of overhead to the generation, making it still suitable for practical deployment.
Our fork of the vLLM project with Angular Steering integrated can be found at https://github.com/  lone17/vllm/tree/feat/steering.
this section cite: ['b17']

Section: G Broader Impacts
The Angular Steering approach presented in this work has several broader societal impacts. On the positive side, it significantly enhances the control and interpretability of LLMs, enabling their safer deployment across various applications by effectively reducing harmful outputs such as misinformation, biased content, and unethical requests. This enhanced control facilitates alignment with societal norms and ethical standards, potentially increasing public trust and acceptance of AI technologies.
Conversely, there is also a potential for negative impacts. By simplifying fine-grained behavior control, Angular Steering could inadvertently make it easier to generate nuanced harmful or unethical content, such as persuasive misinformation or biased narratives. Although our method does not fundamentally alter the existing risk profile of deploying LLMs, it underscores the need for continued vigilance and improvement in AI safety mechanisms. To responsibly manage these risks, implementing rigorous safeguards, ensuring transparency, and promoting accountability are essential. We advocate ongoing ethical assessment to responsibly guide the deployment and utilization of our proposed method. Calm (140) "Today, I am feeling in a state of profound peace and contentment. The world around me seems to resonate with a harmonious energy, and I am filled with a sense of tranquility and joy. Nature's beauty is more vivid, and the air feels crisp and invigorating. I am in tune with the..." 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 6064 68 0 1000 2000 harmless harmful Extraction Point Activation Norm Loading [MathJax]/extensions/MathMenu.js 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 6468 -0.4 -0.2 0 0.2 0.4 harmless harmful Extraction Point Scalar Projections (a) Qwen2.5-3B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 4852 0 200 400 harmless harmful Extraction Point Activation Norm 0 4 8 12 16 20 24 28 32 36 40 44 48 520.4 -0.2 0 0.2 0.4 harmless harmful Extraction Point Scalar Projections (b) Qwen2.5-7B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 0 500 1000 harmless harmful Extraction Point Activation Norm 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 -0.4 -0.2 0 0.2 0.4 harmless harmful Extraction Point Scalar Projections (c) Qwen2.5-14B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 4852 0 10 20 30 40 50 harmless harmful Extraction Point Activation Norm 0 4 8 12 16 20 24 28 32 36 40 44 48 0 4 8 12 16 20 24 28 32 36 40 44 48 52 5660 0 10 20 30 40 harmless harmful Extraction Point Activation Norm 0 4 8 12 16 20 24 28 32 36 40 44 48 52 5660 -0.5 0 0.5 harmless harmful Extraction Point Scalar Projections (e) Llama-3.1-8B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 0 500 1000 1500 harmless harmful Extraction Point Activation Norm 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 -0.4 -0.2 0 0.2 0.4 harmless harmful Extraction Point Scalar Projections (f) gemma-2-9b-it.   8 12 16 20 24 28 32 36 40 44 48 52 56 60 6468 0 0.2 0.4 0.6 0.8 Extraction Point Norm of Refusal Direction 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 6468 0.15 0.2 0.25 0.3 Extraction Point Mean Cosine Score (a) Qwen2.5-3B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 4852 0.2 0.4 0.6 0.8 Extraction Point Norm of Refusal Direction 0 4 8 12 16 20 24 28 32 36 40 44 4852 0.15 0.2 0.25 0.3 0.35 Extraction Point Mean Cosine Score (b) Qwen2.5-7B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 6468 72 76 80 84 88 92 0 0.2 0.4 0.6 0.8 Extraction Point Norm of Refusal Direction 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 0 4 8 12 16 20 24 28 32 36 40 44 48 52 0 0.5 1 Extraction Point Norm of Refusal Direction 0 4 8 12 16 20 24 28 32 36 40 44 48 52 0.1 0.2 0.3 0.4 Extraction Point Mean Cosine Score (d) Llama-3.2-3B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 0.2 0.4 0.6 0.8 1 1.2 Extraction Point Norm of Refusal Direction 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 0.1 0.2 0.3 0.4 Extraction Point Mean Cosine Score (e) Llama-3.1-8B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 6468 72 76 80 0 0.2 0.4 0.6 0.8 Extraction Point Norm of Refusal Direction 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64
this section cite: []

Section: I Additional Results

this section cite: []

Section: I.1 Activations along the model's depth
Fig. 10 (left) demonstrates that the norm of activation vectors increases exponentially across all tested models as the layer depth increases. This behavior is attributable to the additive nature of the residual stream, where each layer's output accumulates onto the previous state. Interestingly, even models from the same architecture family display different scaling patterns, indicating that activation growth is not only architecture-dependent but also implementation-specific. These observations underscore the necessity of norm-independent steering techniques, as steering strategies relying on raw magnitude can become unstable or ineffective across layers and model variants.
Fig. 10 (right) shows a consistent phenomenon across all evaluated models: activations from contrastive prompts, harmful versus harmless, diverge progressively in geometric space as depth in-
0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 -0.2 0 0.2 harmful harmless
this section cite: []

Section: Extraction Point Scalar Projections
(a) Qwen2.5-3B-Instruct 0 4 8 12 16 20 24 28 32 36 40 44 48 520.2 0 0.2 harmful harmless Extraction Point Scalar Projections (b) Qwen2.5-7B-Instruct. 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 -0.4 -0.2 0 0.2 0.4 harmful harmless Extraction Point Scalar Projections (c) Qwen2.5-14B-Instruct 0 4 8 12 16 20 24 28 32 36 40 44 48 520.5 0 0.5 harmful harmless Extraction Point Scalar Projections (d) Llama-3.2-3B-Instruct.  8 12 16 20 24 28 32 36 40 44 48 52 5660 -0.5 0 0.5 harmful harmless Extraction Point Scalar Projections (e) Llama-3.1-8B-Instruct 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 -0.4 -0.2 0 0.2 0.4 harmful harmless Extraction Point Scalar Projections (f) gemma-2-9b-it. Figure 12: Mean scalar projection activations at each layer onto the chosen feature direction d feat for all tested models.
creases. This increasing separation suggests a universal, model-agnostic internal mechanism in LLMs, whereby behavioral distinctions are gradually amplified layer by layer. Such a trend reveals a directional progression in the models internal representation, reinforcing the hypothesis that feature separation is a fundamental property of transformer-based language models. Fig. 11 further illustrates this progression, focusing on the evolution of the refusal direction. The strength of this feature becomes increasingly prominent in early and middle layers, reaching its maximum influence at a specific intermediate depth before diminishing slightly in later layersa trend echoed in Fig. 12. Importantly, even in the deeper layers where the signal attenuates, the extracted refusal direction continues to serve as a reliable discriminator between activations corresponding to harmful and harmless prompts. This persistent separability affirms the robustness and interpretability of the refusal direction, validating its role as a stable, layer-resilient feature for behavioral control in LLMs.
this section cite: []

Section: I.2 Ablation Study: Steering on a random plane.
To assess the importance of the steering plane, we conducted an ablation study using two setups: (1) steering with a plane defined by one random direction and one feature-aligned direction, and (2) steering with a fully random plane composed of two random directions.
As illustrated in Fig. 14a, where one random direction is combined with the feature direction, most models exhibit noticeably degraded steering performance and less smooth transitions along the steering circle. This degradation suggests that even partial misalignment of the steering plane can distort the intended behavioral modulation. An exception is QWEN2.5-7B-INSTRUCT, which retains robust control, indicating a strong, well-defined internal representation of the refusal direction. LLAMA-3.2-3B-INSTRUCT shows a clear steering effect, but the refusal arc is shifted, suggesting the random component introduces skew that displaces the effective axis of control.
Fig. 14b, where both directions are randomly selected, shows that five of the six tested models exhibit minimal to no steering effect. The only partial exception, QWEN2.5-3B-INSTRUCT, displays erratic behavioral changes with a spiky, non-smooth response curve. Closer inspection reveals these outputs are often incoherent or filled with irrelevant content, indicating instability rather than intentional
-0.5 0 0.5 1 -0.5 0 0.5 0 10 20 30 40 50 60 70 chosen direction 1st PC Extraction Point (a) Qwen2.5-3B-Instruct 0 0.5 1 -0.5 0 0.5 1 0 10 20 30 40 50 chosen direction 1st PC Extraction Point (b) Qwen2.5-7B-Instruct.
0 0.5 1 -0.5 0 0.5 1 0 10 20 30 40 50 chosen direction 1st PC Extraction Point (c) Qwen2.5-14B-Instruct 0 0.5 1 -0.5 0 0.5 0 10 20 30 40 50 chosen direction 1st PC Extraction Point (d) Llama-3.2-3B-Instruct. 0 0.5 1 -0.5 0 0.5 0 10 20 30 40 50 60 chosen direction 1st PC Extraction Point (e) Llama-3.1-8B-Instruct 0 0.5 1 -0.5 0 0.5 0 10 20 30 40 50 60 70 80 chosen direction 1st PC Extraction Point (f) gemma-2-9b-it. 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 harmbench llamaguard3 substring_matching feature direction Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct gemma-2-9b-it (a) Steering on a plane spanned by d feat and a random direction. 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 0°1 0°2 0°3 0°4 0°5 0°6 0°7 0°8 0°90°100°1 10°1 20°1 30°1 40°1 50°1 60°1 70°1 80°1 90°2 00°2 10°2 20°2 30°2 40°2 50°2 60°270°280°2 90°3 00°3 10°3 20°3 30°3 40°3 50°0 0.2 0.4 0.6 0.8 1 harmbench llamaguard3 substring_matching feature direction Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct Qwen2.5-14B-Instruct Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct gemma-2-9b-it (b) Steering on a plane spanned by 2 random directions. Figure 14: Ablation study of steering with random direction(s).
this section cite: []

Section: Supplement to "Angular Steering: Behavior Control via Rotation in Activation Space"
Table
of
this section cite: []

Section: H Steering emotion
To test the ability of our Angular Steering (AS) method in controlling other behaviors, we conduct two experiments with changing the emotion of LLMs' generation. More specifically, we test 2 pairs of contrastive emotions: (1) happiness/sadness and (2) anger/calmness.
We use an approach similar to the one used in [39,58] to construct the dataset, then we follow the process described in Section 4 to compute the rotation subspace.
We evaluate on a subset of the Alpaca dataset [48]. We rotate the activation within the rotation subspace at a 10-degree interval to record the generation at each angle, then use EmoLLM [55] to evaluate the emotion of the generated texts.
Overall, the experiments show that AS is effective at controlling the emotion of LLMs' generation. Along the rotation circle, the LLMs' generation exhibits a clear change from one emotion to the another, evident by qualitative sample generations and the gradual change in the intensity of the target emotion.
We report some sample generations for the two pairs of emotions in Tab. 7 and Tab. 8.
this section cite: ['b39', 'b58', 'b48', 'b55']

Section: Results
For the happiness/sadness case, scores closer to 1.0 indicate higher intensity of happiness.
The "happiness score" starts low at 0 degree then increases gradually to 140 degree. It maintains at a high level from 140 degree to 180 degree before dropping along the range from 290 to 310. Finally, it stays at a low level from 310 to 350 degree.
For the anger/calmness case, scores closer to 1.0 indicate higher intensity of anger. The "anger score" starts high at 0 and maintains at that level until the 50 degree mark. Then it gradually decreases along the range from 60 to 120. It stays at a low level from 120 to 180 degree. Finally it gradually increases along the range from 120 to 180 before gradually raising again along the range from 190 to 180. Then it continues to be high for the rest of the rotation range. • The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [NA] Justification: This paper focuses primarily on empirical methods and demonstrations, rather than theoretical proofs. We provide detailed mathematical derivations of our method and, when possible, claims made in the paper.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: Experimental details such as datasets, splits and models are fully described in Section 3 and Appendix D; evaluation metrics are described in each experiment sections (Section 5 and 6); algorithms are described in Section 4 and Appendix B. We also provide the source code for reproducing our results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We provide open access to the source code in the Supplemental Materials so that the results in the paper can be easily reproduced. Our work uses open-source datasets for experiments and evaluations.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/  guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: Detailed descriptions of datasets, evaluation splits and metrics are included in Section 3 and described in more detail in Section 5 and 6. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We report error bars suitably and correctly defined of the experiments. Guidelines:
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We provide sufficient information on the computer resources for all experiments in Appendix E. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: We discuss broader impacts in Appendix G.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: The paper does not release data or models, hence poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: The paper clearly cites the sources of existing assets used in Appendix D. Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: We include details about training and implementation as well as limitations and code for our proposed method. Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing or human subject research. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: No human subjects or IRB approvals are involved. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core methodological contributions of this research do not rely on LLMs in any important, original, or non-standard way. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Refusal in Language Models Is Mediated by a Single Direction Year: (2024-10)
Ref_id:b1 Title: Steering Large Language Model Activations in Sparse Spaces Year: (2025-02)
Ref_id:b2 Title: Diff-in-means concept editing is worst-case optimal: Explaining a result by sam marks and max tegmark Year: (2023)
Ref_id:b3 Title: Mechanistic Interpretability for AI Safety -A Review Year: (2024-04)
Ref_id:b4 Title: Ntk-aware scaled rope allows llama models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation Year: (2023)
Ref_id:b5 Title: Towards monosemanticity: Decomposing language models with dictionary learning Year: (2023)
Ref_id:b6 Title: Extending Context Window of Large Language Models via Positional Interpolation Year: (2023-06)
Ref_id:b7 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b8 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b9 Title: Toy models of superposition Year: (2022)
Ref_id:b10 Title: Xiangnan He, and Tat-Seng Chua. Alphaedit: Null-space constrained knowledge editing for language models Year: (2025)
Ref_id:b11 Title: Scaling and evaluating sparse autoencoders Year: (2024-06)
Ref_id:b12 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b13 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b14 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b15 Title: Catastrophic jailbreak of open-source llms via exploiting generation Year: (2023)
Ref_id:b16 Title: Style Vectors for Steering Generative Large Language Models Year: (2024-03)
Ref_id:b17 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b18 Title: Manish Nagireddy, and Amit Dhurandhar. Programming refusal with conditional activation steering Year: (2024)
Ref_id:b19 Title: Inference-Time Intervention: Eliciting Truthful Answers from a Language Model Year: (2024-06)
Ref_id:b20 Title: Fairsteer: Inference time debiasing for llms with dynamic activation steering Year: (2025)
Ref_id:b21 Title: Truthfulqa: Measuring how models mimic human falsehoods Year: (2022)
Ref_id:b22 Title: The llama 3 herd of models Year: (2024)
Ref_id:b23 Title: tinybenchmarks: evaluating llms with fewer examples Year: (2024)
Ref_id:b24 Title: Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models Year: (2025-03)
Ref_id:b25 Title: The geometry of truth: Emergent linear structure in large language model representations of true/false datasets Year: (2024)
Ref_id:b26 Title: A standardized evaluation framework for automated red teaming and robust refusal Year: (2024)
Ref_id:b27 Title: The hydra effect: Emergent self-repair in language model computations Year: (2023)
Ref_id:b28 Title: Universal adversarial triggers are not universal Year: (2024)
Ref_id:b29 Title:  Year: ()
Ref_id:b30 Title:  Year: (2022)
Ref_id:b31 Title: Emergent linear representations in world models of self-supervised sequence models Year: (2023-12)
Ref_id:b32 Title: Beyond linear steering: Unified multi-attribute control for language models Year: (2025-11)
Ref_id:b33 Title: The Linear Representation Hypothesis and the Geometry of Large Language Models Year: (2024-07)
Ref_id:b34 Title: YaRN: Efficient Context Window Extension of Large Language Models Year: (2023-11)
Ref_id:b35 Title: Householder pseudo-rotation: A novel approach to activation editing in llms with direction-magnitude perspective Year: (2024)
Ref_id:b36 Title: Finetuning aligned language models compromises safety, even when users do not intend to! arXiv preprint Year: (2023)
Ref_id:b37 Title: Spectral editing of activations for large language model alignment Year: (2024)
Ref_id:b38 Title: Qvq: To see the world with wisdom Year: (2024-12)
Ref_id:b39 Title: Steering llama 2 via contrastive activation addition Year: (2024-08)
Ref_id:b40 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b41 Title: Scalable and transferable black-box jailbreaks for language models via persona modulation Year: (2023)
Ref_id:b42 Title: Representation surgery: theory and practice of affine steering Year: (2024)
Ref_id:b43 Title: A strongREJECT for empty jailbreaks Year: (2024)
Ref_id:b44 Title: Steering without side effects: Improving post-deployment control of language models Year: (2024)
Ref_id:b45 Title: Improving instruction-following in language models through activation steering Year: (2024)
Ref_id:b46 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b47 Title: Analysing the generalisation and reliability of steering vectors Year: (2024)
Ref_id:b48 Title: Stanford alpaca: An instruction-following llama model Year: (2023)
Ref_id:b49 Title: Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet Year: (2024)
Ref_id:b50 Title: Linear Representations of Sentiment in Large Language Models Year: (2023-10)
Ref_id:b51 Title: Steering Language Models With Activation Engineering Year: (2024-10)
Ref_id:b52 Title: A Language Model's Guide Through Latent Space Year: (2024-02)
Ref_id:b53 Title: Semantics-adaptive activation intervention for llms via dynamic steering vectors Year: (2024)
Ref_id:b54 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b55 Title: Emollm: Multimodal emotional understanding meets large language models Year: (2024)
Ref_id:b56 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019)
Ref_id:b57 Title: Root Mean Square Layer Normalization Year: (2019-10)
Ref_id:b58 Title: Representation Engineering: A Top-Down Approach to AI Transparency Year: (2023-10)
Ref_id:b59 Title: Universal and transferable adversarial attacks on aligned language models Year: (2023)
