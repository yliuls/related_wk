Title: Provable Gradient Editing of Deep Neural Networks
Abstract: In explainable AI, DNN gradients are used to interpret the prediction; in safetycritical control systems, gradients could encode safety constraints; in scientificcomputing applications, gradients could encode physical invariants. While recent work on provable editing of DNNs has focused on input-output constraints, the problem of enforcing hard constraints on DNN gradients remains unaddressed. We present ProGrad, the first efficient approach for editing the parameters of a DNN to provably enforce hard constraints on the DNN gradients. Given a DNN N with parameters θ, and a set S of pairs (x x x, Q) of input x x x and corresponding linear gradient constraints Q, ProGrad finds new parameters θ θ θ such that (x x x,Q)∈S ∂ ∂x x x N(x x x; θ θ θ) ∈ Q while minimizing the changes ∥ θ θ θ -θ∥. The key contribution is a novel conditional variable gradient of DNNs, which relaxes the NP-hard provable gradient editing problem to a linear program (LP), enabling ProGrad to use an LP solver to efficiently and effectively enforce the gradient constraints. We experimentally evaluated ProGrad via enforcing (i) hard Grad-CAM constraints on IMAGENET ResNet DNNs; (ii) hard Integrated Gradients constraints on Llama 3 and Qwen 3 LLMs; (iii) hard gradient constraints in training a function-approximation DNN as a proxy for safety constraints in control systems and physical invariants in scientific applications. The results highlight the unique capability of ProGrad in enforcing hard constraints on DNN gradients. class: stingray 1 (a) Original image. Cosine: 0% IoU: 0% 1 (b) Reference Grad-CAM on the original image. misclassified: coral reef 1 (c) Misclassified Gaussian-noise corrupted image. Cos: 34.66% IoU: 4.35% 1 (d) Deviated Grad-CAM on the corrupted image. Cos: 100% IoU: 100%

Section: Introduction
Incorporating constraints into deep neural networks (DNNs) can enable learning with less data (e.g., in scientific domains) and provide guarantees about their behavior (e.g., in safety-critical applications). This has led to many recent works on provable editing of DNNs [41,40], which enforce hard constraints on input-output behavior of DNNs. However, the problem of enforcing hard constraints on the gradients of DNNs remains unaddressed. This paper presents ProGrad, which addresses the provable gradient editing problem defined below: Definition 1.1. Given a DNN N : R n → R m with parameters θ, a set S ⊆ x x x | x x x ∈ R n × Q | Q ⊆ R m×n of pairs (x x x, Q), where x x x ∈ R n is an input and Q def = J J J ∈ R m×n | A A A vec(J J J) ≤ b b b is the corresponding linear constraints on the Jacobian ∂ ∂x x x N(x x x; θ θ θ) ∈ R m×n of the DNN N with respect to the input x x x, we use vec(J J J) to denote the Jacobian matrix J J J flattened as a vector. The provable gradient editing problem is to find new parameters θ θ θ such that min ∥ θ θ θ -θ∥ s.t.       In other words, provable gradient editing aims to make minimal changes to the parameters of a DNN to ensure that the DNN is guaranteed to satisfy any affine constraints involving the gradients of the DNN with respect to its input. Hard constraints on the gradient of a function with respect to its input are common in many scientific applications and safety-critical control systems. As an illustrative example, consider the function f (x) in Figure 1(a) whose gradient d dx f should be bounded by g u (x) = x and g l (x) = -x over the domain [0, 3π]. Prior regularization-based training approaches can be used to enforce soft constraints on the gradient. However, as shown in Figure 1(b), although such a DNN N 1 achieves good (low) output and gradient errors, its gradient d dx N 1 exceeds the bounds g u and g l and violates the hard gradient constraints, hence unsafe. In contrast, Figure 1(c) shows a safe DNN N 2 edited by our method ProGrad to satisfy the hard constraints on the gradient, which also achieves better (lower) output and gradient errors.
Another natural application of gradient constraints is to ensure that the DNN has the appropriate gradient-based interpretation for a given input, which is important for explainable AI. Figure 2 shows a use case where we enforce an expected Grad-CAM attribution (Definition 3.5) for an IMAGENET ResNet-152 DNN; Figure 3 shows a use case where we enforce an expected Integrated Gradients attribution (Definition 3.4) for a Llama 3 LLM [9].
Figure 3: Enforcing hard Integrated Gradients (IG) constraints on a Llama 3 LLM [9]. The sentence is from the Stanford Sentiment Treebank 2 (SST-2) dataset [36]. The first row shows the reference IG from a larger LLM, which correctly classifies the sentiment of the sentence as positive.
The second row shows the deviated IG from the smaller LLM, which misclassifies the sentiment of the sentence as negative. The third row shows the edited IG from the repaired smaller LLM edited by ProGrad, whose IG is enforced to be close to the reference IG from the larger LLM, and correctly classifies the sentiment of the sentence as positive. See Section 5.2 for more details.
To the best of our knowledge, ProGrad is the first efficient approach for enforcing hard constraints on the gradients of a DNN that runs in polynomial time in the size of the edited layers (Theorem 4.3).
Our key contribution is a novel conditional variable gradient of DNNs (Definition 4.2), which relaxes the NP-hard provable gradient editing problem to a linear programming (LP) problem, enabling ProGrad to use an LP solver to efficiently and effectively enforce the gradient constraints.
this section cite: ['b40', 'b39', 'b8', 'b8', 'b35']

Section: Related Work
There have been many recent works on incorporating constraints into deep learning. These can be categorized into four categories based on whether they treat constraints as soft or hard constraints, and whether they support input-output constraints or gradient constraints.
For soft constraints, regularized training modifies the loss function to incorporate constraints as regularization and does not guarantee constraint satisfaction. There have been many recent such approaches for incorporating input-output constraints [14,45,23,5,12,17,47,38]. Certified training [21,7,31,1,24,22,20] is a type of regularized training geared towards adversarial robustness. For gradient constraints, Park et al. [25] present a technique for preserving the Grad-CAM attribution when performing network compression. They proposed a new loss function that incorporates the match between the attribution maps during the fine-tuning stage of compression, and present three variations of this matching loss function: EWA, SWA and SSWA. Similar approaches are also used in visual grounding for visual question answering tasks [30]. Physics-informed neural networks (PINNs) [28,3] is another type of regularized training that incorporates DNN gradients to encode physics constraints to solve differential equations using DNNs.
For hard constraints, directly using an SMT solver to incorporate input-output constraints is inefficient and does not scale beyond small DNNs [6,19,8]. More efficient approaches are based on relaxing the problem to solving an LP problem [41,37,40]. APRNN [41] is efficient in enforcing arbitrary affine output constraints for input points. The state-of-the-art provable editing approaches APRNN and PREPARED [40] are also able to handle affine output constraints for input polytopes. However, they do not handle gradient constraints.
To the best of our knowledge, ours is the first approach to efficiently enforce hard constraints on the gradients of DNNs. Though the current work focuses on handling input points, future work could enforce gradient constraints on input polytopes by adapting ideas from APRNN and PREPARED.
DNN verification aims to determine whether a DNN satisfies a given input-output [34,35,50,43,33,49,46,4,2,44] or gradient property such as monotonicity and Lipschitz robustness [15,16,32,13].
this section cite: ['b13', 'b44', 'b22', 'b4', 'b11', 'b16', 'b46', 'b37', 'b20', 'b6', 'b30', 'b0', 'b23', 'b21', 'b19', 'b24', 'b29', 'b27', 'b2', 'b5', 'b18', 'b7', 'b40', 'b36', 'b39', 'b40', 'b39', 'b33', 'b34', 'b49', 'b42', 'b32', 'b48', 'b45', 'b3', 'b1', 'b43', 'b14', 'b15', 'b31', 'b12']

Section: Preliminaries
We use x x x ∈ R to denote a scalar, x x x ∈ R m to denote a column vector, and W W W ∈ R n×m to denote a matrix. Variables in blue with a hat denote LP decision variables, e.g., x x x, z z z, W W W. Variables in blue with a tilde denote the conditional variables, e.g., x x x, z z z. Definition 3.1 (DNN). Consider an L-layer feed-forward ReLU deep neural network (DNN) N with parameters θ
def = {W W W (0) , W W W (1) , . . . , W W W (L-1) , b b b (0) , b b b (1) , . . . , b b b (L-1) }. For each layer 0 ≤ ℓ < L, we use z z z (ℓ) def = W W W (ℓ) x x x (ℓ) + b b b (ℓ)(2)
to denote the pre-activation output z z z (ℓ) after the affine transformation. For a non-last layer ℓ < L-1,
x x x (ℓ+1) def = diag 1 1 1 z z z (ℓ) >0 z z z (ℓ)(3)
denotes the layer output x x x (ℓ+1) after the ReLU activation, where diag 1 1 1 z z z (ℓ) >0 is a diagonal matrix of the indicator vector 1 1 1 z z z (ℓ) >0 , whose i-th element is 1 if z z z (ℓ) i > 0, or 0 otherwise. For the last layer ℓ = L-1, we assume no ReLU activation and use x x x (L) def = z z z (L-1) to denote the layer output. Given the DNN input x x x ∈ R n , we have the first-layer input x x x (0) def = x x x and the network output N (x x x; θ) def = x x x (L) . ■ Definition 3.2 (Gradient of DNNs). For an L-layer ReLU DNN N : R n → R m with parameters θ, the gradient (Jacobian) ∂ ∂x x x N (x x x; θ) of the DNN output N (x x x; θ) ∈ R m with respect to the input x x x ∈ R n is defined by the chain rule as
∂ ∂x x x N (x x x; θ) def = 0 ℓ=L-1 ∂x x x (ℓ+1) ∂z z z (ℓ) ∂z z z (ℓ) ∂x x x (ℓ)(4)
For the ReLU activation x x x (ℓ+1) def = diag 1 1 1 z z z (ℓ) >0 z z z (ℓ) of a non-last layer ℓ < L-1, the gradient ∂x x x (ℓ+1) ∂z z z (ℓ)   of the ℓ-th layer post-activation output x x x (ℓ+1) with respect to the pre-activation output z z z (ℓ) is
∂x x x (ℓ+1) ∂z z z (ℓ) def = diag 1 1 1 z z z (ℓ) >0(5)
and for the last layer ℓ = L-1 without ReLU activation, ∂x x x (L) ∂z z z (L-1) def = diag(1 1 1) is the identity matrix.
For the affine transformation z z z (ℓ
) def = W W W (ℓ) x x x (ℓ) + b b b (ℓ)
, the gradient ∂z z z (ℓ) ∂x x x (ℓ) of the pre-activation output z z z (ℓ) with respect to the ℓ-th layer input x x x (ℓ-1) is defined as
∂z z z (ℓ) ∂x x x (ℓ) def = W W W (ℓ)(6)
■
this section cite: []

Section: Conditional variable output of DNNs for provable output editing
Our method ProGrad extends the conditional variable output of DNNs, introduced by APRNN [41], to the conditional variable gradient of DNNs. We now present how APRNN focuses on the forward pass and constructs the conditional variable output of DNNs. For clarity, we assume the first-layer weight and all-layer biases are variables. In practice, APRNN can freeze the first few layers and only make the rest of the DNN have variable parameters. Definition 3.3 (Conditional variable output of DNNs). Consider an L-layer feed-forward ReLU DNN N with parameters θ θ θ def = { W W W (0) , W W W (1) , . . . , W W W (L-1) , b b b (0) , b b b (1) , . . . , b b b (L-1) }, where the first-layer weight W W W (0) and all-layer biases b b b (ℓ) are variables. For an input x x x ∈ R n , the conditional variable output N (x x x; θ θ θ) ∈ R m with linear activation condition φ φ φ is a linear expression over θ θ θ that is sound if the condition φ φ φ is satisfied. In other words, for any assignment θ to the variable parameters θ θ θ that satisfies the activation condition φ φ φ, N (x x x; θ) = N (x x x; θ). Next we define the conditional variable output and condition φ φ φ (ℓ) for each layer ℓ.
The pre-activation conditional variable output z z z (0) for the first layer ℓ = 0 with variable weight is
z z z (0) def = W W W (0) x x x (0) + b b b (0)(7)
The pre-activation conditional variable output z z z (ℓ) for other layers 0 < ℓ < L with constant weight is
z z z (ℓ) def = W W W (ℓ) x x x (ℓ) + b b b (ℓ)(8)
The conditional variable layer output x x x (ℓ+1) for a non-last layer ℓ < L-1 is defined as
x x x (ℓ+1) def = diag 1 1 1 z z z (ℓ) >0 z z z (ℓ)(9)
with the ℓ-th layer activation condition φ φ φ (ℓ)   φ φ φ (ℓ
) def = diag 1 1 1 z z z (ℓ) >0 z z z (ℓ) > 0 ∧ diag 1 1 1 z z z (ℓ) ≤0 z z z (ℓ) ≤ 0 (10
)
The activation condition φ φ φ (ℓ) is constraining the ReLU activation pattern of the pre-activation conditional variable output z z z (ℓ) to be the same as a constant pre-activation output z z z (ℓ) , so that if φ φ φ (ℓ) is satisfied, diag 1 1 1 z z z (ℓ) >0 = diag 1 1 1 z z z (ℓ) >0 . The choice of the constant z z z (ℓ) is arbitrary, the default choice is the pre-activation output of the original DNN N . For the last layer ℓ = L-1 without ReLU activation, x x x (L) def = z z z (L-1) and φ φ φ (L-1) def = ⊤.
Given the DNN input x x x (0) def = x x x, we have the conditional variable network output N (x x x; θ θ θ)
def = x x x (L)
with the activation condition φ φ φ def = ℓ φ φ φ (ℓ) . ■
this section cite: ['b40']

Section: Gradient-based interpretation methods
Definition 3.4 (Integrated Gradients [39]). Given an input x x x and a baseline input x x x 0 . Let x x x α def = x x x 0 +αd d d be the linearly interpolated points between the baseline x x x 0 and the input x x x, where α ∈ [0, 1] and d d d def = x x xx x x 0 . The integrated gradients (IG) from x x x 0 to x x x is defined as the integral of the gradients ∂ ∂x x x N (x x x α ) along the path, then multiplied by the difference vector d d d:
IG(x x x) def = d d d ⊙ 1 α=0 ∂ ∂x x x N (x x x α )dα(11)
where ⊙ denotes the element-wise product. In practice, this integral is approximated by a left-Riemann sum with m steps:
IG approx (x x x) def = d d d ⊙ 1 m m k=0 ∂ ∂x x x N (x x x k m )(12)
■ Definition 3.5 (Grad-CAM [29]). Given a convolutional DNN N and input x x x, let y y y ∈ R C denote the DNN output, and z z z ∈ R K×H×W denote the pre-activation output of the last convolutional layer of the DNN where K is the number of channels, and H and W are the height and width of the feature map. The Grad-CAM localization map L L L c ∈ R H×W is defined as
L L L c def = k α α α c k z z z k where α α α c k def = 1 H × W i j ∂y y y c ∂z z z k,i,j(13)
where the neuron importance weights α α α c ∈ R K for z z z is the average gradient of the output class c with respect to each channel k of the pre-activation output z z z. Optionally, one can simplify the Grad-CAM localization map L L L c by applying a ReLU activation to ignore the negative attributions. In this paper, we consider the Grad-CAM attribution with any sign, hence don't apply the ReLU activation to L L L c . ■
this section cite: ['b38', 'b28']

Section: Approach
This section presents ProGrad, which solves the provable gradient editing problem of DNNs using linear programming (LP). For ease of exposition, we consider the following provable gradient editing problem for a scalar-valued DNN N : R n → R with gradient ∂ ∂x x x N (x x x; θ) ∈ R n , and assume the first-layer weight and all-layer biases are variables. In practice, ProGrad allows editing only the last few layers and freezing the rest of the DNN, enabling efficient editing because the size of the LP only depends on the edited layers instead of the entire DNN. We defer the extension to the general provable editing problem of Definition 1.1 for vector-valued DNNs, as well as handling multiple inputs and editing only the last few layers to Appendix B. The proofs for all theorems can be found in Appendix A. Definition 4.1. Given a DNN N : R n → R with parameters θ, and an input x x x ∈ R n . The provable gradient editing problem is to find new parameters θ θ θ that
min ∥ θ θ θ -θ∥ s.t. ∂ ∂x x x N (x x x; θ θ θ) ∈ Q (14
)
where ∂ ∂x x x N (x x x; θ θ θ) ∈ R n denotes the gradient of the scalar output of N (x x x; θ θ θ) ∈ R with respect to the input x x x with new parameters θ θ θ.
Q def = z z z ∈ R n | A A Az z z ≤ b b b is a convex polytope denoting the linear constraints for the gradient ∂ ∂x x x N (x x x; θ θ θ).
this section cite: []

Section: Conditional variable gradient of DNNs
Consider a DNN N with variable parameters θ θ θ and input x x x. The exact variable gradient ∂ ∂x x x N (x x x; θ θ θ) of N is highly non-linear, involving quadratic terms from the multiplication between weights and disjunctions from the ReLU activation. Our key insight is to relax the non-linear ∂ ∂x x x N (x x x; θ θ θ) to a linear conditional variable gradient ∂ ∂x x x N (x x x; θ θ θ) that is sound under a linear activation condition φ φ φ. Definition 4.2. The conditional variable gradient ∂ ∂x x x N (x x x; θ θ θ) of a DNN N in terms of θ θ θ with respect to the input x x x, under a poly-size linear activation condition φ φ φ, is a poly-size linear expression that equals the exact variable gradient ∂ ∂x x x N (x x x; θ θ θ) if the activation condition φ φ φ is satisfied.
In other words, for any assignment θ to the variable parameters θ θ θ that satisfies the activation condition φ φ φ, ∂ ∂x x x N (x x x; θ) = ∂ ∂x x x N (x x x; θ). The size of the condition φ φ φ and the expression ∂ ∂x x x N (x x x; θ θ θ) is polynomial in the size of the edited layers of the DNN, i.e., the number of edited layers, parameters, and the input and output dimensions of each edited layer. In practice, ProGrad allows editing only the last few layers of a DNN. We defer the construction of such conditional variable gradient ∂ ∂x x x N (x x x; θ θ θ) of N to Section 4.3, and first show how it can be used to solve the provable gradient editing problem.
this section cite: []

Section: Provable gradient editing via conditional variable gradient of DNNs
The following theorem shows how the conditional variable gradient of DNNs can be used to solve the provable gradient editing problem. Theorem 4.3. Given a provable gradient editing problem (Definition 4.1) for DNN N and parameters θ with input x x x and a gradient constraint
Q def = z z z ∈ R n | A A Az z z ≤ b b b . Let ∂ ∂x x x N (x x x; θ θ θ)
be the conditional variable gradient of the DNN N with respect to the input x x x, under the activation condition φ φ φ. The following linear program can be solved in polynomial time in the size of the edited layers of the DNN N , and its solution is a solution to the provable gradient editing problem.
min ∥ θ θ θ -θ∥ s.t. φ φ φ ∧ A A A ∂ ∂x x x N (x x x; θ θ θ) ≤ b b b (15
)
this section cite: []

Section: Conditional variable gradient of fully-connected ReLU DNN
This section presents how ProGrad focuses on the backward pass and constructs the conditional variable gradient of a fully-connected ReLU DNN N with respect to the input x x x and its parameters θ θ θ.  ℓ) are variables. For an input x x x ∈ R n , the conditional variable gradient ∂ ∂x x x N (x x x; θ θ θ) of a DNN N in terms of θ θ θ with respect to the input x x x is defined by the chain rule as
∂ ∂x x x N (x x x; θ θ θ) def = 0 ℓ=L-1 ∂ x x x (ℓ+1) ∂ z z z (ℓ) ∂ z z z (ℓ) ∂ x x x (ℓ)(16)
with the activation condition φ φ φ def = ℓ φ φ φ (ℓ) defined from each layer ℓ.
For the conditional ReLU activation x x x (ℓ+1
) def = diag 1 1 1 z z z (ℓ) >0 z z z (ℓ) of non-last layer ℓ < L -1, the conditional variable gradient ∂ x x x (ℓ+1) ∂ z z z (ℓ) of the ℓ-th layer post-activation output x x x (ℓ+1) with respect to the pre-activation output z z z (ℓ) is ∂ x x x (ℓ+1) ∂ z z z (ℓ) def = diag 1 1 1 z z z (ℓ) >0(17)
with the ℓ-th layer activation condition φ φ φ (ℓ)   φ φ φ (ℓ
) def = diag 1 1 1 z z z (ℓ) >0 z z z (ℓ) > 0 ∧ diag 1 1 1 z z z (ℓ) ≤0 z z z (ℓ) ≤ 0 (18
)
The activation condition φ φ φ (ℓ) is constraining the ReLU activation pattern of the pre-activation conditional variable output z z z (ℓ) to be the same as a constant pre-activation output z z z
(ℓ) , so that if φ φ φ (ℓ) is satisfied, diag 1 1 1 z z z (ℓ) >0 = diag 1 1 1 z z z (ℓ) >0
. The choice of the constant z z z (ℓ) is arbitrary, the default choice is the pre-activation output of the original DNN N . For the last layer ℓ = L-1 without ReLU activation,
∂ x x x (L) ∂ z z z (L-1)
def = diag(1 1 1) is the identity matrix.
As seen in Equations 18 and 10, ProGrad and APRNN share the same idea of using activation condition to constrain each input in the edit set to lie on a specific linear piece of the edited DNN. Note that the linear piece is not fixed but variable, because it is expressed as a closed-form linear expression in terms of the editable DNN parameters.
For the conditional affine transformation z z z (ℓ
) def = W W W (ℓ) x x x (ℓ) + b b b (ℓ)
of the non-first layer ℓ > 0 with constant weight W W W (ℓ) and conditional variable input x x x (ℓ) , the conditional variable gradient ∂ z z z (ℓ) ∂ x x x (ℓ) of the pre-activation output z z z (ℓ) with respect to the layer input
x x x (ℓ) is ∂ z z z (ℓ) ∂ x x x (ℓ) def = W W W (ℓ)(19)
For the conditional affine transformation z z z (0
) def = W W W (0) x x x (0) + b b b (0)
of the first layer ℓ = 0 with variable weight W W W (0) and constant input x x x (0) , the conditional variable gradient ∂ z z z (0) ∂x x x (0) of the pre-activation output z z z (0) with respect to the layer input
x x x (0) is ∂ z z z (0) ∂x x x (0) def = W W W (0)(20)
■ Theorem 4.5. The conditional variable gradient constructed in Definition 4.4 is valid and satisfies the conditions stated in Definition 4.2.
this section cite: []

Section: Experimental Evaluation
We have implemented ProGrad in PyTorch [26] and use Gurobi [10] as the LP solver. All experiments were run on a machine with dual Intel Xeon Platinum 8362 Processors, 32-Core 2.8GHz with 1.5 TB of memory, SSD, and NVIDIA H100 GPU with 80 GB of GPU memory running Ubuntu 22.04. Additional details can be found in Appendix C.
this section cite: ['b25', 'b9']

Section: Enforcing hard Grad-CAM constraints on ResNet DNNs for IMAGENET
In this experiment, we edit ResNet152 and ResNet50 DNNs from torchvision [18] so that the Grad-CAM attributions [29] for a set of images are ε-close to their expected attributions. We compare ProGrad to the state-of-the-art Grad-CAM fine-tuning methods EWA, SWA and SSWA [25].
Edit set. The edit set consists of misclassified images that have deviated Grad-CAM attributions for their expected class. These images are from the IMAGENET-C dataset [11] that are corrupted with Gaussian noise, whose original uncorrupted version is correctly classified. For each misclassified image in the edit set, we take the Grad-CAM attribution of the corresponding original correctlyclassified image as the expected Grad-CAM. For each DNN, we construct two such edit sets: (i) 100 images from the first 50 classes with ε=1e-2; (ii) 1,000 images from the first 200 classes with ε=5e-2.
Grad-CAM constraints. For each image x x x in the edit sets and its expected Grad-CAM L L L for the expected class, let L L L ′ denote the Grad-CAM on the edited DNN N ′ ; let L L L n and L L L ′ n be the min-max-normalized expected and edited Grad-CAMs; we use ∥L L L n -L L L ′ n ∥ ∞ ≤ ε as the constraint. Results. As shown in Table 1, ProGrad is the only method able to enforce the hard Grad-CAM constraints, achieving a 100% constraint satisfaction rate (Constr. Sat.), and it achieves the best accuracy (Acc.). ProGrad also achieves the best similarity between the edited and expected Grad-CAM as measured by cosine similarity (Min. Cos.) and intersection over union (Min. IoU). In the ResNet152 experiment, ProGrad took 22 minutes for 100 images and 3 hours 51 minutes for 1,000 images; in the ResNet50 experiment, ProGrad took 25 minutes for 100 images and 3 hours 37 minutes for 1,000 images. For each baseline, we performed a grid search over hyperparameters with a time limit of 12 hours, and report the best results with the highest constraint satisfaction rate (Constr. Sat.). Although the baselines (EWA, SWA and SSWA) can improve these similarity metrics, they are unable to enforce the hard constraints.
this section cite: ['b17', 'b28', 'b24', 'b10']

Section: Enforcing hard Integrated Gradients constraints on LLMs for SST-2
In this experiment, we edit Llama-3.2-1b-Instruct [9] and Qwen3-1.7B [48] LLMs so that the Integrated Gradients (IG) attributions [39] for a set of sentences are ε-close to their expected IG attributions as determined by the corresponding teacher LLMs Llama-3.1-8b-Instruct and Qwen3-8B. We compare ProGrad against supervised fine-tuning (SFT) and direct preference optimization (DPO) [27] as baselines.
Edit set. The edit set consists of misclassified sentences that have deviated IG attributions for the expected answer token. These sentences are misclassified samples from the Stanford Sentiment Treebank 2 (SST-2) [36] training set, but are correctly classified by the corresponding teacher LLM. For each sentence in the edit set, we take the corresponding IG from the teacher LLM as the expected IG attribution. For each LLM, we construct two such edit sets: (i) the first 100 misclassified sentences from SST-2 with ε=5e-2; (ii) the first 200 misclassified sentences from SST-2 with ε=1e-1.
IG constraints. For each sentence x x x in the edit set and the corresponding expected IG L L L for the expected answer token, let L L L ′ denote the IG on the edited LLM N ′ ; let L L L n and L L L ′ n denote the min-max-normalized expected and edited IGs. We use ∥L L L n -L L L ′ n ∥ ∞ ≤ ε as the IG constraint. Results. As shown in Table 2, ProGrad is the only method able to enforce the hard IG constraints and achieves the best accuracy in three out of four experiments. ProGrad also achieves the best similarity between the edited and expected IG as measured by cosine similarity (Cos.) and intersection over union (IoU). In the Llama-3.2-1b-Instruct experiment, ProGrad took 9 minutes for 100 samples, and 10 minutes for 200 samples. In the Qwen3-1.7B experiment, ProGrad took 28 minutes for 100 samples, and 32 minutes for 200 samples. For each baseline, we performed a grid search over hyperparameters with a time limit of 4 hours, and report the best results with the highest constraint satisfaction rate (Constr. Sat.). The baselines (SFT and DPO) failed to enforce the hard constraints, barely improved the similarity metrics, and in most cases decreased the accuracy of the edited LLM on the SST-2 validation set.
and the gradient dN dx of DNN N approximates the gradient df dx : R → R of the target function f :
d dx f (x) = x sin(x)(22)
Gradient constraints. The target gradient function d dx f (x) = x sin(x) is bounded by the upper bound function g u (x) = x and the lower bound function g l (x) = -x, and we aim to enforce this hard constraint on the DNN gradient d dx N for all training samples ∀x ∈ D, -x ≤ d dx N (x) ≤ x. Results. Table 3 presents the results of the experiment. Although regularization-based training on both the DNN output and gradient can achieve good (low) errors, the trained DNN is not free of the gradient constraints violations. Applying ProGrad after the regularization-based training to enforce the hard gradient constraints as well as minimize the gradient and output errors can achieve 0% gradient constraints violation and further improve (decrease) the gradient and output errors. ProGrad took 10 seconds to edit this DNN.
this section cite: ['b8', 'b47', 'b38', 'b26', 'b35']

Section: Conclusion
We have presented ProGrad, the first efficient approach for provable gradient editing of DNNs that runs in polynomial time in the size of the edited layers. We presented a novel method for constructing conditional variable gradient of DNNs, enabling ProGrad to use an LP solver to find an edit. To demonstrate the effectiveness of ProGrad, we evaluated ProGrad in enforcing hard Grad-CAM constraints on ResNet DNNs for IMAGENET, enforcing hard Integrated Gradients constraints on Llama 3 and Qwen 3 LLMs, and enforcing hard gradient constraints in training a function-approximation DNN. The results highlight the unique capability of ProGrad in enforcing hard constraints on DNN gradients.
this section cite: []

Section: References
Ref_id:b0 Title: Adversarial training and provable defenses: Bridging the gap Year: (2020)
Ref_id:b1 Title: Fast and precise certification of transformers Year: (2021)
Ref_id:b2 Title: Scientific machine learning through physics-informed neural networks: Where we are and what's next Year: (2022)
Ref_id:b3 Title: Complete verification via multineuron relaxation guided branch-and-bound Year: (2022)
Ref_id:b4 Title: DL2: training and querying neural networks with logic Year: (2019)
Ref_id:b5 Title: Minimal modifications of deep neural networks using verification Year: (2020)
Ref_id:b6 Title: On the effectiveness of interval bound propagation for training verifiably robust models Year: (2019)
Ref_id:b7 Title: Deepsade: Learning neural networks that guarantee domain constraint satisfaction Year: (2024-03)
Ref_id:b8 Title:  Year: (2024)
Ref_id:b9 Title: LLC. Gurobi Optimizer Reference Manual Year: (2022)
Ref_id:b10 Title: Benchmarking neural network robustness to common corruptions and perturbations Year: (2019-05-06)
Ref_id:b11 Title: Multiplexnet: Towards fully satisfied logical constraints in neural networks Year: (2022-06)
Ref_id:b12 Title: Verification of neural control barrier functions with symbolic derivative bounds propagation Year: (2025-11)
Ref_id:b13 Title: Harnessing deep neural networks with logic rules Year: (2016-08)
Ref_id:b14 Title: A general construction for abstract interpretation of higher-order automatic differentiation Year: (2022)
Ref_id:b15 Title: Synthesizing precise static analyzers for automatic differentiation Year: (2023)
Ref_id:b16 Title: Xiaoxing Ma, and Jian L\"{u}. Learning with logical constraints but without shortcut satisfaction Year: (2023)
Ref_id:b17 Title: TorchVision maintainers and contributors. Torchvision: Pytorch's computer vision library Year: (2016)
Ref_id:b18 Title: Mlic: A maxsat-based framework for learning interpretable classification rules Year: (2018)
Ref_id:b19 Title: Connecting certified and adversarial training Year: (2023)
Ref_id:b20 Title: Differentiable abstract interpretation for provably robust neural networks Year: (2018)
Ref_id:b21 Title: Certified training: Small boxes are all you need Year: (2023)
Ref_id:b22 Title: A primal dual formulation for deep learning with constraints Year: (2019)
Ref_id:b23 Title: Ibp regularization for verified adversarial robustness via branch-and-bound Year: (2023)
Ref_id:b24 Title: Attribution preservation in network compression for reliable network interpretation Year: (2020)
Ref_id:b25 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b26 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b27 Title: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations Year: (2019)
Ref_id:b28 Title: Grad-CAM: visual explanations from deep networks via gradient-based localization Year: (2020)
Ref_id:b29 Title: Taking a HINT: leveraging explanations to make vision and language models more grounded Year: (2019)
Ref_id:b30 Title: Fast certified robust training with short warmup Year: (2021)
Ref_id:b31 Title: Efficiently computing local lipschitz constants of neural networks via bound propagation Year: (2022)
Ref_id:b32 Title: Formal verification for neural networks with general nonlinearities via branch-and-bound Year: (2023)
Ref_id:b33 Title: Fast and effective robustness certification Year: (2018-12-03)
Ref_id:b34 Title: An abstract domain for certifying neural networks Year: (2019)
Ref_id:b35 Title: Recursive deep models for semantic compositionality over a sentiment treebank Year: (2013-10-21)
Ref_id:b36 Title:  Year: (2021)
Ref_id:b37 Title: Guaranteed conformance of neurosymbolic models to natural constraints Year: (2023)
Ref_id:b38 Title: Axiomatic attribution for deep networks Year: (2017)
Ref_id:b39 Title: Provable editing of deep neural networks using parametric linear relaxation Year: (2024)
Ref_id:b40 Title: Architecture-preserving provable repair of deep neural networks Year: (2023-06)
Ref_id:b41 Title: TRL: Transformer Reinforcement Learning Year: (2025)
Ref_id:b42 Title: Betacrown: Efficient bound propagation with per-neuron split constraints for neural network robustness verification Year: (2021)
Ref_id:b43 Title: Modelverification. jl: a comprehensive toolbox for formally verifying deep neural networks Year: (2025)
Ref_id:b44 Title: A semantic loss function for deep learning with symbolic knowledge Year: (2018-07)
Ref_id:b45 Title: Fast and complete: Enabling complete neural network verification with rapid and massively parallel incomplete verifiers Year: (2021)
Ref_id:b46 Title: Don't pour cereal into coffee: Differentiable temporal logic for temporal action segmentation Year: (2022)
Ref_id:b47 Title: Qwen3 technical report Year: (2025)
Ref_id:b48 Title: Efficient neural network robustness certification with general activation functions Year: (2018-12-03)
Ref_id:b49 Title: Efficient neural network robustness certification with general activation functions Year: (2018)
