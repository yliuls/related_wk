Title: 
Abstract: A common approach to mechanistic interpretability is to causally manipulate model representations via targeted interventions in order to understand what those representations encode. Here we ask whether such interventions create out-ofdistribution (divergent) representations, and whether this raises concerns about how faithful their resulting explanations are to the target model in its natural state. First, we demonstrate theoretically and empirically that common causal intervention techniques often do shift internal representations away from the natural distribution of the target model. Then, we provide a theoretical analysis of two cases of such divergences: 'harmless' divergences that occur in the behavioral null-space of the layer(s) of interest, and 'pernicious' divergences that activate hidden network pathways and cause dormant behavioral changes. Finally, in an effort to mitigate the pernicious cases, we apply and modify the Counterfactual Latent (CL) loss from Grant (2025) allowing representations from causal interventions to remain closer to the natural distribution, reducing the likelihood of harmful divergences while preserving the interpretive power of the interventions. Together, these results highlight a path towards more reliable interpretability methods. 1

Section: INTRODUCTION
A central goal of mechanistic interpretability is to understand what the internal representations of neural networks (NNs) encode and how this gives rise to their behavior. Perhaps the most powerful approach to pursuing this goal is through causal interventions, where methods such as activation patching and Distributed Alignment Search (DAS) directly manipulate internal representations to test how they affect outputs (Geiger et al., 2021;2023;Wu et al., 2023;Wang et al., 2022;Meng et al., 2023;Nanda, 2022;Csordás et al., 2024). Indeed, even correlational methods such as Sparse Autoencoders (SAEs) and Principal Component Analysis (PCA) often use causal interventions as a final judge for whether the features they identify are truly meaningful (Huang et al., 2024;Dai et al., 2024). Causal interventions thus occupy a central place in making functional claims about neural circuitry (Pearl, 2010;Geiger et al., 2024;2025;Lampinen et al., 2025;Braun et al., 2025).
The use of causal interventions rests on a fundamental assumption that counterfactual model states created by interventions are realistic for the target model. Despite its pervasiveness, however, this assumption is often untested. For example, some activation patching experiments multiply feature values by up to 15x (Lindsey et al., 2025); in these settings, it seems possible that intervened representations diverge significantly from the NN's natural distributions. This raises questions about the reliability of causal interventions for mechanistic interpretability. Do divergent representations change what an intervention can say about an NN's natural mechanisms? When, and to what extent, is it okay for such divergences to occur? When it is not okay, how can we prevent them from occurring?
In this work, we provide both empirical and theoretical insight on these issues. We first demonstrate that divergent representations are a common issue for causal interventions -across a wide range of intervention methods, we find that intervened representations often do diverge from the target NN's natural distribution. We then provide theoretical examples of two types of divergence: 'harmless' divergences that can occur from within-decision-boundary covariance along causal dimensions or from deviations in the null-spaces of the NN layers, and 'pernicious' divergences that activate hidden network pathways and can cause dormant changes to behavior. We provide discussion on Figure 1: Causal interventions can recruit hidden circuits that produce misleadingly confirmatory or dormant behavior. (a) Consider natural pathways (dashed arrows) for two classes A and B that carry activity to different behavioral outputs y. In a hypothetical intervention meant to find path A, patching h 1 with a divergent representation can activate distinct, hidden pathways (solid arrows) that result in misleadingly confirmatory behavior (orange) and/or undetected behavior (red). (b) Consider 2D projections of the neural activity of h 1 for a different network that classifies states into one of 10 classes (denoted by hue). Suppose that natural representations (dark points) lie within well-defined decision boundaries (dashed lines) and covary along causal axes, and that intervened representations (light points) are constructed by patching the first axis from a sampled natural representation. Although these representations diverge from the natural distribution, this can be harmless (top) or pernicious (bottom) depending on the network's functional landscape. In particular, it can be pernicious if the network has a functional landscape where intervened activity unknowingly recruits hidden circuits (visualized as an orange region) or crosses dormant behavioral boundaries (red regions), depending on the claims. how harmless and pernicious cases are not always mutually exclusive, where the harm depends on the specific mechanistic claims. Finally, we provide a broad-stroke, initial solution for mitigating pernicious divergences by minimizing all intervened divergences. We show that we can use the Counterfactual Latent (CL) auxiliary loss introduced in Grant (2025) to reduce all representational divergence in the Boundless DAS setting from Wu et al. (2023) while maintaining the same behavioral accuracy; and we introduce a modified version of the CL loss that targets causal subspaces and show that it can improve out-of-distribution (OOD) intervention performance on synthetic tasks. Although we do not propose this method as the final solution to representational divergence, we pose it as a step towards more reliable interventions.
We summarize our contributions as follows:
1. We show theoretical and empirical examples of divergence between natural and causally intervened representations for a variety of causal methods (Section 3).
2. We provide a theoretical treatment of cases in which divergence can arise innocuously from variation in null-spaces, demonstrating that some divergences can even be desired (Section 4.1) 3. We provide synthetic examples of cases where divergent representations can (1) activate hidden computational pathways while still resulting in hypothesis-affirming behavior, and (2) cause dormant behavioral changes, together raising questions about the mechanistic claims that can be made from patching results alone (Section 4.2).
4. Lastly, we use the CL auxiliary loss from Grant (2025) to minimize patching divergence directly in the 7B Large Language Model (LLM) Boundless DAS experimental setting from Wu et al. (2023), and we introduce a stand-alone, modified CL loss that exclusively minimizes divergence along causal dimensions, improving OOD intervention performance in synthetic settings. Together, these results provide an initial step towards mitigating pernicious divergences (Section 5).
this section cite: ['b9', 'b44', 'b41', 'b25', 'b27', 'b3', 'b16', 'b5', 'b28', 'b11', 'b19', 'b1', 'b22', 'b1', 'b44', 'b1', 'b44']

Section: BACKGROUND AND RELATED WORK

this section cite: []

Section: ACTIVATION PATCHING
Activation patching generally refers to a process of "patching" (i.e. substituting) some portion of neural activity at an intermediate layer into or from a corrupted forward pass of a network (Geiger et al., 2020;Vig et al., 2020;Wang et al., 2022;Meng et al., 2023;Zhang & Nanda, 2024). It can be performed at various granularities such as whole layers, attention heads, or individual neurons. Many forms of activation patching can be unified under the assumption that subspaces, rather than individual neurons, are the atomic units of NN representations (Rumelhart et al., 1986;McClelland et al., 1986;Smolensky, 1988;Elhage et al., 2022;Geiger et al., 2021;Grant et al., 2024). Activation patching at the level of individual neurons can be understood as subspace patching along neuronal axes, and many of its high-level granularities can be understood as specific forms of individual neuron patching (Geiger et al., 2020;Vig et al., 2020;Wang et al., 2022;Meng et al., 2023).
this section cite: ['b8', 'b39', 'b41', 'b25', 'b46', 'b29', 'b24', 'b32', 'b6', 'b9', 'b14', 'b8', 'b39', 'b41', 'b25']

Section: DISTRIBUTED ALIGNMENT SEARCH
Distributed Alignment Search (DAS) (Geiger et al., 2021;2023;Wu et al., 2023) can be understood as a form of activation patching that operates in a transformed basis so that specific, causally relevant subspaces can be manipulated analogously to high-level variables from causal abstractions (CAs) (e.g. symbolic programs). Many cases of individual neuron (coordinate) patching can be understood as specific cases of DAS that use the identity transform. We use DAS in Sections 4 and 5, so here we introduce its theory and background.
DAS finds alignments between neural representational subspaces and causal variables from Causal Abstractions (CAs) by testing the hypothesis that an NN's latent state vector h ∈ R dm can be transformed into a vector z ∈ R dm that consists of orthogonal subspaces encoding interpretable variables. This transformation is performed by a learnable, invertible Alignment Function (AF), z = A(h). We restrict our considerations to linear AFs of the form A(h) = W h where W ∈ R dm×dm is invertible. This transformation allows us to formulate h in terms of interpretable variables and to manipulate encoded values.
For a given CA with variables var i ∈ {var 1 , var 2 , ..., var n }, DAS tests the hypothesis that z is composed of subspaces ⃗ z vari ∈ R dvar i corresponding to each of the variables from the CA. A causally irrelevant subspace ⃗ z extra ∈ R dextra is also included to encode extraneous, functionally irrelevant neural activity (i.e., the behavioral null-space).
A(h) = z =    ⃗ z var1 • • • ⃗ z varn ⃗ z extra   (1)
Here, each ⃗ z vari ∈ R dvar i is a column vector of potentially different lengths, where d vari is the subspace size of var i , and all subspace sizes satisfy d extra
+ n i=1 d vari = d m .
The value of a single causal variable encoded in h can be manipulated through an interchange intervention defined as follows:
ĥ = A -1 ((I -D vari )A(h trg ) + D vari A(h src ))(2)
Here, D var ∈ R dm×dm is a manually defined, block diagonal, binary matrix that defines the subspace size d vari , and I ∈ R dm×dm is the identity matrix. Each D vari has a set of d vari contiguous ones along its diagonal to isolate the dimensions that make up ⃗ z vari . h src is the source vector from which the subspace activity is harvested, h trg is the target vector into which the harvested activity is patched, and ĥ is the resulting intervened vector that replaces h trg in the model's processing. This allows the model to make predictions using a new value of variable var i .
To train A, DAS uses counterfactual behavior c ∼ D as training labels, where c is generated from the CA. c, for a given state of a CA and its context, is the behavior that would have occurred had a causal variable taken a different value and everything else remained the same. c is generated by freezing the state of the environment, changing one or more variable values in the CA, and using the CA to generate new behavior in the same environment using the new values. We train A on intervention samples while keeping the model parameters frozen, minimizing the following objective (for non-sequence-based settings):
L DAS (A) = - 1 N N k=1 log p A c (k) x (k) , ĥ(k) ,(3)
where N is the number of samples in the dataset, c (k) is the counterfactual label in sample k, x (k)  is the model input data, and p A (• | •) is the model's conditional probability distribution given the intervened latent vector, ĥ. We minimize L DAS (A) using gradient descent, backpropagating into A with all model weights frozen. A is evaluated on new intervention data, where the model's accuracy on c following each intervention is referred to as the Interchange Intervention Accuracy (IIA).
this section cite: ['b9', 'b44']

Section: PROBLEMATIC CAUSAL INTERVENTIONS
Prior work has implicitly explored issues related to representational divergence from causal interventions. For example, methods such as causal scrubbing or noising/denoising activation patching (Wang et al., 2022;LawrenceC et al., 2022;Meng et al., 2023;Chen et al., 2025;Zhang & Nanda, 2024) intentionally introduce divergent representations to test the sufficiency, completeness, and faithfulness of proposed circuits. Works such as Wattenberg & Viégas (2024), Méloux et al. (2024), and Chen et al. (2025) also implicitly explore the dangers of divergent intervened representations by showing how circuits and features can be redundant or have combinatorial effects that are difficult to enumerate given current methodologies, while Zhang & Nanda (2024) and Heimersheim & Nanda (2024) point out easy misinterpretations of patching results. Shi et al. (2024) and Wang et al. (2022) provide criteria centered on faithfulness, completeness, and minimality for evaluating circuits through causal interventions. A body of work on counterfactual explanations exists, some of which has explored differences between on-manifold and off-manifold adversarial attacks (Stutz et al., 2019), and some works have explored constraining counterfactual features to the manifold of the dataset (Verma et al., 2024;Tsiourvas et al., 2024). Our proposed method in Section 5 differs in that it trains a principled alignment to generate counterfactual representations and it constrains deviations along causal dimensions. For DAS in particular, Makelov et al. (2023) demonstrate that it is possible to produce an interaction between the null-space and dormant subspaces that affect behavior. Because they define dormant subspaces as those that do not vary across different model inputs, variation along these directions is, by definition, a form of divergent representation. Finally,
Sutter et al. (2025) posit that it is possible to align any causal abstraction to NNs under a number of assumptions including a sufficiently powered, non-linear alignment function, raising questions about what non-linear causal interventions really tell us.
this section cite: ['b41', 'b25', 'b2', 'b46', 'b34', 'b38', 'b37', 'b23']

Section: ARE DIVERGENT REPRESENTATIONS A COMMON PHENOMENON?
We begin by demonstrating that divergent representations are a common (if not likely) outcome of causal interventions, both in theory and in practice. We do not yet consider its perniciousness, however, and we reserve the question of whether and when divergence is harmful for Section 4.
this section cite: []

Section: FOR MOST MANIFOLDS, COORDINATE PATCHING GUARANTEES DIVERGENCE
We first consider a theoretical setting where coordinate-based patching of one or more vector dimensions is performed on a single manifold, similar to what might be done in neuron level activation patching (Vig et al., 2020;Geiger et al., 2021). We prove that in this setting, divergent representations are guaranteed to occur if patching is performed exhaustively. For simplicity, we consider a minimal version of this proof that involves a circular manifold with two dimensions. A more general proof, which applies to most manifold geometries, can be found in Appendix A.2.
Formally, let
M K = { c K + u : ∥u∥ 2 ≤ r K } ⊂ R 2 be a class-K manifold with centroid c K ∈ R 2
and radius r K > 0. Given two native representations h src = c K + u and h trg = c K + v, let us define a coordinate patch (onto class K) that keeps the first coordinate of h src and the second of h trg : Proposition (coordinate patching exceeds the class radius).
ĥ = h src 1 h trg 2 = c K,1 + u 1 c K,2 + v 2 . Then the deviation from c K is ĥ -c K = (u 1 , v 2 ) ⊤ , ∥ ĥ -c K ∥ 2 2 = u 2 1 + v 2 2 . Natural Representation Intervened Representation EMD
this section cite: ['b39', 'b9']

Section: Mean
If h src , h trg ∈ M K (i.e., ∥u∥ 2 ≤ r K and ∥v∥ 2 ≤ r K ), then the patched point ĥ is off-manifold whenever u 2 1 +v 2 2 > r 2 K . In particular, there exist boundary points h src , h trg ∈ ∂M K with u = (r K , 0) and v = (0, r K ) such that ∥ ĥ -c K ∥ 2 = r 2 K + r 2 K = r K √ 2 > r K . Proof. Since ĥ -c K = (u 1 , v 2 ), we have ∥ ĥ -c K ∥ 2 2 = u 2 1 + v 2 2 . Choosing u = (r K , 0) and v = (0, r K )
gives the stated violation.
As noted above, this intuition holds for all manifold shapes other than axis-aligned hyper-rectangles (see Appendix A.2). Thus, in these relatively simple theoretical intervention settings, divergent representations are guaranteed to occur with enough intervention samples.
this section cite: []

Section: MANY EXISTING CAUSAL METHODS EMPIRICALLY PRODUCE DIVERGENCE
Stepping beyond the theoretical setting, we also empirically demonstrate that common, real-world causal interpretability methods often produce divergent representations. The notion that these methods can create divergence has been indirectly explored in previous work through attention patterns in Gaussian noise corrupted activation patching (Zhang & Nanda, 2024). Here we extend this view to three popular causal intervention methods: mean difference vector patching (e.g. Feng & Steinhardt  (2024 Figure 2 shows the top two principal components of the natural and intervened representations for each intervention method, distinguished by color. These results demonstrate that divergence is a common phenomenon in practice and is not specific to any one method. Even simple methods that patch along a single mean direction are subject to divergence, despite high behavioral accuracy (Feng & Steinhardt, 2024;Geiger et al., 2023;Wu et al., 2023). We quantify this divergence in Figure 2(c) using the Earth Mover's Distance (EMD) (Villani, 2009) between the full dimensional natural and intervened distributions, using the corresponding natural-natural comparison as a baseline. We see that the intervened divergence exceeds that of the natural (more metrics and details are in Appendix A.1). Note that this result does not necessarily imply that the respective methods are invalid or that their claims are incorrect; the panels are only meant to show the presence of divergence.
this section cite: ['b46', 'b7', 'b10', 'b44', 'b40']

Section: WHEN ARE DIVERGENT REPRESENTATIONS HARMLESS OR PERNICIOUS?
Having demonstrated that divergent representations are a common phenomenon, we now consider whether divergence is a concerning phenomenon. We propose that divergence is harmless to many functional claims if it exists in the behavioral null-space, and that it can be pernicious if it recruits hidden pathways or causes dormant behavioral changes. However, we stress that the harm is inherently claim dependent, meaning that these forms of divergence are not mutually exclusive.
this section cite: []

Section: HARMLESS CASES
Here we explore a set of cases for which we might consider divergent representations to be harmless to functional claims. First among these are cases in which divergence is bottle-necked into the null-spaces of the next interacting weight matrices. Formally, we define the null-space of a weight matrix
W ∈ R d ′ ×d as N (W ) = {v ∈ R d | W v = 0}.
Neural activity in the null-space of the weights refers to any changes δ ∈ R d for which W (h + δ) = W h. We propose that divergence v ∈ N (W ) is harmless to the computation of W because it is equivalent to adding the zero vector
W (h + v) = W h + W v = W h + 0 = W (h + 0).
Notably, however, this harmlessness does not apply to the sub-computations of the matrix multiplication because v ∈ N (W ) does not imply that W i,j (h j + v j ) = W i,j (h j + 0) for vector row j and matrix row i. Thus, v in this case is potentially harmful to mechanistic claims about individual activation-weight sub-computations, while being harmless to the overall matrix multiplication.
We can generalize this notion beyond matrices to an arbitrary function ψ. Let ψ : R d → R d ′ and let X ⊆ R d . We define the behavioral null-space with respect to X as
N (ψ, X) = { v ∈ R d | ∀x ∈ X, ψ(x + v) = ψ(x) }.(4)
A common case of ψ in practice for a layer ℓ of an NN f consists of all subsequent computations after and including layer ℓ, denoted f ≥ℓ (h). We propose that behaviorally null divergence v is harmless to the overall computation of f ≥ℓ because it is equivalent to adding 0 to the input. However, v can be harmful to claims about a sublayer ℓ + k within f ≥ℓ because f ℓ+k (f ≥ℓ,<ℓ+k (h +v)) is not guaranteed to be equal to f ℓ+k (f ≥ℓ,<ℓ+k (h + 0)) (Sec. 4.2.1). See Appx. A.3 and Algorithm 1 to practically classify harmlessness when N (ψ, X) characterizes the full space of harmless divergence.
Idealized Case Study. We now present an example of harmless divergence in the behavioral nullspace by introducing a behaviorally binary subspace-a vector subspace which causally impacts the outputs of a future processing layer (e.g., classification labels) only through its sign. Formally, let f : R d → R d ′ denote a computational unit (possibly consisting of multiple NN layers and functions). Let sign(•) denote the elementwise sign map sign : R dvar → {-1, 1} dvar , and assume a fixed alignment function A and subspace selection matrix D var ∈ R d×d (Sec. 2.2). A linear subspace Z ⊆ R d is behaviorally binary (with respect to f , D var , and A) iff for all
D var A(h), D var A(h ′ ) ∈ Z, sign(D var A(h)) = sign(D var A(h ′ )) =⇒ f (h) = f (h ′ )(5)
Now, suppose we have an NN with two causal subspaces, zvara ⊆ R dvar a and zvar b ⊆ R dvar b , with values ⃗ z
vara and ⃗ z
var b , for a model input x i , where the bold, tilde notation distinguishes variables from their values. Further assume that zvar b is behaviorally binary and co-varies with, zvara . Using h (xi) and z (xi) from Equation 1 under a given input x i , we use the following definition:
A(h (xi) ) = z (xi) = z (x i ) vara = ⃗ z (xi) vara z (x i ) var b = ⃗ z (xi) var b(6)
Due to the assumption of covariance in zvara and zvar b , it is reasonable to assume that the values ⃗ z ). Now, an interchange intervention on zvar b using a source vector from x low and target vector from x high , will produce:
ẑ = zvara = ⃗ z (xhigh) vara zvar b = ⃗ z (xlow) var b (7)
Because we assumed that the value of ⃗ z in Equation 7will have never existed together in the native distribution, but the behavior will remain unchanged because zvar b is behaviorally binary and its sign has not changed. This divergence is thus harmless to the claim of discovering f 's causal axes.
Summary. We propose that divergences within behavioral null-spaces are harmless to many functional claims about a function ψ's computations when the claim encapsulates (i.e. ignores) the internal, sub-computations of ψ. However, we are not suggesting that the behavioral null-space encompasses the set of all harmless divergences. In general, such an exhaustive set cannot exist without assuming the superiority of some scientific claims/assumptions over others. For example, take the set of all harmless divergences for a specific claim, then modify the claim to assume it permissible to diverge in a manner previously excluded from the harmless set. The modification to the claim also modifies the harmless divergences. Lastly, we note that behaviorally null divergence is not always harmless. Indeed, one could even desire to intervene on the behavioral null-space to causally test that it is null. Thus, we stress that the mechanistic claim for which an intervention is meant to support is important for determining the harmlessness of the divergence.
this section cite: []

Section: PERNICIOUS DIVERGENCE VIA OFF-MANIFOLD ACTIVATION
We now explore pernicious cases of representational divergence involving the concept of hidden pathways, which refer to any unit, vector direction, or subcircuit that is inactive on the natural support of representations for a given context but becomes active and influences behavior under an intervention. Although hidden pathways can be compatible with harmless divergences (Sec. 4.1), they can also undermine claims about natural mechanisms and can prime dormant behavioral changes discussed in Sec. 4.2.2.
Formally, let D denote the data distribution over latent representations h ℓ ∈ R d at layer ℓ, and let S = supp(h ℓ ∼ D) denote its support with S K = supp(h ℓ K ∼ D) the support for class K. Denoting the intended class K following an intervention as subscripted → K, an intervened representation ĥℓ →K is said to be divergent if ĥℓ →K / ∈ S K (e.g. it exists off the natural manifold of class K). We define the convex hull of class-K representations as conv(S
K ) = { i α i h ℓ i,K : α i ≥ 0, i α i = 1}
where the subscript K denotes that h ℓ i,K was taken from class K inputs. Projecting an intervention onto conv(S K ) ensures that it remains within the convex interpolation region of class-K.
this section cite: []

Section: MEAN-DIFFERENCE PATCHING CAN ACTIVATE HIDDEN PATHWAYS
Patching with a mean-difference vector can flip a decision by activating a unit that is silent for all natural class inputs.
Setup. Consider a two-layer circuit with a ReLU nonlinearity. Let h ℓ ∈ R 4 feed into
s = 1 ⊤ h ℓ+1 = 1 ⊤ ReLU(W ℓ h ℓ + b ℓ ), W ℓ ∈ R 3×4 , b ℓ ∈ R 3 , where W ℓ = 0.75 0.25 0 0.5 0 1 0 0 1 1 -1 -1 , b ℓ = -0.5 -0.5 0 .
A positive score (s > 0) indicates class A. Suppose class-A and class-B representations at layer ℓ are
h ℓ A = [1, 0, 1, 0] ⊤ (case 1) [0, 1, 1, 0] ⊤ (case 2) , h ℓ B = [0, 0, 1, 0] ⊤ (case 1) [0, 0, 1, 1] ⊤ (case 2) Evaluating h ℓ A yields h ℓ+1 Acase 1 = [0.25, 0, 0] ⊤ , h ℓ+1
Acase 2 = [0, 0.5, 0] ⊤ , so for class-A, s A ∈ {0.25, 0.5}. For class-B, all outputs are zero, s B = 0.
Mean-difference patching. We construct a mean difference vector between the classes,
δ B→A = µ A -µ B = 1 2 2 i=1 (h ℓ Acase i ) -1 2 2 i=1 (h ℓ Bcase i
) = [0.5, 0.5, 0, -0.5] ⊤ .
Applying this to class-B representations gives
ĥℓ B→A = h ℓ B + δ B→A =
[0.5, 0.5, 1, -0.5] ⊤ (case 1) [0.5, 0.5, 1, 0.5] ⊤ (case 2) After propagation through the circuit:
ĥℓ+1 B→A =
[0, 0, 0.5] ⊤ , ŝcase1 = 0.5 [0.25, 0, 0] ⊤ , ŝcase2 = 0.25
The intervention flips the decision to class-A (ŝ > 0). However, the third hidden unit becomes active only for ĥℓ B→A , never for natural h ℓ A . This new activation is a hidden pathway that was silent under all native samples. Thus the mean-difference patch crosses the decision boundary only by activating an off-manifold circuit.
this section cite: []

Section: If we project ĥℓ
B→A onto conv(S A ) (or equivalently onto a local PCA subspace of S A ), this ReLU state change disappears, and the decision boundary is no longer crossed-confirming that the original effect was driven by divergence rather than a within-manifold causal mechanism. It is unclear what this patching experiment reveals about the natural mechanisms of the model.
this section cite: []

Section: DORMANT BEHAVIORAL CHANGES
Divergent representations can also yield dormant behavioral changes: perturbations that appear behaviorally null in one subset of contexts but alter behavior in others. Formally, let ψ : R d ×C → R d ′ and let C 1 ⊂ C be a subset of contexts. The space of dormant behavioral changes relative to X, C 1 , C is V(ψ, X, C 1 , C) = N (ψ, X, C 1 ) \ N (ψ, X, C).
Illustration. Extend the network from the previous Sec. 4.2.1 by adding one row of zeros to W ℓ and b ℓ , producing h ℓ+1 ∈ R 4 where the final coordinate is always zero. Add a context vector v ∈ R 4 and a final affine layer:
ŷ = W ℓ+1 (h ℓ+1 + v) + b ℓ+1 , W ℓ+1 = 1 1 0.5 0 0 0 0 0 0 0 1 1 , b ℓ+1 = 0 0.25 -1 .
Here, the first argmax index of ŷ = [ŷ 1 , ŷ2 , ŷ3 ] ⊤ corresponds to class predictions A, B, and C.
Assume v = [0, 0, 0, v 4 ] ⊤ . For ĥℓ+1
Bcase 1 →A = [0, 0, 0.5, 0] ⊤ , ŷ1 = 0.25, ŷ2 = 0.25 ŷ3 = (0.5 + v 4 ) -1 = v 4 -0.5. The model predicts class-A when v 4 < 0.75 but switches to class-C when v 4 > 0.75. Notably, v 4 would not naturally cause a class-C prediction below a value of 1 due to the bias threshold. Thus the same intervention that was benign in one context (v 4 < 0.75) produces a behavioral flip in another (1 > v 4 > 0.75) purely due to the latent divergence priming a new pathway.
Implications. Dormant behavioral changes highlight that behaviorally "safe" interventions can still introduce hidden context dependencies. Detecting them would require evaluating across all possible contexts, which is infeasible in practice. Therefore, causal intervention based experiments should ideally (1) report any introduced representational divergence outside of the null-space and (2) test causal interventions for context-sensitivity.
this section cite: []

Section: Summary.
Hidden pathways arise when off-manifold representations activate computations that never occur for natural representations. Such pathways can potentially alter causal conclusions even when behavior appears unchanged. Manifold-preserving projections and ReLU-pattern audits are potentially practical safeguards against these pernicious forms of divergence.
this section cite: []

Section: HOW MIGHT WE AVOID DIVERGENT REPRESENTATIONS?
We have thus far shown that divergent representations are common and that their harm depends on multiple factors. We now consider the question of how such divergences might be avoided. Some existing methods solve this by projecting counterfactual features (in our case, intervened representations) directly to the natural manifold (Verma et al., 2024). However, we seek a method that generates principled interventions that are constrained to be innocuous. In this pursuit, we first apply the Counterfactual Latent (CL) loss from Grant (2025) to the DAS experiments from Wu et al. (2023) and find that we can reduce intervened divergence while preserving accuracy on a Llama based LLM. We then introduce a modified CL loss that only targets causal dimensions, and we show that it can improve OOD intervention accuracy. We emphasize, however, that minimizing divergence magnitude does not guarantee elimination of hidden pathways; it only reduces the risk surface.
this section cite: ['b38', 'b1', 'b44']

Section: APPLYING THE COUNTERFACTUAL LATENT LOSS TO BOUNDLESS DAS
To encourage intervened NN representations to be more similar to the native distribution, we first apply the CL auxiliary loss from Grant (2025) to the Boundless DAS setting in Wu et al. (2023). This auxiliary training objective relies on counterfactual latent (CL) vectors as vector objectives. CL vectors are defined as vectors that encode the same causal variable value(s) that would exist in the intervened vector, ĥ, assuming the interchange intervention was successful. We can obtain CL vectors from sets of natural h vectors from situations and behaviors that are consistent with the values of the CA to which we are aligning. See Figure 3 for a visualization. As an example, assume that we have a CA with causal variables var u , var w , and var extra , and following a causal intervention we expect ĥ to have a value of u for variable var u and w for variable var w . A CL vector h CL for this example can be obtained by averaging over a set of m natural representations,
h CL = 1 m m i=1 h (xi) CL , where each h (xi)
CL has the same variable values: var u = u and var w = w (as labeled by the CA). The CL auxiliary loss L CL introduced in Grant (2025) is composed of the mean of an L2 and a cosine distance objective using CL vectors as labels. Using notation defined in Section 2.2, L CL for a single training sample is defined as follows:
L CL ( ĥ, h CL ) = 1 2 ∥ ĥ -h CL ∥ 2 2 - 1 2 ĥ • h CL ∥ ĥ∥ 2 ∥h CL ∥ 2 (8
)
L CL is combined with the DAS behavioral loss L DAS into a single loss term using ϵ as a tunable hyperparameter: L total = ϵL CL + L DAS . The loss is computed as the mean over batches of samples and optimized using gradient descent (Appendix A.5).
this section cite: ['b1', 'b44', 'b1']

Section: Results:
We applied the CL loss to the Boundless DAS notebook from Wu et al. (2024) which reproduces the main result from Wu et al. (2023) (see Appendix A.4). Figure 3A provides qualitative visualizations of the decreased divergence when applying the CL loss, and Figure 3B shows both IIA and EMD as a function of increasing the CL weight ϵ. For small values of ϵ, the IIA is maintained (and potentially even improved) while EMD decreases, demonstrating that the CL auxiliary loss can directly reduce divergence in practical interpretability settings without sacrificing the interpretability method.
this section cite: ['b45', 'b44']

Section: MODIFIED CL LOSS IMPROVES OOD INTERVENTIONS IN SYNTHETIC MODEL SETTINGS
We next modify the CL loss to work independently of the DAS behavioral loss and show that it improves OOD intervention performance. We simulate an intermediate layer of an NN by constructing a synthetic dataset of h vectors with known feature dimensions and labels y that we use to train a Multi-Layer Perceptron (MLP). The dataset consists of noisy samples around a set of grid points from two feature dimensions with correlational structure. Specifically, we define a set of features as the Cartesian product of two values along the x 1 -axis {-1, 1} and five values along the x 2 -axis {0, 1, 2, 3, 4}, resulting in ten unique coordinates that each correspond to one of ten classes. We add noise and covariance to these feature dimensions and concatenate n extra noise dimensions, resulting in simulated vectors h ∈ R 2+n where n = 16 unless otherwise stated. The feature dimensions of these vectors are shown as the natural distributions in Figures 1(b) and 3. We then train a small MLP on these representations to predict the class labels using a standard cross entropy loss. After training, we perform DAS analyses with either the behavioral loss or the CL loss independently. See Appendix A.5 for further details on dataset construction, MLP training, and DAS training.
We modify the CL loss by applying it to individual causal subspaces only (as discovered through the DAS training). This allows us to construct h vari CL vectors specific to a single causal variable var i . The modified L ′ CL for a single training sample is defined as follows:
ĥvari = A -1 (D vari A( ĥ)), h vari CL = stopgrad(A -1 (D vari A(h CL ))) (9
)
L ′ CL = n i=1 L vari CL = n i=1 1 2 ∥ ĥvari -h vari CL ∥ 2 2 - 1 2 ĥvari • h vari CL ∥ ĥvari ∥ 2 ∥h vari CL ∥ 2 (10
)
Results: Figures 3D and 3E provide a qualitative comparison of intervened and native representations for interventions using a trained DAS rotation matrix. Each dot in the figures shows the values of the feature dimensions for a single representation. The native states are displayed in darker colors and the intervened in lighter. Each hue indicates the ground truth class of the state. We can see a tightening of the intervened representations using the CL loss. Quantitatively, the DAS loss produces EMD values along the feature dimensions of 0.032 ± 0.003 whereas the CL loss produces 0.007 ± 0.001 with IIAs of 0.997 ± 0.001 and 0.9988 ± 0.0005 respectively on training/test sets with held-out classes.
What is the practical utility of reducing divergence? We hypothesized that divergence could influence IIA when transferring the DAS alignment to OOD settings. To test this, we partitioned the synthetic task into a dense and a sparse cluster of classes based on their relative spacing (Appx. A.5.4). We then trained an MLP and alignment on each partition and evaluated the alignment on the held-out partition. The CL loss performed better than the behavioral loss in these OOD settings (Figure 3F). We then regressed OOD IIA on training EMD to find an anti-correlation (coef. -.34, R 2 = .73, F(1,28)=75.28, p < .001), showing that divergence can predict lower OOD performance (Appx. A.6).
this section cite: []

Section: DISCUSSION AND LIMITATIONS
In this work we demonstrated that a variety of common causal interventions can produce representations that diverge from a target model's natural distribution of latent representations. We then showed that although this can have benign effects for some causal claims, it can also activate hidden pathways and trigger dormant behaviors that can perniciously affect other claims. As a step towards mitigating this issue, we provided a broad-stroke solution by directly minimizing the divergence of intervened activity along causal dimensions, mitigating both pernicious and harmless forms of divergence.
A remaining gap in our work is the failure to produce a principled method for classifying harmful divergence for any claim. Additionally, the modified CL loss is confined to a narrow set of simplistic settings and is not specific to pernicious divergence. We look forward to exploring ways to classify and mitigate pernicious divergence through self-supervised means in future work.
Where does this leave us with respect to causal interventions in mechanistic interpretability? Given our theoretical findings, any divergence outside of the null-space of NN layers is potentially pernicious. This poses challenges for aspirations of a complete mechanistic understanding of NNs using existing methods alone. However, we note that many practical mechanistic projects can be satisfied by collecting sufficiently large intervention evaluation datasets, and continued development of methods such as the CL loss can reduce the problem even further. We are optimistic for the future of this field.
Baseline divergence. To ensure that we did not introduce bias from the sampling procedure in the baseline comparison, we use the corresponding ground truth vectors for the intervened vectors when comparing to the baseline divergence. Formally, let H ′ be the set of ground truth natural vectors corresponding to the set Ĥ:
Divergence Baseline = EMD(H, H ′ ).(14)
Local PCA Distance: For each reference point x i , we identified its k nearest neighbors N k (x i ) in Euclidean space and computed the local tangent subspace via PCA. Let U d (x i ) ∈ R D×d denote the top d principal components explaining at least 95% of the local variance. The projection matrix onto this tangent space is P i = U d U ⊤ d . For a query point v, the Local PCA Distance is the orthogonal residual between v and its projection onto the local tangent space at the nearest reference sample:
D PCA (v) = min xi ∥(I -P i ) [ v -x i ]∥ 2 .
Small values indicate that v lies close to the locally linear approximation of the manifold, whereas large residuals reflect departures orthogonal to the manifold surface.
Local Linear Reconstruction Error: We computed an error inspired by Locally Linear Embedding (LLE). Given the same neighborhood N k (v) of k reference points X k = {x 1 , . . . , x k }, we found reconstruction weights w = (w 1 , . . . , w k ) that best express v as a convex combination of its neighbors:
min w v - k j=1 w j x j 2 2 s.t. j w j = 1.
A small Tikhonov regularizer λI was added to the local covariance for numerical stability. The Local Linear Reconstruction Error is the residual norm at the optimum:
D LLR (v) = ∥v -X k w * ∥ 2 .
This metric measures how well v can be expressed as a locally linear interpolation of nearby manifold points; poor reconstruction (large D LLR ) implies off-manifold position or local curvature mismatch.
this section cite: []

Section: Kernel Density (KDE) Density Score:
We estimated a nonparametric probability density function over the reference set using Gaussian kernel density estimation:
p(x) = 1 nh D n i=1 exp - ∥x -x i ∥ 2 2 2h 2 ,
where h is the kernel bandwidth determined by Silverman's rule of thumb or cross-validation. The KDE Density Score for a query point v is its log-density under this model,
S KDE (v) = log p(v),
which inversely reflects off-manifold distance: lower log-density corresponds to less typical or outof-distribution points. To express results on a comparable scale, we report the negative log-density (i.e., -S KDE ), so that larger values consistently indicate greater deviation from the manifold.
this section cite: []

Section: A.1.3 VISUALIZATION
We visualized both original and intervened representations by projecting onto the first two principal components of the covariance matrix of H ′ and Ĥ combined:
PCs = eigenvectors Cov( H ′ Ĥ ) .(15)
The top two principal components were used to plot representations in two dimensions, with colors distinguishing intervention method and condition (Figure 2).
this section cite: []

Section: A.2 CLOSURE UNDER COORDINATE PATCHING CHARACTERIZES CARTESIAN PRODUCTS AND HYPERRECTANGLES
In this section, we will show that axis-aligned hyperrectangles are the only convex manifold shape that does not have source-target vector pairs that produce off-manifold intervened representations.
For vector dimensions S ⊆ [d] = {1, . . . , d} and h src , h trg ∈ R d , define a coordinate patch
Patch S (h src , h trg ) i = h src i , i ∈ S, h trg i , i / ∈ S.
where i refers to the i th vector coordinate. A set M ⊆ R d is patch-closed if Patch S (h src , h trg ) ∈ M for all h src , h trg ∈ M and all S ⊆ [d].
Let π i : R d → R be the i th coordinate projection and write I i := π i (M) = {h i : h ∈ M}.
Theorem A.2 (Patch-closure ⇐⇒ product of projections). Let M ⊆ R d be nonempty and let
I i := π i (M). Then M is patch-closed ⇐⇒ M = I 1 × • • • × I d .
Proof. (⇐) Immediate: if M = i I i , then patching replaces coordinates by elements of the corresponding I i , so the result stays in M.
(⇒) Suppose M is patch-closed. The inclusion M ⊆ i I i is tautological. For the reverse inclusion, fix t = (t 1 , . . . , t d ) with t i ∈ I i . For each i pick h (i) ∈ M with h (i) i = t i . Define ĥ(1) := h (1) and for k ≥ 2 set ĥ(k) := Patch {k} ( ĥ(k-1) , h (k) ). Patch-closure gives ĥ(k) ∈ M, and by construction ĥ(k) j = t j for all j ≤ k. Hence ĥ(d) = t ∈ M, yielding i I i ⊆ M.
this section cite: []

Section: Corollary (Convex case ⇒ hyperrectangle).
If M is also convex, then each I i = π i (M) ⊂ R is convex, hence an interval. Therefore M = i I i is an axis-aligned hyperrectangle (Cartesian product of intervals). Conversely, any axis-aligned hyperrectangle is patch-closed (and convex).
Implication. Consequently, any nonempty convex set in R d that is not an axis-aligned hyperrectangle (e.g., a ball, ellipsoid, or a polytope with non-axis-aligned faces) fails to be patch-closed: there exist h src , h trg ∈ M and S ⊆ [d] such that Patch S (h src , h trg ) / ∈ M.
this section cite: []

Section: A.2.1 ACTIVATION PATCHING IN BALANCED SUBSPACES
Here we include an additional example of pernicious activation patching that assumes the existence of balanced subspaces, defined as one or more behaviorally relevant subspaces that are canceled out by opposing weight values. Before continuing, we note that such subspaces are unlikely to exist in practical models due to the fact that they would only arise in cases where two rows of a weight matrix W ∈ R n×m are non-zero, scalar multiples of one another, assuming h = W x. This example, however, could arise in cases where the input x is low rank and a subset of the columns of two rows in W are scalar multiples of one another.
Consider the case where there exists an NN layer that classifies inputs based on the mean intensity of dimensions 3 and 4 for a latent vector h ∈ R 4 , where the NN layers that produce h are denoted f (x) with data inputs x sampled from the dataset, x ∼ D, and where the layer of interest has a synthetically constructed weight vector w ∈ R 4 where
w ⊤ = [w 1 w 2 w 3 w 4 ] = 1 -1 1 2 1 2 .
The layer is thus defined as follows:
y = w ⊤ f (x (i) ) = w ⊤ h (i) = 1h (i) 1 -1h (i) 2 + 1 2 h (i) 3 + 1 2 h (i) 4(16)
Here, i denotes the index of the data within the dataset. Further assume that some behavioral decision depends on the sign of y, that h 1 and h 2 together form balanced subspaces given w (meaning that for all
x (i) ∼ D, w 1 h (i) 1 = -w 2 h (i)
2 ), and that they are non-dormant, meaning that for some pairs (x (i) , x (j) ) where i ̸ = j, then h
(i) 1 ̸ = h (j)
1 . Under these assumptions, the subspace spanned by [1 0 0 0] ⊤ and [0 1 0 0] ⊤ is not causally affecting the network's output under the natural distribution of h. However, if we intervene on h 1 or h 2 while leaving h 3 and h 4 unchanged, the intervened representation ĥ will diverge and potentially cross the decision boundary.
Concretely, if we set h (i) = [1 1 1 1] ⊤ and h (j) = [3 3 -1 -1] ⊤ and then perform an intervention on h 2 using h (i) as the target and h (j) as the source, we get: ĥ = [1 3 1 1] ⊤ . This will result in a negative value of y, thus crossing its decision boundary using a non-native mechanism. This intervention could be used as experimentally affirming evidence for a mechanistic claim, when in reality we have not addressed the model's original mechanism.
this section cite: []

Section: A.3 PRACTICAL ALGORITHM FOR HARMLESS DIVERGENCE WHEN BEHAVIORAL NULL-SPACE CHARACTERIZES THE FULL HARMLESS SET
In settings where perturbations v ∈ N (ψ, X) are treated as harmless and perturbations v / ∈ N (ψ, X) as harmful, the behavioral null-space formalism suggests a practical procedure for testing the harmlessness of a given divergence v. Let X K ⊂ R d be the set of natural representations for class K, and let xK ∈ R d be an intervened representation for class K. To approximate the natural manifold M K locally around xK , we first select the n nearest neighbors of xK in X K :
N n (x K ) = {x (1) , . . . , x (n) } ⊂ X K .
Let U ∈ R n×d be the matrix whose rows are the neighbors u ⊤ i = (x (i) ) ⊤ , and let
µ K = 1 n n i=1 x (i) ∈ R d
denote their mean. Define the centered data matrix
Ũ =    (x (1) -µ K ) ⊤ . . . (x (n) -µ K ) ⊤    ∈ R n×d .
We compute a rank-r PCA of Ũ , obtaining the top r principal directions Q r ∈ R d×r (columns are orthonormal). The corresponding local projection operator is
Π K (x) = µ K + Q r Q ⊤ r (x -µ K ).
The local projection of the intervened representation is then
xproj = Π K (x K ),
and we define the divergence vector as v = xK -xproj .
We now provide Algorithm 1 as a practical method for classifying divergence as harmless or pernicious. We note, however, this algorithm only approximates harmlessness and is not guaranteed to be successful. This is especially the case for situations prone to dormant behavioral changes (Sec. 4.2.2).
this section cite: []

Section: A.4 CL LOSS APPLIED TO BOUNDLESS DAS
To perform the Boundless DAS experiments, we used the Boundless DAS tutorial provided in the pyvene python package (Wu et al., 2024) which reproduces a main result from Wu et al. (2023), and we included the CL loss as a weighted auxiliary objective as described in Section 5.1. The exact task used in this tutorial is one involving continuous valued features, which resulted in few occurrences of valid CL vectors in the provided dataset. In order to obtain exact CL vectors, we generated a token sequence sample that contained a valid CL vector for each intervention sample in the dataset. We left hyperparameter choices the same across all trainings except for the CL loss weight ϵ.
this section cite: ['b45', 'b44']

Section: A.5 CL LOSS IN SYNTHETIC SETTINGS
Here we continue Section 5 with experimental details and additional experiments and results.
Algorithm 1 Classifying the harmlessness of a divergence vector when the behavioral null-space characterizes harmlessness Require: Intervened representation xK ∈ R d for class K; natural class representations X K ⊂ R d ; evaluation set X eval ⊂ R d ; function ψ : R d → R d ′ ; neighborhood size n; local dimension r; tolerance ϵ ≥ 0. Ensure: Classification of the divergence vector v as harmless or harmful.
1: (Local manifold estimation for class K)
Let N n (x K ) = {x (1) , . . . , x (n) } ⊂ X K be the n nearest neighbors of xK in X K . 2: Compute the mean µ K = 1 n n i=1
x (i) .
3: Form the centered matrix Ũ ∈ R n×d with rows (x (i) -µ K ) ⊤ . 4: Perform rank-r PCA on Ũ to obtain the top r principal directions Q r ∈ R d×r . 5: Define the local projection
Π K (x) = µ K + Q r Q ⊤ r (x -µ K ) and set xproj ← Π K (x K ). 6: Define the divergence vector v ← xK -xproj . 7: (Behavioral test over a broader context) For each x ∈ X eval , compute ∆(ψ, x) = ∥ψ(x + v) -ψ(x)∥. 8: if max x∈X eval ∆(x) ≤ ϵ then 9: return v is HARMLESS. 10: else 11: return v is HARMFUL. 12: end if A.5.1 SYNTHETIC DATASET CONSTRUCTION
The default synthetic task reported in the results section of Section 5.2 was constructed as a dataset of simulated intermediate-layer representations h ∈ R 18 with known ground-truth labels y ∈ {1, . . . , 10} and two causal feature dimensions, where 18comes from 2 feature dimensions plus 16 concatenated noise dimensions is the total feature dimensionality. We split these classes into partition 1 and 2, each consisting of 8 of the 10 classes. The held out classes for partition 1 were contained in partition 2 and visa versa. See Figure 5 for a visualization of the task.
Base feature coordinates. We first defined a grid of base coordinates as the Cartesian product of {-1, 1} along the first feature axis and {0, 1, 2, 3, 4} along the second feature axis:
G = {-1, 1} × {0, 1, 2, 3, 4},(17)
This procedure yields 10 = |G| unique base coordinates, each corresponding to a distinct class label.
Noise and correlation structure. For each base coordinate (x 1 , x 2 ) ∈ G, we generated N noisy samples by adding Gaussian noise with variance 0.1 2 and covariance parameter 0.2. Specifically, each sample was drawn as
x1 x2 ∼ N x 1 x 2 , 0.1 2 0.2 0.2 0.1 2 .(18)
Additional noise dimensions. We augmented each 2D noisy base vector with 16 independent Gaussian noise features, each sampled from N (0, 1), producing final representations h ∈ R 2+16 .
this section cite: []

Section: A.5.2 MLP TRAINING
We trained a feedforward Multi-Layer Perceptron (MLP) classifier to predict the class label y from the synthetic representations h. The MLP was parameterized with:
• input dimensionality d, defaulting to d = 18 as previously described • a 1D batch normalization of d dimensions (Ioffe & Szegedy, 2015),
• one hidden layer of width 128,
• activation function ReLU, • dropout with probability 0.5 to drop (Srivastava et al., 2014),
• a 1D batch normalization of 128 dimensions, Causal Dim 1 Causal Dim 1 Causal Dim 1 Default Task OOD Task Partition Sparse Dense Partition One Two Figure 5: Visualization of the different synthetic tasks used for Figure 3. The Default Task is split into two partitions, both withholding two classes that are contained in the other partition. The OOD task is also split into two partitions, both consisting of 4 classes. The Dense partition consists of a tighter cluster than the Sparse.
this section cite: ['b17', 'b33']

Section: 
Thank you to the PDP Lab and the Stanford Psychology department for funding. Thank you to Noah Goodman and Jay McClelland for thoughtful feedback and discussions. Thank you to the PDP lab and the Stanford Mech Interp community for opportunities to present and for thoughtful discussion.
this section cite: []

Section: LLM USAGE STATEMENT
We used ChatGPT to generally edit for clarity, as well as to improve notational consistency in the behavioral null-space, hidden pathways, and dormant behavioral changes formalizations in Sections 4.1 and 4.2. We also used ChatGPT to provide an initial layout of the proof offered in Appendix A.2 showing that axis-aligned hyperrectangles are the only manifold shape that do not have divergent source-target vector pairs in coordinate patching settings, and we used it to suggest, implement, and generate the initial writeup for the additional divergence measures in Appendix A.1.2.
this section cite: []

Section: A APPENDIX A.1 EMPIRICAL INTERVENED DIVERGENCE METHODOLOGICAL DETAILS

this section cite: []

Section: A.1.1 INTERVENTION METHODS
We considered three families of interpretability interventions that modify hidden-layer representations. In all three, we visualize the residual stream output from the specified layer:
1. Mean Difference Vector Patching (MDVP) (Feng & Steinhardt, 2024), where an intervention vector δ MD ∈ R d is defined as the difference in mean activations between two conditions and then added to or subtracted from activations h ∈ R d . Formally,
ĥ = h + δ MD (11
)
We examine the representations ĥ from a sample size of 100 unique contexts across 4 token positions at each individual layer. We compare the representations to the native cases of the swapped binding positions. We used layer 10 of Meta's Meta-Llama-3-8B-Instruct through Huggingface's transformers package for this task and visualization (Touvron et al., 2023;Wolf et al., 2019). We selected layer 10 as it had the lowest EMD difference of all layers, although, we note that this measure did not necessarily correlate with the subjective interpretation of divergence in the qualitative visualizations. We report the EMD difference in Figure 2(a) as the average over all model layers. 2. Sparse Autoencoder (SAE) Projections (Bloom et al., 2024), where h is projected through a trained encoder E : R d → R k and linear decoder D : R k → R d :
h ′ = D(E(h)).(12)
SAEs are trained with sparsity penalty λ SAE to encourage interpretable basis functions. We offload further experimental details to the referenced SAElens paper and code base. We compare the reconstructed representations to an equal sample size of 2000 vectors from the natural distribution. We used layer 25 of Meta's Meta-Llama-3-8B-Instruct through Huggingface's transformers package for this task and visualization (Touvron et al., 2023;Wolf et al., 2019). We selected layer 25 as it appeared to be the only layer available through SAElens' pretrained SAEs. 3. Distributed Alignment Search (DAS) (Wu et al., 2023), where representations are aligned to a causal abstraction using a learned orthogonal transformation Q ∈ R d×d . See Section 2.2 and Wu et al. (2023) for further detail on the method. We compare the intervened representations to an equal sample size of 1000 vectors from the natural distribution. We used the model and layer specified in Wu et al. (2023) for the visualizations in Figure 2.
this section cite: ['b7', 'b36', 'b43', 'b0', 'b36', 'b43', 'b44', 'b44', 'b44']

Section: A.1.2 MEASURING DIVERGENCE
For each intervened sample, there exists a corresponding ground truth sample that the intervention is meant to approximate. In the case of the mean difference experiments, these ground samples consist of the naturally occurring entity or attribute in the position which the δ M D is meant to approximate. For the SAEs, each reconstructed vector has a corresponding encoded vector. For DAS, the ground truth vectors are equivalent to CL vectors.
this section cite: []

Section: Earth Mover's Distance:
To quantify distributional differences between original and intervened representations, we approximated the Earth Mover's Distance (EMD) (Villani, 2009) including all vector dimensions using the Sinkhorn loss from the GeomLoss python package with a p = 2 and blur = 0.05 (Cuturi, 2013). Let H = {h i } N i=1 denote a set of original representations and Ĥ = { ĥi } N i=1 their intervened counterparts. We computed
EMD(H, Ĥ) = min γ∈Π(µ,ν) 1 N i,j γ ij ∥h i -ĥj ∥ 2 , (13
)
where N is the number of samples in H, µ and ν are the empirical distributions over H and Ĥ, and Π(µ, ν) denotes the set of couplings with marginals µ and ν.
• output layer with 10 logits.
Training was performed with a standard categorical cross-entropy loss:
L CE = - 1 B B i=1 log p θ (y i | h i ),(19)
where B is the batch size and p θ denotes the MLP's predictive distribution over class labels.
We perform the the MLP training on both partitions combined for the default dataset split. Then we perform an alignment function training on each partition independently and test the alignment on the untrained partition. We report the average IIA over both data partitions for DAS analyses over 5 seeds. Note that we use an independent MLP for each partition in the OOD experiment (Appendix A.5.4.
Optimization used stochastic gradient descent with learning rate 0.01 for 300 epochs with early stopping using an Adam optimizer (Kingma & Ba, 2017). The code was implemented in PyTorch.
this section cite: ['b40', 'b4', 'b18']

Section: A.5.3 DAS TRAINING
Following MLP pretraining, we applied Distributed Alignment Search (DAS) with varying intensities of the behavioral loss and contrastive learning (CL) loss terms. Specifically, the DAS objective was
L DAS = ϵ behavior L behavior + ϵ CL L CL ,(20)
where ϵ behavior and ϵ CL are tunable coefficients controlling the strength of each term. We only use values of 0 or 1 for ϵ behavior and we explore values of ϵ CL referring to it as the CL epsilon in figures. We default to an overall learning rate of 0.01 and subspace size of 1 unless otherwise specified. Details of these loss functions are provided in Section 5.
Importantly, we stopped training after loss convergence with a patience of 400 training epochs, and we kept the best DAS alignment matrix decided by IIA, and the best CL alignment matrix by EMD. Furthermore, for these trainings, we used a symmetric invertible linear weight matrix as our alignment function as introduced in Grant et al. (2024). Namely, the linear alignment matrix X is constructed as X = (M M ⊤ + λI)S where M ∈ R dm×dm is a matrix of learned parameters initially sampled from a centered gaussian distribution with a standard deviation of 1 dm , I ∈ R dm×dm is the identity matrix, λ = 0.1 to prevent singular values equal to 0, and S ∈ R dm×dm is a diagonal matrix to learn a sign for each column of X using diagonal values s i,i = Tanh(a i ) + λ(sign(Tanh(a i ))) where each a i is a learned parameter and λ = 0.1 to prevent 0 values. We perform the the alignment function training on both partitions for each synthetic dataset and test each DAS alignment on the untrained partition. We report the average IIA over both data partitions for DAS analyses.
this section cite: ['b14']

Section: A.5.4 OUT-OF-DISTRIBUTION EXPERIMENTAL DETAILS
To perform the OOD CL loss experiments, we partitioned the classes into 2 non-overlapping groups. Two of the 10 classes were excluded entirely. The groups were chosen so that the Sparse set had strictly greater spacing than the Dense set. See a visualization of the Dense and Sparse partitions in Figure 5. A separate MLP training was performed on each partition individually. Then an alignment function was trained on each partition/classifier tuple using the settings specified in Appendix A.5.3. The alignment functions were then tested on the untrained partition. We report IIA values averaged over the performance on each partition.
this section cite: []

Section: A.5.5 FURTHER CL LOSS EXPLORATIONS
In these explorations, we explore DAS learning rate and the number of extra noisy dimensions for the OOD experiments. We show accuracies, EMD divergences, and EMD divergences restricted to the causal dimensions. The EMD values are scaled by the number of extra noisy dimensions. We refer to the EMD measurements along causal dimensions only as the Row EMD. We do this for both the trained partitions and held-out partitions for various DAS trainings.
this section cite: []

Section: A.6 LINEAR REGRESSION
In an effort to establish a more general, concrete relationship between intervened divergence an out-of-distribution (OOD) intervention performance, we performed a linear regression on trained Published as a conference paper at ICLR 2026 alignments using EMD along causal axes (as discovered through the alignment training) as the independent variable and interchange intervention accuracy (IIA) as the dependent variable. We performed these regressions independently on the Default task and the OOD task trainings, each training consisting of two partitions with 5training seeds and 3 types of alignment trainings: DAS behavioral loss only, CL loss only, and DAS+CL loss, creating 30trainings total. We used the statsmodels python package (Seabold & Perktold, 2010) to perform the regression.
Dep. Variable: IIA R-squared: 0.729 Model: OLS Adj. R-squared: 0.719 Method: Least Squares F-statistic: 75.28 Date: Wed, 26 Nov 2025 Prob (F-statistic): 2.00e-09 Time: 11:54:42 Log-Likelihood: 76.575 No. Observations: 30 AIC: -149.2 Df Residuals: 28 BIC: -146.3 Df Model: 1 Covariance Type: nonrobust coef std err t P> |t| [0.025 0.975] Intercept 0.9885 0.004 243.666 0.000 0.980 0.997 Training EMD -0.3424 0.039 -8.677 0.000 -0.423 -0.262 Omnibus: 33.330 Durbin-Watson: 1.903 Prob(Omnibus): 0.000 Jarque-Bera (JB): 98.629 Skew: -2.235 Prob(JB): 3.83e-22 Kurtosis: 10.676 Cond. No. 11.1 0.6 0.8 1.0 Cross Task IIA
this section cite: ['b30']

Section: References
Ref_id:b0 Title:  Year: (2024)
Ref_id:b1 Title: Not all solutions are created equal: An analytical dissociation of functional and representational similarity in deep linear neural networks Year: (2025)
Ref_id:b2 Title: Rethinking circuit completeness in language models: And, or, and adder gates Year: (2025)
Ref_id:b3 Title: Recurrent neural networks learn to store and generate sequences using non-linear representations Year: (2024)
Ref_id:b4 Title: Sinkhorn distances: Lightspeed computation of optimal transport Year: (2013)
Ref_id:b5 Title: Representational analysis of binding in language models Year: (2024)
Ref_id:b6 Title: Toy models of superposition. Transformer Circuits Thread Year: (2022)
Ref_id:b7 Title: How do language models bind entities in context? Year: (2024)
Ref_id:b8 Title: Neural natural language inference models partially embed theories of lexical entailment and negation Year: (2020)
Ref_id:b9 Title: Causal abstractions of neural networks Year: (2021)
Ref_id:b10 Title: Finding alignments between interpretable causal variables and distributed neural representations Year: (2023)
Ref_id:b11 Title: Causal abstraction: A theoretical foundation for mechanistic interpretability Year: (2024)
Ref_id:b12 Title: How causal abstraction underpins computational explanation Year: (2025)
Ref_id:b13 Title: Satchel Grant. Model alignment search Year: (2025)
Ref_id:b14 Title: Emergent symbol-like number variables in artificial neural networks Year: (2024)
Ref_id:b15 Title: How to use and interpret activation patching Year: (2024)
Ref_id:b16 Title: RAVEL: Evaluating interpretability methods on disentangling language model representations Year: (2024-08)
Ref_id:b17 Title: Batch normalization: Accelerating deep network training by reducing internal covariate shift Year: (2015)
Ref_id:b18 Title: Adam: A method for stochastic optimization Year: (2017)
Ref_id:b19 Title: Representation biases: will we achieve complete understanding by analyzing representations? arXiv preprint Year: (2025)
Ref_id:b20 Title: Causal Scrubbing: a method for rigorously testing interpretability hypotheses Year: ()
Ref_id:b21 Title: Less Wrong Year: (2022-12)
Ref_id:b22 Title: On the biology of a large language model. Transformer Circuits Thread Year: (2025)
Ref_id:b23 Title: Is this the subspace you are looking for? an interpretability illusion for subspace activation patching Year: (2023)
Ref_id:b24 Title: Psychological and Biological Models Year: (1986)
Ref_id:b25 Title: Locating and editing factual associations in gpt Year: (2023)
Ref_id:b26 Title: François Portet, and Maxime Peyrard. Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? Year: (2024-10)
Ref_id:b27 Title: Attribution patching: Activation patching at industrial scale Year: (2022)
Ref_id:b28 Title: An Introduction to Causal Inference Year: (2010-02)
Ref_id:b29 Title: Parallel Distributed Processing Year: (1986)
Ref_id:b30 Title: statsmodels: Econometric and statistical modeling with python Year: (2010)
Ref_id:b31 Title: Hypothesis testing the circuit hypothesis in llms Year: (2024)
Ref_id:b32 Title: On the proper treatment of connectionism Year: (1988)
Ref_id:b33 Title: Dropout: A simple way to prevent neural networks from overfitting Year: (2014)
Ref_id:b34 Title: Disentangling adversarial robustness and generalization Year: (2019)
Ref_id:b35 Title: The non-linear representation dilemma: Is causal abstraction enough for mechanistic interpretability? Year: (2025)
Ref_id:b36 Title: Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models Year: (2023)
Ref_id:b37 Title: Manifold-aligned counterfactual explanations for neural networks Year: (2024-05)
Ref_id:b38 Title: Counterfactual explanations and algorithmic recourses for machine learning: A review Year: (2024-10)
Ref_id:b39 Title: Causal mediation analysis for interpreting neural nlp: The case of gender bias Year: (2020)
Ref_id:b40 Title: The wasserstein distances Year: (2009)
Ref_id:b41 Title: Interpretability in the wild: a circuit for indirect object identification in gpt-2 small Year: (2022)
Ref_id:b42 Title: Relational Composition in Neural Networks: A Survey and Call to Action Year: (2024-07)
Ref_id:b43 Title: HuggingFace's Transformers: State-of-the-art Natural Language Processing Year: (2019)
Ref_id:b44 Title: Interpretability at scale: Identifying causal mechanisms in alpaca Year: (2023)
Ref_id:b45 Title: pyvene: A library for understanding and improving PyTorch models via interventions Year: (2024-06)
Ref_id:b46 Title: Towards Best Practices of Activation Patching in Language Models: Metrics and Methods Year: (2024-01)
