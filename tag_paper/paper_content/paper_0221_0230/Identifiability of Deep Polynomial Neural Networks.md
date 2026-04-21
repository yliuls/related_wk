Title: Identifiability of Deep Polynomial Neural Networks
Abstract: Polynomial Neural Networks (PNNs) possess a rich algebraic and geometric structure. However, their identifiability-a key property for ensuring interpretabilityremains poorly understood. In this work, we present a comprehensive analysis of the identifiability of deep PNNs, including architectures with and without bias terms. Our results reveal an intricate interplay between activation degrees and layer widths in achieving identifiability. As special cases, we show that architectures with non-increasing layer widths are generically identifiable under mild conditions, while encoder-decoder networks are identifiable when the decoder widths do not grow too rapidly compared to the activation degrees. Our proofs are constructive and center on a connection between deep PNNs and low-rank tensor decompositions, and Kruskal-type uniqueness theorems. We also settle an open conjecture on the dimension of PNN's neurovarieties, and provide new bounds on the activation degrees required for it to reach the expected dimension.

Section: Introduction
Neural network architectures which use polynomials as activation functions-polynomial neural networks (PNN)-have emerged as architectures that combine competitive experimental performance (capturing high-order interactions between input features) while allowing a fine grained theoretical analysis. On the one hand, PNNs have been employed in many problems in computer vision [1][2][3], image representation [4], physics [5] and finance [6], to name a few. On the other hand, the geometry of function spaces associated with PNNs, called neuromanifolds, can be analyzed using tools from algebraic geometry. Properties of such spaces, such as their dimension, shed light on the impact of a PNN architecture (layer widths and activation degrees) on the expressivity of feedforward, convolutional and self-attention PNN architectures [7][8][9][10][11]. They also determine the landscape of their loss function and the dynamics of their training process [7,12,13].
Moreover, PNNs are also closely linked to low-rank tensor decompositions [14][15][16][17][18], which play a fundamental role in the study of latent variable models due to their identifiability properties [19]. In fact, single-output 2-layer PNNs are equivalent to low-rank symmetric tensors [7]. Identifiabilitywhether the parameters and, consequently, the hidden representations of a NN can be determined from its response up to some equivalence class of trivial ambiguities such as permutations of its neurons-is a key question in NN theory [20][21][22][23][24][25][26][27][28][29][30][31][32]. Identifiability is critical to ensure interpretability in representation learning [33][34][35], to provably obtain disentangled representations [36], and in the study of causal models [37]. It is also critical to understand how the architecture affects the inference process and to support manipulation or "stitching" of pretrained models and representations [35,38,39]. Moreover, it has important links to learning and optimization of PNNs [40,9,13].
Identifiability of deep PNNs is intimately linked to the dimension of their so-called neurovarieties: when this dimension reaches the effective parameter count, the number of possible parametrizations is finite, which means the model is finitely identifiable and the neurovariety is said to be non-defective. In addition, many PNN architectures admit only a single parametrization (i.e., they are globally identifiable).This has been investigated for specific types of self-attention [9] and convolutional [8] layers, and feedforward PNNs without bias [11]. However, current results for feedforward networks only show that finite identifiability holds for very high activation degrees, or for networks with the same widths in every layer [11]. A standing conjecture is that this holds for any PNN with degrees at least quadratic and non-increasing layer widths [11], which parallels identifiability results of ReLU networks [29]. However, a general theory of identifiability of deep PNNs is still missing.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b6', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b6', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b34', 'b37', 'b38', 'b39', 'b8', 'b12', 'b8', 'b7', 'b10', 'b10', 'b10', 'b28']

Section: Our contribution
We provide a comprehensive analysis of the identifiability of deep PNNs considering monomial activation functions. We prove that an L-layer PNN is finitely identifiable if every 2-layer block composed by a pair of two successive layers is finitely identifiable for some subset of their inputs. This surprising result tightly links the identifiability of shallow and deep polynomial networks, which is a key challenge in the general theory of NNs. Moreover, our results reveal an intricate interplay between activation degrees and layer widths in achieving identifiability.
As special cases, we show that architectures with non-increasing layer widths (i.e., pyramidal nets) are generically identifiable, while encoder-decoder (bottleneck) networks are identifiable when the decoder widths do not grow too rapidly compared to the activation degrees. We also show that the minimal activation degrees required to render a PNN identifiable (which is equivalent to its activation thresholds) is only linear in the layer widths, compared to the quadratic bound in [11,Theorem 18]. These results not only settle but generalize conjectures stated in [11]. Moreover, we also address the case of PNNs with biases (which was overlooked in previous theoretical studies) by leveraging a homogenization procedure.
Our proofs are constructive and are based on a connection between deep PNNs and partially symmetric canonical polyadic tensor decompositions (CPD). This allows us to leverage Kruskal-type uniqueness theorems for tensors to obtain identifiability results for 2-layer networks, which serve as the building block in the proof of the finite identifiability of deep nets, which is performed by induction. Our results also shed light on the geometry of the neurovarieties, as they lead to conditions under which its dimension reaches the expected (maximum) value.
this section cite: ['b10', 'b10']

Section: Related works
Polynomial NNs: Several works studied PNNs from the lens of algebraic geometry using their associated neuromanifolds and neurovarieties [7] (in the emerging field of neuroalgebraic geometry [41]) and their close connection to tensor decompositions. Kileel et al. [7] studied the expressivity or feedforward PNNs in terms of the dimension of their neurovarieties. An analysis of the neuromanifolds for several architectures was presented in [10]. Conditions under which training losses do not exhibit bad local minima or spurious valleys were also investigated [13,12,42]. The links between training 2-layer PNNs and low-rank tensor approximation [13] as well as the biases of gradient descent [43] have been established.
Recent work computed the dimensions of neuromanifolds associated with special types of selfattention [9] and convolutional [8] architectures, and also include identifiability results. For feedforward PNNs, finite identifiability was demonstrated for networks with the same widths in every layer [11], while stronger results are available for the 2-layer case with more general polynomial activations [44]. Finite identifiability also holds when the activation degrees are larger than a so-called activation degree threshold [11]. Recent work studied the singularities of PNNs with activations consisting of the sum of monomials with very high activation degrees [45]. PNNs are also linked to factorization machines [46]; this led to the development of efficient tensor-based learning algorithms [47,48]. Note that other types of non-monomial polynomial-type activations [49,50,5,51] have shown excellent performance; however, the geometry of these models is not well known.
this section cite: ['b6', 'b40', 'b6', 'b9', 'b12', 'b11', 'b41', 'b12', 'b42', 'b8', 'b7', 'b10', 'b43', 'b10', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b4', 'b50']

Section: NN identifiability:
Many studies focused on the identifiability of 2-layer NNs with tanh, odd, and ReLU activation functions [20][21][22][23]. Moreover, algorithms to learn 2-layer NNs with unique parameter recovery guarantees have been proposed (see, e.g., [52,53]), however, their extension to NNs with 3 or more layers is challenging and currently uses heuristics [54]. Identifiability of deep NNs under weak genericity assumptions was first studied in the pioneering work of Fefferman [24] for the case of the tanh activation function through the study of its singularities. Recent work extended this result to more general sigmoidal activations [25,26]. Various works focused on deep ReLU nets, which are piecewise linear [28]; they have been shown to be generically identifiable if the number of neurons per layer is non-increasing [29]. Recent work studied the local identifiability of ReLU nets [30][31][32]. Identifiability has also been studied for latent variable/causal modeling, leveraging different types of assumptions (e.g., sparsity, statistical independence, etc.) [55][56][57][58][59][60]. Note that although some of these works tackle deep NNs, their proof techniques are completely different from our approach and do not apply to the case of polynomial activation functions.
Tensors and NNs: Low-rank tensor decompositions had widespread practical impact in the compression of NN weights [61][62][63][64][65]. Moreover, their properties also played a key role in the theory of NNs [18]. This includes the study of the expressivity of convolutional [66] and recurrent [67,68] NNs, and the sample complexity of reinforcement learning parametrized by low-rank transition and reward tensors [69,70]. The decomposability of low-rank symmetric tensors was also paramount in establishing conditions under which 2-layer NNs can (or cannot [71]) be learned in polynomial time and in the development of algorithms with identifiability guarantees [52,72,73]. It was also used to study identifiability of some deep linear networks [74]. However, the use of tensor decompositions in the studying the identifiability of deep nonlinear networks has not yet been investigated.
2 Setup and background 2.1 Polynomial neural networks: with and without bias Polynomial neural networks are functions R d0 → R d L represented as feedforward networks with bias terms and activation functions of the form ρ r (•) = (•) r . Our results hold for both the real and complex valued case (F = R, C), thus, and we prefer to keep the real notation for simplicity. Note that we allow the activation functions to have a different degree r ℓ for each layer.
this section cite: ['b19', 'b20', 'b21', 'b22', 'b51', 'b52', 'b53', 'b23', 'b24', 'b25', 'b27', 'b28', 'b29', 'b30', 'b31', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b17', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b51', 'b71', 'b72', 'b73']

Section: Definition 1 (PNN).
A polynomial neural network (PNN) with biases and architecture (d = (d 0 , d 1 , . . . , d L ), r = (r 1 , . . . , r L-1 )) is a map R d0 → R d L given by a feedforward neural network
PNN d,r [θ] = PNN r [θ] := f L • ρ r L-1 • f L-1 • ρ r L-2 • • • • • ρ r1 • f 1 ,(1)
where f i (x) = W i x + b i are affine maps, with W i ∈ R di×di-1 being the weight matrices and b i ∈ R di the biases, and the activation functions ρ r : R d → R d , defined as ρ r (z) := (z r 1 , . . . , z r d ) are monomial. The parameters θ are given by the entries of the weights W i and biases b i , i.e.,
θ = (w, b), w = (W 1 , W 2 , . . . , W L ), b = (b 1 , b 2 , . . . , b L ).(2)
The vector of degrees r is called the activation degree of PNN r [θ] (we often omit the subscript d if it is clear from the context).
PNNs are algebraic maps and are polynomial vectors, where the total degree is r total = r 1 • • • r L-1 , that is, they belong to the polynomial space (P d,r total ) ×d L , where P d,r denotes the space of dvariate polynomials of degree ≤ r. Most previous works analyzed the simpler case of PNNs without bias, which we refer to as homogeneous. Due to its importance, we consider it explicitly.
this section cite: []

Section: Definition 2 (hPNN).
A PNN is said to be a homogenous PNN (hPNN) when it has no biases (b ℓ = 0 for all ℓ = 1, . . . , L), and is denoted as
hPNN d,r [w] = hPNN r [w] := W L • ρ r L-1 • W L-1 • ρ r L-2 • • • • • ρ r1 • W 1 .(3)
Its parameter set is given by w = (W 1 , W 2 , . . . , W L ).
It is well known that such PNNs are in fact homogeneous polynomial vectors and belong to the polynomial space (H d0,r total ) ×d L , where H d,r ⊂ P d,r denotes the space of homogeneous d-variate polynomials of degree r. hPNNs are also naturally linked to tensors and tensor decompositions, whose properties can be used in their theoretical analysis. Example 3 (Running example). Consider an hPNN with L = 2, r = (2) and d = (3, 2, 2). In such a case the parameter matrices are given as
W 2 = b 11 b 12 b 21 b 22 , W 1 = a 11 a 12 a 13 a 21 a 22 a 23 ,
and the hPNN p = hPNN r [w] is a vector polynomial that admits the expression
p(x) = W 2 ρ 2 (W 1 x) = b 11 b 21 (a 11 x 1 + a 12 x 2 + a 13 x 3 ) 2 + b 12 b 22 (a 21 x 1 + a 22 x 2 + a 23 x 3 ) 2 .
the only monomials that can appear are of the form x i 1 x j 2 x k 3 with i + j + k = 2 thus p is a vector of degree-2 homogeneous polynomials in 3 variables (in our notation, p ∈ (H 3,2 ) 2 ).
this section cite: []

Section: Equivalent PNN representations
It is known that the PNNs admit equivalent representations (i.e., several parameters θ leading to the same function). Indeed, for each hidden layer we can (a) permute the hidden neurons, and (b) rescale the input and output to each activation function since for any a ̸ = 0, (at) r = a r t r . These transformations lead to different sets of parameters that leave the PNN unchanged. We can characterize all such equivalent representations in the following lemma (provided in [7] for the case without biases). Lemma 4. Let PNN d,r [θ] be a PNN with θ as in (2). Let also D ℓ ∈ R d ℓ ×d ℓ be any invertible diagonal matrices and P ℓ ∈ Z d ℓ ×d ℓ (ℓ = 1, . . . , L -1) be permutation matrices, and define the transformed parameters as
W ′ ℓ ← P ℓ D ℓ W ℓ D -r ℓ-1 ℓ-1 P T ℓ-1 , b ′ ℓ ← P ℓ D ℓ b ℓ
θ ′ = ((W ′ 1 , W ′ 2 , . . . , W ′ L ), (b ′ 1 , b ′ 2 , . . . , b ′ L )
). If θ and θ ′ are linked with such a transformation, they are called equivalent (denoted θ ∼ θ ′ ).
this section cite: ['b6', 'b1']

Section: Example 5 (Example 3, continued).
In Example 3 we can take any α, β ̸ = 0 to get
hPNN d,r [w] = α -2 b 11 α -2 b 21 (αa 11 x 1 +αa 12 x 2 +αa 13 x 3 ) 2 + β -2 b 12 β -2 b 22 (βa 21 x 1 +βa 22 x 2 +βa 23 x 3 ) 2 .
which correspond to rescaling rows of W 1 and corresponding columns of W 2 . If we additionally permute them, we get W ′ 1 = P DW 1 , W ′ 2 = W 2 D -2 P T with D = α 0 0 β and P = [
0 1 1 0 ]. This characterization of equivalent representations allows us to define when a PNN is unique. Definition 6 (Unique and finite-to-one representation). The PNN p = PNN d,r [θ] (resp. hPNN p = hPNN d,r [w]) with parameters θ (resp. w) is said have a unique representation if every other representation satisfying p = PNN d,r [θ ′ ] (resp. p = hPNN d,r [w ′ ]) is given by an equivalent set of parameters, i.e., θ ′ ∼ θ (resp. w ′ ∼ w) in the sense of Lemma 4 (i.e., they can be obtained from the permutations and elementwise scalings in Lemma 4). Similarly, a PNN p = PNN d,r [θ] (resp. hPNN p = hPNN d,r [w]) is called finite-to-one if it admits only finitely many non-equivalent representations, that is, the set {θ ′ : PNN d,r [θ ′ ] = p} (resp. {w ′ : hPNN d,r [w ′ ] = p}) contains finitely many non-equivalent parameters. Example 7 (Example 5, continued). Thanks to links with tensor decompositions and their uniqueness, it is known that the hPNN in Example 3 has unique representation if W 2 is invertible and W 1 full row rank (rank 2), see Proposition 35 in Section 4.2. 2.3 Identifiability and link to neurovarieties An immediate question is which PNN/hPNN architectures are expected to admit only a single (or finitely many) non-equivalent representations? This question can be formalized using the notions of global and finite identifiability, which considers a general set of parameters. Definition 8 (Global and finite identifiability). The PNN (resp. hPNN) with architecture (d, r) is said to be globally identifiable if for a general choice of θ = (w, b) ∈ R d ℓ (d ℓ-1 +1) , (resp. w ∈ R d ℓ d ℓ-1 ) (i.e., for all choices of parameters except for a set of Lebesgue measure zero), the network PNN d,r [θ] (resp. hPNN d,r [w]) has a unique representation. Similarly, the PNN (resp. hPNN) with architecture (d, r) is said to be finitely identifiable if for a general choice of θ, (resp. w) the network PNN d,r [θ] (resp. hPNN d,r [w]) is finite-to-one (i.e., it admits only finitely many non-equivalent representations).
In the following, we use the term "identifiable" to refer to finite identifiability unless stated otherwise. Note also that the notion of finite identifiability is much stronger than the related notion of local identifiability (i.e., a model being identifiable only in a neighborhood of a parameterization).
this section cite: []

Section: Example 9 (Example 7, continued).
From Example 7, we see that the hPNN architecture with d = (3, 2, 2), r = (2) is identifiable due to the fact that generic matrices W 1 and W 2 are full rank.
Note that Definition 8 excludes a set of parameters of Lebesgue measure zero. Thus, for an identifiable architecture such as the one mentioned in Example 9, there exists rare sets of pathological parameters for which the hPNN is non-unique (e.g., weight matrices containing collinear rows).
With some abuse of notation, let hPNN d,r [•] be the map taking w to hPNN d,r [w]. Then the image of hPNN d,r [•] is called a neuromanifold, and the neurovariety V d,r is defined as its closure in the Zariski topology 3 . The study of neurovarieties and their properties is a topic of recent interest [7,41,11,10]. More details are given in Appendix A. An important property for our case is the link between identifiability of an hPNN, the dimension of its neurovariety, and the rank of its Jacobian. Proposition 10. The architecture hPNN d,r [•] is finitely identifiable if and only if the dimension of V d,r is equal to the effective number of parameters, i.e., dim
V d,r = L ℓ=1 d ℓ d ℓ-1 - L-1 ℓ=1 d ℓ .
In such case, V d,r is said to be nondefective. Equivalently, the rank of the Jacobian of the map hPNN d,r [•] is maximal and equal to
L ℓ=1 d ℓ d ℓ-1 - L-1 ℓ=1 d ℓ at a general parameter w.
this section cite: ['b6', 'b40', 'b10', 'b9']

Section: Main results

this section cite: []

Section: Main results on the identifiability of deep hPNNs
Although several works have studied the identifiability of 2-layer NNs, tackling the case of deep networks is significantly harder. However, when we consider the opposite statement, i.e., the nonidentifiability of a network, it is much easier to show such connection: in a deep network with L > 2 layers, the lack of identifiability of any 2-layer subnetwork (formed by two consecutive layers) clearly implies that the full network is not identifiable. What our main result shows is that, surprisingly, under mild additional conditions the converse is also true for hPNNs: if the every 2-layer subnetwork is identifiable for some subset of their inputs, then the full network is identifiable as well. This is formalized in the following theorem. Theorem 11 (Localization theorem). Let ((d 0 , . . . , d L ), (r 1 , . . . , r L-1 )) be the hPNN format. For ℓ = 0, . . . , L -2 denote d ℓ := min{d 0 , . . . , d ℓ }. Then the following holds true: if for all ℓ = 1, . . . , L -1 the two-layer architecture hPNN ( d ℓ-1 ,d ℓ ,d ℓ+1 ),r ℓ [•] is finitely identifiable, then the L-layer architecture hPNN d,r [•] is finitely identifiable as well.
The technical proofs are relegated to the appendices. This key result shows a strong relation between the finite identifiability of shallow and deep hPNNs. However, as we move into the deeper layers, the identifiability conditions required by Theorem 11 are stricter than in the shallow case, since the number of inputs is reduced to d ℓ . This can lead to a requirement of larger activation degrees to guarantee identifiability compared to the shallow case.
Theorem 11 allows us to derive identifiability conditions for hPNNs using the link between 2-layer hPNNs and partially symmetric tensor decompositions and their generic uniqueness based on classical Kruskal-type conditions. We use the following sufficient condition for the identifiability of shallow networks. Proposition 12 (Sufficient condition for identifiability of 2-layer hPNN). Let d 0 , d 1 ≥ 2, d 2 ≥ 1 be the layer widths and r ≥ 2 such that
r ≥ 2d 1 -min(d 2 , d 1 ) min(d 1 , d 0 ) -1 . (4
)
Then the 2-layer hPNN with architecture ((d 0 , d 1 , d 2 ), r) is globally identifiable.
this section cite: []

Section: Remark 13.
If the above condition is satisfied for every 2-layer architecture (( d ℓ-1 , d ℓ , d ℓ+1 ), r ℓ ), ℓ = 1, . . . , L -1, then Theorem 11 implies that the L-layer hPNN is finitely identifiable for the L-layer architecture (d, r).
this section cite: []

Section: Remark 14.
Note that for the single output case d L = 1, Equation (4) means the activation degree in the last layer must satisfy r L-1 ≥ 3, in contrast to r ℓ ≥ 2 for ℓ < L -1. Remark 15 (Our bounds are constructive). We note that the condition (4) for identifiability is not the best possible (and can be further improved using much stronger results on generic uniqueness of decompositions, see e.g., [75,Corollary 37]). However, the bound (4) is constructive, and we can use standard polynomial-time tensor algorithms to recover the parameters of the 2-layer hPNN.
this section cite: ['b74']

Section: Implications for specific architectures
Proposition 12 has direct implications for the finite identifiability of several architectures of practical interest, including pyramidal and bottleneck networks, and for the activation thresholds of hPNNs, as shown in the following corollaries. Corollary 16 (Pyramidal hPNNs are always identifiable). The hPNNs with architectures containing non-increasing layer widths (except possibly the last layer), i.e., d 0 ≥ d 1 ≥ • • • d L-1 ≥ 2 and d L ≥ 1, are finitely identifiable for any degrees satisfying
(i) r 1 , . . . , r L-1 ≥ 2 if d L ≥ 2; or (ii) r 1 , . . . , r L-2 ≥ 2, r L-1 ≥ 3 if d L = 1.
Note that, due to the connection between the identifiability of hPNNs and the neurovarieties presented in Proposition 10, a direct consequence of Corollary 16 is that the neurovariety V d,r has expected dimension. This settles a recent conjecture presented in [11,Section 4]. This implication is explained in detail in Appendix A.
Instead of seeking conditions on the layer widths for a fixed (or minimal) degree, a complementary perspective is to determine what are the smallest degrees r ℓ such that a given architecture d is finitely identifiable. Following the terminology introduced in [11], we refer to those values as the activation degree thresholds for identifiability of an hPNN. An upper bound is given in the following corollary: Corollary 17 (Activation degree thresholds for identifiability). For fixed layer widths d = (d 0 , . . . , d L ) with d ℓ ≥ 2, ℓ = 0, . . . , L -1, the hPNNs with architectures (d, (r 1 , . . . , r L-1 )) are finitely identifiable for any degrees satisfying r ℓ ≥ 2d ℓ -1 .
Note that due to Proposition 10, the result in this corollary implies that the neurovariety V d,r has expected dimension. This means that (2d ℓ -1) is also a universal upper bound to the so-called activation thresholds for hPNN expressiveness introduced in [11]. The existence of such activation degree thresholds was conjectured in [7] and recently proved in [11,Theorem 18], but the for a quadratic in d ℓ bound (the bound in Corollary 17 is linear).
this section cite: ['b10', 'b10', 'b10', 'b6', 'b10']

Section: Remark 18 (Admissible layer sizes).
The possible layer sizes in a deep network are tightly linked with the degree of the activation. For example, for r ℓ = 2, identifiability is impossible if
d ℓ > d ℓ-1 (d ℓ-1 +1) 2 (for general r ℓ , a similar bound O(d r ℓ ℓ-1 )
follows from a link with tensor decompositions [76]). Therefore, to allow for larger layer widths, we need to have higher-degree activations.
It is enlightening to consider the admissible layer widths when taking into account the joint effect of layer widths and degrees. By doing this, Proposition 12 can be leveraged to yield identifiability conditions for the case of bottleneck networks, as illustrated in the following corollary. Corollary 19 (Identifiability of bottleneck hPNNs). Consider the "bottleneck" architecture with
d 0 ≥ d 1 ≥ • • • ≥ d b ≤ d b+1 ≤ . . . ≤ d L
and d b ≥ 2. Suppose that r 1 , . . . , r b ≥ 2 and that the decoder part satisfies d ℓ r ℓ ≤ d b -1 for ℓ ∈ {b + 1, . . . , L -1}. Then the bottleneck hPNN is finitely identifiable.
This shows that encoder-decoder hPNNs architectures are identifiable under mild conditions on the layer widths and decoder degrees, providing a polynomial networks-based counterpart to previous studies that analyzed linear autoencoders [77,78].
Note that the width of the bottleneck layer d b constrains the entire decoder part of the architecture: the degrees r ℓ , ℓ ≥ b are constrained according to the width d b . The presence of bottlenecks has also been shown to affect the expressivity of hPNNs in [7,Theorem 19]: for d b = 2d 0 -2 there exists a number of layers L such that for r ℓ ≥ 2 and d 0 ≥ 2, the hPNN neurovariety is non-filling (i.e., its dimension never reaches that of the ambient space) for any choice of widths d 1 , . . . , d b-1 , d b+1 , . . . , d L .
this section cite: ['b75', 'b76', 'b77', 'b6']

Section: PNNs with biases
The identifiability of general PNNs (with biases) can be studied via the properties of hPNNs. The simplest idea is truncation (i.e., taking only higher-order terms of the polynomials), which eliminates biases from PNNs. Such an approach was already taken in [44] for shallow PNNs with general polynomial activation, and is described in Appendix D.3. We will follow a different approach based on the well-known idea of homogenization: we transform a PNN to an equivalent hPNN with structured parameters keeping the information about biases at the expense of increasing the layer widths. Our key result is to show how this can be used to study the identifiability of PNNs with bias terms. The following correspondence is well-known.
this section cite: ['b43']

Section: Definition 20 (Homogenization).
There is a one-to-one mapping between polynomials in d variables of degree r and homogeneous polynomials of the same degree in d + 1 variables. We denote this mapping P d,r → H d+1,r by homog(•), and it acts as follows: for every polynomial p ∈ P d,r , p = homog(p) ∈ H d+1,r (that is p(x 1 , . . . , x d , x d+1 )) is the unique homogeneous polynomial in d + 1 variables such that p(x 1 , . . . , x d , 1) = p(x 1 , . . . , x d ). Example 21. For the polynomial p ∈ P 2,2 in variables (x 1 , x 2 ) given by
p(x 1 , x 2 ) = ax 2 1 + bx 1 x 2 + cx 2 2 + ex 1 + f x 2 + g, its homogenization p = homog(p) ∈ H 3,2 in 3 variables (x 1 , x 2 , x 3 ) is p(x 1 , x 2 , x 3 ) = ax 2 1 + bx 1 x 2 + cx 2 2 + ex 1 x 3 + f x 2 x 3 + gx 2 3
, and we can verify that p(1, x 1 , x 2 ) = p(x 1 , x 2 ).
Similarly, we extend homogenization to polynomial vectors, which gives the following. Example 22. Let f (x) = W 2 ρ r1 (W 1 x + b 1 ) + b 2 , and define extended matrices as
W 1 = W 1 b 1 0 1 ∈ R (d1+1)×(d0+1) , W 2 = [W 2 b 2 ] ∈ R d2×(d1+1)
Then its homogenization f = homog(f ) is an hPNN of format
(d 0 + 1, d 1 + 1, d 2 ) f ( x) = W 2 ρ r1 W 1 x
where x = [x 0 , x 1 , . . . , x d0 , x d0+1 ] T , so that f (x 1 , . . . , x d0 , 1) = f (x 1 , . . . , x d0 ).
The construction in Example 22 similar to the well-known idea of augmenting the network with an artificial (constant) input. The following proposition generalizes this example to the case of multiple layers, by "propagating" the constant input. Proposition 23. Fix the architecture r = (r 1 , . . . , r L-1 ) and d = (d 0 , . . . , d L ). Then a polynomial vector p ∈ (P d0,r total ) ×d L admits a PNN representation p = PNN d,r [(w, b)] with (w, b) as in (2) if and only if its homogenization p = homog(p) admits an hPNN decomposition for the same activation degrees r and extended
d = (d 0 + 1, . . . , d L-1 + 1, d L ), p = hPNN d,r [ w], w = ( W 1 , . . . , W L )
, with matrices given as
W ℓ =    W ℓ b ℓ 0 1 ∈ R (d ℓ +1)×(d ℓ-1 +1) , ℓ < L, W L b L ∈ R (d L )×(d L-1 +1) , ℓ = L.
That is, PNNs are in one-to-one correspondence to hPNNs with increased number of inputs and structured weight matrices.
Uniqueness of PNNs from homogenization: An important consequence of homogenization is that the uniqueness of the homogenized hPNN implies the uniqueness of the original PNN with bias terms, which is a key result to support the application of our identifiability results to general PNNs. Proposition 24. If hPNN r [ w] from Proposition 23 is unique (resp. finite-to-one) as an hPNN (without taking into account the structure), then the original PNN representation PNN r [(w, b)] is unique (resp. finite-to-one).
The proposition follows from the fact that we can always fix the permutation ambiguity for the "artificial" input. Remark 25. Despite the one-to-one correspondence, for generic properties (e.g., finite identifiability) we cannot immediately apply the results from the homogeneous case, because the matrices W ℓ are structured (they form a set of measure zero inside R (d ℓ +1)×(d ℓ-1 +1) ).
However, we can prove that the identifiability of the hPNN implies the identifiability of the PNN. Lemma 26. Let the 2-layer hPNN architecture be finitely (resp. globally) identifiable for ((d 0 + 1, d 1 + 1, d 2 ), r 1 ). Then the PNN architecture with widths (d 0 , d 1 , d 2 ) and degree r 1 is also finitely (resp. globally) identifiable.
Using Lemma 26 and specializing the proof of Theorem 11, we obtain the following result:
Proposition 27. Let ((d 0 , . . . , d L ), (r 1 , . . . , r L-1 )) be the PNN format. For ℓ = 0, . . . , L -2 denote d ℓ = min{d 0 , . . . , d ℓ }.
Then the following holds true: If for all ℓ = 1, . . . , L -1 each two-layer architecture hPNN
( d ℓ-1 +1,d ℓ +1,d ℓ+1 ),r ℓ [•]
is finitely identifiable, then the L-layer PNN with architecture (d, r) is finitely identifiable as well.
In particular, we have the following bounds for generic uniqueness.
Corollary 28. Let ((d 0 , . . . , d L ), (r 1 , . . . , r L-1 )) be such that d ℓ ≥ 1, and r ℓ ≥ 2 satisfy r ℓ ≥ 2(d ℓ + 1) -min(d ℓ + 1, d ℓ+1 ) min(d ℓ , d ℓ-1 ) ,
then the L-layer PNN with architecture (d, r) is finitely identifiable (and globally identifiable if L = 2).
this section cite: []

Section: Remark 29.
For general PNNs with bias, similar conclusions hold to the ones in the hPNN case.
In particular, for fixed layer widths d ℓ ≥ 1, the activation threshold for a PNN architecture (d, r) becomes r ℓ ≥ 2d ℓ + 1. Also, pyramidal PNNs are identifiable in degree 2.
A distinctive feature of PNNs with bias is that they can be identifiable even for architectures with layers containing a single hidden neuron: for d ℓ = 1 and d ℓ+1 ≥ 2 and/or d ℓ-1 = 1, the condition in Corollary 28 is still satisfied when r ℓ ≥ 2.
this section cite: []

Section: Proofs and main tools
Our main results in Theorem 11 translates the identifiability conditions of deep hPNNs into those of shallow hPNNs. Our results are strongly related to the decomposition of partially symmetric tensors (we review basic facts about tensors and tensors decompositions and recall their connection between to hPNNs in later subsections). More details are provided in the appendices, and we list key components of the proof below.
this section cite: []

Section: Identifiability of deep PNNs: necessary conditions
Increasing hidden layers breaks uniqueness. The key insight is that if we add to any architecture a neuron in any hidden layer, then the uniqueness of the hPNN is not possible, which is formalized as following lemma (whose proof is based, in its turn, on tensor decompositions). Lemma 30. Let p = hPNN r [w] be an hPNN of format (d 0 , . . . , d ℓ , . . . , d L ). Then for any ℓ there exists an infinite number of representations of hPNNs p = hPNN r [w] with architecture (d 0 , . . . , d ℓ + 1, . . . , d L ). In particular, the augmented hPNN is not unique (and is not finite-to-one).
Internal features of a unique hPNN are linearly independent. This is an easy consequence of Lemma 30 (as linear dependence would allow for pruning neurons). Lemma 31. For d = (d 0 , . . . , d L ), let p = hPNN r [w] have a unique (or finite-to-one) L-layers decomposition. Consider the output at any ℓ-th internal level ℓ < L after the activations
q ℓ (x) = ρ r ℓ • W ℓ • • • • • ρ r1 • W 1 (x).(5)
Then the elements of
q ℓ (x) = [q ℓ,1 (x) • • • q ℓ,d ℓ (x)]
T are linearly independent polynomials.
Identifiability for hPNNs and Kruskal rank. Identifiability of 2-layer hPNNs, or equivalently uniqueness of CPD is strongly related to the concept of Kruskal rank of a matrix that we define below.
Definition 32. The Kruskal rank of a matrix A (denoted krank{A}) is the maximal number k such that any k columns of A are linearly independent.
This is in contrast with the usual rank, which is the maximal k such that there exist k linearly independent columns. Therefore krank{A} ≤ rank{A}. Note that krank{A} ≥ 2 means that none of the pairs of columns of A are linearly dependent (no columns are pairwise collinear). Using the notion of Kruskal rank, we can state a necessary condition on weight matrices for identifiability of hPNNs, which is a generalization of the well-known necessary condition for the uniqueness of CPD tensor decompositions (6) (i.e., shallow networks), and is a corollary of Lemma 30 and Lemma 31. Proposition 33. As in Lemma 31, let the widths be d = (d 0 , . . . , d L ), and p = hPNN r [w] have a unique (or finite-to-one) L-layers decomposition. Then we have that for all ℓ = 1, . . . , L -1 krank{W T ℓ } ≥ 2, krank{W ℓ+1 } ≥ 1, where krank{W ℓ+1 } ≥ 1 simply means that W ℓ+1 does not have zero columns.
this section cite: []

Section: Shallow hPNNs and tensor decompositions
An order-s tensor T ∈ R m1×•••×ms is an s-way multidimensional array (more details are provided in Appendix B.2 and more background on tensors can be found in [14][15][16]). It is said to have a d-term CPD (canonical polyadic decomposition) if it admits a decomposition into d rank-1 terms T = d j=1 a 1,j ⊗ • • • ⊗ a s,j for a i,j ∈ R mi , with ⊗ being the tensor (outer) product. The CPD is also written compactly as
T = [[A 1 , A 2 , • • • , A s ]] for matrices A i = [a i,1 , • • • , a i,d ] ∈ R mi×d .
T is said to be (partially) symmetric if it is invariant to any permutation of (a subset) of its indices [79]. Concretely, we will consider tensors T partially symmetric on dimensions i ∈ {2, . . . , s}, with CPD that is also partially symmetric, i.e., with
A i , i ≥ 2 satisfying A 2 = A 3 = • • • = A s .
Our main proofs strongly rely on results of [7] on the connection between hPNN and tensors decomposition in the shallow (i.e., 2-layer) case (see also [79]). Proposition 34. There is a one-to-one mapping between partially symmetric tensors F ∈ R d2×d0×•••×d0 and polynomial vectors f ∈ (H d0,r ) ×d2 , which can be written as
F → f (x) = F (1) x ⊗r ,
with F (1) ∈ R d2×d r 0 the first unfolding of F . Under this mapping, the partially symmetric CPD
F = [[W 2 , W T 1 , • • • , W T 1 ]](6)
is mapped to hPNN W 2 ρ r (W 1 x). Thus, uniqueness of hPNN (d0,d1,d2),r [(W 1 , W 2 )] is equivalent to uniqueness of the partially symmetric CPD of F .
Thanks to the link with the partially symmetric CPD, we prove the following Kruskal-based sufficient condition for uniqueness (which is a counterpart of Proposition 33).
Proposition 35. Let p w (x) = W 2 ρ r1 (W 1 x) be a 2-layer hPNN with layer sizes (d 0 , d 1 , d 2 ) satisfying d 0 , d 1 ≥ 2, d 2 ≥ 1. Assume that r ≥ 2, krank{W 2 } ≥ 1, krank{W T 1 } ≥ 2 and that: r ≥ 2d 1 -krank{W 2 } krank{W T 1 } -1 ,
then the 2-layer hPNN p w (x) is unique (or equivalently, the CPD of F in (6) is unique). Remark 36. For 2-layer hPNNs (L = 2), when the activation degree r is high enough Proposition 33 gives both necessary and sufficient conditions for uniqueness due to Proposition 35. Remark 37. Proposition 35 forms the basis of the proof of Proposition 12, which comes from the fact that the Kruskal rank of a generic matrix is equal to its smallest dimension. Remark 38. Proposition 35 is based on basic (Kruskal) uniqueness conditions [80][81][82]. As mentioned in Remark 15, by using more powerful results on generic uniqueness [83,84], we can obtain better bounds for identifiability of 2-layer PNNs. For example, for "bottleneck" architectures (as in Corollary 19), the results of [83, imply that for degrees r ℓ = 2, identifiability holds for decoder layer sizes satisfying a weaker condition
d ℓ ≤ (d b -1)d b 2 (instead of d ℓ r ℓ ≤ d b -1).
this section cite: ['b13', 'b14', 'b15', 'b78', 'b6', 'b78', 'b79', 'b80', 'b81', 'b82', 'b83', 'b82']

Section: Proof of the main result
The proof of Theorem 11 proceeds by induction over the layers ℓ = 1, . . . , L. The key idea is based on a procedure that allows us to prove finite identifiability of the L-th layer given the assumption that the previous layers are identifiable. For this, we introduce a map (last layer map)
ψ[q, W L ] := W L ρ r L-1 (q(x 1 , . . . , x d0 )),(7)
where q is the vector polynomial of degree R = r 1 • • • r L-2 , representing the output of the (L -1)-th linear layer. Then the L-layer hPNN is a composition:
hPNN r [θ, W L ] = ψ[hPNN (r1,...,r L-2 ) [θ], W L ], for θ = (W 1 , . . . , W L-1
). To obtain finite identifiability, we look at the Jacobian of the composite map. The key to this recursion is to show that the Jacobian J ψ (q, W L ) (Jacobian of ψ with respect to the input polynomial vector and W L ) is of maximal possible rank. For this, we construct a "certificate" of finite identifiability q realized by hPNN (r1,...,r L-2 ) [ θ], but of simpler structure which inherits identifiability of a shallow hPNN. Remark 39. For d L = 1, maximality of the rank for J ψ (q, W L ) is closely related to nondefectivity of the variety of sums of powers of forms, which is often proved by establishing Hilbert genericity of an ideal generated by the elements of q (a question raised in Fröberg conjecture, see e.g., [85]).
A key limitation of our techniques is that they only allow for establishing finite identifiability for deep PNNs. There exist recent results linking finite and global identifiability, [75,86] but only for additive decompositions (shallow case). We state, however, the following conjecture. Conjecture 40. Under the assumptions of Theorem 11, the L-layer hPNN is globally identifiable.
Note that the conjecture may be valid only for global identifiability (i.e., for a generic choice of parameters) and not for uniqueness, since it is not true that the composition of unique shallow hPNNs yield a unique deep hPNNs, as shown by the following example.
Example 41. Consider two polynomials: p(x 1 , x 2 ) = (x 2 1 + x 2 2 ) 2 (x 2 1 -x 2 2 ) 2 T . We see that this polynomial vector admits two different representations p(x) = Iρ 2 (W 2 ρ 2 (Ix)) = W 3 ρ 2 1 2 W 2 ρ 2 (W 2 x) ,with
W 2 = 1 1 1 -1 , W 3 = 1 0 1 -1 ,
which are not equivalent. However, each 2-layer subnetwork is unique (see Example 7).
this section cite: ['b84', 'b74', 'b85']

Section: Discussion
In this paper, we presented a comprehensive analysis of the identifiability of deep feedforward PNNs by using their connections to tensor decompositions. Our main result is the localization of identifiability, showing that deep PNNs are finitely identifiable if every 2-layer subnetwork is also finitely identifiable for a subset of their inputs. Our results can be also useful for compression (pruning) neural networks as they give an indication about the architectures that are not reducible. An important perspective is also to understand when two different identifiable PNN architectures can represent the same function, as the identifiable representations can potentially occur for different non-compatible formats (e.g., a PNN in format d = (2, 4, 4, 2) could be potentially pruned to two different identifiable representations, say, d = (2, 3, 4, 2) and d = (2, 4, 3, 2)).
While our results focus on the case of monomial activations, we believe that this approach can be extended for establishing theoretical guarantees for other types of architectures and activation functions.
In fact, the monomial case constitutes as a key first step in addressing general polynomial activations (see, e.g., [45]) which, in turn, can approximate most commonly used activations on compact sets. Moreover, the close connection between PNNs and partially symmetric tensor decompositions (which benefit from efficient computational algorithms based on linear algebra [87]) can also serve as support for the development of computational algorithms based on tensor decompositions for training deep PNNs. In fact, tensor decompositions have been combined with the method of moments to learn small NN architectures (see, e.g., [52,88]), extending such approaches for training deep PNNs with finite datasets is an important direction for future work.
A roadmap to the appendices 4The appendices of the paper contain background on tensor decompositions and neurovarieties, the proofs of the technical results, as well as a discussion on the changes between the originally submitted and final version of the paper. They are organized as follows:
• Appendix A presents background on neurovarieties for homogeneous PNNs. This is a crucial part for understanding the link between finite identifiability of an hPNN, the dimension of its neurovariety and the rank of the Jacobian of its parametrization map.
• Appendix B contains the main technical tools used in the proof the localization theorem and follows the structure of Section 4. In particular, it presents the proofs of necessary conditions for uniqueness (Section 4.1), background on tensor decompositions and Kruskal-based sufficient conditions for the identifiability of 2-layer hPNNs (Section 4.2).
• Appendix C presents the proof of the localization theorem (Theorem 11) and its consequences for several hPNN architectures, as well as some supporting technical results.
• Appendix D presents the proofs for the case of PNNs with biases. Appendix D.3 discusses the idea of truncation, an alternative approach to tackle the PNNs with biases.
• Appendix E discusses necessary and sufficient conditions for the identifiability of hPNNs, as well as changes between the originally submitted and the final version of the paper which were done to correct a mistake in the proof of one of the main results.
A Homogeneous PNNs and neurovarieties hPNNs are often studied through the prism of neurovarieties, using their algebraic structure. Our results have direct implications on the expected dimension of the neurovarieties, as explained in this appendix.
this section cite: ['b44', 'b86', 'b51', 'b87']

Section: References
Ref_id:b0 Title: P-nets: Deep polynomial neural networks Year: (2020)
Ref_id:b1 Title: Augmenting deep classifiers with polynomial neural networks Year: (2022)
Ref_id:b2 Title: Polynet: Polynomial neural network for 3D shape recognition with polyshape representation Year: (2021)
Ref_id:b3 Title: Polynomial neural fields for subband decomposition and manipulation Year: (2022)
Ref_id:b4 Title: Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving PDEs Year: (2021)
Ref_id:b5 Title: Estimating stock closing indices using a GAweighted condensed polynomial neural network Year: (2018)
Ref_id:b6 Title: On the expressive power of deep polynomial neural networks Year: (2019)
Ref_id:b7 Title: On the geometry and optimization of polynomial convolutional networks Year: (2025)
Ref_id:b8 Title: Geometry of lightning selfattention: Identifiability and dimension Year: ()
Ref_id:b9 Title: Geometry of polynomial neural networks Year: (2024)
Ref_id:b10 Title: Activation degree thresholds and expressiveness of polynomial neural networks Year: (2025)
Ref_id:b11 Title: Spurious valleys and clustering behavior of neural networks Year: (2023)
Ref_id:b12 Title: Geometry and optimization of shallow polynomial networks Year: (2025)
Ref_id:b13 Title: Tensor decompositions and applications Year: (2009)
Ref_id:b14 Title: Tensor decomposition for signal processing and machine learning Year: (2017)
Ref_id:b15 Title: Tensor networks for dimensionality reduction and large-scale optimization: Part 1 low-rank tensor decompositions Year: (2016)
Ref_id:b16 Title: Uniqueness of tensor decompositions with applications to polynomial identifiability Year: (2014)
Ref_id:b17 Title: Low-rank tensor decompositions for the theory of neural networks Year: (2026)
Ref_id:b18 Title: Tensor decompositions for learning latent variable models Year: (2014)
Ref_id:b19 Title: Uniqueness of the weights for minimal feedforward nets with a given input-output map Year: (1992)
Ref_id:b20 Title: For neural networks, function determines form Year: (1993)
Ref_id:b21 Title: Uniqueness of weights for neural networks Year: (1993)
Ref_id:b22 Title: Notes on the symmetries of 2layer ReLU-networks Year: (2020)
Ref_id:b23 Title: Reconstructing a neural net from its output Year: (1994)
Ref_id:b24 Title: Affine symmetries and neural network identifiability Year: (2021)
Ref_id:b25 Title: Neural network identifiability for a family of sigmoidal nonlinearities Year: (2022)
Ref_id:b26 Title: Expand-and-cluster: parameter recovery of neural networks Year: (2024)
Ref_id:b27 Title: Reverse-engineering deep ReLU networks Year: (2020)
Ref_id:b28 Title: Functional vs. parametric equivalence of ReLU networks Year: (2020)
Ref_id:b29 Title: An embedding of ReLU networks and an analysis of their identifiability Year: (2022)
Ref_id:b30 Title: Local identifiability of deep ReLU neural networks: the theory Year: (2022)
Ref_id:b31 Title: Parameter identifiability of a deep feedforward ReLU neural network Year: (2023)
Ref_id:b32 Title: Disentanglement via mechanism sparsity regularization: A new principle for nonlinear ICA Year: (2021)
Ref_id:b33 Title: Indeterminacy in generative models: Characterization and strong identifiability Year: (2023)
Ref_id:b34 Title: On the symmetries of deep learning models and their internal representations Year: (2022)
Ref_id:b35 Title: Challenging common assumptions in the unsupervised learning of disentangled representations Year: (2019)
Ref_id:b36 Title: From identifiable causal representations to controllable counterfactual generation: A survey on causal generative modeling Year: (2024)
Ref_id:b37 Title: Linear mode connectivity between multiple models modulo permutation symmetries Year: (2025)
Ref_id:b38 Title: Git re-basin: Merging models modulo permutation symmetries Year: (2023)
Ref_id:b39 Title: Algebraic geometry and statistical learning theory Year: (2009)
Ref_id:b40 Title: Position: Algebra unveils deep learning -an invitation to neuroalgebraic geometry Year: (2025)
Ref_id:b41 Title: Avoiding spurious local minima in deep quadratic networks Year: (2019)
Ref_id:b42 Title: The spectral bias of polynomial neural networks Year: (2022)
Ref_id:b43 Title: Identifiability of an X-rank decomposition of polynomial maps Year: (2017)
Ref_id:b44 Title: Learning on a razor's edge: the singularity bias of polynomial neural networks Year: (2025)
Ref_id:b45 Title: Factorization machines Year: (2010)
Ref_id:b46 Title: Polynomial networks and factorization machines: New insights and efficient training algorithms Year: (2016)
Ref_id:b47 Title: Multi-output polynomial networks and factorization machines Year: (2017)
Ref_id:b48 Title: Ladder polynomial neural networks Year: (2021)
Ref_id:b49 Title: On expressivity and trainability of quadratic networks Year: (2023)
Ref_id:b50 Title: Polynomial composition activations: Unleashing the dynamics of large language models Year: (2025)
Ref_id:b51 Title: Beating the perils of non-convexity: Guaranteed training of neural networks using tensor methods Year: (2015)
Ref_id:b52 Title: Robust and resource efficient identification of shallow neural networks by fewest samples. Information and Inference: A Year: (2021)
Ref_id:b53 Title: Stable recovery of entangled weights: Towards robust identification of deep neural networks from minimal samples Year: (2023)
Ref_id:b54 Title: Identifiability of latent-variable and structural-equation models: from linear to nonlinear Year: (2024)
Ref_id:b55 Title: Variational autoencoders and nonlinear ICA: A unifying framework Year: (2020)
Ref_id:b56 Title: Nonparametric identifiability of causal representations from unknown interventions Year: (2023)
Ref_id:b57 Title: On linear identifiability of learned representations Year: (2021)
Ref_id:b58 Title: On the identifiability of nonlinear ICA: Sparsity and beyond Year: (2022)
Ref_id:b59 Title: Identifiability of deep generative models without auxiliary information Year: (2022)
Ref_id:b60 Title: Speeding-up convolutional neural networks using fine-tuned CP-decomposition Year: (2015)
Ref_id:b61 Title: Tensorizing neural networks Year: (2015)
Ref_id:b62 Title: Tensor decomposition for model reduction in neural networks: A review Year: (2023)
Ref_id:b63 Title: Stable low-rank tensor decomposition for compression of convolutional neural network Year: (2020)
Ref_id:b64 Title: Geometry-aware training of factorized layers in tensor Tucker format. Proceedings Year: (2024)
Ref_id:b65 Title: On the expressive power of deep learning: A tensor analysis Year: (2016)
Ref_id:b66 Title: A tensor decomposition perspective on second-order RNNs Year: (2024)
Ref_id:b67 Title: Expressive power of recurrent neural networks Year: (2018)
Ref_id:b68 Title: Tesseract: Tensorised actors for multi-agent reinforcement learning Year: (2021)
Ref_id:b69 Title: Tensor and matrix low-rank valuefunction approximation in reinforcement learning Year: (2024)
Ref_id:b70 Title: On the connection between learning two-layer neural networks and tensor decomposition Year: (2019)
Ref_id:b71 Title: Learning two-layer neural networks with symmetric inputs Year: (2019)
Ref_id:b72 Title: Efficient algorithms for learning depth-2 neural networks with general ReLU activations Year: (2021)
Ref_id:b73 Title: Multilinear compressive sensing and an application to convolutional linear networks Year: (2019)
Ref_id:b74 Title: From non-defectivity to identifiability Year: (2022)
Ref_id:b75 Title: Tensors: Geometry and applications Year: (2012)
Ref_id:b76 Title: Aleksandrina Goeva, and Cotton Seed. Loss landscapes of regularized linear autoencoders Year: (2019)
Ref_id:b77 Title: Regularized linear autoencoders recover the principal components, eventually Year: (2020)
Ref_id:b78 Title: Symmetric tensors and symmetric tensor rank Year: (2008)
Ref_id:b79 Title: Identifiability results for blind beamforming in incoherent multipath with small delay spread Year: (2001)
Ref_id:b80 Title: On the uniqueness of the canonical polyadic decomposition of third-order tensors-Part II: Uniqueness of the overall decomposition Year: (2013)
Ref_id:b81 Title: On the uniqueness of multilinear decomposition of N-way arrays Year: (2000)
Ref_id:b82 Title: Generic uniqueness conditions for the canonical polyadic decomposition and INDSCAL Year: (2015)
Ref_id:b83 Title: On the dimensions of secant varieties of Segre-Veronese varieties Year: (2013)
Ref_id:b84 Title: Algebraic stories from one and from the other pockets Year: (2018)
Ref_id:b85 Title: Bronowski's conjecture and the identifiability of projective varieties Year: (2024)
Ref_id:b86 Title: Symmetric tensor decomposition by an iterative eigendecomposition algorithm Year: (2016)
Ref_id:b87 Title: Learning a deep convolutional neural network via tensor decomposition. Information and Inference: A Year: (2021)
Ref_id:b88 Title: Real algebraic geometry Year: (2013)
Ref_id:b89 Title: Algebraic compressed sensing Year: (2023)
Ref_id:b90 Title: Semialgebraic geometry of nonnegative tensor rank Year: (2016)
Ref_id:b91 Title: On generic and maximal k-ranks of binary forms Year: (2019)
Ref_id:b92 Title: On uniqueness of power sum decomposition Year: (2025)
Ref_id:b93 Title: The Alexander-Hirschowitz theorem for neurovarieties Year: (2025)
