Title: Transformative or Conservative? Conservation laws for ResNets and Transformers
Abstract: While conservation laws in gradient flow training dynamics are well understood for (mostly shallow) ReLU and linear networks, their study remains largely unexplored for more practical architectures. This paper bridges this gap by deriving and analyzing conservation laws for modern architectures, with a focus on convolutional ResNets and Transformer networks. For this, we first show that basic building blocks such as ReLU (or linear) shallow networks, with or without convolution, have easily expressed conservation laws, and no more than the known ones. In the case of a single attention layer, we also completely describe all conservation laws, and we show that residual blocks have the same conservation laws as the same block without skip connection. We then introduce the notion of conservation laws that depend only on a subset of parameters (corresponding e.g. to a pair of consecutive layers, to a residual block, or to an attention layer). We demonstrate that the characterization of such laws can be reduced to the analysis of the corresponding building block in isolation. Finally, we examine how these newly discovered conservation principles, initially established in the continuous gradient flow regime, persist under discrete optimization dynamics, particularly in the context of Stochastic Gradient Descent (SGD).

Section: Introduction
Understanding the behavior of neural networks during training remains a fundamental challenge in deep learning. A particularly insightful approach to this challenge involves studying conserved functions -quantities that remain invariant throughout the training process. These conserved functions reveal important geometric properties of the training dynamics and serve a dual purpose. First, they provide valuable insights into the implicit bias induced by both the training algorithm and network architecture, by revealing properties that persist from initialization to the final solution (Saxe et al., 2013;Bah et al., 2022;Arora et al., 2018;Tarmoun et al., 2021;Min et al., 2021). Second, they have emerged as crucial tools in theoretical analyses, playing a key role in convergence studies (Du et al., 2018;Arora et al., 2019;Bah et al., 2022;Chizat & Bach, 2020;Ji & Telgarsky, 2019a;Min et al., 2021). Understanding these conservation laws can also be applied to designing new optimization schemes, which no longer preserve these laws but instead enforce them to reach a desired value (e.g., a balanced condition for ReLU networks) in order to potentially accelerate convergence (Saul, 2023;Stock et al., 2019).
this section cite: ['b26', 'b3', 'b0', 'b29', 'b23', 'b11', 'b1', 'b3', 'b9', 'b23', 'b25', 'b28']

Section: Conservation laws.
In the context of Euclidean gradient flow training dynamics, conservation laws in the form of "balancedness conditions" have been established for ReLU and linear networks (Saxe et al., 2013;Du et al., 2018;Arora et al., 2019). Subsequently, (Marcotte et al., 2023) demonstrated the "completeness" of these laws: no additional conservation laws exist for these architectures in the shallow case under Euclidean gradient flows. For these network architectures, (Marcotte et al., 2024) unveiled novel conservation laws when considering alternative optimization algorithms -particularly non-Euclidean gradient flows, as employed in ICNN or NMF, or momentum-based dynamics, also demonstrating their completeness. Furthermore, (Marcotte et al., 2024) revealed that conservation laws under momentum dynamics exhibit fundamentally different characteristics compared to simple gradient flows: these laws are time and velocity-dependent, and are generally fewer in number than in the gradient flow case. For feed-forward networks with single-channel convolutions, conservation laws were also identified under gradient flow dynamics (Du et al., 2018). While the investigation of conservation laws for more sophisticated neural architectures has remained largely unexplored, this paper addresses this gap by extending the analysis to more complex network architectures.
Residual networks (ResNets) constitute a fundamental class of deep learning architectures that revolutionized the field of computer vision through their groundbreaking performance (He et al., 2016). The distinguishing feature of these networks-the incorporation of skip connections-has since become a cornerstone principle in other deep learning architectures, most notably exemplified in Transformer models (Vaswani, 2017). In (Marion et al., 2023), under specific initialization assumptions and incorporating a rescaling operation, the authors demonstrate that the solution reached during traing (i.e. the trained neural network) corresponds to a discretization of a Neural ODE (Chen et al., 2018), thus revealing an implicit bias. This enables leveraging ODE (ordinary differential equation) theory to analyze the trained network.
Transformers. Since their introduction (Vaswani, 2017), Transformers and their multi-head attention mechanism (Bahdanau et al., 2015) have achieved unprecedented performance across domains from natural language processing (Brown et al., 2020) to computer vision (Dosovitskiy et al., 2021). In (Vasudeva et al., 2024), the authors shows that when training self-attention layers with gradient descent, the key-query matrix naturally converges to a hard-margin SVM solution, revealing an implicit bias similar to that observed in linear logistic regression on separable data (Soudry et al., 2018;Ji & Telgarsky, 2019b).
Our contributions. After proving that conservation laws of gradient flows with weight decay are determined by their equivalent without weight decay (Theorem 2.1), we discover new conservation laws and show that these new laws are complete for several basic building blocks of modern networks with skip connections: shallow multi-channel convolution layers (Theorem 3.6), self-attention layers (Corollary 3.9, Corollary 3.10), cross-entropy classification layer (Proposition 3.11), and plain MLP with skip connections (Proposition 3.2). We then explain how these results can be used to analyze deeper networks, via a generic analysis under the lens of the new taylored analysis of conservation laws that only depend on a given subset of parameters (Proposition 4.3). Notable results (Theorem 4.6) include the formal proof that such laws exactly match the laws of the classical blocks considered in isolation. Besides, in the case of residual convolutional networks we show (Theorem 4.7) the absence of conservation law associated to consecutive linear layers that "overlap" two residual blocks. To complete the theoretical analysis we finally show that the conservation laws of gradient flows are also approximately preserved during actual (discrete) SGD dynamics, under appropriate assumptions (Proposition 5.1), with an error bound (27) scaling as O(step-size 2 ) that we showcase in our numerical experiments, Figure 1 and Figure 5.
this section cite: ['b26', 'b11', 'b1', 'b20', 'b21', 'b21', 'b11', 'b13', 'b31', 'b22', 'b8', 'b31', 'b4', 'b7', 'b10', 'b30', 'b27']

Section: Conservation laws for Gradient Flows
In this paper, we consider learning problems where the features are represented as x i ∈ R m and the targets or labels are denoted by y i ∈ Y. In regression tasks, Y is typically defined as R n , while in classification contexts, y i represents categorical labels. In scenarios involving unsupervised or self-supervised learning, y i can be treated as a constant. We denote z i := (x i , y i ) and Z = (x i , y i ) i .
Predictions are generated through a parametric function g(θ, •) : R m → R n (such as a neural network). This function is trained by empirically minimizing a cost function with respect to the parameters θ ∈ Θ ⊆ R D .
L Z (θ) := 1 N i ℓ(g(θ, x i ), y i ),(1)
with ℓ a loss function. In practice, the training dynamics are realized using a gradient descent algorithm with weight decay:
θ k+1 = θ k -τ ∇L Z (θ k ) -λ k τ θ k .(2)
To facilitate mathematical analysis, the training dynamics is simplified by considering a gradient flow (GF) with weight decay (WD). This approach represents the continuous counterpart of Equation ( 2) as τ approaches zero and can be expressed as the first-order ODE where λ(t) ≥ 0:
θ(t) + λ(t)θ(t) weight-decay = -∇L Z (θ(t)),(3)
We aim to understand what quantities are preserved during the dynamic (3) for a variety of neural networks g. The mathematical study of what happens in discrete dynamics is done in Section 5. The analogue of the transition from ( 2) to ( 3) for a simplified version of Adam algorithm is analyzed in Section 2.3.
this section cite: []

Section: Conservation laws of neural networks
A function h(t, θ) is conserved if for each solution θ(t) of the ODE (3) with any initialization and any data-set, one has h(t, θ(t)) that remains constant over time.
Time-dependency: with vs without weight-decay. Our first contribution is the following theorem, which clarifies a fundamental distinction in the temporal dependence of conserved quantities of a dynamic system, whether it includes WD or not. It also demonstrates how the conservation laws with WD can be readily derived from those from the system without WD. See Appendix A for a proof.
Theorem 2.1 (Structure theorem). Let h(t, θ) be a conserved function for the ODE (3).
If for every θ, there exists a data-set Z such that ∇L Z (θ) = 0, then the function H(a) := h(0, a) satisfies h(t, θ) = H θ exp( t 0 λ(s)ds) , ∀t, θ. Thus, conserved functions can be expressed with D variables (instead of D + 1). Moreover, h(t, θ) := H(θ) is a conservation law of (3) without WD (i.e. with λ(t) ≡ 0). Remark 2.2. In particular, when considering (3) with λ(t) ≡ 0, one has h(t, θ) = h(0, θ) for all t and θ.
Given this established correspondence between conserved functions with and without weight-decay, we can now restrict our analysis to time-independent conserved functions h(θ) in the case of gradient descent without WD:
θ(t) = -∇L Z (θ(t)).(4)
Definition and characterization of conservation laws.
Here we recall the main ingredients of the framework for conservation laws from (Marcotte et al., 2023), introducing some formal definitions of intermediate objects and results that hopefully streamline the corresponding reasoning. A function h(θ) is a conservation law for g if for each solution θ(t) of the ODE (4) with any initialization and any data-set, one has h(θ(t)) = h(θ(0)), ∀t. The formal definition of a conservation law in that case is given in (Marcotte et al., 2023, Definition 2.4) (corresponds to the notion of being locally conserved on Θ for any dataset), and we recall an orthogonal characterization of a smooth conservation law (Marcotte et al., 2023, Corollary 2.6, Proposition 2.7): Proposition 2.3. Assume that for each y ∈ Y, the loss ℓ(z, y) is C 2 -differentiable with respect to z ∈ R n . A function h ∈ Cfoot_4 (Θ, R) is a conservation law for g with respect to the loss ℓ if and only if ∇h(θ) ⊥ W g,ℓ θ for all θ ∈ Θ where:
W g,ℓ θ := span Z=(xi,yi)∈(X θ ×Y) N {∇L Z (θ)} = span (x,y)∈X θ ×Y {∂ θ g(θ, x) ⊤ ∇ z ℓ(g(θ, x), y)}⊆ R D , with X θ the set of data points x∈ R m such that g(•, x) is C 2 -differentiable in the neighborhood of θ.
Assumption 2.4. Consider a loss ℓ(z, y). We assume that for every y it is differentiable with respect to z ∈ R n . We define for all z ∈ R n the subspace
V ℓ (z) := span y ∇ z ℓ(z, y)⊆ R n .
We also assume that V ℓ (z) does not depend on z ∈ R n , so we rewrite V ℓ (z) = V ℓ .
In particular, this assumption is satisfied for all classical losses (e.g. mean-square error, Kullblack-Leibler divergence, cross-entropy loss) as stated in (Marcotte et al., 2023, Lemma C.2, Remark C.3), and is known to imply the following direct corollary:
Corollary 2.5. Consider a loss ℓ(z, y) that satisfies Assumption 2.4. Then for all θ ∈ Θ:
W g,ℓ θ = span x∈X θ ,w∈V ℓ {∂ θ g(θ, x) ⊤ w}.
Another useful assumption is the following.
this section cite: ['b20', 'b20', 'b20', 'b20']

Section: Assumption 2.6 (Local reparameterization).
There exists d and ϕ ∈ C 2 (Θ, R d ) such that: for each parameter θ 0 in the open set
Θ ⊆ R D , for each x ∈ X such that θ → g(θ, x) is C 2 in a neighborhood of θ 0 1 , there is a neighborhood Ω of θ 0 and f (•, x) ∈ C 2 (ϕ(Ω), R n ) such that ∀θ ∈ Ω, g(θ, x) = f (ϕ(θ), x).(5)
Such a factorization g(θ, x) = f (ϕ(θ), x) is always possible but never unique: ϕ = id and f = g yield a trivial factorization, and starting from an arbitrary factorization any diffeomorphism ψ yields another one g(θ, x) = f ( φ(θ), x) with f (a, x) := f (ψ(a), x) and φ := ψ -1 • ϕ. The corresponding space W g,ℓ θ can be characterized with any such factorization (Marcotte et al., 2023, Proposition 2.12).
Proposition 2.7. Assume that for every y the loss ℓ(z, y) is C 2 -differentiable with respect to z. Under Assumption 2.6, for all θ ∈ Θ:
W g,ℓ θ = ∂ϕ(θ) ⊤ W f,ℓ ϕ(θ)(6)
with ∂ϕ(θ) ∈ R d×D the Jacobian of ϕ and
W f,ℓ ϕ(θ) := span (x,y)∈X θ ×Y {∂f x (ϕ(θ)) ⊤ ∇ z ℓ(g(θ, x), y)}, where f x (•) := f (•, x).
Under Assumption 2.6, Proposition 2.7 directly rewrites as:
Corollary 2.8. Consider a loss ℓ(z, y) that satisfies Assumption 2.4. Under Assumption 2.6, then for all θ ∈ Θ
W f,ℓ ϕ(θ) = span (x,w)∈X θ ×V ℓ {[∂f x (ϕ(θ))] ⊤ w}.(7)
and, denoting P ℓ (ϕ(θ)) ∈ R d×d the matrix of the projection on this finite-dimensional vector subspace, we have
W g,ℓ θ = range ∂ϕ(θ) ⊤ P ℓ (ϕ(θ)) .(8)
With plain shallow ReLU and linear networks, when V ℓ = R n (this holds with the Euclidean loss or the Kullback-Leibler loss), there happens to a be a "distinguished" choice of ϕ (Marcotte et al., 2023, Lemma 2.13) such that the projection matrix P ℓ is simply the identity, so that all the needed information about W g,ℓ θ is captured in ∂ϕ. The formalism with P ℓ enhances the flexibility of the framework to encompass the variety of possible factorizations via f and ϕ.
In light of (8) we introduce the vectors fields:
χ ℓ i : θ → ∂ϕ(θ) ⊤ P ℓ (ϕ(θ))e i , 1 ≤ i ≤ d(9)
and the linear function space:
W g,ℓ := span{χ ℓ 1 , • • • , χ ℓ d },
so that we have for any θ ∈ Θ, W g,ℓ θ = W g,ℓ (θ), where the trace at θ ∈ Θ of any set W ⊂ C 1 (Θ, R D ) of vector fields is defined as the linear space
W(θ) := span{χ(θ) : χ ∈ W} ⊆ R D .(10)
In particular h ∈ C 1 is a conservation law of g with respect to the loss ℓ if, and only if, its gradient is orthogonal to χ ℓ i (θ) for every i and θ. This property is stable by Lie brackets (for self-containedness see a reminder on the underlying calculus in Appendix B), leading to a new orthogonal characterization of conservation laws (proved in Appendix B).
Proposition 2.9. If ℓ(z, y) satisfies Assumption 2.4 then h ∈ C 1 (Θ, R) is a conservation law for g with respect to ℓ if and only if ∇h(θ) ⊥ Lie(span i {χ ℓ i })(θ) for all θ ∈ Θ.
this section cite: ['b20']

Section: Existence and "number" of conservation laws
Having characterized the conservation laws, we now seek to ascertain the quantity of such laws. However, to comprehend the number that exists, it is essential to establish a notion of independence that eliminates all functional redundancies. We recall the definition of independency from (Marcotte et al., 2023, Definition 2.18):
Definition 2.10. A family of functions h i , 1 ≤ i ≤ N is said to be independent if the vectors ∇h i (θ), 1 ≤ i ≤ N are linearly independent for all θ ∈ Ω.
The following fundamental theorem (Marcotte et al., 2023, Theorem 3.3) provides a formula for the exact number of independent conservation laws.
Theorem 2.11. If W g,ℓ ⊆ C ∞ Θ, R D and dim(Lie(W g,ℓ )(θ)) is locally constant (equal to some k) then each θ ∈ Θ ⊆ R D admits a neighborhood U 0 such that there are D -k smooth (C ∞ ) independent conservation laws h k+1 , • • • , h D of g with respect to ℓ on U 0 .
The proof of this theorem in (Marcotte et al., 2023, Theorem 3.3) relies on Frobenius Theorem (recalled in Theorem C.1) and necessitates a reasoning by contradiction, along with the use of an intermediate functional space. In this article, we present a significantly simplified proof (see Appendix C) that is based solely on the Frobenius Theorem and the characterization of Proposition 2.9.
We now demonstrate why Definition 2.10 effectively eliminates all functional redundancies. The following proposition states (see Appendix D for a proof) that any conservation law can be expressed locally as a function of the independent reference conservation laws obtained in Theorem 2.11. Proposition 2.12. Consider a smooth conservation law h : Θ → R of g with respect to ℓ, and θ ∈ Θ around which dim(Lie(W g,ℓ )(θ)) is locally constant (equal to some integer k). Then, on the neighborhood U 0 of θ given by Theorem 2.11, h can be expressed as a function of the D -k smooth independent conservation laws h k+1 , • • • , h D of g given by Theorem 2.11. Marcotte et al. (2023) developed a code (detailed in their Section 3.3) that calculates the dimension of the trace of the Lie algebra generated by a finite set of vector fields.
this section cite: ['b20', 'b20']

Section: Conservation laws for Adam flows
A simplified version of Adam algorithm (Kingma & Ba, 2014) (full batch, without bias correction and ε, β 1 , β 2 = 0)foot_6 updates the parameters θ, with an estimate of the first and second moments m t , v t using the following equations:
m t = -∇ θ L Z (θ t ), v t = -(∇ θ L Z (θ t )) 2 , θ t+1 = θ t -η m t √ v t = θ t -ηsign(∇ θ L Z (θ t ))(11)
where the square, square-root, and division are done element-wise.
The discrete dynamic (11) corresponds to the explicit Euler scheme of the ODE (informally corresponds to ( 11) when η → 0):
θ = -sign (∇ θ L Z (θ)) .(12)
Remark 2.13. Connections between the Adam algorithm and variants of sign gradient descent (referred to as "variance-adapted sign descent") have been established in (Balles & Hennig, 2018) in the discrete dynamic.
Thus one can adapt the results of Section 2.1 by considering the ODE (12) instead of (4). In particular under the same assumptions as in Proposition 2.3, a conservation law h for g for the Adam flow (12) with respect to the loss ℓ is now characterized by ∇h(θ) ⊥ W g,ℓ θ , ∀θ where
W g,ℓ θ := span Z=(xi,yi)∈(X θ ×Y) N {sign (∇L Z (θ))}.(13)
In particular, the associated space W g,ℓ is locally constant with respect to θ. Thus, the trace of the generated Lie algebra at θ is directly equal to the one of W g,ℓ at θ: by using Theorem 2.11, it suffices to determine the dimension of the trace of W g,ℓ to know the number of independent conservation laws. In the case of a 2-layer linear neural network parameterized by ϕ : (U, V ) ∈ R n×r × R m×r → U V ⊤ (and similarly for an attention layer), we empirically find that there are no conservation laws, except in the special case n = m = r = 1, where there is exactly one conservation law, given by |U | -|V |, as detailed in Appendix R.
this section cite: ['b17', 'b5']

Section: The case of shallow neural networks
Equipped with the general results of the previous section we now provide characterizations of the conservation laws of several elementary networks that serve as building blocks of standard modern network architectures such as ResNets and Transformers. After showing that a basic block has the same conservation laws with or without skip connections, we remind existing laws for shallow ReLU and linear networks, before providing our main contributions of this section : a characterization of the conservation laws of shallow ReLU networks with convolutions, of attention layers, and of crossentropy classification layers. Remark 3.1 (Conservation and invariances). In this section we characterize all conservation laws for elementary networks (except multihead layers where the completeness of the identified laws is left to future work). It is noteworthy that these laws in most cases are intrinsically connected to network invariances (as detailed in Appendix J.1). When considering weight decay regularization and applying the structure theorem Theorem 2.1, a particular consequence of our results is that the conservation laws from the nonregularized case (GF without weight decay) vanish at the optimum. This finding not only aligns with partially known "balancedness" properties ( (Yang et al., 2022, Theorem 1) applies here in the case of a ReLU activation, as the associated elementary networks g θ are homogeneous with respect to hidden neuron rescaling, and our results show that it is also true for linear networks and for cross-entropy classification layers ( 20)); it also provides additional insight on the dynamics of the convergence to balanced parameters.
this section cite: ['b32']

Section: With vs without residual connections
We first establish a simple but generic result for conservation laws in the presence of skip connexions. Given any neural network g(θ, •) with n = m, we consider the residual neural network g(θ, •) defined by:
g(θ, •) : x ∈ R n → x + g(θ, x) ∈ R n .(14)
Proposition 3.2. With respect to any loss satisfying Assumption 2.4 g and g have the same conservation laws.
Proof. Use Proposition 2.3 and Corollary 2.5, and notice that since ∂ θ g = ∂ θ g we have W g,ℓ θ = W g,ℓ θ . Remark 3.3. It is worth noticing that this result does not require the assumption V ℓ = R n .
this section cite: []

Section: ReLU and linear networks: known results
Let us consider U ∈ R n×r , V ∈ R m×r , and we denote θ := (U, V ) the parameters and u k , v k the columns of U and V . In that case, the neural network writes:
g(θ, x) = U σ(V ⊤ x),(15)
where σ = id (resp. σ = ReLU) in the linear case case (resp. ReLU case). The number of parameters is D = (n + m)r.
We recall here the result demonstrated in (Marcotte et al., 2023, Lemma 2.13, Theorem 2.14) which shows that, through an appropriate parameterization ϕ, the study of W g,ℓ can be reduced to the study of a Lie algebra generated by a finite number of 'well-behaved' vector fields. Theorem 3.4. Under Assumption 2.4, if V ℓ = R n , then considering Θ = R D and ϕ(θ) := U V ⊤ for linear neural networks, one has: W f,ℓ ϕ(θ) = R d and W g,ℓ θ = range(∂ϕ(θ) ⊤ ). The same result holds for 2-layer ReLU networks (with or without bias b k ∈ R, 1 ≤ k ≤ r) with Θ ⊆ R D the set of all parameters θ such that hidden neurons define pairwise distinct hyperplanes
H k := {x ∈ R d , v ⊤ k + b k = 0}, and ϕ(θ) := (u k v ⊤ k ) r k=1
Indeed, by Theorem 3.4, computing the associated generated Lie algebra and finally applying Theorem 2.11, the authors are able to determine all conservation laws in these settings (Marcotte et al., 2023, Corollary 4.4): Proposition 3.5. With the assumptions of Theorem 3.4, in the ReLU (resp. linear) case, we have: for any θ ∈ Θ (resp. θ ∈ Θ such that ( U V ) has full rank), there is a neighborhood of θ in which all conservation laws for (15) are functions of
∥u k ∥ 2 -∥v k ∥ 2 (resp. of ⟨u k , u l ⟩ -⟨v k , v l ⟩), 1 ≤ k, l ≤ r.
this section cite: ['b20', 'b20']

Section: ReLU neural networks with convolutions
We now consider the case of a basic block of a one-hidden layer ReLU neural network with convolutions. This means that the input vector x ∈ R m is considered as the concatenation of channels x (i) ∈ R p , 1 ≤ i ≤ c 0 each with p pixels (for images), or p samples (in case of time series), so that m = c 0 p. Accordingly the output of the network is given by (15) where the matrices V and U are made of circulant blocks respectively corresponding to convolutions with filters v j,i , u k,j , 1 ≤ i ≤ c 0 , 1 ≤ j ≤ c 1 (c 1 is the number of channels of the hidden layer), and 1 ≤ k ≤ c 2 (c 2 is the number of channels of the output y = g(θ, x) ∈ R n , considered as the concatenation of channels y (k) ∈ R n1 , so that n = c 2 × n 1 .
More explicitly this corresponds to
g(θ, x) := c1 j=1 u k,j ⋆ σ c0 i=1 v j,i ⋆ x (i) c2 k=1 . (16)
Assuming that the filters all satisfy u k,j ∈ R nu (resp. v i,j ∈ R nv ) the network parameters θ := ((u k,j ) k,j , (v j,i ) j,i ), are of dimension D := c 2 c
1 n u + c 1 c 0 n v = c 1 (c 2 n u + c 0 n v ).
We define Θ conv as the set of all θ such that the matrix V has all its columns that define pairwise distinct hyperplanes.
A conservation law for this network has already been established in (Du et al., 2018, Theorem 2.3) for a singlechannel networks (c 0 = c 1 = c 2 = 1). We find that similar functions are preserved in the multi-channel case, and we demonstrate that there are no others by characterizing the trace of the associated Lie algebra as well as its dimension. The following theorem presents all conservation laws of the network (16). A more general version of this result is proved in Appendix E, which allows for instance to consider strided convolutions (Zhang, 2019) (i.e. with zeros inserted in the filter) in order to define translation invariant CNNs.
Theorem 3.6. Under Assumption 2.4, if V ℓ = R n , then in the neighborhood of each θ ∈ Θ conv there are exactly c 1 independent conservation laws for (16) given by
h j (θ) := c2 k=1 ∥u k,j ∥ 2 - c0 i=1 ∥v j,i ∥ 2 , 1 ≤ j ≤ c 1 . (17)
Remark 3.7. The formulation of the neural network (16) in the multi-channel convolutive case generalizes the one (15) of a simple 2-layer ReLU network without convolution. Specifically, setting p = 1 and identifying c 0 and n v with m, c 1 with r, and c 2 and n u with n, yields (15) and the conservation laws obtained in Proposition 3.5 coincide with the ones given in Theorem 3.6.
this section cite: ['b33']

Section: One attention layer
For an attention layer, the input X ∈ R N ×dim is reshaped as the concatenation x ∈ R m of N tokens x (i) ∈ R dim , with m = N dim. The layer output is
g(θ, x) = softmax(XQ ⊤ KX ⊤ )XV ⊤ O ∈ R N ×dim (18)
(reshaped row by row as a n-dimensional vector with n = N × dim to fit our generic notations), and where:
softmax(A) i = exp(A i ) N k=1 exp(A ik )
, and with Q, K, V, O ∈ R d1×dim . We assume that all the columns of O ⊤ V are non equal to zero. We consider Θ att the set of all parameters that satisfy this condition.
We define ϕ(θ) = (ϕ 1 , ϕ 2 ) with ϕ 1 = Q ⊤ K and ϕ 2 = V ⊤ O the reparametrization such that (up to flattening of matrices into vectors) g(θ, x) = f (ϕ, x) = softmax(Xϕ 1 X ⊤ )Xϕ 2 . The following theorem (see Appendix F for a proof) demonstrates that the parameterization ϕ is, in a some sense, sufficiently rich and allows for reduction to the study of a Lie algebra generated by the vector fields (θ → ∂ϕ(θ) ⊤ e k ) k . Theorem 3.8. Under Assumption 2.4, if V ℓ = R n and N ≥ 2 then W f,ℓ ϕ(θ) = R d , and W g,ℓ θ = range{∂ϕ(θ) ⊤ }, ∀θ ∈ Θ att .
Thanks to this theorem, the analysis boils down to a similar problem as for Proposition 3.5 and allows us to determine all conservation laws. See Appendix G for a proof. Corollary 3.9. Under the assumptions of Theorem 3.8 for each θ ∈ Θ att such that both horizontally concatenated matrices Q, K and V, O have full rank, there is a neighborhood of θ in which all conservation laws for (18) are functions of QQ ⊤ -KK ⊤ , and V V ⊤ -OO ⊤ and viceversa.
For more than one head, the neural network writes
g(θ, X) = H h=1 softmax(XQ ⊤ h K h X ⊤ )XV ⊤ h O h , (19
)
up to matrix flattening, with
Q h , K h , V h , O h ∈ R d 1 H ×dim .
In the case of multi-head attention, we can partially extend our results to obtain a set of conserved quantities (see Corollary 3.10, with proof detailed in Appendix G). However, determining whether this set of conservation laws is complete remains an open problem. Corollary 3.10. For any h = 1, • • • , H, the functions
Q h Q ⊤ h -K h K ⊤ h , V h V ⊤ h -O h O ⊤ h ,
define conservation laws for (19) with respect to any loss ℓ.
this section cite: []

Section: Cross-Entropy Classification Layer
For classification tasks (which we consider in the numerical part Section 5.2), the final layer typically combines a linear transformation with the evaluation of a cross-entropy. This corresponds to using a Kullback-Leibler loss over n classes ℓ(y, y ′ ) := n i=1 y i log (y i /y ′ i ) -y i + y ′ i , together with a soft-max layer g(θ, •) where θ ∈ R n×m : g(θ, x) := softmax(θx), softmax(z) i := e zi j e zj . (20) Note that this loss satisfies Assumption 2.4 and V ℓ = R n . The following proposition (proved in Appendix H) shows that the softmax layer induces a new set of conservation laws with respect to this loss.
Proposition 3.11. With respect to any loss ℓ such that V ℓ = R n , there are exactly m independent conservation laws for the classification layer given by (20): h j (θ) := i θ i,j , j = 1, . . . , m.
this section cite: []

Section: Deeper ResNets and Transformers
In this section, we examine conservation laws for deep networks g(θ, •) (denoted here as g θ (x)) composed of q residual blocks. Specifically, each block l consists of parameters θ l such that θ = (θ 1 , . . . , θ q ) and corresponds to an elementary network denoted g l θ l (x). Thus, the global network g θ can be written as the composition of q elementary networks:
g θ : x ∈ R m → g q θq • • • • • g 1 θ1 (x) ∈ R m .
In case of a classification task using a cross-entropy loss, we consider a last block constitued of a linear (fully-connected) layer and a softmax activation g q+1 θq+1 as in (20), so that θ := (θ 1 , • • • , θ q+1 ) and the global network then writes:
g θ : x ∈ R m → g q+1 θq+1 • g q θq • • • • • g 1 θ1 (x) ∈ R n . Example 4.1 (Convolutive ResNet). A convolutive ResNet corresponds to g θ : x ∈ R m → g θ (x) ∈ R n
with q residual blocks (we recall that here m = c 0 p). Each block l ≤ q has parameters (u l k,j , v l j,i ) and consists of a plain 2-layer convolutive ReLU network (16) with a skip connection:
g l θ l : x → x + U l σ V l x ,(21)
with matrices V l and U l made of circulant blocks respectively corresponding to convolutions with filters v l j,i , u l k,j . Example 4.2 (Transformer). We consider a transformer architecture where g θ : x ∈ R m → g θ (x) ∈ R n (here m = N × dim) consists of alternating residual blocks with either an attention layer with one head, or a 2-layer MLP. We notably omit normalization layers, and restrict to a single head. In that case, up to appropriate harmless matricizations or flattening operations associating the vector x to the token matrix X, each block l ≤ q writes either:
g l θ l : x → vec(X ⊤ + U l σ(U ′ l X ⊤ )),(22)
where θ l = (U l , U ′ l ) corresponds to the two weight matrices of a 2-layer MLP (15), or:
g θ l (x) = vec([X + softmax(XQ ⊤ l K l X ⊤ )XV ⊤ l O l ] ⊤ ),(23)
where θ l = (Q l , K l , V l , O l ) corresponds to the parameters of a single-head attention layer (18).
this section cite: []

Section: Characterization of "block" conservation laws
To analyze conservation laws in this context we focus on laws that depend only on a subset (or block) of parameters.
Considering θ ∈ Θ the parameters of a global network g θ , we focus on θ T a subset of the parameter entries (typically we will consider θ T = θ l for some l, but other scenarios will also be covered), so we can write θ = (θ T , θ T c ), where θ T c gathers the remaining entries of the parameters θ.
The following proposition (see Appendix I) characterizes smooth conservation laws of g that only depend on θ T . Proposition 4.3. Consider a function h ∈ C 1 (Θ, R) that only depends on the coordinates θ T , and for each θ ∈ Θ denote
Θ T c (θ T ) := {η ∈ R T c : (θ T , η) ∈ Θ} (24
)
Consider a loss that satisfies the assumptions of Proposition 2.3 as well as Assumption 2.4. The function h is a conservation law of g with respect to ℓ if, and only if, for every θ ∈ Θ one has
∇ θ T h(θ) ⊥ R θ T (W g,ℓ ), where: R θ T (W g,ℓ ) := span η∈Θ T c (θ T ) w∈V ℓ span x∈X (θ T ,η) {∂ θ T g((θ T , η), x) ⊤ w}.(25)
For concrete examples below, a technical challenge that arises when studying conservation laws that depend solely on θ T , and in comparing them with those of "internal" shallow networks involving only θ T , is the analysis of the set
X θ of input x of the global network g θ around which it is smooth enough (cf the definition of X θ in Proposition 2.
3), rather than the set of inputs of the considered "internal" shallow network. Overall, the important property for our purposes is the density of this set, proved in Appendix K. For every θ ∈ Θ we have X θ = R m . Remark 4.5. Obviously Item 2 only excludes a lowerdimensional set of parameters. We discuss in Appendix L why Item 1 is also a generic condition on the parameters for the example of a residual block associated with a 2-layer ReLU network.
Lemma 4.4. Denote Θ = Θ q × . . . × Θ 1 (or Θ = Θ q+1 × Θ q × . . . × Θ 1 with Θ q+1 = R n×m when there is a last softmax layer) where for each layer 1 ≤ l ≤ q, Θ l is the set of parameters θ l such that 1. g l θ l is an open map 3 ; 2. all the rows of the matrix V l (resp. U ′ l ) from (21) (resp. (22
this section cite: []

Section: Block laws for natural residual blocks
Consider θ T := θ l where l ∈ {1, • • • , q}. The following theorem (See Appendix N for a proof) demonstrates that the conservation laws of the global network g that depend exclusively on θ l are precisely those of the shallow network g l θ l of the l-th residual block. Theorem 4.6. With Θ as in Lemma 4.4, consider the lth residual block of Example 4.1 (resp. Example 4.2), and denote θ T := θ l and θ T c the parameters of all other residual blocks. A function H : θ = (θ T , θ T c ) ∈ Θ → h(θ T ) that only depends on θ T is a conservation law of g with respect to a loss ℓ such that V ℓ = R n if and only if h is a conservation law of the shallow residual network g l (θ l , x) := g l θ l (x) with respect to the Euclidean loss. The same result holds for θ T := θ q+1 when considering a last block (20).
Thus by using Proposition 3.2, the conservation laws of g that only depends on θ l are exactly the ones of (16) (resp. ( 18)), which are described in Theorem 3.6 (resp. Corollary 3.9) for a residual block defined with a 2-layer ReLU networks (resp. with an attention layer).
this section cite: []

Section: Case of blocks overlapping a residual connection
This section focuses exclusively on the case of ResNet (with or without) convolutions as defined in Example 4.1. In the previous section Section 4.2, we examined the conservation laws of the global network g that depend only on the parameters θ l of the l-th block, and we just show that they exactly correspond to the ones of the shallow network g l θ l corresponding to the l-th block. However, it is also possible to recast the global network as a composition of elementary networks that maintain parameter separation while involving a subset of parameters located before and after a residual connection as described in Figure 2 in Appendix O.
Specifically, let us consider l ∈ {1, • • • , q -1} and θ T = (v l+1 j,i , u l k,j ′ ). In that case, θ T corresponds to two consecutive parameter blocks that overlap a skip connection (as described in Figure 2 in Appendix O). Denoting x 1 (resp. y 1 ) the input of the intermediate layer of the l-th (resp. l + 1-th) block, x 2 (resp. y 2 ) the copy of the input of the corresponding block that is transferred via the skip connection, g can be written as a composition of g 1 , g 2 and g 3 with
( y1 y2 ) = g 2 θ2 , ( x1 x2 ) := V l+1 I d ( U l I d ) ( σ id ) ( x1 x2 ) , ( x1 x2 ) = g 3 ( θ3 , x) := V l I d (g l-1 θ l-1 • . . . • g 1 θ1 )(x),
and
g 1 θ1 , ( y1 y2 ) = g q θq • . . . • g l+2 θ l+2 (( U l+1 I d ) ( σ id ) ( y1 y2 ))
where θ1 , θ3 gather all parameters appearing in g 1 and g 3 .
The following proposition (See Appendix P for a proof) shows that there exist no conserved functions of the global network that depend exclusively on θ2 = θ T = (V l+1 , U l ).
Theorem 4.7. Consider a layer index 1 ≤ l ≤ q -1 and Θ defined as in Lemma 4.4 with the exception that for each θ l+1 ∈ Θ l+1 , we further require that the rows of V l+1 are pairwise non-colinear. If n v = p then any conservation law of g with respect to the Euclidean loss that only depends on θ2 is a constant function.
this section cite: []

Section: Numerical confirmation
In the general case of a residual network composed of q blocks, there could potentially exist conservation laws that depend on a larger subset of parameters than those previously analyzed, which helped us demonstrate that we recover exactly the same conservation laws as those of elementary blocks when considering the global network. However, by numerically computing the dimension of span{L Z (θ) : Z} ⊆ W g θ with a sufficiently large batch size to adequately explore the space, we find that there is no additional conservation law when m > 1 for a ResNet with q = 2 residual blocks (see code in our GitHub repository).
When m = 1, numerical results suggest that there are additional conservation laws. It might be that this specific case enables as in (Marcotte et al., 2024, Theorem 4.1) to shed the light on new invariances that give rise to new conservation laws. This is confirmed by the following example: we exhibit a domain Ω where there are more conservation laws than the "block" ones.
Example 4.8. Consider a ReLU neural network with tworesidual blocks, g((u, v, s, t), x) = x + uσ(vx) + sσ(t(x + uσ(vx)) with (u, v, s, t) ∈ Ω ⊆ R 4 and x ∈ R. While there are two "block" conservation laws: u 2 -v 2 and s 2 -t 2 , there are three conservation laws on the set Ω of all parameters such that sgn(t) = sgn(v) = sgn(u). Indeed, for every x,
1. either vx, tx ≤ 0, hence g(θ, x) = x and ∇ θ g(•) = 0; 2. or vx, tx ≥ 0, hence g(θ, x) = x+uvx+stx+stuvx
as vx, tx, tu ≥ 0, and thus
1 x ∇ θ g(•) =: χ 1 (•) is a vector field that does not depend on x.
As the space W g,ℓ = Rχ 1 is spanned by a single non-null vector field, its Lie algebra is itself, and by Theorem 2.11, there are exactly 4 -1 = 3 conservation laws as claimed.
this section cite: []

Section: Conservation laws for discrete dynamics
In practice optimization is performed with a discrete dynamics associated to stochastic gradient descent. To what extent do conservation laws still apply in this context? This is the object of this section.
this section cite: []

Section: (Stochastic) gradient descent as a training dynamic
We consider the ERM problem (1). Instead of using gradient descent (2), we consider the stochastic gradient descent (SGD) method on mini-batches:
θ k+1 = θ k -τ k ∇L Z k (θ k ),
where the sequence of mini-batches (Z k ) k is drawn i.i.d. from the data distribution. The following proposition (see Appendix Q for a proof) shows that a conservation law is approximately preserved by SGD. Proposition 5.1. Let h(θ) be a conservation law of the gradient flow with a bounded Hessian ∀θ, ∥∂ 2 h(θ)∥ ≤ C h . Suppose that the gradients remain bounded in expectation throughout the algorithm:
E Z k ,θ k ∇L Z k (θ k ) 2 2 ≤ C L .
(26)
Then, we have E h(θ k ) -h(θ 0 ) ≤ C h C L 2 k-1 i=0 τ 2 i . (27
)
For a constant step size τ k = τ , this proposition shows that the conservation error grows as |h(θ k ) -h(θ 0 )| = O(τ 2 k), which increases linearly with the number of iterations. If one uses a decaying step size, such as τ k = τ 0 /(k + 1), which ensures convergence of the method, then the error remains bounded: |h(θ k )-h(θ 0 )| = O(τ 2 0 ). The requirement that h has a bounded Hessian holds in the case considered in this paper since the conservation laws we examine are quadratic. The key assumption for the result to hold is the bound on the gradient magnitude in (26). This condition is met if the loss function ℓ and the network g θ are uniformly Lipschitz, though this is not generally true for deep networks. It also holds with an explicit constant C L for smooth convex losses (Bach, 2024), but this setting is quite restrictive. More generally, such bounds hold for smooth losses when the variance of ∇L Z k (θ) is bounded (Garrigos & Gower, 2023), because the iterates are bounded in expectations, E(∥θ k ∥ 2 ) < +∞, though the constant C L may not be explicitly determined. In the numerical experiments presented in Section 5.2, we empirically evaluate the constants to show that, in practice, they remain relatively small, ensuring approximate conservation.
this section cite: ['b2', 'b12']

Section: Numerical experiments
In Figure 1, we train a ResNet-18 on CIFAR-10 ( Krizhevsky, 2009) while tracking the difference between the squared Frobenius norms of consecutive convolutional layers in the first residual block, considering the conservation law h(θ T ) := c1 j=1 h j (θ T ), where h j is defined in (17), with θ T representing the parameters of the first residual block. We vary the learning rate between 10 -3 and 5 × 10 -3 , using stochastic gradient descent (SGD) without momentum or weight decay. For each learning rate, we train 10 models for 50 steps with 10 different random seeds, recording both the loss evolution (bottom) and the evolution of the con- lines show the theoretical slopes Cτ 2 derived from ( 27), confirming that the function is approximately conserved and that the slope coefficient maintains proportionality with τ 2 . Our code is available at our GitHub repository.
servation error |(h(θ k ) -h(θ 0 ))/h(θ 0 )| (top). The dotted
In another experiment, we train a transformer model on the IMDb sentiment analysis dataset (Maas et al., 2011) using SGD optimization. We track the evolution of the Frobenius norm of the conserved matrix identified in Corollary 3.10, specifically examining the query and key matrices from the first attention head in the first layer. Consistent with our ResNet training results presented in Figure 1, we observe that the conservation error scales as O(step-size 2 ) throughout training, which confirms our theoretical bound (27). Furthermore, it is worth noting that the numerical behavior is unchanged whether masking is applied or not. See Appendix O for the associated figures. Our code is available at our GitHub repository
this section cite: ['b18', 'b19']

Section: Conclusion
This paper investigates conservation laws in deep networks (ResNet and Transformer architectures) within gradient flow dynamics and examines their behavior under discrete SGD dynamics. Our analysis does not currently account for transformer normalization layers, and the multi-head attention mechanism is only partially addressed. The integration of these components as well as max-pooling layers presents promising avenues for future research. SHARP ANR Project ANR-23-PEIA-0008 of the PEPR IA, funded in the framework of the France 2030 program.
this section cite: []

Section: References
Ref_id:b0 Title: On the optimization of deep networks: Implicit acceleration by overparameterization Year: (2018)
Ref_id:b1 Title: A convergence analysis of gradient descent for deep linear neural networks Year: (2019)
Ref_id:b2 Title: Learning theory from first principles Year: (2024)
Ref_id:b3 Title: Learning deep linear neural networks: Riemannian gradient flows and convergence to global minimizers Year: (2022)
Ref_id:b4 Title: Neural machine translation by jointly learning to align and translate Year: (2015)
Ref_id:b5 Title: Dissecting adam: The sign, magnitude and variance of stochastic gradients Year: (2018)
Ref_id:b6 Title: Convergence and dynamical behavior of the adam algorithm for nonconvex stochastic optimization Year: (2021)
Ref_id:b7 Title: Language models are few-shot learners Year: (2020)
Ref_id:b8 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b9 Title: Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss Year: (2020)
Ref_id:b10 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b11 Title: Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced Year: (2018)
Ref_id:b12 Title: Handbook of convergence theorems for (stochastic) gradient methods Year: (2023)
Ref_id:b13 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b14 Title: Nonlinear system control Year: (1995)
Ref_id:b15 Title: Gradient descent aligns the layers of deep linear networks Year: (2019)
Ref_id:b16 Title: The implicit bias of gradient descent on nonseparable data Year: (2019)
Ref_id:b17 Title: A method for stochastic optimization Year: (2014)
Ref_id:b18 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b19 Title: Learning word vectors for sentiment analysis Year: (2011-06)
Ref_id:b20 Title: Abide by the law and follow the flow: Conservation laws for gradient flows Year: (2023)
Ref_id:b21 Title: Keep the momentum: Conservation laws beyond euclidean gradient flows Year: (2024)
Ref_id:b22 Title: Implicit regularization of deep residual networks towards neural odes Year: (2023)
Ref_id:b23 Title: On the explicit role of initialization on the convergence and implicit bias of overparametrized linear networks Year: (2021)
Ref_id:b24 Title: Functional dependence Year: (1967)
Ref_id:b25 Title: Weight-balancing fixes and flows for deep learning Year: (2023)
Ref_id:b26 Title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks Year: (2013)
Ref_id:b27 Title: The implicit bias of gradient descent on separable data Year: (2018)
Ref_id:b28 Title: Equi-normalization of neural networks Year: (2019)
Ref_id:b29 Title: Understanding the dynamics of gradient flow in overparameterized linear models Year: (2021)
Ref_id:b30 Title: Implicit bias and fast convergence rates for self-attention Year: (2024)
Ref_id:b31 Title: Attention is all you need Year: (2017)
Ref_id:b32 Title: A better way to decay: Proximal gradient training algorithms for neural nets Year: (2022)
Ref_id:b33 Title: Making convolutional networks shift-invariant again Year: (2019)
