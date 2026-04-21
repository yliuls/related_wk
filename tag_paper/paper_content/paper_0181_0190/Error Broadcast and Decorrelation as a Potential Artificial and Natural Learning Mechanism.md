Title: Error Broadcast and Decorrelation as a Potential Artificial and Natural Learning Mechanism
Abstract: We introduce Error Broadcast and Decorrelation (EBD), a novel learning framework for neural networks that addresses credit assignment by directly broadcasting output errors to individual layers, circumventing weight transport of backpropagation. EBD is rigorously grounded in the stochastic orthogonality property of Minimum Mean Square Error estimators. This fundamental principle states that the error of an optimal estimator is orthogonal to functions of the input. Guided by this insight, EBD defines layerwise loss functions that directly penalize correlations between layer activations and output errors, thereby establishing a principled foundation for error broadcasting. This theoretically sound mechanism naturally leads to the experimentally observed three-factor learning rule and integrates with biologically plausible frameworks to enhance performance and plausibility. Numerical experiments demonstrate EBD's competitive or better performance against other error-broadcast methods on benchmark datasets. Our findings establish EBD as an efficient, biologically plausible, and principled alternative for neural network training. The implementation is available at: https://github.com/meterdogan07/error-broadcast-decorrelation.

Section: Introduction
Neural networks are dominant mathematical models for biological and artificial intelligence. A major challenge in these networks is determining how to adjust individual synaptic weights to optimize a global learning objective, known as the credit assignment problem. In Artificial Neural Networks (ANNs), the most common solution is the backpropagation (BP) algorithm [1].
In contrast to ANNs, the mechanisms for credit assignment in biological neural networks remain poorly understood. While backpropagation is highly effective for training ANNs, it is not directly applicable to biological systems because it relies on biologically implausible assumptions. In its standard form, backpropagation propagates output errors backward through a separate pathway, reusing the same synaptic weights as in the forward pass (Figure 1a). This requirement for weight symmetry is not supported by biological evidence [2]. Although many experimentally motivated models of local synaptic plasticity have been proposed [3], a biologically feasible theory of credit assignment that integrates these mechanisms remains unresolved.
To address the credit assignment problem in biological networks, researchers have proposed methods known as error broadcasting [4][5][6][7][8][9]. These methods involve broadcasting the global output error directly to all layers, often through random projections or fixed pathways, without relying on precise backward paths or symmetric weights (as summarized in Section 1.1). This eliminates the weight symmetry issue inherent in backpropagation. Error broadcasting offers practical benefits for hardware implementation; recent work [10] demonstrates potential for efficient neural network execution. However, despite encouraging progress in both theory and application [11,12], error broadcasting still needs stronger theoretical foundations to fully validate and enhance its training effectiveness.
In this context, we introduce a novel learning framework termed the Error Broadcast and Decorrelation (EBD), which builds on basic error broadcasting by introducing layer-specific objectives grounded in estimation theory. The fundamental principle of EBD is to adjust network weights to minimize the correlation between broadcast output errors and the activations of each layer. This approach is rigorously grounded in Minimum Mean Square Error (MMSE) estimation, where an optimal estimator's error is orthogonal to any measurable function of its input. We leverage this orthogonality principle for EBD, defining layer-specific training losses to drive layer activations (functions of the network input) towards orthogonality with the broadcast error. This enables a more distributed mechanism for credit assignment, alternative to approaches relying solely on an output-defined loss and end-to-end error propagation.
(a) BP (b) EBD (c) Correlation between layer activations and output error. EBD directly broadcasts output errors to layers, simplifying credit assignment and enabling parallel synaptic updates. It offers two key advantages for biologically realistic networks. First, optimizing EBD's loss naturally leads to experimentally observed three-factor learning rules [13,14], which extend Hebbian plasticity by incorporating a neuromodulatory signal (the third factor) modulating synaptic updates based on pre-and postsynaptic activity. Second, by broadcasting errors directly to layers as shown in Figure 1b, it overcomes the weight transport problem inherent in backpropagation and some more biologically plausible credit assignment approaches [15,16].
We demonstrate EBD's utility by applying it to both artificial and biologically realistic neural networks. Benchmark results show EBD matching/exceeding state-of-the-art error-broadcast techniques. Its successful application to a 10-layer biologically plausible network (CorInfoMax-EBD) provides initial evidence of depth scalability for more complex tasks.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15']

Section: Related work and contributions
Several frameworks have been proposed as alternatives to the backpropagation algorithm for modeling credit assignment in biological networks [8]. These include predictive coding [15,17,18], similarity matching [16,19], time-contrastive approaches [20][21][22], forward-only methods [23][24][25], target propagation [26][27][28], random feedback alignment [29], and learned feedback weights [30,31]. Alternative strategies also seek to establish local learning rules by optimizing statistical objectives, such as the Hilbert-Schmidt Independence Criterion (HSIC) bottleneck [32].
Another significant alternative is error-broadcast methods, where output errors are directly transmitted to network layers without relying on precise backward pathways or symmetric weights. Two important examples of this approach are weight and node perturbation algorithms [4,[33][34][35], in which global error signals are broadcast to all network units. These signals reflect the change in overall error caused by individual perturbations in the network's weights or units. A more recent and prominent example of error broadcast is Direct Feedback Alignment (DFA) [6]. In DFA, the output errors are projected onto the hidden layers through fixed random weights, effectively replacing the symmetric backward weights required in traditional backpropagation. The core challenge of weight transport has been tackled by several other methods, many of which also rely on fixed random signals or avoid feedback entirely [36,37]. Encouragingly, a number of these biologically-plausible frameworks have demonstrated the ability to scale effectively to large datasets, underscoring their potential as viable training mechanisms [38]. This approach first emerged as a modification to the feedback alignment approach (which replaced the symmetric weights of the backpropagation algorithm with random ones). DFA has been extended and analyzed in several studies [11,12,[39][40][41], demonstrating its potential in training neural networks with less biologically implausible mechanisms. Clark et al. [9] introduced another broadcast approach for a network with vector units and nonnegative weights for which three factor learning based update rule is applied.
Our framework for error broadcasting differentiates itself through
• a principled method based on the orthogonality property of nonlinear MMSE estimators,
• error projection weights determined by the cross-correlation between the output errors and the layer activations as opposed to random weights of DFA, • dynamic Hebbian updating of projection weights as opposed to fixed weights of DFA, • updates involving arbitrary nonlinear functions of layer activities, encompassing a family of three-factor learning rules, • the option to project layer activities forward to the output layer.
In summary, our approach provides a theoretical grounding for the error broadcasting mechanism and suggests ways to enhance its effectiveness in training networks.
this section cite: ['b7', 'b14', 'b16', 'b17', 'b15', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b3', 'b32', 'b33', 'b34', 'b5', 'b35', 'b36', 'b37', 'b10', 'b11', 'b38', 'b39', 'b40', 'b8']

Section: Error Broadcast and Decorrelation method

this section cite: []

Section: Problem statement
To illustrate our approach, we first assume a multi-layer perceptron (MLP) network with L layers. We label the input x = h (0) ∈ R N (0) and layer activations h (k) ∈ R N (k) for k = 1, . . . , L, where N (k) is layer size. The layer activations are:
h (k) = f (k) (u (k) ), u (k) = W (k) h (k-1) + b (k) ,(1)
where k ∈ {1, . . . L} is the layer index, f (k) are activation functions, W (k) weights, u (k) preactivations and b (k) biases. We consider input-output pairs (x, y) sampled from a joint distribution P (x, y).
The performance criterion is the mean square of the output error ϵ = h (L) -y, i.e., E P (x,y) [∥ϵ∥ 2  2 ].
this section cite: []

Section: Error Broadcast and Decorrelation loss functions
To guide the training of our neural network (which aims to minimize this MSE), we draw inspiration from the fundamental principles of Minimum Mean Square Error (MMSE) estimation theory [42]. This theory defines an ideal estimator, denoted ŷ * (x), which achieves the absolute minimum possible MSE for a given joint data distribution P (x, y). A crucial characteristic of this optimal estimator is its stochastic orthogonality property, which forms the theoretical cornerstone of our EBD approach.
Formally, considering input-output pairs (x, y) drawn from P (x, y), this optimal nonlinear MMSE estimator is given by ŷ * (x) = E[y|x] (its derivation and properties as the optimal MSE-minimizing function are detailed in Appendix A, Lemma A.1). Its estimation error ϵ * = yŷ * (x) satisfies:
E[g(x)ϵ T * ] = 0,(2)
for any properly measurable function g(x) of the input x (see Appendix A, Lemma A.2). This means ϵ * is orthogonal to g(x) (i.e., their expected outer product is zero). While this orthogonality property, stated in Eq. ( 2), is foundational, its application in constructing estimators has predominantly been in linear MMSE estimation. In that context, the estimator ŷ(x) is constrained to be a linear function of x, and under the linearity constraint on the estimator, Eq. ( 2) is restricted to a form where g(x) = x. This restricted orthogonality condition has long been used to derive parameters for linear estimators, such as Wiener-Kolmogorov and Kalman filters [43].
A key aspect of our work is to leverage the full generality of the orthogonality condition Eq. ( 2) for obtaining nonlinear estimators. For such estimators, this condition holds for any measurable function g(x) and, crucially, is not only necessary but also sufficient for MMSE optimality (as established in Appendix B, Theorem B.1). EBD distinctively employs this sufficiency as a constructive principle to train nonlinear estimators, specifically the neural network parameters.
We model the neural network (Eq. ( 1)) as a parameterized nonlinear estimator f Θ (x) = h (L) (x; Θ) and aim to satisfy Eq. ( 2) for its output error ϵ = f Θ (x) -y. We choose g(x) as the network's hidden layer activations h (k) (x; Θ (k) ), where Θ (k) = (W (k) , b (k) ). This choice is motivated because:
(i). Since each hidden-layer activation is a nonlinear function of input x, the output error of an optimal estimator should be stochastically orthogonal to those activations. Figure 1c illustrates this phenomenon by showing the evolution of the average absolute correlation between layer activations and the error signal during backpropagation training of an MLP with three hidden layers on the CIFAR-10 dataset, based on the MSE criterion. Similar correlation declining trends are also observed across different datasets and architectures (see Appendix J). The declining correlation during MSE training reflects the MMSE estimator's stochastic orthogonality of layer activations and output errors., (ii). h (k) depends on layer parameters Θ (k) , enabling their direct updates via differentiation, (iii). if hidden-layer activations form a "rich enough" set of functions of x (as elaborated below), then enforcing error orthogonality to them implies orthogonality to "every" function of x.
Indeed, in Theorem B.2 of Appendix B, we show that when the hidden-layer activations-say, those in the first layer-form a sufficiently rich basis (for example, becoming dense in L 2 (P x ) as network width tends to infinity), enforcing that the output error ϵ be orthogonal to these activations naturally drives the estimator toward the true MMSE solution. Accordingly, we aim to enforce zero correlation between the output error ϵ and hidden layer activations, or more generally their nonlinear functions:
R (k) gϵ = E[g (k) (h (k) )ϵ T ] = 0, k = 0, . . . , L, with the typical choice g (k) (h (k) ) = h (k) .(3)
Building on the orthogonality property and its established sufficiency for optimality (Appendix B, Theorem B.1), we define layer-specific surrogate loss functions that enforce orthogonality conditions with respect to the hidden layer activations. As demonstrated in Section 2.3, these losses yield an alternative to backpropagation by broadcasting output errors directly to network nodes (Figure 1b).
Specifically, based on the stochastic orthogonality condition in Eq. ( 3), we propose minimizing the Frobenius norm of the cross-correlation matrices R (k) gϵ as a replacement for the standard MSE loss. To this end, we define the estimated cross-correlation matrix between a function g (k) of layer activations and the output error for batch m and layer k as
R(k) gϵ [m] = λ R(k) gϵ [m -1] + 1 -λ B G (k) [m]E[m] T ,
where m is the batch index, λ ∈ [0, 1] is the forgetting factor used in the autoregressive estimation, B is the batch size, R(k) gϵ [0] is the initial value hyperparameter for the correlation matrix, and
G (k) [m] = g (k) (h (k) [mB + 1]) . . . g (k) (h (k) [mB + B]) ,(4)
is the matrix of nonlinearly transformed activations of layer k for batch m. In the above equation, mB + l refers to absolute (sequence) index for the l th member of batch-m. Furthermore,
E[m] = [ ϵ[mB + 1] . . . ϵ[mB + B] ] ,(5)
is the output error matrix for batch m. We then define the layer-specific loss function based on the stochastic orthogonality condition for layer k as
J (k) (h (k) , ϵ)[m] = 1 2 R(k) gϵ [m] 2 F ,(6)
where ∥ • ∥ F denotes the Frobenius norm. This loss function captures the sum of the squared magnitudes of all cross-correlations between the components of the output error and the (potentially transformed) activations of layer k. Thus, we refer to the minimization of this loss as decorrelation.
this section cite: ['b41', 'b42']

Section: Error Broadcast and Decorrelation algorithm
The functions in Eq. ( 6) defines individual loss functions for each hidden layer, which are used to adjust the layer parameters. These loss functions can be minimized using a gradient based algorithm.
To minimize the loss for layer k, we compute the gradient of the loss function J (k) (h (k) , ϵ) with respect to the weight W (k) ij . The derivative can be decomposed into two terms:
∂J (k) (h (k) , ϵ) ∂W (k) ij [m] = ζT r R(k) gϵ [m]E[m] ∂G (k) [m] T ∂W (k) ij [∆W (k) 1 [m]]ij + ζT r R(k) gϵ [m] ∂E[m] ∂W (k) ij G (k) [m] T [∆W (k) 2 [m]]ij , where ζ := (1 -λ)/B.
Similarly, the derivative with respect to the bias b (k) i is given by:
∂J (k) (h (k) , ϵ) ∂b (k) i [m] = ζT r R(k) gϵ [m]E[m] ∂G (k) [m] T ∂b (k) i [∆b (k) 1 [m]]i + ζT r R(k) gϵ [m] ∂E[m] ∂b (k) i G (k) [m] T [∆b (k) 2 [m]]i .
Here
∆W (k) 1 , ∆b (k) 1 [m] (∆W (k) 2 , ∆b(k)
2 [m]) represent the components of the gradients containing derivatives of activations (output errors) with respect to the layer parameters. As derived in Appendix C.1, we obtain the closed-form expressions for ∆W (k) 1 [m] and ∆b
(k) 1 [m]: [∆W (k) 1 [m]] ij = ζ (m+1)B n=mB+1 ϑ (k) [n]h (k-1) j [n], [∆b (k) 1 [m]] i = ζ (m+1)B n=mB+1 ϑ (k) [n],(7)
where
ϑ (k) [n] = g ′ (k) i (h (k) i [n])f ′ (k) (u (k) i [n])q (k) i [n], g ′ (k) i
and f ′ (k) denote the derivatives of the nonlinearity g (k) and the activation function f (k) , respectively. The term q (k) [m] is defined as:
q (k) [m] = R(k) gϵ [m] ϵ[m]
, representing the projection of the output error onto the layer activations, with the cross-correlation matrix R(k) gϵ [m] as the transformation matrix. These projections are shown in Figure 1b. For the special case of batchsize, B = 1, the weight update in (7) simplifies to
[∆W (k) 1 [m]] ij = ζg ′ (k) i (h (k) i [m])f ′ (k) (u (k) i [m])q (k) i [m]h (k-1) j [m].(8)
The update terms ∆W (k) 1 [m] and ∆b (k) 1 [m] aim to adjust the activations to gradually become orthogonal to ϵ, as they are based on the derivatives of activations with respect to layer parameters. In contrast, ∆W [m], we eliminate the need for propagation terms, resulting in a completely localized update mechanism for training the neural network. This simplification to localized updates is supported by their positive alignment with backpropagation and full (untruncated) EBD gradient directions, as demonstrated in Appendix E.1 and E.2, respectively. Therefore, we prescribe the Error Broadcast and Decorrelation (EBD) update expressions as:
W (k) [m + 1] = W (k) [m] -µ (k) [m]∆W (k) 1 [m], b (k) [m + 1] = b (k) [m] -µ (k) [m]∆b (k) 1 [m],
for k = 1, . . . , L -1, where µ (k) [m] is the learning rate for layer k at batch m. Although these updates resemble backpropagation, a key difference lies in the error signals: the backpropagated error is replaced by the broadcasted error. Furthermore, the algorithm introduces flexibility by allowing the choice of nonlinearity functions g (k) , which influence the gradient terms ∆W (k) 1 [m] and ∆b
(k) 1 [m].
For the final layer (k = L), we utilize the standard MMSE gradient update:
W (L) [m + 1] = W (L) [m] - µ (L) [m] B (m+1)B n=mB+1 f ′ (k) (u (L) [n]) ⊙ ϵ[n] h (L-1) [n] T , b (L) [m + 1] = b (L) [m] - µ (L) [m] B (m+1)B n=mB+1 f ′ (k) (u (L) [n]) ⊙ ϵ[n],
where f ′ (L) is the derivative of the activation function of the output layer.
this section cite: []

Section: Further EBD algorithm extensions
We propose further extensions to the EBD framework to address potential activation collapse, which can arise when minimizing correlations is the sole objective. To prevent unit-level collapse, we introduce power regularization, while entropy regularization is employed to prevent dimensional collapse. Both regularizations can be implemented in ANNs as well as biologically plausible networks. The biological plausibility of employing these regularizers in MLP-based EBD is discussed in Appendix H.1. Although CorInfoMax-EBD inherently includes entropy regularization, it can also benefit from the addition of power regularization for enhanced stability. Additionally, we introduce forward layer activation projections to improve the algorithm's versatility. We also extend the EBD formulations to more complex architectures, including Convolutional Neural Networks (CNNs) and Locally Connected (LC) networks. For further details on these extensions, please refer to Appendix D.
this section cite: []

Section: Avoiding collapse
A critical challenge with EBD is potential activation collapse, where decorrelation losses (Eq. ( 6)) are minimized by driving activations h (k) → 0, even with non-zero output errors, undermining learning.
To counteract this, we introduce two complementary regularizers:
Power normalization: To prevent total activation collapse, power normalization (Eq. ( 9)) regulates layer activation power around a target level P (k) .
J (k) P (h (k) )[m] = N (k) l=1 (B -1 (m+1)B n=mB+1 h (k) l [n] 2 -P (k) ) 2 ,(9)
which simplifies to
J (k) P (h (k) )[m] = N (k) l=1 (h (k) l [n] 2 -P (k) ) 2 ,(10)
for B = 1.
Layer entropy: To mitigate collapse into low-dimensional subspaces, which restricts expressiveness, we incorporate layer entropy (Eq. ( 11)), building on prior work [44,45].
J (k) E (h (k) )[m] = 1 2 log det(R (k) h [m] + ε (k) I).(11)
Here, R
h [m] is the layer autocorrelation matrix, updated autoregressively with forgetting factor λ E :
R (k) h [m] = λ E R (k) h [m -1] + (1 -λ E ) 1 B H (k) [m]H (k) [m] T ,(12)
where
H (k) [m] = [ h (k) [mB + 1] . . . h (k) [(m + 1)B] ],(13)
is the activation matrix. Gradient derivations for these objectives are in Appendices C.3 and C.4.
this section cite: ['b43', 'b44']

Section: Forward broadcast
In the EBD algorithm (Section 2.3), output errors are broadcast to layers to adjust weights and reduce correlations with activations. To complement this, we introduce forward broadcasting, projecting hidden layer activations onto the output layer to optimize the decorrelation loss by adjusting the final layer's parameters. Details are provided in Appendix C.5.
this section cite: []

Section: Extensions to other network architectures
The EBD approach is independent of network topology. We extend EBD to convolutional neural networks (CNNs) in Appendix D.1 and to locally connected (LC) networks in Appendix D.2.
this section cite: []

Section: EBD for biologically realistic networks
h (k) i h (k-1) j ϵ q (k) i ΔW (k) ij ∝ h (k-1) j ⋅ q (k) i ⋅ g′ i (k) (h (k) i )f′ (k) (u (k) i ) Pre-synaptic Post-synaptic Output Error Synapse Figure 2:
Error-broadcast learning as a three-factor synaptic update.
The presynaptic firing rate h (k-1) j (green, left) projects onto the postsynaptic neuron h In the previous section, we introduced the EBD algorithm within the context of MLP networks. While MLPs can resemble biologically plausible networks depending on the credit assignment mechanism, in this section, we extend the application of the EBD approach to neural networks that exhibit more biologically realistic dynamics and architectures, motivated by its inherent solutions to key neuroscientific challenges: 1) EBD's direct error broadcast naturally resolves the problematic weight symmetry requirement of BP, and 2) its update rules intrinsically manifest as modulated, extended Hebbian mechanisms (three-factor learning), aligning with current understanding of synaptic plasticity. In the following subsections, we explore how EBD relates to the biologically plausible three-factor learning rule and demonstrate its integration with the biologically more realistic CorInfoMax networks [45].
this section cite: ['b44']

Section: Three factor learning rule and EBD
The three-factor learning rule for biological neural networks extends the traditional two-factor Hebbian rule by incorporating a modulatory signal into synaptic updates based on presynaptic and postsynaptic activity [13,46]. While backpropagation can be expressed similarly, it is not typically considered a three-factor rule in neuroscience, as its 'third factor' is a locally tailored signal specific to each neuron requiring a biologically implausible dual network with symmetric weights, unlike global neuromodulatory signals [13]. In contrast, EBD update for batchsize B = 1 in (8) naturally matches the three-factor structure:
∆W (k) ij ∝ g ′ (k) i (h (k) i )f ′ (k) (u (k) i ) Postsynaptic q (k) i Modulatory h (k-1) j Presynaptic ,
where
q (k) i
is the projected global error. Thus, EBD supports various three-factor rules depending on nonlinearity g (k) . For example, g
i (h
(k) i ) = h (k) i 2
yields the error-modulated Hebbian update [11,47]:
∆W (k) ij ∝ h (k) i f ′ (k) (u (k) i ) q (k) i h (k-1) j
. By enabling diverse three-factor updates via different nonlinear functions, EBD holds potential for modeling biologically consistent neural learning processes.
Figure 2 breaks the EBD weight update (8) into its three interacting factors. The presynaptic activity h (k-1) j from the sending unit; the postsynaptic term is g ′(k) i f ′(k) computed from the receiving unit's own activation; and the modulatory broadcast error q (k) i , derived from the network's output error ϵ. Multiplying these three quantities produces the weight change ∆W (k) ij shown beneath the diagram, revealing that EBD naturally realises the classical three-factor learning rule in neural networks.
this section cite: ['b12', 'b45', 'b12', 'b10', 'b46']

Section: CorInfoMax-EBD: CorInfoMax with three factor learning rule
One of the significant advantages of the EBD framework is its flexibility to broadcast output errors into network nodes, which can be leveraged to transform time-contrastive, biologically plausible approaches into non-contrastive forms. To illustrate this property, we propose a modification of the recently introduced CorInfoMax framework [45] (see Appendix F for a summary). The CorInfoMax approach uses correlative information flow between layers as its objective function:
J CI [m] = L-1 k=1 ( Î(ε k ) r (h (k-1) , h (k) )[m] + Î(ε k ) l (h (k) , h (k+1) )[m]
), where,
Î(ε k ) r (h (k) , h (k+1) )[m] = 0.5 log det( Rh (k+1) [m] + ε k I) -0.5 log det( R→ e (k+1) [m] + ε k I), Î(ε k ) l (h (k) , h (k+1) )[m] = 0.5 log det( Rh (k) [m] + ε k I) -0.5 log det( R← e(
k) [m] + ε k I), are alternative forms of correlative mutual information between nodes, defined in terms of the correlation matrices of layer activations, i.e., Rh (k) and forward and backward prediction errors ( R→ e (k+1) and R← e (k) ). Here, forward/backward prediction errors are defined by → e (k+1) [n] = h (k+1) [n] -W (f,k) [m]h (k) [n], ← e (k)
[n] = h (k) [n] -W (b,k) [m]h (k+1) [n], respectively. Here, W (f,k) [m] (W (b,k) [m]) is the forward (backward) prediction matrix for layer k.
This objective leads to network dynamics corresponding to a structure with feedforward and feedback prediction weights, and lateral connections B (k) that maximize layer entropy. In the original work [45], the two-phase EP approach [22] is proposed to train the network weights. As an alternative, we propose employing the EBD update rule to replace the two-phase EP adaptation. The proposed CorInfoMax-EBD algorithm is described by the following update equations defined in Algorithm 1.
this section cite: ['b44', 'b44', 'b21']

Section: Algorithm 1 CorInfoMax-EBD Algorithm for Updating Weights in Layer k
Input: Batch size B, layer index k, iteration step m, learning rates µ (f,k) , µ (b,k) ,
µ (d f ,k) , µ (d b ,k) , µ (d l ,k) , factors λ d , λE, γE, activations H (k) in
Eq. (13), the nonlinear function of layer activations G (k) in Eq. (4) and their derivatives G (k) d in Eq. (28), the derivative of activations F (k) d in Eq. (29), output error E in Eq. (5), prediction errors ← E (k) and → E (k) in Eq. ( 54)-(55), lateral outputs Z (k) in Eq. (56).
Output: Updated weights W (f,k) , W (b,k) , B (k) . Step 1: Update error projection weights: R(k) gϵ [m] = λ d R(k) gϵ [m -1] + 1-λ d B G (k) [m]E[m] T Step 2: Project errors to layer k: Q (k) [m] = R(k) gϵ [m]E[m]
Step 3: Find the gradient of the nonlinear function of activations for layer k:
Φ (k) [m] = F (k) d [m] ⊙ Q (k) [m] ⊙ G (k) d [m]
Step 4: Update forward, backward and lateral weights for layer k:  E), output error E, and lateral weight outputs Z are computed via CorInfoMax network dynamics in [45] (see Appendix F). Integrating EBD enables single-phase updates per input, eliminating the less biologically plausible two-phase mechanism required by CorInfoMax-EP. EP's two-phase approach-separate label-free and label-connected phases-is implausible, as biological neurons unlikely alternate between distinct global phases for learning. Our method simplifies the process, aligns more closely with biological learning, and achieves comparable or superior performance to CorInfoMax-EP (see Section 4).
W (f,k) [m] = W (f,k) [m -1] + B -1 µ (f,k) [m] → E (k) [m] -B -1 µ (d f ,k) [m]Φ (k) [m] H (k-1) [m] T W (b,k) [m] = W (b,k) [m -1] + B -1 µ (b,k) [m] ← E (k) [m] -B -1 µ (d b ,k) [m]Φ (k) [m] H (k+1) [m] T B (k) [m] = λ -1 E B (k-1) [m] -B -1 γEZ (k) [m]Z (k) [m] T -B -1 µ (d l ,k) [m]Φ (k) [m]H (k) [m] T
The CorInfoMax-EBD scheme introduced in this section is more biologically plausible than the earlier MLP-based EBD formulation, as its learning rules can be implemented through local mechanisms such as lateral and three-factor Hebbian/anti-Hebbian updates, realistic neuron models with apical and basal dendrites, and feedback via backward predictors. Additionally, both the entropy and power normalization terms in CorInfoMax are realizable using biologically plausible operations, particularly in the online setting with single-sample updates. See Appendix H.2 for further discussion.
this section cite: ['b55', 'b44']

Section: Numerical experiments
In this section, we evaluate the performance of the proposed Error Broadcast and Decorrelation (EBD) approach on benchmark datasets: MNIST [48] and CIFAR-10/100 [49]. For experiments with MNIST and CIFAR-10 involving MLP, CNN and LC, we use the same architectures used in [9]; while for CIFAR-100 we adopt a CNN architecture closely following that of [41]. We also tested the proposed CorInfoMax-EBD model against the CorInfoMax-EP model of [45]. More details about architectures, implementations, hyperparameter selections, and experimental outputs are provided in the Appendix I. EBD test accuracy results compared to BP (with MSE criterion) and three error-broadcast methods: DFA without and with entropy regularization (DFA-E) [6], global error vector broadcasting (nonnegative-(NN-GEVB) and mixed-sign-(MS-GEVB)) [9] are in Table 1 for both MNIST and CIFAR-10. Under our training setup, BP yielded comparable test accuracies for these datasets with both MSE and Cross-Entropy losses, though we report only the MSE results. In addition, CIFAR-100 results of EBD compared to BP (with Cross-Entropy criterion) and DFA is given in Table 2. Lastly, the test accuracies for biological CorInfoMax networks trained with EP and EBD methods are in Table 3.
These results show that EBD-trained networks achieve equivalent performance on the MNIST dataset and significantly better performance on the CIFAR-10 and CIFAR-100 datasets compared to other error broadcasting methods. These improvements of EBD in Table 1 over DFA can be attributed to the adaptability of error projection weights in EBD. The improvement of CorInfoMax-EBD over CorInfoMax-EP in Table 3 can be attributed to CorInfoMax-EBD incorporating error decorrelation in updating lateral weights, whereas CorInfoMax-EP relies only on (anti-)Hebbian updates. Particularly noteworthy is the performance of CorInfoMax-EBD, which not only substantially improves upon the original CorInfoMax-EP on CIFAR-10 (e.g., 55.79% vs. 50.97% for 3-layers with batch size 20) but also demonstrates encouraging scalability with depth, with a 10-layer CorInfoMax-EBD achieving 96.38% on MNIST and 54.89% on CIFAR-10 using online learning (batch size 1). This highlights EBD's potential in deeper, more complex biological networks.
this section cite: ['b47', 'b48', 'b8', 'b40', 'b44', 'b5', 'b8']

Section: Conclusions, extensions and limitations
Conclusions. We introduced the Error Broadcast and Decorrelation framework, a biologically plausible alternative to backpropagation. EBD addresses the credit assignment problem by minimizing correlations between layer activations and output errors, offering fresh insights into biologically realistic learning. This approach provides a theoretical foundation for existing error broadcast mechanisms and three-factor learning rules in biological neural networks and facilitates flexible implementations in neuromorphic and artificial neural systems. EBD's error-broadcasting mechanism aligns with biological processes using local updates, and notably, has proven effective for training deep recurrent biologically-plausible networks (e.g., the 10-layer CorInfoMax-EBD), thereby addressing a key challenge in effectively scaling deep, biologically plausible learning with local rules. Moreover, EBD's simplicity and parallelism suit efficient hardware, like neuromorphic systems.
Extensions. The MMSE orthogonality property underlying EBD offers significant promise for new algorithms, deeper theoretical understanding, and neural network analysis in both artificial and biological contexts. Further theoretical extensions, drawing from the groundwork laid in Appendix B.2, could focus on deriving tighter convergence guarantees for EBD in practical (finite-width) settings and on investigating the impact of more adaptive choices for the decorrelation functions g (k) . In addition, EBD provides theoretical underpinnings for error-broadcast mechanisms with three-factor learning rules, enabling the conversion of two-phase contrastive methods into a single-phase approach. We are currently unaware of similar theoretical properties for alternative loss functions. Finally, our numerical experiments in Appendix J.2 reveal that similar decorrelation behavior occurs for networks trained with backpropagation and categorical cross entropy loss, suggesting that decorrelation may be a general feature of the learning process and an intriguing avenue for further investigation.
this section cite: []

Section: Impact and limitations.
This paper seeks to advance the fields of Machine Learning and Computational Neuroscience by proposing a novel learning mechanism. As a foundational learning algorithm, we do not identify specific negative societal impacts arising directly from the EBD mechanisms beyond general considerations common to advancements in machine learning. While EBD offers a theoretically-grounded framework for error broadcast based and three factor learning that has yielded competitive (and in some cases, superior) performance against other error-broadcast methods on the presented benchmarks, several aspects warrant future investigation:
Scalability: The current work evaluates EBD on MLP, CNN, LC and recurrent biological networks for image classification tasks like MNIST and CIFAR-10. Results on the 10-layer CorInfoMax-EBD demonstrate the potential to scale EBD to deeper, biologically realistic recurrent architectures using online learning. However, assessing EBD's performance on significantly larger datasets, or its applicability to diverse large-scale architectures in other domains, remains an important open direction. While related methods like DFA have been explored in such contexts [41], comprehensive empirical validation of EBD itself under those conditions is needed.
Computational Cost and Hyperparameters: The dynamic updating of error projection matrices R(k) gϵ and the optional inclusion of regularization terms like layer entropy (discussed in Appendix G and Appendix I.7) contribute to computational and memory overhead compared to simpler schemes like DFA with fixed projectors, or standard backpropagation. EBD also introduces several hyperparameters (e.g., learning rates for decorrelation and regularization, forgetting factors) that require careful tuning, although this offers flexibility. Future work could explore more efficient update mechanisms or automated tuning strategies.
this section cite: ['b40']

Section: References
Ref_id:b0 Title: Learning representations by back-propagating errors Year: (1986)
Ref_id:b1 Title: The recent excitement about neural networks Year: (1989)
Ref_id:b2 Title: Synaptic plasticity forms and functions Year: (2020)
Ref_id:b3 Title: Simple statistical gradient-following algorithms for connectionist reinforcement learning Year: (1992)
Ref_id:b4 Title: Learning curves for stochastic gradient descent in linear feedforward networks Year: (2003)
Ref_id:b5 Title: Direct feedback alignment provides learning in deep neural networks Year: (2016)
Ref_id:b6 Title: Learning in the machine: Random backpropagation and the deep learning channel Year: (2018-07)
Ref_id:b7 Title: Theories of error back-propagation in the brain Year: (2019)
Ref_id:b8 Title: Credit assignment through broadcasting a global error vector Year: (2021)
Ref_id:b9 Title: Optical training of large-scale transformers and deep neural networks with direct feedback alignment Year: (2024)
Ref_id:b10 Title: The influence of learning rule on representation dynamics in wide neural networks Year: (2022)
Ref_id:b11 Title: Principled training of neural networks with direct feedback alignment Year: (2019)
Ref_id:b12 Title: Eligibility traces and plasticity on behavioral time scales: experimental support of neohebbian three-factor learning rules Year: (2018)
Ref_id:b13 Title: Learning with three factors: modulating hebbian plasticity with errors Year: (2017)
Ref_id:b14 Title: An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity Year: (2017)
Ref_id:b15 Title: Contrastive similarity matching for supervised learning Year: (2021)
Ref_id:b16 Title: Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects Year: (1999)
Ref_id:b17 Title: Constrained predictive coding as a biologically plausible model of the cortical hierarchy. Advances in Neural Information Processing Systems Year: (2022)
Ref_id:b18 Title: Unlocking the potential of similarity matching: Scalability, supervision and pretraining Year: (2023)
Ref_id:b19 Title: A learning algorithm for boltzmann machines Year: (1985)
Ref_id:b20 Title: Biologically plausible error-driven learning using local activation differences: The generalized recirculation algorithm Year: (1996)
Ref_id:b21 Title: Equilibrium propagation: Bridging the gap between energy-based models and backpropagation Year: (2017)
Ref_id:b22 Title: The forward-forward algorithm: Some preliminary investigations Year: (2022)
Ref_id:b23 Title: Efficient biologically plausible adversarial training Year: (2023)
Ref_id:b24 Title: Error-driven input modulation: Solving the credit assignment problem without a backward pass Year: (2022)
Ref_id:b25 Title: Learning process in an asymmetric threshold network Year: (1986)
Ref_id:b26 Title: How auto-encoders could provide credit assignment in deep networks via target propagation Year: (2014)
Ref_id:b27 Title: Difference target propagation Year: (2015)
Ref_id:b28 Title: Random synaptic feedback weights support error backpropagation for deep learning Year: (2016)
Ref_id:b29 Title: Backpropagation without weight transport Year: (1994)
Ref_id:b30 Title: Deep learning without weight symmetry Year: (2024)
Ref_id:b31 Title: The hsic bottleneck: Deep learning without back-propagation Year: (2020)
Ref_id:b32 Title: Model-free distributed learning Year: (1990)
Ref_id:b33 Title: A fast stochastic error-descent algorithm for supervised learning and optimization Year: (1992)
Ref_id:b34 Title: Gradient learning in spiking neural networks by dynamic perturbation of conductances Year: (2006)
Ref_id:b35 Title: Deep learning without weight transport Year: (2019)
Ref_id:b36 Title: Learning without feedback: Fixed random learning signals allow for feedforward training of deep neural networks Year: (2021)
Ref_id:b37 Title: Biologically-plausible learning algorithms can scale to large datasets Year: (2019)
Ref_id:b38 Title: Assessing the scalability of biologically-motivated deep learning algorithms and architectures Year: (2018)
Ref_id:b39 Title: Efficient convolutional neural network training with direct feedback alignment Year: (2019)
Ref_id:b40 Title: Direct feedback alignment scales to modern deep learning tasks and architectures Year: (2020)
Ref_id:b41 Title: Probability, Random Variables, and Stochastic Processes Year: (2002)
Ref_id:b42 Title: Linear estimation. Prentice-Hall information and system sciences series Year: (2000)
Ref_id:b43 Title: Selfsupervised learning with an information maximization criterion Year: (2022)
Ref_id:b44 Title: Correlative information maximization: a biologically plausible approach to supervised deep neural networks without weight symmetry Year: (2023)
Ref_id:b45 Title: Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules Year: (2016)
Ref_id:b46 Title: Operant matching is a generic outcome of synaptic plasticity based on the covariance between reward and neural activity Year: (2006)
Ref_id:b47 Title: The mnist database of handwritten digit images for machine learning research Year: (2012)
Ref_id:b48 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b49 Title: Inference and information lecture notes Year: (2017)
Ref_id:b50 Title: Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning Year: (2008)
Ref_id:b51 Title: Uniform approximation of functions with random bases Year: (2008)
Ref_id:b52 Title: On the approximation properties of random ReLU features Year: (2018)
Ref_id:b53 Title: Long-range projections coordinate distributed brain-wide neural activity with a specific spatiotemporal profile Year: (2016)
Ref_id:b54 Title: Bottom-up inputs are required for establishment of top-down connectivity onto cortical layer 1 neurogliaform cells Year: (2021)
Ref_id:b55 Title: MNIST handwritten digit database Year: (2010)
Ref_id:b56 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b57 Title: Updating formulae and a pairwise algorithm for computing sample variances Year: (1982)
Ref_id:b58 Title:  Year: ()
Ref_id:b59 Title:  Year: ()
Ref_id:b60 Title: Implementation details for Direct Feedback Alignment (DFA) and Year: ()
Ref_id:b61 Title: Runtime comparisons for the update rules Year: ()
Ref_id:b62 Title:  Year: ()
Ref_id:b63 Title:  Year: ()
Ref_id:b64 Title: J Calculation of the correlation between layer activations and output error J.1 Correlation in the mean squared error (MSE) criterion-based training Year: ()
Ref_id:b65 Title:  Year: ()
