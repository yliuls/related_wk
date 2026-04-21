Title: Not all solutions are created equal: An analytical dissociation of functional and representational similarity in deep linear neural networks
Abstract: A foundational principle of connectionism is that perception, action, and cognition emerge from parallel computations among simple, interconnected units that generate and rely on neural representations. Accordingly, researchers employ multivariate pattern analysis to decode and compare the neural codes of artificial and biological networks, aiming to uncover their functions. However, there is limited analytical understanding of how a network's representation and function relate, despite this being essential to any quantitative notion of underlying function or functional similarity. We address this question using analysable two-layer linear networks and numerical simulations in nonlinear networks. We find that function and representation are dissociated, allowing representational similarity without functional similarity and vice versa. Further, we show that neither robustness to input noise nor the level of generalization error constrain representations to the task. In contrast, networks robust to parameter noise have limited representational flexibility and must employ task-specific representations. Our findings suggest that representational alignment reflects computational advantages beyond functional alignment alone, with significant implications for interpreting and comparing the representations of connectionist systems.

Section: Introduction
The parallel distributed processing hypothesis posits that function in artificial and biological networks emerges from interactions among simple interconnected units that com-Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). pute with distributed representations (Rumelhart et al. 1986).
Accordingly, one might aim to identify function from networks observables such as connectivity weights and neural activity patterns; however, this is often complicated by the inherent complexity and partial observability of these systems. In particular, the structure of artificial and biological networks is often non-identifiable in the sense that networks can be structurally distinct, yet implement the same input-output mapping. For example, biophysical neuron models can exhibit nearly identical functions at both the neuron (Goldman et al. 2001) and network levels (Prinz et al. 2004) despite considerable variation in their architecture (reviewed in Marder and Goaillard 2006;Albantakis et al. 2024). Similarly, artificial neural networks (ANNs) are almost always non-identifiable due to simple symmetries, such as permutation-invariance of neurons (Sussmann 1992;Albertini and Sontag 1993), scale-invariance of activation functions (Neyshabur et al. 2015a), alongside more complex symmetries arising from feature composition across layers and from finite training data (Refinetti et al. 2021;Arous et al. 2022). As modern networks are deep and heavily overparametrised (Zhang et al. 2021), they are inherently nonidentifiable, with many parametrisations yielding the same input-output behaviour. Determining when parametrisations become identifiable and understanding the consequences of non-identifiability remain open problems (Roeder et al. 2021;Entezari et al. 2022;Vlačić and Bölcskei 2022;Ghosh et al. 2022;Wang and Jordan 2021;Godfrey et al. 2022;Martinelli et al. 2023;Bona-Pellissier et al. 2023;Lampinen et al. 2024;Kori et al. 2024;Marconato et al. 2024).
Even deep linear networks exhibit both trivial and non-trivial symmetries, making their parametrisation non-identifiable.
While any deep linear network can be re-parametrised as a single linear transformation (Laurent and Brecht 2018), it does so through multistage computations that give rise to hidden-layer representations. Moreover, the optimisation landscape of a deep linear network is non-convex and contains a high-dimensional solution manifold (Figure 1) whose shape is determined by the statistics of training data and the network architecture (Baldi and Hornik 1989;Saxe et al. 2014;Arora et al. 2019), making it a useful surrogate for studying representation learning (Saxe et al. 2019;Braun et al. 2022;Dominé et al. 2025). Here, we leverage the analytical tractability of deep linear networks to study functionally equivalent parametrisations at global minimum error. Crucially, these solutions employ different internal representations, which has significant computational consequences, most notably in their affordances for linear decoding, representational similarity analysis (Section 4), and their sensitivity to noise (Section 5).
We now detail our contributions:
• We derive exact parametric equations characterising the complete and distinct subregions of the solution manifold in two-layer linear networks.
• We demonstrate that, although all subregions allow flexible neural representations, some inherently lead to identifiable task-specific representational similarities, while others result in non-identifiable, task-agnostic representational similarities.
• We establish that in contrast to task-specific solutions, task-agnostic solutions are non-identifiable and noncomparable.
• We analytically show that input noise and generalisation error do not constrain representations to task-specific regions, whereas parameter noise does.
• We validate our analytical findings through numerical simulations, demonstrating that these computational principles persist in non-linear neural networks.
All simulations are detailed in Appendix A, and a code repository reproducing all figures is available on GitHub at lukas-braun/dissociating-similarity.
this section cite: ['b70', 'b35', 'b64', 'b57', 'b3', 'b78', 'b4', 'b65', 'b9', 'b90', 'b68', 'b29', 'b81', 'b33', 'b82', 'b34', 'b59', 'b13', 'b49', 'b46', 'b56', 'b50', 'b11', 'b74', 'b7', 'b73', 'b14', 'b24']

Section: Setting and preliminaries
We consider a two-layer linear network (Figure 1A),
ŷn = W 2 W 1 x n ,(1)
trained to minimise the mean-squared error
L MSE = 1 2P P n=1 ||ŷ n -y n || 2 2 (2)
over a dataset D = {(x n , y n )} P n=1 , with inputs x n ∈ R Ni and corresponding targets y n ∈ R No . The input weights W 1 ∈ R N h ×Ni project inputs to hidden-layer neural representation h n = W 1 x n ∈ R N h , which are projected to outputs via the readout weights W 2 ∈ R No×N h . We denote by X = [x 1 , ..., x P ], Y = [y 1 , ..., y P ], and H = [h 1 , ..., h P ] the matrices that contain all inputs, targets and hidden-layer representations, respectively. The network's representational similarity matrix (RSM) is then defined by
RSM = X T W T 1 W 1 X = H T H,(3)
capturing pairwise similarities between inputs in the hidden representational space. Laurent and Brecht (2018) showed that, under the following assumptions:
Assumption 2.1. The loss function is convex and differentiable, e.g., the mean-squared error loss. Assumption 2.2. The network is not bottle-necked, i.e., min (N i , N o ) ≤ N h all local minima of a deep linear network are global and equivalent to the solution of the corresponding single-layer linear regression problem. Notably, this result holds without assumptions on the structure of the training data. In our setting, this permits the following definition (see Appendix C): Definition 2.3. Under Assumptions 2.1 and 2.2, any pair of network weights satisfying
W 2 W 1 Σ xx = Σ yx(4)
is a globally optimal general linear solution (GLS), where
Σ xx = 1 P P n=1 x n x T n and Σ yx = 1 P P n=1 y n x T n (5
)
denote the input and input-output covariance matrices.
Note that the absence of suboptimal minima does not preclude the existence of other critical points, nor does it guarantee convergence of gradient-based algorithms. To distinguish general weight matrices W 1 and W 2 from those that satisfy Equation (4), we denote the latter as optimal weights Ω 1 and Ω 2 . Further, we denote the compact singular value decomposition (cSVD), as defined in Appendix B.1, of the inputs and least-squares solution as
cSVD(X) = ABC T ,(6)
and
cSVD(Σ yx (Σ xx ) + ) = USV T .(7)
In the following, we analytically study the full set of global solutions, the so called solution manifold, and partition it into subregions with distinct representational and computational properties. Crucially, our analysis holds without assumptions on the structure of the training data and irrespective of how a solution is obtained, and thus does not depend on any particular learning or optimisation algorithm. Figure 2A visualises the entire solution manifold and its subregions for a simple example, providing some intuition for the formal definitions and theorems developed next.
this section cite: ['b50']

Section: Partitioning the solution manifold of two-layer linear networks
Two-layer linear networks architectures are typically highly overparametrised, admitting many combinations of input and output weights that achieve the global optimum for a given task. Formally, the set of all such network weights defines the solution manifold,
M = Ω 2 Ω 1 : Ω 2 Ω 1 Σ xx = Σ yx .(8)
We note that useful intuition about the manifold's structure can be gained by viewing it as the set of weight configurations related by invertible linear transformations. For any invertible matrix Q ∈ R N h ×N h , the weight pair
Ω 2 → Ω 2 Q -1 and Ω 1 → QΩ 1 (9
)
implements the same input-output map and thus lies on the same manifold (Baldi and Hornik 1989;Saxe et al. 2014).
In the context of a neural network architecture, these Qtransformations redistribute how information is processed across layers, for example, by rotating and scaling intermediate representations. However, since they preserve rank, they only fully characterise the solution manifold when the input and output dimensions are identical and the task has full rank, conditions that may not be met in real-world scenarios. To refine this view, we partition the input space into three subspaces: Observed and relevant, observed but irrelevant, and unobserved null directions, with corresponding projections P r = VV T , P i = AA T -VV T , and P u = I -AA T . The distinction between relevant and irrelevant directions arises because the input space can exceed the intrinsic dimensionality of the solution manifold (but not vice versa). Specifically,
r = rank(Σ yx ) ≤ min(rank(X), rank(Y)),(10)
so when the input rank exceeds the target rank, some input directions are irrelevant to solving the task. Likewise, the hidden space can be partitioned into subspaces corresponding to the hidden-layer representations of relevant, and irrelevant inputs, and all remaining unoccupied null directions. Importantly, the hidden representations of relevant and irrelevant inputs may overlap, which necessitates compensation and introduces structural constraints on the form of valid solutions. The following parametrized equation fully encapsulates this intricate structure of the solution manifold: Theorem 3.1. Any GLS satisfies
Ω 1 = Q √ SV T + Γ 1 P i + Γ 2 P u
and
Ω 2 = U √ SQ + + Ψ + Γ 3 (I -HH + ),(11)
where Q ∈ R N h ×r is an arbitrary full-column-rank matrix, Γ 1 , Γ 2 ∈ R N h ×Ni are arbitrary matrices subject to the constraint rank(QQ
+ Γ 1 P i ) ≤ rank((I -QQ + )Γ 1 P i ), Ψ = -U √ SQ + Γ 1 P i [(I -QQ + )Γ 1 P i ] + , and Γ 3 ∈ R No×N h
is an arbitrary matrix.
The first terms of Ω 1 and Ω 2 implement the core inputoutput mapping; Γ 1 and Γ 2 project from task-irrelevant and unobserved input directions; Γ 3 projects from the unoccupied hidden space; Ψ cancels interference from irrelevant inputs that are projected into the core; and the rank constraint ensures that such a correction exists. See Appendix C for a detailed proof and Figure 2B for a visualisation.
Next, we partition the solution manifold into distinct regions and subsequently analyse their respective representational and computational properties. For proofs of Theorems 3.3, 3.5 and 3.7 refer to Appendix D, and to Figure 2C-E for visualisations.
Definition 3.2. Any GLS that minimises the norm of the network function,
argmin W1,W2 ||W 2 W 1 || 2 F s.t. W 2 W 1 Σ xx = Σ yx (12
)
is a least-squares solution (LSS).
Theorem 3.3. All LSS satisfy Ω 2 Ω 1 = Σ yx (Σ xx ) + and are exactly parametrised by Ω 1 = Q √ SV T + Γ 1 P i + Γ 2 P u and
Ω 2 = U √ SQ + + Ψ + Φ + Γ 3 (I -Ω 1 Ω + 1 ),(13)
subject to the definitions and constraints in Theorem 3.1, and the additional constraint that rank(HH + Ω 1 P u ) ≤ rank((I -HH + )Ω 1 P u ), and where Φ = -(U √ SQ + + Ψ)Ω 1 P u [(I -HH + )Ω 1 P u ]
+ .
Here, Φ cancels interference from unobserved inputs projected into the occupied hidden space, with the rank constraint ensuring a correction exists.
Definition 3.4. Any GLS for which the norm of the hiddenlayer representations and readout weights is minimised
argmin W1,W2 ||W 1 X|| 2 F + ||W 2 || 2 F s.t. W 2 W 1 Σ xx = Σ yx ,(14)
is a minimum representation-norm solution (MRNS). Theorem 3.5. All MRNS are parametrised by
Ω 2 = M √ NR T and Ω 1 = R √ NO T X + +Γ 2 P u , (15
)
where R ∈ R N h ×r is an arbitrary (semi-)orthonormal matrix, and
cSVD(YCC T ) = MNO T . (16
)
Definition 3.6. Any GLS for which the sum of the norm of the weight matrices is minimised
argmin W1,W2 ||W 1 || 2 F + ||W 2 || 2 F s.t. W 2 W 1 Σ xx = Σ yx (17
)
is a minimum weight-norm solution (MWNS). Theorem 3.7. All MWNS are parametrised by
Ω 2 = U √ SR T and Ω 1 = R √ SV T .(18)
We note, that the relation between MWNS and the singular value decomposition of the least-squares solution has been previously derived under strong assumptions, namely that Σ xx = I, N i = N o and that Σ yx has full rank (Saxe et al. 2019, see appendix S14-S15). Further, we note that Corollary 3.8. MRNS and MWNS are identical if inputs are whitened, i.e., Σ xx = I.
A key difference between the four solution types lies in how they constrain the image and kernel of the weight matrices, which map between input, hidden, and output spaces. GLS impose minimal constraints. LSS restrict projections to and from null spaces in the input and hidden layers. MRNS further reduce freedom in the null space, eliminate irrelevant projections, and constrain the core solution itself. MWNS, the most restrictive class, eliminate all irrelevant and nullspace projections, and enforce balance in the core by requiring equal contributions from the input and output weights. While this perspective clarifies how different subregions of the solution manifold constrain the structure of the weight matrices, it does not address a key question: how these solutions differ in their hidden-layer representations.
this section cite: ['b11', 'b74', 'b73']

Section: Hidden-layer representations
Understanding hidden-layer representations begins with identifying the degrees of freedom in the input weights, which govern how inputs are mapped into hidden space. Intuitively, input weights are constrained only by the need to preserve sufficient task-relevant information for the output weights to solve the task. In this section, we go beyond this intuition by leveraging the exact parametrisations of Ω 1 to precisely characterise the degrees of freedom in hidden-layer representations. Proofs for Corollaries 3.9, 3.11 and 3.12 are in Appendix E. We begin by noting that GLS and LSS differ only in how they handle projections from unobserved input and unoccupied hidden directions and thus implement the same input-output map on the training data. As a result, they exhibit identical degrees of freedom in their hidden-layer representations
H = Q √ SV T X + Γ 1 P i X.(19)
Since Q can be any full-column-rank matrix and Γ 1 is free up to a rank constraint, GLS and LSS support nearly arbitrary hidden-layer representations. To illustrate this, we consider a semantic learning task linking items to positions within a hierarchical structure (Figure 3A,B). For example, we can select a point on the solution manifold where the hidden-layer representations of the items form the shape of an elephant (Figure 3C). Corollary 3.9. GLS and LSS permit any RSM of the form
RSM = X T (V √ SQ T Q √ SV T + V √ SQ T Γ 1 P i + P T i Γ T 1 Q √ SV T + P T i Γ T 1 Γ 1 P i )X.(20)
In our example, this yields a highly structured RSM, yet does not reflect the task structure, i.e., the hierarchical relationships between items (Figure 3C). Accordingly, we make Definition 3.10. Neural representations whose RSM depends on the specific choice of input weights are taskagnostic representations.
In contrast, hidden-layer representations of MRNS
H = R √ NO T X + X (21
)
and MWNS
H = R √ SV T X (22
)
are unique up to an orthogonal transformation R, which includes rotations and reflections. Thus, in both cases, hiddenlayer representations are not unique. In the semantic hierarchy task, this results in representations that appear arbitrary and unstructured (Figure 3D, E). However, Corollary 3.11. The RSM of MRNS is unique and given by
RSM = ONO T . (23
)
Since O and N are fully determined by the training data, the RSM is invariant to the specific choice of input weights. Similarly, Corollary 3.12. The RSM of MWNS is unique and given by
RSM = X T VSV T X.(24)
Again, the RSM is invariant to the specific choice of input weights, as V and S are fully determined by the training data. Accordingly, we make Definition 3.13. Neural representations whose RSM if fully determined by the training data are task-specific representations.
In the semantic hierarchy task, this yields an RSM that reflects the hierarchical structure, where representational similarity increases with proximity in the hierarchy, for MRNS; and an RSM that reflects a combination of input statistics and target hierarchy for MWNS (Figure 3D,E).
In summary, in two-layer linear networks, neural representations and the underlying function are dissociable: the same function can arise from different hidden-layer representations, and the same representations can support different functions. For GLS and LSS this flexibility supports almost arbitrary representations and task-agnostic RSMs. In contrast, MRNS and MWNS impose constraints which determine representations up to orthonormal transformations, and are guaranteed to have unique and task-specific RSMs.
this section cite: []

Section: Implications for neural data analysis
A fundamental challenge in understanding computations and learning in artificial and biological neural networks is linking changes in connectivity and representations to changes in function. However, if representation and function are dissociable, two key observations follow. First, changes in network function need not alter neural representations, as changes in one layer can be offset by compensatory changes in the next. Second, changes in network function can proceed without altering representations, as it can, in principle, occur entirely in downstream layers. We now examine the implications of these findings for representational comparisons, representational drift, and the synaptic stability-plasticity trade-off through a series of illustrative simulations. While exact outcomes depend on task specifics and hyperparameter choices, these are not our focus here; a systematic analytical and numerical exploration is left for future work.
this section cite: []

Section: Linear predictivity
A common method for comparing neural representations is to assess how well activation patterns from a source model or recording can predict those of a target via linear regression (e.g., Yamins et al. 2014;Yamins and DiCarlo 2016). High linear predictivity is often interpreted as evidence that two systems process information similarly. However, since function and hidden-layer representations are dissociable, strong linear predictivity does not necessarily imply functional alignment. We illustrate this by comparing hiddenlayer representations from independent random walks on the solution manifold of LSS, MWNS, and MRNS (Figure 4A).
In each case, we compare either representations from two different network functions (across function) or from the same network function (within function). While R 2 scores are on average slightly higher for within-function comparisons, the main determinant of predictivity is whether the representations are task-agnostic or task-specific. Specifically, in across-function comparisons, R 2 scores are highest when the source representation comes from a LSS (which shares representational degrees of freedom with GLS), fol- lowed by MRNS, and lowest for MWNS. Within-function comparisons yield high R 2 when the source is task-agnostic or when both source and target are of the same type; in contrast, predicting a task-agnostic representation from a task-specific one lead to the lowest R 2 scores. These patterns arise because task-agnostic solutions process both relevant and irrelevant input directions, producing higher-rank representations that cannot be linearly predicted from the lower-rank task-specific ones. These results indicate that linear predictivity is predominantly driven by solution type rather than functional alignment, and may yield misleading conclusions if underlying representational constraints are not explicitly taken into account.
this section cite: ['b86', 'b85']

Section: Representational similarity analysis
Representational similarity analysis (RSA) compares neural activation patterns by evaluation the similarity of RSMs across conditions, stimuli, models, or participants (Kriegeskorte et al. 2008;Haxby et al. 2014). As with linear predictivity, our analytical results show that the interpretability of RSA in terms of functional alignment critically depends on the solution type of the comparanda. We illustrate this in Figure 4B using example trajectories from the previous section. Since task-agnostic solutions (i.e., GLS and LSS) exhibit highly flexible RSMs, correlation coefficients r involving such representations fluctuate unpredictably throughout the random walk, both within and across functions. By contrast, comparing within and across task-specific solutions (i.e., MWNS and MRNS) results in static and consistent r, as they exhibit unique RSMs. However, because MWNS and MRNS induce different unique RSMs, comparisons across types yield imperfect similarity even within the same function. In summary, RSA reliably reflects functional similarity only when representational constraints enforce unique RSMs, underscoring the importance of accounting for solution type in representational comparisons.
this section cite: ['b47', 'b39']

Section: Drifting neural representations
Intuitively, one might expect that if a stimulus elicits stable perception and behaviour, the associated neural repre-sentations should likewise remain stable (Rule et al. 2019;Driscoll et al. 2022). However, this assumptions is challenged by converging evidence of representation drift across species, brain regions and modalities (e.g., Ziv et al. 2013;Driscoll et al. 2017;Schoonover et al. 2021;Marks and Goard 2021;Deitch et al. 2021;Ahmed et al. 2024). Our analysis shows that the existence of a solution manifold allows hidden-layer representations to vary without changing the implemented function. This dissociation implies that stable perception and behaviour do not require stable representations. Indeed, an optimal linear decoder trained on an initial representation rapidly degrades in performance during a random walk on the solution manifold (Figure 4C). Thus, representational drift need not signal functional change, but may instead reflect a reparametrisation within a functionally equivalent subspace.
this section cite: ['b69', 'b24', 'b91', 'b25', 'b74', 'b58', 'b20', 'b2']

Section: Synaptic stability and plasticity
The so-called stability-plasticity dilemma posits that neural systems must remain plastic enough to acquire new knowledge while remaining stable enough to retain previously learned information (Grossberg 1987;Abraham and Robins 2005). This view, grounded in single-neuron and synapselevel intuitions, has motivated continual learning algorithms that explicitly regulate synaptic changes to preserve past knowledge (e.g., Kirkpatrick et al. 2017;Zenke et al. 2017;Aljundi et al. 2018). However, our analysis shows that this dilemma need not apply at the network level, because, independent of the solution type, many distinct configurations of synapses and representations implement the same function (see e.g., Figure 1). This raises important methodological concerns: observing or inducing isolated synaptic or representational changes may not suffice to infer function, learning, or the underlying learning mechanisms in artificial or biological neural networks.
this section cite: ['b37', 'b0', 'b44', 'b88', 'b5']

Section: Advantages of task-specific representations
A natural question arises from the observation that function and representation are dissociable: why do biological and artificial systems often converge to non-arbitrary representations that obey the structure of the task? One possible explanation is that task-specific representations confer computational advantages, creating selective pressure in both biological and artificial systems. Consequently, such representations may emerge as preferred functional implementations, leading to representational alignment across systems.
this section cite: []

Section: Secondary error
One hypothesised advantage of task-specific representations is improved performance on secondary datasets, such as in-or out-of-distribution generalisation. To test this, we identify solutions on the primary-task solution manifold that minimise the error on an unseen secondary dataset D = {(x n , ỹn )} Q n=1 . In Appendix F we derive, Theorem 5.1. The secondary error is minimised by all solutions of the form
Ω 2 Ω 1 = Σ yx Σ + xx + ZP u ,(25)
where
Z = Ỹ -Σ yx Σ + xx X P u X + + Γ Pu ,(26)
with Γ ∈ R No×Ni arbitrary and
Pu = I -P u X(P u X) + .
Since the solution is a LSS with a perturbation in the unobserved null directions of the primary-task inputs, minimising secondary error permits task-agnostic solutions. Thus, observing H gives no information about secondary task performance and secondary error cannot explain the emergence of task-specific representations in two-layer linear networks.
this section cite: []

Section: Sensitivity to noise
Neural systems are subject to a multitude of internal and external sources of noise, which range from variability in incoming sensory signals to fluctuations in synaptic efficacy and spontaneous neural activity (Faisal et al. 2008). Therefore, solutions that exhibit robustness to such noise are advantageous, as they enable the neural circuitry to maintain reliable function (Johnston et al. 2020). The following theorems are derived in Appendix G. Theorem 5.2. The expected loss under additive, independent and identically distributed (i.i.d.), zero-centred input noise ξ xn with variance
σ 2 x is 1 2P P n=1 ||Ω 2 Ω 1 x n + ξ xn -y n || 2 2 = σ 2 x 2 ||Ω 2 Ω 1 || 2 F + c, (27
)
where c is a noise-independent constant that only depends on the training data.
Corollary 5.3. The expected loss under input noise is exclusively minimised by LSS, which include MWNS as a subspace (see Figure 4D).
Thus, while robustness to input noise selects for solutions with minimal functional norm, LSS employ task-agnostic representations, so robustness alone does not ensure representational alignment.
Theorem 5.4. The expected loss under additive, i.i.d., zerocentred noise Ξ 1 and Ξ 2 in the parameters, with variances
σ 2 1 ∝ 1/||X|| 2 F and σ 2 2 ∝ 1/N o is 1 2P P n=1 || (Ω 2 + Ξ 2 ) (Ω 1 + Ξ 1 ) x n -y n || 2 2 = 1 2P ||Ω 1 X|| 2 F + ||Ω 2 || 2 F + c . (28
)
where c is again a noise-independent constant.
Corollary 5.5. The expected loss under parameter noise is exclusively minimised by MRNS (see Figure 4D).
We have scaled noise variances to simplify the analytical expression; without this scaling, the results hold up to multiplicative constants. Robustness to parameter noise thus selects for solutions with task-specific representations, ensuring representational alignment.
Theorem 5.6. Under the assumption that the input data is whitened, i.e., Σ xx = I, the expected loss under additive, i.i.d., zero-centred parameter noise Ξ 1 and Ξ 2 with variance σ 2 1 ∝ 1/N i and σ 2 2 ∝ 1/N o and input noise ξ xn with variance σ 2
x is
1 2P P n=1 || (Ω 2 + Ξ 2 ) (Ω 1 + Ξ 1 ) x n + ξ xn -y n || 2 2 = σ 2 x 2 ||Ω 2 Ω 1 || 2 F + ||Ω 2 || 2 F + ||Ω 1 || 2 F (29
)
+ 1 2P ||Ω 2 || 2 F + ||Ω 1 X|| 2 F + c,
with noise-independent constant c.
Corollary 5.7. Under the stated assumptions and constraints, the expected loss under input and parameter noise is minimised exclusively by MRNS and MWNS.
We note, that if the input data is not whitened, interaction terms render the optimal subspace depends on input statistics, complicating its explicit analytical characterisation.
In summary, neither minimising secondary error nor input noise sensitivity promotes task-specific representations. In contrast, robustness to parameter noise selectively favours solutions with task-specific structure, thereby supporting representational alignment. This suggests that shared representations across biological and artificial systems may arise from implicit or explicit optimisation for parameter robustness, such as regularisation strategies that favour lownorm solutions. However, this reflects an inductive bias rather than a general principle and without explicit justification, functional and representational alignment cannot be assumed to coincide.
this section cite: ['b30', 'b43']

Section: Nonlinear networks
The results presented thus far apply to two-layer linear networks. We now extend our study to emphnonlinear networks (networks with nonlinear activation functions), and show that they exhibit analogous degrees of freedom in representation and associated computational trade-offs as their linear counterparts. Here, we face a challenge: A full characterisation of the solution manifold of general nonlinear network remains analytically intractable, even for two-layer networks (Misiakiewicz and Montanari 2024). However, substantial progress has been made in deriving function-preserving transformations that allow reparametrisation of nonlinear networks while leaving their input-output map invariant (Simsek et al. 2021;Martinelli et al. 2023).
Here, we exploit these function-preserving transformations to construct functionally equivalent reparametrisations of nonlinear networks, which allows us to probe computational differences between networks with minimal and expanded representations, in analogy to the task-specific and taskagnostic representations of Section 3.
this section cite: ['b60', 'b77', 'b59']

Section: Functional invariances in deep ReLU networks
Feedforward networks with any activation function are output-invariant to permuting neurons within a layer, as permuting the rows of one weight matrix and the corresponding columns of the next leaves the network function unchanged (permutation invariance; Sussmann 1992). Feedforward networks with rectified linear unit (ReLU) activation are further invariant to rescaling a neuron's incoming weights by a factor α > 0 and its outgoing weights by 1/α, due to the non-negative homogeneity of ReLU (scale invariance; Neyshabur et al. 2015a). Simsek et al. (2021) and Martinelli et al. (2023) identify additional invariances that fully characterise the manifold of global minima in teacherstudent settings, where one network is trained to replicate the function of another. Although these invariances may not capture all functionally equivalent parametrisations outside of the teacher-student setting, they provide a means to construct network reparametrisations that exactly preserve network function. We realise two of their invariances by inserting hidden-layer neurons with arbitrary incoming weights and zero outgoing weights (nuisance-neuron invariance) and by duplicating hidden-layer neurons while halving the outgoing weights of both the original and the duplicated neuron (duplication invariance). Lastly, one can add any perturbation to the input-layer weights that lies in the unobserved nullspace of the input data while preserving the network's function (input-nullspace invariance).
this section cite: ['b77', 'b59']

Section: Manipulating representations of ReLU networks
We train a two-layer ReLU network with 1024 hidden neurons on the MNIST dataset (LeCun et al. 1998) from small norm random weights, resulting in task-specific representations (Figure 5A, "rich" learning from Jacot et al. 2018;Chizat et al. 2019;Woodworth et al. 2020). Starting from this trained model, we use augmented Lagrangian optimisation to modify the hidden-layer activations of 1024 training inputs such that their representations collectively resemble an image of two elephants, while enforcing that the network's predicted class labels remain unchanged across the entire training set (Figure 5B). This transformation illustrates the representational freedom among nonlinear ReLU networks that make the same class predictions. In addition, even when limited to manipulations that exactly preserve a network's input-output map (a stricter condition than preserving classifications) one can induce a task-agnostic RSM (Figure 5C).
this section cite: ['b51', 'b42', 'b19', 'b84']

Section: Computational advantages in ReLU networks
To complement the analytical results in Section 5.2 on the robustness of task-specific representations in linear networks, we empirically evaluate secondary (test) error and robustness to input and parameter noise across different nonlinear solution types. We train two-layer networks with 1024 hidden dimensions and ReLU activation on the training set of the MNIST digit classification task (LeCun et al. 2010) from 8 random initialisations. These models trained from small initial weights serve our task-trained (minimal) solutions. We next apply four function-preserving transforms defined by the four invariances of Section 6.1 to these task-trained (minimal) networks to construct expanded (non-minimal) parametrisations that exactly preserve the network function. These minimal and non-minimal parametrisations serve as our solution types in the nonlinear setting.
In Figure 5D, we observe the effect of input noise on the task-trained (minimal) and expanded (non-minimal) networks. In accordance with the linear result, only transformations in the unobserved input space have a deleterious effect (input-null). In Figure 5E, we observe the effect of parameter noise on the initial and transformed networks. Nonminimal models (scaled, nuisance, duplicate) degrade in test error more quickly than minimal ones, similarly to what is derived in Section 5.2, though duplicated expansions (duplicate) are more robust due to noise averaging. In contrast, input-nullspace perturbations (input-null) have no effect because transformations are in unobserved input directions, and input noise is absent. Lastly, at nearzero noise levels, we observe that no manipulations inflate the secondary (test) error, consistent with the result of Section 5.1 that generalisation performance does not constrain network representations to be minimal.
this section cite: []

Section: Related work
The solution manifold of artificial networks. The solution manifold of two-layer neural networks was first described by Baldi and Hornik (1989). Subsequent work showed under some assumptions that all minima in deep linear networks are global and equivalent to those in linear regression (Laurent and Brecht 2018). The dissociation between general linear solutions and minimum-norm solutions has been previously studied under a set of strong assumptions (Saxe et al. 2014). Sensitivity to noise for task-specific and task-agnostic rich and lazy solutions has been previously studied numerically in nonlinear neural networks (Flesch et al. 2022) and generalisation and transfer performance of deep linear networks have been previously investigated (Lampinen and Ganguli 2019;Advani et al. 2020;Tahir et al. 2024;Ingrosso et al. 2025) using a teacher-student paradigm (Gardner and Derrida 1989;Riegler and Biehl 1995;Saad and Solla 1995). The relation between representational drift and drift on the solution manifold that results from stochasticity during gradient descent (Chaudhari and Soatto 2018) has been studied on a subpart of the solution manifold in linear networks (Pashakhanloo and Koulakov 2023) and in nonlinear networks, again, relying on the teacher-student setting (Avidan et al. 2023;Li et al. 2024).
Comparing the solutions of artificial and biological networks. Neuroscientists have identified parallels between artificial and biological neural computation (Richards et al. 2019;Saxe et al. 2021;Doerig et al. 2023) from hierarchical feature extraction in visual processing (DiCarlo et al. 2012;Eickenberg et al. 2017;Lindsay 2021) to analogous population dynamics during decision-making tasks (Mante et al. 2013;Chaisangmongkon et al. 2017). The field has developed various methods to quantify this shared representational structure, including representational similarity analysis (Kriegeskorte et al. 2008), linear predictivity (e.g., Yamins et al. 2014;Yamins and DiCarlo 2016), and metricbased methods (Williams et al. 2021); see Klabunde et al. (2025) for an overview of methods. However, recent work has identified significant methodological challenges in comparing representations between artificial neural networks, including confounding effects from stimulus correlations (Cai et al. 2019;Hermann and Lampinen 2020;Dujmović et al. 2023), metric-dependent results (Ding et al. 2021;Bo et al. 2025), and difficulties in matching representations even between identical networks trained with different random initialisations (Han et al. 2023); these negative results suggest further problems when comparing artificial and biological neural networks, where little is known in advance of the computation the biological networks performs. These challenges may apply to existing neural predictivity benchmarks such as Brain-score (Schrimpf et al. 2018;Schrimpf et al. 2020) and the Natural Scenes Dataset (NSD; Allen et al. 2022).
this section cite: ['b11', 'b50', 'b74', 'b31', 'b48', 'b1', 'b79', 'b41', 'b32', 'b67', 'b71', 'b17', 'b10', 'b53', 'b66', 'b23', 'b21', 'b28', 'b54', 'b55', 'b16', 'b47', 'b86', 'b85', 'b83', 'b45', 'b15', 'b40', 'b27', 'b22', 'b12', 'b38', 'b75', 'b76', 'b6']

Section: Discussion
In this work, we give a complete analytical characterisation of the global minima manifold for deep linear networks, and demonstrate that different subregions of this manifold afford different interpretability and computational properties due to their representational structure. We conclude that the use of deep, overparametrised networks poses fundamental challenges for representational analysis, interpretation, and comparison, as the impact of variability in the parametrisation of functionally equivalent representations on these use cases is significant.
Our analysis does not assume a specific learning algorithm and ignores the question of how a specific solution could be attained in practice. However, the computational advantages of task-specific representations detailed in Section 5 are compatible with the view that gradient descent, in particular in overparametrised models, is subject to implicit regularisation that prefers solutions with certain optimality properties for both linear and nonlinear networks (Neyshabur et al. 2015b;Zhang et al. 2017;Neyshabur 2017;Du et al. 2019;Arora et al. 2018;Chizat and Bach 2020;Yun et al. 2021;Vardi and Shamir 2021).
this section cite: ['b89', 'b61', 'b26', 'b8', 'b18', 'b87', 'b80']

Section: References
Ref_id:b0 Title: Memory retention-the synaptic stability versus plasticity dilemma Year: (2005)
Ref_id:b1 Title: High-dimensional dynamics of generalization error in neural networks Year: (2020)
Ref_id:b2 Title: Representational Drift in Barrel Cortex Is Receptive Field Dependent Year: (2024-12-16)
Ref_id:b3 Title: The brain's best kept secret is its degenerate structure Year: (2024)
Ref_id:b4 Title: For neural networks, function determines form Year: (1993)
Ref_id:b5 Title: Memory aware synapses: Learning what (not) to forget Year: (2018)
Ref_id:b6 Title: A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence Year: (2022)
Ref_id:b7 Title: A Convergence Analysis of Gradient Descent for Deep Linear Neural Networks Year: (2019)
Ref_id:b8 Title: On the optimization of deep networks: Implicit acceleration by overparameterization Year: (2018)
Ref_id:b9 Title: High-dimensional limit theorems for SGD: Effective dynamics and critical scaling Year: (2022)
Ref_id:b10 Title: Connecting NTK and NNGP: A Unified Theoretical Framework for Wide Neural Network Learning Dynamics Year: (2023)
Ref_id:b11 Title: Neural Networks and Principal Component Analysis: Learning from Examples without Local Minima Year: (1989-01-01)
Ref_id:b12 Title: Evaluating Representational Similarity Measures from the Lens of Functional Correspondence Year: (2025)
Ref_id:b13 Title: Parameter Identifiability of a Deep Feedforward ReLU Neural Network Year: (2023-11-01)
Ref_id:b14 Title: Exact learning dynamics of deep linear networks with prior knowledge Year: (2022)
Ref_id:b15 Title: Representational Structure or Task Structure? Bias in Neural Representational Similarity Analysis and a Bayesian Method for Reducing Bias Year: (2019-05)
Ref_id:b16 Title: Computing by Robust Transience: How the Fronto-Parietal Network Performs Sequential, Category-Based Decisions Year: (2017-03-22)
Ref_id:b17 Title: Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks Year: (2018)
Ref_id:b18 Title: Implicit Bias of Gradient Descent for Wide Two-layer Neural Networks Trained with the Logistic Loss Year: (2020)
Ref_id:b19 Title: On Lazy Training in Differentiable Programming Year: (2019)
Ref_id:b20 Title: Representational drift in the mouse visual cortex Year: (2021)
Ref_id:b21 Title: How Does the Brain Solve Visual Object Recognition? Year: (2012-09)
Ref_id:b22 Title: Grounding Representation Similarity Through Statistical Testing Year: (2021)
Ref_id:b23 Title: The neuroconnectionist research programme Year: (2023)
Ref_id:b24 Title: Representational drift: Emerging theories for continual learning and experimental future directions Year: (2022)
Ref_id:b25 Title: Dynamic reorganization of neuronal activity patterns in parietal cortex Year: (2017)
Ref_id:b26 Title: Gradient Descent Provably Optimizes Over-parameterized Neural Networks Year: (2019)
Ref_id:b27 Title: Obstacles to inferring mechanistic similarity using representational similarity analysis Year: (2023)
Ref_id:b28 Title: Seeing It All: Convolutional Network Layers Map the Function of the Human Visual System Year: (2017-05-15)
Ref_id:b29 Title: The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks Year: (2022)
Ref_id:b30 Title: Noise in the nervous system Year: (2008)
Ref_id:b31 Title: Orthogonal Representations for Robust Context-Dependent Task Performance in Brains and Neural Networks Year: (2022-06)
Ref_id:b32 Title: Three unfinished works on the optimal storage capacity of networks Year: (1989)
Ref_id:b33 Title: On Pitfalls of Identifiability in Unsupervised Learning. A Note on Year: (2022)
Ref_id:b34 Title: On the symmetries of deep learning models and their internal representations Year: (2022)
Ref_id:b35 Title: Global structure, robustness, and modulation of neuronal models Year: (2001)
Ref_id:b36 Title: Note on the generalized inverse of a matrix product Year: (1966)
Ref_id:b37 Title: Competitive learning: From interactive activation to adaptive resonance Year: (1987)
Ref_id:b38 Title: System identification of neural systems: If we got it right, would we know? Year: (2023)
Ref_id:b39 Title: Decoding neural representational spaces using multivariate pattern analysis Year: (2014)
Ref_id:b40 Title: What shapes feature representations? Exploring datasets, architectures, and training Year: (2020)
Ref_id:b41 Title: Statistical Mechanics of Transfer Learning in Fully Connected Networks in the Proportional Limit Year: (2025-04-30)
Ref_id:b42 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b43 Title: Nonlinear mixed selectivity supports reliable neural computation Year: (2020)
Ref_id:b44 Title: Overcoming catastrophic forgetting in neural networks Year: (2017)
Ref_id:b45 Title: Similarity of Neural Network Models: A Survey of Functional and Representational Measures Year: (2025-05)
Ref_id:b46 Title: Identifiable Object-Centric Representation Learning via Probabilistic Slot Attention Year: (2024)
Ref_id:b47 Title: Representational similarity analysis -connecting the branches of systems neuroscience Year: (2008)
Ref_id:b48 Title: An analytic theory of generalization dynamics and transfer learning in deep linear networks Year: (2019)
Ref_id:b49 Title: Learned feature representations are biased by complexity, learning order, position, and more Year: (2024)
Ref_id:b50 Title: Deep linear networks with arbitrary loss: All local minima are global Year: (2018)
Ref_id:b51 Title: Gradient-based learning applied to document recognition Year: (1998)
Ref_id:b52 Title: MNIST handwritten digit database Year: (2010)
Ref_id:b53 Title: Representations and generalization in artificial and brain neural networks Year: (2024)
Ref_id:b54 Title: Convolutional Neural Networks as a Model of the Visual System: Past, Present, and Future Year: (2021)
Ref_id:b55 Title: Context-Dependent Computation by Recurrent Dynamics in Prefrontal Cortex Year: (2013-11)
Ref_id:b56 Title: All or None: Identifiable Linear Properties of next-Token Predictors in Language Modeling Year: (2024)
Ref_id:b57 Title: Variability, compensation and homeostasis in neuron and network function Year: (2006)
Ref_id:b58 Title: Stimulusdependent representational drift in primary visual cortex Year: (2021)
Ref_id:b59 Title: Expand-and-Cluster: Exact Parameter Recovery of Neural Networks Year: (2023)
Ref_id:b60 Title: Six Lectures on Linearized Neural Networks Year: (2024-10)
Ref_id:b61 Title: Implicit Regularization in Deep Learning Year: (2017)
Ref_id:b62 Title: Path-SGD: Path-Normalized Optimization in Deep Neural Networks Year: (2015)
Ref_id:b63 Title: Stochastic gradient descent-induced drift of representation in a two-layer neural network Year: (2015-04-16)
Ref_id:b64 Title: Similar network activity from disparate circuit parameters Year: (2004)
Ref_id:b65 Title: Classifying high-dimensional Gaussian mixtures: Where kernel methods fail and neural networks succeed Year: (2021)
Ref_id:b66 Title: A deep learning framework for neuroscience Year: (2019)
Ref_id:b67 Title: On-Line Backpropagation in Two-Layered Neural Networks Year: (1995-10)
Ref_id:b68 Title: On Linear Identifiability of Learned Representations Year: (2021)
Ref_id:b69 Title: Causes and consequences of representational drift Year: (2019)
Ref_id:b70 Title: Parallel Distributed Processing: Explorations in the Microstructure of Cognition Year: (1986)
Ref_id:b71 Title: On-Line Learning in Soft Committee Machines Year: (1995-10-01)
Ref_id:b72 Title: If deep learning is the answer, what is the question? Year: (2021-01)
Ref_id:b73 Title: A mathematical theory of semantic development in deep neural networks Year: (2019)
Ref_id:b74 Title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks Year: (2014)
Ref_id:b75 Title: Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like? Year: (2018)
Ref_id:b76 Title: Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence Year: (2020)
Ref_id:b77 Title: Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances Year: (2021)
Ref_id:b78 Title: Uniqueness of the Weights for Minimal Feedforward Nets with a given Input-Output Map Year: (1992-07-01)
Ref_id:b79 Title: Features Are Fate: A Theory of Transfer Learning in High-Dimensional Regression Year: (2024)
Ref_id:b80 Title: Implicit Regularization in ReLU Networks with the Square Loss Year: (2021)
Ref_id:b81 Title: Neural network identifiability for a family of sigmoidal nonlinearities Year: (2022)
Ref_id:b82 Title: Desiderata for Representation Learning: A Causal Perspective Year: (2021)
Ref_id:b83 Title: Generalized Shape Metrics on Neural Representations Year: (2021)
Ref_id:b84 Title: Kernel and Rich Regimes in Overparametrized Models Year: (2020)
Ref_id:b85 Title: Using Goal-Driven Deep Learning Models to Under-stand Sensory Cortex Year: (2016-03)
Ref_id:b86 Title: Performance-Optimized Hierarchical Models Predict Neural Responses in Higher Visual Cortex Year: (2014)
Ref_id:b87 Title: A unifying view on implicit bias in training linear neural networks Year: (2021)
Ref_id:b88 Title: Continual learning through synaptic intelligence Year: (2017)
Ref_id:b89 Title: Understanding deep learning requires rethinking generalization Year: (2017)
Ref_id:b90 Title: Understanding deep learning (still) requires rethinking generalization Year: (2021)
Ref_id:b91 Title: Long-term dynamics of CA1 hippocampal place codes Year: (2013)
