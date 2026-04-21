Title: Characterizing control between interacting subsystems with deep Jacobian estimation
Abstract: Biological function arises through the dynamical interactions of multiple subsystems, including those between brain areas, within gene regulatory networks, and more. A common approach to understanding these systems is to model the dynamics of each subsystem and characterize communication between them. An alternative approach is through the lens of control theory: how the subsystems control one another. This approach involves inferring the directionality, strength, and contextual modulation of control between subsystems. However, methods for understanding subsystem control are typically linear and cannot adequately describe the rich contextual effects enabled by nonlinear complex systems. To bridge this gap, we devise a data-driven nonlinear control-theoretic framework to characterize subsystem interactions via the Jacobian of the dynamics. We address the challenge of learning Jacobians from time-series data by proposing the JacobianODE, a deep learning method that leverages properties of the Jacobian to directly estimate it for arbitrary dynamical systems from data alone. We show that JacobianODE models outperform existing Jacobian estimation methods on challenging systems, including high-dimensional chaos. Applying our approach to a multi-area recurrent neural network (RNN) trained on a working memory selection task, we show that the "sensory" area gains greater control over the "cognitive" area over learning. Furthermore, we leverage the JacobianODE to directly control the trained RNN, enabling precise manipulation of its behavior. Our work lays the foundation for a theoretically grounded and data-driven understanding of interactions among biological subsystems.

Section: Introduction
Complex systems are ubiquitous in nature. These systems exhibit a wide range of behavior and function, in large part through the dynamic interaction of multiple component subsystems within them. One approach to understanding such complex systems is to build detailed models of their underlying dynamics. An alternative and simpler yet powerful approach is offered by control theory, focusing instead on how subsystems influence and regulate one another, and how they can be controlled.
Control theory thus offers a complementary approach to both understanding and manipulating biological systems. The theory describes how inputs must be coordinated with system dynamics to achieve desired behaviors, and can be applied across domains ranging from robotics to biology (Figure 1A). The brain coordinates neural activity across multiple interconnected brain areas, dynamically modulating which regions receive information from which others depending on need and context [1,2]. Interareal interactions play central roles in cognition and consciousness [3][4][5][6][7], in selective attention [8][9][10][11][12][13][14][15][16], decision making [17,18], working memory [19,20], feature binding [21,22], motor control [23][24][25][26][27], and learning and memory [28][29][30][31][32].
A common approach to characterizing interareal interactions is to quantify communication between them, using methods such as reduced-rank regression to define "communication subspaces". These subspaces determine low-dimensional projections that maximally align high-dimensional states of the input area with high-dimensional states of the target area [33][34][35]. However, effective control not only involves alignment of high-variance input states with high-variance target states, but also appropriate alignment of the inputs with the dynamics of the target area. Given connected subsystems A and B, an identical signal from B will have dramatically different control effects on A, depending on whether the signal aligns with stable or unstable directions of A's dynamics: projections onto more unstable eigenvectors can much more readily drive the system to novel states. (For more detail see Appendix C. 1.) Accordingly, recent work in neuroscience has espoused control-theoretic perspectives on interareal interactions (Figure 1B) [24,[36][37][38][39][40][41][42]. The dominant approach has involved linear control [43][44][45][46][47][48][49][50][51][52][53]. However, the inherently nonlinear dynamics of the brain enable richer contextual control than possible to fully model with linear systems, necessitating a nonlinear control approach. One approach to extend control-theoretic analyses to nonlinear systems is by linearizing the nonlinear dynamics through Taylor expansion, which involves the Jacobian. This converts the nonlinear system into a linear state-and time-dependent one [40,[54][55][56], allowing for simple control. Jacobian linearization for control is straightforward with access to analytical expressions for the non-linear system, but it becomes non-trivial in purely data-driven scenarios. Estimating the Jacobian involves conjunctively inferring both a function and its derivative, yet good function approximation need not yield good approximations of its derivatives (see Section 4 and Appendix C.2).
This paper introduces several key contributions:
• Robust data-driven Jacobian estimation. We present JacobianODE, a deep learning-based method for estimating Jacobians from noisy trajectories in high-dimensional dynamical systems. We demonstrate the validity of this approach in several sample systems that include high-dimensional chaos.
• A framework to characterize control between interacting subsystems via data-driven Jacobian estimation. Harnessing our Jacobian estimation method, we devise a rigorous, data-driven approach for nonlinear control-theoretic analysis of how paired interacting systems, including brain areas, drive and regulate each other across different contexts.
• Data-driven inference of control dynamics in trained recurrent neural networks. We apply our data-driven framework to a recurrent neural network (RNN) trained on a working memory task. We show that, purely from data, we can identify key control-theoretic interactions between the areas, and that these interactions crucially evolve over the course of learning to produce the desired behavior.
• Demonstration of accurate control of rich interacting high-dimensional coupled dynamical subsystems. We demonstrate high accuracy in a challenging high-dimensional data-driven nonlinear control task, enabling precise control of the behavior of the RNN.
Overall, our work lays the foundation for data-driven control-theoretic analyses in complex highdimensional nonlinear systems, including the brain.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b23', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b39', 'b53', 'b54', 'b55']

Section: Related work
Interareal communication A wide range of tools have been developed and harnessed to study interareal communication in neural data [34,57]. This includes, but is not limited to, methods based on reduced rank regression [33,35], recurrent neural network models of neural dynamics [40,[58][59][60][61][62][63], Gaussian process factor analysis [64,65], canonical correlation analysis [66][67][68], convergent cross mapping [69][70][71], switching dynamical systems [72,73], granger causality [74,75], dynamic causal mapping [76], point process models [77], and machine learning methods [78,79].
Nonlinear controllability Classical results assess the controllability of nonlinear control systems via the Lie theory [80][81][82][83][84]. Another approach to nonlinear network controllability is based on attractor strength [85]. Liu et al. [54] and Parkes et al. [55] note that the large literature of linear controllability analyses could be extended locally to nonlinear systems with an appropriate linearization method.
Neural network controllability Linear structural network controllability has been implicated across a wide range of contexts, tasks, and neuropsychiatric conditions [36, 42, 44-47, 49, 50]. This approach has been extended to functional brain networks [51][52][53]. Recent work has characterized the subspaces of neural activity that are most feedback controllable as opposed to feedforward controllable using linear methods [43]. Other approaches to neural control analyses consider the identification of structural driver nodes [86], and input novelty [87].
Data-driven Jacobian estimation Besides approaches estimating Jacobians through weighted linear regressions [88,89], some methods have used direct parameterization via neural networks to learn Jacobians of general functions [90,91]. These approaches inform our method but do not explicitly address dynamical systems. Applying path-integral-based Jacobian estimation to dynamical systems is challenging, as the target function (the system's time derivative) is typically unobserved. Beik-Mohammadi et al. [92] utilized this idea in dynamical systems to learn contracting latent dynamics from demonstrations.
3 JacobianODE: learning Jacobians from data
this section cite: ['b33', 'b56', 'b32', 'b34', 'b39', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b75', 'b76', 'b77', 'b78', 'b79', 'b80', 'b81', 'b82', 'b83', 'b84', 'b53', 'b54', 'b50', 'b51', 'b52', 'b42', 'b85', 'b86', 'b87', 'b88', 'b89', 'b90', 'b91']

Section: Jacobian linearization
We consider nonlinear dynamical systems in R n , defined by ẋ(t) = f (x(t)). The Jacobian of the dynamics is a matrix-valued function J f : R n → R n×n (henceforth, J) given by
J(x(t)) = ∂ ∂x f (x(t)) =       ∂f1 ∂x1 ∂f1 ∂x2 . . . ∂f1 ∂xn ∂f2 ∂x1 ∂f2 ∂x2 . . . ∂f2 ∂xn . . . . . . . . . . . . ∂fn ∂x1 ∂fn ∂x2 . . . ∂fn ∂xn       .(1)
At each time t, the Jacobian defines a linear subspace relating input and output changes, capturing how perturbations to the system will propagate. This recasts nonlinear dynamics as linear time-varying dynamics in the tangent space locally along trajectories (formally, δ ẋ(t) = J f (x(t))δx(t), see Figure 2 left, also see Lohmiller and Slotine [93] for a discussion in the context of virtual displacements).
this section cite: ['b92']

Section: Parameterizing differential equations via the Jacobian
We now turn to the problem of how to estimate the Jacobian J from data. We assume that we have access only to observed trajectories of the system, of the form x (j) (t (j) 0 + k∆t), k = 1, 2, ..., j = 1, 2, ...where ∆t is a fixed sampling interval, j indexes the trajectory, and t (j) 0 is a trajectory-specific start time. Crucially, we do not assume access to the function f . Our method estimates the Jacobian directly via a neural network. To do this, we parameterize a neural network function Ĵθ with learnable parameters θ that is then trained to approximate J.
Path integration Following Lorraine and Hossain [91] and Beik-Mohammadi et al. [92], we exploit the fact that the path integral of the Jacobian is path independent in the construction of the loss function. This is because the rows of the Jacobian are conservative vector fields (i.e., they are the gradients of scalar functions). For an intuitive picture, consider the work done by the force of gravity as you climb a mountain. Regardless of the path you take to climb, the resulting work is dependent only on the start and end points of the path. Formally, for the time derivative function f we have that
f (x(t f )) -f (x(t i )) = C J ds = t f ti J(c(r))c ′ (r) dr,(2)
where C is a piecewise smooth curve in R n and c : [t i , t f ] → C is a parameterization of C with c(t i ) = x(t i ) and c(t f ) = x(t f ) (Figure 3A). Given estimates of f (x(t i )) and the Jacobian J, we can then use Equation 2 to approximate f (x(t f )) at any time t f . For these integrals, we use a line between the endpoints as a simple choice of path (Figure 3A). Then, to generate an estimate x(t f + ∆t) of the next step, we can use a standard ordinary differential equation (ODE) integrator (e.g., Euler, fourth-order Runge-Kutta, etc.) to integrate the estimated time derivative f (x(t)) (see Appendix A for more detail). To avoid the need to represent f directly (thereby enabling all gradients to backpropagate through the Jacobian network) we note that we parameterize an estimate of f (x(t i )) in Equation 2in terms of the Jacobian (see Appendix A.1).
this section cite: ['b90', 'b91']

Section: Loss functions
Trajectory reconstruction loss Given an observed trajectory x(t 0 + k∆t), k = 0, ..., T -1 of length T , we can compute the trajectory reconstruction loss, L traj (θ; x), between the true trajectory and the estimated trajectory using an appropriate distance measure d (e.g., mean squared error, see Appendix A.2 for more detail on generating predictions and the trajectory prediction loss).
this section cite: []

Section: Generalized Teacher Forcing
To avoid trajectory divergence in chaotic or noisy systems, we employ Generalized Teacher Forcing, generating recursive predictions partially guided by true states (Figure 3B, and Appendix D.8.1) [94].
this section cite: ['b93']

Section: Loop closure loss
The Jacobian captures how perturbations to the system will propagate along any direction in state space. Estimating it purely from dynamics constrains only the direction of the flow, leaving the full solution underdetermined. To address this, we again exploit the fact that each row of the Jacobian is a conservative vector field. Specifically, we note that for any piecewise smooth loop C loop , we have C loop Jds 2 = 0 (see Figure 3C). Thus, by integrating along loops that contain directions orthogonal to the system's dynamics (and penalizing the deviation from zero), we encourage the estimated Jacobians to capture information about other directions in state space (see Appendices A.3 and D.8.2 for full technical details). To ensure broad coverage of tangent space directions, we form loops from concatenations of line integrals between randomly selected data points. This strategy samples diverse directions from the tangent space while remaining easy to compute. The resulting self-supervised loss term, L loop (θ; x), builds on the loss introduced by Iyer et al. [95]. It constrains Ĵθ to satisfy both the dynamics and conservativity. This improves Jacobian estimation accuracy significantly (see Appendix C.4 for ablation studies).
this section cite: ['b94']

Section: Training loss
We therefore minimize the following loss function with respect to the parameters θ:
L(θ; x) = L traj (θ; x) + λ loop L loop (θ; x)(3)
where λ loop controls the relative weighting of the loop closure loss L loop (θ; x) compared to the trajectory prediction, and is a hyperparameter of the learning procedure (Figure 3D).
this section cite: []

Section: Jacobian estimation in dynamical systems
Data To evaluate the quality of the Jacobian estimation procedure, we apply our approach to several example systems for which the dynamics are known. For this analysis, we used the Van der Pol oscillator [96], Lorenz system [97], and the Lorenz 96 system across three different system sizes (12,32, and 64 dimensional) [98]. All systems were simulated using the dysts package, which samples dynamical systems with respect to the characteristic timescale τ of their Fourier spectrum [99,100]. All training data consisted of 26 trajectories of 12 periods, sampled at 100 time steps per τ . All models were trained on a 10 time-step prediction task with teacher forcing.
To evaluate the performance of the methods in the presence of noise, we trained the models on data with 1%, 5%, and 10% Gaussian observation noise added i.i.d over time, where the percentage is defined via the ratio of the euclidean norm of the noise to the mean euclidean norm of the data.
this section cite: ['b95', 'b96', 'b11', 'b31', 'b97', 'b98', 'b99']

Section: JacobianODE model
For the JacobianODE framework, loop closure loss weights were chosen via line search (Appendix D.8.8), where the neural network J θ was taken to be a four-layer multilayer perceptron (MLP) with hidden layer sizes 256, 1024, 2048 and 2048. Path integration was performed using the trapezoid method from the torchquad package, with each integral discretized into 20 steps [101]. ODE integration was performed using the fourth-order Runge-Kutta (RK4) method from the torchdiffeq package [102]. The JacobianODE models used 15 observed points to generate the initial estimate of f (see Section 3.2 and Appendices A.1 and A.2). All models were built in PyTorch [103]. Full implementation details are provided in appendices A and D.
Baselines We chose two different Jacobian estimation procedures for comparison. The first was a neural ordinary differential equation (NeuralODE) model trained to reproduce the dynamics [102]. The NeuralODE was implemented as a four-layer MLP with hidden layers of the same size as the one used for the JacobianODE model. Jacobians were computed via automatic differentiation. NeuralODEs were regularized via a penalty on the Frobenius norm of the estimated Jacobians to prevent the model from learning unnecessarily large negative eigenvalues (see Appendix D.3) [104][105][106]. We also employed a baseline that estimates Jacobian via a weighted linear regression, which computes locally linear models at each point in the space [88,89] (see Appendix D.4). Performance We tested the approaches on Jacobian estimation on held-out trajectories without noise. The JacobianODE method outperforms the baseline methods in terms of mean Frobenius norm error for virtually all systems (Table 1). This was also true when considering the spectral matrix 2-norm (see Table S3 in Appendix C.3).
We plot performance in Figure 4. While both the JacobianODEs and the NeuralODEs reproduce the observed dynamics (Figure 4A-D), the JacobianODE learns a more accurate estimate of the Jacobian (Figure 4 E,F). In particular, looking at Lyapunov spectra learned by the models, we note that the JacobianODE exceeds the other methods in estimating the full Lyapunov exponent spectrum, indicating a better overall representation of how perturbations along different directions will evolve (Figure 4G,H). 5 Control-theoretic analyses in a task-trained RNN
this section cite: ['b100', 'b101', 'b102', 'b101', 'b103', 'b104', 'b105', 'b87', 'b88']

Section: Characterizing control between subsystems with Jacobian linearization
Consider neural data recorded from two areas, A and B. Concatenating their data into a state vector x ∈ R n , composed of x A ∈ R n A and x B ∈ R n B , and assuming dynamics governed by f , we linearize around a reference trajectory (δ ẋ(t) = J(x(t)) δx(t)). Splitting the Jacobian into block matrices yields:
δ ẋA (t) = J A→A (x(t)) δx A + J B→A (x(t)) δx B δ ẋB (t) = J A→B (x(t)) δx A + J B→B (x(t)) δx B (4
)
where diagonal blocks J A→A ∈ R n A ×n A and J B→B ∈ R n B ×n B represent within-area dynamics, and off-diagonal blocks J B→A ∈ R n A ×n B , J A→B ∈ R n B ×n A represent interareal interactions (Figure 2, right). Explicit separation of each area's control dynamics quantifies the direct influence each exerts on the other, and readily generalizes beyond two areas (see Appendix B.3).
Since Jacobians are time-dependent, we obtain a linear time-varying representation of control dynamics along the trajectory. This enables computation of time-varying reachability ease, capturing how readily each area drives the other toward novel states [107][108][109]. Reachability is quantified via the reachability Gramian, a matrix defining a local metric in tangent space. For the above control system capturing the influence of area B on area A, the time-varying reachability Gramian on the interval [t 0 , t 1 ] is defined as
W r (t 0 , t 1 ) ≜ t1 t0 Φ(t 1 , τ )B(τ )B T (τ )Φ T (t 1 , τ )dτ (5
)
where Φ (computed from J A→A ) denotes the state-transition matrix of the intrinsic dynamics of subsystem A without any input (i.e., δx A (t) = Φ(t, t 0 )δx A (t 0 )), and B(τ ) = J B→A (x(τ )) (see Appendix B.1) [107]. The Gramian W r (t 0 , t 1 ) is symmetric and positive semidefinite for every t 1 > t 0 [107]. Each eigenvalue of the reachability Gramian quantifies how easily the target system can be driven along its corresponding eigenvector. Thus, the trace of the reachability Gramian reflects average ease of reaching new states, and its minimum eigenvalue reflects the ease along the most challenging direction of control.
this section cite: ['b106', 'b107', 'b108', 'b106', 'b106']

Section: Estimating the Jacobian of a task-trained RNN
Task To demonstrate how JacobianODE models could be used in neuroscience, we performed a control-theoretic analysis of a task-trained RNN. We used a working memory selection task from Panichello and Buschman [110] (Figure 5A). On each trial, the network is presented with two of four possible "colors", denoted by one-hot vectors. After a delay, the network is presented with a cue indicating which of the colors to select. The network is then asked to reach a state of sustained activation that corresponds to the selected color.
RNN model To perform this task, we trained a 128-dimensional continuous-time RNN. The RNN had hidden dynamics defined by
τ ḣ(t) = -h + W hh σ(h(t)) + W hi u(t) + b o(t) = W oh h(t)(6)
where W hh defines the internal dynamics, W hi maps the input into the hidden state, W oh maps the hidden state to a four-dimensional output o(t), b is a bias term, and σ is the exponential linear unit activation. The RNN had two 64-neuron areas: a "visual" area (which received sensory input) and a "cognitive" (which output the RNN's color choice) (Figure 5B). To encourage multi-area structure, we initialized the within-area weights with greater connectivity strength than the across-area weights (Figure 5C). Since input comes only to the visual area and output only from the cognitive area, the two areas are forced to interact to solve the task (Figure 5D,E). The RNN solves this task with 100% accuracy.
this section cite: ['b109']

Section: Jacobian reconstruction quality
We trained JacobianODEs on RNN trajectories from the post-cue delay and response portion of the trials. We used the same baselines as in Section 4. JacobianODEs are not only robust to noise, but can also benefit from it (for multiple systems in Table 1, 5% training noise improves estimation). Noise encourages the model to explore how perturbations around the observed trajectory evolve, which is crucial for learning accurate Jacobians in high-dimensional systems. Thus (for both the JacobianODE and NeuralODE) we add a small amount of additional noise to the data during learning. Although the models perform similarly on trajectory reconstruction (Figure 5F,G), on Jacobian estimation, JacobianODEs drastically outperform both baseline models (Figure 5H,I, and Table 1).
this section cite: []

Section: Reachability in the task-trained RNN across contexts
Next, we used the Jacobians learned by the JacobianODE (trained on 5% noise) to evaluate reachability control in the RNN, and compared it to evaluations using ground truth Jacobians. All analyses were performed using the 10 time-step reachability Gramian. We first found that it was on average easier for the visual area to drive the cognitive area, both when considering overall ease (Gramian trace) and worst-case ease (Gramian minimum eigenvalue) (Figure 6A, larger values indicate greater reachability). We next considered how reachability ease varied throughout the delay period. We found that the visual area could drive the cognitive area more easily at the beginning of the delay, with ease decreasing into the middle of the delay period (Figure 6B, bottom). The cognitive area was able to drive the visual area more easily slightly later in the delay period (Figure 6B, top). Finally, we considered whether reachability changes over the course of learning. We found that both directions of reachability increased after learning (Figure 6C). The JacobianODE reproduced all results accurately (Figure 6A-C, comparison with ground truth). Our analysis reveals that reachability is crucial in the RNN's ability to perform the working memory task, with the visual area's ability to drive the cognitive area shortly after the cue being especially important.
this section cite: []

Section: Controlling the task-trained RNN
To further validate our approach, we used JacobianODE-learned Jacobians to control the task-trained RNN (Figure 6D). Given a trial in which the network was cued to respond with a particular color, we tested if we could induce a specific incorrect response by input to the visual area alone. To do so, we implemented Iterative Linear Quadratic Regulator (ILQR) control, which relies on knowledge of the Jacobian [111,112]. The controller guided the network towards the mean hidden state of training trials corresponding to the desired incorrect color. We defined accuracy as the percentage of time points during which the RNN output the desired color. We found that the JacobianODE widely outperformed the baseline models on this task, achieving an accuracy nearing that of the ground truth system (Figure 6E). The JacobianODE was furthermore able to achieve the lowest mean squared error on the desired control trajectory (Figure 6F). This illustrates that while both the JacobianODE and the NeuralODE can learn the dynamics, only the JacobianODE learns a representation of the Jacobian that is sufficient for control.
this section cite: ['b110', 'b111']

Section: Discussion
Extended Jacobian estimation A natural extension of JacobianODE models would add inductive biases for particular classes of dynamics. For example, one could parameterize a negative definite matrix and thus ensure contracting dynamics [92]. Other extensions could include an L1 penalty to encourage sparsity, as well as the inclusion of known dynamic structure (e.g., hierarchical structure, low-rank structure, etc.). In neuroscience, connectomic constraints could be incorporated to capture interactions within a neural circuit.
Limitations A future challenge for JacobianODE models is partially observed dynamics. Recent work has identified that it is possible to learn latent embeddings that approximately recover the true state from partial observation [113][114][115][116][117]. Jacobian-based dynamics learning has been performed in a latent space [92,118], however it is unclear whether this translates to accurate Jacobian estimation, given the challenges related to automatic differentiation and Jacobian estimation presented here. We also note that reachability estimates depend sensitively on several factors: the alignment of cross-subsystem interactions and within-subsystem dynamics, the eigenvectors and eigenvalues of within-subsystem Jacobians, and the way activity propagates within each subsystem (see Appendix C.1). While our method reliably captures broad trends in reachability over time, fine-grained, timepoint-specific comparisons should be interpreted with caution. Finally, although JacobianODE models scale well to moderately high-dimensional systems, their performance in systems that are orders of magnitude larger than those considered here (e.g., recordings of thousands of neurons via calcium imaging) remains to be tested. In many practical settings, this may not be a limitation: neural representations during tasks often exhibit intrinsic dimensionalities comparable to those studied here, enabling JacobianODE models to operate in reduced dimensionality (see Appendix C.7). Notably, most models converged in under 45 minutes on a single GPU, suggesting favorable scaling (see Table S7). Future work should explore the viability of the method in higher dimensions, and assess whether dimensionality reduction strategies (such as latent state models or low-rank Jacobian approximations) can further improve scalability and accuracy.
this section cite: ['b91', 'b112', 'b113', 'b114', 'b115', 'b116', 'b91', 'b117']

Section: References
Ref_id:b0 Title: Frequency of gamma oscillations routes flow of information in the hippocampus Year: (2009-11)
Ref_id:b1 Title: Distributed and dynamical communication: a mechanism for flexible cortico-cortical interactions and its functional roles in visual attention Year: (2024-05-08)
Ref_id:b2 Title: Towards a neurobiological theory of consciousness Year: (1990)
Ref_id:b3 Title: Chapter 3 -neuronal oscillations, coherence, and consciousness Year: (2016-01-01)
Ref_id:b4 Title: Radical embodiment: neural dynamics and consciousness Year: (2001-10-01)
Ref_id:b5 Title: Synchronous neural oscillations and cognitive processes Year: (2003-12)
Ref_id:b6 Title: Spectral fingerprints of large-scale neuronal interactions Year: (2012-01-11)
Ref_id:b7 Title: New vistas for alpha-frequency band oscillations Year: (2007-04)
Ref_id:b8 Title: Top-down versus bottom-up control of attention in the prefrontal and posterior parietal cortices Year: (2007-03-30)
Ref_id:b9 Title: High-frequency, long-range coupling between prefrontal and visual cortex during attention Year: (2009-05-29)
Ref_id:b10 Title: Cell-type-specific synchronization of neural activity in FEF with V4 during attention Year: (2012-02-09)
Ref_id:b11 Title: Correlated neuronal activity and the flow of neural information Year: (2001-08)
Ref_id:b12 Title: Modulation of oscillatory neuronal synchronization by selective visual attention Year: (2001-02-23)
Ref_id:b13 Title: Neuronal synchronization along the dorsal visual pathway reflects the focus of spatial attention Year: (2008-11-26)
Ref_id:b14 Title: Top-down coordination of local cortical state during selective attention Year: (2021-03-03)
Ref_id:b15 Title: Oscillatory responses in cat visual cortex exhibit inter-columnar synchronization which reflects global stimulus properties Year: (1989-03-23)
Ref_id:b16 Title: Cortical network dynamics of perceptual decision-making in the human brain Year: (2011-02-28)
Ref_id:b17 Title: Cortical information flow during flexible sensorimotor decisions Year: (2015-06-19)
Ref_id:b18 Title: Interhemispheric transfer of working memories Year: (2021-03-17)
Ref_id:b19 Title: Content-specific fronto-parietal synchronization during visual working memory Year: (2012-11-23)
Ref_id:b20 Title: Temporal binding and the neural correlates of sensory awareness Year: (2001-01-01)
Ref_id:b21 Title: Visual feature integration and the temporal correlation hypothesis Year: (1995)
Ref_id:b22 Title: Oscillatory activity in sensorimotor cortex of awake monkeys: synchronization of local field potentials and relation to behavior Year: (1996-12)
Ref_id:b23 Title: Thalamic control of cortical dynamics in a model of flexible motor sequencing Year: (2021-06-01)
Ref_id:b24 Title: Primary motor and sensory cortical areas communicate via spatiotemporally coordinated networks at multiple frequencies Year: (2016-05-03)
Ref_id:b25 Title: A neural population mechanism for rapid learning Year: (2018-11-21)
Ref_id:b26 Title: Cortical activity in the null space: permitting preparation without movement Year: (2014-03)
Ref_id:b27 Title: Frequency-specific hippocampal-prefrontal interactions during associative learning Year: (2015-04)
Ref_id:b28 Title: Theta rhythms coordinate hippocampal-prefrontal interactions in a spatial memory task Year: (2005-12)
Ref_id:b29 Title: Coordinated interactions between hippocampal ripples and cortical spindles during slow-wave sleep Year: (1998-11)
Ref_id:b30 Title: Gamma rhythm communication between entorhinal cortex and dentate gyrus neuronal assemblies Year: (2021-04-02)
Ref_id:b31 Title: Increases in functional connectivity between prefrontal cortex and striatum during category learning Year: (2014-07-02)
Ref_id:b32 Title: Cortical areas interact through a communication subspace Year: (2019-04-03)
Ref_id:b33 Title: Statistical methods for dissecting interactions between brain areas Year: (2020-12)
Ref_id:b34 Title: Multiplexed subspaces route neural activity across brain-wide networks Year: (2023-02-12)
Ref_id:b35 Title: Controllability of structural brain networks Year: (2015-10-01)
Ref_id:b36 Title: Neuroscience out of control: control-theoretic perspectives on neural circuit dynamics Year: (2019-10)
Ref_id:b37 Title: Network neuroscience Year: (2017-02-23)
Ref_id:b38 Title: Optimal anticipatory control as a theory of motor preparation: A thalamo-cortical circuit model Year: (2021-05-05)
Ref_id:b39 Title: RNNs of RNNs: Recursive construction of stable assemblies of recurrent neural networks Year: (2022)
Ref_id:b40 Title: iLQR-VAE : control-based learning of input-driven dynamics with applications to neural data Year: (2021-10-06)
Ref_id:b41 Title: Stimulation-based control of dynamic brain networks Year: (2016-09)
Ref_id:b42 Title: Feedback controllability is a normative theory of neural population dynamics Year: (2024-03-29)
Ref_id:b43 Title: Brain network dynamics during working memory are modulated by dopamine and diminished in schizophrenia Year: (2021-06-09)
Ref_id:b44 Title: Mindful attention promotes control of brain network dynamics for self-regulation and discontinues the past from the present Year: (2023)
Ref_id:b45 Title: Structural control energy of resting-state functional brain states reveals less cost-effective brain dynamics in psychosis vulnerability Year: (2021-05)
Ref_id:b46 Title: Uncovering the biological basis of control energy: Structural and metabolic correlates of energy inefficiency in temporal lobe epilepsy Year: (2022)
Ref_id:b47 Title: Receptor-informed network control theory links LSD and psilocybin to a flattening of the brain's control energy landscape Year: (2022-10-03)
Ref_id:b48 Title: Network controllability in the inferior frontal gyrus relates to controlled language variability and susceptibility to TMS Year: (2018-07-11)
Ref_id:b49 Title: Development of the brain network control theory and its implications. Psychoradiology, 4:kkae028 Year: (2024-12-14)
Ref_id:b50 Title: Dynamic causal brain circuits during working memory and their functional controllability Year: (2021-06-29)
Ref_id:b51 Title: Controllability of functional and structural brain networks Year: (2024-01-01)
Ref_id:b52 Title: Controllability of functional brain networks and its clinical significance in first-episode schizophrenia Year: (2023-05-03)
Ref_id:b53 Title: Controllability of complex networks Year: (2011-05-12)
Ref_id:b54 Title: A network control theory pipeline for studying the dynamics of the structural connectome Year: (2024-12-29)
Ref_id:b55 Title: Geometric jacobian linearization and LQR theory Year: (2010)
Ref_id:b56 Title: Identification of interacting neural populations: methods and statistical considerations Year: (2023-09-01)
Ref_id:b57 Title: Inferring brain-wide interactions using data-constrained recurrent neural network models Year: (2021-03-11)
Ref_id:b58 Title: Rethinking brain-wide interactions through multi-region 'network of networks' models Year: (2020-12)
Ref_id:b59 Title: Neuronal dynamics regulating brain and behavioral state transitions Year: (2019-05-02)
Ref_id:b60 Title: Task-dependent changes in the large-scale dynamics and necessity of cortical regions Year: (2019-11-20)
Ref_id:b61 Title: A mechanistic multiarea recurrent network model of decision-making Year: (2021-12-06)
Ref_id:b62 Title: Early selection of task-relevant features through population gating Year: (2023-10-27)
Ref_id:b63 Title: Disentangling the flow of signals between populations of neurons Year: (2022-08-18)
Ref_id:b64 Title: Uncovering motifs of concurrent signaling across multiple neuronal populations Year: (2023-11-02)
Ref_id:b65 Title: Emergent reliability in sensory cortical coding and inter-area communication Year: (2022-05)
Ref_id:b66 Title: Detecting multivariate cross-correlation between brain regions Year: (1962)
Ref_id:b67 Title: Feedforward and feedback interactions between visual cortical areas use different population activity patterns Year: (2022-03-01)
Ref_id:b68 Title: Distinguishing time-delayed causal interactions using convergent cross mapping Year: (2015-10-05)
Ref_id:b69 Title: Untangling brain-wide dynamics in consciousness by cross-embedding Year: (2015-11)
Ref_id:b70 Title: Detecting causality in complex ecosystems Year: (2012-10-26)
Ref_id:b71 Title: Recurrent switching dynamical systems models for multiple interacting neural populations Year: (2020-10-22)
Ref_id:b72 Title: Modeling communication and switching nonlinear dynamics in multi-region neural activity Year: (2022-09-15)
Ref_id:b73 Title: Wiener-granger causality: a well established methodology Year: (2011-09-15)
Ref_id:b74 Title: Granger causality analysis in neuroscience and neuroimaging Year: (2015-02-25)
Ref_id:b75 Title: Dynamic causal modelling Year: (2003-08)
Ref_id:b76 Title: Population burst propagation across interacting areas of the brain Year: (2022-12-01)
Ref_id:b77 Title: Inferring the time-varying coupling of dynamical systems with temporal convolutional autoencoders. eLife Year: (2024-11-12)
Ref_id:b78 Title: Attention for causal relationship discovery from biological neural dynamics Year: (2023-11-12)
Ref_id:b79 Title:  Year: (1991)
Ref_id:b80 Title: Nonlinear systems and differential geometry Year: (1976)
Ref_id:b81 Title: Nonlinear controllability via lie theory Year: (1970-11)
Ref_id:b82 Title: Necessary conditions for controllability of nonlinear networked control systems Year: (2014-06)
Ref_id:b83 Title: Observability and controllability of nonlinear networks: The role of symmetry Year: (2015-01-23)
Ref_id:b84 Title: A geometrical approach to control and controllability of nonlinear dynamical networks Year: (2016-04-14)
Ref_id:b85 Title: Identifying controlling nodes in neuronal networks in different scales Year: (2012-07-27)
Ref_id:b86 Title: Input novelty as a control metric for time varying linear systems Year: (2014-11-21)
Ref_id:b87 Title: Tracking and forecasting ecosystem interactions in real time Year: (1822-01-13)
Ref_id:b88 Title: Capturing the continuous complexity of behaviour in caenorhabditis elegans Year: (2020-10-05)
Ref_id:b89 Title: Estimating the jacobian matrix of an unknown multivariate function from sample values by means of a neural network Year: (2022-04)
Ref_id:b90 Title: Learning functions with structured jacobians Year: (2024)
Ref_id:b91 Title: Neural contractive dynamical systems Year: (2024-01-17)
Ref_id:b92 Title: On contraction analysis for non-linear systems Year: (1998-06-01)
Ref_id:b93 Title: Generalized teacher forcing for learning chaotic dynamics Year: (2023-06-07)
Ref_id:b94 Title: Flexible mapping of abstract domains by grid cells via self-supervised extraction and projection of generalized velocity signals Year: (2024)
Ref_id:b95 Title: LXXXVIII. on "relaxation-oscillations Year: (1926-11-01)
Ref_id:b96 Title: Deterministic nonperiodic flow Year: (1963-03-01)
Ref_id:b97 Title: Predictability: a problem partly solved Year: (1995-09-08)
Ref_id:b98 Title: Chaos as an interpretable benchmark for forecasting and data-driven modelling Year: (2021)
Ref_id:b99 Title: Model scale versus domain knowledge in statistical forecasting of chaotic systems Year: (2023-12-15)
Ref_id:b100 Title: torchquad: Numerical Integration in Arbitrary Dimensions with PyTorch Year: ()
Ref_id:b101 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b102 Title: PyTorch: An imperative style, high-performance deep learning library Year: (2019-12-03)
Ref_id:b103 Title: Robust learning with jacobian regularization Year: (2019-08-07)
Ref_id:b104 Title: Stabilizing machine learning prediction of dynamics: Novel noise-inspired regularization tested with reservoir computing Year: (2024-02-01)
Ref_id:b105 Title: Time-series attribution maps with regularized contrastive learning Year: (2025-02-17)
Ref_id:b106 Title:  Year: (2006)
Ref_id:b107 Title: Empirical differential gramians for nonlinear model reduction Year: (2021)
Ref_id:b108 Title: Minimum energy control for complex networks Year: (2018-02-16)
Ref_id:b109 Title: Shared mechanisms underlie the control of working memory and attention Year: (2021-03-31)
Ref_id:b110 Title: Iterative linear quadratic regulator design for nonlinear biological movement systems Year: (2004-01-01)
Ref_id:b111 Title: Synthesis and stabilization of complex behaviors through online trajectory optimization Year: (2012-10)
Ref_id:b112 Title: Learning latent dynamics for partially observed chaotic systems Year: (2020-10-20)
Ref_id:b113 Title: Deep reconstruction of strange attractors from time series Year: (2020-12-06)
Ref_id:b114 Title: Discovering sparse interpretable dynamics from partial observations Year: (2022-08-12)
Ref_id:b115 Title: Reconstruction, forecasting, and stability of chaotic dynamics from partial data Year: (2023-09-01)
Ref_id:b116 Title: Deep learning delay coordinate dynamics for chaotic attractors from partial observable data Year: (2023-03-30)
Ref_id:b117 Title: Learning neural contracting dynamics: Extended linearization and global guarantees Year: (2024-02-12)
Ref_id:b118 Title:  Year: (2014-07-29)
Ref_id:b119 Title: Practical bifurcation and stability analysis. Interdisciplinary applied mathematics Year: (2009-12-10)
Ref_id:b120 Title: On the computation of lyapunov exponents for continuous dynamical systems Year: (1997)
Ref_id:b121 Title: Computing lyapunov spectra with continuous gram -schmidt orthonormalization Year: (1997-09-01)
Ref_id:b122 Title: Investigation of human-robot interaction stability using lyapunov theory Year: (2008-05)
Ref_id:b123 Title: Control contraction metrics: Convex and intrinsic criteria for nonlinear feedback design Year: (2017-06)
Ref_id:b124 Title: Beyond convexity-contraction and global convergence of gradient descent Year: (2020)
Ref_id:b125 Title: Achieving stable dynamics in neural circuits Year: (2020)
Ref_id:b126 Title: Large motion control of mobile manipulators including vehicle suspension characteristics Year: (2002)
Ref_id:b127 Title: Modified transpose jacobian control of robotic systems Year: (2007-07-01)
Ref_id:b128 Title: Lagrangian jacobian inverse for nonholonomic robotic systems Year: (2015-12)
Ref_id:b129 Title: Autonomic and inflammatory consequences of posttraumatic stress disorder and the link to cardiovascular disease Year: (2015-08-15)
Ref_id:b130 Title: Autonomic nervous system dysfunction in psychiatric disorders and the impact of psychotropic medications: a systematic review and meta-analysis Year: (2016-03)
Ref_id:b131 Title: Autonomic nervous system dysfunction: JACC focus seminar Year: (2019-03-19)
Ref_id:b132 Title: Autonomic dysfunction in neurological disorders Year: (1903)
Ref_id:b133 Title: Network medicine: a networkbased approach to human disease Year: (2011-01)
Ref_id:b134 Title: Systematic localization of common disease-associated variation in regulatory DNA Year: (2012-09-07)
Ref_id:b135 Title: Gene regulatory network inference: evaluation and application to ovarian cancer allows the prioritization of drug targets Year: (2012-05-01)
Ref_id:b136 Title: Transcriptional regulation and its misregulation in disease Year: (2013-03-14)
Ref_id:b137 Title: Gene regulatory networks in disease and ageing Year: (2024-09-12)
Ref_id:b138 Title: Using machine learning to replicate chaotic attractors and calculate lyapunov exponents from data Year: (2017-12)
Ref_id:b139 Title: Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach Year: (2018-01-12)
Ref_id:b140 Title: Reservoir observers: Model-free inference of unmeasured variables in chaotic systems Year: (2017-04-05)
Ref_id:b141 Title: Using machine learning to assess short term causal dependence and infer network links Year: (2019-12-26)
Ref_id:b142 Title: The importance of mixed selectivity in complex cognitive tasks Year: (2013-05-30)
