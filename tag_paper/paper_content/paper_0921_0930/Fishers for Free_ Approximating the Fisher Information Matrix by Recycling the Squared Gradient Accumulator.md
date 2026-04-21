Title: Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator
Abstract: The diagonal of a model's Fisher Information Matrix (the "Fisher diagonal") has frequently been used as a way to measure parameter sensitivity. Typically, the Fisher diagonal is estimated via squared sampled gradients of the model's likelihood with respect to its parameters, averaged over a few hundred or thousand examples -a process which incurs nontrivial computational costs. At the same time, adaptive gradient methods like the ubiquitous Adam optimizer compute a moving average of the squared gradient over the course of training. This paper therefore explores whether an approximation of the Fisher diagonal can be obtained "for free" by recycling the squared gradient accumulator that has already been computed over the course of training. Through a comprehensive set of experiments covering five applications of the Fisher diagonal, we demonstrate that the "Squisher" (Squared gradient accumulator as an approximation of the Fisher) consistently performs similarly to the Fisher diagonal while outperforming baseline methods. Additionally, we clarify the exact differences between the Squisher and the Fisher diagonal and provide empirical quantification of their respective impact.

Section: Introduction
The Fisher Information Matrix (FIM, Fisher, 1922) is a fundamental concept in statistics, capturing how much information an observable random variable carries about an unknown parameter. In machine learning, the FIM has been widely used in optimization, particularly in Natural Gradient Descent (NGD, Amari, 1998). Unlike standard gradient-based methods, which update parameters using the Euclidean gradient, NGD leverages the geometry from 1 University of Toronto & Vector Institute. Correspondence to: YuXin Li <lyx.li@mail.utoronto.ca>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
the statistical manifold of likelihoods and scales updates according to the parameter space's curvature, informing us about the "steepness" or "flatness" of the objective function at a given point in the parameter space (Karakida et al., 2019). This has made the FIM a valuable tool for training neural networks, improving stability and convergence speed (Karakida & Osawa, 2020).
Beyond optimization, recent work has explored the use of the FIM's diagonal (the "Fisher diagonal") as a measure of parameter sensitivity (Ly et al., 2017), including applications in model sparsification (Theis et al., 2018), sparse training (Sung et al., 2021), task similarity measurement (Achille et al., 2019), continual learning (Kirkpatrick et al., 2017), and model merging (Matena & Raffel, 2022;Daheim et al., 2024). These applications leverage the fact that the diagonal elements of the Fisher reflect each parameter's impact on the model output, when considering parameters to be independent, providing a principled approach to understanding and modifying neural nets.
Despite its utility, computing the Fisher diagonal introduces nontrivial computations beyond those for training. While these costs are typically comparable to training on a few hundred or a few thousand examples, the Fisher diagonal requires computing, squaring, then summing per-example gradients on sampled labels. Doing so efficiently is nontrivial in most deep learning frameworks. While there exist specialized solutions (Dangel et al., 2020;Osawa et al., 2023), practitioners often resort to sequential gradient computations (i.e. with a "batch size of one"), which sacrifices parallelization opportunities. Additionally, computing the Fisher diagonal requires access to training data that might not be available -for example, trained models are frequently released without their associated training data. We suspect that these factors can hinder the adoption of methods that require computing the Fisher diagonal -for example, Fisher Merging (Matena & Raffel, 2022) is not implemented in the ubiquitous mergekit library (Goddard et al., 2024), as it does not support merging methods that require computing gradients or accessing external data.
At the same time, modern neural networks are typically trained using adaptive gradient methods such as Adam (Kingma & Ba, 2014) that make use of an accumulator vari-Fishers for Free?
Standard Fisher 1 N N n=1 E p(y|xn,θ) ĝn ĝ⊤ n Empirical Fisher 1 N N n=1 gng ⊤ n Diagonal Fisher 1 N N n=1 g 2 n Joint Fisher 1 N E p(y|Xn,θ) (N ĝ)(N ĝ) ⊤ ĝ = 1 N n ĝn Joint empirical Fisher 1 N (N g)(N g) ⊤ g = 1 N n gn Joint diagonal Fisher 1 N N n=1 gn 2 Squared grad. accumulators v (t) = αv (t-1) + (1 -α) 1 N N n=1 g (t) n 2 Squisher N v (
t) Undo scale Moving average Empirical likelihood Extract diagonal Joint likelihood Square then sum
this section cite: ['b15', 'b14', 'b29', 'b51', 'b0', 'b17', 'b33', 'b2', 'b3', 'b35', 'b33', 'b8', 'b16']

Section: Sum then square
Figure 1. The arrows represent various approximations of the Fisher Information Matrix. The central idea of the paper is highlighted through the recycling symbol. Namely, we show that squared gradient accumulators can be used to approximate the Fisher diagonal (details in Section 2). Terms in blue stem from using a loss function with mean reduction ( 1 /NLsum), and should be omitted when using sum reduction (Lsum). The per-sample gradients gn and "would-be" gradients ĝn are defined in Equations ( 3) and ( 5).
able to store an exponential moving average of the squared gradient over training steps. The squared gradient accumulator is at least superficially similar to the Fisher diagonal in the sense that both compute some kind of average of squared gradients. The fact that such accumulator variables are available "for free" (i.e. without requiring any additional computation) at the end of training, combined with the aforementioned inconveniences of computing the Fisher, raises a natural question for applications that use the Fisher as a notion of parameter importance:
Can we recycle an optimizer's squared gradient accumulator to approximate the Fisher diagonal for free?
The superficial similarities of the Fisher diagonal and squared gradient accumulator suggest this could be trivially possible, and indeed, past work like the paper introducing Adam (Kingma & Ba, 2014) asserts without evidence that the squared gradient accumulator "is an approximation to the diagonal of the Fisher information matrix". However, a closer look reveals important subtleties. Our contribution is to rigorously assess and empirically validate these nuances:
• We discuss the non-trivial connections between the Fisher diagonal and squared gradient accumulators from adaptive optimizers, highlighting important modifications that are necessary to successfully use them in place of the Fisher (Section 2).
• We empirically validate the Squisher (squared gradient accumulator in place of the Fisher) across six settings spanning five key applications.We find that the Squisher achieves comparable performance to the Fisher diagonal and consistently outperforms the baseline across all settings (Section 3). This motivates the Squisher as a practical and efficient alternative that eliminates the computational costs and inconvenience of Fisher-based methods without sacrificing effectiveness.
• We investigate the Squisher's limitations and trade-offs, providing a deeper understanding of when and why it succeeds (Section 4).
On the whole, our work comprehensively establishes the connections between the squared gradient accumulator and the Fisher diagonal, paving the way for broader adoption of Fisher-based techniques in deep learning.
this section cite: ['b16']

Section: Background & Motivation
Here we provide background on, and connect, the Fisher diagonal and squared gradient accumulators (Figure 1).
this section cite: []

Section: The Fisher Information Matrix and its Variants
Standard Fisher Information Matrix For simplicity, we first consider an unscaled loss of the form
L sum (θ) := N n=1 ℓ(f (x n , θ), y n ),(1)
where f (•, θ) is a neural network with parameters θ that processes a data point x n into a prediction which is then scored by comparison with its label y n using a criterion function ℓ. Also define the per-sample gradient g n := ∇ θ ℓ(f (x n , θ), y n ). Later, we will add back the more common scaling and consider L(θ) = 1 /NL sum (θ). The most common loss functions in machine learning, like square or softmax cross-entropy loss, permit the loss in Equation ( 1) to be interpreted as a negative log likelihood for a random variable y representing a target (Martens, 2020), and we can define a per-example likelihood
p(y | x, θ) ∝ exp (-ℓ(f (x, θ), y)) .(2)
Note the relation to the per-sample gradient when y = y n and
x = x n -∇ θ log p(y n | x n , θ) (2) = g n .(3)
The standard Fisher Information Matrix (FIM) is
F std (θ) := N n=1 E ŷn∼p(y|xn,θ) ĝn ĝ⊤ n(4)
with "would-be" gradients
ĝn := -∇ θ log p(ŷ n | x n , θ) = ∇ θ ℓ(f (x n , θ), ŷn ) (5)
that require drawing labels ŷn from the model's likelihood.
this section cite: ['b32']

Section: Empirical Fisher Information
The empirical FIM replaces the model's likelihood with the empirical likelihood implied by the data, p(y | x n , θ) → δ(y -y n ). It consists of per-sample gradients which do not require sampling:
F emp std (θ) := N n=1 E ŷn∼δ(y-yn) ĝn ĝ⊤ n = N n=1 g n g ⊤ n . (6)
While past work has argued against using the empirical FIM in second-order methods like NGD (Kunstner et al., 2019;Thomas et al., 2020), it nevertheless remains popular in applications where the FIM is being used as a measure of parameter importance (e.g. Matena & Raffel, 2022;Theis et al., 2018;Sung et al., 2021;Achille et al., 2019, inter alia) due its lower computational costs. We therefore primarily focus on the empirical FIM in this work.
this section cite: ['b20', 'b52', 'b33', 'b51', 'b0']

Section: Diagonals
Our discussion so far was concerned with Fisher information matrices, which are quadratic in the number of model parameters and therefore prohibitively large in most deep learning applications. Hence, many applications only compute and use the diagonal of the FIM. Note that for a matrix that is a sum of rank one matrices, A = i v i v ⊤ i , which is the case for all the Fisher flavors we previously discussed, the diagonal is diag(A) = i v 2 i where the square is applied elementwise. This leads to simple expressions for the standard diagonal Fisher:
diag(F std (θ)) = N n=1 E ŷn∼p(y|xn,θ)
ĝ2 n and the empirical diagonal Fisher:
diag(F emp std (θ)) = N n=1 g 2 n (7
)
where the square is elementwise.
this section cite: []

Section: Gradient Accumulators
Training deep neural networks is an ill-conditioned and nonconvex optimization problem (Dauphin et al., 2014;Saarinen et al., 1993). The size of typical datasets used in neural network training also necessitates stochastic optimization where different randomly sampled batches of data are used at each training step (LeCun et al., 2002;Robbins & Monro, 1951). Consequently, most optimizers used in neural network training incorporate some mechanism to provide adaptivity to noise and differences in parameter sensitivity (i.e. differences in per-parameter gradient scale). One common approach is to approximate the average squared gradient for each parameter over the training steps. This squared gradient estimate can then be used e.g. to normalize each parameter update by a smoothed estimate of the gradient magnitude.
Specifically, we focus on optimizers with a squared gradient accumulator that takes the form of an exponential moving average (EMA, Roberts, 1959) of squared gradients:foot_0
v (t) = αv (t-1) + (1 -α) 1 N N n=1 g (t) n 2 (8
)
where α is a hyperparameter and g
n is the gradient vector for training example n at training step t. This accumulator mechanism, first introduced in 2012 through concurrent developments of RMSProp (Tieleman & Hinton, 2012) and Adadelta (Zeiler, 2012), has become a fundamental technique for adapting learning rates to stabilize training dynamics. Apart from being used in the ubiquitous Adam optimizer (Kingma & Ba, 2014) and its derivatives (AdamW (Loshchilov, 2017), NAdam (Dozat, 2016), RAdam (Liu et al., 2020), AdamP (Heo et al., 2021), etc.), it is also included in FTRL (McMahan & Streeter, 2012), AMSGrad (Reddi et al., 2018), QHM (Ma & Yarats, 2019), LAMB (You et al., 2020), and many others (Schmidt et al., 2021). In this work, we focus primarily on Adam and AdamW, which represent the most commonly used optimizers in contemporary deep learning research.
this section cite: ['b4', 'b43', 'b22', 'b40', 'b53', 'b62', 'b16', 'b27', 'b6', 'b25', 'b11', 'b34', 'b39', 'b30', 'b61', 'b44']

Section: The Squisher
Joint Fisher Note that both the standard and empirical FIMs in Equations ( 4) and (6) require per-datum (wouldbe) gradients to be squared then accumulated. This is in contrast to the squared gradient accumulator of Equation ( 8), which first aggregates the per-datum gradients, then squares them. Lin et al. (2024) proposed a new Fisher matrix which has a sum-then-square structure that is more compatible with the structure of squared gradient accumulators. Instead of modeling the label as single random variable y, they consider a likelihood for a random vector of labels y = (y 1 , y 2 , . . . , y N ) jointly from inputs X = (x 1 , . . . ,
x N ) via p(y | X, θ) = N n=1 p(y n | x n , θ) .
Their joint Fisher information matrix is
F joint (θ) := E ŷ∼p(y|Xn,θ) ĝ ĝ⊤(9)
with "would-be"
gradients ĝ = -∇ θ log p( ŷ | X, θ) = - N n=1 ∇ θ log p(ŷ n | x n , θ) = N n=1
ĝn and ŷn = [ ŷ] n . As for the standard case, their joint empirical Fisher information matrix follows by replacing the model's likelihood over the joint labels with the likelihood implied by the data, p(y | X, θ) → N n=1 δ(y n -y n ), which gives
F emp joint (θ) := E ŷ∼ N n=1 δ(y n -yn) ĝ ĝ⊤ = gg ⊤(10)
with the empirical gradient g = -
N n=1 ∇ θ log p(y n | x n , θ) = N n=1 ∇ θ ℓ(f (x n , θ), y n ) = ∇ θ L sum (θ). The diagonal joint empirical FIM follows as diag(F emp joint (θ)) = g 2 = N n=1 g n 2 .
The key properties of the joint Fisher matrices in Equations (9) and (10) is that they are squares of sums, and not sums of squares. This provides a theoretical motivation that the square of aggregated gradients indeed corresponds to the diagonal of a Fisher information matrix. In fact, Lin et al. (2024) show that the joint and standard Fisher coincide, F joint (θ) = F std (θ), and hence that both viewsstandard and joint-lead to the same underlying Fisher.
Handling mini-batching Note that the joint Fisher's distribution considers a vector of random variables whose size equals the data set. When using only a subset of data, say a mini-batch X B = (x 1 , . . . , x B ), we can consider the mini-batch version of the joint Fisher, F joint (θ, B), which is defined in terms of the marginal distribution p(y B | X B , θ) with y B = (y 1 , . . . , y B ). Lin et al. (2024) show that N /BF joint (θ, B) is an unbiased estimation of F joint (θ). This property allows us to estimate the Fisher on a larger data set (as would be done when using the standard Fisher in applications) from gradients evaluated on a smaller amount of data (as would be done by an optimizer based on minibatch gradients). Importantly, this estimation is unbiased if we sample labels from the model's likelihood. This ensures that mini-batching itself does not introduce bias in Fisher estimation. However, when labels are replaced with their empirical counterparts (e.g., ground truth labels), the resulting estimator becomes biased. Hence we can think of the standard and joint empirical Fishers as two different biased approximations of the same underlying Fisher. When using batches, we can simply replace N by B in all expressions that follow.
Handling averaged loss functions So far, we have assumed an unscaled loss (1) in our discussion of Fishers. However, most implementations use an average loss
L(θ) := 1 N N n=1 ℓ(f (x n , θ), y n ) = 1 N L sum (θ) . (11
)
Note that we cannot absorb the factor 1 /N into the loss function ℓ and define ℓ scaled = 1 /Nℓ to reduce Equation ( 11) to the form of Equation ( 1), because only ℓ, but not ℓ scaled , corresponds to a probability density via Equation (2). Therefore, we will keep the 1 /N factor separate and use the RHS of Equation ( 11), which means we can re-use the standard Fisher matrices that are defined in terms of the unscaled loss L sum and scale them by 1 /N. Rescaling the joint empirical Fisher matrix from Equation (10), we get
1 N gg ⊤ = 1 N N ∇ θ L(θ) (N ∇ θ L(θ)) ⊤ = N ∇ θ L(θ)(∇ θ L(θ)) ⊤ .
In the diagonal Fisher case, we have
N ∇ θ L(θ) 2 = N 1 N N n=1 g n 2 . (12
)
In practise, re-scaling is often unnecessary as many applications are invariant under scaling the Fisher (Section 3).
Recycling the squared gradient accumulator The term in the parenthesis on the RHS of Equation ( 12) is exactly the quantity whose EMA is computed by the squared gradient accumulator of Equation ( 8). This leads us to a clear path for using the Squisher, i.e. the squared gradient accumulator, as an approximation of the Fisher. Specifically, compared to the Fisher (i.e. the diagonal of the empirical FIM, as commonly used as a measure of parameter importance), the Squisher squares the average gradient over a training batch rather than computing the sum of gradients over an arbitrary collection of datapoints. The Squisher therefore more closely relates to the diagonal of the joint empirical FIM. In addition, the squaring of the average gradient introduces a factor of N difference in scale as in Equation ( 12).
Separately, the Squisher is computed using an EMA of minibatch gradients. The EMA coefficient α is typically tuned to improve training convergence and therefore might not reflect the best value for approximating the Fisher. For example, the default value of α in Adam (where it is referred to as β 2 ) is 0.999, which results in v (t) containing nontrivial contributions from a long history of gradients -the time constant, i.e. the number of steps to reach a rescaling of 1 -1 /e ≈ 63%, is about 10,000 steps. Averaging over such a long history both introduces contributions from gradients computed with respect to "old" parameter values and also results in a biased estimate of the corresponding joint empirical Fisher computed over all of the data the model has been trained on (Lin et al., 2024).
this section cite: []

Section: Experiments
The above discussion reveals a clear way to relate the squared gradient accumulator to a Fisher, but this relation involves various nontrivial approximations. We therefore turn to exploring whether these approximations are problematic through an empirical study covering a wide range of six settings where the Fisher is used, which we outline below. We emphasize that our goal is not to show that either the Fisher or the Squisher is "better" across all of these settings, but rather to test whether or not the approximations made in formulating the Squisher result in differences in performance compared to using the Fisher itself. To ensure reliable results, we base all of our experiments on prior implementations. In most cases, these implementations used either Adam or AdamW for optimization and therefore lent themselves straightforwardly to using the Squisher. We additionally always compare to a "Fisher-free" baseline, i.e. a method that does not involve computing the Fisher and therefore no additional computational costs (like the Squisher). Our high-level results are shown in Figure 2; complete fine-grained results are provided in Appendix A.1.
Naming: As the empirical diagonal standard Fisher from Equation ( 7) is the standard choice as a notion of parameter importance in many applications (Matena & Raffel, 2022;Theis et al., 2018;Sung et al., 2021;Achille et al., 2019), we will drop all prefixes and simply refer to it as the "Fisher" from now on.
this section cite: ['b33', 'b51', 'b0']

Section: Fisher Merging
Model merging aims to cheaply combine individual models into a single model that inherits their capabilities (Utans, 1996;Singh & Jaggi, 2020). Fisher merging (Matena & Raffel, 2022) formulates merging as maximizing the joint likelihood of the individual models' posterior distributions over parameters. To do so, Fisher merging uses the Laplace approximation (MacKay, 2003), where the Fisher is the precision matrix of a Gaussian approximation to this posterior. Fisher Merging then uses a closed-form solution to the likelihood maximizing problem:
θ = M i=1 F i -1 M i=1 F i θ i(13)
where F i and θ i are the Fisher and parameters of model i out of M models being merged. Equation ( 13) corresponds to parameter averaging where parameters with higher corresponding values in the Fisher are given a higher weight when averaging. Noting that Equation ( 13) is invariant to rescaling all F i by the same constant, using either the Squisher (N v (t) ) or the squared gradient accumulator (v (t) ) as Fisher proxies yields the same merge.
Setup We directly follow Tam et al. (2024) and merge eight variants of T5-Large-LM-Adapt (Raffel et al., 2020;Lester et al., 2021) that were fine-tuned on text datasets that have been shown to produce performant multitask models (Zhou et al., 2022). Performance is measured as the average accuracy of the merged model on held-out data from the eight datasets used to fine-tune the individual models. As Fisher-free baseline, we use simple parameter averaging (Utans, 1996;Wortsman et al., 2022). For further experimental details, please see Tam et al. (2024, Section 6.2).
Results Average performance of the Fisher, Squisher, and unweighted parameter averaging are shown in Figure 2. Overall, we found Squisher merging performed considerably better than Fisher merging. We don't interpret this to mean that the Squisher is "better", but rather that it is due to the inherent instability in multitask merging (Tam et al., 2024;Yadav et al., 2024;Ilharco et al., 2022). Since both the Fisher and Squisher work significantly better than the Fisher-free baseline, we conclude that both provide a reliable estimate of parameter importance. Additionally, we found that Squisher merging performance could suffer if the fine-tuned models were not fully trained, likely because the squared gradient accumulator had not observed sufficient training steps. We explore this factor further in Appendix A.2.
this section cite: ['b54', 'b48', 'b33', 'b31', 'b50', 'b38', 'b23', 'b64', 'b54', 'b59', 'b50', 'b60', 'b13']

Section: Model Merging by Uncertainty-Based Gradient Matching (UBGM)
Daheim et al. (2024) uncover that "gradient mismatch" arises when merging models that fall in disparate regions of the loss landscape, potentially leading merged models to fall in high-loss areas, thereby degrading merging performance. Daheim et al. (2024) therefore aim to mitigate gradient mismatch by aligning parameter updates with the optimization trajectories of the individual models. Specifically, given a base model with weights θ 0 and Fisher F 0 , and M fine-tuned models with weights θ i and Fishers F i for i ∈ {1, . . . , M }, the parameters of the merged model θ are given by
θ = θ 0 + F 0 + M i ′ =1 F i ′ -1 M i=1 (F 0 + F i )(θ i -θ 0 ) .
45 50 55 60 Average accuracy Fisher Merging 93 94 Average accuracy UBGM Merging 56 58 60 62 64 66 Accuracy Fisher Pruning 70 75 80 85 Average Accuracy FISH Mask 10 20 30 40 50 MRR (× 100) Task Embedding 55 60 65 70 75 80 Accuracy Continual Learning Fisher Squisher Fisher-free baseline As with Fisher merging, the Squisher simply replaces F i with the corresponding model's squared gradient accumulator at the end of training. Setup We exactly replicate the setup of Daheim et al. (2024) and consider the experiments focused on merging fine-tuned variants of RoBERTa (Liu et al., 2019) on four standard text classification tasks. As for Fisher merging, we measure performance in terms of average accuracy on the individual tasks. Daheim et al. (2024) use AdamW for finetuning, and we re-use their squared gradient accumulator without modification. Further experimental details are provided in Daheim et al. (2024, section 4.3). Like the previous merging setting, we use parameter averaging as baseline.
this section cite: ['b2']

Section: Results
High-level results are shown in Figure 2. In the UBGM Merging setting, we found that the Squisher performed similarly to Fisher. Interestingly, parameter averaging is a relatively strong Fisher-free baseline in this setup, though both Fisher-and Squisher-based UBGM merging outperform it slightly. Therefore, we consider their performance to be comparable in both settings.
this section cite: []

Section: Fisher Pruning
Pruning aims to convert a standard neural network into a "sparse" network where most of the weights are zero (LeCun et al., 1989;Hassibi & Stork, 1992). Theis et al. (2018) formulate pruning as eliminating the parameters that least impact the loss. Given a neural net trained to convergence at θ ⋆ , the optimal perturbation θ = θ ⋆ + δ(i), such that [ θ] i = 0 with minimal increase (ρ) of the loss is given by
δ(i) = -[θ] i F (θ ⋆ ) -1 e i [F (θ ⋆ ) -1 ] i,i , ρ(δ(i)) = [θ ⋆ ] 2 i 2[F (θ ⋆ ) -1 ] i,i
, with e i the canonical i-th basis vector. Assuming a diagonal Fisher produces the pruning statistics ρ(δ(i)) = [θ ⋆ ] 2 i [F (θ ⋆ )] i,i /2. Theis et al. (2018) then retain the parameters corresponding to the k leading pruning statistics.
Setup We base our Fisher pruning experiments on the re-implementation of Lubana et al. (2020). Specifically, we focus on an experiment described in Section 5.3 of Lubana et al. (2020), which involves pruning a VGG-13 (Simonyan, 2014) network trained on CIFAR-100 (Krizhevsky et al., 2009). Lubana et al. (2020) originally used vanilla stochastic gradient descent for training; we therefore modified the implementation to use Adam and tuned hyperparameters to ensure comparable results. Since Fisher pruning retains the top-k parameters, it is insensitive to global rescaling and we can therefore use the squared gradient accumulator as-is for the Squisher. As a Fisher-free baseline, we consider pruning parameters at random.
this section cite: ['b21', 'b10', 'b51', 'b51', 'b28', 'b28', 'b47', 'b19', 'b28']

Section: Results
We present the results for pruning 75% of the model parameters (i.e. reducing the model to 25% of its original size) in Figure 2 and include additional results for 25% and 50% in Appendix A.1. Across all pruning levels, we find that Squisher pruning slightly underperforms Fisher pruning but performs much better than baseline pruning with a random mask. et al. (2021) consider sparse training and fine-tuning, i.e. updating only a small subset of a model's parameters during training. They propose FISH Mask, which uses the Fisher to choose which parameters to update during training. Specifically, the k parameters to update are selected based on their importance measured by the Fisher diagonal:
this section cite: []

Section: FISH Mask

this section cite: []

Section: Sung
{[θ] i | [F (θ)] i,i ≥ sort(diag(F (θ))) k }.
With the Squisher, we instead update only those parameters with the k largest values in the squared gradient accumulator. As with Fisher pruning, we use random masking as Fisherfree baseline.
Setup We focus on the BERT-Large (Devlin et al., 2019) fine-tuning setting described in Section 4.1 of Sung et al. (2021), which uses the FISH mask to reset parameters of fine-tuned models back to the pre-trained values (i.e. their values before fine-tuning). Specifically, 50% of the model's weights are masked, and the remaining 50% are reset to their pre-trained values rather than retaining their fine-tuned values. Fine-tuned, masked models are separately trained and evaluated on nine datasets from the GLUE benchmark (Wang et al., 2018). Full details are available in Sung et al. (2021).
this section cite: ['b5', 'b58']

Section: Results
As shown in Figure 2, the Fisher and Squisher attain extremely similar performance when used in the FISH Mask setting. Combined with our previous finding for pruning, this result suggests that the squared gradient accumulator provides a reliable way to rank parameter importance.
this section cite: []

Section: Task Embedding
The Task2Vec embedding represents tasks as points in a vector space in which the distance aims to capture task similarity (Achille et al., 2019). Task2Vec embeddings are computed using the Fisher of a model trained on a given task and averaging the values across parameter "groups" (e.g. weight matrices). This vector approximates how sensitive different parameters are to some particular task. Task2Vec has been shown to be useful for predicting task similarities, such as semantic or taxonomic relations, and for determining which tasks are best suited for knowledge transfer (Vu et al., 2020), i.e. whether training on task a before training on task b can improve performance on task b. Specifically, the cosine similarity between task vectors F a and F b is
d sim (F a , F b ) = d cos (F a + F b ) -1 F a , (F a + F b ) -1 F b
and is used to predict task transferability, i.e. tasks with more similar task vectors are predicted as being more amenable to knowledge transfer. Since cosine distance is invariant to rescaling, to use the Fisher we simply replace F a and F b with the squared gradient accumulators from the corresponding models.
Setup We consider the setting from Vu et al. (2020), who use Task2Vec to predict task transferability for intermediatetask training (Phang et al., 2018;Pruksachatkun et al., 2020). Specifically, Vu et al. (2020) experiment with predicting the best intermediate task for fine-tuning BERT (Devlin et al., 2019) on a wide range of datasets. We focus on the 22 classification, regression, and question-answering datasets. Since the original fine-tuned models were not released by Vu et al. (2020), we re-fine-tuned BERT on each dataset using the AdamW optimizer. As a Fisher-free baseline, we use the simple and effective heuristic of ranking datasets in terms of their size (because larger datasets tend to be more beneficial for intermediate-task transfer (Vu et al., 2020)). We evaluate each ranking method based on its mean reciprocal rank (Voorhees et al., 1999) which measures how a given embedding method tends to rank the best dataset for intermediate-task transfer.
this section cite: ['b0', 'b57', 'b57', 'b36', 'b37', 'b57', 'b5', 'b57', 'b57', 'b56']

Section: Results
As seen in Figure 2, the Squisher-based task embedding produced a better mean reciprocal rank than the Fisher-based one; both outperformed the dataset size heuristic. This trend held across both classification/regression and question-answering datasets. This confirms that the squared gradient accumulator's values can be used as a reliable measure of task similarity.
this section cite: []

Section: Elastic Weight Consolidation
Continual learning faces the challenge of catastrophic forgetting, where artificial neural networks forget previously learned tasks when training on new tasks. Elastic Weight Consolidation (EWC, Kirkpatrick et al., 2017) aims to mitigate catastrophic forgetting by using the Fisher to avoid changes to model parameters that have a high influence on the performance of previously seen tasks. Specifically, EWC introduces a regularization term that rescales the squared difference between the current parameter value θ and the learned values from the previous task(s) θ by the Fisher F from the previous task:
L EWC (θ) = λ 2 (θ -θ) ⊤ F (θ -θ) .(14)
Setup We focus on task-incremental learning for this study, since EWC has been shown to have poor performance on domain-and class-incremental learning (van de Ven & Tolias, 2019). Task-incremental learning happens when the context identity is known during training, and the model must incrementally learn a set of distinct tasks (Ruvolo & Eaton, 2013). We consider three standard benchmarks for this setting: split MNIST, a split of the original MNIST dataset into five contexts with two digits each (Shin et al., 2017); permuted MNIST, an additional variant of MNIST transformed by applying a fixed, random pixel permutation to each task (Zenke et al., 2017); and split CI-FAR100, similarly split into ten contexts with ten classes each (Krizhevsky, 2009). We use an MLP with 478,410 parameters for split MNIST, a larger MLP with 2,126,100 parameters for Permuted MNIST, and a 5-layer CNN with 393,088 parameters for split CIFAR-100. For each protocol, we replaced the Fisher with the Squisher in the EWC regularizer.
Unlike in previous settings, rescaling the Squisher does change the learning behavior in this setting as it modifies the regularization strength. For EWC, we found that scaling the Squisher computed on batches of size B by N provided best performance, where N is the data set size on which the original Fisher was computed. Although we lack a formal theoretical justification, this serves as a useful heuristic, i.e., one can start with setting λ Squisher = N λ Fisher , and sweep over parameters around this value in a grid search. In practice, when training EWC from scratch, λ Fisher is unknown and must be tuned regardless, so using the Squisher does not introduce any additional tuning burden. We discuss the importance of adjusting the scaling in Section 4. As a Fisher-free baseline, we incrementally train the model without regularization.
this section cite: ['b17', 'b42', 'b46', 'b63', 'b18']

Section: Results
When used in EWC for continual learning, we find that the Squisher performs slightly better than the Fisher across all three continual learning setups (Figure 2 shows results for CIFAR100; results for other protocols are available in Appendix A.1). Using either the Squisher or the Fisher worked significantly better than using no parameterwise rescaling (i.e. the identity matrix instead of the Fisher in Equation ( 14)), suggesting again that the Squisher does capture a reliable notion of parameter importance.
this section cite: []

Section: Summary
Across all of the diverse experimental settings we consider, replacing the Fisher with the Squisher had little impact on performance. While performance degraded marginally in Fisher pruning, it improved in Fisher merging, and was almost identical for the other settings.
In all cases, both the Fisher and Squisher significantly outperformed relevant Fisher-free baselines. This validates that the approximations made when going from the Fisher to the Squisher do not significantly impact performance -at least when using diagonal approximations, as is common practise. More broadly, we can confirm that the Squisher meaningfully reflects the importance of a model's parameters to a similar extent to the Fisher.
We want to emphasize that our goal is not to claim that Squisher is inherently superior or inferior to Fisher. The performance differences are problem-dependent and influenced by training trajectories, but overall remain small relative to baseline performance. These differences are largely attributable to noise within the setting, rather than any fundamental distinction between the two approaches. This is further explained in Appendix A.2.
this section cite: []

Section: Runtime Analysis
The central motivation of this paper is to obtain Fishers for free. The time required to compute the Fisher varies with the model, dataset, and experimental setting; corresponding runtimes are reported in Table 1. In contrast, computing the Squisher incurs virtually no cost, as it simply involves loading pre-computed values and, in the case of EWC, applying a scaling factor. Across all settings, Squisher's runtime remains below 0.1 seconds, and under 1 second for EWC. A detailed breakdown of computation costs across datasets is provided in Appendix A.3.
this section cite: []

Section: Ablation Experiments
When relating the squared gradient accumulator to the Fisher in Section 2.3, we highlighted various approximations made. Although our results from Section 3 confirm that these approximations generally do not harm performance, we would still like to untangle each of their impacts.
To do so, we run a series of ablation experiments in the continual learning setting from Section 3.6. We chose continual learning as it was the only setting where results were dependent on appropriately rescaling the Squisher.foot_1
Squared gradient accumulator without rescaling To measure the importance of rescaling the squared gradient accumulator (as described in Section 2.3), we measure performance when using the squared gradient accumulator directly without tuning the λ term in Equation ( 14). As can be seen in Table 2, not tuning the value and using the default from Fisher provides suboptimal results.
Changing the exponential moving average From the results of Fisher merging, we observed that Squisher's performance could degrade if the fine-tuned models were not fully trained, likely because the squared gradient accumulator had not been exposed to enough training steps. We address this effect by reducing β 2 in Adam, which decreases the emphasis on past squared gradients and places more weight on recent gradients. The optimal setting aligns with the default AdamW value of 0.999, while performance degradation was observed at 0.95. Importantly, our intent is not to tune β 2 for Squisher; rather, we examine whether any standard optimizer hyperparameters can serve as drop-in approximations for the Fisher. We recommend using the β 2 that best supports optimization. Notably, even with unusually low values of β 2 , the Squisher continued to outperform the baseline, highlighting that Squisher remains effective as long as conventional training settings are used. Diagonal joint empirical FIM The squared gradient accumulator uses a moving average of gradients over the course of training, whereas FIMs compute an explicit sum of gradients at the end of training. To measure the importance of this difference, we explicitly square the sum of per-example gradients (i.e. we turn off the moving average in the Squisher), shown in Table 2 as JOINT. We see that its performance values are close to FISHER, suggesting the joint Fisher is a useful approximation.
this section cite: []

Section: Related Work
To the best of our knowledge, there has been no prior work investigating whether the squared gradient accumulator can reliably be used as a measure of parameter importance in place of the Fisher. However, there has been ongoing research on more directly connecting adaptive optimizers to second-order optimization. One such example is IVON (Shen et al., 2024), an extension of Adam that approximates the Hessian via weight perturbations. Notably, similar to our goal, IVON demonstrates that its accumulated statistics can serve as a substitute for the FIM in model merging tasks. However, unlike this work, using IVON's statistics involve a nontrivial deviation from what is currently standard practice for training neural networks.
Similarly, the AdaFisher optimizer (Gomes et al., 2024) replaces Adam's second-moment estimation with a novel block-diagonal approximation of the FIM. This approach yields improved performance, emphasizing the richness of the FIM compared to the squared gradient moving average. As with IVON and the Squisher, recycling the statistics produced by AdaFisher or FAdam could yield a similarly effective replacement for the Fisher.
FAdam is another optimizer that leverages the empirical FIM (Hwang, 2024). The authors reinterpret the secondorder moment in Adam through the lens of the diagonal Fisher, and propose a modified optimizer to address its limitations. While their work offers valuable insights into the relationship between natural gradient methods and adaptive optimization, it lacks a full theoretical proof and comprehensive empirical analysis.
BackPACK (Dangel et al., 2020) is a library that enables effi-cient computation of per-sample gradients and second-order quantities like the empirical Fisher diagonal by extending the backward pass of neural networks. While more efficient than naïve for-loop approaches, it still introduces additional overhead, requires code changes, and lacks full architectural support. For instance, it does not support layer normalization. As a result, the for-loop approach remains common in Fisher-based methods. In contrast, the Squisher introduces no extra cost or code modifications, as it reuses the squared gradient accumulator already present in optimizers like Adam (Kingma & Ba, 2014), making it universally supported without these limitations.
this section cite: ['b45', 'b9', 'b12', 'b3', 'b16']

Section: Conclusion
In this paper, we present a novel technique, the Squisher, that serves as a free approximation for the diagonal Fisher in various Fisher-based methods. The Squisher uses gradient accumulators that are readily available during training to provide an estimate of the Fisher, thereby alleviating the costs associated with calculating the Fisher. We motivate this by formulating the empirical Fisher as a joint Fisher, and hence show its parallels with gradient accumulators. We implement the Squisher as a direct replacement for the Fisher in settings where only the relative importance of parameters matters, and apply a rescaling term in contexts where the magnitude of values is consequential. We show that the Squisher had comparable performance to the Fisher across five settings, and had significant improvements from the baselines. We further discuss the impact of the approximations we made by considering several variants.
In future work, we can explore advanced optimizers which use gradient accumulators that approximate the nondiagonal Fisher, and whether that leads to better performance. Another direction is to maintain a moving average of the Fisher over time, analogous to the exponential moving average used in adaptive optimizers, rather than computing it at a single post-training point. Although impractical during real training, it may be interesting to explore whether it offers better approximations for the true Fisher. By reducing the computational burden of Fisher-based methods, this work advances the democratization of deep learning research. Furthermore, it is currently not common practice to share optimizer weights. However, we hope this will encourage the community to share full training configurations, hyperparameters, and optimizer state dicts, fostering more reproducible and inclusive machine learning research.
tute. We would like to thank Reviewer WT93 for helping us formalize the equivalence between the standard and joint Fisher. This clarification strengthens our motivation to interpret the standard empirical Fisher and Squisher as two distinct empirical approximations of the same underlying Fisher information.
this section cite: []

Section: References
Ref_id:b0 Title: Task2vec: Task embedding for meta-learning Year: (2019)
Ref_id:b1 Title: Natural gradient works efficiently in learning Year: (1998)
Ref_id:b2 Title: Model merging by uncertainty-based gradient matching Year: (2024)
Ref_id:b3 Title: BackPACK: Packing more into backprop Year: (2020)
Ref_id:b4 Title: Identifying and attacking the saddle point problem in high-dimensional non-convex optimization Year: (2014)
Ref_id:b5 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b6 Title: Incorporating nesterov momentum into adam Year: (2016)
Ref_id:b7 Title: On the mathematical foundations of theoretical statistics Year: (1922)
Ref_id:b8 Title: Arcee's mergekit: A toolkit for merging large language models Year: (2024)
Ref_id:b9 Title: Adaptive second order optimization via fisher information Year: (2024)
Ref_id:b10 Title: Second order derivatives for network pruning: Optimal brain surgeon Year: (1992)
Ref_id:b11 Title: Slowing down the slowdown for momentum optimizers on scale-invariant weights Year: (2021)
Ref_id:b12 Title: Adam is a natural gradient optimizer using diagonal empirical fisher information Year: (2024)
Ref_id:b13 Title: Editing models with task arithmetic Year: (2022)
Ref_id:b14 Title: Understanding approximate fisher information for fast convergence of natural gradient descent in wide neural networks Year: (2020)
Ref_id:b15 Title: Universal statistics of fisher information in deep neural networks: Mean field approach Year: (2019)
Ref_id:b16 Title: A method for stochastic optimization Year: (2014)
Ref_id:b17 Title: Overcoming catastrophic forgetting in neural networks Year: (2017)
Ref_id:b18 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b19 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b20 Title: Limitations of the empirical fisher approximation for natural gradient descent Year: (2019)
Ref_id:b21 Title: Optimal brain damage Year: (1989)
Ref_id:b22 Title: Efficient backprop Year: (2002)
Ref_id:b23 Title: The power of scale for parameter-efficient prompt tuning Year: (2021)
Ref_id:b24 Title: Can we remove the square-root in adaptive gradient methods? a second-order perspective Year: ()
Ref_id:b25 Title: On the variance of the adaptive learning rate and beyond Year: (2020)
Ref_id:b26 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b27 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b28 Title: Robust network pruning using orthonormality regularization Year: (2020)
Ref_id:b29 Title: A tutorial on fisher information Year: (2017)
Ref_id:b30 Title: Quasi-hyperbolic momentum and adam for deep learning Year: (2019)
Ref_id:b31 Title: Information theory, inference and learning algorithms Year: (2003)
Ref_id:b32 Title: New insights and perspectives on the natural gradient method Year: (2020)
Ref_id:b33 Title: Merging models with fisherweighted averaging Year: (2022)
Ref_id:b34 Title: Adaptive bound optimization for online learning and stochastic optimization Year: (2012)
Ref_id:b35 Title: Asdl: A unified interface for gradient preconditioning in pytorch Year: (2023)
Ref_id:b36 Title: Sentence encoders on stilts: Supplementary training on intermediate labeleddata tasks Year: (2018)
Ref_id:b37 Title: Intermediate-task transfer learning with pretrained models for natural language understanding Year: (2020)
Ref_id:b38 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b39 Title: On the convergence of adam and beyond Year: (2018)
Ref_id:b40 Title: A stochastic approximation method. The annals of mathematical statistics Year: (1951)
Ref_id:b41 Title: Control chart tests based on geometric moving averages Year: ()
Ref_id:b42 Title: ELLA: An efficient lifelong learning algorithm Year: (2013-06)
Ref_id:b43 Title: Ill-conditioning in neural network training problems Year: (1993)
Ref_id:b44 Title: Descending through a crowded valley-benchmarking deep learning optimizers Year: (2021)
Ref_id:b45 Title: Variational learning is effective for large deep networks Year: (2024)
Ref_id:b46 Title: Continual learning with deep generative replay Year: (2017)
Ref_id:b47 Title: Very deep convolutional networks for largescale image recognition Year: (2014)
Ref_id:b48 Title: Model fusion via optimal transport Year: (2020)
Ref_id:b49 Title: Training neural networks with fixed sparse masks Year: ()
Ref_id:b50 Title: Merging by matching models in task parameter subspaces Year: (2024)
Ref_id:b51 Title: Faster gaze prediction with dense networks and fisher pruning Year: (2018)
Ref_id:b52 Title: On the interplay between noise and curvature and its effect on optimization and generalization Year: (2020)
Ref_id:b53 Title: Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude Year: (2012)
Ref_id:b54 Title: Weight averaging for neural networks and local resampling schemes Year: (1996)
Ref_id:b55 Title: Three scenarios for continual learning Year: (2019)
Ref_id:b56 Title: The trec-8 question answering track evaluation Year: (1999)
Ref_id:b57 Title: Exploring and predicting transferability across nlp tasks Year: (2020)
Ref_id:b58 Title: Glue: A multi-task benchmark and analysis platform for natural language understanding Year: (2018)
Ref_id:b59 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022)
Ref_id:b60 Title: Ties-merging: Resolving interference when merging models Year: (2024)
Ref_id:b61 Title: Large batch optimization for deep learning: Training bert in 76 minutes Year: (2020)
Ref_id:b62 Title: Adadelta: an adaptive learning rate method Year: (2012)
Ref_id:b63 Title: Continual learning through synaptic intelligence Year: (2017)
Ref_id:b64 Title: Not all tasks are born equal: Understanding zero-shot generalization Year: (2022)
