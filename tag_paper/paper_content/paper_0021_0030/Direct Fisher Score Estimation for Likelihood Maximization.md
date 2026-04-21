Title: Direct Fisher Score Estimation for Likelihood Maximization
Abstract: We study the problem of likelihood maximization when the likelihood function is intractable but model simulations are readily available. We propose a sequential, gradient-based optimization method that directly models the Fisher score based on a local score matching technique which uses simulations from a localized region around each parameter iterate. By employing a linear parameterization for the surrogate score model, our technique admits a closed-form, least-squares solution. This approach yields a fast, flexible, and efficient approximation to the Fisher score, effectively smoothing the likelihood objective and mitigating the challenges posed by complex likelihood landscapes. We provide theoretical guarantees for our score estimator, including bounds on the bias introduced by the smoothing. Empirical results on a range of synthetic and real-world problems demonstrate the superior performance of our method compared to existing benchmarks.

Section: Introduction
Implicit simulator-based models are now routine in many scientific fields, such as biology [Csillery et al., 2010], cosmology [Schafer and Freeman, 2012], neuroscience [Sterratt et al., 2011], engineering [Bharti et al., 2021], and other scientific applications [Toni et al., 2009]. In traditional statistical models, there is a prescribed probabilistic model, which provides an explicit parameterization for the data distribution, allowing development of the likelihood function for further inference. In contrast, simulation-based models define the distribution implicitly through a computational simulator. Thus, while simulation of the data for various parameter settings is possible, the probability density function of the data or the likelihood function is often unavailable in closed form. This problem setting is known as likelihood-free inference or simulation-based inference (SBI) [Cranmer et al., 2020].
Traditionally, methods in this setting have focused on a Bayesian inference technique known as approximate Bayesian computation (ABC) [Beaumont et al., 2002]. Fundamentally, the ABC method builds an approximation to the Bayesian posterior distribution by drawing parameter samples from the prior distribution, generating datasets from the drawn parameter values, and filtering parameter values through a rejection algorithm based on the distance of the generated summary statistic of the dataset from the observations. In recent years, there has been a rise of generative modeling or unsupervised learning in machine learning, which aims to recover a data distribution given a set of samples. Such generative models are often built from neural networks [Mohamed and Lakshminarayanan, 2017], and given the fundamental similarity with the SBI problem setting, this has led to significant cross-pollination across the two fields, with the development of many SBI inference methods using generative neural networks [Durkan et al., 2018, Papamakarios et al., 2017].
While significant progress in SBI has come from Bayesian approaches, such methods are often computationally demanding; furthermore, many scientific disciplines retain a preference towards maximum likelihood approaches. In contrast, practitioners who favor faster point estimation through maximum likelihood lack comparably mature tools. To close this gap, we propose a fast, simulationefficient, and robust gradient-based technique for estimating the maximum likelihood for SBI. We build on a popular technique in generative modeling known as score matching [Hyvärinen and Dayan, 2005], which has seen significant use in score-based generative models [Song and Ermon, 2019], now a cornerstone for many state-of-the-art approaches in generative modeling [Yang et al., 2023]. We adapt score matching to estimate the Fisher score, that is, the gradient of the log-likelihood function with respect to the parameters, within a localized region. This estimated gradient can then be used in any first-order gradient-based stochastic optimization algorithm such as stochastic gradient descent (SGD) to obtain an approximate maximum likelihood estimator (MLE) and serves as a potential avenue for uncertainty quantification for the MLE through the empirical Fisher information matrix.
Our contributions in this work are as follows.
• We propose a lightweight, simulation-efficient, and robust method for maximum likelihood estimation of simulator models based on a novel local Fisher score matching technique
• We derive theory for our local Fisher score matching technique and establish a connection with the Gaussian smoothing gradient estimator, offering a unifying perspective for zerothorder optimization techniques and likelihood optimization for SBI
• We demonstrate the effectiveness of our method in real-world experiments for applied machine learning and cosmology problems, showcasing both its efficiency and robust performance compared to existing approaches 2 Background
this section cite: ['b7', 'b3', 'b39', 'b6', 'b1', 'b18', 'b10', 'b21', 'b12', 'b32', 'b43']

Section: Score Matching
Density estimation is the problem of learning a data distribution p D (x) using only an observed dataset, x ∼ p D . An approach to this problem is learning the density with an energy-based model (EBM), which parameterizes the model through its scalar-valued energy function E θ : R k → R where x ∈ X ⊆ R k , giving the model density p θ (x) = exp(-E θ (x))/Z θ .
Since the energy function is an unnormalized density function, it can be flexibly parameterized, usually through a neural network. However, note that the normalization constant Z θ = exp(-E θ (x))dx is still a function of θ, and therefore will still need to be computed in training the EBM through standard likelihood maximization. Since this multidimensional integral is often intractable and requires a costly approximation method, score matching [Hyvärinen and Dayan, 2005] is often used to bypass the computation of the normalization constant. This is done by considering an alternate training objective instead of MLE, based on the score function s θ : R k → R k , where s θ = ∇ x log p θ = -∇ x E θ . In fact, since equivalence in the score amounts to equivalence in the distribution, matching the scores is equivalent to performing density estimation. One starting point is the explicit score matching objective (ESM). Defining the gradient operator on a scalar-valued function as ∇ x := ( ∂ ∂x1 , . . . , ∂ ∂x k ) ⊤ , where ∂ ∂xi is the partial derivative operator for x = (x 1 , . . . , x k ), and the Jacobian operator on a vectorvalued function f : R m → R n as J i,j = [ ∂fi ∂xj ] i,j , we have:
L ESM (θ) = E x∼p D (x) 1 2 ∥s θ (x) -∇ x log p D (x)∥ 2
However, this objective is not tractable due to the need to evaluate ∇ x log p D (x). Hence, this objective is transformed to:
L ESM (θ) = E x∼p D (x) 1 2 ∥s θ (x)∥ 2 + tr (J x s θ (x)) + (constants w.r.t. θ)
Although this objective can be directly estimated, and thus optimized and used in the training of an EBM, it is computationally expensive due to the presence of the Jacobian term, motivating further extensions to the standard score matching objective, such as the denoising score matching objective [Vincent, 2011] and the sliced score matching objective [Song et al., 2020].
this section cite: ['b12', 'b41', 'b33']

Section: Maximum Likelihood Estimation and Fisher Score
Maximum likelihood estimation (MLE) is a foundational tool in statistical inference, under standard regularity conditions, it is consistent and asymptotically efficient [Casella and Berger, 2024, Section 10]. Central to the MLE is the Fisher score, defined as the gradient of the log-likelihood with respect to the parameters, ∇ θ log p(x | θ). From an optimization point of view, the score provides the direction of steepest ascent of the log-likelihood in parameter space, and thus drives gradient-based MLE approaches. From an inferential point of view, the covariance of the Fisher score is equal to the Fisher information matrix (FIM), which, through the Cramér-Rao lower bound [Rao, 1992], lower bounds the variance of any unbiased estimator. Furthermore, the distribution of the MLE is asymptotically normal with covariance equal to the inverse of the FIM, which underpins Wald-type confidence intervals and hypothesis tests [Van der Vaart, 2000, Section 5].
this section cite: ['b25']

Section: Notation and Problem Setup
We consider a statistical model where the data x ∈ X ⊂ R dx are generated from a distribution P θ parameterized by θ ∈ Ω ⊂ R d θ . In the simulation-based inference setting, this statistical model is implicitly defined, so we can draw samples from this model for any choice of θ but the closed-form expression for the probability density function, and hence the likelihood function is not known.
Given a set of N independent and identically distributed observations, D = {x i } N i=1 , drawn from the true data-generating process x i ∼ P θ * , where θ * denotes the true parameter, the maximum likelihood estimator is θMLE = arg max θ p(D | θ).
As the likelihood function L(θ; D) = N i=1 p(x i | θ) is not available for SBI models, typical likelihood maximization cannot be applied directly. We thus propose a Fisher score matching-based estimator, θFSM . Our method is fundamentally a first-order optimization approach, and our main focus is on the direct estimation of the gradient of the log-likelihood function at each parameter iteration, which is done with a novel local Fisher score matching objective. We first discuss our Fisher score estimation technique in Section 3, before proceeding with the MLE procedure in Section 4.
this section cite: []

Section: Likelihood-free Fisher Score Estimation
Score matching [Hyvärinen and Dayan, 2005] is a classical method in density estimation, but is not directly applicable in likelihood gradient maximization, as it typically targets the Stein score, i.e., the gradient with respect to the data ∇ x log p θ (x) instead of the Fisher score, which is the gradient with respect to the parameters ∇ θ log p θ (x). Hence, we propose to adapt score matching into a novel local Fisher score estimation technique which estimates the gradient of the log-likelihood for a fixed parameter point θ t at any data sample
x, ∇ θ ℓ(θ; x) θ=θt = ∇ θ log p x | θ θ=θt .
this section cite: ['b12']

Section: Local Fisher Score Matching Objective
Around the target parameter point θ t , we introduce a local proposal distribution q(θ | θ t ), which we typically take as an isotropic Gaussian distribution, q(θ | θ t ) = N (θ t , σ 2 I). When combined with the statistical model P θ , we induce a joint distribution in both the data and parameter space that has probability density p(x | θ)q(θ | θ t ). Note that by drawing parameter samples from the local proposal distribution and then drawing corresponding data samples for the parameter samples, we can easily draw samples from this joint distribution.
To estimate the score function, we use a score model S W : R dx → R d θ , where S W (x) has parameters W . Our starting point is the adapted, localized score matching least-squares loss for the Fisher score.
J (W ; θ t ) = E x∼p(x|θ),θ∼q(θ|θt) ∇ θ log p(x | θ) -S W (x) 2 (1)
As we are within the simulation-based inference framework, we do not have a closed form expression for the Fisher score ∇ θ log p(x | θ) and hence this objective function is not tractable. We first expand the square of Equation ( 1), which allows us to rewrite J (W ; θ t ) as:
J (W ; θ t ) = E x∼p(x|θ),θ∼q(θ|θt) S W (x) 2 -2 S W (x) ⊤ ∇ θ log p(x | θ) + (constants w.r.t. W )
We focus on the cross-term, E x∼p(x|θ),θ∼q(θ|θt) S W (x) ⊤ ∇ θ log p(x | θ) . Using an integration-byparts trick, this term can be transformed to -E x∼p(x|θ),θ∼q(θ|θt) S W (x) ⊤ ∇ θ log q(θ | θ t ) . Note that we have eliminated the dependence on the intractable likelihood function log p(x | θ). Thus, this allows us to rewrite J (W ; θ t ) as follows. Theorem 3.1 (Local Fisher Score Matching (FSM)). Let J (W ) be defined as in Equation (1). Under suitable boundary conditions, it can be rewritten (up to an additive constant w.r.t. W ) as
J (W ; θ t ) = E x∼p(x|θ),θ∼q(θ|θt) S W (x) 2 + 2 S W (x) ⊤ ∇ θ log q(θ | θ t )(2)
The complete details for Theorem 3.1 are provided in Appendix A.1. Given that we can draw proposal samples {θ (j) } m j=1 where θ (j) ∼ q(θ | θ t ) and corresponding data samples {x
(j) k } n k=1 where x (j) k ∼ p(x | θ (j)
), the objective J (W ; θ t ) can be approximated by Monte Carlo estimation.
Ĵ (W ; θ t ) = 1 m m j=1 1 n n k=1 S W (x (j) k ) 2 + 2 S W (x (j) k ) ⊤ ∇ θ log q(θ | θ t )| θ=θ (j)(3)
Next, we show the optimal solution for the local FSM objective, J (W ; θ t ). The proof of Theorem 3.2 in Appendix A.2. Theorem 3.2 (Bayes-optimal Local Fisher Score). The optimal score model for the FSM objective
J (W ; θ t ), is S * (x; θ t ) = E θ∼p(θ|x,θt) ∇ θ log p(x | θ)
As the score matching objective Equation (1) is taken as an expectation over the parameter proposal distribution q(θ | θ t ), the Bayes-optimal score model for this objective is generally biased and instead of being the true score at the point θ t , it is instead an average of the score over the posterior induced from the proposal distribution and the statistical model, that is, p(θ | x, θ t ). Thus, this score matching objective targets a smoothed likelihood around θ t . We elaborate on this in more detail in Section 5.1.
this section cite: []

Section: Score Model Parameterization
A key aspect of the Fisher score matching technique is the choice of parameterization for the surrogate score model, S W (x; θ t ), which approximates the Fisher score at the target parameter iterate θ t , ∇ θ log p θ (x)| θ=θt . For computational tractability, we propose using a lightweight linear surrogate score model based on the following derivation.
Let the surrogate score model be defined as S W (x; θ t ) = W ⊤ x, where W ∈ R dx×d θ is the weight matrix for our model. Recall that we first draw a set of parameters {θ (j) } m j=1 from the proposal distribution q(θ | θ t ). Then, define the j-th data matrix as X j ∈ R n×dx constructed from n training samples {x j) , and the j-th Gram matrix as G j = X ⊤ j X j . Using the linear score model for the local Fisher score matching objective function in Equation (3) and solving for the first-order conditions, we obtain the normal equation,
(j) k } n k=1 drawn from the model p(x | θ) at θ (
( m j=1 G j ) Ŵ = - m j=1 n k=1 x (j) k ∇ θ log q(θ | θ t )| ⊤ θ=θ (j) (4)
We can thus obtain a closed-form solution for the linear Fisher score matching estimator as:
Ŵ = -( m j=1 G j ) -1 m j=1 n k=1 x (j) k ∇ θ log q(θ | θ t )| ⊤ θ=θ (j) (5)
Once Ŵ is obtained, we can use this to construct our Fisher score estimator Ŝ(x; θ t ) = Ŵ ⊤ x. We provide a complete derivation and further discussion in the Appendix A.3. Although the local Fisher score matching objective is a general framework that is agnostic to the choice of parameterization of the model, using a linear model essentially recasts the model estimation procedure as multivariate linear regression, benefiting from well-understood theory and efficient implementations. Although a linear model might not be sufficient to fully capture the full data-parameter relationship, it provides a strong baseline that we find works well empirically compared to a more flexible neural network-based model, which incurs significant computational costs in the form of an inner optimization loop and increased variance. We provide empirical comparisons with the neural network-based score model in the relevant experimental sections of the appendix, and details of the implementation in Appendix A.4.
this section cite: []

Section: Likelihood-free MLE with Approximate Fisher Score
Using our local Fisher score matching (FSM) method as described in Section 3, we describe how maximum likelihood estimation (MLE) can be performed in the likelihood-free setting. Unlike many SBI methods that attempt to estimate the likelihood globally, our method is inherently sequential by focusing only on a local Fisher score estimation at the parameter point θ t .
Given a set of N independent and identically distributed observations, D = {x i } N i=1 , at a fixed parameter point θ t , we obtain an estimated FSM model Ŝ(x; θ t ) using training samples {θ (j) } m j=1 , {x j) ). As the FSM model is a function of x, we can evaluate it at any observation x i , providing us with an approximate gradient of the log-likelihood evaluated at θ t , ∇θ ℓ(θ t ; D) = N i=1 Ŝ(x i ; θ t ). This can then be used directly in any iterative stochastic gradient-based algorithm such as stochastic gradient descent (SGD) [Robbins and Monro, 1951], Adam [Kingma and Ba, 2015], or RMSProp [Tieleman, 2012], where at each parameter iteration θ t , a new FSM model Ŝ(x; θ t ) is estimated. The FSM-MLE algorithm with SGD is presented in Algorithm 1 Algorithm 1 FSM-MLE Algorithm (SGD) Input: N independent and identically distributed observations D = {x i } N i=1 , initial parameter θ 0 , step size η, and proposal distribution q(θ | θ t ) Initialize t ← 0 while t < T do 1. For current iterate θ t , sample {θ (j) } m j=1 from proposal distribution θ ∼ q(θ | θ t ), and then sample corresponding data samples {x
k } n k=1 , drawn from θ (j) ∼ q(θ | θ t ), x (j) k ∼ p(x | θ ((j)
(j) k } n k=1 , from x (j) k ∼ p(x | θ (j) ). 2. Estimate Fisher score model Ŝ(x; θ t ) using training samples ({θ (j) } m j=1 , {x (j) k } n k=1 ) 3. Set θ t+1 ← θ t + η Ŝ(D; θ t ), where Ŝ(D; θ t ) = N i=1 Ŝ(x i ; θ t ) 4. t ← t + 1 end while 4.1 Fisher Score Proposal Distribution
Our local FSM approach crucially uses a proposal distribution q(θ | θ t ) in the parameter space, defining a local region for the estimation of our Fisher score model. Although most distributions with a differentiable and unbounded density can be used, we use an isotropic Gaussian distribution q(θ | θ t ) = N (θ | θ t , σ 2 I), which has a simple, closed-form solution and direct theoretical interpretation as discussed in Section 5. This introduces a single scalar hyperparameter σ that controls the width of the proposal distribution. While further extensions such as a diagonal covariance matrix or an adaptive covariance could be explored, we keep to the isotropic Gaussian proposal distribution as it provides a simple and effective baseline. We provide further discussion on the choice of proposal distribution and a calibration scheme for σ in Appendix A.5.
this section cite: ['b26', 'b16']

Section: Theoretical Analysis
In this section, we provide a theoretical analysis of our proposed local Fisher score matching technique and the stochastic gradient optimization based on this technique.
this section cite: []

Section: Connection to Gaussian Smoothing
Gaussian smoothing is a popular zeroth-order optimization technique that estimates gradients using only function evaluations when the gradient function is not known [Nesterov andSpokoiny, 2017, Duchi et al., 2012]. As the Gaussian smoothing gradient estimator targets a smoothed function, it is widely applicable even for non-smooth functions, which would not be amenable with standard gradient estimation, and has been shown to be robust to local optima [Starnes et al., 2023] and applicable for many challenging machine learning problems [Salimans et al., 2017].
Although standard Gaussian smoothing is straightforward for black-box optimization problems, note that it is not directly applicable in the simulation-based inference setting as the intractable likelihood L(θ) = p(x | θ) is not explicitly accessible. Nonetheless, we show here that our proposed local Fisher score matching technique can be directly cast as a likelihood-free analogue of Gaussian smoothing. Specifically, under a Gaussian proposal distribution, q(θ | θ t ) = N (θ | θ t , σ 2 I), the Bayes-optimal Fisher score is exactly the gradient of a smoothed likelihood. We provide a full proof of Theorem 5.1 in Appendix A.6. Theorem 5.1 (Equivalence as Gaussian Smoothing). Under an isotropic Gaussian proposal, q(θ Observe that the smoothed likelihood can be further rewritten as
| θ t ) = N (θ | θ t , σ 2 I), the optimal FSM estimator is equivalent to the gradient of the smoothed likelihood ∇ θt l(θ t ; x) = E θ∼p(θ|x,θt) ∇ θ log p(x | θ) where l(θ t ; x) = log p x | θ q θ | θ t dθ and p(θ | x, θ t ) ∝ p(x | θ)q(θ | θ t ) is the induced posterior from the proposal distribution q(θ | θ t )2
l(θ t ; x) = log E z∼N (0,I) L(θ t + σ z; x) .
where L(θ; x) = p(x | θ). This is exactly the Gaussian-smoothed likelihood function, except importantly that explicit evaluations of the likelihood L(θ) were not used. Instead, our Fisher score matching technique only obtains samples from the model p(x | θ) for the FSM estimation. Hence, our method directly inherits many of the robustness benefits of Gaussian smoothing while still being applicable in the SBI setting.
Figure 1 demonstrates the effects of smoothing in a one-dimensional, shifted exponential likelihood model with a single observation. The true likelihood is zero for θ ≥ θMLE , and hence any gradient-based optimization which is initialized beyond the boundary will be stuck in that region. However, using our smoothed likelihood (depicted with differing values of proposal variance σ 2 ), we are able to obtain a non-zero gradient even outside the nominal support, allowing us to successfully optimize the likelihood function.
We can also further view the FSM procedure as a form of Empirical Bayes (EB) [Morris, 1983], by interpreting the proposal distribution q(θ | θ t ) as a local prior centered at θ t , which, together with the simulator model, defines an EB marginal likelihood function l(θ t ; x). Theorem 5.1 then shows that our Bayes-optimal FSM estimator is exactly the hyperparameter gradient of the EB marginal likelihood. Hence, this provides a complementary Bayesian interpretation of our FSM method in addition to the optimization viewpoint of Gaussian smoothing.
this section cite: ['b20', 'b35', 'b29', 'b19']

Section: Properties of the FSM estimator
We now provide theoretical guarantees for our FSM estimator under a Gaussian proposal distribution by characterizing its bias. In particular, by establishing the bias in terms of the smoothing hyperparameter σ, we highlight a fundamental trade-off in the FSM estimation procedure. Theorem 5.2 (Bias characterization of the FSM estimator). Let θ * be the true parameter, and denote x 0 ∼ P θ * as random observations sampled from the true model. Suppose there exists a unique maximum likelihood estimator for this model, and that the log-likelihood is
L-smooth. Recall that g(x 0 ; θ t ) = ∇ θ log p(x 0 | θ)| θ=θt is the true Fisher score, S * (x 0 ; θ t ) = E θ∼p(θ|x,θt) ∇ θ log p(x | θ)
is the optimal FSM estimator. For a fixed parameter point θ t ,
The bias at θ t is bounded by
E x0 S * (x 0 ; θ t ) -g(x 0 ; θ t ) ≤ L √ d σ E x0 [R(x 0 )]
where R(x) = p(x|θ * ) p(x|θt) is a likelihood ratio term and d is the dimension of the parameter space We provide a full proof in Appendix A.7. From Theorem 5.2, we can see that, increasing σ, we increase the bias of the FSM estimator. Intuitively, this is because σ governs the degree of smoothing, which induces a "smearing" effect of the FSM gradient estimates. On the other hand, for the linear FSM estimator, note that in the estimator Ŵ , we have the proposal gradient term
∇ θ log q(θ | θ t )| θ=θ (j) = -1 σ 2 (θ (j) -θ t )
, and hence taking σ → 0 inflates the variance of Ŵ . Thus, there is a fundamental bias-variance trade-off in the choice of σ.
Figure 2 empirically illustrates the bias-variance trade-off of the linear FSM estimator with an isotropic Gaussian proposal distribution for two Gaussian likelihood models with differing curvature. In particular, the figure also shows the effect of the log-likelihood curvature, or the gradient-Lipschitz constant L from Theorem 5.2, on the MSE-optimal choice of the proposal scale σ. When the curvature is stronger (larger L), smoothing tends to introduce more bias, and the optimal σ is smaller to control the bias. Conversely, when curvature is weaker (smaller L), a larger σ is optimal to reduce the variance of the score estimator. Furthermore, note that the likelihood ratio term, R(x) = p(x|θ * ) p(x|θt) encodes the estimation error from using training samples around the parameter iterate points θ t to estimate an FSM estimator that is evaluated at observations x 0 ∼ P θ * . Hence, for parameter iterates θ t that are far from the true parameter θ * , we are likely to get a subpar estimation of the true gradient, while as we approach the true parameter, our estimation is likely to improve. However, increasing σ, we can sample from a wider parameter space and are therefore more likely to obtain parameter samples that cover θ * . Thus, σ also encodes an inherent exploration-exploitation trade-off.
this section cite: []

Section: Convergence Guarantees
As we have shown that our FSM gradient estimator closely relates to the Gaussian smoothing gradient estimator in Section 5.1, we can leverage established results showing the asymptotic convergence of stochastic gradient-based optimization methods with such biased gradient estimators. In particular, instead of using the final parameter iterate of the gradient-based optimization procedure as the approximated MLE θ T ≈ θMLE , we instead propose using an averaged SGD estimator θT = 1 T T t=1 θ t based on Polyak-Ruppert averaging [Polyak andJuditsky, 1992, Ruppert, 1988], which enjoys stronger theoretical guarantees. We provide the relevant convergence arguments in Appendix A.8.
A further benefit is that since we can obtain the quantification of the algorithmic uncertainty using the averaged SGD, θT -θMLE from Appendix A.8 and the statistical uncertainty of the MLE θMLE -θ * from standard statistical theory, we can provide a result showing the quantification of the joint uncertainty using the averaged SGD θT as an approximate MLE.
Theorem 5.3. Let θMLE,N be the MLE for N i.i.d. samples. Suppose that the number of iterations in the optimization algorithm T dominates the number of observations N such that N T → 0 as N, T → ∞. Then, assuming that √ T ( θT -θMLE,N ) = O p (1) uniformly over both N, T and that the standard regularity conditions for the MLE are met, we have as N, T → ∞, √ N ( θT -θ * ) → d N (0, I(θ * ) -1 ) where I(θ * ) is the Fisher information matrix evaluated at the true parameter
We provide the proof in Appendix A.9. Given that the Fisher information matrix I(θ * ) can be approximated using the Fisher score by drawing samples x i ∼ P θ and evaluating
I(θ * ) ≈ 1 N N i=1 ∇ θ log p(x i | θMLE )∇ θ log p(x i | θMLE ) ⊤
, we can also estimate this with our FSM method and take advantage of this result to obtain uncertainty quantification based on Theorem 5.3.
this section cite: ['b24']

Section: Related Work
The method closest to ours is the approximate MLE approach of Bertl et al. [2017], which first estimates the likelihood through kernel density estimation (KDE) before applying a simultaneous perturbation stochastic approximation (SPSA) [Spall, 1992] algorithm, which amounts to using a finite-differences gradient estimator on the likelihood function estimated using KDE. In contrast, our FSM method directly estimates the Fisher score, merging density and gradient estimation into one step and thereby reducing both model complexity and computational overhead. After posting the first version of this manuscript on arXiv, we became aware of related independent work by Sui et al. [2025], which proposes a similar Fisher score matching estimator. Their focus is on Fisher score estimation more broadly, whereas our work targets simulation-based MLE specifically.
The use of MLE in the simulation-based model setting was first addressed in the seminal work on SBI of Diggle and Gratton [1984], although the inference of SBI is more typically addressed within the Bayesian framework, as exemplified by the ABC algorithm. Naturally, since the maximum a posteriori estimate (MAP) of the posterior distribution under a uniform prior corresponds to the MLE within the prior support, Rubio and Johansen [2013] suggested leveraging the ABC algorithm and using KDE to obtain the MLE, and more recent neural surrogate SBI methods, such as SNLE [Papamakarios et al., 2019], while not specifically targeted for MLE, could be used in the same way. Another similar line of research is the work of Ionides et al. [2017] and Park [2023], which develop the MLE methodology in the SBI setting for partially observed Markov models. Research focused on developing SBI methods using score matching is a growing field [Geffner et al., 2023, Sharrock et al., 2024, Jiang et al., 2025], however, this has been limited to amortized Bayesian inference, and, to our knowledge, we are the first work that has adapted score matching for the purpose of direct Fisher score estimation and MLE in the SBI setting.
this section cite: ['b2', 'b34', 'b37', 'b8', 'b27', 'b22', 'b13', 'b23', 'b11', 'b31', 'b14']

Section: Experimental Results
We evaluate our local Fisher score matching (FSM) technique on both controlled numerical studies and challenging real-world SBI problems. 3 For all experiments, we use an isotropic Gaussian proposal distribution q(θ | θ t ) = N (θ t , σ 2 I) with a linear FSM estimator, and an empirical comparison with the neural network-based FSM estimator is provided in the relevant experiment sections in the Appendix.
As a primary baseline, we compare against the approximate MLE method of Bertl et al. [2017], here referred to as KDE-SP, which estimates a log-likelihood via kernel density estimation (KDE) and then uses a simultaneous perturbation (SP) estimator to compute gradients:
∇ℓ(θ) = δ l(θ + ) -l(θ -) 2c
where δ is a Rademacher random vector with i.i.d. entries, θ ± = θ ± cδ, c is a perturbation constant, and l(θ; x obs ) = log p(x obs | θ) is the log-likelihood estimated from the KDE by simulating data samples around the target parameter θ and evaluating at the observations x obs . We provide further details about the implementation in Appendix A.10.
this section cite: ['b2']

Section: Numerical Studies
To investigate the accuracy of gradient estimation and parameter estimation, we begin with a multivariate Gaussian model that features a fixed covariance. This model has a closed-form Fisher score, allowing us to directly compare the estimated gradients from FSM and KDE-SP against the ground truth. Further details and results of this experiment are presented in Appendix A.11. One key aspect of both the FSM and KDE-SP approach is the choice of the hyperparameters, specifically the perturbation constant in KDE-SP and the proposal variance in FSM. In Figure 3, we show the sensitivity of the gradient approximation quality to different choices of this hyperparameter, as the simulation budget increases. Although the gradient approximation of both methods depends strongly on the choice of hyperparameters, we see that the FSM estimate is always able to match the accuracy of the KDE-SP estimate given sufficient simulation budget, even when the hyperparameters are not favorably tuned. We provide the same ablation study in higher-dimensional settings in Appendix A.11. In Figure 4, we show the quality of the resulting parameter estimate for the same multivariate Gaussian model with increasing parameter dimension while keeping the simulation budget fixed. While the FSM gradient is able to maintain the quality of the parameter estimate, the KDE-SP struggles in higher dimensional parameter spaces, likely due to the additional kernel density estimation required.
this section cite: []

Section: LSST Weak Lensing Cosmology Model
In this example, we use the log-normal forward model proposed by Zeghal et al. [2024] and Lanzieri et al. [2025], which simulates the non-Gaussian structure in gravitational weak-lensing. Using the model in the full LSST-Y10 setting, this model is representative of real-world weak-lensing data.
Since the generated data are high-dimensional tomographic convergence maps (5 × 256 × 256), we use a trained ResNet-18 compressor in Alsing et al. [2018], producing a 6-dimensional summary statistic. As an additional benchmark beyond the KDE-SP method, we further implement a standard neural likelihood estimator (NLE) using the SBI package [Boelts et al., 2025], trained with the same total simulation budget given to both the KDE-SP and FSM gradient-based optimization methods. Evaluated at the observations, NLE can be directly optimized to obtain an approximate maximum likelihood estimator. Further details and results of this experiment are presented in Appendix A.12
In Figure 5, we show both the parameter estimation and the accuracy of the prediction. Given the limited simulation budget available, we observe that sequential gradient-based optimization methods outperform the more simulation-intensive NLE approach and that the FSM approach is generally able to achieve better performance with a smaller variance.
7.3 Generator Inversion Task In this section, we tackle the canonical problem of latent inversion of a generator network [Xia et al., 2022]. For a fixed generator G w and a query image x 0 , the goal is to recover a latent vector z such that G w (z) ≈ x 0 . Although typically z is treated as a point estimate, in this setting, we treat it as a latent variable, z ∼ N θ, σ 2 z I and focus on θ as the parameter of interest. Note the marginal likelihood We train a GAN model on a 16 × 16 MNIST dataset and apply the generator inversion task, comparing the direct optimization approach, FSM, and KDE-SP method. From Figure 6 we can see that while the FSM and direct optimization is able to recover the target observation, the KDE-SP struggles to achieve the same pixel quality. This is also reflected in Figure 7, which shows the reconstruction loss for the different methods. More details and results for this experiment are provided in Appendix A.13.
p w (x | θ) = δ (x -G w (z)) N z | θ, σ 2 z I dz
this section cite: ['b44', 'b17', 'b0', 'b42']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: See Section 3 Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: The limitations are briefly discussed in Section 8 and left for future work.
Guidelines:
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

Section: Theory Assumptions and Proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: See Appendix A.8 Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental Result Reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: See Section 7 and Algorithm 1 Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Only publicly accessible datasets are used and code is provided.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental Setting/Details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: See Appendix A.11,A.12,A.13 Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7.
this section cite: []

Section: Experiment Statistical Significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: All figures in Section 7 and Appendix A.11, A.12, and A.13 include either error bars representing the 95% confidence intervals over 100 repeated runs, or boxplots that visualize the distribution of the results.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8. Experiments Compute Resources Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: See Appendix A.11,A.12,and A.13 Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code Of Ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research presented in this paper fully complies with the NeurIPS Code of Ethics Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10. Broader Impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This paper is purely a mathematical work and does not involve direct societal applications.
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
Answer: [NA]
Justification: This paper does not work on language models.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: See Section 7
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New Assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: This paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets ( if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and Research with Human Subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: This paper does not involve crowdsourcing and human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: This paper does not involve crowdsourcing and human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: A Appendix / Supplemental Material
A.1 Fisher score matching objective
Here, we provide a complete theorem and a proof for Theorem 3.1.
Theorem A.1 (Local Fisher Score Matching). Let J (W ) be defined as in Equation (1). Given the following assumptions:
• p(x | θ), q(θ | θ t ) are differentiable with respect to θ, S W (x) is differentiable with respect to x
• ∀x ∈ R dx , lim ∥θ∥→∞ p(x | θ)q(θ | θ t ) = 0 J (W )
can be rewritten (up to an additive constant w.r.t. W ) as
J (W ; θ t ) = E x∼p(x|θ),θ∼q(θ|θt) [∥S W (x)∥ 2 + 2 S W (x) ⊤ ∇ θ log q(θ | θ t )](6)
Proof. We denote the joint distribution over (x, θ) from the distributions p(x | θ) and q(θ | θ t ) as p(x, θ | θ t ). First, we expand the square, and remove terms which are not dependent on the score model parameters W .
J (W ) = E p(x,θ|θt) (∥∇ θ log p(x | θ) -S W (x)∥ 2 ) = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∥∇ θ log p(x | θ) -S W (x)∥ 2 dx dθ = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ){∥∇ θ log p(x | θ)∥ 2 + ∥S W (x)∥ 2 -2∇ θ log p(x | θ) ⊤ S W (x)}dx dθ = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ) ∥S W (x)∥ 2 -2∇ θ log p(x | θ) ⊤ S W (x) dx dθ + (constants w.r.t. W )
Next, by exchanging integrals and using the integration by parts tricks similar to Theorem 1 in Hyvärinen and Dayan [2005],
J (W ) = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∥S W (x)∥ 2 dx dθ -2 θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∇ θ log p(x | θ) ⊤ S W (x)dx dθ = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∥S W (x)∥ 2 dx dθ -2 x∈R dx θ∈R d θ q(θ | θ t )∇ θ p(x | θ) ⊤ S W (x)dθ dx = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∥S W (x)∥ 2 dx dθ -2 x∈R dx θ∈R d θ q(θ | θ t ) d θ i=1 ∂ ∂θ i p(x | θ)S (i) W (x)dθ dx J (W ) = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∥S W (x)∥ 2 dx dθ -2 x∈R dx d θ i=1 S (i) W (x) θ∈R d θ q(θ | θ t ) ∂ ∂θ i p(x | θ)dθ dx = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∥S W (x)∥ 2 dx dθ + 2 x∈R dx d θ i=1 S (i) W (x) θ∈R d θ ∂ ∂θ i q(θ | θ t )p(x | θ)dθ dx
Finally, by further simplification
J (W ) = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∥S W (x)∥ 2 dx dθ + 2 x∈R dx d θ i=1 S (i) W (x) θ∈R d θ q(θ | θ t ) ∂ ∂θ i log q(θ | θ t )p(x | θ)dθ dx = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ)∥S W (x)∥ 2 dx dθ + 2 θ∈R d θ q(θ | θ t ) x∈R dx d θ i=1 S (i) W (x) ∂ ∂θ i log q(θ | θ t )p(x | θ)dx dθ = θ∈R d θ q(θ | θ t ) x∈R dx p(x | θ) d θ i=1 S (i) W (x) 2 + 2S (i) W (x) ∂ ∂θ i log q(θ | θ t ) dx dθ = E q(θ|θt) E p(x|θ) d θ i=1 S (i) W (x) 2 + 2S (i) W (x) ∂ ∂θ i log q(θ | θ t ) = E x∼p(x|θ),θ∼q(θ|θt) [∥S W (x)∥ 2 + 2 S W (x) ⊤ ∇ θ log q(θ | θ t )]
A.2 Bayes-optimal solution to Fisher score matching objective
We present the complete theorem and proof for Theorem 3.2 here. Theorem A.2. For a general differentiable function S : R dx → R d θ ,
S * = argmin S E p(x,θ|θt) ∥∇ θ log p(x | θ) -S(x)∥ 2 = E p(θ|x,θt) ∇ θ log p(x | θ)
Proof. First, observe that since the function S is only a function of x, we have
E p(x,θ|θt) S(x) = E p(x|θt) S(x)
We can decompose the objective function by expanding the square,
argmin S E p(x,θ|θt) ∥∇ θ log p(x | θ) -S(x)∥ 2 = argmin S E p(x,θ|θt) ∥S(x)∥ 2 -2S(x) ⊤ ∇ θ log p(x | θ)
Then, our objective can be equivalently expressed as
E p(x|θt) ∥S(x)∥ 2 -2S(x) ⊤ E p(θ|x,θt) [∇ θ log p(x | θ)] Which has the optimal solution S * (x) = E p(θ|x,θt) ∇ θ log p(x | θ) A.3 Linear Fisher score model parameterization
Here, we provide details of the linear Fisher score model derivation.
Recall that the parameter and data space are θ ∈ R d θ , x
k ∈ R dx , the linear score model weights are W ∈ R dx×d θ , and we defined the data matrix as
X j =    x (j)⊤ 1 . . . x (j)⊤ n    ∈ R n×dx
this section cite: ['b12']

Section: and the corresponding
Gram matrix as G j = X ⊤ j X j . In practice, we include an intercept term in our regression by augmenting the data matrix with a column of ones, i.e., x (j) k 1 ∈ R dx+1 and W as a (d x + 1) × d θ matrix. For simplicity, we omit this intercept term in our derivation.
We start from the empirical version of the local Fisher score matching objective, Equation (3) (replacing averages by sums for simplicity),
Ĵ (W ) = m j=1 n k=1 [∥S W (x (j) k )∥ 2 + 2 S W (x (j) k ) ⊤ ∇ θ log q(θ | θ t )| θ=θ (j) ] Substituting our linear score model, S(x; θ t ) = W ⊤ x, Ĵ (W ) = m j=1 n k=1 ∥W ⊤ x (j) k ∥ 2 + 2 m j=1 n k=1 (x (j)⊤ k W ∇ θ log q(θ | θ t )| θ=θ (j) )
To obtain the first-order conditions, we take derivative with respect to W , for each of the terms separately.
For the first term,
m j=1 n k=1 ∥W ⊤ x (j) k ∥ 2 = m j=1 tr[(X j W ) ⊤ (X j W )] = m j=1 tr(W ⊤ X ⊤ j X j W ) Applying ∂ ∂W gives: m j=1 2X ⊤ j X j W = 2 m j=1 G j W
For the second term, we can similarly apply ∂ ∂W to give:
2 m j=1 n k=1 ∂ ∂W [x (j)⊤ k W ∇ θ log q(θ | θ t )| θ=θ (j) ] = 2 m j=1 n k=1 x (j) k ∇ θ log q(θ | θ t )| ⊤ θ=θ (j)
Combining the two terms, we obtain
∂ ∂W Ĵ (W ) = 2 m j=1 G j W + 2 m j=1 n k=1 x (j) k ∇ θ log q(θ | θ t )| ⊤ θ=θ (j)
Setting this to 0 gives us the normal equations in Equation ( 4).
If the sum of the Gram matrices, m j=1 G j is invertible (otherwise, we may opt to use the ridge penalty), we can directly obtain the linear Fisher score matching estimator in Equation ( 5).
For an isotropic Gaussian proposal, q(θ | θ t ) = N (θ | θ t , σ 2 I), we have that
∇ θt q(θ | θ t ) = -∇ θ q(θ | θ t )
Using the integration-by-parts trick (similarly to the proof in Appendix A.6), we have,
p(x | θ)∇ θt q(θ | θ t )dθ = -p(x | θ)∇ θ q(θ | θ t )dθ = ∇ θ p(x | θ)q(θ | θ t )dθ = ∇ θ log p(x | θ)p(x | θ)q(θ | θ t )dθ
Substituting this expression into ∇ θt ℓ(θ t ; x), we have,
∇ θt ℓ(θ t ; x) = 1 Z(θ t ) p(x | θ)∇ θt q(θ | θ t )dθ = 1 Z(θ t ) ∇ θ log p(x | θ)p(x | θ)q(θ | θ t )dθ = ∇ θ log p(x | θ) p(x | θ)q(θ | θ t ) Z(θ t ) dθ = E θ∼p(θ|x,θt) ∇ θ log p(x | θ)
A.7 Bias of FSM Theorem A.4 (Bias characterization of the FSM estimator). Let θ * be the true parameter, and denote x 0 ∼ P θ * as random observations sampled from the true model. Suppose there exists a unique maximum likelihood estimator for this model, and that the log-likelihood is L-smooth. Recall that g(
x 0 ; θ t ) = ∇ θ log p(x 0 | θ)| θ=θt is the true Fisher score, S * (x 0 ; θ t ) = E θ∼p(θ|x,θt) ∇ θ log p(x | θ)
is the optimal FSM estimator. For a fixed parameter point θ t ,
The bias at θ t is bounded by 1. We first show the bias bound ∥S * (x; θ t ) -g(x; θ t )∥, at a fixed data point x. 10 1 10 2 Parameter Dimension 10 1 10 0 10 1 10 2 Parameter Error: FSM Linear Gradient KDE-SP Gradient FSM NN Gradient favorably with respect to the increase in the number of simulation budgets. However, in Figure 13, the matrix inversion step of the linear FSM method grows cubically with the parameter dimension, and hence causes an increase in the wall-clock time for the FSM method. We note that in practice one can reduce this cost considerably by employing faster linear solvers (e.g., conjugate gradient methods), which can greatly improve scalability in higher dimensions.
E x0 S * (x 0 ; θ t ) -g(x 0 ; θ t ) ≤ L √ d σ E x0 [R(x 0 )] where R(x) = p(x|θ * ) p(
∥S * (x; θ t ) -g(x; θ t )∥ = E θ∼p(θ|x,θt) ∇ θ log p(x | θ) -∇ θ log p(x | θ)| θ=θt ≤ E θ∼p(θ|x,θt) ∇ θ log p(x | θ) -∇ θ log p(x | θ)| θ=θt ≤ L ∥θ -θ t ∥p(θ | x, θ t )dθ ≤ L sup θ p(θ | x, θ t ) q(θ | θ t ) ∥θ -θ t ∥q(θ | θ t )dθ ≤ L √ dσ sup θ p(θ | x, θ t ) q(θ | θ t )
this section cite: []

Section: A.11.4 Additional results on confidence interval construction
We also note that in Figure 14, we provide a simple validation test for the use of the FSM estimate for the Fisher information matrix estimation. This shows that we can recover a well-calibrated confidence interval even with the use of a stochastic Fisher score estimate.
Further details of these experiments are provided in the Appendix A.11.5. 10 0 10 1 10 2 Number of Observations 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Estimated Averaged Coverage of 95% CI FSM Linear Gradient FSM NN Gradient Parametric Bootstrap (B=100) 95% Coverage Figure 14: Estimated coverage of constructed confidence interval (averaged across all parameter dimensions) from the approximated Fisher information matrix estimation with FSM estimates, comparing repeating the FSM estimation procedure with nonparametric bootstrap A.11.5 Experimental details
The gradient comparison experiment corresponding to the plots of Figures 3 and 8 was carried out for a bivariate Gaussian mean model with 10 observations. Observations were generated from a true mean of (1.0, 1.0) (Figure 9 used a 20-dimensional Gaussian with the same setting), and Fisher score estimates were taken at the observation means, which is also the maximum likelihood estimator. The uncertainty was obtained by repeating 100 runs of the score estimation for both methods.
The multivariate Gaussian parameter estimation accuracy in Figure 11 was performed with 100 observations, and with parameter dimensions of d = 5, 20, 50, 100 for 100 optimization steps, using 100 repeated runs as with the previous experiment. Figures 4 and 10 were performed in a similar way, by fixing a total simulation budget of 1000 and increasing parameter dimensions of d = 2, 5, 10, 20, 50, 100.
The true parameters used to generate the observations were similarly taken to be a vectors of ones as with the previous experiment. The (a, c) hyperparameters for the KDE-SP gradient method was
this section cite: []

Section: 
Naturally, our linear score model setup can be extended to include a Frobenius norm penalty λ∥W ∥ 2 F in the objective, leading to a ridge-type solution:
Ŵ = - m j=1 G j + λ I dx -1 m j=1 n k=1 x (j) k ∇ θ log q(θ | θ t )| ⊤ θ=θ (j)
This stabilises the inverse and helps prevent overfitting in finite-sample regimes. In practice, we implement the ridge-type linear Fisher score model.
this section cite: []

Section: A.4 Neural Network Fisher score model parameterization
An alternative to the linear Fisher score model provided in Appendix A.3 is a neural network parameterization of the score model. In this setting, we denote S(x; θ t ) = S ϕ (x) where S ϕ is a neural network with parameters ϕ for a fixed parameter iterate θ t . The parameters ϕ can be obtained by optimizing the FSM objective J (ϕ; θ t ) from Equation (2) (with its Monte Carlo estimate Equation ( 3)) using standard neural network backpropagation. Thus, following Algorithm 1, using a neural network parameterization requires a potentially costly inner optimization loop for each parameter iterate θ t .
this section cite: []

Section: A.5 Fisher score matching proposal distribution
As discussed in Section 5.2, the theoretical optimal choice of the hyperparameter σ depends on the curvature of the log-likelihood function. In practice, the curvature is unknown, and thus selecting the optimal σ is challenging in general. We propose a simple pilot calibration based on grid search, before running the main FSM-MLE procedure, we execute a short pilot FSM-MLE procedure with different candidate value σ k yielding corresponding candidate parameter estimate θσ k . For each σ k , we simulate data at θσ k , and select the candidate σ k which minimizes the discrepancy between the observed data and the simulated data at the candidate parameter estimate. Thus, this procedure selects hyperparameters σ which can produce simulations most consistent with the observations.
A simple annealing schedule which would reduce the hyperparameter σ over the course of the FSM-MLE optimization procedure could also be considered. This would ensure that the smoothing bias discussed in Section 5 vanishes asymptotically, however, such a procedure would be complicated by the increase in the variance of the FSM estimator and numerical instability when σ is too small. While we attempted to implement such an annealing scheme in our experiments, we found that it introduced additional complexity without meaningful performance gains over a fixed σ scheme.
When the likelihood function exhibits strong anisotropic curvature, an isotropic Gaussian proposal is suboptimal. Extending the calibration scheme to diagonal covariances, however, would make grid search scale exponentially with the parameter dimension, making the method computationally prohibitive for higher dimensional problems. Hence, it remains an open question on how to efficiently design scalable procedure to select more expressive proposal distributions.
this section cite: []

Section: A.6 Gaussian smoothing equivalence
We provide here a more detailed derivation of Theorem 5.1. Theorem A.3 (Equivalence as Gaussian Smoothing). Under an isotropic Gaussian proposal, q(θ | θ t ) = N (θ | θ t , σ 2 I), with the assumptions as Theorem A.1, the optimal score matching estimator is equivalent to the gradient of the smoothed likelihood
∇ θt l(θ t ; x) = E θ∼p(θ|x,θt) ∇ θ log p(x | θ) where l(θ t ; x) = log p x | θ q θ | θ t dθ and p(θ | x, θ t ) ∝ p(x | θ)q(θ | θ t ) is the induced posterior from the proposal distribution q(θ | θ t ) Proof. First, define Z(θ t ) = p(x | θ)q(θ | θ t )dθ such that l(θ t ; x) = log Z(θ t ).
Now, observe that,
∇ θt ℓ(θ t ; x) = ∇ θt Z(θ t ) Z(θ t ) = 1 Z(θ t ) p(x | θ)∇ θt q(θ | θ t )dθ Denoting p(x | θ t ) := p(x | θ) q(θ | θ t ) dθ, note that sup θ p(θ|x,θt) q(θ|θt) = p(x| θMLE(x)) p(x|θt)
2. We now take expectation with respect to the true model, x 0 ∼ p(x | θ * ), and the only non-constant term is sup θ p(θ|x,θt) q(θ|θt) .
E x0 sup θ p(θ | x 0 , θ t ) q(θ | θ t ) = p(x 0 | θMLE (x 0 )) p(x 0 | θ t ) p(x 0 | θ * )dx 0 ≈ p(x 0 | θ * ) p(x 0 | θ t ) p(x 0 | θ * )dx 0 = E x0 p(x 0 | θ * ) p(x 0 | θ t )
A.8 Convergence guarantees of FSM
In this section, we provide an asymptotic convergence analysis of the stochastic gradient method based on the local Fisher score matching gradient under a Gaussian proposal distribution. Recall that, based on theoretical development in Section 5.1 and Appendix A.6, we have shown that the FSM estimator, under a isotropic Gaussian proposal distribution, targets a smoothed log-likelihood,
lσ (θ; x) = log p(x | θ ′ )N (θ ′ | θ, σ 2 I)dθ ′
Let the N independent and identically distributed observations be D = {x i } N i=1 and the corresponding smoothed likelihood objective be
lσ (θ; D) = N i=1 lσ (θ; x i )
We define the smoothed maximum likelihood estimator for the dataset D as
θσ = argmax θ N i=1 lσ (θ; x i )
Equivalently, assuming the concavity of the smoothed likelihood function, we can characterize the smoothed maximum likelihood estimator with its first-order optimality condition.
∇ θ lσ (θ; D) = N i=1 S * (x i ; θ) = 0 where S * (x; θ) = ∇ θ lσ (θ; x) is the Bayes-optimal FSM estimator.
In practice, however, we utilize the linear FSM estimator Ŝ(x; θ) as discussed in Section 3.2 and Appendix A.3. In order to reduce the variance of the resulting approximate maximum likelihood estimator, as well as to provide stronger theoretical guarantees, we use the averaged parameter estimate [Polyak and Juditsky, 1992] θT = 1 T T t=1 θ t
In the following, we state an asymptotic convergence result, which can be found in Proposition 2.1 of Jin et al. [2021], which is based on Polyak and Juditsky [1992]. Note that this is a sufficient condition for the strong concavity of the smoothed log-likelihood.
this section cite: ['b24', 'b15', 'b24']

Section: Assumption A.2 (Step Size Condition).
The step-sizes η t > 0 satisfies for all t, ηt-ηt+1 ηt = o(η t ) and ∞ t=1 η (1+λ)/2 t -1/2 < ∞ Assumption A.3 (Unbiasedness and Martingale Noise Control). Define the noise term ξ t = Ŝ(θ t-1 , u t ; σ) -S * (θ t-1 ; σ), which is a martingale difference sequence with respect to F t-1 = σ(u 1 , . . . , u t-1 ), where u t = {(θ i,t , X i,t )} M i=1 represents all the simulations used for the score model estimation at iteration t.
1. For all iterations t ≥ 1, the linear FSM estimator is unbiased:
E[ Ŝ(θ t-1 , u t ; σ) | F t-1 ] = S * (θ t-1 ; σ) 2.
Assume that there exists a constant K > 0 such that for all t ≥ 1, almost surely:
E[∥ξ t ∥ 2 | F t-1 ] + ∥S * (θ t-1 ; σ)∥ 2 ≤ K(1 + ∥θ t-1 -θσ ∥ 2 ) Assumption A.4 (Hessian Bound).
There is a function H(u) with bounded fourth moments, such that the operator norm of ∇ θ Ŝ(θ, u) is bounded, ∥∇ θ Ŝ(θ, u)∥ ≤ H(u) for all θ Theorem A.5. Suppose Assumptions A.1, A.3 and A.4 hold and the sequence of step sizes fulfills A.2. Using the updates of the gradient descent θ t+1 ← θ t + η t Ŝt (x; θ t , σ), we have that the averaged parameter iterates θT = 1 T T t=1 θ t satisfies as T → ∞:
1. θT → a.s. θσ 2. √ T ( θT -θσ ) → d N (0, V ) where V = (∇ 2 θ lσ ( θσ ; D)) -1 E[ Ŝ( θσ ; D) Ŝ( θσ ; D) ⊤ ](∇ 2 θ lσ ( θσ ; D)) -1
A.8.1 Relationship between smoothed MLE and the true MLE Furthermore, we note here that we can establish an upper bound on the distance between the smoothed MLE θσ and the true MLE θ. For simplicity, assume that we only have a single observation in our dataset, x. Then, using the strong concavity in Assumption A.1, L-smoothness of the log-likelihood as in Theorem A.4, and the result from A.7 for a fixed point x, and denoting the gradient of the true log-likelihood as g(θ;
x) = ∇ θ ℓ(θ; x), ∥ θσ -θ∥ ≤ 1 µ ∥g( θσ ; x) -g( θ; x)∥ = 1 µ ∥g( θσ ; x)∥ = 1 µ ∥g( θσ ; x) -S * (x; θσ )∥ = Lσ √ d θ µ sup θ p(θ | x, θσ ) q(θ | θσ )
Thus, we have shown that ∥ θσ -θ∥ is approximately of the order O(σ).
this section cite: []

Section: A.9 Uncertainty quantification of FSM
Here, we show the uncertainty quantification by leveraging the result from Appendix A.8 with classical MLE theory. As before, we denote θT = 1 T T t=1 θ t as the averaged parameter iterate from the FSM-SGD procedure, and for clarity, we denote θMLE,N as the MLE of the true likelihood based on N i.i.d. observations. Our goal will be to characterize the distribution of
√ N ( θT -θ * ).
First, we note that we can decompose θT -θ * into both algorithmic and statistical uncertainty, θT -θ * = ( θT -θMLE,N ) algorithmic uncertainty
+ ( θMLE,N -θ * ) sampling uncertainty
Multiplying by √ N , we obtain the following.
√ N ( θT -θ * ) = √ N ( θT -θMLE,N ) + √ N ( θMLE,N -θ * )
Focusing on the algorithmic error, observe that
√ N ( θT -θMLE,N ) = N T • √ T ( θT -θMLE,N )
Since by assumption we know that N T → 0 and X N,T = √ T ( θT -θMLE,N ) = O p (1) from Appendix A.8, this implies that their product is
N T • X N,T = √ N ( θT -θMLE,N ) → p 0 as N, T → ∞. From classical MLE theory, under standard regularity conditions, √ N ( θMLE,N -θ * ) → d N (0, I(θ * ) -1 )
where I(θ * ) is the Fisher information matrix at the true parameter.
Finally, to combine both results, using Slutsky's theorem,
√ N ( θT -θ * ) → d N (0, I(θ * ) -1 )
as N, T → ∞
this section cite: []

Section: A.10 KDE-SP implementation
We implement the KDE-SP gradient estimator as proposed in Bertl et al. [2017], combining a kernel density estimate (KDE)-based likelihood approximation with a simultaneous perturbation stochastic approximation (SPSA). Specifically, at each iteration t, the approximate gradient of the log-likelihood at θ is given by:
∇ℓ(θ) = δ t l (θ + ) -l (θ -) 2c t ,
where θ + = θ + c t δ t and θ -= θ -c t δ t for a random perturbation δ t . This gradient estimate is then used in an SPSA update of the form
θ t = θ t-1 + α t ∇ℓ (θ t-1 )
Following the specifications in Bertl et al. [2017], we adopt the standard SPSA step size schedule:
α t = a (t + A) α , c t = c t γ
with α = 1, γ = 1/6, and A = ⌊0.1T ⌋, where T is the total number of iterations.
The constants a and c control the initial values of α t and c t . We tune both by performing a grid search over pairs ( a, c ). For each candidate pair, we run a short trial of the SPSA optimization, simulate data from the resulting parameter estimates, and measure prediction error relative to the observed dataset. We then select the pair ( a, c ) that yields the lowest validation error. We also incorporate the KDE modifications proposed in Section 3.2 of Bertl et al. [2017], which refine the KDE-based likelihood approximation. These modifications help stabilize the KDE estimation for high-dimensional problems.
this section cite: ['b2', 'b2', 'b2']

Section: A.11 Additional details and results on numerical studies experiment A.11.1 Additional results on hyperparameter sensitivity
In Figures 8 and 9, we provide additional results on the ablation study showing the sensitivity of the gradient accuracy between the FSM method and the KDE-SP method for different choices of the proposal variance and perturbation constants, respectively. Figure 3 is a subset of the results shown in Figure 8. Even in higher-dimensional settings, we find that the FSM method can match the gradient accuracy of the KDE-SP method across a wide range of hyperparameter choices.
this section cite: []

Section: A.11.2 Additional results on parameter dimension scaling
Figure 10 includes an additional result with the neural network FSM method for the parameter dimension scaling experiment seen in Figure 4. Furthermore, Figure 11 shows the scaling with parameter dimension, with increasing simulation budgets, complementing the results seen in Figure 10. Generally, we find that the linear FSM method performs the best across different parameter dimensions and simulation budgets. The KDE-SP method performs worse in higher dimensions, likely due to the curse of dimensionality affecting the kernel density estimate. The neural network FSM method shows competitive performance in lower dimensions, but its performance quickly degrades in higher dimensions, possibly due to optimization and/or overfitting issues.
this section cite: []

Section: A.11.3 Additional results on wall-clock time
We provide a comparison of the wall-clock time in Figures 12 and 13 for repeated gradient estimation procedures for both the KDE-SP and FSM methods. As we can see in Figure 12, the FSM scales selected from a grid of [10 -2 , 10 -1 , 10 0 , 10 1 , 10 2 , 10 3 ] × [10 -2 , 10 -1 , 10 0 , 10 1 , 10 2 , 10 3 ]. For the FSM-based estimation, the (σ, η) hyperparameters, corresponding to the proposal variance and step size, were tuned in the exact same way as the KDE-SP gradient hyperparameters (using the prediction error), but over a grid of [10 -3 , 10 -2 , 10 -1 ] × [10 -2 , 10 -1 , 10 0 ] instead. The Adam [Kingma and Ba, 2015] optimizer was used for the FSM-based estimation, with averaging over the last 50 iterations of the parameter iterates.
For the wall-clock time comparisons in Figures 12 and 13, each gradient estimation procedure was timed for 1000 runs on a bivariate Gaussian mean model with 10 observations. As both the FSM and KDE-SP gradient estimation was implemented in Python and the JAX package, best attempts were made to equalize the comparison between the two methods. All just-in-time (JIT) compilations for both methods were disabled for the wall-clock tests to remove compilation overhead.
For the confidence interval experiment of Figure 14, a 5 dimensional multivariate Gaussian mean model was used. A step size of 10 -3 with σ = 0.05 was used with the RMSProp [Tieleman, 2012] optimizer. The final Fisher information matrix was estimated by simulating 100000 simulations from the resulting MLE estimate of the optimization run, which was used to construct the confidence interval. This was repeated for 100 runs to obtain an estimated coverage probability.
For the neural network-based FSM method, a standard feedforward neural network with two hidden layers of size 16 with ReLU activations was used. Adam optimizer with a step size of 10 -2 was used to train the neural network for 10 iterations, for each parameter iteration of the MLE optimization procedure.
All experiments in this section were performed on a standard consumer laptop, an Intel i7-11370H CPU with 64GB of RAM.
this section cite: ['b16']

Section: A.12 Additional details on LSST weak lensing experiment
For the weak lensing experiment in Figure 15, 100 iterations of the gradient optimization method were used with both the KDE-SP and FSM estimators, with 100 simulations per iteration, giving a total simulation budget of 10000 simulations for the entire optimization process. The dimension of the parameter space is 6, and the dimension of the summary statistics used is 6 as well.
The same amount of simulations was provided to a neural likelihood estimater, which is a standard masked autoregressive flow model in the package SBI in Python [Boelts et al., 2025]. To mimic a general, uninformative prior, we used the priors for the parameters provided in Table 1 of Zeghal et al. [2024], which are all Gaussian priors, and converted them to a uniform prior by taking three standard deviations from the mean, U[µ -3 * σ, µ + 3 * σ], where the original Gaussian priors are represented as N (µ, σ 2 ). The NLE was trained with 10000 (parameters, data) pairs drawn from this prior, and 5000 iterations with a standard Adam optimizer were used to train the NLE. To optimize the NLE for a specific observational dataset, we evaluated the trained NLE at the specific dataset, and directly differentiated through the NLE model, giving us a deterministic gradient, which is used in a standard gradient-based optimization procedure. The likelihood is optimized until convergence, where there is no longer any change in the estimated likelihood with the NLE.
The hyperparameters (a, c) for the KDE-SP gradient method were selected from a grid of [10 -5 , 10 -4 , 10 -3 , 10 -2 ] × [10 -3 , 10 -2 , 10 -1 , 10 0 ]. For the FSM method, we set σ = 10 -3 and a step size of 10 -2 , with parameter averaging over the final 50 iterations.
For the neural network-based FSM method, a standard feedforward neural network with two hidden layers of size 16 with ReLU activations was used. Adam optimizer with a step size of 10 -2 was used to train the neural network for 10 iterations, for each parameter iteration of the MLE optimization procedure. We find that the neural network-based FSM method did not perform well in this experiment, often suffering from high variance and instability during training.
An RTX 4090 GPU with 24GB of VRAM, 41GB of RAM was used in this experiment.
this section cite: ['b44']

Section: A.13 Additional details on generator inversion task
For the generator inversion task in Figure 16, we trained a standard GAN on a down-scaled 16 × 16 MNIST dataset, giving a data dimension of 256 as no summary statistics were used. We used 500 iterations of the gradient optimization method with both the KDE-SP and the FSM gradient The (a, c) hyperparameters for the KDE-SP gradient method was selected from a grid of [10 -4 , 10 -3 , 5 × 10 -3 , 10 -2 , 5 × 10 -2 ] × [10 -4 , 10 -3 , 5 × 10 -3 , 10 -2 , 5 × 10 -2 ]. For the FSM method, we set σ = 0.2 and a step size of 5 × 10 -2 , with parameter averaging over the last 300 iterations. The latent mean prior, σ z was set at 0.1.
The direct optimization approach was performed by directly minimizing a reconstruction loss (mean squared error in pixel space) between the generated images and the observations, and directly differentiating through the generator network G w . Specifically, we minimize the following loss function.
min θ L(G w (θ), x 0 ) = 1 n n i=1 ∥G w (z i ) -x 0 ∥ 2
where z i ∼ N (θ, σ 2 z I). This is done with the Adam optimizer with a step size of 5 • 10 -1 , and for 1000 iterations, with n = 100 simulations per iteration.
For the neural network-based FSM method, a standard feedforward neural network with two hidden layers of size 16 with ReLU activations was used. Adam optimizer with a step size of 10 -3 was used to train the neural network for 10 iterations, for each parameter iteration of the MLE optimization procedure. Compared to Appendix A.12, we found that the neural network-based FSM method performed better in this experiment, but still generally had subpar performance compared to the linear FSM method and with increased variance.
An RTX 4090 GPU with 24GB of VRAM, 41GB of RAM was used in this experiment.
this section cite: []

Section: References
Ref_id:b0 Title: Massive optimal data compression and density estimation for scalable, likelihood-free inference in cosmology Year: (2018)
Ref_id:b1 Title: Approximate bayesian computation in population genetics Year: (2002)
Ref_id:b2 Title: Approximate maximum likelihood estimation for population genetic inference Year: (2017)
Ref_id:b3 Title: A general method for calibrating stochastic radio channel models with kernels Year: (2021)
Ref_id:b4 Title: sbi reloaded: a toolkit for simulation-based inference workflows Year: ()
Ref_id:b5 Title: Statistical inference Year: (2024)
Ref_id:b6 Title: The frontier of simulation-based inference Year: (2020-05)
Ref_id:b7 Title: Approximate bayesian computation (abc) in practice Year: (2010-07)
Ref_id:b8 Title: Monte carlo methods of inference for implicit statistical models Year: (1984)
Ref_id:b9 Title: Randomized smoothing for stochastic optimization Year: (2012)
Ref_id:b10 Title: Sequential neural methods for likelihood-free inference Year: (2018-11)
Ref_id:b11 Title: Compositional score modeling for simulation-based inference Year: (2023)
Ref_id:b12 Title: Estimation of non-normalized statistical models by score matching Year: (2005)
Ref_id:b13 Title: Monte carlo profile confidence intervals for dynamic systems Year: (2017-07)
Ref_id:b14 Title: Simulation-based inference via langevin dynamics with score matching Year: (2025)
Ref_id:b15 Title: Statistical inference for polyak-ruppert averaged zeroth-order stochastic gradient algorithm Year: (2021)
Ref_id:b16 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b17 Title: Optimal neural summarisation for full-field weak lensing cosmological implicit inference Year: (2025)
Ref_id:b18 Title: Learning in implicit generative models Year: (2017-02)
Ref_id:b19 Title: Parametric empirical bayes inference: theory and applications Year: (1983)
Ref_id:b20 Title: Random gradient-free minimization of convex functions Year: (2017)
Ref_id:b21 Title: Masked autoregressive flow for density estimation Year: (2017)
Ref_id:b22 Title: Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows Year: (2019)
Ref_id:b23 Title: On simulation-based inference for implicitly defined models Year: (2023-11)
Ref_id:b24 Title: Acceleration of stochastic approximation by averaging Year: (1992-07)
Ref_id:b25 Title: Information and the accuracy attainable in the estimation of statistical parameters Year: (1992)
Ref_id:b26 Title: A stochastic approximation method. The annals of mathematical statistics Year: (1951)
Ref_id:b27 Title: A simple approach to maximum intractable likelihood estimation Year: (2013)
Ref_id:b28 Title: Efficient estimations from a slowly convergent robbins-monro process Year: (1988)
Ref_id:b29 Title: Evolution strategies as a scalable alternative to reinforcement learning Year: (2017)
Ref_id:b30 Title: Likelihood-free inference in cosmology: Potential for the estimation of luminosity functions Year: ()
Ref_id:b31 Title: Sequential neural score estimation: likelihood-free inference with conditional score based diffusion models Year: (2024)
Ref_id:b32 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b33 Title: Sliced score matching: A scalable approach to density and score estimation Year: (2020)
Ref_id:b34 Title: Multivariate stochastic approximation using a simultaneous perturbation gradient approximation Year: (1992)
Ref_id:b35 Title: Gaussian smoothing gradient descent for minimizing functions (gsmoothgd) Year: (2023)
Ref_id:b36 Title: Principles of Computational Modelling in Neuroscience. 07 2011 Year: ()
Ref_id:b37 Title: Fisher score matching for simulation-based forecasting and inference Year: (2025)
Ref_id:b38 Title: Divide the gradient by a running average of its recent magnitude Year: (2012)
Ref_id:b39 Title: Toni t, welch d, strelkowa n, ipsen a, stumpf mpapproximate bayesian computation scheme for parameter inference and model selection in dynamical systems Year: (2009)
Ref_id:b40 Title:  Year: (2000)
Ref_id:b41 Title: A connection between score matching and denoising autoencoders Year: (2011-07)
Ref_id:b42 Title: Gan inversion: A survey Year: (2022)
Ref_id:b43 Title: Diffusion models: A comprehensive survey of methods and applications Year: (2023)
Ref_id:b44 Title: Simulation-based inference benchmark for lsst weak lensing cosmology Year: (2024)
