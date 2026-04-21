Title: Identifying Causal Direction via Variational Bayesian Compression
Abstract: Telling apart the cause and effect between two random variables with purely observational data is a challenging problem that finds applications in various scientific disciplines. A key principle utilized in this task is the algorithmic Markov condition, which postulates that the joint distribution, when factorized according to the causal direction, yields a more succinct codelength compared to the anti-causal direction. Previous approaches approximate these codelengths by relying on simple functions or Gaussian processes (GPs) with easily evaluable complexity, compromising between model fitness and computational complexity. To address these limitations, we propose leveraging the variational Bayesian learning of neural networks as an interpretation of the codelengths. This allows the improvement of model fitness, while maintaining the succinctness of the codelengths, and the avoidance of the significant computational complexity of the GP-based approaches. Extensive experiments on both synthetic and real-world benchmarks in cause-effect identification demonstrate the effectiveness of our proposed method, showing promising performance enhancements on several datasets in comparison to most related methods.

Section: Introduction
Cause-effect identification or bivariate causal discoverythe task of telling apart the cause and the effect between two random variables-is a critical task across various scientific disciplines, including biology, economics, and sociology (Pearl, 2009). While randomized controlled trials (RCTs) are considered the most accurate method for identifying these types of causal relationships, especially in medical research (Guyon et al., 2019), they are often impractical due to resource constraints and ethical considerations. Studying passive observations offers a viable alternative for causal inference, despite requiring assumptions on the data generating process to detect the asymmetry between the causal and anti-causal directions. One intuitive interpretation of the causal asymmetry from an information-theoretic perspective is the independence postulate of algorithmic Markov kernels (Janzing & Schölkopf, 2010), which states that the true causal direction must yield the lowest algorithmic Kolmogorov complexity factorization of the joint distribution.
However, because of the incomputability of the Kolmogorov complexity (Li & Vitányi, 2019), approximation methods via the principle of Minimum Description Length (MDL, Grünwald, 2007;Marx & Vreeken, 2017;2019a;b) are proposed to instantiate this complexity empirically. The two-part MDL aims to find the model that minimize two criteria: (1) the complexity of the data given that model and
(2) the model complexity of the model used for modeling the data. The former complexity measures the model fitness, which is commonly evaluated through the log-likelihood of the data given the model. The options for estimating the latter model complexity are more diverse, which usually involve the number of available models and the parameters in each model. As these methods attempt to minimize the total codelength, they are also called compression-based methods.
While these approaches have shown promising results, the estimation of the conditional distributions in these methods relies on traditional regression methods that offer easily evaluable model complexity. If the ground truth conditional models are more complex or too distinct from the predefined ones, this implementation for the model classes can result in lower fitness and higher complexity of data given the model, leading to suboptimal approximations for the algorithmic complexity. Dhir et al. (2024a) overcome this restriction by leveraging Gaussian processes (GPs), though this involves a significant trade-off between flexibility and computational complexity.
To address the limitations of these previous compressionbased methods, we propose the leverage of neural networks for learning the conditional models, which are regarded as universal approximators (Hornik et al., 1989). Moreover, the algorithmic complexity of the networks can be approximated empirically via the concepts of bits-back coding (Hinton & van Camp, 1993;Wallace, 1990) and variational Bayesian coding (Honkela & Valpola, 2004;Louizos et al., 2017;Blier & Ollivier, 2018). In this study, we introduce COMIC-a Bayesian COMpression-based approach to Identifying the Causal direction-that improves the model fitness of compression-based methods without compromising the computability of model complexity and avoids the higher computational complexity of GP-based modeling. Correspondingly, through extensive empirical evaluation on benchmarks for the cause-effect identification task, our approach demonstrates promising results with performance improvements compared to a majority of complexity-based and maximum likelihood-based approaches on multiple benchmarks.
this section cite: ['b46', 'b18', 'b31', 'b17', 'b37', 'b21', 'b19', 'b61', 'b20', 'b35', 'b2']

Section: Contributions
The key contributions of this work can be outlined as follows:
1. We propose the utilization of Bayesian neural networks for modeling the conditional distributions to address the challenge of balancing flexibility and scalability, which is hindering previous complexity/compressionbased methods. By minimizing the variational Bayesian codelength, we can approximate the algorithmic complexity of neural networks.
2. From our proposed encoding scheme of the data given each causal direction, the causal identifiability can be proven. In particular, our models are nonseparable-compatible, which implies that given a sufficient amount of data, the causal direction can be identified via the complexity of our models.
3. The capability of our approach is assessed on both synthetic and real-world bivariate causal discovery benchmarks. In comparison to most related approaches, our method achieves performance improvements in several benchmarks, demonstrating the effectiveness of our approach in identifying the causal direction.
this section cite: []

Section: Related Works
Despite being a well-defined task, the amount of information for determining the causal direction in the bivariate setting is limited, hindering further improvements in both theoretical and empirical results. In the following section, we provide an overview of recent related publications about this task, which includes two popular approaches.
this section cite: []

Section: Functional Causal Models
The earliest functional causal model (FCM, Pearl, 2009) being proposed is the additive noise model (ANM, Shimizu et al., 2006;Hoyer et al., 2008;Bühlmann et al., 2014;Peters et al., 2014), where the effect Y is generated from a function of the cause X and an independent noise E Y (X ⊥ ⊥ E Y ) by adding them as Y := f (X) + E Y . In this model, the cause is assumed to only contribute to the mean, which is can be estimated by mean regression methods (Shimizu et al., 2006;Hoyer et al., 2008;Bühlmann et al., 2014;Peters et al., 2014). From the estimated models, CAM (Bühlmann et al., 2014) determines the causal direction by selecting the one with the greater maximum likelihood, whereas RESIT (Peters et al., 2014) quantifies the independence between the cause and the estimated noise with the Hilbert-Schmidt Independence Criterion (HSIC, Gretton et al., 2005).
Due to the strict assumption on model classes of ANMs, generalization approaches for the ANMs have been introduced to allow for more complex functions, including post nonlinear models (PNLs, Zhang & Hyvärinen, 2009) and heteroscedastic/location-scale noise models (LSNMs, Immer et al., 2023). Specifically, LSNMs assume that the cause not only contributes to the mean but also the scale through another function g (X), resulting in the effect
Y := f (X) + g (X) × E Y .
These models are more flexible than the ANMs due to their ability to also cover multiplicative noise models with the scale functions. LOCI (Immer et al., 2023) chooses the Gaussian likelihood to estimate the mean and scale functions, and recovers the noise from the fitted models. Similar to RESIT, LOCI also consider HSIC as a criterion in addition to the likelihood for predicting the causal direction. Since the Gaussian likelihood may not be robust to epistemic uncertainty, ROCHE (Tran et al., 2024a) suggested a more robust estimation for LSNMs by replacing the Gaussian likelihood with a likelihood based on Student's t-distribution.
Principle of Independent Causal Mechanisms Beside the algorithmic complexity, there are other approaches for interpreting the principle of independent causal mechanisms (ICMs, Peters et al., 2017, Sec. 2.1), which assumes the independence between the marginal distribution of the cause and the conditional distribution of the effect given the cause.
With the assumption of low noise levels and invertible causal mechanisms, IGCI (Daniušis et al., 2010) formulates this independence using orthogonality in information space to distinguish cause and effect, which is implemented by the relative entropy distances. CDCI (Duong & Nguyen, 2022) postulates that due to this ICM principle, the shape of the conditional distributions will be invariant, and compute the variations in shape to find the causal directions.
The algorithmic complexity-based methods (Marx & Vreeken, 2019a;b;Tagasovska et al., 2020) do not only consider the model fitness objective as in FCM-based methods but also examine the complexity of the model. SLOPE and SLOPER (Marx & Vreeken, 2017;2019b) use a set of basis functions to regress the data globally and locally to account for both deterministic and non-deterministic functions, and compute the codelengths for encoding the data with two-part codes. SLOPPY (Marx & Vreeken, 2019a) is an improvement of RECI (Blöbaum et al., 2018) that utilizes regularized regressions of Identifiable Regressionbased Scoring Functions and find the one with the lowest regularized score to find the minimal model in each direction. QCCD (Tagasovska et al., 2020) uses non-parametric conditional quantile regression methods and encodes the data via each quantile model. Our work-COMIC-also belongs to this category where the conditional distributions of the data are modeled by neural networks and encoded by the variational Bayesian coding scheme. This approach allows for more flexibility compared to previous methods while allowing for the balance between model fitness and model complexity. Another study by Dhir et al. (2024a) interprets the principle of ICMs from the view of the Bayesian model selection and proposes using the marginal likelihoods estimated by latent variable Gaussian processes (GPLVM, Titsias & Lawrence, 2010) for identifying the causal direction.
this section cite: ['b51', 'b22', 'b5', 'b47', 'b51', 'b22', 'b5', 'b47', 'b5', 'b47', 'b16', 'b23', 'b23', 'b8', 'b14', 'b55', 'b37', 'b3', 'b55']

Section: Preliminaries
In this work, the assumption of causal sufficiency is adopted, which is similar to previous publications (Immer et al., 2023;Marx & Vreeken, 2017;2019a;b;Mooij et al., 2016;Peters et al., 2014;Tagasovska et al., 2020;Tran et al., 2024a). This means that we assume that there is no hidden confounder between the two random variables. In other words, given two variables X and Y , if they are not independent, either X or Y will be the cause of the other.
As mentioned in Sec. 2, interpreting the principle of independent causal mechanisms is one of two major approaches for determining the direction of a causal relation. The algorithmic interpretation of this independence via the Kolmogorov complexity originates from the postulate about the algorithmic independence of conditionals (Janzing & Schölkopf, 2010, Eq. 26) as follows:
Postulate 1 (Algorithmic Independence of Conditionals, Janzing & Schölkopf, 2010). Let G be causal hypothesis, represented by a directed acyclic graph (DAG), over a set of d variables X 1 , . . . , X d with a joint density
p (X 1 , . . . , X d ), which is lower semi-computable, that is, K (p (X 1 , . . . , X d )) < ∞.
The causal hypothesis is only acceptable if the shortest description (i.e., the Kolmogorov complexity) of the joint density K (p (X 1 , . . . , X d )) is equal to a concatenation of the shortest description of the Markov kernels up to an independent additive constant. This postulate can be described formally as
K (p (X 1 , . . . , X d )) + = d j=1 K (p (X j | PA G,j )) , (1
)
where K (•) is the Kolmogorov complexity, PA G,j denotes the parents of X j in the causal hypothesis G, and + = denotes equality up to a constant, which is independent of p (•).
In the bivariate setting with two random variables X and Y , if X causes Y (denoted as X → Y ), Eq. ( 1) will become
K (p (X, Y )) + = K (p (X)) + K (p (Y | X)) . (2)
Following the algorithmically independent conditionals, the algorithmic independence of Markov kernels has also been postulated by Janzing & Schölkopf (2010); Mooij et al. (2010) as follows Postulate 2 (Algorithmic Independence of Markov Kernels, Janzing & Schölkopf, 2010). If X → Y , the marginal distribution of the cause p (X) and the conditional distribution of the effect given the cause p (Y | X) are algorithmically independent of each other. In other words, their algorithmic mutual information (I A ) will be equal to zero up to an additive constant,
I A (p (X) : p (Y | X)) + = 0,(3)
and this independence does not hold in the other direction.
From these postulates, Mooij et al. (2010, Thm. 1) induce a rule for identifying the causal direction.
Theorem 3.1 (Asymmetry in Complexities of Markov Kernels, Mooij et al., 2010). If X is the cause of Y and Pos. 2 holds, the description of the joint distribution K (p (X, Y )) via the description of the marginal distribution of the cause K (p (X)) and the description of the conditional distribution of the effect given the cause K (p (Y | X)) is the most succinct one, or formally,
K (p (X)) + K (p (Y | X)) + ≤ K (p (Y )) + K (p (X | Y )) .(4)
As a consequence of this rule, a causal indicator score can be obtained for the assumed causal directions of X → Y as follows
∆ X→Y := K (p (X)) + K (p (Y | X)) ,(5)
and vice versa for the remaining direction of Y → X. From these indicator scores, we can infer that
X → Y if ∆ Y →X - ∆ X→Y > 0 and Y → X if ∆ Y →X -∆ X→Y < 0.
The Kolmogorov complexity is not computable in practice (Li & Vitányi, 2019). Hence, the causal indicators in the previous section are substituted by approximating approaches such as Minimum Message Length (MML, Wallace & Freeman, 1987) or Minimum Description Length (MDL, Rissanen, 1978) in previous information-theoretic methods (Mooij et al., 2010;Marx & Vreeken, 2017;2019b;a). MML and MDL share a two-part coding principle where the complexity or codelengthfoot_2 Lfoot_3-p of the data D is computed via a model M ∈ M by combining the complexity (fitness) of the data given that model L 1 (D | M ) and the complexity of the model L 2 (M ) as
L 2-p M (D) := L 1 (D | M ) + L 2 (M ) .(6)
The codelength with a model M * that minimizes this equation is appointed as an instantiation for the algorithmic complexity. If there are multiple solutions for M * , the one with the smallest model complexity L 1 is selected to model the data. From this two-part code, we can attain an approximation for the causal indicator score in Eq. ( 5) as follows
∆2-p X→Y := L 2-p M * X (X) + L 2-p M * Y |X (Y | X) ,(7)
where M * X and
M * Y |X are models that minimize L 2-p M * X (X) and L 2-p M * Y |X (Y | X), respectively.
The definitions of Kolmogorov complexity and algorithmic mutual information, as well as the discussion of the MDL-based instantiation of Kolmogorov complexity in the context of causal discovery, as referenced in this section, are provided in App. A.
this section cite: ['b23', 'b37', 'b43', 'b47', 'b55', 'b42', 'b42', 'b31', 'b62', 'b50', 'b42', 'b37']

Section: COMIC: Bayesian Compression for
Identifying Causal Direction
this section cite: []

Section: Classes of Models for the Conditionals
For the conditional distribution, most previous MDL-related studies choose the set M of candidate functions by predefining a list of basis functions (Marx & Vreeken, 2017;2019b) or using more advanced regression methods such as cubic spline regression (Marx & Vreeken, 2019a). Although these classes of functions have the model codelengths that are easily computable (e.g., through the number of bits of linear parameters in polynomial regressions), the fitness can be compromised when the ground truth models are more complex, leading to suboptimal codelengths. Dhir et al. (2024a) utilize GPLVM (Titsias & Lawrence, 2010) to improve the fitness; however, this substantially increases the computational complexity due to the poor scalability of GPs.
Neural networks are one class of models that can overcome these limitations thanks to their universality in approximations (Hornik et al., 1989) and better scalability. Additionally, the complexity of neural networks has also been well studied and implemented with different encoding methods (Blier & Ollivier, 2018;Louizos et al., 2017;Voita & Titov, 2020). Hence, neural network can be a viable class of models for computing the codelengths of conditional distributions. Moreover, due to their flexibility and capability of approximating a wide range of functions, neural networks allow us to only focus on this class of models where complexity scores are determined solely by their parameters.
The prequential code (Dawid, 1984) and the variational Bayesian code (Honkela & Valpola, 2004) are two effective MDL approaches for encoding the conditional distribution modeled by neural networks (Blier & Ollivier, 2018;Grünwald, 2007;Voita & Titov, 2020). The former encodes the model implicitly through sequential data transmission, as seen in applications such as time series. The latter involves predefining the priors over the parameters and utilizes variational inference to learn the posteriors from the samples. In contrast to the online prequential code, this approach is aligned with the two-part code. Despite these differences in coding strategies, both methods yield consistent results, as noted by Voita & Titov (2020). We opted for the variational Bayesian code to assess the codelengths because it explicitly captures the model complexity and does not necessitate a specific order of data transmission.
this section cite: ['b37', 'b56', 'b21', 'b2', 'b35', 'b60', 'b9', 'b20', 'b2', 'b17', 'b60', 'b60']

Section: Variational Bayesian Code for Evaluating Complexity of Neural Networks
The problem of encoding a conditional distribution p (Y | X) is often defined via a transmitting perspective.
Alice has a dataset D N := x (i) , y (i) N i=1 that needs to be transported to Bob, who has already got the input samples of this dataset x (i) N i=1 . The most efficient method is to encode the conditional distribution p (Y | X) so that Bob can predict the remaining output part y (i) N i=1 of D N . The variational Bayesian code is a two-part code in MDL where both Alice and Bob first designate a class of model M = {p (y | x, θ) | θ ∈ Θ}, where θ represents the parameters of the conditional probability density function p (y | x, θ), and a prior distribution for the parameters with the probability density function p (θ). The corresponding two-part codelength 2 for this setting can be computed as
L 2-p p(θ) y (1:N ) | x (1:N ) := -log p y (1:N ) | x (1:N ) , θ -log p (θ) ,(8)
where the former term corresponds to L 1 , the latter corresponds L 2 in Eq. ( 6), and p y
(1:N ) | x (1:N ) , θ := N i=1 p y (i) | x (i) , θ .
The variational Bayesian code is based on the bits-back coding scheme (Wallace, 1990;Hinton & van Camp, 1993). In this scheme, Alice employs a codelength with redundant code and computes the codelength with respect to that auxiliary information (i.e., the redundant code). After that, Bob will perform the same learning process and observe the choice made by this process to retrieve the auxiliary information and the transmitted data (Honkela & Valpola, 2004).
By applying this scheme, instead of finding a point estimate of the parameters θ * that minimize Eq. ( 8), we introduce redundant code by choosing the parameters from a variational distribution q ϕ (θ) and compute the expected codelength on this distribution. In this scenario, the amount of redun- (Honkela & Valpola, 2004). We can obtain the amount of original information being transmitted by deducting the excessive entropy H (q ϕ (θ)) from the expectation codelength as follows
dant code to encode q ϕ (θ) is its entropy H (q ϕ (θ)) = E q ϕ (θ) [-log q ϕ (θ)]
L var q ϕ (θ) y (1:N ) | x (1:N ) := E q ϕ (w) L 2-p y (1:N ) | x (1:N ) , θ -H (q ϕ (θ)) (9) := -E q ϕ (w) log p y (1:N ) | x (1:N ) , θ + KL (q ϕ (θ) || p (θ)) ,(10)
where the first term corresponds to the fitness of the data given the model and the second term corresponds to the complexity of the model of the two-part MDL principle (Honkela & Valpola, 2004;Louizos et al., 2017).
-L var q ϕ (θ) y (1:N ) | x (1:N )
is known as the evidence lower bound (ELBO) in the variational inference problem (Blei et al., 2017). From this perspective, by minimizing Eq. ( 10), we can expect to achieve the negative logarithm of the evidence in the Bayesian inference problem, which is also known as the Bayesian codelength in MDL coding (Grünwald, 2007) as follows
L Bayes p(θ) y (1:N ) | x (1:N ) := -log p y (1:N ) | x (1:N ) , θ p (θ) dθ.(11)
The gap between the optimal codelength
L Bayes p(θ) y (1:N ) | x (
1:N ) and its upper bound
L var q ϕ (θ) y (1:N ) | x (1:N )
can also be formulated from the variational perspective as follows
L var q ϕ (θ) y (1:N ) | x (1:N ) -L Bayes p(θ) y (1:N ) | x (1:N ) = KL q ϕ (θ) || p θ | x (1:N ) , y (1:N ) ,(12)
where p θ | x (1:N ) , y (1:N ) is the posterior distribution of the parameters given the observed samples. From this formulation, it is obvious that we can retrieve the Bayesian codelength iff. q ϕ (θ) converges to p θ | x (1:N ) , y (1:N ) .
The selection of the priors p (θ) and the variational distributions q ϕ (θ) over the parameters is a necessary step in Bayesian learning. In this work, we employ Gaussian distributions as the family for both the priors p (θ) and the mean-field variational posteriors q ϕ (θ). The details on these priors and variational posteriors are provided in App. B.
this section cite: ['b61', 'b19', 'b20', 'b20', 'b20', 'b35', 'b1', 'b17']

Section: Identifying Causal Direction via the Codelengths
From the formulation of variational Bayesian codelength of the conditional distribution described in the previous section, we can approximate the conditional codelength for the assumed causal direction X → Y using Eq. ( 10) with the chosen likelihood distribution being the Gaussian distribution as
p y (1:N ) | x (1:N ) , θ := N i=1 N y (i) | µ x (i) ; θ , σ 2 x (i) ; θ ,(13)
where µ (•; θ) and σ (•; θ) are modeled via a neural network f Y : R × Θ → R 2 with parameters θ ∈ Θ. In particular, we implement a neural network with one hidden layer and two output nodes f Y,1 (•; θ) and f Y,2 (•; θ), and compute the parameters of the likelihood as follows
µ (•; θ) = f Y,1 (•; θ) and σ (•; θ) = ζ (f Y,2 (•; θ)) , (14
)
where ζ (•) is a positive link function, such as the exponential or softplus function, for ensuring the standard deviation values being positive.
Both x (1:N ) and y (1:N ) are standardized with respect to their corresponding sample means and standard deviations to avoid the scale-based bias on the identifiability (Reisach et al., 2021). Regarding the codelength of the assumed cause X, we adopt standard Gaussian distribution N (x | 0, 1) to encode the data using the marginal codelength, which is a common choice in previous methods (Mooij et al., 2016;Immer et al., 2023). We denote this marginal codelength L N x (1:N ) .
The approximated causal indicator score for this direction is the sum of the two codelengths
∆var X→Y D N := L N x (1:N ) + L var q ϕ * (θ) y (1:N ) | x (1:N ) ,(15)
where q ϕ * (θ) is the model that minimizes the variational Bayesian codelength. The corresponding score ∆Y →X for the reversed direction Y → X is estimated by a similar procedure. Once the scores in both directions are obtained, the difference between them provides a final score:
∆var D N := ∆var Y →X D N -∆var X→Y D N , (16
)
which indicates the inferred causal direction with its absolute value reflecting the confidence of the inference.
If the optimized variational distribution q ϕ * (θ) converges to the posterior distribution p θ | x (1:N ) , y (1:N ) and N (0, 1) is the ground truth distribution of p (X), we can expect ∆X→Y to converge to the Bayesian causal indicator score
∆ Bayes X→Y D N := L N x (1:N ) + L Bayes p(θ) y (1:N ) | x (1:N )(17)
:= -log p D N | M X→Y ,(18)
where p D N | M X→Y is the marginal likelihood of the dataset D N factorized in accordance with the causal model M X→Y . Conversely, ∆ Bayes Y →X D N can be achieved if Y ∼ N (0, 1) and the minimized variational codelength in this case also converges to the Bayesian codelength.
this section cite: ['b49', 'b43', 'b23']

Section: Causal Identifiability
The identifiability of our approach is closely related to the identifiability of Bayesian causal models via marginal likelihoods. First, we introduce the definition of separablecompatibility by Dhir et al. (2024a), which is a necessary condition of two Bayesian causal models being unidentifiable via their marginal likelihoods, regardless of the dataset D N . Definition 4.1 (Separable-Compatibility of Bayesian Causal Models, informally restated from Dhir et al., 2024a). Two causal models M X→Y and M Y →X are separablecompatible if the anti-causal factorizations of M X→Y and M Y →X respectively belong to the same classes of distributions as M Y →X and M X→Y , and the priors of these anti-causal factorizations can also be factorized with respect to the priors of M Y →X and M X→Y .
In this definition, the anti-causal factorization of the causal model M X→Y refers to the factorization of the joint distribution p (X, Y ) into p (Y ) and p (X | Y ) with respect to M X→Y . Similarly, the anti-causal factorization of the causal model M Y →X involves factorizing p (X, Y ) into p (X) and p (Y | X) according to M Y →X . If the anticausal factorization the causal model M X→Y results in the same distributions as the causal factorization o f M Y →X , or vice versa, the two causal models are said to be separablecompatible.
As a result of our Bayesian coding scheme, the identifiability of our method is verifiable through an orthogonal perspective of marginal likelihoods (Dhir et al., 2024a). Let us assume that the approximated scores in Eq. ( 15) would converge to the Bayesian scores in Eq. ( 17), our results on the non-separable-compatibility of Bayesian causal models employed in our method can be presented as follows: ) respectively be the marginal likelihoods of the data given the causal models M X→Y and M Y →X . Because of the non-separablecompatibility in Prop. 4.2, there exists a dataset D N whose causal direction is identifiable via the difference between Bayesian indicator scores:
∆ Bayes D N := ∆ Bayes Y →X D N -∆ Bayes X→Y D N . (19)
Cor. 4.3 implies that in a large sample limit, the Bayesian causal indicator scores defined in Eq. ( 17) can distinguish the cause and effect from observational data. Details on the causal identifiability via marginal likelihoods and the proof of Prop. 4.2 are further discussed in App. C.
this section cite: []

Section: Experiments
Throughout this section, the empirical performance of our COMIC approach is evaluated in comparison with stateof-the-art complexity-based and regression-based methods for bivariate causal discovery. Implementation details and descriptions of the benchmarks related to this section are provided in App. D and E, respectively.
this section cite: []

Section: Experimental Settings
Benchmarks Following previous works (Immer et al., 2023;Marx & Vreeken, 2019a;Tagasovska et al., 2020;Tran et al., 2024a), the experiments in this section utilizes both synthetic and real-world data for evaluation. The synthetic data consists of 12 common benchmarks. The first collection of synthetic datasets, including AN, AN-s, LS, LS-s, and MN-U, are proposed by Tagasovska et al. (2020). The next group of simulated benchmarks, comprising SIM, SIM-c, SIM-G, and SIM-ln, is introduced by Mooij et al. (2016). The remaining synthetic benchmarks consist of the CE-Multi, CE-Net, and CE-Cha datasets described by Goudet et al. (2018). For real-world data, we choose the Tübingen cause-effect pairs (Mooij et al., 2016).  & Vreeken, 2019b) and SLOPE (Marx & Vreeken, 2017), QCCD (Tagasovska et al., 2020), IGCI (with uniform and Gaussian reference measures, Daniušis et al., 2010), GPLVM (Dhir et al., 2024a), LOCI (Immer et al., 2023), and CAM (Bühlmann et al., 2014). The marginal likelihood-based objective in COMIC achieves promising results with enhanced performance compared to methods with the maximum likelihood-based objectives in LOCI and CAM, especially on the real-world Tübingen benchmark. In comparison to most complexity-based methods, COMIC also achieves better performance on multiple synthetic datasets and obtains comparable results on real-world data. on the principle of ICMs include SLOPPY (Marx & Vreeken, 2019a), SLOPE (Marx & Vreeken, 2017;2019b), QCCD (Tagasovska et al., 2020), and IGCI (Daniušis et al., 2010), and GPLVM (Dhir et al., 2024a). For FCM-based methods, we choose CAM (Bühlmann et al., 2014) as a representative for ANM-based methods, and LOCI (Immer et al., 2023) with maximum likelihood scoring to represent LSNM-based methods. Some of the ICM-based methods have different variants, which we also include in the evaluations.
Baselines
this section cite: ['b23', 'b55', 'b55', 'b43', 'b15', 'b43', 'b37', 'b55', 'b8', 'b23', 'b5', 'b37', 'b55', 'b8', 'b5', 'b23']

Section: Evaluation Metrics
The identification result of each pair in a dataset is considered as a sample in a binary classification problem of that dataset. Hence, the accuracy score and the area under receiver operating characteristic curve (AUROC) are commonly utilized as evaluation metrics for the task of bivariate causal discovery. As the ground truth directions in the benchmarks are imbalanced, we compute the bidirectional AUROC (Bi-AUROC, Guyon et al., 2019, Sec. 2.4.3), which is the average of the forward AUROC and the backward AUROC corresponding to X → Y and Y → X, respectively.
this section cite: []

Section: Results & Discussion
The experimental results on all aforementioned benchmarks are visualized in Fig. 1 and 2. The overall results in Fig. 2 demonstrate that our COMIC approach achieves secondbest performance in comparison to complexity-based and maximum likelihood-based methods. Our method is also more consistent across all benchmarks, as indicated by the smaller standard deviation of the accuracy and Bi-AUROC scores compared to most baseline methods. Furthermore, COMIC outperforms the maximum likelihood-based methods, i.e., LOCI and CAM, suggesting that its marginal likelihood-based objective delivers promising results with improved performance relative to methods with the maximum likelihood-based objectives.
GPLVM is the baseline that attains the best overall results. However, it is important to note that GPLVM also requires the most substantial computational resources, with execution on GPUs (as noted in App. D), compared to COMIC and other baselines, which can be executed on CPUs. This high resource demand stems from the computational complexity of Gaussian processes with latent variables and the extensive use of random restarts for hyperparameter selection.
Performance on Synthetic Benchmarks On synthetic AN, AN-s, LS, LS-s, and MN-U benchmarks, COMIC perfectly determines the causal directions in every case. QCCD, GPLVM, and LOCI also achieve equivalent accuracy and Bi-AUROC scores on these datasets. These findings align with theoretical expectation that models identifiable via maximum likelihood are also identifiable via marginal likelihood (Dhir et al., 2024a). IGCI with Gaussian reference measure is another method that excels in this group of datasets. The remaining baseline approaches do not perform as well on the more challenging LS, LS-s, and MN-U sets. In particular, the predictions of SLOPER, SLOPE, IGCI with uniform reference measure, and CAM on these datasets are noticeably suboptimal compared to other baselines.
On the more challenging SIM and SIM-c datasets, our COMIC approach demonstrates performance comparable to complexity-based methods, such as SLOPPY with AIC, SLOPER, and SLOPE. A majority of baseline methods perform decently on SIM-G and SIM-ln, which are more manageable compared to the previous two sets. COMIC achieves the second-highest scores on SIM-ln, with slightly lower accuracy and Bi-AUROC than GPLVM. One unexpected outlier among the baseline models is IGCI, which exhibits the lowest accuracy and Bi-AUROC on SIM-ln, despite its intended focus on telling apart causal directions in low-noise scenarios. Moreover, SLOPPY with AIC, SLOPER, Slope, and QCCD underperform relative to COMIC on the SIM-G dataset with nearly Gaussian distributions over the causes. The diverse causal modeling of CE-Multi results in varying performance across benchmarks. Information-theoretic approaches, including COMIC, SLOPPY with BIC, SLOPER, SLOPE, IGCI with Gaussian reference measure, and GPLVM, tend to perform well on this benchmark. QCCD and uniform-referenced IGCI, while also belonging to this category, perform less effectively. Since CAM is designed for additive noise models, it exhibits inefficacious performance on this dataset. A majority of methods featuring regressions in the learning process can adequately predict causal relations of the CE-Net benchmark. Hence, IGCI is the only baseline tested that underperforms on this benchmark. Due to the difficulty of CE-Cha, the results on this set resemble those on SIM, where most methods struggle, except for GPLVM. Performance on Real-World Benchmark On the realworld Tübingen benchmark, complexity-based methods, including our COMIC approach, GPLVM, SLOPPY with AIC, SLOPE, and QCCD, achieve comparable accuracy and Bi-AUROC scores. This highlights the effectiveness of compression-based methods, whose Bi-AUROC scores are notably higher than the remaining maximum likelihoodbased methods that focus solely on model fitness. As noted by Marx & Vreeken (2019a), SLOPPY with AIC offers greater flexibility for more complex datasets, such as this one, and performs better than the BIC variant. A similar pattern appears in IGCI, where the uniform reference measure variant obtains higher accuracy and Bi-AUROC. Although SLOPER outperforms its respective variant SLOPE on previous benchmarks, it is surpassed by SLOPE on this benchmark.
this section cite: []

Section: Conclusion
In this work, we have proposed COMIC-a neural network compression-based approach for determining the cause and effect via the variational Bayesian code-where a more universal and scalable class of neural networks is utilized for modeling the conditionals to improve fitness and induce better codelengths. With the variational Bayesian coding scheme, the algorithmic complexity of these networks can be assessed empirically to approximate the theoretical Kolmogorov complexity, and its identifiability can also be scrutinized from a marginal likelihood-based perspective. The effectiveness of COMIC has been validated through comprehensive experiments, delivering promising results and demonstrating enhanced performance compared to most related methods based on the compression and FCM regression objectives across multiple benchmarks.
Limitations & Future Work One persistent limitation of our current work is the non-convexity of the learning objective, which may require further investigation into the convergence and consistency. Additionally, the use of standard Gaussian codelength to encode the marginal distribution of the cause can introduce a bias toward "more Gaussian" causes, potentially posing a hindrance in intricate settings.
In future work, we plan to adapt our approach to multivariate settings, explore alternative priors and likelihoods, and consider other marginal likelihood estimation techniques for neural networks, such as Laplace approximation, to enhance its applicability. Regarding the multivariate extension, we also provide a brief discussion in App. H on potential adaptations of the bivariate methods, including ours, to multivariate data.
this section cite: []

Section: References
Ref_id:b0 Title: Variational inference for Dirichlet process mixtures Year: (2006)
Ref_id:b1 Title: Variational inference: A review for statisticians Year: (2017)
Ref_id:b2 Title: The description length of deep learning models Year: (2018)
Ref_id:b3 Title: Cause-effect inference by comparing regression errors Year: (2018)
Ref_id:b4 Title: Prequential MDL for causal structure learning with neural networks Year: (2021)
Ref_id:b5 Title: CAM: Causal additive models, high-dimensional order search and penalized regression Year: (2002)
Ref_id:b6 Title: Wide mean-field Bayesian neural networks ignore the data Year: (2022)
Ref_id:b7 Title: BCD Nets: Scalable variational approaches for Bayesian causal discovery Year: (2021)
Ref_id:b8 Title: Inferring deterministic causal relations Year: (2010)
Ref_id:b9 Title: Present position and potential developments: Some personal views: Statistical theory: The prequential approach Year: (1984)
Ref_id:b10 Title: Bayesian structure learning with generative flow networks Year: (2022)
Ref_id:b11 Title: Bivariate causal discovery using Bayesian model selection Year: (2024)
Ref_id:b12 Title: Continuous Bayesian model selection for multivariate causal discovery Year: (2024)
Ref_id:b13 Title: A meta-learning approach to Bayesian causal discovery Year: ()
Ref_id:b14 Title: Bivariate causal discovery via conditional divergence Year: (2022)
Ref_id:b15 Title: Learning functional causal models with generative neural networks Year: (2018)
Ref_id:b16 Title: Measuring statistical dependence with Hilbert-Schmidt norms Year: (2005)
Ref_id:b17 Title: The minimum description length principle Year: (2007)
Ref_id:b18 Title: Cause effect pairs in machine learning Year: (2019)
Ref_id:b19 Title: Keeping the neural networks simple by minimizing the description length of the weights Year: (1993)
Ref_id:b20 Title: Variational learning and bitsback coding: An information-theoretic view to Bayesian learning Year: (2004)
Ref_id:b21 Title: Multilayer feedforward networks are universal approximators Year: (1989)
Ref_id:b22 Title: Nonlinear causal discovery with additive noise models Year: (2008)
Ref_id:b23 Title: On the identifiability and estimation of causal location-scale noise models Year: (2023)
Ref_id:b24 Title: Causal inference using the algorithmic Markov condition Year: (2010)
Ref_id:b25 Title: Information-geometric approach to inferring causal directions Year: (2012)
Ref_id:b26 Title: Causal discovery toolbox: Uncovering causal relationships in Python Year: (2020)
Ref_id:b27 Title: Don't Confound Yourself: Causality from Biased Data Year: (2024)
Ref_id:b28 Title: Causal discovery with hidden confounders using the algorithmic Markov condition Year: (2023)
Ref_id:b29 Title: A method for stochastic optimization Year: (2015)
Ref_id:b30 Title: On tables of random numbers Year: (1961)
Ref_id:b31 Title: An Introduction to Kolmogorov Complexity and Its Applications Year: (2019)
Ref_id:b32 Title: A skewness-based criterion for addressing heteroscedastic noise in causal discovery Year: ()
Ref_id:b33 Title: Differentiable Bayesian structure learning Year: (2021)
Ref_id:b34 Title: Amortized inference for causal structure learning Year: (2022)
Ref_id:b35 Title: Bayesian compression for deep learning Year: (2017)
Ref_id:b36 Title: Comparison of approximate methods for handling hyperparameters Year: (1999)
Ref_id:b37 Title: Telling cause from effect using MDL-based local and global regression Year: (2017)
Ref_id:b38 Title: Identifiability of cause and effect using regularized regression Year: (2019)
Ref_id:b39 Title: Telling cause from effect by local and global regression Year: (2019)
Ref_id:b40 Title: Formally justifying MDL-based inference of cause and effect Year: ()
Ref_id:b41 Title: Discovering fully oriented causal networks Year: (2021)
Ref_id:b42 Title: Probabilistic latent variable models for distinguishing between cause and effect Year: (2010)
Ref_id:b43 Title: Distinguishing cause from effect using observational data: Methods and benchmarks Year: (2016)
Ref_id:b44 Title: Priors for infinite networks Year: (1996)
Ref_id:b45 Title: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b46 Title:  Year: (2009)
Ref_id:b47 Title: Causal discovery with continuous additive noise models Year: (2014)
Ref_id:b48 Title: Elements of Causal Inference: Foundations and Learning Algorithms Year: (2017)
Ref_id:b49 Title: Beware of the simulated DAG! Causal discovery benchmarks may be easy to game Year: (2021)
Ref_id:b50 Title: Modeling by shortest data description Year: (1978)
Ref_id:b51 Title: A linear non-Gaussian acyclic model for causal discovery Year: (2006)
Ref_id:b52 Title: A formal theory of inductive inference Year: (1964)
Ref_id:b53 Title: A formal theory of inductive inference Year: (1964)
Ref_id:b54 Title: Ladder variational autoencoders Year: (2016)
Ref_id:b55 Title: Distinguishing cause from effect using quantiles: Bivariate quantile causal discovery Year: (2020)
Ref_id:b56 Title: Gaussian process latent variable model Year: (2010)
Ref_id:b57 Title: Differentiable Bayesian structure learning with acyclicity assurance Year: (2023)
Ref_id:b58 Title: Robust estimation of causal heteroscedastic noise models Year: (2024)
Ref_id:b59 Title: Constraining acyclicity of differentiable Bayesian structure learning with topological ordering Year: (2024)
Ref_id:b60 Title: Information-theoretic probing with minimum description length Year: (2020)
Ref_id:b61 Title: Classification by minimum-message-length inference Year: (1990)
Ref_id:b62 Title: Estimation and inference by compact coding Year: (1987)
Ref_id:b63 Title: Computing with infinite networks Year: (1996)
Ref_id:b64 Title: On the identifiability of the post-nonlinear causal model Year: (2009)
Ref_id:b65 Title:  Year: ()
