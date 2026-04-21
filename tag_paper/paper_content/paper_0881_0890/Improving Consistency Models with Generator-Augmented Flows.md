Title: Improving Consistency Models with Generator-Augmented Flows
Abstract: Consistency models imitate the multi-step sampling of score-based diffusion in a single forward pass of a neural network. They can be learned in two ways: consistency distillation and consistency training. The former relies on the true velocity field of the corresponding differential equation, approximated by a pre-trained neural network. In contrast, the latter uses a single-sample Monte Carlo estimate of this velocity field. The related estimation error induces a discrepancy between consistency distillation and training that, we show, still holds in the continuous-time limit. To alleviate this issue, we propose a novel flow that transports noisy data towards their corresponding outputs derived from a consistency model. We prove that this flow reduces the previously identified discrepancy and the noise-data transport cost. Consequently, our method not only accelerates consistency training convergence but also enhances its overall performance. The code is available at: github.com/thibautissenhuth/consistency GC.

Section: Introduction
A large family of diffusion (Ho et al., 2020), score-based (Song et al., 2021;Karras et al., 2022), and flow models (Liu et al., 2023;Lipman et al., 2023) have emerged as stateof-the-art generative models for image generation. Since they are costly to use at inference time -requiring several neural function evaluations -, many distillation techniques have been explored (Salimans and Ho, 2022;Meng et al., 2023;Sauer et al., 2023). One of the most remarkable ap- proach is consistency models (Song et al., 2023;Song and Dhariwal, 2024). Consistency models lead to high-quality one-step generators, that can be trained either by distillation of a pre-trained velocity field (consistency distillation), or as standalone generative models (consistency training) by approximating the velocity field through a one-sample Monte Carlo estimate.
The corresponding estimation error naturally induces a discrepancy between consistency distillation and training. While Song et al. (2023) hinted that it would resolve in the continuous-time limit, we show that this discrepancy persists in both the gradients and values of the loss functions. Interestingly, this discrepancy vanishes when the difference between the target velocity field and its Monte-Carlo approximation approaches zero. However, this is not the case with the independent coupling (IC) between data and noise used to construct the standard estimate. It is unclear how to improve this one-sample estimate without access to the true underlying diffusion model.
The approach we adopt in this paper to alleviate this issue involves altering the velocity field -thereby changing the target flow -to reduce the variance of its one-sample estimator. One possible solution to this problem is to resort to optimal transport (OT) to learn on a deterministic coupling. OT has been succesfully adopted in diffusion (Li et al., 2024), consistency (Dou et al., 2024), and flow matching (Pooladian et al., 2023) models. However, due to the prohibitive cubic complexity of OT solvers (e.g. Hungarian matching algorithm), such methods need to be applied at the minibatch level. This incurs an OT approximation error (Fatras et al., 2021;Sommerfeld et al., 2019) and stochasticity of the data-noise coupling, thus not solving the consistency training issue.
In our approach, we propose to use the consistency model, assumed to be an approximation of the target diffusion flow, to construct additional trajectories. The consistency model serves as a proxy to reduce the expected deviation between the velocity field and its estimator. More precisely, from an intermediate point computed from an IC, we let the consistency model predict the corresponding endpoint, supposedly close to the data distribution. This predicted endpoint is coupled to the same original noise vector, defining a generator-  augmented coupling (GC). We show empirically that the resulting generator-augmented flow presents compelling properties for training consistency models, in particular a reduced deviation between the velocity field and its estimator, and decreased transport costs -as supported by theoretical and empirical evidence. This can be observed in Figure 1. From this, we derive practical algorithms to train consistency models with generator-augmented flows, leading to improved performance and faster convergence compared to standard and OT-based consistency models.
Let us summarize our contributions below.
• We prove that in the continuous-time limit consistency training and consistency distillation loss function converge to different values and we provide a closed-form expression of this discrepancy.
• We propose a novel type of flows that we denote generator-augmented flows. It relies on generatoraugmented coupling (GC) that can be used to train a consistency model.
• We provide theoretical and empirical insights into the advantages of GC. We show that generator-augmented flows have smaller discrepancy to consistency distillation than IC consistency training, and that they reduce data-noise transport costs.
• We derive practical ways to train consistency models with GC. Our approach based on a joint learning strategy leads to faster convergence and improves the performance compared to the base model and OT-based approaches on image generation benchmarks.
this section cite: ['b9', 'b32', 'b10', 'b17', 'b16', 'b24', 'b20', 'b26', 'b33', 'b30', 'b33', 'b15', 'b4', 'b23', 'b6', 'b29']

Section: Notation.
We consider an empirical data distribution p ⋆ and a noise distribution p z (e.g. Gaussian), both defined on R d . We denote by q a joint distribution of samples from p ⋆ and p z . We equip R d with the dot product ⟨x, y⟩ = x ⊤ y and write ∥x∥ = ⟨x, x⟩ 1/2 for the Euclidean norm of x. We use a distance function D : R d × R d → [0, ∞) to measure the distance between two points from R d . sg denotes the stop-gradient operator.
In consistency models, we consider diffusion processes of the form x t = x ⋆ + σ t z, where x ⋆ ∼ p ⋆ , z ∼ p z , and σ t is monotonically increasing for t ∈ [0, T ], where T ∈ R + .
We denote the distribution of x t by p(x t ), or simply p t . Conditional distributions or finite-dimensional joint distributions of x t 's are denoted similarly. When considering a discrete formulation with N intermediate timesteps, we denote the intermediate points as x ti = x ⋆ + σ ti z, where t i is strictly increasing for i ∈ {0, . . . , N }, with t 0 = 0 and t N = T . The values of σ 0 and σ T are chosen to be sufficiently small and large, respectively, so that p 0 ≈ p ⋆ and p T ≈ N (0, σ 2 T I).
this section cite: []

Section: Consistency Distillation Versus Training
In this section, we provide the required background on diffusion and consistency models (Sections 2.1 and 2.2), then discuss the discrepancy between consistency distillation and consistency training (Section 2.3) which we theoretically characterize in continuous-time.
this section cite: []

Section: Flow and Score-Based Diffusion Models
Score-based diffusion models (Ho et al., 2020;Song et al., 2021) can generate data from noise via a multi-step process consisting in numerically solving either a stochastic differential equation (SDE), or equivalently an ordinary differential equation (ODE). Although SDE solvers generally exhibit superior sampling quality, ODEs have desirable properties. Most notably, they define a deterministic mapping from noise to data. Recently, Liu et al. (2023) and Lipman et al. (2023) generalize diffusion to flow models, which are defined by the following probability flow ODE (PF-ODE):
dx = v t (x) dt,(1)
where v t (x) = E[ ẋt |x t = x]
is the velocity field. Note that ẋt is defined as the random variable ẋt = d(x⋆+σtz) dt = σt z, and is not to be confused with the time-derivative of the ODE, v t .
In the context of consistency models (Song et al., 2023;Song and Dhariwal, 2024), the most common choice is v t (x) =σt σ t ∇ x log p t (x) dt, in particular the EDM formulation (Karras et al., 2022) where σ t = t and thus v t (x) = -t∇ x log p t (x). Here, ∇ x log p t , a.k.a. the score function, can be approximated with a neural network s ϕ (x, t) (Vincent, 2011;Song and Ermon, 2019).
this section cite: ['b9', 'b32', 'b33', 'b30', 'b10', 'b31']

Section: Consistency Models
Numerically solving an ODE is costly because it requires multiple expensive evaluations of the velocity function. To alleviate this issue, Song et al. (2023) propose training a consistency model f θ , which learns the output map of the PF-ODE, i.e. its flow, such that:
f θ (x t , σ t ) = x 0 ,(2)
for all (x t , σ t ) ∈ R d × [σ 0 , σ T ] that belong to the trajectory of the PF-ODE ending at (x 0 , σ 0 ).
Equation ( 2) is equivalent to (i) enforcing the boundary condition f θ (x 0 , σ 0 ) = x 0 , and (ii) ensuring that f θ has the same output for any two samples of a single PF-ODE trajectory -the consistency property. (i) is naturally satisfied by the following model parametrization:
f θ (x ti , σ ti ) = c skip (σ ti )x ti + c out (σ ti )F θ (x ti , σ ti ), (3
) where c skip (σ) = σ 2 d σ 2 d +(σ-σ0) 2 , c out (σ) = σ d •(σ-σ0) √ σ 2 d +σ 2 , σ 2
d the variance of data, and F θ is a neural network. This ensures c skip (0) = 1, c out (0) = 0. (ii) is achieved by minimizing the distance between the outputs of two same-trajectory consecutive samples using the consistency loss:
L CD (θ) = E qI(x⋆,z),p(xt i+1 |x⋆,z) λ(σ ti )D sg f θ (x Φ ti , σ ti ) , f θ (x ti+1 , σ ti+1 ) , (4
)
where (x ⋆ , z) is sampled from the independent coupling q I (x ⋆ , z) = p ⋆ (x ⋆ )p z (z), i is an index sampled uniformly at random from {0, 1, . . . , N -1}, x ti+1 = x ⋆ + σ ti+1 z, and x Φ ti is computed by discretizing the PF-ODE with the Euler scheme as follows:
x Φ ti = Φ(x ti+1 , t i+1 ) = x ti+1 + (t i -t i+1 )v ti+1 (x ti+1 ).
(5) This loss can be used to distill a score model into f θ .
In the case of consistency training, Song et al. (2023) circumvent the lack of a score function by noting that
v ti+1 (x) = E[ ẋti+1 |x ti+1 = x].
In light of this, its singlesample Monte Carlo estimate ẋti+1 is used instead in Equation (5) to replace the intractable x Φ ti by x ti = x ⋆ + σ ti z in the consistency loss:
L CT (θ) = E qI(x⋆,z),p(xt i ,xt i+1 |x⋆,z) λ(σ ti )D sg f θ (x ti , σ ti ) , f θ (x ti+1 , σ ti+1 ) . (6)
this section cite: ['b33', 'b33']

Section: Discrepancy Between Consistency Training and Distillation and Velocity Field Estimation
Naturally
, replacing v t by its single-sample estimate ẋt makes consistency training deviate from consistency distillation in discrete time. Still, Song et al. (2023, Theorems 2 and 6) suggest that this discrepancy disappears in continuous-time since L CT (θ) = L CD (θ) + o( 1 /N) and the corresponding gradients are equal in some cases. This equality is then used in work of Lu and Song (2024), concurrent to ours, to train continuous-time consistency models at the cost of an elaborate architectural design. Without disproving these results, we find that scaling issues and lack of generality soften the claim of a closed gap between consistency training and distillation.
Indeed, we provide in the following theorem a thorough theoretical comparison of L CT and L CD . We first prove that they converge to different values in the continuous-time limit. The difference is captured by a regularization term that depends on the discrepancy between the velocity field and its estimate. Moreover, we show that the limits of the scaled gradients do not coincide in the general case, except when the (asymptotic) quadratic loss is used. The proof, and further discussion on why this discrepancy did not appear in Song et al. (2023), can be found in Appendix A.1.
Theorem 1 (Discrepancy between consistency distillation and consistency training objectives). Assume that the distance function is given by D(x, y) = φ(∥x -y∥) for a continuous convex function φ : [0, ∞) → [0, ∞) with φ(x) ∼ Cx α as x → 0 + for some C > 0 and α ≥ 1, and that the timesteps are equally spaced, i.e., t i = iT N . Furthermore, assume that the Jacobian ∂f θ ∂x does not vanish identically. Then the following assertions hold:
(i) The scaled consistency losses N α L CD (θ) and N α L CT (θ) converge as N → ∞. Moreover, the minimization objectives corresponding to these limiting scaled consistency losses are not equivalent, and their difference is given by:
lim N →∞ N α L CT (θ) -L CD (θ) = CT α-1 R(θ),(7)
where R(θ) is defined by
R(θ) = T 0 λ(σ t )E ∥∂ CT f θ ∥ α -∥∂ CD f θ ∥ α dt (8)
and satisfies R(θ) > 0, with
∂ CT f θ = ∂f θ ∂σ (x t , σ t ) σt + ∂f θ ∂x (x t , σ t ) • ẋt ,(9)
∂ CD f θ = ∂f θ ∂σ (x t , σ t ) σt + ∂f θ ∂x (x t , σ t ) • v t (x t ). (10) In particular, if α = 2, R(θ) = T 0 λ(σ t )E ∂f θ ∂x (x t , σ t ) ẋt -v t (x t ) 2 dt.(11)
(ii) The scaled gradient N α-1 ∇ θ L CD (θ) and N α-1 ∇ θ L CT (θ) converge as N → ∞. Moreover, if α ̸ = 2, then their respective limits are not identical as functions of θ:
lim N →∞ N α-1 ∇ θ L CT (θ) ̸ = lim N →∞ N α-1 ∇ θ L CD (θ). (12
)
This theorem reveals that the optimization problems of consistency training and distillation differ not only in discrete time but also in continuous-time. It even highlights a discrepancy between, firstly, the limiting gradients in continuous-time -although they are equal for α = 2 -and, secondly, the gradients of the limiting losses, which differ because of R(θ), even when α = 2.
This analysis shows the importance of employing probability paths whose sample path derivatives ẋt are aligned with the velocity field v t (x t ). In particular, if a diffusion process x t satisfies ẋt = v t (x t ), we have R(θ) = 0 and equal gradients for all α ≥ 1. Hence, for such x t , consistency training and consistency distillation would be reconciled both in discrete time and in the continuous-time limit.
However, it is unclear how to directly improve the singlesample estimation ẋt of v t (x t ). In particular, increasing the number of samples per point x t to reduce its variance is not tractable, as it requires sampling from the inverse diffusion process p(x ⋆ |x t ). Therefore, we adopt an alternative approach to alleviate the discrepancy identified in this section, which involves altering the velocity field -thereby changing the target flow -to reduce the variance of its one-sample estimator. This approach is reminiscent of recent work tackling the data-noise coupling that we discuss in the following section.
this section cite: ['b19', 'b33']

Section: Reducing the Discrepancy with Data-Noise Coupling
Beyond independent coupling (IC). From Section 2.2, it appears that ẋt is computed through an IC q I = p ⋆ (x ⋆ )p z (z) of data and noise, in a similar fashion to flow matching (Lipman et al., 2023;Kingma and Gao, 2024). Making correlated choices of data and noise beyond IC could then help align ẋt and v t (x t ), thereby resolving the discrepancy from the previous section.
The reliance on IC in consistency and flow models is increasingly recognized as a limiting factor. Recent advancements suggest that improved coupling mechanisms could enhance both training efficiency and the quality of generated samples in flow matching (Liu et al., 2023;Pooladian et al., 2023) and diffusion models (Li et al., 2024). By reducing the variance in gradient estimation, enhanced coupling can accelerate training. Additionally, improved coupling could decrease transport costs and straighten trajectories, yielding better-quality samples. In a different context, ReFlow (Liu et al., 2023) leverages couplings provided by the ODE solver in a flow framework, and demonstrates that it reduced transport costs. Moreover, Lee et al. (2023) propose to learn an encoder from data to noise, and use this encoder as a way to construct a coupling when training a flow model.
Couplings based on optimal transport (OT) solvers.
OT is a particularly appealing solution for our alignment problem. Indeed, if we consider a quadratic cost and distributions with bounded supports, OT is a no-collision transport map (Nurbekyan et al., 2020), i.e. x t can be sampled by a unique pair of points (x ⋆ , z). Thus ẋt = v t (x t ), implying R(θ) = 0 in Theorem 1. Several approaches have precisely targeted the reduction of transport cost in flow and consistency models. Pooladian et al. (2023) have more directly explored OT coupling within the framework of flow matching models. They show that deterministic and non-crossing paths enabled by OT with infinite batch size lowers the variance of gradient estimators. Experimentally, they assess the efficacy of OT solvers, such as Hungarian matching and Sinkhorn algorithms, in coupling batches of noise and data points. Dou et al. (2024) have successfully adopted this approach in consistency models, while Li et al. (2024) applied OT to diffusion models. However, due to the prohibitive cubic complexity of OT solvers, OT has to be applied by minibatch for matching samples (x ⋆ , z). Besides an OT approximation error, this incurs the loss of the no-collision property, making R(θ) non-zero in real use-cases. Another line of works using OT tools with score-based models relies on the Schrödinger Bridge formulation (De Bortoli et al., 2021;Shi et al., 2023;Korotin et al., 2024;Tong et al., 2024), which has mostly proven benefits on transfer tasks.
Our approach. In this paper, we use a consistency model as a proxy of the flow of a diffusion process to reduce transport costs. While not fully solving the alignment issue, we will show that our method present reduced transport costs and better alignment than dedicated OT-based methods.
this section cite: ['b16', 'b11', 'b17', 'b23', 'b15', 'b17', 'b14', 'b21', 'b23', 'b4', 'b15', 'b2', 'b27', 'b12']

Section: Consistency Models with Generator-Augmented Flows
Here, we introduce our method, denoted as generatoraugmented flows, which relies on a generator-augmented coupling (GC). We capitalize on the true diffusion flow f (i.e. an ideal consistency model) to map noisy points towards the PF-ODE solution. We present theoretical and empirical evidences that GC not only reduces the data-noise transport cost but also narrows the gap between consistency distillation and consistency training. We will discuss how to train GC consistency models jointly with f in Section 5.
this section cite: []

Section: Generator-Augmented Coupling (GC): Definition and Training Loss
The solution proposed in this work involves harnessing the diffusion flow, computed from a consistency model, to create a novel form of coupling. The idea is to leverage the properties and accumulated knowledge within an ideal consistency model, f , to construct pairs of points. To achieve this, we first sample an intermediate point, which is done as usual by sampling x ⋆ ∼ p ⋆ and z ∼ p z using the IC between the two distributions, and then predict the data point xti via the consistency model:
(x ⋆ , z) ∼ q I ; x ti = x ⋆ + σ ti z; xti = sg( f (x ti , σ ti )).(13)
Although xti depends on the timestep t i , it is important to note that it (supposedly) follows the distribution p 0 . This xti is coupled with z, thereby defining our generatoraugmented coupling (GC) q, which we use to construct the pair of points (x ti , xti+1 ):
(x ti , z) ∼ q; xti = xti + σ ti z; xti+1 = xti + σ ti+1 z.
(14) These intermediate points can serve to define a new consistency training loss: Generator-augmented trajectories satisfy the boundary conditions of diffusion processes. We note the two following important properties of the distribution of xt :
L GC (θ) = E q(xt i ,z),p(xt i ,xt i+1 |xt i ,z) λ(σ ti )D sg(f θ (x ti , σ ti )), f θ (x ti+1 , σ ti+1 ) . (15)
p(x 0 ) = p(x 0 ) ≈ p ⋆ , p(x T ) ≈ p(x T ) ≈ p(σ T z). (16)
The first property is achieved thanks to the boundary condition of the consistency model (c.f. Section 2.1) , and the second property by construction of the diffusion process which ensures that the noise magnitude is significantly larger than xti for large t. However, for the timesteps t ∈ (0, T ) the marginal distributions p(x t ) and p(x t ) do not necessarily coincide.
this section cite: []

Section: Properties of Generator-Augmented Flows
Here, we present some properties of generator-augmented flows that motivate them for training consistency models.
this section cite: []

Section: REDUCING R(θ) WITH GC
In Theorem 1, we proved that the continuous-time consistency training objective decomposes into the sum of the consistency distillation objective and a regularizer term:
L CT (θ) = L CD (θ) + R(θ).
Here, we study a proxy term for R(θ) that is easier to calculate:
Rt = E ẋt -v t (x t ) 2 . (17
)
This quantity measures the expected distance between the true velocity field and its one-sample Monte Carlo estimate.
We study Rt,IC , Rbatch-OT , and Rt,GC . They are the respective proxy regularizer term for each type of probability path. Note that Rt,GC depends on the endpoint predictor, a consistency model, which impacts both probability paths and velocity fields. Our goal is to compare those proxy regularizer terms, in order to demonstrate that GC does lead to a smaller discrepancy than IC. We further motivate the use of this proxy, in regards with Theorem 1, in Appendix A.4.
In the following theorem, proved in Appendix A.2, we show that Rt decays faster for GC than for IC.
Theorem 2. Assume that the data distribution contains more than a single point. Also, assume that the generatoraugmented coupling between the predicted data point xt and noise z is computed via an ideal consistency model f , i.e., the flow of the PF-ODE. Then, as t → ∞,
Rt,GC ≪ Rt,IC .(18)
Empirical validation. Evaluating Rt requires computing the difference between the sample path derivative ẋt and the velocity field v t (x t ). In the EDM setting, this difference can be approximated using a denoiser. Indeed, ẋt = z and v t (
x t ) = E[ ẋt |x t ] = E[z|x t ] = E[ xt-x⋆ t |x t ] = 1 t (x t -D ⋆ (x t , t))
with an optimal denoiser D ⋆ . The optimal denoiser can be approximated by a denoiser network D ϕ . Finally, we have: ẋt -v t (x t ) ≈ z-1 t (x t -D ϕ (x t , t)). Since IC, batch-OT, and GC define different p t 's and v t 's, we train a different denoiser D ϕ for each coupling. In Figure 2, we report the results from the comparison of the three proxy terms on CIFAR-10. We observe that Rt,GC < Rt,batch-OT < Rt,IC and that the gap increases with t, corroborating our theoretical findings (Theorem 2).
this section cite: []

Section: REDUCING TRANSPORT COST WITH GC
Here, we investigate the average transport cost between the noise z ∼ p z and the predicted data point x ∼ p ⋆ as a measure of the efficiency of the data-noise coupling. Recall that the diffusion process is given by x t = x ⋆ + σ t z. Then, knowing that the consistency model f satisfying the boundary condition f (x 0 , σ 0 ) = x 0 , we define the function c(t) as:
c(t) = E qI(x⋆,z) f (x t , σ t ) -z 2 . (19
)
c(0) = E qI(x⋆,z) [∥x 0 -z∥ 2 ]
and c(t) represent the transport costs of, respectively, IC and GC. We show below, with proofs in Appendix A.3, that c(t) is decreasing for σ t close to zero and for large σ t .
Lemma 1 (Transport cost of GC coupling). Assume that f is a continuously differentiable function representing the ground-truth consistency model, i.e. the flow of the PF-ODE induced by the diffusion process
x t . Define w t = z -E[z|x t ] = 1 σt ( ẋt -E[ ẋt | x t ]
). Then:
c ′ (t) = -2 σt E   ∂ f ∂x (x t , σ t ) • w t , w t   .(20)
Corollary 1 (Decreasing transport cost of GC coupling in t → 0 + ). There exists a t * > 0 such that for all t ∈ [0, t * ], the derivative of c(t) takes the form c ′ (t) = -2 σt a t with a t > 0. Hence for σt positive, the cost is decreasing.
In particular, in the EDM setting where σ t = t, c(t) is decreasing for small t.
The proof of this corollary proceeds by noting that for t = 0, the consistency model f (x, t) is an identity function, its Jacobian is an identity matrix, and thus
a t = E[∥w t ∥ 2 ].
Using the continuity of Jacobian elements and invoking intermediate value theorem on a t concludes the proof.
Corollary 2 (Decreasing transport cost of GC coupling in t ≈ t max ). Assume that the consistency model f (x, σ) is a scaling function f (x, σ t ) = σ0 σt x. Then, we have
c ′ (t) = -2 σtσ0 σt E[∥w t ∥ 2 ].
In particular, c(t) is decreasing whenever σ t is increasing.
We note that, while the assumption of the consistency model being a scaling function is strong, it nonetheless bears some degree of truth for t ≈ t max , see Lemma 3 of Appendix A.
this section cite: []

Section: Experimental validation.
As stressed in Section 3, a line of work has brought evidence that reducing the transport cost between noise and data distributions could fasten the training and help produce better samples. We compare the quadratic transport costs involved in IC, batch-OT (Pooladian et al., 2023;Dou et al., 2024), and GC (resp. c(0), c OT (0), and c(t)). Results are presented in Figure 3. Interestingly, GC reduces transport cost more than batch-OT on CIFAR-10 because batch-OT is tied to the batch data points x t whereas our computed xt are not.
this section cite: ['b23', 'b4']

Section: Training With Generator-Augmented Flows for Image Generation
In this section, we present a methodology to train consistency models with GC on unconditional image generation. To construct points drawn from GC trajectories (x ti ), our theory requires an optimal predictor f on intermediate points drawn from IC (x ti ). Thus, this lets us two potential training strategies: (i) pre-train an IC generator, and leverage it to construct GC trajectories that train a GC model; (ii) a joint learning strategy: train a single consistency model from scratch with both types of trajectories. Note that in this second setting, the model is unique: f = f θ . The second option is more appealing, since it is a simple one-stage training. We demonstrate that the joint learning approach improves performance and accelerates convergence compared to standard consistency models.
Our experiments are done on the following datasets: CIFAR-10 (Krizhevsky, 2009), ImageNet (Deng et al., 2009), CelebA (Liu et al., 2015) and LSUN Church (Yu et al., 2015). For the evaluation metrics, we report the  2024)). Details are provided in Appendix D. The code is shared in the supplementary material and will be open-sourced upon publication for reproducibility.
this section cite: ['b3', 'b18']

Section: GC with Pre-Trained Endpoint Predictor
Our theoretical results assume having access to an ideal generator on IC trajectories, meaning that the generator ap- proximates the diffusion flow output. To train a consistency model on GC, we can thus rely on a separate endpoint predictor pre-trained on IC (iCT-IC): f ≡ g ϕ (cf. Section 4.1). This network predicts the endpoint:
xti = g ϕ (x ti , σ ti ).
During the training of the consistency model on GC, g ϕ is kept frozen and considered a proxy of the true flow, as in our theoretical results. In Figure 4, we report the performance of consistency models on CIFAR-10 trained with GC using two different g ϕ : (i) a g ϕ fully trained as standard iCT-IC with 100k training steps; (ii) a weak g ϕ partially trained as iCT-IC with 20k training steps.
Finding 1. Using a partially pre-trained and frozen endpoint predictor, trained on IC trajectories, allows to train a consistency model with GC and which converges faster. However, the performance of the GC model depends on the quality of the endpoint predictor on IC trajectories.
It is important to note that this setup is not practical, as it requires pre-training a standard consistency model. We aim for a training methodology that accelerates convergence and improves performance when training from scratch, without doubling the number of required training iterations.
this section cite: []

Section: GC from Scratch with Joint Learning
In this section, we propose to learn simultaneously a single model on IC and GC trajectories from the start of the training, i.e. f ≡ sg(f θ ) (cf. Section 4.1). Thereby, we combine the training of the ideal IC predictor with the training of GC model based on this predictor. We introduce a joint learning factor µ: at each training step, training pairs are drawn from GC with probability µ, while the remaining pairs are drawn from standard IC. The loss can be written on average as:
L GC-µ (θ) = µL GC (θ) + (1 -µ)L CT (θ)(21)
Table 1. iCT-IC is the standard improved consistency model with independent coupling (Song and Dhariwal, 2024); iCT-OT is iCT with minibatch optimal transport coupling (Pooladian et al., 2023;Dou et al., 2024); iCT-GC (µ = 0.5) is our proposed GC with joint learning.
Dataset Model FID ↓ KID (×10 2 ) ↓ IS ↑ CIFAR-10 iCT-IC 7.42 ± 0.04 0.44 ± 0.03 8.76 ± 0.06 iCT-OT 6.75 ± 0.04 0.36 ± 0.04 8.86 ± 0.09 iCT-GC (µ = 0.5) 5.95 ± 0.05 0.26 ± 0.02 9.10 ± 0.05 ImageNet (32 × 32) iCT-IC 14.89 ± 0.17 1.23 ± 0.05 9.46 ± 0.06 iCT-OT 14.13 ± 0.17 1.18 ± 0.05 9.62 ± 0.06 iCT-GC (µ = 0.5) 13.99 ± 0.28 1.13 ± 0.03 9.77 ± 0.07 CelebA (64 × 64) iCT-IC 15.82 ± 0.13 1.31 ± 0.04 2.33 ± 0.00 iCT-OT 13.63 ± 0.13 1.09 ± 0.03 2.40 ± 0.01 iCT-GC (µ = 0.5) 11.74 ± 0.08 0.91 ± 0.04 2.45 ± 0.01 LSUN Church (64 × 64) iCT-IC 10.58 ± 0.11 0.73 ± 0.03 1.99 ± 0.01 iCT-OT 9.71 ± 0.13 0.64 ± 0.03 2.00 ± 0.01 iCT-GC (µ = 0.5) 9.88 ± 0.07 0.66 ± 0.04 2.14 ± 0.01
We denote this joint learning procedure as GC (µ = •).
Hence, GC (µ = 0) corresponds to the standard IC procedure, while GC (µ = 1) corresponds to training only with GC points.Note that GC (µ = 1) is not expected to work, since our theoretical guarantees assume an optimal IC predictor. The detailed algorithm is presented in Algorithm 1 in Appendix. We apply this joint learning to four image datasets, and include comparisons to iCT with batch-OT (Dou et al., 2024) as an additional baseline. Results across multiple datasets and metrics are presented in Table 1, and visual examples are shown in Figure 8 in Appendix.
Finding 2. Joint learning of IC and GC trajectories consistently improves results compared to the base IC model and outperforms batch-OT in most cases.
As shown in Figure 5, we observe an interesting interpolation phenomenon between µ = 0 and µ = 1. For µ = 0, we recover the steady FID improvement typical of IC training. As µ increases, the convergence of the generative model accelerates. For 0.3 ≤ µ ≤ 0.7, on CIFAR-10, convergence speed and final FID are improved compared to IC and batch-OT models. For µ = 1, the FID score decreases faster than other configurations early in the training process, but it soons increases as training progresses further. It is explained by the poor performance of the predictions on IC yielding a deviation from the ideal IC predictor from Section 4. For the other datasets, we simply chose µ = 0.5 and report those results. We provide further detail on the sensitivity of our results to the choice of µ in Appendices C.1 and D.
this section cite: ['b30', 'b23', 'b4', 'b4']

Section: GC in the ECT Setting
As an additional experiment, we explore the recent ECT setting (Geng et al., 2024) on CIFAR-10, where consis- 6.39 ± 0.20 tency models are fine-tuned from a pre-trained diffusion model. This approach enables training high-quality consistency models in one GPU-hour, though it requires an already trained diffusion model.
We compare IC and GC trajectories in this setting, with both short (approximately one GPU-hour) and long (100k steps, 1 GPU-day) training times. Using the referenced hyperparameters selected by Geng et al. (2024), we observe a consistent advantage for GC, with an optimal µ value of 0.3. These preliminary results, summarized in Table 2, align with our previous findings on the iCT setting, further supporting the effectiveness of GC.
this section cite: ['b7', 'b7']

Section: Conclusion
In this paper, we identify a discrepancy between consistency training and consistency distillation. Building on this theoretical analysis, we introduce generator-augmented flows and show that they reduce a proxy term measuring this discrepancy. Additionally, generator-augmented flows decrease the data-to-noise transport cost, as demonstrated by theory and experiments. Finally, we derive practical algorithms for training consistency models using generatoraugmented flows and demonstrate improved empirical performance.
this section cite: []

Section: References
Ref_id:b0 Title: International Conference on Learning Representations Year: (2018)
Ref_id:b1 Title: Symbolic discovery of optimization algorithms Year: (2023)
Ref_id:b2 Title: Diffusion Schrödinger bridge with appli-cations to score-based generative modeling Year: (2021)
Ref_id:b3 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b4 Title: A unified framework for consistency generative modeling Year: (2024)
Ref_id:b5 Title: The epistemic threat of deepfakes Year: (2021)
Ref_id:b6 Title: Rémi Flamary, Rémi Gribonval, and Nicolas Courty. Minibatch optimal transport distances; analysis and applications Year: (2021)
Ref_id:b7 Title: Consistency models made easy Year: (2024)
Ref_id:b8 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b9 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b10 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b11 Title: Understanding diffusion objectives as the elbo with simple data augmentation Year: (2024)
Ref_id:b12 Title: Light Schrödinger bridge Year: (2024)
Ref_id:b13 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b14 Title: Minimizing trajectory curvature of ODE-based generative models Year: (2023-07)
Ref_id:b15 Title: Immiscible diffusion: Accelerating diffusion training with noise assignment Year: (2024)
Ref_id:b16 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b17 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b18 Title: Deep learning face attributes in the wild Year: (2015)
Ref_id:b19 Title: Simplifying, stabilizing and scaling continuous-time consistency models Year: (2024)
Ref_id:b20 Title: On distillation of guided diffusion models Year: (2023)
Ref_id:b21 Title: No-collision transportation maps Year: (2020)
Ref_id:b22 Title: PyTorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b23 Title: Multisample flow matching: Straightening flows with minibatch couplings Year: (2023-07)
Ref_id:b24 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b25 Title: Improved techniques for training GANs Year: (2016)
Ref_id:b26 Title:  Year: (2023)
Ref_id:b27 Title: Diffusion Schrödinger bridge matching Year: (2023)
Ref_id:b28 Title: Maxim Grechkin, and William Falcon. TorchMetrics -measuring reproducibility in Py-Torch Year: (2022-02)
Ref_id:b29 Title: Optimal transport: Fast probabilistic approximation with exact solvers Year: (2019)
Ref_id:b30 Title: Improved techniques for training consistency models Year: (2024)
Ref_id:b31 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b32 Title: Scorebased generative modeling through stochastic differential equations Year: (2021)
Ref_id:b33 Title: Consistency models Year: (2023-07)
