Title: UNIVERSAL INVERSE DISTILLATION FOR MATCHING MODELS WITH REAL-DATA SUPERVISION (NO GANS)
Abstract: While achieving exceptional generative quality, modern diffusion, flow, and other matching models suffer from slow inference, as they require many steps of iterative generation. Recent distillation methods address this by training efficient onestep generators under the guidance of a pre-trained teacher model. However, these methods are often constrained to only one specific framework, e.g., only to diffusion or only to flow models. Furthermore, these methods are naturally datafree, and to benefit from the usage of real data, it is required to use an additional complex adversarial training with an extra discriminator model. In this paper, we present RealUID, a universal distillation framework for all matching models that seamlessly incorporates real data into the distillation procedure without GANs. Our RealUID approach offers a simple theoretical foundation that covers previous distillation methods for Flow Matching and Diffusion models, and is also extended to their modifications, such as Bridge Matching and Stochastic Interpolants. The code can be found in https://github.com/David-cripto/RealUID.

Section: INTRODUCTION
In generative modeling, the goal is to learn to sample from complex data distributions (e.g., images), and two powerful paradigms for it are the diffusion models (DM) and the flow matching (FM) models. While they share common principles and are even equivalent under certain conditions (Holderrieth et al., 2024;Gao et al., 2025), they are typically studied separately. Diffusion models (Sohl-Dickstein et al., 2015;Ho et al., 2020;Song et al., 2021) transform data into noise through a forward process and then learn a reverse-time stochastic differential equation (SDE) to recover the data distribution. Training minimizes score-matching objectives, yielding unbiased estimates of intermediate scores. Sampling requires simulating the reverse dynamics, which is computationally heavy but delivers high-quality and diverse results. Flow Matching (Lipman et al., 2023;Liu, 2022) instead interpolates between source and target distributions by learning the vector field of an ordinary differential equation (ODE). The field is estimated through unbiased conditional objectives, but the resulting ODE often has curved trajectories, making sampling costly due to expensive integration. Beyond these, Bridge Matching (Peluchetti, 2023;Liu et al., 2022) and Stochastic Interpolants (Albergo et al., 2023) generalize the framework and naturally support data couplings, which are crucial for data-to-data translation. Since all of the above optimize conditional matching objectives to recover an ODE/SDE for generation, we refer to them collectively as matching models.
Despite their success, matching models share a major drawback: sampling is slow, as generation requires integrating many steps of an SDE or ODE. To address this, a range of distillation techniques have been proposed to compress multi-step dynamics into efficient one-step or few-step generators. Although matching models follow a similar mathematical framework, many distillation works consider only one particular framework, e.g., only diffusion models (Zhou et al., 2024a;b), Flow Matching (Huang et al., 2024), or Bridge Matching (Gushchin et al., 2025). Furthermore, these distillation methods are data-free by construction and cannot benefit from the utilization of real data without using additional GAN-based losses. Thus, the following problems remain: 1. Similar distillation techniques developed separately for similar matching models frameworks. 2. Absence of a natural way to incorporate real data in distillation procedures (without GANs).
this section cite: ['b12', 'b4', 'b31', 'b11', 'b33', 'b21', 'b22', 'b28', 'b23', 'b0', 'b13', 'b9']

Section: Contributions.
In this paper, we address these issues and present the following main contributions:
1. We present the Universal Inverse Distillation with real data (RealUID) framework for matching models, including diffusion and flow matching models ( §3) as well as Bridge Matching and Stochastic Interpolants (Appendix C). It unifies previously introduced Flow Generator Matching (FGM), Score Identity Distillation (SiD) and Inverse Bridge Matching Distillation (IBMD) methods ( §3.2) for flow, score and bridge matching models respectively, provides simple yet rigorous theoretical explanations based on a linearization technique, and reveals the connections between these methods and inverse optimization ( §3.3). 2. Our RealUID introduces a novel and natural way to incorporate real data directly into the distillation loss, eliminating the need for extra adversarial losses which require additional discriminator networks used in GANs from the previous works ( §3.4).
this section cite: []

Section: BACKGROUNDS ON TRAINING AND DISTILLING MATCHING MODELS
We describe the Diffusion Models and Flow Matching frameworks ( §2.1) and distillation methods for them ( §2.3). Then, we discuss how real data can be added to distilling methods via GANs ( §2.4)
Preliminaries. We work on the D-dimensional Euclidean space R D . This space is equipped with the standard scalar product ⟨x, y⟩ = D d=1 x d y d , the ℓ 2 -norm ∥x∥ = ⟨x, x⟩ and ℓ 2 -distance ∥x -y∥, ∀x, y ∈ R D . We consider probability distributions from the set P(R D ) of absolutely continuous distributions with finite variance and support on the whole R D .
this section cite: []

Section: DIFFUSION AND FLOW MODELS
Diffusion models (Sohl-Dickstein et al., 2015;Ho et al., 2020;Song et al., 2021) consider a forward noising process p t that gradually transforms clean data p 0 into a noise p T on the time interval [0, T ]:
dx t = f t • x t • dt + g t • dw t , x 0 ∼ p 0 ,
where f t and g t are time-dependent scalars and w t is a standard Wiener process. This process defines a conditional distributions p t (x t |x 0 ) = N (α t x 0 |σ 2 t I), where
α t = exp t 0 f s ds , σ t = t 0 g 2 s exp -2 s 0 f u du ds 1/2 .
Each conditional distribution admits a conditional score function, describing it: s t (x t |x 0 ) := ∇ xt log p t (x t |x 0 ) = -(x t -α t x 0 )/σ 2 t . The reverse dynamics from the noise distribution p T to the data distribution p 0 is provided by the following reverse-time SDE with a reverse-time Wiener process wt :
dx t = (f t • x t -g 2 t • s t (x t ))dt + g t d wt ,
where γ t are some positive weights. The reverse dynamics admits a probability flow ODE (PF-ODE):
dx t = u t (x t )dt, u t (x t ) := (f t • x t -g 2 t • s t (x t )/2), which provides faster inference than the SDE formulation. Flow Matching framework (Lipman et al., 2023;Liu et al., 2023) constructs the flow directly by learning the drift u t (x t ). Specifically, for each data point x 0 ∼ p 0 , one defines a conditional flow p t (x t |x 0 ) with the corresponding conditional vector field u t (x t |x 0 ) generating it via ODE:
dx t = u t (x t |x 0 )dt.
Then, to construct the flow between the data p 0 and noise p T , one needs to compute the unconditional vector field u t (x t ) = E x0∼p0(•|xt) [u t (x t |x 0 )] which generates the flow p t (x t ) = p(x t |x 0 )p(x 0 )dx 0 . It can be done by minimizing the following Conditional Flow Matching (CFM) loss:
L CFM (v, p 0 ) = E t∼[0,T ],x0∼p0,xt∼pt(•|x0) ∥v t (x t ) -u t (x t |x 0 )∥ 2 2 .(2)
In practice, the most popular choice is the Gaussian conditional flows p t (x t |x 0 ) = N (α t x 0 , σ 2 t I). For this conditional flow samples can be obtained as x t = α t x 0 +σ t ϵ, ϵ ∼ N (0, I) and the conditional drift can be calculated as u t (x t |x 0 ) = αt x 0 + σt ϵ.
We recall data-to-data models working with data couplings, such as Bridge Matching and Stochastic Interpolants, in Appendices C.1 and C.2, respectively.
this section cite: ['b31', 'b11', 'b33', 'b21', 'b24']

Section: UNIVERSAL LOSS FOR MATCHING MODELS
From a mathematical point of view, it was shown in (Holderrieth et al., 2024;Gao et al., 2025) that flow and diffusion models basically share the same loss structure. We recall this structure, but use our own notation. We call diffusion and flow models and their extensions as matching models.
A matching model constructs a probability path p t on the time interval [0, T ], transforming the desired data p 0 ∈ P(R D ) to the noise p T ∈ P(R D ). This path is built as a mixture of simple conditional paths p t (•|x 0 ) conditioned on samples x 0 ∼ p 0 , i.e., p t (x t ) = R D p t (x t |x 0 )p 0 (x 0 )dx 0 , ∀x t ∈ R D . The path p t determines the function f p0 : [0, T ] × R D → R D which recovers it (e.g., score function or drift). The conditional paths also determine their own simple conditional functions f p0 (•|x 0 ) so that they express
f p0 t (x t ) = E x0∼p0(•|xt) [f p0 t (x t |x 0 )], where p 0 (•|x t )
is a data distribution p 0 conditioned on sample x t at time t. Since f p0 cannot be computed directly, it is approximated by a trainable function f : [0, T ] × R D → R D via minimizing the squared ℓ 2 -distance between the functions at each time t from [0, T ] and point x t ∼ p t :
∥f t (x t )-f p0 t (x t )∥ 2 = ∥f t (x t )-E x0∼p0(•|xt) [f p0 t (x t |x 0 )]∥ 2 ∝ E x0∼p0(•|xt) [∥f t (x t )-f p0 t (x t |x 0 )∥ 2 ]. We also change the sampling order x t ∼ p t , x 0 ∼ p 0 (•|x t ) to more natural x 0 ∼ p 0 , x t ∼ p t (•|x 0 ).
Definition 1. We define Universal Matching (UM) loss L UM (f, p 0 ) that takes trainable function f and distribution p 0 ∈ P(R D ) as arguments and upon minimization over f returns the function f p0 :
L UM (f, p 0 ):= E t∼[0,T ] E x0∼p0,xt∼pt(•|x0) [∥f t (x t )-f p0 t (x t |x 0 )∥ 2 ], f p0 := arg min f L UM (f, p 0 ).
(3) The notation t ∼ [0, T ] hides time sampling and loss weighting inherent to the given matching model.
this section cite: ['b12', 'b4']

Section: DISTILLATION OF MATCHING-BASED MODELS
To solve the long inference problem of matching models, a line of distillation approaches sharing similar principles was introduced: Score Identity Distillation (Zhou et al., 2024b;a, SiD), Flow Generator Matching (Huang et al., 2024, FGM), and Inverse Bridge Matching Distillation (Gushchin et al., 2025, IBMD), for diffusion, flow, and bridge matching models, respectively.
The SiD approach trains a student generator G θ : Z → R D (parameterized by θ) that produces a distribution p θ 0 from a latent distribution p Z on Z. This approach minimizes the squared ℓ 2 -distance between the known teacher score function s * := arg min s ′ L DSM (s ′ , p * 0 ) on real data p * 0 and the unknown student score function s θ :
min θ E t∼[0,T ] E x θ t ∼p θ t [∥s θ t (x θ t ) -s * t (x θ t )∥ 2 ], s.t. s θ = arg min s ′ L DSM (s ′ , p θ 0 ),(4)
where p θ t is the forward noising process for the generated data p θ 0 . The authors propose the tractable loss with parameter α SiD to approximate the real gradients of (4) :
L SiD (θ) := E t∼[0,T ] E z∼p Z ,x θ 0 =G θ (z),x θ t ∼p θ t [-2ω t • α SiD ∥s * t (x θ t ) -s sg[θ] t (x θ t )∥ 2 + 2ω t ⟨s * t (x θ t ) -s sg[θ] t (x θ t ), s * t (x θ t ) -s θ t (x θ t |x θ 0 )⟩], s θ = arg min s ′ L DSM (s ′ , p θ 0 ), (5
) where w t are normalizing weights and gradients w.r.t. θ are not calculated for the variables under stop-gradient sg[•] operator. The SiD pipeline is two alternating steps: first, refine the fake score s sg [θ] by minimizing DSM loss (1) on new p θ 0 from the previous step. Then, update the generator G θ using the gradient of (5) with frozen s sg [θ] . The α SiD parameter is chosen from the range [0.5, 1.2], although theoretically only the value α SiD = 0.5 restores true gradient as we show in our paper.
The authors of FGM propose a similar approach, but for the flow matching models. Specifically, they also use a generator G θ to produce a distribution p θ 0 , but instead of DSM loss (1), consider CFM loss (2). The method minimizes the squared ℓ 2 -distance between the student and teacher drifts:
min θ E t∼[0,T ] E xt∼p θ t [∥u θ t (x t ) -u * t (x t )∥ 2 ], s.t. u θ := arg min v L CFM (v, p θ 0 ),(6)
where the interpolation path p θ t is constructed between the noise p T and generator p θ 0 distributions. To avoid the same problem of differentiating through arg min operator as in SiD, the authors derive a tractable loss whose gradients match those of (6):
L FGM (θ) := E t∼[0,T ] E z∼p Z ,x θ 0 =G θ (z),x θ t ∼p θ t [-∥u * t (x θ t ) -u sg[θ] t (x θ t )∥ 2(7)
+ 2⟨u * t (x θ t ) -u sg[θ] t (x θ t ), u * t (x θ t ) -u θ t (x θ t |x θ 0 )⟩], s.t. u θ = arg min v L CFM (v, p θ 0 ).
For data-to-data bridge matching models, the IBMD method applies the same idea of minimizing the difference between student and teacher drifts using a similar loss. Notably, all these approaches (SiD, FGM, IBMD) are data-free, i.e., they do not use any real data from p * 0 to train a generator.
this section cite: ['b44']

Section: GANS FOR REAL DATA INCORPORATION
Although FGM and SiD methods exhibit strong performance in one-step generation, the generator in these methods is trained under the guidance of the teacher model alone. This means the generator cannot get more information about the real data that the teacher has learned. For example, it is not expected to correct the teacher's errors. To address this, recent works (Yin et al., 2024a;Zhou et al., 2024a) propose adding real data via GANs (Goodfellow et al., 2014). In such approaches, the encoder of fake model is typically augmented with an additional discriminator head D that distinguishes between the generated and real data noising processes via the following adversarial loss:
L adv = E t∼[0,T ] E x * t ∼p * t ln D t x * t + E x θ t ∼p θ t ln[1 -D t x θ t ] .(8)
The overall objective in such hybrid frameworks (Zhou et al., 2024a) consists of generator loss:
L G θ = L FGM/SiD (θ) + λ G θ adv L adv (θ)
, And fake model/discriminator loss:
L D = L CFM/DSM + λ D adv L adv . Here, λ G θ
adv and λ D adv are weighting coefficients for the adversarial components. Despite empirical gains, the GAN augmentation entails nontrivial costs: it necessitates architectural modifications, such as an auxiliary discriminator head, and inherits the well-known optimization problems of adversarial training, such as non-stationary objectives, mode collapse, and sensitivity to training dynamics.
this section cite: ['b7']

Section: UNIVERSAL DISTILLATION OF MATCHING MODELS WITH REAL DATA
In this section, we present our novel RealUID approach for matching models enhanced by real data. First, we show that the previous data-free distillation methods can be unified under the single UID framework ( §3.1). Then, we describe how this framework is connected to prior works ( §3.2) and inverse optimization ( §3.3). Using this intuition, we propose and discuss the real data modified UID framework (RealUID) with a natural way to incorporate real data without GANs ( §3.4).
this section cite: []

Section: UNIVERSAL INVERSE DISTILLATION
To learn a complex real data distribution p * 0 , one usually trains a teacher function f * := arg min f L UM (f, p * 0 ) that is then used in a multi-step sampling procedure (Def. 1). To avoid time-consuming sampling, one can train a simple student generator G θ : Z → R D with parameters θ to reproduce the real data p * 0 from the distribution p Z on the latent space Z. The teacher function serves as a guide that shows how close the student distribution p θ 0 and the real data p * 0 are. FGM and SiD methods ( §2.3) train such generator via minimizing the squared ℓ 2 -distance between the known teacher function f * and an unknown student function
f θ := arg min f L UM (f, p θ 0 ): E t∼[0,T ] E x θ t ∼p θ t [∥f * t (x θ t ) -f θ t (x θ t )∥ 2 ] = E t∼[0,T ] E x θ t ∼p θ t [∥f * t (x θ t ) -E x θ 0 ∼p θ 0 (•|x θ t ) [f θ t (x θ t |x θ 0 )]∥ 2 ] = E t∼[0,T ] E x θ t ∼p θ t [∥f * t (x θ t )∥ 2 ] -2 • E t∼[0,T ] E x θ t ∼p θ t ,x θ 0 ∼p θ 0 (•|x θ t ) [⟨f * t (x θ t ), f θ t (x θ t |x θ 0 )⟩] + E t∼[0,T ] E x θ t ∼p θ t [∥E x θ 0 ∼p θ 0 (•|x θ t ) [f θ t (x θ t |x θ 0 )]∥ 2 ] not tractable ,(9)
where p θ t is the probability path constructed between generated data p θ 0 and noise p T . The problem is that the final term (9) cannot be calculated directly. This is because it involves the math expectation inside the squared norm, unlike the other terms which are linear in the expectations. It means that a simple estimate of ∥f θ t (x θ t |x θ 0 )∥ 2 using samples x θ 0 and x θ t will be biased. Moreover, to differentiate through the math expectation inside the norm, an explicit dependence of p θ 0 on θ is required, while, in practice, usually only dependence of samples x θ 0 on θ is known.
this section cite: []

Section: Making loss tractable via linearization.
To resolve this, we use the identity ∥a|| 2 = max b∈R D {-∥b∥ 2 + 2⟨b, a⟩}, ∀a ∈ R D . For a fixed time t and point x θ t , we reformulate the squared norm (9) as this identity and parametrize vector b via an auxiliary function δ : [0, T ] × R D → R D :
E t∼[0,T ], x θ t ∼p θ t [∥f * t (x θ t ) -f θ t (x θ t )∥ 2 ] = max δt(x θ t )
E t∼[0,T ],
x θ t ∼p θ t -∥δ t (x θ t )∥ 2 + 2⟨δ t (x θ t ), f * t (x θ t ) -f θ t (x θ t )⟩ = max δt(x θ t )
E t∼[0,T ], (10
x θ t ∼p θ t -∥δ t (x θ t )∥ 2 + 2⟨δ t (x θ t ), f * t (x θ t )⟩ -2⟨δ t (x θ t ), E x θ 0 ∼p θ 0 (•|x θ t ) [f θ t (x θ t |x θ 0 )]⟩
)= max ft(x θ t ) E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) -∥f * t (x θ t )-f t (x θ t )∥ 2 + 2⟨f * t (x θ t )-f t (x θ t ), f * t (x θ t )-f θ t (x θ t |x θ 0 )⟩ (11) = max ft(x θ t ) E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) [∥f * t (x θ t )-f θ t (x θ t |x θ 0 )∥ 2 ] =LUM(f * ,p θ 0 ) -E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) [∥f t (x θ t )-f θ t (x θ t |x θ 0 )∥ 2 ] =LUM(f,p θ 0 ) .(12)
Summary. We build a universal distillation framework as a single min-max optimization (13), implicitly minimizing squared ℓ 2 -distance between teacher and student functions. When real and generated probability paths match, these functions match as well
, and the distance attains its minimum. Theorem 1 (Real data generator minimizes UID loss). Let teacher f * := arg min f L UM (f, p * 0 ) be the minimizer of UM loss (Def. 1) on real data p * 0 ∈ P(R D ). Then, real data generator G θ * s.t. p θ * 0 = p * 0 is a solution to the min-max optimization of Universal Inverse Distillation (UID) loss L UID (f, p θ 0 ) over fake function f and generator distribution p θ 0 :
min θ max f L UID (f, p θ 0 ) := L UM (f * , p θ 0 ) -L UM (f, p θ 0 ) . (13
) Lemma 1 (UID loss minimizes squared ℓ 2 -distance). Maximization of UID loss (13) over fake function f represents the squared ℓ 2 -distance between the student function f θ := arg min f L UM (f, p θ 0 ) and the teacher
f * := arg min f L UM (f, p * 0 ): max f L UID (f, p θ 0 ) = E t∼[0,T ] E x θ t ∼p θ t [∥f * t (x θ t ) -f θ t (x θ t )∥ 2 ].(14)
In UID framework, the trained fake model simply learns the current student function f θ by minimizing UM loss L UM (f, p θ 0 ). Note that for points x θ t out of the generator's domain s.t. p θ t (x θ t ) ≈ 0, the distance (14) vanishes, and the generator cannot receive feedback from the uncovered real data. Moreover, if the teacher function is inaccurate, the generator will learn it with all inaccuracies.
this section cite: []

Section: RELATION TO PRIOR DISTILLATION WORKS
FGM and SiD approaches formulate distillation as a constraint minimization of generator loss subject to the optimal fake model. For generator updates, the explicit UID loss (11) matches FGM loss (7) and SiD loss (5) with α SiD = 0.5. For a fake model, it also minimizes the UM loss on the generated data. The work (Gushchin et al., 2025) was the first to formulate the distillation of bridge matching models in their IBMD framework as a min-max optimization of the single loss (12).
Although previous works derive the same losses, we give a new, simple explanation using a linearization technique. This technique is more powerful and general for handling intractable terms than complex proofs for concrete models from FGM, SiD, IBMD. It allows us to build other distillations, e.g., a loss for minimizing the ℓ 2 -distance instead of the squared one (see Appendix A.4).
this section cite: ['b9']

Section: CONNECTION WITH INVERSE OPTIMIZATION
We derive UID loss (13) by minimizing the squared ℓ 2 -distance between teacher and student functions. However, this loss admits another interpretation: its structure is typical for inverse optimization (Chan et al., 2025). In this framework, one considers a parametric family of optimization problems min f L(f, θ) with objective loss L(f, θ) depending on argument f and parameters θ. The goal is to find the parameters θ * that yield a known, desired solution f * = arg min f L(f, θ * ). One standard way to recover the required parameters is to solve the same min-max problem as (13):
min θ max f {L(f * , θ) -L(f, θ)} ∼ min θ L(f * , θ) -min f {L(f, θ)} .(15)
The inverse problem (15) always has minimum 0 which is attained when θ = θ * .
Although the inverse optimization can handle arbitrary losses L, it does not describe the properties of the optimized functions or how to find solutions. In our case, we show that all losses are tractable and minimize the distances between teacher and student functions (Lemmas 1 and 2).
this section cite: ['b2']

Section: REALUID: NATURAL APPROACH FOR REAL DATA INCORPORATION
Previous distillation methods add real data during training only via GANs with extra discriminator and adversarial loss. We propose a simpler, more natural way that requires no extra models or losses.
Based on intuition from inverse optimization ( §3.3), we see that the min-max inverse problem (15) is compatible with other losses. This allows us to redesign the UM loss (3) to incorporate real data into it. A key constraint is that the loss must still yield the same teacher upon minimization on the real data. Thus, we derive a novel Unified Matching loss with real data -a weighted sum of two UM-like losses on generated and real data parameterized by α, β ∈ (0, 1] which control the weights.
Definition 2. We define Universal Matching loss with real data (RealUM) L α,β R-UM (f, p θ 0 ) that is parametrized by α,β ∈(0,1] and takes trainable function f and generated data p θ 0 as arguments:
L α,β R-UM (f, p θ 0 ) = α • E t∼[0,T ] E x θ 0 ∼p θ 0 ,x θ t ∼p θ t (•|x θ 0 ) ∥f t (x θ t ) - β α f θ (x θ t |x θ 0 )∥ 2 generated data p θ 0 term + (1 -α) • E t∼[0,T ] E x * 0 ∼p * 0 ,x * t ∼p * t (•|x * 0 ) ∥f t (x * t ) - 1 -β 1 -α f * t (x * t |x * 0 )∥ 2 real data p * 0 term .(16)
For α = 1, we consider only β = 1, i.e., the pure generated data term.
RealUM loss (16) for all α, β and UM loss (3) yield the same teacher when input is real data p * 0 , since if we consider only the f -dependent terms in the losses, we have:
L α,β R-UM (f, p * 0 ) ∝ E t,x * 0 ,x * t [[α + (1 -α)] =1 •⟨f t (x * t ), f t (x * t )⟩+2[α • β α + (1 -α) • 1 -β 1 -α =1 ]⟨f t (x * t ), f * t (x * t |x * 0 )⟩] ∝ E t,x * 0 ,x * t [∥f t (x * t ) -f * t (x * t |x * 0 )∥ 2 ] =⇒ arg min f L α,β R-UM (f, p * 0 ) = arg min f L UM (f, p * 0 ) = f * .
Hence, the min-max inverse scheme ( 15) with RealUM loss and the old teacher f * still has a real data generator as a solution, but now real data is incorporated via the real data terms of
L α,β R-UM (f, p θ 0 ): min θ {L α,β R-UM (f * , p θ 0 ) -min f {L α,β R-UM (f, p θ 0 )} ≥0 } p θ 0 =p * 0 = L α,β R-UM (f * , p * 0 ) -min f {L α,β R-UM (f, p * 0 )} =L α,β R-UM (f * ,p * 0 ) = 0. Theorem 2 (Real data generator minimizes RealUID loss). Let teacher f * := arg min f L UM (f, p * 0 ) be the minimizer of UM loss (Def. 1) on real data p * 0 ∈ P(R D ). Then, real data generator G θ * s.t. p θ * 0 = p *
0 is a solution to the min-max optimization of Universal Inverse Distillation loss with real data (RealUID) L α,β R-UID (f, p θ 0 ) over fake function f and generator distribution p θ 0 (see Def. 2):
min θ max f L α,β R-UID (f, p θ 0 ) := L α,β R-UM (f * , p θ 0 ) -L α,β R-UM (f, p θ 0 ) .(17)
We provide analysis of RealUID in Appendix A.1, below we highlight the most important findings.
Role of coefficients α, β. Our RealUID uses real data only to minimize L α,β R-UM (f, p θ 0 ) loss over fake function f . Thus, the trained fake function memorizes both the real data and the generator's current state. In turn, the generator is influenced by the real data indirectly, only via this fake function. As shown in Lemma 2, RealUID implicitly minimizes the rescaled distance (18) between the teacher and generator functions. This distance is still minimal when p θ 0 = p * 0 , alternatively proving Theorem 2. Lemma 2 (Distance minimized by RealUID loss). Maximization of RealUID loss (16) over fake function f represents the weighted squared ℓ 2 -distance between the student function f θ := arg min f L UM (f, p θ 0 ) and the teacher
f * := arg min f L UM (f, p * 0 ) : max f L α,β R-UID (f, p θ 0 ) = E t∼[0,T ], x * t ∼p * t ∥ β α • [p * t (x * t )f * t (x * t ) -p θ t (x * t )f θ t (x * t )] + (p θ t (x * t ) -p * t (x * t )) • f * t (x * t )∥ 2 p * t (x * t )((1 -α)p * t (x * t ) + αp θ t (x * t ))/α 2 . (18
)
The proof of Lemma 2 is located in Appendix A.
1.2. With the help of real data, our RealUID loss now provides the generator with the feedback on the real data domain it needs to cover, i.e., the distance (18) does not vanish for points x t s.t. p θ (x t ) ≈ 0, p * (x t ) ≫ 0 (see Appendix A.1.3). Moreover, if teacher function is inaccurate, RealUID can now provably fix teacher's errors (see Appendix A.1.4).
Choice of coefficients α, β. Lemma 2 shows that, instead of values α and β, actually the values α and β /α determine the balance between real and generated data in the minimized distance ( 18). Furthermore, coefficient α only sets the general scaling of the distance, while β /α plays the most important role, as it determines the relation between f θ t and f * t inside the distance. The value β /α = 1 yields the distance identical to the data-free distance ( 14) up to scaling. Thus, even when α = β < 1 and real data is formally added, it may have no effect on the generator. Excessively low α and β diminish the effect of the generated data terms in the trained fake function, leading to vanishing gradients. The same issue occurs with β /α ≪ 1 in (18), while β /α ≫ 1 diminish the effect of real data in the right term of (18). See complete distance analysis in Appendix A.1.3. Moreover, if teacher function is inaccurate, only the choice β /α ̸ = 1 can fix teacher's errors (see Appendix A.1.4). Hence, good coefficients α, β ∈ (0, 1] can be chosen by first finding good β /α ̸ = 1, as it has the largest impact, and then adjusting α < 1. Both β /α and α should be close to 1.
Comparison with GAN-based methods. Unlike SiD and FGM with GANs (8), we do not use extra adversarial losses and discriminator to incorporate real data. We only modify UM loss, preserving its core structure and fake model architecture. While general adversarial loss is unrelated to the main distillation loss and has uninterpretable scaling hyperparameters, our RealUID loss and weighting coefficients α, β ∈ (0, 1] come naturally from the data-free UID loss. The original UID loss (13), equivalent to FGM (7) and SiD (5) with α SiD = 0.5, is obtained when α = β = 1.
Alternative loss form. Our RealUID is implicitly related to the linearization scheme used to obtain data-free UID ( §3.1). The loss (17) can be derived by splitting each term in the linearized UID loss (10) between real and generated data according to proportions α and β (see Appendix A.1.1). This form helps to prove RealUID's properties and extend it beyond the inversion scheme ( §5).
Extension for Bridge Matching and Stochastic Interpolants frameworks. In Appendix C.3, we demonstrate that our framework can be easily extended to data-to-data matching models by parameterizing the generated data coupling π θ (x 0 , x T ) instead of the data distribution p θ 0 .
this section cite: []

Section: EXPERIMENTS
All our PyTorch implementations and the latest checkpoints are publicly available in https://github.com/David-cripto/RealUID.
This section provides an ablation study and evaluation of our RealUID, assessing both its performance and computational efficiency. We begin in ( §4.1) by detailing the experimental setup based on flow matching models. In ( §4.2), we show that our incorporation of real data via coefficients α, β improves performance, speeds up convergence, and enables effective fine-tuning. In ( §4.3), we assess the benchmark performance and computational demands of RealUID relative to SOTA methods. Additional experimental details and results are provided in Appendix D.
this section cite: []

Section: EXPERIMENTAL SETUP

this section cite: []

Section: Datasets and Evaluation Protocol.
The experiments were conducted on the CIFAR-10 dataset with 32 × 32 resolution (Krizhevsky et al., 2009) and on the CelebA dataset with 64 × 64 resolution (Liu et al., 2015), see Appendix D.3. In line with the prior works (Karras et al., 2019;2022), we report test FID scores (Heusel et al., 2017), computed using 50k generated samples. Implementation Details. We implement our RealUID framework for flow matching models from Appendix B. In contrast to prior studies (Zhou et al., 2024b;a;Huang et al., 2024) which employ the computationally demanding EDM architecture (Karras et al., 2022) our work adopts a more lightweight alternative (Tong et al., 2024). We also train our own flow matching teacher models using CFM loss (2). Further implementation details and efficiency analysis are provided in Appendix D.1.
this section cite: ['b20', 'b25', 'b15', 'b22', 'b10', 'b44', 'b13', 'b17', 'b35']

Section: BENCHMARKING METHODS UNDER A UNIFIED EXPERIMENTAL CONFIGURATION
We evaluate RealUID under a unified experimental protocol (fixed architecture and implementation). We begin by conducting an ablation over α, β to assess the influence of real-data incorporation. We then compare RealUID to a GAN-based alternative, showing that RealUID achieves comparable or superior accuracy. Furthermore, we analyze convergence, indicating that RealUID variants with real data train substantially faster than baselines without real-data. Finally, we explore a fine-tuning stage initialized from strong RealUID checkpoints, showing further performance gains.
Ablation study of coefficients α, β. We restrict the search for optimal parameters α and β to values near 1, specifically α, β ∈ [0.85, 1.0] with increments of 0.02. Setting these parameters too low leads to noisy generated samples. Following the analysis in ( §3.4), we perform a grid search over the values α and β /α instead of the original α and β. The results are reported in Table 1. As a baseline, we highlight the UID model without data incorporation, i.e., our RealUID with α = 1.0, β = 1.0.
As shown in the table, the ratio β /α has the largest impact on the final metrics, while α only adjusts them. Using real data with β /α = 1 or with large values outside the range [0.98, 1.02] consistently degrades performance. In contrast, values β /α = 0.98 or β /α = 1.02 outperform the baseline for a majority of α. Note that these practical results match the theoretical description in ( §3.4). Comparison with GAN-based method. We integrate the GAN-based approach (8) proposed by (Zhou et al., 2024a) as an alternative method for incorporating real data, enabling a direct comparison with our RealUID formulation. We combine the GAN loss with the UID baseline. As shown in Table 1, the best-performing configurations are achieved with GAN losses (λ G θ adv = 0.3, λ D adv = 1). While this setup performs comparably to RealUID (α = 0.92, β = 0.94) in the unconditional setting, it remains clearly inferior to RealUID (α = 0.98, β = 0.96) in the conditional case.
Table 2: This table presents the results of ablation study of our RealUID framework, evaluated using the FID metric under both unconditional and conditional generation setups. The Teacher Flow model with 100 NFE is reported as a reference. The performance of the UID (FGM) baseline without real-data incorporation is indicated in italic. For emphasis, we underline the two counterparts that incorporate real data: the GAN-based and our RealUID methods. The best-performing configurations, obtained via an additional fine-tuning stage, are highlighted in bold. Qualitative results are presented in Appendix D.5.1.
this section cite: []

Section: Model FID (↓)
Teacher Flow (NFE=100) 3.57 UID (FGM) 2.58 Convergence Speed. Our RealUID(α, β) with parameters which are highlighted in Table 1 achieves faster convergence than the UID baseline. For clarity, we present qualitative comparisons in Figure 2. The best RealUID configurations reach the saturated performance level of the baseline after ∼100k iterations, whereas the baseline requires ∼300k iterations to achieve comparable metrics.
UID + GAN (λ Gθ adv = 0.3, λ D adv = 1 | λ Gθ FT = 25, λ D FT = 75) 2.10 RealUID (α = 0.92, β = 0.94 | αFT = 0.92, βFT = 0.86) (Ours) 1.98 Model FID (↓) Teacher Flow (NFE=100) 5.56 UID (FGM) 2.21 UID + GAN (λ Gθ adv = 0.3, λ D adv = 1 | λ Gθ FT = 25, λ D FT =
Fine-tuning stage. We observe that RealUID and GAN frameworks offer substantial flexibility for fine-tuning. In this procedure, the generator is initialized from the best-performing checkpoint obtained during training from scratch of the corresponding framework, while the fake model is initialized from the teacher. Fine-tuning then proceeds with new values α FT and β FT for our RealUID and λ G θ FT and λ D FT for GANs. We present the best-found fine-tuning configurations for both methods in Table 2. Ablation study analyzing the effect of loss coefficients is provided in Appendix D.2. Scaling to larger datasets. In Appendix D.3, we provide the results of the same ablation study on the CelebA dataset with 64 × 64 resolution. Notably, our RealUID performance and the optimal values β /α remain the same across datasets.
this section cite: []

Section: BASELINE COMPARISON
As shown in Tables 3 and 4, our RealUID after fine-tuning consistently outperforms all prior flowbased models on CIFAR-10, significantly surpassing the strongest flow distillation baseline, FGM. Despite its compact and lightweight architecture ( §4.1) with nearly 2× faster inference, it achieves performance comparable to leading diffusion distillation methods SiD (α SiD =1.0\1.2), while falling short of adversarially enhanced models such as SiD 2 A. We hypothesize that this performance gap is attributed to architectural and teacher capacity differences rather than the lack of adversarial loss.
Our latest checkpoints and metrics (Appendix D.4) are available in our repository. Relation to DMD. Instead of minimizing the squared ℓ 2 -distance between the score functions, Distribution Matching Distillation (Luo et al., 2023;Wang et al., 2023;Yin et al., 2024b;a, DMD) approach minimizes the KL divergence between the real and generated data. Its gradients are computed using the generator and teacher score functions, leading to the similar alternating updates.
We would like to highlight that DMD does not fit UID framework. Nevertheless, we investigated an opportunity to incorporate real data into DMD without GANs in Appendix A.5.
this section cite: ['b27', 'b37', 'b44']

Section: BROADER IMPACT
This paper presents work whose goal is to advance the field of Artificial Intelligence, Machine Learning and Generative Modeling. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. CONTENTS 1 Introduction 2 Backgrounds on training and distilling matching models 2.1 Diffusion and flow models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2.2 Universal loss for matching models . . . . . . . . . . . . . . . . . . . . . . . . . . 2.3 Distillation of matching-based models . . . . . . . . . . . . . . . . . . . . . . . . 2.4 GANs for real data incorporation . . . . . . . . . . . . . . . . . . . . . . . . . . . 3 Universal distillation of matching models with real data 3.1 Universal Inverse Distillation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.2 Relation to prior distillation works . . . . . . . . . . . . . . . . . . . . . . . . . . 3.3 Connection with Inverse Optimization . . . . . . . . . . . . . . . . . . . . . . . . 3.4 RealUID: natural approach for real data incorporation . . . . . . . . . . . . . . . . 4 Experiments 4.1 Experimental setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4.2 Benchmarking methods under a unified experimental configuration . . . . . . . . . 4.3 Baseline comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 Discussion and extensions 6 Broader impact 7 LLM Usage A Theoretical proofs and extensions A.1 RealUID theoretical properties . . . . . . . . . . . . . . . . . . . . . . . . . . . . A.1.1 Alternative RealUID split form . . . . . . . . . . . . . . . . . . . . . . . . A.1.2 Proof of RealUID Distance Lemma 2 . . . . . . . . . . . . . . . . . . . . A.1.3 Explanation of the choice of coefficients α and β . . . . . . . . . . . . . . A.1.4 Correction of teacher's errors . . . . . . . . . . . . . . . . . . . . . . . . . A.2 General RealUID loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . A.3 General SiD with real data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . A.4 Normalized UID and RealUID losses for minimizing ℓ 2 -distance . . . . . . . . . . A.5 DMD approach with real data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B RealUID Algorithm for flow matching models C Unified Inverse Distillation with real data for Bridge Matching and Stochastic Interpolants C.1 Bridge Matching . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.2 Stochastic Interpolants . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28 C.3 Objective for general data coupling . . . . . . . . . . . . . . . . . . . . . . . . . . 28 D Experimental details and additional results 29 D.1 CIFAR-10 distillation from scratch . . . . . . . . . . . . . . . . . . . . . . . . . . 29 D.2 CIFAR-10 distillation fine-tuning . . . . . . . . . . . . . . . . . . . . . . . . . . . 30 D.3 CelebA distillation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30 D.4 Further hyperparameters gridsearch . . . . . . . . . . . . . . . . . . . . . . . . . 32 D.5 Example of samples for various methods . . . . . . . . . . . . . . . . . . . . . . . 32 D.5.1 CIFAR-10 generated images . . . . . . . . . . . . . . . . . . . . . . . . . 33 D.5.2 CelebA generated images . . . . . . . . . . . . . . . . . . . . . . . . . . 39 A THEORETICAL PROOFS AND EXTENSIONS In this appendix, we discuss our RealUID framework (Appendix A.1) in theoretical details and provide three extensions of it: General RealUID framework with 3 degrees of freedom (Appendix A.2), General SiD framework with real data (Appendix A.3) and Normalized RealUID framework for minimizing ℓ 2 -distance between teacher and student functions instead of the squared one (Appendix A.4). All proofs are based on the linearization technique and splitting terms in linearized decomposition between real and generated data. We also propose an approach to incorporate real data into DMD framework, which is unsuitable for our RealUID (Appendix A.5). A.1 REALUID THEORETICAL PROPERTIES
In this section, we discuss our RealUID loss in detail. We begin by presenting its alternative form and how it connects linearization technique and real data incorporation (Appendix A.1.1). We then demonstrate that the loss minimizes a squared ℓ 2 -distance between the rescaled teacher and student functions (Appendix A.1.2). Finally, we provide the motivation of the best choice of coefficients α ̸ = β from the perspectives of the better distance (Appendix A.1.3) and the correction of the teacher's errors (Appendix A.1.4).
this section cite: []

Section: A.1.1 ALTERNATIVE REALUID SPLIT FORM
Let us recall the linearization trick that we apply to make the minimized squared norm between the student function f θ and the teacher f * tractable. For each time t and generated point x θ t , we restate this squared norm as the identity ∥a∥ 2 = max b∈R D {-∥b∥ 2 + 2⟨b, a⟩}, ∀a ∈ R and use an auxiliary function δ t (x t ) to parametrize a vector b. In the end, we substitute the student function f θ t (x θ t ) with its conditional and differentiable estimate
f θ t (x θ t |x θ 0 ): E t∼[0,T ], x θ t ∼p θ t [∥f * t (x θ t ) -f θ t (x θ t )∥ 2 ] = max δt(x θ t )
E t∼[0,T ],
x θ t ∼p θ t -∥δ t (x θ t )∥ 2 + 2⟨δ t (x θ t ), f * t (x θ t ) -f θ t (x θ t )⟩ = E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) [-∥δ t (x θ t )∥ 2 + 2⟨δ t (x θ t ), f * t (x θ t )⟩ -2⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩].(19)
In addition, we use parameterization δ = f * -f with a fake model f to obtain our UID loss which matches the previous distillation losses.
Originally, we derived our RealUID loss (17) from the idea of splitting each term in the linearized form of data-free UID (19) between the generated and real data in proportions defined by coefficients α and (1 -α), α and (1 -α) and β and (1 -β). We present the split form of RealUID loss in Lemma 3, and this form completely matches the inverse optimization form defined in Theorem 2. Lemma 3 (RealUID split form). The RealUID loss (17) can be restated as
L α,β R-UID (f, p θ 0 ) = E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) [-α∥δ t (x θ t )∥ 2 + 2α⟨δ t (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩] + E t∼[0,T ],x * 0 ∼p * 0 , x * t ∼p * t (•|x * t ) [-(1 -α)∥δ t (x * t )∥ 2 + 2(1 -α)⟨δ t (x * t ), f * t (x * t )⟩ -2(1 -β)⟨δ t (x * t ), f * t (x * t |x * 0 )⟩],
with the parameterization δ = f * -f .
The idea of splitting coefficients between two data types helps to prove properties of RealUID, and extend our real data incorporation technique to general form (Appendix A.2), SiD framework with α SiD ̸ = 1 2 (Appendix A.3) and new distances (Appendix A.4).
Proof. Putting explicit values for RealUM loss (16) in RealUID loss (17), we get:
L α,β R-UID (f, p θ 0 ) = L α,β R-UM (f * , p θ 0 ) -L α,β R-UM (f, p θ 0 ) = α • E t,x θ 0 ,x θ t ∥f * t (x θ t ) - β α f θ (x θ t |x θ 0 )∥ 2 + (1 -α) • E t,x * 0 ,x * t ∥f * t (x * t ) - 1 -β 1 -α f * t (x * t |x * 0 )∥ 2 -α • E t,x θ 0 ,x θ t ∥f t (x θ t ) - β α f θ (x θ t |x θ 0 )∥ 2 -(1 -α) • E t,x * 0 ,x * t ∥f t (x * t ) - 1 -β 1 -α f * t (x * t |x * 0 )∥ 2 .
Then, we group the factors with the same data type and multipliers:
L α,β R-UID (f, p θ 0 ) = L α,β R-UM (f * , p θ 0 ) -L α,β R-UM (f, p θ 0 ) = E t,x θ 0 ,x θ t α • ∥f * t (x θ t ) - β α f θ (x θ t |x θ 0 )∥ 2 -α • ∥f t (x θ t ) - β α f θ (x θ t |x θ 0 )∥ 2 + E t,x * 0 ,x * t (1 -α) • ∥f * t (x * t ) - 1 -β 1 -α f * t (x * t |x * 0 )∥ 2 -(1 -α) • ∥f t (x * t ) - 1 -β 1 -α f * t (x * t |x * 0 )∥ 2 = E t,x θ 0 ,x θ t α • ∥f * t (x θ t )∥ 2 -2β • ⟨f * t (x θ t ), f θ (x θ t |x θ 0 )⟩ -α • ∥f t (x θ t )∥ 2 + 2β • ⟨f t (x θ t ), f θ (x θ t |x θ 0 )⟩ + E t,x * 0 ,x * t (1 -α) • ∥f * t (x * t )∥ 2 -2(1 -β) • ⟨f * t (x * t ) -f t (x * t ), f * (x * t |x * 0 )⟩ -(1 -α) • ∥f t (x * t )∥ 2 ⟩ = E t,x θ 0 ,x θ t α • (∥f * t (x θ t )∥ 2 -∥f t (x θ t )∥ 2 ) -2β • ⟨f * t (x θ t ) -f t (x θ t ), f θ (x θ t |x θ 0 )⟩ + E t,x * 0 ,x * t (1 -α) • (∥f * t (x * t )∥ 2 -∥f t (x * t )∥ 2 ) -2(1 -β) • ⟨f * t (x * t ) -f t (x * t ), f * (x * t |x * 0 )⟩ = E t,x θ 0 ,x θ t α • (-∥f * t (x θ t ) -f t (x θ t )∥ 2 + 2⟨f * t (x θ t ) -f t (x θ t ), f * t (x θ t )⟩) -E t,x θ 0 ,x θ t 2β • ⟨f * t (x θ t ) -f t (x θ t ), f θ (x θ t |x θ 0 )⟩ + E t,x * 0 ,x * t (1 -α) • (-∥f * t (x * t ) -f t (x * t )∥ 2 + 2⟨f * t (x * t ) -f t (x * t ), f * t (x * t )⟩) -E t,x * 0 ,x * t [2(1 -β) • ⟨f * t (x * t ) -f t (x * t ), f * (x * t |x * 0 )⟩] .
Finally, denoting parameterization δ = f * -f , we obtain the required form:
L α,β R-UID (f, p θ 0 ) = E t,x θ 0 ,x θ t [-α∥δ t (x θ t )∥ 2 + 2α⟨δ t (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩] + E t,x * 0 ,x * t [-(1 -α)∥δ t (x * t )∥ 2 + 2(1 -α)⟨δ t (x * t ), f * t (x * t )⟩ -2(1 -β)⟨δ t (x * t ), f * t (x * t |x * 0 )⟩].
this section cite: []

Section: A.1.2 PROOF OF REALUID DISTANCE LEMMA 2
Proof of Lemma 2. In this proof, we use the split form of our ReaLUID loss from Lemma 3. First, we take math expectation over data points x * 0 . Since the expectation can be taken in a reverse order, i.e.,
E x * 0 ∼p * 0 ,x * t ∼p * t (•|x * 0 ) = E x * t ∼p * t ,x * 0 ∼p * 0 (•|x * t ) , we see that E x * 0 ∼p * 0 ,x * t ∼p * t (•|x * 0 ) [⟨δ t (x * t ), f * t (x * t |x * 0 )⟩] = E x * t ∼p * t [⟨δ t (x * t ), E x * 0 ∼p * 0 (•|x * t ) [f * t (x * t |x * 0 )]⟩] = E x * t ∼p * t [⟨δ t (x * t ), f * t (x * t )⟩].(20)
For the generated data term
E x θ 0 ∼p θ 0 ,x θ t ∼p θ t (•|x θ 0 ) [⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩] = E x θ t ∼p θ t [⟨δ t (x θ t ), f θ t (x θ t )⟩]
, the reasoning is similar. Thus, we can write down RealUID loss in an explicit form with δ t = f * t -f t :
L α,β R-UID (δ, p θ 0 ) = E t∼[0,T ] E x θ t ∼p θ t [-α∥δ t (x θ t )∥ 2 + 2α⟨δ t (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t )⟩] +E t∼[0,T ] E x * t ∼p * t [-(1 -α)∥δ t (x * t )∥ 2 + 2(1 -α)⟨δ t (x * t ), f * t (x * t )⟩ -2(1 -β)⟨δ t (x * t ), f * t (x * t )⟩].(
21) Then, we rescale the generated data terms in RealUID loss (21) using the equality p θ t (x t ) = p θ t (xt) p * t (xt) p * t (x t ) for x t ∈ R D (we assume p * t (x t ) > 0, ∀x t , t) leaving only math expectation w.r.t. the real data, i.e, L α,β R-UID (δ, p θ 0 ) = E t∼[0,T ] x * t ∼p * t -[(1 -α) + α p θ t (x * t ) p * t (x * t ) ]∥δ t (x * t )∥ 2 -E t∼[0,T ] x * t ∼p * t 2β p θ t (x * t ) p * t (x * t ) ⟨δ t (x * t ), f θ t (x * t )⟩ + 2[(β -α) + α p θ t (x * t ) p * t (x * t ) ]⟨δ t (x * t ), f * t (x * t )⟩ .
Finally, we maximize the loss w.r.t. δ t (x * t ) for each x * t and t as a quadratic function. The maximum is achieved when
δ t (x * t ) = [(β -α) + α p θ t (x * t ) p * t (x * t ) ]f * t (x * t ) -β p θ t (x * t ) p * t (x * t ) f θ t (x * t ) [(1 -α) + α p θ t (x * t ) p * t (x * t ) ] or in terms of the fake model f = f * -δ arg max f L α,β R-UID (f, p θ 0 ) (t, x t ) = f * t (x t ) • (1 -β) + f θ t (x t ) • β p θ t (xt) p * t (xt) (1 -α) + α p θ t (xt) p * t (xt) .(22)
The maximum itself equals to
max f L α,β R-UID (f, p θ 0 ) = E t∼[0,T ] E x * t ∼p * t   ∥f * t (x * t ) • ((β -α) + α p θ t (x * t ) p * t (x * t ) ) -f θ t (x * t ) • β p θ t (x * t ) p * t (x * t ) ∥ 2 (1 -α) + α p θ t (x * t ) p * t (x * t )   .
It is easy to see that when p θ 0 = p * 0 and f θ = f * this distance achieves its minimal value 0. Moreover, optimal fake model in this case matches the teacher f * , i.e.,
arg max f L α,β R-UID (f, p * 0 ) (t, x t ) = f * t (x t ) • (1 -β) + f * t (x t ) • β p * t (xt) p * t (xt) (1 -α) + α p * t (xt) p * t (xt) = f * t (x t ).
this section cite: []

Section: A.1.3 EXPLANATION OF THE CHOICE OF COEFFICIENTS α AND β
Here we show that the best way to incorporate real data during generator training is to set β /α ̸ = 1.
Following Lemma 2, we know exactly what distance our RealUID loss implicitly minimizes. Below we examine it for various α, β ∈ (0, 1]:
max f L α,β R-UID (f, p θ 0 ) = xt l t (x t , β, α)dx t , l t (x t , β, α) := α 2 ∥(p * t (x t )( β α -1) + p θ t (x t )) • f * t (x t ) -β α • p θ t (x t ) • f θ t (x t )∥ 2 (1 -α)p * t (x t ) + αp θ t (x t )
, where l t (x t , β, α) denotes the distance for the particular point x t .
The total distance mostly sums up from the two groups of points: incorrectly generated points from the generator's main domain, i.e., p θ t (x t ) ≫ 0, p * (x t ) ≈ 0, and real data points which are not covered by the generator, i.e., p θ t (x t ) ≈ 0, p * (x t ) ≫ 0. For the points out of both domains p θ t (x t ) ≈ 0, p * t (x t ) ≈ 0, the distance tends to 0, as well as for matching points p θ t (x t ) ≈ p * t (x t ).
Choice of coefficients α, β. Next, we consider various coefficients α, β ∈ (0, 1] and how they affect two main groups of points.
• All configurations affect the incorrectly generated points x t : p * t (x t ) ≈ 0, p θ (x t ) ≫ 0:
l t (x t , β, α) ≈ ∥αp θ t (x t ) • f * t (x t ) -βp θ t (x t ) • f θ t (x t )∥ 2 αp θ t (x t ) ≈ β 2 ∥f θ t (x t )∥ 2 α p θ t (x t ) ≫ 0.(23)
Note that increasing β /α > 1 will diminish the weight of the distance in comparison with α = β = 1, while decreasing otherwise will lift the weight up. • Configuration β < α = 1 is unstable for uncovered real data points x t : p θ t (x t ) ≈ 0, p * (x t ) ≫ 0:
l t (x t , β, α) ≈ ∥p * t (x t )(β -1) • f * t (x t ) -βp θ t (x t ) • f θ t (x t )∥ 2 p θ t (x t )
≈ ∞.
• Configuration β = α = 1 (UID loss) does not affect uncovered real data points x t :
p θ t (x t ) ≈ 0, p * (x t ) ≫ 0: l t (x t , β, α) ≈ ∥p θ t (x t ) • f * t (x t ) -p θ t (x t ) • f θ t (x t )∥ 2 p θ t (x t ) = ∥f * t (x t ) -f θ t (x t )∥ 2 p θ t (x t ) ≈ 0.
• Configuration β = α < 1 does not affect uncovered real data points x t : p θ t (x t ) ≈ 0, p * (x t ) ≫ 0:
l t (x t , β, α) ≈ ∥αp θ t (x t )f * t (x t ) -βp θ t (x t )f θ t (x t )∥ 2 (1 -α)p * t (x t ) = ∥αf * t (x t ) -βf θ t (x t )∥ 2 (1 -α) (p θ t (x t )) 2 p * t (x t ) ≈ 0.
Notably, in this configuration, the distance drops even faster than when α = β = 1, what makes it even less preferable.
• Only configuration β /α ̸ = 1 affects the uncovered real data points x t :
p θ t (x t ) ≈ 0, p * (x t ) ≫ 0: l t (x t , β, α) ≈ ∥p * t (x t )(β -α) • f * t (x t ) -βp θ t (x t ) • f θ t (x t )∥ 2 (1 -α)p * t (x t ) ≫ 0.
Visual illustration. We analytically calculate the loss surface l t (x t , α, β) between the FM models transforming one-dimensional real data Gaussian N (µ * , 1) and generated Gaussian N (µ θ , 1) to noise N (0, 1) on the time interval [0, 1]. In this case, the generated and real data interpolations are p θ t (x t ) = N (x t |µ θ (1 -t), t 2 + (1 -t) 2 ) and p * t (x t ) = N (x t |µ * (1 -t), t 2 + (1 -t) 2 ). The unconditional vector field u = f between N (µ, 1) and N (0, 1) can be calculated as
u t (x t ) = E x0∼p0(•|xt) x t -x 0 t = x0 x t -x 0 t • N x t -x 0 (1 -t) t |0, 1 • N (x 0 |µ, 1)dx 0 = a(2t 2 -2t) -bt 2 √ 2π(1 -2t + 2t 2 ) 3 2 exp - (x t -µ(1 -t)) 2 2(1 -2t + 2t 2 ) 2 . (24
)
In Figure 3, we depict the loss surfaces for the fixed time t = 1/3, real data µ * = 2, generated data µ θ = -2 and various pairs of (α, β). We can see that configurations β /α = 1 do not detect the real data sample, even when α = β < 1 and real data is formally used. while β /α ̸ = 1 actually spots both domains, increasing the weight of generator domain when β /α > 1 and decreasing it otherwise.
this section cite: []

Section: A.1.4 CORRECTION OF TEACHER'S ERRORS
In this chapter, we assume that instead of accurate teacher f * = arg min f L UM (f, p * 0 ) we have access only to the arbitrary corrupted teacher f * . We will show that adding real data via our approach with α ̸ = β provably mitigates the teacher's errors in the final generator.
this section cite: []

Section: Minimized distance.
With the corrupted teacher f * and δ = f * -f , our corrupted RealUID loss takes the split form from Lemma 3
L α,β R-UID ( δ, p θ 0 ) = E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) [-α∥ δt (x θ t )∥ 2 + 2α⟨ δt (x θ t ), f * t (x θ t )⟩ -2β⟨ δt (x θ t ), f θ t (x θ t |x θ 0 )⟩] +E t∼[0,T ],x * 0 ∼p * 0 , x * t ∼p * t (•|x * t ) [-(1 -α)∥ δt (x * t )∥ 2 + 2(1 -α)⟨ δt (x * t ), f * t (x * t )⟩ -2(1 -β)⟨ δt (x * t ), f * t (x * t |x * 0 )⟩].
Note that sampled terms f * t (x * t |x * 0 ) and f θ t (x θ t |x θ 0 ) are not affected by the corruption and give the accurate functions f * t (x * t ) = E x * 0 ∼p * 0 (•|x * t ) [f * t (x * t |x * 0 )] and f θ t (x θ t ) = E x θ 0 ∼p θ 0 (•|x θ t ) [f θ t (x θ t |x θ 0 )]:
L α,β R-UID ( δ, p θ 0 ) = E t∼[0,T ] E x θ t ∼p θ t [-α∥ δt (x θ t )∥ 2 + 2α⟨ δt (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t )⟩] +E t∼[0,T ] E x * t ∼p * t [-(1 -α)∥ δt (x * t )∥ 2 + 2(1 -α)⟨ δt (x * t ), f * t (x * t )⟩ -2(1 -β)⟨ δt (x * t ), f * t (x * t )⟩].
Then, we rescale the generated data terms using the equality p θ t (x t ) =
p θ t (xt) p * t (xt) p * t (x t ) for x t ∈ R D (we assume p * t (x t ) > 0, ∀x t , t) leaving only math expectation w.r.t. the real data, i.e,
L α,β R-UID ( δ, p θ 0 ) = E t∼[0,T ],x * t ∼p * t -[(1 -α) + α p θ t (x * t ) p * t (x * t ) ](∥ δt (x * t )∥ 2 +2⟨ δt (x * t ), f * t (x * t )⟩) -E t∼[0,T ], x * t ∼p * t 2⟨ δt (x * t ), (1 -β)f * t (x * t ) +β p θ t (x * t ) p * t (x * t ) f θ t (x * t )⟩ .
Finally, we maximize the loss w.r.t. δt (x * t ) for each x * t and t as a quadratic function
max δ L α,β R-UID ( δ, p θ 0 ) = E t∼[0,T ] E x * t ∼p * t   ∥ f * t (x * t ) • ((1 -α) + α p θ t (x * t ) p * t (x * t ) ) -(1 -β)f * t (x * t ) -β p θ t (x * t ) p * t (x * t ) f θ t (x * t )∥ 2 (1 -α) + α p θ t (x * t ) p * t (x * t )   . (25
)
Hence, max-min optimization of the corrupted RealUID loss implicitly minimizes expected distance (25). However, due to arbitrary function f , we now cannot guarantee that minimum is achived when the relation inside the norm equals 0. Previously, we could use the solution p θ = p * which obviously achieved a minimum of 0. Now, due to the implicit and complex relationship between f θ and p θ , we can neither find an explicit form for the optimal p θ nor guarantee the minimum of 0.
Choice of coefficients α, β. Here we give an intuition on why coefficients β /α ̸ = 1 can fix the teacher's errors, while β /α = 1 cannot. For simplicity, we assume that the minimized distance (25) actually attains minimum of 0 when
((1 -α)p * t (x t ) + αp θ t (x t )) • f * t (x t ) -(1 -β)p * t (x t ) • f * t (x t ) -βp θ t (x t ) • f θ t (x * t ) = 0. (26
)
• In case of α = β = 1, we have f * t = f θ t , i.e., the generator learns the corrupted function.
• In case of α = β < 1, we have
f * t (x t ) = (1 -α)p * t (x t ) (1 -α)p * t (x t ) + αp θ t (x t ) • f * t (x t ) + αp θ t (x t ) (1 -α)p * t (x t ) + αp θ t (x t ) • f θ t (x * t ).
In this convex combination, the corrupted function f * is always between the true teacher function f * and the optimal generator function f θ , i.e., the generator learns even worse function.
• In case of β /α ̸ = 1, there exist intervals of α, β which can give better generator function than the corrupted teacher. For example, coefficients α ̸ = β close to 1 allow to neglect the terms
(1 -α)p * t (x t ) • f * t (x t ) and (1 -β)p * t (x t ) • f * t (x t ) in (26) to get f θ t (x t ) ≈ α β f * t (x t )
. Hence, we can steer f θ towards the true teacher picking β /α < 1 or β /α > 1 depending on the corrupted and clean teacher's values. However, we cannot find all these intervals analytically due to complex distributions and functions.
Note that we derive the same recommendation β /α ̸ = 1 from the perspective of correcting the teacher's errors and from the perspective of the minimized distance surface from Appendix A.1.3.
this section cite: []

Section: Visual illustration.
For visual demonstration, we consider the FM models transforming onedimensional real data Gaussian N (µ * , 1) and generated Gaussian N (µ θ , 1) to noise N (0, 1) on the time interval [0, 1]. In this case, the generated and real data interpolations are
p θ t (x t ) = N (x t |µ θ (1 - Definition 3. We introduce General RealUID loss L α,β,γ R-UID (δ, p θ 0 ) on generated data p θ 0 ∈ P(R D ) with coefficients α, β, γ: L α,β,γ R-UID (δ, p θ 0 ) := E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) [-γ∥δ t (x θ t )∥ 2 + 2α⟨δ t (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩] + E t∼[0,T ],x * 0 ∼p * 0 , x * t ∼p * t (•|x * 0 ) [-(1 -γ)∥δ t (x * t )∥ 2 + 2(1 -α)⟨δ t (x * t ), f * t (x * t )⟩ -2(1 -β)⟨δ t (x * t ), f * t (x * t |x * 0 )⟩].
Optionally, one can change default parameterization δ = f * -f (e.g., with δ = β(f * -f )), and substitute sampled real data term f * t (x * t |x * 0 ) with the unconditional teacher f * t (x * t ) and vice versa.
Theoretical properties. In case of δ = f * -f and γ ̸ = α, the General RealUID loss cannot be expressed as inverse min-max problem (15) for simple losses, since some scalar products do not eliminate each other. Nevertheless, min-max optimization of L α,β,γ R-UID still minimizes the squared ℓ 2 -distance between the weighted teacher and generator functions, attaining minimum when p θ 0 = p * 0 . Lemma 4 (Distance minimized by General RealUID loss). Maximization of General RealUID loss L α,β,γ R-UID over δ represents the weighted squared ℓ 2 -distance between the student function
f θ := arg min f L UM (f, p θ 0 ) and the teacher f * := arg min f L UM (f, p * 0 ): max δ L α,β,γ R-UID (δ, p θ 0 ) = E t∼[0,T ], x * t ∼p * t ∥ β α [p * t (x * t )f * t (x * t ) -p θ t (x * t )f θ t (x * t )] + (p θ t (x * t ) -p * t (x * t ))f * t (x * t )∥ 2 p * t (x * t ) • max{0, (1 -γ)p * t (x * t ) + γp θ t (x * t )}/α 2 . (28
)
The distances being minimized for RealUID (Lemma 2) and General RealUID (Lemma 4) are almost identical except the scale factor. Thus, we keep the same recommendations for choosing coefficients α, β as we discuss in Section 3.4. The factor β /α still has the largest impact within the distance, while α and γ set the scaling. Values β /α and γ should be chosen close to 1, but not exactly 1.
Proof. First, we take math expectation over data points x * 0 . Since the expectation can be taken in a reverse order, i.e.,
E x * 0 ∼p * 0 ,x * t ∼p * t (•|x * 0 ) = E x * t ∼p * t ,x * 0 ∼p * 0 (•|x * t ) , we see that E x * 0 ∼p * 0 ,x * t ∼p * t (•|x * 0 ) [⟨δ t (x * t ), f * t (x * t |x * 0 )⟩] = E x * t ∼p * t ⟨δ t (x * t ), E x * 0 ∼p * 0 (•|x * t ) [f * t (x * t |x * 0 )]⟩ = E x * t ∼p * t [⟨δ t (x * t ), f * t (x * t )⟩].(29)
For the term
E x θ 0 ∼p θ 0 ,x θ t ∼p θ t (•|x θ 0 ) [⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩] = E x θ t ∼p θ t [⟨δ t (x θ t ), f θ t (x θ t )⟩]
, the reasoning is similar. Thus, we write down General RealUID loss (Def. 3) in an explicit form with
δ t = f * t -f t L α,β R-UID (δ, p θ 0 ) = E t∼[0,T ] E x θ t ∼p θ t [-γ∥δ t (x θ t )∥ 2 + 2α⟨δ t (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t )⟩] +E t∼[0,T ] E x * t ∼p * t [-(1 -γ)∥δ t (x * t )∥ 2 + 2(1 -α)⟨δ t (x * t ), f * t (x * t )⟩ -2(1 -β)⟨δ t (x * t ), f * t (x * t )⟩]
. Then, we rescale the generated data terms in the General RealUID loss using the equality
p θ t (x t ) = p θ t (xt) p * t (xt) p * t (x t ) for x t ∈ R D (we assume p * t (x t ) > 0, ∀x t , t) leaving only math expectation w.r.t. the real data, i.e, L α,β,γ R-UID (δ, p θ 0 ) = E t∼[0,T ], x * t ∼p * t -[(1 -γ) + γ p θ t (x * t ) p * t (x * t ) ]∥δ t (x * t )∥ 2 + E t∼[0,T ], x * t ∼p * t 2[(β -α) + α p θ t (x * t ) p * t (x * t ) ]⟨δ t (x * t ), f * t (x * t )⟩ -2β p θ t (x * t ) p * t (x * t ) ⟨δ t (x * t ), f θ t (x * t )⟩ .
Next we maximize the loss w.r.t. δ t (x * t ) for each x * t and t as a quadratic function.
If (1 -γ) • p * t (x * t ) + γ • p θ t (x * t )
≤ 0, then the maximum tends to +∞. Otherwise, the maximum is achieved when
δ t (x * t ) = [(β -α) + α p θ t (x * t ) p * t (x * t ) ]f * t (x * t ) -β p θ t (x * t ) p * t (x * t ) f θ t (x * t ) [(1 -γ) + γ p θ t (x * t ) p * t (x * t ) ] .(30)
The maximum itself equals to
max δ L α,β,γ R-UID (δ, p θ 0 ) = E t∼[0,T ] E x * t ∼p * t   ∥f * t (x * t ) • ((β -α) + α p θ t (x * t ) p * t (x * t ) ) -f θ t (x * t ) • β p θ t (x * t ) p * t (x * t ) ∥ 2 (1 -γ) + γ p θ t (x * t ) p * t (x * t )   .
Alternative parameterization. In the proximity of the solution, when generated data approaches real one, i.e., p θ t ≈ p * t , the optimal δ t (30) approaches
δ t (x * t ) ≈ [(β -α) + α • 1]f * t (x * t ) -β • 1 • f θ t (x * t ) [(1 -γ) + γ • 1] ≈ β(f * t (x * t ) -f θ t (x * t )).
Thus, the parameterization δ t = β(f * t -f t ) may naturally help reach the solution without making the fake model learn extra information about the teacher near the optimum.
In experiments in Tables 1 and 7, this parameterization with the corresponding coefficients γ = α and β yields slightly better metrics from +0.02 to +0.04.
Extra ranges for coefficients α, β, γ New perspective on our RealUID loss allows us to expand the range of feasible configurations for the parameters α, β, and γ. Specifically, it is now possible to set α = 1 for any β, whereas in the original loss (16) this configuration is unavailable due to division by zero in the real data term. Additionally, one can now use values α, β, γ > 1.
However, we observe that in the experiments reported in Tables 1 and 7, these extra configurations are highly unstable and lead to degraded results. This happens due to out-of-domain generated samples and negative quadratic summands leading to infinite losses and metric (28). Hence, we stick to the original ranges α, β, γ ∈ (0, 1].
this section cite: []

Section: A.3 GENERAL SID WITH REAL DATA
Our real data incorporation. We recall that data-free UID loss (Theorem 1) can be restated via linearization technique with δ = f -f * as:
L UID (δ, p θ 0 ) = E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) -∥δ t (x θ t )∥ 2 + 2⟨δ t (x θ t ), f * t (x θ t )⟩ -2⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩ . (31
)
Following alternative definition of our RealUID loss from Lemma 3, one can incorporate real data into data-free loss by splitting each term in the linearized form between generated and real data as:
L α,β R-UID (δ, p θ 0 ) = E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) [-α∥δ t (x θ t )∥ 2 + 2α⟨δ t (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩] + E t∼[0,T ],x * 0 ∼p * 0 , x * t ∼p * t (•|x * 0 ) [-(1 -α)∥δ t (x * t )∥ 2 + 2(1 -α)⟨δ t (x * t ), f * t (x * t )⟩ -2(1 -β)⟨δ t (x * t ), f * t (x * t |x * 0 )⟩].(32)
General data-free SiD. The authors of the SiD framework (Zhou et al., 2024a;b) for diffusion models empirically notice that scaling the first coefficient -∥δ t (x θ t )∥ 2 by the factor 2α SiD in the UID loss (31) for generator updates yields better performance. Hence, we generalize the SiD loss to other matching models. Namely, the General SiD loss for the generator is the following loss with δ = f -f * and parameter α SiD ∈ [0.5, 1.2]:
L SiD (θ) := E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) -2α SiD ∥δ t (x θ t )∥ 2 + 2⟨δ t (x θ t ), f * t (x θ t )⟩ -2⟨δ t (x θ t ), f θ (x θ t |x θ 0 )⟩ ,(33)
while the UM loss (Def. 1) for the fake model remains intact. The same positive effect is observed in experiments with flow matching models in FGM (Huang et al., 2024), where the authors do not calculate the gradient through some loss terms and obtain the General SiD loss (33) with α SiD = 1, achieving better performance.
General SiD with real data. Following the structure of the General SiD loss (33), we propose to scale the first coefficient in our RealUID loss (32) during generator updates. The whole General SiD pipeline with real data (RealSiD), defined by coefficients α, β ∈ (0, 1], α SiD ∈ [0.5, 1.2] and teacher f * , is two alternating steps:
1. Make one or several fake model f update steps, minimizing UM loss with real data L α,β R-UM (f, p θ 0 ):
L α,β R-UM (f, p θ 0 ) := α • E t∼[0,T ] E x θ 0 ∼p θ 0 ,x θ t ∼p θ t (•|x θ 0 ) ∥f t (x θ t ) - β α f θ t (x θ t |x θ 0 )∥ 2 generated data p θ 0 term + (1 -α) • E t∼[0,T ] E x * 0 ∼p * 0 ,x * t ∼p * t (•|x * 0 ) ∥f t (x * t ) - 1 -β 1 -α f * t (x * t |x * 0 )∥ 2 real data p * 0 term .
2. Make a generator update step, minimizing the loss L
α,β R-SiD (θ) := E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) -2α SiD • α∥δ t (x θ t )∥ 2 + 2α⟨δ t (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩ , (34
)
where δ t = f t -f * t . In the SiD framework for diffusion models, the data-free generator SiD loss (33) is additionally normalized, and the SiD loss with real data (34) should be normalized the same way. For more details on normalization, time sampling, weighting, etc., refer to the original articles (Zhou et al., 2024a;b).
this section cite: ['b13']

Section: Experimental validation.
We modify the data-free SiD loss in the official SiD implementation with real data and conduct a short ablation study on unconditional CIFAR-10. The SiD codebase for diffusion models can be found in https://github.com/mingyuanzhou/SiD.
We compare the data-free SiD loss (33) and our RealSiD loss (34) with the best coefficients α, β from Table 1 for both the theoretically justified α SiD = 0.5 and the best practical heuristic α SiD = 1.2. We do not change anything else and use the default hyperparameters and training pipeline as described in (Zhou et al., 2024b). The results are presented in Figure 5. For the accurate α SiD = 0.5, the RealSiD results for diffusion models are similar to those for flow models. Configurations with β /α = 1.02 boost convergence compared to the data-free baseline (α = β = 1), whereas in the case of α = β ̸ = 1, the convergence speed remains close to the baseline.
However, for the heuristic α SiD = 1.2, our best configurations with β /α ̸ = 1.0 either degrade performance compared to the baseline or become unstable. This suggests that heuristical SiD may require a different approach to incorporate real data, or a more careful tuning of the coefficients α, β and other hyperparameters, due to differing architectures and training pipelines.
We would like to highlight that all our analyses and recommendations were justified only for α SiD = 0.5. For other α SiD values, this may not hold true.
this section cite: ['b44']

Section: A.4 NORMALIZED UID AND REALUID LOSSES FOR MINIMIZING ℓ 2 -DISTANCE
Using the linearization technique from ( §3.1), we can estimate the non-squared ℓ 2 -distance between the teacher f * := arg min f L UM (f, p * 0 ) and student f θ := arg min f L UM (f, p θ 0 ) functions. In this case, the connection with the inverse optimization disappears.
For a fixed point x θ t ∼ p θ t and time t ∼ [0, T ], we derive:
∥f * t (x θ t ) -f θ t (x θ t )∥ = max δt(x θ t ) ⟨ δ t (x θ t ) ∥δ t (x θ t )∥ , f * t (x θ t ) -f θ t (x θ t )⟩ = max δt(x θ t ) E x θ 0 ∼p θ 0 (•|x θ t ) ⟨ δ t (x θ t ) ∥δ t (x θ t )∥ , f * t (x θ t )⟩ -⟨ δ t (x θ t ) ∥δ t (x θ t )∥ , f θ t (x θ t |x θ 0 )⟩ . (35
)
With the parameterization δ t = f * t -f t , the Normalized UID loss LUID (f, p θ 0 ) for solving
min θ E t∼[0,T ] E x θ t ∼p θ t [∥f * t (x θ t ) -f θ t (x θ t )∥] is min θ max f    LUID (f, p θ 0 ) := E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) ⟨ f * t (x θ t ) -f t (x θ t ) ∥f * t (x θ t ) -f t (x θ t )∥ , f * t (x θ t ) -f θ t (x θ t |x θ 0 )⟩    . (36
)
Adding real data. Following alternative definition of RealUID loss from Lemma 3, we can incorporate real data in Normalized UID loss (36) as well. We need to split two terms in the linearized form (35) into generated and real data parts with weights α, (1 -α) and β, (1 -β).
Definition 4. We introduce Normalized RealUID loss Lα,β R-UID (f, p θ 0 ) on generated data p θ 0 ∈ P(R D ) with coefficients α, β ∈ (0, 1]:
Lα,β R-UID (f, p θ 0 ) := E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) ⟨ f * t (x θ t ) -f t (x θ t ) ∥f * t (x θ t ) -f t (x θ t )∥ , α • f * t (x θ t ) -β • f θ t (x θ t |x θ 0 )⟩ +E t∼[0,T ],x * 0 ∼p * 0 , x * t ∼p * t (•|x * 0 ) ⟨ f * t (x * t ) -f t (x * t ) ∥f * t (x * t ) -f t (x * t )∥ , (1 -α) • f * t (x * t ) -(1 -β) • f * t (x * t |x * 0 )⟩ .
Similar to the proof of RealUID distance Lemma 2, we can show that min-max optimization of Normalized RealUID loss minimizes the non-squared ℓ 2 -norm between the similar weighted student f θ and teacher f * functions:
max f Lα,β R-UID (f, p θ 0 ) = E t∼[0,T ] E x * t ∼p * t ∥((β -α) + α p θ t (x * t ) p * t (x * t ) ) • f * t (x * t ) -β p θ t (x * t ) p * t (x * t ) • f θ t (x * t )∥ .
This distance attains minimum when p θ 0 = p * 0 , justifying the procedure.
this section cite: []

Section: A.5 DMD APPROACH WITH REAL DATA
Distribution Matching Distillation (Luo et al., 2023;Wang et al., 2023;Yin et al., 2024b;a) (DMD) approach distills Gaussian diffusion models with forward process x t = x 0 + σ t ϵ, ϵ ∼ N (0, I).
This approach minimizes KL divergence E t∼[0,T ] [D KL (p θ t ||p * t )] = E t∼[0,T ] E x θ t ∼p θ t log p θ t (x θ t ) p * t (x θ t ) between the generated data p θ t and the real data p * t . The authors show that the true gradient of E t∼[0,T ] [D KL (p θ t ||p * t )] w.r.t. θ can be computed via the score functions:
E t∼[0,T ] dD KL (p θ t ||p * t ) dθ = E z∼p Z ,x θ 0 =G(z),x θ t ∼p θ t (∇ x θ t ln p θ t (x θ t ) -∇ x θ t ln p * t (x θ t )) dG θ (z) dθ .
Then, this true gradient is estimated with the teacher score function s * := arg min s L DSM (s, p * 0 ) and student score s θ = arg min s L DSM (s, p θ 0 ) obtained via minimizing DSM loss (1):
E t∼[0,T ] dD KL (p θ t ||p * t ) dθ = E t∼[0,T ] E z∼p Z ,x θ 0 =G θ (z),x θ t ∼p θ t (s θ t (x θ t ) -s * t (x θ t )) dG θ dθ .
The final algorithm alternates updates for the fake model and the generator similar to SiD approach.
We would like to highlight that DMD does not fit our UID framework. The UID loss is uniquely determined by its input UM loss. In the case of Diffusion models and DMD, the UM loss is the L DSM (s, p θ 0 ) loss. With this loss, the resulting UID loss becomes exactly the SiD loss, not DMD.
Adding real data. We investigated a theoretical possibility to incorporate real data into the DMD framework. We found that we can use the DSM loss with real data (16) to train the modified student score function s θ,α t = arg min s L α,α R-DSM (s, p θ 0 ) with coefficients α = β:
L α,α R-DSM (s, p θ 0 ) := α • E t∼[0,T ] E x θ 0 ∼p θ 0 ,x θ t ∼p θ t (•|x0) γ t ∥s t (x θ t ) -s θ (x θ t |x θ 0 )∥ 2 generated data p θ 0 term + (1 -α) • E t∼[0,T ] E x * 0 ∼p * 0 ,x * t ∼p * t (•|x * 0 ) γ t ∥s t (x * t ) -s * t (x * t |x * 0 )∥ 2 real data p * 0 term .
Then apply the generator parameters update based on the KL divergence between mixed distributions.
Lemma 5 (DMD with real data). Consider real data distribution p * 0 ∈ P(R D ) and generated by generator G θ distribution p θ 0 ∈ P(R D ). Then, KL divergence between mixed and real data for α ∈ (0, 1] has the following gradients with modified student score s θ,α t := arg min s L α,α R-DSM (s, p θ 0 ) and teacher score s * t := arg min s L DSM (s, p * 0 ):
E t∼[0,T ] dD KL (α • p θ t + (1 -α) • p * t ||p * t ) dθ = E t∼[0,T ],z∼p Z , x θ 0 =G θ (z),x θ t ∼p θ t α(s θ,α t (x θ t ) -s * t (x θ t )) dG θ dθ .
Although this approach is theoretically justified, it requires coefficients α = β which work poorly for our RealUID, see Table 1. In the proof below, we also show that use of coefficients α ̸ = β in the fake model loss leads to the total collapse of a generator. The proof itself follows (Wang et al., 2023).
Proof. We aim to minimize KL divergence between generated distribution p θ 0 and the real data p
* 0 min p θ 0 E(p θ 0 ) := E t∼[0,T ] D KL (α • p θ t + (1 -α) • p * t ||p * t ) .
First, we use (Wang et al., 2023, Lemma 1) which says that, for any two distributions p, q ∈ P(R D ) and point x ∈ R D , we have
δD KL (q||p) δq [x] = log q(x) -log p(x) + 1.
Second, for the parameterization x θ 0 = G θ (z), z ∼ p Z and a fixed point x t , we have (Wang et al., 2023
, Lemma 2) δp θ t (x t ) δp θ 0 [θ] = z p θ t (x t |x θ 0 )p Z (z)dz.
It allows us to obtain
δE(p θ 0 ) δp θ 0 [θ] = E t δD KL ( =:qt α • p θ t (•) + (1 -α) • p * t (•)||p * t (•)) δp θ 0 [θ] = E t δD KL (q t ||p * t ) δq t [x t ] • δq t δp θ t [x t ] • δp θ t (x t ) δp θ 0 [θ] • dx t = E t log(α • p θ t (x t ) + (1 -α) • p * t (x t )) -log(p * t (x t )) + 1 • α • z p θ t (x t |x θ 0 )p Z (z)dz • dx t = E t,ϵ,z [α log(α • p θ t (x θ t ) + (1 -α) • p * t (x θ t )) -α log(p * t (x θ t )) + α] = E t,ϵ,z [α log α • p θ t (x θ t ) p * t (x θ t ) + (1 -α) + α],(37)
where
x θ 0 = G θ (z), x θ t = x θ 0 + σ t ϵ, ϵ ∼ N (0, I).
Finally, we take derivative w.r.t. θ from (37):
∇ θ δE(p θ 0 ) δp θ 0 [θ] = E t,ϵ,z α • ∇ x θ t log α • p θ t (x θ t ) p * t (x θ t ) + (1 -α) • ∂x θ t ∂θ = E t,ϵ,z α • ∇ x θ t log α • p θ t (x θ t ) p * t (x θ t ) + (1 -α) • ∂G θ (z) ∂θ = E t,ϵ,z   α 2 ∇ x θ t p θ t (x θ t ) /p * t (x θ t ) α • p θ t (x θ t ) p * (x θ t ) + (1 -α) • ∂G θ (z) ∂θ   .(38)
Now, we show how to obtain unbiased estimate of this gradient. We minimize the following loss function over the fake model s:
L α,α R-DSM (s, p θ 0 ) := α • E t∼[0,T ] E x θ t ∼p θ t ,x θ 0 ∼p θ 0 (•|xt) γ t ∥s t (x θ t ) -s θ (x θ t |x θ 0 )∥ 2 + (1 -α) • E t∼[0,T ] E x * t ∼p * t ,x * 0 ∼p * 0 (•|x * t ) γ t ∥s t (x * t ) -s * t (x * t |x * 0 )∥ 2 .
This loss is equivalent to the following sequence min s α • E t∼[0,T ], x θ t ∼p θ t [∥s t (x θ t ) -s θ t (x θ t )∥ 2 ] + (1 -α) • E t∼[0,T ], x * t ∼p * t [∥s t (x * t ) -s * t (x * t )∥ 2 ] , min s α • E t∼[0,T ], x θ t ∼p θ t [∥s t (x θ t ) -∇ x θ t log p θ t (x θ t )∥ 2 ] + (1 -α) • E t∼[0,T ], x * t ∼p * t [∥s t (x * t ) -∇ x * t log p * t (x * t )∥ 2 ] , min s E t∼[0,T ],
x * t ∼p * t α • ∥s t (x * t ) -∇ log p θ t (x * t )∥ 2 • p θ t (x * t ) p * t (x * t ) + (1 -α) • ∥s t (x * t ) -∇ log p * t (x * t )∥ 2 .
The optimal solution s θ,α of this quadratic minimization for each point x t and time moment t is
s θ,α t (x t ) = α • p θ t (xt) p * t (xt) • ∇ xt log p θ t (x t ) + (1 -α) • ∇ xt log p * t (x t ) α • p θ t (xt) p * t (xt) + (1 -α) .
Thus, we have the following estimate with modified student score s θ,α and teacher score s
* t (x t ) := ∇ xt log p * t (x t ) s θ,α t (x t ) -s * t (x t ) = α • p θ t (xt) p * t (xt) • ∇ xt log p θ t (x t ) + (1 -α) • ∇ xt log p * t (x t ) α • p θ t (xt) p * t (xt) + (1 -α) -∇ xt log p * t (x t ) = α • p θ t (xt) p * t (xt) • (∇ xt log p θ t (x t ) -∇ xt log p * t (x t )) α • p θ t (xt) p * t (xt) + (1 -α) = α • p θ t (xt) p * t (xt) • ∇ xt log p θ t (xt) p * t (xt) α • p θ t (xt) p * t (xt) + (1 -α) = α • ∇ xt ( p θ t (xt) /p * t (xt)) α • p θ t (xt) p * t (xt) + (1 -α) .
Hence, this estimate completely matches with required gradient (38):
(38) = E t,ϵ,z α • (s θ,α (x θ t ) -s * t (x θ t )) • ∂G θ (z) ∂θ .
The use of other coefficients during student score optimization does not work. For the other student scores s θ,α,β t := arg min s L α,β R-DSM (s, p θ 0 ), the estimate s θ,α
,β t (x t ) -∇ xt log p * t (x t ) does not lead to the necessary difference ∇ xt log p θ t (x t ) -∇ xt log p * t (x t ) = 0, and the optimal generator collapses due to large bias.
this section cite: ['b27', 'b37', 'b44', 'b37', 'b37', 'b37']

Section: B REALUID ALGORITHM FOR FLOW MATCHING MODELS
We provide a practical implementation of our RealUID approach for flow matching models in Algorithm 1. In the loss functions, we retain only the terms dependent on the target parameters. For the fake model, we reformulate the maximization objective as a minimization. We use alternating optimization, updating the fake model K times per one student update for stability.
this section cite: []

Section: Algorithm 1 Real data modified Unified Inversion Distillation (RealUID) for Flow Matching
Input: teacher drift u * , student generator G θ , fake drift u ψ , real data p * 0 , coefficients α, β ∈ (0, 1], generator update steps K, number of iterations N , batch size B, fake drift minimizer Opt st , generator minimizer Opt gen , latent distribution p Z , noise distribution p 1 . 1: for n = 0, . . . , N -1 do
2: Sample generated batch {x θ 0,i = G θ (z i )} B i=1 , z i ∼ p Z and noise batch {x 1,i } B i=1 ∼ p 1 ; 3: Sample time batch {t i } B i=1 ∼ U nif orm[0, 1] and calculate x θ ti,i = (1 -t i )x θ 0,i + t i x 1,i ; 4: if student step (n%(K + 1) ̸ = 0) then 5: Sample real data batch {x * 0,i } B i=1 ∼ p * 0 and calculate x * ti,i = (1 -t i )x * 0,i + t i x 1,i ; 6: Update fake drift parameters ψ via minimizer Opt st step with gradients of 1 B B i=1 α∥u ψ (t i , x sg[θ] ti,i )-β α (x 1,i -x sg[θ] 0,i )∥ 2 +(1 -α)∥u ψ (t i , x * ti,i )-1-β 1-α (x 1,i -x * 0,i )∥ 2 ; 7: else 8: Update generator parameters θ via minimizer Opt gen step with gradients of
1 B B i=1 α∥u * (t i , x θ ti,i ) - β α (x 1,i -x θ 0,i )∥ 2 -α∥u sg[ψ] (t i , x θ ti,i ) - β α (x 1,i -x θ 0,i )∥ 2 ; 9:
end if 10: end for
this section cite: []

Section: C UNIFIED INVERSE DISTILLATION WITH REAL DATA FOR BRIDGE MATCHING AND STOCHASTIC INTERPOLANTS
C.1 BRIDGE MATCHING Bridge Matching (Liu et al., 2022;Peluchetti, 2023) is an extension of diffusion models specifically design to solve data-to-data, e.g., image-to-image problems. Typically, the distribution p T is the distribution of "corrupted data" and p 0 is the distribution of clean data, furthermore, there is some coupling of clean and corrupted data π(x 0 , x T ) with marginals p 0 (x 0 ) and p T (x T ). To construct the diffusion which recovers clean data given a corrupted data, one first needs to build prior process (which often is the same forward process used in diffusions):
dx t = f t (x t )dt + g t dw t ,
where f t (•) is a drift function, g t is a time-dependent scalar noise scheduler and w t is a standard Wiener process. This prior process defines conditional density p t (x t |x 0 ) and the posterior density p t (x t |x 0 , x T ) called "diffusion bridge". To recover p 0 from p T , one can use reverse-time SDE with a reverse-time Wiener process wt :
dx t = f t (x t ) -g 2 t • u t (x t
) dt + g t d wt , where the drift u t (x t ) is learned via solving of the bridge matching problem:
L BM (v, π) = E t∼[0,T ],(x0,x T )∼π,xt∼pt(•|x0,x T ) ∥v t (x t ) -∇ xt log p t (x t |x 0 )∥ 2 .(39)
However, this reverse-time diffusion in general does not guarantee that the produced samples come from the same coupling π(x 0 , x T ) used for training. This happens only if π(x 0 , x T ) solves entropic optimal transport between p 0 and p T . To guarantee the preservance of the coupling π(x 0 , x T ), there exists another version of Bridge Matching called either Augmented Bridge Matching or Conditional Bridge Matching (De Bortoli et al., 2023), which differs only by addition of a condition on x T to the trainable drift v t (x t , x T ):
L ABM (v, π) = E t∼[0,T ],(x0,x T )∼π,xt∼p(•|x0,x T ) ∥v t (x t , x T ) -∇ xt log p t (x t |x 0 )∥ 2 2 .
The learned conditional drift u(x t , x T ) is then used for sampling via the reverse-time SDE starting from a given x T ∼ p T :
dx t = f t (x t ) -g 2 t • u t (x t , x T ) dt + g t d wt .
Table 1 for CIFAR10, the same pairs of coefficients with β /α = 1.02 or β /α = 0.98 yield a significant improvement in quality over the baseline (α = 1.0, β = 1.0), reaching a level comparable to GANs. Training hyperparameters and details. We take the same architecture (Tong et al., 2024) as for the CIFAR-10 dataset, but adapt it to a larger resolution. We train it with Adam (Kingma & Ba, 2014) for 800,000 iterations, using (β 1 , β 2 ) = (0, 0.999), learning rate 5 × 10 -6 and a 500-step linear warm-up. Similar to SiD framework (Zhou et al., 2024a), we do not recommend setting momentum β 1 ̸ = 0 as it is crucial for a successful convergence in our min-max optimization.
To regulate adaptation between the generator and the fake model, the generator is updated once for every K = 5 updates of the fake model, following DMD2 (Yin et al., 2024a). While the SiD framework leverages an EDM architecture (Karras et al., 2022) and updates the generator after a single update of the fake model (K = 1), our RealUID approach becomes unstable for values K < 3 due to the different (Tong et al., 2024) architecture.
We do not use dropout in generator and fake models. We set a batch size of 64 and maintain an EMA of the generator parameters with decay 0.999 (Hunter, 1986). Additionally, at each optimization step we apply ℓ 2 gradient-norm clipping with threshold 1.0 to both the generator and the fake model.
All other details remain the same as described in Appendix D.1 for CIFAR-10.
Teacher training. For CelebA, we train our own teacher model based on the official implementation of the conditional flow matching procedure from (Tong et al., 2024). We use the same pipeline, architectures, and hyperparameters, but with larger networks and a different dataset. The adapted code for teacher training and final checkpoints for distillation can be found in our repository: https://github.com/David-cripto/RealUID.
Fine-tuning. For fine-tuning, we hold the data-free UID baseline (our RealUID with α = 1.0, β = 1.0) and all highlighted GAN and RealUID setups from Table 7 for twice as long, i.e., for 1,600,000 iterations. The best-found configurations and results are reported in Table 8 and  Figure 8: Uncurated samples for unconditional generation by the one-step UID + GAN (λ G θ adv = 0.3, λ D adv = 1|λ G θ FT = 25, λ D FT = 75) trained on CIFAR-10. Quantitative results are reported in Table 2.  2.  2.  2.  D.5.2 CELEBA GENERATED IMAGES Figure 13: Uncurated samples by the one-step data-free baseline UID trained on CelebA. Quantitative results are reported in Table 7.  7.  7.
this section cite: ['b23', 'b28', 'b3', 'b35', 'b17', 'b35', 'b14', 'b35']

Section: 
t), t 2 + (1 -t) 2 ) and p * t (x t ) = N (x t |µ * (1 -t), t 2 + (1 -t) 2 ). The unconditional vector field u = f between N (µ, 1) and N (0, 1) can be calculated as
u t (x t ) = E x0∼p0(•|xt) x t -x 0 t = x0 x t -x 0 t • N x t -x 0 (1 -t) t |0, 1 • N (x 0 |µ, 1)dx 0 = a(2t 2 -2t) -bt 2 √ 2π(1 -2t + 2t 2 ) 3 2 exp - (x t -µ(1 -t)) 2 2(1 -2t + 2t 2 ) 2 .(27)
In Figure 4, we depict the optimal generator mean µ θ and vector field u θ satisfying (26) for various deviations ũ * -u * and fixed time t = 1/3, real data µ * = -2 and point x t = -1.
We can see that with α = β = 1, the generator learns the corrupted vector field, and with α = β < 1, the learned field and means are often even worse. In contrast, with β /α ̸ = 1, the generator can learn vector fields and means which are closer to the real data. Although the generator cannot satisfy relation (26) under large deviations, it still produces better results with the real data.
0.560 0.565 0.570 0.575 0.580 0.585 0.590 u * u * = 0.01 gen u , = -2.09, ( , ) = (0.95, 0.9) gen u , = -2.06, ( , ) = (0.9, 0.95) gen u , = -2.07, ( , ) = (0.9, 0.9) gen u , = -2.07, ( , ) = (1.0, 1.0) real u * , * = -2 corrupted u * 0.50 0.52 0.54 0.56 0.58 0.60 u * u * = 0.05 gen u , = -2.34, ( , ) = (0.95, 0.9) gen u , = -2.25, ( , ) = (0.9, 0.95) gen u , = -2.29, ( , ) = (0.9, 0.9) gen u , = -2.26, ( , ) = (1.0, 1.0) real u * , * = -2 corrupted u * 0.400 0.425 0.450 0.475 0.500 0.525 0.550 0.575 0.600 u * u * = 0.1 gen u , = -2.62, ( , ) = (0.95, 0.9) gen u , = -2.43, ( , ) = (0.9, 0.95) gen u , = -2.52, ( , ) = (0.9, 0.9) gen u , = -2.45, ( , ) = (1.0, 1.0) real u * , * = -2 corrupted u * 0.0 0.1 0.2 0.3 0.4 0.5 0.6 u * u * = 0.5 gen u , = -2.82, ( , ) = (0.95, 0.9) gen u , = -3.14, ( , ) = (0.9, 0.95) gen u , = -2.84, ( , ) = (0.9, 0.9) gen u , = -3.92, ( , ) = (1.0, 1.0) real u * , * = -2 corrupted u * 0.570 0.575 0.580 0.585 0.590 0.595 0.600 u * u * =0.01 gen u , = -1.87, ( , ) = (0.95, 0.9) gen u , = -1.92, ( , ) = (0.9, 0.95) gen u , = -1.9, ( , ) = (0.9, 0.9) gen u , = -1.91, ( , ) = (1.0, 1.0) real u * , * = -2 corrupted u * 0.56 0.58 0.60 0.62 0.64 0.66 0.68 0.70 u * u * =0.1 gen u , = -1.73, ( , ) = (0.95, 0.9) gen u , = -1.7, ( , ) = (0.9, 0.95) gen u , = -1.71, ( , ) = (0.9, 0.9) gen u , = -1.72, ( , ) = (1.0, 1.0) real u * , * = -2 corrupted u *
this section cite: []

Section: A.2 GENERAL REALUID LOSS
Extending our real data incorporation. We recall that UID loss (Theorem 1) can be restated via linearization technique with δ = f * -f as:
L UID (δ, p θ 0 ) = E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) -∥δ t (x θ t )∥ 2 + 2⟨δ t (x θ t ), f * t (x θ t )⟩ -2⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩ .
Following alternative definition of RealUID loss from Lemma 3, one can incorporate real data into data-free loss by splitting each term in the linearized form between generated and real data as:
L α,β R-UID (δ, p θ 0 ) = E t∼[0,T ],x θ 0 ∼p θ 0 , x θ t ∼p θ t (•|x θ 0 ) [-α∥δ t (x θ t )∥ 2 + 2α⟨δ t (x θ t ), f * t (x θ t )⟩ -2β⟨δ t (x θ t ), f θ t (x θ t |x θ 0 )⟩] + E t∼[0,T ],x * 0 ∼p * 0 , x * t ∼p * t (•|x * 0 ) [-(1 -α)∥δ t (x * t )∥ 2 + 2(1 -α)⟨δ t (x * t ), f * t (x * t )⟩ -2(1 -β)⟨δ t (x * t ), f * t (x * t |x * 0 )⟩].
In RealUID loss (17), its three terms are split with proportions α and 1 -α, α and 1 -α and β and 1 -β, respectively. We can go even further and split the first quadratic coefficient -∥δ t (•)∥ 2 using a new parameter γ to create one more degree of freedom. Moreover, we can use other parameterization of δ, since its form does not change the proof of distance lemma.
this section cite: []

Section: C.2 STOCHASTIC INTERPOLANTS
The Stochastic Interpolants framework generalizes Flow Matching and diffusion models, constructing a diffusion or flow between two given distributions p 0 and p T . To do so, one needs to consider the interpolation between any pair of points (x 0 , x T ) which are sampled from the coupling π(x 0 , x T ) with marginals p 0 and p T . The interpolation itself is given by formula
x t = I(t, x 0 , x T ) + γ t ϵ, ϵ ∼ N (0, I), t ∈ [0, T ], where I(0, x 0 , x T ) = x 0 , I(T, x 0 , x T ) = x T , γ 0 = γ T = 0 and γ t > 0 for all t ∈ (0, T ). This interpolant defines a conditional Gaussian path p t (x t |x 0 , x T ). Note that in the original paper (Albergo et al., 2023), the authors consider the time interval [0, 1], but those two intervals are interchangeable by using a change of variable t ′ = T t . Thus, the ODE interpolation between p 0 and p T is given by:
dx t = u t (x t )dt, x 0 ∼ p 0 , where u t (x, x T ) := E[ ẋt |x t = x] = E[∂ t I(t, x 0 , x T ) + γt ϵ|x t = x]
is the unique minimizer of the quadratic objective:
L SI (v, π) = E t∼[0,T ],(x0,x T )∼π, (xt,ϵ)∼p(•|x0,x T ) ∥v t (x t , x T ) -(∂ t I(t, x 0 , x T ) + γt ϵ)∥ 2 . (40
)
The authors also provide a way of matching the score and the SDE drift of the reverse process by solving similar MSE matching problems.
this section cite: ['b0']

Section: C.3 OBJECTIVE FOR GENERAL DATA COUPLING
The essential difference of Bridge Matching and Stochastic Interpolants from diffusion models and Flow Matching with a Gaussian path is that they additionally introduce coupling π(x 0 , x T ) used to sample x t and can work with conditional drifts.
This difference can be easily incorporated to our RealUID distillation framework just by parametrizing the generator G θ to output not the samples from the initial distribution p θ 0 , but from the coupling π θ . One can do it by setting π θ (x 0 , x T ) = p T (x T )π θ 0 (x 0 |x T ), where conditional data distribution
π θ 0 (x 0 |x T ) is parametrized by the student generator G θ : Z × R D → R D conditioned on a sample x T ∼ p T .
This approach is specifically used in Inverse Bridge Matching Distillation (IBMD) (Gushchin et al., 2024). Hence, our Universal Inverse Distillation objective can be written just by substituting student distribution p θ 0 by student coupling π θ , substituting real data p * 0 by real data coupling π * and adding extra conditions. Definition 5. We define Universal Matching loss with real data for general coupling on generated data coupling π θ ∈ P(R D × R D ) with α, β ∈ (0, 1]:
L α,β R-UM-coup (f, π θ ) = α • E t∼[0,T ] E x T ∼p T ,x θ 0 ∼π θ 0 (•|x T ), x θ t ∼p θ t (•|x θ 0 ,x T ) ∥f t (x θ t , x T ) - β α f θ (x θ t |x θ 0 , x T )∥ 2 generated data π θ term + (1 -α) • E t∼[0,T ] E x T ∼p T ,x * 0 ∼π * 0 (•|x T ), x * t ∼p * t (•|x0,x T ) ∥f t (x * t , x T ) - 1 -β 1 -α f * t (x * t |x * 0 , x T )∥ 2 real data π * term .
And the corresponding Universal Inverse Distillation loss with real data for general coupling is:
min θ max f {L α,β R-UID-coup (f, π θ ) := L α,β R-UM-coup (f * , π θ ) -L α,β R-UM-coup (f, π θ )}.
In case of coupling match π θ = π * , the RealUID loss for couplings attains its minimum, i.e.,
min θ max f L α,β R-UID-coup (f, π θ ) = min θ {L α,β R-UM-coup (f * , π θ ) -min f {L α,β R-UM-coup (f, π θ )} ≥0 } = L α,β R-UM-coup (f * , π * ) -min f {L α,β R-UM-coup (f, π * )} =L α,β R-UM-coup (f * ,π * ) = 0.
this section cite: ['b8']

Section: D EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

this section cite: []

Section: D.1 CIFAR-10 DISTILLATION FROM SCRATCH
Codebase, dataset and teachers. Building on the reference codebase and network architectures of (Tong et al., 2024), we implement the training algorithm described in our Algorithm 1. We evaluate the resulting approach on CIFAR-10 (32×32), under both conditional and unconditional settings, benchmarking against established baselines. The codebase implementation is publicly available in https://github.com/atong01/conditional-flow-matching.
Note that in this codebase, the time flow is reversed, i.e., the time t = 0 corresponds to the pure noise, while the time t = 1 is the real data. As an unconditional teacher, we use already trained Conditional Flow Matching checkpoints from the above repository. For conditional setup, we slightly modify the original code and train our own teacher. Our trained checkpoints, along with the code, are located in https://github.com/David-cripto/RealUID.
Training hyperparameters. We train our models with Adam (Kingma & Ba, 2014), using (β 1 , β 2 ) = (0, 0.999), learning rate 3 × 10 -5 and a 500-step linear warm-up. Similar to SiD framework (Zhou et al., 2024a), we do not recommend setting momentum β 1 ̸ = 0 as it is crucial for a successful convergence in our min-max optimization.
To regulate adaptation between the generator and the fake model, the generator is updated once for every K = 5 updates of the fake model, following DMD2 (Yin et al., 2024a). While the SiD framework leverages an EDM architecture (Karras et al., 2022) and updates the generator after a single update of the fake model (K = 1), our RealUID approach becomes unstable for values K < 3 due to the different (Tong et al., 2024) architecture.
We do not use dropout in generator and fake models. We set a batch size of 256 and maintain an EMA of the generator parameters with decay 0.999 (Hunter, 1986). Additionally, at each optimization step we apply ℓ 2 gradient-norm clipping with threshold 1.0 to both the generator and the fake model.
this section cite: ['b35', 'b17', 'b35', 'b14']

Section: Training time.
All distillation experiments were trained for 500,000 gradient updates, corresponding to approximately 5 days. The experiments were executed on a single Ascend910B NPU with 65 GB of VRAM memory.
this section cite: []

Section: Generator parameterization and models initialization.
We parameterize generator G θ (•) using a time-dependent U-Net g θ (0, •) with a fixed time input t = 0 and a one-step integration scheme:
G θ (z) = z + g θ (0, z).
We initialize the model g θ with a teacher model, and the fake model with random weights. Empirically, we observe that this initialization strategy lead to improved performance on the considered datasets.
GAN details. We integrate a GAN loss into our framework in line with SiD 2 A and DMD2 (Zhou et al., 2024a;Yin et al., 2024a). In the original setup of (Zhou et al., 2024a), the adversarial loss employs a coefficient ratio of
λ D adv /λ G θ adv = 10 2 (see
Table 6 in Zhou et al. 2024a), a choice that poses practical difficulties due to the extreme imbalance between the generator and discriminator losses. To mitigate this issue, we adopt the formulation of (Yin et al., 2024a), where the ratio is ≈ 3, and evaluate different coefficient scales (see the results in Table 1). Additionally, we can select the range of times within which adversarial loss is applied between noised generated and real data samples. We found that the best choice is not to take only clear real data or the whole interval [0,1], but rather to take the range of not severely corrupted data, namely times from 0.8 to 1.
Evaluation protocol. We evaluate image quality using the Fréchet Inception Distance (FID; Heusel et al., 2017), computed from 50,000 generated samples following (Karras et al., 2022;2020;2019). In line with SiD (Zhou et al., 2024b), we periodically compute FID during distillation and select the checkpoint achieving the minimum value. To ensure statistical reliability, we repeat the evaluation over 3 independent runs, rather than 10 as in SiD, because the empirical variance of FID in our experiments was below 0.01.
this section cite: ['b10', 'b17', 'b15', 'b44']

Section: Efficiency comparison.
In terms of efficiency, RealUID leverages a lightweight architecture based on (Tong et al., 2024). Therefore, as summarized in Table 5, it achieves nearly 2× faster inference, lower memory usage, and reduced model size compared to recent distillation approaches (Zhou et al., 2024b;a;Huang et al., 2024).
this section cite: ['b35', 'b44', 'b13']

Section: D.2 CIFAR-10 DISTILLATION FINE-TUNING
This section presents an ablation study of the fine-tuning stage over the loss-balancing coefficients for GANs and our RealUID on CIFAR-10. In this stage, the generator is initialized from the bestperforming checkpoint obtained during training from scratch of the corresponding framework, while the fake model is initialized from the teacher model. In the unconditional setup, the best configuration are RealUID with (α = 0.92, β = 0.94) and FID 2.22, and GAN with (λ G θ adv = 0.3, λ D adv = 1) and FID 2.29. In the unconditional setup, it is RealUID with (α = 0.98, β = 0.96) and FID 2.02, and GAN with (λ G θ adv = 0.3, λ D adv = 1) and FID 2.12. Fine-tuning then proceeds with new values α FT and β FT for our RealUID and λ G θ FT and λ D FT for GANs. The results are summarized in Table 6. FT , λ D FT ) for GANs for unconditional (left) and conditional (right) generation. All values report FID ↓, where lower is better. The mark "-" indicates that configuration is infeasible, and the mark "-" shows that the method did not converge. Best results for each method are bolded.
αFT\ βFT αFT 0.92 0.94 0.96 0.98 1.0 1.02 1.04 1.06 1.08 0.92 1.99 1.98 2.02 ---2.04 2.04 2.02 0.94 2.02 2.02 2.04 ---2.07 2.06 -0.96 2.06 2.04 2.09 ---2.08 --0.98 2.07 2.05 2.07 -----λ Gθ FT 0.1 0.3 1.0 5.0 25.0 100.0 λ D FT 0.3 1.0 3.0 15.0 75.0 300.0 FID↓ ---2.25 2.10 2.12 αFT\ βFT αFT 0.92 0.94 0.96 0.98 1.0 1.02 1.04 1.06 1.08 0.92 1.92 1.91 1.99 ---1.96 1.94 1.92 0.94 1.92 1.90 1.88 ---1.96 1.91 -0.96 1.93 1.94 1.87 ---1.96 --0.98 1.91 1.95 1.95 -----λ Gθ FT 0.1 0.3 1.0 5.0 25.0 100.0 λ D FT 0.3 1.0 3.0 15.0 75.0 300.0 FID↓ ---1.94 1.88 2.04
We observe that fine-tuning is highly sensitive to the choice of factor βFT αFT which still brings the main impact. The best factors βFT αFT = 0.94 or βFT αFT = 1.06 are much farther from 1.0 compared to training from scratch (Table 1), i.e., fine-tuning relies more on information from real data rather than on guidance from a teacher. Meanwhile, configurations closer to 1.0 are unstable, underscoring the crucial role of real data. In the case of GANs, small adversarial losses similarly fail to converge, and only high scales which particularly emphasize real data achieve improvement.
Training details. We run fine-tuning with a smaller learning rate 1 × 10 -5 and without warm-up. All other details remain the same as described in Appendix D.1 for training from scratch.
Training time. All fine-tuning experiments were conducted for 100,000 gradient updates, which took a little more than 1 day, starting from the best distillation checkpoints. The experiments were executed on a single Ascend910B NPU with 65 GB of VRAM memory.
this section cite: []

Section: D.3 CELEBA DISTILLATION
In this section, we present the results of the same ablation study from ( §4.2) on the CelebA dataset with higher 64 × 64 resolution (Liu et al., 2015). The results are summarized in
Table 7. Similar to Published as a conference paper at ICLR 2026
this section cite: ['b25']

Section: D.4 FURTHER HYPERPARAMETERS GRIDSEARCH
The primary goal across all experiments in this paper was to study RealUID framework, focusing on the effects of the coefficients α and β, and provide a fair comparison with GANs. For this reason, we kept all other hyperparameters fixed at their standard values. Now that we have identified the optimal settings for RealUID, we can explore other hyperparameters. Below, we provide a list of useful findings, while the latest hyperparameters sets and training pipelines are described in our repository https://github.com/David-cripto/RealUID.
EMA decays. One can track not only a single EMA decay but a range of values, e.g., [0.999, 0.9996, 0.9999], during a single training run. In long-distance training, larger EMA decays can lead to more stable convergence dynamics and better metrics, whether training from scratch or fine-tuning.
this section cite: []

Section: D.5 EXAMPLE OF SAMPLES FOR VARIOUS METHODS
This section presents representative sample outputs from various studies conducted within the RealUID framework.
this section cite: []

Section: References
Ref_id:b0 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b1 Title: Large scale gan training for high fidelity natural image synthesis Year: (2018)
Ref_id:b2 Title: Inverse optimization: Theory and applications Year: (2025)
Ref_id:b3 Title: Augmented bridge matching Year: (2023)
Ref_id:b4 Title: Diffusion models and gaussian flow matching: Two sides of the same coin Year: (2025)
Ref_id:b5 Title: One-step diffusion distillation via deep equilibrium models Year: (2023)
Ref_id:b6 Title: Mean flows for one-step generative modeling Year: (2025)
Ref_id:b7 Title: Generative adversarial nets Year: (2014)
Ref_id:b8 Title: Entropic neural optimal transport via diffusion processes Year: (2024)
Ref_id:b9 Title: Dmitry Baranchuk, and Alexander Korotin. Inverse bridge matching distillation Year: (2025)
Ref_id:b10 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b11 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b12 Title: Generator matching: Generative modeling with arbitrary markov processes Year: (2024)
Ref_id:b13 Title: Flow generator matching Year: (2024)
Ref_id:b14 Title: The exponentially weighted moving average Year: (1986)
Ref_id:b15 Title: A style-based generator architecture for generative adversarial networks Year: (2019)
Ref_id:b16 Title: Analyzing and improving the image quality of stylegan Year: (2020)
Ref_id:b17 Title: Elucidating the design space of diffusionbased generative models Year: (2022)
Ref_id:b18 Title: Consistency trajectory models: Learning probability flow ode trajectory of diffusion Year: (2023)
Ref_id:b19 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b20 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b21 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b22 Title: Rectified flow: A marginal preserving approach to optimal transport Year: (2022)
Ref_id:b23 Title: Let us build bridges: Understanding and extending diffusion generative models Year: (2022)
Ref_id:b24 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b25 Title: Deep learning face attributes in the wild Year: (2015-12)
Ref_id:b26 Title: Simplifying, stabilizing and scaling continuous-time consistency models Year: (2024)
Ref_id:b27 Title: Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models Year: (2023)
Ref_id:b28 Title: Non-denoising forward-time diffusions Year: (2023)
Ref_id:b29 Title: Flow-anchored consistency models Year: (2025)
Ref_id:b30 Title: scaling stylegan to large diverse datasets Year: (2022)
Ref_id:b31 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b32 Title: Improved techniques for training consistency models Year: (2023)
Ref_id:b33 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b34 Title: Inducing metrizability of gan with discriminative normalized linear layer Year: (2023)
Ref_id:b35 Title: Improving and generalizing flow-based generative models with minibatch optimal transport Year: (2024)
Ref_id:b36 Title: Diffusion-gan: Training gans with diffusion Year: (2022)
Ref_id:b37 Title: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation Year: (2023)
Ref_id:b38 Title: Stefano Ermon, and Bin Cui. Consistency flow matching: Defining straight flows with velocity consistency Year: (2024)
Ref_id:b39 Title: Improved distribution matching distillation for fast image synthesis Year: (2024)
Ref_id:b40 Title: One-step diffusion with distribution matching distillation Year: (2024)
Ref_id:b41 Title: Diffusion bridge implicit models Year: (2024)
Ref_id:b42 Title: Inductive moment matching Year: (2025)
Ref_id:b43 Title: Adversarial score identity distillation: Rapidly surpassing the teacher in one step Year: (2024)
Ref_id:b44 Title: Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation Year: (2024)
