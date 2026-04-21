Title: Score-of-Mixture Training: One-Step Generative Model Training Made Simple via Score Estimation of Mixture Distributions
Abstract: We propose Score-of-Mixture Training (SMT), a novel framework for training one-step generative models by minimizing a class of divergences called the α-skew Jensen-Shannon divergence. At its core, SMT estimates the score of mixture distributions between real and fake samples across multiple noise levels. Similar to consistency models, our approach supports both training from scratch (SMT) and distillation using a pretrained diffusion model, which we call Score-of-Mixture Distillation (SMD). It is simple to implement, requires minimal hyperparameter tuning, and ensures stable training. Experiments on CIFAR-10 and ImageNet 64×64 show that SMT/SMD are competitive with and can even outperform existing methods.

Section: Introduction
Fast and efficient sampling is a key characteristic sought after in modern generative samplers. For many years, generative adversarial networks (GANs) (Goodfellow et al., 2014) set the benchmark for high-quality one-step generative sampling. However, due to the inherent training instabilities associated with discriminator training, attention has recently shifted toward diffusion-based generative models (Sohl-Dickstein et al., 2015;Ho et al., 2020;Karras et al., 2022b). These models trade-off sampling efficiency for more stable training and significantly improved downstream sample quality through iterative sampling.
More recently, the diffusion distillation approach has been studied as an appealing option to significantly reduce the number of sampling steps. Early work (Luhman & Luhman, Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). 2021; Salimans & Ho, 2022;Meng et al., 2023;Berthelot et al., 2023) focused on training a student model with a lower sampling budget by condensing multiple teacher denoising steps into one. The most recent works on distillation improve performance further by leveraging a pretrained model for distribution matching via minimization of the reverse KL divergence (Luo et al., 2024a;Yin et al., 2024b;a;Salimans et al., 2024;Xie et al., 2024). While attractive, distillation approaches necessitate a pretrained diffusion model which adds a significant overhead on the required compute.
As yet another alternative, consistency models (Song et al., 2023;Song & Dhariwal, 2024b) and their variants (Kim et al., 2024) have been proposed for training few-step generative models from scratch by simulating the trajectories of the induced probability flow ODE (Song et al., 2020) of a diffusion process. While consistency models have demonstrated promising results in both distillation and training from scratch, training is sensitive to the choice of noise schedule and distance measure (Geng et al., 2025).
In this paper, we tackle the problem of training high-quality one-step generative models more directly, i.e., without simulating an iterative reverse diffusion process for sampling or leveraging a pretrained diffusion model during training. Starting from first principles of statistical divergence minimization, we show that a high-quality one-step generative model can be trained from scratch in a stable manner, via the multi-noise-level denoising score matching (DSM) technique (Vincent, 2011) used in diffusion models. We emphasize that we do not require a simulation of the reverse diffusion process in our framework.
The proposed framework achieves the best of several worlds: (1) a new, simple statistical divergence minimization framework without probability paths of ODE (like GAN), (2) stable training using denoising score matching (like diffusion models), (3) training from scratch without a pretrained diffusion model (like consistency models), and (4) near state-of-the-art one-step image generative performance (like GAN and consistency models). We also demonstrate that the proposed method can be extended to distill from a pretrained diffusion model, and can achieve performance similar to state-of-the-art methods for the same. See Table 1 for the overview of comparison. The rest of the paper is organized as follows: In Sec. 2 we introduce the necessary background and related works central to our proposed method. In Sec. 3 we introduce our novel one-step generative modeling approach and in Sec. 4 we detail how our framework can be modified to perform diffusion distillation. We describe practical implementation details in both the latter sections and present experimental results in Sec. 5. We conclude with remarks in Sec. 6. Proofs and training details are deferred to Appendix.
this section cite: ['b9', 'b47', 'b12', 'b44', 'b37', 'b1', 'b59', 'b45', 'b57', 'b54', 'b50', 'b19', 'b52', 'b8', 'b55']

Section: Preliminaries and Related Work
In one-step generative modeling, we wish to align the generated sample distribution q θ (x) := δ(xg θ (z))q(z) dz with the true data distribution p(x). Here, g θ : Z → X is a parametric neural sampler which is also often called an implicit generative model that transforms samples from a base measure q(z). In this section, we review some popular methods for training generative models, which will serve as preliminaries for our framework. More detailed discussion on the literature is deferred to Appendix B.
this section cite: []

Section: Generative Adversarial Networks.
The most prominent approach in training implicit generative models is the generative adversarial network (GAN) (Goodfellow et al., 2014).
In its most standard and widely used form, it alternates between the gradient steps of discriminator and generator training, which are
min ψ E p(x) [sp(-ℓ ψ (x))] + E q θ (x) [sp(ℓ ψ (x))], (1) min θ E q θ (x) [sp(-ℓ ψ (x))],(2)
respectively, where sp(y) := log(1 + e y ) denotes the softplus function. 1 Here, we will call ℓ ψ (x) the discriminator, which is supposed to capture the log density ratio log p(x) q θ (x) .foot_1 This so-called adversarial training can be understood as min-
definition of discriminator D ψ (x) := exp(ℓ ψ (x)) 1+exp(ℓ ψ (x)) ∈ [0, 1].
imizing the Jensen-Shannon divergence (JSD) with the help of discriminator, via the variational characterization of JSD.
Despite the popularity of GANs, training them is notoriously difficult. Although various techniques have been proposed to regularize the GAN objective-through alternatives to JSD (Nowozin et al., 2016;Arjovsky et al., 2017;Mao et al., 2017), novel regularizers (Miyato et al., 2018), and specialized network architectures (Karras et al., 2021;Brock et al., 2019;Sauer et al., 2022)-the discriminator training remains unstable. This has sparked increasing interest in developing new objectives for training generative models which we briefly discuss below.
this section cite: ['b9', 'b41', 'b0', 'b36', 'b38', 'b16', 'b2', 'b46']

Section: Diffusion Models.
Diffusion models or score-based generative models (Sohl-Dickstein et al., 2015;Ho et al., 2020) are state-of-the-art generative models that are based on the principles of thermodynamic diffusion. Given a forward stochastic differential equation (SDE) process
dx t = f (x t , t)dt + g(t)dw t ,
where f (x t , t) is the drift function, g(t) is the diffusion function, and w t represents a Brownian noise process, diffusion models simulate the reverse (generative) process, which is also an SDE
dx t = [f (x t , t) -g(t) 2 ∇ xt log p(x t )]dt + g(t)d wt .
An equivalent deterministic probability flow ODE with the same marginals as the SDE can also be used in practice:
dx t = f (x t , t) - 1 2 g(t) 2 ∇ xt log p(x t ) dt.
Thus, to generate samples, diffusion models are trained to learn the score of the data distribution at multiple noise levels σ t via denoising score matching (DSM) (Vincent, 2011), i.e., by minimizing
L DSM (θ) = E p(x)q(z)p(t) w(t)∥s θ (x t ; t) -s(x t |x)∥ 2 ,
where p(t) denotes a distribution over different noise levels, x t := x + σ t ϵ, ϵ ∼ N (0, I), and s(x t |x) := ∇ xt log p(x t |x). It is easy to show that s θ (x t ; t) = ∇ xt log p(x t ) using Tweedie's formula (Robbins, 1956). Sampling can then be achieved by Langevin dynamics (Song & Ermon, 2019;Song et al., 2020) or via black-box ODE solvers (Karras et al., 2022b;Lu et al., 2022b;c).
this section cite: ['b47', 'b12', 'b55', 'b43', 'b51', 'b52']

Section: Diffusion Distillation.
In practical applications, running a diffusion model for multiple steps to generate a single sample can be prohibitively expensive. Distilling few-step generative models from a high-quality pretrained diffusion model has thus become popular (Luo et al., 2024a;Yin et al., 2024b;a;Salimans et al., 2024;Xie et al., 2024). To learn the generator's parameters, most, if not all, approaches aim to minimize the reverse Kullback-Leibler divergence (KLD) D KL (q θ ∥p) averaged across multiple noise levels:
D avg KL (q θ ∥p) := E q θ (x)p(t)q(ϵ) [log q θ (x t ) -log p(x t )].
To update the parameters via gradient descent, the gradient of this divergence is computed as
∇ θ D avg KL (q θ ∥p) (3) = E q(z)p(t)q(ϵ) [∇ θ g θ (z)(s q θ (x t ; t) -s p (x t ; t)) | x=g θ (z) ],
where s q θ and s p are the noisy scores of the fake and true samples, respectively. In the distillation setup, a pretrained diffusion model is plugged in as a close proxy to the true noisy score s p (x t ; t), while the fake noisy score s q θ (x t ; t) is trained along with the generator to assist the training.
this section cite: ['b59', 'b45', 'b57']

Section: Consistency Models.
Distillation approaches often rely on pretrained score models and may use expensive regularizers to address issues like mode collapse and improve sample quality (Yin et al., 2024b;Salimans et al., 2024). In contrast, consistency models (Song et al., 2023;Song & Dhariwal, 2024b), which can be trained from scratch, are trained to simulate the underlying probability flow ODE and ensure each sample along the trajectory maps to the origin. Consistency training, however, can be unstable and is known to sensitive to the noise schedule and distance function (Geng et al., 2025). Additionally, the architecture for consistency models need to be carefully chosen, as the approach relies on a single-sample approximation of Tweedie's formula, which is only valid when noise levels are closely spaced.
this section cite: ['b59', 'b45', 'b54', 'b50', 'b8']

Section: Training from Scratch
In this section, we introduce our new framework, Scoreof-Mixture Training (SMT). We describe how to efficiently train one-step generative models from scratch, i.e.,, without a pretrained diffusion model. In Sec. 4, we explain how the framework can be adapted to leverage a pretrained diffusion model when available, referring to this variant as Score-of-Mixture Distillation (SMD).
The key ingredient of this framework is distribution matching using a new family of statistical divergences (Sec. 3.1), whose gradient can be approximated by estimating the score of mixture distributions of real and fake distributions (Sec. 3.3), hence the name Score of Mixture Training. We adopt the concept of multi-noise level learning from diffusion models and propose multi-divergence minimization for stable training (Sec. 3.2). A practical implementation of our method is described in Sec. 3.4, followed by details of the training procedure in Sec. 3.5.
this section cite: []

Section: Minimizing α-Skew Jensen-Shannon Divergences
The crux of the new framework lies in minimizing a class of statistical divergences between p(x) and q θ (x) defined as
D (α) JSD (q θ , p) := 1 α D KL (q θ ∥ αp + (1 -α)q θ ) + 1 1 -α D KL (p ∥ αp + (1 -α)q θ )
for some α ∈ (0, 1), which we call the α-skew Jensen-Shannon divergence (α-JSD) (Nielsen, 2010). This divergence belongs to f -divergences (Csiszár et al., 2004).
Interestingly, α-skew JSD naturally interpolates between the forward Kullback-Leibler divergence (KLD) D KL (p ∥ q θ ) (when α → 0), the standard definition of JSD (when α = 1 2 ), and the reverse KLD D KL (q θ ∥ p) (when α → 1). In contrast to the forward KLD and reverse KLD, the α-skew JSD with α ∈ (0, 1) is well-defined even when there is a support mismatch in p and q θ , which may be the case especially in the beginning of training.
Feature 1: Multi-Divergence Training. Hence, we propose to minimize a weighted sum of the α-JSD's for different α's, as divergences with different α's exploit different geometries between two distributions. For example, it is known that minimizing the forward and reverse KLD leads to mode-covering and mode-seeking behaviors, respectively, and we can enforce better support matching behavior by considering the entire range of α.
To minimize this family of divergences in practice, we consider its gradient expression:
Proposition 3.1. Suppose that E q θ (x) [∇ θ log q θ (x)] = 0. 3
Then, we have
∇ θ D (α) JSD (q θ , p) (4) = 1 α E q(z) ∇ θ g θ (z)(s θ;0 (x) -s θ;α (x)) x=g θ (z) ,
where we define the score of the mixture distribution
s θ;α (x) := ∇ x log(αp(x) + (1 -α)q θ (x)).
This proposition suggests that we can update the generator g θ (z) using this gradient expression, provided that we can estimate the score of the mixture distribution s θ;α (x).
Feature 2: Amortized Score Model. To implement this idea, in this paper, we propose to use an amortized score model (x, α) → s ψ (x; α), to approximate the score of mixture s θ;α (x). Through our experiments we show that learning the scores of mixture over different α's using a single model is effective and helps training. In Sec. 3.3, we explain how we can train the amortized score model (x, α) → s ψ (x; α) using samples from p(x) and q θ (x).
this section cite: ['b40', 'b4']

Section: Learning with Multiple Noise Levels
To achieve stable training, we opt to minimize the divergence at different noise levels by considering the convolved distributions, p t := p * N (0, σ 2 t I D ) and q θ,t := q θ * N (0, σ 2 t I D ). This idea is widely used in the existing distillation methods. We borrow the variance-exploding Gaussian noising process notation from Karras et al. (2022b) where σ t ∈ [σ min , σ max ]. As we also integrate over different α's, the final objective becomes
L gen (θ) := E p(α)p(t) [D (α) JSD (q θ,t , p t )],(5)
where we will prescribe the choice of p(α) in Sec. 3.5. Similar to Eq. ( 4), the gradient of the divergence at noise level t can be approximated via the amortized score as
∇ θ D (α) JSD (q θ,t , p t ) ≈ γ ψ (θ; α, t)(6)
:= E q(z)q(ϵ) ∇ θ g θ (z) s ψ (x t ; 0, t) -s ψ (x t ; α, t) α x=g θ (z) ,
where the amortized score model s ψ (x t ; α, t), which is conditioned on the noise level t, is an estimate of s θ;α,t (x t ) := ∇ xt log(αp(x t ) + (1α)q θ (x t )). We provide a practical implementation of the amortized score model as a small modification of a diffusion model architecture in Sec. 3.4. We remark in passing that this expression can be understood as a generalization of the gradient update of Eq. ( 3) used in the existing reverse-KLD-based distillation schemes.
Finally, we can then approximate the generator gradient as
∇ θ L gen (θ) ≈ E p(α)p(t) [γ ψ (θ; α, t)].
Importantly, similar to existing distillation methods, the gradient only involves the output of the score model, but not its gradient. This is beneficial since such extra gradient information requires expensive backpropagation through the score model to the generator (Zhou et al., 2024).
this section cite: ['b60']

Section: Estimating Score of Mixture Distributions
Estimating the score of the mixture distribution turns out to be as simple as minimizing a mixture of the score matching losses, as stated in the following proposition:
Proposition 3.2. For any α ∈ [0, 1], the minimizer of the objective function
L(ψ; α) = α E p(x) [∥s ψ (x; α) -s p (x)∥ 2 ] +(1 -α) E q θ (x) [∥s ψ (x; α) -s q θ (x)∥ 2 ](7)
satisfies s ψ * (x; α) = s θ;α (x).
Since we train with multiple noise levels, we are interested in the marginal score of x t = x + σ t ϵ, ϵ ∼ N (0, I) at some noise level σ t . We can use denoising score matching (Vincent, 2011) to define an equivalent sample-only objective to learn the score using Tweedie's formula. Namely, to approximate s θ;α,t (x) using the amortized score model s ψ (x; α, t), we can minimize
L score (ψ) := E p(α)p(t) [L score (ψ; α, t)],
where
L score (ψ; α, t)(8)
:= α E p(x)q(ϵ) [∥s ψ (x t ; α, t) + ϵ/σ t ∥ 2 ] + (1 -α) E q θ (x)q(ϵ) [∥s ψ (x t ; α, t) + ϵ/σ t ∥ 2 ].
See Proposition A.1 for a formal statement. In practice, we parametrize the score model in the form of a denoiser and reconstruct the score from the denoiser output via Tweedie's formula; see Appendix C.1.
Feature 3: Leveraging Real and Fake Samples. We remark that our score learning objective seamlessly utilizes both real and fake samples throughout the training, helping the generator better generalize. This is in contrast to some existing diffusion distillation methods, which introduce expensive regularizers to integrate real samples, or backpropagate through the pretrained score model (Yin et al., 2024b;a;Salimans et al., 2024).
this section cite: ['b55', 'b59', 'b45']

Section: Practical Design of Amortized Score Network
With an additional conditioning scheme to embed auxiliary information about α in addition to the noise level σ t , any existing diffusion model backbone can be used to parameterize the amortized score network s ψ (x; α, t). Here, we describe how we can modify the popular UNet-based score architectures (Song et al., 2020;Nichol & Dhariwal, 2021;Karras et al., 2022b) with minimal modifications.
First, drawing from the noise embedding sensitivity analysis by Song & Dhariwal (2024b), we opt for a Fourier embedding c α with a default scale of 16. This choice ensures that the embedding is sufficiently sensitive to fluctuations in α, particularly during the early stage of training.
Then, we concatenate the α-embedding with the embedding of other auxiliary information (e.g., t and labels) and apply a single SiLU (Elfwing et al., 2018) activated linear layer:
c out = silu(W aux c aux + W α c α ).
The rationale behind this choice is as follows: as training progresses, the real and fake distributions begin to overlap, making it natural for the amortized score model to become less sensitive to α. Thanks to the additional linear layer W α after the α-embedding c α , this behavior can be realized when W α ≈ 0, when necessary.
this section cite: ['b52', 'b50', 'b7']

Section: Training
Alternating Training. Our training scheme alternates between the score estimation with the score matching objective in Eq. ( 8), and the generator training with Eq. ( 6), where we plug-in s ψ (x t ; α, t) in place of s θ;α,t (x t ). This is similar in spirit to GAN training, but the DSM technique in our framework in place of the discriminator training naturally stabilizes training. The overall training framework is summarized in Fig. 1 and Alg. 1 in Appendix C.
this section cite: []

Section: Initialization.
We warm up the generator with a standard denoising task as in diffusion models for several steps to better initialize the weights, as we empirically found that initializing the generator with pretrained weights from a denoiser significantly accelerated convergence. The amortized score network is randomly initialized.
this section cite: []

Section: Choice of p(α).
The choice of p(α) is crucial in our framework. To train both the generator and score model, we sample α from a uniform distribution over 1000 equally spaced points in [0, 1], ensuring a dense enough grid to generalize to any α. For score training, we further ensure that 25% of the sampled α's are zero, since this is always used in our gradient update; see Eq. ( 6).
Adaptive Weighting. In practice we compute the gradient with an adaptive weight w(x t , x, α, t) to ensure that the scale of the gradient for each minibatch sample is roughly uniform for different values of α and t. Hence, we modify the generator gradient in Eq. ( 6) as
γ w ψ (θ; α, t) := E q(z) ∇ θ g θ (z)× (9
) w(x t , x, α, t) s ψ (x t ; 0, t) -s ψ (x t ; α, t) α x=g θ (z) ,
where the weighting is defined as w(x t , x, α, t) := w α (x t , t)w DMD (x t , x, t).
Here w DMD is the adaptive noise weighting introduced by (Yin et al., 2024b) (see Eq. (32) in Appendix B) and w α (x t , t) is a new weighting inspired by the pseudo-Huber norm (Song & Dhariwal, 2024a; Geng et al., 2025) w α (x t , t) := α ∥s ψ (x t ; 0, t)s ψ (x t ; 1, t)∥ 2 ∥s ψ (x t ; 0, t)s ψ (x t ; α, t)∥ 2 .
This weighting still preserves the limiting forward KLD behavior of the objective as α → 0 and simplifies to DMD gradient when α = 1. We empirically show the efficacy of our adaptive weighting term w α (x t , t) through ablation studies on the CIFAR-10 dataset in Sec. 5.3; see Fig. 2b.
Regularization with GAN. We empirically found that a GAN-type regularization can accelerate convergence even further in the beginning of training. More concretely, we can train the discriminator ℓ ψ (x t ; t) ≈ log p(x) q θ (x) by the GAN discriminator training in Eq. ( 1). In our implementation, we opt to train a discriminator using a variant based on the α-JSD, as described in Appendix D.2. Given a discriminator ℓ ψ (x t ; t), we minimize a non-saturating version of the α-JSD loss (cf. Eq. ( 2)),
L (α,t) GAN (θ) = E q θ (xt) sp -ℓ ψ (x t ; t) -log α 1 -α . (11
)
The derivation can be found in Appendix D.2. Similar to Yin et al. (2024a), we parameterized the discriminator by a stack of convolution layers, applied on top of an intermediate feature of the amortized score network at α = 1/2.
Similar to DMD2 (Yin et al., 2024a), we implement a GAN discriminator building on top of the score network, with only a few additional MLP layers. This score-model-dependent design allows the full model to benefit from the training stability provided by denoising score matching, while the GAN discriminator loss only trains the small auxiliary MLP.
(For ImageNet, the generator has 296M parameters and the discriminator has 18M.) Thus, the discriminator represents a small fraction of the overall model size and has a negligible impact on training speed. As a result, our use of the GAN regularizer is both efficient and stable.
this section cite: ['b59']

Section: Distilling from Pretrained Diffusion Model
In our development so far, we do not assume access to a pretrained diffusion model. In this section, we show how a practitioner can train a one-step generative model leveraging a pretrained diffusion model, if available, within our framework. The proposed distillation scheme is comparable or even outperforms the state-of-the-art distillation schemes.
this section cite: []

Section: How To Leverage Pretrained Diffusion Model
In the distillation setup, we treat the pretrained diffusion model as the data score s p (x t ; t), and thus training the score of mixture s θ;α (x t ; t) using a single, amortized model may not be the most efficient parameterization. Hence, instead, we consider the following expression
s θ;α (x) = D θ;α (x)s p (x) + (1 -D θ;α (x))s q θ (x),
m h K n g G h z n 2 1 p Z X V v f 2 C x t l b d 3 d v f 2 7 c p B W y e Z o q x F E 5 G o b k g 0 E 1 y y F n A Q r J s q R u J Q s E 4 4 u i r 8 z j 1 T m i f y D s Y p 8 2 M y k D z i l I C R A r v i x Q S G Y Z Q P J o E H Q w Y k s K t O z Z k C L x N 3 T q p o j m Z g f 3 n 9 h G Y x k 0 A F 0 b r n O i n 4 O V H A q W C T s p d p l h I 6 I g P W M 1 S S m G k / n 0 a f 4 B O j 9 H G U K D M S 8 F T 9 f Z G T W O t x H J r N I q h e 9 A r x P 6 + X Q X T p 5 1 y m G T B J Z w 9 F m c C Q 4 K I H 3 O e K U R B j Q w h V 3 G T F d E g U o W D a K p s S 3 M U v L 5 P 2 W c 2 t 1 + q 3 5 9 X G 9 d O s j h I 6 Q s f o F L n o A j X Q D W q i F q L o A T 2 j V / R m P V o v 1 r v 1 M V t d s e Y V
v r G + b m V k O G s c C k j k M W i p a H J G E 0 I H V F F S O t S B D E P U a a 3 u A s 8 5 u 3 R E g a B j d q G B G X o 1 5 A f Y q R 0 l L H P H A 4 U n 3 P T + 5 S 6 E j K 4 a j H i C V X a c k 6 h D / 2 Z b r f M Y t W 2 R o B z h J 7 Q o p g g l r H / H S 6 I Y 4 5 C R R m S M q 2 b U X K T Z B Q F D O S F p x Y k g j h A e q R
L S g N B E E V 7 f B t f U Z d u G o M Q N 2 F G N L o R B E F c K h g j Z G L o 6 d Q k
c i j B O E k A 8 G + Y m k G N F e V r Q p F H C U X W M Y V 8 L s S n m b K c b R J J
o z I T j j J / 8 1 l z s l p 1 w q n + 8 W j k 7 u B 3 H M k Q 2 y S Y r E I f v k i J y S M 1 I h n D y Q J / J C X q 1 H 6 9 l 6 s 9 4 H r R P W M M J 1 8 k v W 5 z e e 8 K P Y < / l a t e x i t >
x fake = g ✓ (z)
< l a t e x i t s h a 1 _ b a s e 6 4 = " j M 0 y M 5 l 2 h u N t / V B w r r y k V c q n 4 4 k = " > A A A C A H i c b V D L S s N A F J 3 U V 6 2 v q A s X b g a L 4 K o k I t V l Q R C X F e w D m h g m 0 0 k 7 d P J g 5 k Y s I S D + i h s X i r j 1 M 9 z 5 N 0 7 a L r T 1 w I X D O f d y 7 z 1 + I r g C y / o 2 S k v L K 6 t r 5 f X K x u b W 9 o 6 5 u 9 d W c S o p a 9 F Y x L L r E 8 U E j 1 g L O A j W T S Q j o S 9 Y x x 9 d F n 7 n n k n F 4 + g W x g l z Q z K I e M A p A S 1 5 5 o E T E h j 6 Q f a Q e 3 C X O S r A A R m x 3 D O r V s 2 a A C 8 S e 0 a q a I a m Z 3 4 5 / Z i m I Y u A C q J U z 7 Y S c D M i g V P B 8 o q T K p Y Q O i I D 1 t M 0 I i F T b j Z 5 I M f H W u n j I J a 6 I s A T 9      6). Bottom: The amortized score model is updated by computing the score of the mixture distribution on both fake and real noisy samples, and then updating the weights using the gradient in Eq. ( 8).
f d E R k K l x q G v O 4 t z 1 b x X i P 9 5 v R S C C z f j U Z I C i + h 0 U Z A K D D E u 0 s B 9 L h k F M d a E U M n 1 r Z g O i S Q U d G Y V H Y I 9 / / I i a Z / W 7 H q t f n N W b V w 9 T u M o o 0 N 0 h E 6 Q j c 5 R A 1 2 j J m o h i n L 0 j F 7 R m / F k v B j v x
h W G D G O z v U c u w s c X E = " > A A A C G H i c b V D L S g M x F M 3 U V 6 2 v U Z d u g k W o I H W m S H V Z E E Q 3 U s E + o K 0 l k
v c 2 W 3 J O B W E N k n M Y 9 H x s a S c R b S p m O K 0 k w i K Q 5 / T t j + 8 K P z 2 H R W S x d G N G i X U D X E / Y g E j W G n J M / e d E K u B H 2 Q y 9 5 x E s t v M k Q G i 9 y r 3 z I p V t c Z A 8 8 S e k g p M 0 f D M L 6 c X k z S k k S I c S 9 m 1 r U S 5 G R a K E U 7 z s p N K m m A y x H 3 a 1 T T C I Z V u N n 4 h R 0 d a 6 a E g F r o i h c b q 7 4 k M h 1 K O Q l 9 3 F g f L W a 8 Q / / O 6 q Q r O 3 Y x F S a p o R C a L g p Q j F a M i D 9 R j g h L F R 5 p g I p i + F Z E B F p g o n V p Z h 2 D P v j x P W i d V u 1 a t X Z 9 W 6 p e P k z h K c A C H c A w 2 n E E d r q A B T S D w A M / w C m / G k / F i v B s f k 9 Y F Y x r h H v
h b X 1 j c 6 u 4 X d r Z 3 d s / K B 8 e t X S S K k K b J O G J 6 o S o K W e C N g 0 z n H a k o h i H n L b D 8 U 3 u t x + p 0 i w R D 2 Y i a R D j U L C I E T R W a v e Q y x G W + u W K V / V m c F e J v y A V W K D R L 3 / 1 B g l J Y y o M 4 a h 1 1 / e k C T J U h h F O p 6 V e q q l E M s Y h 7 V o q M K Y 6 y G b n T t 0 z q w z c K F G 2 h H F n 6 u + J D G O t J 3 F o O 2 M 0 I 7 3 s 5 e J / X j c 1 0 X W Q M S F T Q w W Z L 4 p S 7 p r E z X 9 3 B 0 x R Y v j E E i S K 2 V t d M k K F x N i E 8 h D 8 5 Z d X S e u i 6 t e q t f v L S v 3 2 a R 5 H E U 7 g F M 7 B h y u o w x 0 0 o A k E x v A M r / D m S O f F e X c + 5 q 0 F Z x H h M f y B 8 / k D 7 L + P u w = = < / l a t e x i t > ↵ < l a t e x i t s h a 1 _ b a s e 6 4 = " i P N W W N J l U a 3 h W G D G O z v U c u w s c X E = " > A A A C G H i c b V D L S g M x F M 3 U V 6 2 v U Z d u g k W o I H W m S H V Z E E Q 3 U s E + o K 0 l k 2 b
g n j w o G B V a G q Z b D d 2 6 W Y T d i d i C Q V / h h f / i h c P i n g S P P h v 3 L Q 9 + P V g 4 P H e D D P z w l Q K g 5 7 3 6 Y y M j o 1 P T E 5 N l 2 Z m 5 + Y X 3 M W l c 5 N k m v E a S 2 S i L 0 M w X A r F a y h Q 8 s t U c 4 h D y S / C z k H h X 9 x w b U S i z r C b 8 k Y M 1 0 p E g g F a q e l W A w W h h G a Q G k G D G L D N Q O b H v a s 8 M B H l t 9 h b L 6 w 9 G o B M 2 7 B J c a P p l r 2 K 1 w f 9 S / w h K Z M h T p r u e 9 B K W B Z z h U y C M X X f S 7 G R g 0 b B J O + V g s z w F F g H r n n d U g U x N 4 2 8 / 1 u P r l m l R a N E 2 1 J I + + r 3 i R x i Y 7 p x a D u L 6 8 1 v r x D / 8 + o Z R r u N X K g 0 Q 6 7 Y Y F G U S Y o J L Y K i L
where
D θ;α (x) := αp(x) αp(x) + (1 -α)q θ (x) = σ log p(x) q θ (x) + log α 1 -α .
See Proposition A.2 for a formal statement. In words, we can express the score of mixture s θ;α (x) as a mixture of scores s p and s q θ , where the weight is (D θ;α (x), 1 -D θ;α (x)). This suggests that instead of an amortized modeling of the score of mixture, we can use an alternative parameterization,
s exp ψ (x; α) := D ψ (x; α)s p (x) + (1 -D ψ (x; α))s fake ψ (x),where
D ψ (x; α) := σ ℓ ψ (x) + log α 1 -α .
Here, we can parameterize the discriminator x → ℓ ψ (x) in the same way as we do for the GAN discriminator.
We can extend this to multiple noise levels easily. Hence, an alternative parameterization for s θ;α (x t ; t) is
s exp ψ (x t ; α, t) := D ψ (x t ; α, t)s p (x t ; t)(12)
+ (1 -D ψ (x t ; α, t))s fake ψ (x t ; t), where
D ψ (x t ; α, t) := σ ℓ ψ (x t ; t) + log α 1 -α .(13)
Plugging this explicit score model into Eq. ( 8), we can learn both the fake score model s fake ψ and the discriminator ℓ ψ at different noise levels.
Corollary 4.1. Let α ∈ [0, 1] be fixed and σ t be some fixed noise level. Then, the minimizer of the objective function
L exp (ψ; α, t)(14)
:= α E p(x)q(ϵ) [∥s exp ψ (x t ; α, t) + ϵ/σ t ∥ 2 ] + (1 -α) E q θ (x)q(ϵ) [∥s exp ψ (x t ; α, t) + ϵ/σ t ∥ 2 ] satisfies s fake ψ * (x; t) = s q θ (x; t) and ℓ ψ * (x t ; t) = log p(x t ) q θ (x t )
.
We remark that this new regression objective in Eq. ( 14) provides a new way to compute the log density ratio, as an alternative to the GAN training (see Eq. ( 1)). In Appendix D.3, we establish a connection between this objective for training a discriminator to an existing GAN discriminator objective in the literature.
With this new, explicit parameterization, we can approximate the gradient expression in Eq. ( 6) as
∇ θ D (α) JSD (q θ,t , p t ) ≈ γ exp ψ (θ; α, t)(15)
:= E q(z) D ψ (x t ; α, t)× ∇ θ g θ (z) s fake ψ (x t ; t) -s p (x t , t) α x=g θ (z)
.
this section cite: []

Section: Implementation and Training
Model Architectures. We can leverage any existing diffusion model architectures directly for the fake score s fake ψ (x t ; t). We parametrize the discriminator ℓ ψ (x t ; t) similar to the noise-conditional discriminator in our training from scratch setting (see Sec. 3.5). The difference is that we can train the discriminator by minimizing the DSM loss in Eq. ( 14) naturally, without an additional GAN loss. When training the generator, we plug in this approximate log density ratio into Eq. ( 11) to regularize the generator updates.
Training. We also train in an alternating fashion. Since we have access to a pretrained score model, we use this to initialize the weights of both the generator and the fake score model. We utilize the same sampling distribution for α as in our training from scratch setup (see Sec. 3.5). The procedure is summarized in Fig. 4 and Alg. 2 in Appendix C.
this section cite: []

Section: Experiments
In this section, we first present results on the ImageNet 64 × 64 dataset. We then demonstrate the competitiveness of our method on the CIFAR-10 dataset and conduct a series of ablation studies. We measure performance through sample quality as measured by the Fréchet Inception Distance (FID) (Heusel et al., 2017). The exact hyerparameters, training configurations used and additional results, including an example training dynamics and latent interpolation, can be found in Appendix E. Our implementation can be found at https://github.com/tkj516/   score-of-mixture-training.
this section cite: ['b11']

Section: Class-conditional ImageNet 64x64 Generation
Experimental Setup. We trained class-conditional onestep generative models on ImageNet 64 × 64 (Deng et al., 2009), experimenting with both distillation and training from scratch. In both cases, we used the ADM architecture (Nichol & Dhariwal, 2021) as the base score model architecture, and the discriminator ℓ ψ (x t ; t) was implemented as a stack of convolution layers operating on the bottleneck feature from the score network, similar to DMD2 (Yin et al., 2024a). For training from scratch, we augmented the score architecture using an α-embedding as described in Sec. 3.4. The total number of parameters of the amortized score model remained unchanged otherwise. As a warmup stage, we pretrained the generator on the dataset using a standard diffusion denoising objective for 40k steps to initialize the weights. For distillation, we used a pretrained diffusion model from (Karras et al., 2022b).
Results. We evaluated our method against several published baselines for both training from scratch and distillation. As shown in Table 2, when trained from scratch, our generator with 296M parameters outperforms both consistency training and its improved variant (Song et al., 2023;Song & Dhariwal, 2024a), with a much smaller training budget (200k iterations with batch size of 40 vs. 800k iterations with batch size of 512). Our model also competes favorably with iCT-deep, despite using a generator with half the number of parameters: FID of 3.23 with 296M parameters (ours) vs. 3.25 with 592M parameters (iCT-deep). We observed stable training throughout, without requiring extensive hyperparameter tuning or special noise schedule adjustments as in consistency training, as visualized in Fig. 2a. We also surpass the ECT model (Geng et al., 2025) of similar size and training budget that includes several modifications to induce stability in consistency training. Samples generated using our method can be found in Fig. 3 and Appendix E.
In the distillation setting, our model achieves a competitive FID of 1.48, outperforming several baselines. Notably, we outperform consistency distillation methods, such as multistep consistency distillation (Heek et al., 2024), despite using only a fraction of the model size (256M parameters against 1200M parameters). Our model also surpasses consistency trajectory models (CTM) (Kim et al., 2024), which ensure consistency between random points along the PF ODE trajectory, by simulating the reverse diffusion sampler for an arbitrary number of steps per minibatch, thus resulting in high computational cost.
We also outperform reverse-KLD methods with similar compute or regularizers such as DMD (Yin et al., 2024b) and DMD2 (Yin et al., 2024a) with FIDs of 5.60 and 1.51 respectively. We note that on spending significant extra compute, DMD and DMD2 achieved improved results. For example, in the DMD framework without any GAN regularization, the authors simulate the reverse process of a diffusion model and sample several thousand noise-image pairs to anchor the generator's outputs. Each noise-image pair requires evaluating the diffusion denoiser 256 times for ImageNet 64×64, which is extremely costly in practice. In contrast in the DMD2 framework the authors adopt a lengthy finetuning stage with GAN regularization of 400k steps to further improve results.
We did not resort to any of the above techniques and sought to find an approach that worked best with a single execution of the training pipeline run for 200k steps.
this section cite: ['b5', 'b54', 'b8', 'b10', 'b19', 'b59']

Section: Unconditional CIFAR-10 Generation
Experimental Setup. We evaluated our method on the CIFAR-10 dataset (Krizhevsky et al., 2009) for unconditional one-step generative modeling, considering both training from scratch and distillation. In both cases, we employed a DDPM++ architecture (Song et al., 2020) with EDM preconditioning (Karras et al., 2022b). The discriminator again followed the convolutional stack used in DMD2. For training from scratch, we modified the score model to incorporate the α-embedding (Sec. 3.4) while maintaining a similar network size. To mitigate overfitting due to the dataset's small size, we enabled dropout with p = 0.13,   as in EDM. In the distillation setting, we initialized the generator with a pretrained unconditional diffusion model from (Karras et al., 2022b), using the same UNet backbone and weights. Distillation performed well without dropout.
Results. The last three columns in Table 2 highlight the performance of our method on CIFAR-10 compared to various baselines. In our training from scratch setting, despite utilizing a lower training budget (150k steps with a batch size of 40) than many methods, our approach remains highly competitive. In terms of training budget, the most comparable baseline is ECT, which we are able to outperform without requiring excessive design considerations and hyperparameter tuning. Our distillation results are also competitive. In particular, we outperform Diff-Instruct and DMD2, which are only based on minimizing the reverse KLD. This corroborates the benefit of our multi-divergence minimization approach. Image samples can be found in Appendix E.5.
this section cite: ['b24', 'b52']

Section: Ablation Studies
We use the CIFAR-10 dataset to study the effectiveness of the design choices that we have proposed; see Fig. 2b.
this section cite: []

Section: Choice of Adaptive Gradient Weighting.
Starting with our base objective without the GAN regularizer, we tested our (α, t)-adaptive weighting in Eq. ( 10). Fig. 2b demonstrates the benefits of our weighting scheme, compared to the DMD weight function that only depends on t.
Learning with Single vs. Multiple α's. The α-JSD reduces to the reverse KLD of DMD and other distillation methods, when α = 1. To test the efficacy with multi-α learning, we implemented an amortized variant, training the score model only with α ∈ {0, 1}. Results show that conditioning on a range of α-values not only minimizes multiple divergences but also strengthens the α embedding as a conditioning signal thereby facilitating more accurate divergence minimization.
Accelerated Convergence with GAN Regularizer. We finally verify the benefits of our novel GAN-type regularizer for α-JSD minimization. As demonstrated by the second and fourth curves in Fig. 2b, the GAN regularizer helps accelerate convergence especially in the beginning of training.
this section cite: []

Section: Concluding Remarks
In this paper, we show that high-quality one-step generative models can be trained from scratch and in a stable manner, without simulating the reverse diffusion process or probability flow ODE as in diffusion models and consistency models. The key distinctive idea in our framework is a new multi-divergence minimization paradigm implemented by estimating the score of mixture distributions. For stable training, we borrow multi-level noise learning and denoising score matching techniques from the diffusion literature. Our empirical results show that accurate score estimation facilitates stable minimization of statistical divergences. We hope this work offers a fresh perspective on generative modeling and inspires further research in the field.
Limitations and Future Work. While SMT/SMD achieve strong empirical performance, there is still room for improvement in both architecture and training strategies. Despite achieving highly competitive FID scores for onestep generation from scratch, models capable of few-step generation-such as consistency models-might further improve FID with additional iterations.Finally, given the generality of our framework, we believe these ideas could extend to other complex modalities, including speech and audio synthesis. We leave such directions for future work.
this section cite: []

Section: Impact Statement
We introduce Score-of-Mixture Training, a simple yet effective one-step generative modeling framework that requires minimal design effort and hyperparameter tuning. We hope its ease of implementation will drive further research into efficient, state-of-the-art neural sampling. However, we acknowledge the potential risks of misuse, including the generation of fake, biased, or misleading content. Our work focuses on fundamental research using standard machine learning datasets, but we recognize the importance of ensuring generative models are secure and privacy-preserving to democratize this technology responsibly.
D On GAN Training 20 D.1 On the Non-Saturating Generative Loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20 D.2 GAN-Type Regularization with α-JSD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21 D.3 On Discriminator Training with Mixture Score Matching Loss . . . . . . . . . . . . . . . . . . . . . . . 21 E More on Experiments and Additional Results 22 E.1 Training Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22 E.2 Toy Swiss Roll . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22 E.3 Image Interpolation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23 E.4 Additional Training Curves . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25 E.5 Samples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
this section cite: []

Section: A. Deferred Statements and Proofs

this section cite: []

Section: A.1. Proof of Proposition 3.1
Proof of Proposition 3.1. We can simplify the gradient of each term separately as follows:
∇ θ D KL (q θ ∥αp + (1 -α)q θ ) = E q θ (x) ∇ θ log q θ (x) αp(x) + (1 -α)q θ (x) + E q(z) ∇ θ g θ (z)(s θ;0 (x) -s θ;α (x)) x=g θ (z) , ∇ θ D KL (p∥αp + (1 -α)q θ ) = -E p(x) [∇ θ log(αp(x) + (1 -α)q θ (x))] .
Here, note that in the first expression, we invoke the chain rule: for some function f θ : X → R, we have
∇ θ f θ (g θ (z)) = (∇ θ f θ (x))| x=g θ (z) + ∇ θ g θ (z)(∇ x f θ (x))| x=g θ (z) .
Combining these two terms with the weights, we get the gradient of the α-skew JSD:
∇ θ D (α) JSD (q θ , p) = 1 α ∇ θ D KL (q θ ∥αp + (1 -α)q θ ) + 1 1 -α ∇ θ D KL (p∥αp + (1 -α)q θ ) = 1 α E q(z) ∇ θ g θ (z)(s θ;0 (x) -s θ;α (x)) x=g θ (z) - 1 α(1 -α) E αp(x)+(1-α)q θ (x) [∇ θ log(αp(x) + (1 -α)q θ (x))] + 1 α E q θ (x) [∇ θ log q θ (x)] = 1 α E q(z) ∇ θ g θ (z)(s θ;0 (x) -s θ;α (x)) x=g θ (z)
.
Here, we use the assumption that E q θ (x) [∇ θ log q θ (x)] = 0.
this section cite: []

Section: A.2. Proof of Proposition 3.2
Proof of Proposition 3.2. We can write the objective L(ψ; α) as
L(ψ; α) = (αp(x) + (1 -α)q θ (x))∥s ψ (x; α)∥ 2 -2(αp(x)s p (x) + (1 -α)q θ (x)) ⊺ s ψ (x; α) dx + C = (αp(x) + (1 -α)q θ (x)) s ψ (x; α) - αp(x)s p (x) + (1 -α)q θ (x)s q θ (x) αp(x) + (1 -α)q θ (x) 2 dx + C ′ .
Hence, it is clear that the global minimizer should be
s ψ * (x; α) = αp(x)s p (x) + (1 -α)q θ (x)s q θ (x) αp(x) + (1 -α)q θ (x) = α∇ x p(x) + (1 -α)∇ x q θ (x) αp(x) + (1 -α)q θ (x) = ∇ x (αp(x) + (1 -α)q θ (x)) αp(x) + (1 -α)q θ (x) = ∇ x log(αp(x) + (1 -α)q θ (x)).
this section cite: []

Section: A.3. Deferred Statements
Proposition A.1. Let α ∈ [0, 1] be fixed and σ t be some fixed noise level. Then, the minimizer of the objective function
L score (ψ; α, t) := α E p(x)q(ϵ) [∥s ψ (x t ; α, t) + ϵ/σ t ∥ 2 ] + (1 -α) E q θ (x)q(ϵ) [∥s ψ (x t ; α, t) + ϵ/σ t ∥ 2 ] (16
)
satisfies s ψ * (x t ; α, t) = s θ;α,t (x t ).
Proof. We can write the objective L(ψ; α, t) as
L score (ψ; α, t) = (αp(x t ) + (1 -α)q θ (x t )) s ψ (x t ; α, t) + ϵ σ t 2 dx dϵ.
This is a standard minimum mean square estimation (MMSE) problem for which the global minimizer is the conditional mean,
s ψ * (x t ; α, t) = - 1 σ t E αpt+(1-α)q θ,t [ϵ|x t ] = - 1 σ 2 t E αpt+(1-α)q θ,t [x t -x|x t ] = - 1 σ 2 t x t + 1 σ 2 t E αpt+(1-α)q θ,t [x|x t ] = ∇ xt log(αp(x t ) + (1 -α)q θ (x t )).
Here we use that x t = x + σ t ϵ and make the connection to the marginal score in the last line using Tweedie's formula (Robbins, 1956).
Proposition A.2. Let α ∈ [0, 1], s p (x)
be the data score, s q θ (x) be the score of the generated samples. Then, the score of the mixture distribution can be expressed as
s θ;α (x) = D θ;α (x)s p (x) + (1 -D θ;α (x))s q θ (x),(17)
where
D θ;α (x) := σ log p(x) q θ (x) + log α 1 -α ,(18)
Proof. The amortized score can be expressed as
∇ x log(αp(x) + (1 -α)q θ (x)) = ∇ x (αp(x) + (1 -α)q θ (x)) αp(x) + (1 -α)q θ (x) = αp(x) αp(x) + (1 -α)q θ (x) ∇ x log p(x) + (1 -α)q θ (x) αp(x) + (1 -α)q θ (x) ∇ x log q θ (x) = D(x; α)∇ x log p(x) + (1 -D(x; α))∇ x log q θ (x).
We can now simplify the scaling factor as
D(x; α) = αp(x) αp(x) + (1 -α)q θ (x) = σ log p(x) q θ (x) + log α 1 -α .
this section cite: ['b43']

Section: B. Detailed Discussions on Related Work
B.1. Diffusion Models Prior Work. Sohl-Dickstein et al. (2015) first introduced diffusion probabilistic models (DPMs) as deep variational autoencoders (Kingma, 2014) based on the principles of thermodynamic diffusion with a Markov-chain variational posterior that maximizes the evidence lower bound (ELBO). Several years later, Ho et al. (2020) re-introduced DPMs (DDPMs) with modern neural network architectures and a simplified loss function that set a new state-of-the-art in image generation. Since then, numerous connections to existing literature in statistics, information theory and stochastic differential equations (SDEs) have helped bolster the quality of these models. For example, Song & Ermon (2019) illustrate the equivalence between DDPMs and DSM at multiple noise levels, thus bridging the areas of diffusion-based models and score-based models. Subsequently, Song et al. (2021b) showed that in continuous time, DPMs can be appropriately interpreted as solving for the reverse of a noising process that evolves as an SDE while Kingma et al. (2021) demonstrated that continuous-time DPMs can interpreted as VAEs and that the variational lower bound is invariant to the noise schedule except for its endpoints, thus bolstering its density estimation capabilities. Following the latter discovery, Kong et al. (2023) show that DPMs can in-fact be used for exact likelihood computation by leveraging techniques from information theory. To further improve DPMs, extensive research has gone into the choice of noise schedules, network architectures and loss functions (Nichol & Dhariwal, 2021;Hoogeboom et al., 2023;Karras et al., 2022a;Kingma & Gao, 2024). Many tangentially discovered frameworks such as rectified flows (Liu et al., 2023) and conditional normalizing flows trained with Gaussian conditional flow matching (Lipman et al., 2023), are also particular instances of (Gaussian) diffusion models with specialized noise schedules and weighted loss functions, as show in (Kingma & Gao, 2024).
Formulation. We take the following unified view in our definition of DPMs as inspired by (Kingma & Gao, 2024) and (Karras et al., 2022a). Let p(x) be the data distribution and let λ(t) define a variance exploding noise schedule with distribution p(t) where t ∼ U(0, 1). Under this noise schedule we can define a noisy version of x at noise level σ t as
x t := x + σ t ϵ where ϵ ∼ N (0, I).
Given noisy samples of data, the diffusion objective can be reduced to a weighted denoising objective,
L DPM (ϵ θ ) = 1 2 E p(t)p(x)q(ϵ) w(t)∥ϵ -ϵ θ (x t ; t)∥ 2 ,(20)
where w(t) is a positive scalar-valued weighting function. Note that for the forward process defined in Eq. ( 19), the conditional score is s(x t |x) = -ϵ/σ t . Thus, Eq. ( 20) can be interpreted as a weighted denoising score matching loss (Vincent, 2011) over multiple noise levels,
L DPM (ϵ θ ) = 1 2 E p(t)p(x)p(xt|x) w ′ (t) s(x t |x) + ϵ θ (x t ; t) σ t 2 , (21
)
where w ′ (t) := σ 2 t w(t) and the marginal score estimator is s θ (x t ; t) := -ϵ θ (x t ; t)/σ t . Sampling. It is often beneficial to view DPMs as SDEs (Song et al., 2021b) where the forward process can be expressed as
dx t = f (x t , t)dt + g(t)dw,
where w is a standard Wiener process and x 0 = x. The time reversal of this process (i.e., the generative process) is known to follow the reverse SDE, dx t = f (x t , t)g 2 (t)∇ xt log p(x t ) dt + g(t)d w.
Note that in practice ∇ xt log p(x t ) would be estimated by the score function s θ (x t ; t) from a variant of DSM as in Eq. ( 21).
Sampling can be simulated through techniques such as annealed Langevin dynamics or ancestral sampling (Song et al., 2021b). While the above reverse SDE is stochastic in nature, there also exists a deterministic process known as the probability flow ODE that satisfies the same intermediate marginal distributions,
dx t = f (x t , t) - 1 2 g 2 (t)∇ xt log p(x t ) dt. (22
)
The benefit of the ODE formulation is that it can discretized more coarsely and hence sampling can done in fewer timesteps. Furthermore, sampling is possible by plugging in the updates from Eq. ( 22) into black-box ODE solvers, e.g., the Heun 2 nd order solver (Karras et al., 2022a). Sampling can be sped even further if Eq. ( 22) can be solved exactly. Lu et al. (2022a) show that the exact solution to Eq. ( 22) at timestep t given an initial value at timestep s < t is,
x t = x s + 2 σt σs σ u ϵ θ (x u ; u) dσ u .(23)
Various samplers can be derived by approximating the exponentially weighted integral in different ways. For example, the widely used DDIM sampler (Song et al., 2021a) is an example of a first-order Taylor expansion of the integral term. At the core of all these algorithms is a score estimator/denoiser, which if learned accurately could improve the quality of samples produced.
this section cite: ['b53', 'b21', 'b23', 'b13', 'b29', 'b27', 'b55', 'b53', 'b53']

Section: EDM Diffusion Architecture.
The EDM preconditioning diffusion model utilizes a base DDPM++ architecture from (Song et al., 2021b) for CIFAR-10 and the ADM architecture (Nichol & Dhariwal, 2021) for ImageNet 64 × 64. The EDM model uses a noise schedule that is defined as log σ t ∼ N (-1.2, 1.2 2 ).
Rather than regressing against the unscaled additive noise as in DSM, EDM regresses against the original sample expressed in the following form,
x = σ 2 data σ 2 t + σ 2 data x t + σ t • σ data σ 2 t + σ 2 data g,(25)
where σ data = 0.5. To this end, EDM is parametrized with a denoising neural network,
f θ (x t ; t) = σ 2 data σ 2 t + σ 2 data x t + σ t • σ data σ 2 t + σ 2 data g θ (x t ; t),(26)
which is trained by minimizing
min θ E p(x)q(ϵ)p(t) [ w(t)∥x -f θ (x t ; t)∥ 2 ], where w EDM (t) = σ t σ data σ 2 t + σ 2 data . (27
)
This is equivalent to estimating g by minimizing the objective,
L EDM (g θ ) := E p(x)q(ϵ)p(t) ∥g -g θ (x t ; t) ∥ 2 . (28
)
Using Eq. ( 24) and Eq. ( 25) we can show that,
g = σ 2 t + σ 2 data σ t σ data x - σ data σ t σ 2 t + σ 2 data x t (29
) = - σ 2 t + σ 2 data σ data ϵ + σ t σ 2 t + σ 2 data σ data x t .(30)
Therefore, in terms of Eq. ( 20) the EDM objective boils down to the unified diffusion objective with weighting function,
w(t) = σ 2 t + σ 2 data σ 2 data .(31)
this section cite: ['b53']

Section: B.2. Diffusion Distillation
Achieving state-of-the-art generation results on CIFAR-10 and ImageNet 64 × 64 using a Heun 2 nd order sampler with the EDM architecture requires 35 and 512 function evaluations (FEs) respectively. The goal of diffusion distillation is to distill a teacher model into a student model that can achieve high quality signal generation with few FEs.
The earliest works on distillation such as progressive distillation (Salimans & Ho, 2022) and knowledge distillation (Huang et al., 2023) train a student diffusion model with drastically reduced sampling budget to match the performance of a teacher model that is simulated in reverse. For example, given a teacher diffusion model parametrized as a denoiser f ϕ and a noisy sample x t , a "clean" target x (k) ϕ is constructed by running the teacher model for k steps in reverse. The student denoiser f θ is then optimized by minimizing the loss,
L(ϕ) := E p(x)p(z)p(t) [w(t)∥f θ (x t ; t) -x (k) ϕ ∥ 2 ].
Knowledge distillation on the other hand conditions the student model on intermediate features from the teacher diffusion model so as to regularize the learned weights more effectively and retain knowledge from the teacher model. These methods are expensive as it requires either simulating multiple steps of a teacher diffusion model or additionally probing it for feature extraction.
More recently a class of new diffusion distillation techniques grounded in reverse KL divergence minimization have gained popularity as discussed in Sec. 2. Diff-Instruct (Luo et al., 2024a), DMD (Yin et al., 2024b) and DMD2 (Yin et al., 2024a) all train a one-step generator g θ mapping noise z ∼ N (0, I) to generated samples by updating the generator in the direction of minimizing the reverse KLD,
∇ θ D avg KL (q θ ∥p) = E q(z)p(t)q(ϵ) [∇ θ g θ (z)(s q θ (x t ) -s p (x t )) | x=g θ (z) ],
where
s p (x t ) = ∇ xt log p(x t ) and s q θ (x t ) = ∇ xt log q θ (x t ).
Assuming that the score model was learned using a parametrization similar to EDM, DMD scales the gradient and uses Tweedie's formula (Robbins, 1956) to express it in terms of a pretrained denoiser f ϕ and a denoiser for the fake samples f ψ ,
∇ θ L DMD (θ) = E q(z)p(t)q(ϵ) [w DMD (x t , x, t)∇ θ g θ (z)(f ψ (x t ; t) -f ϕ (x t ; t)) | x=g θ (z) ],
where an adaptive weight is used to ensure that the scale of the gradient is roughly uniform across noise levels,
w DMD (x t , x, t) := σ 2 t ∥x -f ϕ (x t ; t)∥ 1 .(32)
To mitigate mode collapse and enhance sample diversity, DMD employs an ODE-based regularizer by simulating the pretrained diffusion model in reverse. This process generates noise-image pairs, which are then used to further supervise the generator's training. However, collecting this dataset becomes prohibitively expensive for high-dimensional samples.
To address this limitation, DMD2 introduces a GAN-based regularizer, which effectively minimizes the Jensen-Shannon divergence alongside the reverse KLD, or a variant of the forward KLD when implemented in a non-saturating manner. For further details on GAN training, refer to Appendix D.
Several methods build upon the divergence minimization framework by introducing regularizers based on alternative statistical distance measures. For instance, Moment Matching Distillation (MMD) (Salimans et al., 2024), Score Identity Distillation (SiD) (Zhou et al., 2024), and Score Implicit Matching (SiM) (Luo et al., 2024b) align the fake score model with the pretrained score model using a variant of the Fisher divergence:
L Fisher (ψ) := E q θ (x)p(t)q(ϵ) [w ′ (t)∥f ψ (x t ; t) -sg[f ϕ (x t ; t)]∥ 2 ].
Here sg stands for the stop gradient operator. Additionally, both SiD and SiM extend this approach to generator training by minimizing the Fisher divergence, which requires a computationally expensive gradient calculation through the entire score model. To address this, they employ statistical approximations to make these gradient computations more practical.
this section cite: ['b44', 'b14', 'b59', 'b43', 'b45', 'b60']

Section: B.3. Consistency Models
Consistency models are a new class of generative models introduced by Song et al. (2023) that learn a consistency function between all points along the trajectory of the probability flow ODE of a reverse diffusion sampler. Concisely, given points along one such trajectory, x t , t ∈ [ϵ, 1], where x 1 ∼ N (0, I), the consistency function satisfies,
f (x t , t) = x if t = ϵ f (x s , s) s ∈ [ϵ, 1]
Given the boundary condition at the origin, the consistency function can be parametrized using a neural network similar to EDM ,
f θ (x t , t) = σ 2 data (σ t -σ ϵ ) 2 + σ 2 data x t + (σ t -σ ϵ ) • σ data σ 2 t + σ 2 data g θ (x t ; t).
Given a noisy sample x t = x + σ t ϵ, ϵ ∼ N (0, I), first a single step of the probability flow ODE is simulated using the Euler sampler by running one step of sampling using Eq. ( 23),
x s = x t + (t -s)t∇ xt log p(x t )
This can be computed using either a pretrained score model or via a single sample Monte-Carlo estimate. In the latter setting, it is important that the timesteps s and t are very close to each other for the approximation to hold. In consistency distillation a pretrained score model s ϕ is available and a single sampling step along the PF-ODE is simulated as
x ϕ s = x t + (t -s)ts ϕ (x t ; t). Then the consistency function is learned by minimizing L CD (θ) = E p(x)q(ϵ)p(t) [w(t)d(f θ (x t ; t), sg[f θ (x ϕ
t-∆t ; t -∆t)])], where d is some distance measure, w(t) is some positive weighting function and s = t -∆t, with ∆t some fixed timestep difference. Song et al. (2023) initially proposed using the LPIPS distance but subsequent works (Song & Dhariwal, 2024a;Geng et al., 2025) have shown that similar performance can be achieved by using the ℓ 2 distance or a pseudo-Huber norm.
Unlike distillation techniques, consistency models can also be trained from scratch. Assume that s = tδt, δt → 0. Then, the sampling step can be approximated using Tweedie's formula (Robbins, 1956),
x s ≈ x t + (t -s)
xx t t = x + sϵ.
Thus, the consistency function can now be learned by minimizing,
L CT (θ) = E p(x)q(ϵ)p(t) [w(t)d(f θ (x + tϵ; t), sg[f θ (x + (t -δt)ϵ; t -δt)])],
Consistency distillation still lags behind distillation methods based on reverse KL minimization, but consistency training often demonstrates more impressive results. However, consistency training is still inherently unstable and requires careful design of both the noise schedule due to limiting nature of δt and distance measure (Song & Dhariwal, 2024a;Geng et al., 2025). Stabilizing and making this objective simpler is the focus of a lot of current research in the area.
this section cite: ['b54', 'b54', 'b8', 'b43', 'b8']

Section: C. Detailed Description of Score-of-Mixture Training and Distillation

this section cite: []

Section: C.1. Amortized Denoiser
Modern diffusion architectures such as the EDM architecture (Karras et al., 2022b) are specially designed for denoising purposes (see Appendix B). Hence, in practice we choose to train an amortized denoiser, f ψ (x t ; α, t) ≈ E αpt+(1-α)q θ,t [x|x t ], upon which the amortized score can be recovered using Tweedie's formula (Robbins, 1956),
s ψ (x t ; α, t) = - 1 σ 2 t x t + 1 σ 2 t f ψ (x t ; α, t).
The mixture score matching loss in Eq. ( 8) can be expressed with this denoiser as
L denoise gen (ψ; α, t) := α E p(x)q(ϵ) [∥f ψ (x; α, t) -x∥ 2 ] + (1 -α) E q θ (x)q(ϵ) [∥f ψ (x; α, t) -x∥ 2 ].
this section cite: ['b43']

Section: C.2. Score-of-Mixture Training
Here, we present a pseudocode for Score-of-Mixture Training (SMT). See Algorithm 1.
If τ = 0, it boils down to the plug-in reverse KL divergence, and τ = 1 recovers the non-saturating loss. If we consider a gradient with respect to θ, we get
∇ θ E q(z) [log(τ + r ψ (g θ (z)) -1 )] = E q(z) - r ′ ψ (g θ (z)) r ψ (g θ (z))(1 + τ r ψ (g θ (z))) ∇ θ g θ (z)
Here, recall that r ψ (x) ≈ p(x) q θ (x) is supposed to be small for generated samples x = g θ (z). Therefore, the plug-in loss with τ = 0 is inherently prone to vanishing gradient.
this section cite: []

Section: D.2. GAN-Type Regularization with α-JSD
To train a discriminator for our GAN-type regularization, we opt to use a modified GAN discriminator objective defined as
min ψ -α E p(x) [log D ψ (x; α)] -(1 -α) E q θ (x) [log(1 -D ψ (x; α))].
Similar to the vanilla GAN, the optimal discriminator for each θ and α in this case is
D ⋆ ψ (x; α) = αp(x) αp(x) + (1 -α)q θ (x)
Then,the α-JSD can be approximated as
D (α) JSD (q θ , p) ≈ 1 1 -α E p(x) log D ψ (x; α) α + 1 α E q θ (x) log 1 -D ψ (x; α) 1 -α .
Hence, with this approximation, the generator update can be done via
min θ 1 α E q θ (x) log 1 -D ψ (x; α) 1 -α .
In practice, we can use a weighted non-saturating version of the loss as well,
min θ -E q θ (x) log D ψ (x; α) α = min θ E q θ (x) sp -ℓ ψ (x) -log α 1 -α + log α,
where ℓ ψ (x) = log p(x) q θ (x) .
this section cite: []

Section: D.3. On Discriminator Training with Mixture Score Matching Loss
In Sec. 4, we plugged in the explicit parameterization
s exp ψ (x; α) := D ψ (x; α)s p (x) + (1 -D ψ (x; α))s fake ψ (x),
to the mixture regression loss in Eq. ( 7), to train the fake score and the discriminator simultaneously. If we consider an ideal scenario where we have the perfect score models for both p and q, then all we need to train is the discriminator andthe mixture regression objective can be interpreted as a discriminator objective. Here we reveal its connection to an instance of f -GAN discriminator objective.
Let s p (x) and s q (x) be the underlying score functions for p and q, respectively. Then, the explicit parameterization becomes
s exp ψ (x; α) = D ψ (x; α)s p (x) + (1 -D ψ (x; α))s q (x),
and the mixture regression objective becomes only a function of the discriminator, i.e., The samples produced are shown in Fig. 5. Notice how the GAN is unable to perfectly cover the entire continuous mode of the swiss roll. The Diffusion-GAN, a multi-noise level extension of the GAN, covers the mode but also samples from areas of low density. We found the latter to be sensitive to the chosen noise levels in comparison to the methods based on updating the generator using the score.
L(ψ; α) = α E p(x) [∥s exp ψ (x; α) -s p (x)∥ 2 ] + (1 -α) E q θ (x) [∥s exp ψ (x; α) -s q (x)∥ 2 ] = α E p(x) [(1 -D ψ (x; α)) 2 ∥s p (x) -s q (x)∥ 2 ] + (1 -α) E q(x) [D ψ (x; α) 2 ∥s p (x) -s q (x)∥ 2 ] = αp(x)(1 -D ψ (x; α)) 2 + (1 -α)q(x)D ψ (x; α) 2 ∥s p (x) -s q (x)∥ 2 dx.
Our results for training from scratch and distillation are presented in Fig. 5d-f. All three methods successfully capture the modes of the underlying distribution. While the impact of the GAN regularizer is less pronounced than in our highdimensional experiments, we observe that enabling it (as in Fig. 5d) reduces the number of samples in low-density regions compared to Fig. 5e. The distillation results in Fig. 5f appear slightly noisy, likely due to the quality of the pre-trained score model. This highlights the advantage of training from scratch, as it avoids amplifying existing estimation errors in the pre-trained model.
In Fig. 5g, we present an ablation result of the α-sampler, by using only α = 1. This corresponds to minimizing reverse KLD as in DMD and DMD2. Unlike in the high-dimensional setting presented in Fig. 2b, we observe that using the single α = 1 produces visually plausible samples in this low-dimensional synthetic example. However, our method in Fig. 5d (i.e., SMT with GAN regularizer) produces fewer spurious samples compared to Fig. 5g, suggesting the benefit of multiple α values.
this section cite: []

Section: E.3. Image Interpolation
The one-step generator is a mapping between the representation space, which in this case is the space of standard multivariate Gaussian variables, and the space of images. Thus, to study the representation space we followed the approached in (Song et al., 2023) and spherically interpolated between two randomly chosen noise instances z 0 and z 1 ,
z β = sin((1 -β)ψ) sin(ψ) z 0 + sin(βψ) sin(ψ)z 1 ,
where β ∈ [0, 1] and ψ = arccos
z ⊺ 0 z1
∥z0∥2∥z1∥2 . After interpolating between these two points, the interpolated image can be obtained as x β = g θ (z β ) as shown in Figures 6 and 7.
this section cite: ['b54']

Section: E.4. Additional Training Curves
To further demonstrate the stable training dynamics of SMT, we provide additional training curves in Figure 8 showing a consistent decrease in both loss and gradient norm. Notably, at no point do we observe any spikes or instability in either metric. Since we compute the generator's gradient directly, the plotted loss in Figure 8b corresponds to the proxy objective prior to applying automatic differentiation.
this section cite: []

Section: E.5. Samples
We present some image samples generated by SMT and SMD in Figs. 9-12.
this section cite: []

Section: 
t) and α as described in Sec. 3.5 Compute weighted generator gradient γ w ψ (θ; α, t) from Eq. (9) Compute GAN regularizer loss L (α,t) GAN from Eq. (11) Update parameters: θ ← θη gen E p(α)p(t) [w(x fake t , x fake , α, t)γ w ψ (θ; α, t) + λ∇ θ L (α,t) GAN (θ)] Amortized Score Training: Freeze g θ for each sub-iteration do Sample mini-batch of real samples x real ∼ D Sample t ∼ p(t) and α Compute score matching loss L score (ψ; α, t) from Eq. (8) Compute non-saturated discriminator loss L disc (ψ) (see Appendix D) Update parameters: ψ ← ψη score ∇ ψ (E p(α)p(t) [L score (ψ; α, t)] + µL disc (ψ))
end for end for Return: Trained model parameters θ, ψ
this section cite: []

Section: C.3. Score-of-Mixture Distillation
We present an overview figure and pseudocode of Score-of-Mixture Distillation in Fig. 4 and Alg. 2. We highlight the central differences in the distillation training from the training from scratch in Alg. 1.
• The pretrained score model on the real data is available during distillation. Thus, rather than defining an amortized score model that takes in a conditioning variable for α, we show that we only need to learn a score model on the fake samples to make the objective simpler (see Proposition A.2). This is more effective as learning the score of multiple interpolated distributions is a generally more challenging task.
• We can initialize the fake score and generator with the weights from the pretrained model and don't need to run a separate pretraining phase as in training from scratch.
• Notice that in amortized score training in the distillation setting there is no explicit GAN loss to learn the discriminator in Algorithm 2. Learning the discriminator implicit to our proposed parametrization and once learned we can use this to minimize the skewed divergence via a non-saturating GAN regularizer for generator training. R J J o z I T j j J / 8 1 l z s l p 1 w q n + 8 W j k 7 u B 3 H M k Q 2 y S Y r E I f v k i J y S M 1 I h n D y Q J / J C X q 1 H 6 9 l 6 s 9 4 H r R P W M M J 1 8 k v W 5 z e e 8 K P Y < / l a t e x i t >
x fake = g ✓ (z)
< l a t e x i t s h a 1 _ b a s e 6 4 = " j M 0 y M 5 l 2 h u N t / V B w r r y k V c q n 4 4 k = " > A A A C A H i c b V D L S s N A F J 3 U V 6 2 v q A s X b g a L 4 K o k I t V l Q R C X F e w D m h g m 0 0 k 7 d P J g 5 k Y s I S D + i h s X i r j 1 M 9 z 5 N 0 7 a L r T 1 w I X D O f d y 7 z 1 + I r g C y / o 2 S k v L K 6 t r 5 f X K x u b W 9 o 6 5 u 9 d W c S o p a 9 F Y x L L r E 8 U E j 1 g L O A j W T S Q j o S 9 Y x x 9 d F n 7 n n k n F 4 + g W x g l z Q z K I e M A p A S 1 5 5 o E T E h j 6 Q f a Q e 3 C X O S r A A R m x 3 D O r V s 2 a A C 8 S e 0 a q a I a m Z 3 4 5 / Z i m I Y u A C q J U z 7 Y S c D M i g V P B 8 o q T K p Y Q O i I D 1 t M 0 I i F T b j Z 5 I M f H W u n j I J a 6 I s A T 9     15).
f d E R k K l x q G v O 4 t z 1 b x X i P 9 5 v R S C C z f j U Z I C i + h 0 U Z A K D D E u 0 s B 9 L h k F M d a E U M n 1 r Z g O i S Q U d G Y V H Y I 9 / / I i a Z / W 7 H q t f n N W b V w 9 T u M o o 0 N 0 h E 6 Q j c 5 R A 1 2 j J m o h i n L 0 j F 7 R m / F k v B j v x
G I W 7 / E n X / j p O 1 C W w 9 c O J x z L 3 P m h K n g G h z n 2 1 p Z X V v f 2 C x t l b d 3 d v f 2 7 c p B W y e Z o q x F E 5 G o b k g 0 E 1 y y F n A Q r J s q R u J Q s E 4 4 u i r 8 z j 1 T m i f y D s Y p 8 2 M y k D z i l I C R A r v i x Q S G Y Z Q P J o E H Q w Y k s K t O z Z k C L x N 3 T q p o j m Z g f 3 n 9 h G Y x k 0 A F 0 b r n O i n 4 O V H A q W C T s p d p l h I 6 I g P W M 1 S S m G k / n 0 a f 4 B O j 9 H G U K D M S 8 F T 9 f Z G T W O t x H J r N I q h e 9 A r x P 6 + X Q X T p 5 1 y m G T B J Z w 9 F m c C Q 4 K I H 3 O e K U R B j Q w h V 3 G T F d E g U o W D a K p s S 3 M U v L 5 P 2 W c 2 t 1 + q 3 5 9 X G 9 d O s j h I 6 Q s f o F L n o A j X Q D W q i F q L o A T 2 j V / R m P V o v 1 r v 1 M V t d s e Y V H q I / s D 5 / A P O m l O I = < / l a t e x i t > g ✓ < l a t e x i t s h a 1 _ b a s e 6 4 = " i P N W W N J l U a 3 h W G D G O z v U c u w s c X E = " > A A A C G H i c b V D L S g M x F M 3 U V 6 2 v U Z d u g k W o I H W m S H V Z E E Q 3 U s E + o K 0 l k 2 b
h b X 1 j c 6 u 4 X d r Z 3 d s / K B 8 e t X S S K k K b J O G J 6 o S o K W e C N g 0 z n H a k o h i H n L b D 8 U 3 u t x + p 0 i w R D 2 Y i a R D j U L C I E T R W a v e Q y x G W + u W K V / V m c F e J v y A V W K D R L 3 / 1 B g l J Y y o M 4 a h 1 1 / e k C T J U h h F O p 6 V e q q l E M s Y h 7 V o q M K Y 6 y G b n T t 0 z q w z c K F G 2 h H F n 6 u + J D G O t J 3 F o O 2 M 0 I 7 3 s 5 e J / X j c 1 0 X W Q M S F T Q w W Z L 4 p S 7 p r E z X 9 3 B 0 x R Y v j E E i S K 2 V t d M k K F x N i E 8 h D 8 5 Z d X S e u i 6 t e q t f v L S v 3 2 a R 5 H E U 7 g F M 7 B h y u o w x 0 0 o A k E x v A M r / D m S O f F e X c + 5 q 0 F Z x H h M f y B 8 / k D 7 L + P u w = = < / l a t e x i t > ↵ < l a t e x i t s h a 1 _ b a s e 6 4 = " i P N W W N J l U a 3 h W G D G O z v U c u w s c X E = " > A A A C G H i c b V D L S g M x F M 3 U V 6 2 v U Z d u g k W o I H W m S H V Z E E Q 3 U s E + o K 0 l k 2 b a 0 M y D 5 I 5 S h g F / w o 2 / 4 s a F I m 6 7 8 2 9 M H 4 K 2 H g g c z j n h 3 n u c U H A F l v V l p B Y W l 5 Z X 0 q u Z t f W N z S 1 z e 6 e q g k h S V q G B C G T d I Y o J 7 r M K c B
z 2 H R W S x d G N G i X U D X E / Y g E j W G n J M / e d E K u B H 2 Q y 9 5 x E s t v M k Q G i 9 y r 3 z I p V t c Z A 8 8 S e k g p M 0 f D M L 6 c X k z S k k S I c S 9 m 1 r U S 5 G R a K E U 7 z s p N K m m A y x H 3 a 1 T T C I Z V u N n 4 h R 0 d a 6 a E g F r o i h c b q 7 4 k M h 1 K O Q l 9 3 F g f L W a 8 Q / / O 6 q Q r O 3 Y x F S a p o R C a L g p Q j F a M i D 9 R j g h L F R 5 p g I p i + F Z E B F p g o n V p Z h 2 D P v j x P W i d V u 1 a t X Z 9 W 6 p e P k z h K c A C H c A w 2 n E E d r q A B T S D w A M / w C m / G k / F i v B s f k 9 Y F Y x r h H v y B 8 f k D S s m Y c A = = < / l a t e x i t > s ext < l a t e x i t s h a 1 _ b a s e 6 4 = " Y S G E k F y a 2 c D V d 4 1 L Z + N d + j T D l K U = " > A A A C q H i c l V F d i 9 Q w F E 3 r 1 1 q / R n 3 0 J T i s r C J D K 7 I K v i w I I v g y w s 7 O y m Q s t + l t J 0 y a l u R W d i g F / 5 n / w T f / j Z k P Y Z 3 1 x Q u B w z n J z b n n Z o 1 W j u L 4 V x B e u 3 7 j 5 q 2 D 2 9 G d u / f u P x g 8 f H T m 6 t Z K n M h a 1 / Y 8 A 4 d a G Z y Q I o 3 n j U W o M o 3 T b P l + r U + / o X W q N q e 0 a n B e Q W l U o S S Q p 9 L B j 0 h o L E j o z I J E H o k M S 2 U 6 0 K o 0 m P c 8 e s Z F B b T I i s 7 1 q W i c + t o J V 3 C 8 o P 7 o j 3 L R b 8 k C l t i n 9 I 4 L 0 M 0 C X n J 6 z o X 4 v x 7 e v N 7 r I d D k l x w J q 8 o F C b t x n A 6 G 8 S j e F L 8 K k h 0 Y s l 2 N 0 8 F P k d e y r d C Q 1 O D c L I k b m n d g S U m N f S R a h w 3 I J Z Q 4 8 9 B A h W 7 e b Y L u + a F n c l 7 U 1 h 9 D f M N e f t F B 5 d y q y v z N 9 V h u X 1 u T / 9 J m L R V v 5 5 0 y T U t o 5 P a j o t W c a r 7 e G s + V R U l 6 5 Q F I q 7 x X L h f g A y C / 2 8 i H k O y P f B W c v R o l x 6 P j z 6 + H J x + + b + M 4 Y E / Y U 3 b E E v a G n b C P b M w m T A a H w a f g N J i E L 8 J x O A 2 / b K + G w S 7 C x + y v C r P f a + / S M w = = < / l a t e x i t > ( s ext (x fake t ; ↵, t) s ext (x real t ; ↵, t) ) < l a t e x i t s h a 1 _ b a s e 6 4 = " 4 H B t k V C K o P 8 Q X l S 6 X z t N E F C 8 l 2 s = " > A A A C G 3 i c b V D L S s N A F J 3 4 t r 6 q L t 0 M F k F B S i K i g h t B E J c K V o W m l p v p T T s 4 k 4 S Z G 7 G E g p / h x l 9 x 4 0 I R V 4 I L / 8 a k 7 c L X g Q u H c + 5 l 5 p w g U d K S 6 3 4 6 I 6 N j 4 x O T U 9 O l m d m 5 + Y X y 4 t K 5 j V M j s C Z i F Z v L A C w q G W G N J C m 8 T A y C D h R e B N e H h X 9 x g 8 b K O D q j b o I N D e 1 I h l I A 5 V K z v O V r o E 4 Q Z n 4 b t I Z e 0 0 + s v M p 8 G 3 K 8 p d 6 6 T x 0 k 2 O c + q K Q D m 5 w 2 S s 1 y x a 2 6 f f C / x B u S C h v i p F l + 9 1 u x S D V G J B R Y W / f c h B o Z G J J C Y a / k p x Y T E N f Q x n p O I 9 B o G 1 k / W 4 + v 5 U q L h 7 H J J y L e V 7 9 f Z K C t 7 e o g 3 y y S 2 N 9 e I f 7 n 1 V M K 9 x q Z j J K U M B K D h 8 J U c Y p 5 U R R v S Y O C V D c n I I z M / 8 p F B w w I y u s s S v B + R / 5 L z r e q 3 k 5 1 5 3 S 7 c n B 0 N 6 h j i q 2 w V b b O P L b L D t g x O 2 E 1 J t g 9 e 2 T P 7 M V 5 c J 6 c V + d t s D r i D C t c Z j / g f H w B y v G h o A = = < / l a t e x i t > ext (✓; ↵, t) < l a t e x i t s h a 1 _ b a s e 6 4 = " j P 6 r l b u J A v e S e a s v r d a O Z e 8 1 j c 0 = " > A A A C G 3 i c b V B N S 8 N A E N 3 4 b f 2 K e v S y W A Q F K U k R F b w I g n j w o G B V a G q Z b D d 2 6 W Y T d i d i C Q V / h h f / i h c P i n g S P P h v 3 L Q 9 + P V g 4 P H e D D P z w l Q K g 5 7 3 6 Y y M j o 1 P T E 5 N l 2 Z m 5 + Y X 3 M W l c 5 N k m v E a S 2 S i L 0 M w X A r F a y h Q 8 s t U c 4 h D y S / C z k H h X 9 x w b U S i z r C b 8 k Y M 1 0 p E g g F a q e l W A w W h h G a Q G k G D G L D N Q O b H v a s 8 M B H l t 9 h b L 6 w 9 G o B M 2 7 B J c a P p l r 2 K 1 w f 9 S / w h K Z M h T p r u e 9 B K W B Z z h U y C M X X f S 7 G R g 0 b B J O + V g s z w F F g H r n n d U g U x N 4 2 8 / 1 u P r l m l R a N E 2 1 J I + + r 3 i R x i Y 7 p x a D u L 6 8 1 v r x D / 8 + o Z R r u N X K g 0 Q 6 7 Y Y F G U S Y o J L Y K i L
Bottom: Amortized score model training involves computing the score of the mixture distribution on both fake and real samples diffused with noise level t and then updating the weights using the gradient of Eq. ( 14).
this section cite: []

Section: Algorithm 2 Score-of-Mixture Distillation
Inputs: Randomly initialized generator g θ , fake score model s fake ψ , discriminator ℓ ψ , pretrained score model s p , real dataset D, score training sub-iterations = 5, learning rates (η gen , η score ), GAN generator regularizer weight λ Initialization Initialize s fake ψ and g θ with weights from s p Training: Alternating updates of g θ and s ψ for each training iteration do Generator Training: Freeze s fake ψ and ℓ ψ Sample mini-batch of fake samples x fake = g θ (z), z ∼ N (0, I) Sample t ∼ p(t) and α as described in Sec.
3.5 Compute generator gradient γ ext ψ (θ; α, t) from Eq. (15) Compute GAN regularizer loss L (α,t) GAN from Eq. (11) Update parameters: θ ← θη gen E p(α)p(t) [γ ext ψ (θ; α, t) + λ∇ θ L (α,t) GAN (θ)] Amortized Score Training: Freeze g θ for each sub-iteration do Sample mini-batch of real samples x real ∼ D Sample t ∼ p(t) and α Compute score matching loss using explicit parametrization L ext score (ψ; α, t) from Eq. (14) Update parameters: ψ ← ψη score ∇ ψ E p(α)p(t) [L ext score (ψ; α, t)] end for end for Return: Trained model parameters θ, ψ D. On GAN Training The vanilla GAN is min
θ max ψ {E p(x) [log D ψ (x)] + E q θ (x) [log(1 -D ψ (x))]}.
Breaking down, the discriminator training is
min ψ -E p(x) [log D ψ (x)] -E q θ (x) [log(1 -D ψ (x))]},
and the generator training is min
θ E q θ (x) [log(1 -D ψ (x))]. Note that D ψ (x) = r ψ (x) 1 + r ψ (x) = 1 1 + r -1 ψ (x)
.
Further, the sigmoid logit C ψ (x), i.e., D ψ (x) = σ(C ψ (x)), is log r ψ (x). Note that the optimal discriminator for each θ is x) . The non-saturating versions is training generator based on
D ⋆ (x) = p(x) p(x)+q θ (x) or r ⋆ (x) = p(x) q θ (
min θ -E q θ (x) [log D ψ (x))] ≈ E q θ (x) log q θ (x) p(x) + 1 .
The StyleGAN uses the non-saturating loss:
min ψ E p(x) [sp(-log r ψ (x))] + E q θ (x) [sp(log r ψ (x))]}, min θ E q θ (x) [sp(-log r ψ (x))].
Note that sp(y) := log(1 + e y ). Hence, note that
sp(-log r ψ (x)) = log(1 + r -1 ψ (x)) = -log D ψ (x), sp(log r ψ (x)) = log(1 + r ψ (x)) = -log(1 -D ψ (x)).
this section cite: []

Section: D.1. On the Non-Saturating Generative Loss
The original, saturating version of the generator objective is
min θ E q θ (x) [-sp(log r ψ (x))] = E q θ (x) [-log(1 + r ψ (x))], whose gradient is ∇ θ E q(z) [-log(1 + r ψ (g θ (z)))] = E q(z) - r ′ ψ (g θ (z)) 1 + r ψ (g θ (z)) ∇ θ g θ (z) .
Note that the plug-in reverse KL-divergence loss is
min θ E q θ (x) [-log r ψ (x)].
Compared to this, the non-saturating loss has the additional sp(•):
min θ E q θ (x) [sp(-log r ψ (x))] = E q θ (x) [log(1 + r ψ (x) -1 )].
This seems to help prevent vanishing gradients. Consider
min θ E q θ (x) [log(τ + r ψ (x) -1 )] = E q(z) [log(τ + r ψ (g θ (z)) -1 )],
Here, we note that the term ∥s p (x)s q (x)∥ 2 is common in both expectation, and can be safely dropped to train the discriminator, which leads to a simplified objective
L ′ (ψ; α) = α E p(x) [(1 -D ψ (x; α)) 2 ] + (1 -α) E q(x) [D ψ (x; α) 2 ] = αp(x)(1 -D ψ (x; α)) 2 + (1 -α)q(x)D ψ (x; α) 2 dx.
We note that this is equivalent to the discriminator objective induced by the following f -divergence
D fα (p ∥ q) := 1 - p(x)q(x) αp(x) + (1 -α)q(x) dx := D α-LC (p ∥ q),
where f α (r) := (1-α)(1-r) αr+(1-α) is a convex function over [0, ∞) for α ∈ (0, 1). For α = 1 2 , this divergence becomes symmetric in p and q and is known as the Le Cam distance (Le Cam, 2012, p. 47) in the literature (Polyanskiy & Wu, 2019). We thus call the general divergence for α ∈ (0, 1) the α-Le Cam distance. In the GAN literature, this is known as the LSGAN objective (Mao et al., 2017).
As we revealed, our discriminator training in distillation can also be done separately using the α-Le Cam-distance-based objective. However, we conjecture that our score-regression-based end-to-end objective may have benefit, as our primary goal of discriminator training is to use it in the generator update in the form of an approximate score of mixture. We leave the further exploration of such alternative methods as a future work.
this section cite: ['b42', 'b36']

Section: E. More on Experiments and Additional Results
We present some additional experiments and results in this section. We first provide a more detailed training configuration for our experiments in Sec. 5 and then evaluate our proposed method on a synthetic swiss-roll dataset in Appendix E.2. Finally, we present some samples generated from Score-of-Mixture Training and Score-of-Mixture Distillation in Figs. 9-12.
this section cite: []

Section: E.1. Training Configuration
We summarize the detailed training configuration in Table 3.
this section cite: []

Section: E.2. Toy Swiss Roll
We tested our proposed framework and ablated various design choices on a synthetic swiss roll dataset. We followed the dataset setup by Che et al. (2020). We trained models with SMT and SMD and compared this against an amortized version of reverse KL minimization with DMD weighting (α ∈ {0, 1}) similar to the ablations in Sec. 5.3. Additionally we compared against non-score-based baselines including the vanilla GAN and Diffusion-GAN (Wang et al., 2023).
Across all experiments, we use the same generator architecture -a two-layer MLP with a hidden dimension of 128 and leaky ReLU nonlinearity. We train all models for 200k steps on a single NVIDIA 3090 GPU with a batch size of 256. All score-based methods leverage a learning rate of 1e-5 for the generator and 1e-4 for the amortized score (and discriminator when applicable) whereas the GAN-based methods use a learning rate of 1e-4 for both generator and discriminator. We use the AdamW optimizer without any learning rate schedulers.
this section cite: ['b3', 'b56']

Section: References
Ref_id:b0 Title: Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit Year: (2017)
Ref_id:b1 Title: Denoising Diffusion Models with Transitive Closure Time-Distillation Year: (2023)
Ref_id:b2 Title: Large Scale GAN training for High Fidelity Natural Image Synthesis Year: (2019)
Ref_id:b3 Title: Your GAN is Secretly an Energy-Based Model and you should use Discriminator Driven Latent Sampling Year: (2020)
Ref_id:b4 Title: Information Theory and Statistics: A Tutorial Year: (2004)
Ref_id:b5 Title: ImageNet: A Large-Scale Hierarchical Image Database Year: (2009)
Ref_id:b6 Title: Diffusion Models Beat GAns On Image Synthesis Year: (2021)
Ref_id:b7 Title: Sigmoid-weighted Linear Units for Neural Network Function Approxima-tion in Reinforcement Learning Year: (2018)
Ref_id:b8 Title: Consistency Models Made Easy Year: (2025)
Ref_id:b9 Title:  Year: (2014)
Ref_id:b10 Title: Multistep Consistency Models Year: (2024)
Ref_id:b11 Title: GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium Year: (2017)
Ref_id:b12 Title: Denoising Diffusion Probabilistic Models Year: (2020)
Ref_id:b13 Title: Simple Diffusion: End-to-end Diffusion for High Resolution Images Year: (2023)
Ref_id:b14 Title: Knowledge Diffusion for Distillation Year: (2023)
Ref_id:b15 Title: Estimation of Non-Normalized Statistical Models by Score Matching Year: (2005)
Ref_id:b16 Title: A Style-Based Generator Architecture for Generative Adversarial Networks Year: (2021-12)
Ref_id:b17 Title: Elucidating the Design Space of Diffusion-based Generative Models Year: (2022)
Ref_id:b18 Title: Elucidating the Design Space of Diffusion-based Generative Models Year: (2022)
Ref_id:b19 Title: Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion Year: (2024)
Ref_id:b20 Title: Understanding Diffusion Objectives as the ELBO with simple Data Augmentation Year: (2024)
Ref_id:b21 Title: Variational Diffusion Models Year: (2021)
Ref_id:b22 Title: Auto-encoding Variational Bayes Year: (2014)
Ref_id:b23 Title: Informationtheoretic Diffusion Year: (2023)
Ref_id:b24 Title: Learning Multiple Layers of Features from Tiny Images Year: (2009)
Ref_id:b25 Title: Asymptotic methods in statistical decision theory Year: (2012)
Ref_id:b26 Title: Improving the Training of Rectified Flows Year: (2024)
Ref_id:b27 Title: Flow Matching for Generative Modeling Year: (2023)
Ref_id:b28 Title: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow Year: (2022)
Ref_id:b29 Title: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow Year: (2023)
Ref_id:b30 Title: DPMsolver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps Year: (2022)
Ref_id:b31 Title: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps Year: (2022)
Ref_id:b32 Title: Fast Solver for Guided Sampling of Diffusion Probabilistic Models Year: (2022)
Ref_id:b33 Title: Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed Year: (2021)
Ref_id:b34 Title: Diff-Instruct: A Universal Approach for Transferring Knowledge from Pre-Trained Diffusion Models Year: (2024)
Ref_id:b35 Title: One-step Diffusion Distillation through Score Implicit Matching Year: (2024)
Ref_id:b36 Title: Least Squares Generative Adversarial Networks Year: (2017)
Ref_id:b37 Title: On Distillation of Guided Diffusion Models Year: (2023)
Ref_id:b38 Title: Spectral Normalization for Generative Adversarial Networks Year: (2018)
Ref_id:b39 Title: Improved Denoising Diffusion Probabilistic Models Year: (2021)
Ref_id:b40 Title: A Family of Statistical Symmetric Divergences based on Jensen's Inequality Year: (2010)
Ref_id:b41 Title: Training Generative Neural Samplers using Variational Divergence Minimization Year: (2016)
Ref_id:b42 Title: Lecture notes on information theory Year: (2019)
Ref_id:b43 Title: An Empirical Bayes Approach to Statistics Year: (1956)
Ref_id:b44 Title: Progressive Distillation for Fast Sampling of Diffusion Models Year: (2022)
Ref_id:b45 Title: Multistep Distillation of Diffusion Models via Moment Matching Year: (2024)
Ref_id:b46 Title: Stylegan-XL: Scaling Stylegan to Large Diverse Datasets Year: (2022)
Ref_id:b47 Title: Deep Unsupervised Learning using Nonequilibrium Thermodynamics Year: (2015)
Ref_id:b48 Title: Denoising Diffusion Implicit Models Year: ()
Ref_id:b49 Title: Improved Techniques for Training Consistency Models Year: (2024)
Ref_id:b50 Title: Improved Techniques for Training Consistency Models Year: (2024)
Ref_id:b51 Title: Generative Modeling by Estimating Gradients of the Data Distribution Year: (2019)
Ref_id:b52 Title: Sliced Score Matching: A Scalable Approach to Density and Score Estimation Year: (2020)
Ref_id:b53 Title: Score-based Generative Modeling through Stochastic Differential Equations Year: (2021)
Ref_id:b54 Title: Consistency Models Year: (2023-07)
Ref_id:b55 Title: A Connection Between Score Matching and Denoising Autoencoders Year: (2011)
Ref_id:b56 Title: Diffusion-GAN: Training GANs with Diffusion Year: (2023)
Ref_id:b57 Title: EM Distillation for One-Step Diffusion Models Year: (2024)
Ref_id:b58 Title: Improved Distribution Matching Distillation for Fast Image Synthesis Year: (2024)
Ref_id:b59 Title: One-Step Diffusion with Distribution Matching Distillation Year: (2024)
Ref_id:b60 Title: Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation Year: (2024)
