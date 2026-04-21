Title: THE SPACETIME OF DIFFUSION MODELS: AN INFORMATION GEOMETRY PERSPECTIVE
Abstract: We present a novel geometric perspective on the latent space of diffusion models. We first show that the standard pullback approach, utilizing the deterministic probability flow ODE decoder, is fundamentally flawed. It provably forces geodesics to decode as straight segments in data space, effectively ignoring any intrinsic data geometry beyond the ambient Euclidean space. Complementing this view, diffusion also admits a stochastic decoder via the reverse SDE, which enables an information geometric treatment with the Fisher-Rao metric. However, a choice of x T as the latent representation collapses this metric due to memorylessness. We address this by introducing a latent spacetime z = (x t , t) that indexes the family of denoising distributions p(x 0 |x t ) across all noise scales, yielding a nontrivial geometric structure. We prove these distributions form an exponential family and derive simulation-free estimators for curve lengths, enabling efficient geodesic computation. The resulting structure induces a principled Diffusion Edit Distance, where geodesics trace minimal sequences of noise and denoise edits between data. We also demonstrate benefits for transition path sampling in molecular systems, including constrained variants such as low-variance transitions and region avoidance. Code is available at https://github.com/Aalto-QuML/spacetime-geometry.

Section: INTRODUCTION
Diffusion models have emerged as a powerful paradigm for generative modeling, demonstrating remarkable success in learning to model and sample data (Yang et al., 2023). While the underlying mathematical frameworks of training and sampling are well-established (Sohl-Dickstein et al., 2015;Kingma et al., 2021;Song et al., 2021;Lu et al., 2022;Holderrieth et al., 2025), analysing how information evolves through the noisy intermediate states x t for t ∈ [0, T ] remains an open question. Our work addresses this by defining and analyzing the geometric structure of diffusion models, which provides a principled framework for understanding their inner workings.
Figure 1: A geodesic in spacetime is the shortest path between denoising distributions.
In generative models, a common way to study the intrinsic geometry of the data is to pull back the ambient (Euclidean) metric onto the latent space (Arvanitidis et al., 2018;2022). Equipped with this pullback metric, shortest paths (i.e., geodesics) in the latent space decode to realistic transitions along data that lie on a lower-dimensional submanifold.
In a diffusion model, a natural choice for the decoder is the reverse ODE x 0 (x T ), which allows us to derive the pullback geometry of the latents x T . Interestingly, we prove that this leads to latent shortest paths always decoding to linear interpolations in data space, which have little practical utility.
We then turn our attention to the denoising posterior distribution p(x 0 |x t ) given by the reverse SDE. We propose an alternative Fisher-Rao geometry, which measures how the denoising distribution p(x 0 |x t ) changes when manipulating the latent x t .
We introduce the Fisher-Rao metric G(x t , t) that varies with both state and time over the latent spacetime (x t , t) (Fig. 1).
Estimating geodesics in information geometry is usually tractable only for analytic families. Although denoising distributions in diffusion are complex and non-Gaussian, we show that they form an exponential family. This simplifies the geometry and yields a practical method for computing geodesics between any two samples through the spacetime. In the Fisher-Rao setting, curve lengths can be evaluated without running the reverse SDE, which significantly reduces the computational cost.
We demonstrate the utility of the Fisher-Rao geometry in diffusion models in two ways. First, it induces a principled Diffusion Edit Distance on data that admits a clear interpretation: the geodesic between x a and x b traces the minimal sequence of edits, adding just enough noise to forget information specific to x a and then denoising to introduce information specific to x b . The resulting length quantifies the total edit cost. Second, spacetime geodesics allow generating transition paths in molecular systems, where we obtain results competitive with specialized state-of-the-art methods and can incorporate constraints such as avoidance of designated regions in data space.
this section cite: ['b36', 'b31', 'b20', 'b33', 'b23', 'b13', 'b2']

Section: BACKGROUND ON DIFFUSION MODELS
We assume a data distribution q defined on R D , and the forward process
p(x t |x 0 ) = N (x t |α t x 0 , σ 2 t I),(1)
which gradually transforms q into pure noise p T ≈ N (0, σ 2 T I) at time T , where α t , σ t define the forward drift f t and diffusion g t . There exists a denoising SDE reverse process (Anderson, 1982) Reverse SDE:
dx = f t x -g 2 t ∇ log p t (x) dt + g t dW t , x T ∼ p T ,(2)
where p t is the marginal distribution of the forward process (Eq. 1) at time t, and W is a reverse Wiener process. Somewhat unexpectedly, there exists a deterministic Probability Flow ODE (PF-ODE) with matching marginals (Song et al., 2021):
PF ODE: dx = f t x - 1 2 g 2 t ∇ log p t (x) dt, x T ∼ p T .(3)
Assuming we can approximate the score ∇ log p t (Karras et al., 2024), we denote by x T → x 0 (x T ) the deterministic denoiser of solving the PF-ODE from noise x T , while we denote by p(x 0 |x t ) the denoising distributions induced by stochastic sampling of the reverse SDE (Karras et al., 2022).
this section cite: ['b1', 'b33', 'b19', 'b18']

Section: RIEMANNIAN GEOMETRY OF DIFFUSION MODELS
Riemannian geometry equips a latent space Z with a smoothly varying metric tensor G(z) for z ∈ Z. This metric defines inner products and induces the notions of distance and curve length (Do Carmo & Francis, 1992). Several works have developed diffusion models on top of Riemannian manifolds, such as spheres, tori and hyperboloids (De Bortoli et al., 2022;Huang et al., 2022;Thornton et al., 2022). In this paper, we instead study what kind of Riemannian geometries are implicitly induced by the denoiser within a real vector space R D (e.g., images).
In Euclidean geometry, the space is flat, with distances given by the length of straight lines connecting points. In Riemannian spaces, the shortest path between two points is no longer straight, but a curved geodesic. A smooth curve γ : [0, 1] → Z between fixed endpoints γ 0 , γ 1 is a geodesic if it minimizes the length
ℓ(γ) = 1 0 ∥ γs ∥ G ds = 1 0 γT s G(γ s ) γs ds,(4)
or, equivalently, the energy E(γ) = 1 2 1 0 ∥ γs ∥ 2 G ds. We introduce two interpretations of Riemannian geometry G for diffusion models, depending on whether the decoder is deterministic or stochastic. In both cases, we first assume the latent space is the noise space x T , and later relax this to cover the entire noisy sample space x t .
Deterministic sampler: pullback geometry. Let x T → x 0 (x T ) be a deterministic map given by the PF-ODE (Eq. 3) mapping noise to data. We propose the pullback metric (Arvanitidis et al., 2022;Park et al., 2023
) G PB (x T ) = ∂x 0 ∂x T ⊤ ∂x 0 ∂x T ∈ R D×D , x 0 := x 0 (x T ) ∈ R D(5)
which measures how an infinitesimal noise step dx T changes the decoded sample:
x 0 (x T + dx T ) -x 0 (x T ) 2 = dx ⊤ T G PB (x T )dx T + o(∥dx T ∥ 2 ).(6)
Stochastic sampler: information geometry. Alternatively, consider a stochastic decoder that, for each latent x T defines a denoising distribution p(x 0 |x T ) by solving the Reverse SDE ( Eq. 2). We propose the information-geometric viewpoint via the Fisher-Rao metric (Amari, 2016)
G IG (x T ) = E x0∼p(x0|x T ) ∇ x T log p(x 0 |x T ) ∇ x T log p(x 0 |x T ) ⊤ ∈ R D×D ,(7)
which measures how an infinitesimal noise step dx T changes the entire denoising distribution:
KL p(x 0 | x T ) p(x 0 | x T + dx T ) = 1 2 dx ⊤ T G IG (x T )dx T + o(∥dx T ∥ 2 ). (8
)
For a helpful tutorial on information geometry, we refer to Mishra et al. ( 2023).
this section cite: ['b15', 'b34', 'b3', 'b27', 'b0']

Section: PULLBACK GEOMETRY COLLAPSES IN DIFFUSION MODELS
Both pullback and information geometries are, in principle, applicable. We will first show the pullback geometry has fundamental theoretical limitations in diffusion models, rendering it practically useless.
Figure 2: The pullback geodesics curve in noise space, but decode to straight lines in data space.
Assume we estimate a geodesic γ in the noise space x T such that its endpoints decode to x 0 (γ 0 ) = x a and x 0 (γ 1 ) = x b . The pullback energy E(γ) (Eq. 4) can be shown to only depend on the decoded curve x 0 (γ s ) in data space (See Appendix B):
E PB (γ) = 1 2 1 0 d ds x 0 (γ s ) 2 ds. (9
)
The unique minimizer is the constant-speed straight line x s = (1-s)x a +sx b in data space. Since the ODE is bijective, this line has a unique latent preimage γ ⋆ s = x -1 0 (x s ) = x T (x s ), which is thus a pullback geodesic, and the energy reduces to Euclidean distance in data space:
E PB (γ) := 1 2 x a -x b 2 . (10
)
Hence, all pullback geodesics decode to straight segments, ignoring the curvature of the data manifold and undermining downstream applications (See Fig. 2). The same pathology applies for denoised geodesics in the intermediate space x t as well. The core reason for this is that, in diffusion models, the latent and data spaces have the same dimension. The decoder operates directly in the ambient space and, without further dimensional constraints, it cannot capture the intrinsic structure of the data, even if the data lie on a lower-dimensional submanifold. As a result, the standard pullback metric provides no meaningful geometric information. A formal proof and discussion are in Appendix B.
this section cite: []

Section: INFORMATION GEOMETRY WITH DENOISING DECODERS
Under the stochastic view, the decoder is the denoising distribution p(x 0 |x T ) obtained by reversing the diffusion process (Eq. 2). This yields a family of distributions on the data space parametrized with noise vectors x T . The information geometry assigns the Fisher-Rao metric to the latent domain, and geodesic energies/lengths are computed as in Section 3.
The latent spacetime. Diffusion models are "memoryless" (Domingo-Enrich et al., 2025):
p(x T | x 0 ) ≈ p T (x T ) ⇒ p(x 0 | x T ) ≈ q(x 0 ).(11)
Hence p(x 0 | x T ) is (approximately) independent of x T , implying ∇ x T log p(x 0 | x T ) ≈ 0 and a collapse of the Fisher-Rao metric, G IG ≈ 0 (Eq. 7). Consequently, if we identify the latent space with z = x T , all x T become metrically indistinguishable. This could be avoided by choosing z = x t for some t < T ; however, instead of choosing an arbitrary noise level t, we propose to model all noise levels simultaneously by considering points in the (D + 1)-dimensional latent spacetime
z = (x t , t) ∈ R D × (0, T ],(12)
which define the family of all denoising distributions p(x 0 |x t ) across all noise levels (Fig. 1).
this section cite: ['b8']

Section: Why include time?
The resulting Fisher-Rao metric G IG (z) varies with state and time, restoring a nontrivial geometry and enabling navigation across noise levels within a unified structure. Identifying clean data with spacetime points (x, 0), for which p(x 0 | x 0 = x) = δ x , lets geodesics connect clean endpoints through noisy intermediates. This yields (i) a principled notion of distance between data as the length of the shortest spacetime path (Diffusion Edit Distance), and (ii) a mechanism for transition-path sampling via spacetime geodesics; both are demonstrated empirically in Section 6.
Tractable energy estimation. Usually, the information-geometric energy of a discretized curve γ = {z n } N -1 n=0 is approximated via the local-KL approximation (Arvanitidis et al., 2022):
E(γ) ≈ (N -1) N -2 n=0 KL p(• | z n ) p(• | z n+1 ) ,(13)
but such KLs are generally intractable, unless p(•|z) is a simple analytic distribution such as multinomial or Gaussian, which is not the case for denoising distributions p(x 0 |x t ). Nonetheless, we show that in the specific case of the diffusion spacetime, the energy can be tractably estimated.
Proposition 5.1 (Spacetime energy estimation -informal). The energy of discretized spacetime curve γ = {z n } N -1 n=0 with z n = (x tn , t n ) admits an approximation
E(γ) ≈ N -1 2 N -2 n=0 η(z n+1 ) -η(z n ) ⊤ µ(z n+1 ) -µ(z n ) ,(14)
where
η(x t , t) = α t σ 2 t x t , - α 2 t 2σ 2 t , µ(x t , t) = E x 0 | x t , E ∥x 0 ∥ 2 | x t .(15)
The proof (Appendix C) consists of showing that denoising distributions form an exponential family, which admits a simplified energy formula. In practice, we calculate µ(x t , t) with Tweedie's formula over the approximate denoiser x0 (x t ) (See Appendix C.2 for details),
E x 0 | x t ≈ x0 (x t ) E ∥x 0 ∥ 2 | x t ≈ x0 (x t ) 2 + σ 2 t α t div xt x0 (x t ),(16)
where both x0 and div x0 are computed efficiently via Hutchinson's trick (Hutchinson, 1989;Grathwohl et al., 2019), enabling the esimation of µ(x t , t) with a single Jacobian-vector product (JVP).
Spacetime geodesics are simulation-free: the energy calculation requires only N JVPs of the denoiser x0 for a curve discretized into N points.  6 EXPERIMENTS 6.1 SAMPLING TRAJECTORIES We compare the trajectories obtained by solving the PF-ODE x 0 (x T ) (Eq. 3) with geodesics between the same endpoints x 0 , x T . For a toy example of 1D mixture of Gaussians, we observe the geodesics curving less than the PF-ODE trajectories in the early sampling (high t), while being indistinguishable for lower values of t (See Fig. 3 left and Appendix G.1 for details).
We find only marginal perceptual difference between the PF-ODE sampling trajectories and the geodesics in the EDM2 ImageNet-512 model (Karras et al., 2024). The geodesic appears to generate information slightly earlier, but the difference is minor (See Fig. 3 right, and Appendix G.2 for details).
We note that spacetime geodesics are not an alternative sampling method since they require knowing the endpoints beforehand. An investigation into whether our framework can be used to improve sampling strategies is an interesting future research direction.
this section cite: ['b3', 'b16', 'b12', 'b19']

Section: DIFFUSION EDIT DISTANCE
The spacetime geometry yields a principled distance on the data space. We identify clean datum x ∈ R d with the spacetime point (x, 0), corresponding to the Dirac denoising distribution δ x . Given two points x a , x b , we define the Diffusion Edit Distance (DiffED) by
DiffED(x a , x b ) = ℓ(γ),(17)
where γ is the spacetime geodesic between (x a , 0) and (x b , 0). For numerical stability, we anchor endpoints at a small t min > 0 rather than at 0. See Algorithm 4 for DiffED pseudocode.
A spacetime geodesic links two clean data points through intermediate noisy states. It can be interpreted as the minimal sequence of edits: add just enough noise to discard information specific to x a , then remove noise to introduce information specific to x b . The path length is the total edit cost, which is measured by how much the denoising distribution changes along the path. Fig. 4 visualizes the spacetime geodesics: as endpoint similarity decreases, the intermediate points become noisier.
We quantitatively evaluate DiffED on image data. First, we ask whether DiffED correlates with human perception as approximated by Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018). We randomly selected 10 classes in the ImageNet dataset and sampled 20 random image pairs for each. We then evaluated the DiffED and LPIPS for each image pair, and found the correlation to be very low at approximately -7%, suggesting that perceptual similarity and geometric edit cost capture different notions of closeness. We found DiffED to be more closely related to the structural similarity index measure (SSIM) (Wang et al., 2004), which correlates at 53% with DiffED.
To qualitatively compare different notions of image similarity, we order image pairs by their similarity evaluated with multiple metrics: DiffED, LPIPS, SSIM, and Euclidean. We show the results in Fig. Another application of the spacetime geometry is the problem of transition-path sampling (Holdijk et al., 2023;Du et al., 2024;Raja et al., 2025), whose goal is to find probable transition paths between low-energy states. We assume a Boltzmann distribution
q(x) ∝ exp(-U (x)), (18
)
where U is a known energy function, which is a common assumption in molecular dynamics. In this setting, the denoising distribution follows a tractable energy function (See Eq. 60)
p(x 0 |x t ) ∝ q(x 0 )p(x t |x 0 ) ∝ exp -U (x 0 ) -1 2 SNR(t) x 0 -x t /α t 2 -U (x0|xt) .(19)
To construct a transition path between two low-energy states x 1 0 and x 2 0 , we estimate the spacetime geodesic γ between them using a denoiser model x0 (x t ) ≈ E[x 0 |x t ] with Proposition 5.1, as shown in Fig. 5. At each interpolation point s ∈ [0, 1], the geodesic defines a denoising Boltzmann distribution p(x|γ s ) where U (x|γ s ) is the energy at that spacetime location. See Appendix G.3 for details.
this section cite: ['b38', 'b35', 'b14', 'b9', 'b28']

Section: Annealed Langevin Dynamics.
To sample transition paths, we use Langevin dynamics
dx = -∇ x U (x|γ s )dt + √ 2dW t ,(20)
whose stationary distributions are p(x | γ s ) ∝ exp(-U (x|γ s )) for any s. To obtain the trajectories from x 1 0 to x 2 0 , we gradually increase s from 0 to 1 using annealed Langevin (Song & Ermon, 2019).
Figure 6: Transition paths generated with a spacetime geodesic avoid high-energy regions without collapsing to a single path. Compared with MCMC baselines, the spacetime-geodesic method yields transition paths that better avoid high-energy areas, whereas Doob's Lagrangian collapses to generating nearly identical trajectories. Ten sample paths are shown for each method.
After discretizing the geodesic into N points γ n , we alternate between taking K steps of Eq. 20 conditioned on γ n and updating γ n → γ n+1 , as described in Algorithm 1. This approach assumes that p(x|γ n ) is close to p(x|γ n+1 ), and thus x ∼ p(x|γ n ) is a good starting point to Langevin dynamics conditioned on γ n+1 . Alanine dipeptide. We compute a spacetime geodesic connecting two molecular configurations of Alanine Dipeptide, as in Holdijk et al. (2023). In Fig. 5, the energy landscape is visualized over the dihedral angle space, with a neural network used to approximate the potential energy U . Using our trained denoiser x0 (x t ), we estimate the expectation parameter µ, which allows us to compute and visualize a geodesic trajectory through spacetime. Transition paths were generated using Algorithm 1. See Appendix G.3 for details.
Baselines. We considered For each method we generate 1,000 paths and report mean MaxEnergy (lower is better) and its numerical lower bound min γ max s U (γ s ), along with the number of energy evaluations needed for 1,000 paths. To train a base diffusion model for our method, we generated data using Langevin dynamics (16M 1 energy evaluations), a one-time cost that does not scale with the number of generated transition paths.
Results. We show in Table 1 that our method outperforms the baselines in the MaxEnergy obtained along the transition paths. It is also considerably closer to the lower bound than to the next best baseline (MCMC-fixed length) while requiring several orders of magnitude fewer energy function evaluations. In Fig. 6, we show a qualitative comparison of transition paths generated with our method and the baselines. Our proposed method shows improved efficiency in avoiding high-energy 1 +16M is the number of energy function evaluations to generate the training set with Langevin dynamics for the base diffusion model. We did not tune this number, and fewer evaluations may yield comparable performance.
regions compared to MCMC. In contrast, the Doob's Lagrangian method converged to a suboptimal solution, producing nearly identical transition paths. We discuss this in more detail in Appendix H.
this section cite: ['b32', 'b14']

Section: Algorithm 1 Transition Path Sampling with Annealed Langevin Dynamics
Require:
x a , x b ∈ R D endpoints, N γ > 0, T > 0, t min , dt 1: γ ← SPACETIMEGEODESIC(x a , x b )
▷ Approximate spacetime geodesic with Algorithm 3 2: T ← {x := x a } ▷ Initialize chain T at x a 3: for n ∈ {0, . . . , N γ -1} do ▷ Iterate over the points on the geodesic γ n 4:
for t ∈ {1, . . . , T } do 5:
ε ∼ N (0, I) ▷ Sample Gaussian noise 6:
x ← x -∇ x U (x|γ n )dt + √ 2dtε ▷ Langevin update 7:
T ← T ∪ {x} ▷ Append state x to chain 8:
end for 9: end for 10: return T ▷ Return chain 6.4 CONSTRAINED PATH SAMPLING Suppose we would like to impose additional constraints along the geodesic interpolants. This corresponds to penalized optimization (Rygaard et al. (2025) also explore regularized geodesics)
min γ E(γ) + λ 1 0 h(γ s )ds, s.t. γ 0 = (x 1 0 , 0), γ 1 = (x 2 0 , 0) ,(21)
where h : R × R D → R is some penalty function with λ > 0. We demonstrate the principle by (i) penalizing transition path variance, and (ii) imposing regions to avoid in the data space..
Low-variance transitions. Suppose we want the posterior p(x | γ s ) to have a low variance. This concentrates the path around a narrower set of plausible states, more repeatable trajectories, albeit at the cost of reduced coverage. By Eq. 56, higher SNR(t) yields lower denoising variance, so we implement this by penalizing low SNR via h(x, t) = max(-log SNR(t), ρ) for some threshold ρ.
Avoiding restricted regions. Suppose we want to avoid certain regions in the data space in the transition paths. We encode the region to avoid as a denoising distribution p(•|z * ) for some z * = (x * t , t * ) where larger the t * , larger the restricted region. We encode the penalty as KL distance between the denoising distributions (See Appendix D for the derivation)
KL p(•|z * )||p(•|γ s ) = s 0 d du η(γ u ) ⊤ (µ(γ u ) -µ(z * )) du + C (22
) h(γ s ) = min ρ, -KL p(•|z * )||p(•|γ s ) .(23)
In Fig. 7, we compare spacetime geodesics (unconstrained) with low-variance, and region-avoiding spacetime curves. We visualize both the curves and the corresponding transition paths generated with Algorithm 1. This demonstrates that our framework with the penalized optimization (Eq. 21) can incorporate various preferences on the transition paths.
this section cite: ['b30']

Section: RELATED WORKS
We review three directions of research related to ours: (i) studies of latent noise in diffusion models, (ii) applications of information geometry in generative modeling, and (iii) geometric formulations for sampling efficiency.  Information geometry in generative models. Lobashev et al. (2025) introduce the Fisher-Rao metric on families p(x|θ) to study phase-like transitions, where θ is a low-dimensional variable parametrizing a microstate x. In contrast, we place the geometry on diffusion's explicit spacetime coordinates z = (x t , t), induced by the denoising posterior p(x 0 |x t ).
Geometric approaches to sampling. Two recent works also formulate diffusion models geometrically to improve sampling efficiency. Das et al. (2023) optimize the forward noising process by following the shortest geodesic between p 0 and p t under the Fisher-Rao metric, assuming p 0 (x 0 ) to be Gaussian. Ghimire et al. (2023) model both the forward and reverse processes as Wasserstein gradient flows. Our contribution differs: we use information geometry (not optimal transport), focus on the reverse process (not the forward), and only require p 0 to admit a density.
this section cite: ['b5', 'b11']

Section: LIMITATIONS
Although our framework defines geodesics between any noisy samples, optimizing between nearly clean ones is numerically unstable because their denoising distributions collapse to Dirac deltas, making Fisher-Rao (via local KL) distances effectively infinite. Therefore, consistent with diffusion practice (Song et al., 2021;Lu et al., 2022), we choose endpoints with non-negligible noise for tractable optimization (details in Appendix G).
The proposed distance metric DiffED (Section 6.2) is considerably slower (details in Appendix G.2) than established image similarity metrics such as LPIPS (Zhang et al., 2018), or SSIM (Wang et al., 2004). Exploring a distillation strategy involving training a separate model trained to predict DiffED is a possible future research direction.
this section cite: ['b33', 'b23', 'b38', 'b35']

Section: CONCLUSION
We proposed a novel perspective on the latent space of diffusion models by viewing it as a (D + 1)dimensional statistical manifold, with the Fisher-Rao metric inducing a geometrical structure. By leveraging the fact that the denoising distributions form an exponential family, we showed that we can tractably estimate geodesics even for high-dimensional image diffusion models. We visualized our methods for image interpolations and demonstrated their utility in molecular transition path sampling.
This work deepens our understanding of the latent space in diffusion models and has the potential to inspire further research, including the development of novel applications of the spacetime geometric framework, such as enhanced sampling techniques.
this section cite: []

Section: References
Ref_id:b0 Title: Information geometry and its applications Year: (2016)
Ref_id:b1 Title: Reverse-time diffusion equation models Year: (1982)
Ref_id:b2 Title: Latent space oddity: On the curvature of deep generative models Year: (2018)
Ref_id:b3 Title: Pulling back information geometry Year: (2022)
Ref_id:b4 Title: A one-way shooting algorithm for transition path sampling of asymmetric barriers Year: (2016)
Ref_id:b5 Title: Image generation with shortest path diffusion Year: (2023)
Ref_id:b6 Title: Riemannian score-based generative modelling Year: (2022)
Ref_id:b7 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b8 Title: Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control Year: (2025)
Ref_id:b9 Title: Doob's Lagrangian: A sample-efficient variational approach to transition path sampling Year: (2024)
Ref_id:b10 Title: Tweedie's formula and selection bias Year: (2011)
Ref_id:b11 Title: Geometry of score based generative models Year: (2023)
Ref_id:b12 Title: Scalable reversible generative models with free-form continuous dynamics Year: (2019)
Ref_id:b13 Title: Generator matching: Generative modeling with arbitrary Markov processes Year: (2025)
Ref_id:b14 Title: Stochastic optimal control for collective variable free sampling of molecular transition paths Year: (2023)
Ref_id:b15 Title: Riemannian diffusion models Year: (2022)
Ref_id:b16 Title: A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines Year: (1989)
Ref_id:b17 Title: Devil is in the details: Density guidance for detail-aware generation with flow models Year: (2025)
Ref_id:b18 Title: Elucidating the design space of diffusionbased generative models Year: (2022)
Ref_id:b19 Title: Analyzing and improving the training dynamics of diffusion models Year: (2024)
Ref_id:b20 Title: Variational diffusion models Year: (2021)
Ref_id:b21 Title: Understanding diffusion objectives as the ELBO with simple data augmentation Year: (2023)
Ref_id:b22 Title: Hessian geometry of latent space in generative models Year: (2025)
Ref_id:b23 Title: Maximum likelihood training for score-based diffusion odes by high order denoising score matching Year: (2022)
Ref_id:b24 Title: Estimating high order gradients of the data distribution by denoising Year: (2021)
Ref_id:b25 Title: Information geometry for the working information theorist. arXiv Year: (2023)
Ref_id:b26 Title: OpenMP application program interface version 3.0 Year: (2008)
Ref_id:b27 Title: Understanding the latent space of diffusion models through the lens of Riemannian geometry Year: (2023)
Ref_id:b28 Title: Action-minimization meets generative modeling: Efficient transition path sampling with the Onsager-Machlup functional Year: (2025)
Ref_id:b29 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b30 Title: A likely geometry of generative models Year: (2025)
Ref_id:b31 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b32 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b33 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b34 Title: Riemannian diffusion Schrödinger bridge Year: (2022)
Ref_id:b35 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b36 Title: Diffusion models: A comprehensive survey of methods and applications Year: (2023)
Ref_id:b37 Title: Probability density geodesics in image diffusion latent space Year: (2025)
Ref_id:b38 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
