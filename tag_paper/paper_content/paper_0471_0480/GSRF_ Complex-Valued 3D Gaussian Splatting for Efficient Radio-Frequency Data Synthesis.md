Title: GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis
Abstract: Synthesizing radio-frequency (RF) data given the transmitter and receiver positions, e.g., received signal strength indicator (RSSI), is critical for wireless networking and sensing applications, such as indoor localization. However, it remains challenging due to complex propagation interactions, including reflection, diffraction, and scattering. State-of-the-art neural radiance field (NeRF)-based methods achieve high-fidelity RF data synthesis but are limited by long training times and high inference latency. We introduce GSRF, a framework that extends 3D Gaussian Splatting (3DGS) from the optical domain to the RF domain, enabling efficient RF data synthesis. GSRF realizes this adaptation through three key innovations: First, it introduces complex-valued 3D Gaussians with a hybrid Fourier-Legendre basis to model directional and phase-dependent radiance. Second, it employs orthographic splatting for efficient ray-Gaussian intersection identification. Third, it incorporates a complex-valued ray tracing algorithm, executed on RF-customized CUDA kernels and grounded in wavefront propagation principles, to synthesize RF data in real time. Evaluated across various RF technologies, GSRF preserves highfidelity RF data synthesis while achieving significant improvements in training efficiency, shorter training time, and reduced inference latency.† This work was partially done when Kang Yang was a PhD student in Dr. Wan Du's group at UC Merced.

Section: Introduction
Wireless networks, e.g., WiFi and Fifth Generation (5G) cellular networks, are increasingly tasked with supporting both communication and sensing applications through deep learning (DL) models, including indoor localization [1,2,3]. However, training these DL models requires large-scale radio-frequency (RF) datasets, e.g., received signal strength indicator (RSSI) measurements across different transmitter and receiver positions within a 3D space, which are typically collected through site surveys. These site surveys involve labor-intensive and time-consuming RF signal measurements across numerous transmitter-receiver locations [4,5,6].
Inspired by the success of generative models in computer vision [7,8,9,10], a natural alternative approach is to synthesize RF data through propagation modeling, which computes the received RF signal at a receiver given a transmitter emitting signals from a specific position [11]. However, generating high-fidelity RF data is challenging due to complex propagation interactions between RF signals and surrounding objects, including reflection, diffraction, and scattering.
Neural Radiance Field (NeRF) [7]-based methods [12,13] address these challenges by extending NeRF to the RF domain, achieving state-of-the-art fidelity in RF data synthesis. These NeRFbased methods adopt continuous RF scene representations to effectively model complex RF interactions. However, their stochastic sampling process and Multi-layer Perceptron (MLP) optimization are computationally intensive and slow, limiting real-time applicability. Efficient training and inference in the RF domain are crucial for applications such as real-time localization and tracking [14,6]. This paper proposes GSRF, an efficient RF data synthesis framework that extends 3D Gaussian Splatting (3DGS) [8,15], developed for real-time novel view synthesis, to the RF domain. However, this adaptation introduces challenges due to inherent differences between visible light and RF signals:
• (i) Directional and Phase Modeling. In 3DGS [8,16], the color attribute of a Gaussian distribution is parameterized by spherical harmonics (SH) coefficients [17,8] to capture directional variations caused by optical propagation effects such as reflections and shading. In contrast, RF signals with centimeter-scale wavelengths exhibit complex phenomena such as diffraction [18] and phasedependent interference (constructive and destructive), which SH coefficients struggle to capture [19].
• (ii) Data Capture Mechanism. In visible light, images are captured by camera sensors (e.g., CMOS or CCD) on a 2D image plane, allowing splatting through classical transformation matrices that project 3D Gaussians onto the plane to identify ray-Gaussian intersections. In contrast, RF signals are collected by antenna arrays over a spherical region centered at the RF antenna. This fundamental difference makes splatting algorithms designed for visible light unsuitable for the RF domain.
• (iii) Rendering Algorithm. In 3DGS, point-based rendering algorithm aggregates amplitude-based attributes, e.g., color, to compute pixel values along each ray. In contrast, RF signal synthesis needs to consider both amplitude and phase to model interference patterns. This necessitates a complex-valued rendering algorithm, along with CUDA kernels that jointly process amplitude and phase information.
By tackling the three challenges above, we make the following key contributions:
• Fourier-Legendre Radiance Fields. A scene is represented using 3D Gaussian distributions, each characterized by four attributes: a mean and covariance matrix, along with two RF-specific attributes, which are complex-valued RF radiance and transmittance. The directional radiance is modeled using a Fourier-Legendre Expansion (FLE) [20]. FLE leverages Fourier basis functions for the azimuthal angle α and Legendre polynomials for the elevation angle β, with complex coefficients c ml ∈ C encoding both amplitude and phase. Additionally, the complex-valued transmittance models signal amplitude attenuation and phase shifts as the RF signal propagates through a Gaussian.
• Orthographic Splatting. To determine ray-Gaussian intersections, GSRF introduces an orthographic splatting method for the RF domain. GSRF operates on the Ray Emitting Spherical Surface (RESS), a spherical region where RF signals are captured. Each 3D Gaussian is then splatted onto this region via orthographic projection, enabling identification of intersecting Gaussians for each ray.
• Complex-Valued Ray Tracing. GSRF incorporates a complex-valued ray tracing algorithm for RF signals, executed on RF-customized CUDA kernels. Building on the Huygens-Fresnel principle [21], which states that each point on a wavefront acts as a source of secondary wavelets, GSRF models each Gaussian as an RF source. GSRF emits rays from the RESS, identifies intersecting Gaussians through the adapted splatting method, and employs a complex-valued ray tracing algorithm to jointly process amplitude and phase attributes along each ray, computing the received RF signal data.
• GSRF is trained with an RF-customized loss function derived from both time and frequency domains using 2D Fourier transforms to capture the intricate propagation characteristics of RF signals.
We evaluate GSRF on various RF technologies, including radio-frequency identification (RFID), Bluetooth Low Energy (BLE), and 5G networks, to synthesize different types of RF data, including RSSI, spatial spectra, and complex-valued channel state information (CSI). Results show that GSRF achieves significantly higher efficiency than existing methods, with improvements in training data efficiency, training time, and inference latency. We release our code at this GitHub repository.
2 Preliminaries RF Signal Propagation Characteristics. Wireless systems, such as WiFi, rely on RF signals propagating between transmitters and receivers [22,23]. A transmitted signal can be represented as:
s(t) = Ae j(2πfct+θ) ,(1)
where A is the amplitude, f c is the carrier frequency (e.g., 2.4 GHz), and θ is the initial phase. As the signal propagates through the scene, it encounters obstacles that cause reflections, diffraction, and scattering, resulting in multiple propagation paths. The received signal is the sum of these paths:
r(t) = N i=1 A i e jϕi s (t -τ i ) , ϕ i = 2πf c τ i + θ i , τ i = d i c (2
)
where N is the number of paths, c is the RF signal speed, A i is the attenuated amplitude, ϕ i is the phase shift, τ i is the time delay of path length d i , and θ i is the phase change from reflections.
The phase greatly affects the received signal, as illustrated in the following example. For two paths with lengths d 1 = 3 m and d 2 = 3.0625 m at a carrier frequency of f c = 2.4 GHz, the corresponding delays are τ 1 = 10 ns and τ 2 = 10.208 ns. The phase shifts are ϕ 1 = 0 and ϕ 2 = π, resulting in a phase difference of ∆ϕ = π, which causes destructive interference, i.e., the two signals cancel each other out, leading to a reduction or complete loss of signal strength. Conversely, when ∆ϕ ≈ 0, constructive interference occurs, amplifying the signal [22]. Therefore, synthesizing RF data requires modeling these amplitude and phase interactions across all paths.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b6', 'b11', 'b12', 'b13', 'b5', 'b7', 'b14', 'b7', 'b15', 'b7', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b20']

Section: 3D Gaussian Splatting (3DGS).
It is a real-time rendering technique for novel view synthesis in 3D scenes [8]. It represents a 3D scene as a collection of 3D Gaussian ellipsoids {ζ 1 , . . . , ζ K }, where each Gaussian primitive ζ k is defined by a 3D Gaussian distribution:
G k (x; µ k , Σ k ) = exp - 1 2 (x -µ k ) T Σ -1 k (x -µ k ) ,(3)
where µ k ∈ R 3 is the center position and Σ k ∈ R 3×3 is the covariance matrix. It is decomposed as:
Σ k = R k S k S T k R T
k , where R k and S k are learnable rotation and scaling matrices that ensure positive semi-definiteness [8]. Each Gaussian also includes an opacity term ρ k ∈ [0, 1] and SH coefficients sh k ∈ R d , making each Gaussian primitive represented as:
ζ k = (µ k , R k , S k , ρ k , sh k ) .
To render an image, 3DGS projects these 3D Gaussians onto a 2D image plane, forming 2D Gaussians:
G 2D k r; µ 2D k , Σ 2D k with µ 2D k = π 2D (µ k ) , Σ 2D k = JW Σ k W T J T , (4
)
where J is the Jacobian of the projective transformation, and W is the world-to-camera transformation matrix [8]. The pixel color Ĉ (r) at location r ∈ R 2 is computed via α-blending:
Ĉ (r) = k∈Sr ω 2D k (r) c (sh k , r) ,(5)
where S r ⊆ {1, . . . , K} is the subset of indices of Gaussians that contribute to pixel r. The term ω 2D k (r) represents the contribution of each Gaussian, computed as:
ω 2D k (r) = ρ k G 2D k r; µ 2D k , Σ 2D k k-1 j=1 1 -ρ j G 2D j r; µ 2D j , Σ 2D j ,(6)
where the Gaussians are ordered by increasing depth (i.e., from front to back) to ensure correct rendering. Finally, c (sh k , r) is the color decoded from the SH coefficients sh k .
this section cite: ['b7', 'b7', 'b7']

Section: Related Work
Conventional RF data synthesis methods include simulations [24,25,26], empirical models [22,27,28], and physics-unaware DL models [29,30,31], but all suffer from low modeling fidelity due to inherent limitations. Simulations require accurate scene Computer-Aided Design (CAD) models, which are often unavailable. Empirical models oversimplify propagation with limited parameters, predicting only coarse signal power. Physics-unaware DL models map inputs to labels but fail to capture the underlying physics of RF propagation. NeRF-based methods [12,13,32] introduce voxel-based scene representations to capture scene impact on RF signal propagation and employ ray tracing algorithms to achieve state-of-the-art fidelity in RF data synthesis. However, they suffer from low efficiency, requiring long training times and exhibiting high inference latency. This work proposes a 3DGS-based method to achieve high training and inference efficiency.
Two recent works, RF-3DGS [33] and WRF-GS [34], propose 3DGS-inspired techniques for RF data synthesis, yet both face limitations. RF-3DGS [33] employs a two-stage training process to learn scene representations using Gaussian primitives defined by mean, covariance, opacity, and path loss. First, optical 3DGS optimizes mean, covariance, and opacity from visual images, then these parameters are fixed to train path loss with RF data. However, merging visible light and RF signals is challenging due to their distinct properties. Moreover, visual data is often unavailable in RF domains.
WRF-GS [34] assigns each Gaussian four attributes: mean, covariance, radiance, and attenuation. It adopts a NeRF-inspired approach to learn radiance and attenuation by optimizing a large MLP with each Gaussian's position as input. This dependence on a computationally intensive MLP results in inefficiency and introduces NeRF-like bottlenecks: (i) dense querying of the MLPs for attribute prediction during training and inference, (ii) expensive backpropagation through deep networks for every Gaussian update, which scales poorly with scene complexity, and (iii) high inference latency due to per-query MLP evaluations (e.g., for novel transmitter positions). In contrast, GSRF eliminates MLP regressors entirely by directly optimizing per-Gaussian attributes as learnable parameters. Combined with Fourier-Legendre radiance fields, orthographic splatting, and complex-valued ray tracing, this design achieves faster training and inference compared to WRF-GS.
WRF-GS+ is an extension of WRF-GS [34]. It introduces deformable Gaussians that decouple static components (e.g., path loss) and dynamic components (e.g., multipath) via learned offsets, thereby improving synthesis quality and mitigating the inefficiencies of WRF-GS's MLP-based attributes. While effective, this approach remains distinct from GSRF, which offers a unified, complexvalued, MLP-free pipeline for RF propagation modeling; nevertheless, deformable mechanisms could be explored in future extensions of our framework.
this section cite: ['b22', 'b23', 'b24', 'b20', 'b25', 'b26', 'b27', 'b28', 'b29', 'b11', 'b12', 'b30', 'b31', 'b32', 'b31', 'b32', 'b32']

Section: Methodology

this section cite: []

Section: Problem Formulation
Given a transmitter at a fixed position emitting RF signals (e.g., a WiFi router) and a receiver (e.g., a smartphone) distributed throughout a scene, the objective is to synthesize the received RF data. Formally, for a transmitter located at t = (x tx , y tx , z tx ) and a set of receiver positions {r i } N i=1 , where r i = (x rx,i , y rx,i , z rx,i ), the goal is to estimate a model with parameters θ that synthesizes the received RF data S i at each receiver r i :
θ * = argmax θ p {S i } N i=1 | t, {r i } N i=1 , θ ,(7)
where S i ∈ C represents the received complex-valued RF data at receiver r i , encapsulating both amplitude and phase. For specific RF technologies or applications, S i may represent a scalar signal power S i ∈ R or a spatial spectrum S i ∈ R Naz×Nel over azimuth α and elevation β angles. For example, with a one-degree angular resolution, we have N az = 360 and N el = 180.
Model Overview. Figure 1 illustrates the GSRF. First, the scene is represented using 3D Gaussian distributions, each characterized by a mean µ k ∈ R 3 , a covariance matrix Σ k ∈ R 3×3 , and two RFspecific complex-valued attributes: radiance ψ k ∈ C and transmittance ρ k ∈ C. To initialize the mean and covariance matrix, the scene is partitioned into equal-sized cubes, deriving initial scene point clouds without Structure-from-Motion (SfM) algorithms [35,36], which are inapplicable to the RF domain. Next, each 3D Gaussian is projected onto the receiver's receiving region, using orthographic projection to efficiently identify ray-Gaussian intersections. For each ray, intersecting Gaussians are sorted by depth, and a complex-valued ray tracing algorithm is applied to compute the received signal. Model optimization is performed by minimizing the loss function. Explicit gradients are computed to update the primitives via stochastic gradient descent, adjusting parameters (µ k , Σ k , ψ k , ρ k ) and refining primitive density through gradient-driven cloning, splitting, or removal.
this section cite: ['b33', 'b34']

Section: Fourier-Legendre Radiance Fields
Each Gaussian primitive in GSRF is represented as a tuple:
ζ k = (µ k , R k , S k , ψ k , ρ k ) with Σ k = R k S k S T k R T k .(8)
The pair (µ k , R k , S k ) defines a 3D Gaussian distribution resembling an ellipsoid, representing a probability distribution in 3D space. Its probability density function (PDF) is given by Equation (3).
The transmittance ρ k ∈ C models the effect of an RF signal passing through the k-th Gaussian, resulting in an amplitude reduction |ρ k | and a phase shift ∠ρ k . According to Maxwell's equations [37], transmittance depends on the material properties at the Gaussian's location µ k . Therefore, ρ k primarily captures the physical interaction of the RF signal with the medium.
The radiance ψ k ∈ C represents the complex-valued RF signal emitted by the k-th Gaussian. To model its directional dependency, ψ k is defined using a Fourier-Legendre Expansion (FLE) [20], which leverages Fourier basis functions for the azimuthal angle α and Legendre polynomials for the elevation angle β. This approach is physically grounded in the Huygens-Fresnel principle [21], which posits that each point on a wavefront, such as the k-th Gaussian at position µ k , acts as a source of secondary spherical wavelets. The emitted RF signal is modeled as a solution to the wave equation in spherical coordinates, where spherical harmonics-comprising Fourier functions e imα and associated Legendre polynomials P m l (cos β)-form a complete basis for representing directional wave fields on the unit sphere. Specifically, for a direction (α, β), the radiance is expressed as:
ψ k (α, β) = L l=0 l m=-l c (k) ml e imα P m l (cos β) ,(9)
where c (k) ml ∈ C are complex coefficients encoding the amplitude and phase of the radiance for the k-th Gaussian. This representation effectively captures phase-dependent interference crucial, as the separation of α and β aligns with their geometric roles in spherical coordinates, while the complex coefficients model the interference effects stemming from the wave nature of RF signals.
this section cite: ['b2', 'b35', 'b18', 'b19']

Section: Orthographic Splatting
For a receiver positioned at r = (x rx , y rx , z rx ), rays are emitted to sample the RF signal across various directions around the receiver r. Each ray is parameterized as:
γ(d) = r + dv, d ≥ r rx ,(10)
where d is the distance along the ray from the receiver, and v is the unit direction vector. Therefore, rays are emitted from the Ray Emitting Spherical Surface (RESS), which is a sphere centered at r with radius r rx , and extend outward. For a one-degree angular resolution, N az = 360 and N el = 180, resulting in a total of 360 × 180 rays being emitted, covering all directions around the receiver.
this section cite: []

Section: 2D Receiving RF Plane.
To enable splatting in the RF domain, where an image plane is absent, we map the RESS onto a 2D RF plane. Consider a point p = (x, y, z) ∈ R 3 on the RESS, satisfying ∥p-r∥ = r rx . We transform the Cartesian coordinates of p into spherical coordinates (ζ, α, β), where ζ is the radial distance, α ∈ [0, 2π) is the azimuthal angle, and β ∈ [-π/2, π/2] is the elevation angle:
ζ = x 2 + y 2 + z 2 = r rx , α = arctan 2(y, x), β = π 2 -arccos z r rx .(11)
We then project α and β onto a 2D grid with one-degree resolution, defined as:
u = α • 180 π , v = β • 180 π + 90 ,(12)
where ⌊•⌋ denotes the floor function. The resulting coordinates (u, v) define the 2D RF plane.
Splatting Process. Each 3D Gaussian, with mean µ k ∈ R 3 and covariance Σ k ∈ R 3×3 , is projected onto the 2D RF plane to identify ray-Gaussian intersections. The unit direction vector from the receiver position r ∈ R 3 to the k-th Gaussian center µ k is given by:
vk = µ k -r ∥µ k -r∥ 2 . (13
)
The projected center µ 2D k = (u k , v k ) is computed from Equations ( 11) and (12), with the vector vk as input. The 3D covariance Σ k is projected onto the 2D plane as Σ 2D k = JΣ k J T , where J is the Jacobian matrix, and the 2D spread is approximated by radius r k = 3 √ λ max , with λ max as the largest eigenvalue of Σ 2D k . Rays at points (u, v) intersect the Gaussian if:
(u -u k ) 2 + (v -v k ) 2 ≤ r k .
this section cite: ['b11']

Section: Complex-Valued Ray Tracing Algorithm
The received signal S ∈ C for a ray is computed by aggregating the contributions from all intersecting Gaussians, considering their geometric influence, radiance, and transmittance. The intersecting Gaussians are sorted in ascending order of their distance from the receiver along the ray path to ensure correct accumulation of transmittance effects. The received signal is computed as:
S = Kintr k=1 G k (x rep,k ; µ k , Σ k ) Gaussian weight • |ψ k | e j∠ψ k Complex radiance • k-1 m=1 |ρ m | e j∠ρm Cumulative transmittance (14
)
where K intr is the number of Gaussians intersecting the ray, and G k (x; µ k , Σ k ) is the probability density function of the k-th Gaussian, evaluated at the representative intersection point x rep,k ∈ R 3 , which is the midpoint of the intersection points between the ray trajectory and the ellipsoid defined by the Gaussian's mean µ k ∈ R 3 and covariance Σ k ∈ R 3×3 . The term G k (x rep,k ; µ k , Σ k ) weights the radiance based on the Gaussian's density at the intersection point, while the product term accumulates both amplitude attenuation and phase shifts from all preceding Gaussians, capturing both amplitude reduction and phase shifts during propagation. The detail of proor is provided in Appendix A.
this section cite: []

Section: Loss Function.
The loss function is designed based on the receiver antenna type.
ANTENNA ARRAY. For a receiver equipped with antenna arrays, the signal power across all directions is represented as a ground-truth spatial spectrum matrix S ∈ R Naz×Nel , spanning N az azimuth and N el elevation angles. The predicted spatial spectrum is denoted as Ŝ ∈ R Naz×Nel . The loss function L combines the L 1 loss, the Structural Similarity Index Measure (SSIM) loss, and a Fourier-based loss:
L = (1 -λ 1 -λ 2 ) L 1 + λ 1 L SSIM + λ 2 L Fourier ,(15)
where
L 1 = 1 NazNel Naz u=1
Nel v=1 Ŝ(u, v) -S(u, v) measures the average absolute difference between the predicted and ground-truth spectra. The term L SSIM captures spatial RF pattern similarity across directions. The term L Fourier quantifies the difference in the frequency domain:
L Fourier = 1 N az N el fu,fv F Ŝ (f u , f v ) -F (S) (f u , f v ) 2 ,(16)
where F Ŝ (f u , f v ) and F (S) (f u , f v ) are the 2D Fourier transforms of the predicted and groundtruth spectra, respectively, with f u and f v representing the frequency indices in the azimuth and elevation dimensions. This term promotes consistency in the frequency domain, which is important for learning RF propagation behavior. The squared magnitude penalizes discrepancies in both amplitude and phase, enhancing the fidelity of synthesized signals.
SINGLE ANTENNA. For a receiver equipped with a single antenna, the ground-truth received signal S represents either a real-valued power measurement or a complex-valued signal encompassing both amplitude and phase information. The synthesized signal Ŝ is computed as Ŝ = Naz u=1 Nel v=1 Ŝu,v , where Ŝu,v denotes the synthesized signal contribution from the ray at azimuth index u ∈ {1, . . . , N az } and elevation index v ∈ {1, . . . , N el }. The loss function L is defined as L = Ŝ -S 1 if S is real-valued (RF signal power), and as L = Ŝ -S 2 2 if S is complexvalued, penalizing both amplitude and phase errors.
Gradient-Based Gaussian Primitive Optimization. GSRF initializes the number of Gaussians and their primitives based on the scene's point clouds, which are obtained by partitioning the scene into equal-sized cubes. After calculating the loss function, the optimization of Gaussian primitives is performed through gradient-based strategies, as detailed in Appendix B.
Fast Differentiable RF Signal Renderer for Gaussians. In GSRF, we develop two CUDA kernels to enable efficient forward and backward computations for differentiable RF signal synthesis using Gaussian primitives. Implementation details of the CUDA kernels are provided in Appendix C. To reduce computational overhead, gradients are explicitly calculated as described in Appendix D.
this section cite: []

Section: Experiments
Our method is implemented in PyTorch with CUDA. Further implementation details and hyperparameter settings are provided in Appendix E, and additional experiments are presented in Appendix F.  TASK. Given a transmitter sending RF signals at location (x tx , y tx , z tx ), the goal is to synthesize the spatial spectrum received by the receiver (equipped with an antenna array). The spatial spectrum, represented as a 360 × 90 matrix, captures the signal power from all directions around the receiver, covering azimuth and elevation angles at a one-degree resolution. The elevation angle is limited to 90 • as only the front hemisphere of the antenna array is considered [12].
this section cite: ['b11']

Section: RFID Spatial Spectrum Synthesis
DATASET. The publicly released RFID dataset from NeRF 2 [12], collected in real-world indoor environments, is employed. It contains 6,123 transmitter (RFID tag) locations and their corresponding spatial spectra, received by a receiver equipped with a 4 × 4 antenna array operating at the 915 MHz frequency band. The dataset is randomly split by default into 70% for training and 30% for testing. METRICS. We employ the two metrics: • (i) Mean Squared Error (MSE)↓: This metric calculates the average of the squared differences in signal power between the synthesized spectrum and the ground truth for each entry.
• (ii) Peak Signal-to-Noise Ratio (PSNR, in dB)↑: Treating the spatial spectrum as an image, PSNR measures structural similarity, with higher values indicating better quality. BASELINES. We compare GSRF with NeRF 2 [12] and WRF-GS [34]. Other simulation-based or physics-unaware DL-based methods, such as MATLAB simulation [26], DCGAN [39], and VAE [30], perform worse on the same RFID dataset [12] compared to NeRF 2 .
this section cite: ['b11', 'b11', 'b32', 'b24', 'b37', 'b28', 'b11']

Section: Ϭ͘Ϭ

this section cite: []

Section: Overall Performance.
To evaluate GSRF's performance in scenarios with insufficient data, we randomly select 220 instances from the training dataset instead of using the full training data. This creates a sparse dataset with a measurement density of 0.8 measurements/ft 3 . Figure 2 presents the real-collected spatial spectra for four randomly selected transmitter positions (first row), alongside those generated by baseline models.
Visually, the spectra synthesized by GSRF more closely match the ground truth compared to those by NeRF 2 . Figure 3 then shows the Cumulative Distribution Function (CDF) of the two metric scores on the testing data. GSRF achieves median improvements of 21.2% in PSNR and 56.4% in MSE over NeRF 2 , while outperforming WRF-GS by 5.7% and 19.3%, respectively. This superiority stems from two advantages: First, our complex-valued Gaussian representation explicitly models phase interactions throughout the architecture, which is critical for RF signal propagation, whereas WRF-GS relies on real-valued Gaussian primitives. Second, the Fourier-Legendre radiance basis in GSRF provides directional resolution beyond the spherical harmonics used in WRF-GS, enabling finer capture of diffraction and scattering effects. These innovations allow GSRF to learn more physically accurate scene representations even from sparse measurements. Ϭ ϭϬϬ ϮϬϬ ϯϬϬ ϰϬϬ /ŶĨĞƌĞŶĐĞƚŝŵĞ;ŵƐͿ Ϭ͘Ϭ Ϭ͘Ϯ Ϭ͘ϰ Ϭ͘ϲ Ϭ͘ϴ ϭ͘Ϭ & EĞZ& 2 tZ&Ͳ'' ^Z& Figure 5: Test times for spectrum synthesis.
Training & Inference Efficiency. Training time is measured by running each method on a computer equipped with GeForce RTX 3080Ti GPU.
Inference time for each model is also recorded. Figure 4 illustrates that our method achieves convergence in 0.27 hours, which is 18.56× faster than NeRF 2 (5.01 hours) and 5.96× faster than WRF-GS (1.61 hours). For inference latency, as shown in Figure 5, our method synthesizes spatial spectra in 4.18 ms, yielding an 84.39× speedup over NeRF 2 (352.73 ms) and a 1.81× speedup over WRF-GS (7.58 ms). This acceleration is due to: First, our explicit Gaussian representation with FLE eliminates the need for MLP queries, which are required in both NeRF 2 and WRF-GS. Although WRF-GS also employs Gaussians, it still relies on a large MLP to query each Gaussian primitive's values using the Gaussian mean as input.
Second, the hybrid CUDA-based ray tracer optimizes complex-valued operations through explicit gradient computation. These optimizations enable our method to support real-time applications, e.g., sub-millisecond tracking in 5G networks. Measurement Density. Figure 6 compares the MSE of GSRF (trained on the dataset with a density of 0.8 measurements/ft 3 ) to NeRF 2 (varying densities ranging from 0.8 to 15.5). The densities are obtained by random sampling from the original 70% training set. GSRF achieves a comparable MSE to NeRF 2 trained on the dataset with a density of 7.8. This indicates that GSRF requires 9.8× less training data to achieve comparable spectrum synthesis quality to NeRF 2 . The improvement arises from GSRF's 3D Gaussian-based scene representation, which focuses on object features rather than empty space, making it more efficient than NeRF 2 's voxel-based fields. More results for WRF-GS [34] are provided in Appendix F.4.
this section cite: ['b32']

Section: 5G Complex-Valued CSI Synthesis
TASK. This task demonstrates GSRF's effectiveness in synthesising complex-valued signals. In 5G Orthogonal Frequency-Division Multiplexing (OFDM) modulation, downlink and uplink operate on different frequency bands [40]. Given uplink complex-valued CSI, the objective is to predict the downlink CSI. The rationale for this task lies in the shared physical propagation environment, which correlates uplink and downlink CSI [41]. Furthermore, uplink CSI can serve as a transmitter position indicator due to its uniqueness across different positions [12,42].
DATASET. The public Argos dataset [43] is employed. It is collected in outdoor environments, where a base station with 104 antennas measures CSI from signals sent by clients. Each CSI measurement includes 52 subcarriers. Following prior works [12,30,41], the first 26 subcarriers are treated as the uplink channel, and the remaining N = 26 as the downlink channel. The dataset contains 100,000 measurements and is randomly split into 70% for training and 30% for testing.
METRICS. We adopt the Signal-to-Noise Ratio (SNR) [30] to quantify synthesized CSI quality:
SNR = -10 log 10 ∥S -Ŝ∥ 2 2 • ∥S∥ -2 2 , (17
)
where S, Ŝ ∈ C N are the ground truth and synthesized CSI vectors, respectively.
BASELINES. We compare GSRF with NeRF 2 [12] and include two additional baselines:
• R2F2 [41]: Extracts the number of propagation paths and each path's parameters to estimate CSI.
• FIRE [30]: Uses the VAE [44] to predict the downlink CSI by learning the latent distribution.
Overall Performance. To demonstrate GSRF's efficiency, it is trained on only 30% of the raw training data portion, while the baselines are trained on the full training set. All methods are evaluated on the same testing data. Since GSRF requires three-dimensional transmitter locations, we train an autoencoder [45] using 26 uplink subcarriers as input to reconstruct them. The autoencoder's hidden layer is set to three dimensions, representing the transmitter locations. Figure 7 illustrates a prediction example from GSRF, where the two curves (blue and red) nearly overlap, demonstrating its high prediction accuracy. Figure 8 quantifies the SNR of the four methods. GSRF achieves a mean SNR of 20.99 dB, outperforming R2F2 and FIRE. Additionally, GSRF achieves comparable CSI synthesis quality to NeRF 2 while using 3× less training data, highlighting its training efficiency. ZϮ&Ϯ &/Z EĞZ& 2 '^Z& Ϭ ϱ ϭϬ ϭϱ ϮϬ Ϯϱ ^EZ;ĚͿ Figure 8: Channel CSI prediction SNR.
It is worth noting that NeRF 2 also performs phaseaware modeling through an MLP that regresses amplitude and phase from voxel and transmitter coordinates. While this enables high-quality CSI synthesis, the volumetric ray-based querying of the MLP introduces significant computational cost during both training and inference. In contrast, GSRF integrates phase modeling directly into Gaussian primitives via Fourier-Legendre basis expansion, avoiding MLP regressors entirely. This explicit, complex-valued representation allows efficient gradient updates with lightweight CUDA operations, leading to faster convergence and inference without sacrificing accuracy. Thus, the comparable SNR to NeRF 2 does not diminish the contribution of GSRF, but instead underscores its ability to achieve phase-aware synthesis with substantially greater efficiency.
this section cite: ['b38', 'b39', 'b11', 'b40', 'b41', 'b11', 'b28', 'b39', 'b28', 'b11', 'b39', 'b28', 'b42', 'b43']

Section: BLE Real-Valued RSSI Synthesis
TASK. This task verifies that GSRF supports single-antenna setups for capturing a single realvalued RSSI. Given a transmitter (BLE node) sending signals from location (x tx , y tx , z tx ), the goal is to synthesize the RSSI (in dBm) received by a receiver (BLE gateway with a single antenna). The measured RSSI represents the aggregate signal power from all directions [12]. Additionally, we conduct a fingerprint-based localization application to demonstrate GSRF's sensing advantages.
DATASET. The public BLE dataset [12], collected in an elderly nursing home, is employed. Twentyone receivers operating at 2.4 GHz frequency band to capture RSSI. The dataset contains 6,000 transmitter positions, each paired with a 21-dimensional tuple of RSSI readings from the 21 receivers. Ϭ Ϯ ϰ ϲ ϴ ƌƌŽƌ;ŵͿ Ϭ͘Ϭ Ϭ͘Ϯ Ϭ͘ϰ Ϭ͘ϲ Ϭ͘ϴ ϭ͘Ϭ & EĞZ& 2 '^Z& Figure 10: BLE-based localization error. METRICS. RSSI synthesis error is the absolute difference between predictions and ground truth.
BASELINES. We compare GSRF with NeRF 2 . Other empirical and DL methods, e.g., MRI [46], are excluded because they perform worse than NeRF 2 on the same testing dataset [12].
Overall Performance. To evaluate the performance of GSRF in scenarios with sufficient data, both models (GSRF and NeRF 2 ) are trained on the full training dataset. Figure 9 indicates that GSRF achieves an average RSSI error of 4.09 dBm, compared to NeRF 2 's 6.09 dBm. This represents a 32.79% improvement, highlighting GSRF's effectiveness fo single-antenna receivers. The performance gain stems from GSRF's flexible 3D Gaussian-based explicit scene representation, which efficiently utilizes training data by focusing on objects rather than large empty space and aligning with object geometry. Training and inference times are reported in Appendix F.2, Figures 11 and 12. These results demonstrate that GSRF achieves a 15.82-fold decrease in training time and a 78.98-fold reduction in inference time for RSSI synthesis. Additional results for WRF-GS [34] are in Appendix F.2.
this section cite: ['b11', 'b11', 'b44', 'b11', 'b32']

Section: BLE-Based Localization.
In fingerprinting-based localization, the RSSI value from an unknown transmitter queries a fingerprint database containing pairs of transmitter positions and corresponding RSSI values. The K Nearest Neighbors (KNN) identifies the K nearest matches and estimates the unknown transmitter position as the average of these K positions [29]. We generate synthetic datasets using GSRF and NeRF 2 to build the fingerprint database for comparison. Figure 10 shows that GSRF outperforms NeRF 2 by 31.40% on average. This improvement in localization accuracy demonstrates that high-fidelity synthesized databases generated by GSRF enhance localization applications, eliminating the need for time-consuming and labor-intensive manually collected fingerprinting databases.
this section cite: ['b27']

Section: Ablation Study
We evaluate our design components using the RFID dataset introduced in Section 5.1. All versions are trained on the full training set, with results presented in Table 1. FLE-Based Radiance. We employ FLE coefficients to model the directional radiance of each Gaussian in GSRF, unlike the SH coefficients used in 3DGS for visible light rendering [8]. Experimental comparisons (first vs. last column) show that our FLE coefficients achieve better spectrum synthesis compared to SH coefficients. This improvement arises from FLE's ability to capture intricate RF signal interactions, e.g., phase-dependent interference, while SH is more suitable for smooth optical functions. Both FLE and SH are implemented with a degree of L = 3, resulting in 16 coefficients each. Thus, GSRF enhances spatial spectrum synthesis by leveraging FLE's capability to handle complex-valued radiance fields.
this section cite: ['b7']

Section: Phase Information.
Each Gaussian primitive represents radiance and transmittance as complexvalued attributes to capture RF signal propagation effects, such as constructive and destructive interference. Removing the phase channel while retaining only the amplitude results in a 8.37% reduction in PSNR compared to the full model with phase inclusion (second vs. last column). This performance drop underscores the importance of phase information in synthesizing RF signal data.
Fourier Loss L Fourier . We evaluate the impact of Fourier loss L Fourier by comparing the model without this loss term (third column) against the full model (last column). Removing Fourier loss L Fourier results in a 6.28% reduction in PSNR, indicating that frequency-domain alignment enhances the fidelity of the synthesized spatial spectra. Incorporating L Fourier enables our model to better preserve frequency-domain properties, enhancing overall RF data synthesis quality.
this section cite: []

Section: Discussion
Despite its advancements, our method has two main limitations. It achieves efficient RF signal synthesis when training data is available for a specific scene but lacks spatial generality for zero-shot inference in unseen environments. It is also optimized for static settings: when the scene changes (e.g., moving obstacles or structural modifications), retraining or fine-tuning is required, limiting temporal adaptation. To address these issues, we outline two complementary directions: improving spatial generality across environments and enabling temporal adaptability to dynamic scenes.
For spatial generality, future work will explore pre-training GSRF on large and diverse multi-scene RF datasets to learn transferable priors that capture common propagation patterns across environments. This could involve designing domain-general encoders that disentangle scene-invariant propagation features (e.g., free-space loss, reflection/diffraction signatures) from scene-specific geometry, and leveraging domain-adaptation strategies to enable rapid adaptation to new environments with only a few samples. Another promising direction is hierarchical Gaussian representations, where global Gaussians encode universal priors while local Gaussians specialize to environment-specific details.
For temporal adaptability, we propose a deformable 3DGS extension that supports dynamic RF scene rendering: a shared set of complex-valued 3D Gaussians represents the baseline RF field, while a lightweight deformation module models time-varying changes without full retraining.
A spatiotemporal encoder factors the 4D space-time volume (x, y, z, t) into six compact 2D planes (xy, yz, xz, xt, yt, zt), reducing parameter complexity from R 4 ×C to 6R 2 ×C and preserving locality for efficient CUDA querying; spatial planes capture multipath effects (reflection/diffraction), and temporal planes capture motion-induced changes. A small multi-head decoder then predicts per-Gaussian deformations (position/rotation/scale), while complex-valued attributes (e.g., radiance ψ k , transmittance ρ k ) are preserved for phase-aware modeling. Together, these directions aim to make GSRF both broadly generalizable and responsive to real-world dynamics.
this section cite: []

Section: Conclusion
This paper introduces GSRF, a novel complex-valued 3DGS-based framework for efficient RF signal data synthesis. We customize 3D Gaussian primitives with complex-valued attributes and integrate an RF-specific CUDA-enabled ray tracing algorithm for efficient scene representation and received signal computation. Extensive experiments validate GSRF's efficiency, demonstrating significant improvements in training and inference speed while maintaining high-fidelity RF data synthesis.
this section cite: []

Section: References
Ref_id:b0 Title: WiFi sensing with channel state information: A survey Year: (2019)
Ref_id:b1 Title: WiTAG: Seamless WiFi Backscater Communication Year: (2020)
Ref_id:b2 Title: RALoRa: Rateless-Enabled Link Adaptation for LoRa Networking Year: (2024)
Ref_id:b3 Title: Site Survey and Radio Frequency Planning for the Deployment of Next Generation WLAN Year: (2018)
Ref_id:b4 Title: Site Survey Guidelines for WLAN Deployment Year: (2023)
Ref_id:b5 Title: Generative Diffusion Model-Assisted Efficient Fingerprinting for in-Orchard Localization Year: (2025)
Ref_id:b6 Title: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis Year: (2020)
Ref_id:b7 Title: 3D Gaussian Splatting for Real-Time Radiance Field Rendering Year: ()
Ref_id:b8 Title: Denoising Diffusion Probabilistic Models Year: (2020)
Ref_id:b9 Title: Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy Year: (2021)
Ref_id:b10 Title: Link Quality Modeling for LoRa Networks in Orchards Year: ()
Ref_id:b11 Title: NeRF 2 : Neural Radio-Frequency Radiance Fields Year: ()
Ref_id:b12 Title: NeWRF: A Deep Learning Framework for Wireless Radiation Field Reconstruction and Channel Prediction Year: (2024)
Ref_id:b13 Title: NeRF-VINS: A Real-time Neural Radiance Field Mapbased Visual-Inertial Navigation System Year: ()
Ref_id:b14 Title: Qi Tian, and Xinggang Wang. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering Year: ()
Ref_id:b15 Title: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Year: ()
Ref_id:b16 Title: A Unifm Geometrical Theory of Diffraction for an Edge in a Perfectly Conducting Surface Year: (1974)
Ref_id:b17 Title: Using Spherical Harmonics for Modeling Antenna Patterns Year: (2012)
Ref_id:b18 Title: Spherical Wave Expansion With Arbitrary Origin for Near-Field Antenna Measurements Year: (2017)
Ref_id:b19 Title: Principles of Optics Year: (2013)
Ref_id:b20 Title: Wireless Communications: Principles and Practice Year: (1996)
Ref_id:b21 Title: FLog: Automated Modeling of Link Quality for LoRa Networks in Orchards Year: ()
Ref_id:b22 Title:  Year: (2024)
Ref_id:b23 Title: WiNeRT: Towards Neural Ray Tracing for Wireless Channel Modelling and Differentiable Simulations Year: ()
Ref_id:b24 Title:  Year: (2024)
Ref_id:b25 Title: Mobile Communication Systems Year: (2012)
Ref_id:b26 Title: Empirical Formula for Propagation Loss in Land Mobile Radio Services Year: (1980)
Ref_id:b27 Title: Comparative Study of Different BLE Fingerprint Reconstruction Techniques Year: ()
Ref_id:b28 Title: FIRE: enabling reciprocity for FDD MIMO systems Year: (2021)
Ref_id:b29 Title: On the Spatial Predictability of Communication Channels Year: (2012)
Ref_id:b30 Title: GWRF: A Generalizable Wireless Radiance Field for Wireless Signal Propagation Modeling Year: (2025)
Ref_id:b31 Title: RF-3DGS Year: (2024)
Ref_id:b32 Title: Wireless Radiation Field Reconstruction with 3D Gaussian Splatting Year: (2024)
Ref_id:b33 Title: Photo Tourism: Exploring Photo Collections in 3D Year: (2006)
Ref_id:b34 Title: Modeling the World from Internet Photo Collections Year: (2008)
Ref_id:b35 Title:  Year: (1873)
Ref_id:b36 Title: OrchLoc: In-Orchard Localization via a Single LoRa Gateway and Generative Diffusion Model-based Fingerprinting Year: ()
Ref_id:b37 Title: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks Year: (2015)
Ref_id:b38 Title: 5G NR release 16 and millimeter wave integrated access and backhaul Year: (2020)
Ref_id:b39 Title: Eliminating Channel Feedback in Next-Generation Cellular Networks Year: (2016)
Ref_id:b40 Title: mD-Track: Leveraging multi-dimensionality for passive indoor Wi-Fi tracking Year: (2019)
Ref_id:b41 Title: Understanding real many-antenna MU-MIMO channels Year: (2016)
Ref_id:b42 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b43 Title: An introduction to autoencoders Year: (2022)
Ref_id:b44 Title: MRI: Model-Based Radio Interpolation for Indoor War-Walking Year: (2014)
Ref_id:b45 Title: Backpropagation and stochastic gradient descent method Year: (1993)
Ref_id:b46 Title: Strict Positive Definiteness of a Product of Covariance Functions Year: (2011)
Ref_id:b47 Title:  Year: (2016)
Ref_id:b48 Title:  Year: (2024)
Ref_id:b49 Title: General-Purpose Deep Tracking Platform across Protocols for the Internet of Things Year: (2020)
Ref_id:b50 Title: Three-Dimensional Indoor Positioning with 802.11az Fingerprinting and Deep Learning Year: (2024)
