Title: Learnable Burst-Encodable Time-of-Flight Imaging for High-Fidelity Long-Distance Depth Sensing
Abstract: Long-distance depth imaging holds great promise for applications such as autonomous driving and robotics. Direct time-of-flight (dToF) imaging offers highprecision, long-distance depth sensing, yet demands ultra-short pulse light sources and high-resolution time-to-digital converters. In contrast, indirect time-of-flight (iToF) imaging often suffers from phase wrapping and low signal-to-noise ratio (SNR) as the sensing distance increases. In this paper, we introduce a novel ToF imaging paradigm, termed Burst-Encodable Time-of-Flight (BE-ToF), which facilitates high-fidelity, long-distance depth imaging. Specifically, the BE-ToF system emits light pulses in burst mode and estimates the phase delay of the reflected signal over the entire burst period, thereby effectively avoiding the phase wrapping inherent to conventional iToF systems. Moreover, to address the low SNR caused by light attenuation over increasing distances, we propose an end-to-end learnable framework that jointly optimizes the coding functions and the depth reconstruction network. A specialized double well function and first-order difference term are incorporated into the framework to ensure the hardware implementability of the coding functions. The proposed approach is rigorously validated through comprehensive simulations and real-world prototype experiments, demonstrating its effectiveness and practical applicability. The code is available at: https://github.com/ComputationalPerceptionLab/BE-ToF.

Section: Introduction
Achieving high-precision depth imaging over long distances has remained a fundamental objective in fields such as computer vision, robotics, and autonomous systems. Time-of-flight (ToF) imaging [1,2,3], as a key approach to depth imaging, can be further categorized into direct ToF (dToF) and indirect ToF (iToF) based on differences in working principles. Direct ToF imaging [4] estimates depth by directly measuring the round-trip time of light, enabling high-precision and long-range sensing. Despite its advantages, this approach requires ultra-short pulsed light sources and highresolution time-to-digital converters (TDCs), imposing stringent hardware demands that increase system complexity and cost, thereby limiting its practicality for widespread deployment. Indirect ToF systems [5,6,7,8,9], in contrast, emit amplitude-modulated continuous wave (AMCW) signals and infer depth by analyzing the phase shift between the transmitted and received signals. Due to their relatively lower hardware complexity and cost, iToF systems offer a more practical and hardware-friendly solution. Nevertheless, existing iToF technologies face significant challenges in long-range imaging, primarily due to phase wrapping [10] and low signal-to-noise ratio (SNR) resulting from optical attenuation [11]. To address the phase wrapping, dual-frequency modulation techniques [12,13] have been proposed, albeit at the cost of increased computational complexity and stricter hardware synchronization requirements. Alternative approaches have sought to mitigate phase wrapping under single-frequency modulation by incorporating scene priors [5,14], however, these methods do not fundamentally resolve the intrinsic ambiguity introduced by periodic modulation.
In this paper, we propose a novel ToF imaging paradigm termed Burst-Encodable Time-of-Flight (BE-ToF). Our BE-ToF system operates in a low-frequency burst mode for light pulse modulation and demodulation, such that the phase of the reflected signal sweeps the entire range [0, 2π] within a single, long burst period. This facilitates high-fidelity, long-distance depth imaging using only single frequency modulation. Moreover, considering the significant variation in SNRs caused by the light-falloff, we propose an end-to-end learnable framework that jointly optimizes the coding functions and the depth reconstruction network, thereby ensuring high-precision depth estimation. In particular, we incorporate constraints based on double well function and first-order difference to ensure the hardware implementability of the learned coding functions. We evaluate our method on a synthetic dataset and compare it with conventional iToF approaches, including single-frequency and multi-frequency modulation techniques. Finally, we built a prototype system to prove the effectiveness of our method in real-world experiments.
In general, we make the following contributions:
• We present a novel Burst-Encodable Time-of-Flight imaging system that enables highfidelity long-distance depth sensing using only a single modulation frequency, thereby fundamentally mitigating the issue of phase wrapping inherent in traditional iToF systems.
• We propose an end-to-end learnable framework that jointly optimizes the coding functions and the depth reconstruction network to ensure high-precision depth estimation across varying distances.
• We uniquely incorporate double well function and first-order difference as loss function to ensure the hardware implementability of the learned coding functions.
• We develop a prototype of our BE-ToF system and demonstrate its superior performance on both synthetic datasets and real-world scenarios.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b4', 'b13']

Section: Related Work
ToF imaging. Time-of-flight (ToF) imaging has become a widely adopted and effective modality for depth acquisition. Direct ToF (dToF) enables long-range depth estimation by measuring the round-trip time of short optical pulses [15]. However, attaining high precision with dToF places stringent demands on the illumination and timing hardware, typically requiring nanosecond-or even picosecond-scale pulse widths [4,16,17] as well as TDCs with tens-of-picoseconds resolution and low timing jitter [18,19]. These requirements substantially hinder practical implementation and large-scale deployment. In contrast, indirect ToF (iToF) imaging leverages cost-effective CMOS sensors to achieve high-resolution depth estimation, yet it is inherently susceptible to phase ambiguity due to phase wrapping in long-range scenarios. A common strategy to alleviate this issue is multi-frequency modulation [12,20,21], where low modulation frequencies extend the maximum unambiguous range and high frequencies preserve depth precision. For example, Hanto et al. [22] developed a ToF LiDAR range finder based on dual-modulation frequency switching to extend the imaging range, and Su et al. [23] proposed an end-to-end ToF framework for high-quality depth reconstruction under multi-frequency modulation. Nevertheless, multi-frequency operation often increases hardware complexity and computational cost. Alternatively, single-frequency phase unwrapping has been explored using amplitude correction [5], surface normal constraints [14], and RGB fusion [24], but such approaches typically rely heavily on scene priors and may degrade under challenging conditions. In this paper, we propose Burst-Encodable Time-of-Flight Imaging (BE-ToF) to fundamentally address phase wrapping in iToF, enabling high-fidelity, long-distance depth estimation.
End-to-end learning. End-to-end learning is a method aimed at jointly optimizing optical systems and reconstruction algorithms. Metzler et al. [25,26] obtained high dynamic range (HDR) images from a single-shot by jointly optimizing the optical encoder and the electronic decoder. Nie et al. [27] leveraged an end-to-end network for hyperspectral reconstruction, enabling simultaneous learning of optimized camera spectral response functions and a mapping for spectral reconstruction. For dense 3D localization microscopy, Nehme et al. [28] proposed a deep STORM-based method to achieve end-to-end optimization of point spread function engineering and accurate 3D localization.
To achieve extended depth of field (EDOF), Sitzmann et al. [29] proposed to jointly optimize the optical system and the reconstruction algorithm's parameters to achieve achromatic EDOF imaging. Guo et al. [30] put forward an end-to-end framework capable of jointly optimizing the coding functions and the exposure time to improve the accuracy of fluorescence lifetime imaging. Moreover, in iToF imaging, Chugunov et al. [31] proposed to jointly learn a microlens amplitude mask and an encoder-decoder network to reduce flying pixels in depth captures. Li et al. [11] put forward a Fisher-information-guided framework for the joint optimization of the coding functions and the reconstruction network. Given the remarkable potential of end-to-end learning in elevating imaging performance, we propose an end-to-end learnable framework that jointly optimizes the coding functions and the depth reconstruction network of our BE-ToF, ensuring high-quality depth performance across varying distances.
this section cite: ['b14', 'b3', 'b15', 'b16', 'b17', 'b18', 'b11', 'b19', 'b20', 'b21', 'b22', 'b4', 'b13', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b10']

Section: Learnable Burst-Encodable Time-of-Flight Imaging

this section cite: []

Section: Modulated Signal
Reflected Signal In this section, we first introduce the working principle of our BE-ToF. As shown in Fig. 1(a), conventional iToF is fundamentally constrained by a trade-off between maximum unambiguous range and depth precision, governed by the modulation period T m . To handle this, our BE-ToF performs short-period light pulse modulation/demodulation in a low-frequency burst mode. As illustrated in Fig. 1(b), within each long burst period T burst , a single modulated signal is emitted. When the reflected signal returns with a phase shift ϕ, it can be demodulated by coding functions with controllable time delay τ . Specifically, the total phase shift ϕ can be decomposed into two components: ϕ 1 , which is primarily determined by the controllable time delay τ , and ϕ 2 , which can be recovered using demodulation techniques like 4-step phase shift [5] or deep learning [23,11]. In summary, the depth d can be defined as Eq. 1
d = c (ϕ 1 + ϕ 2 ) T burst 4π = cτ 2 + D(ϕ 2 ) , (1
)
where c is the light speed and D(ϕ 2 ) represents the demodulation process of ϕ 2 .
Thus, in our BE-ToF system, the maximum unambiguous range d mur is primarily determined by burst period T burst , as defined in Eq. 2
d mur = c 2f burst = c • T burst 2 . (2
)
Regarding depth precision, since we divide the phase delay ϕ into two components ϕ 1 and ϕ 2 , where ϕ 1 is entirely determined by the time delay τ . Consequently, the depth error in our BE-ToF system mainly arises during the demodulation of ϕ 2 . Thus, the depth error ϵ d of our BE-ToF system can be represented as Eq. 3
ϵ d = c • ϵ ϕ2 4πf m = c • ϵ ϕ2 • T m 4π , (3
)
where ϵ ϕ2 is the phase error due to several factors like photon noise, readout noise, and multi-path interference. For fixed phase error, the depth error is chiefly governed by the modulation/demodulation period T m .
Based on the above analysis, BE-ToF substantially extends the maximum unambiguous range while maintaining the same depth precision as conventional iToF, thereby enabling long distance and highfidelity depth imaging. Moreover, the depth sensing range of our BE-ToF spans from c•τ 2 to c•(τ +Tm) 2 , which can be flexibly adjusted by tuning the time delay τ . Based on the operating principle of BE-ToF, which can be realized with a pulsed laser and an exposure-encodable camera [32], we propose an end-to-end imaging framework to ensure highquality reconstruction across varying distances and SNRs. As shown in Fig. 2(a), the framework comprises two components: a differentiable forward model that synthesizes BE-ToF measurements in our simulation pipeline, and a reconstruction network that estimates depth from multiple measurements. By jointly optimizing the coding functions and the reconstruction network, the system delivers accurate depth reconstructions under challenging conditions.
this section cite: ['b4', 'b22', 'b10', 'b31']

Section: Differential BE-ToF Imaging Model
In this section, we first establish the differentiable forward model of our BE-ToF for end-to-end optimization, as illustrated in Fig. 2(b). Assuming M (t) is the modulated signal emitted by pulse laser, the reflected signal of scene point s ∈ R 3 can be defined as Eq. 4
R(s, t) = ρ s M (t -2 d(s) c ) + I amb ,(4)
where ρ s is the inherent reflectance of the scene point s, I amb is the ambient light, d(s) denotes the depth value of point s. Furthermore, considering the attenuation of light intensity with distance during propagation, we incorporate the attenuation function into our model as Eq. 5
R(s, t) = F d(s) ρ s M (t -2 d(s) c ) + I amb , (5
)
where F d(s) is the attenuation coefficient of the emitted light M (t) at depth d(s), which is typically inversely proportional to the square of the distance [33]. Finally, the whole BE-ToF imaging process can be formulated as Eq. 6
I i (s) = ∫ τ +Tm τ R(s, t)D i (t) dt, i ∈ 1, ..., K , (6
)
where I i (s) is the measurement value of the camera, D i (t) denotes the coding functions and K denotes the number of measurements. Taking into account the inherent noise of the sensor, the final measurement can be expressed as Eq. 7
X i (s) = I i (s) + n d + n r , n d ∼ P(E(n d )), n r ∼ N (0, σ 2 r ) ,(7)
where n d is the dark noise following the Poisson distribution with expectation E(n d ) and n r is the readout noise following Gaussian distribution with standard deviation σ r .
Considering that X i (s) contains three unknowns: ρ s , I amb , d(s). Therefore, at least K ≥ 3 measurements are required to solve for the depth d(s). With the proposed differentiable forward model, we can simulate the K measurements of the BE-ToF imaging process. To recover high-fidelity depth map from this set of measurements, we propose a Restormer-based Spatial-Channel Fusion Network(RSCF-Net). As shown in Fig. 3, our network adopts Restormer [34] as the backbone, featuring a four-level encoder-decoder structure in which each level comprises multiple Restormer blocks. In contrast to the conventional skip connections used in the original Restormer, we integrate an Efficient Channel Attention (ECA) module [35] to enhance the fusion of features between encoder and decoder branches. Furthermore, recognizing the inherent differences between depth reconstruction and the image restoration tasks for which Restormer was originally designed, we augment our network with two additional components: the Channel Feature Extraction Block (CFEB) and the Multi-scale Feature Fusion Block (MFFB). The CFEB is composed of multiple residual-connected 1×1 convolutional layers, designed to extract inter-channel relationships across multiple per-pixel measurements. On the other hand, the MFFB emphasizes spatial structure by performing preliminary depth estimation at each decoder level and progressively integrating features from multiple scales in a coarse-to-fine manner. The outputs of CFEB and MFFB are subsequently fused to produce the final high-fidelity depth map.
this section cite: ['b32', 'b33', 'b34']

Section: Reconstruction Network

this section cite: []

Section: Loss Function
During the training process, we jointly optimize the coding functions and the reconstruction network. Given that our exposure encodable camera supports only binary coding functions, we enforce hardware implementability by applying constraints based on a double well function and first-order difference. Additionally, Fisher information is incorporated into the loss to improve reconstruction quality, while Mean Squared Error (MSE) is used as the objective to guide the final output. Here we give more details about these losses.
Mean Squared Error Loss. We employ MSE as the fidelity loss to supervise the predicted depth map, as defined in Eq. 8
L M SE = ∑ s ∥d pre (s) -d gt (s)∥ 2 2 . (8
)
Fisher Guidance Loss. The SNR is one of the key factors influencing the quality of ToF imaging. Inspired by [11], we introduce the fisher guidance loss to enhance the quality of our depth reconstruction, which can be summarized as Eq. 9
L f isher = - ∑ s K ∑ i=1 [ 1 2σ 4 i (s) + 1 σ 2 i (s) ] [ ∂E(I i (s)) ∂d ] 2 , (9
)
where E(I i (s)) is the expectation of I i (s) and σ i (s
) = √ E(I i (s)) + E(n d ) + σ 2 r .
Figure 4: Demonstration of the double well function with two identical minima located at x = 0 and x = 1.
Double Well Function Loss. To enable the optimization of binary coding functions within the differentiable physical model. We introduce the double well function from quantum mechanics [36], which is formulated in Eq. 10 f dw (x) = 4(x -0.5) 4 -2(x -0.5) 2 . (10) As shown in Fig. 4, this function has two valleys at x = 0 and x = 1, thereby encouraging the coding functions to converge toward binary states during the optimization process. Therefore, our double well function loss can be defined as Eq. 11
L dw = K ∑ i=1 M ∑ j=1 f dw (D i (t j )) , (11
)
where M is the sampling points on each coding function.
First-order Difference Loss. Although the double well function effectively constrains the coding functions to a binary state, we observe that the learned functions often exhibit extremely narrow peaks, which pose challenges for practical hardware implementation. To mitigate this issue, we introduce a first-order difference loss, as defined in Eq. 12. By minimizing the first-order difference loss, narrow peaks can be effectively suppressed, thus ensuring feasibility for hardware implementation.
L 1st = K ∑ i=1 M -1 ∑ j=1 |D i (t j+1 ) -D i (t j )| . (12
)
Finally, our complete loss can be summarized as Eq. 13
L = L M SE + γ 1 L f isher + γ 2 L dw + γ 3 L 1st , (13
)
where γ 1 , γ 2 and γ 3 are loss balance coefficients.
this section cite: ['b10', 'b35']

Section: Synthetic Assessment

this section cite: []

Section: Implementation Details
Dataset. We use the NYU-V2 dataset [37] to train and test our end-to-end framework. The NYU-V2 dataset is a high-quality RGB-D dataset captured by Kinect. It contains a total of 1449 pairs of precisely aligned RGB and depth images collected from 464 indoor scenes, which enables its extensive application in academic research. For each RGB-D pair, we first apply intrinsic image decomposition [38] to the RGB image to obtain reflectance and ambient light maps. Subsequently, as detailed in Sec. 3.1, given the reflectance ρ s , ambient light I amb , and depth d(s), we can synthesize multiple BE-ToF measurements. We divide the dataset in detail, using 1000 pairs of data as the training set and the remaining 449 pairs as the test set [39,40].
this section cite: ['b36', 'b37', 'b38', 'b39']

Section: Incremental Training Method.
In our BE-ToF system, the SNR varies not only with distance but also significantly under the same distance due to ambient light I amb . Therefore, we introduce an incremental training strategy [41] to ensure robust depth estimation of our network under varying SNR levels. Specifically, for each distance, we define three distinct SNR scenarios arranged from high to low. The network is trained with input data of varying SNRs, progressively transitioning from high to low every 10 epochs. When data of all SNRs are traversed, samples with random SNR are generated and fed to the network for the convergence of the network. including FisherToF [11] under single frequency modulation; Sine/Square + PS algorithm [5] and Sine + DeepToF [23] under dual frequency modulation.
Training Parameters. We choose K in Eq. 6 as 4 and M in Eq. 11 as 1000. The number of restormer blocks in the network is set to [4,6,6,8]. We train the network for 200 epochs using the ADAM optimizer [42] with a batch size of 20. The learning rate is initialized at 0.01 and decays by a factor of 0.7 every 10 epochs. The loss balance coefficients γ 1 and γ 2 are empirically set to 5e-4 and 5e-2 initially, and are updated to 5e-5 and 1 after 40 epochs. γ 3 is always set to 5. Xavier initialization is used for the learnable coding functions. All experiments are conducted on the PyTorch platform [43], using an NVIDIA GeForce RTX 4090 GPU.
this section cite: ['b40', 'b10', 'b4', 'b22', 'b3', 'b5', 'b5', 'b7', 'b41', 'b42']

Section: Comparison with the State-of-the-art Methods
To demonstrate the superiority of our method, we conduct a detailed comparison with traditional iToF approaches, including single frequency modulation and dual frequency modulation. The scenarios encompass multiple distance ranges (0-3m, 30-33m, 60-63m, and 90-93m) combined with varying SNRs, specifically high (H snr = 22 dB), medium(M snr = 19 dB) and low (L snr = 16 dB).
As shown in Fig. 5, we first compare our method with FisherToF [11] under single frequency modulation. While FisherToF achieves precise depth reconstruction at close range, it still suffers from the rapid decline in imaging quality over distance. We then compare our method with a variety of dual frequency modulation approaches, including sinusoid and square coding functions with Phase Shift (PS) algorithm [5] and the learning-based DeepToF we perform a thorough comparison of our RSCF-Net with existing depth reconstruction networks with the same learned coding functions, including DeepToF [23], MaskToF [31] and FisherToF [11].
The quantitative results in Tab. 1 (c) confirm the effectiveness of our network. Fig. 6 presents the visual results of different methods across four distances under low SNR conditions, intuitively demonstrating the advantages of our approach.  We first perform ablations on the proposed double well function loss L dw and first-order difference loss L 1st . Since the learned coding functions are primarily used to control the camera's exposure, they must be strictly binary. As illustrated in Fig. 7, the absence of L dw and L 1st results in coding functions that are entirely impractical to implement in hardware. With only the L dw , the coding functions do converge to binary states; however, the proliferation of narrow peaks still makes them impossible to implement on real hardware.
this section cite: ['b10', 'b4', 'b22', 'b30', 'b10']

Section: Ablation Study
In the next, we present a quantitative analysis to evaluate the impact of the fisher guidance loss and different network blocks. The values in Tab. 2 represent the average MAE measured under different SNRs at the same distance. The experimental results demonstrate the effectiveness of the introduced Fisher loss in guiding the network to learn an optimal coding functions. The ablation studies on different network blocks further validate the significant improvement in reconstruction quality brought by the proposed CFEB and MFFB.
this section cite: []

Section: Physical Experiment Results
Hardware Prototype. As shown in Fig. 8, to validate the effectiveness of our BE-ToF approach in real world scenarios, we built a prototype system comprising a solid-state pulsed laser and an exposure-encodable ICMOS sensor. The laser operates at 532 nm with a 5 ns pulse width, a fixed 1 kHz repetition rate, and up to 1 mJ single-pulse energy. To realize area illumination, we homogenize the beam with a diffuser and expand it using a beam expander. The ICMOS is fitted with a zoom lens (300-800 mm) and supports a minimum exposure gate of 3 ns. Timing synchronization is provided by a fast photodiode that detects each laser pulse and issues a hardware trigger to a signal generator, which then drives the ICMOS with the learned coding functions. This hardware chain achieves picosecond-scale synchronization, ensuring high-quality imaging.
Experimental Results. As shown in Fig. 9, we evaluate our approach across diverse indoor and outdoor scenarios, including a hand model and a kettle indoors and a stone model and a stair out-   doors. All experiments use the same settings as in simulation: we apply the coding functions learned in simulation and reconstruct with RSCF-Net. The modulation period T m is fixed at 20 ns, and the burst period T burst is set to 1ms, corresponding to the laser repetition rate of 1 kHz. We perform a detailed comparison against other methods, including square coding function and several reconstruction networks. Quantitative results are summarized in Tab. 3. Ground truth is acquired via a time-delay scan at the minimum exposure time (3 ns) with a 1 ns step. Both qualitative and quantitative results demonstrate that our system consistently achieves centimeter-level depth accuracy across these scenarios and outperforms existing methods.
this section cite: []

Section: Conclusion, Limitations, and Broader Impact
In conclusion, we propose a novel ToF imaging paradigm, termed BE-ToF. The BE-ToF system enables long-distance high-fidelity depth imaging by modulating and demodulating pulsed signals in burst mode using only single-frequency modulation. Additionally, we introduce a learnable endto-end framework that jointly optimizes binary coding functions and the reconstruction network to effectively handle varying SNRs across different distances, achieving state-of-the-art performance.
Limitations. Despite achieving both long-distance and high-fidelity depth imaging, our BE-ToF system is subject to limitations in its imaging range. As shown in Fig. 1, the operational range is confined between c•τ 2 and c•(τ +Tm) 2 , with higher precision resulting in a narrower imaging range. We are currently exploring several promising directions to mitigate these limitations. First, we can exploit BE-ToFs flexible time-delay control to perform temporal scanning and synthesize a widerange depth map. Second, because temporal scanning can incur significant latency, we favor a coarse-to-fine strategy: first capture a wide-range, low-resolution depth map, then use BE-ToF to selectively acquire high-precision depth in regions of interest (ROIs).
this section cite: []

Section: References
Ref_id:b0 Title: Breaking the temporal and frequency congestion of lidar by parallel chaos Year: (2023)
Ref_id:b1 Title: Automotive lidar technology: A survey Year: (2021)
Ref_id:b2 Title: Performance evaluation of the 1st and 2nd generation kinect for multimedia applications Year: (2015)
Ref_id:b3 Title: Estimation of optical pathlength through tissue from direct time of flight measurement Year: (1988)
Ref_id:b4 Title: Time-of-flight cameras: principles, methods and applications Year: (2012)
Ref_id:b5 Title: Time-of-flight cameraan introduction Year: (2014)
Ref_id:b6 Title: Theoretical and experimental error analysis of continuous-wave time-of-flight range cameras Year: (2009)
Ref_id:b7 Title: itof-flow-based high frame rate depth imaging Year: (2024)
Ref_id:b8 Title: Alignment-free 3d motion compensation for itof imaging via local linear transfer-enhanced joint optimization Year: (2025)
Ref_id:b9 Title: An overview of depth cameras and range scanners based on time-of-flight technologies. Machine vision and applications Year: (2016)
Ref_id:b10 Title: Fisher information guidance for learned time-of-flight imaging Year: (2022)
Ref_id:b11 Title: A twofold modulation frequency laser range finder Year: (2002)
Ref_id:b12 Title: Analysis of errors in tof range imaging with dual-frequency modulation Year: (2011)
Ref_id:b13 Title: Fast single-frequency time-of-flight range imaging Year: (2015)
Ref_id:b14 Title: High-resolution longdistance depth imaging lidar with ultra-low timing jitter superconducting nanowire single-photon detectors Year: (2025)
Ref_id:b15 Title: Improving the depth sensitivity of time-resolved measurements by extracting the distribution of times-of-flight Year: (2013)
Ref_id:b16 Title: A 220 m-range direct time-of-flight 688× 384 cmos image sensor with sub-photon signal extraction (spse) pixels using vertical avalanche photodiodes and 6 khz light pulse counters Year: (2018)
Ref_id:b17 Title: A 64×64-pixels digital silicon photomultiplier direct tof sensor with 100-mphotons/s/pixel background rejection and imaging/altimeter mode with 0.14% precision up to 6 km for spacecraft navigation and landing Year: (2016)
Ref_id:b18 Title: A near-infrared single-photon detector for direct timeof-flight measurement using time-to-amplitude-digital hybrid conversion method Year: (2023)
Ref_id:b19 Title: Multi-frequency phase unwrapping for time-of-flight cameras Year: (2010)
Ref_id:b20 Title: Tackling 3d tof artifacts through learning and the flat dataset Year: (2018)
Ref_id:b21 Title: Time of flight lidar employing dual-modulation frequencies switching for optimizing unambiguous range extension and high resolution Year: (2023)
Ref_id:b22 Title: Deep end-to-end time-of-flight imaging Year: (2018)
Ref_id:b23 Title: Wild tofu: Improving range and quality of indirect time-of-flight depth with rgb fusion in challenging environments Year: (2021)
Ref_id:b24 Title: Deep optics for single-shot high-dynamic-range imaging Year: (2020)
Ref_id:b25 Title: Learning rank-1 diffractive optics for single-shot high dynamic range imaging Year: (2020)
Ref_id:b26 Title: Deeply learned filter response functions for hyperspectral reconstruction Year: (2018)
Ref_id:b27 Title: Deepstorm3d: dense 3d localization microscopy and psf design by deep learning Year: (2020)
Ref_id:b28 Title: End-to-end optimization of optics and image processing for achromatic extended depth of field and super-resolution imaging Year: (2018)
Ref_id:b29 Title: End-to-end fluorescence lifetime imaging with optimized encoding and exposure allocation Year: (2024)
Ref_id:b30 Title: Mask-tof: Learning microlens masks for flying pixel correction in time-of-flight imaging Year: (2021)
Ref_id:b31 Title: Coded exposure deblurring: Optimized codes for psf estimation and invertibility Year: (2009)
Ref_id:b32 Title: Light fall-off stereo Year: (2007)
Ref_id:b33 Title: Restormer: Efficient transformer for high-resolution image restoration Year: (2022)
Ref_id:b34 Title: Eca-net: Efficient channel attention for deep convolutional neural networks Year: (2020)
Ref_id:b35 Title: The double-well potential in quantum mechanics: a simple, numerically exact formulation Year: (2012)
Ref_id:b36 Title: Indoor segmentation and support inference from rgbd images Year: (2012)
Ref_id:b37 Title: Intrinsic image decomposition using structure-texture separation and surface normals Year: (2014)
Ref_id:b38 Title: Deformable kernel networks for joint image filtering Year: (2021)
Ref_id:b39 Title: Towards fast and accurate real-world depth super-resolution: Benchmark dataset and baseline Year: (2021)
Ref_id:b40 Title: Curriculum learning Year: (2009)
Ref_id:b41 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b42 Title:  Year: (2017)
Ref_id:b43 Title: Practical coding function design for time-of-flight imaging Year: (2019)
Ref_id:b44 Title: Coded time of flight cameras: sparse deconvolution to address multipath interference and recover time profiles Year: (2013)
Ref_id:b45 Title: A dataset and evaluation methodology for depth estimation on 4d light fields Year: (2017)
Ref_id:b46 Title: Sun rgb-d: A rgb-d scene understanding benchmark suite Year: (2015)
