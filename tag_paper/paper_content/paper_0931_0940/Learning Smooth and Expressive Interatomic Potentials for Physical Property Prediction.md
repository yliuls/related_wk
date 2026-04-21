Title: Learning Smooth and Expressive Interatomic Potentials for Physical Property Prediction
Abstract: Machine learning interatomic potentials (MLIPs) have become increasingly effective at approximating quantum mechanical calculations at a fraction of the computational cost. However, lower errors on held out test sets do not always translate to improved results on downstream physical property prediction tasks. In this paper, we propose testing MLIPs on their practical ability to conserve energy during molecular dynamic simulations. If passed, improved correlations are found between test errors and their performance on physical property prediction tasks. We identify choices which may lead to models failing this test, and use these observations to improve upon highly-expressive models. The resulting model, eSEN, provides state-of-the-art results on a range of physical property prediction tasks, including materials stability prediction, thermal conductivity prediction, and phonon calculations.

Section: Introduction
Density Functional Theory (DFT), which models the electrons in materials and molecules, serves as the foundation for many modern drug and materials discovery workflows. Unfortunately, DFT calculations are notoriously computationally intensive, scaling cubically with the number of electrons in the system: O(n 3 ). Machine learning interatomic potentials (MLIPs) are promising in approximating and expediting DFT calculations. With increasing data set sizes and model innovations, MLIPs have shown substantial improvements in accuracy and generalization capabilities (Batatia et al., 2023;Merchant et al., 2023;Yang et al., 2024;Barroso-Luque et al., 2024). 1 Fundamental AI Research (FAIR) at Meta. Correspondence to: Xiang Fu <xiangfu@meta.com>, C. Lawrence Zitnick <zitnick@meta.com>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
Predicting physical properties in chemistry and materials science often requires complex workflows involving numerous evaluations of DFT or MLIPs. For example, in molecular dynamics (MD) simulations, forces are predicted over thousands to millions of time steps. However, the MLIP literature has mostly focused on assessing models based on energy and force predictions over static DFT test sets rather than directly assessing their performance in complex simulations. This approach has limitations, as improved accuracy on test sets does not always lead to better predictions of physical properties (Póta et al., 2024;Loew et al., 2024).
In this paper, we address two questions: Why does higher test accuracy sometimes fail to enhance a model's ability to predict physical properties, and how can we improve MLIPs to excel in this area? We first outline four critical property prediction tasks and identify the properties required for an MLIP to succeed in these tasks. These properties entail learning a conservative model with continuous and bounded energy derivatives, indicating a smoothly-varying and physically meaningful energy landscape. To test whether these properties hold, we propose testing the ability of MLIPs to practically conserve energy in MD simulations. We demonstrate models that pass this test have a higher correlation between test errors and property prediction accuracy.
Building on these insights, we present a novel MLIP called eSEN and training approach that achieves state-ofthe-art (SOTA) performance on complex property prediction tasks. Specifically, our model is capable of running energy-conserving MD simulations for out-of-distribution systems (Figure 1 (a)). For materials stability prediction, eSEN achieves a leading F1 score of 0.831 and a κ SRME of 0.340 on the compliant Matbench-Discovery benchmark (Riebesell et al., 2023;Póta et al., 2024). Previous models are only able to excel in one of these metrics ( Figure 1 (b,c)). We also achieve a SOTA F1 score of 0.925 and κ SRME of 0.170 on the non-compliant category. On the MDR Phonon benchmark (Loew et al., 2024), SOTA results are found (Figure 1 (d)). Finally, eSEN achieves the highest test accuracy on the SPICE-MACE-OFF dataset (Kovács et al., 2023). Our model (eSEN) achieves the best performance on all benchmarks. A higher correlation between test-set energy MAE and physical property prediction performance can be observed among energyconserving models. All models are trained on MPTrj.
this section cite: ['b8', 'b49', 'b90', 'b4', 'b60', 'b48', 'b66', 'b60', 'b48', 'b40']

Section: Preliminaries

this section cite: []

Section: Machine learning interatomic potentials
Under the Born-Oppenheimer approximation (Oppenheimer, 1927) utilized by DFT (Parr et al., 1979), the Potential Energy Surface (PES) can be written as a function of positions, r, and atomic numbers, a: E(r, a). Per-atom forces can be calculated by taking the negative gradient of the PES with respect to the atom positions, F = -∇ r E.
For periodic systems such as inorganic materials, the lattice parameters l are also considered (E(r, a, l)), and the stress σ may also be calculated, which can be understood as the gradient of the potential energy surface with respect to the lattice parameters.
The goal of an MLIP (Unke et al., 2021b) is to predict the exact same properties as DFT from a training dataset of DFT calculations (Chanussot et al., 2021;Riebesell et al., 2023;Loew et al., 2024). The most straightforward benchmark for MLIPs is to evaluate the model on a held-out test set of DFT calculations, and compare models based on the mean absolute error (MAE) or root mean squared error (RMSE) of energies, forces, or stresses. To bridge the gap between these performance metrics and practical applicability, we need to ensure they correlate with physical property prediction tasks, such as those described next.
this section cite: ['b54', 'b57', 'b15', 'b66', 'b48']

Section: Physical property prediction tasks
Geometry optimization/relaxation. Many computational chemistry and materials science tasks rely on atomic systems being in stable configurations, which correspond to minima of the PES. Stable states are found by minimizing the potential energy using an optimization procedure that iteratively updates atom positions based on the predicted forces (F = -∇ r E). Given that many physical properties are evaluated at or near equilibrium states, geometry optimization (also referred to as "relaxation") is usually the first step in most computational workflows.
this section cite: []

Section: MD simulations.
Simulating the time evolution of atomic systems enables us to gain understanding of various chemical and biological processes, as well as enabling the calculation of macroscopic properties, such as liquid densities, that can be experimentally verified. For the task of molecular dynamics simulation, we typically use a potential to compute the per-atom forces which are then used to numerically integrate Newton's equations of motion. In this work, we will focus on the microcanonical ensemble (NVE), where the number of particles (N), the volume of the system (V), and the energy of the system (E) are kept constant.
Phonon and thermal conductivity calculations. Precise predictions of phonon band structures and vibrational modes are essential for understanding various material properties, including dynamical stability, thermal stability (Bartel, 2022;Fultz, 2010), thermal conductivity (Razeghi, 2002), and optoelectronic behavior (Ganose et al., 2021). The calculation of phonon band structures requires the MLIP to accurately predict higher-order derivatives and capture the subtle curvature of the true PES around critical points. Recent work (Póta et al., 2024) has demonstrated the usage of MLIPs in predicting thermal conductivity (κ) by solving the Wigner transport equation (Simoncelli et al., 2022). In order to accurately predict κ, MLIPs must reliably capture both harmonic and anharmonic phonon behavior, which necessitates the calculation of second and third derivatives of the learned PES.
this section cite: ['b5', 'b27', 'b28', 'b60', 'b72']

Section: Desideratum for physical property prediction
We begin the section by defining what it means for an MLIP to be energy conserving, which is a fundamental principle for applications such as MD simulations (Tuckerman, 2023). For many physical property prediction tasks that probe the higher-order derivatives of the PES it is also important that the PES's derivatives are well-behaved (they exist and are bounded). To indicate whether a PES meets these criteria, we discuss how an MLIP's ability to conserve energy given fixed simulation settings may be used.
this section cite: ['b80']

Section: Conservative forces
For a force model to be conservative, the work done by moving in a closed path must be zero, i.e., the integration of the forces along any path that starts and ends at the same point is zero:
F • dr = 0 (1
)
This property holds if the forces are calculated as the negative derivative of the PES with respect to the atom positions (Unke et al., 2021b). However, predicting forces as derivatives requires an additional backpropagation step through the network, which increases the computational cost of the MLIP. Alternatively, some networks (Liao et al., 2023;Neumann et al., 2024) directly predict forces using a separate force head to increase efficiency 1 . Although direct-force models can achieve high accuracy, their non-conservative nature leads to significantly larger errors in certain property prediction tasks (Fu et al., 2023;Loew et al., 2024;Póta et al., 2024;Bigi et al., 2024).
this section cite: ['b45', 'b52', 'b26', 'b48', 'b60', 'b12']

Section: Bounded energy derivatives
Conservative forces is a necessary but not sufficient condition for an MLIP to demonstrate energy conservation in MD. In practice, MD simulations use a finite-order numerical integration algorithm and a finite time step ∆t, which introduces truncation errors. The most commonly used integrator for the NVE ensemble is the Verlet algorithm-a second-order integrator. The Verlet integrator is known to approximately conserve the total energy of the system in long-time simulations. As shown by Theorem 5.1 of Hairer et al. 2003, the total energy drift of a simulation satisfies
|E(r T , a) -E(r 0 , a)| ≤ C∆t 2 + C N ∆t N T, (2
)
where T , 0 ≤ T ≤ ∆t -N , is the total simulation time, N is a positive integer representing the highest order for which the N th-order derivative of E is continuously differentiable with a bounded derivative, and r 0 and r T are the starting and ending positions of the atoms in the simulation respectively. The constants C and C N are independent of T and ∆t. The energy drift bound contains two terms: the first term represents a time-independent fluctuation of O(∆t 2 ), and the second term represents the long-term energy conservation. The proof for this theorem is long and technical, for 1 Strictly speaking, direct-force models are not truly "potentials", but rather (non-conservative) "force fields". which we refer interested readers to Hairer et al. 2003 andHairer et al. 2006 for more details.
In Equation (2), the ∆t N in the second term and the bound on the simulation time T ≤ ∆t -N implies that the PES must be continuously differentiable to high order for energy conservation in long-time simulations. The critical constant C N depends on the bounds of the derivatives of E up to the (N + 1)th order. This implies that, given a fixed time step size, E and its higher-order derivatives up to the (N )th order all need to be continuously differentiable with bounded derivatives to maintain long-time conservation. If the derivatives of a PES are more tightly bound, approximate energy conservation will be maintained even at larger step sizes ∆t. Therefore, the magnitude of ∆t for which the energy is stable can be viewed as a proxy for the derivative bounds of the estimated PES. Alternatively, if a certain time step is known to be stable when using DFT, we can determine whether an MLIP has similar bounds on higher-order derivatives by testing whether it is also stable using the same time step.
this section cite: ['b35', 'b35', 'b36']

Section: eSEN
We propose equivariant Smooth Energy Network (eSEN), a new MLIP architecture that improves upon architectures that demonstrate high test accuracies to achieve effective physical property predictions. eSEN is a message-passing neural network that conducts multiple blocks of edgewise and nodewise neural processing. Initially, all nodes are embedded as multi-channel spherical harmonic representations. Each eSEN layer block updates the node embedding by conducting an edgewise convolution, followed by a nodewise feed-forward network with normalization layers and residual connections between all layers.
A model diagram is shown in Figure 2. eSEN utilizes the same SO2 convolution layer from the equivariant spherical channel network (eSCN) architecture (Passaro & Zitnick, 2023) inside the edgewise convolution block. Compared to eSCN, our edgewise convolution blocks first concatenate the source and target node embedding, then apply two SO2 convolution layers with an intermediate non-linearity. We also add an envelope function (details in Section 5) which is not in eSCN. The nodewise feed-forward layer uses two equivariant linear layers and an intermediate SiLU-based gated non-linearity (Weiler et al., 2018;Geiger & Smidt, 2022), which is the same as Equiformer (Liao & Smidt, 2022). Unlike eSCN and EquiformerV2 (Liao et al., 2023), which projects the spherical-harmonics channels onto spatial grids for nodewise processing, the nodewise layers in eSEN do not discretize the node representations. As we demonstrate in Section 5, this design improves the ability of the model to conserve energy. Normalization is performed using the equivariant layer normalization (Ba, 2016) proposed by Equiformer (Liao & Smidt, 2022). In the next section, we conduct an in-depth analysis of the key design choices for energy conservation, which we argue is important for accurate physical property prediction.
this section cite: ['b58', 'b87', 'b44', 'b45', 'b3', 'b44']

Section: Design choices for enhancing physical property prediction
As discussed in Section 3, having conservative forces with continuous and bounded energy derivatives are properties an MLIP should obey for MD simulations. It can also be seen as a prerequisite for the MLIP to accurately capture higher-order behavior of the PES and thus high accuracy in physical property prediction tasks such as phonon calculations. Motivated by this observation, we identify design choices that impact a model's ability to conserve energy and whether its PES varies smoothly. These design choices can be categorized into three aspects: (1) conservative vs. direct-force prediction;
(2) discretization of the representation; and (3) obtaining a continuous and smoothly varying PES. For many of these design aspects, their impact on the desired properties is not well understood.
To quantify whether an MLIP's PES is continuous and smoothly varying, we measure the ability of the resulting MLIP to conserve energy during MD simulations with a predetermined fixed time step. We trained eSEN models under the same hyperparameters while ablating one design choice at a time. We construct out-of-distribution (OOD) MD simulation tasks for both inorganic materials and organic molecules using models trained on the MPTrj (Jain et al., 2013;Deng et al., 2023) and the SPICE-MACE-OFF (Eastman et al., 2023;Kovács et al., 2023) datasets. For inorganic materials, we compute an average conservation error over 81 NVE MD simulations of 100 ps based on the TM23 dataset's simulation settings (Owen et al., 2024). For organic molecules, we compute an average conservation error over 7 NVE MD simulations of 100 ps based on the MD22 dataset's simulation settings (Chmiela et al., 2023).
All eSEN models are 2-layer with 3.2M trainable parameters. We include details regarding the task protocol in Appendix A.
this section cite: ['b38', 'b21', 'b23', 'b40', 'b55', 'b19']

Section: Direct-force prediction
Models that directly predict forces F from the atomic configuration may produce forces that are inconsistent with the energy prediction, i.e., F ̸ = -∇ r Ê, and more importantly are unlikely to be conservative. From the perspective of minimizing the test error, the direct-force approach has strong motivations: it avoids the backward pass for force prediction, which significantly improves model efficiency and enables low-precision training which further accelerates training. Empirically, current SOTA accuracy on the OC20, OC22, and Matbench-Discovery (Chanussot et al., 2021;Tran et al., 2023;Riebesell et al., 2023) benchmarks are achieved by direct-force models. Despite this, the directforce formulation results in significant energy drift in MD simulations, as shown in Figure 4 (a1, a2). For this reason, we compute forces as the negative gradient of the PES with respect to the atom positions in eSEN. Direct-force pre-training. Although direct-force models are not suitable for certain physical property prediction tasks, they may still offer advantages (Bigi et al., 2024;Amin et al., 2025). We demonstrate their efficiency can offer significant benefit as a pre-training strategy for a conservative model.
Figure 3 shows the validation loss of 2-layer eSEN models trained on the MPTrj dataset: direct-force, conservative, and conservative fine-tuning from a pre-trained direct-force backbone. We start from a direct-force model trained for 60 epochs, remove its direct-force prediction head, and fine-
(a2) ((a1)
) (b1) (c1) (b2
)c2
Figure 4. Conservation error on the TM23 task (top row) and MD22 task (bottom row) for ablating design choices of eSEN. Models that conserve energy are bolded in the legends.
tune using conservative force prediction. The conservative fine-tuned model achieves a lower validation loss after being trained for 40 epochs compared to the from-scratch conservative model being trained for 100 epochs. The fine-tuning strategy also reduces the wallclock time for model training by 40%. The strategy of combining direct-force pre-training and conservative fine-tuning is also shown to be effective under different data/model settings (Bigi et al., 2024).
this section cite: ['b15', 'b79', 'b66', 'b12', 'b1', 'b12']

Section: Representation discretization
As proposed by Cohen & Welling 2016 and later used in eSCN and EquiformerV2 (Zitnick et al., 2022;Passaro & Zitnick, 2023;Liao et al., 2023), non-linearities may be performed by projecting the spherical harmonics to a discrete grid. A 1 × 1 convolution or pointwise non-linearity may then be applied to this grid, which then get projected back to the spherical-harmonics space. The non-linear step may introduce higher-frequency signals than cannot be properly represented by the spherical harmonics, i.e., they are beyond the Nyquist frequency. This can lead to sampling errors that break strict equivariance and energy conservation. This problem can be mitigated by sampling the grid at higher resolutions as shown in Figure 4 (b1, b2). In eSEN, we instead use the SiLU-based equivariant Gated non-linearity (Weiler et al., 2018;Geiger & Smidt, 2022) that performs the nonlinearity directly in the spherical harmonic representation. This does not require a projection to a discrete grid, so the model is perfectly equivariant and conservative up to numerical accuracy.
this section cite: ['b20', 'b93', 'b58', 'b45', 'b87']

Section: Smoothly varying PES
Subtle choices in the design of MLIPs can have a significant impact on whether a PES varies smoothly and can even lead to the presence of discontinuities. These include how neighboring atoms are chosen, whether envelope functions are used near atom distance cutoffs, and which basis functions are used to embed pairwise atom distances. We discuss each of these in turn.
A maximum number of neighbors limit in graph construction has been found to improve training efficiency without compromising test error (Liao et al., 2023;Qu & Krishnapriyan, 2024). However, it results in a discontinuity in the learned PES as the nearest-K neighbors may change drastically under a small perturbation of the atom positions.
As shown in Figure 4 (c1, c2), having a maximum neighbor limit breaks energy conservation. In eSEN, instead of limiting the number of neighbors, we use the common approach of applying a distance cutoff (6 Å) under which all neighbors are kept.
Envelope functions were first introduced in the DimeNet architecture (Gasteiger et al., 2020a) to improve model smoothness. The radial basis function used in MLIPs is not twice continuously differentiable due to the use of a finite cutoff during graph construction. By applying a polynomial envelope function on the edge messages, the values in an edge message and its first/higher-order derivatives with respect to atom positions decays to 0 when the edge distance approaches the cutoff distance. Figure 4 (c1, c2) shows a model fails to conserve energy without the envelope function.
Radial basis functions are commonly used to embed interatomic distances (Bartók et al., 2013). A larger number of basis functions (512 in Passaro & Zitnick 2023, as opposed to 10 in eSEN's default setting) allows higher-frequency signals to pass through the network. This can lead to the PES being more sensitive to small shifts in the atom positions.
In our experiments, using a large number of basis functions breaks conservation for the TM23 tasks, but is able to conserve energy for the MD22 tasks. Using a Bessel radial basis function (as opposed to a Gaussian radial basis in the default setting) does not impact conservation properties in both tasks.
this section cite: ['b45', 'b61', 'b6', 'b58']

Section: Ablation studies
Many of the architecture choices described above have negligible impact on the test set errors as shown in Table 1. However, as shown in Figure 4, they can have a dramatic impact on whether a model is conservative in practice. If a model is found to be conservative, stronger correlations are found between test errors and property prediction tasks (Figure 1 and Figure 6).
this section cite: []

Section: Experiments
In the previous section, we demonstrated the design of eSEN results in its ability to be energy-conserving in MD simulations. In this section, we evaluate eSEN in physical property prediction tasks: (1) materials stability prediction based on geometry optimization; (2) thermal conductivity prediction; and (3) phonon calculation. We also demonstrate the correlation between test energy MAE and physical property prediction tasks for eSEN.
this section cite: []

Section: Matbench Discovery
The Matbench-Discovery benchmark evaluates a model's ability to predict ground-state (0 K) thermodynamic stability through geometry optimization and energy prediction. It is a widely used benchmark for evaluating ML models in materials discovery. The compliant benchmark only includes models trained on the MPTrj (Jain et al., 2013;Deng et al., 2023) dataset or its subset, which facilitate a fair comparison of model architectures. The F1 score is the primary metric used to rank models. We train an eSEN with 30M parameters on MPTrj for 60 epochs of direct-force pre-training and 40 epochs of conservative fine-tuning. DeNS (Liao et al., 2024) is used during direct-force pre-training. As shown in Table 2, eSEN-30M-MP achieves an F1 score of 0.831-the highest among all compliant models. eSEN-30M-MP also achieves the lowest root mean square deviation (RMSD) when comparing the relaxed structures to the ground truth DFT reference.
The thermal conductivity prediction task requires accurate modeling of harmonic and anharmonic phonons in materials, which tests the accuracy of second and third order derivatives of the learned PES. The primary metric is the symmetric relative mean error in predicting thermal conductivity (κ SRME ). We follow the protocol set forth in the Matbench-Discovery benchmark (Riebesell et al., 2023;Póta et al., 2024) to predict thermal conductivity κ. After running a structural relaxation, κ is computed using second and third order force constants obtained from phonon calculations using the supercell method.
As shown in
Table 2, our model achieves a κ SRME of 0.340 under the default evaluation protocol proposed by Póta et al. 2024. Notably, our model excels in both the F1 score and κ SRME , while all previous models only achieve SOTA performance on one or the other of these metrics. The non-compliant Matbench-Discovery benchmark includes models trained on datasets other than MPTrj. eSEN-30M-OAM is an eSEN model with 30 million parameters pre-trained on the OMat24 (Barroso-Luque et al., 2024) dataset then fine-tuned on the subsampled Alexandria (sAlex) dataset (Barroso-Luque et al., 2024; Schmidt et al., 2024) and MPTrj dataset. As shown in Table 3, eSEN-30M-OAM achieves the best performance among all noncompliant models with an F1 score of 0.925, a κ SRME of 0.170, and an RMSD of 0.0608, significantly advancing state-of-the-art.
this section cite: ['b38', 'b21', 'b46', 'b66', 'b60']

Section: MDR phonon benchmark
The MDR Phonon benchmark (Loew et al., 2024) assesses the performance of MLIPs in predicting key phonon properties, including maximum phonon frequency (ω max ), entropy (S), free energy (F ) and heat capacity at constant volume  Our results are consistent with those reported by Loew et al. 2024, showing that conservative MLIPs significantly outperform direct-force models in terms of prediction accuracy when tested using phonon calculations with a displacement of 0.01 Å. The high error of direct-force models can be largely attributed to high-frequency prediction errors at small displacements (Loew et al., 2024). Increasing the displacement used in the finite-difference phonon calculations to 0.2 Å can considerably improve prediction accuracy of direct-force models (with caveats). We include a more detailed analysis of the relationship between atom displacement and phonon prediction in Appendix B.
In physical phonon calculations, we expect the results to converge as the displacement goes to zero. By examining the resulting phonon band structure, we can gain insight into this behavior. Figure 5 presents the predicted phonon band structure and density of states for three representative materials using eSEN. The predicted phonon bands exhibit convergence as the displacement decreases. In contrast, Figures  acoustic branches and spurious imaginary frequencies.
While we find that the OMat-trained models (without sAlex/MPTrj finetuning) may provide a lower error on phonon prediction (7/7/2/2 for ω max /S/F /C V MAE), we refrain from direct comparison due to mismatch in level of theory. We attribute this result to the softening issue of the sAlex and MPTrj dataset (Deng et al., 2025;Barroso-Luque et al., 2024), which OMat24 addresses. We refer interested
this section cite: ['b48', 'b48', 'b48', 'b22', 'b4']

Section: SPICE-MACE-OFF
We train and evaluate eSEN models on the SPICE-MACE-OFF dataset (Kovács et al., 2023), which is built upon the SPICE dataset (Eastman et al., 2023). As shown in Table 5, eSEN with 6.5M parameters outperforms MACE-OFF-L (4.7M parameters) and EscAIP (45M parameters, direct-force) on all test-set splits for both energy and force MAE. We also include results for eSEN with 3.2M parameters, which has inference efficiency similar to MACE-4.7M, while achieving lower test energy/force MAE. More details on the inference efficiency benchmark are included in Appendix C.
this section cite: ['b40', 'b23']

Section: Test-set error for model development
In Figure 1, we showed the correlation between test error and physical property prediction tasks for different architectures. Figure 6 demonstrates this correlation for different variants of eSEN (with a 1k-materials subset of the MDR Phonon benchmark for efficiency). In particular, among models that pass the MD energy conservation test, a strong
40 50 60 70 Energy MAE (meV/atom) 0.4 0.6 0.8 1.0 1.2 SRME 40 50 60 70 Energy MAE (meV/atom) 20 40 60 80 100 120 140 160 Entropy MAE (J/K/mol) eSEN-30M eSEN-6.5M eSEN-3.2M eSEN, Bessel eSEN, direct-force eSEN, discrete eSEN, neighbor limit eSEN, no envelope eSEN, Nbasis=512 Figure 6. Test error correlation across several property prediction tasks for eSEN variants. Conservative models are shown as boxes and those found to not conserve as crosses. Note metrics for conservative models have a stronger correlation with test set errors.
correlation between test error and κ SRME /vibrational entropy MAE can be observed. We include experimental details about Figure 1 and Figure 6 in Appendix A and additional results for other phonon properties in Appendix B.
7. Related works MLIP architectures have made significant progress since their initial proposal (Behler & Parrinello, 2007). These architectures are usually symmetry-preserving (Smith et al., 2017;Schütt et al., 2017;Gilmer et al., 2017;Chmiela et al., 2017;Artrith et al., 2017;Unke & Meuwly, 2018;Zhang et al., 2018;Zubatyuk et al., 2019;Smith et al., 2020;Kovács et al., 2021), with increasingly expressive atom environment embeddings and message-passing operations (Gasteiger et al., 2020b;2021;Schütt et al., 2021;Liu et al., 2021;Unke et al., 2021a;Chen & Ong, 2022;Deng et al., 2023;Cheng, 2024;Yin et al., 2025). Notably, equiv-ariant architectures based on spherical harmonics representations (Thomas et al., 2018;Thölke & De Fabritiis, 2021;Batzner et al., 2022;Musaelian et al., 2022;Batatia et al., 2022;Passaro & Zitnick, 2023;Liao et al., 2023;Bochkarev et al., 2024;Park et al., 2024;Batatia et al., 2025) have shown strong performance on large-scale datasets. Meanwhile, the high computational cost of these architectures has sparked significant interest in scalable architectures that may not respect physical principles such as energy conservation (Langer et al., 2024;Brehmer et al., 2024;Hu et al., 2021;Yang et al., 2024;Qu & Krishnapriyan, 2024;Neumann et al., 2024;Rhodes et al., 2025). These models have demonstrated strong performance in accuracy, scalability, and relaxation tasks (Chanussot et al., 2021;Riebesell et al., 2023). While their non-physical nature may make them unsuitable for direct usage in some physical property prediction tasks, they may still provide benefit by using the pre-training strategy proposed in this paper and Bigi et al. 2024, distilling them to conservative models (Amin et al., 2025), or combining them with a conservative model using multiple-time-step integration (Bigi et al., 2024).
MLIPs and physical observables. While MLIPs continue to improve, it is necessary to evaluate them in realistic tasks that are relevant to scientific discovery. Physical property prediction benchmarks that involve geometry optimization (Riebesell et al., 2023;Lan et al., 2023;Wander et al., 2024), MD simulations (Fu et al., 2023;Kovács et al., 2023;Moore et al., 2024;Sabanes Zariquiey et al., 2024;Eastman et al., 2024), vibrational analysis and phonon calculations (Póta et al., 2024;Loew et al., 2024;Wines & Choudhary, 2024), and others are increasing in scale with broader applications and wider adoption. Training strategies for learning from physical observables (Wang et al., 2020;Greener, 2024;Röcken et al., 2024;Raja et al., 2024) and the higher-order derivatives of the PES (Fang et al., 2024;Williams et al., 2025) are promising directions to further improve MLIPs for predicting physical properties.
this section cite: ['b11', 'b73', 'b70', 'b33', 'b18', 'b2', 'b81', 'b92', 'b94', 'b74', 'b39', 'b71', 'b47', 'b16', 'b21', 'b17', 'b91', 'b76', 'b75', 'b10', 'b51', 'b7', 'b58', 'b45', 'b13', 'b56', 'b9', 'b42', 'b14', 'b37', 'b90', 'b61', 'b52', 'b65', 'b15', 'b66', 'b12', 'b1', 'b12', 'b66', 'b41', 'b85', 'b26', 'b40', 'b50', 'b68', 'b24', 'b60', 'b48', 'b89', 'b86', 'b34', 'b67', 'b62', 'b25', 'b88']

Section: Discussion
We identify conservative forces and a smoothly-varying PES as two important properties for MLIPs to consistently perform well in physical property prediction tasks. We offer an analysis of design choices to enhance these two properties. The resulting eSEN architecture bridges the gap between the test-set error and downstream applications, achieving SOTA performance in force/energy prediction, geometry optimization, phonon calculations, and thermal conductivity prediction. This implies it may be possible to use test error as a proxy metric for evaluating model performance during development, if a model passes energy conservation tests. This can accelerate innovations in MLIPs, since benchmarking physical properties usually requires significant domain knowledge and is usually time-consuming, whereas evaluating test set error is straightforward and efficient.
this section cite: []

Section: References
Ref_id:b0 Title: Practical methods in ab initio lattice dynamics Year: (1997)
Ref_id:b1 Title: Towards fast, specialized machine learning force fields: Distilling foundation models via energy hessians Year: (2025)
Ref_id:b2 Title: Efficient and accurate machine-learning interpolation of atomic energies in compositions with many species Year: (2017)
Ref_id:b3 Title: Layer normalization Year: (2016)
Ref_id:b4 Title: Open materials 2024 (omat24) inorganic materials dataset and models Year: (2024)
Ref_id:b5 Title: Review of computational approaches to predict the thermodynamic stability of inorganic solids Year: (2022)
Ref_id:b6 Title: On representing chemical environments Year: (2013)
Ref_id:b7 Title: Higher order equivariant message passing neural networks for fast and accurate force fields Year: (2022)
Ref_id:b8 Title: A foundation model for atomistic materials chemistry Year: (2023)
Ref_id:b9 Title: The design space of e (3)-equivariant atom-centred interatomic potentials Year: (2025)
Ref_id:b10 Title: E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials Year: (2022)
Ref_id:b11 Title: Generalized neural-network representation of high-dimensional potential-energy surfaces Year: (2007)
Ref_id:b12 Title: The dark side of the forces: assessing non-conservative force models for atomistic machine learning Year: (2024)
Ref_id:b13 Title: Graph atomic cluster expansion for semilocal interactions beyond equivariant message passing Year: (2024-06)
Ref_id:b14 Title: Does equivariance matter at scale? arXiv preprint Year: (2024)
Ref_id:b15 Title: Open catalyst 2020 (oc20) dataset and community challenges Year: (2021)
Ref_id:b16 Title: A universal graph deep learning interatomic potential for the periodic table Year: (2022)
Ref_id:b17 Title: Cartesian atomic cluster expansion for machine learning interatomic potentials. npj Computational Materials Year: (2024)
Ref_id:b18 Title: Machine learning of accurate energy-conserving molecular force fields Year: (2017)
Ref_id:b19 Title: Accurate global machine learning force fields for molecules with hundreds of atoms Year: (2023)
Ref_id:b20 Title: Group equivariant convolutional networks Year: (2016)
Ref_id:b21 Title: Chgnet as a pretrained universal neural network potential for charge-informed atomistic modelling Year: (2023)
Ref_id:b22 Title: Systematic softening in universal machine learning interatomic potentials Year: (2025)
Ref_id:b23 Title: Spice, a dataset of drug-like molecules and peptides for training machine learning potentials Year: (2023)
Ref_id:b24 Title: Nutmeg and spice: models and data for biomolecular machine learning Year: (2024)
Ref_id:b25 Title: Phonon predictions with e (3)-equivariant graph neural networks Year: (2024)
Ref_id:b26 Title: Forces are not enough: Benchmark and critical evaluation for machine learning force fields with molecular simulations Year: (2023)
Ref_id:b27 Title: Vibrational thermodynamics of materials Year: (2010)
Ref_id:b28 Title: Efficient calculation of carrier scattering rates from first principles Year: (2021)
Ref_id:b29 Title: Directional message passing for molecular graphs Year: (2020)
Ref_id:b30 Title: Directional message passing for molecular graphs Year: (2020)
Ref_id:b31 Title: Gemnet: Universal directional graph neural networks for molecules Year: (2021)
Ref_id:b32 Title: e3nn: Euclidean neural networks Year: (2022)
Ref_id:b33 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b34 Title: Differentiable simulation to develop molecular dynamics force fields for disordered proteins Year: (2024)
Ref_id:b35 Title: Geometric numerical integration illustrated by the störmer-verlet method Year: (2003)
Ref_id:b36 Title: Geometric numerical integration Year: (2006)
Ref_id:b37 Title: A graph neural network for large-scale quantum calculations Year: (2021)
Ref_id:b38 Title: The materials project: a materials genome approach to accelerating materials innovation Year: (2013)
Ref_id:b39 Title: Linear atomic cluster expansion force fields for organic molecules: beyond rmse Year: (2021)
Ref_id:b40 Title: Transferable machine learning force fields for organic molecules Year: (2023)
Ref_id:b41 Title: Adsorbml: a leap in efficiency for adsorption energy calculations using generalizable machine learning potentials Year: (2023)
Ref_id:b42 Title: Probing the effects of broken symmetries in machine learning Year: (2024)
Ref_id:b43 Title: The atomic simulation environment-a python library for working with atoms Year: (2017)
Ref_id:b44 Title: Equiformer: Equivariant graph attention transformer for 3d atomistic graphs Year: (2022)
Ref_id:b45 Title: Equiformerv2: Improved equivariant transformer for scaling to higherdegree representations Year: (2023)
Ref_id:b46 Title: Generalizing denoising to non-equilibrium structures improves equivariant force fields Year: (2024)
Ref_id:b47 Title: Spherical message passing for 3d molecular graphs Year: (2021)
Ref_id:b48 Title: Universal machine learning interatomic potentials are ready for phonons Year: (2024)
Ref_id:b49 Title: Scaling deep learning for materials discovery Year: (2023)
Ref_id:b50 Title: Computing hydration free energies of small molecules with first principles accuracy Year: (2024)
Ref_id:b51 Title: Learning local equivariant representations for large-scale atomistic dynamics Year: (2022)
Ref_id:b52 Title: A fast, scalable neural network potential Year: (2024)
Ref_id:b53 Title: Scalable parallel programming with cuda: Is cuda the parallel programming model that application developers have been waiting for? Year: (2008)
Ref_id:b54 Title: Zur quantentheorie der molekeln [on the quantum theory of molecules] Year: (1927)
Ref_id:b55 Title: Complexity of many-body interactions in transition metals via machine-learned force fields from the tm23 data set Year: (2024)
Ref_id:b56 Title: Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations Year: (2024)
Ref_id:b57 Title: Local density functional theory of atoms and molecules Year: (1979)
Ref_id:b58 Title: Reducing so (3) convolutions to so (2) for efficient equivariant gnns Year: (2023)
Ref_id:b59 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b60 Title: Thermal conductivity predictions with foundation atomistic models Year: (2024)
Ref_id:b61 Title: The importance of being scalable: Improving the speed and accuracy of neural network interatomic potentials across chemical domains Year: (2024)
Ref_id:b62 Title: Stability-aware training of neural network interatomic potentials with differentiable boltzmann estimators Year: (2024)
Ref_id:b63 Title: Thermal Properties of Crystals Year: ()
Ref_id:b64 Title:  Year: (2002)
Ref_id:b65 Title: Orb-v3: atomistic simulation at scale Year: (2025)
Ref_id:b66 Title: Matbench discovery-an evaluation framework for machine learning crystal stability prediction Year: (2023)
Ref_id:b67 Title: Predicting solvation free energies with an implicit solvent machine learning potential Year: (2024)
Ref_id:b68 Title: Enhancing protein-ligand binding affinity predictions using neural network potentials Year: (2024)
Ref_id:b69 Title: Improving machine-learning models in materials science through large datasets Year: (2024)
Ref_id:b70 Title: Schnet: A continuous-filter convolutional neural network for modeling quantum interactions Year: (2017)
Ref_id:b71 Title: Equivariant message passing for the prediction of tensorial properties and molecular spectra Year: (2021)
Ref_id:b72 Title: Wigner formulation of thermal transport in solids Year: (2022)
Ref_id:b73 Title: Ani-1: an extensible neural network potential with dft accuracy at force field computational cost Year: (2017)
Ref_id:b74 Title: The ani-1ccx and ani-1x data sets, coupled-cluster and density functional theory properties for molecules Year: (2020)
Ref_id:b75 Title: Equivariant transformers for neural network based molecular potentials Year: (2021)
Ref_id:b76 Title: Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds Year: (2018)
Ref_id:b77 Title: Distributions of phonon lifetimes in brillouin zones Year: (2015-03)
Ref_id:b78 Title: Implementation strategies in phonopy and phono3py Year: ()
Ref_id:b79 Title: The open catalyst 2022 (oc22) dataset and challenges for oxide electrocatalysts Year: (2023)
Ref_id:b80 Title: Statistical mechanics: theory and molecular simulation Year: (2023)
Ref_id:b81 Title: A reactive, scalable, and transferable model for molecular energies from a neural network approach based on local information Year: (2018)
Ref_id:b82 Title: Learning force fields with electronic degrees of freedom and nonlocal effects Year: (2021)
Ref_id:b83 Title:  Year: (2021)
Ref_id:b84 Title: The effect of lattice vibrations on substitutional alloy thermodynamics Year: (2002)
Ref_id:b85 Title: Accelerating transition state energy calculations with pre-trained graph neural networks Year: (2024)
Ref_id:b86 Title: Differentiable molecular simulations for control and learning Year: (2020)
Ref_id:b87 Title: 3d steerable cnns: Learning rotationally equivariant features in volumetric data Year: (2018)
Ref_id:b88 Title: Hessian qm9: A quantum chemistry database of molecular hessians in implicit solvents Year: (2025)
Ref_id:b89 Title: Chips-ff: Evaluating universal machine learning force fields for material properties Year: (2024)
Ref_id:b90 Title: A deep learning atomistic model across elements, temperatures and pressures Year: (2024)
Ref_id:b91 Title: Alphanet: Scaling up local frame-based atomistic foundation model Year: (2025)
Ref_id:b92 Title: End-to-end symmetry preserving inter-atomic potential energy model for finite and extended systems Year: (2018)
Ref_id:b93 Title: Spherical channels for modeling atomic interactions Year: (2022)
Ref_id:b94 Title: Accurate and transferable multitask prediction of chemical properties with an atoms-in-molecules neural network Year: (2019)
