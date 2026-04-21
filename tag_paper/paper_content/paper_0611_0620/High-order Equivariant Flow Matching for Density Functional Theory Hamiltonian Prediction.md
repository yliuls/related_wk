Title: High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction
Abstract: Density functional theory (DFT) is a fundamental method for simulating quantum chemical properties, but it remains expensive due to the iterative self-consistent field (SCF) process required to solve the Kohn-Sham equations. Recently, deep learning methods are gaining attention as a way to bypass this step by directly predicting the Hamiltonian. However, they rely on deterministic regression and do not consider the highly structured nature of Hamiltonians. In this work, we propose QHFLOW, a high-order equivariant flow matching framework that generates Hamiltonian matrices conditioned on molecular geometry. Flow matching models continuous-time trajectories between simple priors and complex targets, learning the structured distributions over Hamiltonians instead of direct regression. To further incorporate symmetry, we use a neural architecture that predicts SE(3)-equivariant vector fields, improving accuracy and generalization across diverse geometries. To further enhance physical fidelity, we additionally introduce a fine-tuning scheme to align predicted orbital energies with the target. QHFLOW achieves state-of-the-art performance, reducing Hamiltonian error by 73% on MD17 and 53% on QH9 compared to the previous best model. Moreover, we further show that QHFLOW accelerates the DFT process without trading off the solution quality when initializing SCF iterations with the predicted Hamiltonian, significantly reducing the number of iterations and runtime.

Section: Introduction
Density functional theory (DFT) [1,2] is a cornerstone of modern computational physics [3,4], chemistry [5,6], and materials science [7,8]. It offers a powerful trade-off between accuracy and computational efficiency in predicting the electronic properties of atoms, molecules, and solids [9][10][11]. The practical implementation of DFT relies mainly on solving the Kohn-Sham equations [12], which describe a system of non-interacting electrons that yield the same ground-state density as the interacting system [12]. These equations are typically solved using the self-consistent field (SCF) method [13][14][15], an iterative procedure that updates the electron density until convergence. To this end, at each iteration, the Kohn-Sham Hamiltonian must be reconstructed from the kinetic operator, external potential, Coulomb (Hartree) potential, and exchange-correlation potential [16,17]. However, this reconstruction of the Hamiltonian becomes computationally demanding for larger systems, limiting the scalability of conventional DFT approaches [18].
To address this challenge, recent work has explored machine learning models to predict the Hamiltonian directly from atomic configurations, with the aim of bypassing or accelerating the SCF loop [19][20][21][22][23][24][25][26]. A key challenge in this task is ensuring that the model preserves the physical symmetries of molecular systems, particularly SE(3) symmetry, which governs how Hamiltonians transform under rotations and translations due to their construction in a spherical harmonics basis. SE(3)equivariant networks such as PhisNet [20], QHNet [23], WANet [25], and SHNet [26] have been proposed to address this, using tensor representation and Clebsch-Gordan products to enforce rotational equivariance. These approaches rely on pointwise regression, which may lead to high predictive uncertainty and often struggle to capture the structural correlations present in the Hamiltonian matrix.
Contribution. In this work, we introduce QHFLOW, a high-order SE(3)-equivariant flow matching for generating Kohn-Sham Hamiltonians. Our work uses flow matching to learn a neural ordinary equation (ODE) that transforms a prior distribution to the target distribution of Hamiltonians. We also parameterize the vector fields using high-order SE(3)-equivariant neural networks to ensure that symmetry is preserved not only in the output, but throughout the entire ODE trajectory [27,28].
To be specific, QHFLOW generates Hamiltonians conditioned on molecular geometries using equivariant architectures and is trained to match the distribution of Hamiltonians obtained from DFT. To this end, we design two types of SE(3)-invariant priors to support equivariant flow-based learning: a Gaussian orthogonal ensemble (GOE) and a tensor expansion-based (TE) prior that includes a group-theoretic structure. To further enhance the accuracy of energy-related properties, we introduce a fine-tuning stage that aligns the predicted orbital energies with those from the DFT, inspired by the weight alignment loss in WANet [25]. We provide an overview of QHFLOW in Figure 1.
Our main contributions are as follows:
• Flow matching for Hamiltonian prediction: We are the first to formulate Hamiltonian prediction as a generative problem, learning a trajectory to generate Kohn-Sham Hamiltonians conditioned on molecular geometry using SE(3)-equivariant vector fields.
• Symmetry-aware prior distributions: We introduce two SE(3)-invariant priors, based on the Gaussian orthogonal ensemble (GOE) and the tensor expansion (TE), ensuring equivariant flow matching and symmetrical initialization of Hamiltonian priors.
• Energy alignment fine-tuning: We introduce a fine-tuning strategy that aligns the predicted orbital energies with target orbital energies. This step encourages the model to match the target orbital energies, resulting in physically consistent property predictions.
• Empirical performance: Extensive experiments on MD17 and QH9 demonstrate that QHFLOW outperforms recent methods in terms of predicting the Hamiltonian MAE, orbital energy prediction, HOMO, LUMO and gap energy accuracy. We also show that using QH-FLOW's predictions as initialization for the SCF procedure leads to significant acceleration of the DFT process, reducing the number of iterations and total computation time.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b19', 'b22', 'b24', 'b25', 'b26', 'b27', 'b24']

Section: Related work
Hamiltonian matrix prediction. Learning-based approaches for Hamiltonian matrix prediction have emerged as promising alternatives to the computationally demanding traditional DFT method.
Early work such as SchNOrb [19] extends SchNet [29] to enable accurate prediction of molecular orbitals. PhiSNet [20] introduces an architecture that explicitly respects SE(3)-symmetry using tensor representations and tensor product operations. QHNet [23] improves computational efficiency by reducing the number of high-order tensor products, and WANet [25] further extends this direction to larger molecular systems by incorporating a physically grounded loss function. Recently, SPHNet [26] improves scalability through adaptive path selection in the tensor product. Despite these advances, all preceding methods treat Hamiltonian prediction as a pointwise regression task. In contrast, our approach leverages flow matching to model the structural correlation in the Hamiltonian.
Flow matching for prediction. Flow matching [30,31] has gained significant attention in generative modeling for its ability to leverage an ordinary differential equation (ODE)-based formulation, enabling faster inference than diffusion models [27]. Flow matching has been increasingly applied to machine learning problems in chemistry and drug discovery [32]. In particular, recent work has shown its effectiveness in regression-style applications such as protein structure prediction [33], crystal structure prediction [34,35], and protein binding site identification [36], where modeling complex target distributions is crucial. In this work, we further extend flow matching to Hamiltonian matrix prediction. By framing the task as a distributional matching problem, our model learns structured outputs with greater accuracy and physical fidelity than conventional regression-based methods, illustrating the broader utility of flow matching in scientific prediction tasks.
this section cite: ['b18', 'b28', 'b19', 'b22', 'b24', 'b25', 'b29', 'b30', 'b26', 'b31', 'b32', 'b33', 'b34', 'b35']

Section: Preliminary

this section cite: []

Section: Kohn-Sham density functional theory and Roothaan-Hall equation
Kohn-Sham density functional theory (KS-DFT). The Kohn-Sham (KS) equations [1,12] are simplified analogs of the many-body Schrödinger equation [37] that retain a one-to-one correspondence with the ground-state density of the interacting electron system. This framework has revolutionized quantum mechanical simulations of electronic structures in atoms, molecules, and solids [9][10][11], offering a practical balance between computational efficiency and predictive accuracy.
Consider an N -particle system associated with the wavefunction Ψ : R 3×N → C. In principle, the system is governed by the stationary Schrödinger equation ĤΨ = ϵΨ, which describes the quantum behavior of a system using the Hamiltonian operator Ĥ and the total energy ϵ. A direct solution is generally intractable, as its computational cost scales exponentially with the number of particles N .
this section cite: ['b0', 'b11', 'b36', 'b8', 'b9', 'b10']

Section: Kohn-Sham density functional theory (KS-DFT) provides a computationally feasible alternative.
Instead of solving the complex many-body wavefunction Ψ, KS-DFT uses the ground-state electron density ρ(r) as its fundamental variable. This is achieved by introducing a fictitious system of noninteracting electrons that share the same ground-state density with the real, interacting wavefunction Ψ. The behavior of non-interacting system is described by a set of N single-particle orbitals, {ψ i } N i=1 , which are the solutions to the KS equations:
H[ρ]ψ i = ϵ i ψ i , ρ(r) = N n=1 |ψ i (r i )| 2 ,(1)
where ρ : R 3 → R + is the electron density, H[ρ] is the effective single-particle KS Hamiltonian constructed from ρ, and ϵ i ∈ R is the orbital energy. The H[ρ] contains the external, Hartree, and exchange-correlation terms. We provide a detailed explanation in Appendix A.1.
Roothann-Hall (RH) equation. In practice, the KS-equations are projected into a finite basis set, converting differential equations into the matrix eigenvalue problem known as RH-equation [38,39]:
H[C]C = SCϵ.(2)
Here, C ∈ R B×N denotes the set of coefficients where each row {C bi } B b=1 express an orbital ψ i as a linear combination of B basis functions {ϕ b (r)} B b=1 i.e., ψ i (r) = B i=1 C bi ϕ b (r). Next, H[C] is the Hamiltonian matrix H ∈ R B×B constructed by the coefficient C, and ϵ ∈ R N ×N is the diagonal matrix of orbital energies i.e., diag(ϵ 1 , . . . , ϵ N ). Finally, S ∈ R B×B is the matrix of overlap between basis functions S bb ′ = ϕ * b (r)ϕ b ′ (r)dr where ϕ * b denotes the complex conjugate of ϕ b . The basis functions {ϕ b (r)} is selected to match the system geometry and commonly chosen between Slater-type orbitals (STOs) [40,15], Gaussian-type orbitals (GTOs) [41], or plane waves [42,10].
this section cite: ['b37', 'b38', 'b39', 'b14', 'b40', 'b41', 'b9']

Section: SO(3)-equivariance in RH-DFT SO(3)-equivariance and irreducible representations.
Formally, a map f is said to be SO(3)equivariant if R • f (x) = f (R • x) for all R ∈ SO(3) where • denotes the SO(3) group action. This equivariance can be described in terms of the representation theory [43]. A representation of SO(3) is a matrix-valued function that describes the transforms under rotations, and it is called irreducible if it cannot be decomposed into smaller invariant subspaces. These irreducible representations (irreps) form the fundamental components for constructing SO(3)-equivariant function, and irrep vector is the vector that transforms under a specific irrep. Each SO(3) irrep is indexed by a non-negative integer l, and represented by a Wigner D-matrix D (ℓ) (R) ∈ C (2ℓ+1)×(2ℓ+1) . These matrices characterize how scalar (ℓ = 0), vector (ℓ = 1), and higher-order tensor features (ℓ ≥ 2) transform under rotation. A rank-ℓ irrep vector w (ℓ) ∈ R (2ℓ+1) transforms under rotation R as: High-order SO(3)-equivariance of RH-DFT.
w ℓ R -→ R • w (ℓ) = D (ℓ) (R
The Hamiltonian matrix in RH-DFT exhibits a highly structured form of equivariance. As shown in Figure 2, the matrix can be organized into blocks, where each block corresponds to interactions between two sets of orbital basis functions indexed by pairs of quantum principal numbers and angular momentum numbers, i.e., (n 1 , ℓ 1 ) and (n 2 , ℓ 2 ). For example, a block labeled 1s-2p denotes interactions between (n 1 , ℓ 1 ) = (1, 0) and (n 2 , ℓ 2 ) = (2, 1). These orbital groupings induce a higher-order structure, where the Hamiltonian must remain equivariant not only at the global matrix level but also within each individual block. This makes Hamiltonian modeling more challenging than conventional equivariant tasks such as force or energy prediction, as it requires explicit handling of block-level symmetry constraints.
This symmetry arises naturally in RH-DFT when the basis set is constructed from spherical harmonics, which serve as irrep vectors of SO(3). In particular, standard basis functions such as STO or GTO are defined using spherical harmonics Y m ℓ , where ℓ and m denote the angular momentum and magnetic quantum numbers, respectively [44]. When using such spherical-type basis, the full orbital basis set {ϕ b } B b=1 can be partitioned into K groups, with the k-th group {ϕ n k ℓ k m } ℓ k m=-ℓ k sharing the same principal quantum number n k and angular momentum quantum number ℓ k . Each group spans a (2ℓ k + 1) functions indexed by m, forming a rank-ℓ k irrep vector. We assume the columns of the coefficient matrix C are ordered according to these groupings. Under a rotation R ∈ SO(3), the matrix C transforms as follows:
C R -→ R • C = D(R)C, D(R) = D (ℓ1) (R) ⊕ • • • ⊕ D (ℓ K ) (R), (3
)
where each
D (ℓ k ) ∈ R (2ℓ k +1)×(2ℓ k +1)
is the rank-ℓ k Wigner D-matrix and ⊕ denotes the direct sum over irreps. The Hamiltonian matrix H, being a function of C, also satisfies the SO(3)-equivariance:
H [C] R -→ R • H [C] = H D(R)C = D(R)H [C] D(R) -1 .(4)
We provide further details about the SO(3)-equivariance structure of Hamiltonian submatrix and the direct sum of Wigner D-matrices in Appendix A.4.
4 Quantum Hamiltonian flow matching (QHFLOW) 4.1 Flow matching for Hamiltonian matrix Problem formulation. Our goal is to train a model that predicts the Hamiltonian matrix H from a molecular configuration M = (x, h). Here, for a molecule M with M atoms, x ∈ R M ×3 and h ∈ R M ×D denotes the atom-wise coordinates and D-dimensional atom-wise features, respectively. We train the model using a atomic dataset A with a sample (M, H) corresponding to a molecular configuration M and its Hamiltonian matrix H obtained from conventional RH-DFT solvers.
this section cite: ['b42', 'b43']

Section: Conditional flow matching (CFM).
We first propose the flow matching for Hamiltonian matrix, by parameterizing our model as a continuous normalizing flow (CNF) model. Our CNF aims to learn a time-dependent vector field v t (•|M) that outputs the Hamiltonian matrix H from solving the ODE d dt H t = v t (H t |M) for time t ∈ (0, 1]. The ODE starts from H 0 sampled from the prior distribution p 0 defined over the space R B×B of Hamiltonian matrices and outputs a Dirac distribution centered at the Hamiltonian matrix H. To this end, we learn the vector field v θ t ≈ v t to match the distribution induced from the empirical distribution (M, H 1 ) ∼ A associated with conditional probability paths p t (H t |H). Importantly, the conditional path p t (H t |H) is expressed using a closed-form conditional vector field u t (•|H 1 ): d dt H t = u t (H t |H 1 , M) for H 0 ∼ p 0 . Following previous works [45,35], we parameterize the conditional probability path using linear interpolation between H 0 and H 1 , resulting in the conditional vector field u t (H t |H 1 , M) = H1-Ht 1-t . To train the CNF, we use the conditional flow matching objective defined as follows:
L CFM = E (H,M)∼A,t∼U (0,1),Ht∼pt(•|H) v θ t (H t ) -u t (H t |H 1 , M) 2 2 ,(5)
where U denotes an uniform distribution and ∥ • ∥ 2 2 denotes the Frobeneous norm of a matrix. In practice, we parameterize the vector field by v θ t =
H θ 1 (Ht,M)-Ht 1-t using a neural network H θ 1 (•).
This results in the final loss function:
L CFM = E (H,M)∼A,t∼U (0,1),Ht∼pt(•|H) 1 (1 -t) 2 H θ 1 (H t , M) -H 1,M 2 2 ,(6)
which one could interpret as training H θ 1 to approximate the true Hamiltonian matrix H from its noisy versions H t . We provide a full description of the training and sampling algorithms in Appendix B.1.
Fine-tuning for energy alignment. In practice, many important quantum properties derived from DFT rely directly on orbital energies. To improve the accuracy of such downstream quantities including orbital energies, highest occupied molecular orbital (HOMO) energy, loweset unoccupied molecular orbital (LUMO) energy, and the HOMO-LUMO gap, we introduce a fine-tuning strategy that explicitly aligns the predicted orbital energies with their ground-truth DFT references. This approach is motivated by the weighted alignment loss (WALoss) introduced in WANet [25], which improves prediction quality by aligning the energy of predicted and reference matrix. Rather than incorporating this loss during full training, we adopt it as a post hoc fine-tuning objective.
Specifically, we define a fine-tuning objective that aligns the approximate orbital energies ε derived from our predicted Hamiltonian matrix H θ 1 from CNF with the ground-truth orbital energies ϵ computed from RH-DFT as defined in Equation (2):
L FT = E (H,M)∼A,t∼U (0,1),Ht∼pt(•|H) ε H θ 1 (H t , M) -ϵ 2 2(7)
Here, the ground-truth orbital energies ϵ are computed via the identity ϵ = C ⊤ HC, which follows from the orthonormality condition C ⊤ SC = I, where C is the ground-truth coefficient matrix and S is the overlap matrix. Following WALoss, we estimate ε using the predicted Hamiltonian: ε = C ⊤ H θ 1 C ≈ (C θ ) ⊤ H θ 1 C θ , where C θ is the coefficient matrix obtained by solving the RH-DFT for the predicted Hamiltonian H θ 1 . We provide the implementation details in Appendix B.2.
this section cite: ['b44', 'b34', 'b24']

Section: SE(3)-equivariant flow matching with equivariant priors.
Here, we introduce our approach to design our QHFLOW to satisfy SE(3)-equivariance. We follow the framework of Song et al. [27] to learn an equivariant vector field under an invariant prior. To handle translational symmetry, our networks operate in a mean-free system by centering the molecular coordinates X at the origin, i.e., set X → X -1 N 11 ⊤ X given a vector 1 ∈ R N ×1 of ones. To make QHFLOW satisfy equivariance, we (1) parameterize the vector field v θ t using a SO(3)-equivariant neural network [23] and (2) propose Gaussian orthogonal ensemble (GOE) and tensor expansionbased (TE) priors for the Hamiltonian. The main challenge is to design new priors that are invariant to the rotation of the Hamiltonian matrix as described in Equation (4).
this section cite: ['b26', 'b22', 'b1']

Section: Gaussian orthogonal ensemble (GOE) prior.
We here introduce a simple rotational invariant prior, the GOE prior. A GOE is defined over matrices H ∈ R n×n , with each element drawn from a Gaussian distribution: H ij ∼ N (0, σ 2 ). The log-density is proportional to the Frobenius norm of the matrix, which is invariant under any transformations associated with SO(3) irreps, i.e., the 3D-rotation matrix (ℓ = 1) and even the high-rank Wigner D-matrices (ℓ ≥ 2). Therefore, the prior satisfies the desired invariance, i.e., p(H) = p(D(R)HD(R) -1 ) for all R ∈ SO(3). We provide the full proof for the SO(3)-invariance in Appendix C.1.
this section cite: []

Section: Tensor expansion-based (TE) prior.
We also introduce TE prior, an SO(3)-invariant distribution constructed by first sampling an irrep vector w (ℓ) , then applying tensor expansion to produce a matrix H (ℓ1,ℓ2) ∈ R (2ℓ1+1)×(2ℓ2+1) that transforms covariantly under rotation R ∈ SO(3):
H (ℓ1,ℓ2) R -→ D (ℓ1) (R) H (ℓ1,ℓ2) D (ℓ2) (R) -1 .(8)
To this end, we first design an SO(3)-invariant distribution of irrep vector p(w (ℓ3) ) by decomposing the vector into a radial and rotational parts:
w (ℓ) = rD (ℓ) (R)w (ℓ0)
where r is a scalar norm and w (ℓ0) is a fixed unit-norm irrep vector. i.e., Y ℓ (r 0 ) for fixed direction r0 . By sampling r independently of R, and choosing R uniformly on SO(3), the resulting prior p(w ℓ ) is SO(3)-invariant. In practice, we take r ∼ Normal(µ, σ 2 ) to ensure positivity, and R ∼ Uniform(SO(3)).
Next, we apply tensor expansion, which maps a rank-ℓ irrep vector w (ℓ) ∈ R (2ℓ+1) into a matrix of shape (2ℓ 1 +1)×(2ℓ 2 +1), indexed by the (m 1 , m 2 ) satisfying -ℓ 1 ≤ m 1 ≤ ℓ 1 and -ℓ 2 ≤ m 2 ≤ ℓ 2 , respectively. Each entry of the resulting matrix is defined as follows:
⊗w (ℓ) (ℓ1,ℓ2) (m1,m2) = ℓ m=-ℓ C (ℓ,m) (ℓ1,m1),(ℓ2,m2) w (ℓ) m ,(9)
where C denotes Clebsch-Gordan coefficients, ⊗ denotes the tensor expansion operation. Note that the superscript (ℓ 1 , ℓ 2 ) indicates the output irrep structure and can be omitted when clear from context. The rotation of this expanded matrix is represented by the Wigner D-matrix as follows:
⊗w (ℓ) R -→ D (ℓ1) (R) ⊗w (ℓ) D (ℓ2) (R) -1 .(10)
See Appendix C.2 for proof. This property allows us to construct SO(3)-invariant distributions over Hamiltonians. We formalize this in the following theorem: Theorem 1. Let H = ⊗w (ℓ) (ℓ1,ℓ2) , where an irrep vector w (ℓ) ∼ p(w (ℓ) ) is drawn from a SO(3)invariant distribution. Then the induced distribution over H is invariant under SO(3) transformation:
p(H) = p(D (ℓ1) (R)HD (ℓ2) (R) -1 ), ∀R ∈ SO(3). (11
)
We provide proof in Appendix C.3. Using the expansion, we can model the invariant distributions for each sub-matrix of Hamiltonian H. We further extend this construction to build global invariant distributions over full Hamiltonian matrices H by composing multiple high-order irrep vector via tensor expansion. We provide a detailed construction of this multi-block formulation in Appendix C.4.
this section cite: []

Section: Model implementation
We construct QHFLOW by extending QHNet [23], an SE(3)-equivariant GNN for Hamiltonian prediction. Our architecture introduces time conditioning and two additional inputs: the current Hamiltonian H t and the overlap matrix S. The model takes (M, H t , S, t) and predicts the target Hamiltonian H θ 1 as described in Equation (6). We provided the implementation details in Appendix D.1.
Feature intialization. We first construct the atom-wise features h i , s i based on (1) extracting atomwise submatrices H i , S i from H t , S and (2) mapping the submatrices into SO(3) irrep vectors based on Wigner-Eckart projections [46], i.e., set h i = ProjBlock (H i ) , s i = ProjBlock (S i ) where ProjBlock flattens and applies change of basis transformations to the input matrix. We further encode time t through tensor field network (TFN) [47] layer and sinusoidal encoding e t = f time (t), i.e., set m i = TFN (x i , h i , e t ).
Message-passing layers. To propagate information between neighbors, our model updates the irrep vectors h i and s i using SO(3)-equivariant attention-based layers adapted from Equiformer [48]:
h i ← EquiBlock h i , (h j , d ij ) j∈Ni , e t , s i ← EquiBlock s i , (s j , d ij ) j∈Ni , e t ,
where N i denotes the neighbors of atom i and d ij is distance between atom i and j. We then apply a channel mixing module, Mix, which linearly combines the set of irrep vectors by flattening and reordering them according to their corresponding irrep indices while preserving SE(3)-equivariance. This module update the m i in the residual manner:
m i ← TFN m i + Mix(h i , s i , m i ), (m j , d ij ) j∈Ni , e t .(12)
This update process is repeated for a fixed number of iterations, allowing the model to integrate geometric and physical information across the molecular graph.
Hamiltonian reconstruction. The atom-wise irrep vectors m i are passed into two TFN modules to predict Hamiltonian components. The first TFN s computes atom-wise irreps w ii using messages from neighboring atoms, while the next TFN p predicts pairwise irreps w ij between atom i and j:
w ii = TFN s m i , {(m k , d ik )} k∈Ni , w ij = TFN p m i , m j , d ij .(13)
Each output consists of a set of irrep vectors,
w ii = {w (ℓ)
ii } ℓ∈Li and w ij = {w (ℓ) ij } ℓ∈Li , where L i is the set of angular momentum ℓ corresponding to the atom i. Then Hamiltonian blocks are constructed via tensor expansion ⊗ and learnable weights F :
H (ℓ1,ℓ2) ij = ℓ∈Li F (ℓ1,ℓ2,ℓ) ij ⊗w (ℓ) ij (ℓ1,ℓ2) ,(14)
where ℓ 1 ∈ L i and ℓ 2 ∈ L j , and the case i = j corresponds to diagonal block. These blocks are then assembled into the full Hamiltonian matrix Ĥ, which is structured into K × K sub-matrices corresponding to all pairwise combinations of the K groups of basis functions. All operations in this reconstruction preserve SE(3)-equivariance, ensuring that both Ĥ and the learned vector field v θ t remain equivariant under any SE(3) transformation of the molecular geometry.
this section cite: ['b22', 'b5', 'b45', 'b46', 'b47']

Section: Experiments
In this section, we present experimental results that demonstrate the effectiveness of QHFLOW. We evaluate the performance on a public benchmark and compare it with competitive baselines [19,20,23,26] when metrics are available. We provide training settings and hyperparameters in Appendix E.
this section cite: ['b18', 'b19', 'b22', 'b25']

Section: Datasets.
We evaluate our model on two datasets: MD17 [49,19] and QH9 [50,51,24]. MD17 contains DFT Hamiltonians along molecular dynamics trajectories for four small molecules, water (4,900 structures), ethanol (30,000), malondialdehyde (26,978), and uracil (30,000). The PBE exchange-correlation [52,53] functional with a GTO basis set is used for DFT.
QH9 consists of two subsets: stable and dynamic-300k, each supporting in-distribution (id) and outof-distribution (ood) splits. The stable includes 130,831 molecules with up to 29 atoms. The id split randomly samples molecules, while the ood split groups them by size. The dynamic-300k contains 2,998 molecules, each with 100 perturbed geometries. The geo split assigns geometries randomly across splits for each molecule; the mol split partitions at the molecule level to test on unseen molecular identities. All QH9 Hamiltonians are computed using B3LYP [54] with the def2-SVP basis set. We provide the details about the dataset in Appendix E.1.
Evaluation metrics. Our evaluation metrics include the mean absolute error (MAE) of the Hamiltonian H, the MAE of occupied orbital energies (ϵ occ ), and the similarity score of the orbital coefficient matrix (S c ). To assess physical fidelity on QH9, we also report the energy of LUMO (ϵ LUMO ), HOMO (ϵ HOMO ), and HOMO-LUMO gap (ϵ ∆ ). We provide details of the metrics in Appendix E.2.
this section cite: ['b48', 'b18', 'b49', 'b50', 'b23', 'b25', 'b51', 'b52', 'b53']

Section: Performance on MD17 dataset
We extensively evaluate QHFLOW on the widely used MD17 benchmark dataset to assess its accuracy in predicting Hamiltonian matrices in diverse molecular geometries. As shown in Table 1, we compare QHFlow with baseline models in four representative molecules, and it consistently outperformed all existing methods. With up to a 73% reduction in Hamiltonian prediction error, QHFlow demonstrates strong predictive performance. This improvement highlights its ability to capture fine-grained variations in quantum Hamiltonians resulting from subtle geometric changes, an essential capability for accurately modeling physical and chemical behavior.
this section cite: []

Section: Performance on QH9 dataset
We evaluate QHFLOW on the QH9 dataset, a more challenging benchmark that includes various molecular sizes and compositions. This setting requires the model to generalize effectively across chemical and geometric variations. We trained on four different data splits. As shown in Table 2, QHFLOW consistently outperforms all prior methods in all metrics. In particular, it shows significant improvements in properties sensitive to eigenvalues such as ϵ LUMO , ϵ HOMO , and ϵ ∆ , showing the benefits of the flow-based model in capturing a physically meaningful structure in Hamiltonians. These results demonstrate QHFLOW's strong potential as a surrogate model for DFT.
We also apply our energy fine-tuning stage to a pretrained QHFLOW using the weighted alignment loss (WAloss) [25] for an additional 60,000 steps, which we denote as WA-FT. As shown in Table 2, this improves predictions of orbital energies and their downstream task, confirming that energy alignment fine-tuning serves as an effective inductive bias. We compare training from scratch with the WALoss and flow matching objectives in Appendix F.1.
this section cite: ['b24']

Section: DFT acceleration performance
Here, we show that our QHFLOW can accelerate DFT through initializing the SCF iterations [13][14][15] using the predicted Hamiltonian matrix, replacing the conventional SCF initialization methods.
Preliminaries on SCF method. Starting from an initial Hamiltonian H (0) , each SCF iteration solves the generalized eigenvalue problem H (k) C (k) = SC (k) ϵ (k) to obtain the coefficients C (k) . The coefficients are then used to construct the density ρ (k) and subsequently update the Hamiltonian to H (k+1) through the mapping C (k) Q -→ ρ (k) H -→ H (k+1) , where Q -→ and H -→ denote the density construction and Hamiltonian update operators, respectively. The iteration continues until convergence. DFT acceleration via SCF initialization. We use QHFLOW's prediction Ĥ to replace the conventional Hamiltonian guess by beginning the SCF loop closer to convergence. Performance is summarized with four relative metrics normalized by the conventional DFT baseline metric: SCF Iter Ratio: ratio of SCF iteration count, SCF T Ratio: ratio of SCF wall time, Inf T Ratio: ratio of inference time as a fraction of baseline SCF wall time, Total T Ratio: Ratio of net runtime after adding inference overhead. We provide the definition of metric in Appendix E.2.
We evaluate QHFLOW on 300 test molecules from the QH9 dataset, selected via a fixed index slice using a 3-step ODE-based sampling. As shown in Figure 3, QHFLOW significantly accelerates the convergence of SCF. For instance, on the QH9 id benchmark, QHFLOW achieves convergence in 46% of the runtime required by conventional DFT, corresponding to 54% relative speedup. Compared to previous ML-based initializations methods like QHNet which requires 53% of the conventional runtime, QHFLOW provides an additional reduction of 7% points. These results demonstrate the practical utility of QHFlow for accelerating real-world DFT simulations.
this section cite: ['b12', 'b13', 'b14']

Section: Ablation studies
Here, we ablate our design choices, with more results (e.g., orbital energy metrics) in Appendix F.3.
this section cite: []

Section: GOE, TE-based priors.
We explore the effects of the flow matching prior for Hamiltonian prediction by comparing the two priors introduced in Section 4: GOE and TE prior. The GOE prior treats each symmetric matrix as structure-free noise, whereas the TE prior injects group-theoretic bias that matches the blockwise symmetry of Hamiltonians.
Table 3 compares GOE and TE priors, where the TE prior results are identical to those of QHFlow reported in Table 2.
We found that the TE prior consistently yields lower errors than the GOE prior across all splits. This highlights the importance of designing task-aligned and symmetry-aware priors for accurate Hamiltonian prediction. Predictive variance. To evaluate the robustness of QHFLOW to stochastic initialization, we repeat inference 5 times per molecule using the TE prior, changing only the seed that generates the initial noise sample. Results are shown in Table 4, with all metrics reported as mean±std over five runs. The predicted Hamiltonians exhibit a mean absolute deviation of just 0.03% and a standard deviation of predicted energy below 0.3µE h , much smaller than the prevalent chemical accuracy thresholds. These results confirm that QHFLOW produces stable and consistent predictions despite stochastic initialization.
this section cite: []

Section: Conclusion
In this work, we introduce QHFLOW, a high-order equivariant flow-based generative model designed to predict Hamiltonian matrices with high accuracy and physical consistency. By designing SE(3)invariant priors GOE and TE, QHFLOW improves the accuracy and generalization across diverse geometries. Extensive experiments on the MD17 and QH9 datasets demonstrate that QHFLOW not only achieves state-of-the-art performance across multiple evaluation metrics but also significantly improves efficiency in downstream SCF calculations. Our results highlight the power of flow-based learning for scientific tasks and establish QHFLOW as a scalable and physically grounded framework for accelerating quantum chemistry simulations.
this section cite: []

Section: Limitation
We did not fully verify the generalizability of our approach, e.g., extension to new architectures (WANet [25], SPHNet [26]) or datasets (PubChemQH [25]), due to the lack of resources, public codes, and public datasets. Our flow-based formulation introduces computational overhead for solving ODEs. In the future, one could consider removing the overhead with the one-step generative models [55]. Our experiments are limited to gas-phase molecules with up to 29 atoms and two commonly used XC functionals (PBE and B3LYP). The algorithms are left to be verified for more general settings, such as periodic solids or higher-accuracy functionals (e.g., hybrid or double-hybrid). We also acknowledge that our datasets consist of relatively small molecules. One could further test the scalability of our approach once new datasets on larger systems are available. We also note that our work does not give a theoretical explanation of how Hamiltonian prediction error translates to SCF acceleration. Developing a theoretically grounded algorithm specifically for SCF acceleration would be an interesting future research.
this section cite: ['b24', 'b25', 'b24', 'b54']

Section: Broader Impact
This work focuses on advancing machine learning methods for quantum chemistry applications and does not involve human subjects, personal data, or sensitive information. As such, we believe that there are no direct ethical concerns associated with this research. Nevertheless, we acknowledge that improved computational tools for molecular modeling could have downstream applications in sensitive areas such as pharmaceuticals or materials development. We encourage responsible and ethical use of our methods in accordance with applicable laws and scientific guidelines.
this section cite: []

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Discussions on the methods and empirical results can be found in Sections 4 and 5.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Discussion on the experiments results are found in the Section 5.
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

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: We provide the full set of assumptions and proof of the theretical statements in Appendix B.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: We provide the main objective function of our work with the description of the implementation and hyperparamters.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We used the publicly available data. Code will be on GitHub after publication.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: Details are in Appendix D.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: Improvement percentages (> 50%) are discussed in Section 5. Also, we provide the variance of our model prediciton which is related with the nature of the generative model.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: Discussions are in Appendix D.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: The research conforms the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes]
Justification: The work has positive impact on the drug discovery and material science, and this is discussed in Section 1 and 2.
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
Justification: The work does not present issues of high-risk misuse.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All data sources and baseline models are open-sourced. They have been properly credited and mentioned, as outlined in Section 4 and Appendix D.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: No new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: No crowdsourcing experiments were conducted.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: Our methodology is not involved with the LLM. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: A Additional Preliminary

this section cite: []

Section: A.1 Kohn-Sham density functional theory
Density functional theory (DFT) is a cornerstone of quantum chemistry and materials science, offering a computationally tractable framework for approximating the electronic structure of manybody systems. As directly solving the many-electron Schrödinger equation [37] is prohibitively expensive for systems with many particles, DFT reformulates the problem by expressing ground-state properties as a functional of the electron density ρ(r). Since ρ(r) depends only on three spatial coordinates, this significantly reduces the computational cost regardless of the number of electrons.
In practical applications, DFT is most commonly implemented via the Kohn-Sham formulation [1,12], which introduces a fictitious system of non-interacting electrons designed to reproduce the true ground-state density of the real interacting system. The wavefunction of this non-interacting system, often referred to as the Kohn-Sham wavefunction Ψ KS , is defined as follows. Since electrons are fermions, Ψ KS must obey the Pauli exclusion principle. This requires the wavefunction to be antisymmetric with respect to the exchange of any two electrons, a property that is achieved by constructing Ψ KS as a Slater determinant built from the N single-particle Kohn-Sham orbitals {ψ i } N i=1 :
Ψ KS (r 1 , . . . , r N ) = 1 √ N ! det[ψ i (r j )] N i,j=1 = 1 √ N ! ψ 1 (r 1 ) ψ 2 (r 1 ) • • • ψ N (r 1 ) ψ 1 (r 2 ) ψ 2 (r 2 ) • • • ψ N (r 2 ) . . . . . . . . . . . . ψ 1 (r N ) ψ 2 (r N ) • • • ψ N (r N )(15)
Originally, the electron density is calculated from the wavefunction Ψ:
ρ(r) = N • • • |Ψ(r, r 2 , . . . , r N )|dr 2 • • • dr N .(16)
However, assuming orthonormal orbitals, the density expression simplifies to:
ρ(r) = N i=1 |ψ i (r)| 2 . (17
)
Instead of explicitly constructing the full KS wavefunction, which scales exponentially with system size due to the determinant, we can directly obtain the electron density from the single-particle orbitals ψ i . The single-particle orbitals {ψ i (r)} and their corresponding energies {ϵ i } are found by solving the Kohn-Sham (KS) equations:
H[ρ]ψ i (r) = - 1 2 ∇ 2 + V ext (r) + V H [ρ](r) + V xc [ρ](r) ψ i (r) = ϵ i ψ i (r),(18)
Here, the entire term in brackets is the effective KS Hamiltonian H[ρ], which is a functional of the electron density ρ. It consists of the Laplacian operator (∇ 2 ) for the kinetic energy of the noninteracting electrons, the external potential (V ext ) from the atomic nuclei, the Hartree potential (V H ) representing classical electrostatic repulsion, and the crucial exchange-correlation potential (V xc ), which encapsulates all complex many-body quantum effects.
The ground-state electron density ρ(r) is then constructed from the N occupied KS orbitals obtained from solving these equations:
ρ(r) = N i=1 |ψ i (r)| 2 . (19
)
It is important to note that the KS orbital ψ i is not, by itself, a physical wavefunction. It is a mathematical construct introduced solely to yield the same ground-state electron density as the real, interacting system. Additionally, the Hamiltonian H[ρ] depends on the density ρ, and ρ itself is constructed from the orbitals ψ i that solve the KS equations, therefore forming a self-consistent problem.
this section cite: ['b36', 'b0', 'b11']

Section: A.2 Group theory
We briefly introduce key concepts from group theory, providing the necessary background to understand the motivation behind equivariant designs in Hamiltonian prediction.
this section cite: []

Section: Groups.
A group G is a non-empty set equipped with a binary operation • : G × G → G, satisfying the following axioms:
• Associativity: For all a, b, c ∈ G, (a • b) • c = a • (b • c).
• Identity: There exists an identity element e ∈ G such that for all a ∈ G, e • a = a • e = a.
• Inverse: For each a ∈ G, there exists an inverse a -foot_1 ∈ G such that a • a -1 = a -1 • a = e.
• Closure: For all a, b ∈ G, the result a • b is also in G.
We typically simplify notation by omitting •, writing ab instead of a • b. Groups can be finite or infinite, discrete or continuous, compact or non-compact. Group theory [43] provides a rigorous mathematical foundation for describing symmetries in physical systems [56], an essential perspective for designing physically informed machine learning models.
this section cite: ['b42', 'b55']

Section: Group actions.
To make group theory applicable to structures in mathematics and physics, we define how groups act on sets. A group G is said to act on a set X if there exists a function • X : G×X → X 1 such that for all g, h ∈ G and x ∈ X, the following conditions hold: (1) e • X x = x, where e is the identity element of G, and (2) (gh)
• X x = g • X (h • X x).
These conditions ensure that the group operation is compatible with its action on the set. In practice, such group actions often correspond to geometric transformations, such as translations, rotations, or reflections, on points or functions defined over Euclidean space.
Equivariance. When working with structured data such as molecular geometries or fields, it is often desirable that the operations we apply preserve the symmetries of the underlying space. A function f : X → Y is said to be equivariant with respect to group actions • X on the input space X and • Y on the output space Y, if it satisfies
f (g • X x) = g • Y f (x), ∀g ∈ G, x ∈ X . (20
)
This means that applying a group transformation before or after the function yields consistent results.
In physical systems, such as those governed by rotational symmetry (e.g., SO(3) in molecules), equivariance ensures that rotated inputs yield correspondingly rotated outputs. This property is critical in designing models that generalize well across symmetrically equivalent configurations.
Group representations. Group actions on vector spaces are formalized through the notion of representations. A representation of a group G on a vector space V is a homomorphism ρ : G → GL(V ), where GL(V ) denotes the group of invertible linear transformations on V . This mapping associates each group element with a linear operator that describes how the group acts on the space. The vector space V is referred to as the representation space, and its dimension defines the representation's size.
In finite-dimensional real spaces V = R d , these representations are often expressed as invertible matrices D(g) ∈ R d×d . For instance, the group SO(3) of 3D rotations is represented by orthogonal matrices g ∈ SO(3) and its 3D rotation matrix R ∈ R 3×3 acting on vectors x ∈ R 3 via matrix multiplication D(g)x = Rx.
Two representations D(g) and D ′ (g) are said to be equivalent if they are related by a change of basis:
D ′ (g) = Q -1 D(g)Q,(21)
for some invertible matrix Q. Such equivalence indicates that the two representations describe the same group action under different coordinate systems.
this section cite: []

Section: A.3 Irreducible representations Irreducible representations (Irreps).
A representation ρ : G → GL(V ) of a group G on a vector space V is called irreducible if it does not contain any nontrivial invariant subspace. More formally, a subspace W ⊆ V is said to be invariant under the representation ρ if for all group elements g ∈ G and all vectors w ∈ W , the action satisfies:
ρ(g)w ∈ W.(22)
Then, the representation ρ is irreducible if the only invariant subspaces it admits are the trivial subspace {0} and the entire representation space V .
Irreducible representations play a fundamental role because any finite-dimensional unitary representation of a compact group can be uniquely decomposed into a direct sum of irreducible representations, up to isomorphism [57]. This decomposition is foundational in analyzing the symmetry properties and constructing equivariant neural network architectures.
For the rotation group SO(3), the irreps are completely characterized by non-negative integers ℓ ∈ N 0 . Each irreducible representation D (ℓ) : SO(3) → C (2ℓ+1)×(2ℓ+1) has dimension 2l+1 and is explicitly realized by the action on spherical harmonics Y l m (θ, ϕ). For a function f : S 2 → C can be expanded in the basis of spherical harmonics as:
f (θ, ϕ) = ∞ l=0 l m=-l f lm Y l m (θ, ϕ),(23)
where f lm ∈ C is the complex coefficients of the spherical harmonics. The action of a rotation R ∈ SO(3) on f is defined as:
(R • f )(θ, ϕ) = f (R -1 • (θ, ϕ)) = ∞ l=0 ℓ m=-ℓ ℓ m ′ =-ℓ D (ℓ) mm ′ (R) f ℓm ′ Y ℓ m (θ, ϕ),(24)
where D (ℓ) (R) is the Wigner D-matrix corresponding to the irreducible representation labeled by ℓ, acting linearly on the vector of expansion coefficients f (ℓ) = (f -ℓ , f -ℓ+1 , . . . , f ℓ ) ∈ C (2ℓ+1) . Explicitly, under rotation:
f (ℓ) R -→ D (ℓ) (R)f (ℓ) .(25)
Thus, irreps serve as the fundamental building blocks in constructing functions and features that transform equivariantly under the action of SO(3).
this section cite: ['b56']

Section: Clebsch-Gordan tensor product.
Given two irreps D (ℓ1) , D (ℓ2) of SO(3), their tensor product D (ℓ1) ⊗ D (ℓ2) acts on the tensor product space and is generally reducible. It decomposes into a direct sum of irreps:
D (ℓ1) ⊗ CG D (ℓ2) ∼ = ℓ1+ℓ2 ℓ=|ℓ1-ℓ2| D (ℓ) ,(26)
where the decomposition is governed by Clebsch-Gordan (CG) coefficients. CG coefficients enable a basis transformation from the product space to irreps, playing crucial roles in quantum angular momentum coupling and equivariant neural networks.
Explicitly, for vectors u (ℓ1) ∈ C (2l1+1) and v (ℓ2) ∈ C (2ℓ2+1) , their CG tensor product produces an irreducible feature w (ℓ) ∈ C (2ℓ+1) :
w (ℓ) m = m1,m2 C (ℓ,m) (ℓ1,m1),(ℓ2,m2) u (ℓ1) m1 v (ℓ2) m2 .(27)
This construction ensures that w (ℓ) transforms correctly under R ∈ SO(3):
D (ℓ1) (R)u (ℓ1) ⊗ CG D (ℓ2) (R)v (ℓ2) = D (ℓ) (R)w (ℓ) .(28)
Thus, CG tensor products ensure equivariant transformations, fundamental to maintaining consistency and symmetry in equivariant neural architectures.
this section cite: []

Section: Tensor field networks (TFN).
The TFN [47] is a widely used architecture that achieves SE(3)equivariance by explicitly encoding features as irreps of SO(3) and using spherical harmonics to process directional information. Each feature vector at a node is decomposed into spherical tensor components V (ℓ) ∈ C (2ℓ+1) , where ℓ denotes the order of the irreducible representation. The key component of the TFN layer is equivariant convolution and self-interaction. For the equivariant convolution, the filter function is generated by the spherical harmonic functions Y ℓ m (r ij ) are applied to the direction between atoms i and j, i.e., rij = (r i -r j )/∥r i -r j ∥, and are modulated by a learnable radial function R(∥r ij ∥), implemented as an MLP to produce the filter:
F (ℓin,ℓ f ) (r ij ) = R (ℓin,ℓ f ) (∥r ij ∥)Y ℓ f m (r ij ).(29)
The equivariant convolution aggregates information from neighboring nodes set N using the tensor product between the filters and neighboring features:
Ṽ (ℓout) i = j∈N F (ℓin,ℓ f ) c (r ij ) ⊗ V (ℓin) j (ℓout) ,(30)
where
ℓ out ∈ {|ℓ in -ℓ f |, . . . , (ℓ in + ℓ f )}.
This convolution ensures that the result transforms equivariantly under rotation.
The self-interaction step applies a linear transformation over the channel dimension to each irreps independently:
Ṽ (ℓ) icm = c ′ w cc ′ V (ℓ) ic ′ m ,(31)
preserving equivariance due to the linearity and diagonal structure in the irrep indices. By construction, all operations in TFNs are equivariant to SE(3), allowing them to model tensorial and directional quantities while respecting physical symmetries. This makes them especially well-suited for applications in molecular modeling, quantum chemistry, and materials science, where equivariance is crucial for generalization and physical accuracy.
this section cite: ['b46']

Section: A.4 Symmetry and equivariance of RH-DFT

this section cite: []

Section: Spherical harmonics and atomic orbital basis.
To numerically solve the Kohn-Sham equations, electronic wavefunctions are typically expanded in a basis of atomic orbitals. These orbitals are constructed using spherical harmonics Y ℓ m (θ, ϕ), which form a complete orthonormal basis on the sphere S 2 and transform under the irreducible representations D (ℓ) of the rotation group SO(3). An atomic orbital basis function takes the form [58]:
ϕ nℓm (r) = R nℓ (∥r∥)Y ℓ m (θ, ϕ), (32
)
where R nℓ (∥r∥) is a radial function (e.g., Gaussian [41] or Slater-type [40,15]), and Y ℓ m captures the angular component. Here, n is the principal quantum number, ℓ is the orbital angular momentum quantum number, and m ∈ {-ℓ, . . . , ℓ} is the magnetic quantum number [44].
These basis functions are central to Kohn-Sham DFT calculations, as molecular orbitals are expressed as linear combinations of atomic orbitals:
ψ i (r) = α=(nℓm) C αi ϕ nℓm (r -r α ),(33)
where C ∈ R B×O is the orbital coefficient matrix, B is the number of basis functions, and O is the number of occupied orbitals.
this section cite: ['b57', 'b40', 'b39', 'b14', 'b43']

Section: Block Hamiltonian matrix.
The Hamiltonian matrix in RH-DFT can be decomposed according to the angular momentum quantum numbers ℓ grouping associated with the atomic orbitals. Specifically, let H (ℓ,ℓ ′ ) ∈ C (2ℓ+1)×(2ℓ ′ +1) denote the submatrix representing the coupling between two angular momentum numbers ℓ and ℓ ′ . Under a rotation R ∈ SO(3), each such sub-matrix H (ℓ,ℓ ′ ) transforms equivariantly according to the SO(3) irreps as follows:
H (ℓ,ℓ ′ ) [D(R)C] = D (ℓ) (R) H (ℓ,ℓ ′ ) [C] D (ℓ ′ ) (R) -1 ,(34)
where atomic orbitals are grouped by their ℓ and ℓ ′ , D(R) is block diagonal Wigner-D matrix, and D ℓ and D ℓ ′ are Wigner D-matrices for ℓ and ℓ ′ , respectively. These submatrices H (ℓ,ℓ ′ ) , called orbital-wise Hamiltonian blocks, are the minimal building blocks of the full Hamiltonian that respect SO(3)-equivariance and have (2ℓ + 1) × (2ℓ ′ + 1) elements which corresponds to the combination of the magnetic quantum number m and m ′ .
Full Hamiltonian matrix. In practice, basis functions are grouped not only by their angular momentum, but also by the atom they are centered on. For atom i, we denote its angular momentum support as L i = {ℓ 1 , ℓ 2 , . . .}, where each ℓ k corresponds to its atomic shell (e.g., s, p, d) represented in the basis set.
From this atomic partitioning, the Hamiltonian can be reorganized into block form where each submatrix H (ℓ,ℓ ′ ) ij corresponds to the interaction between angular momentum ℓ ∈ L i of atom i and ℓ ′ ∈ L j of atom j. The full Hamiltonian H is thus a structured matrix composed of K × K blocks, where K = i |L i | is the total number of irreps (angular momentum groups) across all atoms, where |L i | denotes the cardinality of the set.
To construct the full Hamiltonian matrix, we first form the atom-pair matrix H ij for every atom-pair (i, j). This atom-pair matrix gathers all interactions between the angular momentum orbital groupings on the two atoms:
H ij =      H (ℓ0,ℓ ′ 0 ) ij H (ℓ0,ℓ ′ 1 ) ij • • • H (ℓ1,ℓ ′ 0 ) ij H (ℓ1,ℓ ′ 1 ) ij • • • . . . . . . . . .      , ℓ k ∈ L i , ℓ ′ k ∈ L j ,(35)
where
H (ℓ,ℓ ′ ) ij ∈ C (2ℓ+1)×(2ℓ ′ +1)
denotes the interaction between angular momentum ℓ ∈ L i of atom i and ℓ ′ ∈ L j of atom j.
Next, we place these atom-pair blocks into a global block matrix to obtain the full RH Hamiltonian matrix: The full Hamiltonian matrix H ∈ C n×n , where n = ℓ (2ℓ + 1), is assembled by concatenating the atom-pair blocks:
H =    H ii H ji • • • H ij H jj • • • . . . . . . . . .    , n = i∈M ℓ∈Li (2ℓ + 1),(36)
for every atom-pair (i, j) in the molecule M.
For better intuition, we illustrate a schematic structure of the Hamiltonian matrix in Figure 4a. Here, blocks corresponding to s, p, and d orbitals are arranged according to their angular momentum, highlighting the blockwise symmetry pattern induced by the underlying spherical harmonics basis.
Rotational equivariance of full Hamiltonian. The global action of a rotation R on the basis is encoded by the block-diagonal matrix as follows Equation (3):
D(R) = D (ℓ1) (R) ⊕ • • • ⊕ D (ℓ K ) (R) =     D (ℓ1) (R) 0 • • • 0 D (ℓ2) (R) • • • . . . . . . . . .     ,(37)
where each block D ℓ (R) corresponds to the Wigner D-matrix for angular momentum l. The Hamiltonian matrix as a whole then transforms under rotation as:
H[C] R -→ H[R • C] = R • H[C] = D(R) H[C] D(R) -1 . (38
)
This global equivariance condition guarantees that the Hamiltonian transforms consistently with molecular rotations, preserving the physical structure of the system. Leveraging this symmetry is critical for constructing SE(3)-equivariant machine learning models, which not only respect physical laws but also improve model generalization and interpretability in quantum chemistry tasks. For better intuition, we illustrate a schematic structure of the Wigner D-matrix in Figure 4b.
this section cite: []

Section: B Training and sampling algorithms, and fine-tuning objective B.1 Training and sampling algorithms
We investigate two types of prior distributions for the initial Hamiltonian H 0 : Gaussian orthogonal ensemble (GOE) and tensor expansion-based (TE) priors. For the GOE prior, we sample each matrix entry independently as M ij ∼ N (0, σ 2 ), where we set σ 2 = 1.0 for MD17 and σ 2 = 0.1 for QH9.
For TE prior, we sample each irreducible component w l by drawing its radial norm from LogNormal(1, σ 2 = 0.1) and applying a uniform SO(3) rotation to construct the full equivariant basis, which is then mapped to the Hamiltonian space via tensor expansion.
These choices provide flexibility in modeling different types of initial distributions, depending on the target dataset and symmetry constraints. Formal definitions and proofs of equivariance for both priors are provided in Appendix C.1 and Appendix C.4.
The training and sampling procedures based on the above rotationally invariant priors are summarized in Algorithm 1 and Algorithm 2, respectively.
this section cite: []

Section: Algorithm 1 QHFlow training procedure
Require: Dataset of molecular configurations {M i }, target Hamiltonians H 1,M , overlap matrix S M , model f θ 1: for each training step do
H (θ) 1,M ← f θ (H t k , S, M, t k ) 4: Compute conditional vector field: v t k ,θ ← (H (θ) 1,M -H t k )/(1 -t k ) 5: Update Hamiltonian: H t k+1 ← H t k + (t k+1 -t k ) • v t k ,θ 6: end for 7: Output Ĥ1 = H t K
These algorithms ensure that training aligns the predicted vector field with the true conditional flow, while inference produces a final Hamiltonian via integration through the learned ODE trajectory. This design ensures SE(3)-equivariance and physical consistency throughout the training and prediction processes.
this section cite: []

Section: B.2 Fine-tuning objective
After pre-training with the flow matching objective, we further fine-tune QHFlow to enhance its spectral accuracy by introducing energy alignment objectives. To implement the energy alignment objective, we adopt the weighted alignment loss (WALoss) from WANet [25].
Property of the RH-equation. The relationship between the Hamiltonian matrix H, the coefficient matrix C, and the orbital energy matrix ϵ is governed by the Roothaan-Hall equation:
HC = SCϵ, (39
)
where S is overlap matrix. Using orthonormality condition C ⊤ SC = I, this relation simplifies to:
C ⊤ HC = ϵ.(40)
This identity forms the foundation for designing spectral alignment objectives.
Weighted alignment loss (WALoss). WALoss encourages the predicted Hamiltonian Ĥ to match the spectral structure of the converged SCF Hamiltonian H ⋆ . Let C ⋆ and ϵ ⋆ denote the eigenvectors and eigenvalues of H ⋆ , respectively. We define WALoss as:
L WALoss = diag (C ⋆ ) ⊤ ĤC ⋆ -ϵ ⋆ 2 , (41
)
where diag(•) extracts the diagonal elements and ∥ • ∥ 2 denotes the L2 norm.
To emphasize physically important states, we place larger weights on eigenvalues up to the Lowest unoccupied molecular orbital (LUMO) i.e., the k + 1 lowest-energy orbitals. By projecting Ĥ onto the fixed eigenbasis C ⋆ , WALoss promotes alignment with the ground-truth spectrum. However, it is important to note that (C ⋆ ) ⊤ ĤC ⋆ does not exactly diagonalize Ĥ unless Ĥ = H ⋆ . Nevertheless, minimizing this discrepancy helps guide Ĥ toward the correct eigenspectrum, improving orbital energy prediction and SCF behavior.
Overall fine-tuning objective. Our final finetuning objective combines the flow matching loss with WALoss:
L total = L CFM + λ WA L WA ,(42)
where λ WA is the weighting coefficients controlling the contributions of the energy alignment terms, and we chose λ WA = 2.0 for the finetuning.
this section cite: ['b24']

Section: C Proofs and additional details C.1 Proof of SE(3)-equivariance of GOE
Let M ∈ R n×n be a symmetric matrix drawn from the Gaussian orthogonal ensemble (GOE), characterized by:
E[M ij ] = 0, Var(M ij ) = σ 2
, with M ij = M ji , and independent entries for i ≤ j.
Let D = D(R) denote an orthogonal matrix representing a Wigner D-transformation corresponding to a rotation R ∈ SO(3), satisfying orthogonality i.e. DD ⊤ = I. We define the rotated matrix as:
M ′ = DMD ⊤ , with M ′ ij = k,l D ik M kl D jl .(44)
We aim to show that the distribution of M ′ is the same as that of M, i.e., that GOE is invariant under conjugation by orthogonal transformations:
ρ(M) = ρ(DMD ⊤ ).(45)
To establish this, it suffices to verify that the first and second moments of the entries in M ′ match those of GOE.
this section cite: []

Section: Mean.
Since E[M kl ] = 0 for all k, l, we have:
E[M ′ ij ] = k,l D ik D jl E[M kl ] = 0.(46)
Covariance. We now compute the covariance of the entries M ′ ab and M ′ cd :
E[M ′ ab M ′ cd ] = E      i,j D ai D bj M ij     k,l D ck D dl M kl      (47
) = i,j,k,l D ai D bj D ck D dl E[M ij M kl ].(48)
Using the independence and zero mean of GOE entries, we know:
E[M ij M kl ] = σ 2 δ ik δ jl .(49)
Thus,
E[M ′ ab M ′ cd ] = σ 2 i,j D ai D bj D ci D dj (50
) = σ 2   i D ai D ci     j D bj D dj   = σ 2 (DD ⊤ ) ac (DD ⊤ ) bd = σ 2 δ ac δ bd .(51)
This matches the covariance structure of the original GOE matrix. Since both M and M ′ are jointly Gaussian with zero mean and identical covariances, we conclude:
DMD ⊤ ∼ M, i.e., GOE is invariant under orthogonal conjugation.(52)
This proves that the GOE prior is SO(3)-invariant, even for the high-rank Wigner D-matrices ℓ ≥ 2.
this section cite: []

Section: C.2 Proof of the SE(3)-equivariance of tensor expansion operation
Since the w (ℓ) is the irrep vector, the m-th entry w ℓ m transforms under rotation R ∈ SO(3) as:
w (ℓ) m R -→ R • w (ℓ) m =   m ′ D (ℓ) mm ′ (R) w (ℓ) m ′   ,(53)
where 2ℓ2+1) is the Wigner D-matrix.
D (ℓ) (R) ∈ C (2ℓ1+1)×(
By substituting above Equation (53) in Equation ( 9), the entry of rotated tensor expansion is
⊗w (ℓ) (ℓ1,ℓ2) (m1,m2) R -→ ⊗ R • w (ℓ) = m C (ℓ,m) (ℓ1,m1),(ℓ2,m2)   m ′ D (ℓ) mm ′ (R)w (ℓ) m ′   ,(54)
and we can exchange the order of summation:
⊗w (ℓ) (ℓ1,ℓ2) (m1,m2) R -→ m ′ m C (ℓ,m) (ℓ1,m1),(ℓ2,m2) D (ℓ) mm ′ (R) w (ℓ) m ′ .(55)
Now we can use a crucial identity from representation theory: Theorem 2. The Clebsch-Gordan coefficients provide a basis change between the product representation
D (ℓ1) (R) ⊗ D (ℓ2) (R) and the irreducible representation D (ℓ) (R): m C (ℓ,m) (ℓ1,m1),(ℓ2,m2) D (ℓ) mm ′ (R) = m ′ 1 ,m ′ 2 D (ℓ1) m1m ′ 1 (R)D (ℓ2) m2m ′ 2 (R)C (ℓ,m ′ ) (ℓ1,m ′ 1 ),(ℓ2,m ′ 2 ) .(56)
We can substitute the identity, then we have:
⊗w (ℓ) (ℓ1,l2) (m1,m2) R -→ m ′ 1 ,m ′ 2 D (ℓ1) m1m ′ 1 (R)D (ℓ2) m2m ′ 2 (R)   m ′ C (ℓ,m ′ ) (ℓ1,m ′ 1 ),(ℓ2,m ′ 2 ) w (ℓ) m ′   ,(57)
and the last sum implies the
m ′ C (ℓ,m ′ ) (ℓ1,m ′ 1 ),(ℓ2,m ′ 2 ) w(ℓ)
m ′ = ⊗w (ℓ) (ℓ1,ℓ2) (m ′ 1 ,m ′ 2 )(58)
Therefore,
⊗w (ℓ) (ℓ1,ℓ2) (m1,m2) R -→ m ′ 1 ,m ′ 2 D (ℓ1) m1m ′ 1 (R)D (ℓ2) m2m ′ 2 (R) ⊗w (ℓ3) (ℓ1,ℓ2) (m ′ 1 ,m ′ 2 ) ,(59)
in the matrix form,
⊗w (ℓ) R -→ D (ℓ1) (R) ⊗w (ℓ) D (ℓ2) (R) -1 .(60)
this section cite: ['b52']

Section: C.3 Proof of the invariance of the tensor expansion based (TE) prior
To construct an SO(3)-invariant distribution over matrices via tensor expansion, we begin with a rotationally invariant distribution over irreducible features w (ℓ) ∈ C (2ℓ+1) . We define this distribution by factorizing it into two independent components:
• A magnitude r = ∥w (ℓ) ∥ drawn from a radial distribution, e.g., r ∼ Normal(1, σ 2 ).
• A direction sampled uniformly on the sphere S 2 , induced by a Haar-uniform rotation R ∈ SO(3), applied to a canonical unit-norm feature vector w
(ℓ) 0 , such that w (ℓ) = rD (ℓ) (R) w (ℓ) 0 .
This construction ensures that w (ℓ) is distributed isotropically, i.e., p(w (ℓ) ) = p(D (ℓ) (R)w (ℓ) ) for any R ∈ SO(3). Now, from the tensor expansion defined as:
H (ℓ1,ℓ2) = ⊗w (ℓ) (ℓ1,ℓ2) ,(61)
we know from the previous subsection that this construction transforms under SO(3) as:
H (ℓ1,ℓ2) R -→ D (ℓ1) (R)H (ℓ1,ℓ2) D (ℓ2) (R) -1 .(62)
To prove invariance, consider the pushforward distribution:
p(H (ℓ1,ℓ2) ) = p(w (ℓ) ) = p(D (ℓ) (R)w (ℓ) ) ∀R ∈ SO(3).(63)
Because H (ℓ1,ℓ2) is a deterministic, equivariant function of w (ℓ) , and p(w (ℓ) ) is rotation-invariant, the induced distribution p(H (ℓ1,ℓ2) ) must also be invariant under conjugation by D (ℓ1) (R) and D (ℓ2) (R). Thus:
p(H (ℓ1,ℓ2) ) = p(D (ℓ1) (R) H (ℓ1,ℓ2) D (ℓ2) (R) -1 ).(64)
This proves that the prior distribution constructed via tensor expansion from rotationally invariant w (ℓ) yields an SO(3)-equivariant (conjugation-invariant) distribution over matrix-valued outputs.
this section cite: []

Section: C.4 Proof of the invariance of the multiple tensor expansion prior
We now generalize the single-irrep vector expansion to construct a full Hamiltonian matrix H ∈ R n×n composed of multiple irrep vectors components. Let the atomic orbital basis consist of a set of irrep vectors indexed by L = {ℓ 1 , ℓ 2 , . . . , ℓ B }, where each angular momentum ℓ i corresponds to a subspace of dimension (2ℓ i + 1).
For each irrep vector rank ℓ i , we define a random irrep vector w (ℓi) ∈ C (2ℓi+1) , sampled independently from a rotationally invariant distribution:
w (ℓi) = r i D (ℓi) (R i )w (ℓi) 0 , r i ∼ LogNormal(1, σ 2 ), R i ∼ Uniform(SO(3)).(65)
We then define a factorized prior:
p(w (ℓ1) , . . . , w (ℓ B ) ) = B i=1 p i (w (ℓi) ),(66)
with each p i (w (ℓi) ) being SO(3)-invariant.
Using the tensor expansion, each pair (ℓ i , ℓ j ) generates a block of the Hamiltonian matrix via:
H (ℓi,ℓj ) = ⊗ℓi,ℓj w (ℓ k ) , for some ℓ k ∈ {|ℓ i -ℓ j |, . . . , ℓ i + ℓ j }.(67)
The full Hamiltonian is then assembled as:
H = i,j H (ℓi,ℓj ) .(68)
From the single-component proof, we know that each block transforms as:
H (ℓi,ℓj ) R -→ D (ℓi) (R)H (ℓi,ℓj ) D (ℓj ) (R) -1 .(69)
Therefore, under a global rotation R ∈ SO(3), the full matrix H transforms as:
H R -→ D(R) H D(R) -1 ,(70)
where D(R) is the block-diagonal rotation operator acting on the full basis.
Since each w (ℓi) is sampled independently from an SO(3)-invariant distribution, the joint distribution of all irreducible components p(H) remains invariant under this global conjugation:
p(H) = p D(R) H D(R) -1 , ∀R ∈ SO(3).(71)
This completes the proof that the full Hamiltonian matrix constructed via multiple tensor expansions, using independently sampled spherical features from invariant priors, defines an SO(3)-invariant distribution over matrices.
this section cite: []

Section: D Implementation details

this section cite: []

Section: D.1 Model implementation
We build our model upon the official QHNet [23] and DEQHNet [59] codebases, which implement SE(3)-equivariant GNN for Hamiltonian matrix prediction. QHNet is selected as our backbone for its balance of architectural simplicity and strong predictive performance. Although more recent models such as WANet [25] and SPHNet [26] have demonstrated improvements, their codebases are not publicly available and thus are not considered here. Notably, our method is model-agnostic and could be paired with more expressive backbones if desired.
We extend QHNet by incorporating additional physically meaningful inputs: the current Hamiltonian matrix H t , the overlap matrix S, the molecular configuration M, and the time t. Unlike the original QHNet, which processes only M, our model is designed to handle H t , and S as well, following symmetry-preserving strategies inspired by DEQHNet and OrbNet-Equi [60].
Project Block. To map the equivariant matrix features H and S into the SO(3) irrep vectors, we used diagonal reduction in OrbNet [60] based on the Wigner-Eckart theorem [46]. For atom i, we first extract the atom-wise matrices, H i := H ii (see Equation ( 35)). Each element of H i carries two orbital index, (ℓ, m) and (ℓ ′ , m ′ ). We project these matrices onto rank-ℓ o irrep vectors h
(ℓo) i = [h (ℓo,-ℓo) i , . . . h (ℓo,ℓo) i ] ∈ C (2ℓo+1) by h (ℓo,mo) i = (ℓ,ℓ ′ ) (m,m ′ ) H (ℓ,m;ℓ ′ ,m ′ ) i Q (ℓ,m;ℓ ′ ,m ′ ) i,(ℓo,mo) ,(72)
where
H (ℓ,m;ℓ ′ ,m ′ ) i is an element of submatrix H (ℓ,ℓ ′ ) i
that corresponds to the m and m ′ index, and a set of atom-wise projection weights Q
(ℓ,m;ℓ ′ ,m ′ ) i,(ℓo,mo)
which plays the role of Clebsch-Gordan-like projection weights. Because the map in Equation ( 72) is an SO(3) tensor contraction, the resulting feature vectors remain equivariant. The same procedure is applied to overlap matrix S, giving two parallel streams of SE(3)-equivariant atomic features that are subsequently fused in the network.
this section cite: ['b22', 'b58', 'b24', 'b25', 'b59', 'b59', 'b45']

Section: Construction of projection weights.
To construct the coefficients atom-wise projection coefficients, we compute three-center integrals involving orbital basis function Φ
(ℓ,m) i and auxiliary spherical-type basis functions Φ(l,m) i : Q (ℓ,m;ℓ ′ ,m ′ ) i,(ℓo,mo) = Φ (ℓ,m) i (r) * Φ (ℓ ′ ,m ′ ) i (r) Φ(ℓo,mo) i (r) dr.
In practice, the angular parts of these integrals are related with the Clebsch-Gordan coefficients due to their relation to spherical harmonics:
Q (ℓ,m;ℓ ′ ,m ′ ) i,(ℓo,mo) ∝ S 2 Y ℓ m (r)Y ℓ ′ m (r) Y ℓo mo (r) * dr.(74)
this section cite: []

Section: D.2 Training objective
Our training objective adopts a residual learning strategy, following prior works such as WANet and SHNet. Rather than directly predicting the converged Hamiltonian, the model learns to approximate the residual between the initial and converged Hamiltonians using a time-dependent vector field within the conditional flow matching framework. We define the residual target as done in prior works:
H 1,M := H ⋆ M -H (0) M ,(75)
where
H (0
)
M is the initial guess of Hamiltonian matrixfoot_2 , and H ⋆ M is the SCF solution. The final prediction of Hamiltonian is obtained by summing the predicted residual and the initial Hamiltonian guess.
The conditional flow matching loss is defined as:
L = E (H,M)∼A,t∼U (0,1),Ht∼pt v t,θ (H t,M ) -u t (H t,M | H 1,M ) 2 2 (76) = E (H,M)∼A,t∼U (0,1),Ht∼pt 1 (1 -t) 2 H (θ) 1,M -H 1,M 2 2 , (77
)
where the second line follows from the analytical form of the conditional velocity field u t , showing that the model is penalized based on the squared residual error at each time step. This formulation encourages smooth convergence from the initial guess to the target solution. To simplify the objective, we omit the 1/(1 -t) 2 time-scaling for our implementation.
this section cite: []

Section: E Experimental study settings E.1 Dataset preparation
To demonstrate the effectiveness of flow-matching-based training, we conduct experiments on two molecular datasets: MD17 and QH9. The MD17 represents a relatively simple task compared to the QH9, focusing solely on small systems and their conformational space. The PubChemQH9 is not considered since their dataset and codebase are not publicly released.
this section cite: []

Section: MD17.
The MD17 [49,19] dataset consists of quantum chemical simulations for four small organic molecules: water (H 2 O), ethanol (C 2 H 5 OH), malondialdehyde (CH 2 (CHO) 2 ), and uracil (C 4 H 4 N 2 O 2 ). It provides a comprehensive set of molecular properties, including geometries, total energies, forces, Kohn-Sham Hamiltonian matrices, and overlap matrices. All reference computations were implemented via the ORCA electronic structure package [61] using the PBE exchange-correlation functional [52,53] and the def2-SVP Gaussian-type orbital (GTO) basis set. We follow the standard data split protocol used in prior work [19,20,23] to divide each molecule's conformational data into training, validation, and test sets. The detailed dataset statistics are summarized in Table 5 and MOs in the table imply molecular orbitals (i.e.. s,p,d,f) Table 5: The statistics of MD17 dataset [19].
Dataset # of structures Train Val Test # of atoms # of orbitals # of occupied MOs Water 4,900 500 500 3,900 3 24 5 Ethanol 30,000 25,000 500 4,500 9 72 10 Malondialdehyde 26,978 25,000 500 1,478 9 90 19 Uracil 30,000 25,000 500 4,500 12 132 26
QH9. The QH9 dataset [24] is a large-scale quantum chemistry benchmark designed to support the training and evaluation of machine learning models for Hamiltonian matrix prediction across diverse chemical structures. It is based on the QM9 [50,51] molecular dataset and includes 130,831 Hamiltonian matrices from stable molecular geometries, as well as 2,698 molecular dynamics trajectories. The dataset covers small organic molecules composed of up to nine heavy atoms (C, N, O, and F). All Hamiltonians were computed using the PySCF [62] quantum chemistry package with the B3LYP [54] exchange-correlation functional and the def2-SVP Gaussian-type orbital (GTO) basis set. We provide the detailed dataset statistics in Table 6.
QH9 is organized into two main subsets, QH9-stable and QH9-dynamic-300k, and provides four standard evaluation splits: stable-id, stable-ood, dynamic-300k-geo, and dynamic-300k-mol. The id split randomly partitions the QH9-stable subset into training, validation, and test sets, while the ood split is constructed based on molecular size. Specifically, it groups molecules with 3-20 atoms in the training set, molecules with 21-22 atoms in the validation set, and molecules with 23-29 atoms in the test set. This ood setup allows for a rigorous evaluation of out-of-distribution (OOD) generalization in terms of molecular complexity.
The geo and mol splits are based on molecular dynamics (MD) trajectories, where each molecule is associated with 100 geometry snapshots. In the geo split, geometries are randomly assigned within each molecule: 80 for training, 10 for validation, and 10 for testing. Thus, while all molecules are present across the splits, the specific conformations are disjoint, enabling an assessment of geometric generalization. In contrast, the mol split divides the 2,698 molecules themselves into disjoint sets with an 80/10/10 train/validation/test ratio. All 100 geometries of a molecule are assigned to the same subset. This split presents a more challenging task than the geo split, as it requires generalization to entirely unseen molecular identities rather than just new conformations.
this section cite: ['b48', 'b18', 'b60', 'b51', 'b52', 'b18', 'b19', 'b22', 'b18', 'b23', 'b49', 'b50', 'b61', 'b53']

Section: E.2 Evaluation Metrics
To comprehensively evaluate the performance of Hamiltonian prediction models, we adopt several metrics that measure accuracy, physical fidelity, and downstream impact on quantum chemical properties. Below are detailed descriptions of each metric used in our experiments.
Table 6: The statistics of QH9 dataset [24].
this section cite: ['b23']

Section: Dataset # of structures # of Molecules Train Val Test
Stable-id 130,831 130,831 104,664 13,083 13,084 Stable-ood 130,831 130,831 104,001 17,495 9,335 Dynamic-300k-geo 299,800 2,998 239,840 29,980 29,980 Dynamic-300k-mol 299,800 2,998 239,800 29,900 30,100
Hamiltonian MAE. This metric measures the mean absolute error (MAE) between the predicted Hamiltonian matrix Ĥ and the ground-truth matrix H ⋆ :
MAE(H) = 1 n 2 n i,j=1 Ĥij -H ⋆ ij 2 2 , (78
)
where n is the length of the Hamiltonian matrix (H ∈ R n×n ), and ∥ • ∥ 2 2 is the Frobenius norm. This metric directly reflects the quality of the predicted Hamiltonian in element-wise terms.
Occupied orbital energy MAE (ϵ occ ). This metric measures the MAE between the predicted and true orbital energies for only the occupied orbitals. Let ε and ϵ ⋆ be predicted and the ground-truth generalized eigenvalues of the Hamiltonian, respectively, and let I occ be the set of indices of the occupied orbitals:
MAE(ϵ occ ) = 1 card(I occ ) i∈Iocc ∥ε i -ϵ ⋆ i ∥ 2 2 . (79
)
This is particularly important for capturing the energy spectrum relevant to ground-state electronic properties. We identify the occupied orbitals by selecting the ⌊N/2⌋ lowest eigenvalues, where N is the number of electrons in the system.
this section cite: []

Section: Orbital coefficient similarity score (S c ).
To compare the predicted and ground-truth molecular orbital coefficient matrices Ĉ and C ⋆ , we use a cosine similarity-based score averaged over all columns:
S C ( Ĉ, C ⋆ ) = Ĉ, C ⋆ = i C i C ⋆ i ∥ Ĉ∥∥C ⋆ ∥ , (80
)
where C i is the i-th column vector of the coefficient matrix and ⟨•, •⟩ denotes the inner product. This metric evaluates how well the predicted orbitals align with the ground truth.
HOMO, LUMO, and energy gap MAE (ϵ HOMO , ϵ LUMO , ϵ ∆ ). These metrics quantify the predictive accuracy of specific frontier orbital energies, which are critical for determining electronic properties such as reactivity and charge transport. The highest occupied molecular orbital (HOMO) and lowest unoccupied molecular orbital (LUMO) energies, along with their difference (the HOMO-LUMO gap), are particularly sensitive to model generalization, especially in challenging settings like QH9, where predictions are evaluated on unseen molecular structures or conformations. These metrics are computed as follows:
ϵ HOMO = εHOMO -ϵ ⋆ HOMO ,(81)
ϵ LUMO = εLUMO -ϵ ⋆ LUMO ,(82)
ϵ ∆ = (ε LUMO -εHOMO ) -(ϵ ⋆ LUMO -ϵ ⋆ HOMO ) .(83)
Here, ϵ HOMO corresponds to the ⌊N/2⌋-th lowest eigenvalue, and ϵ LUMO to the (⌊N/2⌋ + 1)-th, where N is the number of electrons in the system.
this section cite: []

Section: SCF Iter Ratio.
This metric measures the reduction in the number of SCF iterations required when using a predicted Hamiltonian as the initial guess, relative to a standard reference initialization (e.g., from a minimal basis or atomic superposition guess). It is defined as
SCF Iter Ratio = Iter pred Iter ref ,(84)
where Iter pred is the number of SCF iterations with the predicted initialization, and Iter ref is the number of iterations under the reference setup. In our setting, we used the reference value, which is measured from the conventional DFT metric to compare the acceleration of our method.
this section cite: []

Section: SCF T Ratio.
This represents the reduction in total wall-clock time spent during SCF convergence using the predicted Hamiltonian. It reflects actual runtime savings in quantum chemical simulations:
SCF T Ratio = T SCF-pred T SCF-ref ,(85)
where T SCF-pred and T SCF-ref denote the SCF runtimes under predicted and reference initializations, respectively.
Inf T Ratio. This metric captures the inference overhead of the predictive model itself. It is the fraction of time required to generate the predicted Hamiltonian relative to the baseline SCF time:
Inf T Ratio = T inference T SCF-ref ,(86)
where T inference is the runtime of the Hamiltonian prediction model.
Total T Ratio. This quantifies the net cost of using the predictive model, combining both inference and SCF convergence times. It provides a holistic view of overall computational efficiency:
Total T Ratio = T inference + T SCF-pred T SCF-ref .(87)
Note. All energy values are reported in units of Hartree (1E h = 27.211eV ), and similarity scores are unitless with values in [0, 1], where higher is better. While the SCF Iter Ratio is a deterministic metric reflecting algorithmic convergence behavior, the wall-clock timing ratios (SCF T, Inf T, and Total T) are subject to variability due to system-level factors. In our experiments, all timing evaluations were conducted on A100-80GB GPUs. However, due to resource allocation via SLURM and shared CPU environments, precise time measurements were not always reproducible. As such, reported time ratios should be interpreted as indicative trends rather than absolute benchmarks.
this section cite: []

Section: E.3 Experimental setup
Environment. Experiments were conducted using a single GPU per model. For the MD17 dataset, we used NVIDIA RTX 3090 and A5000 GPUs, while for the QH9 dataset, experiments were performed on NVIDIA A100 80GB GPUs. Our implementation is based on PyTorch 2.1.2 and PyG 2.3.0, both compiled with CUDA 12.1. Detailed package versions [62][63][64] and environment specifications will be released upon publication for full reproducibility.
Training was performed on a SLURM-managed cluster, where slight variability in runtime was observed due to system-wide GPU and CPU resource contention. Random seeds were fixed where possible to ensure training stability; however, minor non-determinism remains due to the inherent variability of distributed computing environments.
Training QHFlow took approximately 2-4 days on MD17 and 7-10 days on QH9, which is slightly longer than QHNet due to the added complexity of incorporating the current Hamiltonian and orbital information as inputs. Finetuning on QH9 required an additional 3-4 days. While the total training time could be further reduced with optimized infrastructure, our experiments were conducted under a SLURM-managed environment with periodic reinitialization every 72 hours. Importantly, QHFlow is a model-agnostic framework; with more scalable architectures, training efficiency can be significantly improved.
this section cite: ['b61', 'b62', 'b63']

Section: Hyperparameters.
For fair comparison, we adopted the same hyperparameters used by the baseline model QHNet [23] whenever possible. Our QHFlow shares the majority of its architecture and training settings with QHNet to ensure that performance improvements are attributable to the proposed flow matching design rather than hyperparameter tuning. A summary of the key hyperparameters used across different datasets is provided in Table 7.
For evaluating DFT acceleration performance, we used the PySCF [62] framework with the B3LYP exchange-correlation functional. The numerical integration grid level was set to 3, and the maximum number of SCF cycles was 50. All other parameters followed the default settings of the PySCF RKS implementation.
this section cite: ['b22', 'b61']

Section: F Additional experimental results and limitations

this section cite: []

Section: F.1 Ablation study of the finetuning objective
We conduct additional experiments to evaluate various fine-tuning strategies to improve the Hamiltonian prediction performance of QHFlow after standard flow matching pretraining. In this section, we describe each strategy in detail and provide empirical observations.
For the WALoss implementation, we follow the coefficient settings proposed in WANet [25] (λ WA = 2.0). During fine-tuning, we train for an additional 60,000 steps starting from the pretrained model with TE prior, using an initial learning rate of 1 × 10 -5 and a polynomial learning rate scheduler.
Full Flow Matching with WA Loss training from scratch (WA-Full). In this strategy, we train the model from scratch by jointly optimizing the flow matching objective and WALoss for 260,000 steps, matching the default training steps as done in WANet. However, this approach significantly degraded the Hamiltonian error. We hypothesize that imposing strong spectral constraints too early disrupts the learning of the global Hamiltonian structure, leading to unstable convergence and poor generalization.
WA Loss finetuning (WA-FT). After completing standard flow matching pre-training, we fine-tune the model solely using WALoss, which encourages alignment between the predicted Hamiltonian's eigenstructure and the ground-truth SCF solutions. This approach improves orbital energy prediction and SCF convergence compared to pure flow matching training, confirming the value of adding spectral supervision during fine-tuning.
Summary. Table 8 summarizes the quantitative results of all fine-tuning strategies. Here, w/o-FT refers to QHFlow trained solely with the flow matching objective using the TE prior. Among the approaches, WA-FT achieves the best balance between Hamiltonian prediction accuracy and practical DFT acceleration, demonstrating the effectiveness of spectral alignment in the fine-tuning stage.
this section cite: ['b24']

Section: F.2 Full results of DFT acceleration via SCF initialization
For completeness, we include in Figure 5 the figure with the inset from Figure 3. As shown in Figure 5, the inference time of QHFLOW is longer than that of QHNet because QHFLOW performs multi-step inference for ODE integration. However, this additional cost is negligible compared to the overall DFT computation pipeline, as reflected in the SCF T ratio.
this section cite: []

Section: F.3 Full results of ablation study
For completeness, we provide the full ablation study results in Table 9, and Table 10, including additional metrics beyond those presented in the main text. While the main manuscript focused primarily on Hamiltonian prediction error (H), occupied orbital energy error (ϵ occ ), and electronic density accuracy (S c ) for clarity, here we also report the LUMO (ϵ LUMO ), HOMO (ϵ HOMO ), and HOMO-LUMO gap (ϵ ∆ ) energy errors.
Overall, the trends observed in the full table are consistent with those discussed in the main text. In particular, the improvements in Hamiltonian accuracy and density prediction are accompanied by consistent reductions in orbital energy errors (ϵ LUMO , ϵ HOMO ) and gap errors (ϵ ∆ ), further demonstrating the broad effectiveness of our proposed methods. These results reinforce the conclusion that flow matching enhance both Hamiltonian structure prediction and downstream electronic properties.
Also, to better visualize the role of the prior distributions, we provide interpolation trajectories from the GOE and TE prior toward target Hamiltonians. Figure 6 shows example interpolations for both settings, highlighting how the initial distribution affects the learned flow dynamics. As seen, the TE prior results in smoother and more structured trajectories compared to the GOE prior, which starts from unstructured noise.
this section cite: []

Section: F.4 Ablation studies on model design
In this section, we present additional ablation studies on the architectural design choices of QH-FLOW, as summarized in Table 11, to clarify their individual roles and contributions to the overall performance. Regression. QHFLOW (Regression) is a baseline that directly predicts the Hamiltonian correction matrix (∆H = H ⋆ -H (0) ) using the same objective described above, without incorporating any physical priors or structural constraints. This variant isolates the effect of the flow-matching mechanism itself and evaluates how well the model performs under a purely regression-based setting. Although this approach achieves improved Hamiltonian prediction compared to QHNet, its overall performance remains inferior to the flow-matching version.
Overlap matrix S. We further examine the impact of incorporating the overlap matrix embedding, which encodes pairwise correlations among atomic orbitals. Comparing models with and without S embedding shows that leveraging overlap information significantly enhances model stability and accuracy across both Hamiltonian prediction and SCF convergence.
this section cite: []

Section: F.5 Analysis of error distribution of the predictions
To better understand the properties of QHFLOW, we move beyond single-number summaries and visualise the entire error distribution for six key quantities Hamiltonian MAE (H), occupied-orbital MAE (ϵ occ ), ϵ HOMO , ϵ LUMO , ϵ ∆ , and S c on all four QH9 splits (id, ood, geo, mol). Figure 7 shows log-scale violin plots for QHNet* (grey), QHFLOW (orange), and its WA-finetuned variant (cyan).
(only the similarity score is kept on a linear axis.) Each violin displays the full empirical distribution; the thick dashed line marks the median, and the two thinner lines divide the inter-quartile ranges.
Across every split and metric, the QHFLOW violins are both lower and narrower. Median Hamiltonian error is reduced by roughly one order of magnitude on id and geo, and by a factor of 3-5 on the tougher ood and mol sets. In addition, the long error tails of QHNet* largely vanish, demonstrating that our flow objective mitigates the occasional catastrophic failure modes that plague point-wise regression models.
Figure 8 plots log H against log ϵ occ for every test geometry. QHFLOW occupies the bottom-left corner of each panel, while QHNet* points fan out towards larger joint errors, especially in ood and mol, where chemical diversity is highest. These results show the QHFLOW superior performance on both the geometric and unseen molecules.
Finally, Figure 9 compares Hamiltonian error versus atom count. QHFLOW remains flat up to the largest molecules in QH9 (29atoms). This suggests that modeling a distribution over Hamiltonians confers a form of size transferability.  QHNet* Ours WA-FT 10 1 10 2 10 3 logH [ Eh] H (id) QHNet* Ours WA-FT 10 1 10 2 10 3 10 4 10 5 log occ [ Eh] occ (id) QHNet* Ours WA-FT 20 40 60 80 100 c [%] c (id) QHNet* Ours WA-FT 10 2 10 0 10 2 10 4 log HOMO [ Eh] HOMO (id) QHNet* Ours WA-FT 10 2 10 0 10 2 10 4 10 6 log LUMO [ Eh] LUMO (id) QHNet* Ours WA-FT 10 1 10 1 10 3 10 5 log [ Eh]
(id) QHNet* Ours WA-FT 10 1 10 2 logH [ Eh] H (ood) QHNet* Ours WA-FT 10 2 10 3 10 4 10 5 log occ [ Eh] occ (ood) QHNet* Ours WA-FT 20 40 60 80 100 c [%] c (ood) QHNet* Ours WA-FT 10 1 10 1 10 3 10 5 log HOMO [ Eh] HOMO (ood) QHNet* Ours WA-FT 10 2 10 0 10 2 10 4 10 6 log LUMO [ Eh] LUMO (ood) QHNet* Ours WA-FT 10 1 10 1 10 3 10 5 log [ Eh]
(ood) QHNet* Ours WA-FT 10 1 10 2 logH [ Eh] H (geo) QHNet* Ours WA-FT 10 1 10 2 10 3 10 4 10 5 log occ [ Eh] occ (geo) QHNet* Ours WA-FT 20 40 60 80 100 c [%] c (geo) QHNet* Ours WA-FT 10 2 10 0 10 2 10 4 log HOMO [ Eh] HOMO (geo) QHNet* Ours WA-FT 10 2 10 0 10 2 10 4 10 6 log LUMO [ Eh] LUMO (geo) QHNet* Ours WA-FT 10 1 10 1 10 3 10 5 log [ Eh]   versus atom count for the four QH9 splits. QHNet* (gray) errors rise with molecular size in ood, whereas QHFLOW (orange) and WA-FT (cyan) remain nearly flat up to 29 atoms. This suggests that modelling the full Hamiltonian distribution with symmetry-aware priors confers robust transferability to larger, more complex molecules.
this section cite: []

Section: References
Ref_id:b0 Title: Inhomogeneous electron gas Year: (1964)
Ref_id:b1 Title: Density functional theory of atoms and molecules Year: (1979-11-03)
Ref_id:b2 Title: Density functional theory: An introduction Year: (2000)
Ref_id:b3 Title: Perspective: Fifty years of density-functional theory in chemical physics Year: (2014)
Ref_id:b4 Title: Perspective on density functional theory Year: ()
Ref_id:b5 Title: Thirty years of density functional theory in computational chemistry: an overview and extensive assessment of 200 density functionals Year: (2017)
Ref_id:b6 Title: Density functional theory in materials science Year: (2013)
Ref_id:b7 Title: Dft exchange: sharing perspectives on the workhorse of quantum chemistry and materials science Year: (2022)
Ref_id:b8 Title: Classical dynamical density functional theory: from fundamentals to applications Year: (2020)
Ref_id:b9 Title: Electronic structure: basic theory and practical methods Year: (2020)
Ref_id:b10 Title: A review on the use of dft for the prediction of the properties of nanomaterials Year: (2021)
Ref_id:b11 Title: Self-consistent equations including exchange and correlation effects Year: (1965)
Ref_id:b12 Title: A generalized self-consistent field method Year: (1953)
Ref_id:b13 Title: Self-consistent field approach to the many-electron problem Year: (1959)
Ref_id:b14 Title: Self-consistent molecular-orbital methods. i. use of gaussian expansions of slater-type atomic orbitals Year: (1969)
Ref_id:b15 Title: An introduction to hartree-fock molecular orbital theory. School of Chemistry and Biochemistry Georgia Institute of Technology Year: (2000)
Ref_id:b16 Title: Quantum materials: Experiments and theory Year: (2016)
Ref_id:b17 Title: A deep learning framework to emulate density functional theory Year: (2023)
Ref_id:b18 Title: Unifying machine learning and quantum chemistry with a deep neural network for molecular wavefunctions Year: (2007)
Ref_id:b19 Title: )-equivariant prediction of molecular wavefunctions and electronic densities Year: (2007)
Ref_id:b20 Title: Deep-learning density functional theory hamiltonian for efficient ab initio electronic-structure calculation Year: (2022)
Ref_id:b21 Title: General framework for e (3)-equivariant neural network representation of density functional theory hamiltonian Year: (2023)
Ref_id:b22 Title: Efficient and equivariant graph networks for predicting quantum hamiltonian Year: (2023)
Ref_id:b23 Title: Qh9: A quantum hamiltonian prediction benchmark for qm9 molecules Year: (2023)
Ref_id:b24 Title: Enhancing the scalability and applicability of kohn-sham hamiltonians for molecular systems Year: (2025)
Ref_id:b25 Title: Efficient and scalable density functional theory hamiltonian prediction through adaptive sparsity Year: (2025)
Ref_id:b26 Title: Equivariant flow matching with hybrid probability transport for 3d molecule generation Year: (2023)
Ref_id:b27 Title: Equivariant flow matching Year: (2023)
Ref_id:b28 Title: Schnet-a deep learning architecture for molecules and materials Year: ()
Ref_id:b29 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b30 Title: Improving and generalizing flow-based generative models with minibatch optimal transport Year: (2023)
Ref_id:b31 Title: Et-flow: Equivariant flow-matching for molecular conformer generation Year: (2025)
Ref_id:b32 Title: Alphafold meets flow matching for generating protein ensembles Year: (2024)
Ref_id:b33 Title: Flowmm: Generating materials with riemannian flow matching Year: (2024)
Ref_id:b34 Title: Flow matching for structure prediction of metal-organic frameworks Year: (2024)
Ref_id:b35 Title: Harmonic self-conditioned flow matching for joint multi-ligand docking and binding site design Year: (2024)
Ref_id:b36 Title: An undulatory theory of the mechanics of atoms and molecules Year: (1926)
Ref_id:b37 Title: New developments in molecular orbital theory Year: (1951-04)
Ref_id:b38 Title: The molecular orbital theory of chemical valency viii. a method of calculating ionization potentials Year: (1083-03)
Ref_id:b39 Title: Atomic shielding constants Year: (1930)
Ref_id:b40 Title: Molecular integrals over gaussian basis functions Year: (1994)
Ref_id:b41 Title: Ground-state properties of cosi 2 determined by a total-energy pseudopotential method Year: (1994)
Ref_id:b42 Title: Groups and symmetries Year: (2010)
Ref_id:b43 Title: Introduction to quantum mechanics Year: (2018)
Ref_id:b44 Title: Improved motif-scaffolding with se (3) flow matching Year: (2024)
Ref_id:b45 Title: On the matrices which reduce the kronecker products of representations of sr groups Year: (1993)
Ref_id:b46 Title: Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds Year: (2018)
Ref_id:b47 Title: Equiformer: Equivariant graph attention transformer for 3d atomistic graphs Year: (2022)
Ref_id:b48 Title: Machine learning of accurate energy-conserving molecular force fields Year: (2017)
Ref_id:b49 Title: Enumeration of 166 billion organic small molecules in the chemical universe database gdb-17 Year: (2012)
Ref_id:b50 Title: Electronic spectra from tddft and machine learning in chemical space Year: (2015)
Ref_id:b51 Title: Generalized gradient approximation made simple Year: (1996)
Ref_id:b52 Title: Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for h to rn: Design and assessment of accuracy Year: (2005)
Ref_id:b53 Title: Ab initio calculation of vibrational absorption and circular dichroism spectra using density functional force fields Year: (1994)
Ref_id:b54 Title: Optimal flow matching: Learning straight trajectories in just one step Year: (2024)
Ref_id:b55 Title: Group theory in physics: An introduction Year: (1997)
Ref_id:b56 Title: Group theory: and its application to the quantum mechanics of atomic spectra Year: (2012)
Ref_id:b57 Title: A chemist's guide to density functional theory Year: (2015)
Ref_id:b58 Title: Infusing self-consistency into density functional theory hamiltonian prediction via deep equilibrium models Year: (2024)
Ref_id:b59 Title: Orbnet: Deep learning for quantum chemistry using symmetry-adapted atomic-orbital features Year: (2020)
Ref_id:b60 Title: The orca quantum chemistry program package Year: (2020)
Ref_id:b61 Title: Pyscf: the pythonbased simulations of chemistry framework Year: (1340)
Ref_id:b62 Title: Euclidean neural networks Year: (2022)
Ref_id:b63 Title: The atomic simulation environment-a python library for working with atoms Year: (2017)
